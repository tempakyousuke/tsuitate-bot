//! **王手中の反則経済 P0-7: 反則トークンの影の価格**（issue #31。H1 の主測定）。
//!
//! 王手手番の**直後**（受理手を指した後）の盤面・信念・乱数を固定し、
//! **審判の累積反則数だけを +1 した** paired continuation を共通乱数で走らせて、
//! スコア差 = 「反則1個の controlled direct effect」を出す。
//! 情報価値（どの手を試したか・相手が何を観測したか）は固定されているので、
//! 出てくるのは**純粋なトークン価格**だけ。
//!
//! - **単位が違う**: 影の価格は勝率単位、`foul_cost` は評価点単位。比べられるのは
//!   **残数に対する曲線の形・比率**だけで、「水準が過小」はここでは判定しない
//! - 反実仮想なので**ログには反則を足さない**（足すと相手の信念と自分の
//!   `foul_tried` まで動いて「情報価値を固定した」ことにならない）。
//!   審判の累積数（`StartState.fouls`）と自分の視界（`PlayerView.fouls`）だけを動かす。
//!   継続側の全コードが累積数をそこから読むことは
//!   `selfplay` のテスト（`継続の累積反則数は開始状態から読む`）で固定してある
//! - **残り1を0にする treatment は継続を始めず即 foul_limit 負け**
//!   （開始時点で 10 反則なら、その手番の次の反則を待たずに負けが確定している）
//!
//! 裁定は `selfplay::play_continuation` = 通常アリーナと同じ関数。
//! **時計は無効**（途中局面の残り時間は復元できない）。
//!
//! usage:
//!   TSUITATE_THINK_BUDGET_MS=700 cargo run --release --bin check_price -- \
//!     [--seeds 2] [--per-game 2] [--opponent estimator_v14] [--jobs N] [--shard i/n] \
//!     [--out out.jsonl] <records/*.jsonl...>
//!   cargo run --release --bin check_price -- report [--allow-incomplete] <out-*.jsonl...>

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tsuitate_bot::check_economy::cluster_ratio_ci;
use tsuitate_bot::checkpoint::stable_hash;
use tsuitate_bot::mate_economy::force_move;
use tsuitate_bot::protocol::Color;
use tsuitate_bot::scenario_core::{clone_log, make_view, side_idx};
use tsuitate_bot::selfplay::{GameResult, MAX_FOULS, StartState, mix, play_continuation};
use tsuitate_bot::shogi::Position;
use tsuitate_bot::strategy;
use tsuitate_bot::truth_replay::{for_each_decision_full, parse_bot_and_end};

const ROW_SCHEMA: u32 = 1;

fn die(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(2);
}

/// 継続の開始点（王手手番の**直後**）
struct Point {
    game: String,
    /// 決定点の手数（王手手番のほう）
    move_number: u32,
    /// 手番開始時の bot の累計反則（＝残り = 10 − これ）
    fouls_before: u32,
    /// その手番で積んだ反則
    fouls_in_turn: u32,
    /// 受理手を指した**後**の状態
    pos: Position,
    logs: [tsuitate_bot::observation::ObservationLog; 2],
    fouls: [u32; 2],
    plies: u32,
    bot: Color,
}

impl Point {
    /// 継続開始時点の bot の累計反則（= 手番開始時 + その手番の反則）
    fn fouls_me(&self) -> u32 {
        self.fouls[side_idx(self.bot)]
    }
    fn remaining(&self) -> u32 {
        MAX_FOULS.saturating_sub(self.fouls_me())
    }
    fn opp_remaining(&self) -> u32 {
        MAX_FOULS.saturating_sub(self.fouls[side_idx(self.bot.other())])
    }
    /// 手数帯（終盤ほど1反則の価値が変わりうる）
    fn band(&self) -> &'static str {
        match self.move_number {
            0..=49 => "序中盤(0-49)",
            50..=89 => "中盤(50-89)",
            _ => "終盤(90-)",
        }
    }
}

fn walk_jsonl(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            walk_jsonl(&p, out);
        } else if p.extension().is_some_and(|e| e == "jsonl") {
            out.push(p);
        }
    }
}

fn collect_records(specs: &[String]) -> Vec<PathBuf> {
    let mut out = vec![];
    for s in specs {
        let p = PathBuf::from(s);
        if p.is_dir() {
            walk_jsonl(&p, &mut out);
        } else if p.exists() {
            out.push(p);
        } else {
            eprintln!("見つかりません: {s}");
        }
    }
    out.sort();
    out.dedup();
    out
}

/// 継続の乱数（bot 側・相手側）。**arm に依らず (局, 決定点, seed) だけ**から作る
/// = control と treatment は共通乱数のペアになる（issue #28 の教訓）。
fn continuation_seeds(game: &str, ply: u32, seed: u64) -> (u64, u64) {
    let base = stable_hash(seed, &format!("{game}#{ply}"));
    (mix(base ^ 0x0C0F_FEE0), mix(base ^ 0x0BEE_F000))
}

/// 1本の継続を走らせて bot 側のスコア（勝1 / 分0.5 / 負0）を返す
fn run_continuation(p: &Point, extra_foul: bool, seed: u64, opponent: &str) -> serde_json::Value {
    let me_i = side_idx(p.bot);
    let mut fouls = p.fouls;
    if extra_foul {
        fouls[me_i] += 1;
    }
    let arm = if extra_foul { "treatment" } else { "control" };
    // **残り1を0にする treatment は継続を始めない**: 累計が上限に達した時点で
    // 反則負けが確定しているので、その状態から指し継ぐのは反実仮想として不整合
    if fouls[me_i] >= MAX_FOULS {
        return serde_json::json!({
            "schema": ROW_SCHEMA,
            "game": p.game, "move_number": p.move_number, "seed": seed, "arm": arm,
            "score": 0.0, "reason": "foul_limit", "plies": p.plies, "added_plies": 0,
            "fouls_me": fouls[me_i], "remaining": p.remaining(),
            "opp_remaining": p.opp_remaining(), "band": p.band(),
            "think_mean_ms": 0.0, "immediate_loss": true,
        });
    }
    let logs = [clone_log(&p.logs[0]), clone_log(&p.logs[1])];
    let (seed_me, seed_opp) = continuation_seeds(&p.game, p.move_number, seed);
    let mut strats: [Option<Box<dyn strategy::Strategy>>; 2] = [None, None];
    for i in 0..2 {
        let color = if i == 0 { Color::Sente } else { Color::Gote };
        let (name, s) = if i == me_i {
            ("estimator", seed_me)
        } else {
            (opponent, seed_opp)
        };
        let mut strat = strategy::make_seeded(name, s)
            .unwrap_or_else(|| die(&format!("未知の戦略名: {name}")));
        let view = make_view(&p.pos, color, &fouls);
        strategy::prewarm_strategy(&mut *strat, &view, &logs[i]);
        strats[i] = Some(strat);
    }
    let start = StartState {
        pos: p.pos.clone(),
        logs,
        fouls,
        plies: p.plies,
    };
    let out = play_continuation(
        [strats[0].take().unwrap(), strats[1].take().unwrap()],
        start,
        0,
    );
    let score = match out.result {
        GameResult::Win(w) if w == p.bot => 1.0,
        GameResult::Win(_) => 0.0,
        GameResult::Draw => 0.5,
    };
    let think = &out.think_us[me_i];
    let mean_ms = if think.is_empty() {
        0.0
    } else {
        think.iter().sum::<u64>() as f64 / think.len() as f64 / 1000.0
    };
    serde_json::json!({
        "schema": ROW_SCHEMA,
        "game": p.game, "move_number": p.move_number, "seed": seed, "arm": arm,
        "score": score, "reason": out.reason, "plies": out.plies,
        "added_plies": out.added_plies,
        "fouls_me": fouls[me_i], "remaining": p.remaining(),
        "opp_remaining": p.opp_remaining(), "band": p.band(),
        "think_mean_ms": mean_ms, "immediate_loss": false,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "report") {
        run_report(&args[1..]);
        return;
    }
    let mut seeds: u64 = 2;
    let mut per_game: usize = 2;
    let mut opponent = "estimator_v14".to_string();
    let mut jobs: usize =
        std::thread::available_parallelism().map_or(1, |n| n.get().saturating_sub(2).max(1));
    let mut shard = (0usize, 1usize);
    let mut out_path: Option<String> = None;
    let mut allow_opponent_mismatch = false;
    let mut allow_incomplete = false;
    let mut specs: Vec<String> = vec![];
    let mut i = 0;
    let need = |v: Option<&String>, what: &str| -> String {
        v.cloned()
            .unwrap_or_else(|| die(&format!("{what} には値が必要です")))
    };
    while i < args.len() {
        match args[i].as_str() {
            "--seeds" => {
                seeds = need(args.get(i + 1), "--seeds")
                    .parse()
                    .unwrap_or_else(|_| die("--seeds は整数"));
                i += 2;
            }
            "--per-game" => {
                per_game = need(args.get(i + 1), "--per-game")
                    .parse()
                    .unwrap_or_else(|_| die("--per-game は整数"));
                i += 2;
            }
            "--opponent" => {
                opponent = need(args.get(i + 1), "--opponent");
                i += 2;
            }
            "--jobs" => {
                jobs = need(args.get(i + 1), "--jobs")
                    .parse::<usize>()
                    .unwrap_or_else(|_| die("--jobs は整数"))
                    .max(1);
                i += 2;
            }
            "--shard" => {
                let v = need(args.get(i + 1), "--shard");
                let (a, b) = v.split_once('/').unwrap_or_else(|| die("--shard は i/n"));
                shard = (
                    a.parse().unwrap_or_else(|_| die("--shard は i/n")),
                    b.parse().unwrap_or_else(|_| die("--shard は i/n")),
                );
                if shard.1 == 0 || shard.0 >= shard.1 {
                    die("--shard は 0 <= i < n");
                }
                i += 2;
            }
            "--out" => {
                out_path = Some(need(args.get(i + 1), "--out"));
                i += 2;
            }
            "--allow-opponent-mismatch" => {
                allow_opponent_mismatch = true;
                i += 1;
            }
            "--allow-incomplete" => {
                allow_incomplete = true;
                i += 1;
            }
            s if s.starts_with("--") => die(&format!("未知のオプション: {s}")),
            s => {
                specs.push(s.to_string());
                i += 1;
            }
        }
    }
    if specs.is_empty() {
        die("記録ファイル（またはディレクトリ）を指定してください");
    }
    // **`--seeds 0` で空の集計を作れてはいけない**（issue #28 が塞いだ穴と同じ）
    if seeds == 0 {
        die("--seeds は 1 以上にしてください（0 だと継続を1局も走らせずに影の価格が 0 になる）");
    }
    if per_game == 0 {
        die("--per-game は 1 以上にしてください");
    }
    let files = collect_records(&specs);
    if files.is_empty() {
        die("記録ファイルが見つかりません");
    }
    let cfg = tsuitate_bot::config::ambient();

    // ---- 継続の開始点を集める --------------------------------------------
    use sha2::Digest as _;
    let mut digest = sha2::Sha256::new();
    let mut points: Vec<Point> = vec![];
    let mut games = 0u32;
    let mut broken = 0u32;
    let mut record_opponents: BTreeMap<String, u32> = BTreeMap::new();
    let mut mismatched: Vec<String> = vec![];
    for path in &files {
        let name = path.to_string_lossy().to_string();
        let Ok(content) = std::fs::read_to_string(path) else {
            broken += 1;
            continue;
        };
        sha2::Digest::update(&mut digest, name.as_bytes());
        sha2::Digest::update(&mut digest, content.as_bytes());
        let Some((bot, end)) = parse_bot_and_end(&content) else {
            broken += 1;
            continue;
        };
        *record_opponents
            .entry(end.opponent.username.clone())
            .or_insert(0) += 1;
        // 元対局の相手と継続対局の相手が違うと、価格が「相手が変わった効果」と
        // 混ざる（issue #28 PR #30 レビュー指摘①と同じ）
        if end.opponent.username != opponent {
            mismatched.push(name.clone());
            if !allow_opponent_mismatch {
                continue;
            }
        }
        let short = Path::new(&name)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or(name.clone());
        let mut found: Vec<Point> = vec![];
        let ok = for_each_decision_full(&end, |d| {
            if d.side != bot || !d.pos.in_check(bot) {
                return;
            }
            let Some(accepted) = end.moves.get(d.decision_id as usize).map(|m| m.usi.clone())
            else {
                return;
            };
            // 受理手を**共有規約**（審判と同じ観測の記録）で指した後の状態
            let logs = [clone_log(&d.logs[0]), clone_log(&d.logs[1])];
            let forced = force_move(d.pos, &logs, *d.fouls, d.side, &[accepted]);
            if forced.played.is_none() {
                return;
            }
            found.push(Point {
                game: short.clone(),
                move_number: d.pos.move_number(),
                fouls_before: d.fouls[side_idx(d.side)] - d.fouls_this_turn,
                fouls_in_turn: d.fouls_this_turn,
                pos: forced.pos,
                logs: forced.logs,
                fouls: forced.fouls,
                plies: d.plies + 1,
                bot,
            });
        });
        if !ok {
            broken += 1;
            continue;
        }
        games += 1;
        // **1局から採る決定点は `--per-game` 本まで**（同じ局の相互排他的な未来を
        // 足し合わせない。残り反則の水準が散るように、累計反則の順で等間隔に採る）
        found.sort_by_key(|p| (p.fouls_me(), p.move_number));
        if found.len() > per_game {
            let step = found.len() as f64 / per_game as f64;
            let keep: BTreeSet<usize> = (0..per_game)
                .map(|k| ((k as f64 * step) as usize).min(found.len() - 1))
                .collect();
            found = found
                .into_iter()
                .enumerate()
                .filter(|(i, _)| keep.contains(i))
                .map(|(_, p)| p)
                .collect();
        }
        points.extend(found);
    }
    if !mismatched.is_empty() && !allow_opponent_mismatch {
        eprintln!(
            "元対局の相手が --opponent {opponent} と一致しない記録が {} 件あります",
            mismatched.len()
        );
        eprintln!(
            "記録の相手: {}",
            record_opponents
                .iter()
                .map(|(k, n)| format!("{k} {n}局"))
                .collect::<Vec<_>>()
                .join(" / ")
        );
        die("影の価格が「相手が変わった効果」と混ざるので中止しました（--allow-opponent-mismatch で強行）");
    }
    points.sort_by(|a, b| (&a.game, a.move_number).cmp(&(&b.game, b.move_number)));
    // シャードは**局単位**
    if shard.1 > 1 {
        let mut names: Vec<String> = points.iter().map(|p| p.game.clone()).collect();
        names.sort();
        names.dedup();
        let keep: HashSet<String> = names
            .into_iter()
            .enumerate()
            .filter(|(i, _)| i % shard.1 == shard.0)
            .map(|(_, n)| n)
            .collect();
        points.retain(|p| keep.contains(&p.game));
    }
    if points.is_empty() {
        die("継続の開始点がありません");
    }
    println!(
        "記録 {} 件（壊れ {broken}）/ 局 {games} / 開始点 {}（1局 {per_game} 本まで）",
        files.len(),
        points.len()
    );
    println!(
        "相手 {opponent} / 思考予算 {}ms / seeds {seeds} / jobs {jobs} / shard {}/{} / config {}",
        cfg.think_budget_ms,
        shard.0,
        shard.1,
        cfg.fingerprint(),
    );
    println!("source_fingerprint {}", env!("TSUITATE_SOURCE_FINGERPRINT"));

    // ---- 継続（control / treatment のペア）---------------------------------
    let units: Vec<(usize, u64, bool)> = (0..points.len())
        .flat_map(|p| (0..seeds).flat_map(move |s| [(p, s, false), (p, s, true)]))
        .collect();
    let effective_jobs = jobs.min(units.len()).max(1);
    let next = Arc::new(Mutex::new(0usize));
    let lines: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(vec![]));
    let points = Arc::new(points);
    let started = std::time::Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..effective_jobs {
            let next = Arc::clone(&next);
            let lines = Arc::clone(&lines);
            let points = Arc::clone(&points);
            let units = &units;
            let opponent = opponent.as_str();
            scope.spawn(move || {
                loop {
                    let ui = {
                        let mut g = next.lock().unwrap();
                        if *g >= units.len() {
                            break;
                        }
                        let v = *g;
                        *g += 1;
                        v
                    };
                    let (pi, seed, extra) = units[ui];
                    let row = run_continuation(&points[pi], extra, seed, opponent);
                    lines.lock().unwrap().push(row);
                }
            });
        }
    });
    let mut lines = Arc::try_unwrap(lines).ok().unwrap().into_inner().unwrap();
    lines.sort_by_key(|v| {
        (
            v["game"].as_str().unwrap_or("").to_string(),
            v["move_number"].as_u64().unwrap_or(0),
            v["seed"].as_u64().unwrap_or(0),
            v["arm"].as_str().unwrap_or("").to_string(),
        )
    });
    eprintln!(
        "継続 {} 局 / {:.1}分",
        lines.len(),
        started.elapsed().as_secs_f64() / 60.0
    );

    let records_fingerprint: String =
        digest.finalize().iter().map(|b| format!("{b:02x}")).collect();
    let experiment = serde_json::json!({
        "opponent": opponent,
        "budget_ms": cfg.think_budget_ms,
        "seeds": seeds,
        "per_game": per_game,
        "jobs": effective_jobs,
        "shard_total": shard.1,
        "config": cfg.fingerprint(),
        "source_fingerprint": env!("TSUITATE_SOURCE_FINGERPRINT"),
        "records": records_fingerprint,
    });
    let meta = serde_json::json!({
        "schema": ROW_SCHEMA,
        "type": "meta",
        "experiment": experiment,
        "shard": shard.0,
        "games": games,
        "points": points.len(),
        "points_detail": points
            .iter()
            .map(|p| serde_json::json!({
                "game": p.game,
                "move_number": p.move_number,
                "remaining": p.remaining(),
                "fouls_before": p.fouls_before,
                "fouls_in_turn": p.fouls_in_turn,
            }))
            .collect::<Vec<_>>(),
        "record_opponents": record_opponents,
    });
    if let Some(path) = &out_path {
        let mut s = format!("{meta}\n");
        for l in &lines {
            s.push_str(&format!("{l}\n"));
        }
        match std::fs::write(path, s) {
            Ok(()) => println!("JSONL: {path}"),
            Err(e) => eprintln!("{path}: 書けません: {e}"),
        }
    }
    report(&[meta], &lines, allow_incomplete);
}

// ---- 集計 ------------------------------------------------------------------

/// ペア（control / treatment）の差を局ごとに畳む
fn pairs(rows: &[serde_json::Value]) -> Vec<(String, f64, u32, u32, String)> {
    let mut by: BTreeMap<(String, u64, u64), BTreeMap<String, serde_json::Value>> = BTreeMap::new();
    for r in rows {
        by.entry((
            r["game"].as_str().unwrap_or("?").to_string(),
            r["move_number"].as_u64().unwrap_or(0),
            r["seed"].as_u64().unwrap_or(0),
        ))
        .or_default()
        .insert(r["arm"].as_str().unwrap_or("?").to_string(), r.clone());
    }
    let mut out = vec![];
    for ((game, _, _), m) in by {
        let (Some(c), Some(t)) = (m.get("control"), m.get("treatment")) else {
            continue;
        };
        let d = t["score"].as_f64().unwrap_or(0.0) - c["score"].as_f64().unwrap_or(0.0);
        out.push((
            game,
            d,
            c["remaining"].as_u64().unwrap_or(0) as u32,
            c["opp_remaining"].as_u64().unwrap_or(0) as u32,
            c["band"].as_str().unwrap_or("?").to_string(),
        ));
    }
    out
}

/// 局ごとの cluster bootstrap（統計単位は元対局。seed も決定点も独立標本にしない）
fn cluster(diffs: &[(String, f64)]) -> (f64, f64, f64) {
    let mut by_game: BTreeMap<String, (f64, f64)> = BTreeMap::new();
    for (g, d) in diffs {
        let e = by_game.entry(g.clone()).or_default();
        e.0 += d;
        e.1 += 1.0;
    }
    let v: Vec<(f64, f64)> = by_game.values().copied().collect();
    let num: f64 = v.iter().map(|(a, _)| a).sum();
    let den: f64 = v.iter().map(|(_, b)| b).sum();
    let point = if den > 0.0 { num / den } else { f64::NAN };
    let (lo, hi) = cluster_ratio_ci(&v, 0.05, 0x31_2028);
    (point, lo, hi)
}

fn report(metas: &[serde_json::Value], rows: &[serde_json::Value], allow_incomplete: bool) {
    for msg in check_inputs(metas, rows) {
        if allow_incomplete {
            eprintln!("警告: {msg}");
        } else {
            die(&msg);
        }
    }
    let exp = metas
        .first()
        .map(|m| m["experiment"].clone())
        .unwrap_or_default();
    let all = pairs(rows);
    println!("\n=== P0-7: 反則トークンの影の価格（issue #31 / H1 の主測定）===");
    println!(
        "  実験: 相手 {} / 予算 {}ms / seeds {} / 1局あたり {} 点 / 記録 {} / code {}",
        exp["opponent"], exp["budget_ms"], exp["seeds"], exp["per_game"], exp["records"],
        exp["source_fingerprint"],
    );
    if all.is_empty() {
        println!("  ペアが1つもありません（control / treatment が揃っていない）");
        return;
    }
    let (d, lo, hi) = cluster(&all.iter().map(|(g, d, ..)| (g.clone(), *d)).collect::<Vec<_>>());
    println!(
        "  ペア {} / 元対局 {} / **影の価格（treatment − control、勝率単位）: {d:+.4} [{lo:+.4}, {hi:+.4}]**",
        all.len(),
        all.iter().map(|(g, ..)| g).collect::<BTreeSet<_>>().len(),
    );
    println!(
        "  ※ 負なら「反則1個が勝率をこれだけ下げる」。`foul_cost` は評価点単位なので\
水準は比べられない。読むのは**残数に対する曲線の形**だけ"
    );

    let table = |title: &str, key: &dyn Fn(&(String, f64, u32, u32, String)) -> String| {
        let mut by: BTreeMap<String, Vec<(String, f64)>> = BTreeMap::new();
        for p in &all {
            by.entry(key(p)).or_default().push((p.0.clone(), p.1));
        }
        println!("  {title}:");
        for (k, v) in by {
            let (d, lo, hi) = cluster(&v);
            println!(
                "    {k:<16} n={:<5} {d:+.4} [{lo:+.4}, {hi:+.4}]",
                v.len()
            );
        }
    };
    table("残り反則別（曲線の形。現行 foul_cost は残数だけの静的関数）", &|p| {
        format!("残り{}", p.2)
    });
    table("手数帯別", &|p| p.4.clone());
    table("相手の残り反則別（効果修飾）", &|p| {
        format!(
            "相手残り{}",
            match p.3 {
                0..=3 => "0-3".to_string(),
                4..=6 => "4-6".to_string(),
                _ => "7-10".to_string(),
            }
        )
    });

    // 終局理由と即負け（機構が壊れていないかの確認）
    let mut reasons: BTreeMap<(String, String), u32> = BTreeMap::new();
    let mut immediate = 0;
    let mut think: Vec<f64> = vec![];
    for r in rows {
        *reasons
            .entry((
                r["arm"].as_str().unwrap_or("?").to_string(),
                r["reason"].as_str().unwrap_or("?").to_string(),
            ))
            .or_insert(0) += 1;
        if r["immediate_loss"].as_bool().unwrap_or(false) {
            immediate += 1;
        }
        think.push(r["think_mean_ms"].as_f64().unwrap_or(0.0));
    }
    println!(
        "  残り1→0 で継続を始めなかった treatment: {immediate} 本（即 foul_limit 負けとして 0 点）"
    );
    println!("  終局理由:");
    for ((arm, reason), n) in &reasons {
        println!("    {arm} / {reason}: {n}");
    }
    println!(
        "  継続の平均思考: {:.0}ms",
        if think.is_empty() {
            0.0
        } else {
            think.iter().sum::<f64>() / think.len() as f64
        }
    );
}

fn check_inputs(metas: &[serde_json::Value], rows: &[serde_json::Value]) -> Vec<String> {
    let mut out = vec![];
    if metas.is_empty() {
        return vec!["meta 行がありません".into()];
    }
    let first = &metas[0]["experiment"];
    if first["seeds"].as_u64().unwrap_or(0) == 0 {
        out.push("meta の seeds が 0 です（継続 0 局のまま影の価格 0 を出せてしまう）".into());
    }
    for (i, m) in metas.iter().enumerate().skip(1) {
        if m["experiment"] != *first {
            let diff: Vec<String> = first
                .as_object()
                .map(|o| {
                    o.keys()
                        .filter(|k| first[k.as_str()] != m["experiment"][k.as_str()])
                        .map(|k| {
                            format!("{k}: {} vs {}", first[k.as_str()], m["experiment"][k.as_str()])
                        })
                        .collect()
                })
                .unwrap_or_default();
            out.push(format!(
                "meta の実験キーが {i} 本目で食い違います（違う実験を混ぜている）: {}",
                diff.join(" / ")
            ));
        }
    }
    let total = first["shard_total"].as_u64().unwrap_or(1) as usize;
    let mut shards: Vec<u64> = metas.iter().filter_map(|m| m["shard"].as_u64()).collect();
    let n = shards.len();
    shards.sort_unstable();
    shards.dedup();
    if shards.len() != n {
        out.push("同じシャードの JSONL を2回渡しています（行が二重に数えられる）".into());
    }
    if shards.len() != total {
        out.push(format!(
            "シャードが欠けています（{shards:?} / 全 {total}）: 判定は出せない"
        ));
    }
    // **ペアの欠落**は致命的（片側だけの継続は差が取れないので黙って落ちる）
    let seeds = first["seeds"].as_u64().unwrap_or(0);
    let mut seen: BTreeMap<(String, u64, String), usize> = BTreeMap::new();
    for r in rows {
        *seen
            .entry((
                r["game"].as_str().unwrap_or("?").to_string(),
                r["move_number"].as_u64().unwrap_or(0),
                r["arm"].as_str().unwrap_or("?").to_string(),
            ))
            .or_default() += 1;
    }
    let mut lacks: Vec<String> = vec![];
    for m in metas {
        for d in m["points_detail"].as_array().into_iter().flatten() {
            let g = d["game"].as_str().unwrap_or("?").to_string();
            let mn = d["move_number"].as_u64().unwrap_or(0);
            for arm in ["control", "treatment"] {
                let got = seen
                    .get(&(g.clone(), mn, arm.to_string()))
                    .copied()
                    .unwrap_or(0) as u64;
                if got != seeds {
                    lacks.push(format!("{g}#{mn} {arm}: {got}/{seeds}"));
                }
            }
        }
    }
    if !lacks.is_empty() {
        out.push(format!(
            "ペアが {} 箇所欠けています（片側だけの継続は差が取れず黙って落ちる）: {}{}",
            lacks.len(),
            lacks.iter().take(3).cloned().collect::<Vec<_>>().join(" / "),
            if lacks.len() > 3 { " ..." } else { "" }
        ));
    }
    out
}

fn run_report(args: &[String]) {
    let allow_incomplete = args.iter().any(|a| a == "--allow-incomplete");
    let paths: Vec<String> = args
        .iter()
        .filter(|a| !a.starts_with("--"))
        .cloned()
        .collect();
    if paths.is_empty() {
        die("report には JSONL を指定してください");
    }
    let mut metas = vec![];
    let mut rows = vec![];
    for p in &paths {
        let text =
            std::fs::read_to_string(p).unwrap_or_else(|e| die(&format!("{p} を読めません: {e}")));
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(line)
                .unwrap_or_else(|_| die(&format!("{p}: JSON として読めない行があります")));
            let schema = v["schema"].as_u64().unwrap_or(0) as u32;
            if schema != ROW_SCHEMA {
                die(&format!(
                    "{p}: schema {schema} は集計できません（現行 {ROW_SCHEMA}）"
                ));
            }
            if v["type"] == "meta" {
                metas.push(v);
            } else {
                rows.push(v);
            }
        }
    }
    println!("JSONL {} 本 / 行 {}", paths.len(), rows.len());
    report(&metas, &rows, allow_incomplete);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exp() -> serde_json::Value {
        serde_json::json!({
            "opponent": "estimator_v14", "budget_ms": 700, "seeds": 2, "per_game": 2,
            "jobs": 4, "shard_total": 1, "config": "c", "source_fingerprint": "s",
            "records": "r",
        })
    }

    fn meta(e: serde_json::Value, shard: u64) -> serde_json::Value {
        serde_json::json!({
            "schema": ROW_SCHEMA, "type": "meta", "experiment": e, "shard": shard,
            "games": 10, "points": 1,
            "points_detail": [{"game": "g1", "move_number": 41, "remaining": 5,
                               "fouls_before": 4, "fouls_in_turn": 1}],
        })
    }

    fn row(arm: &str, seed: u64, score: f64) -> serde_json::Value {
        serde_json::json!({
            "schema": ROW_SCHEMA, "game": "g1", "move_number": 41, "seed": seed, "arm": arm,
            "score": score, "reason": "checkmate", "plies": 90, "added_plies": 10,
            "fouls_me": 5, "remaining": 5, "opp_remaining": 7, "band": "中盤(50-89)",
            "think_mean_ms": 700.0, "immediate_loss": false,
        })
    }

    fn full() -> Vec<serde_json::Value> {
        let mut v = vec![];
        for seed in 0..2 {
            v.push(row("control", seed, 1.0));
            v.push(row("treatment", seed, 0.0));
        }
        v
    }

    #[test]
    fn 揃ったペアは契約を通り差が出る() {
        assert!(check_inputs(&[meta(exp(), 0)], &full()).is_empty());
        let p = pairs(&full());
        assert_eq!(p.len(), 2);
        let (d, _, _) = cluster(&p.iter().map(|(g, d, ..)| (g.clone(), *d)).collect::<Vec<_>>());
        assert!((d + 1.0).abs() < 1e-9, "treatment が全敗なら −1.0: {d}");
    }

    #[test]
    fn 片側だけの継続は契約で止まる() {
        let rows: Vec<serde_json::Value> = full()
            .into_iter()
            .filter(|r| !(r["arm"] == "treatment" && r["seed"] == 1))
            .collect();
        let problems = check_inputs(&[meta(exp(), 0)], &rows);
        assert!(
            problems.iter().any(|p| p.contains("ペアが")),
            "{problems:?}"
        );
        // 片側だけの unit は差の標本からも落ちる（黙って落ちることの明示）
        assert_eq!(pairs(&rows).len(), 1);
    }

    #[test]
    fn seedが0のmetaは判定を出さない() {
        let mut e = exp();
        e["seeds"] = serde_json::json!(0);
        let problems = check_inputs(&[meta(e, 0)], &[]);
        assert!(problems.iter().any(|p| p.contains("seeds")), "{problems:?}");
    }

    #[test]
    fn 継続の乱数はarmに依らない() {
        // control と treatment は共通乱数（差が「反則1個」だけになる）
        assert_eq!(
            continuation_seeds("g1", 41, 0),
            continuation_seeds("g1", 41, 0)
        );
        assert_ne!(
            continuation_seeds("g1", 41, 0),
            continuation_seeds("g1", 41, 1)
        );
        assert_ne!(
            continuation_seeds("g1", 41, 0),
            continuation_seeds("g1", 43, 0)
        );
    }
}
