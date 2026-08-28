//! **王手中の反則経済 P0-6: 継続の局所効果**（issue #31。**検証セットでだけ回す**）。
//!
//! 王手中の1手番を方策どおりに指させ（非合法なら実対局と同じく反則を積んで次候補へ）、
//! そのあとを通常方策で終局まで指し継いで 勝1 / 分0.5 / 負0 で数える。
//! P0-5 が出すのは「反則が減ったか・真実指標が良くなったか」で、**勝敗への変換**は
//! ここでしか測れない。
//!
//! **これは一手番の局所効果であって、全局反復適用の下界でも上界でもない**
//! （反復すれば節約が将来にも効いて大きくなることも、状態分布が変わって逆転する
//! こともある）。P1 の arena へ進むための局所的な有効性確認と位置づける。
//!
//! - **1元対局につき estimand ごとに最大1決定点**（同じ元対局の相互排他的な
//!   未来を足し合わせない）。候補が複数あるときは最初の対象手番
//! - estimand は2つ: `foul`（実戦で反則した王手手番＝改善側）と
//!   `nofoul`（反則0の王手手番＝**新しいプローブを足す害**の非劣性）
//! - 継続の乱数は **arm に依らず `(局, 決定点, seed)` だけ**から作る（共通乱数）
//! - Δ の分母は**全対局数**（対象の手番が無かった局は 0）。CI は元対局単位の
//!   cluster bootstrap。**シャードが欠けたら判定を出さない**（#28 の契約）
//!
//! 門（issue #31 で事前登録）:
//!
//! - 反則あり: 主 arm の **`Δpolicy − Δcurrent ≥ +0.04`（CI 下限 > 0）**
//!   かつ foul_limit・破滅率が悪化しない
//! - 反則0: **非劣性**（`≥ −0.01` かつ CI 下限 > −0.02）かつ即時反則・破滅率・
//!   foul_limit が悪化しない。ここで改善は要求しない
//! - **β-order は反則経済施策なので即時反則の非増加を必須**にする。
//!   **full β は反則増を許し**、勝率改善と foul_limit 非悪化で判定する
//!
//! **arm は `--policy` で外から固定する**（P0-4 / P0-5 を見てから主 arm を1本
//! 決める設計なので、水準をコードに埋めない）。発見セットを見てから検証セットで
//! 水準を変えてはいけない。
//!
//! usage:
//!   TSUITATE_THINK_BUDGET_MS=700 cargo run --release --bin check_continue -- \
//!     [--seeds 4] [--opponent estimator_v14] [--policy alpha@k2] \
//!     [--jobs N] [--shard i/n] [--out out.jsonl] <records/*.jsonl...>
//!   cargo run --release --bin check_continue -- report [--allow-incomplete] <out-*.jsonl...>
//!
//! **思考予算はプロセス env で両側・両段に同じ値が効く**（`bin/mate_continue` と
//! 同じ規約）。つまりここで測るのは「その予算での方策」の局所効果で、
//! ランキングも継続も同じ動作点に揃う。

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tsuitate_bot::check_economy::entry_replayed;
use tsuitate_bot::check_policy::{EntrySetup, Policy, UpdateRule, entry_setup, simulate};
use tsuitate_bot::checkpoint::stable_hash;
use tsuitate_bot::mate_economy::force_move;
use tsuitate_bot::observation::Observation;
use tsuitate_bot::protocol::Color;
use tsuitate_bot::scenario_core::{Replayed, clone_log, make_view, side_idx};
use tsuitate_bot::selfplay::{GameResult, StartState, mix, play_continuation};
use tsuitate_bot::shogi::Position;
use tsuitate_bot::strategy::{self, EvalParams};
use tsuitate_bot::truth_replay::{for_each_decision_full, parse_bot_and_end};

const ROW_SCHEMA: u32 = 1;

fn die(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(2);
}

struct Point {
    game: String,
    move_number: u32,
    estimand: &'static str,
    /// 手番開始時（反則0）の状態
    entry: Replayed,
    /// 決定点の真実局面（裁定用。方策には渡さない）
    truth: Position,
    bot: Color,
    /// 実戦の反則列＋受理手（`baseline` arm が強制する列）
    record_order: Vec<String>,
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

/// 継続の乱数（bot 側・相手側）。**arm に依らず (局, 決定点, seed) だけ**から作る。
///
/// 強制した手を hash に混ぜると arm ごとに別の乱数列で継続することになり、
/// `Δpolicy − Δcurrent` が共通乱数のペア差にならない（issue #28 の教訓④）。
fn continuation_seeds(game: &str, ply: u32, seed: u64) -> (u64, u64) {
    let base = stable_hash(seed, &format!("{game}#{ply}"));
    (mix(base ^ 0x31C0_FFEE), mix(base ^ 0x31BE_EF00))
}

/// 手番を `order` の順に強制して指させ、そのあとを終局まで指し継ぐ
fn run_arm(p: &Point, order: &[String], seed: u64, opponent: &str) -> serde_json::Value {
    let me_i = side_idx(p.bot);
    let logs = [clone_log(&p.entry.logs[0]), clone_log(&p.entry.logs[1])];
    // 適用は共有規約（審判と同じ観測の記録・反則の積み方）
    let forced = force_move(&p.entry.pos, &logs, p.entry.fouls, p.bot, order);
    let forced_fouls = forced.forced_fouls;
    let base = |score: f64, reason: &str, plies: u32, added: u32, think: f64, extra: u32| {
        serde_json::json!({
            "schema": ROW_SCHEMA,
            "game": p.game, "move_number": p.move_number, "estimand": p.estimand,
            "seed": seed, "opponent": opponent,
            "score": score, "reason": reason, "plies": plies, "added_plies": added,
            // **即時反則** = その手番で積んだ反則（β-order の門はここを見る）
            "immediate_fouls": forced_fouls,
            "added_fouls_me": forced_fouls + extra,
            "think_mean_ms": think,
            "foul_limit": reason == "foul_limit",
        })
    };
    if forced.foul_limit || forced.played.is_none() {
        // その手番で反則上限に達した（＝その場で反則負け）／候補を出せなかった
        return base(0.0, "foul_limit", p.entry.plies, 0, 0.0, 0);
    }
    let pos = forced.pos;
    let logs = forced.logs;
    let fouls = forced.fouls;
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
        let view = make_view(&pos, color, &fouls);
        strategy::prewarm_strategy(&mut *strat, &view, &logs[i]);
        strats[i] = Some(strat);
    }
    let start = StartState { pos, logs, fouls, plies: p.entry.plies + 1 };
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
    let mut row = base(
        score,
        out.reason,
        out.plies,
        out.added_plies,
        mean_ms,
        out.added_fouls[me_i],
    );
    row["played"] = serde_json::json!(forced.played);
    row
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "report") {
        run_report(&args[1..]);
        return;
    }
    let mut seeds: u64 = 4;
    let mut opponent = "estimator_v14".to_string();
    // **主 arm は外から固定する**（P0-4 / P0-5 を見てから1本決める）
    let mut policy_tags: Vec<String> = vec!["current".into()];
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
            "--opponent" => {
                opponent = need(args.get(i + 1), "--opponent");
                i += 2;
            }
            "--policy" => {
                policy_tags = need(args.get(i + 1), "--policy")
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
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
        die("--seeds は 1 以上にしてください（0 だと継続を1局も走らせずに Δ が全部 0 になる）");
    }
    // `current` は必ず対照として要る（Δpolicy − Δcurrent が門なので）
    if !policy_tags.iter().any(|t| t == "current") {
        policy_tags.insert(0, "current".into());
    }
    let policies: Vec<(String, Policy)> = policy_tags
        .iter()
        .map(|t| {
            (
                t.clone(),
                Policy::parse(t).unwrap_or_else(|| die(&format!("未知の方策: {t}"))),
            )
        })
        .collect();
    let files = collect_records(&specs);
    if files.is_empty() {
        die("記録ファイルが見つかりません");
    }
    let cfg = tsuitate_bot::config::ambient();
    let params = EvalParams::default();
    let eval_particles = strategy::eval_particles_for_budget(cfg.think_budget_ms);

    // ---- 決定点（1元対局につき estimand ごとに最大1つ）----------------------
    use sha2::Digest as _;
    let mut digest = sha2::Sha256::new();
    let mut points: Vec<Point> = vec![];
    let mut games = 0u32;
    let mut broken = 0u32;
    let mut record_opponents: BTreeMap<String, u32> = BTreeMap::new();
    let mut mismatched = 0u32;
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
        if end.opponent.username != opponent {
            mismatched += 1;
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
            let post = Replayed {
                pos: d.pos.clone(),
                logs: [clone_log(&d.logs[0]), clone_log(&d.logs[1])],
                fouls: *d.fouls,
                plies: d.plies,
                injected_fouls: vec![],
                oracle: None,
            };
            let Some(entry) = entry_replayed(&post, d.side, d.fouls_this_turn) else {
                return;
            };
            let events = post.logs[side_idx(d.side)].events();
            let mut record_order: Vec<String> = events
                [events.len() - d.fouls_this_turn as usize..]
                .iter()
                .filter_map(|e| match e {
                    Observation::MyFoul { usi, .. } => Some(usi.clone()),
                    _ => None,
                })
                .collect();
            let estimand = if record_order.is_empty() { "nofoul" } else { "foul" };
            if let Some(accepted) = end.moves.get(d.decision_id as usize) {
                record_order.push(accepted.usi.clone());
            }
            if record_order.is_empty() {
                return;
            }
            found.push(Point {
                game: short.clone(),
                move_number: d.pos.move_number(),
                estimand,
                truth: d.pos.clone(),
                entry,
                bot,
                record_order,
            });
        });
        if !ok {
            broken += 1;
            continue;
        }
        games += 1;
        // **estimand ごとに最初の1つだけ**（同じ元対局の相互排他的な未来を
        // 足し合わせない。issue #28 P0-6 の契約と同じ）
        for estimand in ["foul", "nofoul"] {
            if let Some(p) = found.iter().position(|p| p.estimand == estimand) {
                points.push(found.remove(p));
            }
        }
    }
    if mismatched > 0 && !allow_opponent_mismatch {
        eprintln!(
            "元対局の相手が --opponent {opponent} と一致しない記録が {mismatched} 件あります: {}",
            record_opponents
                .iter()
                .map(|(k, n)| format!("{k} {n}局"))
                .collect::<Vec<_>>()
                .join(" / ")
        );
        die("Δ が「受けの効果」と「相手が変わった効果」の混合になるので中止しました（--allow-opponent-mismatch で強行）");
    }
    points.sort_by(|a, b| (&a.game, a.move_number).cmp(&(&b.game, b.move_number)));
    // シャードは**局単位**（同じ局の arm は必ず同じシャードで走る）
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
        die("対象の決定点がありません");
    }
    println!(
        "記録 {} 件（壊れ {broken}）/ 局 {games} / 決定点 {}（反則あり {} / 反則0 {}）",
        files.len(),
        points.len(),
        points.iter().filter(|p| p.estimand == "foul").count(),
        points.iter().filter(|p| p.estimand == "nofoul").count(),
    );
    println!(
        "相手 {opponent} / 方策 {} / 思考予算 {}ms / seeds {seeds} / jobs {jobs} / shard {}/{}",
        policy_tags.join(","),
        cfg.think_budget_ms,
        shard.0,
        shard.1,
    );
    println!("source_fingerprint {}", env!("TSUITATE_SOURCE_FINGERPRINT"));

    // ---- 方策ごとの強制列（決定点 × seed）--------------------------------
    // P0-5 と**同じ配管**（`entry_setup` → `simulate`）で作る。simulate の
    // `sequence` は「反則した手…受理された手」なので、そのまま `force_move` へ
    // 渡せば実対局と同じ裁定を再現する
    let points = Arc::new(points);
    let orders: Arc<Mutex<BTreeMap<(usize, u64), BTreeMap<String, Vec<String>>>>> =
        Arc::new(Mutex::new(BTreeMap::new()));
    let policy_units: Vec<(usize, u64)> = (0..points.len())
        .flat_map(|p| (0..seeds).map(move |s| (p, s)))
        .collect();
    let policy_jobs = jobs.min(policy_units.len().max(1)).max(1);
    run_parallel(jobs, policy_units.len(), |ui| {
        let (pi, seed) = policy_units[ui];
        let p = &points[pi];
        let Some(EntrySetup { moves, p0, updater, .. }) =
            entry_setup(&p.entry, &p.truth, seed, &params, eval_particles)
        else {
            return;
        };
        let fouls_before = p.entry.fouls[side_idx(p.entry.pos.turn())];
        let opp_fouls = p.entry.fouls[side_idx(p.entry.pos.turn().other())];
        let mut by_arm: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (tag, policy) in &policies {
            let out = simulate(
                policy,
                &moves,
                &p0,
                &params,
                fouls_before,
                opp_fouls,
                UpdateRule::Shadow(&updater),
            );
            if !out.sequence.is_empty() {
                by_arm.insert(tag.clone(), out.sequence);
            }
        }
        orders.lock().unwrap().insert((pi, seed), by_arm);
    });
    let orders = Arc::try_unwrap(orders).ok().unwrap().into_inner().unwrap();

    // ---- 継続（同じ強制列は1回だけ走らせて該当 arm へ配る）-----------------
    let mut units: Vec<(usize, u64, Vec<String>, Vec<String>)> = vec![];
    for (pi, p) in points.iter().enumerate() {
        for seed in 0..seeds {
            let mut by_order: BTreeMap<Vec<String>, Vec<String>> = BTreeMap::new();
            by_order
                .entry(p.record_order.clone())
                .or_default()
                .push("baseline".to_string());
            if let Some(by_arm) = orders.get(&(pi, seed)) {
                for (arm, order) in by_arm {
                    by_order.entry(order.clone()).or_default().push(arm.clone());
                }
            }
            for (order, arms) in by_order {
                units.push((pi, seed, order, arms));
            }
        }
    }
    let lines: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(vec![]));
    let continuation_jobs = jobs.min(units.len().max(1)).max(1);
    let started = std::time::Instant::now();
    {
        let lines = Arc::clone(&lines);
        let points = Arc::clone(&points);
        let units = &units;
        let opponent = opponent.as_str();
        run_parallel(jobs, units.len(), move |ui| {
            let (pi, seed, order, arms) = &units[ui];
            let row = run_arm(&points[*pi], order, *seed, opponent);
            let mut out = lines.lock().unwrap();
            for arm in arms {
                // 同じ強制列 = 同じ継続なので、arm ごとに同じ結果を配る
                let mut r = row.clone();
                r["arm"] = serde_json::json!(arm);
                out.push(r);
            }
        });
    }
    let mut lines = Arc::try_unwrap(lines).ok().unwrap().into_inner().unwrap();
    lines.sort_by_key(|v| {
        (
            v["game"].as_str().unwrap_or("").to_string(),
            v["move_number"].as_u64().unwrap_or(0),
            v["arm"].as_str().unwrap_or("").to_string(),
            v["seed"].as_u64().unwrap_or(0),
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
        "policies": policy_tags,
        "policy_jobs": policy_jobs,
        "continuation_jobs": continuation_jobs,
        "games": games,
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
                "estimand": p.estimand,
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

/// `n` 個の unit を `jobs` 本のスレッドで回す（実効並列度は unit 数で clamp）
fn run_parallel(jobs: usize, n: usize, f: impl Fn(usize) + Sync + Send) {
    if n == 0 {
        return;
    }
    let next = Arc::new(Mutex::new(0usize));
    let effective = jobs.min(n).max(1);
    let f = &f;
    std::thread::scope(|scope| {
        for _ in 0..effective {
            let next = Arc::clone(&next);
            scope.spawn(move || {
                loop {
                    let ui = {
                        let mut g = next.lock().unwrap();
                        if *g >= n {
                            break;
                        }
                        let v = *g;
                        *g += 1;
                        v
                    };
                    f(ui);
                }
            });
        }
    });
}

// ---- 集計 ------------------------------------------------------------------

/// 局ごとの寄与（arm 別の平均値）。cluster bootstrap の統計単位は元対局
fn contributions(
    rows: &[&serde_json::Value],
    key: &dyn Fn(&serde_json::Value) -> f64,
) -> BTreeMap<String, BTreeMap<String, f64>> {
    // game → arm → 値の平均
    let mut sums: BTreeMap<String, BTreeMap<String, (f64, f64)>> = BTreeMap::new();
    for r in rows {
        let e = sums
            .entry(r["game"].as_str().unwrap_or("?").to_string())
            .or_default()
            .entry(r["arm"].as_str().unwrap_or("?").to_string())
            .or_default();
        e.0 += key(r);
        e.1 += 1.0;
    }
    sums.into_iter()
        .map(|(g, m)| {
            (
                g,
                m.into_iter()
                    .map(|(a, (s, n))| (a, if n > 0.0 { s / n } else { 0.0 }))
                    .collect(),
            )
        })
        .collect()
}

/// Δ（= Σ_局 寄与 / 全対局数）と、元対局単位の cluster bootstrap CI
fn delta_ci(
    contrib: &BTreeMap<String, BTreeMap<String, f64>>,
    arm: &str,
    base: Option<&str>,
    games_total: usize,
) -> (f64, f64, f64) {
    if games_total == 0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let mut vals: Vec<f64> = contrib
        .values()
        .map(|m| {
            let a = m.get(arm).copied().unwrap_or(0.0);
            match base {
                Some(b) => a - m.get(b).copied().unwrap_or(0.0),
                None => a,
            }
        })
        .collect();
    // **対象の手番が無かった局も寄与 0 の cluster として標本に入れる**
    // （再標本化を「対象があった局」だけに限ると分散が過小になる）
    vals.resize(vals.len().max(games_total), 0.0);
    let point = vals.iter().sum::<f64>() / vals.len() as f64;
    let mut state: u64 = 0x3120_2606;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let reps = 2000;
    let mut draws: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let mut s = 0.0;
        for _ in 0..vals.len() {
            s += vals[next() % vals.len()];
        }
        draws.push(s / vals.len() as f64);
    }
    draws.sort_by(f64::total_cmp);
    (
        point,
        draws[(reps as f64 * 0.025) as usize],
        draws[(reps as f64 * 0.975) as usize],
    )
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
    let games_total = exp["games"].as_u64().unwrap_or(0) as usize;
    println!("\n=== P0-6: 継続の局所効果（issue #31。検証セット）===");
    println!(
        "  実験: 相手 {} / 予算 {}ms / seeds {} / 方策 {} / 全 {games_total}局 / 記録 {} / code {}",
        exp["opponent"], exp["budget_ms"], exp["seeds"], exp["policies"], exp["records"],
        exp["source_fingerprint"],
    );
    // シャードが揃っているか（Δ の分母は全対局数なので、欠けると分子だけが落ちる）
    let shard_total = exp["shard_total"].as_u64().unwrap_or(1) as usize;
    let mut seen: Vec<u64> = metas.iter().filter_map(|m| m["shard"].as_u64()).collect();
    seen.sort_unstable();
    seen.dedup();
    let complete = seen.len() == shard_total;
    if games_total == 0 {
        println!("  meta の games が 0（Δ の分母が取れない）");
        return;
    }

    for estimand in ["foul", "nofoul"] {
        let sel: Vec<&serde_json::Value> =
            rows.iter().filter(|r| r["estimand"] == estimand).collect();
        if sel.is_empty() {
            continue;
        }
        let arms: BTreeSet<String> = sel
            .iter()
            .map(|r| r["arm"].as_str().unwrap_or("?").to_string())
            .collect();
        let score = contributions(&sel, &|r| r["score"].as_f64().unwrap_or(0.0));
        let fouls = contributions(&sel, &|r| r["immediate_fouls"].as_f64().unwrap_or(0.0));
        let limit = contributions(&sel, &|r| {
            f64::from(u8::from(r["foul_limit"].as_bool().unwrap_or(false)))
        });
        println!(
            "\n--- estimand {estimand}（決定点のある局 {} / 継続 {} 本）---",
            score.len(),
            sel.len()
        );
        println!(
            "  {:<18} {:>9} {:>26} {:>10} {:>10}",
            "arm", "Δ", "Δ − Δcurrent [CI]", "即時反則", "反則負け%"
        );
        for arm in &arms {
            let (d, _, _) = delta_ci(&score, arm, None, games_total);
            let (dd, lo, hi) = delta_ci(&score, arm, Some("current"), games_total);
            let (f, _, _) = delta_ci(&fouls, arm, None, games_total);
            let (fl, _, _) = delta_ci(&limit, arm, None, games_total);
            println!(
                "  {arm:<18} {d:>+9.4} {:>26} {f:>10.3} {:>10.1}",
                if arm == "current" {
                    "—".to_string()
                } else {
                    format!("{dd:+.4} [{lo:+.4}, {hi:+.4}]")
                },
                100.0 * fl,
            );
        }
        if !complete {
            continue;
        }
        // 門（issue #31 で事前登録）
        println!("  判定:");
        for arm in arms.iter().filter(|a| *a != "current" && *a != "baseline") {
            let (dd, lo, hi) = delta_ci(&score, arm, Some("current"), games_total);
            let (df, _, _) = delta_ci(&fouls, arm, Some("current"), games_total);
            let (dl, _, _) = delta_ci(&limit, arm, Some("current"), games_total);
            let verdict = if estimand == "foul" {
                // 反則あり: Δpolicy − Δcurrent ≥ +0.04 かつ CI 下限 > 0 かつ
                // foul_limit・破滅が悪化しない
                if dd >= 0.04 && lo > 0.0 && dl <= 0.0 {
                    "**門を超える**"
                } else if dd >= 0.04 && lo > 0.0 {
                    "改善量は門を超えるが反則負けが悪化（不合格）"
                } else {
                    "不合格"
                }
            } else {
                // 反則0: 非劣性（≥ −0.01 かつ CI 下限 > −0.02）かつ即時反則が悪化しない
                if dd >= -0.01 && lo > -0.02 && df <= 0.0 {
                    "**非劣性を満たす**"
                } else if dd >= -0.01 && lo > -0.02 {
                    "非劣性は満たすが即時反則が増えている（β-order なら不合格）"
                } else {
                    "非劣性を満たさない"
                }
            };
            println!(
                "    {arm:<18} Δ差 {dd:+.4} [{lo:+.4}, {hi:+.4}] / 即時反則 {df:+.3} / 反則負け {dl:+.4} → {verdict}"
            );
        }
    }
    if !complete {
        println!(
            "\n  **判定は出せない**（シャード {}/{} しか揃っていない。Δ の分母は全対局数なので、\
欠けたシャードのぶんだけ分子だけが落ちて Δ が過小に出る）",
            seen.len(),
            shard_total
        );
    }
    println!(
        "\n  ※ baseline（実戦の反則列＋受理手を強制して指し直し）が 0 から離れているぶんは\
「単に指し直したから」の差。Δ から引いて読むこと"
    );
    println!(
        "  ※ **一手番の局所効果**であって全局反復適用の下界でも上界でもない\
（P1 の arena へ進むための有効性確認）"
    );
    let mut reasons: BTreeMap<(String, String), u32> = BTreeMap::new();
    let mut think: Vec<f64> = vec![];
    for r in rows {
        *reasons
            .entry((
                r["arm"].as_str().unwrap_or("?").to_string(),
                r["reason"].as_str().unwrap_or("?").to_string(),
            ))
            .or_insert(0) += 1;
        think.push(r["think_mean_ms"].as_f64().unwrap_or(0.0));
    }
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
        return vec!["meta 行がありません（Δ の分母が取れない）".into()];
    }
    let first = &metas[0]["experiment"];
    if first["seeds"].as_u64().unwrap_or(0) == 0 {
        out.push("meta の seeds が 0 です（継続 0 局のまま Δ を 0 にできてしまう）".into());
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
    let mut shards: Vec<u64> = metas.iter().filter_map(|m| m["shard"].as_u64()).collect();
    let n = shards.len();
    shards.sort_unstable();
    shards.dedup();
    if shards.len() != n {
        out.push("同じシャードの JSONL を2回渡しています（行が二重に数えられる）".into());
    }
    let total = first["shard_total"].as_u64().unwrap_or(1) as usize;
    if shards.len() != total {
        out.push(format!(
            "シャードが欠けています（{shards:?} / 全 {total}）: Δ の分子だけが落ちる"
        ));
    }
    // 重複行
    let mut keys: HashSet<(String, u64, u64, String)> = HashSet::new();
    let mut dups = 0;
    for r in rows {
        let k = (
            r["game"].as_str().unwrap_or("?").to_string(),
            r["move_number"].as_u64().unwrap_or(0),
            r["seed"].as_u64().unwrap_or(0),
            r["arm"].as_str().unwrap_or("?").to_string(),
        );
        if !keys.insert(k) {
            dups += 1;
        }
    }
    if dups > 0 {
        out.push(format!("重複行が {dups} 件あります"));
    }
    // **meta が宣言した決定点に対して行が揃っているか**（ある seed の全 arm が
    // まとめて欠けても検出する。issue #28 PR #30 レビュー2巡目 [P1]）
    let seeds = first["seeds"].as_u64().unwrap_or(0);
    let mut want: Vec<String> = vec!["baseline".into()];
    for p in first["policies"].as_array().into_iter().flatten() {
        want.push(p.as_str().unwrap_or("?").to_string());
    }
    let mut seen_rows: BTreeMap<(String, u64, String), usize> = BTreeMap::new();
    for r in rows {
        *seen_rows
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
            for arm in &want {
                let got = seen_rows
                    .get(&(g.clone(), mn, arm.clone()))
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
            "meta が宣言した決定点に対して行が {} 箇所欠けています（Δ の分子が欠ける）: {}{}",
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
            "opponent": "estimator_v14", "budget_ms": 700, "seeds": 2,
            "policies": ["current", "alpha@k2"], "policy_jobs": 3, "continuation_jobs": 3,
            "games": 104, "shard_total": 1, "config": "c", "source_fingerprint": "s",
            "records": "r",
        })
    }

    fn meta(e: serde_json::Value, shard: u64) -> serde_json::Value {
        serde_json::json!({
            "schema": ROW_SCHEMA, "type": "meta", "experiment": e, "shard": shard,
            "games": 104, "points": 1,
            "points_detail": [{"game": "g1", "move_number": 41, "estimand": "foul"}],
        })
    }

    fn row(arm: &str, seed: u64, score: f64) -> serde_json::Value {
        serde_json::json!({
            "schema": ROW_SCHEMA, "game": "g1", "move_number": 41, "estimand": "foul",
            "seed": seed, "arm": arm, "score": score, "reason": "checkmate",
            "plies": 90, "added_plies": 10, "immediate_fouls": 1, "added_fouls_me": 1,
            "think_mean_ms": 700.0, "foul_limit": false,
        })
    }

    fn full() -> Vec<serde_json::Value> {
        let mut v = vec![];
        for seed in 0..2 {
            v.push(row("baseline", seed, 0.0));
            v.push(row("current", seed, 0.0));
            v.push(row("alpha@k2", seed, 1.0));
        }
        v
    }

    #[test]
    fn 揃った入力は契約を通る() {
        assert!(check_inputs(&[meta(exp(), 0)], &full()).is_empty());
    }

    #[test]
    fn armが欠けた決定点を検出する() {
        let rows: Vec<serde_json::Value> = full()
            .into_iter()
            .filter(|r| !(r["arm"] == "alpha@k2" && r["seed"] == 1))
            .collect();
        let problems = check_inputs(&[meta(exp(), 0)], &rows);
        assert!(problems.iter().any(|p| p.contains("alpha@k2")), "{problems:?}");
    }

    #[test]
    fn シャードが欠けたら止まる() {
        let mut e = exp();
        e["shard_total"] = serde_json::json!(2);
        let problems = check_inputs(&[meta(e, 0)], &full());
        assert!(
            problems.iter().any(|p| p.contains("シャードが欠けて")),
            "{problems:?}"
        );
    }

    /// **Δ の分母は全対局数**（対象の手番が無かった局は寄与 0）。
    /// 分母を「対象があった局」に取り替えると Δ が水増しされる
    #[test]
    fn deltaの分母は全対局数() {
        let rows = full();
        let refs: Vec<&serde_json::Value> = rows.iter().collect();
        let c = contributions(&refs, &|r| r["score"].as_f64().unwrap_or(0.0));
        assert_eq!(c.len(), 1, "1局ぶんの寄与");
        // 1局だけスコア 1.0 → 全104局で割るので Δ = 1/104
        let (d, lo, hi) = delta_ci(&c, "alpha@k2", None, 104);
        assert!((d - 1.0 / 104.0).abs() < 1e-9, "{d}");
        assert!(lo <= d && d <= hi);
        // ペア差（current は 0 なので同じ値）
        let (dd, _, _) = delta_ci(&c, "alpha@k2", Some("current"), 104);
        assert!((dd - 1.0 / 104.0).abs() < 1e-9, "{dd}");
        // 分母を1局にすると 1.0 になる = 分母の取り違えは Δ を100倍にする
        let (bad, _, _) = delta_ci(&c, "alpha@k2", None, 1);
        assert!((bad - 1.0).abs() < 1e-9);
    }

    #[test]
    fn 継続の乱数はarmに依らない() {
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
