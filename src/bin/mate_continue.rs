//! **一手強制の継続診断**（issue #28 P0-6）。
//!
//! 詰み負けした各局の「最後に受けられた決定点」から**一手だけ強制**して、
//! そのあとを通常方策で終局まで指し継ぎ、勝ち=1 / 引分=0.5 / 負け=0 で数える。
//! 受けが勝敗へ**どれだけ変換されうるか**の上限と、実現可能な推定値を出す:
//!
//! - `Δoracle` = Σ_局（候補生成に載る真実の安全手のうち最良の継続平均スコア）/ G
//! - `Δpolicy` = Σ_局（P0-3 のオフライン方策が選ぶ手の継続平均スコア）/ G
//!
//! G は**記録の全対局数**（受けの機会が無かった局は 0 として数える）。
//! `Δoracle < 0.04` なら issue #28 の P0 は即中止（受けても救われない）。
//!
//! 裁定は `selfplay::play_continuation` = 通常アリーナと同じ関数（MAX_PLIES・
//! 反則上限・王手/捕獲通知・終局判定を共有）。**時計は無効**（途中局面の残り
//! 時間は復元できない。本番相当で時間切れ0の実測があるので落としてよい）。
//!
//! **`Δoracle` は最良手を選ぶぶん楽観**（勝者の呪い）なので、seed を前半後半に
//! 割って「前半で選び後半で測る」正直版も併せて出す。中止判定は上限側で見る
//! ので楽観は保守的に働くが、`Δpolicy` との差を読むときは正直版を見ること。
//!
//! **replay 対照を必ず取る**: 元対局が負けだからといって「実戦の手を強制した
//! 継続」が 0 になるとは限らない（両者が指し直すので別の対局になる）。
//! `baseline` はその実測で、Δ の一部が「単に指し直したから」でないことを見る。
//!
//! usage:
//!   TSUITATE_THINK_BUDGET_MS=700 cargo run --release --bin mate_continue -- \
//!     [--seeds 2] [--max-safe 4] [--opponent estimator_v14] [--policy-w 4] \
//!     [--jobs N] [--shard i/n] [--out out.jsonl] <records/*.jsonl...>
//!   cargo run --release --bin mate_continue -- report <out-*.jsonl...>
//!
//! **思考予算はプロセス env**（`TSUITATE_THINK_BUDGET_MS`）。両側に同じ値が
//! 効く（凍結相手も同じ env を読む）ので、片側だけ厚くならない。

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tsuitate_bot::checkpoint::stable_hash;
use tsuitate_bot::mate::mate_moves_in_1_fast;
use tsuitate_bot::mate_economy::{analyze_game, force_move, rescored_usis};
use tsuitate_bot::observation::Observation;
use tsuitate_bot::protocol::{Color, GameEndPayload};
use tsuitate_bot::scenario_core::{
    Replayed, build_estimator, clone_log, make_view, ranking_one_with, side_idx,
    weighted_unique_particles,
};
use tsuitate_bot::selfplay::{GameResult, StartState, mix, play_continuation};
use tsuitate_bot::shogi::{Position, ShogiMove};
use tsuitate_bot::strategy::{self, candidate_moves};
use tsuitate_bot::truth_replay::{for_each_decision_full, load_bot_and_end};

const ROW_SCHEMA: u32 = 1;

fn usage() -> &'static str {
    "usage: cargo run --release --bin mate_continue -- [--seeds 2] [--max-safe 4] \
     [--opponent estimator_v14] [--policy-w 4] [--jobs N] [--shard i/n] [--out out.jsonl] \
     <records/*.jsonl...>\n   または: mate_continue report <out-*.jsonl...>"
}

fn die(msg: &str) -> ! {
    eprintln!("{msg}");
    eprintln!("{}", usage());
    std::process::exit(1);
}

/// 強制する手の由来
#[derive(Clone, Debug, PartialEq, Eq)]
enum Arm {
    /// 実戦で指した手（指し直しの対照）
    Baseline,
    /// 真実の安全手（オラクル）
    Oracle,
    /// P0-3 のオフライン方策（厳密粒子の q / taint 込みの q）
    PolicyStrict,
    PolicyAll,
    /// 同じ経路で w=0（= 現行方策そのもの）。**再決定のぶんを切り分ける対照**:
    /// `Δpolicy − Δpolicy(w=0)` が危険量ペナルティ自身の効果になる
    PolicyW0,
}

impl Arm {
    fn tag(&self) -> &'static str {
        match self {
            Arm::Baseline => "baseline",
            Arm::Oracle => "oracle",
            Arm::PolicyStrict => "policy_strict",
            Arm::PolicyAll => "policy_all",
            Arm::PolicyW0 => "policy_w0",
        }
    }
}

struct Point {
    game: String,
    ply: usize,
    rep: Replayed,
    foul_tried: HashSet<String>,
    bot: Color,
    /// 真実の安全手のうち候補生成に載ったもの（オラクルの母集団）
    safe_covered: Vec<String>,
    /// そのうち今回強制したもの（`--max-safe` で絞った後）
    safe_used: Vec<String>,
    played: String,
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


/// 決定点の状態を復元する（`bin/mate_probe` と同じ規約: その手番の反則を
/// 消化した後の状態と、そこまでに試した `foul_tried`）
fn decision_state(end: &GameEndPayload, want: usize) -> Option<(Replayed, HashSet<String>)> {
    let mut out = None;
    for_each_decision_full(end, |d| {
        if d.decision_id as usize != want {
            return;
        }
        let mut foul_tried: HashSet<String> = HashSet::new();
        for e in d.logs[side_idx(d.side)].events().iter().rev() {
            match e {
                Observation::MyFoul { usi, .. } if foul_tried.len() < d.fouls_this_turn as usize => {
                    foul_tried.insert(usi.clone());
                }
                _ => break,
            }
        }
        out = Some((
            Replayed {
                pos: d.pos.clone(),
                logs: [clone_log(&d.logs[0]), clone_log(&d.logs[1])],
                fouls: *d.fouls,
                plies: d.plies,
                injected_fouls: vec![],
                oracle: None,
            },
            foul_tried,
        ));
    });
    out
}

/// 一手強制して継続対局を回し、bot 側のスコア（勝1 / 分0.5 / 負0）を返す。
///
/// `order` は**優先順位つきの候補列**。先頭が真実で非合法なら、実対局と同じく
/// 反則を1つ積んで（手番は変わらない）次の候補へ進む。オフライン方策
/// （`Δpolicy`）は bot の信念で選ぶので**非合法な手を選びうる**ので、
/// ここを「非合法なら測定不能」にすると方策の実力を過大評価する。
/// 反則が上限に達したらその場で反則負け（スコア 0）。
///
/// 近似であることを明示しておく: 実対局なら反則の観測を食った推定器が
/// **選び直す**が、ここでは同じランキングの次点を取る（P0-3 のオフライン
/// ランキングは1回しか作らない）。
fn forced_continuation(
    p: &Point,
    order: &[String],
    seed: u64,
    opponent: &str,
) -> Option<serde_json::Value> {
    // 適用は共有規約（審判と同じ観測の記録・反則の積み方）
    let forced = force_move(&p.rep.pos, &p.rep.logs, p.rep.fouls, p.bot, order);
    let forced_fouls = forced.forced_fouls;
    let fouls = forced.fouls;
    if forced.foul_limit {
        return Some(serde_json::json!({
            "schema": ROW_SCHEMA,
            "game": p.game,
            "ply": p.ply + 1,
            "usi": "",
            "requested": order.first().cloned().unwrap_or_default(),
            "seed": seed,
            "opponent": opponent,
            "score": 0.0,
            "reason": "foul_limit",
            "plies": p.rep.plies,
            "added_plies": 0,
            "added_fouls_me": forced_fouls,
            "added_fouls_opp": 0,
            "forced_fouls": forced_fouls,
            "think_mean_ms": 0.0,
        }));
    }
    let usi = forced.played.clone()?;
    let usi = usi.as_str();
    let pos = forced.pos;
    let logs = forced.logs;

    let me_i = side_idx(p.bot);
    let key = format!("{}#{}#{}", p.game, p.ply, order.first().map_or("", String::as_str));
    let base = stable_hash(seed, &key);
    let seed_me = mix(base ^ 0x0C0F_FEE0);
    let seed_opp = mix(base ^ 0x0BEE_F000);

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
    let start = StartState {
        pos,
        logs,
        fouls,
        plies: p.rep.plies + 1,
    };
    let strategies = [strats[0].take().unwrap(), strats[1].take().unwrap()];
    let out = play_continuation(strategies, start, 0);
    let score = match out.result {
        GameResult::Win(w) if w == p.bot => 1.0,
        GameResult::Win(_) => 0.0,
        GameResult::Draw => 0.5,
    };
    let think: Vec<u64> = out.think_us[me_i].clone();
    let mean_ms = if think.is_empty() {
        0.0
    } else {
        think.iter().sum::<u64>() as f64 / think.len() as f64 / 1000.0
    };
    Some(serde_json::json!({
        "schema": ROW_SCHEMA,
        "game": p.game,
        "ply": p.ply + 1,
        "usi": usi,
        "requested": order.first().cloned().unwrap_or_default(),
        "seed": seed,
        "opponent": opponent,
        "score": score,
        "reason": out.reason,
        "plies": out.plies,
        "added_plies": out.added_plies,
        "added_fouls_me": out.added_fouls[me_i] + forced_fouls,
        "added_fouls_opp": out.added_fouls[1 - me_i],
        "forced_fouls": forced_fouls,
        "think_mean_ms": mean_ms,
    }))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "report") {
        report_files(&args[1..]);
        return;
    }

    let mut seeds: u64 = 2;
    let mut max_safe: usize = 4;
    let mut opponent = "estimator_v14".to_string();
    let mut policy_w: f64 = 4.0;
    let mut jobs: usize =
        std::thread::available_parallelism().map_or(1, |n| n.get().saturating_sub(2).max(1));
    let mut shard = (0usize, 1usize);
    let mut out_path: Option<String> = None;
    let mut specs: Vec<String> = vec![];
    let mut i = 0;
    while i < args.len() {
        let need = |v: Option<&String>, what: &str| -> String {
            v.cloned().unwrap_or_else(|| die(&format!("{what} には値が必要です")))
        };
        match args[i].as_str() {
            "--seeds" => {
                seeds = need(args.get(i + 1), "--seeds").parse().unwrap_or_else(|_| die("--seeds は整数"));
                i += 2;
            }
            "--max-safe" => {
                max_safe = need(args.get(i + 1), "--max-safe").parse().unwrap_or_else(|_| die("--max-safe は整数"));
                i += 2;
            }
            "--opponent" => {
                opponent = need(args.get(i + 1), "--opponent");
                i += 2;
            }
            "--policy-w" => {
                policy_w = need(args.get(i + 1), "--policy-w").parse().unwrap_or_else(|_| die("--policy-w は数値"));
                i += 2;
            }
            "--jobs" => {
                jobs = need(args.get(i + 1), "--jobs").parse::<usize>().unwrap_or_else(|_| die("--jobs は整数")).max(1);
                i += 2;
            }
            "--shard" => {
                let v = need(args.get(i + 1), "--shard");
                let (a, b) = v.split_once('/').unwrap_or_else(|| die("--shard は i/n の形式"));
                shard = (
                    a.parse().unwrap_or_else(|_| die("--shard は i/n の形式")),
                    b.parse().unwrap_or_else(|_| die("--shard は i/n の形式")),
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
    let files = collect_records(&specs);
    if files.is_empty() {
        die("記録ファイルが見つかりません");
    }
    if seeds % 2 != 0 {
        eprintln!("注意: --seeds が奇数だと「前半で選び後半で測る」正直版が偏ります");
    }

    let cfg = tsuitate_bot::config::ambient();
    let budget_ms = cfg.think_budget_ms;
    let scale = budget_ms as f64 / 900.0;

    // ---- 対象の決定点 ----------------------------------------------------
    let mut points: Vec<Point> = vec![];
    let mut games_total = 0u32;
    let mut games_mated = 0u32;
    let mut games_with_defense = 0u32;
    let mut broken = 0u32;
    let mut dropped_safe = 0usize;
    for (gi, path) in files.iter().enumerate() {
        let name = path.to_string_lossy().to_string();
        let Some((bot, end)) = load_bot_and_end(&name) else {
            broken += 1;
            continue;
        };
        let mut mates_of = |p: &Position, _ply: usize| mate_moves_in_1_fast(p);
        let Some(g) = analyze_game(&end, bot, &mut mates_of) else {
            broken += 1;
            continue;
        };
        games_total += 1;
        if !g.bot_mated {
            continue;
        }
        games_mated += 1;
        // シャードは**局単位**で割る（同じ局の arm は必ず同じシャードで走る）
        if shard.1 > 1 && gi % shard.1 != shard.0 {
            continue;
        }
        let Some(turn) = g.last_defense_point() else {
            continue;
        };
        let Some(decision_idx) = turn.decision_idx else {
            continue;
        };
        let Some((rep, foul_tried)) = decision_state(&end, decision_idx) else {
            continue;
        };
        let game = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| name.clone());
        // 漏斗の2段目: 真実の安全手のうち bot の候補生成に載るものだけが
        // オラクルの母集団（載らない手は方策が原理的に選べない）
        let view = make_view(&rep.pos, bot, &rep.fouls);
        let cands: HashSet<String> = candidate_moves(&view, &foul_tried)
            .into_iter()
            .map(|(usi, _)| usi)
            .collect();
        let mut safe_covered: Vec<String> = turn
            .safe
            .iter()
            .map(ShogiMove::to_usi)
            .filter(|u| cands.contains(u))
            .collect();
        safe_covered.sort();
        if safe_covered.is_empty() {
            continue;
        }
        games_with_defense += 1;
        // **黙って切らない**: 上限で落とした本数を記録して最後に出す。
        // 選び方は決定論的（USI の辞書順で等間隔）: 現行方策の順位で選ぶと
        // オラクルが現行方策に引きずられる
        let mut safe_used = safe_covered.clone();
        if safe_used.len() > max_safe && max_safe > 0 {
            let step = safe_used.len() as f64 / max_safe as f64;
            safe_used = (0..max_safe)
                .map(|k| safe_covered[((k as f64 * step) as usize).min(safe_covered.len() - 1)].clone())
                .collect();
            safe_used.dedup();
            dropped_safe += safe_covered.len() - safe_used.len();
        }
        let played = end
            .moves
            .get(decision_idx)
            .map(|m| m.usi.clone())
            .unwrap_or_default();
        points.push(Point {
            game,
            ply: decision_idx,
            rep,
            foul_tried,
            bot,
            safe_covered,
            safe_used,
            played,
        });
    }

    println!(
        "記録 {} 件（壊れ {broken}）/ 局 {games_total}（詰み負け {games_mated}・受けが候補にあった {games_with_defense}）",
        files.len()
    );
    println!(
        "対象の決定点 {}（--max-safe {max_safe} で落とした安全手 {dropped_safe}本）/ 相手 {opponent} / \
         思考予算 {budget_ms}ms / seeds {seeds} / jobs {jobs} / shard {}/{} / policy_w {policy_w}",
        points.len(),
        shard.0,
        shard.1,
    );
    if points.is_empty() {
        die("継続する決定点がありません");
    }

    // ---- P0-3 のオフライン方策が選ぶ手（seed ごと）------------------------
    // ランキングと粒子は `bin/mate_probe` と同じ構築（同じ seed・同じ scale）
    let policy_units: Vec<(usize, u64)> = (0..points.len())
        .flat_map(|p| (0..seeds).map(move |s| (p, s)))
        .collect();
    // 方策は argmax だけでなく**順序**を持つ（先頭が非合法なら次点へ = 実対局の反則）
    let policy: Arc<Mutex<HashMap<(usize, u64), (Vec<String>, Vec<String>, Vec<String>)>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let points_ref = Arc::new(points);
    run_parallel(jobs, policy_units.len(), |ui| {
        let (pi, seed) = policy_units[ui];
        let p = &points_ref[pi];
        let Some((_, ranking)) = ranking_one_with(&p.rep, seed, "estimator", &p.foul_tried) else {
            return;
        };
        let est = build_estimator(&p.rep, seed, scale, |_, _| {});
        let particles = weighted_unique_particles(&est);
        let rows = tsuitate_bot::mate_economy::build_rows(&p.rep.pos, &ranking, &particles);
        // 反則上限（10回）を超えて並べても意味が無いので先頭 12 本で足りる
        let mut strict = rescored_usis(&rows, policy_w, false);
        let mut all = rescored_usis(&rows, policy_w, true);
        let mut w0 = rescored_usis(&rows, 0.0, false);
        strict.truncate(12);
        all.truncate(12);
        w0.truncate(12);
        policy.lock().unwrap().insert((pi, seed), (strict, all, w0));
    });
    let policy = Arc::try_unwrap(policy).ok().unwrap().into_inner().unwrap();

    // ---- 継続対局の unit ---------------------------------------------------
    let mut units: Vec<(usize, u64, Arm, Vec<String>)> = vec![];
    for (pi, p) in points_ref.iter().enumerate() {
        for seed in 0..seeds {
            units.push((pi, seed, Arm::Baseline, vec![p.played.clone()]));
            for usi in &p.safe_used {
                units.push((pi, seed, Arm::Oracle, vec![usi.clone()]));
            }
            if let Some((strict, all, w0)) = policy.get(&(pi, seed)) {
                if !strict.is_empty() {
                    units.push((pi, seed, Arm::PolicyStrict, strict.clone()));
                }
                if !all.is_empty() {
                    units.push((pi, seed, Arm::PolicyAll, all.clone()));
                }
                if !w0.is_empty() {
                    units.push((pi, seed, Arm::PolicyW0, w0.clone()));
                }
            }
        }
    }
    let lines: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(vec![]));
    let started = std::time::Instant::now();
    {
        let units = &units;
        let points_ref = Arc::clone(&points_ref);
        let lines = Arc::clone(&lines);
        run_parallel(jobs, units.len(), move |ui| {
            let (pi, seed, arm, order) = &units[ui];
            let p = &points_ref[*pi];
            if let Some(mut v) = forced_continuation(p, order, *seed, &opponent) {
                v["arm"] = serde_json::json!(arm.tag());
                v["safe_covered"] = serde_json::json!(p.safe_covered.len());
                v["safe_used"] = serde_json::json!(p.safe_used.len());
                lines.lock().unwrap().push(v);
            }
        });
    }
    let mut lines = Arc::try_unwrap(lines).ok().unwrap().into_inner().unwrap();
    lines.sort_by_key(|v| {
        (
            v["game"].as_str().unwrap_or("").to_string(),
            v["arm"].as_str().unwrap_or("").to_string(),
            v["usi"].as_str().unwrap_or("").to_string(),
            v["seed"].as_u64().unwrap_or(0),
        )
    });
    eprintln!(
        "継続 {} 局 / {:.1}分",
        lines.len(),
        started.elapsed().as_secs_f64() / 60.0
    );

    let meta = serde_json::json!({
        "schema": ROW_SCHEMA,
        "type": "meta",
        "games": games_total,
        "games_mated": games_mated,
        "games_with_defense": games_with_defense,
        "points": points_ref.len(),
        "seeds": seeds,
        "max_safe": max_safe,
        "dropped_safe": dropped_safe,
        "opponent_meta": true,
        "budget_ms": budget_ms,
        "policy_w": policy_w,
        "shard": format!("{}/{}", shard.0, shard.1),
        "config": cfg.fingerprint(),
        "source_fingerprint": env!("TSUITATE_SOURCE_FINGERPRINT"),
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
    report(&[meta], &lines);
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

fn report_files(paths: &[String]) {
    if paths.is_empty() {
        die("report には JSONL を指定してください");
    }
    let mut metas = vec![];
    let mut rows = vec![];
    for p in paths {
        let Ok(text) = std::fs::read_to_string(p) else {
            die(&format!("{p}: 読めません"));
        };
        for line in text.lines() {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
                die(&format!("{p}: JSON として読めない行があります"));
            };
            if v["schema"].as_u64() != Some(u64::from(ROW_SCHEMA)) {
                die(&format!("{p}: schema が {ROW_SCHEMA} ではありません（撤回済みの記録は混ぜない）"));
            }
            if v["type"] == "meta" {
                metas.push(v);
            } else {
                rows.push(v);
            }
        }
    }
    report(&metas, &rows);
}

/// 局ごとの平均スコアを arm 別に畳み、Δ と cluster bootstrap CI を出す
fn report(metas: &[serde_json::Value], rows: &[serde_json::Value]) {
    // **シャードは同じ記録集合を見て局単位で割る**ので、`games` は合計ではなく
    // 全シャードで同じ値になる（合計するとシャード数だけ Δ の分母が膨らむ）。
    // 食い違ったら「違う記録集合を混ぜた」ということなので止める
    let game_counts: Vec<u64> = metas.iter().map(|m| m["games"].as_u64().unwrap_or(0)).collect();
    if game_counts.windows(2).any(|w| w[0] != w[1]) {
        die("meta の games がシャード間で食い違います（違う記録集合を混ぜている）");
    }
    let games_total: u64 = game_counts.first().copied().unwrap_or(0);
    let games_mated: u64 = metas
        .iter()
        .map(|m| m["games_mated"].as_u64().unwrap_or(0))
        .max()
        .unwrap_or(0);
    // シャードが揃っているか（Δ の分母は全対局数なので、1シャードだけ集計すると
    // 分母はそのままで分子だけが欠ける = **Δ が机上で小さくなる**）。
    // 揃っていなければ判定を出さない
    let mut shard_ids: Vec<(usize, usize)> = vec![];
    for m in metas {
        let spec = m["shard"].as_str().unwrap_or("0/1");
        if let Some((a, b)) = spec.split_once('/') {
            if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
                shard_ids.push((a, b));
            }
        }
    }
    let shard_total = shard_ids.first().map_or(1, |s| s.1);
    let mut seen: Vec<usize> = shard_ids.iter().map(|s| s.0).collect();
    seen.sort_unstable();
    seen.dedup();
    let shards_complete = shard_ids.iter().all(|s| s.1 == shard_total) && seen.len() == shard_total;

    // 継続した局と落とした安全手はシャードで**割った**ぶんなので合計する
    let dropped: u64 = metas
        .iter()
        .map(|m| m["dropped_safe"].as_u64().unwrap_or(0))
        .sum();
    if games_total == 0 {
        eprintln!("meta 行がありません（Δ の分母が取れない）");
        return;
    }
    // game → arm → usi → seed → score
    type Bucket = BTreeMap<String, BTreeMap<String, BTreeMap<u64, f64>>>;
    let mut per_game: BTreeMap<String, Bucket> = BTreeMap::new();
    for r in rows {
        let game = r["game"].as_str().unwrap_or("?").to_string();
        let arm = r["arm"].as_str().unwrap_or("?").to_string();
        let usi = r["usi"].as_str().unwrap_or("?").to_string();
        let seed = r["seed"].as_u64().unwrap_or(0);
        let score = r["score"].as_f64().unwrap_or(0.0);
        per_game
            .entry(game)
            .or_default()
            .entry(arm)
            .or_default()
            .entry(usi)
            .or_default()
            .insert(seed, score);
    }

    let mean = |v: &[f64]| -> f64 {
        if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
    };
    // 局ごとの寄与（0 = 受けの機会が無かった局）
    let mut contrib: Vec<Contrib> = vec![];
    for (game, arms) in &per_game {
        let arm_mean = |tag: &str| -> f64 {
            arms.get(tag).map_or(0.0, |by_usi| {
                let all: Vec<f64> = by_usi.values().flat_map(|m| m.values().copied()).collect();
                mean(&all)
            })
        };
        let oracle = arms.get("oracle").map_or(0.0, |by_usi| {
            by_usi
                .values()
                .map(|m| mean(&m.values().copied().collect::<Vec<_>>()))
                .fold(0.0, f64::max)
        });
        // 正直版: seed を前半（選ぶ）/ 後半（測る）に割る
        let oracle_honest = arms.get("oracle").map_or(0.0, |by_usi| {
            let seeds: Vec<u64> = by_usi
                .values()
                .flat_map(|m| m.keys().copied())
                .collect::<HashSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            let mut seeds = seeds;
            seeds.sort_unstable();
            if seeds.len() < 2 {
                return f64::NAN;
            }
            let half = seeds.len() / 2;
            let (train, test) = seeds.split_at(half);
            let best = by_usi.iter().max_by(|a, b| {
                let sa = mean(&train.iter().filter_map(|s| a.1.get(s).copied()).collect::<Vec<_>>());
                let sb = mean(&train.iter().filter_map(|s| b.1.get(s).copied()).collect::<Vec<_>>());
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal).then_with(|| b.0.cmp(a.0))
            });
            best.map_or(f64::NAN, |(_, m)| {
                mean(&test.iter().filter_map(|s| m.get(s).copied()).collect::<Vec<_>>())
            })
        });
        contrib.push(Contrib {
            baseline: arm_mean("baseline"),
            oracle,
            oracle_honest,
            policy_strict: arm_mean("policy_strict"),
            policy_all: arm_mean("policy_all"),
            policy_w0: arm_mean("policy_w0"),
        });
        let _ = game;
    }

    let g = games_total as f64;
    let sum = |f: &dyn Fn(&Contrib) -> f64| -> f64 {
        contrib.iter().map(|c| f(c)).filter(|v| v.is_finite()).sum::<f64>() / g
    };
    let d_base = sum(&|c| c.baseline);
    let d_oracle = sum(&|c| c.oracle);
    let d_honest = sum(&|c| c.oracle_honest);
    let d_ps = sum(&|c| c.policy_strict);
    let d_pa = sum(&|c| c.policy_all);
    let d_w0 = sum(&|c| c.policy_w0);

    println!("\n=== P0-6: 一手強制の継続診断 ===");
    println!(
        "全 {games_total}局（詰み負け {games_mated}・継続した局 {}）/ 継続 {} 局ぶん{}",
        contrib.len(),
        rows.len(),
        if dropped > 0 {
            format!("・--max-safe で落とした安全手 {dropped}本（Δoracle は下界）")
        } else {
            String::new()
        }
    );
    // 受けの機会が無かった局は寄与 0 の cluster として**必ず標本に入れる**
    // （再標本化を「受けがあった局」だけに限ると分散が過小になる）
    let ci = |f: &dyn Fn(&Contrib) -> f64| -> (f64, f64) {
        cluster_bootstrap(&contrib, f, games_total as usize)
    };
    let (lo_o, hi_o) = ci(&|c| c.oracle);
    let (lo_ps, hi_ps) = ci(&|c| c.policy_strict);
    let (lo_pa, hi_pa) = ci(&|c| c.policy_all);
    let (lo_b, hi_b) = ci(&|c| c.baseline);
    let (lo_w0, hi_w0) = ci(&|c| c.policy_w0);
    println!("  baseline（実戦の手を強制して指し直し）: {:+.4} [{lo_b:+.4}, {hi_b:+.4}]", d_base);
    println!("  Δoracle（最良の安全手・楽観）:          {:+.4} [{lo_o:+.4}, {hi_o:+.4}]", d_oracle);
    println!("  Δoracle（前半で選び後半で測る正直版）:  {:+.4}", d_honest);
    println!("  Δpolicy（厳密粒子の q）:                {:+.4} [{lo_ps:+.4}, {hi_ps:+.4}]", d_ps);
    println!("  Δpolicy（taint 込みの q）:              {:+.4} [{lo_pa:+.4}, {hi_pa:+.4}]", d_pa);
    println!(
        "  対照 w=0（同じ経路で再決定するだけ）:   {:+.4} [{lo_w0:+.4}, {hi_w0:+.4}] \
         → 危険量ペナルティ自身の効果は 厳密 {:+.4} / taint {:+.4}",
        d_w0,
        d_ps - d_w0,
        d_pa - d_w0
    );
    if shards_complete {
        println!(
            "  判定: Δoracle {} 0.04 → {}",
            if d_oracle < 0.04 { "<" } else { "≥" },
            if d_oracle < 0.04 {
                "**即中止**（受けても救われないので受け項は律速でない）"
            } else {
                "P1 へ進む必要条件のうち Δoracle は満たす（Δpolicy ≥ 0.04 と P0-3 の較正も要る）"
            }
        );
    } else {
        println!(
            "  判定: **出せない**（シャード {}/{} しか揃っていない。Δ の分母は全対局数なので、\
             欠けたシャードのぶんだけ分子だけが落ちて Δ が過小に出る）",
            seen.len(),
            shard_total
        );
    }
    println!(
        "  ※ baseline が 0 から離れているぶんは「指し直したから」の差。Δ から引いて読むこと"
    );

    // 終局理由の内訳（機構が壊れていないかの確認）
    let mut reasons: BTreeMap<(String, String), u32> = BTreeMap::new();
    let mut think: Vec<f64> = vec![];
    let mut fouls: Vec<f64> = vec![];
    for r in rows {
        let arm = r["arm"].as_str().unwrap_or("?").to_string();
        let reason = r["reason"].as_str().unwrap_or("?").to_string();
        *reasons.entry((arm, reason)).or_insert(0) += 1;
        think.push(r["think_mean_ms"].as_f64().unwrap_or(0.0));
        fouls.push(r["added_fouls_me"].as_f64().unwrap_or(0.0));
    }
    let mut arm_rows: BTreeMap<String, u32> = BTreeMap::new();
    for r in rows {
        *arm_rows
            .entry(r["arm"].as_str().unwrap_or("?").to_string())
            .or_insert(0) += 1;
    }
    println!(
        "  arm ごとの継続数: {}（方策が取れなかった決定点×seed は 0 として数えるので、\
         policy_* が baseline より少ないときは Δpolicy がそのぶん下振れする）",
        arm_rows
            .iter()
            .map(|(a, n)| format!("{a} {n}"))
            .collect::<Vec<_>>()
            .join(" / ")
    );
    println!("  終局理由:");
    for ((arm, reason), n) in &reasons {
        println!("    {arm} / {reason}: {n}");
    }
    let forced: Vec<f64> = rows
        .iter()
        .map(|r| r["forced_fouls"].as_f64().unwrap_or(0.0))
        .collect();
    println!(
        "  継続の平均: 思考 {:.0}ms / 追加反則 {:.2}回（うち強制手が非合法で焼いた反則 {:.2}回）",
        mean(&think),
        mean(&fouls),
        mean(&forced)
    );
}

/// 元対局単位の cluster bootstrap（percentile 95% CI）。
/// seed は独立標本として数えない（checkpoint arena と同じ規約）。
/// 局ごとの寄与（arm 別の平均スコア）。cluster bootstrap の統計単位
struct Contrib {
    baseline: f64,
    oracle: f64,
    oracle_honest: f64,
    policy_strict: f64,
    policy_all: f64,
    policy_w0: f64,
}

fn cluster_bootstrap(
    contrib: &[Contrib],
    f: &dyn Fn(&Contrib) -> f64,
    games_total: usize,
) -> (f64, f64) {
    if contrib.is_empty() || games_total == 0 {
        return (f64::NAN, f64::NAN);
    }
    let mut vals: Vec<f64> = contrib
        .iter()
        .map(|c| f(c))
        .map(|v| if v.is_finite() { v } else { 0.0 })
        .collect();
    // 受けの機会が無かった局（寄与 0）も cluster として数える
    vals.resize(vals.len().max(games_total), 0.0);
    let denom = vals.len() as f64;
    // 決定論的な線形合同法（同じ入力なら同じ CI）
    let mut state: u64 = 0x2026_0827;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let reps = 2000;
    let mut draws: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let mut s = 0.0;
        for _ in 0..vals.len() {
            s += vals[next() % vals.len()];
        }
        draws.push(s / denom);
    }
    draws.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    (draws[(reps as f64 * 0.025) as usize], draws[(reps as f64 * 0.975) as usize])
}
