//! **王手中の反則経済 P0-4: w\* 監査**（issue #31。runtime には何も入らない）。
//!
//! 「浮かせたい候補がそもそも在るのか」を、方策を変える前に測る
//! （`TSUITATE_PROBE_AUDIT` と同じ考え方）。アリーナ記録の**王手中で反則した
//! 手番**を復元し、決定点ごとに:
//!
//! - **k\***: 王手中の反則コストを何倍にすれば**玉の手が首位になるか**。
//!   価格は玉の手の (1−p) 側にも効くので、score 差 ÷ 傾斜ではなく
//!   **P1-α と同じ score 関数を k で再計算して交点を探す**
//! - **反則注入後の再決定**: 実戦の反則列を食わせて `Estimator::update` を
//!   通した本物の再ランキングが、「初回ランキングの次点を取る」静的方策と
//!   **違う手**を選ぶか（H2 の材料。反則が減るか増えるかは問わない）
//!
//! 中止条件（issue #31）: **k\* ≤ 3 で反転する決定点が 20% 未満 かつ
//! 反則注入後に首位が変わる決定点も 20% 未満**なら α / β の枝は中止
//! （m061/m036 型の「候補が全部悪手」）。
//!
//! usage:
//!   TSUITATE_THINK_BUDGET_MS=2000 cargo run --release --bin check_probe -- \
//!     [--seeds 3] [--jobs N] [--limit N] [--shard i/n] [--kmax 30] \
//!     [--opponent estimator_v14] [--types nonking_king,nonking_nonking] \
//!     [--out data/check_probe.csv] <records...>

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tsuitate_bot::check::CheckSolver;
use tsuitate_bot::check_economy::{
    CheckMoveKind, PricedMove, classify_move_kind, hypothesis_stats, k_star, priced_moves,
};
use tsuitate_bot::observation::{Observation, ObservationLog};
use tsuitate_bot::protocol::Color;
use tsuitate_bot::scenario_core::{Replayed, clone_log, make_view, ranking_one_with, side_idx};
use tsuitate_bot::shogi::parse_usi;
use tsuitate_bot::strategy::EvalParams;
use tsuitate_bot::truth_replay::{for_each_decision_full, load_bot_and_end};

fn die(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(2);
}

/// 監査対象の決定点（王手中で実際に反則した手番1つ）
struct Point {
    game: String,
    /// 決定点の手数（1始まりの手目 = pos.move_number()）
    move_number: u32,
    /// 型（最初の反則の手種 → 受理された手の手種）の粗い束
    type_tag: &'static str,
    /// 実戦の反則列（順番どおり）と受理された手
    fouls: Vec<String>,
    accepted: String,
    /// 手番開始時（反則0）の状態と、実戦の反則を食った後の状態
    entry: Replayed,
    post: Replayed,
    /// 開始時の残り反則
    remaining: u32,
    /// 王手駒仮説の質（真の王手駒の重みシェア / 正規化エントロピー）
    true_hyp_share: f64,
    hyp_entropy: f64,
}

/// 1 unit（決定点 × seed）の結果
struct UnitOut {
    point: usize,
    seed: u64,
    /// 初回ランキングの首位（乱数を除いた決定的スコア順）
    top1: String,
    top1_is_king: bool,
    /// 最良の玉の手と、そのスコア・p_legal
    king_best: Option<String>,
    p_top1: f64,
    p_king: f64,
    /// 首位と最良の玉の手の決定的スコア差
    score_gap: f64,
    /// 玉の手が首位になる反則コスト倍率（None = kmax までに反転しない）。
    /// 首位が既に玉の手なら 1.0 になるので、門の集計からは
    /// `top1_is_king` の unit を外す（反転していないため）
    k_star: Option<f64>,
    /// 初回ランキングの首位が**実戦の最初の反則と一致**したか
    /// （オフライン復元が実戦の選択をどれだけ再現しているかの検査）
    matches_record: bool,
    /// 「初回ランキングの次点」（実戦の反則列を除いた最上位）
    next_static: Option<String>,
    /// 反則注入後の実再決定の首位
    top1_after: Option<String>,
    /// 実再決定が静的な次点と違う手を選んだか（H2）
    changed: Option<bool>,
    /// 最良の玉の手の p_legal が反則注入後にどう動いたか
    p_king_after: Option<f64>,
    candidates: usize,
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

/// 反則コストの**床適用前の基準値**（`strategy::evaluate` と同じ式）。
///
/// `CandidateScore::foul_cost` は床（`last_foul_guard`）を適用した後の値なので、
/// P1-α の `max(k × base, 床)` を再現するには基準値のほうが要る。
fn base_foul_cost(params: &EvalParams, you: u32, opp: u32) -> f64 {
    let fouls_left = (10u32.saturating_sub(you)).max(1) as f64;
    let opp_left = (10u32.saturating_sub(opp)).max(1) as f64;
    params.foul_cost_base
        * (10.0 / fouls_left).powf(params.foul_cost_pow)
        * (opp_left / 10.0).powf(params.foul_diff_pow)
}

/// 手番開始時（反則0）の状態を作る。
///
/// `for_each_decision_full` が渡すのは**その手番の反則を全部食った後**の状態
/// なので、両者のログ末尾から反則の観測を `fouls_this_turn` 本だけ落とす
/// （手番側は `MyFoul`・相手側は `OpponentFoul`。`add_foul_obs` が対で積む）。
/// 反則は局面を変えないので `pos` はそのまま。
fn entry_replayed(post: &Replayed, side: Color, fouls_this_turn: u32) -> Option<Replayed> {
    let n = fouls_this_turn as usize;
    let mut logs = [ObservationLog::default(), ObservationLog::default()];
    for (idx, log) in logs.iter_mut().enumerate() {
        let events = post.logs[idx].events();
        let keep = events.len().checked_sub(n)?;
        // 落とす末尾が本当に反則の観測かを確かめる（規約が変わったら止める）
        for e in &events[keep..] {
            let ok = if idx == side_idx(side) {
                matches!(e, Observation::MyFoul { .. })
            } else {
                matches!(e, Observation::OpponentFoul { .. })
            };
            if !ok {
                return None;
            }
        }
        for e in &events[..keep] {
            log.record(e.clone());
        }
    }
    let mut fouls = post.fouls;
    fouls[side_idx(side)] = fouls[side_idx(side)].checked_sub(fouls_this_turn)?;
    Some(Replayed {
        pos: post.pos.clone(),
        logs,
        fouls,
        plies: post.plies,
        injected_fouls: vec![],
        oracle: None,
    })
}

fn clone_replayed(rep: &Replayed) -> Replayed {
    Replayed {
        pos: rep.pos.clone(),
        logs: [clone_log(&rep.logs[0]), clone_log(&rep.logs[1])],
        fouls: rep.fouls,
        plies: rep.plies,
        injected_fouls: rep.injected_fouls.clone(),
        oracle: rep.oracle.clone(),
    }
}

/// 型（粗い束）のタグ。P0-1 の表と同じ分類
fn type_tag(first: CheckMoveKind, accepted: CheckMoveKind) -> &'static str {
    use tsuitate_bot::check_economy::CoarseKind::*;
    match (first.coarse(), accepted.coarse()) {
        (Drop, _) => "drop",
        (NonKingBoard, King) => "nonking_king",
        (NonKingBoard, _) => "nonking_nonking",
        (King, King) => "king_king",
        (King, _) => "king_nonking",
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut seeds: u64 = 3;
    let mut jobs: usize =
        std::thread::available_parallelism().map_or(1, |n| n.get().saturating_sub(2).max(1));
    let mut limit = usize::MAX;
    let mut kmax = 30.0f64;
    let mut shard: Option<(usize, usize)> = None;
    let mut opponent: Option<String> = None;
    let mut types: Vec<String> = vec!["nonking_king".into(), "nonking_nonking".into()];
    let mut out_csv: Option<String> = None;
    let mut specs: Vec<String> = vec![];
    let mut i = 0;
    let num = |v: Option<&String>, what: &str| -> usize {
        v.and_then(|s| s.parse().ok())
            .unwrap_or_else(|| die(&format!("{what} には数値が必要です")))
    };
    while i < args.len() {
        match args[i].as_str() {
            "--seeds" => {
                seeds = num(args.get(i + 1), "--seeds") as u64;
                i += 2;
            }
            "--jobs" => {
                jobs = num(args.get(i + 1), "--jobs").max(1);
                i += 2;
            }
            "--limit" => {
                limit = num(args.get(i + 1), "--limit");
                i += 2;
            }
            "--kmax" => {
                kmax = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| die("--kmax には数値が必要です"));
                i += 2;
            }
            "--shard" => {
                let s = args.get(i + 1).cloned().unwrap_or_else(|| die("--shard i/n"));
                let (a, b) = s.split_once('/').unwrap_or_else(|| die("--shard i/n"));
                let (a, b) = (
                    a.parse::<usize>().unwrap_or_else(|_| die("--shard i/n")),
                    b.parse::<usize>().unwrap_or_else(|_| die("--shard i/n")),
                );
                if b == 0 || a >= b {
                    die("--shard は i/n（0 ≤ i < n）");
                }
                shard = Some((a, b));
                i += 2;
            }
            "--opponent" => {
                opponent = Some(
                    args.get(i + 1)
                        .cloned()
                        .unwrap_or_else(|| die("--opponent には戦略名が必要です")),
                );
                i += 2;
            }
            "--types" => {
                types = args
                    .get(i + 1)
                    .unwrap_or_else(|| die("--types には型のタグが必要です"))
                    .split(',')
                    .map(str::to_string)
                    .collect();
                i += 2;
            }
            "--out" => {
                out_csv = Some(
                    args.get(i + 1)
                        .cloned()
                        .unwrap_or_else(|| die("--out にはパスが必要です")),
                );
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
    let cfg = tsuitate_bot::config::ambient();
    let params = EvalParams::default();

    // ---- 決定点の収集（粒子は回さない）-----------------------------------
    let mut points: Vec<Point> = vec![];
    let mut games = 0u32;
    let mut broken = 0u32;
    let mut mismatched = 0u32;
    let mut skipped = 0u32;
    let mut record_opponents: BTreeMap<String, u32> = BTreeMap::new();
    let mut type_counts: BTreeMap<String, u32> = BTreeMap::new();
    for path in &files {
        let name = path.to_string_lossy().to_string();
        let Some((bot, end)) = load_bot_and_end(&name) else {
            broken += 1;
            continue;
        };
        // ガントレットの記録は相手ごとに artifact が分かれる。混ざったまま
        // 監査すると「どの相手との対局分布か」が言えなくなる
        *record_opponents
            .entry(end.opponent.username.clone())
            .or_insert(0) += 1;
        if opponent.as_ref().is_some_and(|o| *o != end.opponent.username) {
            mismatched += 1;
            continue;
        }
        games += 1;
        let short = Path::new(&name)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or(name.clone());
        let mut found: Vec<Point> = vec![];
        let ok = for_each_decision_full(&end, |d| {
            if d.side != bot || d.fouls_this_turn == 0 || !d.pos.in_check(bot) {
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
                skipped += 1;
                return;
            };
            // この手番の反則列（ログ末尾の MyFoul。順番どおり）
            let events = post.logs[side_idx(d.side)].events();
            let fouls: Vec<String> = events[events.len() - d.fouls_this_turn as usize..]
                .iter()
                .filter_map(|e| match e {
                    Observation::MyFoul { usi, .. } => Some(usi.clone()),
                    _ => None,
                })
                .collect();
            let accepted = end.moves[d.decision_id as usize].usi.clone();
            // 手種は bot の意図（CheckSolver の仮説）で分ける。P0-1 と同じ規約
            let view = make_view(&entry.pos, d.side, &entry.fouls);
            let log = &entry.logs[side_idx(d.side)];
            let mut solver = CheckSolver::new(&view, &[], &[], log);
            let Some(first) = fouls.first().and_then(|u| parse_usi(u)) else {
                skipped += 1;
                return;
            };
            let Some(acc_mv) = parse_usi(&accepted) else {
                skipped += 1;
                return;
            };
            let first_kind = classify_move_kind(&first, &view, solver.as_mut());
            let acc_kind = classify_move_kind(&acc_mv, &view, solver.as_mut());
            let tag = type_tag(first_kind, acc_kind);
            *type_counts.entry(tag.to_string()).or_insert(0) += 1;
            if !types.iter().any(|t| t == tag) {
                return;
            }
            let (share, entropy) = hypothesis_stats(solver.as_ref(), d.pos, bot);
            found.push(Point {
                game: short.clone(),
                move_number: d.pos.move_number(),
                type_tag: tag,
                fouls,
                accepted,
                remaining: 10u32.saturating_sub(entry.fouls[side_idx(d.side)]),
                entry,
                post,
                true_hyp_share: share.unwrap_or(f64::NAN),
                hyp_entropy: entropy.unwrap_or(f64::NAN),
            });
        });
        if !ok {
            broken += 1;
            continue;
        }
        points.extend(found);
    }
    // 局単位で切る（同じ局の決定点がシャードをまたがないように game 名で割る）
    if let Some((si, sn)) = shard {
        let mut names: Vec<String> = points.iter().map(|p| p.game.clone()).collect();
        names.sort();
        names.dedup();
        let keep: HashSet<String> = names
            .into_iter()
            .enumerate()
            .filter(|(i, _)| i % sn == si)
            .map(|(_, n)| n)
            .collect();
        points.retain(|p| keep.contains(&p.game));
    }
    if points.len() > limit {
        points.truncate(limit);
    }
    if points.is_empty() {
        die("監査対象の決定点がありません（--types と記録を確認）");
    }
    println!(
        "記録 {} 件（壊れ {broken} / 相手不一致 {mismatched} / 復元できず {skipped}）/ 局 {games}",
        files.len()
    );
    println!(
        "記録の相手: {}",
        record_opponents
            .iter()
            .map(|(k, n)| format!("{k} {n}局"))
            .collect::<Vec<_>>()
            .join(" / ")
    );
    println!(
        "王手中で反則した手番の型: {}",
        type_counts
            .iter()
            .map(|(k, n)| format!("{k} {n}"))
            .collect::<Vec<_>>()
            .join(" / ")
    );
    println!(
        "監査対象 {} 決定点（型 {}）/ seeds {seeds} / jobs {jobs} / kmax {kmax}\n\
         思考予算 {}ms / config {}",
        points.len(),
        types.join(","),
        cfg.think_budget_ms,
        cfg.fingerprint(),
    );
    println!("source_fingerprint {}", env!("TSUITATE_SOURCE_FINGERPRINT"));

    // ---- 実行（決定点 × seed）。1 unit で「初回」と「反則注入後」を両方測る --
    let units: Vec<(usize, u64)> = (0..points.len())
        .flat_map(|p| (0..seeds).map(move |s| (p, s)))
        .collect();
    let effective_jobs = jobs.min(units.len()).max(1);
    let next = Arc::new(Mutex::new(0usize));
    let results: Arc<Mutex<Vec<UnitOut>>> = Arc::new(Mutex::new(vec![]));
    let points = Arc::new(points);
    let started = std::time::Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..effective_jobs {
            let next = Arc::clone(&next);
            let results = Arc::clone(&results);
            let points = Arc::clone(&points);
            let units = &units;
            let params = &params;
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
                    let (pi, seed) = units[ui];
                    let p = &points[pi];
                    let side = p.entry.pos.turn();
                    let king = p.entry.pos.king_square(side);
                    let Some((_, entry_rank)) =
                        ranking_one_with(&p.entry, seed, "estimator", &HashSet::new())
                    else {
                        eprintln!("{} {}手目: 初回ランキングなし", p.game, p.move_number);
                        continue;
                    };
                    let priced = priced_moves(&entry_rank, king);
                    let Some(top1) = priced.first() else { continue };
                    let king_best = priced.iter().find(|m| m.is_king);
                    let base = base_foul_cost(
                        params,
                        p.entry.fouls[side_idx(side)],
                        p.entry.fouls[side_idx(side.other())],
                    );
                    // `CandidateScore::foul_cost` は床（`last_foul_guard`）適用後の
                    // 値 = max(base, 床) なので、その最大値がそのまま床になる
                    // （床が効いていなければ base と一致する）。P1-α の
                    // `check_cost = max(k × base, 床)` をそのまま再現する
                    let floor = priced.iter().map(|m| m.foul_cost).fold(0.0f64, f64::max);
                    let k = k_star(&priced, base, floor, kmax);
                    // 「初回ランキングの次点」= 実戦の反則列を除いた最上位。
                    // 静的方策（現行 combine_score の暗黙の仮定）そのもの
                    let tried: HashSet<String> = p.fouls.iter().cloned().collect();
                    let next_static = priced
                        .iter()
                        .find(|m| !tried.contains(&m.usi))
                        .map(|m| m.usi.clone());
                    // 実再決定（`MyFoul` を食った推定器で全候補を評価し直す）
                    let post = clone_replayed(&p.post);
                    let after = ranking_one_with(&post, seed, "estimator", &tried);
                    let after_priced: Option<Vec<PricedMove>> =
                        after.as_ref().map(|(_, r)| priced_moves(r, king));
                    let (top1_after, p_king_after) = match &after_priced {
                        Some(r) => (
                            r.first().map(|m| m.usi.clone()),
                            r.iter().find(|m| m.is_king).map(|m| m.p_legal),
                        ),
                        None => (None, None),
                    };
                    let changed = match (&next_static, &top1_after) {
                        (Some(a), Some(b)) => Some(a != b),
                        _ => None,
                    };
                    results.lock().unwrap().push(UnitOut {
                        point: pi,
                        seed,
                        top1: top1.usi.clone(),
                        top1_is_king: top1.is_king,
                        king_best: king_best.map(|m| m.usi.clone()),
                        p_top1: top1.p_legal,
                        p_king: king_best.map_or(f64::NAN, |m| m.p_legal),
                        score_gap: king_best
                            .map_or(f64::NAN, |m| top1.det_score - m.det_score),
                        k_star: k,
                        matches_record: p.fouls.first().is_some_and(|f| *f == top1.usi),
                        next_static,
                        top1_after,
                        changed,
                        p_king_after,
                        candidates: entry_rank.len(),
                    });
                }
            });
        }
    });
    let mut results = Arc::try_unwrap(results).ok().unwrap().into_inner().unwrap();
    results.sort_by_key(|r| (r.point, r.seed));
    eprintln!(
        "{} unit / {:.1}秒",
        results.len(),
        started.elapsed().as_secs_f64()
    );

    if let Some(path) = &out_csv {
        write_csv(path, &points, &results);
    }
    report(&points, &results, seeds);
}

fn write_csv(path: &str, points: &[Point], results: &[UnitOut]) {
    let mut s = String::from(
        "game,move_number,type,remaining,fouls_actual,accepted,seed,candidates,top1,top1_is_king,\
king_best,p_top1,p_king,score_gap,k_star,next_static,top1_after,changed,p_king_after,\
matches_record,true_hyp_share,hyp_entropy\n",
    );
    for r in results {
        let p = &points[r.point];
        s.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{:.4},{:.4},{:.4},{},{},{},{},{},{},{:.4},{:.4}\n",
            p.game,
            p.move_number,
            p.type_tag,
            p.remaining,
            p.fouls.len(),
            p.accepted,
            r.seed,
            r.candidates,
            r.top1,
            u8::from(r.top1_is_king),
            r.king_best.clone().unwrap_or_default(),
            r.p_top1,
            r.p_king,
            r.score_gap,
            r.k_star.map_or(String::new(), |k| format!("{k:.2}")),
            r.next_static.clone().unwrap_or_default(),
            r.top1_after.clone().unwrap_or_default(),
            r.changed.map_or(String::new(), |c| u8::from(c).to_string()),
            r.p_king_after.map_or(String::new(), |p| format!("{p:.4}")),
            u8::from(r.matches_record),
            p.true_hyp_share,
            p.hyp_entropy,
        ));
    }
    if let Err(e) = std::fs::write(path, s) {
        eprintln!("CSV を書けません（{path}）: {e}");
    } else {
        eprintln!("CSV: {path}");
    }
}

/// 決定点ごとに seed 多数決を取る（seed は独立標本として数えない）
fn majority(vals: &[bool]) -> Option<bool> {
    if vals.is_empty() {
        return None;
    }
    let yes = vals.iter().filter(|v| **v).count();
    Some(yes * 2 > vals.len())
}

fn report(points: &[Point], results: &[UnitOut], seeds: u64) {
    println!("\n=== P0-4 w* 監査（issue #31）===");
    println!("決定点 {} / seed {seeds} / unit {}", points.len(), results.len());
    let mut by_type: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, p) in points.iter().enumerate() {
        by_type.entry(p.type_tag).or_default().push(i);
    }
    let mut all: Vec<usize> = (0..points.len()).collect();
    all.sort();
    for (tag, idxs) in by_type
        .iter()
        .map(|(t, v)| (*t, v.clone()))
        .chain(std::iter::once(("**合計**", all)))
    {
        let mut k3 = vec![];
        let mut changed = vec![];
        let mut already_king = vec![];
        let mut reproduced = vec![];
        let mut k_values: Vec<f64> = vec![];
        let mut unreverted = 0usize;
        let mut p_king_delta: Vec<f64> = vec![];
        for &pi in &idxs {
            let rows: Vec<&UnitOut> = results.iter().filter(|r| r.point == pi).collect();
            if rows.is_empty() {
                continue;
            }
            let ak: Vec<bool> = rows.iter().map(|r| r.top1_is_king).collect();
            let is_already = majority(&ak);
            if let Some(m) = is_already {
                already_king.push(m);
            }
            if let Some(m) = majority(&rows.iter().map(|r| r.matches_record).collect::<Vec<_>>()) {
                reproduced.push(m);
            }
            // **門の分母は「首位が玉以外」の決定点だけ**。首位が既に玉の手なら
            // 価格を上げるまでもないので「反転した」には数えない
            let rev: Vec<bool> = rows
                .iter()
                .filter(|r| !r.top1_is_king)
                .map(|r| r.k_star.is_some_and(|k| k <= 3.0))
                .collect();
            if let Some(m) = majority(&rev) {
                k3.push(m);
            }
            let ch: Vec<bool> = rows.iter().filter_map(|r| r.changed).collect();
            if let Some(m) = majority(&ch) {
                changed.push(m);
            }
            for r in rows.iter().filter(|r| !r.top1_is_king) {
                match r.k_star {
                    Some(k) => k_values.push(k),
                    None => unreverted += 1,
                }
            }
            for r in &rows {
                if let (Some(after), true) = (r.p_king_after, r.p_king.is_finite()) {
                    p_king_delta.push(after - r.p_king);
                }
            }
        }
        let pct = |v: &[bool]| -> String {
            if v.is_empty() {
                return "-".into();
            }
            let n = v.iter().filter(|b| **b).count();
            format!("{n}/{} ({:.1}%)", v.len(), 100.0 * n as f64 / v.len() as f64)
        };
        k_values.sort_by(f64::total_cmp);
        let median = if k_values.is_empty() {
            "-".to_string()
        } else {
            format!("{:.2}", k_values[k_values.len() / 2])
        };
        let mean_delta = if p_king_delta.is_empty() {
            "-".to_string()
        } else {
            format!(
                "{:+.3}",
                p_king_delta.iter().sum::<f64>() / p_king_delta.len() as f64
            )
        };
        println!("{tag}: 決定点 {}", idxs.len());
        println!("  初回ランキングの首位が実戦の最初の反則と一致: {}", pct(&reproduced));
        println!(
            "  現行の首位が既に玉の手（価格の出番なし・門の分母から外す）: {}",
            pct(&already_king)
        );
        println!(
            "  **k* ≤ 3 で玉の手が首位になる: {}**（首位が玉以外の unit のみ。k* の中央値 {median} / kmax までに反転しない unit {unreverted}）",
            pct(&k3)
        );
        println!("  **反則注入後の実再決定が静的な次点と違う: {}**", pct(&changed));
        println!("  最良の玉の手の p_legal の変化（反則注入の前後）: {mean_delta}");
    }
    println!(
        "\n中止条件（issue #31）: **k* ≤ 3 で反転する決定点が 20% 未満 かつ\n\
         反則注入後に首位が変わる決定点も 20% 未満**なら α / β の枝は中止"
    );
}
