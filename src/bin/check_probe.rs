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
//!     [--seeds 3] [--jobs N] [--limit N] [--shard i/n] [--kmax 30] [--allow-incomplete] \
//!     [--opponent estimator_v14] [--types nonking_king,nonking_nonking] \
//!     [--out data/check_probe.csv] <records...>

use std::collections::{BTreeMap, BTreeSet, HashSet};
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
use tsuitate_bot::truth_replay::{for_each_decision_full, parse_bot_and_end};

/// CSV / 集約の契約バージョン。**古い schema は集計から弾く**
/// （撤回済みの数字が横断表へ戻らないように。issue #28 の契約と同じ）
const ROW_SCHEMA: u32 = 1;

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

/// 集計の単位（CSV の1行 = 決定点 × seed）。**その場の実行でもシャードの
/// CSV からでも同じ集計を通す**ための共通形
#[derive(Clone)]
struct Row {
    game: String,
    move_number: u32,
    type_tag: String,
    seed: u64,
    top1_is_king: bool,
    k_star: Option<f64>,
    changed: Option<bool>,
    matches_record: bool,
    p_king: f64,
    p_king_after: Option<f64>,
}

impl Row {
    /// 決定点のキー（seed は独立標本として数えないので、ここで束ねる）
    fn point_key(&self) -> (String, u32) {
        (self.game.clone(), self.move_number)
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
    // `report` は既存の CSV を集約するだけ（粒子を回さない）
    if args.first().is_some_and(|a| a == "report") {
        run_report(&args[1..]);
        return;
    }
    let mut seeds: u64 = 3;
    let mut jobs: usize =
        std::thread::available_parallelism().map_or(1, |n| n.get().saturating_sub(2).max(1));
    let mut limit = usize::MAX;
    let mut kmax = 30.0f64;
    let mut allow_incomplete = false;
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
            "--allow-incomplete" => {
                allow_incomplete = true;
                i += 1;
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
    // **`--seeds 0` で「対象なし」を成功終了にできてはいけない**（issue #28 が
    // `mate_continue` で塞いだ穴と同じ: 判定を空の集計で偽造できる）
    if seeds == 0 {
        die("--seeds は 1 以上にしてください（0 だとランキングを1本も取らずに集計が空になる）");
    }
    let files = collect_records(&specs);
    if files.is_empty() {
        die("記録ファイルが見つかりません");
    }
    let cfg = tsuitate_bot::config::ambient();
    let params = EvalParams::default();
    use sha2::Digest as _;

    // ---- 決定点の収集（粒子は回さない）-----------------------------------
    let mut points: Vec<Point> = vec![];
    let mut games = 0u32;
    let mut broken = 0u32;
    let mut mismatched = 0u32;
    let mut skipped = 0u32;
    let mut record_opponents: BTreeMap<String, u32> = BTreeMap::new();
    let mut type_counts: BTreeMap<String, u32> = BTreeMap::new();
    // **解析に渡したのと同じ bytes** から記録集合の指紋を作る（ディスクを
    // 読み直すと TOCTOU になる。issue #28 PR #30 レビュー3巡目の教訓）
    let mut digest = sha2::Sha256::new();
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
        // ガントレットの記録は相手ごとに artifact が分かれる。混ざったまま
        // 監査すると「どの相手との対局分布か」が言えなくなる
        *record_opponents
            .entry(end.opponent.username.clone())
            .or_insert(0) += 1;
        if opponent.as_ref().is_some_and(|o| *o != end.opponent.username) {
            mismatched += 1;
            continue;
        }
        let short = Path::new(&name)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or(name.clone());
        let mut found: Vec<Point> = vec![];
        // 途中で壊れた棋譜は `found` ごと捨てるので、型の分布もこの局の分は
        // 捨てる（`--types` を選ぶ表が監査した決定点と食い違わないように）
        let mut local_types: BTreeMap<String, u32> = BTreeMap::new();
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
            *local_types.entry(tag.to_string()).or_insert(0) += 1;
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
        games += 1;
        for (k, v) in local_types {
            *type_counts.entry(k).or_insert(0) += v;
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
    let dropped = Arc::new(Mutex::new(0usize));
    let results: Arc<Mutex<Vec<UnitOut>>> = Arc::new(Mutex::new(vec![]));
    let points = Arc::new(points);
    let started = std::time::Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..effective_jobs {
            let next = Arc::clone(&next);
            let dropped = Arc::clone(&dropped);
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
                        *dropped.lock().unwrap() += 1;
                        continue;
                    };
                    let priced = priced_moves(&entry_rank, king);
                    let Some(top1) = priced.first() else {
                        *dropped.lock().unwrap() += 1;
                        continue;
                    };
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
    let dropped = *dropped.lock().unwrap();
    if dropped > 0 {
        eprintln!("ランキングが取れずに落とした unit: {dropped}");
    }
    eprintln!(
        "{} unit / {:.1}秒",
        results.len(),
        started.elapsed().as_secs_f64()
    );

    let records_fingerprint: String =
        digest.finalize().iter().map(|b| format!("{b:02x}")).collect();
    let (shard_i, shard_n) = shard.unwrap_or((0, 1));
    let meta = serde_json::json!({
        "schema": ROW_SCHEMA,
        // **相手を meta に持つ**（`records` 指紋はディレクトリ全体のハッシュなので、
        // 同じ記録集合から `--opponent` だけ変えた2本は指紋が一致してしまう。
        // 相手ごとに分けるのが判定の前提なので、混ぜたら集約で止める）
        "opponent": opponent.clone().unwrap_or_default(),
        "shard": shard_i,
        "shards": shard_n,
        "seeds": seeds,
        "types": types.join(","),
        "kmax": kmax,
        "budget_ms": cfg.think_budget_ms,
        "config": cfg.fingerprint(),
        "source_fingerprint": env!("TSUITATE_SOURCE_FINGERPRINT"),
        "records": records_fingerprint,
        "points": points.len(),
    });
    if let Some(path) = &out_csv {
        write_csv(path, &points, &results, &meta);
    }
    let rows: Vec<Row> = results
        .iter()
        .map(|r| Row {
            game: points[r.point].game.clone(),
            move_number: points[r.point].move_number,
            type_tag: points[r.point].type_tag.to_string(),
            seed: r.seed,
            top1_is_king: r.top1_is_king,
            k_star: r.k_star,
            changed: r.changed,
            matches_record: r.matches_record,
            p_king: r.p_king,
            p_king_after: r.p_king_after,
        })
        .collect();
    check_completeness(&rows, seeds, allow_incomplete);
    check_points_present(&rows, points.len(), allow_incomplete);
    if shard_n > 1 {
        println!(
            "\n**シャード {shard_i}/{shard_n} の部分集計**（判定は `check_probe report` で全シャードを集めてから）"
        );
    }
    report(&rows);
}

fn write_csv(path: &str, points: &[Point], results: &[UnitOut], meta: &serde_json::Value) {
    // 1行目は **meta**（集約が「同じ実験の独立サンプルか」を検査する）。
    // シャードを跨いで seeds / types / 予算 / config / コード版 / 記録集合が
    // 一致していなければ集計は失敗させる（issue #28 の契約と同じ）
    let mut s = format!("#meta {meta}\n");
    s.push_str(
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

/// **決定点ごとに seed が全部揃っているか**を検査する。
///
/// 揃っていない決定点があると、多数決の分母が決定点ごとに変わり
/// （2/3 seed の点は同数で「反転しない」側へ倒れる）、門の割合が黙って動く。
/// `--allow-incomplete` を付けたときだけ警告へ落とす（issue #28 の契約と同じ）。
fn check_completeness(rows: &[Row], seeds: u64, allow_incomplete: bool) {
    let mut per_point: BTreeMap<(String, u32), BTreeSet<u64>> = BTreeMap::new();
    for r in rows {
        per_point.entry(r.point_key()).or_default().insert(r.seed);
    }
    let bad: Vec<String> = per_point
        .iter()
        .filter(|(_, s)| s.len() as u64 != seeds)
        .map(|((g, mn), s)| format!("{g} {mn}手目（seed {}/{seeds}）", s.len()))
        .collect();
    if bad.is_empty() {
        return;
    }
    let msg = format!(
        "seed が揃っていない決定点が {} 件あります（多数決の分母が決定点ごとに変わる）:\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
    if allow_incomplete {
        eprintln!("警告: {msg}");
    } else {
        die(&format!("{msg}\n（承知のうえで集計するなら --allow-incomplete）"));
    }
}

/// **決定点そのものが消えていないか**を検査する。
///
/// 全 seed でランキングが取れなかった決定点は行が1つも残らないので、
/// `check_completeness`（行がある決定点しか見ない）では捕まらない。
/// 門の分母が黙って縮むので、meta の決定点数と突き合わせる。
fn check_points_present(rows: &[Row], expected: usize, allow_incomplete: bool) {
    let seen: BTreeSet<(String, u32)> = rows.iter().map(|r| r.point_key()).collect();
    if seen.len() == expected {
        return;
    }
    let msg = format!(
        "決定点が {} 件しか残っていません（対象 {expected} 件）: 門の分母が縮む",
        seen.len()
    );
    if allow_incomplete {
        eprintln!("警告: {msg}");
    } else {
        die(&format!("{msg}\n（承知のうえで集計するなら --allow-incomplete）"));
    }
}

/// 型ごとの集計値（表示と検査で同じものを使う）
#[derive(Default)]
struct Stats {
    points: usize,
    /// 決定点ごとの多数決（seed は独立標本として数えない）
    reproduced: Vec<bool>,
    already_king: Vec<bool>,
    /// **首位が玉以外**の unit だけで取る「k* ≤ 3 で反転したか」
    k3: Vec<bool>,
    changed: Vec<bool>,
    k_values: Vec<f64>,
    unreverted: usize,
    p_king_delta: Vec<f64>,
}

/// 決定点ごとに畳んだ集計。型ごと＋合計を返す
fn summarize(rows: &[Row]) -> Vec<(String, Stats)> {
    let mut points: BTreeMap<(String, u32), Vec<&Row>> = BTreeMap::new();
    for r in rows {
        points.entry(r.point_key()).or_default().push(r);
    }
    let mut by_type: BTreeMap<String, Vec<(String, u32)>> = BTreeMap::new();
    for (key, rs) in &points {
        by_type.entry(rs[0].type_tag.clone()).or_default().push(key.clone());
    }
    let all: Vec<(String, u32)> = points.keys().cloned().collect();
    let mut out = vec![];
    for (tag, keys) in by_type
        .into_iter()
        .chain(std::iter::once(("**合計**".to_string(), all)))
    {
        let mut st = Stats { points: keys.len(), ..Stats::default() };
        for key in &keys {
            let rs = &points[key];
            // **決定点の分類は多数決で1回だけ決める**（seed は独立標本として
            // 数えない）。首位が既に玉の手の決定点は「価格の出番なし」に入れ、
            // 門の分母からは丸ごと外す: 多数決で既に玉の手なのに、少数側の
            // 1 unit だけで「反転した」を分母1で数えると門の割合が水増しされる
            let is_already = majority(&rs.iter().map(|r| r.top1_is_king).collect::<Vec<_>>());
            if let Some(m) = is_already {
                st.already_king.push(m);
            }
            if let Some(m) = majority(&rs.iter().map(|r| r.matches_record).collect::<Vec<_>>()) {
                st.reproduced.push(m);
            }
            if is_already == Some(false) {
                // 門の分母は「首位が玉以外」の決定点だけ。その中でも
                // 首位が玉の手だった seed は問い自体が立たないので除く
                let rev: Vec<bool> = rs
                    .iter()
                    .filter(|r| !r.top1_is_king)
                    .map(|r| r.k_star.is_some_and(|k| k <= 3.0))
                    .collect();
                if let Some(m) = majority(&rev) {
                    st.k3.push(m);
                }
                for r in rs.iter().filter(|r| !r.top1_is_king) {
                    match r.k_star {
                        Some(k) => st.k_values.push(k),
                        None => st.unreverted += 1,
                    }
                }
            }
            if let Some(m) = majority(&rs.iter().filter_map(|r| r.changed).collect::<Vec<_>>()) {
                st.changed.push(m);
            }
            for r in rs {
                if let (Some(after), true) = (r.p_king_after, r.p_king.is_finite()) {
                    st.p_king_delta.push(after - r.p_king);
                }
            }
        }
        st.k_values.sort_by(f64::total_cmp);
        out.push((tag, st));
    }
    out
}

/// 集計の表示（その場の実行でも、シャードの CSV を集めた `report` でも同じ経路）
fn report(rows: &[Row]) {
    let seeds: BTreeSet<u64> = rows.iter().map(|r| r.seed).collect();
    let points: BTreeSet<(String, u32)> = rows.iter().map(|r| r.point_key()).collect();
    println!("\n=== P0-4 w* 監査（issue #31）===");
    println!(
        "決定点 {} / seed {} / unit {}",
        points.len(),
        seeds.len(),
        rows.len()
    );
    let pct = |v: &[bool]| -> String {
        if v.is_empty() {
            return "-".into();
        }
        let n = v.iter().filter(|b| **b).count();
        format!("{n}/{} ({:.1}%)", v.len(), 100.0 * n as f64 / v.len() as f64)
    };
    for (tag, st) in summarize(rows) {
        let median = if st.k_values.is_empty() {
            "-".to_string()
        } else {
            format!("{:.2}", st.k_values[st.k_values.len() / 2])
        };
        let mean_delta = if st.p_king_delta.is_empty() {
            "-".to_string()
        } else {
            format!(
                "{:+.3}",
                st.p_king_delta.iter().sum::<f64>() / st.p_king_delta.len() as f64
            )
        };
        println!("{tag}: 決定点 {}", st.points);
        println!(
            "  初回ランキングの首位が実戦の最初の反則と一致: {}",
            pct(&st.reproduced)
        );
        println!(
            "  現行の首位が既に玉の手（価格の出番なし・門の分母から外す）: {}",
            pct(&st.already_king)
        );
        println!(
            "  **k* ≤ 3 で玉の手が首位になる: {}**（首位が玉以外の unit のみ。k* の中央値 {median} / kmax までに反転しない unit {}）",
            pct(&st.k3),
            st.unreverted,
        );
        println!("  **反則注入後の実再決定が静的な次点と違う: {}**", pct(&st.changed));
        println!("  最良の玉の手の p_legal の変化（反則注入の前後）: {mean_delta}");
    }
    println!(
        "\n中止条件（issue #31）: **k* ≤ 3 で反転する決定点が 20% 未満 かつ\n\
         反則注入後に首位が変わる決定点も 20% 未満**なら α / β の枝は中止"
    );
}

/// シャードの CSV を集約する（`check_probe report [--shards N] <csv...>`）。
///
/// **欠けたシャードを黙って少ない分母で報告しない**（issue #19 / #28 の教訓）。
/// meta が食い違う CSV（別の seed 数・型・予算・config・コード版・記録集合）も
/// 混ぜない。
fn run_report(args: &[String]) {
    let mut want_shards: Option<usize> = None;
    let mut allow_incomplete = false;
    let mut paths: Vec<String> = vec![];
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--shards" => {
                want_shards = Some(
                    args.get(i + 1)
                        .and_then(|s| s.parse().ok())
                        .unwrap_or_else(|| die("--shards には数値が必要です")),
                );
                i += 2;
            }
            "--allow-incomplete" => {
                allow_incomplete = true;
                i += 1;
            }
            s if s.starts_with("--") => die(&format!("未知のオプション: {s}")),
            s => {
                paths.push(s.to_string());
                i += 1;
            }
        }
    }
    if paths.is_empty() {
        die("CSV を指定してください");
    }
    let mut rows: Vec<Row> = vec![];
    let mut metas: Vec<serde_json::Value> = vec![];
    for path in &paths {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| die(&format!("{path} を読めません: {e}")));
        let mut lines = text.lines();
        let meta: serde_json::Value = lines
            .next()
            .and_then(|l| l.strip_prefix("#meta "))
            .and_then(|j| serde_json::from_str(j).ok())
            .unwrap_or_else(|| die(&format!("{path}: meta 行がありません（schema 0 の CSV）")));
        if meta["schema"].as_u64() != Some(u64::from(ROW_SCHEMA)) {
            die(&format!(
                "{path}: schema {} は集計できません（現行 {ROW_SCHEMA}）",
                meta["schema"]
            ));
        }
        let header = lines.next().unwrap_or_default();
        let cols: Vec<&str> = header.split(',').collect();
        let idx = |name: &str| -> usize {
            cols.iter()
                .position(|c| *c == name)
                .unwrap_or_else(|| die(&format!("{path}: 列 {name} がありません")))
        };
        let (i_game, i_mn, i_type, i_seed, i_king, i_k, i_ch, i_match, i_pk, i_pka) = (
            idx("game"),
            idx("move_number"),
            idx("type"),
            idx("seed"),
            idx("top1_is_king"),
            idx("k_star"),
            idx("changed"),
            idx("matches_record"),
            idx("p_king"),
            idx("p_king_after"),
        );
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split(',').collect();
            if f.len() != cols.len() {
                die(&format!("{path}: 列数が合わない行があります"));
            }
            rows.push(Row {
                game: f[i_game].to_string(),
                move_number: f[i_mn].parse().unwrap_or(0),
                type_tag: f[i_type].to_string(),
                seed: f[i_seed].parse().unwrap_or(0),
                top1_is_king: f[i_king] == "1",
                k_star: f[i_k].parse().ok(),
                changed: match f[i_ch] {
                    "1" => Some(true),
                    "0" => Some(false),
                    _ => None,
                },
                matches_record: f[i_match] == "1",
                p_king: f[i_pk].parse().unwrap_or(f64::NAN),
                p_king_after: f[i_pka].parse().ok(),
            });
        }
        metas.push(meta);
    }
    // 実験キーの一致（shard 番号だけが違うこと）
    let key = |m: &serde_json::Value| -> String {
        let mut m = m.clone();
        if let Some(o) = m.as_object_mut() {
            o.remove("shard");
            o.remove("points");
        }
        m.to_string()
    };
    let first = key(&metas[0]);
    for (path, m) in paths.iter().zip(&metas) {
        if key(m) != first {
            die(&format!(
                "{path}: meta が他と食い違います（別の実験を混ぜている）\n  {}\n  {first}",
                key(m)
            ));
        }
    }
    let shards: BTreeSet<u64> = metas.iter().filter_map(|m| m["shard"].as_u64()).collect();
    let total = metas[0]["shards"].as_u64().unwrap_or(1) as usize;
    // `--shards` は**上書きではなく突き合わせ**（上書きを許すと「1本しか無いのに
    // --shards 1 で通す」で欠落検査を素通りできる）
    if let Some(want) = want_shards {
        if want != total {
            die(&format!(
                "--shards {want} は CSV の meta（shards {total}）と食い違います"
            ));
        }
    }
    if shards.len() != total || shards.iter().max().map_or(0, |m| *m as usize + 1) != total {
        die(&format!(
            "シャードが欠けています（{:?} / 全 {total}）: 決定点の分母が狂うので中止",
            shards
        ));
    }
    // 同じ (決定点, seed) が2度出てくる = シャードの重複
    let mut seen = HashSet::new();
    for r in &rows {
        if !seen.insert((r.game.clone(), r.move_number, r.seed)) {
            die("同じ (決定点, seed) が重複しています（シャードの割り当てがおかしい）");
        }
    }
    let seeds = metas[0]["seeds"].as_u64().unwrap_or_else(|| die("meta に seeds がありません"));
    if seeds == 0 {
        die("meta の seeds が 0 です（空の集計で判定を偽造できてしまう）");
    }
    check_completeness(&rows, seeds, allow_incomplete);
    let expected: usize = metas
        .iter()
        .map(|m| m["points"].as_u64().unwrap_or_else(|| die("meta に points がありません")) as usize)
        .sum();
    check_points_present(&rows, expected, allow_incomplete);
    println!("CSV {} 本 / シャード {total} / 行 {}", paths.len(), rows.len());
    println!("meta {}", first);
    report(&rows);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(game: &str, mn: u32, seed: u64, king: bool, k: Option<f64>) -> Row {
        Row {
            game: game.into(),
            move_number: mn,
            type_tag: "nonking_king".into(),
            seed,
            top1_is_king: king,
            k_star: k,
            changed: Some(false),
            matches_record: true,
            p_king: 0.8,
            p_king_after: Some(0.9),
        }
    }

    #[test]
    fn 首位が既に玉の手の決定点は門の分母から外れる() {
        let rows = vec![
            // 決定点A: 3 seed とも首位が既に玉の手 → k3 には入らない
            row("a", 10, 0, true, Some(1.0)),
            row("a", 10, 1, true, Some(1.0)),
            row("a", 10, 2, true, Some(1.0)),
            // 決定点B: 首位はプローブで、k* ≤ 3 で反転する
            row("b", 20, 0, false, Some(2.0)),
            row("b", 20, 1, false, Some(2.5)),
            row("b", 20, 2, false, None),
        ];
        let st = summarize(&rows);
        let total = &st.iter().find(|(t, _)| t == "**合計**").unwrap().1;
        assert_eq!(total.points, 2);
        // 分母は決定点B だけ（A は「価格の出番なし」に数える）
        assert_eq!(total.k3, vec![true]);
        assert_eq!(total.already_king, vec![true, false]);
        assert_eq!(total.unreverted, 1);
    }

    #[test]
    fn 多数決で既に玉の手の決定点は少数側のunitでも門に入らない() {
        // 3 seed 中 2 つで首位が既に玉の手 → その決定点は「価格の出番なし」。
        // 残り 1 seed が k*≤3 で反転しても、分母1で「反転した」に数えない
        let rows = vec![
            row("a", 10, 0, true, Some(1.0)),
            row("a", 10, 1, true, Some(1.0)),
            row("a", 10, 2, false, Some(2.0)),
        ];
        let st = summarize(&rows);
        let total = &st.iter().find(|(t, _)| t == "**合計**").unwrap().1;
        assert_eq!(total.already_king, vec![true]);
        assert!(total.k3.is_empty(), "門の分母に入らない: {:?}", total.k3);
        assert!(total.k_values.is_empty(), "k* の分布にも混ぜない");
    }

    #[test]
    fn seedは独立標本として数えず決定点ごとに多数決を取る() {
        // 3 seed 中 1 つだけ反転 → その決定点は「反転しない」
        let rows = vec![
            row("a", 10, 0, false, Some(2.0)),
            row("a", 10, 1, false, Some(9.0)),
            row("a", 10, 2, false, None),
        ];
        let st = summarize(&rows);
        let total = &st.iter().find(|(t, _)| t == "**合計**").unwrap().1;
        assert_eq!(total.k3, vec![false]);
        assert_eq!(majority(&[true, true, false]), Some(true));
        assert_eq!(majority(&[true, false]), Some(false), "同数は「反転しない」側");
        assert_eq!(majority(&[]), None);
    }
}
