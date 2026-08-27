//! **盤上版の被詰めろのオフライン較正と argmax シミュレーション**（issue #28 P0-3）。
//!
//! 記録（`arena-records` / `records/`）の真実を再生し、詰み負けの局の
//! 「最後に受けられた決定点」と、**同じ手数帯の非危険な対照決定点**で、
//! 現行 estimator の候補ランキングと粒子集合を作って次を測る:
//!
//! - 真実の「この手を指すと相手に一手詰めが生じる」ラベルに対する、
//!   粒子上の詰み質量 q の較正（Brier・PR・信頼度別）。**厳密粒子 / taint 別**
//! - **対照決定点での偽陽性率**（ここを測らないと誤爆が見えない）
//! - 受け手（真実の安全手）の現行順位・top-k 包含率・argmax が受けへ変わる最小 w
//! - 現行 `mate_risk_w`（打ち詰みのみ）の発火量 = 盤上詰みが視野外であることの確認
//!
//! **runtime には何も入らない**（`mate_economy` の診断関数を呼ぶだけ）。
//! 再スコアは P1-B の形（`gain` 側から引いて `combine_score` を通す。
//! issue #24 の教訓⑦: 最終 score へ直接足すのは等価でない）。
//!
//! usage:
//!   TSUITATE_THINK_BUDGET_MS=2000 cargo run --release --bin mate_probe -- \
//!     [--seeds 3] [--jobs N] [--controls 1] [--limit N] [--out data/mate_probe.csv] \
//!     [--emit <dir> <接頭辞>] <records/ または *.jsonl ...>
//!
//! **思考予算はプロセス env から読む**（`TSUITATE_THINK_BUDGET_MS`）。粒子の
//! 構築（`build_estimator`）と候補評価（`ranking_one`）で同じ scale を使うため、
//! ここで勝手に上書きはしない。
//!
//! 判定の注意（CLAUDE.md の教訓）: 壁時計予算なので**同じ設定でも粒子数は揺れる**。
//! 3シードは切り分け用で、合否に使う量は本数を増やしてから見ること。

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tsuitate_bot::kifu::kif_body;
use tsuitate_bot::mate::mate_moves_in_1_fast;
use tsuitate_bot::mate_economy::{
    MateKind, MateRow, analyze_game, control_points, min_w_to_flip, pick,
};
use tsuitate_bot::observation::Observation;
use tsuitate_bot::protocol::GameEndPayload;
use tsuitate_bot::scenario_core::{
    Replayed, build_estimator, clone_log, ranking_one_with, side_idx, weighted_unique_particles,
};
use tsuitate_bot::shogi::{Position, ShogiMove};
use tsuitate_bot::truth_replay::{for_each_decision_full, load_bot_and_end};

fn usage() -> &'static str {
    "usage: cargo run --release --bin mate_probe -- [--seeds 3] [--jobs N] [--controls 1] \
     [--limit N] [--out PATH.csv] [--emit <dir> <接頭辞>] <records/*.jsonl...>"
}

fn die(msg: &str) -> ! {
    eprintln!("{msg}");
    eprintln!("{}", usage());
    std::process::exit(1);
}

/// 危険 / 対照の決定点1つ
struct Point {
    game: String,
    /// 決定点の添字（= 0 始まりの ply）
    ply: usize,
    danger: bool,
    /// その決定点で真実上の安全手（危険な決定点だけ非空）
    safe: HashSet<String>,
    /// 実戦でその決定点から指した手
    played: String,
    /// 危険な決定点の詰め手の分類
    kind: Option<MateKind>,
    /// 実際に詰まされた手（次の相手番で終局したとき）
    executed: Option<String>,
    rep: Replayed,
    foul_tried: HashSet<String>,
}

/// 1 unit（決定点 × seed）の結果
struct UnitOut {
    point: usize,
    seed: u64,
    chosen: String,
    rows: Vec<MateRow>,
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

/// 決定点の状態（真実の局面・両者の観測ログ・累積反則）を復元する。
///
/// `truth_replay` が渡す状態は**その手番の反則試行を消化した後**なので、
/// 実戦でその決定点に立ったときと同じ（`foul_tried` も復元して渡す）。
fn decision_states(
    end: &GameEndPayload,
    want: &HashSet<usize>,
) -> HashMap<usize, (Replayed, HashSet<String>)> {
    let mut out = HashMap::new();
    for_each_decision_full(end, |d| {
        let idx = d.decision_id as usize;
        if !want.contains(&idx) {
            return;
        }
        // この手番で既に試した反則（ログ末尾の MyFoul。手番は変わらない）
        let mut foul_tried: HashSet<String> = HashSet::new();
        for e in d.logs[side_idx(d.side)].events().iter().rev() {
            match e {
                Observation::MyFoul { usi, .. } if foul_tried.len() < d.fouls_this_turn as usize => {
                    foul_tried.insert(usi.clone());
                }
                _ => break,
            }
        }
        let rep = Replayed {
            pos: d.pos.clone(),
            logs: [clone_log(&d.logs[0]), clone_log(&d.logs[1])],
            fouls: *d.fouls,
            plies: d.plies,
            injected_fouls: vec![],
            oracle: None,
        };
        out.insert(idx, (rep, foul_tried));
    });
    out
}

fn usi_set(moves: &[ShogiMove]) -> HashSet<String> {
    moves.iter().map(ShogiMove::to_usi).collect()
}

/// `--emit`: 決定点を `scenarios/` 形式の kif で書き出す（`bin/rank_probe` へ流す用）
fn emit_kif(dir: &str, prefix: &str, n: usize, p: &Point, end: &GameEndPayload) {
    let moves: Vec<String> = end.moves.iter().map(|m| m.usi.clone()).collect();
    let fouls: Vec<(u32, String)> = end
        .foul_attempts
        .iter()
        .map(|f| (f.move_number, f.usi.clone()))
        .collect();
    let ending = if end.reason == "foul_limit" {
        Some("反則負け")
    } else if end.reason == "checkmate" {
        Some("詰み")
    } else {
        None
    };
    let target = p.safe.iter().min().cloned().unwrap_or_default();
    let desc = format!(
        "{} の{}手目（詰み経済 P0-3）: 次の相手番の詰め手は{}・真実上の安全手 {}本・実戦は {}{}",
        p.game,
        p.ply + 1,
        p.kind.map_or("なし", MateKind::label),
        p.safe.len(),
        p.played,
        p.executed
            .as_ref()
            .map(|m| format!("・次の相手番で {m} と詰まされた"))
            .unwrap_or_default(),
    );
    let mut out = format!(
        "*scenario ply={} target={target} desc={}\n",
        p.ply,
        desc.replace('\n', " ")
    );
    out.push_str("棋戦：arena\n手合割：平手\n先手：先手\n後手：後手\n手数----指手---------消費時間--\n");
    match kif_body(&moves, &fouls, ending) {
        Ok(body) => out.push_str(&body),
        Err(e) => {
            eprintln!("{}: KIF 生成に失敗（飛ばします）: {e}", p.game);
            return;
        }
    }
    let path = Path::new(dir).join(format!("{prefix}-{n:02}.kif"));
    if let Err(e) = std::fs::write(&path, out) {
        eprintln!("{}: 書けません: {e}", path.display());
    } else {
        eprintln!("書き出し: {}", path.display());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut seeds: u64 = 3;
    let mut jobs: usize = std::thread::available_parallelism().map_or(1, |n| n.get().saturating_sub(2).max(1));
    let mut controls: usize = 1;
    let mut limit: usize = usize::MAX;
    let mut out_csv: Option<String> = None;
    let mut emit: Option<(String, String)> = None;
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
            "--controls" => {
                controls = num(args.get(i + 1), "--controls");
                i += 2;
            }
            "--limit" => {
                limit = num(args.get(i + 1), "--limit");
                i += 2;
            }
            "--out" => {
                out_csv = Some(args.get(i + 1).cloned().unwrap_or_else(|| die("--out にはパスが必要です")));
                i += 2;
            }
            "--emit" => {
                let dir = args.get(i + 1).cloned().unwrap_or_else(|| die("--emit <dir> <接頭辞>"));
                let prefix = args.get(i + 2).cloned().unwrap_or_else(|| die("--emit <dir> <接頭辞>"));
                emit = Some((dir, prefix));
                i += 3;
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
    let budget_ms = cfg.think_budget_ms;
    // 粒子構築の scale は評価側と同じ規約（既定 2000ms が scale 2000/900）
    let scale = budget_ms as f64 / 900.0;

    // ---- 決定点の収集 -----------------------------------------------------
    let mut points: Vec<Point> = vec![];
    let mut ends: HashMap<String, GameEndPayload> = HashMap::new();
    let mut games = 0u32;
    let mut mated_games = 0u32;
    let mut broken = 0u32;
    let mut danger_found = 0usize;
    for path in &files {
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
        games += 1;
        if g.bot_mated {
            mated_games += 1;
        }
        // P0-3 の対象は「詰まされた局の最後に受けられた決定点」
        if !g.bot_mated {
            continue;
        }
        let Some(turn) = g.last_defense_point() else {
            continue;
        };
        let Some(decision_idx) = turn.decision_idx else {
            continue;
        };
        danger_found += 1;
        let game = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| name.clone());
        let mut want: HashSet<usize> = HashSet::from([decision_idx]);
        let ctrl = control_points(&g, decision_idx, controls, &want);
        want.extend(ctrl.iter().copied());
        let states = decision_states(&end, &want);
        let played_at = |idx: usize| {
            end.moves
                .get(idx)
                .map(|m| m.usi.clone())
                .unwrap_or_else(|| "-".into())
        };
        if let Some((rep, foul_tried)) = states.get(&decision_idx) {
            points.push(Point {
                game: game.clone(),
                ply: decision_idx,
                danger: true,
                safe: usi_set(&turn.safe),
                played: played_at(decision_idx),
                kind: Some(turn.kind),
                executed: turn.executed.as_ref().map(ShogiMove::to_usi),
                rep: clone_replayed(rep),
                foul_tried: foul_tried.clone(),
            });
        }
        for c in ctrl {
            if let Some((rep, foul_tried)) = states.get(&c) {
                points.push(Point {
                    game: game.clone(),
                    ply: c,
                    danger: false,
                    safe: HashSet::new(),
                    played: played_at(c),
                    kind: None,
                    executed: None,
                    rep: clone_replayed(rep),
                    foul_tried: foul_tried.clone(),
                });
            }
        }
        ends.insert(game, end);
    }

    // --limit は危険な決定点の数で切る（対照はその局のぶんだけ残す）。
    // **黙って切らない**: 落とした件数を必ず出す（CLAUDE.md「no silent caps」）
    let mut dropped_games = 0usize;
    if danger_found > limit {
        let keep: HashSet<String> = points
            .iter()
            .filter(|p| p.danger)
            .take(limit)
            .map(|p| p.game.clone())
            .collect();
        dropped_games = danger_found - keep.len();
        points.retain(|p| keep.contains(&p.game));
    }
    if points.is_empty() {
        die("対象の決定点がありません（詰み負けの局が無い / 受けが無い）");
    }
    let n_danger = points.iter().filter(|p| p.danger).count();
    let n_control = points.len() - n_danger;

    if let Some((dir, prefix)) = &emit {
        std::fs::create_dir_all(dir).unwrap_or_else(|e| die(&format!("{dir}: {e}")));
        let mut n = 0usize;
        for p in points.iter().filter(|p| p.danger) {
            n += 1;
            if let Some(end) = ends.get(&p.game) {
                emit_kif(dir, prefix, n, p, end);
            }
        }
    }

    println!(
        "記録 {} 件（壊れ {broken}）/ 局 {games}（bot の詰み負け {mated_games}）\n\
         決定点: 危険 {n_danger} / 対照 {n_control}（--limit で落とした局 {dropped_games}）\n\
         思考予算 {budget_ms}ms / seeds {seeds} / jobs {jobs} / config {}",
        files.len(),
        cfg.fingerprint(),
    );
    println!("source_fingerprint {}", env!("TSUITATE_SOURCE_FINGERPRINT"));

    // ---- 実行（決定点 × seed）-------------------------------------------
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
                    let Some((chosen, ranking)) =
                        ranking_one_with(&p.rep, seed, "estimator", &p.foul_tried)
                    else {
                        eprintln!("{} ply{}: ランキングなし（定跡・候補ゼロ）", p.game, p.ply + 1);
                        continue;
                    };
                    // 粒子は診断の規約（bin/scenario diag / GUI と同じ構築）で作る。
                    // 同じ seed・同じ scale なので候補評価が見た集合と同じになる
                    let est = build_estimator(&p.rep, seed, scale, |_, _| {});
                    let particles = weighted_unique_particles(&est);
                    let rows = tsuitate_bot::mate_economy::build_rows(&p.rep.pos, &ranking, &particles);
                    results.lock().unwrap().push(UnitOut {
                        point: pi,
                        seed,
                        chosen,
                        rows,
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

fn write_csv(path: &str, points: &[Point], results: &[UnitOut]) {
    let mut s = String::from(
        "game,ply,danger,seed,usi,rank,det_score,gain,p_legal,foul_cost,adjust_det,depth2,\
         mate_risk,q_strict,q_all,strict_share,truth_legal,truth_allows_mate,truth_kind,\
         threat_kind,safe,played,chosen\n",
    );
    for u in results {
        let p = &points[u.point];
        for r in &u.rows {
            s.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                p.game,
                p.ply + 1,
                u32::from(p.danger),
                u.seed,
                r.usi,
                r.rank,
                r.det_score,
                r.gain,
                r.p_legal,
                r.foul_cost,
                r.adjust_det,
                u32::from(r.depth2),
                r.mate_risk,
                r.mass.strict,
                r.mass.all,
                r.mass.strict_share,
                u32::from(r.truth_legal),
                r.truth_allows_mate.map_or(String::new(), |b| u32::from(b).to_string()),
                r.truth_kind.map_or("", MateKind::tag),
                p.kind.map_or("", MateKind::tag),
                u32::from(p.safe.contains(&r.usi)),
                u32::from(r.usi == p.played),
                u32::from(r.usi == u.chosen),
            ));
        }
    }
    match std::fs::write(path, s) {
        Ok(()) => println!("CSV: {path}"),
        Err(e) => eprintln!("{path}: 書けません: {e}"),
    }
}

/// PR / 較正 / 偽陽性の集計
fn report(points: &[Point], results: &[UnitOut], seeds: u64) {
    let w_grid: Vec<f64> = {
        let mut v = vec![];
        let mut w = 0.0;
        while w <= 60.0 + 1e-9 {
            v.push(w);
            w += 0.5;
        }
        v
    };
    let report_ws = [2.0, 4.0, 8.0, 16.0, 30.0];

    for &use_taint in &[false, true] {
        let label = if use_taint { "taint込み" } else { "厳密粒子" };
        println!("\n=== 較正（{label}の詰み質量 q）===");
        // 行の集計（真実で合法な行だけ。非合法な手は「その世界で詰まれない」ので
        // ラベルが定義できない）
        let mut brier = [(0.0f64, 0usize); 2]; // [危険, 対照]
        let mut pos_n = [0usize; 2];
        let mut bins: Vec<[(f64, usize, usize); 2]> = vec![[(0.0, 0, 0); 2]; 10];
        let mut kind_pos: HashMap<&str, usize> = HashMap::new();
        let mut kind_q: HashMap<&str, f64> = HashMap::new();
        let taus = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9];
        // しきい値ごとの (予測陽性, そのうち真陽性)
        let mut tally: Vec<(usize, usize)> = vec![(0, 0); taus.len()];
        let mut fp_control: Vec<(usize, usize)> = vec![(0, 0); taus.len()];
        let mut total_pos = 0usize;
        for u in results {
            let d = usize::from(!points[u.point].danger);
            for r in &u.rows {
                let Some(label) = r.truth_allows_mate else {
                    continue;
                };
                let q = r.q(use_taint);
                let y = f64::from(u8::from(label));
                brier[d].0 += (q - y) * (q - y);
                brier[d].1 += 1;
                pos_n[d] += usize::from(label);
                let b = ((q * 10.0).floor() as usize).min(9);
                bins[b][d].0 += q;
                bins[b][d].1 += 1;
                bins[b][d].2 += usize::from(label);
                if label {
                    total_pos += 1;
                    if let Some(k) = r.truth_kind {
                        *kind_pos.entry(k.tag()).or_insert(0) += 1;
                        *kind_q.entry(k.tag()).or_insert(0.0) += q;
                    }
                }
                for (ti, &t) in taus.iter().enumerate() {
                    if q >= t {
                        tally[ti].0 += 1;
                        if label {
                            tally[ti].1 += 1;
                        }
                        if d == 1 {
                            fp_control[ti].0 += usize::from(!label);
                        }
                    }
                    if d == 1 {
                        fp_control[ti].1 += usize::from(!label);
                    }
                }
            }
        }
        let show = |t: (f64, usize)| if t.1 == 0 { f64::NAN } else { t.0 / t.1 as f64 };
        println!(
            "  行数: 危険 {} (陽性 {}) / 対照 {} (陽性 {})",
            brier[0].1, pos_n[0], brier[1].1, pos_n[1]
        );
        println!(
            "  Brier: 危険 {:.4} / 対照 {:.4}（基底率予測はそれぞれ {:.4} / {:.4}）",
            show(brier[0]),
            show(brier[1]),
            base_brier(pos_n[0], brier[0].1),
            base_brier(pos_n[1], brier[1].1),
        );
        println!("  しきい値ごとの PR と対照の偽陽性率:");
        for (ti, &t) in taus.iter().enumerate() {
            let (predicted, tp) = tally[ti];
            let prec = if predicted == 0 { f64::NAN } else { tp as f64 / predicted as f64 };
            let rec = if total_pos == 0 { f64::NAN } else { tp as f64 / total_pos as f64 };
            let (fp, neg) = fp_control[ti];
            let fpr = if neg == 0 { f64::NAN } else { fp as f64 / neg as f64 };
            println!(
                "    q≥{t:.2}: 予測陽性 {predicted} / 適合率 {prec:.3} / 再現率 {rec:.3} / 対照の偽陽性率 {fpr:.4}"
            );
        }
        println!("  信頼度別（q のビン: 平均q → 実際の陽性率 / 件数）:");
        for (b, row) in bins.iter().enumerate() {
            let n = row[0].1 + row[1].1;
            if n == 0 {
                continue;
            }
            let sum_q = row[0].0 + row[1].0;
            let pos = row[0].2 + row[1].2;
            println!(
                "    [{:.1},{:.1}): 平均q {:.3} → 実際 {:.3}（{n}件 = 危険 {} / 対照 {}）",
                b as f64 / 10.0,
                (b + 1) as f64 / 10.0,
                sum_q / n as f64,
                pos as f64 / n as f64,
                row[0].1,
                row[1].1,
            );
        }
        if !kind_pos.is_empty() {
            print!("  陽性行の分類別（平均q）:");
            let mut ks: Vec<_> = kind_pos.iter().collect();
            ks.sort();
            for (k, n) in ks {
                print!(" {k} {n}件 q={:.3}", kind_q[k] / *n as f64);
            }
            println!();
        }

        // ---- argmax シミュレーション（危険な決定点だけ）-------------------
        println!("=== argmax シミュレーション（{label}・危険な決定点）===");
        let mut n_units = 0usize;
        let mut blind_units = 0usize;
        let mut argmax_safe0 = 0usize;
        let mut best_safe_rank: Vec<usize> = vec![];
        let mut in_top = [0usize; 2]; // top8, top17
        let mut flips: Vec<Option<f64>> = vec![];
        let mut mate_risk_fired = 0usize;
        let mut at_w = vec![0usize; report_ws.len()];
        let mut covered_safe = 0usize;
        let mut total_safe = 0usize;
        let mut no_safe_units = 0usize;
        for u in results {
            let p = &points[u.point];
            if !p.danger {
                continue;
            }
            n_units += 1;
            let strict_share = u.rows.first().map_or(0.0, |r| r.mass.strict_share);
            if strict_share == 0.0 {
                blind_units += 1;
            }
            if u.rows.iter().any(|r| r.mate_risk > 0.0) {
                mate_risk_fired += 1;
            }
            let safe_rows: Vec<&MateRow> = u
                .rows
                .iter()
                .filter(|r| r.truth_allows_mate == Some(false))
                .collect();
            // 漏斗の2段目: 真実の安全手のうち候補生成に載った割合
            covered_safe += u
                .rows
                .iter()
                .filter(|r| p.safe.contains(&r.usi))
                .count();
            total_safe += p.safe.len();
            if safe_rows.is_empty() {
                no_safe_units += 1;
                continue;
            }
            let best = safe_rows.iter().map(|r| r.rank).min().unwrap_or(usize::MAX);
            best_safe_rank.push(best);
            in_top[0] += usize::from(best < 8);
            in_top[1] += usize::from(best < 17);
            let safe_usis: HashSet<String> = safe_rows.iter().map(|r| r.usi.clone()).collect();
            if pick(&u.rows, 0.0, use_taint).is_some_and(|r| safe_usis.contains(&r.usi)) {
                argmax_safe0 += 1;
            }
            for (wi, &w) in report_ws.iter().enumerate() {
                if pick(&u.rows, w, use_taint).is_some_and(|r| safe_usis.contains(&r.usi)) {
                    at_w[wi] += 1;
                }
            }
            flips.push(min_w_to_flip(&u.rows, &safe_usis, use_taint, &w_grid));
        }
        let pct = |a: usize, b: usize| if b == 0 { f64::NAN } else { a as f64 * 100.0 / b as f64 };
        println!(
            "  決定点×seed {n_units}（厳密粒子ゼロ {blind_units} = {:.1}%・現行 mate_risk が発火 {mate_risk_fired}）",
            pct(blind_units, n_units)
        );
        if !best_safe_rank.is_empty() {
            let mut sorted = best_safe_rank.clone();
            sorted.sort_unstable();
            println!(
                "  受け手の現行順位（乱数を除く）: 中央値 {} / 平均 {:.1} / top8 {:.1}% / top17 {:.1}%",
                sorted[sorted.len() / 2],
                sorted.iter().sum::<usize>() as f64 / sorted.len() as f64,
                pct(in_top[0], sorted.len()),
                pct(in_top[1], sorted.len()),
            );
            println!(
                "  argmax が受け: w=0 {:.1}% / {}",
                pct(argmax_safe0, best_safe_rank.len()),
                report_ws
                    .iter()
                    .zip(&at_w)
                    .map(|(w, n)| format!("w={w:.0} {:.1}%", pct(*n, best_safe_rank.len())))
                    .collect::<Vec<_>>()
                    .join(" / "),
            );
            // 最小 w は「w=0 で既に受けを選んでいる決定点」を除いて見る
            // （0 を混ぜると中央値が 0 に張り付いて必要な強度が読めない）
            let mut needed: Vec<f64> = flips
                .iter()
                .filter_map(|f| *f)
                .filter(|&w| w > 0.0)
                .collect();
            needed.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let need_n = flips.iter().filter(|f| f.map(|w| w > 0.0).unwrap_or(true)).count();
            println!(
                "  w=0 では受けでない {need_n} 決定点のうち、w を上げると受けになる {}（最小 w の中央値 {}・最大 {}）",
                needed.len(),
                if needed.is_empty() {
                    "-".to_string()
                } else {
                    format!("{:.1}", needed[needed.len() / 2])
                },
                needed.last().map_or("-".to_string(), |w| format!("{w:.1}")),
            );
            println!(
                "  受けの候補生成カバー率: {covered_safe}/{total_safe}（真実の安全手が候補に載った割合）\
                 ・候補内に安全手が1本も無い決定点 {no_safe_units}"
            );
        }
    }

    // seed 間のばらつき（壁時計予算なので同じ設定でも粒子数が揺れる）
    if seeds > 1 {
        let mut per_seed: HashMap<u64, (f64, f64, usize)> = HashMap::new();
        for u in results {
            if !points[u.point].danger {
                continue;
            }
            let e = per_seed.entry(u.seed).or_insert((0.0, 0.0, 0));
            let max_q = |taint: bool| {
                u.rows
                    .iter()
                    .filter(|r| r.truth_allows_mate == Some(true))
                    .map(|r| r.q(taint))
                    .fold(0.0, f64::max)
            };
            e.0 += max_q(false);
            e.1 += max_q(true);
            e.2 += 1;
        }
        let mut ks: Vec<_> = per_seed.keys().copied().collect();
        ks.sort();
        print!("\nseed 間のばらつき（危険な決定点の「最大の陽性 q」の平均・厳密/taint）:");
        for k in ks {
            let (st, ta, n) = per_seed[&k];
            print!(" seed{k} {:.3}/{:.3}", st / n as f64, ta / n as f64);
        }
        println!("\n  ※ 壁時計予算なので同じ設定でも粒子数は揺れる。少数シードの差は判定に使わないこと");
    }
}

fn base_brier(pos: usize, n: usize) -> f64 {
    if n == 0 {
        return f64::NAN;
    }
    let p = pos as f64 / n as f64;
    p * (1.0 - p)
}
