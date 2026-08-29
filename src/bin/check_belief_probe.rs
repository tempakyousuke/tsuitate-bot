//! **王手駒仮説の希釈 P0-1: runtime の伝達の分解**
//! （issue #36。runtime には何も入らない）。
//!
//! #31 P0-3 が出した「真の王手駒の重みシェア 0.035」は
//! `CheckSolver::new(&view, &[], &[], &log)` = **粒子投票なしのソルバー単体の事前**で、
//! runtime ではそこから
//!
//! ```text
//! ソルバー q → 粒子投票後の q → prior_legal との積 → blend_p_legal →
//! cap/min 系の補正 → score → rank
//! ```
//!
//! と6段を通って初めて選択に届く。**仮説シェアは選択への伝達量ではない**ので、
//! ここは各段を記述的に並べるだけで**中止の門を置かない**（因果はオラクル
//! arm = P0-2 でしか言えない）。
//!
//! 出力の主眼は「どこで信念が消えているか」:
//!
//! - **真の王手駒を玉以外で取る手**（`true_capture`）と
//!   **誤仮説マスへの捕獲の最大値**（`false_capture`）を対にして各段を並べる
//! - 層は `particles_vote_check` の有無 × 厳密/taint × 反則あり/なし ×
//!   型（#31 の `TurnType` 相当）× 王手駒の種別（打ち／盤上／捕獲つき）。
//!   **両王手は別層**（単王手が主 estimand）
//! - **coverage failure**（真の王手駒を玉以外で取る候補が無い手番）は欠測に
//!   せず別に数える（候補に無ければ完璧な信念でも救えない = この issue の上限の外）
//! - 母集団は **bot の全王手中手番**（反則0も、反則だけ積んで受理手なしで
//!   終局した終端手番も含む）。復元できない手番は attrition として本数を出す
//!
//! usage:
//!   TSUITATE_THINK_BUDGET_MS=2000 cargo run --release --bin check_belief_probe -- \
//!     [--seeds 3] [--jobs N] [--limit N] [--shard i/n] [--opponent estimator_v14] \
//!     [--out data/check_belief.jsonl] <records...>
//!   cargo run --release --bin check_belief_probe -- report [--shards N] <out-*.jsonl...>
//!
//! **思考予算は記録を取ったアリーナと揃える**（違うと粒子数が変わり、
//! 「その決定点で実際に見えていたランキング」でなくなる）。

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tsuitate_bot::check::CheckSolver;
use tsuitate_bot::check_belief::{Attrition, Point, decision_points, focus};
use tsuitate_bot::check_economy::{
    classify_move_kind, hypothesis_share, sq as sq_str, true_checkers,
};
use tsuitate_bot::check_policy::{EntrySetup, PStage, entry_setup, price_at};
use tsuitate_bot::scenario_core::side_idx;
use tsuitate_bot::shogi::parse_usi;
use tsuitate_bot::strategy::{self, EvalParams};
use tsuitate_bot::truth_replay::parse_bot_and_end;

/// JSONL の契約バージョン。**古い schema は集計から弾く**（issue #28 / #31 の契約）
const ROW_SCHEMA: u32 = 1;

fn die(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(2);
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

/// 王手駒の種別（打ち／盤上／捕獲つき）。観測（捕獲マス）と記録（相手の USI）から
fn checker_kind(p: &Point) -> &'static str {
    if p.opp_captured_at.is_some() {
        return "with_capture";
    }
    match p.opp_last_usi.as_deref() {
        Some(u) if u.contains('*') => "drop",
        Some(_) => "board",
        None => "unknown",
    }
}

struct Collected {
    game: String,
    point: Point,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "report") {
        run_report(&args[1..]);
        return;
    }
    let mut seeds: u64 = 3;
    let mut jobs: usize =
        std::thread::available_parallelism().map_or(1, |n| n.get().saturating_sub(2).max(1));
    let mut limit = usize::MAX;
    let mut shard = (0usize, 1usize);
    let mut opponent: Option<String> = None;
    let mut allow_incomplete = false;
    let mut out_path: Option<String> = None;
    let mut specs: Vec<String> = vec![];
    let need = |v: Option<&String>, what: &str| -> String {
        v.cloned()
            .unwrap_or_else(|| die(&format!("{what} には値が必要です")))
    };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seeds" => {
                seeds = need(args.get(i + 1), "--seeds")
                    .parse()
                    .unwrap_or_else(|_| die("--seeds は整数"));
                i += 2;
            }
            "--jobs" => {
                jobs = need(args.get(i + 1), "--jobs")
                    .parse::<usize>()
                    .unwrap_or_else(|_| die("--jobs は整数"))
                    .max(1);
                i += 2;
            }
            "--limit" => {
                limit = need(args.get(i + 1), "--limit")
                    .parse()
                    .unwrap_or_else(|_| die("--limit は整数"));
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
            "--opponent" => {
                opponent = Some(need(args.get(i + 1), "--opponent"));
                i += 2;
            }
            "--allow-incomplete" => {
                allow_incomplete = true;
                i += 1;
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
    // `--seeds 0` で「対象なし」を成功終了にできてはいけない（issue #28 の契約）
    if seeds == 0 {
        die("--seeds は 1 以上にしてください（0 だと1本も走らせずに集計が空になる）");
    }
    let files = collect_records(&specs);
    if files.is_empty() {
        die("記録ファイルが見つかりません");
    }

    let cfg = tsuitate_bot::config::ambient();
    let params = EvalParams::default();
    let eval_particles = strategy::eval_particles_for_budget(cfg.think_budget_ms);

    // ---- 母集団の取り出し（粒子は回さない）--------------------------------
    let mut points: Vec<Collected> = vec![];
    let mut at = Attrition::default();
    let mut games = 0u32;
    let mut broken = 0u32;
    let mut mismatched = 0u32;
    let mut record_opponents: BTreeMap<String, u32> = BTreeMap::new();
    use sha2::Digest as _;
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
        let Some(found) = decision_points(&end, bot, &mut at) else {
            broken += 1;
            continue;
        };
        games += 1;
        for point in found {
            points.push(Collected { game: short.clone(), point });
        }
    }
    points.sort_by(|a, b| {
        (&a.game, a.point.move_number).cmp(&(&b.game, b.point.move_number))
    });
    // シャードは**局単位**（同じ局の決定点がシャードをまたがない）
    if shard.1 > 1 {
        let mut names: Vec<String> = points.iter().map(|p| p.game.clone()).collect();
        names.sort();
        names.dedup();
        let keep: BTreeSet<String> = names
            .into_iter()
            .enumerate()
            .filter(|(i, _)| i % shard.1 == shard.0)
            .map(|(_, n)| n)
            .collect();
        points.retain(|p| keep.contains(&p.game));
    }
    if points.len() > limit {
        points.truncate(limit);
    }
    if points.is_empty() {
        die("対象の決定点がありません（記録と --opponent を確認）");
    }

    println!(
        "記録 {} 件（壊れ {broken} / 相手不一致 {mismatched}）/ 局 {games}",
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
        "王手中の bot の手番 {} （うち終端 {} / 復元できず {}）→ 対象 {}",
        at.turns,
        at.terminal,
        at.unreplayable,
        points.len()
    );
    println!(
        "seeds {seeds} / jobs {jobs} / shard {}/{} / 思考予算 {}ms / eval_particles {eval_particles}",
        shard.0, shard.1, cfg.think_budget_ms
    );
    println!("config {}", cfg.fingerprint());
    println!("source_fingerprint {}", env!("TSUITATE_SOURCE_FINGERPRINT"));

    // ---- 実行（決定点 × seed）---------------------------------------------
    let units: Vec<(usize, u64)> = (0..points.len())
        .flat_map(|p| (0..seeds).map(move |s| (p, s)))
        .collect();
    let effective_jobs = jobs.min(units.len()).max(1);
    let next = Arc::new(Mutex::new(0usize));
    let lines: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(vec![]));
    let dropped: Arc<Mutex<BTreeSet<(String, u32)>>> = Arc::new(Mutex::new(BTreeSet::new()));
    let points = Arc::new(points);
    let started = std::time::Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..effective_jobs {
            let next = Arc::clone(&next);
            let lines = Arc::clone(&lines);
            let dropped = Arc::clone(&dropped);
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
                    match run_unit(&points[pi], seed, params, eval_particles) {
                        Some(row) => lines.lock().unwrap().push(row),
                        None => {
                            dropped
                                .lock()
                                .unwrap()
                                .insert((points[pi].game.clone(), points[pi].point.move_number));
                        }
                    }
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
        )
    });
    let dropped = Arc::try_unwrap(dropped).ok().unwrap().into_inner().unwrap();
    eprintln!(
        "{} 行 / 落とした決定点 {} / {:.1}分",
        lines.len(),
        dropped.len(),
        started.elapsed().as_secs_f64() / 60.0
    );

    let records_fingerprint: String =
        digest.finalize().iter().map(|b| format!("{b:02x}")).collect();
    let experiment = serde_json::json!({
        "opponent": opponent.clone().unwrap_or_default(),
        "budget_ms": cfg.think_budget_ms,
        "eval_particles": eval_particles,
        "seeds": seeds,
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
        "attrition": {
            "check_turns": at.turns,
            "terminal": at.terminal,
            "unreplayable": at.unreplayable,
        },
        "points_detail": points
            .iter()
            .map(|p| serde_json::json!({
                "game": p.game,
                "move_number": p.point.move_number,
                "estimand": p.point.estimand(),
                "terminal": p.point.terminal,
            }))
            .collect::<Vec<_>>(),
        "dropped": dropped
            .iter()
            .map(|(g, mn)| serde_json::json!({ "game": g, "move_number": mn }))
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
    if shard.1 > 1 {
        println!(
            "\n**シャード {}/{} の部分集計**（判定は `check_belief_probe report` で全シャードを集めてから）",
            shard.0, shard.1
        );
    }
    report(&[meta], &lines, allow_incomplete);
}

fn stage_json(s: &PStage, usi: &str, gain: f64, score: f64, rank: usize) -> serde_json::Value {
    serde_json::json!({
        "usi": usi,
        "base_prior": s.base_prior,
        "solver_p": s.solver_p,
        "prior": s.prior,
        "particle_legal": if s.particle_legal.is_finite() { serde_json::json!(s.particle_legal) } else { serde_json::Value::Null },
        "blended": s.blended,
        "cap": s.cap,
        "final_p": s.final_p,
        "gain": gain,
        "score": score,
        "rank": rank,
    })
}

/// 1 unit（決定点 × seed）を回して JSONL の行を返す
fn run_unit(
    c: &Collected,
    seed: u64,
    params: &EvalParams,
    eval_particles: usize,
) -> Option<serde_json::Value> {
    let p = &c.point;
    let side = p.entry.pos.turn();
    let king = p.entry.pos.king_square(side);
    let setup = entry_setup(&p.entry, &p.truth, seed, params, eval_particles)?;
    let EntrySetup { view, log, moves, p0, updater, snapshot, .. } = setup;
    // **ソルバー単体**（粒子投票なし）= #31 P0-3 が測った 0.035 と同じ経路
    let solo = CheckSolver::new(&view, &[], &[], &log);
    let checkers = true_checkers(&p.truth, p.bot);
    let (share_solo, ent_solo) = solo
        .as_ref()
        .map(|s| hypothesis_share(&s.hypotheses_debug(), &checkers))
        .unwrap_or((None, None));
    // **runtime と同じ**（粒子投票・捕獲ブースト・反則減衰を通した）仮説
    let stages = updater.stages_after(&moves, &[]);
    let (share_rt, ent_rt) = hypothesis_share(&stages.hypotheses, &checkers);
    // 健全性: 分解の `final_p` は初回ランキングの p_legal を再現するはず
    let repro_err = p0
        .iter()
        .zip(&stages.per_move)
        .map(|(a, b)| (a - b.final_p).abs())
        .fold(0.0f64, f64::max);

    // 投票者（王手を説明する粒子）の有無と重み
    let pool: &[(tsuitate_bot::shogi::Position, f64)] = if snapshot.strict.is_empty() {
        &snapshot.taint
    } else {
        &snapshot.strict
    };
    let total_w: f64 = pool.iter().map(|(_, w)| w).sum();
    let mut voter_w = 0.0;
    let mut voter_true_w = 0.0;
    for (pos, w) in pool {
        if !pos.in_check(p.bot) {
            continue;
        }
        voter_w += w;
        let mine = true_checkers(pos, p.bot);
        if mine.len() == checkers.len()
            && mine.iter().all(|m| checkers.iter().any(|c| c == m))
        {
            voter_true_w += w;
        }
    }

    // 順位は**乱数を除いたスコア**で付け直す（issue #24 の教訓②。同点は USI の辞書順）
    let price = price_at(params, p.entry.fouls[side_idx(side)], p.entry.fouls[side_idx(side.other())]);
    let mut scored: Vec<(usize, f64)> = moves
        .iter()
        .enumerate()
        .map(|(i, m)| (i, m.score(p0[i], price.current)))
        .collect();
    scored.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| moves[a.0].usi.cmp(&moves[b.0].usi))
    });
    let mut rank_of = vec![usize::MAX; moves.len()];
    for (r, (i, _)) in scored.iter().enumerate() {
        rank_of[*i] = r + 1;
    }
    // 誤仮説マスは**その決定の仮説集合**から取る（真実で敵駒がいるマス全部へ
    // 広げると、信念とは無関係な捕獲を代表にしてしまう）
    let hyp_squares: Vec<_> = stages.hypotheses.iter().map(|(s, _, _)| *s).collect();
    let f = focus(&moves, &p.truth, p.bot, &hyp_squares);
    let focus_json = |idx: Option<usize>| -> serde_json::Value {
        match idx {
            Some(i) => stage_json(
                &stages.per_move[i],
                &moves[i].usi,
                moves[i].gain,
                moves[i].score(p0[i], price.current),
                rank_of[i],
            ),
            None => serde_json::Value::Null,
        }
    };

    // 型（#31 と同じ規約: bot の意図 = `captures_checker` で分ける）
    let mut solver2 = CheckSolver::new(&view, &[], &[], &log);
    let first_kind = p
        .record_fouls
        .first()
        .and_then(|u| parse_usi(u))
        .map(|m| classify_move_kind(&m, &view, solver2.as_mut()).tag());
    let acc_kind = p
        .record_accepted
        .as_deref()
        .and_then(parse_usi)
        .map(|m| classify_move_kind(&m, &view, solver2.as_mut()).tag());
    // 首位が玉の手か（希釈の帰結として玉逃げへ倒れているか）
    let top = scored.first().map(|(i, _)| *i);

    Some(serde_json::json!({
        "schema": ROW_SCHEMA,
        "game": c.game,
        "move_number": p.move_number,
        "estimand": p.estimand(),
        "terminal": p.terminal,
        "seed": seed,
        "checkers": checkers.len(),
        "double_check": checkers.len() > 1,
        "checker_kind": checker_kind(p),
        "checker_squares": checkers.iter().map(|(s, _)| sq_str(*s)).collect::<Vec<_>>(),
        "first_foul_kind": first_kind,
        "accepted_kind": acc_kind,
        "record_fouls": p.record_fouls.len(),
        "can_vote": stages.can_vote,
        "use_taint": stages.use_taint,
        "strict_n": stages.strict_n,
        "hyp_n": stages.hypotheses.len(),
        "share_solver": share_solo,
        "entropy_solver": ent_solo,
        "share_runtime": share_rt,
        "entropy_runtime": ent_rt,
        "voter_share": if total_w > 0.0 { voter_w / total_w } else { 0.0 },
        "voter_true_share": if total_w > 0.0 { voter_true_w / total_w } else { 0.0 },
        "candidates": moves.len(),
        "coverage": f.true_capture.is_some(),
        "true_capture": focus_json(f.true_capture),
        "false_capture": focus_json(f.false_capture),
        "top_is_king": top.map(|i| moves[i].is_king),
        "top_usi": top.map(|i| moves[i].usi.clone()),
        "repro_err": repro_err,
        "king": king.map(sq_str),
    }))
}

// ---- 集計 ------------------------------------------------------------------

#[derive(Default)]
struct Acc {
    n: usize,
    share_solver: Vec<f64>,
    share_runtime: Vec<f64>,
    entropy: Vec<f64>,
    coverage: usize,
    true_rank: Vec<f64>,
    true_p: Vec<f64>,
    false_p: Vec<f64>,
    true_solver_p: Vec<f64>,
    false_solver_p: Vec<f64>,
    top_king: usize,
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

impl Acc {
    fn push(&mut self, r: &serde_json::Value) {
        self.n += 1;
        if let Some(v) = r["share_solver"].as_f64() {
            self.share_solver.push(v);
        }
        if let Some(v) = r["share_runtime"].as_f64() {
            self.share_runtime.push(v);
        }
        if let Some(v) = r["entropy_runtime"].as_f64() {
            self.entropy.push(v);
        }
        if r["coverage"].as_bool().unwrap_or(false) {
            self.coverage += 1;
        }
        if r["top_is_king"].as_bool().unwrap_or(false) {
            self.top_king += 1;
        }
        if let Some(t) = r["true_capture"].as_object() {
            if let Some(v) = t["rank"].as_f64() {
                self.true_rank.push(v);
            }
            if let Some(v) = t["final_p"].as_f64() {
                self.true_p.push(v);
            }
            if let Some(v) = t["solver_p"].as_f64() {
                self.true_solver_p.push(v);
            }
        }
        if let Some(t) = r["false_capture"].as_object() {
            if let Some(v) = t["final_p"].as_f64() {
                self.false_p.push(v);
            }
            if let Some(v) = t["solver_p"].as_f64() {
                self.false_solver_p.push(v);
            }
        }
    }

    fn row(&self, label: &str) -> String {
        format!(
            "  {label:<22} {:>5} {:>8.3} {:>8.3} {:>8.3} {:>8.1} {:>9.3} {:>9.3} {:>8.1} {:>8.1}",
            self.n,
            mean(&self.share_solver),
            mean(&self.share_runtime),
            mean(&self.entropy),
            100.0 * self.coverage as f64 / self.n.max(1) as f64,
            mean(&self.true_solver_p),
            mean(&self.true_p),
            mean(&self.true_rank),
            100.0 * self.top_king as f64 / self.n.max(1) as f64,
        )
    }
}

const HEAD: &str = "  層                         n  q単体  q投票後  エントロピ  被覆%   真捕獲p_s   真捕獲p    順位   玉首位%";

fn report(metas: &[serde_json::Value], rows: &[serde_json::Value], allow_incomplete: bool) {
    if rows.is_empty() {
        println!("行がありません");
        return;
    }
    let complete = check_inputs(metas, rows, allow_incomplete);
    let mut by: BTreeMap<String, Acc> = BTreeMap::new();
    let mut all = Acc::default();
    let mut repro = 0.0f64;
    let mut single = Acc::default();
    for r in rows {
        repro = repro.max(r["repro_err"].as_f64().unwrap_or(0.0));
        let dbl = r["double_check"].as_bool().unwrap_or(false);
        all.push(r);
        if dbl {
            by.entry("両王手（別層）".into()).or_default().push(r);
            continue;
        }
        single.push(r);
        let est = r["estimand"].as_str().unwrap_or("?");
        by.entry(format!("estimand={est}")).or_default().push(r);
        let vote = if r["can_vote"].as_bool().unwrap_or(false) { "投票あり" } else { "投票なし" };
        by.entry(format!("{vote}")).or_default().push(r);
        let pool = if r["use_taint"].as_bool().unwrap_or(false) { "taint" } else { "厳密" };
        by.entry(format!("プール={pool}")).or_default().push(r);
        by.entry(format!("王手駒={}", r["checker_kind"].as_str().unwrap_or("?")))
            .or_default()
            .push(r);
        if let Some(k) = r["first_foul_kind"].as_str() {
            by.entry(format!("初反則={k}")).or_default().push(r);
        }
        if r["terminal"].as_bool().unwrap_or(false) {
            by.entry("終端手番".into()).or_default().push(r);
        }
    }
    println!("\n## P0-1 伝達の分解（記述のみ。中止の門は置かない）\n");
    println!("健全性: 分解の final_p と初回ランキングの p_legal の最大差 {repro:.4}");
    if repro > 1e-9 {
        println!("  **ずれている**: `evaluate` の経路（cap / min 系）を取りこぼしている可能性");
    }
    println!("{HEAD}");
    println!("{}", all.row("全体"));
    println!("{}", single.row("単王手（主）"));
    for (k, a) in &by {
        println!("{}", a.row(k));
    }
    println!(
        "\ncoverage failure（真の王手駒を玉以外で取る候補が無い）: {} / {} 行",
        all.n - all.coverage,
        all.n
    );
    println!(
        "誤仮説マスへの捕獲: ソルバー p {:.3} / 最終 p {:.3}（真捕獲は {:.3} / {:.3}）",
        mean(&all.false_solver_p),
        mean(&all.false_p),
        mean(&all.true_solver_p),
        mean(&all.true_p),
    );
    if !complete {
        println!("\n**INCOMPLETE**: 入力が揃っていません（判定に使わないこと）");
    }
}

/// 入力の完全性（シャードの欠落・重複・実験キーの一致）
fn check_inputs(
    metas: &[serde_json::Value],
    rows: &[serde_json::Value],
    allow_incomplete: bool,
) -> bool {
    let mut problems: Vec<String> = vec![];
    let mut experiments: BTreeSet<String> = BTreeSet::new();
    let mut shards: BTreeSet<u64> = BTreeSet::new();
    let mut totals: BTreeSet<u64> = BTreeSet::new();
    let mut want: BTreeSet<(String, u32)> = BTreeSet::new();
    let mut dropped: BTreeSet<(String, u32)> = BTreeSet::new();
    let mut seeds = 0u64;
    for m in metas {
        if m["schema"].as_u64() != Some(ROW_SCHEMA as u64) {
            problems.push(format!("schema {} は集計対象外", m["schema"]));
            continue;
        }
        experiments.insert(m["experiment"].to_string());
        shards.insert(m["shard"].as_u64().unwrap_or(0));
        totals.insert(m["experiment"]["shard_total"].as_u64().unwrap_or(1));
        seeds = seeds.max(m["experiment"]["seeds"].as_u64().unwrap_or(0));
        for d in m["points_detail"].as_array().into_iter().flatten() {
            want.insert((
                d["game"].as_str().unwrap_or("?").to_string(),
                d["move_number"].as_u64().unwrap_or(0) as u32,
            ));
        }
        for d in m["dropped"].as_array().into_iter().flatten() {
            dropped.insert((
                d["game"].as_str().unwrap_or("?").to_string(),
                d["move_number"].as_u64().unwrap_or(0) as u32,
            ));
        }
    }
    if experiments.len() > 1 {
        problems.push(format!("実験キーが {} 種類（別実験の混入）", experiments.len()));
    }
    if let Some(&total) = totals.iter().next_back() {
        if totals.len() > 1 {
            problems.push("shard_total が食い違う".into());
        } else if shards.len() as u64 != total {
            problems.push(format!("シャードが {}/{total} しか揃っていない", shards.len()));
        }
    }
    let mut seen: BTreeSet<(String, u32, u64)> = BTreeSet::new();
    for r in rows {
        let key = (
            r["game"].as_str().unwrap_or("?").to_string(),
            r["move_number"].as_u64().unwrap_or(0) as u32,
            r["seed"].as_u64().unwrap_or(0),
        );
        if !seen.insert(key.clone()) {
            problems.push(format!("重複行 {key:?}"));
        }
    }
    for (g, mn) in &want {
        if dropped.contains(&(g.clone(), *mn)) {
            continue;
        }
        for s in 0..seeds {
            if !seen.contains(&(g.clone(), *mn, s)) {
                problems.push(format!("欠測 {g} {mn} seed{s}"));
            }
        }
    }
    if problems.is_empty() {
        return true;
    }
    println!("\n### 入力の問題（{} 件）", problems.len());
    for p in problems.iter().take(20) {
        println!("  {p}");
    }
    if !allow_incomplete {
        eprintln!("入力が不完全です（--allow-incomplete で警告へ降格）");
        std::process::exit(1);
    }
    false
}

fn run_report(args: &[String]) {
    let mut allow_incomplete = false;
    let mut paths: Vec<String> = vec![];
    for a in args {
        match a.as_str() {
            "--allow-incomplete" => allow_incomplete = true,
            s if s.starts_with("--") => die(&format!("未知のオプション: {s}")),
            s => paths.push(s.to_string()),
        }
    }
    if paths.is_empty() {
        die("JSONL を指定してください");
    }
    let mut metas = vec![];
    let mut rows = vec![];
    for p in &paths {
        let Ok(content) = std::fs::read_to_string(p) else {
            die(&format!("{p}: 読めません"));
        };
        for line in content.lines().filter(|l| !l.trim().is_empty()) {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
                die(&format!("{p}: JSON として読めない行があります"));
            };
            if v["type"] == "meta" {
                metas.push(v);
            } else if v["schema"].as_u64() == Some(ROW_SCHEMA as u64) {
                rows.push(v);
            }
        }
    }
    report(&metas, &rows, allow_incomplete);
}
