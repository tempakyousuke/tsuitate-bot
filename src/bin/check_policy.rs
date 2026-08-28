//! **王手中の反則経済 P0-5: オフライン方策シミュレーション**
//! （issue #31。runtime には何も入らない）。
//!
//! アリーナ記録の**王手中の手番**を復元し、候補ごとの gain / p_legal と
//! **ランキングを作ったのと同じ粒子**（`ParticleSnapshot`）を取り、方策を
//! 反則するたび更新しながら**真実で裁定**する。
//!
//! estimand は2つ（issue #31 の P0-6 の門と同じ割り方）:
//!
//! - `foul` … 実戦で反則した王手手番（改善側）
//! - `nofoul` … 反則0で受理された王手手番から**同数**を決定論的に抽出した対照。
//!   新しいプローブを足す害（即時反則が増えていないか）はここで見る
//!
//! 方策（すべて王手中限定。arm 名は `check_policy::Policy::tag`）:
//!
//! | arm | 内容 |
//! | --- | --- |
//! | `current` | 現行 `combine_score`（更新規則は下記の3本を別 arm で持つ） |
//! | `alpha@k*` | 反則価格 ×k（`c = max(k × base, 床)`。床は倍率化しない） |
//! | `beta@k*l*` | 動的継続価値（`c_eff = max(0.5c, c − λ·ΔV)`、残り1回では無効） |
//! | `beta_order@l*` | 現行がプローブを選ぶ決定に限りプローブ同士だけ並べ替え |
//! | `solver_greedy` | 解消確率の argmax（参考） |
//!
//! **γ（手種別の較正マップ）は測らない**: P0-2 の方向つき較正誤差が
//! v14 +0.025 [−0.008, +0.057] / v13 −0.009 [−0.055, +0.036] と門
//! （CI 下限 > 0.1・両相手で同方向）に届かず**符号すら逆**だったので、
//! H3 は棄却済み（`docs/check-economy-p0.md`）。落ちた枝に arm を足すと
//! 「有限個の arm の最大値」を無駄に広げるだけになる。
//!
//! 反則後の更新規則は issue の「4本の対照」に対応する:
//!
//! 1. 記録された実反則列 … 出力の `record_fouls`（再現の基準）
//! 2. `current@static` … 初回ランキングの次点を取る静的方策
//!    （= 現行 `combine_score` が暗黙に置く「次善手固定」の仮定そのもの）
//! 3. `current@real` … **実再決定**（`MyFoul` を注入して `Estimator::update` と
//!    ランキングを走らせる。正解基準・思考予算をまるごと使う）
//! 4. それ以外の arm … **p-only shadow update**（snapshot ベースの仮想更新）
//!
//! `current@shadow` と `current@real` の差が**近似通過条件**の材料。
//!
//! usage:
//!   TSUITATE_THINK_BUDGET_MS=2000 cargo run --release --bin check_policy -- \
//!     [--seeds 4] [--jobs N] [--limit N] [--shard i/n] [--opponent estimator_v14] \
//!     [--alpha-k 1.5,2,3] [--beta-lambda 0.5,1] [--beta-k 1] [--no-real] \
//!     [--out data/check_policy.jsonl] <records...>
//!   cargo run --release --bin check_policy -- report [--shards N] <out-*.jsonl...>
//!
//! **思考予算は記録を取ったアリーナと揃える**（違うと粒子数が変わり、
//! 「その決定点で実際に見えていたランキング」でなくなる）。

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tsuitate_bot::check::CheckSolver;
use tsuitate_bot::check_economy::{
    CheckMoveKind, classify_move_kind, cluster_ratio_ci, entry_replayed,
};
use tsuitate_bot::check_policy::{
    CalibrationSums, EntrySetup, Policy, PolicyMove, SimOutcome, UpdateRule,
    entry_setup as check_policy_entry, fmt_num, policy_moves, simulate, truth_after,
};
use tsuitate_bot::observation::Observation;
use tsuitate_bot::protocol::Color;
use tsuitate_bot::scenario_core::{Replayed, clone_log, make_view, side_idx};
use tsuitate_bot::shogi::{Position, ShogiMove, parse_usi};
use tsuitate_bot::strategy::{self, EvalParams};
use tsuitate_bot::truth_replay::{for_each_decision_full, parse_bot_and_end};

/// JSONL の契約バージョン。**古い schema は集計から弾く**（issue #28 / #31 の契約）
const ROW_SCHEMA: u32 = 1;

fn die(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(2);
}

/// 監査対象の決定点（王手中の bot の手番1つ）
struct Point {
    game: String,
    move_number: u32,
    /// `foul`（実戦で反則した手番）/ `nofoul`（反則0の対照）
    estimand: &'static str,
    /// 型（P0-1 の粗い束）。反則0の手番は `no_foul`
    type_tag: String,
    /// 実戦の反則列（順番どおり）
    record_fouls: Vec<String>,
    /// 実戦で受理された手（復元の健全性検査用。方策には渡さない）
    record_accepted: String,
    /// 手番開始時（反則0）の状態
    entry: Replayed,
    /// 決定点の真実局面（裁定用。方策には渡さない）
    truth: Position,
    bot: Color,
}

/// 1 arm の結果（決定点 × seed × arm）
struct ArmOut {
    arm: String,
    out: SimOutcome,
    truth: tsuitate_bot::check_policy::TruthAfter,
    /// この arm のシミュレーションに掛かった実時間（µs。P1 のコスト見積り）
    sim_us: u64,
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

fn type_tag(first: CheckMoveKind, accepted: Option<CheckMoveKind>) -> String {
    use tsuitate_bot::check_economy::CoarseKind::*;
    let Some(acc) = accepted else {
        return "unfinished".to_string();
    };
    match (first.coarse(), acc.coarse()) {
        (Drop, _) => "drop",
        (NonKingBoard, King) => "nonking_king",
        (NonKingBoard, _) => "nonking_nonking",
        (King, King) => "king_king",
        (King, _) => "king_nonking",
    }
    .to_string()
}

fn parse_list(s: &str, what: &str) -> Vec<f64> {
    let mut v: Vec<f64> = s
        .split(',')
        .filter(|t| !t.trim().is_empty())
        .map(|t| {
            t.trim()
                .parse()
                .unwrap_or_else(|_| die(&format!("{what} は数値のカンマ区切り")))
        })
        .collect();
    v.sort_by(f64::total_cmp);
    v.dedup();
    v
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "report") {
        run_report(&args[1..]);
        return;
    }
    let mut seeds: u64 = 4;
    let mut jobs: usize =
        std::thread::available_parallelism().map_or(1, |n| n.get().saturating_sub(2).max(1));
    let mut limit = usize::MAX;
    let mut shard = (0usize, 1usize);
    let mut opponent: Option<String> = None;
    let mut alpha_ks = vec![1.5, 2.0, 3.0];
    let mut beta_lambdas = vec![0.5, 1.0];
    let mut beta_k = 1.0f64;
    let mut with_real = true;
    let mut allow_incomplete = false;
    let mut out_path: Option<String> = None;
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
            "--alpha-k" => {
                alpha_ks = parse_list(&need(args.get(i + 1), "--alpha-k"), "--alpha-k");
                i += 2;
            }
            "--beta-lambda" => {
                beta_lambdas = parse_list(&need(args.get(i + 1), "--beta-lambda"), "--beta-lambda");
                i += 2;
            }
            "--beta-k" => {
                beta_k = need(args.get(i + 1), "--beta-k")
                    .parse()
                    .unwrap_or_else(|_| die("--beta-k は数値"));
                i += 2;
            }
            "--no-real" => {
                with_real = false;
                i += 1;
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
    // **`--seeds 0` で「対象なし」を成功終了にできてはいけない**
    // （issue #28 が `mate_continue` で塞いだ穴と同じ: 空の集計で判定を偽造できる）
    if seeds == 0 {
        die("--seeds は 1 以上にしてください（0 だと1本もシミュレーションせずに集計が空になる）");
    }
    let files = collect_records(&specs);
    if files.is_empty() {
        die("記録ファイルが見つかりません");
    }

    let cfg = tsuitate_bot::config::ambient();
    let params = EvalParams::default();
    let eval_particles = strategy::eval_particles_for_budget(cfg.think_budget_ms);

    // ---- 決定点の収集（粒子は回さない）-----------------------------------
    let mut fouled: Vec<Point> = vec![];
    let mut clean: Vec<Point> = vec![];
    let mut games = 0u32;
    let mut broken = 0u32;
    let mut mismatched = 0u32;
    let mut skipped = 0u32;
    let mut record_opponents: BTreeMap<String, u32> = BTreeMap::new();
    // **解析に渡したのと同じ bytes** から記録集合の指紋を作る（ディスクを読み直すと
    // TOCTOU。issue #28 PR #30 レビュー3巡目の教訓）
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
                skipped += 1;
                return;
            };
            let events = post.logs[side_idx(d.side)].events();
            let record_fouls: Vec<String> = events[events.len() - d.fouls_this_turn as usize..]
                .iter()
                .filter_map(|e| match e {
                    Observation::MyFoul { usi, .. } => Some(usi.clone()),
                    _ => None,
                })
                .collect();
            // 型は P0-1 と同じ規約（bot の意図 = `captures_checker` で分ける）
            let view = make_view(&entry.pos, d.side, &entry.fouls);
            let log = &entry.logs[side_idx(d.side)];
            let mut solver = CheckSolver::new(&view, &[], &[], log);
            let accepted = end.moves.get(d.decision_id as usize).map(|m| m.usi.clone());
            let acc_kind = accepted
                .as_deref()
                .and_then(parse_usi)
                .map(|m| classify_move_kind(&m, &view, solver.as_mut()));
            let tag = match record_fouls.first().and_then(|u| parse_usi(u)) {
                Some(first) => {
                    let k = classify_move_kind(&first, &view, solver.as_mut());
                    type_tag(k, acc_kind)
                }
                None => "no_foul".to_string(),
            };
            let point = Point {
                game: short.clone(),
                move_number: d.pos.move_number(),
                estimand: if record_fouls.is_empty() { "nofoul" } else { "foul" },
                type_tag: tag,
                record_accepted: accepted.clone().unwrap_or_default(),
                record_fouls,
                truth: d.pos.clone(),
                entry,
                bot,
            };
            found.push(point);
        });
        if !ok {
            broken += 1;
            continue;
        }
        games += 1;
        for p in found {
            if p.estimand == "foul" { fouled.push(p) } else { clean.push(p) }
        }
    }
    // **反則0の対照は「同数」を決定論的に抽出する**（issue #31 の非劣性 estimand）。
    // 現行方策の順位で選ぶと対照が方策に引きずられるので、(局, 手数) の辞書順で
    // 等間隔に取る。全記録を見てから割るので、シャードが違っても同じ標本になる
    fouled.sort_by(|a, b| (&a.game, a.move_number).cmp(&(&b.game, b.move_number)));
    clean.sort_by(|a, b| (&a.game, a.move_number).cmp(&(&b.game, b.move_number)));
    let want = fouled.len();
    if clean.len() > want && want > 0 {
        let step = clean.len() as f64 / want as f64;
        let keep: BTreeSet<usize> = (0..want)
            .map(|k| ((k as f64 * step) as usize).min(clean.len() - 1))
            .collect();
        clean = clean
            .into_iter()
            .enumerate()
            .filter(|(i, _)| keep.contains(i))
            .map(|(_, p)| p)
            .collect();
    }
    let mut points: Vec<Point> = fouled;
    points.extend(clean);
    points.sort_by(|a, b| (&a.game, a.move_number).cmp(&(&b.game, b.move_number)));
    // シャードは**局単位**（同じ局の決定点がシャードをまたがない）
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
    if points.len() > limit {
        points.truncate(limit);
    }
    if points.is_empty() {
        die("対象の決定点がありません（記録と --opponent を確認）");
    }

    // ---- arm の定義 --------------------------------------------------------
    let mut policies: Vec<(String, Policy)> = vec![("current".into(), Policy::Current)];
    for k in &alpha_ks {
        policies.push((Policy::Alpha { k: *k }.tag(), Policy::Alpha { k: *k }));
    }
    for l in &beta_lambdas {
        let b = Policy::Beta { k: beta_k, lambda: *l };
        policies.push((b.tag(), b));
        let o = Policy::BetaOrder { lambda: *l };
        policies.push((o.tag(), o));
    }
    policies.push(("solver_greedy".into(), Policy::SolverGreedy));

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
        "決定点 {}（反則あり {} / 反則0の対照 {}）/ seeds {seeds} / jobs {jobs} / shard {}/{}",
        points.len(),
        points.iter().filter(|p| p.estimand == "foul").count(),
        points.iter().filter(|p| p.estimand == "nofoul").count(),
        shard.0,
        shard.1,
    );
    println!(
        "arm: {} + current@static{} / 思考予算 {}ms / eval_particles {eval_particles} / config {}",
        policies
            .iter()
            .map(|(t, _)| t.as_str())
            .collect::<Vec<_>>()
            .join(","),
        if with_real { " + current@real" } else { "（--no-real）" },
        cfg.think_budget_ms,
        cfg.fingerprint(),
    );
    println!("source_fingerprint {}", env!("TSUITATE_SOURCE_FINGERPRINT"));

    // ---- 実行（決定点 × seed）---------------------------------------------
    let units: Vec<(usize, u64)> = (0..points.len())
        .flat_map(|p| (0..seeds).map(move |s| (p, s)))
        .collect();
    let effective_jobs = jobs.min(units.len()).max(1);
    let next = Arc::new(Mutex::new(0usize));
    let lines: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(vec![]));
    // ランキングが取れなかった決定点は**明示的に meta へ残す**（定跡手・候補ゼロ）。
    // 黙って行が欠けると `check_inputs` が「seed ごと欠けた」と区別できず、
    // 2時間走った集約が最後に失敗する。逆に記録しておけば、記録に無い欠落
    // （= 本当に落ちた行）は今までどおり検出できる
    let dropped: Arc<Mutex<BTreeSet<(String, u32)>>> = Arc::new(Mutex::new(BTreeSet::new()));
    let points = Arc::new(points);
    let policies = Arc::new(policies);
    let started = std::time::Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..effective_jobs {
            let next = Arc::clone(&next);
            let lines = Arc::clone(&lines);
            let dropped = Arc::clone(&dropped);
            let points = Arc::clone(&points);
            let policies = Arc::clone(&policies);
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
                    match run_unit(
                        &points[pi],
                        seed,
                        &policies,
                        params,
                        eval_particles,
                        with_real,
                    ) {
                        Some(rows) => lines.lock().unwrap().extend(rows),
                        None => {
                            dropped
                                .lock()
                                .unwrap()
                                .insert((points[pi].game.clone(), points[pi].move_number));
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
            v["arm"].as_str().unwrap_or("").to_string(),
        )
    });
    let dropped = Arc::try_unwrap(dropped).ok().unwrap().into_inner().unwrap();
    if !dropped.is_empty() {
        eprintln!(
            "ランキングが取れずに落とした決定点: {} 件（meta の dropped に残す）",
            dropped.len()
        );
    }
    eprintln!(
        "{} 行 / {:.1}分",
        lines.len(),
        started.elapsed().as_secs_f64() / 60.0
    );

    let records_fingerprint: String =
        digest.finalize().iter().map(|b| format!("{b:02x}")).collect();
    // **判定に効く実験キー**（全シャードで完全一致していることを `report` が要求する）
    let experiment = serde_json::json!({
        "opponent": opponent.clone().unwrap_or_default(),
        "budget_ms": cfg.think_budget_ms,
        "eval_particles": eval_particles,
        "seeds": seeds,
        "arms": policies.iter().map(|(t, _)| t.clone()).collect::<Vec<_>>(),
        "with_real": with_real,
        "beta_k": fmt_num(beta_k),
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
        // **期待する行の骨格を meta 自身に残す**（ある seed の全 arm がまとめて
        // 欠けても検出できるように。issue #28 PR #30 レビュー2巡目 [P1]）
        "points_detail": points
            .iter()
            .map(|p| serde_json::json!({
                "game": p.game,
                "move_number": p.move_number,
                "estimand": p.estimand,
                "type": p.type_tag,
                "record_fouls": p.record_fouls.len(),
            }))
            .collect::<Vec<_>>(),
        // **ランキングが取れなかった決定点**（定跡手・候補ゼロ）。行が無いのが
        // 正常なので期待から外す。記録に無い欠落は今までどおり失敗させる
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
            "\n**シャード {}/{} の部分集計**（判定は `check_policy report` で全シャードを集めてから）",
            shard.0, shard.1
        );
    }
    report(&[meta], &lines, allow_incomplete);
}

/// 1 unit（決定点 × seed）を回して JSONL の行を返す
fn run_unit(
    p: &Point,
    seed: u64,
    policies: &[(String, Policy)],
    params: &EvalParams,
    eval_particles: usize,
    with_real: bool,
) -> Option<Vec<serde_json::Value>> {
    let side = p.entry.pos.turn();
    let king = p.entry.pos.king_square(side);
    // **ランキングを作ったのと同じ粒子**で仮想更新する（issue #28 P0-3 の教訓）。
    // prewarm は1回だけで、実再決定はこの instance の clone に反則を食わせた継続
    let setup = check_policy_entry(&p.entry, &p.truth, seed, params, eval_particles)?;
    let EntrySetup { strat, log: entry_log, moves, p0, updater, .. } = setup;
    // **健全性**: 反則0での仮想更新は初回ランキングの p_legal を再現するはず。
    // ずれるなら `evaluate` の経路（min キャップや別のノブ）を取りこぼしている
    let p_repro = updater.p_after(&moves, &[]);
    let repro_err = p0
        .iter()
        .zip(&p_repro)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);

    let fouls_before = p.entry.fouls[side_idx(side)];
    let opp_fouls = p.entry.fouls[side_idx(side.other())];
    let mut cal = CalibrationSums::default();
    cal.add(&moves, &p0);

    let mut arms: Vec<ArmOut> = vec![];
    let mut run = |arm: String, policy: &Policy, rule: UpdateRule<'_>| {
        let t = std::time::Instant::now();
        let out = simulate(policy, &moves, &p0, params, fouls_before, opp_fouls, rule);
        let truth = truth_after(&p.truth, p.bot, out.accepted.as_deref());
        arms.push(ArmOut {
            arm,
            out,
            truth,
            sim_us: t.elapsed().as_micros() as u64,
        });
    };
    // 2. 静的方策（現行 `combine_score` が暗黙に置く仮定そのもの）
    run("current@static".into(), &Policy::Current, UpdateRule::Static);
    // 4. p-only shadow update（全 arm 共通の更新規則）
    for (tag, policy) in policies {
        run(format!("{tag}@shadow"), policy, UpdateRule::Shadow(&updater));
    }
    // 3. 実再決定（正解基準。呼び出しごとに思考予算をまるごと使う）
    if with_real {
        let mut real = |fouls: &[ShogiMove]| -> Option<Vec<PolicyMove>> {
            let mut b = strat.clone_boxed()?;
            let mut post_log = clone_log(&entry_log);
            let mut post_fouls = p.entry.fouls;
            for m in fouls {
                post_fouls[side_idx(side)] += 1;
                post_log.record(Observation::MyFoul {
                    move_number: p.entry.pos.move_number(),
                    usi: m.to_usi(),
                });
            }
            let post_view = make_view(&p.entry.pos, side, &post_fouls);
            let tried: HashSet<String> = fouls.iter().map(ShogiMove::to_usi).collect();
            b.choose(&post_view, &post_log, &tried)?;
            let r = b.last_ranking()?.to_vec();
            let mut s = CheckSolver::new(&post_view, &[], fouls, &post_log);
            Some(policy_moves(&r, &post_view, &p.truth, s.as_mut(), king))
        };
        let t = std::time::Instant::now();
        let out = simulate(
            &Policy::Current,
            &moves,
            &p0,
            params,
            fouls_before,
            opp_fouls,
            UpdateRule::Real(&mut real),
        );
        let truth = truth_after(&p.truth, p.bot, out.accepted.as_deref());
        arms.push(ArmOut {
            arm: "current@real".into(),
            out,
            truth,
            sim_us: t.elapsed().as_micros() as u64,
        });
    }

    Some(
        arms.into_iter()
            .map(|a| {
                serde_json::json!({
                    "schema": ROW_SCHEMA,
                    "game": p.game,
                    "move_number": p.move_number,
                    "estimand": p.estimand,
                    "type": p.type_tag,
                    "seed": seed,
                    "arm": a.arm,
                    "fouls": a.out.fouls,
                    "record_fouls": p.record_fouls.len(),
                    "foul_limit": a.out.foul_limit,
                    "accepted": a.out.accepted.clone().unwrap_or_default(),
                    "accepted_kind": a.out.accepted_kind.map(|k| k.tag()).unwrap_or(""),
                    // 復元の健全性（P0-4 と同じ読み方）: シミュレーションが
                    // 実戦の選択をどれだけ再現しているか。**判定の前にここを見る**
                    "record_accepted_matches":
                        a.out.accepted.as_deref() == Some(p.record_accepted.as_str()),
                    "record_first_matches": a.out.sequence.first().map(String::as_str)
                        == p.record_fouls.first().map(String::as_str)
                            .or(Some(p.record_accepted.as_str())),
                    "sequence": a.out.sequence,
                    "updates": a.out.updates,
                    "sim_us": a.sim_us,
                    "material": a.truth.material,
                    "mated_in_1": a.truth.mated_in_1,
                    "next_check": a.truth.next_check,
                    "truth_accepted": a.truth.accepted,
                    "candidates": moves.len(),
                    "repro_err": repro_err,
                    "calibration": cal.to_json(),
                })
            })
            .collect(),
    )
}

// ---- 集計 ------------------------------------------------------------------

#[derive(Default, Clone)]
struct ArmStats {
    n: usize,
    fouls: Vec<f64>,
    foul_limit: usize,
    heavy: usize,
    accepted: usize,
    mated: usize,
    next_check: usize,
    material: Vec<f64>,
    sim_us: Vec<f64>,
    updates: Vec<f64>,
    /// 局ごとの (反則の和, 決定点数) — cluster bootstrap の統計単位
    by_game: BTreeMap<String, (f64, f64)>,
    /// 局ごとの (被一手詰めの数, 決定点数)
    mate_by_game: BTreeMap<String, (f64, f64)>,
}

impl ArmStats {
    fn push(&mut self, r: &serde_json::Value) {
        let game = r["game"].as_str().unwrap_or("?").to_string();
        let fouls = r["fouls"].as_f64().unwrap_or(0.0);
        self.n += 1;
        self.fouls.push(fouls);
        if r["foul_limit"].as_bool().unwrap_or(false) {
            self.foul_limit += 1;
        }
        if fouls >= 8.0 {
            self.heavy += 1;
        }
        if r["truth_accepted"].as_bool().unwrap_or(false) {
            self.accepted += 1;
            self.material.push(r["material"].as_f64().unwrap_or(0.0));
        }
        if r["mated_in_1"].as_bool().unwrap_or(false) {
            self.mated += 1;
        }
        if r["next_check"].as_bool().unwrap_or(false) {
            self.next_check += 1;
        }
        self.sim_us.push(r["sim_us"].as_f64().unwrap_or(0.0));
        self.updates.push(r["updates"].as_f64().unwrap_or(0.0));
        let e = self.by_game.entry(game.clone()).or_default();
        e.0 += fouls;
        e.1 += 1.0;
        let m = self.mate_by_game.entry(game).or_default();
        m.0 += f64::from(u8::from(r["mated_in_1"].as_bool().unwrap_or(false)));
        m.1 += 1.0;
    }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn pct(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[(((s.len() - 1) as f64) * q).round() as usize]
}

/// 局ごとの**ペア差**（arm − current）の cluster bootstrap CI。
///
/// 統計単位は元対局（seed も決定点も独立標本として数えない）。
fn paired_ci(
    rows: &[&serde_json::Value],
    arm: &str,
    base: &str,
    key: &dyn Fn(&serde_json::Value) -> f64,
) -> (f64, f64, f64) {
    // (局, 決定点, seed) → arm → 値
    let mut by: BTreeMap<(String, u64, u64), BTreeMap<String, f64>> = BTreeMap::new();
    for r in rows {
        by.entry((
            r["game"].as_str().unwrap_or("?").to_string(),
            r["move_number"].as_u64().unwrap_or(0),
            r["seed"].as_u64().unwrap_or(0),
        ))
        .or_default()
        .insert(r["arm"].as_str().unwrap_or("?").to_string(), key(r));
    }
    let mut clusters: BTreeMap<String, (f64, f64)> = BTreeMap::new();
    for ((game, _, _), m) in &by {
        if let (Some(a), Some(b)) = (m.get(arm), m.get(base)) {
            let e = clusters.entry(game.clone()).or_default();
            e.0 += a - b;
            e.1 += 1.0;
        }
    }
    let v: Vec<(f64, f64)> = clusters.values().copied().collect();
    let num: f64 = v.iter().map(|(a, _)| a).sum();
    let den: f64 = v.iter().map(|(_, b)| b).sum();
    let point = if den > 0.0 { num / den } else { f64::NAN };
    let (lo, hi) = cluster_ratio_ci(&v, 0.05, 0x31_2026);
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
    println!("\n=== P0-5: オフライン方策シミュレーション（issue #31）===");
    println!(
        "  実験: 相手 {} / 予算 {}ms / seeds {} / arms {} / 記録 {} / code {}",
        exp["opponent"], exp["budget_ms"], exp["seeds"], exp["arms"], exp["records"],
        exp["source_fingerprint"],
    );
    // 健全性: 仮想更新が初回ランキングの p_legal を再現できているか
    let repro: Vec<f64> = rows
        .iter()
        .filter(|r| r["arm"] == "current@static")
        .map(|r| r["repro_err"].as_f64().unwrap_or(0.0))
        .collect();
    println!(
        "  健全性: 反則0での仮想更新と初回ランキングの p_legal の最大差 平均 {:.4} / p95 {:.4}",
        mean(&repro),
        pct(&repro, 0.95)
    );
    println!(
        "         （大きいなら `evaluate` の p_legal 経路を取りこぼしている。判定の前にここを見る）"
    );
    // **復元の健全性**（P0-4 と同じ読み方）: オフラインの再現が実戦の選択を
    // どれだけ再現できているか。低いと「実戦のその決定そのもの」ではなく
    // 「同じ状態から現行方策が引き直したときの分布」を見ていることになる
    for arm in ["current@static", "current@real"] {
        let sel: Vec<&serde_json::Value> = rows.iter().filter(|r| r["arm"] == arm).collect();
        if sel.is_empty() {
            continue;
        }
        let rate = |k: &str| -> f64 {
            let v: Vec<f64> = sel
                .iter()
                .map(|r| f64::from(u8::from(r[k].as_bool().unwrap_or(false))))
                .collect();
            100.0 * mean(&v)
        };
        println!(
            "  復元（{arm}）: 最初に試した手が実戦と一致 {:.1}% / 受理手が実戦と一致 {:.1}%",
            rate("record_first_matches"),
            rate("record_accepted_matches")
        );
    }

    for estimand in ["foul", "nofoul"] {
        let sel: Vec<&serde_json::Value> =
            rows.iter().filter(|r| r["estimand"] == estimand).collect();
        if sel.is_empty() {
            continue;
        }
        let mut by_arm: BTreeMap<String, ArmStats> = BTreeMap::new();
        for r in &sel {
            by_arm
                .entry(r["arm"].as_str().unwrap_or("?").to_string())
                .or_default()
                .push(r);
        }
        let points: BTreeSet<(String, u64)> = sel
            .iter()
            .map(|r| {
                (
                    r["game"].as_str().unwrap_or("?").to_string(),
                    r["move_number"].as_u64().unwrap_or(0),
                )
            })
            .collect();
        println!(
            "\n--- estimand {estimand}（決定点 {} / unit {}）---",
            points.len(),
            by_arm.values().map(|s| s.n).max().unwrap_or(0)
        );
        let record_fouls = mean(
            &sel.iter()
                .filter(|r| r["arm"] == "current@static")
                .map(|r| r["record_fouls"].as_f64().unwrap_or(0.0))
                .collect::<Vec<_>>(),
        );
        println!("  実戦の反則/手番（記録）: {record_fouls:.2}");
        println!(
            "  {:<22} {:>7} {:>8} {:>7} {:>7} {:>8} {:>8} {:>9} {:>9}",
            "arm", "反則/番", "Δ vs cur", "上限%", "≥8%", "受理%", "被詰1%", "材料差", "更新/番"
        );
        for (arm, st) in &by_arm {
            let n = st.n.max(1) as f64;
            let (d, lo, hi) = if arm == "current@shadow" {
                (0.0, 0.0, 0.0)
            } else {
                paired_ci(&sel, arm, "current@shadow", &|r| {
                    r["fouls"].as_f64().unwrap_or(0.0)
                })
            };
            println!(
                "  {:<22} {:>7.2} {:>+8.2} {:>7.1} {:>7.1} {:>8.1} {:>8.1} {:>9.2} {:>9.2}{}",
                arm,
                mean(&st.fouls),
                d,
                100.0 * st.foul_limit as f64 / n,
                100.0 * st.heavy as f64 / n,
                100.0 * st.accepted as f64 / n,
                100.0 * st.mated as f64 / n,
                mean(&st.material),
                mean(&st.updates),
                if arm == "current@shadow" {
                    String::new()
                } else {
                    format!("  [{lo:+.2}, {hi:+.2}]")
                }
            );
        }
        // 受理直後の被一手詰め（真実ベース。受理手の gain は自己正当化するので主指標）
        println!("  被一手詰めのペア差（vs current@shadow、元対局 cluster CI）:");
        for arm in by_arm.keys().filter(|a| *a != "current@shadow") {
            let (d, lo, hi) = paired_ci(&sel, arm, "current@shadow", &|r| {
                f64::from(u8::from(r["mated_in_1"].as_bool().unwrap_or(false)))
            });
            println!("    {arm:<22} {d:+.4} [{lo:+.4}, {hi:+.4}]");
        }
        // 型別の反則（α を型で切らずに一律に掛けると型2 が沈む、を見る）
        let mut types: BTreeSet<String> = BTreeSet::new();
        for r in &sel {
            types.insert(r["type"].as_str().unwrap_or("?").to_string());
        }
        if types.len() > 1 {
            println!("  型別の反則/手番:");
            print!("    {:<22}", "arm");
            for t in &types {
                print!(" {t:>18}");
            }
            println!();
            for arm in by_arm.keys() {
                print!("    {arm:<22}");
                for t in &types {
                    let v: Vec<f64> = sel
                        .iter()
                        .filter(|r| r["arm"] == arm.as_str() && r["type"] == t.as_str())
                        .map(|r| r["fouls"].as_f64().unwrap_or(0.0))
                        .collect();
                    print!(" {:>18.2}", mean(&v));
                }
                println!();
            }
        }
    }

    // 近似通過条件（issue #31 の事前登録）
    approximation_gate(rows);
    calibration(rows);
    println!(
        "\n  コスト: 仮想更新 1回の実時間 p95 {:.0}µs（P1-β は候補×候補ぶん掛かる。思考 p95 の悪化 +200ms 以内が P1 の門）",
        pct(
            &rows
                .iter()
                .filter(|r| r["arm"] == "current@shadow" && r["updates"].as_f64().unwrap_or(0.0) > 0.0)
                .map(|r| r["sim_us"].as_f64().unwrap_or(0.0)
                    / r["updates"].as_f64().unwrap_or(1.0))
                .collect::<Vec<_>>(),
            0.95
        )
    );
}

/// **近似通過条件**（事前登録）: β の対現行改善量が正のときだけ比率を定義し、
/// 仮想更新と実再決定の「受理までの反則数/手番」の差がその **25% 以下**。
/// 改善量が 0 以下なら β は不採用。
fn approximation_gate(rows: &[serde_json::Value]) {
    let sel: Vec<&serde_json::Value> = rows.iter().filter(|r| r["estimand"] == "foul").collect();
    if sel.is_empty() {
        return;
    }
    let has_real = sel.iter().any(|r| r["arm"] == "current@real");
    println!("\n--- 近似通過条件（p-only shadow update vs 実再決定）---");
    if !has_real {
        println!("  実再決定の arm がありません（--no-real で回した）。判定は出せない");
        return;
    }
    let (gap, glo, ghi) = paired_ci(&sel, "current@shadow", "current@real", &|r| {
        r["fouls"].as_f64().unwrap_or(0.0)
    });
    let agree = {
        let mut by: BTreeMap<(String, u64, u64), BTreeMap<String, String>> = BTreeMap::new();
        for r in &sel {
            by.entry((
                r["game"].as_str().unwrap_or("?").to_string(),
                r["move_number"].as_u64().unwrap_or(0),
                r["seed"].as_u64().unwrap_or(0),
            ))
            .or_default()
            .insert(
                r["arm"].as_str().unwrap_or("?").to_string(),
                r["accepted"].as_str().unwrap_or("").to_string(),
            );
        }
        let pairs: Vec<bool> = by
            .values()
            .filter_map(|m| Some(m.get("current@shadow")? == m.get("current@real")?))
            .collect();
        if pairs.is_empty() {
            f64::NAN
        } else {
            pairs.iter().filter(|b| **b).count() as f64 / pairs.len() as f64
        }
    };
    println!("  受理までの反則数の差（shadow − real）: {gap:+.3} [{glo:+.3}, {ghi:+.3}]");
    println!("  受理手が一致した割合: {:.1}%", 100.0 * agree);
    // β の対現行改善量（反則が減る = 負の差なので符号を反転して「改善量」にする）
    let best_beta = rows
        .iter()
        .filter(|r| r["estimand"] == "foul")
        .filter_map(|r| r["arm"].as_str())
        .filter(|a| a.starts_with("beta"))
        .collect::<BTreeSet<_>>();
    let mut best: Option<(String, f64)> = None;
    for arm in best_beta {
        let (d, _, _) = paired_ci(&sel, arm, "current@shadow", &|r| {
            r["fouls"].as_f64().unwrap_or(0.0)
        });
        let improve = -d;
        if best.as_ref().is_none_or(|(_, b)| improve > *b) {
            best = Some((arm.to_string(), improve));
        }
    }
    match best {
        Some((arm, improve)) if improve > 0.0 => {
            let ratio = gap.abs() / improve;
            println!(
                "  β の対現行改善量の最大は {arm} の {improve:+.3}（反則/手番）→ 近似誤差 / 改善量 = {ratio:.2} → {}",
                if ratio <= 0.25 {
                    "**通過**（仮想更新で事前スコアを作ってよい）"
                } else {
                    "**不通過**（P1-β は候補ごとに推定器を clone して仮想 MyFoul を注入する設計が要る）"
                }
            );
        }
        Some((arm, improve)) => println!(
            "  β の対現行改善量が 0 以下（{arm} で {improve:+.3}）→ **β は不採用**（比率は定義しない）"
        ),
        None => println!("  β の arm がありません"),
    }
}

/// 候補分布上の較正（**現行方策が選んだ手ではなく候補全体**）。
///
/// P0-2 は「選ばれた手」の較正なので、順位を変えた後の候補分布へは外挿できない。
/// ここは初回ランキングの全候補を真実でラベル付けして手種ごとに出す。
fn calibration(rows: &[serde_json::Value]) {
    // 1決定点×seed につき1回だけ数える（arm ごとに同じ値が入っている）
    let mut by_kind: BTreeMap<String, BTreeMap<String, (f64, f64, f64)>> = BTreeMap::new();
    let mut seen: HashSet<(String, u64, u64)> = HashSet::new();
    for r in rows {
        let key = (
            r["game"].as_str().unwrap_or("?").to_string(),
            r["move_number"].as_u64().unwrap_or(0),
            r["seed"].as_u64().unwrap_or(0),
        );
        if !seen.insert(key.clone()) {
            continue;
        }
        let Some(obj) = r["calibration"].as_object() else {
            continue;
        };
        for (kind, v) in obj {
            let e = by_kind
                .entry(kind.clone())
                .or_default()
                .entry(key.0.clone())
                .or_default();
            e.0 += v["p_sum"].as_f64().unwrap_or(0.0);
            e.1 += v["legal"].as_f64().unwrap_or(0.0);
            e.2 += v["n"].as_f64().unwrap_or(0.0);
        }
    }
    if by_kind.is_empty() {
        return;
    }
    println!("\n--- 候補分布上の較正（真実ラベル・元対局 cluster CI）---");
    println!(
        "  {:<18} {:>9} {:>9} {:>9} {:>24}",
        "手種", "候補数", "平均予測", "合法率", "方向つき較正誤差"
    );
    for (kind, by_game) in &by_kind {
        let n: f64 = by_game.values().map(|(_, _, n)| n).sum();
        let psum: f64 = by_game.values().map(|(p, _, _)| p).sum();
        let legal: f64 = by_game.values().map(|(_, l, _)| l).sum();
        if n <= 0.0 {
            continue;
        }
        // 誤差の cluster: 局ごとの (予測の和 − 合法数, 候補数)
        let clusters: Vec<(f64, f64)> = by_game.values().map(|(p, l, n)| (p - l, *n)).collect();
        let (lo, hi) = cluster_ratio_ci(&clusters, 0.05, 0x31_2027);
        println!(
            "  {:<18} {:>9.0} {:>9.3} {:>9.3} {:>+11.3} [{lo:+.3}, {hi:+.3}]",
            kind,
            n,
            psum / n,
            legal / n,
            (psum - legal) / n
        );
    }
}

/// `report` の入力契約。破っている点を全部返す
fn check_inputs(metas: &[serde_json::Value], rows: &[serde_json::Value]) -> Vec<String> {
    let mut out = vec![];
    if metas.is_empty() {
        return vec!["meta 行がありません".into()];
    }
    let first = &metas[0]["experiment"];
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
    // シャードが揃っているか（欠けると決定点の分母が縮む）
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
            "シャードが欠けています（{:?} / 全 {total}）: 決定点の分母が縮むので判定は出せない",
            shards
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
    // まとめて欠けても検出する）
    let seeds = first["seeds"].as_u64().unwrap_or(0);
    let mut want_arms: Vec<String> = vec!["current@static".into()];
    for a in first["arms"].as_array().into_iter().flatten() {
        want_arms.push(format!("{}@shadow", a.as_str().unwrap_or("?")));
    }
    if first["with_real"].as_bool().unwrap_or(false) {
        want_arms.push("current@real".into());
    }
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
    let dropped: HashSet<(String, u64)> = metas
        .iter()
        .flat_map(|m| m["dropped"].as_array().cloned().unwrap_or_default())
        .map(|d| {
            (
                d["game"].as_str().unwrap_or("?").to_string(),
                d["move_number"].as_u64().unwrap_or(0),
            )
        })
        .collect();
    let mut lacks: Vec<String> = vec![];
    for m in metas {
        for d in m["points_detail"].as_array().into_iter().flatten() {
            let g = d["game"].as_str().unwrap_or("?").to_string();
            let mn = d["move_number"].as_u64().unwrap_or(0);
            // ランキングが取れなかった決定点は行が無いのが正常
            if dropped.contains(&(g.clone(), mn)) {
                continue;
            }
            for arm in &want_arms {
                let got = seen.get(&(g.clone(), mn, arm.clone())).copied().unwrap_or(0) as u64;
                if got != seeds {
                    lacks.push(format!("{g}#{mn} {arm}: {got}/{seeds}"));
                }
            }
        }
    }
    if !lacks.is_empty() {
        out.push(format!(
            "meta が宣言した決定点に対して行が {} 箇所欠けています: {}{}",
            lacks.len(),
            lacks.iter().take(3).cloned().collect::<Vec<_>>().join(" / "),
            if lacks.len() > 3 { " ..." } else { "" }
        ));
    }
    out
}

fn run_report(args: &[String]) {
    let mut allow_incomplete = false;
    let mut want_shards: Option<usize> = None;
    let mut paths: Vec<String> = vec![];
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--allow-incomplete" => {
                allow_incomplete = true;
                i += 1;
            }
            "--shards" => {
                want_shards = Some(
                    args.get(i + 1)
                        .and_then(|s| s.parse().ok())
                        .unwrap_or_else(|| die("--shards には数値が必要です")),
                );
                i += 2;
            }
            s if s.starts_with("--") => die(&format!("未知のオプション: {s}")),
            s => {
                paths.push(s.to_string());
                i += 1;
            }
        }
    }
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
    // `--shards` は**上書きではなく突き合わせ**
    if let Some(want) = want_shards {
        let total = metas
            .first()
            .and_then(|m| m["experiment"]["shard_total"].as_u64())
            .unwrap_or(1) as usize;
        if want != total {
            die(&format!(
                "--shards {want} は JSONL の meta（shard_total {total}）と食い違います"
            ));
        }
    }
    println!("JSONL {} 本 / 行 {}", paths.len(), rows.len());
    report(&metas, &rows, allow_incomplete);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exp(opponent: &str) -> serde_json::Value {
        serde_json::json!({
            "opponent": opponent, "budget_ms": 2000, "eval_particles": 106, "seeds": 2,
            "arms": ["current", "alpha@k2"], "with_real": true, "beta_k": "1",
            "jobs": 4, "shard_total": 1, "config": "c", "source_fingerprint": "s", "records": "r",
        })
    }

    fn meta(e: serde_json::Value, shard: u64) -> serde_json::Value {
        serde_json::json!({
            "schema": ROW_SCHEMA, "type": "meta", "experiment": e, "shard": shard,
            "games": 10, "points": 1,
            "points_detail": [{"game": "g1", "move_number": 41, "estimand": "foul",
                               "type": "nonking_king", "record_fouls": 2}],
        })
    }

    fn row(arm: &str, seed: u64, fouls: u32) -> serde_json::Value {
        serde_json::json!({
            "schema": ROW_SCHEMA, "game": "g1", "move_number": 41, "estimand": "foul",
            "type": "nonking_king", "seed": seed, "arm": arm, "fouls": fouls,
            "record_fouls": 2, "foul_limit": false, "accepted": "5i4h",
            "truth_accepted": true, "mated_in_1": false, "next_check": false,
            "material": 0.0, "updates": 1, "sim_us": 100, "repro_err": 0.0,
        })
    }

    fn full() -> Vec<serde_json::Value> {
        let mut v = vec![];
        for seed in 0..2 {
            for arm in ["current@static", "current@shadow", "alpha@k2@shadow", "current@real"] {
                v.push(row(arm, seed, 1));
            }
        }
        v
    }

    #[test]
    fn 揃った入力は契約を通る() {
        assert!(check_inputs(&[meta(exp("estimator_v14"), 0)], &full()).is_empty());
    }

    #[test]
    fn 実験キーが違うjsonlは混ぜられない() {
        let metas = vec![meta(exp("estimator_v14"), 0), meta(exp("estimator_v13"), 1)];
        let problems = check_inputs(&metas, &full());
        assert!(
            problems.iter().any(|p| p.contains("opponent")),
            "{problems:?}"
        );
    }

    #[test]
    fn armごと欠けた行を検出する() {
        // ある arm の seed が丸ごと欠けると Δ の分母が狂う
        let rows: Vec<serde_json::Value> = full()
            .into_iter()
            .filter(|r| !(r["arm"] == "alpha@k2@shadow" && r["seed"] == 1))
            .collect();
        let problems = check_inputs(&[meta(exp("estimator_v14"), 0)], &rows);
        assert!(
            problems.iter().any(|p| p.contains("alpha@k2@shadow")),
            "{problems:?}"
        );
    }

    /// ランキングが取れなかった決定点は meta の `dropped` に残し、期待から外す。
    /// **記録に無い欠落は今までどおり失敗させる**（黙って行が消えるのと区別する）
    #[test]
    fn 落とした決定点は期待から外れる() {
        let mut m = meta(exp("estimator_v14"), 0);
        // 行が1つも無い状態は、dropped に載っていなければ失敗
        assert!(!check_inputs(&[m.clone()], &[]).is_empty());
        m["dropped"] = serde_json::json!([{"game": "g1", "move_number": 41}]);
        assert!(
            check_inputs(&[m.clone()], &[]).is_empty(),
            "dropped に載っていれば行が無くても通る"
        );
        // 別の決定点の欠落は依然として失敗する
        m["dropped"] = serde_json::json!([{"game": "g1", "move_number": 99}]);
        assert!(!check_inputs(&[m], &[]).is_empty());
    }

    #[test]
    fn シャードが欠けたら判定を出さない() {
        let mut e = exp("estimator_v14");
        e["shard_total"] = serde_json::json!(2);
        let problems = check_inputs(&[meta(e, 0)], &full());
        assert!(
            problems.iter().any(|p| p.contains("シャードが欠けて")),
            "{problems:?}"
        );
    }

    #[test]
    fn ペア差は元対局単位で畳む() {
        // 同じ局の2 seed は1つの cluster。arm が片方しか無い unit は落とす
        let rows = full();
        let refs: Vec<&serde_json::Value> = rows.iter().collect();
        let (d, lo, hi) = paired_ci(&refs, "alpha@k2@shadow", "current@shadow", &|r| {
            r["fouls"].as_f64().unwrap_or(0.0)
        });
        assert_eq!(d, 0.0, "同じ反則数ならペア差は 0");
        assert_eq!((lo, hi), (0.0, 0.0));
    }
}
