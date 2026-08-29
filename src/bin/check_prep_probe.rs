//! **被王手の前の準備 P0-2: 較正プローブ**（issue #34。runtime には何も入らない）。
//!
//! 「王手が来たときの出口の本数と質は、**王手が来る前の自駒配置**で決まって
//! いるか」を、方策を変える前に測る（`bin/collision_probe` / `bin/eval_features` と
//! 同じ作り）。粒子を回さないので軽い。
//!
//! **単位は「bot の全 handoff」**（= bot が受理手を指し終えた時点。P1 が変えられる
//! 配置そのもの）。被王手手番だけに条件づけると
//! `F → 被王手 ← 相手の攻撃力 → K・反則` の collider が開き、形に直接効果が無くても
//! 分離してしまう。**主目的変数は「次の bot 手番に発生した王手中反則数」**
//! （次に王手されなければ 0）。
//!
//! 段は3つで、前の段が落ちたら次へ進まない（issue #34 の中止条件）:
//!
//! 1. **構成概念検証**: 「実際の王手駒を玉以外で真に合法に取れたか」（oracle）を
//!    F1 の起点別被覆がどれだけ再現するか。**ここで落ちたら反則回帰へ進まない**
//! 2. **発火母数**: 観測された王手中反則のうち、直前の handoff の候補手の中に
//!    F を改善する到達可能な手があった割合（`anchor_probe` の教訓）
//! 3. **主 estimand**: 層を固定したうえで主特徴量を Q90 / Q10 に置いたときの
//!    次手番の王手中反則数の**周辺平均差**（hurdle 分解。`check_prep::hurdle`）
//!
//! usage:
//!   cargo run --release --bin check_prep_probe -- [--out data/check_prep.csv]
//!     [--shard i/n] [--opponent estimator_v14] [--origin-weights <spec>]
//!     [--limit N] [--no-firing] <records...>
//!   cargo run --release --bin check_prep_probe -- report [--allow-incomplete]
//!     [--bootstrap N] <csv...>

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use tsuitate_bot::board::{Coord, parse_usi_square};
use tsuitate_bot::check::CheckSolver;
use tsuitate_bot::check_economy::{hypothesis_stats, true_checkers, view_from_model};
use tsuitate_bot::check_prep::{
    OriginClass, OriginWeights, Snapshot, cheb, cluster_bootstrap,
    decision_snapshots, features, hurdle, k_def_truth, known_enemy_squares,
    nonking_capture_of_checker, origin_squares, pieces_after,
};
use tsuitate_bot::model::GameModel;
use tsuitate_bot::observation::ObservationLog;
use tsuitate_bot::protocol::{Color, Role, VisiblePiece};
use tsuitate_bot::scenario_core::clone_log;
use tsuitate_bot::shogi::{Position, ShogiMove, parse_usi, piece_value};
use tsuitate_bot::strategy::{candidate_moves, candidate_moves_with_log};
use tsuitate_bot::truth_replay::{add_move_obs, parse_bot_and_end, side_idx};

/// CSV / 集約の契約バージョン。**古い schema は集計から弾く**
const ROW_SCHEMA: u32 = 1;

/// 反則負けになる累計反則数（サイトの judge.ts と同じ）
const FOUL_LIMIT: u32 = 10;

fn die(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(2);
}

// ---------------------------------------------------------------------------
// 行の定義（extract と report が同じ形を通る）
// ---------------------------------------------------------------------------

/// bot の handoff 1つ = CSV の1行
#[derive(Clone, Debug, Default)]
struct Row {
    game: String,
    decision_id: u64,
    move_number: u32,
    gote: bool,
    // --- 主特徴量（handoff 時点）と差分 ---
    f1_cov_w: f64,
    f1_cov_frac: f64,
    f1_total: u32,
    f1_cov: u32,
    f1_cov_sure: u32,
    f1_by_class: [(u32, u32); 3],
    f1_by_dist: [(u32, u32); 4],
    d_f1_cov_w: f64,
    f2_open_dirs: u32,
    f2_open_len: u32,
    f2_flight_all: u32,
    f2_flight_shielded: u32,
    f2_king_home: bool,
    f2_known_enemy_dist: Option<u32>,
    f3_open_uncovered: u32,
    f3_open_covered: u32,
    f3_own_occupied: u32,
    adj_total: u32,
    adj_covered: u32,
    f13_cov: f64,
    d_f3_open_uncovered: i32,
    /// `entry − handoff`（王手手が形を壊した量）。王手されなければ None
    e_f1_cov_w: Option<f64>,
    e_f3_open_uncovered: Option<i32>,
    // --- 層・共変量 ---
    king_edge_dist: u32,
    own_pieces: u32,
    hand_count: u32,
    material_diff: f64,
    king_density: u32,
    remaining: u32,
    candidates: u32,
    // --- 結果 ---
    next_is_check: bool,
    fouls_in_check: u32,
    terminal_turn: bool,
    k_def_truth: Option<u32>,
    k_def_gen: Option<u32>,
    k_def_runtime: Option<u32>,
    // --- 機構（被王手条件付き。総効果の回帰には入れない）---
    checker_role: Option<Role>,
    checker_drop: Option<bool>,
    checker_capture: Option<bool>,
    checker_dist: Option<u32>,
    double_check: Option<bool>,
    hyp_n: Option<usize>,
    true_hyp_share: Option<f64>,
    hyp_entropy: Option<f64>,
    blind: Option<bool>,
    // --- 構成概念検証（oracle）---
    oracle_nonking_capture: Option<bool>,
    origin_covered: Option<bool>,
    /// そのうち**確定利き**（歩・桂・金銀の歩進）で被覆していたか。
    /// レイだけの被覆は隠れた敵駒に遮られうる楽観値なので分けて数える
    origin_covered_sure: Option<bool>,
    origin_in_c: Option<bool>,
    origin_class: Option<OriginClass>,
    origin_dist: Option<u32>,
    // --- 発火母数 ---
    improvable_any: Option<bool>,
    improvable_origin: Option<bool>,
}

const HEADER: &str = "game,decision_id,move_number,gote,f1_cov_w,f1_cov_frac,f1_total,f1_cov,\
f1_cov_sure,f1_knight_t,f1_knight_c,f1_line_t,f1_line_c,f1_diag_t,f1_diag_c,f1_d2_t,f1_d2_c,\
f1_d3_t,f1_d3_c,f1_d4_t,f1_d4_c,f1_d5_t,f1_d5_c,d_f1_cov_w,f2_open_dirs,f2_open_len,\
f2_flight_all,f2_flight_shielded,f2_king_home,f2_known_enemy_dist,f3_open_uncovered,\
f3_open_covered,f3_own_occupied,adj_total,adj_covered,f13_cov,d_f3_open_uncovered,e_f1_cov_w,e_f3_open_uncovered,\
king_edge_dist,own_pieces,hand_count,material_diff,king_density,remaining,candidates,\
next_is_check,fouls_in_check,terminal_turn,k_def_truth,k_def_gen,k_def_runtime,checker_role,\
checker_drop,checker_capture,checker_dist,double_check,hyp_n,true_hyp_share,hyp_entropy,blind,\
oracle_nonking_capture,origin_covered,origin_covered_sure,origin_in_c,origin_class,origin_dist,\
improvable_any,\
improvable_origin";

fn ob(v: Option<bool>) -> String {
    v.map_or(String::new(), |b| u8::from(b).to_string())
}
fn ou<T: std::fmt::Display>(v: Option<T>) -> String {
    v.map_or(String::new(), |x| x.to_string())
}
fn of(v: Option<f64>) -> String {
    v.map_or(String::new(), |x| format!("{x:.4}"))
}

impl Row {
    /// **列は `HEADER` と同じ順に積む**（書式文字列で並べると、列を足したときに
    /// 静かにずれる）。`cargo test` が列数の一致を検査する
    fn to_csv(&self) -> String {
        let c = self.f1_by_class;
        let d = self.f1_by_dist;
        let mut v: Vec<String> = vec![
            self.game.clone(),
            self.decision_id.to_string(),
            self.move_number.to_string(),
            u8::from(self.gote).to_string(),
            format!("{:.4}", self.f1_cov_w),
            format!("{:.4}", self.f1_cov_frac),
            self.f1_total.to_string(),
            self.f1_cov.to_string(),
            self.f1_cov_sure.to_string(),
        ];
        for (t, cv) in c.iter().chain(d.iter()) {
            v.push(t.to_string());
            v.push(cv.to_string());
        }
        v.extend([
            format!("{:+.4}", self.d_f1_cov_w),
            self.f2_open_dirs.to_string(),
            self.f2_open_len.to_string(),
            self.f2_flight_all.to_string(),
            self.f2_flight_shielded.to_string(),
            u8::from(self.f2_king_home).to_string(),
            ou(self.f2_known_enemy_dist),
            self.f3_open_uncovered.to_string(),
            self.f3_open_covered.to_string(),
            self.f3_own_occupied.to_string(),
            self.adj_total.to_string(),
            self.adj_covered.to_string(),
            format!("{:.4}", self.f13_cov),
            format!("{:+}", self.d_f3_open_uncovered),
            of(self.e_f1_cov_w),
            ou(self.e_f3_open_uncovered),
            self.king_edge_dist.to_string(),
            self.own_pieces.to_string(),
            self.hand_count.to_string(),
            format!("{:+.2}", self.material_diff),
            self.king_density.to_string(),
            self.remaining.to_string(),
            self.candidates.to_string(),
            u8::from(self.next_is_check).to_string(),
            self.fouls_in_check.to_string(),
            u8::from(self.terminal_turn).to_string(),
            ou(self.k_def_truth),
            ou(self.k_def_gen),
            ou(self.k_def_runtime),
            self.checker_role.map_or(String::new(), |r| format!("{r:?}")),
            ob(self.checker_drop),
            ob(self.checker_capture),
            ou(self.checker_dist),
            ob(self.double_check),
            ou(self.hyp_n),
            of(self.true_hyp_share),
            of(self.hyp_entropy),
            ob(self.blind),
            ob(self.oracle_nonking_capture),
            ob(self.origin_covered),
            ob(self.origin_covered_sure),
            ob(self.origin_in_c),
            self.origin_class.map_or(String::new(), |c| c.tag().to_string()),
            ou(self.origin_dist),
            ob(self.improvable_any),
            ob(self.improvable_origin),
        ]);
        v.join(",")
    }

    /// **壊れた値を欠測・0 へ黙って変換しない**（issue #31 PR #32 の教訓:
    /// 不正な値が「分母から脱落」に化けると門の割合が静かに動く）。
    /// 空文字を許すのは設計上 optional な列だけ。
    fn from_csv(path: &str, cols: &[&str], v: &[&str]) -> Row {
        let get = |name: &str| -> &str {
            let i = cols
                .iter()
                .position(|c| *c == name)
                .unwrap_or_else(|| die(&format!("{path}: 列 {name} がありません")));
            v.get(i).copied().unwrap_or_else(|| die(&format!("{path}: 列 {name} が欠けた行")))
        };
        let num = |name: &str| -> f64 {
            get(name)
                .parse()
                .unwrap_or_else(|_| die(&format!("{path}: {name} が数値でない: {}", get(name))))
        };
        let int = |name: &str| -> u32 {
            get(name)
                .parse()
                .unwrap_or_else(|_| die(&format!("{path}: {name} が整数でない: {}", get(name))))
        };
        let opt_int = |name: &str| -> Option<u32> {
            let s = get(name);
            if s.is_empty() {
                None
            } else {
                Some(s.parse().unwrap_or_else(|_| {
                    die(&format!("{path}: {name} が整数でない: {s}"))
                }))
            }
        };
        let opt_bool = |name: &str| -> Option<bool> {
            match get(name) {
                "" => None,
                "0" => Some(false),
                "1" => Some(true),
                other => die(&format!("{path}: {name} が 0/1 でない: {other}")),
            }
        };
        let opt_f = |name: &str| -> Option<f64> {
            let s = get(name);
            if s.is_empty() {
                None
            } else {
                Some(s.parse().unwrap_or_else(|_| {
                    die(&format!("{path}: {name} が数値でない: {s}"))
                }))
            }
        };
        let b = |name: &str| -> bool { opt_bool(name).unwrap_or_else(|| die(&format!("{path}: {name} は必須"))) };
        Row {
            game: get("game").to_string(),
            decision_id: num("decision_id") as u64,
            move_number: int("move_number"),
            gote: b("gote"),
            f1_cov_w: num("f1_cov_w"),
            f1_cov_frac: num("f1_cov_frac"),
            f1_total: int("f1_total"),
            f1_cov: int("f1_cov"),
            f1_cov_sure: int("f1_cov_sure"),
            f1_by_class: [
                (int("f1_knight_t"), int("f1_knight_c")),
                (int("f1_line_t"), int("f1_line_c")),
                (int("f1_diag_t"), int("f1_diag_c")),
            ],
            f1_by_dist: [
                (int("f1_d2_t"), int("f1_d2_c")),
                (int("f1_d3_t"), int("f1_d3_c")),
                (int("f1_d4_t"), int("f1_d4_c")),
                (int("f1_d5_t"), int("f1_d5_c")),
            ],
            d_f1_cov_w: num("d_f1_cov_w"),
            f2_open_dirs: int("f2_open_dirs"),
            f2_open_len: int("f2_open_len"),
            f2_flight_all: int("f2_flight_all"),
            f2_flight_shielded: int("f2_flight_shielded"),
            f2_king_home: b("f2_king_home"),
            f2_known_enemy_dist: opt_int("f2_known_enemy_dist"),
            f3_open_uncovered: int("f3_open_uncovered"),
            f3_open_covered: int("f3_open_covered"),
            f3_own_occupied: int("f3_own_occupied"),
            adj_total: int("adj_total"),
            adj_covered: int("adj_covered"),
            f13_cov: num("f13_cov"),
            d_f3_open_uncovered: num("d_f3_open_uncovered") as i32,
            e_f1_cov_w: opt_f("e_f1_cov_w"),
            e_f3_open_uncovered: {
                let s = get("e_f3_open_uncovered");
                if s.is_empty() {
                    None
                } else {
                    Some(s.parse().unwrap_or_else(|_| die(&format!("{path}: e_f3 が整数でない"))))
                }
            },
            king_edge_dist: int("king_edge_dist"),
            own_pieces: int("own_pieces"),
            hand_count: int("hand_count"),
            material_diff: num("material_diff"),
            king_density: int("king_density"),
            remaining: int("remaining"),
            candidates: int("candidates"),
            next_is_check: b("next_is_check"),
            fouls_in_check: int("fouls_in_check"),
            terminal_turn: b("terminal_turn"),
            k_def_truth: opt_int("k_def_truth"),
            k_def_gen: opt_int("k_def_gen"),
            k_def_runtime: opt_int("k_def_runtime"),
            checker_role: None,
            checker_drop: opt_bool("checker_drop"),
            checker_capture: opt_bool("checker_capture"),
            checker_dist: opt_int("checker_dist"),
            double_check: opt_bool("double_check"),
            hyp_n: opt_int("hyp_n").map(|v| v as usize),
            true_hyp_share: opt_f("true_hyp_share"),
            hyp_entropy: opt_f("hyp_entropy"),
            blind: opt_bool("blind"),
            oracle_nonking_capture: opt_bool("oracle_nonking_capture"),
            origin_covered: opt_bool("origin_covered"),
            origin_covered_sure: opt_bool("origin_covered_sure"),
            origin_in_c: opt_bool("origin_in_c"),
            origin_class: OriginClass::ALL
                .iter()
                .copied()
                .find(|c| c.tag() == get("origin_class")),
            origin_dist: opt_int("origin_dist"),
            improvable_any: opt_bool("improvable_any"),
            improvable_origin: opt_bool("improvable_origin"),
        }
    }
}

// ---------------------------------------------------------------------------
// 抽出
// ---------------------------------------------------------------------------

fn walk_jsonl(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
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

/// 記録の `chose` から「その決定点で厳密粒子がゼロだったか」を引く表。
/// **その手番の最初の chose だけ**を見る（反則後の再選択は別状態）。
fn blind_by_move(content: &str) -> BTreeMap<u32, Option<bool>> {
    let mut out: BTreeMap<u32, Option<bool>> = BTreeMap::new();
    for line in content.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        if v["type"] != "chose" {
            continue;
        }
        if let Some(mn) = v["move_number"].as_u64() {
            out.entry(mn as u32)
                .or_insert_with(|| v["debug"]["sample_slots"].as_u64().map(|s| s == 0));
        }
    }
    out
}

fn king_edge_dist(k: Coord) -> u32 {
    let f = i32::from(k.file).min(10 - i32::from(k.file));
    let r = i32::from(k.rank).min(10 - i32::from(k.rank));
    (f.min(r) - 1).max(0) as u32
}

fn material(pos: &Position, c: Color) -> f64 {
    let board: f64 = pos
        .pieces()
        .filter(|(_, p)| p.color == c)
        .map(|(_, p)| piece_value(p.role))
        .sum();
    let hand: f64 = pos.hand_map(c).iter().map(|(r, n)| piece_value(*r) * f64::from(*n)).sum();
    board + hand
}

/// その handoff の視界から、`entry`（相手の着手後・反則0）の状態を作る。
/// 観測の作り方は `truth_replay` の共有関数だけを使う。
fn entry_state(prev: &Snapshot, opp_mv: &ShogiMove) -> (Position, [ObservationLog; 2]) {
    let mut pos = prev.pos.clone();
    let mut logs = [clone_log(&prev.logs[0]), clone_log(&prev.logs[1])];
    let captured = pos.play_unchecked(opp_mv);
    add_move_obs(prev.side, opp_mv, captured, &pos, &mut logs);
    (pos, logs)
}

struct Extract {
    rows: Vec<Row>,
    games: u32,
    broken: u32,
    mismatched: u32,
    opponents: BTreeMap<String, u32>,
}

#[allow(clippy::too_many_lines)]
fn extract(
    files: &[PathBuf],
    opponent: Option<&str>,
    weights: &OriginWeights,
    adj_share: f64,
    firing: bool,
    digest: &mut sha2::Sha256,
) -> Extract {
    let mut out = Extract {
        rows: vec![],
        games: 0,
        broken: 0,
        mismatched: 0,
        opponents: BTreeMap::new(),
    };
    for path in files {
        let name = path.to_string_lossy().to_string();
        let Ok(content) = std::fs::read_to_string(path) else {
            out.broken += 1;
            continue;
        };
        sha2::Digest::update(digest, name.as_bytes());
        sha2::Digest::update(digest, content.as_bytes());
        let Some((bot, end)) = parse_bot_and_end(&content) else {
            out.broken += 1;
            continue;
        };
        *out.opponents.entry(end.opponent.username.clone()).or_insert(0) += 1;
        if opponent.is_some_and(|o| o != end.opponent.username) {
            out.mismatched += 1;
            continue;
        }
        let Some(snaps) = decision_snapshots(&end) else {
            out.broken += 1;
            continue;
        };
        let blind = blind_by_move(&content);
        let short = Path::new(&name)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or(name.clone());
        out.games += 1;

        for i in 0..snaps.len() {
            let s = &snaps[i];
            if s.side != bot || s.terminal {
                continue;
            }
            // 相手が指していない（bot の手で終局）なら次の bot 手番が無い = 打ち切り
            let (Some(next), Some(entry_snap)) = (snaps.get(i + 1), snaps.get(i + 2)) else {
                continue;
            };
            let Some(opp_mv) = end.moves.get(i + 1).and_then(|m| parse_usi(&m.usi)) else {
                continue;
            };
            let remaining = FOUL_LIMIT.saturating_sub(next.fouls[side_idx(bot)]);
            if remaining == 0 {
                continue; // 既に反則負けしている（次の手番に反則の余地が無い）
            }

            let pre_pieces = s.pos.pieces_of(bot);
            let handoff_pieces = next.pos.pieces_of(bot);
            let pre_f = features(&pre_pieces, bot, &known_enemy_squares(&s.logs[side_idx(bot)]));
            let known_handoff = known_enemy_squares(&next.logs[side_idx(bot)]);
            let hand_f = features(&handoff_pieces, bot, &known_handoff);
            let Some(king) = hand_f.king else { continue };

            let (entry_pos, entry_logs) = entry_state(next, &opp_mv);
            debug_assert_eq!(entry_pos.fingerprint(), entry_snap.pos.fingerprint());
            let in_check = entry_pos.in_check(bot);
            let entry_pieces = entry_pos.pieces_of(bot);
            let entry_f =
                features(&entry_pieces, bot, &known_enemy_squares(&entry_logs[side_idx(bot)]));

            let mut row = Row {
                game: short.clone(),
                decision_id: i as u64,
                move_number: s.pos.move_number(),
                gote: bot == Color::Gote,
                f1_cov_w: hand_f.f1.cov_weighted(weights),
                f1_cov_frac: hand_f.f1.cov_frac(),
                f1_total: hand_f.f1.total(),
                f1_cov: hand_f.f1.covered(),
                f1_cov_sure: hand_f.f1.covered_sure(),
                f1_by_class: [
                    (hand_f.f1.by_class[0].0, hand_f.f1.by_class[0].1),
                    (hand_f.f1.by_class[1].0, hand_f.f1.by_class[1].1),
                    (hand_f.f1.by_class[2].0, hand_f.f1.by_class[2].1),
                ],
                f1_by_dist: hand_f.f1.by_dist,
                d_f1_cov_w: hand_f.f1.cov_weighted(weights) - pre_f.f1.cov_weighted(weights),
                f2_open_dirs: hand_f.f2.open_dirs,
                f2_open_len: hand_f.f2.open_len,
                f2_flight_all: hand_f.f2.flight_all,
                f2_flight_shielded: hand_f.f2.flight_shielded,
                f2_king_home: hand_f.f2.king_home,
                f2_known_enemy_dist: hand_f.f2.known_enemy_to_origin,
                f3_open_uncovered: hand_f.f3.open_uncovered,
                f3_open_covered: hand_f.f3.open_covered,
                f3_own_occupied: hand_f.f3.own_occupied,
                adj_total: hand_f.f3.adj_total,
                adj_covered: hand_f.f3.adj_covered,
                f13_cov: hand_f.f13_cov(weights, adj_share),
                d_f3_open_uncovered: hand_f.f3.open_uncovered as i32
                    - pre_f.f3.open_uncovered as i32,
                e_f1_cov_w: in_check
                    .then(|| entry_f.f1.cov_weighted(weights) - hand_f.f1.cov_weighted(weights)),
                e_f3_open_uncovered: in_check.then(|| {
                    entry_f.f3.open_uncovered as i32 - hand_f.f3.open_uncovered as i32
                }),
                king_edge_dist: king_edge_dist(king),
                own_pieces: handoff_pieces.len() as u32,
                hand_count: next.pos.hand_map(bot).values().sum(),
                material_diff: material(&next.pos, bot) - material(&next.pos, bot.other()),
                king_density: handoff_pieces
                    .iter()
                    .filter_map(|p| parse_usi_square(&p.square))
                    .filter(|c| *c != king && cheb(*c, king) <= 2)
                    .count() as u32,
                remaining,
                next_is_check: in_check,
                fouls_in_check: if in_check { entry_snap.fouls_this_turn } else { 0 },
                terminal_turn: entry_snap.terminal,
                ..Row::default()
            };

            // 候補数（handoff の決定点。反則試行は除かない = 素の候補生成）
            let empty: HashSet<String> = HashSet::new();
            let dec_model = GameModel::from_log(bot, &s.logs[side_idx(bot)]);
            let dec_view =
                view_from_model(&dec_model, s.pos.in_check(bot), s.pos.move_number());
            row.candidates = candidate_moves(&dec_view, &empty).len() as u32;

            if in_check {
                row.k_def_truth = Some(k_def_truth(&entry_pos));
                let model = GameModel::from_log(bot, &entry_logs[side_idx(bot)]);
                let view = view_from_model(&model, true, entry_pos.move_number());
                let genc = candidate_moves(&view, &empty);
                row.k_def_gen =
                    Some(genc.iter().filter(|(_, mv)| entry_pos.is_legal(mv)).count() as u32);
                let rt = candidate_moves_with_log(&view, &empty, Some(&entry_logs[side_idx(bot)]));
                row.k_def_runtime =
                    Some(rt.iter().filter(|(_, mv)| entry_pos.is_legal(mv)).count() as u32);

                let checkers = true_checkers(&entry_pos, bot);
                row.double_check = Some(checkers.len() > 1);
                // 王手駒は複数ありうる。**最も近い**ものを代表列にし、
                // 「全部除去できるか」は oracle 側で手番単位に判定する
                if let Some((sq, role)) = checkers
                    .iter()
                    .copied()
                    .min_by_key(|(sq, _)| cheb(*sq, king))
                {
                    row.checker_role = Some(role);
                    row.checker_dist = Some(cheb(sq, king));
                    let origins = origin_squares(king, bot);
                    let o = origins.iter().find(|o| o.sq == sq);
                    row.origin_in_c = Some(o.is_some());
                    row.origin_class = o.map(|o| o.class);
                    row.origin_dist = o.map(|o| u32::from(o.dist));
                    let cov = tsuitate_bot::check_prep::coverage_of(&handoff_pieces, bot);
                    row.origin_covered = Some(checkers.iter().all(|(s, _)| cov.covered(*s)));
                    row.origin_covered_sure =
                        Some(checkers.iter().all(|(s, _)| cov.covered_sure(*s)));
                }
                row.checker_drop =
                    Some(matches!(opp_mv, ShogiMove::Drop { .. }));
                row.checker_capture = Some(match opp_mv {
                    ShogiMove::Board { to, .. } => next.pos.piece_at(to).is_some(),
                    ShogiMove::Drop { .. } => false,
                });
                if let Some(solver) = CheckSolver::new(&view, &[], &[], &entry_logs[side_idx(bot)])
                {
                    row.hyp_n = Some(solver.hypotheses_debug().len());
                    let (share, ent) = hypothesis_stats(Some(&solver), &entry_pos, bot);
                    row.true_hyp_share = share;
                    row.hyp_entropy = ent;
                }
                row.blind = blind.get(&entry_pos.move_number()).copied().flatten();
                row.oracle_nonking_capture = Some(nonking_capture_of_checker(&entry_pos, bot));

                if firing {
                    let (any, origin) = firing_potential(
                        &dec_view,
                        &s.logs[side_idx(bot)],
                        &pre_pieces,
                        bot,
                        weights,
                        hand_f.f1.cov_weighted(weights),
                        &checkers.iter().map(|(s, _)| *s).collect::<Vec<_>>(),
                    );
                    row.improvable_any = Some(any);
                    row.improvable_origin = Some(origin);
                }
            } else if firing {
                let (any, _) = firing_potential(
                    &dec_view,
                    &s.logs[side_idx(bot)],
                    &pre_pieces,
                    bot,
                    weights,
                    hand_f.f1.cov_weighted(weights),
                    &[],
                );
                row.improvable_any = Some(any);
            }
            out.rows.push(row);
        }
    }
    out
}

/// **発火母数**（`anchor_probe` の教訓）: その決定点の候補手の中に
/// 「主特徴量を実際に改善する手」「実際の王手起点を被覆する手」があったか。
///
/// 前者は「F がそもそも動かせるか」、後者は**後知恵の上限**（どこから来るかを
/// 知っていれば防げたか）。
fn firing_potential(
    view: &tsuitate_bot::protocol::PlayerView,
    log: &ObservationLog,
    pre_pieces: &[VisiblePiece],
    bot: Color,
    weights: &OriginWeights,
    played: f64,
    checkers: &[Coord],
) -> (bool, bool) {
    let empty: HashSet<String> = HashSet::new();
    let mut any = false;
    let mut origin = false;
    for (_, mv) in candidate_moves_with_log(view, &empty, Some(log)) {
        let after = pieces_after(pre_pieces, &mv);
        let f = features(&after, bot, &[]);
        if f.f1.cov_weighted(weights) > played + 1e-9 {
            any = true;
        }
        if !checkers.is_empty() && !origin {
            let cov = tsuitate_bot::check_prep::coverage_of(&after, bot);
            if checkers.iter().all(|c| cov.covered(*c)) {
                origin = true;
            }
        }
        if any && (origin || checkers.is_empty()) {
            break;
        }
    }
    (any, origin)
}

// ---------------------------------------------------------------------------
// 集計
// ---------------------------------------------------------------------------

/// 局ごとに束ねる（cluster bootstrap の単位は元対局）
fn clusters(rows: &[Row]) -> Vec<Vec<Row>> {
    let mut by: BTreeMap<String, Vec<Row>> = BTreeMap::new();
    for r in rows {
        by.entry(r.game.clone()).or_default().push(r.clone());
    }
    by.into_values().collect()
}

fn ratio(num: u32, den: u32) -> f64 {
    if den == 0 { f64::NAN } else { f64::from(num) / f64::from(den) }
}

/// **実王手起点の分布**（距離重み・`adj_share` を決めるための発見セットの材料）。
///
/// issue #34 の手順は「距離重みは事前に置かず、**発見セットの実王手起点の
/// 距離分布**から決めて検証セットでは固定する」なので、その分布をここで出す。
fn report_origin_distribution(rows: &[Row]) {
    let checked: Vec<&Row> = rows.iter().filter(|r| r.next_is_check).collect();
    if checked.is_empty() {
        return;
    }
    println!("\n## 実王手起点の分布（距離重みと adj_share を決める材料）");
    let adjacent = checked.iter().filter(|r| r.origin_in_c == Some(false)).count();
    println!(
        "  隣接（距離1 = F3 の領分）{}/{} ({:.1}%) / C(K)（距離2以上 = F1 の領分）{} ({:.1}%)",
        adjacent,
        checked.len(),
        ratio(adjacent as u32, checked.len() as u32) * 100.0,
        checked.len() - adjacent,
        ratio((checked.len() - adjacent) as u32, checked.len() as u32) * 100.0,
    );
    let mut by_class: BTreeMap<&str, u32> = BTreeMap::new();
    let mut by_dist: BTreeMap<u32, u32> = BTreeMap::new();
    for r in &checked {
        if let Some(c) = r.origin_class {
            *by_class.entry(c.label()).or_insert(0) += 1;
        }
        if let Some(d) = r.origin_dist {
            *by_dist.entry(d).or_insert(0) += 1;
        }
    }
    println!("  C(K) 側のクラス別: {by_class:?} / 距離別: {by_dist:?}");
    let drops = checked.iter().filter(|r| r.checker_drop == Some(true)).count();
    println!(
        "  打ち王手 {}/{} ({:.1}%) / 捕獲つき王手 {} / 両王手 {}",
        drops,
        checked.len(),
        ratio(drops as u32, checked.len() as u32) * 100.0,
        checked.iter().filter(|r| r.checker_capture == Some(true)).count(),
        checked.iter().filter(|r| r.double_check == Some(true)).count(),
    );
    println!(
        "  → **この分布が主特徴量の形を決める**。隣接が多数なら、F1（距離2以上）単独では\n     実王手起点の大半を分母に持たない（`f13_cov` の `--adj-share` を実測値で固定して検証セットへ送る）"
    );
}

/// **K の3定義の突き合わせ**（`K_def_truth` が多く `K_def_runtime` が少ないなら
/// 律速は形でなく候補生成、という切り分け）
fn report_k_defs(rows: &[Row]) {
    let checked: Vec<&Row> = rows.iter().filter(|r| r.next_is_check).collect();
    if checked.is_empty() {
        return;
    }
    let gen_lt = checked
        .iter()
        .filter(|r| matches!((r.k_def_gen, r.k_def_truth), (Some(g), Some(t)) if g < t))
        .count();
    let rt_lt = checked
        .iter()
        .filter(|r| matches!((r.k_def_runtime, r.k_def_gen), (Some(a), Some(b)) if a < b))
        .count();
    println!(
        "  K の突き合わせ: 候補生成が真の解消手を取りこぼした手番 {gen_lt}/{} / runtime 生成が診断側より少なかった手番 {rt_lt}（**多いなら律速は形でなく候補生成**）",
        checked.len()
    );
}

/// 段1: **構成概念検証**。F1 の起点別被覆が oracle（実際の王手駒を玉以外で
/// 真に合法に取れたか）をどれだけ再現するか。ここで落ちたら次へ進まない
fn report_construct(rows: &[Row], cl: &[Vec<Row>]) -> bool {
    let checked: Vec<&Row> = rows.iter().filter(|r| r.next_is_check).collect();
    println!("\n## 段1: 構成概念検証（王手起点の被覆 vs oracle）");
    if checked.is_empty() {
        println!("  被王手の事例がありません");
        return false;
    }
    let mut tab = [[0u32; 2]; 2]; // [被覆][oracle]
    for r in &checked {
        let (Some(c), Some(o)) = (r.origin_covered, r.oracle_nonking_capture) else { continue };
        tab[usize::from(c)][usize::from(o)] += 1;
    }
    let n: u32 = tab.iter().flatten().sum();
    println!(
        "  被王手 {} 件 / oracle 陽性（玉以外で王手駒を真に合法に取れた）{} 件 ({:.1}%)",
        checked.len(),
        tab[0][1] + tab[1][1],
        ratio(tab[0][1] + tab[1][1], n) * 100.0,
    );
    // **適合率が主指標**。「被覆なし → oracle 陰性」はほぼ論理的帰結（被覆は
    // 真の利きの楽観上限なので、被覆が無ければ真の取り手も無い）で、
    // handoff から entry の間に王手手が形を壊した場合にだけ例外が出る。
    // したがってリフトは下駄を履いており、実質の中身は適合率のほう
    let prec = |rs: &[&Row], sure: bool| -> Option<f64> {
        let (mut n, mut k) = (0u32, 0u32);
        for r in rs.iter().filter(|r| r.next_is_check) {
            let flag = if sure { r.origin_covered_sure } else { r.origin_covered };
            let (Some(true), Some(o)) = (flag, r.oracle_nonking_capture) else { continue };
            n += 1;
            k += u32::from(o);
        }
        (n > 0).then(|| f64::from(k) / f64::from(n))
    };
    let refs: Vec<&Row> = rows.iter().collect();
    let sure_n = checked.iter().filter(|r| r.origin_covered_sure == Some(true)).count();
    println!(
        "  被覆あり {} 件（うち確定利き {sure_n} 件）→ oracle 陽性 {} / 被覆なし {} 件 → oracle 陽性 {}",
        tab[1][0] + tab[1][1],
        tab[1][1],
        tab[0][0] + tab[0][1],
        tab[0][1],
    );
    println!(
        "  注: 「被覆なし → oracle 陰性」はほぼ**論理的帰結**（被覆は真の利きの楽観上限）なので、\n     リフトではなく**適合率**が構成概念検証の中身。例外は handoff から entry の間に王手手が形を壊した場合"
    );
    for (label, sure) in [("被覆あり（確定＋レイ）", false), ("確定利きのみ", true)] {
        match prec(&refs, sure) {
            Some(v) => {
                let (lo, hi) =
                    cluster_bootstrap(cl, |rs| prec(rs, sure), 0.05, 0x2034_0001 + u64::from(sure));
                println!("  適合率 {label}: {:.3} [95% CI {lo:.3}, {hi:.3}]", v);
            }
            None => println!("  適合率 {label}: 判定不能（該当なし）"),
        }
    }
    let lift = |rs: &[&Row]| -> Option<f64> {
        let (mut on, mut off) = ((0u32, 0u32), (0u32, 0u32));
        for r in rs.iter().filter(|r| r.next_is_check) {
            let (Some(c), Some(o)) = (r.origin_covered, r.oracle_nonking_capture) else {
                continue;
            };
            let s = if c { &mut on } else { &mut off };
            s.0 += 1;
            s.1 += u32::from(o);
        }
        (on.0 > 0 && off.0 > 0)
            .then(|| f64::from(on.1) / f64::from(on.0) - f64::from(off.1) / f64::from(off.0))
    };
    let (Some(p), Some(l)) = (prec(&refs, false), lift(&refs)) else {
        println!("  → 構成概念検証: **判定不能**（片側が空。反則回帰へ進まない）");
        return false;
    };
    let (plo, _) = cluster_bootstrap(cl, |rs| prec(rs, false), 0.05, 0x2034_0001);
    let (llo, lhi) = cluster_bootstrap(cl, |rs| lift(rs), 0.05, 0x2034_0011);
    println!("  リフト（参考）: {l:+.3} [95% CI {llo:+.3}, {lhi:+.3}]");
    // 事前登録の門: 適合率の CI 下限 > 0.5 かつ リフトの CI 下限 > 0
    let pass = plo > 0.5 && llo > 0.0;
    println!(
        "  → 構成概念検証（門: 適合率の CI 下限 > 0.5 かつ リフトの CI 下限 > 0）: {}",
        if pass {
            format!("通過（適合率 {p:.3}・下限 {plo:.3}）")
        } else {
            format!("**不通過**（適合率 {p:.3}・下限 {plo:.3} / リフト下限 {llo:+.3}）→ 反則回帰へ進まない")
        }
    );
    pass
}

/// 段2: **発火母数**。改善可能でなければ、相関があっても項にできない
fn report_firing(rows: &[Row], cl: &[Vec<Row>]) {
    println!("\n## 段2: 発火母数");
    let fouled: Vec<&Row> = rows.iter().filter(|r| r.fouls_in_check > 0).collect();
    if fouled.is_empty() || fouled.iter().all(|r| r.improvable_any.is_none()) {
        println!("  （--no-firing で計算していないか、王手中反則の事例がありません）");
        return;
    }
    let frac = |rs: &[&Row], f: fn(&Row) -> Option<bool>| -> Option<f64> {
        let (mut n, mut k) = (0u32, 0u32);
        for r in rs.iter().filter(|r| r.fouls_in_check > 0) {
            if let Some(v) = f(r) {
                n += 1;
                k += u32::from(v);
            }
        }
        (n > 0).then(|| f64::from(k) / f64::from(n))
    };
    let refs: Vec<&Row> = rows.iter().collect();
    for (name, f) in [
        ("主特徴量を改善する候補があった", (|r: &Row| r.improvable_any) as fn(&Row) -> Option<bool>),
        ("実際の王手起点を被覆する候補があった（後知恵の上限）", |r: &Row| r.improvable_origin),
    ] {
        match frac(&refs, f) {
            Some(v) => {
                let (lo, hi) = cluster_bootstrap(cl, |rs| frac(rs, f), 0.05, 0x2034_0002);
                println!(
                    "  王手中反則のうち {name}: {:.1}% [95% CI {:.1}, {:.1}]（門: 20% 未満なら中止）",
                    v * 100.0,
                    lo * 100.0,
                    hi * 100.0
                );
            }
            None => println!("  {name}: 判定不能"),
        }
    }
    let movable = rows.iter().filter(|r| r.improvable_any == Some(true)).count();
    let known = rows.iter().filter(|r| r.improvable_any.is_some()).count();
    println!(
        "  参考: 全 handoff のうち主特徴量を動かせた決定点 {}/{} ({:.1}%)",
        movable,
        known,
        ratio(movable as u32, known as u32) * 100.0
    );
}

/// 3段の分解（①王手になる確率 ②K ③反則 0/1/2+）。回帰の前に生の分布を出す
fn report_decomposition(rows: &[Row]) {
    println!("\n## 分解（主 estimand の前に生の分布）");
    let n = rows.len() as u32;
    let checked: Vec<&Row> = rows.iter().filter(|r| r.next_is_check).collect();
    println!(
        "  ① 次の相手手が王手になった handoff: {}/{n} ({:.1}%)",
        checked.len(),
        ratio(checked.len() as u32, n) * 100.0
    );
    let mean = |f: &dyn Fn(&Row) -> Option<u32>| -> f64 {
        let vals: Vec<u32> = checked.iter().filter_map(|r| f(r)).collect();
        if vals.is_empty() {
            f64::NAN
        } else {
            vals.iter().sum::<u32>() as f64 / vals.len() as f64
        }
    };
    println!(
        "  ② 王手された場合の出口: K_def_truth 平均 {:.2} / K_def_gen {:.2} / K_def_runtime {:.2}",
        mean(&|r| r.k_def_truth),
        mean(&|r| r.k_def_gen),
        mean(&|r| r.k_def_runtime),
    );
    let le1 = checked.iter().filter(|r| r.k_def_truth.is_some_and(|k| k <= 1)).count();
    println!(
        "     真に合法な解消手 ≤1本の手番: {}/{} ({:.1}%)（**反則した手番に条件づけない分布**）",
        le1,
        checked.len(),
        ratio(le1 as u32, checked.len() as u32) * 100.0
    );
    let mut hist = [0u32; 3];
    for r in &checked {
        hist[(r.fouls_in_check.min(2)) as usize] += 1;
    }
    println!(
        "  ③ 王手された場合の反則: 0回 {} / 1回 {} / 2回以上 {}",
        hist[0], hist[1], hist[2]
    );
    let cens = checked.iter().filter(|r| r.fouls_in_check >= r.remaining).count();
    println!(
        "     残り反則で打ち切られた（`y == 残り`）事例: {cens} 件（第2段は上側確率で扱う）"
    );
}

/// 主 estimand と secondary（Holm 補正）
fn report_estimand(rows: &[Row], cl: &[Vec<Row>], boots: usize) {
    println!("\n## 段3: 主 estimand（層を固定した Q90 − Q10 の周辺平均差）");
    let feats: Vec<(&str, fn(&Row) -> f64)> = vec![
        ("f1_cov_w（主）", |r: &Row| r.f1_cov_w),
        ("f1_cov_sure/total", |r: &Row| {
            if r.f1_total == 0 { 0.0 } else { f64::from(r.f1_cov_sure) / f64::from(r.f1_total) }
        }),
        ("adj_cov_frac", |r: &Row| {
            if r.adj_total == 0 { 0.0 } else { f64::from(r.adj_covered) / f64::from(r.adj_total) }
        }),
        ("f13_cov（F1 と隣接の混合）", |r: &Row| r.f13_cov),
        ("f3_open_uncovered", |r: &Row| f64::from(r.f3_open_uncovered)),
        ("f3_own_occupied", |r: &Row| f64::from(r.f3_own_occupied)),
        ("f2_open_len", |r: &Row| f64::from(r.f2_open_len)),
        ("f2_flight_all", |r: &Row| f64::from(r.f2_flight_all)),
    ];
    let mut results: Vec<(String, f64, f64, f64, f64)> = vec![];
    for (name, f) in &feats {
        let build = |rs: &[&Row]| -> Vec<hurdle::Row> {
            rs.iter()
                .map(|r| hurdle::Row {
                    x: f(r),
                    z: covariates(r),
                    y: r.fouls_in_check.min(r.remaining),
                    cap: r.remaining,
                })
                .collect()
        };
        let refs: Vec<&Row> = rows.iter().collect();
        let all = build(&refs);
        let mut xs: Vec<f64> = all.iter().map(|r| r.x).collect();
        let lo_q = hurdle::quantile(&mut xs.clone(), 0.1);
        let hi_q = hurdle::quantile(&mut xs, 0.9);
        if (hi_q - lo_q).abs() < 1e-12 {
            println!("  {name}: Q10 と Q90 が同じ値（変動が無い）ので判定不能");
            continue;
        }
        let Some(point) = hurdle::contrast(&all, lo_q, hi_q, 1e-4) else {
            println!("  {name}: 当てはめが収束しませんでした");
            continue;
        };
        let stat = |rs: &[&Row]| hurdle::contrast(&build(rs), lo_q, hi_q, 1e-4);
        let (blo, bhi) = bootstrap_n(cl, stat, 0.05, 0x2034_0003, boots);
        // 両側のブートストラップ p 値（0 を跨ぐ側の割合の2倍）
        let p = boot_p(cl, stat, 0x2034_0003, boots);
        results.push((format!("{name}"), point, blo, bhi, p));
    }
    // Holm 補正（主特徴量は事前登録の1本なので補正の対象は secondary）
    let mut order: Vec<usize> = (1..results.len()).collect();
    order.sort_by(|a, b| results[*a].4.total_cmp(&results[*b].4));
    let m = order.len();
    let mut adj = vec![0.0; results.len()];
    let mut running: f64 = 0.0;
    for (rank, idx) in order.iter().enumerate() {
        running = running.max(results[*idx].4 * (m - rank) as f64);
        adj[*idx] = running.min(1.0);
    }
    for (i, (name, point, lo, hi, p)) in results.iter().enumerate() {
        if i == 0 {
            println!(
                "  **{name}**: {point:+.4} 反則/手番 [95% CI {lo:+.4}, {hi:+.4}] / p={p:.4}（事前登録の主 estimand。補正なし）"
            );
        } else {
            println!(
                "  {name}: {point:+.4} [95% CI {lo:+.4}, {hi:+.4}] / p={p:.4} → Holm 補正後 p={:.4}",
                adj[i]
            );
        }
    }
    println!(
        "  （**符号の向きは事前に仮定しない**。指導2「逃げる」と指導3「固める」は逆向きに働きうるので、\n    発見セットで符号を確かめてから検証セットで固定する）"
    );
}

/// 層・共変量（総効果の回帰で調整する。**`K_def_*` と機構列は入れない** =
/// 狙っている媒介変数と結果側の変数だから）
fn covariates(r: &Row) -> Vec<f64> {
    vec![
        f64::from(r.move_number),
        f64::from(r.king_edge_dist),
        f64::from(r.own_pieces),
        f64::from(r.hand_count),
        r.material_diff,
        f64::from(r.king_density),
        f64::from(r.remaining),
        f64::from(r.candidates),
        f64::from(u8::from(r.gote)),
    ]
}

fn bootstrap_n(
    cl: &[Vec<Row>],
    stat: impl Fn(&[&Row]) -> Option<f64>,
    alpha: f64,
    seed: u64,
    reps: usize,
) -> (f64, f64) {
    let draws = boot_draws(cl, stat, seed, reps);
    if draws.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let lo = ((draws.len() as f64) * (alpha / 2.0)) as usize;
    let hi = (((draws.len() as f64) * (1.0 - alpha / 2.0)) as usize).min(draws.len() - 1);
    (draws[lo], draws[hi])
}

fn boot_draws(
    cl: &[Vec<Row>],
    stat: impl Fn(&[&Row]) -> Option<f64>,
    seed: u64,
    reps: usize,
) -> Vec<f64> {
    let mut state = seed | 1;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let mut draws = vec![];
    let mut buf: Vec<&Row> = vec![];
    for _ in 0..reps {
        buf.clear();
        for _ in 0..cl.len() {
            buf.extend(cl[next() % cl.len()].iter());
        }
        if let Some(v) = stat(&buf) {
            if v.is_finite() {
                draws.push(v);
            }
        }
    }
    draws.sort_by(|a, b| a.total_cmp(b));
    draws
}

fn boot_p(
    cl: &[Vec<Row>],
    stat: impl Fn(&[&Row]) -> Option<f64>,
    seed: u64,
    reps: usize,
) -> f64 {
    let draws = boot_draws(cl, stat, seed, reps);
    if draws.is_empty() {
        return 1.0;
    }
    let below = draws.iter().filter(|v| **v <= 0.0).count() as f64 / draws.len() as f64;
    (2.0 * below.min(1.0 - below)).min(1.0)
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("report") {
        report_mode(&args[1..]);
        return;
    }
    let mut out_csv: Option<String> = None;
    let mut shard: Option<(usize, usize)> = None;
    let mut opponent: Option<String> = None;
    let mut weights = OriginWeights::UNIFORM;
    let mut weights_spec = String::new();
    let mut adj_share = 0.5f64;
    let mut limit = usize::MAX;
    let mut firing = true;
    let mut boots = 500usize;
    let mut specs: Vec<String> = vec![];
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                out_csv = Some(args.get(i + 1).cloned().unwrap_or_else(|| die("--out <path>")));
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
                    args.get(i + 1).cloned().unwrap_or_else(|| die("--opponent <戦略名>")),
                );
                i += 2;
            }
            "--origin-weights" => {
                weights_spec =
                    args.get(i + 1).cloned().unwrap_or_else(|| die("--origin-weights <spec>"));
                weights = OriginWeights::parse(&weights_spec)
                    .unwrap_or_else(|| die("--origin-weights の書式が不正です"));
                i += 2;
            }
            "--adj-share" => {
                adj_share = args
                    .get(i + 1)
                    .and_then(|v| v.parse::<f64>().ok())
                    .filter(|v| (0.0..=1.0).contains(v))
                    .unwrap_or_else(|| die("--adj-share は 0〜1"));
                i += 2;
            }
            "--limit" => {
                limit = args
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| die("--limit <N>"));
                i += 2;
            }
            "--bootstrap" => {
                boots = args
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| die("--bootstrap <N>"));
                i += 2;
            }
            "--no-firing" => {
                firing = false;
                i += 1;
            }
            other if other.starts_with("--") => die(&format!("不明なオプション: {other}")),
            other => {
                specs.push(other.to_string());
                i += 1;
            }
        }
    }
    if specs.is_empty() {
        die("記録（records/*.jsonl）を渡してください");
    }
    let mut files = collect_records(&specs);
    // シャードは**元対局単位**で割る（決定点で割ると同じ局が複数シャードに散り、
    // cluster bootstrap の単位が壊れる）
    if let Some((a, b)) = shard {
        files = files.into_iter().enumerate().filter(|(i, _)| i % b == a).map(|(_, f)| f).collect();
    }
    files.truncate(limit);
    if files.is_empty() {
        die("記録が1つも見つかりません");
    }

    use sha2::Digest as _;
    let mut digest = sha2::Sha256::new();
    let started = std::time::Instant::now();
    let ex = extract(&files, opponent.as_deref(), &weights, adj_share, firing, &mut digest);
    let fingerprint: String = digest.finalize().iter().map(|b| format!("{b:02x}")).collect();
    eprintln!(
        "{} 局 / {} handoff / {:.1}秒（壊れた記録 {} / 相手不一致で除外 {}）",
        ex.games,
        ex.rows.len(),
        started.elapsed().as_secs_f64(),
        ex.broken,
        ex.mismatched,
    );
    if ex.opponents.len() > 1 {
        eprintln!("  記録の相手: {:?}（--opponent で絞ること）", ex.opponents);
    }
    if ex.rows.is_empty() {
        die("handoff が1つも取れませんでした");
    }
    let (shard_i, shard_n) = shard.unwrap_or((0, 1));
    let meta = serde_json::json!({
        "schema": ROW_SCHEMA,
        "opponent": opponent.clone().unwrap_or_default(),
        "shard": shard_i,
        "shards": shard_n,
        "origin_weights": weights_spec,
        "adj_share": adj_share,
        "firing": firing,
        "source_fingerprint": env!("TSUITATE_SOURCE_FINGERPRINT"),
        "records": fingerprint,
        "games": ex.games,
        "rows": ex.rows.len(),
    });
    if let Some(path) = &out_csv {
        let mut s = format!("#meta {meta}\n{HEADER}\n");
        for r in &ex.rows {
            s.push_str(&r.to_csv());
            s.push('\n');
        }
        if let Some(dir) = Path::new(path).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        std::fs::write(path, s).unwrap_or_else(|e| die(&format!("{path} が書けません: {e}")));
        eprintln!("CSV: {path}");
    }
    if shard_n > 1 {
        println!(
            "\n**シャード {shard_i}/{shard_n} の部分集計**（判定は `check_prep_probe report` で全シャードを集めてから）"
        );
    }
    run_report(&ex.rows, boots);
}

fn run_report(rows: &[Row], boots: usize) {
    let cl = clusters(rows);
    println!(
        "\n=== 被王手の前の準備 P0-2（issue #34）: handoff {} 件 / 元対局 {} 局 ===",
        rows.len(),
        cl.len()
    );
    report_decomposition(rows);
    report_k_defs(rows);
    report_origin_distribution(rows);
    let ok = report_construct(rows, &cl);
    report_firing(rows, &cl);
    if ok {
        report_estimand(rows, &cl, boots);
    } else {
        println!(
            "\n## 段3: 主 estimand — **走らせない**（構成概念検証が不通過。issue #34 の中止条件）"
        );
    }
}

fn report_mode(args: &[String]) {
    let mut allow_incomplete = false;
    let mut boots = 500usize;
    let mut paths: Vec<String> = vec![];
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--allow-incomplete" => {
                allow_incomplete = true;
                i += 1;
            }
            "--bootstrap" => {
                boots = args
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| die("--bootstrap <N>"));
                i += 2;
            }
            other if other.starts_with("--") => die(&format!("不明なオプション: {other}")),
            other => {
                paths.push(other.to_string());
                i += 1;
            }
        }
    }
    if paths.is_empty() {
        die("CSV を渡してください");
    }
    let mut rows: Vec<Row> = vec![];
    let mut metas: Vec<serde_json::Value> = vec![];
    let mut seen_shards: Vec<usize> = vec![];
    for path in &paths {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| die(&format!("{path} が読めません: {e}")));
        let mut lines = text.lines();
        let meta: serde_json::Value = lines
            .next()
            .and_then(|l| l.strip_prefix("#meta "))
            .and_then(|j| serde_json::from_str(j).ok())
            .unwrap_or_else(|| die(&format!("{path}: meta 行がありません")));
        if meta["schema"].as_u64() != Some(u64::from(ROW_SCHEMA)) {
            die(&format!(
                "{path}: schema {} は集計できません（現行 {ROW_SCHEMA}）",
                meta["schema"]
            ));
        }
        seen_shards.push(meta["shard"].as_u64().unwrap_or(0) as usize);
        metas.push(meta);
        let header = lines.next().unwrap_or_default();
        let cols: Vec<&str> = header.split(',').collect();
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let v: Vec<&str> = line.split(',').collect();
            rows.push(Row::from_csv(path, &cols, &v));
        }
    }
    // **同じ実験の独立サンプルか**を検査する（別実験の混入と重複は失敗）
    let key = |m: &serde_json::Value| {
        (
            m["opponent"].clone(),
            m["origin_weights"].clone(),
            m["adj_share"].clone(),
            m["source_fingerprint"].clone(),
            m["shards"].clone(),
            m["firing"].clone(),
        )
    };
    if metas.windows(2).any(|w| key(&w[0]) != key(&w[1])) {
        die("meta が食い違う CSV が混ざっています（相手 / 起点重み / コード版 / シャード数 / firing を揃えること）");
    }
    let shards = metas[0]["shards"].as_u64().unwrap_or(1) as usize;
    seen_shards.sort_unstable();
    seen_shards.dedup();
    if seen_shards.len() != shards {
        let msg = format!(
            "シャードが揃っていません（期待 {shards} / 実際 {:?}）。**元対局が欠けると分母が狂う**ので判定を出さない",
            seen_shards
        );
        if allow_incomplete {
            eprintln!("警告: {msg}");
        } else {
            die(&msg);
        }
    }
    let mut ids: Vec<(String, u64)> =
        rows.iter().map(|r| (r.game.clone(), r.decision_id)).collect();
    ids.sort();
    let before = ids.len();
    ids.dedup();
    if ids.len() != before {
        die("同じ (元対局, 決定点) の行が重複しています（同じシャードを2回渡していませんか）");
    }
    println!("入力 {} 本 / 相手 {}", paths.len(), metas[0]["opponent"]);
    run_report(&rows, boots);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(f1: f64, y: u32, check: bool) -> Row {
        Row {
            game: "g".into(),
            f1_cov_w: f1,
            f1_cov_frac: f1,
            remaining: 10,
            next_is_check: check,
            fouls_in_check: y,
            ..Row::default()
        }
    }

    #[test]
    fn csvの往復で値が保たれる() {
        let mut r = row(0.25, 2, true);
        r.game = "rec-1".into();
        r.decision_id = 7;
        r.move_number = 33;
        r.k_def_truth = Some(4);
        r.origin_class = Some(OriginClass::Diag);
        r.origin_dist = Some(3);
        r.origin_covered = Some(true);
        r.oracle_nonking_capture = Some(false);
        r.improvable_any = Some(true);
        r.e_f3_open_uncovered = Some(-2);
        let line = r.to_csv();
        let cols: Vec<&str> = HEADER.split(',').collect();
        let vals: Vec<&str> = line.split(',').collect();
        assert_eq!(cols.len(), vals.len(), "ヘッダと列数が一致する");
        let back = Row::from_csv("t", &cols, &vals);
        assert_eq!(back.game, "rec-1");
        assert_eq!(back.decision_id, 7);
        assert_eq!(back.move_number, 33);
        assert!((back.f1_cov_w - 0.25).abs() < 1e-4);
        assert_eq!(back.k_def_truth, Some(4));
        assert_eq!(back.origin_class, Some(OriginClass::Diag));
        assert_eq!(back.origin_dist, Some(3));
        assert_eq!(back.origin_covered, Some(true));
        assert_eq!(back.oracle_nonking_capture, Some(false));
        assert_eq!(back.improvable_any, Some(true));
        assert_eq!(back.e_f3_open_uncovered, Some(-2));
        assert_eq!(back.fouls_in_check, 2);
    }

    #[test]
    fn 王手されなかったhandoffの機構列は欠測のまま() {
        let r = row(0.4, 0, false);
        let line = r.to_csv();
        let cols: Vec<&str> = HEADER.split(',').collect();
        let vals: Vec<&str> = line.split(',').collect();
        let back = Row::from_csv("t", &cols, &vals);
        assert!(back.k_def_truth.is_none() && back.origin_covered.is_none());
        assert!(!back.next_is_check && back.fouls_in_check == 0);
    }

    #[test]
    fn clusterは元対局で束ねる() {
        let mut a = row(0.1, 0, false);
        a.game = "g1".into();
        let mut b = row(0.2, 1, true);
        b.game = "g1".into();
        b.decision_id = 1;
        let mut c = row(0.3, 0, false);
        c.game = "g2".into();
        let cl = clusters(&[a, b, c]);
        assert_eq!(cl.len(), 2);
        assert_eq!(cl[0].len(), 2);
    }
}
