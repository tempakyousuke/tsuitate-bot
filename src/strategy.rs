//! 指し手の選択。
//!
//! `Strategy` trait の実装を差し替えて強さを比較する（bin/arena.rs で対戦できる）。
//! - `Heuristic`: サイト内蔵の簡易botと同じ「前進を好むヒューリスティック＋乱数」
//! - `EstimatorStrategy`: 観測履歴から相手局面の粒子集合を維持し（estimator.rs）、
//!   候補手を粒子平均で評価する

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use rand::Rng;

use crate::board::{
    Coord, Promotion, defend_targets, drop_targets, make_usi_drop, make_usi_move, make_usi_square,
    move_targets, parse_usi_square, promotion_choice,
};
use crate::check::CheckSolver;
use crate::estimator::{EPS_INFO, Estimator, opp_reply_weights};
use crate::likelihood::{FITTED_THETA, ParticleCtx, particle_features, particle_log_weight};
use crate::model::GameModel;
use crate::observation::{Observation, ObservationLog};
use crate::opening::OpeningBook;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::protocol::{Color, PlayerView, Role, VisiblePiece};
use crate::shogi::{Position, ShogiMove, parse_usi, piece_value, promote_role, unpromote_role};

/// 1インスタンス = 1対局。対局開始ごとに `make` で作り直す。
pub trait Strategy {
    /// 自分の手番で呼ばれる。foul_tried の手は除外すること。
    /// None を返したら投了（指せる手がない）。
    fn choose(
        &mut self,
        view: &PlayerView,
        log: &ObservationLog,
        foul_tried: &HashSet<String>,
    ) -> Option<String>;

    fn name(&self) -> &'static str;

    /// 直近の choose 時点の内部状態（対局記録のデバッグ用）。推定系のみ実装する
    fn debug_state(&self) -> Option<serde_json::Value> {
        None
    }

    /// 直近の choose 時点の全候補評価（スコア降順）。scenario-gui のデバッグ表示用。
    /// 現行 estimator のみ実装する（凍結版は編集しないので既定 None のまま）。
    /// 定跡で指した手番・候補ゼロの手番は None
    fn last_ranking(&self) -> Option<&[CandidateScore]> {
        None
    }

    /// 観測ログを内部推定器に先行反映する（候補評価はしない）。
    /// 実対局では choose が自分の手番ごとに呼ばれて推定器が逐次更新される
    /// （リプレイ予算も手番ごとに与えられる）。局面再現実験（bin/scenario）が
    /// 履歴の途中時点の update を再現するために使う。既定は何もしない
    fn prewarm(&mut self, _view: &PlayerView, _log: &ObservationLog) {}

    /// 現在の内部状態ごと複製する（bin/scenario のバッチ実行が、prewarm 済みの
    /// 推定器を「決定点ごとのスナップショット」として使い回すため）。
    /// 対応しない戦略は None（凍結版は編集しないので既定 None のまま。
    /// その場合バッチ実行は従来どおり毎回作り直しにフォールバックする）
    fn clone_boxed(&self) -> Option<Box<dyn Strategy>> {
        None
    }
}

/// 蓄積済みの観測ログを「自分の手番ごとの逐次 update」で戦略に温めさせる。
/// 一括 update だとリプレイ予算が1回分しか与えられず、長い履歴では粒子が
/// 完全枯渇する（kakunari の69手を一括で食わせるとユニーク粒子0になる）。
/// bin/scenario.rs（棋譜の途中局面再現）と webhook_session.rs
/// （プロセス再起動後などのコールドスタート復元）の両方が使う共通部品
pub fn prewarm_strategy(strat: &mut dyn Strategy, view: &PlayerView, full: &ObservationLog) {
    prewarm_strategy_with_budget(strat, view, full, None);
}

/// 履歴を逐次prewarmするが、HTTP webhookのコールドスタートなどでは時間上限を
/// 設ける。途中で打ち切っても、次の choose が残りのイベントを通常updateする。
pub fn prewarm_strategy_with_budget(
    strat: &mut dyn Strategy,
    view: &PlayerView,
    full: &ObservationLog,
    max_duration: Option<Duration>,
) {
    let deadline = max_duration.map(|d| Instant::now() + d);
    let mut running = ObservationLog::default();
    for e in full.events() {
        if matches!(e, Observation::MyMove { .. } | Observation::MyFoul { .. }) {
            if deadline.is_some_and(|d| Instant::now() >= d) {
                break;
            }
            strat.prewarm(view, &running);
        }
        running.record(e.clone());
    }
}

pub const DEFAULT_STRATEGY: &str = "estimator";

/// 戦略名からインスタンスを作る。未知の名前は None。
/// `estimator_vN` はアリーナ比較用の凍結版（src/frozen/）
/// シード付きで戦略を作る（SPSA の f+/f− 評価で対局条件を揃える共通乱数法用）。
/// シード注入に対応していない戦略は通常の make にフォールバックする
/// （その場合、その戦略側の乱数はペアリングされない）
pub fn make_seeded(name: &str, seed: u64) -> Option<Box<dyn Strategy + Send>> {
    match name {
        "estimator" => Some(Box::new(EstimatorStrategy::with_params_line_seed(
            EvalParams::default(),
            None,
            Some(seed),
        ))),
        "estimator_rush" => {
            let idx = OpeningBook::line_index("居飛車速攻")?;
            Some(Box::new(EstimatorStrategy::with_params_line_seed(
                EvalParams::default(),
                Some(idx),
                Some(seed),
            )))
        }
        "estimator_v6" => Some(Box::new(
            crate::frozen::estimator_v6::EstimatorV6::with_seed(seed),
        )),
        "estimator_v7" => Some(Box::new(
            crate::frozen::estimator_v7::EstimatorV7::with_seed(seed),
        )),
        "estimator_v8" => Some(Box::new(
            crate::frozen::estimator_v8::EstimatorV8::with_seed(seed),
        )),
        "estimator_v9" => Some(Box::new(
            crate::frozen::estimator_v9::EstimatorV9::with_seed(seed),
        )),
        "estimator_v10" => Some(Box::new(
            crate::frozen::estimator_v10::EstimatorV10::with_seed(seed),
        )),
        "estimator_v11" => Some(Box::new(
            crate::frozen::estimator_v11::EstimatorV11::with_seed(seed),
        )),
        "estimator_v12" => Some(Box::new(
            crate::frozen::estimator_v12::EstimatorV12::with_seed(seed),
        )),
        "estimator_v13" => Some(Box::new(
            crate::frozen::estimator_v13::EstimatorV13::with_seed(seed),
        )),
        _ => make(name),
    }
}

pub fn make(name: &str) -> Option<Box<dyn Strategy + Send>> {
    match name {
        "heuristic" => Some(Box::new(Heuristic)),
        "estimator" => Some(Box::new(EstimatorStrategy::new())),
        // Claude（対話セッション）が直接指す実験用（bridge.rs）。アリーナでは使わない
        "bridge" => Some(Box::new(crate::bridge::FileBridge::new())),
        // 定跡特化チューニングの基準用: 居飛車速攻ラインだけを指す現行estimator
        "estimator_rush" => {
            let idx = OpeningBook::line_index("居飛車速攻")?;
            Some(Box::new(EstimatorStrategy::with_params_and_line(
                EvalParams::default(),
                Some(idx),
            )))
        }
        "estimator_v6" => Some(Box::new(crate::frozen::estimator_v6::EstimatorV6::new())),
        "estimator_v7" => Some(Box::new(crate::frozen::estimator_v7::EstimatorV7::new())),
        "estimator_v8" => Some(Box::new(crate::frozen::estimator_v8::EstimatorV8::new())),
        "estimator_v9" => Some(Box::new(crate::frozen::estimator_v9::EstimatorV9::new())),
        "estimator_v10" => Some(Box::new(crate::frozen::estimator_v10::EstimatorV10::new())),
        "estimator_v11" => Some(Box::new(crate::frozen::estimator_v11::EstimatorV11::new())),
        "estimator_v12" => Some(Box::new(crate::frozen::estimator_v12::EstimatorV12::new())),
        "estimator_v13" => Some(Box::new(crate::frozen::estimator_v13::EstimatorV13::new())),
        _ => None,
    }
}

/// 1候補の評価内訳（`Strategy::last_ranking` 用）。
/// score = combine_score(gain, p_legal, foul_cost) + adjust で、
/// depth2=true の候補は gain が2手読みで再構築された値
#[derive(Debug, Clone, serde::Serialize)]
pub struct CandidateScore {
    pub usi: String,
    /// 2手読みを掛ける前の score。静的評価だけの順位を診断するために保持する。
    pub static_score: f64,
    /// 2手読みを掛ける前の gain。depth2=true の候補では `gain` と異なる。
    pub static_gain: f64,
    pub score: f64,
    pub gain: f64,
    pub p_legal: f64,
    pub foul_cost: f64,
    /// gain の外側の補正（タイブレーク乱数・手戻り減点・ブラインド玉攻め等）
    pub adjust: f64,
    /// 2手読み（上位 depth2_top_k 候補の再評価）を通ったか
    pub depth2: bool,
    /// gain のうち王手駒の除去期待値（checker_removal_w × removal_term）分。
    /// 王手中の候補にだけ非ゼロが入る（gain には加算済みの内訳表示）
    pub checker_removal: f64,
    /// gain から引かれた捕獲の賭け分散ペナルティ
    /// （capture_bet_var_w × p_hit(1−p_hit) × E[捕獲価値|hit]、王手外のみ）。
    /// 正の値 = そのぶん gain が減っている（内訳表示。gain には控除済み）
    pub capture_bet_penalty: f64,
    /// gain に加算された詰めろ生成ボーナス（mate_threat_w × 成立確率 × 健全度）
    pub mate_threat: f64,
    /// gain から引かれた被詰めろペナルティ（mate_risk_w × 危険確率 × 健全度）。
    /// 正の値 = そのぶん gain が減っている
    pub mate_risk: f64,
    /// gain から引かれた自玉8近傍の穴の減点（king_hole_w × 穴の数）。正の値
    pub king_holes: f64,
    /// gain に加算された valueネット項（value_nn_w × (勝率相当 − 0.5)）。符号つき
    pub value_nn: f64,
    /// 粒子加重の期待駒得（この手で取れる敵駒の交換価値）。gain には加算済み
    pub capture_value: f64,
    /// 静的な取られリスク項の粒子加重平均（gain からは控除済みの正の値）。
    /// 2手読みを通った候補では depth2_replace 分が実測へ置き換わっている
    pub risk: f64,
    /// gain に加算された紐の項（link_w × 紐のついた自駒の価値合計。V3）。
    /// 候補間でほぼ定数なので、見るのは**候補どうしの差**
    pub link: f64,
    /// gain に加算された成りポテンシャルの差分
    /// （promo_potential_w × (着手後 − 現局面)。符号つき）
    pub promo: f64,
    /// gain から引かれた持ち駒オプションの不足分
    /// （hand_option_w × (その駒の最良打ちポテンシャル − この打ちマスの実現値)。
    /// 打つ手にだけ非ゼロが入る正の値。gain には控除済み）
    pub hand_option: f64,
    /// gain から引かれた盤上駒の減価（board_discount_w × 盤上の自駒の価値合計。V5）。
    /// 正の値 = そのぶん gain が減っている。これも見るのは差
    pub board_discount: f64,
}

/// 前進を好むヒューリスティック＋乱数（従来実装）
pub struct Heuristic;

impl Strategy for Heuristic {
    fn choose(
        &mut self,
        view: &PlayerView,
        _log: &ObservationLog,
        foul_tried: &HashSet<String>,
    ) -> Option<String> {
        choose_move(view, foul_tried)
    }

    fn name(&self) -> &'static str {
        "heuristic"
    }
}

/// 候補手を生成してスコア最大の手を返す。foul_tried の手は除外。
/// 候補が尽きたら None（呼び出し側で投了する）。
pub fn choose_move(view: &PlayerView, foul_tried: &HashSet<String>) -> Option<String> {
    let mut rng = rand::rng();
    let mut best: Option<(String, f64)> = None;
    let consider = |usi: String, score: f64, best: &mut Option<(String, f64)>| {
        if foul_tried.contains(&usi) {
            return;
        }
        if best.as_ref().is_none_or(|(_, s)| score > *s) {
            *best = Some((usi, score));
        }
    };

    let color = view.your_color;
    for piece in &view.your_pieces {
        let Some(from) = parse_usi_square(&piece.square) else {
            continue;
        };
        for to in move_targets(&view.your_pieces, piece, color) {
            let promote = promotion_choice(piece.role, from, to, color) != Promotion::None;
            // 前進を好む（先手は rank 減少が前進）
            let advance = match color {
                Color::Sente => (from.rank - to.rank) as f64,
                Color::Gote => (to.rank - from.rank) as f64,
            };
            let mut score = advance + rng.random_range(0.0..4.0);
            if promote {
                score += 3.0;
            }
            if piece.role == Role::King {
                score -= 2.0; // 玉は無闇に動かさない
            }
            consider(make_usi_move(from, to, promote), score, &mut best);
        }
    }

    for (&role, &count) in &view.your_hand {
        if count == 0 {
            continue;
        }
        for to in drop_targets(&view.your_pieces, role, color) {
            if let Some(usi) = make_usi_drop(role, to) {
                // 打ちは控えめに（乱数のみ）
                consider(usi, rng.random_range(0.0..3.0), &mut best);
            }
        }
    }

    best.map(|(usi, _)| usi)
}

/// 評価に使う粒子数の基準値（スケール1.0時）。実際の値は思考予算に比例する
const EVAL_PARTICLES: usize = 192;

/// 1手の思考予算（ms）の既定値。TSUITATE_THINK_BUDGET_MS で上書きできる。
/// このリポジトリのアリーナは 1000秒+3秒 なので既定はやや厚めに使う。
/// 本番サイト（300秒+3秒）へのデプロイ時は環境変数で絞って
/// 思考時間（=強さ）を調整する（例: 900 で v5 相当の予算）
const DEFAULT_THINK_BUDGET_MS: u64 = 2000;
/// スケール1.0の基準予算。v5 までの暗黙の実測上限（p99 ≒ 900ms）
const REFERENCE_BUDGET_MS: f64 = 900.0;

/// 思考予算（ms）。`TSUITATE_CAND_THINK_BUDGET_MS` > `TSUITATE_THINK_BUDGET_MS` > 既定値。
///
/// **`TSUITATE_THINK_BUDGET_MS` は凍結版（`frozen/estimator_v6..v11`）も読む**ので、
/// アリーナの `-f env=` で渡すと**両側の予算が動いて比較にならない**。
/// 候補側だけ予算を変えたいとき（予算-強さ曲線の測定、時間配分の検証。
/// docs/improvement-plan-2026-07-26-yaneuraou.md 項目1）は
/// `TSUITATE_CAND_THINK_BUDGET_MS` を使う: この名前は現行 `strategy.rs` しか
/// 知らないので、凍結版は既定（または `TSUITATE_THINK_BUDGET_MS`）のまま残る
fn think_budget_ms() -> u64 {
    for name in ["TSUITATE_CAND_THINK_BUDGET_MS", "TSUITATE_THINK_BUDGET_MS"] {
        if let Some(ms) = std::env::var(name).ok().and_then(|v| v.parse().ok()) {
            return ms;
        }
    }
    DEFAULT_THINK_BUDGET_MS
}

/// taint 粒子の玉を移設するとき、移設先に空きマスを優先するか（既定 on、
/// `TSUITATE_TAINT_KING_EMPTY=0` で従来挙動）。従来は候補マスの駒と無条件に
/// 入れ替えていたため、玉位置を直すたびに別の駒が根拠なく飛んでいた
fn taint_king_prefer_empty() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| !std::env::var("TSUITATE_TAINT_KING_EMPTY").is_ok_and(|v| v == "0"))
}

/// 厳密粒子が全滅した決定で、評価本体（`expected`）を taint 粒子へ落とすか。
/// 既定は無効（従来挙動）。`TSUITATE_EVAL_TAINT_FALLBACK=1` で有効。
/// 凍結版はこの名前を知らないので候補側にだけ効く
fn eval_taint_fallback() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("TSUITATE_EVAL_TAINT_FALLBACK").is_ok_and(|v| v == "1")
    })
}

/// 王手中の玉の手の gain を「玉の手全体の平均」に揃えるか（既定 on、
/// `TSUITATE_CHECK_KING_GAIN_MEAN=0` で従来挙動。凍結版はこの名前を知らない）。
/// = 玉の手**どうし**の序列は p_legal（CheckSolver の解消確率）と反則コストだけで
/// 決める。「王手中の候補序列は p_legal が支配すべき」（value_nn / 詰めろ2項 /
/// capture_bet_var の王手中ゲートと同じ教義）を玉の手の evaluate 本体にも適用する。
///
/// 発端は king-evade.kif（対人局レビュー 2026-07-29 の64手目、追加反則65回/20試行）。
/// 粒子が退化した終盤の王手では幻の敵駒が gain を両方向に歪める:
/// - 正解の逃げ手が**負**に沈む（実測: 7a が p_legal 0.887 と解消確率最上位なのに
///   露出リスク・圧力の幻で gain −1.5。min 形の combine_score は負の gain を
///   p_legal で割り引かないため全額効く）
/// - 間違った逃げ手が**正**に浮く（実測: 6b/7b が gain +1.1 前後。幻の gain 差
///   ±1 が p_legal の差 0.3 の序列を上書きする）
///
/// **0 への固定ではなく平均に揃えるのが要点**: 0 固定版は king-evade の反則を
/// 65→29 に減らしたが、玉の手全体の水準が下がって除去期待値つきの非玉プローブが
/// 相対的に浮上し、kakutori 17→50・dragon-check-drop 19→41 と押し出しの反則が
/// 爆発した（玉の逃げへの一律割引がアリーナ3シード全て悪化した king-prior の
/// 第1版と同型）。平均への揃えは玉の手どうしの分散（幻ノイズ）だけを消し、
/// 玉の手 vs 玉以外（合駒・捕獲プローブ）の相対水準を保存する。
/// 玉以外の手はそのまま: 駒を差し出す手の駒損は実在のコストで、王手駒捕獲の
/// 価値は removal_term（仮説条件付き期待値）が持つ
fn check_king_gain_mean() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("TSUITATE_CHECK_KING_GAIN_MEAN").map_or(true, |v| v != "0")
    })
}

/// V1（利き数）のノブ。**既定は両方 無効**（＝従来の二値の利き判定）。
///
/// やねうら王 Lv4 の「利き数」（+R30）をついたてへ持ち込む実験だったが、
/// 200局×3形態（統合 53.3% / 紐だけ 50.0% / 圧力だけ 49.7%）がいずれも
/// 対照 56.5%±6.9 を下回り、狙いの機械指標（只取られ 1050→1113、
/// 損な交換 1934→2128）も改善しなかったため既定から外した。
/// `attack_count` 自体は V3（予防的な紐）・V2（距離重み）で使えるので残す。
/// `TSUITATE_V1_PRESSURE=1` / `TSUITATE_V1_DEFENDED=1` で再度有効化できる
fn v1_pressure_multiplicity() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("TSUITATE_V1_PRESSURE").is_ok_and(|v| v == "1"))
}

fn v1_defended_by_count() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("TSUITATE_V1_DEFENDED").is_ok_and(|v| v == "1"))
}

/// 思考予算に比例して各種の粒子数・読み幅を決める
#[derive(Debug, Clone, Copy)]
struct SearchBudget {
    /// 推定器へ渡すスケール（粒子数・リプレイ予算）
    scale: f64,
    /// 評価に使うユニーク粒子数の上限
    eval_particles: usize,
    /// 王周辺圧力を測る粒子数
    pressure_samples: usize,
    /// valueネット（value_nn.rs）を評価する粒子数
    nn_samples: usize,
    /// 2手読みする上位候補数
    depth2_top_k: usize,
    /// 2手読みに使う粒子数
    depth2_particles: usize,
    /// 詰めろ生成（mate.rs::drop_mate）を判定する粒子数
    mate_samples: usize,
    /// 構想（自分の手 → 相手の応手 → 自分の手）を読む粒子数。
    /// **思考予算をここに使う**: 予算を増やしても強くならないのは、粒子数と
    /// 読み幅を比例させるだけで「同じことを多くやって」いるからで、
    /// 深さは買えていなかった（2000ms で飽和、8000ms でも +0.5pt）
    plan_particles: usize,
}

impl SearchBudget {
    fn from_ms(ms: u64) -> Self {
        let scale = (ms as f64 / REFERENCE_BUDGET_MS).clamp(0.25, 8.0);
        let f = |base: usize, lo: usize, hi: usize| ((base as f64 * scale) as usize).clamp(lo, hi);
        SearchBudget {
            scale,
            eval_particles: f(EVAL_PARTICLES, 48, 2048),
            pressure_samples: f(PRESSURE_SAMPLES, 8, 64),
            nn_samples: f(NN_SAMPLES, 16, 256),
            depth2_top_k: f(DEPTH2_TOP_K, 4, 32),
            depth2_particles: f(DEPTH2_PARTICLES, 16, 384),
            mate_samples: f(MATE_SAMPLES, 2, 32),
            plan_particles: f(PLAN_PARTICLES, 4, 128),
        }
    }
}

/// 王周辺圧力を測る粒子数の基準値（スケール1.0時）
const PRESSURE_SAMPLES: usize = 16;

/// valueネットを評価する粒子数の基準値（スケール1.0時）。forward pass自体は
/// 約0.6µs/回だが、transition特徴量の利き走査が粒子×候補ごとに掛かるため
/// 圧力項（PRESSURE_SAMPLES）と同様に粒子数を絞る
const NN_SAMPLES: usize = 48;

/// 2手読み（相手応手のサンプル再評価）を行う上位候補数の基準値（スケール1.0時）。
/// 1手読みの静的リスク項は近似なので、有望手だけ実際の応手分布で検算する
const DEPTH2_TOP_K: usize = 8;
/// 2手読みに使う粒子数の基準値（1候補あたり・スケール1.0時）
const DEPTH2_PARTICLES: usize = 48;
/// 応手で詰まされる場合のペナルティ（壊滅的なのでSPSA対象にしない）
const DEPTH2_MATE_PEN: f64 = 30.0;

/// 被詰めろのうち「玉で打った駒を取る以外に受けがない」形（MateThreat::
/// IfSupported）を数える割合。相手の支え駒が実際にそのマスへ利いている確率の
/// 代理で、真の詰み（Mate）より軽い。SPSA対象にはしない（mate_risk_w で
/// まとめて調整できる。分けても勾配が立たない）
const MATE_RISK_IF_SUPPORTED: f64 = 0.5;

/// 詰めろ生成の判定に使う粒子数の基準値（スケール1.0時）。1粒子あたり
/// 「玉の利き線上の空きマス × 持ち駒種」ぶんの詰み判定が走るので、
/// 圧力項（PRESSURE_SAMPLES）よりさらに絞る
const MATE_SAMPLES: usize = 6;

/// 構想（自分の手 → 相手の応手 → 自分の手）を読む粒子数の基準値。
/// 1粒子あたり `PLAN_REPLY_SAMPLES` 本の応手サンプル × `legal_moves()` 1回。
/// **思考予算をここに使う**方針なので depth2 と同オーダーで取る
/// （既存の項は予算を増やしても飽和していて、深さだけが未使用の余地だった）
const PLAN_PARTICLES: usize = 16;

/// 駒交換で動く価値: 盤上価値と持ち駒価値（基本駒種）の平均。
/// 素の駒は piece_value と一致し、成駒は取られても相手の持ち駒に入るのは
/// 基本駒種ぶんなので割り引かれる（と金を取り返された反動 = (6+1)/2 = 3.5）。
/// 逆に成駒を取る側の得も同じ理由で割り引く
pub(crate) fn exchange_value(role: Role) -> f64 {
    (piece_value(role) + piece_value(unpromote_role(role))) / 2.0
}

/// 着手後の自駒の利き被覆マス数（自分に見える盤面だけの近似）。
/// 相手の駒は見えないため飛び駒は自駒にだけ遮られる楽観値
fn coverage_after(view: &PlayerView, mv: &ShogiMove) -> f64 {
    own_effects_after(view, mv, None, &EvalParams::default()).coverage
}

/// 玉からの距離による利きの価値の減衰（V2。docs/yaneuraou-lessons.md 1-2）。
///
/// やねうら王 Lv3（利きの価値を玉からの距離で重み付ける）は連載中で最大の
/// 伸び（+R200）。向こうの optimizer が出した実測値は距離0〜8で
/// 1024 / 496 / 297 / 272 / 184 / 166 / 146 / 116 / 117 だが、**テーブルを
/// 写すのではなく**、それによく一致する近似式 `1/(1+d)` を使う
/// （0.5 / 0.333 / 0.25 … 対 実測 0.484 / 0.290 / 0.266 …。
/// やねうら王のコメントにある手決めの式 `83×1024/(d+1)` と同じ形）。
/// ついたての最適点は向こうと違うはずなので、係数は SPSA で別途調整する。
///
/// **底歩が可動性ゼロでも価値がある**のはこの重み付けで説明できる:
/// 利きは1マスでも、それが自玉の隣（距離1）なら 0.5。隅で誰も脅かしていない
/// と金は利きが2マスあっても両玉から遠い（距離7〜8）ので 0.25 前後にしかならない
fn king_dist_weight(d: i8) -> f64 {
    1.0 / (1.0 + d as f64)
}

/// チェビシェフ距離（将棋の玉が届くまでの手数と同じ尺度）
fn cheb(a: Coord, b: Coord) -> i8 {
    (a.file - b.file).abs().max((a.rank - b.rank).abs())
}

/// マスごとの「相手玉の信念位置からの距離重み」の期待値
/// `Σ_k P(玉=k) × king_dist_weight(dist(マス, k))`。
///
/// 相手玉の位置は粒子の信念なので、候補ごとに粒子を舐め直すと重い。
/// **決定点ごとに1度だけ**81マスぶん作って使い回す（`blind_king_attack` が
/// 玉位置分布を1度だけ作るのと同じ方針）。分布が空なら None = この項は無効
fn opp_king_effect_weights(dist: &[(Coord, f64)]) -> Option<[f64; 81]> {
    if dist.is_empty() {
        return None;
    }
    let total: f64 = dist.iter().map(|&(_, p)| p).sum();
    if total <= 0.0 {
        return None;
    }
    let mut w = [0.0f64; 81];
    for file in 1..=9i8 {
        for rank in 1..=9i8 {
            let sq = Coord { file, rank };
            let acc: f64 = dist
                .iter()
                .map(|&(k, p)| p * king_dist_weight(cheb(sq, k)))
                .sum();
            w[crate::belief_features::sq_index(sq)] = acc / total;
        }
    }
    Some(w)
}

/// 自駒だけで決まる評価量（粒子不要・ノイズゼロ）。4項が同じ
/// 「着手後の自駒配置と、その利き」を必要とするので1回の走査でまとめて出す
#[derive(Debug, Default, Clone, Copy)]
struct OwnEffects {
    /// 自駒の利き被覆マス数（索敵網の広さ。`coverage_w`）
    coverage: f64,
    /// 自玉8近傍のうち「玉以外の自駒の利きが無い」マスの数（V4。`king_hole_w`）
    king_holes: f64,
    /// 紐のついた自駒の価値合計（V3。`link_w`）
    linked_value: f64,
    /// **自玉からの距離で重み付けた自駒の利き**の総和（V2。`effect_own_w`）。
    /// 底歩のように「可動性はほぼ無いが自玉の隣に利いている駒」を正しく
    /// 拾うための量（可動性そのものではない、が要点）
    effect_own: f64,
    /// **相手玉の信念位置からの距離で重み付けた自駒の利き**の総和
    /// （V2。`effect_opp_w`）。相手玉の位置は粒子の信念なので、
    /// マスごとに `Σ_k P(玉=k) × w(dist)` を決定点ごとに1度だけ先に作る
    effect_opp: f64,
    /// **打ち当て露出**（`drop_hit_evac_w`）: 敵陣にいる自分の大駒（飛角竜馬）の
    /// うち、頭（自分の前進方向1マス先）が盤内かつ自駒で塞がれていないものの
    /// 交換価値合計。「相手が歩を持っているか」のゲートは呼び出し側
    /// （choose → evaluate）が観測ログから掛ける
    drop_hit_exposure: f64,
    /// **成りで増える利きのポテンシャル**（`promo_potential_w`）: 未成駒ごとの
    /// 「成ったら増える利き数 × 減衰^(成りマスまでの手数)」の合計。
    /// `promo_potential()` の doc 参照。`promo_potential_w == 0` か王手中は
    /// 計算しない（0 のまま）
    promo_potential: f64,
    /// **大駒の成り道**（`major_promo_path_w`）: 飛・角・香の素材の成り得を
    /// 成りマスまでの距離で減衰させた総和。`major_promo_path()` の doc 参照
    major_promo_path: f64,
    /// **この手で盤上に増える自駒の価値**（V5。`board_discount_w`）。
    /// 打ち = 打った駒の価値、成り = 増えた価値、それ以外 = 0。
    ///
    /// やねうら王 Lv2 は盤上の全駒に一律の減点を掛けるが、そのまま写すと
    /// 盤上価値の合計（70前後）が gain の定数オフセットになり、w=0.1 で
    /// **全候補の gain が −7 ほど下がって負に振り切る**。`combine_score` は
    /// `(p_legal×gain).min(gain)` なので、gain が負だと p_legal の割引が
    /// 丸ごと効かなくなり反則確率の序列が壊れる（threat_value の差分化が
    /// vs v9 −12pt で否定されたのと同じ罠）。候補間の差は増分だけなので、
    /// **増分だけを持つ**ことで順位への効果を保ったままゼロ点を動かさない
    board_material_added: f64,
}

/// 着手後の自駒配置を作り、`OwnEffects` の各項をまとめて計算する。
///
/// ついたてで**相手の駒が見えなくても完全に既知**な情報だけを使う
/// （自駒の位置と利き）。飛び駒は自駒にだけ遮られる楽観値になる。
///
/// `opp_king_w` は相手玉の信念からの距離重み（`opp_king_effect_weights`）。
/// None なら V2 の相手玉側（`effect_opp`）は 0 のまま
fn own_effects_after(
    view: &PlayerView,
    mv: &ShogiMove,
    opp_king_w: Option<&[f64; 81]>,
    params: &EvalParams,
) -> OwnEffects {
    let mut pieces: Vec<VisiblePiece> = view.your_pieces.clone();
    match *mv {
        ShogiMove::Board { from, to, promote } => {
            let from_usi = make_usi_square(from);
            let Some(p) = pieces.iter_mut().find(|p| p.square == from_usi) else {
                return OwnEffects::default();
            };
            if promote {
                if let Some(r) = promote_role(p.role) {
                    p.role = r;
                }
            }
            p.square = make_usi_square(to);
        }
        ShogiMove::Drop { role, to } => pieces.push(VisiblePiece {
            square: make_usi_square(to),
            role,
        }),
    }

    let king = pieces
        .iter()
        .find(|p| p.role == Role::King)
        .and_then(|p| parse_usi_square(&p.square));

    // 移動できるマス（自駒のマスを含まない）= 索敵網の広さ
    let mut covered: HashSet<Coord> = HashSet::new();
    // 利かせているマス（自駒のマスを含む）= 紐の判定用。玉の利きも数える
    // （玉で取り返す形も守りとしては成立する。玉を除くのは V4 の穴の側だけ）
    let mut defended: HashSet<Coord> = HashSet::new();
    // 玉以外の自駒が利かせているマス（V4 の穴の判定用）
    let mut defended_nonking: HashSet<Coord> = HashSet::new();
    // V2 の副産物: 駒ごとの「働き」= その駒の利きを玉からの距離で重み付けた和。
    // 紐（V3）を「守る価値のある駒か」で重み付けるのに使う（下記 linked_value）
    let mut work_by_sq: HashMap<Coord, f64> = HashMap::new();
    for p in &pieces {
        covered.extend(move_targets(&pieces, p, view.your_color));
        let d = defend_targets(&pieces, p, view.your_color);
        if let Some(sq) = parse_usi_square(&p.square) {
            let work: f64 = d
                .iter()
                .map(|&s| {
                    king.map_or(0.0, |k| king_dist_weight(cheb(s, k)))
                        + opp_king_w.map_or(0.0, |w| w[crate::belief_features::sq_index(s)])
                })
                .sum();
            work_by_sq.insert(sq, work);
        }
        if p.role != Role::King {
            defended_nonking.extend(d.iter().copied());
        }
        defended.extend(d);
    }
    let occupied: HashSet<Coord> = pieces
        .iter()
        .filter_map(|p| parse_usi_square(&p.square))
        .collect();
    let mut king_holes = 0.0;
    if let Some(king) = king {
        for df in -1..=1i8 {
            for dr in -1..=1i8 {
                if df == 0 && dr == 0 {
                    continue;
                }
                let c = Coord {
                    file: king.file + df,
                    rank: king.rank + dr,
                };
                if !(1..=9).contains(&c.file) || !(1..=9).contains(&c.rank) {
                    continue; // 盤外は穴ではない（壁として機能する）
                }
                if !defended_nonking.contains(&c) && !occupied.contains(&c) {
                    king_holes += 1.0;
                }
            }
        }
    }

    // V3: 紐のついた自駒（玉を除く）の価値合計。自分の利きは自分のマスへは
    // 届かないので、`defended` に載っている = 別の自駒が守っている。
    //
    // **働きによる重み付け**（2026-07-28、ユーザー指摘）: 素の交換価値だけで
    // 数えると「隅で何もしていないと金」に紐をつける手が、実戦的に価値のある
    // 紐と同じ重みで評価される（発端: watch-estimator 17手目の L*1b で、
    // 1一のと金と打った香が**相互に守り合う**ので単発の打ちで得られる紐の
    // 最大値になっていた）。守る価値は駒の材料価値だけでなく**その駒が
    // 働いているか**にも依るので、V2 の距離重み付き利きで係数を掛ける。
    //
    // `link_work_w` は補間ノブ（0 = 従来どおり係数1で挙動不変、1 = 完全に
    // 働きで重み付け）。飽和の基準 `link_work_ref()` 以上の働きがあれば係数1。
    //
    // **王手中は重み付けしない**（＝従来どおり満額の紐）。v12 の実測で
    // 「紐は王手中もゲートしてはいけない」（ゲートすると kakutori の反則が
    // 7 → 51）と分かっており、王手中の紐は反則経済に効いている。働きで
    // 薄めるとその効果を部分的に打ち消す（重み付け版の実測: kakutori の反則
    // 18 → 38、dragon-check-drop 12 → 23）。**遊び駒に紐をつけるな**という
    // 狙いは王手中には関係がないので、ここだけ従来の挙動に戻すのが素直
    let lw = if view.you_in_check {
        0.0
    } else {
        params.link_work_w
    };
    let work_ref = params.link_work_ref.max(1e-6);
    let linked_value = pieces
        .iter()
        .filter(|p| p.role != Role::King)
        .filter_map(|p| parse_usi_square(&p.square).map(|c| (c, p.role)))
        .filter(|(c, _)| defended.contains(c))
        .map(|(c, role)| {
            let factor = if lw == 0.0 {
                1.0
            } else {
                let work = work_by_sq.get(&c).copied().unwrap_or(0.0);
                (1.0 - lw) + lw * (work / work_ref).min(1.0)
            };
            exchange_value(role) * factor
        })
        .sum();

    // V5: この手で盤上に増える自駒の価値（打ち＝打った駒、成り＝増えたぶん）
    let board_material_added = match *mv {
        ShogiMove::Drop { role, .. } => piece_value(role),
        ShogiMove::Board { from, promote: true, .. } => view
            .your_pieces
            .iter()
            .find(|p| p.square == make_usi_square(from))
            .and_then(|p| promote_role(p.role).map(|r| piece_value(r) - piece_value(p.role)))
            .unwrap_or(0.0),
        ShogiMove::Board { .. } => 0.0,
    };

    // V2: 利きを「玉からの距離」で重み付ける。自玉側は完全既知なのでノイズゼロ、
    // 相手玉側は粒子の信念（決定点ごとに1度だけ作った期待重み）を使う。
    // 数えるのは `defended`（自駒の乗ったマスも含む利き）: 底歩が自玉の隣の
    // 自駒を守っている形も「利き」として拾う必要がある
    let mut effect_own = 0.0f64;
    let mut effect_opp = 0.0f64;
    for &sq in &defended {
        if let Some(k) = king {
            effect_own += king_dist_weight(cheb(sq, k));
        }
        if let Some(w) = opp_king_w {
            effect_opp += w[crate::belief_features::sq_index(sq)];
        }
    }

    OwnEffects {
        coverage: covered.len() as f64,
        king_holes,
        linked_value,
        effect_own,
        effect_opp,
        drop_hit_exposure: drop_hit_exposure(&pieces, view.your_color),
        promo_potential: if params.promo_potential_w != 0.0 && !view.you_in_check {
            promo_potential(&pieces, view.your_color)
        } else {
            0.0
        },
        major_promo_path: if params.major_promo_path_w != 0.0 && !view.you_in_check {
            major_promo_path(&pieces, view.your_color)
        } else {
            0.0
        },
        board_material_added,
    }
}

/// 打ち当て露出（`drop_hit_evac_w` の素材）。自分の大駒（飛角竜馬）のうち、
/// **頭**（自分の前進方向に1マス先 = 相手の歩がそこへ打たれると次の相手の手で
/// この駒を取れるマス）が盤内かつ自駒で塞がれていないものの交換価値の合計。
///
/// 頭のマスに**敵駒**がいる可能性（打てない）は見えないので無視する =
/// 露出は上界。逆に相手の二歩制約も相手の歩の配置が見えないので無視する。
/// 紐（自分の利き）は軽減に数えない: 歩打ちは観測されないため取られてから
/// 取り返しても「大駒 ↔ 歩」の交換にしかならない。
///
/// **2026-08-03 に自陣側へ拡張**（ユーザー指摘）: 初版は敵陣（最奥3段）の大駒
/// だけを数えていたが、「相手が歩を入手した瞬間に飛車の頭へ歩を打たれる」危険は
/// **自陣に居る飛車にも同じように生じる**（quest31-m021 の先手飛車2八は
/// 2七へ打たれうるのに露出0と数えられていた）。`TSUITATE_DROP_HIT_ALL_RANKS=0`
/// で旧挙動（敵陣のみ）へ戻せる。
/// この拡張は `major_promo_path_w` の動機とも繋がる: 先に成って敵陣へ入って
/// しまえば、自陣で歩打ちに晒され続ける状態から抜けられる
fn drop_hit_exposure(pieces: &[VisiblePiece], me: Color) -> f64 {
    let occupied: HashSet<Coord> = pieces
        .iter()
        .filter_map(|p| parse_usi_square(&p.square))
        .collect();
    pieces
        .iter()
        .filter(|p| {
            matches!(
                p.role,
                Role::Rook | Role::Bishop | Role::Dragon | Role::Horse
            )
        })
        .filter_map(|p| parse_usi_square(&p.square).map(|c| (c, p.role)))
        .filter(|&(c, _)| {
            // 段の向き: 先手は rank 減少方向へ前進（advance_bias と同じ規約）
            drop_hit_all_ranks()
                || match me {
                    Color::Sente => c.rank <= 3,
                    Color::Gote => c.rank >= 7,
                }
        })
        .filter(|&(c, _)| {
            let head = Coord {
                file: c.file,
                rank: match me {
                    Color::Sente => c.rank - 1,
                    Color::Gote => c.rank + 1,
                },
            };
            (1..=9).contains(&head.rank) && !occupied.contains(&head)
        })
        .map(|(_, role)| exchange_value(role))
        .sum()
}

/// 打ち当て露出を**全段**で数えるか（既定 on、`TSUITATE_DROP_HIT_ALL_RANKS=0` で
/// 旧挙動 = 敵陣の大駒のみ）。`drop_hit_exposure` の doc 参照
fn drop_hit_all_ranks() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| !std::env::var("TSUITATE_DROP_HIT_ALL_RANKS").is_ok_and(|v| v == "0"))
}

/// 成りポテンシャルの手数減衰（1手遠いごとに半減）
const PROMO_POTENTIAL_DECAY: f64 = 0.5;
/// 成りマス探索（BFS）の手数上限。0.5^8 ≈ 0.004 で寄与が消えるので打ち切る
const PROMO_BFS_MAX_DEPTH: u32 = 8;

/// **成りで増える利きのポテンシャル**（`promo_potential_w` の素材）。
///
/// やねうら王に成りの専用項は無く、成りの価値は「利きの全マス評価（Lv3、+R200）が
/// 成った瞬間に増える」＋「深い探索が成りの未来を実際に読む」から出てくる
/// （docs/yaneuraou-lessons.md）。この bot は探索が1〜2手で打ち切られるので、
/// 探索の代わりに**将来の利き増加を静的な勾配で前借り**する。旧 tokin_probe_w
/// （歩専用・削除済み）の一般化で、駒種ハードコードなしに
/// 「歩→と金 +5利き ≫ 銀→成銀 +1利き」の序列が自然に出る。
///
/// 未成駒ごとに: Δ利き（成り駒種の利き数 − 現駒種の利き数、成りマス上で測る）
/// × 減衰^(成りマスまでの最短手数)。成りマスと手数は自駒ブロッカーだけを見る
/// BFS の下限（相手駒は見えないので楽観値 = `move_targets` と同じ規約）。
/// 自駒に道を塞がれて成れない駒（と金の裏に打った香など）はポテンシャル 0。
/// 成り済みの駒は「実現した利き増加」を減衰なしの満額で数える —— これが無いと
/// 成る手自体が「ポテンシャルを失う手」になって差分が負に出る。
///
/// 呼び出し側は `drop_hit_evac_w` と同じ**差分形**（着手後 − 現局面）で gain へ
/// 加算するので、無関係な候補は 0 でゼロ点が動かない（threat_value 差分化の罠を
/// 回避）。自駒だけで決まるので粒子不要・ノイズゼロ
fn promo_potential(pieces: &[VisiblePiece], me: Color) -> f64 {
    let occupied: HashSet<Coord> = pieces
        .iter()
        .filter_map(|p| parse_usi_square(&p.square))
        .collect();
    let mut total = 0.0f64;
    for p in pieces {
        let Some(origin) = parse_usi_square(&p.square) else {
            continue;
        };
        if unpromote_role(p.role) != p.role {
            // 成り済み: 実現した利き増加（成る手の差分を正にするための対）
            total += promo_effect_gain(&occupied, origin, unpromote_role(p.role), origin, me);
        } else {
            total += piece_promo_potential(&occupied, origin, p.role, me);
        }
    }
    total
}

/// **大駒（飛・角・香）の成り道**の価値（`major_promo_path_w`、2026-08-03）。
///
/// `promo_potential` は成りの価値を **Δ利き**（成り駒種と現駒種の利きマス数の差）で
/// 測るので、飛→龍が +4 利きにしかならず**歩→と金の +5 より小さく**なる。
/// ところが実戦的には「自分の飛車の前を自駒がどいて次手で成れる」は決定的な差で、
/// ユーザーが quest31-m021 の 3二と（2三のと金をどけて先手飛車2八の前を開ける手）を
/// 本命とした理由がまさにこれ。実測でも `promo_potential` の寄与は +0.300 しかなく、
/// **飛車の道を開けない 3一と（+0.400）のほうが高い**＝この概念は表現されていない。
///
/// そこで大駒だけを対象に、**素材の成り得**（龍−飛 = 2.5、馬−角 = 2.0、成香−香 = 3.0）を
/// 成りマスまでの手数で減衰させて足す。距離は `promo_distance`（自駒ブロッカーのみ
/// 考慮の BFS）を共用するので、**自駒がどけば距離が縮んで値が跳ねる**。
/// 歩・桂・銀は対象外（`promo_potential` の領分。ここで足すと垂れ歩が支配的になる）。
/// 差分形（着手後−現局面）で gain 内・王手中は無効・粒子不要（自駒だけで決まる）
fn major_promo_path(pieces: &[VisiblePiece], me: Color) -> f64 {
    let occupied: HashSet<Coord> = pieces
        .iter()
        .filter_map(|p| parse_usi_square(&p.square))
        .collect();
    let mut total = 0.0f64;
    for p in pieces {
        if !matches!(p.role, Role::Rook | Role::Bishop | Role::Lance) {
            continue; // 成り済み・小駒は対象外
        }
        let Some(at) = parse_usi_square(&p.square) else {
            continue;
        };
        let Some(promoted) = promote_role(p.role) else {
            continue;
        };
        let gain = exchange_value(promoted) - exchange_value(p.role);
        if let Some((d, _)) = promo_distance(&occupied, at, p.role, me) {
            total += gain * PROMO_POTENTIAL_DECAY.powi(d as i32);
        }
    }
    total
}

/// 未成駒 `role` がマス `at` にいる（または打たれる）ときの成りポテンシャル
/// 「Δ利き × 減衰^(成りマスまでの手数)」。`promo_potential()` の1駒ぶんと、
/// 持ち駒オプション（`hand_option_w`）の打ちマス実現値の共通部品
fn piece_promo_potential(occupied: &HashSet<Coord>, at: Coord, role: Role, me: Color) -> f64 {
    if promote_role(role).is_none() {
        return 0.0;
    }
    match promo_distance(occupied, at, role, me) {
        Some((d, sq)) => {
            promo_effect_gain(occupied, at, role, sq, me) * PROMO_POTENTIAL_DECAY.powi(d as i32)
        }
        None => 0.0,
    }
}

/// 持ち駒オプション価値（`hand_option_w`）の決定点コンテキスト。
/// choose() が1度だけ作り、evaluate() が打つ手の不足分を引くのに使う
struct HandOption {
    /// 自駒の占有マス（`piece_promo_potential` のブロッカー）
    occupied: HashSet<Coord>,
    /// 手持ち駒種ごとの最良打ちポテンシャル h(r) = max_s pp(r, s)。
    /// s は自分視点の合法な打ちマス（二歩・行き所を除外 = `drop_targets`）。
    /// h が 0 の駒種（金・全マス塞がり）は載せない = 減点なし
    best: HashMap<Role, f64>,
}

/// 手持ち駒ごとの最良打ちポテンシャルを計算する（`hand_option_w` の素材）。
/// 自駒配置と持ち駒だけで決まるので粒子不要・ノイズゼロ
fn hand_option_context(view: &PlayerView) -> HandOption {
    let me = view.your_color;
    let occupied: HashSet<Coord> = view
        .your_pieces
        .iter()
        .filter_map(|p| parse_usi_square(&p.square))
        .collect();
    let mut best = HashMap::new();
    for (&role, &count) in &view.your_hand {
        if count == 0 || promote_role(role).is_none() {
            continue;
        }
        let h = crate::board::drop_targets(&view.your_pieces, role, me)
            .into_iter()
            .map(|s| piece_promo_potential(&occupied, s, role, me))
            .fold(0.0f64, f64::max);
        if h > 0.0 {
            best.insert(role, h);
        }
    }
    HandOption { occupied, best }
}

/// 駒種 `base` がマス `at` で成ったときに増える利き数（負なら 0）。
/// `vacated`（駒の元位置）は空きマスとして扱う。利きは `defend_targets` と
/// 同じ規約（自駒の乗ったマスも利きに数え、レイはそこで止まる）
fn promo_effect_gain(
    occupied: &HashSet<Coord>,
    vacated: Coord,
    base: Role,
    at: Coord,
    me: Color,
) -> f64 {
    let Some(promoted) = promote_role(base) else {
        return 0.0;
    };
    (count_effects(occupied, vacated, promoted, at, me)
        - count_effects(occupied, vacated, base, at, me))
    .max(0.0)
}

/// 駒種 `role` がマス `at` から利かせているマス数（`vacated` は空き扱い）
fn count_effects(occupied: &HashSet<Coord>, vacated: Coord, role: Role, at: Coord, me: Color) -> f64 {
    use crate::board::{on_board, orient, rays, steps};
    let blocked = |c: Coord| c != vacated && occupied.contains(&c);
    let mut n = 0.0;
    for &delta in steps(role) {
        let (df, dr) = orient(delta, me);
        let c = Coord {
            file: at.file + df,
            rank: at.rank + dr,
        };
        if on_board(c) {
            n += 1.0;
        }
    }
    for &delta in rays(role) {
        let (df, dr) = orient(delta, me);
        let mut c = Coord {
            file: at.file + df,
            rank: at.rank + dr,
        };
        while on_board(c) {
            n += 1.0;
            if blocked(c) {
                break;
            }
            c = Coord {
                file: c.file + df,
                rank: c.rank + dr,
            };
        }
    }
    n
}

/// `origin` の未成駒 `role` が成る手を指せるまでの最短手数と、その成りマス。
/// 自駒ブロッカーだけを見た BFS の下限（`move_targets` と同じ規約で、
/// 自駒のいるマスには行けずレイもそこで止まる。`origin` は空き扱い）。
/// すでに敵陣にいて動ける駒は (1, 現在地) —— 敵陣からの移動は着地がどこでも成れる。
/// `PROMO_BFS_MAX_DEPTH` 手以内に成りマスへ届かなければ None
fn promo_distance(
    occupied: &HashSet<Coord>,
    origin: Coord,
    role: Role,
    me: Color,
) -> Option<(u32, Coord)> {
    use crate::board::{on_board, orient, rays, steps};
    let in_zone = |c: Coord| match me {
        Color::Sente => c.rank <= 3,
        Color::Gote => c.rank >= 7,
    };
    let blocked = |c: Coord| c != origin && occupied.contains(&c);
    let targets = |s: Coord, out: &mut Vec<Coord>| {
        for &delta in steps(role) {
            let (df, dr) = orient(delta, me);
            let c = Coord {
                file: s.file + df,
                rank: s.rank + dr,
            };
            if on_board(c) && !blocked(c) {
                out.push(c);
            }
        }
        for &delta in rays(role) {
            let (df, dr) = orient(delta, me);
            let mut c = Coord {
                file: s.file + df,
                rank: s.rank + dr,
            };
            while on_board(c) && !blocked(c) {
                out.push(c);
                c = Coord {
                    file: c.file + df,
                    rank: c.rank + dr,
                };
            }
        }
    };
    let mut buf = vec![];
    if in_zone(origin) {
        targets(origin, &mut buf);
        if !buf.is_empty() {
            return Some((1, origin));
        }
        return None; // 敵陣内で動けない駒（と金の裏の香など）は成れない
    }
    let mut visited: HashSet<Coord> = HashSet::new();
    visited.insert(origin);
    let mut frontier = vec![origin];
    for depth in 1..=PROMO_BFS_MAX_DEPTH {
        let mut next_frontier = vec![];
        for &s in &frontier {
            buf.clear();
            targets(s, &mut buf);
            for &t in &buf {
                if in_zone(t) {
                    return Some((depth, t));
                }
                if visited.insert(t) {
                    next_frontier.push(t);
                }
            }
        }
        if next_frontier.is_empty() {
            return None;
        }
        frontier = next_frontier;
    }
    None
}

/// この手の後、**自玉の8近傍のうち「玉以外の自駒の利きが無い」マスの数**（0〜8）。
///
/// docs/yaneuraou-lessons.md の V4（やねうら王 Lv8 で +R35）。ついたてと相性が
/// 良いのは、**自分の駒だけで計算できる**から: 相手の駒が見えなくても自玉の位置も
/// 自駒の利きも完全既知なので、粒子を使わずノイズゼロで測れる。
///
/// 玉自身の利きは除く（玉が自分で守っているマスは「支えがある」とは言えない。
/// 玉で取り返す形は詰みへ直結するため。やねうら王も「玉以外の味方の利き」で数える）。
/// 相手の駒が見えないので、やねうら王の「そこが空きか敵駒なら減点・味方の駒が
/// あるなら加点」の区別はできない。**自駒が乗っているマスは穴に数えない**
/// （壁として機能するため）という近似にする
fn king_holes_after(view: &PlayerView, mv: &ShogiMove) -> f64 {
    own_effects_after(view, mv, None, &EvalParams::default()).king_holes
}

/// アンチドロー（終盤の寄せ）: 増幅を始める手数（plies）
const ANTI_DRAW_START: f64 = 60.0;
/// 増幅が最大になる手数。アリーナの手数上限200の手前で全開にする
const ANTI_DRAW_FULL: f64 = 160.0;
/// リードの正規化単位（歩換算。8 ≒ 飛車1枚のリードでほぼフル増幅）
const ANTI_DRAW_LEAD_UNIT: f64 = 8.0;

/// 終盤の攻め増幅係数。手数が進むほど・素材リードがあるほど大きくなる。
/// 互角でも弱く掛けて膠着を破りにいくが、負けているときは掛けない
/// （負けているときの引き分けは0.5勝ぶんの価値がある）
fn endgame_push(move_number: u32, lead: f64) -> f64 {
    let ramp = ((f64::from(move_number) - ANTI_DRAW_START) / (ANTI_DRAW_FULL - ANTI_DRAW_START))
        .clamp(0.0, 1.0);
    (ramp * (0.3 + (lead / ANTI_DRAW_LEAD_UNIT).clamp(-0.3, 1.2))).max(0.0)
}

/// **自分側の配置**（盤上の自駒と持ち駒）の指紋。
///
/// ついたてでは自分側の情報は完全既知なので、この量は**粒子を使わず
/// ノイズゼロ**で計算できる。同じ指紋がまた出る = 「その間に何も起きていない
/// のに同じ形へ戻った」ということで、無意味な往復の直接の証拠になる
/// （相手に駒を取られた／自分が取ったなら持ち駒か盤上が変わるので指紋も変わる）。
fn own_config_fingerprint<'a>(
    pieces: impl Iterator<Item = (&'a str, Role)>,
    hand: &HashMap<Role, u32>,
) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut items: Vec<(&str, Role)> = pieces.collect();
    items.sort_unstable();
    let mut hands: Vec<(u8, u32)> = hand
        .iter()
        .filter(|&(_, &n)| n > 0)
        .map(|(&r, &n)| (r as u8, n))
        .collect();
    hands.sort_unstable();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    items.hash(&mut h);
    hands.hash(&mut h);
    h.finish()
}

/// これまでに出現した自分側の配置と、その出現回数。
/// 受理された自分の手のたびに1つ記録する（初期配置も1回として数える）
fn own_config_history(my_color: Color, log: &ObservationLog) -> HashMap<u64, u32> {
    let mut model = GameModel::new(my_color);
    let mut counts: HashMap<u64, u32> = HashMap::new();
    let mut record = |m: &GameModel| {
        let pieces = m.my_pieces();
        let hand = m.my_hand();
        let fp = own_config_fingerprint(pieces.iter().map(|p| (p.square.as_str(), p.role)), &hand);
        *counts.entry(fp).or_insert(0) += 1;
    };
    record(&model);
    for e in log.events() {
        model.apply(e);
        if matches!(e, Observation::MyMove { .. }) {
            record(&model);
        }
    }
    counts
}

/// 候補手を指した**後**の自分側配置の指紋。
///
/// 取りが成立するかは指してみないと分からない（相手の駒は見えない）ので、
/// **取らなかった場合**の配置で数える。実際に取れたなら持ち駒が増えて
/// 次からは別の指紋になるので、往復の連鎖はそこで自然に切れる
fn own_config_fingerprint_after(view: &PlayerView, mv: &ShogiMove) -> u64 {
    let mut pieces: Vec<(String, Role)> = view
        .your_pieces
        .iter()
        .map(|p| (p.square.clone(), p.role))
        .collect();
    let mut hand: HashMap<Role, u32> = view.your_hand.iter().map(|(&r, &n)| (r, n)).collect();
    match *mv {
        ShogiMove::Board { from, to, promote } => {
            let from_usi = make_usi_square(from);
            if let Some(p) = pieces.iter_mut().find(|p| p.0 == from_usi) {
                if promote {
                    if let Some(r) = promote_role(p.1) {
                        p.1 = r;
                    }
                }
                p.0 = make_usi_square(to);
            }
        }
        ShogiMove::Drop { role, to } => {
            if let Some(n) = hand.get_mut(&role) {
                *n = n.saturating_sub(1);
            }
            pieces.push((make_usi_square(to), role));
        }
    }
    own_config_fingerprint(pieces.iter().map(|(s, r)| (s.as_str(), *r)), &hand)
}


/// 観測から確実に分かる素材リード（歩換算・相対値）。
/// 自分の駒の増減は取った駒（持ち駒に入る）と取られた駒を両方含み、
/// 相手側は鏡像（自分が+vなら相手は-v）なので、リード = 自分の変化×2。
/// 成りは基本駒種で数える（成駒を取った得は過小評価だが単調な信号としては十分）
fn material_lead(view: &PlayerView) -> f64 {
    let current: f64 = view
        .your_pieces
        .iter()
        .map(|p| piece_value(unpromote_role(p.role)))
        .sum::<f64>()
        + view
            .your_hand
            .iter()
            .map(|(r, n)| piece_value(*r) * f64::from(*n))
            .sum::<f64>();
    let initial: f64 = Position::initial()
        .pieces()
        .filter(|(_, p)| p.color == view.your_color)
        .map(|(_, p)| piece_value(p.role))
        .sum();
    2.0 * (current - initial)
}

/// evaluate() の結果。最終スコアでなく内訳を保持し、2手読みが
/// gain を組み替えた後に同じ最終式を適用し直せるようにする
/// （min形の非線形式に対して後から線形補正すると負のgainで壊れるため）
struct EvalOut {
    /// 期待値＋バイアス項（合法確率・反則コストを含まない）
    gain: f64,
    /// 静的な取られリスク項（mover/hidden の max）の粒子加重平均。
    /// 2手読みがこの分をサンプル実測で置き換える
    risk_mean: f64,
    p_legal: f64,
    foul_cost: f64,
    /// gain のうち王手駒の除去期待値（checker_removal_w × removal_term）分。
    /// 王手中の候補にだけ入る（内訳表示用。gain には加算済み）
    checker_removal: f64,
    /// gain から引かれた捕獲の賭け分散ペナルティ
    /// （capture_bet_var_w × p_hit(1−p_hit) × E[捕獲価値|hit]）。
    /// 正の値 = そのぶん gain が減っている（内訳表示用。gain には控除済み）
    capture_bet_penalty: f64,
    /// gain に加算された詰めろ生成ボーナス（内訳表示用）
    mate_threat: f64,
    /// gain から引かれた被詰めろペナルティ（正の値。内訳表示用）
    mate_risk: f64,
    /// gain から引かれた自玉8近傍の穴の減点（正の値。内訳表示用）
    king_holes: f64,
    /// gain に加算された valueネット項（符号つき。内訳表示用）
    value_nn: f64,
    /// 粒子加重の期待駒得（内訳表示用。gain には加算済み）
    capture_value: f64,
    /// gain に加算された紐の項（V3。内訳表示用）
    link: f64,
    /// gain に加算された成りポテンシャルの差分（符号つき。内訳表示用）
    promo: f64,
    /// gain から引かれた持ち駒オプションの不足分（正の値。内訳表示用）
    hand_option: f64,
    /// gain から引かれた盤上駒の減価（V5。正の値。内訳表示用）
    board_discount: f64,
    /// 打ちプローブの反則情報価値（drop_probe_w）。反則の失敗枝の期待値なので
    /// gain には含めず、combine_score の外側（(1−p_legal) 側）で加算する
    foul_probe: f64,
}

impl EvalOut {
    fn score(&self) -> f64 {
        combine_score(self.gain, self.p_legal, self.foul_cost) + self.foul_probe
    }
}

/// 最終スコア: 期待値が負の手を p_legal で割り引かない（min の形）。
/// 割り引くと「合法確率が低いほどスコアが高い」= わざと反則に寄る手が
/// 選ばれてしまう。反則しても手番は残るので悪い局面からは逃げられず、
/// 反則の価値は「次善手の価値 − 反則コスト」でしかない
pub(crate) fn combine_score(gain: f64, p_legal: f64, foul_cost: f64) -> f64 {
    (p_legal * gain).min(gain) - (1.0 - p_legal) * foul_cost
}

/// evaluate() まわりの調整可能パラメータ。Default が現行の手調整値。
/// bin/tune.rs の SPSA がこれを最適化する（凍結版は各自のコピーを持ち依存しない）
#[derive(Debug, Clone)]
pub struct EvalParams {
    /// 王手ボーナスの基本値
    pub check_bonus: f64,
    /// 王手ボーナスの相手反則数スケール
    pub check_foul_scale: f64,
    /// 着手駒の取られリスク重み（駒を取った直後 = 位置がバレている）
    pub mover_w_captured: f64,
    /// 着手駒の取られリスク重み（静かな手）
    pub mover_w_quiet: f64,
    /// 着手駒の取られリスク重みへの加算（王手をかけた手）。王手宣言は「王を攻撃
    /// できる（マス,駒種）」まで仮説を絞らせるので、相手は反則覚悟の探り取りで
    /// 王手駒を高確率で回収できる（対人実戦: 竜の王手→2反則で竜を取られた）。
    /// 旧 mover_w_check は quiet/captured との max で不感帯があった
    /// （SPSAで勾配が立たない）ため、非負の加算に変更
    pub mover_check_extra: f64,
    /// 捕獲後の残留露見リスク（自駒価値に掛ける割合）。取ったマスは相手に
    /// 通知されるため、粒子に守り駒が見えなくても取り返しの下限リスクを敷く。
    /// 等価な取りなら安い駒で取る、というタイブレークにもなる
    /// （対人実戦: 成桂でも取れる角を竜で取って竜を回収された）
    pub capture_reveal_risk: f64,
    /// 敵陣リスク下限の「静かな進入」係数（捕獲時は 1.0）
    pub camp_known_quiet: f64,
    /// 敵陣の守られ事前確率のスケール（1.0 で 0.25/0.2/0.15）
    pub camp_scale: f64,
    /// 露出リスクの基本重み
    pub exposed_base: f64,
    /// 露出リスクの既知度係数
    pub exposed_known: f64,
    /// 初期配置から動いていない駒の既知度
    pub home_knownness: f64,
    /// 紐つき割引（着手駒）
    pub recapture_defended: f64,
    /// 紐つき割引（露出駒）
    pub exposed_defended: f64,
    /// 相手玉周辺への攻め圧力の重み
    pub attack_w: f64,
    /// 自玉周辺への相手圧力の重み
    pub pressure_w: f64,
    /// 反則コストの基本値
    pub foul_cost_base: f64,
    /// 反則コストの急峻さ（残り反則数に対する冪）
    pub foul_cost_pow: f64,
    /// 前進バイアス
    pub advance_w: f64,
    /// 成りバイアス
    pub promote_bias: f64,
    /// 打ちバイアス
    pub drop_bias: f64,
    /// p(合法) 事前確率の擬似観測数
    pub prior_weight: f64,
    /// 粒子退化時に prior_weight へ加算する上限（ユニーク粒子が減るほど事前を信じる。
    /// 少数の複製・偏った粒子への過信 = 「自信過剰な間違い」を防ぐ）
    pub prior_weight_degen: f64,
    /// 着手後に自分が当たりを付けている敵駒の価値への重み（露出リスクの鏡像）。
    /// 1手読みでは見えない「次の駒得」（飛車頭への歩打ち等）を作る手に価値を与える
    pub threat_w: f64,
    /// 探索ボーナス: 着地マスの敵駒有無について粒子が割れているほど加点。
    /// 取れても空振りでも観測が推定を絞る（情報の価値）
    pub info_bonus: f64,
    /// 大駒（飛・角）が初期位置に残っていることへのペナルティ（1枚あたり）。
    /// 初期位置の大駒は位置が予測可能で、開いた筋の背後を歩・桂で狙われる
    /// （対人50局で頻発）。展開を促す勾配を作り、動かせば消える
    pub big_home_penalty: f64,
    /// 相手の持ち駒による「打ち込み王手の受け入れ面積」への重み。
    /// 相手の持ち駒は既知（=取られた自駒）で、飛を持たれたら玉への開いた直線、
    /// 金銀なら玉の隣接空きマスがすべて王手打ちの入口になる。
    /// 持ち駒が空なら居玉でもコストゼロ（一律の玉移動推奨はしない）
    pub hand_drop_w: f64,
    /// 手戻り減点
    pub backtrack_penalty: f64,
    /// 直前に動かした駒をまた動かす手の減点（雑なシャッフルの抑制。
    /// 駒得や王手が絡む手は期待値側が勝つので実質影響しない）
    pub shuffle_penalty: f64,
    /// 【C-7 P1 で未使用化】ソフト救済粒子の評価重み減衰。フィルタ側の
    /// EPS_INFO（estimator.rs）へ統合された。SPSAベクタのレイアウト互換のため
    /// フィールドは残す（調整しても無効）
    pub soft_decay: f64,
    /// 王探しの情報利得: 粒子間で王手判定が割れる手への p(1-p) 加点
    pub king_probe_bonus: f64,
    /// 利き被覆1マスあたりの加点（自駒のみ考慮の近似被覆）
    pub coverage_w: f64,
    /// 2手読みで静的リスク項をサンプル実測に置き換える割合（0=従来、1=全面置換）
    pub depth2_replace: f64,
    /// 2手読みの**楽観方向の置き換え**の上限（0 = 従来と同一挙動、1 = 楽観禁止 =
    /// 実効リスクが静的リスクを下回らない）。
    ///
    /// 置き換えは実効リスク `(1−w)·risk_mean + w·|delta|` の形になる
    /// （w = depth2_replace）。相手の応手方策は取り返しを確率的にしか選ばないので
    /// **|delta| < risk_mean になりやすく、静的リスクが大きいほど戻し額が大きい**
    /// = 危険な捕獲ほど加点される、という逆転が起きる（quest31-m030f1 の実測:
    /// 1二飛の反則で「3三は守られている」信念が 7.2%→57.7% に上がり静的評価は
    /// 正しく 6.844→5.882 に下がったのに、2手読みの戻しが +0.70→+1.85 に増えて
    /// 最終 gain は 7.547→7.734 と上がった。F3、
    /// docs/improvement-plan-2026-08-02-quest31.md）。
    ///
    /// この項は緩和（relief）を `(1−cap)·risk_mean` で頭打ちにする。悲観方向
    /// （delta が静的リスクより大きい損失を見つけた場合）は制限しない
    pub depth2_optimism_cap: f64,
    /// 2手読みで応手に王手を掛けられた場合のペナルティ
    pub depth2_check_pen: f64,
    /// 2手読みの取り返し補償の割引（取り返し自体への反撃リスクの近似）
    pub depth2_recap_discount: f64,
    /// 反則コストの残数差項: ×(相手残数/10)^pow。相手が反則上限に近いほど
    /// 自分の反則は相対的に安い（反則レースの相対価値。0=従来）
    pub foul_diff_pow: f64,
    /// 王手の反則誘発価値の上限加速: check_foul_scale 項に ×(10/相手残数)^accel。
    /// 相手が反則負けに近づくほど1回の誘発の限界価値が跳ねる（0=従来）
    pub check_limit_accel: f64,
    /// 粒子上のvalueネット（value_nn.rs、NN段階③）の重み。粒子ごとに
    /// (state特徴量16 + transition特徴量6) → 勝率相当[0,1] を推論し、
    /// 重み付き平均の (avg − 0.5) をこの係数で歩価値スケールへ換算して
    /// gain に加算する。手作り項が横並びになる静かな局面の序列付けが狙い
    /// （54手目9二香: 意味を問わない advance_bias だけで手が決まる問題）。
    /// 0 = NN無効（従来と同一挙動）
    pub value_nn_w: f64,
    /// 王手中の仮説条件付き「王手駒の除去期待値」（CheckSolver::removal_term、
    /// 歩価値スケール）の重み。王手駒のマスを取る手には+交換価値、王手駒を
    /// 盤に残す解消手には−残存脅威を、受理を条件付けた仮説の事後分布で
    /// 平均して gain へ加算する。p_legal は合法性しか平均しないため、粒子が
    /// 真の王手駒を外している局面では捕獲の価値が評価のどこにも現れない
    /// （kakutori.kif）ことへの対応。旧 CHECK_CAPTURE_P_LEGAL_FLOOR
    /// （一律0.35のp_legal下限）の置き換え。0 = 無効（従来と同一挙動）
    pub checker_removal_w: f64,
    /// 捕獲の賭け分散ペナルティの重み。p_hit(1−p_hit) × E[捕獲価値|hit] を
    /// gain から引く（王手中は無効）。占有が五分に近いマスへの高額な捕獲賭けは
    /// 空振り分岐の認識悪化（信念の前提崩壊＋進出駒の孤立）を素の期待値が
    /// 数えないことへの補正。0 = 無効（従来と同一挙動）
    pub capture_bet_var_w: f64,
    /// 詰めろ生成（この手の後、次の自分の手番で持ち駒打ちの一手詰めが
    /// 成立する）ボーナス。粒子上での成立確率 × 粒子健全度で掛ける。
    /// ついたて将棋では相手から脅威が見えないので詰めろは受けられにくく、
    /// 通常将棋より実効価値が高い（2026-07-25 の対人局: 58手目 N*6六 で
    /// 詰めろ、bot は受けを選ばず 60手目 G*7八 で詰み）。0 = 無効
    pub mate_threat_w: f64,
    /// 被詰めろペナルティ。この手の後、**相手**が持ち駒打ちで一手詰めにできる
    /// 状態を残すことへの減点。詰みの成立条件は自玉の逃げ道と自駒（=完全既知）と
    /// 相手の持ち駒（=取られた自駒なので既知）がほぼ決めるので、相手の盤上の
    /// 支え駒が見えなくても評価できる（玉で取る以外に受けがない
    /// `MateThreat::IfSupported` を MATE_RISK_IF_SUPPORTED 倍で数える）。0 = 無効
    pub mate_risk_w: f64,
    /// 幻の詰みゲート（quest31 レビュー 2026-08-02、m027/m029 の 4一龍が発端）。
    /// 粒子上の詰み（+1000）は「真の局面がこの粒子なら勝ち」として正しいが、
    /// 較正の悪い信念の裾が詰みを主張すると全候補を乗っ取る（実測: 玉が初期配置
    /// 近辺に残る粒子質量 8〜9% だけで gain 77〜91。真の玉は既に移動済みで、
    /// 外れ枝は敵陣密集地帯で龍がタダ死にする）。詰み質量 q = 詰み粒子重み/合法重み
    /// に対し、寄与を 1000×q×(q/(q+q0)) と凸にゲートする: 裾の幻詰み（q≈0.1）は
    /// 材料スケールまで沈み、合意の詰み（q→1）はほぼ満額。0 = 従来と同一挙動
    pub mate_gate_q0: f64,
    /// **大駒の成り道**（`major_promo_path` の doc 参照、2026-08-03）。
    /// 飛・角・香の素材の成り得を成りマスまでの距離で減衰させた総和の差分。
    /// 自駒がどいて大駒の道が開くと跳ねる。0 = 従来と同一挙動
    pub major_promo_path_w: f64,
    /// **当たっている自駒の複数枚計上**（`exposed_capture_risk` の doc 参照、
    /// 2026-08-03）。上位3件を `t0 + w·t1 + w²·t2` で数える。0 = 従来の max。
    /// 相手は1手に1枚しか取れないので主項は最大値のまま置き、2枚目以降を
    /// 割り引いて足す。実用域は 0.2〜0.4（4七歩成と安い探り打ちの差が ±0.3 の
    /// スケールなので、歩1枚ぶんの当たりが効くのはこのあたり）
    pub exposed_multi_w: f64,
    /// **鉢合わせ**（`exposed_capture_risk` 内、2026-08-03）。敵歩の正面に
    /// 立っている自駒の当たりを `1 + w` 倍する。相手がその駒の位置を知らなくても
    /// 「歩を突く」という普通の手で取られるので、`knownness` では表現できない。
    /// 実測は 1.50倍（= w=0.5）で対人・アリーナとも一致。0 = 従来挙動
    pub exposed_pawn_head_w: f64,
    /// **ブラインド玉攻め加点を攻め駒の生存で割り引く**（2026-08-03）。
    /// 係数 = `1/(1 + w×max(0, 着地マスの期待被覆枚数 − 自分の守り枚数))`。
    /// 0 = 従来挙動（無防備な駒を信念上の玉の隣へ置くほど加点が最大になる）。
    /// 実用域は 0.5〜1.5（w=1 で「相手が2枚利かせて自分の紐なし」の打ちが 1/3 に）
    pub blind_attack_survive_w: f64,
    /// **taint 粒子の占有合意で打ちの反則確率を下げる**（2026-08-03、ユーザー指摘の
    /// 38手目 `S*4g` が発端）。厳密粒子が全滅した決定では `p_legal` が
    /// `prior_legal` だけで決まるが、その打ち側は**マスに依らない定数**
    /// （盤全体の平均空きマス率 q）なので、**確実に埋まっているマスへの打ちも
    /// 空きマスと同じ 0.74 で生き残る**。
    ///
    /// 実測（quest31 の38手目、後手番）: 4七には先手の歩がいて、しかも
    /// **taint 粒子は 100% それを当てている**（自分の4六歩が前進を塞いでいて、
    /// 動けば取りが発生して観測されるので論理的にも確定する）。にもかかわらず
    /// `S*4g` は p_legal=0.74 で候補上位に残り、反則を1回捨てることになる。
    ///
    /// 打ちは「占有マス = 反則」なので、taint の合意占有率をそのまま反則確率に
    /// 使える（gain 側と違い**駒種の当て違いに鈍感** = taint を信用する範囲が狭い）。
    /// **安全方向のみ**: `p_legal = min(prior, 1 − w×p_occ)` で、空きマスの
    /// 打ちを押し上げることはしない（反則マス記憶系4種が全滅した領域なので
    /// 楽観方向へは動かさない）。0 = 従来と同一挙動
    pub taint_occ_legal_w: f64,
    /// 打ちプローブの反則情報価値（quest31 レビュー 2026-08-03、m015 の
    /// 2二歩打が発端。ユーザー指摘「2二とが高いだけでなく2二歩打が低いのも問題」）。
    /// combine_score は「反則の価値 = 次善手の価値 − 反則コスト」と仮定するが、
    /// **打ちの反則は占有の確定という情報を買う**: 打ちマスが相手駒に塞がれて
    /// いて（反則枝）、かつ自分の利きが既にそのマスに当たっているなら、
    /// 次の手で確定した駒を回収できる（実戦の人間: 角がいそうな2二へ歩を打ち、
    /// 反則ならと金で角を取る。駒は失わず反則1回だけ消費）。
    /// 反則枝の期待値 w × Σ(粒子重み × 占有駒の交換価値 × 自利きあり)/全質量 を
    /// combine_score の**外側**（(1−p_legal) 側）へ加算する。攻め側の利きが
    /// 無いマスへのプローブ（ただ情報だけ）は対象外 = 打ち得スパムにならない。
    /// 王手中・taint 粒子では無効。0 = 従来と同一挙動
    pub drop_probe_w: f64,
    /// 玉隣接への無支え進入ペナルティ（quest31 レビュー 2026-08-02、F1/F2 の
    /// 共通形への対応）。粒子上の相手玉の8近傍に、自分の利きの支えが無い駒で
    /// 入る手は、王手宣言・接触で即座に存在がバレて玉や近傍の守りに回収される
    /// （実測: 直接王手の53〜56%が即取られ、大半が取り返しなし）。
    /// 該当粒子で w × 着手駒の交換価値 を引く。駒価値スケールなので
    /// 龍の突進（12）は大きく沈み、と金の進入（3.5）は軽い。
    /// mover_check_extra（全王手対象）より狭く、4七歩成のような
    /// 「安い駒の前進」への巻き添えが小さい。王手中は無効。0 = 従来と同一挙動
    pub king_adj_entry_w: f64,
    /// 自玉8近傍の「玉以外の自駒の利きが無いマス」1個あたりの減点
    /// （docs/yaneuraou-lessons.md の V4。やねうら王 Lv8 で +R35）。
    ///
    /// **ついたてと相性が良い理由**: この量は自分の駒だけで計算できるので
    /// 粒子が要らず、相手が見えなくても**正確**に測れる。既存の `hand_drop_w`
    /// （相手の持ち駒による打ち込み王手の受け入れ面積）は「打ち込まれる面積」を
    /// 測る対の項で、両方あって初めて「面積 × 支えの無さ」が表現できる。
    /// `mate.rs` の被詰めろとも噛み合う（打ち一手詰めの成立条件は、まさに
    /// 玉の近傍に支えの無いマクがあること）。0 = 無効（従来と同一挙動）
    pub king_hole_w: f64,
    /// 紐のついた自駒1点（交換価値）あたりの加点
    /// （docs/yaneuraou-lessons.md の V3。やねうら王 Lv7 で +R25、
    /// 向こうは駒価値の 0.8〜1.0% を加点している）。
    ///
    /// **ついたてでは将棋よりこの項の価値が高いはず**という理屈:
    /// 将棋は「狙われてから紐をつける」で間に合うが、ついたては相手の攻めが
    /// 見えないので狙われたことに気づけない。事前に紐がついている駒は、
    /// 気づかないまま只取られされる確率がそもそも低い。
    /// 既存の紐（`recapture_defended` / `exposed_defended`）は**すでに
    /// 攻撃されている駒にしか効かない**ので、この事前の勾配は無かった。
    /// 自駒同士の連結は完全既知なので粒子不要・ノイズゼロ
    pub link_w: f64,
    /// **盤上の自駒**の駒価値1点あたりの減点（V5。docs/yaneuraou-lessons.md 1-6）。
    /// やねうら王 Lv2 は `score -= piece_value × 104/1024`（≒10%）**だけで +R50**
    /// と、連載中2番目に大きい単独項。含意は「同じ駒なら持ち駒のほうが価値が高い」。
    ///
    /// 盤上の合計は候補間でほぼ定数なので、**順位に効くのは差分だけ**:
    /// - 打ち → 打った駒の価値ぶん盤上が増える = 打つこと自体のコスト
    /// - 成り → 増えた価値ぶんのコスト（やねうら王も同じ扱い）
    /// - 取り → 自分の盤上は変わらず持ち駒が増えるので、相対的に得になる
    ///
    /// **ついたてでは持ち駒の優位がさらに大きいはずだという理屈**: 持ち駒は
    /// (a) 相手から見えない、(b) 任意のマスに打てる（＝索敵ユニットにもなる）。
    /// **ただし逆向きの力もある**: 打ちマス反則は反則原因の最多カテゴリで、
    /// 持ち駒は「打てるマスが分からない」というついたて固有のコストを負う。
    /// どちらが勝つかは実測でしか分からないので、SPSA の範囲は**正負にまたがらせる**。
    ///
    /// 発端は watch-estimator 17手目の `L*1b`（自分のと金の直前に香を打って
    /// その香が最後まで一度も動けない手）。紐（`link_w`）と低い取られリスクが
    /// 加点する一方、**打つこと自体のコストがゼロ**だったのが症状の半分。
    /// 0 = 無効（従来と同一挙動）
    pub board_discount_w: f64,
    /// **自玉からの距離で重み付けた自駒の利き**への加点（V2。やねうら王 Lv3 は
    /// これ単独で **+R200** と連載中で最大。docs/yaneuraou-lessons.md 1-2）。
    ///
    /// 既存の `coverage_w` は「利き被覆マス数を**全マス平等に**数える」量で、
    /// SPSA が 0.0013（実質ゼロ）まで潰した。やねうら王の知見は
    /// 「利きの価値は玉からの距離に強く依存する」なので、**同じ利き情報でも
    /// 重み付けを変えれば生き返る**はずだ、という賭け。
    ///
    /// **底歩の説明が付くのがこの形の要点**: 可動性はほぼゼロでも、利きが
    /// 自玉の隣（距離1）なら重みは 0.5。隅で何も脅かしていないと金は
    /// 利きが2マスあっても両玉から遠いので小さくなる。**可動性そのものでは
    /// 「動かないが働いている駒」を取りこぼす**（ユーザー指摘、2026-07-28）。
    ///
    /// 自玉の位置は完全既知なのでこちら側は**粒子不要・ノイズゼロ**。0 = 無効
    pub effect_own_w: f64,
    /// **相手玉の信念位置からの距離で重み付けた自駒の利き**への加点（V2 の攻め側）。
    /// 相手玉の位置は粒子の信念なので、`blind_king_attack` と同じく玉位置分布を
    /// 決定点ごとに1度だけ作って使う（`opp_king_effect_weights`）。
    /// 玉位置の不確かさで自然に薄まるので、信念が割れている局面では小さくなる。
    ///
    /// やねうら王は自玉側より相手玉側をやや重く見ている（`their > our` が全距離で
    /// 一貫）。ついたて側の既存 SPSA も `pressure_w`(0.0918) > `attack_w`(0.0434) と
    /// 同じ非対称を独立に見つけているので、初期値・範囲もその想定でよい。0 = 無効
    pub effect_opp_w: f64,
    /// 紐（V3）を**守られる駒の「働き」**で重み付ける度合い。
    /// 0 = 交換価値だけで数える（従来）、1 = 完全に働きで重み付け。
    ///
    /// 発端はユーザー指摘（2026-07-28）: 「1二香車の筋の悪さは、そこにいるだけだと
    /// ほとんど価値のないと金に対して紐をつけていることにも起因する。紐をつける際に
    /// その駒の働きも考慮する必要がある」。素の交換価値だけで数えると、隅で何も
    /// していないと金に紐をつける手が、実戦的に価値のある紐と同じ重みになる。
    ///
    /// 働きは V2（玉距離重み付き利き）を流用する。**可動性ではない**のが要点で、
    /// 底歩は利き1マスでも自玉の隣なので働きを確保する（「動けない駒は価値なし」
    /// では底歩を取りこぼす）。**王手中は適用しない**（v12 の実測で「紐は王手中も
    /// ゲートしてはいけない」＝ゲートすると kakutori の反則が 7→51）。
    ///
    /// 実測（200局×2シード、vs v12）: 対照 51.5/52.0% に対し 53.5/54.3%、
    /// 反則/局 6.64 → 6.37、詰み 39 → 50。シナリオでは狙いの悪手が
    /// 20/20 → 2/20 になり、良い打ち（kakudo の R*2d、mate-net の R*7h）は維持
    pub link_work_w: f64,
    /// 働きの飽和基準。これ以上の働きがある駒は係数1（＝満額の紐）になる。
    /// `1/(1+d)` は距離1で0.5・距離8で0.11 と平坦なので、基準値の取り方で
    /// 「遊び駒」と「働いている駒」の分離度が決まる。
    /// 実測では 2.0 が最良（3.0 はシナリオでは良いが勝率が落ちる）
    pub link_work_ref: f64,
    /// **自分側の配置が過去に出現した回数**1回あたりの減点。
    ///
    /// 既存の `backtrack_penalty` / `shuffle_penalty` は**直前の1手しか見ない
    /// うえに固定値**なので、何度繰り返しても同じ額しか引かれない。実戦で
    /// 3四角↔2五角を6手繰り返した局（watch-estimator-20260728-122107 の
    /// 57〜67手目）を `rank_probe` で見ると、手戻り減点 −0.369 は効いているのに
    /// **ブラインド玉攻めボーナス +1.8 が上回っていた**: 角が動くたびに
    /// 「信念上の敵玉マスへ利きを作る手」として加点され直すため、
    /// 同じ攻めを何度でも作り直せてしまう。
    /// `endgame_push`（手戻り減点を増幅する仕組み）も ANTI_DRAW_START=60手 から
    /// しか立ち上がらないのでこの局面では効いていなかった。
    ///
    /// そこで「直前の手の逆か」ではなく**同じ配置に戻った回数**で数える。
    /// 往復を続けると 0→1→1→2→2→3… と単調に増えるので、繰り返すほど重くなる。
    /// 自分側の配置は完全既知なので粒子不要・ノイズゼロ。相手に取られたり
    /// 自分が取ったりすれば指紋が変わって回数がリセットされる ＝
    /// 「**何も起きていないのに同じ形へ戻る**」ときだけ効く。0 = 無効
    pub repeat_penalty_w: f64,
    /// **構想の読み**（自分の手 → 自分の次の手）の利得に掛ける重み。
    ///
    /// 既存の2手読みは「自分の手 → **相手の**応手」しか見ないので、
    /// 「**支えを作ってから出る**」型の組み立てを評価できない。
    /// 発端（watch-estimator-20260728-130424 の55手目、ユーザー指摘）:
    /// `9五歩打 → 9四金打` は歩で支えてから金を出す2手の構想だが、
    /// `G*9d` 単独ではリスク −3.078（他の金打ちは −0.8前後）で113位、
    /// `P*9e` も「ただ歩を打っただけ」で31位にしかならない。
    ///
    /// 相手の応手を挟まない楽観値なので、そのまま足すと攻めが軽くなりすぎる。
    /// 1未満の重みで割り引く前提。0 = 無効（従来と同一挙動）
    pub plan_w: f64,
    /// **王手の強さ**による値付け（2026-07-29、tuyoi_oote/rei2 のユーザー指導）。
    /// 王手の期待反則数は相手に残る合法解消手数 K に強く依存する
    /// （実測 800局: K≤2 の王手は直後の実反則 0.81〜1.17回/王手、
    /// K≥3 は 0.40〜0.46回 = 約2.4倍。bin/analyze の「王手の強さ」）。
    /// 粒子ごとに K = 王手後の相手の合法手数を数え、
    /// g(K) = CHECK_STRENGTH_CURVE/(1+K) − CHECK_STRENGTH_CENTER を w 倍して加算する。
    /// 中心化（平均的な王手 K≈3.7 で g≈0）してあるので、強い王手（K=1 で +0.69w）を
    /// 押し上げ、解消の多い王手（K=10 で −0.29w）を抑える**再配分**であり、
    /// 王手全体への食欲は従来どおり check_bonus / check_foul_scale の責務のまま。
    /// K は詰み判定で既に呼んでいた legal_moves() の流用なので追加コストほぼゼロ。
    /// 受け側の選択肢 N は実測でアリーナでは常に広い（N<10 出現ゼロ）ため v1 では
    /// 見ない（対人・終盤で効く可能性はメモリ strong-check-few-resolutions 参照）。
    /// 0 = 無効（従来と同一挙動）
    pub check_strength_w: f64,
    /// **逃げマス被覆**の凸ボーナス（2026-07-29、対人局レビュー
    /// human-play-review-2026-07-29 の N*6四が発端）。粒子上の相手玉の隣接マスの
    /// うち「逃げ先になり得る」（相手自身の駒に塞がれていない）のに自分の利きが
    /// 当たっていないマス数 U に対し、w × 1/(1+U) を加点する。
    ///
    /// 既存の `attack_w`（king_zone_pressure）は被覆マス数に**線形**なので、
    /// 「最後の逃げ道を塞ぐ手」（U 1→0）と「既に厚い包囲へ5本目の利きを足す手」の
    /// 増分が同じになる。王手の期待反則数は相手の解消手数 K にほぼ反比例する
    /// （実測: K≤2 の王手は実反則約2.4倍、メモリ strong-check-few-resolutions）ので、
    /// U が小さいほど1マスの限界価値が跳ねる**凸形**でなければ「強い王手が成立する
    /// 形を作る」勾配にならない。check_strength_w（王手時点の再配分）が勝率中立
    /// だったのは王手の瞬間には形が決まっているからで、これはその前段の中期項。
    /// 王手中は無効（CheckSolver の領分）。0 = 無効
    pub escape_cover_w: f64,
    /// **守り駒の捕獲**ボーナス（同レビューの 6一成銀が発端。ユーザー聞き取りで
    /// 「金取り＋守り駒削減が主目的」と確認済み）。粒子上の相手玉の8近傍にいる
    /// 相手駒を取る手へ、交換価値とは**別に**フラットな加点。玉の守りが1枚減る
    /// 価値（以後の王手が強くなる・詰み網が作りやすくなる）は素材価値に含まれない。
    /// 王手中は無効（CheckSolver の領分）。0 = 無効
    pub defender_capture_w: f64,
    /// **打ち当て露出**（2026-07-30、対人局レビュー human-play-review-2026-07-29 の
    /// 竜退避拒否 = scenarios/dragon-evac.kif が発端）。敵陣にいる自分の大駒
    /// （飛角竜馬）は、相手が歩を持った瞬間に「頭への歩打ち → 次の手で捕獲」の
    /// 的になる。ついたてでは**その歩打ちは観測されない**ので、当たりに気づいて
    /// から逃げることが原理的にできない（発端の実戦: 8七竜の頭8八へ見えない歩、
    /// bot は S*7八 の紐付けを選んで次手で竜が歩に取られた）。事前の退避だけが
    /// 対策で、これは flee-probe（bot相手の追撃実被害2.7%）とは逆に**人間・bot
    /// どちらが相手でも実行される脅威**（歩打ちは安く、頭の歩は定石）。
    ///
    /// 使う情報は全て完全既知: 自駒の位置と、相手の持ち駒
    /// （`GameModel::opponent_hand` = 取られた自駒。相手が見えない打ちで消費した
    /// 分は引けないが「脅威が既に着地している」ケースなので上界のままでよい =
    /// 発端の局面がまさにこれ）。粒子不要・ノイズゼロ（link_w / king_hole_w と
    /// 同じクラス）。
    ///
    /// 形は**露出の差分** `w × (現局面の露出 − 着手後の露出)`: 露出 = 敵陣の
    /// 大駒のうち頭（自分の前進方向1マス先）が盤内かつ自駒で塞がれていないもの
    /// の交換価値合計。差分なので大半の候補は 0（gain のゼロ点を動かさない =
    /// threat_value 差分化 −12pt の罠を回避）、退避・頭を自駒で埋める手が正、
    /// 相手が歩を持つ局面で大駒を敵陣へ突っ込む手が負になる。
    /// 王手中は無効（CheckSolver の領分）。0 = 無効
    pub drop_hit_evac_w: f64,
    /// **成りで増える利きのポテンシャル**（2026-07-30、駒種特化2項
    /// knight_bait_w / tokin_probe_w の削除で残った「歩を敵陣側へ働かせる勾配」の
    /// 穴を一般項で埋める。ユーザー方針: 成り価値は素材差でなく**成ることで
    /// 増える利きの多さ**で測る = やねうら王の利き評価（Lv3）が成りを自動で
    /// 値付けする構造の静的近似）。
    ///
    /// 未成駒ごとの「Δ利き × 減衰^(成りマスまでの手数)」の合計を、着手後 −
    /// 現局面の**差分形**で gain へ加算（`promo_potential()` の doc 参照）。
    /// 歩→と金 +5利き ≫ 香(深)→成香 +4〜5 ≫ 銀→成銀 +1 の序列が駒種
    /// ハードコードなしで出る。垂れ歩・香の成り込みが浮き、自駒に道を塞がれた
    /// 打ち（L*1b 型）は 0。検証セットは tokin-bet / lance-selfdrop /
    /// lance-tether / pawn-tether / lance-for-pawn（9b9a+ が副指標）。
    /// 王手中は無効（CheckSolver の領分）。0 = 無効
    pub promo_potential_w: f64,
    /// **持ち駒のオプション価値**（2026-07-30、観戦局レビューの悪手8件の根
    /// ①「打つより持つ」②「同じ仕事なら最安の駒で」= 持ち駒経済の一般項。
    /// promo_potential_w の対で、検証セットは lance-for-pawn（不合格計19/20が
    /// 発端）・lance-tether / pawn-tether / pawn-hoard、副指標 9b9a+）。
    ///
    /// 手持ち駒 r の保持価値 h(r) = 空きマス全体での最良打ちポテンシャル
    /// max_s pp(r, s)（pp = promo_potential と同じ「Δ利き × 減衰^成りまで手数」。
    /// 二歩・行き所は除外）。打つ手 (r, s) に **不足分 w × (h(r) − pp(r, s))** を
    /// gain から引く。最良マスへの打ち（垂れ歩など）は減点 0、ポテンシャルを
    /// 捨てる打ち（と金の裏の香 L*1b・自陣の香打ち L*9i）ほど満額に近づく。
    ///
    /// 性質: どの打ちにも**加点はしない**（安全方向のみ = 打ちを浮かせる
    /// 副作用が無い）・移動の手は 0 でゼロ点不動・自駒と持ち駒だけで決まるので
    /// 粒子不要・ノイズゼロ。成れない駒（金）は h=0 で対象外。
    /// 王手中は無効（合駒は CheckSolver の領分）。0 = 無効
    pub hand_option_w: f64,
}

/// check_strength_w の g(K) 曲線: 実測の「王手直後の実反則数」の K 依存を
/// 双曲線で近似（K=1: 1.2, K=2: 0.8, K=5: 0.4 — 実測バケット平均に整合）
const CHECK_STRENGTH_CURVE: f64 = 2.4;
/// g(K) の中心化定数（平均的な王手 K≈3.7 で g≈0 になるよう
/// CHECK_STRENGTH_CURVE/(1+3.7) を丸めた値）
const CHECK_STRENGTH_CENTER: f64 = 0.51;

impl Default for EvalParams {
    fn default() -> Self {
        // SPSA第2ラウンドの収束点（2026-07-14、60反復×2×40局 vs estimator_v5、
        // 共通乱数法・tuning/tune-round2.jsonl、最終中心点の追加評価 score=0.675）。
        // 第1ラウンド（2026-07-11）からの主な動き: check_bonus 大幅減
        // （0.75→0.16。王手自体より check_foul_scale 側=相手の反則蓄積で加点）、
        // prior_weight_degen 増（4.7→8.0、退化時は事前をさらに信頼）、
        // threat_w 増（0.31→0.46）、coverage_w はほぼゼロへ
        // （利き被覆の一律加点は効かず、と金・王探しの個別項が残った）
        EvalParams {
            check_bonus: 0.1619,
            check_foul_scale: 0.0983,
            mover_w_captured: 0.8042,
            mover_w_quiet: 0.7312,
            mover_check_extra: 0.0622,
            capture_reveal_risk: 0.1313,
            camp_known_quiet: 0.4472,
            camp_scale: 0.1252,
            exposed_base: 0.4576,
            exposed_known: 0.1659,
            home_knownness: 0.0027,
            recapture_defended: 0.4692,
            exposed_defended: 0.3031,
            attack_w: 0.0434,
            pressure_w: 0.0918,
            foul_cost_base: 0.637,
            foul_cost_pow: 1.3331,
            advance_w: 0.0699,
            promote_bias: 0.1466,
            drop_bias: 0.2616,
            prior_weight: 4.9065,
            prior_weight_degen: 7.9515,
            threat_w: 0.4586,
            info_bonus: 0.64,
            big_home_penalty: 0.3156,
            hand_drop_w: 0.0757,
            backtrack_penalty: 0.3685,
            shuffle_penalty: 0.2996,
            soft_decay: 0.6753,
            king_probe_bonus: 0.2451,
            coverage_w: 0.0013,
            depth2_replace: 0.6205,
            // 2手読みの楽観上限（2026-08-03、F3）。0 = 従来と同一挙動。
            // アリーナがペア比較で中立〜負（cap=1.0 で −1.5pt、0.5 で −6.5pt）
            // だったため 0 のまま
            depth2_optimism_cap: 0.0,
            // taint 占有合意による打ちの反則回避（2026-08-03）。0 = 従来と同一挙動
            taint_occ_legal_w: 0.0,
            // 大駒の成り道（2026-08-03）。0 = 従来と同一挙動
            major_promo_path_w: 0.0,
            exposed_multi_w: 0.0,
            exposed_pawn_head_w: 0.0,
            blind_attack_survive_w: 0.0,
            depth2_check_pen: 0.178,
            depth2_recap_discount: 0.7612,
            // 反則経済の新項（2026-07-16、オラクル測定で36ptの伸びしろを確認後に追加）。
            // 0 = 従来と同一挙動。SPSA第4ラウンド（反則経済マスク）の調整対象
            foul_diff_pow: 0.0,
            check_limit_accel: 0.0,
            // valueネット統合（2026-07-22、NN段階③フェーズ2）。NNの候補間スコア差は
            // 0.1〜0.2程度（pairwise margin=0.1で学習）なので、6.0で0.6〜1.2歩相当。
            // w選定スイープ（w=3/6/10 × 5シナリオ）: w=3はgold-checkの悪手を
            // 変えられず（17/20）、w=6で2/20に反転。王手中の反則増（dragon-check-
            // drop）は you_in_check ゲートで遮断したうえでの採用値
            value_nn_w: 6.0,
            // 仮説条件付き除去期待値（2026-07-24、p_legalフロアの置き換え）。
            // w スイープ（kakutori 捕獲率: w=0.5で10/20, w=1で19/20, w=2で18/20 /
            // dragon-check-drop: w=1で玉逃げ20/20維持・反則18→28 /
            // keima: w=1で捕獲20/20維持）から採用。挙動は「捕獲プローブ→
            // 反則観測→仮説減衰→真の捕獲」の系列で、プローブ反則が少し増える
            // 対価はアリーナの反則経済で判定した
            checker_removal_w: 1.0,
            // 捕獲賭け分散（2026-07-24、play-estimator-20260724 16手目
            // 「8八と > 8八歩打」レビューを受けて追加）。0 = 従来と同一挙動。
            // w スイープ: tokin-bet で 8g8h(と金の五分賭け) は w=1で gain 6.4→4.5
            // (1位のまま)、w=2で2.5(P*8fと拮抗1位)、w=2.5で0.8(44位まで沈み
            // P*8f を選択 = 人間レビューの意図どおり)。アリーナ104局 vs v10:
            // w=1 59.2%±9.5 / w=2 62.5%±9.3。keima 20/20・kakutori 19/20
            // （王手中ゲートで不変）を確認して 2.5 を採用
            capture_bet_var_w: 2.5,
            // 詰めろ生成（2026-07-25、対人局 58手目 N*6六 のレビューを受けて追加）。
            // 0 = 従来と同一挙動。未調整の新項なので w スイープで決める
            mate_threat_w: 0.0,
            mate_risk_w: 0.0,
            // 幻の詰みゲート（2026-08-02、quest31 レビュー）。0 = 従来と同一挙動。
            // 既定 4.0 はシナリオ実測で採用（2026-08-03、シナリオ重視のユーザー方針）:
            // quest31-m027 の幻詰み 4一龍 20/20→2/20・m029 の龍 19/20→6/20、
            // 回帰なし（m026/m028 ベースライン同等・ansatsu 19〜20/20・keima 20/20・
            // dragon-check-drop / king-evade / mate-net-attack / tokin-bet 維持）。
            // q=0.3 の詰みは 26 残り材料項より優位、q=0.1 の幻詰みは 7 で材料スケール
            mate_gate_q0: 4.0,
            // 玉隣接への無支え進入（2026-08-02、同上）。0 = 従来と同一挙動
            king_adj_entry_w: 0.0,
            // 打ちプローブの反則情報価値（2026-08-03）。0 = 従来と同一挙動。
            // 形は p_occ²(1−p_occ)×予算² ゲート（evaluate の foul_probe 参照。
            // 変遷: p_occ² 版 w=1.6 はシナリオ全緑だがアリーナ 36.8%・反則/局
            // 6.4→8.2 でレース負け。占有確定マスの再プローブループと中終盤の
    // 乱発が原因で、(1−p_occ)・予算² の2ゲートを追加して w を再較正）。
            // quest31-m015 の 2二歩打（人間の実戦手）0/20→20/20 と
            // m026/m028 の 4七歩成 20/20 維持を両立する値
            drop_probe_w: 4.5,
            // 自玉8近傍の穴（2026-07-26、docs/yaneuraou-lessons.md の V4）。
            // 0 = 従来と同一挙動。未調整の新項なので w スイープで決める
            king_hole_w: 0.0,
            // 予防的な紐（2026-07-26、V3）。やねうら王 Lv7 の比率（駒価値の 0.8%
            // ＝ w 0.008）から始めたが、w スイープの実測はその**約10倍**が最適だった。
            // 2シード×200局の合算で w=0.06 が 61.2%±4.8 / w=0.12 が 59.8%±4.8
            // （vs v11。単一シードでは 0.06 と 0.12 の順位が入れ替わるので
            // 「0.12 が頂点」は取らない）。w=0.25 以上は 52.3%→46.5% と崩れる:
            // 反則は単調に減り続ける（6.48→5.18）のに勝率が落ちるのは、守りが
            // 目的化して攻めなくなり手数だけ伸びる変質（85手→112手）。
            // 平坦部の下端を採るのは、手数が短く引き分け化のリスクが小さいため
            link_w: 0.06,
            board_discount_w: 0.0,
            effect_own_w: 0.0,
            effect_opp_w: 0.0,
            // 2026-07-28 採用（200局×2シードで再現。詳細は link_work_w の doc）
            link_work_w: 1.0,
            link_work_ref: 2.0,
            repeat_penalty_w: 0.3,
            plan_w: 0.0,
            // 王手の強さ（2026-07-29）。未調整の新項なので w スイープで決める
            check_strength_w: 0.0,
            // 逃げマス被覆・守り駒捕獲（2026-07-29、対人局レビュー）。
            // 未調整の新項なので w スイープで決める
            escape_cover_w: 0.0,
            defender_capture_w: 0.0,
            // 打ち当て露出（2026-07-30、dragon-evac）。未調整の新項なので
            // w スイープで決める
            drop_hit_evac_w: 0.0,
            // 成りで増える利きのポテンシャル（2026-07-30 採用）。w スイープの実測で
            // 0.2（0.5 は垂れ歩の常用化・成り済み駒のマス好み副作用で過剰）。
            // シナリオ大幅改善・ペアアリーナ3シード中立で採用（CLAUDE.md 参照）
            promo_potential_w: 0.2,
            // 持ち駒のオプション価値（2026-07-30）。未調整の新項なので
            // w スイープで決める
            hand_option_w: 0.0,
        }
    }
}

/// SPSA用のパラメータ仕様（名前と探索範囲）。to_vec/from_vec と同じ順序
pub struct ParamSpec {
    pub name: &'static str,
    pub lo: f64,
    pub hi: f64,
}

impl EvalParams {
    pub const SPECS: [ParamSpec; 65] = [
        ParamSpec {
            name: "check_bonus",
            lo: 0.0,
            hi: 3.0,
        },
        ParamSpec {
            name: "check_foul_scale",
            lo: 0.0,
            hi: 0.5,
        },
        ParamSpec {
            name: "mover_w_captured",
            lo: 0.0,
            hi: 1.5,
        },
        ParamSpec {
            name: "mover_w_quiet",
            lo: 0.0,
            hi: 1.5,
        },
        ParamSpec {
            name: "mover_check_extra",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            name: "capture_reveal_risk",
            lo: 0.0,
            hi: 0.6,
        },
        ParamSpec {
            name: "camp_known_quiet",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            name: "camp_scale",
            lo: 0.0,
            hi: 3.0,
        },
        ParamSpec {
            name: "exposed_base",
            lo: 0.0,
            hi: 1.5,
        },
        ParamSpec {
            name: "exposed_known",
            lo: 0.0,
            hi: 1.5,
        },
        ParamSpec {
            name: "home_knownness",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            name: "recapture_defended",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            name: "exposed_defended",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            name: "attack_w",
            lo: 0.0,
            hi: 0.5,
        },
        ParamSpec {
            name: "pressure_w",
            lo: 0.0,
            hi: 0.6,
        },
        ParamSpec {
            name: "foul_cost_base",
            lo: 0.2,
            hi: 6.0,
        },
        ParamSpec {
            name: "foul_cost_pow",
            lo: 0.5,
            hi: 3.0,
        },
        ParamSpec {
            name: "advance_w",
            lo: -0.1,
            hi: 0.3,
        },
        ParamSpec {
            name: "promote_bias",
            lo: -0.2,
            hi: 0.6,
        },
        ParamSpec {
            name: "drop_bias",
            lo: -0.5,
            hi: 0.3,
        },
        ParamSpec {
            name: "prior_weight",
            lo: 0.5,
            hi: 16.0,
        },
        ParamSpec {
            name: "prior_weight_degen",
            lo: 0.0,
            hi: 32.0,
        },
        ParamSpec {
            name: "threat_w",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            name: "info_bonus",
            lo: 0.0,
            hi: 2.0,
        },
        ParamSpec {
            name: "big_home_penalty",
            lo: 0.0,
            hi: 1.5,
        },
        ParamSpec {
            name: "hand_drop_w",
            lo: 0.0,
            hi: 0.5,
        },
        ParamSpec {
            name: "backtrack_penalty",
            lo: 0.0,
            hi: 1.5,
        },
        ParamSpec {
            name: "shuffle_penalty",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            name: "soft_decay",
            lo: 0.05,
            hi: 1.0,
        },
        ParamSpec {
            name: "king_probe_bonus",
            lo: 0.0,
            hi: 1.5,
        },
        ParamSpec {
            name: "coverage_w",
            lo: 0.0,
            hi: 0.1,
        },
        ParamSpec {
            name: "depth2_replace",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            name: "depth2_check_pen",
            lo: 0.0,
            hi: 1.5,
        },
        ParamSpec {
            name: "depth2_recap_discount",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            name: "foul_diff_pow",
            lo: 0.0,
            hi: 3.0,
        },
        ParamSpec {
            name: "check_limit_accel",
            lo: 0.0,
            hi: 3.0,
        },
        ParamSpec {
            name: "value_nn_w",
            lo: 0.0,
            hi: 10.0,
        },
        ParamSpec {
            name: "checker_removal_w",
            lo: 0.0,
            hi: 2.0,
        },
        ParamSpec {
            name: "capture_bet_var_w",
            lo: 0.0,
            hi: 3.0,
        },
        ParamSpec {
            name: "mate_threat_w",
            lo: 0.0,
            hi: 6.0,
        },
        ParamSpec {
            name: "mate_risk_w",
            lo: 0.0,
            hi: 6.0,
        },
        ParamSpec {
            name: "king_hole_w",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            name: "link_w",
            // 上限は実測の崩れる手前まで（0.25 で 52.3%、0.5 で 46.5%）。
            // やねうら王の比率 0.008 を上限にすると最適点が範囲外になる
            lo: 0.0,
            hi: 0.25,
        },
        ParamSpec {
            // V2。利き1本あたりの加点なので、被覆マス数（40〜80）× 距離重み
            // （0.1〜0.5）= 10〜30 スケールの量に掛かる。coverage_w が 0.0013 に
            // 潰された前例があるので範囲は小さめから
            name: "effect_own_w",
            lo: 0.0,
            hi: 0.2,
        },
        ParamSpec {
            name: "effect_opp_w",
            lo: 0.0,
            hi: 0.2,
        },
        ParamSpec {
            // 0 = 交換価値だけ、1 = 完全に働きで重み付け。補間なので [0,1]
            name: "link_work_w",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            // 飽和基準。小さいほど「遊び駒」の判定が緩い
            name: "link_work_ref",
            lo: 0.5,
            hi: 4.0,
        },
        ParamSpec {
            // 出現回数1回あたり。ブラインド玉攻めボーナス（実測 +1.8）を
            // 数回の往復で上回れる水準まで範囲を取る
            name: "repeat_penalty_w",
            lo: 0.0,
            hi: 2.0,
        },
        ParamSpec {
            // 相手の応手を挟まない楽観値なので 1 未満で割り引く前提
            name: "plan_w",
            lo: 0.0,
            hi: 0.5,
        },
        ParamSpec {
            name: "board_discount_w",
            // やねうら王は 104/1024 ≒ 0.102。ついたては持ち駒優位（不可視・
            // 索敵）と打ちマス反則（打てるマスが分からない）が逆向きに働くので、
            // **符号を固定せず正負にまたがらせる**（1-6 の指摘）
            lo: -0.2,
            hi: 0.2,
        },
        ParamSpec {
            // g(K) の振れ幅は ±0.7 なので、w=3 で約 ±2歩相当の再配分
            name: "check_strength_w",
            lo: 0.0,
            hi: 3.0,
        },
        ParamSpec {
            // 値域は 1/(1+U) ∈ [0.11, 1.0]。w=2 で「最後の逃げ道を塞ぐ手」が
            // 約1歩相当になるスケール
            name: "escape_cover_w",
            lo: 0.0,
            hi: 3.0,
        },
        ParamSpec {
            // 捕獲1回あたりのフラット加点（交換価値の外）。歩1〜2枚相当まで
            name: "defender_capture_w",
            lo: 0.0,
            hi: 3.0,
        },
        ParamSpec {
            // 差分の振れ幅は大駒1枚の交換価値（8〜10.75）。w=1 は「留まれば
            // 確実に死ぬ」相当なので過大、実用域は 0.1〜0.5 を想定
            name: "drop_hit_evac_w",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            // Δ利きの振れ幅は歩→と金の +5 × 減衰。垂れ歩(d=1)で 2.5 なので
            // w=0.2 で旧 tokin_probe_w（0.2×1.0）と同スケール
            name: "promo_potential_w",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            // 不足分の振れ幅は最良打ちポテンシャル（歩→と金で 2〜2.5）。
            // w=0.5 で悪い打ちが歩1枚強沈む。実用域は 0.2〜1.0 を想定
            name: "hand_option_w",
            lo: 0.0,
            hi: 2.0,
        },
        ParamSpec {
            // 幻の詰みゲートの半飽和点。q0=4 で q=0.1 の幻詰みが
            // 材料スケール（〜7）まで沈み、q=0.3 の詰みは 26 残る。
            // 既定 4.0 が中心に来るよう範囲は 0〜8
            name: "mate_gate_q0",
            lo: 0.0,
            hi: 8.0,
        },
        ParamSpec {
            // 着手駒の交換価値（3.5〜12）× 玉隣接粒子質量に掛かる。
            // w=1 で「隣接粒子では確実に取られる」相当なので実用域は 0.3〜0.8
            name: "king_adj_entry_w",
            lo: 0.0,
            hi: 1.5,
        },
        ParamSpec {
            // 占有駒の交換価値 × p_occ²(1−p_occ) × 予算² に掛かる。
            // p_occ(1−p_occ) ゲートのピークは p=2/3 で ≈0.15 なので、
            // 実効値は交換価値の w×0.15 程度。w=4.5 で 2二歩打（p=0.65・角8）が
            // +5.3 = 2二と直行を上回る水準
            name: "drop_probe_w",
            lo: 0.0,
            hi: 8.0,
        },
        ParamSpec {
            // 実効リスクが静的リスクの何割を下回れないか。1 で楽観禁止
            name: "depth2_optimism_cap",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            // 1 で taint の合意占有率をそのまま反則確率に使う（安全方向のみ）
            name: "taint_occ_legal_w",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            // 差分の振れ幅は龍−飛=2.5 × 減衰の変化（最大 0.5 程度）。
            // w=1 で「次手で成れる形を作る手」が約1歩強
            name: "major_promo_path_w",
            lo: 0.0,
            hi: 3.0,
        },
        ParamSpec {
            // 2枚目以降の当たりの割引率。1 だと3枚まで満額で数えて
            // 「相手は1手に1枚しか取れない」に反するので上限は 0.8
            name: "exposed_multi_w",
            lo: 0.0,
            hi: 0.8,
        },
        ParamSpec {
            // 実測は 1.50倍 = w 0.5。倍率なので範囲は 0〜1.5（2.5倍まで）
            name: "exposed_pawn_head_w",
            lo: 0.0,
            hi: 1.5,
        },
        ParamSpec {
            // 1/(1+w×不足枚数) の形。w=1 で2枚不足なら 1/3、w=3 で 1/7
            name: "blind_attack_survive_w",
            lo: 0.0,
            hi: 3.0,
        },
    ];

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.check_bonus,
            self.check_foul_scale,
            self.mover_w_captured,
            self.mover_w_quiet,
            self.mover_check_extra,
            self.capture_reveal_risk,
            self.camp_known_quiet,
            self.camp_scale,
            self.exposed_base,
            self.exposed_known,
            self.home_knownness,
            self.recapture_defended,
            self.exposed_defended,
            self.attack_w,
            self.pressure_w,
            self.foul_cost_base,
            self.foul_cost_pow,
            self.advance_w,
            self.promote_bias,
            self.drop_bias,
            self.prior_weight,
            self.prior_weight_degen,
            self.threat_w,
            self.info_bonus,
            self.big_home_penalty,
            self.hand_drop_w,
            self.backtrack_penalty,
            self.shuffle_penalty,
            self.soft_decay,
            self.king_probe_bonus,
            self.coverage_w,
            self.depth2_replace,
            self.depth2_check_pen,
            self.depth2_recap_discount,
            self.foul_diff_pow,
            self.check_limit_accel,
            self.value_nn_w,
            self.checker_removal_w,
            self.capture_bet_var_w,
            self.mate_threat_w,
            self.mate_risk_w,
            self.king_hole_w,
            self.link_w,
            self.effect_own_w,
            self.effect_opp_w,
            self.link_work_w,
            self.link_work_ref,
            self.repeat_penalty_w,
            self.plan_w,
            self.board_discount_w,
            self.check_strength_w,
            self.escape_cover_w,
            self.defender_capture_w,
            self.drop_hit_evac_w,
            self.promo_potential_w,
            self.hand_option_w,
            self.mate_gate_q0,
            self.king_adj_entry_w,
            self.drop_probe_w,
            self.depth2_optimism_cap,
            self.taint_occ_legal_w,
            self.major_promo_path_w,
            self.exposed_multi_w,
            self.exposed_pawn_head_w,
            self.blind_attack_survive_w,
        ]
    }

    pub fn from_vec(v: &[f64]) -> EvalParams {
        assert_eq!(v.len(), Self::SPECS.len());
        EvalParams {
            check_bonus: v[0],
            check_foul_scale: v[1],
            mover_w_captured: v[2],
            mover_w_quiet: v[3],
            mover_check_extra: v[4],
            capture_reveal_risk: v[5],
            camp_known_quiet: v[6],
            camp_scale: v[7],
            exposed_base: v[8],
            exposed_known: v[9],
            home_knownness: v[10],
            recapture_defended: v[11],
            exposed_defended: v[12],
            attack_w: v[13],
            pressure_w: v[14],
            foul_cost_base: v[15],
            foul_cost_pow: v[16],
            advance_w: v[17],
            promote_bias: v[18],
            drop_bias: v[19],
            prior_weight: v[20],
            prior_weight_degen: v[21],
            threat_w: v[22],
            info_bonus: v[23],
            big_home_penalty: v[24],
            hand_drop_w: v[25],
            backtrack_penalty: v[26],
            shuffle_penalty: v[27],
            soft_decay: v[28],
            king_probe_bonus: v[29],
            coverage_w: v[30],
            depth2_replace: v[31],
            depth2_check_pen: v[32],
            depth2_recap_discount: v[33],
            foul_diff_pow: v[34],
            check_limit_accel: v[35],
            value_nn_w: v[36],
            checker_removal_w: v[37],
            capture_bet_var_w: v[38],
            mate_threat_w: v[39],
            mate_risk_w: v[40],
            king_hole_w: v[41],
            link_w: v[42],
            effect_own_w: v[43],
            effect_opp_w: v[44],
            link_work_w: v[45],
            link_work_ref: v[46],
            repeat_penalty_w: v[47],
            plan_w: v[48],
            board_discount_w: v[49],
            check_strength_w: v[50],
            escape_cover_w: v[51],
            defender_capture_w: v[52],
            drop_hit_evac_w: v[53],
            promo_potential_w: v[54],
            hand_option_w: v[55],
            mate_gate_q0: v[56],
            king_adj_entry_w: v[57],
            drop_probe_w: v[58],
            depth2_optimism_cap: v[59],
            taint_occ_legal_w: v[60],
            major_promo_path_w: v[61],
            exposed_multi_w: v[62],
            exposed_pawn_head_w: v[63],
            blind_attack_survive_w: v[64],
        }
    }
}

/// 観測履歴から相手局面を推定して指す戦略。
///
/// 候補手（自分に見える範囲の疑似合法手）を、推定粒子の平均で評価する:
/// - 駒得の期待値（その粒子でそのマスに相手駒がいるか）
/// - 反則確率（粒子上で非合法な割合）× 反則コスト（残り反則数が減るほど高い）
/// - 指した直後に取り返されるリスク（粒子上での相手の即時駒取り）
/// - 王手・詰みボーナス
#[derive(Clone)]
pub struct EstimatorStrategy {
    est: Option<Estimator>,
    book: Option<OpeningBook>,
    /// Some なら定跡をこのラインに固定する（定跡特化チューニング用）
    book_line: Option<usize>,
    params: EvalParams,
    /// 思考予算に応じた粒子数・読み幅（TSUITATE_THINK_BUDGET_MS 由来）
    budget: SearchBudget,
    /// Some なら推定器・定跡選択・タイブレークの乱数をこのシードから導出する
    /// （SPSA の共通乱数法用。None は従来どおりエントロピー由来）
    seed: Option<u64>,
    /// 評価タイブレーク用の乱数（seed があれば決定論的）
    rng: StdRng,
    /// 直近の choose 時点の内部状態（記録用）
    last_debug: Option<serde_json::Value>,
    /// 直近の choose 時点の全候補評価（スコア降順、scenario-gui 用）
    last_ranking: Option<Vec<CandidateScore>>,
}

impl EstimatorStrategy {
    pub fn new() -> Self {
        Self::with_params(EvalParams::default())
    }

    /// 内部推定器への参照（scenario_core の継ぎ足し等価性テスト用）
    pub fn estimator(&self) -> Option<&Estimator> {
        self.est.as_ref()
    }

    /// パラメータを差し替えて作る（bin/tune.rs のSPSA評価用）
    pub fn with_params(params: EvalParams) -> Self {
        Self::with_params_line_seed(params, None, None)
    }

    /// パラメータと定跡ライン固定を指定して作る（定跡特化チューニング用）
    pub fn with_params_and_line(params: EvalParams, book_line: Option<usize>) -> Self {
        Self::with_params_line_seed(params, book_line, None)
    }

    /// シードつきで作る（SPSA の f+/f− 評価で対局条件を揃える共通乱数法用）
    pub fn with_params_line_seed(
        params: EvalParams,
        book_line: Option<usize>,
        seed: Option<u64>,
    ) -> Self {
        let params = apply_env_param_overrides(params);
        EstimatorStrategy {
            est: None,
            book: None,
            book_line,
            params,
            budget: SearchBudget::from_ms(think_budget_ms()),
            seed,
            rng: match seed {
                Some(s) => StdRng::seed_from_u64(s ^ 0xA5A5_5A5A_DEAD_BEEF),
                None => StdRng::seed_from_u64(rand::rng().random()),
            },
            last_debug: None,
            last_ranking: None,
        }
    }
}

/// TSUITATE_*_W 系 env による EvalParams の上書き（運用ノブ。デプロイ時の
/// 切り戻し・w スイープ用）。SPSA（with_params 経由）でも env が設定されて
/// いればそちらを優先する。bin/tune はこの関数で「調整対象ノブが env に
/// 潰されていないか」を起動時に検査する
pub fn apply_env_param_overrides(params: EvalParams) -> EvalParams {
    // valueネット重みの運用ノブ（デプロイ時の切り戻し・アブレーション用）。
    // SPSA（with_params 経由）でも env が設定されていればそちらを優先する
    let params = match std::env::var("TSUITATE_VALUE_NN_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
    {
        Some(w) => EvalParams {
            value_nn_w: w,
            ..params
        },
        None => params,
    };
    // 除去期待値項の運用ノブ（w スイープ・切り戻し用）
    let params = match std::env::var("TSUITATE_CHECKER_REMOVAL_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
    {
        Some(w) => EvalParams {
            checker_removal_w: w,
            ..params
        },
        None => params,
    };
    // 捕獲賭け分散ペナルティの運用ノブ（w スイープ・切り戻し用）
    let params = match std::env::var("TSUITATE_CAPTURE_BET_VAR_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
    {
        Some(w) => EvalParams {
            capture_bet_var_w: w,
            ..params
        },
        None => params,
    };
    // 詰めろ生成ボーナスの運用ノブ（w スイープ・切り戻し用）
    let params = match std::env::var("TSUITATE_MATE_THREAT_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
    {
        Some(w) => EvalParams {
            mate_threat_w: w,
            ..params
        },
        None => params,
    };
    // 被詰めろペナルティの運用ノブ（w スイープ・切り戻し用）
    let params = match std::env::var("TSUITATE_MATE_RISK_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
    {
        Some(w) => EvalParams {
            mate_risk_w: w,
            ..params
        },
        None => params,
    };
    // 自玉8近傍の穴の運用ノブ（w スイープ・切り戻し用）
    let params = match std::env::var("TSUITATE_KING_HOLE_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
    {
        Some(w) => EvalParams {
            king_hole_w: w,
            ..params
        },
        None => params,
    };
    // 予防的な紐（V3）の運用ノブ（w スイープ・切り戻し用。0 で従来挙動）
    let params = match std::env::var("TSUITATE_LINK_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
    {
        Some(w) => EvalParams { link_w: w, ..params },
        None => params,
    };
    // 紐の働き重み付け（V3 の拡張）の運用ノブ（w スイープ・切り戻し用）
    let params = match std::env::var("TSUITATE_LINK_WORK_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
    {
        Some(w) => EvalParams {
            link_work_w: w,
            ..params
        },
        None => params,
    };
    let params = match std::env::var("TSUITATE_LINK_WORK_REF")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
    {
        Some(r) => EvalParams {
            link_work_ref: r,
            ..params
        },
        None => params,
    };
    // 王手の強さ（解消手数Kによる値付け）の運用ノブ（w スイープ・切り戻し用）
    let params = match std::env::var("TSUITATE_CHECK_STRENGTH_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite())
    {
        Some(w) => EvalParams {
            check_strength_w: w,
            ..params
        },
        None => params,
    };
    // 王手の自己露見リスク（王手宣言で位置がバレて着手駒が取られる）の
    // 運用ノブ。既定 0.0622 は実測（直接王手の53〜56%が即取られ）に対して
    // 過小の疑いがあり、quest31-m021 の 4一と（幻の金への王手込み突進）の
    // スイープ用に env を開ける
    let params = match std::env::var("TSUITATE_MOVER_CHECK_EXTRA")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        Some(w) => EvalParams {
            mover_check_extra: w,
            ..params
        },
        None => params,
    };
    // 大駒の成り道（0 で従来挙動）
    let params = match std::env::var("TSUITATE_MAJOR_PROMO_PATH_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        Some(w) => EvalParams {
            major_promo_path_w: w,
            ..params
        },
        None => params,
    };
    // ブラインド玉攻め加点の生存割引（0 で従来挙動）
    let params = match std::env::var("TSUITATE_BLIND_ATTACK_SURVIVE_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        Some(w) => EvalParams {
            blind_attack_survive_w: w,
            ..params
        },
        None => params,
    };
    // 鉢合わせ（敵歩の正面。0 で従来挙動）
    let params = match std::env::var("TSUITATE_EXPOSED_PAWN_HEAD_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        Some(w) => EvalParams {
            exposed_pawn_head_w: w,
            ..params
        },
        None => params,
    };
    // 位置を知られている駒の当たり重み（既定 0.1659。較正の実測は
    // 「知られている駒は知られていない駒の 6.6〜7.9倍取られる」= 大幅に不足。
    // 既定は変えずスイープ経路だけ用意する。bin/collision_probe 参照）
    let params = match std::env::var("TSUITATE_EXPOSED_KNOWN")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        Some(w) => EvalParams {
            exposed_known: w,
            ..params
        },
        None => params,
    };
    // 当たっている自駒の複数枚計上（0 で従来の max）
    let params = match std::env::var("TSUITATE_EXPOSED_MULTI_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        Some(w) => EvalParams {
            exposed_multi_w: w,
            ..params
        },
        None => params,
    };
    // taint 占有合意による打ちの反則回避（0 で従来挙動）
    let params = match std::env::var("TSUITATE_TAINT_OCC_LEGAL_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        Some(w) => EvalParams {
            taint_occ_legal_w: w,
            ..params
        },
        None => params,
    };
    // 2手読みの楽観置き換えの上限（F3。0 で従来挙動）
    let params = match std::env::var("TSUITATE_DEPTH2_OPTIMISM_CAP")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        Some(w) => EvalParams {
            depth2_optimism_cap: w,
            ..params
        },
        None => params,
    };
    // 打ちプローブの反則情報価値（w スイープ・切り戻し用。0 で従来挙動）
    let params = match std::env::var("TSUITATE_DROP_PROBE_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        Some(w) => EvalParams {
            drop_probe_w: w,
            ..params
        },
        None => params,
    };
    // 玉隣接への無支え進入ペナルティ（w スイープ・切り戻し用。0 で従来挙動）
    let params = match std::env::var("TSUITATE_KING_ADJ_ENTRY_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        Some(w) => EvalParams {
            king_adj_entry_w: w,
            ..params
        },
        None => params,
    };
    // 幻の詰みゲート（w スイープ・切り戻し用。0 で従来挙動）
    let params = match std::env::var("TSUITATE_MATE_GATE_Q0")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        Some(w) => EvalParams {
            mate_gate_q0: w,
            ..params
        },
        None => params,
    };
    // 構想の読み（自分の手 → 自分の次の手）の運用ノブ（0 で従来挙動）
    let params = match std::env::var("TSUITATE_PLAN_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        Some(w) => EvalParams { plan_w: w, ..params },
        None => params,
    };
    // 同じ自陣形への往復の累積減点（0 で従来挙動）
    let params = match std::env::var("TSUITATE_REPEAT_PENALTY_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        Some(w) => EvalParams {
            repeat_penalty_w: w,
            ..params
        },
        None => params,
    };
    // 玉距離重み付き利き（V2）の運用ノブ（w スイープ用。0 で従来挙動）
    let params = match std::env::var("TSUITATE_EFFECT_OWN_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite())
    {
        Some(w) => EvalParams {
            effect_own_w: w,
            ..params
        },
        None => params,
    };
    let params = match std::env::var("TSUITATE_EFFECT_OPP_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite())
    {
        Some(w) => EvalParams {
            effect_opp_w: w,
            ..params
        },
        None => params,
    };
    // 盤上駒の減価（V5）の運用ノブ（w スイープ用。0 で従来挙動。
    // やねうら王の比率は 0.102。負の値も許す＝持ち駒より盤上を好む側）
    let params = match std::env::var("TSUITATE_BOARD_DISCOUNT_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite())
    {
        Some(w) => EvalParams {
            board_discount_w: w,
            ..params
        },
        None => params,
    };
    // 逃げマス被覆（凸）の運用ノブ（w スイープ用。0 で従来挙動）
    let params = match std::env::var("TSUITATE_ESCAPE_COVER_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        Some(w) => EvalParams {
            escape_cover_w: w,
            ..params
        },
        None => params,
    };
    // 守り駒捕獲ボーナスの運用ノブ（w スイープ用。0 で従来挙動）
    let params = match std::env::var("TSUITATE_DEFENDER_CAPTURE_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        Some(w) => EvalParams {
            defender_capture_w: w,
            ..params
        },
        None => params,
    };
    // 打ち当て露出の運用ノブ（w スイープ用。0 で従来挙動）
    let params = match std::env::var("TSUITATE_DROP_HIT_EVAC_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        Some(w) => EvalParams {
            drop_hit_evac_w: w,
            ..params
        },
        None => params,
    };
    // 成りポテンシャルの運用ノブ（w スイープ用。0 で従来挙動）
    let params = match std::env::var("TSUITATE_PROMO_POTENTIAL_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        Some(w) => EvalParams {
            promo_potential_w: w,
            ..params
        },
        None => params,
    };
    // 持ち駒オプション価値の運用ノブ（w スイープ用。0 で従来挙動）
    let params = match std::env::var("TSUITATE_HAND_OPTION_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        Some(w) => EvalParams {
            hand_option_w: w,
            ..params
        },
        None => params,
    };
    params
}

impl Default for EstimatorStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl Strategy for EstimatorStrategy {
    fn clone_boxed(&self) -> Option<Box<dyn Strategy>> {
        Some(Box::new(self.clone()))
    }

    fn prewarm(&mut self, view: &PlayerView, log: &ObservationLog) {
        let budget = self.budget;
        let seed = self.seed;
        let est = self.est.get_or_insert_with(|| match seed {
            Some(s) => Estimator::with_seed_and_scale(view.your_color, s, budget.scale),
            None => Estimator::with_scale(view.your_color, budget.scale),
        });
        est.update(log);
    }

    fn choose(
        &mut self,
        view: &PlayerView,
        log: &ObservationLog,
        foul_tried: &HashSet<String>,
    ) -> Option<String> {
        let budget = self.budget;
        let seed = self.seed;
        // 定跡・候補ゼロで早期 return したとき前の手番のランキングが残らないように
        self.last_ranking = None;
        let est = self.est.get_or_insert_with(|| match seed {
            Some(s) => Estimator::with_seed_and_scale(view.your_color, s, budget.scale),
            None => Estimator::with_scale(view.your_color, budget.scale),
        });
        est.update(log);

        // 序盤定跡（静かな間だけ）。ブック中も推定器の update は回して粒子を保つ
        let book_line = self.book_line;
        let book = self.book.get_or_insert_with(|| match (book_line, seed) {
            (Some(idx), _) => OpeningBook::with_line(view.your_color, idx),
            (None, Some(s)) => OpeningBook::with_seed(view.your_color, s),
            (None, None) => OpeningBook::new(view.your_color),
        });
        if let Some(usi) = book.next(view, log, foul_tried) {
            return Some(usi);
        }

        let mut candidates = candidate_moves(view, foul_tried);
        if view.you_in_check {
            // 王手中: 解消しえない手は（王手駒がどこにいても）王手放置で必ず反則に
            // なるので候補から外す。全滅したら元の候補に戻す（投了よりは反則のほうが
            // 手番を失わないぶんまし。真に詰みならサーバー側で終局している）
            let filtered: Vec<_> = candidates
                .iter()
                .filter(|(_, mv)| may_resolve_check(view, mv))
                .cloned()
                .collect();
            if !filtered.is_empty() {
                candidates = filtered;
            }
        }
        if candidates.is_empty() {
            return None;
        }

        // 同一指紋の粒子は質量を畳み込んでユニーク化して評価に使う
        // （ESSリサンプリング後は複製数が事後質量。ただし p(合法) ブレンドの
        // 実効 n はユニーク数で数える = 複製は独立な証拠ではない）。
        // ソフト救済の減衰はフィルタが logw へ課金済み（EPS_INFO）。
        // 粒子尤度モデル（likelihood.rs）で真の局面に近い粒子を厚くする。
        // 相手玉の位置で層化して抽出する（stratified_sample 参照）。
        // 粒子が完全に枯渇していても、事前確率だけで安全側の評価が成り立つ
        let particle_ctx = ParticleCtx {
            // 直近で自駒が取られたマス（相手の駒がそこに着地した）
            opp_landed_last: log.events().iter().rev().find_map(|e| match e {
                Observation::OpponentMoved {
                    captured_my_piece_at: Some(sq),
                    ..
                } => parse_usi_square(sq),
                _ => None,
            }),
        };
        let sample = stratified_sample(
            est.particles(),
            est.info_miss(),
            est.phys_taint(),
            est.log_weights(),
            view.your_color,
            &particle_ctx,
            budget.eval_particles,
            &mut self.rng,
        );

        // 相手の盤上駒数の概算（取った枚数ぶん減る。相手の打ちで戻る分は無視）
        let my_captures = log
            .events()
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    Observation::MyMove {
                        captured: Some(_),
                        ..
                    }
                )
            })
            .count();
        let opp_board_n = (20 - my_captures.min(19)) as f64;

        // 直前に受理された自分の手（手戻りシャッフルの抑制に使う）
        let last_my_move = log.events().iter().rev().find_map(|e| match e {
            Observation::MyMove { usi, .. } => parse_usi(usi),
            _ => None,
        });

        // クリーン粒子が全滅しているときだけ taint 粒子を取り出す（C-7 P3 / D4:
        // 嘘の盤面だが直近まで観測と整合していた歴史なので、用途を限定すれば
        // ブラインドの手探りより役立つ）。王手ソルバーの仮説投票・玉攻め・
        // ハング回避リスクで共有する（重複計算を避ける）。
        // **上限つき**（長手数の対局で持続したブラインドはユニーク taint 粒子が
        // 数百〜数千に膨らみうる。候補手ごとに O(particles×pieces) の被覆度
        // 走査があるため無制限だと思考予算を溶かす — 125te/132te シナリオの
        // 実測で検出。重み上位だけに絞る（自己正規化する関数群なので偏りは
        // 軽微、末尾は寄与が薄い）
        //
        // taint 粒子は物理制約を緩めた「嘘の盤面」なので**敵玉の位置まで嘘になる**。
        // 王手宣言の履歴から健全に絞れる候補集合（deduce::opp_king_candidates）へ
        // 玉を引き戻してから使う（実測: 対人局 50手目で 6d/6e/7e/7d/7b に 6.7% の
        // 質量が乗っていたが、演繹上は 3d/4e/5a/5e/6a/6b/6c の7マスしかあり得ない）。
        // **棄却ではなく移動**なのは、玉位置の質量を潰す事故を避けるため（ansatsu 回帰の教訓）
        let opp_color = view.your_color.other();
        // 玉位置ネットの分布（king_belief_nn）。ノブが両方0（既定）なら評価もしない。
        // 使うのはブラインド決定だけ（belief_gain_w と同じ規約）
        let net_king_dist: Vec<(Coord, f64)> =
            if sample.is_empty() && (king_net_w() > 0.0 || king_net_proj()) {
                let ctx = crate::belief_features::BeliefContext::from_log(view.your_color, log);
                let cands = crate::deduce::opp_king_candidates(view.your_color, log);
                crate::king_belief_nn::king_distribution(&ctx, &cands)
            } else {
                vec![]
            };
        let taint_owned: Vec<(Position, f64)> = if sample.is_empty() {
            let mut pool = taint_particles(est);
            if pool.len() > TAINT_POOL_CAP {
                pool.select_nth_unstable_by(TAINT_POOL_CAP, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                pool.truncate(TAINT_POOL_CAP);
            }
            // 切り戻し・アブレーション用ノブ（既定は有効）
            if std::env::var("TSUITATE_TAINT_KING_FIX").is_ok_and(|v| v == "0") {
                pool.iter().map(|&(p, w)| (p.clone(), w)).collect()
            } else {
                let cands = crate::deduce::opp_king_candidates(view.your_color, log);
                let net = (king_net_proj() && !net_king_dist.is_empty())
                    .then_some(net_king_dist.as_slice());
                project_taint_kings(&pool, &cands, opp_color, net)
            }
        } else {
            vec![]
        };
        let taint_pool: Vec<(&Position, f64)> =
            taint_owned.iter().map(|(p, w)| (p, *w)).collect();

        // 王手中は粒子に依存しない制約推論で「王手を解消する確率」を出す
        // （粒子が枯渇する終盤の反則バースト対策。check.rs 参照）。
        // taint 投票は駒得・リスク・p(合法) には混ぜない
        let mut check_solver = if view.you_in_check {
            let fouls: Vec<ShogiMove> = foul_tried.iter().filter_map(|u| parse_usi(u)).collect();
            let votes = if sample.is_empty() {
                &taint_pool
            } else {
                &sample
            };
            CheckSolver::new(view, votes, &fouls, log)
        } else {
            None
        };

        // 相手が位置を知っている自駒（露出）の地図
        let known = knownness_map(view, log, self.params.home_knownness);

        // 2手読み用: 自分が駒を取ったマス（露見）と自分の手が触れたマス
        // （estimator の my_capture_sq / my_touched_sq と同じ定義）。
        // my_fouls_this_turn はこの手番でここまでに自分が試みた反則の回数
        // （反則リトライ中は >0）。相手は反則宣言の回数を観測しているので、
        // 応手予測の my_foul_count_last_turn 特徴量として渡す
        let mut my_capture_squares: Vec<Coord> = vec![];
        let mut my_touched_squares: Vec<Coord> = vec![];
        let mut my_fouls_this_turn: u32 = 0;
        // 自分が打ちの反則をしたマス（局を通じて累積。`foul_tride` と違い
        // 受理でクリアされない）。`drop_probe_repeat_gate` 参照
        let mut my_drop_foul_squares = [false; 81];
        for e in log.events() {
            match e {
                Observation::MyMove { usi, captured, .. } => {
                    my_fouls_this_turn = 0;
                    if let Some(mv) = parse_usi(usi) {
                        let to = match mv {
                            ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
                        };
                        if captured.is_some() {
                            my_capture_squares.push(to);
                        }
                        if let ShogiMove::Board { from, .. } = mv {
                            my_touched_squares.push(from);
                        }
                        my_touched_squares.push(to);
                    }
                }
                Observation::MyFoul { usi, .. } => {
                    my_fouls_this_turn += 1;
                    // 自分が「打ちの反則」をしたマス = そこに相手駒がいることが
                    // 確定した（かつ以後も自分が取るまで動かないとは限らないが、
                    // **もう一度打っても新しい情報は買えない**）。
                    // `drop_probe_repeat_gate` のループ対策に使う
                    if let Some(ShogiMove::Drop { to, .. }) = parse_usi(usi) {
                        my_drop_foul_squares[crate::belief_features::sq_index(to)] = true;
                    }
                }
                _ => {}
            }
        }

        // 同じ自陣形へ戻る手の累積減点用（`repeat_penalty_w`）。
        // 自分側の配置は完全既知なので粒子不要・ノイズゼロ
        let own_config_history = if self.params.repeat_penalty_w != 0.0 {
            own_config_history(view.your_color, log)
        } else {
            HashMap::new()
        };

        // アンチドロー: 終盤にリードがあるほど攻め項を増幅して膠着を破る。
        // 手戻り/シャッフルの減点も同時に強めて「その場で回る」手を締め出す
        let push = endgame_push(view.move_number, material_lead(view));
        let params = {
            let mut p = self.params.clone();
            if push > 0.0 {
                p.check_bonus *= 1.0 + push;
                p.attack_w *= 1.0 + push;
                p.advance_w *= 1.0 + 0.5 * push;
                p.backtrack_penalty *= 1.0 + push;
                p.shuffle_penalty *= 1.0 + push;
            }
            p
        };

        // ブラインド時の玉攻め勾配（C-7 P3 追補）+ 局所被覆度ビリーフ（追補2）:
        // taint_pool の玉位置分布だけを抽出して攻めへ使う。個々の駒種・位置は
        // 特定しない「マスへの利き枚数密度」（ユーザーの実際の推論=
        // 「５七への相手利き≥2枚の確率が低い」に対応）は blind_hang_risk が
        // 受け（ハング回避）に使う
        let blind_king_dist: Vec<(Coord, f64)> = {
            let taint_dist = if taint_pool.is_empty() {
                vec![]
            } else {
                taint_king_distribution(&taint_pool, opp_color)
            };
            // 玉位置ネットのブレンド（`king_net_w` 参照。既定 0 = 従来挙動）
            if king_net_w() > 0.0 && !net_king_dist.is_empty() {
                blend_king_dist(&taint_dist, &net_king_dist, king_net_w())
            } else {
                taint_dist
            }
        };
        // 着地マスごとの被覆度をキャッシュ（成り/不成の同一着地マス等での
        // 重複走査を避ける）
        let mut coverage_cache: HashMap<Coord, f64> = HashMap::new();
        // ブラインドハング回避リスクは**既定で無効**（実験用オプトイン）。
        // codex レビュー: 5g（真実利き1枚）で期待値0.03（ほぼ0と誤信）、
        // 4h（真実1枚）で期待値1.48（過大評価）という較正不良は「ノイズの多い
        // 弱い特徴」ではなく「明確な誤誘導」水準で、blind_king_attack の
        // ボーナスを重み1.0の piece_value×coverage が簡単に相殺してしまう
        // （kakunari continue の指し継ぎが 2a1c 主体の無目的手へ逆戻りした
        // 実測とも整合）。局所被覆度は玉位置と違い複数駒の相対位置が同時に
        // 正しくないと当たらない複合情報で、taint の単純な force_apply では
        // 再現できない（ユーザーの実践知見どおり）。再設計するまでは無効
        let hang_risk_enabled = std::env::var("TSUITATE_ENABLE_HANG_RISK").is_ok();
        let debug_check_enabled = std::env::var("TSUITATE_DEBUG_CHECK").is_ok();

        // 詰めろ判定のプール: 厳密粒子があればそれ、全滅していれば taint 粒子。
        // 終盤のブラインド（＝詰めろが決まる局面）ほど厳密粒子は枯れているので、
        // sample だけを見ると項が発火しない（実測: 発端の対人局 57手目は厳密0個・
        // taint の玉位置信念は真実 7i に 71.6%）。
        // taint 粒子でも**自駒配置・持ち駒・相手の持ち駒は真実と同期**している
        // （estimator.rs::force_apply）ので、自玉の逃げ道と相手の打てる駒種で
        // ほぼ決まる被詰めろ判定は taint でも信頼できる。嘘が乗るのは相手の
        // 盤上駒＝支え駒の有無で、そこは IfSupported 側が吸収する
        let mate_pool: &[(&Position, f64)] = if sample.is_empty() { &taint_pool } else { &sample };

        // 評価本体（`expected`）の粒子プール。厳密粒子が全滅した決定では
        // 駒得期待値・情報利得・攻め圧力・valueネットが**丸ごとゼロ**になり、
        // 着手が前進バイアスと事前確率だけで決まっていた（実測: 全決定の26%。
        // src/hits.rs の「厳密粒子ゼロ」）。mate_pool・blind_king_attack・
        // CheckSolver 投票が既に持っている「厳密が全滅していれば taint」の規約を
        // 評価本体にも広げる。既定は無効（挙動不変）で、
        // `TSUITATE_EVAL_TAINT_FALLBACK=1` で有効化する。
        //
        // **p_legal には混ぜない**: taint 粒子は反則の説明（打ちマス占有など）を
        // 緩和して生かした粒子なので、合法性の証拠としては使えない。
        // 反則マス記憶系4種が全滅した領域でもあるので、供給チャネルは
        // gain 側（駒得・攻め）に限定する
        let eval_taint = sample.is_empty() && !taint_pool.is_empty() && eval_taint_fallback();
        let eval_pool: &[(&Position, f64)] = if eval_taint { &taint_pool } else { &sample };

        // ブラインド時の取り返し（観測だけで決まるので粒子の有無に依らず作れる。
        // 実際に使うのは厳密粒子ゼロの決定だけ = evaluate 側で判定する）
        let blind_recapture = blind_recapture_target(view, log);
        // ブラインド時の信念ネット供給（NN段階②、`belief_gain_w`）。
        // 81マスぶんの forward pass は決定点ごとに1回で足りる（粒子に依存しない）。
        // 重み0（既定）なら計算もしない = 挙動不変
        let blind_belief_board: Option<([f64; 81], f64)> =
            (belief_gain_w() > 0.0 && sample.is_empty()).then(|| {
                let ctx = crate::belief_features::BeliefContext::from_log(view.your_color, log);
                (
                    crate::belief_nn::board_occupancy(&ctx),
                    blind_capture_estimate(view, log),
                )
            });
        let blind_belief = blind_belief_board.as_ref().map(|(o, v)| (o, *v));
        // 自駒の利き枚数（`blind_attack_survive_w` の守り枚数と
        // `adds_focal_attacker` の争点判定で共用。決定点ごとに1回）
        let own_attack = own_attack_counts(view);
        // ブラインド決定での taint 占有合意（打ちの反則回避。`taint_occ_legal_w`）。
        // 打ちマスの占有は「駒種が当たっているか」に依らないので、gain 側より
        // taint を信用する範囲が狭い。決定点ごとに1回作れば全候補で使える
        let taint_occ_board: Option<[f64; 81]> = (params.taint_occ_legal_w != 0.0
            && sample.is_empty()
            && !taint_pool.is_empty())
        .then(|| {
            let mut occ = [0.0f64; 81];
            let mut mass = 0.0f64;
            for &(p, w) in &taint_pool {
                mass += w;
                for (sq, pc) in p.pieces() {
                    if pc.color == opp_color {
                        occ[crate::belief_features::sq_index(sq)] += w;
                    }
                }
            }
            if mass > 0.0 {
                for v in occ.iter_mut() {
                    *v /= mass;
                }
            }
            occ
        });
        // V2（玉距離重み付き利き）の相手玉側。玉位置の信念は評価に使う粒子から
        // 取る（厳密が生きていればそちら、全滅していれば taint = mate_pool と
        // 同じ規約）。重みが両方0（既定）なら 81マスぶんの表も作らない
        // 紐の働き重み付け（`link_work_w`）も相手玉側の距離重みを使うので、
        // effect_opp_w が 0 でもそちらが有効なら表を作る（作り忘れると
        // 「自玉側だけの働き」という別物の量になる）
        let opp_king_w: Option<[f64; 81]> = (params.effect_opp_w != 0.0 || params.link_work_w != 0.0)
            .then(|| {
                let src: &[(&Position, f64)] = if sample.is_empty() { &taint_pool } else { &sample };
                opp_king_effect_weights(&taint_king_distribution(src, opp_color))
            })
            .flatten();

        // 評価の前提条件の発火率（src/hits.rs）。`expected` の内側にある項
        // （駒得期待値・valueネット等）は厳密粒子が全滅すると丸ごと無効になるので、
        // それがどれくらいの頻度で起きているかを測る
        if crate::hits::enabled() {
            crate::hits::flag("王手中", view.you_in_check);
            crate::hits::flag("厳密粒子ゼロ", sample.is_empty());
            crate::hits::flag(
                "value_nn の前提充足",
                !sample.is_empty() && !view.you_in_check,
            );
            crate::hits::flag("taint 評価へ落ちた", eval_taint);
            // 厳密も taint も無い = 事前確率と粒子に依らない項だけで指している決定
            crate::hits::flag("粒子が全く無い", sample.is_empty() && taint_pool.is_empty());
        }

        // 打ち当て露出（`drop_hit_evac_w`）の現局面の露出。相手の持ち駒は
        // 取られた自駒から完全に決まる（GameModel::opponent_hand。相手が
        // 見えない打ちで消費した分は引けないが上界のままでよい = doc 参照）。
        // 歩を持たれていなければ脅威が無いので項ごと無効にする。王手中も無効
        // （CheckSolver の領分）
        let drop_hit_expo_before: Option<f64> = (params.drop_hit_evac_w != 0.0
            && !view.you_in_check)
            .then(|| {
                let opp_hand = GameModel::from_log(view.your_color, log).opponent_hand();
                (opp_hand.get(&Role::Pawn).copied().unwrap_or(0) > 0)
                    .then(|| drop_hit_exposure(&view.your_pieces, view.your_color))
            })
            .flatten();

        // 成りポテンシャル（`promo_potential_w`）の現局面の値。着手後との差分を
        // gain へ加算する。王手中は無効（CheckSolver の領分）
        let promo_pot_before: Option<f64> = (params.promo_potential_w != 0.0
            && !view.you_in_check)
            .then(|| promo_potential(&view.your_pieces, view.your_color));
        // 大駒の成り道（`major_promo_path_w`）の現局面の値。同じく差分で使う
        let major_path_before: Option<f64> = (params.major_promo_path_w != 0.0
            && !view.you_in_check)
            .then(|| major_promo_path(&view.your_pieces, view.your_color));

        // 持ち駒オプション価値（`hand_option_w`）の決定点コンテキスト。
        // 王手中は無効（合駒は CheckSolver の領分）
        let hand_option: Option<HandOption> = (params.hand_option_w != 0.0
            && !view.you_in_check)
            .then(|| hand_option_context(view));

        let rng = &mut self.rng;
        // 王手中の玉の手の gain 平均化判定用（check_king_gain_mean の doc 参照）
        let my_king = king_square(view);
        let is_king_move = |mv: &ShogiMove| {
            matches!(*mv, ShogiMove::Board { from, .. } if Some(from) == my_king)
        };
        // 平均化の対象となる玉の手（直前に自駒が取られたマスへの玉捕獲は
        // 観測確実な取り返しなので除外。平均化ブロックの doc 参照）
        let equalized_king_move = |mv: &ShogiMove| {
            is_king_move(mv)
                && !matches!(
                    (*mv, blind_recapture),
                    (ShogiMove::Board { to, .. }, Some((sq, _))) if to == sq
                )
        };
        // valueネットのstate特徴量キャッシュ（sample と同じ並び。候補間で共通なので
        // 手番ごとに1回だけ計算する）
        let mut nn_state_cache: Vec<Option<[f64; crate::value_features::VALUE_FEATURES]>> =
            vec![None; eval_pool.len()];
        // 1段目: 全候補を1手読み（静的リスク項つき）で評価する。
        // (usi, mv, 内訳, gain外の補正, 1段目スコア)
        let mut scored: Vec<(String, ShogiMove, EvalOut, f64, f64)> = vec![];
        for (usi, mv) in candidates {
            let mut prior = prior_legal(view, &mv, opp_board_n);
            if view.you_in_check {
                prior *= match check_solver.as_mut() {
                    Some(solver) => solver.resolve_probability(&mv).clamp(0.02, 1.0),
                    // ソルバーが作れないときは従来の粗い事前確率
                    // （玉移動 > 取り/合駒の順）に落とす
                    None => in_check_prior(view, &mv),
                };
            }
            let mut out = evaluate(
                view,
                &mv,
                eval_pool,
                eval_taint,
                mate_pool,
                prior,
                &known,
                &params,
                budget,
                &mut nn_state_cache,
                blind_recapture,
                blind_belief,
                taint_occ_board.as_ref(),
                opp_king_w.as_ref(),
                drop_hit_expo_before,
                promo_pot_before,
                major_path_before,
                hand_option.as_ref(),
                &my_drop_foul_squares,
            );
            // 王手中: 仮説条件付きの「王手駒の除去期待値」（check.rs::removal_term）。
            // 王手駒のマスを取る手は受理された未来で脅威ごと駒を排除し、玉逃げ等の
            // 解消手は王手駒を盤に残す。この差は粒子が真の王手駒を外している局面
            // （kakutori.kif）では gain に現れないため、CheckSolver の仮説分布で
            // 補正する。gain の内側（= combine_score の p_legal 割引の内側）に
            // 置くこと: 王手中の加点を外側に置くと反則確実な手が素通りする
            // （dragon-check-drop の教訓）
            if view.you_in_check && params.checker_removal_w != 0.0 {
                if let Some(term) = check_solver
                    .as_mut()
                    .and_then(|solver| solver.removal_term(&mv))
                {
                    out.checker_removal = params.checker_removal_w * term;
                    out.gain += out.checker_removal;
                }
            }
            if debug_check_enabled && view.you_in_check {
                eprintln!(
                    "DEBUG {usi}: prior={prior:.4} gain={:.3} p_legal={:.4} foul_cost={:.3} score={:.4}",
                    out.gain,
                    out.p_legal,
                    out.foul_cost,
                    out.score()
                );
            }
            // gain の外側の補正（タイブレーク乱数・手戻り/シャッフル減点）は
            // 2手読み後の再計算でも同じ値を使うので分離して持つ
            let mut adjust = rng.random_range(0.0..0.01);
            if !blind_king_dist.is_empty() {
                // 攻め加点は p(合法) で割り引く（加点が実現するのは手が受理された
                // ときだけ）。adjust は combine_score の外側に加算されるため、
                // 割引がないと反則確実な手の攻めボーナスが反則コストを素通りで
                // 上書きする。王手中が顕著（dragon-check-drop.kif: 解消確率ゼロの
                // G*5h が信念上の敵玉 5i/4h への利きで +1.7 を得て正解の玉逃げ
                // 5c4d を逆転）だが、平時のブラインドでも taint 粒子は反則の説明
                // （打ちマス占有など）を緩和しているため同じ穴が開く
                let attack = blind_king_attack(view, &mv, &blind_king_dist);
                // **攻め駒の生存で割り引く**（`blind_attack_survive_w`、2026-08-03、
                // ユーザー指摘「初期位置の玉に王手をかけてしまうのを危惧した」）。
                // 厳密粒子ゼロでは `expected`（駒得・取られリスク）が丸ごと消えるので、
                // この加点だけが残って対抗する項が無い ＝ **無防備な駒を信念上の玉の
                // 隣へ置くほど加点が最大になる**。quest31-m040 の実測: 5九へ利く
                // 銀打ち（5七/5九/7七/5八/6八/4八）が adjust +0.59〜0.90 を得て
                // 上位を占め、4七の争点を支える 3八銀打（5九に利かないので +0.002）が
                // 78位に沈む。ユーザー判定は前者が悪手・後者が本命。
                //
                // 解消手が「王手駒を取る」しかない王手は駒の献上でしかない
                // （メモリ strong-check-few-resolutions）。着地マスの taint 由来の
                // 期待被覆枚数から自分の守り枚数を引いた分だけ加点を減衰させる。
                // 加点側だけを絞る形なので、`blind_hang_risk`（着地マス全般への
                // 一律減点、既定 off）と違い普通の前進手には影響しない
                let survive = if params.blind_attack_survive_w > 0.0
                    && attack > 0.0
                    && !taint_pool.is_empty()
                {
                    let to = match mv {
                        ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
                    };
                    let cov = *coverage_cache
                        .entry(to)
                        .or_insert_with(|| taint_square_coverage(&taint_pool, to, opp_color));
                    // 着地マスを守る自駒の枚数（着手駒自身は自分のマスを守れないので
                    // 移動元から to へ利いていたぶんだけ引く）
                    let mut def = f64::from(own_attack[crate::belief_features::sq_index(to)]);
                    if let ShogiMove::Board { from, .. } = mv {
                        if own_defends_from(view, from, to) {
                            def -= 1.0;
                        }
                    }
                    1.0 / (1.0 + params.blind_attack_survive_w * (cov - def).max(0.0))
                } else {
                    1.0
                };
                adjust += out.p_legal * BLIND_KING_ATTACK_W * attack * survive;
            }
            if hang_risk_enabled && !taint_pool.is_empty() {
                adjust -= BLIND_HANG_RISK_W
                    * blind_hang_risk(view, &mv, &taint_pool, opp_color, &mut coverage_cache);
            }
            // 手戻り（直前の手をそのまま逆に戻す）は膠着の典型なので減点。
            // 直前に動かした駒をまた動かすだけの手も雑なシャッフルとして軽く減点
            if let (
                Some(ShogiMove::Board {
                    from: pf, to: pt, ..
                }),
                ShogiMove::Board { from, to, .. },
            ) = (last_my_move, mv)
            {
                if from == pt && to == pf {
                    adjust -= params.backtrack_penalty;
                } else if from == pt {
                    adjust -= params.shuffle_penalty;
                }
            }
            // 同じ自陣形へ戻る手の累積減点。上の2つは直前の1手しか見ず固定額
            // なので、往復を繰り返しても減点が増えない（`repeat_penalty_w` の
            // doc 参照: 実戦で角が 3四↔2五 を6手繰り返した局面では、
            // ブラインド玉攻めの +1.8 が固定の −0.369 を上回り続けていた）
            if params.repeat_penalty_w != 0.0 {
                let seen = own_config_history
                    .get(&own_config_fingerprint_after(view, &mv))
                    .copied()
                    .unwrap_or(0);
                adjust -= params.repeat_penalty_w * f64::from(seen);
            }
            let score = out.score() + adjust;
            scored.push((usi, mv, out, adjust, score));
        }

        // 王手中の玉の手は gain を「玉の手全体の平均」に揃える
        // （doc は check_king_gain_mean。分散＝幻の敵駒ノイズだけを消し、
        // 玉の手 vs 非玉プローブの相対水準は保存する）。
        // **直前に自駒が取られたマスへの玉捕獲は除外する**: そこに相手駒が
        // いるのは観測事実で、その捕獲 gain（blind_recapture / 粒子の駒得）は
        // 幻ではない。巻き込むと確実な取り返しのベイトが消える
        // （実測: recap-dragon の 6a7a が 19/20 → 13/20 に落ちた）
        if view.you_in_check && check_king_gain_mean() {
            let king_idx: Vec<usize> = scored
                .iter()
                .enumerate()
                .filter(|(_, s)| equalized_king_move(&s.1))
                .map(|(i, _)| i)
                .collect();
            if king_idx.len() > 1 {
                let mean =
                    king_idx.iter().map(|&i| scored[i].2.gain).sum::<f64>() / king_idx.len() as f64;
                for &i in &king_idx {
                    scored[i].2.gain = mean;
                    scored[i].4 = scored[i].2.score() + scored[i].3;
                }
            }
        }

        // 2段目: 上位候補だけ相手の応手をサンプルして再評価。
        // gain 内の静的リスク項の depth2_replace 分を実測の期待損失で
        // 置き換えて（一致するなら無変化）、最終式を適用し直す
        scored.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
        // **争点への利き足し**の予約枠（`adds_focal_attacker` の doc 参照）。
        // 静的スコアでは必ず沈むので、上位N本の足切りとは別枠で読む機会を作る。
        // 王手中は CheckSolver の領分なので予約しない
        let focal_k = depth2_focal_k();
        let focal_reserved: HashSet<usize> = if focal_k > 0 && !view.you_in_check {
            scored
                .iter()
                .enumerate()
                .skip(budget.depth2_top_k)
                .filter(|(_, s)| adds_focal_attacker(view, &s.1, &own_attack))
                .map(|(i, _)| i)
                .take(focal_k)
                .collect()
        } else {
            HashSet::new()
        };
        // (usi, 選択手の p_legal, スコア)
        let mut best: Option<(String, f64, f64)> = None;
        let mut ranking: Vec<CandidateScore> = vec![];
        for (i, (usi, mv, out, adjust, score)) in scored.into_iter().enumerate() {
            // 平均化した玉の手は2手読みで gain を再構成しない（応手サンプルも
            // 同じ幻の粒子が源で、揃えた序列が壊れるだけ）
            let depth2 = (i < budget.depth2_top_k || focal_reserved.contains(&i))
                && !(view.you_in_check && check_king_gain_mean() && equalized_king_move(&mv));
            let (final_gain, final_score) = if depth2 {
                let delta = depth2_delta(
                    view,
                    &mv,
                    &sample,
                    &known,
                    &my_capture_squares,
                    &my_touched_squares,
                    my_fouls_this_turn,
                    &params,
                    budget,
                    &mut *rng,
                );
                // 構想（自分の手 → 相手の応手 → 自分の手）は depth2_delta の
                // 内側へ統合した（`plan_w`）。応手を挟まない外付けの加点は
                // 200局で -6pt と明確に負だったので廃止
                //
                // relief（正 = 静的リスクより楽観に置き換わる量）は
                // `(1−depth2_optimism_cap)×risk_mean` で頭打ちにする。相手の応手
                // 方策は取り返しを確率的にしか選ばないので |delta| < risk_mean に
                // なりやすく、上限が無いと**静的リスクが大きいほど加点が増える**
                // （F3。depth2_optimism_cap の doc 参照）。悲観方向は制限しない
                let relief = params.depth2_replace * (out.risk_mean + delta);
                let max_relief = (1.0 - params.depth2_optimism_cap) * out.risk_mean;
                let gain2 = out.gain + relief.min(max_relief);
                (
                    gain2,
                    combine_score(gain2, out.p_legal, out.foul_cost) + out.foul_probe + adjust,
                )
            } else {
                (out.gain, score)
            };
            ranking.push(CandidateScore {
                usi: usi.clone(),
                static_score: score,
                static_gain: out.gain,
                score: final_score,
                gain: final_gain,
                p_legal: out.p_legal,
                foul_cost: out.foul_cost,
                adjust,
                depth2,
                checker_removal: out.checker_removal,
                capture_bet_penalty: out.capture_bet_penalty,
                mate_threat: out.mate_threat,
                mate_risk: out.mate_risk,
                king_holes: out.king_holes,
                value_nn: out.value_nn,
                capture_value: out.capture_value,
                risk: out.risk_mean,
                link: out.link,
                promo: out.promo,
                hand_option: out.hand_option,
                board_discount: out.board_discount,
            });
            if best.as_ref().is_none_or(|(_, _, s)| final_score > *s) {
                best = Some((usi, out.p_legal, final_score));
            }
        }
        ranking.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        // 評価項の発火率フック（TSUITATE_DBG_HITS=1 のときだけ）。
        // 「中立だった変更が効いていないのか発火していないのか」の切り分け用
        if crate::hits::enabled() {
            crate::hits::observe_ranking(&ranking);
        }
        self.last_ranking = Some(ranking);

        let mut debug = debug_summary(est, &sample, push);
        // 選択手の p(合法) 予測を記録へ残す（C-7 P3 の前提整備: アリーナ真実の
        // 受理/反則と突き合わせて Brier/logloss を測る。bin/analyze 参照）
        if let (Some((_, p_legal, _)), Some(obj)) = (&best, debug.as_object_mut()) {
            obj.insert(
                "p_legal".into(),
                serde_json::json!(((p_legal * 1000.0).round()) / 1000.0),
            );
        }
        self.last_debug = Some(debug);
        best.map(|(usi, _, _)| usi)
    }

    fn name(&self) -> &'static str {
        "estimator"
    }

    fn debug_state(&self) -> Option<serde_json::Value> {
        self.last_debug.clone()
    }

    fn last_ranking(&self) -> Option<&[CandidateScore]> {
        self.last_ranking.as_deref()
    }
}

/// 評価用の粒子サンプルを相手玉の位置で**層化抽出**する。
///
/// 従来は penalty 昇順の先頭から eval_particles 件を採っていたが、層内の並びは
/// 生存順で相関しており、少数の玉位置仮説群だけで候補を評価する偏りがあった。
/// 設計（2026-07-15 のレビュー指摘対応込み）:
/// - 採用数は**必ず eval_particles 以下**（カバレッジ枠→D'Hondt式の質量比例配分）
/// - 層内は決定的シャッフルで代表抽出（生存順バイアスを切る。rng は対局シード由来）
/// - 出力は層をまたぐ**ラウンドロビン順**: 先頭 k 件しか見ない評価
///   （王周辺圧力・2手読み）でも玉位置の分布が近似される
/// - 採らなかった質量は同層の採用粒子へ再配分（層合計の重みを保存）
/// - **multiplicity 畳み込み**（C-7 P1）: ESS リサンプリング後は複製数そのものが
///   事後質量なので、同一指紋の個体は捨てずに質量 Σexp(logw) を畳み込む
///   （旧「最良個体で代表」だとリサンプリングの結果が評価時に消える —
///   2026-07-17 codex レビュー最重要指摘）
/// - 重み和は較正アンカー legacy_mass へ正規化する: info_miss 昇順の先頭
///   min(eval, unique) 件の EPS_INFO^info_miss 和（= 旧方式の soft 重み和の後継。
///   複製は独立な証拠ではないので、p(合法) ブレンドの実効 n はユニーク数で数える）
/// - 粒子尤度モデル（likelihood.rs、アリーナ真実で教師あり学習）の exp(θ·φ) を
///   乗じる: 真の局面に近い粒子ほど評価に効く。相対的な再重み付けなので
///   合計質量（較正）は変えない
/// - 推定器の観測尤度の対数重み（Estimator::log_weights、SIR の重み更新）は
///   個体質量の側で効く: 観測を「相手が指しにくい手」でしか説明できない粒子
///   （幻の角の飛び込み王手等）を粒子間で相対的に軽くする。
///   ソフト減衰はフィルタが logw へ課金済み（EPS_INFO）なのでここでは掛けない
fn stratified_sample<'a>(
    particles: &'a [Position],
    info_miss: &[u8],
    phys_taint: &[u8],
    log_weights: &[f64],
    my_color: Color,
    ctx: &ParticleCtx,
    eval_particles: usize,
    rng: &mut StdRng,
) -> Vec<(&'a Position, f64)> {
    let opp = my_color.other();
    // ユニーク化: 同一指紋の質量 logΣexp(logw) と最小 info_miss を畳み込む。
    // 物理不整合（phys_taint>0）の粒子は**通常サンプルから除外**する
    // （C-7 P3 / D4: 嘘の盤面を駒得・リスク・p(合法) に混ぜない。
    // 必要な補助評価は別途作る taint_pool を直接使う）
    struct Unique<'a> {
        pos: &'a Position,
        mass_log: f64,
        min_miss: u8,
        logl: f64,
    }
    let mut seen: HashMap<u64, usize> = HashMap::new();
    let mut uniques: Vec<Unique> = vec![];
    for (i, pos) in particles.iter().enumerate() {
        if phys_taint.get(i).copied().unwrap_or(0) > 0 {
            continue;
        }
        let lw = log_weights.get(i).copied().unwrap_or(0.0);
        let miss = info_miss.get(i).copied().unwrap_or(0);
        match seen.entry(pos.fingerprint()) {
            std::collections::hash_map::Entry::Vacant(e) => {
                let logl =
                    particle_log_weight(&particle_features(pos, my_color, ctx), &FITTED_THETA);
                e.insert(uniques.len());
                uniques.push(Unique {
                    pos,
                    mass_log: lw,
                    min_miss: miss,
                    logl,
                });
            }
            std::collections::hash_map::Entry::Occupied(e) => {
                let u = &mut uniques[*e.get()];
                u.mass_log = logaddexp(u.mass_log, lw);
                u.min_miss = u.min_miss.min(miss);
            }
        }
    }
    if uniques.is_empty() {
        return vec![];
    }
    // 較正アンカー: 旧方式（penalty昇順の先頭 min(eval, unique) 件の soft 重み和）の
    // 後継。ソフト減衰の較正はフィルタと同じ EPS_INFO^info_miss で数える。
    // **尤度・logw 適用前**のベース重みで計るのは従来どおり（p(合法) ブレンドの
    // 実効質量 n が尤度分布に引きずられて prior_weight の較正が崩れるため —
    // 2026-07-16 レビュー指摘）。ESS リサンプリングで複製が増えても n は
    // ユニーク数でしか増えない = リサンプリングは確信を偽装しない
    let mut miss_sorted: Vec<u8> = uniques.iter().map(|u| u.min_miss).collect();
    miss_sorted.sort_unstable();
    let legacy_mass: f64 = miss_sorted
        .iter()
        .take(eval_particles)
        .map(|&m| EPS_INFO.powi(i32::from(m)))
        .sum();
    // 分布重み: 個体質量 × 粒子尤度 = exp(mass_log + logl)（オーバーフロー対策で
    // max を引く。全体スケールは最後に legacy_mass へ正規化されるので相対値だけが
    // 意味を持つ）
    let max_logl = uniques
        .iter()
        .map(|u| u.mass_log + u.logl)
        .fold(f64::MIN, f64::max);
    let uniques: Vec<(&Position, f64)> = uniques
        .into_iter()
        .map(|u| (u.pos, (u.mass_log + u.logl - max_logl).exp()))
        .collect();

    // 玉位置で層化（質量降順）
    let mut index: HashMap<Option<Coord>, usize> = HashMap::new();
    let mut strata: Vec<(Vec<(&Position, f64)>, f64)> = vec![];
    for (pos, w) in uniques {
        let k = pos.king_square(opp);
        let i = *index.entry(k).or_insert_with(|| {
            strata.push((vec![], 0.0));
            strata.len() - 1
        });
        strata[i].0.push((pos, w));
        strata[i].1 += w;
    }
    strata.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // 採用枠の配分（合計は eval_particles を超えない）:
    // まずカバレッジ枠（各層 MIN_STRATUM 件まで、質量降順のラウンドロビン）、
    // 残り予算は D'Hondt（mass/(quota+1) が最大の層へ1件ずつ）で質量比例に配る
    const MIN_STRATUM: usize = 4;
    let n = strata.len();
    let mut quotas = vec![0usize; n];
    let mut budget = eval_particles;
    'coverage: for _ in 0..MIN_STRATUM {
        for i in 0..n {
            if budget == 0 {
                break 'coverage;
            }
            if quotas[i] < strata[i].0.len() {
                quotas[i] += 1;
                budget -= 1;
            }
        }
    }
    while budget > 0 {
        let mut best: Option<(usize, f64)> = None;
        for i in 0..n {
            if quotas[i] >= strata[i].0.len() {
                continue;
            }
            let score = strata[i].1 / (quotas[i] as f64 + 1.0);
            if best.is_none_or(|(_, s)| score > s) {
                best = Some((i, score));
            }
        }
        let Some((i, _)) = best else {
            break; // 全層が member 数まで採用済み
        };
        quotas[i] += 1;
        budget -= 1;
    }

    // 層内の採用: 重み付き systematic resampling。
    // 選択確率 ∝ 重みで quota 件を等間隔に引き、各出力へ**等重み**（層質量/quota）を
    // 割り当てる。「重み比例で選び、さらに元の重みも配る」と二重適用になり
    // 低重み粒子の期待寄与を過小評価する（2026-07-15 追加レビュー指摘）。
    // 等重み割当なら任意の quota で E[粒子iの寄与] = w_i の不偏性が成り立つ
    // （同一粒子が複数スロットに乗ることもあるが合計質量は固定）。
    // 出力後に層内を一様シャッフル（等重みなので不偏のまま）して、
    // prefix利用時の生存順相関を切る
    let resampled: Vec<Vec<(&Position, f64)>> = strata
        .iter()
        .zip(&quotas)
        .map(|((members, mass), &q)| {
            if q == 0 || *mass <= 0.0 {
                return vec![];
            }
            let unit = mass / q as f64;
            let offset: f64 = rng.random_range(0.0..unit);
            let mut out: Vec<(&Position, f64)> = Vec::with_capacity(q);
            let mut cum = 0.0;
            let mut idx = 0;
            for k in 0..q {
                let target = offset + k as f64 * unit;
                while idx + 1 < members.len() && cum + members[idx].1 <= target {
                    cum += members[idx].1;
                    idx += 1;
                }
                out.push((members[idx].0, unit));
            }
            for i in (1..out.len()).rev() {
                let j = rng.random_range(0..=i);
                out.swap(i, j);
            }
            out
        })
        .collect();
    // 層をまたぐラウンドロビン出力（prefixしか見ない評価でも層化が効く）
    let max_quota = quotas.iter().copied().max().unwrap_or(0);
    let mut sample: Vec<(&Position, f64)> = vec![];
    for round in 0..max_quota {
        for stratum in &resampled {
            if let Some(&entry) = stratum.get(round) {
                sample.push(entry);
            }
        }
    }

    // 旧方式の重み和へ正規化（較正の維持）
    let sample_mass: f64 = sample.iter().map(|(_, w)| w).sum();
    if sample_mass > 0.0 {
        let norm = legacy_mass / sample_mass;
        for (_, w) in sample.iter_mut() {
            *w *= norm;
        }
    }
    sample
}

/// taint 粒子を王手ソルバー投票に使う深さの上限（それ以上は嘘が深すぎる）
const TAINT_VOTE_MAX: u8 = 6;
/// ブラインド時の玉攻めボーナスの重み（クリーン粒子全滅時のみ。
/// taint 粒子から抽出した**玉位置分布だけ**を使い、盤面の嘘は評価に入れない。
/// kakunari 実測: 玉位置信念は 91.8% で真実に集中するのに、評価が使えず
/// 無目的手を選んでいた）
const BLIND_KING_ATTACK_W: f64 = 2.0;

/// ブラインド時の取り返しボーナスの重み（クリーン粒子全滅時のみ）。
///
/// **発端**（2026-07-26、`bin/recapture_probe` と発火率フック）: 厳密粒子は
/// 決定の26%で全滅し、そのとき `expected`（駒得期待値を含む評価本体）が
/// 丸ごとゼロになる。実測した1局面（相手の龍が自分の銀を取った直後）では
/// シード6本中4本が「龍を取り返す 6a7a」を gain 11〜13 で1位に選ぶのに、
/// **厳密粒子ゼロの2本では同じ手が gain 0.045 の89位**まで落ちていた。
/// 只取られと取り返し逃しの主因はここ（評価が低いのではなく、消えている）。
///
/// 埋め方は `blind_king_attack` と同じ思想: **嘘の盤面（taint 粒子）は使わず、
/// 観測だけから確実に言えること**を使う。相手が直前の手で自駒を取ったマスには
/// **相手の駒が確実にいる**（観測 `OpponentMoved{captured_my_piece_at}` そのもの）。
/// 駒種は不明なので、相手の盤上に残っているはずの駒の平均交換価値で見積もる
/// （持ち駒の内訳まで含めて観測から一意に決まる。`blind_capture_estimate`）。
/// taint フォールバックを評価本体へ広げる案は**別途200局で不採用**になっている
/// （taint は物理制約を緩めた盤面なので、その 7a に駒がいる保証すら無い）
const BLIND_RECAPTURE_W: f64 = 1.0;

fn blind_recapture_w() -> f64 {
    static V: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("TSUITATE_BLIND_RECAPTURE_W")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(BLIND_RECAPTURE_W)
    })
}

/// 「直前の相手手が自駒を取ったマス」と、そこにいる相手駒の期待交換価値。
///
/// 観測だけで決まる（粒子を使わない）:
/// - マスは `OpponentMoved{captured_my_piece_at}` そのもの。直前の相手手に限る
///   （それ以前のマスは相手がもう動かしているかもしれない）
/// - 駒種は不明だが、**相手の盤上に残っている駒の多重集合は観測から一意に決まる**:
///   初期配置 − 自分が取った駒（`MyMove{captured}`）＋ 相手が自分から取った駒
///   （打ち直されて盤に戻りうる）。その平均交換価値を見積もりに使う。
///   玉は除く（玉で取り返しに来る形は稀で、平均を押し上げるだけ）
fn blind_recapture_target(view: &PlayerView, log: &ObservationLog) -> Option<(Coord, f64)> {
    let square = log.events().iter().rev().find_map(|e| match e {
        Observation::OpponentMoved {
            captured_my_piece_at,
            ..
        } => Some(captured_my_piece_at.as_deref().and_then(parse_usi_square)),
        _ => None,
    })??;
    let opp = view.your_color.other();
    let mut roles: Vec<Role> = Position::initial()
        .pieces_of(opp)
        .iter()
        .map(|p| p.role)
        .filter(|r| *r != Role::King)
        .collect();
    for e in log.events() {
        // 自分が取った駒は相手の盤上から消えている（自分の持ち駒になる）
        if let Observation::MyMove {
            captured: Some(role),
            ..
        } = e
        {
            if let Some(i) = roles.iter().position(|r| r == role) {
                roles.swap_remove(i);
            }
        }
    }
    // 近似: 相手が自分から取った駒（相手の持ち駒 → 打てば盤に戻る）は数えない。
    // 打たれた駒の多くは歩なので、数えないぶん見積もりはやや高めに出る
    if roles.is_empty() {
        return None;
    }
    let mean = roles.iter().map(|&r| exchange_value(r)).sum::<f64>() / roles.len() as f64;
    Some((square, mean))
}

/// 信念ネット（NN段階②）を**厳密粒子ゼロの決定でだけ** gain 側へ供給する重み。
/// 既定 0 = 挙動不変。`TSUITATE_BELIEF_GAIN_W` で上書き（凍結版は知らない名前）。
///
/// 位置づけ: `BLIND_RECAPTURE_W` の**盤面全体への一般化**。あちらは
/// 「直前に自駒が取られたマスには相手駒が確実にいる」という観測1点だけを使い、
/// こちらは81マスの占有確率を使う。実測（`bin/belief_probe`、アリーナ
/// ホールドアウト30局）で、**厳密粒子ゼロの決定は全体の 29.2%** を占め、
/// そこでの粒子の占有信念は AUC 0.685（ほぼ無情報）／対数損失 1.0657 なのに対し、
/// 信念ネットは AUC 0.821 ／ 0.4418。粒子が「無い」のではなく「嘘」なので、
/// taint フォールバック（`TSUITATE_EVAL_TAINT_FALLBACK`）とは供給源が違う。
///
/// **p_legal には混ぜない**（反則マス記憶系4種が全滅したチャネル）。
/// **既存粒子の再重み付けでもない**（nn-particle-likelihood で飽和済み）。
/// 「粒子が1つも無いときに信念を供給する」チャネルに限定する
fn belief_gain_w() -> f64 {
    static V: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("TSUITATE_BELIEF_GAIN_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(0.0)
    })
}

/// 玉位置ネット（king_belief_nn = 候補集合内 softmax のカテゴリカル分布）を
/// **厳密粒子ゼロの決定でだけ**ブラインド玉攻めの玉位置分布へブレンドする係数 λ。
/// 既定 0 = ネットを評価すらしない（挙動不変）。`TSUITATE_KING_NET_W` で上書き
/// （凍結版は知らない名前 = `-f env=` は候補側にだけ効く）。
///
/// 位置づけ: 段階②の pointwise 信念ネットは「マスごとの独立確率」なので
/// 玉1枚の同時分布を表現できなかった（belief-net-stage2 の保留理由の一つ）。
/// 玉位置は攻めの律速と特定済み（v2-bottleneck-is-king-belief）なので、
/// 玉に特化したカテゴリカル分布として学習し直した。
/// 供給先は `blind_king_dist`（taint 粒子の玉位置分布とのブレンド）:
/// p = (1−λ)·p_taint + λ·p_net。taint が空でもネットだけで供給できる
fn king_net_w() -> f64 {
    static V: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("TSUITATE_KING_NET_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite())
            .map(|v| v.clamp(0.0, 1.0))
            .unwrap_or(0.0)
    })
}

/// 玉位置ネットで taint 粒子の玉移設先を選ぶ（`project_taint_kings` の誘導）。
/// **既定 on**（`TSUITATE_KING_NET_PROJ=0` で従来の最近傍移動へ切り戻し。
/// TAINT_KING_FIX と同じ規約、凍結版は知らない名前）。有効時は移設が必要な
/// 粒子を、ネット分布の CDF を等間隔の分位点で引いて割り当てる
/// （決定的・分布比例。全部 argmax に集めると玉位置の多様性が潰れる）。
///
/// 実測（2026-07-31、ペア3シード 200局×3 vs v13、match_seed=20260801〜03）:
/// 56.0/61.5/53.0 = **56.8% vs 対照（main）50.2%（+6.7pt、3シード全て対照以上）**。
/// ガントレット（100局）v6 90.4 / v7 75.0 / v8 73.1 / v9 69.2 / v10 71.2 /
/// v11 59.6 / v12 60.6 / v13 51.0%。反則/局・手数・思考時間は変質なし。
/// ブレンド（king_net_w）は単独中立・投影との複合では投影の利得を打ち消す
/// （50.8%）ため既定0のまま
fn king_net_proj() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| !std::env::var("TSUITATE_KING_NET_PROJ").is_ok_and(|v| v == "0"))
}

/// taint 玉位置分布とネット分布のブレンド: p = (1−λ)·p_taint + λ·p_net。
/// どちらも正規化済みなので結果も正規化されている（taint が空なら実質ネットのみ）
fn blend_king_dist(
    taint: &[(Coord, f64)],
    net: &[(Coord, f64)],
    lambda: f64,
) -> Vec<(Coord, f64)> {
    let mut acc: std::collections::BTreeMap<(i8, i8), f64> = std::collections::BTreeMap::new();
    let taint_total: f64 = taint.iter().map(|(_, p)| p).sum();
    for (c, p) in taint {
        *acc.entry((c.file, c.rank)).or_insert(0.0) += (1.0 - lambda) * p;
    }
    for (c, p) in net {
        *acc.entry((c.file, c.rank)).or_insert(0.0) += lambda * p;
    }
    let z: f64 = if taint_total > 0.0 { 1.0 } else { lambda };
    if z <= 0.0 {
        return vec![];
    }
    acc.into_iter()
        .map(|((file, rank), p)| (Coord { file, rank }, p / z))
        .collect()
}

/// ブラインド時に使う「盤上に残っている相手駒の平均交換価値」。
/// `blind_recapture_target` の駒種会計と同じもの（あちらはマスとセットで返す）
fn blind_capture_estimate(view: &PlayerView, log: &ObservationLog) -> f64 {
    let opp = view.your_color.other();
    let mut roles: Vec<Role> = Position::initial()
        .pieces_of(opp)
        .iter()
        .map(|p| p.role)
        .filter(|r| *r != Role::King)
        .collect();
    for e in log.events() {
        if let Observation::MyMove {
            captured: Some(role),
            ..
        } = e
        {
            if let Some(i) = roles.iter().position(|r| r == role) {
                roles.swap_remove(i);
            }
        }
    }
    if roles.is_empty() {
        return 0.0;
    }
    roles.iter().map(|&r| exchange_value(r)).sum::<f64>() / roles.len() as f64
}
/// ブラインド時のハング回避リスクの重み（クリーン粒子全滅時のみ。追補2）。
/// 個々の駒種・位置を特定しない「マスへの相手利き枚数の期待値」を使い、
/// 着地マスの被覆度が高いほど期待損失（駒の価値×密度）を引く。今までは
/// 全滅すると exposed_capture_risk 等が完全に働かず、ただ取られるリスクへの
/// 認識がゼロになっていた
const BLIND_HANG_RISK_W: f64 = 1.0;
/// taint_pool の上限（重み上位のみ使用。長手数対局での計算量爆発対策）
const TAINT_POOL_CAP: usize = 256;

/// taint 粒子を指紋でユニーク化し、深度減衰つきの重みで合算して返す
/// （taint_king_distribution・taint_square_coverage の共通部品。
/// 深い taint は信用が下がるので 0.5^(taint-1) で減衰し、
/// taint > TAINT_VOTE_MAX は除外する）
/// taint 粒子の敵玉を、観測から健全に絞れる候補集合 `cands` の中へ引き戻す。
///
/// taint は物理制約を緩めて延命させた盤面なので敵玉の位置も嘘になりうるが、
/// 「王手宣言の履歴」から絞れる玉位置は自分側の完全既知情報だけで決まるので、
/// そちらを信じてよい。**棄却せず移動する**のは玉位置の質量を潰さないため
/// （ansatsu 回帰: 王手中の再重み付けで玉位置の質量を潰すと悪化する）。
///
/// 移動先は現在位置から最も近い候補（チェビシェフ距離、同点は決定的に file→rank 順）。
/// その粒子で移動先に相手駒が居れば**入れ替える**（駒種の多重集合＝持ち駒会計を壊さない）。
///
/// `net` が Some（`TSUITATE_KING_NET_PROJ=1`）なら、移設先を最近傍でなく
/// 玉位置ネットの分布から選ぶ: 移設が必要な粒子（重み降順で数えた通し番号 j）に
/// CDF^{-1}((j+0.5)/m) の候補マスを割り当てる（決定的な分位点サンプル =
/// 分布に比例した割り当てで、全部を argmax に集めて多様性を潰さない）
fn project_taint_kings(
    pool: &[(&Position, f64)],
    cands: &std::collections::BTreeSet<Coord>,
    opp: Color,
    net: Option<&[(Coord, f64)]>,
) -> Vec<(Position, f64)> {
    // ネット誘導時は「移設が必要な粒子の数」を先に数えて分位点を等間隔に切る
    let need_move = |pos: &Position| -> Option<Coord> {
        let k = pos.king_square(opp)?;
        (!cands.is_empty() && !cands.contains(&k)).then_some(k)
    };
    let movers = net.map(|_| pool.iter().filter(|(p, _)| need_move(p).is_some()).count());
    let mut mover_idx = 0usize;
    pool.iter()
        .map(|&(pos, w)| {
            let Some(k) = need_move(pos) else {
                return (pos.clone(), w);
            };
            // **移設先は空きマスを優先する**: 下の移設は移設先の駒と入れ替えるので、
            // 埋まっているマスを選ぶと「その駒が別のマスへ瞬間移動した」という
            // 余計な嘘が1つ増える（玉の位置を直すために別の駒の位置を壊す）。
            // 候補の中に空きが1つも無いときだけ従来どおり入れ替える。
            // `TSUITATE_TAINT_KING_EMPTY=0` で従来挙動（アブレーション用）
            let empty_here = |c: &Coord| taint_king_prefer_empty() && pos.piece_at(*c).is_none();
            let target = match (net, movers) {
                (Some(dist), Some(m)) if m > 0 => {
                    let u = (mover_idx as f64 + 0.5) / m as f64;
                    mover_idx += 1;
                    // 空きマスだけで分位点を切り直す（全滅なら元の分布へ戻す）
                    let open: Vec<&(Coord, f64)> =
                        dist.iter().filter(|(c, _)| empty_here(c)).collect();
                    let (slice, total): (&[&(Coord, f64)], f64) = if open.is_empty() {
                        (&[], 0.0)
                    } else {
                        let t: f64 = open.iter().map(|(_, p)| p).sum();
                        (&open, t)
                    };
                    if slice.is_empty() || total <= 0.0 {
                        let mut acc = 0.0f64;
                        let mut chosen = dist.last().map(|(c, _)| *c);
                        for (c, p) in dist {
                            acc += p;
                            if acc >= u {
                                chosen = Some(*c);
                                break;
                            }
                        }
                        chosen
                    } else {
                        let mut acc = 0.0f64;
                        let mut chosen = slice.last().map(|(c, _)| *c);
                        for (c, p) in slice {
                            acc += p / total;
                            if acc >= u {
                                chosen = Some(*c);
                                break;
                            }
                        }
                        chosen
                    }
                }
                _ => {
                    let dist =
                        |a: Coord, b: Coord| (a.file - b.file).abs().max((a.rank - b.rank).abs());
                    cands
                        .iter()
                        .filter(|c| empty_here(c))
                        .min_by_key(|&&c| dist(c, k))
                        .or_else(|| cands.iter().min_by_key(|&&c| dist(c, k)))
                        .copied()
                }
            };
            let Some(t) = target else {
                return (pos.clone(), w);
            };
            let mut next = pos.clone();
            let displaced = next.piece_at(t);
            // 「玉を直すために他の駒を根拠なく飛ばした」割合。空きマス優先の
            // 効果はここで測る（TSUITATE_DBG_HITS=1）
            if crate::hits::enabled() {
                crate::hits::flag("taint_king_displaces_piece", displaced.is_some());
            }
            next.set(t, pos.piece_at(k));
            next.set(k, displaced);
            (next, w)
        })
        .collect()
}

fn taint_particles(est: &Estimator) -> Vec<(&Position, f64)> {
    let max_lw = est
        .log_weights()
        .iter()
        .zip(est.phys_taint())
        .filter(|&(_, &t)| t > 0 && t <= TAINT_VOTE_MAX)
        .map(|(&lw, _)| lw)
        .fold(f64::MIN, f64::max);
    if max_lw == f64::MIN {
        return vec![];
    }
    let mut seen: HashMap<u64, usize> = HashMap::new();
    let mut out: Vec<(&Position, f64)> = vec![];
    for ((pos, &t), &lw) in est
        .particles()
        .iter()
        .zip(est.phys_taint())
        .zip(est.log_weights())
    {
        if t == 0 || t > TAINT_VOTE_MAX {
            continue;
        }
        let w = (lw - max_lw).exp() * 0.5f64.powi(i32::from(t) - 1);
        match seen.entry(pos.fingerprint()) {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(out.len());
                out.push((pos, w));
            }
            std::collections::hash_map::Entry::Occupied(e) => out[*e.get()].1 += w,
        }
    }
    out
}

/// taint 粒子から相手玉の位置分布（正規化済み）だけを抽出する。
/// 深い taint は玉位置も信用が下がるので投票と同じ減衰・上限を適用
fn taint_king_distribution(particles: &[(&Position, f64)], opp: Color) -> Vec<(Coord, f64)> {
    let mut tally: HashMap<Coord, f64> = HashMap::new();
    let mut total = 0.0f64;
    for (pos, w) in particles {
        let Some(sq) = pos.king_square(opp) else {
            continue;
        };
        *tally.entry(sq).or_insert(0.0) += w;
        total += w;
    }
    if total <= 0.0 {
        return vec![];
    }
    tally.into_iter().map(|(sq, w)| (sq, w / total)).collect()
}

/// 指定マスへの相手利き枚数の期待値（taint 粒子由来）。個々の駒種・位置は
/// 特定せず**密度だけ**を見る — kakunari 分析でのユーザーの実際の推論
/// （「５七への相手利き≥2枚の確率が低い」）に対応する部品。
/// 攻め（信念マスへ利きを作る）だけでなく受け（信念被覆度が高いマスへの
/// 着地を避ける）にも使える
fn taint_square_coverage(particles: &[(&Position, f64)], sq: Coord, opp: Color) -> f64 {
    if particles.is_empty() {
        return 0.0;
    }
    let mut total_w = 0.0f64;
    let mut weighted_count = 0.0f64;
    for (pos, w) in particles {
        let n = pos
            .pieces()
            .filter(|(from, p)| p.color == opp && pos.attacks(*from, sq))
            .count();
        weighted_count += w * n as f64;
        total_w += w;
    }
    if total_w <= 0.0 {
        0.0
    } else {
        weighted_count / total_w
    }
}

/// ブラインド時の玉攻めボーナス: 候補手の着地駒が「信念上の玉マス」へ利きを
/// 作る度合い。自駒だけの盤（相手駒は不可視なので候補手生成と同じ仮定）で
/// 着地点からの利きを判定する — taint 粒子の盤面（嘘を含む）は使わない
fn blind_king_attack(view: &PlayerView, mv: &ShogiMove, dist: &[(Coord, f64)]) -> f64 {
    if dist.is_empty() {
        return 0.0;
    }
    // 自駒だけの盤面を作って候補手を適用する
    let mut pos = Position::empty(view.your_color);
    for p in &view.your_pieces {
        let (Some(sq), role) = (parse_usi_square(&p.square), p.role) else {
            continue;
        };
        pos.set(
            sq,
            Some(crate::shogi::Piece {
                color: view.your_color,
                role,
            }),
        );
    }
    for (role, n) in &view.your_hand {
        pos.set_hand(view.your_color, *role, *n as u8);
    }
    if !pos.is_pseudo_legal(mv) {
        return 0.0;
    }
    pos.play_unchecked(mv);
    let to = match *mv {
        ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
    };
    dist.iter()
        .map(|&(k, p)| if pos.attacks(to, k) { p } else { 0.0 })
        .sum()
}

/// ブラインド時のハング回避リスク: 着地マスの taint 由来の被覆度（期待利き
/// 枚数）× 着地する自駒の価値。相手駒は不可視なので着地駒の役割（成りを
/// 反映）だけで価値を決める。取り（着地に既に自駒がある＝取られる駒がない）
/// は対象外。cache は着地マスごとの被覆度の使い回し（成り/不成の同一着地マス
/// 等で同じスキャンを繰り返さない。長手数対局での計算量対策）
fn blind_hang_risk(
    view: &PlayerView,
    mv: &ShogiMove,
    taint_pool: &[(&Position, f64)],
    opp: Color,
    cache: &mut HashMap<Coord, f64>,
) -> f64 {
    let (to, role) = match *mv {
        ShogiMove::Board { from, to, promote } => {
            let Some(p) = view
                .your_pieces
                .iter()
                .find(|p| p.square == make_usi_square(from))
            else {
                return 0.0;
            };
            let role = if promote {
                promote_role(p.role).unwrap_or(p.role)
            } else {
                p.role
            };
            (to, role)
        }
        ShogiMove::Drop { role, to } => (to, role),
    };
    let coverage = *cache
        .entry(to)
        .or_insert_with(|| taint_square_coverage(taint_pool, to, opp));
    piece_value(role) * coverage
}

/// log(exp(a) + exp(b))（オーバーフロー安全）
fn logaddexp(a: f64, b: f64) -> f64 {
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    if hi == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    hi + (lo - hi).exp().ln_1p()
}

/// 記録用の推定サマリ: 粒子の健全性・ユニーク数・相手玉の位置分布（上位）。
/// 事後分析で「推定が外れていたのか、評価が悪かったのか」を切り分けるために残す
fn debug_summary(est: &Estimator, sample: &[(&Position, f64)], push: f64) -> serde_json::Value {
    let opp = est.my_color().other();
    // 層化で少数派にも最低枠が付くため、件数でなく重みで集計する
    let mut king_votes: HashMap<Coord, f64> = HashMap::new();
    let mut total_w = 0.0f64;
    // systematic resampling は同じ粒子を複数スロットに乗せうるので、
    // ユニーク数はスロット数（sample.len()）と別に指紋で数える
    let mut fingerprints = HashSet::new();
    for (pos, w) in sample {
        total_w += w;
        fingerprints.insert(pos.fingerprint());
        if let Some(sq) = pos.king_square(opp) {
            *king_votes.entry(sq).or_default() += w;
        }
    }
    let mut top: Vec<(Coord, f64)> = king_votes.into_iter().collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let n = total_w.max(1e-9);
    let opp_king_top: Vec<serde_json::Value> = top
        .iter()
        .take(3)
        .map(|(sq, votes)| {
            serde_json::json!({
                "sq": make_usi_square(*sq),
                "p": ((votes / n) * 1000.0).round() / 1000.0,
            })
        })
        .collect();
    serde_json::json!({
        "healthy": est.healthy(),
        "unique_particles": fingerprints.len(),
        "sample_slots": sample.len(),
        "soft_particles": est.info_miss().iter().filter(|&&p| p > 0).count(),
        "taint_particles": est.phys_taint().iter().filter(|&&t| t > 0).count(),
        "ess": (est.last_ess() * 10.0).round() / 10.0,
        "resamples": est.resamples(),
        "endgame_push": (push * 100.0).round() / 100.0,
        "opp_king_top": opp_king_top,
    })
}

/// 自分に見える範囲の候補手（foul_tried を除く）。bin/analyze の検証でも使う
pub fn candidate_moves(
    view: &PlayerView,
    foul_tried: &HashSet<String>,
) -> Vec<(String, ShogiMove)> {
    let color = view.your_color;
    let mut out = vec![];
    let push = |usi: String, out: &mut Vec<(String, ShogiMove)>| {
        if !foul_tried.contains(&usi) {
            if let Some(mv) = parse_usi(&usi) {
                out.push((usi, mv));
            }
        }
    };
    for piece in &view.your_pieces {
        let Some(from) = parse_usi_square(&piece.square) else {
            continue;
        };
        for to in move_targets(&view.your_pieces, piece, color) {
            match promotion_choice(piece.role, from, to, color) {
                Promotion::None => push(make_usi_move(from, to, false), &mut out),
                Promotion::Forced => push(make_usi_move(from, to, true), &mut out),
                Promotion::Optional => {
                    // 成れるなら成る（不成が有利な局面はまれなので候補を絞る）
                    push(make_usi_move(from, to, true), &mut out);
                }
            }
        }
    }
    for (&role, &count) in &view.your_hand {
        if count == 0 {
            continue;
        }
        for to in drop_targets(&view.your_pieces, role, color) {
            if let Some(usi) = make_usi_drop(role, to) {
                push(usi, &mut out);
            }
        }
    }
    out
}

/// 自玉のマス（PlayerView の自駒リストから引く）
fn king_square(view: &PlayerView) -> Option<Coord> {
    view.your_pieces
        .iter()
        .find(|p| p.role == Role::King)
        .and_then(|p| parse_usi_square(&p.square))
}

/// 王手されているとき、この手が王手を解消しうるか（自分に見える情報だけで判定）。
/// 解消手段は (a) 玉を動かす (b) 王手駒を取る (c) 合駒。王手駒の位置は不明でも
/// (b) の着地点は自玉に利きが通るマス（クイーンライン上か桂の利き元）、
/// (c) は玉と王手駒の間（クイーンライン上）に限られる。
/// どれにも該当しない手は王手放置で必ず反則になる
fn may_resolve_check(view: &PlayerView, mv: &ShogiMove) -> bool {
    let Some(king) = king_square(view) else {
        return true; // 玉が見つからないなら判定不能（除外しない）
    };
    let on_ray = |to: Coord| {
        let df = to.file - king.file;
        let dr = to.rank - king.rank;
        (df != 0 || dr != 0) && (df == 0 || dr == 0 || df.abs() == dr.abs())
    };
    // 相手の桂が自玉に利くマス（桂の王手は取るしかなく、合駒では防げない）
    let knight_source = |to: Coord| {
        let dr = match view.your_color {
            Color::Sente => -2, // 相手（後手）の桂は rank+2 へ利く → 利き元は rank-2 側
            Color::Gote => 2,
        };
        (to.file - king.file).abs() == 1 && to.rank - king.rank == dr
    };
    match *mv {
        ShogiMove::Board { from, to, .. } => {
            if from == king {
                return true; // 玉を動かす
            }
            on_ray(to) || knight_source(to)
        }
        // 打ちは駒を取れないので合駒（ライン上）のみ
        ShogiMove::Drop { to, .. } => on_ray(to),
    }
}

/// 王手中の p(合法) 補正係数。玉移動が最も解消しやすく、
/// 取り/合駒は王手駒の位置に当たっている必要があるので低め
fn in_check_prior(view: &PlayerView, mv: &ShogiMove) -> f64 {
    match *mv {
        ShogiMove::Board { from, .. } if Some(from) == king_square(view) => 0.5,
        _ => 0.25,
    }
}

/// 観測ゼロでも成り立つ p(合法) の事前確率。
/// 経路上の「中身の見えないマス」1つごとに空である確率 q を掛ける。
/// 打ちは着地点が空である確率 q（隠れた相手駒の上に打つのが典型的な反則源）
fn prior_legal(view: &PlayerView, mv: &ShogiMove, opp_board_n: f64) -> f64 {
    let my_n = view.your_pieces.len() as f64;
    let q = (1.0 - opp_board_n / (81.0 - my_n)).clamp(0.05, 1.0);
    match *mv {
        ShogiMove::Board { from, to, .. } => {
            let df = to.file - from.file;
            let dr = to.rank - from.rank;
            let aligned = df == 0 || dr == 0 || df.abs() == dr.abs();
            // 候補手は自駒には塞がれていないので、中間マスはすべて未知マス
            let unknown = if aligned {
                (df.abs().max(dr.abs()) - 1).max(0)
            } else {
                0 // 桂・1マス移動
            };
            q.powi(unknown as i32)
        }
        ShogiMove::Drop { .. } => q,
    }
}

/// 相手が位置を知っている自駒の地図（マス → 既知度 0.0〜1.0）。
///
/// 対人対局の分析（records/ 2026-07-08）より: 相手は (a) 自駒が死んだマス =
/// こちらの駒がいるマス、(b) 初期配置から動いていない駒、に当たりを付けて
/// 一方的に駒を回収してくる。ついたて将棋で相手に漏れる自駒の位置情報は
/// この2種類が主なので、露出リスクの重み付けに使う
/// - 1.0: 駒を取って位置が暴露し、以降動いていない駒
/// - home_knownness: 初期配置から一度も動いていない駒（相手は初期配置を知っている）
fn knownness_map(
    view: &PlayerView,
    log: &ObservationLog,
    home_knownness: f64,
) -> HashMap<Coord, f64> {
    let mut revealed: HashSet<Coord> = HashSet::new();
    let mut touched: HashSet<Coord> = HashSet::new();
    for e in log.events() {
        match e {
            Observation::MyMove { usi, captured, .. } => match parse_usi(usi) {
                Some(ShogiMove::Board { from, to, .. }) => {
                    revealed.remove(&from);
                    if captured.is_some() {
                        revealed.insert(to);
                    } else {
                        revealed.remove(&to);
                    }
                    touched.insert(from);
                    touched.insert(to);
                }
                Some(ShogiMove::Drop { to, .. }) => {
                    // 打った駒の位置は相手から見えない
                    revealed.remove(&to);
                    touched.insert(to);
                }
                None => {}
            },
            Observation::OpponentMoved {
                captured_my_piece_at: Some(sq),
                ..
            } => {
                if let Some(c) = parse_usi_square(sq) {
                    revealed.remove(&c);
                }
            }
            _ => {}
        }
    }

    let initial = Position::initial();
    let mut map = HashMap::new();
    for piece in &view.your_pieces {
        let Some(sq) = parse_usi_square(&piece.square) else {
            continue;
        };
        let k = if revealed.contains(&sq) {
            1.0
        } else if !touched.contains(&sq)
            && initial
                .piece_at(sq)
                .is_some_and(|p| p.color == view.your_color && p.role == piece.role)
        {
            home_knownness
        } else {
            0.0
        };
        if k > 0.0 {
            map.insert(sq, k);
        }
    }
    map
}

/// 敵陣のマスが（見えない駒に）守られている事前確率。
/// 粒子が枯渇・偏っていて守り駒を見落としていても、敵陣への単騎突入
/// （対人5局で歩→高価な駒の損な交換が9回）を抑えるための下限に使う
fn camp_defended_prior(to: Coord, me: Color, camp_scale: f64) -> f64 {
    let depth_from_back = match me {
        Color::Sente => to.rank,     // 相手（後手）の陣は rank 1..=3
        Color::Gote => 10 - to.rank, // 相手（先手）の陣は rank 7..=9
    };
    camp_scale
        * match depth_from_back {
            1 => 0.25,
            2 => 0.2,
            3 => 0.15,
            _ => 0.0,
        }
}

/// 候補手をユニーク粒子の加重平均で評価する（重み = ソフト救済の減衰）
#[allow(clippy::too_many_arguments)]
fn evaluate(
    view: &PlayerView,
    mv: &ShogiMove,
    particles: &[(&Position, f64)],
    // particles が taint 粒子（情報系の制約を緩めて生かした粒子）かどうか。
    // 真なら **p_legal は事前確率のみ**にする: taint は「反則の説明」を
    // 緩和して生き残った粒子なので、合法性の証拠に使ってはいけない
    // （厳密粒子ゼロのときの従来挙動 p_legal = prior と一致する）
    particles_are_taint: bool,
    // 詰めろ生成の判定に使う粒子プール。厳密粒子があれば `particles` と同じ、
    // 全滅していれば taint 粒子（choose() が渡す）。終盤のブラインドでは
    // 厳密粒子が枯れるので、詰めろが効く局面ほど `particles` は空になる
    mate_pool: &[(&Position, f64)],
    prior: f64,
    known: &HashMap<Coord, f64>,
    params: &EvalParams,
    budget: SearchBudget,
    // valueネットのstate特徴量キャッシュ（particles と同じ並び。候補間で共通なので
    // choose() が1手番ぶん保持し、最初に使う候補の評価時に遅延計算する）
    nn_state_cache: &mut [Option<[f64; crate::value_features::VALUE_FEATURES]>],
    // 直前に自駒を取られたマスと、そこにいる敵駒の期待交換価値（観測のみで決まる）。
    // 厳密粒子が全滅した決定でだけ使う（`blind_recapture_target`）
    blind_recapture_target: Option<(Coord, f64)>,
    // 信念ネットのマスごと占有確率と、盤上に残る相手駒の平均交換価値。
    // 上と同じく**厳密粒子が全滅した決定でだけ**使う（`belief_gain_w`）。
    // 重みが 0 なら choose() が None を渡すので計算もしない
    blind_belief: Option<(&[f64; 81], f64)>,
    // ブラインド決定での taint 占有合意（`taint_occ_legal_w`）。打ちの反則確率
    // にだけ使う（Some のときだけ有効。choose() が w≠0・厳密粒子ゼロの
    // ゲートを掛けて渡す）
    taint_occ: Option<&[f64; 81]>,
    // V2: マスごとの「相手玉の信念位置からの距離重み」（`opp_king_effect_weights`）。
    // 決定点ごとに1度だけ作って全候補で使い回す
    opp_king_w: Option<&[f64; 81]>,
    // 打ち当て露出（`drop_hit_evac_w`）の**現局面**の露出。Some のときだけ項が
    // 有効（choose() が「w≠0・相手の確定持ち駒に歩あり・王手中でない」の
    // ゲートを掛けて渡す）。着手後の露出との差分を gain へ加点する
    drop_hit_expo_before: Option<f64>,
    // 成りポテンシャル（`promo_potential_w`）の**現局面**の値。Some のときだけ
    // 項が有効（choose() が「w≠0・王手中でない」のゲートを掛けて渡す）。
    // 着手後との差分を gain へ加点する
    promo_pot_before: Option<f64>,
    // 大駒の成り道（`major_promo_path_w`）の**現局面**の値。同じく差分で使う
    major_path_before: Option<f64>,
    // 持ち駒オプション価値（`hand_option_w`）の決定点コンテキスト。Some のとき
    // だけ項が有効（choose() が「w≠0・王手中でない」のゲートを掛けて渡す）。
    // 打つ手に「最良打ちポテンシャルとの不足分」の減点を掛ける
    hand_option: Option<&HandOption>,
    // 自分が打ちの反則をしたマス（`drop_probe_repeat_gate` の再プローブ判定）
    drop_foul_squares: &[bool; 81],
) -> EvalOut {
    let me = view.your_color;
    let opp = me.other();
    let mut legal = 0.0f64;
    let mut value_sum = 0.0;
    let mut risk_sum = 0.0;
    // 着地マスに敵駒がいた（=駒を取れた）粒子の重み。探索ボーナスの不一致度に使う
    let mut capture_hits = 0.0f64;
    // 捕獲価値の重み付き和（賭け分散ペナルティの stake = E[捕獲価値|hit] 用）
    let mut capture_value_sum = 0.0f64;
    // 王手になった粒子の重み。王探しの情報利得（判定が割れるほど価値）に使う
    let mut check_hits = 0.0f64;
    // 詰みになった粒子の重み。加点はループ後に凸ゲート（mate_gate_q0）を通す
    let mut mate_hits = 0.0f64;
    // 打ちプローブの反則情報価値の材料（drop_probe_w。反則枝の粒子だけが足す）
    let mut probe_val_sum = 0.0f64;
    // 「占有かつ自利きあり」の粒子質量（プローブ項の凸ゲート用）
    let mut probe_mass = 0.0f64;
    // 王周辺の圧力は粒子間の分散が小さいわりに計算が重い（9マス×利き走査）ので
    // 少数の粒子でだけ測って平均する（数は思考予算に比例）
    let pressure_samples = budget.pressure_samples;
    let mut pressure_sum = 0.0;
    let mut attack_sum = 0.0;
    let mut danger_sum = 0.0;
    // 逃げマス被覆（escape_cover_w）。圧力項と同じ少数粒子サンプルで測る
    let mut escape_sum = 0.0;
    let mut pressure_n = 0usize;
    // 圧力項もソフト粒子の重みで加重する（他の項と同じ扱い）
    let mut pressure_w_sum = 0.0f64;
    // valueネット（粒子=真の局面仮説ごとの勝率相当を重み付き平均）。
    // 圧力項と同じく少数の粒子でだけ測る（transition特徴量の利き走査が重い）
    let mut nn_sum = 0.0f64;
    let mut nn_w_sum = 0.0f64;
    let mut nn_n = 0usize;

    for (pi, &(pos, w)) in particles.iter().enumerate() {
        if !pos.is_legal(mv) {
            // 打ちプローブの反則情報価値の材料（drop_probe_w）: この粒子で
            // 打ちマスが相手駒に塞がれていて、かつ自分の利きが既に当たって
            // いるなら、反則の失敗枝が「占有の確定 → 次手で回収」に変換できる
            if params.drop_probe_w != 0.0 && !view.you_in_check && !particles_are_taint {
                if let ShogiMove::Drop { to, .. } = *mv {
                    if let Some(p) = pos.piece_at(to) {
                        if p.color == opp && pos.attack_count(to, me) > 0 {
                            probe_val_sum += w * exchange_value(p.role);
                            probe_mass += w;
                        }
                    }
                }
            }
            continue;
        }
        legal += w;
        let mut v = 0.0;

        // 駒得（盤上価値で数える。成駒を取れば大きい）
        let mut captured_value = 0.0;
        if let ShogiMove::Board { to, .. } = *mv {
            if let Some(p) = pos.piece_at(to) {
                if p.color == opp {
                    captured_value = exchange_value(p.role);
                }
            }
        }
        v += captured_value;
        if captured_value > 0.0 {
            capture_hits += w;
            capture_value_sum += w * captured_value;
        }

        let mut next = pos.clone();
        next.play_unchecked(mv);

        // 王手・詰み。ついたて将棋では王手された側は王手駒の位置が見えず
        // 手探りの反則をしやすい（反則10回で負け）ので、王手自体が得点源。
        // 相手の反則が溜まっているほど価値が上がり、上限（反則負け）に
        // 近づくほど1回の誘発の限界価値が跳ねるので check_limit_accel で加速する
        // （オラクル測定 2026-07-16: 王手中反則の完全知識だけで vs v6 +9.5pt）
        let gives_check = next.in_check(opp);
        if gives_check {
            let opp_fouls_left = f64::from(10u32.saturating_sub(view.fouls.opponent).max(1));
            let accel = (10.0 / opp_fouls_left).powf(params.check_limit_accel);
            v += params.check_bonus
                + params.check_foul_scale * f64::from(view.fouls.opponent) * accel;
            check_hits += w;
            // K = この粒子での相手の合法解消手数（王手中の合法手 = 解消手）。
            // 0 なら詰み。それ以外は「王手の強さ」で再配分する（check_strength_w
            // の doc 参照。逃げ場・合駒・王手駒の取りが少ない王手ほど、受け側は
            // 正解を引くまで反則を積む — tuyoi_oote / rei2 のユーザー指導）
            let resolutions = next.legal_moves().len();
            if resolutions == 0 {
                // 詰み（真の局面がこの粒子なら勝ち）。ここでは質量だけ数え、
                // 加点はループ後に詰み質量 q の凸ゲート（mate_gate_q0）を通す:
                // 較正の悪い信念の裾（q≈0.1 の幻詰み）が +1000 で全候補を
                // 乗っ取るのを防ぐ（quest31-m027/m029 の 4一龍、gain 77〜91）
                mate_hits += w;
            } else if params.check_strength_w != 0.0 {
                v += params.check_strength_w
                    * (CHECK_STRENGTH_CURVE / (1.0 + resolutions as f64)
                        - CHECK_STRENGTH_CENTER);
            }
        }

        // 守り駒の捕獲: 相手玉（この粒子）の8近傍にいる相手駒を取る手は、
        // 交換価値とは別に「守りが1枚減る」価値を持つ（対人局の6一成銀:
        // 金取り＋守り駒削減が主目的、というユーザー聞き取り。
        // human-play-review-2026-07-29）。王手中は無効（CheckSolver の領分）
        if params.defender_capture_w != 0.0 && !view.you_in_check && captured_value > 0.0 {
            if let (ShogiMove::Board { to, .. }, Some(ok)) = (*mv, pos.king_square(opp)) {
                if cheb(to, ok) <= 1 {
                    v += params.defender_capture_w;
                }
            }
        }

        // 取られリスクは「相手がこの駒の位置を知っているか」で重みを分ける。
        // 駒を取った直後は取られたマスが相手に通知される → 着手駒の位置は確実にバレて
        // いて、取り返しはほぼ実行される。それ以外の駒への当たりは相手から見えない
        // （推定はされうる）ので薄く見積もる
        let to = match *mv {
            ShogiMove::Board { to, .. } => to,
            ShogiMove::Drop { to, .. } => to,
        };
        // 相手が取れるのは1手で1枚なので、重み付きリスクの最大値だけを引く。
        // 敵陣への着手は「粒子には見えない守り駒がいる」事前確率を下限に敷く
        // （駒を取った直後は位置が確実にバレているので下限をフルに、静かな
        // 進入は相手からまだ見えないので薄く適用する）
        // 王手をかけた手は王手宣言で位置の仮説が絞られ、相手は反則覚悟の
        // 探り取りで回収に来る（人間の実証済み戦術）ので、露見扱いにする
        let mut mover_w = if captured_value > 0.0 {
            params.mover_w_captured
        } else {
            params.mover_w_quiet
        };
        if gives_check {
            mover_w += params.mover_check_extra;
        }
        let own_after = next
            .piece_at(to)
            .map(|p| exchange_value(p.role))
            .unwrap_or(0.0);
        // 玉隣接への無支え進入（king_adj_entry_w）: この粒子で着地マスが相手玉の
        // 8近傍にあり自分の利きの支えが無いなら、王手宣言・接触で存在がバレて
        // 玉や近傍の守りに回収される前提の回収リスクを駒価値スケールで敷く
        // （実測: 直接王手の53〜56%が即取られ。quest31 の 4一龍/4一と が発端）。
        // 王手中は無効（回避手の序列は CheckSolver の領分）
        if params.king_adj_entry_w != 0.0 && !view.you_in_check {
            if let Some(ok) = next.king_square(opp) {
                if cheb(to, ok) <= 1 && next.attack_count(to, me) == 0 {
                    v -= params.king_adj_entry_w * own_after;
                }
            }
        }
        let known_factor = if captured_value > 0.0 {
            1.0
        } else {
            params.camp_known_quiet
        };
        let mut floor = own_after * camp_defended_prior(to, me, params.camp_scale) * known_factor;
        if captured_value > 0.0 {
            // 取ったマスは相手に通知される。粒子に守りが見えなくても
            // 取り返しの残留リスクを敷く（= 等価な取りは安い駒で取る）
            floor = floor.max(own_after * params.capture_reveal_risk);
        }
        let mover_risk =
            mover_w * recapture_risk(&next, me, to, params.recapture_defended).max(floor);
        let hidden_risk = exposed_capture_risk(&next, me, Some(to), known, params);
        // ここの max は**外さない**（2026-08-03 実測）。着手駒の危険と
        // 置き去りの危険を `top + w·min` で足す版は、着手駒側に
        // `capture_reveal_risk` の床（取ったマスは相手に通知される）が
        // 乗っているぶん**取る手を余計に罰する**: quest31-m026 で
        // 4七歩成 と 2六歩打 の差が w=0/0.3/0.5 で 0.322/0.272/0.259 と
        // 逆に縮んだ。複数枚計上は `exposed_capture_risk` の内側だけに留める
        let risk = mover_risk.max(hidden_risk);
        v -= risk;
        risk_sum += w * risk;

        // 自分が敵駒に当たりを付けている価値（露出リスクの鏡像）。
        // 1手読みでは見えない「次の駒得」を作る手（大駒の頭への歩打ち等）に価値を与える
        v += params.threat_w * threat_value(&next, me);

        // 王の安全度と攻撃圧力（利き走査が重いので少数の粒子でだけ測って平均する）
        if pressure_n < pressure_samples {
            // 自玉の周囲に当たっている相手の利き（守り）
            pressure_sum += w * king_zone_pressure(&next, me, opp);
            // 相手玉の周囲に当たっている自分の利き（攻め）。王手にならない攻め駒の
            // 集結にも報酬を与える（王手/詰みボーナスだけだと攻めを組み立てない）
            attack_sum += w * king_zone_pressure(&next, opp, me);
            // 相手の持ち駒による王手打ちの受け入れ面積（対局実験の教訓:
            // 飛車を持たれた瞬間、玉への開いた直線はすべて即王手の入口になる）
            danger_sum += w * drop_check_danger(&next, me);
            // 逃げマス被覆（凸）: 相手玉の未被覆の逃げ先が残り U 個のとき
            // 1/(1+U)。「最後の逃げ道を塞ぐ手」ほど増分が大きい
            // （escape_cover_w の doc 参照）。王手中は無効
            if params.escape_cover_w != 0.0 && !view.you_in_check {
                escape_sum += w * escape_cover_value(&next, opp, me);
            }
            pressure_w_sum += w;
            pressure_n += 1;
        }

        // valueネット: 学習時の規約（state=指す前の局面・指す側視点、transition=
        // その一手。docs/nn-value-phase1.md）どおり、粒子=真の局面仮説として推論する。
        // state特徴量は候補間で共通なので粒子単位にキャッシュする。
        // **自分が王手されている間は無効**: 王手回避は CheckSolver（制約推論）の
        // 領分で、NNの加点が回避プローブの反則試行を増やす実測があった
        // （dragon-check-drop で w=6 時に反則負け2/20が発生。w選定スイープ
        // 2026-07-22）。王手中の候補序列は p_legal（解消確率）が支配すべき
        if params.value_nn_w != 0.0 && !view.you_in_check && nn_n < budget.nn_samples {
            let state = nn_state_cache[pi]
                .get_or_insert_with(|| crate::value_features::value_features(pos, me));
            let trans = crate::value_features::transition_features(pos, mv, &next, me);
            let mut f = [0.0f64;
                crate::value_features::VALUE_FEATURES + crate::value_features::TRANSITION_FEATURES];
            f[..crate::value_features::VALUE_FEATURES].copy_from_slice(state);
            f[crate::value_features::VALUE_FEATURES..].copy_from_slice(&trans);
            nn_sum += w * crate::value_nn::value_nn_forward(&f);
            nn_w_sum += w;
            nn_n += 1;
        }

        value_sum += w * v;
    }

    // 粒子の証拠と事前確率のブレンド（粒子ゼロなら事前そのもの）。
    // 粒子が退化している（実効重みが評価上限に届かない）ほど事前の重みを
    // 増やし、少数の偏った粒子への過信を防ぐ。ソフト粒子は重みぶんしか
    // 数えないので、退化度にも自然に反映される
    let n: f64 = particles.iter().map(|(_, w)| w).sum();
    let degen = 1.0 - (n / budget.eval_particles as f64).min(1.0);
    let w = params.prior_weight + params.prior_weight_degen * degen;
    let mut p_legal = if particles_are_taint {
        prior
    } else {
        (legal + prior * w) / (n + w)
    };
    // taint 占有合意で打ちの反則確率を締める（`taint_occ_legal_w`）。
    // 打ちマスが埋まっていれば必ず反則なので、合意占有率をそのまま反則確率に
    // 使える。**安全方向のみ**（min）: 空きマスの打ちを押し上げはしない
    if let (Some(occ), ShogiMove::Drop { to, .. }) = (taint_occ, *mv) {
        let p_occ = occ[crate::belief_features::sq_index(to)];
        p_legal = p_legal.min(1.0 - params.taint_occ_legal_w * p_occ).max(0.0);
    }
    let p_legal = p_legal;
    // 賭け分散ペナルティの内訳（ランキング表示用に expected の外へ持ち出す）
    let mut capture_bet_penalty = 0.0;
    // valueネット項の内訳（同上。発火率フック src/hits.rs が使う）
    let mut value_nn_term = 0.0;
    // ブラインド時（厳密粒子ゼロ）の取り返し。粒子が無いと `expected` が丸ごと
    // ゼロになり、**位置が確実に分かっている敵駒を取る手**の駒得まで消える
    // （実測: 龍を取り返す手が gain 11〜13 → 0.045）。観測だけで決まる量なので
    // 粒子に依らずここで補う。取った後の取り返されリスク（下限）も同じ式で引く
    let blind_recapture = if particles.is_empty() {
        blind_recapture_target
            .filter(|&(sq, _)| matches!(*mv, ShogiMove::Board { to, .. } if to == sq))
            .map(|(sq, value)| {
                let own_after = view
                    .your_pieces
                    .iter()
                    .find(|p| {
                        matches!(*mv, ShogiMove::Board { from, .. } if p.square == make_usi_square(from))
                    })
                    .map(|p| exchange_value(p.role))
                    .unwrap_or(0.0);
                let _ = sq;
                blind_recapture_w()
                    * (value - params.mover_w_captured * own_after * params.capture_reveal_risk)
            })
            .unwrap_or(0.0)
    } else {
        0.0
    };
    // ブラインド時の信念ネット供給（NN段階②）。blind_recapture の一般化で、
    // 「相手駒が確実にいる1マス」の代わりに **81マスの占有確率**を使う。
    // 素の期待値は blind_recapture と同じ形（p=1 なら一致する）。
    //
    // 打ちは対象外: 占有マスへの打ちは捕獲ではなく反則なので、供給先は
    // p_legal 側になってしまう（反則マス記憶系4種が全滅したチャネル）。
    // 直前に取られたマスは blind_recapture が p=1 で見ているので二重計上しない。
    //
    // **賭け分散の凹割引を通常経路と同じ式で引く**（`capture_bet_var_w`）:
    // 素の p×stake は空振り分岐（賭けの前提が崩れ、進出駒だけが未知領域に
    // 残る側）の質を数えない。初版（割引なし・王手中ゲート無し）は
    // 200局ペア比較で3シャードとも対照割れ（42.8% vs 47.3%）し、
    // **平均手数 104.5 → 112.8・手数上限による引き分け 0 → 5** という
    // 「投機的な捕獲で手数だけ伸びる」形が出た。p が中間値のときに最も
    // 加点される素の式ではこの分岐が数えられていない。
    //
    // **王手中は無効**: 候補の序列は解消確率（CheckSolver）が支配すべきで、
    // 王手中の攻め加点は回避プローブの反則を増やす（value_nn_w / mate_*_w /
    // capture_bet_var_w と同じゲート）。初版はシナリオ dragon-check-drop の
    // 反則を 12 → 29 に増やしていた
    let blind_belief_gain = match (particles.is_empty(), blind_belief, *mv) {
        (true, Some((occ, mean_value)), ShogiMove::Board { from, to, .. })
            if !view.you_in_check && blind_recapture_target.is_none_or(|(sq, _)| sq != to) =>
        {
            let p = occ[crate::belief_features::sq_index(to)];
            let own_after = view
                .your_pieces
                .iter()
                .find(|q| q.square == make_usi_square(from))
                .map(|q| exchange_value(q.role))
                .unwrap_or(0.0);
            let stake = mean_value - params.mover_w_captured * own_after * params.capture_reveal_risk;
            belief_gain_w() * (p * stake - params.capture_bet_var_w * p * (1.0 - p) * mean_value)
        }
        _ => 0.0,
    };
    // 攻め圧力は粒子の健全度でゲートする。退化した粒子は間違った玉位置に
    // 固まりやすく、「誰もいない場所への攻め」が加点され続ける
    // （対人実戦: 終盤の成桂の徘徊）。健全度が低いときは確実な項だけ残す
    let confidence = (n / budget.eval_particles as f64).min(1.0);
    let expected = if legal > 0.0 {
        // 探索ボーナス: 着地マスの敵駒有無について粒子が割れているほど、
        // 指せば（取れても空でも）推定が絞れる。捕獲の期待値とは別の情報の価値
        let p_hit = capture_hits / legal;
        // 王探し: 王手判定が粒子間で割れる手は、指せば王手宣言の有無で
        // 玉位置仮説が絞れる（互角膠着で「玉が見つからない」を崩す勾配）
        let p_chk = check_hits / legal;
        // valueネット項: 勝率相当[0,1]の重み付き平均を中心化して歩価値スケールへ。
        // gain の内側（= combine_score の p_legal 割引を受ける側）に置くことで、
        // 反則確実な手への加点素通り（dragon-check-drop の教訓）を構造的に防ぐ
        value_nn_term = if nn_w_sum > 0.0 {
            params.value_nn_w * (nn_sum / nn_w_sum - 0.5)
        } else {
            0.0
        };
        // 捕獲の賭け分散ペナルティ: 期待駒得が「占有が割れているマスへの
        // 大きな捕獲」1本に集中している手を凹に割り引く。素の期待値
        // p×stake は空振り分岐（賭けの前提が崩れ、進出駒だけが未知領域に
        // 残る側）の質の悪さを見ないため、信念が五分に近いほど・賭け金
        // （stake = E[捕獲価値|hit]）が大きいほど p(1−p)×stake で課金する。
        // 同じ1ビット（マスの占有）を買うなら安い駒のプローブが相対的に
        // 浮く設計（play-estimator-20260724 16手目: 8八と>8八歩打の逆転が発端）。
        // 王手中は無効: 王手駒捕獲の序列は CheckSolver（removal_term・p_legal）の
        // 領分で、五分の信念での捕獲プローブはむしろ推奨挙動（kakutori）
        if capture_hits > 0.0 && !view.you_in_check {
            capture_bet_penalty =
                params.capture_bet_var_w * p_hit * (1.0 - p_hit) * (capture_value_sum / capture_hits);
        }
        // 粒子上の詰みの加点。q = 詰みを主張する質量の割合に対し
        // 1000×q×(q/(q+q0)) の凸ゲート: q0=0 は従来（1000×q）と同一挙動、
        // q0>0 では裾の幻詰みが材料スケールへ沈み、合意の詰みはほぼ満額残る
        let mate_term = if mate_hits > 0.0 {
            let q = mate_hits / legal;
            1000.0 * q * (q / (q + params.mate_gate_q0))
        } else {
            0.0
        };
        value_sum / legal + mate_term - capture_bet_penalty
            + params.info_bonus * p_hit * (1.0 - p_hit)
            + params.king_probe_bonus * p_chk * (1.0 - p_chk)
            + value_nn_term
            + (params.attack_w * confidence * attack_sum
                + params.escape_cover_w * confidence * escape_sum
                - params.pressure_w * pressure_sum
                - params.hand_drop_w * danger_sum)
                / pressure_w_sum.max(1e-9)
    } else {
        0.0
    };

    // 反則コスト: 手番は失わないが反則数を消費する。残りが少ないほど急激に高価。
    // 序盤の「安い反則で情報を得る」は低コスト側で自然に許容される。
    // 勝敗は反則レース（先に10回）なので、コストは絶対値でなく**残数差の相対価値**:
    // 相手が上限間際（残数小）なら自分の1反則は相対的に安い（foul_diff_pow で調整。
    // 0 = 従来どおり自分の残数のみ。tune-round3 の分析でスコアと反則差の相関0.75）
    let fouls_left = (10u32.saturating_sub(view.fouls.you)).max(1) as f64;
    let opp_fouls_left = (10u32.saturating_sub(view.fouls.opponent)).max(1) as f64;
    let foul_cost = params.foul_cost_base
        * (10.0 / fouls_left).powf(params.foul_cost_pow)
        * (opp_fouls_left / 10.0).powf(params.foul_diff_pow);

    // 前進の弱い事前バイアス（推定が薄い序盤に駒をぶつけに行くため）
    let advance_bias = match *mv {
        ShogiMove::Board { from, to, promote } => {
            let adv = match me {
                Color::Sente => (from.rank - to.rank) as f64,
                Color::Gote => (to.rank - from.rank) as f64,
            };
            params.advance_w * adv + if promote { params.promote_bias } else { 0.0 }
        }
        ShogiMove::Drop { .. } => params.drop_bias,
    };

    // 大駒を初期位置に置き続けるペナルティ（この手の後に残る枚数分）。
    // 動かす手だけペナルティが軽くなるので、展開への勾配になる
    let development = -params.big_home_penalty * big_home_after(view, mv);

    // 利き被覆（広い索敵網）。粒子に依存しない自明な情報だけで計算できる
    let own_effects = own_effects_after(view, mv, opp_king_w, params);
    let coverage = params.coverage_w * own_effects.coverage;
    // 自玉8近傍の支えの無いマス（V4）。自駒だけで決まるので粒子に依らない
    let king_holes = params.king_hole_w * own_effects.king_holes;
    // V3（予防的な紐、やねうら王 Lv7 で +R25）: 紐のついた自駒の価値合計。
    // **ついたてでは将棋よりこの項の価値が高いはず**という理屈がある:
    // 将棋なら「狙われてから紐をつける」で間に合うが、ついたては相手の攻めが
    // 見えないので狙われたことに気づけない。事前に紐がついている駒は、
    // 気づかないまま只取られされる確率がそもそも低い。
    // 既存の紐（recapture_risk / exposed_capture_risk の割引）は
    // **すでに攻撃されている駒にしか効かない**ので、この事前の勾配が無かった。
    // 自駒同士の連結は完全既知なので粒子不要・ノイズゼロで計算できる。
    //
    // **王手中もそのまま効かせる**（value_nn / 詰めろ2項 / capture_bet_var とは
    // 逆の判断）。他の項と同じ王手中ゲートを一度入れたが、実測で不採用にした:
    //
    // | 条件 | vs v9 | vs v10 | vs v11 | kakutori 捕獲 | kakutori 反則 |
    // |---|---|---|---|---|---|
    // | ゲート無し | 60.3% | 61.3% | 61.5% | 3/20 | **7** |
    // | ゲート有り | 58.0% | 51.8% | 52.0% | 18/20 | **51** |
    // （各200局・match_seed=20260728 のペア比較）
    //
    // kakutori の「捕獲 17/20 → 3/20」は一見回帰だが**反則が 51 → 7 に激減**して
    // おり、実際には別の合法手（3g2g、12/20）で王手を解消できている
    // （反則7 < 選択12 なのでこの手は合法）。捕獲率は「王手駒を見つけられるか」の
    // 代理指標として置いたもので、より安く王手を解消できるならそちらが良い。
    // 王手中は反則が最も出る場面（analyze: 反則の約半数）なので、
    // ここで紐を切ると反則経済がそのまま悪化する（vs v11 の反則/局 6.20 → 6.71）
    let link = params.link_w * own_effects.linked_value;

    // 打ち当て露出の差分（`drop_hit_evac_w` の doc 参照）。退避・頭を自駒で
    // 埋める手が正、相手が歩を持つ局面で大駒を敵陣へ突っ込む手が負。
    // 大半の候補は 0 なので gain のゼロ点は動かない
    let drop_hit_evac = drop_hit_expo_before
        .map(|before| params.drop_hit_evac_w * (before - own_effects.drop_hit_exposure))
        .unwrap_or(0.0);

    // 成りポテンシャルの差分（`promo_potential_w` の doc 参照）。未成駒を
    // 成りマスへ近づける手・垂れ歩・成る手自体が正、後退・自駒の道を塞ぐ
    // 打ちが負。大半の候補は 0 なので gain のゼロ点は動かない
    let promo = promo_pot_before
        .map(|before| params.promo_potential_w * (own_effects.promo_potential - before))
        .unwrap_or(0.0);

    // 大駒の成り道の差分（`major_promo_path` の doc 参照）。自駒がどいて
    // 飛・角・香の成り道が開く手が正、自分で塞ぐ手が負。promo と同じく
    // 大半の候補は 0 なので gain のゼロ点は動かない
    let major_path = major_path_before
        .map(|before| params.major_promo_path_w * (own_effects.major_promo_path - before))
        .unwrap_or(0.0);

    // 持ち駒オプションの不足分（`hand_option_w` の doc 参照）。打つ手にだけ
    // 「その駒種の最良打ちポテンシャル − この打ちマスでの実現値」を引く。
    // 最良マスへの打ち（垂れ歩）は 0、ポテンシャルを捨てる打ちほど満額。
    // 移動の手は 0 なのでゼロ点は動かない
    let hand_option_pen = match (hand_option, mv) {
        (Some(ctx), &ShogiMove::Drop { role, to }) => ctx
            .best
            .get(&role)
            .map(|&h| {
                params.hand_option_w
                    * (h - piece_promo_potential(&ctx.occupied, to, role, me)).max(0.0)
            })
            .unwrap_or(0.0),
        _ => 0.0,
    };

    // V5（盤上駒の減価、やねうら王 Lv2 で +R50）: 「同じ駒なら持ち駒のほうが
    // 価値が高い」。この手で**盤上に増えた自駒の価値**にだけ比例して引く
    // （打ち＝打った駒、成り＝増えたぶん。盤上の合計は定数なので持たない
    // ＝ gain のゼロ点を動かさない。`board_material_added` の doc 参照）。
    // 紐（link）と同じく自駒だけで決まるので粒子不要・ノイズゼロ
    let board_discount = params.board_discount_w * own_effects.board_material_added;

    // V2（玉距離重み付き利き、やねうら王 Lv3 で +R200）。自玉側は完全既知で
    // ノイズゼロ、相手玉側は粒子の信念で薄まる。既存の coverage（全マス平等）が
    // SPSA で潰されたのは重み付けが無かったせいだ、という賭け
    let effect_value =
        params.effect_own_w * own_effects.effect_own + params.effect_opp_w * own_effects.effect_opp;

    // 詰めろ生成: この手の後、次の自分の手番で持ち駒打ちの一手詰めが成立するか
    // （mate.rs::drop_mate）。ついたて将棋では相手に脅威が見えないので、詰めろは
    // 実質「次で詰む」に近い（発端の対人局 2026-07-25: 58手目 N*6六 の詰めろに
    // bot は受けを選ばず 60手目 G*7八 で詰み）。既存の攻め項（check_bonus /
    // attack_w / blind_king_attack）はどれも「王手でも駒得でもない、詰み網を
    // 完成させる静かな手」を評価できない。
    //
    // **`expected` の外**に置くのは、詰めろが効く終盤ほど厳密粒子が枯れて
    // `legal == 0`（= expected が丸ごとゼロ）になるため。ただし gain の内側なので
    // combine_score の p_legal 割引は受ける（反則確実な手への攻め加点素通りを
    // 防ぐ dragon-check-drop の教訓）。
    // **王手中は両方とも無効**: 候補の序列は解消確率（CheckSolver）が支配すべき。
    // 攻め側は加点が回避プローブの反則を増やす実測があり（value_nn_w と同じゲート）、
    // 受け側は「合法な回避手だけを一律に減点する」形になるため、合法手全体が
    // 反則水位を割って反則が爆発する（removal_term の対称形で踏んだ罠と同型。
    // check.rs の doc コメント参照）
    //
    // 被詰めろ（`mate_risk_w`）は同じ判定の鏡像。ただし相手の**盤上の支え駒**は
    // 見えないので、真の詰みだけを数えると発端の局面（支えが不可視の 6六桂）を
    // 取りこぼす。「玉で取る以外に受けがない」= 支えが1枚あれば詰み
    // （MateThreat::IfSupported）も MATE_RISK_IF_SUPPORTED 倍で数える。
    // 詰みの成立条件は自玉の逃げ道・自駒・相手の持ち駒（いずれも既知）が
    // ほぼ決めるので、この形なら不可視情報にほとんど依存しない
    let (mate_threat, mate_risk) = if (params.mate_threat_w != 0.0 || params.mate_risk_w != 0.0)
        && !view.you_in_check
    {
        // 厳密粒子は正確なぶん退化度（confidence）で割り引く。taint 粒子は
        // 重み自体に 0.5^(taint-1) の減衰が入っているのでそのまま使う
        let conf = if particles.is_empty() { 1.0 } else { confidence };
        let mut threat = 0.0f64;
        let mut risk = 0.0f64;
        let mut tot = 0.0f64;
        for (pos, w) in mate_pool.iter().take(budget.mate_samples) {
            if !pos.is_legal(mv) {
                continue;
            }
            let mut next = (*pos).clone();
            next.play_unchecked(mv);
            tot += w;
            // 既に詰ましている手は粒子ループの +1000 側で評価済み（受けも不要）
            if next.in_check(opp) && !next.has_any_legal_move() {
                continue;
            }
            if params.mate_threat_w != 0.0 && crate::mate::drop_mate(&next, me).is_some() {
                threat += w;
            }
            if params.mate_risk_w != 0.0 {
                match crate::mate::drop_mate_threat(&next, opp) {
                    Some((_, crate::mate::MateThreat::Mate)) => risk += w,
                    Some((_, crate::mate::MateThreat::IfSupported)) => {
                        risk += w * MATE_RISK_IF_SUPPORTED
                    }
                    None => {}
                }
            }
        }
        if tot > 0.0 {
            (
                params.mate_threat_w * conf * (threat / tot),
                params.mate_risk_w * conf * (risk / tot),
            )
        } else {
            (0.0, 0.0)
        }
    } else {
        (0.0, 0.0)
    };

    let gain = expected + advance_bias + development + coverage + mate_threat - mate_risk
        - king_holes
        + link
        + drop_hit_evac
        + promo
        + major_path
        - hand_option_pen
        - board_discount
        + effect_value
        + blind_recapture
        + blind_belief_gain;
    // 打ちプローブの反則情報価値: 反則枝（占有粒子）の期待回収値 ×
    // p_occ(1−p_occ) の情報ゲート × 残り反則予算の2乗。形の根拠:
    // - p_occ を掛ける（実効 p_occ²·(1−p_occ)）: 線形だと占有質量 8% のマスへの
    //   探り（quest31-m026/m028 の P*2f、飛車 9.5 の期待値が僅差を逆転）まで
    //   浮く。人間の使い分けは「居そうなときだけ探る」（2二=65% は探る・
    //   2六=8% は探らない）
    // - (1−p_occ) を掛ける: 占有が**確定**したマスの再プローブは情報価値ゼロ
    //   （正解は既存の利きで取る手）。selfplay/実対局の foul_tried は受理後に
    //   クリアされるので、この因子が無いと「反則で確定 → 次手も同じマスへ
    //   打つ」の反則ループが最大ブーストになる
    // - 残り反則予算 (fouls_left/10)² : 勝敗は反則レース（アリーナの78%が
    //   反則負け決着）なので対価は残budget比例。人間のプローブも
    //   「序盤の反則が安いうちに」が定石（15手目の2二歩打 = 反則0の局面）。
    //   ゲート無しの初版はアリーナで反則/局 6.4→8.2・36.8%（−13.7pt）
    // 全粒子質量 n で正規化するので (1−p_legal) の重みを内包している
    let foul_probe = if n > 0.0 && probe_mass > 0.0 {
        let p_occ = probe_mass / n;
        let budget_frac = f64::from(10u32.saturating_sub(view.fouls.you)) / 10.0;
        // **再プローブのゲート**（`drop_probe_repeat_gate`、2026-08-03、
        // ユーザー指摘の61手目 4七歩打が発端）。`(1−p_occ)` はループ対策
        // としては効くが、**確信が高いほど価値が上がる**という人間の使い方を
        // 巻き添えで消す: quest31 の61手目は 4七の占有 p_occ≈0.89・5七金の
        // 支えつきで、人間は「反則なら占有が確定して金で取れる、空なら
        // 支え付きで争点を占拠できる」と読んで歩を打った。bot は
        // (1−p_occ)=0.11 でこの手を78位に沈め、代わりに**支えていた金自身**を
        // 4七へ動かす手（ユーザー判定=悪手）を5位に置いていた。
        //
        // ループ対策は占有確率でなく「**そのマスへ既に打って反則したか**」で
        // 判定するのが正しい（観測ログの MyFoul から正確に取れる。foul_tried と
        // 違い受理でクリアされない）。既に確定したマスは価値ゼロ、まだ試して
        // いないマスは p_occ² で単調増加させる（m026/m028 の「占有8%への
        // 安い探り」を抑える p_occ² は保つ）
        let repeat = if drop_probe_repeat_gate() {
            match *mv {
                ShogiMove::Drop { to, .. } => {
                    if drop_foul_squares[crate::belief_features::sq_index(to)] {
                        0.0
                    } else {
                        1.0
                    }
                }
                ShogiMove::Board { .. } => 1.0,
            }
        } else {
            1.0 - p_occ
        };
        params.drop_probe_w * (probe_val_sum / n) * p_occ * repeat * budget_frac * budget_frac
    } else {
        0.0
    };
    EvalOut {
        gain,
        risk_mean: if legal > 0.0 { risk_sum / legal } else { 0.0 },
        p_legal,
        foul_cost,
        checker_removal: 0.0,
        capture_bet_penalty,
        mate_threat,
        mate_risk,
        king_holes,
        value_nn: value_nn_term,
        capture_value: if legal > 0.0 {
            capture_value_sum / legal
        } else {
            0.0
        },
        link,
        promo,
        hand_option: hand_option_pen,
        board_discount,
        foul_probe,
    }
}

/// 2手読み: 候補手の後の相手応手の損失を方策加重の**期待値**で評価する。
/// （露見度で割引した駒損 − 取り返し補償、被王手/被詰みペナルティ）。
/// 静的リスク項（EvalOut::risk_mean）の置き換え先。値は「加点」方向（通常は負）。
///
/// 旧実装は応手を1手サンプルしていたため、低確率の大損失を引いたかどうかで
/// 候補順位が揺れた（モンテカルロノイズ）。応手の列挙と重みは既に計算している
/// ので、駒損が出る応手（自駒を取る手）は全て厳密に評価して重み平均し、
/// 静かな応手は駒損ゼロ・王手ペナルティのみを少数サンプルで近似する
#[allow(clippy::too_many_arguments)]
fn depth2_delta(
    view: &PlayerView,
    mv: &ShogiMove,
    particles: &[(&Position, f64)],
    known: &HashMap<Coord, f64>,
    my_captures: &[Coord],
    my_touched: &[Coord],
    my_fouls_this_turn: u32,
    params: &EvalParams,
    budget: SearchBudget,
    rng: &mut impl rand::Rng,
) -> f64 {
    let me = view.your_color;
    let to = match *mv {
        ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
    };
    // 被王手/被詰みの評価（clone+play が要るのでここに集約）
    let check_pen = |next2: &mut Position| -> f64 {
        if next2.in_check(me) {
            let mut p = params.depth2_check_pen;
            if next2.legal_moves().is_empty() {
                p += DEPTH2_MATE_PEN;
            }
            p
        } else {
            0.0
        }
    };
    let mut sum = 0.0;
    let mut n = 0.0;
    // 構想（3手目）の累積。相手の応手を1手挟んだ**後**の自分の駒得を見る。
    // 応手を挟まない楽観版（plan_bonus）は 200局で -6pt と明確に負だった
    let mut plan_sum = 0.0;
    let mut plan_n = 0.0;
    for (i, (pos, w)) in particles.iter().take(budget.depth2_particles).enumerate() {
        if !pos.is_legal(mv) {
            continue;
        }
        let mut next = (*pos).clone();
        let my_capture = next.play_unchecked(mv);
        let gives_check = next.in_check(me.other());
        n += w;
        // この候補手で駒を取った場合、捕獲通知でそのマスは相手に露見する。
        // 応手予測の既知地点に加えないと、最有力の応手である「即時の取り返し」に
        // PREDICT_RECAPTURE_BOOST が掛からず、捕獲手を過度に楽観視してしまう
        let extended;
        let known_for_reply: &[Coord] = if my_capture.is_some() {
            extended = [my_captures, &[to]].concat();
            &extended
        } else {
            my_captures
        };
        let replies = opp_reply_weights(&next, me, known_for_reply, my_touched, my_fouls_this_turn);
        let total_rw: f64 = replies.iter().map(|(_, rw)| rw).sum();
        if replies.is_empty() || total_rw <= 0.0 {
            continue; // 応手なし（詰み/ステイルメイト）は stage1 のボーナス側で評価済み
        }
        let mut exp_delta = 0.0;
        // 静かな応手（駒損なし）: 重みを溜めて王手ペナルティだけ後でサンプル近似
        let mut quiet: Vec<(ShogiMove, f64)> = vec![];
        let mut quiet_w = 0.0;
        for (reply, rw) in &replies {
            let reply_to = match *reply {
                ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
            };
            let lost = next
                .piece_at(reply_to)
                .filter(|p| p.color == me)
                .map(|p| exchange_value(p.role))
                .unwrap_or(0.0);
            if lost <= 0.0 {
                quiet_w += rw;
                quiet.push((*reply, *rw));
                continue;
            }
            let mut next2 = next.clone();
            next2.play_unchecked(reply);
            // 露見度スケール: 着手駒は stage1 の mover_w と同じ規則、
            // それ以外の駒は exposed_capture_risk と同じ knownness 重み。
            // 粒子上の応手はこちらの駒が全部見えてしまうので、実戦で相手が
            // その取りを狙える確率で割り引く（情報非対称の担保）
            let scale = if reply_to == to {
                let mut s = if my_capture.is_some() {
                    params.mover_w_captured
                } else {
                    params.mover_w_quiet
                };
                if gives_check {
                    s += params.mover_check_extra;
                }
                s
            } else {
                let knownness = known.get(&reply_to).copied().unwrap_or(0.0);
                params.exposed_base + params.exposed_known * knownness
            };
            // 取り返し補償: 応手の駒に自分の利きが残っていれば取り返せる
            let comp = if !next2.in_check(me) && next2.is_attacked(reply_to, me) {
                params.depth2_recap_discount
                    * next2
                        .piece_at(reply_to)
                        .map(|p| exchange_value(p.role))
                        .unwrap_or(0.0)
            } else {
                0.0
            };
            let d = -scale * (lost - comp).max(0.0) - check_pen(&mut next2);
            exp_delta += rw * d;
        }
        if quiet_w > 0.0 {
            // 静かな応手の被王手率は低頻度なので2サンプルで近似する
            let samples = quiet.len().min(2);
            let mut pen = 0.0;
            for _ in 0..samples {
                let mut t = rng.random_range(0.0..quiet_w);
                let mut chosen = &quiet[quiet.len() - 1].0;
                for (r, rw) in &quiet {
                    t -= rw;
                    if t <= 0.0 {
                        chosen = r;
                        break;
                    }
                }
                let mut next2 = next.clone();
                next2.play_unchecked(chosen);
                pen += check_pen(&mut next2);
            }
            exp_delta -= quiet_w * pen / samples as f64;
        }
        sum += w * (exp_delta / total_rw);

        // --- 構想（自分の手 → 相手の応手 → 自分の手）---
        //
        // やねうら王のような完全情報エンジンは、この「自分→相手→自分」を
        // 深さぶん展開するのが探索そのもの。ついたて側で深さが買えないのは
        // **1ノードの単価**が違う（粒子集合の上で評価するので NNUE の差分計算に
        // 比べて3桁重い）ことと、期待値を取るまで打ち切れないので αβ カットが
        // 効かないことによる。そこでここでは
        // 「応手を1つサンプル → 自分の最善の駒得だけ見る」に絞る。
        // 全評価（evaluate）ではなく駒得だけなので、legal_moves 1回ぶんで済む。
        //
        // 応手を挟まない版（旧 plan_bonus）は 200局で 45.5%（対照 51.5%）と
        // 明確に負だった: 「支えを作れば次に出られる」を数える一方で、相手が
        // その間に支えを崩したり、より速い攻めを通したりするのを見ていなかった
        if params.plan_w != 0.0 && i < budget.plan_particles {
            // 応手は**複数サンプルして平均**する。1本だけだと分散が大きく、
            // 「たまたま楽な応手を引いた候補」が浮く（旧版の負けの一因）
            let mut acc = 0.0f64;
            for _ in 0..PLAN_REPLY_SAMPLES {
                let mut t = rng.random_range(0.0..total_rw);
                let mut chosen = &replies[replies.len() - 1].0;
                for (r, rw) in &replies {
                    t -= rw;
                    if t <= 0.0 {
                        chosen = r;
                        break;
                    }
                }
                let mut next2 = next.clone();
                next2.play_unchecked(chosen);
                // ここで手番は自分。相手の応手を織り込んだうえでの自分の最善手。
                // 3手目は**駒得と王手**だけ見る（全評価を回すと粒子×応手×候補で
                // 桁が違う。やねうら王が深く読めるのは1ノードの単価が3桁安い
                // からで、こちらは同じ深さを同じ値段では買えない）
                let mut best = 0.0f64;
                for m2 in next2.legal_moves() {
                    let to2 = match m2 {
                        ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
                    };
                    let gain3 = match next2.piece_at(to2).filter(|p| p.color != me) {
                        Some(target) => {
                            // 取り返される見込みは「着地マスに相手の利きが
                            // 残っているか」で近似する（候補ごとの clone+play は重い）
                            let comp = if next2.is_attacked(to2, me.other()) {
                                match m2 {
                                    ShogiMove::Board { from, .. } => next2
                                        .piece_at(from)
                                        .map(|p| exchange_value(p.role))
                                        .unwrap_or(0.0),
                                    ShogiMove::Drop { role, .. } => exchange_value(role),
                                }
                            } else {
                                0.0
                            };
                            exchange_value(target.role) - comp
                        }
                        None => 0.0,
                    };
                    if gain3 > best {
                        best = gain3;
                    }
                }
                acc += best;
            }
            plan_sum += w * acc / f64::from(PLAN_REPLY_SAMPLES);
            plan_n += w;
        }
    }
    let base = if n > 0.0 { sum / n } else { 0.0 };
    let plan = if plan_n > 0.0 {
        params.plan_w * (plan_sum / plan_n)
    } else {
        0.0
    };
    base + plan
}

/// 構想（3手目）で相手の応手をサンプルする本数。1本だけだと分散が大きく、
/// 「たまたま楽な応手を引いた候補」が浮いてしまう
const PLAN_REPLY_SAMPLES: u32 = 3;

/// この手の後も初期位置に残っている自分の大駒（飛・角）の枚数
fn big_home_after(view: &PlayerView, mv: &ShogiMove) -> f64 {
    let (rook_home, bishop_home) = match view.your_color {
        Color::Sente => (Coord { file: 2, rank: 8 }, Coord { file: 8, rank: 8 }),
        Color::Gote => (Coord { file: 8, rank: 2 }, Coord { file: 2, rank: 2 }),
    };
    let from = match *mv {
        ShogiMove::Board { from, .. } => Some(from),
        ShogiMove::Drop { .. } => None,
    };
    let mut n = 0.0;
    for piece in &view.your_pieces {
        let Some(sq) = parse_usi_square(&piece.square) else {
            continue;
        };
        let home = (piece.role == Role::Rook && sq == rook_home)
            || (piece.role == Role::Bishop && sq == bishop_home);
        if home && from != Some(sq) {
            n += 1.0;
        }
    }
    n
}

/// 自分が当たりを付けている敵駒の最大価値（露出リスクの鏡像）。
/// 紐つき（相手が守っている）なら取ったときに取り返されるぶん割り引く。
/// 玉への当たりは王手であり合法性・王手ボーナス側で扱うので除く。
///
/// **枚数版**（`threat_by_count`、`TSUITATE_THREAT_BY_COUNT`）: 守りの判定を
/// 「1枚でも守られていれば割り引く」から「守り枚数 ≥ 攻め枚数なら割り引く」へ
/// 変える（`recapture_risk` の V1 の鏡像）。ユーザー指摘 2026-08-03:
/// 「いると確信している駒を複数枚の利きで取りに行くことは実戦でもやること」—
/// 従来形は当たりの有無が二値なので、**既に当たっている駒へ利きを足す手**
/// （取りの準備）に一切の価値が付かず、単独駒での投機的な捕獲にしか勝ち目がない
/// （quest31-m021: 4一の金（と信じている駒）へ3二とで利きを足す手が、
/// 単騎の 4一と に 2.2 点負ける）
fn threat_value(pos: &Position, me: Color) -> f64 {
    let opp = me.other();
    let by_count = threat_by_count();
    let mut best = 0.0f64;
    for (sq, piece) in pos.pieces() {
        if piece.color != opp || piece.role == Role::King {
            continue;
        }
        // 安いゲートを先に置く（attack_count は当たっている駒にだけ払う）
        if !pos.is_attacked(sq, me) {
            continue;
        }
        let defended = if by_count {
            pos.attack_count(sq, opp) >= pos.attack_count(sq, me)
        } else {
            pos.is_attacked(sq, opp)
        };
        let gain = exchange_value(piece.role) * if defended { 0.45 } else { 1.0 };
        best = best.max(gain);
    }
    best
}

/// **争点への利き足し**か（2手読みの予約枠、`depth2_focal_k`）。
///
/// 着手した駒が「**自分が既に利かせていて、自駒が乗っていないマス**」へ
/// 新たに利きを足すなら真。＝これから取り合いになるマスへ味方を足す手
/// （ユーザーの言う「焦点への利き集中」。quest31-m040 の 3八銀打が典型で、
/// 4六歩が既に利かせている4七へ銀の利きを足す）。
///
/// **静的評価では必ず負になる類型**なのが要点: 持ち駒を晒して、その手自体では
/// 何も得ない。価値は「4七歩成 → 同X → 同銀」の3手先にしかないので、
/// 静的スコアの上位N本だけを2手読みに回す従来の足切りでは**読まれることすら
/// ない**（実測: 3八銀打は 67〜98位で depth2=false）。良い利き足しと悪い利き足し
/// （5八銀打 = 玉の隣で即取られ）を静的に区別しようとすると、マージンの薄い
/// 他の手を巻き込んで壊れる（king_adj_entry_w / hand_option_w の失敗）ので、
/// **区別は読みに任せて、読む機会だけを与える**。
///
/// 自駒の配置だけで決まるので粒子不要・ノイズゼロ
fn adds_focal_attacker(view: &PlayerView, mv: &ShogiMove, own_attack_before: &[u8; 81]) -> bool {
    let (to, role) = match *mv {
        ShogiMove::Board { from, to, promote } => {
            let from_usi = make_usi_square(from);
            let Some(p) = view.your_pieces.iter().find(|p| p.square == from_usi) else {
                return false;
            };
            let role = if promote {
                promote_role(p.role).unwrap_or(p.role)
            } else {
                p.role
            };
            (to, role)
        }
        ShogiMove::Drop { role, to } => (to, role),
    };
    // 着手後の盤で、その駒がどこへ利くか
    let mut pieces: Vec<VisiblePiece> = view.your_pieces.clone();
    if let ShogiMove::Board { from, .. } = *mv {
        let from_usi = make_usi_square(from);
        pieces.retain(|p| p.square != from_usi);
    }
    let moved = VisiblePiece {
        square: make_usi_square(to),
        role,
    };
    let own_occupied: HashSet<String> = pieces.iter().map(|p| p.square.clone()).collect();
    pieces.push(moved.clone());
    defend_targets(&pieces, &moved, view.your_color)
        .into_iter()
        .any(|s| {
            s != to
                && own_attack_before[crate::belief_features::sq_index(s)] >= 1
                && !own_occupied.contains(&make_usi_square(s))
        })
}

/// マス `from` にいる自駒が `to` へ利いているか（`blind_attack_survive_w` の
/// 守り枚数から着手駒自身を除くため）
fn own_defends_from(view: &PlayerView, from: Coord, to: Coord) -> bool {
    let from_usi = make_usi_square(from);
    view.your_pieces
        .iter()
        .find(|p| p.square == from_usi)
        .is_some_and(|p| defend_targets(&view.your_pieces, p, view.your_color).contains(&to))
}

/// 自駒が今どのマスへ何枚利かせているか（`adds_focal_attacker` の前提）
fn own_attack_counts(view: &PlayerView) -> [u8; 81] {
    let mut n = [0u8; 81];
    for p in &view.your_pieces {
        for s in defend_targets(&view.your_pieces, p, view.your_color) {
            let i = crate::belief_features::sq_index(s);
            n[i] = n[i].saturating_add(1);
        }
    }
    n
}

/// 打ちプローブの再プローブ判定を「既に打って反則したマスか」で行うか
/// （`TSUITATE_DROP_PROBE_REPEAT_GATE=1`、既定 off = 従来の `(1−p_occ)`）。
/// `foul_probe` の doc 参照
fn drop_probe_repeat_gate() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("TSUITATE_DROP_PROBE_REPEAT_GATE").is_ok_and(|v| v == "1")
    })
}

/// 2手読みへ予約する「争点への利き足し」の本数（`TSUITATE_DEPTH2_FOCAL_K`、
/// 既定 0 = 従来挙動）。`adds_focal_attacker` の doc 参照
fn depth2_focal_k() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("TSUITATE_DEPTH2_FOCAL_K")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
    })
}

/// マス `sq` の「`by` 側から見た正面手前」= `by` の歩がそこから `sq` へ進めるマス。
/// `exposed_pawn_head_w`（鉢合わせ）の判定に使う
fn pawn_front_of(sq: Coord, by: Color) -> Option<Coord> {
    let rank = match by {
        Color::Sente => sq.rank + 1,
        Color::Gote => sq.rank - 1,
    };
    (1..=9).contains(&rank).then_some(Coord {
        file: sq.file,
        rank,
    })
}

/// `threat_value` の守り判定を枚数で行うか（既定 off = 従来の二値）。
/// `threat_value` の doc 参照
fn threat_by_count() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("TSUITATE_THREAT_BY_COUNT").is_ok_and(|v| v == "1"))
}

/// 着手駒（マス to にいる自駒）が次の相手番で取られるリスク。
/// 紐つきなら取り返せるぶん割り引く（相手のどの駒で取るかは不明なので近似）
fn recapture_risk(pos: &Position, me: Color, to: Coord, defended_discount: f64) -> f64 {
    let opp = me.other();
    let Some(piece) = pos.piece_at(to).filter(|p| p.color == me) else {
        return 0.0;
    };
    if piece.role == Role::King {
        return 0.0;
    }
    let attackers = pos.attack_count(to, opp);
    if attackers == 0 {
        return 0.0;
    }
    // V1: 「紐が1本でもあれば割り引く」から「守り枚数が攻め枚数以上なら割り引く」へ。
    // 2枚で狙われて1枚で守っている駒は、取り返しても駒損が残るので割り引かない
    let defenders = pos.attack_count(to, me);
    let defended = if v1_defended_by_count() {
        defenders >= attackers
    } else {
        defenders > 0
    };
    exchange_value(piece.role) * if defended { defended_discount } else { 1.0 }
}

/// 次の相手番で失いうる駒の概算: 相手の利きが当たっている自駒の最大重み付き価値。
/// 自分の利きも当たっている（紐つき）なら取り返せるぶん割り引く。
/// 相手がその駒の位置を知っているほど（knownness_map）実際に取られやすいので
/// 重みを引き上げる。位置が漏れていない駒は従来通り薄く見積もる。
/// exclude（着手駒のマス）は recapture_risk 側で別の重みで数えるので除外する。
/// 合法手の完全列挙（ピン考慮など）はコストに見合わないので利きベースの近似。
///
/// **複数枚版**（`exposed_multi_w`、既定0 = 従来の max）: 上位3件を
/// `t0 + w·t1 + w²·t2` で数える。相手は1手に1枚しか取れないので最大値を主項に
/// 置く形は保つが、max だけだと**2枚目以降の当たりが完全に消える**。
/// ユーザー指摘 2026-08-03（quest31-m026 の 4七歩成）: 当たっている自駒が
/// 1一香（紐なし・龍に当たり・初期位置なので相手の既知度も高い）と
/// 4六歩（2枚に当たられて紐なし）の2枚あるとき、max は必ず香を取るので
/// **「取られやすい歩を逃がしつつ取って成る」手の緊急性が評価上ゼロになる**。
/// 逃げの価値は「動かせばその駒が集合から抜ける」ことで自動的に差分になる
fn exposed_capture_risk(
    pos: &Position,
    me: Color,
    exclude: Option<Coord>,
    known: &HashMap<Coord, f64>,
    params: &EvalParams,
) -> f64 {
    let opp = me.other();
    // 上位3件を降順に保つ（Vec を作らずホットパスの割り当てを避ける）
    let mut top = [0.0f64; 3];
    for (sq, piece) in pos.pieces() {
        if piece.color != me || piece.role == Role::King {
            continue; // 玉が当たっているなら王手なので合法性の側で処理される
        }
        if exclude == Some(sq) {
            continue;
        }
        let attackers = pos.attack_count(sq, opp);
        if attackers == 0 {
            continue;
        }
        // V1: recapture_risk と同じく守り枚数 vs 攻め枚数で判定する
        let defenders = pos.attack_count(sq, me);
        let defended = if v1_defended_by_count() {
            defenders >= attackers
        } else {
            defenders > 0
        };
        let knownness = known.get(&sq).copied().unwrap_or(0.0);
        // **鉢合わせ**（`exposed_pawn_head_w`）: 敵歩の正面に立つ駒は、相手が
        // 歩を突くという普通の手だけで取られる = 位置を知られている必要がない。
        // 実測（bin/collision_probe、相手が位置を知らない駒だけで比較）:
        // 対人83局 8.01% vs 5.33%、アリーナ50局 5.36% vs 3.58% = **どちらも1.50倍**
        let head = params.exposed_pawn_head_w != 0.0
            && pawn_front_of(sq, opp).is_some_and(|c| {
                pos.piece_at(c)
                    .is_some_and(|p| p.color == opp && p.role == Role::Pawn)
            });
        let weight = (params.exposed_base + params.exposed_known * knownness)
            * if head {
                1.0 + params.exposed_pawn_head_w
            } else {
                1.0
            };
        let loss = exchange_value(piece.role)
            * if defended {
                params.exposed_defended
            } else {
                1.0
            }
            * weight;
        if loss > top[0] {
            top[2] = top[1];
            top[1] = top[0];
            top[0] = loss;
        } else if loss > top[1] {
            top[2] = top[1];
            top[1] = loss;
        } else if loss > top[2] {
            top[2] = loss;
        }
    }
    let w = params.exposed_multi_w;
    top[0] + w * top[1] + w * w * top[2]
}

/// 相手の持ち駒による「王手打ちの受け入れ面積」。
/// 相手の持ち駒はこの粒子上で正確に分かる（取られた自駒 − 打たれた駒）。
/// - 飛: 玉からの縦横の空き直線の長さ（その各マスが王手打ちの入口）
/// - 角: 斜めの空き直線の長さ
/// - 香: 相手の香が王手できる側の1直線
/// - 金/銀: 玉の隣接空きマス（打てば即王手）
/// - 歩: 玉頭の1マス
/// 持ち駒が空ならゼロ = 居玉そのものは咎めない
pub(crate) fn drop_check_danger(pos: &Position, me: Color) -> f64 {
    let Some(king) = pos.king_square(me) else {
        return 0.0;
    };
    let opp = me.other();
    let on_board = |c: &Coord| (1..=9).contains(&c.file) && (1..=9).contains(&c.rank);
    let ray_len = |df: i8, dr: i8| -> f64 {
        let mut n = 0;
        let mut c = Coord {
            file: king.file + df,
            rank: king.rank + dr,
        };
        while on_board(&c) && pos.piece_at(c).is_none() {
            n += 1;
            c = Coord {
                file: c.file + df,
                rank: c.rank + dr,
            };
        }
        n as f64
    };

    let mut danger = 0.0;
    if pos.hand_count(opp, Role::Rook) > 0 {
        danger += ray_len(1, 0) + ray_len(-1, 0) + ray_len(0, 1) + ray_len(0, -1);
    }
    if pos.hand_count(opp, Role::Bishop) > 0 {
        danger += ray_len(1, 1) + ray_len(1, -1) + ray_len(-1, 1) + ray_len(-1, -1);
    }
    // 相手の香・歩は「相手から見て前へ」利くので、自玉側から見ると
    // 自分の陣の奥方向の直線・玉頭が入口になる
    let toward = if me == Color::Sente { -1 } else { 1 };
    if pos.hand_count(opp, Role::Lance) > 0 {
        danger += ray_len(0, toward);
    }
    if pos.hand_count(opp, Role::Pawn) > 0 {
        let head = Coord {
            file: king.file,
            rank: king.rank + toward,
        };
        if on_board(&head) && pos.piece_at(head).is_none() {
            danger += 1.0;
        }
    }
    let generals = pos.hand_count(opp, Role::Gold) > 0 || pos.hand_count(opp, Role::Silver) > 0;
    if generals {
        let mut air = 0.0;
        for df in -1..=1i8 {
            for dr in -1..=1i8 {
                if df == 0 && dr == 0 {
                    continue;
                }
                let c = Coord {
                    file: king.file + df,
                    rank: king.rank + dr,
                };
                if on_board(&c) && pos.piece_at(c).is_none() {
                    air += 0.5;
                }
            }
        }
        danger += air;
    }
    danger
}

/// 1マスに m 枚の利きがあるときの価値（1枚を 1.0 とした逓減）。
///
/// docs/yaneuraou-lessons.md の V1。やねうら王 Lv4 の実測式
/// `6365 - 0.8525^(m-1) × 5341` を 1枚 = 1024 で正規化したもので、
/// 2枚目 1.77・3枚目 2.42・…と**明確に逓減するが飽和はしない**。
/// 「2枚で狙われている」と「1枚で狙われている」を区別できるようにするのが目的で、
/// 枚数に線形だと大駒1枚の睨みと歩3枚の押さえが同じ重みになってしまう
fn effect_multiplicity_value(m: u8) -> f64 {
    if m == 0 {
        return 0.0;
    }
    // 逓減の実測値（m=1..=8）。9枚以上は 8枚と同じに丸める（盤上で起きない）
    const TABLE: [f64; 8] = [1.000, 1.769, 2.419, 2.973, 3.446, 3.849, 4.192, 4.485];
    TABLE[(m as usize - 1).min(TABLE.len() - 1)]
}

/// owner 玉の周囲8マス（と玉のマス）に当たっている by 側の利きの重み付き総和。
///
/// V1 以前は「攻撃されているマスの数」（各マス0/1）だった。同じ1マスに
/// 2枚利いている形（＝実際に破られる形）と1枚だけ睨んでいる形が区別できず、
/// 玉頭に駒を足す攻めにも、支えを1枚増やす受けにも勾配が立たなかった
pub(crate) fn king_zone_pressure(pos: &Position, owner: Color, by: Color) -> f64 {
    let Some(king) = pos.king_square(owner) else {
        return 0.0;
    };
    let mut pressure = 0.0;
    for df in -1..=1i8 {
        for dr in -1..=1i8 {
            let c = crate::board::Coord {
                file: king.file + df,
                rank: king.rank + dr,
            };
            if (1..=9).contains(&c.file) && (1..=9).contains(&c.rank) {
                pressure += if v1_pressure_multiplicity() {
                    effect_multiplicity_value(pos.attack_count(c, by))
                } else {
                    pos.is_attacked(c, by) as u8 as f64
                };
            }
        }
    }
    pressure
}

/// owner 玉の隣接8マスのうち「逃げ先になり得る」（owner 自身の駒に塞がれて
/// いない）のに by の利きが当たっていないマス数 U を数え、1/(1+U) を返す。
/// U=0（全逃げ先を被覆）で 1.0、U=8 で 0.11 の凸形。king_zone_pressure が
/// 被覆マス数に線形なのと違い、「最後の逃げ道」ほど1マスの価値が跳ねる
/// （escape_cover_w の doc 参照）。by の駒がいるマスは玉に取られ得るので、
/// 支え（他の by 駒の利き）が無ければ未被覆と数える
pub(crate) fn escape_cover_value(pos: &Position, owner: Color, by: Color) -> f64 {
    let Some(king) = pos.king_square(owner) else {
        return 0.0;
    };
    let mut uncovered = 0u32;
    for df in -1..=1i8 {
        for dr in -1..=1i8 {
            if df == 0 && dr == 0 {
                continue;
            }
            let c = crate::board::Coord {
                file: king.file + df,
                rank: king.rank + dr,
            };
            if !(1..=9).contains(&c.file) || !(1..=9).contains(&c.rank) {
                continue;
            }
            if pos.piece_at(c).is_some_and(|p| p.color == owner) {
                continue;
            }
            if !pos.is_attacked(c, by) {
                uncovered += 1;
            }
        }
    }
    1.0 / (1.0 + f64::from(uncovered))
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::protocol::{ClockState, FoulCounts, GameStatus, VisiblePiece};

    pub(crate) fn minimal_view(pieces: Vec<VisiblePiece>, hand: HashMap<Role, u32>) -> PlayerView {
        PlayerView {
            game_id: "g".into(),
            your_color: Color::Sente,
            your_pieces: pieces,
            your_hand: hand,
            turn: Color::Sente,
            move_number: 1,
            clocks: ClockState {
                sente_ms: 300_000,
                gote_ms: 300_000,
                running: Some(Color::Sente),
                server_time: 0,
            },
            fouls: FoulCounts {
                you: 0,
                opponent: 0,
            },
            you_in_check: false,
            opponent_in_check: false,
            status: GameStatus::Playing,
        }
    }

    #[test]
    fn endgame_push_ramps_with_moves_and_lead() {
        // 序盤は掛けない
        assert_eq!(endgame_push(1, 10.0), 0.0);
        assert_eq!(endgame_push(59, 10.0), 0.0);
        // 終盤リードありで強く掛かる
        assert!(endgame_push(160, ANTI_DRAW_LEAD_UNIT) > 1.0);
        // 互角でも弱く掛けて膠着を破りにいく
        let even = endgame_push(160, 0.0);
        assert!(even > 0.0 && even < 0.5, "even={even}");
        // 負けているときは掛けない（引き分けは0.5勝の価値）
        assert_eq!(endgame_push(160, -10.0), 0.0);
        // 手数で単調増加
        assert!(endgame_push(100, 8.0) < endgame_push(160, 8.0));
    }

    #[test]
    fn material_lead_is_relative_and_symmetric() {
        let initial_pieces: Vec<VisiblePiece> = Position::initial()
            .pieces()
            .filter(|(_, p)| p.color == Color::Sente)
            .map(|(sq, p)| VisiblePiece {
                square: crate::board::make_usi_square(sq),
                role: p.role,
            })
            .collect();
        // 歩を1枚取った（持ち駒+1、盤上そのまま）→ 相対リード+2
        let mut hand = HashMap::new();
        hand.insert(Role::Pawn, 1);
        let view = minimal_view(initial_pieces.clone(), hand);
        assert!((material_lead(&view) - 2.0).abs() < 1e-9);
        // 飛車を1枚失った → 相対リードは飛車価値の2倍のマイナス
        // （相手の持ち駒に飛車が入るぶんも含む）
        let without_rook: Vec<VisiblePiece> = initial_pieces
            .into_iter()
            .filter(|p| p.role != Role::Rook)
            .collect();
        let view = minimal_view(without_rook, HashMap::new());
        let expected = -2.0 * piece_value(Role::Rook);
        assert!((material_lead(&view) - expected).abs() < 1e-9);
    }

    #[test]
    fn drop_hit_exposure_counts_open_headed_majors_in_enemy_camp() {
        let vp = |file, rank, role| VisiblePiece {
            square: crate::board::make_usi_square(Coord { file, rank }),
            role,
        };
        // 先手の竜が2段目（敵陣）、頭=1段目が空き → 露出は竜の交換価値
        let pieces = vec![vp(5, 2, Role::Dragon), vp(5, 9, Role::King)];
        assert!(
            (drop_hit_exposure(&pieces, Color::Sente) - exchange_value(Role::Dragon)).abs() < 1e-9
        );
        // 頭を自駒が塞ぐと露出ゼロ（相手はそこへ歩を打てない）
        let pieces = vec![vp(5, 2, Role::Dragon), vp(5, 1, Role::Gold), vp(5, 9, Role::King)];
        assert_eq!(drop_hit_exposure(&pieces, Color::Sente), 0.0);
        // 最奥段は頭が盤外 → 露出ゼロ
        let pieces = vec![vp(5, 1, Role::Dragon)];
        assert_eq!(drop_hit_exposure(&pieces, Color::Sente), 0.0);
        // 敵陣外も数える（2026-08-03 の拡張。相手が歩を持てば自陣の飛車の頭にも
        // 打たれるので、露出は段に依らない）。旧挙動は TSUITATE_DROP_HIT_ALL_RANKS=0
        let pieces = vec![vp(5, 4, Role::Dragon)];
        assert!(
            (drop_hit_exposure(&pieces, Color::Sente) - exchange_value(Role::Dragon)).abs() < 1e-9
        );
        // 自陣の飛車（8段目）も対象: 頭の7段目が空いていれば露出する
        let pieces = vec![vp(2, 8, Role::Rook)];
        assert!(
            (drop_hit_exposure(&pieces, Color::Sente) - exchange_value(Role::Rook)).abs() < 1e-9
        );
        // 自陣の飛車でも頭が自駒（歩）で塞がっていれば露出ゼロ
        let pieces = vec![vp(2, 8, Role::Rook), vp(2, 7, Role::Pawn)];
        assert_eq!(drop_hit_exposure(&pieces, Color::Sente), 0.0);
        // 大駒以外は数えない
        let pieces = vec![vp(5, 2, Role::Gold)];
        assert_eq!(drop_hit_exposure(&pieces, Color::Sente), 0.0);
        // 後手: 8七の竜の頭は8八（dragon-evac の実局面と同じ向き）
        let pieces = vec![vp(8, 7, Role::Dragon)];
        assert!(
            (drop_hit_exposure(&pieces, Color::Gote) - exchange_value(Role::Dragon)).abs() < 1e-9
        );
        // 飛角馬も対象（角 @敵陣・頭空き）
        let pieces = vec![vp(2, 3, Role::Bishop)];
        assert!(
            (drop_hit_exposure(&pieces, Color::Sente) - exchange_value(Role::Bishop)).abs() < 1e-9
        );
    }

    #[test]
    fn promo_potential_scales_with_distance_and_effect_gain() {
        let vp = |file, rank, role| VisiblePiece {
            square: crate::board::make_usi_square(Coord { file, rank }),
            role,
        };
        // 先手の歩: 4段目（成りマスまで1手）は 7段目（4手）より大きい
        let near = promo_potential(&[vp(5, 4, Role::Pawn)], Color::Sente);
        let far = promo_potential(&[vp(5, 7, Role::Pawn)], Color::Sente);
        assert!(near > far, "near={near} far={far}");
        assert!(far > 0.0);
        // 歩→と金は Δ利き5（中央）× 0.5^1
        assert!((near - 5.0 * 0.5).abs() < 1e-9, "near={near}");
        // 前に自分の歩がいる歩は成れない（道を塞がれた駒はポテンシャル 0）
        let blocked =
            promo_potential(&[vp(5, 4, Role::Pawn), vp(5, 3, Role::Pawn)], Color::Sente);
        let solo_5c = promo_potential(&[vp(5, 3, Role::Pawn)], Color::Sente);
        assert!((blocked - solo_5c).abs() < 1e-9, "5四の歩の寄与が 0 になるはず");
        // 後手向き: 6段目の歩（成りマス7段目まで1手）も同じ値
        let gote = promo_potential(&[vp(5, 6, Role::Pawn)], Color::Gote);
        assert!((gote - 5.0 * 0.5).abs() < 1e-9, "gote={gote}");
    }

    #[test]
    fn promo_potential_zero_for_lance_behind_own_tokin() {
        // watch-…223458 の悪手 L*1b の形: と金1一の裏に打った香は永久に動けない
        let vp = |file, rank, role| VisiblePiece {
            square: crate::board::make_usi_square(Coord { file, rank }),
            role,
        };
        let with_lance = promo_potential(
            &[vp(1, 1, Role::Tokin), vp(1, 2, Role::Lance)],
            Color::Sente,
        );
        let without = promo_potential(&[vp(1, 1, Role::Tokin)], Color::Sente);
        assert!((with_lance - without).abs() < 1e-9, "塞がれた香の寄与は 0");
    }

    #[test]
    fn hand_option_shortfall_zero_at_best_square_full_when_blocked() {
        let vp = |file, rank, role| VisiblePiece {
            square: crate::board::make_usi_square(Coord { file, rank }),
            role,
        };
        // 歩1枚持ち: 最良打ちマス（成りマスまで1手）での不足分は 0、
        // 自陣深くの打ちは満額に近い
        let mut hand = HashMap::new();
        hand.insert(Role::Pawn, 1);
        let view = minimal_view(vec![vp(5, 9, Role::King)], hand);
        let ctx = hand_option_context(&view);
        let h = ctx.best[&Role::Pawn];
        assert!((h - 5.0 * 0.5).abs() < 1e-9, "h={h}");
        let near = piece_promo_potential(
            &ctx.occupied,
            Coord { file: 5, rank: 3 },
            Role::Pawn,
            Color::Sente,
        );
        let deep = piece_promo_potential(
            &ctx.occupied,
            Coord { file: 5, rank: 8 },
            Role::Pawn,
            Color::Sente,
        );
        assert!((h - near).abs() < 1e-9, "敵陣近くの打ちは不足分 0");
        assert!(h - deep > 1.0, "自陣深くの打ちは不足分が大きい: h={h} deep={deep}");
    }

    #[test]
    fn hand_option_lance_behind_tokin_forfeits_everything() {
        // lance-for-pawn の悪手 L*1b の形: と金1一の裏への香打ちは実現値 0 =
        // 最良打ちポテンシャルを丸ごと捨てる
        let vp = |file, rank, role| VisiblePiece {
            square: crate::board::make_usi_square(Coord { file, rank }),
            role,
        };
        let mut hand = HashMap::new();
        hand.insert(Role::Lance, 1);
        let view = minimal_view(
            vec![vp(1, 1, Role::Tokin), vp(5, 9, Role::King)],
            hand,
        );
        let ctx = hand_option_context(&view);
        let h = ctx.best[&Role::Lance];
        assert!(h > 0.0);
        let blocked = piece_promo_potential(
            &ctx.occupied,
            Coord { file: 1, rank: 2 },
            Role::Lance,
            Color::Sente,
        );
        assert_eq!(blocked, 0.0, "と金の裏の香は成りへの道が無い");
    }

    #[test]
    fn hand_option_skips_unpromotable_and_nifu_blocked() {
        let vp = |file, rank, role| VisiblePiece {
            square: crate::board::make_usi_square(Coord { file, rank }),
            role,
        };
        // 金は成れないので対象外（打ちに減点が掛からない）
        let mut hand = HashMap::new();
        hand.insert(Role::Gold, 1);
        let view = minimal_view(vec![vp(5, 9, Role::King)], hand);
        assert!(!hand_option_context(&view).best.contains_key(&Role::Gold));
        // 全筋に自歩がいれば歩は打てるマスが無い = h 無し
        let mut pieces: Vec<VisiblePiece> = (1..=9).map(|f| vp(f, 7, Role::Pawn)).collect();
        pieces.push(vp(5, 9, Role::King));
        let mut hand = HashMap::new();
        hand.insert(Role::Pawn, 1);
        let view = minimal_view(pieces, hand);
        assert!(!hand_option_context(&view).best.contains_key(&Role::Pawn));
    }

    #[test]
    fn promo_potential_promotion_move_realizes_gain() {
        // 成る手は「ポテンシャル（減衰つき）→ 実現値（満額）」で差分が正になる
        let vp = |file, rank, role| VisiblePiece {
            square: crate::board::make_usi_square(Coord { file, rank }),
            role,
        };
        let before = promo_potential(&[vp(9, 3, Role::Lance)], Color::Sente);
        let after = promo_potential(&[vp(9, 3, Role::Promotedlance)], Color::Sente);
        assert!(before > 0.0);
        assert!(after > before, "after={after} before={before}");
    }

    #[test]
    fn own_effects_promo_potential_diff_favors_advance_and_deep_drop() {
        let vp = |file, rank, role| VisiblePiece {
            square: crate::board::make_usi_square(Coord { file, rank }),
            role,
        };
        let mut hand = HashMap::new();
        hand.insert(Role::Pawn, 1);
        let view = minimal_view(vec![vp(5, 5, Role::Pawn), vp(5, 9, Role::King)], hand);
        let params = EvalParams {
            promo_potential_w: 1.0,
            ..EvalParams::default()
        };
        // 歩を進める手 > 玉を動かす手（歩のポテンシャルが 0.5^2 → 0.5^1 に上がる）
        let advance = ShogiMove::Board {
            from: Coord { file: 5, rank: 5 },
            to: Coord { file: 5, rank: 4 },
            promote: false,
        };
        let idle = ShogiMove::Board {
            from: Coord { file: 5, rank: 9 },
            to: Coord { file: 5, rank: 8 },
            promote: false,
        };
        let adv_pot = own_effects_after(&view, &advance, None, &params).promo_potential;
        let idle_pot = own_effects_after(&view, &idle, None, &params).promo_potential;
        assert!(adv_pot > idle_pot, "adv={adv_pot} idle={idle_pot}");
        // 垂れ歩（4段目打ち）> 自陣打ち（8段目）
        let deep = ShogiMove::Drop {
            role: Role::Pawn,
            to: Coord { file: 7, rank: 4 },
        };
        let shallow = ShogiMove::Drop {
            role: Role::Pawn,
            to: Coord { file: 7, rank: 8 },
        };
        let deep_pot = own_effects_after(&view, &deep, None, &params).promo_potential;
        let shallow_pot = own_effects_after(&view, &shallow, None, &params).promo_potential;
        assert!(deep_pot > shallow_pot, "deep={deep_pot} shallow={shallow_pot}");
        // w=0 なら計算ごとスキップ（切り戻しノブの担保）
        let params_off = EvalParams {
            promo_potential_w: 0.0,
            ..EvalParams::default()
        };
        let off = own_effects_after(&view, &advance, None, &params_off);
        assert_eq!(off.promo_potential, 0.0);
    }

    #[test]
    fn own_effects_drop_hit_exposure_tracks_evacuation_and_blocking() {
        let vp = |file, rank, role| VisiblePiece {
            square: crate::board::make_usi_square(Coord { file, rank }),
            role,
        };
        // 先手の竜が2二（敵陣）・頭2一空き、金を1枚持っている想定
        let mut hand = HashMap::new();
        hand.insert(Role::Gold, 1);
        // 2七に自分の歩を置いておく（頭を塞げる退避先を作るため）
        let view = minimal_view(
            vec![vp(2, 2, Role::Dragon), vp(2, 7, Role::Pawn), vp(5, 9, Role::King)],
            hand,
        );
        let params = EvalParams::default();
        // 退避 = **頭が自駒で塞がるマスへ動く**こと（2二→2八、頭2七は自分の歩）。
        // 2026-08-03 の拡張で「敵陣から出れば安全」ではなくなった
        let evac = ShogiMove::Board {
            from: Coord { file: 2, rank: 2 },
            to: Coord { file: 2, rank: 8 },
            promote: false,
        };
        assert_eq!(own_effects_after(&view, &evac, None, &params).drop_hit_exposure, 0.0);
        // 敵陣を出るだけ（2二→2六、頭2五は空き）では露出は消えない
        let shallow = ShogiMove::Board {
            from: Coord { file: 2, rank: 2 },
            to: Coord { file: 2, rank: 6 },
            promote: false,
        };
        assert!(
            (own_effects_after(&view, &shallow, None, &params).drop_hit_exposure
                - exchange_value(Role::Dragon))
            .abs()
                < 1e-9
        );
        // 無関係の手（玉を動かす）では露出が残る
        let idle = ShogiMove::Board {
            from: Coord { file: 5, rank: 9 },
            to: Coord { file: 5, rank: 8 },
            promote: false,
        };
        assert!(
            (own_effects_after(&view, &idle, None, &params).drop_hit_exposure
                - exchange_value(Role::Dragon))
            .abs()
                < 1e-9
        );
        // 頭（2一）へ持ち駒を打って塞ぐと露出が消える
        let block = ShogiMove::Drop {
            role: Role::Gold,
            to: Coord { file: 2, rank: 1 },
        };
        assert_eq!(own_effects_after(&view, &block, None, &params).drop_hit_exposure, 0.0);
    }

    /// F3: 2手読みの緩和（relief）の上限。cap=0 は従来と同一挙動でなければ
    /// ならない（depth2_replace<1 なので relief は常に risk_mean 未満）
    #[test]
    fn depth2の楽観上限はcap0で挙動を変えない() {
        let relief_of = |risk_mean: f64, delta: f64, cap: f64| {
            let w = EvalParams::default().depth2_replace;
            let relief = w * (risk_mean + delta);
            relief.min((1.0 - cap) * risk_mean)
        };
        // cap=0: 従来どおり relief = w*(risk_mean+delta) がそのまま通る
        let plain = EvalParams::default().depth2_replace * (4.0 + -1.0);
        assert!((relief_of(4.0, -1.0, 0.0) - plain).abs() < 1e-9);
        // cap=1: 楽観方向の置き換えを禁止（実効リスクは静的リスクのまま）
        assert!(relief_of(4.0, -1.0, 1.0).abs() < 1e-9);
        // 悲観方向（delta が静的リスクより大きい損失）は cap に関係なく通る
        assert!(relief_of(4.0, -9.0, 1.0) < 0.0);
        // リスクゼロの手（静かな手）は cap の影響を受けない
        assert!(relief_of(0.0, 0.0, 1.0).abs() < 1e-9);
    }

    #[test]
    fn combine_score_handles_gain_signs() {
        // 正のgain: p_legal で割り引かれる
        assert!((combine_score(2.0, 0.5, 0.0) - 1.0).abs() < 1e-9);
        // 負のgain: 割り引かない（min形。反則に寄るインセンティブを作らない）
        assert!((combine_score(-2.0, 0.5, 0.0) + 2.0).abs() < 1e-9);
        // 反則コストは (1-p_legal) 倍で引かれる
        assert!((combine_score(0.0, 0.75, 1.0) + 0.25).abs() < 1e-9);
        // 2手読みのリスク置換で符号が変わるケース: gain=-0.5 → +0.5 に
        // 再構築した場合、min形が正側の割引へ正しく切り替わる
        let before = combine_score(-0.5, 0.5, 0.0);
        let after = combine_score(0.5, 0.5, 0.0);
        assert!((before + 0.5).abs() < 1e-9);
        assert!((after - 0.25).abs() < 1e-9);
    }

    #[test]
    fn search_budget_scales_with_think_time() {
        let base = SearchBudget::from_ms(900);
        assert_eq!(base.eval_particles, EVAL_PARTICLES);
        assert_eq!(base.depth2_top_k, DEPTH2_TOP_K);
        let big = SearchBudget::from_ms(2000);
        assert!(big.eval_particles > base.eval_particles);
        assert!(big.depth2_top_k > base.depth2_top_k);
        assert!(big.depth2_particles > base.depth2_particles);
        // 極端な予算でも上限で頭打ち
        assert!(SearchBudget::from_ms(600_000).eval_particles <= 2048);
        // 本番向けに絞れば従来より軽くなる
        let small = SearchBudget::from_ms(450);
        assert!(small.eval_particles < base.eval_particles);
    }

    #[test]
    fn eval_params_specs_to_vec_from_vec_stay_aligned() {
        fn changed_indices(a: &[f64], b: &[f64]) -> Vec<usize> {
            a.iter()
                .zip(b)
                .enumerate()
                .filter_map(|(i, (&x, &y))| ((x - y).abs() > 1e-12).then_some(i))
                .collect()
        }

        fn spec_index(name: &str) -> usize {
            EvalParams::SPECS
                .iter()
                .position(|s| s.name == name)
                .unwrap_or_else(|| panic!("SPECSに {name} がない"))
        }

        let base = EvalParams::default();
        let base_vec = base.to_vec();
        assert_eq!(base_vec.len(), EvalParams::SPECS.len());
        assert_eq!(EvalParams::from_vec(&base_vec).to_vec(), base_vec);

        for i in 0..base_vec.len() {
            let mut v = base_vec.clone();
            v[i] += 1.0;
            let round = EvalParams::from_vec(&v).to_vec();
            assert_eq!(changed_indices(&base_vec, &round), vec![i]);
        }

        macro_rules! assert_field_index {
            ($field:ident) => {{
                let mut p = base.clone();
                p.$field += 1.0;
                assert_eq!(
                    changed_indices(&base_vec, &p.to_vec()),
                    vec![spec_index(stringify!($field))]
                );
            }};
        }

        // 全項目を網羅する（一部だけ並べていた頃に drop_probe_w の
        // SPECS 位置ズレを見逃した。SPECS の順序は to_vec の順序と一致が必須で、
        // ズレると SPSA が別の項目の範囲・名前で調整してしまう）
        assert_field_index!(check_bonus);
        assert_field_index!(check_foul_scale);
        assert_field_index!(mover_w_captured);
        assert_field_index!(mover_w_quiet);
        assert_field_index!(mover_check_extra);
        assert_field_index!(capture_reveal_risk);
        assert_field_index!(camp_known_quiet);
        assert_field_index!(camp_scale);
        assert_field_index!(exposed_base);
        assert_field_index!(exposed_known);
        assert_field_index!(home_knownness);
        assert_field_index!(recapture_defended);
        assert_field_index!(exposed_defended);
        assert_field_index!(attack_w);
        assert_field_index!(pressure_w);
        assert_field_index!(foul_cost_base);
        assert_field_index!(foul_cost_pow);
        assert_field_index!(advance_w);
        assert_field_index!(promote_bias);
        assert_field_index!(drop_bias);
        assert_field_index!(prior_weight);
        assert_field_index!(prior_weight_degen);
        assert_field_index!(threat_w);
        assert_field_index!(info_bonus);
        assert_field_index!(big_home_penalty);
        assert_field_index!(hand_drop_w);
        assert_field_index!(backtrack_penalty);
        assert_field_index!(shuffle_penalty);
        assert_field_index!(soft_decay);
        assert_field_index!(king_probe_bonus);
        assert_field_index!(coverage_w);
        assert_field_index!(depth2_replace);
        assert_field_index!(depth2_check_pen);
        assert_field_index!(depth2_recap_discount);
        assert_field_index!(foul_diff_pow);
        assert_field_index!(check_limit_accel);
        assert_field_index!(value_nn_w);
        assert_field_index!(checker_removal_w);
        assert_field_index!(capture_bet_var_w);
        assert_field_index!(mate_threat_w);
        assert_field_index!(mate_risk_w);
        assert_field_index!(king_hole_w);
        assert_field_index!(link_w);
        assert_field_index!(effect_own_w);
        assert_field_index!(effect_opp_w);
        assert_field_index!(link_work_w);
        assert_field_index!(link_work_ref);
        assert_field_index!(repeat_penalty_w);
        assert_field_index!(plan_w);
        assert_field_index!(board_discount_w);
        assert_field_index!(check_strength_w);
        assert_field_index!(escape_cover_w);
        assert_field_index!(defender_capture_w);
        assert_field_index!(drop_hit_evac_w);
        assert_field_index!(promo_potential_w);
        assert_field_index!(hand_option_w);
        assert_field_index!(mate_gate_q0);
        assert_field_index!(king_adj_entry_w);
        assert_field_index!(drop_probe_w);
        assert_field_index!(depth2_optimism_cap);
        assert_field_index!(taint_occ_legal_w);
        assert_field_index!(major_promo_path_w);
        assert_field_index!(exposed_multi_w);
        assert_field_index!(exposed_pawn_head_w);
        assert_field_index!(blind_attack_survive_w);

        // 既定値は自分の SPECS 範囲内にあること（SPSA の中心点が
        // クランプで別の値へ化けるのを防ぐ。位置ズレの二重の網でもある）
        for (spec, v) in EvalParams::SPECS.iter().zip(&base_vec) {
            assert!(
                spec.lo <= *v && *v <= spec.hi,
                "{} の既定 {v} が範囲 [{}, {}] の外",
                spec.name,
                spec.lo,
                spec.hi
            );
        }
    }

    #[test]
    fn check_strength_curve_matches_measured_fouls() {
        // g(K) の符号と単調性: 強い王手（K=1..2）は正、平均的（K≈3.7）でほぼ0、
        // 解消の多い王手は負（再配分であって王手全体の食欲を変えない）
        let g = |k: usize| CHECK_STRENGTH_CURVE / (1.0 + k as f64) - CHECK_STRENGTH_CENTER;
        assert!(g(1) > 0.5);
        assert!(g(2) > 0.2);
        assert!(g(4) < 0.05, "g(4)={}", g(4));
        assert!(g(10) < -0.2);
        for k in 1..20 {
            assert!(g(k) > g(k + 1), "単調減少");
        }
    }

    #[test]
    fn exposed_multi_w_counts_the_second_hanging_piece() {
        use crate::shogi::Piece;
        // 先手の香 1a と歩 4f が、どちらも紐なしで後手の駒に当たられている。
        // max だけだと香が常に勝ち、歩の危険は評価上ゼロになる
        // （quest31-m026 の 4七歩成 が本命なのに緊急性が見えなかった構図）
        let mut pos = Position::empty(Color::Sente);
        let put = |pos: &mut Position, file: i8, rank: i8, color, role| {
            pos.set(Coord { file, rank }, Some(Piece { color, role }));
        };
        put(&mut pos, 1, 1, Color::Sente, Role::Lance);
        put(&mut pos, 4, 6, Color::Sente, Role::Pawn);
        // 後手の龍が 2a から 1a を、歩が 4e から 4f を狙う
        put(&mut pos, 2, 1, Color::Gote, Role::Dragon);
        put(&mut pos, 4, 5, Color::Gote, Role::Pawn);
        let known = HashMap::new();

        let base = EvalParams::default();
        assert_eq!(base.exposed_multi_w, 0.0, "既定は従来の max");
        let max_only = exposed_capture_risk(&pos, Color::Sente, None, &known, &base);

        // 香だけを取り除くと max は歩に落ちる = 歩ぶんの寄与を単独で測れる
        let mut without_lance = pos.clone();
        without_lance.set(Coord { file: 1, rank: 1 }, None);
        let pawn_only = exposed_capture_risk(&without_lance, Color::Sente, None, &known, &base);
        assert!(pawn_only > 0.0);
        assert!(
            max_only > pawn_only,
            "香のほうが高いので max は香を取る（歩の寄与は0）"
        );

        // 複数枚版では歩が w 倍で足される
        let w = 0.3;
        let multi = EvalParams {
            exposed_multi_w: w,
            ..base.clone()
        };
        let both = exposed_capture_risk(&pos, Color::Sente, None, &known, &multi);
        assert!((both - (max_only + w * pawn_only)).abs() < 1e-9);

        // 歩を動かして当たりを外せば、その差分だけリスクが下がる（＝逃げの価値）
        let mut moved = pos.clone();
        moved.set(Coord { file: 4, rank: 6 }, None);
        put(&mut moved, 4, 5, Color::Sente, Role::Tokin);
        let after = exposed_capture_risk(&moved, Color::Sente, None, &known, &multi);
        assert!(
            after < both,
            "取られそうな歩を動かせば複数枚版ではリスクが下がる"
        );
        // 従来の max では同じ手でリスクが動かない（香が支配し続ける）
        let after_max = exposed_capture_risk(&moved, Color::Sente, None, &known, &base);
        assert!((after_max - max_only).abs() < 1e-9);
    }

    #[test]
    fn adds_focal_attacker_picks_support_of_a_contested_square() {
        // 後手番。4六に自分の歩がいて4七へ利いている（=争点）
        let mut view = minimal_view(
            vec![
                VisiblePiece {
                    square: "4f".into(),
                    role: Role::Pawn,
                },
                VisiblePiece {
                    square: "6c".into(),
                    role: Role::King,
                },
            ],
            HashMap::new(),
        );
        view.your_color = Color::Gote;
        view.turn = Color::Gote;
        let own = own_attack_counts(&view);
        assert_eq!(own[crate::belief_features::sq_index(Coord { file: 4, rank: 7 })], 1);

        // 3八銀打: 銀が4七へ利きを足す → 予約対象
        let s3h = ShogiMove::Drop {
            role: Role::Silver,
            to: Coord { file: 3, rank: 8 },
        };
        assert!(adds_focal_attacker(&view, &s3h, &own));

        // 9八銀打: 何の争点にも絡まない → 対象外
        let s9h = ShogiMove::Drop {
            role: Role::Silver,
            to: Coord { file: 9, rank: 8 },
        };
        assert!(!adds_focal_attacker(&view, &s9h, &own));

        // 歩を1マス進めるだけの手も対象外（自分が既に利かせているマスへ
        // 足すわけではない）
        let push = ShogiMove::Board {
            from: Coord { file: 4, rank: 6 },
            to: Coord { file: 4, rank: 7 },
            promote: false,
        };
        assert!(!adds_focal_attacker(&view, &push, &own));
    }

    #[test]
    fn exposed_pawn_head_w_scales_only_the_head_on_piece() {
        use crate::shogi::Piece;
        // 先手の歩 4f が後手の歩 4e の正面に立っている（鉢合わせ）
        let mut pos = Position::empty(Color::Sente);
        pos.set(
            Coord { file: 4, rank: 6 },
            Some(Piece {
                color: Color::Sente,
                role: Role::Pawn,
            }),
        );
        pos.set(
            Coord { file: 4, rank: 5 },
            Some(Piece {
                color: Color::Gote,
                role: Role::Pawn,
            }),
        );
        let known = HashMap::new();
        let base = EvalParams::default();
        assert_eq!(base.exposed_pawn_head_w, 0.0, "既定は従来挙動");
        let plain = exposed_capture_risk(&pos, Color::Sente, None, &known, &base);
        assert!(plain > 0.0);

        let head = EvalParams {
            exposed_pawn_head_w: 0.5,
            ..base.clone()
        };
        let scaled = exposed_capture_risk(&pos, Color::Sente, None, &known, &head);
        assert!((scaled - plain * 1.5).abs() < 1e-9, "正面の駒は 1+w 倍");

        // 同じ当たりでも正面でなければ倍率は掛からない（後手の銀が斜めから）
        let mut side = Position::empty(Color::Sente);
        side.set(
            Coord { file: 4, rank: 6 },
            Some(Piece {
                color: Color::Sente,
                role: Role::Pawn,
            }),
        );
        side.set(
            Coord { file: 3, rank: 5 },
            Some(Piece {
                color: Color::Gote,
                role: Role::Silver,
            }),
        );
        let a = exposed_capture_risk(&side, Color::Sente, None, &known, &base);
        let b = exposed_capture_risk(&side, Color::Sente, None, &known, &head);
        assert!(a > 0.0);
        assert!((a - b).abs() < 1e-9);
    }

    #[test]
    fn escape_cover_value_is_convex_in_uncovered_squares() {
        // 後手玉 5a: 逃げ先候補は 4a,6a,4b,5b,6b の5マス（盤端）
        let king = Coord { file: 5, rank: 1 };
        let mut pos = Position::empty(Color::Sente);
        pos.set(
            king,
            Some(crate::shogi::Piece {
                color: Color::Gote,
                role: Role::King,
            }),
        );
        // 利き無し → U=5 → 1/6
        assert!((escape_cover_value(&pos, Color::Gote, Color::Sente) - 1.0 / 6.0).abs() < 1e-9);

        // 先手金を 5c へ: 4b,5b,6b を被覆 → U=2 → 1/3
        pos.set(
            Coord { file: 5, rank: 3 },
            Some(crate::shogi::Piece {
                color: Color::Sente,
                role: Role::Gold,
            }),
        );
        assert!((escape_cover_value(&pos, Color::Gote, Color::Sente) - 1.0 / 3.0).abs() < 1e-9);

        // 後手自身の歩が 4a を塞ぐ → 逃げ先候補から除外 → U=1 → 1/2。
        // 凸性: 被覆1マスの増分が U が小さいほど大きい（1/6→1/3→1/2）
        pos.set(
            Coord { file: 4, rank: 1 },
            Some(crate::shogi::Piece {
                color: Color::Gote,
                role: Role::Pawn,
            }),
        );
        assert!((escape_cover_value(&pos, Color::Gote, Color::Sente) - 0.5).abs() < 1e-9);

        // 銀を 5b へ: 残る 6a を被覆（5b 自体は金の利きが支える）→ U=0 → 1.0
        pos.set(
            Coord { file: 5, rank: 2 },
            Some(crate::shogi::Piece {
                color: Color::Sente,
                role: Role::Silver,
            }),
        );
        assert!((escape_cover_value(&pos, Color::Gote, Color::Sente) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn exchange_value_discounts_promoted_pieces() {
        // 素の駒は piece_value と一致
        assert_eq!(exchange_value(Role::Silver), piece_value(Role::Silver));
        // と金の反動は (盤上6 + 持ち駒1) / 2 = 3.5 で歩由来の駒として安い
        assert!((exchange_value(Role::Tokin) - 3.5).abs() < 1e-9);
        assert!(exchange_value(Role::Tokin) < exchange_value(Role::Silver));
        // 龍も持ち駒に入るのは飛車ぶん
        assert!(exchange_value(Role::Dragon) < piece_value(Role::Dragon));
        // 元手が安い成駒ほど反動が小さい（と金 < 成香 < 成桂 < 成銀）
        assert!(exchange_value(Role::Tokin) < exchange_value(Role::Promotedlance));
        assert!(exchange_value(Role::Promotedlance) < exchange_value(Role::Promotedknight));
        assert!(exchange_value(Role::Promotedknight) < exchange_value(Role::Promotedsilver));
    }

    #[test]
    fn promotion_widens_coverage() {
        // 3d の歩: 利きは 3c の1マス。成れば金の利き6マスに広がる
        let view = minimal_view(
            vec![VisiblePiece {
                square: "3d".into(),
                role: Role::Pawn,
            }],
            HashMap::new(),
        );
        let quiet = coverage_after(&view, &parse_usi("3d3c").unwrap());
        let promo = coverage_after(&view, &parse_usi("3d3c+").unwrap());
        assert_eq!(quiet, 1.0);
        assert_eq!(promo, 6.0, "と金は金の利き（6マス）");
    }

    /// ブラインド取り返し: 対象マスは**直前の相手手**で取られたマスに限る
    /// （それ以前のマスは相手がもう動かしているかもしれない）。
    /// 見積もりは相手の盤上に残っている駒の平均交換価値
    #[test]
    fn blind_recapture_target_uses_the_latest_capture_only() {
        let view = minimal_view(
            vec![VisiblePiece {
                square: "5i".into(),
                role: Role::King,
            }],
            HashMap::new(),
        );
        let mut log = ObservationLog::default();
        log.record(Observation::OpponentMoved {
            move_number: 2,
            captured_my_piece_at: Some("7g".into()),
        });
        let (sq, value) = blind_recapture_target(&view, &log).expect("直前の手が捕獲なら対象あり");
        assert_eq!(sq, Coord { file: 7, rank: 7 });
        assert!(value > 0.0, "相手の盤上駒の平均交換価値が出る");

        // 直前の相手手が捕獲でなければ対象なし（古い捕獲マスは使わない）
        log.record(Observation::OpponentMoved {
            move_number: 4,
            captured_my_piece_at: None,
        });
        assert!(blind_recapture_target(&view, &log).is_none());
    }

    /// NN段階②のブラインド供給は blind_recapture の一般化なので、**同じ駒種会計**を
    /// 使わなければならない（p=1 のとき blind_recapture と一致する設計）。
    /// 会計がズレると「一般化」ではなく別物の項になる
    #[test]
    fn blind_capture_estimate_matches_blind_recapture_value() {
        let view = minimal_view(
            vec![VisiblePiece {
                square: "5i".into(),
                role: Role::King,
            }],
            HashMap::new(),
        );
        let mut log = ObservationLog::default();
        log.record(Observation::OpponentMoved {
            move_number: 2,
            captured_my_piece_at: Some("7g".into()),
        });
        let (_, value) = blind_recapture_target(&view, &log).unwrap();
        assert!((blind_capture_estimate(&view, &log) - value).abs() < 1e-12);

        // 自分が相手の飛車を取ったら、相手の盤上に残る駒の平均は下がる
        log.record(Observation::MyMove {
            move_number: 3,
            usi: "2h2b".into(),
            captured: Some(Role::Rook),
        });
        let after = blind_capture_estimate(&view, &log);
        assert!(after < value, "取った駒は相手の盤上から消える: {after} < {value}");
        let (_, value_after) = blind_recapture_target(&view, &log).unwrap();
        assert!((after - value_after).abs() < 1e-12);
    }

    /// 同じ自陣形へ戻る手の累積カウント。往復を続けるほど回数が増えること、
    /// 途中で駒を取られたら（＝何かが起きたら）別の形になって数え直しになることを
    /// 確かめる。既存の backtrack_penalty は直前の1手しか見ず固定額なので、
    /// 実戦の 3四角↔2五角 6連続（watch-estimator-20260728-122107）を止められなかった
    #[test]
    fn own_config_repeat_count_grows_with_shuffling() {
        let mut log = ObservationLog::default();
        let mv = |n: u32, usi: &str| Observation::MyMove {
            move_number: n,
            usi: usi.into(),
            captured: None,
        };
        // 玉を 5九 → 5八 → 5九 → 5八 と往復させる（5八は初期局面で空きマス。
        // 自駒の上へ動かす手を書くと GameModel がその駒を上書きしてしまい、
        // 「同じ形へ戻った」ことにならないので注意）
        log.record(mv(1, "5i5h"));
        log.record(mv(3, "5h5i"));
        log.record(mv(5, "5i5h"));
        let counts = own_config_history(Color::Sente, &log);

        // 「5八に玉がいる」形は 1手目と5手目の2回出ている
        let mut model = GameModel::new(Color::Sente);
        for e in log.events() {
            model.apply(e);
        }
        let now = model.my_pieces();
        let fp_now = own_config_fingerprint(
            now.iter().map(|p| (p.square.as_str(), p.role)),
            &model.my_hand(),
        );
        assert_eq!(counts.get(&fp_now), Some(&2), "5八の形は2回出ている");

        // ここから 5九へ戻す手を指すと、その形は初期局面と3手目の2回ぶん既出
        let view = minimal_view(now.clone(), model.my_hand());
        let back = own_config_fingerprint_after(&view, &parse_usi("5h5i").unwrap());
        assert_eq!(counts.get(&back), Some(&2), "5九の形も2回出ている＝往復の証拠");

        // 一度も出ていない形（別の駒を動かす）は 0
        let fresh = own_config_fingerprint_after(&view, &parse_usi("2g2f").unwrap());
        assert_eq!(counts.get(&fresh), None, "新しい形は既出でない");
    }

    /// 間に駒を取られたら形が変わるので、往復カウントは持ち越さない
    /// （「何も起きていないのに同じ形へ戻る」ときだけ効かせるための性質）
    #[test]
    fn own_config_repeat_resets_when_material_changes() {
        let mut log = ObservationLog::default();
        log.record(Observation::MyMove {
            move_number: 1,
            usi: "2h5h".into(),
            captured: None,
        });
        // 相手に 5八の飛車を取られる
        log.record(Observation::OpponentMoved {
            move_number: 2,
            captured_my_piece_at: Some("5h".into()),
        });
        let counts = own_config_history(Color::Sente, &log);
        let mut model = GameModel::new(Color::Sente);
        for e in log.events() {
            model.apply(e);
        }
        let view = minimal_view(model.my_pieces(), model.my_hand());
        // 飛車を失った後の形は、それ以前のどの形とも一致しない
        let fresh = own_config_fingerprint_after(&view, &parse_usi("2g2f").unwrap());
        assert_eq!(counts.get(&fresh), None);
        assert_eq!(counts.len(), 2, "初期形と 5八飛の形の2つだけが記録されている");
    }

    /// V2 の要点は「可動性」ではなく「**利きがどこを向いているか**」であること。
    /// 底歩（可動性ほぼゼロ・自玉の隣に利く）と、隅で何もしていないと金
    /// （可動性はあるが両玉から遠い）が分かれなければ意味がない
    /// （ユーザー指摘、2026-07-28: 「動けない駒＝価値なし」では底歩を取りこぼす）
    #[test]
    fn effect_own_separates_bottom_pawn_from_idle_corner_tokin() {
        // 自玉 5九。底歩 5八（利きは 5七の1マスだけ = 可動性は最低）
        let bottom_pawn = minimal_view(
            vec![
                VisiblePiece {
                    square: "5i".into(),
                    role: Role::King,
                },
                VisiblePiece {
                    square: "5h".into(),
                    role: Role::Pawn,
                },
            ],
            HashMap::new(),
        );
        // 同じ玉位置で、と金が隅 1一（利きは 2一・1二の2マス = 可動性は上）
        let corner_tokin = minimal_view(
            vec![
                VisiblePiece {
                    square: "5i".into(),
                    role: Role::King,
                },
                VisiblePiece {
                    square: "1a".into(),
                    role: Role::Tokin,
                },
            ],
            HashMap::new(),
        );
        // 玉を 5九→4九 に動かす（どちらも同じ手）ことで、駒の配置だけが違う
        // 2局面の effect_own を比べる
        let a = own_effects_after(&bottom_pawn, &parse_usi("5i4i").unwrap(), None, &EvalParams::default());
        let b = own_effects_after(&corner_tokin, &parse_usi("5i4i").unwrap(), None, &EvalParams::default());
        assert!(
            a.effect_own > b.effect_own,
            "底歩（可動性ゼロだが玉の近く）が隅のと金（可動性はあるが遠い）より \
             高く評価されるべき: 底歩={} 隅のと金={}",
            a.effect_own,
            b.effect_own
        );
        // 可動性そのものは逆順（と金2マス > 底歩1マス）であることも確かめておく。
        // ここが逆順でなければ、このテストは V2 の性質を検証していない
        assert!(
            b.coverage > a.coverage,
            "可動性は隅のと金のほうが大きいはず: 底歩={} 隅のと金={}",
            a.coverage,
            b.coverage
        );
    }

    /// V5（盤上駒の減価）は「この手で盤上に増えた自駒の価値」だけを持つ。
    /// 盤上の合計を持つと gain のゼロ点が下がり、combine_score の min 形で
    /// p_legal 割引が消える（threat_value の差分化で踏んだ罠）
    #[test]
    fn board_material_added_counts_only_the_increment() {
        let view = minimal_view(
            vec![
                VisiblePiece {
                    square: "5i".into(),
                    role: Role::King,
                },
                VisiblePiece {
                    square: "2d".into(),
                    role: Role::Pawn,
                },
            ],
            HashMap::from([(Role::Lance, 1)]),
        );
        // 打ち: 打った駒の価値そのもの
        let drop = own_effects_after(&view, &parse_usi("L*1b").unwrap(), None, &EvalParams::default());
        assert_eq!(drop.board_material_added, piece_value(Role::Lance));
        // 静かな盤上の手: 増分ゼロ
        let quiet = own_effects_after(&view, &parse_usi("2d2c").unwrap(), None, &EvalParams::default());
        assert_eq!(quiet.board_material_added, 0.0);
        // 成り: 増えたぶんだけ（歩1 → と6）
        let promo = own_effects_after(&view, &parse_usi("2d2c+").unwrap(), None, &EvalParams::default());
        assert_eq!(
            promo.board_material_added,
            piece_value(Role::Tokin) - piece_value(Role::Pawn)
        );
    }

    /// 既定（重み0）では V5 が働かない = 従来と同じ挙動
    #[test]
    fn board_discount_is_disabled_by_default() {
        assert_eq!(EvalParams::default().board_discount_w, 0.0);
    }

    /// 既定（重み0）ではブラインド供給が働かない = 従来と同じ挙動
    #[test]
    fn belief_gain_is_disabled_by_default() {
        assert_eq!(
            belief_gain_w(),
            0.0,
            "TSUITATE_BELIEF_GAIN_W の既定は0（挙動不変）でなければならない"
        );
    }

    /// 玉位置ネットの既定: ブレンドは無効（単独中立・投影と複合すると
    /// 投影の利得を打ち消す）、投影は有効（ペア3シード +6.7pt で採用）
    #[test]
    fn king_net_defaults() {
        assert_eq!(
            king_net_w(),
            0.0,
            "TSUITATE_KING_NET_W の既定は0（ブレンドは不採用）でなければならない"
        );
        assert!(
            king_net_proj(),
            "TSUITATE_KING_NET_PROJ の既定は on（2026-07-31 採用。=0 で切り戻し）"
        );
    }

    /// blend_king_dist: λ=0.5 の混合が正規化を保ち、taint 空ならネットのみになる
    #[test]
    fn blend_king_dist_normalizes() {
        let a = Coord { file: 5, rank: 1 };
        let b = Coord { file: 5, rank: 2 };
        let taint = vec![(a, 1.0)];
        let net = vec![(a, 0.25), (b, 0.75)];
        let mixed = blend_king_dist(&taint, &net, 0.5);
        let total: f64 = mixed.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-9, "正規化が壊れた: {total}");
        let pa = mixed.iter().find(|(c, _)| *c == a).unwrap().1;
        assert!((pa - (0.5 + 0.5 * 0.25)).abs() < 1e-9);
        // taint 空: ネット分布そのもの
        let only_net = blend_king_dist(&[], &net, 0.3);
        let pb = only_net.iter().find(|(c, _)| *c == b).unwrap().1;
        assert!((pb - 0.75).abs() < 1e-9);
    }

    /// ネット誘導の玉移設: 分位点割り当てが分布に比例し、候補内に収まる
    #[test]
    fn project_taint_kings_follows_net_distribution() {
        // 敵玉だけ 9i に居る粒子を4つ用意し、候補 {5a,5b}・ネット {5a:0.75, 5b:0.25}
        // へ移設する → 3粒子が 5a、1粒子が 5b に割り当たるはず
        let mut pos = Position::empty(Color::Sente);
        let k = Coord { file: 9, rank: 9 };
        pos.set(
            k,
            Some(crate::shogi::Piece {
                color: Color::Gote,
                role: Role::King,
            }),
        );
        let pool: Vec<(&Position, f64)> = vec![(&pos, 1.0); 4];
        let a = Coord { file: 5, rank: 1 };
        let b = Coord { file: 5, rank: 2 };
        let cands: std::collections::BTreeSet<Coord> = [a, b].into_iter().collect();
        let net = vec![(a, 0.75), (b, 0.25)];
        let out = project_taint_kings(&pool, &cands, Color::Gote, Some(&net));
        let kings: Vec<Coord> = out
            .iter()
            .filter_map(|(p, _)| p.king_square(Color::Gote))
            .collect();
        assert_eq!(kings.iter().filter(|&&c| c == a).count(), 3);
        assert_eq!(kings.iter().filter(|&&c| c == b).count(), 1);
        // ネット無しは従来どおり最近傍（9i から近いのは 5b = rank 差が小さい方…
        // チェビシェフ距離は 5a が max(4,8)=8・5b が max(4,7)=7 なので 5b）
        let out = project_taint_kings(&pool, &cands, Color::Gote, None);
        assert!(
            out.iter()
                .all(|(p, _)| p.king_square(Color::Gote) == Some(b))
        );
    }

    /// V3: 紐は「移動できるマス」ではなく「利かせているマス」で数える。
    /// `move_targets` は自駒のいるマスを除くので、そのままでは紐が常にゼロになる
    #[test]
    fn linked_value_counts_pieces_defended_by_another_piece() {
        let gold = VisiblePiece {
            square: "5h".into(),
            role: Role::Gold,
        };
        let king = VisiblePiece {
            square: "5i".into(),
            role: Role::King,
        };
        // 玉(5i)だけが金(5h)を守っている形。玉の利きも紐に数える
        let view = minimal_view(vec![gold.clone(), king.clone()], HashMap::new());
        let alone = own_effects_after(&view, &parse_usi("5h5g").unwrap(), None, &EvalParams::default());
        assert_eq!(alone.linked_value, 0.0, "5g へ出れば玉から離れて紐が切れる");
        let stay = own_effects_after(&view, &parse_usi("5i4i").unwrap(), None, &EvalParams::default());
        assert!(
            stay.linked_value > 0.0,
            "玉が 4i へ寄っても金 5h は玉の利きに入ったまま"
        );

        // 銀を足すと紐が増える（価値の合計で数える）
        let mut pieces = vec![gold, king];
        pieces.push(VisiblePiece {
            square: "4h".into(),
            role: Role::Silver,
        });
        let view2 = minimal_view(pieces, HashMap::new());
        let two = own_effects_after(&view2, &parse_usi("5i4i").unwrap(), None, &EvalParams::default());
        assert!(two.linked_value > stay.linked_value);
    }

    #[test]
    fn king_holes_counts_unsupported_neighbours() {
        // 先手玉5九だけの盤。隣接8マスのうち盤内は5マス（4八,5八,6八,4九,6九）で
        // 玉自身の利きは数えないので、守りが無ければ全部が穴
        let lone_king = vec![VisiblePiece { square: "5i".into(), role: Role::King }];
        let view = minimal_view(lone_king.clone(), HashMap::new());
        // 玉から離れたマスへ歩を打つ = 近傍の穴は減らない
        let far = king_holes_after(&view, &parse_usi("P*1e").unwrap());
        assert_eq!(far, 5.0, "盤内の近傍5マスが全部穴のはず: {far}");

        // 5八へ金を打つと、そのマスが埋まり（占有）、金の利きが 4八/6八/5七… を守る
        let filled = king_holes_after(&view, &parse_usi("G*5h").unwrap());
        assert!(filled < far, "支えを足せば穴は減る: {filled} < {far}");
    }

    #[test]
    fn king_holes_ignores_the_kings_own_effect() {
        // 玉の利きを数えてしまうと近傍が全部「守られている」ことになり、
        // 項が常にゼロになる（やねうら王も「玉以外の味方の利き」で数える）
        let view = minimal_view(
            vec![VisiblePiece { square: "5e".into(), role: Role::King }],
            HashMap::new(),
        );
        // 盤の中央なので近傍8マスすべてが盤内。玉の利きを除けば全部穴
        let holes = king_holes_after(&view, &parse_usi("5e5f").unwrap());
        assert_eq!(holes, 8.0, "玉自身の利きは支えに数えない: {holes}");
    }


    /// 相手玉を kf筋・自陣に歩を1枚置いた盤（指紋がユニークになるよう pawn_sq を変える）
    fn synth_position(king_file: i8, pawn_rank: i8) -> Position {
        let mut pos = Position::empty(Color::Sente);
        pos.set(
            Coord { file: 5, rank: 9 },
            Some(crate::shogi::Piece {
                color: Color::Sente,
                role: Role::King,
            }),
        );
        pos.set(
            Coord {
                file: king_file,
                rank: 1,
            },
            Some(crate::shogi::Piece {
                color: Color::Gote,
                role: Role::King,
            }),
        );
        pos.set(
            Coord {
                file: 5,
                rank: pawn_rank,
            },
            Some(crate::shogi::Piece {
                color: Color::Sente,
                role: Role::Pawn,
            }),
        );
        pos
    }

    #[test]
    fn stratified_sample_respects_count_cap_and_prefix_diversity() {
        let mut rng = StdRng::seed_from_u64(1);
        // 9層（玉位置 file 1..=9）× 各6粒子 = 54ユニーク
        let mut particles = vec![];
        for kf in 1..=9i8 {
            for pr in 2..=7i8 {
                particles.push(synth_position(kf, pr));
            }
        }
        let miss = vec![0u8; particles.len()];
        // 上限16 < 層数9×最低枠4=36: 件数は必ず16以下
        let sample = stratified_sample(
            &particles,
            &miss,
            &vec![0u8; particles.len()],
            &vec![0.0f64; particles.len()],
            Color::Sente,
            &ParticleCtx::default(),
            16,
            &mut rng,
        );
        assert!(sample.len() <= 16, "len={}", sample.len());
        // ラウンドロビン順: 先頭9件で9層すべての玉位置が現れる
        let prefix_kings: HashSet<_> = sample
            .iter()
            .take(9)
            .map(|(p, _)| p.king_square(Color::Gote))
            .collect();
        assert_eq!(prefix_kings.len(), 9, "prefixが層化されていない");
        // 上限が大きい場合も件数はユニーク数以下・重みは旧方式と一致
        // （不変条件①: 全ユニーク・logw=0・ソフトなしなら重み和 = ユニーク数）
        let sample = stratified_sample(
            &particles,
            &miss,
            &vec![0u8; particles.len()],
            &vec![0.0f64; particles.len()],
            Color::Sente,
            &ParticleCtx::default(),
            512,
            &mut rng,
        );
        assert_eq!(sample.len(), 54);
        let mass: f64 = sample.iter().map(|(_, w)| w).sum();
        assert!((mass - 54.0).abs() < 1e-6, "mass={mass}");
    }

    #[test]
    fn stratified_sample_excludes_tainted_particles() {
        // 物理不整合（phys_taint>0）の粒子は通常サンプルに混ざらない（C-7 P3）
        let mut rng = StdRng::seed_from_u64(11);
        let clean = synth_position(1, 2);
        let tainted = synth_position(2, 3);
        let particles = vec![clean.clone(), tainted.clone()];
        let miss = vec![0u8, 0u8];
        let taints = vec![0u8, 1u8];
        let logw = vec![0.0f64, 0.0];
        let sample = stratified_sample(
            &particles,
            &miss,
            &taints,
            &logw,
            Color::Sente,
            &ParticleCtx::default(),
            16,
            &mut rng,
        );
        assert!(!sample.is_empty());
        assert!(
            sample
                .iter()
                .all(|(p, _)| p.fingerprint() == clean.fingerprint()),
            "taint 粒子がサンプルに混ざっている"
        );
        // 較正: ユニーク1件（クリーンのみ）
        let mass: f64 = sample.iter().map(|(_, w)| w).sum();
        assert!((mass - 1.0).abs() < 1e-6, "mass={mass}");
    }

    #[test]
    fn multiplicity_survives_unique_folding() {
        // 不変条件②（C-7 P1）: ESSリサンプリング後の複製数は事後質量。
        // 同一指紋3個+別指紋1個（全て logw=0）→ 質量比はちょうど 3:1 になり、
        // 合計は較正アンカー（ユニーク2件×1.0）へ正規化される
        let a = synth_position(1, 2);
        let b = synth_position(1, 4); // 同じ玉位置 = 同じ層、別指紋
        let particles = vec![a.clone(), a.clone(), a.clone(), b.clone()];
        let miss = vec![0u8; 4];
        let logw = vec![0.0f64; 4];
        let a_fp = a.fingerprint();
        let trials = 200;
        let mut a_share_sum = 0.0;
        for seed in 0..trials {
            let mut rng = StdRng::seed_from_u64(seed);
            let sample = stratified_sample(
                &particles,
                &miss,
                &vec![0u8; particles.len()],
                &logw,
                Color::Sente,
                &ParticleCtx::default(),
                16,
                &mut rng,
            );
            let total: f64 = sample.iter().map(|(_, w)| w).sum();
            assert!((total - 2.0).abs() < 1e-6, "較正: ユニーク2件で mass=2.0");
            let a_mass: f64 = sample
                .iter()
                .filter(|(p, _)| p.fingerprint() == a_fp)
                .map(|(_, w)| w)
                .sum();
            a_share_sum += a_mass / total;
        }
        let avg = a_share_sum / trials as f64;
        assert!(
            (avg - 0.75).abs() < 0.05,
            "multiplicity が評価重みに反映されていない: a_share={avg}（期待 0.75）"
        );
    }

    #[test]
    fn stratum_representative_is_weight_proportional() {
        // 同一層に重み 1.0 と 0.125（logw = ln 0.125。フィルタが課金済みの想定）の
        // 2粒子。quota=1 のとき層代表は重み比例（重い側 ≈ 89%）で選ばれるべき。
        // 一様シャッフルだと 50% になる（回帰: 2026-07-15 追加レビュー）
        let strict = synth_position(1, 2);
        let soft = synth_position(1, 3); // 同じ玉位置 = 同じ層、別指紋
        let particles = vec![strict.clone(), soft];
        let miss = vec![0u8, 0u8];
        let logw = vec![0.0f64, 0.125f64.ln()];
        let strict_fp = strict.fingerprint();
        let mut strict_hits = 0;
        let trials = 400;
        for seed in 0..trials {
            let mut rng = StdRng::seed_from_u64(seed);
            let sample = stratified_sample(
                &particles,
                &miss,
                &vec![0u8; particles.len()],
                &logw,
                Color::Sente,
                &ParticleCtx::default(),
                1,
                &mut rng,
            );
            assert_eq!(sample.len(), 1);
            if sample[0].0.fingerprint() == strict_fp {
                strict_hits += 1;
            }
        }
        let share = strict_hits as f64 / trials as f64;
        // 期待値 1.0/(1.0+0.125) ≒ 0.889。一様（0.5）とも過剰（→1.0）とも
        // 区別できる両側の閾値で検証
        assert!(
            share > 0.84 && share < 0.94,
            "strictの代表率が重み比例になっていない: {share}"
        );
    }

    #[test]
    fn resampling_does_not_double_apply_weights() {
        // 同一層に [1.0, 1.0, 0.125（logw課金済み）] の3粒子、quota=2。
        // 軽い粒子の期待質量シェアは 0.125/2.125 ≒ 5.9%。
        // 「重み比例で選び、さらに元の重みも配る」二重適用だと ~1.8% に沈む
        // （2026-07-15 追加レビューの回帰テスト）
        let s1 = synth_position(1, 2);
        let s2 = synth_position(1, 4);
        let soft = synth_position(1, 6); // 同じ玉位置 = 同じ層
        let soft_fp = soft.fingerprint();
        let particles = vec![s1, s2, soft];
        let miss = vec![0u8, 0u8, 0u8];
        let logw = vec![0.0f64, 0.0, 0.125f64.ln()];
        let trials = 400;
        let mut share_sum = 0.0;
        for seed in 0..trials {
            let mut rng = StdRng::seed_from_u64(1000 + seed);
            let sample = stratified_sample(
                &particles,
                &miss,
                &vec![0u8; particles.len()],
                &logw,
                Color::Sente,
                &ParticleCtx::default(),
                2,
                &mut rng,
            );
            let total: f64 = sample.iter().map(|(_, w)| w).sum();
            let soft_mass: f64 = sample
                .iter()
                .filter(|(p, _)| p.fingerprint() == soft_fp)
                .map(|(_, w)| w)
                .sum();
            share_sum += soft_mass / total.max(1e-9);
        }
        let avg = share_sum / trials as f64;
        assert!(
            avg > 0.03 && avg < 0.09,
            "軽い粒子の期待寄与が歪んでいる: avg={avg}（期待 ≒ 0.059）"
        );
    }

    #[test]
    fn stratified_sample_keeps_soft_evidence_calibration() {
        let mut rng = StdRng::seed_from_u64(2);
        // 20ユニーク全てが info_miss=1（フィルタが logw へ ln(EPS_INFO) 課金済み）。
        // 較正アンカー = min(16,20) × EPS_INFO（ソフトは証拠として EPS_INFO 人分）
        let particles: Vec<Position> = (2..=7)
            .flat_map(|pr| (1..=4).map(move |kf| synth_position(kf, pr)))
            .take(20)
            .collect();
        let miss = vec![1u8; particles.len()];
        let logw = vec![EPS_INFO.ln(); particles.len()];
        let sample = stratified_sample(
            &particles,
            &miss,
            &vec![0u8; particles.len()],
            &logw,
            Color::Sente,
            &ParticleCtx::default(),
            16,
            &mut rng,
        );
        assert!(sample.len() <= 16);
        let mass: f64 = sample.iter().map(|(_, w)| w).sum();
        let expected = 16.0 * EPS_INFO;
        assert!(
            (mass - expected).abs() < 1e-6,
            "ソフト証拠の較正が崩れている: mass={mass}（期待{expected}）"
        );
    }

    #[test]
    fn chooses_some_move() {
        let view = minimal_view(
            vec![VisiblePiece {
                square: "7g".into(),
                role: Role::Pawn,
            }],
            HashMap::new(),
        );
        assert_eq!(
            choose_move(&view, &HashSet::new()),
            Some("7g7f".to_string())
        );
    }

    #[test]
    fn skips_fouled_moves_and_resigns_when_exhausted() {
        let view = minimal_view(
            vec![VisiblePiece {
                square: "7g".into(),
                role: Role::Pawn,
            }],
            HashMap::new(),
        );
        let mut tried = HashSet::new();
        tried.insert("7g7f".to_string());
        assert_eq!(choose_move(&view, &tried), None);
    }

    #[test]
    fn may_resolve_check_filters_hopeless_moves() {
        // 先手玉 5i。ライン外への手・桂の利き元以外は王手を解消しえない
        let view = minimal_view(
            vec![
                VisiblePiece {
                    square: "5i".into(),
                    role: Role::King,
                },
                VisiblePiece {
                    square: "7g".into(),
                    role: Role::Pawn,
                },
            ],
            HashMap::new(),
        );
        let ok = |usi: &str| may_resolve_check(&view, &parse_usi(usi).unwrap());
        assert!(ok("5i5h"), "玉移動は常に候補");
        assert!(
            ok("7g5g"),
            "自玉と同段（ライン上）への移動は合駒/取りになりうる"
        );
        assert!(ok("7g5e"), "架空の手でも判定対象はライン（5筋）上の着地点");
        assert!(!ok("7g7f"), "ライン外への移動は王手放置が確定");
    }

    #[test]
    fn may_resolve_check_knight_source_and_drops() {
        let view = minimal_view(
            vec![VisiblePiece {
                square: "5i".into(),
                role: Role::King,
            }],
            HashMap::new(),
        );
        let mv = |usi: &str| parse_usi(usi).unwrap();
        // 4g/6g は相手桂の利き元 → 盤上の駒での取りは候補
        assert!(may_resolve_check(&view, &mv("4f4g")));
        // 打ちは駒を取れないので桂の利き元でも解消しえない
        assert!(!may_resolve_check(&view, &mv("P*4g")));
        // ライン上への打ちは合駒
        assert!(may_resolve_check(&view, &mv("P*5e")));
        assert!(!may_resolve_check(&view, &mv("P*4e")));
    }

    #[test]
    fn estimator_in_check_prefers_resolving_moves() {
        // 粒子が王手を反映していなくても（空ログ = 初期局面粒子）、
        // you_in_check なら解消しうる手（ここでは玉移動のみ）しか指さない
        let mut view = minimal_view(
            vec![
                VisiblePiece {
                    square: "5i".into(),
                    role: Role::King,
                },
                VisiblePiece {
                    square: "7g".into(),
                    role: Role::Pawn,
                },
            ],
            HashMap::new(),
        );
        view.you_in_check = true;
        let mut strat = EstimatorStrategy::new();
        let log = ObservationLog::default();
        let usi = strat.choose(&view, &log, &HashSet::new()).unwrap();
        assert!(
            usi.starts_with("5i"),
            "王手中は玉移動を選ぶはず（選ばれた手: {usi}）"
        );
    }

    #[test]
    fn make_knows_heuristic() {
        assert!(make("heuristic").is_some());
        assert!(make("nonsense").is_none());
    }

    #[test]
    fn make_knows_frozen_versions() {
        assert!(make("estimator").is_some());
        assert!(make("estimator_v6").is_some());
        assert!(make("estimator_v7").is_some());
        assert!(make("estimator_v8").is_some());
        assert!(make("estimator_v9").is_some());
        assert!(make("estimator_v10").is_some());
        assert!(make("estimator_v11").is_some());
        assert!(make("estimator_v12").is_some());
        assert!(make_seeded("estimator_v12", 1).is_some());
        assert!(make("estimator_v13").is_some());
        assert!(make_seeded("estimator_v13", 1).is_some());
        // 破棄済みの凍結版は登録されていない
        assert!(make("estimator_v5").is_none());
    }
}
