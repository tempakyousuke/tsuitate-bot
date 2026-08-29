//! 指し手の選択。
//!
//! `Strategy` trait の実装を差し替えて強さを比較する（bin/arena.rs で対戦できる）。
//! 既定（`DEFAULT_STRATEGY`）は `estimator`。
//! - `Heuristic`: サイト内蔵の簡易botと同じ「前進を好むヒューリスティック＋乱数」
//! - `EstimatorStrategy`: 観測履歴から相手局面の粒子集合を維持し（estimator.rs）、
//!   候補手を粒子平均で評価する
//! - `estimator_v6` … `estimator_v14`: `frozen/` の凍結版（アリーナの基準）

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rand::Rng;

use crate::board::{
    defend_targets, drop_targets, make_usi_drop, make_usi_move, make_usi_square, move_targets,
    parse_usi_square, promotion_choice, Coord, Promotion,
};
use crate::check::CheckSolver;
use crate::estimator::{opp_reply_weights, Estimator, EPS_INFO};
use crate::likelihood::{particle_features, particle_log_weight, ParticleCtx, FITTED_THETA};
use crate::model::GameModel;
use crate::observation::{stale_king_foul_dests, Observation, ObservationLog};
use crate::opening::OpeningBook;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::protocol::{Color, PlayerView, Role, VisiblePiece};
use crate::shogi::{parse_usi, piece_value, promote_role, unpromote_role, Position, ShogiMove};

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

    /// **診断専用**: 直近の choose が評価に使った粒子プールを保存させる
    /// （既定 off = コピーもしないので実対局の挙動とコストは不変）。
    /// issue #28 の P0-3 / P0-6 が「ランキングを作ったのと同じ粒子」で
    /// 危険質量を測るために使う。凍結版は編集しないので既定の no-op のまま
    fn set_capture_particles(&mut self, _on: bool) {}

    /// [`set_capture_particles`] を有効にしていたときの直近のスナップショット。
    /// 定跡で指した手番・候補ゼロの手番は None
    fn last_particles(&self) -> Option<&ParticleSnapshot> {
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

    /// **オラクル錨**（bin/scenario 専用）: 観測ログの接頭辞 `log_prefix` の
    /// 時点の**真実の局面** `truth` を内部推定器に与え、以後の観測だけを
    /// 通常どおり処理させる（`Estimator::oracle_anchor`）。「直前の相手手だけが
    /// 分からない」状態を粒子生成なしで作るための切り分け用。対応しない戦略
    /// （凍結版）は false を返す。**実対局・アリーナからは呼ばない**
    fn oracle_anchor(
        &mut self,
        _view: &PlayerView,
        _log_prefix: &ObservationLog,
        _truth: &Position,
    ) -> bool {
        false
    }
}

/// `prewarm_strategy` の「先頭 `skip` イベントは prewarm せずに記録だけする」版。
/// オラクル錨（`Strategy::oracle_anchor`）で接頭辞まで帳簿を進めた戦略に、
/// 残りのイベントだけを実対局と同じ規約（自分手番イベントの直前で prewarm）で
/// 食わせるために使う
pub fn prewarm_strategy_from(
    strat: &mut dyn Strategy,
    view: &PlayerView,
    full: &ObservationLog,
    skip: usize,
) {
    let mut running = ObservationLog::default();
    for (i, e) in full.events().iter().enumerate() {
        if i >= skip && matches!(e, Observation::MyMove { .. } | Observation::MyFoul { .. }) {
            strat.prewarm(view, &running);
        }
        running.record(e.clone());
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
        "estimator_v14" => Some(Box::new(
            crate::frozen::estimator_v14::EstimatorV14::with_seed(seed),
        )),
        _ => make(name),
    }
}

/// **設定を尊重する戦略名か**（issue #21）。
///
/// 現行 estimator 系だけが [`crate::config::StrategyConfig`] を instance で持つ。
/// 凍結版 v6〜v14 は凍結時点のコピーの中でプロセス env を直接読むので、
/// config を渡しても**無視される**。取り違えを検出できるように名前で分ける。
pub fn honors_config(name: &str) -> bool {
    matches!(name, "estimator" | "estimator_rush")
}

/// 設定とシードを明示して作る（arena / checkpoint arena の arm 用）。
///
/// config を尊重しない戦略名（凍結版）では **None** を返す: プロセス env を
/// 触らずに arm 固有のノブを渡せるのは現行 estimator だけで、凍結版に対して
/// 「設定したつもり」になるのが PR #20 で見つかった事故そのものだから。
pub fn make_seeded_with_config(
    name: &str,
    seed: u64,
    config: Arc<crate::config::StrategyConfig>,
) -> Option<Box<dyn Strategy + Send>> {
    make_with_config(name, Some(seed), config)
}

/// [`make_seeded_with_config`] の **seed 任意**版。
///
/// `seed: None` は `make` と同じくエントロピー由来（対局ごとに違う乱数）。
/// **ノブの有無で乱数条件が変わってはいけない**ので、共通乱数法を使わない
/// 経路（`ARENA_MATCH_SEED` 未指定の通常アリーナ）はここを通すこと
/// —— `Some(0)` に落とすと全対局で候補が同じシードになり、対照との比較が
/// ノブ以外の理由で崩れる（PR #22 レビュー指摘1）。
pub fn make_with_config(
    name: &str,
    seed: Option<u64>,
    config: Arc<crate::config::StrategyConfig>,
) -> Option<Box<dyn Strategy + Send>> {
    Some(Box::new(estimator_with_config(name, seed, config)?))
}

/// [`make_with_config`] の実体（テストが seed の扱いを直接見られるよう
/// boxing の手前で切ってある）。
fn estimator_with_config(
    name: &str,
    seed: Option<u64>,
    config: Arc<crate::config::StrategyConfig>,
) -> Option<EstimatorStrategy> {
    let book_line = match name {
        "estimator" => None,
        // 定跡の読み込みも config のパスで行う
        "estimator_rush" => {
            let _cfg = crate::config::scoped(&config);
            Some(OpeningBook::line_index("居飛車速攻")?)
        }
        _ => return None,
    };
    Some(EstimatorStrategy::with_config(
        config,
        EvalParams::default(),
        book_line,
        seed,
    ))
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
        "estimator_v14" => Some(Box::new(crate::frozen::estimator_v14::EstimatorV14::new())),
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
    /// gain に加算された自玉近傍の敵駒排除ボーナス
    /// （own_zone_capture_w。粒子加重平均、内訳表示用）
    pub own_zone: f64,
    /// **プローブ影の監査**（`TSUITATE_PROBE_AUDIT=1`）: 玉/経路プローブを
    /// **重み1で計算した値**（スコアには足さない）。「w をいくつにすれば
    /// この候補が首位を取れたか」= w* = 首位との差 / この値、を後から出すため
    pub probe_unit: f64,
    /// プローブの反則質量（この手が違法だと言う粒子の割合。監査用）
    pub probe_mass: f64,
    /// 反則質量のうち**最大の回収先マス**が占める割合（原因駒の集中度。
    /// 低いと「どの駒が原因か割れている」= 指し直しで取りに行けない）
    pub probe_concentration: f64,
    /// 打ちプローブの反則情報価値（drop_probe_w）。gain の**外側**
    /// （combine_score の (1−p_legal) 側）へ加算されている正の値。
    /// `score = combine_score(gain, p_legal, foul_cost) + foul_probe + adjust`
    pub foul_probe: f64,
    /// `adjust` に含まれるタイブレーク乱数（0〜0.01）。学習・回帰の特徴量から
    /// 乱数を外せるように分離して持つ（`adjust - tiebreak` が決定的な補正）
    pub tiebreak: f64,
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
/// **アリーナ（1000秒+3秒）も本番サイト（300秒+3秒）も既定の 2000 のまま**でよい
/// （2026-07-26 実測: 300秒+3秒・100局で時間切れ0・クロック消費13.9%）。
/// 900 へ絞ると −14.5pt、8000 へ増やしても +0.5pt で飽和しており、
/// もう強さの調整ノブではない。候補側だけ変える版比較・スイープには
/// `TSUITATE_CAND_THINK_BUDGET_MS` を使う（v6〜v11 と v14 以降の凍結版はこの名前を
/// 知らない。**v12 / v13 は凍結時に持ち込んでいて読む**）。
pub(crate) const DEFAULT_THINK_BUDGET_MS: u64 = 2000;

// 予算の解決は `crate::config::StrategyConfig::from_source`
// （`TSUITATE_CAND_THINK_BUDGET_MS` > `TSUITATE_THINK_BUDGET_MS` > 既定値）。
// **凍結版 v6〜v11 / v12・v13 は自分のコピーの中で env を読む**ので、
// プロセス env で予算を渡すと相手側の予算まで動く。候補側だけ変えたいときは
// config（`StrategyConfig::think_budget_ms`）で渡すこと（issue #21）
/// スケール1.0の基準予算。v5 までの暗黙の実測上限（p99 ≒ 900ms）
const REFERENCE_BUDGET_MS: f64 = 900.0;

/// taint 粒子の玉位置を `deduce::opp_king_candidates` の候補集合へ引き戻すか
/// （既定 on、`TSUITATE_TAINT_KING_FIX=0` で従来挙動）
fn taint_king_fix() -> bool {
    crate::config::current(|c| c.strategy.taint_king_fix)
}

/// taint 粒子の玉を移設するとき、移設先に空きマスを優先するか（既定 on、
/// `TSUITATE_TAINT_KING_EMPTY=0` で従来挙動）。従来は候補マスの駒と無条件に
/// 入れ替えていたため、玉位置を直すたびに別の駒が根拠なく飛んでいた
fn taint_king_prefer_empty() -> bool {
    crate::config::current(|c| c.strategy.taint_king_prefer_empty)
}

/// 厳密粒子が全滅した決定で、評価本体（`expected`）を taint 粒子へ落とすか。
/// 既定は無効（従来挙動）。`TSUITATE_EVAL_TAINT_FALLBACK=1` で有効。
/// 凍結版はこの名前を知らないので候補側にだけ効く
fn eval_taint_fallback() -> bool {
    crate::config::current(|c| c.strategy.eval_taint_fallback)
}

/// 診断（issue #31 P0-5）が `evaluate` と同じ粒子プールの選び方を再現するための入口。
///
/// `eval_pool` は「厳密粒子が空 かつ taint があり かつ このノブが有効」のときだけ
/// taint に落ちる（既定は無効 = 厳密のみ）。`p_legal` のブレンドが
/// `particles_are_taint` で分岐するので、診断側も同じ判定を引く必要がある。
pub fn eval_taint_fallback_enabled() -> bool {
    eval_taint_fallback()
}

/// taint フォールバック中の**攻め項の倍率**（`TSUITATE_EVAL_TAINT_ATTACK`、
/// 既定 0 = 攻め項は供給しない）。
///
/// 発端は 2026-08-10 の実測: 素の taint フォールバックは材料・リスクの供給で
/// m067 を 3.10→7.65 点に改善する一方、**王手ボーナスが玉の信念位置への
/// 駒捨て王手**（7七桂成 8e7g+ = 支え付きマスへの桂捨て。ユーザー採点は
/// 全ブロックで 0〜2点）を 9〜10/20 まで押し上げ、採点済みシナリオでは
/// m100 7.70→3.10・m106/m110/m114/m116/m138 と一様に得点を落とした。
/// taint 粒子の玉位置ビリーフは攻め先の決定には足りない（メモリ
/// v2-bottleneck-is-king-belief、blind_attack_survive_w と同じ構図）ので、
/// 供給チャネルを材料・リスク側に限定する。1.0 で従来のフォールバック相当
fn eval_taint_attack_w() -> f64 {
    crate::config::current(|c| c.strategy.eval_taint_attack_w)
}

/// 終盤の紐減衰（`TSUITATE_LINK_ENDGAME_DAMPEN`、既定 0。
/// 0 で切り戻し）。
/// ブラインド決定でのみ `link_w /= 1 + w × endgame_push`。
///
/// quest31 終盤の 3三角成（2d3c+）固執が発端: 駒得ゼロなのに
/// `link≈2.4` + `promo` で玉筋の打ち（P*7c / G*7c）を押し下げる。
/// `link_w=0` なら G*7c（採点8）が首位に来るが、序中盤の予防的な紐
/// （v12）まで消すのは困る。厳密粒子が生きている決定では紐を触らず、
/// taint の玉位置に紐の働きが引っ張られるブラインド終盤だけ減衰する。
fn link_endgame_dampen() -> f64 {
    crate::config::current(|c| c.strategy.link_endgame_dampen)
}

/// ブラインド終盤の紐減衰の既定。0 = 無効（env 作業点は 40）。
/// 手数窓は quest31 の 3三角成帯向けで、汎化根拠が無いので既定オンにしない。
const LINK_ENDGAME_DAMPEN: f64 = 0.0;
/// env 有効時のみ。quest31 の 3三角成帯（127〜）向けの作業点。
const LINK_ENDGAME_DAMPEN_MIN_MOVE: u32 = 110;

/// 持ち駒の資産損（`TSUITATE_HAND_ASSET_W`、既定 `HAND_ASSET_W`。
/// 0 で切り戻し）。打つ手に `w × exchange_value(role)` を gain から引く
/// （仕事がある打ちは 0）。
///
/// **金・銀・桂**と、**敵陣以外への角・飛打ち**と**自陣への歩打ち**に限定する。
/// PR#1 全駒種版は kakudo の R*2d と lance/pawn-tether を巻き込みアリーナ
/// −7pt だった。quest31 の G*5e 濫発は金銀、N*4g / N*5d は桂、
/// B*1h 逃避は自陣、B*6f / B*8f / B*2f は中段の無目的角打ち、
/// P*4h / P*5h は自陣の歩打ち。敵陣の角飛打ち（B*3c / B*4a / B*3f で
/// 玉近接なら仕事あり）と敵陣の歩打ち（P*7c / P*4f）は対象外。
/// 仕事: 金は自玉 8 近傍、銀は玉頭2マス／敵陣かつ玉筋が読めるときの
/// 敵玉近接（金は玉候補そのもの・玉筋隣接を除く）／安い駒の裏付け当たり。
/// 打つ手だけ・王手中無効・粒子不要。
/// env 有効時の手数窓は quest31 終盤向け作業点（汎化根拠なし・既定オフ）。
/// 凍結版はこの名前を知らない。
fn hand_asset_w() -> f64 {
    crate::config::current(|c| c.strategy.hand_asset_w)
}

/// 金銀桂＋敵陣以外の大駒打ち＋自陣歩の無目的打ち課税。既定 0
/// （2026-08-12 に PR#1 の −7.4pt 容疑で確定。env 作業点は 1.0）。
const HAND_ASSET_W: f64 = 0.0;
/// env 有効時のみ。quest31 終盤向け作業点（汎化根拠なし）。
const HAND_ASSET_MIN_MOVE: u32 = 110;

/// 玉の既知脅威への接近減点（`TSUITATE_KING_KNOWN_APPROACH_W`、既定
/// `KING_KNOWN_APPROACH_W`。0 で切り戻し）。
/// 観測で位置が確定している敵駒マスへ近づく玉の手へ `w × Δcloseness` を
/// gain から引く。quest31-m099 の 6八玉（既知の 5六へ筋だけ寄る）が発端。
/// 脅威マスは `king_threat_evidence`（捕獲マス＋歴代の非歩打ち反則）。
/// **王手中も有効**（m099 は王手逃げの序列問題。CheckSolver は解消確率だけ）。
/// 粒子不要。そのマス自体を取る手は対象外。凍結版はこの名前を知らない。
fn king_known_approach_w() -> f64 {
    crate::config::current(|c| c.strategy.king_known_approach_w)
}

/// 玉の既知脅威接近。既定 0（2026-08-12 に PR#1 の −7.4pt 容疑で確定。
/// env 作業点は 1.0。PR#1 計測時は 2.0）。
const KING_KNOWN_APPROACH_W: f64 = 0.0;
/// env 有効時のみ。quest31-m099 向け作業点（汎化根拠なし）。
const KING_KNOWN_APPROACH_MIN_MOVE: u32 = 90;

/// 大駒成りの遠方ペナルティ（`TSUITATE_PROMOTE_FAR_W`、既定 0 = 無効。
/// quest31 コンボ計測時の作業点は 1.0）。
/// 角・飛の**実現する成り**で、着地が `deduce::opp_king_candidates` から
/// 遠いとき `w × max(0, d_min−1)` を gain から引く。
///
/// quest31 終盤の 3三角成（2d3c+）／3二角成（4a3b+）固執が発端: 採点 0 なのに
/// `promote_bias` + `promo_potential` で玉筋の打ち（P*7c / G*7c）を押し下げる。
/// `promo_king_prox` は将来の成りポテンシャル側で、**今成る手**の固定ボーナスを
/// 削らない。こちらは成る手そのものへの課税。玉候補 1 マス以内（隣接）だけ免税
/// （寄せの成りを壊さない）。免税を 2 にすると 4a3b+/4a6c+ が
/// d_min=2 で逃げ残る（combo_far_v1 実測で 4a3b+ が20残）。
/// 観測裏付けのある捕獲成は対象外。王手中無効・粒子不要。
/// 凍結版はこの名前を知らない。
fn promote_far_w() -> f64 {
    crate::config::current(|c| c.strategy.promote_far_w)
}

/// 大駒成りの遠方ペナルティ。既定 0（env 作業点は 2.5。PR#1 計測時は 1.0）。
const PROMOTE_FAR_W: f64 = 0.0;
/// m081/m083 の 4a3b+ 以降を残し、序盤の大駒成り（材料として正しい手）は
/// 課税しない。
const PROMOTE_FAR_MIN_MOVE: u32 = 80;

/// 玉筋の歩前進（`TSUITATE_KING_FILE_PAWN_W`、既定 0。env 作業点は 1.2）。
/// **敵陣**の歩（前進・成り・打ち）が、玉候補筋の中央値から距離 ≤2 のとき
/// `w / (1+d_file)` を gain に足す（P*7c / 7c7b+ 型）。
/// 自陣・中段の歩突き（9六歩・8六歩・7六歩・4f4g+）は加点しない。
/// **盤上の入口段**（先手3段・後手7段 = 5f5g+ 型）も加点しない。打ちは残す。
/// 王手中無効・粒子不要。凍結版はこの名前を知らない。
fn king_file_pawn_w() -> f64 {
    crate::config::current(|c| c.strategy.king_file_pawn_w)
}

const KING_FILE_PAWN_W: f64 = 0.0;

/// 中段の玉筋歩前進（`TSUITATE_KING_FILE_PAWN_MID_W`、既定
/// `KING_FILE_PAWN_MID_W`。0 で切り戻し）。手数 80..=86、自陣段からの
/// 前進1マスが玉候補筋の**中央値以外**（距離 1..=2）のとき `w` を足す。
///
/// 発端は quest31-m081/m083 の 7六歩・9六歩（8点）が 4a3b+（0点）の下に
/// 居ること。敵陣限定の `king_file_pawn_w` は 7g7f を加点しない。
/// 中央値そのもの（実測で 8六歩）は m081 で未収載の 4 点、m097 以降では
/// 7〜9 点の手を置き換えるので除外する。手数上限 86 は m087 の G*6c（8点）
/// を 8g8f の仮4点へ逃がさないため。玉筋が読めていないときは 0。
/// 王手中無効・粒子不要。凍結版はこの名前を知らない。
fn king_file_pawn_mid_w() -> f64 {
    crate::config::current(|c| c.strategy.king_file_pawn_mid_w)
}

const KING_FILE_PAWN_MID_W: f64 = 0.0;
const KING_FILE_PAWN_MID_MIN_MOVE: u32 = 80;
const KING_FILE_PAWN_MID_MAX_MOVE: u32 = 86;

/// 終盤の玉の非捕獲逃げ（`TSUITATE_KING_ENDGAME_FLEE_W`、既定
/// `KING_ENDGAME_FLEE_W`。0 で切り戻し）。手数 125 以降の玉の移動で、
/// 着地が観測裏付けの占有マスでなければ `w` を gain から引く。
///
/// 発端は quest31-m140 の 8a9b / 8a8b / 8a7a（1〜2点）が 8c8b（8点）の上に
/// 居ること。w=3 では seed0 で玉逃げが依然 1〜3 位だったので 8.0。
/// 取る手は残す（recap-dragon）。王手中も空マス逃げは課税する
/// （m140 は打ち込み王手で 8b が裏付けに乗らず、非王手ゲートだと
/// 8a9b が CheckSolver の p_legal で首位のまま）。king-evade は
/// 手数 125 未満なので手数ゲートで守る。粒子不要。凍結版はこの名前を知らない。
fn king_endgame_flee_w() -> f64 {
    crate::config::current(|c| c.strategy.king_endgame_flee_w)
}

const KING_ENDGAME_FLEE_W: f64 = 0.0;
const KING_ENDGAME_FLEE_MIN_MOVE: u32 = 125;

/// 終盤の金が自玉へ隣接する盤上移動（`TSUITATE_GOLD_JOIN_KING_W`、既定
/// `GOLD_JOIN_KING_W`。0 で切り戻し）。手数 125 以降、**王手中**に金が
/// 自玉の 8 近傍へ寄る盤上移動へ `w` を足す（既に隣接は 0）。
///
/// 発端は quest31-m140 の 8c8b（8点）vs 8a9b（2点）。打ち込み王手は
/// 観測されないので CheckSolver の仮説に 8b が乗らず、8c8b の p_legal が
/// 薄い（kakutori 同型）。gain 内（p_legal 割引の内側）では w=4 でも
/// 8a9b を逆転できなかったので **combine_score の外側**（`foul_probe` と同じ）
/// に足す。盤上の金移動なので打ち反則スパムにはならない。非王手の寄りは
/// m138 の未採点 8c8b へ逃げるので対象外。粒子不要。凍結版はこの名前を知らない。
fn gold_join_king_w() -> f64 {
    crate::config::current(|c| c.strategy.gold_join_king_w)
}

const GOLD_JOIN_KING_W: f64 = 0.0;
const GOLD_JOIN_KING_MIN_MOVE: u32 = 125;

/// 終盤、自玉に既に隣接している金が玉筋へ動く手（`TSUITATE_GOLD_KING_FILE_W`、
/// 既定 `GOLD_KING_FILE_W`。0 で切り戻し）。手数 125 以降、**非王手**で
/// 金が玉の 8 近傍から玉筋（同じ file）の **距離 2** へ動く手へ `w` を足す。
///
/// 発端は quest31-m130 の 7b8c（6点）vs 5f5g+（2点）。金は 7b で既に
/// 玉 8a に隣接しているので `gold_join` は 0。距離 1（7b8b）まで入れると
/// 未採点の 8b へ逃げる。歩成り課税は金銀手持ちゲートのため、盤上の金
/// だけでは 5f5g+ を沈められない。王手中は `gold_join` / CheckSolver の領分。
/// 粒子不要。凍結版はこの名前を知らない。
fn gold_king_file_w() -> f64 {
    crate::config::current(|c| c.strategy.gold_king_file_w)
}

const GOLD_KING_FILE_W: f64 = 0.0;
const GOLD_KING_FILE_MIN_MOVE: u32 = 125;

/// 終盤の桂の敵陣成り課税（`TSUITATE_KNIGHT_LATE_PROMO_W`、既定
/// `KNIGHT_LATE_PROMO_W`。0 で切り戻し）。手数 100..=136、桂が敵陣へ
/// **成って**入る盤上移動へ `w` を引く。不成は手数
/// `KNIGHT_LATE_NONPROMO_MIN_MOVE` 以降だけ `KNIGHT_LATE_NONPROMO_SCALE`
/// 倍（既定 0.5 = 3.0 点）。
///
/// 発端は quest31-m100 の 8e7g+（2点）が 5f5g+（8点）の上に居ること。
/// 不成まで 100 手から課税すると 8e7g 不成（5点）が沈み、8c8d（0点）へ
/// 押し出される（`c8e80a1` の 0.5 倍で m100 が 6.20→0.60）。不成を免税に
/// すると m116/m118/m120 で 8e7g 不成（2点）が 5f5g+（6点）と混ざって
/// 4.40 に落ちる（`c863904`）。不成税は 110 手以降に限る。
/// 手数 137 以降の 8e7g+（m138 = 4点）は `knight_endgame_promo_w` の加点側。
/// 序中盤の 4d3b+ は min で守る。王手中無効・粒子不要。凍結版はこの名前を知らない。
fn knight_late_promo_w() -> f64 {
    crate::config::current(|c| c.strategy.knight_late_promo_w)
}

const KNIGHT_LATE_PROMO_W: f64 = 0.0;
const KNIGHT_LATE_PROMO_MIN_MOVE: u32 = 100;
const KNIGHT_LATE_PROMO_MAX_MOVE: u32 = 136;
/// 不成の敵陣進入は成り税のこの倍率。100 手からの課税は m100 を 0 点混在へ
/// 押し出したので、`KNIGHT_LATE_NONPROMO_MIN_MOVE` 以降に限る。
const KNIGHT_LATE_NONPROMO_SCALE: f64 = 0.5;
const KNIGHT_LATE_NONPROMO_MIN_MOVE: u32 = 110;

/// 終盤の桂の敵陣成り加点（`TSUITATE_KNIGHT_ENDGAME_PROMO_W`、既定
/// `KNIGHT_ENDGAME_PROMO_W`。0 で切り戻し）。手数 137 以降、桂が敵陣へ
/// **成って**入る盤上移動へ `w` を足す（`knight_late_promo` の税の逆）。
///
/// 発端は quest31-m138 の 8e7g+（4点）。歩 force で 5f5g+（0点）を沈めると
/// P*4g へ逃げるので、採点済みの受け皿を押し上げる。不成は加点しない
/// （eval に無い 8e7g 不成へ逃げるため）。王手中無効・粒子不要。
fn knight_endgame_promo_w() -> f64 {
    crate::config::current(|c| c.strategy.knight_endgame_promo_w)
}

const KNIGHT_ENDGAME_PROMO_W: f64 = 0.0;
const KNIGHT_ENDGAME_PROMO_MIN_MOVE: u32 = 137;

/// 終盤、自陣の桂が中段へ出る手（`TSUITATE_KNIGHT_CAMP_EXIT_W`、既定
/// `KNIGHT_CAMP_EXIT_W`。0 で切り戻し）。手数 120 以降、自陣の桂が
/// 自陣でも敵陣でもないマスへ動く手へ `w` を足す。ただし**初期玉筋
/// （5筋）へ近づく手は 0**（m124 の 6b5d = 0点）。
///
/// 発端は quest31-m124 の 6b7d（8点）vs 5f5g+（5点）。玉候補の median へ
/// 近づくフィルタは、信念が初期5筋に残っていると 6b5d を押し上げて
/// 6.40→1.60 に壊した（`da1a869`）。敵陣進入は `knight_late_promo` の
/// 課税側。王手中無効・粒子不要。
fn knight_camp_exit_w() -> f64 {
    crate::config::current(|c| c.strategy.knight_camp_exit_w)
}

const KNIGHT_CAMP_EXIT_W: f64 = 0.0;
const KNIGHT_CAMP_EXIT_MIN_MOVE: u32 = 120;

/// 終盤の銀が自陣から出る手（`TSUITATE_SILVER_CAMP_EXIT_W`、既定
/// `SILVER_CAMP_EXIT_W`。0 で切り戻し）。手数 100 以降、自陣の銀が
/// 自陣の外へ動く手へ `w` を足す。
///
/// 発端は quest31-m106 の 7c6d（後手・自陣3段目→中段、8点）vs 8e7g+（2点）。
/// 桂成課税と対で、吊るされた銀を進める側を押し上げる。王手中無効・粒子不要。
fn silver_camp_exit_w() -> f64 {
    crate::config::current(|c| c.strategy.silver_camp_exit_w)
}

const SILVER_CAMP_EXIT_W: f64 = 0.0;
const SILVER_CAMP_EXIT_MIN_MOVE: u32 = 100;

/// 玉筋の金打ち（`TSUITATE_KING_FILE_GOLD_W`、既定 0）。
/// 敵陣かつ玉候補筋の中央値から距離 ≤2 の金打ちへ `w / (1+d_file)` を
/// gain に足す。m101 で G*8c / G*8a（0点）を玉筋として押し上げ
/// 2.40→0.80 に回帰したため既定オフ。env から試せる。銀は対象外。
fn king_file_gold_w() -> f64 {
    crate::config::current(|c| c.strategy.king_file_gold_w)
}

const KING_FILE_GOLD_W: f64 = 0.0;

/// と金の玉筋逸れ（`TSUITATE_TOKIN_FILE_DRIFT_W`、既定
/// `TOKIN_FILE_DRIFT_W`。0 で切り戻し）。敵陣のと金が、裏付けの無い空きマスへ
/// 動く手へ `w × exchange_value(Tokin)` を gain から引く。
///
/// 発端は quest31-m046 の 4g4h / 4g5h（採点 2）vs 3h4i 不成（10）。
/// 接近免税は 4g5h が同点で繰り上がるだけだったので外す。残す免税は
/// 玉筋滞在・大駒の筋空け・相手最奥段（2a1a の香取り）・裏付け捕獲。
/// 王手中無効・粒子不要。
fn tokin_file_drift_w() -> f64 {
    crate::config::current(|c| c.strategy.tokin_file_drift_w)
}

/// と金の玉筋逸れの既定。接近免税なしでも 4g5h / P*2b へ逃げるだけで、
/// 受け皿なしのフル suite は 5.503→5.460。コードは残して既定オフ。
const TOKIN_FILE_DRIFT_W: f64 = 0.0;

/// 終盤の歩成り課税（`TSUITATE_PAWN_OFFFILE_W`、既定 `PAWN_OFFFILE_W`。
/// 0 で切り戻し）。金または銀を持っているとき、中段から敵陣へ入る
/// **歩成り**へ `w` を gain から引き、裏付けの無い捕獲期待値もキャンセルする。
/// 発火中は `king_file_pawn_w` を掛けない（+1.2 が税を打ち消す）。
///
/// 発端は quest31 終盤の 5f5g+ 固執（m098/m126/m130）。手数 125 以降に
/// 限るので m110 の 5f5g+（7点）と m116/m120（6点）は対象外。
/// **既に敵陣にいる歩の前進**（m127 の 7c7b+ = 9点）は課税しない。
/// 不成・手持ちゲート無しの 5.0 は m134/m136 の妥当な 5f5g+（4〜5点）まで
/// B*7g へ沈めたので、成り＋金銀手持ちに戻す。手数 137 以降だけ
/// 手持ちゲートを外す（m138 の 5f5g+ = 0 点。金は盤上にいる）。
/// 歩打ちは垂れ歩保護のため除外。王手中無効・粒子不要。
fn pawn_offfile_w() -> f64 {
    crate::config::current(|c| c.strategy.pawn_offfile_w)
}

/// 終盤の歩成り課税の既定。手数 125 以降・金銀手持ち・成りのみ。
/// 手数 `PAWN_OFFFILE_FORCE_MIN_MOVE` 以降は手持ちゲートを外す
/// （m138 の 5f5g+ = 0 点。金は盤上 8c にいるので手持ちゲートだと発火しない）。
const PAWN_OFFFILE_W: f64 = 0.0;
const PAWN_OFFFILE_MIN_MOVE: u32 = 125;
const PAWN_OFFFILE_FORCE_MIN_MOVE: u32 = 137;

/// 遠方の大駒成り捕獲の幻の駒得キャンセル（`TSUITATE_FAR_MAJOR_PROMO_CAPTURE_W`、
/// 既定 `FAR_MAJOR_PROMO_CAPTURE_W`。0 で切り戻し）。
/// 角・飛の**成る手**が、観測裏付けの無いマスへ入り、かつ
/// `promote_far_amount > 0`（玉から遠い／近づかない）のとき、
/// 粒子の捕獲期待値を `w` 倍だけ gain から引く（w=1 で全額キャンセル）。
///
/// 発端は quest31 の 4a3b+ / 2d3c+（採点 0 の幻の角成り込み）。
/// `unbacked_gs_capture_w` を大駒まで広げると正しい捕獲まで殺し
/// （フル suite 5.326）、信念ネット占有も敵陣では「居る」と見なす。
/// こちらは **成る × 遠い × 裏付け無し** の交差だけなので、
/// recap-dragon（裏付け）と玉隣の成り捕獲（amount=0）は残る。
/// 王手中無効・粒子不要（捕獲期待値は evaluate が出した値を外側で削る）。
fn far_major_promo_capture_w() -> f64 {
    crate::config::current(|c| c.strategy.far_major_promo_capture_w)
}

/// 遠方の大駒成り捕獲キャンセルの既定。複合フル suite で 4a3b+ は減らず
/// 他項の副作用が勝った（5.503→5.281）。コードは残して既定オフ。
const FAR_MAJOR_PROMO_CAPTURE_W: f64 = 0.0;

/// 自陣の金銀桂の空きマス移動課税（`TSUITATE_OWN_CAMP_IDLE_W`、既定
/// `OWN_CAMP_IDLE_W`。0 で切り戻し）。自陣への非捕獲移動へ
/// `w × exchange_value` を gain から引く。
///
/// 発端は quest31-m046 の 7a7b。幾何だけでは m027 の 3i3h（採点 7）と
/// 区別できないので **手数 40 以降だけ** 掛ける（3i3h は 27 手目）。
/// 王手中は無効（m086 の 7a7b は 8 点の受け）。
fn own_camp_idle_w() -> f64 {
    crate::config::current(|c| c.strategy.own_camp_idle_w)
}

const OWN_CAMP_IDLE_W: f64 = 0.0;
const OWN_CAMP_IDLE_MIN_MOVE: u32 = 40;

/// 角・馬の非前進空きマス移動（`TSUITATE_BISHOP_RETREAT_W`、既定
/// `BISHOP_RETREAT_W`。0 で切り戻し）。敵陣に入らず前進もしない
/// 盤上移動へ `w × exchange_value` を gain から引く。
///
/// 発端は quest31 の 2d3e（3五角、採点 0〜2 が m081/m083/m085 で首位）。
/// 先手なら rank が増える＝敵陣から遠ざかる。kakudo の 7i2d は rank が
/// 減る前進なので免税。裏付け捕獲は免税。打ちは HAND_ASSET の領分。
/// 0.25 は m055〜m063 の 3c5c（0点の龍横滑り）を復活させたので 0.5。
/// 王手中無効・粒子不要。
/// **手数 `BISHOP_RETREAT_MIN_MOVE` 以降**（m055 の 3c5c は残し、
/// 序盤の角の横移動はアリーナの攻め手段）。
/// 凍結版はこの名前を知らない。
fn bishop_retreat_w() -> f64 {
    crate::config::current(|c| c.strategy.bishop_retreat_w)
}

const BISHOP_RETREAT_W: f64 = 0.0;
/// m055（55手）の 3c5c を残す下限。
const BISHOP_RETREAT_MIN_MOVE: u32 = 50;

/// 終盤の敵陣成銀の筋替え（`TSUITATE_ENDGAME_CAMP_GENERAL_W`、既定
/// `ENDGAME_CAMP_GENERAL_W`。0 で切り戻し）。手数 125 以降、敵陣にいる
/// 成銀が敵陣へ**玉筋（中央値）へ近づく筋替え**をする手へ `w` を足す。
///
/// 発端は quest31-m145 の 7c8b（成銀、10点）。方向無しの筋替えは
/// 7c6b（逆方向・未収載の仮4点）へ逃避した。打ちは HAND_ASSET の領分。
/// 王手中無効・粒子不要。
fn endgame_camp_general_w() -> f64 {
    crate::config::current(|c| c.strategy.endgame_camp_general_w)
}

const ENDGAME_CAMP_GENERAL_W: f64 = 0.0;
const ENDGAME_CAMP_GENERAL_MIN_MOVE: u32 = 125;

/// 裏付け無しの敵陣進入課税（`TSUITATE_UNBACKED_CAMP_W`、既定
/// `UNBACKED_CAMP_W`）。歩・香・桂・玉以外が、観測裏付けの無い敵陣マスへ
/// 入る手へ `w × exchange_value(着手駒)` を gain から引く。
///
/// 発端は quest31-m021 の 4一と（幻の金）と 4a3b+ / 2d3c+（幻の駒得で
/// 大駒が敵陣へ成り込む）。`capture_bet_var_w` は p_hit≈1 で消え、
/// `material_degen_q0` は粒子質量で縮めるが信念が自信を持って間違うと
/// 残る。こちらは**観測の裏付けが無い敵陣マス**という安全方向だけの静的税。
/// 裏付けマス（取られた/非歩打ち反則）への取り返しは免税。
/// 王手中無効・粒子不要。
/// **手数 `UNBACKED_CAMP_MIN_MOVE` 以降**（と金は対象外なので m021 は
/// KING_ADJ 側。大駒の幻成り込みは 80 手以降だけ課税）。
/// 凍結版はこの名前を知らない。
fn unbacked_camp_w() -> f64 {
    crate::config::current(|c| c.strategy.unbacked_camp_w)
}

const UNBACKED_CAMP_W: f64 = 0.0;
/// と金は対象外なので m021 は KING_ADJ 側。こちらは 4a3b+ / 2d3c+ の
/// 大駒成り込み用で、序中盤から掛けるとアリーナの正しい敵陣進入まで殺す。
const UNBACKED_CAMP_MIN_MOVE: u32 = 80;

/// 金銀・大駒の裏付け無し捕獲を gain から削る係数（`TSUITATE_UNBACKED_GS_CAPTURE_W`、
/// 既定 `UNBACKED_GS_CAPTURE_W`。0 で切り戻し）。
///
/// `material_degen_q0` は粒子質量が厚いと縮まない。観測裏付けの無いマスで
/// 「駒が居る」と信じた捕獲（quest31-m081 の 6c6b 幻の金、4a3b+ 幻の駒得）は
/// p_hit≈1 で `capture_bet_var` も消える。こちらは**金銀と大駒の盤上移動**
/// の期待駒得を `w` 倍だけ引く（w=1 で幻の駒得を全額キャンセル）。
/// 打ちは捕獲にならない。王手中無効（kakutori）。裏付けマスは満額残す
/// （recap-dragon）。と金・歩・桂・香は対象外（2c3b の銀取り・4七歩成）。
/// 凍結版はこの名前を知らない。
fn unbacked_gs_capture_w() -> f64 {
    crate::config::current(|c| c.strategy.unbacked_gs_capture_w)
}

/// 金銀の裏付け無し捕獲をキャンセルする既定（2026-08-13。m081 の 6c6b）。
/// 大駒まで広げると 5試行フル suite が 5.326 まで落ちたため金銀だけ。
const UNBACKED_GS_CAPTURE_W: f64 = 0.0;
/// m081 の 6c6b は残し、序中盤の正しい金銀捕獲まで殺さない。
const UNBACKED_GS_CAPTURE_MIN_MOVE: u32 = 80;

/// 信念ネット占有による裏付け無し捕獲の縮小（`TSUITATE_BELIEF_OCC_CAP_W`、
/// 既定 `BELIEF_OCC_CAP_W`。0 で切り戻し）。
///
/// `material_degen_q0` は粒子質量が厚いと縮まない。生存粒子が全員で空きマス
/// に駒を置くと相対重みは動かず、p_hit≈1 の幻の駒得が残る（m067: 厳密9個が
/// 4一に飛車 85.9%。taint 側は空 95.8% で正しかった）。信念ネットの占有は
/// 粒子より当たる（対数損失 0.62→0.38）ので、**観測裏付けの無い捕獲**の
/// 期待駒得をネット占有へ安全方向だけ寄せる。
///
/// 寄せ方: ネット占有 p_occ が盤面平均事前（0.25）を下回るときだけ、
/// mix = w × (1 − p_occ/0.25) で粒子の p_hit を p_occ へ混ぜ、
/// 差分ぶんの期待駒得を引く。
///
/// **既定 0**（2026-08-16。対 v13 12局・seed 20260816 で発火 32%・駒得
/// 876/962・2勝9敗1分。同条件でオフに戻すと 8勝4敗・駒得 842/829。
/// ネットが空きと見たマスへの実在大駒捕獲まで沈み、対 bot の勝率を落とす）。
/// 手数ゲートは付けない。quest31 の 2d3c+ 向けに env から試せる。
/// 全駒種版は 5.379、自信過剰ギャップは 5.351 で不採用。金銀は
/// `unbacked_gs_capture_w`。王手中無効・裏付けマスは満額。
/// 凍結版はこの名前を知らない。
fn belief_occ_cap_w() -> f64 {
    crate::config::current(|c| c.strategy.belief_occ_cap_w)
}

const BELIEF_OCC_CAP_W: f64 = 0.0;
/// 信念ネットが「空き寄り」と見なす占有の上界。これ以上なら粒子の
/// p_hit を上書きしない（盤面平均 prior_occ ≈ 0.21 の少し上）。
const BELIEF_OCC_EMPTY_PRIOR: f64 = 0.25;

/// 裏付け無し捕獲の期待駒得を、信念ネット占有が空き寄りのときだけ縮める。
/// `capture_ev` = 粒子の E[捕獲価値]、`p_hit` = 粒子の占有率、`p_occ` = ネット。
fn belief_occ_cap_shrink(capture_ev: f64, p_hit: f64, p_occ: f64, w: f64) -> f64 {
    if w <= 0.0 || capture_ev <= 0.0 || p_hit <= 1e-9 {
        return 0.0;
    }
    let empty = ((BELIEF_OCC_EMPTY_PRIOR - p_occ) / BELIEF_OCC_EMPTY_PRIOR).clamp(0.0, 1.0);
    if empty <= 0.0 {
        return 0.0;
    }
    let mix = (w * empty).clamp(0.0, 1.0);
    let effective_p = (1.0 - mix) * p_hit + mix * p_occ.min(p_hit);
    capture_ev * (1.0 - effective_p / p_hit).max(0.0)
}

/// 相手の初期金位置への金銀の当たり（`TSUITATE_HOME_GOLD_ATTACK_W`、既定
/// `HOME_GOLD_ATTACK_W`）。capture≈0 のときだけ足す。
///
/// 発端は quest31-m046 の 3h4i 不成（10点）。捕獲信念がある手と
/// move_number<44（m042 の 4f4g+）は加点しない。
///
/// w=3 の一律加点は m042 と m054 で回帰。裏付け捕獲ゲートは m054 の 4g が
/// capture≈0 のとき外れ、3h4i が 4/5 で選ばれた（5.426、m054 10→2）。
/// 同筋の別マスへ動ける駒（3h4g が候補に載っている）なら加点しない。
/// 3h2i は 2 筋なので m046 の受け皿は残る。
fn home_gold_attack_w() -> f64 {
    crate::config::current(|c| c.strategy.home_gold_attack_w)
}

const HOME_GOLD_ATTACK_W: f64 = 0.0;
const HOME_GOLD_MIN_MOVE: u32 = 44;

/// と金が玉筋へ寄る手の加点（`TSUITATE_TOKIN_APPROACH_W`、既定 0）。
/// 敵陣のと金が、玉候補筋の中央値への筋距離を縮める手へ `w / (1+d_to)` を足す。
///
/// 発端は quest31-m021 の 2c3b。実測で玉筋が読めていないと中央値が端に寄り
/// 2c1c（1点）を 5/5 で選んだため既定オフ。`king_files_focused` ゲート付き。
fn tokin_approach_w() -> f64 {
    crate::config::current(|c| c.strategy.tokin_approach_w)
}

const TOKIN_APPROACH_W: f64 = 0.0;

/// 玉隣接への高い駒の無支え進入（`TSUITATE_KING_ADJ_HEAVY_W`、既定
/// `KING_ADJ_HEAVY_W`。0 で切り戻し）。
/// 観測裏付けの無い玉候補 8 近傍へ、歩香桂玉以外の駒が**盤上から**入る手へ
/// `w × exchange_value` を gain から引く。
///
/// 発端は quest31-m021 の 4一と（3a4a）。unbacked_camp のと金課税は
/// 本命の 2c3b（敵陣のと金移動）まで同じ税が乗って相対差が消える。
/// こちらは玉隣だけなので 4a（5a の 8 近傍）は沈み、3b（チェビシェフ 2）は
/// 免税。筋外れのと金（2c1c）は課税するが、相手最奥段（2a1a の香取り）は
/// 免税。打ちは HAND_ASSET / drop_probe の領分（S*4b を巻き込まない）。
/// 歩は対象外（4七歩成を巻き込まない = 旧 `king_adj_entry_w` の失敗）。
/// 玉筋が読めていない序盤では発火しない。王手中無効・粒子不要。
/// **手数 `KING_ADJ_HEAVY_MIN_MOVE` 以降**（m021 の 3a4a は残す）。
/// 凍結版はこの名前を知らない。
fn king_adj_heavy_w() -> f64 {
    crate::config::current(|c| c.strategy.king_adj_heavy_w)
}

/// 玉隣の高い駒進入課税の既定（2026-08-13。m021 の 3a4a 対策）。
/// 0.5 では suite で 3a4a が 5/5 残ったため 1.5（tokin 3.5×1.5=5.25）
const KING_ADJ_HEAVY_W: f64 = 0.0;
/// m021（21手）の 3a4a を残す下限。
const KING_ADJ_HEAVY_MIN_MOVE: u32 = 20;

/// 桂銀香の任意成り課税（`TSUITATE_OWN_CAMP_MINOR_PROMO_W`、既定
/// `OWN_CAMP_MINOR_PROMO_W`。0 で切り戻し）。
/// 桂・銀・香が成る手へ `w` を gain から引く。不成は advance_bias が
/// 既に `promote_bias` を付けているので加点しない（二重計上を避ける）。
///
/// 発端は quest31-m046 の 3h4i+（後手・2 点）vs 3h4i（10 点）。成ると金が
/// 取れなかったときに王手が掛かって銀を失う。GEN_NONPROMOTE が無効なら
/// 不成が無いので発火しない。強制成り（行き所のない駒）は対象外。
/// 王手中無効・粒子不要。凍結版はこの名前を知らない。
fn own_camp_minor_promo_w() -> f64 {
    crate::config::current(|c| c.strategy.own_camp_minor_promo_w)
}

const OWN_CAMP_MINOR_PROMO_W: f64 = 0.0;

/// 玉候補への接近ボーナス（`TSUITATE_KING_CAND_ATTACK_W`、既定 **4.0**、
/// 0 で従来挙動 = 切り戻しノブ）。
///
/// `deduce::opp_king_candidates`（**健全**＝真の玉を絶対に落とさない候補集合）が
/// 鋭いとき、着地マスの近接度 `Σ_k 0.5^cheb(to,k)`（盤の最大で正規化）を
/// `w × 近接度 × 安さ係数` で gain へ加点する（近接度は `king_cand_prox_map`、
/// 安さ係数は歩を 1.0 に正規化した `2/(1+交換価値)`。**持ち込んだ駒**で数え、
/// 成る手も成り後の駒種にはしない。成り後だと 7c7b+ がと金扱いになり、
/// 候補の隣への垂れ歩（quest31-m121 の P*5b、採点0）に負ける）。
///
/// 発端は quest31 終盤（plies 67〜148）の実測: 王手宣言のおかげで **先手側の
/// 玉候補は 6〜17 マス**まで絞れていて真の玉（8二/8一）を必ず含むのに、
/// この情報を使う項は `promote_far_w`（成りへの課税）しか無い。評価は
/// 終盤ブラインドで `link`（紐）が支配し（実測: gain 3.7 のうち link 2.8）、
/// 玉から一番遠い 3二角成／3三角成が首位を占め続けていた。採点済み eval の
/// 高得点手は P*7c / G*7c / 7c7b+ / 7d7c+ と**玉の隣接圏に集中**する。
///
/// 粒子不要・ノイズゼロ（自分の観測だけで決まる）。**候補集合が鋭いときだけ**
/// 発火するので中盤（|cands| が 30〜50）では素通りする。王手中は無効。
/// v13 以前の凍結版はこの名前を知らない（v14 は読む）。
///
/// 既定 4.0 は 2026-08-13 の作業点（`promote_far_w` を外したときの作業点。
/// PR#1 の hand_asset / link_endgame / king_known_approach /
/// material_degen は既定 0 のまま）。安全解消ゲート既定 on は
/// arena v13 48.1%・suite 4.614・kakutori 2/20 で不合格。
/// **既定オンは deduce が鋭いときだけ発火する接近ボーナスに限る**
/// （`king_belief_prox_w` は既定 0。ネット接近は deduce が鈍い大半の
/// 局面で発火し、5.9 構成のアリーナ合算 54.0% vs 対照 56.3% に入っていた）。
/// 2026-08-19 に estimator_v14 として凍結（suite 5.197、vs v13 200局 60.0%）。
fn king_cand_attack_w() -> f64 {
    crate::config::current(|c| c.strategy.king_cand_attack_w)
}

/// 玉候補接近ボーナスの既定（2026-08-13 作業点）。0 で切り戻し
const KING_CAND_ATTACK_W: f64 = 4.0;

/// 玉候補マスへ利く手への追加加点（`TSUITATE_KING_CAND_CHECK_W`、既定 **1.0**、
/// 0 で無効 = 切り戻しノブ）。
/// `king_cand_attack_w`（着地マスの近接度）と同じゲート・同じ安さ係数で、
/// 「その駒が実際に玉候補へ利くか」を `blind_king_attack` で測って加点する
/// （分布は候補集合上の一様分布 = 粒子不要）
fn king_cand_check_w() -> f64 {
    crate::config::current(|c| c.strategy.king_cand_check_w)
}

/// 玉候補へ利く手の加点既定（2026-08-13 作業点）
const KING_CAND_CHECK_W: f64 = 1.0;

/// `king_cand_attack_w` が発火する玉候補集合の上限（`TSUITATE_KING_CAND_ATTACK_GATE`、
/// 既定 20）。これを超えると候補が盤の半分に散っていて近接度が
/// 「敵陣へ前進」以上の情報を持たない
fn king_cand_attack_gate() -> usize {
    crate::config::current(|c| c.strategy.king_cand_attack_gate)
}

/// 着手後に着地マスを守っている自駒の枚数（着手駒自身は自分のマスを守れないので
/// 移動元から着地へ利いていたぶんを引く。`blind_attack_survive_w` と同じ規約）
fn landing_def(view: &PlayerView, mv: &ShogiMove, own_attack: &[u8; 81]) -> f64 {
    let to = match *mv {
        ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
    };
    let mut def = f64::from(own_attack[crate::belief_features::sq_index(to)]);
    if let ShogiMove::Board { from, .. } = *mv {
        if own_defends_from(view, from, to) {
            def -= 1.0;
        }
    }
    def.max(0.0)
}

/// 玉候補接近ボーナスの**支えゲートの強さ**（`TSUITATE_LANDING_SUPPORT_W`、
/// 既定 **0.7**、0 でゲート無し = 切り戻しノブ）。
/// 係数は `(1−w) + w×min(着手後の支え枚数,2)/2` で、
/// w=1 なら支え0枚の接近は加点ゼロ、w=0.5 なら半額。採点済み eval の局面内
/// 回帰では支え枚数が +0.60点あるが、**独立した加算項にすると玉から遠い
/// 「支えのある無意味なマス」まで加点する**ので接近側へ掛ける
fn landing_support_w() -> f64 {
    crate::config::current(|c| c.strategy.landing_support_w)
}

/// 接近ボーナスの支えゲート既定（2026-08-13 作業点。m145 の裸の金打ち対策）
const LANDING_SUPPORT_W: f64 = 0.7;

/// **玉位置ネット**による接近ボーナス（`TSUITATE_KING_BELIEF_PROX_W`、
/// 既定 **0**、env 作業点は 5.0）。
/// `king_cand_attack_w` の**王手を掛けていない側用の対**。
///
/// `deduce::opp_king_candidates` は「自分が王手を宣言した履歴」から絞るので、
/// 王手をあまり掛けられていない側では 35〜55 マスに散ってゲートを通らない
/// （quest_20260731 の実測: 先手は 67手目以降ずっと 6〜17 マスなのに後手は
/// 全局面で 35〜55）。残る失点上位 `5f5g+` / `8a7c` / `4g4h` はすべて後手側の
/// 決定で、接近ボーナスが原理的に発火していないのが上限になっている。
///
/// そこで **deduce が鈍いときだけ**、玉位置ネット（`king_belief_nn`）の分布を
/// 使って同じ近接度を作る。粒子版は全水準で負けた（情報が無かったのではなく
/// 情報源が違った）。分布が散っているとき（実効サポート数 1/Σp² が
/// `king_cand_attack_gate` 超）は使わない。
/// deduce が鋭いときは `king_cand_attack_w` の領分なので発火しない（二重計上
/// を避ける。ただしこの排他は `king_cand_attack_w` / `king_cand_check_w` の
/// どちらかがオンで deduce 側のマップが作られたときだけ成立し、本ノブ単独で
/// 有効化すると deduce が鋭くてもネット経路が発火する）。王手中は無効。
///
/// **既定 0**: ネット接近は deduce が鈍い＝対局の大半で発火する。
/// 5.9 構成（本ノブ w=5 込み）はシナリオ 5.901 でもアリーナ合算 54.0% vs
/// 対照 56.3% で不採用。安全解消ゲート不合格のあと接近束ごと既定オンに
/// していたが、アリーナを下げる側の項を既定に残さない。
fn king_belief_prox_w() -> f64 {
    crate::config::current(|c| c.strategy.king_belief_prox_w)
}

/// 玉位置ネット接近ボーナスの既定。0 = 無効（env 作業点は 5.0。粒子版は不発）
const KING_BELIEF_PROX_W: f64 = 0.0;

/// `promote_far_w` の課税を**全駒種**へ広げるか（`TSUITATE_PROMOTE_FAR_ALL=1`、
/// 既定 0 = 角・飛だけ）。課税額は着手駒の交換価値で頭打ちにする
fn promote_far_all() -> bool {
    crate::config::current(|c| c.strategy.promote_far_all)
}

/// `king_cand_attack_w` を**厳密粒子ゼロの決定だけ**に限るか
/// （`TSUITATE_KING_CAND_ATTACK_BLIND=1`、既定 0 = 常に発火）。
/// 厳密粒子が生きている決定では駒得・攻め圧力が情報を持つので、
/// 玉候補の幾何だけで引っ張る必要は薄い、という仮説の検証用
fn king_cand_attack_blind_only() -> bool {
    crate::config::current(|c| c.strategy.king_cand_attack_blind_only)
}

/// 成って王手する手の露見ペナルティ（`TSUITATE_PROMOTE_CHECK_REVEAL_W`、
/// 既定 0。env 作業点は 1.2）。歩・角・飛の**成る王手**
/// は宣言で位置が露見し、安い駒でも回収されやすい（quest31-m095 の
/// 7三歩成 vs 不成が発端。ユーザー指導: 歩・飛・角の不成価値は「成ると
/// 王手が増え宣言で露見するのを避ける」ついたて固有）。
///
/// **粒子不要**（`deduce::opp_king_candidates` 上の幾何）。m095 は厳密粒子
/// ゼロのブラインド決定で、粒子ループ内の減点は `expected=0` に消える。
/// 歩→と金は着地の8近傍に玉候補がいれば発火。角・飛は自駒ブロッカー
/// だけを見た利きで判定する。詰み級の寄せは候補が着地そのもののとき
/// （取って詰み）は対象外。
/// **玉筋が読めていない序盤では発火しない**（広い候補だと全ての歩成りが
/// 「誰かの8近傍」になり 4七歩成クラスを巻き込む）。
/// **手数 `PROMOTE_CHECK_REVEAL_MIN_MOVE..=MAX_MOVE`**（m095=2点・m101=2点の
/// 7d7c+ は課税、m103/m111/m119/m121 の 7d7c+/7c7b+ =10/9 点は課税しない。
/// 同じ USI でも終盤は既知玉への寄せ）。下限は序中盤の成る王手（相手の
/// 解消反則を誘う手）を潰さないため。凍結版はこの名前を知らない。
fn promote_check_reveal_w() -> f64 {
    crate::config::current(|c| c.strategy.promote_check_reveal_w)
}

/// 成る王手の露見ペナルティ。既定 0（env 作業点は 1.2。局面フェーズ依存の
/// 信号はあるが、手数窓は quest31 の m101=2点 / m103=10点の間に引いた境界）。
const PROMOTE_CHECK_REVEAL_W: f64 = 0.0;
/// env 有効時のみ。quest31-m095 向け作業点（汎化根拠なし）。
const PROMOTE_CHECK_REVEAL_MIN_MOVE: u32 = 90;
/// env 有効時のみ。m101（7d7c+ = 2）まで課税、m103（=10）以降は切る。
const PROMOTE_CHECK_REVEAL_MAX_MOVE: u32 = 102;
/// **成ると王手が増える手にだけ不成の双子を作り、その露見を値付けする**
/// （`TSUITATE_NONPROMOTE_CHECK_W`、既定 0 = 無効。生成と減点の両方を
/// この1本のノブが担う）。
///
/// 発端は quest_20260731 の採点済み eval に出る同型の6局面（2026-08-15）:
/// 46/48手目の 4九銀不成(3h4i)=10 vs 4九銀成(3h4i+)=2、95手目の
/// 7三歩不成(7d7c)=10 vs 7三歩成(7d7c+)=2、101手目 6 vs 2、50手目 6 vs 3。
/// ユーザーの採点コメントが判定条件そのものを述べている:
/// 「4九銀成は金が取れなかった場合、**王手がかかってしまい**、そのまま銀が
/// 取られる展開になるので、4九銀不成の方がいい」/「成ると王手がかかって
/// しまい、取られてしまう…成らずとしておくと、駒が取れない場合にこの歩の
/// 存在を相手は観測できず、次に7二歩成ができる」。
///
/// つまり不成の価値は**成った駒種の利きが玉候補に届くかどうか**で決まる。
/// `TSUITATE_GEN_NONPROMOTE=1`（全駒種の不成を生成）が eval で負けたのは
/// 条件を見ずに双子を作るためで（7二歩不成=1「取れることが確定している
/// ので不成にする意味が全くない」が最大の失点源になった）、ここでは
/// **成りが玉候補へ新たに利きを作る手だけ**に双子を絞る。
///
/// 減点は成り側へ `w`（gain 内）。双子が存在する手にしか掛からないので、
/// 逃げ場のない成り（`Promotion::Forced`）や王手が増えない成りは不動。
/// **粒子不要**（`deduce::opp_king_candidates` 上の幾何）で、m095 のような
/// 厳密粒子ゼロのブラインド決定でも効く。王手中は無効。
/// 凍結版はこの名前を知らない。
fn nonpromote_check_w() -> f64 {
    crate::config::current(|c| c.strategy.nonpromote_check_w)
}

/// 双子を作る駒種（`TSUITATE_NONPROMOTE_CHECK_ROLES`、既定 `minor` =
/// 銀・桂・香）。`all` で歩・角・飛も対象にする。
///
/// **歩を既定で外すのは実測による**（2026-08-15、700ms・10シード）:
/// 全駒種版は「成りが最善」の局面を軒並み壊した（m121 10.00→1.00・
/// m127 9.00→1.00・m111 10.00→3.00・m020/m040 −6・m103 −6）。歩の成りは
/// **王手が掛かること自体が狙い**の場合があり、ユーザーの採点コメントが
/// まさに両側を述べている: 95手目「成ると王手がかかってしまい、取られて
/// しまう」（不成=10）vs 111手目「王手がかからないと、この歩を無視して
/// 別の手を指される」（成=10）。どちらになるかは「成った駒がそこで
/// 生き残れるか」で決まり、ブラインド決定では観測から判定できない。
/// 歩を含めた合計は −57.9/146件、銀桂香だけなら +11.7 と符号が逆になる
fn nonpromote_check_roles() -> String {
    crate::config::current(|c| c.strategy.nonpromote_check_roles.clone())
}

fn nonpromote_check_role_ok(role: Role) -> bool {
    match nonpromote_check_roles().as_str() {
        "all" => true,
        "silver" => role == Role::Silver,
        _ => matches!(role, Role::Silver | Role::Knight | Role::Lance),
    }
}

/// 双子を作る最小の王手確率（`TSUITATE_NONPROMOTE_CHECK_P`、既定 0.2）。
fn nonpromote_check_p() -> f64 {
    crate::config::current(|c| c.strategy.nonpromote_check_p)
}

/// 成ることで**新たに**玉へ利きが生じる確率（不成では生じない分だけ）。
/// `promote_checks_king_cand` との違いは2点:
/// - **差分**であること（不成でも王手なら「成ると露見する」理由にならない）
/// - 候補集合の**真偽でなく玉位置ネットの質量**で測ること。deduce の候補は
///   王手をあまり掛けていない側では 35〜55 マスに散るので、真偽で見ると
///   隣のマスへの成り（quest31-m046 の 2九銀成）まで発火してしまう
///   （実測: 真偽版は 46手目の Optional 成りの半分以上で発火した）
fn promotion_check_mass(
    view: &PlayerView,
    from: Coord,
    to: Coord,
    role: Role,
    dist: &[(Coord, f64)],
    certain_occ: &[bool; 81],
) -> f64 {
    if !nonpromote_check_role_ok(role) {
        return 0.0;
    }
    // **着地の占有が観測で確定しているなら成る**: 露見が損になるのは
    // 「取れなかった場合」に駒だけ晒されるからで、確実に取れるマスなら
    // 取った時点で相手に通知される（露見コストは既に払っている）。
    // ユーザー採点がこの区別を裏づける（2026-08-15）: 95手目
    // 7三歩不成=10「相手が7三に駒を打っていなかった場合、王手がかかって
    // しまい、取られてしまう」に対し、7二の銀が既知の 121/127手目は
    // 7二歩成=9/10・不成=1、同じ 6b7c でも占有確定の 135/143手目は
    // 銀不成=1/5
    if certain_occ[crate::belief_features::sq_index(to)] {
        return 0.0;
    }
    let Some(promo_role) = promote_role(role) else {
        return 0.0;
    };
    let me = view.your_color;
    let mut own = [false; 81];
    for p in &view.your_pieces {
        let Some(c) = parse_usi_square(&p.square) else {
            continue;
        };
        if c == from {
            continue; // 動かす駒は vacate
        }
        own[crate::belief_features::sq_index(c)] = true;
    }
    let mut mass = 0.0;
    for &(k, p) in dist {
        if k == to {
            continue; // 玉が着地そのもの＝取る王手は「寄せ」
        }
        if own[crate::belief_features::sq_index(k)] {
            continue; // 自駒マスに玉は居ない
        }
        if piece_attacks_sq(promo_role, me, to, k, &own) && !piece_attacks_sq(role, me, to, k, &own)
        {
            mass += p;
        }
    }
    mass
}

/// 不成の双子を作る/減点する対象か（王手確率がしきい値以上）。
fn promotion_adds_check(
    view: &PlayerView,
    from: Coord,
    to: Coord,
    role: Role,
    dist: &[(Coord, f64)],
    certain_occ: &[bool; 81],
) -> bool {
    promotion_check_mass(view, from, to, role, dist, certain_occ) >= nonpromote_check_p()
}

/// `nonpromote_check_w` 用の玉位置分布（deduce 候補上の玉位置ネット）。
fn nonpromote_king_dist(view: &PlayerView, log: &ObservationLog) -> Vec<(Coord, f64)> {
    let ctx = crate::belief_features::BeliefContext::from_log(view.your_color, log);
    let cands = crate::deduce::opp_king_candidates(view.your_color, log);
    crate::king_belief_nn::king_distribution(&ctx, &cands)
}

/// 成る手が `deduce` 玉候補のいずれかに王手を掛けるか（観測のみ）。
fn promote_checks_king_cand(
    view: &PlayerView,
    from: Coord,
    to: Coord,
    role: Role,
    cands: &std::collections::BTreeSet<Coord>,
) -> bool {
    let me = view.your_color;
    // 着地に自駒は無い（合法候補の前提）。玉候補が着地そのもの＝取る王手は
    // 「寄せ」なので露見ペナルティの対象外
    let mut own = [false; 81];
    for p in &view.your_pieces {
        let Some(c) = parse_usi_square(&p.square) else {
            continue;
        };
        if c == from {
            continue; // 動かす駒は vacate
        }
        own[crate::belief_features::sq_index(c)] = true;
    }
    let promo_role = match role {
        Role::Pawn => Role::Tokin,
        Role::Bishop => Role::Horse,
        Role::Rook => Role::Dragon,
        _ => return false,
    };
    for &k in cands {
        if k == to {
            continue;
        }
        if own[crate::belief_features::sq_index(k)] {
            continue; // 自駒マスに玉は居ない
        }
        if piece_attacks_sq(promo_role, me, to, k, &own) {
            return true;
        }
    }
    false
}

/// 自駒ブロッカーのみを考慮した利き（ついたての「自分側だけ見える」利き）。
fn piece_attacks_sq(role: Role, me: Color, from: Coord, target: Coord, own: &[bool; 81]) -> bool {
    let df = i32::from(target.file - from.file);
    let dr = i32::from(target.rank - from.rank);
    let adf = df.abs();
    let adr = dr.abs();
    let clear_ray = || -> bool {
        let steps = adf.max(adr);
        if steps <= 1 {
            return true;
        }
        let step_f = df.signum();
        let step_r = dr.signum();
        for s in 1..steps {
            let sq = Coord {
                file: from.file + (step_f * s) as i8,
                rank: from.rank + (step_r * s) as i8,
            };
            if own[crate::belief_features::sq_index(sq)] {
                return false;
            }
        }
        true
    };
    match role {
        Role::Tokin
        | Role::Gold
        | Role::Promotedlance
        | Role::Promotedknight
        | Role::Promotedsilver => {
            // 金相当: 前3・左右・直後（斜め後ろ以外の隣接）
            let forward = match me {
                Color::Sente => -1,
                Color::Gote => 1,
            };
            adf <= 1 && adr <= 1 && !(adf == 0 && adr == 0) && !(adf == 1 && dr == -forward)
        }
        Role::Horse => {
            if adf <= 1 && adr <= 1 && !(adf == 0 && adr == 0) {
                return true;
            }
            adf == adr && adf > 0 && clear_ray()
        }
        Role::Dragon => {
            if adf <= 1 && adr <= 1 && !(adf == 0 && adr == 0) {
                return true;
            }
            ((adf == 0 && adr > 0) || (adr == 0 && adf > 0)) && clear_ray()
        }
        _ => false,
    }
}

/// 大駒成りの遠方量（`promote_far_w` の材料）。
/// - 着地の最小チェビシェフが 1 を超えたぶん
/// - **玉へ近づかない**成り（`d_to ≥ d_from`）は最低 1
///   （玉候補の裾が 3b 近傍に残っているとき 4a3b+ が免税で残る対策。
///   combo_far_v2 実測で 4a3b+ がなお 18）
/// 候補が空なら 0（減点しない = 安全方向）。
fn promote_far_amount(from: Coord, to: Coord, cands: &std::collections::BTreeSet<Coord>) -> f64 {
    let Some(d_to) = cands.iter().map(|&k| chebyshev(to, k)).min() else {
        return 0.0;
    };
    let d_from = cands
        .iter()
        .map(|&k| chebyshev(from, k))
        .min()
        .unwrap_or(d_to);
    let mut amount = f64::from(d_to.saturating_sub(1));
    if d_to >= d_from {
        amount = amount.max(1.0);
    }
    amount
}

fn chebyshev(a: Coord, b: Coord) -> i32 {
    i32::from((a.file - b.file).abs().max((a.rank - b.rank).abs()))
}

fn in_enemy_camp(to: Coord, me: Color) -> bool {
    match me {
        Color::Sente => to.rank <= 3,
        Color::Gote => to.rank >= 7,
    }
}

fn in_own_camp(to: Coord, me: Color) -> bool {
    match me {
        Color::Sente => to.rank >= 7,
        Color::Gote => to.rank <= 3,
    }
}

/// 相手の最奥段（先手なら 1 段目、後手なら 9 段目）。香・金の初期位置。
fn on_enemy_back_rank(to: Coord, me: Color) -> bool {
    match me {
        Color::Sente => to.rank == 1,
        Color::Gote => to.rank == 9,
    }
}

/// 玉候補の筋の中央値。空なら None。
fn king_file_median(cands: &std::collections::BTreeSet<Coord>) -> Option<i8> {
    let n = cands.len();
    if n == 0 {
        return None;
    }
    let mut files: Vec<i8> = cands.iter().map(|k| k.file).collect();
    files.sort_unstable();
    Some(files[n / 2])
}

/// 玉の筋が「読める」か。中央値 ±2 に候補の 2/3 以上がいるときだけ真。
/// 初期玉から半径が大きく広がった全盤候補では 9 筋ほぼ均等なので偽になる。
fn king_files_focused(cands: &std::collections::BTreeSet<Coord>, median: i8) -> bool {
    let n = cands.len();
    if n == 0 {
        return false;
    }
    let near = cands
        .iter()
        .filter(|k| (k.file - median).abs() <= 2)
        .count();
    near * 3 >= n * 2
}

/// 中段の玉筋歩前進量（`king_file_pawn_mid_w`）。手数 80..=86・玉筋が読めて
/// いるときだけ。自陣段からの前進に限り、中央値の筋そのものは 0
/// （8六歩の未採点逃避）。敵陣は `king_file_pawn_amount` の領分。
fn king_file_pawn_mid_amount(
    from: Coord,
    to: Coord,
    me: Color,
    cands: &std::collections::BTreeSet<Coord>,
    move_number: u32,
) -> f64 {
    if !(KING_FILE_PAWN_MID_MIN_MOVE..=KING_FILE_PAWN_MID_MAX_MOVE).contains(&move_number) {
        return 0.0;
    }
    let forward = match me {
        Color::Sente => to.file == from.file && to.rank == from.rank - 1,
        Color::Gote => to.file == from.file && to.rank == from.rank + 1,
    };
    // 自陣段から出る歩だけ（7g7f）。既に中段の 9f9e は終盤の仮4点逃避になる。
    if !forward || !in_own_camp(from, me) || in_enemy_camp(to, me) || in_own_camp(to, me) {
        return 0.0;
    }
    let Some(median) = king_file_median(cands) else {
        return 0.0;
    };
    if !king_files_focused(cands, median) {
        return 0.0;
    }
    let d_file = (to.file - median).abs();
    // 中央値そのもの（実測 8六）は除外。側筋 7六・9六だけ満額。
    if d_file < 1 || d_file > 2 {
        return 0.0;
    }
    1.0
}

/// 終盤の玉の非捕獲逃げ量（`king_endgame_flee_w`）。
/// 王手中の空マス逃げも含む。裏付け占有（取り返し）は 0。
fn king_endgame_flee_amount(to: Coord, view: &PlayerView, backed: Option<&[bool; 81]>) -> f64 {
    if view.move_number < KING_ENDGAME_FLEE_MIN_MOVE {
        return 0.0;
    }
    if backed.is_some_and(|b| b[crate::belief_features::sq_index(to)]) {
        return 0.0;
    }
    1.0
}

/// 終盤の金が自玉へ隣接する量（`gold_join_king_w`）。
/// 王手中に金が玉へ寄るときだけ 1。打ち込み王手は観測されないので
/// 占有裏付けは要求しない。
fn gold_join_king_amount(
    from: Coord,
    to: Coord,
    king: Coord,
    move_number: u32,
    in_check: bool,
) -> f64 {
    if !in_check || move_number < GOLD_JOIN_KING_MIN_MOVE {
        return 0.0;
    }
    if chebyshev(from, king) <= 1 {
        return 0.0;
    }
    if chebyshev(to, king) == 1 {
        1.0
    } else {
        0.0
    }
}

/// 終盤、自玉隣接の金が玉筋へ動く量（`gold_king_file_w`）。
/// 非王手・既に隣接・着地が玉と同じ筋かつ距離 2 のときだけ 1。
fn gold_king_file_amount(
    from: Coord,
    to: Coord,
    king: Coord,
    move_number: u32,
    in_check: bool,
) -> f64 {
    if in_check || move_number < GOLD_KING_FILE_MIN_MOVE {
        return 0.0;
    }
    if chebyshev(from, king) != 1 {
        return 0.0;
    }
    if to.file != king.file {
        return 0.0;
    }
    if chebyshev(to, king) == 2 {
        1.0
    } else {
        0.0
    }
}

/// 終盤の桂の敵陣進入課税量（`knight_late_promo_w`）。成りは 1、
/// 不成は 110 手以降だけ scale。
fn knight_late_promo_amount(
    role: Role,
    _from: Coord,
    to: Coord,
    promote: bool,
    me: Color,
    move_number: u32,
) -> f64 {
    if role != Role::Knight
        || !(KNIGHT_LATE_PROMO_MIN_MOVE..=KNIGHT_LATE_PROMO_MAX_MOVE).contains(&move_number)
    {
        return 0.0;
    }
    if !in_enemy_camp(to, me) {
        return 0.0;
    }
    if promote {
        1.0
    } else if move_number >= KNIGHT_LATE_NONPROMO_MIN_MOVE {
        KNIGHT_LATE_NONPROMO_SCALE
    } else {
        0.0
    }
}

/// 終盤の桂の敵陣成り加点量（`knight_endgame_promo_w`）。手数 137 以降・成りのみ。
fn knight_endgame_promo_amount(
    role: Role,
    to: Coord,
    promote: bool,
    me: Color,
    move_number: u32,
) -> f64 {
    if role != Role::Knight || !promote || move_number < KNIGHT_ENDGAME_PROMO_MIN_MOVE {
        return 0.0;
    }
    if in_enemy_camp(to, me) {
        1.0
    } else {
        0.0
    }
}

/// 終盤、自陣の桂が中段へ出る量（`knight_camp_exit_w`）。
/// 初期玉筋（5筋）へ近づく手は 0。
fn knight_camp_exit_amount(role: Role, from: Coord, to: Coord, me: Color, move_number: u32) -> f64 {
    if role != Role::Knight || move_number < KNIGHT_CAMP_EXIT_MIN_MOVE {
        return 0.0;
    }
    if !in_own_camp(from, me) || in_own_camp(to, me) || in_enemy_camp(to, me) {
        return 0.0;
    }
    // 5筋は両者の初期玉。信念の median がここに残ると 6b5d が「接近」に見える。
    if (to.file - 5).abs() < (from.file - 5).abs() {
        return 0.0;
    }
    1.0
}

/// 終盤の銀が自陣から出る量（`silver_camp_exit_w`）。
fn silver_camp_exit_amount(role: Role, from: Coord, to: Coord, me: Color, move_number: u32) -> f64 {
    if role != Role::Silver || move_number < SILVER_CAMP_EXIT_MIN_MOVE {
        return 0.0;
    }
    if in_own_camp(from, me) && !in_own_camp(to, me) {
        1.0
    } else {
        0.0
    }
}

/// 玉筋の歩前進量（`king_file_pawn_w`）。敵陣への前進1マスかつ中央値の筋距離 ≤2。
///
/// 入口段の盤上加点除外は 5f5g+ が 5〜6 点の局面まで沈み、B*2f / P*4g へ
/// 流出してネット負だったので戻す。終盤の悪い 5f5g+ は手数ゲートの歩成り
/// 課税でも m110（7点）を壊すため、ここでは触らない。
fn king_file_pawn_amount(
    from: Coord,
    to: Coord,
    me: Color,
    cands: &std::collections::BTreeSet<Coord>,
) -> f64 {
    let forward = match me {
        Color::Sente => to.file == from.file && to.rank == from.rank - 1,
        Color::Gote => to.file == from.file && to.rank == from.rank + 1,
    };
    if !forward || !in_enemy_camp(to, me) || cands.is_empty() {
        return 0.0;
    }
    king_file_pawn_drop_amount(to, cands)
}

/// 敵陣（または玉筋）への歩打ち。中央値の筋距離 ≤2 なら `1/(1+d_file)`。
/// 全盤に広がった候補でも中央値は 5 筋付近なので 9六歩は加点 0。
fn king_file_pawn_drop_amount(to: Coord, cands: &std::collections::BTreeSet<Coord>) -> f64 {
    let Some(median) = king_file_median(cands) else {
        return 0.0;
    };
    let d_file = (to.file - median).abs();
    if d_file > 2 {
        return 0.0;
    }
    1.0 / (1.0 + f64::from(d_file))
}

/// 金または銀を持っているか（歩の玉筋外れ課税の機会損失ゲート）。
fn has_attacking_general(view: &PlayerView) -> bool {
    view.your_hand.get(&Role::Gold).copied().unwrap_or(0) > 0
        || view.your_hand.get(&Role::Silver).copied().unwrap_or(0) > 0
}

/// 終盤の敵陣歩進入量（`pawn_offfile_w`）。手数 125 以降かつ裏付け無し。
/// 既に敵陣にいる歩の前進は 0（m127 の 7c7b+）。
fn pawn_late_promo_amount(
    from: Coord,
    to: Coord,
    me: Color,
    move_number: u32,
    backed_hit: bool,
) -> f64 {
    if move_number < PAWN_OFFFILE_MIN_MOVE || backed_hit {
        return 0.0;
    }
    if !in_enemy_camp(to, me) {
        return 0.0;
    }
    if in_enemy_camp(from, me) {
        return 0.0;
    }
    1.0
}

/// 終盤の敵陣成銀の筋替え量（`endgame_camp_general_w`）。
/// 玉筋（中央値）へ近づくときだけ 1。
fn endgame_camp_general_amount(
    role: Role,
    from: Coord,
    to: Coord,
    me: Color,
    move_number: u32,
    cands: &std::collections::BTreeSet<Coord>,
) -> f64 {
    if move_number < ENDGAME_CAMP_GENERAL_MIN_MOVE {
        return 0.0;
    }
    if role != Role::Promotedsilver {
        return 0.0;
    }
    if from.file == to.file {
        return 0.0;
    }
    if !in_enemy_camp(from, me) || !in_enemy_camp(to, me) {
        return 0.0;
    }
    let Some(median) = king_file_median(cands) else {
        return 0.0;
    };
    if (to.file - median).abs() >= (from.file - median).abs() {
        return 0.0;
    }
    1.0
}

/// と金の空きマス移動課税量（`tokin_file_drift_w`）。
/// 玉筋に留まる・大駒の筋を空ける・裏付け捕獲は 0。
fn tokin_file_drift_amount(
    view: &PlayerView,
    from: Coord,
    to: Coord,
    me: Color,
    cands: &std::collections::BTreeSet<Coord>,
    backed: &[bool; 81],
) -> f64 {
    if !in_enemy_camp(to, me) {
        return 0.0;
    }
    if backed[crate::belief_features::sq_index(to)] {
        return 0.0;
    }
    // 相手最奥段の香・金取り（m019 の 2a1a）は逸れではない
    if on_enemy_back_rank(to, me) {
        return 0.0;
    }
    if let Some(median) = king_file_median(cands) {
        let d0 = (from.file - median).abs();
        let d1 = (to.file - median).abs();
        // 玉筋に留まるだけ免税。近づき免税は 4g5h（採点 2）が繰り上がる。
        if d0 == 0 && d1 == 0 {
            return 0.0;
        }
    }
    if tokin_vacates_major_file(view, from, to) {
        return 0.0;
    }
    1.0
}

fn tokin_vacates_major_file(view: &PlayerView, from: Coord, to: Coord) -> bool {
    if from.file == to.file {
        return false;
    }
    view.your_pieces.iter().any(|p| {
        matches!(
            p.role,
            Role::Rook | Role::Dragon | Role::Bishop | Role::Horse
        ) && parse_usi_square(&p.square).is_some_and(|c| c.file == from.file)
    })
}

/// 裏付け無し敵陣進入の課税量（`unbacked_camp_w`）。
/// 角・飛・馬・龍は打ちも移動も課税。金銀は**盤上移動だけ**
/// （m081 の 6c6b 幻捕獲。G*7c のような打ちは HAND_ASSET の領分）。
/// と金・歩は対象外（2a3a / 4七歩成）。
fn unbacked_camp_amount(
    role: Role,
    to: Coord,
    me: Color,
    backed: &[bool; 81],
    is_drop: bool,
) -> f64 {
    if !in_enemy_camp(to, me) {
        return 0.0;
    }
    if backed[crate::belief_features::sq_index(to)] {
        return 0.0;
    }
    match role {
        Role::Bishop | Role::Rook | Role::Horse | Role::Dragon => exchange_value(role),
        Role::Gold | Role::Silver if !is_drop => exchange_value(role),
        _ => 0.0,
    }
}

/// 金銀の敵陣進入税は、粒子が「そこに駒が居る」と信じているときだけ掛ける。
/// capture≈0 の 3h4i（実在の金が見えていない）を 4g4h へ流出させない。
fn unbacked_camp_needs_capture(role: Role, capture_value: f64) -> bool {
    match role {
        Role::Gold | Role::Silver => capture_value >= 0.5,
        _ => true,
    }
}

/// 相手の金の初期マス（先手が攻めるなら 4a/6a、後手なら 4i/6i）。
fn opp_gold_homes(me: Color) -> [Coord; 2] {
    match me {
        Color::Sente => [Coord { file: 4, rank: 1 }, Coord { file: 6, rank: 1 }],
        Color::Gote => [Coord { file: 4, rank: 9 }, Coord { file: 6, rank: 9 }],
    }
}

/// 同一駒が初期金の筋の別マスへ動けるなら、空の初期金への加点を抑える。
/// m054 の 3h4g は 4 筋なので 3h4i をブロック。m046 は自と金が 4g にいて
/// 3h4g が無く、3h2i は 2 筋なので通す。捕獲信念も裏付けも使わない
/// （m054 の 4g は capture≈0 でも候補に載る）。
fn home_gold_file_sibling(to: Coord, me: Color) -> bool {
    opp_gold_homes(me)
        .iter()
        .any(|&h| h.file == to.file && h != to)
}

/// 相手の初期金位置への金銀の当たり量（`home_gold_attack_w`）。
/// 成る手（3h4i+）は 0。観測裏付けのあるマスは捕獲側が既に数える。
fn home_gold_attack_amount(
    role: Role,
    to: Coord,
    me: Color,
    backed: &[bool; 81],
    promote: bool,
) -> f64 {
    if promote {
        return 0.0;
    }
    if !matches!(role, Role::Gold | Role::Silver) {
        return 0.0;
    }
    if backed[crate::belief_features::sq_index(to)] {
        return 0.0;
    }
    if opp_gold_homes(me).iter().any(|&h| h == to) {
        1.0
    } else {
        0.0
    }
}

/// と金が玉筋へ寄る量（`tokin_approach_w`）。
fn tokin_file_approach_amount(
    from: Coord,
    to: Coord,
    me: Color,
    cands: &std::collections::BTreeSet<Coord>,
) -> f64 {
    if !in_enemy_camp(to, me) {
        return 0.0;
    }
    let Some(median) = king_file_median(cands) else {
        return 0.0;
    };
    if !king_files_focused(cands, median) {
        return 0.0;
    }
    let d0 = (from.file - median).abs();
    let d1 = (to.file - median).abs();
    if d1 >= d0 {
        return 0.0;
    }
    1.0 / (1.0 + f64::from(d1))
}

/// 自陣の金銀桂の空きマス移動量（`own_camp_idle_w`）。
fn own_camp_idle_amount(
    role: Role,
    to: Coord,
    me: Color,
    backed: &[bool; 81],
    move_number: u32,
) -> f64 {
    if move_number < OWN_CAMP_IDLE_MIN_MOVE {
        return 0.0;
    }
    if !matches!(role, Role::Gold | Role::Silver | Role::Knight) {
        return 0.0;
    }
    if !in_own_camp(to, me) {
        return 0.0;
    }
    if backed[crate::belief_features::sq_index(to)] {
        return 0.0;
    }
    exchange_value(role)
}

/// 角・馬の非前進空きマス移動量（`bishop_retreat_w`）。
/// 前進（先手なら rank 減、後手なら rank 増）と敵陣進入・裏付けは 0。
fn bishop_retreat_amount(
    role: Role,
    from: Coord,
    to: Coord,
    me: Color,
    backed: &[bool; 81],
) -> f64 {
    if !matches!(role, Role::Bishop | Role::Horse) {
        return 0.0;
    }
    if in_enemy_camp(to, me) {
        return 0.0;
    }
    if backed[crate::belief_features::sq_index(to)] {
        return 0.0;
    }
    let advancing = match me {
        Color::Sente => to.rank < from.rank,
        Color::Gote => to.rank > from.rank,
    };
    if advancing {
        return 0.0;
    }
    exchange_value(role)
}

/// 持ち駒資産損の対象駒・着地か。金銀桂は全域、角飛は敵陣以外、歩は自陣だけ。
fn hand_asset_drop_taxable(role: Role, to: Coord, me: Color) -> bool {
    match role {
        Role::Gold | Role::Silver | Role::Knight => true,
        Role::Bishop | Role::Rook => !in_enemy_camp(to, me),
        Role::Pawn => in_own_camp(to, me),
        _ => false,
    }
}

/// 玉隣接への高い駒の無支え進入量（`king_adj_heavy_w`）。
/// **と金だけ**（m021 の 3a4a / 2c1c）。金銀大駒まで広げるとアリーナの
/// 寄せ・支えまで課税して反則押し出しになる（対 v13 46.6% の残差）。
fn king_adj_heavy_amount(
    role: Role,
    to: Coord,
    me: Color,
    cands: &std::collections::BTreeSet<Coord>,
    backed: &[bool; 81],
) -> f64 {
    if role != Role::Tokin {
        return 0.0;
    }
    if backed[crate::belief_features::sq_index(to)] {
        return 0.0;
    }
    let Some(median) = king_file_median(cands) else {
        return 0.0;
    };
    if !king_files_focused(cands, median) {
        return 0.0;
    }
    let adjacent = cands
        .iter()
        .any(|&k| chebyshev(to, k) <= 1 && (k.file - median).abs() <= 2);
    if !adjacent {
        // 玉筋から外れたと金（2c1c）。最奥段の香取り（2a1a）は免税
        let off_file = (to.file - median).abs() > 2 && !on_enemy_back_rank(to, me);
        if !off_file {
            return 0.0;
        }
    }
    exchange_value(role)
}

/// 桂銀香の任意成り課税量（`own_camp_minor_promo_w`）。成る手だけ 1。
/// 強制成りと、成れない移動は 0。
fn own_camp_minor_promo_amount(
    role: Role,
    from: Coord,
    to: Coord,
    promote: bool,
    me: Color,
) -> f64 {
    if !promote || !gen_nonpromote() {
        return 0.0;
    }
    if !matches!(role, Role::Silver | Role::Knight | Role::Lance) {
        return 0.0;
    }
    match promotion_choice(role, from, to, me) {
        Promotion::Optional => 1.0,
        _ => 0.0,
    }
}

/// 金銀打ちが自玉の守りか。
/// - 金: 自玉の 8 近傍（quest31-m055 の G*5g / G*5h。玉が 6h に居るとき
///   5g は斜め隣接。隣接銀打ち S*5h は対象外のまま）。
///   手数ゲートは m033–m049 の G*5h / G*5i を沈められず（次の 2 点手が
///   繰り上がるだけ）、m055 を巻き込んだので外した。
/// - 銀: 同じ筋の玉頭ちょうど2マスだけ（G*5g 型。S*5h / S*3h は課税）
fn own_king_drop_is_defensive(role: Role, to: Coord, king: Coord, me: Color) -> bool {
    if role == Role::Gold && chebyshev(to, king) <= 1 {
        return true;
    }
    if to.file != king.file {
        return false;
    }
    let forward = match me {
        Color::Sente => king.rank - to.rank,
        Color::Gote => to.rank - king.rank,
    };
    forward == 2
}

/// 玉の既知脅威への接近量（`king_known_approach_w` の材料）。
/// チェビシェフが縮むときはその差分、同じ危険圏（≤2）で筋/段だけ寄るとき
/// は 0.5（quest31-m099: 7g→6h は 5f へ dist=2 のまま筋だけ寄る）。
/// 脅威マス自体を取る手は 0。
fn king_known_approach_amount(from: Coord, to: Coord, backed: &[bool; 81]) -> f64 {
    let mut amount = 0.0f64;
    for t in crate::belief_features::all_squares() {
        if !backed[crate::belief_features::sq_index(t)] || to == t {
            continue;
        }
        let d0 = chebyshev(from, t);
        let d1 = chebyshev(to, t);
        if d1 < d0 {
            amount += (d0 - d1) as f64;
        } else if d1 <= 2 {
            let file_closer = (to.file - t.file).abs() < (from.file - t.file).abs();
            let rank_closer = (to.rank - t.rank).abs() < (from.rank - t.rank).abs();
            if file_closer || rank_closer {
                amount += 0.5;
            }
        }
    }
    amount
}

/// 打ちの「仕事」があるか（`hand_asset_w`）。
///
/// - **金の自玉 8 近傍** / **銀の玉頭ちょうど2マス**: 守り打ち
///   （quest31-m055 の G*5g。隣接の S*5h や斜めの S*3h は対象外 = m027）
/// - **安い駒**（歩香桂）: 裏付け占有への当たり
/// - **高い駒**: 敵陣かつ玉筋が読めるときの敵玉近接、またはその近くの裏付けへの当たり。
///   金に限り、玉候補そのもの／玉筋（中央値）の 8 近傍への打ちは近接免税しない
///   （m087 の G*7c、m145 の G*8c）。銀の寄せ打ちは残す。
///
/// 発端は quest31-m062 の `G*1b`。玉候補が41マスに拡散している局面で
/// 「昔取られた 2c に当たる」だけで金打ちが免税され、端への無目的な
/// 金打ちが link 加点ごと生き残っていた。敵玉近接は `king_files_focused`
/// のときだけ数える（len≤8 の旧ゲートは終盤の G*6c まで巻き込む）。
/// 金銀の敵玉仕事は**敵陣に限る**（G*5e が中段の裾候補で免税される穴）。
fn drop_has_hand_asset_work(
    view: &PlayerView,
    role: Role,
    to: Coord,
    backed: &[bool; 81],
    king_cands: &std::collections::BTreeSet<Coord>,
) -> bool {
    let me = view.your_color;
    if matches!(role, Role::Gold | Role::Silver) {
        if let Some(k) = king_square(view) {
            if own_king_drop_is_defensive(role, to, k, me) {
                return true;
            }
        }
    }
    let mut pieces = view.your_pieces.clone();
    pieces.push(VisiblePiece {
        square: make_usi_square(to),
        role,
    });
    let p = pieces.last().expect("just pushed");
    let attack_backed: Vec<Coord> = crate::board::defend_targets(&pieces, p, me)
        .into_iter()
        .filter(|&s| backed[crate::belief_features::sq_index(s)])
        .collect();
    let cheap = matches!(role, Role::Pawn | Role::Lance | Role::Knight);
    if cheap && !attack_backed.is_empty() {
        return true;
    }
    let Some(median) = king_file_median(king_cands) else {
        return false;
    };
    if !king_files_focused(king_cands, median) {
        return false;
    }
    // 金を玉筋から 3 筋以上外して打つのは仕事なし（m145 の G*6b）。
    if role == Role::Gold && (to.file - median).abs() > 2 {
        return false;
    }
    let near_king = |c: Coord| {
        king_cands
            .iter()
            .any(|k| (k.file - c.file).abs() <= 2 && (k.rank - c.rank).abs() <= 2)
    };
    // 金銀の敵玉近接は敵陣だけ（G*5e が中段の裾で免税される対策）
    if matches!(role, Role::Gold | Role::Silver) && !in_enemy_camp(to, me) {
        return false;
    }
    if near_king(to) {
        // 金を玉候補そのもの、または玉筋（中央値の筋）の隣接へ打つのは
        // 仕事ではなくプローブ／裸の打ち込み（m087 の G*7c、m145 の G*8c）。
        // 裏付け当たりへも落とさない（7c が何かに当たっても 1 点のまま）。
        // 銀は 4b 型の寄せ打ちを残す。チェビシェフ2 の金（m087 の G*6c）は免税。
        if role == Role::Gold {
            let on_king = king_cands.contains(&to);
            let adj_median_king = king_cands
                .iter()
                .any(|&k| chebyshev(to, k) <= 1 && k.file == median);
            return !on_king && !adj_median_king;
        }
        return true;
    }
    // 高い駒: 玉の近くの裏付けを取る／当てる打ちだけ仕事
    !cheap && attack_backed.iter().any(|&s| near_king(s))
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
    crate::config::current(|c| c.strategy.check_king_gain_mean)
}

/// 王手中に「ほぼ確実な解消手」があるのに低い p の手を選んで反則するのを止める
/// （`TSUITATE_CHECK_SAFE_RESOLVE`、**既定 off**、`1` で有効）。
///
/// 対 v13 104局の analyze: 王手中反則 247回 → ソルバー方策なら 139回、という
/// 診断は正しい。しかし既定 on・p_max≥0.70 で切る版は CI ガントレット
/// （run 32094628907、104局）で **v13 48.1%**・kakutori **2/20**
/// （suite run 32094626767、平均 4.614）と両方壊した。
/// 仮説希釈で p_max が 0.70 を超える kakutori 型では、正しい捕獲プローブまで
/// `p_max-0.25` 未満として捨てていた。有効時は王手駒捕獲と玉の手を残す。
/// 凍結版はこの名前を知らない。
fn check_safe_resolve_enabled() -> bool {
    crate::config::current(|c| c.strategy.check_safe_resolve_enabled)
}

const CHECK_SAFE_RESOLVE_PMAX: f64 = 0.70;
const CHECK_SAFE_RESOLVE_MARGIN: f64 = 0.25;

fn check_safe_resolve_active(p_max: f64) -> bool {
    p_max >= CHECK_SAFE_RESOLVE_PMAX
}

fn check_safe_resolve_thresh(p_max: f64) -> f64 {
    p_max - CHECK_SAFE_RESOLVE_MARGIN
}

/// 安全解消ゲートで残す手か。閾値以上に加え、王手駒捕獲と玉の手は
/// 仮説希釈で p が低くても残す（kakutori の捕獲プローブを落とさない）。
fn check_safe_resolve_keep(p: f64, thresh: f64, captures_checker: bool, is_king: bool) -> bool {
    p + 1e-12 >= thresh || captures_checker || is_king
}

/// 成りが任意の移動で**不成も候補に生成する**か（既定は無効 =
/// 従来の「成れるなら成る」。`TSUITATE_GEN_NONPROMOTE=1` で有効。
/// 2026-08-09 に採否保留・既定0で確定）。
/// 凍結版は自前の candidate_moves を持つのでこの名前を知らない。
///
/// 発端は quest_20260731 の95手目（人間の ７三歩**成らず**）。不成の価値は
/// 駒種で2系統ある（2026-08-08 ユーザーの指導）:
/// - **歩・飛・角**（成りは利きの純増）: 成ると信念上の玉へ王手が掛かる手筋が
///   増え、王手は**宣言で露見**して駒を失いやすい（blind_attack_survive_w と
///   同じ経済）。不成なら**バレずに侵入**でき、歩は次の成りも狙える
///   = ついたて固有の価値
/// - **桂・銀（・香）**（成ると元の利きを失う）: 桂の独特の跳び・銀の斜め後ろ
///   への退路・香の縦の走りを**保つ**ための不成 = 通常将棋と共通の価値。
///   着手後の駒種が変わるので評価の標準項（threat/露出/被覆）が自然に扱える
/// 従来は生成段階で不成を刈っていたため、評価側（mover_check_extra /
/// promo_potential）に判断させる機会すら無かった（実戦の 7d7c が make_eval の
/// スケルトンにも載らない）。駒種フィルタは置かない（駒種特化を足さない方針）
/// 不成を生成する駒種か（`TSUITATE_GEN_NONPROMOTE`）。
/// `1` は全駒種、`minor` は**銀・桂・香だけ**（元の利きを保つ系）
fn gen_nonpromote_for(role: Role) -> bool {
    if gen_nonpromote_minor() {
        return matches!(role, Role::Silver | Role::Knight | Role::Lance);
    }
    gen_nonpromote()
}

/// `TSUITATE_GEN_NONPROMOTE=minor`（銀桂香だけ不成を生成）か
fn gen_nonpromote_minor() -> bool {
    crate::config::current(|c| c.strategy.gen_nonpromote_minor)
}

fn gen_nonpromote() -> bool {
    crate::config::current(|c| c.strategy.gen_nonpromote)
}

/// 成る手の取られリスクを**成る前の駒価値**で数えるか（**既定 on** =
/// 2026-08-21 採用。`TSUITATE_PROMO_RISK_PREROLE=0` で従来挙動 = 切り戻しノブ。
/// 2026-08-09 の保留は GEN_NONPROMOTE 込みの計測で、単体+capture_retreat の
/// 2点は suite 1008→939 / 5.293→5.531・アリーナ対照同等以上で採用）。
/// **v14 はこの名前を読む**（v13 以前は知らない）。
///
/// GEN_NONPROMOTE の初回計測（2026-08-08）で露呈した歪みへの対応:
/// 4七歩成（quest31-m016/018/020/024/026/028 の本命・従来20/20）が不成の
/// 4七歩へ 20/20 で流れた。原因は `recapture_risk` と床が着手後の駒種
/// （と金 ≈3）でリスクを数えるため、成ると取り返し損の見積もりが歩(1)の
/// 3倍になり、promo_potential の実現加点（0.2×Δ利き5 ≈ 1.0）では覆せない。
/// **取り返される分岐で相手が得るのは元駒相当**（と金→持ち駒の歩）なので、
/// 成/不成のリスク差は本来ほぼゼロ: リスクは「持ち込んだ駒」の価値で数え、
/// 成りの付加価値は生き残った分岐（threat / promo 実現）でだけ実現させる
fn promo_risk_prerole() -> bool {
    crate::config::current(|c| c.strategy.promo_risk_prerole)
}

/// 捕獲直後の手戻り免除・退避加点（`TSUITATE_CAPTURE_RETREAT_W`、
/// **既定 0.08** = 2026-08-21 に PREROLE と対で採用。`0` で従来挙動）。
///
/// 直前に受理された手が**駒を取った移動**で、今手がその厳密な逆（from/to 入替）
/// かつ**不成**なら「取って逃げる」なので `backtrack_penalty` を免除し、
/// `w × exchange_value(着手駒)` を adjust へ加点する。
///
/// 成りを除外する理由: 捕獲→成り返り（例: 3二角成）は退避でなく再突入で、
/// 加点すると quest31-m087/m089 の 4a3b+（0点）が銀不成を押しのける
/// （実測: w=0.08 で両シナリオ 5.6→0）。発端 m024 の 3b4b は不成。
/// 粒子不要（観測の captured のみ）。**v14 はこの名前を読む**（v13 以前は知らない）。
fn capture_retreat_w() -> f64 {
    crate::config::current(|c| c.strategy.capture_retreat_w)
}

/// 捕獲直後の手戻り免除。2026-08-21 に既定 0.08 で採用（PREROLE の対）。
const CAPTURE_RETREAT_W: f64 = 0.08;

/// V1（利き数）のノブ。**既定は両方 無効**（＝従来の二値の利き判定）。
///
/// やねうら王 Lv4 の「利き数」（+R30）をついたてへ持ち込む実験だったが、
/// 200局×3形態（統合 53.3% / 紐だけ 50.0% / 圧力だけ 49.7%）がいずれも
/// 対照 56.5%±6.9 を下回り、狙いの機械指標（只取られ 1050→1113、
/// 損な交換 1934→2128）も改善しなかったため既定から外した。
/// `attack_count` 自体は V3（予防的な紐）・V2（距離重み）で使えるので残す。
/// `TSUITATE_V1_PRESSURE=1` / `TSUITATE_V1_DEFENDED=1` で再度有効化できる
fn v1_pressure_multiplicity() -> bool {
    crate::config::current(|c| c.strategy.v1_pressure_multiplicity)
}

fn v1_defended_by_count() -> bool {
    crate::config::current(|c| c.strategy.v1_defended_by_count)
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

/// **評価側の設定**（issue #21）。`TSUITATE_*` を構成境界で一度だけ解釈した
/// 結果で、strategy instance が [`crate::config::StrategyConfig`] として持つ。
///
/// フィールドは同名のアクセサ関数と1対1で、既定値・検証（範囲クリップ）は
/// アクセサの doc コメントに残してある。**プロセス env は読まない**ので、
/// 候補へノブを渡しても同じプロセスの凍結相手には効かない。
#[derive(Clone, Debug, PartialEq)]
pub struct StrategyKnobs {
    /// `taint_king_prefer_empty()` の解決値。
    pub taint_king_prefer_empty: bool,
    /// `eval_taint_fallback()` の解決値。
    pub eval_taint_fallback: bool,
    /// `eval_taint_attack_w()` の解決値。
    pub eval_taint_attack_w: f64,
    /// `link_endgame_dampen()` の解決値。
    pub link_endgame_dampen: f64,
    /// `hand_asset_w()` の解決値。
    pub hand_asset_w: f64,
    /// `king_known_approach_w()` の解決値。
    pub king_known_approach_w: f64,
    /// `promote_far_w()` の解決値。
    pub promote_far_w: f64,
    /// `king_file_pawn_w()` の解決値。
    pub king_file_pawn_w: f64,
    /// `king_file_pawn_mid_w()` の解決値。
    pub king_file_pawn_mid_w: f64,
    /// `king_endgame_flee_w()` の解決値。
    pub king_endgame_flee_w: f64,
    /// `gold_join_king_w()` の解決値。
    pub gold_join_king_w: f64,
    /// `gold_king_file_w()` の解決値。
    pub gold_king_file_w: f64,
    /// `knight_late_promo_w()` の解決値。
    pub knight_late_promo_w: f64,
    /// `knight_endgame_promo_w()` の解決値。
    pub knight_endgame_promo_w: f64,
    /// `knight_camp_exit_w()` の解決値。
    pub knight_camp_exit_w: f64,
    /// `silver_camp_exit_w()` の解決値。
    pub silver_camp_exit_w: f64,
    /// `king_file_gold_w()` の解決値。
    pub king_file_gold_w: f64,
    /// `tokin_file_drift_w()` の解決値。
    pub tokin_file_drift_w: f64,
    /// `pawn_offfile_w()` の解決値。
    pub pawn_offfile_w: f64,
    /// `far_major_promo_capture_w()` の解決値。
    pub far_major_promo_capture_w: f64,
    /// `own_camp_idle_w()` の解決値。
    pub own_camp_idle_w: f64,
    /// `bishop_retreat_w()` の解決値。
    pub bishop_retreat_w: f64,
    /// `endgame_camp_general_w()` の解決値。
    pub endgame_camp_general_w: f64,
    /// `unbacked_camp_w()` の解決値。
    pub unbacked_camp_w: f64,
    /// `unbacked_gs_capture_w()` の解決値。
    pub unbacked_gs_capture_w: f64,
    /// `belief_occ_cap_w()` の解決値。
    pub belief_occ_cap_w: f64,
    /// `home_gold_attack_w()` の解決値。
    pub home_gold_attack_w: f64,
    /// `tokin_approach_w()` の解決値。
    pub tokin_approach_w: f64,
    /// `king_adj_heavy_w()` の解決値。
    pub king_adj_heavy_w: f64,
    /// `own_camp_minor_promo_w()` の解決値。
    pub own_camp_minor_promo_w: f64,
    /// `king_cand_attack_w()` の解決値。
    pub king_cand_attack_w: f64,
    /// `king_cand_check_w()` の解決値。
    pub king_cand_check_w: f64,
    /// `king_cand_attack_gate()` の解決値。
    pub king_cand_attack_gate: usize,
    /// `landing_support_w()` の解決値。
    pub landing_support_w: f64,
    /// `king_belief_prox_w()` の解決値。
    pub king_belief_prox_w: f64,
    /// `promote_far_all()` の解決値。
    pub promote_far_all: bool,
    /// `king_cand_attack_blind_only()` の解決値。
    pub king_cand_attack_blind_only: bool,
    /// `promote_check_reveal_w()` の解決値。
    pub promote_check_reveal_w: f64,
    /// `nonpromote_check_w()` の解決値。
    pub nonpromote_check_w: f64,
    /// `nonpromote_check_p()` の解決値。
    pub nonpromote_check_p: f64,
    /// `check_king_gain_mean()` の解決値。
    pub check_king_gain_mean: bool,
    /// `check_safe_resolve_enabled()` の解決値。
    pub check_safe_resolve_enabled: bool,
    /// `gen_nonpromote_minor()` の解決値。
    pub gen_nonpromote_minor: bool,
    /// `gen_nonpromote()` の解決値。
    pub gen_nonpromote: bool,
    /// `promo_risk_prerole()` の解決値。
    pub promo_risk_prerole: bool,
    /// `capture_retreat_w()` の解決値。
    pub capture_retreat_w: f64,
    /// `v1_pressure_multiplicity()` の解決値。
    pub v1_pressure_multiplicity: bool,
    /// `v1_defended_by_count()` の解決値。
    pub v1_defended_by_count: bool,
    /// `drop_hit_all_ranks()` の解決値。
    pub drop_hit_all_ranks: bool,
    /// `promo_decay()` の解決値。
    pub promo_decay: f64,
    /// `promo_realized_floor()` の解決値。
    pub promo_realized_floor: f64,
    /// `king_prox_exclude_self()` の解決値。
    pub king_prox_exclude_self: bool,
    /// `check_foul_prior_boost()` の解決値。
    pub check_foul_prior_boost: f64,
    /// `stale_threat_w()` の解決値。
    pub stale_threat_w: f64,
    /// `blind_recapture_w()` の解決値。
    pub blind_recapture_w: f64,
    /// `blind_home_risk_w()` の解決値。
    pub blind_home_risk_w: f64,
    /// `eval_weight_cap()` の解決値。
    pub eval_weight_cap: f64,
    /// `king_repeat_foul_w()` の解決値。
    pub king_repeat_foul_w: f64,
    /// `last_foul_guard()` の解決値。
    pub last_foul_guard: f64,
    /// `last_foul_guard_2()` の解決値。
    pub last_foul_guard_2: f64,
    /// `last_foul_guard_3()` の解決値。
    pub last_foul_guard_3: f64,
    /// `blind_home_drop_occ_w()` の解決値。
    pub blind_home_drop_occ_w: f64,
    /// `blind_home_floor()` の解決値。
    pub blind_home_floor: f64,
    /// `blind_home_lambda()` の解決値。
    pub blind_home_lambda: f64,
    /// `belief_gain_w()` の解決値。
    pub belief_gain_w: f64,
    /// `king_net_w()` の解決値。
    pub king_net_w: f64,
    /// `king_net_proj()` の解決値。
    pub king_net_proj: bool,
    /// `king_probe_w()` の解決値。
    pub king_probe_w: f64,
    /// `path_probe_w()` の解決値。
    pub path_probe_w: f64,
    /// `king_sensor_w()` の解決値。
    pub king_sensor_w: f64,
    /// `sensor_p_push()` の解決値。
    pub sensor_p_push: f64,
    /// `sensor_p_promo()` の解決値。
    pub sensor_p_promo: f64,
    /// `probe_audit()` の解決値。
    pub probe_audit: bool,
    /// `probe_anchor_decay()` の解決値。
    pub probe_anchor_decay: f64,
    /// `probe_threat_w()` の解決値。
    pub probe_threat_w: f64,
    /// `drop_probe_repeat_gate()` の解決値。
    pub drop_probe_repeat_gate: bool,
    /// `depth2_focal_k()` の解決値。
    pub depth2_focal_k: usize,
    /// `threat_by_count()` の解決値。
    pub threat_by_count: bool,
    /// `nonpromote_check_roles()` の解決値。
    pub nonpromote_check_roles: String,
    /// `taint_king_fix()` の解決値。
    pub taint_king_fix: bool,
    /// 局所被覆度リスクを有効化する（`TSUITATE_ENABLE_HANG_RISK`）。
    pub enable_hang_risk: bool,
    /// 王手解消の診断出力（`TSUITATE_DEBUG_CHECK`）。
    pub debug_check: bool,
}

impl StrategyKnobs {
    pub(crate) fn from_source(src: &crate::config::EnvSource) -> Self {
        StrategyKnobs {
            taint_king_prefer_empty: { !src.var("TSUITATE_TAINT_KING_EMPTY").is_ok_and(|v| v == "0") },
            eval_taint_fallback: { src.var("TSUITATE_EVAL_TAINT_FALLBACK").is_ok_and(|v| v == "1") },
            eval_taint_attack_w: { src.var("TSUITATE_EVAL_TAINT_ATTACK")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
            .unwrap_or(0.0) },
            link_endgame_dampen: { src.var("TSUITATE_LINK_ENDGAME_DAMPEN")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(LINK_ENDGAME_DAMPEN) },
            hand_asset_w: { src.var("TSUITATE_HAND_ASSET_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(HAND_ASSET_W) },
            king_known_approach_w: { src.var("TSUITATE_KING_KNOWN_APPROACH_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(KING_KNOWN_APPROACH_W) },
            promote_far_w: { src.var("TSUITATE_PROMOTE_FAR_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(PROMOTE_FAR_W) },
            king_file_pawn_w: { src.var("TSUITATE_KING_FILE_PAWN_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(KING_FILE_PAWN_W) },
            king_file_pawn_mid_w: { src.var("TSUITATE_KING_FILE_PAWN_MID_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(KING_FILE_PAWN_MID_W) },
            king_endgame_flee_w: { src.var("TSUITATE_KING_ENDGAME_FLEE_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(KING_ENDGAME_FLEE_W) },
            gold_join_king_w: { src.var("TSUITATE_GOLD_JOIN_KING_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(GOLD_JOIN_KING_W) },
            gold_king_file_w: { src.var("TSUITATE_GOLD_KING_FILE_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(GOLD_KING_FILE_W) },
            knight_late_promo_w: { src.var("TSUITATE_KNIGHT_LATE_PROMO_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(KNIGHT_LATE_PROMO_W) },
            knight_endgame_promo_w: { src.var("TSUITATE_KNIGHT_ENDGAME_PROMO_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(KNIGHT_ENDGAME_PROMO_W) },
            knight_camp_exit_w: { src.var("TSUITATE_KNIGHT_CAMP_EXIT_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(KNIGHT_CAMP_EXIT_W) },
            silver_camp_exit_w: { src.var("TSUITATE_SILVER_CAMP_EXIT_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(SILVER_CAMP_EXIT_W) },
            king_file_gold_w: { src.var("TSUITATE_KING_FILE_GOLD_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(KING_FILE_GOLD_W) },
            tokin_file_drift_w: { src.var("TSUITATE_TOKIN_FILE_DRIFT_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(TOKIN_FILE_DRIFT_W) },
            pawn_offfile_w: { src.var("TSUITATE_PAWN_OFFFILE_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(PAWN_OFFFILE_W) },
            far_major_promo_capture_w: { src.var("TSUITATE_FAR_MAJOR_PROMO_CAPTURE_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(FAR_MAJOR_PROMO_CAPTURE_W) },
            own_camp_idle_w: { src.var("TSUITATE_OWN_CAMP_IDLE_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(OWN_CAMP_IDLE_W) },
            bishop_retreat_w: { src.var("TSUITATE_BISHOP_RETREAT_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(BISHOP_RETREAT_W) },
            endgame_camp_general_w: { src.var("TSUITATE_ENDGAME_CAMP_GENERAL_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(ENDGAME_CAMP_GENERAL_W) },
            unbacked_camp_w: { src.var("TSUITATE_UNBACKED_CAMP_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(UNBACKED_CAMP_W) },
            unbacked_gs_capture_w: { src.var("TSUITATE_UNBACKED_GS_CAPTURE_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(UNBACKED_GS_CAPTURE_W) },
            belief_occ_cap_w: { src.var("TSUITATE_BELIEF_OCC_CAP_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(BELIEF_OCC_CAP_W) },
            home_gold_attack_w: { src.var("TSUITATE_HOME_GOLD_ATTACK_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(HOME_GOLD_ATTACK_W) },
            tokin_approach_w: { src.var("TSUITATE_TOKIN_APPROACH_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(TOKIN_APPROACH_W) },
            king_adj_heavy_w: { src.var("TSUITATE_KING_ADJ_HEAVY_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(KING_ADJ_HEAVY_W) },
            own_camp_minor_promo_w: { src.var("TSUITATE_OWN_CAMP_MINOR_PROMO_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(OWN_CAMP_MINOR_PROMO_W) },
            king_cand_attack_w: { src.var("TSUITATE_KING_CAND_ATTACK_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(KING_CAND_ATTACK_W) },
            king_cand_check_w: { src.var("TSUITATE_KING_CAND_CHECK_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(KING_CAND_CHECK_W) },
            king_cand_attack_gate: { src.var("TSUITATE_KING_CAND_ATTACK_GATE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(20) },
            landing_support_w: { src.var("TSUITATE_LANDING_SUPPORT_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(LANDING_SUPPORT_W) },
            king_belief_prox_w: { src.var("TSUITATE_KING_BELIEF_PROX_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(KING_BELIEF_PROX_W) },
            promote_far_all: { src.var("TSUITATE_PROMOTE_FAR_ALL").is_ok_and(|v| v == "1") },
            king_cand_attack_blind_only: { src.var("TSUITATE_KING_CAND_ATTACK_BLIND").is_ok_and(|v| v == "1") },
            promote_check_reveal_w: { src.var("TSUITATE_PROMOTE_CHECK_REVEAL_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(PROMOTE_CHECK_REVEAL_W) },
            nonpromote_check_w: { src.var("TSUITATE_NONPROMOTE_CHECK_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(0.0) },
            nonpromote_check_p: { src.var("TSUITATE_NONPROMOTE_CHECK_P")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(0.2) },
            check_king_gain_mean: { src.var("TSUITATE_CHECK_KING_GAIN_MEAN").map_or(true, |v| v != "0") },
            check_safe_resolve_enabled: { src.var("TSUITATE_CHECK_SAFE_RESOLVE").is_ok_and(|v| v != "0") },
            gen_nonpromote_minor: { src.var("TSUITATE_GEN_NONPROMOTE").is_ok_and(|v| v == "minor") },
            gen_nonpromote: { src.var("TSUITATE_GEN_NONPROMOTE").is_ok_and(|v| v == "1") },
            promo_risk_prerole: { src.var("TSUITATE_PROMO_RISK_PREROLE").map_or(true, |v| v == "1") },
            capture_retreat_w: { src.var("TSUITATE_CAPTURE_RETREAT_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(CAPTURE_RETREAT_W) },
            v1_pressure_multiplicity: { src.var("TSUITATE_V1_PRESSURE").is_ok_and(|v| v == "1") },
            v1_defended_by_count: { src.var("TSUITATE_V1_DEFENDED").is_ok_and(|v| v == "1") },
            drop_hit_all_ranks: { !src.var("TSUITATE_DROP_HIT_ALL_RANKS").is_ok_and(|v| v == "0") },
            promo_decay: { src.var("TSUITATE_PROMO_DECAY")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| (0.1..=1.0).contains(v))
            .unwrap_or(PROMO_POTENTIAL_DECAY) },
            promo_realized_floor: { src.var("TSUITATE_PROMO_REALIZED_FLOOR")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| (0.0..=1.0).contains(v))
            .unwrap_or(0.0) },
            king_prox_exclude_self: { src.var("TSUITATE_KING_PROX_EXCLUDE_SELF")
            .ok()
            .map(|v| v != "0")
            .unwrap_or(true) },
            check_foul_prior_boost: { src.var("TSUITATE_CHECK_FOUL_PRIOR_BOOST")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|v: &f64| v.is_finite() && *v >= 0.0)
            .unwrap_or(3.0) },
            stale_threat_w: { src.var("TSUITATE_STALE_THREAT_W")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v: &f64| v.is_finite() && (0.0..=1.0).contains(v))
            .unwrap_or(0.0) },
            blind_recapture_w: { src.var("TSUITATE_BLIND_RECAPTURE_W")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(BLIND_RECAPTURE_W) },
            blind_home_risk_w: { src.var("TSUITATE_BLIND_HOME_RISK_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(0.0) },
            eval_weight_cap: { src.var("TSUITATE_EVAL_WEIGHT_CAP")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
            .unwrap_or(1.0) },
            king_repeat_foul_w: { src.var("TSUITATE_KING_REPEAT_FOUL_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(KING_REPEAT_FOUL_W) },
            last_foul_guard: { src.var("TSUITATE_LAST_FOUL_GUARD")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(LAST_FOUL_GUARD) },
            last_foul_guard_2: { src.var("TSUITATE_LAST_FOUL_GUARD_2")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(LAST_FOUL_GUARD_2) },
            last_foul_guard_3: { src.var("TSUITATE_LAST_FOUL_GUARD_3")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(LAST_FOUL_GUARD_3) },
            blind_home_drop_occ_w: { src.var("TSUITATE_BLIND_HOME_DROP_OCC_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
            .unwrap_or(0.0) },
            blind_home_floor: { src.var("TSUITATE_BLIND_HOME_FLOOR")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
            .unwrap_or(0.2) },
            blind_home_lambda: { src.var("TSUITATE_BLIND_HOME_LAMBDA")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(0.045) },
            belief_gain_w: { src.var("TSUITATE_BELIEF_GAIN_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(0.0) },
            king_net_w: { src.var("TSUITATE_KING_NET_W")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite())
            .map(|v| v.clamp(0.0, 1.0))
            .unwrap_or(0.0) },
            king_net_proj: { !src.var("TSUITATE_KING_NET_PROJ").is_ok_and(|v| v == "0") },
            king_probe_w: { probe_env(src, "TSUITATE_KING_PROBE_W", 0.0) },
            path_probe_w: { probe_env(src, "TSUITATE_PATH_PROBE_W", 0.0) },
            king_sensor_w: { probe_env(src, "TSUITATE_KING_SENSOR_W", 0.0) },
            sensor_p_push: { probe_env(src, "TSUITATE_SENSOR_P_PUSH", 0.3).min(1.0) },
            sensor_p_promo: { probe_env(src, "TSUITATE_SENSOR_P_PROMO", 0.8).min(1.0) },
            probe_audit: { src.var("TSUITATE_PROBE_AUDIT").is_ok_and(|v| v == "1") },
            probe_anchor_decay: { probe_env(src, "TSUITATE_PROBE_ANCHOR_DECAY", 0.5).clamp(0.0, 1.0) },
            probe_threat_w: { probe_env(src, "TSUITATE_PROBE_THREAT_W", 0.0) },
            drop_probe_repeat_gate: { src.var("TSUITATE_DROP_PROBE_REPEAT_GATE").is_ok_and(|v| v == "1") },
            depth2_focal_k: { src.var("TSUITATE_DEPTH2_FOCAL_K")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0) },
            threat_by_count: { src.var("TSUITATE_THREAT_BY_COUNT").is_ok_and(|v| v == "1") },
            nonpromote_check_roles: {
                src.var("TSUITATE_NONPROMOTE_CHECK_ROLES")
                    .unwrap_or_else(|_| "minor".into())
            },
            taint_king_fix: !src.var("TSUITATE_TAINT_KING_FIX").is_ok_and(|v| v == "0"),
            enable_hang_risk: src.var("TSUITATE_ENABLE_HANG_RISK").is_ok(),
            debug_check: src.var("TSUITATE_DEBUG_CHECK").is_ok(),
        }
    }
}

#[cfg(test)]
fn coverage_after(view: &PlayerView, mv: &ShogiMove) -> f64 {
    own_effects_after(view, mv, None, None, &EvalParams::default()).coverage
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
    promo_prox: Option<&[f64; 81]>,
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
        ShogiMove::Board {
            from,
            promote: true,
            ..
        } => view
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

    // 打つ手なら打ったマス（promo_potential の旧価格付け対象）
    let dropped_at = match *mv {
        ShogiMove::Drop { to, .. } => Some(to),
        _ => None,
    };
    OwnEffects {
        coverage: covered.len() as f64,
        king_holes,
        linked_value,
        effect_own,
        effect_opp,
        drop_hit_exposure: drop_hit_exposure(&pieces, view.your_color),
        promo_potential: if params.promo_potential_w != 0.0 && !view.you_in_check {
            promo_potential(&pieces, view.your_color, promo_prox, dropped_at)
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
    crate::config::current(|c| c.strategy.drop_hit_all_ranks)
}

/// 成りポテンシャルの手数減衰（既定: 1手遠いごとに半減）。
/// `TSUITATE_PROMO_DECAY` で上書き可（スイープ経路。0.5〜0.9 が実用域）。
/// ついたて将棋では静かな前進は相手から観測されず妨害されにくいので、
/// 通常将棋の感覚より高い生存率（緩い減衰）が正当化されうる
/// （「静かな準備の手の複数手先の価値」2026-08-06）
fn promo_decay() -> f64 {
    crate::config::current(|c| c.strategy.promo_decay)
}
/// 成りポテンシャルの手数減衰の既定値
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
/// 回避）。自駒だけで決まるので粒子不要・ノイズゼロ。
///
/// `king_prox` は**敵玉候補集合への近接重み**（`promo_king_prox_map`、既定 None =
/// 従来挙動）。quest31-m083 が発端: 実現ボーナスが方向を見ないため、敵玉
/// （王手履歴からほぼ確定）から遠ざかる 3二角成 が、正解の 7六歩 を
/// promo 差 +0.36 で上回っていた。ユーザーの原則「4三のと金は相手玉から
/// 遠すぎる。突くなら7・8筋」を、演繹（deduce::opp_king_candidates =
/// 観測のみ由来でノイズゼロ）への近さで実装する。差分の両側が同じマップを
/// 使うので差分形の性質は保たれる
fn promo_potential(
    pieces: &[VisiblePiece],
    me: Color,
    king_prox: Option<&[f64; 81]>,
    // 直前の手で打たれたマス（Drop の差分評価時のみ Some）。そこにいる駒は
    // **旧価格**（既定 decay 0.5・prox なし）で値付けする: 打つ手は持ち駒を
    // 消費するので、行進の価格改定（decay 緩和×prox）で釣り上げると
    // 垂れ歩が常用化する（実測: decay0.8/w0.5 で P*2f が 19/20。
    // occasional-probe「たまに」方針の保護）
    dropped_at: Option<Coord>,
) -> f64 {
    let occupied: HashSet<Coord> = pieces
        .iter()
        .filter_map(|p| parse_usi_square(&p.square))
        .collect();
    let mut total = 0.0f64;
    for p in pieces {
        let Some(origin) = parse_usi_square(&p.square) else {
            continue;
        };
        if dropped_at == Some(origin) {
            // 旧 decay（緩和なし）＋ prox は適用。prox まで免除すると
            // 盤上の駒だけが方向割引されて打ちが相対的に浮く
            // （実測: P*2f が 16〜20/20 に再浮上）
            total += piece_promo_potential_with(
                &occupied,
                origin,
                p.role,
                me,
                king_prox,
                PROMO_POTENTIAL_DECAY,
            );
            continue;
        }
        if unpromote_role(p.role) != p.role {
            // 成り済み: 実現した利き増加（成る手の差分を正にするための対）。
            // 近接重みは**今いるマス**で掛けるが、実現分は現物なので
            // 床 `promo_realized_floor()` より下へは割り引かない
            // （床なしだと「今できる成り捕獲」まで方向割引で沈み、
            // 無意味な歩突きに負ける。実測: m016 の 7三歩 20/20）
            let prox = king_prox
                .map_or(1.0, |m| m[crate::belief_features::sq_index(origin)])
                .max(promo_realized_floor());
            total +=
                prox * promo_effect_gain(&occupied, origin, unpromote_role(p.role), origin, me);
        } else {
            // 未成: 近接重みは**成りマス**で掛ける（piece_promo_potential 内）。
            // 行進中の歩は現在地が玉から遠くても成り先が玉の隣なら満額に近い
            total += piece_promo_potential(&occupied, origin, p.role, me, king_prox);
        }
    }
    total
}

/// 実現済みの成りの prox 床（`TSUITATE_PROMO_REALIZED_FLOOR`、既定 0 = 床なし）
fn promo_realized_floor() -> f64 {
    crate::config::current(|c| c.strategy.promo_realized_floor)
}

/// 残存する相手駒（玉除く）の平均交換価値（`foul_occ_attack_w` の素材）。
/// 観測のみで決まる: 初期の19枚から自分が取った駒種を引く
fn mean_remaining_opp_value(log: &ObservationLog) -> f64 {
    use Role::*;
    let mut remaining: Vec<Role> = vec![
        Pawn, Pawn, Pawn, Pawn, Pawn, Pawn, Pawn, Pawn, Pawn, Lance, Lance, Knight, Knight, Silver,
        Silver, Gold, Gold, Bishop, Rook,
    ];
    for e in log.events() {
        if let Observation::MyMove {
            captured: Some(role),
            ..
        } = e
        {
            let base = unpromote_role(*role);
            if let Some(pos) = remaining.iter().position(|&r| r == base) {
                remaining.swap_remove(pos);
            }
        }
    }
    if remaining.is_empty() {
        0.0
    } else {
        remaining.iter().map(|&r| exchange_value(r)).sum::<f64>() / remaining.len() as f64
    }
}

/// `promo_king_prox` の近接マップ。マスごとに `(1-w) + w × 1/(1+d_min)`
/// （d_min = 敵玉候補集合への最小チェビシェフ距離）。候補集合が空なら None
/// （= 重みなし）。deduce 由来なので粒子不要・ノイズゼロ
fn promo_king_prox_map(w: f64, cands: &std::collections::BTreeSet<Coord>) -> Option<[f64; 81]> {
    if cands.is_empty() {
        return None;
    }
    let mut map = [1.0f64; 81];
    for (i, slot) in map.iter_mut().enumerate() {
        // belief_features::sq_index の逆写像（(file-1)*9 + (rank-1)）
        let sq = Coord {
            file: (i / 9) as i8 + 1,
            rank: (i % 9) as i8 + 1,
        };
        let d = cands.iter().map(|&k| cheb(sq, k)).min().unwrap_or(8);
        *slot = (1.0 - w) + w / (1.0 + d as f64);
    }
    Some(map)
}

/// `king_cand_attack_w` の近接マップ。マスごとに `Σ_k 0.5^cheb(sq,k) / |cands|`。
///
/// 候補集合全体の平均なので、候補が1点に絞れているときは隣接で 0.5・
/// 2マス離れで 0.25 と急峻に落ち、候補が広いほど全マスで平坦になる
/// （＝情報が無いときは自然に効かなくなる）
/// 近接マップから**着地マス自身**（距離0）を除くか
/// （`TSUITATE_KING_PROX_EXCLUDE_SELF`、**既定 on**、`0` で従来挙動）。
///
/// 着手できたということは玉はそのマスに居ない（打ちなら反則、移動なら玉は
/// 取れない）ので、距離0の項は「玉に近い」根拠にならない。従来版は玉候補
/// マスそのものへの垂れ歩に最大加点を与えていた（quest31-m119 の P*5b、
/// ユーザー採点0）。`king_cand_attack_w` が既定 0 のときは得点中立だったが、
/// 接近ボーナスを既定オンにしたので、距離0を残すと P*5b 型が最大加点を
/// 受ける。v13 以前の凍結版はこの名前を知らない（v14 は読む）。
fn king_prox_exclude_self() -> bool {
    crate::config::current(|c| c.strategy.king_prox_exclude_self)
}

fn king_cand_prox_map(cands: &std::collections::BTreeSet<Coord>) -> [f64; 81] {
    let mut map = [0.0f64; 81];
    for (i, slot) in map.iter_mut().enumerate() {
        let sq = Coord {
            file: (i / 9) as i8 + 1,
            rank: (i % 9) as i8 + 1,
        };
        // 着地マス自身（距離0）は数えない: そこへ着手できたということは
        // **玉はそこに居ない**（打ちなら反則、移動なら玉は取れない）ので、
        // 「玉に近い」根拠にならない。数えていた版は玉候補マスそのものへの
        // 無意味な垂れ歩（quest31-m119 の P*5b、採点0）へ最大加点を与えていた
        *slot = cands
            .iter()
            .filter(|&&k| !king_prox_exclude_self() || k != sq)
            .map(|&k| 0.5f64.powi(i32::from(cheb(sq, k))))
            .sum::<f64>();
    }
    // 最大が 1 になるよう正規化する（候補が散っているほど生の値は小さいので、
    // 正規化しないと w の意味が候補集合の広さで変わる）。**最短距離でなく
    // 候補全体の和**を使うのは、候補が固まっている側（玉が居そうな一帯）を
    // 素直に重くしたいから
    let max = map.iter().cloned().fold(0.0f64, f64::max);
    if max > 0.0 {
        for slot in map.iter_mut() {
            *slot /= max;
        }
    }
    map
}

/// 玉位置分布からの近接マップ（`king_belief_prox_w`）。`king_cand_prox_map` の
/// 確率重み版で、こちらも盤の最大で正規化する
fn king_dist_prox_map(dist: &[(Coord, f64)]) -> [f64; 81] {
    let mut map = [0.0f64; 81];
    for (i, slot) in map.iter_mut().enumerate() {
        let sq = Coord {
            file: (i / 9) as i8 + 1,
            rank: (i % 9) as i8 + 1,
        };
        // 着地マス自身は数えない（`king_cand_prox_map` と同じ理由）
        *slot = dist
            .iter()
            .filter(|&&(k, _)| !king_prox_exclude_self() || k != sq)
            .map(|&(k, p)| p * 0.5f64.powi(i32::from(cheb(sq, k))))
            .sum::<f64>();
    }
    let max = map.iter().cloned().fold(0.0f64, f64::max);
    if max > 0.0 {
        for slot in map.iter_mut() {
            *slot /= max;
        }
    }
    map
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
/// 「Δ利き × 減衰^(成りマスまでの手数) × 成りマスの敵玉近接重み」。
/// `promo_potential()` の1駒ぶんと、持ち駒オプション（`hand_option_w`）の
/// 打ちマス実現値の共通部品。`king_prox` は None なら重みなし
fn piece_promo_potential(
    occupied: &HashSet<Coord>,
    at: Coord,
    role: Role,
    me: Color,
    king_prox: Option<&[f64; 81]>,
) -> f64 {
    // 減衰の緩和（TSUITATE_PROMO_DECAY）は**歩だけ**に適用する。歩の行進は
    // 正面の駒にしか止められない低リスクの前進だが、桂銀香の進出は被捕獲
    // リスクが高く緩い減衰の根拠が無い（実測: decay0.8 で 8一桂の跳び道を
    // 玉が開ける手に candy が乗り、m076 の玉捕獲が 0→20 に再発した）
    let decay = if role == Role::Pawn {
        promo_decay()
    } else {
        PROMO_POTENTIAL_DECAY
    };
    piece_promo_potential_with(occupied, at, role, me, king_prox, decay)
}

/// 減衰を明示指定する版（打たれた駒の旧価格付け用。`promo_potential` の doc 参照）
fn piece_promo_potential_with(
    occupied: &HashSet<Coord>,
    at: Coord,
    role: Role,
    me: Color,
    king_prox: Option<&[f64; 81]>,
    decay: f64,
) -> f64 {
    if promote_role(role).is_none() {
        return 0.0;
    }
    match promo_distance(occupied, at, role, me) {
        Some((d, sq)) => {
            let prox = king_prox.map_or(1.0, |m| m[crate::belief_features::sq_index(sq)]);
            prox * promo_effect_gain(occupied, at, role, sq, me) * decay.powi(d as i32)
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
            .map(|s| piece_promo_potential(&occupied, s, role, me, None))
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
fn count_effects(
    occupied: &HashSet<Coord>,
    vacated: Coord,
    role: Role,
    at: Coord,
    me: Color,
) -> f64 {
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
#[cfg(test)]
fn king_holes_after(view: &PlayerView, mv: &ShogiMove) -> f64 {
    own_effects_after(view, mv, None, None, &EvalParams::default()).king_holes
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
    /// gain に加算された自玉近傍の敵駒排除ボーナス（own_zone_capture_w。
    /// 粒子加重平均、内訳表示用）
    own_zone: f64,
    /// 打ちプローブの反則情報価値（drop_probe_w）。反則の失敗枝の期待値なので
    /// gain には含めず、combine_score の外側（(1−p_legal) 側）で加算する
    foul_probe: f64,
    /// 監査用（`CandidateScore::probe_unit` の doc）
    probe_unit: f64,
    probe_mass: f64,
    probe_concentration: f64,
    /// `adjust` に含まれるタイブレーク乱数（`rng.random_range(0.0..0.01)`）。
    /// evaluate() の外（候補ループ）で引いて書き戻す。学習用のエクスポートが
    /// 「乱数を除いた adjust」を出すために分離して持つ（`CandidateScore::tiebreak`）
    tiebreak: f64,
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
pub fn combine_score(gain: f64, p_legal: f64, foul_cost: f64) -> f64 {
    (p_legal * gain).min(gain) - (1.0 - p_legal) * foul_cost
}

/// 粒子の合法性投票と事前確率のブレンド（`p_legal` の本体）。
///
/// `legal` = その手が合法だった粒子の重み和 / `n` = 粒子の重み和 /
/// `prior` = 観測ゼロの事前確率（王手中は CheckSolver の解消確率を掛けたもの）。
/// 粒子が退化している（実効重みが評価上限 `eval_particles` に届かない）ほど
/// 事前の重みを増やし、少数の偏った粒子への過信を防ぐ。
///
/// **`evaluate` と、issue #31 P0-5 の p-only shadow update が同じ式を使う**ため
/// に切り出してある（別々に書くと「仮想更新が実再決定とずれた」のか
/// 「式が食い違っていた」のかを分けられない）。ここは抽出しただけで
/// 挙動は変わらない。
pub fn blend_p_legal(
    legal: f64,
    n: f64,
    prior: f64,
    particles_are_taint: bool,
    eval_particles: usize,
    params: &EvalParams,
) -> f64 {
    if particles_are_taint {
        // taint 粒子は「反則の説明」を持たないので合法性の投票には使わない
        return prior;
    }
    let degen = 1.0 - (n / eval_particles as f64).min(1.0);
    let w = params.prior_weight + params.prior_weight_degen * degen;
    (legal + prior * w) / (n + w)
}

/// 思考予算 `ms` のときの評価粒子数の上限（`blend_p_legal` の退化度の分母）。
///
/// 診断（issue #31 P0-5）が `evaluate` と同じ退化度を再現するために公開する。
pub fn eval_particles_for_budget(ms: u64) -> usize {
    SearchBudget::from_ms(ms).eval_particles
}

/// 王手中の反則1回あたりの `prior_weight` 倍率（診断が同じ boost を掛けるため）。
///
/// `choose` は `foul_tried` の本数だけこの倍率を `prior_weight` /
/// `prior_weight_degen` へ掛ける（2026-08-20 の「被王手の確かめの経済」）。
pub fn check_foul_prior_factor(fouls_this_turn: usize) -> f64 {
    1.0 + check_foul_prior_boost() * fouls_this_turn as f64
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
    /// **錨外し**（`evaluate` の anchor_move_pen 参照、2026-08-04）。
    /// 争点マスを支えている自駒自身をそこへ動かす手へ
    /// `w × 交換価値 ÷ (1 + 着手後の残り守り枚数)` の減点。0 = 従来挙動。
    /// 実用域は 0.2〜0.6（w=0.4 で金の単独錨外しが約2歩ぶん）
    pub anchor_move_w: f64,
    /// **玉で取る手の露見実効価値**（2026-08-04、quest31-m076 が発端）。
    /// 取ったマスは相手に通知されるので取り手の位置は露見するが、玉は
    /// `exchange_value=0` のため `capture_reveal_risk` の床と `blind_recapture` の
    /// 露見コストが**完全に免除**され、ブラインド決定（粒子全滅で expected が
    /// 消えた状態）では玉が「最も安全な取り手」に化ける（実測: 銀でも取れる
    /// 6二の駒を玉で取る手が 12/20 で首位。ユーザー判定は「普通は銀で取った
    /// 方が良い」）。露見するのが玉自身のときは交換価値の代わりにこの実効価値を
    /// 使う（0 = 従来挙動）。玉の位置露見は材料でなく詰みリスクなので
    /// 実用域は大駒級の 5〜12
    pub king_capture_reveal: f64,
    /// **成りポテンシャルの敵玉近接重み**（2026-08-05、quest31-m083 の
    /// 3二角成が発端）。`promo_potential` の各駒寄与に
    /// `(1-w) + w × 1/(1+d_min)`（d_min = `deduce::opp_king_candidates` への
    /// 最小チェビシェフ距離）を掛ける。促成の実現ボーナスが方向を見ないため、
    /// 敵玉から遠ざかる成り（3二角成）が正解の玉方向の手（7六歩）を上回っていた。
    /// ユーザー原則「と金は相手玉の近くに作る。突くなら7・8筋」の実装。
    /// deduce 由来なので粒子不要・ノイズゼロ（promo_potential の設計を保つ）。
    /// 0 = 従来挙動。w=1 で「玉候補の隣 1/2 vs 8マス先 1/9」
    pub promo_king_prox: f64,
    /// **この手番の打ち反則で確定した駒への当たり**（2026-08-07、quest31-m090f1
    /// が発端）。自分の打ちが反則になったマスには相手駒がいることが100%確定し、
    /// しかも反則では手番が変わらないので**情報は今この瞬間まで新鮮**。
    /// なのに従来はそのマスへ利きを付ける手（実戦の人間: 7八歩打の反則 →
    /// 7七歩打で確定駒に当てる）に何の価値も付かず、73〜81位に沈んでいた。
    /// drop_probe_w（プローブ = 情報を買う）の**回収側**。
    /// 着手駒（玉以外）が着手後にそのマスへ利きを持つ手へ
    /// `w × 残存敵駒の平均交換価値` を gain 内（p_legal 割引の内側）へ加点。
    /// 観測のみ由来・粒子不要。王手中は無効。0 = 従来挙動
    pub foul_occ_attack_w: f64,
    /// **材料の退化ゲート**（既定 0 = 従来と同一挙動。quest31 コンボ計測時の
    /// 作業点は 0.3）。駒得の期待値を粒子質量の
    /// 薄さで縮める: g = c(1+q0)/(c+q0)（c = confidence）。少数の生存粒子は
    /// 「自信を持って間違う」ため（実測 2026-08-10: 厳密粒子9個の決定点で
    /// 真実が空きマスの4一に飛車85.9%の信念、そこへの4一成桂が浮く）。
    ///
    /// **縮めるのは観測裏付けの無い捕獲だけ**（2026-08-10 の全捕獲版は
    /// m032/m063 など正しい捕獲まで殺し不採用。次に試すなら「裏付け無し」に
    /// 絞れ、という教訓どおり）。裏付け = 相手が自駒を取ったマス（居場所が
    /// 通知される）＋この手番の非歩打ち反則マス。幻の 3三角成 / 3二角成 /
    /// 7七桂成クラスを沈める一方、観測確実な取り返しは満額残す。
    /// `TSUITATE_MATERIAL_DEGEN_Q0` で上書き可・SPSA対応。凍結版は知らない
    pub material_degen_q0: f64,
    /// **自玉近傍の敵駒の排除**（2026-08-20、arena-recap01 が発端。ユーザー
    /// 「リスクを恐れすぎて逆にリスクを作っている。玉の近くの敵駒を排除する
    /// ことに加点する項を作っていい」）。粒子上で自玉のチェビシェフ距離 ≤2 に
    /// いる相手駒を取る手へ、交換価値とは別に
    /// w × 距離(1:1.0/2:0.5) × 駒種危険度(歩香桂0.25/金銀成小駒0.75/大駒1.0)
    /// × 観測裏付け(あり1.0/なし0.5) を加点（codex 相談 2026-08-20 の3係数化。
    /// 粒子重み付きなので占有確率も暗黙に掛かる）。`defender_capture_w`
    /// （相手玉の守り駒）の自玉側の鏡像。王手中は無効（CheckSolver の領分）。
    /// **既定 10.0**（2026-08-20 採用。確定 suite +0.158・アリーナ2シードで
    /// vs v13 57.2%/vs v14 56.2% と両基準対照超え）。
    /// `TSUITATE_OWN_ZONE_CAPTURE_W` で上書き可（0 で切り戻し）・SPSA対応。
    /// 凍結版は知らない
    pub own_zone_capture_w: f64,
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
            promote_bias: 0.4,
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
            // （CLAUDE.md: 未検証。既定オンにしない）
            taint_occ_legal_w: 0.0,
            // 大駒の成り道（2026-08-03）。0 = 従来と同一挙動
            major_promo_path_w: 0.0,
            exposed_multi_w: 0.0,
            exposed_pawn_head_w: 0.0,
            // ブラインド玉攻めの生存割引（2026-08-03 実装、2026-08-05 採用）。
            // 0 で従来挙動へ切り戻し
            blind_attack_survive_w: 1.0,
            anchor_move_w: 0.0,
            // 玉で取る手の露見実効価値（2026-08-04 実装、2026-08-05 採用）。
            // 0 で従来挙動へ切り戻し。w=10 × capture_reveal_risk ≒ 1.3点の
            // リスク床で、玉でしか取れない駒の捕獲は gain が勝って生き残る
            king_capture_reveal: 10.0,
            // 成りポテンシャルの敵玉近接重み。既定 0（2026-08-06 不採用）
            promo_king_prox: 0.0,
            // 打ち反則で確定した駒への当たり（2026-08-07 実装、2026-08-08 採用）。
            // 0 で従来挙動へ切り戻し。drop_probe_w（情報を買う）の回収側
            foul_occ_attack_w: 2.0,
            // 材料の退化ゲート。既定 0（2026-08-10: 正しい捕獲まで殺す）
            material_degen_q0: 0.0,
            own_zone_capture_w: 10.0,
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
    pub const SPECS: [ParamSpec; 71] = [
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
        ParamSpec {
            // 交換価値（3.5〜12）に掛かる。w=1 で「単独の錨を外す手」が
            // 駒価値まるごとの減点になるので実用域は 1 未満
            name: "anchor_move_w",
            lo: 0.0,
            hi: 1.5,
        },
        ParamSpec {
            // 駒価値スケール（capture_reveal_risk の床に乗る）。龍=12 が上限
            name: "king_capture_reveal",
            lo: 0.0,
            hi: 12.0,
        },
        ParamSpec {
            // (1-w) + w/(1+d) のブレンド比なので 0〜1
            name: "promo_king_prox",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            // 残存敵駒の平均交換価値（約3）に掛かる。w=1.5 で銀1枚ぶん相当
            name: "foul_occ_attack_w",
            lo: 0.0,
            hi: 3.0,
        },
        ParamSpec {
            // 材料の退化ゲートの半減点（confidence のスケール）。0 = 無効
            name: "material_degen_q0",
            lo: 0.0,
            hi: 1.0,
        },
        ParamSpec {
            // 自玉近傍（距離≤2）の敵駒を取る手への加点。0 = 無効。
            // hi は3係数化（dist×role×backed、最大でも 1.0×1.0×1.0）後の実効値
            // で決めてある: w=16 で大駒@距離1 が 16、成駒@距離2 が 6
            name: "own_zone_capture_w",
            lo: 0.0,
            hi: 16.0,
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
            self.anchor_move_w,
            self.king_capture_reveal,
            self.promo_king_prox,
            self.foul_occ_attack_w,
            self.material_degen_q0,
            self.own_zone_capture_w,
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
            anchor_move_w: v[65],
            king_capture_reveal: v[66],
            promo_king_prox: v[67],
            foul_occ_attack_w: v[68],
            material_degen_q0: v[69],
            own_zone_capture_w: v[70],
        }
    }
}

/// **診断専用**の粒子スナップショット（issue #28 P0-3 / P0-6）。
///
/// `choose` が実際に評価へ渡したプールそのもの:
/// `strict` は `stratified_sample` の戻り（物理整合の粒子だけ・評価重みつき）、
/// `taint` は物理制約を緩めた粒子のプール（厳密が全滅したときに評価が落ちる先）。
///
/// **同じ seed でもう一度 `Estimator` を作り直しても同じ集合にはならない**
/// （`update` は壁時計デッドラインまで若返らせるので、実行のたびに粒子数が
/// 揺れる）。「ランキングを作ったのと同じ粒子」で危険質量を測るには、
/// 評価が見たプールをそのまま持ち出すしかない。
#[derive(Clone, Debug, Default)]
pub struct ParticleSnapshot {
    pub strict: Vec<(Position, f64)>,
    pub taint: Vec<(Position, f64)>,
}

impl ParticleSnapshot {
    /// `(局面, 重み, 厳密か)` の並び（診断側が扱いやすい形）
    pub fn entries(&self) -> Vec<(&Position, f64, bool)> {
        self.strict
            .iter()
            .map(|(p, w)| (p, *w, true))
            .chain(self.taint.iter().map(|(p, w)| (p, *w, false)))
            .collect()
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
    /// **診断専用**（既定 false）: 直近の choose が評価に使った粒子プールを
    /// 保存する。off のときはコピーもしないので実対局・アリーナの挙動と
    /// コストは完全に不変（issue #28 P0-3 / P0-6 が「ランキングを作ったのと
    /// 同じ粒子」で危険質量を測るために使う）
    capture_particles: bool,
    last_particles: Option<ParticleSnapshot>,
    /// **この instance の設定**（issue #21）。trait メソッドの入口で
    /// `crate::config::scoped` に設置するので、同じプロセスの別 instance
    /// （凍結相手・別 arm の候補）とは混ざらない
    config: Arc<crate::config::StrategyConfig>,
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

    /// シードつきで作る（SPSA の f+/f− 評価で対局条件を揃える共通乱数法用）。
    /// 設定は ambient（プロセス env を一度だけ解釈したもの）を使う。
    pub fn with_params_line_seed(
        params: EvalParams,
        book_line: Option<usize>,
        seed: Option<u64>,
    ) -> Self {
        Self::with_config(crate::config::ambient(), params, book_line, seed)
    }

    /// **設定を明示して作る**（issue #21 の入口）。arena / checkpoint arena が
    /// arm ごとに別の設定を渡すのはここ。プロセス env は触らないので、
    /// 同じプロセスで走る凍結相手は既定値のまま動く。
    pub fn with_config(
        config: Arc<crate::config::StrategyConfig>,
        params: EvalParams,
        book_line: Option<usize>,
        seed: Option<u64>,
    ) -> Self {
        let params = apply_param_overrides(params, &config.source);
        EstimatorStrategy {
            est: None,
            book: None,
            book_line,
            params,
            budget: SearchBudget::from_ms(config.think_budget_ms),
            seed,
            rng: match seed {
                Some(s) => StdRng::seed_from_u64(s ^ 0xA5A5_5A5A_DEAD_BEEF),
                None => StdRng::seed_from_u64(rand::rng().random()),
            },
            last_debug: None,
            last_ranking: None,
            capture_particles: false,
            last_particles: None,
            config,
        }
    }

    /// この instance の設定（記録・検査用）。
    pub fn config(&self) -> &Arc<crate::config::StrategyConfig> {
        &self.config
    }

    /// 注入されたシード（None = エントロピー由来）。テスト・診断用。
    pub fn seed(&self) -> Option<u64> {
        self.seed
    }
}

/// TSUITATE_*_W 系 env による EvalParams の上書き（運用ノブ。デプロイ時の
/// 切り戻し・w スイープ用）。SPSA（with_params 経由）でも env が設定されて
/// いればそちらを優先する。bin/tune はこの関数で「調整対象ノブが env に
/// 潰されていないか」を起動時に検査する
pub fn apply_env_param_overrides(params: EvalParams) -> EvalParams {
    apply_param_overrides(params, &crate::config::current_config().source)
}

/// `EvalParams` の上書きを **config から** 当てる（issue #21）。
/// env はここでは読まない: 呼び出し側が [`crate::config::EnvSource`] を
/// 明示的に渡すので、候補と凍結相手へ別々の値を安全に与えられる。
pub fn apply_param_overrides(params: EvalParams, src: &crate::config::EnvSource) -> EvalParams {
    // 実現成りの固定バイアス（既存SPSA項のスイープ・切り戻し用）
    let params = match src.var("TSUITATE_PROMOTE_BIAS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite())
    {
        Some(w) => EvalParams {
            promote_bias: w,
            ..params
        },
        None => params,
    };
    // valueネット重みの運用ノブ（デプロイ時の切り戻し・アブレーション用）。
    // SPSA（with_params 経由）でも env が設定されていればそちらを優先する
    let params = match src.var("TSUITATE_VALUE_NN_W")
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
    let params = match src.var("TSUITATE_CHECKER_REMOVAL_W")
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
    let params = match src.var("TSUITATE_CAPTURE_BET_VAR_W")
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
    let params = match src.var("TSUITATE_MATE_THREAT_W")
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
    let params = match src.var("TSUITATE_MATE_RISK_W")
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
    let params = match src.var("TSUITATE_KING_HOLE_W")
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
    let params = match src.var("TSUITATE_LINK_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
    {
        Some(w) => EvalParams {
            link_w: w,
            ..params
        },
        None => params,
    };
    // 紐の働き重み付け（V3 の拡張）の運用ノブ（w スイープ・切り戻し用）
    let params = match src.var("TSUITATE_LINK_WORK_W")
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
    let params = match src.var("TSUITATE_LINK_WORK_REF")
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
    let params = match src.var("TSUITATE_CHECK_STRENGTH_W")
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
    let params = match src.var("TSUITATE_MOVER_CHECK_EXTRA")
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
    let params = match src.var("TSUITATE_MAJOR_PROMO_PATH_W")
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
    // 錨外し（0 で従来挙動）
    let params = match src.var("TSUITATE_ANCHOR_MOVE_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        Some(w) => EvalParams {
            anchor_move_w: w,
            ..params
        },
        None => params,
    };
    // ブラインド玉攻め加点の生存割引（0 で従来挙動）
    let params = match src.var("TSUITATE_BLIND_ATTACK_SURVIVE_W")
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
    let params = match src.var("TSUITATE_EXPOSED_PAWN_HEAD_W")
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
    let params = match src.var("TSUITATE_EXPOSED_KNOWN")
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
    let params = match src.var("TSUITATE_EXPOSED_MULTI_W")
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
    let params = match src.var("TSUITATE_TAINT_OCC_LEGAL_W")
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
    // 同一駒の連続移動減点（既定 0.2996 = SPSA 収束値）。撤去スイープ用:
    // 固有の守備範囲が「正当な継続手」（quest31-m073 の 4二成桂→5二成桂）と
    // 重なっている疑いがあり、膠着側は backtrack_penalty / repeat_penalty_w が
    // 上位互換でカバーする
    let params = match src.var("TSUITATE_SHUFFLE_PENALTY")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        Some(w) => EvalParams {
            shuffle_penalty: w,
            ..params
        },
        None => params,
    };
    // 打ち反則で確定した駒への当たり（0 で従来挙動）
    let params = match src.var("TSUITATE_FOUL_OCC_ATTACK_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        Some(w) => EvalParams {
            foul_occ_attack_w: w,
            ..params
        },
        None => params,
    };
    // 材料の退化ゲート（0 で従来挙動）
    let params = match src.var("TSUITATE_MATERIAL_DEGEN_Q0")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        Some(w) => EvalParams {
            material_degen_q0: w,
            ..params
        },
        None => params,
    };
    // 自玉近傍の敵駒の排除（0 で従来挙動）
    let params = match src.var("TSUITATE_OWN_ZONE_CAPTURE_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        Some(w) => EvalParams {
            own_zone_capture_w: w,
            ..params
        },
        None => params,
    };
    // 成りポテンシャルの敵玉近接重み（0 で従来挙動）
    let params = match src.var("TSUITATE_PROMO_KING_PROX")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
    {
        Some(w) => EvalParams {
            promo_king_prox: w,
            ..params
        },
        None => params,
    };
    // 玉で取る手の露見実効価値（0 で従来挙動）
    let params = match src.var("TSUITATE_KING_CAPTURE_REVEAL")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        Some(w) => EvalParams {
            king_capture_reveal: w,
            ..params
        },
        None => params,
    };
    // 2手読みの楽観置き換えの上限（F3。0 で従来挙動）
    let params = match src.var("TSUITATE_DEPTH2_OPTIMISM_CAP")
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
    let params = match src.var("TSUITATE_DROP_PROBE_W")
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
    let params = match src.var("TSUITATE_KING_ADJ_ENTRY_W")
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
    let params = match src.var("TSUITATE_MATE_GATE_Q0")
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
    let params = match src.var("TSUITATE_PLAN_W")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        Some(w) => EvalParams {
            plan_w: w,
            ..params
        },
        None => params,
    };
    // 同じ自陣形への往復の累積減点（0 で従来挙動）
    let params = match src.var("TSUITATE_REPEAT_PENALTY_W")
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
    let params = match src.var("TSUITATE_EFFECT_OWN_W")
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
    let params = match src.var("TSUITATE_EFFECT_OPP_W")
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
    let params = match src.var("TSUITATE_BOARD_DISCOUNT_W")
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
    let params = match src.var("TSUITATE_ESCAPE_COVER_W")
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
    let params = match src.var("TSUITATE_DEFENDER_CAPTURE_W")
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
    let params = match src.var("TSUITATE_DROP_HIT_EVAC_W")
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
    let params = match src.var("TSUITATE_PROMO_POTENTIAL_W")
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
    let params = match src.var("TSUITATE_HAND_OPTION_W")
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
        let _cfg = crate::config::scoped(&self.config);
        let budget = self.budget;
        let seed = self.seed;
        let est = self.est.get_or_insert_with(|| match seed {
            Some(s) => Estimator::with_seed_and_scale(view.your_color, s, budget.scale),
            None => Estimator::with_scale(view.your_color, budget.scale),
        });
        est.update(log);
    }

    fn oracle_anchor(
        &mut self,
        view: &PlayerView,
        log_prefix: &ObservationLog,
        truth: &Position,
    ) -> bool {
        let _cfg = crate::config::scoped(&self.config);
        let budget = self.budget;
        let seed = self.seed;
        let est = self.est.get_or_insert_with(|| match seed {
            Some(s) => Estimator::with_seed_and_scale(view.your_color, s, budget.scale),
            None => Estimator::with_scale(view.your_color, budget.scale),
        });
        est.oracle_anchor(log_prefix, truth);
        true
    }

    fn choose(
        &mut self,
        view: &PlayerView,
        log: &ObservationLog,
        foul_tried: &HashSet<String>,
    ) -> Option<String> {
        // この instance の設定をこのスレッドへ設置する（issue #21）。
        // 評価・推定の実装は深い呼び出しの奥から定数を引くので、引数で
        // 引き回す代わりに入口で設置し、抜けるときに元へ戻す
        let _cfg = crate::config::scoped(&self.config);
        let budget = self.budget;
        let seed = self.seed;
        // 定跡・候補ゼロで早期 return したとき前の手番のランキング（と診断の
        // 粒子スナップショット）が残らないように
        self.last_ranking = None;
        self.last_particles = None;
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

        let mut candidates = candidate_moves_with_log(view, foul_tried, Some(log));
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

        // 直前に受理された自分の手（手戻りシャッフルの抑制／捕獲直後の退避判定）
        let last_my_move = log.events().iter().rev().find_map(|e| match e {
            Observation::MyMove { usi, captured, .. } => {
                parse_usi(usi).map(|mv| (mv, captured.is_some()))
            }
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
            if !taint_king_fix() {
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
        let taint_pool: Vec<(&Position, f64)> = taint_owned.iter().map(|(p, w)| (p, *w)).collect();
        // 診断（issue #28）のスナップショット。**既定 off なのでコピーもしない**
        if self.capture_particles {
            self.last_particles = Some(ParticleSnapshot {
                strict: sample.iter().map(|&(p, w)| (p.clone(), w)).collect(),
                taint: taint_pool.iter().map(|&(p, w)| (p.clone(), w)).collect(),
            });
        }

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
        // 争点マス（`anchor_move_w`）: 自分が取ったマスと、自駒を取られたマス。
        // どちらも観測だけで正確に決まる（相手も取られたマスを通知されるので
        // 双方が注目しているマス = 争点）
        let mut contested_squares = [false; 81];
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
                            contested_squares[crate::belief_features::sq_index(to)] = true;
                        }
                        if let ShogiMove::Board { from, .. } = mv {
                            my_touched_squares.push(from);
                        }
                        my_touched_squares.push(to);
                    }
                }
                Observation::OpponentMoved {
                    captured_my_piece_at: Some(sq),
                    ..
                } => {
                    if let Some(c) = parse_usi_square(sq) {
                        contested_squares[crate::belief_features::sq_index(c)] = true;
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
            // **王手中の反則が積もるほど事前（CheckSolver）を信じる**
            // （2026-08-20、arena-check-foul02。同一手番の各反則は「粒子の
            // 合法性投票がこの決定で外れている」直接の証拠なので、p_legal
            // ブレンドの prior_weight を反則数に比例して増やす。反則0では
            // 従来と同一挙動。`TSUITATE_CHECK_FOUL_PRIOR_BOOST=0` で無効）
            if view.you_in_check && !foul_tried.is_empty() {
                let f = 1.0 + check_foul_prior_boost() * foul_tried.len() as f64;
                p.prior_weight *= f;
                p.prior_weight_degen *= f;
            }
            if push > 0.0 {
                p.check_bonus *= 1.0 + push;
                p.attack_w *= 1.0 + push;
                p.advance_w *= 1.0 + 0.5 * push;
                p.backtrack_penalty *= 1.0 + push;
                p.shuffle_penalty *= 1.0 + push;
                // 終盤の紐減衰（`link_endgame_dampen`）。攻め増幅の対で、
                // 遠方の相互守りが玉攻めを潰さないようにする。
                // **ブラインド決定に限定**: 厳密粒子がある局面では紐の働き重みが
                // まだ使え、序中盤の予防的な紐（v12）を壊したくない
                let damp = link_endgame_dampen();
                if damp > 0.0
                    && sample.is_empty()
                    && view.move_number >= LINK_ENDGAME_DAMPEN_MIN_MOVE
                {
                    p.link_w /= 1.0 + damp * push;
                }
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
        let hang_risk_enabled = crate::config::current(|c| c.strategy.enable_hang_risk);
        let debug_check_enabled = crate::config::current(|c| c.strategy.debug_check);

        // 詰めろ判定のプール: 厳密粒子があればそれ、全滅していれば taint 粒子。
        // 終盤のブラインド（＝詰めろが決まる局面）ほど厳密粒子は枯れているので、
        // sample だけを見ると項が発火しない（実測: 発端の対人局 57手目は厳密0個・
        // taint の玉位置信念は真実 7i に 71.6%）。
        // taint 粒子でも**自駒配置・持ち駒・相手の持ち駒は真実と同期**している
        // （estimator.rs::force_apply）ので、自玉の逃げ道と相手の打てる駒種で
        // ほぼ決まる被詰めろ判定は taint でも信頼できる。嘘が乗るのは相手の
        // 盤上駒＝支え駒の有無で、そこは IfSupported 側が吸収する
        let mate_pool: &[(&Position, f64)] = if sample.is_empty() {
            &taint_pool
        } else {
            &sample
        };

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
        // ブラインド進入リスク／home占有割引の擬似粒子（`blind_home_risk_w` /
        // `blind_home_drop_occ_w`。観測のみ由来。使うのは厳密粒子ゼロの
        // 決定だけ = evaluate 側で判定する）
        let blind_home = ((blind_home_risk_w() != 0.0 || blind_home_drop_occ_w() > 0.0)
            && !view.you_in_check)
            .then(|| blind_home_position(view, log));
        // 信念ネットのマスごと占有（NN段階②）。81マスぶんの forward pass は
        // 決定点ごとに1回。`belief_gain_w`（ブラインド時の gain 供給）と
        // `belief_occ_cap_w`（厳密粒子が居ても裏付け無し捕獲をネット占有へ
        // 寄せる）が共有する。どちらも0なら計算しない
        let belief_occ_board: Option<[f64; 81]> =
            (belief_occ_cap_w() > 0.0 || (belief_gain_w() > 0.0 && sample.is_empty())).then(|| {
                let ctx = crate::belief_features::BeliefContext::from_log(view.your_color, log);
                crate::belief_nn::board_occupancy(&ctx)
            });
        let blind_belief_mean =
            (belief_gain_w() > 0.0 && sample.is_empty()).then(|| blind_capture_estimate(view, log));
        let blind_belief = match (belief_occ_board.as_ref(), blind_belief_mean) {
            (Some(o), Some(v)) => Some((o, v)),
            _ => None,
        };
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
        let opp_king_w: Option<[f64; 81]> = (params.effect_opp_w != 0.0
            || params.link_work_w != 0.0)
            .then(|| {
                let src: &[(&Position, f64)] = if sample.is_empty() {
                    &taint_pool
                } else {
                    &sample
                };
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

        // 成りポテンシャルの敵玉近接マップ（`promo_king_prox`）。deduce の
        // 玉候補集合から決定点ごとに1度だけ作る（粒子不要・ノイズゼロ）
        let promo_prox: Option<[f64; 81]> = (params.promo_king_prox != 0.0
            && params.promo_potential_w != 0.0
            && !view.you_in_check)
            .then(|| {
                let cands = crate::deduce::opp_king_candidates(view.your_color, log);
                promo_king_prox_map(params.promo_king_prox, &cands)
            })
            .flatten();
        // 成りポテンシャル（`promo_potential_w`）の現局面の値。着手後との差分を
        // gain へ加算する。王手中は無効（CheckSolver の領分）
        let promo_pot_before: Option<f64> = (params.promo_potential_w != 0.0 && !view.you_in_check)
            .then(|| {
                promo_potential(
                    &view.your_pieces,
                    view.your_color,
                    promo_prox.as_ref(),
                    None,
                )
            });
        // 大駒の成り道（`major_promo_path_w`）の現局面の値。同じく差分で使う
        let major_path_before: Option<f64> = (params.major_promo_path_w != 0.0
            && !view.you_in_check)
            .then(|| major_promo_path(&view.your_pieces, view.your_color));

        // 持ち駒オプション価値（`hand_option_w`）の決定点コンテキスト。
        // 王手中は無効（合駒は CheckSolver の領分）
        let hand_option: Option<HandOption> =
            (params.hand_option_w != 0.0 && !view.you_in_check).then(|| hand_option_context(view));

        // 未裏付け大駒への新規 threat の鮮度（`stale_threat_w`）。1.0 = 減衰なし
        let stale_freshness = {
            let w = stale_threat_w();
            if w > 0.0 {
                let u = unaccounted_opp_moves(log);
                (1.0 - w) + w * 0.8f64.powi(u.min(60) as i32)
            } else {
                1.0
            }
        };
        // 相手駒の占有が観測で裏付けられたマス（`material_degen_q0` の
        // 「裏付け無し捕獲だけ縮める」ゲート／`hand_asset_w` の仕事判定／
        // `own_zone_capture_w` の裏付け係数／`stale_threat_w`。決定点ごとに1回）
        let opp_occ_backed = (params.material_degen_q0 > 0.0
            || params.own_zone_capture_w > 0.0
            || stale_threat_w() > 0.0
            || hand_asset_w() > 0.0
            || promote_far_w() > 0.0
            || unbacked_camp_w() > 0.0
            || unbacked_gs_capture_w() > 0.0
            || belief_occ_cap_w() > 0.0
            || home_gold_attack_w() > 0.0
            || king_adj_heavy_w() > 0.0
            || tokin_file_drift_w() > 0.0
            || own_camp_idle_w() > 0.0
            || bishop_retreat_w() > 0.0
            || pawn_offfile_w() > 0.0
            || far_major_promo_capture_w() > 0.0)
            .then(|| opp_occupancy_evidence(view, log));
        // 玉接近減点の脅威マス（歴代の非歩打ち反則を含む。上記より広い）
        let king_threats = (king_known_approach_w() > 0.0
            && view.move_number >= KING_KNOWN_APPROACH_MIN_MOVE)
            .then(|| king_threat_evidence(log));
        // 持ち駒資産損の玉候補（`hand_asset_w`）。王手中は無効
        let hand_asset_kings: Option<std::collections::BTreeSet<Coord>> =
            (hand_asset_w() > 0.0 && !view.you_in_check && view.move_number >= HAND_ASSET_MIN_MOVE)
                .then(|| crate::deduce::opp_king_candidates(view.your_color, log));
        // 大駒成り遠方 / 玉筋歩 / 裏付け無し敵陣進入の玉候補。王手中は無効
        let promote_far_kings: Option<std::collections::BTreeSet<Coord>> = ((promote_far_w()
            > 0.0
            || king_file_pawn_w() > 0.0
            || king_file_pawn_mid_w() > 0.0
            || king_file_gold_w() > 0.0
            || tokin_approach_w() > 0.0
            || unbacked_camp_w() > 0.0
            || king_adj_heavy_w() > 0.0
            || tokin_file_drift_w() > 0.0
            || pawn_offfile_w() > 0.0
            || far_major_promo_capture_w() > 0.0
            || knight_camp_exit_w() > 0.0
            || endgame_camp_general_w() > 0.0)
            && !view.you_in_check)
            .then(|| crate::deduce::opp_king_candidates(view.your_color, log));
        // 玉候補への接近ボーナスの近接マップ（`king_cand_attack_w`）。
        // 候補集合が**鋭いときだけ**（`king_cand_attack_gate` 以下）作る:
        // 40〜55マスに散らばった候補への近接は「敵陣へ前進」以上の意味を
        // 持たない。王手中は無効（CheckSolver の領分）
        let king_cand_set: Option<std::collections::BTreeSet<Coord>> = ((king_cand_attack_w()
            > 0.0
            || king_cand_check_w() > 0.0
            || landing_support_w() > 0.0)
            && !view.you_in_check)
            .then(|| {
                let cands = crate::deduce::opp_king_candidates(view.your_color, log);
                (!cands.is_empty() && cands.len() <= king_cand_attack_gate()).then_some(cands)
            })
            .flatten();
        let king_cand_prox: Option<[f64; 81]> =
            king_cand_set.as_ref().map(|c| king_cand_prox_map(c));
        // 玉位置ネットによる近接マップ（`king_belief_prox_w`）。
        // deduce の玉候補が鈍い側（＝王手をあまり掛けていない側）でだけ使う
        let king_belief_dist: Option<Vec<(Coord, f64)>> =
            (king_belief_prox_w() > 0.0 && !view.you_in_check && king_cand_prox.is_none())
                .then(|| {
                    // 情報源は**玉位置ネット**（`king_belief_nn`）。粒子ではない:
                    // `bin/king_cands` の実測（quest_20260731、14手目以降）で
                    // 後手側は真マス確率 0.272（一様 0.021 の13倍）・top1 46%・
                    // 実効サポート 7.2 と鋭いのに対し、粒子由来の分布は
                    // ブラインド top1 42%（メモリ）で、実際に接近ボーナスへ
                    // 流すと全水準で負けた（h1/h2/h3 実測）。
                    // ネットは deduce と**相補的**（先手は deduce が鋭くネットが
                    // top1 7%、後手はその逆）なので、deduce が鈍い側でだけ使う
                    let ctx = crate::belief_features::BeliefContext::from_log(view.your_color, log);
                    let cands = crate::deduce::opp_king_candidates(view.your_color, log);
                    let dist = crate::king_belief_nn::king_distribution(&ctx, &cands);
                    if dist.is_empty() {
                        return None;
                    }
                    // 実効サポート数（1/Σp²）が広いときは「敵陣へ前進」以上の情報が
                    // 無いので使わない。閾値は deduce 側と同じノブを共用する
                    let eff = 1.0 / dist.iter().map(|&(_, p)| p * p).sum::<f64>();
                    (eff <= king_cand_attack_gate() as f64).then_some(dist)
                })
                .flatten();
        let king_belief_prox: Option<[f64; 81]> =
            king_belief_dist.as_ref().map(|d| king_dist_prox_map(d));
        // 成る王手の露見ペナルティ用玉候補（`promote_check_reveal_w`）。
        // ブラインド決定でも効かせるため粒子不要。王手中は無効
        let promote_check_kings: Option<std::collections::BTreeSet<Coord>> =
            (promote_check_reveal_w() > 0.0
                && !view.you_in_check
                && (PROMOTE_CHECK_REVEAL_MIN_MOVE..=PROMOTE_CHECK_REVEAL_MAX_MOVE)
                    .contains(&view.move_number))
            .then(|| crate::deduce::opp_king_candidates(view.your_color, log));
        // 不成の双子がある成り手への露見減点（`nonpromote_check_w`）。
        // 生成側と同じ条件・同じ候補集合を使う
        let nonpromote_check_kings: Option<(Vec<(Coord, f64)>, [bool; 81])> =
            (nonpromote_check_w() > 0.0 && !view.you_in_check)
                .then(|| (nonpromote_king_dist(view, log), king_threat_evidence(log)))
                .filter(|(d, _)| !d.is_empty());

        // この手番の打ち反則で占有が確定したマスと、残存敵駒の平均交換価値
        // （`foul_occ_attack_w`）。反則では手番が変わらないので情報は完全に新鮮。
        // 平均価値は観測のみで決まる: 相手の初期19枚（玉除く）− 自分が取った駒
        let turn_foul_occ: Option<([bool; 81], f64)> = (params.foul_occ_attack_w != 0.0
            && !view.you_in_check)
            .then(|| {
                let mut arr = [false; 81];
                let mut any = false;
                for u in foul_tried {
                    if let Some(ShogiMove::Drop { to, .. }) = parse_usi(u) {
                        arr[crate::belief_features::sq_index(to)] = true;
                        any = true;
                    }
                }
                any.then(|| (arr, mean_remaining_opp_value(log)))
            })
            .flatten();

        let rng = &mut self.rng;
        // 王手中の玉の手の gain 平均化判定用（check_king_gain_mean の doc 参照）
        let my_king = king_square(view);
        // 過去手番の玉反則行き先（`king_repeat_foul_w`）。同手番は foul_tried。
        let repeat_w = king_repeat_foul_w();
        let stale_king_dests = (repeat_w > 0.0)
            .then(|| stale_king_foul_dests(log, view.your_color, view.move_number))
            .unwrap_or_default();
        // 玉プローブ・経路プローブ・玉センサーのコンテキスト（`ProbeCtx` の doc）。
        // 王手中は無効（CheckSolver の領分）
        let probe_ctx: Option<ProbeCtx> = (!view.you_in_check
            && (king_probe_w() > 0.0
                || path_probe_w() > 0.0
                || king_sensor_w() > 0.0
                || probe_audit()))
            .then(|| ProbeCtx {
                my_king,
                stale_king_dests: stale_king_foul_dests(log, view.your_color, view.move_number),
                revealed: opp_occupancy_evidence(view, log),
                king_probe_w: king_probe_w(),
                path_probe_w: path_probe_w(),
                king_sensor_w: king_sensor_w(),
                p_push: sensor_p_push(),
                p_promo: sensor_p_promo(),
            });
        if crate::hits::enabled() {
            crate::hits::flag("king_repeat_stale", !stale_king_dests.is_empty());
        }
        let is_king_move =
            |mv: &ShogiMove| matches!(*mv, ShogiMove::Board { from, .. } if Some(from) == my_king);
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
                blind_home.as_ref(),
                blind_belief,
                taint_occ_board.as_ref(),
                opp_king_w.as_ref(),
                drop_hit_expo_before,
                promo_pot_before,
                major_path_before,
                promo_prox.as_ref(),
                turn_foul_occ.as_ref(),
                hand_option.as_ref(),
                &my_drop_foul_squares,
                &own_attack,
                &contested_squares,
                opp_occ_backed.as_ref(),
                stale_freshness,
                hand_asset_kings.as_ref(),
                king_threats.as_ref(),
                belief_occ_board.as_ref(),
                probe_ctx.as_ref(),
            );
            // 過去手番で玉が反則した行き先への再訪（`king_repeat_foul_w`）。
            // p_legal を安全方向に落とすだけ。gain 内の課税は anchor_move の
            // 教訓で反則経済を壊すので使わない。
            out.p_legal *= king_repeat_legal_factor(&mv, my_king, &stale_king_dests, repeat_w);
            // 王手中: 既知敵駒（未説明手数込みの known_enemy_squares）が
            // カバーする玉の行き先は p_legal を min で締める。粒子の合法性
            // 投票が「既知マスの駒はもう動いた」と信じて玉逃げを 0.7 に保ち、
            // 反則を積む（arena-check-foul02: 5三のと金が 4二/5二/6二 を全部
            // カバーしているのに玉逃げ3連発）。min 専用・玉の Board 移動のみ
            if view.you_in_check {
                if let Some(cap) = check_solver
                    .as_ref()
                    .and_then(|s| s.known_covered_king_move_cap(&mv))
                {
                    out.p_legal = out.p_legal.min(cap);
                }
            }
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
            // 大駒成りの遠方ペナルティ（`promote_far_w` の doc 参照）。
            // gain の内側: 成りの固定ボーナスと同レイヤで綱引きさせる。
            if let (
                Some(cands),
                ShogiMove::Board {
                    from,
                    to,
                    promote: true,
                },
            ) = (promote_far_kings.as_ref(), mv)
            {
                let w = promote_far_w();
                let role = view
                    .your_pieces
                    .iter()
                    .find(|p| p.square == make_usi_square(from))
                    .map(|p| p.role);
                let target_role = if promote_far_all() {
                    // 全駒種へ拡張（`TSUITATE_PROMOTE_FAR_ALL=1`）。
                    // 玉から遠い「意味のない成り」は歩・桂・銀でも同じ理屈で
                    // 課税されるべき（実測の失点上位は 4a3b+/5f5g+/7d7c+/4a6c+ と
                    // ほぼ全部が成る手）。ただし課税額は駒の交換価値で頭打ちに
                    // する（歩の成りに角と同じ 5 点を課すのは過剰）
                    role.is_some()
                } else {
                    matches!(role, Some(Role::Bishop | Role::Rook))
                };
                if w > 0.0 && view.move_number >= PROMOTE_FAR_MIN_MOVE && target_role {
                    // 観測裏付けの占有マスへの成り捕獲は「材料」なので免税
                    let backed_hit = opp_occ_backed
                        .as_ref()
                        .is_some_and(|b| b[crate::belief_features::sq_index(to)]);
                    if !backed_hit {
                        let mut pen = w * promote_far_amount(from, to, cands);
                        if promote_far_all() {
                            if let Some(r) = role {
                                pen = pen.min(exchange_value(r));
                            }
                        }
                        out.gain -= pen;
                    }
                }
                // 遠方の大駒成り捕獲（`far_major_promo_capture_w`）。
                // 裏付け無しかつ玉から遠い成りの幻の駒得を外側で削る。
                let cw = far_major_promo_capture_w();
                if cw > 0.0 && matches!(role, Some(Role::Bishop | Role::Rook)) {
                    let backed_hit = opp_occ_backed
                        .as_ref()
                        .is_some_and(|b| b[crate::belief_features::sq_index(to)]);
                    if !backed_hit
                        && out.capture_value > 0.0
                        && promote_far_amount(from, to, cands) > 0.0
                    {
                        out.gain -= cw * out.capture_value;
                    }
                }
            }
            // 玉筋の歩前進・打ち（`king_file_pawn_w`）と金打ち（`king_file_gold_w`）。
            // gain の内側。敵陣の歩だけ（7c7b+ / P*7c）。9六歩・4f4g+ は加点しない。
            if !view.you_in_check {
                if let Some(cands) = promote_far_kings.as_ref() {
                    let pw = king_file_pawn_w();
                    let gw = king_file_gold_w();
                    // 終盤の歩成り（`pawn_offfile_w`）。金銀を持つ手数 125 以降、
                    // または手数 137 以降（手持ちゲートなし）。中段→敵陣の歩成り。
                    // 既に敵陣の前進は除外。不成は対象外（m134/m136 の妥当な
                    // 5f5g+ を守る）。発火中は king_file_pawn を掛けない。
                    let ow = pawn_offfile_w();
                    let mut late_pawn_promo = false;
                    let pawn_offfile_gate = has_attacking_general(view)
                        || view.move_number >= PAWN_OFFFILE_FORCE_MIN_MOVE;
                    if ow > 0.0 && pawn_offfile_gate {
                        if let ShogiMove::Board {
                            from,
                            to,
                            promote: true,
                        } = mv
                        {
                            let is_pawn = view
                                .your_pieces
                                .iter()
                                .find(|p| p.square == make_usi_square(from))
                                .is_some_and(|p| p.role == Role::Pawn);
                            if is_pawn {
                                let backed_hit = opp_occ_backed
                                    .as_ref()
                                    .is_some_and(|b| b[crate::belief_features::sq_index(to)]);
                                let amt = pawn_late_promo_amount(
                                    from,
                                    to,
                                    view.your_color,
                                    view.move_number,
                                    backed_hit,
                                );
                                if amt > 0.0 {
                                    late_pawn_promo = true;
                                    out.gain -= ow * amt;
                                    if out.capture_value > 0.0 {
                                        out.gain -= out.capture_value;
                                    }
                                }
                            }
                        }
                    }
                    match mv {
                        ShogiMove::Board { from, to, .. } if !late_pawn_promo => {
                            let is_pawn = view
                                .your_pieces
                                .iter()
                                .find(|p| p.square == make_usi_square(from))
                                .is_some_and(|p| p.role == Role::Pawn);
                            if is_pawn {
                                if pw > 0.0 {
                                    out.gain += pw
                                        * king_file_pawn_amount(from, to, view.your_color, cands);
                                }
                                let mw = king_file_pawn_mid_w();
                                if mw > 0.0 {
                                    out.gain += mw
                                        * king_file_pawn_mid_amount(
                                            from,
                                            to,
                                            view.your_color,
                                            cands,
                                            view.move_number,
                                        );
                                }
                            }
                        }
                        ShogiMove::Drop {
                            role: Role::Pawn,
                            to,
                        } if pw > 0.0 && in_enemy_camp(to, view.your_color) => {
                            out.gain += pw * king_file_pawn_drop_amount(to, cands);
                        }
                        ShogiMove::Drop {
                            role: Role::Gold,
                            to,
                        } if gw > 0.0 && in_enemy_camp(to, view.your_color) => {
                            out.gain += gw * king_file_pawn_drop_amount(to, cands);
                        }
                        _ => {}
                    }
                }
            }
            // 裏付け無しの敵陣進入（`unbacked_camp_w`）。gain の内側。
            // 金銀は粒子が捕獲を信じているときだけ（3h4i の capture=0 を守る）。
            if !view.you_in_check {
                let uw = unbacked_camp_w();
                if uw > 0.0 && view.move_number >= UNBACKED_CAMP_MIN_MOVE {
                    if let Some(backed) = opp_occ_backed.as_ref() {
                        let landing = match mv {
                            ShogiMove::Board { from, to, .. } => view
                                .your_pieces
                                .iter()
                                .find(|p| p.square == make_usi_square(from))
                                .map(|p| (p.role, to, false)),
                            ShogiMove::Drop { role, to } => Some((role, to, true)),
                        };
                        if let Some((role, to, is_drop)) = landing {
                            if unbacked_camp_needs_capture(role, out.capture_value) {
                                out.gain -= uw
                                    * unbacked_camp_amount(
                                        role,
                                        to,
                                        view.your_color,
                                        backed,
                                        is_drop,
                                    );
                            }
                        }
                    }
                }
            }
            // 相手の初期金位置への金銀当たりは、全候補の捕獲が見えてから
            // 後段で足す（同一駒に 3h4g 型の捕獲があるときの 3h4i 逃避を防ぐ）。
            // と金の玉筋接近（`tokin_approach_w`）。
            if !view.you_in_check {
                let tw = tokin_approach_w();
                if tw > 0.0 {
                    if let (Some(cands), ShogiMove::Board { from, to, .. }) =
                        (promote_far_kings.as_ref(), mv)
                    {
                        let is_tokin = view
                            .your_pieces
                            .iter()
                            .find(|p| p.square == make_usi_square(from))
                            .is_some_and(|p| p.role == Role::Tokin);
                        if is_tokin {
                            out.gain +=
                                tw * tokin_file_approach_amount(from, to, view.your_color, cands);
                        }
                    }
                }
            }
            // 玉隣接への高い駒の無支え進入（`king_adj_heavy_w`）。盤上の移動だけ。
            if !view.you_in_check {
                let hw = king_adj_heavy_w();
                if hw > 0.0 && view.move_number >= KING_ADJ_HEAVY_MIN_MOVE {
                    if let (Some(cands), Some(backed), ShogiMove::Board { from, to, .. }) =
                        (promote_far_kings.as_ref(), opp_occ_backed.as_ref(), mv)
                    {
                        let role = view
                            .your_pieces
                            .iter()
                            .find(|p| p.square == make_usi_square(from))
                            .map(|p| p.role);
                        if let Some(role) = role {
                            out.gain -= hw
                                * king_adj_heavy_amount(role, to, view.your_color, cands, backed);
                        }
                    }
                }
            }
            // と金の玉筋逸れ（`tokin_file_drift_w`）。盤上のと金移動だけ。
            // 粒子が捕獲を信じている手（m029 の 3b3c 桂取り）は免税。
            if !view.you_in_check {
                let dw = tokin_file_drift_w();
                if dw > 0.0 {
                    if let (Some(cands), Some(backed), ShogiMove::Board { from, to, .. }) =
                        (promote_far_kings.as_ref(), opp_occ_backed.as_ref(), mv)
                    {
                        let is_tokin = view
                            .your_pieces
                            .iter()
                            .find(|p| p.square == make_usi_square(from))
                            .is_some_and(|p| p.role == Role::Tokin);
                        if is_tokin && out.capture_value < 0.5 {
                            out.gain -= dw
                                * exchange_value(Role::Tokin)
                                * tokin_file_drift_amount(
                                    view,
                                    from,
                                    to,
                                    view.your_color,
                                    cands,
                                    backed,
                                );
                        }
                    }
                }
            }
            // 自陣の金銀桂の空きマス移動（`own_camp_idle_w`）。
            if !view.you_in_check {
                let iw = own_camp_idle_w();
                if iw > 0.0 {
                    if let (Some(backed), ShogiMove::Board { from, to, .. }) =
                        (opp_occ_backed.as_ref(), mv)
                    {
                        let role = view
                            .your_pieces
                            .iter()
                            .find(|p| p.square == make_usi_square(from))
                            .map(|p| p.role);
                        if let Some(role) = role {
                            out.gain -= iw
                                * own_camp_idle_amount(
                                    role,
                                    to,
                                    view.your_color,
                                    backed,
                                    view.move_number,
                                );
                        }
                    }
                }
            }
            // 角・馬の非前進空きマス移動（`bishop_retreat_w`）。
            if !view.you_in_check {
                let bw = bishop_retreat_w();
                if bw > 0.0 && view.move_number >= BISHOP_RETREAT_MIN_MOVE {
                    if let (Some(backed), ShogiMove::Board { from, to, .. }) =
                        (opp_occ_backed.as_ref(), mv)
                    {
                        let role = view
                            .your_pieces
                            .iter()
                            .find(|p| p.square == make_usi_square(from))
                            .map(|p| p.role);
                        if let Some(role) = role {
                            out.gain -=
                                bw * bishop_retreat_amount(role, from, to, view.your_color, backed);
                        }
                    }
                }
            }
            // 終盤の敵陣成銀の筋替え（`endgame_camp_general_w`）。玉筋へ近づくときだけ。
            if !view.you_in_check {
                let gw = endgame_camp_general_w();
                if gw > 0.0 {
                    if let (Some(cands), ShogiMove::Board { from, to, .. }) =
                        (promote_far_kings.as_ref(), mv)
                    {
                        let role = view
                            .your_pieces
                            .iter()
                            .find(|p| p.square == make_usi_square(from))
                            .map(|p| p.role);
                        if let Some(role) = role {
                            out.gain += gw
                                * endgame_camp_general_amount(
                                    role,
                                    from,
                                    to,
                                    view.your_color,
                                    view.move_number,
                                    cands,
                                );
                        }
                    }
                }
            }
            // 桂銀香の任意成り課税（`own_camp_minor_promo_w`）。
            // 終盤の桂敵陣成り課税（`knight_late_promo_w`）・終盤の桂成り加点
            // （`knight_endgame_promo_w`）・桂の自陣脱出（`knight_camp_exit_w`）
            // と銀の自陣脱出（`silver_camp_exit_w`）も同じ role を使う。
            if !view.you_in_check {
                if let ShogiMove::Board { from, to, promote } = mv {
                    let role = view
                        .your_pieces
                        .iter()
                        .find(|p| p.square == make_usi_square(from))
                        .map(|p| p.role);
                    if let Some(role) = role {
                        let mw = own_camp_minor_promo_w();
                        if mw > 0.0 {
                            out.gain -= mw
                                * own_camp_minor_promo_amount(
                                    role,
                                    from,
                                    to,
                                    promote,
                                    view.your_color,
                                );
                        }
                        let nw = knight_late_promo_w();
                        if nw > 0.0 {
                            out.gain -= nw
                                * knight_late_promo_amount(
                                    role,
                                    from,
                                    to,
                                    promote,
                                    view.your_color,
                                    view.move_number,
                                );
                        }
                        let np = knight_endgame_promo_w();
                        if np > 0.0 {
                            out.gain += np
                                * knight_endgame_promo_amount(
                                    role,
                                    to,
                                    promote,
                                    view.your_color,
                                    view.move_number,
                                );
                        }
                        let nx = knight_camp_exit_w();
                        if nx > 0.0 {
                            out.gain += nx
                                * knight_camp_exit_amount(
                                    role,
                                    from,
                                    to,
                                    view.your_color,
                                    view.move_number,
                                );
                        }
                        let sw = silver_camp_exit_w();
                        if sw > 0.0 {
                            out.gain += sw
                                * silver_camp_exit_amount(
                                    role,
                                    from,
                                    to,
                                    view.your_color,
                                    view.move_number,
                                );
                        }
                    }
                }
            }
            // 玉候補への接近ボーナス（`king_cand_attack_w` の doc 参照）。
            // gain の内側（攻め加点は p_legal 割引の内側に置く。
            // dragon-check-drop の教訓）。着地マスの近接度を「着地する駒の
            // 安さ」で割る（同じ仕事なら最安の駒で）
            if let Some((map, term_w)) = king_cand_prox
                .as_ref()
                .filter(|_| !king_cand_attack_blind_only() || sample.is_empty())
                .map(|m| (m, king_cand_attack_w()))
                .or_else(|| king_belief_prox.as_ref().map(|m| (m, king_belief_prox_w())))
            {
                let (to, role) = match mv {
                    ShogiMove::Drop { role, to } => (to, role),
                    ShogiMove::Board {
                        from,
                        to,
                        promote: _,
                    } => {
                        let r = view
                            .your_pieces
                            .iter()
                            .find(|p| p.square == make_usi_square(from))
                            .map(|p| p.role);
                        // 安さは**持ち込んだ駒**で数える（成り後の駒種ではない）。
                        // 歩が玉隣で成るとと金扱いの cheapness ≈0.4 になり、
                        // 候補の隣への垂れ歩（m121 の P*5b、採点0）が
                        // 本命の 7c7b+（採点10）を cheapness 差で上回っていた。
                        match r {
                            Some(r) => (to, r),
                            None => (to, Role::King),
                        }
                    }
                };
                if role != Role::King {
                    let prox = map[crate::belief_features::sq_index(to)];
                    // 玉候補マスへ**実際に利く**手への追加加点
                    // （`king_cand_check_w`）。採点済み eval の局面内回帰では
                    // 距離を制御してもなお +0.78 点ぶんの説明力がある
                    if king_cand_check_w() > 0.0 {
                        // 分布は deduce 側なら候補集合上の一様分布、
                        // ネット側なら玉位置ネットの分布そのもの
                        let dist: Option<Vec<(Coord, f64)>> = if king_cand_prox.is_some() {
                            king_cand_set.as_ref().map(|cands| {
                                cands
                                    .iter()
                                    .map(|&k| (k, 1.0 / cands.len() as f64))
                                    .collect()
                            })
                        } else {
                            king_belief_dist.clone()
                        };
                        if let Some(dist) = dist {
                            let frac = blind_king_attack(view, &mv, &dist);
                            out.gain +=
                                king_cand_check_w() * frac * 2.0 / (1.0 + exchange_value(role));
                        }
                    }
                    // 「同じ仕事なら最安の駒で」（`foul_occ_attack_w` と同じ規約）。
                    // 歩を 1.0 に正規化した安さ係数を掛ける: 裸の銀・金を信念上の
                    // 玉の隣へ投げる手（m040 の教訓）は歩の垂らしより小さい加点に
                    // なる。採点済み eval の局面内回帰でも歩 +0.53 / 角 −0.53 と
                    // 「安い駒の手ほど良い」が出ている
                    // 床は敷かない: 0.4 で止めると角・飛の成り込み（4a6c+ 型、
                    // 採点2）が玉の近くというだけで 40% の加点を受け、
                    // 狙いの「安い駒で寄せる」から外れる（supp 実測の回帰）
                    let cheapness = 2.0 / (1.0 + exchange_value(role));
                    // **支えのある接近だけを満額にする**（2026-08-13 の supp2 実測。
                    // 玉候補の隣というだけで裸の金銀打ちが浮く m145 の G*8c（採点0）
                    // 対策で、m040 の「無防備な駒を信念上の玉の隣へ置くほど加点が
                    // 最大になる」と同型の穴）。採点回帰でも支え枚数は +0.60点ある
                    // 支えゲート（`landing_support_w` = ゲートの強さ）。
                    // **独立した加算項にしてはいけない**: 玉候補ゲートの中とはいえ
                    // 「支えのあるマスならどこでも加点」になり、飛車に支えられた
                    // 無意味な 2二歩打が浮く（supp3 実測で P*2b が 0.85→3.9回/
                    // シードに増え、失点 +0.154 の最大要因になった）。接近ボーナスへ
                    // 掛けることで「支えのある寄せだけ満額」に限定する
                    let sw = landing_support_w();
                    let support = if sw > 0.0 {
                        (1.0 - sw) + sw * landing_def(view, &mv, &own_attack).min(2.0) / 2.0
                    } else {
                        1.0
                    };
                    out.gain += term_w * prox * cheapness * support;
                }
            }
            // 歩・角・飛の成る王手の露見ペナルティ（`promote_check_reveal_w`）。
            // ブラインド決定でも効くよう粒子不要（deduce 玉候補の幾何）。
            if let (
                Some(cands),
                ShogiMove::Board {
                    from,
                    to,
                    promote: true,
                },
            ) = (promote_check_kings.as_ref(), mv)
            {
                let w = promote_check_reveal_w();
                let role = view
                    .your_pieces
                    .iter()
                    .find(|p| p.square == make_usi_square(from))
                    .map(|p| p.role);
                if w > 0.0
                    && (PROMOTE_CHECK_REVEAL_MIN_MOVE..=PROMOTE_CHECK_REVEAL_MAX_MOVE)
                        .contains(&view.move_number)
                    && matches!(role, Some(Role::Pawn | Role::Bishop | Role::Rook))
                    && king_file_median(cands).is_some_and(|m| king_files_focused(cands, m))
                    && promote_checks_king_cand(view, from, to, role.unwrap(), cands)
                {
                    out.gain -= w;
                }
            }
            // 不成の双子がある成り手（＝成ると玉候補への利きが増える手）への
            // 露見減点（`nonpromote_check_w`）。生成側と同一条件なので、
            // 減点が掛かる手には必ず不成の逃げ道がある
            if let (Some((cands, occ_ev)), ShogiMove::Board { from, to, promote }) =
                (nonpromote_check_kings.as_ref(), mv)
            {
                let role = view
                    .your_pieces
                    .iter()
                    .find(|p| p.square == make_usi_square(from))
                    .map(|p| p.role);
                if let Some(r) = role {
                    if promotion_choice(r, from, to, view.your_color) == Promotion::Optional {
                        let mass = promotion_check_mass(view, from, to, r, cands, occ_ev);
                        if mass >= nonpromote_check_p() {
                            // 双子の間の**中心化再配分**にする（成りへ減点だけ
                            // だと、成り駒の利き・交換価値の差ぶん不成が沈んだ
                            // ままで第三の手が繰り上がる。実測 2026-08-15:
                            // 減点のみの版は m046/m048 で 4九銀成が消えた代わりに
                            // 5八と（採点2）が出て 4九銀不成（採点10）は浮かなかった）。
                            // 露見は王手が実際に掛かったときだけ起きるので
                            // 期待値（質量）でスケールする
                            let d = nonpromote_check_w() * mass;
                            out.gain += if promote { -d } else { d };
                        }
                    }
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
            let tiebreak = rng.random_range(0.0..0.01);
            out.tiebreak = tiebreak;
            let mut adjust = tiebreak;
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
            // 直前に動かした駒をまた動かすだけの手も雑なシャッフルとして軽く減点。
            // ただし直前が**捕獲**で今手がその厳密な逆かつ不成なら「取って逃げる」
            // （`capture_retreat_w` / quest31-m024: 同飛→3b4b）。観測のみ。
            // 成り逆（4a3b+ 型）は再突入なので対象外（m087/m089 回帰の教訓）。
            if let (
                Some((
                    ShogiMove::Board {
                        from: pf, to: pt, ..
                    },
                    last_captured,
                )),
                ShogiMove::Board {
                    from, to, promote, ..
                },
            ) = (last_my_move, mv)
            {
                let retreat_w = capture_retreat_w();
                let capture_retreat =
                    retreat_w > 0.0 && last_captured && !promote && from == pt && to == pf;
                if from == pt && to == pf {
                    if !capture_retreat {
                        adjust -= params.backtrack_penalty;
                    }
                } else if from == pt {
                    adjust -= params.shuffle_penalty;
                }
                if capture_retreat {
                    let val = view
                        .your_pieces
                        .iter()
                        .find(|p| p.square == make_usi_square(from))
                        .map(|p| exchange_value(p.role))
                        .unwrap_or(0.0);
                    adjust += retreat_w * val;
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
                // 玉接近減点は幻の分散ではないので平均化の外に出す
                // （m099: 王手逃げ 6八 vs 8八。平均化が −w·Δcloseness まで消す）
                let approach_pens: Vec<f64> = king_idx
                    .iter()
                    .map(|&i| match (king_threats.as_ref(), &scored[i].1) {
                        (Some(th), ShogiMove::Board { from, to, .. }) => {
                            let w = king_known_approach_w();
                            if w <= 0.0 {
                                0.0
                            } else {
                                w * king_known_approach_amount(*from, *to, th)
                            }
                        }
                        _ => 0.0,
                    })
                    .collect();
                for (j, &i) in king_idx.iter().enumerate() {
                    scored[i].2.gain += approach_pens[j];
                }
                let mean =
                    king_idx.iter().map(|&i| scored[i].2.gain).sum::<f64>() / king_idx.len() as f64;
                for (j, &i) in king_idx.iter().enumerate() {
                    scored[i].2.gain = mean - approach_pens[j];
                    scored[i].4 = scored[i].2.score() + scored[i].3;
                }
            }
        }

        // 王手中にほぼ確実な解消手があるなら、そこから大きく落ちる手を捨てる
        // （`check_safe_resolve_enabled`。既定 off。有効時も王手駒捕獲と玉の手は残す）
        if view.you_in_check && check_safe_resolve_enabled() {
            if let Some(solver) = check_solver.as_mut() {
                if !scored.is_empty() {
                    let ps: Vec<f64> = scored
                        .iter()
                        .map(|s| solver.resolve_probability(&s.1))
                        .collect();
                    let p_max = ps.iter().copied().fold(0.0_f64, f64::max);
                    if check_safe_resolve_active(p_max) {
                        let thresh = check_safe_resolve_thresh(p_max);
                        let before = scored.len();
                        let mut i = 0;
                        scored.retain(|s| {
                            let keep = check_safe_resolve_keep(
                                ps[i],
                                thresh,
                                solver.captures_checker(&s.1),
                                is_king_move(&s.1),
                            );
                            i += 1;
                            keep
                        });
                        if crate::hits::enabled() {
                            crate::hits::flag("check_safe_resolve", scored.len() < before);
                        }
                    }
                }
            }
        }

        // 相手の初期金位置への金銀当たり（`home_gold_attack_w`）。
        // 不成のみ・手数 ≥44・同一駒が初期金の筋の別マスへ動けないときだけ。
        // m042 の 4f4g+ と m054 の 3h4g を守りつつ、m046 の 3h4i を受け皿にする。
        if !view.you_in_check {
            let hw = home_gold_attack_w();
            if hw > 0.0 {
                if let Some(backed) = opp_occ_backed.as_ref() {
                    let file_sibling_from: HashSet<Coord> = scored
                        .iter()
                        .filter_map(|(_, mv, _, _, _)| match mv {
                            ShogiMove::Board { from, to, .. }
                                if home_gold_file_sibling(*to, view.your_color)
                                    && view.your_pieces.iter().any(|p| {
                                        p.square == make_usi_square(*from)
                                            && matches!(p.role, Role::Gold | Role::Silver)
                                    }) =>
                            {
                                Some(*from)
                            }
                            _ => None,
                        })
                        .collect();
                    for (_, mv, out, adjust, score) in scored.iter_mut() {
                        let ShogiMove::Board {
                            from, to, promote, ..
                        } = *mv
                        else {
                            continue;
                        };
                        if view.move_number < HOME_GOLD_MIN_MOVE
                            || file_sibling_from.contains(&from)
                            || out.capture_value >= 0.5
                        {
                            continue;
                        }
                        let Some(role) = view
                            .your_pieces
                            .iter()
                            .find(|p| p.square == make_usi_square(from))
                            .map(|p| p.role)
                        else {
                            continue;
                        };
                        let bonus = hw
                            * home_gold_attack_amount(role, to, view.your_color, backed, promote);
                        if bonus > 0.0 {
                            out.gain += bonus;
                            *score = out.score() + *adjust;
                        }
                    }
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
                own_zone: out.own_zone,
                probe_unit: out.probe_unit,
                probe_mass: out.probe_mass,
                probe_concentration: out.probe_concentration,
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
                foul_probe: out.foul_probe,
                tiebreak: out.tiebreak,
            });
            if best.as_ref().is_none_or(|(_, _, s)| final_score > *s) {
                best = Some((usi, out.p_legal, final_score));
            }
        }
        ranking.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // **プローブ影の監査**（`TSUITATE_PROBE_AUDIT=1`、codex 提案 2026-08-22）。
        // 重み1のプローブ値 B と首位との差 Δ から「w* = Δ/B（この候補を首位に
        // するのに必要な重み）」を出す。w* の分布を見れば「効果量が足りないだけ」
        // なのか「そもそも良い候補が m036 しかない」のかが、アリーナを回さずに分かる
        if probe_audit() {
            if let Some(top) = ranking.first().map(|c| c.score) {
                for c in &ranking {
                    if c.probe_unit <= 0.0 {
                        continue;
                    }
                    let gap = top - c.score;
                    let w_star = if c.probe_unit > 0.0 { gap / c.probe_unit } else { f64::INFINITY };
                    eprintln!(
                        "PROBEAUDIT\t{}\t{}\t{:.4}\t{:.4}\t{:.2}\t{:.3}\t{:.3}",
                        view.move_number, c.usi, gap, c.probe_unit, w_star, c.probe_mass,
                        c.probe_concentration
                    );
                }
            }
        }
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

    fn set_capture_particles(&mut self, on: bool) {
        self.capture_particles = on;
        if !on {
            self.last_particles = None;
        }
    }

    fn last_particles(&self) -> Option<&ParticleSnapshot> {
        self.last_particles.as_ref()
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
        // **由来タグを見るときは併合キーにも混ぜる**（codex 指摘 2026-08-22）:
        // 物理指紋は盤・持ち駒・手番だけなので、「観測に裏付けられた駒がそこに
        // いる粒子」と「同じ盤面だが幻として置かれた粒子」が併合され、先に来た
        // 方のアンカー表で代表されてしまう（実測: quest_0809-m036 で本物の
        // プローブ信号が 26% まで薄まっていた）。物理指紋そのものは真実照合・
        // 尤度フィットが依存しているので変えない
        let key = if anchors_needed() {
            pos.fingerprint() ^ pos.anchors().fingerprint()
        } else {
            pos.fingerprint()
        };
        match seen.entry(key) {
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

    // **指紋シェアの上限**（`eval_weight_cap`、既定 1.0 = 無効）: 崩壊した
    // 粒子集合では、生き残った少数の指紋の**複製数**がそのまま重みになるので
    // 「1個の粒子が信念の85%」という点質量ができる（実測 2026-08-10、
    // m067 の 4一: 厳密粒子9個で飛車85.9%、真実は空きマス。ユーザー指摘
    // 「厳密粒子の信念が偏りすぎている」）。複製数は崩壊前のリサンプリングの
    // 遺物で独立な証拠ではないので、1指紋の取り分に上限を敷いて残りへ配り直す。
    // 健全な集合（どの指紋も上限未満）では何も変わらない
    let cap_share = eval_weight_cap();
    if cap_share < 1.0 && sample.len() > 1 {
        let total: f64 = sample.iter().map(|(_, w)| w).sum();
        if total > 0.0 {
            let mut per_fp: HashMap<u64, f64> = HashMap::new();
            for (p, w) in sample.iter() {
                *per_fp.entry(p.fingerprint()).or_insert(0.0) += w;
            }
            let cap = cap_share * total;
            let over = per_fp.values().any(|&m| m > cap);
            if over {
                // 超過した指紋だけ縮める。残りへの配り直しは直後の
                // legacy_mass への再正規化が比例配分でやってくれる
                for (p, w) in sample.iter_mut() {
                    let m = per_fp[&p.fingerprint()];
                    if m > cap {
                        *w *= cap / m;
                    }
                }
            }
            crate::hits::flag("eval_weight_cap_fired", over);
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

/// 王手中の反則1回あたりの prior_weight 倍率の増分
/// （`TSUITATE_CHECK_FOUL_PRIOR_BOOST`、既定 3.0、0 で従来挙動）
fn check_foul_prior_boost() -> f64 {
    crate::config::current(|c| c.strategy.check_foul_prior_boost)
}

/// **未裏付け大駒への新規 threat の鮮度減衰**（`TSUITATE_STALE_THREAT_W`、
/// 既定 0 = 無効。2026-08-21、codex 設計相談。quest31 の P*2f/P*2g 濫発が発端 =
/// ユーザー指摘「もう飛車は動かされてる可能性が高い」）。
/// threat_value の対象が飛角（成り込み）で、その現在マスが観測裏付け
/// （opp_occupancy_evidence）に無いとき、**着手で新たに増えた threat 増分だけ**を
/// freshness = (1−w) + w×0.8^未説明手数 で薄める。既存 threat の基準値・
/// 裏付けありの大駒・小駒は満額のまま（blind_home の失敗 = 安全方向の
/// 課税による合法悪手への押し出し、とは介入面が違う: これは「根拠の古い
/// 成功報酬を満額払わない」）。凍結版はこの名前を知らない
fn stale_threat_w() -> f64 {
    crate::config::current(|c| c.strategy.stale_threat_w)
}

/// 未説明の相手手数（捕獲観測のない OpponentMoved の数 = その駒が動けた機会）。
/// check.rs の KNOWN_STAY_PER_UNACC と同じ尺度
fn unaccounted_opp_moves(log: &ObservationLog) -> u32 {
    log.events()
        .iter()
        .filter(|e| {
            matches!(
                e,
                Observation::OpponentMoved {
                    captured_my_piece_at: None,
                    ..
                }
            )
        })
        .count() as u32
}

fn blind_recapture_w() -> f64 {
    crate::config::current(|c| c.strategy.blind_recapture_w)
}

/// ブラインド進入リスク（`TSUITATE_BLIND_HOME_RISK_W`、既定 0 = 無効）の重み。
/// 厳密粒子ゼロの決定では `expected` ごと mover リスクが消え、敵陣の
/// 初期配置マスへの成り込みが無コストで浮く（quest31 の 1三角成 = 初期位置の
/// 歩を取り、1一の香に取り返される歩角交換。粒子が生きた seed では
/// リスク −3.2〜−6.4 で沈むのにブラインド seed ではリスク 0 で 16 位。
/// 2026-08-09 ユーザー指摘が発端）
fn blind_home_risk_w() -> f64 {
    crate::config::current(|c| c.strategy.blind_home_risk_w)
}

/// 評価サンプルで**1指紋が持てる重みの上限シェア**
/// （`TSUITATE_EVAL_WEIGHT_CAP`、既定 1.0 = 無効）。
/// 崩壊した粒子集合では複製数がそのまま重みになり「1個の粒子が信念の85%」
/// という点質量ができる（実測 2026-08-10: m067 の 4一 で厳密粒子9個・
/// 飛車85.9%・真実は空きマス。ユーザー指摘「厳密粒子の信念が偏りすぎている」）。
/// 複製数は崩壊前のリサンプリングの遺物で独立な証拠ではない。
/// 推奨 0.4〜0.6。凍結版はこの名前を知らない
fn eval_weight_cap() -> f64 {
    crate::config::current(|c| c.strategy.eval_weight_cap)
}

/// 過去手番で玉が反則した行き先への再訪割引（`TSUITATE_KING_REPEAT_FOUL_W`、
/// 既定 0 = 無効。env 作業点は 0.8）。
///
/// 2026-08-19 打ち切り: 既定オンしない。機構と env は残す。
/// 原因駒が別マスで消えても汚名が残る（3七型）ためマス記憶は人間判断と矛盾する。
///
/// 一つの対局の中で同じ玉移動の反則を繰り返すのは、原因だった相手駒が
/// 動いた・無くなった・飛角の利きが遮られた、という観測が無いのに同じ
/// マスへ玉を出すこと。同手番の同一 USI は `foul_tried` 済みで、ここは
/// **手番をまたいだ行き先**だけ。`p_legal *= (1-w)` の安全方向のみ。
/// 解除は観測できたときだけ（そのマスで取った / 玉がそこへ受理された）。
/// 王手中も有効（行き先が過去に失敗した逃げは、CheckSolver の仮説が
/// 薄くても再試行しない）。玉の逃げ全体を割り引く第1版
/// （`check_king_prior`）とは違い、汚名のあるマスだけが対象。
/// 凍結版はこの名前を知らない。
fn king_repeat_foul_w() -> f64 {
    crate::config::current(|c| c.strategy.king_repeat_foul_w)
}

/// 玉行き先の再訪割引。既定 0（作業点は 0.8）。
const KING_REPEAT_FOUL_W: f64 = 0.0;

/// 玉の手が汚名のある行き先なら `1-w`、それ以外は 1。
fn king_repeat_legal_factor(
    mv: &ShogiMove,
    king: Option<Coord>,
    stale: &HashSet<Coord>,
    w: f64,
) -> f64 {
    if w <= 0.0 {
        return 1.0;
    }
    match *mv {
        ShogiMove::Board { from, to, .. } if Some(from) == king && stale.contains(&to) => {
            (1.0 - w).clamp(0.0, 1.0)
        }
        _ => 1.0,
    }
}

/// 残り反則1回（次の反則で即負け）のときの反則コストの床
/// （`TSUITATE_LAST_FOUL_GUARD`、**既定 60**、0 で従来挙動 = 切り戻しノブ）。
/// 材料スケールの gain（〜10）では反則リスクを正当化できないが、
/// 粒子合意の詰み（〜1000×q）は通る水準。凍結版はこの名前を知らない。
/// 実測（2026-08-10 採用）: suite 1263→1230・アリーナ ペア3シード
/// 57.1% vs 対照 51.8%（+5.3pt、3シード全勝）・反則/局は対照水準
/// （= 当時は残り2回以上のプローブ経済は不変）
fn last_foul_guard() -> f64 {
    crate::config::current(|c| c.strategy.last_foul_guard)
}

/// `last_foul_guard` の既定値（2026-08-10 採用）。0 で従来挙動へ切り戻し
const LAST_FOUL_GUARD: f64 = 60.0;

/// 残り反則2回の床（`TSUITATE_LAST_FOUL_GUARD_2`、既定 0。env 作業点は 36）。
/// 既定の急峻化は残り2回でも約5.4点。b939bd3 以降の「対v13 60%」向け
/// 拡張なので既定オンにしない。凍結版はこの名前を知らない。
fn last_foul_guard_2() -> f64 {
    crate::config::current(|c| c.strategy.last_foul_guard_2)
}

const LAST_FOUL_GUARD_2: f64 = 0.0;

/// 残り反則3回の床（`TSUITATE_LAST_FOUL_GUARD_3`、既定 0。env 作業点は 16）。
/// 既定の急峻化は残り3回で約3.1点。凍結版はこの名前を知らない。
fn last_foul_guard_3() -> f64 {
    crate::config::current(|c| c.strategy.last_foul_guard_3)
}

const LAST_FOUL_GUARD_3: f64 = 0.0;

/// 残り反則予算に応じた反則コストの床。残り1→2→3 の順に見る。
/// 詰みスケール（〜1000×q）はどの床でも通る。
fn apply_foul_budget_floors(fouls_left: f64, mut foul_cost: f64) -> f64 {
    if fouls_left <= 1.0 {
        foul_cost = foul_cost.max(last_foul_guard());
    } else if fouls_left <= 2.0 {
        foul_cost = foul_cost.max(last_foul_guard_2());
    } else if fouls_left <= 3.0 {
        foul_cost = foul_cost.max(last_foul_guard_3());
    }
    foul_cost
}

/// 反則コストの**床適用前の基準値**（P1-α の `max(k × base, 床)` の base）。
///
/// 診断（issue #31 の P0-4 / P0-5）が価格を付け替えるときに、
/// `evaluate` と同じ式を引くための入口。
pub fn base_foul_cost_for(params: &EvalParams, you: u32, opponent: u32) -> f64 {
    let fouls_left = f64::from((10u32.saturating_sub(you)).max(1));
    let opp_fouls_left = f64::from((10u32.saturating_sub(opponent)).max(1));
    params.foul_cost_base
        * (10.0 / fouls_left).powf(params.foul_cost_pow)
        * (opp_fouls_left / 10.0).powf(params.foul_diff_pow)
}

/// 残り反則数に応じたガード床（`last_foul_guard` 系）だけを返す。
pub fn foul_cost_floor_for(you: u32) -> f64 {
    apply_foul_budget_floors(f64::from((10u32.saturating_sub(you)).max(1)), 0.0)
}

/// 反則コストの実効値（床適用後）＝ `evaluate` が使う値そのもの。
pub fn foul_cost_for(params: &EvalParams, you: u32, opponent: u32) -> f64 {
    let fouls_left = f64::from((10u32.saturating_sub(you)).max(1));
    apply_foul_budget_floors(fouls_left, base_foul_cost_for(params, you, opponent))
}

/// ブラインドの home 占有による**打ちの p_legal 割引**の重み
/// （`TSUITATE_BLIND_HOME_DROP_OCC_W`、既定 0 = 無効）。
/// ブラインド決定では打ちの p_legal が事前確率のみ（マスに依らずほぼ定数）に
/// なり、初期配置マスへの打ち（S*1c = 敵歩の上）がほぼ無割引で反則を連発する
/// （m121/m123 で追加反則 15〜35回/20試行）。擬似粒子の占有 × 鮮度を
/// 安全方向のみ（min）で反則確率に使う。taint_occ_legal_w と同じ形。
/// codex 相談（2026-08-09）で「占有反則と受理後リスクの二重課税を避けつつ
/// まず (b) 占有割引から」と助言された実装順
fn blind_home_drop_occ_w() -> f64 {
    crate::config::current(|c| c.strategy.blind_home_drop_occ_w)
}

/// home 残存の**鮮度減衰**のパラメータ（codex 相談 2026-08-09 の推奨形:
/// `freshness(T) = floor + (1−floor)·exp(−λ·T)`、T = 相手の消化手数）。
/// 相手が動くほど初期配置の情報は古くなるので、home 由来の課金・割引を
/// 一様に減衰させる。歩・香・桂は動きにくいので減衰を半分にする
/// （1三角成の香の取り返しのような隅の物理を保つ）。
/// `TSUITATE_BLIND_HOME_FLOOR` / `TSUITATE_BLIND_HOME_LAMBDA` でスイープ可
fn blind_home_floor() -> f64 {
    crate::config::current(|c| c.strategy.blind_home_floor)
}

fn blind_home_lambda() -> f64 {
    crate::config::current(|c| c.strategy.blind_home_lambda)
}

/// マス上の home 駒の鮮度（そこにまだ居る確率の近似）。role は擬似粒子で
/// そのマスに置いた駒種（None なら速い側の減衰 = 保守的）
fn blind_home_freshness(role: Option<Role>, view: &PlayerView) -> f64 {
    let t = f64::from(view.move_number.saturating_sub(1)) / 2.0;
    let lambda = match role {
        Some(Role::Pawn | Role::Lance | Role::Knight) => blind_home_lambda() * 0.5,
        _ => blind_home_lambda(),
    };
    let floor = blind_home_floor();
    floor + (1.0 - floor) * (-lambda * t).exp()
}

/// ブラインド決定の home 事前モデル: 相手駒を初期配置に**全部**置いた盤面と、
/// 駒種ごとの生存率（残枚数/初期枚数。自分が取った駒は成駒を生駒へ戻して数える）。
///
/// 取った駒を盤面から個別に除かないのが要点（codex 相談 2026-08-09）:
/// どの1枚を取ったかは観測から分からないので、走査順で除くと「実在する
/// 1三の歩」まで消える（初版のバグ。多くの歩を取った終盤で S*1c の占有割引が
/// 発火しなかった）。代わりに占有・リスクの重みへ survival(role) を掛けて
/// 確率質量として扱う。自駒が居るマスの相手駒だけは確実に居ないので置かない。
/// 用途はブラインド決定での mover リスク・打ちの p_legal 割引だけで、
/// 駒得側の供給には使わない（反則マス記憶系が全滅した領域なので安全方向のみ）
struct BlindHome {
    pos: Position,
    /// 自分がそのマスで駒を取ったことがあるか（観測で確定。そのマスの
    /// home 駒は確実にもう居ない）
    captured_at: [bool; 81],
    /// 単一駒（角・飛）の生存率。1枚しか無いので「その駒種を取った」＝
    /// home 駒が消えた、が枚数証拠だけで確定する。複数駒（歩香桂銀金）は
    /// 「どの1枚を取ったか」が分からず、取られやすいのは動いた駒に偏る
    /// （隅の home 残留駒ではない）ので枚数証拠は使わない —
    /// 初版の駒数プール方式は打ち直し→再捕獲の二重計上で歩の生存率が
    /// 0 になり、実在する1三の歩の占有割引が発火しなかった
    bishop_survival: f64,
    rook_survival: f64,
}

impl BlindHome {
    fn survival_at(&self, sq: Coord, role: Role) -> f64 {
        if self.captured_at[crate::belief_features::sq_index(sq)] {
            return 0.0;
        }
        match unpromote_role(role) {
            Role::Bishop => self.bishop_survival,
            Role::Rook => self.rook_survival,
            _ => 1.0,
        }
    }
}

fn blind_home_position(view: &PlayerView, log: &ObservationLog) -> BlindHome {
    let me = view.your_color;
    let opp = me.other();
    let mut pos = Position::empty(me);
    for p in &view.your_pieces {
        if let Some(sq) = parse_usi_square(&p.square) {
            pos.set(
                sq,
                Some(crate::shogi::Piece {
                    color: me,
                    role: p.role,
                }),
            );
        }
    }
    for (sq, p) in Position::initial().pieces() {
        if p.color == opp && pos.piece_at(sq).is_none() {
            pos.set(
                sq,
                Some(crate::shogi::Piece {
                    color: opp,
                    role: p.role,
                }),
            );
        }
    }
    let mut captured_at = [false; 81];
    let mut bishop_survival = 1.0f64;
    let mut rook_survival = 1.0f64;
    for e in log.events() {
        if let Observation::MyMove {
            captured: Some(role),
            usi,
            ..
        } = e
        {
            if let Some(ShogiMove::Board { to, .. }) = parse_usi(usi) {
                captured_at[crate::belief_features::sq_index(to)] = true;
            }
            match unpromote_role(*role) {
                Role::Bishop => bishop_survival = 0.0,
                Role::Rook => rook_survival = 0.0,
                _ => {}
            }
        }
    }
    BlindHome {
        pos,
        captured_at,
        bishop_survival,
        rook_survival,
    }
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
/// 相手駒の占有が観測で裏付けられたマス。
///
/// `material_degen_q0` が「幻の駒得」だけを縮めるための証拠集合:
/// - 相手が自駒を取ったマス（`OpponentMoved.captured_my_piece_at`）=
///   取った駒の位置が通知されるので占有は確定
/// - この手番の非歩打ち反則マス = 候補生成が二歩・行き所を既に除外しているので
///   着地点に相手駒がいる（`exclude_moves_on_known_opponent` と同じ規約）
///
/// 自分が取ったマスは対象外（取った時点で相手駒は消えており、再占有の証拠に
/// ならない）。歩打ち反則も対象外（二歩の可能性がある）。
fn opp_occupancy_evidence(view: &PlayerView, log: &ObservationLog) -> [bool; 81] {
    let mut backed = [false; 81];
    for e in log.events() {
        match e {
            Observation::OpponentMoved {
                captured_my_piece_at: Some(sq),
                ..
            } => {
                if let Some(c) = parse_usi_square(sq) {
                    backed[crate::belief_features::sq_index(c)] = true;
                }
            }
            Observation::MyFoul { move_number, usi } if *move_number == view.move_number => {
                if view.you_in_check {
                    continue;
                }
                if let Some(ShogiMove::Drop { role, to }) = parse_usi(usi) {
                    if role != Role::Pawn {
                        backed[crate::belief_features::sq_index(to)] = true;
                    }
                }
            }
            _ => {}
        }
    }
    backed
}

/// 玉接近減点用の脅威マス（`king_known_approach_w`）。
///
/// `opp_occupancy_evidence` より広い: **手番を問わず**非歩の打ち反則マスを残す。
/// 発端の m099 では 41 手目の `S*5f` 反則で 5六が確定しており、99 手目でも
/// そこに歩が居る（現行の「今手番の打ち反則だけ」では消えてしまう）。
/// 歩打ち反則は二歩の可能性があるので除外。自分がそのマスで取ったら消す
/// （駒はもう居ない）。
fn king_threat_evidence(log: &ObservationLog) -> [bool; 81] {
    let mut backed = [false; 81];
    for e in log.events() {
        match e {
            Observation::OpponentMoved {
                captured_my_piece_at: Some(sq),
                ..
            } => {
                if let Some(c) = parse_usi_square(sq) {
                    backed[crate::belief_features::sq_index(c)] = true;
                }
            }
            Observation::MyFoul { usi, .. } => {
                if let Some(ShogiMove::Drop { role, to }) = parse_usi(usi) {
                    if role != Role::Pawn {
                        backed[crate::belief_features::sq_index(to)] = true;
                    }
                }
            }
            Observation::MyMove {
                usi,
                captured: Some(_),
                ..
            } => {
                if let Some(ShogiMove::Board { to, .. }) = parse_usi(usi) {
                    backed[crate::belief_features::sq_index(to)] = false;
                }
            }
            _ => {}
        }
    }
    backed
}

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
    crate::config::current(|c| c.strategy.belief_gain_w)
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
    crate::config::current(|c| c.strategy.king_net_w)
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
    crate::config::current(|c| c.strategy.king_net_proj)
}

/// taint 玉位置分布とネット分布のブレンド: p = (1−λ)·p_taint + λ·p_net。
/// どちらも正規化済みなので結果も正規化されている（taint が空なら実質ネットのみ）
fn blend_king_dist(taint: &[(Coord, f64)], net: &[(Coord, f64)], lambda: f64) -> Vec<(Coord, f64)> {
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
    candidate_moves_with_log(view, foul_tried, None)
}

/// `candidate_moves` の観測ログつき版。`nonpromote_check_w` の不成生成は
/// `deduce::opp_king_candidates`（観測のみ）を要るのでログが必要。
/// ログ無しの呼び出し（bin/analyze の検証）では従来どおり成り一択になる。
pub fn candidate_moves_with_log(
    view: &PlayerView,
    foul_tried: &HashSet<String>,
    log: Option<&ObservationLog>,
) -> Vec<(String, ShogiMove)> {
    let color = view.your_color;
    // 成ると玉候補への利きが増える手だけ不成の双子を作る（`nonpromote_check_w`）
    let nonpromote_gate: Option<(Vec<(Coord, f64)>, [bool; 81])> = log
        .filter(|_| nonpromote_check_w() > 0.0 && !view.you_in_check)
        .map(|l| (nonpromote_king_dist(view, l), king_threat_evidence(l)))
        .filter(|(d, _)| !d.is_empty());
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
                    // 成れるなら成る、が既定。=1 で不成も生成して評価側に
                    // 判断させる（gen_nonpromote の doc 参照）
                    push(make_usi_move(from, to, true), &mut out);
                    // `TSUITATE_GEN_NONPROMOTE=minor` なら**銀・桂・香だけ**
                    // 不成を生成する。採点済み eval の実測（2026-08-14）:
                    // 銀の不成は 3九銀不成(3h4i)=10 が2局面とも最善で一貫して
                    // 良いのに対し、歩の不成は 10/6/3/1/1 と文脈依存で、
                    // 全駒種で生成すると 7二歩不成（採点1「取れることが確定して
                    // いるので不成にする意味が全くない」）が最大の失点源になる。
                    // 「元の利きを保つ」系（銀桂香）は通常将棋と共通の理屈で
                    // 安定して良く、「王手が増えて宣言で露見するのを避ける」系
                    // （歩飛角）はついたて固有で局面依存、という2系統の差が
                    // そのまま出た形
                    if gen_nonpromote_for(piece.role)
                        || nonpromote_gate.as_ref().is_some_and(|(d, occ)| {
                            promotion_adds_check(view, from, to, piece.role, d, occ)
                        })
                    {
                        push(make_usi_move(from, to, false), &mut out);
                    }
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
pub fn in_check_prior(view: &PlayerView, mv: &ShogiMove) -> f64 {
    match *mv {
        ShogiMove::Board { from, .. } if Some(from) == king_square(view) => 0.5,
        _ => 0.25,
    }
}

/// 観測ゼロでも成り立つ p(合法) の事前確率。
/// 経路上の「中身の見えないマス」1つごとに空である確率 q を掛ける。
/// 打ちは着地点が空である確率 q（隠れた相手駒の上に打つのが典型的な反則源）
pub fn prior_legal(view: &PlayerView, mv: &ShogiMove, opp_board_n: f64) -> f64 {
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
    // ブラインドの home 事前モデル（相手駒を初期配置に置いた盤面＋駒種生存率。
    // `blind_home_risk_w` / `blind_home_drop_occ_w`。choose() が
    // 「w≠0・王手中でない」のゲートを掛けて渡し、厳密粒子が全滅した決定で
    // だけ使う）
    blind_home: Option<&BlindHome>,
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
    // 成りポテンシャルの敵玉近接マップ（`promo_king_prox`。deduce 由来）。
    // None なら重みなし = 従来挙動。promo_pot_before と着手後の両方が
    // 同じマップを使う（差分形の整合）
    promo_prox: Option<&[f64; 81]>,
    // この手番の打ち反則で占有が確定したマスと残存敵駒の平均交換価値
    // （`foul_occ_attack_w`。choose() が「w≠0・王手中でない・今手番に
    // 打ち反則あり」のゲートを掛けて渡す）
    turn_foul_occ: Option<&([bool; 81], f64)>,
    // 持ち駒オプション価値（`hand_option_w`）の決定点コンテキスト。Some のとき
    // だけ項が有効（choose() が「w≠0・王手中でない」のゲートを掛けて渡す）。
    // 打つ手に「最良打ちポテンシャルとの不足分」の減点を掛ける
    hand_option: Option<&HandOption>,
    // 自分が打ちの反則をしたマス（`drop_probe_repeat_gate` の再プローブ判定）
    drop_foul_squares: &[bool; 81],
    // 着手前の自駒の利き枚数（`anchor_move_w` の錨の判定）
    own_attack_before: &[u8; 81],
    // 争点マス = そこで駒が取られた/取ったマス（`anchor_move_w` の争点ゲート）
    contested_squares: &[bool; 81],
    // 相手駒の占有が観測で裏付けられたマス（`material_degen_q0` /
    // `hand_asset_w`）。None ならノブ無効時（choose が渡さない）
    opp_occ_backed: Option<&[bool; 81]>,
    // 未裏付け大駒への新規 threat の鮮度（`stale_threat_w`。1.0 = 減衰なし）
    stale_freshness: f64,
    // 持ち駒資産損（`hand_asset_w`）の玉候補。Some のときだけ項が有効
    // （choose が「w≠0・王手中でない」のゲートを掛けて渡す）
    hand_asset_kings: Option<&std::collections::BTreeSet<Coord>>,
    // 玉接近減点の脅威マス（`king_known_approach_w`）。歴代非歩打ち反則込み
    king_threats: Option<&[bool; 81]>,
    // 信念ネットのマスごと占有（`belief_occ_cap_w`）。厳密粒子が居る決定でも
    // 裏付け無し捕獲の期待駒得をネット占有へ安全方向だけ寄せる。None なら無効
    belief_occ: Option<&[f64; 81]>,
    // 玉プローブ・経路プローブ・玉センサー（`ProbeCtx` の doc）。choose() が
    // 「いずれかの w≠0・王手中でない」のゲートを掛けて渡す
    probe_ctx: Option<&ProbeCtx>,
) -> EvalOut {
    let me = view.your_color;
    let opp = me.other();
    // 玉プローブ / 経路プローブの材料（反則枝の粒子だけが足す）。
    // 回収先のマスごとに貯めて**最後に最大のマスだけ**を採る（codex 指摘。
    // 反則は「どの駒が原因か」を教えないので、粒子ごとの最良を足すと
    // 取りに行けない別マスの回収価値まで二重計上になる）
    let mut king_probe_by_sq = [0.0f64; 81];
    let mut king_probe_mass = 0.0f64;
    let mut path_probe_by_sq = [0.0f64; 81];
    let mut path_probe_mass = 0.0f64;
    let mut legal = 0.0f64;
    let mut value_sum = 0.0;
    let mut risk_sum = 0.0;
    // 着地マスに敵駒がいた（=駒を取れた）粒子の重み。探索ボーナスの不一致度に使う
    let mut capture_hits = 0.0f64;
    // 捕獲価値の重み付き和（賭け分散ペナルティの stake = E[捕獲価値|hit] 用）
    let mut capture_value_sum = 0.0f64;
    // 自玉近傍の敵駒排除ボーナスの粒子加重和（own_zone_capture_w の内訳表示用）
    let mut own_zone_sum = 0.0f64;
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
    // taint フォールバック中の攻め項の倍率（既定 0 = 材料・リスクだけ供給）。
    // 厳密粒子で評価している通常の決定では 1.0 = 従来どおり
    let attack_scale = if particles_are_taint {
        eval_taint_attack_w()
    } else {
        1.0
    };

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
            // 玉プローブ / 経路プローブの材料（`ProbeCtx` の doc）。
            // **回収先のマスごとに集計する**（codex 指摘 2026-08-22）: 反則が
            // 教えるのは「その行き先が違法」だけで**どの駒が原因か**ではない。
            // 粒子ごとに最良の原因駒を選んで足すと、粒子 A が「4七の飛」・
            // 粒子 B が「3三の角」と割れている場合に**両方の回収価値を計上**して
            // しまうが、指し直しで取りに行けるのは1マスだけ
            if let (Some(ctx), ShogiMove::Board { from, to, .. }) = (probe_ctx, *mv) {
                if !particles_are_taint {
                    if Some(from) == ctx.my_king {
                        // 監査モードは w=0 でも材料を集める（スコアには足さない）
                        if ctx.king_probe_w > 0.0 || probe_audit() {
                            let mut hit = false;
                            for (sq, p) in pos.pieces() {
                                if p.color != opp || !pos.attacks(sq, to) {
                                    continue;
                                }
                                let aw = anchor_weight(pos, sq);
                                if aw <= 0.0 || !my_nonking_attacks(pos, sq, me) {
                                    continue;
                                }
                                king_probe_by_sq[crate::belief_features::sq_index(sq)] +=
                                    w * aw * recovery_value(pos, sq, p.role, me);
                                hit = true;
                            }
                            if hit {
                                king_probe_mass += w;
                            }
                        }
                    } else if ctx.path_probe_w > 0.0 || probe_audit() {
                        if let Some((sq, v)) = path_probe_material(pos, from, to, me) {
                            path_probe_by_sq[crate::belief_features::sq_index(sq)] += w * v;
                            path_probe_mass += w;
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

        // 玉センサー（`ProbeCtx` の doc）: 玉の手だけ、現在位置との差分
        if let (Some(ctx), ShogiMove::Board { from, to, .. }) = (probe_ctx, *mv) {
            if ctx.king_sensor_w > 0.0 && Some(from) == ctx.my_king && !particles_are_taint {
                let after = king_sensor_value(&next, to, me, ctx);
                let before = king_sensor_value(pos, from, me, ctx);
                v += ctx.king_sensor_w * (after - before);
            }
        }

        // 王手・詰み。ついたて将棋では王手された側は王手駒の位置が見えず
        // 手探りの反則をしやすい（反則10回で負け）ので、王手自体が得点源。
        // 相手の反則が溜まっているほど価値が上がり、上限（反則負け）に
        // 近づくほど1回の誘発の限界価値が跳ねるので check_limit_accel で加速する
        // （オラクル測定 2026-07-16: 王手中反則の完全知識だけで vs v6 +9.5pt）
        let gives_check = next.in_check(opp);
        if gives_check {
            let opp_fouls_left = f64::from(10u32.saturating_sub(view.fouls.opponent).max(1));
            let accel = (10.0 / opp_fouls_left).powf(params.check_limit_accel);
            // taint フォールバック中は攻め項を絞る（`eval_taint_attack_w`。
            // 玉位置ビリーフが攻め先の決定に足りない = 駒捨て王手の濫発）
            v += attack_scale
                * (params.check_bonus
                    + params.check_foul_scale * f64::from(view.fouls.opponent) * accel);
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
                v += attack_scale
                    * params.check_strength_w
                    * (CHECK_STRENGTH_CURVE / (1.0 + resolutions as f64) - CHECK_STRENGTH_CENTER);
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
        // 自玉近傍の敵駒の排除（own_zone_capture_w）: 自玉の距離≤2 にいる相手駒を
        // 取る手は、交換価値とは別に「自陣の脅威が1枚減る」価値を持つ（arena-recap01:
        // 1二で角を取られた直後、飛車で取り返せば自玉3一の隣接圏から敵駒が消える
        // のに、幻の紐（1五の香 52%）へのリスクが勝って 3五歩。取り返さなかった
        // 成駒が30手後の詰みの材料になった）。王手中は無効。
        // 係数3つ（codex 相談 2026-08-20 の改良: 初版の定額 w=10 は suite +0.12 だが
        // 隣接の歩を取る手にも飛車級の別価値を与えていた）:
        // - 距離: 1 は満額 / 2 は半額
        // - 駒種の危険度: 生の歩香桂 0.25 / 金銀・成小駒 0.75 / 飛角竜馬 1.0
        //   （近いだけの香と玉頭のと金を区別する）
        // - 観測裏付け: 着地マスの相手駒が観測で裏付けられていなければ 0.5
        //   （幻の駒の排除に満額を払わない。opp_occupancy_evidence と共通）
        if params.own_zone_capture_w != 0.0 && !view.you_in_check && captured_value > 0.0 {
            if let (ShogiMove::Board { to, .. }, Some(mk)) = (*mv, pos.king_square(me)) {
                let dist_mult = match cheb(to, mk) {
                    1 => 1.0,
                    2 => 0.5,
                    _ => 0.0,
                };
                if dist_mult > 0.0 {
                    let role_mult = match pos.piece_at(to).map(|p| p.role) {
                        Some(Role::Pawn | Role::Lance | Role::Knight) => 0.25,
                        Some(
                            Role::Silver
                            | Role::Gold
                            | Role::Tokin
                            | Role::Promotedlance
                            | Role::Promotedknight
                            | Role::Promotedsilver,
                        ) => 0.75,
                        Some(Role::Bishop | Role::Rook | Role::Horse | Role::Dragon) => 1.0,
                        _ => 0.0,
                    };
                    // None は呼び出しゲート（choose の opp_occ_backed 計算条件に
                    // own_zone_capture_w > 0 を含めてある）の取りこぼしなので、
                    // 安全側 = 裏付け無し扱いに落とす（幻の駒の排除に満額を払わない）
                    debug_assert!(opp_occ_backed.is_some(), "own_zone_capture_w > 0 なのに opp_occ_backed が None");
                    let backed_mult = match opp_occ_backed {
                        Some(backed) if backed[crate::belief_features::sq_index(to)] => 1.0,
                        _ => 0.5,
                    };
                    let bonus =
                        params.own_zone_capture_w * dist_mult * role_mult * backed_mult;
                    v += bonus;
                    own_zone_sum += w * bonus;
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
        // 成る手のリスクは成る前の駒価値で数える（promo_risk_prerole の doc 参照。
        // 既定は従来どおり own_after = 成った後の駒種）
        let own_risk = match *mv {
            ShogiMove::Board {
                from,
                promote: true,
                ..
            } if promo_risk_prerole() => pos
                .piece_at(from)
                .map(|p| exchange_value(p.role))
                .unwrap_or(own_after),
            _ => own_after,
        };
        // 玉隣接への無支え進入（king_adj_entry_w）: この粒子で着地マスが相手玉の
        // 8近傍にあり自分の利きの支えが無いなら、王手宣言・接触で存在がバレて
        // 玉や近傍の守りに回収される前提の回収リスクを駒価値スケールで敷く
        // （実測: 直接王手の53〜56%が即取られ。quest31 の 4一龍/4一と が発端）。
        // 王手中は無効（回避手の序列は CheckSolver の領分）
        if params.king_adj_entry_w != 0.0 && !view.you_in_check {
            if let Some(ok) = next.king_square(opp) {
                if cheb(to, ok) <= 1 && next.attack_count(to, me) == 0 {
                    v -= params.king_adj_entry_w * own_risk;
                }
            }
        }
        let known_factor = if captured_value > 0.0 {
            1.0
        } else {
            params.camp_known_quiet
        };
        let mut floor = own_risk * camp_defended_prior(to, me, params.camp_scale) * known_factor;
        if captured_value > 0.0 {
            // 取ったマスは相手に通知される。粒子に守りが見えなくても
            // 取り返しの残留リスクを敷く（= 等価な取りは安い駒で取る）。
            // 玉は exchange_value=0 でこの床を素通りしていた（quest31-m076:
            // 銀で取れる駒を玉で取る手が首位）ので、露見するのが玉のときは
            // 実効価値 king_capture_reveal に置き換える（既定0 = 従来挙動）。
            // **王手中は適用しない**: 王手宣言で自玉の位置は既に漏れており、
            // 王手駒の玉捕獲は CheckSolver / check_king_prior の領分
            // （観測確実な取り返しベイト recap-dragon が 16→14 に沈む実測）
            let reveal_after =
                if !view.you_in_check && next.piece_at(to).is_some_and(|p| p.role == Role::King) {
                    params.king_capture_reveal.max(own_risk)
                } else {
                    own_risk
                };
            floor = floor.max(reveal_after * params.capture_reveal_risk);
        }
        let mut recap = recapture_risk(&next, me, to, params.recapture_defended);
        if own_risk != own_after && own_after > 0.0 {
            // recapture_risk は着手後の駒種で価値を数えるので成る前の価値へ換算
            recap *= own_risk / own_after;
        }
        let mover_risk = mover_w * recap.max(floor);
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
        v += params.threat_w
            * if stale_freshness < 1.0 {
                threat_value_stale(pos, &next, me, opp_occ_backed, stale_freshness)
            } else {
                threat_value(&next, me)
            };

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
            // 不成候補（gen_nonpromote 時のみ存在）の NN は**成り側の双子**で
            // 評価する: NN の学習データは「成れるなら成る」の自動成りのみで、
            // 成/不成の差分は分布外ノイズ（実測 2026-08-09: ΔNN が seed により
            // ±0.8 振れ、リスク・静的gainが同額なのに確定捕獲の 4七歩不成が
            // 成りの上へ浮く = ユーザー指摘の悪手）。成/不成の差は手作り項
            // （露見・利き・promo）だけに担わせる
            let promoted_twin = match *mv {
                ShogiMove::Board {
                    from,
                    to,
                    promote: false,
                } if pos.piece_at(from).is_some_and(|p| {
                    promotion_choice(p.role, from, to, me) == Promotion::Optional
                }) =>
                {
                    let mv2 = ShogiMove::Board {
                        from,
                        to,
                        promote: true,
                    };
                    let mut next2 = pos.clone();
                    next2.play_unchecked(&mv2);
                    Some((mv2, next2))
                }
                _ => None,
            };
            let trans = match &promoted_twin {
                Some((mv2, next2)) => {
                    crate::value_features::transition_features(pos, mv2, next2, me)
                }
                None => crate::value_features::transition_features(pos, mv, &next, me),
            };
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
    let mut p_legal = blend_p_legal(
        legal,
        n,
        prior,
        particles_are_taint,
        budget.eval_particles,
        params,
    );
    // taint 占有合意で打ちの反則確率を締める（`taint_occ_legal_w`）。
    // 打ちマスが埋まっていれば必ず反則なので、合意占有率をそのまま反則確率に
    // 使える。**安全方向のみ**（min）: 空きマスの打ちを押し上げはしない
    if let (Some(occ), ShogiMove::Drop { to, .. }) = (taint_occ, *mv) {
        let p_occ = occ[crate::belief_features::sq_index(to)];
        p_legal = p_legal.min(1.0 - params.taint_occ_legal_w * p_occ).max(0.0);
    }
    // ブラインドの home 占有による打ちの p_legal 割引（`blind_home_drop_occ_w`、
    // 安全方向のみ = min）。初期配置マスへの打ちは擬似粒子の占有 × 生存率 ×
    // 鮮度の確率で反則になる。空きマスの打ちは押し上げない
    if blind_home_drop_occ_w() > 0.0 && particles.is_empty() {
        if let (Some(bh), ShogiMove::Drop { to, .. }) = (blind_home, *mv) {
            if let Some(p) = bh.pos.piece_at(to).filter(|p| p.color == opp) {
                let p_occ = bh.survival_at(to, p.role) * blind_home_freshness(Some(p.role), view);
                p_legal = p_legal.min(1.0 - blind_home_drop_occ_w() * p_occ).max(0.0);
            }
        }
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
                    .map(|p| {
                        // 玉の露見コスト免除の穴を塞ぐ（上の床と同じ規約。
                        // 王手中は適用しない = recap-dragon の取り返しベイト保護）
                        if p.role == Role::King && !view.you_in_check {
                            params.king_capture_reveal
                        } else {
                            exchange_value(p.role)
                        }
                    })
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
            let stake =
                mean_value - params.mover_w_captured * own_after * params.capture_reveal_risk;
            belief_gain_w() * (p * stake - params.capture_bet_var_w * p * (1.0 - p) * mean_value)
        }
        _ => 0.0,
    };
    // ブラインド進入リスク（`blind_home_risk_w`、既定 0）: 厳密粒子ゼロの
    // 決定では expected ごと mover リスクが消え、敵陣の初期配置マスへの
    // 成り込み（1三角成 = 初期位置の歩を取り 1一の香に取り返される歩角交換）が
    // 無コストで浮く。相手駒を初期配置に置いた擬似粒子1個に対して
    // **mover リスクだけ**を通常経路と同じ式（露見床・取り返し・prerole 換算）で
    // 課金する。駒得側は供給しない（安全方向のみ）。blind_recapture のマスは
    // あちらが同じ式でリスクを引くので二重計上しない。王手中は無効
    let blind_home_risk = match (particles.is_empty(), blind_home, *mv) {
        (true, Some(bh), ShogiMove::Board { from, to, .. })
            if !view.you_in_check
                && blind_recapture_target.is_none_or(|(sq, _)| sq != to)
                && bh.pos.is_legal(mv) =>
        {
            let occ_piece = bh.pos.piece_at(to).filter(|p| p.color == opp);
            let captured_value = occ_piece.map(|p| exchange_value(p.role)).unwrap_or(0.0);
            let mut next_p = bh.pos.clone();
            next_p.play_unchecked(mv);
            let own_after = next_p
                .piece_at(to)
                .map(|p| exchange_value(p.role))
                .unwrap_or(0.0);
            // 成る手のリスク価格は通常経路と同じ規約（promo_risk_prerole）
            let own_risk = match *mv {
                ShogiMove::Board {
                    from: f2,
                    promote: true,
                    ..
                } if promo_risk_prerole() => bh
                    .pos
                    .piece_at(f2)
                    .map(|p| exchange_value(p.role))
                    .unwrap_or(own_after),
                _ => own_after,
            };
            let _ = from;
            let mover_w = if captured_value > 0.0 {
                params.mover_w_captured
            } else {
                params.mover_w_quiet
            };
            let known_factor = if captured_value > 0.0 {
                1.0
            } else {
                params.camp_known_quiet
            };
            let mut floor =
                own_risk * camp_defended_prior(to, me, params.camp_scale) * known_factor;
            if captured_value > 0.0 {
                floor = floor.max(own_risk * params.capture_reveal_risk);
            }
            let mut recap = recapture_risk(&next_p, me, to, params.recapture_defended);
            if own_risk != own_after && own_after > 0.0 {
                recap *= own_risk / own_after;
            }
            // 鮮度減衰 × 生存率（codex 相談 2026-08-09）: 相手が動くほど home
            // 事前は古くなり、取った駒種ほど残っていない。着地マスの占有駒の
            // 鮮度×生存率で全体を割り引く（捕獲でなければ保守的に速い側の減衰）。
            // home 事前が古い局面で正しい侵入まで課税していた過課金
            // （m119/m044/m034）への対策
            let fresh = match occ_piece {
                Some(p) => bh.survival_at(to, p.role) * blind_home_freshness(Some(p.role), view),
                None => blind_home_freshness(None, view),
            };
            blind_home_risk_w() * fresh * mover_w * recap.max(floor)
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
            capture_bet_penalty = params.capture_bet_var_w
                * p_hit
                * (1.0 - p_hit)
                * (capture_value_sum / capture_hits);
        }
        // **材料の退化ゲート**（`material_degen_q0`、既定 0 = 従来と同一挙動）:
        // 駒得の期待値は退化した粒子集合でも満額で効く（confidence ゲートは
        // 攻め項にしか掛かっていなかった）。実測（2026-08-10、m067 の
        // `scenario diag`）: 生存した厳密粒子が9個しかない決定点で、真実は
        // 空きマスの 4一 に **飛車 85.9%** の信念が立ち、そこへの成桂
        // （4一成桂）が浮いていた（ユーザー指摘の手）。同型の 8e7g+ も
        // 駒得 +3.575 が駆動源。少数粒子の合意は「自信を持って間違う」ので、
        // 質量が薄いほど駒得期待値を縮める。
        //
        // **縮めるのは観測裏付けの無い捕獲だけ**: 2026-08-10 の全捕獲版は
        // m032/m063 の正しい捕獲まで殺し不採用。裏付けマス（相手が自駒を
        // 取った／この手番の非歩打ち反則）への捕獲は満額残し、幻の
        // 3三角成クラスだけを沈める。
        // g = c(1+q0)/(c+q0): q0=0 で g=1（従来）、q0>0 で c 小さいほど強く縮む
        let degen_gate = if params.material_degen_q0 > 0.0 {
            confidence * (1.0 + params.material_degen_q0) / (confidence + params.material_degen_q0)
        } else {
            1.0
        };
        let capture_to_backed = match (*mv, opp_occ_backed) {
            (ShogiMove::Board { to, .. }, Some(backed)) => {
                backed[crate::belief_features::sq_index(to)]
            }
            _ => false,
        };
        let capture_ev = if legal > 0.0 && capture_hits > 0.0 {
            capture_value_sum / legal
        } else {
            0.0
        };
        let material_shrink = if degen_gate < 1.0 && capture_ev > 0.0 && !capture_to_backed {
            (1.0 - degen_gate) * capture_ev
        } else {
            0.0
        };
        // 信念ネット占有キャップ（`belief_occ_cap_w`、既定 0）。質量ゲートの後
        // の残り駒得に対して、ネットが空き寄りと見ているマスへの裏付け無し
        // **大駒**捕獲だけ縮める。金銀キャンセルより先に掛け、残りを
        // gs_unbacked が受け取る（二重控除しない）
        let remaining_after_degen = (capture_ev - material_shrink).max(0.0);
        let belief_occ_shrink = if belief_occ_cap_w() > 0.0
            && !view.you_in_check
            && remaining_after_degen > 0.0
            && !capture_to_backed
        {
            match (*mv, belief_occ) {
                (ShogiMove::Board { from, to, .. }, Some(occ)) => {
                    let mover_major = view
                        .your_pieces
                        .iter()
                        .find(|p| p.square == make_usi_square(from))
                        .is_some_and(|p| {
                            matches!(
                                p.role,
                                Role::Bishop | Role::Rook | Role::Horse | Role::Dragon
                            )
                        });
                    if !mover_major {
                        0.0
                    } else {
                        let p_occ = occ[crate::belief_features::sq_index(to)];
                        let shrink = belief_occ_cap_shrink(
                            remaining_after_degen,
                            p_hit,
                            p_occ,
                            belief_occ_cap_w(),
                        );
                        if crate::hits::enabled() {
                            crate::hits::flag("belief_occ_cap", shrink > 0.05);
                        }
                        shrink
                    }
                }
                _ => 0.0,
            }
        } else {
            0.0
        };
        let remaining_after_belief = (remaining_after_degen - belief_occ_shrink).max(0.0);
        // 金銀の裏付け無し捕獲をキャンセル（`unbacked_gs_capture_w`）。
        // 信念ネットも「居る」と見ているマスで金銀が幻の駒得を残すときの床。
        // 王手中・裏付けマス・と金歩桂香は対象外。
        let gs_unbacked_capture = if unbacked_gs_capture_w() > 0.0
            && !view.you_in_check
            && view.move_number >= UNBACKED_GS_CAPTURE_MIN_MOVE
            && remaining_after_belief > 0.0
            && !capture_to_backed
        {
            let mover_gs = matches!(
                *mv,
                ShogiMove::Board { from, .. } if view
                    .your_pieces
                    .iter()
                    .find(|p| p.square == make_usi_square(from))
                    .is_some_and(|p| matches!(p.role, Role::Gold | Role::Silver))
            );
            if mover_gs {
                unbacked_gs_capture_w() * remaining_after_belief
            } else {
                0.0
            }
        } else {
            0.0
        };
        // 粒子上の詰みの加点。q = 詰みを主張する質量の割合に対し
        // 1000×q×(q/(q+q0)) の凸ゲート: q0=0 は従来（1000×q）と同一挙動、
        // q0>0 では裾の幻詰みが材料スケールへ沈み、合意の詰みはほぼ満額残る
        let mate_term = if mate_hits > 0.0 {
            let q = mate_hits / legal;
            1000.0 * q * (q / (q + params.mate_gate_q0))
        } else {
            0.0
        };
        // 攻め側の項（王探し・玉周りの圧力・逃げマス被覆）は taint
        // フォールバック中は attack_scale で絞る。守り側（自玉への圧力・
        // 相手の打ち王手の危険）は絞らない = 安全方向は残す
        value_sum / legal + mate_term
            - capture_bet_penalty
            - material_shrink
            - belief_occ_shrink
            - gs_unbacked_capture
            + params.info_bonus * p_hit * (1.0 - p_hit)
            + attack_scale * params.king_probe_bonus * p_chk * (1.0 - p_chk)
            + value_nn_term
            + (attack_scale * params.attack_w * confidence * attack_sum
                + attack_scale * params.escape_cover_w * confidence * escape_sum
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
    // 残り反則が少ないときのガード。既定の急峻化は残り1回でも約13点、
    // 残り2回で約5.4点にしかならず、gain 5〜8 の手が「10%反則でも指す」
    // 計算を通して反則負けまで打ち尽くす（アリーナの ~76% が foul_limit）。
    // 床は材料スケールでは正当化できないが、粒子合意の詰み（〜1000×q）は通る。
    let foul_cost = foul_cost_for(params, view.fouls.you, view.fouls.opponent);

    // 前進の弱い事前バイアス（推定が薄い序盤に駒をぶつけに行くため）
    // 成りの固定ボーナス（`promote_bias`）の駒種分け（2026-08-10）:
    // - 歩・角・飛: 成ると利きが増え／ついたてでは王手露見回避の不成もあるが、
    //   成り側へ `promote_bias` を付ける（従来どおり）
    // - 桂・銀・香: 成ると元の利き（跳び・斜め後ろ・縦走り）を失うので、
    //   **不成側**へ同額を付ける（成り側は 0）。不成が候補に載る前提なので
    //   **`gen_nonpromote()` が有効なときだけ**付け替える（従来生成は成り一択
    //   なので、付け替えたままだと桂銀香の成りだけ promote_bias を失う）。
    //   発端は quest31 の 4九銀成（3h4i+、不成=10 / 成=2）
    let advance_bias = match *mv {
        ShogiMove::Board { from, to, promote } => {
            let adv = match me {
                Color::Sente => (from.rank - to.rank) as f64,
                Color::Gote => (to.rank - from.rank) as f64,
            };
            let role = view
                .your_pieces
                .iter()
                .find(|p| p.square == make_usi_square(from))
                .map(|p| p.role);
            let promo_bonus = if gen_nonpromote() || gen_nonpromote_minor() {
                match (promote, role) {
                    (true, Some(Role::Silver | Role::Knight | Role::Lance)) => 0.0,
                    (false, Some(r @ (Role::Silver | Role::Knight | Role::Lance)))
                        if promotion_choice(r, from, to, me) != Promotion::None =>
                    {
                        params.promote_bias
                    }
                    (true, _) => params.promote_bias,
                    (false, _) => 0.0,
                }
            } else if promote {
                params.promote_bias
            } else {
                0.0
            };
            params.advance_w * adv + promo_bonus
        }
        ShogiMove::Drop { .. } => params.drop_bias,
    };

    // 大駒を初期位置に置き続けるペナルティ（この手の後に残る枚数分）。
    // 動かす手だけペナルティが軽くなるので、展開への勾配になる
    let development = -params.big_home_penalty * big_home_after(view, mv);

    // 利き被覆（広い索敵網）。粒子に依存しない自明な情報だけで計算できる
    let own_effects = own_effects_after(view, mv, opp_king_w, promo_prox, params);
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
                    * (h - piece_promo_potential(&ctx.occupied, to, role, me, None)).max(0.0)
            })
            .unwrap_or(0.0),
        _ => 0.0,
    };

    // 持ち駒の資産損（`hand_asset_w` の doc 参照）。仕事の無い打ちだけを
    // 駒価値で課税。移動の手は 0 なのでゼロ点は動かない
    let hand_asset_pen = match (hand_asset_kings, opp_occ_backed, mv) {
        (Some(cands), Some(backed), &ShogiMove::Drop { role, to }) => {
            let w = hand_asset_w();
            let taxable = hand_asset_drop_taxable(role, to, view.your_color);
            if w <= 0.0 || !taxable || drop_has_hand_asset_work(view, role, to, backed, cands) {
                0.0
            } else {
                let amount = if role == Role::Pawn {
                    2.0
                } else {
                    exchange_value(role)
                };
                w * amount
            }
        }
        _ => 0.0,
    };

    // 玉の既知脅威への接近（`king_known_approach_w` の doc 参照）。
    // 王手中も有効: m099 は 8五桂の王手で、逃げ先の序列（8八 vs 6八）が
    // CheckSolver の解消確率だけでは決まらない。脅威マスを取る手は amount=0。
    let king_approach_pen = match (king_threats, mv) {
        (Some(threats), &ShogiMove::Board { from, to, .. }) if king_square(view) == Some(from) => {
            let w = king_known_approach_w();
            if w <= 0.0 {
                0.0
            } else {
                w * king_known_approach_amount(from, to, threats)
            }
        }
        _ => 0.0,
    };

    // 終盤の玉の非捕獲逃げ（`king_endgame_flee_w`）。王手中の空マスも課税。
    let king_flee_pen = match mv {
        &ShogiMove::Board { from, to, .. } if king_square(view) == Some(from) => {
            let w = king_endgame_flee_w();
            if w <= 0.0 {
                0.0
            } else {
                w * king_endgame_flee_amount(to, view, opp_occ_backed)
            }
        }
        _ => 0.0,
    };

    // 終盤の金が自玉へ隣接する（`gold_join_king_w`）。王手中の盤上移動。
    // CheckSolver が打ち込み王手駒を仮説に持てないので p_legal 割引の
    // **外側**（`foul_probe` に加算）へ置く。
    // 非王手で既に隣接している金の玉筋移動は `gold_king_file_w`（gain 内）。
    let (gold_join, gold_file_guard) = match mv {
        &ShogiMove::Board { from, to, .. } => match king_square(view) {
            Some(king)
                if view
                    .your_pieces
                    .iter()
                    .find(|p| p.square == make_usi_square(from))
                    .is_some_and(|p| p.role == Role::Gold) =>
            {
                let join = {
                    let w = gold_join_king_w();
                    if w > 0.0 {
                        w * gold_join_king_amount(
                            from,
                            to,
                            king,
                            view.move_number,
                            view.you_in_check,
                        )
                    } else {
                        0.0
                    }
                };
                let guard = {
                    let w = gold_king_file_w();
                    if w > 0.0 {
                        w * gold_king_file_amount(
                            from,
                            to,
                            king,
                            view.move_number,
                            view.you_in_check,
                        )
                    } else {
                        0.0
                    }
                };
                (join, guard)
            }
            _ => (0.0, 0.0),
        },
        _ => (0.0, 0.0),
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
    let (mate_threat, mate_risk) =
        if (params.mate_threat_w != 0.0 || params.mate_risk_w != 0.0) && !view.you_in_check {
            // 厳密粒子は正確なぶん退化度（confidence）で割り引く。taint 粒子は
            // 重み自体に 0.5^(taint-1) の減衰が入っているのでそのまま使う
            let conf = if particles.is_empty() {
                1.0
            } else {
                confidence
            };
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

    // **錨外し**（`anchor_move_w`、2026-08-04、ユーザー指摘の61手目が発端。
    // codex 相談で「anchor removal penalty」として一般化）。
    // 争点マス `to` を支えている自駒**自身**をそこへ動かすと、占拠と守りを
    // 同じ駒に背負わせることになり、読みが外れて空だった／取り返された
    // ときに取り返す駒が残らない。実例: 61手目の 4七金 — 4七を守っているのは
    // 5七金だけで、その金を4七へ動かすと外部の支えがゼロになる（人間の判定は
    // 悪手で、正着は支えを残したまま安い歩を打つ P*4g）。
    //
    // **`gain` の内側**に置くこと（= p_legal 割引を受ける）。同じ着想を
    // `foul_probe`（combine_score の外側）へ載せた版は反則コストを迂回して
    // 反則/局 6.4→8.1・−12.8pt だった（`drop_probe_repeat_gate` の doc）。
    //
    // 自駒の配置だけで決まるので粒子不要・ノイズゼロ。王手中は CheckSolver の
    // 領分なので無効。玉の手は recapture_risk が元々ゼロ扱いなので対象外
    let anchor_move_pen = if params.anchor_move_w != 0.0 && !view.you_in_check {
        match *mv {
            ShogiMove::Board { from, to, .. } => {
                let idx = crate::belief_features::sq_index(to);
                // **争点ゲート**（第1版に無くて −20.6pt を出した因子。
                // codex の `contested(to)`）。争点は観測だけで正確に分かる:
                // **そこで駒が取られた/取ったマス**（相手は取られたマスを
                // 通知されるので双方が注目している）。61手目の4七はまさに
                // 58手目に自分の歩が取られたマス。全マスに当たらないので
                // 発火率が低い（`anchor_move_fires` フックで実測する）
                let contested = contested_squares[idx];
                // 着手後に `to` を守る自駒の枚数（着手駒自身を除く）
                let after = own_attack_before[idx]
                    .saturating_sub(u8::from(own_defends_from(view, from, to)));
                let mover = view
                    .your_pieces
                    .iter()
                    .find(|p| p.square == make_usi_square(from))
                    .filter(|p| p.role != Role::King);
                // 争点へ乗ったあと**自分の支えが一枚も残らない**ときだけ罰する
                // （残っていれば取り返せるので形は壊れていない）
                let fires = matches!(mover, Some(_)) && contested && after == 0;
                // `hits::flag` は内部でグローバル Mutex を取るので、
                // 既存の呼び出しと同じく enabled() でガードすること
                if crate::hits::enabled() {
                    crate::hits::flag("anchor_move_fires", fires);
                }
                match mover {
                    Some(p) if fires => params.anchor_move_w * exchange_value(p.role),
                    _ => 0.0,
                }
            }
            ShogiMove::Drop { .. } => 0.0,
        }
    } else {
        0.0
    };

    // この手番の打ち反則で占有が確定したマスへ、着手駒（玉以外）が着手後に
    // 利きを付ける手への加点（`foul_occ_attack_w`。EvalParams の doc 参照）。
    // 反則直後は手番が保たれるので確定情報は完全に新鮮 = 次の相手の1手より
    // 前にこの当たりが立つ。玉は露見リスク側（king_capture_reveal）の領分
    let foul_occ_attack = match turn_foul_occ {
        Some((occ, mean_val)) if !view.you_in_check => {
            let mut pieces = view.your_pieces.clone();
            let moved: Option<usize> = match *mv {
                ShogiMove::Board { from, to, promote } => {
                    let from_usi = make_usi_square(from);
                    pieces
                        .iter()
                        .position(|p| p.square == from_usi)
                        .filter(|&i| pieces[i].role != Role::King)
                        .inspect(|&i| {
                            if promote {
                                if let Some(r) = promote_role(pieces[i].role) {
                                    pieces[i].role = r;
                                }
                            }
                            pieces[i].square = make_usi_square(to);
                        })
                }
                ShogiMove::Drop { role, to } => {
                    pieces.push(VisiblePiece {
                        square: make_usi_square(to),
                        role,
                    });
                    Some(pieces.len() - 1)
                }
            };
            match moved {
                Some(i) => {
                    let p = pieces[i].clone();
                    let hits_confirmed = crate::board::defend_targets(&pieces, &p, me)
                        .iter()
                        .any(|&s| occ[crate::belief_features::sq_index(s)]);
                    if hits_confirmed {
                        // 「同じ仕事なら最安の駒で」: 当てる駒が高いほど、
                        // 取り返された/かわされたときの損が大きい。
                        // 素の実測では 6六桂打（悪手・桂3.5）が 7七歩打
                        // （正解・歩1.0）と同点で上に来た
                        params.foul_occ_attack_w * mean_val / (1.0 + exchange_value(p.role))
                    } else {
                        0.0
                    }
                }
                None => 0.0,
            }
        }
        _ => 0.0,
    };

    let gain = expected + advance_bias + development + coverage + mate_threat
        - mate_risk
        - king_holes
        - anchor_move_pen
        + link
        + drop_hit_evac
        + promo
        + major_path
        - hand_option_pen
        - hand_asset_pen
        - king_approach_pen
        - king_flee_pen
        + gold_file_guard
        - board_discount
        + effect_value
        + foul_occ_attack
        + blind_recapture
        + blind_belief_gain
        - blind_home_risk;
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
    // 監査用（スコアには足さない。`CandidateScore::probe_unit` の doc）
    let mut probe_unit = 0.0f64;
    let mut probe_mass = 0.0f64;
    let mut probe_concentration = 0.0f64;
    // 玉プローブ / 経路プローブ（`ProbeCtx` の doc）。全粒子質量 n で正規化するので
    // (1−p_legal) の重みを内包している。玉プローブは占有の凸ゲートを掛けない
    // （失うものが無い確かめ方なので「居そうなときだけ」の制約は要らない）が、
    // 過去手番で玉が反則した行き先は価値ゼロ（再プローブ）
    let foul_probe = foul_probe
        + match (probe_ctx, *mv) {
            (Some(ctx), ShogiMove::Board { from, to, .. }) if n > 0.0 => {
                let budget_frac = f64::from(10u32.saturating_sub(view.fouls.you)) / 10.0;
                // **凸ゲートは掛けない**（由来タグ導入後。codex 指摘 2026-08-22）:
                // drop_probe_w の p_occ² は「確信が低いマスを探らない」ためだが、
                // ここでは**由来タグ**（観測に裏付けられた駒か）が判別を担う。
                // 質量ゲートを重ねると age 減衰と合わせて実効二乗になり、本物の
                // 信号まで 7 倍沈む（実測: quest_0809-m036 のプローブ寄与が
                // 期待 1.8 に対し 0.046 まで落ちていた）。
                // 幻の抑制は anchor_weight が 0 を返すことで既に効いている。
                //
                // ただし **(1 − p) の情報価値は要る**（drop_probe_w と同じ経済。
                // CLAUDE.md の教訓「反則1回で買えるのは情報であり、p_occ が高い
                // マスに買う情報は無い」）。既に確信しているマスへ玉を出しても
                // 学ぶことは無く、その駒は普通に取ればよい。これが無いと
                // quest31-m057 で「94% の粒子が 5八 を塞いでいる」玉の手が
                // score 37.9 まで跳ねた（2026-08-22 実測）
                let best = |acc: &[f64; 81]| acc.iter().copied().fold(0.0, f64::max);
                if Some(from) == ctx.my_king {
                    if king_probe_mass > 0.0 && !ctx.stale_king_dests.contains(&to) {
                        let info = 1.0 - (king_probe_mass / n).clamp(0.0, 1.0);
                        let unit = (best(&king_probe_by_sq) / n) * info * budget_frac * budget_frac;
                        probe_unit = unit;
                        probe_mass = king_probe_mass / n;
                        probe_concentration = if king_probe_mass > 0.0 {
                            (best(&king_probe_by_sq) / king_probe_mass).min(1.0)
                        } else {
                            0.0
                        };
                        ctx.king_probe_w * unit
                    } else {
                        0.0
                    }
                } else if path_probe_mass > 0.0 {
                    let info = 1.0 - (path_probe_mass / n).clamp(0.0, 1.0);
                    let unit = (best(&path_probe_by_sq) / n) * info * budget_frac * budget_frac;
                    probe_unit = unit;
                    probe_mass = path_probe_mass / n;
                    ctx.path_probe_w * unit
                } else {
                    0.0
                }
            }
            _ => 0.0,
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
        own_zone: if legal > 0.0 { own_zone_sum / legal } else { 0.0 },
        capture_value: if legal > 0.0 {
            capture_value_sum / legal
        } else {
            0.0
        },
        link,
        promo,
        hand_option: hand_option_pen,
        board_discount,
        foul_probe: foul_probe + gold_join,
        probe_unit,
        probe_mass,
        probe_concentration,
        // 候補ループが引いた乱数を後から書き戻す（evaluate は乱数を持たない）
        tiebreak: 0.0,
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
        // 成る手のリスクは成る前の駒価値で数える（promo_risk_prerole。
        // stage1 の own_risk と同じ規約を、応手で着手駒が取られる分岐にも適用）
        let mover_prerole = match *mv {
            ShogiMove::Board {
                from,
                promote: true,
                ..
            } if promo_risk_prerole() => pos.piece_at(from).map(|p| exchange_value(p.role)),
            _ => None,
        };
        for (reply, rw) in &replies {
            let reply_to = match *reply {
                ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
            };
            let mut lost = next
                .piece_at(reply_to)
                .filter(|p| p.color == me)
                .map(|p| exchange_value(p.role))
                .unwrap_or(0.0);
            if reply_to == to && lost > 0.0 {
                if let Some(pre) = mover_prerole {
                    lost = pre;
                }
            }
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
/// threat_value の**鮮度減衰版**（`stale_threat_w` 有効時のみ。doc は
/// stale_threat_w 参照）。対象駒ごとに「着手で新たに増えた threat 増分」
/// created = max(0, after − before) を計算し、対象が**観測裏付けのない
/// 飛角（成駒込み）**のときだけ effective = after − (1−freshness)×created と
/// する。既存 threat の基準値は沈めない・当たりを外す手を免税しない・
/// 全候補の gain を負側へ平行移動しない（threat-value-delta の失敗回避）
fn threat_value_stale(
    pos: &Position,
    next: &Position,
    me: Color,
    backed: Option<&[bool; 81]>,
    freshness: f64,
) -> f64 {
    let opp = me.other();
    let by_count = threat_by_count();
    let gain_on = |b: &Position, sq: Coord, role: Role| -> f64 {
        if !b.is_attacked(sq, me) {
            return 0.0;
        }
        let defended = if by_count {
            b.attack_count(sq, opp) >= b.attack_count(sq, me)
        } else {
            b.is_attacked(sq, opp)
        };
        exchange_value(role) * if defended { 0.45 } else { 1.0 }
    };
    let mut best = 0.0f64;
    for (sq, piece) in next.pieces() {
        if piece.color != opp || piece.role == Role::King {
            continue;
        }
        let after = gain_on(next, sq, piece.role);
        if after <= 0.0 {
            continue;
        }
        let is_major = matches!(unpromote_role(piece.role), Role::Rook | Role::Bishop);
        let backed_here = backed.is_some_and(|b| b[crate::belief_features::sq_index(sq)]);
        let eff = if is_major && !backed_here {
            // 相手駒は自分の着手中は動かないので、同じマスの同じ駒が before
            let before = if pos
                .piece_at(sq)
                .is_some_and(|p| p.color == opp && p.role == piece.role)
            {
                gain_on(pos, sq, piece.role)
            } else {
                0.0
            };
            let created = (after - before).max(0.0);
            after - (1.0 - freshness) * created
        } else {
            after
        };
        best = best.max(eff);
    }
    best
}

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

/// **反則枝の回収価値の一般化＋玉センサー**（案2、2026-08-22、quest_0809 36手目の
/// 2三玉（玉プローブ、7点）が発端。ユーザー指摘「歩がすぐ前進してこなくても
/// 2三に玉を置いておけば、王手が掛かった時点で成ってきたと判断できる」）。
///
/// 既存の `foul_probe`（drop_probe_w）は「打ちマスが埋まっていて自分の利きが
/// 当たっている」反則枝だけに情報価値を付ける。同じ経済が玉の手と移動手にもある:
/// - **玉プローブ**（`king_probe_w`）: 玉の行き先 K に利いている敵駒 a が原因で
///   反則 → 「K に利く駒がいる」が確定。a のマスを自分の玉以外の駒が当てて
///   いれば次手で回収できる（価値 = a の交換価値 ＋ a が自駒に当てている脅威の
///   解消 × 0.5）。玉は取られないので、合法枝に mover リスクが無い = 失うもの
///   の無い確かめ方（桂・香で 1三 を確かめると 1四 に歩が残る世界線で只取られ）
/// - **経路プローブ**（`path_probe_w`）: 飛び駒の移動の経路を塞ぐ敵駒 b が原因で
///   反則 → 「経路上に駒がいる」。b のマスを当てていれば回収（32手目 3一角 =
///   経路 2二 のプローブ兼退避、8点）
/// - **玉センサー**（`king_sensor_w`、gain 内・玉の手だけ・差分形）: 相手の
///   **既知の歩**（捕獲で露見したマスと同じ筋でそこより前方 = 二歩則で同じ歩）
///   の成り経路上のマス S が新しい玉位置への王手になり、かつ S を自分の玉以外の
///   駒が当てているなら、成られた瞬間に**王手宣言で観測できて只取りできる**。
///   価値 = P(S に到達) × (駒価値＋脅威解消×0.5)、P は較正値
///   （`bin/pawn_push_probe`: 露見歩の前進 ≈0.3/手、次で成れる歩 ≈0.8）の積。
///   現在の玉位置との差分で加点する（玉を動かさない手は 0）
/// すべて王手中は無効・taint 粒子は無効。反則枝の項は `foul_probe` と同じく
/// combine_score の外側に加え、残り反則予算 (fouls_left/10)² と玉反則の再訪
/// ゲート（`stale_king_foul_dests`）を掛ける。凍結版はこれらの名前を知らない
pub(crate) struct ProbeCtx {
    pub(crate) my_king: Option<Coord>,
    /// 過去手番で玉が反則した行き先（再プローブの価値ゼロ）
    pub(crate) stale_king_dests: HashSet<Coord>,
    /// 相手が自駒を取ったマス（露見歩の同定に使う）
    pub(crate) revealed: [bool; 81],
    pub(crate) king_probe_w: f64,
    pub(crate) path_probe_w: f64,
    pub(crate) king_sensor_w: f64,
    pub(crate) p_push: f64,
    pub(crate) p_promo: f64,
}

fn probe_env(src: &crate::config::EnvSource, name: &str, default: f64) -> f64 {
    src.var(name)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(default)
}

fn king_probe_w() -> f64 {
    crate::config::current(|c| c.strategy.king_probe_w)
}

fn path_probe_w() -> f64 {
    crate::config::current(|c| c.strategy.path_probe_w)
}

fn king_sensor_w() -> f64 {
    crate::config::current(|c| c.strategy.king_sensor_w)
}

fn sensor_p_push() -> f64 {
    crate::config::current(|c| c.strategy.sensor_p_push)
}

fn sensor_p_promo() -> f64 {
    crate::config::current(|c| c.strategy.sensor_p_promo)
}

/// 自分の玉以外の駒が sq へ利いているか（回収できるか）
fn my_nonking_attacks(pos: &Position, sq: Coord, me: Color) -> bool {
    pos.pieces()
        .any(|(from, p)| p.color == me && p.role != Role::King && pos.attacks(from, sq))
}

/// **由来タグによる重み**（`shogi::Anchors`、2026-08-22）: 反則の原因駒が
/// 「観測に裏付けられた駒」なら age に応じた重み、裏付けの無い幻なら 0。
///
/// これが無いと、プローブ系の項は**幻の大駒に最も強く反応する**（実測:
/// quest31-m017 の 6八 へ利く幻の寄与は quest_0809-m036 の本物のと金の7倍。
/// 質量も駒価値も幻の方が大きく、凸ゲートでは分離できなかった）
fn anchor_weight(pos: &Position, at: Coord) -> f64 {
    match pos.anchors().age_at(at) {
        Some(age) => probe_anchor_decay().powi(i32::from(age)),
        None => 0.0,
    }
}

/// 由来タグ（`shogi::Anchors`）を追跡する必要があるか。**消費者がいるときだけ
/// 追跡する**ので、既定（プローブ系がすべて 0）ではアンカーは常に空になり、
/// 粒子の重複除去キーも従来と同一 = 既定挙動が変わらない
pub(crate) fn anchors_needed() -> bool {
    king_probe_w() > 0.0 || path_probe_w() > 0.0 || probe_audit()
}

/// プローブ影の監査モード（`TSUITATE_PROBE_AUDIT=1`）。プローブ値を重み1で
/// 計算して stderr へ出すが**スコアには足さない**ので、方策は
/// 「由来タグ追跡だけオン」の対照のまま。`CandidateScore::probe_unit` の doc
fn probe_audit() -> bool {
    crate::config::current(|c| c.strategy.probe_audit)
}

fn probe_anchor_decay() -> f64 {
    crate::config::current(|c| c.strategy.probe_anchor_decay)
}

/// 原因駒 a（マス at）の回収価値。既定は**交換価値だけ**（codex 指摘 2026-08-22:
/// 「a が当てている自駒の価値 × 0.5」は較正されておらず、その駒を除けば守りの
/// 価値が丸ごと実現する前提になっている）。`TSUITATE_PROBE_THREAT_W` で
/// 脅威解消ぶんを足せる（アブレーション用、既定 0）。
///
/// 反則しても手番は変わらないので相手の応手を挟まずに取りに行ける = 交換価値を
/// 満額で見てよい。ただし `my_nonking_attacks` は幾何的な利きしか見ておらず
/// ピン等で実際には取れない場合があるので、なお楽観側の見積もり
fn recovery_value(pos: &Position, at: Coord, role: Role, me: Color) -> f64 {
    let tw = probe_threat_w();
    if tw <= 0.0 {
        return exchange_value(role);
    }
    let threatened = pos
        .pieces()
        .filter(|(sq, p)| p.color == me && p.role != Role::King && pos.attacks(at, *sq))
        .map(|(_, p)| exchange_value(p.role))
        .fold(0.0, f64::max);
    exchange_value(role) + tw * threatened
}

fn probe_threat_w() -> f64 {
    crate::config::current(|c| c.strategy.probe_threat_w)
}

/// 経路プローブの材料: from→to の経路を最初に塞ぐ敵駒を自分が当てていれば
/// (そのマス, 回収価値)
fn path_probe_material(pos: &Position, from: Coord, to: Coord, me: Color) -> Option<(Coord, f64)> {
    let opp = me.other();
    let df = (to.file - from.file).signum();
    let dr = (to.rank - from.rank).signum();
    let sf = (to.file - from.file).abs();
    let sr = (to.rank - from.rank).abs();
    if sf != 0 && sr != 0 && sf != sr {
        return None;
    }
    let n = sf.max(sr);
    for i in 1..n {
        let sq = Coord {
            file: from.file + df * i,
            rank: from.rank + dr * i,
        };
        if let Some(p) = pos.piece_at(sq) {
            if p.color == opp && my_nonking_attacks(pos, sq, me) {
                let v = anchor_weight(pos, sq) * recovery_value(pos, sq, p.role, me);
                return (v > 0.0).then_some((sq, v));
            }
            return None;
        }
    }
    None
}

/// 相手の歩 sq が「既知」か: 相手が自駒を取ったマスと同じ筋で、そこと同じか
/// それより前方（相手から見て）にいる = 二歩則で同じ歩（estimator の
/// `opp_pawn_intent_factor` と同じ同定）
fn opp_pawn_is_known(sq: Coord, opp: Color, revealed: &[bool; 81]) -> bool {
    (1..=9).any(|r| {
        let s = Coord { file: sq.file, rank: r };
        revealed[crate::belief_features::sq_index(s)]
            && match opp {
                Color::Sente => sq.rank <= r,
                Color::Gote => sq.rank >= r,
            }
    })
}

fn in_zone_for(sq: Coord, c: Color) -> bool {
    match c {
        Color::Sente => sq.rank <= 3,
        Color::Gote => sq.rank >= 7,
    }
}

/// 玉センサーの価値（玉が k にいるとき）: 既知の相手の歩ごとに、成り経路上の
/// マス S で「S の歩/と金が k へ王手 かつ 自分が S を玉以外で当てている」の
/// うち P(到達)×回収価値が最大のものを足す
fn king_sensor_value(pos: &Position, k: Coord, me: Color, ctx: &ProbeCtx) -> f64 {
    let opp = me.other();
    let dr: i8 = match opp {
        Color::Sente => -1,
        Color::Gote => 1,
    };
    let mut total = 0.0;
    for (sq, p) in pos.pieces() {
        if p.color != opp || p.role != Role::Pawn || !opp_pawn_is_known(sq, opp, &ctx.revealed) {
            continue;
        }
        let mut best = 0.0f64;
        let mut prob = 1.0;
        let mut cur = sq;
        for _ in 0..4 {
            let s = Coord {
                file: cur.file,
                rank: cur.rank + dr,
            };
            if !(1..=9).contains(&s.rank) {
                break;
            }
            // 自駒の上へは「取って」到達、相手駒なら止まる
            let blocked_by_opp = pos.piece_at(s).is_some_and(|q| q.color == opp);
            if blocked_by_opp {
                break;
            }
            let promotes = in_zone_for(s, opp);
            prob *= if promotes { ctx.p_promo } else { ctx.p_push };
            // S に着いた駒が k へ利くか（成れば と金 = 金の利き）
            let role = if promotes { Role::Tokin } else { Role::Pawn };
            let mut probe = pos.clone();
            probe.set(sq, None);
            probe.set(s, Some(crate::shogi::Piece { color: opp, role }));
            if probe.attacks(s, k) && my_nonking_attacks(&probe, s, me) {
                best = best.max(prob * recovery_value(&probe, s, role, me));
            }
            if pos.piece_at(s).is_some() {
                break; // 自駒を取ったところで経路は止まる
            }
            cur = s;
        }
        total += best;
    }
    total
}

/// 打ちプローブの再プローブ判定を「既に打って反則したマスか」で行うか
/// （`TSUITATE_DROP_PROBE_REPEAT_GATE=1`、既定 off = 従来の `(1−p_occ)`）。
/// `foul_probe` の doc 参照
fn drop_probe_repeat_gate() -> bool {
    crate::config::current(|c| c.strategy.drop_probe_repeat_gate)
}

/// 2手読みへ予約する「争点への利き足し」の本数（`TSUITATE_DEPTH2_FOCAL_K`、
/// 既定 0 = 従来挙動）。`adds_focal_attacker` の doc 参照
fn depth2_focal_k() -> usize {
    crate::config::current(|c| c.strategy.depth2_focal_k)
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
    crate::config::current(|c| c.strategy.threat_by_count)
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
    use std::collections::{HashMap, HashSet};

    use super::*;

    /// **ノブの有無で乱数条件が変わってはいけない**（PR #22 レビュー指摘1）。
    ///
    /// config 付きの生成が seed を `Option` で受けないと、呼び出し側が
    /// `seed.unwrap_or(0)` に落として「候補だけ全対局が同じシード」になり、
    /// 対照との比較がノブ以外の理由で崩れる。
    #[test]
    fn config付き生成はseedの扱いを変えない() {
        let base = std::sync::Arc::new(crate::config::StrategyConfig::defaults());
        let knobbed = std::sync::Arc::new(crate::config::StrategyConfig::from_source(
            crate::config::EnvSource::from_pairs([("TSUITATE_HAND_ASSET_W", "0.5")]),
        ));
        assert_ne!(base.fingerprint(), knobbed.fingerprint(), "前提: ノブは効いている");

        for cfg in [&base, &knobbed] {
            // seed 未指定はエントロピー由来のまま（Some(0) へ落ちない）
            assert_eq!(
                estimator_with_config("estimator", None, cfg.clone())
                    .expect("estimator")
                    .seed(),
                None
            );
            // 指定したときはその値がそのまま入る
            assert_eq!(
                estimator_with_config("estimator", Some(7), cfg.clone())
                    .expect("estimator")
                    .seed(),
                Some(7)
            );
        }
        // 公開 API 同士の関係（make_seeded_with_config は Some(seed) 版）
        assert!(make_with_config("estimator", None, base.clone()).is_some());
        assert!(make_seeded_with_config("estimator", 7, base.clone()).is_some());
        // 凍結版へは渡せない（黙って無視されるのを防ぐ）
        assert!(make_with_config("estimator_v14", None, base).is_none());
    }
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
    fn king_repeat_legal_factor_は汚名マスの玉手だけを割り引く() {
        let king = Coord { file: 5, rank: 9 };
        let dest = Coord { file: 5, rank: 8 };
        let other = Coord { file: 4, rank: 9 };
        let stale = HashSet::from([dest]);
        let king_to_stale = ShogiMove::Board {
            from: king,
            to: dest,
            promote: false,
        };
        let king_elsewhere = ShogiMove::Board {
            from: king,
            to: other,
            promote: false,
        };
        let pawn = ShogiMove::Board {
            from: Coord { file: 7, rank: 7 },
            to: dest,
            promote: false,
        };
        let drop = ShogiMove::Drop {
            role: Role::Gold,
            to: dest,
        };
        assert!(
            (king_repeat_legal_factor(&king_to_stale, Some(king), &stale, 0.8) - 0.2).abs() < 1e-12
        );
        assert_eq!(
            king_repeat_legal_factor(&king_elsewhere, Some(king), &stale, 0.8),
            1.0
        );
        assert_eq!(
            king_repeat_legal_factor(&pawn, Some(king), &stale, 0.8),
            1.0
        );
        assert_eq!(
            king_repeat_legal_factor(&drop, Some(king), &stale, 0.8),
            1.0
        );
        assert_eq!(
            king_repeat_legal_factor(&king_to_stale, Some(king), &stale, 0.0),
            1.0
        );
        assert_eq!(
            king_repeat_legal_factor(&king_to_stale, Some(king), &stale, 1.5),
            0.0
        );
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
        let pieces = vec![
            vp(5, 2, Role::Dragon),
            vp(5, 1, Role::Gold),
            vp(5, 9, Role::King),
        ];
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
        let near = promo_potential(&[vp(5, 4, Role::Pawn)], Color::Sente, None, None);
        let far = promo_potential(&[vp(5, 7, Role::Pawn)], Color::Sente, None, None);
        assert!(near > far, "near={near} far={far}");
        assert!(far > 0.0);
        // 歩→と金は Δ利き5（中央）× 0.5^1
        assert!((near - 5.0 * 0.5).abs() < 1e-9, "near={near}");
        // 前に自分の歩がいる歩は成れない（道を塞がれた駒はポテンシャル 0）
        let blocked = promo_potential(
            &[vp(5, 4, Role::Pawn), vp(5, 3, Role::Pawn)],
            Color::Sente,
            None,
            None,
        );
        let solo_5c = promo_potential(&[vp(5, 3, Role::Pawn)], Color::Sente, None, None);
        assert!(
            (blocked - solo_5c).abs() < 1e-9,
            "5四の歩の寄与が 0 になるはず"
        );
        // 後手向き: 6段目の歩（成りマス7段目まで1手）も同じ値
        let gote = promo_potential(&[vp(5, 6, Role::Pawn)], Color::Gote, None, None);
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
            None,
            None,
        );
        let without = promo_potential(&[vp(1, 1, Role::Tokin)], Color::Sente, None, None);
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
            None,
        );
        let deep = piece_promo_potential(
            &ctx.occupied,
            Coord { file: 5, rank: 8 },
            Role::Pawn,
            Color::Sente,
            None,
        );
        assert!((h - near).abs() < 1e-9, "敵陣近くの打ちは不足分 0");
        assert!(
            h - deep > 1.0,
            "自陣深くの打ちは不足分が大きい: h={h} deep={deep}"
        );
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
        let view = minimal_view(vec![vp(1, 1, Role::Tokin), vp(5, 9, Role::King)], hand);
        let ctx = hand_option_context(&view);
        let h = ctx.best[&Role::Lance];
        assert!(h > 0.0);
        let blocked = piece_promo_potential(
            &ctx.occupied,
            Coord { file: 1, rank: 2 },
            Role::Lance,
            Color::Sente,
            None,
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
        let before = promo_potential(&[vp(9, 3, Role::Lance)], Color::Sente, None, None);
        let after = promo_potential(&[vp(9, 3, Role::Promotedlance)], Color::Sente, None, None);
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
        let adv_pot = own_effects_after(&view, &advance, None, None, &params).promo_potential;
        let idle_pot = own_effects_after(&view, &idle, None, None, &params).promo_potential;
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
        let deep_pot = own_effects_after(&view, &deep, None, None, &params).promo_potential;
        let shallow_pot = own_effects_after(&view, &shallow, None, None, &params).promo_potential;
        assert!(
            deep_pot > shallow_pot,
            "deep={deep_pot} shallow={shallow_pot}"
        );
        // w=0 なら計算ごとスキップ（切り戻しノブの担保）
        let params_off = EvalParams {
            promo_potential_w: 0.0,
            ..EvalParams::default()
        };
        let off = own_effects_after(&view, &advance, None, None, &params_off);
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
            vec![
                vp(2, 2, Role::Dragon),
                vp(2, 7, Role::Pawn),
                vp(5, 9, Role::King),
            ],
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
        assert_eq!(
            own_effects_after(&view, &evac, None, None, &params).drop_hit_exposure,
            0.0
        );
        // 敵陣を出るだけ（2二→2六、頭2五は空き）では露出は消えない
        let shallow = ShogiMove::Board {
            from: Coord { file: 2, rank: 2 },
            to: Coord { file: 2, rank: 6 },
            promote: false,
        };
        assert!(
            (own_effects_after(&view, &shallow, None, None, &params).drop_hit_exposure
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
            (own_effects_after(&view, &idle, None, None, &params).drop_hit_exposure
                - exchange_value(Role::Dragon))
            .abs()
                < 1e-9
        );
        // 頭（2一）へ持ち駒を打って塞ぐと露出が消える
        let block = ShogiMove::Drop {
            role: Role::Gold,
            to: Coord { file: 2, rank: 1 },
        };
        assert_eq!(
            own_effects_after(&view, &block, None, None, &params).drop_hit_exposure,
            0.0
        );
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

    /// 残り反則 1/2/3 回の床。材料スケールのコストは押し上げ、詰みスケールは通す。
    #[test]
    fn foul_budget_floors_raise_material_cost_not_mate() {
        if std::env::var("TSUITATE_LAST_FOUL_GUARD").is_err()
            && std::env::var("TSUITATE_LAST_FOUL_GUARD_2").is_err()
            && std::env::var("TSUITATE_LAST_FOUL_GUARD_3").is_err()
        {
            assert!((apply_foul_budget_floors(1.0, 5.0) - LAST_FOUL_GUARD).abs() < 1e-12);
            // 2・3回の床は既定 0（材料コストはそのまま）
            assert!((apply_foul_budget_floors(2.0, 5.0) - 5.0).abs() < 1e-12);
            assert!((apply_foul_budget_floors(3.0, 3.0) - 3.0).abs() < 1e-12);
            assert!((apply_foul_budget_floors(4.0, 2.0) - 2.0).abs() < 1e-12);
            assert!((apply_foul_budget_floors(1.0, 1000.0) - 1000.0).abs() < 1e-12);
            assert_eq!(LAST_FOUL_GUARD_2, 0.0);
            assert_eq!(LAST_FOUL_GUARD_3, 0.0);
        }
        assert!((EvalParams::default().taint_occ_legal_w - 0.0).abs() < 1e-12);
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
        assert_field_index!(anchor_move_w);
        assert_field_index!(king_capture_reveal);
        assert_field_index!(promo_king_prox);
        assert_field_index!(foul_occ_attack_w);
        assert_field_index!(material_degen_q0);
        assert_field_index!(own_zone_capture_w);

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
        assert_eq!(
            own[crate::belief_features::sq_index(Coord { file: 4, rank: 7 })],
            1
        );

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

    /// 相手駒の占有証拠: 相手が取ったマスは常に、この手番の非歩打ち反則も。
    /// 歩打ち反則と自分が取ったマスは対象外（material_degen の裏付けゲート）
    #[test]
    fn opp_occupancy_evidence_marks_capture_and_nonpawn_foul_only() {
        let mut view = minimal_view(
            vec![VisiblePiece {
                square: "5i".into(),
                role: Role::King,
            }],
            HashMap::from([(Role::Silver, 1), (Role::Pawn, 1)]),
        );
        view.move_number = 10;
        let mut log = ObservationLog::default();
        log.record(Observation::OpponentMoved {
            move_number: 2,
            captured_my_piece_at: Some("7g".into()),
        });
        // 自分が取ったマスは裏付けにならない
        log.record(Observation::MyMove {
            move_number: 5,
            usi: "2b2a+".into(),
            captured: Some(Role::Silver),
        });
        // この手番の銀打ち反則 → 裏付け
        log.record(Observation::MyFoul {
            move_number: 10,
            usi: "S*4g".into(),
        });
        // 歩打ち反則は二歩の可能性があるので対象外
        log.record(Observation::MyFoul {
            move_number: 10,
            usi: "P*5e".into(),
        });
        // 前の手番の打ち反則は対象外（盤が動いている）
        log.record(Observation::MyFoul {
            move_number: 8,
            usi: "S*3c".into(),
        });

        let backed = opp_occupancy_evidence(&view, &log);
        assert!(backed[crate::belief_features::sq_index(Coord { file: 7, rank: 7 })]);
        assert!(backed[crate::belief_features::sq_index(Coord { file: 4, rank: 7 })]);
        assert!(!backed[crate::belief_features::sq_index(Coord { file: 2, rank: 1 })]);
        assert!(!backed[crate::belief_features::sq_index(Coord { file: 5, rank: 5 })]);
        assert!(!backed[crate::belief_features::sq_index(Coord { file: 3, rank: 3 })]);
    }

    /// 玉の既知脅威接近: m099 型（7g→6h は 5f へ dist=2 のまま筋だけ寄る → 0.5、
    /// 7g→8h は遠ざかる → 0）。脅威マスへの捕獲は 0
    #[test]
    fn king_known_approach_partial_file_close() {
        let mut backed = [false; 81];
        backed[crate::belief_features::sq_index(Coord { file: 5, rank: 6 })] = true;
        let from = Coord { file: 7, rank: 7 };
        let toward = Coord { file: 6, rank: 8 };
        let away = Coord { file: 8, rank: 8 };
        let capture = Coord { file: 5, rank: 6 };
        assert!((king_known_approach_amount(from, toward, &backed) - 0.5).abs() < 1e-9);
        assert_eq!(king_known_approach_amount(from, away, &backed), 0.0);
        assert_eq!(king_known_approach_amount(from, capture, &backed), 0.0);
        // チェビシェフが縮むときは整数差分
        let closer = Coord { file: 6, rank: 6 };
        assert!((king_known_approach_amount(from, closer, &backed) - 1.0).abs() < 1e-9);
    }

    /// m099 決定点で歴代 S*5f が脅威に残っていること
    #[test]
    fn king_threat_evidence_quest31_m099_has_5f() {
        let text = std::fs::read_to_string("scenarios/quest31-m099.kif").expect("kif");
        let kifu = crate::kifu::parse_kif(&text).expect("parse");
        let rep = crate::scenario_core::replay(&kifu, 98);
        let side = rep.pos.turn();
        let log = &rep.logs[crate::scenario_core::side_idx(side)];
        let threats = king_threat_evidence(log);
        let sq5f = crate::belief_features::sq_index(Coord { file: 5, rank: 6 });
        assert!(
            threats[sq5f],
            "5f should be a king threat from historical S*5f foul"
        );
        let from = Coord { file: 7, rank: 7 };
        let toward = Coord { file: 6, rank: 8 };
        let away = Coord { file: 8, rank: 8 };
        let a6 = king_known_approach_amount(from, toward, &threats);
        let a8 = king_known_approach_amount(from, away, &threats);
        assert!(
            a6 > a8,
            "7g6h should approach threats more than 7g8h: 6h={a6} 8h={a8}"
        );
    }

    /// 玉脅威: 過去手番の非歩打ち反則も残し、そこで取ったら消える
    #[test]
    fn king_threat_evidence_keeps_historical_nonpawn_foul() {
        let mut log = ObservationLog::default();
        log.record(Observation::MyFoul {
            move_number: 41,
            usi: "S*5f".into(),
        });
        log.record(Observation::OpponentMoved {
            move_number: 42,
            captured_my_piece_at: Some("7g".into()),
        });
        // 歩打ち反則は載せない
        log.record(Observation::MyFoul {
            move_number: 50,
            usi: "P*3c".into(),
        });
        let threats = king_threat_evidence(&log);
        assert!(threats[crate::belief_features::sq_index(Coord { file: 5, rank: 6 })]);
        assert!(threats[crate::belief_features::sq_index(Coord { file: 7, rank: 7 })]);
        assert!(!threats[crate::belief_features::sq_index(Coord { file: 3, rank: 3 })]);

        // 5f で取ったら消える
        log.record(Observation::MyMove {
            move_number: 51,
            usi: "4f5f".into(),
            captured: Some(Role::Silver),
        });
        let threats2 = king_threat_evidence(&log);
        assert!(!threats2[crate::belief_features::sq_index(Coord { file: 5, rank: 6 })]);
        assert!(threats2[crate::belief_features::sq_index(Coord { file: 7, rank: 7 })]);
    }

    /// 持ち駒資産損の「仕事」判定（安い駒の裏付け当たり／鋭い玉近接）
    #[test]
    fn drop_hand_asset_work_backed_or_near_king() {
        let view = minimal_view(
            vec![VisiblePiece {
                square: "5i".into(),
                role: Role::King,
            }],
            HashMap::from([(Role::Gold, 1), (Role::Silver, 1), (Role::Pawn, 1)]),
        );
        let mut backed = [false; 81];
        // 7g に裏付け占有
        backed[crate::belief_features::sq_index(Coord { file: 7, rank: 7 })] = true;
        let mut kings = std::collections::BTreeSet::new();
        kings.insert(Coord { file: 5, rank: 1 });

        // 安い駒（歩）は裏付けへの当たりだけで work（先手歩は 7h→7g）
        assert!(drop_has_hand_asset_work(
            &view,
            Role::Pawn,
            Coord { file: 7, rank: 8 },
            &backed,
            &kings,
        ));
        // 高い駒（金）は玉から遠い裏付けへの当たりだけでは免税しない
        assert!(!drop_has_hand_asset_work(
            &view,
            Role::Gold,
            Coord { file: 7, rank: 6 },
            &backed,
            &kings,
        ));
        // 金を 1b に打つ: 裏付けにも玉（5a）にも遠い → 仕事なし
        assert!(!drop_has_hand_asset_work(
            &view,
            Role::Gold,
            Coord { file: 1, rank: 2 },
            &backed,
            &kings,
        ));
        // 銀を 5c に打つ: 敵陣かつ玉 5a へチェビシェフ2 → work
        assert!(drop_has_hand_asset_work(
            &view,
            Role::Silver,
            Coord { file: 5, rank: 3 },
            &backed,
            &kings,
        ));
        // 高い駒でも、中段の玉近接だけでは免税しない（G*5e 型）
        let mut kings_near = std::collections::BTreeSet::new();
        kings_near.insert(Coord { file: 7, rank: 5 });
        assert!(
            !drop_has_hand_asset_work(
                &view,
                Role::Gold,
                Coord { file: 7, rank: 6 },
                &backed,
                &kings_near,
            ),
            "gold drop on rank 6 is not enemy camp"
        );
        // 敵陣の金打ちは鋭い玉候補の近くなら work
        assert!(drop_has_hand_asset_work(
            &view,
            Role::Gold,
            Coord { file: 5, rank: 3 },
            &backed,
            &kings,
        ));
        // 金を玉の 8 近傍（5b）へ打つのはプローブなので仕事なし
        assert!(
            !drop_has_hand_asset_work(
                &view,
                Role::Gold,
                Coord { file: 5, rank: 2 },
                &backed,
                &kings,
            ),
            "G*5b adjacent to median-file king is not hand-asset work"
        );
        // 候補が全筋に拡散していると敵玉近接は数えない
        for f in 1..=9i8 {
            kings.insert(Coord { file: f, rank: 1 });
        }
        assert!(
            !drop_has_hand_asset_work(
                &view,
                Role::Silver,
                Coord { file: 5, rank: 3 },
                &backed,
                &kings,
            ),
            "unfocused king belief must not exempt far-from-own-king drops"
        );
        // 自玉の玉頭2マス（5g）の金銀は守り打ち。銀の隣接（5h）や斜め（3h）は対象外。
        // 金の隣接（5h）は 8 近傍なので守り（m055 の G*5h）。
        assert!(
            drop_has_hand_asset_work(
                &view,
                Role::Gold,
                Coord { file: 5, rank: 7 },
                &backed,
                &kings,
            ),
            "G*5g on the king's file is defensive work"
        );
        assert!(
            drop_has_hand_asset_work(
                &view,
                Role::Gold,
                Coord { file: 5, rank: 8 },
                &backed,
                &kings,
            ),
            "G*5h adjacent to own king is defensive for gold"
        );
        assert!(
            !drop_has_hand_asset_work(
                &view,
                Role::Silver,
                Coord { file: 5, rank: 8 },
                &backed,
                &kings,
            ),
            "S*5h adjacent to own king is NOT defensive for silver"
        );
        assert!(
            !drop_has_hand_asset_work(
                &view,
                Role::Silver,
                Coord { file: 3, rank: 8 },
                &backed,
                &kings,
            ),
            "S*3h (chebyshev 2, different file) is not defensive work"
        );
        // 安い駒の裏付け当たりは拡散していても work
        assert!(drop_has_hand_asset_work(
            &view,
            Role::Pawn,
            Coord { file: 7, rank: 8 },
            &backed,
            &kings,
        ));
        // 高い駒は拡散時に裏付け当たりでも免税しない（m062 の G*1b 型）
        assert!(!drop_has_hand_asset_work(
            &view,
            Role::Gold,
            Coord { file: 7, rank: 6 },
            &backed,
            &kings,
        ));
    }

    /// m027 の S*3h は自玉の斜め2マスなので仕事なし。S*4b は敵陣の玉近接。
    #[test]
    fn hand_asset_s3h_quest31_m027_should_be_taxed() {
        let text = std::fs::read_to_string("scenarios/quest31-m027.kif").expect("kif");
        let kifu = crate::kifu::parse_kif(&text).expect("parse");
        let rep = crate::scenario_core::replay(&kifu, 26);
        let side = rep.pos.turn();
        let view = crate::scenario_core::make_view(&rep.pos, side, &rep.fouls);
        let log = &rep.logs[crate::scenario_core::side_idx(side)];
        let kings = crate::deduce::opp_king_candidates(side, log);
        let backed = opp_occupancy_evidence(&view, log);
        assert!(
            !drop_has_hand_asset_work(
                &view,
                Role::Silver,
                Coord { file: 3, rank: 8 },
                &backed,
                &kings
            ),
            "S*3h should be taxed at m027 (king_cands={})",
            kings.len()
        );
        // S*4b は敵陣の玉近接なら仕事あり。玉筋が読めない局面では免税しない
        let work_4b = drop_has_hand_asset_work(
            &view,
            Role::Silver,
            Coord { file: 4, rank: 2 },
            &backed,
            &kings,
        );
        if let Some(median) = king_file_median(&kings) {
            if king_files_focused(&kings, median) {
                assert!(
                    work_4b,
                    "S*4b should have enemy-camp king work at m027 (kings={kings:?})"
                );
            }
        }
    }

    /// m087 の G*6c（8点）は仕事、G*7c（1点）は玉筋隣接なので課税。
    /// m145 の G*8c（0点）も玉筋隣接。
    #[test]
    fn hand_asset_gold_adj_median_king_is_not_work() {
        let text = std::fs::read_to_string("scenarios/quest31-m087.kif").expect("kif");
        let kifu = crate::kifu::parse_kif(&text).expect("parse");
        let rep = crate::scenario_core::replay(&kifu, 86);
        let side = rep.pos.turn();
        let view = crate::scenario_core::make_view(&rep.pos, side, &rep.fouls);
        let log = &rep.logs[crate::scenario_core::side_idx(side)];
        let kings = crate::deduce::opp_king_candidates(side, log);
        let backed = opp_occupancy_evidence(&view, log);
        assert!(
            drop_has_hand_asset_work(
                &view,
                Role::Gold,
                Coord { file: 6, rank: 3 },
                &backed,
                &kings
            ),
            "G*6c should keep work at m087 (kings={kings:?})"
        );
        assert!(
            !drop_has_hand_asset_work(
                &view,
                Role::Gold,
                Coord { file: 7, rank: 3 },
                &backed,
                &kings
            ),
            "G*7c should be taxed at m087 (kings={kings:?})"
        );

        let text = std::fs::read_to_string("scenarios/quest31-m145.kif").expect("kif");
        let kifu = crate::kifu::parse_kif(&text).expect("parse");
        let rep = crate::scenario_core::replay(&kifu, 144);
        let side = rep.pos.turn();
        let view = crate::scenario_core::make_view(&rep.pos, side, &rep.fouls);
        let log = &rep.logs[crate::scenario_core::side_idx(side)];
        let kings = crate::deduce::opp_king_candidates(side, log);
        let backed = opp_occupancy_evidence(&view, log);
        assert!(
            !drop_has_hand_asset_work(
                &view,
                Role::Gold,
                Coord { file: 8, rank: 3 },
                &backed,
                &kings
            ),
            "G*8c should be taxed at m145 (kings={kings:?})"
        );
        assert!(
            !drop_has_hand_asset_work(
                &view,
                Role::Gold,
                Coord { file: 9, rank: 2 },
                &backed,
                &kings
            ),
            "G*9b is a king candidate and should be taxed at m145"
        );
        assert!(
            !drop_has_hand_asset_work(
                &view,
                Role::Gold,
                Coord { file: 6, rank: 2 },
                &backed,
                &kings
            ),
            "G*6b is 3 files off the median king file and should be taxed at m145"
        );
    }

    /// m062 の G*1b が広い玉候補近接で work 免除されていないか
    #[test]
    fn hand_asset_g1b_quest31_m062_should_be_taxed() {
        let text = std::fs::read_to_string("scenarios/quest31-m062.kif").expect("kif");
        let kifu = crate::kifu::parse_kif(&text).expect("parse");
        let rep = crate::scenario_core::replay(&kifu, 61);
        let side = rep.pos.turn();
        let view = crate::scenario_core::make_view(&rep.pos, side, &rep.fouls);
        let log = &rep.logs[crate::scenario_core::side_idx(side)];
        let kings = crate::deduce::opp_king_candidates(side, log);
        let backed = opp_occupancy_evidence(&view, log);
        let to = Coord { file: 1, rank: 2 };
        let work = drop_has_hand_asset_work(&view, Role::Gold, to, &backed, &kings);
        assert!(
            !work,
            "G*1b should NOT have hand-asset work at m062 (king_cands={})",
            kings.len()
        );
    }

    /// 大駒成り遠方: 隣接着地かつ接近なら 0。非接近は最低 1。以遠は超過分
    #[test]
    fn promote_far_amount_free_within_one_if_approaching() {
        let mut cands = std::collections::BTreeSet::new();
        cands.insert(Coord { file: 7, rank: 1 });
        let from_far = Coord { file: 2, rank: 4 };
        // 2d→3c: 着地 d=4 → 3、かつ非接近でも max で 3
        assert!(
            (promote_far_amount(from_far, Coord { file: 3, rank: 3 }, &cands) - 3.0).abs() < 1e-9
        );
        // 7c→7b: 着地隣接 d=1 → 0、かつ接近（2→1）なので免税
        assert_eq!(
            promote_far_amount(
                Coord { file: 7, rank: 3 },
                Coord { file: 7, rank: 2 },
                &cands
            ),
            0.0
        );
        // 4a→3b: 玉 7a からは遠ざかる。裾に 3a がいても非接近なら最低 1
        let mut cands2 = cands.clone();
        cands2.insert(Coord { file: 3, rank: 1 });
        let amt = promote_far_amount(
            Coord { file: 4, rank: 1 },
            Coord { file: 3, rank: 2 },
            &cands2,
        );
        assert!(
            amt >= 1.0,
            "non-approaching promote near a fringe cand should still be taxed: {amt}"
        );
        // 空集合は 0
        assert_eq!(
            promote_far_amount(
                Coord { file: 2, rank: 4 },
                Coord { file: 3, rank: 3 },
                &std::collections::BTreeSet::new()
            ),
            0.0
        );
    }

    /// と金の8近傍王手（m095: 7c のと金 → 7b の玉）と、遠い玉への非発火
    #[test]
    fn promote_check_reveal_tokin_adjacent() {
        let own = [false; 81];
        let to = Coord { file: 7, rank: 3 }; // 7c
        assert!(piece_attacks_sq(
            Role::Tokin,
            Color::Sente,
            to,
            Coord { file: 7, rank: 2 }, // 7b
            &own
        ));
        assert!(!piece_attacks_sq(
            Role::Tokin,
            Color::Sente,
            to,
            Coord { file: 5, rank: 1 }, // 5a（遠い）
            &own
        ));
        // 斜め後ろは金相当なので不可（先手なら rank+1 斜め）
        assert!(!piece_attacks_sq(
            Role::Tokin,
            Color::Sente,
            to,
            Coord { file: 6, rank: 4 }, // 6d
            &own
        ));
    }

    /// 不成の双子を作る条件（`nonpromote_check_w`）は「成ると玉候補への
    /// 利きが**増える**」こと。quest31 の 46手目でユーザーが 4九銀不成=10 /
    /// 4九銀成=2 と採点した根拠（成銀が 5九の玉に王手 → 宣言で露見）を固定し、
    /// 同時に隣の 2九銀成が対象外（王手が増えない）であることも確かめる
    #[test]
    fn nonpromote_check_quest31_m046_silver() {
        let text = std::fs::read_to_string("scenarios/quest31-m046.kif").expect("kif");
        let kifu = crate::kifu::parse_kif(&text).expect("parse");
        let rep = crate::scenario_core::replay(&kifu, 45);
        let side = rep.pos.turn();
        let view = crate::scenario_core::make_view(&rep.pos, side, &[0, 0]);
        let log = &rep.logs[crate::scenario_core::side_idx(side)];
        let dist = nonpromote_king_dist(&view, log);
        let occ = crate::strategy::king_threat_evidence(log);
        let from = Coord { file: 3, rank: 8 }; // 3h
        let good = promotion_check_mass(
            &view,
            from,
            Coord { file: 4, rank: 9 },
            Role::Silver,
            &dist,
            &occ,
        );
        let other = promotion_check_mass(
            &view,
            from,
            Coord { file: 2, rank: 9 },
            Role::Silver,
            &dist,
            &occ,
        );
        assert!(
            good > other,
            "3h4i+ (checks the 5i king) must carry more check mass than 3h2i+: {good} vs {other}"
        );
        // 双子の乱造チェック: 成りが選べる手のうち発火するのはごく一部
        let mut optional = 0usize;
        let mut fires = 0usize;
        for piece in &view.your_pieces {
            let Some(f) = parse_usi_square(&piece.square) else {
                continue;
            };
            for to in move_targets(&view.your_pieces, piece, side) {
                if promotion_choice(piece.role, f, to, side) != Promotion::Optional {
                    continue;
                }
                optional += 1;
                if promotion_check_mass(&view, f, to, piece.role, &dist, &occ) >= good {
                    fires += 1;
                }
            }
        }
        assert!(
            fires * 3 <= optional,
            "twin generation should stay narrow: {fires}/{optional} (3h4i+ mass={good})"
        );
    }

    /// 捕獲直後の手戻り免除ノブの既定は定数と一致する（2026-08-21 採用: 0.08）。
    #[test]
    fn capture_retreat_w_default_off() {
        let w = std::env::var("TSUITATE_CAPTURE_RETREAT_W").ok();
        if w.is_none() {
            assert!((capture_retreat_w() - CAPTURE_RETREAT_W).abs() < 1e-12);
            assert_eq!(CAPTURE_RETREAT_W, 0.08);
        }
    }

    /// m133 の 2d3c+ は玉候補から遠く、寄せ筋の接近成りより重い
    #[test]
    fn promote_far_quest31_m133_2d3c_is_far() {
        let text = std::fs::read_to_string("scenarios/quest31-m133.kif").expect("kif");
        let kifu = crate::kifu::parse_kif(&text).expect("parse");
        let rep = crate::scenario_core::replay(&kifu, 132);
        let side = rep.pos.turn();
        let log = &rep.logs[crate::scenario_core::side_idx(side)];
        let kings = crate::deduce::opp_king_candidates(side, log);
        let far = promote_far_amount(
            Coord { file: 2, rank: 4 },
            Coord { file: 3, rank: 3 },
            &kings,
        );
        let near = promote_far_amount(
            Coord { file: 7, rank: 4 },
            Coord { file: 7, rank: 3 },
            &kings,
        );
        assert!(
            far > near && far >= 1.0,
            "2d3c+ landing should be farther than 7c: far={far} near={near} kings={kings:?}"
        );
    }

    /// 7四歩→7三歩は 6二玉候補に筋距離1、1四歩は距離5で加点0。
    /// 中段の 7六歩は敵陣ではないので 0（9六歩・8六歩の乗っ取り対策）。
    #[test]
    fn king_file_pawn_prefers_king_files() {
        let mut cands = std::collections::BTreeSet::new();
        cands.insert(Coord { file: 6, rank: 2 });
        let seven = king_file_pawn_amount(
            Coord { file: 7, rank: 4 },
            Coord { file: 7, rank: 3 },
            Color::Sente,
            &cands,
        );
        let one = king_file_pawn_amount(
            Coord { file: 1, rank: 4 },
            Coord { file: 1, rank: 3 },
            Color::Sente,
            &cands,
        );
        assert!((seven - 0.5).abs() < 1e-9, "d_file=1 → 1/2, got {seven}");
        assert_eq!(one, 0.0, "d_file=5 is outside the ≤2 window");
        // 中段の歩突きは加点しない
        assert_eq!(
            king_file_pawn_amount(
                Coord { file: 7, rank: 7 },
                Coord { file: 7, rank: 6 },
                Color::Sente,
                &cands,
            ),
            0.0
        );
        // 横移動は前進ではない
        assert_eq!(
            king_file_pawn_amount(
                Coord { file: 7, rank: 6 },
                Coord { file: 8, rank: 6 },
                Color::Sente,
                &cands,
            ),
            0.0
        );
        // 敵陣の歩打ち: 7c は筋距離1、2c は距離4
        let drop7 = king_file_pawn_drop_amount(Coord { file: 7, rank: 3 }, &cands);
        let drop2 = king_file_pawn_drop_amount(Coord { file: 2, rank: 3 }, &cands);
        assert!((drop7 - 0.5).abs() < 1e-9);
        assert_eq!(drop2, 0.0);

        // 全盤の広い候補でも中央値は 5筋。9六歩は中段なので 0、7c 打ちは距離2で 1/3。
        let all: std::collections::BTreeSet<Coord> = (1..=9)
            .flat_map(|file| (1..=9).map(move |rank| Coord { file, rank }))
            .collect();
        assert_eq!(
            king_file_pawn_amount(
                Coord { file: 9, rank: 7 },
                Coord { file: 9, rank: 6 },
                Color::Sente,
                &all,
            ),
            0.0,
            "9g9f is not in enemy camp"
        );
        assert!(
            (king_file_pawn_drop_amount(Coord { file: 7, rank: 3 }, &all) - 1.0 / 3.0).abs() < 1e-9
        );
        // 4f4g+ 相当: 自陣への歩前進は 0
        assert_eq!(
            king_file_pawn_amount(
                Coord { file: 4, rank: 6 },
                Coord { file: 4, rank: 7 },
                Color::Sente,
                &cands,
            ),
            0.0
        );
    }

    #[test]
    fn king_file_pawn_mid_boosts_76_after_move_80_when_focused() {
        let mut cands = std::collections::BTreeSet::new();
        cands.insert(Coord { file: 7, rank: 2 });
        cands.insert(Coord { file: 8, rank: 2 });
        cands.insert(Coord { file: 9, rank: 1 });
        let seven = king_file_pawn_mid_amount(
            Coord { file: 7, rank: 7 },
            Coord { file: 7, rank: 6 },
            Color::Sente,
            &cands,
            81,
        );
        let nine = king_file_pawn_mid_amount(
            Coord { file: 9, rank: 7 },
            Coord { file: 9, rank: 6 },
            Color::Sente,
            &cands,
            81,
        );
        let eight = king_file_pawn_mid_amount(
            Coord { file: 8, rank: 7 },
            Coord { file: 8, rank: 6 },
            Color::Sente,
            &cands,
            81,
        );
        assert_eq!(seven, 1.0, "7g7f at ply 81 on side file");
        assert_eq!(nine, 1.0, "9g9f at ply 81 on side file");
        assert_eq!(eight, 0.0, "8g8f is the median file: do not boost");
        assert_eq!(
            king_file_pawn_mid_amount(
                Coord { file: 7, rank: 7 },
                Coord { file: 7, rank: 6 },
                Color::Sente,
                &cands,
                21,
            ),
            0.0,
            "opening 7g7f is not boosted"
        );
        assert_eq!(
            king_file_pawn_mid_amount(
                Coord { file: 7, rank: 7 },
                Coord { file: 7, rank: 6 },
                Color::Sente,
                &cands,
                87,
            ),
            0.0,
            "m087 G*6c must not be displaced by 7g7f"
        );
        // 既に中段にいる 9f9e は対象外
        assert_eq!(
            king_file_pawn_mid_amount(
                Coord { file: 9, rank: 6 },
                Coord { file: 9, rank: 5 },
                Color::Sente,
                &cands,
                81,
            ),
            0.0,
            "9f9e is already mid-board"
        );
        // 敵陣は mid ではなく camp 側
        assert_eq!(
            king_file_pawn_mid_amount(
                Coord { file: 7, rank: 4 },
                Coord { file: 7, rank: 3 },
                Color::Sente,
                &cands,
                81,
            ),
            0.0
        );
        // 玉筋が全盤に拡散していると 0
        let all: std::collections::BTreeSet<Coord> = (1..=9)
            .flat_map(|file| (1..=9).map(move |rank| Coord { file, rank }))
            .collect();
        assert_eq!(
            king_file_pawn_mid_amount(
                Coord { file: 7, rank: 7 },
                Coord { file: 7, rank: 6 },
                Color::Sente,
                &all,
                81,
            ),
            0.0
        );
    }

    #[test]
    fn king_endgame_flee_taxes_empty_king_run_after_125() {
        let mut view = minimal_view(
            vec![VisiblePiece {
                square: "8a".into(),
                role: Role::King,
            }],
            HashMap::new(),
        );
        view.move_number = 140;
        let to = Coord { file: 9, rank: 2 }; // 9b
        assert_eq!(king_endgame_flee_amount(to, &view, None), 1.0);
        view.move_number = 80;
        assert_eq!(king_endgame_flee_amount(to, &view, None), 0.0);
        view.move_number = 140;
        view.you_in_check = true;
        assert_eq!(
            king_endgame_flee_amount(to, &view, None),
            1.0,
            "m140 は王手中の空マス逃げ 8a9b を課税する"
        );
        view.you_in_check = false;
        let mut backed = [false; 81];
        backed[crate::belief_features::sq_index(to)] = true;
        assert_eq!(
            king_endgame_flee_amount(to, &view, Some(&backed)),
            0.0,
            "backed recapture squares are not taxed"
        );
    }

    #[test]
    fn gold_join_king_boosts_8c8b_in_check() {
        let king = Coord { file: 8, rank: 1 }; // 8a
        let from = Coord { file: 8, rank: 3 }; // 8c
        let to = Coord { file: 8, rank: 2 }; // 8b
        assert_eq!(
            gold_join_king_amount(from, to, king, 140, true),
            1.0,
            "m140 の 8c8b は王手中の金が玉へ寄る"
        );
        assert_eq!(
            gold_join_king_amount(from, to, king, 140, false),
            0.0,
            "非王手の寄りは m138 の未採点逃避になるので対象外"
        );
        assert_eq!(
            gold_join_king_amount(from, to, king, 80, true),
            0.0,
            "手数 125 未満は対象外"
        );
        assert_eq!(
            gold_join_king_amount(to, from, king, 140, true),
            0.0,
            "既に隣接している金の移動は 0"
        );
        assert_eq!(
            gold_join_king_amount(from, Coord { file: 7, rank: 3 }, king, 140, true),
            0.0,
            "玉に隣接しない着地は 0"
        );
    }

    #[test]
    fn gold_king_file_boosts_7b8c_not_in_check() {
        let king = Coord { file: 8, rank: 1 }; // 8a
        let from = Coord { file: 7, rank: 2 }; // 7b
        let to = Coord { file: 8, rank: 3 }; // 8c
        assert_eq!(
            gold_king_file_amount(from, to, king, 130, false),
            1.0,
            "m130 の 7b8c は隣接金が玉筋へ動く"
        );
        assert_eq!(
            gold_king_file_amount(from, to, king, 130, true),
            0.0,
            "王手中は gold_join / CheckSolver の領分"
        );
        assert_eq!(
            gold_king_file_amount(from, to, king, 80, false),
            0.0,
            "手数 125 未満は対象外"
        );
        assert_eq!(
            gold_king_file_amount(
                Coord { file: 8, rank: 3 },
                Coord { file: 8, rank: 2 },
                king,
                140,
                false,
            ),
            0.0,
            "既に隣接していない金（8c）は gold_join の領分"
        );
        assert_eq!(
            gold_king_file_amount(from, Coord { file: 7, rank: 3 }, king, 130, false),
            0.0,
            "玉筋以外（7c）は 0"
        );
        assert_eq!(
            gold_king_file_amount(from, Coord { file: 8, rank: 2 }, king, 130, false),
            0.0,
            "距離 1 の 7b8b は未採点逃避なので対象外"
        );
    }

    #[test]
    fn knight_late_promo_taxes_optional_after_100() {
        let gote = Color::Gote;
        let from = Coord { file: 8, rank: 5 }; // 8e
        let to = Coord { file: 7, rank: 7 }; // 7g
        assert_eq!(
            knight_late_promo_amount(Role::Knight, from, to, true, gote, 106),
            1.0,
            "m106 の 8e7g+ は終盤の桂の敵陣進入"
        );
        assert_eq!(
            knight_late_promo_amount(Role::Knight, from, to, false, gote, 100),
            0.0,
            "m100 の 8e7g 不成（5点）は 110 手前なので免税。0.5 倍でも 8c8d へ落ちた"
        );
        assert_eq!(
            knight_late_promo_amount(Role::Knight, from, to, false, gote, 116),
            KNIGHT_LATE_NONPROMO_SCALE,
            "m116 の 8e7g 不成（2点）は 110 手以降の弱い税"
        );
        assert_eq!(
            knight_late_promo_amount(Role::Knight, from, to, true, gote, 40),
            0.0,
            "40手前後の 4d3b+ クラスは手数ゲートで守る"
        );
        assert_eq!(
            knight_late_promo_amount(Role::Knight, from, to, true, gote, 138),
            0.0,
            "m138 の 8e7g+（4点）は手数上限で免税"
        );
        assert_eq!(
            knight_late_promo_amount(Role::Silver, from, to, true, gote, 106),
            0.0,
            "銀成りは own_camp_minor の領分"
        );
        assert_eq!(
            knight_late_promo_amount(
                Role::Knight,
                Coord { file: 8, rank: 9 },
                Coord { file: 7, rank: 7 },
                false,
                Color::Sente,
                137,
            ),
            0.0,
            "先手の 8i7g は自陣側なので課税しない（手数上限でも 0）"
        );
    }

    #[test]
    fn knight_endgame_promo_boosts_8e7g_after_137() {
        let gote = Color::Gote;
        let to = Coord { file: 7, rank: 7 }; // 7g
        assert_eq!(
            knight_endgame_promo_amount(Role::Knight, to, true, gote, 138),
            1.0,
            "m138 の 8e7g+（4点）は終盤の桂成り加点"
        );
        assert_eq!(
            knight_endgame_promo_amount(Role::Knight, to, false, gote, 138),
            0.0,
            "不成は加点しない"
        );
        assert_eq!(
            knight_endgame_promo_amount(Role::Knight, to, true, gote, 100),
            0.0,
            "手数 136 までは課税側"
        );
        assert_eq!(
            knight_endgame_promo_amount(Role::Silver, to, true, gote, 138),
            0.0,
            "銀は対象外"
        );
    }

    #[test]
    fn knight_camp_exit_boosts_6b7d_after_120() {
        let gote = Color::Gote;
        let from = Coord { file: 6, rank: 2 }; // 6b
        let to = Coord { file: 7, rank: 4 }; // 7d
        assert_eq!(
            knight_camp_exit_amount(Role::Knight, from, to, gote, 124),
            1.0,
            "m124 の 6b7d は自陣から中段へ出て 5筋から遠ざかる"
        );
        assert_eq!(
            knight_camp_exit_amount(Role::Knight, from, Coord { file: 5, rank: 4 }, gote, 124),
            0.0,
            "6b5d（0点）は初期玉筋へ近づく"
        );
        assert_eq!(
            knight_camp_exit_amount(Role::Knight, from, to, gote, 100),
            0.0,
            "手数 120 未満は対象外"
        );
        assert_eq!(
            knight_camp_exit_amount(
                Role::Knight,
                Coord { file: 8, rank: 5 },
                Coord { file: 7, rank: 7 },
                gote,
                124
            ),
            0.0,
            "敵陣進入は knight_late_promo の領分"
        );
        assert_eq!(
            knight_camp_exit_amount(
                Role::Knight,
                Coord { file: 8, rank: 9 },
                Coord { file: 7, rank: 7 },
                Color::Sente,
                137
            ),
            0.0,
            "先手の 8i7g は自陣に留まる"
        );
    }

    #[test]
    fn silver_camp_exit_boosts_own_camp_advance_after_100() {
        let gote = Color::Gote;
        let from = Coord { file: 7, rank: 3 }; // 7c、後手の自陣
        let to = Coord { file: 6, rank: 4 }; // 6d、中段
        assert_eq!(
            silver_camp_exit_amount(Role::Silver, from, to, gote, 106),
            1.0,
            "m106 の 7c6d は自陣から出る銀"
        );
        assert_eq!(
            silver_camp_exit_amount(Role::Silver, from, to, gote, 80),
            0.0,
            "手数 100 未満は対象外"
        );
        assert_eq!(
            silver_camp_exit_amount(Role::Gold, from, to, gote, 106),
            0.0,
            "金は対象外"
        );
        assert_eq!(
            silver_camp_exit_amount(
                Role::Silver,
                Coord { file: 7, rank: 8 },
                Coord { file: 6, rank: 7 },
                gote,
                106
            ),
            0.0,
            "敵陣の銀は自陣脱出ではない"
        );
        assert_eq!(
            silver_camp_exit_amount(Role::Silver, from, Coord { file: 7, rank: 2 }, gote, 106),
            0.0,
            "自陣内の移動は加点しない"
        );
    }

    #[test]
    fn king_known_approach_w_default_off() {
        let w = std::env::var("TSUITATE_KING_KNOWN_APPROACH_W").ok();
        if w.is_none() {
            assert!((king_known_approach_w() - KING_KNOWN_APPROACH_W).abs() < 1e-12);
            assert_eq!(KING_KNOWN_APPROACH_W, 0.0);
        }
    }

    #[test]
    fn hand_asset_w_default_off() {
        let w = std::env::var("TSUITATE_HAND_ASSET_W").ok();
        if w.is_none() {
            assert!((hand_asset_w() - HAND_ASSET_W).abs() < 1e-12);
            assert_eq!(HAND_ASSET_W, 0.0);
        }
    }

    #[test]
    fn king_adj_heavy_w_default_off() {
        let w = std::env::var("TSUITATE_KING_ADJ_HEAVY_W").ok();
        if w.is_none() {
            assert!((king_adj_heavy_w() - KING_ADJ_HEAVY_W).abs() < 1e-12);
            assert_eq!(KING_ADJ_HEAVY_W, 0.0);
        }
    }

    #[test]
    fn own_camp_minor_promo_w_default_off() {
        let w = std::env::var("TSUITATE_OWN_CAMP_MINOR_PROMO_W").ok();
        if w.is_none() {
            assert!((own_camp_minor_promo_w() - OWN_CAMP_MINOR_PROMO_W).abs() < 1e-12);
            assert_eq!(OWN_CAMP_MINOR_PROMO_W, 0.0);
        }
    }

    #[test]
    fn promote_check_reveal_w_default_off() {
        let w = std::env::var("TSUITATE_PROMOTE_CHECK_REVEAL_W").ok();
        if w.is_none() {
            assert!((promote_check_reveal_w() - PROMOTE_CHECK_REVEAL_W).abs() < 1e-12);
            assert_eq!(PROMOTE_CHECK_REVEAL_W, 0.0);
        }
    }

    /// GEN は既定 off のまま。PREROLE は 2026-08-21 に既定 on で採用。
    #[test]
    fn gen_nonpromote_and_prerole_default_off() {
        if std::env::var("TSUITATE_GEN_NONPROMOTE").is_err() {
            assert!(!gen_nonpromote());
        }
        if std::env::var("TSUITATE_PROMO_RISK_PREROLE").is_err() {
            assert!(promo_risk_prerole());
        }
    }

    /// 角の 3三 は敵陣・裏付け無しで課税、と金と歩は免税
    #[test]
    fn unbacked_camp_taxes_expensive_unbacked_entry() {
        let me = Color::Sente;
        let mut backed = [false; 81];
        let bishop_3c =
            unbacked_camp_amount(Role::Bishop, Coord { file: 3, rank: 3 }, me, &backed, false);
        assert!(
            (bishop_3c - exchange_value(Role::Bishop)).abs() < 1e-9,
            "bishop into 3c: {bishop_3c}"
        );
        let tokin_4a =
            unbacked_camp_amount(Role::Tokin, Coord { file: 4, rank: 1 }, me, &backed, false);
        assert_eq!(tokin_4a, 0.0, "tokin working in camp is not taxed");
        let pawn_7c =
            unbacked_camp_amount(Role::Pawn, Coord { file: 7, rank: 3 }, me, &backed, false);
        assert_eq!(pawn_7c, 0.0, "pawns are cheap probes");
        let bishop_4g =
            unbacked_camp_amount(Role::Bishop, Coord { file: 4, rank: 7 }, me, &backed, false);
        assert_eq!(bishop_4g, 0.0, "rank 7 is not sente's enemy camp");
        let gold_board =
            unbacked_camp_amount(Role::Gold, Coord { file: 6, rank: 2 }, me, &backed, false);
        assert!(
            (gold_board - exchange_value(Role::Gold)).abs() < 1e-9,
            "gold board move into enemy camp (6c6b) is taxed"
        );
        let gold_drop =
            unbacked_camp_amount(Role::Gold, Coord { file: 7, rank: 3 }, me, &backed, true);
        assert_eq!(gold_drop, 0.0, "gold drops are HAND_ASSET's job");
        backed[crate::belief_features::sq_index(Coord { file: 3, rank: 3 })] = true;
        let recap =
            unbacked_camp_amount(Role::Bishop, Coord { file: 3, rank: 3 }, me, &backed, false);
        assert_eq!(recap, 0.0, "backed recapture is exempt");
    }

    #[test]
    fn unbacked_camp_gold_needs_believed_capture() {
        assert!(
            unbacked_camp_needs_capture(Role::Gold, 5.3),
            "6c6b believed gold capture is taxed"
        );
        assert!(
            !unbacked_camp_needs_capture(Role::Silver, 0.0),
            "3h4i with capture=0 must not be taxed"
        );
        assert!(
            unbacked_camp_needs_capture(Role::Bishop, 0.0),
            "majors are taxed even without capture (2d3c+)"
        );
    }

    #[test]
    fn home_gold_attack_hits_opp_gold_home_not_tokin() {
        let backed = [false; 81];
        let gote_4i = home_gold_attack_amount(
            Role::Silver,
            Coord { file: 4, rank: 9 },
            Color::Gote,
            &backed,
            false,
        );
        assert_eq!(gote_4i, 1.0, "3h4i is sente's gold home");
        let sente_4a = home_gold_attack_amount(
            Role::Gold,
            Coord { file: 4, rank: 1 },
            Color::Sente,
            &backed,
            false,
        );
        assert_eq!(sente_4a, 1.0, "gold to 4a is gote's gold home");
        let promo = home_gold_attack_amount(
            Role::Silver,
            Coord { file: 4, rank: 9 },
            Color::Gote,
            &backed,
            true,
        );
        assert_eq!(promo, 0.0, "3h4i+ must not get the gold-home bonus");
        let tokin = home_gold_attack_amount(
            Role::Tokin,
            Coord { file: 4, rank: 1 },
            Color::Sente,
            &backed,
            false,
        );
        assert_eq!(tokin, 0.0, "tokin 3a4a must not get the gold-home bonus");
        let miss = home_gold_attack_amount(
            Role::Silver,
            Coord { file: 3, rank: 9 },
            Color::Gote,
            &backed,
            false,
        );
        assert_eq!(miss, 0.0, "3i is silver home, not gold");
        let mut hit = [false; 81];
        hit[crate::belief_features::sq_index(Coord { file: 4, rank: 9 })] = true;
        assert_eq!(
            home_gold_attack_amount(
                Role::Silver,
                Coord { file: 4, rank: 9 },
                Color::Gote,
                &hit,
                false,
            ),
            0.0,
            "backed recapture is already counted as capture"
        );
    }

    #[test]
    fn home_gold_file_sibling_blocks_same_file_not_other_file() {
        let me = Color::Gote;
        let four_g = Coord { file: 4, rank: 7 };
        let two_i = Coord { file: 2, rank: 9 };
        let four_i = Coord { file: 4, rank: 9 };
        assert!(
            home_gold_file_sibling(four_g, me),
            "m054 3h4g is on gold-home file 4"
        );
        assert!(
            !home_gold_file_sibling(two_i, me),
            "phantom 3h2i is file 2, must not block 3h4i"
        );
        assert!(
            !home_gold_file_sibling(four_i, me),
            "gold-home itself is the boosted move, not a sibling"
        );
        let sente = Color::Sente;
        assert!(
            home_gold_file_sibling(Coord { file: 4, rank: 2 }, sente),
            "4b shares file with gote gold home 4a"
        );
        assert!(
            !home_gold_file_sibling(Coord { file: 5, rank: 1 }, sente),
            "5a is not a gold-home file"
        );
    }

    #[test]
    fn tokin_approach_rewards_closing_king_file() {
        let mut cands = std::collections::BTreeSet::new();
        cands.insert(Coord { file: 5, rank: 1 });
        let close = tokin_file_approach_amount(
            Coord { file: 2, rank: 3 },
            Coord { file: 3, rank: 2 },
            Color::Sente,
            &cands,
        );
        assert!(
            (close - 1.0 / 3.0).abs() < 1e-9,
            "2c3b d 3→2 → 1/3, got {close}"
        );
        let recede = tokin_file_approach_amount(
            Coord { file: 2, rank: 3 },
            Coord { file: 1, rank: 3 },
            Color::Sente,
            &cands,
        );
        assert_eq!(recede, 0.0, "2c1c recedes from file 5");
        let adj = tokin_file_approach_amount(
            Coord { file: 3, rank: 1 },
            Coord { file: 4, rank: 1 },
            Color::Sente,
            &cands,
        );
        assert!(
            (adj - 0.5).abs() < 1e-9,
            "3a4a also approaches (king_adj tax is larger)"
        );
    }

    #[test]
    fn unbacked_gs_capture_and_home_gold_defaults() {
        if std::env::var("TSUITATE_UNBACKED_GS_CAPTURE_W").is_err() {
            assert!((unbacked_gs_capture_w() - UNBACKED_GS_CAPTURE_W).abs() < 1e-12);
            assert_eq!(UNBACKED_GS_CAPTURE_W, 0.0);
        }
        if std::env::var("TSUITATE_UNBACKED_CAMP_W").is_err() {
            assert!((unbacked_camp_w() - UNBACKED_CAMP_W).abs() < 1e-12);
            assert_eq!(UNBACKED_CAMP_W, 0.0);
        }
        if std::env::var("TSUITATE_HOME_GOLD_ATTACK_W").is_err() {
            assert!((home_gold_attack_w() - HOME_GOLD_ATTACK_W).abs() < 1e-12);
            assert_eq!(HOME_GOLD_ATTACK_W, 0.0);
        }
        if std::env::var("TSUITATE_TOKIN_APPROACH_W").is_err() {
            assert!((tokin_approach_w() - TOKIN_APPROACH_W).abs() < 1e-12);
            assert_eq!(TOKIN_APPROACH_W, 0.0);
        }
        if std::env::var("TSUITATE_PROMOTE_FAR_W").is_err() {
            assert!((promote_far_w() - PROMOTE_FAR_W).abs() < 1e-12);
            assert_eq!(PROMOTE_FAR_W, 0.0);
        }
        if std::env::var("TSUITATE_BELIEF_OCC_CAP_W").is_err() {
            assert!((belief_occ_cap_w() - BELIEF_OCC_CAP_W).abs() < 1e-12);
            assert_eq!(BELIEF_OCC_CAP_W, 0.0);
        }
        if std::env::var("TSUITATE_CHECK_SAFE_RESOLVE").is_err() {
            assert!(!check_safe_resolve_enabled());
        }
        if std::env::var("TSUITATE_KING_CAND_ATTACK_W").is_err() {
            assert!((king_cand_attack_w() - KING_CAND_ATTACK_W).abs() < 1e-12);
            assert_eq!(KING_CAND_ATTACK_W, 4.0);
        }
        if std::env::var("TSUITATE_KING_CAND_CHECK_W").is_err() {
            assert!((king_cand_check_w() - KING_CAND_CHECK_W).abs() < 1e-12);
            assert_eq!(KING_CAND_CHECK_W, 1.0);
        }
        if std::env::var("TSUITATE_LANDING_SUPPORT_W").is_err() {
            assert!((landing_support_w() - LANDING_SUPPORT_W).abs() < 1e-12);
            assert_eq!(LANDING_SUPPORT_W, 0.7);
        }
        if std::env::var("TSUITATE_KING_BELIEF_PROX_W").is_err() {
            assert!((king_belief_prox_w() - KING_BELIEF_PROX_W).abs() < 1e-12);
            assert_eq!(KING_BELIEF_PROX_W, 0.0);
        }
        if std::env::var("TSUITATE_KING_PROX_EXCLUDE_SELF").is_err() {
            assert!(king_prox_exclude_self());
        }
    }

    /// 接近ボーナスは玉候補の隣の安い駒を、遠い大駒成り込みより高くする。
    /// 着地マス自身（玉候補そのものへの打ち）は最大加点にしない。
    #[test]
    fn king_cand_prox_prefers_adjacent_cheap_over_far_bishop() {
        let mut cands = std::collections::BTreeSet::new();
        cands.insert(Coord { file: 8, rank: 2 }); // 8b
        cands.insert(Coord { file: 8, rank: 1 });
        cands.insert(Coord { file: 7, rank: 1 });
        let map = king_cand_prox_map(&cands);
        let idx = |file: i8, rank: i8| ((file - 1) * 9 + (rank - 1)) as usize;
        let adj = map[idx(7, 2)]; // 7b、玉候補の隣
        let far = map[idx(3, 3)]; // 3c、3三角成の着地
        let on_king = map[idx(8, 2)]; // 8b そのもの（着手できた＝玉は居ない）
        assert!(adj > far * 4.0, "7b prox {adj} should dwarf 3c prox {far}");
        assert!(
            adj > on_king,
            "landing on a king-candidate square must not be the peak: 7b={adj} 8b={on_king}"
        );
        let pawn = KING_CAND_ATTACK_W * adj * (2.0 / (1.0 + exchange_value(Role::Pawn)));
        let bishop = KING_CAND_ATTACK_W * far * (2.0 / (1.0 + exchange_value(Role::Bishop)));
        assert!(
            pawn > bishop * 8.0,
            "adjacent pawn bonus {pawn} should beat far bishop promo {bishop}"
        );
        // 成る歩の安さはと金でなく歩。と金換算だと cheapness が約半分になり、
        // 玉隣の成り込みが少し遠い垂れ歩（m121 の P*5b）に負ける
        let pawn_c = 2.0 / (1.0 + exchange_value(Role::Pawn));
        let tokin_c = 2.0 / (1.0 + exchange_value(Role::Tokin));
        assert!(
            pawn_c > tokin_c * 1.5,
            "prerole pawn cheapness {pawn_c} must dwarf tokin {tokin_c}"
        );
    }

    /// 王手中の安全解消ゲート: p_max が高いときだけ低い p を捨て、
    /// 王手駒捕獲と玉の手は仮説希釈でも残す。
    #[test]
    fn check_safe_resolve_keeps_diluted_and_drops_ignored_safe() {
        assert!(!check_safe_resolve_active(0.69));
        assert!(check_safe_resolve_active(0.70));
        assert!(check_safe_resolve_active(0.91));
        let thresh = check_safe_resolve_thresh(0.91);
        assert!(0.91 + 1e-12 >= thresh);
        assert!(0.70 + 1e-12 >= thresh);
        assert!(0.41 + 1e-12 < thresh);
        assert!(0.17 + 1e-12 < thresh);
        let thresh70 = check_safe_resolve_thresh(0.70);
        assert!(0.70 + 1e-12 >= thresh70);
        assert!(0.44 + 1e-12 < thresh70);
        assert!(check_safe_resolve_keep(0.70, thresh70, false, false));
        assert!(!check_safe_resolve_keep(0.44, thresh70, false, false));
        assert!(
            check_safe_resolve_keep(0.17, thresh70, true, false),
            "kakutori の捕獲プローブは残す"
        );
        assert!(
            check_safe_resolve_keep(0.17, thresh70, false, true),
            "玉の手は残す"
        );
    }

    /// 信念ネット占有キャップ: 空き寄り（p_occ < 0.25）のときだけ縮み、
    /// ネットも居ると見ているマスは粒子の捕獲期待値を残す。
    /// 呼び出し側は大駒の移動に限定。既定 w=0（env で試す）。
    #[test]
    fn belief_occ_cap_shrinks_only_when_net_says_empty() {
        let ev = 5.0;
        // ネットも居る（0.40 ≥ 0.25）→ 縮まない
        assert_eq!(belief_occ_cap_shrink(ev, 1.0, 0.40, 1.0), 0.0);
        // 完全な空き → 全額
        let empty = belief_occ_cap_shrink(ev, 1.0, 0.0, 1.0);
        assert!(
            (empty - ev).abs() < 1e-9,
            "p_occ=0 は幻の駒得を全額キャンセル: {empty}"
        );
        // 裾の空き: mix = 1-0.05/0.25 = 0.8、effective_p = 0.2+0.8*0.05 = 0.24
        let tail = belief_occ_cap_shrink(ev, 1.0, 0.05, 1.0);
        assert!(
            (tail - ev * 0.76).abs() < 1e-9,
            "p_occ=0.05 は 76% 縮小: {tail}"
        );
        // w=0 は無効
        assert_eq!(belief_occ_cap_shrink(ev, 1.0, 0.0, 0.0), 0.0);
        // 裏付け相当: p_hit が既にネット以下なら縮まない
        assert_eq!(belief_occ_cap_shrink(ev, 0.05, 0.05, 1.0), 0.0);
    }

    #[test]
    fn king_adj_heavy_taxes_tokin_next_to_king_not_two_away() {
        let mut cands = std::collections::BTreeSet::new();
        cands.insert(Coord { file: 5, rank: 1 });
        let backed = [false; 81];
        let me = Color::Sente;
        let four_a =
            king_adj_heavy_amount(Role::Tokin, Coord { file: 4, rank: 1 }, me, &cands, &backed);
        assert!(
            (four_a - exchange_value(Role::Tokin)).abs() < 1e-9,
            "3a4a is adjacent to 5a: {four_a}"
        );
        let three_b =
            king_adj_heavy_amount(Role::Tokin, Coord { file: 3, rank: 2 }, me, &cands, &backed);
        assert_eq!(three_b, 0.0, "2c3b is chebyshev 2 from 5a");
        let one_c =
            king_adj_heavy_amount(Role::Tokin, Coord { file: 1, rank: 3 }, me, &cands, &backed);
        assert!(
            (one_c - exchange_value(Role::Tokin)).abs() < 1e-9,
            "2c1c is off the king file: {one_c}"
        );
        let one_a =
            king_adj_heavy_amount(Role::Tokin, Coord { file: 1, rank: 1 }, me, &cands, &backed);
        assert_eq!(one_a, 0.0, "2a1a is enemy back rank (lance take)");
        let pawn =
            king_adj_heavy_amount(Role::Pawn, Coord { file: 4, rank: 1 }, me, &cands, &backed);
        assert_eq!(pawn, 0.0, "pawns are exempt (4七歩成)");
        assert_eq!(
            king_adj_heavy_amount(Role::Gold, Coord { file: 4, rank: 1 }, me, &cands, &backed),
            0.0,
            "gold next to king is not taxed (arena mid-game)"
        );
        assert_eq!(
            king_adj_heavy_amount(
                Role::Bishop,
                Coord { file: 4, rank: 1 },
                me,
                &cands,
                &backed
            ),
            0.0,
            "bishop next to king is not taxed (arena mid-game)"
        );
        let mut backed_hit = [false; 81];
        backed_hit[crate::belief_features::sq_index(Coord { file: 4, rank: 1 })] = true;
        assert_eq!(
            king_adj_heavy_amount(
                Role::Tokin,
                Coord { file: 4, rank: 1 },
                me,
                &cands,
                &backed_hit
            ),
            0.0,
            "backed recapture is exempt"
        );
        // 全盤候補は focused でない
        let all: std::collections::BTreeSet<Coord> = (1..=9)
            .flat_map(|file| (1..=9).map(move |rank| Coord { file, rank }))
            .collect();
        assert_eq!(
            king_adj_heavy_amount(Role::Tokin, Coord { file: 4, rank: 1 }, me, &all, &backed),
            0.0,
            "unfocused king files must not fire"
        );
    }

    #[test]
    fn tokin_file_drift_taxes_idle_except_major_file_and_king_file() {
        let mut cands = std::collections::BTreeSet::new();
        cands.insert(Coord { file: 5, rank: 9 });
        let backed = [false; 81];
        let gote_view = minimal_view(
            vec![VisiblePiece {
                square: "4g".into(),
                role: Role::Tokin,
            }],
            HashMap::new(),
        );
        // 4g4h: 同じ筋の空前進
        let drift = tokin_file_drift_amount(
            &gote_view,
            Coord { file: 4, rank: 7 },
            Coord { file: 4, rank: 8 },
            Color::Gote,
            &cands,
            &backed,
        );
        assert_eq!(drift, 1.0, "4g4h wanders on a side file");
        // 4g5h: 5 筋へ近づくが空きマス逸れ。接近免税は外した
        let approach_idle = tokin_file_drift_amount(
            &gote_view,
            Coord { file: 4, rank: 7 },
            Coord { file: 5, rank: 8 },
            Color::Gote,
            &cands,
            &backed,
        );
        assert_eq!(
            approach_idle, 1.0,
            "4g5h is still idle (approach is not exempt)"
        );
        // 2c3b: 2 筋に飛車がいるので道を空ける
        let rook_view = minimal_view(
            vec![
                VisiblePiece {
                    square: "2c".into(),
                    role: Role::Tokin,
                },
                VisiblePiece {
                    square: "2a".into(),
                    role: Role::Rook,
                },
            ],
            HashMap::new(),
        );
        let opens = tokin_file_drift_amount(
            &rook_view,
            Coord { file: 2, rank: 3 },
            Coord { file: 3, rank: 2 },
            Color::Sente,
            &cands,
            &backed,
        );
        assert_eq!(opens, 0.0, "2c3b vacates the rook's file");
        // 5c5b: 玉筋に留まる
        let on_file = tokin_file_drift_amount(
            &gote_view,
            Coord { file: 5, rank: 3 },
            Coord { file: 5, rank: 2 },
            Color::Sente,
            &cands,
            &backed,
        );
        assert_eq!(on_file, 0.0, "tokin staying on the king file is exempt");
        let mut hit = [false; 81];
        hit[crate::belief_features::sq_index(Coord { file: 3, rank: 1 })] = true;
        let cap = tokin_file_drift_amount(
            &gote_view,
            Coord { file: 2, rank: 1 },
            Coord { file: 3, rank: 1 },
            Color::Sente,
            &cands,
            &hit,
        );
        assert_eq!(cap, 0.0, "backed capture is exempt");
        let lance_take = tokin_file_drift_amount(
            &gote_view,
            Coord { file: 2, rank: 1 },
            Coord { file: 1, rank: 1 },
            Color::Sente,
            &cands,
            &backed,
        );
        assert_eq!(lance_take, 0.0, "2a1a is enemy back rank (lance take)");
    }

    #[test]
    fn own_camp_idle_taxes_silver_retreat_not_enemy_entry() {
        let backed = [false; 81];
        let idle = own_camp_idle_amount(
            Role::Silver,
            Coord { file: 7, rank: 2 },
            Color::Gote,
            &backed,
            46,
        );
        assert!(
            (idle - exchange_value(Role::Silver)).abs() < 1e-9,
            "7a7b is gote own-camp idle after move 40: {idle}"
        );
        let early = own_camp_idle_amount(
            Role::Silver,
            Coord { file: 7, rank: 2 },
            Color::Gote,
            &backed,
            27,
        );
        assert_eq!(early, 0.0, "m027 3i3h-class must stay exempt");
        let entry = own_camp_idle_amount(
            Role::Silver,
            Coord { file: 4, rank: 9 },
            Color::Gote,
            &backed,
            46,
        );
        assert_eq!(entry, 0.0, "3h4i is enemy camp for gote");
    }

    #[test]
    fn hand_asset_drop_taxable_bishop_midboard_not_enemy() {
        let me = Color::Sente;
        assert!(
            hand_asset_drop_taxable(Role::Bishop, Coord { file: 6, rank: 6 }, me),
            "B*6f mid-board is taxable"
        );
        assert!(
            hand_asset_drop_taxable(Role::Bishop, Coord { file: 1, rank: 8 }, me),
            "B*1h own camp stays taxable"
        );
        assert!(
            !hand_asset_drop_taxable(Role::Bishop, Coord { file: 4, rank: 1 }, me),
            "B*4a enemy camp is exempt (m069)"
        );
        assert!(
            !hand_asset_drop_taxable(Role::Bishop, Coord { file: 3, rank: 3 }, me),
            "B*3c enemy camp is exempt"
        );
        assert!(
            !hand_asset_drop_taxable(Role::Pawn, Coord { file: 4, rank: 6 }, me),
            "P*4f mid-board pawn stays exempt (drop probe)"
        );
        assert!(
            hand_asset_drop_taxable(Role::Pawn, Coord { file: 4, rank: 8 }, me),
            "P*4h own-camp pawn stays taxable"
        );
    }

    #[test]
    fn bishop_retreat_taxes_sideways_not_forward() {
        let backed = [false; 81];
        let me = Color::Sente;
        let retreat = bishop_retreat_amount(
            Role::Bishop,
            Coord { file: 2, rank: 4 },
            Coord { file: 3, rank: 5 },
            me,
            &backed,
        );
        assert!(
            (retreat - exchange_value(Role::Bishop)).abs() < 1e-9,
            "2d3e is a retreat for sente: {retreat}"
        );
        let forward = bishop_retreat_amount(
            Role::Bishop,
            Coord { file: 7, rank: 9 },
            Coord { file: 2, rank: 4 },
            me,
            &backed,
        );
        assert_eq!(forward, 0.0, "kakudo 7i2d advances toward enemy camp");
        let into_camp = bishop_retreat_amount(
            Role::Bishop,
            Coord { file: 4, rank: 4 },
            Coord { file: 3, rank: 3 },
            me,
            &backed,
        );
        assert_eq!(into_camp, 0.0, "4d3c enters enemy camp");
        let gold = bishop_retreat_amount(
            Role::Gold,
            Coord { file: 2, rank: 4 },
            Coord { file: 3, rank: 5 },
            me,
            &backed,
        );
        assert_eq!(gold, 0.0, "only bishop/horse");
    }

    #[test]
    fn tokin_file_drift_and_promote_far_defaults() {
        if std::env::var("TSUITATE_TOKIN_FILE_DRIFT_W").is_err() {
            assert!((tokin_file_drift_w() - TOKIN_FILE_DRIFT_W).abs() < 1e-12);
            assert_eq!(TOKIN_FILE_DRIFT_W, 0.0);
        }
        if std::env::var("TSUITATE_KING_FILE_PAWN_W").is_err() {
            assert!((king_file_pawn_w() - KING_FILE_PAWN_W).abs() < 1e-12);
            assert_eq!(KING_FILE_PAWN_W, 0.0);
        }
        if std::env::var("TSUITATE_LINK_ENDGAME_DAMPEN").is_err() {
            assert!((link_endgame_dampen() - LINK_ENDGAME_DAMPEN).abs() < 1e-12);
            assert_eq!(LINK_ENDGAME_DAMPEN, 0.0);
        }
        if std::env::var("TSUITATE_PROMOTE_FAR_W").is_err() {
            assert!((promote_far_w() - PROMOTE_FAR_W).abs() < 1e-12);
            assert_eq!(PROMOTE_FAR_W, 0.0);
        }
        if std::env::var("TSUITATE_KING_ADJ_HEAVY_W").is_err() {
            assert_eq!(KING_ADJ_HEAVY_W, 0.0);
        }
        if std::env::var("TSUITATE_OWN_CAMP_IDLE_W").is_err() {
            assert!((own_camp_idle_w() - OWN_CAMP_IDLE_W).abs() < 1e-12);
            assert_eq!(OWN_CAMP_IDLE_W, 0.0);
        }
        if std::env::var("TSUITATE_BISHOP_RETREAT_W").is_err() {
            assert!((bishop_retreat_w() - BISHOP_RETREAT_W).abs() < 1e-12);
            assert_eq!(BISHOP_RETREAT_W, 0.0);
        }
        if std::env::var("TSUITATE_PAWN_OFFFILE_W").is_err() {
            assert!((pawn_offfile_w() - PAWN_OFFFILE_W).abs() < 1e-12);
            assert_eq!(PAWN_OFFFILE_W, 0.0);
        }
        if std::env::var("TSUITATE_ENDGAME_CAMP_GENERAL_W").is_err() {
            assert!((endgame_camp_general_w() - ENDGAME_CAMP_GENERAL_W).abs() < 1e-12);
            assert_eq!(ENDGAME_CAMP_GENERAL_W, 0.0);
        }
        if std::env::var("TSUITATE_FAR_MAJOR_PROMO_CAPTURE_W").is_err() {
            assert!((far_major_promo_capture_w() - FAR_MAJOR_PROMO_CAPTURE_W).abs() < 1e-12);
            assert_eq!(FAR_MAJOR_PROMO_CAPTURE_W, 0.0);
        }
        if std::env::var("TSUITATE_KING_FILE_PAWN_MID_W").is_err() {
            assert!((king_file_pawn_mid_w() - KING_FILE_PAWN_MID_W).abs() < 1e-12);
            assert_eq!(KING_FILE_PAWN_MID_W, 0.0);
        }
        if std::env::var("TSUITATE_KING_ENDGAME_FLEE_W").is_err() {
            assert!((king_endgame_flee_w() - KING_ENDGAME_FLEE_W).abs() < 1e-12);
            assert_eq!(KING_ENDGAME_FLEE_W, 0.0);
        }
        if std::env::var("TSUITATE_GOLD_JOIN_KING_W").is_err() {
            assert!((gold_join_king_w() - GOLD_JOIN_KING_W).abs() < 1e-12);
            assert_eq!(GOLD_JOIN_KING_W, 0.0);
        }
        if std::env::var("TSUITATE_GOLD_KING_FILE_W").is_err() {
            assert!((gold_king_file_w() - GOLD_KING_FILE_W).abs() < 1e-12);
            assert_eq!(GOLD_KING_FILE_W, 0.0);
        }
        if std::env::var("TSUITATE_KNIGHT_LATE_PROMO_W").is_err() {
            assert!((knight_late_promo_w() - KNIGHT_LATE_PROMO_W).abs() < 1e-12);
            assert_eq!(KNIGHT_LATE_PROMO_W, 0.0);
        }
        if std::env::var("TSUITATE_KNIGHT_ENDGAME_PROMO_W").is_err() {
            assert!((knight_endgame_promo_w() - KNIGHT_ENDGAME_PROMO_W).abs() < 1e-12);
            assert_eq!(KNIGHT_ENDGAME_PROMO_W, 0.0);
        }
        if std::env::var("TSUITATE_KNIGHT_CAMP_EXIT_W").is_err() {
            assert!((knight_camp_exit_w() - KNIGHT_CAMP_EXIT_W).abs() < 1e-12);
            assert_eq!(KNIGHT_CAMP_EXIT_W, 0.0);
        }
        if std::env::var("TSUITATE_SILVER_CAMP_EXIT_W").is_err() {
            assert!((silver_camp_exit_w() - SILVER_CAMP_EXIT_W).abs() < 1e-12);
            assert_eq!(SILVER_CAMP_EXIT_W, 0.0);
        }
    }

    #[test]
    fn pawn_late_promo_taxes_endgame_not_midgame_or_backed() {
        let me = Color::Gote;
        let from = Coord { file: 5, rank: 6 }; // 5f、中段
        let to = Coord { file: 5, rank: 7 }; // 5g、後手の敵陣
        assert_eq!(
            pawn_late_promo_amount(from, to, me, 130, false),
            1.0,
            "m130 の 5f5g+ は課税"
        );
        assert_eq!(
            pawn_late_promo_amount(from, to, me, 110, false),
            0.0,
            "m110 の 5f5g+（7点）は手数 125 未満で免税"
        );
        assert_eq!(
            pawn_late_promo_amount(from, to, me, 26, false),
            0.0,
            "序中盤の歩成りは対象外"
        );
        assert_eq!(
            pawn_late_promo_amount(from, to, me, 130, true),
            0.0,
            "裏付け捕獲は免税"
        );
        assert_eq!(
            pawn_late_promo_amount(
                Coord { file: 5, rank: 5 },
                Coord { file: 5, rank: 6 },
                me,
                130,
                false
            ),
            0.0,
            "敵陣でなければ対象外"
        );
        assert_eq!(
            pawn_late_promo_amount(
                Coord { file: 7, rank: 3 },
                Coord { file: 7, rank: 2 },
                Color::Sente,
                127,
                false
            ),
            0.0,
            "m127 の 7c7b+ は既に敵陣なので免税"
        );
    }

    #[test]
    fn endgame_camp_general_boosts_promoted_silver_in_camp() {
        let me = Color::Sente;
        let from = Coord { file: 7, rank: 3 }; // 7c
        let to = Coord { file: 8, rank: 2 }; // 8b
                                             // m145: 7a/8a/9a/9b/9c → median file 9
        let mut kings = std::collections::BTreeSet::new();
        kings.insert(Coord { file: 7, rank: 1 });
        kings.insert(Coord { file: 8, rank: 1 });
        kings.insert(Coord { file: 9, rank: 1 });
        kings.insert(Coord { file: 9, rank: 2 });
        kings.insert(Coord { file: 9, rank: 3 });
        assert_eq!(
            endgame_camp_general_amount(Role::Promotedsilver, from, to, me, 145, &kings),
            1.0,
            "m145 の 7c8b は玉筋へ近づくので加点"
        );
        assert_eq!(
            endgame_camp_general_amount(Role::Promotedsilver, from, to, me, 80, &kings),
            0.0,
            "手数 125 未満は対象外"
        );
        assert_eq!(
            endgame_camp_general_amount(
                Role::Gold,
                Coord { file: 7, rank: 2 },
                Coord { file: 7, rank: 3 },
                Color::Gote,
                130,
                &kings
            ),
            0.0,
            "後手の 7b7c は自陣なので加点しない"
        );
        assert_eq!(
            endgame_camp_general_amount(Role::Pawn, from, to, me, 145, &kings),
            0.0,
            "歩は対象外"
        );
        assert_eq!(
            endgame_camp_general_amount(
                Role::Promotedsilver,
                Coord { file: 7, rank: 3 },
                Coord { file: 7, rank: 2 },
                me,
                145,
                &kings
            ),
            0.0,
            "m145 の 7c7b は同筋なので加点しない"
        );
        assert_eq!(
            endgame_camp_general_amount(
                Role::Promotedsilver,
                Coord { file: 7, rank: 3 },
                Coord { file: 6, rank: 2 },
                me,
                145,
                &kings
            ),
            0.0,
            "m145 の 7c6b は玉筋から遠ざかるので加点しない"
        );
        assert_eq!(
            endgame_camp_general_amount(Role::Gold, from, to, me, 145, &kings),
            0.0,
            "素の金は対象外（m139 の 6b5a を押し上げない）"
        );
    }

    #[test]
    fn promote_far_amount_positive_for_sideways_bishop_promo() {
        // 4a3b+: 玉が 8a 付近なら着地は遠く、from からも近づかない
        let mut cands = std::collections::BTreeSet::new();
        cands.insert(Coord { file: 8, rank: 1 });
        cands.insert(Coord { file: 7, rank: 1 });
        let from = Coord { file: 4, rank: 1 };
        let to = Coord { file: 3, rank: 2 };
        let amt = promote_far_amount(from, to, &cands);
        assert!(amt >= 1.0, "4a3b+ recedes from the king file: {amt}");
        // 玉の隣へ成り込む手は免税（d_to=1 かつ近づく）
        let near_to = Coord { file: 7, rank: 2 };
        let near_from = Coord { file: 6, rank: 3 };
        let near = promote_far_amount(near_from, near_to, &cands);
        assert_eq!(near, 0.0, "approaching adjacent promo is free");
    }

    #[test]
    fn own_camp_minor_promo_taxes_silver_optional_promote() {
        // 後手の 3h4i+（quest31-m046）: 敵陣での任意成り。
        // 課税は GEN_NONPROMOTE が有効なときだけ（不成が無いと成/不成の差が無い）。
        let gote = Color::Gote;
        let from = Coord { file: 3, rank: 8 };
        let to = Coord { file: 4, rank: 9 };
        assert_eq!(
            promotion_choice(Role::Silver, from, to, gote),
            Promotion::Optional
        );
        if std::env::var("TSUITATE_GEN_NONPROMOTE").is_err() {
            assert_eq!(
                own_camp_minor_promo_amount(Role::Silver, from, to, true, gote),
                0.0,
                "GEN 既定オフでは課税しない"
            );
        }
        assert_eq!(
            own_camp_minor_promo_amount(Role::Silver, from, to, false, gote),
            0.0
        );
        // 先手の歩成りは対象外
        assert_eq!(
            own_camp_minor_promo_amount(
                Role::Pawn,
                Coord { file: 4, rank: 4 },
                Coord { file: 4, rank: 3 },
                true,
                Color::Sente
            ),
            0.0,
            "pawn promotes should not be taxed"
        );
        // 先手が自陣で銀を動かしても成れない
        assert_eq!(
            own_camp_minor_promo_amount(
                Role::Silver,
                Coord { file: 3, rank: 8 },
                Coord { file: 4, rank: 9 },
                true,
                Color::Sente
            ),
            0.0,
            "sente cannot promote on rank 9"
        );
    }

    #[test]
    fn own_king_drop_defensive_gold_adjacent_silver_head_two() {
        let king = Coord { file: 5, rank: 9 };
        let me = Color::Sente;
        // 銀は玉頭2マスだけ
        assert!(!own_king_drop_is_defensive(
            Role::Silver,
            Coord { file: 5, rank: 8 },
            king,
            me,
        ));
        assert!(!own_king_drop_is_defensive(
            Role::Silver,
            Coord { file: 4, rank: 9 },
            king,
            me,
        ));
        assert!(own_king_drop_is_defensive(
            Role::Silver,
            Coord { file: 5, rank: 7 },
            king,
            me,
        ));
        assert!(!own_king_drop_is_defensive(
            Role::Silver,
            Coord { file: 3, rank: 8 },
            king,
            me,
        ));
        // 金は 8 近傍も守り（m055 の G*5h / 斜め G*4h）
        assert!(own_king_drop_is_defensive(
            Role::Gold,
            Coord { file: 5, rank: 8 },
            king,
            me,
        ));
        assert!(own_king_drop_is_defensive(
            Role::Gold,
            Coord { file: 4, rank: 9 },
            king,
            me,
        ));
        assert!(own_king_drop_is_defensive(
            Role::Gold,
            Coord { file: 5, rank: 7 },
            king,
            me,
        ));
        assert!(!own_king_drop_is_defensive(
            Role::Gold,
            Coord { file: 5, rank: 5 },
            king,
            me,
        ));
        assert!(!own_king_drop_is_defensive(
            Role::Gold,
            Coord { file: 3, rank: 8 },
            king,
            me,
        ));
    }

    #[test]
    fn in_own_camp_is_bottom_three_ranks() {
        assert!(in_own_camp(Coord { file: 1, rank: 8 }, Color::Sente));
        assert!(!in_own_camp(Coord { file: 3, rank: 6 }, Color::Sente));
        assert!(in_own_camp(Coord { file: 5, rank: 2 }, Color::Gote));
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
        assert!(
            after < value,
            "取った駒は相手の盤上から消える: {after} < {value}"
        );
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
        assert_eq!(
            counts.get(&back),
            Some(&2),
            "5九の形も2回出ている＝往復の証拠"
        );

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
        assert_eq!(
            counts.len(),
            2,
            "初期形と 5八飛の形の2つだけが記録されている"
        );
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
        let a = own_effects_after(
            &bottom_pawn,
            &parse_usi("5i4i").unwrap(),
            None,
            None,
            &EvalParams::default(),
        );
        let b = own_effects_after(
            &corner_tokin,
            &parse_usi("5i4i").unwrap(),
            None,
            None,
            &EvalParams::default(),
        );
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
        let drop = own_effects_after(
            &view,
            &parse_usi("L*1b").unwrap(),
            None,
            None,
            &EvalParams::default(),
        );
        assert_eq!(drop.board_material_added, piece_value(Role::Lance));
        // 静かな盤上の手: 増分ゼロ
        let quiet = own_effects_after(
            &view,
            &parse_usi("2d2c").unwrap(),
            None,
            None,
            &EvalParams::default(),
        );
        assert_eq!(quiet.board_material_added, 0.0);
        // 成り: 増えたぶんだけ（歩1 → と6）
        let promo = own_effects_after(
            &view,
            &parse_usi("2d2c+").unwrap(),
            None,
            None,
            &EvalParams::default(),
        );
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
        assert!(out
            .iter()
            .all(|(p, _)| p.king_square(Color::Gote) == Some(b)));
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
        let alone = own_effects_after(
            &view,
            &parse_usi("5h5g").unwrap(),
            None,
            None,
            &EvalParams::default(),
        );
        assert_eq!(alone.linked_value, 0.0, "5g へ出れば玉から離れて紐が切れる");
        let stay = own_effects_after(
            &view,
            &parse_usi("5i4i").unwrap(),
            None,
            None,
            &EvalParams::default(),
        );
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
        let two = own_effects_after(
            &view2,
            &parse_usi("5i4i").unwrap(),
            None,
            None,
            &EvalParams::default(),
        );
        assert!(two.linked_value > stay.linked_value);
    }

    #[test]
    fn king_holes_counts_unsupported_neighbours() {
        // 先手玉5九だけの盤。隣接8マスのうち盤内は5マス（4八,5八,6八,4九,6九）で
        // 玉自身の利きは数えないので、守りが無ければ全部が穴
        let lone_king = vec![VisiblePiece {
            square: "5i".into(),
            role: Role::King,
        }];
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
            vec![VisiblePiece {
                square: "5e".into(),
                role: Role::King,
            }],
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

    /// 粒子スナップショット（issue #28 の診断）は**明示的に on にしたときだけ**
    /// 保存する。既定はコピーもしないので実対局の挙動とコストは不変
    #[test]
    fn 粒子スナップショットは診断のときだけ保存する() {
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
        // 王手中にすると定跡を通らずに候補評価まで進む
        view.you_in_check = true;
        let mut strat = EstimatorStrategy::new();
        let log = ObservationLog::default();

        assert!(strat.choose(&view, &log, &HashSet::new()).is_some());
        assert!(
            strat.last_particles().is_none(),
            "既定では保存しない（実対局のコストを増やさない）"
        );

        strat.set_capture_particles(true);
        assert!(strat.choose(&view, &log, &HashSet::new()).is_some());
        let snap = strat.last_particles().expect("on にしたら残る");
        assert!(
            !snap.strict.is_empty() || !snap.taint.is_empty(),
            "評価に使ったプールが空ではない"
        );
        assert_eq!(snap.entries().len(), snap.strict.len() + snap.taint.len());
        assert!(snap.entries().iter().filter(|e| e.2).count() == snap.strict.len());

        strat.set_capture_particles(false);
        assert!(strat.last_particles().is_none(), "off に戻したら捨てる");
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
        assert!(make("estimator_v14").is_some());
        assert!(make_seeded("estimator_v14", 1).is_some());
        // 破棄済みの凍結版は登録されていない
        assert!(make("estimator_v5").is_none());
    }
}
