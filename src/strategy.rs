//! 指し手の選択。
//!
//! `Strategy` trait の実装を差し替えて強さを比較する（bin/arena.rs で対戦できる）。
//! - `Heuristic`: サイト内蔵の簡易botと同じ「前進を好むヒューリスティック＋乱数」
//! - `EstimatorStrategy`: 観測履歴から相手局面の粒子集合を維持し（estimator.rs）、
//!   候補手を粒子平均で評価する

use std::collections::{HashMap, HashSet};

use rand::Rng;

use crate::board::{
    Coord, Promotion, drop_targets, make_usi_drop, make_usi_move, make_usi_square, move_targets,
    parse_usi_square, promotion_choice,
};
use crate::check::CheckSolver;
use crate::estimator::{Estimator, predict_opp_reply};
use crate::opening::OpeningBook;
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{Color, PlayerView, Role};
use crate::shogi::{Position, ShogiMove, parse_usi, piece_value};

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
}

pub const DEFAULT_STRATEGY: &str = "estimator";

/// 戦略名からインスタンスを作る。未知の名前は None。
/// `estimator_vN` はアリーナ比較用の凍結版（src/frozen/）
pub fn make(name: &str) -> Option<Box<dyn Strategy>> {
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
        "estimator_v2" => Some(Box::new(crate::frozen::estimator_v2::EstimatorV2::new())),
        "estimator_v3" => Some(Box::new(crate::frozen::estimator_v3::EstimatorV3::new())),
        "estimator_v4" => Some(Box::new(crate::frozen::estimator_v4::EstimatorV4::new())),
        "estimator_v5" => Some(Box::new(crate::frozen::estimator_v5::EstimatorV5::new())),
        _ => None,
    }
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

/// 評価に使う粒子数の上限（思考時間の予算。粒子は estimator 側で最大400）。
/// フィッシャー300秒+3秒に対し1手1〜2秒が目安。96粒子で平均370ms程度だったので
/// 精度側（反則率の低下）に予算を振る
const EVAL_PARTICLES: usize = 192;

/// ソフト救済された粒子の評価重み（0.5^penalty）。厳密整合の粒子=1.0
const SOFT_PARTICLE_DECAY: f64 = 0.5;

/// 2手読み（相手応手のサンプル再評価）を行う上位候補数。
/// 1手読みの静的リスク項は近似なので、有望手だけ実際の応手分布で検算する
const DEPTH2_TOP_K: usize = 8;
/// 2手読みに使う粒子数（1候補あたり）。応手の合法手列挙が重いので絞る
const DEPTH2_PARTICLES: usize = 48;
/// 静的リスク項をサンプル実測に置き換える割合（0=従来、1=全面置換）
const DEPTH2_REPLACE: f64 = 0.7;
/// 応手で王手を掛けられた場合のペナルティ（王手中は反則リスクが跳ねる）
const DEPTH2_CHECK_PEN: f64 = 0.3;
/// 応手で詰まされる場合のペナルティ
const DEPTH2_MATE_PEN: f64 = 30.0;
/// 取り返し補償の割引（取り返し自体がさらに取り返されるリスクの近似）
const DEPTH2_RECAP_DISCOUNT: f64 = 0.7;

/// evaluate() の結果。2手読みの再評価に必要な内訳つき
struct EvalOut {
    score: f64,
    /// 静的な取られリスク項（mover/hidden の max）の粒子加重平均。
    /// 2手読みがこの分をサンプル実測で置き換える
    risk_mean: f64,
    p_legal: f64,
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
    /// 着手駒の取られリスク重み（王手をかけた手）。王手宣言は「王を攻撃できる
    /// （マス,駒種）」まで仮説を絞らせるので、相手は反則覚悟の探り取りで
    /// 王手駒を高確率で回収できる（対人実戦: 竜の王手→2反則で竜を取られた）
    pub mover_w_check: f64,
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
}

impl Default for EvalParams {
    fn default() -> Self {
        // SPSA第1ラウンドの収束点（2026-07-11、60反復×2×30局 vs estimator_v5）。
        // 手調整からの主な学び: 反則コスト減（探り反則の価値）、info_bonus増、
        // 打ちの一律減点は撤回（drop_bias正）、露出評価は小さい値で復活
        // （camp_scale 0.16 / exposed_known 0.11 / home_knownness 0.15 —
        // 手動で入れた0.25〜1.0は過剰、ゼロも過小だった）。
        // hand_drop_w のみ未チューニング（第1ラウンド後に追加した項）
        EvalParams {
            check_bonus: 0.748,
            check_foul_scale: 0.047,
            mover_w_captured: 0.988,
            mover_w_quiet: 0.671,
            mover_w_check: 0.506,
            capture_reveal_risk: 0.155,
            camp_known_quiet: 0.403,
            camp_scale: 0.159,
            exposed_base: 0.569,
            exposed_known: 0.113,
            home_knownness: 0.147,
            recapture_defended: 0.323,
            exposed_defended: 0.42,
            attack_w: 0.062,
            pressure_w: 0.151,
            foul_cost_base: 1.005,
            foul_cost_pow: 1.518,
            advance_w: 0.057,
            promote_bias: 0.175,
            drop_bias: 0.16,
            prior_weight: 4.649,
            prior_weight_degen: 4.718,
            threat_w: 0.305,
            info_bonus: 0.832,
            big_home_penalty: 0.352,
            hand_drop_w: 0.08,
            backtrack_penalty: 0.363,
            shuffle_penalty: 0.249,
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
    pub const SPECS: [ParamSpec; 28] = [
        ParamSpec { name: "check_bonus", lo: 0.0, hi: 3.0 },
        ParamSpec { name: "check_foul_scale", lo: 0.0, hi: 0.5 },
        ParamSpec { name: "mover_w_captured", lo: 0.0, hi: 1.5 },
        ParamSpec { name: "mover_w_quiet", lo: 0.0, hi: 1.5 },
        ParamSpec { name: "mover_w_check", lo: 0.0, hi: 1.5 },
        ParamSpec { name: "capture_reveal_risk", lo: 0.0, hi: 0.6 },
        ParamSpec { name: "camp_known_quiet", lo: 0.0, hi: 1.0 },
        ParamSpec { name: "camp_scale", lo: 0.0, hi: 3.0 },
        ParamSpec { name: "exposed_base", lo: 0.0, hi: 1.5 },
        ParamSpec { name: "exposed_known", lo: 0.0, hi: 1.5 },
        ParamSpec { name: "home_knownness", lo: 0.0, hi: 1.0 },
        ParamSpec { name: "recapture_defended", lo: 0.0, hi: 1.0 },
        ParamSpec { name: "exposed_defended", lo: 0.0, hi: 1.0 },
        ParamSpec { name: "attack_w", lo: 0.0, hi: 0.5 },
        ParamSpec { name: "pressure_w", lo: 0.0, hi: 0.6 },
        ParamSpec { name: "foul_cost_base", lo: 0.2, hi: 6.0 },
        ParamSpec { name: "foul_cost_pow", lo: 0.5, hi: 3.0 },
        ParamSpec { name: "advance_w", lo: -0.1, hi: 0.3 },
        ParamSpec { name: "promote_bias", lo: -0.2, hi: 0.6 },
        ParamSpec { name: "drop_bias", lo: -0.5, hi: 0.3 },
        ParamSpec { name: "prior_weight", lo: 0.5, hi: 16.0 },
        ParamSpec { name: "prior_weight_degen", lo: 0.0, hi: 32.0 },
        ParamSpec { name: "threat_w", lo: 0.0, hi: 1.0 },
        ParamSpec { name: "info_bonus", lo: 0.0, hi: 2.0 },
        ParamSpec { name: "big_home_penalty", lo: 0.0, hi: 1.5 },
        ParamSpec { name: "hand_drop_w", lo: 0.0, hi: 0.5 },
        ParamSpec { name: "backtrack_penalty", lo: 0.0, hi: 1.5 },
        ParamSpec { name: "shuffle_penalty", lo: 0.0, hi: 1.0 },
    ];

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.check_bonus,
            self.check_foul_scale,
            self.mover_w_captured,
            self.mover_w_quiet,
            self.mover_w_check,
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
        ]
    }

    pub fn from_vec(v: &[f64]) -> EvalParams {
        assert_eq!(v.len(), Self::SPECS.len());
        EvalParams {
            check_bonus: v[0],
            check_foul_scale: v[1],
            mover_w_captured: v[2],
            mover_w_quiet: v[3],
            mover_w_check: v[4],
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
pub struct EstimatorStrategy {
    est: Option<Estimator>,
    book: Option<OpeningBook>,
    /// Some なら定跡をこのラインに固定する（定跡特化チューニング用）
    book_line: Option<usize>,
    params: EvalParams,
    /// 直近の choose 時点の内部状態（記録用）
    last_debug: Option<serde_json::Value>,
}

impl EstimatorStrategy {
    pub fn new() -> Self {
        Self::with_params(EvalParams::default())
    }

    /// パラメータを差し替えて作る（bin/tune.rs のSPSA評価用）
    pub fn with_params(params: EvalParams) -> Self {
        Self::with_params_and_line(params, None)
    }

    /// パラメータと定跡ライン固定を指定して作る（定跡特化チューニング用）
    pub fn with_params_and_line(params: EvalParams, book_line: Option<usize>) -> Self {
        EstimatorStrategy {
            est: None,
            book: None,
            book_line,
            params,
            last_debug: None,
        }
    }
}

impl Default for EstimatorStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl Strategy for EstimatorStrategy {
    fn choose(
        &mut self,
        view: &PlayerView,
        log: &ObservationLog,
        foul_tried: &HashSet<String>,
    ) -> Option<String> {
        let est = self
            .est
            .get_or_insert_with(|| Estimator::new(view.your_color));
        est.update(log);

        // 序盤定跡（静かな間だけ）。ブック中も推定器の update は回して粒子を保つ
        let book_line = self.book_line;
        let book = self.book.get_or_insert_with(|| match book_line {
            Some(idx) => OpeningBook::with_line(view.your_color, idx),
            None => OpeningBook::new(view.your_color),
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

        // 複製粒子を指紋で除いたユニーク粒子だけを評価に使う
        // （複製は独立な証拠ではないので p(合法) を過信させる）。
        // ソフト救済された粒子（penalty>0）は重み 0.5^penalty で薄く数える。
        // 粒子は penalty 昇順なので厳密整合の粒子から先に採用される。
        // 粒子が完全に枯渇していても、事前確率だけで安全側の評価が成り立つ
        let mut seen = HashSet::new();
        let mut sample: Vec<(&Position, f64)> = vec![];
        for (pos, pen) in est.particles().iter().zip(est.penalties()) {
            if sample.len() >= EVAL_PARTICLES {
                break;
            }
            if seen.insert(pos.fingerprint()) {
                sample.push((pos, SOFT_PARTICLE_DECAY.powi(i32::from(*pen))));
            }
        }

        // 相手の盤上駒数の概算（取った枚数ぶん減る。相手の打ちで戻る分は無視）
        let my_captures = log
            .events()
            .iter()
            .filter(|e| matches!(e, Observation::MyMove { captured: Some(_), .. }))
            .count();
        let opp_board_n = (20 - my_captures.min(19)) as f64;

        // 直前に受理された自分の手（手戻りシャッフルの抑制に使う）
        let last_my_move = log.events().iter().rev().find_map(|e| match e {
            Observation::MyMove { usi, .. } => parse_usi(usi),
            _ => None,
        });

        // 王手中は粒子に依存しない制約推論で「王手を解消する確率」を出す
        // （粒子が枯渇する終盤の反則バースト対策。check.rs 参照）
        let mut check_solver = if view.you_in_check {
            let fouls: Vec<ShogiMove> =
                foul_tried.iter().filter_map(|u| parse_usi(u)).collect();
            let positions: Vec<&Position> = sample.iter().map(|(p, _)| *p).collect();
            CheckSolver::new(view, &positions, &fouls, log)
        } else {
            None
        };

        // 相手が位置を知っている自駒（露出）の地図
        let known = knownness_map(view, log, self.params.home_knownness);

        // 2手読み用: 自分が駒を取ったマス（露見）と自分の手が触れたマス
        // （estimator の my_capture_sq / my_touched_sq と同じ定義）
        let mut my_capture_squares: Vec<Coord> = vec![];
        let mut my_touched_squares: Vec<Coord> = vec![];
        for e in log.events() {
            if let Observation::MyMove { usi, captured, .. } = e {
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
        }

        let mut rng = rand::rng();
        // 1段目: 全候補を1手読み（静的リスク項つき）で評価する
        let mut scored: Vec<(String, ShogiMove, EvalOut, f64)> = vec![];
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
            let out = evaluate(view, &mv, &sample, prior, &known, &self.params);
            let mut score = out.score + rng.random_range(0.0..0.01);
            // 手戻り（直前の手をそのまま逆に戻す）は膠着の典型なので減点。
            // 直前に動かした駒をまた動かすだけの手も雑なシャッフルとして軽く減点
            if let (
                Some(ShogiMove::Board { from: pf, to: pt, .. }),
                ShogiMove::Board { from, to, .. },
            ) = (last_my_move, mv)
            {
                if from == pt && to == pf {
                    score -= self.params.backtrack_penalty;
                } else if from == pt {
                    score -= self.params.shuffle_penalty;
                }
            }
            scored.push((usi, mv, out, score));
        }

        // 2段目: 上位候補だけ相手の応手をサンプルして再評価。
        // 静的リスク項の DEPTH2_REPLACE 分を実測の期待損失で置き換える
        // （risk_mean を足し戻し、サンプルの delta を足す。両者が一致するなら無変化）
        scored.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        let mut best: Option<(String, f64)> = None;
        for (i, (usi, mv, out, score)) in scored.into_iter().enumerate() {
            let final_score = if i < DEPTH2_TOP_K {
                let delta = depth2_delta(
                    view,
                    &mv,
                    &sample,
                    &known,
                    &my_capture_squares,
                    &my_touched_squares,
                    &self.params,
                    &mut rng,
                );
                score + out.p_legal * DEPTH2_REPLACE * (out.risk_mean + delta)
            } else {
                score
            };
            if best.as_ref().is_none_or(|(_, s)| final_score > *s) {
                best = Some((usi, final_score));
            }
        }

        self.last_debug = Some(debug_summary(est, &sample));
        best.map(|(usi, _)| usi)
    }

    fn name(&self) -> &'static str {
        "estimator"
    }

    fn debug_state(&self) -> Option<serde_json::Value> {
        self.last_debug.clone()
    }
}

/// 記録用の推定サマリ: 粒子の健全性・ユニーク数・相手玉の位置分布（上位）。
/// 事後分析で「推定が外れていたのか、評価が悪かったのか」を切り分けるために残す
fn debug_summary(est: &Estimator, sample: &[(&Position, f64)]) -> serde_json::Value {
    let opp = est.my_color().other();
    let mut king_votes: HashMap<Coord, u32> = HashMap::new();
    for (pos, _) in sample {
        if let Some(sq) = pos.king_square(opp) {
            *king_votes.entry(sq).or_default() += 1;
        }
    }
    let mut top: Vec<(Coord, u32)> = king_votes.into_iter().collect();
    top.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    let n = sample.len().max(1) as f64;
    let opp_king_top: Vec<serde_json::Value> = top
        .iter()
        .take(3)
        .map(|(sq, votes)| {
            serde_json::json!({
                "sq": make_usi_square(*sq),
                "p": *votes as f64 / n,
            })
        })
        .collect();
    serde_json::json!({
        "healthy": est.healthy(),
        "unique_particles": sample.len(),
        "soft_particles": est.penalties().iter().filter(|&&p| p > 0).count(),
        "opp_king_top": opp_king_top,
    })
}

/// 自分に見える範囲の候補手（foul_tried を除く）。bin/analyze の検証でも使う
pub fn candidate_moves(view: &PlayerView, foul_tried: &HashSet<String>) -> Vec<(String, ShogiMove)> {
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
fn evaluate(
    view: &PlayerView,
    mv: &ShogiMove,
    particles: &[(&Position, f64)],
    prior: f64,
    known: &HashMap<Coord, f64>,
    params: &EvalParams,
) -> EvalOut {
    let me = view.your_color;
    let opp = me.other();
    let mut legal = 0.0f64;
    let mut value_sum = 0.0;
    let mut risk_sum = 0.0;
    // 着地マスに敵駒がいた（=駒を取れた）粒子の重み。探索ボーナスの不一致度に使う
    let mut capture_hits = 0.0f64;
    // 王周辺の圧力は粒子間の分散が小さいわりに計算が重い（9マス×利き走査）ので
    // 少数の粒子でだけ測って平均する
    const PRESSURE_SAMPLES: usize = 16;
    let mut pressure_sum = 0.0;
    let mut attack_sum = 0.0;
    let mut danger_sum = 0.0;
    let mut pressure_n = 0usize;

    for (pos, w) in particles {
        let w = *w;
        if !pos.is_legal(mv) {
            continue;
        }
        legal += w;
        let mut v = 0.0;

        // 駒得（盤上価値で数える。成駒を取れば大きい）
        let mut captured_value = 0.0;
        if let ShogiMove::Board { to, .. } = *mv {
            if let Some(p) = pos.piece_at(to) {
                if p.color == opp {
                    captured_value = piece_value(p.role);
                }
            }
        }
        v += captured_value;
        if captured_value > 0.0 {
            capture_hits += w;
        }

        let mut next = (*pos).clone();
        next.play_unchecked(mv);

        // 王手・詰み。ついたて将棋では王手された側は王手駒の位置が見えず
        // 手探りの反則をしやすい（反則10回で負け）ので、王手自体が得点源。
        // 相手の反則が溜まっているほど価値が上がる
        let gives_check = next.in_check(opp);
        if gives_check {
            v += params.check_bonus + params.check_foul_scale * f64::from(view.fouls.opponent);
            if next.legal_moves().is_empty() {
                v += 1000.0; // 詰み（真の局面がこの粒子なら勝ち）
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
            mover_w = mover_w.max(params.mover_w_check);
        }
        let own_after = next
            .piece_at(to)
            .map(|p| piece_value(p.role))
            .unwrap_or(0.0);
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
        let risk = mover_risk.max(hidden_risk);
        v -= risk;
        risk_sum += w * risk;

        // 自分が敵駒に当たりを付けている価値（露出リスクの鏡像）。
        // 1手読みでは見えない「次の駒得」を作る手（大駒の頭への歩打ち等）に価値を与える
        v += params.threat_w * threat_value(&next, me);

        // 王の安全度と攻撃圧力（利き走査が重いので少数の粒子でだけ測って平均する）
        if pressure_n < PRESSURE_SAMPLES {
            // 自玉の周囲に当たっている相手の利き（守り）
            pressure_sum += king_zone_pressure(&next, me, opp);
            // 相手玉の周囲に当たっている自分の利き（攻め）。王手にならない攻め駒の
            // 集結にも報酬を与える（王手/詰みボーナスだけだと攻めを組み立てない）
            attack_sum += king_zone_pressure(&next, opp, me);
            // 相手の持ち駒による王手打ちの受け入れ面積（対局実験の教訓:
            // 飛車を持たれた瞬間、玉への開いた直線はすべて即王手の入口になる）
            danger_sum += drop_check_danger(&next, me);
            pressure_n += 1;
        }

        value_sum += w * v;
    }

    // 粒子の証拠と事前確率のブレンド（粒子ゼロなら事前そのもの）。
    // 粒子が退化している（実効重みが評価上限に届かない）ほど事前の重みを
    // 増やし、少数の偏った粒子への過信を防ぐ。ソフト粒子は重みぶんしか
    // 数えないので、退化度にも自然に反映される
    let n: f64 = particles.iter().map(|(_, w)| w).sum();
    let degen = 1.0 - (n / EVAL_PARTICLES as f64).min(1.0);
    let w = params.prior_weight + params.prior_weight_degen * degen;
    let p_legal = (legal + prior * w) / (n + w);
    let expected = if legal > 0.0 {
        // 探索ボーナス: 着地マスの敵駒有無について粒子が割れているほど、
        // 指せば（取れても空でも）推定が絞れる。捕獲の期待値とは別の情報の価値
        let p_hit = capture_hits / legal;
        // 攻め圧力は粒子の健全度でゲートする。退化した粒子は間違った玉位置に
        // 固まりやすく、「誰もいない場所への攻め」が加点され続ける
        // （対人実戦: 終盤の成桂の徘徊）。健全度が低いときは確実な項だけ残す
        let confidence = (n / EVAL_PARTICLES as f64).min(1.0);
        value_sum / legal
            + params.info_bonus * p_hit * (1.0 - p_hit)
            + (params.attack_w * confidence * attack_sum
                - params.pressure_w * pressure_sum
                - params.hand_drop_w * danger_sum)
                / pressure_n.max(1) as f64
    } else {
        0.0
    };

    // 反則コスト: 手番は失わないが反則数を消費する。残りが少ないほど急激に高価。
    // 序盤の「安い反則で情報を得る」は低コスト側で自然に許容される
    let fouls_left = (10u32.saturating_sub(view.fouls.you)).max(1) as f64;
    let foul_cost = params.foul_cost_base * (10.0 / fouls_left).powf(params.foul_cost_pow);

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

    // 期待値が負の手を p_legal で割り引かない（min の形）。
    // 割り引くと「合法確率が低いほどスコアが高い」= わざと反則に寄る手が
    // 選ばれてしまう。反則しても手番は残るので悪い局面からは逃げられず、
    // 反則の価値は「次善手の価値 − 反則コスト」でしかない
    let gain = expected + advance_bias + development;
    EvalOut {
        score: (p_legal * gain).min(gain) - (1.0 - p_legal) * foul_cost,
        risk_mean: if legal > 0.0 { risk_sum / legal } else { 0.0 },
        p_legal,
    }
}

/// 2手読み: 候補手の後に相手の応手を粒子上でサンプルし、実測の期待損失
/// （露見度で割引した駒損 − 取り返し補償、被王手/被詰みペナルティ）を返す。
/// 静的リスク項（EvalOut::risk_mean）の置き換え先。値は「加点」方向（通常は負）
#[allow(clippy::too_many_arguments)]
fn depth2_delta(
    view: &PlayerView,
    mv: &ShogiMove,
    particles: &[(&Position, f64)],
    known: &HashMap<Coord, f64>,
    my_captures: &[Coord],
    my_touched: &[Coord],
    params: &EvalParams,
    rng: &mut impl rand::Rng,
) -> f64 {
    let me = view.your_color;
    let to = match *mv {
        ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
    };
    let mut sum = 0.0;
    let mut n = 0.0;
    for (pos, w) in particles.iter().take(DEPTH2_PARTICLES) {
        if !pos.is_legal(mv) {
            continue;
        }
        let mut next = (*pos).clone();
        let my_capture = next.play_unchecked(mv);
        let gives_check = next.in_check(me.other());
        n += w;
        let Some(reply) = predict_opp_reply(&next, me, my_captures, my_touched, rng) else {
            continue; // 応手なし（詰み/ステイルメイト）は stage1 のボーナス側で評価済み
        };
        let reply_to = match reply {
            ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
        };
        let lost = next
            .piece_at(reply_to)
            .filter(|p| p.color == me)
            .map(|p| piece_value(p.role))
            .unwrap_or(0.0);
        let mut next2 = next.clone();
        next2.play_unchecked(&reply);
        let mut d = 0.0;
        if lost > 0.0 {
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
                    s = s.max(params.mover_w_check);
                }
                s
            } else {
                let knownness = known.get(&reply_to).copied().unwrap_or(0.0);
                params.exposed_base + params.exposed_known * knownness
            };
            // 取り返し補償: 応手の駒に自分の利きが残っていれば取り返せる
            let comp = if !next2.in_check(me) && next2.is_attacked(reply_to, me) {
                DEPTH2_RECAP_DISCOUNT
                    * next2
                        .piece_at(reply_to)
                        .map(|p| piece_value(p.role))
                        .unwrap_or(0.0)
            } else {
                0.0
            };
            d -= scale * (lost - comp).max(0.0);
        }
        if next2.in_check(me) {
            d -= DEPTH2_CHECK_PEN;
            if next2.legal_moves().is_empty() {
                d -= DEPTH2_MATE_PEN;
            }
        }
        sum += w * d;
    }
    if n > 0.0 { sum / n } else { 0.0 }
}

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
/// 玉への当たりは王手であり合法性・王手ボーナス側で扱うので除く
fn threat_value(pos: &Position, me: Color) -> f64 {
    let opp = me.other();
    let mut best = 0.0f64;
    for (sq, piece) in pos.pieces() {
        if piece.color != opp || piece.role == Role::King {
            continue;
        }
        if !pos.is_attacked(sq, me) {
            continue;
        }
        let defended = pos.is_attacked(sq, opp);
        let gain = piece_value(piece.role) * if defended { 0.45 } else { 1.0 };
        best = best.max(gain);
    }
    best
}

/// 着手駒（マス to にいる自駒）が次の相手番で取られるリスク。
/// 紐つきなら取り返せるぶん割り引く（相手のどの駒で取るかは不明なので近似）
fn recapture_risk(pos: &Position, me: Color, to: Coord, defended_discount: f64) -> f64 {
    let opp = me.other();
    let Some(piece) = pos.piece_at(to).filter(|p| p.color == me) else {
        return 0.0;
    };
    if piece.role == Role::King || !pos.is_attacked(to, opp) {
        return 0.0;
    }
    let defended = pos.is_attacked(to, me);
    piece_value(piece.role) * if defended { defended_discount } else { 1.0 }
}

/// 次の相手番で失いうる駒の概算: 相手の利きが当たっている自駒の最大重み付き価値。
/// 自分の利きも当たっている（紐つき）なら取り返せるぶん割り引く。
/// 相手がその駒の位置を知っているほど（knownness_map）実際に取られやすいので
/// 重みを引き上げる。位置が漏れていない駒は従来通り薄く見積もる。
/// exclude（着手駒のマス）は recapture_risk 側で別の重みで数えるので除外する。
/// 合法手の完全列挙（ピン考慮など）はコストに見合わないので利きベースの近似
fn exposed_capture_risk(
    pos: &Position,
    me: Color,
    exclude: Option<Coord>,
    known: &HashMap<Coord, f64>,
    params: &EvalParams,
) -> f64 {
    let opp = me.other();
    let mut worst = 0.0f64;
    for (sq, piece) in pos.pieces() {
        if piece.color != me || piece.role == Role::King {
            continue; // 玉が当たっているなら王手なので合法性の側で処理される
        }
        if exclude == Some(sq) {
            continue;
        }
        if !pos.is_attacked(sq, opp) {
            continue;
        }
        let defended = pos.is_attacked(sq, me);
        let knownness = known.get(&sq).copied().unwrap_or(0.0);
        let weight = params.exposed_base + params.exposed_known * knownness;
        let loss = piece_value(piece.role)
            * if defended { params.exposed_defended } else { 1.0 }
            * weight;
        worst = worst.max(loss);
    }
    worst
}

/// 相手の持ち駒による「王手打ちの受け入れ面積」。
/// 相手の持ち駒はこの粒子上で正確に分かる（取られた自駒 − 打たれた駒）。
/// - 飛: 玉からの縦横の空き直線の長さ（その各マスが王手打ちの入口）
/// - 角: 斜めの空き直線の長さ
/// - 香: 相手の香が王手できる側の1直線
/// - 金/銀: 玉の隣接空きマス（打てば即王手）
/// - 歩: 玉頭の1マス
/// 持ち駒が空ならゼロ = 居玉そのものは咎めない
fn drop_check_danger(pos: &Position, me: Color) -> f64 {
    let Some(king) = pos.king_square(me) else {
        return 0.0;
    };
    let opp = me.other();
    let on_board = |c: &Coord| (1..=9).contains(&c.file) && (1..=9).contains(&c.rank);
    let ray_len = |df: i8, dr: i8| -> f64 {
        let mut n = 0;
        let mut c = Coord { file: king.file + df, rank: king.rank + dr };
        while on_board(&c) && pos.piece_at(c).is_none() {
            n += 1;
            c = Coord { file: c.file + df, rank: c.rank + dr };
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
        let head = Coord { file: king.file, rank: king.rank + toward };
        if on_board(&head) && pos.piece_at(head).is_none() {
            danger += 1.0;
        }
    }
    let generals =
        pos.hand_count(opp, Role::Gold) > 0 || pos.hand_count(opp, Role::Silver) > 0;
    if generals {
        let mut air = 0.0;
        for df in -1..=1i8 {
            for dr in -1..=1i8 {
                if df == 0 && dr == 0 {
                    continue;
                }
                let c = Coord { file: king.file + df, rank: king.rank + dr };
                if on_board(&c) && pos.piece_at(c).is_none() {
                    air += 0.5;
                }
            }
        }
        danger += air;
    }
    danger
}

/// owner 玉の周囲8マス（と玉のマス）に当たっている by 側の利きの数
fn king_zone_pressure(pos: &Position, owner: Color, by: Color) -> f64 {
    let Some(king) = pos.king_square(owner) else {
        return 0.0;
    };
    let mut pressure = 0;
    for df in -1..=1i8 {
        for dr in -1..=1i8 {
            let c = crate::board::Coord {
                file: king.file + df,
                rank: king.rank + dr,
            };
            if (1..=9).contains(&c.file)
                && (1..=9).contains(&c.rank)
                && pos.is_attacked(c, by)
            {
                pressure += 1;
            }
        }
    }
    pressure as f64
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
            fouls: FoulCounts { you: 0, opponent: 0 },
            you_in_check: false,
            opponent_in_check: false,
            status: GameStatus::Playing,
        }
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
        assert_eq!(choose_move(&view, &HashSet::new()), Some("7g7f".to_string()));
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
        assert!(ok("7g5g"), "自玉と同段（ライン上）への移動は合駒/取りになりうる");
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
        assert!(make("estimator_v2").is_some());
        assert!(make("estimator_v3").is_some());
        assert!(make("estimator_v4").is_some());
        assert!(make("estimator_v5").is_some());
    }
}
