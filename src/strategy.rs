//! 指し手の選択。
//!
//! `Strategy` trait の実装を差し替えて強さを比較する（bin/arena.rs で対戦できる）。
//! - `Heuristic`: サイト内蔵の簡易botと同じ「前進を好むヒューリスティック＋乱数」
//! - `EstimatorStrategy`: 観測履歴から相手局面の粒子集合を維持し（estimator.rs）、
//!   候補手を粒子平均で評価する

use std::collections::HashSet;

use rand::Rng;

use crate::board::{
    Promotion, drop_targets, make_usi_drop, make_usi_move, move_targets, parse_usi_square,
    promotion_choice,
};
use crate::estimator::Estimator;
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
}

pub const DEFAULT_STRATEGY: &str = "estimator";

/// 戦略名からインスタンスを作る。未知の名前は None。
/// `estimator_vN` はアリーナ比較用の凍結版（src/frozen/）
pub fn make(name: &str) -> Option<Box<dyn Strategy>> {
    match name {
        "heuristic" => Some(Box::new(Heuristic)),
        "estimator" => Some(Box::new(EstimatorStrategy::new())),
        "estimator_v1" => Some(Box::new(crate::frozen::estimator_v1::EstimatorV1::new())),
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

/// 評価に使う粒子数の上限（思考時間の予算。粒子は estimator 側で最大400）
const EVAL_PARTICLES: usize = 96;

/// 観測履歴から相手局面を推定して指す戦略。
///
/// 候補手（自分に見える範囲の疑似合法手）を、推定粒子の平均で評価する:
/// - 駒得の期待値（その粒子でそのマスに相手駒がいるか）
/// - 反則確率（粒子上で非合法な割合）× 反則コスト（残り反則数が減るほど高い）
/// - 指した直後に取り返されるリスク（粒子上での相手の即時駒取り）
/// - 王手・詰みボーナス
pub struct EstimatorStrategy {
    est: Option<Estimator>,
}

impl EstimatorStrategy {
    pub fn new() -> Self {
        EstimatorStrategy { est: None }
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

        let candidates = candidate_moves(view, foul_tried);
        if candidates.is_empty() {
            return None;
        }

        // 複製粒子を指紋で除いたユニーク粒子だけを評価に使う
        // （複製は独立な証拠ではないので p(合法) を過信させる）。
        // 粒子が完全に枯渇していても、事前確率だけで安全側の評価が成り立つ
        let mut seen = HashSet::new();
        let mut sample: Vec<&Position> = vec![];
        for pos in est.particles() {
            if sample.len() >= EVAL_PARTICLES {
                break;
            }
            if seen.insert(pos.fingerprint()) {
                sample.push(pos);
            }
        }

        // 相手の盤上駒数の概算（取った枚数ぶん減る。相手の打ちで戻る分は無視）
        let my_captures = log
            .events()
            .iter()
            .filter(|e| matches!(e, Observation::MyMove { captured: Some(_), .. }))
            .count();
        let opp_board_n = (20 - my_captures.min(19)) as f64;

        let mut rng = rand::rng();
        let mut best: Option<(String, f64)> = None;
        for (usi, mv) in candidates {
            let prior = prior_legal(view, &mv, opp_board_n);
            let score = evaluate(view, &mv, &sample, prior) + rng.random_range(0.0..0.01);
            if best.as_ref().is_none_or(|(_, s)| score > *s) {
                best = Some((usi, score));
            }
        }
        best.map(|(usi, _)| usi)
    }

    fn name(&self) -> &'static str {
        "estimator"
    }
}

/// 自分に見える範囲の候補手（foul_tried を除く）
fn candidate_moves(view: &PlayerView, foul_tried: &HashSet<String>) -> Vec<(String, ShogiMove)> {
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

/// 事前確率の重み（擬似観測数）。粒子が少ない・偏っているときほど事前が効く
const PRIOR_WEIGHT: f64 = 4.0;

/// 候補手をユニーク粒子の平均で評価する
fn evaluate(view: &PlayerView, mv: &ShogiMove, particles: &[&Position], prior: f64) -> f64 {
    let me = view.your_color;
    let opp = me.other();
    let mut legal = 0usize;
    let mut value_sum = 0.0;
    // 王周辺の圧力は粒子間の分散が小さいわりに計算が重い（9マス×利き走査）ので
    // 少数の粒子でだけ測って平均する
    const PRESSURE_SAMPLES: usize = 16;
    let mut pressure_sum = 0.0;
    let mut pressure_n = 0usize;

    for pos in particles {
        if !pos.is_legal(mv) {
            continue;
        }
        legal += 1;
        let mut v = 0.0;

        // 駒得（盤上価値で数える。成駒を取れば大きい）
        if let ShogiMove::Board { to, .. } = *mv {
            if let Some(p) = pos.piece_at(to) {
                if p.color == opp {
                    v += piece_value(p.role);
                }
            }
        }

        let mut next = (*pos).clone();
        next.play_unchecked(mv);

        // 王手・詰み
        if next.in_check(opp) {
            v += 0.8;
            if next.legal_moves().is_empty() {
                v += 1000.0; // 詰み（真の局面がこの粒子なら勝ち）
            }
        }

        // 取り返され・タダ取られリスク: 利きの当たっている自駒の最大値。
        // 相手にはこちらの駒が見えないので、確実に取られるわけではない → 割引き
        v -= 0.6 * exposed_capture_risk(&next, me);

        // 王の安全度: 自玉の周囲8マスに当たっている相手の利きの数
        if pressure_n < PRESSURE_SAMPLES {
            pressure_sum += king_zone_pressure(&next, me);
            pressure_n += 1;
        }

        value_sum += v;
    }

    // 粒子の証拠と事前確率のブレンド（粒子ゼロなら事前そのもの）
    let n = particles.len() as f64;
    let p_legal = (legal as f64 + prior * PRIOR_WEIGHT) / (n + PRIOR_WEIGHT);
    let expected = if legal > 0 {
        value_sum / legal as f64 - 0.2 * pressure_sum / pressure_n.max(1) as f64
    } else {
        0.0
    };

    // 反則コスト: 手番は失わないが反則数を消費する。残りが少ないほど急激に高価。
    // 序盤の「安い反則で情報を得る」は低コスト側で自然に許容される
    let fouls_left = (10u32.saturating_sub(view.fouls.you)).max(1) as f64;
    let foul_cost = 1.5 * (10.0 / fouls_left).powf(1.5);

    // 前進の弱い事前バイアス（推定が薄い序盤に駒をぶつけに行くため）
    let advance_bias = match *mv {
        ShogiMove::Board { from, to, promote } => {
            let adv = match me {
                Color::Sente => (from.rank - to.rank) as f64,
                Color::Gote => (to.rank - from.rank) as f64,
            };
            0.05 * adv + if promote { 0.1 } else { 0.0 }
        }
        ShogiMove::Drop { .. } => -0.05,
    };

    p_legal * (expected + advance_bias) - (1.0 - p_legal) * foul_cost
}

/// 次の相手番で失いうる駒の概算: 相手の利きが当たっている自駒の最大価値。
/// 自分の利きも当たっている（紐つき）なら取り返せるぶん割り引く。
/// 合法手の完全列挙（ピン考慮など）はコストに見合わないので利きベースの近似
fn exposed_capture_risk(pos: &Position, me: Color) -> f64 {
    let opp = me.other();
    let mut worst = 0.0f64;
    for (sq, piece) in pos.pieces() {
        if piece.color != me || piece.role == Role::King {
            continue; // 玉が当たっているなら王手なので合法性の側で処理される
        }
        if !pos.is_attacked(sq, opp) {
            continue;
        }
        let defended = pos.is_attacked(sq, me);
        let loss = piece_value(piece.role) * if defended { 0.4 } else { 1.0 };
        worst = worst.max(loss);
    }
    worst
}

/// 自玉の周囲8マス（と玉のマス）に当たっている相手の利きの数
fn king_zone_pressure(pos: &Position, me: Color) -> f64 {
    let Some(king) = pos.king_square(me) else {
        return 0.0;
    };
    let opp = me.other();
    let mut pressure = 0;
    for df in -1..=1i8 {
        for dr in -1..=1i8 {
            let c = crate::board::Coord {
                file: king.file + df,
                rank: king.rank + dr,
            };
            if (1..=9).contains(&c.file)
                && (1..=9).contains(&c.rank)
                && pos.is_attacked(c, opp)
            {
                pressure += 1;
            }
        }
    }
    pressure as f64
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::protocol::{ClockState, FoulCounts, GameStatus, OpponentInfo, VisiblePiece};

    pub(crate) fn minimal_view(pieces: Vec<VisiblePiece>, hand: HashMap<Role, u32>) -> PlayerView {
        PlayerView {
            game_id: "g".into(),
            your_color: Color::Sente,
            opponent: OpponentInfo {
                username: "aite".into(),
                rating: 1500,
                is_bot: false,
            },
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
    fn make_knows_heuristic() {
        assert!(make("heuristic").is_some());
        assert!(make("nonsense").is_none());
    }

    #[test]
    fn make_knows_frozen_versions() {
        assert!(make("estimator").is_some());
        assert!(make("estimator_v1").is_some());
    }
}
