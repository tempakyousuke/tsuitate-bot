//! 指し手の選択。
//!
//! 現状はサイト内蔵の簡易botと同じ「前進を好むヒューリスティック＋乱数」。
//! 将来ここを観測履歴（observation.rs）ベースの情報集合探索に置き換える。

use std::collections::HashSet;

use rand::Rng;

use crate::board::{
    Promotion, drop_targets, make_usi_drop, make_usi_move, move_targets, parse_usi_square,
    promotion_choice,
};
use crate::protocol::{Color, PlayerView, Role};

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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::protocol::{ClockState, FoulCounts, GameStatus, OpponentInfo, VisiblePiece};

    fn minimal_view(pieces: Vec<VisiblePiece>, hand: HashMap<Role, u32>) -> PlayerView {
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
}
