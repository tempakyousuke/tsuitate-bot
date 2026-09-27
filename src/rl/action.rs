//! 行動の符号化（AlphaZero 将棋と同じ 139 種 × 81 マス）と行動マスク。
//!
//! 手番側を常に先手向きに正規化する（後手なら盤を180度回す）ので、
//! 「前」は常に段が減る方向になる。
//!
//! 行動番号 = 種類 × 81 + マス（方策ヘッドの出力 [139, 9, 9] を平らにした並び）。
//! - 種類 0..64: 8方向 × 距離 1〜8（不成）。マス = 移動元
//! - 種類 64, 65: 桂の2方向（不成）。マス = 移動元
//! - 種類 66..132: 上の 66 種の成り
//! - 種類 132..139: 打ち（`HAND_ROLES` 順）。マス = 打ち先
//!
//! マスクは**自駒だけを見た候補手**（`board.rs`）から、その手番で既に反則した手を除いたもの。
//! 見えている範囲で確定する反則（自駒のマス・自駒の飛び越え・二歩・行き所なし）は落ち、
//! 見えない相手駒による反則は残る。**真の合法手はすべてマスクに含まれる**
//! （テストで常時検査。漏れた手は永久に指せなくなる）。

use std::collections::HashSet;

use crate::board::{
    Coord, Promotion, drop_targets, make_usi_drop, make_usi_move, move_targets,
    parse_usi_square, promotion_choice,
};
use crate::protocol::{Color, PlayerView};
use crate::shogi::{HAND_ROLES, ShogiMove, hand_index, parse_usi};

pub const NUM_SQUARES: usize = 81;
/// 方向×距離 64 ＋ 桂 2
const MOVE_KINDS: usize = 66;
const DROP_BASE: usize = MOVE_KINDS * 2;
pub const NUM_ACTION_KINDS: usize = DROP_BASE + HAND_ROLES.len();
pub const NUM_ACTIONS: usize = NUM_ACTION_KINDS * NUM_SQUARES;

/// 正規化後の8方向 (筋の差, 段の差)。段が減るのが「前」
const DIRS: [(i8, i8); 8] = [
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
];
const KNIGHT_DIRS: [(i8, i8); 2] = [(1, -2), (-1, -2)];

/// 手番側から見た座標へ（後手なら180度回す。自分自身が逆変換）
pub fn normalize(c: Coord, color: Color) -> Coord {
    match color {
        Color::Sente => c,
        Color::Gote => Coord {
            file: 10 - c.file,
            rank: 10 - c.rank,
        },
    }
}

/// 正規化済み座標 → マス番号（段優先: (段-1)×9 + (筋-1)）
pub fn square_index(c: Coord) -> usize {
    (c.rank as usize - 1) * 9 + (c.file as usize - 1)
}

pub fn index_square(i: usize) -> Coord {
    Coord {
        file: (i % 9) as i8 + 1,
        rank: (i / 9) as i8 + 1,
    }
}

/// 移動の差分 → 種類（成りを含まない 0..66）
fn move_kind(df: i8, dr: i8) -> Option<usize> {
    if let Some(k) = KNIGHT_DIRS.iter().position(|&d| d == (df, dr)) {
        return Some(64 + k);
    }
    let dist = df.abs().max(dr.abs());
    if dist == 0 || dist > 8 {
        return None;
    }
    // 直線（縦・横・斜め）でなければ符号化できない
    if df != 0 && dr != 0 && df.abs() != dr.abs() {
        return None;
    }
    let unit = (df.signum(), dr.signum());
    let d = DIRS.iter().position(|&x| x == unit)?;
    Some(d * 8 + (dist as usize - 1))
}

/// 指し手 → 行動番号。盤外・符号化できない差分は None
pub fn encode_move(mv: &ShogiMove, color: Color) -> Option<usize> {
    match *mv {
        ShogiMove::Board { from, to, promote } => {
            let (f, t) = (normalize(from, color), normalize(to, color));
            let kind = move_kind(t.file - f.file, t.rank - f.rank)?;
            let kind = if promote { kind + MOVE_KINDS } else { kind };
            Some(kind * NUM_SQUARES + square_index(f))
        }
        ShogiMove::Drop { role, to } => {
            let kind = DROP_BASE + hand_index(role)?;
            Some(kind * NUM_SQUARES + square_index(normalize(to, color)))
        }
    }
}

pub fn encode_usi(usi: &str, color: Color) -> Option<usize> {
    encode_move(&parse_usi(usi)?, color)
}

/// 行動番号 → 指し手。盤外へ出る番号は None（マスクの外なので通常は起きない）
pub fn decode_action(action: usize, color: Color) -> Option<ShogiMove> {
    if action >= NUM_ACTIONS {
        return None;
    }
    let (kind, sq) = (action / NUM_SQUARES, action % NUM_SQUARES);
    let at = index_square(sq);
    if kind >= DROP_BASE {
        return Some(ShogiMove::Drop {
            role: HAND_ROLES[kind - DROP_BASE],
            to: normalize(at, color),
        });
    }
    let promote = kind >= MOVE_KINDS;
    let base = kind % MOVE_KINDS;
    let (df, dr) = if base >= 64 {
        KNIGHT_DIRS[base - 64]
    } else {
        let (ux, uy) = DIRS[base / 8];
        let dist = (base % 8) as i8 + 1;
        (ux * dist, uy * dist)
    };
    let to = Coord {
        file: at.file + df,
        rank: at.rank + dr,
    };
    if !(1..=9).contains(&to.file) || !(1..=9).contains(&to.rank) {
        return None;
    }
    Some(ShogiMove::Board {
        from: normalize(at, color),
        to: normalize(to, color),
        promote,
    })
}

pub fn decode_usi(action: usize, color: Color) -> Option<String> {
    Some(decode_action(action, color)?.to_usi())
}

/// 自駒だけを見た候補手（USI）。成りが任意なら成・不成の両方、強制なら成りだけ
pub fn candidate_usis(view: &PlayerView) -> Vec<String> {
    let color = view.your_color;
    let mut out = vec![];
    for piece in &view.your_pieces {
        let Some(from) = parse_usi_square(&piece.square) else {
            continue;
        };
        for to in move_targets(&view.your_pieces, piece, color) {
            match promotion_choice(piece.role, from, to, color) {
                Promotion::None => out.push(make_usi_move(from, to, false)),
                Promotion::Optional => {
                    out.push(make_usi_move(from, to, false));
                    out.push(make_usi_move(from, to, true));
                }
                Promotion::Forced => out.push(make_usi_move(from, to, true)),
            }
        }
    }
    for role in HAND_ROLES {
        if view.your_hand.get(&role).copied().unwrap_or(0) == 0 {
            continue;
        }
        for to in drop_targets(&view.your_pieces, role, color) {
            out.extend(make_usi_drop(role, to));
        }
    }
    out
}

/// 行動マスク（長さ `NUM_ACTIONS`）。その手番で反則済みの手は除く
pub fn legal_mask(view: &PlayerView, foul_tried: &HashSet<String>) -> Vec<bool> {
    let mut mask = vec![false; NUM_ACTIONS];
    for usi in candidate_usis(view) {
        if foul_tried.contains(&usi) {
            continue;
        }
        if let Some(a) = encode_usi(&usi, view.your_color) {
            mask[a] = true;
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::referee::{Referee, StepResult};
    use rand::{Rng, SeedableRng, rngs::StdRng};

    #[test]
    fn 行動数は139かける81() {
        assert_eq!(NUM_ACTION_KINDS, 139);
        assert_eq!(NUM_ACTIONS, 11_259);
    }

    #[test]
    fn 全行動の往復が一致する() {
        for color in [Color::Sente, Color::Gote] {
            let mut decodable = 0;
            for a in 0..NUM_ACTIONS {
                if let Some(mv) = decode_action(a, color) {
                    decodable += 1;
                    assert_eq!(encode_move(&mv, color), Some(a), "{a} {mv:?}");
                }
            }
            assert!(decodable > 5000);
        }
    }

    #[test]
    fn 先後で同じ形の手は同じ行動になる() {
        // 先手 7六歩 と 後手 3四歩 は正規化すると同じ「前へ1マス」
        assert_eq!(
            encode_usi("7g7f", Color::Sente),
            encode_usi("3c3d", Color::Gote)
        );
        assert_eq!(
            encode_usi("P*5e", Color::Sente),
            encode_usi("P*5e", Color::Gote)
        );
        assert_ne!(encode_usi("2h2c", Color::Sente), encode_usi("2h2c+", Color::Sente));
    }

    /// ランダム対局の全局面で「真の合法手 ⊆ マスク」かつ「反則済みの手はマスク外」
    #[test]
    fn 真の合法手はすべてマスクに含まれる() {
        let mut rng = StdRng::seed_from_u64(20260927);
        let mut checked = 0usize;
        for game in 0..40 {
            let mut referee = Referee::new();
            loop {
                let side = referee.to_move();
                let view = referee.view(side, [0, 0], game);
                let mask = legal_mask(&view, referee.foul_tried(side));
                for mv in referee.position().legal_moves() {
                    let a = encode_move(&mv, side).expect("合法手は符号化できる");
                    assert!(mask[a], "合法手 {} がマスクに無い", mv.to_usi());
                    checked += 1;
                }
                for usi in referee.foul_tried(side) {
                    assert!(!mask[encode_usi(usi, side).unwrap()], "反則済み {usi}");
                }
                // マスク内から一様に選ぶ（反則も起きる = foul_tried の経路も通る）
                let actions: Vec<usize> = (0..NUM_ACTIONS).filter(|&a| mask[a]).collect();
                if actions.is_empty() {
                    break;
                }
                let a = actions[rng.random_range(0..actions.len())];
                let usi = decode_usi(a, side).unwrap();
                if let StepResult::Ended { .. } = referee.step(&usi, 0) {
                    break;
                }
            }
        }
        assert!(checked > 10_000, "検査した合法手が少なすぎる: {checked}");
    }
}
