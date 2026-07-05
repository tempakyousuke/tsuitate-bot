//! 盤座標と「自分の駒だけを考慮した」候補手生成。
//! tsuitate リポジトリの src/lib/shared/coords.ts / move-hints.ts の移植。
//!
//! ついたて将棋では相手の駒が見えないため、ここで出す候補は
//! 「相手の駒がいなければ指せる手」にすぎない。実際の合法性はサーバーが判定する。

use std::collections::HashSet;

use crate::protocol::{Color, Role, VisiblePiece};

/// file 1〜9（右から）、rank 1〜9（上から。1='a'）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Coord {
    pub file: i8,
    pub rank: i8,
}

const RANK_LETTERS: &[u8] = b"abcdefghi";

pub fn parse_usi_square(sq: &str) -> Option<Coord> {
    let bytes = sq.as_bytes();
    if bytes.len() != 2 {
        return None;
    }
    let file = (bytes[0] as char).to_digit(10)? as i8;
    let rank = RANK_LETTERS.iter().position(|&c| c == bytes[1])? as i8 + 1;
    if (1..=9).contains(&file) {
        Some(Coord { file, rank })
    } else {
        None
    }
}

pub fn make_usi_square(c: Coord) -> String {
    format!("{}{}", c.file, RANK_LETTERS[c.rank as usize - 1] as char)
}

pub fn make_usi_move(from: Coord, to: Coord, promote: bool) -> String {
    format!(
        "{}{}{}",
        make_usi_square(from),
        make_usi_square(to),
        if promote { "+" } else { "" }
    )
}

/// 持ち駒として打てる駒の USI 文字
fn drop_letter(role: Role) -> Option<char> {
    match role {
        Role::Pawn => Some('P'),
        Role::Lance => Some('L'),
        Role::Knight => Some('N'),
        Role::Silver => Some('S'),
        Role::Gold => Some('G'),
        Role::Bishop => Some('B'),
        Role::Rook => Some('R'),
        _ => None,
    }
}

pub fn make_usi_drop(role: Role, to: Coord) -> Option<String> {
    Some(format!("{}*{}", drop_letter(role)?, make_usi_square(to)))
}

/// [dFile, dRank] 先手視点（rank減少=前進）
pub(crate) type Delta = (i8, i8);

const GOLD_STEPS: &[Delta] = &[(0, -1), (1, -1), (-1, -1), (1, 0), (-1, 0), (0, 1)];
const KING_STEPS: &[Delta] = &[
    (0, -1),
    (1, -1),
    (-1, -1),
    (1, 0),
    (-1, 0),
    (0, 1),
    (1, 1),
    (-1, 1),
];
const DIAGONALS: &[Delta] = &[(1, -1), (-1, -1), (1, 1), (-1, 1)];
const ORTHOGONALS: &[Delta] = &[(0, -1), (0, 1), (1, 0), (-1, 0)];

pub(crate) fn steps(role: Role) -> &'static [Delta] {
    match role {
        Role::Pawn => &[(0, -1)],
        Role::Knight => &[(1, -2), (-1, -2)],
        Role::Silver => &[(0, -1), (1, -1), (-1, -1), (1, 1), (-1, 1)],
        Role::Gold
        | Role::Tokin
        | Role::Promotedlance
        | Role::Promotedknight
        | Role::Promotedsilver => GOLD_STEPS,
        Role::King => KING_STEPS,
        // 馬・龍の「1マス」部分
        Role::Horse => ORTHOGONALS,
        Role::Dragon => DIAGONALS,
        _ => &[],
    }
}

pub(crate) fn rays(role: Role) -> &'static [Delta] {
    match role {
        Role::Lance => &[(0, -1)],
        Role::Bishop | Role::Horse => DIAGONALS,
        Role::Rook | Role::Dragon => ORTHOGONALS,
        _ => &[],
    }
}

pub(crate) fn on_board(c: Coord) -> bool {
    (1..=9).contains(&c.file) && (1..=9).contains(&c.rank)
}

/// 先手視点のデルタを自分の色に合わせる（後手は前後反転）
pub(crate) fn orient((df, dr): Delta, color: Color) -> Delta {
    match color {
        Color::Sente => (df, dr),
        Color::Gote => (-df, -dr),
    }
}

fn occupied_set(pieces: &[VisiblePiece]) -> HashSet<Coord> {
    pieces
        .iter()
        .filter_map(|p| parse_usi_square(&p.square))
        .collect()
}

/// 盤上の自駒の移動候補（自駒にだけ塞がれる。相手駒は考慮しない）
pub fn move_targets(pieces: &[VisiblePiece], piece: &VisiblePiece, color: Color) -> Vec<Coord> {
    let own = occupied_set(pieces);
    let Some(from) = parse_usi_square(&piece.square) else {
        return vec![];
    };
    let mut targets = vec![];

    let push = |c: Coord, targets: &mut Vec<Coord>| -> bool {
        if !on_board(c) || own.contains(&c) {
            return false; // 自駒のいるマスには行けない（レイもここで止まる）
        }
        targets.push(c);
        true
    };

    for &delta in steps(piece.role) {
        let (df, dr) = orient(delta, color);
        push(
            Coord {
                file: from.file + df,
                rank: from.rank + dr,
            },
            &mut targets,
        );
    }
    for &delta in rays(piece.role) {
        let (df, dr) = orient(delta, color);
        let mut c = Coord {
            file: from.file + df,
            rank: from.rank + dr,
        };
        while push(c, &mut targets) {
            c = Coord {
                file: c.file + df,
                rank: c.rank + dr,
            };
        }
    }
    targets
}

/// 持ち駒を打てる候補（自駒のないマス。歩は自歩の筋と行き所を除外）
pub fn drop_targets(pieces: &[VisiblePiece], role: Role, color: Color) -> Vec<Coord> {
    let own = occupied_set(pieces);
    let own_pawn_files: HashSet<i8> = pieces
        .iter()
        .filter(|p| p.role == Role::Pawn)
        .filter_map(|p| parse_usi_square(&p.square).map(|c| c.file))
        .collect();
    let mut targets = vec![];
    for file in 1..=9 {
        for rank in 1..=9 {
            let c = Coord { file, rank };
            if own.contains(&c) {
                continue;
            }
            if role == Role::Pawn && own_pawn_files.contains(&file) {
                continue; // 二歩（見えている範囲）
            }
            if dead_end_rank(role, rank, color) {
                continue;
            }
            targets.push(c);
        }
    }
    targets
}

/// 行き所のない駒になる段か（移動先・打ち先の共通判定）
pub fn dead_end_rank(role: Role, rank: i8, color: Color) -> bool {
    let from_top = match color {
        Color::Sente => rank,
        Color::Gote => 10 - rank,
    };
    match role {
        Role::Pawn | Role::Lance => from_top <= 1,
        Role::Knight => from_top <= 2,
        _ => false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Promotion {
    None,
    Optional,
    Forced,
}

fn promotable(role: Role) -> bool {
    matches!(
        role,
        Role::Pawn | Role::Lance | Role::Knight | Role::Silver | Role::Bishop | Role::Rook
    )
}

/// 敵陣3段か
fn in_promotion_zone(rank: i8, color: Color) -> bool {
    match color {
        Color::Sente => rank <= 3,
        Color::Gote => rank >= 7,
    }
}

/// この移動の成りの扱い
pub fn promotion_choice(role: Role, from: Coord, to: Coord, color: Color) -> Promotion {
    if !promotable(role) {
        return Promotion::None;
    }
    if !in_promotion_zone(from.rank, color) && !in_promotion_zone(to.rank, color) {
        return Promotion::None;
    }
    if dead_end_rank(role, to.rank, color) {
        return Promotion::Forced;
    }
    Promotion::Optional
}

#[cfg(test)]
mod tests {
    use super::*;

    fn piece(square: &str, role: Role) -> VisiblePiece {
        VisiblePiece {
            square: square.to_string(),
            role,
        }
    }

    #[test]
    fn usi_square_roundtrip() {
        let c = parse_usi_square("7g").unwrap();
        assert_eq!(c, Coord { file: 7, rank: 7 });
        assert_eq!(make_usi_square(c), "7g");
        assert_eq!(parse_usi_square("0a"), None);
        assert_eq!(parse_usi_square("1j"), None);
    }

    #[test]
    fn usi_move_and_drop() {
        let from = parse_usi_square("8h").unwrap();
        let to = parse_usi_square("2b").unwrap();
        assert_eq!(make_usi_move(from, to, true), "8h2b+");
        assert_eq!(
            make_usi_drop(Role::Pawn, parse_usi_square("5e").unwrap()),
            Some("P*5e".to_string())
        );
        assert_eq!(make_usi_drop(Role::Tokin, parse_usi_square("5e").unwrap()), None);
    }

    #[test]
    fn pawn_moves_forward_only() {
        let pieces = vec![piece("7g", Role::Pawn)];
        let targets = move_targets(&pieces, &pieces[0], Color::Sente);
        assert_eq!(targets, vec![Coord { file: 7, rank: 6 }]);
        // 後手は逆方向
        let targets = move_targets(&pieces, &pieces[0], Color::Gote);
        assert_eq!(targets, vec![Coord { file: 7, rank: 8 }]);
    }

    #[test]
    fn own_piece_blocks_step_and_ray() {
        let pieces = vec![piece("7g", Role::Pawn), piece("7f", Role::Silver)];
        // 歩の前に自分の銀 → 歩は動けない
        assert!(move_targets(&pieces, &pieces[0], Color::Sente).is_empty());

        let pieces = vec![piece("1i", Role::Lance), piece("1f", Role::Pawn)];
        // 香のレイは自駒の手前で止まる
        let targets = move_targets(&pieces, &pieces[0], Color::Sente);
        assert_eq!(
            targets,
            vec![Coord { file: 1, rank: 8 }, Coord { file: 1, rank: 7 }]
        );
    }

    #[test]
    fn knight_jumps() {
        let pieces = vec![piece("8i", Role::Knight), piece("8h", Role::Rook)];
        let targets = move_targets(&pieces, &pieces[0], Color::Sente);
        assert_eq!(
            targets,
            vec![Coord { file: 9, rank: 7 }, Coord { file: 7, rank: 7 }]
        );
    }

    #[test]
    fn horse_combines_step_and_ray() {
        let pieces = vec![piece("5e", Role::Horse)];
        let targets = move_targets(&pieces, &pieces[0], Color::Sente);
        // 直交4ステップ + 斜め4レイ（5eから各方向4マス） = 4 + 16
        assert_eq!(targets.len(), 20);
    }

    #[test]
    fn pawn_drop_respects_nifu_and_dead_end() {
        let pieces = vec![piece("7g", Role::Pawn)];
        let targets = drop_targets(&pieces, Role::Pawn, Color::Sente);
        // 7筋（二歩）と1段目（行き所なし）が除外され、自駒マスもない
        assert!(targets.iter().all(|c| c.file != 7));
        assert!(targets.iter().all(|c| c.rank != 1));
        // 9筋 × (9-1)段 - 7筋分(8) = 64
        assert_eq!(targets.len(), 8 * 9 - 8);
    }

    #[test]
    fn knight_drop_dead_end() {
        let targets = drop_targets(&[], Role::Knight, Color::Sente);
        assert!(targets.iter().all(|c| c.rank >= 3));
        let targets = drop_targets(&[], Role::Knight, Color::Gote);
        assert!(targets.iter().all(|c| c.rank <= 7));
    }

    #[test]
    fn promotion_rules() {
        let from = parse_usi_square("2c").unwrap();
        let to = parse_usi_square("2b").unwrap();
        assert_eq!(
            promotion_choice(Role::Silver, from, to, Color::Sente),
            Promotion::Optional
        );
        // 歩が1段目へ → 強制成り
        let to1 = parse_usi_square("2a").unwrap();
        let from2 = parse_usi_square("2b").unwrap();
        assert_eq!(
            promotion_choice(Role::Pawn, from2, to1, Color::Sente),
            Promotion::Forced
        );
        // 敵陣に無関係な移動は成れない
        let f = parse_usi_square("5g").unwrap();
        let t = parse_usi_square("5f").unwrap();
        assert_eq!(promotion_choice(Role::Rook, f, t, Color::Sente), Promotion::None);
        // 金・玉・成駒は成れない
        assert_eq!(promotion_choice(Role::Gold, from, to, Color::Sente), Promotion::None);
    }
}
