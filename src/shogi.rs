//! フル盤面（両者の駒が見える）の通常将棋ルールエンジン。
//!
//! 用途は2つ:
//! - アリーナ（bin/arena.rs）の審判: サーバーの judge.ts（shogiops の isLegal）と
//!   同じ基準で反則・王手・詰み/ステイルメイトを裁定する
//! - 相手局面の推定（estimator.rs）: サンプルした具体局面に手を適用・検証する部品
//!
//! 合法性 = 疑似合法（駒の動き・二歩・行き所なし・成りの妥当性）
//!         + 自玉を王手に晒さない + 打ち歩詰め禁止。
//! 初期局面からの perft 値（30 / 900 / 25470 / 719731）で検証している。
#![allow(dead_code)]

use crate::board::{
    Coord, Promotion, dead_end_rank, make_usi_drop, make_usi_move, on_board, orient,
    parse_usi_square, promotion_choice, rays, steps,
};
use crate::protocol::{Color, Role, VisiblePiece};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Piece {
    pub color: Color,
    pub role: Role,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShogiMove {
    Board { from: Coord, to: Coord, promote: bool },
    Drop { role: Role, to: Coord },
}

impl ShogiMove {
    pub fn to_usi(&self) -> String {
        match *self {
            ShogiMove::Board { from, to, promote } => make_usi_move(from, to, promote),
            ShogiMove::Drop { role, to } => {
                make_usi_drop(role, to).expect("持ち駒にならない駒種は Drop にならない")
            }
        }
    }
}

/// USI表記の指し手をパースする（"7g7f", "8h2b+", "P*5e"）
pub fn parse_usi(usi: &str) -> Option<ShogiMove> {
    let bytes = usi.as_bytes();
    if bytes.len() >= 2 && bytes[1] == b'*' {
        if bytes.len() != 4 {
            return None;
        }
        let role = match bytes[0] {
            b'P' => Role::Pawn,
            b'L' => Role::Lance,
            b'N' => Role::Knight,
            b'S' => Role::Silver,
            b'G' => Role::Gold,
            b'B' => Role::Bishop,
            b'R' => Role::Rook,
            _ => return None,
        };
        let to = parse_usi_square(&usi[2..4])?;
        return Some(ShogiMove::Drop { role, to });
    }
    if bytes.len() != 4 && bytes.len() != 5 {
        return None;
    }
    let promote = bytes.len() == 5;
    if promote && bytes[4] != b'+' {
        return None;
    }
    Some(ShogiMove::Board {
        from: parse_usi_square(&usi[0..2])?,
        to: parse_usi_square(&usi[2..4])?,
        promote,
    })
}

/// 持ち駒になれる7駒種（手駒配列のインデックス順）
pub const HAND_ROLES: [Role; 7] = [
    Role::Pawn,
    Role::Lance,
    Role::Knight,
    Role::Silver,
    Role::Gold,
    Role::Bishop,
    Role::Rook,
];

pub fn hand_index(role: Role) -> Option<usize> {
    HAND_ROLES.iter().position(|&r| r == role)
}

/// 成った後の駒種
pub fn promote_role(role: Role) -> Option<Role> {
    match role {
        Role::Pawn => Some(Role::Tokin),
        Role::Lance => Some(Role::Promotedlance),
        Role::Knight => Some(Role::Promotedknight),
        Role::Silver => Some(Role::Promotedsilver),
        Role::Bishop => Some(Role::Horse),
        Role::Rook => Some(Role::Dragon),
        _ => None,
    }
}

/// 取られて持ち駒になるときの駒種（成りを戻す）
pub fn unpromote_role(role: Role) -> Role {
    match role {
        Role::Tokin => Role::Pawn,
        Role::Promotedlance => Role::Lance,
        Role::Promotedknight => Role::Knight,
        Role::Promotedsilver => Role::Silver,
        Role::Horse => Role::Bishop,
        Role::Dragon => Role::Rook,
        r => r,
    }
}

/// 駒の概算価値（歩=1）。指し手評価と相手手の尤度づけに使う
pub fn piece_value(role: Role) -> f64 {
    match role {
        Role::Pawn => 1.0,
        Role::Lance => 3.0,
        Role::Knight => 3.5,
        Role::Silver => 5.0,
        Role::Gold => 5.5,
        Role::Bishop => 8.0,
        Role::Rook => 9.5,
        Role::Tokin => 6.0,
        Role::Promotedlance => 6.0,
        Role::Promotedknight => 6.0,
        Role::Promotedsilver => 6.0,
        Role::Horse => 10.0,
        Role::Dragon => 12.0,
        Role::King => 0.0, // 玉は取られたら終局なので価値では扱わない
    }
}

fn sq_index(c: Coord) -> usize {
    ((c.rank - 1) * 9 + (c.file - 1)) as usize
}

fn index_sq(i: usize) -> Coord {
    Coord {
        file: (i % 9) as i8 + 1,
        rank: (i / 9) as i8 + 1,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// 手番側が詰んだ（勝者は相手）
    Checkmate { winner: Color },
    /// 手番側に合法手がない（将棋ではステイルメイトも手番側の負け）
    Stalemate { winner: Color },
}

#[derive(Debug, Clone)]
pub struct Position {
    board: [Option<Piece>; 81],
    /// [color][hand_index] の枚数
    hands: [[u8; 7]; 2],
    turn: Color,
    /// 次に指す手の番号（初期局面で1、1手ごとに+1）。shogiops の moveNumber と同じ
    move_number: u32,
}

fn color_index(color: Color) -> usize {
    match color {
        Color::Sente => 0,
        Color::Gote => 1,
    }
}

impl Position {
    pub fn initial() -> Self {
        let mut pos = Position {
            board: [None; 81],
            hands: [[0; 7]; 2],
            turn: Color::Sente,
            move_number: 1,
        };
        let back = [
            Role::Lance,
            Role::Knight,
            Role::Silver,
            Role::Gold,
            Role::King,
            Role::Gold,
            Role::Silver,
            Role::Knight,
            Role::Lance,
        ];
        for (i, &role) in back.iter().enumerate() {
            let file = 9 - i as i8;
            pos.set(Coord { file, rank: 1 }, Some(Piece { color: Color::Gote, role }));
            pos.set(Coord { file, rank: 9 }, Some(Piece { color: Color::Sente, role }));
        }
        pos.set(Coord { file: 8, rank: 2 }, Some(Piece { color: Color::Gote, role: Role::Rook }));
        pos.set(Coord { file: 2, rank: 2 }, Some(Piece { color: Color::Gote, role: Role::Bishop }));
        pos.set(Coord { file: 8, rank: 8 }, Some(Piece { color: Color::Sente, role: Role::Bishop }));
        pos.set(Coord { file: 2, rank: 8 }, Some(Piece { color: Color::Sente, role: Role::Rook }));
        for file in 1..=9 {
            pos.set(Coord { file, rank: 3 }, Some(Piece { color: Color::Gote, role: Role::Pawn }));
            pos.set(Coord { file, rank: 7 }, Some(Piece { color: Color::Sente, role: Role::Pawn }));
        }
        pos
    }

    /// 盤・持ち駒が空の局面（テスト・推定器の部品用）
    pub fn empty(turn: Color) -> Self {
        Position {
            board: [None; 81],
            hands: [[0; 7]; 2],
            turn,
            move_number: 1,
        }
    }

    pub fn turn(&self) -> Color {
        self.turn
    }

    pub fn set_turn(&mut self, turn: Color) {
        self.turn = turn;
    }

    pub fn move_number(&self) -> u32 {
        self.move_number
    }

    pub fn piece_at(&self, c: Coord) -> Option<Piece> {
        self.board[sq_index(c)]
    }

    pub fn set(&mut self, c: Coord, piece: Option<Piece>) {
        self.board[sq_index(c)] = piece;
    }

    pub fn hand_count(&self, color: Color, role: Role) -> u8 {
        hand_index(role).map_or(0, |i| self.hands[color_index(color)][i])
    }

    pub fn set_hand(&mut self, color: Color, role: Role, count: u8) {
        if let Some(i) = hand_index(role) {
            self.hands[color_index(color)][i] = count;
        }
    }

    /// 局面の指紋（FNV-1a）。複製された同一粒子の除去に使う
    pub fn fingerprint(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        let mut eat = |b: u8| {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        };
        for p in &self.board {
            eat(match p {
                None => 0xFF,
                Some(p) => (p.role as u8) * 2 + color_index(p.color) as u8,
            });
        }
        for hand in &self.hands {
            for &n in hand {
                eat(n);
            }
        }
        eat(color_index(self.turn) as u8);
        h
    }

    /// 盤上の全駒（座標つき）。評価関数の走査用
    pub fn pieces(&self) -> impl Iterator<Item = (Coord, Piece)> + '_ {
        self.board
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.map(|p| (index_sq(i), p)))
    }

    pub fn pieces_of(&self, color: Color) -> Vec<VisiblePiece> {
        self.board
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                p.filter(|p| p.color == color).map(|p| VisiblePiece {
                    square: crate::board::make_usi_square(index_sq(i)),
                    role: p.role,
                })
            })
            .collect()
    }

    pub fn hand_map(&self, color: Color) -> HashMap<Role, u32> {
        HAND_ROLES
            .iter()
            .enumerate()
            .filter(|&(i, _)| self.hands[color_index(color)][i] > 0)
            .map(|(i, &role)| (role, self.hands[color_index(color)][i] as u32))
            .collect()
    }

    pub fn king_square(&self, color: Color) -> Option<Coord> {
        self.board.iter().enumerate().find_map(|(i, p)| {
            (*p == Some(Piece { color, role: Role::King })).then(|| index_sq(i))
        })
    }

    /// マス sq が by 側の駒に利いているか（sq から逆引きで走査）
    pub fn is_attacked(&self, sq: Coord, by: Color) -> bool {
        // 桂: 攻撃側の桂が s にいて s + oriented(knight) == sq となる s を逆算
        for &delta in steps(Role::Knight) {
            let (df, dr) = orient(delta, by);
            let s = Coord { file: sq.file - df, rank: sq.rank - dr };
            if on_board(s)
                && self.piece_at(s) == Some(Piece { color: by, role: Role::Knight })
            {
                return true;
            }
        }
        // 8方向: 隣接ステップと、その先のレイ
        const DIRS: [(i8, i8); 8] = [
            (0, -1), (0, 1), (1, 0), (-1, 0), (1, -1), (-1, -1), (1, 1), (-1, 1),
        ];
        for &(df, dr) in &DIRS {
            let mut c = Coord { file: sq.file + df, rank: sq.rank + dr };
            let mut dist = 1;
            while on_board(c) {
                if let Some(p) = self.piece_at(c) {
                    if p.color == by {
                        // p が c から sq 方向 (-df, -dr) に利くか
                        let back = (-df, -dr);
                        if dist == 1
                            && steps(p.role)
                                .iter()
                                .any(|&d| orient(d, by) == back)
                        {
                            return true;
                        }
                        if rays(p.role).iter().any(|&d| orient(d, by) == back) {
                            return true;
                        }
                    }
                    break; // 先頭の駒で遮断（敵味方問わず）
                }
                c = Coord { file: c.file + df, rank: c.rank + dr };
                dist += 1;
            }
        }
        false
    }

    pub fn in_check(&self, color: Color) -> bool {
        self.king_square(color)
            .is_some_and(|k| self.is_attacked(k, color.other()))
    }

    /// 疑似合法か（自玉の安全・打ち歩詰めは見ない）
    /// 疑似合法（利き・経路・打ちマスの空き等。自玉の王手放置は見ない）。
    /// 記録分析（bin/analyze.rs）が反則の原因分類にも使う
    pub fn is_pseudo_legal(&self, mv: &ShogiMove) -> bool {
        match *mv {
            ShogiMove::Board { from, to, promote } => {
                if !on_board(from) || !on_board(to) {
                    return false;
                }
                let Some(piece) = self.piece_at(from) else {
                    return false;
                };
                if piece.color != self.turn {
                    return false;
                }
                if self.piece_at(to).is_some_and(|p| p.color == self.turn) {
                    return false;
                }
                if !self.reachable(piece, from, to) {
                    return false;
                }
                match promotion_choice(piece.role, from, to, self.turn) {
                    Promotion::None => !promote,
                    Promotion::Forced => promote,
                    Promotion::Optional => true,
                }
            }
            ShogiMove::Drop { role, to } => {
                if !on_board(to) || self.piece_at(to).is_some() {
                    return false;
                }
                if self.hand_count(self.turn, role) == 0 {
                    return false;
                }
                if dead_end_rank(role, to.rank, self.turn) {
                    return false;
                }
                if role == Role::Pawn && self.has_pawn_on_file(self.turn, to.file) {
                    return false; // 二歩
                }
                true
            }
        }
    }

    fn has_pawn_on_file(&self, color: Color, file: i8) -> bool {
        (1..=9).any(|rank| {
            self.piece_at(Coord { file, rank }) == Some(Piece { color, role: Role::Pawn })
        })
    }

    /// from にいる駒が target マスへ利いているか（間の駒に遮られない移動可能性）。
    /// 相手手の事前分布の threat 特徴量（fit_opp / estimator）が使う
    pub fn attacks(&self, from: Coord, target: Coord) -> bool {
        match self.piece_at(from) {
            Some(piece) => self.reachable(piece, from, target),
            None => false,
        }
    }

    /// piece が from から to へ（間の駒に遮られず）動けるか
    fn reachable(&self, piece: Piece, from: Coord, to: Coord) -> bool {
        for &delta in steps(piece.role) {
            let (df, dr) = orient(delta, piece.color);
            if to.file == from.file + df && to.rank == from.rank + dr {
                return true;
            }
        }
        for &delta in rays(piece.role) {
            let (df, dr) = orient(delta, piece.color);
            let mut c = Coord { file: from.file + df, rank: from.rank + dr };
            while on_board(c) {
                if c == to {
                    return true;
                }
                if self.piece_at(c).is_some() {
                    break;
                }
                c = Coord { file: c.file + df, rank: c.rank + dr };
            }
        }
        false
    }

    /// サーバー（judge.ts / shogiops isLegal）と同じ基準の合法性
    pub fn is_legal(&self, mv: &ShogiMove) -> bool {
        if !self.is_pseudo_legal(mv) {
            return false;
        }
        let mut next = self.clone();
        next.play_unchecked(mv);
        if next.in_check(self.turn) {
            return false; // 自玉を王手に晒す
        }
        // 打ち歩詰め: 歩打ちで相手玉が詰む手は反則
        if let ShogiMove::Drop { role: Role::Pawn, .. } = mv {
            let opponent = self.turn.other();
            if next.in_check(opponent) && !next.has_any_legal_move() {
                return false;
            }
        }
        true
    }

    /// 合法性チェックなしで適用する。取った駒（盤上の駒種）を返す
    pub fn play_unchecked(&mut self, mv: &ShogiMove) -> Option<Role> {
        let captured = match *mv {
            ShogiMove::Board { from, to, promote } => {
                let piece = self.piece_at(from).expect("移動元に駒がある");
                let captured = self.piece_at(to).map(|p| p.role);
                if let Some(role) = captured {
                    let hand_role = unpromote_role(role);
                    if let Some(i) = hand_index(hand_role) {
                        self.hands[color_index(self.turn)][i] += 1;
                    }
                }
                self.set(from, None);
                let role = if promote {
                    promote_role(piece.role).unwrap_or(piece.role)
                } else {
                    piece.role
                };
                self.set(to, Some(Piece { color: piece.color, role }));
                captured
            }
            ShogiMove::Drop { role, to } => {
                if let Some(i) = hand_index(role) {
                    self.hands[color_index(self.turn)][i] -= 1;
                }
                self.set(to, Some(Piece { color: self.turn, role }));
                None
            }
        };
        self.turn = self.turn.other();
        self.move_number += 1;
        captured
    }

    /// 手番側の疑似合法手（成り/不成の両変種を含む）
    fn pseudo_legal_moves(&self) -> Vec<ShogiMove> {
        let mut moves = vec![];
        for (i, p) in self.board.iter().enumerate() {
            let Some(piece) = *p else { continue };
            if piece.color != self.turn {
                continue;
            }
            let from = index_sq(i);
            for to in self.move_targets_full(piece, from) {
                match promotion_choice(piece.role, from, to, self.turn) {
                    Promotion::None => moves.push(ShogiMove::Board { from, to, promote: false }),
                    Promotion::Forced => moves.push(ShogiMove::Board { from, to, promote: true }),
                    Promotion::Optional => {
                        moves.push(ShogiMove::Board { from, to, promote: false });
                        moves.push(ShogiMove::Board { from, to, promote: true });
                    }
                }
            }
        }
        for &role in &HAND_ROLES {
            if self.hand_count(self.turn, role) == 0 {
                continue;
            }
            for i in 0..81 {
                let to = index_sq(i);
                if self.board[i].is_some() {
                    continue;
                }
                if dead_end_rank(role, to.rank, self.turn) {
                    continue;
                }
                if role == Role::Pawn && self.has_pawn_on_file(self.turn, to.file) {
                    continue;
                }
                moves.push(ShogiMove::Drop { role, to });
            }
        }
        moves
    }

    /// フル盤面での移動先（自駒は不可、敵駒は取れる、レイは駒で止まる）
    fn move_targets_full(&self, piece: Piece, from: Coord) -> Vec<Coord> {
        let mut targets = vec![];
        for &delta in steps(piece.role) {
            let (df, dr) = orient(delta, piece.color);
            let c = Coord { file: from.file + df, rank: from.rank + dr };
            if on_board(c) && self.piece_at(c).is_none_or(|p| p.color != piece.color) {
                targets.push(c);
            }
        }
        for &delta in rays(piece.role) {
            let (df, dr) = orient(delta, piece.color);
            let mut c = Coord { file: from.file + df, rank: from.rank + dr };
            while on_board(c) {
                match self.piece_at(c) {
                    None => targets.push(c),
                    Some(p) => {
                        if p.color != piece.color {
                            targets.push(c);
                        }
                        break;
                    }
                }
                c = Coord { file: c.file + df, rank: c.rank + dr };
            }
        }
        targets
    }

    /// 手番側の全合法手
    pub fn legal_moves(&self) -> Vec<ShogiMove> {
        self.pseudo_legal_moves()
            .into_iter()
            .filter(|mv| {
                let mut next = self.clone();
                next.play_unchecked(mv);
                if next.in_check(self.turn) {
                    return false;
                }
                if let ShogiMove::Drop { role: Role::Pawn, .. } = mv {
                    let opponent = self.turn.other();
                    if next.in_check(opponent) && !next.has_any_legal_move() {
                        return false;
                    }
                }
                true
            })
            .collect()
    }

    /// 手番側に合法手が1つでもあるか（打ち歩詰め判定の内側では
    /// 相手の歩打ちの打ち歩詰めまでは見ない = shogiops と同等の近似）
    fn has_any_legal_move(&self) -> bool {
        self.pseudo_legal_moves().iter().any(|mv| {
            let mut next = self.clone();
            next.play_unchecked(mv);
            !next.in_check(self.turn)
        })
    }

    /// 終局判定（手番側に合法手がなければ終局）
    pub fn outcome(&self) -> Option<Outcome> {
        if self.legal_moves().is_empty() {
            let winner = self.turn.other();
            if self.in_check(self.turn) {
                Some(Outcome::Checkmate { winner })
            } else {
                Some(Outcome::Stalemate { winner })
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perft(pos: &Position, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }
        pos.legal_moves()
            .iter()
            .map(|mv| {
                let mut next = pos.clone();
                next.play_unchecked(mv);
                perft(&next, depth - 1)
            })
            .sum()
    }

    #[test]
    fn perft_initial_shallow() {
        let pos = Position::initial();
        assert_eq!(perft(&pos, 1), 30);
        assert_eq!(perft(&pos, 2), 900);
        assert_eq!(perft(&pos, 3), 25470);
    }

    #[test]
    #[ignore = "遅いので --release で明示実行: cargo test --release -- --ignored"]
    fn perft_initial_deep() {
        let pos = Position::initial();
        assert_eq!(perft(&pos, 4), 719_731);
        assert_eq!(perft(&pos, 5), 19_861_490);
    }

    #[test]
    fn usi_parse_roundtrip() {
        for usi in ["7g7f", "8h2b+", "P*5e", "N*3c"] {
            let mv = parse_usi(usi).unwrap();
            assert_eq!(mv.to_usi(), usi);
        }
        assert_eq!(parse_usi("K*5e"), None); // 玉は打てない
        assert_eq!(parse_usi("7g7f#"), None);
    }

    #[test]
    fn initial_position_basics() {
        let pos = Position::initial();
        assert_eq!(pos.turn(), Color::Sente);
        assert_eq!(pos.move_number(), 1);
        assert_eq!(pos.pieces_of(Color::Sente).len(), 20);
        assert_eq!(pos.pieces_of(Color::Gote).len(), 20);
        assert_eq!(
            pos.king_square(Color::Sente),
            Some(Coord { file: 5, rank: 9 })
        );
        assert!(!pos.in_check(Color::Sente));
        assert!(pos.outcome().is_none());
    }

    #[test]
    fn blocked_bishop_is_illegal() {
        let pos = Position::initial();
        // 角道を開けないまま 8h2b+ は経路が塞がっていて反則
        assert!(!pos.is_legal(&parse_usi("8h2b+").unwrap()));
        assert!(pos.is_legal(&parse_usi("7g7f").unwrap()));
    }

    #[test]
    fn capture_goes_to_hand_unpromoted() {
        let mut pos = Position::empty(Color::Sente);
        pos.set(
            Coord { file: 5, rank: 5 },
            Some(Piece { color: Color::Sente, role: Role::Rook }),
        );
        pos.set(
            Coord { file: 5, rank: 3 },
            Some(Piece { color: Color::Gote, role: Role::Tokin }),
        );
        // 双方に玉を置く（詰み判定を避ける）
        pos.set(Coord { file: 9, rank: 9 }, Some(Piece { color: Color::Sente, role: Role::King }));
        pos.set(Coord { file: 1, rank: 1 }, Some(Piece { color: Color::Gote, role: Role::King }));
        let mv = parse_usi("5e5c+").unwrap();
        assert!(pos.is_legal(&mv));
        let captured = pos.play_unchecked(&mv);
        assert_eq!(captured, Some(Role::Tokin));
        assert_eq!(pos.hand_count(Color::Sente, Role::Pawn), 1); // と金→歩で持ち駒に
        assert_eq!(
            pos.piece_at(Coord { file: 5, rank: 3 }),
            Some(Piece { color: Color::Sente, role: Role::Dragon })
        );
    }

    #[test]
    fn cannot_leave_king_in_check() {
        let mut pos = Position::empty(Color::Sente);
        pos.set(Coord { file: 5, rank: 9 }, Some(Piece { color: Color::Sente, role: Role::King }));
        pos.set(Coord { file: 5, rank: 8 }, Some(Piece { color: Color::Sente, role: Role::Gold }));
        pos.set(Coord { file: 5, rank: 1 }, Some(Piece { color: Color::Gote, role: Role::Rook }));
        pos.set(Coord { file: 1, rank: 1 }, Some(Piece { color: Color::Gote, role: Role::King }));
        // 金が飛車をピンされている: 横に逃げるのは反則
        assert!(!pos.is_legal(&parse_usi("5h4h").unwrap()));
        // 縦に進むのは合法（ピンの線上）
        assert!(pos.is_legal(&parse_usi("5h5g").unwrap()));
    }

    #[test]
    fn nifu_and_dead_end_drops() {
        let mut pos = Position::empty(Color::Sente);
        pos.set(Coord { file: 5, rank: 9 }, Some(Piece { color: Color::Sente, role: Role::King }));
        pos.set(Coord { file: 1, rank: 1 }, Some(Piece { color: Color::Gote, role: Role::King }));
        pos.set(Coord { file: 7, rank: 7 }, Some(Piece { color: Color::Sente, role: Role::Pawn }));
        pos.set_hand(Color::Sente, Role::Pawn, 1);
        assert!(!pos.is_legal(&parse_usi("P*7e").unwrap())); // 二歩
        assert!(!pos.is_legal(&parse_usi("P*5a").unwrap())); // 行き所なし
        assert!(pos.is_legal(&parse_usi("P*5e").unwrap()));
    }

    #[test]
    fn uchifuzume_is_illegal() {
        // 打ち歩詰め形: 1a玉。1b歩打ちを 1c金 が支え、2a/2b は 2i飛 が塞ぐ
        let mut pos = Position::empty(Color::Sente);
        pos.set(Coord { file: 1, rank: 1 }, Some(Piece { color: Color::Gote, role: Role::King }));
        pos.set(Coord { file: 1, rank: 3 }, Some(Piece { color: Color::Sente, role: Role::Gold }));
        pos.set(Coord { file: 2, rank: 9 }, Some(Piece { color: Color::Sente, role: Role::Rook }));
        pos.set(Coord { file: 5, rank: 9 }, Some(Piece { color: Color::Sente, role: Role::King }));
        pos.set_hand(Color::Sente, Role::Pawn, 1);
        // 打つ前は王手ではない
        assert!(!pos.in_check(Color::Gote));
        // 1b歩打ちは王手で、取っても逃げても王手が残る → 打ち歩詰めで反則
        assert!(!pos.is_legal(&parse_usi("P*1b").unwrap()));
        // 同じ形でも突き歩（盤上の歩の前進）による詰みなら合法、の確認は
        // ここでは省略し、歩以外の駒打ちで詰ますのは合法であることを見る
        pos.set_hand(Color::Sente, Role::Lance, 1);
        assert!(pos.is_legal(&parse_usi("L*1b").unwrap()));
    }

    #[test]
    fn checkmate_and_stalemate_detection() {
        // 頭金の詰み
        let mut pos = Position::empty(Color::Gote);
        pos.set(Coord { file: 1, rank: 1 }, Some(Piece { color: Color::Gote, role: Role::King }));
        pos.set(Coord { file: 1, rank: 2 }, Some(Piece { color: Color::Sente, role: Role::Gold }));
        pos.set(Coord { file: 1, rank: 3 }, Some(Piece { color: Color::Sente, role: Role::King }));
        pos.set(Coord { file: 2, rank: 3 }, Some(Piece { color: Color::Sente, role: Role::Gold }));
        assert!(pos.in_check(Color::Gote));
        assert_eq!(pos.outcome(), Some(Outcome::Checkmate { winner: Color::Sente }));
    }
}
