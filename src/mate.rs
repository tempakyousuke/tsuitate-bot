//! 一手詰めの検出。攻め（詰めろ生成ボーナス）と受け（被詰めろペナルティ）の
//! 共通部品で、`bin/analyze` の被詰めろオラクル指標もここを使う。
//!
//! **持ち駒打ちに限定**した一手詰めだけを見る。理由は2つ:
//! - 費用: 全合法手からの一手詰め探索は候補手×粒子のホットパスに載らない
//!   （合法手150手級 × 相手の合法手生成）。打ちなら候補は玉の近傍・利き線上の
//!   空きマス × 持ち駒種に絞れるので1〜2桁安い
//! - 頻度: ついたて将棋の終盤は双方が持ち駒を多く抱えるうえ、相手からは
//!   脅威が見えないので受けられない。実戦で決め手になる詰みは打ちが支配的
//!   （2026-07-25 の対人局 58手目 N*6六 →（受けられず）60手目 G*7八 詰み が発端）
//!
//! 歩打ちは打ち歩詰めが反則なので原理的に詰みにならない。候補から外している。

use crate::board::{Coord, on_board, orient, rays, steps};
use crate::protocol::{Color, Role};
use crate::shogi::{Position, ShogiMove};

/// 打ちで王手になりうる (マス, 駒種)。空きマスのみ・持ち駒があるものだけ。
/// 玉のマスから8方向へ空きマスを辿り、「そのマスの `side` の `role` が
/// 玉へ利くか」を移動テーブル（board::steps/rays）から逆引きする。
fn drop_check_candidates(pos: &Position, side: Color, king: Coord) -> Vec<(Coord, Role)> {
    // 歩は打ち歩詰めが反則なので除外（玉は打てない）
    const DROPPABLE: [Role; 5] = [Role::Lance, Role::Silver, Role::Gold, Role::Bishop, Role::Rook];
    let mut out = Vec::new();

    // 桂は跳ぶので直線走査に乗らない。玉へ利く2マスを逆算する
    if pos.hand_count(side, Role::Knight) > 0 {
        for &delta in steps(Role::Knight) {
            let (df, dr) = orient(delta, side);
            let s = Coord { file: king.file - df, rank: king.rank - dr };
            if on_board(s) && pos.piece_at(s).is_none() {
                out.push((s, Role::Knight));
            }
        }
    }

    const DIRS: [(i8, i8); 8] = [
        (0, -1), (0, 1), (1, 0), (-1, 0), (1, -1), (-1, -1), (1, 1), (-1, 1),
    ];
    for (df, dr) in DIRS {
        let mut c = Coord { file: king.file + df, rank: king.rank + dr };
        let mut dist = 1;
        while on_board(c) {
            if pos.piece_at(c).is_some() {
                break; // 先頭の駒で遮断（打てるのは空きマスだけ）
            }
            let back = (-df, -dr); // c から玉へ向かう方向
            for role in DROPPABLE {
                if pos.hand_count(side, role) == 0 {
                    continue;
                }
                let hit = (dist == 1
                    && steps(role).iter().any(|&d| orient(d, side) == back))
                    || rays(role).iter().any(|&d| orient(d, side) == back);
                if hit {
                    out.push((c, role));
                }
            }
            c = Coord { file: c.file + df, rank: c.rank + dr };
            dist += 1;
        }
    }
    out
}

/// 王手された側（`side`）の逃げ道の形。
enum Escape {
    /// 合法手なし = 詰み
    None,
    /// 「玉で打った駒（`drop_sq`）を取る」以外に合法手がない
    OnlyKingTakesDrop,
    /// それ以外の受けがある
    Other,
}

/// 逃げ道の形を、全合法手を構築せずに判定する（`Other` が確定した時点で打ち切る）。
/// 合法性の基準は `Position::has_any_legal_move` と同じ（打ち歩詰めの入れ子は見ない）
fn escape_shape(pos: &Position, side: Color, drop_sq: Coord) -> Escape {
    let king = pos.king_square(side);
    let mut only_king_takes = true;
    let mut any = false;
    for mv in pos.pseudo_legal_moves() {
        let mut next = pos.clone();
        next.play_unchecked(&mv);
        if next.in_check(side) {
            continue;
        }
        any = true;
        let king_takes_drop = matches!(
            mv,
            ShogiMove::Board { from, to, .. } if to == drop_sq && Some(from) == king
        );
        if !king_takes_drop {
            only_king_takes = false;
            break;
        }
    }
    match (any, only_king_takes) {
        (false, _) => Escape::None,
        (true, true) => Escape::OnlyKingTakesDrop,
        (true, false) => Escape::Other,
    }
}

/// 打ち一手詰めの成立度。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MateThreat {
    /// 盤上の駒だけで詰んでいる（打った駒を玉で取れない）
    Mate,
    /// 打った駒を玉で取る以外に受けがない = **打ちマスに支えが1枚あれば詰み**。
    /// 支えは相手（守る側）から見て不可視な駒でありうるので、受け側の評価では
    /// これも危険信号として扱う。逆に攻め側は自分の支え駒を正確に知っているので
    /// `Mate` だけを使う
    IfSupported,
}

/// `side` が手番を得たら持ち駒打ちで一手詰めにできるか（できるならその手と成立度）。
///
/// `pos` の手番は**無視して** `side` の手番として判定する。詰めろ（次の自分の
/// 手番での脅威）の判定に使うため、相手の手番を挟んだ局面でも呼べる必要がある。
///
/// `Mate` を見つけたら即返す。無ければ `IfSupported` の最初の1件を返す。
pub fn drop_mate_threat(pos: &Position, side: Color) -> Option<(ShogiMove, MateThreat)> {
    let opp = side.other();
    let king = pos.king_square(opp)?;
    let mut base = pos.clone();
    base.set_turn(side);
    let mut if_supported = None;
    for (to, role) in drop_check_candidates(&base, side, king) {
        let mv = ShogiMove::Drop { role, to };
        if !base.is_legal(&mv) {
            continue;
        }
        let mut next = base.clone();
        next.play_unchecked(&mv);
        if !next.in_check(opp) {
            continue;
        }
        match escape_shape(&next, opp, to) {
            Escape::None => return Some((mv, MateThreat::Mate)),
            Escape::OnlyKingTakesDrop if if_supported.is_none() => {
                if_supported = Some((mv, MateThreat::IfSupported));
            }
            _ => {}
        }
    }
    if_supported
}

/// `side` が手番を得たら持ち駒打ちで（盤上の駒だけで）一手詰めにできるか
pub fn drop_mate(pos: &Position, side: Color) -> Option<ShogiMove> {
    match drop_mate_threat(pos, side) {
        Some((mv, MateThreat::Mate)) => Some(mv),
        _ => None,
    }
}

/// 手番側の一手詰め（全合法手を見る完全版。診断・記録分析用で重い）
pub fn mate_in_1(pos: &Position) -> Option<ShogiMove> {
    let opp = pos.turn().other();
    for mv in pos.legal_moves() {
        let mut next = pos.clone();
        next.play_unchecked(&mv);
        if next.in_check(opp) && !next.has_any_legal_move() {
            return Some(mv);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shogi::parse_usi;

    /// 発端の実戦局面（scenarios/play-estimator-20260725-090838.kif の 58手目まで、
    /// 先手番）。後手は直前に打った 6六桂の支えで G*7八 の一手詰めを持っている
    fn play_estimator_ply58() -> Position {
        use crate::shogi::Piece;
        let mut pos = Position::empty(Color::Sente);
        let mut put = |file: i8, rank: i8, color: Color, role: Role| {
            pos.set(Coord { file, rank }, Some(Piece { color, role }));
        };
        use Color::{Gote as G, Sente as S};
        use Role::*;
        // 後手
        put(9, 1, G, Gold);
        put(4, 1, G, Gold);
        put(3, 1, G, Bishop);
        put(2, 1, G, Knight);
        put(1, 1, G, Lance);
        put(3, 2, G, Silver);
        put(5, 3, G, King);
        for f in [9, 7, 6, 4, 3, 2, 1] {
            put(f, 3, G, Pawn);
        }
        put(5, 4, G, Pawn);
        put(6, 6, G, Knight); // 58手目の打ち（先手からは不可視）
        // 先手
        put(7, 9, S, King);
        put(9, 9, S, Lance);
        put(1, 9, S, Lance);
        put(4, 9, S, Gold);
        put(3, 9, S, Silver);
        put(7, 7, S, Bishop);
        put(3, 6, S, Silver);
        put(1, 7, S, Knight);
        for (f, r) in [(9, 7), (8, 7), (6, 7), (4, 7), (3, 7), (2, 7)] {
            put(f, r, S, Pawn);
        }
        put(7, 6, S, Pawn);
        put(5, 6, S, Pawn);
        put(1, 6, S, Pawn);
        pos.set_hand(S, Knight, 1);
        pos.set_hand(S, Rook, 1);
        for role in [Gold, Lance, Pawn, Rook, Silver] {
            pos.set_hand(G, role, 1);
        }
        pos
    }

    #[test]
    fn drop_mate_finds_the_real_game_mate() {
        let pos = play_estimator_ply58();
        // 先手番。後手は「手番を得たら」G*7八 で詰ませられる = 先手は詰めろを掛けられている
        assert_eq!(pos.turn(), Color::Sente);
        let mate = drop_mate(&pos, Color::Gote).expect("後手の打ち一手詰めがあるはず");
        assert_eq!(mate.to_usi(), "G*7h");
        // 先手側には詰めろは掛かっていない
        assert!(drop_mate(&pos, Color::Sente).is_none());
    }

    #[test]
    fn drop_mate_is_prevented_by_filling_the_square() {
        let pos = play_estimator_ply58();
        // 7八を埋める・玉が逃げる・支えの桂を取る、のいずれでも詰めろは消える
        for usi in ["R*7h", "N*7h", "7i8i", "7i6i", "6g6f", "7g6f"] {
            let mut next = pos.clone();
            let mv = parse_usi(usi).unwrap();
            assert!(next.is_legal(&mv), "非合法: {usi}");
            next.play_unchecked(&mv);
            assert!(
                drop_mate(&next, Color::Gote).is_none(),
                "{usi} の後も打ち一手詰めが残っている"
            );
        }
        // 逆に、玉から離れた攻め手では詰めろが残る（実戦の選択手）
        let mut next = pos.clone();
        next.play_unchecked(&parse_usi("N*8c").unwrap());
        assert!(drop_mate(&next, Color::Gote).is_some());
    }

    /// 受け側の判定は「相手の支え駒が見えない」前提で働く必要がある。
    /// 6六桂を取り除いた（= bot の粒子が持っていない）盤面でも、
    /// G*7八 が IfSupported として拾えることを確認する
    #[test]
    fn mate_risk_is_detected_without_seeing_the_supporting_piece() {
        let mut pos = play_estimator_ply58();
        pos.set(Coord { file: 6, rank: 6 }, None); // 支えの桂は先手からは不可視
        // 支えが無いので真の詰みではない
        assert!(drop_mate(&pos, Color::Gote).is_none());
        // が、「玉で取る以外に受けがない」形は残る = 支え1枚で詰む危険信号
        let (mv, kind) = drop_mate_threat(&pos, Color::Gote).expect("IfSupported があるはず");
        assert_eq!(mv.to_usi(), "G*7h");
        assert_eq!(kind, MateThreat::IfSupported);

        // 受けの手を指せば消える（支えの桂を取る手は、桂が見えていないので対象外）
        for usi in ["R*7h", "N*7h", "7i8i", "7i6i", "R*8h"] {
            let mut next = pos.clone();
            let m = parse_usi(usi).unwrap();
            assert!(next.is_legal(&m), "非合法: {usi}");
            next.play_unchecked(&m);
            assert!(
                drop_mate_threat(&next, Color::Gote).is_none(),
                "{usi} の後も被詰めろが残っている"
            );
        }
        // 攻め手（実戦の選択）では残る
        let mut next = pos.clone();
        next.play_unchecked(&parse_usi("N*8c").unwrap());
        assert_eq!(
            drop_mate_threat(&next, Color::Gote).map(|(_, k)| k),
            Some(MateThreat::IfSupported)
        );
    }

    #[test]
    fn drop_mate_agrees_with_full_mate_search() {
        let pos = play_estimator_ply58();
        let mut gote_turn = pos.clone();
        gote_turn.set_turn(Color::Gote);
        // 打ち限定の詰みは全合法手探索の詰みでもある
        assert_eq!(
            mate_in_1(&gote_turn).map(|m| m.to_usi()),
            Some("G*7h".to_string())
        );
    }
}
