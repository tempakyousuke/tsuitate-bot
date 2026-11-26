//! **手の区分**: 指した手を「何をしに行く手か」で 12 種に分ける（補助ヘッドのラベル。2026-11-24 ユーザーとの議論）。
//!
//! 学習側（tsuitate-nn の `--aux-class`）は、区分ごとの期待値（その区分の手を指したときの価値）を予測する補助ヘッドの
//! ラベルに使う。方策の入力には使わない（補助損失の教師だけ）。
//!
//! **判定は手番側の観測だけで決まる**（入力は `Strategy::choose` と同じ `(PlayerView, ObservationLog, foul_tried)` と手）。
//! 実際に駒を取れたか・王手になったかは結果なので使わず、手を選んだ時点で見えている「狙い」で分ける。だから同じ関数を
//! 学習の収集でも推論の診断でも通せる。
//!
//! 王手されていない手番は、上から順に当てはめて最初に当たった区分（排他的）:
//!
//! | 区分 | 判定 |
//! | --- | --- |
//! | 0 玉を動かす | 自玉の手 |
//! | 1 守りを固める | 自玉の8近傍の「穴」（盤上で、自駒がおらず玉以外の自駒の利きも無いマス）が減る |
//! | 2 確定した駒を取りに行く | 相手の駒がいると確定したマス（直前に取られたマス・打ちの反則など。`deduced_facts`）への盤上の手 |
//! | 3 王手を狙う | 相手玉の候補（`opp_king_candidates_tight`、`KING_TARGET_MAX` マス以下のとき）に新しく利きが届く |
//! | 4 逃げる | 相手の駒がいると確定したマスの隣か、そこから相手の桂が跳べるマスにいた駒を動かす |
//! | 5 寄せの準備 | 相手玉の候補の8近傍（候補自身と自駒のマスは除く）に新しく利きが届く |
//! | 6 当たりをかける | 相手の駒がいると確定したマスか、相手の初期配置のマスで自分の駒がまだ通っていないマスに、新しく利きが届く |
//! | 7 その他 | 残り全部 |
//!
//! 順番が効く例: 玉の隣にいると確定した相手の駒を取り、その結果として穴が減る手は、2 ではなく 1（守りを固める）になる。
//! 2 に入るのは、取っても自玉の周りの穴が減らない手だけ。
//!
//! 王手されている手番は既存の王手中の区分（`check_economy::CheckMoveKind` と同じ分け方）:
//! 8 玉の手 / 9 王手駒の候補のマスを取りに行く盤上の手 / 10 その他の盤上の手（移動合など）/ 11 打ち（合駒打ち）。
//! 王手駒の候補はマスクの推論（`deduced_facts().checkers`）を使う。
//!
//! 利きは `board::defend_targets`（自駒にだけ遮られる楽観値。相手の駒は見えない）で、玉以外の自駒の利きの和。
//! 「新しく利きが届く」は着手後の利きの集合から着手前の集合を引いたもの。

use std::collections::{BTreeSet, HashSet};

use crate::board::{Coord, defend_targets, parse_usi_square};
use crate::deduce_king::opp_king_candidates_tight;
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{Color, PlayerView, Role, VisiblePiece};
use crate::rl::action::deduced_facts;
use crate::shogi::{Position, ShogiMove, parse_usi, promote_role};

/// 区分の数（平時 8 + 王手中 4）
pub const NUM_MOVE_CLASSES: usize = 12;

/// 区分の名前（添字 = `MoveClass as usize`。学習側は `tsuitate_rl.MOVE_CLASS_NAMES` で引く）
pub const MOVE_CLASS_NAMES: [&str; NUM_MOVE_CLASSES] = [
    "king",
    "guard",
    "capture_known",
    "aim_check",
    "flee",
    "approach",
    "threat",
    "other",
    "check_king",
    "check_capture",
    "check_board",
    "check_drop",
];

/// 相手玉の候補がこれより多いときは、王手を狙う・寄せの準備の区分を使わない（候補が広いと「相手陣のどこかへ利きを
/// 足す手」がほぼ全部当たってしまい、区分が意味を持たない）。9 = 3×3 ぶん
pub const KING_TARGET_MAX: usize = 9;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MoveClass {
    King = 0,
    Guard = 1,
    CaptureKnown = 2,
    AimCheck = 3,
    Flee = 4,
    Approach = 5,
    Threat = 6,
    Other = 7,
    CheckKing = 8,
    CheckCapture = 9,
    CheckBoard = 10,
    CheckDrop = 11,
}

impl MoveClass {
    pub const ALL: [MoveClass; NUM_MOVE_CLASSES] = [
        MoveClass::King,
        MoveClass::Guard,
        MoveClass::CaptureKnown,
        MoveClass::AimCheck,
        MoveClass::Flee,
        MoveClass::Approach,
        MoveClass::Threat,
        MoveClass::Other,
        MoveClass::CheckKing,
        MoveClass::CheckCapture,
        MoveClass::CheckBoard,
        MoveClass::CheckDrop,
    ];

    pub fn name(self) -> &'static str {
        MOVE_CLASS_NAMES[self as usize]
    }
}

/// 手 `usi` の区分。`usi` が読めない・動かす自駒が無いときは None
pub fn classify(
    view: &PlayerView,
    log: &ObservationLog,
    foul_tried: &HashSet<String>,
    usi: &str,
) -> Option<MoveClass> {
    classify_with(view, log, foul_tried, usi, &mut || {
        opp_king_candidates_tight(view.your_color, log)
    })
}

/// `classify` と同じだが、相手玉の候補（`opp_king_candidates_tight` と同じ値）を呼び出し側から渡す。
/// 候補は局の頭からログを読み直すので重い（ランダム対局で1手番 約0.8ms）。RL 環境は局ごとの途中状態
/// （`encode::DeducedCache`）から渡す。要るときだけ呼ぶ
pub fn classify_with(
    view: &PlayerView,
    log: &ObservationLog,
    foul_tried: &HashSet<String>,
    usi: &str,
    king_candidates: &mut dyn FnMut() -> BTreeSet<Coord>,
) -> Option<MoveClass> {
    let mv = parse_usi(usi)?;
    let color = view.your_color;
    let pieces = &view.your_pieces;
    let king = pieces
        .iter()
        .find(|p| p.role == Role::King)
        .and_then(|p| parse_usi_square(&p.square));
    let moving = match mv {
        ShogiMove::Board { from, .. } => Some(
            pieces
                .iter()
                .find(|p| parse_usi_square(&p.square) == Some(from))?,
        ),
        ShogiMove::Drop { .. } => None,
    };
    let is_king_move = moving.is_some_and(|p| p.role == Role::King);

    if view.you_in_check {
        return Some(match mv {
            ShogiMove::Drop { .. } => MoveClass::CheckDrop,
            _ if is_king_move => MoveClass::CheckKing,
            ShogiMove::Board { to, .. } => {
                if deduced_facts(view, log, foul_tried).checkers.contains(&to) {
                    MoveClass::CheckCapture
                } else {
                    MoveClass::CheckBoard
                }
            }
        });
    }
    if is_king_move {
        return Some(MoveClass::King);
    }

    let after = pieces_after(pieces, &mv, moving);
    if let Some(k) = king
        && king_holes(&after, k, color) < king_holes(pieces, k, color)
    {
        return Some(MoveClass::Guard);
    }

    let facts = deduced_facts(view, log, foul_tried);
    let occupied: HashSet<Coord> = facts.occupied.iter().copied().collect();
    if let ShogiMove::Board { to, .. } = mv
        && occupied.contains(&to)
    {
        return Some(MoveClass::CaptureKnown);
    }

    let before_att = attacks(pieces, color);
    let new_att: HashSet<Coord> = attacks(&after, color)
        .difference(&before_att)
        .copied()
        .collect();
    let own_after: HashSet<Coord> = after
        .iter()
        .filter_map(|p| parse_usi_square(&p.square))
        .collect();
    let cands = king_candidates();
    let aim_king = cands.len() <= KING_TARGET_MAX;
    if aim_king && cands.iter().any(|c| new_att.contains(c)) {
        return Some(MoveClass::AimCheck);
    }

    if let ShogiMove::Board { from, .. } = mv
        && occupied
            .iter()
            .any(|&x| threatened_from(x, from, color.other()))
    {
        return Some(MoveClass::Flee);
    }

    if aim_king
        && near_candidates(&cands, &own_after)
            .iter()
            .any(|c| new_att.contains(c))
    {
        return Some(MoveClass::Approach);
    }

    let mut suspected = unvisited_initial_squares(color, log, &own_after);
    suspected.extend(occupied.iter().copied());
    if suspected.iter().any(|c| new_att.contains(c)) {
        return Some(MoveClass::Threat);
    }
    Some(MoveClass::Other)
}

/// 着手後の自駒（盤上）。駒を取っても自駒の配置は動かした駒の分しか変わらない
fn pieces_after(
    pieces: &[VisiblePiece],
    mv: &ShogiMove,
    moving: Option<&VisiblePiece>,
) -> Vec<VisiblePiece> {
    match *mv {
        ShogiMove::Board { from, to, promote } => {
            let role = moving.map_or(Role::Pawn, |p| {
                if promote {
                    promote_role(p.role).unwrap_or(p.role)
                } else {
                    p.role
                }
            });
            pieces
                .iter()
                .filter(|p| parse_usi_square(&p.square) != Some(from))
                .cloned()
                .chain(std::iter::once(VisiblePiece {
                    square: crate::board::make_usi_square(to),
                    role,
                }))
                .collect()
        }
        ShogiMove::Drop { role, to } => pieces
            .iter()
            .cloned()
            .chain(std::iter::once(VisiblePiece {
                square: crate::board::make_usi_square(to),
                role,
            }))
            .collect(),
    }
}

/// 玉以外の自駒の利き（自駒のマスも含む。`defend_targets`）
fn attacks(pieces: &[VisiblePiece], color: Color) -> HashSet<Coord> {
    pieces
        .iter()
        .filter(|p| p.role != Role::King)
        .flat_map(|p| defend_targets(pieces, p, color))
        .collect()
}

/// 自玉 `king` の8近傍の穴の数（盤上で、自駒がおらず玉以外の自駒の利きも無いマス。`strategy.rs` の king_hole と同じ）
fn king_holes(pieces: &[VisiblePiece], king: Coord, color: Color) -> usize {
    let att = attacks(pieces, color);
    let own: HashSet<Coord> = pieces
        .iter()
        .filter_map(|p| parse_usi_square(&p.square))
        .collect();
    neighbors(king)
        .filter(|c| !own.contains(c) && !att.contains(c))
        .count()
}

fn neighbors(c: Coord) -> impl Iterator<Item = Coord> {
    (-1..=1i8)
        .flat_map(move |df| (-1..=1i8).map(move |dr| (df, dr)))
        .filter(|&d| d != (0, 0))
        .map(move |(df, dr)| Coord {
            file: c.file + df,
            rank: c.rank + dr,
        })
        .filter(|c| crate::board::on_board(*c))
}

/// 相手の駒が `x` にいるとき、`sq` の自駒に当たりがかかり得るか: 隣（どの駒種でも1マスの利きはあり得る）か、
/// `x` の相手の桂が跳べるマス。飛び駒の遠い利きは駒種が分からないと広すぎるので見ない
fn threatened_from(x: Coord, sq: Coord, opp: Color) -> bool {
    let d = (x.file - sq.file).abs().max((x.rank - sq.rank).abs());
    d == 1 || crate::deduce::piece_attacks_square(Role::Knight, opp, x, sq)
}

/// 相手玉の候補の8近傍のうち、候補自身でも自駒のマスでもないもの
fn near_candidates(cands: &BTreeSet<Coord>, own: &HashSet<Coord>) -> HashSet<Coord> {
    cands
        .iter()
        .flat_map(|&c| neighbors(c))
        .filter(|c| !cands.contains(c) && !own.contains(c))
        .collect()
}

/// 相手の初期配置のマスのうち、自分の駒がまだ一度も止まったり通ったりしていないもの（受理された自分の手の
/// 着地点と、飛び駒の通り道。今の自駒のマスも除く）。駒が出ていった後に戻ることはあるが、序盤〜中盤の
/// 「まだそこにいそうな駒」の近似として使う
fn unvisited_initial_squares(
    color: Color,
    log: &ObservationLog,
    own: &HashSet<Coord>,
) -> HashSet<Coord> {
    let opp = color.other();
    let mut squares: HashSet<Coord> = Position::initial()
        .pieces()
        .filter(|(_, p)| p.color == opp)
        .map(|(c, _)| c)
        .filter(|c| !own.contains(c))
        .collect();
    for ev in log.events() {
        let Observation::MyMove { usi, .. } = ev else {
            continue;
        };
        match parse_usi(usi) {
            Some(ShogiMove::Board { from, to, .. }) => {
                for c in path(from, to) {
                    squares.remove(&c);
                }
            }
            Some(ShogiMove::Drop { to, .. }) => {
                squares.remove(&to);
            }
            None => {}
        }
    }
    squares
}

/// from から to への移動で通る・止まるマス（from は含まない）。直線でなければ（桂）着地点だけ
fn path(from: Coord, to: Coord) -> Vec<Coord> {
    let df = to.file - from.file;
    let dr = to.rank - from.rank;
    let aligned = df == 0 || dr == 0 || df.abs() == dr.abs();
    if !aligned {
        return vec![to];
    }
    let n = df.abs().max(dr.abs());
    let (sf, sr) = (df.signum(), dr.signum());
    (1..=n)
        .map(|k| Coord {
            file: from.file + sf * k,
            rank: from.rank + sr * k,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::action::{decode_usi, legal_mask};
    use crate::rl::env::{EnvGame, random_action};

    fn sq(s: &str) -> Coord {
        parse_usi_square(s).unwrap()
    }

    #[test]
    fn 区分の名前と並びがそろっている() {
        for (i, c) in MoveClass::ALL.iter().enumerate() {
            assert_eq!(*c as usize, i);
            assert_eq!(c.name(), MOVE_CLASS_NAMES[i]);
        }
    }

    #[test]
    fn 初期局面の手の区分() {
        let env = EnvGame::selfplay(None, 0);
        let view = env.referee().view(Color::Sente, [0, 0], 0);
        let log = ObservationLog::default();
        let none = HashSet::new();
        let c = |u: &str| classify(&view, &log, &none, u).unwrap();
        assert_eq!(c("5i4h"), MoveClass::King);
        // 角道を開ける 7g7f: 角の利きが 3c（相手の初期配置の歩）に届く = 当たりをかける
        assert_eq!(c("7g7f"), MoveClass::Threat);
        // 端歩 1g1f: 新しい利きは 1e だけで、相手の初期配置にも玉の候補の周りにも届かない
        assert_eq!(c("1g1f"), MoveClass::Other);
    }

    #[test]
    fn 穴の数は自駒と利きで決まる() {
        let pos = Position::initial();
        let pieces = pos.pieces_of(Color::Sente);
        // 初期局面の 5九玉の8近傍（盤上は5マス）: 4八・5八・6八 は金銀の利き、4九・6九 は金で埋まる
        assert_eq!(king_holes(&pieces, sq("5i"), Color::Sente), 0);
        let no_gold: Vec<_> = pieces
            .iter()
            .filter(|p| p.role != Role::Gold)
            .cloned()
            .collect();
        assert!(king_holes(&no_gold, sq("5i"), Color::Sente) > 0);
    }

    #[test]
    fn 通り道は直線なら途中も含む() {
        assert_eq!(path(sq("2h"), sq("2e")), vec![sq("2g"), sq("2f"), sq("2e")]);
        assert_eq!(path(sq("2i"), sq("3g")), vec![sq("3g")]);
    }

    /// ランダム対局の全手番で、マスクの中の手はどれも区分が付き、王手中の手番は王手中の区分、
    /// そうでなければ平時の区分になる。区分は観測だけで決まる（同じ入力なら同じ区分）
    #[test]
    fn ランダム対局の全手番で区分が付く() {
        let mut counts = [0usize; NUM_MOVE_CLASSES];
        let mut rng = 12345u64;
        let mut next = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 11) as f64 / (1u64 << 53) as f64
        };
        for game in 0..3 {
            let mut env = EnvGame::selfplay(None, game);
            while env.outcome().is_none() {
                let side = env.to_move();
                let referee = env.referee();
                let view = referee.view(side, [0, 0], game);
                let log = referee.log(side);
                let foul = referee.foul_tried(side);
                let mask = legal_mask(&view, log, foul);
                for a in mask
                    .iter()
                    .enumerate()
                    .filter(|(_, m)| **m)
                    .map(|(a, _)| a)
                    .step_by(31)
                {
                    let usi = decode_usi(a, side).unwrap();
                    let c = classify(&view, log, foul, &usi).expect("マスクの中の手には区分が付く");
                    assert_eq!(classify(&view, log, foul, &usi), Some(c));
                    assert_eq!(view.you_in_check, c as usize >= 8, "{usi} {c:?}");
                    counts[c as usize] += 1;
                }
                let a = random_action(env.mask(), next()).unwrap();
                let want = classify(&view, log, foul, &decode_usi(a, side).unwrap());
                // 環境は局ごとの途中状態から相手玉の候補を作る（作り直したものと同じ区分になる）
                assert_eq!(env.move_class(a), want);
                env.step(a).unwrap();
            }
        }
        eprintln!("区分ごとの数（マスクの中の手の間引き）: {counts:?}");
        for c in [
            MoveClass::King,
            MoveClass::Guard,
            MoveClass::Threat,
            MoveClass::Other,
            MoveClass::CheckKing,
        ] {
            assert!(counts[c as usize] > 0, "{c:?} が1度も出ない: {counts:?}");
        }
    }
}
