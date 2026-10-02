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
//! マスクは**自駒だけを見た候補手**（`board.rs`）から、その手番で既に反則した手を除き、
//! さらに**観測から反則が確定する手**（`Deduction`）を除いたもの。
//! 見えている範囲で確定する反則（自駒のマス・自駒の飛び越え・二歩・行き所なし）と、
//! 観測から論理的に確定する反則（王手を解消し得ない手など）は落ち、
//! 見えない相手駒による反則のうち確定しないものは残る。**真の合法手はすべてマスクに含まれる**
//! （テストで常時検査。漏れた手は永久に指せなくなる）。
//!
//! 観測からの除外（`MASK_VERSION` 2、2026-10-02〜）を持たない旧マスクは `basic_legal_mask`。
//! 凍結版 rl_v15〜rl_v21 はこちらで学習・凍結されている。

use std::collections::HashSet;

use crate::board::{
    Coord, Promotion, drop_targets, make_usi_drop, make_usi_move, move_targets,
    parse_usi_square, promotion_choice,
};
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{Color, PlayerView, Role};
use crate::shogi::{HAND_ROLES, ShogiMove, hand_index, parse_usi};

/// マスクの版。1 = 自駒視点だけ（`basic_legal_mask`）、2 = 観測からの確定反則も除く。
/// 凍結版の同一性テストは、凍結ファイルがこの定数を持つかでどちらのマスクと比べるかを決める
pub const MASK_VERSION: u32 = 2;

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

/// 旧マスク（`MASK_VERSION` 1）: 自駒視点の候補手から、その手番で反則済みの手だけを除く
pub fn basic_legal_mask(view: &PlayerView, foul_tried: &HashSet<String>) -> Vec<bool> {
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

/// 行動マスク（長さ `NUM_ACTIONS`）。反則済みの手と、観測から反則が確定する手（`Deduction`）を除く。
///
/// 除外で候補が尽きたときは旧マスクへ戻す（推論が健全なら、真の局面に合法手がある限り
/// 尽きることはない。尽きる＝推論の前提が崩れているので、指せなくなるより安全側に倒す）
pub fn legal_mask(view: &PlayerView, log: &ObservationLog, foul_tried: &HashSet<String>) -> Vec<bool> {
    let deduction = Deduction::new(view, log, foul_tried);
    let mut mask = vec![false; NUM_ACTIONS];
    let mut basic = vec![false; NUM_ACTIONS];
    let mut any = false;
    for usi in candidate_usis(view) {
        if foul_tried.contains(&usi) {
            continue;
        }
        let Some(mv) = parse_usi(&usi) else { continue };
        let Some(a) = encode_move(&mv, view.your_color) else {
            continue;
        };
        basic[a] = true;
        if !deduction.rules_out(&usi, &mv) {
            mask[a] = true;
            any = true;
        }
    }
    if any { mask } else { basic }
}

/// 王手駒がいうるマスの1仮説。`dir` は玉から見た方向（桂は None）、`dist` は玉からの距離
#[derive(Debug, Clone, Copy)]
struct CheckerHyp {
    square: Coord,
    dir: Option<(i8, i8)>,
    dist: i8,
}

/// 観測から論理的に確定する反則。3種類あり、どれも**真の合法手を落とさない**ことだけを条件にしている
/// （「たぶん反則」は落とさない。それはネットの領分）。
///
/// 1. **反則した手の成り／不成の片割れ**: 両者の違いは移動後の駒種だけで、自玉が取られるか
///    （= 合法か）は盤上の占有にしか依らない。行き所の有無は候補生成が自駒視点で落とし済み
/// 2. **直前に相手が駒を取ったマス**（相手の着手駒がいまそこにいる）: そこへの打ちと、
///    そこを飛び越える移動
/// 3. **王手中に王手を解消し得ない手**: 王手駒がいうるマスの集合 H を作り、H のどの仮説に
///    対しても解消しない手を落とす。H は
///    - 相手の直前の手が駒を取っていなければ: 玉から8方向に最初の自駒の手前までのマスと、
///      桂の王手マス（自駒がいないもの）。王手駒はこのどこかにいる
///    - 駒を取っていれば（取られたマス X）: 直前の自分の手は合法だった＝相手の手の前は王手が
///      無かったので、王手駒は「X に来た着手駒」か「着手駒の元のマス O が空いて通った
///      飛び駒（開き王手）」しかない。O は玉の筋の上の空きマスで、その先に飛び駒が入る余地が
///      あり、X へ1手で行ける位置（直線か桂跳び）に限られる。開き王手の飛び駒は O の先の
///      筋の上のマス
///
///    仮説 h に対して手が王手を解消し得ない条件:
///    - 玉以外の手: 着地点が h（駒取り。打ちは除く）でも、玉と h の間（合駒）でもない
///    - 玉の手: h が2マス以上離れた筋の上（＝その方向へ利く飛び駒）で、移動先が同じ筋の上
///      （h の側へ1マス、または玉の真後ろ）。玉が動くと元のマスが空くので真後ろも利きの中。
///      h が隣接・桂のときは駒種が分からないので玉の手は落とさない
struct Deduction {
    twins: HashSet<String>,
    known_opponent: Option<Coord>,
    check: Option<(Coord, Vec<CheckerHyp>)>,
}

const ALL_DIRS: [(i8, i8); 8] = [
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
];

fn on_board(c: Coord) -> bool {
    (1..=9).contains(&c.file) && (1..=9).contains(&c.rank)
}

fn step(c: Coord, (df, dr): (i8, i8), k: i8) -> Coord {
    Coord {
        file: c.file + df * k,
        rank: c.rank + dr * k,
    }
}

/// from→to が直線の移動で、途中（両端を除く）に square を通るか
fn passes_through(from: Coord, to: Coord, square: Coord) -> bool {
    let (df, dr) = (to.file - from.file, to.rank - from.rank);
    if (df == 0 && dr == 0) || (df != 0 && dr != 0 && df.abs() != dr.abs()) {
        return false;
    }
    let unit = (df.signum(), dr.signum());
    let n = df.abs().max(dr.abs());
    (1..n).any(|k| step(from, unit, k) == square)
}

/// 何かの駒が o から x へ1手で行けうるか（直線か桂跳び。駒種と遮りは見ない＝上界）
fn reachable(o: Coord, x: Coord) -> bool {
    let (df, dr) = (x.file - o.file, x.rank - o.rank);
    if df == 0 && dr == 0 {
        return false;
    }
    df == 0 || dr == 0 || df.abs() == dr.abs() || (df.abs() == 1 && dr.abs() == 2)
}

impl Deduction {
    fn new(view: &PlayerView, log: &ObservationLog, foul_tried: &HashSet<String>) -> Self {
        let twins = foul_tried
            .iter()
            .filter_map(|usi| match parse_usi(usi)? {
                ShogiMove::Board { from, to, promote } => Some(make_usi_move(from, to, !promote)),
                ShogiMove::Drop { .. } => None,
            })
            .collect();

        // 直前の相手の手（自分の最後の受理手より後にあるもの）が取ったマス
        let mut known_opponent = None;
        for event in log.events().iter().rev() {
            match event {
                Observation::OpponentMoved {
                    captured_my_piece_at,
                    ..
                } => {
                    known_opponent = captured_my_piece_at.as_deref().and_then(parse_usi_square);
                    break;
                }
                Observation::MyMove { .. } => break,
                _ => {}
            }
        }

        let check = if view.you_in_check {
            Self::checker_hyps(view, known_opponent)
        } else {
            None
        };
        Deduction {
            twins,
            known_opponent,
            check,
        }
    }

    /// 王手駒がいうるマスの集合（上の doc の H）。前提が崩れている（空になる）ときは None
    fn checker_hyps(view: &PlayerView, captured_at: Option<Coord>) -> Option<(Coord, Vec<CheckerHyp>)> {
        let own: HashSet<Coord> = view
            .your_pieces
            .iter()
            .filter_map(|p| parse_usi_square(&p.square))
            .collect();
        let king = view
            .your_pieces
            .iter()
            .find(|p| p.role == Role::King)
            .and_then(|p| parse_usi_square(&p.square))?;

        // 8方向の「最初の自駒の手前まで」
        let rays: Vec<((i8, i8), Vec<Coord>)> = ALL_DIRS
            .iter()
            .map(|&d| {
                let mut squares = vec![];
                let mut c = step(king, d, 1);
                while on_board(c) && !own.contains(&c) {
                    squares.push(c);
                    c = step(c, d, 1);
                }
                (d, squares)
            })
            .collect();
        // 相手の桂が玉を狙える位置（相手から見て前へ跳んでくる）
        let forward = if view.your_color == Color::Sente { -2 } else { 2 };
        let knights: Vec<Coord> = [1, -1]
            .iter()
            .map(|&df| Coord {
                file: king.file + df,
                rank: king.rank + forward,
            })
            .filter(|&c| on_board(c) && !own.contains(&c))
            .collect();

        let ray_hyp = |d: (i8, i8), i: usize, c: Coord| CheckerHyp {
            square: c,
            dir: Some(d),
            dist: i as i8 + 1,
        };
        let knight_hyp = |c: Coord| CheckerHyp {
            square: c,
            dir: None,
            dist: 0,
        };

        let mut hyps = vec![];
        match captured_at {
            None => {
                for (d, squares) in &rays {
                    hyps.extend(squares.iter().enumerate().map(|(i, &c)| ray_hyp(*d, i, c)));
                }
                hyps.extend(knights.iter().map(|&c| knight_hyp(c)));
            }
            Some(x) => {
                // 着手駒そのものの王手
                for (d, squares) in &rays {
                    if let Some(i) = squares.iter().position(|&c| c == x) {
                        hyps.push(ray_hyp(*d, i, x));
                    }
                }
                if knights.contains(&x) {
                    hyps.push(knight_hyp(x));
                }
                // 開き王手: 元のマス o の先の筋の上
                for (d, squares) in &rays {
                    let opens = squares
                        .iter()
                        .enumerate()
                        .take(squares.len().saturating_sub(1))
                        .find(|&(_, &o)| o != x && reachable(o, x));
                    if let Some((oi, _)) = opens {
                        hyps.extend(
                            squares
                                .iter()
                                .enumerate()
                                .skip(oi + 1)
                                .map(|(i, &c)| ray_hyp(*d, i, c)),
                        );
                    }
                }
            }
        }
        if hyps.is_empty() { None } else { Some((king, hyps)) }
    }

    fn rules_out(&self, usi: &str, mv: &ShogiMove) -> bool {
        if self.twins.contains(usi) {
            return true;
        }
        if let Some(x) = self.known_opponent {
            let blocked = match *mv {
                ShogiMove::Drop { to, .. } => to == x,
                ShogiMove::Board { from, to, .. } => passes_through(from, to, x),
            };
            if blocked {
                return true;
            }
        }
        let Some((king, hyps)) = &self.check else {
            return false;
        };
        let (to, is_king, is_drop) = match *mv {
            ShogiMove::Board { from, to, .. } => (to, from == *king, false),
            ShogiMove::Drop { to, .. } => (to, false, true),
        };
        hyps.iter().all(|h| {
            if is_king {
                if to == h.square {
                    return false;
                }
                match h.dir {
                    Some(d) if h.dist >= 2 => to == step(*king, d, 1) || to == step(*king, d, -1),
                    _ => false,
                }
            } else {
                if !is_drop && to == h.square {
                    return false;
                }
                let blocks = h.dir.is_some_and(|d| (1..h.dist).any(|k| step(*king, d, k) == to));
                !blocks
            }
        })
    }
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
                let mask = legal_mask(&view, referee.log(side), referee.foul_tried(side));
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

    /// 王手を多く含む対局（真実を見て王手・駒取りを優先する偏った乱択）で、観測からの除外が
    /// 真の合法手を1つも落とさない。マスクの「尽きたら旧マスクへ戻す」安全弁を通さずに
    /// `Deduction::rules_out` を直接検査する
    #[test]
    fn 観測からの除外は真の合法手を落とさない() {
        // 局数と seed は env で増やせる（既定は CI 向けの軽い量）:
        // `RL_MASK_SOUNDNESS_GAMES=20000 RL_MASK_SOUNDNESS_SEED=1 cargo test --release --lib 観測からの除外`
        let games: u32 = std::env::var("RL_MASK_SOUNDNESS_GAMES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(300);
        let seed: u64 = std::env::var("RL_MASK_SOUNDNESS_SEED")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(20261002);
        let mut rng = StdRng::seed_from_u64(seed);
        let (mut legal_checked, mut check_states, mut capture_checks, mut ruled_out) = (0, 0, 0, 0);
        for game in 0..games {
            let mut referee = Referee::new();
            while referee.ended().is_none() {
                let side = referee.to_move();
                let view = referee.view(side, [0, 0], game);
                let log = referee.log(side);
                let tried = referee.foul_tried(side);
                let deduction = Deduction::new(&view, log, tried);
                let legal = referee.position().legal_moves();
                for mv in &legal {
                    let usi = mv.to_usi();
                    assert!(
                        !deduction.rules_out(&usi, mv),
                        "局 {game}: 合法手 {usi} を除外した（王手中 {}・直前の捕獲 {:?}）",
                        view.you_in_check,
                        deduction.known_opponent
                    );
                    legal_checked += 1;
                }
                let basic = basic_legal_mask(&view, tried);
                let mask = legal_mask(&view, log, tried);
                ruled_out += (0..NUM_ACTIONS).filter(|&a| basic[a] && !mask[a]).count();
                if view.you_in_check {
                    check_states += 1;
                    capture_checks += deduction.known_opponent.is_some() as usize;
                }

                // 半分は真実を見て王手（なければ駒取り）を選ぶ。残りはマスク内から一様に（反則も起きる）
                let gives_check = |mv: &ShogiMove| {
                    let mut p = referee.position().clone();
                    p.play_unchecked(mv);
                    p.in_check(p.turn())
                };
                let pick = if rng.random_bool(0.5) {
                    let checks: Vec<&ShogiMove> = legal.iter().filter(|mv| gives_check(mv)).collect();
                    if checks.is_empty() {
                        None
                    } else {
                        Some(checks[rng.random_range(0..checks.len())].to_usi())
                    }
                } else {
                    None
                };
                let usi = match pick {
                    Some(u) => u,
                    None => {
                        let actions: Vec<usize> = (0..NUM_ACTIONS).filter(|&a| mask[a]).collect();
                        if actions.is_empty() {
                            break;
                        }
                        decode_usi(actions[rng.random_range(0..actions.len())], side).unwrap()
                    }
                };
                referee.step(&usi, 0);
            }
        }
        println!(
            "合法手 {legal_checked} / 王手中 {check_states}（うち駒取りの王手 {capture_checks}）/ 除外 {ruled_out}"
        );
        assert!(legal_checked > 100_000, "検査した合法手が少なすぎる: {legal_checked}");
        assert!(check_states > 2_000, "王手中の局面が少なすぎる: {check_states}");
        assert!(capture_checks > 300, "駒取りの王手が少なすぎる: {capture_checks}");
        assert!(ruled_out > 10_000, "除外が働いていない: {ruled_out}");
    }

    /// 対局記録の真実を再生し、全試行の直前で「観測からの除外が合法手を落とさない」ことを確かめる。
    /// あわせて、記録された反則のうち除外に当たる（= 避けられた）ものを数える。
    /// `RL_RECORDS_DIR=<dir> cargo test --release --lib 記録の全局面 -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn 記録の全局面で観測からの除外は合法手を落とさない() {
        let dir = std::env::var("RL_RECORDS_DIR").expect("RL_RECORDS_DIR");
        let (mut games, mut legal_checked, mut fouls, mut avoidable, mut check_fouls, mut check_avoidable) =
            (0, 0usize, 0, 0, 0, 0);
        // サブディレクトリも辿る（CI の artifact をシャードごとに展開したままでよい）
        let mut paths = vec![];
        let mut stack = vec![std::path::PathBuf::from(&dir)];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).unwrap() {
                let p = e.unwrap().path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|x| x == "jsonl") {
                    paths.push(p);
                }
            }
        }
        paths.sort();
        for path in paths {
            let Ok(content) = std::fs::read_to_string(&path) else { continue };
            let Some((_, end)) = crate::truth_replay::parse_bot_and_end(&content) else {
                continue;
            };
            let replayed = crate::rl::records::replay(&end, |r, a| {
                let view = r.view(a.side, [0, 0], 0);
                let deduction = Deduction::new(&view, r.log(a.side), r.foul_tried(a.side));
                for mv in r.position().legal_moves() {
                    let usi = mv.to_usi();
                    assert!(
                        !deduction.rules_out(&usi, &mv),
                        "{}: 合法手 {usi} を除外した（王手中 {}）",
                        path.display(),
                        view.you_in_check
                    );
                    legal_checked += 1;
                }
                if a.foul {
                    let out = parse_usi(&a.usi).is_some_and(|mv| deduction.rules_out(&a.usi, &mv));
                    fouls += 1;
                    avoidable += out as usize;
                    if view.you_in_check {
                        check_fouls += 1;
                        check_avoidable += out as usize;
                    }
                }
            });
            if replayed.is_ok() {
                games += 1;
            }
        }
        println!(
            "{games} 局・合法手 {legal_checked} を検査。反則 {fouls} のうち除外に当たる {avoidable}\
             （王手中 {check_fouls} のうち {check_avoidable}）"
        );
        assert!(games > 0);
    }

    /// SFEN の盤面から手番側の駒だけを取り出した視界（王手中）
    fn view_from_sfen(board: &str, color: Color, hand: &[(Role, u32)]) -> PlayerView {
        use crate::protocol::{ClockState, FoulCounts, GameStatus, VisiblePiece};
        let mut pieces = vec![];
        for (r, row) in board.split('/').enumerate() {
            let mut file = 9i8;
            let mut promoted = false;
            for ch in row.chars() {
                if let Some(n) = ch.to_digit(10) {
                    file -= n as i8;
                    continue;
                }
                if ch == '+' {
                    promoted = true;
                    continue;
                }
                let mine = ch.is_ascii_uppercase() == (color == Color::Sente);
                let role = match (ch.to_ascii_lowercase(), promoted) {
                    ('p', false) => Role::Pawn,
                    ('l', false) => Role::Lance,
                    ('n', false) => Role::Knight,
                    ('s', false) => Role::Silver,
                    ('g', _) => Role::Gold,
                    ('b', false) => Role::Bishop,
                    ('r', false) => Role::Rook,
                    ('k', _) => Role::King,
                    ('p', true) => Role::Tokin,
                    ('l', true) => Role::Promotedlance,
                    ('n', true) => Role::Promotedknight,
                    ('s', true) => Role::Promotedsilver,
                    ('b', true) => Role::Horse,
                    ('r', true) => Role::Dragon,
                    other => panic!("{other:?}"),
                };
                if mine {
                    let sq = Coord {
                        file,
                        rank: r as i8 + 1,
                    };
                    pieces.push(VisiblePiece {
                        square: crate::board::make_usi_square(sq),
                        role,
                    });
                }
                file -= 1;
                promoted = false;
            }
        }
        PlayerView {
            game_id: "t".into(),
            your_color: color,
            your_pieces: pieces,
            your_hand: hand.iter().copied().collect(),
            turn: color,
            move_number: 1,
            clocks: ClockState {
                sente_ms: 0,
                gote_ms: 0,
                running: None,
                server_time: 0,
            },
            fouls: FoulCounts { you: 0, opponent: 0 },
            you_in_check: true,
            opponent_in_check: false,
            status: GameStatus::Playing,
        }
    }

    fn check_log(captured_at: Option<&str>, in_check: Color) -> ObservationLog {
        let mut log = ObservationLog::default();
        log.record(Observation::OpponentMoved {
            move_number: 1,
            captured_my_piece_at: captured_at.map(str::to_string),
        });
        log.record(Observation::Check { in_check });
        log
    }

    fn allowed(view: &PlayerView, log: &ObservationLog, tried: &HashSet<String>, usi: &str) -> bool {
        legal_mask(view, log, tried)[encode_usi(usi, view.your_color).unwrap()]
    }

    /// webhook の実戦（2026-10-02、gameId 8b91eec3…）: 39手目に 3九の金を取られて 7九の玉に王手。
    /// 開き王手は起こりえない（玉の筋の空きマスの先に飛び駒の余地が無い）ので王手駒は 3九の飛車か竜。
    /// rl_v20 は 6九玉・8九玉・8九銀・6八金をすべて反則していた
    #[test]
    fn 取られたマスから王手駒を特定して横の筋の玉逃げを落とす() {
        let view = view_from_sfen(
            "9/9/9/2P4G1/4B4/9/PPNPPPP1P/2S1G4/L1K5L",
            Color::Sente,
            &[(Role::Pawn, 1)],
        );
        let log = check_log(Some("3i"), Color::Sente);
        let tried = HashSet::new();
        for usi in ["7i6i", "7i8i", "7h8i", "5h6h"] {
            assert!(!allowed(&view, &log, &tried, usi), "{usi} は確定反則");
        }
        // 合駒（7八銀→6九 が実戦で通った手）・玉の斜め逃げは残る
        for usi in ["7h6i", "5h5i", "7i8h", "7i6h"] {
            assert!(allowed(&view, &log, &tried, usi), "{usi} は残す");
        }
    }

    /// webhook の実戦（gameId 84acc854…）: 1一玉に駒を取らない王手。2二玉・2一玉が反則した後、
    /// 玉の筋に関わらない竜・金・角の手と、反則した 7五竜成 の不成を指して反則負け
    #[test]
    fn 王手の筋に関わらない手と反則した手の片割れを落とす() {
        let view = view_from_sfen("l7k/8l/p3pp1pp/5b3/9/4g4/6+p2/2rg5/9", Color::Gote, &[]);
        let log = check_log(None, Color::Gote);
        let tried: HashSet<String> = ["1a2b", "1a2a", "7h7e+"].iter().map(|s| s.to_string()).collect();
        for usi in ["7h7e", "6h5i", "4d5e"] {
            assert!(!allowed(&view, &log, &tried, usi), "{usi} は確定反則");
        }
        // 2二・3三への合駒、1筋の横への合駒は残る
        for usi in ["4d3c", "7h7a", "4d2b"] {
            assert!(allowed(&view, &log, &tried, usi), "{usi} は残す");
        }
    }

    /// 直前に相手が駒を取ったマスへは打てず、そこを飛び越えられない（王手でなくても）
    #[test]
    fn 直前に取られたマスへの打ちと飛び越えを落とす() {
        let mut view = view_from_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPP1PP/1B5R1/LNSGKGSNL",
            Color::Sente,
            &[(Role::Pawn, 1)],
        );
        view.you_in_check = false;
        let mut log = ObservationLog::default();
        log.record(Observation::OpponentMoved {
            move_number: 1,
            captured_my_piece_at: Some("3e".into()),
        });
        let tried = HashSet::new();
        assert!(!allowed(&view, &log, &tried, "P*3e"));
        assert!(allowed(&view, &log, &tried, "P*3f"));
    }
}
