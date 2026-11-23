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
//! 凍結版 rl_v15〜rl_v22 はこちらで学習・凍結されている。

use std::collections::HashSet;

use crate::board::{
    Coord, Promotion, drop_targets, make_usi_drop, make_usi_move, move_targets, orient,
    parse_usi_square, promotion_choice, rays, steps,
};
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{Color, PlayerView, Role};
use crate::shogi::{
    HAND_ROLE_TOTAL, HAND_ROLES, ShogiMove, hand_index, parse_usi, promote_role, unpromote_role,
};

/// マスクの版。1 = 自駒視点だけ、2 = 観測からの確定反則も除く、
/// 3 = 開き王手の仮説を自駒の遮りで絞る、4 = 相手に残り得る駒種とその動きも見る。
/// 5 = 直前に王手で逃げた升への戻りを、王手が残ると確認できる場合に除く。
/// 6 = 打ち歩詰めを否定できる歩打ちの反則からも占有を推定する。
pub const MASK_VERSION: u32 = 6;

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
    legal_mask_with_version(view, log, foul_tried, MASK_VERSION)
}

pub(super) fn legal_mask_with_version(
    view: &PlayerView,
    log: &ObservationLog,
    foul_tried: &HashSet<String>,
    mask_version: u32,
) -> Vec<bool> {
    let candidates = candidate_usis(view);
    let deduction = Deduction::new(view, log, foul_tried, &candidates.iter().cloned().collect(), mask_version);
    let mut mask = vec![false; NUM_ACTIONS];
    let mut basic = vec![false; NUM_ACTIONS];
    let mut any = false;
    for usi in candidates {
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

/// マスクの推論（`Deduction`）が確定させた事実のうち、盤のマスで表せるもの（エンコーダ v2 の入力）。
/// どれも真の盤面と矛盾しない（マスクが真の合法手を落とさないのと同じ根拠）
#[derive(Debug, Default, Clone, PartialEq)]
pub struct DeducedFacts {
    /// 相手の駒がいると確定したマス
    pub occupied: Vec<Coord>,
    /// 飛び越え反則ごとの間のマス（それぞれ少なくとも1マスに相手の駒がいる）
    pub blocked: Vec<Vec<Coord>>,
    /// 王手中: 王手駒がいうるマス（重複なし。王手中でない・前提が崩れているときは空）
    pub checkers: Vec<Coord>,
}

pub fn deduced_facts(view: &PlayerView, log: &ObservationLog, foul_tried: &HashSet<String>) -> DeducedFacts {
    let candidates: HashSet<String> = candidate_usis(view).into_iter().collect();
    // 学習済み方策への観測入力はマスクv2の時点の推論で固定する。
    let d = Deduction::new(view, log, foul_tried, &candidates, 2);
    let mut occupied: Vec<Coord> = d.occupied.into_iter().collect();
    occupied.sort_by_key(|c| (c.file, c.rank));
    let blocked = d
        .blocked_beyond
        .iter()
        .map(|&(f, unit, dist)| (1..dist).map(|k| step(f, unit, k)).collect())
        .collect();
    let mut checkers: Vec<Coord> = d
        .check
        .map(|(_, hyps)| hyps.into_iter().map(|h| h.square).collect())
        .unwrap_or_default();
    checkers.sort_by_key(|c| (c.file, c.rank));
    checkers.dedup();
    DeducedFacts {
        occupied,
        blocked,
        checkers,
    }
}

/// 王手駒がいうるマスの1仮説。`dir` は玉から見た方向（桂は None）、`dist` は玉からの距離
#[derive(Debug, Clone, Copy)]
struct CheckerHyp {
    square: Coord,
    dir: Option<(i8, i8)>,
    dist: i8,
}

/// 観測から論理的に確定する反則。どれも**真の合法手を落とさない**ことだけを条件にしている
/// （「たぶん反則」は落とさない。それはネットの領分）。
///
/// 1. **反則した手の成り／不成の片割れ**: 両者の違いは移動後の駒種だけで、自玉が取られるか
///    （= 合法か）は盤上の占有にしか依らない。行き所の有無は候補生成が自駒視点で落とし済み
/// 2. **相手の駒がいると確定したマス**（`occupied`）への打ちと、そこを飛び越える移動。確定の根拠は
///    - 直前に相手が駒を取ったマス（相手の着手駒がいまそこにいる）
///    - 王手中でないときの**打ちの反則**: 打ちで自玉が危なくなることは無く、二歩・
///      行き所は候補生成が落とし済み。歩は打ち歩詰めを否定できるときだけ占有を推定する
///    - 王手中でないときの**間が1マスの飛び越え反則**（下の 3 で原因が遮りと確定したもの）:
///      遮っている駒は間の1マスにしかいられない
/// 3. **飛び越え反則の先**（`blocked_beyond`）: 王手中でないときに2マス以上の直線移動 F→T が
///    反則で、F の駒がピンされえない（玉の筋で最初の自駒でない、または移動方向がその筋と平行）
///    なら、原因は「F と T の間に相手の駒がいる」しかない（T に相手の駒がいれば駒取りで合法）。
///    同じ F から同じ方向へ T より先へ進む手も必ず遮られる
/// 4. **王手中に王手を解消し得ない手**: 王手駒がいうるマスの集合 H を作り、H のどの仮説に
///    対しても解消しない手を落とす。H は
///    - 相手の直前の手が駒を取っていなければ: 玉から8方向に最初の自駒の手前までのマスと、
///      桂の王手マス（自駒がいないもの）。王手駒はこのどこかにいる
///    - 駒を取っていれば（取られたマス X）: 直前の自分の手は合法だった＝相手の手の前は王手が
///      無かったので、王手駒は「X に来た着手駒」か「着手駒の元のマス O が空いて通った
///      飛び駒（開き王手）」しかない。O は玉の筋の上の空きマスで、その先に飛び駒が入る余地が
///      あり、X へ1手で行ける位置に限られる。マスクv3から自駒の遮り、v4から相手に残り得る
///      駒種ごとの動きも調べる。開き王手の飛び駒は O の先の筋の上のマス
///
///    仮説 h に対して手が王手を解消し得ない条件:
///    - 玉以外の手: 着地点が h（駒取り。打ちは除く）でも、玉と h の間（合駒）でもない
///    - 玉の手: h が2マス以上離れた筋の上（＝その方向へ利く飛び駒）で、移動先が同じ筋の上
///      （h の側へ1マス、または玉の真後ろ）。玉が動くと元のマスが空くので真後ろも利きの中。
///      h が隣接・桂のときは駒種が分からないので玉の手は落とさない
struct Deduction {
    twins: HashSet<String>,
    /// 相手の駒がいると確定したマス（直前に取られたマスを含む）
    occupied: HashSet<Coord>,
    /// 飛び越え反則した (元のマス, 方向, 距離)。同じ元・方向でこれより遠い手は遮られる
    blocked_beyond: Vec<(Coord, (i8, i8), i8)>,
    check: Option<(Coord, Vec<CheckerHyp>)>,
    king_return: Option<Coord>,
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

/// 自分の盤上駒と持駒で総数を使い切った駒種は、相手の着手駒になれない。
/// それ以外は失われた駒や相手の持駒かもしれないので、成り・不成とも候補に残す。
fn possible_opponent_roles(view: &PlayerView) -> Vec<Role> {
    let mut own = [0u32; HAND_ROLES.len()];
    for piece in &view.your_pieces {
        if let Some(i) = hand_index(unpromote_role(piece.role)) {
            own[i] += 1;
        }
    }
    let mut roles = vec![Role::King];
    for (i, role) in HAND_ROLES.into_iter().enumerate() {
        let hand = view.your_hand.get(&role).copied().unwrap_or(0);
        if own[i].saturating_add(hand) < HAND_ROLE_TOTAL[i] {
            roles.push(role);
            roles.extend(promote_role(role));
        }
    }
    roles
}

/// 着手前の駒種の利きで到達できるか。途中の遮りは呼び出し側で調べる。
fn role_reaches(role: Role, color: Color, from: Coord, to: Coord) -> bool {
    let (df, dr) = (to.file - from.file, to.rank - from.rank);
    let distance = df.abs().max(dr.abs());
    steps(role).iter().any(|&d| orient(d, color) == (df, dr))
        || (distance > 0 && rays(role).iter().any(|&d| step(from, orient(d, color), distance) == to))
}

/// 打ち先が空だと仮定して、歩が王手できる升に相手玉がいないと証明する。
/// 飛び駒の途中がある場合は、空と仮定した打ち先だけを通る利きに限る。
fn pawn_drop_cannot_mate(view: &PlayerView, to: Coord) -> bool {
    let target = step(to, orient((0, -1), view.your_color), 1);
    view.your_pieces.iter().any(|piece| {
        let Some(from) = parse_usi_square(&piece.square) else { return false };
        let oriented = |d| orient(d, view.your_color);
        from == target
            || steps(piece.role).iter().any(|&d| step(from, oriented(d), 1) == target)
            || rays(piece.role).iter().any(|&d| {
                let d = oriented(d);
                let next = step(from, d, 1);
                next == target || (next == to && step(from, d, 2) == target)
            })
    })
}

impl Deduction {
    /// `candidates` は自駒視点の候補手（`candidate_usis`）。反則から推論するのは候補手の反則だけ
    /// （候補外の反則は原因が成りの選択や行き所でありうるので、占有・遮りの根拠にしない）
    fn new(
        view: &PlayerView,
        log: &ObservationLog,
        foul_tried: &HashSet<String>,
        candidates: &HashSet<String>,
        mask_version: u32,
    ) -> Self {
        let twins = foul_tried
            .iter()
            .filter_map(|usi| match parse_usi(usi)? {
                ShogiMove::Board { from, to, promote } => Some(make_usi_move(from, to, !promote)),
                ShogiMove::Drop { .. } => None,
            })
            .collect();

        // 直前の相手の手（自分の最後の受理手より後にあるもの）が取ったマス
        let mut known_opponent = None;
        let mut opponent_moved = false;
        for event in log.events().iter().rev() {
            match event {
                Observation::OpponentMoved {
                    captured_my_piece_at,
                    ..
                } => {
                    opponent_moved = true;
                    known_opponent = captured_my_piece_at.as_deref().and_then(parse_usi_square);
                    break;
                }
                Observation::MyMove { .. } => break,
                _ => {}
            }
        }

        // 動けないと確定している相手の歩（deduce::immobile_opponent_pawns）は使わない:
        // ログを毎回先頭から辿るのでマスクのコストが倍になる割に、記録483局で反則2件にしか効かない
        let mut occupied: HashSet<Coord> = known_opponent.into_iter().collect();

        let mut blocked_beyond = vec![];
        if !view.you_in_check {
            let king = view
                .your_pieces
                .iter()
                .find(|p| p.role == Role::King)
                .and_then(|p| parse_usi_square(&p.square));
            let own: HashSet<Coord> = view
                .your_pieces
                .iter()
                .filter_map(|p| parse_usi_square(&p.square))
                .collect();
            // f が玉の筋で最初の自駒なら、その筋の方向（＝ピンされうる向き）
            let pin_dir = |f: Coord| -> Option<(i8, i8)> {
                let k = king?;
                ALL_DIRS.iter().copied().find(|&d| {
                    let mut c = step(k, d, 1);
                    while on_board(c) && !own.contains(&c) {
                        c = step(c, d, 1);
                    }
                    c == f
                })
            };
            for usi in foul_tried.iter().filter(|u| candidates.contains(*u)) {
                match parse_usi(usi) {
                    // 歩（マスクv6〜）: `pawn_drop_cannot_mate` は「自駒が利いているマスに相手玉はいない」を使う。
                    // これは自分の手番で相手玉に王手がかかっていない（相手が自玉を王手に晒したまま指していない）
                    // ことが前提なので、`opponent_in_check` で確かめる。正規の対局の自分の手番では常に偽で、
                    // 効くのは手で組んだ局面・シナリオの開始局面のような、正規でない局面だけ
                    Some(ShogiMove::Drop { role, to })
                        if role != Role::Pawn
                            || (mask_version >= 6 && opponent_moved && !view.opponent_in_check
                                && pawn_drop_cannot_mate(view, to)) =>
                    {
                        occupied.insert(to);
                    }
                    Some(ShogiMove::Board { from, to, .. }) if king.is_some() => {
                        let (df, dr) = (to.file - from.file, to.rank - from.rank);
                        let dist = df.abs().max(dr.abs());
                        if dist < 2 || (df != 0 && dr != 0 && df.abs() != dr.abs()) {
                            continue;
                        }
                        let unit = (df.signum(), dr.signum());
                        let pinnable = pin_dir(from)
                            .is_some_and(|d| d != unit && d != (-unit.0, -unit.1));
                        if pinnable {
                            continue;
                        }
                        blocked_beyond.push((from, unit, dist));
                        if dist == 2 {
                            occupied.insert(step(from, unit, 1));
                        }
                    }
                    _ => {}
                }
            }
        }

        let check = if view.you_in_check {
            Self::checker_hyps(view, known_opponent, mask_version)
        } else {
            None
        };
        let king_return = if mask_version >= 5 {
            check.as_ref().and_then(|(king, hyps)| {
                let [hyp] = hyps.as_slice() else { return None };
                // 取った駒自身の桂王手だけが残る。開き王手の可能性があれば使わない。
                if hyp.dir.is_none() && known_opponent == Some(hyp.square) {
                    Self::checked_king_origin(log, view.your_color, *king, hyp.square)
                } else {
                    None
                }
            })
        } else {
            None
        };
        Deduction {
            twins,
            occupied,
            blocked_beyond,
            check,
            king_return,
        }
    }

    fn checked_king_origin(log: &ObservationLog, color: Color, king: Coord, captured_at: Coord) -> Option<Coord> {
        let mut events = log.events().iter().rev();
        let accepted = |event: &&Observation| {
            matches!(event, Observation::MyMove { .. } | Observation::OpponentMoved { .. })
        };
        let Observation::OpponentMoved { move_number: capture_ply, .. } = events.find(accepted)? else {
            return None;
        };
        let Observation::MyMove { move_number: escape_ply, usi, captured: None } =
            events.find(accepted)? else {
                return None;
            };
        let ShogiMove::Board { from, to, promote: false } = parse_usi(usi)? else { return None };
        if to != king || from.rank == captured_at.rank || escape_ply.checked_add(1) != Some(*capture_ply) {
            return None;
        }

        // 戻る升と桂の捕獲先は段が違うので、その桂は前の王手駒ではない。
        // 捕獲先はもともと自駒で塞がっていたので、桂の到着で前の王手筋も遮れない。
        let mut was_checked = false;
        for event in events {
            match event {
                Observation::Check { in_check } if *in_check == color => was_checked = true,
                Observation::OpponentMoved { move_number, .. } => {
                    return (was_checked && move_number.checked_add(1) == Some(*escape_ply)).then_some(from);
                }
                Observation::MyMove { .. } => break,
                _ => {}
            }
        }
        None
    }

    /// 王手駒がいうるマスの集合（上の doc の H）。前提が崩れている（空になる）ときは None
    fn checker_hyps(
        view: &PlayerView,
        captured_at: Option<Coord>,
        mask_version: u32,
    ) -> Option<(Coord, Vec<CheckerHyp>)> {
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
        let opponent_roles = (mask_version >= 4 && captured_at.is_some())
            .then(|| possible_opponent_roles(view));
        let hyps = checker_hyps_on(
            &own, king, view.your_color, captured_at, mask_version >= 3, opponent_roles.as_deref(),
        );
        if hyps.is_empty() { None } else { Some((king, hyps)) }
    }

    fn rules_out(&self, usi: &str, mv: &ShogiMove) -> bool {
        if self.twins.contains(usi) {
            return true;
        }
        let on_occupied = self.occupied.iter().any(|&x| match *mv {
            ShogiMove::Drop { to, .. } => to == x,
            ShogiMove::Board { from, to, .. } => passes_through(from, to, x),
        });
        if on_occupied {
            return true;
        }
        if let ShogiMove::Board { from, to, .. } = *mv {
            let beyond = self.blocked_beyond.iter().any(|&(f, unit, dist)| {
                f == from && (2..=8).any(|k| k > dist && step(f, unit, k) == to)
            });
            if beyond {
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
        if is_king && self.king_return == Some(to) {
            return true;
        }
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

/// 王手駒がいうるマスの仮説（`Deduction` の doc の H）。自駒のマス `own`・自玉 `king`・王手した相手の手が取ったマス
/// `captured_at` を使う。行動マスクv4以降では `opponent_roles` で移動元を絞る。空なら前提が崩れている。
fn checker_hyps_on(
    own: &HashSet<Coord>,
    king: Coord,
    color: Color,
    captured_at: Option<Coord>,
    check_capture_paths: bool,
    opponent_roles: Option<&[Role]>,
) -> Vec<CheckerHyp> {
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
    let forward = if color == Color::Sente { -2 } else { 2 };
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
                    .find(|&(_, &o)| {
                        o != x
                            && reachable(o, x)
                            && (!check_capture_paths
                                || !own.iter().any(|&piece| passes_through(o, x, piece)))
                            && opponent_roles.is_none_or(|roles| {
                                roles.iter().any(|&role| role_reaches(role, color.other(), o, x))
                            })
                    });
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
    hyps
}

/// 王手を受けた時点の、王手駒がいうるマス（重複なし。並びは筋・段の昇順）。エンコーダが過去の王手についても
/// 同じ規則で候補を出すのに使う（`own` と `king` はその時点の自駒。開き王手の候補は自駒の遮りで絞る = マスク v3 と同じ）
pub(crate) fn checker_squares_at(
    own: &HashSet<Coord>,
    king: Coord,
    color: Color,
    captured_at: Option<Coord>,
) -> Vec<Coord> {
    let mut out: Vec<Coord> = checker_hyps_on(own, king, color, captured_at, true, None)
        .into_iter()
        .map(|h| h.square)
        .collect();
    out.sort_by_key(|c| (c.file, c.rank));
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::referee::{Referee, StepResult};
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn deduce(view: &PlayerView, log: &ObservationLog, tried: &HashSet<String>) -> Deduction {
        Deduction::new(view, log, tried, &candidate_usis(view).into_iter().collect(), MASK_VERSION)
    }

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
                let deduction = deduce(&view, log, tried);
                let legal = referee.position().legal_moves();
                for mv in &legal {
                    let usi = mv.to_usi();
                    assert!(
                        !deduction.rules_out(&usi, mv),
                        "局 {game}: 合法手 {usi} を除外した（王手中 {}・確定占有 {:?}・反則済み {tried:?}）",
                        view.you_in_check,
                        deduction.occupied
                    );
                    legal_checked += 1;
                }
                let basic = basic_legal_mask(&view, tried);
                let mask = legal_mask(&view, log, tried);
                ruled_out += (0..NUM_ACTIONS).filter(|&a| basic[a] && !mask[a]).count();
                if view.you_in_check {
                    check_states += 1;
                    let captured = log.events().iter().rev().find_map(|e| match e {
                        Observation::OpponentMoved {
                            captured_my_piece_at,
                            ..
                        } => Some(captured_my_piece_at.is_some()),
                        _ => None,
                    });
                    capture_checks += captured.unwrap_or(false) as usize;
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
                let deduction = deduce(&view, r.log(a.side), r.foul_tried(a.side));
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

    #[test]
    fn 捕獲先への経路が自駒に遮られる開き王手候補を除く() {
        let log = check_log(Some("4d"), Color::Gote);
        let tried = HashSet::new();
        let mut view = view_from_sfen(
            "l6nl/r2kg1gb1/np3sp1p/p1pp1B1p1/9/1PP3KPP/P2P2P2/2R3GS1/LNS1G2NL",
            Color::Gote,
            &[(Role::Pawn, 2)],
        );
        let escape = "6b5c";
        let action = encode_usi(escape, Color::Gote).unwrap();

        // 8四→4四は7四歩・6四歩を越せない。5三は4四から6二への王手線上。
        assert!(legal_mask_with_version(&view, &log, &tried, 2)[action]);
        assert!(!allowed(&view, &log, &tried, escape));

        // 8四飛→4四で9五角の王手が通る場合は、5三玉で逃げられる。
        view.your_pieces.retain(|piece| !matches!(piece.square.as_str(), "7d" | "6d"));
        assert!(allowed(&view, &log, &tried, escape));
    }

    #[test]
    fn 相手に飛車がないと分かるとき横へ飛ぶ開き王手の仮説を除く() {
        let log = check_log(Some("1c"), Color::Sente);
        let tried = HashSet::new();
        let view = view_from_sfen(
            "2g4n1/5Rs2/2pp2pkl/4Pp3/1ns1gbS2/6PpK/+l1PP1+l2P/R1G3g2/1N1+b2S1L",
            Color::Sente,
            &[(Role::Pawn, 1)],
        );
        let escape = "1f1e";
        let action = encode_usi(escape, Color::Sente).unwrap();
        // 第7局の104試行目。4三→1三へ横に走れる飛車は先手の4二と9八にある。
        assert!(legal_mask_with_version(&view, &log, &tried, 3)[action]);
        assert!(!allowed(&view, &log, &tried, escape));
        assert!(allowed(&view, &log, &tried, "1f2f"));

        // #165 の入力用王手候補は v3 のまま。除外器v4の仮説とはここで異なる。
        let own = view.your_pieces.iter().filter_map(|p| parse_usi_square(&p.square)).collect();
        let king = parse_usi_square("1f").unwrap();
        let captured_at = parse_usi_square("1c");
        let checker = parse_usi_square("5b").unwrap();
        let input_squares = checker_squares_at(&own, king, Color::Sente, captured_at);
        assert!(input_squares.contains(&checker));
        assert!(!Deduction::checker_hyps(&view, captured_at, 4).unwrap().1
            .iter().any(|hyp| hyp.square == checker));

        // 後手飛車が4三から1三へ取り、5二角の開き王手。1四の後手桂が飛車筋を遮る。
        let view = view_from_sfen(
            "k8/4bR3/8r/8n/9/6P1K/8P/9/9",
            Color::Sente,
            &[],
        );
        assert!(allowed(&view, &log, &tried, escape));
    }

    #[test]
    fn 盤上の馬と持駒の角で総数が尽きれば斜めに飛ぶ開き王手の仮説を除く() {
        let log = check_log(Some("1b"), Color::Sente);
        let tried = HashSet::new();
        let mut view = view_from_sfen(
            "1k7/5R2l/9/9/9/6P1K/8P/R8/1+B7",
            Color::Sente,
            &[(Role::Bishop, 1)],
        );
        let escape = "1f1e";
        let action = encode_usi(escape, Color::Sente).unwrap();
        // 3四→1二の斜め移動には角か馬が要る。両方とも先手の保有駒。
        assert!(legal_mask_with_version(&view, &log, &tried, 3)[action]);
        assert!(!allowed(&view, &log, &tried, escape));

        // 3四角→1二で4三角の筋が開いた局面では、1五玉が成立する。
        view = view_from_sfen("1k7/5R2b/5b3/9/9/6P1K/8P/R8/9", Color::Sente, &[]);
        assert!(allowed(&view, &log, &tried, escape));
    }

    #[test]
    fn 捕獲元の到達判定は成駒の利きと先後の向きを使う() {
        let from = parse_usi_square("5e").unwrap();
        let down = parse_usi_square("4g").unwrap();
        assert!(role_reaches(Role::Knight, Color::Gote, from, down));
        assert!(!role_reaches(Role::Knight, Color::Sente, from, down));
        let diagonal = parse_usi_square("4d").unwrap();
        assert!(role_reaches(Role::Dragon, Color::Sente, from, diagonal));
        assert!(!role_reaches(Role::Rook, Color::Sente, from, diagonal));
    }

    #[test]
    fn 駒を取った桂の王手でも直前の王手が残る升に玉を戻せない() {
        let view = view_from_sfen(
            "l2g4l/1ks2+Ps2/pp2+Np1p1/2Np4p/1N2B4/1PP1n4/PK1P+bPSPP/3+r1g3/L1SG4L",
            Color::Gote,
            &[(Role::Pawn, 1)],
        );
        let mut log = ObservationLog::default();
        log.record(Observation::OpponentMoved { move_number: 58, captured_my_piece_at: None });
        log.record(Observation::Check { in_check: Color::Gote });
        log.record(Observation::MyMove { move_number: 59, usi: "7c8b".into(), captured: None });
        log.record(Observation::OpponentMoved {
            move_number: 60, captured_my_piece_at: Some("7d".into()),
        });
        log.record(Observation::Check { in_check: Color::Gote });
        let tried = HashSet::new();
        let action = encode_usi("8b7c", Color::Gote).unwrap();
        // 第17局67試行目。旧マスクの候補に残った△7三玉だけがこの規則に当たる。
        assert!(legal_mask_with_version(&view, &log, &tried, 4)[action]);
        assert!(!allowed(&view, &log, &tried, "8b7c"));
        assert!(allowed(&view, &log, &tried, "8b9b"));
    }

    #[test]
    fn 前の王手駒が移動して合法になった升には玉を戻せる() {
        use crate::selfplay::StartState;
        use crate::shogi::{Piece, Position};

        let mut pos = Position::empty(Color::Sente);
        for (square, color, role) in [
            ("1i", Color::Sente, Role::King),
            ("6e", Color::Sente, Role::Rook),
            ("7c", Color::Gote, Role::King),
            ("8e", Color::Gote, Role::Pawn),
        ] {
            pos.set(parse_usi_square(square).unwrap(), Some(Piece { color, role }));
        }
        let mut referee = Referee::from_start(StartState {
            pos, logs: Default::default(), fouls: [0, 0], plies: 0,
        });
        // 7五飛で7三玉に王手、8二玉に逃げた後、同じ飛車が8五へ移って再び王手。
        for usi in ["6e7e", "7c8b", "7e8e"] {
            assert!(matches!(referee.step(usi, 0), StepResult::Accepted), "{usi}");
        }
        let side = Color::Gote;
        let view = referee.view(side, [0, 0], 0);
        assert!(view.you_in_check);
        assert!(referee.position().is_legal(&parse_usi("8b7c").unwrap()));
        assert!(allowed(&view, referee.log(side), referee.foul_tried(side), "8b7c"));
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

    /// 王手中でないときの歩以外の打ちの反則 → 打ち先に相手の駒がいると確定する
    #[test]
    fn 打ちの反則から打ち先の占有を確定させる() {
        let mut view = view_from_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL",
            Color::Sente,
            &[(Role::Gold, 1), (Role::Silver, 1), (Role::Pawn, 1)],
        );
        view.you_in_check = false;
        let log = ObservationLog::default();
        let tried: HashSet<String> = ["G*5e".to_string()].into();
        // 同じマスへの別の駒の打ちと、そこを飛び越える移動は確定反則
        assert!(!allowed(&view, &log, &tried, "S*5e"));
        // 歩の打ちの反則は打ち歩詰めがありうるので根拠にしない
        let tried_pawn: HashSet<String> = ["P*5e".to_string()].into();
        assert!(allowed(&view, &log, &tried_pawn, "S*5e"));
    }

    #[test]
    fn 打ち歩詰めを否定できる反則から飛車の途中の占有を確定する() {
        let mut view = view_from_sfen(
            "ln1g1g1nl/2s4k1/ppp1pp1pp/5bpR1/8P/3PPP3/PPP2KP2/3G2+n2/LN6L",
            Color::Sente,
            &[(Role::Rook, 1), (Role::Bishop, 1), (Role::Silver, 1), (Role::Pawn, 1)],
        );
        view.you_in_check = false;
        let mut log = ObservationLog::default();
        log.record(Observation::OpponentMoved { move_number: 57, captured_my_piece_at: None });
        let tried = ["B*2b".to_string(), "P*2c".to_string()].into();
        let action = encode_usi("2d2b", Color::Sente).unwrap();

        // 第7局63試行目。2三が空なら2四飛は2二に利くので、2三歩打は詰みになり得ない。
        assert!(legal_mask_with_version(&view, &log, &tried, 5)[action]);
        assert!(!allowed(&view, &log, &tried, "2d2b"));
        assert!(!allowed(&view, &log, &tried, "S*2c"));
        assert!(allowed(&view, &log, &tried, "2d2c+"));
        assert!(allowed(&view, &log, &tried, "2d2e"));
    }

    #[test]
    fn 打ち歩詰めの反則で空の打ち先を占有扱いしない() {
        use crate::selfplay::StartState;
        use crate::shogi::{Piece, Position};

        let mut pos = Position::empty(Color::Gote);
        for (square, color, role) in [
            ("1a", Color::Gote, Role::King),
            ("8c", Color::Gote, Role::Pawn),
            ("1c", Color::Sente, Role::Gold),
            ("2i", Color::Sente, Role::Rook),
            ("5i", Color::Sente, Role::King),
        ] {
            pos.set(parse_usi_square(square).unwrap(), Some(Piece { color, role }));
        }
        pos.set_hand(Color::Sente, Role::Pawn, 1);
        pos.set_hand(Color::Sente, Role::Lance, 1);
        let mut referee = Referee::from_start(StartState {
            pos, logs: Default::default(), fouls: [0, 0], plies: 0,
        });
        assert!(matches!(referee.step("8c8d", 0), StepResult::Accepted));
        assert!(matches!(referee.step("P*1b", 0), StepResult::Foul));
        assert!(referee.position().is_pseudo_legal(&parse_usi("P*1b").unwrap()));
        let side = Color::Sente;
        let view = referee.view(side, [0, 0], 0);
        assert!(referee.position().is_legal(&parse_usi("L*1b").unwrap()));
        assert!(allowed(&view, referee.log(side), referee.foul_tried(side), "L*1b"));
    }

    /// 飛び越え反則（ピンされえない駒）→ 同じ方向でそれより先は遮られる。間が1マスなら
    /// そのマスの占有が確定し、そこへの打ちも落ちる（ユーザー指摘）
    #[test]
    fn 飛び越え反則の先と間の1マスを落とす() {
        let mut view = view_from_sfen(
            "4k4/9/9/9/9/9/9/7R1/4K4",
            Color::Sente,
            &[(Role::Gold, 1)],
        );
        view.you_in_check = false;
        let log = ObservationLog::default();
        let tried: HashSet<String> = ["2h2d".to_string()].into();
        for usi in ["2h2c", "2h2b", "2h2a", "2h2c+", "2h2a+"] {
            assert!(!allowed(&view, &log, &tried, usi), "{usi} は遮られる");
        }
        for usi in ["2h2e", "2h2g", "2h5h"] {
            assert!(allowed(&view, &log, &tried, usi), "{usi} は残す");
        }
        // 間が1マス（2h→2f の間は 2g だけ）なら 2g の占有が確定する
        let tried2: HashSet<String> = ["2h2f".to_string()].into();
        assert!(!allowed(&view, &log, &tried2, "G*2g"));
        assert!(!allowed(&view, &log, &tried2, "2h2e"));
        assert!(allowed(&view, &log, &tried2, "G*2f"));
    }

    /// ピンされうる駒（玉の筋で最初の自駒）が筋から外れる飛び越え反則は、原因がピンかも
    /// しれないので推論しない
    #[test]
    fn ピンされうる駒の反則からは推論しない() {
        // 5九玉の真上 5八に飛車。横へ動く反則はピン（5筋の上に相手の飛び駒）でもありうる
        let mut view = view_from_sfen("4k4/9/9/9/9/9/9/4R4/4K4", Color::Sente, &[]);
        view.you_in_check = false;
        let log = ObservationLog::default();
        let tried: HashSet<String> = ["5h3h".to_string()].into();
        assert!(allowed(&view, &log, &tried, "5h2h"));
        // 筋に沿う動き（縦）の反則なら原因は遮りしかない
        let tried_up: HashSet<String> = ["5h5d".to_string()].into();
        assert!(!allowed(&view, &log, &tried_up, "5h5c"));
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
