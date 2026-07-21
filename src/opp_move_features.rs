//! 相手（人間）の指し手モデル用の特徴量。`src/bin/fit_opp.rs`（線形モデルの
//! 条件付きMLEフィット）と`src/bin/export_opp_move_data.rs`（NN学習データの
//! 書き出し）の両方から使う共通定義。
//!
//! **定義は estimator.rs の opp_move_weight 関連ヘルパ
//! （moved_is_minor/deep_unsupported/hangs_on_landing等）と一致させること**
//! （学習と推論の特徴量がズレると意味がない）。

use std::collections::HashSet;

use crate::board::Coord;
use crate::protocol::{Color, Role};
use crate::shogi::{Position, ShogiMove};

pub const OPP_MOVE_FEATURES: usize = 13;

pub const FEATURE_NAMES: [&str; OPP_MOVE_FEATURES] = [
    "advance",          // 前進量（段）
    "promote_minor",    // 成り（歩・香・桂）
    "promote_major",    // 成り（銀・角・飛）
    "is_drop",          // 持ち駒を打つ
    "threat_known", // 位置が既知の相手駒（自分の駒が死んだマス）へ新たに当たりを付ける
    "threat_home",  // 初期位置から動いていない相手駒へ新たに当たりを付ける
    "is_king_move", // 玉を動かす（基礎傾向）
    "king_flee",    // 玉が危険地点（自駒が死んだマス = 相手駒の露見地点）から遠ざかる
    "deep_unsup_pawn",  // 敵陣（3段）への紐なし着地（歩・香・桂）
    "deep_unsup_piece", // 敵陣（3段）への紐なし着地（銀以上の駒）
    "hang_minor", // 相手の利きがあるマスへの紐なし着地（歩・香・桂、取りは除く）
    "hang_major", // 同（銀以上）
    "opp_foul_count_this_turn", // この手番で相手が最終的な着手に至るまでに
    // 試みた反則の回数。反則の具体的な中身は「ついたて」の公平性上どちらの
    // プレイヤーにも相手には明かされないが、回数（Observation::OpponentFoul
    // のcount）は実戦でもリアルタイムに観測できる。反則を重ねた末の着手は
    // 探り直し・方針転換の産物であることが多いはずで、学習データ（真実の
    // foul_attempts）と実戦観測（累計countの差分）の両方から同じ値が求まる
];

pub fn advance_of(mv: &ShogiMove, mover: Color) -> f64 {
    match *mv {
        ShogiMove::Board { from, to, .. } => match mover {
            Color::Sente => (from.rank - to.rank) as f64,
            Color::Gote => (to.rank - from.rank) as f64,
        },
        ShogiMove::Drop { .. } => 0.0,
    }
}

pub fn to_square(mv: &ShogiMove) -> Coord {
    match *mv {
        ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
    }
}

/// 動かす駒種（移動前の役）。歩・香・桂を「小駒」とみなす
pub fn moved_is_minor(pos: &Position, mv: &ShogiMove) -> bool {
    let role = match *mv {
        ShogiMove::Board { from, .. } => pos.piece_at(from).map(|p| p.role),
        ShogiMove::Drop { role, .. } => Some(role),
    };
    matches!(role, Some(Role::Pawn | Role::Lance | Role::Knight))
}

/// 動かした駒（着地点 to）が対象マスのどれかへ新たに利きを付けたか。
/// 「新たに」= 移動元からは利いていなかった（打ちは常に新規）
pub fn newly_threatens(
    pos: &Position,
    next: &Position,
    mv: &ShogiMove,
    targets: &HashSet<Coord>,
) -> bool {
    let to = to_square(mv);
    targets.iter().any(|&s| {
        if s == to || !next.attacks(to, s) {
            return false;
        }
        match *mv {
            ShogiMove::Board { from, .. } => !pos.attacks(from, s),
            ShogiMove::Drop { .. } => true,
        }
    })
}

/// チェビシェフ距離（玉の歩数）
fn dist(a: Coord, b: Coord) -> i8 {
    (a.file - b.file).abs().max((a.rank - b.rank).abs())
}

/// 玉の移動が危険地点集合から遠ざかる手か（最近接距離が増える）
pub fn flees_danger(from: Coord, to: Coord, danger: &HashSet<Coord>) -> bool {
    let near = |sq: Coord| danger.iter().map(|&d| dist(sq, d)).min();
    match (near(from), near(to)) {
        (Some(a), Some(b)) => b > a,
        _ => false,
    }
}

/// 敵陣（成れる3段）への紐なし着地か。着地点に自分の別の駒の利きが無い
pub fn deep_unsupported(next: &Position, mv: &ShogiMove, mover: Color) -> bool {
    let to = to_square(mv);
    let deep = match mover {
        Color::Sente => to.rank <= 3,
        Color::Gote => to.rank >= 7,
    };
    deep && !next
        .pieces()
        .any(|(sq, p)| p.color == mover && sq != to && next.attacks(sq, to))
}

/// 相手の利きがあるマスへの紐なし着地か（取りは除く = 交換ではなく差し出し）
pub fn hangs_on_landing(pos: &Position, next: &Position, mv: &ShogiMove, mover: Color) -> bool {
    let to = to_square(mv);
    if pos.piece_at(to).is_some() {
        return false; // 取り（交換の文脈）は対象外
    }
    let opp = mover.other();
    let attacked = next
        .pieces()
        .any(|(sq, p)| p.color == opp && next.attacks(sq, to));
    attacked
        && !next
            .pieces()
            .any(|(sq, p)| p.color == mover && sq != to && next.attacks(sq, to))
}

/// 初期位置から一度も動いていない bot 駒のマス（相手はここを推論で狙ってくる）
pub fn home_squares(pos: &Position, bot: Color, bot_touched: &HashSet<Coord>) -> HashSet<Coord> {
    let initial = Position::initial();
    initial
        .pieces()
        .filter(|(sq, p)| {
            p.color == bot
                && !bot_touched.contains(sq)
                && pos.piece_at(*sq).is_some_and(|cur| cur.color == bot && cur.role == p.role)
        })
        .map(|(sq, _)| sq)
        .collect()
}

/// 候補手1つぶんの特徴量ベクトル。`pos`は着手前、`next`は着手後、
/// `known_squares`は位置が既知の相手駒（自分の駒が死んだマス）、
/// `homes`は初期位置から動いていない自分側の駒のマス、
/// `foul_count_this_turn`はこの手番で相手がここまでに試みた反則の回数
/// （候補手によらずこの手番内で共通の値。全候補行に同じ値が入る）
pub fn opp_move_features(
    pos: &Position,
    next: &Position,
    mv: &ShogiMove,
    mover: Color,
    known_squares: &HashSet<Coord>,
    homes: &HashSet<Coord>,
    foul_count_this_turn: u32,
) -> [f64; OPP_MOVE_FEATURES] {
    let (is_king, flee) = match *mv {
        ShogiMove::Board { from, to, .. } => {
            let is_king = pos.piece_at(from).is_some_and(|p| p.role == Role::King);
            (is_king, is_king && flees_danger(from, to, known_squares))
        }
        ShogiMove::Drop { .. } => (false, false),
    };
    let minor = moved_is_minor(pos, mv);
    let promotes = matches!(mv, ShogiMove::Board { promote: true, .. });
    let deep_unsup = deep_unsupported(next, mv, mover);
    let hang = hangs_on_landing(pos, next, mv, mover);
    [
        advance_of(mv, mover),
        (promotes && minor) as u8 as f64,
        (promotes && !minor) as u8 as f64,
        matches!(mv, ShogiMove::Drop { .. }) as u8 as f64,
        newly_threatens(pos, next, mv, known_squares) as u8 as f64,
        newly_threatens(pos, next, mv, homes) as u8 as f64,
        is_king as u8 as f64,
        flee as u8 as f64,
        (deep_unsup && minor) as u8 as f64,
        (deep_unsup && !minor) as u8 as f64,
        (hang && minor) as u8 as f64,
        (hang && !minor) as u8 as f64,
        f64::from(foul_count_this_turn),
    ]
}
