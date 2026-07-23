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

pub const OPP_MOVE_FEATURES: usize = 24;

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
    //
    // --- ここから駒種特化ブロック（NN段階①-b、2026-07-22）。動かす駒の
    // 駒種one-hot（玉は既存is_king_moveがあるので全ゼロ）・成駒フラグ・
    // 移動距離・初期配置マスからの移動。狙いは (1) kakutoriで露呈した
    // 「角・飛の長距離移動」の表現力欠如（12特徴量では駒種を区別できず
    // 真の王手駒への信念が立たない）、(2) home_lance_move（未動の隅香車
    // 割り引き）の駒種横断への一般化（「未観測の駒は初期配置のまま」の原則）
    "moved_pawn",
    "moved_lance",
    "moved_knight",
    "moved_silver",
    "moved_gold",
    "moved_bishop",
    "moved_rook",
    "moved_promoted", // 動かす駒が成駒（one-hotは成る前の駒種側に立つ）
    "move_dist",      // 移動距離（チェビシェフ。打ちは0）
    "from_home",      // その駒種の初期配置マスからの移動（未動駒の近似）
    "my_foul_count_last_turn", // 直前の相手（=このモデルから見た敵側）手番で
    // 相手が試みた反則の回数。opp_foul_count_this_turn の逆方向版
    // （2026-07-23）: 敵の反則宣言は指し手側も観測できる（回数のみ・中身は
    // 不明）。敵が反則を重ねた直後は「敵はこちらの駒配置を読み違えている」
    // シグナルであり、突撃・様子見などの方針変化が学習できるはず。
    // v9凍結（反則に反応する教師の自己対局データが取れるようになった）で
    // ブートストラップ依存が解消されたため追加
];

/// 駒種特化ブロック（末尾10特徴量）。one-hotは成る前の駒種（unpromote）で
/// 立て、玉は全ゼロ（既存のis_king_moveが担う）。打ちはone-hotのみ
/// （dist=0, from_home=false, promoted=false）
pub fn piece_type_features(pos: &Position, mv: &ShogiMove, mover: Color) -> [f64; 10] {
    let (role_raw, dist, from_home) = match *mv {
        ShogiMove::Board { from, to, .. } => {
            let Some(p) = pos.piece_at(from) else {
                return [0.0; 10];
            };
            let dist = (from.file - to.file).abs().max((from.rank - to.rank).abs());
            (p.role, f64::from(dist), is_home_square(p.role, mover, from))
        }
        ShogiMove::Drop { role, .. } => (role, 0.0, false),
    };
    let base = crate::shogi::unpromote_role(role_raw);
    let one_hot = |r: Role| (base == r) as u8 as f64;
    [
        one_hot(Role::Pawn),
        one_hot(Role::Lance),
        one_hot(Role::Knight),
        one_hot(Role::Silver),
        one_hot(Role::Gold),
        one_hot(Role::Bishop),
        one_hot(Role::Rook),
        (role_raw != base) as u8 as f64,
        dist,
        from_home as u8 as f64,
    ]
}

/// マス sq がその駒種（成っていない駒）の初期配置マスか。
/// 「まだ初期配置マスに立っている＝未動」の近似（実際は一度動いて戻った
/// 可能性もあるが、旧home_lance_moveと同じ近似を全駒種へ一般化した）
pub fn is_home_square(role: Role, mover: Color, sq: Coord) -> bool {
    let home_rank = |sente: i8, gote: i8| match mover {
        Color::Sente => sente,
        Color::Gote => gote,
    };
    match role {
        Role::Pawn => sq.rank == home_rank(7, 3),
        Role::Lance => sq.rank == home_rank(9, 1) && (sq.file == 1 || sq.file == 9),
        Role::Knight => sq.rank == home_rank(9, 1) && (sq.file == 2 || sq.file == 8),
        Role::Silver => sq.rank == home_rank(9, 1) && (sq.file == 3 || sq.file == 7),
        Role::Gold => sq.rank == home_rank(9, 1) && (sq.file == 4 || sq.file == 6),
        Role::King => sq.rank == home_rank(9, 1) && sq.file == 5,
        Role::Bishop => match mover {
            Color::Sente => sq.file == 8 && sq.rank == 8,
            Color::Gote => sq.file == 2 && sq.rank == 2,
        },
        Role::Rook => match mover {
            Color::Sente => sq.file == 2 && sq.rank == 8,
            Color::Gote => sq.file == 8 && sq.rank == 2,
        },
        _ => false, // 成駒は初期配置に存在しない
    }
}

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
/// （候補手によらずこの手番内で共通の値。全候補行に同じ値が入る）、
/// `my_foul_count_last_turn`は直前の敵側（moverの相手）手番での反則回数
/// （同じく全候補行で共通）
pub fn opp_move_features(
    pos: &Position,
    next: &Position,
    mv: &ShogiMove,
    mover: Color,
    known_squares: &HashSet<Coord>,
    homes: &HashSet<Coord>,
    foul_count_this_turn: u32,
    my_foul_count_last_turn: u32,
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
    let pt = piece_type_features(pos, mv, mover);
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
        pt[0],
        pt[1],
        pt[2],
        pt[3],
        pt[4],
        pt[5],
        pt[6],
        pt[7],
        pt[8],
        pt[9],
        f64::from(my_foul_count_last_turn),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shogi::parse_usi;

    fn board(from: (i8, i8), to: (i8, i8)) -> ShogiMove {
        ShogiMove::Board {
            from: Coord { file: from.0, rank: from.1 },
            to: Coord { file: to.0, rank: to.1 },
            promote: false,
        }
    }

    /// 初期配置マス判定が Position::initial の実配置と全マス・全駒種で一致する
    /// （ハードコードした座標のずれをエンジン側の初期局面で照合する）
    #[test]
    fn is_home_square_matches_initial_position() {
        let initial = Position::initial();
        for color in [Color::Sente, Color::Gote] {
            for file in 1..=9i8 {
                for rank in 1..=9i8 {
                    let sq = Coord { file, rank };
                    for role in [
                        Role::Pawn,
                        Role::Lance,
                        Role::Knight,
                        Role::Silver,
                        Role::Gold,
                        Role::Bishop,
                        Role::Rook,
                        Role::King,
                    ] {
                        let expect = initial
                            .piece_at(sq)
                            .is_some_and(|p| p.color == color && p.role == role);
                        assert_eq!(
                            is_home_square(role, color, sq),
                            expect,
                            "role={role:?} color={color:?} sq={sq:?}"
                        );
                    }
                }
            }
        }
    }

    /// 旧home_lance_move相当のケース: 隅の香車の初手は lance one-hot + from_home が
    /// 同時に立つ（NNがこの組で旧-1.3割り引きを表現できることの前提）
    #[test]
    fn piece_type_features_flags_home_lance() {
        let pos = Position::initial();
        let f = piece_type_features(&pos, &board((1, 1), (1, 2)), Color::Gote);
        assert_eq!(f[1], 1.0, "lance one-hot");
        assert_eq!(f[9], 1.0, "from_home");
        assert_eq!(f[8], 1.0, "dist=1");
        // 桂馬マスからの手は香車扱いしない（後手2一は桂馬）
        let g = piece_type_features(&pos, &board((2, 1), (1, 3)), Color::Gote);
        assert_eq!(g[1], 0.0);
        assert_eq!(g[2], 1.0, "knight one-hot");
        assert_eq!(g[9], 1.0, "桂馬も初期配置マスからなら from_home");
    }

    /// 長距離移動の距離と、打ち・玉・成駒の扱い
    #[test]
    fn piece_type_features_dist_drop_and_king() {
        let mut pos = Position::initial();
        // 角道を開けて後手角が2二→8八の長距離移動（dist=6）
        for usi in ["7g7f", "3c3d", "2g2f"] {
            pos.play_unchecked(&parse_usi(usi).unwrap());
        }
        let f = piece_type_features(&pos, &board((2, 2), (8, 8)), Color::Gote);
        assert_eq!(f[5], 1.0, "bishop one-hot");
        assert_eq!(f[8], 6.0, "dist");
        assert_eq!(f[9], 1.0, "2二は後手角の初期配置マス");
        // 打ちは one-hot のみ（dist=0, from_home=0）
        let drop = ShogiMove::Drop { role: Role::Pawn, to: Coord { file: 5, rank: 5 } };
        let d = piece_type_features(&pos, &drop, Color::Gote);
        assert_eq!(d[0], 1.0);
        assert_eq!(d[8], 0.0);
        assert_eq!(d[9], 0.0);
        // 玉は one-hot 全ゼロ（is_king_move が既存特徴量にある）だが from_home は立つ
        let k = piece_type_features(&pos, &board((5, 1), (4, 2)), Color::Gote);
        assert_eq!(&k[..8], &[0.0; 8], "玉はone-hotなし・成駒フラグOFF");
        assert_eq!(k[9], 1.0, "5一は後手玉の初期配置マス");
    }
}
