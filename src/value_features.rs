//! 局面（真の情報、両者視点）の数値特徴量。
//!
//! `likelihood.rs::particle_features` と同じ発想（手作り特徴量・名前付き配列）
//! だが、あちらは「粒子1個の尤もらしさ」を測る相手視点の特徴、こちらは
//! 「局面そのものの優劣」を測る両者視点の特徴。学習データ書き出し
//! （`bin/export_value_data`）専用で、真の `Position`（両者の駒配置が既知）
//! からのみ計算する。将来 evaluate() へ推論を統合するときも、特徴量の定義は
//! ここに一本化し、学習側（tsuitate-nn）とズレないようにする。

use crate::protocol::{Color, Role};
use crate::shogi::{Position, piece_value};
use crate::strategy::{drop_check_danger, king_zone_pressure};

pub const VALUE_FEATURES: usize = 14;

pub const VALUE_FEATURE_NAMES: [&str; VALUE_FEATURES] = [
    "material_diff",     // 自分の駒価値合計（盤上+持ち駒） − 相手の同値
    "my_hand_value",      // 自分の持ち駒価値合計
    "opp_hand_value",      // 相手の持ち駒価値合計
    "king_pressure_on_me", // 自玉周囲8マスへの相手の利き数
    "king_pressure_on_opp", // 相手玉周囲8マスへの自分の利き数
    "drop_check_danger_me", // 自玉への打ち込み王手の受け入れ面積（相手持ち駒基準）
    "drop_check_danger_opp", // 相手玉への同（自分の持ち駒基準）
    "my_in_check",        // 自分が王手されている
    "opp_in_check",        // 相手が王手されている
    "my_pieces_in_opp_camp", // 敵陣3段にいる自分の駒数（歩・玉除く）
    "opp_pieces_in_my_camp", // 自陣3段にいる相手の駒数（歩・玉除く）
    "my_max_hanging",      // 相手の利きが当たり自分の紐が無い自分の駒の最大価値
    "opp_max_hanging",      // 同、相手側（=自分が取れる駒の最大価値）
    "ply_progress",        // 手数を100で割った進行度（局面フェーズの粗い指標）
];

fn camp_rank_range(owner: Color) -> std::ops::RangeInclusive<i8> {
    // owner の敵陣（盤の奥3段）
    match owner {
        Color::Sente => 1..=3,
        Color::Gote => 7..=9,
    }
}

fn board_value(pos: &Position, color: Color) -> f64 {
    pos.pieces()
        .filter(|(_, p)| p.color == color)
        .map(|(_, p)| piece_value(p.role))
        .sum()
}

fn hand_value(pos: &Position, color: Color) -> f64 {
    pos.hand_map(color)
        .iter()
        .map(|(r, n)| piece_value(*r) * f64::from(*n))
        .sum()
}

fn material_sum(pos: &Position, color: Color) -> f64 {
    board_value(pos, color) + hand_value(pos, color)
}

/// `color` の駒（歩・玉除く）のうち、`color` から見た敵陣（盤の奥3段）に
/// いる枚数。攻め込みの深さ（自分が攻めているなら my_pieces、相手が攻めて
/// いるなら opp_pieces として呼ぶ）
fn pieces_in_enemy_camp(pos: &Position, color: Color) -> f64 {
    let range = camp_rank_range(color);
    pos.pieces()
        .filter(|(sq, p)| {
            p.color == color
                && !matches!(p.role, Role::Pawn | Role::Tokin | Role::King)
                && range.contains(&sq.rank)
        })
        .count() as f64
}

/// `color` の駒（玉除く）のうち、相手の利きが当たっていて自分の紐が無い
/// （取り返せない）駒の最大価値。33手目5八四金（scenarios/gold-check.kif）の
/// ような「利きが確定している駒への無防備な接近」を捉えるための特徴量
/// （元々の12特徴量にはこれが無く、まさに動機となった局面を判別できなかった）
fn max_hanging_value(pos: &Position, color: Color) -> f64 {
    let opp = color.other();
    pos.pieces()
        .filter(|(sq, p)| {
            p.color == color
                && p.role != Role::King
                && pos.is_attacked(*sq, opp)
                && !pos.is_attacked(*sq, color)
        })
        .map(|(_, p)| piece_value(p.role))
        .fold(0.0, f64::max)
}

/// 局面特徴量。`me` は評価する側（手番側とは限らない。学習データ書き出し側で
/// 手番ごとに `me` を指定して両方の視点を作れる）
pub fn value_features(pos: &Position, me: Color) -> [f64; VALUE_FEATURES] {
    let opp = me.other();
    [
        material_sum(pos, me) - material_sum(pos, opp),
        hand_value(pos, me),
        hand_value(pos, opp),
        king_zone_pressure(pos, me, opp),
        king_zone_pressure(pos, opp, me),
        drop_check_danger(pos, me),
        drop_check_danger(pos, opp),
        f64::from(pos.in_check(me)),
        f64::from(pos.in_check(opp)),
        pieces_in_enemy_camp(pos, me),
        pieces_in_enemy_camp(pos, opp),
        max_hanging_value(pos, me),
        max_hanging_value(pos, opp),
        f64::from(pos.move_number()) / 100.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Coord;
    use crate::shogi::parse_usi;

    #[test]
    fn initial_position_is_symmetric() {
        let pos = Position::initial();
        let sente = value_features(&pos, Color::Sente);
        let gote = value_features(&pos, Color::Gote);
        // 初期局面は完全対称: 駒得差・持ち駒価値・王手はどちらの視点でも0
        assert_eq!(sente[0], 0.0);
        assert_eq!(gote[0], 0.0);
        assert_eq!(sente[7], 0.0);
        assert_eq!(sente[8], 0.0);
        // 敵陣進出数も初期局面では0（自陣の駒を敵陣進出と誤カウントしないこと）
        assert_eq!(sente[9], 0.0, "my_pieces_in_opp_camp");
        assert_eq!(sente[10], 0.0, "opp_pieces_in_my_camp");
    }

    #[test]
    fn material_diff_and_hand_value_reflect_captures() {
        let mut pos = Position::initial();
        // kifu.rsのテストで合法性検証済みの手順（後手角を初期位置のまま角で取る）
        for usi in ["7g7f", "3a3b", "8h2b+"] {
            pos.play_unchecked(&parse_usi(usi).unwrap());
        }
        let f = value_features(&pos, Color::Sente);
        assert!(f[0] > 0.0, "角を取った直後は先手が駒得のはず: {}", f[0]);
        assert!(f[1] > 0.0, "取った角が先手の持ち駒に入っているはず: {}", f[1]);
        assert_eq!(f[2], 0.0, "後手の持ち駒はまだ空のはず: {}", f[2]);
        // 成った馬が2二（後手陣3段目）にいる = 先手の敵陣進出1
        assert_eq!(f[9], 1.0, "my_pieces_in_opp_camp: 馬が後手陣にいる");
        assert_eq!(f[10], 0.0, "opp_pieces_in_my_camp: 後手はまだ先手陣に駒がない");
    }

    #[test]
    fn ply_progress_increases_with_moves() {
        let mut pos = Position::initial();
        let before = value_features(&pos, Color::Sente)[13];
        pos.play_unchecked(&parse_usi("7g7f").unwrap());
        let after = value_features(&pos, Color::Sente)[13];
        assert!(after > before);
    }

    #[test]
    fn hanging_piece_is_detected() {
        // 33手目5八四金相当の最小再現: 自分の駒が紐なしで相手の利きに
        // 当たっている局面では my_max_hanging がその駒の価値になる
        let mut pos = Position::empty(Color::Sente);
        pos.set(
            Coord { file: 9, rank: 9 },
            Some(crate::shogi::Piece { color: Color::Sente, role: Role::King }),
        );
        pos.set(
            Coord { file: 1, rank: 1 },
            Some(crate::shogi::Piece { color: Color::Gote, role: Role::King }),
        );
        // 先手の金が4五に紐なしで浮いている
        pos.set(
            Coord { file: 4, rank: 5 },
            Some(crate::shogi::Piece { color: Color::Sente, role: Role::Gold }),
        );
        // 後手の金が4四から先手の金を直接攻撃（互いに向き合う）
        pos.set(
            Coord { file: 4, rank: 4 },
            Some(crate::shogi::Piece { color: Color::Gote, role: Role::Gold }),
        );
        // 後手の歩が4三から後手の金を守る（紐つき）
        pos.set(
            Coord { file: 4, rank: 3 },
            Some(crate::shogi::Piece { color: Color::Gote, role: Role::Pawn }),
        );
        let f = value_features(&pos, Color::Sente);
        assert_eq!(f[11], piece_value(Role::Gold), "my_max_hanging: 先手の金が浮いている");
        assert_eq!(f[12], 0.0, "opp_max_hanging: 後手の金は歩に守られていて紐つき");
    }
}
