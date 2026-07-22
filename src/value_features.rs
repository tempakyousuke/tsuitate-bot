//! 局面（真の情報、両者視点）の数値特徴量。
//!
//! `likelihood.rs::particle_features` と同じ発想（手作り特徴量・名前付き配列）
//! だが、あちらは「粒子1個の尤もらしさ」を測る相手視点の特徴、こちらは
//! 「局面そのものの優劣」を測る両者視点の特徴。学習データ書き出し
//! （`bin/export_value_data`）専用で、真の `Position`（両者の駒配置が既知）
//! からのみ計算する。将来 evaluate() へ推論を統合するときも、特徴量の定義は
//! ここに一本化し、学習側（tsuitate-nn）とズレないようにする。

use crate::protocol::{Color, Role};
use crate::shogi::{Position, ShogiMove, piece_value};
use crate::strategy::{drop_check_danger, exchange_value, king_zone_pressure};

pub const VALUE_FEATURES: usize = 16;

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
    "my_pieces_in_opp_camp", // 敵陣3段にいる自分の駒数（歩・と金・玉除く）
    "opp_pieces_in_my_camp", // 自陣3段にいる相手の駒数（歩・と金・玉除く）
    "my_max_hanging",      // 相手の利きが当たり自分の紐が無い自分の駒の最大価値
    "opp_max_hanging",      // 同、相手側（=自分が取れる駒の最大価値）
    "my_max_exchange_loss", // 相手に取られた場合の最悪交換損失（取り返しの補償を差し引いた後）
    "opp_max_exchange_loss", // 同、相手側（=自分が仕掛けられる最悪の交換損失）
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

/// `color` の駒（歩・と金・玉除く）のうち、`color` から見た敵陣（盤の奥3段）に
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

/// マス sq を攻撃している `by` 側の駒のうち、最も安い exchange_value（取り返す/
/// 取られる際に実際に使われるはずの駒。攻撃側は損を最小化するため最安の駒で
/// 取る）。1枚も無ければ None
///
/// 近似: `attacks()`（利きの有無）だけを見ており、ピンで動けない駒や
/// 取ると自玉が王手になる駒も攻撃駒に数える（既存の`max_hanging_value`と
/// 同じ近似方針）。厳密な合法性チェックは局面ごとに指し手を構築する必要があり
/// コストが高いため、学習データの特徴量としては許容範囲としている
/// （codexレビュー指摘、2026-07-20。pairwiseの教師信号としてのノイズ源になる
/// 可能性は残る）
fn min_attacker_exchange_value(pos: &Position, sq: crate::board::Coord, by: Color) -> Option<f64> {
    pos.pieces()
        .filter(|(from, p)| p.color == by && pos.attacks(*from, sq))
        .map(|(_, p)| exchange_value(p.role))
        .fold(None, |acc: Option<f64>, v| Some(acc.map_or(v, |a| a.min(v))))
}

/// `color` の駒（歩・と金・玉除く。歩は打ち歩詰め等の特殊性が強く exchange_value の
/// 前提が崩れやすいため除外）のうち、相手に取られた場合の最悪の交換損失
/// （取り返せるなら相手の攻め駒の exchange_value を補償として差し引く）。
/// kakudo局面（scenarios/kakudo.kif、R*2d vs P*2h）のような「取られる駒の
/// 価値の高さ」を、single hangingでは表現できない紐つき交換でも捉えるための特徴量
/// （2026-07-20、codexレビュー指摘: max_hanging_valueは紐なしの即取りしか
/// 表せず、飛車を切って角を得る/歩を切って角を得るの損得差を区別できない）
fn max_exchange_loss(pos: &Position, color: Color) -> f64 {
    let opp = color.other();
    pos.pieces()
        .filter(|(_, p)| p.color == color && !matches!(p.role, Role::King | Role::Pawn))
        .filter_map(|(sq, p)| {
            // 相手は損を最小化するため最安の攻め駒で取ってくる想定
            let attacker = min_attacker_exchange_value(pos, sq, opp)?;
            let loss = exchange_value(p.role);
            // 取り返せる（sq を自分の他の駒も攻撃している）なら、取り返して
            // 得る相手の攻め駒の価値ぶんを補償として差し引く
            let can_recapture = min_attacker_exchange_value(pos, sq, color).is_some();
            let comp = if can_recapture { attacker } else { 0.0 };
            Some((loss - comp).max(0.0))
        })
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
        max_exchange_loss(pos, me),
        max_exchange_loss(pos, opp),
        f64::from(pos.move_number()) / 100.0,
    ]
}

pub const TRANSITION_FEATURES: usize = 6;

pub const TRANSITION_FEATURE_NAMES: [&str; TRANSITION_FEATURES] = [
    "moved_piece_value",           // 直前に着手された駒（動いた/打たれた駒）の価値
    "moved_piece_hanging_value",   // 同、紐なしで即取られる状態なら価値、そうでなければ0
    "moved_piece_exchange_loss",   // 同、紐つきでも駒種の交換で損する額（取り返しの補償を差し引いた後）
    "captured_value",              // その着手で取った相手駒の価値（打つ手・非取りなら0）
    "net_capture_then_recapture",  // captured_value − moved_piece_exchange_loss（この一手の実質損得）
    "gives_check",                 // その着手で相手に王手をかけたか
];

/// 直前の着手（`mv`）固有の特徴量。`max_hanging_value`/`max_exchange_loss`は
/// 盤面全体でのworst-caseを返すため、無関係などこか別の駒のリスクが大きいと
/// その着手自体が生むリスクの差がmaxに埋もれて消える（kakudo局面 R*2d vs P*2h
/// で実際に発生・codexレビューで指摘、2026-07-20）。この関数は着手で動いた/
/// 打たれた駒**だけ**に絞ることでその埋没を避ける。`mover` は着手した側
pub fn transition_features(
    before: &Position,
    mv: &ShogiMove,
    after: &Position,
    mover: Color,
) -> [f64; TRANSITION_FEATURES] {
    let opp = mover.other();
    let to = match *mv {
        ShogiMove::Board { to, .. } => to,
        ShogiMove::Drop { to, .. } => to,
    };
    let moved_role = after
        .piece_at(to)
        .expect("着手直後は to に自駒があるはず")
        .role;
    let moved_value = piece_value(moved_role);

    let hanging = if after.is_attacked(to, opp) && !after.is_attacked(to, mover) {
        moved_value
    } else {
        0.0
    };

    let exchange_loss = min_attacker_exchange_value(after, to, opp).map_or(0.0, |attacker| {
        let loss = exchange_value(moved_role);
        let can_recapture = min_attacker_exchange_value(after, to, mover).is_some();
        let comp = if can_recapture { attacker } else { 0.0 };
        (loss - comp).max(0.0)
    });

    // exchange_value に揃える（captured_value - exchange_loss = net の両辺が
    // 同じ「持ち駒化後の実質価値」基準でないと差し引きの意味がズレる。
    // codexレビュー指摘、2026-07-20: ここだけpiece_valueのままだと、と金等
    // 成駒を取った際の得を過大評価し、net_capture_then_recaptureが歪む）
    let captured_value = match *mv {
        ShogiMove::Board { to, .. } => before.piece_at(to).map_or(0.0, |p| exchange_value(p.role)),
        ShogiMove::Drop { .. } => 0.0,
    };

    [
        moved_value,
        hanging,
        exchange_loss,
        captured_value,
        captured_value - exchange_loss,
        f64::from(after.in_check(opp)),
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
        let before = value_features(&pos, Color::Sente)[15];
        pos.play_unchecked(&parse_usi("7g7f").unwrap());
        let after = value_features(&pos, Color::Sente)[15];
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

    #[test]
    fn defended_piece_still_shows_exchange_loss() {
        // kakudo局面相当の最小再現: 飛車が「紐つき」（取り返せる）なので
        // my_max_hangingは0だが、取り返しても金と飛車の交換では駒種で損する
        // （my_max_exchange_lossはその損失=9.5-5.5=4.0を検出するはず）
        let mut pos = Position::empty(Color::Sente);
        pos.set(
            Coord { file: 9, rank: 9 },
            Some(crate::shogi::Piece { color: Color::Sente, role: Role::King }),
        );
        pos.set(
            Coord { file: 1, rank: 1 },
            Some(crate::shogi::Piece { color: Color::Gote, role: Role::King }),
        );
        // 先手の飛車が4五、後手の金が4四から攻撃
        pos.set(
            Coord { file: 4, rank: 5 },
            Some(crate::shogi::Piece { color: Color::Sente, role: Role::Rook }),
        );
        pos.set(
            Coord { file: 4, rank: 4 },
            Some(crate::shogi::Piece { color: Color::Gote, role: Role::Gold }),
        );
        // 先手の歩が4六から飛車を守る（紐つき。金を取り返せる）
        pos.set(
            Coord { file: 4, rank: 6 },
            Some(crate::shogi::Piece { color: Color::Sente, role: Role::Pawn }),
        );
        let f = value_features(&pos, Color::Sente);
        assert_eq!(f[11], 0.0, "my_max_hanging: 飛車は紐つきなのでハングではない");
        assert!(
            (f[13] - 4.0).abs() < 1e-9,
            "my_max_exchange_loss: 飛車(9.5)を切られ金(5.5)を取り返しても4.0損: {}",
            f[13]
        );
    }
}
