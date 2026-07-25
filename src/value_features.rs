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

/// 局面側の特徴量数（既存16 + 接近4）。列順は
/// [既存16, 接近4] で固定する（学習CSV・推論の組み立ての両方がこの順）
pub const VALUE_FEATURES: usize = 20;

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
    // 接近特徴量（2026-07-25追加。詳細は APPROACH_DIST_CAP 付近のdocコメント）
    "my_prey_proximity",  // 自分の駒から見た「価値×近さ」が最大の敵駒
    "opp_prey_proximity", // 同、相手側
    "my_king_siege",      // 相手玉へ寄っている自分の駒の 価値×近さ の総和
    "opp_king_siege",     // 同、相手側（自玉が寄せられている度合い）
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
    let approach = approach_state_features(pos, me);
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
        approach[0],
        approach[1],
        approach[2],
        approach[3],
    ]
}

/// 接近特徴量（NN段階③の追補、2026-07-25）。
///
/// 動機: 手作り評価にもvalueネットにも「**多手かけて敵駒・敵玉へ近づく**」を
/// 測る量が無い。前向きの項は `threat_value`（既に当たっている＝1手先）・
/// `king_zone_pressure`（隣接8マスのみ）・`knight_bait_value`（歩→桂だけの
/// BFS距離）で、距離勾配を持つのは最後の1つ、しかも駒種特化。その結果
/// 「取られないが何の脅威にもならない手」と「駒を寄せて次に取りに行く手」が
/// 同じ評価になる（scenarios/watch-estimator-vs-estimator-20260725-203544.kif
/// の34手目 8八歩が発端）。
///
/// ここでは `deduce::distance_empty_board`（空盤BFS＝手数の admissible な下限。
/// (駒種,色,出発マス) でメモ化済み）で距離を測り、`knight_bait_value` の
/// 駒種一般化にあたる量を state / transition の両方に足す。
/// **遮蔽を見ない下限**なので「近いのに実は届かない」を過大評価するが、
/// 学習側が重みを決めるので手作り項ほど致命的ではない（同じ近似方針は
/// max_hanging_value 等の既存特徴量でも採っている）。
///
/// 距離が定義できない（到達不能）ときの打ち切り手数
const APPROACH_DIST_CAP: u32 = 8;
/// 1手ぶんの減衰。距離1（空盤なら次の手でそのマスへ行ける）を 1.0 とする
const APPROACH_DECAY: f64 = 0.6;

/// `from` にいる `role`（成駒ならその成駒自身を base に渡す）が `target` へ
/// 到達するのに要する空盤上の最短手数。成れる駒は成った経路も含めた最短
fn moves_to_reach(
    role: Role,
    color: Color,
    from: crate::board::Coord,
    target: crate::board::Coord,
) -> Option<u32> {
    let plain = crate::deduce::distance_empty_board(role, color, from, target, false);
    let promoted = crate::deduce::distance_empty_board(role, color, from, target, true);
    plain.into_iter().chain(promoted).min()
}

/// 距離を [0,1] の「近さ」へ落とす。距離1 = 1.0（次の手で取れる位置）
fn closeness(dist: Option<u32>) -> f64 {
    let d = dist.unwrap_or(APPROACH_DIST_CAP).min(APPROACH_DIST_CAP);
    APPROACH_DECAY.powi(d.saturating_sub(1) as i32)
}

/// 紐つき（相手が守っている）獲物の割引率。取っても取り返されるので、
/// 寄せる価値はそのぶん低い（threat_value の THREAT_DEFENDED_DISCOUNT と同じ発想）
const PREY_DEFENDED_DISCOUNT: f64 = 0.45;

/// `from` にいる `role` の駒から見た「獲物の近さ」= 敵駒（玉除く）のうち
/// `価値 × 近さ` が最大のもの。**これが1手先だけを見る threat_value の一般化**で、
/// 距離2・3の獲物にも減衰した価値がつくので、寄せていく途中の手に勾配が立つ。
/// 紐つきの獲物は割り引く（2026-07-26 修正。当初は生の価値を使っていたが、
/// 「取り返される獲物へ寄る価値」を過大評価していた）
fn prey_proximity_from(
    pos: &Position,
    me: Color,
    role: Role,
    from: crate::board::Coord,
) -> f64 {
    let opp = me.other();
    pos.pieces()
        .filter(|(_, p)| p.color == opp && p.role != Role::King)
        .map(|(sq, p)| {
            let defended = if pos.is_attacked(sq, opp) { PREY_DEFENDED_DISCOUNT } else { 1.0 };
            exchange_value(p.role) * defended * closeness(moves_to_reach(role, me, from, sq))
        })
        .fold(0.0, f64::max)
}

/// `me` の駒（玉除く）全体で見た獲物の近さの最大値
fn prey_proximity(pos: &Position, me: Color) -> f64 {
    pos.pieces()
        .filter(|(_, p)| p.color == me && p.role != Role::King)
        .map(|(sq, p)| prey_proximity_from(pos, me, p.role, sq))
        .fold(0.0, f64::max)
}

/// `me` の駒（玉除く）が相手玉へどれだけ集まっているか（価値×近さの総和）。
/// king_zone_pressure が隣接8マスしか見ないのに対し、こちらは寄せの途中の
/// 「まだ届いていないが向かっている駒」も数える
fn king_siege(pos: &Position, me: Color) -> f64 {
    let Some(king) = pos.king_square(me.other()) else {
        return 0.0;
    };
    pos.pieces()
        .filter(|(_, p)| p.color == me && p.role != Role::King)
        .map(|(sq, p)| exchange_value(p.role) * closeness(moves_to_reach(p.role, me, sq, king)))
        .sum()
}

pub const APPROACH_STATE_FEATURES: usize = 4;

pub const APPROACH_STATE_FEATURE_NAMES: [&str; APPROACH_STATE_FEATURES] = [
    "my_prey_proximity",  // 自分の駒から見た「価値×近さ」が最大の敵駒
    "opp_prey_proximity", // 同、相手側
    "my_king_siege",      // 相手玉へ寄っている自分の駒の 価値×近さ の総和
    "opp_king_siege",     // 同、相手側（自玉が寄せられている度合い）
];

/// 局面側の接近特徴量。`value_features` と同じ規約（`me` 視点）
pub fn approach_state_features(pos: &Position, me: Color) -> [f64; APPROACH_STATE_FEATURES] {
    let opp = me.other();
    [
        prey_proximity(pos, me),
        prey_proximity(pos, opp),
        king_siege(pos, me),
        king_siege(pos, opp),
    ]
}

pub const APPROACH_TRANSITION_FEATURES: usize = 5;

pub const APPROACH_TRANSITION_FEATURE_NAMES: [&str; APPROACH_TRANSITION_FEATURES] = [
    "king_approach",       // この手で着手駒が相手玉へ縮めた手数（打ちは0）
    "prey_approach",       // この手で着手駒の「獲物の近さ」が増えた量（打ちは0）
    "king_closeness_after", // 着手後の着手駒→相手玉の近さ（打ちも定義できる）
    // 「紐をつけながら寄せる」の観点（2026-07-26、人間レビュー指摘）。
    // 将棋では単に近づくのではなく、取られても取り返せる形で寄せることが要る。
    // 実測（1536局・静かな手7.7万件）でも接近の実現利得は駒種で符号が逆転して
    // いた（歩 -1.10 / 香桂銀 -0.64 / 金飛角と成駒 +1.18）ので、裸の距離だけでは
    // 足りない
    "moved_piece_supported", // 着手後、着地マスを自分の他の駒が守っているか
    "supported_prey_approach", // 紐つきで寄せたぶんだけの prey_approach
];

/// 着手側の接近特徴量。`transition_features` と同じ規約（`mover` が指した側）。
/// 盤上手は from/to の差分、打つ手は差分を持たないので 0 とし、
/// 着手後の近さ（`king_closeness_after`）だけで表現する
pub fn approach_transition_features(
    before: &Position,
    mv: &ShogiMove,
    after: &Position,
    mover: Color,
) -> [f64; APPROACH_TRANSITION_FEATURES] {
    let to = match *mv {
        ShogiMove::Board { to, .. } => to,
        ShogiMove::Drop { to, .. } => to,
    };
    let role_after = after
        .piece_at(to)
        .expect("着手直後は to に自駒があるはず")
        .role;
    let opp_king = after.king_square(mover.other());

    let dist_after = opp_king.and_then(|k| moves_to_reach(role_after, mover, to, k));
    let (king_approach, prey_approach) = match *mv {
        ShogiMove::Board { from, .. } => {
            let role_before = before
                .piece_at(from)
                .expect("着手前は from に自駒があるはず")
                .role;
            let dist_before = opp_king.and_then(|k| moves_to_reach(role_before, mover, from, k));
            let d_before = dist_before.unwrap_or(APPROACH_DIST_CAP).min(APPROACH_DIST_CAP);
            let d_after = dist_after.unwrap_or(APPROACH_DIST_CAP).min(APPROACH_DIST_CAP);
            (
                f64::from(d_before) - f64::from(d_after),
                prey_proximity_from(after, mover, role_after, to)
                    - prey_proximity_from(before, mover, role_before, from),
            )
        }
        ShogiMove::Drop { .. } => (0.0, 0.0),
    };
    // 紐 = 着地マスを自分の**他の**駒が守っているか（自分自身は自マスを攻撃しない）
    let supported = f64::from(after.is_attacked(to, mover));
    [
        king_approach,
        prey_approach,
        closeness(dist_after),
        supported,
        if supported > 0.0 { prey_approach } else { 0.0 },
    ]
}

/// 着手側の特徴量数（既存6 + 接近5）。列順は [既存6, 接近5]
pub const TRANSITION_FEATURES: usize = 11;

pub const TRANSITION_FEATURE_NAMES: [&str; TRANSITION_FEATURES] = [
    "moved_piece_value",           // 直前に着手された駒（動いた/打たれた駒）の価値
    "moved_piece_hanging_value",   // 同、紐なしで即取られる状態なら価値、そうでなければ0
    "moved_piece_exchange_loss",   // 同、紐つきでも駒種の交換で損する額（取り返しの補償を差し引いた後）
    "captured_value",              // その着手で取った相手駒の価値（打つ手・非取りなら0）
    "net_capture_then_recapture",  // captured_value − moved_piece_exchange_loss（この一手の実質損得）
    "gives_check",                 // その着手で相手に王手をかけたか
    "king_approach",               // この手で着手駒が相手玉へ縮めた手数（打ちは0）
    "prey_approach",               // この手で着手駒の「獲物の近さ」が増えた量（打ちは0）
    "king_closeness_after",        // 着手後の着手駒→相手玉の近さ（打ちも定義できる）
    "moved_piece_supported",       // 着手後、着地マスを自分の他の駒が守っているか
    "supported_prey_approach",     // 紐つきで寄せたぶんだけの prey_approach
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

    let approach = approach_transition_features(before, mv, after, mover);
    [
        moved_value,
        hanging,
        exchange_loss,
        captured_value,
        captured_value - exchange_loss,
        f64::from(after.in_check(opp)),
        approach[0],
        approach[1],
        approach[2],
        approach[3],
        approach[4],
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

    fn put(pos: &mut Position, file: i8, rank: i8, color: Color, role: Role) {
        pos.set(
            Coord { file, rank },
            Some(crate::shogi::Piece { color, role }),
        );
    }

    /// 玉だけの空盤（接近特徴量のテスト用。玉が無いと king_siege が0になる）
    fn kings_only() -> Position {
        let mut pos = Position::empty(Color::Sente);
        put(&mut pos, 5, 9, Color::Sente, Role::King);
        put(&mut pos, 5, 1, Color::Gote, Role::King);
        pos
    }

    #[test]
    fn king_approach_counts_moves_saved_toward_enemy_king() {
        let mut pos = kings_only();
        // 先手の金5五 → 5四は敵玉5一へ1手ぶん近づく
        put(&mut pos, 5, 5, Color::Sente, Role::Gold);
        let mv = parse_usi("5e5d").unwrap();
        let mut after = pos.clone();
        after.play_unchecked(&mv);
        let f = approach_transition_features(&pos, &mv, &after, Color::Sente);
        assert_eq!(f[0], 1.0, "king_approach: 1手ぶん縮む");
        // 遠ざかる手は負
        let back = parse_usi("5e5f").unwrap();
        let mut after_back = pos.clone();
        after_back.play_unchecked(&back);
        let g = approach_transition_features(&pos, &back, &after_back, Color::Sente);
        assert_eq!(g[0], -1.0, "king_approach: 遠ざかると負");
        // 近いほうが king_closeness_after は大きい
        assert!(f[2] > g[2]);
    }

    #[test]
    fn prey_approach_rewards_closing_on_enemy_pieces() {
        let mut pos = kings_only();
        put(&mut pos, 5, 5, Color::Sente, Role::Gold);
        // 敵の飛車を9三に置く（金からは遠い）
        put(&mut pos, 9, 3, Color::Gote, Role::Rook);
        let toward = parse_usi("5e6d").unwrap();
        let mut after = pos.clone();
        after.play_unchecked(&toward);
        let f = approach_transition_features(&pos, &toward, &after, Color::Sente);
        assert!(f[1] > 0.0, "獲物へ近づく手は prey_approach が正: {}", f[1]);
    }

    #[test]
    fn drops_have_no_delta_but_keep_closeness() {
        let pos = kings_only();
        let mv = parse_usi("G*5b").unwrap();
        let mut after = pos.clone();
        after.play_unchecked(&mv);
        let f = approach_transition_features(&pos, &mv, &after, Color::Sente);
        assert_eq!(f[0], 0.0, "打つ手に差分は無い");
        assert_eq!(f[1], 0.0);
        assert!(f[2] > 0.5, "敵玉の隣に打てば近さは最大級: {}", f[2]);
    }

    #[test]
    fn king_siege_grows_as_pieces_close_in() {
        let mut far = kings_only();
        put(&mut far, 5, 8, Color::Sente, Role::Gold);
        let mut near = kings_only();
        put(&mut near, 5, 2, Color::Sente, Role::Gold);
        let f_far = approach_state_features(&far, Color::Sente);
        let f_near = approach_state_features(&near, Color::Sente);
        assert!(
            f_near[2] > f_far[2],
            "my_king_siege: 寄っているほうが大きい {} vs {}",
            f_near[2],
            f_far[2]
        );
        // 相手視点では鏡像（自玉が寄せられている度合い）に入る
        let g = approach_state_features(&near, Color::Gote);
        assert!((g[3] - f_near[2]).abs() < 1e-9, "opp_king_siege は鏡像");
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
