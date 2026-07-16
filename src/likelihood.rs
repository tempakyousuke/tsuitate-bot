//! 粒子の尤度モデル（教師あり学習）。
//!
//! 推定器の粒子は「観測と整合するか」の二値でしか選別されず、整合する粒子は
//! （ソフト救済の減衰を除き）同じ重みで評価に入る。しかし観測と整合する局面の
//! 中にも「真の局面に近いもの」と「ありそうにないもの」がある。
//!
//! アリーナ記録は審判の真実（全手順）を持つので、「観測列 → 真の局面」の教師
//! データが無制限に作れる（bin/fit_particles）。真の局面を粒子群の中で判別する
//! 条件付き最尤推定 P(真 | 候補粒子) ∝ exp(θ·φ) をフィットし、係数をここへ
//! 反映する。評価側（strategy.rs の stratified_sample）は粒子重みに
//! exp(θ·φ) を平均1へ正規化して乗じる（相対的な再重み付けなので
//! p(合法) の事前ブレンドの較正は変えない）。
//!
//! 特徴量は推論時に「観測から分かる文脈＋粒子自身」だけで計算できるものに限る
//! （公平性: 真実は使わない）。

use crate::board::Coord;
use crate::protocol::{Color, Role};
use crate::shogi::Position;

pub const PARTICLE_FEATURES: usize = 8;

pub const FEATURE_NAMES: [&str; PARTICLE_FEATURES] = [
    "king_moved",    // 相手玉が初期位置から動いた
    "king_advance",  // 相手玉の前進量（段。負=後退はない）
    "king_shift",    // 相手玉の横ずれ量（筋）
    "pawn_advance",  // 相手の歩（と金含む）の平均前進量
    "pieces_home",   // 初期位置に残っている相手駒の割合（0..1）
    "at_my_death",   // 直近で自駒が死んだマスに相手駒がいる（取った駒は残留しがち）
    "in_my_camp",    // 自陣（3段）内の相手駒数
    "past_mid",      // 中央線を越えて自分側にいる相手駒数（歩・玉以外）
];

/// フィット済み係数（bin/fit_particles の出力を反映する）。
/// 全ゼロ = 重み1の恒等（フィット前は挙動に影響しない）
pub const FITTED_THETA: [f64; PARTICLE_FEATURES] = [0.0; PARTICLE_FEATURES];

/// 推論時に観測から分かる文脈
#[derive(Debug, Clone, Copy, Default)]
pub struct ParticleCtx {
    /// 直近で自駒が取られたマス（相手の駒がそこに着地した）
    pub opp_landed_last: Option<Coord>,
}

/// 相手側の前進量（段）: 初期配置側から自分側へ何段進んだか
fn advance_of(rank: i8, home_rank: i8, opp: Color) -> f64 {
    match opp {
        Color::Gote => f64::from(rank - home_rank),
        Color::Sente => f64::from(home_rank - rank),
    }
}

/// 粒子の特徴量。my_color は自分（観測者）の色
pub fn particle_features(
    pos: &Position,
    my_color: Color,
    ctx: &ParticleCtx,
) -> [f64; PARTICLE_FEATURES] {
    let opp = my_color.other();
    let initial = Position::initial();

    // 玉の3特徴
    let king_home = initial.king_square(opp);
    let king = pos.king_square(opp);
    let (king_moved, king_advance, king_shift) = match (king, king_home) {
        (Some(k), Some(h)) => (
            f64::from(k != h),
            advance_of(k.rank, h.rank, opp).max(0.0),
            f64::from((k.file - h.file).abs()),
        ),
        _ => (1.0, 0.0, 0.0),
    };

    // 歩（と金含む）の平均前進量。相手歩の初期段: 後手=3段目 / 先手=7段目
    let pawn_home = match opp {
        Color::Gote => 3,
        Color::Sente => 7,
    };
    let mut pawn_adv = 0.0;
    let mut pawns = 0.0;
    // 初期位置に残っている相手駒（種類まで一致）の数
    let mut home_count = 0.0;
    let mut in_my_camp = 0.0;
    let mut past_mid = 0.0;
    for (sq, p) in pos.pieces() {
        if p.color != opp {
            continue;
        }
        if matches!(p.role, Role::Pawn | Role::Tokin) {
            pawn_adv += advance_of(sq.rank, pawn_home, opp).max(0.0);
            pawns += 1.0;
        }
        // 自陣3段（自分側の端から3段）
        let in_camp = match my_color {
            Color::Sente => sq.rank >= 7,
            Color::Gote => sq.rank <= 3,
        };
        if in_camp {
            in_my_camp += 1.0;
        }
        // 中央線越え（歩・玉以外）
        let past = match my_color {
            Color::Sente => sq.rank >= 6,
            Color::Gote => sq.rank <= 4,
        };
        if past && !matches!(p.role, Role::Pawn | Role::Tokin | Role::King) {
            past_mid += 1.0;
        }
    }
    for (sq, p) in initial.pieces() {
        if p.color == opp
            && pos
                .piece_at(sq)
                .is_some_and(|cur| cur.color == opp && cur.role == p.role)
        {
            home_count += 1.0;
        }
    }

    let at_my_death = ctx
        .opp_landed_last
        .map(|s| f64::from(pos.piece_at(s).is_some_and(|p| p.color == opp)))
        .unwrap_or(0.0);

    [
        king_moved,
        king_advance,
        king_shift,
        if pawns > 0.0 { pawn_adv / pawns } else { 0.0 },
        home_count / 20.0,
        at_my_death,
        in_my_camp,
        past_mid,
    ]
}

/// θ·φ（対数重み）。重みは exp(θ·φ) で、呼び出し側が平均1へ正規化する
pub fn particle_log_weight(features: &[f64; PARTICLE_FEATURES], theta: &[f64; PARTICLE_FEATURES]) -> f64 {
    features.iter().zip(theta).map(|(f, t)| f * t).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shogi::parse_usi;

    #[test]
    fn features_reflect_opponent_development() {
        let mut pos = Position::initial();
        let ctx = ParticleCtx::default();
        let before = particle_features(&pos, Color::Sente, &ctx);
        assert_eq!(before[0], 0.0, "初期局面: 玉は動いていない");
        assert_eq!(before[4], 1.0, "初期局面: 全駒が初期位置");
        assert_eq!(before[3], 0.0, "初期局面: 歩は前進していない");

        // 後手が 3c3d（歩を1段前進）→ pawn_advance が正に、home比率が下がる
        pos.play_unchecked(&parse_usi("7g7f").unwrap());
        pos.play_unchecked(&parse_usi("3c3d").unwrap());
        let after = particle_features(&pos, Color::Sente, &ctx);
        assert!(after[3] > 0.0, "歩の前進が反映される: {}", after[3]);
        assert!(after[4] < 1.0, "初期位置の駒が減る: {}", after[4]);

        // at_my_death: 2b の駒の有無
        let ctx = ParticleCtx {
            opp_landed_last: Some(Coord { file: 2, rank: 2 }),
        };
        let f = particle_features(&pos, Color::Sente, &ctx);
        assert_eq!(f[5], 1.0, "2bには後手の角がいる");
    }

    #[test]
    fn zero_theta_gives_zero_log_weight() {
        let pos = Position::initial();
        let f = particle_features(&pos, Color::Sente, &ParticleCtx::default());
        assert_eq!(particle_log_weight(&f, &FITTED_THETA), 0.0);
    }
}
