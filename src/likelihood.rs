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
/// 2026-07-16 フィット（CI run 29468501253、600局・6157決定点、
/// 実効候補数 59.3→32.9、真実が上位半分に入る率 77.9%）。
/// 主な補正: 実際の相手は粒子の想定より歩を突き駒を展開している
/// （pawn_advance / pieces_home）、玉は想定ほど動かない（king_moved）、
/// 大駒の中央線越えは過大評価だった（past_mid）
pub const FITTED_THETA: [f64; PARTICLE_FEATURES] = [
    -0.815, // king_moved
    0.543,  // king_advance
    0.248,  // king_shift
    2.532,  // pawn_advance
    -2.051, // pieces_home
    -0.073, // at_my_death
    -0.050, // in_my_camp
    -1.377, // past_mid
];

/// 推論時に観測から分かる文脈
#[derive(Debug, Clone, Copy, Default)]
pub struct ParticleCtx {
    /// 直近で自駒が取られたマス（相手の駒がそこに着地した）
    pub opp_landed_last: Option<Coord>,
    /// 相手の着手数（NN版の文脈特徴量。グループ内で不変なので softmax では
    /// 単独では効かず、粒子特徴量との相互作用としてだけ効く）
    pub opp_moves: u32,
    /// 取られた自駒の数（相手の持ち駒＋打ち戻しの上限）
    pub my_dead: u32,
    /// いま自玉が王手されているか
    pub you_in_check: bool,
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

/// NN版の特徴量次元（線形8特徴量の上位互換。likelihood.rs のNN化 =
/// ロードマップ段階①の残り。学習は tsuitate-nn/train_particle.py、
/// 推論は particle_nn.rs の手書き forward pass）
pub const PARTICLE_NN_FEATURES: usize = 26;

pub const NN_FEATURE_NAMES: [&str; PARTICLE_NN_FEATURES] = [
    // 線形モデルと同じ定義の8特徴量
    "king_moved",
    "king_advance",
    "king_shift",
    "pawn_advance",
    "pieces_home",
    "at_my_death",
    "in_my_camp",
    "past_mid",
    // 駒種別の初期配置残存（「未観測の駒は初期配置のまま」の駒種分解。
    // home_lance_move / from_home と同じ原則の粒子判別版）
    "pawns_home",   // 初期マスに残る歩 /9
    "lances_home",  // /2
    "knights_home", // /2
    "silvers_home", // /2
    "golds_home",   // /2
    "bishop_home",  // 0/1
    "rook_home",    // 0/1
    // 進出・成り・持ち駒
    "pawn_advance_max",    // 歩（と金含む）の最大前進量
    "nonpawn_advance_max", // 歩・玉以外が敵陣（相手の3段）を出た最大段数
    "promoted_count",      // 成駒の数 /5
    "opp_hand_count",      // 相手の持ち駒数 /5（取った駒をまだ打っていない数。粒子ごとに違う）
    // 自分の駒（既知）との相互作用
    "attacked_by_me",     // 自分の利きが当たっている相手駒数 /5
    "hanging_to_me",      // うち相手の紐が無い駒数 /5
    "defended_frac",      // 相手の駒（玉以外）のうち紐つきの割合
    "king_zone_attackers", // 自玉とその周囲8マスへ利かせている相手駒数 /5
    // 文脈（グループ内で不変。NNの相互作用項としてだけ効く）
    "ply",          // 相手の着手数 /50
    "my_dead",      // 取られた自駒数 /10
    "you_in_check", // 0/1
];

/// NN版の粒子特徴量。先頭8個は線形版 `particle_features` と同じ値
pub fn particle_nn_features(
    pos: &Position,
    my_color: Color,
    ctx: &ParticleCtx,
) -> [f64; PARTICLE_NN_FEATURES] {
    let opp = my_color.other();
    let initial = Position::initial();
    let base = particle_features(pos, my_color, ctx);

    // 駒種別のhome残存カウント
    let mut home_by_role = [0.0f64; 7]; // Pawn..Rook の順
    for (sq, p) in initial.pieces() {
        if p.color != opp || p.role == Role::King {
            continue;
        }
        if pos
            .piece_at(sq)
            .is_some_and(|cur| cur.color == opp && cur.role == p.role)
        {
            let i = match p.role {
                Role::Pawn => 0,
                Role::Lance => 1,
                Role::Knight => 2,
                Role::Silver => 3,
                Role::Gold => 4,
                Role::Bishop => 5,
                Role::Rook => 6,
                _ => continue,
            };
            home_by_role[i] += 1.0;
        }
    }

    let pawn_home_rank = match opp {
        Color::Gote => 3,
        Color::Sente => 7,
    };
    // 敵陣（相手側の3段）の境界: そこを出た段数で非歩駒の進出を測る
    let camp_edge = match opp {
        Color::Gote => 3,
        Color::Sente => 7,
    };
    let mut pawn_adv_max = 0.0f64;
    let mut nonpawn_adv_max = 0.0f64;
    let mut promoted = 0.0f64;
    let mut attacked_by_me = 0.0f64;
    let mut hanging_to_me = 0.0f64;
    let mut defended = 0.0f64;
    let mut nonking = 0.0f64;
    for (sq, p) in pos.pieces() {
        if p.color != opp {
            continue;
        }
        match p.role {
            Role::Pawn | Role::Tokin => {
                pawn_adv_max = pawn_adv_max.max(advance_of(sq.rank, pawn_home_rank, opp).max(0.0));
            }
            Role::King => {}
            _ => {
                nonpawn_adv_max =
                    nonpawn_adv_max.max(advance_of(sq.rank, camp_edge, opp).max(0.0));
            }
        }
        if matches!(
            p.role,
            Role::Tokin
                | Role::Promotedlance
                | Role::Promotedknight
                | Role::Promotedsilver
                | Role::Horse
                | Role::Dragon
        ) {
            promoted += 1.0;
        }
        if p.role != Role::King {
            nonking += 1.0;
            let def = pos.is_attacked(sq, opp);
            if def {
                defended += 1.0;
            }
            if pos.is_attacked(sq, my_color) {
                attacked_by_me += 1.0;
                if !def {
                    hanging_to_me += 1.0;
                }
            }
        }
    }

    let opp_hand: f64 = pos
        .hand_map(opp)
        .values()
        .map(|&c| f64::from(c))
        .sum();

    // 自玉とその周囲8マスへの利き（王手駒仮説の妥当性判別に効かせたい）
    let mut king_zone_attackers = 0.0f64;
    if let Some(k) = pos.king_square(my_color) {
        for (sq, p) in pos.pieces() {
            if p.color != opp {
                continue;
            }
            let mut hits = false;
            for df in -1i8..=1 {
                for dr in -1i8..=1 {
                    let t = Coord {
                        file: k.file + df,
                        rank: k.rank + dr,
                    };
                    if (1..=9).contains(&t.file)
                        && (1..=9).contains(&t.rank)
                        && pos.attacks(sq, t)
                    {
                        hits = true;
                        break;
                    }
                }
                if hits {
                    break;
                }
            }
            if hits {
                king_zone_attackers += 1.0;
            }
        }
    }

    [
        base[0],
        base[1],
        base[2],
        base[3],
        base[4],
        base[5],
        base[6],
        base[7],
        home_by_role[0] / 9.0,
        home_by_role[1] / 2.0,
        home_by_role[2] / 2.0,
        home_by_role[3] / 2.0,
        home_by_role[4] / 2.0,
        home_by_role[5],
        home_by_role[6],
        pawn_adv_max,
        nonpawn_adv_max,
        promoted / 5.0,
        opp_hand / 5.0,
        attacked_by_me / 5.0,
        hanging_to_me / 5.0,
        if nonking > 0.0 { defended / nonking } else { 0.0 },
        king_zone_attackers / 5.0,
        f64::from(ctx.opp_moves) / 50.0,
        f64::from(ctx.my_dead) / 10.0,
        f64::from(ctx.you_in_check),
    ]
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
            ..ParticleCtx::default()
        };
        let f = particle_features(&pos, Color::Sente, &ctx);
        assert_eq!(f[5], 1.0, "2bには後手の角がいる");
    }

    #[test]
    fn nn_features_track_piece_type_development() {
        let mut pos = Position::initial();
        let ctx = ParticleCtx::default();
        let f0 = particle_nn_features(&pos, Color::Sente, &ctx);
        // 先頭8個は線形版と同じ値
        let lin = particle_features(&pos, Color::Sente, &ctx);
        for i in 0..PARTICLE_FEATURES {
            assert_eq!(f0[i], lin[i], "feature {i} が線形版とずれている");
        }
        assert_eq!(f0[8], 1.0, "初期局面: 歩は全部home");
        assert_eq!(f0[9], 1.0, "香車は全部home");
        assert_eq!(f0[13], 1.0, "角はhome");
        assert_eq!(f0[14], 1.0, "飛車はhome");
        assert_eq!(f0[17], 0.0, "成駒なし");
        assert_eq!(f0[18], 0.0, "持ち駒なし");
        assert_eq!(f0[15], 0.0, "歩の前進なし");

        pos.play_unchecked(&parse_usi("7g7f").unwrap());
        pos.play_unchecked(&parse_usi("3c3d").unwrap());
        let f1 = particle_nn_features(&pos, Color::Sente, &ctx);
        assert!((f1[8] - 8.0 / 9.0).abs() < 1e-9, "歩1枚がhomeを離れた: {}", f1[8]);
        assert_eq!(f1[15], 1.0, "歩の最大前進=1: {}", f1[15]);

        let ctx = ParticleCtx {
            opp_moves: 25,
            my_dead: 3,
            you_in_check: true,
            ..ParticleCtx::default()
        };
        let f2 = particle_nn_features(&pos, Color::Sente, &ctx);
        assert_eq!(f2[23], 0.5);
        assert_eq!(f2[24], 0.3);
        assert_eq!(f2[25], 1.0);
    }

    /// NN特徴量抽出は stratified_sample でユニーク粒子ごとに1回呼ばれる
    /// （1手あたり数百回オーダー）。利き判定（attacked_by_me / defended /
    /// king_zone）を含むため forward pass より重いが、粒子512個ぶんでも
    /// 思考予算（900ms〜）の数%に収まることをガードする
    #[test]
    fn nn_feature_extraction_is_fast_enough() {
        let mut pos = Position::initial();
        for usi in ["7g7f", "3c3d", "8h2b+", "3a2b", "B*4e", "8c8d"] {
            pos.play_unchecked(&parse_usi(usi).unwrap());
        }
        let ctx = ParticleCtx {
            opp_moves: 3,
            my_dead: 1,
            ..ParticleCtx::default()
        };
        let n = 2_000u32;
        let start = std::time::Instant::now();
        let mut acc = 0.0f64;
        for _ in 0..n {
            acc += particle_nn_features(std::hint::black_box(&pos), Color::Sente, &ctx)[19];
        }
        let elapsed = start.elapsed();
        std::hint::black_box(acc);
        eprintln!("{n}回の特徴量抽出: {elapsed:?}（1回あたり{:?}）", elapsed / n);
        // release実測は数µs/回のオーダー。512粒子×数µs ≈ 数ms/手。
        // debugは1桁以上遅いので閾値を緩める
        let threshold = if cfg!(debug_assertions) { 400e-6 } else { 40e-6 };
        assert!(
            elapsed.as_secs_f64() / f64::from(n) < threshold,
            "特徴量抽出が遅すぎる: {elapsed:?} / {n}回"
        );
    }

    #[test]
    fn zero_theta_gives_zero_log_weight() {
        let pos = Position::initial();
        let f = particle_features(&pos, Color::Sente, &ParticleCtx::default());
        assert_eq!(particle_log_weight(&f, &[0.0; PARTICLE_FEATURES]), 0.0);
    }

    #[test]
    fn fitted_theta_prefers_developed_opponent() {
        // フィット済み係数の健全性: 「歩を突いて駒が展開した」局面のほうが
        // 初期局面のままより対数重みが高い（実対局の分布に合う方向）
        let ctx = ParticleCtx::default();
        let initial = Position::initial();
        let mut developed = Position::initial();
        for usi in ["7g7f", "3c3d", "2g2f", "8c8d"] {
            developed.play_unchecked(&parse_usi(usi).unwrap());
        }
        let w_initial =
            particle_log_weight(&particle_features(&initial, Color::Sente, &ctx), &FITTED_THETA);
        let w_dev = particle_log_weight(
            &particle_features(&developed, Color::Sente, &ctx),
            &FITTED_THETA,
        );
        assert!(
            w_dev > w_initial,
            "展開した相手のほうが尤度が高いはず: dev={w_dev} initial={w_initial}"
        );
    }
}
