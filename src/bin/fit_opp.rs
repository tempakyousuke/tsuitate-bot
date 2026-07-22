//! 対局記録から相手（人間）の指し手モデルをフィットする。
//!
//! 推定器（estimator.rs）の粒子は opp_move_weight の事前分布で相手手をサンプル
//! しており、この分布の精度が「観測と整合する粒子を維持できるか」= 粒子枯渇の
//! 速さを支配する。対人50局で41手目以降の健全率12%まで枯渇しており、事前分布の
//! 改善が最大のレバー。
//!
//! game:end の全公開棋譜から（局面, 人間が実際に選んだ手）の組を集め、
//! 特徴量の線形和のソフトマックス P(m) ∝ exp(θ·f(m)) を最尤推定する。
//! 出力された係数は estimator.rs の事前分布に手で反映する。
//!
//! **条件付きフィットにする理由**: 推定器の sample_opp_move は観測（取られたマス・
//! 王手宣言の有無）と整合する手に絞ってから事前分布でサンプルする。絞り込み後は
//! 「駒を取るか」「王手か」は全候補で同値になり正規化で消えるので、ここでは
//! 選ばれた手と同じ観測クラス（同じ捕獲マス・同じ王手有無）の候補だけを分母に
//! 入れて、クラス内で判別できる特徴量だけをフィットする。
//!
//! 使い方: cargo run --release --bin fit_opp -- records/*.jsonl

use std::collections::HashSet;

use tsuitate_bot::board::Coord;
use tsuitate_bot::opp_move_features::{
    FEATURE_NAMES, OPP_MOVE_FEATURES as D, home_squares, opp_move_features, to_square,
};
use tsuitate_bot::protocol::{Color, GameEndPayload};
use tsuitate_bot::shogi::{Position, ShogiMove, parse_usi};

struct Sample {
    /// 観測クラス内の各候補手の特徴量
    features: Vec<[f64; D]>,
    /// 選ばれた手のインデックス
    chosen: usize,
}

/// 1局から（人間手番の局面, 選択手）のサンプル列を作る。
/// 分母は選ばれた手と同じ観測クラス（同じ捕獲マス・同じ王手有無）の候補のみ
fn extract_samples(bot: Color, end: &GameEndPayload, samples: &mut Vec<Sample>) {
    let human = bot.other();
    let mut pos = Position::initial();
    // 人間側の駒が死んだマス（人間はそこに bot 駒がいることを知っている）
    let mut human_lost_at: HashSet<Coord> = HashSet::new();
    // bot の駒が動いたマス（初期位置のまま動いていない駒の判定に使う）
    let mut bot_touched: HashSet<Coord> = HashSet::new();

    for m in &end.moves {
        let Some(mv) = parse_usi(&m.usi) else { return };
        if m.by_color == human && pos.turn() == human {
            let homes = home_squares(&pos, bot, &bot_touched);
            let foul_count_this_turn = end
                .foul_attempts
                .iter()
                .filter(|f| f.by_color == human && f.move_number == pos.move_number())
                .count() as u32;
            // 選ばれた手の観測クラス
            let chosen_to = to_square(&mv);
            let chosen_capture = pos
                .piece_at(chosen_to)
                .filter(|p| p.color == bot)
                .map(|_| chosen_to);
            let mut chosen_next = pos.clone();
            chosen_next.play_unchecked(&mv);
            let chosen_check = chosen_next.in_check(bot);

            let mut features = Vec::new();
            let mut chosen = None;
            for lm in pos.legal_moves() {
                let to = to_square(&lm);
                let capture = pos.piece_at(to).filter(|p| p.color == bot).map(|_| to);
                if capture != chosen_capture {
                    continue; // 観測クラス違い（捕獲マスが異なる）
                }
                let mut next = pos.clone();
                next.play_unchecked(&lm);
                if next.in_check(bot) != chosen_check {
                    continue; // 観測クラス違い（王手宣言が異なる）
                }
                if lm == mv {
                    chosen = Some(features.len());
                }
                features.push(opp_move_features(
                    &pos,
                    &next,
                    &lm,
                    human,
                    &human_lost_at,
                    &homes,
                    foul_count_this_turn,
                ));
            }
            if let Some(chosen) = chosen {
                if features.len() >= 2 {
                    samples.push(Sample { features, chosen });
                }
            }
        }

        // 状態更新（両者の手を真の局面に適用）
        let to = to_square(&mv);
        let captured_color = pos.piece_at(to).map(|p| p.color);
        if m.by_color == bot && captured_color == Some(human) {
            human_lost_at.insert(to);
        }
        if m.by_color == bot {
            if let ShogiMove::Board { from, .. } = mv {
                bot_touched.insert(from);
            }
            bot_touched.insert(to);
        }
        pos.play_unchecked(&mv);
    }
}

/// 対数尤度と勾配（ソフトマックス。L2正則化つき）
fn log_likelihood(samples: &[Sample], theta: &[f64; D], l2: f64) -> (f64, [f64; D]) {
    let mut ll = 0.0;
    let mut grad = [0.0f64; D];
    for s in samples {
        let scores: Vec<f64> = s
            .features
            .iter()
            .map(|f| f.iter().zip(theta).map(|(a, b)| a * b).sum())
            .collect();
        let max = scores.iter().cloned().fold(f64::MIN, f64::max);
        let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
        let z: f64 = exps.iter().sum();
        ll += scores[s.chosen] - max - z.ln();
        for (f, e) in s.features.iter().zip(&exps) {
            let p = e / z;
            for i in 0..D {
                grad[i] -= p * f[i];
            }
        }
        for i in 0..D {
            grad[i] += s.features[s.chosen][i];
        }
    }
    let n = samples.len() as f64;
    for i in 0..D {
        grad[i] = grad[i] / n - l2 * theta[i];
        // llはサンプル平均で返す
    }
    (ll / n - 0.5 * l2 * theta.iter().map(|t| t * t).sum::<f64>(), grad)
}

/// NN化（v9）以前の線形版 opp_move_weight 相当での平均対数尤度。
/// 駒取り項は観測クラス内で定数なので 0 として比較する近似ベースライン
fn current_prior_ll(samples: &[Sample]) -> f64 {
    let idx = |name: &str| {
        FEATURE_NAMES
            .iter()
            .position(|&n| n == name)
            .unwrap_or_else(|| panic!("FEATURE_NAMES に {name} がない"))
    };
    let advance = idx("advance");
    let promote_minor = idx("promote_minor");
    let promote_major = idx("promote_major");
    let is_drop = idx("is_drop");
    let mut ll = 0.0;
    for s in samples {
        let weights: Vec<f64> = s
            .features
            .iter()
            .map(|f| {
                let mut w = 1.0;
                w += 0.25 * f[advance].max(0.0);
                if f[promote_minor] > 0.0 || f[promote_major] > 0.0 {
                    w += 1.0; // promote
                }
                if f[is_drop] > 0.0 {
                    w *= 0.5; // drop
                }
                w.max(0.05)
            })
            .collect();
        let z: f64 = weights.iter().sum();
        ll += (weights[s.chosen] / z).ln();
    }
    ll / samples.len() as f64
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("使い方: fit_opp <records/*.jsonl>");
        std::process::exit(1);
    }

    let mut samples: Vec<Sample> = vec![];
    let mut games = 0;
    for path in &paths {
        let Ok(content) = std::fs::read_to_string(path) else {
            continue;
        };
        let mut bot_color: Option<Color> = None;
        let mut end: Option<GameEndPayload> = None;
        for line in content.lines() {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            match v["type"].as_str() {
                Some("match") => {
                    bot_color = serde_json::from_value(v["your_color"].clone()).ok()
                }
                Some("end") => end = serde_json::from_value(v["payload"].clone()).ok(),
                _ => {}
            }
        }
        if let (Some(bot), Some(end)) = (bot_color, end) {
            games += 1;
            extract_samples(bot, &end, &mut samples);
        }
    }
    println!("{games}局から {} サンプル（人間の着手）を抽出", samples.len());
    // 特徴量ごとの基礎統計: 候補内の出現率 vs 選択された手での出現率
    for i in 0..D {
        let cand: usize = samples
            .iter()
            .map(|s| s.features.iter().filter(|f| f[i] > 0.0).count())
            .sum();
        let total: usize = samples.iter().map(|s| s.features.len()).sum();
        let chosen = samples
            .iter()
            .filter(|s| s.features[s.chosen][i] > 0.0)
            .count();
        println!(
            "  {:>16}: 候補 {:.2}% / 選択 {:.2}%",
            FEATURE_NAMES[i],
            100.0 * cand as f64 / total as f64,
            100.0 * chosen as f64 / samples.len() as f64
        );
    }

    // 勾配上昇（凸なので単純でよい。学習率は発散したら半分に）
    let mut theta = [0.0f64; D];
    let l2 = 0.01;
    let mut lr = 0.5;
    let (mut prev_ll, _) = log_likelihood(&samples, &theta, l2);
    for step in 0..2000 {
        let (ll, grad) = log_likelihood(&samples, &theta, l2);
        if ll < prev_ll - 1e-12 {
            lr *= 0.5;
        }
        prev_ll = ll;
        for i in 0..D {
            theta[i] += lr * grad[i];
        }
        if step % 200 == 0 {
            println!("  step {step}: 平均対数尤度 {ll:.4}");
        }
        if grad.iter().map(|g| g * g).sum::<f64>().sqrt() < 1e-5 {
            break;
        }
    }

    let (final_ll, _) = log_likelihood(&samples, &theta, 0.0);
    let uniform_ll: f64 = -(samples
        .iter()
        .map(|s| (s.features.len() as f64).ln())
        .sum::<f64>()
        / samples.len() as f64);
    let hand_ll = current_prior_ll(&samples);
    // top-1: フィット済みモデルの argmax が実際の手と一致した割合
    let top1 = samples
        .iter()
        .filter(|s| {
            let best = s
                .features
                .iter()
                .enumerate()
                .max_by(|a, b| {
                    let sa: f64 = a.1.iter().zip(&theta).map(|(x, t)| x * t).sum();
                    let sb: f64 = b.1.iter().zip(&theta).map(|(x, t)| x * t).sum();
                    sa.total_cmp(&sb)
                })
                .map(|(i, _)| i);
            best == Some(s.chosen)
        })
        .count() as f64
        / samples.len() as f64;

    println!("\n=== フィット結果 ===");
    for (name, t) in FEATURE_NAMES.iter().zip(&theta) {
        println!("  {name:>14}: {t:+.3}");
    }
    println!("\n平均対数尤度: 一様 {uniform_ll:.3} / 現行手調整 {hand_ll:.3} / フィット {final_ll:.3}");
    println!(
        "パープレキシティ: 一様 {:.1} / 現行 {:.1} / フィット {:.1}（小さいほど良い）",
        (-uniform_ll).exp(),
        (-hand_ll).exp(),
        (-final_ll).exp()
    );
    println!("top-1一致率: {:.1}%", top1 * 100.0);
    println!("\n採用するときは estimator.rs の opp_move_weight を exp(θ·f) 形式に置き換える。");
}
