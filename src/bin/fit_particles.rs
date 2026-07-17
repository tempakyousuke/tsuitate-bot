//! アリーナ記録から粒子の尤度モデル（likelihood.rs）をフィットする。
//!
//! アリーナ記録には bot の観測列と審判の真実（全手順）が両方入っている。
//! 各「相手の着手」観測時点で推定器を回し、そのユニーク粒子を負例、
//! 真の局面を正例とする条件付き最尤推定 P(真 | 候補) ∝ exp(θ·φ) を解く
//! （bin/fit_opp と同じ方法論の局面版）。
//!
//! 出力された係数は likelihood.rs の FITTED_THETA に手で反映する。
//! 評価指標: 真の局面へ割り当てた事後確率（一様 = 1/候補数 と比較）。
//!
//! 使い方: cargo run --release --bin fit_particles -- <records/*.jsonl>
//! 環境変数: FIT_MAX_POINTS_PER_GAME（既定20）: 1局から取る決定点の上限

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use tsuitate_bot::board::parse_usi_square;
use tsuitate_bot::estimator::Estimator;
use tsuitate_bot::likelihood::{
    FEATURE_NAMES, PARTICLE_FEATURES as D, ParticleCtx, particle_features,
};
use tsuitate_bot::observation::{Observation, ObservationLog};
use tsuitate_bot::protocol::{Color, GameEndPayload};
use tsuitate_bot::shogi::{Position, parse_usi};

struct Sample {
    /// 候補（粒子＋真の局面）の特徴量
    features: Vec<[f64; D]>,
    /// 候補の固定対数オフセット（推論側のベース重み = 指紋ごとの
    /// logΣexp(logw)（max正規化）。ソフト減衰 EPS_INFO も logw に課金済み。
    /// 推論の重みは base×exp(θ·φ) なので、学習側の softmax にも同じオフセットを
    /// 入れないと分布がずれる）
    offsets: Vec<f64>,
    /// 真の局面のインデックス
    truth: usize,
}

/// 推論側と同じ思考予算スケール（strategy.rs の SearchBudget と同じ式）
fn inference_scale() -> f64 {
    let ms: f64 = std::env::var("TSUITATE_THINK_BUDGET_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000.0);
    (ms / 900.0).clamp(0.25, 8.0)
}

/// 1局から（観測時点の粒子群, 真の局面）のサンプル列を作る。
/// 決定点は「相手の着手」観測ごと。推定器は記録の観測列をそのまま再生する
fn extract_samples(
    bot: Color,
    observations: &[Observation],
    end: &GameEndPayload,
    max_points: usize,
    game_seed: u64,
) -> Vec<Sample> {
    // 真実の局面列（moves[0..k] 適用後 = truth[k]）
    let mut truth_positions = vec![Position::initial()];
    for m in &end.moves {
        let Some(mv) = parse_usi(&m.usi) else {
            return vec![];
        };
        let mut next = truth_positions.last().unwrap().clone();
        next.play_unchecked(&mv);
        truth_positions.push(next);
    }

    // 推論側と同じスケールの推定器で負例分布を合わせる
    let mut est = Estimator::with_seed_and_scale(bot, game_seed, inference_scale());
    let mut log = ObservationLog::default();
    let mut samples = vec![];
    // 決定点の間引き: 固定周期（% stride）は初回を必ず落とし本数も過小になるので、
    // 等間隔のインデックス集合 round((j+0.5)·n/k) を使う
    let opp_moves = observations
        .iter()
        .filter(|o| matches!(o, Observation::OpponentMoved { .. }))
        .count();
    let k = opp_moves.min(max_points.max(1));
    let targets: HashSet<usize> = (0..k)
        .map(|j| ((j as f64 + 0.5) * opp_moves as f64 / k as f64) as usize)
        .collect();
    let mut opp_move_idx = 0usize;
    let mut opp_landed_last: Option<tsuitate_bot::board::Coord> = None;

    let mut i = 0usize;
    while i < observations.len() {
        let event = &observations[i];
        let measure = match event {
            Observation::OpponentMoved {
                move_number,
                captured_my_piece_at,
            } => {
                if let Some(sq) = captured_my_piece_at.as_deref().and_then(parse_usi_square) {
                    opp_landed_last = Some(sq);
                }
                Some(*move_number)
            }
            _ => None,
        };
        log.record(event.clone());
        // 着手直後の Check は同じ着手の観測なので、update の前に対で入れる
        // （実戦では自分の手番でまとめて update するため常に対になっている。
        //  着手だけ入れて update すると王手宣言の制約が丸ごと落ちる）
        if matches!(
            event,
            Observation::OpponentMoved { .. } | Observation::MyMove { .. }
        ) {
            if let Some(check @ Observation::Check { .. }) = observations.get(i + 1) {
                log.record(check.clone());
                i += 1;
            }
        }
        let should_update = matches!(
            event,
            Observation::OpponentMoved { .. } | Observation::MyFoul { .. }
        );
        if should_update {
            est.update(&log);
        }
        i += 1;
        let Some(mn) = measure else { continue };
        let point = opp_move_idx;
        opp_move_idx += 1;
        if !targets.contains(&point) {
            continue;
        }
        // 観測 move_number = 適用後の値 → その時点までに mn-1 手が指されている
        let Some(truth) = truth_positions.get(mn as usize - 1) else {
            continue;
        };
        let ctx = ParticleCtx { opp_landed_last };
        // ユニーク粒子（真実と同一指紋の粒子は truth 側に統合）。
        // 各候補には推論側のベース重み ln(soft_decay^penalty) をオフセットとして持たせる
        let truth_fp = truth.fingerprint();
        // 推論側のベース重み: 指紋ごとの logΣexp(logw - max)（multiplicity 畳み込み。
        // stratified_sample と同じ規約）
        let max_lw = est
            .log_weights()
            .iter()
            .copied()
            .fold(f64::MIN, f64::max);
        let mut mass: HashMap<u64, f64> = HashMap::new();
        for (pos, &lw) in est.particles().iter().zip(est.log_weights()) {
            *mass.entry(pos.fingerprint()).or_insert(0.0) += (lw - max_lw).exp();
        }
        let mut seen = HashSet::new();
        seen.insert(truth_fp);
        let mut features = vec![particle_features(truth, bot, &ctx)];
        let mut offsets = vec![0.0]; // 真実は完全整合のベース重み1（= log 0.0）とみなす
        for pos in est.particles() {
            if features.len() > 64 {
                break;
            }
            if seen.insert(pos.fingerprint()) {
                features.push(particle_features(pos, bot, &ctx));
                offsets.push(mass[&pos.fingerprint()].ln());
            }
        }
        if features.len() >= 8 {
            samples.push(Sample {
                features,
                offsets,
                truth: 0,
            });
        }
    }
    samples
}

/// 対数尤度と勾配（ソフトマックス、L2つき）。fit_opp と同型。
/// score = 固定オフセット（推論側のベース重み）+ θ·φ。オフセットは定数なので
/// 勾配の形は変わらない
fn log_likelihood(samples: &[Sample], theta: &[f64; D], l2: f64) -> (f64, [f64; D]) {
    let mut ll = 0.0;
    let mut grad = [0.0f64; D];
    for s in samples {
        let scores: Vec<f64> = s
            .features
            .iter()
            .zip(&s.offsets)
            .map(|(f, off)| off + f.iter().zip(theta).map(|(a, b)| a * b).sum::<f64>())
            .collect();
        let max = scores.iter().cloned().fold(f64::MIN, f64::max);
        let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
        let z: f64 = exps.iter().sum();
        ll += scores[s.truth] - max - z.ln();
        for (f, e) in s.features.iter().zip(&exps) {
            let p = e / z;
            for i in 0..D {
                grad[i] -= p * f[i];
            }
        }
        for i in 0..D {
            grad[i] += s.features[s.truth][i];
        }
    }
    let n = samples.len() as f64;
    for g in grad.iter_mut() {
        *g /= n;
    }
    for i in 0..D {
        grad[i] -= l2 * theta[i];
    }
    (
        ll / n - 0.5 * l2 * theta.iter().map(|t| t * t).sum::<f64>(),
        grad,
    )
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("使い方: fit_particles <records/*.jsonl>");
        std::process::exit(1);
    }
    let max_points: usize = std::env::var("FIT_MAX_POINTS_PER_GAME")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);

    // 記録の読み込みと決定点抽出（推定器の再生が重いのでスレッドに分散）
    let samples: Mutex<Vec<Sample>> = Mutex::new(vec![]);
    let games = Mutex::new(0usize);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let chunk = paths.len().div_ceil(threads);
    std::thread::scope(|scope| {
        for (t, chunk_paths) in paths.chunks(chunk.max(1)).enumerate() {
            let samples = &samples;
            let games = &games;
            scope.spawn(move || {
                for (i, path) in chunk_paths.iter().enumerate() {
                    let Ok(content) = std::fs::read_to_string(path) else {
                        continue;
                    };
                    let mut bot_color: Option<Color> = None;
                    let mut observations: Vec<Observation> = vec![];
                    let mut end: Option<GameEndPayload> = None;
                    for line in content.lines() {
                        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
                            continue;
                        };
                        match v["type"].as_str() {
                            Some("match") => {
                                bot_color = serde_json::from_value(v["your_color"].clone()).ok()
                            }
                            Some("obs") => {
                                if let Ok(obs) = serde_json::from_value(v["event"].clone()) {
                                    observations.push(obs);
                                }
                            }
                            Some("end") => {
                                end = serde_json::from_value(v["payload"].clone()).ok()
                            }
                            _ => {}
                        }
                    }
                    let (Some(bot), Some(end)) = (bot_color, end) else {
                        continue;
                    };
                    let seed = (t * 1_000_003 + i) as u64;
                    let s = extract_samples(bot, &observations, &end, max_points, seed);
                    samples.lock().unwrap().extend(s);
                    *games.lock().unwrap() += 1;
                }
            });
        }
    });
    let samples = samples.into_inner().unwrap();
    let games = games.into_inner().unwrap();
    println!(
        "{games}局から {} 決定点（平均候補数 {:.1}）を抽出",
        samples.len(),
        samples.iter().map(|s| s.features.len()).sum::<usize>() as f64
            / samples.len().max(1) as f64
    );
    if samples.is_empty() {
        std::process::exit(1);
    }

    // 勾配上昇
    let mut theta = [0.0f64; D];
    let l2 = 0.01;
    let mut lr = 0.5;
    let (mut prev_ll, _) = log_likelihood(&samples, &theta, l2);
    for step in 0..3000 {
        let (ll, grad) = log_likelihood(&samples, &theta, l2);
        if ll < prev_ll - 1e-12 {
            lr *= 0.5;
        }
        prev_ll = ll;
        for i in 0..D {
            theta[i] += lr * grad[i];
        }
        if step % 300 == 0 {
            println!("  step {step}: 平均対数尤度 {ll:.4}");
        }
        if grad.iter().map(|g| g * g).sum::<f64>().sqrt() < 1e-5 {
            break;
        }
    }

    // 指標: 真の局面の事後確率（一様と比較）と、真実が上位半分に入る率
    let (final_ll, _) = log_likelihood(&samples, &theta, 0.0);
    let uniform_ll: f64 = -(samples
        .iter()
        .map(|s| (s.features.len() as f64).ln())
        .sum::<f64>()
        / samples.len() as f64);
    let top_half = samples
        .iter()
        .filter(|s| {
            let score = |f: &[f64; D], off: f64| -> f64 {
                off + f.iter().zip(&theta).map(|(a, b)| a * b).sum::<f64>()
            };
            let ts = score(&s.features[s.truth], s.offsets[s.truth]);
            let better = s
                .features
                .iter()
                .zip(&s.offsets)
                .filter(|(f, off)| score(f, **off) > ts)
                .count();
            better * 2 < s.features.len()
        })
        .count() as f64
        / samples.len() as f64;

    println!("\n=== フィット結果 ===");
    for (name, t) in FEATURE_NAMES.iter().zip(&theta) {
        println!("  {name:>14}: {t:+.3}");
    }
    println!(
        "\n真の局面の平均対数事後確率: 一様 {uniform_ll:.3} / フィット {final_ll:.3}（改善 {:+.3}）",
        final_ll - uniform_ll
    );
    println!(
        "実効候補数（perplexity）: 一様 {:.1} → フィット {:.1}（小さいほど真実に質量が寄る）",
        (-uniform_ll).exp(),
        (-final_ll).exp()
    );
    println!("真実がスコア上位半分に入る率: {:.1}%", top_half * 100.0);
    println!("\n採用するときは likelihood.rs の FITTED_THETA へ反映する。");
}
