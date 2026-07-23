//! 粒子尤度NN（likelihood.rs のNN化）の学習データ書き出し。
//!
//! bin/fit_particles と同じ決定点抽出（アリーナ記録の観測列を推定器で再生し、
//! 各「相手の着手」観測時点のユニーク粒子群＋真の局面を1グループとする）を行い、
//! 条件付き学習用のCSVを出力する。1行=1候補局面、同じ (game_id, decision_id) の
//! 複数行が1グループ。chosen=1 の行が真の局面。offset 列は推論側のベース対数重み
//! （指紋質量 logΣexp(logw)。学習側の softmax にも同じオフセットを入れないと
//! 分布がずれる — fit_particles と同じ理由）。
//!
//! 特徴量の定義は likelihood.rs の particle_nn_features に一本化する
//! （学習/推論のズレ防止。opp_move_features.rs と同じ方針）。
//!
//! 使い方: cargo run --release --bin export_particle_data -- records/*.jsonl > data.csv
//! 環境変数: FIT_MAX_POINTS_PER_GAME（既定20）: 1局から取る決定点の上限
//!
//! 注意: game_id は1回の実行内でのみ一意。複数回のexport出力を後で連結すると
//! game_id が衝突して別対局が同一グループ/同一対局扱いになるので、
//! 全記録を1回の実行にまとめて渡すこと

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use tsuitate_bot::board::parse_usi_square;
use tsuitate_bot::estimator::Estimator;
use tsuitate_bot::likelihood::{
    NN_FEATURE_NAMES, PARTICLE_NN_FEATURES as D, ParticleCtx, particle_nn_features,
};
use tsuitate_bot::observation::{Observation, ObservationLog};
use tsuitate_bot::protocol::{Color, GameEndPayload};
use tsuitate_bot::shogi::{Position, parse_usi};

/// 推論側と同じ思考予算スケール（strategy.rs の SearchBudget と同じ式）
fn inference_scale() -> f64 {
    let ms: f64 = std::env::var("TSUITATE_THINK_BUDGET_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000.0);
    (ms / 900.0).clamp(0.25, 8.0)
}

/// 1局ぶんの決定点をCSV行として書き出す（fit_particles::extract_samples の
/// CSV版。抽出条件・オフセット規約はあちらと揃えること）
fn export_game(
    game_id: usize,
    bot: Color,
    observations: &[Observation],
    end: &GameEndPayload,
    max_points: usize,
    game_seed: u64,
    buf: &mut Vec<String>,
) {
    let mut truth_positions = vec![Position::initial()];
    for m in &end.moves {
        let Some(mv) = parse_usi(&m.usi) else {
            return;
        };
        let mut next = truth_positions.last().unwrap().clone();
        next.play_unchecked(&mv);
        truth_positions.push(next);
    }

    let mut est = Estimator::with_seed_and_scale(bot, game_seed, inference_scale());
    let mut log = ObservationLog::default();
    let opp_moves_total = observations
        .iter()
        .filter(|o| matches!(o, Observation::OpponentMoved { .. }))
        .count();
    let k = opp_moves_total.min(max_points.max(1));
    let targets: HashSet<usize> = (0..k)
        .map(|j| ((j as f64 + 0.5) * opp_moves_total as f64 / k as f64) as usize)
        .collect();
    let mut opp_move_idx = 0usize;
    let mut opp_landed_last: Option<tsuitate_bot::board::Coord> = None;
    let mut my_dead = 0u32;
    let mut decision_id = 0u64;

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
                    my_dead += 1;
                }
                Some(*move_number)
            }
            _ => None,
        };
        log.record(event.clone());
        // 着手直後の Check は同じ着手の観測なので、update の前に対で入れる
        let mut you_in_check = false;
        if matches!(
            event,
            Observation::OpponentMoved { .. } | Observation::MyMove { .. }
        ) {
            if let Some(check @ Observation::Check { in_check }) = observations.get(i + 1) {
                you_in_check = *in_check == bot;
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
        let Some(truth) = truth_positions.get(mn as usize - 1) else {
            continue;
        };
        let ctx = ParticleCtx {
            opp_landed_last,
            opp_moves: opp_move_idx as u32,
            my_dead,
            you_in_check,
        };
        let truth_fp = truth.fingerprint();
        let max_lw = est
            .log_weights()
            .iter()
            .copied()
            .fold(f64::MIN, f64::max);
        // 物理不整合（phys_taint>0）の粒子は学習候補に入れない
        let mut mass: HashMap<u64, f64> = HashMap::new();
        for ((pos, &lw), &taint) in est
            .particles()
            .iter()
            .zip(est.log_weights())
            .zip(est.phys_taint())
        {
            if taint > 0 {
                continue;
            }
            *mass.entry(pos.fingerprint()).or_insert(0.0) += (lw - max_lw).exp();
        }
        let mut seen = HashSet::new();
        seen.insert(truth_fp);
        // 真実は完全整合のベース重み1（= log 0.0）とみなす（fit_particles と同じ規約）
        let mut rows: Vec<(f64, u8, [f64; D])> =
            vec![(0.0, 1, particle_nn_features(truth, bot, &ctx))];
        for (pos, &taint) in est.particles().iter().zip(est.phys_taint()) {
            if rows.len() > 64 {
                break;
            }
            if taint > 0 {
                continue;
            }
            if seen.insert(pos.fingerprint()) {
                rows.push((
                    mass[&pos.fingerprint()].ln(),
                    0,
                    particle_nn_features(pos, bot, &ctx),
                ));
            }
        }
        if rows.len() < 8 {
            continue;
        }
        for (offset, chosen, features) in rows {
            let feat_csv = features
                .iter()
                .map(|f| format!("{f:.6}"))
                .collect::<Vec<_>>()
                .join(",");
            buf.push(format!("{game_id},{decision_id},{offset:.6},{chosen},{feat_csv}"));
        }
        decision_id += 1;
    }
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("使い方: export_particle_data <records/*.jsonl>");
        std::process::exit(1);
    }
    let max_points: usize = std::env::var("FIT_MAX_POINTS_PER_GAME")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);

    println!("game_id,decision_id,offset,chosen,{}", NN_FEATURE_NAMES.join(","));

    let out: Mutex<Vec<String>> = Mutex::new(vec![]);
    let games = Mutex::new(0usize);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let chunk = paths.len().div_ceil(threads);
    std::thread::scope(|scope| {
        for (t, chunk_paths) in paths.chunks(chunk.max(1)).enumerate() {
            let out = &out;
            let games = &games;
            let base = t * chunk.max(1);
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
                    let mut buf = vec![];
                    export_game(base + i, bot, &observations, &end, max_points, seed, &mut buf);
                    if !buf.is_empty() {
                        out.lock().unwrap().extend(buf);
                        *games.lock().unwrap() += 1;
                    }
                }
            });
        }
    });
    let rows = out.into_inner().unwrap();
    for r in &rows {
        println!("{r}");
    }
    eprintln!(
        "{}局から {} 行を書き出し",
        games.into_inner().unwrap(),
        rows.len()
    );
}
