//! 対局記録の観測ログを現在の推定器にリプレイし、粒子の健全性を測る。
//!
//! 記録の debug（chose 行）には「その対局を指した当時のコード」での粒子状態が
//! 残っているので、事前分布などを変えた後にこれを実行すると、同一データでの
//! 新旧比較になる（相手手の事前分布の改善が枯渇をどれだけ遅らせたか等）。
//!
//! 使い方: cargo run --release --bin replay_health -- records/*.jsonl

use std::collections::HashSet;

use tsuitate_bot::estimator::Estimator;
use tsuitate_bot::observation::{Observation, ObservationLog};
use tsuitate_bot::protocol::Color;

fn bucket(move_number: u32) -> usize {
    match move_number {
        0..=20 => 0,
        21..=40 => 1,
        _ => 2,
    }
}

const BUCKET_NAMES: [&str; 3] = ["1-20", "21-40", "41+"];
/// 記録側 debug と同じ条件（strategy.rs の EVAL_PARTICLES）でユニーク数を数える
const SAMPLE_CAP: usize = 192;

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("使い方: replay_health <records/*.jsonl>");
        std::process::exit(1);
    }

    // (ユニーク粒子数の合計, 健全だった回数, 観測点数)
    let mut new_stats = [(0u64, 0u64, 0u64); 3];
    let mut old_stats = [(0u64, 0u64, 0u64); 3];
    let mut games = 0;

    for path in &paths {
        let Ok(content) = std::fs::read_to_string(path) else {
            continue;
        };
        let mut bot_color: Option<Color> = None;
        let mut events: Vec<Observation> = vec![];
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
                        events.push(obs);
                    }
                }
                Some("chose") => {
                    // 記録当時のコードでの粒子状態（比較の基準）
                    if let Some(dbg) = v["debug"].as_object() {
                        let mn = v["move_number"].as_u64().unwrap_or(0) as u32;
                        let b = bucket(mn);
                        old_stats[b].0 += dbg["unique_particles"].as_u64().unwrap_or(0);
                        old_stats[b].1 += dbg["healthy"].as_bool().unwrap_or(false) as u64;
                        old_stats[b].2 += 1;
                    }
                }
                _ => {}
            }
        }
        let Some(bot) = bot_color else { continue };
        games += 1;

        let mut est = Estimator::new(bot);
        let mut log = ObservationLog::default();
        let mut i = 0usize;
        while i < events.len() {
            let event = &events[i];
            let measure_at = match event {
                Observation::OpponentMoved { move_number, .. } => Some(*move_number),
                _ => None,
            };
            let should_update = matches!(
                event,
                Observation::OpponentMoved { .. } | Observation::MyFoul { .. }
            );
            log.record(event.clone());
            // 着手直後の Check は同じ着手の観測なので update の前に対で入れる
            // （実戦では自分の手番でまとめて update するため常に対になっている）
            if matches!(
                event,
                Observation::OpponentMoved { .. } | Observation::MyMove { .. }
            ) {
                if let Some(check @ Observation::Check { .. }) = events.get(i + 1) {
                    log.record(check.clone());
                    i += 1;
                }
            }
            i += 1;
            if should_update {
                est.update(&log);
            }
            if let Some(mn) = measure_at {
                let mut seen = HashSet::new();
                for pos in est.particles() {
                    if seen.len() >= SAMPLE_CAP {
                        break;
                    }
                    seen.insert(pos.fingerprint());
                }
                let b = bucket(mn);
                new_stats[b].0 += seen.len() as u64;
                new_stats[b].1 += est.healthy() as u64;
                new_stats[b].2 += 1;
            }
        }
    }

    println!("{games}局をリプレイ");
    println!("手数        記録当時（ユニーク/健全率）   現在のコード（ユニーク/健全率）");
    for b in 0..3 {
        let fmt = |(u, h, n): (u64, u64, u64)| {
            if n == 0 {
                "-".to_string()
            } else {
                format!("{:>5.0} / {:>3.0}% (n={n})", u as f64 / n as f64, h as f64 / n as f64 * 100.0)
            }
        };
        println!(
            "{:>5}手   {:>28}   {:>28}",
            BUCKET_NAMES[b],
            fmt(old_stats[b]),
            fmt(new_stats[b])
        );
    }
}
