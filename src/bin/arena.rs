//! 戦略同士をローカルで対戦させて勝率を測るアリーナ。
//! 対局ループ・裁定はライブラリ側（selfplay.rs）にあり、tune とも共用する。
//!
//! 使い方:
//!   cargo run --release --bin arena -- [対局数] [戦略A] [戦略B]
//!   cargo run --release --bin arena -- [対局数] [候補] [基準1] [基準2] ...
//!
//! 基準を複数並べるとガントレット: 候補が各基準と [対局数] ずつ対戦する。
//! 新戦略は直近の凍結版だけでなく過去の凍結版すべてに勝ち越すこと
//! （v2 に勝つが v1 に負ける、という非推移性の検出。src/frozen/ 参照）。

use tsuitate_bot::selfplay::{
    FISCHER_INCREMENT_MS, FISCHER_INITIAL_MS, MatchStats, run_match_with, run_match_with_seeds,
    thread_count,
};
use tsuitate_bot::strategy;

/// 思考時間の要約（平均 / p99 / 最大、ミリ秒）
fn think_summary(think_us: &[u64]) -> String {
    if think_us.is_empty() {
        return "-".into();
    }
    let mut sorted = think_us.to_vec();
    sorted.sort_unstable();
    let mean = sorted.iter().sum::<u64>() as f64 / sorted.len() as f64 / 1000.0;
    let p99 = sorted[(sorted.len() * 99 / 100).min(sorted.len() - 1)] as f64 / 1000.0;
    let max = *sorted.last().unwrap() as f64 / 1000.0;
    format!("平均 {mean:.1}ms / p99 {p99:.1}ms / 最大 {max:.1}ms")
}

fn print_match(stats: &MatchStats, name_a: &str, name_b: &str) {
    let (rate, ci) = stats.rate_and_ci();
    println!(
        "A={name_a}: {}勝 / B={name_b}: {}勝 / 引き分け {}",
        stats.wins_a, stats.wins_b, stats.draws
    );
    println!("Aの勝率（引き分け除く）: {:.1}% ± {:.1}%", rate * 100.0, ci * 100.0);
    println!(
        "終局理由: 詰み {} / ステイルメイト {} / 反則負け {} / 投了 {} / 時間切れ {} / 手数上限 {}",
        stats.checkmate, stats.stalemate, stats.foul_limit, stats.resign, stats.timeout,
        stats.max_plies
    );
    println!(
        "平均手数 {:.1} / 平均反則 A {:.2}（うち王手中 {:.2}） B {:.2}（うち王手中 {:.2}）",
        stats.total_plies as f64 / stats.games.max(1) as f64,
        stats.fouls_a as f64 / stats.games.max(1) as f64,
        stats.fouls_in_check_a as f64 / stats.games.max(1) as f64,
        stats.fouls_b as f64 / stats.games.max(1) as f64,
        stats.fouls_in_check_b as f64 / stats.games.max(1) as f64
    );
    println!("思考時間 A: {}", think_summary(&stats.think_us_a));
    println!("思考時間 B: {}", think_summary(&stats.think_us_b));
}

/// 1マッチアップの集計を機械可読に書き出す（CIのシャード集約用）。
/// think時間は生配列を持ち回れないため、シャード内の要約値だけを残す
fn summary_json(candidate: &str, baseline: &str, stats: &MatchStats) -> serde_json::Value {
    let quant = |us: &[u64]| -> (f64, f64) {
        if us.is_empty() {
            return (0.0, 0.0);
        }
        let mut sorted = us.to_vec();
        sorted.sort_unstable();
        (
            sorted.iter().sum::<u64>() as f64 / sorted.len() as f64 / 1000.0,
            sorted[(sorted.len() * 99 / 100).min(sorted.len() - 1)] as f64 / 1000.0,
        )
    };
    let (a_avg, a_p99) = quant(&stats.think_us_a);
    let (b_avg, b_p99) = quant(&stats.think_us_b);
    serde_json::json!({
        "candidate": candidate,
        "baseline": baseline,
        "games": stats.games,
        "wins_a": stats.wins_a,
        "wins_b": stats.wins_b,
        "draws": stats.draws,
        "checkmate": stats.checkmate,
        "stalemate": stats.stalemate,
        "foul_limit": stats.foul_limit,
        "resign": stats.resign,
        "timeout": stats.timeout,
        "max_plies": stats.max_plies,
        "total_plies": stats.total_plies,
        "fouls_a": stats.fouls_a,
        "fouls_b": stats.fouls_b,
        "fouls_in_check_a": stats.fouls_in_check_a,
        "fouls_in_check_b": stats.fouls_in_check_b,
        "think_avg_ms_a": a_avg,
        "think_p99_ms_a": a_p99,
        "think_avg_ms_b": b_avg,
        "think_p99_ms_b": b_p99,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // ARENA_MATCH_SEED: 対局条件（定跡・推定器シード等）を決定論化する共通seed。
    // アブレーション比較（版Aと版Bを同じ対局条件列で戦わせて差分を見る）用
    let match_seed: Option<u64> = std::env::var("ARENA_MATCH_SEED")
        .ok()
        .and_then(|v| v.parse().ok());
    let games: u32 = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(100);
    let candidate = args.get(2).cloned().unwrap_or_else(|| "heuristic".into());
    let opponents: Vec<String> = if args.len() > 3 {
        args[3..].to_vec()
    } else {
        vec!["heuristic".into()]
    };
    for name in std::iter::once(&candidate).chain(&opponents) {
        if strategy::make(name).is_none() {
            eprintln!("未知の戦略名です: {name}");
            std::process::exit(1);
        }
    }

    let mut results: Vec<(String, MatchStats)> = vec![];
    for (opp_idx, opp) in opponents.iter().enumerate() {
        println!(
            "=== アリーナ: {candidate} (A) vs {opp} (B), {games}局（先後交代・フィッシャー{}秒+{}秒・並列{}{}） ===",
            FISCHER_INITIAL_MS / 1000,
            FISCHER_INCREMENT_MS / 1000,
            thread_count().min(games.max(1) as usize),
            match match_seed {
                Some(s) => format!("・seed {s}"),
                None => String::new(),
            },
        );
        let stats = match match_seed {
            Some(seed) => run_match_with_seeds(
                games,
                // 基準ごとにずらす（同じ基準に対してだけ同一条件列になる）
                seed ^ (opp_idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                &|gs| strategy::make_seeded(&candidate, gs.seed).expect("検証済みの戦略名"),
                &|gs| strategy::make_seeded(opp, gs.seed).expect("検証済みの戦略名"),
            ),
            None => run_match_with(
                games,
                &|| strategy::make(&candidate).expect("検証済みの戦略名"),
                &|| strategy::make(opp).expect("検証済みの戦略名"),
            ),
        };
        print_match(&stats, &candidate, opp);
        println!();
        results.push((opp.clone(), stats));
    }

    // ARENA_JSON: 集計をJSONL（1行=1マッチアップ）で書き出す（CIのシャード集約用）
    if let Ok(path) = std::env::var("ARENA_JSON") {
        if !path.is_empty() {
            let lines: Vec<String> = results
                .iter()
                .map(|(opp, stats)| summary_json(&candidate, opp, stats).to_string())
                .collect();
            std::fs::write(&path, lines.join("\n") + "\n")
                .unwrap_or_else(|e| eprintln!("ARENA_JSON を書き込めません（{path}）: {e}"));
        }
    }

    // ガントレット時のみ総合サマリ（非推移性の一覧確認用）
    if results.len() > 1 {
        println!("=== 総合: {candidate} の対戦成績 ===");
        let mut total = MatchStats::default();
        for (opp, stats) in &results {
            let (rate, ci) = stats.rate_and_ci();
            println!(
                "vs {opp}: {:.1}% ± {:.1}% ({}-{}-{})",
                rate * 100.0,
                ci * 100.0,
                stats.wins_a,
                stats.wins_b,
                stats.draws
            );
            total.wins_a += stats.wins_a;
            total.wins_b += stats.wins_b;
            total.draws += stats.draws;
        }
        let (rate, ci) = total.rate_and_ci();
        println!(
            "合計: {:.1}% ± {:.1}% ({}-{}-{})",
            rate * 100.0,
            ci * 100.0,
            total.wins_a,
            total.wins_b,
            total.draws
        );
    }
}
