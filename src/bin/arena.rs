//! 戦略同士をローカルで対戦させて勝率を測るアリーナ。
//! 対局ループ・裁定はライブラリ側（selfplay.rs）にあり、tune とも共用する。
//!
//! 使い方:
//!   cargo run --release --bin arena -- [対局数] [戦略A] [戦略B]
//!   cargo run --release --bin arena -- [対局数] [候補] [基準1] [基準2] ...
//!
//! 基準を複数並べるとガントレット: 候補が各基準と [対局数] ずつ対戦する。
//! 新戦略の合格条件は凍結版への勝ち越し。既定の対象は v9 以降（src/frozen/ 参照）。
//! 引数を省略したときの戦略名は `heuristic`（本番既定の `estimator` ではない）。

use std::collections::BTreeMap;
use std::sync::Arc;

use tsuitate_bot::selfplay::{
    MatchStats, fischer_increment_ms, fischer_initial_ms, run_match_with, run_match_with_seeds,
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
    // クロック消費率: 支給された持ち時間のうち実際に使った割合。
    // 時間配分（docs/improvement-plan-2026-07-26-yaneuraou.md 項目A）の伸びしろ
    let clock_line = |used_us: &[u64], granted_ms: u64, min_ms: Option<i64>| -> String {
        let used_ms = used_us.iter().sum::<u64>() as f64 / 1000.0;
        let pct = if granted_ms > 0 {
            used_ms / granted_ms as f64 * 100.0
        } else {
            0.0
        };
        format!(
            "消費 {:.1}% （{:.0}秒 / 支給 {:.0}秒）/ 残り最小 {}",
            pct,
            used_ms / 1000.0,
            granted_ms as f64 / 1000.0,
            match min_ms {
                Some(ms) => format!("{:.1}秒", ms as f64 / 1000.0),
                None => "-".into(),
            }
        )
    };
    println!(
        "クロック A: {}",
        clock_line(&stats.think_us_a, stats.clock_granted_ms_a, stats.clock_min_ms_a)
    );
    println!(
        "クロック B: {}",
        clock_line(&stats.think_us_b, stats.clock_granted_ms_b, stats.clock_min_ms_b)
    );
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
        // 時間配分の検証用（項目A）。消費 = think の総和、支給 = 初期＋加算の総和
        "clock_used_ms_a": stats.think_us_a.iter().sum::<u64>() / 1000,
        "clock_used_ms_b": stats.think_us_b.iter().sum::<u64>() / 1000,
        "clock_granted_ms_a": stats.clock_granted_ms_a,
        "clock_granted_ms_b": stats.clock_granted_ms_b,
        "clock_min_ms_a": stats.clock_min_ms_a,
        "clock_min_ms_b": stats.clock_min_ms_b,
        "fischer_initial_ms": fischer_initial_ms(),
        "fischer_increment_ms": fischer_increment_ms(),
        // **候補の実効予算**（数値）。`ARENA_CAND_KNOBS` はプロセス env を触らずに
        // candidate_config へ重ねるので、env を読むと実際と食い違う
        // （PR #22 再レビュー P2）。基準側は凍結版が読むプロセス env のまま
        "think_budget_ms_a": candidate_config().think_budget_ms,
        // 候補側だけに効かせたノブと、両側の実効設定の指紋（issue #21）
        "cand_knobs": cand_knobs(),
        "cand_config": candidate_config().fingerprint(),
        // **基準側の実効挙動**の指紋（凍結版は版のソース・その版が読む env の実効値・
        // 共有モデルの pin から作る）。現行 config の指紋をそのまま入れると
        // 全 baseline で同じ値になり、実効設定を表さない（PR #22 レビュー指摘4）
        "baseline_behavior": baseline_fingerprint(baseline),
    })
}

/// **候補側だけに効かせるノブ**（`ARENA_CAND_KNOBS="K=V K=V"`、issue #21）。
///
/// `-f env=` はプロセス env なので**両側に効く**（凍結版は自分のコピーの中で
/// env を読む）。候補側だけ変えたいときはこちらを使う: 値は
/// `StrategyConfig` として候補の instance にだけ渡り、プロセス env は動かない。
/// **凍結版は config を尊重しない**ので、候補が凍結版のときは使えない。
fn cand_knobs() -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let Ok(spec) = std::env::var("ARENA_CAND_KNOBS") else {
        return out;
    };
    for token in spec.split([',', ' ', '\n']).filter(|s| !s.trim().is_empty()) {
        let Some((k, v)) = token.trim().split_once('=') else {
            eprintln!("ARENA_CAND_KNOBS の書式は K=V です: {token}");
            std::process::exit(1);
        };
        if !k.starts_with("TSUITATE_") {
            eprintln!("ARENA_CAND_KNOBS は TSUITATE_* だけ許可されます: {k}");
            std::process::exit(1);
        }
        out.insert(k.to_string(), v.to_string());
    }
    // **綴り間違い・無効値の関門**（PR #22 レビュー指摘2）。接頭辞だけ見ていると
    // `TSUITATE_HAND_ASSET_WW=0.5` が通り、実効値は既定のままの実験が
    // 正常完走してしまう
    let check = tsuitate_bot::config::check_overrides(
        &tsuitate_bot::config::EnvSource::from_process(),
        &out,
    );
    if !check.unknown.is_empty() {
        eprintln!(
            "ARENA_CAND_KNOBS に戦略が読まないキーがあります（綴り間違い？）: {}",
            check.unknown.join(", ")
        );
        std::process::exit(1);
    }
    if !check.ineffective.is_empty() {
        eprintln!(
            "警告: ARENA_CAND_KNOBS の次のキーは実効値を変えませんでした\n                      （既定値と同じ値か、解釈できない・範囲外の値）: {}",
            check.ineffective.join(", ")
        );
    }
    out
}

/// **基準側の実効挙動**の指紋。凍結版は版のソース・その版が読む env の実効値・
/// 共有モデルの pin から、現行 estimator 系は ambient config の指紋から作る。
fn baseline_fingerprint(name: &str) -> String {
    let env: BTreeMap<String, String> = std::env::vars()
        .filter(|(k, _)| k.starts_with("TSUITATE_"))
        .collect();
    tsuitate_bot::frozen::behavior_fingerprint(name, &env)
        .unwrap_or_else(|| tsuitate_bot::config::ambient().fingerprint())
}

/// 候補側の実効設定（プロセス env にノブを重ねたもの）。
fn candidate_config() -> Arc<tsuitate_bot::config::StrategyConfig> {
    use tsuitate_bot::config::{EnvSource, StrategyConfig};
    static C: std::sync::OnceLock<Arc<StrategyConfig>> = std::sync::OnceLock::new();
    C.get_or_init(|| Arc::new(candidate_config_from(&EnvSource::from_process(), &cand_knobs())))
        .clone()
}

/// [`candidate_config`] の純粋版（プロセス env を読まないのでテストできる）。
fn candidate_config_from(
    base: &tsuitate_bot::config::EnvSource,
    knobs: &BTreeMap<String, String>,
) -> tsuitate_bot::config::StrategyConfig {
    tsuitate_bot::config::StrategyConfig::from_source(base.with_overrides(knobs.clone()))
}

/// 候補側の戦略を作る（ノブがあれば config で渡す）。
fn make_candidate(name: &str, seed: Option<u64>) -> Box<dyn strategy::Strategy + Send> {
    let knobs = cand_knobs();
    if knobs.is_empty() {
        return match seed {
            Some(s) => strategy::make_seeded(name, s),
            None => strategy::make(name),
        }
        .expect("検証済みの戦略名");
    }
    if !strategy::honors_config(name) {
        eprintln!(
            "ARENA_CAND_KNOBS は {name} には渡せません（凍結版は config を尊重せず\n             \
             凍結時点の env を読むため、黙って無視されます）"
        );
        std::process::exit(1);
    }
    // **seed は素通しする**。`seed.unwrap_or(0)` にすると ARENA_MATCH_SEED 未指定の
    // 通常アリーナで候補が全局 seed 0 になり、「ノブの有無」で乱数条件まで変わって
    // 対照との比較が崩れる（PR #22 レビュー指摘1）
    strategy::make_with_config(name, seed, candidate_config()).expect("検証済みの戦略名")
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
            fischer_initial_ms() / 1000,
            fischer_increment_ms() / 1000,
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
                &|gs| make_candidate(&candidate, Some(gs.seed)),
                &|gs| strategy::make_seeded(opp, gs.seed).expect("検証済みの戦略名"),
            ),
            None => run_match_with(
                games,
                &|| make_candidate(&candidate, None),
                &|| strategy::make(opp).expect("検証済みの戦略名"),
            ),
        };
        print_match(&stats, &candidate, opp);
        println!();
        results.push((opp.clone(), stats));
    }

    // 評価項の発火率（TSUITATE_DBG_HITS=1 のときだけ）。候補・基準の両側を
    // まとめた値だが、凍結版は last_ranking を作らないので実質候補側の統計
    if let Some(table) = tsuitate_bot::hits::dump() {
        println!("{table}");
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

#[cfg(test)]
mod tests {
    use super::*;
    use tsuitate_bot::config::EnvSource;

    /// **`cand_env` の思考予算が候補の実効値として反映される**
    /// （PR #22 再レビュー P2）。`ARENA_CAND_KNOBS` はプロセス env を触らないので、
    /// 記録側が env を読むと `cand_knobs` と `think_budget_ms_a` が食い違う。
    #[test]
    fn 候補ノブの思考予算が実効設定へ入る() {
        let base = EnvSource::empty();
        let knob = |k: &str, v: &str| -> BTreeMap<String, String> {
            [(k.to_string(), v.to_string())].into_iter().collect()
        };
        // 既定（凍結時点から変わらない基準値）
        assert_eq!(
            candidate_config_from(&base, &BTreeMap::new()).think_budget_ms,
            2000
        );
        // 候補専用の名前
        assert_eq!(
            candidate_config_from(&base, &knob("TSUITATE_CAND_THINK_BUDGET_MS", "777"))
                .think_budget_ms,
            777
        );
        // 共通の名前でも候補 config には効く（プロセス env は触らない）
        assert_eq!(
            candidate_config_from(&base, &knob("TSUITATE_THINK_BUDGET_MS", "555")).think_budget_ms,
            555
        );
        // 候補専用が優先
        let mut both = knob("TSUITATE_THINK_BUDGET_MS", "555");
        both.insert("TSUITATE_CAND_THINK_BUDGET_MS".into(), "777".into());
        assert_eq!(candidate_config_from(&base, &both).think_budget_ms, 777);
        // **基準側（プロセス env）は動かない**: base だけで解決した値は既定のまま
        assert_eq!(
            tsuitate_bot::config::StrategyConfig::from_source(base).think_budget_ms,
            2000
        );
    }
}
