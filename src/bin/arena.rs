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

use tsuitate_bot::config;

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

/// 基準ごとにずらした**実効 match seed**。
///
/// `ARENA_MATCH_SEED` をそのまま使うと、ガントレットの全基準が同じ対局条件列に
/// なる（同じ基準に対してだけ同一条件列にしたい）。**式はここ1箇所**に置く:
/// 対局の生成（`run_match_with_seeds`）と記録（`ARENA_GAMES_JSON` / `ARENA_JSON`）で
/// 食い違うと、記録された seed から対局条件を復元できなくなる。
fn baseline_match_seed(seed_env: u64, opp_idx: usize) -> u64 {
    seed_env ^ (opp_idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// `ARENA_MATCH_SEED` / `ARENA_MATCH_SEED_BASE` / `ARENA_SHARD` の整合性検査。
///
/// **base と shard は下流へ渡すラベルでしかなく、実際に対局条件を決めるのは
/// `ARENA_MATCH_SEED` だけ**。provenance 関門（check-prep.yml）は summary の
/// `match_seed_base` を見るので、ラベルと実物が食い違うと
/// 「`expect_match_seed=20260829` を指定したのに対局は別の seed で回っていた」を
/// 通してしまう（PR #35 レビュー3巡目 [P2]。実際に
/// `ARENA_MATCH_SEED=7 ARENA_MATCH_SEED_BASE=20260829 ARENA_SHARD=0` で再現した）。
///
/// そこで **3値のどれか1つでも指定されたら**（＝CI 経路だと分かったら）3値が揃い
/// `match_seed == base + shard` であることを要求し、違えば run 自体を落とす。
/// **式の複製を check-prep 側へ増やさない**ための場所でもある: 照合は
/// 「起動時にここで検査済み」という不変量に乗る。
///
/// base / shard を渡さないローカル実行（`ARENA_MATCH_SEED` だけ・何も無し）は従来どおり。
fn check_seed_provenance(
    match_seed: Option<u64>,
    base: Option<u64>,
    shard: Option<u64>,
) -> Result<(), String> {
    if base.is_none() && shard.is_none() {
        // ラベルを名乗っていないので検査対象外（ローカル実行）
        return Ok(());
    }
    let (Some(seed), Some(base), Some(shard)) = (match_seed, base, shard) else {
        return Err(format!(
            "ARENA_MATCH_SEED / _BASE / ARENA_SHARD は3つ揃えて指定してください\
             （片欠けだと記録の match_seed_base が実際の対局条件と対応しません）: \
             match_seed={match_seed:?} base={base:?} shard={shard:?}"
        ));
    };
    let expect = base.checked_add(shard).ok_or_else(|| {
        format!("ARENA_MATCH_SEED_BASE + ARENA_SHARD が桁あふれします: {base} + {shard}")
    })?;
    if seed != expect {
        return Err(format!(
            "ARENA_MATCH_SEED が base + shard と一致しません: \
             match_seed={seed} だが base={base} + shard={shard} = {expect}。\
             記録される match_seed_base は実際の対局条件と対応しないので中止します"
        ));
    }
    Ok(())
}

/// **実行の識別子**（PR #41 レビュー4巡目 [P1]）。
///
/// `match_seed_base` は**実験条件**であって実行の識別子ではない: 同じ base で
/// 取り直した2つの run から shard を半分ずつ選ぶと、base は1値・shard 集合は
/// 完全・局数も一致して、下流の「複数 run を混ぜない」検査をすべて通ってしまう。
/// 壁時計予算で同じ seed でも結果が揺れるこのリポジトリでは、それは
/// 「門付近での取り直し」を機械検査できないことと同じ。そこで
/// `ARENA_GAMES_JSON` を書く run には実行の識別子を必須にする:
/// CI は `GITHUB_RUN_ID` / `GITHUB_RUN_ATTEMPT`（Actions が全ジョブに立てる
/// 既定 env。re-run attempt も別実行として区別する）、ローカルは明示的な
/// `ARENA_EXPERIMENT_ID`（attempt は 1）。どちらも無ければ起動時に落とす
/// （書き出し時に落とすと対局が丸ごと無駄になる）。
fn resolve_run_identity(
    github_run_id: Option<String>,
    github_run_attempt: Option<String>,
    experiment_id: Option<String>,
) -> Result<(String, u64), String> {
    if let Some(id) = github_run_id.filter(|s| !s.trim().is_empty()) {
        // ID があるのに attempt が無い・数値でないときは黙って 1 にしない
        // （check_seed_provenance と同じ姿勢: 欠測は「一致」ではない）
        let attempt = github_run_attempt
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| "GITHUB_RUN_ID があるのに GITHUB_RUN_ATTEMPT がありません".to_string())?
            .parse::<u64>()
            .map_err(|_| "GITHUB_RUN_ATTEMPT は非負整数で指定してください".to_string())?;
        return Ok((id.trim().to_string(), attempt));
    }
    if let Some(id) = experiment_id.filter(|s| !s.trim().is_empty()) {
        return Ok((id.trim().to_string(), 1));
    }
    Err("ARENA_GAMES_JSON には実行の識別子が必須です（同じ base seed で取り直した\n\
         複数 run の混入検査に使う）。CI は GITHUB_RUN_ID / GITHUB_RUN_ATTEMPT（自動）、\n\
         ローカルは ARENA_EXPERIMENT_ID=<一意な名前> を指定してください"
        .into())
}

/// validation manifest の指紋（issue #40 の held-out 採否用、PR #41 レビュー4巡目）。
///
/// 処置ノブのような「P1 の後に決まる可変部分」は合算器の定数にできないので、
/// **計測前に commit した manifest ファイル**を `ARENA_BALANCE_MANIFEST=<path>` で
/// 指し、その sha256 を games.jsonl の全行へ焼き込む。合算器
/// （`checkpoint_arena arena-balance --manifest`）は同じファイルの指紋と
/// 行の指紋の一致を要求するので、「manifest と違う設定で測った run」や
/// 「計測後に manifest を書き換えた」が判定へ混ざらない。起動時に検査する
/// （対局を回してから落とすと 600 局が無駄になる。PR #41 レビュー5巡目で
/// 検査を3点足した）:
/// - candidate 側の実効ノブが manifest の `cand_knobs` と一致（対照 = ノブなしは空でよい）
/// - manifest が `candidate` / `think_budget_ms` を持つなら実効値と一致
/// - **診断オラクル（`ARENA_ORACLE_A`）の run には焼き込めない**: 候補の反則を
///   審判が握りつぶす上限測定を通常の held-out として通してはいけない
fn balance_manifest_fingerprint(candidate: &str) -> Option<String> {
    let path = std::env::var("ARENA_BALANCE_MANIFEST")
        .ok()
        .filter(|p| !p.trim().is_empty())?;
    let bail = |msg: &str| -> ! {
        eprintln!("{msg}");
        std::process::exit(1);
    };
    if !oracle_mode().is_empty() {
        bail(&format!(
            "ARENA_ORACLE_A が有効な run に ARENA_BALANCE_MANIFEST（{path}）は焼き込めません\n\
             （オラクルは診断用の上限測定で、held-out 採否の記録にはならない）"
        ));
    }
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| bail(&format!("ARENA_BALANCE_MANIFEST を読めません（{path}）: {e}")));
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_else(|e| {
        bail(&format!("ARENA_BALANCE_MANIFEST（{path}）が JSON として読めません: {e}"))
    });
    let want: BTreeMap<String, String> = match v.get("cand_knobs").and_then(|x| x.as_object()) {
        Some(o) => o
            .iter()
            .map(|(k, val)| {
                (
                    k.clone(),
                    val.as_str().map(str::to_string).unwrap_or_else(|| val.to_string()),
                )
            })
            .collect(),
        None => bail(&format!(
            "ARENA_BALANCE_MANIFEST（{path}）に cand_knobs（object）がありません"
        )),
    };
    let actual = cand_knobs();
    if !actual.is_empty() && actual != want {
        bail(&format!(
            "ARENA_CAND_KNOBS が manifest（{path}）の cand_knobs と一致しません。\n\
             candidate run は manifest の処置ノブそのもの、対照 run はノブなしで回してください"
        ));
    }
    // **candidate / think_budget_ms は必須**（PR #41 レビュー6巡目 [P2]:
    // 「あれば検査」だと、欠けた manifest でも指紋を焼いて 600 局を完走し、
    // 4 run 後の合算器で初めて判定不能になる。契約どおり起動時に落とす）
    let mc = v
        .get("candidate")
        .and_then(|x| x.as_str())
        .unwrap_or_else(|| {
            bail(&format!(
                "ARENA_BALANCE_MANIFEST（{path}）に candidate（文字列）がありません（合算器の必須フィールド）"
            ))
        });
    if mc != candidate {
        bail(&format!(
            "candidate {candidate} が manifest（{path}）の candidate {mc} と一致しません"
        ));
    }
    let mb = v
        .get("think_budget_ms")
        .and_then(|x| x.as_u64())
        .unwrap_or_else(|| {
            bail(&format!(
                "ARENA_BALANCE_MANIFEST（{path}）に think_budget_ms（非負整数）がありません（合算器の必須フィールド）"
            ))
        });
    let eff = budget_of(candidate, &candidate_config());
    if eff != Some(mb) {
        bail(&format!(
            "候補の実効思考予算 {eff:?} が manifest（{path}）の think_budget_ms {mb} と一致しません"
        ));
    }
    use sha2::Digest as _;
    Some(
        sha2::Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
    )
}

/// 診断オラクルの実効値（`ARENA_ORACLE_A`。空 = 通常対局）。games.jsonl へ
/// 必ず記録する: 記録が無いと「対照は通常・候補は nofoul」の run を合算器が
/// 識別できず、審判が候補の反則を握りつぶした診断 run を held-out として
/// 通せてしまう（PR #41 レビュー5巡目 [P1]）
fn oracle_mode() -> String {
    std::env::var("ARENA_ORACLE_A")
        .map(|v| v.trim().to_string())
        .unwrap_or_default()
}

/// 数値 env を読む。**指定されているのに parse できないときは None で黙らない**
/// （黙って None にすると「指定していない」と区別できず、上の整合性検査を
/// すり抜ける）。
fn env_u64(key: &str) -> Result<Option<u64>, String> {
    match std::env::var(key) {
        Err(_) => Ok(None),
        Ok(v) if v.trim().is_empty() => Ok(None),
        Ok(v) => v
            .trim()
            .parse()
            .map(Some)
            .map_err(|_| format!("{key} は非負整数で指定してください: {v:?}")),
    }
}

/// 1マッチアップの集計を機械可読に書き出す（CIのシャード集約用）。
/// think時間は生配列を持ち回れないため、シャード内の要約値だけを残す
///
/// `seed` は `(ARENA_MATCH_SEED_BASE, ARENA_SHARD, ARENA_MATCH_SEED, 実効 seed)`。
/// **base と shard を明示して残す**のが要点（PR #35 レビュー2巡目 [P1]）:
/// 記録に残るのは XOR 済みの実効値なので、下流の診断が
/// 「この run はどの base seed で回したか」を式の複製なしに照合できるようにする。
fn summary_json(
    candidate: &str,
    baseline: &str,
    stats: &MatchStats,
    seed: (Option<u64>, Option<u64>, Option<u64>, Option<u64>),
) -> serde_json::Value {
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
        // candidate_config へ重ねるので env を読むと実際と食い違い、逆に候補が
        // 凍結版のときは candidate_config が実効ではない（版ごとの規則で読む）。
        // 予算の概念が無い heuristic は null（PR #22 再レビュー P2）
        "think_budget_ms_a": budget_of(candidate, &candidate_config()),
        // 候補側だけに効かせたノブと、両側の実効設定の指紋（issue #21）
        "cand_knobs": cand_knobs(),
        // **候補が config を尊重しない戦略のときは現行 config の指紋を入れない**
        // （凍結版は自分のファイル内の規則で動くので実効設定ではない。
        //  heuristic のように該当しない戦略は null）。PR #22 再レビュー P2
        "cand_config": behavior_of(candidate, &candidate_config()),
        // **基準側の実効挙動**の指紋（凍結版は版のソース・その版が読む env の実効値・
        // 共有モデルの pin から作る）。現行 config の指紋をそのまま入れると
        // 全 baseline で同じ値になり、実効設定を表さない（PR #22 レビュー指摘4）
        "baseline_behavior": behavior_of(baseline, &tsuitate_bot::config::ambient()),
        // **対局条件列の出どころ**（下流の診断が実験条件を機械的に照合するのに使う）。
        // `match_seed_env` は shard ずらし後の `ARENA_MATCH_SEED`、
        // `match_seed` はそこから基準ごとにずらした実効値
        "match_seed_base": seed.0,
        "match_seed_shard": seed.1,
        "match_seed_env": seed.2,
        "match_seed": seed.3,
    })
}

/// **候補側だけに効かせるノブ**（`ARENA_CAND_KNOBS="K=V K=V"`、issue #21）。
///
/// `-f env=` はプロセス env なので**両側に効く**（凍結版は自分のコピーの中で
/// env を読む）。候補側だけ変えたいときはこちらを使う: 値は
/// `StrategyConfig` として候補の instance にだけ渡り、プロセス env は動かない。
/// **凍結版は config を尊重しない**ので、候補が凍結版のときは使えない。
/// 実行時 cwd の HEAD（`ARENA_GAMES_JSON` の突き合わせ検査用）。
/// git が無い環境では空文字（検査は「両方空なら一致」で通る）
fn git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

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

/// プロセス env の `TSUITATE_*`（凍結版が読むのはこちら）。
fn process_env() -> BTreeMap<String, String> {
    std::env::vars()
        .filter(|(k, _)| k.starts_with("TSUITATE_"))
        .collect()
}

/// **戦略 `name` の実効挙動**の指紋（PR #22 レビュー指摘4 / 再レビュー P2）。
///
/// - config を尊重する現行 estimator 系 … 渡された instance config の指紋
/// - 凍結版 … 版のソース・その版が読む env の実効値・共有モデルの pin
/// - どちらでもない（`heuristic` など）… `null`（設定という概念が無い）
fn behavior_of(name: &str, cfg: &tsuitate_bot::config::StrategyConfig) -> Option<String> {
    behavior_of_with_env(name, cfg, &process_env())
}

/// **戦略 `name` の実効思考予算**。判定は [`behavior_of`] と同じ切り分け。
fn budget_of(name: &str, cfg: &tsuitate_bot::config::StrategyConfig) -> Option<u64> {
    budget_of_with_env(name, cfg, &process_env())
}

// 以下は**プロセス env を読まない純粋版**。凍結版が見る env を引数で受けるので、
// テストが呼び出し元の shell に残った `TSUITATE_*` に左右されない
// （PR #22 再レビュー P3。プロセス env をテスト内で set/remove すると
//  並列テストと競合するので、注入する形にしてある）。

fn behavior_of_with_env(
    name: &str,
    cfg: &tsuitate_bot::config::StrategyConfig,
    env: &BTreeMap<String, String>,
) -> Option<String> {
    if strategy::honors_config(name) {
        return Some(cfg.fingerprint());
    }
    tsuitate_bot::frozen::behavior_fingerprint(name, env)
}

fn budget_of_with_env(
    name: &str,
    cfg: &tsuitate_bot::config::StrategyConfig,
    env: &BTreeMap<String, String>,
) -> Option<u64> {
    if strategy::honors_config(name) {
        return Some(cfg.think_budget_ms);
    }
    tsuitate_bot::frozen::effective_think_budget_ms(name, env)
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
    // **shard ずらしの前の base seed と shard 番号**（CI が渡す。ローカル実行では
    // 空でよい）。記録に残るのは XOR 済みの実効値なので、これが無いと下流の診断は
    // 「どの base seed で回した run か」を式の複製なしには照合できない。
    // **ラベルと実物の一致は起動時にここで検査する**（PR #35 レビュー3巡目 [P2]）
    let (match_seed, match_seed_base, shard_index) = match (|| {
        let seed = env_u64("ARENA_MATCH_SEED")?;
        let base = env_u64("ARENA_MATCH_SEED_BASE")?;
        let shard = env_u64("ARENA_SHARD")?;
        check_seed_provenance(seed, base, shard)?;
        Ok::<_, String>((seed, base, shard))
    })() {
        Ok(v) => v,
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(1);
        }
    };
    // ARENA_GAMES_JSON を書く run は**実行の識別子**（と、あれば manifest 指紋）を
    // 起動時に解決する。書き出し時に落とすと数時間の対局が丸ごと無駄になる
    let games_json_path = std::env::var("ARENA_GAMES_JSON").ok().filter(|p| !p.is_empty());
    let run_identity: Option<(String, u64)> = games_json_path.as_ref().map(|_| {
        resolve_run_identity(
            std::env::var("GITHUB_RUN_ID").ok(),
            std::env::var("GITHUB_RUN_ATTEMPT").ok(),
            std::env::var("ARENA_EXPERIMENT_ID").ok(),
        )
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        })
    });
    // 候補 run が対照に指す run（arena.yml の `-f pair_with=`）。合算器が
    // 「対照の取り違え」を機械検査するために記録へ残す
    let pair_with = std::env::var("ARENA_PAIR_WITH").ok().filter(|s| !s.trim().is_empty());
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
    // manifest の照合は candidate の実効設定（名前・ノブ・予算）が要るのでここで
    let manifest_fp = balance_manifest_fingerprint(&candidate);

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
                baseline_match_seed(seed, opp_idx),
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

    // ARENA_GAMES_JSON: **1行=1対局**で書き出す（issue #19 の P0 の残り:
    // 同じ ARENA_MATCH_SEED の2本を局ごとに突き合わせて Var(delta) を実測する）。
    // 集計サマリー（ARENA_JSON）では勝敗の合計しか残らず、ペア差の分散が測れない。
    // `match_seed` が無いと局ごとの対局条件が揃わない = ペアにならないので、
    // そのときは何も書かずに理由を出す
    if let Ok(path) = std::env::var("ARENA_GAMES_JSON") {
        if !path.is_empty() {
            match match_seed {
                None => eprintln!(
                    "ARENA_GAMES_JSON は ARENA_MATCH_SEED と併用してください\n                                  （局ごとの対局条件が揃わないとペア差になりません）"
                ),
                Some(seed) => {
                    let knobs = cand_knobs();
                    let commit = git_commit();
                    let mut lines: Vec<String> = vec![];
                    for (opp_idx, (opp, stats)) in results.iter().enumerate() {
                        // 基準ごとのずらしは run_match_with_seeds の呼び出しと同じ関数
                        let match_seed_eff = baseline_match_seed(seed, opp_idx);
                        let mut games: Vec<&_> = stats.per_game.iter().collect();
                        games.sort_by_key(|g| g.game_no);
                        for g in games {
                            lines.push(
                                serde_json::json!({
                                    // schema 2: **実効条件の指紋を必須にした**
                                    // （PR #23 レビュー指摘2。名前と時計しか見ていないと、
                                    //  同じ `estimator_v14` でも共通 env・共有モデル pin・
                                    //  予算が違う2 run を「ペア」として受理してしまう）
                                    // schema 3: **base seed と shard を必須にした**
                                    // （PR #41 レビュー3巡目。記録に残る match_seed は
                                    //  シャードずらし＋基準ごとの XOR を掛けた実効値なので、
                                    //  「同じ run か」「shard が揃っているか」を下流が
                                    //  base / shard で検査できる必要がある）
                                    // schema 4: **実行の識別子を必須にした**
                                    // （PR #41 レビュー4巡目。base は実験条件であって
                                    //  実行の識別子ではないので、同じ base で取り直した
                                    //  複数 run の shard 混ぜはこれ無しでは検出できない）
                                    // schema 5: **診断オラクルを必須記録にした**
                                    // （同5巡目。無いと「対照は通常・候補は nofoul」を
                                    //  合算器が識別できず、審判が候補の反則を握りつぶした
                                    //  診断 run を held-out として通せる）
                                    "schema": 5,
                                    "oracle": oracle_mode(),
                                    "run_id": run_identity.as_ref().expect("起動時に解決済み").0,
                                    "run_attempt": run_identity.as_ref().expect("起動時に解決済み").1,
                                    // 候補 run が対照に指した run（無ければ null）。
                                    // 合算器が対照 run_id との一致を検査する
                                    "pair_with": pair_with,
                                    // 計測前に commit した validation manifest の指紋
                                    // （ARENA_BALANCE_MANIFEST。無い run は null =
                                    //  合算器の #40 採否では判定不能）
                                    "balance_manifest": manifest_fp,
                                    // **arm を突き合わせる前提の一部**。env アブレーション
                                    // なら両 run で同じでなければならない（違うなら
                                    // 測っているのは別 revision の差でもある）
                                    "commit": commit,
                                    "candidate": candidate,
                                    "baseline": opp,
                                    "match_seed": match_seed_eff,
                                    // ローカルの単発 run（CI の base/shard env なし）は
                                    // base = seed / shard = 0 とみなす（seed == base + shard）
                                    "match_seed_base": match_seed_base.unwrap_or(seed),
                                    "match_seed_shard": shard_index.unwrap_or(0),
                                    "game_no": g.game_no,
                                    "a_is_sente": g.a_is_sente,
                                    "score_a": g.score_a,
                                    "reason": g.reason,
                                    "plies": g.plies,
                                    "fouls_a": g.fouls_a,
                                    "fouls_b": g.fouls_b,
                                    "fouls_in_check_a": g.fouls_in_check_a,
                                    "fouls_in_check_b": g.fouls_in_check_b,
                                    "think_ms_a": g.think_ms_a,
                                    "think_ms_b": g.think_ms_b,
                                    "moves_a": g.moves_a,
                                    "moves_b": g.moves_b,
                                    // **両側の実効予算**。片側だけだと 700ms と 2000ms の
                                    // 2 run が「同じ条件のペア」として通る
                                    "think_budget_ms_a": budget_of(&candidate, &candidate_config()),
                                    "think_budget_ms_b": budget_of(opp, &config::ambient()),
                                    "cand_knobs": knobs,
                                    // 候補と固定相手の**実効挙動**の指紋（summary と同じ規約）
                                    "cand_config": behavior_of(&candidate, &candidate_config()),
                                    "baseline_behavior": behavior_of(opp, &config::ambient()),
                                    // 両側に効くプロセス env（`-f env=`）。ここが違えば
                                    // 固定相手の挙動も違う
                                    "shared_env": process_env(),
                                    "clock": [fischer_initial_ms(), fischer_increment_ms()],
                                })
                                .to_string(),
                            );
                        }
                    }
                    std::fs::write(&path, lines.join("\n") + "\n").unwrap_or_else(|e| {
                        eprintln!("ARENA_GAMES_JSON を書き込めません（{path}）: {e}")
                    });
                }
            }
        }
    }

    // ARENA_JSON: 集計をJSONL（1行=1マッチアップ）で書き出す（CIのシャード集約用）
    if let Ok(path) = std::env::var("ARENA_JSON") {
        if !path.is_empty() {
            let lines: Vec<String> = results
                .iter()
                .enumerate()
                .map(|(opp_idx, (opp, stats))| {
                    let seed = (
                        match_seed_base,
                        shard_index,
                        match_seed,
                        match_seed.map(|s| baseline_match_seed(s, opp_idx)),
                    );
                    summary_json(&candidate, opp, stats, seed).to_string()
                })
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

    /// **実効 seed の式は1箇所**（PR #35 レビュー2巡目 [P1]）。
    ///
    /// 記録に残るのは XOR 済みの実効値なので、対局生成と記録で式が食い違うと
    /// 「記録された seed から対局条件を復元する」ができなくなる。定数が動いたら
    /// 過去の記録と突き合わせられなくなるので、ここで固定する。
    #[test]
    fn 実効match_seedは基準ごとにずれる() {
        // 同じ base でも基準が違えば別の条件列
        assert_ne!(baseline_match_seed(20260828, 0), baseline_match_seed(20260828, 1));
        // 同じ (base, 基準) なら決定論的
        assert_eq!(baseline_match_seed(20260828, 0), baseline_match_seed(20260828, 0));
        // **XOR なので base を復元できる**（下流の診断はこの性質に乗る）
        for opp_idx in 0..4 {
            let eff = baseline_match_seed(20260828, opp_idx);
            let m = eff ^ 20260828u64;
            assert_eq!(m, (opp_idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        }
        // 定数そのもの（過去の記録との互換）
        assert_eq!(baseline_match_seed(0, 0), 0x9E37_79B9_7F4A_7C15);
    }

    /// **base seed のラベルと実際の対局条件の一致を起動時に検査する**
    /// （PR #35 レビュー3巡目 [P2]）。
    ///
    /// provenance 関門は summary の `match_seed_base` しか見ないので、
    /// ラベルだけ正しく実物が違う run を通せてしまう。レビューが再現した
    /// `ARENA_MATCH_SEED=7 ARENA_MATCH_SEED_BASE=20260829 ARENA_SHARD=0` を
    /// ここで固定する。
    #[test]
    fn base_seedのラベルと実効seedの食い違いは起動時に落ちる() {
        // レビューの再現ケース: ラベルは 20260829 なのに対局は 7 由来
        assert!(check_seed_provenance(Some(7), Some(20260829), Some(0)).is_err());
        // 正しい組（arena.yml が渡す形）
        assert!(check_seed_provenance(Some(20260829), Some(20260829), Some(0)).is_ok());
        assert!(check_seed_provenance(Some(20260831), Some(20260829), Some(2)).is_ok());
        // **片欠けも落とす**（base だけ名乗って shard を渡さない等）
        assert!(check_seed_provenance(Some(20260829), Some(20260829), None).is_err());
        assert!(check_seed_provenance(Some(20260829), None, Some(0)).is_err());
        assert!(check_seed_provenance(None, Some(20260829), Some(0)).is_err());
        // 桁あふれは通さない
        assert!(check_seed_provenance(Some(0), Some(u64::MAX), Some(1)).is_err());
        // ローカル実行（ラベルを名乗らない）は従来どおり
        assert!(check_seed_provenance(None, None, None).is_ok());
        assert!(check_seed_provenance(Some(20260829), None, None).is_ok());
    }

    /// **games.jsonl を書く run は実行の識別子が必須**（PR #41 レビュー4巡目 [P1]）。
    ///
    /// `match_seed_base` は実験条件であって実行の識別子ではないので、同じ base で
    /// 取り直した複数 run の shard 混ぜはこれ無しでは検出できない。CI は
    /// GITHUB_RUN_ID / GITHUB_RUN_ATTEMPT、ローカルは ARENA_EXPERIMENT_ID。
    #[test]
    fn 実行の識別子はgithub_run_idかexperiment_idのどちらかで必須() {
        let s = |v: &str| Some(v.to_string());
        // CI 経路（re-run attempt も識別子の一部）
        assert_eq!(
            resolve_run_identity(s("33604671318"), s("2"), None),
            Ok(("33604671318".into(), 2))
        );
        // ローカル経路（attempt は 1）
        assert_eq!(
            resolve_run_identity(None, None, s("issue40-local-a")),
            Ok(("issue40-local-a".into(), 1))
        );
        // GITHUB_RUN_ID があるなら experiment id より優先（CI の実体が勝つ）
        assert_eq!(
            resolve_run_identity(s("123"), s("1"), s("x")),
            Ok(("123".into(), 1))
        );
        // **ID があるのに attempt が欠測・非数値なら黙って 1 にしない**
        assert!(resolve_run_identity(s("123"), None, None).is_err());
        assert!(resolve_run_identity(s("123"), s(""), None).is_err());
        assert!(resolve_run_identity(s("123"), s("abc"), None).is_err());
        // どちらも無ければ落とす（空文字は未指定と同じ）
        assert!(resolve_run_identity(None, None, None).is_err());
        assert!(resolve_run_identity(s(""), None, s("  ")).is_err());
    }

    /// **summary に base seed と shard が残る**。これが無いと、下流は
    /// XOR 済みの実効値しか見られず `expect_match_seed` と照合できない
    #[test]
    fn summaryはbase_seedとshardを残す() {
        let stats = MatchStats::default();
        let v = summary_json(
            "estimator",
            "estimator_v14",
            &stats,
            (Some(20260828), Some(2), Some(20260830), Some(baseline_match_seed(20260830, 0))),
        );
        assert_eq!(v["match_seed_base"], 20260828);
        assert_eq!(v["match_seed_shard"], 2);
        assert_eq!(v["match_seed_env"], 20260830);
        assert_eq!(v["match_seed"], baseline_match_seed(20260830, 0));
        // seed 無しの run では null（欠測として下流が扱えるように）
        let v = summary_json("estimator", "estimator_v14", &stats, (None, None, None, None));
        assert!(v["match_seed_base"].is_null() && v["match_seed"].is_null());
    }

    /// **記録は候補の種別に合った実効値になる**（PR #22 再レビュー P2 の適用範囲）。
    ///
    /// `summary_json` は任意の戦略を A 側に取れるので、候補が凍結版・heuristic の
    /// ときに現行 estimator の config を記録すると嘘になる。
    #[test]
    fn 記録は候補の種別に合った実効値になる() {
        let cand_cfg = candidate_config_from(
            &EnvSource::empty(),
            &[(
                "TSUITATE_CAND_THINK_BUDGET_MS".to_string(),
                "777".to_string(),
            )]
            .into_iter()
            .collect(),
        );
        // **凍結版が見る env は注入する**（呼び出し元の shell に
        // TSUITATE_THINK_BUDGET_MS が残っていても結果が変わらないように）
        let env = BTreeMap::new();
        let budget = |n: &str| budget_of_with_env(n, &cand_cfg, &env);
        let behavior = |n: &str| behavior_of_with_env(n, &cand_cfg, &env);

        // 現行 estimator 系は instance config がそのまま実効
        assert_eq!(budget("estimator"), Some(777));
        assert!(behavior("estimator").is_some());
        // **凍結版は候補でも instance config を使わない**（版ごとの読み取り規則）。
        // v14 は候補専用の名前を読まないので、env 未設定なら既定 2000
        assert_eq!(budget("estimator_v14"), Some(2000));
        assert_ne!(
            behavior("estimator_v14"),
            behavior("estimator_v13"),
            "版が違えば指紋も違う"
        );
        assert_ne!(behavior("estimator_v14"), behavior("estimator"));
        // 予算・設定の概念が無い戦略は null
        assert_eq!(budget("heuristic"), None);
        assert_eq!(behavior("heuristic"), None);

        // **凍結版は注入した env に従う**（共通の名前はどの版も読む）
        let with_common: BTreeMap<String, String> =
            [("TSUITATE_THINK_BUDGET_MS".to_string(), "555".to_string())]
                .into_iter()
                .collect();
        assert_eq!(
            budget_of_with_env("estimator_v14", &cand_cfg, &with_common),
            Some(555)
        );
        // 現行 estimator は instance config が優先（env には従わない）
        assert_eq!(
            budget_of_with_env("estimator", &cand_cfg, &with_common),
            Some(777)
        );
    }

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
