//! 評価パラメータ（strategy::EvalParams）のSPSAチューニング。
//!
//! 目的関数はアリーナのスコア率（勝ち=1 / 引き分け=0.5 / 負け=0）で、
//! 基準戦略との対局を SPSA の2点評価（θ+cΔ / θ−cΔ）で繰り返し、
//! 正規化座標（各パラメータの探索範囲 [lo,hi] を [0,1] に写像）上で更新する。
//! SPSA は評価がノイジーでも次元数によらず1反復2評価で勾配を推定できるので、
//! 「100局±10pt」のアリーナを目的関数にする用途に向く。
//!
//! 使い方:
//!   cargo run --release --bin tune -- [反復数=40] [評価あたり対局数=60] [基準...=estimator_v5]
//!
//! - 進捗と各反復のパラメータは tune-log.jsonl に追記する
//! - **再開**: tune-log.jsonl が存在すれば最後の反復のθから自動で続きを実行する
//!   （反復番号も引き継ぐ。最初からやり直すときはファイルを削除する）
//! - 最後に「最終パラメータ」を出力する。採用は人間が判断し、
//!   strategy.rs の Default を書き換えてガントレットで確認する

use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;

use tsuitate_bot::selfplay::run_match_with;
use tsuitate_bot::strategy::{self, EstimatorStrategy, EvalParams};

/// 正規化座標 u ∈ [0,1]^d とパラメータの相互変換
fn to_params(u: &[f64]) -> EvalParams {
    let v: Vec<f64> = EvalParams::SPECS
        .iter()
        .zip(u)
        .map(|(spec, &ui)| spec.lo + ui.clamp(0.0, 1.0) * (spec.hi - spec.lo))
        .collect();
    EvalParams::from_vec(&v)
}

fn to_unit(params: &EvalParams) -> Vec<f64> {
    EvalParams::SPECS
        .iter()
        .zip(params.to_vec())
        .map(|(spec, v)| ((v - spec.lo) / (spec.hi - spec.lo)).clamp(0.0, 1.0))
        .collect()
}

/// スコア率（勝ち1 / 引き分け0.5）を基準戦略ごとに測って平均する
fn fitness(params: &EvalParams, games_per_eval: u32, baselines: &[String]) -> f64 {
    let per = (games_per_eval / baselines.len() as u32).max(2);
    let mut total = 0.0;
    for baseline in baselines {
        let stats = run_match_with(
            per,
            &|| Box::new(EstimatorStrategy::with_params(params.clone())),
            &|| strategy::make(baseline).expect("検証済みの戦略名"),
        );
        total += stats.score_rate();
    }
    total / baselines.len() as f64
}

fn log_line(file: &mut std::fs::File, value: &serde_json::Value) {
    use std::io::Write;
    if let Ok(line) = serde_json::to_string(value) {
        let _ = writeln!(file, "{line}");
        let _ = file.flush();
    }
}

fn params_json(params: &EvalParams) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (spec, v) in EvalParams::SPECS.iter().zip(params.to_vec()) {
        map.insert(
            spec.name.to_string(),
            serde_json::Value::from((v * 10000.0).round() / 10000.0),
        );
    }
    serde_json::Value::Object(map)
}

/// params_json の逆変換。ログに無いパラメータ（旧バージョンのログ等）は既定値のまま
fn params_from_json(v: &serde_json::Value) -> Option<EvalParams> {
    let obj = v.as_object()?;
    let mut vals = EvalParams::default().to_vec();
    for (i, spec) in EvalParams::SPECS.iter().enumerate() {
        if let Some(x) = obj.get(spec.name).and_then(|x| x.as_f64()) {
            vals[i] = x;
        }
    }
    Some(EvalParams::from_vec(&vals))
}

/// tune-log.jsonl の最後の反復から（次の反復番号, θ）を復元する
fn resume_state() -> Option<(u32, EvalParams)> {
    let content = std::fs::read_to_string("tune-log.jsonl").ok()?;
    let mut last: Option<(u32, EvalParams)> = None;
    for line in content.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if v["type"] == "iter" {
            let k = v["k"].as_u64().unwrap_or(0) as u32;
            if let Some(p) = params_from_json(&v["theta"]) {
                last = Some((k + 1, p));
            }
        }
    }
    last
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iterations: u32 = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(40);
    let games_per_eval: u32 = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(60);
    let baselines: Vec<String> = if args.len() > 3 {
        args[3..].to_vec()
    } else {
        vec!["estimator_v5".into()]
    };
    for name in &baselines {
        if strategy::make(name).is_none() {
            eprintln!("未知の戦略名です: {name}");
            std::process::exit(1);
        }
    }

    // SPSA係数（正規化座標）。c0: 摂動幅（範囲の8%）、a0/A/α/γ: 標準的な減衰
    let c0 = 0.08;
    let a0 = 0.15;
    let big_a = 10.0;
    let alpha = 0.602;
    let gamma = 0.101;

    let d = EvalParams::SPECS.len();
    let mut u = to_unit(&EvalParams::default());
    let mut start_k = 1u32;
    if let Some((next_k, params)) = resume_state() {
        u = to_unit(&params);
        start_k = next_k;
        println!(
            "tune-log.jsonl から再開: 反復{start_k}〜（最初からやり直すときはファイルを削除）"
        );
    }
    if start_k > iterations {
        println!("指定の反復数（{iterations}）は既に完了しています");
        return;
    }
    let mut rng = StdRng::seed_from_u64(rand::rng().random());
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("tune-log.jsonl")
        .expect("tune-log.jsonl を開けない");

    println!(
        "SPSA開始: 反復{start_k}〜{iterations} × 2評価 × {games_per_eval}局, 基準 {baselines:?}, {d}次元"
    );
    log_line(
        &mut log,
        &serde_json::json!({
            "type": "start",
            "iterations": iterations,
            "games_per_eval": games_per_eval,
            "baselines": baselines,
            "initial": params_json(&EvalParams::default()),
        }),
    );

    // 最良評価点（ノイズがあるので参考値。最終的な採用判断はガントレットで行う）
    let mut best_score = f64::MIN;
    let mut best_params = EvalParams::default();

    for k in start_k..=iterations {
        let ck = c0 / (k as f64).powf(gamma);
        let ak = a0 / (big_a + k as f64).powf(alpha);
        let delta: Vec<f64> = (0..d)
            .map(|_| if rng.random_bool(0.5) { 1.0 } else { -1.0 })
            .collect();

        let u_plus: Vec<f64> = u
            .iter()
            .zip(&delta)
            .map(|(ui, di)| (ui + ck * di).clamp(0.0, 1.0))
            .collect();
        let u_minus: Vec<f64> = u
            .iter()
            .zip(&delta)
            .map(|(ui, di)| (ui - ck * di).clamp(0.0, 1.0))
            .collect();

        let p_plus = to_params(&u_plus);
        let p_minus = to_params(&u_minus);
        let f_plus = fitness(&p_plus, games_per_eval, &baselines);
        let f_minus = fitness(&p_minus, games_per_eval, &baselines);

        // 勾配上昇（最大化）。Δ_i = ±1 なので 1/Δ_i = Δ_i
        let g = (f_plus - f_minus) / (2.0 * ck);
        for (ui, di) in u.iter_mut().zip(&delta) {
            *ui = (*ui + ak * g * di).clamp(0.0, 1.0);
        }

        for (score, params) in [(f_plus, &p_plus), (f_minus, &p_minus)] {
            if score > best_score {
                best_score = score;
                best_params = params.clone();
            }
        }

        let current = to_params(&u);
        println!(
            "[{k}/{iterations}] f+={f_plus:.3} f-={f_minus:.3} |g|={:.3} best={best_score:.3}",
            g.abs()
        );
        log_line(
            &mut log,
            &serde_json::json!({
                "type": "iter",
                "k": k,
                "f_plus": f_plus,
                "f_minus": f_minus,
                "theta": params_json(&current),
            }),
        );
    }

    let final_params = to_params(&u);
    println!("\n=== 最終パラメータ（SPSA収束点） ===");
    println!("{}", serde_json::to_string_pretty(&params_json(&final_params)).unwrap());
    println!("\n=== 最良評価点（参考: score={best_score:.3}、ノイズ込み） ===");
    println!("{}", serde_json::to_string_pretty(&params_json(&best_params)).unwrap());
    log_line(
        &mut log,
        &serde_json::json!({
            "type": "done",
            "final": params_json(&final_params),
            "best": params_json(&best_params),
            "best_score": best_score,
        }),
    );
    println!("\n採用する場合は strategy.rs の EvalParams::default を最終パラメータで書き換え、");
    println!("フルガントレット（全凍結版に勝ち越し）で確認すること。");
}
