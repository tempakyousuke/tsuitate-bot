//! 評価パラメータ（strategy::EvalParams）のSPSAチューニング。
//!
//! 目的関数はアリーナのスコア率（勝ち=1 / 引き分け=0.5 / 負け=0）で、
//! 基準戦略との対局を SPSA の2点評価（θ+cΔ / θ−cΔ）で繰り返し、
//! 正規化座標（各パラメータの探索範囲 [lo,hi] を [0,1] に写像）上で更新する。
//! SPSA は評価がノイジーでも次元数によらず1反復2評価で勾配を推定できるので、
//! 「100局±10pt」のアリーナを目的関数にする用途に向く。
//!
//! ノイズ対策（共通乱数法）: f+ と f− は同じ match_seed 列で評価する。
//! 対局番号から先後・定跡ライン・推定器シード・タイブレークが決定論的に
//! 決まるため（selfplay::GameSeeds）、両評価の差分から共通の運要素が消える。
//! 評価順（f+先/f−先）も反復ごとに入れ替えてドリフトを打ち消す。
//! 注意: 推定器の時間打ち切り（壁時計）は決定論化できないため完全一致はしない。
//!
//! 使い方:
//!   cargo run --release --bin tune -- [反復数=40] [評価あたり対局数=60] [基準...=estimator_v7]
//!
//! - 進捗と各反復のパラメータは TUNE_LOG（既定 tune-log.jsonl）に追記する
//! - **再開**: ログが存在すれば最後の反復のθから自動で続きを実行する。
//!   再開時は start イベントの設定（基準・局数・定跡固定・思考予算・パラメータ空間・
//!   ランシード）と一致するか検証し、不一致なら停止する（TUNE_FORCE_RESUME=1 で強行）
//! - 最後に最終中心点を追加評価して done に記録する。採用は人間が判断し、
//!   strategy.rs の Default を書き換えてガントレット（CI・200局）で確認する
//!
//! 環境変数:
//! - TUNE_LOG: ログファイルのパス（既定 tune-log.jsonl。実験ごとに分ける）
//! - TUNE_SEED: ランシード（既定はエントロピー。再開時は start から引き継ぐ）
//! - TUNE_CANDIDATE_LINE: 候補側の定跡をこのライン名に固定する
//!   （例: 居飛車速攻。基準側を固定するには estimator_rush を基準に指定する）
//! - TUNE_FORCE_RESUME=1: 設定不一致でも再開を強行する

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use tsuitate_bot::opening::OpeningBook;
use tsuitate_bot::selfplay::{MatchStats, run_match_with_seeds};
use tsuitate_bot::strategy::{self, EstimatorStrategy, EvalParams};

/// ログファイルのパス（実験ごとに分けられる）
fn log_path() -> String {
    std::env::var("TUNE_LOG").unwrap_or_else(|_| "tune-log.jsonl".into())
}

/// SplitMix64（selfplay::player_seed と同系の撹拌。シード導出に使う）
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// 調整空間: 各パラメータの調整対象マスクと有効範囲。
/// - TUNE_PARAMS（カンマ区切りの名前）: 調整する項目だけを動かす。未指定なら全項目。
///   方式変更に関係する項目に絞り、収束済み・無関係の項目を固定するために使う
/// - TUNE_SPAN（0.01..=1.0、既定1.0）: 調整対象の有効範囲を「現在の既定値を中心に
///   元範囲×span の幅」へ局所化する。広い旧範囲のままだと中心が端に近い項目は
///   初回摂動から片側クリップになる（2026-07-15 レビュー指摘）
struct TuneSpace {
    active: Vec<bool>,
    lo: Vec<f64>,
    hi: Vec<f64>,
    span: f64,
}

fn build_space(center: &EvalParams) -> Result<TuneSpace, String> {
    let list = std::env::var("TUNE_PARAMS").ok();
    let span: f64 = match std::env::var("TUNE_SPAN") {
        Ok(v) => v
            .parse()
            .map_err(|_| format!("TUNE_SPAN を数値として読めません: {v}"))?,
        Err(_) => 1.0,
    };
    build_space_from(list.as_deref(), span, center)
}

fn build_space_from(
    list: Option<&str>,
    span: f64,
    center: &EvalParams,
) -> Result<TuneSpace, String> {
    let d = EvalParams::SPECS.len();
    let mut active = vec![true; d];
    if let Some(list) = list {
        active = vec![false; d];
        for name in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let Some(i) = EvalParams::SPECS.iter().position(|s| s.name == name) else {
                return Err(format!("TUNE_PARAMS に未知のパラメータ名: {name}"));
            };
            active[i] = true;
        }
        if !active.iter().any(|&a| a) {
            return Err("TUNE_PARAMS に有効なパラメータ名がありません".into());
        }
    }
    if !(0.01..=1.0).contains(&span) {
        return Err(format!("TUNE_SPAN は 0.01..=1.0 の範囲で指定してください: {span}"));
    }
    let centers = center.to_vec();
    let mut lo = Vec::with_capacity(d);
    let mut hi = Vec::with_capacity(d);
    for (i, spec) in EvalParams::SPECS.iter().enumerate() {
        if active[i] && span < 1.0 {
            let half = span * (spec.hi - spec.lo) / 2.0;
            lo.push((centers[i] - half).max(spec.lo));
            hi.push((centers[i] + half).min(spec.hi));
        } else {
            lo.push(spec.lo);
            hi.push(spec.hi);
        }
    }
    Ok(TuneSpace { active, lo, hi, span })
}

/// 正規化座標 u ∈ [0,1]^d とパラメータの相互変換（space の有効範囲上）
fn to_params(u: &[f64], space: &TuneSpace) -> EvalParams {
    let v: Vec<f64> = space
        .lo
        .iter()
        .zip(&space.hi)
        .zip(u)
        .map(|((&lo, &hi), &ui)| lo + ui.clamp(0.0, 1.0) * (hi - lo))
        .collect();
    EvalParams::from_vec(&v)
}

fn to_unit(params: &EvalParams, space: &TuneSpace) -> Vec<f64> {
    space
        .lo
        .iter()
        .zip(&space.hi)
        .zip(params.to_vec())
        .map(|((&lo, &hi), v)| ((v - lo) / (hi - lo)).clamp(0.0, 1.0))
        .collect()
}

/// 基準ごとの対局数。切り捨てで指定局数を下回らないよう切り上げ、
/// 先後を揃えるため偶数に丸める
fn games_per_baseline(games_per_eval: u32, baselines: usize) -> u32 {
    let per = games_per_eval.div_ceil(baselines as u32).max(2);
    per + (per % 2)
}

/// クリップ済みの実際の評価点から勾配を推定する。
/// 通常は u_plus - u_minus = 2ckΔ_i だが、境界クリップで片側が縮んだ次元は
/// 実際に動いた距離を分母に使う。両点が同一（両側クリップ）の次元は勾配0
fn spsa_gradient(f_plus: f64, f_minus: f64, u_plus: &[f64], u_minus: &[f64]) -> Vec<f64> {
    u_plus
        .iter()
        .zip(u_minus)
        .map(|(&p, &m)| {
            let denom = p - m;
            if denom.abs() < 1e-12 {
                0.0
            } else {
                (f_plus - f_minus) / denom
            }
        })
        .collect()
}

/// ミリ秒統計（平均/p99/最大）
fn think_ms_stats(us: &[u64]) -> (f64, f64, f64) {
    if us.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut sorted = us.to_vec();
    sorted.sort_unstable();
    let avg = sorted.iter().sum::<u64>() as f64 / sorted.len() as f64 / 1000.0;
    let p99 = sorted[(sorted.len() * 99 / 100).min(sorted.len() - 1)] as f64 / 1000.0;
    let max = *sorted.last().unwrap() as f64 / 1000.0;
    (avg, p99, max)
}

/// 1基準ぶんの対局内訳（引き分け化・時間使用などのスコア外の変質を検出するため）
fn stats_json(baseline: &str, stats: &MatchStats) -> serde_json::Value {
    let (a_avg, a_p99, a_max) = think_ms_stats(&stats.think_us_a);
    let (b_avg, b_p99, b_max) = think_ms_stats(&stats.think_us_b);
    serde_json::json!({
        "baseline": baseline,
        "score_rate": stats.score_rate(),
        "wins": stats.wins_a,
        "losses": stats.wins_b,
        "draws": stats.draws,
        "endings": {
            "checkmate": stats.checkmate,
            "foul_limit": stats.foul_limit,
            "timeout": stats.timeout,
            "max_plies": stats.max_plies,
            "resign": stats.resign,
            "stalemate": stats.stalemate,
        },
        "avg_plies": stats.total_plies as f64 / stats.games.max(1) as f64,
        "fouls_per_game": {
            "candidate": stats.fouls_a as f64 / stats.games.max(1) as f64,
            "baseline": stats.fouls_b as f64 / stats.games.max(1) as f64,
        },
        "think_ms": {
            "candidate": { "avg": a_avg, "p99": a_p99, "max": a_max },
            "baseline": { "avg": b_avg, "p99": b_p99, "max": b_max },
        },
    })
}

/// スコア率を基準戦略ごとに測って平均する。
/// match_seeds は基準ごとの対局シード列で、同じ値で呼べば同じ対局条件になる
/// （f+/f− のペアリング）。詳細内訳も返す
fn fitness(
    params: &EvalParams,
    games_per_eval: u32,
    baselines: &[String],
    candidate_line: Option<usize>,
    match_seeds: &[u64],
) -> (f64, Vec<serde_json::Value>) {
    let per = games_per_baseline(games_per_eval, baselines.len());
    let mut total = 0.0;
    let mut details = vec![];
    for (baseline, &seed) in baselines.iter().zip(match_seeds) {
        let stats = run_match_with_seeds(
            per,
            seed,
            &|gs| {
                Box::new(EstimatorStrategy::with_params_line_seed(
                    params.clone(),
                    candidate_line,
                    Some(gs.seed),
                ))
            },
            &|gs| strategy::make_seeded(baseline, gs.seed).expect("検証済みの戦略名"),
        );
        total += stats.score_rate();
        details.push(stats_json(baseline, &stats));
    }
    (total / baselines.len() as f64, details)
}

/// ログ書き込み。失敗したら即座に落とす（Spot VM等で黙って進捗を失わないため）
fn log_line(file: &mut std::fs::File, value: &serde_json::Value) {
    use std::io::Write;
    let line = serde_json::to_string(value).expect("チューニングログをシリアライズできない");
    writeln!(file, "{line}")
        .and_then(|_| file.flush())
        .expect("チューニングログへ書き込めない（ディスク障害？進捗を失うため停止）");
}

/// 表示用（1e-4に丸め）。再開は theta_raw（完全精度）を使う
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

/// パラメータ空間の指紋（名前・有効範囲・調整対象）。再開時の互換性検証に使う
fn specs_json(space: &TuneSpace) -> serde_json::Value {
    serde_json::Value::Array(
        EvalParams::SPECS
            .iter()
            .enumerate()
            .map(|(i, s)| {
                serde_json::json!({
                    "name": s.name,
                    "lo": space.lo[i],
                    "hi": space.hi[i],
                    "active": space.active[i],
                })
            })
            .collect(),
    )
}

/// 今回のランの設定（start イベントに記録し、再開時に一致を検証する）
fn config_json(
    games_per_eval: u32,
    baselines: &[String],
    candidate_line_name: &Option<String>,
    space: &TuneSpace,
) -> serde_json::Value {
    serde_json::json!({
        "games_per_eval": games_per_eval,
        "baselines": baselines,
        "candidate_line": candidate_line_name,
        "think_budget_ms": std::env::var("TSUITATE_THINK_BUDGET_MS").ok(),
        "span": space.span,
        "specs": specs_json(space),
    })
}

struct Resume {
    next_k: u32,
    u: Vec<f64>,
    run_seed: Option<u64>,
    /// これまでの評価点の最良（eval イベントから復元）
    best: Option<(f64, Vec<f64>)>,
    /// 直近 start イベントの設定（互換性検証用）
    config: Option<serde_json::Value>,
}

/// ログから再開状態を復元する
fn resume_state(space: &TuneSpace) -> Option<Resume> {
    let content = std::fs::read_to_string(log_path()).ok()?;
    let mut resume: Option<Resume> = None;
    let mut run_seed = None;
    let mut config = None;
    let mut best: Option<(f64, Vec<f64>)> = None;
    for line in content.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        match v["type"].as_str() {
            Some("start") => {
                run_seed = v["run_seed"].as_u64();
                config = Some(v["config"].clone());
            }
            Some("eval") => {
                if let (Some(score), Some(u)) = (v["score"].as_f64(), unit_vec(&v["u"])) {
                    if best.as_ref().is_none_or(|(s, _)| score > *s) {
                        best = Some((score, u));
                    }
                }
            }
            Some("iter") => {
                let k = v["k"].as_u64().unwrap_or(0) as u32;
                // 完全精度の u を優先し、旧形式ログは丸めた theta から復元
                let u = unit_vec(&v["u"])
                    .or_else(|| params_from_json(&v["theta"]).map(|p| to_unit(&p, space)));
                if let Some(u) = u {
                    resume = Some(Resume {
                        next_k: k + 1,
                        u,
                        run_seed: None,
                        best: None,
                        config: None,
                    });
                }
            }
            _ => {}
        }
    }
    resume.map(|mut r| {
        r.run_seed = run_seed;
        r.best = best;
        r.config = config;
        r
    })
}

fn unit_vec(v: &serde_json::Value) -> Option<Vec<f64>> {
    let arr = v.as_array()?;
    if arr.len() != EvalParams::SPECS.len() {
        return None;
    }
    arr.iter().map(|x| x.as_f64()).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iterations: u32 = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(40);
    let games_per_eval: u32 = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(60);
    let baselines: Vec<String> = if args.len() > 3 {
        args[3..].to_vec()
    } else {
        vec!["estimator_v7".into()]
    };
    for name in &baselines {
        if strategy::make(name).is_none() {
            eprintln!("未知の戦略名です: {name}");
            std::process::exit(1);
        }
    }

    // 候補側の定跡固定（定跡特化チューニング）
    let candidate_line_name = std::env::var("TUNE_CANDIDATE_LINE").ok();
    let candidate_line = match &candidate_line_name {
        Some(name) => match OpeningBook::line_index(name) {
            Some(idx) => {
                println!("候補側の定跡を「{name}」に固定します");
                Some(idx)
            }
            None => {
                eprintln!("定跡ライン「{name}」が joseki.json にありません");
                std::process::exit(1);
            }
        },
        None => None,
    };

    // SPSA係数（正規化座標）。c0: 摂動幅（範囲の8%）、a0/A/α/γ: 標準的な減衰
    let c0 = 0.08;
    let a0 = 0.15;
    let big_a = 10.0;
    let alpha = 0.602;
    let gamma = 0.101;

    let d = EvalParams::SPECS.len();
    let space = match build_space(&EvalParams::default()) {
        Ok(space) => space,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    let active_count = space.active.iter().filter(|&&a| a).count();
    let config = config_json(games_per_eval, &baselines, &candidate_line_name, &space);
    let mut u = to_unit(&EvalParams::default(), &space);
    let mut start_k = 1u32;
    let mut run_seed: u64 = std::env::var("TUNE_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| rand::rng().random());
    // 最良評価点（ノイズがあるので参考値。最終的な採用判断はガントレットで行う）
    let mut best_score = f64::MIN;
    let mut best_u = u.clone();

    let mut resumed = false;
    if let Some(resume) = resume_state(&space) {
        // 設定の互換性を検証（不一致のまま続けると異なる目的関数を混ぜてしまう）
        if let Some(prev) = &resume.config {
            if *prev != config && std::env::var("TUNE_FORCE_RESUME").as_deref() != Ok("1") {
                eprintln!("再開しようとしたログと現在の設定が一致しません:");
                eprintln!("  ログ側: {prev}");
                eprintln!("  現在  : {config}");
                eprintln!("別ランは TUNE_LOG を分けるか、ログを消すか、TUNE_FORCE_RESUME=1 で強行してください");
                std::process::exit(1);
            }
        } else {
            eprintln!(
                "警告: ログに start イベントが無い（旧形式）。設定の一致は検証できません"
            );
        }
        u = resume.u;
        start_k = resume.next_k;
        if let Some(seed) = resume.run_seed {
            run_seed = seed; // シード列を引き継いで対局条件の連続性を保つ
        }
        if let Some((score, bu)) = resume.best {
            best_score = score;
            best_u = bu;
        }
        resumed = true;
        println!("{} から再開: 反復{start_k}〜（最初からやり直すときはファイルを削除）", log_path());
    }
    if start_k > iterations {
        println!("指定の反復数（{iterations}）は既に完了しています");
        return;
    }
    let mut rng = StdRng::seed_from_u64(mix(run_seed ^ 0x5EED));
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path())
        .expect("チューニングログを開けない");

    let per = games_per_baseline(games_per_eval, baselines.len());
    println!(
        "SPSA開始: 反復{start_k}〜{iterations} × 2評価 × {}局（{per}局×{}基準）, seed={run_seed}, {active_count}/{d}次元, span={}",
        per * baselines.len() as u32,
        baselines.len(),
        space.span,
    );
    if !resumed {
        log_line(
            &mut log,
            &serde_json::json!({
                "type": "start",
                "iterations": iterations,
                "run_seed": run_seed,
                "config": config,
                "initial": params_json(&EvalParams::default()),
            }),
        );
    }

    for k in start_k..=iterations {
        let ck = c0 / (k as f64).powf(gamma);
        let ak = a0 / (big_a + k as f64).powf(alpha);
        // 調整対象外の次元は摂動しない（u_plus=u_minus → 勾配0 → 更新なし）
        let delta: Vec<f64> = (0..d)
            .map(|i| {
                let sign = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
                if space.active[i] { sign } else { 0.0 }
            })
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

        // 共通乱数法: f+ と f− に同じ対局シード列を使う。
        // 評価順も反復ごとに入れ替える（実行環境のドリフト対策）
        let iter_seed = mix(run_seed ^ u64::from(k));
        let match_seeds: Vec<u64> = (0..baselines.len())
            .map(|i| mix(iter_seed ^ (i as u64 + 1)))
            .collect();
        let plus_first = (iter_seed >> 7) & 1 == 0;

        let p_plus = to_params(&u_plus, &space);
        let p_minus = to_params(&u_minus, &space);
        let eval = |params: &EvalParams| {
            fitness(params, games_per_eval, &baselines, candidate_line, &match_seeds)
        };
        let ((f_plus, det_plus), (f_minus, det_minus)) = if plus_first {
            let plus = eval(&p_plus);
            (plus, eval(&p_minus))
        } else {
            let minus = eval(&p_minus);
            (eval(&p_plus), minus)
        };
        for (which, u_pt, f, det) in [
            ("plus", &u_plus, f_plus, &det_plus),
            ("minus", &u_minus, f_minus, &det_minus),
        ] {
            log_line(
                &mut log,
                &serde_json::json!({
                    "type": "eval", "k": k, "which": which,
                    "u": u_pt, "score": f, "stats": det,
                }),
            );
            if f > best_score {
                best_score = f;
                best_u = u_pt.clone();
            }
        }

        // 勾配上昇（最大化）。境界クリップ時は実際に動いた距離を分母に使う
        let g = spsa_gradient(f_plus, f_minus, &u_plus, &u_minus);
        for (ui, gi) in u.iter_mut().zip(&g) {
            *ui = (*ui + ak * gi).clamp(0.0, 1.0);
        }

        let g_norm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        println!(
            "[{k}/{iterations}] f+={f_plus:.3} f-={f_minus:.3} |g|={g_norm:.3} best={best_score:.3}"
        );
        log_line(
            &mut log,
            &serde_json::json!({
                "type": "iter",
                "k": k,
                "f_plus": f_plus,
                "f_minus": f_minus,
                "plus_first": plus_first,
                "u": u,
                "theta": params_json(&to_params(&u, &space)),
            }),
        );
    }

    // 最終中心点は勾配更新の結果であってまだ評価されていないので、ここで測る
    let final_params = to_params(&u, &space);
    let final_seeds: Vec<u64> = (0..baselines.len())
        .map(|i| mix(run_seed ^ 0x000F_17A1 ^ (i as u64 + 1)))
        .collect();
    let (final_score, final_det) = fitness(
        &final_params,
        games_per_eval,
        &baselines,
        candidate_line,
        &final_seeds,
    );
    let best_params = to_params(&best_u, &space);
    println!("\n=== 最終パラメータ（SPSA収束点、score={final_score:.3}） ===");
    println!("{}", serde_json::to_string_pretty(&params_json(&final_params)).unwrap());
    println!("\n=== 最良評価点（参考: score={best_score:.3}、ノイズ込み） ===");
    println!("{}", serde_json::to_string_pretty(&params_json(&best_params)).unwrap());
    log_line(
        &mut log,
        &serde_json::json!({
            "type": "done",
            "final": params_json(&final_params),
            "final_u": u,
            "final_score": final_score,
            "final_stats": final_det,
            "best": params_json(&best_params),
            "best_score": best_score,
        }),
    );
    println!("\n採用する場合は strategy.rs の EvalParams::default を最終パラメータで書き換え、");
    println!("フルガントレット（CI・全凍結版に勝ち越し・僅差なら200局）で確認すること。");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn games_split_is_even_and_covers_request() {
        assert_eq!(games_per_baseline(60, 1), 60);
        assert_eq!(games_per_baseline(60, 4), 16); // ceil(15)→15→偶数化16
        assert_eq!(games_per_baseline(40, 3), 14); // ceil(13.3)=14
        assert_eq!(games_per_baseline(1, 1), 2);   // 最低2局（先後1局ずつ）
    }

    #[test]
    fn gradient_uses_actual_perturbation_distance() {
        // 通常次元: 分母 0.2
        let g = spsa_gradient(0.6, 0.5, &[0.6], &[0.4]);
        assert!((g[0] - 0.5).abs() < 1e-9);
        // 片側クリップ: 実際の距離 0.1 を使う
        let g = spsa_gradient(0.6, 0.5, &[0.1], &[0.0]);
        assert!((g[0] - 1.0).abs() < 1e-9);
        // 両側クリップ（同一点）: 勾配なし
        let g = spsa_gradient(0.6, 0.5, &[0.0], &[0.0]);
        assert_eq!(g[0], 0.0);
        // 符号: Δ=-1 の次元では分母が負になり勾配の向きが正しく反転する
        let g = spsa_gradient(0.6, 0.5, &[0.4], &[0.6]);
        assert!((g[0] + 0.5).abs() < 1e-9);
    }

    #[test]
    fn unit_roundtrip_preserves_params() {
        let p = EvalParams::default();
        let space = build_space_from(None, 1.0, &p).unwrap();
        let u = to_unit(&p, &space);
        let q = to_params(&u, &space);
        for (a, b) in p.to_vec().iter().zip(q.to_vec()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn tune_space_masks_and_localizes() {
        let center = EvalParams::default();
        // マスク: 指定した項目だけ active
        let space =
            build_space_from(Some("soft_decay, depth2_replace"), 1.0, &center).unwrap();
        let active: Vec<&str> = EvalParams::SPECS
            .iter()
            .zip(&space.active)
            .filter(|(_, a)| **a)
            .map(|(s, _)| s.name)
            .collect();
        assert_eq!(active, vec!["soft_decay", "depth2_replace"]);
        // 未知の名前はエラー
        assert!(build_space_from(Some("no_such_param"), 1.0, &center).is_err());
        // span: active な項目の有効範囲が現在値の近傍に縮む（元範囲内にクランプ）
        let space = build_space_from(Some("prior_weight"), 0.5, &center).unwrap();
        let i = EvalParams::SPECS
            .iter()
            .position(|s| s.name == "prior_weight")
            .unwrap();
        let c = center.to_vec()[i];
        let full = EvalParams::SPECS[i].hi - EvalParams::SPECS[i].lo;
        assert!(space.hi[i] - space.lo[i] <= full * 0.5 + 1e-9);
        assert!(space.lo[i] <= c && c <= space.hi[i]);
        assert!(space.lo[i] >= EvalParams::SPECS[i].lo);
        assert!(space.hi[i] <= EvalParams::SPECS[i].hi);
        // 非activeの項目は元範囲のまま（中心点の値が保存される）
        let j = EvalParams::SPECS.iter().position(|s| s.name == "check_bonus").unwrap();
        assert_eq!(space.lo[j], EvalParams::SPECS[j].lo);
        assert_eq!(space.hi[j], EvalParams::SPECS[j].hi);
        // roundtrip が局所空間でも成り立つ
        let u = to_unit(&center, &space);
        let q = to_params(&u, &space);
        for (a, b) in center.to_vec().iter().zip(q.to_vec()) {
            assert!((a - b).abs() < 1e-9);
        }
    }
}
