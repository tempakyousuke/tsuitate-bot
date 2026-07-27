//! 信念ネット（NN段階②）のゲート1用: **現行の粒子フィルタが持っている
//! マスごとの占有信念**を、学習データと同じ決定点・同じ特徴量つきで書き出す。
//!
//! 目的は「ネットが現行粒子より当たるか」を同じ土俵で測ること
//! （particle-likelihood-nn の教訓: オフライン較正だけで勝ったつもりに
//! ならない。ここで負けるなら段階②はそもそも入口で止める）。
//!
//! 出力は `export_belief_data` と同じ列に `p_all` / `p_strict` / `n_strict` を
//! 足したもの。1決定点ごとに推定器を実戦と同じ順序で逐次 update するため
//! **重い**（対局1局ぶんの思考に相当）ので、少数の記録に対して使う。
//!
//! 使い方:
//!   cargo run --release --bin belief_probe -- [--stride N] [--budget MS] [--seed S] \
//!       [--max-games N] <records/*.jsonl...> > probe.csv
//!
//! 集計（対数損失・Brier・較正）は stderr のサマリーに出る。

use tsuitate_bot::belief_features::{
    BELIEF_FEATURE_NAMES, BeliefContext, all_squares, label, role_class, sq_index,
};
use tsuitate_bot::protocol::Color;
use tsuitate_bot::scenario_core::{IncrementalEstimator, weighted_unique_particles};
use tsuitate_bot::truth_replay::{for_each_decision, load_end};

const REFERENCE_BUDGET_MS: f64 = 900.0;

fn fmt(x: f64) -> String {
    if x == 0.0 {
        "0".to_string()
    } else if x == 1.0 {
        "1".to_string()
    } else {
        format!("{x:.4}")
    }
}

#[derive(Default)]
struct Score {
    n: u64,
    occ: u64,
    logloss: f64,
    brier: f64,
}

impl Score {
    fn add(&mut self, p: f64, occ: bool) {
        let p = p.clamp(1e-4, 1.0 - 1e-4);
        self.n += 1;
        self.occ += u64::from(occ);
        self.logloss -= if occ { p.ln() } else { (1.0 - p).ln() };
        let d = p - f64::from(occ);
        self.brier += d * d;
    }
    fn report(&self, name: &str) {
        if self.n == 0 {
            eprintln!("{name}: データなし");
            return;
        }
        let n = self.n as f64;
        eprintln!(
            "{name}: n={} 占有率={:.4} 対数損失={:.4} Brier={:.4}",
            self.n,
            self.occ as f64 / n,
            self.logloss / n,
            self.brier / n
        );
    }
}

fn arg_num<T: std::str::FromStr>(args: &mut Vec<String>, name: &str, default: T) -> T {
    if let Some(i) = args.iter().position(|a| a == name) {
        let v = args.get(i + 1).and_then(|s| s.parse().ok());
        args.drain(i..=(i + 1).min(args.len() - 1));
        return v.unwrap_or(default);
    }
    default
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let stride = arg_num(&mut args, "--stride", 3usize).max(1);
    let budget_ms = arg_num(&mut args, "--budget", 2000u64);
    let seed = arg_num(&mut args, "--seed", 20260727u64);
    let max_games = arg_num(&mut args, "--max-games", usize::MAX);
    if args.is_empty() {
        eprintln!(
            "使い方: belief_probe [--stride N] [--budget MS] [--seed S] [--max-games N] <records/*.jsonl...>"
        );
        std::process::exit(1);
    }
    let scale = (budget_ms as f64 / REFERENCE_BUDGET_MS).clamp(0.25, 8.0);

    println!(
        "game_id,decision_id,side,sq,{},occ,role,promoted,p_all,p_strict,n_strict",
        BELIEF_FEATURE_NAMES.join(",")
    );

    // 基準（対照）: その決定点の一様事前 = 相手の盤上駒数 ÷ 未知マス数。
    // 粒子の対数損失がこれを下回っていなければ、そもそも信念になっていない
    let mut prior = Score::default();
    let mut all = Score::default();
    let mut strict = Score::default();
    let mut blind_all = Score::default();
    let mut decisions = 0u64;
    let mut blind_decisions = 0u64;
    let mut games = 0u64;

    for path in args.iter().take(max_games) {
        let Some(end) = load_end(path) else {
            continue;
        };
        let game_id = games;
        // 先後それぞれの推定器を逐次 update する（実戦と同じ順序）
        let mut ests = [
            IncrementalEstimator::new(Color::Sente, seed, scale),
            IncrementalEstimator::new(Color::Gote, seed ^ 0x9e37, scale),
        ];
        let mut buf: Vec<String> = vec![];
        let ok = for_each_decision(&end, |pos, side, log, decision_id| {
            let idx = usize::from(side == Color::Gote);
            // 間引いても推定器は毎手番食わせる（実戦と同じ履歴になるように）
            ests[idx].feed(log);
            if !(decision_id as usize).is_multiple_of(stride) {
                return;
            }
            let est = ests[idx].est();
            let parts = weighted_unique_particles(est);
            let total: f64 = parts.iter().map(|&(_, w, _)| w).sum();
            let total_strict: f64 = parts.iter().filter(|&&(_, _, s)| s).map(|&(_, w, _)| w).sum();
            let n_strict = parts.iter().filter(|&&(_, _, s)| s).count();
            let ctx = BeliefContext::from_log(side, log);
            let opp = side.other();
            let base_prior = ctx.features(all_squares().next().unwrap())[0];

            decisions += 1;
            if n_strict == 0 {
                blind_decisions += 1;
            }
            for sq in all_squares() {
                if ctx.is_mine(sq) {
                    continue;
                }
                let (occ, role) = label(pos, side, sq);
                let (base, promoted) = role_class(role);
                let mut mass = 0.0;
                let mut mass_strict = 0.0;
                for &(p, w, s) in &parts {
                    if p.piece_at(sq).is_some_and(|q| q.color == opp) {
                        mass += w;
                        if s {
                            mass_strict += w;
                        }
                    }
                }
                let p_all = if total > 0.0 { mass / total } else { base_prior };
                let p_strict = if total_strict > 0.0 {
                    mass_strict / total_strict
                } else {
                    base_prior
                };
                prior.add(base_prior, occ);
                all.add(p_all, occ);
                strict.add(p_strict, occ);
                if n_strict == 0 {
                    blind_all.add(p_all, occ);
                }
                let feats: Vec<String> = ctx.features(sq).iter().map(|&x| fmt(x)).collect();
                buf.push(format!(
                    "{game_id},{decision_id},{},{},{},{},{},{},{},{},{}",
                    u8::from(side == Color::Gote),
                    sq_index(sq),
                    feats.join(","),
                    u8::from(occ),
                    base,
                    promoted,
                    fmt(p_all),
                    fmt(p_strict),
                    n_strict
                ));
            }
        });
        if ok {
            for line in &buf {
                println!("{line}");
            }
            games += 1;
            eprintln!("  {games}局目 完了: {path}");
        } else {
            eprintln!("  棋譜を再生できませんでした: {path}。この局はスキップ");
        }
    }

    eprintln!("--- 粒子の占有信念（{games}局 / 決定点{decisions}）---");
    prior.report("一様事前");
    all.report("粒子（全）");
    strict.report("粒子（厳密のみ）");
    eprintln!(
        "厳密粒子ゼロの決定点: {blind_decisions} / {decisions} ({:.1}%)",
        100.0 * blind_decisions as f64 / decisions.max(1) as f64
    );
    blind_all.report("粒子（全）／厳密ゼロの決定点だけ");
}
