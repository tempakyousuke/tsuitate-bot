//! 相手玉の位置ビリーフの質を決定点ごとに測る診断（変異救済の効果測定用）。
//!
//! belief_probe のマスごと占有信念は全駒種で薄まるため、攻めの律速と特定済みの
//! **玉位置**（docs/yaneuraou-lessons.md 1-2）だけを取り出して測る:
//! 各決定点で「真の相手玉マス」への信念質量（taint 込み = p_all / 厳密のみ =
//! p_strict）と top-1 一致率を集計する。厳密粒子ゼロの決定点（ブラインド）
//! だけの集計も出す — そこで玉位置を供給しているのは taint 粒子なので、
//! 変異救済（TSUITATE_MUT_RESCUE）の効果はここに出るはず。
//!
//! belief_probe と同じく**重い**（1局 ≒ 対局1局ぶんの思考）。対照も複数 seed
//! 取ること（probe-control-needs-multiple-seeds）。
//!
//! 使い方:
//!   cargo run --release --bin king_probe -- [--stride N] [--budget MS] [--seed S] \
//!       [--max-games N] <records/*.jsonl...>

use tsuitate_bot::belief_features::BeliefContext;
use tsuitate_bot::deduce::opp_king_candidates;
use tsuitate_bot::king_belief_nn::king_distribution;
use tsuitate_bot::protocol::Color;
use tsuitate_bot::scenario_core::{IncrementalEstimator, weighted_unique_particles};
use tsuitate_bot::truth_replay::{for_each_decision, load_end};

const REFERENCE_BUDGET_MS: f64 = 900.0;

#[derive(Default)]
struct Score {
    n: u64,
    logloss: f64,
    hit: u64,
}

impl Score {
    fn add(&mut self, p_true: f64, top1: bool) {
        self.n += 1;
        self.logloss -= p_true.clamp(1e-4, 1.0).ln();
        self.hit += u64::from(top1);
    }
    fn report(&self, name: &str) {
        if self.n == 0 {
            eprintln!("{name}: データなし");
            return;
        }
        let n = self.n as f64;
        eprintln!(
            "{name}: n={} 対数損失={:.4} top1一致={:.1}%",
            self.n,
            self.logloss / n,
            100.0 * self.hit as f64 / n
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
            "使い方: king_probe [--stride N] [--budget MS] [--seed S] [--max-games N] <records/*.jsonl...>"
        );
        std::process::exit(1);
    }
    let scale = (budget_ms as f64 / REFERENCE_BUDGET_MS).clamp(0.25, 8.0);

    let mut all = Score::default();
    let mut strict_only = Score::default();
    let mut blind_all = Score::default();
    // 玉位置ネット（king_belief_nn）。粒子と**同一の決定点**で測る（ゲート1）
    let mut net_all = Score::default();
    let mut net_blind = Score::default();
    // ブレンド p=(1−λ)·p_taint+λ·p_net の λ 選定用（供給先 blind_king_dist と
    // 同じ量になるのはブラインド決定点だけなので、そこだけ測る）
    const LAMBDAS: [f64; 3] = [0.25, 0.5, 0.75];
    let mut blend_blind: [Score; 3] = Default::default();
    let mut decisions = 0u64;
    let mut blind_decisions = 0u64;
    let mut games = 0u64;

    for path in args.iter().take(max_games) {
        let Some(end) = load_end(path) else {
            continue;
        };
        let mut ests = [
            IncrementalEstimator::new(Color::Sente, seed, scale),
            IncrementalEstimator::new(Color::Gote, seed ^ 0x9e37, scale),
        ];
        let ok = for_each_decision(&end, |pos, side, log, decision_id| {
            let idx = usize::from(side == Color::Gote);
            ests[idx].feed(log);
            if !(decision_id as usize).is_multiple_of(stride) {
                return;
            }
            let Some(true_king) = pos.king_square(side.other()) else {
                return;
            };
            let parts = weighted_unique_particles(ests[idx].est());
            let mut mass_true = 0.0f64;
            let mut mass_total = 0.0f64;
            let mut strict_true = 0.0f64;
            let mut strict_total = 0.0f64;
            let mut best = (0.0f64, None::<tsuitate_bot::board::Coord>);
            let mut per_sq: std::collections::HashMap<tsuitate_bot::board::Coord, f64> =
                std::collections::HashMap::new();
            let mut n_strict = 0usize;
            for (pp, w, s) in &parts {
                let Some(sq) = pp.king_square(side.other()) else {
                    continue;
                };
                mass_total += w;
                *per_sq.entry(sq).or_insert(0.0) += w;
                if sq == true_king {
                    mass_true += w;
                }
                if *s {
                    n_strict += 1;
                    strict_total += w;
                    if sq == true_king {
                        strict_true += w;
                    }
                }
            }
            for (&sq, &m) in &per_sq {
                if m > best.0 {
                    best = (m, Some(sq));
                }
            }
            if mass_total <= 0.0 {
                return;
            }
            decisions += 1;
            let p_all = mass_true / mass_total;
            let top1 = best.1 == Some(true_king);
            all.add(p_all, top1);
            if strict_total > 0.0 {
                strict_only.add(strict_true / strict_total, top1);
            }
            // 玉位置ネット（同一決定点）
            let ctx = BeliefContext::from_log(side, log);
            let cands = opp_king_candidates(side, log);
            let net = king_distribution(&ctx, &cands);
            let p_net = net
                .iter()
                .find(|(c, _)| *c == true_king)
                .map_or(0.0, |(_, p)| *p);
            let net_top1 = net
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .is_some_and(|(c, _)| *c == true_king);
            net_all.add(p_net, net_top1);
            if n_strict == 0 {
                blind_decisions += 1;
                blind_all.add(p_all, top1);
                net_blind.add(p_net, net_top1);
                // ブレンド（粒子分布は per_sq を正規化して使う）
                for (li, &lambda) in LAMBDAS.iter().enumerate() {
                    let mut mix: std::collections::HashMap<tsuitate_bot::board::Coord, f64> =
                        std::collections::HashMap::new();
                    for (&sq, &m) in &per_sq {
                        *mix.entry(sq).or_insert(0.0) += (1.0 - lambda) * m / mass_total;
                    }
                    for (sq, p) in &net {
                        *mix.entry(*sq).or_insert(0.0) += lambda * p;
                    }
                    let p_mix = mix.get(&true_king).copied().unwrap_or(0.0);
                    let mix_top1 = mix
                        .iter()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .is_some_and(|(sq, _)| *sq == true_king);
                    blend_blind[li].add(p_mix, mix_top1);
                }
            }
        });
        if !ok {
            eprintln!("  リプレイ不整合: {path}");
        }
        games += 1;
        eprintln!("  {games}局目 完了: {path}");
    }

    eprintln!("--- 相手玉の位置ビリーフ（{games}局 / 決定点{decisions}）---");
    eprintln!(
        "厳密粒子ゼロの決定点: {blind_decisions} / {decisions} ({:.1}%)",
        100.0 * blind_decisions as f64 / decisions.max(1) as f64
    );
    all.report("全粒子（taint込み）");
    strict_only.report("厳密のみ（厳密が生きている決定点だけ）");
    blind_all.report("全粒子／厳密ゼロの決定点だけ");
    net_all.report("玉位置ネット（全決定点）");
    net_blind.report("玉位置ネット／厳密ゼロの決定点だけ");
    for (li, &lambda) in LAMBDAS.iter().enumerate() {
        blend_blind[li].report(&format!("ブレンド λ={lambda}／厳密ゼロの決定点だけ"));
    }
}
