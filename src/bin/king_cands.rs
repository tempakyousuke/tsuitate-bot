//! 各決定点で `deduce::opp_king_candidates` が何マスに絞れているか、真の相手玉が
//! その中にいるか（健全性）を一覧する診断ツール（粒子を回さないので一瞬で終わる）。
//!
//! usage: cargo run --release --bin king_cands -- <kifへのパス> [開始手目] [終了手目]

use tsuitate_bot::deduce::opp_king_candidates;
use tsuitate_bot::kifu::parse_kif;
use tsuitate_bot::scenario_core::{replay, side_idx};
use tsuitate_bot::board::Coord;
use tsuitate_bot::protocol::Color;
use tsuitate_bot::belief_features::BeliefContext;
use tsuitate_bot::king_belief_nn::king_distribution;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("kif path");
    let from: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let to: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let text = std::fs::read_to_string(path).expect("read kif");
    let kifu = parse_kif(&text).expect("parse kif");
    let last = kifu.plies.len().min(to.saturating_sub(1));
    for ply in from..=last + 1 {
        if ply == 0 || ply > kifu.plies.len() + 1 {
            continue;
        }
        let r = replay(&kifu, ply - 1);
        let side = r.pos.turn();
        let log = &r.logs[side_idx(side)];
        let cands = opp_king_candidates(side, log);
        let truth = r.pos.king_square(side.other());
        let sound = truth.is_some_and(|t| cands.contains(&t));
        let fmt = |c: &Coord| format!("{}{}", c.file, c.rank);
        let list: Vec<String> = cands.iter().map(fmt).collect();
        // 玉位置ネット（粒子不要）の分布の質: 真のマスへの確率・top1一致・
        // 実効サポート数（1/Σp²）。**後手側は deduce が鈍いので、接近ボーナスを
        // 駆動できるだけの集中があるか**をここで見る
        let ctx = BeliefContext::from_log(side, log);
        let dist = king_distribution(&ctx, &cands);
        let p_true = truth
            .and_then(|t| dist.iter().find(|(c, _)| *c == t).map(|(_, p)| *p))
            .unwrap_or(0.0);
        let top = dist
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let top1 = top.is_some_and(|(c, _)| Some(*c) == truth);
        let eff = if dist.is_empty() {
            0.0
        } else {
            1.0 / dist.iter().map(|(_, p)| p * p).sum::<f64>()
        };
        println!(
            "{ply}\t{}\t{}\t{}\t{}\tp_true={p_true:.3}\ttop1={}\teff={eff:.1}",
            if side == Color::Sente {
                "S"
            } else {
                "G"
            },
            cands.len(),
            truth.map(|t| fmt(&t)).unwrap_or_default(),
            if sound { "ok" } else { "VIOLATION" },
            u8::from(top1),
        );
        println!("\t\t{}", list.join(" "));
    }
}
