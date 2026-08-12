//! 各決定点で `deduce::opp_king_candidates` が何マスに絞れているか、真の相手玉が
//! その中にいるか（健全性）を一覧する診断ツール（粒子を回さないので一瞬で終わる）。
//!
//! usage: cargo run --release --bin king_cands -- <kifへのパス> [開始手目] [終了手目]

use tsuitate_bot::deduce::opp_king_candidates;
use tsuitate_bot::kifu::parse_kif;
use tsuitate_bot::scenario_core::{replay, side_idx};
use tsuitate_bot::board::Coord;
use tsuitate_bot::protocol::Color;

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
        println!(
            "{ply}\t{}\t{}\t{}\t{}",
            if side == Color::Sente {
                "S"
            } else {
                "G"
            },
            cands.len(),
            truth.map(|t| fmt(&t)).unwrap_or_default(),
            if sound { "ok" } else { "VIOLATION" },
        );
        println!("\t\t{}", list.join(" "));
    }
}
