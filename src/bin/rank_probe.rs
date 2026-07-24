//! 指定シナリオ・指定plyの候補手ランキング内訳をCLIへ出す（scenario-guiのランキング相当）。
//! 上位8件と focus 指定の手（順位も表示）を score/gain/p_legal/foul_cost/adjust に分解する。
//! 評価項の変更が特定局面の序列にどう効くかを、GUIを立てずに複数シードで見るための診断ツール。
//!
//! usage: cargo run --release --bin rank_probe -- <kifパスまたは名前> <ply> [シード数=5] [focus手,focus手...]
//!   例: TSUITATE_CAPTURE_BET_VAR_W=2 TSUITATE_THINK_BUDGET_MS=5000 \
//!       cargo run --release --bin rank_probe -- scenarios/tokin-bet.kif 15 5 "P*8h,8g8h"
use tsuitate_bot::scenario_core::{load_scenario, ranking_one, replay};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let spec = args.get(1).expect("usage: rank_probe <kif> <ply> [seeds] [focus,focus...]");
    let ply: usize = args.get(2).expect("ply").parse().expect("ply");
    let seeds: u64 = args.get(3).map(|s| s.parse().expect("seeds")).unwrap_or(5);
    let focus: Vec<String> = args
        .get(4)
        .map(|s| s.split(',').map(str::to_string).collect())
        .unwrap_or_default();
    let sc = load_scenario(spec, Some(ply), None, None).expect("load");
    let rep = replay(&sc.kifu, sc.ply);
    println!("手番: {:?} / {}手目を考えさせる", rep.pos.turn(), ply + 1);
    for seed in 0..seeds {
        let Some((chosen, ranking)) = ranking_one(&rep, seed, "estimator") else {
            println!("seed={seed}: ランキングなし");
            continue;
        };
        println!("== seed={seed} 選択={chosen} ==");
        for (i, c) in ranking.iter().enumerate() {
            let is_focus = focus.iter().any(|f| f == &c.usi);
            if i < 8 || is_focus {
                let extras = {
                    let mut s = String::new();
                    if c.checker_removal != 0.0 {
                        s.push_str(&format!(" 除去EV={:+.3}", c.checker_removal));
                    }
                    if c.capture_bet_penalty != 0.0 {
                        s.push_str(&format!(" 賭けpen=-{:.3}", c.capture_bet_penalty));
                    }
                    s
                };
                println!(
                    "{:>3} {}{:<6} score={:8.3} gain={:8.3} p_legal={:.3} foul_cost={:7.3} adjust={:7.3} depth2={}{extras}",
                    i + 1,
                    if is_focus { "*" } else { " " },
                    c.usi,
                    c.score,
                    c.gain,
                    c.p_legal,
                    c.foul_cost,
                    c.adjust,
                    c.depth2
                );
            }
        }
    }
}
