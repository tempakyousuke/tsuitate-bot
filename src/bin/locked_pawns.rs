//! `deduce::immobile_opponent_pawns`（動けないと確定できる相手の歩）の
//! 鋭さの測定。棋譜を再生して、各決定点で何マス確定できるかを両者視点で数える。
//!
//! 健全性（真実を落とさない）は deduce のテストが担保しているので、ここでは
//! 「実戦でどれだけ効くか」だけを見る。
//!
//! usage: cargo run --release --bin locked_pawns -- <kifへのパス...>

use tsuitate_bot::board::make_usi_square;
use tsuitate_bot::deduce::immobile_opponent_pawns;
use tsuitate_bot::protocol::{Color, Role};
use tsuitate_bot::scenario_core::{load_scenario, replay, side_idx};

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: locked_pawns <kifへのパス...>");
        std::process::exit(2);
    }
    for path in &paths {
        let kifu = match load_scenario(path, Some(0), None, None) {
            Ok(sc) => sc.kifu,
            Err(e) => {
                eprintln!("{path}: 読み込み失敗: {e}");
                continue;
            }
        };
        println!("=== {path}（{}手） ===", kifu.plies.len());
        for me in [Color::Sente, Color::Gote] {
            let mut total = 0usize;
            let mut points = 0usize;
            let mut last: Vec<String> = vec![];
            for ply in 0..=kifu.plies.len() {
                let rep = replay(&kifu, ply);
                let locked = immobile_opponent_pawns(me, &rep.logs[side_idx(me)]);
                // 健全性の再確認（bin でも回しておく: 棋譜を足したときの網）
                for sq in &locked {
                    let truth = rep.pos.piece_at(*sq);
                    assert!(
                        truth.is_some_and(|p| p.color == me.other() && p.role == Role::Pawn),
                        "ply={ply} {me:?}視点: {} は歩でない（{truth:?}）",
                        make_usi_square(*sq)
                    );
                }
                total += locked.len();
                points += 1;
                last = locked.iter().map(|c| make_usi_square(*c)).collect();
            }
            println!(
                "  {me:?}視点: 平均 {:.1} マス確定 / 最終局面 {}件 {:?}",
                total as f64 / points as f64,
                last.len(),
                last
            );
        }
    }
}
