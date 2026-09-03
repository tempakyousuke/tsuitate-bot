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
                    if c.mate_threat != 0.0 {
                        s.push_str(&format!(" 詰めろ={:+.3}", c.mate_threat));
                    }
                    if c.mate_risk != 0.0 {
                        s.push_str(&format!(" 被詰めろ=-{:.3}", c.mate_risk));
                    }
                    if c.king_holes != 0.0 {
                        s.push_str(&format!(" 玉穴=-{:.3}", c.king_holes));
                    }
                    if c.own_zone != 0.0 {
                        s.push_str(&format!(" 玉圏排除={:+.3}", c.own_zone));
                    }
                    if c.promote_bias != 0.0 {
                        s.push_str(&format!(" 成りbias={:+.3}", c.promote_bias));
                    }
                    if c.drop_bias != 0.0 {
                        s.push_str(&format!(" 打ちbias={:+.3}", c.drop_bias));
                    }
                    // 常時出す内訳（CandidateScore にあるのに表示していなかった）。
                    // gain の差がどこから来ているかは、これが無いと env の
                    // アブレーションで1項ずつ潰すしかない
                    s.push_str(&format!(
                        " NN={:+.3} 駒得={:+.3} リスク=-{:.3} 紐={:+.3} 盤上減価=-{:.3} 静的gain={:.3}",
                        c.value_nn,
                        c.capture_value,
                        c.risk,
                        c.link,
                        c.board_discount,
                        c.static_gain
                    ));
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
        // 詰めろ2項の発火率と gain の符号（w が強すぎて評価構造を壊していないかの診断）。
        // combine_score は (p_legal×gain).min(gain) なので、gain が負に振り切ると
        // p_legal による割引が効かなくなる = 序列付けの性質が変わる
        let n = ranking.len();
        let risk_n = ranking.iter().filter(|c| c.mate_risk > 0.0).count();
        let threat_n = ranking.iter().filter(|c| c.mate_threat > 0.0).count();
        let neg_n = ranking.iter().filter(|c| c.gain < 0.0).count();
        if risk_n > 0 || threat_n > 0 {
            let risk_max = ranking.iter().map(|c| c.mate_risk).fold(0.0, f64::max);
            println!(
                "   [発火] 被詰めろ {risk_n}/{n}（最大 -{risk_max:.3}）/ 詰めろ {threat_n}/{n} / gain<0 {neg_n}/{n}"
            );
        }
    }
    // 評価項・前提条件の発火率（TSUITATE_DBG_HITS=1 のときだけ）。
    // 「この局面で厳密粒子が生きていたか」= expected が効いていたかが分かる
    if let Some(table) = tsuitate_bot::hits::dump() {
        println!("{table}");
    }
}
