//! 実対局の局面再現実験。
//!
//! Shogi Quest の実棋譜（真実の手順＋反則試行）をリプレイして特定局面を再現し、
//! estimator の選択・信念分布・終盤遂行を調べる。シナリオ:
//! - keima: 対 likealorigstorn 戦（2026-07）の29手目 ▲８五桂（王手）。
//!   同歩(8d8e)で桂を取り返せるか。人間は反則1回（6d6e）ののち同歩だった
//! - kakunari: 対 dkuhouho8 戦（2026-07）の70手目 △５七角成(7i5g+)。
//!   馬捨てで５七に金を釣り出し、５八飛の成り込み（5h5g+）を通す決め手。
//!   指せるか、その後勝ち切れるか（実戦は76手目まで指して先手の反則負け）
//!
//! 反則試行も MyFoul / OpponentFoul として両者の観測ログに再現する
//! （反則回数は foul_limit の残量と推定器の制約の両方に効く）。
//!
//! usage:
//!   cargo run --release --bin scenario -- <シナリオ> [試行数=20] [戦略=estimator]
//!   cargo run --release --bin scenario -- <シナリオ> diag [推定器数=10]
//!   cargo run --release --bin scenario -- <シナリオ> continue [対局数=10] [手番側戦略] [相手戦略]

use std::collections::{HashMap, HashSet};

use tsuitate_bot::board::{make_usi_square, parse_usi_square};
use tsuitate_bot::estimator::Estimator;
use tsuitate_bot::observation::{Observation, ObservationLog};
use tsuitate_bot::protocol::{ClockState, Color, FoulCounts, GameStatus, PlayerView};
use tsuitate_bot::shogi::{Outcome, Position, ShogiMove, parse_usi, unpromote_role};
use tsuitate_bot::strategy;

/// 1手ぶんの真実: 受理された手と、その直前の同じ手番側の反則試行
struct Ply {
    usi: &'static str,
    fouls: &'static [&'static str],
}

const fn p(usi: &'static str) -> Ply {
    Ply { usi, fouls: &[] }
}

const fn pf(usi: &'static str, fouls: &'static [&'static str]) -> Ply {
    Ply { usi, fouls }
}

struct Scenario {
    name: &'static str,
    desc: &'static str,
    /// 注目している手（一致したら出力に印をつける）
    target: &'static str,
    plies: &'static [Ply],
    /// diag で相手駒の利き枚数分布を測るマス（着地の安全性検証用）
    diag_squares: &'static [&'static str],
}

/// 対 likealorigstorn 戦の1〜29手目。29手目 7g8e が桂跳ね王手
const KEIMA: &[Ply] = &[
    p("7g7f"), p("3a3b"), p("5g5f"), p("2b3a"), p("5f5e"), p("5a6b"),
    p("2h5h"), p("5c5d"), p("5i4h"), p("7c7d"), p("7i6h"), p("8c8d"),
    p("6h5g"), p("6b7c"), p("5g5f"), p("6c6d"), p("4h3h"), p("9c9d"),
    p("6i6h"), p("9d9e"), p("6h5g"), p("9e9f"), p("4g4f"), p("9f9g+"),
    p("8h6f"), p("P*9h"), p("8i7g"), p("9h9i+"), p("7g8e"),
];

/// 対 dkuhouho8 戦の1〜69手目。70手目 7i5g+（５七角成）が注目手
const KAKUNARI: &[Ply] = &[
    p("7g7f"), p("3a3b"), p("6i7h"), p("1c1d"), p("7h7g"), p("2b1c"),
    p("4i5h"), p("5a4b"), p("6g6f"), p("7c7d"), p("5h6g"), p("8a7c"),
    p("5g5f"), p("8c8d"), p("8g8f"), p("8d8e"), p("4g4f"), p("8e8f"),
    p("7i7h"), p("8f8g+"), p("4f4e"),
    pf("8g8h", &["P*8h"]),
    p("4e4d"), p("8h8i"), p("4d4c+"), p("3b4c"), p("P*8c"),
    pf("8b8c", &["8b8e", "P*8c"]),
    p("7h8g"), p("8i7i"), p("2h8h"),
    pf("7i6i", &["8c8i+"]),
    p("5i5h"), p("P*8f"), p("8h8i"), p("8f8g+"), p("7g8g"), p("8c8g+"),
    p("8i8g"), p("P*8e"),
    pf("8g8i", &["8g8c+"]),
    p("4c3d"), p("8i6i"), p("N*5g"), p("6i8i"), p("B*6i"), p("8i6i"),
    p("5g6i+"), p("5h6i"), p("1c7i"),
    pf("6i5h", &["6i6h"]),
    p("R*6i"), p("P*4f"),
    pf("P*4h", &["G*5h"]),
    p("R*4e"), p("4b3b"), p("4e8e"), p("7c8e"), p("N*5d"), p("8e7g+"),
    p("5h4g"), p("4h4i+"), p("B*2b"), p("6i5i+"), p("2b1a+"), p("R*5h"),
    p("L*4c"), p("G*4e"), p("4c4a+"),
];

const SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "keima",
        desc: "29手目 ▲８五桂（王手）を受けた後手番。真の解消手は 8d8e（同歩・桂得）ほか玉逃げ",
        target: "8d8e",
        plies: KEIMA,
        diag_squares: &[],
    },
    Scenario {
        name: "kakunari",
        desc: "70手目の後手番。注目手は 7i5g+（５七角成の馬捨て → 71同金に 5h5g+ で決まり）",
        target: "7i5g+",
        plies: KAKUNARI,
        diag_squares: &["5g", "4h"],
    },
];

/// リプレイ結果: 真実の局面と両者の観測ログ・反則数。[0]=先手, [1]=後手
struct Replayed {
    pos: Position,
    logs: [ObservationLog; 2],
    fouls: [u32; 2],
    plies: u32,
}

fn side_idx(c: Color) -> usize {
    if c == Color::Sente { 0 } else { 1 }
}

/// 棋譜（反則試行込み）を裁定つきでリプレイし、selfplay.rs と同じ規約で
/// 両者の観測ログを構築する
fn replay(plies: &[Ply]) -> Replayed {
    let mut pos = Position::initial();
    let mut logs = [ObservationLog::default(), ObservationLog::default()];
    let mut fouls = [0u32; 2];
    for ply in plies {
        let side = pos.turn();
        for f in ply.fouls {
            let mv = parse_usi(f).expect("反則試行のUSI解析失敗");
            assert!(!pos.is_legal(&mv), "反則のはずの手が合法: {f}");
            fouls[side_idx(side)] += 1;
            logs[side_idx(side)].record(Observation::MyFoul {
                move_number: pos.move_number(),
                usi: (*f).to_string(),
            });
            logs[side_idx(side.other())].record(Observation::OpponentFoul {
                count: fouls[side_idx(side)],
            });
        }
        let mv = parse_usi(ply.usi).expect("USI解析失敗");
        assert!(pos.is_legal(&mv), "棋譜の手が非合法: {}", ply.usi);
        let captured = pos.play_unchecked(&mv);
        let move_number = pos.move_number();
        let captured_sq = captured.map(|_| match mv {
            ShogiMove::Board { to, .. } => make_usi_square(to),
            ShogiMove::Drop { .. } => unreachable!("打ちでは駒を取れない"),
        });
        logs[side_idx(side)].record(Observation::MyMove {
            move_number,
            usi: ply.usi.to_string(),
            captured: captured.map(unpromote_role),
        });
        logs[side_idx(side.other())].record(Observation::OpponentMoved {
            move_number,
            captured_my_piece_at: captured_sq,
        });
        if pos.in_check(pos.turn()) {
            let in_check = pos.turn();
            for log in logs.iter_mut() {
                log.record(Observation::Check { in_check });
            }
        }
    }
    let plies = plies.len() as u32;
    Replayed { pos, logs, fouls, plies }
}

/// 実対局と同じ「自分の手番ごとの逐次 update」を再現して戦略の推定器を温める。
/// 一括 update だとリプレイ予算が1回分しか与えられず、長い棋譜では粒子が
/// 完全枯渇する（kakunari の69手を一括で食わせるとユニーク粒子0になる）
fn prewarm_strategy(
    strat: &mut Box<dyn tsuitate_bot::strategy::Strategy>,
    view: &PlayerView,
    full: &ObservationLog,
) {
    let mut running = ObservationLog::default();
    for e in full.events() {
        if matches!(e, Observation::MyMove { .. } | Observation::MyFoul { .. }) {
            strat.prewarm(view, &running);
        }
        running.record(e.clone());
    }
}

fn clone_log(log: &ObservationLog) -> ObservationLog {
    let mut out = ObservationLog::default();
    for e in log.events() {
        out.record(e.clone());
    }
    out
}

fn make_view(pos: &Position, color: Color, fouls: &[u32; 2]) -> PlayerView {
    PlayerView {
        game_id: "scenario".into(),
        your_color: color,
        your_pieces: pos.pieces_of(color),
        your_hand: pos.hand_map(color),
        turn: pos.turn(),
        move_number: pos.move_number(),
        clocks: ClockState {
            sente_ms: 900_000,
            gote_ms: 900_000,
            running: Some(pos.turn()),
            server_time: 0,
        },
        fouls: FoulCounts {
            you: fouls[side_idx(color)],
            opponent: fouls[side_idx(color.other())],
        },
        you_in_check: pos.in_check(color),
        opponent_in_check: pos.in_check(color.other()),
        status: GameStatus::Playing,
    }
}

/// 手番側の一手の選択を試行する。反則は観測として与えて指し直させる
/// （実対局と同じ）。受理された手と反則列を返す
fn choice_trials(sc: &Scenario, rep: &Replayed, trials: u64, name: &str) {
    let side = rep.pos.turn();
    println!("局面: {}", sc.desc);
    println!(
        "手番: {:?} / ここまでの反則 先手{} 後手{} / 戦略: {name} / 試行 {trials} 回",
        side, rep.fouls[0], rep.fouls[1]
    );
    println!();

    let mut final_tally: HashMap<String, u32> = HashMap::new();
    let mut total_fouls = 0u32;
    for seed in 0..trials {
        let mut strat = strategy::make_seeded(name, seed).expect("未知の戦略名");
        let mut log = clone_log(&rep.logs[side_idx(side)]);
        prewarm_strategy(&mut strat, &make_view(&rep.pos, side, &rep.fouls), &log);
        let mut foul_tried: HashSet<String> = HashSet::new();
        let mut fouls = rep.fouls;
        let mut foul_seq: Vec<String> = vec![];
        let accepted = loop {
            let view = make_view(&rep.pos, side, &fouls);
            let Some(usi) = strat.choose(&view, &log, &foul_tried) else {
                break "resign".to_string();
            };
            let legal = parse_usi(&usi).is_some_and(|mv| rep.pos.is_legal(&mv));
            if legal {
                break usi;
            }
            fouls[side_idx(side)] += 1;
            log.record(Observation::MyFoul {
                move_number: rep.pos.move_number(),
                usi: usi.clone(),
            });
            foul_tried.insert(usi.clone());
            foul_seq.push(usi);
            if fouls[side_idx(side)] >= 10 {
                break "foul_limit".to_string();
            }
        };
        let note = if accepted == sc.target { "（注目手）" } else { "" };
        let foul_note = if foul_seq.is_empty() {
            String::new()
        } else {
            format!(" 反則: {}", foul_seq.join(", "))
        };
        println!("seed {seed:2}: {accepted}{note}{foul_note}");
        *final_tally.entry(accepted).or_insert(0) += 1;
        total_fouls += foul_seq.len() as u32;
    }

    println!();
    let mut sorted: Vec<_> = final_tally.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    println!("受理された手の内訳:");
    for (usi, n) in sorted {
        let mark = if usi == sc.target { " ← 注目手" } else { "" };
        println!("  {usi}: {n}/{trials}{mark}");
    }
    println!("追加の反則の総数: {total_fouls}");
}

/// 粒子集合の診断: 王手駒の分布・相手玉位置の分布・注目マスへの相手利き枚数。
/// 粒子は複製で偏るので指紋でユニーク化して数える（strategy.rs の評価と同じ発想）。
/// 玉位置・利き枚数は厳密整合粒子（penalty=0）だけで集計する
fn diagnose_particles(sc: &Scenario, rep: &Replayed, n_estimators: u64) {
    let side = rep.pos.turn();
    // 既定の思考予算 2000ms 相当（SearchBudget::from_ms は非公開なので直書き）。
    // SCENARIO_SCALE_MULT で粒子数・再生成予算を実運用の何倍にも増やせる
    // （枯渇が予算不足か構造的かの切り分け用）
    let mult: f64 = std::env::var("SCENARIO_SCALE_MULT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let scale = 2000.0 / 900.0 * mult;
    let king_sq = rep.pos.king_square(side).expect("手番側の玉");
    let log = &rep.logs[side_idx(side)];
    let diag_sqs: Vec<_> = sc
        .diag_squares
        .iter()
        .map(|s| (*s, parse_usi_square(s).expect("diag_squares のマス解析失敗")))
        .collect();

    let mut checker_tally: HashMap<String, u32> = HashMap::new();
    let mut opp_king_tally: HashMap<String, u32> = HashMap::new();
    // マスごとの相手利き枚数（0,1,2,3+）の度数
    let mut cover_tally: Vec<[u32; 4]> = vec![[0; 4]; diag_sqs.len()];
    let mut total_unique = 0u32;
    let mut strict_unique = 0u32;
    for seed in 0..n_estimators {
        let mut est = Estimator::with_seed_and_scale(side, seed, scale);
        // 実対局と同じ逐次 update（prewarm_strategy と同じ理由）
        let mut running = ObservationLog::default();
        let mut turn_no = 0;
        for e in log.events() {
            if matches!(e, Observation::MyMove { .. } | Observation::MyFoul { .. }) {
                est.update(&running);
                turn_no += 1;
                if seed == 0 {
                    eprintln!(
                        "  [seed0] 手番{turn_no}: 粒子 {} (healthy={})",
                        est.particles().len(),
                        est.healthy()
                    );
                }
            }
            running.record(e.clone());
        }
        est.update(&running);
        if seed == 0 {
            eprintln!(
                "  [seed0] 最終: 粒子 {} (healthy={})",
                est.particles().len(),
                est.healthy()
            );
        }
        let mut seen: HashSet<u64> = HashSet::new();
        for (pp, &penalty) in est.particles().iter().zip(est.penalties()) {
            if !seen.insert(pp.fingerprint()) {
                continue;
            }
            total_unique += 1;
            if rep.pos.in_check(side) {
                let checkers: Vec<String> = pp
                    .pieces()
                    .filter(|(from, pc)| pc.color == side.other() && pp.attacks(*from, king_sq))
                    .map(|(from, pc)| format!("{:?}@{}", pc.role, make_usi_square(from)))
                    .collect();
                let key = if checkers.is_empty() {
                    "（王手なし）".to_string()
                } else {
                    checkers.join("+")
                };
                *checker_tally.entry(key).or_insert(0) += 1;
            }
            if penalty > 0 {
                continue;
            }
            strict_unique += 1;
            if let Some(sq) = pp.king_square(side.other()) {
                *opp_king_tally
                    .entry(make_usi_square(sq))
                    .or_insert(0) += 1;
            }
            for (i, (_, sq)) in diag_sqs.iter().enumerate() {
                let n = pp
                    .pieces()
                    .filter(|(from, pc)| {
                        pc.color == side.other()
                            && pc.role != tsuitate_bot::protocol::Role::King
                            && pp.attacks(*from, *sq)
                    })
                    .count();
                cover_tally[i][n.min(3)] += 1;
            }
        }
    }

    println!(
        "粒子診断: 推定器 {n_estimators} 個ぶんのユニーク粒子 {total_unique} 個（うち厳密整合 {strict_unique}。玉位置・利きは厳密のみで集計）"
    );
    if rep.pos.in_check(side) {
        println!();
        println!("王手駒の分布（粒子内で手番側の玉に利いている相手駒）:");
        let mut sorted: Vec<_> = checker_tally.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        for (key, n) in sorted {
            println!(
                "  {key}: {n} ({:.1}%)",
                100.0 * n as f64 / total_unique as f64
            );
        }
    }
    println!();
    println!("相手玉の位置分布（上位）:");
    let mut sorted: Vec<_> = opp_king_tally.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    for (sq, n) in sorted.iter().take(8) {
        println!(
            "  {sq}: {n} ({:.1}%)",
            100.0 * *n as f64 / strict_unique as f64
        );
    }
    for (i, (name, _)) in diag_sqs.iter().enumerate() {
        let t = &cover_tally[i];
        println!();
        println!(
            "{name} への相手利き枚数（玉を除く）: 0枚 {:.1}% / 1枚 {:.1}% / 2枚 {:.1}% / 3枚以上 {:.1}%",
            100.0 * t[0] as f64 / strict_unique as f64,
            100.0 * t[1] as f64 / strict_unique as f64,
            100.0 * t[2] as f64 / strict_unique as f64,
            100.0 * t[3] as f64 / strict_unique as f64,
        );
    }
}

/// 局面から bot 同士で終局まで指し継ぐ（selfplay.rs の裁定を簡略移植。時計なし）。
/// 反則数はリプレイ時点から引き継ぐ（foul_limit 10 は累計）
fn continue_games(sc: &Scenario, rep: &Replayed, games: u64, name_me: &str, name_opp: &str) {
    let me = rep.pos.turn();
    println!("局面: {}", sc.desc);
    println!(
        "手番側 {:?} = {name_me} / 相手側 = {name_opp} / ここまでの反則 先手{} 後手{} / {games} 局",
        me, rep.fouls[0], rep.fouls[1]
    );
    println!();

    let mut wins_me = 0u32;
    let mut reasons: HashMap<String, u32> = HashMap::new();
    let mut first_moves: HashMap<String, u32> = HashMap::new();
    let mut win_plies: Vec<u32> = vec![];
    for seed in 0..games {
        let mut strats = [
            strategy::make_seeded(
                if me == Color::Sente { name_me } else { name_opp },
                seed ^ 0x5E17E_u64,
            )
            .expect("未知の戦略名"),
            strategy::make_seeded(
                if me == Color::Gote { name_me } else { name_opp },
                seed ^ 0x607E_u64,
            )
            .expect("未知の戦略名"),
        ];
        let mut pos = rep.pos.clone();
        let mut logs = [clone_log(&rep.logs[0]), clone_log(&rep.logs[1])];
        for (i, strat) in strats.iter_mut().enumerate() {
            let color = if i == 0 { Color::Sente } else { Color::Gote };
            prewarm_strategy(strat, &make_view(&rep.pos, color, &rep.fouls), &logs[i]);
        }
        let mut fouls = rep.fouls;
        let mut foul_tried: [HashSet<String>; 2] = [HashSet::new(), HashSet::new()];
        let mut plies = rep.plies;
        let mut first_move: Option<String> = None;

        let (winner, reason): (Option<Color>, String) = loop {
            if plies >= 200 {
                break (None, "max_plies".into());
            }
            let side = pos.turn();
            let i = side_idx(side);
            let view = make_view(&pos, side, &fouls);
            let Some(usi) = strats[i].choose(&view, &logs[i], &foul_tried[i]) else {
                break (Some(side.other()), "resign".into());
            };
            let legal = parse_usi(&usi).is_some_and(|mv| pos.is_legal(&mv));
            if !legal {
                fouls[i] += 1;
                foul_tried[i].insert(usi.clone());
                logs[i].record(Observation::MyFoul {
                    move_number: pos.move_number(),
                    usi,
                });
                logs[1 - i].record(Observation::OpponentFoul { count: fouls[i] });
                if fouls[i] >= 10 {
                    break (Some(side.other()), "foul_limit".into());
                }
                continue;
            }
            let mv = parse_usi(&usi).unwrap();
            let captured = pos.play_unchecked(&mv);
            plies += 1;
            foul_tried[i].clear();
            if side == me && first_move.is_none() {
                first_move = Some(usi.clone());
            }
            let move_number = pos.move_number();
            let captured_sq = captured.map(|_| match mv {
                ShogiMove::Board { to, .. } => make_usi_square(to),
                ShogiMove::Drop { .. } => unreachable!(),
            });
            logs[i].record(Observation::MyMove {
                move_number,
                usi,
                captured: captured.map(unpromote_role),
            });
            logs[1 - i].record(Observation::OpponentMoved {
                move_number,
                captured_my_piece_at: captured_sq,
            });
            if pos.in_check(pos.turn()) {
                let in_check = pos.turn();
                for log in logs.iter_mut() {
                    log.record(Observation::Check { in_check });
                }
            }
            match pos.outcome() {
                Some(Outcome::Checkmate { winner }) => break (Some(winner), "checkmate".into()),
                Some(Outcome::Stalemate { winner }) => break (Some(winner), "stalemate".into()),
                None => {}
            }
        };

        let first = first_move.unwrap_or_else(|| "-".into());
        let won = winner == Some(me);
        if won {
            wins_me += 1;
            win_plies.push(plies - rep.plies);
        }
        println!(
            "game {seed:2}: 初手 {first}{} → {} ({reason}, +{}手, 反則 先手{} 後手{})",
            if first == sc.target { "（注目手）" } else { "" },
            match winner {
                Some(c) if c == me => "勝ち",
                Some(_) => "負け",
                None => "引き分け",
            },
            plies - rep.plies,
            fouls[0],
            fouls[1],
        );
        *reasons.entry(reason).or_insert(0) += 1;
        *first_moves.entry(first).or_insert(0) += 1;
    }

    println!();
    println!("手番側の勝ち: {wins_me}/{games}");
    if !win_plies.is_empty() {
        win_plies.sort_unstable();
        println!(
            "勝ち局の追加手数: 中央値 {} / 最短 {} / 最長 {}",
            win_plies[win_plies.len() / 2],
            win_plies[0],
            win_plies[win_plies.len() - 1]
        );
    }
    let mut sorted: Vec<_> = first_moves.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    println!("初手の内訳:");
    for (usi, n) in sorted {
        let mark = if usi == sc.target { " ← 注目手" } else { "" };
        println!("  {usi}: {n}/{games}{mark}");
    }
    let mut sorted: Vec<_> = reasons.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    println!("終局理由:");
    for (r, n) in sorted {
        println!("  {r}: {n}/{games}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sc_name = args.first().map(String::as_str).unwrap_or("keima");
    let Some(sc) = SCENARIOS.iter().find(|s| s.name == sc_name) else {
        eprintln!(
            "未知のシナリオ: {sc_name}（候補: {}）",
            SCENARIOS
                .iter()
                .map(|s| s.name)
                .collect::<Vec<_>>()
                .join(", ")
        );
        std::process::exit(1);
    };

    let rep = replay(sc.plies);
    match args.get(1).map(String::as_str) {
        Some("diag") => {
            let n = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
            diagnose_particles(sc, &rep, n);
        }
        Some("continue") => {
            let games = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
            let name_me = args.get(3).map(String::as_str).unwrap_or("estimator");
            let name_opp = args.get(4).map(String::as_str).unwrap_or("estimator");
            continue_games(sc, &rep, games, name_me, name_opp);
        }
        mode => {
            let trials = mode.and_then(|s| s.parse().ok()).unwrap_or(20);
            let name = args.get(2).map(String::as_str).unwrap_or("estimator");
            choice_trials(sc, &rep, trials, name);
        }
    }
}
