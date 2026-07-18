//! 対局記録（records/*.jsonl）の事後分析。
//!
//! game:end の全公開棋譜（真実）をリプレイし、bot視点の問題を集計する:
//! - 反則の原因分類（見えない駒に経路を塞がれた / 王手放置 / 自ら王手に飛び込んだ / 打ちマスに駒）
//! - 駒交換の損得（取った直後に取り返されたか、そのネット価値）
//! - タダ取られ（守られていない駒を只で取られた）
//! - 1手詰みの存在（参考値: botからは玉位置が見えないため「逃し」を責める指標ではなく、
//!   玉位置推定が当たっていれば勝てた機会の総量を測る）
//! - 王手ソルバー（check.rs）の再現検証: 記録上の王手中の反則それぞれについて、
//!   その時点の観測だけからソルバーが選んだ手が真の局面で合法だったかを判定する
//!
//! 使い方: cargo run --release --bin analyze -- records/*.jsonl

use std::collections::{HashMap, HashSet};

use tsuitate_bot::check::CheckSolver;
use tsuitate_bot::model::GameModel;
use tsuitate_bot::observation::{Observation, ObservationLog};
use tsuitate_bot::protocol::{
    ClockState, Color, FoulCounts, FoulRecord, GameEndPayload, GameStatus, PlayerView,
};
use tsuitate_bot::shogi::{Outcome, Position, ShogiMove, parse_usi, piece_value};
use tsuitate_bot::strategy::candidate_moves;

struct GameRecord {
    file: String,
    bot_color: Color,
    strategy: String,
    observations: Vec<Observation>,
    end: GameEndPayload,
    /// (選択時の p_legal 予測, 実際に合法だったか)。chose イベントの
    /// debug.p_legal と、その手の受理/反則の突き合わせ（C-7 P3 の較正測定）
    p_legal_outcomes: Vec<(f64, bool)>,
}

fn load(path: &str) -> Option<GameRecord> {
    let content = std::fs::read_to_string(path).ok()?;
    let mut bot_color = None;
    let mut strategy = String::new();
    let mut observations = vec![];
    let mut end = None;
    let mut p_legal_outcomes = vec![];
    // 直近の chose イベントの (usi, p_legal)。次の MyMove/MyFoul 観測と照合する
    let mut pending_chose: Option<(String, f64)> = None;
    for line in content.lines() {
        let v: serde_json::Value = serde_json::from_str(line).ok()?;
        match v["type"].as_str() {
            Some("match") => {
                bot_color = serde_json::from_value(v["your_color"].clone()).ok();
                strategy = v["strategy"].as_str().unwrap_or("?").to_string();
            }
            Some("chose") => {
                if let (Some(usi), Some(p)) =
                    (v["usi"].as_str(), v["debug"]["p_legal"].as_f64())
                {
                    pending_chose = Some((usi.to_string(), p));
                }
            }
            Some("obs") => {
                if let Ok(obs) = serde_json::from_value::<Observation>(v["event"].clone()) {
                    match (&obs, &pending_chose) {
                        (Observation::MyMove { usi, .. }, Some((cu, p))) if usi == cu => {
                            p_legal_outcomes.push((*p, true));
                            pending_chose = None;
                        }
                        (Observation::MyFoul { usi, .. }, Some((cu, p))) if usi == cu => {
                            p_legal_outcomes.push((*p, false));
                            pending_chose = None;
                        }
                        _ => {}
                    }
                    observations.push(obs);
                }
            }
            Some("end") => {
                end = serde_json::from_value(v["payload"].clone()).ok();
            }
            _ => {}
        }
    }
    Some(GameRecord {
        file: path.to_string(),
        bot_color: bot_color?,
        strategy,
        observations,
        end: end?,
        p_legal_outcomes,
    })
}

/// 観測ログの復元から PlayerView 相当を作る（王手ソルバーの再現検証用）
fn view_from_model(model: &GameModel, in_check: bool) -> PlayerView {
    PlayerView {
        game_id: "replay".into(),
        your_color: model.my_color(),
        your_pieces: model.my_pieces(),
        your_hand: model.my_hand(),
        turn: model.my_color(),
        move_number: 0,
        clocks: ClockState {
            sente_ms: 0,
            gote_ms: 0,
            running: None,
            server_time: 0,
        },
        fouls: FoulCounts {
            you: model.my_fouls(),
            opponent: model.opponent_fouls(),
        },
        you_in_check: in_check,
        opponent_in_check: false,
        status: GameStatus::Playing,
    }
}

/// 記録上「王手中に反則した手番」それぞれを、ソルバー方策（解消確率の argmax）で
/// 最初から指し直し、合法手に到達するまでの反則回数を実際の反則回数と比較する。
/// 実運用と同じく、反則するたびにその手を仮説消去へ回して選び直す。
/// 戻り値: (検証した手番数, 実際の反則合計, ソルバー方策での反則合計)
fn simulate_check_solver(rec: &GameRecord, positions: &[Position], bot: Color) -> (u32, u32, u32) {
    // 手番ごとの実際の反則回数と、その手番の最初の反則の直前までの観測数
    let mut turns: Vec<(u32, usize, u32)> = vec![]; // (move_number, obs_prefix, actual_fouls)
    for (i, obs) in rec.observations.iter().enumerate() {
        let Observation::MyFoul { move_number, .. } = obs else {
            continue;
        };
        match turns.last_mut() {
            Some((mn, _, n)) if *mn == *move_number => *n += 1,
            _ => turns.push((*move_number, i, 1)),
        }
    }

    let mut tested = 0;
    let mut actual_total = 0;
    let mut solver_total = 0;
    for (move_number, prefix, actual) in turns {
        let idx = (move_number as usize).saturating_sub(1);
        let Some(truth) = positions.get(idx) else {
            continue;
        };
        if !truth.in_check(bot) {
            continue; // 王手以外の反則（経路封鎖など）はソルバーの対象外
        }
        let mut log = ObservationLog::default();
        for prev in &rec.observations[..prefix] {
            log.record(prev.clone());
        }
        let model = GameModel::from_log(bot, &log);
        let view = view_from_model(&model, true);

        let mut fouls: Vec<ShogiMove> = vec![];
        let mut tried: HashSet<String> = HashSet::new();
        let mut sim_fouls = 0u32;
        let mut sequence: Vec<String> = vec![];
        loop {
            if sim_fouls >= 10 {
                break; // 反則負け相当
            }
            let candidates = candidate_moves(&view, &tried);
            if candidates.is_empty() {
                break;
            }
            let Some(mut solver) = CheckSolver::new(&view, &[], &fouls, &log) else {
                break;
            };
            let best = candidates
                .iter()
                .map(|(usi, mv)| (usi.clone(), *mv, solver.resolve_probability(mv)))
                .max_by(|a, b| a.2.total_cmp(&b.2));
            let Some((usi, mv, p)) = best else { break };
            if truth.is_legal(&mv) {
                sequence.push(format!("{usi}(p{p:.2})○"));
                break;
            }
            sequence.push(format!("{usi}(p{p:.2})×"));
            sim_fouls += 1;
            fouls.push(mv);
            tried.insert(usi);
        }
        tested += 1;
        actual_total += actual;
        solver_total += sim_fouls;
        println!(
            "  王手手番 {move_number}手目: 実際の反則 {actual}回 → ソルバー方策 {sim_fouls}回 [{}]",
            sequence.join(" ")
        );
    }
    (tested, actual_total, solver_total)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum FoulCause {
    /// 経路上に見えない相手駒があって届かない（or 移動先の自駒 = 起きないはず）
    Blocked,
    /// 王手を受けていて、その手では解消できなかった（攻め駒の位置を知らない）
    CheckUnresolved,
    /// 王手は受けていないのに、指すと自玉が王手になる（ピン・利きへの飛び込み）
    IntoCheck,
    /// 持ち駒を打とうとしたマスに見えない駒があった
    DropOccupied,
    /// 打ち歩詰め
    PawnDropMate,
    /// 上記以外（想定外）
    Other,
}

fn classify_foul(pos: &Position, foul: &FoulRecord) -> FoulCause {
    let Some(mv) = parse_usi(&foul.usi) else {
        return FoulCause::Other;
    };
    // is_legal と同じ順で原因を切り分ける
    if !pos.is_pseudo_legal(&mv) {
        return match mv {
            ShogiMove::Board { .. } => FoulCause::Blocked,
            ShogiMove::Drop { .. } => FoulCause::DropOccupied,
        };
    }
    let mut probe = pos.clone();
    probe.play_unchecked(&mv);
    if probe.in_check(pos.turn()) {
        return if pos.in_check(pos.turn()) {
            FoulCause::CheckUnresolved
        } else {
            FoulCause::IntoCheck
        };
    }
    if let ShogiMove::Drop { .. } = mv {
        return FoulCause::PawnDropMate;
    }
    FoulCause::Other
}

fn cause_label(c: FoulCause) -> &'static str {
    match c {
        FoulCause::Blocked => "経路が見えない駒に塞がれた",
        FoulCause::CheckUnresolved => "王手を解消できない手（攻め駒の位置不明）",
        FoulCause::IntoCheck => "自ら王手に飛び込んだ（ピン/見えない利き）",
        FoulCause::DropOccupied => "打ちマスに見えない駒",
        FoulCause::PawnDropMate => "打ち歩詰め",
        FoulCause::Other => "その他",
    }
}

/// bot の手番で1手詰み（相手玉）が存在するか
fn has_mate_in_one(pos: &Position) -> Option<String> {
    for mv in pos.legal_moves() {
        let mut next = pos.clone();
        next.play_unchecked(&mv);
        if let Some(Outcome::Checkmate { winner }) = next.outcome() {
            if winner == pos.turn() {
                return Some(mv.to_usi());
            }
        }
    }
    None
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("使い方: analyze <records/*.jsonl>");
        std::process::exit(1);
    }

    let mut total_fouls: HashMap<FoulCause, u32> = HashMap::new();
    let mut total_bot_captured = 0.0;
    let mut total_bot_lost = 0.0;
    let mut total_free_losses = 0.0;
    let mut total_bad_trades = 0.0;
    let mut total_missed_mates = 0;
    let mut total_check_tested = 0;
    let mut total_check_solved = 0;
    let mut total_recap_ops = 0;
    let mut total_recap_taken = 0;
    let mut total_recap_missed_good = 0;
    let mut games = 0;
    let mut bot_wins = 0;
    let mut p_legal_all: Vec<(f64, bool)> = vec![];

    for path in &paths {
        let Some(rec) = load(path) else {
            eprintln!("読めませんでした（終局まで到達していない記録？）: {path}");
            continue;
        };
        games += 1;
        p_legal_all.extend(rec.p_legal_outcomes.iter().copied());
        let bot = rec.bot_color;
        let bot_won = matches!(
            (rec.end.result.as_str(), bot),
            ("sente_win", Color::Sente) | ("gote_win", Color::Gote)
        );
        if bot_won {
            bot_wins += 1;
        }
        println!("\n=== {} ===", rec.file);
        println!(
            "bot={:?} ({}) vs {} / 結果: {} ({}) {}",
            bot,
            rec.strategy,
            rec.end.opponent.username,
            rec.end.result,
            rec.end.reason,
            if bot_won { "→ bot勝ち" } else { "→ bot負け" },
        );

        // 反則の原因分類（局面 moveNumber = その時点までに moveNumber-1 手が指されている）
        let mut positions = vec![Position::initial()];
        for m in &rec.end.moves {
            let mut next = positions.last().unwrap().clone();
            let Some(mv) = parse_usi(&m.usi) else {
                eprintln!("  棋譜の手をパースできません: {}", m.usi);
                break;
            };
            next.play_unchecked(&mv);
            positions.push(next);
        }

        let mut fouls_here: HashMap<FoulCause, u32> = HashMap::new();
        for foul in rec.end.foul_attempts.iter().filter(|f| f.by_color == bot) {
            let idx = (foul.move_number as usize).saturating_sub(1);
            if idx >= positions.len() {
                continue;
            }
            let cause = classify_foul(&positions[idx], foul);
            *fouls_here.entry(cause).or_default() += 1;
            *total_fouls.entry(cause).or_default() += 1;
            println!(
                "  反則 {}手目 {}: {}",
                foul.move_number,
                foul.usi,
                cause_label(cause)
            );
        }
        let _ = fouls_here;

        // 駒の損得: 各手の捕獲を追い、直後の取り返しをペアにする
        let mut bot_captured = 0.0;
        let mut bot_lost = 0.0;
        let mut free_losses: Vec<String> = vec![];
        let mut bad_trades: Vec<String> = vec![];
        for (i, m) in rec.end.moves.iter().enumerate() {
            let pos = &positions[i];
            let Some(mv) = parse_usi(&m.usi) else { break };
            let to = match mv {
                ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
            };
            let captured = pos.piece_at(to).map(|p| p.role);
            let Some(role) = captured else { continue };
            let v = piece_value(role);
            if m.by_color == bot {
                bot_captured += v;
            } else {
                bot_lost += v;
                // 取り返したか（次の bot の正規手が同じマスを取ったか）
                let recaptured = rec.end.moves.get(i + 1).is_some_and(|n| {
                    n.by_color == bot
                        && parse_usi(&n.usi).is_some_and(|nm| match nm {
                            ShogiMove::Board { to: t, .. } | ShogiMove::Drop { to: t, .. } => {
                                t == to && positions[i + 1].piece_at(t).is_some()
                            }
                        })
                });
                if !recaptured {
                    // 守られていたのに取り返さなかったのか、そもそも守っていなかったのか
                    free_losses.push(format!(
                        "{}手目 {} で {:?}(価値{v:.0}) を只取られ",
                        i + 1,
                        m.usi,
                        role
                    ));
                }
            }
            // bot が取った直後に取り返された交換のネット
            if m.by_color == bot {
                if let Some(n) = rec.end.moves.get(i + 1) {
                    if n.by_color != bot {
                        if let Some(nm) = parse_usi(&n.usi) {
                            let nt = match nm {
                                ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
                            };
                            if nt == to {
                                if let Some(lost) = positions[i + 1].piece_at(nt) {
                                    let net = v - piece_value(lost.role);
                                    if net < -1.5 {
                                        bad_trades.push(format!(
                                            "{}手目 {}: {:?}(価値{:.0}) を取ったが {:?}(価値{:.0}) を取り返され ネット{net:+.0}",
                                            i + 1,
                                            m.usi,
                                            role,
                                            v,
                                            lost.role,
                                            piece_value(lost.role),
                                        ));
                                        total_bad_trades += -net;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        total_bot_captured += bot_captured;
        total_bot_lost += bot_lost;
        for l in &free_losses {
            println!("  {l}");
        }
        total_free_losses += free_losses.len() as f64;
        for t in &bad_trades {
            println!("  {t}");
        }
        println!("  駒得収支: 取った {bot_captured:.0} / 取られた {bot_lost:.0}（歩=1換算）");

        // 王手ソルバーの再現検証（王手中に反則した手番それぞれを指し直す）
        let (tested, actual, sim) = simulate_check_solver(&rec, &positions, bot);
        let _ = tested;
        total_check_tested += actual;
        total_check_solved += sim;

        // 取り返し機会: 相手に駒を取られた直後の bot 手番で、そのマスを合法に
        // 取り返せたか（bot は取られたマス = 相手駒の現在地を正確に知っている）
        for (i, m) in rec.end.moves.iter().enumerate() {
            if m.by_color == bot {
                continue;
            }
            let Some(mv) = parse_usi(&m.usi) else { break };
            let to = match mv {
                ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
            };
            if positions[i].piece_at(to).is_none() {
                continue; // 捕獲ではない
            }
            let Some(after) = positions.get(i + 1) else { break };
            if after.turn() != bot || after.outcome().is_some() {
                continue;
            }
            total_recap_ops += 1;
            let attacker_value = piece_value(after.piece_at(to).map(|p| p.role).unwrap());
            let recaps: Vec<ShogiMove> = after
                .legal_moves()
                .into_iter()
                .filter(|lm| matches!(lm, ShogiMove::Board { to: t, .. } if *t == to))
                .collect();
            let actually = rec.end.moves.get(i + 1).and_then(|n| parse_usi(&n.usi));
            let took = actually.is_some_and(|am| match am {
                ShogiMove::Board { to: t, .. } | ShogiMove::Drop { to: t, .. } => t == to,
            });
            if took {
                total_recap_taken += 1;
            } else if let Some(best) = recaps.first() {
                // 取り返し後にさらに取り返されるか（真の局面で）
                let mut probe = after.clone();
                let own = match best {
                    ShogiMove::Board { from, .. } => {
                        after.piece_at(*from).map(|p| piece_value(p.role)).unwrap_or(0.0)
                    }
                    ShogiMove::Drop { .. } => 0.0,
                };
                probe.play_unchecked(best);
                let exposed = probe.is_attacked(to, bot.other());
                let net = attacker_value - if exposed { own } else { 0.0 };
                if net > 0.5 {
                    total_recap_missed_good += 1;
                    println!(
                        "  取り返し逃し {}手目: {} で {:.0} を回収できた（推定ネット{net:+.0}）が {} を選択",
                        i + 2,
                        best.to_usi(),
                        attacker_value,
                        rec.end
                            .moves
                            .get(i + 1)
                            .map(|n| n.usi.as_str())
                            .unwrap_or("-"),
                    );
                }
            }
        }

        // 詰み逃し: bot 手番の各局面で1手詰みがあったか
        for (i, pos) in positions.iter().enumerate() {
            if pos.turn() != bot {
                continue;
            }
            if pos.outcome().is_some() {
                break;
            }
            if let Some(mate) = has_mate_in_one(pos) {
                let played = rec.end.moves.get(i).map(|m| m.usi.as_str()).unwrap_or("-");
                // 実際に詰ませた手なら逃していない
                if i + 1 == positions.len() - 1
                    && positions.last().unwrap().outcome().is_some()
                {
                    continue;
                }
                println!(
                    "  1手詰みが存在 {}手目: {mate}（実際は {played}。玉位置は不可視なので参考値）",
                    i + 1
                );
                total_missed_mates += 1;
            }
        }
    }

    println!("\n=== 集計（{games}局 bot {bot_wins}勝）===");
    println!("反則の原因:");
    let mut causes: Vec<_> = total_fouls.iter().collect();
    causes.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
    for (cause, n) in causes {
        println!("  {:>3}回  {}", n, cause_label(*cause));
    }
    println!("駒得収支合計: 取った {total_bot_captured:.0} / 取られた {total_bot_lost:.0}");
    println!("只取られ回数: {total_free_losses:.0} / 損な交換の累計損失: {total_bad_trades:.0}");
    println!(
        "取り返し: 機会{total_recap_ops}回中 実行{total_recap_taken}回 / 得だったのに逃した{total_recap_missed_good}回"
    );
    println!("1手詰みの存在（参考値・玉位置は不可視）: {total_missed_mates}回");
    if total_check_tested > 0 {
        println!(
            "王手中の反則: 実際 {total_check_tested}回 → ソルバー方策なら {total_check_solved}回"
        );
    }
    // p_legal の較正（C-7 P3）: 選択手の合法確率予測 vs 実際の受理/反則。
    // Brier = mean((p-y)^2)（小さいほど良い）。参考: 常に基底率を答える予測の Brier
    if !p_legal_all.is_empty() {
        let n = p_legal_all.len() as f64;
        let base_rate = p_legal_all.iter().filter(|(_, y)| *y).count() as f64 / n;
        let brier: f64 = p_legal_all
            .iter()
            .map(|(p, y)| {
                let y = if *y { 1.0 } else { 0.0 };
                (p - y) * (p - y)
            })
            .sum::<f64>()
            / n;
        let base_brier = base_rate * (1.0 - base_rate);
        let logloss: f64 = p_legal_all
            .iter()
            .map(|(p, y)| {
                let p = p.clamp(1e-6, 1.0 - 1e-6);
                if *y { -p.ln() } else { -(1.0 - p).ln() }
            })
            .sum::<f64>()
            / n;
        println!(
            "p_legal 較正（{}手 合法率{:.1}%）: Brier {:.4}（基底率予測 {:.4}）/ logloss {:.4}",
            p_legal_all.len(),
            base_rate * 100.0,
            brier,
            base_brier,
            logloss,
        );
    }
}
