//! アリーナ記録から「取られた駒を取り返す手」がなぜ選ばれないかを測る。
//!
//! 取り返しは**ついたてで唯一「相手の駒の位置が確実に分かっている」場面**
//! （観測 `OpponentMoved{captured_my_piece_at}` = 相手駒の現在地）なので、
//! ここで取りに行けていないなら信念ではなく評価の問題になる。
//! `bin/analyze` の実測では機会713回中の実行が235回（33%）しかない。
//!
//! bot視点の定義: 直前の相手手が自駒を取ったマス X（観測そのもの）。
//! 「取り返し手」= X へ動かす自分の候補手。真実盤面は診断
//! （X にいる敵駒の価値・取り返した後に取り返されるか）にだけ使う。
//!
//! 使い方:
//!   TSUITATE_THINK_BUDGET_MS=900 cargo run --release --bin recapture_probe -- records/*.jsonl
//!   cargo run --release --bin recapture_probe -- --seed 42 --jobs 4 records/*.jsonl

use std::collections::HashSet;
use std::time::Instant;

use tsuitate_bot::board::{Coord, make_usi_square};
use tsuitate_bot::observation::{Observation, ObservationLog};
use tsuitate_bot::protocol::{Color, GameEndPayload, Role};
use tsuitate_bot::scenario_core::{clone_log, make_view, side_idx};
use tsuitate_bot::selfplay::mix;
use tsuitate_bot::shogi::{Position, ShogiMove, parse_usi, piece_value, unpromote_role};
use tsuitate_bot::strategy::{self, CandidateScore};

struct GameRecord {
    file: String,
    bot_color: Color,
    end: GameEndPayload,
}

fn load(path: &str) -> Option<GameRecord> {
    let content = std::fs::read_to_string(path).ok()?;
    let mut bot_color = None;
    let mut end = None;
    for line in content.lines() {
        let v: serde_json::Value = serde_json::from_str(line).ok()?;
        match v["type"].as_str() {
            Some("match") => bot_color = serde_json::from_value(v["your_color"].clone()).ok(),
            Some("end") => end = serde_json::from_value(v["payload"].clone()).ok(),
            _ => {}
        }
    }
    Some(GameRecord {
        file: path.to_string(),
        bot_color: bot_color?,
        end: end?,
    })
}

#[derive(Debug, Default, Clone)]
struct Summary {
    /// 取り返し機会（＝直前に自駒を取られた bot 手番）
    cases: u64,
    /// うち候補手に「X へ動かす手」が1つでもあった件数
    with_candidate: u64,
    /// bot が実際に取り返しを選んだ件数
    chosen_recapture: u64,
    /// 取り返し手の順位（1始まり）
    static_rank_sum: f64,
    final_rank_sum: f64,
    depth2_in: u64,
    /// 取り返し手と選択手のスコア内訳の差（取り返し − 選択手）
    d_score: f64,
    d_gain: f64,
    d_p_legal: f64,
    d_foul_cost: f64,
    d_adjust: f64,
    d_capture_bet: f64,
    /// gain の内訳の差（取り返し − 選択手）
    d_capture_value: f64,
    d_risk: f64,
    /// 取り返し手そのものの内訳（絶対値）
    recap_capture_value: f64,
    recap_risk: f64,
    /// 取り返し手・選択手それぞれの p_legal の絶対値
    p_legal_recap: f64,
    p_legal_chosen: f64,
    /// 真実側の診断
    truth_profitable: u64,
    truth_profitable_taken: u64,
    captured_value_sum: f64,
    /// 取り返しに使う自駒の価値（安い駒で取れるか）
    recapture_piece_value_sum: f64,
}

impl Summary {
    fn merge(&mut self, o: &Summary) {
        self.cases += o.cases;
        self.with_candidate += o.with_candidate;
        self.chosen_recapture += o.chosen_recapture;
        self.static_rank_sum += o.static_rank_sum;
        self.final_rank_sum += o.final_rank_sum;
        self.depth2_in += o.depth2_in;
        self.d_score += o.d_score;
        self.d_gain += o.d_gain;
        self.d_p_legal += o.d_p_legal;
        self.d_foul_cost += o.d_foul_cost;
        self.d_adjust += o.d_adjust;
        self.d_capture_bet += o.d_capture_bet;
        self.d_capture_value += o.d_capture_value;
        self.d_risk += o.d_risk;
        self.recap_capture_value += o.recap_capture_value;
        self.recap_risk += o.recap_risk;
        self.p_legal_recap += o.p_legal_recap;
        self.p_legal_chosen += o.p_legal_chosen;
        self.truth_profitable += o.truth_profitable;
        self.truth_profitable_taken += o.truth_profitable_taken;
        self.captured_value_sum += o.captured_value_sum;
        self.recapture_piece_value_sum += o.recapture_piece_value_sum;
    }
}

fn rerank(
    view: &tsuitate_bot::protocol::PlayerView,
    log: &ObservationLog,
    foul_tried: &HashSet<String>,
    seed: u64,
) -> Option<(String, Vec<CandidateScore>)> {
    let mut strat = strategy::make_seeded("estimator", seed).expect("estimator がありません");
    let warm = clone_log(log);
    strategy::prewarm_strategy(&mut *strat, view, &warm);
    let chosen = strat.choose(view, log, foul_tried)?;
    let ranking = strat.last_ranking()?.to_vec();
    Some((chosen, ranking))
}

fn moves_to(usi: &str, target: Coord) -> bool {
    match parse_usi(usi) {
        // 打ちは「取り返し」にならない（相手駒がいるマスには打てない）
        Some(ShogiMove::Board { to, .. }) => to == target,
        _ => false,
    }
}

/// 記録の再生（flee_probe と同じ手順。観測列を両者ぶん作る）
fn add_foul_obs(
    side: Color,
    usi: String,
    pos: &Position,
    logs: &mut [ObservationLog; 2],
    fouls: &mut [u32; 2],
) {
    fouls[side_idx(side)] += 1;
    logs[side_idx(side)].record(Observation::MyFoul {
        move_number: pos.move_number(),
        usi,
    });
    logs[side_idx(side.other())].record(Observation::OpponentFoul {
        count: fouls[side_idx(side)],
    });
}

fn add_move_obs(
    side: Color,
    mv: &ShogiMove,
    captured: Option<Role>,
    pos: &Position,
    logs: &mut [ObservationLog; 2],
) {
    let usi = mv.to_usi();
    let move_number = pos.move_number();
    let captured_sq = captured.map(|_| match *mv {
        ShogiMove::Board { to, .. } => make_usi_square(to),
        ShogiMove::Drop { .. } => unreachable!("打ちは駒を取れない"),
    });
    logs[side_idx(side)].record(Observation::MyMove {
        move_number,
        usi,
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

/// 集計対象。`all` は全機会、`missed` は「真実では得なのに取り返さなかった」機会だけ
/// （平均が「取り返すべきでない機会」の外れ値に引きずられないよう分ける）
#[derive(Default)]
struct Summaries {
    all: Summary,
    missed: Summary,
}

fn probe_game(rec: &GameRecord, file_seed: u64, sum: &mut Summaries) -> bool {
    let bot = rec.bot_color;
    let mut pos = Position::initial();
    let mut logs = [ObservationLog::default(), ObservationLog::default()];
    let mut fouls = [0u32; 2];
    let mut foul_idx = 0usize;
    let mut fouls_sorted = rec.end.foul_attempts.clone();
    fouls_sorted.sort_by_key(|f| f.move_number);
    // 直前の相手手が自駒を取ったマス（観測そのもの）
    let mut captured_at: Option<Coord> = None;

    for (move_idx, m) in rec.end.moves.iter().enumerate() {
        let side = pos.turn();
        while foul_idx < fouls_sorted.len()
            && fouls_sorted[foul_idx].move_number == pos.move_number()
            && fouls_sorted[foul_idx].by_color == side
        {
            add_foul_obs(
                side,
                fouls_sorted[foul_idx].usi.clone(),
                &pos,
                &mut logs,
                &mut fouls,
            );
            foul_idx += 1;
        }

        // 直前の相手手が自駒を取っていれば、この bot 手番が取り返し機会。
        // captured_at はループ末尾で毎手更新されるのでここでクリアは要らない
        if side == bot {
            if let Some(target) = captured_at {
                measure(rec, &pos, target, move_idx, &logs, &fouls, &fouls_sorted, file_seed, sum);
            }
        }

        let Some(mv) = parse_usi(&m.usi) else {
            eprintln!("  棋譜の手をパースできません: {} ({})", m.usi, rec.file);
            return false;
        };
        if m.by_color != side || !pos.is_legal(&mv) {
            eprintln!("  棋譜の手が局面と合いません: {} ({})", m.usi, rec.file);
            return false;
        }
        let captured = pos.play_unchecked(&mv);
        add_move_obs(side, &mv, captured, &pos, &mut logs);
        // 相手が自駒を取った → 次の bot 手番が取り返し機会
        captured_at = match (side != bot, captured, &mv) {
            (true, Some(_), ShogiMove::Board { to, .. }) => Some(*to),
            _ => None,
        };
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn measure(
    rec: &GameRecord,
    pos: &Position,
    target: Coord,
    move_idx: usize,
    logs: &[ObservationLog; 2],
    fouls: &[u32; 2],
    fouls_sorted: &[tsuitate_bot::protocol::FoulRecord],
    file_seed: u64,
    sums: &mut Summaries,
) {
    let sum = &mut sums.all;
    let bot = rec.bot_color;
    let view = make_view(pos, bot, fouls);
    let log = &logs[side_idx(bot)];
    let tried: HashSet<String> = fouls_sorted
        .iter()
        .filter(|f| f.move_number == pos.move_number() && f.by_color == bot)
        .map(|f| f.usi.clone())
        .collect();
    let seed = mix(file_seed ^ u64::from(pos.move_number()));
    let Some((chosen, ranking)) = rerank(&view, log, &tried, seed) else {
        return;
    };
    sum.cases += 1;

    // 真実側の診断: X にいる敵駒の価値と、取り返しが得か（取り返し後に取られるか）
    let attacker_value = pos
        .piece_at(target)
        .filter(|p| p.color != bot)
        .map(|p| piece_value(p.role))
        .unwrap_or(0.0);
    let sum = &mut sums.all;
    sum.captured_value_sum += attacker_value;
    let profitable = pos
        .legal_moves()
        .into_iter()
        .filter(|mv| matches!(mv, ShogiMove::Board { to, .. } if *to == target))
        .map(|mv| {
            let own = match mv {
                ShogiMove::Board { from, .. } => {
                    pos.piece_at(from).map(|p| piece_value(p.role)).unwrap_or(0.0)
                }
                ShogiMove::Drop { .. } => 0.0,
            };
            let mut probe = pos.clone();
            probe.play_unchecked(&mv);
            let exposed = probe.is_attacked(target, bot.other());
            attacker_value - if exposed { own } else { 0.0 }
        })
        .fold(f64::NEG_INFINITY, f64::max);
    let profitable = profitable.is_finite() && profitable > 0.5;
    if profitable {
        sum.truth_profitable += 1;
    }

    // 候補の中の最良の取り返し手（最終スコア順・静的スコア順それぞれの順位）
    let mut by_static: Vec<&CandidateScore> = ranking.iter().collect();
    by_static.sort_by(|a, b| {
        b.static_score
            .partial_cmp(&a.static_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let static_rank = by_static.iter().position(|c| moves_to(&c.usi, target));
    let final_rank = ranking.iter().position(|c| moves_to(&c.usi, target));
    let (Some(sr), Some(fr)) = (static_rank, final_rank) else {
        return; // 取り返し手が候補に無い（届く自駒がない）
    };
    let recap = &ranking[fr];
    let Some(chosen_score) = ranking.iter().find(|c| c.usi == chosen) else {
        return;
    };
    let took = moves_to(&chosen, target);
    let mut one = Summary {
        with_candidate: 1,
        static_rank_sum: sr as f64 + 1.0,
        final_rank_sum: fr as f64 + 1.0,
        depth2_in: recap.depth2 as u64,
        d_score: recap.score - chosen_score.score,
        d_gain: recap.gain - chosen_score.gain,
        d_p_legal: recap.p_legal - chosen_score.p_legal,
        d_foul_cost: recap.foul_cost - chosen_score.foul_cost,
        d_adjust: recap.adjust - chosen_score.adjust,
        d_capture_bet: recap.capture_bet_penalty - chosen_score.capture_bet_penalty,
        d_capture_value: recap.capture_value - chosen_score.capture_value,
        d_risk: recap.risk - chosen_score.risk,
        recap_capture_value: recap.capture_value,
        recap_risk: recap.risk,
        p_legal_recap: recap.p_legal,
        p_legal_chosen: chosen_score.p_legal,
        chosen_recapture: took as u64,
        // 真実側の値（部分集合の集計でも出せるよう one に載せる）
        captured_value_sum: attacker_value,
        ..Summary::default()
    };
    if let Some(ShogiMove::Board { from, .. }) = parse_usi(&recap.usi) {
        one.recapture_piece_value_sum = pos
            .piece_at(from)
            .map(|p| piece_value(p.role))
            .unwrap_or(0.0);
    }
    sum.merge(&one);
    if took && profitable {
        sum.truth_profitable_taken += 1;
    }
    // 「真実では得なのに取り返さなかった」機会だけの集計（本題の部分集合）
    if profitable && !took {
        sums.missed.cases += 1;
        sums.missed.merge(&one);
    }
    let _ = move_idx;
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut seed = 7u64;
    let mut files: Vec<String> = vec![];
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--seed" => seed = it.next().and_then(|v| v.parse().ok()).unwrap_or(seed),
            _ => files.push(a),
        }
    }
    if files.is_empty() {
        eprintln!("usage: recapture_probe [--seed N] <records/*.jsonl>");
        std::process::exit(1);
    }
    let started = Instant::now();
    let mut sums = Summaries::default();
    let mut games = 0u64;
    for (i, f) in files.iter().enumerate() {
        let Some(rec) = load(f) else {
            eprintln!("読めません: {f}");
            continue;
        };
        let mut local = Summaries::default();
        if probe_game(&rec, mix(seed ^ i as u64), &mut local) {
            sums.all.merge(&local.all);
            sums.missed.merge(&local.missed);
            games += 1;
        }
        if (i + 1) % 10 == 0 {
            eprintln!("  {}/{} 局（{:.0}秒）", i + 1, files.len(), started.elapsed().as_secs_f64());
        }
    }

    let sum = &sums.all;
    let n = sum.with_candidate.max(1) as f64;
    println!("=== 取り返し probe（{games}局・{}件の機会）===", sum.cases);
    println!(
        "候補に取り返し手あり: {} 件（{:.1}%）/ 実際に取り返した: {} 件（{:.1}%）",
        sum.with_candidate,
        sum.with_candidate as f64 / sum.cases.max(1) as f64 * 100.0,
        sum.chosen_recapture,
        sum.chosen_recapture as f64 / n * 100.0
    );
    println!(
        "取り返し手の順位: 静的 {:.1} 位 / 最終 {:.1} 位 / 2手読み入り {:.1}%",
        sum.static_rank_sum / n,
        sum.final_rank_sum / n,
        sum.depth2_in as f64 / n * 100.0
    );
    println!(
        "スコア差（取り返し − 選択手）: score {:+.3} = gain {:+.3} / p_legal {:+.3} / foul_cost {:+.3} / adjust {:+.3} / 賭け分散 {:+.3}",
        sum.d_score / n,
        sum.d_gain / n,
        sum.d_p_legal / n,
        sum.d_foul_cost / n,
        sum.d_adjust / n,
        sum.d_capture_bet / n
    );
    println!(
        "p_legal 絶対値: 取り返し {:.3} / 選択手 {:.3}",
        sum.p_legal_recap / n,
        sum.p_legal_chosen / n
    );
    println!(
        "真実: 取り返しが得だった {} 件 / うち実行 {} 件（{:.1}%）/ 取られた駒を取った敵駒の平均価値 {:.2} / 取り返しに使う自駒の平均価値 {:.2}",
        sum.truth_profitable,
        sum.truth_profitable_taken,
        sum.truth_profitable_taken as f64 / sum.truth_profitable.max(1) as f64 * 100.0,
        sum.captured_value_sum / sum.cases.max(1) as f64,
        sum.recapture_piece_value_sum / n
    );

    // 本題: 真実では得なのに取り返さなかった機会だけを見る
    let m = &sums.missed;
    let k = m.with_candidate.max(1) as f64;
    println!();
    println!("=== 得なのに取り返さなかった {} 件の内訳 ===", m.with_candidate);
    println!(
        "取り返し手の順位: 静的 {:.1} 位 / 最終 {:.1} 位 / 2手読み入り {:.1}%",
        m.static_rank_sum / k,
        m.final_rank_sum / k,
        m.depth2_in as f64 / k * 100.0
    );
    println!(
        "スコア差（取り返し − 選択手）: score {:+.3} = gain {:+.3} / p_legal {:+.3} / adjust {:+.3}",
        m.d_score / k,
        m.d_gain / k,
        m.d_p_legal / k,
        m.d_adjust / k
    );
    println!(
        "  gain の内訳の差: 期待駒得 {:+.3} / 取られリスク {:+.3}（正 = 取り返し手の方がリスクが高い）",
        m.d_capture_value / k,
        m.d_risk / k
    );
    println!(
        "  取り返し手そのもの: 期待駒得 {:.3} / 取られリスク {:.3} / p_legal {:.3}",
        m.recap_capture_value / k,
        m.recap_risk / k,
        m.p_legal_recap / k
    );
    println!(
        "  取り返しに使う自駒の平均価値 {:.2}（取られた駒を取った敵駒 {:.2}）",
        m.recapture_piece_value_sum / k,
        m.captured_value_sum / k
    );
}
