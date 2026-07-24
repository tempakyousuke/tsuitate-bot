//! アリーナ記録から「既知の当たりが付いた自駒」の逃げ手がなぜ選ばれないかを測る。
//!
//! bot視点の en-prise 判定は観測履歴だけから作る:
//! 相手が自駒を取ったマスを「位置を観測した敵駒」とし、そのマスから駒種不明の
//! 敵駒が自駒を攻撃しうるなら当たりとみなす。真実盤面の敵駒種・現在位置は使わない。
//! 真実盤面は、逃げなかった後に次の相手手で実際に取られたかを見る診断だけに使う。
//!
//! 使い方:
//!   TSUITATE_THINK_BUDGET_MS=900 cargo run --release --bin flee_probe -- records/*.jsonl
//!   cargo run --release --bin flee_probe -- --seed 42 --jobs 4 records/*.jsonl

use std::collections::{HashMap, HashSet};
use std::sync::mpsc;
use std::time::Instant;

use tsuitate_bot::board::{make_usi_square, parse_usi_square, Coord};
use tsuitate_bot::observation::{Observation, ObservationLog};
use tsuitate_bot::protocol::{Color, GameEndPayload, PlayerView, Role};
use tsuitate_bot::scenario_core::{clone_log, make_view, side_idx};
use tsuitate_bot::selfplay::mix;
use tsuitate_bot::shogi::{parse_usi, piece_value, unpromote_role, Position, ShogiMove};
use tsuitate_bot::strategy::{self, CandidateScore};

#[derive(Clone)]
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

#[derive(Debug, Clone)]
struct EnPrisePiece {
    square: Coord,
    role: Role,
    attackers: Vec<Coord>,
}

#[derive(Debug, Default, Clone)]
struct Summary {
    cases: u64,
    positions: HashSet<(u64, u32)>,
    static_rank_sum: f64,
    final_rank_sum: f64,
    depth2_in: u64,
    chosen_flee: u64,
    record_flee: u64,
    actually_taken_next: u64,
    exchange_loss_sum: f64,
    diff_score_sum: f64,
    diff_gain_sum: f64,
    diff_p_legal_sum: f64,
    diff_foul_cost_sum: f64,
    diff_adjust_sum: f64,
    diff_checker_removal_sum: f64,
    diff_capture_bet_penalty_sum: f64,
}

impl Summary {
    fn add_case(&mut self, key: (u64, u32), m: &Measurement) {
        self.cases += 1;
        self.positions.insert(key);
        self.static_rank_sum += m.static_rank as f64;
        self.final_rank_sum += m.final_rank as f64;
        self.depth2_in += m.depth2_in as u64;
        self.chosen_flee += m.chosen_flee as u64;
        self.record_flee += m.record_flee as u64;
        self.actually_taken_next += m.taken_next as u64;
        self.exchange_loss_sum += m.exchange_loss;
        self.diff_score_sum += m.diff.score;
        self.diff_gain_sum += m.diff.gain;
        self.diff_p_legal_sum += m.diff.p_legal;
        self.diff_foul_cost_sum += m.diff.foul_cost;
        self.diff_adjust_sum += m.diff.adjust;
        self.diff_checker_removal_sum += m.diff.checker_removal;
        self.diff_capture_bet_penalty_sum += m.diff.capture_bet_penalty;
    }

    fn merge(&mut self, other: Summary) {
        self.cases += other.cases;
        self.positions.extend(other.positions);
        self.static_rank_sum += other.static_rank_sum;
        self.final_rank_sum += other.final_rank_sum;
        self.depth2_in += other.depth2_in;
        self.chosen_flee += other.chosen_flee;
        self.record_flee += other.record_flee;
        self.actually_taken_next += other.actually_taken_next;
        self.exchange_loss_sum += other.exchange_loss_sum;
        self.diff_score_sum += other.diff_score_sum;
        self.diff_gain_sum += other.diff_gain_sum;
        self.diff_p_legal_sum += other.diff_p_legal_sum;
        self.diff_foul_cost_sum += other.diff_foul_cost_sum;
        self.diff_adjust_sum += other.diff_adjust_sum;
        self.diff_checker_removal_sum += other.diff_checker_removal_sum;
        self.diff_capture_bet_penalty_sum += other.diff_capture_bet_penalty_sum;
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct ScoreDiff {
    score: f64,
    gain: f64,
    p_legal: f64,
    foul_cost: f64,
    adjust: f64,
    checker_removal: f64,
    capture_bet_penalty: f64,
}

#[derive(Debug)]
struct Measurement {
    static_rank: usize,
    final_rank: usize,
    depth2_in: bool,
    chosen_flee: bool,
    record_flee: bool,
    taken_next: bool,
    exchange_loss: f64,
    diff: ScoreDiff,
}

fn known_enemy_squares(log: &ObservationLog) -> HashSet<Coord> {
    let mut known = HashSet::new();
    for e in log.events() {
        match e {
            Observation::OpponentMoved {
                captured_my_piece_at: Some(sq),
                ..
            } => {
                if let Some(sq) = parse_usi_square(sq) {
                    known.insert(sq);
                }
            }
            Observation::MyMove { usi, .. } => {
                if let Some(mv) = parse_usi(usi) {
                    let to = match mv {
                        ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
                    };
                    known.remove(&to);
                }
            }
            _ => {}
        }
    }
    known
}

fn blocked_by_own_squares(from: Coord, to: Coord, own: &HashSet<Coord>) -> bool {
    let df = to.file - from.file;
    let dr = to.rank - from.rank;
    let steps = df.abs().max(dr.abs());
    if steps <= 1 {
        return false;
    }
    let sf = df.signum();
    let sr = dr.signum();
    (1..steps).any(|k| {
        own.contains(&Coord {
            file: from.file + sf * k,
            rank: from.rank + sr * k,
        })
    })
}

fn visible_own_squares(view: &PlayerView) -> HashSet<Coord> {
    view.your_pieces
        .iter()
        .filter_map(|p| parse_usi_square(&p.square))
        .collect()
}

fn one_step(df: i8, dr: i8, dirs: &[(i8, i8)]) -> bool {
    dirs.iter().any(|&(a, b)| df == a && dr == b)
}

fn ray(df: i8, dr: i8, dirs: &[(i8, i8)]) -> bool {
    dirs.iter().any(|&(a, b)| {
        let n = if a == 0 { dr / b } else { df / a };
        n > 0 && df == a * n && dr == b * n
    })
}

/// 駒種不明の既知敵駒が from から target を攻撃しうるか。
/// 敵駒種・真実の占有は見ず、盤上の自駒だけを経路遮断として使う。
fn possible_unknown_attack(from: Coord, target: Coord, attacker: Color, view: &PlayerView) -> bool {
    possible_unknown_attack_with_own(from, target, attacker, &visible_own_squares(view))
}

fn possible_unknown_attack_with_own(
    from: Coord,
    target: Coord,
    attacker: Color,
    own: &HashSet<Coord>,
) -> bool {
    if from == target {
        return false;
    }
    let df = target.file - from.file;
    let raw_dr = target.rank - from.rank;
    let dr = match attacker {
        Color::Sente => raw_dr,
        Color::Gote => -raw_dr,
    };

    let gold = [(0, -1), (-1, -1), (1, -1), (-1, 0), (1, 0), (0, 1)];
    let silver = [(0, -1), (-1, -1), (1, -1), (-1, 1), (1, 1)];
    let king = [
        (0, -1),
        (-1, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (0, 1),
        (-1, 1),
        (1, 1),
    ];
    let bishop = [(-1, -1), (1, -1), (-1, 1), (1, 1)];
    let rook = [(0, -1), (-1, 0), (1, 0), (0, 1)];

    let short = one_step(df, dr, &king)
        || one_step(df, dr, &gold)
        || one_step(df, dr, &silver)
        || (df == 0 && dr == -1)
        || (df.abs() == 1 && dr == -2);
    let long = (df == 0 && dr < 0) || ray(df, dr, &bishop) || ray(df, dr, &rook);
    (short || long) && !blocked_by_own_squares(from, target, own)
}

fn en_prise_pieces(view: &PlayerView, log: &ObservationLog) -> Vec<EnPrisePiece> {
    let known = known_enemy_squares(log);
    if known.is_empty() {
        return vec![];
    }
    let attacker = view.your_color.other();
    let mut out = vec![];
    for p in &view.your_pieces {
        if p.role == Role::King {
            continue;
        }
        let Some(square) = parse_usi_square(&p.square) else {
            continue;
        };
        let attackers: Vec<Coord> = known
            .iter()
            .copied()
            .filter(|&a| possible_unknown_attack(a, square, attacker, view))
            .collect();
        if !attackers.is_empty() {
            out.push(EnPrisePiece {
                square,
                role: p.role,
                attackers,
            });
        }
    }
    out
}

fn still_attacked_after_move(
    from: Coord,
    to: Coord,
    attackers: &[Coord],
    captured_attacker: Option<Coord>,
    view: &PlayerView,
) -> bool {
    let attacker = view.your_color.other();
    let mut own = visible_own_squares(view);
    own.remove(&from);
    own.insert(to);
    attackers.iter().any(|&a| {
        Some(a) != captured_attacker && possible_unknown_attack_with_own(a, to, attacker, &own)
    })
}

fn is_flee_move(usi: &str, ep: &EnPrisePiece, view: &PlayerView) -> bool {
    let Some(ShogiMove::Board { from, to, .. }) = parse_usi(usi) else {
        return false;
    };
    if from != ep.square {
        return false;
    }
    let captured_attacker = ep.attackers.iter().copied().find(|&a| a == to);
    !still_attacked_after_move(from, to, &ep.attackers, captured_attacker, view)
}

fn rerank(
    view: &PlayerView,
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

fn rank_by_static(ranking: &[CandidateScore]) -> Vec<&CandidateScore> {
    let mut v: Vec<&CandidateScore> = ranking.iter().collect();
    v.sort_by(|a, b| {
        b.static_score
            .partial_cmp(&a.static_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    v
}

fn diff(flee: &CandidateScore, chosen: &CandidateScore) -> ScoreDiff {
    ScoreDiff {
        score: flee.score - chosen.score,
        gain: flee.gain - chosen.gain,
        p_legal: flee.p_legal - chosen.p_legal,
        foul_cost: flee.foul_cost - chosen.foul_cost,
        adjust: flee.adjust - chosen.adjust,
        checker_removal: flee.checker_removal - chosen.checker_removal,
        capture_bet_penalty: flee.capture_bet_penalty - chosen.capture_bet_penalty,
    }
}

fn true_damage(
    ep: &EnPrisePiece,
    pos: &Position,
    move_idx: usize,
    end: &GameEndPayload,
    record_usi: &str,
    bot: Color,
) -> (bool, f64) {
    if is_flee_usi(record_usi, ep.square) {
        return (false, 0.0);
    }
    let Some(bot_mv) = parse_usi(record_usi) else {
        return (false, 0.0);
    };
    let mut after_bot = pos.clone();
    if !after_bot.is_legal(&bot_mv) {
        return (false, 0.0);
    }
    after_bot.play_unchecked(&bot_mv);
    if after_bot.piece_at(ep.square).is_none_or(|p| p.color != bot) {
        return (false, 0.0);
    }
    let Some(opp_rec) = end.moves.get(move_idx + 1) else {
        return (false, 0.0);
    };
    if opp_rec.by_color == bot {
        return (false, 0.0);
    }
    let Some(opp_mv) = parse_usi(&opp_rec.usi) else {
        return (false, 0.0);
    };
    let captured = match opp_mv {
        ShogiMove::Board { to, .. } => to == ep.square && after_bot.piece_at(to).is_some(),
        ShogiMove::Drop { .. } => false,
    };
    if !captured {
        return (false, 0.0);
    }
    let lost_value = after_bot
        .piece_at(ep.square)
        .map(|p| piece_value(p.role))
        .unwrap_or_else(|| piece_value(ep.role));
    let mut after_opp = after_bot.clone();
    after_opp.play_unchecked(&opp_mv);
    let recaptured_value = end
        .moves
        .get(move_idx + 2)
        .filter(|m| m.by_color == bot)
        .and_then(|m| parse_usi(&m.usi))
        .and_then(|mv| match mv {
            ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } if to == ep.square => {
                after_opp.piece_at(to).map(|p| piece_value(p.role))
            }
            _ => None,
        })
        .unwrap_or(0.0);
    (true, lost_value - recaptured_value)
}

fn is_flee_usi(usi: &str, from: Coord) -> bool {
    matches!(parse_usi(usi), Some(ShogiMove::Board { from: f, .. }) if f == from)
}

fn measure_piece(
    ep: &EnPrisePiece,
    ranking: &[CandidateScore],
    chosen: &str,
    record_usi: &str,
    view: &PlayerView,
    damage: (bool, f64),
) -> Option<Measurement> {
    let static_ranking = rank_by_static(ranking);
    let mut static_best: Option<(&CandidateScore, usize)> = None;
    for (i, c) in static_ranking.iter().enumerate() {
        if is_flee_move(&c.usi, ep, view) {
            static_best = Some((c, i + 1));
            break;
        }
    }
    let mut final_best: Option<(&CandidateScore, usize)> = None;
    for (i, c) in ranking.iter().enumerate() {
        if is_flee_move(&c.usi, ep, view) {
            final_best = Some((c, i + 1));
            break;
        }
    }
    let (static_best, static_rank) = static_best?;
    let (final_best, final_rank) = final_best?;
    let chosen_score = ranking.iter().find(|c| c.usi == chosen)?;
    Some(Measurement {
        static_rank,
        final_rank,
        depth2_in: static_best.depth2 || final_best.depth2,
        chosen_flee: is_flee_move(chosen, ep, view),
        record_flee: is_flee_move(record_usi, ep, view),
        taken_next: damage.0,
        exchange_loss: damage.1,
        diff: diff(final_best, chosen_score),
    })
}

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

fn probe_game(
    file_id: u64,
    rec: &GameRecord,
    file_seed: u64,
    summaries: &mut HashMap<Option<Role>, Summary>,
) -> bool {
    let bot = rec.bot_color;
    let mut pos = Position::initial();
    let mut logs = [ObservationLog::default(), ObservationLog::default()];
    let mut fouls = [0u32; 2];
    let mut foul_idx = 0usize;
    let mut fouls_sorted = rec.end.foul_attempts.clone();
    fouls_sorted.sort_by_key(|f| f.move_number);

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

        if side == bot {
            let view = make_view(&pos, bot, &fouls);
            let log = &logs[side_idx(bot)];
            let eps = en_prise_pieces(&view, log);
            if !eps.is_empty() {
                let tried: HashSet<String> = fouls_sorted
                    .iter()
                    .filter(|f| f.move_number == pos.move_number() && f.by_color == bot)
                    .map(|f| f.usi.clone())
                    .collect();
                let seed = mix(file_seed ^ u64::from(pos.move_number()));
                if let Some((chosen, ranking)) = rerank(&view, log, &tried, seed) {
                    for ep in eps {
                        let damage = true_damage(&ep, &pos, move_idx, &rec.end, &m.usi, bot);
                        if let Some(measurement) =
                            measure_piece(&ep, &ranking, &chosen, &m.usi, &view, damage)
                        {
                            let key = (file_id, pos.move_number());
                            summaries
                                .entry(None)
                                .or_default()
                                .add_case(key, &measurement);
                            summaries
                                .entry(Some(ep.role))
                                .or_default()
                                .add_case(key, &measurement);
                        }
                    }
                }
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
    }
    true
}

fn stable_hash(s: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

fn merge_summaries(dst: &mut HashMap<Option<Role>, Summary>, src: HashMap<Option<Role>, Summary>) {
    for (role, summary) in src {
        dst.entry(role).or_default().merge(summary);
    }
}

#[derive(Debug)]
struct FileResult {
    path: String,
    ok: bool,
    elapsed_secs: f64,
    summaries: HashMap<Option<Role>, Summary>,
}

fn process_file(path: String, seed_base: u64) -> FileResult {
    let start = Instant::now();
    let file_hash = stable_hash(&path);
    let file_seed = mix(seed_base ^ file_hash);
    let mut summaries = HashMap::new();
    let ok = match load(&path) {
        Some(rec) => probe_game(file_hash, &rec, file_seed, &mut summaries),
        None => {
            eprintln!("読めませんでした（終局まで到達していない記録？）: {path}");
            false
        }
    };
    FileResult {
        path,
        ok,
        elapsed_secs: start.elapsed().as_secs_f64(),
        summaries,
    }
}

fn role_label(role: Option<Role>) -> &'static str {
    match role {
        None => "全体",
        Some(Role::Pawn) => "歩",
        Some(Role::Lance) => "香",
        Some(Role::Knight) => "桂",
        Some(Role::Silver) => "銀",
        Some(Role::Gold) => "金",
        Some(Role::Bishop) => "角",
        Some(Role::Rook) => "飛",
        Some(Role::Tokin) => "と",
        Some(Role::Promotedlance) => "成香",
        Some(Role::Promotedknight) => "成桂",
        Some(Role::Promotedsilver) => "成銀",
        Some(Role::Horse) => "馬",
        Some(Role::Dragon) => "竜",
        Some(Role::King) => "玉",
    }
}

fn pct(n: u64, d: u64) -> f64 {
    if d == 0 {
        0.0
    } else {
        100.0 * n as f64 / d as f64
    }
}

fn avg(x: f64, n: u64) -> f64 {
    if n == 0 {
        0.0
    } else {
        x / n as f64
    }
}

fn print_summary(games: u64, summaries: &HashMap<Option<Role>, Summary>) {
    println!("\n=== 自駒 en-prise 逃げ計測（{games}局）===");
    println!(
        "bot視点判定: 相手が自駒を取った観測マスから、駒種不明の敵駒が自駒を攻撃しうるなら当たり。真実盤面は判定に未使用。"
    );
    println!("差分は「最終スコア最良の逃げ手 - estimator再実行の選択手」。正なら逃げ手が上回る。");
    println!(
        "{:<6} {:>7} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>9} {:>9}",
        "駒種",
        "局面",
        "件数",
        "静的rank",
        "2手候補",
        "最終rank",
        "再実行逃げ",
        "記録逃げ",
        "次手被取",
        "交換損",
        "Δscore",
        "Δgain",
    );
    let mut keys: Vec<_> = summaries.keys().copied().collect();
    keys.sort_by_key(|k| (k.is_some(), *k));
    for key in keys {
        let s = &summaries[&key];
        println!(
            "{:<6} {:>7} {:>7} {:>8.2} {:>7.1}% {:>8.2} {:>7.1}% {:>7.1}% {:>7.1}% {:>9.2} {:>9.3} {:>9.3}",
            role_label(key),
            s.positions.len(),
            s.cases,
            avg(s.static_rank_sum, s.cases),
            pct(s.depth2_in, s.cases),
            avg(s.final_rank_sum, s.cases),
            pct(s.chosen_flee, s.cases),
            pct(s.record_flee, s.cases),
            pct(s.actually_taken_next, s.cases - s.record_flee),
            avg(s.exchange_loss_sum, s.cases),
            avg(s.diff_score_sum, s.cases),
            avg(s.diff_gain_sum, s.cases),
        );
    }
    if let Some(s) = summaries.get(&None) {
        println!("\n=== 差分内訳（全体平均、逃げ手 - 選択手）===");
        println!("Δscore={:+.3}", avg(s.diff_score_sum, s.cases));
        println!("Δgain={:+.3}", avg(s.diff_gain_sum, s.cases));
        println!("Δp_legal={:+.3}", avg(s.diff_p_legal_sum, s.cases));
        println!("Δfoul_cost={:+.3}", avg(s.diff_foul_cost_sum, s.cases));
        println!("Δadjust={:+.3}", avg(s.diff_adjust_sum, s.cases));
        println!(
            "Δchecker_removal={:+.3}",
            avg(s.diff_checker_removal_sum, s.cases)
        );
        println!(
            "Δcapture_bet_penalty={:+.3}",
            avg(s.diff_capture_bet_penalty_sum, s.cases)
        );
    }
}

fn default_jobs() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(2))
        .unwrap_or(1)
        .max(1)
}

fn parse_args() -> (u64, usize, Vec<String>) {
    let mut seed = 0u64;
    let mut jobs = default_jobs();
    let mut paths = vec![];
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--seed" {
            seed = args
                .next()
                .expect("--seed には数値が必要です")
                .parse()
                .expect("--seed を数値として読めません");
        } else if arg == "--jobs" {
            jobs = args
                .next()
                .expect("--jobs には数値が必要です")
                .parse::<usize>()
                .expect("--jobs を数値として読めません")
                .max(1);
        } else {
            paths.push(arg);
        }
    }
    (seed, jobs, paths)
}

fn main() {
    let (seed, jobs, paths) = parse_args();
    if paths.is_empty() {
        eprintln!("使い方: flee_probe [--seed N] [--jobs N] <records/*.jsonl>");
        std::process::exit(1);
    }

    let total = paths.len();
    let jobs = jobs.min(total.max(1));
    eprintln!("flee_probe: {total}ファイル / jobs={jobs} / seed={seed}");

    let mut summaries: HashMap<Option<Role>, Summary> = HashMap::new();
    let mut games = 0u64;
    let started = Instant::now();
    let (tx, rx) = mpsc::channel();
    std::thread::scope(|scope| {
        for worker in 0..jobs {
            let tx = tx.clone();
            let paths = &paths;
            scope.spawn(move || {
                let mut i = worker;
                while i < paths.len() {
                    let result = process_file(paths[i].clone(), seed);
                    tx.send(result).expect("進捗channel送信失敗");
                    i += jobs;
                }
            });
        }
        drop(tx);
        for done in 1..=total {
            let result = rx.recv().expect("進捗channel受信失敗");
            let cases = result.summaries.get(&None).map(|s| s.cases).unwrap_or(0);
            eprintln!(
                "[{done}/{total}] {} {:.1}s {} cases={cases}",
                if result.ok { "完了" } else { "スキップ" },
                result.elapsed_secs,
                result.path,
            );
            if result.ok {
                games += 1;
                merge_summaries(&mut summaries, result.summaries);
            }
        }
    });
    eprintln!(
        "flee_probe: 処理完了 {:.1}s / 有効{games}/{total}ファイル",
        started.elapsed().as_secs_f64()
    );
    print_summary(games, &summaries);
}
