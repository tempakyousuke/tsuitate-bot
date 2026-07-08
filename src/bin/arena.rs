//! 戦略同士をローカルで対戦させて勝率を測るアリーナ。
//!
//! サーバー（judge.ts / game-room.ts）と同じ裁定を再現する:
//! - 反則（フル盤面で非合法な手）は手番を変えずカウント。MAX_FOULS=10 で反則負け
//! - フィッシャー時計 300秒+3秒 をシミュレート。choose() の壁時計を消費し、
//!   時間切れは負け。加算は受理された手の後のみ（反則では加算しない）
//! - 各戦略に見えるのは自分の PlayerView 相当と観測イベントのみ（公平性の担保）
//! - 王手宣言は両者に、取った駒種は指した側に、取られたマスは相手側に通知
//! - 詰み・ステイルメイト・投了（choose が None）・手数上限で終局
//!
//! 使い方:
//!   cargo run --release --bin arena -- [対局数] [戦略A] [戦略B]
//!   cargo run --release --bin arena -- [対局数] [候補] [基準1] [基準2] ...
//!
//! 基準を複数並べるとガントレット: 候補が各基準と [対局数] ずつ対戦する。
//! 新戦略は直近の凍結版だけでなく過去の凍結版すべてに勝ち越すこと
//! （v2 に勝つが v1 に負ける、という非推移性の検出。src/frozen/ 参照）。

use std::collections::HashSet;
use std::time::Instant;

use tsuitate_bot::observation::{Observation, ObservationLog};
use tsuitate_bot::protocol::{
    ClockState, Color, FoulCounts, GameStatus, PlayerView,
};
use tsuitate_bot::shogi::{Outcome, Position, parse_usi, unpromote_role};
use tsuitate_bot::strategy::{self, Strategy};

const MAX_FOULS: u32 = 10;
/// これを超えたら引き分け扱い（千日手検出は未実装のため）
const MAX_PLIES: u32 = 400;
/// フィッシャー時計（サイト仕様: 300秒+3秒）
const FISCHER_INITIAL_MS: i64 = 300_000;
const FISCHER_INCREMENT_MS: i64 = 3_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GameResult {
    Win(Color),
    Draw,
}

#[derive(Debug, Default)]
struct MatchStats {
    games: u32,
    wins_a: u32,
    wins_b: u32,
    draws: u32,
    /// 終局理由ごとの回数（A勝ち, B勝ち問わず）
    checkmate: u32,
    stalemate: u32,
    foul_limit: u32,
    resign: u32,
    timeout: u32,
    max_plies: u32,
    total_plies: u64,
    fouls_a: u64,
    fouls_b: u64,
    /// うち王手を受けている局面での反則（王手ソルバーの効果測定用）
    fouls_in_check_a: u64,
    fouls_in_check_b: u64,
    /// 1手ごとの思考時間（マイクロ秒）。時間切れ検証と改善の副作用チェック用
    think_us_a: Vec<u64>,
    think_us_b: Vec<u64>,
}

impl MatchStats {
    /// 引き分けを除いた A の勝率と、二項近似の95%信頼区間の半幅
    fn rate_and_ci(&self) -> (f64, f64) {
        let decisive = self.wins_a + self.wins_b;
        if decisive == 0 {
            return (0.5, 0.0);
        }
        let rate = self.wins_a as f64 / decisive as f64;
        let se = (rate * (1.0 - rate) / decisive as f64).sqrt();
        (rate, se * 1.96)
    }
}

struct PlayerState {
    strategy: Box<dyn Strategy>,
    log: ObservationLog,
    fouls: u32,
    fouls_in_check: u32,
    foul_tried: HashSet<String>,
    clock_ms: i64,
    think_us: Vec<u64>,
}

fn make_view(
    pos: &Position,
    color: Color,
    fouls: &[u32; 2],
    clocks_ms: &[i64; 2],
    game_no: u32,
) -> PlayerView {
    let idx = |c: Color| if c == Color::Sente { 0 } else { 1 };
    PlayerView {
        game_id: format!("arena-{game_no}"),
        your_color: color,
        your_pieces: pos.pieces_of(color),
        your_hand: pos.hand_map(color),
        turn: pos.turn(),
        move_number: pos.move_number(),
        clocks: ClockState {
            sente_ms: clocks_ms[0],
            gote_ms: clocks_ms[1],
            running: Some(pos.turn()),
            server_time: 0,
        },
        fouls: FoulCounts {
            you: fouls[idx(color)],
            opponent: fouls[idx(color.other())],
        },
        you_in_check: pos.in_check(color),
        opponent_in_check: pos.in_check(color.other()),
        status: GameStatus::Playing,
    }
}

/// 1局対戦する。players[0] が先手。
fn play_game(players: &mut [PlayerState; 2], game_no: u32) -> (GameResult, &'static str, u32) {
    let mut pos = Position::initial();
    let idx = |c: Color| if c == Color::Sente { 0usize } else { 1usize };
    let mut plies = 0u32;

    loop {
        if plies >= MAX_PLIES {
            return (GameResult::Draw, "max_plies", plies);
        }
        let side = pos.turn();
        let fouls = [players[0].fouls, players[1].fouls];
        let clocks_ms = [players[0].clock_ms, players[1].clock_ms];
        let view = make_view(&pos, side, &fouls, &clocks_ms, game_no);

        let mover = &mut players[idx(side)];
        let started = Instant::now();
        let choice = mover.strategy.choose(&view, &mover.log, &mover.foul_tried);
        let elapsed = started.elapsed();
        mover.think_us.push(elapsed.as_micros() as u64);
        mover.clock_ms -= elapsed.as_millis() as i64;
        if mover.clock_ms <= 0 {
            return (GameResult::Win(side.other()), "timeout", plies);
        }
        let Some(usi) = choice else {
            return (GameResult::Win(side.other()), "resign", plies);
        };

        let legal = parse_usi(&usi).is_some_and(|mv| pos.is_legal(&mv));
        if !legal {
            // 反則: 手番は変わらない（judge.ts と同じ）。フィッシャー加算もしない
            let in_check = pos.in_check(side);
            let mover = &mut players[idx(side)];
            mover.fouls += 1;
            if in_check {
                mover.fouls_in_check += 1;
            }
            mover.foul_tried.insert(usi.clone());
            let foul_count = mover.fouls;
            mover.log.record(Observation::MyFoul {
                move_number: pos.move_number(),
                usi,
            });
            players[idx(side.other())]
                .log
                .record(Observation::OpponentFoul { count: foul_count });
            if foul_count >= MAX_FOULS {
                return (GameResult::Win(side.other()), "foul_limit", plies);
            }
            continue;
        }

        let mv = parse_usi(&usi).unwrap();
        let captured = pos.play_unchecked(&mv);
        plies += 1;
        players[idx(side)].foul_tried.clear();
        players[idx(side)].clock_ms += FISCHER_INCREMENT_MS;

        // 通知（game-room.ts と同じ内容・同じ moveNumber 規約 = 適用後の値）
        let move_number = pos.move_number();
        let captured_square = captured.map(|_| match mv {
            tsuitate_bot::shogi::ShogiMove::Board { to, .. } => {
                tsuitate_bot::board::make_usi_square(to)
            }
            tsuitate_bot::shogi::ShogiMove::Drop { .. } => unreachable!("打ちでは駒を取れない"),
        });
        players[idx(side)].log.record(Observation::MyMove {
            move_number,
            usi,
            captured: captured.map(unpromote_role),
        });
        players[idx(side.other())].log.record(Observation::OpponentMoved {
            move_number,
            captured_my_piece_at: captured_square,
        });
        if pos.in_check(pos.turn()) {
            let in_check = pos.turn();
            for p in players.iter_mut() {
                p.log.record(Observation::Check { in_check });
            }
        }

        match pos.outcome() {
            Some(Outcome::Checkmate { winner }) => {
                return (GameResult::Win(winner), "checkmate", plies);
            }
            Some(Outcome::Stalemate { winner }) => {
                return (GameResult::Win(winner), "stalemate", plies);
            }
            None => {}
        }
    }
}

/// 1局ぶんの結果を stats に集計する
fn record_game(
    stats: &mut MatchStats,
    a_is_sente: bool,
    players: [PlayerState; 2],
    result: GameResult,
    reason: &str,
    plies: u32,
) {
    let [sente, gote] = players;
    let (pa, pb) = if a_is_sente {
        (sente, gote)
    } else {
        (gote, sente)
    };
    stats.fouls_a += pa.fouls as u64;
    stats.fouls_b += pb.fouls as u64;
    stats.fouls_in_check_a += pa.fouls_in_check as u64;
    stats.fouls_in_check_b += pb.fouls_in_check as u64;
    stats.think_us_a.extend(pa.think_us);
    stats.think_us_b.extend(pb.think_us);
    stats.total_plies += plies as u64;
    match reason {
        "checkmate" => stats.checkmate += 1,
        "stalemate" => stats.stalemate += 1,
        "foul_limit" => stats.foul_limit += 1,
        "resign" => stats.resign += 1,
        "timeout" => stats.timeout += 1,
        _ => stats.max_plies += 1,
    }
    match result {
        GameResult::Draw => stats.draws += 1,
        GameResult::Win(winner) => {
            let a_won = (winner == Color::Sente) == a_is_sente;
            if a_won {
                stats.wins_a += 1;
            } else {
                stats.wins_b += 1;
            }
        }
    }
}

/// 対局の並列数。既定はコア数-2（推定器の時間予算・思考時間計測が
/// コア競合で歪みすぎないよう全コアは使わない）。ARENA_THREADS で上書き可
fn thread_count() -> usize {
    if let Some(n) = std::env::var("ARENA_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        return n.max(1);
    }
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(2))
        .unwrap_or(1)
        .max(1)
}

/// A と B を games 局（先後交代で）対戦させる。
/// 対局同士は独立なのでスレッドに分散する。game_no の偶奇で先後を決めるため
/// ラウンドロビンに割っても先後バランスは保たれる。
/// 注意: 並列実行中の思考時間はコア競合ぶん逐次実行より長めに出る
fn run_match(games: u32, name_a: &str, name_b: &str) -> MatchStats {
    let threads = thread_count().min(games.max(1) as usize);
    let mut stats = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    let mut local = MatchStats::default();
                    let mut game_no = t as u32;
                    while game_no < games {
                        // 偶数局は A が先手
                        let a_is_sente = game_no % 2 == 0;
                        let new_player = |name: &str| PlayerState {
                            strategy: strategy::make(name).unwrap(),
                            log: ObservationLog::default(),
                            fouls: 0,
                            fouls_in_check: 0,
                            foul_tried: HashSet::new(),
                            clock_ms: FISCHER_INITIAL_MS,
                            think_us: Vec::new(),
                        };
                        let (sente_name, gote_name) = if a_is_sente {
                            (name_a, name_b)
                        } else {
                            (name_b, name_a)
                        };
                        let mut players = [new_player(sente_name), new_player(gote_name)];
                        let (result, reason, plies) = play_game(&mut players, game_no);
                        record_game(&mut local, a_is_sente, players, result, reason, plies);
                        game_no += threads as u32;
                    }
                    local
                })
            })
            .collect();
        let mut merged = MatchStats::default();
        for h in handles {
            let local = h.join().expect("対局スレッドが panic した");
            merged.wins_a += local.wins_a;
            merged.wins_b += local.wins_b;
            merged.draws += local.draws;
            merged.checkmate += local.checkmate;
            merged.stalemate += local.stalemate;
            merged.foul_limit += local.foul_limit;
            merged.resign += local.resign;
            merged.timeout += local.timeout;
            merged.max_plies += local.max_plies;
            merged.total_plies += local.total_plies;
            merged.fouls_a += local.fouls_a;
            merged.fouls_b += local.fouls_b;
            merged.fouls_in_check_a += local.fouls_in_check_a;
            merged.fouls_in_check_b += local.fouls_in_check_b;
            merged.think_us_a.extend(local.think_us_a);
            merged.think_us_b.extend(local.think_us_b);
        }
        merged
    });
    stats.games = games;
    stats
}

/// 思考時間の要約（平均 / p99 / 最大、ミリ秒）
fn think_summary(think_us: &[u64]) -> String {
    if think_us.is_empty() {
        return "-".into();
    }
    let mut sorted = think_us.to_vec();
    sorted.sort_unstable();
    let mean = sorted.iter().sum::<u64>() as f64 / sorted.len() as f64 / 1000.0;
    let p99 = sorted[(sorted.len() * 99 / 100).min(sorted.len() - 1)] as f64 / 1000.0;
    let max = *sorted.last().unwrap() as f64 / 1000.0;
    format!("平均 {mean:.1}ms / p99 {p99:.1}ms / 最大 {max:.1}ms")
}

fn print_match(stats: &MatchStats, name_a: &str, name_b: &str) {
    let (rate, ci) = stats.rate_and_ci();
    println!(
        "A={name_a}: {}勝 / B={name_b}: {}勝 / 引き分け {}",
        stats.wins_a, stats.wins_b, stats.draws
    );
    println!("Aの勝率（引き分け除く）: {:.1}% ± {:.1}%", rate * 100.0, ci * 100.0);
    println!(
        "終局理由: 詰み {} / ステイルメイト {} / 反則負け {} / 投了 {} / 時間切れ {} / 手数上限 {}",
        stats.checkmate, stats.stalemate, stats.foul_limit, stats.resign, stats.timeout,
        stats.max_plies
    );
    println!(
        "平均手数 {:.1} / 平均反則 A {:.2}（うち王手中 {:.2}） B {:.2}（うち王手中 {:.2}）",
        stats.total_plies as f64 / stats.games.max(1) as f64,
        stats.fouls_a as f64 / stats.games.max(1) as f64,
        stats.fouls_in_check_a as f64 / stats.games.max(1) as f64,
        stats.fouls_b as f64 / stats.games.max(1) as f64,
        stats.fouls_in_check_b as f64 / stats.games.max(1) as f64
    );
    println!("思考時間 A: {}", think_summary(&stats.think_us_a));
    println!("思考時間 B: {}", think_summary(&stats.think_us_b));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let games: u32 = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(100);
    let candidate = args.get(2).cloned().unwrap_or_else(|| "heuristic".into());
    let opponents: Vec<String> = if args.len() > 3 {
        args[3..].to_vec()
    } else {
        vec!["heuristic".into()]
    };
    for name in std::iter::once(&candidate).chain(&opponents) {
        if strategy::make(name).is_none() {
            eprintln!("未知の戦略名です: {name}");
            std::process::exit(1);
        }
    }

    let mut results: Vec<(String, MatchStats)> = vec![];
    for opp in &opponents {
        println!(
            "=== アリーナ: {candidate} (A) vs {opp} (B), {games}局（先後交代・フィッシャー{}秒+{}秒・並列{}） ===",
            FISCHER_INITIAL_MS / 1000,
            FISCHER_INCREMENT_MS / 1000,
            thread_count().min(games.max(1) as usize)
        );
        let stats = run_match(games, &candidate, opp);
        print_match(&stats, &candidate, opp);
        println!();
        results.push((opp.clone(), stats));
    }

    // ガントレット時のみ総合サマリ（非推移性の一覧確認用）
    if results.len() > 1 {
        println!("=== 総合: {candidate} の対戦成績 ===");
        let mut total = MatchStats::default();
        for (opp, stats) in &results {
            let (rate, ci) = stats.rate_and_ci();
            println!(
                "vs {opp}: {:.1}% ± {:.1}% ({}-{}-{})",
                rate * 100.0,
                ci * 100.0,
                stats.wins_a,
                stats.wins_b,
                stats.draws
            );
            total.wins_a += stats.wins_a;
            total.wins_b += stats.wins_b;
            total.draws += stats.draws;
        }
        let (rate, ci) = total.rate_and_ci();
        println!(
            "合計: {:.1}% ± {:.1}% ({}-{}-{})",
            rate * 100.0,
            ci * 100.0,
            total.wins_a,
            total.wins_b,
            total.draws
        );
    }
}
