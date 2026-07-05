//! 戦略同士をローカルで対戦させて勝率を測るアリーナ。
//!
//! サーバー（judge.ts / game-room.ts）と同じ裁定を再現する:
//! - 反則（フル盤面で非合法な手）は手番を変えずカウント。MAX_FOULS=10 で反則負け
//! - 各戦略に見えるのは自分の PlayerView 相当と観測イベントのみ（公平性の担保）
//! - 王手宣言は両者に、取った駒種は指した側に、取られたマスは相手側に通知
//! - 詰み・ステイルメイト・投了（choose が None）・手数上限で終局
//!
//! 使い方: cargo run --release --bin arena -- [対局数] [戦略A] [戦略B]

use std::collections::HashSet;

use tsuitate_bot::observation::{Observation, ObservationLog};
use tsuitate_bot::protocol::{
    ClockState, Color, FoulCounts, GameStatus, OpponentInfo, PlayerView,
};
use tsuitate_bot::shogi::{Outcome, Position, parse_usi, unpromote_role};
use tsuitate_bot::strategy::{self, Strategy};

const MAX_FOULS: u32 = 10;
/// これを超えたら引き分け扱い（千日手検出は未実装のため）
const MAX_PLIES: u32 = 400;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GameResult {
    Win(Color),
    Draw,
}

#[derive(Debug, Default)]
struct Tally {
    wins_a: u32,
    wins_b: u32,
    draws: u32,
    /// 終局理由ごとの回数（A勝ち, B勝ち問わず）
    checkmate: u32,
    stalemate: u32,
    foul_limit: u32,
    resign: u32,
    max_plies: u32,
    total_plies: u64,
    fouls_a: u64,
    fouls_b: u64,
}

struct PlayerState {
    strategy: Box<dyn Strategy>,
    log: ObservationLog,
    fouls: u32,
    foul_tried: HashSet<String>,
}

fn make_view(pos: &Position, color: Color, fouls: &[u32; 2], game_no: u32) -> PlayerView {
    let idx = |c: Color| if c == Color::Sente { 0 } else { 1 };
    PlayerView {
        game_id: format!("arena-{game_no}"),
        your_color: color,
        opponent: OpponentInfo {
            username: "arena".into(),
            rating: 1500,
            is_bot: true,
        },
        your_pieces: pos.pieces_of(color),
        your_hand: pos.hand_map(color),
        turn: pos.turn(),
        move_number: pos.move_number(),
        clocks: ClockState {
            sente_ms: 300_000,
            gote_ms: 300_000,
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
        let view = make_view(&pos, side, &fouls, game_no);

        let mover = &mut players[idx(side)];
        let Some(usi) = mover.strategy.choose(&view, &mover.log, &mover.foul_tried) else {
            return (GameResult::Win(side.other()), "resign", plies);
        };

        let legal = parse_usi(&usi).is_some_and(|mv| pos.is_legal(&mv));
        if !legal {
            // 反則: 手番は変わらない（judge.ts と同じ）
            let mover = &mut players[idx(side)];
            mover.fouls += 1;
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let games: u32 = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(100);
    let name_a = args.get(2).cloned().unwrap_or_else(|| "heuristic".into());
    let name_b = args.get(3).cloned().unwrap_or_else(|| "heuristic".into());
    for name in [&name_a, &name_b] {
        if strategy::make(name).is_none() {
            eprintln!("未知の戦略名です: {name}");
            std::process::exit(1);
        }
    }

    println!("アリーナ: {name_a} (A) vs {name_b} (B), {games}局（先後交代）");
    let mut tally = Tally::default();

    for game_no in 0..games {
        // 偶数局は A が先手
        let a_is_sente = game_no % 2 == 0;
        let new_player = |name: &str| PlayerState {
            strategy: strategy::make(name).unwrap(),
            log: ObservationLog::default(),
            fouls: 0,
            foul_tried: HashSet::new(),
        };
        let (sente_name, gote_name) = if a_is_sente {
            (&name_a, &name_b)
        } else {
            (&name_b, &name_a)
        };
        let mut players = [new_player(sente_name), new_player(gote_name)];
        let (result, reason, plies) = play_game(&mut players, game_no);

        let fouls_sente = players[0].fouls as u64;
        let fouls_gote = players[1].fouls as u64;
        if a_is_sente {
            tally.fouls_a += fouls_sente;
            tally.fouls_b += fouls_gote;
        } else {
            tally.fouls_a += fouls_gote;
            tally.fouls_b += fouls_sente;
        }
        tally.total_plies += plies as u64;
        match reason {
            "checkmate" => tally.checkmate += 1,
            "stalemate" => tally.stalemate += 1,
            "foul_limit" => tally.foul_limit += 1,
            "resign" => tally.resign += 1,
            _ => tally.max_plies += 1,
        }
        match result {
            GameResult::Draw => tally.draws += 1,
            GameResult::Win(winner) => {
                let a_won = (winner == Color::Sente) == a_is_sente;
                if a_won {
                    tally.wins_a += 1;
                } else {
                    tally.wins_b += 1;
                }
            }
        }
    }

    let decisive = tally.wins_a + tally.wins_b;
    let rate = if decisive > 0 {
        tally.wins_a as f64 / decisive as f64
    } else {
        0.5
    };
    // 二項近似の95%信頼区間
    let se = if decisive > 0 {
        (rate * (1.0 - rate) / decisive as f64).sqrt()
    } else {
        0.0
    };
    println!("A={name_a}: {}勝 / B={name_b}: {}勝 / 引き分け {}", tally.wins_a, tally.wins_b, tally.draws);
    println!(
        "Aの勝率（引き分け除く）: {:.1}% ± {:.1}%",
        rate * 100.0,
        se * 196.0
    );
    println!(
        "終局理由: 詰み {} / ステイルメイト {} / 反則負け {} / 投了 {} / 手数上限 {}",
        tally.checkmate, tally.stalemate, tally.foul_limit, tally.resign, tally.max_plies
    );
    println!(
        "平均手数 {:.1} / 平均反則 A {:.2} B {:.2}",
        tally.total_plies as f64 / games as f64,
        tally.fouls_a as f64 / games as f64,
        tally.fouls_b as f64 / games as f64
    );
}
