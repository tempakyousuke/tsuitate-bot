//! 棋譜の連続した決定点の候補ランキングをまとめて出力する診断ツール。
//!
//! docs/quest_20260731.md のような「終局まで bot の候補手を評価していく」レビューの
//! スケルトン生成用（scenario-gui の「ランキング」の一括版）。同一手番側は
//! prewarm 済み戦略を継ぎ足して共有する（choice_trials_batch と同じ規約なので
//! 結果は決定点ごとに作り直した場合と同じ意味論）。
//!
//! usage:
//!   cargo run --release --bin rank_dump -- <kifへのパス> <開始手目> <終了手目> [top_n=7] [seed=0]
//!
//! 手目は「その手を考えさせる」番号（= 手目-1 手まで再生）。出力は markdown の
//! 「## N手目」スケルトン（stdout）。進捗は stderr。

use std::collections::HashSet;

use tsuitate_bot::kifu::parse_kif;
use tsuitate_bot::observation::{Observation, ObservationLog};
use tsuitate_bot::protocol::Role;
use tsuitate_bot::scenario_core::{clone_log, make_view, replay, resolve_foul, side_idx};
use tsuitate_bot::shogi::{Position, ShogiMove, parse_usi};
use tsuitate_bot::strategy::{self, Strategy};

fn kanji_rank(r: i8) -> char {
    "一二三四五六七八九".chars().nth(r as usize - 1).unwrap()
}

fn role_name(role: Role) -> &'static str {
    match role {
        Role::Pawn => "歩",
        Role::Lance => "香",
        Role::Knight => "桂",
        Role::Silver => "銀",
        Role::Gold => "金",
        Role::Bishop => "角",
        Role::Rook => "飛",
        Role::King => "玉",
        Role::Tokin => "と",
        Role::Promotedlance => "成香",
        Role::Promotedknight => "成桂",
        Role::Promotedsilver => "成銀",
        Role::Horse => "馬",
        Role::Dragon => "龍",
    }
}

/// doc の表記に合わせた日本語表記（例: 4七歩成 / 2二歩打 / 3一と）。
/// 移動元の区別は付けない（併記する USI が一意な識別子）
fn jp_move(pos: &Position, usi: &str) -> String {
    let Some(mv) = parse_usi(usi) else {
        return usi.to_string();
    };
    match mv {
        ShogiMove::Board { from, to, promote } => {
            let name = pos.piece_at(from).map(|p| role_name(p.role)).unwrap_or("?");
            format!(
                "{}{}{}{}",
                to.file,
                kanji_rank(to.rank),
                name,
                if promote { "成" } else { "" }
            )
        }
        ShogiMove::Drop { role, to } => {
            format!("{}{}{}打", to.file, kanji_rank(to.rank), role_name(role))
        }
    }
}

struct Chain {
    strat: Box<dyn Strategy>,
    running: ObservationLog,
    consumed: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: rank_dump <kifへのパス> <開始手目> <終了手目> [top_n=7] [seed=0]");
        std::process::exit(2);
    }
    let path = &args[0];
    let start: usize = args[1].parse().expect("開始手目を読めません");
    let end: usize = args[2].parse().expect("終了手目を読めません");
    let top_n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(7);
    let seed: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);

    let text = std::fs::read_to_string(path).expect("kif を読めません");
    let kifu = parse_kif(&text).expect("kif をパースできません");
    let last = end.min(kifu.plies.len() + 1);

    let mut chains: [Option<Chain>; 2] = [None, None];
    for m in start..=last {
        let ply = m - 1;
        let rep = replay(&kifu, ply);
        let side = rep.pos.turn();
        let idx = side_idx(side);
        let chain = chains[idx].get_or_insert_with(|| Chain {
            strat: strategy::make_seeded("estimator", seed).expect("estimator を作れません"),
            running: ObservationLog::default(),
            consumed: 0,
        });
        let view = make_view(&rep.pos, side, &rep.fouls);
        let log = &rep.logs[idx];
        let events = log.events();
        while chain.consumed < events.len() {
            let e = &events[chain.consumed];
            if matches!(e, Observation::MyMove { .. } | Observation::MyFoul { .. }) {
                chain.strat.prewarm(&view, &chain.running);
            }
            chain.running.record(e.clone());
            chain.consumed += 1;
        }
        let show_scores = std::env::var("RANK_DUMP_SCORES").is_ok_and(|v| v == "1");
        // 実戦でこの手の前に試みられた反則（*illegal 行）。棋譜の最終決定
        // （反則負けなど、plies に手が無い決定点）は trailing_fouls を使う
        let real_fouls = if ply < kifu.plies.len() {
            kifu.plies[ply].fouls.clone()
        } else {
            kifu.trailing_fouls.clone()
        };
        // RANK_DUMP_FOULS_ONLY=1: 反則を挟む決定点の「反則後」サブ状態だけを
        // 出力する（反則前のリストは既に doc にある前提。チェーンは全手で進める）
        let fouls_only = std::env::var("RANK_DUMP_FOULS_ONLY").is_ok_and(|v| v == "1");

        let print_ranking =
            |snap: &dyn Strategy, chosen: &Option<String>, pos: &Position, top_n: usize| {
                match snap.last_ranking() {
                    Some(ranking) => {
                        for c in ranking.iter().take(top_n) {
                            if show_scores {
                                println!(
                                    "{}({}) score={:.3} gain={:.3} static_gain={:.3} capture={:.3} p_legal={:.2} adjust={:+.3} depth2={}",
                                    jp_move(pos, &c.usi),
                                    c.usi,
                                    c.score,
                                    c.gain,
                                    c.static_gain,
                                    c.capture_value,
                                    c.p_legal,
                                    c.adjust,
                                    c.depth2
                                );
                            } else {
                                println!("{}({})", jp_move(pos, &c.usi), c.usi);
                            }
                        }
                    }
                    None => println!(
                        "（ランキングなし: 定跡または候補ゼロ。choose={}）",
                        chosen.as_deref().unwrap_or("投了")
                    ),
                }
            };

        if !fouls_only {
            let mut snap = chain
                .strat
                .clone_boxed()
                .expect("estimator は clone_boxed 対応のはず");
            let chosen = snap.choose(&view, log, &HashSet::new());
            println!();
            println!("## {m}手目");
            print_ranking(&*snap, &chosen, &rep.pos, top_n);
        }

        // 反則を挟む決定点: 反則観測を1つずつ足した「反則後」の状態を、
        // choice_trial_body と同じ規約（MyFoul 記録・反則カウント・foul_tried）で
        // 再現してランキングを出す
        if !real_fouls.is_empty() {
            let mut log2 = clone_log(log);
            let mut fouls_arr = rep.fouls;
            let mut foul_tried: HashSet<String> = HashSet::new();
            for raw in &real_fouls {
                let usi = resolve_foul(&rep.pos, side, raw);
                fouls_arr[idx] += 1;
                log2.record(Observation::MyFoul {
                    move_number: rep.pos.move_number(),
                    usi: usi.clone(),
                });
                foul_tried.insert(usi.clone());
                let view2 = make_view(&rep.pos, side, &fouls_arr);
                let mut snap = chain
                    .strat
                    .clone_boxed()
                    .expect("estimator は clone_boxed 対応のはず");
                let chosen = snap.choose(&view2, &log2, &foul_tried);
                println!();
                println!("### {m}手目（{}({})の反則後）", jp_move(&rep.pos, &usi), usi);
                print_ranking(&*snap, &chosen, &rep.pos, top_n);
            }
        }
        eprintln!("[{m}/{last}] 完了");
    }
}
