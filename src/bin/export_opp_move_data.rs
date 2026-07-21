//! NN学習データ書き出し: 対局記録（records/*.jsonl、人間の実戦棋譜）から
//! 「同一観測クラスの候補手集合 + 実際に選ばれた手」をCSVで出力する。
//!
//! 特徴量の定義は`src/opp_move_features.rs`に一本化する（`bin/fit_opp`の
//! 線形フィットと同じ特徴量・同じ抽出ロジックを使う。学習/推論のズレ防止）。
//!
//! 1行=1候補手。同じ`(game_id, decision_id)`の複数行が1つの決定点＝候補集合
//! （softmax条件付き学習の1グループに対応）。`game_id`は対局単位の
//! train/val分割用（nn-value-phase1のリーク教訓を踏襲）。
//!
//! 使い方: cargo run --release --bin export_opp_move_data -- records/*.jsonl > data.csv

use std::collections::HashSet;

use tsuitate_bot::opp_move_features::{FEATURE_NAMES, home_squares, opp_move_features, to_square};
use tsuitate_bot::protocol::{Color, GameEndPayload};
use tsuitate_bot::shogi::{Position, ShogiMove, parse_usi};

fn load(path: &str) -> Option<(Color, GameEndPayload)> {
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
    Some((bot_color?, end?))
}

/// 1局ぶんの決定点をCSV行としてbufへ積む。パース失敗があれば全体を破棄する
/// （export_value_data.rsと同じ理由: 真実が壊れている局の一部だけが
/// 紛れ込むのを防ぐ）
fn export_game(game_id: u64, bot: Color, end: &GameEndPayload, buf: &mut Vec<String>) -> bool {
    let human = bot.other();
    let mut pos = Position::initial();
    let mut human_lost_at: HashSet<_> = HashSet::new();
    let mut bot_touched: HashSet<_> = HashSet::new();
    let mut decision_id = 0u64;

    for m in &end.moves {
        let Some(mv) = parse_usi(&m.usi) else {
            return false;
        };
        if m.by_color == human && pos.turn() == human {
            let homes = home_squares(&pos, bot, &bot_touched);
            // この手番で人間が最終的な着手に至るまでに試みた反則の回数。
            // 反則の中身は「ついたて」の公平性上どちらのプレイヤーにも
            // 相手には明かされないが、回数（Observation::OpponentFoulの
            // count）は実戦でもリアルタイムに観測できる
            let foul_count_this_turn = end
                .foul_attempts
                .iter()
                .filter(|f| f.by_color == human && f.move_number == pos.move_number())
                .count() as u32;
            let chosen_to = to_square(&mv);
            let chosen_capture = pos
                .piece_at(chosen_to)
                .filter(|p| p.color == bot)
                .map(|_| chosen_to);
            let mut chosen_next = pos.clone();
            chosen_next.play_unchecked(&mv);
            let chosen_check = chosen_next.in_check(bot);

            let mut rows: Vec<String> = Vec::new();
            let mut saw_chosen = false;
            for lm in pos.legal_moves() {
                let to = to_square(&lm);
                let capture = pos.piece_at(to).filter(|p| p.color == bot).map(|_| to);
                if capture != chosen_capture {
                    continue;
                }
                let mut next = pos.clone();
                next.play_unchecked(&lm);
                if next.in_check(bot) != chosen_check {
                    continue;
                }
                let is_chosen = lm == mv;
                saw_chosen |= is_chosen;
                let f = opp_move_features(
                    &pos,
                    &next,
                    &lm,
                    human,
                    &human_lost_at,
                    &homes,
                    foul_count_this_turn,
                );
                let feats: Vec<String> = f.iter().map(|x| x.to_string()).collect();
                rows.push(format!(
                    "{game_id},{decision_id},{},{}",
                    feats.join(","),
                    is_chosen as u8
                ));
            }
            if saw_chosen && rows.len() >= 2 {
                buf.extend(rows);
                decision_id += 1;
            }
        }

        let to = to_square(&mv);
        let captured_color = pos.piece_at(to).map(|p| p.color);
        if m.by_color == bot && captured_color == Some(human) {
            human_lost_at.insert(to);
        }
        if m.by_color == bot {
            if let ShogiMove::Board { from, .. } = mv {
                bot_touched.insert(from);
            }
            bot_touched.insert(to);
        }
        pos.play_unchecked(&mv);
    }
    true
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("使い方: export_opp_move_data <records/*.jsonl...> > data.csv");
        std::process::exit(1);
    }

    println!("game_id,decision_id,{},chosen", FEATURE_NAMES.join(","));

    let mut games = 0u64;
    let mut rows = 0u64;
    for path in &paths {
        let Some((bot, end)) = load(path) else {
            eprintln!("読めませんでした（終局まで到達していない記録？）: {path}");
            continue;
        };
        let mut buf: Vec<String> = vec![];
        if export_game(games, bot, &end, &mut buf) {
            for line in &buf {
                println!("{line}");
            }
            rows += buf.len() as u64;
            games += 1;
        } else {
            eprintln!("  棋譜の手をパースできませんでした: {path}。この局はスキップ");
        }
    }
    eprintln!(
        "書き出し完了: {games}局 / {rows}行（候補手単位） / 特徴量{}次元",
        FEATURE_NAMES.len()
    );
}
