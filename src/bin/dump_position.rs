//! 一時的な調査用ツール: game.moves の USI 列を再生し、指定手数時点の
//! 盤面・持ち駒を表示する。人間対局の事後分析（局面レビュー）専用。
use std::env;
use std::fs;

use tsuitate_bot::shogi::{Position, parse_usi};

fn role_char(role: tsuitate_bot::protocol::Role) -> &'static str {
    use tsuitate_bot::protocol::Role::*;
    match role {
        Pawn => "P",
        Lance => "L",
        Knight => "N",
        Silver => "S",
        Gold => "G",
        Bishop => "B",
        Rook => "R",
        King => "K",
        Tokin => "+P",
        Promotedlance => "+L",
        Promotedknight => "+N",
        Promotedsilver => "+S",
        Horse => "+B",
        Dragon => "+R",
    }
}

fn print_board(pos: &Position) {
    use tsuitate_bot::board::Coord;
    use tsuitate_bot::protocol::Color;
    for rank in 1..=9 {
        let mut row = String::new();
        for file in (1..=9).rev() {
            let c = Coord { file, rank };
            match pos.piece_at(c) {
                Some(p) => {
                    let s = role_char(p.role);
                    let s = if p.color == Color::Gote {
                        format!("v{s}")
                    } else {
                        s.to_string()
                    };
                    row.push_str(&format!("{s:>4}"));
                }
                None => row.push_str("   ."),
            }
        }
        println!("{rank}: {row}");
    }
    for color in [Color::Sente, Color::Gote] {
        let hand = pos.hand_map(color);
        let mut items: Vec<_> = hand.into_iter().filter(|(_, n)| *n > 0).collect();
        items.sort_by_key(|(r, _)| format!("{r:?}"));
        println!(
            "{color:?} hand: {}",
            items
                .iter()
                .map(|(r, n)| format!("{}x{n}", role_char(*r)))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    if let Some(k) = pos.king_square(Color::Sente) {
        println!("sente king: {}{}", k.file, (b'a' + (k.rank - 1) as u8) as char);
    }
    if let Some(k) = pos.king_square(Color::Gote) {
        println!("gote king: {}{}", k.file, (b'a' + (k.rank - 1) as u8) as char);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = &args[1];
    let upto: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(usize::MAX);
    let content = fs::read_to_string(path).unwrap();
    let v: serde_json::Value = serde_json::from_str(&content).unwrap();
    let moves = v["moves"].as_array().unwrap();
    let mut pos = Position::initial();
    for (i, m) in moves.iter().enumerate() {
        if i >= upto {
            break;
        }
        let usi = m["usi"].as_str().unwrap();
        let mv = parse_usi(usi).unwrap_or_else(|| panic!("bad usi {usi}"));
        pos.play_unchecked(&mv);
        println!("--- after move {} ({}) ---", i + 1, usi);
    }
    print_board(&pos);
}
