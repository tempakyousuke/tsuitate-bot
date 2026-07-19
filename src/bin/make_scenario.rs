//! 一時的な調査用ツール: 対局記録(JSON)からshogi-quest風KIFを生成する。
//! 人間対局の局面をscenarios/*.kifへ落とし込むための変換専用（使い捨て）。
use std::env;
use std::fs;

use tsuitate_bot::board::Coord;
use tsuitate_bot::protocol::{Color, Role};
use tsuitate_bot::shogi::{Position, ShogiMove, parse_usi};

const KANJI_RANKS: [char; 9] = ['一', '二', '三', '四', '五', '六', '七', '八', '九'];

fn role_kanji(role: Role) -> &'static str {
    use Role::*;
    match role {
        Pawn => "歩",
        Lance => "香",
        Knight => "桂",
        Silver => "銀",
        Gold => "金",
        Bishop => "角",
        Rook => "飛",
        King => "玉",
        Tokin => "と",
        Promotedlance => "成香",
        Promotedknight => "成桂",
        Promotedsilver => "成銀",
        Horse => "馬",
        Dragon => "龍",
    }
}

fn role_foulcode(role: Role) -> &'static str {
    use Role::*;
    match role {
        Pawn => "FU",
        Lance => "KY",
        Knight => "KE",
        Silver => "GI",
        Gold => "KI",
        Bishop => "KA",
        Rook => "HI",
        King => "OU",
        Tokin => "TO",
        Promotedlance => "NY",
        Promotedknight => "NK",
        Promotedsilver => "NG",
        Horse => "UM",
        Dragon => "RY",
    }
}

fn unpromote(role: Role) -> Role {
    use Role::*;
    match role {
        Tokin => Pawn,
        Promotedlance => Lance,
        Promotedknight => Knight,
        Promotedsilver => Silver,
        Horse => Bishop,
        Dragon => Rook,
        r => r,
    }
}

fn move_line(no: usize, pos: &Position, usi: &str, prev_to: Option<Coord>) -> String {
    let mv = parse_usi(usi).unwrap_or_else(|| panic!("bad usi {usi}"));
    match mv {
        ShogiMove::Drop { to, role } => {
            format!(
                "{no} {}{}{}打",
                to.file,
                KANJI_RANKS[(to.rank - 1) as usize],
                role_kanji(role)
            )
        }
        ShogiMove::Board { from, to, promote } => {
            let piece = pos.piece_at(from).unwrap_or_else(|| panic!("no piece at from for {usi}"));
            // 成る手は成る前の駒名（例: 角成）、既に成っている駒はそのまま
            let name = if promote { role_kanji(unpromote(piece.role)) } else { role_kanji(piece.role) };
            let dest = if Some(to) == prev_to {
                "同　".to_string()
            } else {
                format!("{}{}", to.file, KANJI_RANKS[(to.rank - 1) as usize])
            };
            let suffix = if promote { "成" } else { "" };
            format!("{no} {dest}{name}{suffix}({}{})", from.file, from.rank)
        }
    }
}

fn foul_code(pos: &Position, usi: &str) -> String {
    let mv = parse_usi(usi).unwrap_or_else(|| panic!("bad foul usi {usi}"));
    match mv {
        ShogiMove::Drop { to, role } => format!("00{}{}{}", to.file, to.rank, role_foulcode(role)),
        ShogiMove::Board { from, to, promote } => {
            let piece = pos.piece_at(from).unwrap_or_else(|| panic!("no piece at from for foul {usi}"));
            let role = if promote { promote_role(piece.role) } else { piece.role };
            format!("{}{}{}{}{}", from.file, from.rank, to.file, to.rank, role_foulcode(role))
        }
    }
}

fn promote_role(role: Role) -> Role {
    use Role::*;
    match role {
        Pawn => Tokin,
        Lance => Promotedlance,
        Knight => Promotedknight,
        Silver => Promotedsilver,
        Bishop => Horse,
        Rook => Dragon,
        r => r,
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = &args[1];
    let bot_color: Color = if args.get(2).map(|s| s.as_str()) == Some("sente") {
        Color::Sente
    } else {
        Color::Gote
    };
    let ply: usize = args[3].parse().unwrap();
    let diag: &str = args.get(4).map(|s| s.as_str()).unwrap_or("");

    let content = fs::read_to_string(path).unwrap();
    let v: serde_json::Value = serde_json::from_str(&content).unwrap();
    let moves = v["moves"].as_array().unwrap();
    let fouls = v["foulAttempts"].as_array().unwrap();

    let mut out = String::new();
    out.push_str("棋戦：Shogi Quest\n手合割：平手\n先手：先手\n後手：後手\n手数----指手---------消費時間--\n");
    out.push_str(&format!(
        "*scenario ply={ply} target={} diag={diag} desc=人間対局の再現（bot={bot_color:?}）\n",
        parse_usi(moves[ply]["usi"].as_str().unwrap()).unwrap().to_usi()
    ));

    let mut pos = Position::initial();
    let mut prev_to: Option<Coord> = None;
    for (i, m) in moves.iter().enumerate() {
        let no = i + 1;
        let usi = m["usi"].as_str().unwrap();
        // このplyの手を指す前の、同じ手番側の反則試行
        let my_fouls: Vec<&serde_json::Value> = fouls
            .iter()
            .filter(|f| f["moveNumber"].as_u64() == Some(no as u64))
            .collect();

        out.push_str(&move_line(no, &pos, usi, prev_to));
        out.push('\n');
        if !my_fouls.is_empty() {
            let codes: Vec<String> = my_fouls
                .iter()
                .map(|f| foul_code(&pos, f["usi"].as_str().unwrap()))
                .collect();
            out.push_str(&format!("*illegal:{}\n", codes.join(",")));
        }

        let mv = parse_usi(usi).unwrap();
        prev_to = Some(match mv {
            ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
        });
        pos.play_unchecked(&mv);
    }
    // 終局後（最終手より後）の反則試行は trailing として、終局宣言行の後に出す
    let last_no = moves.len();
    let trailing: Vec<&serde_json::Value> = fouls
        .iter()
        .filter(|f| f["moveNumber"].as_u64() > Some(last_no as u64))
        .collect();
    if !trailing.is_empty() {
        out.push_str(&format!("{} 反則負け\n", last_no + 1));
        let codes: Vec<String> = trailing
            .iter()
            .map(|f| foul_code(&pos, f["usi"].as_str().unwrap()))
            .collect();
        out.push_str(&format!("*illegal:{}\n", codes.join(",")));
    }
    print!("{out}");
}
