//! 一時的な調査用ツール: 対局記録(JSON)からshogi-quest風KIFを生成する。
//! 人間対局の局面をscenarios/*.kifへ落とし込むための変換専用（使い捨て）。
//! KIF 整形の本体は kifu::kif_body（scenario-gui の対局モードと共用）。
use std::env;
use std::fs;

use tsuitate_bot::kifu::kif_body;
use tsuitate_bot::protocol::Color;
use tsuitate_bot::shogi::parse_usi;

fn usage() -> &'static str {
    "usage: cargo run --bin make_scenario -- <moves.json> <sente|gote> <ply> [diag]"
}

fn exit_usage(msg: &str) -> ! {
    eprintln!("{msg}");
    eprintln!("{}", usage());
    std::process::exit(1);
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 3 {
        exit_usage("引数が不足しています");
    }
    let path = &args[0];
    let bot_color: Color = match args[1].as_str() {
        "sente" => Color::Sente,
        "gote" => Color::Gote,
        color => exit_usage(&format!("手番は sente か gote を指定してください: {color}")),
    };
    let ply: usize = args[2]
        .parse()
        .unwrap_or_else(|_| exit_usage(&format!("ply を数値として読めません: {}", args[2])));
    let diag: Option<&str> = args.get(3).map(String::as_str).filter(|s| !s.is_empty());

    let content = fs::read_to_string(path).unwrap();
    let v: serde_json::Value = serde_json::from_str(&content).unwrap();
    let moves: Vec<String> = v["moves"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["usi"].as_str().unwrap().to_string())
        .collect();
    let fouls: Vec<(u32, String)> = v["foulAttempts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| {
            (
                f["moveNumber"].as_u64().unwrap() as u32,
                f["usi"].as_str().unwrap().to_string(),
            )
        })
        .collect();

    let mut out = String::new();
    out.push_str("棋戦：Shogi Quest\n手合割：平手\n先手：先手\n後手：後手\n手数----指手---------消費時間--\n");
    out.push_str(&format!(
        "*scenario ply={ply} target={}",
        parse_usi(&moves[ply]).unwrap().to_usi()
    ));
    if let Some(diag) = diag {
        out.push_str(&format!(" diag={diag}"));
    }
    out.push_str(&format!(" desc=人間対局の再現（bot={bot_color:?}）\n"));

    // 終局後（最終手より後）の反則試行があれば「反則負け」行の後の trailing にする
    let ending = if fouls.iter().any(|(mn, _)| *mn as usize > moves.len()) {
        Some("反則負け")
    } else {
        None
    };
    out.push_str(&kif_body(&moves, &fouls, ending).unwrap_or_else(|e| {
        eprintln!("KIF生成に失敗: {e}");
        std::process::exit(1);
    }));
    print!("{out}");
}
