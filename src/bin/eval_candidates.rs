//! オフライン検証用の一時ツール: moves.json の指定手数（ply）まで再生し、
//! 指定した候補USI手それぞれについて「着手後の局面」の value_features を
//! CSVで出力する。学習済みvalueネット（tsuitate-nn）へ食わせて、
//! 「どの候補を高く評価するか」を確認するのに使う。
//!
//! 使い方: cargo run --release --bin eval_candidates -- <moves.json> <ply> <me:sente|gote> <USI手...>

use tsuitate_bot::protocol::Color;
use tsuitate_bot::shogi::{Position, parse_usi};
use tsuitate_bot::value_features::{VALUE_FEATURE_NAMES, value_features};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let ply: usize = args[2].parse().unwrap();
    let me = if args[3] == "sente" { Color::Sente } else { Color::Gote };
    let candidates = &args[4..];

    let content = std::fs::read_to_string(path).unwrap();
    let v: serde_json::Value = serde_json::from_str(&content).unwrap();
    let moves = v["moves"].as_array().unwrap();

    let mut base = Position::initial();
    for m in moves.iter().take(ply) {
        let usi = m["usi"].as_str().unwrap();
        let mv = parse_usi(usi).unwrap_or_else(|| panic!("bad usi {usi}"));
        base.play_unchecked(&mv);
    }

    // codexレビュー指摘: me は「候補手を指す側」でなければ評価の符号が反転する。
    // CLI引数の指定ミスをここで検出する（着手後にpos.turn()が反転してからでは
    // 気づけない）
    if base.turn() != me {
        eprintln!(
            "エラー: 指定した me（{me:?}）はこの局面の手番（{:?}）と一致しません",
            base.turn()
        );
        std::process::exit(1);
    }

    println!("usi,{}", VALUE_FEATURE_NAMES.join(","));
    for usi in candidates {
        let mv = parse_usi(usi).unwrap_or_else(|| panic!("bad candidate usi {usi}"));
        if !base.is_legal(&mv) {
            eprintln!("{usi}: この局面では非合法（スキップ）");
            continue;
        }
        let mut pos = base.clone();
        pos.play_unchecked(&mv);
        let f = value_features(&pos, me);
        let row: Vec<String> = f.iter().map(|x| x.to_string()).collect();
        println!("{usi},{}", row.join(","));
    }
}
