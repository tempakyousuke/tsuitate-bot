//! オフライン検証用の一時ツール: moves.json または scenarios/*.kif の指定手数（ply）
//! まで再生し、指定した候補USI手それぞれについて (state_features, transition_features)
//! をCSVで出力する。学習済みvalueネット（tsuitate-nn）へ食わせて、
//! 「どの候補を高く評価するか」を確認するのに使う。
//!
//! 使い方: cargo run --release --bin eval_candidates -- <moves.json|foo.kif> <ply> <me:sente|gote> <USI手...>
//!
//! state_featuresは**着手前（base、meの手番）の局面**、transition_featuresは
//! その状態から`mv`を指した結果を、いずれも me 視点で計算する
//! （export_value_data.rsと同じ規約: 学習データは常に「指す前の局面・指した側の
//! 視点」。2026-07-20、codexレビュー指摘: 一時期「着手後・相手視点」に変えたが、
//! これは学習データの時点（指す前）とズレる誤った修正だったため差し戻した。
//! 真の問題は me!=pos.turn() ではなく「着手後の局面を使っていたこと」だった）。

use tsuitate_bot::kifu::parse_kif;
use tsuitate_bot::protocol::Color;
use tsuitate_bot::shogi::{Position, parse_usi};
use tsuitate_bot::value_features::{
    TRANSITION_FEATURE_NAMES, VALUE_FEATURE_NAMES, transition_features, value_features,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let ply: usize = args[2].parse().unwrap();
    let me = if args[3] == "sente" { Color::Sente } else { Color::Gote };
    let candidates = &args[4..];

    let content = std::fs::read_to_string(path).unwrap();
    let move_usis: Vec<String> = if path.ends_with(".kif") {
        let kifu = parse_kif(&content).unwrap_or_else(|e| panic!("{path}: {e}"));
        kifu.plies.iter().map(|p| p.mv.to_usi()).collect()
    } else {
        let v: serde_json::Value = serde_json::from_str(&content).unwrap();
        v["moves"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["usi"].as_str().unwrap().to_string())
            .collect()
    };

    let mut base = Position::initial();
    for usi in move_usis.iter().take(ply) {
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

    println!(
        "usi,{},{}",
        VALUE_FEATURE_NAMES.join(","),
        TRANSITION_FEATURE_NAMES.join(",")
    );
    for usi in candidates {
        let mv = parse_usi(usi).unwrap_or_else(|| panic!("bad candidate usi {usi}"));
        if !base.is_legal(&mv) {
            eprintln!("{usi}: この局面では非合法（スキップ）");
            continue;
        }
        let mut pos = base.clone();
        pos.play_unchecked(&mv);
        // stateは着手前（base, me視点）、transitionはbase→posの遷移をme視点で
        let f = value_features(&base, me);
        let t = transition_features(&base, &mv, &pos, me);
        let row: Vec<String> = f.iter().chain(t.iter()).map(|x| x.to_string()).collect();
        println!("{usi},{}", row.join(","));
    }
}
