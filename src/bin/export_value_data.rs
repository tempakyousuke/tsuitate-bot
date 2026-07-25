//! 学習データ書き出し: 対局記録（records/*.jsonl、ARENA_RECORD_DIR互換）から
//! 真の棋譜をreplayし、各手番の局面ごとに (value_features, ラベル) をCSVで出力する。
//!
//! ラベルはその手番側から見た対局結果（勝ち=1.0・負け=0.0・引き分け=0.5）。
//! 特徴量の定義は src/value_features.rs に一本化する（学習側とズレないため）。
//!
//! 使い方: cargo run --release --bin export_value_data -- <records/*.jsonl...> > data.csv

use tsuitate_bot::protocol::{Color, GameEndPayload};
use tsuitate_bot::shogi::{Position, parse_usi};
use tsuitate_bot::value_features::{
    TRANSITION_FEATURE_NAMES, TRANSITION_FEATURES, VALUE_FEATURE_NAMES, VALUE_FEATURES,
    transition_features, value_features,
};

/// `mat_swing` を測る先読み手数（両者の手を数えた ply 数。8 = 自分の手番4回ぶん）。
///
/// **最終勝敗ラベルがこの領域では弱い教師である**ことへの対処（2026-07-26）。
/// アリーナ実測で対局の約66%が反則負けで終わっており（詰みは33%）、
/// label（勝敗）は反則レースにほぼ支配されていて局面の良し悪しと直交する。
/// 「多手かけて駒を取りに行く」の価値は、実際に**その後N手で駒得が実現したか**を
/// 見るほうが密で素直な信号になる。
///
/// 同一局面で観測できる継続は実際に指した1本だけなので、pairwise（同じ局面の
/// 別の手どうしの優劣）としては作れない。ここはラベル側の補助信号として入れ、
/// 学習側で `--swing-weight` で勝敗ラベルとブレンドする
const SWING_HORIZON: usize = 8;

fn load_end_payload(path: &str) -> Option<GameEndPayload> {
    let content = std::fs::read_to_string(path).ok()?;
    for line in content.lines() {
        let v: serde_json::Value = serde_json::from_str(line).ok()?;
        if v["type"] == "end" {
            return serde_json::from_value(v["payload"].clone()).ok();
        }
    }
    None
}

fn outcome_value(result: &str, side: Color) -> Option<f64> {
    match (result, side) {
        ("draw", _) => Some(0.5),
        ("sente_win", Color::Sente) | ("gote_win", Color::Gote) => Some(1.0),
        ("sente_win", Color::Gote) | ("gote_win", Color::Sente) => Some(0.0),
        _ => None,
    }
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("使い方: export_value_data <records/*.jsonl...> > data.csv");
        std::process::exit(1);
    }

    // game_id列でゲーム単位のグループ分割ができるようにする（codexレビュー指摘:
    // 行単位のランダム分割だと同じ対局の別手番が学習/検証の両方に混じり
    // データリークになる）。ply列は手数（1始まり）
    // 接近特徴量は value_features / transition_features の末尾に畳み込み済み
    // （列順 = 推論側 evaluate() の組み立て順。ズレると学習と推論が静かに食い違う）
    println!(
        "game_id,ply,{},{},label,mat_swing",
        VALUE_FEATURE_NAMES.join(","),
        TRANSITION_FEATURE_NAMES.join(",")
    );

    let mut games = 0u64;
    let mut rows = 0u64;
    for path in &paths {
        let Some(end) = load_end_payload(path) else {
            eprintln!("読めませんでした（終局まで到達していない記録？）: {path}");
            continue;
        };
        let Some(win_value_sente) = outcome_value(&end.result, Color::Sente) else {
            eprintln!("resultが不明: {path} ({})", end.result);
            continue;
        };

        // 1局ぶんをバッファし、全手のreplayが成功した場合だけ出力する
        // （codexレビュー指摘: パース失敗で打ち切ると、真実が壊れている
        // 可能性がある局の一部だけが結果ラベルつきで紛れ込む）
        // (行の中身, 手番側, 先手視点の駒得) を溜めてから、N手先の駒得スイングを
        // 後ろ向きに埋める（実現した多手の因果を教師にするため。下記 mat_swing）
        let mut buf: Vec<(String, Color)> = vec![];
        let mut material_sente: Vec<f64> = vec![];
        let mut pos = Position::initial();
        let mut ok = true;
        for (i, m) in end.moves.iter().enumerate() {
            let side = pos.turn();
            let label = if side == Color::Sente {
                win_value_sente
            } else {
                1.0 - win_value_sente
            };
            let f = value_features(&pos, side);
            // f[0] = material_diff（手番側視点）。先手視点へ揃えて記録する
            material_sente.push(if side == Color::Sente { f[0] } else { -f[0] });

            let Some(mv) = parse_usi(&m.usi) else {
                eprintln!("  棋譜の手をパースできません: {} ({path})。この局はスキップ", m.usi);
                ok = false;
                break;
            };
            let before = pos.clone();
            pos.play_unchecked(&mv);
            let t = transition_features(&before, &mv, &pos, side);

            let row: Vec<String> =
                f.iter().chain(t.iter()).map(|x| x.to_string()).collect();
            buf.push((format!("{games},{},{},{label}", i + 1, row.join(",")), side));
        }
        if ok {
            // 局末（最終手の後）の駒得も足して、終盤の行が末尾を参照できるようにする
            material_sente.push(value_features(&pos, Color::Sente)[0]);
            for (i, (line, side)) in buf.iter().enumerate() {
                let j = (i + SWING_HORIZON).min(material_sente.len() - 1);
                let swing_sente = material_sente[j] - material_sente[i];
                let swing = if *side == Color::Sente { swing_sente } else { -swing_sente };
                println!("{line},{swing}");
            }
            rows += buf.len() as u64;
            games += 1;
        }
    }
    eprintln!(
        "書き出し完了: {games}局 / {rows}行 / 特徴量{}次元",
        VALUE_FEATURES + TRANSITION_FEATURES
    );
}
