//! pairwise補助loss用のデータ書き出し: 対局記録から真の棋譜をreplayし、
//! 各手番の局面ごとに合法手を全列挙、transition_featuresの
//! net_capture_then_recapture（駒得-交換損失の実質収支）が最大の手と
//! 最小の手のペアを1件出力する。
//!
//! 狙い: docs/nn-value-phase1.md参照。最終勝敗ラベルだけでは
//! 「駒を危険にさらす手」と「攻めている・優勢な手」が交絡し、
//! transition特徴量の符号を学習できなかった（codexレビュー指摘、2026-07-20）。
//! net_capture_then_recaptureは手作り特徴量として単体では正しい向きを
//! 検証済み（gold-check/kakudo）なので、それ自体を教師にして
//! 「同じ局面内でのこの特徴量の向き」を補助lossとして直接教える。
//!
//! 使い方: cargo run --release --bin export_pairwise_data -- <records/*.jsonl...> > pairwise.csv
//!
//! 出力列: game_id,ply,state(16),good_transition(6),bad_transition(6),quality_gap
//! quality_gapが小さい（局面内で手による差がほぼ無い）ペアは学習ノイズに
//! なるだけなので閾値未満は出力しない

use tsuitate_bot::protocol::GameEndPayload;
use tsuitate_bot::shogi::{Position, parse_usi};
use tsuitate_bot::value_features::{
    TRANSITION_FEATURE_NAMES, VALUE_FEATURE_NAMES, transition_features, value_features,
};

const MIN_QUALITY_GAP: f64 = 0.5;

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

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("使い方: export_pairwise_data <records/*.jsonl...> > pairwise.csv");
        std::process::exit(1);
    }

    let state_names: Vec<String> = VALUE_FEATURE_NAMES.iter().map(|s| s.to_string()).collect();
    let good_names: Vec<String> =
        TRANSITION_FEATURE_NAMES.iter().map(|s| format!("good_{s}")).collect();
    let bad_names: Vec<String> =
        TRANSITION_FEATURE_NAMES.iter().map(|s| format!("bad_{s}")).collect();
    println!(
        "game_id,ply,{},{},{},quality_gap",
        state_names.join(","),
        good_names.join(","),
        bad_names.join(",")
    );

    let mut games = 0u64;
    let mut pairs = 0u64;
    for path in &paths {
        let Some(end) = load_end_payload(path) else {
            eprintln!("読めませんでした（終局まで到達していない記録？）: {path}");
            continue;
        };

        let mut pos = Position::initial();
        let mut ok = true;
        let mut buf: Vec<String> = vec![];
        for (i, m) in end.moves.iter().enumerate() {
            let side = pos.turn();
            let legals = pos.legal_moves();

            let mut best: Option<(f64, [f64; 6])> = None;
            let mut worst: Option<(f64, [f64; 6])> = None;
            for mv in &legals {
                let mut after = pos.clone();
                after.play_unchecked(mv);
                let t = transition_features(&pos, mv, &after, side);
                let quality = t[4]; // net_capture_then_recapture
                if best.is_none_or(|(q, _)| quality > q) {
                    best = Some((quality, t));
                }
                if worst.is_none_or(|(q, _)| quality < q) {
                    worst = Some((quality, t));
                }
            }

            if let (Some((qg, good)), Some((qb, bad))) = (best, worst) {
                let gap = qg - qb;
                if gap >= MIN_QUALITY_GAP {
                    let state = value_features(&pos, side);
                    let state_s: Vec<String> = state.iter().map(|x| x.to_string()).collect();
                    let good_s: Vec<String> = good.iter().map(|x| x.to_string()).collect();
                    let bad_s: Vec<String> = bad.iter().map(|x| x.to_string()).collect();
                    buf.push(format!(
                        "{games},{},{},{},{},{gap}",
                        i + 1,
                        state_s.join(","),
                        good_s.join(","),
                        bad_s.join(",")
                    ));
                }
            }

            let Some(mv) = parse_usi(&m.usi) else {
                eprintln!("  棋譜の手をパースできません: {} ({path})。この局はスキップ", m.usi);
                ok = false;
                break;
            };
            pos.play_unchecked(&mv);
        }
        if ok {
            for line in &buf {
                println!("{line}");
            }
            pairs += buf.len() as u64;
            games += 1;
        }
    }
    eprintln!("書き出し完了: {games}局 / {pairs}ペア");
}
