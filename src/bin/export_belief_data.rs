//! 信念ネット（NN段階②）の学習データ書き出し。
//!
//! 対局記録（`records/*.jsonl` / `ARENA_RECORD_DIR` 互換）の**真実**（game:end の
//! 全手順＋反則試行）を再生し、各手番の決定点で
//! 「その手番側の観測履歴から作った特徴量（`belief_features.rs`）」と
//! 「真の局面での相手駒の有無・駒種」を対にして CSV へ出す。
//!
//! - 1行 = 1マス。`(game_id, decision_id)` が同じ行が1つの決定点（81マスから
//!   自駒のマスを除いたもの）
//! - **両者ぶん**の決定点を出す（観測履歴はどちらの側も真実から再構成できる）
//! - `game_id` は対局単位の train/val 分割用（nn-value-phase1 のリーク教訓）
//!
//! 使い方:
//!   cargo run --release --bin export_belief_data -- [--stride N] <records/*.jsonl...> > data.csv
//!
//! `--stride N`（既定 3）は決定点の間引き。1決定点あたり60行以上出るので、
//! 全決定点を出すと数百万行になる。

use tsuitate_bot::belief_features::{
    BELIEF_FEATURE_NAMES, BeliefContext, all_squares, label, role_class, sq_index,
};
use tsuitate_bot::protocol::Color;
use tsuitate_bot::truth_replay::{for_each_decision, load_end};

/// 数値を短く書く（0 が大半なので CSV が肥大化しないように）
pub fn fmt(x: f64) -> String {
    if x == 0.0 {
        "0".to_string()
    } else if x == 1.0 {
        "1".to_string()
    } else {
        format!("{x:.4}")
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut stride = 3usize;
    if let Some(i) = args.iter().position(|a| a == "--stride") {
        stride = args
            .get(i + 1)
            .and_then(|s| s.parse().ok())
            .filter(|&n: &usize| n > 0)
            .unwrap_or(3);
        args.drain(i..=i + 1);
    }
    if args.is_empty() {
        eprintln!("使い方: export_belief_data [--stride N] <records/*.jsonl...> > data.csv");
        std::process::exit(1);
    }

    println!(
        "game_id,decision_id,side,sq,{},occ,role,promoted",
        BELIEF_FEATURE_NAMES.join(",")
    );

    let mut games = 0u64;
    let mut rows = 0u64;
    let mut skipped = 0u64;
    for path in &args {
        let Some(end) = load_end(path) else {
            skipped += 1;
            continue;
        };
        // 1局ぶんをバッファし、全手の再生が成功した場合だけ出力する
        // （壊れた棋譜の一部だけが紛れ込むのを防ぐ。export_value_data と同じ規約）
        let mut buf: Vec<String> = vec![];
        let game_id = games;
        let ok = for_each_decision(&end, |pos, side, log, decision_id| {
            if !(decision_id as usize).is_multiple_of(stride) {
                return;
            }
            let ctx = BeliefContext::from_log(side, log);
            for sq in all_squares() {
                if ctx.is_mine(sq) {
                    continue;
                }
                let (occ, role) = label(pos, side, sq);
                let (base, promoted) = role_class(role);
                let feats: Vec<String> = ctx.features(sq).iter().map(|&x| fmt(x)).collect();
                buf.push(format!(
                    "{game_id},{decision_id},{},{},{},{},{},{}",
                    u8::from(side == Color::Gote),
                    sq_index(sq),
                    feats.join(","),
                    u8::from(occ),
                    base,
                    promoted
                ));
            }
        });
        if ok {
            for line in &buf {
                println!("{line}");
            }
            rows += buf.len() as u64;
            games += 1;
        } else {
            eprintln!("  棋譜を再生できませんでした: {path}。この局はスキップ");
            skipped += 1;
        }
    }
    eprintln!(
        "書き出し完了: {games}局 / {rows}行（マス単位） / 特徴量{}次元 / stride={stride} / スキップ{skipped}件",
        BELIEF_FEATURE_NAMES.len()
    );
}
