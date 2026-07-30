//! 玉位置ネット（81マス上のカテゴリカル分布）の学習データ書き出し。
//!
//! 段階②の pointwise 信念ネットは「マスごとの独立確率」なので複数駒の
//! 同時制約を表現できないが、**玉は1枚**なので候補マス集合上の softmax
//! （カテゴリカル分布）として学習すればその限界を回避できる。
//!
//! - 1行 = 1候補マス。`(game_id, decision_id, side)` が同じ行が1つの決定点で、
//!   その中の `is_king=1` はちょうど1行（候補集合内 softmax の教師）
//! - 候補集合は `deduce::opp_king_candidates`（健全 = 真実を落とさない）に、
//!   自駒マスの除外（自駒と同じマスに相手玉は居られない）を足したもの。
//!   推論側（king_belief_nn）も同じマスクを使うこと
//! - 真の玉が候補外だったら演繹の健全性違反なので計数して決定点ごと捨てる
//!   （実測0のはず。0でなければ deduce のバグ）
//! - 候補が1マスしかない決定点は softmax の教師にならないので出さない
//! - 特徴量は belief_features の35次元 + KING_EXTRA（玉専用）
//!
//! 使い方:
//!   cargo run --release --bin export_king_belief_data -- [--stride N] <records/*.jsonl...> > data.csv

use tsuitate_bot::belief_features::{
    BELIEF_FEATURE_NAMES, BeliefContext, KING_EXTRA_FEATURE_NAMES, sq_index,
};
use tsuitate_bot::deduce::opp_king_candidates;
use tsuitate_bot::protocol::Color;
use tsuitate_bot::truth_replay::{for_each_decision, load_end};

fn fmt(x: f64) -> String {
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
    let mut stride = 2usize;
    if let Some(i) = args.iter().position(|a| a == "--stride") {
        stride = args
            .get(i + 1)
            .and_then(|s| s.parse().ok())
            .filter(|&n: &usize| n > 0)
            .unwrap_or(2);
        args.drain(i..=i + 1);
    }
    if args.is_empty() {
        eprintln!("使い方: export_king_belief_data [--stride N] <records/*.jsonl...> > data.csv");
        std::process::exit(1);
    }

    println!(
        "game_id,decision_id,side,n_cands,sq,{},{},is_king",
        BELIEF_FEATURE_NAMES.join(","),
        KING_EXTRA_FEATURE_NAMES.join(",")
    );

    let mut games = 0u64;
    let mut rows = 0u64;
    let mut decisions = 0u64;
    let mut singletons = 0u64;
    let mut violations = 0u64;
    let mut skipped = 0u64;
    let mut cand_sum = 0u64;
    for path in &args {
        let Some(end) = load_end(path) else {
            skipped += 1;
            continue;
        };
        let mut buf: Vec<String> = vec![];
        let game_id = games;
        let ok = for_each_decision(&end, |pos, side, log, decision_id| {
            if !(decision_id as usize).is_multiple_of(stride) {
                return;
            }
            let Some(true_king) = pos.king_square(side.other()) else {
                return;
            };
            let ctx = BeliefContext::from_log(side, log);
            let mut cands = opp_king_candidates(side, log);
            // 自駒マスの除外は健全（自駒と相手玉は同居できない）
            cands.retain(|&c| !ctx.is_mine(c));
            if !cands.contains(&true_king) {
                violations += 1;
                return;
            }
            if cands.len() < 2 {
                singletons += 1;
                return;
            }
            decisions += 1;
            cand_sum += cands.len() as u64;
            for sq in &cands {
                let feats: Vec<String> = ctx
                    .features(*sq)
                    .iter()
                    .chain(ctx.king_extra(*sq).iter())
                    .map(|&x| fmt(x))
                    .collect();
                buf.push(format!(
                    "{game_id},{decision_id},{},{},{},{},{}",
                    u8::from(side == Color::Gote),
                    cands.len(),
                    sq_index(*sq),
                    feats.join(","),
                    u8::from(*sq == true_king)
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
        "書き出し完了: {games}局 / 決定点{decisions}（平均候補{:.1}マス） / {rows}行 / stride={stride}",
        cand_sum as f64 / decisions.max(1) as f64
    );
    eprintln!(
        "  候補1マスの決定点（教師にならない）: {singletons} / 健全性違反: {violations} / スキップ: {skipped}件"
    );
    if violations > 0 {
        eprintln!("  ★健全性違反が0でない — deduce::opp_king_candidates のバグの可能性");
    }
}
