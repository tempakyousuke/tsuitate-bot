//! 診断: 推定器の相手モデル（`opp_reply_weights` = 粒子を進めるときと探索の応手に使う opp_move NN）が、
//! 実際の相手の手にどれだけ確率を置くかを、記録の真の手順で測る（runtime には何も入らない）。
//!
//! 受理された各手の直前の真の局面で、指した側を「相手」と見て `opp_reply_weights` を引き
//! （露見マス・触ったマスは渡さない。版どうしの比べ合いにだけ使う）、実際の手の確率を集計する。
//! 粒子は観測と整合する手をこのモデルからサンプルして相手の駒を動かすので、ここが低い対局ほど
//! 粒子が真の局面を追いにくい（`docs/rl-deepnash-design.md` の「探索が MMD の版で伸びない理由」）。
//!
//! ```sh
//! cargo run --release --bin opp_model_fit -- records_a/ [records_b/ ...]
//! ```

use std::collections::BTreeMap;

use tsuitate_bot::estimator::opp_reply_weights;
use tsuitate_bot::shogi::{Position, parse_usi};
use tsuitate_bot::truth_replay::load_end;

#[derive(Default)]
struct Acc {
    n: f64,
    logp: f64,
    top1: f64,
    top5: f64,
    zero: f64,
}

impl Acc {
    fn add(&mut self, p: f64, rank: usize) {
        self.n += 1.0;
        if p > 0.0 {
            self.logp += p.ln();
        } else {
            self.zero += 1.0;
            self.logp += (1e-9f64).ln();
        }
        self.top1 += f64::from(u8::from(rank == 0));
        self.top5 += f64::from(u8::from(rank < 5));
    }

    fn line(&self) -> String {
        if self.n == 0.0 {
            return "—".into();
        }
        format!(
            "{:>5} 手 / 平均 log p {:+.2} / 1位 {:.1}% / 5位以内 {:.1}% / 確率0 {:.1}%",
            self.n,
            self.logp / self.n,
            100.0 * self.top1 / self.n,
            100.0 * self.top5 / self.n,
            100.0 * self.zero / self.n
        )
    }
}

fn band(ply: usize) -> &'static str {
    match ply {
        0..=20 => "1-20手",
        21..=40 => "21-40手",
        41..=60 => "41-60手",
        _ => "61手〜",
    }
}

fn main() {
    let dirs: Vec<String> = std::env::args().skip(1).collect();
    assert!(!dirs.is_empty(), "記録のディレクトリ（arena の ARENA_RECORD_DIR）を渡す");
    for dir in dirs {
        let mut all = Acc::default();
        let mut by: BTreeMap<(&'static str, &'static str), Acc> = BTreeMap::new();
        let mut games = 0;
        let mut paths: Vec<_> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).collect();
        paths.sort();
        for path in paths {
            let Some(end) = load_end(path.to_str().unwrap()) else { continue };
            games += 1;
            let mut pos = Position::initial();
            for (i, m) in end.moves.iter().enumerate() {
                let Some(mv) = parse_usi(&m.usi) else { break };
                // 指した側 = 相手。`my_color` はその反対側
                let weights = opp_reply_weights(&pos, m.by_color.other(), &[], &[], 0);
                let total: f64 = weights.iter().map(|x| x.1).sum();
                let p_mv = weights.iter().find(|x| x.0 == mv).map_or(0.0, |x| x.1);
                let rank = weights.iter().filter(|x| x.1 > p_mv).count();
                let p = if total > 0.0 { p_mv / total } else { 0.0 };
                let kind = if m.usi.contains('*') { "打ち" } else { "盤上" };
                all.add(p, rank);
                by.entry((band(i + 1), kind)).or_default().add(p, rank);
                if !pos.is_legal(&mv) {
                    break;
                }
                pos.play_unchecked(&mv);
            }
        }
        println!("== {dir}（{games} 局）");
        println!("  全体   {}", all.line());
        for ((b, k), acc) in &by {
            println!("  {b} {k} {}", acc.line());
        }
    }
}
