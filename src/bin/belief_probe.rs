//! 指定シナリオ・指定plyの手番側が「そのマスに何がいると思っているか」を出す
//! （scenario-gui の玉位置ビリーフを任意のマス・任意の駒種へ一般化した診断）。
//!
//! 「この手を選んだのは、相手の駒がここにいると誤解しているからでは？」という
//! 疑いを潰すための道具。重み規約は `king_belief` と同じ（`weighted_unique_particles` の
//! 重みをシードをまたいで素で合算する。厳密粒子だけの列も同時に出す）。
//!
//! `dump` を付けると、その仮説を持つ粒子の盤面と持ち駒を実際に印字する。
//! **信念が漏れているのか、合法な仮説なのかはここまで見ないと判別できない**
//! （実例 2026-07-25: 8九に先手桂を置く0.2%の粒子は、後手が8九で取った桂を
//! 先手が4七で取り返して8九へ打ち直した = 観測と矛盾しない筋だった）。
//!
//! usage: cargo run --release --bin belief_probe -- <kifパスまたは名前> <ply> <マス,マス,...> [シード数=3] [ダンプ=マス:駒種]
//!   例: TSUITATE_THINK_BUDGET_MS=2000 \
//!       cargo run --release --bin belief_probe -- scenarios/foo.kif 33 8i,8h 3 8i:Knight
use std::collections::HashMap;
use tsuitate_bot::board::{make_usi_square, parse_usi_square};
use tsuitate_bot::protocol::Color;
use tsuitate_bot::scenario_core::{
    build_estimator, load_scenario, replay, weighted_unique_particles,
};

/// dump で印字する粒子の上限（多いだけで読めないので少数に絞る）
const DUMP_MAX: usize = 3;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let spec = args
        .get(1)
        .expect("usage: belief_probe <kif> <ply> <マス,...> [シード数] [マス:駒種]");
    let ply: usize = args.get(2).expect("ply").parse().expect("ply");
    let squares: Vec<_> = args
        .get(3)
        .expect("マス")
        .split(',')
        .map(|s| parse_usi_square(s).expect("USIマス（例 8i）"))
        .collect();
    let seeds: u64 = args.get(4).map(|s| s.parse().expect("シード数")).unwrap_or(3);
    let dump = args.get(5).map(|d| {
        let (sq, role) = d.split_once(':').expect("ダンプ指定は マス:駒種（例 8i:Knight）");
        (parse_usi_square(sq).expect("USIマス"), role.to_string())
    });

    let sc = load_scenario(spec, Some(ply), None, None).expect("load");
    let rep = replay(&sc.kifu, sc.ply);
    let me = rep.pos.turn();
    // 粒子数・再生成予算のスケール（SearchBudget::from_ms は非公開なので
    // bin/scenario の diag と同じ直書き。既定は思考予算 2000ms 相当）
    let budget_ms: f64 = std::env::var("TSUITATE_THINK_BUDGET_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000.0);
    let scale = budget_ms / 900.0;
    println!("手番: {me:?} / {}手目の直前 / 予算スケール {scale:.2}", ply + 1);
    for &sq in &squares {
        let truth = match rep.pos.piece_at(sq) {
            None => "空".to_string(),
            Some(p) => format!("{:?}{:?}", p.color, p.role),
        };
        println!("真実 {}: {truth}", make_usi_square(sq));
    }

    // マスごとに「駒種キー -> 重み」。厳密粒子（情報制約も物理制約も緩めていない）
    // だけの集計を別に持つ: taint 頼りの信念か、厳密粒子も支持する信念かで
    // 疑いの重さが変わる
    let mut tally: Vec<HashMap<String, f64>> = vec![HashMap::new(); squares.len()];
    let mut tally_strict: Vec<HashMap<String, f64>> = vec![HashMap::new(); squares.len()];
    let mut mass = 0.0f64;
    let mut mass_strict = 0.0f64;
    let mut dumped = 0usize;

    for seed in 0..seeds.max(1) {
        let est = build_estimator(&rep, seed, scale, |_, _| {});
        let parts = weighted_unique_particles(&est);
        println!(
            "seed={seed}: ユニーク粒子 {} / うち厳密 {}",
            parts.len(),
            parts.iter().filter(|(_, _, strict)| *strict).count()
        );
        for (pos, w, strict) in parts {
            mass += w;
            if strict {
                mass_strict += w;
            }
            for (i, &sq) in squares.iter().enumerate() {
                let key = match pos.piece_at(sq) {
                    None => "空".to_string(),
                    Some(p) => format!("{:?}{:?}", p.color, p.role),
                };
                *tally[i].entry(key.clone()).or_insert(0.0) += w;
                if strict {
                    *tally_strict[i].entry(key).or_insert(0.0) += w;
                }
            }
            if let Some((dsq, drole)) = &dump {
                if dumped < DUMP_MAX
                    && pos.piece_at(*dsq).is_some_and(|p| &format!("{:?}", p.role) == drole)
                {
                    dumped += 1;
                    print_particle(pos, w, strict);
                }
            }
        }
    }

    for (i, &sq) in squares.iter().enumerate() {
        println!("--- {} ---", make_usi_square(sq));
        let mut rows: Vec<_> = tally[i].iter().collect();
        rows.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (key, w) in rows {
            let s = tally_strict[i].get(key).copied().unwrap_or(0.0);
            println!(
                "  {key:14} 全体 {:6.2}%  厳密 {:6.2}%",
                w / mass.max(1e-12) * 100.0,
                if mass_strict > 0.0 { s / mass_strict * 100.0 } else { 0.0 }
            );
        }
    }
    if mass_strict <= 0.0 {
        println!("（厳密粒子が全滅しているので厳密列は全ゼロ = 評価は taint 頼り）");
    }
}

fn print_particle(pos: &tsuitate_bot::shogi::Position, w: f64, strict: bool) {
    println!("  [粒子 重み={w:.4} 厳密={strict}]");
    let mut pieces: Vec<String> = pos
        .pieces()
        .map(|(c, p)| format!("{}{:?}{:?}", make_usi_square(c), p.color, p.role))
        .collect();
    pieces.sort();
    println!("   盤: {}", pieces.join(" "));
    println!("   先手持駒: {:?}", pos.hand_map(Color::Sente));
    println!("   後手持駒: {:?}", pos.hand_map(Color::Gote));
}
