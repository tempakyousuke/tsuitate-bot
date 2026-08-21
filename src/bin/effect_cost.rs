//! 利き走査のコストを実測する（docs/yaneuraou-lessons.md の E に着手する前の必須測定）。
//!
//! やねうら王は各マスの利き数を差分で維持している（`extra/long_effect.h`）。
//! ついたて側で同じことをやると評価粒子数の上限が上がるはずだが、差分更新は
//! バグると静かに壊れるので、**投資に見合うコストなのかを先に数字にする**。
//!
//! 測るもの:
//! 1. 1決定あたりの `Position::is_attacked` / `board::move_targets` の呼び出し回数
//! 2. その2つの1回あたりの実測コスト（同じ局面でのマイクロベンチ）
//! 3. 1 × 2 が思考時間の何%を占めるか
//!
//! usage:
//!   実対局モード（本命。アリーナと同じ経路＝推定器は差分更新される）:
//!     cargo run --release --features effect-profile --bin effect_cost -- game [対局数=2]
//!   シナリオモード（1局面の内訳を見る。**推定器の構築コミ**なので share は薄めに出る）:
//!     cargo run --release --features effect-profile --bin effect_cost -- <kif|名前> <ply> [シード数=3]

use std::time::Instant;

use tsuitate_bot::board::move_targets;
use tsuitate_bot::effect_profile::{
    IS_ATTACKED, LEGAL_MOVES, MOVE_TARGETS, OPP_MOVE_NN, VALUE_NN, take,
};
use tsuitate_bot::protocol::VisiblePiece;
use tsuitate_bot::scenario_core::{load_scenario, ranking_one, replay};
use tsuitate_bot::shogi::Position;

fn main() {
    if cfg!(not(feature = "effect-profile")) {
        eprintln!("--features effect-profile を付けて実行すること（カウンタが無効です）");
        std::process::exit(1);
    }
    let args: Vec<String> = std::env::args().collect();
    let spec = args
        .get(1)
        .expect("usage: effect_cost game [games] | effect_cost <kif|名前> <ply> [seeds]");
    if spec == "game" {
        let games: u32 = args.get(2).map(|s| s.parse().expect("games")).unwrap_or(2);
        run_game_mode(games);
        return;
    }
    let ply: usize = args.get(2).expect("ply").parse().expect("ply");
    let seeds: u64 = args.get(3).map(|s| s.parse().expect("seeds")).unwrap_or(3);

    let sc = load_scenario(spec, Some(ply), None, None).expect("シナリオを読めません");
    let rep = replay(&sc.kifu, sc.ply);
    println!("{spec} ply={ply} 手番={:?} / seeds={seeds}", rep.pos.turn());

    // 1) 1決定あたりの呼び出し回数と壁時計
    let mut total_ms = 0.0;
    let mut total_attacked = 0u64;
    let mut total_targets = 0u64;
    for seed in 0..seeds {
        take(&IS_ATTACKED);
        take(&MOVE_TARGETS);
        let started = Instant::now();
        let got = ranking_one(&rep, seed, "estimator");
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        let a = take(&IS_ATTACKED);
        let t = take(&MOVE_TARGETS);
        println!(
            "  seed={seed}: legal_moves {} / opp_move_nn {} / value_nn {}",
            take(&LEGAL_MOVES),
            take(&OPP_MOVE_NN),
            take(&VALUE_NN),
        );
        println!(
            "  seed={seed}: {ms:.0}ms / is_attacked {a} 回 / move_targets {t} 回{}",
            match got {
                Some((usi, r)) => format!(" / 候補 {} 手・選択 {usi}", r.len()),
                None => " / ランキングなし".into(),
            }
        );
        total_ms += ms;
        total_attacked += a;
        total_targets += t;
    }
    let decisions = seeds as f64;

    // 2) 1回あたりのコスト（この局面でのマイクロベンチ）
    let pos = rep.pos.clone();
    let me = rep.pos.turn();
    let ns_is_attacked = bench_is_attacked(&pos);
    let pieces = pos.pieces_of(me);
    let ns_move_targets = bench_move_targets(&pieces, me);

    // 3) 占有率
    let est_ms = (total_attacked as f64 * ns_is_attacked + total_targets as f64 * ns_move_targets)
        / 1_000_000.0;
    println!();
    println!("平均: {:.0}ms/決定", total_ms / decisions);
    println!(
        "  is_attacked   {:>9.0} 回/決定 × {ns_is_attacked:.0}ns = {:.0}ms",
        total_attacked as f64 / decisions,
        total_attacked as f64 * ns_is_attacked / 1_000_000.0 / decisions
    );
    println!(
        "  move_targets  {:>9.0} 回/決定 × {ns_move_targets:.0}ns = {:.0}ms",
        total_targets as f64 / decisions,
        total_targets as f64 * ns_move_targets / 1_000_000.0 / decisions
    );
    println!(
        "  利き走査の推定シェア: {:.1}%（{:.0}ms / {:.0}ms）",
        est_ms / total_ms * 100.0,
        est_ms / decisions,
        total_ms / decisions
    );
    println!();
    println!("判断の目安（改善計画 項目3）: 3割未満なら E（差分計算）には着手しない");
}

/// 実対局モード: アリーナと同じ経路（推定器は毎手の差分更新）で走らせ、
/// 「思考時間の合計」に対する利き走査のシェアを出す。**これが本番の profile**。
/// シナリオモードは推定器を毎回ゼロから構築するので分母が水増しされる
fn run_game_mode(games: u32) {
    use tsuitate_bot::selfplay::run_match_with;
    use tsuitate_bot::strategy;

    take(&IS_ATTACKED);
    take(&MOVE_TARGETS);
    println!("実対局モード: estimator vs estimator_v11 を {games}局");
    let started = Instant::now();
    let stats = run_match_with(
        games,
        &|| strategy::make("estimator").expect("戦略名"),
        &|| strategy::make("estimator_v11").expect("戦略名"),
    );
    let wall = started.elapsed().as_secs_f64();
    let attacked = take(&IS_ATTACKED);
    let targets = take(&MOVE_TARGETS);
    let think_ms: f64 = stats
        .think_us_a
        .iter()
        .chain(&stats.think_us_b)
        .map(|us| *us as f64 / 1000.0)
        .sum();
    let decisions = (stats.think_us_a.len() + stats.think_us_b.len()) as f64;

    // 1回あたりのコストは初期局面と中盤局面の平均で見る（実対局は局面が変わるので）
    let mut pos = Position::initial();
    let ns_attacked_open = bench_is_attacked(&pos);
    for usi in ["7g7f", "3c3d", "8h2b+", "3a2b", "B*4e"] {
        if let Some(mv) = tsuitate_bot::shogi::parse_usi(usi) {
            pos.play_unchecked(&mv);
        }
    }
    let ns_attacked_mid = bench_is_attacked(&pos);
    let ns = (ns_attacked_open + ns_attacked_mid) / 2.0;
    let est_ms = attacked as f64 * ns / 1_000_000.0;

    println!(
        "  {games}局 / 決定 {decisions:.0} 回 / 思考時間合計 {:.1}秒（壁時計 {wall:.1}秒・並列あり）",
        think_ms / 1000.0
    );
    println!(
        "  is_attacked  {attacked} 回（{:.0} 回/決定）× {ns:.0}ns = {:.1}秒",
        attacked as f64 / decisions.max(1.0),
        est_ms / 1000.0
    );
    println!("  move_targets {targets} 回（{:.0} 回/決定）", targets as f64 / decisions.max(1.0));
    println!(
        "  利き走査（is_attacked）の推定シェア: {:.1}%",
        est_ms / think_ms * 100.0
    );
    // 「利きでないなら何なのか」の内訳。粒子の相手手サンプルと2つのNNが候補
    let legal = take(&LEGAL_MOVES);
    let opp_nn = take(&OPP_MOVE_NN);
    let val_nn = take(&VALUE_NN);
    let ns_legal = bench_legal_moves(&pos);
    let ns_opp_nn = bench_opp_move_nn();
    let ns_val_nn = bench_value_nn();
    let share = |calls: u64, ns: f64| -> (f64, f64) {
        let ms = calls as f64 * ns / 1_000_000.0;
        (ms / 1000.0, ms / think_ms * 100.0)
    };
    for (name, calls, ns) in [
        ("legal_moves ", legal, ns_legal),
        ("opp_move_nn ", opp_nn, ns_opp_nn),
        ("value_nn    ", val_nn, ns_val_nn),
    ] {
        let (s, pct) = share(calls, ns);
        println!(
            "  {name} {calls} 回（{:.0}/決定）× {ns:.0}ns = {s:.1}秒 → {pct:.1}%",
            calls as f64 / decisions.max(1.0)
        );
    }
    println!();
    println!("判断の目安（改善計画 項目3）: 3割未満なら E（差分計算）には着手しない");
}

/// 全81マス × 先後の is_attacked を繰り返して1回あたりのnsを出す
fn bench_is_attacked(pos: &Position) -> f64 {
    use tsuitate_bot::board::Coord;
    use tsuitate_bot::protocol::Color;
    let squares: Vec<Coord> = (1..=9)
        .flat_map(|file| (1..=9).map(move |rank| Coord { file, rank }))
        .collect();
    let reps = 200;
    let started = Instant::now();
    let mut sink = 0u64;
    for _ in 0..reps {
        for &sq in &squares {
            sink += pos.is_attacked(sq, Color::Sente) as u64;
            sink += pos.is_attacked(sq, Color::Gote) as u64;
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    // カウンタ自身の加算コストも含んだ値だが、fetch_add(Relaxed) は数ns
    take(&IS_ATTACKED);
    elapsed * 1e9 / (reps as f64 * squares.len() as f64 * 2.0)
}

fn bench_legal_moves(pos: &Position) -> f64 {
    let reps = 2000;
    let started = Instant::now();
    let mut sink = 0usize;
    for _ in 0..reps {
        sink += pos.legal_moves().len();
    }
    let elapsed = started.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    take(&LEGAL_MOVES);
    take(&IS_ATTACKED);
    elapsed * 1e9 / reps as f64
}

fn bench_opp_move_nn() -> f64 {
    let f = [0.3f64; 26];
    let reps = 200_000;
    let started = Instant::now();
    let mut sink = 0.0;
    for _ in 0..reps {
        sink += tsuitate_bot::opp_move_nn::opp_move_nn_forward(&f);
    }
    let elapsed = started.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    take(&OPP_MOVE_NN);
    elapsed * 1e9 / reps as f64
}

fn bench_value_nn() -> f64 {
    let f = [0.3f64; 22];
    let reps = 200_000;
    let started = Instant::now();
    let mut sink = 0.0;
    for _ in 0..reps {
        sink += tsuitate_bot::value_nn::value_nn_forward(&f);
    }
    let elapsed = started.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    take(&VALUE_NN);
    elapsed * 1e9 / reps as f64
}

fn bench_move_targets(pieces: &[VisiblePiece], color: tsuitate_bot::protocol::Color) -> f64 {
    if pieces.is_empty() {
        return 0.0;
    }
    let reps = 2000;
    let started = Instant::now();
    let mut sink = 0usize;
    for _ in 0..reps {
        for p in pieces {
            sink += move_targets(pieces, p, color).len();
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    take(&MOVE_TARGETS);
    elapsed * 1e9 / (reps as f64 * pieces.len() as f64)
}
