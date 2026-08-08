//! 実対局の局面再現実験。
//!
//! Shogi Quest からエクスポートした棋譜（`scenarios/*.kif`。真実の手順＋反則試行）を
//! リプレイして任意の局面を再現し、戦略の選択・粒子の信念分布・終盤遂行を調べる。
//!
//! シナリオの追加手順:
//! 1. Shogi Quest の棋譜をそのまま `scenarios/<名前>.kif` に保存する
//! 2. `*scenario ply=<N> [diag=<マス,マス>] [target=<USI>] [desc=<説明>]` の1行を
//!    ファイル内（どこでも可）に足す。ply=N は「N手目まで再生して N+1手目を
//!    考えさせる」。target 省略時は棋譜で実際に指された N+1手目が注目手になる
//! 3. `cargo run --release --bin scenario -- <名前>` で選択実験が走る
//!    （リプレイ時に全手・全反則試行を裁定検証するので、棋譜の欠落や
//!    パース誤りは即 panic で分かる）
//!
//! 反則試行は MyFoul / OpponentFoul として両者の観測ログに再現する
//! （反則回数は foul_limit の残量と推定器の制約の両方に効く）。
//! 推定器は実対局と同じく「自分の手番ごと」に逐次 update する（prewarm）。
//!
//! 読み込み・リプレイ・一手選択の本体は `scenario_core`（scenario-gui と共有）。
//!
//! usage:
//!   cargo run --release --bin scenario -- <名前|path.kif> [試行数=20] [戦略=estimator]
//!   cargo run --release --bin scenario -- <名前> diag [推定器数=10]
//!   cargo run --release --bin scenario -- <名前> continue [対局数=10] [手番側戦略] [相手戦略]
//!   cargo run --release --bin scenario -- suite [試行数=10] [戦略=estimator]
//!   共通フラグ: --ply N / --target USI / --diag 5g,4h （*scenario 行より優先）

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use tsuitate_bot::board::{Coord, make_usi_square, parse_usi_square};
use tsuitate_bot::estimator::Estimator;
use tsuitate_bot::observation::Observation;
use tsuitate_bot::protocol::{Color, Role};
use tsuitate_bot::scenario_core::{
    ChoiceStats, Replayed, Scenario, apply_scenario_fouls, build_estimator, choice_trials,
    choice_trials_batch, choice_trials_batch_range_with, clone_log, kifu_key, load_scenario,
    make_view, replay, scenarios_dir, side_idx, weighted_unique_particles,
};
use tsuitate_bot::shogi::{Outcome, Position, ShogiMove, parse_usi, unpromote_role};
use tsuitate_bot::strategy;

fn usage() -> &'static str {
    "usage:
  cargo run --release --bin scenario -- <名前|path.kif> [試行数=20] [戦略=estimator]
  cargo run --release --bin scenario -- <名前> diag [推定器数=10]
  cargo run --release --bin scenario -- <名前> continue [対局数=10] [手番側戦略] [相手戦略]
  cargo run --release --bin scenario -- suite [試行数=10] [戦略=estimator]
  cargo run --release --bin scenario -- merge <TRIAL行入りファイル...>
  共通フラグ: --ply N / --target USI / --diag 5g,4h
  suite用フラグ: --seed-base N / --tsv / --only 名前,..."
}

fn exit_usage(msg: &str) -> ! {
    eprintln!("{msg}");
    eprintln!("{}", usage());
    std::process::exit(1);
}

fn parse_u64_arg(label: &str, value: &str) -> u64 {
    value
        .parse()
        .unwrap_or_else(|_| exit_usage(&format!("{label} を数値として読めません: {value}")))
}

fn validate_strategy_name(name: &str) {
    if strategy::make_seeded(name, 0).is_none() {
        exit_usage(&format!("未知の戦略名です: {name}"));
    }
}

/// 手番側の一手の選択を試行する（choice_trials の逐次表示つきラッパー）
fn run_choice_trials(
    sc: &Scenario,
    rep: &Replayed,
    trials: u64,
    name: &str,
    verbose: bool,
) -> ChoiceStats {
    let side = rep.pos.turn();
    if verbose {
        println!("局面: {}", sc.desc);
        println!(
            "手番: {:?}（{}手目）/ ここまでの反則 先手{} 後手{} / 戦略: {name} / 試行 {trials} 回",
            side,
            sc.ply + 1,
            rep.fouls[0],
            rep.fouls[1]
        );
        println!();
    }

    let stats = choice_trials(rep, trials, name, |seed, accepted, foul_seq| {
        if verbose {
            let note = if accepted == sc.target {
                "（注目手）"
            } else if sc.bad.iter().any(|b| b == accepted) {
                "（不合格）"
            } else {
                ""
            };
            let foul_note = if foul_seq.is_empty() {
                String::new()
            } else {
                format!(" 反則: {}", foul_seq.join(", "))
            };
            println!("seed {seed:2}: {accepted}{note}{foul_note}");
        }
    });

    if verbose {
        println!();
        print_tally(sc, &stats, trials);
    }
    stats
}

/// 受理された手の内訳・不合格計・追加反則の表示（単発実行とバッチ実行で共通）
fn print_tally(sc: &Scenario, stats: &ChoiceStats, trials: u64) {
    println!("受理された手の内訳:");
    for (usi, n) in &stats.tally {
        let mark = if *usi == sc.target {
            " ← 注目手"
        } else if sc.bad.iter().any(|b| b == usi) {
            " ← 不合格"
        } else {
            ""
        };
        println!("  {usi}: {n}/{trials}{mark}");
    }
    if !sc.bad.is_empty() {
        println!("不合格計: {}/{trials}", stats.bad_hits(&sc.bad));
    }
    if !sc.scores.is_empty() {
        let (mean, unscored) = stats.mean_score(&sc.scores);
        println!("平均得点: {mean:.2}/10（未採点の選択 {unscored}/{trials}）");
    }
    println!("追加の反則の総数: {}", stats.total_fouls);
}

/// 複数シナリオの一括選択試行。同一棋譜×同一手番側のシナリオは推定器の構築を
/// 継ぎ足しで共有する（詳細は scenario_core::choice_trials_batch）
fn run_batch(specs: &[String], trials: u64, name: &str) {
    let mut loaded: Vec<(Scenario, Replayed)> = vec![];
    for spec in specs {
        let sc = match load_scenario(spec, None, None, None) {
            Ok(sc) => sc,
            Err(e) => {
                eprintln!("{spec}: 読み込み失敗: {e}");
                std::process::exit(1);
            }
        };
        let mut rep = replay(&sc.kifu, sc.ply);
        apply_scenario_fouls(&mut rep, &sc.fouls);
        if let Some(outcome) = rep.pos.outcome() {
            eprintln!("{spec}: ply={} の局面は終局しています（{outcome:?}）", sc.ply);
            std::process::exit(1);
        }
        loaded.push((sc, rep));
    }
    println!(
        "バッチ: {} 件 / 戦略 {name} / 各 {trials} 試行（同一棋譜×同一手番は構築を共有）",
        loaded.len()
    );
    println!();
    let items: Vec<(&Scenario, &Replayed)> = loaded.iter().map(|(sc, rep)| (sc, rep)).collect();
    let width = loaded.iter().map(|(sc, _)| sc.name.len()).max().unwrap_or(0);
    let stats = choice_trials_batch(&items, trials, name, |i, seed, accepted, foul_seq| {
        let sc = items[i].0;
        let note = if accepted == sc.target {
            "（注目手）"
        } else if sc.bad.iter().any(|b| b == accepted) {
            "（不合格）"
        } else {
            ""
        };
        let foul_note = if foul_seq.is_empty() {
            String::new()
        } else {
            format!(" 反則: {}", foul_seq.join(", "))
        };
        println!("{:<width$} seed {seed:2}: {accepted}{note}{foul_note}", sc.name);
    });
    for ((sc, _), st) in loaded.iter().zip(&stats) {
        println!();
        println!("===== {}（{}手目） =====", sc.name, sc.ply + 1);
        print_tally(sc, st, trials);
    }
}

/// シナリオ局面の**真実の盤面**を表示する（レビュー中に「その駒は本当に
/// 当たっているのか」を確かめるため。審判視点なので bot には見えない情報）。
/// 手番側の各駒について、相手の利き枚数と自分の紐の枚数も併記する
fn print_truth_board(sc: &Scenario, rep: &Replayed) {
    let side = rep.pos.turn();
    println!("局面: {}", sc.desc);
    println!(
        "手番: {side:?}（{}手目）/ ここまでの反則 先手{} 後手{}",
        sc.ply + 1,
        rep.fouls[0],
        rep.fouls[1]
    );
    println!();
    println!("    9   8   7   6   5   4   3   2   1");
    for rank in 1..=9 {
        let mut row = String::new();
        for file in (1..=9).rev() {
            let c = Coord { file, rank };
            row.push_str(&match rep.pos.piece_at(c) {
                Some(p) => {
                    let s = role_char(p.role);
                    format!("{:>4}", if p.color == Color::Gote { format!("v{s}") } else { s.into() })
                }
                None => "   .".to_string(),
            });
        }
        println!("{rank}: {row}");
    }
    for color in [Color::Sente, Color::Gote] {
        let mut items: Vec<_> = rep
            .pos
            .hand_map(color)
            .into_iter()
            .filter(|(_, n)| *n > 0)
            .collect();
        items.sort_by_key(|(r, _)| format!("{r:?}"));
        println!(
            "{color:?} 持ち駒: {}",
            items
                .iter()
                .map(|(r, n)| format!("{}x{n}", role_char(*r)))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    println!();
    println!("手番側（{side:?}）の駒の当たり（相手の利き枚数 > 0 のものだけ）:");
    let mut any = false;
    for (sq, p) in rep.pos.pieces() {
        if p.color != side {
            continue;
        }
        let att = rep.pos.attack_count(sq, side.other());
        if att == 0 {
            continue;
        }
        any = true;
        println!(
            "  {} {:?}: 相手の利き{}枚 / 自分の紐{}枚",
            make_usi_square(sq),
            p.role,
            att,
            rep.pos.attack_count(sq, side)
        );
    }
    if !any {
        println!("  （当たっている駒は無い）");
    }
}

fn role_char(role: Role) -> &'static str {
    match role {
        Role::Pawn => "P",
        Role::Lance => "L",
        Role::Knight => "N",
        Role::Silver => "S",
        Role::Gold => "G",
        Role::Bishop => "B",
        Role::Rook => "R",
        Role::King => "K",
        Role::Tokin => "+P",
        Role::Promotedlance => "+L",
        Role::Promotedknight => "+N",
        Role::Promotedsilver => "+S",
        Role::Horse => "+B",
        Role::Dragon => "+R",
    }
}

/// 粒子集合の診断: 王手駒の分布・相手玉位置の分布・注目マスへの相手利き枚数。
/// 粒子は複製で偏るので指紋でユニーク化して数える（strategy.rs の評価と同じ発想）。
/// 玉位置・利き枚数は厳密整合粒子（penalty=0）だけで集計する
fn diagnose_particles(sc: &Scenario, rep: &Replayed, n_estimators: u64) {
    let side = rep.pos.turn();
    // 既定の思考予算 2000ms 相当（SearchBudget::from_ms は非公開なので直書き）。
    // SCENARIO_SCALE_MULT で粒子数・再生成予算を実運用の何倍にも増やせる
    // （枯渇が予算不足か構造的かの切り分け用）
    let mult: f64 = std::env::var("SCENARIO_SCALE_MULT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let scale = 2000.0 / 900.0 * mult;
    let king_sq = rep.pos.king_square(side).expect("手番側の玉");
    let diag_sqs: Vec<_> = sc
        .diag_squares
        .iter()
        .map(|s| {
            (
                s.clone(),
                parse_usi_square(s).expect("diag のマス解析失敗"),
            )
        })
        .collect();

    // 集計は評価側と同じ重み（0.5^penalty × 推定器ごとに正規化した exp(logw)）で行う
    let mut checker_tally: HashMap<String, f64> = HashMap::new();
    let mut checker_mass = 0.0f64;
    let mut opp_king_tally: HashMap<String, f64> = HashMap::new();
    let mut all_king_tally: HashMap<String, f64> = HashMap::new();
    let mut all_king_mass = 0.0f64;
    // マスごとの相手利き枚数（0,1,2,3+）の重み質量
    let mut cover_tally: Vec<[f64; 4]> = vec![[0.0; 4]; diag_sqs.len()];
    // マスごとの「そこにいる相手駒の駒種」の重み質量（捕獲期待値の内訳）
    let mut occ_tally: Vec<HashMap<String, f64>> = vec![HashMap::new(); diag_sqs.len()];
    // 同（taint 粒子。厳密が全滅した決定点ではこちらが評価の実体になる）
    let mut taint_occ_tally: Vec<HashMap<String, f64>> = vec![HashMap::new(); diag_sqs.len()];
    // taint 粒子だけでの同集計（strategy.rs の taint_particles/taint_square_coverage
    // と同じ重み規約 = 0.5^(taint-1) 減衰・taint<=6・taint内max_lwで正規化）。
    // 「局所被覆度ビリーフ」が真実とどれだけ一致するかの直接測定
    let mut taint_cover_tally: Vec<[f64; 4]> = vec![[0.0; 4]; diag_sqs.len()];
    let mut taint_cover_mass = 0.0f64;
    let mut strict_mass = 0.0f64;
    let mut total_unique = 0u32;
    let mut strict_unique = 0u32;
    for seed in 0..n_estimators {
        // 構築は scenario_core と共有（GUI の玉位置ビリーフと同じ粒子集合になる）
        let est = build_estimator(rep, seed, scale, |est: &Estimator, turn_no: usize| {
            if seed != 0 {
                return;
            }
            let strict = est
                .info_miss()
                .iter()
                .zip(est.phys_taint())
                .filter(|&(&m, &t)| m == 0 && t == 0)
                .count();
            let taint = est.phys_taint().iter().filter(|&&t| t > 0).count();
            let (repaired, revived) = est.rejuv_stats();
            eprintln!(
                "  [seed0] 手番{turn_no}: 粒子 {} (厳密{} taint{} healthy={} 修復{} 復活{})",
                est.particles().len(),
                strict,
                taint,
                est.healthy(),
                repaired,
                revived,
            );
        });
        if seed == 0 {
            let (repaired, revived) = est.rejuv_stats();
            eprintln!(
                "  [seed0] 最終: 粒子 {} (healthy={} 修復{} 復活{})",
                est.particles().len(),
                est.healthy(),
                repaired,
                revived,
            );
            let fails = est.fail_report();
            if !fails.is_empty() {
                let top: Vec<String> = fails
                    .iter()
                    .take(12)
                    .map(|(i, c)| format!("c{i}×{c}"))
                    .collect();
                eprintln!("  [seed0] 失敗制約: {}", top.join(" "));
            }
        }
        // ユニーク粒子と評価重み（指紋ごとの正規化 Σexp(logw)。phys_taint>0 は
        // 非厳密扱い: 王手駒の分布には出るが玉位置・利きの「厳密のみ」集計から
        // 外れる）。規約は scenario_core と共有＝GUI の玉位置ビリーフと同じ
        for (pp, w, strict) in weighted_unique_particles(&est) {
            let penalty = u8::from(!strict);
            total_unique += 1;
            // taint 込みの全粒子での玉位置分布（ε_phys の信念の質の診断用。
            // 厳密のみの分布とは別枠で集計する）
            if let Some(sq) = pp.king_square(side.other()) {
                *all_king_tally.entry(make_usi_square(sq)).or_insert(0.0) += w;
                all_king_mass += w;
            }
            if rep.pos.in_check(side) {
                let checkers: Vec<String> = pp
                    .pieces()
                    .filter(|(from, pc)| pc.color == side.other() && pp.attacks(*from, king_sq))
                    .map(|(from, pc)| format!("{:?}@{}", pc.role, make_usi_square(from)))
                    .collect();
                let key = if checkers.is_empty() {
                    "（王手なし）".to_string()
                } else {
                    checkers.join("+")
                };
                *checker_tally.entry(key).or_insert(0.0) += w;
                checker_mass += w;
            }
            if penalty > 0 {
                continue;
            }
            strict_unique += 1;
            strict_mass += w;
            if let Some(sq) = pp.king_square(side.other()) {
                *opp_king_tally.entry(make_usi_square(sq)).or_insert(0.0) += w;
            }
            for (i, (_, sq)) in diag_sqs.iter().enumerate() {
                let n = pp
                    .pieces()
                    .filter(|(from, pc)| {
                        pc.color == side.other()
                            && pc.role != Role::King
                            && pp.attacks(*from, *sq)
                    })
                    .count();
                cover_tally[i][n.min(3)] += w;
                // そのマスに何の駒がいると思っているか（捕獲期待値の内訳）。
                // 「取られたマス」は占有が確定するので駒種の分布だけが問題になる
                let key = match pp.piece_at(*sq) {
                    Some(pc) if pc.color == side.other() => format!("{:?}", pc.role),
                    Some(_) => "自駒".to_string(),
                    None => "空".to_string(),
                };
                *occ_tally[i].entry(key).or_insert(0.0) += w;
            }
        }
        // taint 粒子だけの被覆度集計（strategy.rs の taint_particles/
        // taint_square_coverage と**同じ規約**で計算する。診断が本番と
        // 食い違うと較正の数字が無意味になる — codex レビュー指摘:
        // ①玉も利き枚数に含める（本番の taint_square_coverage は role
        // フィルタなし）②TAINT_POOL_CAP と同じ上限を適用 ③clean/soft 粒子の
        // fingerprint を taint 専用マップへ誤って引いてパニックしない
        // （タプルで直接持ち運び、HashMap の再引きをしない）
        const TAINT_VOTE_MAX: u8 = 6;
        const TAINT_POOL_CAP: usize = 256;
        let max_taint_lw = est
            .log_weights()
            .iter()
            .zip(est.phys_taint())
            .filter(|&(_, &t)| t > 0 && t <= TAINT_VOTE_MAX)
            .map(|(&lw, _)| lw)
            .fold(f64::MIN, f64::max);
        if max_taint_lw != f64::MIN {
            let mut seen_t: HashMap<u64, usize> = HashMap::new();
            let mut taint_uniques: Vec<(&Position, f64)> = vec![];
            for ((pp, &t), &lw) in est
                .particles()
                .iter()
                .zip(est.phys_taint())
                .zip(est.log_weights())
            {
                if t == 0 || t > TAINT_VOTE_MAX {
                    continue;
                }
                let w = (lw - max_taint_lw).exp() * 0.5f64.powi(i32::from(t) - 1);
                match seen_t.entry(pp.fingerprint()) {
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert(taint_uniques.len());
                        taint_uniques.push((pp, w));
                    }
                    std::collections::hash_map::Entry::Occupied(e) => {
                        taint_uniques[*e.get()].1 += w;
                    }
                }
            }
            if taint_uniques.len() > TAINT_POOL_CAP {
                taint_uniques.select_nth_unstable_by(TAINT_POOL_CAP, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                taint_uniques.truncate(TAINT_POOL_CAP);
            }
            for (pp, w) in taint_uniques {
                taint_cover_mass += w;
                for (i, (_, sq)) in diag_sqs.iter().enumerate() {
                    let n = pp
                        .pieces()
                        .filter(|(from, pc)| pc.color == side.other() && pp.attacks(*from, *sq))
                        .count();
                    taint_cover_tally[i][n.min(3)] += w;
                    let key = match pp.piece_at(*sq) {
                        Some(pc) if pc.color == side.other() => format!("{:?}", pc.role),
                        Some(_) => "自駒".to_string(),
                        None => "空".to_string(),
                    };
                    *taint_occ_tally[i].entry(key).or_insert(0.0) += w;
                }
            }
        }
    }

    println!(
        "粒子診断: 推定器 {n_estimators} 個ぶんのユニーク粒子 {total_unique} 個（うち厳密整合 {strict_unique}。\
         集計は評価重み = 指紋ごとの正規化 Σexp(logw)（ソフト減衰は logw 課金済み）。玉位置・利きは厳密のみ）"
    );
    // taint 込みの全粒子での相手玉位置（真実との突き合わせ。ε_phys の
    // 「玉位置の信念は保てているか」の直接測定）
    let truth_king = rep
        .pos
        .king_square(side.other())
        .map(make_usi_square)
        .unwrap_or_else(|| "-".into());
    println!();
    println!("相手玉の位置分布（taint 込み全粒子、重み付き。真実 = {truth_king}）:");
    let mut sorted_all: Vec<_> = all_king_tally.into_iter().collect();
    sorted_all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (sq, m) in sorted_all.iter().take(8) {
        let mark = if *sq == truth_king { " ←真実" } else { "" };
        println!("  {sq}: {:.1}%{mark}", 100.0 * m / all_king_mass.max(1e-12));
    }
    if rep.pos.in_check(side) && checker_mass > 0.0 {
        println!();
        println!("王手駒の分布（粒子内で手番側の玉に利いている相手駒。重み付き）:");
        let mut sorted: Vec<_> = checker_tally.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (key, m) in sorted {
            let pct = 100.0 * m / checker_mass.max(1e-12);
            if pct < 0.05 {
                continue;
            }
            println!("  {key}: {pct:.1}%");
        }
    }
    if strict_unique > 0 {
        println!();
        println!("相手玉の位置分布（上位、重み付き）:");
        let mut sorted: Vec<_> = opp_king_tally.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (sq, m) in sorted.iter().take(8) {
            println!("  {sq}: {:.1}%", 100.0 * m / strict_mass.max(1e-12));
        }
    }
    // マスへの相手利き枚数: 真実 vs 厳密粒子 vs taint粒子（局所被覆度ビリーフの
    // 較正）。全滅ケース（strict_unique==0）でこそ taint 側の較正が主役になる
    for (i, (name, sq)) in diag_sqs.iter().enumerate() {
        // 真実の被覆度（審判が持つ全手順から直接計算。ground truth）。
        // 厳密粒子の集計は玉を除く、taint（本番の taint_square_coverage と
        // 同じ規約）は玉を含む — 列ごとに対応する真実を分けて表示する
        // （codex レビュー指摘: 診断と本番の規約不一致は較正の数字を無意味にする）
        let truth_no_king = rep
            .pos
            .pieces()
            .filter(|(from, pc)| {
                pc.color == side.other() && pc.role != Role::King && rep.pos.attacks(*from, *sq)
            })
            .count();
        let truth_with_king = rep
            .pos
            .pieces()
            .filter(|(from, pc)| pc.color == side.other() && rep.pos.attacks(*from, *sq))
            .count();
        println!();
        // そのマスの駒種ビリーフ（真実と並べる）。「取られたマス」のように占有が
        // 確定しているマスでは、捕獲期待値の誤差は丸ごと駒種の誤りになる。
        // **厳密粒子が全滅した決定点では taint 側が評価の実体**になるので両方出す
        let truth_occ = match rep.pos.piece_at(*sq) {
            Some(pc) if pc.color == side.other() => format!("{:?}", pc.role),
            Some(_) => "自駒".to_string(),
            None => "空".to_string(),
        };
        let show_occ = |tally: &HashMap<String, f64>, mass: f64| -> String {
            let mut occ: Vec<(String, f64)> = tally.clone().into_iter().collect();
            occ.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            occ.iter()
                .take(6)
                .map(|(k, v)| {
                    let mark = if *k == truth_occ { "←真実" } else { "" };
                    format!("{k} {:.1}%{mark}", 100.0 * v / mass.max(1e-12))
                })
                .collect::<Vec<_>>()
                .join(" / ")
        };
        if strict_unique > 0 {
            println!(
                "{name} の駒種ビリーフ（厳密粒子。真実={truth_occ}）: {}",
                show_occ(&occ_tally[i], strict_mass)
            );
        }
        if taint_cover_mass > 0.0 {
            println!(
                "{name} の駒種ビリーフ（taint粒子。真実={truth_occ}）: {}",
                show_occ(&taint_occ_tally[i], taint_cover_mass)
            );
        }
        println!("{name} への相手利き枚数:");
        if strict_unique > 0 {
            let t = &cover_tally[i];
            println!(
                "  厳密粒子（玉を除く、真実={truth_no_king}枚）: 0枚 {:.1}% / 1枚 {:.1}% / 2枚 {:.1}% / 3枚以上 {:.1}%",
                100.0 * t[0] / strict_mass.max(1e-12),
                100.0 * t[1] / strict_mass.max(1e-12),
                100.0 * t[2] / strict_mass.max(1e-12),
                100.0 * t[3] / strict_mass.max(1e-12),
            );
        } else {
            println!("  厳密粒子: なし（フィルタ死）");
        }
        if taint_cover_mass > 0.0 {
            let tt = &taint_cover_tally[i];
            let expected: f64 = (0..4).map(|k| k as f64 * tt[k]).sum::<f64>() / taint_cover_mass;
            println!(
                "  taint粒子（玉を含む、本番と同じ規約。真実={truth_with_king}枚）: 0枚 {:.1}% / 1枚 {:.1}% / 2枚 {:.1}% / 3枚以上 {:.1}%（期待値 {:.2}枚）",
                100.0 * tt[0] / taint_cover_mass,
                100.0 * tt[1] / taint_cover_mass,
                100.0 * tt[2] / taint_cover_mass,
                100.0 * tt[3] / taint_cover_mass,
                expected,
            );
        }
    }
}

/// 局面から bot 同士で終局まで指し継ぐ（selfplay.rs の裁定を簡略移植。時計なし）。
/// 反則数はリプレイ時点から引き継ぐ（foul_limit 10 は累計）
fn continue_games(sc: &Scenario, rep: &Replayed, games: u64, name_me: &str, name_opp: &str) {
    let me = rep.pos.turn();
    println!("局面: {}", sc.desc);
    println!(
        "手番側 {:?} = {name_me} / 相手側 = {name_opp} / ここまでの反則 先手{} 後手{} / {games} 局",
        me, rep.fouls[0], rep.fouls[1]
    );
    println!();

    let mut wins_me = 0u32;
    let mut reasons: HashMap<String, u32> = HashMap::new();
    let mut first_moves: HashMap<String, u32> = HashMap::new();
    let mut win_plies: Vec<u32> = vec![];
    for seed in 0..games {
        let mut strats = [
            strategy::make_seeded(
                if me == Color::Sente { name_me } else { name_opp },
                seed ^ 0x5E17E_u64,
            )
            .expect("未知の戦略名"),
            strategy::make_seeded(
                if me == Color::Gote { name_me } else { name_opp },
                seed ^ 0x607E_u64,
            )
            .expect("未知の戦略名"),
        ];
        let mut pos = rep.pos.clone();
        let mut logs = [clone_log(&rep.logs[0]), clone_log(&rep.logs[1])];
        for (i, strat) in strats.iter_mut().enumerate() {
            let color = if i == 0 { Color::Sente } else { Color::Gote };
            strategy::prewarm_strategy(&mut **strat, &make_view(&rep.pos, color, &rep.fouls), &logs[i]);
        }
        let mut fouls = rep.fouls;
        let mut foul_tried: [HashSet<String>; 2] = [HashSet::new(), HashSet::new()];
        let mut plies = rep.plies;
        let mut first_move: Option<String> = None;

        let (winner, reason): (Option<Color>, String) = loop {
            // 足切り（*scenario limit=N）: 必勝局面の遂行実験では、決めるべき
            // 手数で決められなかった時点で不合格（引き分け=負け扱いで集計）
            if plies >= sc.limit {
                break (None, "limit".into());
            }
            let side = pos.turn();
            let i = side_idx(side);
            let view = make_view(&pos, side, &fouls);
            let Some(usi) = strats[i].choose(&view, &logs[i], &foul_tried[i]) else {
                break (Some(side.other()), "resign".into());
            };
            let legal = parse_usi(&usi).is_some_and(|mv| pos.is_legal(&mv));
            if !legal {
                fouls[i] += 1;
                foul_tried[i].insert(usi.clone());
                logs[i].record(Observation::MyFoul {
                    move_number: pos.move_number(),
                    usi,
                });
                logs[1 - i].record(Observation::OpponentFoul { count: fouls[i] });
                if fouls[i] >= 10 {
                    break (Some(side.other()), "foul_limit".into());
                }
                continue;
            }
            let mv = parse_usi(&usi).unwrap();
            let captured = pos.play_unchecked(&mv);
            plies += 1;
            foul_tried[i].clear();
            if side == me && first_move.is_none() {
                first_move = Some(usi.clone());
            }
            let move_number = pos.move_number();
            let captured_sq = captured.map(|_| match mv {
                ShogiMove::Board { to, .. } => make_usi_square(to),
                ShogiMove::Drop { .. } => unreachable!(),
            });
            logs[i].record(Observation::MyMove {
                move_number,
                usi,
                captured: captured.map(unpromote_role),
            });
            logs[1 - i].record(Observation::OpponentMoved {
                move_number,
                captured_my_piece_at: captured_sq,
            });
            if pos.in_check(pos.turn()) {
                let in_check = pos.turn();
                for log in logs.iter_mut() {
                    log.record(Observation::Check { in_check });
                }
            }
            match pos.outcome() {
                Some(Outcome::Checkmate { winner }) => break (Some(winner), "checkmate".into()),
                Some(Outcome::Stalemate { winner }) => break (Some(winner), "stalemate".into()),
                None => {}
            }
        };

        let first = first_move.unwrap_or_else(|| "-".into());
        let won = winner == Some(me);
        if won {
            wins_me += 1;
            win_plies.push(plies - rep.plies);
        }
        println!(
            "game {seed:2}: 初手 {first}{} → {} ({reason}, +{}手, 反則 先手{} 後手{})",
            if first == sc.target {
                "（注目手）"
            } else if sc.bad.iter().any(|b| *b == first) {
                "（不合格）"
            } else {
                ""
            },
            match winner {
                Some(c) if c == me => "勝ち",
                Some(_) => "負け",
                None => "引き分け",
            },
            plies - rep.plies,
            fouls[0],
            fouls[1],
        );
        *reasons.entry(reason).or_insert(0) += 1;
        *first_moves.entry(first).or_insert(0) += 1;
    }

    println!();
    println!("手番側の勝ち: {wins_me}/{games}");
    if !win_plies.is_empty() {
        win_plies.sort_unstable();
        println!(
            "勝ち局の追加手数: 中央値 {} / 最短 {} / 最長 {}",
            win_plies[win_plies.len() / 2],
            win_plies[0],
            win_plies[win_plies.len() - 1]
        );
    }
    let mut sorted: Vec<_> = first_moves.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    println!("初手の内訳:");
    for (usi, n) in sorted {
        let mark = if usi == sc.target {
            " ← 注目手"
        } else if sc.bad.iter().any(|b| *b == usi) {
            " ← 不合格"
        } else {
            ""
        };
        println!("  {usi}: {n}/{games}{mark}");
    }
    let mut sorted: Vec<_> = reasons.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    println!("終局理由:");
    for (r, n) in sorted {
        println!("  {r}: {n}/{games}");
    }
}

/// scenarios/*.kif を全部回して注目手一致率を表にする（回帰テスト用）。
///
/// `seed_base` でシード列を `seed_base..seed_base+trials` にずらせる
/// （CI のシードシャード実行: 各ランナーが trials=1 で別シードを担当し、
/// merge モードで合算する）。`tsv` なら試行ごとの機械可読行
/// `TRIAL\t名前\tシード\t受理手\t反則数` も出力する。
/// `only` はシナリオ名（ファイル名から .kif を除いたもの）のフィルタ。
///
/// (棋譜, 手番側) のグループごとにワーカースレッドへ分配して並列実行する
/// （bin/tune のシナリオ目的関数と同じ分割。ワーカー数は物理コア数まで =
/// choose の壁時計予算を奪い合わない範囲。`SCENARIO_WORKERS` で上書き可）
fn run_suite(trials: u64, name: &str, seed_base: u64, tsv: bool, only: &[String]) {
    let dir = scenarios_dir();
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{} を読めません: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "kif"))
        .filter(|p| {
            only.is_empty()
                || p.file_stem()
                    .is_some_and(|s| only.iter().any(|o| o == &s.to_string_lossy()))
        })
        .collect();
    paths.sort();
    println!(
        "スイート: {} 件 / 戦略 {name} / 各 {trials} 試行 seed {seed_base}〜（同一棋譜×同一手番は構築を共有）",
        paths.len()
    );
    println!();
    let mut loaded: Vec<(Scenario, Replayed)> = vec![];
    for path in paths {
        let sc = match load_scenario(&path.to_string_lossy(), None, None, None) {
            Ok(sc) => sc,
            Err(e) => {
                println!("{}: 読み込み失敗: {e}", path.display());
                continue;
            }
        };
        let mut rep = replay(&sc.kifu, sc.ply);
        apply_scenario_fouls(&mut rep, &sc.fouls);
        loaded.push((sc, rep));
    }

    // (棋譜, 手番側) のパーティション（choice_trials_batch の内部グループと同じ）。
    // 重いグループ（Σply 大）から着手してワーカーの負荷を均す
    let mut parts: Vec<((u64, tsuitate_bot::protocol::Color), Vec<usize>)> = vec![];
    for (i, (sc, rep)) in loaded.iter().enumerate() {
        let key = (kifu_key(&sc.kifu), rep.pos.turn());
        match parts.iter_mut().find(|(k, _)| *k == key) {
            Some((_, v)) => v.push(i),
            None => parts.push((key, vec![i])),
        }
    }
    parts.sort_by_key(|(_, idxs)| {
        std::cmp::Reverse(idxs.iter().map(|&i| loaded[i].0.ply).max().unwrap_or(0))
    });

    let workers = std::env::var("SCENARIO_WORKERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(std::num::NonZeroUsize::get)
                .unwrap_or(4)
        })
        .clamp(1, parts.len().max(1));

    let total = loaded.len() as u64 * trials;
    let done = std::sync::atomic::AtomicU64::new(0);
    let next = std::sync::atomic::AtomicUsize::new(0);
    // パーティションごとの結果: (統計, TSV行 = (グローバルindex, シード, 受理手, 反則数))
    type Rows = Vec<(usize, u64, String, u32)>;
    let results: Vec<std::sync::Mutex<Option<(Vec<ChoiceStats>, Rows)>>> =
        parts.iter().map(|_| std::sync::Mutex::new(None)).collect();
    let factory: &(dyn Fn(u64) -> Box<dyn strategy::Strategy + Send> + Sync) =
        &|s| strategy::make_seeded(name, s).expect("未知の戦略名");
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                loop {
                    let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if k >= parts.len() {
                        break;
                    }
                    let idxs = &parts[k].1;
                    let items: Vec<(&Scenario, &Replayed)> =
                        idxs.iter().map(|&i| (&loaded[i].0, &loaded[i].1)).collect();
                    let mut rows: Rows = vec![];
                    let stats = choice_trials_batch_range_with(
                        &items,
                        seed_base..seed_base + trials,
                        factory,
                        |il, seed, accepted, foul_seq| {
                            rows.push((idxs[il], seed, accepted.to_string(), foul_seq.len() as u32));
                            let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                            eprint!("\r{d}/{total} 試行");
                        },
                    );
                    *results[k].lock().unwrap() = Some((stats, rows));
                }
            });
        }
    });
    eprintln!();

    let mut stats_by_idx: Vec<Option<ChoiceStats>> = (0..loaded.len()).map(|_| None).collect();
    let mut all_rows: Rows = vec![];
    for ((_, idxs), cell) in parts.iter().zip(results) {
        let (stats, rows) = cell.into_inner().unwrap().expect("パーティション未評価");
        for (&i, st) in idxs.iter().zip(stats) {
            stats_by_idx[i] = Some(st);
        }
        all_rows.extend(rows);
    }

    for ((sc, _), stats) in loaded.iter().zip(&stats_by_idx) {
        let stats = stats.as_ref().expect("全シナリオが評価されている");
        let hits = stats.target_hits(&sc.target);
        let others: Vec<String> = stats
            .tally
            .iter()
            .filter(|(usi, _)| *usi != sc.target)
            .take(3)
            .map(|(usi, n)| format!("{usi}×{n}"))
            .collect();
        let bad_note = if sc.bad.is_empty() {
            String::new()
        } else {
            format!(" 不合格計 {}/{trials}", stats.bad_hits(&sc.bad))
        };
        let score_note = if sc.scores.is_empty() {
            String::new()
        } else {
            let (mean, unscored) = stats.mean_score(&sc.scores);
            format!(" 得点 {mean:.2}/10(未採点{unscored})")
        };
        println!(
            "{:<12} {}手目 注目手 {:<6} {hits}/{trials}{bad_note}{score_note} 反則{} 他: {}",
            sc.name,
            sc.ply + 1,
            sc.target,
            stats.total_fouls,
            others.join(" ")
        );
    }
    if tsv {
        println!();
        all_rows.sort();
        for (i, seed, accepted, fouls) in &all_rows {
            println!("TRIAL\t{}\t{seed}\t{accepted}\t{fouls}", loaded[*i].0.name);
        }
    }
}

/// シードシャード実行の TSV（`suite --tsv` の `TRIAL` 行）を合算して
/// suite と同じ集計の markdown 表を出す（scenario.yml の aggregate ジョブ用）。
/// 集計の意味論（target_hits / bad_hits / mean_score）は ChoiceStats を
/// そのまま使うので suite と完全に一致する
fn run_merge(files: &[String]) {
    // シナリオ名 → (受理手カウント, 反則合計, 試行数)
    let mut agg: std::collections::BTreeMap<String, (HashMap<String, u32>, u32, u32)> =
        std::collections::BTreeMap::new();
    for f in files {
        let text = std::fs::read_to_string(f)
            .unwrap_or_else(|e| exit_usage(&format!("{f} を読めません: {e}")));
        for line in text.lines() {
            let mut cols = line.split('\t');
            if cols.next() != Some("TRIAL") {
                continue;
            }
            let (Some(name), Some(_seed), Some(accepted), Some(fouls)) =
                (cols.next(), cols.next(), cols.next(), cols.next())
            else {
                exit_usage(&format!("TRIAL 行を読めません: {line}"));
            };
            let fouls: u32 = fouls
                .trim()
                .parse()
                .unwrap_or_else(|_| exit_usage(&format!("TRIAL 行の反則数を読めません: {line}")));
            let e = agg.entry(name.to_string()).or_default();
            *e.0.entry(accepted.to_string()).or_insert(0) += 1;
            e.1 += fouls;
            e.2 += 1;
        }
    }
    if agg.is_empty() {
        exit_usage("TRIAL 行が見つかりません（suite --tsv の出力を渡してください）");
    }
    let mut bad_total = 0u32;
    let mut bad_n = 0u32;
    let mut score_sum = 0.0f64;
    let mut score_n = 0u32;
    let mut unscored_total = 0u32;
    println!("| シナリオ | 手目 | 注目手 | 選択 | 不合格 | 得点/10 | 反則 | 他の選択 |");
    println!("|---|---|---|---|---|---|---|---|");
    for (sc_name, (tally, fouls, trials)) in &agg {
        let sc = match load_scenario(sc_name, None, None, None) {
            Ok(sc) => sc,
            Err(e) => {
                println!("| {sc_name} | - | 読み込み失敗: {e} | - | - | - | - | - |");
                continue;
            }
        };
        let mut tally: Vec<(String, u32)> = tally.clone().into_iter().collect();
        tally.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let stats = ChoiceStats {
            tally,
            total_fouls: *fouls,
        };
        let hits = stats.target_hits(&sc.target);
        let bad_cell = if sc.bad.is_empty() {
            "-".to_string()
        } else {
            let b = stats.bad_hits(&sc.bad);
            bad_total += b;
            bad_n += 1;
            format!("{b}/{trials}")
        };
        let score_cell = if sc.scores.is_empty() {
            "-".to_string()
        } else {
            let (mean, unscored) = stats.mean_score(&sc.scores);
            score_sum += mean;
            score_n += 1;
            unscored_total += unscored;
            format!("{mean:.2}(未採点{unscored})")
        };
        let others: Vec<String> = stats
            .tally
            .iter()
            .filter(|(usi, _)| *usi != sc.target)
            .take(3)
            .map(|(usi, n)| format!("{usi}×{n}"))
            .collect();
        println!(
            "| {} | {} | {} | {hits}/{trials} | {bad_cell} | {score_cell} | {} | {} |",
            sc.name,
            sc.ply + 1,
            sc.target,
            stats.total_fouls,
            others.join(" ")
        );
    }
    println!();
    if bad_n > 0 {
        println!("**不合格計の合計: {bad_total}（{bad_n}件）**");
    }
    if score_n > 0 {
        println!(
            "**平均得点の全体平均: {:.3}/10（{score_n}件、未採点合計{unscored_total}）**",
            score_sum / f64::from(score_n)
        );
    }
}

fn main() {
    // フラグ（--ply N / --target USI / --diag 5g,4h / suite用 --seed-base N /
    // --tsv / --only 名前,...）を先に抜き取る
    let mut ply_flag: Option<usize> = None;
    let mut target_flag: Option<String> = None;
    let mut diag_flag: Option<String> = None;
    let mut seed_base: u64 = 0;
    let mut tsv = false;
    let mut only: Vec<String> = vec![];
    let mut args: Vec<String> = vec![];
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--ply" => {
                let Some(value) = raw.get(i + 1) else {
                    exit_usage("--ply には値が必要です");
                };
                let ply = value
                    .parse()
                    .unwrap_or_else(|_| exit_usage(&format!("--ply を数値として読めません: {value}")));
                ply_flag = Some(ply);
                i += 2;
            }
            "--target" => {
                let Some(value) = raw.get(i + 1) else {
                    exit_usage("--target には値が必要です");
                };
                if parse_usi(value).is_none() {
                    exit_usage(&format!("--target をUSIとして読めません: {value}"));
                }
                target_flag = Some(value.clone());
                i += 2;
            }
            "--diag" => {
                let Some(value) = raw.get(i + 1) else {
                    exit_usage("--diag には値が必要です");
                };
                diag_flag = Some(value.clone());
                i += 2;
            }
            "--seed-base" => {
                let Some(value) = raw.get(i + 1) else {
                    exit_usage("--seed-base には値が必要です");
                };
                seed_base = parse_u64_arg("--seed-base", value);
                i += 2;
            }
            "--tsv" => {
                tsv = true;
                i += 1;
            }
            "--only" => {
                let Some(value) = raw.get(i + 1) else {
                    exit_usage("--only には値が必要です");
                };
                only = value
                    .split([',', ' '])
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect();
                i += 2;
            }
            _ => {
                args.push(raw[i].clone());
                i += 1;
            }
        }
    }

    let spec = args.first().map(String::as_str).unwrap_or("keima");
    if spec == "suite" {
        let trials = args.get(1).map(|s| parse_u64_arg("試行数", s)).unwrap_or(10);
        let name = args.get(2).map(String::as_str).unwrap_or("estimator");
        validate_strategy_name(name);
        run_suite(trials, name, seed_base, tsv, &only);
        return;
    }
    if spec == "merge" {
        if args.len() < 2 {
            exit_usage("merge には TRIAL 行入りのファイルを1つ以上指定してください");
        }
        run_merge(&args[1..]);
        return;
    }
    if spec == "batch" {
        // batch <名前...> [試行数] [戦略名]（数値は試行数、戦略として解釈できる
        // 名前は戦略、それ以外はシナリオ名として扱う）
        let mut specs: Vec<String> = vec![];
        let mut trials = 20u64;
        let mut name = "estimator".to_string();
        for a in &args[1..] {
            if let Ok(n) = a.parse::<u64>() {
                trials = n;
            } else if strategy::make(a).is_some() {
                name = a.clone();
            } else {
                specs.push(a.clone());
            }
        }
        if specs.is_empty() {
            exit_usage("batch にはシナリオ名を1つ以上指定してください");
        }
        run_batch(&specs, trials, &name);
        return;
    }

    let sc = match load_scenario(spec, ply_flag, target_flag, diag_flag) {
        Ok(sc) => sc,
        Err(e) => {
            eprintln!("{e}");
            eprintln!(
                "シナリオは {} の .kif か、.kif ファイルパスで指定してください",
                scenarios_dir().display()
            );
            std::process::exit(1);
        }
    };
    let mut rep = replay(&sc.kifu, sc.ply);
    apply_scenario_fouls(&mut rep, &sc.fouls);
    if let Some(outcome) = rep.pos.outcome() {
        eprintln!(
            "ply={} の局面は終局しています（{outcome:?}）。--ply を見直してください",
            sc.ply
        );
        std::process::exit(1);
    }

    match args.get(1).map(String::as_str) {
        Some("board") => print_truth_board(&sc, &rep),
        Some("diag") => {
            let n = args
                .get(2)
                .map(|s| parse_u64_arg("推定器数", s))
                .unwrap_or(10);
            diagnose_particles(&sc, &rep, n);
        }
        Some("continue") => {
            let games = args
                .get(2)
                .map(|s| parse_u64_arg("対局数", s))
                .unwrap_or(10);
            let name_me = args.get(3).map(String::as_str).unwrap_or("estimator");
            let name_opp = args.get(4).map(String::as_str).unwrap_or("estimator");
            validate_strategy_name(name_me);
            validate_strategy_name(name_opp);
            continue_games(&sc, &rep, games, name_me, name_opp);
        }
        Some(mode) => {
            let trials = mode.parse().unwrap_or_else(|_| {
                exit_usage(&format!("第2引数は試行数、diag、continue のいずれかです: {mode}"))
            });
            let name = args.get(2).map(String::as_str).unwrap_or("estimator");
            validate_strategy_name(name);
            run_choice_trials(&sc, &rep, trials, name, true);
        }
        None => {
            validate_strategy_name("estimator");
            run_choice_trials(&sc, &rep, 20, "estimator", true);
        }
    }
}
