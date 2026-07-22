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
//! usage:
//!   cargo run --release --bin scenario -- <名前|path.kif> [試行数=20] [戦略=estimator]
//!   cargo run --release --bin scenario -- <名前> diag [推定器数=10]
//!   cargo run --release --bin scenario -- <名前> continue [対局数=10] [手番側戦略] [相手戦略]
//!   cargo run --release --bin scenario -- suite [試行数=10] [戦略=estimator]
//!   共通フラグ: --ply N / --target USI / --diag 5g,4h （*scenario 行より優先）

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use tsuitate_bot::board::{make_usi_drop, make_usi_move, make_usi_square, parse_usi_square};
use tsuitate_bot::estimator::Estimator;
use tsuitate_bot::kifu::{Kifu, RawFoul, parse_kif};
use tsuitate_bot::observation::{Observation, ObservationLog};
use tsuitate_bot::protocol::{ClockState, Color, FoulCounts, GameStatus, PlayerView, Role};
use tsuitate_bot::shogi::{Outcome, Position, ShogiMove, parse_usi, unpromote_role};
use tsuitate_bot::strategy;

struct Scenario {
    name: String,
    desc: String,
    /// 注目している手（一致したら出力に印をつける）。既定は棋譜の ply+1 手目
    target: String,
    /// 何手目まで再生するか（ply+1 手目を考えさせる）
    ply: usize,
    /// diag で相手駒の利き枚数分布を測るマス
    diag_squares: Vec<String>,
    /// continue の足切り手数（**通算**の手数。必勝局面の遂行実験で、これを
    /// 超えたら不合格 = 引き分け扱いで打ち切る）。既定 200
    limit: u32,
    kifu: Kifu,
}

fn scenarios_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("scenarios")
}

fn usage() -> &'static str {
    "usage:
  cargo run --release --bin scenario -- <名前|path.kif> [試行数=20] [戦略=estimator]
  cargo run --release --bin scenario -- <名前> diag [推定器数=10]
  cargo run --release --bin scenario -- <名前> continue [対局数=10] [手番側戦略] [相手戦略]
  cargo run --release --bin scenario -- suite [試行数=10] [戦略=estimator]
  共通フラグ: --ply N / --target USI / --diag 5g,4h"
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

fn load_scenario(
    spec: &str,
    ply_flag: Option<usize>,
    target_flag: Option<String>,
    diag_flag: Option<String>,
) -> Result<Scenario, String> {
    let path = if spec.contains('/') || spec.ends_with(".kif") {
        PathBuf::from(spec)
    } else {
        scenarios_dir().join(format!("{spec}.kif"))
    };
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("{} を読めません: {e}", path.display()))?;
    let kifu = parse_kif(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let directive_ply: Option<usize> = kifu.directives.get("ply").and_then(|s| s.parse().ok());
    let ply = ply_flag
        .or(directive_ply)
        .ok_or("再生する手数が不明です（--ply か *scenario ply= を指定）")?;
    if ply > kifu.plies.len() {
        return Err(format!(
            "ply={ply} が棋譜の手数 {} を超えています",
            kifu.plies.len()
        ));
    }
    let target = target_flag
        .or_else(|| kifu.directives.get("target").cloned())
        .or_else(|| kifu.plies.get(ply).map(|p| p.mv.to_usi()))
        .unwrap_or_default();
    let diag_squares: Vec<String> = diag_flag
        .or_else(|| kifu.directives.get("diag").cloned())
        .map(|s| {
            s.split(',')
                .map(|x| x.trim())
                .filter(|x| !x.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    for sq in &diag_squares {
        parse_usi_square(sq).ok_or_else(|| format!("diag のマスを読めません: {sq}"))?;
    }
    let limit: u32 = kifu
        .directives
        .get("limit")
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| spec.to_string());
    // --ply で局面を変えたときは directive の説明（元の ply 前提）を使わない
    let desc = kifu
        .directives
        .get("desc")
        .filter(|_| Some(ply) == directive_ply)
        .cloned()
        .unwrap_or_else(|| format!("{ply}手目まで再生し、{}手目を考えさせる", ply + 1));
    Ok(Scenario {
        name,
        desc,
        target,
        ply,
        diag_squares,
        limit,
        kifu,
    })
}

/// リプレイ結果: 真実の局面と両者の観測ログ・反則数。[0]=先手, [1]=後手
struct Replayed {
    pos: Position,
    logs: [ObservationLog; 2],
    fouls: [u32; 2],
    plies: u32,
}

fn side_idx(c: Color) -> usize {
    if c == Color::Sente { 0 } else { 1 }
}

/// 反則試行を USI に解決する。駒コードは「移動後の駒」なので、盤上の移動元が
/// 生駒でコードが成駒なら成る手と判断する
fn resolve_foul(pos: &Position, side: Color, f: &RawFoul) -> String {
    match f {
        RawFoul::Drop { role, to } => {
            make_usi_drop(*role, *to).expect("打てない駒種の反則試行")
        }
        RawFoul::Board { from, to, role } => {
            let piece = pos
                .piece_at(*from)
                .expect("反則試行の移動元に駒がない（棋譜とKIFの不整合）");
            assert_eq!(piece.color, side, "反則試行の移動元が相手の駒");
            assert_eq!(
                unpromote_role(piece.role),
                unpromote_role(*role),
                "反則試行の駒コードと盤上の駒種が不一致"
            );
            // 駒コードは移動後の駒種: 盤上が生駒でコードが成駒なら成る手。
            // 盤上が成駒でコードが生駒に戻る組み合わせは存在しない（KIF不整合）
            let piece_promoted = piece.role != unpromote_role(piece.role);
            let code_promoted = *role != unpromote_role(*role);
            assert!(
                piece_promoted <= code_promoted,
                "反則試行のコードが生駒なのに盤上は成駒（KIF不整合）: {from:?}"
            );
            make_usi_move(*from, *to, code_promoted && !piece_promoted)
        }
    }
}

/// 棋譜（反則試行込み）を upto 手まで裁定つきでリプレイし、selfplay.rs と
/// 同じ規約で両者の観測ログを構築する
fn replay(kifu: &Kifu, upto: usize) -> Replayed {
    let mut pos = Position::initial();
    let mut logs = [ObservationLog::default(), ObservationLog::default()];
    let mut fouls = [0u32; 2];
    for ply in &kifu.plies[..upto] {
        let side = pos.turn();
        for f in &ply.fouls {
            let usi = resolve_foul(&pos, side, f);
            let mv = parse_usi(&usi).expect("反則試行のUSI解析失敗");
            assert!(!pos.is_legal(&mv), "反則のはずの手が合法: {usi}");
            fouls[side_idx(side)] += 1;
            logs[side_idx(side)].record(Observation::MyFoul {
                move_number: pos.move_number(),
                usi,
            });
            logs[side_idx(side.other())].record(Observation::OpponentFoul {
                count: fouls[side_idx(side)],
            });
        }
        let usi = ply.mv.to_usi();
        let mv = parse_usi(&usi).expect("USI解析失敗");
        assert!(pos.is_legal(&mv), "棋譜の手が非合法: {usi}");
        let captured = pos.play_unchecked(&mv);
        let move_number = pos.move_number();
        let captured_sq = captured.map(|_| match mv {
            ShogiMove::Board { to, .. } => make_usi_square(to),
            ShogiMove::Drop { .. } => unreachable!("打ちでは駒を取れない"),
        });
        logs[side_idx(side)].record(Observation::MyMove {
            move_number,
            usi,
            captured: captured.map(unpromote_role),
        });
        logs[side_idx(side.other())].record(Observation::OpponentMoved {
            move_number,
            captured_my_piece_at: captured_sq,
        });
        if pos.in_check(pos.turn()) {
            let in_check = pos.turn();
            for log in logs.iter_mut() {
                log.record(Observation::Check { in_check });
            }
        }
    }
    Replayed {
        pos,
        logs,
        fouls,
        plies: upto as u32,
    }
}

/// 実対局と同じ「自分の手番ごとの逐次 update」を再現して戦略の推定器を温める。
/// 一括 update だとリプレイ予算が1回分しか与えられず、長い棋譜では粒子が
/// 完全枯渇する（kakunari の69手を一括で食わせるとユニーク粒子0になる）
fn prewarm_strategy(
    strat: &mut Box<dyn tsuitate_bot::strategy::Strategy>,
    view: &PlayerView,
    full: &ObservationLog,
) {
    let mut running = ObservationLog::default();
    for e in full.events() {
        if matches!(e, Observation::MyMove { .. } | Observation::MyFoul { .. }) {
            strat.prewarm(view, &running);
        }
        running.record(e.clone());
    }
}

fn clone_log(log: &ObservationLog) -> ObservationLog {
    let mut out = ObservationLog::default();
    for e in log.events() {
        out.record(e.clone());
    }
    out
}

fn make_view(pos: &Position, color: Color, fouls: &[u32; 2]) -> PlayerView {
    PlayerView {
        game_id: "scenario".into(),
        your_color: color,
        your_pieces: pos.pieces_of(color),
        your_hand: pos.hand_map(color),
        turn: pos.turn(),
        move_number: pos.move_number(),
        clocks: ClockState {
            sente_ms: 900_000,
            gote_ms: 900_000,
            running: Some(pos.turn()),
            server_time: 0,
        },
        fouls: FoulCounts {
            you: fouls[side_idx(color)],
            opponent: fouls[side_idx(color.other())],
        },
        you_in_check: pos.in_check(color),
        opponent_in_check: pos.in_check(color.other()),
        status: GameStatus::Playing,
    }
}

struct ChoiceStats {
    /// (受理された手, 回数) を回数降順で
    tally: Vec<(String, u32)>,
    total_fouls: u32,
}

impl ChoiceStats {
    fn target_hits(&self, target: &str) -> u32 {
        self.tally
            .iter()
            .find(|(usi, _)| usi == target)
            .map(|(_, n)| *n)
            .unwrap_or(0)
    }
}

/// 手番側の一手の選択を試行する。反則は観測として与えて指し直させる（実対局と同じ）
fn choice_trials(
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

    let mut final_tally: HashMap<String, u32> = HashMap::new();
    let mut total_fouls = 0u32;
    for seed in 0..trials {
        let mut strat = strategy::make_seeded(name, seed).expect("未知の戦略名");
        let mut log = clone_log(&rep.logs[side_idx(side)]);
        prewarm_strategy(&mut strat, &make_view(&rep.pos, side, &rep.fouls), &log);
        let mut foul_tried: HashSet<String> = HashSet::new();
        let mut fouls = rep.fouls;
        let mut foul_seq: Vec<String> = vec![];
        let accepted = loop {
            let view = make_view(&rep.pos, side, &fouls);
            let Some(usi) = strat.choose(&view, &log, &foul_tried) else {
                break "resign".to_string();
            };
            let legal = parse_usi(&usi).is_some_and(|mv| rep.pos.is_legal(&mv));
            if legal {
                break usi;
            }
            fouls[side_idx(side)] += 1;
            log.record(Observation::MyFoul {
                move_number: rep.pos.move_number(),
                usi: usi.clone(),
            });
            foul_tried.insert(usi.clone());
            foul_seq.push(usi);
            if fouls[side_idx(side)] >= 10 {
                break "foul_limit".to_string();
            }
        };
        if verbose {
            let note = if accepted == sc.target { "（注目手）" } else { "" };
            let foul_note = if foul_seq.is_empty() {
                String::new()
            } else {
                format!(" 反則: {}", foul_seq.join(", "))
            };
            println!("seed {seed:2}: {accepted}{note}{foul_note}");
        }
        *final_tally.entry(accepted).or_insert(0) += 1;
        total_fouls += foul_seq.len() as u32;
    }

    let mut tally: Vec<_> = final_tally.into_iter().collect();
    tally.sort_by(|a, b| b.1.cmp(&a.1));
    if verbose {
        println!();
        println!("受理された手の内訳:");
        for (usi, n) in &tally {
            let mark = if *usi == sc.target { " ← 注目手" } else { "" };
            println!("  {usi}: {n}/{trials}{mark}");
        }
        println!("追加の反則の総数: {total_fouls}");
    }
    ChoiceStats { tally, total_fouls }
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
    let log = &rep.logs[side_idx(side)];
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
    // taint 粒子だけでの同集計（strategy.rs の taint_particles/taint_square_coverage
    // と同じ重み規約 = 0.5^(taint-1) 減衰・taint<=6・taint内max_lwで正規化）。
    // 「局所被覆度ビリーフ」が真実とどれだけ一致するかの直接測定
    let mut taint_cover_tally: Vec<[f64; 4]> = vec![[0.0; 4]; diag_sqs.len()];
    let mut taint_cover_mass = 0.0f64;
    let mut strict_mass = 0.0f64;
    let mut total_unique = 0u32;
    let mut strict_unique = 0u32;
    for seed in 0..n_estimators {
        let mut est = Estimator::with_seed_and_scale(side, seed, scale);
        // 実対局と同じ逐次 update（prewarm_strategy と同じ理由）
        let mut running = ObservationLog::default();
        let mut turn_no = 0;
        for e in log.events() {
            if matches!(e, Observation::MyMove { .. } | Observation::MyFoul { .. }) {
                est.update(&running);
                turn_no += 1;
                if seed == 0 {
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
                }
            }
            running.record(e.clone());
        }
        est.update(&running);
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
        // 推定器内の logw を max で正規化（評価側 stratified_sample と同じ規約）
        let max_logw = est
            .log_weights()
            .iter()
            .copied()
            .fold(f64::MIN, f64::max);
        // 重複局面は質量 Σexp(logw) を畳み込む（C-7 P1: multiplicity 保持。
        // ソフト減衰はフィルタが logw へ課金済み。info_miss は厳密判定にだけ使う。
        // phys_taint>0 は非厳密扱い: 王手駒の分布には出るが玉位置・利きの
        // 「厳密のみ」集計からは外れる）
        let mut mass: HashMap<u64, (f64, u8)> = HashMap::new();
        for (((pp, &miss), &taint), &lw) in est
            .particles()
            .iter()
            .zip(est.info_miss())
            .zip(est.phys_taint())
            .zip(est.log_weights())
        {
            let miss_eff = if taint > 0 { miss.max(1) } else { miss };
            let e = mass.entry(pp.fingerprint()).or_insert((0.0, miss_eff));
            e.0 += (lw - max_logw).exp();
            e.1 = e.1.min(miss_eff);
        }
        let mut seen: HashSet<u64> = HashSet::new();
        for pp in est.particles() {
            if !seen.insert(pp.fingerprint()) {
                continue;
            }
            let (w, penalty) = mass[&pp.fingerprint()];
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
            prewarm_strategy(strat, &make_view(&rep.pos, color, &rep.fouls), &logs[i]);
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
            if first == sc.target { "（注目手）" } else { "" },
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
        let mark = if usi == sc.target { " ← 注目手" } else { "" };
        println!("  {usi}: {n}/{games}{mark}");
    }
    let mut sorted: Vec<_> = reasons.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    println!("終局理由:");
    for (r, n) in sorted {
        println!("  {r}: {n}/{games}");
    }
}

/// scenarios/*.kif を全部回して注目手一致率を表にする（回帰テスト用）
fn run_suite(trials: u64, name: &str) {
    let dir = scenarios_dir();
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{} を読めません: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "kif"))
        .collect();
    paths.sort();
    println!("スイート: {} 件 / 戦略 {name} / 各 {trials} 試行", paths.len());
    println!();
    for path in paths {
        let sc = match load_scenario(&path.to_string_lossy(), None, None, None) {
            Ok(sc) => sc,
            Err(e) => {
                println!("{}: 読み込み失敗: {e}", path.display());
                continue;
            }
        };
        let rep = replay(&sc.kifu, sc.ply);
        let stats = choice_trials(&sc, &rep, trials, name, false);
        let hits = stats.target_hits(&sc.target);
        let others: Vec<String> = stats
            .tally
            .iter()
            .filter(|(usi, _)| *usi != sc.target)
            .take(3)
            .map(|(usi, n)| format!("{usi}×{n}"))
            .collect();
        println!(
            "{:<12} {}手目 注目手 {:<6} {hits}/{trials} 反則{} 他: {}",
            sc.name,
            sc.ply + 1,
            sc.target,
            stats.total_fouls,
            others.join(" ")
        );
    }
}

fn main() {
    // フラグ（--ply N / --target USI / --diag 5g,4h）を先に抜き取る
    let mut ply_flag: Option<usize> = None;
    let mut target_flag: Option<String> = None;
    let mut diag_flag: Option<String> = None;
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
        run_suite(trials, name);
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
    let rep = replay(&sc.kifu, sc.ply);
    if let Some(outcome) = rep.pos.outcome() {
        eprintln!(
            "ply={} の局面は終局しています（{outcome:?}）。--ply を見直してください",
            sc.ply
        );
        std::process::exit(1);
    }

    match args.get(1).map(String::as_str) {
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
            choice_trials(&sc, &rep, trials, name, true);
        }
        None => {
            validate_strategy_name("estimator");
            choice_trials(&sc, &rep, 20, "estimator", true);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(name: &str) -> Scenario {
        load_scenario(name, None, None, None).unwrap()
    }

    /// 手動翻訳で検証済みだった USI 列とパーサーの出力が一致すること
    #[test]
    fn keimaの棋譜はUSI列と反則が既知の正解に一致する() {
        let sc = load(&scenarios_dir().join("keima.kif").to_string_lossy());
        let expected = [
            "7g7f", "3a3b", "5g5f", "2b3a", "5f5e", "5a6b", "2h5h", "5c5d", "5i4h",
            "7c7d", "7i6h", "8c8d", "6h5g", "6b7c", "5g5f", "6c6d", "4h3h", "9c9d",
            "6i6h", "9d9e", "6h5g", "9e9f", "4g4f", "9f9g+", "8h6f", "P*9h", "8i7g",
            "9h9i+", "7g8e", "8d8e",
        ];
        let usi: Vec<String> = sc.kifu.plies.iter().map(|p| p.mv.to_usi()).collect();
        assert_eq!(usi, expected);
        assert_eq!(sc.ply, 29);
        assert_eq!(sc.target, "8d8e"); // 30手目（同歩）が自動導出される
        // 30手目の前の反則試行 = 6465FU
        assert_eq!(sc.kifu.plies[29].fouls.len(), 1);
    }

    #[test]
    fn kakunariの棋譜はUSI列と反則が既知の正解に一致する() {
        let sc = load(&scenarios_dir().join("kakunari.kif").to_string_lossy());
        let expected = [
            "7g7f", "3a3b", "6i7h", "1c1d", "7h7g", "2b1c", "4i5h", "5a4b", "6g6f",
            "7c7d", "5h6g", "8a7c", "5g5f", "8c8d", "8g8f", "8d8e", "4g4f", "8e8f",
            "7i7h", "8f8g+", "4f4e", "8g8h", "4e4d", "8h8i", "4d4c+", "3b4c", "P*8c",
            "8b8c", "7h8g", "8i7i", "2h8h", "7i6i", "5i5h", "P*8f", "8h8i", "8f8g+",
            "7g8g", "8c8g+", "8i8g", "P*8e", "8g8i", "4c3d", "8i6i", "N*5g", "6i8i",
            "B*6i", "8i6i", "5g6i+", "5h6i", "1c7i", "6i5h", "R*6i", "P*4f", "P*4h",
            "R*4e", "4b3b", "4e8e", "7c8e", "N*5d", "8e7g+", "5h4g", "4h4i+", "B*2b",
            "6i5i+", "2b1a+", "R*5h", "L*4c", "G*4e", "4c4a+", "7i5g+", "6g5g",
            "5h5g+", "4g3h", "4i3i", "3h2h", "5i4h",
        ];
        let usi: Vec<String> = sc.kifu.plies.iter().map(|p| p.mv.to_usi()).collect();
        assert_eq!(usi, expected);
        assert_eq!(sc.ply, 69);
        assert_eq!(sc.target, "7i5g+"); // 70手目が自動導出される
        // 反則試行の総数（69手目まで7件 + 71/73/75手目の前に4件。終局後の4件は trailing）
        let n_fouls: usize = sc.kifu.plies.iter().map(|p| p.fouls.len()).sum();
        assert_eq!(n_fouls, 11);
        assert_eq!(sc.kifu.trailing_fouls.len(), 4);
    }

    /// リプレイの裁定検証（合法手は合法・反則試行は非合法）が全編通ること
    #[test]
    fn 収録シナリオは裁定つきリプレイが通る() {
        for name in ["keima", "kakunari"] {
            let sc = load(&scenarios_dir().join(format!("{name}.kif")).to_string_lossy());
            let rep = replay(&sc.kifu, sc.kifu.plies.len());
            assert!(rep.plies > 0);
        }
        // kakunari は後手5反則・先手2反則で70手目を迎える
        let sc = load(&scenarios_dir().join("kakunari.kif").to_string_lossy());
        let rep = replay(&sc.kifu, sc.ply);
        assert_eq!(rep.fouls, [2, 5]);
        assert_eq!(rep.pos.turn(), Color::Gote);
    }
}
