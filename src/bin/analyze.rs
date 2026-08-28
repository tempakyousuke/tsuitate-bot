//! 対局記録（records/*.jsonl）の事後分析。
//!
//! game:end の全公開棋譜（真実）をリプレイし、bot視点の問題を集計する:
//! - 反則の原因分類（見えない駒に経路を塞がれた / 王手放置 / 自ら王手に飛び込んだ / 打ちマスに駒）
//! - 駒交換の損得（取った直後に取り返されたか、そのネット価値）
//! - タダ取られ（守られていない駒を只で取られた）
//! - 1手詰みの存在（参考値: botからは玉位置が見えないため「逃し」を責める指標ではなく、
//!   玉位置推定が当たっていれば勝てた機会の総量を測る）
//! - 被詰めろ（相手番の局面で相手に1手詰めが存在した回数）。相手が実際に詰みを
//!   見つけたかに依らないので、相手が弱い環境でも自玉の受けの失敗を直接測れる
//! - 王手ソルバー（check.rs）の再現検証: 記録上の王手中の反則それぞれについて、
//!   その時点の観測だけからソルバーが選んだ手が真の局面で合法だったかを判定する
//! - 被王手時に汚名マスへの玉移動が真に合法だったのに選ばず反則した回数
//!   （`KING_REPEAT_FOUL_W` が正しい王手解消を沈める機会）
//!
//! 使い方: cargo run --release --bin analyze -- records/*.jsonl

use std::collections::{HashMap, HashSet};

use tsuitate_bot::board::Coord;
use tsuitate_bot::check::CheckSolver;
use tsuitate_bot::check_economy::{
    CheckFoulReason, CheckMoveKind, CheckTurn, TurnType, check_turns, classify_check_foul,
    cluster_ratio_ci, sq as sq_name, true_checkers, view_from_model,
};
use tsuitate_bot::mate::{mate_moves_in_1, mate_moves_in_1_fast};
use tsuitate_bot::mate_economy::{MateKind, fold_episodes, threat_turns};
use tsuitate_bot::model::GameModel;
use tsuitate_bot::observation::{stale_king_foul_dests, Observation, ObservationLog};
use tsuitate_bot::protocol::{Color, FoulRecord, GameEndPayload, Role};
use tsuitate_bot::shogi::{Outcome, Position, ShogiMove, parse_usi, piece_value};
use tsuitate_bot::strategy::candidate_moves;

struct GameRecord {
    file: String,
    bot_color: Color,
    strategy: String,
    observations: Vec<Observation>,
    end: GameEndPayload,
    /// (選択時の p_legal 予測, 実際に合法だったか, move_number)。chose イベントの
    /// debug.p_legal と、その手の受理/反則の突き合わせ（C-7 P3 の較正測定）。
    /// move_number は王手中の手番だけに絞った較正（王手中の p_legal 過信の診断）用
    p_legal_outcomes: Vec<(f64, bool, u32)>,
    /// (決定点の move_number, USI) → 記録された p_legal。
    /// `p_legal_outcomes` と違い**決定点の手数**で引ける（MyMove の
    /// move_number は適用後の値なので -1 して揃える）。issue #31 の
    /// 手種別・順番別の較正が、試行1つずつを手種へ割り当てるのに使う
    p_legal_by_attempt: HashMap<(u32, String), f64>,
    /// 決定点（positions のインデックス）→ その決定の chose debug で観測できる
    /// 「厳密粒子ゼロだったか」。`debug.sample_slots`（評価に使った厳密粒子の
    /// スロット数）が 0 なら true。
    ///
    /// **その手番の最初の `chose` だけを見る**。反則後の再選択は同じ move_number で
    /// 何度も来るが、粒子の状態は反則の観測を食った後の別物なので混ぜられない。
    /// 最初の選択に `sample_slots` が無ければ（定跡手・旧記録の `debug: null`）
    /// その決定は `Some(None)` = 判定不能として登録し、**再選択の値で埋めない**。
    /// P0-2 の漏斗の3段目（2手読みは厳密粒子ゼロには効かない）で使う
    blind_decisions: HashMap<usize, Option<bool>>,
}

fn load(path: &str) -> Option<GameRecord> {
    let content = std::fs::read_to_string(path).ok()?;
    parse_record(path, &content)
}

/// 記録（JSONL）1局ぶんの本体。テストから直接呼べるように path と分けてある
fn parse_record(path: &str, content: &str) -> Option<GameRecord> {
    let mut bot_color = None;
    let mut strategy = String::new();
    let mut observations = vec![];
    let mut end = None;
    let mut p_legal_outcomes = vec![];
    let mut p_legal_by_attempt: HashMap<(u32, String), f64> = HashMap::new();
    let mut blind_decisions: HashMap<usize, Option<bool>> = HashMap::new();
    // 直近の chose イベントの (usi, p_legal)。次の MyMove/MyFoul 観測と照合する
    let mut pending_chose: Option<(String, f64)> = None;
    for line in content.lines() {
        let v: serde_json::Value = serde_json::from_str(line).ok()?;
        match v["type"].as_str() {
            Some("match") => {
                bot_color = serde_json::from_value(v["your_color"].clone()).ok();
                strategy = v["strategy"].as_str().unwrap_or("?").to_string();
            }
            Some("chose") => {
                // p_legal の無い chose（定跡手・旧記録）で古い pending を残さない
                // （後続の同一 USI に誤対応するのを防ぐ — codex レビュー指摘）
                pending_chose = None;
                if let (Some(usi), Some(p)) =
                    (v["usi"].as_str(), v["debug"]["p_legal"].as_f64())
                {
                    pending_chose = Some((usi.to_string(), p));
                }
                // chose の move_number はその決定点の手数（着手前）なので
                // positions のインデックスは -1（MyMove の -2 とは規約が違う）。
                // **最初の chose を必ず登録する**: sample_slots が無いときに
                // 未登録のまま残すと、同じ手番の反則後の再選択（粒子が反則の
                // 観測を食った後の別状態）がその手番の「最初の決定」として
                // 登録されてしまう
                if let Some(mn) = v["move_number"].as_u64() {
                    blind_decisions
                        .entry((mn as usize).saturating_sub(1))
                        .or_insert_with(|| {
                            v["debug"]["sample_slots"].as_u64().map(|slots| slots == 0)
                        });
                }
            }
            Some("obs") => {
                if let Ok(obs) = serde_json::from_value::<Observation>(v["event"].clone()) {
                    match (&obs, &pending_chose) {
                        (Observation::MyMove { usi, move_number, .. }, Some((cu, p)))
                            if usi == cu =>
                        {
                            p_legal_outcomes.push((*p, true, *move_number));
                            p_legal_by_attempt
                                .insert((move_number.saturating_sub(1), usi.clone()), *p);
                            pending_chose = None;
                        }
                        (Observation::MyFoul { usi, move_number }, Some((cu, p)))
                            if usi == cu =>
                        {
                            p_legal_outcomes.push((*p, false, *move_number));
                            p_legal_by_attempt.insert((*move_number, usi.clone()), *p);
                            pending_chose = None;
                        }
                        _ => {}
                    }
                    observations.push(obs);
                }
            }
            Some("end") => {
                end = serde_json::from_value(v["payload"].clone()).ok();
            }
            _ => {}
        }
    }
    Some(GameRecord {
        file: path.to_string(),
        bot_color: bot_color?,
        strategy,
        observations,
        end: end?,
        p_legal_outcomes,
        p_legal_by_attempt,
        blind_decisions,
    })
}

/// ソルバー方策が「破滅した」とみなす反則数（issue #31 の実測では 8〜10回）
const SOLVER_CATASTROPHE: u32 = 8;

/// 王手中の反則経済（issue #31 P0-1/P0-2/P0-3）の集計器。
///
/// **粒子を回さない**（記録の観測列＋真実の棋譜＋CheckSolver だけ）ので
/// アリーナ記録に対してそのまま常設できる。エンジンの順位・gain が要る量
/// （k\* 監査・方策シミュレーション）は P0-4/P0-5 の領分。
#[derive(Default)]
struct CheckEconomy {
    /// 型別の (手番数, 反則数)
    types: HashMap<TurnType, (u32, u32)>,
    /// 最初の反則の着手先（真実）: [王手駒がいた, 敵駒はいたが王手駒でない, 空]
    first_foul_target: [u32; 3],
    /// 整合性検査（受理手が決定点の真実局面で合法か）。反則は手番を変えない
    /// ので常に一致するはずで、ずれたら手数の対応がおかしい
    accepted_consistent: (u32, u32),
    /// 反則した手番の開始時点で**真に合法だった候補の本数** [1本, 2〜3本, 4本以上]。
    /// 1本しか無い手番は価格では避けられない（確かめが必要）
    legal_outs_hist: [u32; 3],
    /// 反則した手番のうち、**ソルバー最善が最初から真に合法**だった手番
    /// （= 1反則も要らなかった。issue の「64手番」に対応）
    best_legal_was_top: (u32, u32),
    /// 受理された**非玉手**の開始時ソルバー p 順位 [1位, 2〜3位, 4位以上, 不明]
    accepted_nonking_rank: [u32; 4],
    /// 反則した王手手番の開始時の**残り反則**（index = 残り回数 0..=10）
    remaining_hist: [u32; 11],
    /// P0-2 の較正: (手種, 順番束 0/1/2+) → (n, Σ予測, 合法数)
    calib: HashMap<(CheckMoveKind, usize), (u32, f64, u32)>,
    /// 捕獲試みの**方向つき較正誤差**の cluster（元対局ごとの (Σ(p−y), n)）
    capture_gap_clusters: Vec<(f64, f64)>,
    /// 王手中の bot の手番の総数（**反則0の手番も含む分母**）
    turns_total: u32,
    fouls_total: u32,
    /// 真の王手駒仮説の重みシェアとエントロピー（反則あり/なしの手番で分ける）。
    /// 「知らない（希釈）」と「知っていて払う」を分ける量
    hyp_share: [(u32, f64, f64); 2],
}

impl CheckEconomy {
    /// 1局ぶんの王手手番を畳む。`turns` は check_economy::check_turns の出力
    fn add_game(&mut self, turns: &[CheckTurn]) {
        let mut gap_sum = 0.0;
        let mut gap_n = 0.0;
        for turn in turns {
            self.turns_total += 1;
            self.fouls_total += turn.fouls() as u32;
            let e = self.types.entry(turn.turn_type()).or_insert((0, 0));
            e.0 += 1;
            e.1 += turn.fouls() as u32;
            if let Some(first) = turn.first_foul() {
                let bucket = if first.truth_checker_at_to {
                    0
                } else if first.truth_enemy_at_to {
                    1
                } else {
                    2
                };
                self.first_foul_target[bucket] += 1;
                let remaining = 10usize.saturating_sub(turn.fouls_before as usize);
                self.remaining_hist[remaining.min(10)] += 1;
            }
            if turn.accepted_legal_at_entry.is_some() {
                self.accepted_consistent.1 += 1;
                if turn.accepted_legal_at_entry == Some(true) {
                    self.accepted_consistent.0 += 1;
                }
            }
            if turn.fouls() > 0 {
                let outs = match turn.legal_candidates_at_entry {
                    0 | 1 => 0,
                    2..=3 => 1,
                    _ => 2,
                };
                self.legal_outs_hist[outs] += 1;
                self.best_legal_was_top.1 += 1;
                if turn.best_legal_rank_at_entry == Some(1) {
                    self.best_legal_was_top.0 += 1;
                }
            }
            if let (Some(acc), true) = (turn.accepted_attempt(), turn.fouls() > 0) {
                if acc.kind != CheckMoveKind::King {
                    let bucket = match turn.accepted_rank_at_entry {
                        Some(1) => 0,
                        Some(2..=3) => 1,
                        Some(_) => 2,
                        None => 3,
                    };
                    self.accepted_nonking_rank[bucket] += 1;
                }
            }
            if let (Some(share), Some(ent)) = (turn.true_hyp_share, turn.hyp_entropy) {
                let slot = usize::from(turn.fouls() > 0);
                self.hyp_share[slot].0 += 1;
                self.hyp_share[slot].1 += share;
                self.hyp_share[slot].2 += ent;
            }
            for a in &turn.attempts {
                let Some(p) = a.p_legal else { continue };
                let order = a.order.min(2);
                let c = self.calib.entry((a.kind, order)).or_insert((0, 0.0, 0));
                c.0 += 1;
                c.1 += p;
                if a.was_legal {
                    c.2 += 1;
                }
                if a.kind == CheckMoveKind::CheckerCapture {
                    gap_sum += p - if a.was_legal { 1.0 } else { 0.0 };
                    gap_n += 1.0;
                }
            }
        }
        // 捕獲試みが1件も無かった局も cluster として数える（分母は元対局）
        self.capture_gap_clusters.push((gap_sum, gap_n));
    }

    fn report(&self) {
        if self.turns_total == 0 {
            return;
        }
        println!("\n--- 王手中の反則経済（issue #31 P0-1/P0-2）---");
        let pct = |a: u32, b: u32| -> f64 {
            if b == 0 { 0.0 } else { f64::from(a) * 100.0 / f64::from(b) }
        };
        println!(
            "王手中の手番: {} / 反則 {}（分母は反則0の手番も含む全王手手番）",
            self.turns_total, self.fouls_total
        );
        for t in TurnType::ALL {
            let (turns, fouls) = self.types.get(&t).copied().unwrap_or((0, 0));
            println!(
                "  型 {:<34} 手番 {:>4} ({:>4.1}%) / 反則 {:>4}",
                t.label(),
                turns,
                pct(turns, self.turns_total),
                fouls,
            );
        }
        let ft = self.first_foul_target;
        println!(
            "  最初の反則の着手先（真実）: 王手駒がいた {} / 敵駒はいたが王手駒でない {} / 空 {}",
            ft[0], ft[1], ft[2]
        );
        let (ok, all) = self.accepted_consistent;
        if all > 0 && ok != all {
            println!(
                "  **整合性検査に失敗**: 受理手が決定点の真実局面で合法 {ok}/{all}（手数の対応がずれている）"
            );
        }
        let outs = self.legal_outs_hist;
        let outs_n: u32 = outs.iter().sum();
        if outs_n > 0 {
            println!(
                "  反則した手番の開始時に真に合法だった候補: 1本以下 {} / 2〜3本 {} / 4本以上 {}（1本の手番は価格では避けられない）",
                outs[0], outs[1], outs[2]
            );
            let (top, tot) = self.best_legal_was_top;
            println!(
                "  そのうちソルバー最善が最初から合法だった手番: {top}/{tot} ({:.1}% = 1反則も要らなかった)",
                pct(top, tot)
            );
        }
        let r = self.accepted_nonking_rank;
        let rn: u32 = r.iter().sum();
        if rn > 0 {
            println!(
                "  受理された非玉手の開始時ソルバーp順位: 1位 {} / 2〜3位 {} / 4位以上 {} / 不明 {}（上位なら「単に次点へ進んだだけ」。エンジン順位は P0-4）",
                r[0], r[1], r[2], r[3]
            );
        }
        for (slot, name) in [(0usize, "反則0"), (1, "反則あり")] {
            let (n, share, ent) = self.hyp_share[slot];
            if n > 0 {
                println!(
                    "  王手駒仮説（{name}の手番 {n}）: 真の王手駒の重みシェア 平均 {:.3} / 正規化エントロピー 平均 {:.3}",
                    share / f64::from(n),
                    ent / f64::from(n),
                );
            }
        }
        let rem: Vec<String> = (1..=10)
            .rev()
            .map(|k| format!("{k}:{}", self.remaining_hist[k]))
            .collect();
        println!("  反則した王手手番の開始時の残り反則: {}", rem.join(" "));

        if self.calib.is_empty() {
            println!(
                "  p_legal 較正: 記録に chose.debug.p_legal がありません（旧記録。P0-2 は arena-records に対して回すこと）"
            );
            return;
        }
        println!("  p_legal 較正（手種 × 手番内の順番。記録の chose.debug.p_legal）:");
        for kind in CheckMoveKind::ALL {
            for order in 0..3 {
                let Some(&(n, sum_p, legal)) = self.calib.get(&(kind, order)) else {
                    continue;
                };
                if n == 0 {
                    continue;
                }
                let mean_p = sum_p / f64::from(n);
                let rate = f64::from(legal) / f64::from(n);
                println!(
                    "    {:<16} 順番{:<3} n={:>4} 平均予測 {:.3} / 合法率 {:.3} / 差 {:+.3}",
                    kind.label(),
                    if order == 2 { "2+".to_string() } else { order.to_string() },
                    n,
                    mean_p,
                    rate,
                    mean_p - rate,
                );
            }
        }
        // H3 の判定は Brier ではなく**方向つき較正誤差**（元対局 cluster CI）。
        // 門: 下限 > 0.1（そして v13 / v14 の両相手で同方向）
        let n: f64 = self.capture_gap_clusters.iter().map(|c| c.1).sum();
        if n > 0.0 {
            let gap: f64 = self.capture_gap_clusters.iter().map(|c| c.0).sum::<f64>() / n;
            let (lo, hi) = cluster_ratio_ci(&self.capture_gap_clusters, 0.05, 0x2026_0828);
            println!(
                "  捕獲試みの方向つき較正誤差（平均予測 − 合法率）: {gap:+.3} [95% CI {lo:+.3}, {hi:+.3}]（n={n:.0}手 / {}局。門: CI下限 > 0.1 かつ v13/v14 で同方向）",
                self.capture_gap_clusters.len(),
            );
        }
    }
}

/// ソルバー方策の**破滅**（issue #31 P0-3）: 解消確率の argmax で指し直すと
/// かえって 8 反則以上を積む手番。P1 の方策がソルバーの p を強く信じるほど
/// この破滅を継承するので、原因（仮説集合に真の王手駒が無い／`legal_under` が
/// 支え駒を置かない盲点）を手番ごとに記録する。
#[derive(Default)]
struct SolverCatastrophes {
    turns: u32,
    /// 真の王手駒の仮説集合での状態 [マス・駒種とも一致, マスのみ一致, なし]
    hypothesis_status: [u32; 3],
    /// ソルバーが積んだ反則の原因
    reasons: HashMap<CheckFoulReason, u32>,
}

impl SolverCatastrophes {
    fn report(&self, threshold: u32) {
        println!(
            "  ソルバー方策の破滅（{threshold}反則以上）: {}手番",
            self.turns
        );
        if self.turns == 0 {
            return;
        }
        let h = self.hypothesis_status;
        println!(
            "    真の王手駒の仮説: マス・駒種とも一致 {} / マスのみ一致 {} / 仮説集合に無い {}",
            h[0], h[1], h[2]
        );
        let mut reasons: Vec<_> = self.reasons.iter().collect();
        reasons.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
        let line: Vec<String> = reasons
            .iter()
            .map(|(r, n)| format!("{} {}", r.label(), n))
            .collect();
        println!("    反則の原因（真実から分類）: {}", line.join(" / "));
    }
}

/// 記録上「王手中に反則した手番」それぞれを、ソルバー方策（解消確率の argmax）で
/// 最初から指し直し、合法手に到達するまでの反則回数を実際の反則回数と比較する。
/// 実運用と同じく、反則するたびにその手を仮説消去へ回して選び直す。
/// 戻り値: (検証した手番数, 実際の反則合計, ソルバー方策での反則合計)
fn simulate_check_solver(
    rec: &GameRecord,
    positions: &[Position],
    bot: Color,
    cat: &mut SolverCatastrophes,
) -> (u32, u32, u32) {
    // 手番ごとの実際の反則回数と、その手番の最初の反則の直前までの観測数
    let mut turns: Vec<(u32, usize, u32)> = vec![]; // (move_number, obs_prefix, actual_fouls)
    for (i, obs) in rec.observations.iter().enumerate() {
        let Observation::MyFoul { move_number, .. } = obs else {
            continue;
        };
        match turns.last_mut() {
            Some((mn, _, n)) if *mn == *move_number => *n += 1,
            _ => turns.push((*move_number, i, 1)),
        }
    }

    let mut tested = 0;
    let mut actual_total = 0;
    let mut solver_total = 0;
    for (move_number, prefix, actual) in turns {
        let idx = (move_number as usize).saturating_sub(1);
        let Some(truth) = positions.get(idx) else {
            continue;
        };
        if !truth.in_check(bot) {
            continue; // 王手以外の反則（経路封鎖など）はソルバーの対象外
        }
        let mut log = ObservationLog::default();
        for prev in &rec.observations[..prefix] {
            log.record(prev.clone());
        }
        let model = GameModel::from_log(bot, &log);
        let view = view_from_model(&model, true, truth.move_number());

        let mut fouls: Vec<ShogiMove> = vec![];
        let mut tried: HashSet<String> = HashSet::new();
        let mut sim_fouls = 0u32;
        let mut sequence: Vec<String> = vec![];
        loop {
            if sim_fouls >= 10 {
                break; // 反則負け相当
            }
            let candidates = candidate_moves(&view, &tried);
            if candidates.is_empty() {
                break;
            }
            let Some(mut solver) = CheckSolver::new(&view, &[], &fouls, &log) else {
                break;
            };
            let best = candidates
                .iter()
                .map(|(usi, mv)| (usi.clone(), *mv, solver.resolve_probability(mv)))
                .max_by(|a, b| a.2.total_cmp(&b.2));
            let Some((usi, mv, p)) = best else { break };
            if truth.is_legal(&mv) {
                sequence.push(format!("{usi}(p{p:.2})○"));
                break;
            }
            sequence.push(format!("{usi}(p{p:.2})×"));
            sim_fouls += 1;
            fouls.push(mv);
            tried.insert(usi);
        }
        tested += 1;
        actual_total += actual;
        solver_total += sim_fouls;
        println!(
            "  王手手番 {move_number}手目: 実際の反則 {actual}回 → ソルバー方策 {sim_fouls}回 [{}]",
            sequence.join(" ")
        );
        // P0-3: 破滅した手番（ソルバーのほうが大きく損をする）の原因を記録する。
        // 「ソルバーに従え」は解ではないので、P1 の方策がこの破滅を継承しないか
        // を見る材料にする
        if sim_fouls >= SOLVER_CATASTROPHE {
            cat.turns += 1;
            let checkers = true_checkers(truth, bot);
            let hyps = CheckSolver::new(&view, &[], &[], &log)
                .map(|s| s.hypotheses_debug())
                .unwrap_or_default();
            let status = if checkers
                .iter()
                .any(|(s, r)| hyps.iter().any(|(hs, hr, _)| hs == s && hr == r))
            {
                0
            } else if checkers.iter().any(|(s, _)| hyps.iter().any(|(hs, _, _)| hs == s)) {
                1
            } else {
                2
            };
            cat.hypothesis_status[status] += 1;
            let mut reasons: Vec<String> = vec![];
            for mv in &fouls {
                let reason = classify_check_foul(truth, bot, mv);
                *cat.reasons.entry(reason).or_insert(0) += 1;
                reasons.push(reason.label().to_string());
            }
            println!(
                "    [破滅] 真の王手駒 {} / 仮説 {} / 原因 {}",
                checkers
                    .iter()
                    .map(|(s, r)| format!("{}{:?}", sq_name(*s), r))
                    .collect::<Vec<_>>()
                    .join(","),
                ["マス・駒種一致", "マスのみ一致", "仮説集合に無い"][status],
                reasons.join(","),
            );
        }
    }
    (tested, actual_total, solver_total)
}

/// 王手中の実反則それぞれについて「その瞬間にソルバーが知っていた最善解消確率
/// p_max」と「実際に選んだ（反則になった）手の解消確率 p_chosen」を監査する。
/// p_max が高い（ほぼ確実な解消手を既に知っていた）のに低い p の手で反則して
/// いる質量が大きければ、王手中の gain が p_legal を上書きしている構造的な
/// 欠陥がある（966 vs 611 のソルバー方策ギャップの内訳を取る診断。2026-07-29）。
/// 戻り値: [p_max≥0.9, 0.5..0.9, <0.5] ごとの (反則数, p_chosen の合計)。
/// solver_p_outcomes には王手中の全決定（反則・受理とも）の
/// (ソルバー単体の p, 実際に合法だったか) を積む（エンジンの p_legal 較正との
/// 比較用。ソルバー単体のほうが当たるなら evaluate() のブレンドが希釈している）
/// 王手中の決定について「真の王手駒だけを置いた単独盤面なら合法」だった手の
/// 手種別の実際の合法率を測る。CheckSolver::legal_under は仮説の王手駒1枚
/// しか置かないため、見えない支え駒・利きの被覆を無視して玉の手を過信する
/// （実測: p=1.00 と断言した玉での王手駒捕獲が支え付きで反則 = 健全性違反）。
/// ここで測る「単独仮説合法 → 実際合法」の手種別の率が、resolve_probability に
/// 掛けるべき事前確率の実測値になる。
/// 種別: [玉で王手駒マスを捕獲, 玉の自陣方向への後退, 玉のその他の移動,
///        玉以外の捕獲, 打ち, その他]
/// 後退を別枠にするのは rei3 のユーザー指導「自陣奥への退路は仮説によらず
/// ほぼ確実に合法」の検証（一律の玉の手割引はアリーナ3シードで悪化した）
fn tally_single_hyp_legality(
    truth: &Position,
    bot: Color,
    mv: &ShogiMove,
    was_legal: bool,
    counts: &mut [(u32, u32); 6],
) {
    let Some(king) = truth.king_square(bot) else {
        return;
    };
    // 真の王手駒（複数=両王手もそのまま全部置く）
    let checkers: Vec<(Coord, Role)> = truth
        .pieces()
        .filter(|(sq, p)| p.color != bot && truth.attacks(*sq, king))
        .map(|(sq, p)| (sq, p.role))
        .collect();
    if checkers.is_empty() {
        return;
    }
    // 単独盤面: 自駒・自持ち駒・真の王手駒だけ
    let mut base = Position::empty(bot);
    for (sq, p) in truth.pieces() {
        if p.color == bot {
            base.set(sq, Some(p));
        }
    }
    for (&role, &n) in &truth.hand_map(bot) {
        base.set_hand(bot, role, n as u8);
    }
    for &(sq, role) in &checkers {
        base.set(
            sq,
            Some(tsuitate_bot::shogi::Piece {
                color: bot.other(),
                role,
            }),
        );
    }
    if !base.is_legal(mv) {
        return; // 単独仮説でも反則予測 → ソルバーの過信の対象外
    }
    let to = match *mv {
        ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
    };
    let kind = match *mv {
        ShogiMove::Board { from, .. } if from == king => {
            if checkers.iter().any(|&(sq, _)| sq == to) {
                0 // 玉で王手駒を捕獲
            } else {
                let backward = match bot {
                    Color::Sente => to.rank > from.rank,
                    Color::Gote => to.rank < from.rank,
                };
                if backward {
                    1 // 玉の自陣方向への後退
                } else {
                    2 // 玉のその他の移動（横・前進）
                }
            }
        }
        ShogiMove::Board { .. } if truth.piece_at(to).is_some_and(|p| p.color != bot) => 3,
        ShogiMove::Board { .. } => 5,
        ShogiMove::Drop { .. } => 4,
    };
    counts[kind].0 += 1;
    if was_legal {
        counts[kind].1 += 1;
    }
}

fn audit_check_fouls(
    rec: &GameRecord,
    positions: &[Position],
    bot: Color,
    solver_p_outcomes: &mut Vec<(f64, bool)>,
    single_hyp_counts: &mut [(u32, u32); 6],
) -> [(u32, f64); 3] {
    let mut buckets = [(0u32, 0.0f64); 3];
    for (i, obs) in rec.observations.iter().enumerate() {
        // 着手時局面: MyFoul は move_number-1、MyMove は適用後の値なので -2
        let (move_number, usi, was_legal) = match obs {
            Observation::MyFoul { move_number, usi } => (*move_number, usi, false),
            Observation::MyMove { move_number, usi, .. } => {
                (move_number.saturating_sub(1), usi, true)
            }
            _ => continue,
        };
        let idx = (move_number as usize).saturating_sub(1);
        let Some(truth) = positions.get(idx) else {
            continue;
        };
        if !truth.in_check(bot) {
            continue;
        }
        let Some(chosen_mv) = parse_usi(usi) else {
            continue;
        };
        tally_single_hyp_legality(truth, bot, &chosen_mv, was_legal, single_hyp_counts);
        let mut log = ObservationLog::default();
        for prev in &rec.observations[..i] {
            log.record(prev.clone());
        }
        // この手番でここまでに試した反則（仮説消去に使う）
        let mut fouls: Vec<ShogiMove> = vec![];
        let mut tried: HashSet<String> = HashSet::new();
        for prev in &rec.observations[..i] {
            if let Observation::MyFoul { move_number: mn, usi: u } = prev {
                if *mn == move_number {
                    if let Some(m) = parse_usi(u) {
                        fouls.push(m);
                    }
                    tried.insert(u.clone());
                }
            }
        }
        let model = GameModel::from_log(bot, &log);
        let view = view_from_model(&model, true, truth.move_number());
        let Some(mut solver) = CheckSolver::new(&view, &[], &fouls, &log) else {
            continue;
        };
        let p_chosen = solver.resolve_probability(&chosen_mv);
        solver_p_outcomes.push((p_chosen, was_legal));
        if was_legal {
            continue; // p_max 別の内訳は反則だけ数える
        }
        let p_max = candidate_moves(&view, &tried)
            .iter()
            .map(|(_, mv)| solver.resolve_probability(mv))
            .fold(0.0f64, f64::max);
        let bucket = if p_max >= 0.9 {
            0
        } else if p_max >= 0.5 {
            1
        } else {
            2
        };
        buckets[bucket].0 += 1;
        buckets[bucket].1 += p_chosen;
        // p_max≥0.9 なのに選ばれなかった反則の性質: 捕獲試み（プローブ＝
        // kakutori 型の意図された挙動でありうる）か、打ちの合駒か、その他か
        if bucket == 0 {
            let kind = match chosen_mv {
                ShogiMove::Board { to, .. }
                    if truth.piece_at(to).is_some_and(|p| p.color != bot) =>
                {
                    "捕獲試み(真に敵駒あり)"
                }
                ShogiMove::Board { .. } => "盤上移動(捕獲空振り含む)",
                ShogiMove::Drop { .. } => "打ち(合駒試み)",
            };
            println!(
                "  王手中反則(安全手p{p_max:.2}を無視) {}手目 {}: p{:.2} {} [{:?}]",
                move_number,
                usi,
                p_chosen,
                kind,
                classify_foul(truth, &FoulRecord {
                    move_number,
                    by_color: bot,
                    usi: usi.clone(),
                }),
            );
        }
    }
    buckets
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum FoulCause {
    /// 経路上に見えない相手駒があって届かない（or 移動先の自駒 = 起きないはず）
    Blocked,
    /// 王手を受けていて、その手では解消できなかった（攻め駒の位置を知らない）
    CheckUnresolved,
    /// 王手は受けていないのに、指すと自玉が王手になる（ピン・利きへの飛び込み）
    IntoCheck,
    /// 持ち駒を打とうとしたマスに見えない駒があった
    DropOccupied,
    /// 打ち歩詰め
    PawnDropMate,
    /// 上記以外（想定外）
    Other,
}

fn classify_foul(pos: &Position, foul: &FoulRecord) -> FoulCause {
    let Some(mv) = parse_usi(&foul.usi) else {
        return FoulCause::Other;
    };
    // is_legal と同じ順で原因を切り分ける
    if !pos.is_pseudo_legal(&mv) {
        return match mv {
            ShogiMove::Board { .. } => FoulCause::Blocked,
            ShogiMove::Drop { .. } => FoulCause::DropOccupied,
        };
    }
    let mut probe = pos.clone();
    probe.play_unchecked(&mv);
    if probe.in_check(pos.turn()) {
        return if pos.in_check(pos.turn()) {
            FoulCause::CheckUnresolved
        } else {
            FoulCause::IntoCheck
        };
    }
    if let ShogiMove::Drop { .. } = mv {
        return FoulCause::PawnDropMate;
    }
    FoulCause::Other
}

/// Blocked/DropOccupied 反則について、真の局面から「実際に駒があった」
/// マスを一意に特定する（analyze は真の棋譜を持つので、bot視点では
/// 一意化できない経路封鎖も truth で厳密に分かる）。占有マス反則の
/// 再訪率測定に使う
fn true_occupied_square(pos: &Position, mv: &ShogiMove) -> Option<Coord> {
    match *mv {
        ShogiMove::Drop { to, .. } => pos.piece_at(to).map(|_| to),
        ShogiMove::Board { from, to, .. } => {
            let df = to.file - from.file;
            let dr = to.rank - from.rank;
            let aligned = df == 0 || dr == 0 || df.abs() == dr.abs();
            let steps = df.abs().max(dr.abs());
            if !aligned || steps <= 1 {
                return None;
            }
            let sf = df.signum();
            let sr = dr.signum();
            (1..steps)
                .map(|k| Coord {
                    file: from.file + sf * k,
                    rank: from.rank + sr * k,
                })
                .find(|&sq| pos.piece_at(sq).is_some())
        }
    }
}

fn cause_label(c: FoulCause) -> &'static str {
    match c {
        FoulCause::Blocked => "経路が見えない駒に塞がれた",
        FoulCause::CheckUnresolved => "王手を解消できない手（攻め駒の位置不明）",
        FoulCause::IntoCheck => "自ら王手に飛び込んだ（ピン/見えない利き）",
        FoulCause::DropOccupied => "打ちマスに見えない駒",
        FoulCause::PawnDropMate => "打ち歩詰め",
        FoulCause::Other => "その他",
    }
}

/// bot の手番で1手詰み（相手玉）が存在するか
/// 王手中に汚名マスへの玉移動が真に合法だったのに、それを選ばずに失敗した回数。
/// `TSUITATE_KING_REPEAT_FOUL_W` が正しい王手解消を沈める機会の上限。
struct CheckStaleKingMissed {
    /// 王手中の決定で、汚名マスへの玉移動が真局面で合法だった回数
    legal_escape_decisions: u32,
    /// そのうち反則した（その合法な玉移動を選ばなかった）
    missed_fouls: u32,
    /// その合法な玉移動を選んで解消した
    used: u32,
}

fn king_board_move(from: Coord, to: Coord) -> ShogiMove {
    ShogiMove::Board {
        from,
        to,
        promote: false,
    }
}

/// 汚名マスへの玉移動のうち、真局面で王手を解消できる行き先。
fn legal_stale_king_escapes(
    pos: &Position,
    bot: Color,
    stale: &HashSet<Coord>,
) -> Vec<Coord> {
    let Some(king) = pos.king_square(bot) else {
        return vec![];
    };
    if !pos.in_check(bot) {
        return vec![];
    }
    stale
        .iter()
        .copied()
        .filter(|&to| pos.is_legal(&king_board_move(king, to)))
        .collect()
}

fn tally_check_stale_king_missed(
    observations: &[Observation],
    positions: &[Position],
    bot: Color,
    print: bool,
) -> CheckStaleKingMissed {
    let mut out = CheckStaleKingMissed {
        legal_escape_decisions: 0,
        missed_fouls: 0,
        used: 0,
    };
    let mut prefix = ObservationLog::default();
    for obs in observations {
        match obs {
            Observation::MyFoul { move_number, usi } => {
                let idx = (*move_number as usize).saturating_sub(1);
                if let Some(pos) = positions.get(idx) {
                    if pos.in_check(bot) {
                        let stale = stale_king_foul_dests(&prefix, bot, *move_number);
                        let legal = legal_stale_king_escapes(pos, bot, &stale);
                        if !legal.is_empty() {
                            out.legal_escape_decisions += 1;
                            out.missed_fouls += 1;
                            if print {
                                let king = pos.king_square(bot).unwrap();
                                let escapes = legal
                                    .iter()
                                    .map(|&to| king_board_move(king, to).to_usi())
                                    .collect::<Vec<_>>()
                                    .join("/");
                                println!(
                                    "  王手中・汚名マスへの玉逃げが真に合法なのに反則: {}手目 {}（合法な解消 {escapes}）",
                                    move_number, usi
                                );
                            }
                        }
                    }
                }
            }
            Observation::MyMove {
                move_number, usi, ..
            } => {
                // MyMove の move_number は適用後。決定時は −1。
                let decision_mn = move_number.saturating_sub(1);
                let idx = (*move_number as usize).saturating_sub(2);
                if let Some(pos) = positions.get(idx) {
                    if pos.in_check(bot) {
                        let stale = stale_king_foul_dests(&prefix, bot, decision_mn);
                        let legal = legal_stale_king_escapes(pos, bot, &stale);
                        if !legal.is_empty() {
                            out.legal_escape_decisions += 1;
                            if let Some(ShogiMove::Board { from, to, .. }) = parse_usi(usi) {
                                if pos.king_square(bot) == Some(from) && legal.contains(&to) {
                                    out.used += 1;
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        prefix.record(obs.clone());
    }
    out
}

/// `mate_moves_in_1_fast` の実測コスト（issue #28 P1-A: `mate_samples` と
/// 対象候補数を決める前に、**実戦の局面**で単発コストを測る）。
///
/// 初期局面の perft から出した movegen の平均単価は、詰み直前の終盤で
/// 相手の合法応手をほぼ全走査する最悪ケースの代理にならない。
#[derive(Default)]
struct MateCost {
    /// 1呼び出しの所要ナノ秒と、その局面が終盤（90手目以降）か
    samples: Vec<(u64, bool)>,
    /// `TSUITATE_MATE_COST=1` のときだけ測る完全版（全合法手）との比
    full_ns: u64,
    fast_ns: u64,
    compared: u32,
}

impl MateCost {
    /// 計測つきで `mate_moves_in_1_fast` を呼ぶ
    fn measure(&mut self, pos: &Position, ply: usize) -> Vec<ShogiMove> {
        let t0 = std::time::Instant::now();
        let out = mate_moves_in_1_fast(pos);
        let ns = u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.samples.push((ns, ply >= 90));
        if compare_full_search() {
            let t1 = std::time::Instant::now();
            let full = mate_moves_in_1(pos);
            self.full_ns += u64::try_from(t1.elapsed().as_nanos()).unwrap_or(0);
            self.fast_ns += ns;
            self.compared += 1;
            assert_eq!(
                full.iter().map(|m| m.to_usi()).collect::<HashSet<_>>(),
                out.iter().map(|m| m.to_usi()).collect::<HashSet<_>>(),
            );
        }
        out
    }

    fn report(&self) {
        if self.samples.is_empty() {
            return;
        }
        let pct = |v: &[u64], q: f64| -> f64 {
            if v.is_empty() {
                return 0.0;
            }
            let i = ((v.len() - 1) as f64 * q).round() as usize;
            v[i] as f64 / 1000.0
        };
        let line = |label: &str, mut v: Vec<u64>| {
            v.sort_unstable();
            println!(
                "  {label}: {}回 / p50 {:.1}µs / p95 {:.1}µs / 最大 {:.1}µs",
                v.len(),
                pct(&v, 0.5),
                pct(&v, 0.95),
                pct(&v, 1.0),
            );
        };
        println!("mate_moves_in_1_fast の単発コスト（issue #28 P1-A）:");
        line("全決定点", self.samples.iter().map(|&(ns, _)| ns).collect());
        let endgame: Vec<u64> = self
            .samples
            .iter()
            .filter(|&&(_, e)| e)
            .map(|&(ns, _)| ns)
            .collect();
        if !endgame.is_empty() {
            line("終盤（90手目以降）", endgame);
        }
        if self.compared > 0 {
            println!(
                "  完全版（全合法手）との比: 絞り込み {:.1}µs/回 vs 完全版 {:.1}µs/回 = {:.1}倍速（{}局面）",
                self.fast_ns as f64 / f64::from(self.compared) / 1000.0,
                self.full_ns as f64 / f64::from(self.compared) / 1000.0,
                self.full_ns as f64 / self.fast_ns.max(1) as f64,
                self.compared,
            );
        }
    }
}

/// `TSUITATE_MATE_COST=1` で完全版との突き合わせ計測を有効にする（既定は絞り込み版だけ）
fn compare_full_search() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TSUITATE_MATE_COST").as_deref() == Ok("1"))
}

/// 被詰めろの**エピソード**（連続した相手番の被詰めろを1つに畳んだもの）。
///
/// 手番単位で数えると「1回の危険が何手番続いたか」と「何回危険に入ったか」が
/// 混ざる（issue #28 P0-1: 局が長くなっただけの増加と区別できない）。
struct MateEpisode {
    /// 最初に被詰めろになった相手番の positions インデックス
    start_idx: usize,
    /// 連続した被詰めろ手番の数 = 受ける機会の回数
    turns: usize,
    /// エピソード内の全詰め手の union で分類
    kind: MateKind,
    /// 実際に詰まされたなら、その手が打ちだったか
    executed_drop: Option<bool>,
    /// 危険へ入った手番（`start_idx` の直前の bot 決定点）で真実上あった安全手の本数
    safe_at_entry: Option<usize>,
    /// 安全手が真実上あった手番の数
    safe_turns: usize,
    /// そのうち安全手が bot の候補生成（自駒のみ）に載っていた手番の数
    covered_turns: usize,
    /// 最後に回避できた相手番の positions インデックス
    last_avoidable: Option<usize>,
    /// 決定点が厳密粒子ゼロだった手番の数 / 記録から判定できた手番の数
    blind_turns: usize,
    known_turns: usize,
}

/// bot の各決定点（positions のインデックス）に対応する観測ログの prefix 長。
/// 反則を挟む手番は最初の試行の直前で切る（その手番の入口の視界）
fn decision_prefixes(rec: &GameRecord) -> HashMap<usize, usize> {
    let mut out: HashMap<usize, usize> = HashMap::new();
    for (i, obs) in rec.observations.iter().enumerate() {
        // MyMove の move_number は適用後の値なので -2、MyFoul は手番維持なので -1
        let idx = match obs {
            Observation::MyMove { move_number, .. } => (*move_number as usize).saturating_sub(2),
            Observation::MyFoul { move_number, .. } => (*move_number as usize).saturating_sub(1),
            _ => continue,
        };
        out.entry(idx).or_insert(i);
    }
    out
}

/// 被詰めろエピソードと受けの漏斗（issue #28 P0-2）。
///
/// 被詰めろ局面は相手番なので**一手巻き戻して** bot の直前の決定点を見る。
/// 段ごとに:
/// 1. 真実上、指した後に相手の一手詰めが消える手があったか（理論上限）
/// 2. その手が bot の候補生成（自駒のみ）に載るか
/// 3. その決定点に厳密粒子があったか（2手読みは厳密粒子ゼロには効かない）
fn mate_defense_episodes(
    rec: &GameRecord,
    positions: &[Position],
    bot: Color,
    cost: &mut MateCost,
) -> Vec<MateEpisode> {
    let prefixes = decision_prefixes(rec);
    // 被詰めろの手番・詰め手の分類・一手巻き戻した決定点の安全手は
    // `mate_economy` の共有定義（P0-3 / P0-6 の診断と同じ数え方）
    let played = |i: usize| rec.end.moves.get(i).and_then(|m| parse_usi(&m.usi));
    let mut mates_of = |p: &Position, ply: usize| cost.measure(p, ply);
    let turns = threat_turns(positions, bot, &mut mates_of, &played);

    let mut out = vec![];
    for ep in fold_episodes(&turns) {
        let first = &turns[ep.turns[0]];
        let mut e = MateEpisode {
            start_idx: first.idx,
            turns: ep.turns.len(),
            kind: first.kind,
            executed_drop: None,
            safe_at_entry: first.decision_idx.map(|_| first.safe.len()),
            safe_turns: 0,
            covered_turns: 0,
            last_avoidable: None,
            blind_turns: 0,
            known_turns: 0,
        };
        for &ti in &ep.turns {
            let t = &turns[ti];
            e.kind = e.kind.merge(t.kind);
            if let Some(mv) = &t.executed {
                e.executed_drop = Some(matches!(mv, ShogiMove::Drop { .. }));
            }
            let Some(decision_idx) = t.decision_idx else {
                continue;
            };
            if !t.safe.is_empty() {
                e.safe_turns += 1;
                e.last_avoidable = Some(t.idx);
                // 漏斗の2段目: その安全手が bot の候補生成（自駒のみ）に載るか
                let safe_usi: HashSet<String> = t.safe.iter().map(ShogiMove::to_usi).collect();
                if let Some(&prefix) = prefixes.get(&decision_idx) {
                    let mut log = ObservationLog::default();
                    for prev in &rec.observations[..prefix] {
                        log.record(prev.clone());
                    }
                    let model = GameModel::from_log(bot, &log);
                    let view = view_from_model(
                        &model,
                        positions[decision_idx].in_check(bot),
                        positions[decision_idx].move_number(),
                    );
                    if candidate_moves(&view, &HashSet::new())
                        .iter()
                        .any(|(usi, _)| safe_usi.contains(usi))
                    {
                        e.covered_turns += 1;
                    }
                }
            }
            // 漏斗の3段目: その決定点に厳密粒子があったか
            if let Some(blind) = rec.blind_decisions.get(&decision_idx).copied().flatten() {
                e.known_turns += 1;
                e.blind_turns += usize::from(blind);
            }
        }
        out.push(e);
    }
    out
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("使い方: analyze <records/*.jsonl>");
        std::process::exit(1);
    }

    let mut total_fouls: HashMap<FoulCause, u32> = HashMap::new();
    let mut total_bot_captured = 0.0;
    let mut total_bot_lost = 0.0;
    let mut total_free_losses = 0.0;
    let mut total_exchange_settlements = 0u32;
    let mut total_bad_trades = 0.0;
    let mut total_missed_mates = 0;
    // 被詰めろオラクル: bot が指した後（相手番）の局面で、相手に一手詰めが
    // 存在したか。**相手が実際に見つけたかに依らない**ので、相手が詰みを
    // 実行してこない環境（アリーナの凍結版同士は互いに玉位置が見えない）でも
    // 受けの改善を直接測れる。nofoul オラクルと同じ趣旨の診断
    let mut total_mate_allowed = 0u32;
    let mut total_mate_executed = 0u32;
    let mut games_mate_allowed = 0u32;
    // issue #28 P0-1/P0-2: エピソード単位・排他的分類・手数調整・受けの漏斗
    let mut total_mate_episodes = 0u32;
    let mut total_mate_episodes_endgame = 0u32;
    let mut total_decisions = 0u32;
    let mut total_decisions_endgame = 0u32;
    let mut episode_len_hist = [0u32; 6]; // 1,2,3,4,5,6手番以上
    let mut total_kind_drop_only = 0u32;
    let mut total_kind_board_only = 0u32;
    let mut total_kind_both = 0u32;
    let mut total_executed_drop = 0u32;
    let mut total_executed_board = 0u32;
    let mut total_safe_at_entry = 0u32;
    let mut total_episodes_with_safe = 0u32;
    let mut total_episodes_covered = 0u32;
    let mut total_funnel_blind = 0u32;
    let mut total_funnel_known = 0u32;
    let mut total_first_threat_ply = 0u64;
    let mut total_first_threat_games = 0u32;
    // 攻め／受けを分けた終局分布（P0-1）。全終局に占める詰みの比率ではなく
    // 候補側（bot）の詰み負け率・詰み勝ち率で見る
    let mut games_bot_mated = 0u32;
    let mut games_bot_mates = 0u32;
    // 攻め側の一手詰めの排他的分類（参考値・玉位置は不可視）
    let mut total_missed_drop_only = 0u32;
    let mut total_missed_board_only = 0u32;
    let mut total_missed_both = 0u32;
    let mut mate_cost = MateCost::default();
    // issue #31 P0-1〜P0-3: 王手中の反則経済（粒子不要）
    let mut check_econ = CheckEconomy::default();
    let mut catastrophes = SolverCatastrophes::default();
    let mut total_check_turns = 0;
    let mut total_check_actual_fouls = 0;
    let mut total_check_solved = 0;
    // 王手中の実反則の監査: その瞬間のソルバー p_max 別の (反則数, p_chosen合計)
    let mut check_foul_audit = [(0u32, 0.0f64); 3];
    // 王手駒の即取られ: bot の手が直接王手（動かした/打った駒自身が敵玉へ利く）
    // になり、その駒を相手の直後の手で取られた回数。玉位置ビリーフが外れている
    // ときの典型（信念上の玉へ向けた王手が、実際には守られたマスへ着地している）
    // を数える。取り返せなかったもの（駒損の確定）は別勘定
    let mut total_bot_checks = 0u32;
    let mut total_checker_lost = 0u32;
    let mut total_checker_lost_free = 0u32;
    // 王手の強さ: bot が王手をかけた各局面（開き王手含む・詰みは除く）で
    // 相手に残る合法解消手数（王手中の合法手 = 解消手）。少ないほど相手は
    // 正解を引くまで反則を積みやすい（強い王手。tuyoi_oote の教訓:
    // 解消手1手の王手は反則負けを直接狙える）。[1, 2, 3..=5, 6..=10, 11+]
    let mut check_resolution_hist = [0u32; 5];
    let mut check_resolution_sum = 0u64;
    let mut check_resolution_n = 0u64;
    // 王手の強さの本体（ユーザー指摘 2026-07-29）: 解消手数 K は分母でしかなく、
    // 期待反則数は「受け側が試しうる選択肢 N」との比で決まる。
    // N = 相手の視界（相手駒＋持ち駒のみ。bot 駒を消した盤面）での合法手数
    // = 王手駒が見えないので王手フィルタが掛からない「試しうる手」の上界。
    // 実際に直後の相手手番で出た反則数も対で数え、定義の妥当性を実測で検証する
    let mut chk_options_sum = 0u64;
    let mut chk_actual_fouls = 0u64;
    // (回数, 直後の実反則) を「解消≤2手 × 選択肢の広い/狭い」と「解消≥3手」で分ける
    let mut chk_strong_wide = (0u32, 0u32);
    let mut chk_strong_narrow = (0u32, 0u32);
    let mut chk_weak = (0u32, 0u32);
    // 玉移動の解消手の有無での分割（rei3 のユーザー指摘 2026-07-29）:
    // 玉の移動（逃げ/取り）は受け側が最初に試す自然なクラスで、しかも自陣側への
    // 退路は仮説によらずほぼ確実に合法（見えない駒の利きが通りにくい）。
    // 解消手に玉移動が含まれる王手は K が小さくても反則を稼げないはず
    let mut chk_kesc = [(0u32, 0u32); 4]; // [K≤2玉逃げあり, K≤2なし, K≥3あり, K≥3なし]
    // 手段クラス数（rei2 の第2次元の実測、2026-07-29）: 期待反則数は K の逆数
    // だけでなく「受け側が区別できない手段クラスの数 × 解消手の互いに素性」で
    // 決まるはず。受け側は攻め側の持ち駒を被captureから正確に知っているので、
    // 王手時の攻め側の持ち駒の多様性（歩以外の異なる駒種数）が打ち王手の
    // 仮説クラス数の代理になる。「強い王手が成立する形を作る」中期項
    // （持ち駒の多様性への値付け）に先立つ相関の検証
    // [K≤2, K≥3] × [持ち駒クラス 0〜1 / 2 / 3+] で (回数, 直後の実反則)
    let mut chk_hand = [[(0u32, 0u32); 3]; 2];
    let mut total_recap_ops = 0;
    let mut total_recap_taken = 0;
    let mut total_recap_missed_good = 0;
    let mut games = 0;
    let mut bot_wins = 0;
    let mut p_legal_all: Vec<(f64, bool)> = vec![];
    // 王手中の手番だけの p_legal 較正（王手中の過信の診断）
    let mut p_legal_check: Vec<(f64, bool)> = vec![];
    // 王手中の全決定に対する CheckSolver 単体の p（エンジン p_legal との比較）
    let mut solver_p_check: Vec<(f64, bool)> = vec![];
    // 「真の王手駒の単独仮説なら合法」だった手の手種別 (試行, 実際合法)
    let mut single_hyp = [(0u32, 0u32); 6];
    // 占有マス反則（Blocked/DropOccupied）が、同じ対局内で過去の占有マス
    // 反則が実際に示していたマスと重なっていたか。「反則が起きたマスを
    // 覚えて避ける」系の対策（Guide::occupies/path_blocks）が原理的に
    // 防げる範囲の上限を測る診断
    let mut total_occupancy_fouls = 0u32;
    let mut total_repeat_avoidable = 0u32;
    // 玉移動反則の行き先再訪（`TSUITATE_KING_REPEAT_FOUL_W` の理論上限）。
    // 同手番の同一 USI は foul_tried 済みなので、項が狙うのは手番をまたいだ
    // 汚名マス（解除証拠なし）だけ。raw は同手番の再試行も含む参照値
    let mut total_king_move_fouls = 0u32;
    let mut total_king_repeat_dest = 0u32;
    let mut total_king_stale_repeat = 0u32;
    // 被王手時: 汚名マスへの玉移動が真に合法（解消できる）なのに選ばず反則した回数。
    // KING_REPEAT_FOUL_W が正しい王手解消を沈める機会の上限
    let mut total_check_stale_king_legal = 0u32;
    let mut total_check_stale_king_missed = 0u32;
    let mut total_check_stale_king_used = 0u32;
    // 無意味な往復: bot が指した後の**自陣形**（盤上の自駒＋持ち駒）が、
    // その対局で既に出現していた回数。ついたてでは自分側は完全既知なので
    // ノイズゼロで測れる。「何も起きていないのに同じ形へ戻った」= 手番を
    // 捨てているので、`repeat_penalty_w` が狙う現象そのもの。
    // **この頻度が改善の天井**になる（実測 2026-07-28: 200局で 0.9%）
    let mut total_bot_moves = 0u32;
    let mut total_repeat_configs = 0u32;
    let mut games_with_repeat = 0u32;

    for path in &paths {
        let Some(rec) = load(path) else {
            eprintln!("読めませんでした（終局まで到達していない記録？）: {path}");
            continue;
        };
        games += 1;
        let bot = rec.bot_color;
        let bot_won = matches!(
            (rec.end.result.as_str(), bot),
            ("sente_win", Color::Sente) | ("gote_win", Color::Gote)
        );
        if bot_won {
            bot_wins += 1;
        }
        println!("\n=== {} ===", rec.file);
        println!(
            "bot={:?} ({}) vs {} / 結果: {} ({}) {}",
            bot,
            rec.strategy,
            rec.end.opponent.username,
            rec.end.result,
            rec.end.reason,
            if bot_won { "→ bot勝ち" } else { "→ bot負け" },
        );

        // 反則の原因分類（局面 moveNumber = その時点までに moveNumber-1 手が指されている）
        let mut positions = vec![Position::initial()];
        for m in &rec.end.moves {
            let mut next = positions.last().unwrap().clone();
            let Some(mv) = parse_usi(&m.usi) else {
                eprintln!("  棋譜の手をパースできません: {}", m.usi);
                break;
            };
            next.play_unchecked(&mv);
            positions.push(next);
        }

        // p_legal 較正の分母（全手番と、王手中の手番だけの2系統）。
        // 着手時の局面: MyMove の move_number は適用後の値なので -2、
        // MyFoul は手番維持なので -1（selfplay.rs の moveNumber 規約）
        for &(p, y, mn) in &rec.p_legal_outcomes {
            p_legal_all.push((p, y));
            let idx = (mn as usize).saturating_sub(if y { 2 } else { 1 });
            if positions.get(idx).is_some_and(|pos| pos.in_check(bot)) {
                p_legal_check.push((p, y));
            }
        }

        let mut known_risky_squares: HashSet<Coord> = HashSet::new();
        let mut bot_fouls: Vec<_> = rec
            .end
            .foul_attempts
            .iter()
            .filter(|f| f.by_color == bot)
            .collect();
        bot_fouls.sort_by_key(|f| f.move_number);
        for foul in bot_fouls {
            let idx = (foul.move_number as usize).saturating_sub(1);
            if idx >= positions.len() {
                continue;
            }
            let cause = classify_foul(&positions[idx], foul);
            *total_fouls.entry(cause).or_default() += 1;
            println!(
                "  反則 {}手目 {}: {}",
                foul.move_number,
                foul.usi,
                cause_label(cause)
            );
            if matches!(cause, FoulCause::Blocked | FoulCause::DropOccupied) {
                if let Some(mv) = parse_usi(&foul.usi) {
                    if let Some(sq) = true_occupied_square(&positions[idx], &mv) {
                        total_occupancy_fouls += 1;
                        if known_risky_squares.contains(&sq) {
                            total_repeat_avoidable += 1;
                        }
                        known_risky_squares.insert(sq);
                    }
                }
            }
        }
        {
            let mut seen_king_dests: HashSet<Coord> = HashSet::new();
            let mut prefix = ObservationLog::default();
            for obs in &rec.observations {
                if let Observation::MyFoul { move_number, usi } = obs {
                    if let Some(ShogiMove::Board { from, to, .. }) = parse_usi(usi) {
                        let idx = (*move_number as usize).saturating_sub(1);
                        if positions.get(idx).is_some_and(|pos| {
                            pos.piece_at(from)
                                .is_some_and(|p| p.color == bot && p.role == Role::King)
                        }) {
                            total_king_move_fouls += 1;
                            if seen_king_dests.contains(&to) {
                                total_king_repeat_dest += 1;
                            }
                            let stale = stale_king_foul_dests(&prefix, bot, *move_number);
                            if stale.contains(&to) {
                                total_king_stale_repeat += 1;
                            }
                            seen_king_dests.insert(to);
                        }
                    }
                }
                prefix.record(obs.clone());
            }
        }
        {
            let missed = tally_check_stale_king_missed(
                &rec.observations,
                &positions,
                bot,
                true,
            );
            total_check_stale_king_legal += missed.legal_escape_decisions;
            total_check_stale_king_missed += missed.missed_fouls;
            total_check_stale_king_used += missed.used;
        }
        // 駒の損得: 各手の捕獲を追い、直後の取り返しをペアにする
        let mut bot_captured = 0.0;
        let mut bot_lost = 0.0;
        let mut free_losses: Vec<String> = vec![];
        let mut bad_trades: Vec<String> = vec![];
        // 無意味な往復の頻度（bot 側の自陣形の再出現）。真の局面から
        // bot 側だけを射影して数える（bot は自分側を完全に知っているので、
        // これは bot が実際に持っている情報だけで判定できる量）
        {
            let mut seen: HashMap<Vec<(String, Role)>, u32> = HashMap::new();
            let mut repeats_here = 0u32;
            for (i, m) in rec.end.moves.iter().enumerate() {
                if m.by_color != bot {
                    continue;
                }
                let after = &positions[i + 1.min(positions.len() - 1 - i)];
                let mut own: Vec<(String, Role)> = after
                    .pieces_of(bot)
                    .iter()
                    .map(|p| (p.square.clone(), p.role))
                    .collect();
                own.sort();
                let e = seen.entry(own).or_insert(0);
                total_bot_moves += 1;
                if *e > 0 {
                    total_repeat_configs += 1;
                    repeats_here += 1;
                }
                *e += 1;
            }
            if repeats_here > 0 {
                games_with_repeat += 1;
            }
        }

        for (i, m) in rec.end.moves.iter().enumerate() {
            let pos = &positions[i];
            let Some(mv) = parse_usi(&m.usi) else { break };
            let to = match mv {
                ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
            };
            let captured = pos.piece_at(to).map(|p| p.role);
            let Some(role) = captured else { continue };
            let v = piece_value(role);
            if m.by_color == bot {
                bot_captured += v;
            } else {
                bot_lost += v;
                let exchange_settlement = i > 0
                    && rec.end.moves[i - 1].by_color == bot
                    && parse_usi(&rec.end.moves[i - 1].usi).is_some_and(|pm| {
                        let prev_to = match pm {
                            ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
                        };
                        prev_to == to && positions[i - 1].piece_at(prev_to).is_some()
                    });
                // 取り返したか（次の bot の正規手が同じマスを取ったか）
                let recaptured = rec.end.moves.get(i + 1).is_some_and(|n| {
                    n.by_color == bot
                        && parse_usi(&n.usi).is_some_and(|nm| match nm {
                            ShogiMove::Board { to: t, .. } | ShogiMove::Drop { to: t, .. } => {
                                t == to && positions[i + 1].piece_at(t).is_some()
                            }
                        })
                });
                if exchange_settlement {
                    total_exchange_settlements += 1;
                }
                if !recaptured && !exchange_settlement {
                    // 守られていたのに取り返さなかったのか、そもそも守っていなかったのか
                    free_losses.push(format!(
                        "{}手目 {} で {:?}(価値{v:.0}) を只取られ",
                        i + 1,
                        m.usi,
                        role
                    ));
                }
            }
            // bot が取った直後に取り返された交換のネット
            if m.by_color == bot {
                if let Some(n) = rec.end.moves.get(i + 1) {
                    if n.by_color != bot {
                        if let Some(nm) = parse_usi(&n.usi) {
                            let nt = match nm {
                                ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
                            };
                            if nt == to {
                                if let Some(lost) = positions[i + 1].piece_at(nt) {
                                    let net = v - piece_value(lost.role);
                                    if net < -1.5 {
                                        bad_trades.push(format!(
                                            "{}手目 {}: {:?}(価値{:.0}) を取ったが {:?}(価値{:.0}) を取り返され ネット{net:+.0}",
                                            i + 1,
                                            m.usi,
                                            role,
                                            v,
                                            lost.role,
                                            piece_value(lost.role),
                                        ));
                                        total_bad_trades += -net;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        total_bot_captured += bot_captured;
        total_bot_lost += bot_lost;
        for l in &free_losses {
            println!("  {l}");
        }
        total_free_losses += free_losses.len() as f64;
        for t in &bad_trades {
            println!("  {t}");
        }
        println!("  駒得収支: 取った {bot_captured:.0} / 取られた {bot_lost:.0}（歩=1換算）");

        // 王手駒の即取られ（変異救済 / 玉位置ビリーフ系の効果測定用）
        for (i, m) in rec.end.moves.iter().enumerate() {
            if m.by_color != bot {
                continue;
            }
            let Some(mv) = parse_usi(&m.usi) else { break };
            let to = match mv {
                ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
            };
            let Some(after) = positions.get(i + 1) else { break };
            if !after.in_check(bot.other()) {
                continue;
            }
            // 王手の強さ（相手の合法解消手数 K と、受け側の選択肢 N）。
            // 詰みは outcome 側で数える
            if after.outcome().is_none() {
                let resolutions = after.legal_moves();
                let k = resolutions.len();
                // 解消手に玉の移動（逃げ/取り）が含まれるか
                let king_escape = resolutions.iter().any(|rm| {
                    matches!(rm, ShogiMove::Board { from, .. }
                        if after.piece_at(*from).is_some_and(|p| p.role == Role::King))
                });
                let bucket = match k {
                    0 | 1 => 0,
                    2 => 1,
                    3..=5 => 2,
                    6..=10 => 3,
                    _ => 4,
                };
                check_resolution_hist[bucket] += 1;
                check_resolution_sum += k as u64;
                check_resolution_n += 1;
                // N: 相手の視界（bot 駒を消した盤面）で試せる「異なる確かめ」の数。
                // 王手駒が見えないので王手解消フィルタは掛からない。
                // 「マス X を（駒を問わず）取りに行く/塞ぐ」は1回の実験なので、
                // 玉以外の移動は行き先ごとに1、打ちも行き先ごとに1へ重複排除する
                // （玉の移動だけは逃げ/取りとして別勘定。成/不成も自然に統合）
                let mut view = after.clone();
                for file in 1..=9i8 {
                    for rank in 1..=9i8 {
                        let sq = Coord { file, rank };
                        if view.piece_at(sq).is_some_and(|p| p.color == bot) {
                            view.set(sq, None);
                        }
                    }
                }
                let mut probes: HashSet<(u8, Coord)> = HashSet::new();
                for vm in view.legal_moves() {
                    let key = match vm {
                        ShogiMove::Board { from, to, .. } => {
                            let is_king =
                                view.piece_at(from).is_some_and(|p| p.role == Role::King);
                            (u8::from(is_king), to)
                        }
                        ShogiMove::Drop { to, .. } => (2u8, to),
                    };
                    probes.insert(key);
                }
                let n_opts = probes.len();
                chk_options_sum += n_opts as u64;
                // 実際に直後の相手手番で出た反則（move_number = i+2 手目）
                let actual = rec
                    .end
                    .foul_attempts
                    .iter()
                    .filter(|f| f.by_color != bot && f.move_number as usize == i + 2)
                    .count() as u32;
                chk_actual_fouls += u64::from(actual);
                let slot = if k <= 2 {
                    if n_opts >= 10 {
                        &mut chk_strong_wide
                    } else {
                        &mut chk_strong_narrow
                    }
                } else {
                    &mut chk_weak
                };
                slot.0 += 1;
                slot.1 += actual;
                let kesc_idx = match (k <= 2, king_escape) {
                    (true, true) => 0,
                    (true, false) => 1,
                    (false, true) => 2,
                    (false, false) => 3,
                };
                chk_kesc[kesc_idx].0 += 1;
                chk_kesc[kesc_idx].1 += actual;
                let hand_classes = [
                    Role::Lance,
                    Role::Knight,
                    Role::Silver,
                    Role::Gold,
                    Role::Bishop,
                    Role::Rook,
                ]
                .iter()
                .filter(|&&r| after.hand_count(bot, r) > 0)
                .count();
                let hc = match hand_classes {
                    0 | 1 => 0,
                    2 => 1,
                    _ => 2,
                };
                chk_hand[usize::from(k > 2)][hc].0 += 1;
                chk_hand[usize::from(k > 2)][hc].1 += actual;
                if k <= 2 {
                    println!(
                        "  強い王手 {}手目 {}: 解消{k}手{} / 選択肢{n_opts}手 / 直後の反則{actual}回",
                        i + 1,
                        m.usi,
                        if king_escape { "（玉逃げ可）" } else { "（玉逃げ不可）" },
                    );
                }
            }
            let Some(opp_king) = after.king_square(bot.other()) else {
                continue;
            };
            // 以降は直接王手のみ（開き王手は動かした駒が王手駒ではない）
            if !after.attacks(to, opp_king) {
                continue;
            }
            total_bot_checks += 1;
            let taken = rec.end.moves.get(i + 1).is_some_and(|n| {
                n.by_color != bot
                    && parse_usi(&n.usi).is_some_and(|nm| match nm {
                        ShogiMove::Board { to: t, .. } | ShogiMove::Drop { to: t, .. } => t == to,
                    })
            });
            if !taken {
                continue;
            }
            total_checker_lost += 1;
            let recaptured = rec.end.moves.get(i + 2).is_some_and(|n| {
                n.by_color == bot
                    && parse_usi(&n.usi).is_some_and(|nm| match nm {
                        ShogiMove::Board { to: t, .. } | ShogiMove::Drop { to: t, .. } => t == to,
                    })
            });
            if !recaptured {
                total_checker_lost_free += 1;
            }
            println!(
                "  王手駒の即取られ {}手目 {}: {:?}{}",
                i + 1,
                m.usi,
                after.piece_at(to).map(|p| p.role),
                if recaptured {
                    "（取り返しあり）"
                } else {
                    "（取り返しなし）"
                }
            );
        }

        // issue #31 P0-1/P0-2: 王手中の手番の型分類と、手種別・順番別の較正。
        // **分母は反則0の手番も含む全王手手番**（新しいプローブを足す害の
        // estimand がそこなので、反則した手番だけを数えてはいけない）
        let econ_turns = check_turns(
            &rec.observations,
            &positions,
            bot,
            &rec.p_legal_by_attempt,
        );
        for turn in &econ_turns {
            if turn.fouls() == 0 {
                continue;
            }
            println!(
                "  王手手番 {}手目（残り反則{}）: {} → 型 {} / 受理手の開始時 {}",
                turn.move_number,
                10u32.saturating_sub(turn.fouls_before),
                turn.sequence(),
                turn.turn_type().label(),
                match (turn.accepted_legal_at_entry, turn.accepted_rank_at_entry) {
                    (Some(true), Some(r)) => format!("合法・ソルバーp {r}位"),
                    (Some(true), None) => "合法".to_string(),
                    (Some(false), _) => "反則（情報が要った）".to_string(),
                    (None, _) => "受理手なし".to_string(),
                },
            );
        }
        check_econ.add_game(&econ_turns);

        // 王手ソルバーの再現検証（王手中に反則した手番それぞれを指し直す）
        let (tested, actual, sim) = simulate_check_solver(&rec, &positions, bot, &mut catastrophes);
        total_check_turns += tested;
        total_check_actual_fouls += actual;
        total_check_solved += sim;
        let audit =
            audit_check_fouls(&rec, &positions, bot, &mut solver_p_check, &mut single_hyp);
        for (t, a) in check_foul_audit.iter_mut().zip(audit) {
            t.0 += a.0;
            t.1 += a.1;
        }

        // 取り返し機会: 相手に駒を取られた直後の bot 手番で、そのマスを合法に
        // 取り返せたか（bot は取られたマス = 相手駒の現在地を正確に知っている）
        for (i, m) in rec.end.moves.iter().enumerate() {
            if m.by_color == bot {
                continue;
            }
            let Some(mv) = parse_usi(&m.usi) else { break };
            let to = match mv {
                ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
            };
            if positions[i].piece_at(to).is_none() {
                continue; // 捕獲ではない
            }
            let Some(after) = positions.get(i + 1) else { break };
            if after.turn() != bot || after.outcome().is_some() {
                continue;
            }
            total_recap_ops += 1;
            let attacker_value = piece_value(after.piece_at(to).map(|p| p.role).unwrap());
            let recaps: Vec<ShogiMove> = after
                .legal_moves()
                .into_iter()
                .filter(|lm| matches!(lm, ShogiMove::Board { to: t, .. } if *t == to))
                .collect();
            let actually = rec.end.moves.get(i + 1).and_then(|n| parse_usi(&n.usi));
            let took = actually.is_some_and(|am| match am {
                ShogiMove::Board { to: t, .. } | ShogiMove::Drop { to: t, .. } => t == to,
            });
            if took {
                total_recap_taken += 1;
            } else if let Some((best, net)) = recaps
                .iter()
                .map(|mv| {
                    // 取り返し後にさらに取り返されるか（真の局面で）
                    let mut probe = after.clone();
                    let own = match mv {
                        ShogiMove::Board { from, .. } => {
                            after.piece_at(*from).map(|p| piece_value(p.role)).unwrap_or(0.0)
                        }
                        ShogiMove::Drop { .. } => 0.0,
                    };
                    probe.play_unchecked(mv);
                    let exposed = probe.is_attacked(to, bot.other());
                    let net = attacker_value - if exposed { own } else { 0.0 };
                    (mv, net)
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
            {
                if net > 0.5 {
                    total_recap_missed_good += 1;
                    println!(
                        "  取り返し逃し {}手目: {} で {:.0} を回収できた（推定ネット{net:+.0}）が {} を選択",
                        i + 2,
                        best.to_usi(),
                        attacker_value,
                        rec.end
                            .moves
                            .get(i + 1)
                            .map(|n| n.usi.as_str())
                            .unwrap_or("-"),
                    );
                }
            }
        }

        // 終局の内訳（P0-1: 全終局に占める詰み比率でなく、候補側の詰み負け／詰み勝ち）
        if let Some(Outcome::Checkmate { winner }) = positions.last().and_then(|p| p.outcome()) {
            if winner == bot {
                games_bot_mates += 1;
            } else {
                games_bot_mated += 1;
            }
        }

        // 詰み逃し: bot 手番の各局面で1手詰みがあったか
        for (i, pos) in positions.iter().enumerate() {
            if pos.turn() != bot {
                continue;
            }
            if pos.outcome().is_some() {
                break;
            }
            let mates = mate_cost.measure(pos, i + 1);
            let Some(kind) = MateKind::of(&mates) else {
                continue;
            };
            let played = rec.end.moves.get(i).map(|m| m.usi.as_str()).unwrap_or("-");
            // 実際に詰ませた手なら逃していない
            if i + 1 == positions.len() - 1 && positions.last().unwrap().outcome().is_some() {
                continue;
            }
            println!(
                "  1手詰みが存在 {}手目: {}（{}・実際は {played}。玉位置は不可視なので参考値）",
                i + 1,
                mates[0].to_usi(),
                kind.label(),
            );
            total_missed_mates += 1;
            match kind {
                MateKind::DropOnly => total_missed_drop_only += 1,
                MateKind::BoardOnly => total_missed_board_only += 1,
                MateKind::Both => total_missed_both += 1,
            }
        }

        // 被詰めろ（issue #28 P0-1/P0-2）: 相手番の各局面で相手に一手詰めが
        // あったかを**エピソード単位**で畳み、受けの漏斗（真実上の受けの有無 →
        // 候補生成に載るか → 厳密粒子があったか）と詰め手の排他的分類を出す
        let episodes = mate_defense_episodes(&rec, &positions, bot, &mut mate_cost);
        // 手数調整のための分母: bot の決定点（相手番でない・終局前）
        for (i, pos) in positions.iter().enumerate() {
            if pos.turn() != bot || pos.outcome().is_some() {
                continue;
            }
            total_decisions += 1;
            if i + 1 >= 90 {
                total_decisions_endgame += 1;
            }
        }
        if !episodes.is_empty() {
            games_mate_allowed += 1;
            total_first_threat_ply += episodes[0].start_idx as u64 + 1;
            total_first_threat_games += 1;
        }
        for ep in &episodes {
            total_mate_episodes += 1;
            total_mate_allowed += ep.turns as u32;
            if ep.start_idx + 1 >= 90 {
                total_mate_episodes_endgame += 1;
            }
            episode_len_hist[(ep.turns - 1).min(episode_len_hist.len() - 1)] += 1;
            match ep.kind {
                MateKind::DropOnly => total_kind_drop_only += 1,
                MateKind::BoardOnly => total_kind_board_only += 1,
                MateKind::Both => total_kind_both += 1,
            }
            if let Some(by_drop) = ep.executed_drop {
                total_mate_executed += 1;
                if by_drop {
                    total_executed_drop += 1;
                } else {
                    total_executed_board += 1;
                }
            }
            if ep.safe_at_entry.is_some_and(|n| n > 0) {
                total_safe_at_entry += 1;
            }
            if ep.safe_turns > 0 {
                total_episodes_with_safe += 1;
            }
            if ep.covered_turns > 0 {
                total_episodes_covered += 1;
            }
            total_funnel_blind += ep.blind_turns as u32;
            total_funnel_known += ep.known_turns as u32;
            println!(
                "  被詰めろ {}手目〜 {}手番（{}{}）: 入口の安全手 {} / 受けられた手番 {}（候補生成に載った {}）{}{}",
                ep.start_idx + 1,
                ep.turns,
                ep.kind.label(),
                if ep.known_turns > 0 {
                    format!("・厳密粒子ゼロ {}/{}", ep.blind_turns, ep.known_turns)
                } else {
                    String::new()
                },
                ep.safe_at_entry
                    .map_or("-".to_string(), |n| n.to_string()),
                ep.safe_turns,
                ep.covered_turns,
                ep.last_avoidable
                    .map_or(String::new(), |i| format!("・最後に回避できたのは {}手目", i + 1)),
                match ep.executed_drop {
                    Some(true) => "・実行された(打ち)",
                    Some(false) => "・実行された(盤上)",
                    None => "",
                },
            );
        }
    }

    println!("\n=== 集計（{games}局 bot {bot_wins}勝）===");
    println!("反則の原因:");
    let mut causes: Vec<_> = total_fouls.iter().collect();
    causes.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
    for (cause, n) in causes {
        println!("  {:>3}回  {}", n, cause_label(*cause));
    }
    println!("駒得収支合計: 取った {total_bot_captured:.0} / 取られた {total_bot_lost:.0}");
    println!(
        "只取られ回数（交換決済除外）: {total_free_losses:.0} / 交換決済: {total_exchange_settlements}件 / 損な交換の累計損失: {total_bad_trades:.0}"
    );
    println!(
        "取り返し: 機会{total_recap_ops}回中 実行{total_recap_taken}回 / 得だったのに逃した{total_recap_missed_good}回"
    );
    println!(
        "1手詰みの存在（参考値・玉位置は不可視）: {total_missed_mates}回（打ちのみ {total_missed_drop_only} / 盤上のみ {total_missed_board_only} / 両方 {total_missed_both}）"
    );
    println!("\n--- 詰み経済（issue #28 P0-1/P0-2）---");
    let pct = |a: u32, b: u32| -> f64 {
        if b == 0 { 0.0 } else { f64::from(a) * 100.0 / f64::from(b) }
    };
    println!(
        "終局: bot の詰み負け {games_bot_mated}/{games}局 ({:.1}%) / 詰み勝ち {games_bot_mates}/{games}局 ({:.1}%)",
        pct(games_bot_mated, games),
        pct(games_bot_mates, games),
    );
    println!(
        "被詰めろ: {total_mate_episodes}エピソード / {total_mate_allowed}手番 / {games_mate_allowed}局（実際に詰まされた {total_mate_executed}回 = 打ち {total_executed_drop} / 盤上移動 {total_executed_board}）"
    );
    println!(
        "  露出（手数調整）: bot の決定点 {total_decisions}（終盤90手以降 {total_decisions_endgame}）→ 100決定点あたり {:.2}エピソード / 終盤 {:.2}エピソード",
        if total_decisions == 0 { 0.0 } else { f64::from(total_mate_episodes) * 100.0 / f64::from(total_decisions) },
        if total_decisions_endgame == 0 { 0.0 } else { f64::from(total_mate_episodes_endgame) * 100.0 / f64::from(total_decisions_endgame) },
    );
    if total_first_threat_games > 0 {
        println!(
            "  初回被詰めろの手数: 平均 {:.1}手目（{total_first_threat_games}局）",
            total_first_threat_ply as f64 / f64::from(total_first_threat_games),
        );
    }
    println!(
        "  エピソード長（受ける機会の手番数）: 1:{} 2:{} 3:{} 4:{} 5:{} 6以上:{}",
        episode_len_hist[0], episode_len_hist[1], episode_len_hist[2],
        episode_len_hist[3], episode_len_hist[4], episode_len_hist[5],
    );
    println!(
        "  詰め手の排他的分類: 打ちのみ {total_kind_drop_only} / 盤上のみ {total_kind_board_only} / 両方 {total_kind_both}（**現行の mate_risk_w は打ち詰みしか見ない**）"
    );
    println!(
        "  受けの漏斗: 入口で真実上の安全手あり {total_safe_at_entry}/{total_mate_episodes} ({:.1}%) → どこかの手番で受けられた {total_episodes_with_safe} → その安全手が候補生成に載った {total_episodes_covered}",
        pct(total_safe_at_entry, total_mate_episodes),
    );
    if total_funnel_known > 0 {
        println!(
            "  被詰めろ手番の決定点が厳密粒子ゼロ: {total_funnel_blind}/{total_funnel_known} ({:.1}%。2手読みはここには効かない)",
            pct(total_funnel_blind, total_funnel_known),
        );
    }
    mate_cost.report();
    println!(
        "王手駒の即取られ: {total_checker_lost}回（うち取り返しなし {total_checker_lost_free}回）/ 直接王手 {total_bot_checks}回"
    );
    if check_resolution_n > 0 {
        println!(
            "王手の強さ（相手の合法解消手数K）: 1手:{} 2手:{} 3〜5手:{} 6〜10手:{} 11手以上:{} / 平均 {:.1}手",
            check_resolution_hist[0],
            check_resolution_hist[1],
            check_resolution_hist[2],
            check_resolution_hist[3],
            check_resolution_hist[4],
            check_resolution_sum as f64 / check_resolution_n as f64,
        );
        let per = |c: (u32, u32)| -> String {
            if c.0 == 0 {
                "-".into()
            } else {
                format!("{}回→反則{}回 ({:.2}回/王手)", c.0, c.1, f64::from(c.1) / f64::from(c.0))
            }
        };
        println!(
            "  受け側の選択肢N 平均 {:.1}手 / 王手直後の実反則 合計{} ({:.2}回/王手)",
            chk_options_sum as f64 / check_resolution_n as f64,
            chk_actual_fouls,
            chk_actual_fouls as f64 / check_resolution_n as f64,
        );
        println!(
            "  解消≤2手×選択肢≥10: {} / 解消≤2手×選択肢<10: {} / 解消≥3手: {}",
            per(chk_strong_wide),
            per(chk_strong_narrow),
            per(chk_weak),
        );
        println!(
            "  玉移動の解消手の有無: K≤2玉逃げあり {} / K≤2玉逃げなし {} / K≥3あり {} / K≥3なし {}",
            per(chk_kesc[0]),
            per(chk_kesc[1]),
            per(chk_kesc[2]),
            per(chk_kesc[3]),
        );
        println!(
            "  攻め側の持ち駒クラス数（歩以外の駒種数）:\n    K≤2: 0〜1種 {} / 2種 {} / 3種以上 {}\n    K≥3: 0〜1種 {} / 2種 {} / 3種以上 {}",
            per(chk_hand[0][0]),
            per(chk_hand[0][1]),
            per(chk_hand[0][2]),
            per(chk_hand[1][0]),
            per(chk_hand[1][1]),
            per(chk_hand[1][2]),
        );
    }
    if total_occupancy_fouls > 0 {
        println!(
            "占有マス反則（打ちマス/経路封鎖）の再訪率: {total_repeat_avoidable}/{total_occupancy_fouls}（同一局内で過去の占有反則マスと一致。反則マスを覚える対策の理論上限）"
        );
    }
    if total_king_move_fouls > 0 {
        println!(
            "玉移動反則の行き先再訪: {total_king_repeat_dest}/{total_king_move_fouls}（同一局内で既に同じ to へ玉反則。同手番の再試行含む） / うち手番またぎ汚名マス {total_king_stale_repeat}（解除証拠なし。KING_REPEAT_FOUL_W の発火上限）"
        );
    }
    if total_check_stale_king_legal > 0 {
        println!(
            "王手中の汚名マス玉逃げ: 真に解消できる決定 {total_check_stale_king_legal} / その合法手で解消 {total_check_stale_king_used} / 選ばず反則 {total_check_stale_king_missed}（KING_REPEAT_FOUL_W が正しい王手解消を沈める機会）"
        );
    }
    if total_bot_moves > 0 {
        println!(
            "無意味な往復（自陣形の再出現）: {total_repeat_configs}/{total_bot_moves}手 ({:.1}%) / {games_with_repeat}局（自分側は完全既知なのでノイズゼロ。repeat_penalty_w が狙う現象の頻度＝改善の天井）",
            100.0 * f64::from(total_repeat_configs) / f64::from(total_bot_moves)
        );
    }
    if total_check_actual_fouls > 0 {
        println!(
            "王手中の反則: {total_check_turns}手番で実際 {total_check_actual_fouls}回 → ソルバー方策なら {total_check_solved}回"
        );
        let audited: u32 = check_foul_audit.iter().map(|c| c.0).sum();
        if audited > 0 {
            let fmt = |c: (u32, f64)| -> String {
                if c.0 == 0 {
                    "-".into()
                } else {
                    format!("{}回 (選択手の平均p {:.2})", c.0, c.1 / f64::from(c.0))
                }
            };
            println!(
                "  反則時点のソルバー最善p_max別: ≥0.9 {} / 0.5〜0.9 {} / <0.5 {}",
                fmt(check_foul_audit[0]),
                fmt(check_foul_audit[1]),
                fmt(check_foul_audit[2]),
            );
        }
    }
    check_econ.report();
    catastrophes.report(SOLVER_CATASTROPHE);

    // p_legal の較正（C-7 P3）: 選択手の合法確率予測 vs 実際の受理/反則。
    // Brier = mean((p-y)^2)（小さいほど良い）。参考: 常に基底率を答える予測の Brier
    if !p_legal_all.is_empty() {
        let n = p_legal_all.len() as f64;
        let base_rate = p_legal_all.iter().filter(|(_, y)| *y).count() as f64 / n;
        let brier: f64 = p_legal_all
            .iter()
            .map(|(p, y)| {
                let y = if *y { 1.0 } else { 0.0 };
                (p - y) * (p - y)
            })
            .sum::<f64>()
            / n;
        let base_brier = base_rate * (1.0 - base_rate);
        let logloss: f64 = p_legal_all
            .iter()
            .map(|(p, y)| {
                let p = p.clamp(1e-6, 1.0 - 1e-6);
                if *y { -p.ln() } else { -(1.0 - p).ln() }
            })
            .sum::<f64>()
            / n;
        println!(
            "p_legal 較正（{}手 合法率{:.1}%）: Brier {:.4}（基底率予測 {:.4}）/ logloss {:.4}",
            p_legal_all.len(),
            base_rate * 100.0,
            brier,
            base_brier,
            logloss,
        );
    }
    // 王手中の手番だけの較正。全体より悪ければ「王手中の p_legal 過信」が
    // 王手中反則（ソルバー方策とのギャップ）の説明になる
    if !p_legal_check.is_empty() {
        let n = p_legal_check.len() as f64;
        let base_rate = p_legal_check.iter().filter(|(_, y)| *y).count() as f64 / n;
        let brier: f64 = p_legal_check
            .iter()
            .map(|(p, y)| {
                let y = if *y { 1.0 } else { 0.0 };
                (p - y) * (p - y)
            })
            .sum::<f64>()
            / n;
        let mean_p: f64 = p_legal_check.iter().map(|(p, _)| p).sum::<f64>() / n;
        println!(
            "p_legal 較正・王手中のみ（{}手 合法率{:.1}% 平均予測{:.1}%）: Brier {:.4}（基底率予測 {:.4}）",
            p_legal_check.len(),
            base_rate * 100.0,
            mean_p * 100.0,
            brier,
            base_rate * (1.0 - base_rate),
        );
    }
    if !solver_p_check.is_empty() {
        let n = solver_p_check.len() as f64;
        let base_rate = solver_p_check.iter().filter(|(_, y)| *y).count() as f64 / n;
        let brier: f64 = solver_p_check
            .iter()
            .map(|(p, y)| {
                let y = if *y { 1.0 } else { 0.0 };
                (p - y) * (p - y)
            })
            .sum::<f64>()
            / n;
        let mean_p: f64 = solver_p_check.iter().map(|(p, _)| p).sum::<f64>() / n;
        println!(
            "CheckSolver単体p 較正・王手中のみ（{}手 合法率{:.1}% 平均予測{:.1}%）: Brier {:.4}（基底率予測 {:.4}）",
            solver_p_check.len(),
            base_rate * 100.0,
            mean_p * 100.0,
            brier,
            base_rate * (1.0 - base_rate),
        );
    }
    if single_hyp.iter().any(|c| c.0 > 0) {
        let rate = |c: (u32, u32)| -> String {
            if c.0 == 0 {
                "-".into()
            } else {
                format!("{}/{} ({:.0}%)", c.1, c.0, 100.0 * f64::from(c.1) / f64::from(c.0))
            }
        };
        println!(
            "単独仮説合法→実際合法の率（真の王手駒を置いた単独盤面で合法だった手。ソルバー legal_under の過信の実測）:\n  玉で王手駒捕獲 {} / 玉の後退（自陣方向） {} / 玉の横・前進 {} / 玉以外の捕獲 {} / 打ち {} / その他 {}",
            rate(single_hyp[0]),
            rate(single_hyp[1]),
            rate(single_hyp[2]),
            rate(single_hyp[3]),
            rate(single_hyp[4]),
            rate(single_hyp[5]),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tsuitate_bot::board::parse_usi_square;
    use tsuitate_bot::shogi::Piece;

    /// 反則を挟む手番の `blind_decisions` は**最初の選択**だけを見る。
    /// 初回が `debug: null`（定跡手・旧記録）なら、その手番は判定不能のままで、
    /// 反則後の再選択の `sample_slots` で埋めてはいけない（PR #29 レビュー指摘）
    #[test]
    fn 反則後の再選択で最初の決定の粒子状態を埋めない() {
        let jsonl = r#"{"type":"match","your_color":"sente","strategy":"estimator","game_id":"t"}
{"type":"chose","move_number":12,"usi":"7g7f","debug":null}
{"type":"obs","event":{"kind":"my_foul","move_number":12,"usi":"7g7f"}}
{"type":"chose","move_number":12,"usi":"2g2f","debug":{"sample_slots":426}}
{"type":"obs","event":{"kind":"my_move","move_number":13,"usi":"2g2f","captured":null}}
{"type":"chose","move_number":14,"usi":"2f2e","debug":{"sample_slots":0}}
{"type":"obs","event":{"kind":"my_move","move_number":15,"usi":"2f2e","captured":null}}
{"type":"end","payload":{"result":"draw","reason":"draw","finalSfen":"","moves":[],"foulAttempts":[],"ratingChange":{"you":{"before":0,"after":0},"opponent":{"before":0,"after":0}},"opponent":{"username":"x","rating":0}}}
"#;
        let rec = parse_record("t.jsonl", jsonl).expect("記録をパースできる");
        // 12手目 = positions[11]: 初回に sample_slots が無いので判定不能のまま
        assert_eq!(rec.blind_decisions.get(&11), Some(&None));
        // 14手目 = positions[13]: 初回に値があるのでそのまま使う
        assert_eq!(rec.blind_decisions.get(&13), Some(&Some(true)));
    }

    fn sq(usi: &str) -> Coord {
        parse_usi_square(usi).unwrap()
    }

    /// 先手玉5i・後手飛5d・後手玉8a。5筋の王手。
    /// 合法逃げは 4i/6i/4h/6h。5h は筋上で非合法。
    fn rook_file_check() -> Position {
        let mut pos = Position::empty(Color::Sente);
        pos.set(
            sq("5i"),
            Some(Piece {
                color: Color::Sente,
                role: Role::King,
            }),
        );
        pos.set(
            sq("5d"),
            Some(Piece {
                color: Color::Gote,
                role: Role::Rook,
            }),
        );
        pos.set(
            sq("8a"),
            Some(Piece {
                color: Color::Gote,
                role: Role::King,
            }),
        );
        pos
    }

    fn positions_at(idx: usize, pos: Position) -> Vec<Position> {
        let mut v = vec![Position::empty(Color::Sente); idx + 1];
        v[idx] = pos;
        v
    }

    #[test]
    fn 汚名マスへの玉逃げが真に合法なら列挙する() {
        let pos = rook_file_check();
        assert!(pos.in_check(Color::Sente));
        assert!(pos.is_legal(&parse_usi("5i4h").unwrap()));
        assert!(!pos.is_legal(&parse_usi("5i5h").unwrap()));
        let stale = HashSet::from([sq("4h"), sq("5h")]);
        let mut legal = legal_stale_king_escapes(&pos, Color::Sente, &stale);
        legal.sort();
        assert_eq!(legal, vec![sq("4h")]);
    }

    #[test]
    fn 王手中に合法な汚名マス玉逃げがあるのに反則したらmissed() {
        let positions = positions_at(11, rook_file_check());
        let observations = vec![
            Observation::MyFoul {
                move_number: 10,
                usi: "5i4h".into(),
            },
            Observation::OpponentMoved {
                move_number: 11,
                captured_my_piece_at: None,
            },
            Observation::MyFoul {
                move_number: 12,
                usi: "5i5h".into(),
            },
        ];
        let t = tally_check_stale_king_missed(&observations, &positions, Color::Sente, false);
        assert_eq!(t.legal_escape_decisions, 1);
        assert_eq!(t.missed_fouls, 1);
        assert_eq!(t.used, 0);
    }

    #[test]
    fn 王手中に合法な汚名マス玉逃げを選んだらused() {
        let positions = positions_at(11, rook_file_check());
        let observations = vec![
            Observation::MyFoul {
                move_number: 10,
                usi: "5i4h".into(),
            },
            Observation::OpponentMoved {
                move_number: 11,
                captured_my_piece_at: None,
            },
            Observation::MyMove {
                move_number: 13,
                usi: "5i4h".into(),
                captured: None,
            },
        ];
        let t = tally_check_stale_king_missed(&observations, &positions, Color::Sente, false);
        assert_eq!(t.legal_escape_decisions, 1);
        assert_eq!(t.missed_fouls, 0);
        assert_eq!(t.used, 1);
    }

    #[test]
    fn 同手番の玉反則は汚名に入らないのでmissedにしない() {
        let positions = positions_at(11, rook_file_check());
        let observations = vec![
            Observation::MyFoul {
                move_number: 12,
                usi: "5i4h".into(),
            },
            Observation::MyMove {
                move_number: 13,
                usi: "5i6i".into(),
                captured: None,
            },
        ];
        let t = tally_check_stale_king_missed(&observations, &positions, Color::Sente, false);
        assert_eq!(t.legal_escape_decisions, 0);
        assert_eq!(t.missed_fouls, 0);
        assert_eq!(t.used, 0);
    }

    #[test]
    fn 汚名が非合法マスだけならmissedにしない() {
        let positions = positions_at(11, rook_file_check());
        let observations = vec![
            Observation::MyFoul {
                move_number: 10,
                usi: "5i5h".into(),
            },
            Observation::OpponentMoved {
                move_number: 11,
                captured_my_piece_at: None,
            },
            Observation::MyFoul {
                move_number: 12,
                usi: "5i4i".into(),
            },
        ];
        let t = tally_check_stale_king_missed(&observations, &positions, Color::Sente, false);
        assert_eq!(t.legal_escape_decisions, 0);
        assert_eq!(t.missed_fouls, 0);
    }

    #[test]
    fn 王手でなければ数えない() {
        let mut pos = Position::empty(Color::Sente);
        pos.set(
            sq("5i"),
            Some(Piece {
                color: Color::Sente,
                role: Role::King,
            }),
        );
        pos.set(
            sq("8a"),
            Some(Piece {
                color: Color::Gote,
                role: Role::King,
            }),
        );
        let positions = positions_at(11, pos);
        let observations = vec![
            Observation::MyFoul {
                move_number: 10,
                usi: "5i4h".into(),
            },
            Observation::OpponentMoved {
                move_number: 11,
                captured_my_piece_at: None,
            },
            Observation::MyFoul {
                move_number: 12,
                usi: "5i5h".into(),
            },
        ];
        let t = tally_check_stale_king_missed(&observations, &positions, Color::Sente, false);
        assert_eq!(t.legal_escape_decisions, 0);
        assert_eq!(t.missed_fouls, 0);
        assert_eq!(t.used, 0);
    }
}
