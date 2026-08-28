//! **王手中の反則経済 P0-5 の共有定義**（issue #31。runtime には何も入らない）。
//!
//! 王手中の1手番を「候補 × p_legal × 価格」の小さな世界として取り出し、
//! 方策（現行 / α 価格 / β 動的継続価値 / β-order / ソルバー貪欲）を
//! **真実に対して裁定しながら**回すための道具を置く。
//!
//! ここに置くのは3つ:
//!
//! 1. [`PolicyMove`] … 1候補の最小の入力。`score = combine_score(gain, p, c) +
//!    foul_probe + adjust` を **p と c の両方**について厳密に引き直せる形
//!    （P0-4 の [`crate::check_economy::PricedMove`] は価格しか動かせない）
//! 2. [`ShadowUpdater`] … **p-only shadow update**。仮想の反則を観測したあとの
//!    p_legal を「m が合法だった粒子を落とす＋ソルバーを反則込みで作り直す」
//!    だけで作る（gain・リスク・`removal_term` は再計算しない）。
//!    正解基準は実再決定（`Estimator::update` を通す）で、両者の差は
//!    P0-5 の**近似通過条件**として測る
//! 3. [`Policy`] と [`simulate`] … 方策を反則するたび更新しながら真実で裁定する
//!
//! **p_legal のブレンドは `strategy::blend_p_legal` を呼ぶ**（`evaluate` と
//! 同じ関数）。別々に書くと「仮想更新が実再決定とずれた」のか「式が食い違って
//! いた」のかを分けられない。

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::check::CheckSolver;
use crate::check_economy::CheckMoveKind;
use crate::observation::ObservationLog;
use crate::protocol::PlayerView;
use crate::shogi::{Position, ShogiMove, parse_usi};
use crate::strategy::{
    self, CandidateScore, EvalParams, blend_p_legal, combine_score, prior_legal,
};

/// 王手中の1候補（**p と価格の両方**を付け替えて score を引き直せる最小の入力）。
///
/// `score = combine_score(gain, p_legal, cost) + foul_probe + adjust_det` は
/// `strategy.rs` の最終式そのもの。`adjust_det` は**タイブレーク乱数を除いた**
/// gain 外の補正（`CandidateScore::adjust − tiebreak`）で、乱数を残すと
/// 同点の順位に乱数の並び順が入る（issue #24 の教訓②）。
#[derive(Clone, Debug)]
pub struct PolicyMove {
    pub usi: String,
    pub mv: ShogiMove,
    pub kind: CheckMoveKind,
    pub gain: f64,
    /// 初回ランキングの p_legal（shadow update で置き換わる）
    pub p_legal: f64,
    pub foul_probe: f64,
    /// gain 外の補正から乱数を除いたもの
    pub adjust_det: f64,
    pub is_king: bool,
    /// `CheckSolver` 単体の解消確率（ソルバー貪欲 arm 用。方策以外では使わない）
    pub solver_p: f64,
    /// **真実で合法か**。裁定にだけ使い、方策へは絶対に渡さない
    pub truth_legal: bool,
}

impl PolicyMove {
    /// p と反則コストを与えたときの決定的スコア
    pub fn score(&self, p: f64, cost: f64) -> f64 {
        combine_score(self.gain, p, cost) + self.foul_probe + self.adjust_det
    }
}

/// `ranking` を [`PolicyMove`] へ落とす（真実の合法性ラベルつき）。
///
/// `solver` は手番開始時のもの。`kind` は **bot の意図**（`captures_checker`）で
/// 分ける規約を `check_economy::classify_move_kind` と共有する。
pub fn policy_moves(
    ranking: &[CandidateScore],
    view: &PlayerView,
    truth: &Position,
    solver: Option<&mut CheckSolver>,
    king: Option<crate::board::Coord>,
) -> Vec<PolicyMove> {
    let mut solver = solver;
    let mut out = vec![];
    for c in ranking {
        let Some(mv) = parse_usi(&c.usi) else { continue };
        let solver_p = solver.as_mut().map_or(0.5, |s| s.resolve_probability(&mv));
        let kind = crate::check_economy::classify_move_kind(&mv, view, solver.as_deref_mut());
        out.push(PolicyMove {
            usi: c.usi.clone(),
            kind,
            gain: c.gain,
            p_legal: c.p_legal,
            foul_probe: c.foul_probe,
            adjust_det: c.adjust - c.tiebreak,
            is_king: crate::check_economy::is_king_move(&c.usi, king),
            solver_p,
            truth_legal: truth.is_legal(&mv),
            mv,
        });
    }
    out
}

/// **p-only shadow update**（issue #31 P0-5 の仮想更新）。
///
/// 反則 `m` を観測したあとの p_legal を、
///
/// - `m` が合法だった粒子を落とす（合法性は物理制約なので厳密・taint にも効く）
/// - `CheckSolver` を「その手番の反則列」込みで作り直す（＝ `observe_foul`）
///
/// の2つだけで作る。**gain・リスク・`removal_term` は再計算しない**ので、
/// 実対局の再決定（若返り・粒子の再生成・2手読み）とは別物になる。
/// その差そのものが P0-5 の測定対象（近似通過条件）。
pub struct ShadowUpdater {
    /// 手番開始時の視界（反則数はステップごとに差し替える）
    view: PlayerView,
    log: ObservationLog,
    /// 評価に渡った厳密粒子（`ParticleSnapshot::strict`）
    strict: Vec<(Position, f64)>,
    /// 全滅時のフォールバック先（`ParticleSnapshot::taint`）
    taint: Vec<(Position, f64)>,
    opp_board_n: f64,
    params: EvalParams,
    eval_particles: usize,
    /// 手番開始時の bot の累計反則
    fouls_before: u32,
    opp_fouls: u32,
}

impl ShadowUpdater {
    pub fn new(
        view: &PlayerView,
        log: &ObservationLog,
        strict: &[(Position, f64)],
        taint: &[(Position, f64)],
        params: &EvalParams,
        eval_particles: usize,
    ) -> Self {
        ShadowUpdater {
            view: view.clone(),
            log: crate::scenario_core::clone_log(log),
            strict: strict.to_vec(),
            taint: taint.to_vec(),
            opp_board_n: opp_board_count(log),
            params: params.clone(),
            eval_particles,
            fouls_before: view.fouls.you,
            opp_fouls: view.fouls.opponent,
        }
    }

    /// `fouls` を観測したあとの各候補の p_legal。
    ///
    /// `fouls` は**その手番でこれまでに反則した手**（順番は問わない）。
    /// 空なら手番開始時の p を返す（初回ランキングの p と一致するはずで、
    /// その一致率が「式を正しく再現できているか」の健全性検査になる）。
    pub fn p_after(&self, moves: &[PolicyMove], fouls: &[ShogiMove]) -> Vec<f64> {
        // 反則は「その手が真実で非合法だった」ことの証拠なので、その手が
        // 合法な粒子は物理的に棄却される（taint 粒子も物理制約は守っている）
        let keep = |pool: &[(Position, f64)]| -> Vec<(Position, f64)> {
            pool.iter()
                .filter(|(p, _)| !fouls.iter().any(|m| p.is_legal(m)))
                .cloned()
                .collect()
        };
        let strict = keep(&self.strict);
        let taint = keep(&self.taint);
        // `evaluate` と同じプールの選び方（既定では厳密のみ）
        let use_taint =
            strict.is_empty() && !taint.is_empty() && strategy::eval_taint_fallback_enabled();
        let pool: &[(Position, f64)] = if use_taint { &taint } else { &strict };
        // ソルバーの投票は「厳密が全滅していれば taint」（`choose` と同じ規約）
        let votes_src: &[(Position, f64)] = if strict.is_empty() { &taint } else { &strict };
        let votes: Vec<(&Position, f64)> = votes_src.iter().map(|(p, w)| (p, *w)).collect();

        let mut view = self.view.clone();
        view.fouls.you = self.fouls_before + fouls.len() as u32;
        view.fouls.opponent = self.opp_fouls;
        let mut solver = CheckSolver::new(&view, &votes, fouls, &self.log);
        // 王手中の反則が積もるほど事前（CheckSolver）を信じる（`choose` と同じ）
        let mut params = self.params.clone();
        let f = strategy::check_foul_prior_factor(fouls.len());
        params.prior_weight *= f;
        params.prior_weight_degen *= f;

        let n: f64 = pool.iter().map(|(_, w)| w).sum();
        moves
            .iter()
            .map(|m| {
                let mut prior = prior_legal(&view, &m.mv, self.opp_board_n);
                prior *= match solver.as_mut() {
                    Some(s) => s.resolve_probability(&m.mv).clamp(0.02, 1.0),
                    None => strategy::in_check_prior(&view, &m.mv),
                };
                let legal: f64 = pool
                    .iter()
                    .filter(|(p, _)| p.is_legal(&m.mv))
                    .map(|(_, w)| w)
                    .sum();
                let mut p = blend_p_legal(
                    legal,
                    n,
                    prior,
                    use_taint,
                    self.eval_particles,
                    &params,
                );
                // 既知敵駒がカバーする玉の行き先の上限（`choose` と同じ min 専用）
                if let Some(cap) = solver
                    .as_ref()
                    .and_then(|s| s.known_covered_king_move_cap(&m.mv))
                {
                    p = p.min(cap);
                }
                p.clamp(0.0, 1.0)
            })
            .collect()
    }
}

/// 手番開始時（反則0）の決定を作り直した結果（P0-5 と P0-6 で共有する）。
pub struct EntrySetup {
    /// **prewarm 済みの戦略 instance**。実再決定はこれを `clone_boxed` して
    /// 反則の観測を食わせた**継続**として作る（別々に組み直すと壁時計
    /// デッドラインのぶん粒子集合が別物になる。PR #32 レビュー [P1]）
    pub strat: Box<dyn crate::strategy::Strategy>,
    pub view: PlayerView,
    pub log: ObservationLog,
    pub moves: Vec<PolicyMove>,
    /// 初回ランキングの p_legal（`moves` と同じ並び）
    pub p0: Vec<f64>,
    pub updater: ShadowUpdater,
}

/// 手番開始時の状態からランキング・粒子・候補を作る。
///
/// `ranking_and_particles` と同じことをするが、**prewarm 済みの instance を
/// 返す**ので実再決定（`UpdateRule::Real`）の継続に使える。
/// 定跡手・候補ゼロでランキングが取れなければ `None`。
pub fn entry_setup(
    entry: &crate::scenario_core::Replayed,
    truth: &Position,
    seed: u64,
    params: &EvalParams,
    eval_particles: usize,
) -> Option<EntrySetup> {
    use crate::scenario_core::{clone_log, make_view, prewarm_for_trial, side_idx};
    let side = entry.pos.turn();
    let king = entry.pos.king_square(side);
    let log = clone_log(&entry.logs[side_idx(side)]);
    let view = make_view(&entry.pos, side, &entry.fouls);
    let mut strat = crate::strategy::make_seeded("estimator", seed)?;
    strat.set_capture_particles(true);
    prewarm_for_trial(&mut *strat, entry);
    strat.choose(&view, &log, &HashSet::new())?;
    let ranking = strat.last_ranking()?.to_vec();
    let snapshot = strat.last_particles().cloned().unwrap_or_default();
    let mut solver = CheckSolver::new(&view, &[], &[], &log);
    let moves = policy_moves(&ranking, &view, truth, solver.as_mut(), king);
    if moves.is_empty() {
        return None;
    }
    let p0: Vec<f64> = moves.iter().map(|m| m.p_legal).collect();
    let updater = ShadowUpdater::new(
        &view,
        &log,
        &snapshot.strict,
        &snapshot.taint,
        params,
        eval_particles,
    );
    Some(EntrySetup { strat, view, log, moves, p0, updater })
}

/// 相手の盤上駒数の見積り（`prior_legal` の引数。`choose` と同じ式）
pub fn opp_board_count(log: &ObservationLog) -> f64 {
    let my_captures = log
        .events()
        .iter()
        .filter(|e| {
            matches!(
                e,
                crate::observation::Observation::MyMove {
                    captured: Some(_),
                    ..
                }
            )
        })
        .count();
    (20 - my_captures.min(19)) as f64
}

/// 王手中の方策（すべて「王手中の手番だけ」に効く）。
#[derive(Clone, Debug, PartialEq)]
pub enum Policy {
    /// 現行 `combine_score`（失敗枝は候補共通の定数 = 次善手固定の仮定）
    Current,
    /// α: 反則価格 ×k。`c = max(k × base, 床)`（**床は倍率化しない**）
    Alpha { k: f64 },
    /// β: 動的継続価値。`c = max(k × base, 床)` を先に決め、
    /// `c_min = 0.5c`、`c_eff(m) = max(c_min, c − λ·ΔV(m))`。
    /// 残り1回では β 全体を無効（失敗枝は次の選択でなく反則負け）
    Beta { k: f64, lambda: f64 },
    /// β-order: **現行がプローブを選ぶ決定に限り、プローブ同士だけ**を β で
    /// 並べ替える（反則を減らす目的に限定した版）
    BetaOrder { lambda: f64 },
    /// ソルバー貪欲（参考。gain を無視して解消確率の argmax）
    SolverGreedy,
}

impl Policy {
    pub fn tag(&self) -> String {
        match self {
            Policy::Current => "current".into(),
            Policy::Alpha { k } => format!("alpha@k{}", fmt_num(*k)),
            Policy::Beta { k, lambda } => {
                format!("beta@k{}l{}", fmt_num(*k), fmt_num(*lambda))
            }
            Policy::BetaOrder { lambda } => format!("beta_order@l{}", fmt_num(*lambda)),
            Policy::SolverGreedy => "solver_greedy".into(),
        }
    }

    /// β 系か（ΔV の計算が要るか）
    pub fn needs_delta_v(&self) -> bool {
        matches!(self, Policy::Beta { .. } | Policy::BetaOrder { .. })
    }

    /// `tag()` の逆。P0-6 が「主 arm を1本だけ固定する」ために使う
    /// （P0-4 / P0-5 を見てから水準を選ぶので、arm 名は文字列で渡す）
    pub fn parse(tag: &str) -> Option<Policy> {
        let num = |s: &str| s.parse::<f64>().ok().filter(|v| v.is_finite());
        match tag {
            "current" => Some(Policy::Current),
            "solver_greedy" => Some(Policy::SolverGreedy),
            t => {
                if let Some(k) = t.strip_prefix("alpha@k") {
                    return Some(Policy::Alpha { k: num(k)? });
                }
                if let Some(l) = t.strip_prefix("beta_order@l") {
                    return Some(Policy::BetaOrder { lambda: num(l)? });
                }
                if let Some(rest) = t.strip_prefix("beta@k") {
                    let (k, l) = rest.split_once('l')?;
                    return Some(Policy::Beta { k: num(k)?, lambda: num(l)? });
                }
                None
            }
        }
    }
}

pub fn fmt_num(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        format!("{}", v.round() as i64)
    } else {
        format!("{v}")
    }
}

/// **β が α と床を迂回しない**ための下限比（`c_min = BETA_C_MIN_RATIO × c`）。
/// β は最大でもその候補の実効コストを半減するだけ（issue #31 で事前登録）
pub const BETA_C_MIN_RATIO: f64 = 0.5;

/// その手番の価格（`base` / 床 / 実効コスト）
#[derive(Clone, Copy, Debug)]
pub struct Price {
    pub base: f64,
    pub floor: f64,
    pub current: f64,
    /// 残り反則（10 − 累計）
    pub remaining: u32,
}

pub fn price_at(params: &EvalParams, you: u32, opponent: u32) -> Price {
    Price {
        base: strategy::base_foul_cost_for(params, you, opponent),
        floor: strategy::foul_cost_floor_for(you),
        current: strategy::foul_cost_for(params, you, opponent),
        remaining: 10u32.saturating_sub(you),
    }
}

/// 1ステップぶんの選択（方策・p・価格・除外集合から首位を決める）。
///
/// 順位は**乱数を除いたスコア**で付け、同点は USI の辞書順で割る。
/// `delta_v` は β 系のときだけ Some（候補と同じ並び）。
pub fn pick(
    policy: &Policy,
    moves: &[PolicyMove],
    p: &[f64],
    price: &Price,
    excluded: &HashSet<String>,
    delta_v: Option<&[f64]>,
) -> Option<usize> {
    let live: Vec<usize> = (0..moves.len())
        .filter(|i| !excluded.contains(&moves[*i].usi))
        .collect();
    if live.is_empty() {
        return None;
    }
    if *policy == Policy::SolverGreedy {
        return live
            .into_iter()
            .max_by(|a, b| {
                moves[*a]
                    .solver_p
                    .total_cmp(&moves[*b].solver_p)
                    .then_with(|| moves[*b].usi.cmp(&moves[*a].usi))
            });
    }
    let costs = effective_costs(policy, moves, price, delta_v);
    let best = |set: &[usize]| -> Option<usize> {
        set.iter().copied().max_by(|a, b| {
            moves[*a]
                .score(p[*a], costs[*a])
                .total_cmp(&moves[*b].score(p[*b], costs[*b]))
                .then_with(|| moves[*b].usi.cmp(&moves[*a].usi))
        })
    };
    if let Policy::BetaOrder { .. } = policy {
        // **現行がプローブを選ぶ決定に限り、プローブ同士だけ**を並べ替える
        let cur_costs = effective_costs(&Policy::Current, moves, price, None);
        let cur = live.iter().copied().max_by(|a, b| {
            moves[*a]
                .score(p[*a], cur_costs[*a])
                .total_cmp(&moves[*b].score(p[*b], cur_costs[*b]))
                .then_with(|| moves[*b].usi.cmp(&moves[*a].usi))
        })?;
        if moves[cur].is_king {
            return Some(cur);
        }
        let probes: Vec<usize> = live.iter().copied().filter(|i| !moves[*i].is_king).collect();
        return best(&probes).or(Some(cur));
    }
    best(&live)
}

/// 候補ごとの実効反則コスト
pub fn effective_costs(
    policy: &Policy,
    moves: &[PolicyMove],
    price: &Price,
    delta_v: Option<&[f64]>,
) -> Vec<f64> {
    match policy {
        Policy::Current | Policy::SolverGreedy => vec![price.current; moves.len()],
        Policy::Alpha { k } => vec![(k * price.base).max(price.floor); moves.len()],
        Policy::Beta { k, lambda } => {
            let c = (k * price.base).max(price.floor);
            beta_costs(c, *lambda, moves.len(), price.remaining, delta_v)
        }
        Policy::BetaOrder { lambda } => {
            // β-order は価格の水準を動かさない（並べ替えだけ）ので c は現行
            beta_costs(price.current, *lambda, moves.len(), price.remaining, delta_v)
        }
    }
}

fn beta_costs(
    c: f64,
    lambda: f64,
    n: usize,
    remaining: u32,
    delta_v: Option<&[f64]>,
) -> Vec<f64> {
    // 残り1回では β を無効にする（失敗枝は「次の選択」ではなく反則負け）
    let Some(dv) = delta_v.filter(|_| remaining > 1) else {
        return vec![c; n];
    };
    let c_min = BETA_C_MIN_RATIO * c;
    (0..n)
        .map(|i| (c - lambda * dv.get(i).copied().unwrap_or(0.0)).max(c_min).min(c.max(c_min)))
        .collect()
}

/// **ΔV(m) = Vpost(m) − Vpre(m)**（候補固有の動的継続価値）。
///
/// `Vpre(m)` は「m を除いた次候補の価値」、`Vpost(m)` は「m の反則を観測した
/// 後の次候補の価値」（m を除外・残り反則を1減らした価格で）。
///
/// **水準の差は引く**: 反則を1つ払えば価格は全候補で上がり p も全体に動くので、
/// 生の `Vpost − Vpre` には m に依らない定数が乗る。issue の設計
/// 「同じ参照候補を引いた候補間 advantage」を、参照＝**候補平均**で実現する
/// （どの m を選んでも同じ量だけ引くので候補間の差は保たれ、β が価格の
/// 水準を動かさない = α と役割が分離する）。
///
/// 計算量を抑えるため、**首位から `c` 以内**の候補だけ実際に仮想更新する
/// （β が動かせる score は高々 `(1−p)·(c − c_min) ≤ 0.5c` なので、それ以上
/// 離れた候補は β では首位になれない。落とした候補の ΔV は 0）。
pub fn delta_v(
    updater: &ShadowUpdater,
    moves: &[PolicyMove],
    p: &[f64],
    c_pre: f64,
    c_post: f64,
    fouls_so_far: &[ShogiMove],
    excluded: &HashSet<String>,
) -> Vec<f64> {
    let live: Vec<usize> = (0..moves.len())
        .filter(|i| !excluded.contains(&moves[*i].usi))
        .collect();
    let mut out = vec![0.0; moves.len()];
    if live.len() < 2 {
        return out;
    }
    // **その方策自身の価格で測る**（α×β では c = max(k×base, 床) が現行価格と
    // 違うので、現行価格で順位を付けると別の世界の ΔV になる）
    let score_pre = |i: usize| moves[i].score(p[i], c_pre);
    let leader = live
        .iter()
        .copied()
        .map(score_pre)
        .fold(f64::NEG_INFINITY, f64::max);
    // 首位から (c − c_min) 以内の候補だけが β で動きうる（doc の上界。
    // 余裕を見て c まで広げる）
    let targets: Vec<usize> = live
        .iter()
        .copied()
        .filter(|i| score_pre(*i) >= leader - c_pre)
        .collect();
    if targets.len() < 2 {
        return out;
    }
    let mut raw: Vec<(usize, f64)> = vec![];
    for &m in &targets {
        // Vpre(m): m を除いた次候補の価値（現行の p と価格）
        let vpre = live
            .iter()
            .copied()
            .filter(|i| *i != m)
            .map(score_pre)
            .fold(f64::NEG_INFINITY, f64::max);
        // Vpost(m): m の反則を観測した後（m を除外・残り反則が1減った価格）
        let mut fouls = fouls_so_far.to_vec();
        fouls.push(moves[m].mv);
        let p_post = updater.p_after(moves, &fouls);
        let vpost = live
            .iter()
            .copied()
            .filter(|i| *i != m)
            .map(|i| moves[i].score(p_post[i], c_post))
            .fold(f64::NEG_INFINITY, f64::max);
        if vpre.is_finite() && vpost.is_finite() {
            raw.push((m, vpost - vpre));
        }
    }
    if raw.is_empty() {
        return out;
    }
    let mean = raw.iter().map(|(_, v)| v).sum::<f64>() / raw.len() as f64;
    for (i, v) in raw {
        out[i] = v - mean;
    }
    out
}

/// 方策シミュレーションの結果（1手番ぶん）
#[derive(Clone, Debug)]
pub struct SimOutcome {
    /// 受理された手（反則負け・候補切れなら None）
    pub accepted: Option<String>,
    pub accepted_kind: Option<CheckMoveKind>,
    /// この手番で積んだ反則の数
    pub fouls: u32,
    /// 反則列（順番どおり）
    pub sequence: Vec<String>,
    /// 累計が上限（10）に達した = その場で反則負け
    pub foul_limit: bool,
    /// 反則の観測ごとに p を更新した回数（＝仮想更新の呼び出し回数）
    pub updates: u32,
}

/// 反則後の更新規則（issue #31 P0-5 の「4本の対照」のうち 2 / 3 / 4）。
pub enum UpdateRule<'a> {
    /// 更新しない（現行 `combine_score` が暗黙に置く「次善手固定」の仮定そのもの）
    Static,
    /// **p-only shadow update**（snapshot ベースの仮想更新。p だけを作り直す）
    Shadow(&'a ShadowUpdater),
    /// **実再決定**: 反則の観測を注入して `Estimator::update` とランキングを
    /// 走らせる（正解基準）。gain・リスク・`removal_term` まで作り直されるので
    /// **候補リストごと差し替える**。呼び出しごとに思考予算をまるごと使う
    Real(&'a mut dyn FnMut(&[ShogiMove]) -> Option<Vec<PolicyMove>>),
}

/// 方策を反則するたび更新しながら**真実で裁定**する。
///
/// 非合法な手を選んだら実対局と同じく反則を1つ積んで（手番は変わらない）
/// 次の選択へ進む。累計が `MAX_FOULS` に達したらその場で反則負け。
pub fn simulate(
    policy: &Policy,
    moves: &[PolicyMove],
    p0: &[f64],
    params: &EvalParams,
    fouls_before: u32,
    opp_fouls: u32,
    mut rule: UpdateRule<'_>,
) -> SimOutcome {
    let mut excluded: HashSet<String> = HashSet::new();
    let mut sequence = vec![];
    let mut fouls_here = 0u32;
    let mut updates = 0u32;
    // 実再決定は候補リストごと差し替わるので、ローカルに持つ
    let mut moves: Vec<PolicyMove> = moves.to_vec();
    let mut p: Vec<f64> = p0.to_vec();
    let mut fouls_mv: Vec<ShogiMove> = vec![];
    // 候補を全部試し切っても終わらない可能性はないが、上限で守る
    for _ in 0..=(moves.len() + crate::selfplay::MAX_FOULS as usize) {
        let you = fouls_before + fouls_here;
        if you >= crate::selfplay::MAX_FOULS {
            return SimOutcome {
                accepted: None,
                accepted_kind: None,
                fouls: fouls_here,
                sequence,
                foul_limit: true,
                updates,
            };
        }
        let price = price_at(params, you, opp_fouls);
        let price_after = price_at(params, you + 1, opp_fouls);
        let dv = if policy.needs_delta_v() {
            // ΔV は**その方策自身の価格**で測る（β-order は現行価格、
            // β は `max(k × base, 床)`）
            let c_of = |pr: &Price| match policy {
                Policy::Beta { k, .. } => (k * pr.base).max(pr.floor),
                _ => pr.current,
            };
            match &rule {
                UpdateRule::Shadow(u) => Some(delta_v(
                    u,
                    &moves,
                    &p,
                    c_of(&price),
                    c_of(&price_after),
                    &fouls_mv,
                    &excluded,
                )),
                // 仮想更新が使えない対照（Static / Real）では β は ΔV を持てない
                _ => None,
            }
        } else {
            None
        };
        let Some(i) = pick(policy, &moves, &p, &price, &excluded, dv.as_deref()) else {
            return SimOutcome {
                accepted: None,
                accepted_kind: None,
                fouls: fouls_here,
                sequence,
                foul_limit: false,
                updates,
            };
        };
        sequence.push(moves[i].usi.clone());
        if moves[i].truth_legal {
            return SimOutcome {
                accepted: Some(moves[i].usi.clone()),
                accepted_kind: Some(moves[i].kind),
                fouls: fouls_here,
                sequence,
                foul_limit: false,
                updates,
            };
        }
        excluded.insert(moves[i].usi.clone());
        fouls_mv.push(moves[i].mv);
        fouls_here += 1;
        match &mut rule {
            UpdateRule::Static => {}
            UpdateRule::Shadow(u) => {
                p = u.p_after(&moves, &fouls_mv);
                updates += 1;
            }
            UpdateRule::Real(f) => {
                // 実再決定は gain も adjust も作り直されるので候補リストごと入れ替える
                if let Some(next) = f(&fouls_mv) {
                    p = next.iter().map(|m| m.p_legal).collect();
                    moves = next;
                }
                updates += 1;
            }
        }
    }
    SimOutcome {
        accepted: None,
        accepted_kind: None,
        fouls: fouls_here,
        sequence,
        foul_limit: false,
        updates,
    }
}

/// **受理直後の真実ベース指標**（受理手の現行 gain は自己正当化するので補助）。
#[derive(Clone, Copy, Debug, Default)]
pub struct TruthAfter {
    /// bot 側から見た材料差（自分の駒価値合計 − 相手の駒価値合計）
    pub material: f64,
    /// 受理直後に相手が**一手詰め**を持つか
    pub mated_in_1: bool,
    /// 受理直後に相手が王手を掛けられるか（次の被王手）
    pub next_check: bool,
    /// そもそも受理できたか（false = 反則負け or 候補切れ）
    pub accepted: bool,
}

pub fn truth_after(truth: &Position, bot: crate::protocol::Color, usi: Option<&str>) -> TruthAfter {
    let Some(mv) = usi.and_then(parse_usi) else {
        return TruthAfter::default();
    };
    if !truth.is_legal(&mv) {
        return TruthAfter::default();
    }
    let mut next = truth.clone();
    next.play_unchecked(&mv);
    let mated = !crate::mate::mate_moves_in_1_fast(&next).is_empty();
    let next_check = next.legal_moves().iter().any(|m| {
        let mut probe = next.clone();
        probe.play_unchecked(m);
        probe.in_check(bot)
    });
    TruthAfter {
        material: material_diff(&next, bot),
        mated_in_1: mated,
        next_check,
        accepted: true,
    }
}

fn material_diff(pos: &Position, me: crate::protocol::Color) -> f64 {
    let mut sum = 0.0;
    for (_, p) in pos.pieces() {
        let v = crate::strategy::exchange_value(p.role);
        sum += if p.color == me { v } else { -v };
    }
    for color in [crate::protocol::Color::Sente, crate::protocol::Color::Gote] {
        for (role, n) in pos.hand_map(color) {
            let v = crate::strategy::exchange_value(role) * f64::from(n);
            sum += if color == me { v } else { -v };
        }
    }
    sum
}

/// 候補分布上の較正の集計（**現行方策が選んだ手ではなく候補全体**）。
///
/// P0-2 の較正は「選ばれた手」のものなので、順位を変えた後の候補分布へは
/// 外挿できない。ここは初回ランキングの全候補を真実でラベル付けして、
/// 手種ごとに `(平均予測 − 合法率)` を出すための和を貯める。
#[derive(Clone, Debug, Default)]
pub struct CalibrationSums {
    /// 手種 → (予測の和, 合法だった数, 候補数)
    pub by_kind: BTreeMap<String, (f64, f64, f64)>,
}

impl CalibrationSums {
    pub fn add(&mut self, moves: &[PolicyMove], p: &[f64]) {
        for (m, p) in moves.iter().zip(p) {
            let e = self.by_kind.entry(m.kind.tag().to_string()).or_default();
            e.0 += p;
            e.1 += f64::from(u8::from(m.truth_legal));
            e.2 += 1.0;
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        let map: HashMap<String, serde_json::Value> = self
            .by_kind
            .iter()
            .map(|(k, (sp, sl, n))| {
                (k.clone(), serde_json::json!({ "p_sum": sp, "legal": sl, "n": n }))
            })
            .collect();
        serde_json::json!(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{Color, Role};

    fn mv(usi: &str) -> ShogiMove {
        parse_usi(usi).unwrap()
    }

    fn pm(usi: &str, gain: f64, p: f64, king: bool, legal: bool) -> PolicyMove {
        PolicyMove {
            usi: usi.into(),
            mv: mv(usi),
            kind: if king {
                CheckMoveKind::King
            } else {
                CheckMoveKind::CheckerCapture
            },
            gain,
            p_legal: p,
            foul_probe: 0.0,
            adjust_det: 0.0,
            is_king: king,
            solver_p: p,
            truth_legal: legal,
        }
    }

    fn price(current: f64) -> Price {
        Price { base: current, floor: 0.0, current, remaining: 9 }
    }

    #[test]
    fn 価格を上げるとプローブより玉の手が浮く() {
        // プローブ: gain 8.0 / p 0.4 → score 3.2 − 0.6c、
        // 玉の手: gain 2.0 / p 0.9 → 1.8 − 0.1c（交点は c = 2.8）
        let moves = vec![pm("5f5e", 8.0, 0.4, false, false), pm("5i4h", 2.0, 0.9, true, true)];
        let p: Vec<f64> = moves.iter().map(|m| m.p_legal).collect();
        let ex = HashSet::new();
        let pr = price(1.0);
        assert_eq!(
            pick(&Policy::Current, &moves, &p, &pr, &ex, None),
            Some(0),
            "現行価格ではプローブが首位"
        );
        // α で価格を 4 倍にすると玉の手が首位（交点は 3 倍）
        let a = Policy::Alpha { k: 4.0 };
        let costs = effective_costs(&a, &moves, &pr, None);
        assert_eq!(costs, vec![4.0, 4.0], "床は倍率化しない");
        assert_eq!(pick(&a, &moves, &p, &pr, &ex, None), Some(1));
    }

    #[test]
    fn 床は倍率より優先される() {
        let moves = vec![pm("5f5e", 3.0, 0.4, false, false), pm("5i4h", 2.0, 0.9, true, true)];
        let pr = Price { base: 1.0, floor: 60.0, current: 60.0, remaining: 1 };
        // k=1.5 でも床 60 のほうが高いので実効コストは床
        let costs = effective_costs(&Policy::Alpha { k: 1.5 }, &moves, &pr, None);
        assert_eq!(costs, vec![60.0, 60.0]);
    }

    #[test]
    fn betaはcminを下回らず残り1回では無効() {
        let moves = vec![pm("5f5e", 3.0, 0.4, false, false), pm("5i4h", 2.0, 0.9, true, true)];
        let pr = Price { base: 2.0, floor: 0.0, current: 2.0, remaining: 5 };
        // ΔV が大きくても c_min = 0.5c を下回らない（β は α と床を迂回しない）
        let dv = vec![100.0, 0.0];
        let costs = effective_costs(&Policy::Beta { k: 1.0, lambda: 1.0 }, &moves, &pr, Some(&dv));
        assert!((costs[0] - 1.0).abs() < 1e-9, "c_min = 0.5 × 2.0: {costs:?}");
        assert!((costs[1] - 2.0).abs() < 1e-9);
        // ΔV が負なら価格は上がるが、c を超えない（β は割引だけ）
        let dv = vec![-100.0, 0.0];
        let costs = effective_costs(&Policy::Beta { k: 1.0, lambda: 1.0 }, &moves, &pr, Some(&dv));
        assert!((costs[0] - 2.0).abs() < 1e-9, "上側は c で頭打ち: {costs:?}");
        // 残り1回では β 全体が無効
        let last = Price { base: 2.0, floor: 0.0, current: 2.0, remaining: 1 };
        let costs =
            effective_costs(&Policy::Beta { k: 1.0, lambda: 1.0 }, &moves, &last, Some(&[9.0, 0.0]));
        assert_eq!(costs, vec![2.0, 2.0]);
    }

    /// `delta_v` が「首位から c 以内」で枝刈りしてよい根拠。
    ///
    /// β が動かせる score は高々 `(1−p)·(c − c_min) ≤ c − c_min = 0.5c` なので、
    /// 首位から c 以上離れた候補は β では首位になれない（枝刈りは近似ではなく上界）。
    #[test]
    fn betaが動かせるスコアはc_minまでの差で頭打ち() {
        let pr = Price { base: 2.0, floor: 0.0, current: 2.0, remaining: 5 };
        let moves = vec![pm("5f5e", 3.0, 0.4, false, false), pm("5i4h", 2.0, 0.9, true, true)];
        for dv in [-100.0, -1.0, 0.0, 1.0, 100.0] {
            let costs = effective_costs(
                &Policy::Beta { k: 1.0, lambda: 1.0 },
                &moves,
                &pr,
                Some(&[dv, dv]),
            );
            for (i, m) in moves.iter().enumerate() {
                let shift = (m.score(m.p_legal, costs[i]) - m.score(m.p_legal, pr.current)).abs();
                let bound = (1.0 - BETA_C_MIN_RATIO) * pr.current;
                assert!(shift <= bound + 1e-9, "ΔV={dv} で {shift} > {bound}");
            }
        }
    }

    #[test]
    fn beta_orderは玉の手が首位のときは何もしない() {
        // 現行の首位が玉の手 → β-order は並べ替えない
        let moves = vec![pm("5f5e", 1.0, 0.4, false, false), pm("5i4h", 5.0, 0.9, true, true)];
        let p: Vec<f64> = moves.iter().map(|m| m.p_legal).collect();
        let pr = price(1.0);
        let ex = HashSet::new();
        assert_eq!(pick(&Policy::Current, &moves, &p, &pr, &ex, None), Some(1));
        assert_eq!(
            pick(&Policy::BetaOrder { lambda: 1.0 }, &moves, &p, &pr, &ex, Some(&[9.0, 0.0])),
            Some(1),
            "玉の手が首位ならプローブへは触らない"
        );
    }

    #[test]
    fn simulateは非合法な手で反則を積んで次へ進む() {
        // 首位が真実で非合法 → 反則1つ積んで次点が受理される
        let moves = vec![pm("5f5e", 8.0, 0.4, false, false), pm("5i4h", 2.0, 0.9, true, true)];
        let p: Vec<f64> = moves.iter().map(|m| m.p_legal).collect();
        let out = simulate(
            &Policy::Current,
            &moves,
            &p,
            &EvalParams::default(),
            0,
            0,
            UpdateRule::Static,
        );
        assert_eq!(out.fouls, 1);
        assert_eq!(out.accepted.as_deref(), Some("5i4h"));
        assert!(!out.foul_limit);
        assert_eq!(out.sequence, vec!["5f5e", "5i4h"]);
    }

    #[test]
    fn simulateは反則上限で打ち切る() {
        // 候補が全部非合法 & 開始時点で9反則 → 1回積んで反則負け
        let moves = vec![pm("5f5e", 8.0, 0.4, false, false), pm("5i4h", 2.0, 0.9, true, false)];
        let p: Vec<f64> = moves.iter().map(|m| m.p_legal).collect();
        let out = simulate(
            &Policy::Current,
            &moves,
            &p,
            &EvalParams::default(),
            9,
            0,
            UpdateRule::Static,
        );
        assert!(out.foul_limit, "{out:?}");
        assert_eq!(out.fouls, 1);
        assert_eq!(out.accepted, None);
    }

    #[test]
    fn 同点はusiの辞書順で割る() {
        // 完全同点の2候補。安定ソートに任せず辞書順で決める（issue #24 の教訓②）
        let moves = vec![pm("5i4h", 2.0, 0.5, true, true), pm("5i6h", 2.0, 0.5, true, true)];
        let p = vec![0.5, 0.5];
        let ex = HashSet::new();
        assert_eq!(pick(&Policy::Current, &moves, &p, &price(1.0), &ex, None), Some(0));
    }

    #[test]
    fn arm名は往復する() {
        // P0-6 は主 arm を文字列で受け取るので、tag() と parse() は往復すること
        for p in [
            Policy::Current,
            Policy::SolverGreedy,
            Policy::Alpha { k: 1.5 },
            Policy::Alpha { k: 3.0 },
            Policy::Beta { k: 1.0, lambda: 0.5 },
            Policy::BetaOrder { lambda: 1.0 },
        ] {
            assert_eq!(Policy::parse(&p.tag()), Some(p.clone()), "{}", p.tag());
        }
        for bad in ["", "alpha", "alpha@k", "alpha@kx", "beta@k1", "beta@k1lx", "nope"] {
            assert_eq!(Policy::parse(bad), None, "{bad}");
        }
    }

    #[test]
    fn 材料差は初期局面で0() {
        // 両者の盤上駒＋持ち駒を交換価値で数える。初期局面は対称なので 0
        let pos = Position::initial();
        assert!(material_diff(&pos, Color::Sente).abs() < 1e-9);
        // 先手の歩を1枚落とすと、その交換価値ぶんだけ負になる
        let mut down = pos.clone();
        down.set(crate::board::parse_usi_square("7g").unwrap(), None);
        let d = material_diff(&down, Color::Sente);
        assert!(d < 0.0 && d > -3.0, "歩1枚ぶん: {d}");
        assert!((d + material_diff(&down, Color::Gote)).abs() < 1e-9, "符号が対称");
    }

    #[test]
    fn 真実指標は受理できなかったら空になる() {
        let mut pos = Position::empty(Color::Sente);
        pos.set(crate::board::parse_usi_square("5i").unwrap(), Some(crate::shogi::Piece {
            color: Color::Sente,
            role: Role::King,
        }));
        pos.set(crate::board::parse_usi_square("8a").unwrap(), Some(crate::shogi::Piece {
            color: Color::Gote,
            role: Role::King,
        }));
        let none = truth_after(&pos, Color::Sente, None);
        assert!(!none.accepted);
        let ok = truth_after(&pos, Color::Sente, Some("5i5h"));
        assert!(ok.accepted);
    }
}
