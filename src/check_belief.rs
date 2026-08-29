//! **王手駒仮説の希釈の共有定義**（issue #36。runtime には何も入らない）。
//!
//! #31 P0-3 で見えた破滅の署名は「知らない」ではなく**希釈**だった:
//! 8反則以上を積んだ6手番のすべてで真の王手駒はマス・駒種とも一致で仮説集合に
//! **載っていた**のに、一様に近い 30 本前後の仮説に埋もれて正しい捕獲の
//! `p_legal` が薄まる（真の王手駒の重みシェア 0.035〜0.037）。
//!
//! ここに置くのは3つ:
//!
//! 1. [`Belief`] … 仮説重みの介入 arm（オラクル・誤誘導・直前手の演繹）と
//!    そのタグ。`check.rs` の**診断専用フック**へ落とす
//! 2. [`decision_points`] … 母集団の取り出し。**bot の全王手中手番**で、
//!    反則0も、反則だけ積んで受理手なしで終局した**終端手番**も含む
//!    （`for_each_decision_full` は受理手を単位に回すので終端を返さない。
//!    #34 の `check_prep::decision_snapshots` と同じ穴）
//! 3. [`Focus`] … 「真の王手駒を玉以外で取る手」と「誤仮説マスへの捕獲の最大値」の
//!    対。P0-1 はこの2つの伝達（ソルバー q → 事前 → ブレンド → cap → score → rank）を
//!    並べる
//!
//! **仮説シェアは選択への伝達量ではない**（シェア 0.30 でも順位が変わらないことも、
//! 0.05 でも粒子ブレンドに消されることもある）ので、P0-1 に中止の門は置かない。
//! 因果はオラクル arm（P0-2）でしか言えない。

use std::collections::HashMap;

use crate::board::Coord;
use crate::check::HypothesisDiag;
use crate::check_economy::true_checkers;
use crate::protocol::{Color, Role};
use crate::shogi::{Position, ShogiMove};

/// `k = ∞`（真仮説以外を潰す）の内部表現。`f64::INFINITY` を掛けると
/// 正規化が NaN になるので有限の巨大値にする
pub const ORACLE_INF: f64 = 1e9;

/// 仮説重みへの介入 arm（issue #36 P0-2）。**runtime には入らない**。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Belief {
    /// 現行（介入なし）。`Oracle { k: 1.0 }` は**これと bit-exact に一致する**
    /// ことを恒等対照として必須にする
    Current,
    /// 単王手の真仮説（マス・駒種とも一致）× k。
    /// **上限とは呼ばない**: 候補生成に真の捕獲手が無ければ届かないし、
    /// p-only で当てた場合は `removal_term` を初回の値に固定するので
    /// 歪みの向きが一意でない
    Oracle { k: f64 },
    /// **adversarial stress test**: 真でない仮説のうち現行重みが最大のもの × k。
    /// 学習事前の誤りの分布とは一致しないので、許容誤り率への換算はしない
    Misdirected { k: f64 },
    /// H1 ① の演繹（直前の相手手の1手で説明できない仮説の除去。学習なし）
    DeduceLastMove,
}

impl Belief {
    pub fn tag(&self) -> String {
        match self {
            Belief::Current => "current".into(),
            Belief::Oracle { k } => format!("oracle@k{}", fmt_k(*k)),
            Belief::Misdirected { k } => format!("oracle_misdirected@k{}", fmt_k(*k)),
            Belief::DeduceLastMove => "deduce_last_move".into(),
        }
    }

    pub fn parse(tag: &str) -> Option<Belief> {
        let num = |s: &str| -> Option<f64> {
            if s == "inf" {
                return Some(ORACLE_INF);
            }
            s.parse::<f64>().ok().filter(|v| v.is_finite() && *v >= 0.0)
        };
        match tag {
            "current" => Some(Belief::Current),
            "deduce_last_move" => Some(Belief::DeduceLastMove),
            t => {
                if let Some(k) = t.strip_prefix("oracle_misdirected@k") {
                    return Some(Belief::Misdirected { k: num(k)? });
                }
                if let Some(k) = t.strip_prefix("oracle@k") {
                    return Some(Belief::Oracle { k: num(k)? });
                }
                None
            }
        }
    }

    /// 真実を要る arm か（`check_continue` が「真実を渡し忘れていないか」を検査する）
    pub fn needs_truth(&self) -> bool {
        matches!(self, Belief::Oracle { .. } | Belief::Misdirected { .. })
    }

    /// この arm の診断フック。`hyps` は**現行の**仮説（誤誘導 arm の
    /// 「最大重みの誤仮説」を決めるのに要る。順序は決定論的）
    pub fn diag(
        &self,
        truth: &Position,
        bot: Color,
        hyps: &[(Coord, Role, f64)],
    ) -> HypothesisDiag {
        let mut factors: HashMap<(Coord, Role), f64> = HashMap::new();
        match self {
            Belief::Current => {}
            Belief::DeduceLastMove => {
                return HypothesisDiag { factors, deduce_last_move: true };
            }
            Belief::Oracle { k } => {
                let checkers = true_checkers(truth, bot);
                // **単王手のみ主 estimand**（両王手は 2 仮説をともに ×k しても
                // 重みが半々になるだけで、一方しか解消しない非合法手に ≈0.5 が付く）
                if checkers.len() == 1 {
                    for c in checkers {
                        factors.insert(c, *k);
                    }
                    // k=∞ は「真仮説以外を潰す」= 他を ×0（全滅時は
                    // `apply_diag` が元へ戻す）
                    if *k >= ORACLE_INF {
                        for (s, r, _) in hyps {
                            if !factors.contains_key(&(*s, *r)) {
                                factors.insert((*s, *r), 0.0);
                            }
                        }
                    }
                }
            }
            Belief::Misdirected { k } => {
                let checkers = true_checkers(truth, bot);
                let worst = hyps
                    .iter()
                    .filter(|(s, r, _)| !checkers.iter().any(|(cs, cr)| cs == s && cr == r))
                    // 同点は (マス, 駒種) の辞書順で決める（乱数を入れない）
                    .max_by(|a, b| {
                        a.2.total_cmp(&b.2)
                            .then_with(|| (b.0.file, b.0.rank).cmp(&(a.0.file, a.0.rank)))
                    });
                if let Some((s, r, _)) = worst {
                    factors.insert((*s, *r), *k);
                    if *k >= ORACLE_INF {
                        for (hs, hr, _) in hyps {
                            if (*hs, *hr) != (*s, *r) {
                                factors.insert((*hs, *hr), 0.0);
                            }
                        }
                    }
                }
            }
        }
        HypothesisDiag { factors, deduce_last_move: false }
    }

    /// この arm の診断フックを設置して `f` を走らせる。
    /// `Belief::Current` は**フックを設置しない**（恒等対照が bit-exact に
    /// なるように、no-op のスコープすら張らない）
    pub fn scoped<R>(
        &self,
        truth: &Position,
        bot: Color,
        hyps: &[(Coord, Role, f64)],
        f: impl FnOnce() -> R,
    ) -> R {
        let d = self.diag(truth, bot, hyps);
        if d.is_noop() {
            return f();
        }
        crate::check::scoped_hypothesis_diag(&d, f)
    }
}

fn fmt_k(k: f64) -> String {
    if k >= ORACLE_INF {
        "inf".into()
    } else {
        crate::check_policy::fmt_num(k)
    }
}

/// 1 arm の指定（`[belief|policy][@shadow|@real]`）。
///
/// **タグの規約は `bin/check_policy` と `bin/check_continue` で共通**にする
/// （同じ arm 名が違う配管を指していると、P0-2 の削減量と P0-2b の勝率差が
/// 別物の測定になる）。
///
/// - `current` … 現行（`Belief::Current` + `Policy::Current`）
/// - `alpha@k2` … issue #31 の価格 arm（belief は現行）
/// - `oracle@k2@shadow` … issue の **`oracle_p_only@k2`**。p だけを付け替え、
///   gain・`removal_term` は初回ランキングの値に固定する
/// - `oracle@kinf@real` … issue の **`oracle_full_score@real`**（P0-2 の主 arm）。
///   初回ランキングもオラクルの下で作り直し、反則ごとに実再決定する
/// - `oracle_misdirected@k4@shadow` … adversarial stress test
/// - `deduce_last_move@shadow` … H1 ① の演繹（学習なし）
#[derive(Clone, Debug, PartialEq)]
pub struct ArmSpec {
    pub tag: String,
    pub belief: Belief,
    pub policy: crate::check_policy::Policy,
    /// 反則ごとに**実再決定**するか（false = p-only shadow update）
    pub real: bool,
}

impl ArmSpec {
    pub fn parse(tag: &str) -> Option<ArmSpec> {
        use crate::check_policy::Policy;
        let (core, real) = match tag.strip_suffix("@real") {
            Some(c) => (c, true),
            None => (tag.strip_suffix("@shadow").unwrap_or(tag), false),
        };
        let (belief, policy) = match Policy::parse(core) {
            Some(p) => (Belief::Current, p),
            None => (Belief::parse(core)?, Policy::Current),
        };
        Some(ArmSpec { tag: tag.to_string(), belief, policy, real })
    }
}

/// 1 arm ぶんの結果（`run_arm`）。
pub struct ArmRun {
    pub out: crate::check_policy::SimOutcome,
    /// その arm が手番開始時に見た p_legal（恒等対照の検査に使う）
    pub p_entry: Vec<f64>,
}

/// 1 arm を回す（**P0-2 と P0-2b で同じ配管**）。
///
/// `entry_hyps` は手番開始時の**ソルバー単体**の仮説（誤誘導 arm の対象を
/// 決定論的に選ぶため）。真実は裁定と arm の定義にだけ使い、方策へは渡さない。
pub fn run_arm(
    setup: &crate::check_policy::EntrySetup,
    entry: &crate::scenario_core::Replayed,
    truth: &Position,
    bot: Color,
    spec: &ArmSpec,
    entry_hyps: &[(Coord, Role, f64)],
    params: &crate::strategy::EvalParams,
) -> Option<ArmRun> {
    use std::collections::HashSet;

    use crate::check::CheckSolver;
    use crate::check_policy::{UpdateRule, policy_moves, simulate};
    use crate::observation::Observation;
    use crate::scenario_core::{clone_log, make_view, side_idx};

    let side = entry.pos.turn();
    let king = entry.pos.king_square(side);
    let fouls_before = entry.fouls[side_idx(side)];
    let opp_fouls = entry.fouls[side_idx(side.other())];
    let b = &spec.belief;
    if !spec.real {
        // p-only: gain・`removal_term` は初回ランキングの値のまま
        let p = b.scoped(truth, bot, entry_hyps, || {
            setup.updater.p_after(&setup.moves, &[])
        });
        let out = b.scoped(truth, bot, entry_hyps, || {
            simulate(
                &spec.policy,
                &setup.moves,
                &p,
                params,
                fouls_before,
                opp_fouls,
                UpdateRule::Shadow(&setup.updater),
            )
        });
        return Some(ArmRun { out, p_entry: p });
    }
    // 実再決定: 初回ランキングも arm の信念の下で作り直す
    let entry_view = make_view(&entry.pos, side, &entry.fouls);
    let mut b0 = setup.strat.clone_boxed()?;
    let moves = b.scoped(truth, bot, entry_hyps, || {
        b0.choose(&entry_view, &setup.log, &HashSet::new())?;
        let r = b0.last_ranking()?.to_vec();
        let mut s = CheckSolver::new(&entry_view, &[], &[], &setup.log);
        Some(policy_moves(&r, &entry_view, truth, s.as_mut(), king))
    })?;
    if moves.is_empty() {
        return None;
    }
    let p0: Vec<f64> = moves.iter().map(|m| m.p_legal).collect();
    let mut real = |fouls: &[ShogiMove]| -> Option<Vec<crate::check_policy::PolicyMove>> {
        let mut c = setup.strat.clone_boxed()?;
        let mut post_log = clone_log(&setup.log);
        let mut post_fouls = entry.fouls;
        for m in fouls {
            post_fouls[side_idx(side)] += 1;
            post_log.record(Observation::MyFoul {
                move_number: entry.pos.move_number(),
                usi: m.to_usi(),
            });
        }
        let post_view = make_view(&entry.pos, side, &post_fouls);
        let tried: HashSet<String> = fouls.iter().map(ShogiMove::to_usi).collect();
        c.choose(&post_view, &post_log, &tried)?;
        let r = c.last_ranking()?.to_vec();
        let mut s = CheckSolver::new(&post_view, &[], fouls, &post_log);
        Some(policy_moves(&r, &post_view, truth, s.as_mut(), king))
    };
    let out = b.scoped(truth, bot, entry_hyps, || {
        simulate(
            &spec.policy,
            &moves,
            &p0,
            params,
            fouls_before,
            opp_fouls,
            UpdateRule::Real(&mut real),
        )
    });
    Some(ArmRun { out, p_entry: p0 })
}

/// 手番開始時の**ソルバー単体**の仮説（`run_arm` の `entry_hyps`）
pub fn entry_hypotheses(setup: &crate::check_policy::EntrySetup) -> Vec<(Coord, Role, f64)> {
    crate::check::CheckSolver::new(&setup.view, &[], &[], &setup.log)
        .map(|s| s.hypotheses_debug())
        .unwrap_or_default()
}

/// 注目する2手（P0-1 の伝達の分解の単位）。
///
/// - `true_capture` … **真の王手駒を玉以外で取る手**のうち現行順位が最上位のもの。
///   無ければ **coverage failure**（候補に無ければ完璧な信念でも救えない =
///   この issue の上限の外）
/// - `false_capture` … 真の王手駒**以外**のマスへの捕獲のうち最上位のもの
///   （希釈が「どこへ質量を配ってしまっているか」の代表）
#[derive(Clone, Debug, Default)]
pub struct Focus {
    pub true_capture: Option<usize>,
    pub false_capture: Option<usize>,
}

/// `moves` から [`Focus`] を作る。`moves` は現行順位の順に並んでいること。
pub fn focus(
    moves: &[crate::check_policy::PolicyMove],
    truth: &Position,
    bot: Color,
) -> Focus {
    let checkers = true_checkers(truth, bot);
    let mut out = Focus::default();
    for (i, m) in moves.iter().enumerate() {
        if m.is_king {
            continue; // 玉での捕獲は `legal_under` の盲点の領分（非目的）
        }
        let ShogiMove::Board { to, .. } = m.mv else { continue };
        let hits_checker = checkers.iter().any(|(s, _)| *s == to);
        // 「取る手」= 真実で相手の駒がいるマスへの移動（裁定側の情報なので
        // 方策には渡さない。分解の注目手を選ぶためだけに使う）
        let is_capture = truth
            .piece_at(to)
            .is_some_and(|p| p.color == bot.other());
        if hits_checker && out.true_capture.is_none() {
            out.true_capture = Some(i);
        } else if is_capture && !hits_checker && out.false_capture.is_none() {
            out.false_capture = Some(i);
        }
    }
    out
}

/// 王手中の bot の決定点（**終端手番を含む**）。
pub struct Point {
    /// 手番開始時（この手番の反則を食う前）の状態
    pub entry: crate::scenario_core::Replayed,
    /// 決定点の真実局面（裁定・注目手の選択にだけ使う）
    pub truth: Position,
    pub bot: Color,
    pub move_number: u32,
    /// 実戦の反則列（順番どおり）
    pub record_fouls: Vec<String>,
    /// 実戦で受理された手（終端手番は `None`）
    pub record_accepted: Option<String>,
    /// 直前の相手手の USI（王手駒の種別を打ち／盤上に分けるためだけに使う。
    /// **方策へは渡さない**）
    pub opp_last_usi: Option<String>,
    /// 直前の相手手が自駒を取ったマス（観測で分かる = 捕獲つきの王手）
    pub opp_captured_at: Option<String>,
    /// 受理手が無い（反則だけ積んで終局した）手番
    pub terminal: bool,
}

impl Point {
    pub fn estimand(&self) -> &'static str {
        if self.record_fouls.is_empty() { "nofoul" } else { "foul" }
    }
}

/// 母集団の取り出しで落ちた本数（**attrition として必ず出す**: 改善対象の
/// 最悪ケース＝終端手番が系統的に欠測すると門が甘くなる）。
#[derive(Clone, Copy, Debug, Default)]
pub struct Attrition {
    /// 王手中の bot の手番の総数
    pub turns: u32,
    /// うち終端手番
    pub terminal: u32,
    /// 手番開始時の状態を復元できなかった本数
    pub unreplayable: u32,
}

/// 1局から王手中の bot の決定点を全部取り出す（**終端手番を含む**）。
///
/// 復元できなかった手番は [`Attrition`] に積むだけで黙って落とさない。
pub fn decision_points(
    end: &crate::protocol::GameEndPayload,
    bot: Color,
    at: &mut Attrition,
) -> Option<Vec<Point>> {
    use crate::check_economy::entry_replayed;
    use crate::observation::Observation;
    use crate::scenario_core::{clone_log, side_idx};
    let snaps = crate::check_prep::decision_snapshots(end)?;
    let mut out = vec![];
    for s in &snaps {
        if s.side != bot || !s.pos.in_check(bot) {
            continue;
        }
        at.turns += 1;
        if s.terminal {
            at.terminal += 1;
        }
        let post = crate::scenario_core::Replayed {
            pos: s.pos.clone(),
            logs: [clone_log(&s.logs[0]), clone_log(&s.logs[1])],
            fouls: s.fouls,
            plies: 0,
            injected_fouls: vec![],
            oracle: None,
        };
        let Some(entry) = entry_replayed(&post, bot, s.fouls_this_turn) else {
            at.unreplayable += 1;
            continue;
        };
        let events = post.logs[side_idx(bot)].events();
        let record_fouls: Vec<String> = events[events.len() - s.fouls_this_turn as usize..]
            .iter()
            .filter_map(|e| match e {
                Observation::MyFoul { usi, .. } => Some(usi.clone()),
                _ => None,
            })
            .collect();
        // 直前の相手手（種別の層分けに使う。観測ではなく記録から取る）
        let opp_last_usi = (s.decision_id > 0)
            .then(|| end.moves.get(s.decision_id as usize - 1).map(|m| m.usi.clone()))
            .flatten();
        let opp_captured_at = post.logs[side_idx(bot)]
            .events()
            .iter()
            .rev()
            .find_map(|e| match e {
                Observation::OpponentMoved { captured_my_piece_at, .. } => {
                    Some(captured_my_piece_at.clone())
                }
                _ => None,
            })
            .flatten();
        out.push(Point {
            opp_last_usi,
            opp_captured_at,
            move_number: s.pos.move_number(),
            truth: s.pos.clone(),
            record_accepted: if s.terminal {
                None
            } else {
                end.moves.get(s.decision_id as usize).map(|m| m.usi.clone())
            },
            terminal: s.terminal,
            record_fouls,
            entry,
            bot,
        });
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn belief_tags_roundtrip() {
        for b in [
            Belief::Current,
            Belief::Oracle { k: 2.0 },
            Belief::Oracle { k: ORACLE_INF },
            Belief::Misdirected { k: 4.0 },
            Belief::DeduceLastMove,
        ] {
            assert_eq!(Belief::parse(&b.tag()), Some(b), "{}", b.tag());
        }
        assert_eq!(Belief::parse("oracle@kinf"), Some(Belief::Oracle { k: ORACLE_INF }));
        assert!(Belief::parse("oracle@k-1").is_none());
        assert!(Belief::parse("nope").is_none());
    }

    #[test]
    fn armタグはbeliefとpolicyの両方を読む() {
        use crate::check_policy::Policy;
        let a = ArmSpec::parse("oracle@kinf@real").unwrap();
        assert_eq!(a.belief, Belief::Oracle { k: ORACLE_INF });
        assert_eq!(a.policy, Policy::Current);
        assert!(a.real);
        let b = ArmSpec::parse("alpha@k2").unwrap();
        assert_eq!(b.belief, Belief::Current);
        assert_eq!(b.policy, Policy::Alpha { k: 2.0 });
        assert!(!b.real);
        let c = ArmSpec::parse("current@real").unwrap();
        assert_eq!((c.belief, c.policy, c.real), (Belief::Current, Policy::Current, true));
        assert_eq!(ArmSpec::parse("deduce_last_move@shadow").unwrap().belief, Belief::DeduceLastMove);
        assert!(ArmSpec::parse("oracle@k?@real").is_none());
        assert!(ArmSpec::parse("nope").is_none());
    }

    #[test]
    fn 恒等対照は介入しない() {
        // `oracle@k1` は「真仮説 ×1」= 何も変えない。`current` と同じく
        // フックの中身が空であることを型で確かめる（bit-exact の前提）
        let truth = Position::initial();
        let d = Belief::Oracle { k: 1.0 }.diag(&truth, Color::Sente, &[]);
        assert!(d.factors.values().all(|v| *v == 1.0));
        assert!(!d.deduce_last_move);
        assert!(Belief::Current.diag(&truth, Color::Sente, &[]).is_noop());
    }
}
