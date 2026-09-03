//! 評価項の発火率フック（やねうら王 `source/misc.h` の `dbg_hit_on` /
//! `dbg_mean_of` に相当する考え方。docs/yaneuraou-lessons.md 付録）。
//!
//! **解こうとしている問題**: これまで「アリーナ中立」で据え置いた評価項
//! （詰めろ2項・king_hole_w）は、*効いていないのか* *そもそも発火していないのか*
//! が区別できていない。事後の `bin/analyze` や `bin/rank_probe` は特定の
//! 局面・記録しか見ないので、対局全体での発火率が分からない。
//!
//! ここでは対局中の全決定について、項ごとに
//!
//! - **発火局面率**: その項が非ゼロだった候補を1つ以上含む決定の割合
//! - **発火候補率・平均寄与**: 非ゼロだった候補の割合と、その寄与の平均（歩価値）
//! - **top1入替率**: その項を外すと選択手（argmax）が変わる決定の割合 ← 本命
//!
//! を数える。寄与があっても順位を変えなければ勝率に効きようがないので、
//! 3つ目が「中立だった理由」の一次診断になる。
//!
//! `TSUITATE_DBG_HITS=1` のときだけ集計する（既定は完全に無効で、
//! ホットパスには `enabled()` の一度きりの読み出ししか残らない）。
//! 出力は `dump()` を呼んだ側（`bin/arena` の終了時など）が表示する。

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use crate::strategy::CandidateScore;

pub fn enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("TSUITATE_DBG_HITS").is_ok_and(|v| v == "1"))
}

#[derive(Debug, Default, Clone)]
struct TermStat {
    decisions: u64,
    fired_decisions: u64,
    candidates: u64,
    fired_candidates: u64,
    sum_abs: f64,
    top1_flips: u64,
}

fn stats() -> &'static Mutex<BTreeMap<&'static str, TermStat>> {
    static S: OnceLock<Mutex<BTreeMap<&'static str, TermStat>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// 条件の発火率（やねうら王 `dbg_hit_on(cond)` に相当）。(発火, 総数)
fn flags() -> &'static Mutex<BTreeMap<&'static str, (u64, u64)>> {
    static S: OnceLock<Mutex<BTreeMap<&'static str, (u64, u64)>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// 条件が成り立った割合を数える。`enabled()` のときだけ呼ぶこと。
///
/// 「その評価項が発火する前提条件（粒子が生きている・王手でない等）を
/// そもそも満たしているか」を測るためのもの。項の寄与を見る
/// `observe_ranking` と違い、評価の**手前**の条件を見る
pub fn flag(name: &'static str, cond: bool) {
    let mut map = flags().lock().expect("hits のフラグロック");
    let e = map.entry(name).or_insert((0, 0));
    e.0 += cond as u64;
    e.1 += 1;
}

/// gain の内側にある項の一覧。値は **gain への符号つき寄与**
/// （正 = gain を押し上げている）。ここに足せば自動で集計対象になる
fn terms() -> &'static [(&'static str, fn(&CandidateScore) -> f64)] {
    &[
        ("value_nn", |c| c.value_nn),
        ("checker_removal", |c| c.checker_removal),
        ("capture_bet_var", |c| -c.capture_bet_penalty),
        ("mate_threat", |c| c.mate_threat),
        ("mate_risk", |c| -c.mate_risk),
        ("king_hole", |c| -c.king_holes),
        ("link", |c| c.link),
        ("promo", |c| c.promo),
        ("promote_bias", |c| c.promote_bias),
        ("drop_bias", |c| c.drop_bias),
        ("hand_option", |c| -c.hand_option),
        ("board_discount", |c| -c.board_discount),
    ]
}

/// 1決定ぶんの候補ランキングを集計する。`choose()` の末尾から呼ぶ。
///
/// top1入替の判定はランキングに残っている内訳から**最終スコアを組み直して**
/// 行う（`gain - 寄与` を `combine_score` に通し直す）。スコアから寄与を
/// 引くだけでは駄目で、`combine_score` は min 形の非線形式なので
/// p_legal 割引の掛かり方まで変わりうる
pub fn observe_ranking(ranking: &[CandidateScore]) {
    let local = tally(ranking);
    let mut map = stats().lock().expect("hits の集計ロック");
    for (name, st) in local {
        let e = map.entry(name).or_default();
        e.decisions += st.decisions;
        e.fired_decisions += st.fired_decisions;
        e.candidates += st.candidates;
        e.fired_candidates += st.fired_candidates;
        e.sum_abs += st.sum_abs;
        e.top1_flips += st.top1_flips;
    }
}

/// スコア関数 f の argmax を**決定的に**取る: 乱数（`tiebreak`）は f の側で
/// 除いてあり、同点は USI の辞書順（小さいほうが首位）。issue #24 の教訓②
/// （タイブレーク乱数を順位に残さない・完全同点を安定ソートに任せない）を
/// shadow 順位にもそのまま適用する（issue #40 P0-2 の事前登録規約）
fn det_argmax<'a>(
    ranking: &'a [CandidateScore],
    f: impl Fn(&CandidateScore) -> f64,
) -> &'a CandidateScore {
    ranking
        .iter()
        .max_by(|a, b| {
            f(a).partial_cmp(&f(b))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.usi.cmp(&a.usi))
        })
        .expect("空でないことを確認済み")
}

/// 1決定ぶんの集計（純関数。テスト対象）。
///
/// top1入替の判定は**乱数を除いた決定的スコア**で行う:
/// - 現首位: `score − tiebreak`
/// - 項除去後: `combine_score(gain − 寄与, p_legal, foul_cost) + foul_probe +
///   (adjust − tiebreak)`（score の組み立てと同じ形。`foul_probe` は
///   combine_score の外側の項なので落とさない）
/// どちらも同点は USI 辞書順。乱数込みで数えると、W=0 の A/A でも
/// 同点候補の乱数の並びだけで top1_flips が非ゼロになる
fn tally(ranking: &[CandidateScore]) -> Vec<(&'static str, TermStat)> {
    if ranking.is_empty() {
        return vec![];
    }
    let best = det_argmax(ranking, |c| c.score - c.tiebreak);
    let mut local: Vec<(&'static str, TermStat)> = vec![];
    for (name, get) in terms() {
        let mut st = TermStat {
            decisions: 1,
            candidates: ranking.len() as u64,
            ..TermStat::default()
        };
        for c in ranking {
            let v = get(c);
            if v != 0.0 {
                st.fired_candidates += 1;
                st.sum_abs += v.abs();
            }
        }
        if st.fired_candidates > 0 {
            st.fired_decisions = 1;
            // その項を外したときの argmax（乱数を除いた決定的スコア）
            let alt = det_argmax(ranking, |c| {
                crate::strategy::combine_score(c.gain - get(c), c.p_legal, c.foul_cost)
                    + c.foul_probe
                    + c.adjust
                    - c.tiebreak
            });
            if alt.usi != best.usi {
                st.top1_flips = 1;
            }
        }
        local.push((name, st));
    }
    local
}

/// 集計結果の表。無効時・データなしなら None
pub fn dump() -> Option<String> {
    if !enabled() {
        return None;
    }
    let map = stats().lock().expect("hits の集計ロック").clone();
    let fl = flags().lock().expect("hits のフラグロック").clone();
    if map.is_empty() && fl.is_empty() {
        return None;
    }
    let pct = |num: u64, den: u64| -> f64 {
        if den == 0 {
            0.0
        } else {
            num as f64 / den as f64 * 100.0
        }
    };
    let mut out = String::from("=== 評価項の発火率（TSUITATE_DBG_HITS）===\n");
    out.push_str("項               発火局面   発火候補   平均寄与   top1入替(全/発火時)\n");
    for (name, s) in &map {
        out.push_str(&format!(
            "{name:<16} {:>7.1}% {:>9.1}% {:>10.3} {:>9.1}% /{:>6.1}%\n",
            pct(s.fired_decisions, s.decisions),
            pct(s.fired_candidates, s.candidates),
            if s.fired_candidates > 0 {
                s.sum_abs / s.fired_candidates as f64
            } else {
                0.0
            },
            pct(s.top1_flips, s.decisions),
            pct(s.top1_flips, s.fired_decisions),
        ));
    }
    if !fl.is_empty() {
        out.push_str("--- 条件の発火率（評価に入る手前の前提）---\n");
        for (name, (hit, total)) in &fl {
            out.push_str(&format!(
                "{name:<24} {:>7.1}%  ({hit}/{total})\n",
                pct(*hit, *total)
            ));
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(usi: &str, gain: f64, mate_risk: f64) -> CandidateScore {
        CandidateScore {
            usi: usi.into(),
            static_score: gain,
            static_gain: gain,
            score: crate::strategy::combine_score(gain, 1.0, 0.0),
            gain,
            p_legal: 1.0,
            foul_cost: 0.0,
            adjust: 0.0,
            depth2: false,
            checker_removal: 0.0,
            capture_bet_penalty: 0.0,
            foul_probe: 0.0,
            tiebreak: 0.0,
            probe_unit: 0.0,
            probe_mass: 0.0,
            probe_concentration: 0.0,
            mate_threat: 0.0,
            mate_risk,
            king_holes: 0.0,
            value_nn: 0.0,
            capture_value: 0.0,
            risk: 0.0,
            link: 0.0,
            promo: 0.0,
            promote_bias: 0.0,
            drop_bias: 0.0,
            hand_option: 0.0,
            board_discount: 0.0,
            own_zone: 0.0,
        }
    }

    fn get<'a>(t: &'a [(&'static str, TermStat)], name: &str) -> &'a TermStat {
        &t.iter().find(|(n, _)| *n == name).expect("項が集計対象").1
    }

    /// 項を外すと argmax が変わる決定を top1入替として数える。
    /// A は素の gain 2.0 が被詰めろ 1.5 の控除で 0.5 に沈み、top1 は B になる
    /// （= mate_risk が選択を決めている）
    #[test]
    fn counts_top1_flip_when_term_decides() {
        let t = tally(&[cand("B", 1.0, 0.0), cand("A", 0.5, 1.5)]);
        let risk = get(&t, "mate_risk");
        assert_eq!(risk.decisions, 1);
        assert_eq!(risk.fired_decisions, 1);
        assert_eq!(risk.fired_candidates, 1);
        assert_eq!(risk.top1_flips, 1);
        assert!((risk.sum_abs - 1.5).abs() < 1e-9);
    }

    /// 寄与があっても順位を変えなければ top1入替は立たない
    /// （「効いていない」と「発火していない」の区別が本フックの目的）
    #[test]
    fn firing_without_changing_the_choice_is_not_a_flip() {
        // 沈むのは元々2位の A だけなので、外しても top1 は B のまま
        let t = tally(&[cand("B", 3.0, 0.0), cand("A", 0.5, 1.5)]);
        let risk = get(&t, "mate_risk");
        assert_eq!(risk.fired_decisions, 1);
        assert_eq!(risk.top1_flips, 0);
    }

    /// 発火していない項は局面を数えるだけ（分母は全決定）
    #[test]
    fn silent_terms_count_decisions_only() {
        let t = tally(&[cand("B", 1.0, 0.0), cand("A", 0.5, 0.0)]);
        let threat = get(&t, "mate_threat");
        assert_eq!(threat.decisions, 1);
        assert_eq!(threat.fired_decisions, 0);
        assert_eq!(threat.candidates, 2);
        assert_eq!(threat.fired_candidates, 0);
    }

    /// shadow 順位は**乱数を除き、同点は USI 辞書順**（issue #24 教訓②）。
    /// 決定的スコアが完全同点の2候補で、乱数だけが順位を分けている決定は
    /// 項を外しても首位が変わらない = top1入替を数えない
    #[test]
    fn tiebreak_randomness_never_flips_top1() {
        let mut a = cand("A", 1.0, 0.5);
        let mut b = cand("B", 1.0, 0.5);
        // 乱数込みでは B が首位、乱数を除けば同点 → USI 辞書順で A が首位
        a.tiebreak = 0.001;
        a.score += a.tiebreak;
        a.adjust += a.tiebreak;
        b.tiebreak = 0.009;
        b.score += b.tiebreak;
        b.adjust += b.tiebreak;
        let t = tally(&[b, a]);
        let risk = get(&t, "mate_risk");
        assert_eq!(risk.fired_decisions, 1);
        assert_eq!(risk.top1_flips, 0, "両候補が対称なので外しても首位（辞書順 A）は不変");
    }

    /// `foul_probe` は combine_score の外側の項なので、項除去後のスコアからも
    /// 落とさない（落とすと foul_probe で首位に立っている候補が全項の
    /// 「除去後」で沈み、無関係な項に top1入替が立つ）
    #[test]
    fn foul_probe_is_kept_in_shadow_score() {
        let mut probe = cand("A", 1.0, 0.0);
        probe.foul_probe = 2.0;
        probe.score += probe.foul_probe;
        // B は mate_risk が発火しているが、外しても A（foul_probe 込み 3.0）が首位のまま
        let t = tally(&[probe, cand("B", 2.0, 0.5)]);
        let risk = get(&t, "mate_risk");
        assert_eq!(risk.fired_decisions, 1);
        assert_eq!(risk.top1_flips, 0);
    }
}
