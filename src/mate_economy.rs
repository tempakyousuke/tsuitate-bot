//! 詰み経済の**共有定義**（issue #28）。
//!
//! `bin/analyze`（P0-1/P0-2 の集計）・`bin/mate_probe`（P0-3 の較正と argmax
//! シミュレーション）・`bin/mate_continue`（P0-6 の一手強制の継続診断）が
//! **同じ定義**で「被詰めろ」「安全手」「詰め手の分類」を数えるための場所。
//!
//! ここに置くのは runtime に一切入らない診断だけ。評価・推定の挙動は変えない
//! （`combine_score` を読むのは P0-3 のオフライン再スコアだけで、書き戻さない）。

use std::collections::HashSet;

use crate::mate::{has_mate_in_1_fast, mate_moves_in_1_fast};
use crate::observation::ObservationLog;
use crate::protocol::{Color, GameEndPayload};
use crate::shogi::{Outcome, Position, ShogiMove, parse_usi};
use crate::strategy::CandidateScore;

/// 詰め手集合の排他的分類（issue #28 P0-2）。
///
/// 従来の `打ち詰み` フラグは `mate::drop_mate` の有無だったので「打ちでも
/// 盤上移動でも詰む」局面が打ち側にだけ数えられ、**排他的な分類になって
/// いなかった**。全詰め手を列挙して union で分類する。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MateKind {
    DropOnly,
    BoardOnly,
    Both,
}

impl MateKind {
    pub fn of(moves: &[ShogiMove]) -> Option<Self> {
        let drop = moves.iter().any(|m| matches!(m, ShogiMove::Drop { .. }));
        let board = moves.iter().any(|m| matches!(m, ShogiMove::Board { .. }));
        match (drop, board) {
            (true, true) => Some(MateKind::Both),
            (true, false) => Some(MateKind::DropOnly),
            (false, true) => Some(MateKind::BoardOnly),
            (false, false) => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            MateKind::DropOnly => "打ちのみ",
            MateKind::BoardOnly => "盤上のみ",
            MateKind::Both => "両方",
        }
    }

    /// 集計キー用の短い英字ラベル（CSV / JSONL）
    pub fn tag(self) -> &'static str {
        match self {
            MateKind::DropOnly => "drop_only",
            MateKind::BoardOnly => "board_only",
            MateKind::Both => "both",
        }
    }

    /// 2つの分類を union として畳む（エピソード内の手番をまとめるとき）
    pub fn merge(self, other: MateKind) -> MateKind {
        if self == other { self } else { MateKind::Both }
    }
}

/// 記録の真実（`game:end` の全手順）を初期局面から再生した局面列。
/// `positions[i]` は「i 手目を指す直前」= 決定点 i の局面。
/// 棋譜が壊れていたら None（その局は丸ごと捨てる）。
pub fn replay_positions(end: &GameEndPayload) -> Option<Vec<Position>> {
    let mut positions = vec![Position::initial()];
    for m in &end.moves {
        let mut next = positions.last().expect("初期局面がある").clone();
        let mv = parse_usi(&m.usi)?;
        if !next.is_legal(&mv) {
            return None;
        }
        next.play_unchecked(&mv);
        positions.push(next);
    }
    Some(positions)
}

/// 手番側の**安全手**（指した後に相手の一手詰めが消える手）の全量。
///
/// 真実の盤面を見るので理論上限であって、bot が到達できるとは限らない
/// （候補生成に載るか・信念が支持するかは漏斗の後段。P0-2）。
pub fn safe_moves_at(decision: &Position) -> Vec<ShogiMove> {
    let mut out = vec![];
    for mv in decision.legal_moves() {
        let mut next = decision.clone();
        next.play_unchecked(&mv);
        if !has_mate_in_1_fast(&next) {
            out.push(mv);
        }
    }
    out
}

/// bot が被詰めろになっている**相手番**1つ分。
pub struct ThreatTurn {
    /// 相手番の局面のインデックス（`replay_positions` の添字）
    pub idx: usize,
    /// この手番に存在した詰め手の全量
    pub mates: Vec<ShogiMove>,
    pub kind: MateKind,
    /// 実際にこの手番で詰まされたなら、その手
    pub executed: Option<ShogiMove>,
    /// 一手巻き戻した bot の決定点（`idx - 1`。手番が bot でなければ None）
    pub decision_idx: Option<usize>,
    /// その決定点で真実上あった安全手（decision_idx が None なら空）
    pub safe: Vec<ShogiMove>,
}

impl ThreatTurn {
    pub fn avoidable(&self) -> bool {
        self.decision_idx.is_some() && !self.safe.is_empty()
    }
}

/// 被詰めろの手番を列挙し、一手巻き戻した bot の決定点の安全手まで数える。
///
/// `mates_of` は詰め手の列挙（コスト計測を挟めるように注入する。中身は
/// `mate::mate_moves_in_1_fast`）。`ply` は 1 始まりの手数（計測のラベル用）。
pub fn threat_turns(
    positions: &[Position],
    bot: Color,
    mates_of: &mut dyn FnMut(&Position, usize) -> Vec<ShogiMove>,
    played: &dyn Fn(usize) -> Option<ShogiMove>,
) -> Vec<ThreatTurn> {
    let mut out = vec![];
    for (idx, pos) in positions.iter().enumerate() {
        if pos.turn() == bot || pos.outcome().is_some() {
            continue;
        }
        let mates = mates_of(pos, idx + 1);
        let Some(kind) = MateKind::of(&mates) else {
            continue;
        };
        // 実際に詰まされたか（この相手番の手で終局したか）
        let executed = positions
            .get(idx + 1)
            .and_then(|p| p.outcome())
            .and_then(|o| match o {
                Outcome::Checkmate { winner } if winner != bot => played(idx),
                _ => None,
            });
        let decision_idx = (idx >= 1 && positions[idx - 1].turn() == bot).then_some(idx - 1);
        let safe = decision_idx.map_or(vec![], |d| safe_moves_at(&positions[d]));
        out.push(ThreatTurn {
            idx,
            mates,
            kind,
            executed,
            decision_idx,
            safe,
        });
    }
    out
}

/// 被詰めろの**エピソード**（連続した相手番を1つに畳んだもの）。
///
/// 手番単位で数えると「1回の危険が何手番続いたか」と「何回危険に入ったか」が
/// 混ざる（issue #28 P0-1: 局が長くなっただけの増加と区別できない）。
pub struct MateEpisode {
    /// `threat_turns` の戻り値への添字（時系列順・1つ以上）
    pub turns: Vec<usize>,
}

impl MateEpisode {
    /// 最後に**受けられた**手番（安全手が真実上あった最後の手番）
    pub fn last_avoidable<'a>(&self, turns: &'a [ThreatTurn]) -> Option<&'a ThreatTurn> {
        self.turns
            .iter()
            .rev()
            .map(|&i| &turns[i])
            .find(|t| t.avoidable())
    }
}

/// 連続した被詰めろ手番（相手番なので添字は2ずつ進む）を1エピソードへ畳む。
pub fn fold_episodes(turns: &[ThreatTurn]) -> Vec<MateEpisode> {
    let mut out: Vec<MateEpisode> = vec![];
    let mut prev: Option<usize> = None;
    for (i, t) in turns.iter().enumerate() {
        let continues = prev == Some(t.idx.saturating_sub(2));
        prev = Some(t.idx);
        match out.last_mut() {
            Some(ep) if continues => ep.turns.push(i),
            _ => out.push(MateEpisode { turns: vec![i] }),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 記録1局ぶんの被詰めろ解析（P0-3 / P0-6 の入口）
// ---------------------------------------------------------------------------

/// 記録1局を再生した被詰めろの全量。
pub struct GameDefense {
    /// `positions[i]` = i 手目を指す直前の真実の局面
    pub positions: Vec<Position>,
    pub turns: Vec<ThreatTurn>,
    pub episodes: Vec<MateEpisode>,
    /// bot が詰まされて負けた局か
    pub bot_mated: bool,
    /// bot の決定点（終局前）の添字。手数調整の分母
    pub bot_decisions: Vec<usize>,
}

impl GameDefense {
    /// 「最後に受けられた決定点」（詰み負けの局なら、詰まされる前に真実上
    /// 受けが残っていた最後の bot の決定点）。
    ///
    /// P0-3 の argmax シミュレーションと P0-6 の一手強制はここから始める。
    pub fn last_defense_point(&self) -> Option<&ThreatTurn> {
        self.episodes
            .iter()
            .rev()
            .find_map(|ep| ep.last_avoidable(&self.turns))
    }
}

/// 記録の終局ペイロードから被詰めろを解析する。棋譜が壊れていたら None。
pub fn analyze_game(
    end: &GameEndPayload,
    bot: Color,
    mates_of: &mut dyn FnMut(&Position, usize) -> Vec<ShogiMove>,
) -> Option<GameDefense> {
    let positions = replay_positions(end)?;
    let played = |i: usize| end.moves.get(i).and_then(|m| parse_usi(&m.usi));
    let turns = threat_turns(&positions, bot, mates_of, &played);
    let episodes = fold_episodes(&turns);
    let bot_mated = matches!(
        positions.last().and_then(|p| p.outcome()),
        Some(Outcome::Checkmate { winner }) if winner != bot
    );
    let bot_decisions = (0..positions.len())
        .filter(|&i| positions[i].turn() == bot && positions[i].outcome().is_none())
        .collect();
    Some(GameDefense {
        positions,
        turns,
        episodes,
        bot_mated,
        bot_decisions,
    })
}

/// **対照の決定点**（P0-3 の偽陽性率）: 危険な決定点と同じ手数帯にある、
/// 被詰めろに繋がらなかった bot の決定点を近い順に選ぶ。
///
/// 対照を取らないと「危険を当てた」ようにしか見えず、**誤爆が見えない**
/// （issue #28 P0-3）。同じ局・同じ手数帯から取るのは、局面分布の違いを
/// 偽陽性率に混ぜないため。
pub fn control_points(
    g: &GameDefense,
    danger_ply: usize,
    want: usize,
    exclude: &HashSet<usize>,
) -> Vec<usize> {
    let dangerous: HashSet<usize> = g.turns.iter().filter_map(|t| t.decision_idx).collect();
    let mut cands: Vec<usize> = g
        .bot_decisions
        .iter()
        .copied()
        .filter(|i| !dangerous.contains(i) && !exclude.contains(i))
        .collect();
    cands.sort_by_key(|&i| (i.abs_diff(danger_ply), i));
    cands.truncate(want);
    cands
}

// ---------------------------------------------------------------------------
// P0-6: 一手強制（審判と同じ規約で適用する）
// ---------------------------------------------------------------------------

/// 一手強制の適用結果。
pub struct Forced {
    pub pos: Position,
    pub logs: [ObservationLog; 2],
    pub fouls: [u32; 2],
    /// 実際に受理された手（候補が全部非合法／反則上限なら None）
    pub played: Option<String>,
    /// 非合法な候補を試して積んだ反則の数
    pub forced_fouls: u32,
    /// 反則が上限（`selfplay::MAX_FOULS`）に達したか = その場で反則負け
    pub foul_limit: bool,
}

/// 優先順位つきの候補列 `order` の先頭から順に指させる。
///
/// **非合法なら実対局と同じく反則を1つ積んで次の候補へ進む**（手番は変わらない）。
/// オフライン方策は bot の信念で選ぶので真実で非合法な手を選びうるので、
/// そこを「測定不能」にすると方策の実力を過大評価する。観測の記録は審判と
/// 同じ関数（`truth_replay::add_move_obs` / `add_foul_obs`）を通す。
pub fn force_move(
    start: &Position,
    logs: &[ObservationLog; 2],
    fouls: [u32; 2],
    side: Color,
    order: &[String],
) -> Forced {
    let mut pos = start.clone();
    let mut logs = [
        crate::scenario_core::clone_log(&logs[0]),
        crate::scenario_core::clone_log(&logs[1]),
    ];
    let mut fouls = fouls;
    let mut forced_fouls = 0u32;
    let idx = if side == Color::Sente { 0 } else { 1 };
    for usi in order {
        let Some(mv) = parse_usi(usi) else { continue };
        if pos.is_legal(&mv) {
            let captured = pos.play_unchecked(&mv);
            crate::truth_replay::add_move_obs(side, &mv, captured, &pos, &mut logs);
            return Forced {
                pos,
                logs,
                fouls,
                played: Some(usi.clone()),
                forced_fouls,
                foul_limit: false,
            };
        }
        crate::truth_replay::add_foul_obs(side, usi.clone(), &pos, &mut logs, &mut fouls);
        forced_fouls += 1;
        if fouls[idx] >= crate::selfplay::MAX_FOULS {
            return Forced {
                pos,
                logs,
                fouls,
                played: None,
                forced_fouls,
                foul_limit: true,
            };
        }
    }
    Forced {
        pos,
        logs,
        fouls,
        played: None,
        forced_fouls,
        foul_limit: false,
    }
}

// ---------------------------------------------------------------------------
// P0-3: 盤上版のオフライン試作（runtime には入れない）
// ---------------------------------------------------------------------------

/// 粒子上の被詰め質量。分母は**その粒子集合の総重み**（合法でない粒子は
/// 詰みを許さないので分子に入らない = 「反則になる手はその世界では詰まれない」）。
#[derive(Clone, Copy, Debug, Default)]
pub struct MateMass {
    /// taint 込みの全粒子での質量 [0,1]
    pub all: f64,
    /// 厳密整合粒子だけでの質量 [0,1]（厳密が全滅していれば 0）
    pub strict: f64,
    /// 厳密粒子の重みシェア（0 なら「ブラインド決定」= 初版の P1-B は効かない）
    pub strict_share: f64,
}

/// 候補手を各粒子へ適用し、「相手に一手詰めが生じる」重みの割合を測る。
///
/// `particles` は **`choose` が評価に使ったプールそのもの**
/// （`strategy::ParticleSnapshot::entries`。`strict` は `stratified_sample` の
/// 戻り、`taint` は厳密が全滅したときの落とし先）。
/// **同じ seed で `build_estimator` を回し直したものを渡してはいけない**:
/// `Estimator::update` は壁時計デッドラインまで若返らせるので、同じ seed でも
/// 2回の実行で粒子集合が変わり、「ランキングは集合A・q は集合B」になる。
pub fn particle_mate_mass(particles: &[(&Position, f64, bool)], mv: &ShogiMove) -> MateMass {
    let (mut num_all, mut den_all) = (0.0f64, 0.0f64);
    let (mut num_strict, mut den_strict) = (0.0f64, 0.0f64);
    for &(pos, w, strict) in particles {
        den_all += w;
        if strict {
            den_strict += w;
        }
        if !pos.is_legal(mv) {
            continue;
        }
        let mut next = pos.clone();
        next.play_unchecked(mv);
        if has_mate_in_1_fast(&next) {
            num_all += w;
            if strict {
                num_strict += w;
            }
        }
    }
    MateMass {
        all: if den_all > 0.0 { num_all / den_all } else { 0.0 },
        strict: if den_strict > 0.0 {
            num_strict / den_strict
        } else {
            0.0
        },
        strict_share: if den_all > 0.0 {
            den_strict / den_all
        } else {
            0.0
        },
    }
}

/// 1候補手ぶんのオフライン行（P0-3 の CSV 1行）。
#[derive(Clone, Debug)]
pub struct MateRow {
    pub usi: String,
    /// タイブレーク乱数を除いた現行スコア（`score - tiebreak`）
    pub det_score: f64,
    /// 乱数を除いた現行順位（0 始まり。`eval_rank::det_order` と同じ規約）
    pub rank: usize,
    pub gain: f64,
    pub p_legal: f64,
    pub foul_cost: f64,
    pub foul_probe: f64,
    /// 乱数を除いた `adjust`
    pub adjust_det: f64,
    pub depth2: bool,
    /// 現行の被詰めろ項（打ち詰みのみ）の発火量。0 なら視野外
    pub mate_risk: f64,
    pub mass: MateMass,
    /// 真実の盤面でこの手が合法か
    pub truth_legal: bool,
    /// 真実の盤面で指した後に相手の一手詰めが生じるか（非合法なら None）
    pub truth_allows_mate: Option<bool>,
    /// 真実で許してしまう詰め手の排他的分類（許さないなら None）
    pub truth_kind: Option<MateKind>,
}

impl MateRow {
    /// 危険質量（`taint` を使うかで切り替える。初版の P1-B は厳密粒子のみ）
    pub fn q(&self, use_taint: bool) -> f64 {
        if use_taint {
            self.mass.all
        } else {
            self.mass.strict
        }
    }

    /// **オフラインの再スコア**（P1-B の形）。
    ///
    /// 危険量を `gain` 側へ引く（issue #24 の教訓⑦: 最終 score へ直接足すのは
    /// 等価でない。`combine_score` の内側に置かないと合法性の割引を迂回する）。
    /// `w = 0` なら `det_score` に一致する（`rescore_matches_det_score` が検査）。
    pub fn rescore(&self, w: f64, use_taint: bool) -> f64 {
        crate::strategy::combine_score(
            self.gain - w * self.q(use_taint),
            self.p_legal,
            self.foul_cost,
        ) + self.foul_probe
            + self.adjust_det
    }
}

/// 現行ランキング（乱数込みのスコア順）と粒子集合から P0-3 の行を作る。
///
/// 順位は**乱数を除いたスコア**で付け直す（issue #24 の教訓②: 同点を安定
/// ソートに任せると乱数の並び順がそのまま順位に残るので USI の辞書順で割る）。
pub fn build_rows(
    truth: &Position,
    ranking: &[CandidateScore],
    particles: &[(&Position, f64, bool)],
) -> Vec<MateRow> {
    let order = crate::eval_rank::det_order(ranking);
    let mut rank_of = vec![0usize; ranking.len()];
    for (r, &i) in order.iter().enumerate() {
        rank_of[i] = r;
    }
    let mut out = Vec::with_capacity(ranking.len());
    for (i, c) in ranking.iter().enumerate() {
        let Some(mv) = parse_usi(&c.usi) else { continue };
        let truth_legal = truth.is_legal(&mv);
        let (truth_allows_mate, truth_kind) = if truth_legal {
            let mut next = truth.clone();
            next.play_unchecked(&mv);
            let mates = mate_moves_in_1_fast(&next);
            let kind = MateKind::of(&mates);
            (Some(kind.is_some()), kind)
        } else {
            (None, None)
        };
        out.push(MateRow {
            usi: c.usi.clone(),
            det_score: c.score - c.tiebreak,
            rank: rank_of[i],
            gain: c.gain,
            p_legal: c.p_legal,
            foul_cost: c.foul_cost,
            foul_probe: c.foul_probe,
            adjust_det: c.adjust - c.tiebreak,
            depth2: c.depth2,
            mate_risk: c.mate_risk,
            mass: particle_mate_mass(particles, &mv),
            truth_legal,
            truth_allows_mate,
            truth_kind,
        });
    }
    out.sort_by_key(|r| r.rank);
    out
}

/// 再スコアの argmax（同点は USI の辞書順で割る = 乱数を残さない）。
pub fn pick<'a>(rows: &'a [MateRow], w: f64, use_taint: bool) -> Option<&'a MateRow> {
    rows.iter().reduce(|best, row| {
        let (a, b) = (row.rescore(w, use_taint), best.rescore(w, use_taint));
        if a > b || (a == b && row.usi < best.usi) {
            row
        } else {
            best
        }
    })
}

/// 再スコアの降順に並べた USI（同点は辞書順）。
///
/// `bin/mate_continue` の `Δpolicy` は先頭が真実で非合法だったときに次点へ
/// 進む（実対局の反則と同じ）ので、argmax だけでなく順序が要る。
pub fn rescored_usis(rows: &[MateRow], w: f64, use_taint: bool) -> Vec<String> {
    let mut idx: Vec<usize> = (0..rows.len()).collect();
    idx.sort_by(|&a, &b| {
        let (x, y) = (rows[a].rescore(w, use_taint), rows[b].rescore(w, use_taint));
        y.partial_cmp(&x)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| rows[a].usi.cmp(&rows[b].usi))
    });
    idx.into_iter().map(|i| rows[i].usi.clone()).collect()
}

/// argmax が安全手（`safe`）へ変わる最小の w。`grid` の昇順に走査する。
///
/// 再スコアは w に対して単調でない（候補ごとに沈む速さが違う）ので、
/// 二分探索でなく格子の走査で「最初に安全手が argmax になる w」を返す。
pub fn min_w_to_flip(
    rows: &[MateRow],
    safe: &HashSet<String>,
    use_taint: bool,
    grid: &[f64],
) -> Option<f64> {
    grid.iter()
        .copied()
        .find(|&w| pick(rows, w, use_taint).is_some_and(|r| safe.contains(&r.usi)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{
        FoulRecord, GameEndPayload, MoveRecord, OpponentInfo, RatingChange, RatingChangePair,
    };

    fn end_of(moves: &[(&str, Color)]) -> GameEndPayload {
        GameEndPayload {
            result: "win".into(),
            reason: "checkmate".into(),
            final_sfen: String::new(),
            moves: moves
                .iter()
                .map(|&(usi, by_color)| MoveRecord {
                    usi: usi.into(),
                    by_color,
                    ms: 0,
                    fouls_before: 0,
                })
                .collect(),
            foul_attempts: Vec::<FoulRecord>::new(),
            rating_change: RatingChangePair {
                you: RatingChange { before: 0, after: 0 },
                opponent: RatingChange { before: 0, after: 0 },
            },
            opponent: OpponentInfo {
                username: String::new(),
                rating: 0,
                is_bot: true,
            },
        }
    }

    /// 実戦局面（`mate::play_estimator_ply58`）で、被詰めろの手番・詰め手の
    /// 分類・一手巻き戻した決定点の安全手が揃うこと。
    /// 後手は G*7八 の一手詰めを持っているので、先手の手番はこの直前が
    /// 「最後に受けられた決定点」になる
    #[test]
    fn 被詰めろの手番と詰め手の分類と安全手を数える() {
        let decision = crate::mate::play_estimator_ply58();
        assert_eq!(decision.turn(), Color::Sente);
        // 相手番（後手が指す直前）の局面 = bot（先手）が被詰めろになっている手番
        let mut threatened = decision.clone();
        threatened.set_turn(Color::Gote);
        let positions = vec![decision.clone(), threatened];

        let mut mates_of = |p: &Position, _ply: usize| mate_moves_in_1_fast(p);
        let played = |_i: usize| None;
        let turns = threat_turns(&positions, Color::Sente, &mut mates_of, &played);
        assert_eq!(turns.len(), 1, "被詰めろの手番はこの1つ");
        let t = &turns[0];
        assert_eq!(t.idx, 1);
        assert_eq!(t.decision_idx, Some(0), "一手巻き戻した bot の決定点");
        assert!(
            t.mates.iter().any(|m| m.to_usi() == "G*7h"),
            "実戦の詰め手が列挙に入る"
        );
        assert_eq!(t.kind, MateKind::of(&t.mates).unwrap());
        assert!(t.avoidable(), "真実上の受けがある");
        // 受けの理論上限（真実を見た安全手）は「7八を埋める・玉が逃げる・
        // 支えの桂を取る」を含み、全合法手の一部でしかない
        let safe: HashSet<String> = t.safe.iter().map(ShogiMove::to_usi).collect();
        for usi in ["R*7h", "N*7h", "7i8i", "6g6f"] {
            assert!(safe.contains(usi), "安全手に {usi} が入るはず");
        }
        assert!(
            t.safe.len() < positions[0].legal_moves().len(),
            "危険な手もあるので全合法手が安全手にはならない"
        );
    }

    #[test]
    fn 連続した被詰めろ手番は1エピソードへ畳む() {
        let mk = |idx: usize| ThreatTurn {
            idx,
            mates: vec![],
            kind: MateKind::DropOnly,
            executed: None,
            decision_idx: None,
            safe: vec![],
        };
        // 10,12,14 は連続（相手番は2つおき）。18 は別エピソード
        let turns = vec![mk(10), mk(12), mk(14), mk(18)];
        let eps = fold_episodes(&turns);
        assert_eq!(eps.len(), 2);
        assert_eq!(eps[0].turns, vec![0, 1, 2]);
        assert_eq!(eps[1].turns, vec![3]);
    }

    #[test]
    fn 最後に受けられた手番はsafeのある最後の手番() {
        let mk = |idx: usize, safe: bool| ThreatTurn {
            idx,
            mates: vec![],
            kind: MateKind::BoardOnly,
            executed: None,
            decision_idx: Some(idx - 1),
            safe: if safe {
                vec![parse_usi("7g7f").unwrap()]
            } else {
                vec![]
            },
        };
        let turns = vec![mk(10, true), mk(12, true), mk(14, false)];
        let eps = fold_episodes(&turns);
        assert_eq!(eps.len(), 1);
        let last = eps[0].last_avoidable(&turns).expect("受けられた手番がある");
        assert_eq!(last.idx, 12, "安全手のある最後の手番");
    }

    #[test]
    fn 安全手は指した後に相手の一手詰めが消える手() {
        let pos = Position::initial();
        let safe = safe_moves_at(&pos);
        // 初期局面はどう指しても相手に一手詰めは無い = 全合法手が安全手
        assert_eq!(safe.len(), pos.legal_moves().len());
    }

    /// 記録の終局ペイロードから局面列・bot の決定点・詰み負け判定を作る
    #[test]
    fn 記録の真実を再生して決定点と終局を数える() {
        let end = end_of(&[
            ("7g7f", Color::Sente),
            ("3c3d", Color::Gote),
            ("8h2b+", Color::Sente),
            ("3a2b", Color::Gote),
        ]);
        let positions = replay_positions(&end).expect("再生できる");
        assert_eq!(positions.len(), 5);
        let mut mates_of = |p: &Position, _ply: usize| mate_moves_in_1_fast(p);
        let g = analyze_game(&end, Color::Sente, &mut mates_of).expect("解析できる");
        assert_eq!(g.bot_decisions, vec![0, 2, 4], "先手の決定点（最終局面も次の手番）");
        assert!(!g.bot_mated, "詰み負けしていない");
        assert!(g.turns.is_empty(), "この手順では被詰めろは無い");
        assert!(g.last_defense_point().is_none());
        // 壊れた棋譜（非合法手）は None
        let broken = end_of(&[("9i9a", Color::Sente)]);
        assert!(replay_positions(&broken).is_none());
        assert!(analyze_game(&broken, Color::Sente, &mut mates_of).is_none());
    }

    /// 対照の決定点は「危険でない bot の決定点」を手数の近い順に取る
    #[test]
    fn 対照の決定点は同じ手数帯から取る() {
        let end = end_of(&[
            ("7g7f", Color::Sente),
            ("3c3d", Color::Gote),
            ("2g2f", Color::Sente),
            ("8c8d", Color::Gote),
            ("2f2e", Color::Sente),
            ("8d8e", Color::Gote),
        ]);
        let mut mates_of = |p: &Position, _ply: usize| mate_moves_in_1_fast(p);
        let g = analyze_game(&end, Color::Sente, &mut mates_of).expect("解析できる");
        assert_eq!(g.bot_decisions, vec![0, 2, 4, 6]);
        let picked = control_points(&g, 4, 2, &HashSet::from([4]));
        assert_eq!(picked, vec![2, 6], "手数の近い順（除外した 4 は入らない）");
    }

    /// オフライン再スコアは w=0 で現行スコアに一致し、q>0 の手だけを沈める
    /// （issue #24 の教訓⑦: `gain` 側へ引いて `combine_score` を通す）
    #[test]
    fn 再スコアはw0で現行スコアに一致する() {
        let row = |usi: &str, gain: f64, q: f64| {
            let (p_legal, foul_cost, foul_probe, adjust) = (0.8, 6.0, 0.0, 0.25);
            MateRow {
                usi: usi.into(),
                det_score: crate::strategy::combine_score(gain, p_legal, foul_cost)
                    + foul_probe
                    + adjust,
                rank: 0,
                gain,
                p_legal,
                foul_cost,
                foul_probe,
                adjust_det: adjust,
                depth2: false,
                mate_risk: 0.0,
                mass: MateMass {
                    all: q,
                    strict: q,
                    strict_share: 1.0,
                },
                truth_legal: true,
                truth_allows_mate: Some(q > 0.5),
                truth_kind: None,
            }
        };
        let risky = row("2f2e", 3.0, 0.9);
        let safe = row("7g7f", 2.5, 0.0);
        for r in [&risky, &safe] {
            assert!((r.rescore(0.0, false) - r.det_score).abs() < 1e-12);
        }
        let rows = vec![risky.clone(), safe.clone()];
        assert_eq!(pick(&rows, 0.0, false).unwrap().usi, "2f2e", "現行は危険な手");
        // 危険量に十分な重みを掛ければ安全手が argmax になる
        let flip = min_w_to_flip(
            &rows,
            &HashSet::from(["7g7f".to_string()]),
            false,
            &[0.0, 0.5, 1.0, 2.0, 4.0],
        );
        assert_eq!(flip, Some(1.0));
        // 安全手（q=0）は w を上げても沈まない
        assert!((safe.rescore(30.0, false) - safe.det_score).abs() < 1e-12);
    }

    /// 再スコアの順序と argmax は同じ規約（同点は辞書順）で並ぶ
    #[test]
    fn 再スコアの順序の先頭はargmaxと一致する() {
        let row = |usi: &str, gain: f64, q: f64| MateRow {
            usi: usi.into(),
            det_score: 0.0,
            rank: 0,
            gain,
            p_legal: 1.0,
            foul_cost: 0.0,
            foul_probe: 0.0,
            adjust_det: 0.0,
            depth2: false,
            mate_risk: 0.0,
            mass: MateMass {
                all: q,
                strict: q,
                strict_share: 1.0,
            },
            truth_legal: true,
            truth_allows_mate: None,
            truth_kind: None,
        };
        let rows = vec![row("2f2e", 3.0, 1.0), row("7g7f", 2.0, 0.0), row("1g1f", 2.0, 0.0)];
        for w in [0.0, 1.0, 4.0] {
            let order = rescored_usis(&rows, w, false);
            assert_eq!(order.len(), rows.len());
            assert_eq!(order[0], pick(&rows, w, false).unwrap().usi);
        }
        // 同点（7g7f と 1g1f）は辞書順で割れる
        assert_eq!(rescored_usis(&rows, 4.0, false)[..2], ["1g1f", "7g7f"]);
    }

    /// 一手強制は非合法な候補で反則を積んで次点へ進む（審判と同じ規約）
    #[test]
    fn 強制手が非合法なら反則を積んで次点へ進む() {
        let pos = crate::mate::play_estimator_ply58();
        let logs = [ObservationLog::default(), ObservationLog::default()];
        // 9九の香が塞いでいるので 9八香打ちは非合法、7八飛打ちは合法
        let illegal = "L*9h".to_string();
        assert!(!pos.is_legal(&parse_usi(&illegal).unwrap()));
        let legal = "R*7h".to_string();
        assert!(pos.is_legal(&parse_usi(&legal).unwrap()));

        let f = force_move(&pos, &logs, [2, 3], Color::Sente, &[illegal, legal.clone()]);
        assert_eq!(f.played.as_deref(), Some(legal.as_str()));
        assert_eq!(f.forced_fouls, 1);
        assert!(!f.foul_limit);
        assert_eq!(f.fouls, [3, 3], "先手の反則だけが増える");
        // 両者のログに観測が入る（自分は反則と着手、相手は反則宣言と着手）
        assert_eq!(f.logs[0].events().len(), 2);
        assert_eq!(f.logs[1].events().len(), 2);

        // 反則上限に達したら受理せずに止まる
        let f = force_move(
            &pos,
            &logs,
            [crate::selfplay::MAX_FOULS - 1, 0],
            Color::Sente,
            &["L*9h".to_string(), legal],
        );
        assert!(f.foul_limit);
        assert_eq!(f.played, None);
    }

    #[test]
    fn 分類は排他的で_unionで畳める() {
        let d = parse_usi("G*5b").unwrap();
        let b = parse_usi("7g7f").unwrap();
        assert_eq!(MateKind::of(&[d.clone()]), Some(MateKind::DropOnly));
        assert_eq!(MateKind::of(&[b.clone()]), Some(MateKind::BoardOnly));
        assert_eq!(MateKind::of(&[d, b]), Some(MateKind::Both));
        assert_eq!(MateKind::of(&[]), None);
        assert_eq!(
            MateKind::DropOnly.merge(MateKind::BoardOnly),
            MateKind::Both
        );
        assert_eq!(MateKind::DropOnly.merge(MateKind::DropOnly), MateKind::DropOnly);
    }
}
