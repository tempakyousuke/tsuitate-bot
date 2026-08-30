//! 王手中の反則経済の**共有定義**（issue #31）。
//!
//! `bin/analyze`（P0-1 の型分類・P0-2 の較正・P0-3 の破滅の原因）と、以後の
//! 診断（P0-4 の w\* 監査・P0-5 の方策シミュレーション・P0-7 の影の価格）が
//! **同じ定義**で「王手中の手番」「手種」「手番内の順番」を数えるための場所。
//!
//! ここに置くのは runtime に一切入らない診断だけ。評価・推定の挙動は変えない
//! （mate_economy.rs と同じ位置づけ。別々に数えると較正の数字が意味を失う）。

use std::collections::{HashMap, HashSet};

use crate::board::make_usi_square;
use crate::check::CheckSolver;
use crate::model::GameModel;
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{ClockState, Color, FoulCounts, GameStatus, PlayerView, Role};
use crate::board::Coord;
use crate::shogi::{Position, ShogiMove, parse_usi};
use crate::strategy::{CandidateScore, candidate_moves};

/// 観測ログの復元から PlayerView 相当を作る（ソルバーの再現用）。
///
/// 診断はどれも「その決定点で bot が見ていた視界」を作り直すので、規約
/// （手番・持ち駒・反則数・**手数**の出どころ）を1箇所に閉じる。
///
/// **`move_number` は必ず決定点の手数を渡す**（`scenario_core::make_view` と
/// 同じ規約）: `CheckSolver` は既知の敵駒マスの鮮度を
/// `view.move_number − 8` で切るので、0 を渡すと窓が無制限になり、
/// 実戦では載らない古い幻の駒を base に載せた別物のソルバーになる。
pub fn view_from_model(model: &GameModel, in_check: bool, move_number: u32) -> PlayerView {
    PlayerView {
        game_id: "replay".into(),
        your_color: model.my_color(),
        your_pieces: model.my_pieces(),
        your_hand: model.my_hand(),
        turn: model.my_color(),
        move_number,
        clocks: ClockState { sente_ms: 0, gote_ms: 0, running: None, server_time: 0 },
        fouls: FoulCounts { you: model.my_fouls(), opponent: model.opponent_fouls() },
        you_in_check: in_check,
        opponent_in_check: false,
        status: GameStatus::Playing,
    }
}

/// 王手中の手種（**bot の視界での意図**で分ける）。
///
/// 「捕獲試み」は真実の駒配置ではなく `CheckSolver::captures_checker`
/// （王手駒仮説のマスへ玉以外で移動し、その仮説の下で解消する手）で判定する。
/// 真実で敵駒がいたかは別の列（`CheckAttempt::truth_enemy_at_to`）に持つ:
/// 空振りの捕獲試みも意図としては捕獲プローブなので、同じ束に入れないと
/// 「プローブの較正」が測れない。
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum CheckMoveKind {
    /// 玉の手（逃げ・玉での捕獲）
    King,
    /// 玉以外の盤上手で、王手駒仮説のマスを取りに行く手
    CheckerCapture,
    /// 玉以外のその他の盤上手（移動合駒・空振り）
    OtherBoard,
    /// 打ち（合駒打ち）
    Drop,
}

impl CheckMoveKind {
    pub const ALL: [CheckMoveKind; 4] = [
        CheckMoveKind::King,
        CheckMoveKind::CheckerCapture,
        CheckMoveKind::OtherBoard,
        CheckMoveKind::Drop,
    ];

    pub fn label(self) -> &'static str {
        match self {
            CheckMoveKind::King => "玉の手",
            CheckMoveKind::CheckerCapture => "捕獲試み",
            CheckMoveKind::OtherBoard => "移動合駒/その他",
            CheckMoveKind::Drop => "打ち",
        }
    }

    pub fn tag(self) -> &'static str {
        match self {
            CheckMoveKind::King => "king",
            CheckMoveKind::CheckerCapture => "checker_capture",
            CheckMoveKind::OtherBoard => "other_board",
            CheckMoveKind::Drop => "drop",
        }
    }

    /// 型分類（P0-1 の表）が使う粗い束
    pub fn coarse(self) -> CoarseKind {
        match self {
            CheckMoveKind::King => CoarseKind::King,
            CheckMoveKind::CheckerCapture | CheckMoveKind::OtherBoard => CoarseKind::NonKingBoard,
            CheckMoveKind::Drop => CoarseKind::Drop,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CoarseKind {
    King,
    NonKingBoard,
    Drop,
}

/// 手番内の1試行（反則 or 受理）
#[derive(Clone, Debug)]
pub struct CheckAttempt {
    pub usi: String,
    pub mv: ShogiMove,
    pub kind: CheckMoveKind,
    /// 手番内の順番（0 = その手番の最初の試行）
    pub order: usize,
    pub was_legal: bool,
    /// 記録に残っていた選択時の p_legal（`chose` の debug。定跡手などは None）
    pub p_legal: Option<f64>,
    /// その時点の CheckSolver 単体の解消確率
    pub solver_p: f64,
    /// 真実: 着手先に敵駒がいたか / それが王手駒だったか
    pub truth_enemy_at_to: bool,
    pub truth_checker_at_to: bool,
}

/// 王手中の bot の手番1つ（反則0の手番も含む）
#[derive(Clone, Debug)]
pub struct CheckTurn {
    pub move_number: u32,
    /// 決定点の真実局面（positions のインデックス）
    pub position_index: usize,
    /// 手番開始時の bot の累計反則（残り = 10 − これ）
    pub fouls_before: u32,
    pub attempts: Vec<CheckAttempt>,
    /// 受理された試行（`attempts` のインデックス）。その手番で終局したら None
    pub accepted: Option<usize>,
    /// 受理された手が**手番開始時点の真実局面でも合法**だったか。
    ///
    /// 反則は手番を変えないので真実局面は手番中ずっと同じ = これは常に true。
    /// **整合性検査**として残す（false が出るなら決定点と真実局面の
    /// 手数対応がずれている）。「最初から指せた手に落ち着いた」の実質的な
    /// 中身は `legal_candidates_at_entry` と `best_legal_rank_at_entry` の方
    pub accepted_legal_at_entry: Option<bool>,
    /// 開始時点で**真に合法だった候補手の本数**（＝この手番に用意されていた
    /// 出口の数）。1本しか無い手番の確かめは価格では避けられない
    pub legal_candidates_at_entry: usize,
    /// 真に合法な候補のうち**ソルバー p が最上位のもの**の順位（1始まり）。
    /// 1 なら「ソルバー最善が最初から合法だったのに反則してから到達した」
    pub best_legal_rank_at_entry: Option<usize>,
    /// 開始時に `CheckSolver` を作れたか（両王手で仮説が全滅すると作れない）。
    /// false の手番は**ソルバー順位が付かない**ので、集計では「判定不能」に
    /// 分けること（0 として数えると率が下振れする）
    pub solver_at_entry: bool,
    /// 受理された手の**開始時点のソルバー p 順位**（1始まり）。
    ///
    /// エンジンの順位ではない（それは粒子が要るので P0-4 の rank_probe の領分）。
    /// 「情報が生産的だった」のか「単に次点へ進んだだけ」かの粗い代理。
    pub accepted_rank_at_entry: Option<usize>,
    /// 開始時点の候補手数とソルバー最善 p
    pub candidates_at_entry: usize,
    pub p_max_at_entry: f64,
    /// 開始時点の王手駒仮説の本数
    pub hypotheses_at_entry: usize,
    /// **真の王手駒（マス・駒種とも一致）に載っている仮説の重みシェア**。
    /// 「知らない」のか「知っていて払う」のかを分ける量: シェアが低ければ
    /// 仮説が希釈されていて、正しい捕獲の p まで薄まる（kakutori の構図）
    pub true_hyp_share: Option<f64>,
    /// 仮説重みの正規化エントロピー（0 = 1点に確信、1 = 一様）
    pub hyp_entropy: Option<f64>,
}

impl CheckTurn {
    pub fn fouls(&self) -> usize {
        self.attempts.iter().filter(|a| !a.was_legal).count()
    }

    pub fn first_foul(&self) -> Option<&CheckAttempt> {
        self.attempts.iter().find(|a| !a.was_legal)
    }

    pub fn accepted_attempt(&self) -> Option<&CheckAttempt> {
        self.accepted.and_then(|i| self.attempts.get(i))
    }

    /// 手番内の反則列（手種の並び）。P0-1 の列⑤
    pub fn sequence(&self) -> String {
        self.attempts
            .iter()
            .map(|a| {
                format!(
                    "{}{}{}",
                    a.usi,
                    match a.kind {
                        CheckMoveKind::King => "[玉]",
                        CheckMoveKind::CheckerCapture => "[捕]",
                        CheckMoveKind::OtherBoard => "[移]",
                        CheckMoveKind::Drop => "[打]",
                    },
                    if a.was_legal { "○" } else { "×" }
                )
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn turn_type(&self) -> TurnType {
        let Some(first) = self.first_foul() else {
            return TurnType::NoFoul;
        };
        let Some(acc) = self.accepted_attempt() else {
            return TurnType::Unfinished;
        };
        match (first.kind.coarse(), acc.kind.coarse()) {
            (CoarseKind::Drop, _) => TurnType::DropOrOther,
            (CoarseKind::NonKingBoard, CoarseKind::King) => TurnType::NonKingThenKing,
            (CoarseKind::NonKingBoard, _) => TurnType::NonKingThenNonKing,
            (CoarseKind::King, CoarseKind::King) => TurnType::KingThenKing,
            (CoarseKind::King, _) => TurnType::KingThenNonKing,
        }
    }
}

/// P0-1 の型（最初の反則の手種 → 受理された手の手種）
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum TurnType {
    /// 反則なしで受理（分母。**新しいプローブを足す害**の estimand でもある）
    NoFoul,
    /// 受理手を持たない（その手番で終局した）
    Unfinished,
    /// 玉以外の盤上手で反則 → 玉逃げで受理（最大のまとまり）
    NonKingThenKing,
    /// 玉以外の盤上手で反則 → 玉以外で受理（kakutori 型）
    NonKingThenNonKing,
    /// 玉逃げで反則 → 玉逃げで受理（逃げ先の選別）
    KingThenKing,
    /// 玉逃げで反則 → 玉以外で受理
    KingThenNonKing,
    /// 打ちで反則（＋その他）
    DropOrOther,
}

impl TurnType {
    pub const ALL: [TurnType; 7] = [
        TurnType::NonKingThenKing,
        TurnType::NonKingThenNonKing,
        TurnType::KingThenKing,
        TurnType::KingThenNonKing,
        TurnType::DropOrOther,
        TurnType::Unfinished,
        TurnType::NoFoul,
    ];

    pub fn label(self) -> &'static str {
        match self {
            TurnType::NoFoul => "反則0で受理",
            TurnType::Unfinished => "受理手なし（その手番で終局）",
            TurnType::NonKingThenKing => "玉以外の盤上手で反則 → 玉の手で受理",
            TurnType::NonKingThenNonKing => "玉以外の盤上手で反則 → 玉以外で受理",
            TurnType::KingThenKing => "玉の手で反則 → 玉の手で受理",
            TurnType::KingThenNonKing => "玉の手で反則 → 玉以外で受理",
            TurnType::DropOrOther => "打ちで反則",
        }
    }
}

/// 真の王手駒（複数 = 両王手ならその全部）
pub fn true_checkers(truth: &Position, bot: Color) -> Vec<(Coord, Role)> {
    let Some(king) = truth.king_square(bot) else {
        return vec![];
    };
    truth
        .pieces()
        .filter(|(sq, p)| p.color != bot && truth.attacks(*sq, king))
        .map(|(sq, p)| (sq, p.role))
        .collect()
}

/// 記録（観測列＋真実の局面列）から**王手中の bot の手番**を全部取り出す。
///
/// 反則0の手番も返す（P0-1 の分母・P0-6 の非劣性 estimand がそこを使う）。
///
/// `p_legal_by` は `(決定点の move_number, USI) → 記録された p_legal`。
pub fn check_turns(
    observations: &[Observation],
    positions: &[Position],
    bot: Color,
    p_legal_by: &HashMap<(u32, String), f64>,
) -> Vec<CheckTurn> {
    let mut turns = vec![];
    for group in decision_groups(observations) {
        let move_number = group[0].0;
        let idx = (move_number as usize).saturating_sub(1);
        let Some(truth) = positions.get(idx) else { continue };
        if !truth.in_check(bot) {
            continue;
        }
        let entry_obs = group[0].1;
        let mut log = ObservationLog::default();
        for prev in &observations[..entry_obs] {
            log.record(prev.clone());
        }
        let model = GameModel::from_log(bot, &log);
        let fouls_before = model.my_fouls();
        let view = view_from_model(&model, true, truth.move_number());
        let checkers: HashSet<Coord> =
            true_checkers(truth, bot).into_iter().map(|(s, _)| s).collect();

        // 開始時点（反則0）のソルバー順位。エンジンの順位ではないので
        // 「上位なら単に次点へ進んだだけ」の粗い代理としてのみ使う
        let empty: HashSet<String> = HashSet::new();
        let entry_candidates = candidate_moves(&view, &empty);
        let mut entry_rank: Vec<(String, f64)> = vec![];
        let mut p_max_at_entry = 0.0f64;
        let mut hypotheses_at_entry = 0;
        let mut solver_at_entry = false;
        let mut true_hyp_share = None;
        let mut hyp_entropy = None;
        if let Some(mut solver) = CheckSolver::new(&view, &[], &[], &log) {
            solver_at_entry = true;
            for (usi, mv) in &entry_candidates {
                let p = solver.resolve_probability(mv);
                p_max_at_entry = p_max_at_entry.max(p);
                entry_rank.push((usi.clone(), p));
            }
            hypotheses_at_entry = solver.hypotheses_debug().len();
            let (share, entropy) = hypothesis_stats(Some(&solver), truth, bot);
            true_hyp_share = share;
            hyp_entropy = entropy;
        }
        // 同点は USI の辞書順で割る（安定ソートに任せると生成順が順位に残る）
        entry_rank.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let mut attempts: Vec<CheckAttempt> = vec![];
        let mut accepted = None;
        let mut fouls_so_far: Vec<ShogiMove> = vec![];
        for (order, (_, _, usi, was_legal)) in group.iter().enumerate() {
            let Some(mv) = parse_usi(usi) else { continue };
            let mut solver = CheckSolver::new(&view, &[], &fouls_so_far, &log);
            let solver_p = solver.as_mut().map_or(0.5, |s| s.resolve_probability(&mv));
            let kind = classify_move_kind(&mv, &view, solver.as_mut());
            let to = match mv {
                ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
            };
            let truth_enemy_at_to = truth.piece_at(to).is_some_and(|p| p.color != bot);
            attempts.push(CheckAttempt {
                usi: usi.clone(),
                mv,
                kind,
                order,
                was_legal: *was_legal,
                p_legal: p_legal_by.get(&(move_number, usi.clone())).copied(),
                solver_p,
                truth_enemy_at_to,
                truth_checker_at_to: checkers.contains(&to),
            });
            if *was_legal {
                accepted = Some(attempts.len() - 1);
            } else {
                fouls_so_far.push(mv);
            }
        }

        let accepted_legal_at_entry = accepted.map(|i| truth.is_legal(&attempts[i].mv));
        let legal_candidates_at_entry =
            entry_candidates.iter().filter(|(_, mv)| truth.is_legal(mv)).count();
        let best_legal_rank_at_entry = entry_rank
            .iter()
            .position(|(u, _)| {
                entry_candidates
                    .iter()
                    .any(|(cu, mv)| cu == u && truth.is_legal(mv))
            })
            .map(|r| r + 1);
        let accepted_rank_at_entry = accepted.and_then(|i| {
            entry_rank
                .iter()
                .position(|(u, _)| *u == attempts[i].usi)
                .map(|r| r + 1)
        });

        turns.push(CheckTurn {
            move_number,
            position_index: idx,
            fouls_before,
            attempts,
            accepted,
            accepted_legal_at_entry,
            legal_candidates_at_entry,
            best_legal_rank_at_entry,
            solver_at_entry,
            accepted_rank_at_entry,
            candidates_at_entry: entry_candidates.len(),
            p_max_at_entry,
            hypotheses_at_entry,
            true_hyp_share,
            hyp_entropy,
        });
    }
    turns
}

/// 観測列を bot の**決定点**（1手番 = 反則列 + 受理手）ごとに束ねる。
///
/// 手数の対応の規約はここ1箇所に閉じる: 反則は手番を変えないので
/// `MyFoul` の `move_number` はその決定点の手数そのまま、`MyMove` は
/// **適用後**の値なので 1 引く（selfplay.rs / client.rs の moveNumber 規約）。
/// 戻り値の各要素は `(決定点の move_number, 観測列のインデックス, USI, 受理されたか)`。
pub fn decision_groups(observations: &[Observation]) -> Vec<Vec<(u32, usize, String, bool)>> {
    let mut raw: Vec<(u32, usize, String, bool)> = vec![];
    for (i, obs) in observations.iter().enumerate() {
        match obs {
            Observation::MyFoul { move_number, usi } => {
                raw.push((*move_number, i, usi.clone(), false))
            }
            Observation::MyMove { move_number, usi, .. } => {
                raw.push((move_number.saturating_sub(1), i, usi.clone(), true))
            }
            _ => {}
        }
    }
    let mut out: Vec<Vec<(u32, usize, String, bool)>> = vec![];
    for item in raw {
        match out.last_mut() {
            // 受理手の後は必ず新しい手番（同じ手数の手番が2つ続くことはない）
            Some(g) if g[0].0 == item.0 && !g.last().is_some_and(|l| l.3) => g.push(item),
            _ => out.push(vec![item]),
        }
    }
    out
}

/// 手種の判定。捕獲試みかどうかは**ソルバーの仮説**で決める（bot の意図）。
///
/// P0-4 の監査（`bin/check_probe`）も同じ規約で型を分けるので公開する。
pub fn classify_move_kind(
    mv: &ShogiMove,
    view: &PlayerView,
    solver: Option<&mut CheckSolver>,
) -> CheckMoveKind {
    let king = view
        .your_pieces
        .iter()
        .find(|p| p.role == Role::King)
        .and_then(|p| crate::board::parse_usi_square(&p.square));
    match *mv {
        ShogiMove::Drop { .. } => CheckMoveKind::Drop,
        ShogiMove::Board { from, .. } if Some(from) == king => CheckMoveKind::King,
        ShogiMove::Board { .. } => {
            if solver.is_some_and(|s| s.captures_checker(mv)) {
                CheckMoveKind::CheckerCapture
            } else {
                CheckMoveKind::OtherBoard
            }
        }
    }
}

/// 王手駒仮説の質: `(真の王手駒（マス・駒種とも一致）に載っている重みシェア,
/// 重みの正規化エントロピー)`。
///
/// **「知らない（希釈）」と「知っていて払う」を分ける量**。シェアが低く
/// エントロピーが 1 に近いほど、仮説集合は「どの駒が王手しているか」を
/// ほとんど知らない（正しい捕獲の p まで薄まる = kakutori の構図）。
pub fn hypothesis_stats(
    solver: Option<&CheckSolver>,
    truth: &Position,
    bot: Color,
) -> (Option<f64>, Option<f64>) {
    let Some(solver) = solver else {
        return (None, None);
    };
    hypothesis_share(&solver.hypotheses_debug(), &true_checkers(truth, bot))
}

/// [`hypothesis_stats`] の本体（仮説リストと真の王手駒から直に計算する版）。
///
/// issue #36 の P0-1 は runtime と同じ投票つきの仮説を別経路で作るので、
/// **定義を1か所にしておかないと 0.035 の水準が経路ごとに食い違う**。
pub fn hypothesis_share(
    hyps: &[(Coord, Role, f64)],
    checkers: &[(Coord, Role)],
) -> (Option<f64>, Option<f64>) {
    let total: f64 = hyps.iter().map(|(_, _, w)| w.max(0.0)).sum();
    if total <= 0.0 {
        return (None, None);
    }
    let hit: f64 = hyps
        .iter()
        .filter(|(hs, hr, _)| checkers.iter().any(|(s, r)| s == hs && r == hr))
        .map(|(_, _, w)| w.max(0.0))
        .sum();
    let h: f64 = hyps
        .iter()
        .map(|(_, _, w)| w.max(0.0) / total)
        .filter(|q| *q > 0.0)
        .map(|q| -q * q.ln())
        .sum();
    // 一様分布を 1 に正規化する（仮説の本数は局面ごとに違う）
    let max_h = (hyps.len().max(2) as f64).ln();
    (Some(hit / total), Some(h / max_h))
}

/// 反則の原因（**真実**から見た理由）。P0-3 の破滅手番の内訳に使う
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum CheckFoulReason {
    /// 王手駒がまだ玉を攻めている（仮説の駒違い・遮断できていない）
    CheckerRemains,
    /// 王手駒は消えたが、そのマスが真実では他の敵駒に守られていた（支え駒）。
    /// `legal_under` が仮説の王手駒1枚しか置かない盲点そのもの
    DefendedCapture,
    /// 玉の行き先・自駒の移動が別の敵駒の利きに入った（見えない利き・ピン）
    HiddenAttack,
    /// 経路が塞がれていた（擬似合法ですらない）
    Blocked,
    /// 打ちマスに駒があった
    DropOccupied,
    Other,
}

impl CheckFoulReason {
    pub fn label(self) -> &'static str {
        match self {
            CheckFoulReason::CheckerRemains => "王手駒が残る（仮説違い）",
            CheckFoulReason::DefendedCapture => "捕獲先が支えられていた",
            CheckFoulReason::HiddenAttack => "別の敵駒の利き（見えない利き/ピン）",
            CheckFoulReason::Blocked => "経路が塞がれていた",
            CheckFoulReason::DropOccupied => "打ちマスに駒",
            CheckFoulReason::Other => "その他",
        }
    }
}

/// 王手中の反則1つの原因を真実から分類する
pub fn classify_check_foul(truth: &Position, bot: Color, mv: &ShogiMove) -> CheckFoulReason {
    if !truth.is_pseudo_legal(mv) {
        return match mv {
            ShogiMove::Board { .. } => CheckFoulReason::Blocked,
            ShogiMove::Drop { .. } => CheckFoulReason::DropOccupied,
        };
    }
    let before: HashSet<Coord> = true_checkers(truth, bot).into_iter().map(|(s, _)| s).collect();
    let mut probe = truth.clone();
    probe.play_unchecked(mv);
    if !probe.in_check(bot) {
        return CheckFoulReason::Other; // 真に合法（= 反則でない）はここへ来ない
    }
    let after: Vec<(Coord, Role)> = true_checkers(&probe, bot);
    if after.iter().any(|(sq, _)| before.contains(sq)) {
        return CheckFoulReason::CheckerRemains;
    }
    // 王手駒は消えたのに王手が残る = 別の駒。取りに行った先が守られていたのか、
    // 逃げ/移動先が別の利きに入ったのかを分ける
    let to = match *mv {
        ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
    };
    if before.contains(&to) && after.iter().any(|(sq, _)| probe.attacks(*sq, to)) {
        CheckFoulReason::DefendedCapture
    } else {
        CheckFoulReason::HiddenAttack
    }
}

/// **手番開始時（反則0）の状態**を、その手番の反則を消化した後の状態から作る。
///
/// `truth_replay::for_each_decision_full` が渡すのは**その手番の反則を全部
/// 食った後**の状態なので、両者のログ末尾から反則の観測を `fouls_this_turn`
/// 本だけ落とす（手番側は `MyFoul`・相手側は `OpponentFoul`。
/// `add_foul_obs` が対で積む）。反則は局面を変えないので `pos` はそのまま。
///
/// 落とす末尾が本当に反則の観測かを確かめてから落とす（規約が変わったら
/// `None` を返して止める）。P0-4（`bin/check_probe`）と P0-5
/// （`bin/check_policy`）が同じ復元を使う。
pub fn entry_replayed(
    post: &crate::scenario_core::Replayed,
    side: Color,
    fouls_this_turn: u32,
) -> Option<crate::scenario_core::Replayed> {
    use crate::scenario_core::side_idx;
    let n = fouls_this_turn as usize;
    let mut logs = [ObservationLog::default(), ObservationLog::default()];
    for (idx, log) in logs.iter_mut().enumerate() {
        let events = post.logs[idx].events();
        let keep = events.len().checked_sub(n)?;
        for e in &events[keep..] {
            let ok = if idx == side_idx(side) {
                matches!(e, Observation::MyFoul { .. })
            } else {
                matches!(e, Observation::OpponentFoul { .. })
            };
            if !ok {
                return None;
            }
        }
        for e in &events[..keep] {
            log.record(e.clone());
        }
    }
    let mut fouls = post.fouls;
    fouls[side_idx(side)] = fouls[side_idx(side)].checked_sub(fouls_this_turn)?;
    Some(crate::scenario_core::Replayed {
        pos: post.pos.clone(),
        logs,
        fouls,
        plies: post.plies,
        injected_fouls: vec![],
        oracle: None,
    })
}

/// 反則コストを動かしたときの並べ替えを再計算するための最小の入力
/// （issue #31 の P0-4 の k\* 監査と、P1-α の再計算が**同じ式**を使う）。
///
/// `score = min(p·G, G) − (1−p)·c + foul_probe + adjust` なので、コストを
/// `c → c_k` へ変えた効果は `−(1−p)(c_k − c)` の付け替えで厳密に出る
/// （gain も p_legal も再計算しない = α が変えるのは価格だけ）。
#[derive(Clone, Debug)]
pub struct PricedMove {
    pub usi: String,
    pub p_legal: f64,
    /// **乱数を除いた**現行スコア（`CandidateScore::score − tiebreak`）
    pub det_score: f64,
    /// その決定で実際に使われた反則コスト（床適用後）
    pub foul_cost: f64,
    pub is_king: bool,
}

impl PricedMove {
    /// 反則コストを `cost` に置き換えたときの決定的スコア
    pub fn score_at(&self, cost: f64) -> f64 {
        self.det_score - (1.0 - self.p_legal) * (cost - self.foul_cost)
    }
}

/// `ranking` を価格再計算の形へ落とす。順位は**乱数を除いたスコア**で付け直し、
/// 同点は USI の辞書順で割る（安定ソートに任せると生成順が順位に残る。
/// issue #24 の教訓②）。
pub fn priced_moves(ranking: &[CandidateScore], king: Option<Coord>) -> Vec<PricedMove> {
    let mut out: Vec<PricedMove> = ranking
        .iter()
        .map(|c| PricedMove {
            usi: c.usi.clone(),
            p_legal: c.p_legal,
            det_score: c.score - c.tiebreak,
            foul_cost: c.foul_cost,
            is_king: is_king_move(&c.usi, king),
        })
        .collect();
    out.sort_by(|a, b| b.det_score.total_cmp(&a.det_score).then_with(|| a.usi.cmp(&b.usi)));
    out
}

/// その USI が玉の移動か（打ちと他の駒の移動は false）
pub fn is_king_move(usi: &str, king: Option<Coord>) -> bool {
    match parse_usi(usi) {
        Some(ShogiMove::Board { from, .. }) => Some(from) == king,
        _ => false,
    }
}

/// コスト `cost` での首位（同点は USI の辞書順）
pub fn argmax_at(moves: &[PricedMove], cost: f64) -> Option<&PricedMove> {
    moves.iter().min_by(|a, b| {
        b.score_at(cost)
            .total_cmp(&a.score_at(cost))
            .then_with(|| a.usi.cmp(&b.usi))
    })
}

/// **k\***: 玉の手が首位になる最小の反則コスト倍率（`kmax` までに反転しなければ None）。
///
/// 価格は玉の手の (1−p) 側にも効くので「score 差 ÷ プローブの傾斜」では出ない。
/// P1-α と同じ `check_cost = max(k × base, 床)` で**全候補を再計算して交点を探す**
/// （床の折れをそのまま扱えるように細かい格子で argmax を追う）。
pub fn k_star(moves: &[PricedMove], base: f64, floor: f64, kmax: f64) -> Option<f64> {
    const STEP: f64 = 0.02;
    // **格子は整数から作って小数第2位へ丸める**。`k += STEP` の累積だと
    // k=3.00 が 3.0000000000000018 になり、「k\* ≤ 3」の判定が
    // その場の集計（生の値）と CSV 経由の集計（`{:.2}` で丸めた値）で
    // 食い違う（門の割合が経路によって変わる）
    let steps = ((kmax - 1.0) / STEP).floor() as i64;
    for i in 0..=steps.max(0) {
        let k = ((1.0 + i as f64 * STEP) * 100.0).round() / 100.0;
        if k > kmax {
            break;
        }
        if argmax_at(moves, (k * base).max(floor))?.is_king {
            return Some(k);
        }
    }
    None
}

/// 元対局単位の cluster bootstrap（percentile CI）。
///
/// 統計単位は**元対局**（seed も手番も独立標本として数えない）。
/// `clusters` は「局ごとの (分子の和, 分母の和)」で、比の CI を出す。
/// 決定論的な線形合同法なので、同じ入力なら同じ CI になる。
pub fn cluster_ratio_ci(clusters: &[(f64, f64)], alpha: f64, seed: u64) -> (f64, f64) {
    if clusters.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mut state = seed | 1;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let reps = 2000;
    let mut draws: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let (mut num, mut den) = (0.0, 0.0);
        for _ in 0..clusters.len() {
            let (n, d) = clusters[next() % clusters.len()];
            num += n;
            den += d;
        }
        draws.push(if den > 0.0 { num / den } else { f64::NAN });
    }
    draws.retain(|v| v.is_finite());
    if draws.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    draws.sort_by(|a, b| a.total_cmp(b));
    let lo = ((draws.len() as f64) * (alpha / 2.0)) as usize;
    let hi = (((draws.len() as f64) * (1.0 - alpha / 2.0)) as usize).min(draws.len() - 1);
    (draws[lo], draws[hi])
}

/// マスの表示（診断出力用）
pub fn sq(c: Coord) -> String {
    make_usi_square(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 型分類は最初の反則と受理手の粗い束で決まる() {
        let mk = |kind: CheckMoveKind, was_legal: bool, order: usize| CheckAttempt {
            usi: "7g7f".into(),
            mv: parse_usi("7g7f").unwrap(),
            kind,
            order,
            was_legal,
            p_legal: None,
            solver_p: 0.5,
            truth_enemy_at_to: false,
            truth_checker_at_to: false,
        };
        let turn = |attempts: Vec<CheckAttempt>| {
            let accepted = attempts.iter().position(|a| a.was_legal);
            CheckTurn {
                move_number: 1,
                position_index: 0,
                fouls_before: 0,
                attempts,
                accepted,
                accepted_legal_at_entry: None,
                legal_candidates_at_entry: 0,
                best_legal_rank_at_entry: None,
                solver_at_entry: true,
                accepted_rank_at_entry: None,
                candidates_at_entry: 0,
                p_max_at_entry: 0.0,
                hypotheses_at_entry: 0,
                true_hyp_share: None,
                hyp_entropy: None,
            }
        };
        assert_eq!(turn(vec![mk(CheckMoveKind::King, true, 0)]).turn_type(), TurnType::NoFoul);
        assert_eq!(
            turn(vec![
                mk(CheckMoveKind::CheckerCapture, false, 0),
                mk(CheckMoveKind::King, true, 1)
            ])
            .turn_type(),
            TurnType::NonKingThenKing
        );
        // 移動合駒も「玉以外の盤上手」の束に入る
        assert_eq!(
            turn(vec![
                mk(CheckMoveKind::OtherBoard, false, 0),
                mk(CheckMoveKind::CheckerCapture, true, 1)
            ])
            .turn_type(),
            TurnType::NonKingThenNonKing
        );
        assert_eq!(
            turn(vec![mk(CheckMoveKind::King, false, 0), mk(CheckMoveKind::King, true, 1)])
                .turn_type(),
            TurnType::KingThenKing
        );
        // 受理手が無い（その手番で終局）は別枠。反則0の手番と混ぜない
        assert_eq!(
            turn(vec![mk(CheckMoveKind::King, false, 0)]).turn_type(),
            TurnType::Unfinished
        );
    }


    fn obs_foul(mn: u32, usi: &str) -> Observation {
        Observation::MyFoul { move_number: mn, usi: usi.into() }
    }
    fn obs_move(mn: u32, usi: &str) -> Observation {
        Observation::MyMove { move_number: mn, usi: usi.into(), captured: None }
    }

    #[test]
    fn 決定点の束ね方と手数の対応() {
        let obs = vec![
            obs_move(2, "7g7f"),           // 1手目の決定 → 決定点 1
            Observation::OpponentMoved { move_number: 3, captured_my_piece_at: None },
            obs_foul(3, "5i5h"),           // 3手目の決定（反則は手数そのまま）
            obs_foul(3, "5i4h"),
            obs_move(4, "5i6h"),           // 同じ3手目の決定の受理手
            Observation::OpponentMoved { move_number: 5, captured_my_piece_at: None },
            obs_move(6, "6h7h"),           // 5手目の決定
        ];
        let groups = decision_groups(&obs);
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].iter().map(|g| g.0).collect::<Vec<_>>(), vec![1]);
        // 反則2回＋受理1回が1つの手番に束ねられる
        assert_eq!(groups[1].len(), 3);
        assert_eq!(groups[1].iter().map(|g| g.0).collect::<Vec<_>>(), vec![3, 3, 3]);
        assert!(groups[1].iter().map(|g| g.3).eq([false, false, true]));
        assert_eq!(groups[2][0].0, 5);
    }

    #[test]
    fn 受理手のあとに同じ手数の手番が続いても分ける() {
        // 反則が無い連続手番（相手手の観測が欠けた記録）でも、受理手で
        // 手番を閉じるので2つの決定が1つに混ざらない
        let obs = vec![obs_move(2, "7g7f"), obs_move(2, "2g2f")];
        let groups = decision_groups(&obs);
        assert_eq!(groups.len(), 2);
    }

    fn piece(color: Color, role: Role) -> crate::shogi::Piece {
        crate::shogi::Piece { color, role }
    }
    fn at(usi: &str) -> Coord {
        crate::board::parse_usi_square(usi).unwrap()
    }

    #[test]
    fn 王手中の反則の原因を真実から分類する() {
        // 先手玉5i / 後手飛5d の筋王手。5h は筋上なので解消しない
        let mut pos = Position::empty(Color::Sente);
        pos.set(at("5i"), Some(piece(Color::Sente, Role::King)));
        pos.set(at("5d"), Some(piece(Color::Gote, Role::Rook)));
        pos.set(at("8a"), Some(piece(Color::Gote, Role::King)));
        assert!(pos.in_check(Color::Sente));
        assert_eq!(
            classify_check_foul(&pos, Color::Sente, &parse_usi("5i5h").unwrap()),
            CheckFoulReason::CheckerRemains
        );

        // 支え駒: 後手金5h が王手、その後ろの後手飛5a が 5h を守る。
        // 玉で取ると（仮説の王手駒1枚だけの盤面では合法なのに）飛に取られる
        let mut sup = Position::empty(Color::Sente);
        sup.set(at("5i"), Some(piece(Color::Sente, Role::King)));
        sup.set(at("5h"), Some(piece(Color::Gote, Role::Gold)));
        sup.set(at("5a"), Some(piece(Color::Gote, Role::Rook)));
        sup.set(at("8a"), Some(piece(Color::Gote, Role::King)));
        assert!(sup.in_check(Color::Sente));
        assert_eq!(
            classify_check_foul(&sup, Color::Sente, &parse_usi("5i5h").unwrap()),
            CheckFoulReason::DefendedCapture
        );

        // 見えない利き: 飛5d の王手から 4i へ逃げると 4a の香に取られる
        let mut hid = Position::empty(Color::Sente);
        hid.set(at("5i"), Some(piece(Color::Sente, Role::King)));
        hid.set(at("5d"), Some(piece(Color::Gote, Role::Rook)));
        hid.set(at("4a"), Some(piece(Color::Gote, Role::Lance)));
        hid.set(at("8a"), Some(piece(Color::Gote, Role::King)));
        assert_eq!(
            classify_check_foul(&hid, Color::Sente, &parse_usi("5i4i").unwrap()),
            CheckFoulReason::HiddenAttack
        );
    }

    fn priced(usi: &str, p: f64, det: f64, king: bool) -> PricedMove {
        PricedMove { usi: usi.into(), p_legal: p, det_score: det, foul_cost: 1.0, is_king: king }
    }

    #[test]
    fn k_starは価格が両側に効くことを織り込む() {
        // プローブ: gain が高いが合法確率 0.4 / 玉の手: 低いが 0.9。
        // 価格を上げるとプローブのほうが速く沈むので、どこかで逆転する
        let moves = vec![priced("5f5e", 0.4, 3.0, false), priced("5i4h", 0.9, 2.0, true)];
        assert!(!argmax_at(&moves, 1.0).unwrap().is_king, "現行価格ではプローブが首位");
        let k = k_star(&moves, 1.0, 1.0, 30.0).expect("有限の k で反転する");
        // 交点は (3.0−2.0)/((1−0.4)−(1−0.9)) = 2.0 → コスト 1.0 の 3 倍
        assert!((k - 3.0).abs() < 0.05, "k* = {k}");
        assert!(argmax_at(&moves, k).unwrap().is_king);
        assert!(!argmax_at(&moves, (k - 0.1).max(1.0)).unwrap().is_king);
    }

    #[test]
    fn k_starは小数第2位で丸めた格子を返す() {
        // `k += 0.02` の累積だと格子点が 3.0000000000000018 のようにずれ、
        // 「k* ≤ 3」の判定がその場の集計（生の値）と CSV 経由の集計
        // （`{:.2}` で丸めた値）で食い違う。丸めた格子なら経路によらず一致する
        let moves = vec![priced("5f5e", 0.4, 3.0, false), priced("5i4h", 0.9, 2.0, true)];
        let k = k_star(&moves, 1.0, 1.0, 30.0).unwrap();
        // 交点はコスト 3.0（= k 3.0）だが、**同点は USI の辞書順**で割るので
        // ちょうど交点では玉の手 5i4h が 5f5e に負ける。玉の手が strict に
        // 首位を取る最初の格子点は 3.02
        assert_eq!(k, 3.02);
        assert_eq!(format!("{k:.2}").parse::<f64>().unwrap(), k, "CSV の round-trip と一致");
        // 格子点は全部 2 桁で表せる（累積誤差が乗らない）
        for kmax in [1.5, 4.0, 30.0] {
            let far = vec![priced("5f5e", 0.4, 100.0, false), priced("5i4h", 0.9, 0.0, true)];
            if let Some(k) = k_star(&far, 1.0, 1.0, kmax) {
                assert_eq!(format!("{k:.2}").parse::<f64>().unwrap(), k, "kmax={kmax}");
            }
        }
    }

    #[test]
    fn 合法確率が同じか低い玉の手は価格では浮かない() {
        // 玉の手のほうが p が低ければ、価格を上げるほど差は開く（分母が 0 以下）
        let moves = vec![priced("5f5e", 0.9, 3.0, false), priced("5i4h", 0.4, 2.0, true)];
        assert_eq!(k_star(&moves, 1.0, 1.0, 30.0), None);
    }

    #[test]
    fn 床は倍率より優先される() {
        // 残り1回のガード床（60）が効いている決定では、k=1 でもコストは床
        let moves = vec![priced("5f5e", 0.4, 3.0, false), priced("5i4h", 0.9, 2.0, true)];
        // 床 60 なら k=1 の時点で既に玉の手が首位（= 反転に価格の上乗せは要らない）
        assert!(argmax_at(&moves, 60.0).unwrap().is_king);
        assert_eq!(k_star(&moves, 1.0, 60.0, 30.0), Some(1.0));
    }

    #[test]
    fn cluster_ratio_ci_は決定論的で点推定を挟む() {
        let clusters: Vec<(f64, f64)> = (0..40)
            .map(|i| if i % 4 == 0 { (0.0, 1.0) } else { (1.0, 1.0) })
            .collect();
        let (lo, hi) = cluster_ratio_ci(&clusters, 0.05, 31);
        let (lo2, hi2) = cluster_ratio_ci(&clusters, 0.05, 31);
        assert_eq!((lo, hi), (lo2, hi2));
        assert!(lo < 0.75 && 0.75 < hi, "点推定 0.75 を挟む: {lo}..{hi}");
        // 標本が1クラスタしか無ければ幅は 0（縮退。判定には使えないことを明示）
        let (l, h) = cluster_ratio_ci(&[(1.0, 2.0)], 0.05, 1);
        assert_eq!((l, h), (0.5, 0.5));
    }
}
