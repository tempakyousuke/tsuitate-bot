//! RL 環境の1局ぶん（docs/rl-deepnash-design.md の「5. Python 環境」）。
//!
//! `Referee`（裁定）に、学習側から見た都合を足したもの:
//! - 行動番号で1手進める（`action` の符号化）
//! - **相手を既存の Rust 戦略にする評価モード**: 相手の手番は内部で自動的に指し、
//!   学習側の手番でだけ止まる
//! - 自駒視点の候補が1つも残らない（全部反則済み等）手番は投了として終局させる
//!
//! 並列化と numpy 変換は PyO3 側（`rl-env/`）が持つ。ここは Python に依存しないので
//! `cargo test` で検査できる。

use crate::board::Coord;
use crate::protocol::{Color, Role};
use crate::referee::{Referee, StepResult};
use crate::rl::action::{NUM_ACTIONS, decode_usi, index_square, legal_mask, normalize};
use crate::rl::encode::{DeducedCache, encode_into_cached};
use crate::rl::forced::ForcedPlan;
use crate::selfplay::{GameResult, StartState, fischer_initial_ms};
use crate::shogi::{ShogiMove, parse_usi};
use crate::strategy::Strategy;

/// `EnvGame::truth_into` の相手駒のラベルの駒種の並び（観測の自駒プレーンと同じ `Role` の宣言順）
pub const TRUTH_ROLES: [Role; 14] = [
    Role::Pawn,
    Role::Lance,
    Role::Knight,
    Role::Silver,
    Role::Gold,
    Role::Bishop,
    Role::Rook,
    Role::King,
    Role::Tokin,
    Role::Promotedlance,
    Role::Promotedknight,
    Role::Promotedsilver,
    Role::Horse,
    Role::Dragon,
];

/// 玉の周りの集計の半径（チェビシェフ距離。1 = 8近傍、2 = 距離2以内の24マス）
pub const GUARD_RADII: [i8; 2] = [1, 2];

/// **玉の周りを固める戦法**の集計（docs/rl-deepnash-design.md の「防御特化（玉の周りの固め）」）。
///
/// スタイル特化モデルの報酬と監視に使う。自分の配置は完全既知の情報なので、どれも各側が
/// 自分で数えられる量（報酬に使っても相手の情報は漏れない）。添字は `[先手, 後手]`、
/// 半径の添字は `GUARD_RADII` の並び。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StyleStats {
    /// 占有率（自玉の周りの盤上のマスのうち、玉以外の自駒がいる割合）の合計。
    /// 開始局面と受理手の直後ごとに両者ぶん足す
    pub guard_sum: [[f64; 2]; 2],
    /// `guard_sum` の標本数
    pub samples: u32,
    /// 自玉から距離2以内への打ち（受理）
    pub near_drops: [u32; 2],
    /// 自玉から距離2以内への打ちの反則のうち、**打ったマスに相手の駒がいたもの**（＝相手の駒の検知）。
    /// 王手を防がない打ち・打ち歩詰めなど、マスが空いていた反則は含めない
    pub near_drop_fouls: [u32; 2],
    /// 自玉から距離2以内での駒取り
    pub near_captures: [u32; 2],
}

impl StyleStats {
    /// 対局を通した占有率の平均（`radius_idx` は `GUARD_RADII` の添字）。[先手, 後手]、∈ [0, 1]
    pub fn guard_mean(&self, radius_idx: usize) -> [f64; 2] {
        if self.samples == 0 {
            return [0.0, 0.0];
        }
        let n = f64::from(self.samples);
        [self.guard_sum[0][radius_idx] / n, self.guard_sum[1][radius_idx] / n]
    }

    fn sample(&mut self, referee: &Referee) {
        for (i, color) in [Color::Sente, Color::Gote].into_iter().enumerate() {
            for (r, &radius) in GUARD_RADII.iter().enumerate() {
                self.guard_sum[i][r] += guard_fraction(referee, color, radius);
            }
        }
        self.samples += 1;
    }
}

/// `color` の玉から距離 `radius` 以内の盤上のマス（玉のマスを除く）のうち、玉以外の自駒が
/// いる割合。盤端の玉はマス数が減るので割合で数える（端にいるだけで損をしないように）
pub fn guard_fraction(referee: &Referee, color: Color, radius: i8) -> f64 {
    let pos = referee.position();
    let Some(k) = pos.king_square(color) else {
        return 0.0;
    };
    let (mut squares, mut own) = (0u32, 0u32);
    for df in -radius..=radius {
        for dr in -radius..=radius {
            if df == 0 && dr == 0 {
                continue;
            }
            let c = Coord { file: k.file + df, rank: k.rank + dr };
            if !crate::board::on_board(c) {
                continue;
            }
            squares += 1;
            if pos.piece_at(c).is_some_and(|p| p.color == color) {
                own += 1;
            }
        }
    }
    if squares == 0 { 0.0 } else { f64::from(own) / f64::from(squares) }
}

fn near(a: Coord, b: Coord, radius: i8) -> bool {
    (a.file - b.file).abs() <= radius && (a.rank - b.rank).abs() <= radius
}

/// 終局の記録
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Outcome {
    pub result: GameResult,
    pub reason: &'static str,
    pub plies: u32,
    pub fouls: [u32; 2],
}

impl Outcome {
    /// [先手, 後手] の報酬（勝ち +1 / 負け −1 / 引き分け 0）
    pub fn rewards(&self) -> [f32; 2] {
        match self.result {
            GameResult::Win(Color::Sente) => [1.0, -1.0],
            GameResult::Win(Color::Gote) => [-1.0, 1.0],
            GameResult::Draw => [0.0, 0.0],
        }
    }

    /// 決着の速さ ∈ [0, 1] = max(0, (horizon − 手数) / horizon)。手数は受理手の数（反則は数えない）。
    /// 引き分けは 0（`max_plies` は決着していない）。`horizon` = 0 も 0
    pub fn speed(&self, horizon: u32) -> f32 {
        if matches!(self.result, GameResult::Draw) || horizon == 0 {
            return 0.0;
        }
        (horizon.saturating_sub(self.plies) as f32) / horizon as f32
    }

    /// 速攻ボーナス（docs/rl-deepnash-design.md の「速攻特化」）。[先手, 後手] の加算分で、
    /// **勝った側に λ × 速さ**を足す。`SpeedSum::Zero` なら負けた側から同じ額を引く（零和）。
    /// 引き分けは両者 0
    pub fn speed_bonus(&self, lambda: f32, horizon: u32, sum: SpeedSum) -> [f32; 2] {
        let GameResult::Win(winner) = self.result else {
            return [0.0, 0.0];
        };
        let b = lambda * self.speed(horizon);
        let loser = match sum {
            SpeedSum::Own => 0.0,
            SpeedSum::Zero => -b,
        };
        match winner {
            Color::Sente => [b, loser],
            Color::Gote => [loser, b],
        }
    }
}

/// 速攻ボーナスの配り方
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeedSum {
    /// 勝った側だけに足す（非零和。負けた側の報酬は勝敗のまま = 「遅く負ける」動機を作らない）
    Own,
    /// 勝った側に足し、負けた側から同じ額を引く（零和。自己対局の R-NaD の前提を保つ代わりに、
    /// 負けそうな側は手数を延ばす動機を持つ）
    Zero,
}

impl SpeedSum {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "own" => Some(Self::Own),
            "zero" => Some(Self::Zero),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvError {
    /// 行動番号が範囲外、またはマスクの外（学習側のマスク適用漏れ）
    MaskedAction(usize),
    /// 終局済みの局に行動を渡した
    GameOver,
}

/// 序盤の反則を数える手数（各側の受理手。`EnvGame::fouls_early`）。速攻どうしでは駒が7〜8手目でぶつかるので、
/// 接触の前の反則（`fouls_precontact`）はほぼ 0 になる。相手の序盤が変わったときの「いつもの形を前提にした手」の
/// 反則はこちらで見る
pub const EARLY_MOVES: u32 = 12;

/// 1つの側の序盤の強制の状態（`rl::forced`）
#[derive(Debug, Clone, Default)]
struct ForcedState {
    plan: Option<ForcedPlan>,
    /// 強制して受理された手の数（= ラインの進み）
    played: u32,
    /// 強制をやめた理由（一度やめたら戻らない）
    stop: Option<&'static str>,
}

/// 1つの側の序盤の強制の記録（終局後も読める。`EnvGame::forced_stats`）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ForcedStats {
    /// 割り当てたライン（強制しない側は None）
    pub line: Option<u32>,
    /// 強制して受理された手の数
    pub moves: u32,
    /// やめた理由: "contact"（どちらかの駒取り・王手）/ "length"（予定の手数を指し終えた。指し終えた時点で決まる）/
    /// "foul"（強制した手が反則）/ "not_in_mask"（マスクに無い。検査済みのラインでは起きない）/
    /// "game_end"（予定の途中で、その側の次の手番が来る前に終局した）/ "active"（対局中でまだ強制している）。
    /// 強制しない側は None
    pub stop: Option<&'static str>,
}

pub struct EnvGame {
    referee: Referee,
    /// 評価モードで学習側が持つ色（None = 自己対局。両者とも学習側）
    learner: Option<Color>,
    opponent: Option<Box<dyn Strategy + Send>>,
    game_no: u32,
    /// 今の手番（学習側）のマスク。終局後は空
    mask: Vec<bool>,
    outcome: Option<Outcome>,
    style: StyleStats,
    /// 確定事実のプレーンの途中状態（先手・後手の観測ごと。この局の中でだけ使い回す）
    deduced: [DeducedCache; 2],
    /// 序盤の強制（[先手, 後手]。強制しない局は両方とも予定なし）
    forced: [ForcedState; 2],
    /// 最初の接触（どちらかの駒取り・王手。両者に見える）が起きた時点の受理手の数
    contact_ply: Option<u32>,
    /// 最初の接触より前の反則の数 [先手, 後手]
    fouls_precontact: [u32; 2],
    /// 各側の受理手の数と、受理手が `EARLY_MOVES` に届くまでの反則の数 [先手, 後手]
    own_moves: [u32; 2],
    fouls_early: [u32; 2],
}

impl EnvGame {
    /// 自己対局の1局。`start` が None なら初期局面から
    pub fn selfplay(start: Option<StartState>, game_no: u32) -> Self {
        Self::build(start, None, None, game_no, [None, None])
    }

    /// 序盤を強制する1局（初期局面から。`plans` は [先手, 後手]、`rl::forced`）。
    /// `learner` / `opponent` が None なら自己対局、Some なら評価モード
    pub fn with_forced(
        learner: Option<Color>,
        opponent: Option<Box<dyn Strategy + Send>>,
        game_no: u32,
        plans: [Option<ForcedPlan>; 2],
    ) -> Self {
        assert_eq!(learner.is_some(), opponent.is_some(), "評価モードは学習側の色と相手の両方を持つ");
        Self::build(None, learner, opponent, game_no, plans)
    }

    /// 評価モードの1局。`opponent` が `learner` の相手側を指す
    pub fn versus(
        start: Option<StartState>,
        learner: Color,
        opponent: Box<dyn Strategy + Send>,
        game_no: u32,
    ) -> Self {
        Self::build(start, Some(learner), Some(opponent), game_no, [None, None])
    }

    fn build(
        start: Option<StartState>,
        learner: Option<Color>,
        opponent: Option<Box<dyn Strategy + Send>>,
        game_no: u32,
        plans: [Option<ForcedPlan>; 2],
    ) -> Self {
        let referee = start.map_or_else(Referee::new, Referee::from_start);
        let [s, g] = plans;
        let mut game = Self {
            referee,
            learner,
            opponent,
            game_no,
            mask: vec![],
            outcome: None,
            style: StyleStats::default(),
            deduced: Default::default(),
            forced: [ForcedState { plan: s, ..Default::default() }, ForcedState { plan: g, ..Default::default() }],
            contact_ply: None,
            fouls_precontact: [0, 0],
            own_moves: [0, 0],
            fouls_early: [0, 0],
        };
        game.style.sample(&game.referee);
        game.settle();
        game
    }

    /// 終局していればその記録
    pub fn outcome(&self) -> Option<Outcome> {
        self.outcome
    }

    /// 今の手番（学習側）の色
    pub fn to_move(&self) -> Color {
        self.referee.to_move()
    }

    pub fn game_no(&self) -> u32 {
        self.game_no
    }

    pub fn referee(&self) -> &Referee {
        &self.referee
    }

    pub fn mask(&self) -> &[bool] {
        &self.mask
    }

    /// 玉の周りを固める戦法の集計（終局後も読める）
    pub fn style(&self) -> &StyleStats {
        &self.style
    }

    /// 序盤の強制の記録 [先手, 後手]（終局後も読める）
    pub fn forced_stats(&self) -> [ForcedStats; 2] {
        let running = if self.outcome.is_none() { "active" } else { "game_end" };
        self.forced.each_ref().map(|f| ForcedStats {
            line: f.plan.as_ref().map(|p| p.line_id),
            moves: f.played,
            stop: f.plan.as_ref().map(|_| f.stop.unwrap_or(running)),
        })
    }

    /// 最初の接触（どちらかの駒取り・王手）が起きた時点の受理手の数（起きていなければ None）
    pub fn contact_ply(&self) -> Option<u32> {
        self.contact_ply
    }

    /// 最初の接触より前の反則の数 [先手, 後手]
    pub fn fouls_precontact(&self) -> [u32; 2] {
        self.fouls_precontact
    }

    /// 各側の受理手が `EARLY_MOVES` に届くまでの反則の数 [先手, 後手]
    pub fn fouls_early(&self) -> [u32; 2] {
        self.fouls_early
    }

    /// 審判へ1手渡し、玉の周りの集計を更新する（学習側・相手側の両方の手がここを通る）
    fn play(&mut self, usi: &str) -> StepResult {
        let side = self.referee.to_move();
        let i = usize::from(side == Color::Gote);
        let pos = self.referee.position();
        let king = pos.king_square(side);
        let mv = parse_usi(usi);
        let target = match mv {
            Some(ShogiMove::Board { to, .. }) | Some(ShogiMove::Drop { to, .. }) => Some(to),
            None => None,
        };
        let is_drop = matches!(mv, Some(ShogiMove::Drop { .. }));
        // 行き先に相手の駒がいるか（真実の盤面で判定する。打ちなら必ず反則になり、それが検知）
        let hits_opponent = target
            .is_some_and(|t| pos.piece_at(t).is_some_and(|p| p.color != side));
        let near_king = matches!((king, target), (Some(k), Some(t)) if near(k, t, 2));

        let result = self.referee.step(usi, 0);
        let accepted = match result {
            StepResult::Accepted => true,
            StepResult::Foul => false,
            StepResult::Ended { move_accepted, .. } => move_accepted,
        };
        if near_king {
            match (accepted, is_drop, hits_opponent) {
                (true, true, _) => self.style.near_drops[i] += 1,
                // 王手を防がない打ち・打ち歩詰めの反則は検知ではないので数えない
                (false, true, true) => self.style.near_drop_fouls[i] += 1,
                (true, false, true) => self.style.near_captures[i] += 1,
                _ => {}
            }
        }
        if accepted {
            self.style.sample(&self.referee);
        }
        let fouled = matches!(result, StepResult::Foul | StepResult::Ended { move_accepted: false, .. });
        if fouled && self.own_moves[i] < EARLY_MOVES {
            self.fouls_early[i] += 1;
        }
        if accepted {
            self.own_moves[i] += 1;
        }
        if self.contact_ply.is_none() {
            // 接触 = 駒取りか王手（どちらも両者の観測に入る）。打ちが受理されたなら行き先は空いていた
            if accepted && (hits_opponent || self.referee.position().in_check(self.referee.to_move())) {
                self.contact_ply = Some(self.referee.plies());
            } else if fouled {
                self.fouls_precontact[i] += 1;
            }
        }
        result
    }

    /// 今の手番の観測を書く（`obs` の長さでエンコーダの版を選ぶ。`rl::encode::encode_into`）
    /// v2 の確定事実は手番側ごとの途中状態を使い回す（結果は `encode_into` と同じ）
    pub fn observe_into(&mut self, obs: &mut [f32]) {
        let side = self.referee.to_move();
        let view = self.referee.view(side, clocks(), self.game_no);
        let cache = &mut self.deduced[(side == Color::Gote) as usize];
        encode_into_cached(&view, self.referee.log(side), self.referee.foul_tried(side), obs, cache);
    }

    /// 今の手番から見た**真実の盤面のラベル**を書く（`out` は長さ 81、マス番号は観測と同じ
    /// `action::square_index` で手番側を先手向きに正規化）。値は −1 = 自駒（観測で既知）、
    /// 0 = 空き、1〜14 = 相手の駒（`TRUTH_ROLES` の添字 + 1）。
    ///
    /// **学習の補助損失・診断専用**（相手駒の位置は観測に無い情報なので、方策の入力には使わない）
    pub fn truth_into(&self, out: &mut [i8]) {
        debug_assert_eq!(out.len(), 81);
        let side = self.referee.to_move();
        let pos = self.referee.position();
        for (i, o) in out.iter_mut().enumerate() {
            let c = normalize(index_square(i), side);
            *o = match pos.piece_at(c) {
                None => 0,
                Some(p) if p.color == side => -1,
                Some(p) => TRUTH_ROLES.iter().position(|&r| r == p.role).map_or(0, |k| k as i8 + 1),
            };
        }
    }

    /// 行動が今の手番で受け付けられるか（状態は変えない）。並列環境はバッチ全体を
    /// これで検査してから `step` する（1局の不正で他の局だけ進むのを防ぐため）
    pub fn check_action(&self, action: usize) -> Result<(), EnvError> {
        if self.outcome.is_some() {
            return Err(EnvError::GameOver);
        }
        if !self.mask.get(action).copied().unwrap_or(false) {
            return Err(EnvError::MaskedAction(action));
        }
        Ok(())
    }

    /// 学習側の手番に行動を適用し、次の学習側の手番まで進める。終局したら Some
    pub fn step(&mut self, action: usize) -> Result<Option<Outcome>, EnvError> {
        self.check_action(action)?;
        let side = self.referee.to_move();
        let usi = decode_usi(action, side).ok_or(EnvError::MaskedAction(action))?;
        if let StepResult::Ended { result, reason, .. } = self.play(&usi) {
            self.finish(result, reason);
        } else {
            self.settle();
        }
        Ok(self.outcome)
    }

    /// 序盤の強制の次の手（強制しない・やめた・やめるなら None。やめる理由はここで記録し、
    /// 以後その局では戻らない）
    fn forced_move(&mut self, side: Color) -> Option<String> {
        let i = usize::from(side == Color::Gote);
        let state = &self.forced[i];
        let plan = state.plan.as_ref()?;
        if state.stop.is_some() {
            return None;
        }
        // 予定の手数を指し終えたら、その時点で "length" にしている（settle）。ここでの "length" は
        // 手で作った予定がラインより長いとき
        let stop = if self.contact_ply.is_some() {
            Some("contact")
        } else if let Some(&a) = plan.line.get(state.played as usize).filter(|_| state.played < plan.limit) {
            let view = self.referee.view(side, clocks(), self.game_no);
            let mask = legal_mask(&view, self.referee.log(side), self.referee.foul_tried(side));
            if mask.get(a).copied().unwrap_or(false) {
                return decode_usi(a, side);
            }
            // 自駒だけの盤で検査したラインで、接触も反則もまだ無いなら起きない
            Some("not_in_mask")
        } else {
            Some("length")
        };
        self.forced[i].stop = stop;
        None
    }

    /// 相手（Rust 戦略）の手番と序盤の強制の手を指し進め、学習側の手番で止めてマスクを作る
    fn settle(&mut self) {
        loop {
            if let Some((result, reason)) = self.referee.ended() {
                self.finish(result, reason);
                return;
            }
            let side = self.referee.to_move();
            // 強制の手は学習側・相手側どちらの手番でも、その側が選んだ手と同じ経路（play）で指す
            // （両者の観測・反則済みの手・集計が、その側が自分でその手を選んだときと同じになる）
            if let Some(usi) = self.forced_move(side) {
                let i = usize::from(side == Color::Gote);
                let result = self.play(&usi);
                let f = &mut self.forced[i];
                match result {
                    StepResult::Accepted | StepResult::Ended { move_accepted: true, .. } => {
                        f.played += 1;
                        if f.plan.as_ref().is_some_and(|p| f.played >= p.limit.min(p.line.len() as u32)) {
                            f.stop = Some("length");
                        }
                    }
                    // 見えない相手の駒に遮られた等。反則は数え、同じ手番を方策（相手）に渡す
                    StepResult::Foul | StepResult::Ended { move_accepted: false, .. } => f.stop = Some("foul"),
                }
                if let StepResult::Ended { result, reason, .. } = result {
                    self.finish(result, reason);
                    return;
                }
                continue;
            }
            let learner_turn = self.learner.is_none_or(|l| l == side);
            if learner_turn {
                let view = self.referee.view(side, clocks(), self.game_no);
                self.mask = legal_mask(&view, self.referee.log(side), self.referee.foul_tried(side));
                if !self.mask.iter().any(|&m| m) {
                    // 自駒視点の候補が尽きた（全部反則済み）= 指せる手がない
                    self.finish(GameResult::Win(side.other()), "no_moves");
                }
                return;
            }
            let view = self.referee.view(side, clocks(), self.game_no);
            let opponent = self.opponent.as_mut().expect("評価モードには相手がいる");
            let choice = opponent.choose(&view, self.referee.log(side), self.referee.foul_tried(side));
            let Some(usi) = choice else {
                self.finish(GameResult::Win(side.other()), "resign");
                return;
            };
            if let StepResult::Ended { result, reason, .. } = self.play(&usi) {
                self.finish(result, reason);
                return;
            }
        }
    }

    fn finish(&mut self, result: GameResult, reason: &'static str) {
        self.mask.clear();
        self.outcome = Some(Outcome {
            result,
            reason,
            plies: self.referee.plies(),
            fouls: [
                self.referee.fouls(Color::Sente),
                self.referee.fouls(Color::Gote),
            ],
        });
    }
}

/// 時計は持たない（戦略は clocks を読まない。値は arena の初期持ち時間を見せておく）
fn clocks() -> [i64; 2] {
    [fischer_initial_ms(), fischer_initial_ms()]
}

/// 行動マスク中の行動を一様に選ぶ（テスト・ベースライン用）
pub fn random_action(mask: &[bool], r: f64) -> Option<usize> {
    debug_assert_eq!(mask.len(), NUM_ACTIONS);
    let n = mask.iter().filter(|&&m| m).count();
    if n == 0 {
        return None;
    }
    let k = ((r * n as f64) as usize).min(n - 1);
    mask.iter()
        .enumerate()
        .filter(|&(_, &m)| m)
        .nth(k)
        .map(|(a, _)| a)
}

#[cfg(test)]
mod tests {
    #[test]
    fn 真実のラベルは手番側から見た相手駒を正規化して返す() {
        use crate::rl::action::square_index;
        let mut g = EnvGame::selfplay(None, 0);
        for side in [Color::Sente, Color::Gote] {
            assert_eq!(g.to_move(), side);
            let mut t = [0i8; 81];
            g.truth_into(&mut t);
            // 初手（先手）と2手目（後手）: どちらも相手の玉は正規化後の 5一
            let king = TRUTH_ROLES.iter().position(|&r| r == Role::King).unwrap() as i8 + 1;
            assert_eq!(t[square_index(Coord { file: 5, rank: 1 })], king);
            assert_eq!(t[square_index(Coord { file: 5, rank: 9 })], -1);
            assert_eq!(t.iter().filter(|&&v| v > 0).count(), 20);
            assert_eq!(t.iter().filter(|&&v| v < 0).count(), 20);
            if side == Color::Sente {
                // 先手の 7六歩（初手は相手の陣に触れない）
                let a = (0..NUM_ACTIONS).find(|&a| g.mask()[a] && g.check_action(a).is_ok()
                    && decode_usi(a, Color::Sente).as_deref() == Some("7g7f")).unwrap();
                g.step(a).unwrap();
            }
        }
    }

    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn play_random(game: &mut EnvGame, rng: &mut StdRng) -> Outcome {
        loop {
            if let Some(o) = game.outcome() {
                return o;
            }
            let a = random_action(game.mask(), rng.random()).expect("未終局ならマスクは空でない");
            game.step(a).unwrap();
        }
    }

    #[test]
    fn 自己対局は必ず終局し報酬は零和() {
        let mut rng = StdRng::seed_from_u64(3);
        for g in 0..20 {
            let mut game = EnvGame::selfplay(None, g);
            let o = play_random(&mut game, &mut rng);
            let [s, t] = o.rewards();
            assert_eq!(s + t, 0.0);
            assert!(o.plies <= crate::selfplay::MAX_PLIES);
        }
    }

    #[test]
    fn 評価モードは学習側の手番でだけ止まる() {
        let mut rng = StdRng::seed_from_u64(4);
        for (g, learner) in [(0, Color::Sente), (1, Color::Gote)] {
            let opp = crate::strategy::make("heuristic").unwrap();
            let mut game = EnvGame::versus(None, learner, opp, g);
            while game.outcome().is_none() {
                assert_eq!(game.to_move(), learner);
                let a = random_action(game.mask(), rng.random()).unwrap();
                game.step(a).unwrap();
            }
        }
    }

    #[test]
    fn マスク外の行動と終局後の行動は拒否する() {
        let mut game = EnvGame::selfplay(None, 0);
        let bad = game.mask().iter().position(|&m| !m).unwrap();
        assert_eq!(game.step(bad), Err(EnvError::MaskedAction(bad)));
        assert_eq!(game.step(NUM_ACTIONS), Err(EnvError::MaskedAction(NUM_ACTIONS)));

        let mut rng = StdRng::seed_from_u64(5);
        play_random(&mut game, &mut rng);
        assert_eq!(game.step(0), Err(EnvError::GameOver));
    }

    #[test]
    fn 初期局面の玉の周りの占有率() {
        let game = EnvGame::selfplay(None, 0);
        // 5九玉の8近傍で盤上は5マス、うち自駒は金2枚
        assert_eq!(guard_fraction(game.referee(), Color::Sente, 1), 0.4);
        assert_eq!(guard_fraction(game.referee(), Color::Gote, 1), 0.4);
        let s = game.style();
        assert_eq!(s.samples, 1);
        assert_eq!(s.guard_mean(0), [0.4, 0.4]);
    }

    #[test]
    fn 玉の周りの集計は受理手ごとに標本を取り値域に収まる() {
        let mut rng = StdRng::seed_from_u64(6);
        let mut drops = 0;
        for g in 0..10 {
            let mut game = EnvGame::selfplay(None, g);
            let mut accepted = 0;
            loop {
                if game.outcome().is_some() {
                    break;
                }
                let a = random_action(game.mask(), rng.random()).unwrap();
                let before = game.referee().plies();
                game.step(a).unwrap();
                if game.referee().plies() > before {
                    accepted += 1;
                }
            }
            let s = game.style();
            assert_eq!(s.samples, 1 + accepted, "開始局面＋受理手ごと");
            for r in 0..GUARD_RADII.len() {
                for v in s.guard_mean(r) {
                    assert!((0.0..=1.0).contains(&v));
                }
            }
            drops += s.near_drops.iter().sum::<u32>() + s.near_drop_fouls.iter().sum::<u32>();
        }
        assert!(drops > 0, "ランダム対局でも玉の近くへの打ちは起きる");
    }

    #[test]
    fn 検知として数えるのは相手の駒がいたマスへの打ちの反則だけ() {
        use crate::protocol::Role;
        use crate::rl::action::encode_usi;
        use crate::shogi::{Piece, Position};
        let sq = |file, rank| Coord { file, rank };
        let put = |pos: &mut Position, c, color, role| pos.set(c, Some(Piece { color, role }));
        // 先手 5九玉が 1九の飛車で王手されている。6八に後手の銀
        let mut pos = Position::empty(Color::Sente);
        put(&mut pos, sq(5, 9), Color::Sente, Role::King);
        put(&mut pos, sq(5, 1), Color::Gote, Role::King);
        put(&mut pos, sq(1, 9), Color::Gote, Role::Rook);
        put(&mut pos, sq(6, 8), Color::Gote, Role::Silver);
        pos.set_hand(Color::Sente, Role::Pawn, 1);
        let start = StartState { pos, logs: Default::default(), fouls: [0, 0], plies: 0 };
        let mut game = EnvGame::selfplay(Some(start), 0);

        // 王手を防がない空きマスへの打ち（反則だが検知ではない）
        game.step(encode_usi("P*5h", Color::Sente).unwrap()).unwrap();
        assert_eq!(game.referee().fouls(Color::Sente), 1);
        assert_eq!(game.style().near_drop_fouls, [0, 0]);

        // 後手の銀がいるマスへの打ち（検知）
        game.step(encode_usi("P*6h", Color::Sente).unwrap()).unwrap();
        assert_eq!(game.referee().fouls(Color::Sente), 2);
        assert_eq!(game.style().near_drop_fouls, [1, 0]);
    }

    #[test]
    fn 評価モードでは相手の手も集計に入る() {
        let opp = crate::strategy::make("heuristic").unwrap();
        // 学習側が後手なら、構築時に相手（先手）の初手が指されて標本が2つになる
        let game = EnvGame::versus(None, Color::Gote, opp, 0);
        assert_eq!(game.style().samples, 2);
    }

    #[test]
    fn 速攻ボーナスは勝った側に速さに比例して付く() {
        let o = |result, plies| Outcome { result, reason: "checkmate", plies, fouls: [0, 0] };
        let sente = o(GameResult::Win(Color::Sente), 50);
        assert_eq!(sente.speed(200), 0.75);
        assert_eq!(sente.speed_bonus(2.0, 200, SpeedSum::Own), [1.5, 0.0]);
        assert_eq!(sente.speed_bonus(2.0, 200, SpeedSum::Zero), [1.5, -1.5]);

        let gote = o(GameResult::Win(Color::Gote), 100);
        assert_eq!(gote.speed_bonus(1.0, 200, SpeedSum::Own), [0.0, 0.5]);
        assert_eq!(gote.speed_bonus(1.0, 200, SpeedSum::Zero), [-0.5, 0.5]);

        // 速いほど大きく、horizon 以上の手数では 0（負にはならない）
        let fast = o(GameResult::Win(Color::Sente), 30).speed(100);
        let slow = o(GameResult::Win(Color::Sente), 70).speed(100);
        assert!(fast > slow);
        assert_eq!(o(GameResult::Win(Color::Sente), 120).speed(100), 0.0);
        assert_eq!(o(GameResult::Win(Color::Sente), 120).speed(0), 0.0);

        // 引き分けには付かない
        let draw = o(GameResult::Draw, 10);
        assert_eq!(draw.speed(200), 0.0);
        assert_eq!(draw.speed_bonus(5.0, 200, SpeedSum::Zero), [0.0, 0.0]);

        assert_eq!(SpeedSum::parse("own"), Some(SpeedSum::Own));
        assert_eq!(SpeedSum::parse("zero"), Some(SpeedSum::Zero));
        assert_eq!(SpeedSum::parse("both"), None);
    }

    fn plan(moves: &[&str], limit: u32) -> crate::rl::forced::ForcedPlan {
        use crate::rl::action::encode_usi;
        let line: Vec<usize> = moves.iter().map(|u| encode_usi(u, Color::Sente).unwrap()).collect();
        crate::rl::forced::ForcedPlan { line: line.into(), line_id: 7, limit }
    }

    fn act(game: &EnvGame, usi: &str) -> usize {
        crate::rl::action::encode_usi(usi, game.to_move()).unwrap()
    }

    /// その側の試行（受理手と反則）を順に並べる
    fn attempts(game: &EnvGame, side: Color) -> Vec<String> {
        game.referee()
            .log(side)
            .events()
            .iter()
            .filter_map(|e| match e {
                crate::observation::Observation::MyMove { usi, .. }
                | crate::observation::Observation::MyFoul { usi, .. } => Some(usi.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn 強制した手はその側が自分で選んだときと同じ観測を残す() {
        use crate::rl::records::same_observations;
        let mut rng = StdRng::seed_from_u64(11);
        let lines = [
            (["7g7f", "6g6f", "2h6h", "5i4h", "4h3h"].as_slice(), ["2g2f", "2f2e", "2e2d"].as_slice()),
            (["5g5f", "2h5h", "5f5e", "5i4h"].as_slice(), ["7i7h", "8h7i", "5i4h", "5g5f", "3g3f"].as_slice()),
            (["9g9f", "8h9g", "2h8h", "7i7h", "5i6h"].as_slice(), ["7g7f", "2h3h", "3g3f", "3f3e"].as_slice()),
        ];
        for (g, (ls, lg)) in lines.into_iter().enumerate() {
            for limits in [(9, 9), (2, 1), (1, 9)] {
                let plans = [Some(plan(ls, limits.0.min(ls.len() as u32))), Some(plan(lg, limits.1.min(lg.len() as u32)))];
                let mut a = EnvGame::with_forced(None, None, g as u32, plans);
                play_random(&mut a, &mut rng);
                let stats = a.forced_stats();
                assert!(stats.iter().all(|s| s.moves >= 1 && s.line == Some(7)), "{stats:?}");
                // 強制しない局で、同じ試行を両者が自分で選ぶ
                let mut b = EnvGame::selfplay(None, g as u32);
                let mut next = [0usize, 0];
                let want = [attempts(&a, Color::Sente), attempts(&a, Color::Gote)];
                while b.outcome().is_none() {
                    let side = b.to_move();
                    let i = usize::from(side == Color::Gote);
                    let a_ = act(&b, &want[i][next[i]]);
                    next[i] += 1;
                    b.step(a_).unwrap();
                }
                for side in [Color::Sente, Color::Gote] {
                    assert!(same_observations(a.referee().log(side).events(), b.referee().log(side).events()));
                }
                assert_eq!(a.outcome(), b.outcome());
                assert_eq!(a.style(), b.style());
                assert_eq!(a.contact_ply(), b.contact_ply());
                assert_eq!(a.fouls_precontact(), b.fouls_precontact());
                assert_eq!(b.forced_stats(), [ForcedStats::default(); 2]);
            }
        }
    }

    #[test]
    fn 接触で両者とも強制をやめる() {
        // 先手だけ強制。後手（方策の代わりにテストが指す）の 8七歩成で先手の駒が取られる
        let line = ["7g7f", "6g6f", "2h6h", "5i4h", "4h3h", "3i4h"];
        let mut game = EnvGame::with_forced(None, None, 0, [Some(plan(&line, 6)), None]);
        for usi in ["8c8d", "8d8e", "8e8f"] {
            assert_eq!(game.to_move(), Color::Gote);
            game.step(act(&game, usi)).unwrap();
        }
        assert_eq!(game.contact_ply(), None);
        game.step(act(&game, "8f8g+")).unwrap();
        assert_eq!(game.contact_ply(), Some(8));
        // 先手は4手強制したところで接触したので、その手番から方策が指す
        assert_eq!(game.to_move(), Color::Sente);
        assert_eq!(game.forced_stats()[0].moves, 4);
        assert_eq!(game.forced_stats()[0].stop, Some("contact"));
        assert_eq!(game.forced_stats()[1], ForcedStats::default());

        // 両者とも強制し、強制した手どうしがぶつかる（後手の 2五歩が先手の歩を取る）
        let s = ["2g2f", "2f2e", "7g7f", "6g6f"];
        let g = ["8g8f", "8f8e", "8e8d"]; // 後手から見ると 2c2d 2d2e 2e2f
        let game = EnvGame::with_forced(None, None, 1, [Some(plan(&s, 4)), Some(plan(&g, 3))]);
        assert_eq!(game.contact_ply(), Some(4));
        let stats = game.forced_stats();
        assert_eq!((stats[0].moves, stats[1].moves), (2, 2));
        assert_eq!(game.to_move(), Color::Sente);
        assert_eq!(stats[0].stop, Some("contact"));
        // 後手の次の手番はまだ来ていない（来ればそこで "contact" でやめる）
        assert_eq!(stats[1].stop, Some("active"));

        // 予定を指し終えた後に接触しても、やめた理由は "length"（指し終えた時点で決まる）
        let rush = ["2g2f", "2f2e", "2e2d"];
        let mut game = EnvGame::with_forced(None, None, 2, [Some(plan(&rush, 3)), None]);
        game.step(act(&game, "9c9d")).unwrap();
        assert_eq!(game.forced_stats()[0].stop, Some("active"));
        game.step(act(&game, "9d9e")).unwrap(); // ここで先手が3手目を強制で指し終える
        assert_eq!(game.forced_stats()[0].stop, Some("length"));
        game.step(act(&game, "2c2d")).unwrap(); // 後手が先手の 2四の歩を取る（接触）
        assert_eq!(game.contact_ply(), Some(6));
        assert_eq!(game.forced_stats()[0], ForcedStats { line: Some(7), moves: 3, stop: Some("length") });
    }

    #[test]
    fn 強制した手が反則ならその側の強制をやめて同じ手番を方策に渡す() {
        // 先手の 4四角は 5五の後手の歩に遮られて反則になる（先手からは見えない）
        let line = ["7g7f", "9g9f", "8h4d", "5i4h"];
        let mut game = EnvGame::with_forced(None, None, 0, [Some(plan(&line, 4)), None]);
        game.step(act(&game, "5c5d")).unwrap();
        game.step(act(&game, "5d5e")).unwrap();
        assert_eq!(game.to_move(), Color::Sente);
        assert_eq!(game.referee().fouls(Color::Sente), 1);
        assert_eq!(game.fouls_precontact(), [1, 0]);
        assert!(game.referee().foul_tried(Color::Sente).contains("8h4d"));
        assert!(!game.mask()[act(&game, "8h4d")], "反則済みの手はマスクから落ちる");
        assert_eq!(game.forced_stats()[0].moves, 2);
        assert_eq!(game.forced_stats()[0].stop, Some("foul"));
        // 後手にも反則の数は見えている（反則は観測に入る）
        assert!(game.referee().log(Color::Gote).events().iter().any(|e| matches!(
            e,
            crate::observation::Observation::OpponentFoul { count: 1 }
        )));
    }

    #[test]
    fn 序盤の反則は各側の受理手が既定の手数に届くまで数える() {
        // 先手は 4四角で反則（5五の後手の歩に遮られる）。後手の受理手は数に入らない
        let mut game = EnvGame::selfplay(None, 0);
        for usi in ["7g7f", "5c5d", "9g9f", "5d5e"] {
            game.step(act(&game, usi)).unwrap();
        }
        game.step(act(&game, "8h4d")).unwrap();
        assert_eq!((game.fouls_early(), game.fouls_precontact()), ([1, 0], [1, 0]));
        // 受理手が EARLY_MOVES に届いた後の反則は数えない（接触の前なので fouls_precontact には入る）
        let mut game = EnvGame::selfplay(None, 1);
        let mut sente = vec!["7g7f"];
        let mut gote = vec!["5c5d", "5d5e"];
        while sente.len() < EARLY_MOVES as usize {
            sente.push(if sente.len() % 2 == 1 { "2h3h" } else { "3h2h" });
        }
        while gote.len() < sente.len() {
            gote.push(if gote.len() % 2 == 0 { "8b7b" } else { "7b8b" });
        }
        for (s, g) in sente.iter().zip(&gote) {
            game.step(act(&game, s)).unwrap();
            game.step(act(&game, g)).unwrap();
        }
        game.step(act(&game, "8h4d")).unwrap();
        assert_eq!(game.referee().fouls(Color::Sente), 1);
        assert_eq!((game.fouls_early(), game.fouls_precontact()), ([0, 0], [1, 0]));
    }

    #[test]
    fn 予定の手数でやめ_評価モードの相手側も強制できる() {
        let line = ["7g7f", "6g6f", "2h6h", "5i4h"];
        let mut game = EnvGame::with_forced(None, None, 0, [Some(plan(&line, 2)), None]);
        game.step(act(&game, "3c3d")).unwrap();
        game.step(act(&game, "4c4d")).unwrap();
        assert_eq!(game.to_move(), Color::Sente);
        assert_eq!(attempts(&game, Color::Sente), ["7g7f", "6g6f"]);
        assert_eq!(game.forced_stats()[0].stop, Some("length"));

        // 評価モード: 学習側が先手、相手（heuristic）の後手を強制する。相手の最初の手はラインどおり
        let opp = crate::strategy::make("heuristic").unwrap();
        let mut game = EnvGame::with_forced(Some(Color::Sente), Some(opp), 0, [None, Some(plan(&line, 4))]);
        let mut rng = StdRng::seed_from_u64(12);
        for _ in 0..4 {
            assert_eq!(game.to_move(), Color::Sente);
            let a = random_action(game.mask(), rng.random()).unwrap();
            game.step(a).unwrap();
        }
        let got = attempts(&game, Color::Gote);
        if game.contact_ply().is_none() {
            assert_eq!(got[..4], ["3c3d", "4c4d", "8b4b", "5a6b"]);
        }
        assert!(got.len() >= 4);
    }

    #[test]
    fn 強制しない局はこれまでと同じ() {
        for g in 0..6 {
            let mut a = EnvGame::selfplay(None, g);
            let mut b = EnvGame::with_forced(None, None, g, [None, None]);
            let (mut ra, mut rb) = (StdRng::seed_from_u64(g.into()), StdRng::seed_from_u64(g.into()));
            let (oa, ob) = (play_random(&mut a, &mut ra), play_random(&mut b, &mut rb));
            assert_eq!(oa, ob);
            for side in [Color::Sente, Color::Gote] {
                assert!(crate::rl::records::same_observations(
                    a.referee().log(side).events(),
                    b.referee().log(side).events()
                ));
            }
            assert_eq!(a.forced_stats(), [ForcedStats::default(); 2]);
        }
    }

    #[test]
    fn 終局済みの開始局面はすぐ終局する() {
        let start = StartState {
            pos: crate::shogi::Position::initial(),
            logs: Default::default(),
            fouls: [0, 0],
            plies: crate::selfplay::MAX_PLIES,
        };
        let game = EnvGame::selfplay(Some(start), 0);
        assert_eq!(game.outcome().map(|o| o.reason), Some("max_plies"));
    }
}
