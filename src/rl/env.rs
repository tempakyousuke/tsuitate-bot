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

use crate::protocol::Color;
use crate::referee::{Referee, StepResult};
use crate::rl::action::{NUM_ACTIONS, decode_usi, legal_mask};
use crate::rl::encode::{OBS_LEN, encode_into};
use crate::selfplay::{GameResult, StartState, fischer_initial_ms};
use crate::strategy::Strategy;

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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvError {
    /// 行動番号が範囲外、またはマスクの外（学習側のマスク適用漏れ）
    MaskedAction(usize),
    /// 終局済みの局に行動を渡した
    GameOver,
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
}

impl EnvGame {
    /// 自己対局の1局。`start` が None なら初期局面から
    pub fn selfplay(start: Option<StartState>, game_no: u32) -> Self {
        Self::build(start, None, None, game_no)
    }

    /// 評価モードの1局。`opponent` が `learner` の相手側を指す
    pub fn versus(
        start: Option<StartState>,
        learner: Color,
        opponent: Box<dyn Strategy + Send>,
        game_no: u32,
    ) -> Self {
        Self::build(start, Some(learner), Some(opponent), game_no)
    }

    fn build(
        start: Option<StartState>,
        learner: Option<Color>,
        opponent: Option<Box<dyn Strategy + Send>>,
        game_no: u32,
    ) -> Self {
        let referee = start.map_or_else(Referee::new, Referee::from_start);
        let mut game = Self {
            referee,
            learner,
            opponent,
            game_no,
            mask: vec![],
            outcome: None,
        };
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

    /// 今の手番の観測を書く（`obs` は長さ `OBS_LEN`）
    pub fn observe_into(&self, obs: &mut [f32]) {
        debug_assert_eq!(obs.len(), OBS_LEN);
        let side = self.referee.to_move();
        let view = self.referee.view(side, clocks(), self.game_no);
        encode_into(&view, self.referee.log(side), self.referee.foul_tried(side), obs);
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
        if let StepResult::Ended { result, reason, .. } = self.referee.step(&usi, 0) {
            self.finish(result, reason);
        } else {
            self.settle();
        }
        Ok(self.outcome)
    }

    /// 相手（Rust 戦略）の手番を指し進め、学習側の手番で止めてマスクを作る
    fn settle(&mut self) {
        loop {
            if let Some((result, reason)) = self.referee.ended() {
                self.finish(result, reason);
                return;
            }
            let side = self.referee.to_move();
            let learner_turn = self.learner.is_none_or(|l| l == side);
            if learner_turn {
                let view = self.referee.view(side, clocks(), self.game_no);
                self.mask = legal_mask(&view, self.referee.foul_tried(side));
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
            if let StepResult::Ended { result, reason, .. } = self.referee.step(&usi, 0) {
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
