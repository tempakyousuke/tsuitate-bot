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
use crate::protocol::Color;
use crate::referee::{Referee, StepResult};
use crate::rl::action::{NUM_ACTIONS, decode_usi, legal_mask};
use crate::rl::encode::{OBS_LEN, encode_into};
use crate::selfplay::{GameResult, StartState, fischer_initial_ms};
use crate::shogi::{ShogiMove, parse_usi};
use crate::strategy::Strategy;

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
    style: StyleStats,
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
            style: StyleStats::default(),
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
        result
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
        if let StepResult::Ended { result, reason, .. } = self.play(&usi) {
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
