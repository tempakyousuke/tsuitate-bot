//! 手単位で進める審判（arena の対局ループと RL 環境が共有する裁定）。
//!
//! サーバー（judge.ts / game-room.ts）と同じ裁定:
//! - 反則（フル盤面で非合法な手）は手番を変えずカウント。`MAX_FOULS` で反則負け
//! - 受理なら、指した側へ `MyMove`（取った駒種）、相手へ `OpponentMoved`（取られたマス）、
//!   王手なら両者へ `Check`
//! - 詰み・ステイルメイト・手数上限で終局
//!
//! 時計・投了・診断用オラクルは**対局者側の都合**なので持たない（`selfplay.rs` が持つ）。
//! RL 環境は時計を持たず、この型だけで1局を進める（docs/rl-deepnash-design.md）。

use std::collections::HashSet;

use crate::observation::{Observation, ObservationLog};
use crate::protocol::{
    ClockState, Color, FoulCounts, FoulRecord, GameStatus, MoveRecord, PlayerView,
};
use crate::selfplay::{GameResult, GameTruth, MAX_FOULS, MAX_PLIES, StartState};
use crate::shogi::{Outcome, Position, ShogiMove, parse_usi, unpromote_role};

fn idx(c: Color) -> usize {
    if c == Color::Sente { 0 } else { 1 }
}

/// `Referee::step` の結果
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepResult {
    /// 受理された。手番は相手へ移った
    Accepted,
    /// 反則。手番は変わらない
    Foul,
    /// 終局。`move_accepted` はこの手が受理されたか（詰み・ステイルメイト・手数上限は true、
    /// 反則負けは false。**受理された終局手にもフィッシャー加算が付く**ので呼び出し側が区別する。
    /// 終局済みの審判へ `step` した場合も false）
    Ended {
        result: GameResult,
        reason: &'static str,
        move_accepted: bool,
    },
}

pub struct Referee {
    pos: Position,
    plies: u32,
    logs: [ObservationLog; 2],
    fouls: [u32; 2],
    fouls_in_check: [u32; 2],
    /// その手番で反則した手（受理されたら空に戻る）
    foul_tried: [HashSet<String>; 2],
    truth: GameTruth,
    /// 終局していればその結果（以後の `step` は何も記録せずこれを返す）
    ended: Option<(GameResult, &'static str)>,
}

impl Referee {
    /// 初期局面から
    pub fn new() -> Self {
        Self::from_start(StartState {
            pos: Position::initial(),
            logs: [ObservationLog::default(), ObservationLog::default()],
            fouls: [0, 0],
            plies: 0,
        })
    }

    /// 途中局面から（checkpoint arena・RL の指し継ぎ）。`StartState` は手番境界
    /// （反則をまだ試していない時点）なので `foul_tried` は空で始める。
    ///
    /// **開始時点で終局済みの局面も受け付ける**（RL の指し継ぎは終局そのものを開始点に
    /// 選びうる）。その場合は `ended()` が Some を返し、`step` は何も記録しない。
    /// 判定の優先は詰み・ステイルメイト > 反則負け > 手数上限
    pub fn from_start(start: StartState) -> Self {
        let StartState {
            pos,
            logs,
            fouls,
            plies,
        } = start;
        Self {
            pos,
            plies,
            logs,
            fouls,
            fouls_in_check: [0, 0],
            foul_tried: [HashSet::new(), HashSet::new()],
            truth: GameTruth {
                moves: vec![],
                foul_attempts: vec![],
            },
            ended: None,
        }
        .with_start_terminal()
    }

    fn with_start_terminal(mut self) -> Self {
        self.ended = match self.pos.outcome() {
            Some(Outcome::Checkmate { winner }) => Some((GameResult::Win(winner), "checkmate")),
            Some(Outcome::Stalemate { winner }) => Some((GameResult::Win(winner), "stalemate")),
            None => [Color::Sente, Color::Gote]
                .into_iter()
                .find(|&c| self.fouls[idx(c)] >= MAX_FOULS)
                .map(|c| (GameResult::Win(c.other()), "foul_limit"))
                .or_else(|| {
                    self.max_plies_reached()
                        .then_some((GameResult::Draw, "max_plies"))
                }),
        };
        self
    }

    /// 終局していればその結果
    pub fn ended(&self) -> Option<(GameResult, &'static str)> {
        self.ended
    }

    pub fn to_move(&self) -> Color {
        self.pos.turn()
    }

    /// 真実の盤面（審判と学習時の補助教師だけが見てよい）
    pub fn position(&self) -> &Position {
        &self.pos
    }

    pub fn plies(&self) -> u32 {
        self.plies
    }

    pub fn log(&self, color: Color) -> &ObservationLog {
        &self.logs[idx(color)]
    }

    pub fn foul_tried(&self, color: Color) -> &HashSet<String> {
        &self.foul_tried[idx(color)]
    }

    pub fn fouls(&self, color: Color) -> u32 {
        self.fouls[idx(color)]
    }

    /// 開始以降に王手を受けている局面でした反則の数
    pub fn fouls_in_check(&self, color: Color) -> u32 {
        self.fouls_in_check[idx(color)]
    }

    /// 手数上限に達しているか（途中局面から始めた直後にも成り立ちうる）
    pub fn max_plies_reached(&self) -> bool {
        self.plies >= MAX_PLIES
    }

    /// 手番側から見た視界。`clocks_ms` は [先手, 後手]（時計を持たない RL 環境は任意の値でよい。
    /// 戦略は clocks を読まない）
    pub fn view(&self, color: Color, clocks_ms: [i64; 2], game_no: u32) -> PlayerView {
        let pos = &self.pos;
        PlayerView {
            game_id: format!("arena-{game_no}"),
            your_color: color,
            your_pieces: pos.pieces_of(color),
            your_hand: pos.hand_map(color),
            turn: pos.turn(),
            move_number: pos.move_number(),
            clocks: ClockState {
                sente_ms: clocks_ms[0],
                gote_ms: clocks_ms[1],
                running: Some(pos.turn()),
                server_time: 0,
            },
            fouls: FoulCounts {
                you: self.fouls[idx(color)],
                opponent: self.fouls[idx(color.other())],
            },
            you_in_check: pos.in_check(color),
            opponent_in_check: pos.in_check(color.other()),
            status: GameStatus::Playing,
        }
    }

    /// 手番側がこの手を指したら受理されるか（診断用オラクルの判定用。状態は変えない）
    pub fn is_legal(&self, usi: &str) -> bool {
        parse_usi(usi).is_some_and(|mv| self.pos.is_legal(&mv))
    }

    /// 手番側の `foul_tried` にだけ積む（診断用オラクルが反則を握りつぶすとき。
    /// 反則カウントも観測も発生しない）
    pub fn suppress_foul(&mut self, usi: String) {
        let side = self.pos.turn();
        self.foul_tried[idx(side)].insert(usi);
    }

    /// 手番側の手を裁定する。`think_ms` は真実の記録（`MoveRecord.ms`）に残すだけ
    pub fn step(&mut self, usi: &str, think_ms: u64) -> StepResult {
        if let Some((result, reason)) = self.ended {
            return StepResult::Ended {
                result,
                reason,
                move_accepted: false,
            };
        }
        let side = self.pos.turn();
        let Some(mv) = parse_usi(usi).filter(|mv| self.pos.is_legal(mv)) else {
            return self.foul(side, usi);
        };
        self.accept(side, usi, mv, think_ms)
    }

    fn foul(&mut self, side: Color, usi: &str) -> StepResult {
        let i = idx(side);
        self.fouls[i] += 1;
        if self.pos.in_check(side) {
            self.fouls_in_check[i] += 1;
        }
        self.foul_tried[i].insert(usi.to_string());
        self.truth.foul_attempts.push(FoulRecord {
            move_number: self.pos.move_number(),
            by_color: side,
            usi: usi.to_string(),
        });
        self.logs[i].record(Observation::MyFoul {
            move_number: self.pos.move_number(),
            usi: usi.to_string(),
        });
        let count = self.fouls[i];
        self.logs[idx(side.other())]
            .record(Observation::OpponentFoul { count });
        if count >= MAX_FOULS {
            return self.end(GameResult::Win(side.other()), "foul_limit", false);
        }
        StepResult::Foul
    }

    fn accept(&mut self, side: Color, usi: &str, mv: ShogiMove, think_ms: u64) -> StepResult {
        let i = idx(side);
        self.truth.moves.push(MoveRecord {
            usi: usi.to_string(),
            by_color: side,
            ms: think_ms,
            fouls_before: self.fouls[i],
        });
        let captured = self.pos.play_unchecked(&mv);
        self.plies += 1;
        self.foul_tried[i].clear();

        // 通知（game-room.ts と同じ内容・同じ moveNumber 規約 = 適用後の値）
        let move_number = self.pos.move_number();
        let captured_square = captured.map(|_| match mv {
            ShogiMove::Board { to, .. } => crate::board::make_usi_square(to),
            ShogiMove::Drop { .. } => unreachable!("打ちでは駒を取れない"),
        });
        self.logs[i].record(Observation::MyMove {
            move_number,
            usi: usi.to_string(),
            captured: captured.map(unpromote_role),
        });
        self.logs[idx(side.other())].record(Observation::OpponentMoved {
            move_number,
            captured_my_piece_at: captured_square,
        });
        if self.pos.in_check(self.pos.turn()) {
            let in_check = self.pos.turn();
            for log in self.logs.iter_mut() {
                log.record(Observation::Check { in_check });
            }
        }

        match self.pos.outcome() {
            Some(Outcome::Checkmate { winner }) => {
                self.end(GameResult::Win(winner), "checkmate", true)
            }
            Some(Outcome::Stalemate { winner }) => {
                self.end(GameResult::Win(winner), "stalemate", true)
            }
            None if self.max_plies_reached() => self.end(GameResult::Draw, "max_plies", true),
            None => StepResult::Accepted,
        }
    }

    fn end(&mut self, result: GameResult, reason: &'static str, move_accepted: bool) -> StepResult {
        self.ended = Some((result, reason));
        StepResult::Ended {
            result,
            reason,
            move_accepted,
        }
    }

    /// 真実の記録（全手順・反則試行）を取り出す
    pub fn take_truth(&mut self) -> GameTruth {
        std::mem::replace(
            &mut self.truth,
            GameTruth {
                moves: vec![],
                foul_attempts: vec![],
            },
        )
    }
}

impl Default for Referee {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::parse_usi_square;
    use crate::protocol::Role;
    use crate::shogi::Piece;

    fn start(pos: Position, fouls: [u32; 2], plies: u32) -> StartState {
        StartState {
            pos,
            logs: [ObservationLog::default(), ObservationLog::default()],
            fouls,
            plies,
        }
    }

    /// 後手玉 5一 が先手の金 5二・5三 で詰んでいる局面（後手番）
    fn mated() -> Position {
        let sq = |s: &str| parse_usi_square(s).unwrap();
        let piece = |color, role| Some(Piece { color, role });
        let mut pos = Position::empty(Color::Gote);
        pos.set(sq("5i"), piece(Color::Sente, Role::King));
        pos.set(sq("5a"), piece(Color::Gote, Role::King));
        pos.set(sq("5b"), piece(Color::Sente, Role::Gold));
        pos.set(sq("5c"), piece(Color::Sente, Role::Gold));
        pos
    }

    #[test]
    fn 詰み済みの局面から始めると終局として扱う() {
        let mut referee = Referee::from_start(start(mated(), [0, 0], 30));
        assert_eq!(
            referee.ended(),
            Some((GameResult::Win(Color::Sente), "checkmate"))
        );
        // 終局後の step は何も記録しない（反則にも数えない）
        let r = referee.step("5a4a", 0);
        assert_eq!(
            r,
            StepResult::Ended {
                result: GameResult::Win(Color::Sente),
                reason: "checkmate",
                move_accepted: false
            }
        );
        assert_eq!(referee.fouls(Color::Gote), 0);
        assert!(referee.log(Color::Gote).events().is_empty());
        assert_eq!(referee.plies(), 30);
    }

    #[test]
    fn 手数上限と反則上限に達した開始局面も終局として扱う() {
        let r = Referee::from_start(start(Position::initial(), [0, 0], MAX_PLIES));
        assert_eq!(r.ended(), Some((GameResult::Draw, "max_plies")));
        let r = Referee::from_start(start(Position::initial(), [0, MAX_FOULS], 40));
        assert_eq!(r.ended(), Some((GameResult::Win(Color::Sente), "foul_limit")));
        let r = Referee::from_start(start(Position::initial(), [3, 4], 40));
        assert_eq!(r.ended(), None);
    }
}
