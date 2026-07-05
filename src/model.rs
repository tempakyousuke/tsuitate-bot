//! 観測履歴（observation.rs）から対局の時系列モデルを再構成する。
//!
//! sync でもらう PlayerView に頼らず、観測だけから
//! 「自駒の現配置・持ち駒・自分の着手列・相手の手数・取られた駒」を導出する。
//! 本番では sync の PlayerView と照合し、ズレ（切断中の取りこぼし等）を検出する。
//! 推定器（estimator.rs）は生の観測列と本モデルの両方を部品として使う。

use std::collections::HashMap;

use crate::board::{Coord, make_usi_square, parse_usi_square};
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{Color, PlayerView, Role, VisiblePiece};
use crate::shogi::{Position, ShogiMove, hand_index, parse_usi, promote_role, HAND_ROLES};

#[derive(Debug, Clone)]
pub struct GameModel {
    my_color: Color,
    /// 自駒の現配置
    pieces: HashMap<Coord, Role>,
    /// 持ち駒（HAND_ROLES 順）
    hand: [u8; 7],
    /// 受理された自分の着手（USI、時系列順）
    my_moves: Vec<String>,
    /// 相手の着手回数
    opponent_moves: u32,
    /// 取られた自駒（マスと、取られた時点の駒種）
    lost_pieces: Vec<(Coord, Role)>,
    my_fouls: u32,
    opponent_fouls: u32,
    /// 再構成中に観測と矛盾を検出したら false（以後の値は信用できない）
    consistent: bool,
}

impl GameModel {
    pub fn new(my_color: Color) -> Self {
        let initial = Position::initial();
        let pieces = initial
            .pieces_of(my_color)
            .iter()
            .filter_map(|p| parse_usi_square(&p.square).map(|c| (c, p.role)))
            .collect();
        GameModel {
            my_color,
            pieces,
            hand: [0; 7],
            my_moves: vec![],
            opponent_moves: 0,
            lost_pieces: vec![],
            my_fouls: 0,
            opponent_fouls: 0,
            consistent: true,
        }
    }

    pub fn from_log(my_color: Color, log: &ObservationLog) -> Self {
        let mut model = GameModel::new(my_color);
        for event in log.events() {
            model.apply(event);
        }
        model
    }

    pub fn apply(&mut self, event: &Observation) {
        match event {
            Observation::MyMove { usi, captured, .. } => {
                match parse_usi(usi) {
                    Some(ShogiMove::Board { from, to, promote }) => {
                        match self.pieces.remove(&from) {
                            Some(role) => {
                                let role = if promote {
                                    promote_role(role).unwrap_or(role)
                                } else {
                                    role
                                };
                                self.pieces.insert(to, role);
                            }
                            None => self.consistent = false,
                        }
                    }
                    Some(ShogiMove::Drop { role, to }) => {
                        match hand_index(role) {
                            Some(i) if self.hand[i] > 0 => {
                                self.hand[i] -= 1;
                                self.pieces.insert(to, role);
                            }
                            _ => self.consistent = false,
                        }
                    }
                    None => self.consistent = false,
                }
                if let Some(role) = captured {
                    match hand_index(*role) {
                        Some(i) => self.hand[i] += 1,
                        None => self.consistent = false,
                    }
                }
                self.my_moves.push(usi.clone());
            }
            Observation::OpponentMoved {
                captured_my_piece_at,
                ..
            } => {
                self.opponent_moves += 1;
                if let Some(sq) = captured_my_piece_at {
                    match parse_usi_square(sq).and_then(|c| self.pieces.remove(&c).map(|r| (c, r)))
                    {
                        Some(lost) => self.lost_pieces.push(lost),
                        None => self.consistent = false,
                    }
                }
            }
            Observation::MyFoul { .. } => self.my_fouls += 1,
            Observation::OpponentFoul { count } => self.opponent_fouls = *count,
            Observation::Check { .. } => {}
        }
    }

    pub fn my_color(&self) -> Color {
        self.my_color
    }

    pub fn consistent(&self) -> bool {
        self.consistent
    }

    pub fn my_pieces(&self) -> Vec<VisiblePiece> {
        self.pieces
            .iter()
            .map(|(&c, &role)| VisiblePiece {
                square: make_usi_square(c),
                role,
            })
            .collect()
    }

    pub fn my_hand(&self) -> HashMap<Role, u32> {
        HAND_ROLES
            .iter()
            .enumerate()
            .filter(|&(i, _)| self.hand[i] > 0)
            .map(|(i, &role)| (role, self.hand[i] as u32))
            .collect()
    }

    pub fn my_moves(&self) -> &[String] {
        &self.my_moves
    }

    pub fn opponent_moves(&self) -> u32 {
        self.opponent_moves
    }

    pub fn lost_pieces(&self) -> &[(Coord, Role)] {
        &self.lost_pieces
    }

    pub fn my_fouls(&self) -> u32 {
        self.my_fouls
    }

    pub fn opponent_fouls(&self) -> u32 {
        self.opponent_fouls
    }

    /// 相手の持ち駒（取られた自駒の成りを戻したもの）。
    /// ついたて将棋では相手の持ち駒は観測から完全に決まる
    pub fn opponent_hand(&self) -> HashMap<Role, u32> {
        let mut hand: HashMap<Role, u32> = HashMap::new();
        for (_, role) in &self.lost_pieces {
            *hand.entry(crate::shogi::unpromote_role(*role)).or_insert(0) += 1;
        }
        hand
    }

    /// sync の PlayerView と照合し、ズレの説明文を返す（一致なら None）
    pub fn diff_view(&self, view: &PlayerView) -> Option<String> {
        let mut diffs = vec![];
        let mut view_pieces: Vec<(String, Role)> = view
            .your_pieces
            .iter()
            .map(|p| (p.square.clone(), p.role))
            .collect();
        view_pieces.sort();
        let mut model_pieces: Vec<(String, Role)> = self
            .my_pieces()
            .iter()
            .map(|p| (p.square.clone(), p.role))
            .collect();
        model_pieces.sort();
        if view_pieces != model_pieces {
            diffs.push(format!(
                "盤上: view={view_pieces:?} model={model_pieces:?}"
            ));
        }
        let view_hand: HashMap<Role, u32> = view
            .your_hand
            .iter()
            .filter(|&(_, &n)| n > 0)
            .map(|(&r, &n)| (r, n))
            .collect();
        if view_hand != self.my_hand() {
            diffs.push(format!(
                "持ち駒: view={view_hand:?} model={:?}",
                self.my_hand()
            ));
        }
        if view.fouls.you != self.my_fouls {
            diffs.push(format!(
                "反則数: view={} model={}",
                view.fouls.you, self.my_fouls
            ));
        }
        if diffs.is_empty() { None } else { Some(diffs.join(" / ")) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shogi::unpromote_role;

    /// 台本どおりにフル盤面で対局を進め、両者の観測ログを生成して
    /// GameModel の再構成が真の局面（の自分側射影）と一致することを確かめる
    #[test]
    fn reconstruction_matches_full_simulation() {
        // 角交換を含む数手（すべて合法手）
        let script = [
            "7g7f", "3c3d", "8h2b+", // 先手が角を取って成る
            "3a2b", // 後手が馬を銀で取り返す
            "B*5e", // 先手が取った角を打つ
            "8c8d",
        ];
        let mut pos = Position::initial();
        let mut logs = HashMap::from([
            (Color::Sente, ObservationLog::default()),
            (Color::Gote, ObservationLog::default()),
        ]);
        for usi in script {
            let mover = pos.turn();
            let mv = parse_usi(usi).unwrap();
            assert!(pos.is_legal(&mv), "台本の手が非合法: {usi}");
            let captured = pos.play_unchecked(&mv);
            let move_number = pos.move_number();
            logs.get_mut(&mover).unwrap().record(Observation::MyMove {
                move_number,
                usi: usi.into(),
                captured: captured.map(unpromote_role),
            });
            let captured_at = captured.map(|_| match mv {
                ShogiMove::Board { to, .. } => make_usi_square(to),
                ShogiMove::Drop { .. } => unreachable!(),
            });
            logs.get_mut(&mover.other())
                .unwrap()
                .record(Observation::OpponentMoved {
                    move_number,
                    captured_my_piece_at: captured_at,
                });
        }

        for color in [Color::Sente, Color::Gote] {
            let model = GameModel::from_log(color, &logs[&color]);
            assert!(model.consistent(), "{color:?} のモデルが矛盾");
            let mut expect: Vec<(String, Role)> = pos
                .pieces_of(color)
                .iter()
                .map(|p| (p.square.clone(), p.role))
                .collect();
            expect.sort();
            let mut got: Vec<(String, Role)> = model
                .my_pieces()
                .iter()
                .map(|p| (p.square.clone(), p.role))
                .collect();
            got.sort();
            assert_eq!(got, expect, "{color:?} の盤上再構成が不一致");
            assert_eq!(model.my_hand(), pos.hand_map(color), "{color:?} の持ち駒が不一致");
        }

        // 先手: 角得で馬を失った → 持ち駒は角1（5eに打ったので0）
        let sente = GameModel::from_log(Color::Sente, &logs[&Color::Sente]);
        assert_eq!(sente.my_hand().get(&Role::Bishop), None);
        assert_eq!(sente.opponent_moves(), 3);
        assert_eq!(sente.lost_pieces().len(), 1); // 馬（2bの成り角）を取られた
        assert_eq!(sente.lost_pieces()[0].1, Role::Horse);
        // 相手（後手）の持ち駒は馬→角
        assert_eq!(sente.opponent_hand().get(&Role::Bishop), Some(&1));
    }

    #[test]
    fn foul_and_check_events_are_counted() {
        let mut log = ObservationLog::default();
        log.record(Observation::MyFoul {
            move_number: 1,
            usi: "8h2b+".into(),
        });
        log.record(Observation::OpponentFoul { count: 2 });
        log.record(Observation::Check {
            in_check: Color::Sente,
        });
        let model = GameModel::from_log(Color::Sente, &log);
        assert!(model.consistent());
        assert_eq!(model.my_fouls(), 1);
        assert_eq!(model.opponent_fouls(), 2);
        // 反則では盤面は変わらない
        assert_eq!(model.my_pieces().len(), 20);
    }

    #[test]
    fn inconsistent_log_is_flagged() {
        let mut log = ObservationLog::default();
        // 駒がないマスからの移動
        log.record(Observation::MyMove {
            move_number: 1,
            usi: "5e5d".into(),
            captured: None,
        });
        let model = GameModel::from_log(Color::Sente, &log);
        assert!(!model.consistent());
    }
}
