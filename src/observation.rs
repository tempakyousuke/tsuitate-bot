//! 対局中の観測履歴。
//!
//! ついたて将棋で自分が得られる情報は
//! - 自分の指し手が「受理された/反則だった」（反則理由は不明）
//! - 取った駒の種類 / 自駒が取られたマス
//! - 王手宣言・相手の反則宣言
//! がすべて。これが bot が得る情報の全量で、`GameModel` が自駒配置を再構成し、
//! `Estimator` が相手局面の粒子集合を維持する入力になる。
//! 対局記録（`record.rs`）と終局サマリも同じ履歴を書く。

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::board::Coord;
use crate::protocol::{Color, Role};
use crate::shogi::{parse_usi, ShogiMove};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Observation {
    /// 受理された自分の指し手
    MyMove {
        move_number: u32,
        usi: String,
        captured: Option<Role>,
    },
    /// 反則になった自分の指し手（手番は変わっていない）
    MyFoul { move_number: u32, usi: String },
    /// 相手の着手（内容は不明）
    OpponentMoved {
        move_number: u32,
        captured_my_piece_at: Option<String>,
    },
    /// 相手の反則宣言
    OpponentFoul { count: u32 },
    /// 王手宣言
    Check { in_check: Color },
}

#[derive(Debug, Default)]
pub struct ObservationLog {
    events: Vec<Observation>,
}

impl ObservationLog {
    pub fn record(&mut self, obs: Observation) {
        self.events.push(obs);
    }

    pub fn clear(&mut self) {
        self.events.clear();
    }

    pub fn events(&self) -> &[Observation] {
        &self.events
    }

    /// 終局時のサマリ（デバッグ用）
    pub fn summary(&self) -> String {
        let my_moves = self
            .events
            .iter()
            .filter(|e| matches!(e, Observation::MyMove { .. }))
            .count();
        let my_fouls = self
            .events
            .iter()
            .filter(|e| matches!(e, Observation::MyFoul { .. }))
            .count();
        let captures = self
            .events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    Observation::MyMove {
                        captured: Some(_),
                        ..
                    }
                )
            })
            .count();
        let lost = self
            .events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    Observation::OpponentMoved {
                        captured_my_piece_at: Some(_),
                        ..
                    }
                )
            })
            .count();
        format!(
            "自分の着手 {my_moves}（うち駒取り {captures}）/ 反則 {my_fouls} / 取られた駒 {lost}"
        )
    }
}

/// 初期玉位置（先手 5i・後手 5a）
pub(crate) fn initial_king_sq(color: Color) -> Coord {
    match color {
        Color::Sente => Coord { file: 5, rank: 9 },
        Color::Gote => Coord { file: 5, rank: 1 },
    }
}

/// 過去手番で玉が反則した行き先のうち、解除証拠がまだ無いマス。
///
/// 同手番の同一 USI は `foul_tried` が既に除外する。ここは**手番をまたいだ**
/// 再訪だけを対象にする（`current_move_number` の `MyFoul` は無視）。
///
/// 解除（安全方向の例外。観測できたときだけ汚名を落とす）:
/// 1. そのマスで駒を取った（居た駒が無くなった）
/// 2. その後、玉が実際にそのマスへ受理された
///
/// 相手の静かな移動・飛角の路線遮断は観測できないので解除しない。
/// 「推測材料が少ないのに同じ場所へ玉を動かすのは危険」が前提。
pub fn stale_king_foul_dests(
    log: &ObservationLog,
    color: Color,
    current_move_number: u32,
) -> HashSet<Coord> {
    let mut king = initial_king_sq(color);
    let mut fouled: HashSet<Coord> = HashSet::new();
    let mut lifted: HashSet<Coord> = HashSet::new();
    for e in log.events() {
        match e {
            Observation::MyFoul { move_number, usi } if *move_number != current_move_number => {
                if let Some(ShogiMove::Board { from, to, .. }) = parse_usi(usi) {
                    if from == king {
                        fouled.insert(to);
                        lifted.remove(&to);
                    }
                }
            }
            Observation::MyMove { usi, captured, .. } => {
                if let Some(ShogiMove::Board { from, to, .. }) = parse_usi(usi) {
                    if captured.is_some() {
                        lifted.insert(to);
                    }
                    if from == king {
                        lifted.insert(to);
                        king = to;
                    }
                }
            }
            _ => {}
        }
    }
    fouled.into_iter().filter(|d| !lifted.contains(d)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::parse_usi_square;

    fn sq(usi: &str) -> Coord {
        parse_usi_square(usi).unwrap()
    }

    #[test]
    fn 手番をまたいだ玉反則の行き先は汚名が残る() {
        let mut log = ObservationLog::default();
        log.record(Observation::MyFoul {
            move_number: 10,
            usi: "5i5h".into(),
        });
        log.record(Observation::OpponentMoved {
            move_number: 11,
            captured_my_piece_at: None,
        });
        let stale = stale_king_foul_dests(&log, Color::Sente, 12);
        assert_eq!(stale, HashSet::from([sq("5h")]));
    }

    #[test]
    fn 今手番の玉反則は汚名に入れない() {
        let mut log = ObservationLog::default();
        log.record(Observation::MyFoul {
            move_number: 12,
            usi: "5i5h".into(),
        });
        let stale = stale_king_foul_dests(&log, Color::Sente, 12);
        assert!(stale.is_empty());
    }

    #[test]
    fn 行き先で取れば汚名は消える() {
        let mut log = ObservationLog::default();
        log.record(Observation::MyFoul {
            move_number: 10,
            usi: "5i5h".into(),
        });
        log.record(Observation::OpponentMoved {
            move_number: 11,
            captured_my_piece_at: None,
        });
        log.record(Observation::MyMove {
            move_number: 12,
            usi: "4i5h".into(),
            captured: Some(Role::Pawn),
        });
        let stale = stale_king_foul_dests(&log, Color::Sente, 13);
        assert!(stale.is_empty());
    }

    #[test]
    fn 玉がそのマスへ受理されれば汚名は消える() {
        let mut log = ObservationLog::default();
        log.record(Observation::MyFoul {
            move_number: 10,
            usi: "5i6h".into(),
        });
        log.record(Observation::OpponentMoved {
            move_number: 11,
            captured_my_piece_at: None,
        });
        log.record(Observation::MyMove {
            move_number: 12,
            usi: "5i6h".into(),
            captured: None,
        });
        let stale = stale_king_foul_dests(&log, Color::Sente, 13);
        assert!(stale.is_empty());
    }

    #[test]
    fn 無関係な相手の捕獲では汚名は消えない() {
        let mut log = ObservationLog::default();
        log.record(Observation::MyFoul {
            move_number: 10,
            usi: "5i5h".into(),
        });
        log.record(Observation::OpponentMoved {
            move_number: 11,
            captured_my_piece_at: Some("7g".into()),
        });
        let stale = stale_king_foul_dests(&log, Color::Sente, 12);
        assert_eq!(stale, HashSet::from([sq("5h")]));
    }

    #[test]
    fn 後手の初期玉からも行き先を追う() {
        let mut log = ObservationLog::default();
        log.record(Observation::MyFoul {
            move_number: 9,
            usi: "5a5b".into(),
        });
        log.record(Observation::OpponentMoved {
            move_number: 10,
            captured_my_piece_at: None,
        });
        let stale = stale_king_foul_dests(&log, Color::Gote, 11);
        assert_eq!(stale, HashSet::from([sq("5b")]));
    }

    #[test]
    fn 解除後の再反則は再び汚名になる() {
        let mut log = ObservationLog::default();
        log.record(Observation::MyFoul {
            move_number: 10,
            usi: "5i5h".into(),
        });
        log.record(Observation::MyMove {
            move_number: 12,
            usi: "4i5h".into(),
            captured: Some(Role::Silver),
        });
        log.record(Observation::OpponentMoved {
            move_number: 13,
            captured_my_piece_at: None,
        });
        log.record(Observation::MyFoul {
            move_number: 14,
            usi: "5i5h".into(),
        });
        log.record(Observation::OpponentMoved {
            move_number: 15,
            captured_my_piece_at: None,
        });
        let stale = stale_king_foul_dests(&log, Color::Sente, 16);
        assert_eq!(stale, HashSet::from([sq("5h")]));
    }

    #[test]
    fn 玉以外の反則は汚名にしない() {
        let mut log = ObservationLog::default();
        log.record(Observation::MyFoul {
            move_number: 10,
            usi: "2g2f".into(),
        });
        log.record(Observation::MyFoul {
            move_number: 10,
            usi: "P*5e".into(),
        });
        let stale = stale_king_foul_dests(&log, Color::Sente, 11);
        assert!(stale.is_empty());
    }

    /// 視聴棋譜: 4八玉が 3七玉 で反則したあと、相手桂が 4五→3七→2九成 と通過し
    /// 2九で成桂を取り返した局面。3七では何も取っておらず玉も受理されていないので、
    /// 原因の桂が盤上から消えても汚名は残る。
    #[test]
    fn 桂が3七を通過して2九で取られても3七の汚名は消えない() {
        let mut log = ObservationLog::default();
        log.record(Observation::MyMove {
            move_number: 1,
            usi: "7g7f".into(),
            captured: None,
        });
        log.record(Observation::OpponentMoved {
            move_number: 2,
            captured_my_piece_at: None,
        });
        log.record(Observation::MyMove {
            move_number: 3,
            usi: "5i4h".into(),
            captured: None,
        });
        log.record(Observation::OpponentMoved {
            move_number: 4,
            captured_my_piece_at: None,
        });
        log.record(Observation::MyMove {
            move_number: 5,
            usi: "3g3f".into(),
            captured: None,
        });
        log.record(Observation::OpponentMoved {
            move_number: 6,
            captured_my_piece_at: None,
        });
        log.record(Observation::MyFoul {
            move_number: 7,
            usi: "4h3g".into(),
        });
        log.record(Observation::MyMove {
            move_number: 7,
            usi: "4g4f".into(),
            captured: None,
        });
        log.record(Observation::OpponentMoved {
            move_number: 8,
            captured_my_piece_at: None,
        });
        log.record(Observation::MyMove {
            move_number: 9,
            usi: "4f4e".into(),
            captured: None,
        });
        log.record(Observation::OpponentMoved {
            move_number: 10,
            captured_my_piece_at: Some("2i".into()),
        });
        log.record(Observation::MyMove {
            move_number: 11,
            usi: "2h2i".into(),
            captured: Some(Role::Promotedknight),
        });
        let stale = stale_king_foul_dests(&log, Color::Sente, 13);
        assert_eq!(stale, HashSet::from([sq("3g")]));
    }
}
