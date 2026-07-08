//! 対局中の観測履歴。
//!
//! ついたて将棋で自分が得られる情報は
//! - 自分の指し手が「受理された/反則だった」（反則理由は不明）
//! - 取った駒の種類 / 自駒が取られたマス
//! - 王手宣言・相手の反則宣言
//! がすべて。将来の思考エンジンはこの履歴から「相手局面の情報集合」を構築する。
//! 現フェーズでは記録と終局時のサマリ出力のみ（フィールドは将来のエンジンが読む）。
#![allow(dead_code)]

use serde::Serialize;

use crate::protocol::{Color, Role};

#[derive(Debug, Clone, Serialize)]
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
            .filter(|e| matches!(e, Observation::MyMove { captured: Some(_), .. }))
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
