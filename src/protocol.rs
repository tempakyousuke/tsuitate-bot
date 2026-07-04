//! サイト側のイベント型契約（tsuitate リポジトリ src/lib/shared/events.ts / game-types.ts）の Rust 版。
//!
//! サーバーから届くのは「自分から見える情報」だけ。相手の駒・持ち駒・指し手の内容は含まれない。
//! 契約の完全な写しとして保持するため、現時点で未使用のフィールドも定義している。
#![allow(dead_code)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Color {
    Sente,
    Gote,
}

impl Color {
    pub fn other(self) -> Color {
        match self {
            Color::Sente => Color::Gote,
            Color::Gote => Color::Sente,
        }
    }
}

/// shogiops と同じ駒種名（サイト側 PieceRole と同一の文字列表現）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Pawn,
    Lance,
    Knight,
    Silver,
    Gold,
    Bishop,
    Rook,
    King,
    Tokin,
    Promotedlance,
    Promotedknight,
    Promotedsilver,
    Horse,
    Dragon,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisiblePiece {
    /// USI表記のマス（例: "7g"）
    pub square: String,
    pub role: Role,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpponentInfo {
    pub username: String,
    pub rating: i32,
    #[serde(default)]
    pub is_bot: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClockState {
    pub sente_ms: i64,
    pub gote_ms: i64,
    pub running: Option<Color>,
    pub server_time: i64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FoulCounts {
    pub you: u32,
    pub opponent: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GameStatus {
    Playing,
    Ended,
}

/// 対局中に自分へ送られる可視情報の全量（PlayerView）
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlayerView {
    pub game_id: String,
    pub your_color: Color,
    pub opponent: OpponentInfo,
    pub your_pieces: Vec<VisiblePiece>,
    pub your_hand: HashMap<Role, u32>,
    pub turn: Color,
    pub move_number: u32,
    pub clocks: ClockState,
    pub fouls: FoulCounts,
    pub you_in_check: bool,
    pub opponent_in_check: bool,
    pub status: GameStatus,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MatchFoundPayload {
    pub game_id: String,
    pub your_color: Color,
    pub opponent: OpponentInfo,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MoveAcceptedPayload {
    pub move_number: u32,
    /// 取った駒（持ち駒に入る）
    pub captured: Option<Role>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpponentMovedPayload {
    pub move_number: u32,
    /// 自駒が取られたマス
    pub captured_your_piece_at: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GameEndPayload {
    pub result: String,
    pub reason: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Ack {
    pub ok: bool,
    pub error: Option<String>,
}

/// game:move の ack。ok=false かつ reason="foul" なら指し直し
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MoveAck {
    pub ok: bool,
    pub reason: Option<String>,
    pub foul_count: Option<u32>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SyncAck {
    pub state: Option<PlayerView>,
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_uses_shogiops_names() {
        assert_eq!(
            serde_json::to_string(&Role::Promotedlance).unwrap(),
            "\"promotedlance\""
        );
        let r: Role = serde_json::from_str("\"tokin\"").unwrap();
        assert_eq!(r, Role::Tokin);
    }

    #[test]
    fn player_view_deserializes() {
        let json = r#"{
            "gameId": "g1",
            "yourColor": "sente",
            "opponent": { "username": "aite", "rating": 1500, "isBot": false },
            "yourPieces": [{ "square": "7g", "role": "pawn" }],
            "yourHand": { "pawn": 2 },
            "turn": "sente",
            "moveNumber": 1,
            "clocks": { "senteMs": 300000, "goteMs": 300000, "running": "sente", "serverTime": 0 },
            "fouls": { "you": 0, "opponent": 1 },
            "youInCheck": false,
            "opponentInCheck": false,
            "status": "playing"
        }"#;
        let view: PlayerView = serde_json::from_str(json).unwrap();
        assert_eq!(view.your_color, Color::Sente);
        assert_eq!(view.your_hand[&Role::Pawn], 2);
        assert_eq!(view.status, GameStatus::Playing);
    }
}
