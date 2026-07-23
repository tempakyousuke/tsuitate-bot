//! 「ついたて将棋ビューワー」（tsuboshun氏運営、tsuitate リポジトリとは無関係の
//! 第三者サイト）の webhook bot API契約。
//!
//! 真実は運営者提供のサンプル
//! （<https://github.com/tsuboshun/tsuitate-sample-bot> README「dispatcher からの
//! リクエストモデル」節）。Socket.IO常時接続の protocol.rs とは無関係の
//! 別プロトコルなので型を混ぜない。

use std::collections::HashMap;

use serde::Deserialize;

use crate::protocol::Color;

pub const INFO_NONE: u8 = 0;
pub const INFO_FOUL: u8 = 1;
pub const INFO_FOUL_UNDER_CHECK: u8 = 2;
pub const INFO_CHECK: u8 = 3;
pub const INFO_CHECKMATE: u8 = 4;

pub fn is_foul_info(info: u8) -> bool {
    matches!(info, INFO_FOUL | INFO_FOUL_UNDER_CHECK)
}

pub fn is_check_info(info: u8) -> bool {
    matches!(info, INFO_CHECK | INFO_CHECKMATE)
}

/// "b"/"w" を Color に変換する（protocol.rs の Sente/Gote と同じ意味）
pub fn parse_bw_color(s: &str) -> Option<Color> {
    match s {
        "b" => Some(Color::Sente),
        "w" => Some(Color::Gote),
        _ => None,
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BotTurnRequest {
    #[serde(rename = "type")]
    pub kind: String,
    #[allow(dead_code)]
    pub request_id: String,
    pub game_id: String,
    /// "b" | "w"
    pub color: String,
    #[allow(dead_code)]
    pub number: u32,
    pub ply: u32,
    #[allow(dead_code)]
    pub deadline_ms: u64,
    pub positions: HashMap<String, PositionEntry>,
    pub game: GameInfo,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PositionEntry {
    #[allow(dead_code)]
    pub sfen: String,
    #[serde(default)]
    pub fouls: Option<FoulsField>,
    #[serde(default)]
    pub last_move: Option<String>,
    #[serde(default)]
    pub last_info: Option<u8>,
    #[serde(default)]
    pub last_capture: Option<String>,
    #[serde(default)]
    pub was_promotion: Option<bool>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct FoulsField {
    pub b: u32,
    pub w: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GameInfo {
    #[serde(rename = "type")]
    pub kind: String,
    pub required_players: RequiredPlayers,
    #[serde(default)]
    pub param: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct RequiredPlayers {
    pub b: u32,
    pub w: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sample_first_move_payload() {
        let json = r#"{
            "type": "your_turn",
            "requestId": "r1",
            "gameId": "g1",
            "color": "b",
            "number": 0,
            "ply": 0,
            "deadlineMs": 1000,
            "game": {
                "type": "ついたて",
                "gameKind": 1,
                "promotionRank": 3,
                "drawMoveCount": 150,
                "requiredPlayers": { "b": 1, "w": 1 }
            },
            "positions": {
                "0": {
                    "sfen": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
                    "fouls": { "b": 9, "w": 9 },
                    "times": { "b": 300, "w": 300 },
                    "byoyomiActive": { "b": false, "w": false }
                }
            }
        }"#;
        let req: BotTurnRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.kind, "your_turn");
        assert_eq!(req.ply, 0);
        assert_eq!(req.game.kind, "ついたて");
        assert_eq!(req.game.required_players.b, 1);
        assert_eq!(parse_bw_color(&req.color), Some(Color::Sente));
        let p0 = &req.positions["0"];
        assert_eq!(p0.fouls.unwrap().b, 9);
        assert!(p0.last_move.is_none());
    }

    #[test]
    fn parses_masked_opponent_capture() {
        let json = r#"{
            "sfen": "rbsgk/4R/5/P4/KGSB1 w P 2",
            "fouls": { "b": 9, "w": 9 },
            "lastMove": "+1512HI",
            "lastCapture": "P",
            "lastInfo": 3
        }"#;
        let entry: PositionEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.last_move.as_deref(), Some("+1512HI"));
        assert_eq!(entry.last_capture.as_deref(), Some("P"));
        assert_eq!(entry.last_info, Some(INFO_CHECK));
    }
}
