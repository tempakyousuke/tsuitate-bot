//! 対局記録（JSONL）。
//!
//! 1対局 = 1ファイル。botが観測できた全情報（観測イベント）と自分の着手選択
//! （思考時間つき）・終局結果を追記していく。次期評価関数・推定器の改善の
//! 参考データにする。相手の実際の手はbotからは見えないため、ここには入らない
//! （ローカルdevサーバー対局ならサーバーDBの games.moves に全手順が残る）。

use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::json;

use crate::observation::Observation;
use crate::protocol::{Color, GameEndPayload};

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub struct GameRecorder {
    writer: BufWriter<File>,
    path: PathBuf,
}

impl GameRecorder {
    pub fn create(
        dir: &str,
        game_id: &str,
        your_color: Color,
        strategy: &str,
    ) -> std::io::Result<GameRecorder> {
        create_dir_all(dir)?;
        let path = PathBuf::from(dir).join(format!("{}-{game_id}.jsonl", now_ms()));
        let file = File::create(&path)?;
        let mut rec = GameRecorder {
            writer: BufWriter::new(file),
            path,
        };
        rec.write_line(&json!({
            "type": "match",
            "ts": now_ms(),
            "game_id": game_id,
            "your_color": your_color,
            "strategy": strategy,
        }));
        Ok(rec)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// 途中経過が消えないよう1行ごとにflushする（対局は長時間に及ぶ）
    fn write_line(&mut self, value: &impl Serialize) {
        match serde_json::to_string(value) {
            Ok(line) => {
                if writeln!(self.writer, "{line}").and_then(|_| self.writer.flush()).is_err() {
                    eprintln!("対局記録の書き込みに失敗しました: {}", self.path.display());
                }
            }
            Err(e) => eprintln!("対局記録のシリアライズに失敗: {e}"),
        }
    }

    pub fn observation(&mut self, obs: &Observation) {
        self.write_line(&json!({ "type": "obs", "ts": now_ms(), "event": obs }));
    }

    /// 自分が選んだ手（受理/反則が確定する前）と思考時間
    pub fn chosen(&mut self, move_number: u32, usi: &str, think_ms: u64) {
        self.write_line(&json!({
            "type": "chose",
            "ts": now_ms(),
            "move_number": move_number,
            "usi": usi,
            "think_ms": think_ms,
        }));
    }

    pub fn resigned(&mut self, move_number: u32) {
        self.write_line(&json!({ "type": "resign", "ts": now_ms(), "move_number": move_number }));
    }

    /// 終局。game:end は全公開（完全棋譜・反則試行・終局図・相手の身元）なので
    /// ペイロードを丸ごと残す。これが分析用の「真実」になる
    pub fn end(&mut self, payload: &GameEndPayload, summary: &str) {
        self.write_line(&json!({
            "type": "end",
            "ts": now_ms(),
            "payload": payload,
            "summary": summary,
        }));
    }

    /// サーバー再起動などで対局が消えた（正規の game:end が来ない）場合
    pub fn aborted(&mut self, reason: &str, summary: &str) {
        self.write_line(&json!({
            "type": "aborted",
            "ts": now_ms(),
            "reason": reason,
            "summary": summary,
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{
        FoulRecord, MoveRecord, OpponentInfo, RatingChange, RatingChangePair, Role,
    };

    #[test]
    fn writes_jsonl_lines() {
        let dir = std::env::temp_dir().join(format!("tsuitate-bot-record-test-{}", now_ms()));
        let dir_str = dir.to_str().unwrap().to_string();
        let mut rec = GameRecorder::create(&dir_str, "g1", Color::Sente, "estimator").unwrap();
        rec.observation(&Observation::MyMove {
            move_number: 1,
            usi: "7g7f".into(),
            captured: Some(Role::Pawn),
        });
        rec.chosen(1, "7g7f", 123);
        rec.end(
            &GameEndPayload {
                result: "sente_win".into(),
                reason: "checkmate".into(),
                final_sfen: "sfen".into(),
                moves: vec![MoveRecord {
                    usi: "7g7f".into(),
                    by_color: Color::Sente,
                    ms: 1000,
                    fouls_before: 0,
                }],
                foul_attempts: vec![FoulRecord {
                    move_number: 2,
                    by_color: Color::Gote,
                    usi: "2b8h".into(),
                }],
                rating_change: RatingChangePair {
                    you: RatingChange {
                        before: 1500,
                        after: 1510,
                    },
                    opponent: RatingChange {
                        before: 1500,
                        after: 1490,
                    },
                },
                opponent: OpponentInfo {
                    username: "aite".into(),
                    rating: 1500,
                    is_bot: false,
                },
            },
            "s",
        );
        let path = rec.path().to_path_buf();
        drop(rec);

        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<serde_json::Value> = content
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[0]["type"], "match");
        assert_eq!(lines[0]["your_color"], "sente");
        assert_eq!(lines[1]["event"]["kind"], "my_move");
        assert_eq!(lines[1]["event"]["captured"], "pawn");
        assert_eq!(lines[2]["usi"], "7g7f");
        assert_eq!(lines[3]["payload"]["result"], "sente_win");
        assert_eq!(lines[3]["payload"]["moves"][0]["byColor"], "sente");
        assert_eq!(lines[3]["payload"]["opponent"]["username"], "aite");

        std::fs::remove_dir_all(&dir).ok();
    }
}
