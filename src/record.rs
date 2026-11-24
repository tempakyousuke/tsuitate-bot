//! 対局記録（JSONL）。
//!
//! 1対局 = 1ファイル。botが観測できた全情報（観測イベント）と自分の着手選択
//! （思考時間つき）・終局結果を追記していく。`bin/analyze`・NN学習データの
//! 書き出し・局面レビューの入力になる。相手の実際の手はbotからは見えないため、
//! ここには入らない
//! （ローカルdevサーバー対局ならサーバーDBの games.moves に全手順が残る）。

use std::fs::{File, OpenOptions, create_dir_all};
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

    /// 既存の対局記録の続きへ追記する（再起動後の復帰対局。`load_record` で読み直したファイルを渡す）。
    /// 1対局 = 1ファイルを保つために新しいファイルを作らない。途中まで書かれた最後の行が残っていても
    /// 次の行と混ざらないよう、改行で終わっていなければ改行を足す
    pub fn resume(path: &Path) -> std::io::Result<GameRecorder> {
        let ends_with_newline = std::fs::read(path)?.last().is_none_or(|&b| b == b'\n');
        let file = OpenOptions::new().append(true).open(path)?;
        let mut rec = GameRecorder {
            writer: BufWriter::new(file),
            path: path.to_path_buf(),
        };
        if !ends_with_newline {
            writeln!(rec.writer)?;
        }
        Ok(rec)
    }

    /// 復帰の印（読み直した観測の数と、視界から作り直して足した観測の数）
    pub fn resumed(&mut self, restored: usize, repaired: usize) {
        self.write_line(&json!({
            "type": "resumed",
            "ts": now_ms(),
            "restored": restored,
            "repaired": repaired,
        }));
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

    /// 記録の由来（アリーナの run 識別・相手・shard・game 番号など）。
    /// **match ヘッダの直後に1行**書く。arena-balance の粒子数 veto が
    /// 「この record は判定対象の run のものか」を games.jsonl と突き合わせる
    /// のに使う（PR #41 レビュー8巡目: 由来の無い record は別 run の粒子数で
    /// veto を通せてしまう）
    pub fn meta(&mut self, fields: &serde_json::Map<String, serde_json::Value>) {
        let mut obj = serde_json::Map::new();
        obj.insert("type".into(), json!("meta"));
        obj.insert("ts".into(), json!(now_ms()));
        for (k, v) in fields {
            obj.insert(k.clone(), v.clone());
        }
        self.write_line(&serde_json::Value::Object(obj));
    }

    /// 自分が選んだ手（受理/反則が確定する前）と思考時間。
    /// debug は戦略の内部状態サマリ（推定の健全性・相手玉分布など。ないなら None）
    pub fn chosen(
        &mut self,
        move_number: u32,
        usi: &str,
        think_ms: u64,
        debug: Option<&serde_json::Value>,
    ) {
        self.write_line(&json!({
            "type": "chose",
            "ts": now_ms(),
            "move_number": move_number,
            "usi": usi,
            "think_ms": think_ms,
            "debug": debug,
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

/// 読み直した対局記録（再起動後の復帰対局で、観測ログを作り直すのに使う）
#[derive(Debug)]
pub struct LoadedRecord {
    pub path: PathBuf,
    pub game_id: String,
    pub your_color: Color,
    /// 記録された観測（時系列順）
    pub events: Vec<Observation>,
    /// 最後に選んだ手で、受理・反則の観測がまだ無いもの（選んだ時点の手数, USI）。
    /// 送った直後（結果を受け取る前）に落ちた場合だけ残る
    pub pending: Option<(u32, String)>,
    /// 終局・中断・投了の行がある（もう続きは無い）
    pub finished: bool,
}

/// `dir` の中から対局 `game_id` の記録（`<時刻>-<対局ID>.jsonl`）を探す。複数あればいちばん新しいもの
pub fn find_record(dir: &str, game_id: &str) -> Option<PathBuf> {
    let suffix = format!("-{game_id}.jsonl");
    let mut best: Option<(u64, PathBuf)> = None;
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let name = entry.file_name();
        let Some(ts) = name.to_str().and_then(|n| n.strip_suffix(&suffix)).and_then(|p| p.parse::<u64>().ok())
        else {
            continue;
        };
        if best.as_ref().is_none_or(|(b, _)| ts > *b) {
            best = Some((ts, entry.path()));
        }
    }
    best.map(|(_, p)| p)
}

/// 対局記録を読み直す。読めない行（落ちる瞬間に途中まで書かれた最後の行）は捨てる。
/// ヘッダ（`match` の行）が無い・壊れているファイルは Err
pub fn load_record(path: &Path) -> std::io::Result<LoadedRecord> {
    let bad = |why: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{}: {why}", path.display()));
    let text = std::fs::read_to_string(path)?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: serde_json::Value =
        lines.next().and_then(|l| serde_json::from_str(l).ok()).ok_or_else(|| bad("ヘッダが読めない"))?;
    if header["type"] != "match" {
        return Err(bad("先頭が match の行でない"));
    }
    let game_id = header["game_id"].as_str().ok_or_else(|| bad("game_id が無い"))?.to_string();
    let your_color: Color =
        serde_json::from_value(header["your_color"].clone()).map_err(|_| bad("your_color が読めない"))?;
    let mut rec = LoadedRecord { path: path.to_path_buf(), game_id, your_color, events: vec![], pending: None, finished: false };
    for line in lines {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        match v["type"].as_str() {
            Some("obs") => {
                let Ok(event) = serde_json::from_value::<Observation>(v["event"].clone()) else {
                    continue;
                };
                // 選んだ手の結果（受理か反則）が届いた
                if matches!(event, Observation::MyMove { .. } | Observation::MyFoul { .. }) {
                    rec.pending = None;
                }
                rec.events.push(event);
            }
            Some("chose") => {
                if let (Some(n), Some(usi)) = (v["move_number"].as_u64(), v["usi"].as_str()) {
                    rec.pending = Some((n as u32, usi.to_string()));
                }
            }
            Some("end") | Some("aborted") | Some("resign") => rec.finished = true,
            _ => {}
        }
    }
    Ok(rec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{
        FoulRecord, MoveRecord, OpponentInfo, RatingChange, RatingChangePair, Role,
    };

    /// 書いた記録を読み直すと同じ観測が戻り、結果がまだ届いていない手だけが `pending` に残る。
    /// 続きは同じファイルへ追記され、途中まで書かれた行は捨てる
    #[test]
    fn 記録を読み直して続きを追記する() {
        let dir = std::env::temp_dir().join(format!("tsuitate-bot-record-resume-{}-{}", std::process::id(), now_ms()));
        let dir_str = dir.to_str().unwrap().to_string();
        let mut rec = GameRecorder::create(&dir_str, "ab-c_D", Color::Gote, "rl_policy@x").unwrap();
        let path = rec.path().to_path_buf();
        let obs = [
            Observation::OpponentMoved { move_number: 2, captured_my_piece_at: None },
            Observation::MyFoul { move_number: 2, usi: "5a5b".into() },
            Observation::MyMove { move_number: 3, usi: "3c3d".into(), captured: None },
            Observation::Check { in_check: Color::Sente },
            Observation::OpponentFoul { count: 1 },
            Observation::OpponentMoved { move_number: 4, captured_my_piece_at: Some("3d".into()) },
        ];
        rec.chosen(2, "5a5b", 5, None);
        rec.observation(&obs[0]);
        rec.observation(&obs[1]);
        rec.chosen(2, "3c3d", 5, None);
        for o in &obs[2..] {
            rec.observation(o);
        }
        drop(rec);
        assert_eq!(find_record(&dir_str, "ab-c_D"), Some(path.clone()));
        assert_eq!(find_record(&dir_str, "c_D"), None, "対局IDの一部が同じだけの記録は拾わない");
        let loaded = load_record(&path).unwrap();
        assert_eq!((loaded.game_id.as_str(), loaded.your_color, loaded.finished), ("ab-c_D", Color::Gote, false));
        assert!(loaded.pending.is_none(), "結果が届いた手は残らない");
        let json = |e: &[Observation]| e.iter().map(|x| serde_json::to_string(x).unwrap()).collect::<Vec<_>>();
        assert_eq!(json(&loaded.events), json(&obs));

        // 手を選んで送った直後に落ちた: 記録の最後は chose の行と、途中まで書かれた行
        let mut rec = GameRecorder::resume(&path).unwrap();
        rec.chosen(4, "2b3c", 7, None);
        drop(rec);
        let mut raw = std::fs::read_to_string(&path).unwrap();
        raw.push_str("{\"type\":\"obs\",\"event\":{\"kind\":\"my_mo");
        std::fs::write(&path, raw).unwrap();
        let loaded = load_record(&path).unwrap();
        assert_eq!(loaded.pending, Some((4, "2b3c".to_string())));
        assert_eq!(loaded.events.len(), obs.len());

        // 続きの追記は途中まで書かれた行と混ざらない
        let mut rec = GameRecorder::resume(&path).unwrap();
        rec.resumed(loaded.events.len(), 1);
        rec.observation(&Observation::MyMove { move_number: 5, usi: "2b3c".into(), captured: Some(Role::Pawn) });
        drop(rec);
        let loaded = load_record(&path).unwrap();
        assert_eq!(loaded.events.len(), obs.len() + 1);
        assert!(loaded.pending.is_none() && !loaded.finished);
        std::fs::remove_dir_all(&dir).ok();
    }

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
        rec.chosen(1, "7g7f", 123, None);
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
