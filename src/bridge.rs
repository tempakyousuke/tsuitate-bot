//! Claude（対話セッション）が直接指すためのファイル橋渡し戦略。
//!
//! 手番が来るたびに bridge/turn.json へ自分の可視情報の全量（PlayerView 相当・
//! 観測履歴・この手番で反則になった手）を書き出し、bridge/move.txt に
//! 指し手（USI。"resign" で投了）が書かれるまでブロックして待つ。
//! 対局実験用であり、思考は接続の外側（対話セッション）で行われる。

use std::collections::HashSet;
use std::path::PathBuf;
use std::thread::sleep;
use std::time::Duration;

use serde_json::json;

use crate::observation::ObservationLog;
use crate::protocol::PlayerView;
use crate::strategy::Strategy;

pub struct FileBridge {
    dir: PathBuf,
    seq: u64,
}

impl FileBridge {
    pub fn new() -> Self {
        FileBridge {
            dir: PathBuf::from("bridge"),
            seq: 0,
        }
    }
}

impl Default for FileBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl Strategy for FileBridge {
    fn choose(
        &mut self,
        view: &PlayerView,
        log: &ObservationLog,
        foul_tried: &HashSet<String>,
    ) -> Option<String> {
        if std::fs::create_dir_all(&self.dir).is_err() {
            return None;
        }
        self.seq += 1;
        let move_path = self.dir.join("move.txt");
        let _ = std::fs::remove_file(&move_path);

        let hand: Vec<serde_json::Value> = view
            .your_hand
            .iter()
            .filter(|(_, n)| **n > 0)
            .map(|(r, n)| json!({ "role": r, "count": n }))
            .collect();
        let events: Vec<serde_json::Value> = log
            .events()
            .iter()
            .map(|e| serde_json::to_value(e).unwrap_or(serde_json::Value::Null))
            .collect();
        let turn = json!({
            "seq": self.seq,
            "move_number": view.move_number,
            "your_color": view.your_color,
            "your_pieces": view.your_pieces,
            "your_hand": hand,
            "clocks": { "sente_ms": view.clocks.sente_ms, "gote_ms": view.clocks.gote_ms },
            "fouls": { "you": view.fouls.you, "opponent": view.fouls.opponent },
            "you_in_check": view.you_in_check,
            "opponent_in_check": view.opponent_in_check,
            "foul_tried_this_turn": foul_tried.iter().collect::<Vec<_>>(),
            "observations": events,
        });
        let tmp = self.dir.join("turn.json.tmp");
        if std::fs::write(&tmp, serde_json::to_string_pretty(&turn).ok()?).is_err() {
            return None;
        }
        let _ = std::fs::rename(&tmp, self.dir.join("turn.json"));
        println!("bridge: 手番情報を書き出しました（seq {}）。move.txt を待ちます", self.seq);

        loop {
            if let Ok(usi) = std::fs::read_to_string(&move_path) {
                let usi = usi.trim().to_string();
                if usi.is_empty() {
                    sleep(Duration::from_millis(300));
                    continue;
                }
                let _ = std::fs::remove_file(&move_path);
                if usi == "resign" {
                    return None;
                }
                return Some(usi);
            }
            sleep(Duration::from_millis(300));
        }
    }

    fn name(&self) -> &'static str {
        "bridge"
    }
}
