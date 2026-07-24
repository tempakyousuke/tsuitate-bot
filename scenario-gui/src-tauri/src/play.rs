//! 人間 vs bot の対局モード。
//!
//! 審判は selfplay.rs と同じ裁定を再現する: 反則（フル盤面で非合法な手）は
//! 手番を変えずカウント、累計10回で反則負け、王手宣言は両者へ、
//! 詰み・ステイルメイトで終局。時計はシミュレートしない（GUIのデバッグ対局）。
//! bot に見えるのは PlayerView 相当と観測ログのみ（実対局と同じ情報制約）。
//! 人間側の盤面フィルタ（自駒のみ表示）はフロントエンドが行う。
//! 対局の真実（全手順・反則試行）から kifu::kif_body で `.kif` を書き出し、
//! そのままリプレイ・シナリオ実験（bin/scenario / eval_tally）へ流せる。

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde::Serialize;
use tauri::State;

use tsuitate_bot::board::{
    Promotion, drop_targets, make_usi_square, move_targets, parse_usi_square, promotion_choice,
};
use tsuitate_bot::kifu::{kif_body, role_kanji};
use tsuitate_bot::observation::{Observation, ObservationLog};
use tsuitate_bot::protocol::{Color, Role};
use tsuitate_bot::scenario_core::{make_view, scenarios_dir, side_idx};
use tsuitate_bot::shogi::{Outcome, Position, ShogiMove, parse_usi, unpromote_role};
use tsuitate_bot::strategy::{self, Strategy};

use crate::{LastMove, Snapshot, snapshot_of, with_budget};

const MAX_FOULS: u32 = 10;

fn mark(c: Color) -> &'static str {
    if c == Color::Sente { "▲先手" } else { "△後手" }
}

fn reason_ja(reason: &str) -> &'static str {
    match reason {
        "checkmate" => "詰み",
        "stalemate" => "ステイルメイト",
        "resign" => "投了",
        "foul_limit" => "反則10回",
        _ => "終局",
    }
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PlayOutcomeOut {
    winner: Option<Color>,
    reason: String,
}

/// 人間手番の入力候補（自駒だけを考慮した move-hints 相当。実際の合法性は
/// 審判が判定するので、候補どおりに指しても反則になりうる = 実対局と同じ）
#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PlayHint {
    /// 打ちのときは None
    from: Option<String>,
    role: Role,
    to: String,
    /// "none" | "optional" | "forced"
    promotion: &'static str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlayView {
    engine: String,
    seed: u64,
    budget_ms: u32,
    human_color: Color,
    /// 真実の局面（人間側では相手駒を隠す表示はフロントエンドが行う）
    snapshot: Snapshot,
    /// 人間の手番のときだけ非空
    hints: Vec<PlayHint>,
    /// このコマンドで起きた人間向けイベント（時系列）
    events: Vec<String>,
    total_moves: usize,
    outcome: Option<PlayOutcomeOut>,
    /// bot の直前の手で取られた自駒のマス（盤ハイライト用。自分が指すと消える）
    captured_square: Option<String>,
}

pub struct PlaySession {
    engine: String,
    seed: u64,
    budget_ms: u32,
    human: Color,
    pos: Position,
    bot: Box<dyn Strategy + Send>,
    bot_log: ObservationLog,
    bot_foul_tried: HashSet<String>,
    /// [先手, 後手] の反則累計
    fouls: [u32; 2],
    /// 真実の全手順（USI）
    moves: Vec<String>,
    /// 反則試行 (試行時点の move_number, USI)。両者ぶん（手番側しか反則できない
    /// ので move_number から側は一意に決まる = kif_body の入力そのまま）
    foul_attempts: Vec<(u32, String)>,
    last_move: Option<LastMove>,
    captured_square: Option<String>,
    outcome: Option<PlayOutcomeOut>,
}

impl PlaySession {
    fn new(engine: String, seed: u64, budget_ms: u32, human: Color) -> Result<Self, String> {
        // 思考予算は戦略の構築時に env から読まれる（eval と同じ仕組み）
        let bot = with_budget(budget_ms, || strategy::make_seeded(&engine, seed))
            .ok_or_else(|| format!("未知の戦略名です: {engine}"))?;
        Ok(Self {
            engine,
            seed,
            budget_ms,
            human,
            pos: Position::initial(),
            bot,
            bot_log: ObservationLog::default(),
            bot_foul_tried: HashSet::new(),
            fouls: [0, 0],
            moves: vec![],
            foul_attempts: vec![],
            last_move: None,
            captured_square: None,
            outcome: None,
        })
    }

    fn bot_color(&self) -> Color {
        self.human.other()
    }

    fn finish(&mut self, winner: Color, reason: &str, events: &mut Vec<String>) {
        events.push(if winner == self.human {
            format!("終局: あなたの勝ち（{}）", reason_ja(reason))
        } else {
            format!("終局: botの勝ち（{}）", reason_ja(reason))
        });
        self.outcome = Some(PlayOutcomeOut {
            winner: Some(winner),
            reason: reason.into(),
        });
    }

    /// 受理された手を盤へ適用し、観測・王手宣言・終局判定を行う（selfplay と同じ規約）
    fn apply_move(&mut self, usi: &str, mv: &ShogiMove, events: &mut Vec<String>) {
        let mover = self.pos.turn();
        let captured = self.pos.play_unchecked(mv);
        let move_number = self.pos.move_number();
        let captured_sq = captured.map(|_| match mv {
            ShogiMove::Board { to, .. } => make_usi_square(*to),
            ShogiMove::Drop { .. } => unreachable!("打ちでは駒を取れない"),
        });
        self.moves.push(usi.to_string());
        let (from, to) = match mv {
            ShogiMove::Board { from, to, .. } => {
                (Some(make_usi_square(*from)), make_usi_square(*to))
            }
            ShogiMove::Drop { to, .. } => (None, make_usi_square(*to)),
        };
        self.last_move = Some(LastMove {
            usi: usi.to_string(),
            from,
            to,
        });
        if mover == self.human {
            self.captured_square = None;
            self.bot_log.record(Observation::OpponentMoved {
                move_number,
                captured_my_piece_at: captured_sq,
            });
            events.push(match captured {
                Some(r) => format!("あなた: {usi}（{}を取りました）", role_kanji(unpromote_role(r))),
                None => format!("あなた: {usi}"),
            });
        } else {
            self.bot_foul_tried.clear();
            self.bot_log.record(Observation::MyMove {
                move_number,
                usi: usi.to_string(),
                captured: captured.map(unpromote_role),
            });
            events.push(match &captured_sq {
                Some(sq) => format!("相手が着手し、あなたの駒が {sq} で取られました"),
                None => "相手が着手しました".into(),
            });
            self.captured_square = captured_sq;
        }
        if self.pos.in_check(self.pos.turn()) {
            let in_check = self.pos.turn();
            self.bot_log.record(Observation::Check { in_check });
            events.push(format!("王手宣言: {}の玉に王手", mark(in_check)));
        }
        match self.pos.outcome() {
            Some(Outcome::Checkmate { winner }) => self.finish(winner, "checkmate", events),
            Some(Outcome::Stalemate { winner }) => self.finish(winner, "stalemate", events),
            None => {}
        }
    }

    fn human_move(&mut self, usi: &str) -> Result<Vec<String>, String> {
        if self.outcome.is_some() {
            return Err("対局は終了しています".into());
        }
        if self.pos.turn() != self.human {
            return Err("あなたの手番ではありません".into());
        }
        let mv = parse_usi(usi).ok_or_else(|| format!("USIを解釈できません: {usi}"))?;
        let mut events = vec![];
        if self.pos.is_legal(&mv) {
            self.apply_move(usi, &mv, &mut events);
        } else {
            // 反則: 手番は変わらずカウント（サーバーと同じ。理由は通知されない）
            let idx = side_idx(self.human);
            self.fouls[idx] += 1;
            let count = self.fouls[idx];
            self.foul_attempts.push((self.pos.move_number(), usi.to_string()));
            self.bot_log.record(Observation::OpponentFoul { count });
            events.push(format!("反則: {usi}（あなたの反則 累計{count}回）"));
            if count >= MAX_FOULS {
                self.finish(self.bot_color(), "foul_limit", &mut events);
            }
        }
        Ok(events)
    }

    /// bot の1手（受理されるか終局まで反則を繰り返す）。思考時間ぶんブロックする
    fn bot_move(&mut self) -> Result<Vec<String>, String> {
        if self.outcome.is_some() {
            return Err("対局は終了しています".into());
        }
        let bot_color = self.bot_color();
        if self.pos.turn() != bot_color {
            return Err("botの手番ではありません".into());
        }
        let mut events = vec![];
        loop {
            let view = make_view(&self.pos, bot_color, &self.fouls);
            let Some(usi) = self.bot.choose(&view, &self.bot_log, &self.bot_foul_tried) else {
                events.push("botが投了しました".into());
                self.finish(self.human, "resign", &mut events);
                return Ok(events);
            };
            if parse_usi(&usi).is_some_and(|mv| self.pos.is_legal(&mv)) {
                let mv = parse_usi(&usi).unwrap();
                self.apply_move(&usi, &mv, &mut events);
                return Ok(events);
            }
            let idx = side_idx(bot_color);
            self.fouls[idx] += 1;
            let count = self.fouls[idx];
            self.foul_attempts.push((self.pos.move_number(), usi.clone()));
            self.bot_foul_tried.insert(usi.clone());
            self.bot_log.record(Observation::MyFoul {
                move_number: self.pos.move_number(),
                usi,
            });
            events.push(format!("相手が反則しました（累計{count}回）"));
            if count >= MAX_FOULS {
                self.finish(self.human, "foul_limit", &mut events);
                return Ok(events);
            }
        }
    }

    fn resign(&mut self) -> Vec<String> {
        let mut events = vec![];
        if self.outcome.is_none() {
            events.push("あなたは投了しました".into());
            self.finish(self.bot_color(), "resign", &mut events);
        }
        events
    }

    /// 人間手番の入力候補（board.rs の move-hints 移植 = 自駒だけを考慮）
    fn hints(&self) -> Vec<PlayHint> {
        if self.outcome.is_some() || self.pos.turn() != self.human {
            return vec![];
        }
        let pieces = self.pos.pieces_of(self.human);
        let mut out = vec![];
        for p in &pieces {
            let Some(from) = parse_usi_square(&p.square) else {
                continue;
            };
            for t in move_targets(&pieces, p, self.human) {
                let promotion = match promotion_choice(p.role, from, t, self.human) {
                    Promotion::None => "none",
                    Promotion::Optional => "optional",
                    Promotion::Forced => "forced",
                };
                out.push(PlayHint {
                    from: Some(p.square.clone()),
                    role: p.role,
                    to: make_usi_square(t),
                    promotion,
                });
            }
        }
        for (role, n) in self.pos.hand_map(self.human) {
            if n == 0 {
                continue;
            }
            for t in drop_targets(&pieces, role, self.human) {
                out.push(PlayHint {
                    from: None,
                    role,
                    to: make_usi_square(t),
                    promotion: "none",
                });
            }
        }
        out
    }

    fn view(&self, events: Vec<String>) -> PlayView {
        PlayView {
            engine: self.engine.clone(),
            seed: self.seed,
            budget_ms: self.budget_ms,
            human_color: self.human,
            snapshot: snapshot_of(&self.pos, &self.fouls, self.last_move.clone()),
            hints: self.hints(),
            events,
            total_moves: self.moves.len(),
            outcome: self.outcome.clone(),
            captured_square: self.captured_square.clone(),
        }
    }

    /// 対局の真実から `.kif` 全文を組み立てる（ply を与えるとシナリオ指定つき）
    fn kif_text(&self, ply: Option<usize>, desc: Option<&str>) -> Result<String, String> {
        let (sente_name, gote_name) = if self.human == Color::Sente {
            ("人間".to_string(), self.engine.clone())
        } else {
            (self.engine.clone(), "人間".to_string())
        };
        let mut out = format!(
            "棋戦：Shogi Quest\n手合割：平手\n先手：{sente_name}\n後手：{gote_name}\n\
             手数----指手---------消費時間--\n"
        );
        let mut directive = String::from("*scenario");
        if let Some(p) = ply {
            if p >= self.moves.len() {
                return Err(format!(
                    "ply={p} が手数 {} 以上です（考えさせる手がありません）",
                    self.moves.len()
                ));
            }
            directive.push_str(&format!(" ply={p} target={}", self.moves[p]));
        }
        let desc_text = desc.filter(|s| !s.trim().is_empty()).map(str::to_string).unwrap_or_else(|| {
            format!(
                "GUI対局の再現（人間={}, bot={} seed={} 予算={}ms）",
                if self.human == Color::Sente { "先手" } else { "後手" },
                self.engine,
                self.seed,
                self.budget_ms,
            )
        });
        directive.push_str(&format!(" desc={desc_text}\n"));
        out.push_str(&directive);
        let ending = match self.outcome.as_ref().map(|o| o.reason.as_str()) {
            Some("resign") => Some("投了"),
            Some("foul_limit") => Some("反則負け"),
            // 詰み・ステイルメイトは最終手で終局が確定する（終局行なし）。
            // 対局中の書き出しは trailing 反則があるときだけ kif_body が「中断」を入れる
            _ => None,
        };
        out.push_str(&kif_body(&self.moves, &self.foul_attempts, ending)?);
        Ok(out)
    }
}

#[derive(Default)]
pub struct PlayState(pub Arc<Mutex<Option<PlaySession>>>);

/// bot 思考中はセッションのロックが数秒保持されるので、全コマンドを
/// spawn_blocking で回して UI スレッドを塞がない
async fn on_session<T: Send + 'static>(
    state: &State<'_, PlayState>,
    f: impl FnOnce(&mut Option<PlaySession>) -> Result<T, String> + Send + 'static,
) -> Result<T, String> {
    let arc = state.0.clone();
    tauri::async_runtime::spawn_blocking(move || f(&mut arc.lock().unwrap()))
        .await
        .map_err(|e| format!("実行スレッドの異常終了: {e}"))?
}

fn require(slot: &mut Option<PlaySession>) -> Result<&mut PlaySession, String> {
    slot.as_mut().ok_or_else(|| "対局が開始されていません".into())
}

#[tauri::command]
pub async fn play_start(
    state: State<'_, PlayState>,
    engine: String,
    human_color: Color,
    seed: u64,
    budget_ms: u32,
) -> Result<PlayView, String> {
    on_session(&state, move |slot| {
        let session = PlaySession::new(engine, seed, budget_ms, human_color)?;
        let events = vec![format!(
            "対局開始: あなたは{}、bot={}（seed={} 予算={}ms）",
            mark(human_color), session.engine, session.seed, session.budget_ms
        )];
        let view = session.view(events);
        *slot = Some(session);
        Ok(view)
    })
    .await
}

#[tauri::command]
pub async fn play_human_move(
    state: State<'_, PlayState>,
    usi: String,
) -> Result<PlayView, String> {
    on_session(&state, move |slot| {
        let session = require(slot)?;
        let events = session.human_move(&usi)?;
        Ok(session.view(events))
    })
    .await
}

#[tauri::command]
pub async fn play_bot_move(state: State<'_, PlayState>) -> Result<PlayView, String> {
    on_session(&state, |slot| {
        let session = require(slot)?;
        let events = session.bot_move()?;
        Ok(session.view(events))
    })
    .await
}

#[tauri::command]
pub async fn play_resign(state: State<'_, PlayState>) -> Result<PlayView, String> {
    on_session(&state, |slot| {
        let session = require(slot)?;
        let events = session.resign();
        Ok(session.view(events))
    })
    .await
}

/// フロント再マウント時の状態復元用（イベントなしの現在ビュー）
#[tauri::command]
pub async fn play_view(state: State<'_, PlayState>) -> Result<PlayView, String> {
    on_session(&state, |slot| {
        let session = require(slot)?;
        Ok(session.view(vec![]))
    })
    .await
}

/// 対局を `.kif` へ書き出す。file_name が相対ならリポジトリの scenarios/ に置く。
/// 戻り値は書き出した絶対パス
#[tauri::command]
pub async fn play_export(
    state: State<'_, PlayState>,
    file_name: String,
    ply: Option<usize>,
    desc: Option<String>,
) -> Result<String, String> {
    on_session(&state, move |slot| {
        let session = require(slot)?;
        if session.moves.is_empty() {
            return Err("まだ指し手がありません".into());
        }
        let text = session.kif_text(ply, desc.as_deref())?;
        let mut path = PathBuf::from(file_name.trim());
        if path.file_name().is_none() {
            return Err("ファイル名を指定してください".into());
        }
        if path.is_relative() {
            path = scenarios_dir().join(path);
        }
        if path.extension().is_none_or(|e| e != "kif") {
            path.set_extension("kif");
        }
        std::fs::write(&path, text).map_err(|e| format!("{} に書けません: {e}", path.display()))?;
        Ok(path.to_string_lossy().into_owned())
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tsuitate_bot::kifu::parse_kif;
    use tsuitate_bot::scenario_core::replay;

    /// 人間役が真実の合法手を先頭から指す即席プレイヤーで1局回し、
    /// 書き出した KIF が parse_kif → 裁定つき replay まで通ること
    /// （= scenario / GUI リプレイにそのまま流せる形式であること）
    #[test]
    fn 対局からkif書き出しまでの一巡が裁定つきreplayを通る() {
        let mut s =
            PlaySession::new("heuristic".into(), 7, 500, Color::Sente).expect("session");
        for _ in 0..60 {
            if s.outcome.is_some() {
                break;
            }
            if s.pos.turn() == s.human {
                let mv = s.pos.legal_moves().into_iter().next();
                match mv {
                    Some(mv) => {
                        s.human_move(&mv.to_usi()).unwrap();
                    }
                    None => break,
                }
            } else {
                s.bot_move().unwrap();
            }
        }
        assert!(!s.moves.is_empty());
        let text = s.kif_text(Some(0), None).unwrap();
        let kifu = parse_kif(&text).unwrap();
        assert_eq!(kifu.plies.len(), s.moves.len());
        assert_eq!(kifu.directives.get("ply").unwrap(), "0");
        assert_eq!(kifu.directives.get("target").unwrap(), &s.moves[0]);
        // 裁定つき replay（合法手は合法・反則試行は非合法を assert する）が全編通る
        let rep = replay(&kifu, kifu.plies.len());
        // foul_limit 終局は最後の手番の反則が trailing になり replay には現れない
        if s.outcome.as_ref().map(|o| o.reason.as_str()) != Some("foul_limit") {
            assert_eq!(rep.fouls, s.fouls);
        }
    }

    /// 現行 estimator でもセッションが回ること（思考予算の配線のスモークテスト）
    #[test]
    fn estimatorが後手で1手指せる() {
        let mut s =
            PlaySession::new("estimator".into(), 3, 500, Color::Sente).expect("session");
        s.human_move("7g7f").unwrap();
        let events = s.bot_move().unwrap();
        assert_eq!(s.moves.len(), 2, "{events:?}");
        assert_eq!(s.pos.turn(), Color::Sente);
    }

    /// 人間の反則は手番を変えずカウントされ、観測が bot 側へ届くこと
    #[test]
    fn 人間の反則は手番維持でカウントされkifに残る() {
        let mut s =
            PlaySession::new("heuristic".into(), 1, 500, Color::Sente).expect("session");
        // 初期局面で 1一 の香を前に = 相手駒（見えない盤上の駒）で塞がれた非合法手…
        // ではなく確実に非合法な手として、自玉を残したまま王を2マス動かす手を使う
        let events = s.human_move("5i5g").unwrap();
        assert_eq!(s.fouls, [1, 0]);
        assert_eq!(s.pos.turn(), Color::Sente, "反則では手番が変わらない");
        assert!(events.iter().any(|e| e.contains("反則")), "{events:?}");
        // 合法手を指すと受理される
        s.human_move("7g7f").unwrap();
        assert_eq!(s.moves, vec!["7g7f".to_string()]);
        let text = s.kif_text(None, None).unwrap();
        assert!(text.contains("*illegal:"), "{text}");
        let kifu = parse_kif(&text).unwrap();
        assert_eq!(kifu.plies[0].fouls.len(), 1);
    }
}
