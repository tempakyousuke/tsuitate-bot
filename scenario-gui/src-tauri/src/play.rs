//! 対局モード。対局者は先後それぞれ「人間」か「bot」で、
//! 人間 vs bot（実戦レビュー用）と bot vs bot（観戦・回帰用）の両方を回す。
//!
//! 審判は selfplay.rs と同じ裁定を再現する: 反則（フル盤面で非合法な手）は
//! 手番を変えずカウント、累計10回で反則負け、王手宣言は両者へ、
//! 詰み・ステイルメイトで終局。時計はシミュレートしない（GUIのデバッグ対局）。
//! bot に見えるのは PlayerView 相当と**その bot 自身の**観測ログのみ
//! （実対局と同じ情報制約。bot 同士でも互いの盤面は覗けない）。
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
use tsuitate_bot::rl::plane_view::{PlaneView, plane_views};
use tsuitate_bot::scenario_core::{make_view, scenarios_dir, side_idx};
use tsuitate_bot::shogi::{Outcome, Position, ShogiMove, parse_usi, unpromote_role};
use tsuitate_bot::strategy::{self, Strategy};

use crate::{LastMove, Snapshot, snapshot_of, with_budget};

const MAX_FOULS: u32 = 10;

const SIDES: [Color; 2] = [Color::Sente, Color::Gote];

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
    /// [先手, 後手] の表示名（人間は "人間"、bot はエンジン名）
    names: [String; 2],
    /// [先手, 後手] の seed（人間側は None）
    seeds: [Option<u64>; 2],
    /// [先手が bot か, 後手が bot か]
    is_bot: [bool; 2],
    budget_ms: u32,
    /// 人間側の色。bot 同士の観戦では None
    human_color: Option<Color>,
    /// 真実の局面（人間側では相手駒を隠す表示はフロントエンドが行う）
    snapshot: Snapshot,
    /// 人間の手番のときだけ非空
    hints: Vec<PlayHint>,
    /// このコマンドで起きたイベント（時系列）
    events: Vec<String>,
    total_moves: usize,
    outcome: Option<PlayOutcomeOut>,
    /// bot の直前の手で取られた人間の駒のマス（盤ハイライト用。自分が指すと消える）
    captured_square: Option<String>,
}

/// bot 側の対局者。観測ログと反則試行メモは **bot ごとに独立**
/// （bot 同士でも互いの情報は混ざらない）
struct BotSide {
    engine: String,
    seed: u64,
    strat: Box<dyn Strategy + Send>,
    log: ObservationLog,
    foul_tried: HashSet<String>,
}

/// 人間側の対局者。指し手は GUI から来るが、観測ログと反則試行メモは bot と同じ形で持つ
/// （その席から見た入力プレーンを出すため。bot には渡らない）
#[derive(Default)]
struct HumanSide {
    log: ObservationLog,
    foul_tried: HashSet<String>,
}

enum Player {
    Human(HumanSide),
    Bot(BotSide),
}

impl Player {
    fn name(&self) -> String {
        match self {
            Player::Human(_) => "人間".into(),
            Player::Bot(b) => b.engine.clone(),
        }
    }

    fn seed(&self) -> Option<u64> {
        match self {
            Player::Human(_) => None,
            Player::Bot(b) => Some(b.seed),
        }
    }

    /// この席の観測ログと、今の手番で反則した手
    fn seat(&self) -> (&ObservationLog, &HashSet<String>) {
        match self {
            Player::Human(h) => (&h.log, &h.foul_tried),
            Player::Bot(b) => (&b.log, &b.foul_tried),
        }
    }

    fn seat_mut(&mut self) -> (&mut ObservationLog, &mut HashSet<String>) {
        match self {
            Player::Human(h) => (&mut h.log, &mut h.foul_tried),
            Player::Bot(b) => (&mut b.log, &mut b.foul_tried),
        }
    }
}

/// ある席の観測から作った入力プレーン（`play_planes` / `kifu_planes` の戻り値）
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanesOut {
    /// どちらの席の観測か
    side: Color,
    /// いまの手数（次に指す手が何手目か）
    move_number: u32,
    /// その席が手番か（手番でない席は「相手が考えている間」の観測）
    to_move: bool,
    pub(crate) planes: Vec<PlaneView>,
}

impl PlanesOut {
    /// `side` の席の入力プレーン。エンコーダが読むのと同じ3つ（視界・観測ログ・今の手番の反則試行）から作る。
    /// 対局中の表示（`PlaySession::planes`）と棋譜の再生（`kifu_planes`）が同じ関数を通る
    pub(crate) fn of(
        pos: &Position,
        fouls: &[u32; 2],
        side: Color,
        log: &ObservationLog,
        foul_tried: &HashSet<String>,
    ) -> Self {
        let view = make_view(pos, side, fouls);
        Self {
            side,
            move_number: pos.move_number(),
            to_move: pos.turn() == side,
            planes: plane_views(&view, log, foul_tried),
        }
    }
}

fn make_bot(engine: &str, seed: u64) -> Result<Player, String> {
    let strat = strategy::make_seeded(engine, seed)
        .ok_or_else(|| format!("未知の戦略名です: {engine}"))?;
    Ok(Player::Bot(BotSide {
        engine: engine.to_string(),
        seed,
        strat,
        log: ObservationLog::default(),
        foul_tried: HashSet::new(),
    }))
}

pub struct PlaySession {
    budget_ms: u32,
    /// [先手, 後手]
    players: [Player; 2],
    /// 人間側の色。bot 同士の観戦では None
    human: Option<Color>,
    pos: Position,
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
    /// 人間 vs bot
    fn new_vs_bot(
        engine: String,
        seed: u64,
        budget_ms: u32,
        human: Color,
    ) -> Result<Self, String> {
        // 思考予算は戦略の構築時に env から読まれる（eval と同じ仕組み）
        let bot = with_budget(budget_ms, || make_bot(&engine, seed))?;
        let players = if human == Color::Sente {
            [Player::Human(HumanSide::default()), bot]
        } else {
            [bot, Player::Human(HumanSide::default())]
        };
        Ok(Self::with_players(players, Some(human), budget_ms))
    }

    /// bot vs bot（観戦）
    fn new_bots(
        sente_engine: String,
        sente_seed: u64,
        gote_engine: String,
        gote_seed: u64,
        budget_ms: u32,
    ) -> Result<Self, String> {
        let players = with_budget(budget_ms, || {
            Ok::<_, String>([
                make_bot(&sente_engine, sente_seed)?,
                make_bot(&gote_engine, gote_seed)?,
            ])
        })?;
        Ok(Self::with_players(players, None, budget_ms))
    }

    fn with_players(players: [Player; 2], human: Option<Color>, budget_ms: u32) -> Self {
        Self {
            budget_ms,
            players,
            human,
            pos: Position::initial(),
            fouls: [0, 0],
            moves: vec![],
            foul_attempts: vec![],
            last_move: None,
            captured_square: None,
            outcome: None,
        }
    }

    fn player(&self, c: Color) -> &Player {
        &self.players[side_idx(c)]
    }

    fn is_bot(&self, c: Color) -> bool {
        matches!(self.player(c), Player::Bot(_))
    }

    /// 観戦ログ用の呼び名（"▲先手（estimator）"）
    fn label(&self, c: Color) -> String {
        format!("{}（{}）", mark(c), self.player(c).name())
    }

    /// 両者の観測ログへ、**その席自身の視点で**記録する。
    /// f は「その色の対局者から見た観測」を返す（人間側のログは入力プレーンの表示にだけ使う）
    fn record_each(&mut self, f: impl Fn(Color) -> Observation) {
        for c in SIDES {
            self.players[side_idx(c)].seat_mut().0.record(f(c));
        }
    }

    /// `side` の席の観測から作った入力プレーン（エンコーダが読むのと同じ3つ = 視界・観測ログ・
    /// 今の手番の反則試行から作る）
    fn planes(&self, side: Color) -> PlanesOut {
        let (log, foul_tried) = self.player(side).seat();
        PlanesOut::of(&self.pos, &self.fouls, side, log, foul_tried)
    }

    /// 振り返り用に、ここまでの対局を一時ファイルの `.kif` へ書き出してパスを返す（scenarios/ は汚さない）。
    /// リプレイ画面はパスから棋譜を読むので、候補手分析・入力プレーンの再生がそのまま使える
    fn review_file(&self) -> Result<String, String> {
        if self.moves.is_empty() && self.foul_attempts.is_empty() {
            return Err("まだ指し手がありません".into());
        }
        let text = self.kif_text(None, None)?;
        let dir = std::env::temp_dir().join("tsuitate-scenario-gui");
        std::fs::create_dir_all(&dir).map_err(|e| format!("{} を作れません: {e}", dir.display()))?;
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let path = dir.join(format!("review-{stamp}.kif"));
        std::fs::write(&path, text).map_err(|e| format!("{} に書けません: {e}", path.display()))?;
        Ok(path.to_string_lossy().into_owned())
    }

    fn finish(&mut self, winner: Color, reason: &str, events: &mut Vec<String>) {
        let who = match self.human {
            Some(h) if winner == h => "あなたの勝ち".to_string(),
            Some(_) => "botの勝ち".to_string(),
            None => format!("{}の勝ち", self.label(winner)),
        };
        events.push(format!("終局: {who}（{}）", reason_ja(reason)));
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

        // 観測: 指した側は MyMove、相手側は OpponentMoved（取られたマスだけが分かる）
        let usi_owned = usi.to_string();
        let captured_role = captured.map(unpromote_role);
        let captured_at = captured_sq.clone();
        self.record_each(|c| {
            if c == mover {
                Observation::MyMove {
                    move_number,
                    usi: usi_owned.clone(),
                    captured: captured_role,
                }
            } else {
                Observation::OpponentMoved {
                    move_number,
                    captured_my_piece_at: captured_at.clone(),
                }
            }
        });
        // 受理されたので、指した側の反則試行メモは破棄
        self.players[side_idx(mover)].seat_mut().1.clear();

        let captured_ja = captured.map(|r| role_kanji(unpromote_role(r)));
        match self.human {
            // 人間視点: 相手（bot）の手は「取られたマス」しか分からない
            Some(h) if mover == h => {
                self.captured_square = None;
                events.push(match captured_ja {
                    Some(k) => format!("あなた: {usi}（{k}を取りました）"),
                    None => format!("あなた: {usi}"),
                });
            }
            Some(_) => {
                events.push(match &captured_sq {
                    Some(sq) => format!("相手が着手し、あなたの駒が {sq} で取られました"),
                    None => "相手が着手しました".into(),
                });
                self.captured_square = captured_sq;
            }
            // 観戦: 真実をそのまま出す
            None => {
                let label = self.label(mover);
                events.push(match captured_ja {
                    Some(k) => format!("{label}: {usi}（{k}を取りました）"),
                    None => format!("{label}: {usi}"),
                });
            }
        }

        if self.pos.in_check(self.pos.turn()) {
            let in_check = self.pos.turn();
            self.record_each(|_| Observation::Check { in_check });
            events.push(format!("王手宣言: {}の玉に王手", mark(in_check)));
        }
        match self.pos.outcome() {
            Some(Outcome::Checkmate { winner }) => self.finish(winner, "checkmate", events),
            Some(Outcome::Stalemate { winner }) => self.finish(winner, "stalemate", events),
            None => {}
        }
    }

    /// 反則を1件計上する（手番は変えない）。10回目なら終局させる
    fn count_foul(&mut self, side: Color, usi: String, events: &mut Vec<String>) {
        let idx = side_idx(side);
        self.fouls[idx] += 1;
        let count = self.fouls[idx];
        let move_number = self.pos.move_number();
        self.foul_attempts.push((move_number, usi.clone()));

        // 観測: 反則した側は自分の手が分かる（理由は不明）、相手側は回数だけ届く
        let usi_owned = usi.clone();
        self.record_each(|c| {
            if c == side {
                Observation::MyFoul {
                    move_number,
                    usi: usi_owned.clone(),
                }
            } else {
                Observation::OpponentFoul { count }
            }
        });
        self.players[idx].seat_mut().1.insert(usi.clone());

        events.push(match self.human {
            Some(h) if side == h => format!("反則: {usi}（あなたの反則 累計{count}回）"),
            Some(_) => format!("相手が反則しました（累計{count}回）"),
            None => format!("{}: 反則 {usi}（累計{count}回）", self.label(side)),
        });
        if count >= MAX_FOULS {
            self.finish(side.other(), "foul_limit", events);
        }
    }

    fn human_move(&mut self, usi: &str) -> Result<Vec<String>, String> {
        if self.outcome.is_some() {
            return Err("対局は終了しています".into());
        }
        let Some(human) = self.human else {
            return Err("観戦モードでは人間が指せません".into());
        };
        if self.pos.turn() != human {
            return Err("あなたの手番ではありません".into());
        }
        let mv = parse_usi(usi).ok_or_else(|| format!("USIを解釈できません: {usi}"))?;
        let mut events = vec![];
        if self.pos.is_legal(&mv) {
            self.apply_move(usi, &mv, &mut events);
        } else {
            // 反則: 手番は変わらずカウント（サーバーと同じ。理由は通知されない）
            self.count_foul(human, usi.to_string(), &mut events);
        }
        Ok(events)
    }

    /// 手番側 bot の1手（受理されるか終局まで反則を繰り返す）。思考時間ぶんブロックする
    fn bot_move(&mut self) -> Result<Vec<String>, String> {
        if self.outcome.is_some() {
            return Err("対局は終了しています".into());
        }
        let side = self.pos.turn();
        if !self.is_bot(side) {
            return Err("botの手番ではありません".into());
        }
        let mut events = vec![];
        loop {
            let view = make_view(&self.pos, side, &self.fouls);
            let chosen = match &mut self.players[side_idx(side)] {
                Player::Bot(b) => b.strat.choose(&view, &b.log, &b.foul_tried),
                Player::Human(_) => unreachable!("bot 手番を確認済み"),
            };
            let Some(usi) = chosen else {
                events.push(match self.human {
                    Some(_) => "botが投了しました".into(),
                    None => format!("{}が投了しました", self.label(side)),
                });
                self.finish(side.other(), "resign", &mut events);
                return Ok(events);
            };
            if let Some(mv) = parse_usi(&usi).filter(|mv| self.pos.is_legal(mv)) {
                self.apply_move(&usi, &mv, &mut events);
                return Ok(events);
            }
            self.count_foul(side, usi, &mut events);
            if self.outcome.is_some() {
                return Ok(events);
            }
        }
    }

    fn resign(&mut self) -> Result<Vec<String>, String> {
        let Some(human) = self.human else {
            return Err("観戦モードでは投了できません（自動再生を止めて書き出してください）".into());
        };
        let mut events = vec![];
        if self.outcome.is_none() {
            events.push("あなたは投了しました".into());
            self.finish(human.other(), "resign", &mut events);
        }
        Ok(events)
    }

    /// 人間手番の入力候補（board.rs の move-hints 移植 = 自駒だけを考慮）
    fn hints(&self) -> Vec<PlayHint> {
        let Some(human) = self.human else {
            return vec![];
        };
        if self.outcome.is_some() || self.pos.turn() != human {
            return vec![];
        }
        let pieces = self.pos.pieces_of(human);
        let mut out = vec![];
        for p in &pieces {
            let Some(from) = parse_usi_square(&p.square) else {
                continue;
            };
            for t in move_targets(&pieces, p, human) {
                let promotion = match promotion_choice(p.role, from, t, human) {
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
        for (role, n) in self.pos.hand_map(human) {
            if n == 0 {
                continue;
            }
            for t in drop_targets(&pieces, role, human) {
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
            names: [self.players[0].name(), self.players[1].name()],
            seeds: [self.players[0].seed(), self.players[1].seed()],
            is_bot: [self.is_bot(Color::Sente), self.is_bot(Color::Gote)],
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
        let sente_name = self.players[0].name();
        let gote_name = self.players[1].name();
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
        let desc_text = desc
            .filter(|s| !s.trim().is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| match self.human {
                Some(h) => format!(
                    "GUI対局の再現（人間={}, bot={} seed={} 予算={}ms）",
                    if h == Color::Sente { "先手" } else { "後手" },
                    self.player(h.other()).name(),
                    self.player(h.other()).seed().unwrap_or(0),
                    self.budget_ms,
                ),
                None => format!(
                    "GUI観戦の再現（先手={} seed={} / 後手={} seed={} 予算={}ms）",
                    sente_name,
                    self.players[0].seed().unwrap_or(0),
                    gote_name,
                    self.players[1].seed().unwrap_or(0),
                    self.budget_ms,
                ),
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
        let session = PlaySession::new_vs_bot(engine, seed, budget_ms, human_color)?;
        let events = vec![format!(
            "対局開始: あなたは{}、bot={}（seed={} 予算={}ms）",
            mark(human_color),
            session.player(human_color.other()).name(),
            session.player(human_color.other()).seed().unwrap_or(0),
            session.budget_ms,
        )];
        let view = session.view(events);
        *slot = Some(session);
        Ok(view)
    })
    .await
}

/// bot 同士の対局を開始する（観戦モード）
#[tauri::command]
pub async fn play_start_bots(
    state: State<'_, PlayState>,
    sente_engine: String,
    sente_seed: u64,
    gote_engine: String,
    gote_seed: u64,
    budget_ms: u32,
) -> Result<PlayView, String> {
    on_session(&state, move |slot| {
        let session = PlaySession::new_bots(
            sente_engine,
            sente_seed,
            gote_engine,
            gote_seed,
            budget_ms,
        )?;
        let events = vec![format!(
            "観戦開始: {} vs {}（予算={}ms）",
            session.label(Color::Sente),
            session.label(Color::Gote),
            session.budget_ms,
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
        let events = session.resign()?;
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

/// `side` の席の観測から作った入力プレーン（方策ネットのエンコーダ v7 の全プレーン）。
/// 対人モードで bot 側を指定すると相手の駒の位置が見える（「真実を表示」と同じデバッグ用途）
#[tauri::command]
pub async fn play_planes(state: State<'_, PlayState>, side: Color) -> Result<PlanesOut, String> {
    on_session(&state, move |slot| Ok(require(slot)?.planes(side))).await
}

/// ここまでの対局を振り返り用の一時ファイルへ書き出し、そのパスを返す（リプレイ画面で開く）
#[tauri::command]
pub async fn play_review(state: State<'_, PlayState>) -> Result<String, String> {
    on_session(&state, |slot| require(slot)?.review_file()).await
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
            PlaySession::new_vs_bot("heuristic".into(), 7, 500, Color::Sente).expect("session");
        let human = s.human.unwrap();
        for _ in 0..60 {
            if s.outcome.is_some() {
                break;
            }
            if s.pos.turn() == human {
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
            PlaySession::new_vs_bot("estimator".into(), 3, 500, Color::Sente).expect("session");
        s.human_move("7g7f").unwrap();
        let events = s.bot_move().unwrap();
        assert_eq!(s.moves.len(), 2, "{events:?}");
        assert_eq!(s.pos.turn(), Color::Sente);
    }

    /// 人間の反則は手番を変えずカウントされ、観測が bot 側へ届くこと
    #[test]
    fn 人間の反則は手番維持でカウントされkifに残る() {
        let mut s =
            PlaySession::new_vs_bot("heuristic".into(), 1, 500, Color::Sente).expect("session");
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

    /// bot 同士でも 1手送りで交互に進み、書き出した KIF が裁定つき replay を通ること
    #[test]
    fn bot同士の対局が交互に進みkifがreplayを通る() {
        let mut s = PlaySession::new_bots("heuristic".into(), 5, "heuristic".into(), 6, 300)
            .expect("session");
        assert!(s.human.is_none());
        assert!(s.hints().is_empty(), "観戦では人間の候補手は出ない");
        assert!(s.human_move("7g7f").is_err(), "観戦では人間が指せない");
        for _ in 0..40 {
            if s.outcome.is_some() {
                break;
            }
            s.bot_move().unwrap();
        }
        assert!(s.moves.len() >= 2);
        let kifu = parse_kif(&s.kif_text(None, None).unwrap()).unwrap();
        assert_eq!(kifu.plies.len(), s.moves.len());
        let rep = replay(&kifu, kifu.plies.len());
        if s.outcome.as_ref().map(|o| o.reason.as_str()) != Some("foul_limit") {
            assert_eq!(rep.fouls, s.fouls);
        }
    }

    /// bot 同士でも観測ログは各 bot 独立で、自分の手は MyMove・相手の手は
    /// OpponentMoved として届く（＝互いの盤面は覗けない）
    #[test]
    fn bot同士の観測ログは視点ごとに独立している() {
        let mut s = PlaySession::new_bots("heuristic".into(), 1, "heuristic".into(), 2, 300)
            .expect("session");
        s.bot_move().unwrap(); // 先手が1手
        s.bot_move().unwrap(); // 後手が1手
        let mut kinds = vec![];
        for c in SIDES {
            let Player::Bot(b) = s.player(c) else {
                unreachable!()
            };
            let obs: Vec<&'static str> = b
                .log
                .events()
                .iter()
                .map(|o| match o {
                    Observation::MyMove { .. } => "my",
                    Observation::OpponentMoved { .. } => "opp",
                    _ => "other",
                })
                .collect();
            kinds.push(obs);
        }
        // 先手は「自分→相手」、後手は「相手→自分」の順で観測している
        assert_eq!(kinds[0], vec!["my", "opp"], "先手の観測");
        assert_eq!(kinds[1], vec!["opp", "my"], "後手の観測");
    }

    fn plane<'a>(out: &'a PlanesOut, label: &str) -> &'a PlaneView {
        out.planes
            .iter()
            .find(|p| p.label == label)
            .unwrap_or_else(|| panic!("プレーン「{label}」が無い"))
    }

    fn squares(p: &PlaneView) -> Vec<&str> {
        p.cells.keys().map(String::as_str).collect()
    }

    /// 入力プレーンは席ごとの観測から作られる。人間の席にも観測ログと反則試行メモがあり、
    /// 反則した手番の途中・受理された後・相手の席のどれでも、その席から見た値になる
    #[test]
    fn 入力プレーンは席ごとの観測から作られる() {
        let mut s =
            PlaySession::new_vs_bot("heuristic".into(), 1, 500, Color::Sente).expect("session");
        s.human_move("5i5g").unwrap(); // 反則（手番は変わらない）
        let mid = s.planes(Color::Sente);
        assert!(mid.to_move);
        assert_eq!(plane(&mid, "今の手番の反則数").note.as_deref(), Some("1回"));
        assert_eq!(squares(plane(&mid, "今の手番の反則: 移動先")), ["5g"]);
        assert_eq!(squares(plane(&mid, "今の手番の反則: 移動元")), ["5i"]);

        s.human_move("7g7f").unwrap();
        s.bot_move().unwrap();
        let mine = s.planes(Color::Sente);
        assert_eq!(mine.move_number, 3);
        assert!(mine.to_move);
        assert_eq!(squares(plane(&mine, "自駒 玉")), ["5i"]);
        assert_eq!(squares(plane(&mine, "1手前の移動元")), ["7g"]);
        assert_eq!(squares(plane(&mine, "1手前の移動先")), ["7f"]);
        assert_eq!(plane(&mine, "自分の反則数").note.as_deref(), Some("1回"));
        assert_eq!(plane(&mine, "今の手番の反則数").note.as_deref(), Some("0回"));
        assert_eq!(plane(&mine, "手数").note.as_deref(), Some("3手目"));
        // 受理された後は「過去の反則」に移る
        assert!(squares(plane(&mine, "今の手番の反則: 移動先")).is_empty());
        assert_eq!(squares(plane(&mine, "過去の反則: 移動先（鮮度）")), ["5g"]);

        // bot の席（後手）: 盤は実際のマスのまま（180度回した値を戻してある）
        let theirs = s.planes(Color::Gote);
        assert!(!theirs.to_move);
        assert_eq!(squares(plane(&theirs, "自駒 玉")), ["5a"]);
        assert_eq!(plane(&theirs, "相手の反則数").note.as_deref(), Some("1回"));
        assert_eq!(plane(&theirs, "自分の反則数").note.as_deref(), Some("0回"));
    }

    fn hint_usi(h: &PlayHint) -> String {
        match &h.from {
            Some(from) => format!("{from}{}{}", h.to, if h.promotion == "forced" { "+" } else { "" }),
            None => tsuitate_bot::board::make_usi_drop(h.role, parse_usi_square(&h.to).unwrap()).unwrap(),
        }
    }

    /// 「当時の入力」の再生: 振り返り用に書き出した棋譜から `kifu_planes_of` で作る入力は、対局中にその場で
    /// 見えていた入力（`planes`）と同じになる。手番の頭・人間が反則した後・終局行の後に残る反則の後、の
    /// どの時点でも、両方の席で比べる
    #[test]
    fn 棋譜から再生した入力は対局中の入力と同じ() {
        let mut s =
            PlaySession::new_vs_bot("heuristic".into(), 3, 300, Color::Sente).expect("session");
        // ((手数, その手番の反則の数), [先手の席, 後手の席] の入力)
        let mut live: Vec<((usize, usize), [Vec<PlaneView>; 2])> = vec![];
        fn snap(live: &mut Vec<((usize, usize), [Vec<PlaneView>; 2])>, s: &PlaySession, k: usize) {
            live.push((
                (s.moves.len(), k),
                [s.planes(Color::Sente).planes, s.planes(Color::Gote).planes],
            ));
        }
        // 人間の手番: 自駒だけを見た候補のうち真の盤では反則になる手を、手番ごとに最大2回試してから合法手を指す
        let human_fouls = |s: &mut PlaySession, live: &mut Vec<((usize, usize), [Vec<PlaneView>; 2])>| {
            let fouls: Vec<String> = s
                .hints()
                .iter()
                .map(hint_usi)
                .filter(|u| parse_usi(u).is_some_and(|mv| !s.pos.is_legal(&mv)))
                .take(2)
                .collect();
            for (k, usi) in fouls.iter().enumerate() {
                if s.outcome.is_some() {
                    break;
                }
                s.human_move(usi).unwrap();
                snap(live, s, k + 1);
            }
            fouls.len()
        };
        let mut total_fouls = 0;
        snap(&mut live, &s, 0);
        while s.outcome.is_none() && s.moves.len() < 44 {
            if s.pos.turn() == Color::Sente {
                // 反則負けにならない範囲で反則を混ぜる
                if s.fouls[0] < 6 {
                    total_fouls += human_fouls(&mut s, &mut live);
                }
                let mv = s.pos.legal_moves().into_iter().next().expect("合法手");
                s.human_move(&mv.to_usi()).unwrap();
            } else {
                s.bot_move().unwrap();
            }
            if s.outcome.is_none() {
                snap(&mut live, &s, 0);
            }
        }
        // 最後は人間の手番の反則で止める（終局行の後に残る反則試行の再生）
        if s.outcome.is_none() && s.pos.turn() == Color::Sente && s.fouls[0] < 8 {
            total_fouls += human_fouls(&mut s, &mut live);
        }
        assert!(total_fouls >= 3, "反則を挟んだ手番が足りない（{total_fouls} 回）");

        let path = s.review_file().unwrap();
        let kifu = parse_kif(&std::fs::read_to_string(&path).unwrap()).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(kifu.plies.len(), s.moves.len());
        for ((ply, k), [sente, gote]) in &live {
            for (side, want) in [(Color::Sente, sente), (Color::Gote, gote)] {
                let got = crate::kifu_planes_of(&kifu, *ply, *k, side).planes;
                assert!(got == *want, "{ply}手・反則{k}回の後・{side:?}の席の入力が対局中と違う");
            }
        }
        assert!(live.len() >= 20, "比べた時点が少ない（{}）", live.len());
    }

    /// ブラウザ単体の UI 確認（src/mock.ts、`?mock=1`）用の、対局モードのフィクスチャ生成。
    /// 人間の席を方策ネットに指させて中盤まで進めた対人の局面と、両方の席の入力プレーンを書く。
    /// `FIXTURE_DIR=$(pwd)/public/fixtures cargo test --manifest-path src-tauri/Cargo.toml -- --ignored fixtures`
    #[test]
    #[ignore]
    fn dump_play_fixtures() {
        let out = std::env::var("FIXTURE_DIR").expect("FIXTURE_DIR を指定してください");
        std::fs::create_dir_all(&out).unwrap();
        let engine = "rl_scratch5_v1";
        let mut s = PlaySession::new_vs_bot(engine.into(), 11, 500, Color::Sente).expect("session");
        let mut stand_in = strategy::make_seeded(engine, 12).expect("人間の席の代役");
        // 王手を受けた人間の手番（55手目以降）で止める。来なければ 90手目で止める
        while s.outcome.is_none() {
            let human_turn = s.pos.turn() == Color::Sente;
            let in_check = s.pos.in_check(Color::Sente);
            if human_turn && ((s.moves.len() >= 54 && in_check) || s.moves.len() >= 90) {
                break;
            }
            if human_turn {
                let view = make_view(&s.pos, Color::Sente, &s.fouls);
                let (log, tried) = s.player(Color::Sente).seat();
                let usi = stand_in.choose(&view, log, tried).expect("代役の手");
                s.human_move(&usi).unwrap();
            } else {
                s.bot_move().unwrap();
            }
        }
        let write = |name: &str, v: serde_json::Value| {
            std::fs::write(format!("{out}/{name}.json"), serde_json::to_string(&v).unwrap()).unwrap();
        };
        let events = vec![format!("（モック）{}手目まで進めた対人の局面", s.moves.len())];
        write("play_view", serde_json::to_value(s.view(events)).unwrap());
        write("play_planes_sente", serde_json::to_value(s.planes(Color::Sente)).unwrap());
        write("play_planes_gote", serde_json::to_value(s.planes(Color::Gote)).unwrap());
    }
}
