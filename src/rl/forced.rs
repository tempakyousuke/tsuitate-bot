//! **序盤の強制**（学習の自己対局で序盤を散らす。docs/rl-opening-diversity.md）。
//!
//! 方策ネットの序盤は約40,000更新から飛車先の歩の速攻1本に固まり、ほかの手の確率は 1e-18
//! まで落ちている（指されないので学ぶ機会もない）。そこで局ごと・**先後それぞれ独立に**確率 p で、
//! その側の最初の数手を定跡のラインから指させる。強制した手は env の中で指す（`EnvGame::settle`）
//! ので学習側の行には現れず、学習器から見ると「環境の手」（チャンス手）になる。
//!
//! - **独立に引く**のが要点: 強制されなかった側は、相手が強制されたかどうかを自分の観測から
//!   推せない（駒がぶつかるまで相手の手は観測に入らない）。両者をまとめて強制すると、
//!   「自分の序盤が定跡なら相手も定跡」という相関が漏れる
//! - **抜ける条件は `crate::opening::OpeningBook::next` と同じ**: どちらかの駒取り・王手
//!   （= 両者に見える接触）、自分の反則、ラインを指し終えた。ただし定跡の最後の**駒取りの手は
//!   強制しない**（初期配置の相手の駒に触れる手の手前で切る）。ぶつかった後の判断は方策が指す
//! - 定跡の手が実際には反則（見えない相手の駒に遮られた等）なら、**そのまま反則として指して**
//!   から抜ける（`opening.rs` と同じ）。真実の盤面で合法かを先に調べて黙って抜けると、
//!   方策が指し始める局面が見えない情報と相関する
//! - `opening.rs` と `joseki.json` は凍結版の共有 pin（`frozen::SHARED_MODEL_PINS`）なので
//!   触らない。ラインは呼び出し側（Python の学習）が読んで渡す

use std::sync::Arc;

use crate::board::{Coord, Promotion, make_usi_square, move_targets, parse_usi_square, promotion_choice};
use crate::protocol::Color;
use crate::rl::action::{decode_usi, encode_usi};
use crate::selfplay::mix;
use crate::shogi::{Position, ShogiMove, parse_usi, promote_role};

/// 序盤の乱数列の塩（相手戦略の seed `mix(seed ^ mix(game_no))` と混ざらないように）
const OPENING_SALT: u64 = 0x6f70_656e_696e_6721;
/// 本番bot で対局ID から乱数列を作るときの初期値
const GAME_ID_SALT: u64 = 0x6761_6d65_5f69_6421;

/// どの側を強制の対象にするか
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForcedSides {
    /// 先後どちらも（それぞれ独立に確率 p）
    Both,
    Sente,
    Gote,
    /// 評価モードの学習側だけ
    Learner,
    /// 評価モードの相手側だけ
    Opponent,
}

impl ForcedSides {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "both" => Some(Self::Both),
            "sente" => Some(Self::Sente),
            "gote" => Some(Self::Gote),
            "learner" => Some(Self::Learner),
            "opponent" => Some(Self::Opponent),
            _ => None,
        }
    }

    /// 学習側と相手側を区別する指定か（自己対局の env では使えない）
    pub fn needs_learner(self) -> bool {
        matches!(self, Self::Learner | Self::Opponent)
    }

    fn eligible(self, color: Color, learner: Option<Color>) -> bool {
        match self {
            Self::Both => true,
            Self::Sente => color == Color::Sente,
            Self::Gote => color == Color::Gote,
            Self::Learner => learner == Some(color),
            Self::Opponent => learner.is_some_and(|l| l != color),
        }
    }
}

/// 1局の1つの側に割り当てた強制の予定
#[derive(Debug, Clone)]
pub struct ForcedPlan {
    /// 先手視点で正規化した行動番号の並び（`action::encode_usi(_, Sente)`。後手は同じ番号を
    /// `decode_usi(_, Gote)` で読むと点対称にミラーした手になる）
    pub line: Arc<[usize]>,
    /// ブックの中のラインの添字（ラインの名前は一意でないので添字で数える）
    pub line_id: u32,
    /// 強制する手数の上限（≤ ラインの長さ）
    pub limit: u32,
}

/// 強制の材料（ラインの集まりと引き方）
#[derive(Debug, Clone)]
pub struct ForcedBook {
    lines: Vec<Arc<[usize]>>,
    /// 重みの累積（最後が 1）
    cdf: Vec<f64>,
    prob: f64,
    len: Option<(u32, u32)>,
    sides: ForcedSides,
}

impl ForcedBook {
    /// `lines` は先手視点の USI の並び（`joseki.json` の `moves`）。各ラインは自駒だけの盤で
    /// 指し切れるかを検査し、**初期配置の相手の駒に触れる手の手前で切る**（`quiet_prefix`）。
    /// `weights` は None なら一様。`len` は None ならライン（切った後）を最後まで、
    /// `Some((lo, hi))` なら手数を {lo..hi} から一様に引いてラインの長さで頭打ちにする
    pub fn new(
        lines: &[Vec<String>],
        weights: Option<&[f64]>,
        prob: f64,
        len: Option<(u32, u32)>,
        sides: ForcedSides,
    ) -> Result<Self, String> {
        if !(prob.is_finite() && (0.0..=1.0).contains(&prob)) {
            return Err(format!("opening_prob は 0〜1（{prob}）"));
        }
        if lines.is_empty() {
            return Err("openings が空".into());
        }
        if let Some((lo, hi)) = len {
            if lo == 0 || lo > hi {
                return Err(format!("opening_len は 1 ≤ lo ≤ hi（{lo}, {hi}）"));
            }
        }
        let mut encoded = Vec::with_capacity(lines.len());
        for (i, line) in lines.iter().enumerate() {
            let quiet = quiet_prefix(line).map_err(|e| format!("ライン {i}: {e}"))?;
            let actions: Vec<usize> = quiet
                .iter()
                .map(|u| {
                    let a = encode_usi(u, Color::Sente).ok_or_else(|| format!("ライン {i}: {u} を符号化できない"))?;
                    // 正規の表記か（成りの指定など）。記録・反則済みの手の照合は文字列で行うので、
                    // 方策が同じ手を選んだときと同じ文字列になることを保証する
                    if decode_usi(a, Color::Sente).as_deref() != Some(u.as_str()) {
                        return Err(format!("ライン {i}: {u} は正規の USI ではない"));
                    }
                    Ok(a)
                })
                .collect::<Result<_, String>>()?;
            encoded.push(Arc::from(actions));
        }
        let weights: Vec<f64> = match weights {
            None => vec![1.0; lines.len()],
            Some(w) if w.len() != lines.len() => {
                return Err(format!("opening_weights の長さ {} がラインの数 {} と違う", w.len(), lines.len()));
            }
            Some(w) => w.to_vec(),
        };
        if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err("opening_weights は有限で 0 以上".into());
        }
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Err("opening_weights の合計が 0".into());
        }
        let mut acc = 0.0;
        let mut cdf: Vec<f64> = weights
            .iter()
            .map(|w| {
                acc += w / total;
                acc
            })
            .collect();
        *cdf.last_mut().expect("ラインは1本以上") = 1.0;
        Ok(Self { lines: encoded, cdf, prob, len, sides })
    }

    /// **本番bot用**: 対局ID から自分の側の予定を引く（同じ対局IDなら何度呼んでも同じ予定 =
    /// webhook のセッション復元・作り直しでも同じラインを指し続ける）
    pub fn plan_for_game(&self, game_id: &str, color: Color) -> Option<ForcedPlan> {
        let seed = game_id.bytes().fold(GAME_ID_SALT, |h, b| mix(h ^ u64::from(b)));
        let [sente, gote] = self.plans(seed, 0, None);
        match color {
            Color::Sente => sente,
            Color::Gote => gote,
        }
    }

    pub fn sides(&self) -> ForcedSides {
        self.sides
    }

    /// ライン `line_id` を（切った後の）最後まで指す予定。確率・手数の引き方は見ない
    /// （`open:<ライン>:<戦略名>` の相手が毎局同じラインを指すため）
    pub fn fixed_plan(&self, line_id: usize) -> Option<ForcedPlan> {
        let line = self.lines.get(line_id)?;
        Some(ForcedPlan { line: Arc::clone(line), line_id: line_id as u32, limit: line.len() as u32 })
    }

    /// 切った後の各ラインの手数（検査・表示用）
    pub fn line_lengths(&self) -> Vec<usize> {
        self.lines.iter().map(|l| l.len()).collect()
    }

    /// 局 `k`（`seed` の乱数列の中の通し番号）の [先手, 後手] の予定。`learner` は評価モードの
    /// 学習側の色（自己対局は None）。
    ///
    /// 乱数は側ごとに3つ（強制するか・ライン・手数）を**常に**引く: p を変えても、両方の p で
    /// 強制される側のラインと手数は同じになる（p の比較が対になる）
    pub fn plans(&self, seed: u64, k: u64, learner: Option<Color>) -> [Option<ForcedPlan>; 2] {
        let base = mix(mix(seed ^ OPENING_SALT) ^ mix(k));
        let unit = |j: u64| (mix(base.wrapping_add(j)) >> 11) as f64 / (1u64 << 53) as f64;
        let mut out = [None, None];
        for (ci, color) in [Color::Sente, Color::Gote].into_iter().enumerate() {
            let j = 3 * ci as u64;
            let (u_force, u_line, u_len) = (unit(j), unit(j + 1), unit(j + 2));
            if !self.sides.eligible(color, learner) || u_force >= self.prob {
                continue;
            }
            let line_id = self.cdf.iter().position(|&c| u_line < c).unwrap_or(self.cdf.len() - 1);
            let line = &self.lines[line_id];
            let n = line.len() as u32;
            let limit = match self.len {
                None => n,
                Some((lo, hi)) => (lo + ((u_len * f64::from(hi - lo + 1)) as u32).min(hi - lo)).min(n),
            };
            out[ci] = Some(ForcedPlan { line: Arc::clone(line), line_id: line_id as u32, limit });
        }
        out
    }
}

/// 定跡ファイル（`rl_openings.json` / `joseki.json` の形 = `{"lines":[{"moves":[..],"category"?,"weight"?}]}`）を
/// 読んでブックを作る。重みは学習（`train_rnad.py --open-weight category`）の既定と同じく
/// **系統（category）ごとに同じ確率**で、系統の中はラインの `weight`（無ければ 1）に比例させる。
/// category の無いラインはそれぞれ1系統として数える
pub fn book_from_json(json: &str, prob: f64, len: Option<(u32, u32)>, sides: ForcedSides) -> Result<ForcedBook, String> {
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| format!("定跡ファイルが JSON でない: {e}"))?;
    let entries = v["lines"].as_array().ok_or("定跡ファイルに lines が無い")?;
    let mut lines = Vec::with_capacity(entries.len());
    let mut cats = Vec::with_capacity(entries.len());
    let mut raw = Vec::with_capacity(entries.len());
    for (i, l) in entries.iter().enumerate() {
        let moves = l["moves"]
            .as_array()
            .ok_or_else(|| format!("ライン {i} に moves が無い"))?
            .iter()
            .map(|m| m.as_str().map(str::to_string).ok_or_else(|| format!("ライン {i} の手が文字列でない")))
            .collect::<Result<Vec<_>, _>>()?;
        lines.push(moves);
        cats.push(l["category"].as_str().map_or_else(|| format!("#{i}"), str::to_string));
        raw.push(l.get("weight").and_then(serde_json::Value::as_f64).unwrap_or(1.0));
    }
    let mut cat_total: std::collections::HashMap<&str, f64> = std::collections::HashMap::new();
    for (c, w) in cats.iter().zip(&raw) {
        *cat_total.entry(c).or_default() += w;
    }
    let weights: Vec<f64> = cats
        .iter()
        .zip(&raw)
        .map(|(c, w)| {
            let t = cat_total[c.as_str()];
            if t > 0.0 { w / t } else { 0.0 }
        })
        .collect();
    ForcedBook::new(&lines, Some(&weights), prob, len, sides)
}

/// **本番bot の序盤の強制**（`main.rs` と `bin/webhook_bot.rs` が起動時に1回読む）。
///
/// - `TSUITATE_OPEN_BOOK`: 定跡ファイルのパス（未設定・空ならオフ = Ok(None)）
/// - `TSUITATE_OPEN_P`: 1局で自分の側を強制する確率（既定 1.0）
/// - `TSUITATE_OPEN_LEN`: `lo:hi` で強制する手数を局ごとに一様に引く（既定はラインを接触の手前まで）
///
/// ラインは対局ID から決める（`ForcedBook::plan_for_game`）。止め方は学習・arena と同じ（`ForcedOpening`）。
/// 戦略の強さのノブ（`config::STRATEGY_ENV_KEYS`）ではなく bot の運用設定なので config.rs は通さない
pub fn bot_book_from_env() -> Result<Option<ForcedBook>, String> {
    let Some(path) = std::env::var("TSUITATE_OPEN_BOOK").ok().filter(|s| !s.trim().is_empty()) else {
        return Ok(None);
    };
    let json = std::fs::read_to_string(&path).map_err(|e| format!("TSUITATE_OPEN_BOOK: {path} を読めません: {e}"))?;
    let prob = match std::env::var("TSUITATE_OPEN_P").ok().filter(|s| !s.trim().is_empty()) {
        None => 1.0,
        Some(s) => s.trim().parse::<f64>().map_err(|_| format!("TSUITATE_OPEN_P が数でない: {s}"))?,
    };
    let len = match std::env::var("TSUITATE_OPEN_LEN").ok().filter(|s| !s.trim().is_empty()) {
        None => None,
        Some(s) => Some(
            s.trim()
                .split_once(':')
                .and_then(|(lo, hi)| Some((lo.parse().ok()?, hi.parse().ok()?)))
                .ok_or_else(|| format!("TSUITATE_OPEN_LEN は lo:hi の形: {s}"))?,
        ),
    };
    book_from_json(&json, prob, len, ForcedSides::Both).map(Some).map_err(|e| format!("TSUITATE_OPEN_BOOK: {path}: {e}"))
}

/// 本番bot の対局ごとの戦略を、予定があれば `ForcedOpening` で包む（予定が無ければそのまま返す）
pub fn wrap_for_game(
    inner: Box<dyn crate::strategy::Strategy + Send>,
    book: Option<&ForcedBook>,
    game_id: &str,
    color: Color,
) -> Box<dyn crate::strategy::Strategy + Send> {
    match book.and_then(|b| b.plan_for_game(game_id, color)) {
        Some(plan) => Box::new(ForcedOpening::new(inner, plan)),
        None => inner,
    }
}

/// 定跡を固定した相手の戦略名の接頭辞（`open:<ライン>:<戦略名>`）
pub const OPEN_PREFIX: &str = "open:";

/// 学習用のブック（`rl_openings.json`）。`open:` の相手のラインはここの添字で指す
fn training_book() -> &'static ForcedBook {
    static B: std::sync::OnceLock<ForcedBook> = std::sync::OnceLock::new();
    B.get_or_init(|| {
        book_from_json(include_str!("../../rl_openings.json"), 1.0, None, ForcedSides::Both)
            .expect("rl_openings.json は検査済み（学習用の定跡ファイルの全ラインが通る）")
    })
}

/// **定跡を固定した相手**（arena の基準用、2026-11-07）。`open:<ライン>:<戦略名>` は、中の戦略に
/// `rl_openings.json` のライン <ライン>（0 始まりの添字）を毎局・先後どちらでも指させる
/// （後手は点対称に読む）。止め方は学習の強制と同じ（`ForcedOpening`。接触・王手・自分の反則で抜ける）。
///
/// 用途は「強制なしの方策が苦手な序盤」を探すこと（候補は強制なし、相手だけ固定）と、苦手だった
/// ラインを固定の相手としてガントレットに混ぜて区切りごとの変化を見ること。ラインは添字で指すので、
/// `rl_openings.json` にラインを足すときは**末尾に足す**（途中に挟むと過去の記録の名前とずれる）。
/// 中の戦略名は凍結版などの名前（`rl_policy:release:...` は arena.yml の重みの取得が拾わない）。
/// 読めない名前・範囲外のラインは None
///
/// **`open:book:<戦略名>`**（2026-11-12）は、ラインを固定せずに**局ごとにブックから1本引く**相手
/// （系統ごとに同じ確率 = 本番bot の `TSUITATE_OPEN_BOOK`・p = 1 と同じ引き方）。「定跡に従う相手」を
/// 1つの基準として測る用で、ラインは局の seed から決まる（seed なしはエントロピー）
pub fn fixed_opening_from_name(name: &str, seed: Option<u64>) -> Option<Box<dyn crate::strategy::Strategy + Send>> {
    let (line, inner) = name.strip_prefix(OPEN_PREFIX)?.split_once(':')?;
    let plan = if line == "book" {
        use rand::Rng;
        let s = seed.unwrap_or_else(|| rand::rng().random());
        // p = 1・両側が対象のブックなので必ず予定が付く。ラインは先手視点の並びで、後手は点対称に読む
        let [plan, _] = training_book().plans(mix(s ^ OPENING_SALT), 0, None);
        plan?
    } else {
        training_book().fixed_plan(line.parse().ok()?)?
    };
    let inner = match seed {
        Some(s) => crate::strategy::make_seeded(inner, s),
        None => crate::strategy::make(inner),
    }?;
    Some(Box::new(ForcedOpening::new(inner, plan)))
}

/// **序盤を強制する戦略の包み**（arena の自己対局用。`selfplay` が `ARENA_OPEN_BOOK` のときに使う）。
///
/// 予定（`ForcedPlan`）がある間はラインの手を返し、`EnvGame::forced_move` と同じ条件でやめて以後は
/// 中の戦略に任せる: どちらかの駒取り・王手（自分の観測に入る接触）、自分の反則、予定の手数を指し終えた。
/// 判定に使うのは自分の `PlayerView` と観測ログだけ（相手の強制の有無は分からない = 学習と同じ情報条件）。
/// 中の戦略は強制した手番では呼ばれないが、推定器は観測ログを差分で食うので後から追いつく
///
/// 中の戦略の型は引数にしてある（arena は `dyn Strategy`、本番bot は webhook のスレッドをまたぐので
/// `dyn Strategy + Send`）
pub struct ForcedOpening<S: crate::strategy::Strategy + ?Sized = dyn crate::strategy::Strategy> {
    inner: Box<S>,
    plan: ForcedPlan,
    stopped: bool,
    /// 直前の choose が強制の手だったか（`debug_state` を中の戦略の古い値にしないため）
    last_forced: bool,
}

impl<S: crate::strategy::Strategy + ?Sized> ForcedOpening<S> {
    pub fn new(inner: Box<S>, plan: ForcedPlan) -> Self {
        Self { inner, plan, stopped: false, last_forced: false }
    }

    fn forced_move(
        &mut self,
        view: &crate::protocol::PlayerView,
        log: &crate::observation::ObservationLog,
        foul_tried: &std::collections::HashSet<String>,
    ) -> Option<String> {
        use crate::observation::Observation;
        if self.stopped {
            return None;
        }
        let mut played = 0u32;
        for e in log.events() {
            match e {
                Observation::MyMove { captured: None, .. } => played += 1,
                Observation::MyMove { captured: Some(_), .. }
                | Observation::OpponentMoved { captured_my_piece_at: Some(_), .. }
                | Observation::Check { .. }
                | Observation::MyFoul { .. } => {
                    self.stopped = true;
                    return None;
                }
                Observation::OpponentMoved { .. } | Observation::OpponentFoul { .. } => {}
            }
        }
        let color = view.your_color;
        let next = self
            .plan
            .line
            .get(played as usize)
            .filter(|_| played < self.plan.limit)
            .copied()
            .filter(|&a| crate::rl::action::legal_mask(view, log, foul_tried).get(a).copied().unwrap_or(false))
            .and_then(|a| decode_usi(a, color));
        if next.is_none() {
            self.stopped = true;
        }
        next
    }
}

impl<S: crate::strategy::Strategy + ?Sized> crate::strategy::Strategy for ForcedOpening<S> {
    fn choose(
        &mut self,
        view: &crate::protocol::PlayerView,
        log: &crate::observation::ObservationLog,
        foul_tried: &std::collections::HashSet<String>,
    ) -> Option<String> {
        if let Some(usi) = self.forced_move(view, log, foul_tried) {
            self.last_forced = true;
            return Some(usi);
        }
        self.last_forced = false;
        self.inner.choose(view, log, foul_tried)
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn debug_state(&self) -> Option<serde_json::Value> {
        if self.last_forced { None } else { self.inner.debug_state() }
    }

    fn prewarm(&mut self, view: &crate::protocol::PlayerView, log: &crate::observation::ObservationLog) {
        self.inner.prewarm(view, log);
    }
}

/// 先手視点のラインを自駒だけの盤で指し進めて検査し、**初期配置の後手の駒に触れる手**
/// （行き先か、飛び駒の通り道に後手の初期配置の駒がある手 = 相手が動いていなければ駒取りか
/// 反則になる手）の手前で切る。
///
/// 検査は `opening.rs` の `loaded_lines_replay_on_own_view` と同じ（移動元に自駒がある・
/// その駒がそこへ動ける・成りの指定が正しい）。打ちは扱わない。切った結果が空ならエラー
pub fn quiet_prefix(line: &[String]) -> Result<Vec<String>, String> {
    let initial = Position::initial();
    let mut pieces = initial.pieces_of(Color::Sente);
    let mut out = vec![];
    for usi in line {
        let Some(ShogiMove::Board { from, to, promote }) = parse_usi(usi) else {
            return Err(format!("{usi} は盤上の手ではない（打ちは使えない）"));
        };
        let idx = pieces
            .iter()
            .position(|p| parse_usi_square(&p.square) == Some(from))
            .ok_or_else(|| format!("{usi}: 移動元に自駒がない"))?;
        if !move_targets(&pieces, &pieces[idx], Color::Sente).contains(&to) {
            return Err(format!("{usi}: その駒はそこへ動けない"));
        }
        let choice = promotion_choice(pieces[idx].role, from, to, Color::Sente);
        match (promote, choice) {
            (true, Promotion::None) => return Err(format!("{usi}: 成れない手に + が付いている")),
            (false, Promotion::Forced) => return Err(format!("{usi}: 成りが強制の手なのに + がない")),
            _ => {}
        }
        if touches_initial_opponent(&initial, from, to) {
            break;
        }
        if promote {
            pieces[idx].role = promote_role(pieces[idx].role).ok_or_else(|| format!("{usi}: 成れない駒種"))?;
        }
        pieces[idx].square = make_usi_square(to);
        out.push(usi.clone());
    }
    if out.is_empty() {
        return Err("初手から相手の初期配置の駒に触れる（強制できる手が無い）".into());
    }
    Ok(out)
}

/// 行き先か、飛び駒の通り道（両端を除く）に後手の初期配置の駒があるか
fn touches_initial_opponent(initial: &Position, from: Coord, to: Coord) -> bool {
    let gote_at = |c: Coord| initial.piece_at(c).is_some_and(|p| p.color == Color::Gote);
    if gote_at(to) {
        return true;
    }
    let (df, dr) = (to.file - from.file, to.rank - from.rank);
    // 桂の跳びは通り道を持たない（直線・斜めの動きだけ途中のマスを見る）
    if df != 0 && dr != 0 && df.abs() != dr.abs() {
        return false;
    }
    let (sf, sr) = (df.signum(), dr.signum());
    let mut c = Coord { file: from.file + sf, rank: from.rank + sr };
    while c != to {
        if gote_at(c) {
            return true;
        }
        c = Coord { file: c.file + sf, rank: c.rank + sr };
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(moves: &[&str]) -> Vec<String> {
        moves.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn 初期配置の相手の駒に触れる手の手前で切る() {
        // 居飛車速攻: 2三歩成（2c の後手の歩を取る）の手前まで
        assert_eq!(quiet_prefix(&line(&["2g2f", "2f2e", "2e2d", "2d2c+"])).unwrap(), line(&["2g2f", "2f2e", "2e2d"]));
        // 中飛車の桂の跳び（行き先 5c に後手の歩）
        let nakabisha = line(&["7g7f", "2h5h", "5g5f", "8h6f", "8i7g", "6f7e", "5f5e", "7g6e", "5e5d", "6e5c"]);
        assert_eq!(quiet_prefix(&nakabisha).unwrap().len(), 9);
        // 飛び駒の通り道に後手の初期配置の駒（角道 8h→2b は 3c の歩に遮られる）
        assert_eq!(quiet_prefix(&line(&["7g7f", "8h2b+"])).unwrap(), line(&["7g7f"]));
        // 空いたマスへの角の移動は切らない
        assert_eq!(quiet_prefix(&line(&["7g7f", "8h6f"])).unwrap().len(), 2);
    }

    #[test]
    fn 不正なラインは理由つきで拒む() {
        assert!(quiet_prefix(&line(&["P*5e"])).unwrap_err().contains("打ち"));
        assert!(quiet_prefix(&line(&["5e5d"])).unwrap_err().contains("移動元"));
        assert!(quiet_prefix(&line(&["2h2c"])).unwrap_err().contains("動けない")); // 自分の歩に遮られる
        assert!(quiet_prefix(&line(&["7g7f+"])).unwrap_err().contains("成れない"));
        assert!(quiet_prefix(&line(&["2h2b+"])).unwrap_err().contains("動けない"));
        // 切る位置より後ろは読まない（解釈できない手があっても通る）
        assert_eq!(quiet_prefix(&line(&["2g2f", "2f2e", "2e2d", "2d2c+", "x"])).unwrap().len(), 3);
        assert!(ForcedBook::new(&[line(&["7g7f", "2d2c+"])], None, 0.5, None, ForcedSides::Both)
            .unwrap_err()
            .contains("ライン 0"));
        assert!(ForcedBook::new(&[line(&["2g2f"])], Some(&[1.0, 2.0]), 0.5, None, ForcedSides::Both).is_err());
        assert!(ForcedBook::new(&[line(&["2g2f"])], Some(&[0.0]), 0.5, None, ForcedSides::Both).is_err());
        assert!(ForcedBook::new(&[line(&["2g2f"])], None, 1.5, None, ForcedSides::Both).is_err());
        assert!(ForcedBook::new(&[line(&["2g2f"])], None, 0.5, Some((0, 3)), ForcedSides::Both).is_err());
        assert!(ForcedBook::new(&[line(&["2g2f"])], None, 0.5, Some((3, 2)), ForcedSides::Both).is_err());
        assert!(ForcedBook::new(&[], None, 0.5, None, ForcedSides::Both).is_err());
    }

    #[test]
    fn 後手は同じ行動番号を点対称にミラーした手として読む() {
        let book = ForcedBook::new(&[line(&["7g7f", "2h6h", "5i4h"])], None, 1.0, None, ForcedSides::Both).unwrap();
        let [Some(s), Some(g)] = book.plans(1, 0, None) else { panic!("p=1 なら両者とも強制") };
        let read = |p: &ForcedPlan, c| p.line.iter().map(|&a| decode_usi(a, c).unwrap()).collect::<Vec<_>>();
        assert_eq!(read(&s, Color::Sente), line(&["7g7f", "2h6h", "5i4h"]));
        assert_eq!(read(&g, Color::Gote), line(&["3c3d", "8b4b", "5a6b"]));
    }

    #[test]
    fn 予定は乱数列と通し番号で決まり_pを変えても強制される側の中身は同じ() {
        let lines: Vec<Vec<String>> = vec![
            line(&["7g7f", "6g6f", "2h6h", "5i4h"]),
            line(&["5g5f", "2h5h", "5f5e", "5i4h"]),
            line(&["2g2f", "2f2e", "2e2d"]),
        ];
        let lo = ForcedBook::new(&lines, None, 0.3, Some((1, 4)), ForcedSides::Both).unwrap();
        let hi = ForcedBook::new(&lines, None, 0.8, Some((1, 4)), ForcedSides::Both).unwrap();
        let key = |p: &Option<ForcedPlan>| p.as_ref().map(|p| (p.line_id, p.limit));
        let (mut n_lo, mut n_hi, mut lines_seen, mut limits_seen) = (0, 0, [0; 3], [0; 5]);
        for k in 0..2000u64 {
            let a = lo.plans(7, k, None);
            let b = hi.plans(7, k, None);
            assert_eq!(a.clone().map(|p| key(&p)), lo.plans(7, k, None).map(|p| key(&p)), "同じ番号なら同じ予定");
            for c in 0..2 {
                if let Some(p) = &a[c] {
                    n_lo += 1;
                    assert_eq!(key(&b[c]), Some((p.line_id, p.limit)), "p を上げても強制される側は同じ");
                }
                if let Some(p) = &b[c] {
                    n_hi += 1;
                    lines_seen[p.line_id as usize] += 1;
                    limits_seen[p.limit as usize] += 1;
                    assert!(p.limit >= 1 && p.limit as usize <= p.line.len());
                }
            }
        }
        // 側ごとに独立に p（4000 側）。二項分布の ±5σ に入る
        assert!((1000..1400).contains(&n_lo), "{n_lo}");
        assert!((3000..3400).contains(&n_hi), "{n_hi}");
        assert!(lines_seen.iter().all(|&n| n > 900), "{lines_seen:?}");
        // 3手のラインは 4 で頭打ちになるので limit 4 は残りの2本のぶんだけ
        assert!(limits_seen[1] > 700 && limits_seen[3] > 700 && limits_seen[4] > 300, "{limits_seen:?}");
        assert_ne!(
            lo.plans(7, 0, None).map(|p| key(&p)),
            lo.plans(8, 0, None).map(|p| key(&p)),
            "乱数列を変えると別の予定（この組はたまたま同じにならない）"
        );
    }

    #[test]
    fn 側の指定() {
        let lines = vec![line(&["7g7f"])];
        let only = |sides, learner| {
            let book = ForcedBook::new(&lines, None, 1.0, None, sides).unwrap();
            book.plans(3, 0, learner).map(|p| p.is_some())
        };
        assert_eq!(only(ForcedSides::Both, None), [true, true]);
        assert_eq!(only(ForcedSides::Sente, None), [true, false]);
        assert_eq!(only(ForcedSides::Gote, None), [false, true]);
        assert_eq!(only(ForcedSides::Learner, Some(Color::Gote)), [false, true]);
        assert_eq!(only(ForcedSides::Opponent, Some(Color::Gote)), [true, false]);
        assert_eq!(only(ForcedSides::Learner, None), [false, false]);
        let none = ForcedBook::new(&lines, None, 0.0, None, ForcedSides::Both).unwrap();
        assert!(none.plans(3, 0, None).iter().all(Option::is_none));
    }

    /// 中の戦略: 呼ばれた回数を数え、決めた手を順に返す
    struct Scripted {
        moves: Vec<&'static str>,
        calls: std::rc::Rc<std::cell::Cell<u32>>,
    }
    impl crate::strategy::Strategy for Scripted {
        fn choose(
            &mut self,
            _view: &crate::protocol::PlayerView,
            _log: &crate::observation::ObservationLog,
            _foul_tried: &std::collections::HashSet<String>,
        ) -> Option<String> {
            let i = self.calls.get();
            self.calls.set(i + 1);
            self.moves.get(i as usize).map(|s| s.to_string())
        }
        fn name(&self) -> &'static str {
            "scripted"
        }
    }

    #[test]
    fn 包みは接触までラインを指し_以後は中の戦略に任せる() {
        use crate::referee::Referee;
        use crate::strategy::Strategy;
        // 先手は居飛車速攻（2四歩まで強制）、後手は強制なしで 8四歩→8五歩→2四同歩（接触）
        let book = ForcedBook::new(&[line(&["2g2f", "2f2e", "2e2d", "2d2c+"])], None, 1.0, None, ForcedSides::Sente).unwrap();
        let [Some(plan), None] = book.plans(0, 0, None) else { panic!("先手だけ強制") };
        let s_calls = std::rc::Rc::new(std::cell::Cell::new(0));
        let g_calls = std::rc::Rc::new(std::cell::Cell::new(0));
        let mut sente = ForcedOpening::new(Box::new(Scripted { moves: vec!["5i4h"], calls: s_calls.clone() }), plan);
        let mut gote = Scripted { moves: vec!["8c8d", "8d8e", "2c2d"], calls: g_calls.clone() };
        let mut referee = Referee::new();
        let mut played = vec![];
        for _ in 0..7 {
            let side = referee.to_move();
            let view = referee.view(side, [0, 0], 0);
            let mover: &mut dyn Strategy = if side == Color::Sente { &mut sente } else { &mut gote };
            let usi = mover.choose(&view, referee.log(side), referee.foul_tried(side)).unwrap();
            referee.step(&usi, 0);
            played.push(usi);
        }
        assert_eq!(played, line(&["2g2f", "8c8d", "2f2e", "8d8e", "2e2d", "2c2d", "5i4h"]));
        assert_eq!(s_calls.get(), 1, "先手の中の戦略は接触の後の1回だけ呼ばれる");
        assert_eq!(g_calls.get(), 3);
        assert_eq!(sente.name(), "scripted");
    }

    #[test]
    fn 定跡ファイルから系統ごとに同じ確率で引く() {
        let json = r#"{"lines":[
            {"moves":["7g7f"],"category":"a"},
            {"moves":["2g2f"],"category":"a"},
            {"moves":["5g5f"],"category":"b"}]}"#;
        let book = book_from_json(json, 1.0, None, ForcedSides::Both).unwrap();
        let mut n = [0u32; 3];
        for k in 0..4000u64 {
            for p in book.plans(5, k, None).into_iter().flatten() {
                n[p.line_id as usize] += 1;
            }
        }
        // 8000 側のうち b は約半分、a の2本は約1/4ずつ
        assert!((3700..4300).contains(&n[2]), "{n:?}");
        assert!((1800..2200).contains(&n[0]) && (1800..2200).contains(&n[1]), "{n:?}");
        assert!(book_from_json(include_str!("../../rl_openings.json"), 0.5, None, ForcedSides::Both).is_ok());
    }

    #[test]
    fn 本番botの予定は対局のidで決まる() {
        let json = include_str!("../../rl_openings.json");
        let book = book_from_json(json, 1.0, None, ForcedSides::Both).unwrap();
        let key = |id: &str, c| book.plan_for_game(id, c).map(|p| (p.line_id, p.limit));
        assert!(key("game-abc", Color::Sente).is_some(), "p=1 なら必ず強制");
        assert_eq!(key("game-abc", Color::Sente), key("game-abc", Color::Sente), "同じ対局IDなら同じ予定");
        let ids: std::collections::HashSet<u32> =
            (0..200).filter_map(|i| key(&format!("g{i}"), Color::Gote)).map(|(id, _)| id).collect();
        assert!(ids.len() > 5, "対局IDが違えばラインが散る: {ids:?}");
        let off = book_from_json(json, 0.0, None, ForcedSides::Both).unwrap();
        assert!(off.plan_for_game("game-abc", Color::Sente).is_none());
        // 予定が無ければ包まない（名前も中の戦略のまま）
        let inner = crate::strategy::make("heuristic").unwrap();
        assert_eq!(wrap_for_game(inner, Some(&off), "game-abc", Color::Sente).name(), "heuristic");
    }

    #[test]
    fn 定跡を固定した相手は先後どちらでもそのラインから指す() {
        use crate::referee::Referee;
        let lines = book_lines("rl_openings.json", include_str!("../../rl_openings.json"));
        let k = lines.len() - 1;
        let first = quiet_prefix(&lines[k]).unwrap()[0].clone();
        let mirrored = decode_usi(encode_usi(&first, Color::Sente).unwrap(), Color::Gote).unwrap();
        let name = format!("open:{k}:heuristic");
        for seed in [Some(1), Some(2), None] {
            let make = || match seed {
                Some(s) => crate::strategy::make_seeded(&name, s),
                None => crate::strategy::make(&name),
            };
            let mut referee = Referee::new();
            let view = referee.view(Color::Sente, [0, 0], 0);
            let sente = make().unwrap().choose(&view, referee.log(Color::Sente), referee.foul_tried(Color::Sente));
            assert_eq!(sente.as_deref(), Some(first.as_str()), "先手は seed={seed:?} でもライン {k} の初手");
            // 後手は同じラインを点対称に読む（先手の手は相手の駒に触れない端歩）
            referee.step("9g9f", 0);
            let view = referee.view(Color::Gote, [0, 0], 0);
            let gote = make().unwrap().choose(&view, referee.log(Color::Gote), referee.foul_tried(Color::Gote));
            assert_eq!(gote.as_deref(), Some(mirrored.as_str()));
        }
        assert_eq!(crate::strategy::make(&name).unwrap().name(), "heuristic", "記録上の名前は中の戦略のまま");
        for bad in [format!("open:{}:heuristic", lines.len()), "open:x:heuristic".into(), "open:0:no_such".into(), "open:0".into()] {
            assert!(crate::strategy::make(&bad).is_none(), "{bad} は作れない");
        }
    }

    #[test]
    fn ブックから引く相手は局ごとにラインが変わり同じseedなら同じライン() {
        use crate::referee::Referee;
        let lines = book_lines("rl_openings.json", include_str!("../../rl_openings.json"));
        let firsts: std::collections::HashSet<String> =
            lines.iter().map(|l| quiet_prefix(l).unwrap()[0].clone()).collect();
        let first_move = |seed: Option<u64>| {
            let referee = Referee::new();
            let view = referee.view(Color::Sente, [0, 0], 0);
            let mut s = match seed {
                Some(x) => crate::strategy::make_seeded("open:book:heuristic", x),
                None => crate::strategy::make("open:book:heuristic"),
            }
            .unwrap();
            s.choose(&view, referee.log(Color::Sente), referee.foul_tried(Color::Sente)).unwrap()
        };
        let seen: std::collections::HashSet<String> = (0..200).map(|s| first_move(Some(s))).collect();
        assert!(seen.is_subset(&firsts), "初手はどれかのラインの初手");
        assert!(seen.len() >= 4, "局（seed）ごとにラインが変わる（初手 {} 通り）", seen.len());
        for s in [3, 17, 99] {
            assert_eq!(first_move(Some(s)), first_move(Some(s)), "同じ seed なら同じライン");
        }
        assert!(firsts.contains(&first_move(None)), "seed なしでも作れる");
        assert_eq!(crate::strategy::make("open:book:heuristic").unwrap().name(), "heuristic");
        assert!(crate::strategy::make("open:book:no_such").is_none());
    }

    fn book_lines(name: &str, json: &str) -> Vec<Vec<String>> {
        let v: serde_json::Value = serde_json::from_str(json).unwrap_or_else(|e| panic!("{name} は JSON: {e}"));
        v["lines"]
            .as_array()
            .unwrap_or_else(|| panic!("{name} に lines が無い"))
            .iter()
            .map(|l| l["moves"].as_array().unwrap().iter().map(|m| m.as_str().unwrap().to_string()).collect())
            .collect()
    }

    #[test]
    fn 定跡ファイルの全ラインが通る() {
        let lines = book_lines("joseki.json", include_str!("../../joseki.json"));
        assert_eq!(lines.len(), 34);
        let book = ForcedBook::new(&lines, None, 1.0, None, ForcedSides::Both).unwrap();
        let lens = book.line_lengths();
        // 居飛車速攻（ライン 1）は 2三歩成の手前の3手
        assert_eq!(lens[1], 3);
        assert!(lens.iter().all(|&n| (1..=16).contains(&n)), "{lens:?}");
    }

    /// 学習用のブック（`rl_openings.json`。`train_rnad.py --open-book` に渡す）。joseki.json と違って
    /// 凍結版の pin ではないので自由に足してよいが、足したラインが env で拒まれないことはここで守る
    #[test]
    fn 学習用の定跡ファイルの全ラインが通る() {
        let json = include_str!("../../rl_openings.json");
        let lines = book_lines("rl_openings.json", json);
        assert!(!lines.is_empty());
        let book = ForcedBook::new(&lines, None, 1.0, None, ForcedSides::Both).unwrap();
        let lens = book.line_lengths();
        assert!(lens.iter().all(|&n| n >= 1), "接触の手前で 0 手になるライン: {lens:?}");
        // 系統ごとの重み（train_rnad.py の既定 --open-weight category）に使うので、系統と名前は必ず書く
        let v: serde_json::Value = serde_json::from_str(json).unwrap();
        for (i, l) in v["lines"].as_array().unwrap().iter().enumerate() {
            for key in ["name", "category"] {
                assert!(l[key].as_str().is_some_and(|s| !s.is_empty()), "ライン {i} に {key} が無い");
            }
            if let Some(w) = l.get("weight") {
                assert!(w.as_f64().is_some_and(|w| w.is_finite() && w >= 0.0), "ライン {i} の weight: {w}");
            }
        }
    }
}
