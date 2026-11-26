//! **scratch6 系列の凍結版 v2**（2026-11-26）: 方策ネット（DeepNash 路線の R-NaD）。エンコーダ v6（142枚）でゼロから育てた系列の 50,000更新（rl_scratch6_v1 の続きで run の最後。rl_scratch6_v1 に 60.8%、同じ更新数の rl_scratch5_v2 に 56.2%、rl_mmdhandex_v2 に 81.2%〔各 3,700局〕）
//!
//! `scripts/freeze_rl.py` が生成した。**編集しない**（改善は `src/rl/` で行う）。
//! 推論の一式（行動の符号化・観測のテンソル化・手書きの推論）は凍結時点の固定コピーで、
//! 重みは `models/rl_scratch6_v2.bin` をバイナリに埋め込む（sha256 `4fac42f8f0664082494cb15a0ae2ca9a4d3d7bc5852a3a474e9e2bc4a1897a04`）。
//! **序盤の定跡を内蔵する**: 毎局 `models/rl_scratch6_v2.openings.json`（凍結時点の `rl_openings.json` の固定コピー、
//! sha256 `0c84ece8ba03504ee93afe71989bec2b511ab17221ec5f25d7612a084e3b0325`）から系統ごとに同じ確率で1本引き、接触・王手・自分の反則まで
//! そのラインを指す（`open:book:rl_policy:<重み>` と同じ seed なら同じライン・同じ手）。序盤を 100% 強制して
//! 学習したネットは初期局面から自分で指したことが無いため。強制の仕組みも固定コピー（`forced`）。
//! 探索なしで1手 約10ms。思考予算・実行時 env は持たない。

#![allow(clippy::all)]

use std::collections::HashSet;
use std::sync::OnceLock;

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::observation::ObservationLog;
use crate::protocol::PlayerView;
use crate::strategy::Strategy;

/// 埋め込んだ重み
const WEIGHTS: &[u8] = include_bytes!("../../models/rl_scratch6_v2.bin");
/// 埋め込んだ重みの sha256（凍結時点の値。テストで検査する）
pub const WEIGHTS_SHA256: &str = "4fac42f8f0664082494cb15a0ae2ca9a4d3d7bc5852a3a474e9e2bc4a1897a04";

fn net() -> &'static policy_net::PolicyNet {
    static NET: OnceLock<policy_net::PolicyNet> = OnceLock::new();
    NET.get_or_init(|| policy_net::PolicyNet::from_bytes(WEIGHTS).expect("凍結版の重みが読めない"))
}

/// scratch6 系列の凍結版 v2 の方策（序盤の強制の中身。softmax からサンプリングする）
pub struct RlScratch6V2Net {
    rng: StdRng,
    last: Option<serde_json::Value>,
}

impl RlScratch6V2Net {
    pub fn new() -> Self {
        Self { rng: StdRng::from_rng(&mut rand::rng()), last: None }
    }

    pub fn with_seed(seed: u64) -> Self {
        Self { rng: StdRng::seed_from_u64(seed), last: None }
    }
}

impl Default for RlScratch6V2Net {
    fn default() -> Self {
        Self::new()
    }
}

impl Strategy for RlScratch6V2Net {
    fn choose(
        &mut self,
        view: &PlayerView,
        log: &ObservationLog,
        foul_tried: &HashSet<String>,
    ) -> Option<String> {
        // src/rl/policy.rs の rl_policy（サンプリング）と同じ手順
        let mask = action::legal_mask(view, log, foul_tried);
        let legal: Vec<usize> = (0..mask.len()).filter(|&a| mask[a]).collect();
        if legal.is_empty() {
            return None;
        }
        let (logits, value) = net().forward(&encode::encode_planes(net().planes(), view, log, foul_tried));
        let max = legal.iter().map(|&a| logits[a]).fold(f32::MIN, f32::max);
        let weights: Vec<f64> = legal.iter().map(|&a| f64::from(logits[a] - max).exp()).collect();
        let total: f64 = weights.iter().sum();
        let mut r = self.rng.random::<f64>() * total;
        let mut pick = *legal.last().unwrap();
        for (&a, w) in legal.iter().zip(&weights) {
            if r < *w {
                pick = a;
                break;
            }
            r -= w;
        }
        let prob = weights[legal.iter().position(|&a| a == pick).unwrap()] / total;
        self.last = Some(serde_json::json!({ "p": prob, "value": value }));
        action::decode_usi(pick, view.your_color)
    }

    fn name(&self) -> &'static str {
        "rl_scratch6_v2"
    }

    fn debug_state(&self) -> Option<serde_json::Value> {
        self.last.clone()
    }
}

/// 埋め込んだ定跡（凍結時点の `rl_openings.json` の固定コピー）
const OPENINGS: &str = include_str!("../../models/rl_scratch6_v2.openings.json");
/// 埋め込んだ定跡の sha256（凍結時点の値。テストで検査する）
pub const OPENINGS_SHA256: &str = "0c84ece8ba03504ee93afe71989bec2b511ab17221ec5f25d7612a084e3b0325";

fn book() -> &'static forced::ForcedBook {
    static BOOK: OnceLock<forced::ForcedBook> = OnceLock::new();
    BOOK.get_or_init(|| {
        forced::book_from_json(OPENINGS, 1.0, None, forced::ForcedSides::Both).expect("凍結版の定跡が読めない")
    })
}

/// scratch6 系列の凍結版 v2 の戦略（`strategy::make("rl_scratch6_v2")`）。序盤は内蔵の定跡から1本引いて指し、以後は方策に任せる
pub struct RlScratch6V2(forced::ForcedOpening<RlScratch6V2Net>);

impl RlScratch6V2 {
    pub fn new() -> Self {
        Self::with_seed(rand::rng().random())
    }

    /// ラインも方策の乱数も `seed` から決まる（`open:book:rl_policy:<重み>` に同じ seed を渡したときと同じ）
    pub fn with_seed(seed: u64) -> Self {
        let plan = forced::book_plan(book(), seed).expect("p = 1 のブックは必ず予定が付く");
        Self(forced::ForcedOpening::new(Box::new(RlScratch6V2Net::with_seed(seed)), plan))
    }
}

impl Default for RlScratch6V2 {
    fn default() -> Self {
        Self::new()
    }
}

impl Strategy for RlScratch6V2 {
    fn choose(
        &mut self,
        view: &PlayerView,
        log: &ObservationLog,
        foul_tried: &HashSet<String>,
    ) -> Option<String> {
        self.0.choose(view, log, foul_tried)
    }

    fn name(&self) -> &'static str {
        "rl_scratch6_v2"
    }

    fn debug_state(&self) -> Option<serde_json::Value> {
        self.0.debug_state()
    }
}

/// 序盤の強制（`src/rl/forced.rs` のブック・予定・包みの凍結時点の固定コピー）
pub mod forced {
    // **序盤の強制**（学習の自己対局で序盤を散らす。docs/rl-opening-diversity.md）。
    //
    // 方策ネットの序盤は約40,000更新から飛車先の歩の速攻1本に固まり、ほかの手の確率は 1e-18
    // まで落ちている（指されないので学ぶ機会もない）。そこで局ごと・**先後それぞれ独立に**確率 p で、
    // その側の最初の数手を定跡のラインから指させる。強制した手は env の中で指す（`EnvGame::settle`）
    // ので学習側の行には現れず、学習器から見ると「環境の手」（チャンス手）になる。
    //
    // - **独立に引く**のが要点: 強制されなかった側は、相手が強制されたかどうかを自分の観測から
    //   推せない（駒がぶつかるまで相手の手は観測に入らない）。両者をまとめて強制すると、
    //   「自分の序盤が定跡なら相手も定跡」という相関が漏れる
    // - **抜ける条件は `crate::opening::OpeningBook::next` と同じ**: どちらかの駒取り・王手
    //   （= 両者に見える接触）、自分の反則、ラインを指し終えた。ただし定跡の最後の**駒取りの手は
    //   強制しない**（初期配置の相手の駒に触れる手の手前で切る）。ぶつかった後の判断は方策が指す
    // - 定跡の手が実際には反則（見えない相手の駒に遮られた等）なら、**そのまま反則として指して**
    //   から抜ける（`opening.rs` と同じ）。真実の盤面で合法かを先に調べて黙って抜けると、
    //   方策が指し始める局面が見えない情報と相関する
    // - `opening.rs` と `joseki.json` は凍結版の共有 pin（`frozen::SHARED_MODEL_PINS`）なので
    //   触らない。ラインは呼び出し側（Python の学習）が読んで渡す

    use std::sync::Arc;

    use crate::board::{Coord, Promotion, make_usi_square, move_targets, parse_usi_square, promotion_choice};
    use crate::protocol::Color;
    use super::action::{decode_usi, encode_usi};
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

    /// ブックから **seed で1本引いた予定**（`open:book:<戦略名>` と、定跡を内蔵した凍結版が使う）。
    /// p = 1・両側が対象のブックなら必ず予定が付く。ラインは先手視点の並びで、後手は点対称に読む
    pub fn book_plan(book: &ForcedBook, seed: u64) -> Option<ForcedPlan> {
        let [plan, _] = book.plans(mix(seed ^ OPENING_SALT), 0, None);
        plan
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
                .filter(|&a| super::action::legal_mask(view, log, foul_tried).get(a).copied().unwrap_or(false))
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

    // 乱数の混ぜ方（`selfplay::mix` の凍結時点の固定コピー）
    fn mix(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// 行動の符号化とマスク（`src/rl/action.rs` の凍結時点の固定コピー）
pub mod action {
    // 行動の符号化（AlphaZero 将棋と同じ 139 種 × 81 マス）と行動マスク。
    //
    // 手番側を常に先手向きに正規化する（後手なら盤を180度回す）ので、
    // 「前」は常に段が減る方向になる。
    //
    // 行動番号 = 種類 × 81 + マス（方策ヘッドの出力 [139, 9, 9] を平らにした並び）。
    // - 種類 0..64: 8方向 × 距離 1〜8（不成）。マス = 移動元
    // - 種類 64, 65: 桂の2方向（不成）。マス = 移動元
    // - 種類 66..132: 上の 66 種の成り
    // - 種類 132..139: 打ち（`HAND_ROLES` 順）。マス = 打ち先
    //
    // マスクは**自駒だけを見た候補手**（`board.rs`）から、その手番で既に反則した手を除き、
    // さらに**観測から反則が確定する手**（`Deduction`）を除いたもの。
    // 見えている範囲で確定する反則（自駒のマス・自駒の飛び越え・二歩・行き所なし）と、
    // 観測から論理的に確定する反則（王手を解消し得ない手など）は落ち、
    // 見えない相手駒による反則のうち確定しないものは残る。**真の合法手はすべてマスクに含まれる**
    // （テストで常時検査。漏れた手は永久に指せなくなる）。
    //
    // 観測からの除外（`MASK_VERSION` 2、2026-10-02〜）を持たない旧マスクは `basic_legal_mask`。
    // 凍結版 rl_v15〜rl_v22 はこちらで学習・凍結されている。

    use std::collections::HashSet;

    use crate::board::{
        Coord, Promotion, drop_targets, make_usi_drop, make_usi_move, move_targets, orient,
        parse_usi_square, promotion_choice, rays, steps,
    };
    use crate::observation::{Observation, ObservationLog};
    use crate::protocol::{Color, PlayerView, Role};
    use crate::shogi::{
        HAND_ROLE_TOTAL, HAND_ROLES, ShogiMove, hand_index, parse_usi, promote_role, unpromote_role,
    };

    /// マスクの版。1 = 自駒視点だけ、2 = 観測からの確定反則も除く、
    /// 3 = 開き王手の仮説を自駒の遮りで絞る、4 = 相手に残り得る駒種とその動きも見る。
    /// 5 = 直前に王手で逃げた升への戻りを、王手が残ると確認できる場合に除く。
    /// 6 = 打ち歩詰めを否定できる歩打ちの反則からも占有を推定する。
    pub const MASK_VERSION: u32 = 6;

    pub const NUM_SQUARES: usize = 81;
    /// 方向×距離 64 ＋ 桂 2
    const MOVE_KINDS: usize = 66;
    const DROP_BASE: usize = MOVE_KINDS * 2;
    pub const NUM_ACTION_KINDS: usize = DROP_BASE + HAND_ROLES.len();
    pub const NUM_ACTIONS: usize = NUM_ACTION_KINDS * NUM_SQUARES;

    /// 正規化後の8方向 (筋の差, 段の差)。段が減るのが「前」
    const DIRS: [(i8, i8); 8] = [
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
    ];
    const KNIGHT_DIRS: [(i8, i8); 2] = [(1, -2), (-1, -2)];

    /// 手番側から見た座標へ（後手なら180度回す。自分自身が逆変換）
    pub fn normalize(c: Coord, color: Color) -> Coord {
        match color {
            Color::Sente => c,
            Color::Gote => Coord {
                file: 10 - c.file,
                rank: 10 - c.rank,
            },
        }
    }

    /// 正規化済み座標 → マス番号（段優先: (段-1)×9 + (筋-1)）
    pub fn square_index(c: Coord) -> usize {
        (c.rank as usize - 1) * 9 + (c.file as usize - 1)
    }

    pub fn index_square(i: usize) -> Coord {
        Coord {
            file: (i % 9) as i8 + 1,
            rank: (i / 9) as i8 + 1,
        }
    }

    /// 移動の差分 → 種類（成りを含まない 0..66）
    fn move_kind(df: i8, dr: i8) -> Option<usize> {
        if let Some(k) = KNIGHT_DIRS.iter().position(|&d| d == (df, dr)) {
            return Some(64 + k);
        }
        let dist = df.abs().max(dr.abs());
        if dist == 0 || dist > 8 {
            return None;
        }
        // 直線（縦・横・斜め）でなければ符号化できない
        if df != 0 && dr != 0 && df.abs() != dr.abs() {
            return None;
        }
        let unit = (df.signum(), dr.signum());
        let d = DIRS.iter().position(|&x| x == unit)?;
        Some(d * 8 + (dist as usize - 1))
    }

    /// 指し手 → 行動番号。盤外・符号化できない差分は None
    pub fn encode_move(mv: &ShogiMove, color: Color) -> Option<usize> {
        match *mv {
            ShogiMove::Board { from, to, promote } => {
                let (f, t) = (normalize(from, color), normalize(to, color));
                let kind = move_kind(t.file - f.file, t.rank - f.rank)?;
                let kind = if promote { kind + MOVE_KINDS } else { kind };
                Some(kind * NUM_SQUARES + square_index(f))
            }
            ShogiMove::Drop { role, to } => {
                let kind = DROP_BASE + hand_index(role)?;
                Some(kind * NUM_SQUARES + square_index(normalize(to, color)))
            }
        }
    }

    pub fn encode_usi(usi: &str, color: Color) -> Option<usize> {
        encode_move(&parse_usi(usi)?, color)
    }

    /// 行動番号 → 指し手。盤外へ出る番号は None（マスクの外なので通常は起きない）
    pub fn decode_action(action: usize, color: Color) -> Option<ShogiMove> {
        if action >= NUM_ACTIONS {
            return None;
        }
        let (kind, sq) = (action / NUM_SQUARES, action % NUM_SQUARES);
        let at = index_square(sq);
        if kind >= DROP_BASE {
            return Some(ShogiMove::Drop {
                role: HAND_ROLES[kind - DROP_BASE],
                to: normalize(at, color),
            });
        }
        let promote = kind >= MOVE_KINDS;
        let base = kind % MOVE_KINDS;
        let (df, dr) = if base >= 64 {
            KNIGHT_DIRS[base - 64]
        } else {
            let (ux, uy) = DIRS[base / 8];
            let dist = (base % 8) as i8 + 1;
            (ux * dist, uy * dist)
        };
        let to = Coord {
            file: at.file + df,
            rank: at.rank + dr,
        };
        if !(1..=9).contains(&to.file) || !(1..=9).contains(&to.rank) {
            return None;
        }
        Some(ShogiMove::Board {
            from: normalize(at, color),
            to: normalize(to, color),
            promote,
        })
    }

    pub fn decode_usi(action: usize, color: Color) -> Option<String> {
        Some(decode_action(action, color)?.to_usi())
    }

    /// 自駒だけを見た候補手（USI）。成りが任意なら成・不成の両方、強制なら成りだけ
    pub fn candidate_usis(view: &PlayerView) -> Vec<String> {
        let color = view.your_color;
        let mut out = vec![];
        for piece in &view.your_pieces {
            let Some(from) = parse_usi_square(&piece.square) else {
                continue;
            };
            for to in move_targets(&view.your_pieces, piece, color) {
                match promotion_choice(piece.role, from, to, color) {
                    Promotion::None => out.push(make_usi_move(from, to, false)),
                    Promotion::Optional => {
                        out.push(make_usi_move(from, to, false));
                        out.push(make_usi_move(from, to, true));
                    }
                    Promotion::Forced => out.push(make_usi_move(from, to, true)),
                }
            }
        }
        for role in HAND_ROLES {
            if view.your_hand.get(&role).copied().unwrap_or(0) == 0 {
                continue;
            }
            for to in drop_targets(&view.your_pieces, role, color) {
                out.extend(make_usi_drop(role, to));
            }
        }
        out
    }

    /// 旧マスク（`MASK_VERSION` 1）: 自駒視点の候補手から、その手番で反則済みの手だけを除く
    pub fn basic_legal_mask(view: &PlayerView, foul_tried: &HashSet<String>) -> Vec<bool> {
        let mut mask = vec![false; NUM_ACTIONS];
        for usi in candidate_usis(view) {
            if foul_tried.contains(&usi) {
                continue;
            }
            if let Some(a) = encode_usi(&usi, view.your_color) {
                mask[a] = true;
            }
        }
        mask
    }

    /// 行動マスク（長さ `NUM_ACTIONS`）。反則済みの手と、観測から反則が確定する手（`Deduction`）を除く。
    ///
    /// 除外で候補が尽きたときは旧マスクへ戻す（推論が健全なら、真の局面に合法手がある限り
    /// 尽きることはない。尽きる＝推論の前提が崩れているので、指せなくなるより安全側に倒す）
    pub fn legal_mask(view: &PlayerView, log: &ObservationLog, foul_tried: &HashSet<String>) -> Vec<bool> {
        legal_mask_with_version(view, log, foul_tried, MASK_VERSION)
    }

    pub(super) fn legal_mask_with_version(
        view: &PlayerView,
        log: &ObservationLog,
        foul_tried: &HashSet<String>,
        mask_version: u32,
    ) -> Vec<bool> {
        let candidates = candidate_usis(view);
        let deduction = Deduction::new(view, log, foul_tried, &candidates.iter().cloned().collect(), mask_version);
        let mut mask = vec![false; NUM_ACTIONS];
        let mut basic = vec![false; NUM_ACTIONS];
        let mut any = false;
        for usi in candidates {
            if foul_tried.contains(&usi) {
                continue;
            }
            let Some(mv) = parse_usi(&usi) else { continue };
            let Some(a) = encode_move(&mv, view.your_color) else {
                continue;
            };
            basic[a] = true;
            if !deduction.rules_out(&usi, &mv) {
                mask[a] = true;
                any = true;
            }
        }
        if any { mask } else { basic }
    }

    /// マスクの推論（`Deduction`）が確定させた事実のうち、盤のマスで表せるもの（エンコーダ v2 の入力）。
    /// どれも真の盤面と矛盾しない（マスクが真の合法手を落とさないのと同じ根拠）
    #[derive(Debug, Default, Clone, PartialEq)]
    pub struct DeducedFacts {
        /// 相手の駒がいると確定したマス
        pub occupied: Vec<Coord>,
        /// 飛び越え反則ごとの間のマス（それぞれ少なくとも1マスに相手の駒がいる）
        pub blocked: Vec<Vec<Coord>>,
        /// 王手中: 王手駒がいうるマス（重複なし。王手中でない・前提が崩れているときは空）
        pub checkers: Vec<Coord>,
    }

    pub fn deduced_facts(view: &PlayerView, log: &ObservationLog, foul_tried: &HashSet<String>) -> DeducedFacts {
        let candidates: HashSet<String> = candidate_usis(view).into_iter().collect();
        // 学習済み方策への観測入力はマスクv2の時点の推論で固定する。
        let d = Deduction::new(view, log, foul_tried, &candidates, 2);
        let mut occupied: Vec<Coord> = d.occupied.into_iter().collect();
        occupied.sort_by_key(|c| (c.file, c.rank));
        let blocked = d
            .blocked_beyond
            .iter()
            .map(|&(f, unit, dist)| (1..dist).map(|k| step(f, unit, k)).collect())
            .collect();
        let mut checkers: Vec<Coord> = d
            .check
            .map(|(_, hyps)| hyps.into_iter().map(|h| h.square).collect())
            .unwrap_or_default();
        checkers.sort_by_key(|c| (c.file, c.rank));
        checkers.dedup();
        DeducedFacts {
            occupied,
            blocked,
            checkers,
        }
    }

    /// 王手駒がいうるマスの1仮説。`dir` は玉から見た方向（桂は None）、`dist` は玉からの距離
    #[derive(Debug, Clone, Copy)]
    struct CheckerHyp {
        square: Coord,
        dir: Option<(i8, i8)>,
        dist: i8,
    }

    /// 観測から論理的に確定する反則。どれも**真の合法手を落とさない**ことだけを条件にしている
    /// （「たぶん反則」は落とさない。それはネットの領分）。
    ///
    /// 1. **反則した手の成り／不成の片割れ**: 両者の違いは移動後の駒種だけで、自玉が取られるか
    ///    （= 合法か）は盤上の占有にしか依らない。行き所の有無は候補生成が自駒視点で落とし済み
    /// 2. **相手の駒がいると確定したマス**（`occupied`）への打ちと、そこを飛び越える移動。確定の根拠は
    ///    - 直前に相手が駒を取ったマス（相手の着手駒がいまそこにいる）
    ///    - 王手中でないときの**打ちの反則**: 打ちで自玉が危なくなることは無く、二歩・
    ///      行き所は候補生成が落とし済み。歩は打ち歩詰めを否定できるときだけ占有を推定する
    ///    - 王手中でないときの**間が1マスの飛び越え反則**（下の 3 で原因が遮りと確定したもの）:
    ///      遮っている駒は間の1マスにしかいられない
    /// 3. **飛び越え反則の先**（`blocked_beyond`）: 王手中でないときに2マス以上の直線移動 F→T が
    ///    反則で、F の駒がピンされえない（玉の筋で最初の自駒でない、または移動方向がその筋と平行）
    ///    なら、原因は「F と T の間に相手の駒がいる」しかない（T に相手の駒がいれば駒取りで合法）。
    ///    同じ F から同じ方向へ T より先へ進む手も必ず遮られる
    /// 4. **王手中に王手を解消し得ない手**: 王手駒がいうるマスの集合 H を作り、H のどの仮説に
    ///    対しても解消しない手を落とす。H は
    ///    - 相手の直前の手が駒を取っていなければ: 玉から8方向に最初の自駒の手前までのマスと、
    ///      桂の王手マス（自駒がいないもの）。王手駒はこのどこかにいる
    ///    - 駒を取っていれば（取られたマス X）: 直前の自分の手は合法だった＝相手の手の前は王手が
    ///      無かったので、王手駒は「X に来た着手駒」か「着手駒の元のマス O が空いて通った
    ///      飛び駒（開き王手）」しかない。O は玉の筋の上の空きマスで、その先に飛び駒が入る余地が
    ///      あり、X へ1手で行ける位置に限られる。マスクv3から自駒の遮り、v4から相手に残り得る
    ///      駒種ごとの動きも調べる。開き王手の飛び駒は O の先の筋の上のマス
    ///
    ///    仮説 h に対して手が王手を解消し得ない条件:
    ///    - 玉以外の手: 着地点が h（駒取り。打ちは除く）でも、玉と h の間（合駒）でもない
    ///    - 玉の手: h が2マス以上離れた筋の上（＝その方向へ利く飛び駒）で、移動先が同じ筋の上
    ///      （h の側へ1マス、または玉の真後ろ）。玉が動くと元のマスが空くので真後ろも利きの中。
    ///      h が隣接・桂のときは駒種が分からないので玉の手は落とさない
    struct Deduction {
        twins: HashSet<String>,
        /// 相手の駒がいると確定したマス（直前に取られたマスを含む）
        occupied: HashSet<Coord>,
        /// 飛び越え反則した (元のマス, 方向, 距離)。同じ元・方向でこれより遠い手は遮られる
        blocked_beyond: Vec<(Coord, (i8, i8), i8)>,
        check: Option<(Coord, Vec<CheckerHyp>)>,
        king_return: Option<Coord>,
    }

    const ALL_DIRS: [(i8, i8); 8] = [
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
    ];

    fn on_board(c: Coord) -> bool {
        (1..=9).contains(&c.file) && (1..=9).contains(&c.rank)
    }

    fn step(c: Coord, (df, dr): (i8, i8), k: i8) -> Coord {
        Coord {
            file: c.file + df * k,
            rank: c.rank + dr * k,
        }
    }

    /// from→to が直線の移動で、途中（両端を除く）に square を通るか
    fn passes_through(from: Coord, to: Coord, square: Coord) -> bool {
        let (df, dr) = (to.file - from.file, to.rank - from.rank);
        if (df == 0 && dr == 0) || (df != 0 && dr != 0 && df.abs() != dr.abs()) {
            return false;
        }
        let unit = (df.signum(), dr.signum());
        let n = df.abs().max(dr.abs());
        (1..n).any(|k| step(from, unit, k) == square)
    }

    /// 何かの駒が o から x へ1手で行けうるか（直線か桂跳び。駒種と遮りは見ない＝上界）
    fn reachable(o: Coord, x: Coord) -> bool {
        let (df, dr) = (x.file - o.file, x.rank - o.rank);
        if df == 0 && dr == 0 {
            return false;
        }
        df == 0 || dr == 0 || df.abs() == dr.abs() || (df.abs() == 1 && dr.abs() == 2)
    }

    /// 自分の盤上駒と持駒で総数を使い切った駒種は、相手の着手駒になれない。
    /// それ以外は失われた駒や相手の持駒かもしれないので、成り・不成とも候補に残す。
    fn possible_opponent_roles(view: &PlayerView) -> Vec<Role> {
        let mut own = [0u32; HAND_ROLES.len()];
        for piece in &view.your_pieces {
            if let Some(i) = hand_index(unpromote_role(piece.role)) {
                own[i] += 1;
            }
        }
        let mut roles = vec![Role::King];
        for (i, role) in HAND_ROLES.into_iter().enumerate() {
            let hand = view.your_hand.get(&role).copied().unwrap_or(0);
            if own[i].saturating_add(hand) < HAND_ROLE_TOTAL[i] {
                roles.push(role);
                roles.extend(promote_role(role));
            }
        }
        roles
    }

    /// 着手前の駒種の利きで到達できるか。途中の遮りは呼び出し側で調べる。
    fn role_reaches(role: Role, color: Color, from: Coord, to: Coord) -> bool {
        let (df, dr) = (to.file - from.file, to.rank - from.rank);
        let distance = df.abs().max(dr.abs());
        steps(role).iter().any(|&d| orient(d, color) == (df, dr))
            || (distance > 0 && rays(role).iter().any(|&d| step(from, orient(d, color), distance) == to))
    }

    /// 打ち先が空だと仮定して、歩が王手できる升に相手玉がいないと証明する。
    /// 飛び駒の途中がある場合は、空と仮定した打ち先だけを通る利きに限る。
    fn pawn_drop_cannot_mate(view: &PlayerView, to: Coord) -> bool {
        let target = step(to, orient((0, -1), view.your_color), 1);
        view.your_pieces.iter().any(|piece| {
            let Some(from) = parse_usi_square(&piece.square) else { return false };
            let oriented = |d| orient(d, view.your_color);
            from == target
                || steps(piece.role).iter().any(|&d| step(from, oriented(d), 1) == target)
                || rays(piece.role).iter().any(|&d| {
                    let d = oriented(d);
                    let next = step(from, d, 1);
                    next == target || (next == to && step(from, d, 2) == target)
                })
        })
    }

    impl Deduction {
        /// `candidates` は自駒視点の候補手（`candidate_usis`）。反則から推論するのは候補手の反則だけ
        /// （候補外の反則は原因が成りの選択や行き所でありうるので、占有・遮りの根拠にしない）
        fn new(
            view: &PlayerView,
            log: &ObservationLog,
            foul_tried: &HashSet<String>,
            candidates: &HashSet<String>,
            mask_version: u32,
        ) -> Self {
            let twins = foul_tried
                .iter()
                .filter_map(|usi| match parse_usi(usi)? {
                    ShogiMove::Board { from, to, promote } => Some(make_usi_move(from, to, !promote)),
                    ShogiMove::Drop { .. } => None,
                })
                .collect();

            // 直前の相手の手（自分の最後の受理手より後にあるもの）が取ったマス。
            // ログの末尾の相手の手が本当に直前の手かを、手数で確かめる: 今の手数 `now` に対して、直前の相手の手に付く
            // 手数は適用後の規約（審判・サーバー）で `now`、適用前の規約（〜2026-11-24 の webhook のセッション）で
            // `now − 1`。1つ前の相手の手は `now − 2` / `now − 3` なので取り違えない。再接続で通知を取りこぼすと
            // （自分の手の受理と相手の応手の両方が欠けると）末尾の相手の手は2手前のもので、取ったマスの駒はもう動いたか、
            // 自分が取ったかもしれない。そのまま使うと占有と王手駒の候補を誤り、真の合法手を落とす（その形の決定点の
            // 7%。2026-11-23 の計測）。合わなければ「直前の相手の手は分からない」として扱う（王手駒の候補は広いほうへ戻る）
            let now = view.move_number;
            let mut known_opponent = None;
            let mut opponent_moved = false;
            for event in log.events().iter().rev() {
                match event {
                    Observation::OpponentMoved {
                        move_number,
                        captured_my_piece_at,
                    } => {
                        if *move_number == now || move_number.checked_add(1) == Some(now) {
                            opponent_moved = true;
                            known_opponent = captured_my_piece_at.as_deref().and_then(parse_usi_square);
                        }
                        break;
                    }
                    Observation::MyMove { .. } => break,
                    _ => {}
                }
            }

            // 動けないと確定している相手の歩（deduce::immobile_opponent_pawns）は使わない:
            // ログを毎回先頭から辿るのでマスクのコストが倍になる割に、記録483局で反則2件にしか効かない
            let mut occupied: HashSet<Coord> = known_opponent.into_iter().collect();

            let mut blocked_beyond = vec![];
            if !view.you_in_check {
                let king = view
                    .your_pieces
                    .iter()
                    .find(|p| p.role == Role::King)
                    .and_then(|p| parse_usi_square(&p.square));
                let own: HashSet<Coord> = view
                    .your_pieces
                    .iter()
                    .filter_map(|p| parse_usi_square(&p.square))
                    .collect();
                // f が玉の筋で最初の自駒なら、その筋の方向（＝ピンされうる向き）
                let pin_dir = |f: Coord| -> Option<(i8, i8)> {
                    let k = king?;
                    ALL_DIRS.iter().copied().find(|&d| {
                        let mut c = step(k, d, 1);
                        while on_board(c) && !own.contains(&c) {
                            c = step(c, d, 1);
                        }
                        c == f
                    })
                };
                for usi in foul_tried.iter().filter(|u| candidates.contains(*u)) {
                    match parse_usi(usi) {
                        // 歩（マスクv6〜）: `pawn_drop_cannot_mate` は「自駒が利いているマスに相手玉はいない」を使う。
                        // これは自分の手番で相手玉に王手がかかっていない（相手が自玉を王手に晒したまま指していない）
                        // ことが前提なので、`opponent_in_check` で確かめる。正規の対局の自分の手番では常に偽で、
                        // 効くのは手で組んだ局面・シナリオの開始局面のような、正規でない局面だけ
                        Some(ShogiMove::Drop { role, to })
                            if role != Role::Pawn
                                || (mask_version >= 6 && opponent_moved && !view.opponent_in_check
                                    && pawn_drop_cannot_mate(view, to)) =>
                        {
                            occupied.insert(to);
                        }
                        Some(ShogiMove::Board { from, to, .. }) if king.is_some() => {
                            let (df, dr) = (to.file - from.file, to.rank - from.rank);
                            let dist = df.abs().max(dr.abs());
                            if dist < 2 || (df != 0 && dr != 0 && df.abs() != dr.abs()) {
                                continue;
                            }
                            let unit = (df.signum(), dr.signum());
                            let pinnable = pin_dir(from)
                                .is_some_and(|d| d != unit && d != (-unit.0, -unit.1));
                            if pinnable {
                                continue;
                            }
                            blocked_beyond.push((from, unit, dist));
                            if dist == 2 {
                                occupied.insert(step(from, unit, 1));
                            }
                        }
                        _ => {}
                    }
                }
            }

            let check = if view.you_in_check {
                Self::checker_hyps(view, known_opponent, mask_version)
            } else {
                None
            };
            let king_return = if mask_version >= 5 {
                check.as_ref().and_then(|(king, hyps)| {
                    let [hyp] = hyps.as_slice() else { return None };
                    // 取った駒自身の桂王手だけが残る。開き王手の可能性があれば使わない。
                    if hyp.dir.is_none() && known_opponent == Some(hyp.square) {
                        Self::checked_king_origin(log, view.your_color, *king, hyp.square)
                    } else {
                        None
                    }
                })
            } else {
                None
            };
            Deduction {
                twins,
                occupied,
                blocked_beyond,
                check,
                king_return,
            }
        }

        fn checked_king_origin(log: &ObservationLog, color: Color, king: Coord, captured_at: Coord) -> Option<Coord> {
            let mut events = log.events().iter().rev();
            let accepted = |event: &&Observation| {
                matches!(event, Observation::MyMove { .. } | Observation::OpponentMoved { .. })
            };
            let Observation::OpponentMoved { move_number: capture_ply, .. } = events.find(accepted)? else {
                return None;
            };
            let Observation::MyMove { move_number: escape_ply, usi, captured: None } =
                events.find(accepted)? else {
                    return None;
                };
            let ShogiMove::Board { from, to, promote: false } = parse_usi(usi)? else { return None };
            if to != king || from.rank == captured_at.rank || escape_ply.checked_add(1) != Some(*capture_ply) {
                return None;
            }

            // 戻る升と桂の捕獲先は段が違うので、その桂は前の王手駒ではない。
            // 捕獲先はもともと自駒で塞がっていたので、桂の到着で前の王手筋も遮れない。
            let mut was_checked = false;
            for event in events {
                match event {
                    Observation::Check { in_check } if *in_check == color => was_checked = true,
                    Observation::OpponentMoved { move_number, .. } => {
                        return (was_checked && move_number.checked_add(1) == Some(*escape_ply)).then_some(from);
                    }
                    Observation::MyMove { .. } => break,
                    _ => {}
                }
            }
            None
        }

        /// 王手駒がいうるマスの集合（上の doc の H）。前提が崩れている（空になる）ときは None
        fn checker_hyps(
            view: &PlayerView,
            captured_at: Option<Coord>,
            mask_version: u32,
        ) -> Option<(Coord, Vec<CheckerHyp>)> {
            let own: HashSet<Coord> = view
                .your_pieces
                .iter()
                .filter_map(|p| parse_usi_square(&p.square))
                .collect();
            let king = view
                .your_pieces
                .iter()
                .find(|p| p.role == Role::King)
                .and_then(|p| parse_usi_square(&p.square))?;
            let opponent_roles = (mask_version >= 4 && captured_at.is_some())
                .then(|| possible_opponent_roles(view));
            let hyps = checker_hyps_on(
                &own, king, view.your_color, captured_at, mask_version >= 3, opponent_roles.as_deref(),
            );
            if hyps.is_empty() { None } else { Some((king, hyps)) }
        }

        fn rules_out(&self, usi: &str, mv: &ShogiMove) -> bool {
            if self.twins.contains(usi) {
                return true;
            }
            let on_occupied = self.occupied.iter().any(|&x| match *mv {
                ShogiMove::Drop { to, .. } => to == x,
                ShogiMove::Board { from, to, .. } => passes_through(from, to, x),
            });
            if on_occupied {
                return true;
            }
            if let ShogiMove::Board { from, to, .. } = *mv {
                let beyond = self.blocked_beyond.iter().any(|&(f, unit, dist)| {
                    f == from && (2..=8).any(|k| k > dist && step(f, unit, k) == to)
                });
                if beyond {
                    return true;
                }
            }
            let Some((king, hyps)) = &self.check else {
                return false;
            };
            let (to, is_king, is_drop) = match *mv {
                ShogiMove::Board { from, to, .. } => (to, from == *king, false),
                ShogiMove::Drop { to, .. } => (to, false, true),
            };
            if is_king && self.king_return == Some(to) {
                return true;
            }
            hyps.iter().all(|h| {
                if is_king {
                    if to == h.square {
                        return false;
                    }
                    match h.dir {
                        Some(d) if h.dist >= 2 => to == step(*king, d, 1) || to == step(*king, d, -1),
                        _ => false,
                    }
                } else {
                    if !is_drop && to == h.square {
                        return false;
                    }
                    let blocks = h.dir.is_some_and(|d| (1..h.dist).any(|k| step(*king, d, k) == to));
                    !blocks
                }
            })
        }
    }

    /// 王手駒がいうるマスの仮説（`Deduction` の doc の H）。自駒のマス `own`・自玉 `king`・王手した相手の手が取ったマス
    /// `captured_at` を使う。行動マスクv4以降では `opponent_roles` で移動元を絞る。空なら前提が崩れている。
    fn checker_hyps_on(
        own: &HashSet<Coord>,
        king: Coord,
        color: Color,
        captured_at: Option<Coord>,
        check_capture_paths: bool,
        opponent_roles: Option<&[Role]>,
    ) -> Vec<CheckerHyp> {
        // 8方向の「最初の自駒の手前まで」
        let rays: Vec<((i8, i8), Vec<Coord>)> = ALL_DIRS
            .iter()
            .map(|&d| {
                let mut squares = vec![];
                let mut c = step(king, d, 1);
                while on_board(c) && !own.contains(&c) {
                    squares.push(c);
                    c = step(c, d, 1);
                }
                (d, squares)
            })
            .collect();
        // 相手の桂が玉を狙える位置（相手から見て前へ跳んでくる）
        let forward = if color == Color::Sente { -2 } else { 2 };
        let knights: Vec<Coord> = [1, -1]
            .iter()
            .map(|&df| Coord {
                file: king.file + df,
                rank: king.rank + forward,
            })
            .filter(|&c| on_board(c) && !own.contains(&c))
            .collect();

        let ray_hyp = |d: (i8, i8), i: usize, c: Coord| CheckerHyp {
            square: c,
            dir: Some(d),
            dist: i as i8 + 1,
        };
        let knight_hyp = |c: Coord| CheckerHyp {
            square: c,
            dir: None,
            dist: 0,
        };

        let mut hyps = vec![];
        match captured_at {
            None => {
                for (d, squares) in &rays {
                    hyps.extend(squares.iter().enumerate().map(|(i, &c)| ray_hyp(*d, i, c)));
                }
                hyps.extend(knights.iter().map(|&c| knight_hyp(c)));
            }
            Some(x) => {
                // 着手駒そのものの王手
                for (d, squares) in &rays {
                    if let Some(i) = squares.iter().position(|&c| c == x) {
                        hyps.push(ray_hyp(*d, i, x));
                    }
                }
                if knights.contains(&x) {
                    hyps.push(knight_hyp(x));
                }
                // 開き王手: 元のマス o の先の筋の上
                for (d, squares) in &rays {
                    let opens = squares
                        .iter()
                        .enumerate()
                        .take(squares.len().saturating_sub(1))
                        .find(|&(_, &o)| {
                            o != x
                                && reachable(o, x)
                                && (!check_capture_paths
                                    || !own.iter().any(|&piece| passes_through(o, x, piece)))
                                && opponent_roles.is_none_or(|roles| {
                                    roles.iter().any(|&role| role_reaches(role, color.other(), o, x))
                                })
                        });
                    if let Some((oi, _)) = opens {
                        hyps.extend(
                            squares
                                .iter()
                                .enumerate()
                                .skip(oi + 1)
                                .map(|(i, &c)| ray_hyp(*d, i, c)),
                        );
                    }
                }
            }
        }
        hyps
    }

    /// 王手を受けた時点の、王手駒がいうるマス（重複なし。並びは筋・段の昇順）。エンコーダが過去の王手についても
    /// 同じ規則で候補を出すのに使う（`own` と `king` はその時点の自駒。開き王手の候補は自駒の遮りで絞る = マスク v3 と同じ）
    pub(crate) fn checker_squares_at(
        own: &HashSet<Coord>,
        king: Coord,
        color: Color,
        captured_at: Option<Coord>,
    ) -> Vec<Coord> {
        let mut out: Vec<Coord> = checker_hyps_on(own, king, color, captured_at, true, None)
            .into_iter()
            .map(|h| h.square)
            .collect();
        out.sort_by_key(|c| (c.file, c.rank));
        out.dedup();
        out
    }

}

/// 観測のテンソル化（`src/rl/encode.rs` の凍結時点の固定コピー）
pub mod encode {
    // 観測 → 入力テンソル（docs/rl-deepnash-design.md の「2. 観測エンコード」）。
    //
    // 入力は `Strategy::choose` と同じ `(PlayerView, ObservationLog, foul_tried)` だけ。
    // 学習中の環境も arena・本番の方策 Strategy もこの関数を通るので、学習時と推論時で
    // 特徴量が食い違わない。observation.rs にない情報（相手駒の位置など）は受け取れない。
    //
    // 出力は `[NUM_PLANES, 9, 9]` を平らにした f32 列（プレーン優先、マス番号は
    // `action::square_index`）。盤は手番側を先手向きに正規化する。

    use std::collections::HashSet;

    use crate::board::{Coord, on_board, orient, parse_usi_square, rays, steps};
    use crate::deduce::{ImmobilePawns, KingCandidates};
    use crate::deduce_king::KingCandidatesTight;
    use crate::model::GameModel;
    use crate::observation::{Observation, ObservationLog};
    use crate::protocol::{Color, PlayerView, Role};
    use super::action::{checker_squares_at, deduced_facts, normalize, square_index};
    // 凍結時点の手数・反則の上限（入力の正規化に使う）
    const MAX_FOULS: u32 = 10;
    const MAX_PLIES: u32 = 200;
    use crate::shogi::{HAND_ROLES, Piece, Position, ShogiMove, hand_index, parse_usi, unpromote_role};

    /// 自駒の駒種（`Role` の宣言順）
    pub(crate) const ROLES: [Role; 14] = [
        Role::Pawn,
        Role::Lance,
        Role::Knight,
        Role::Silver,
        Role::Gold,
        Role::Bishop,
        Role::Rook,
        Role::King,
        Role::Tokin,
        Role::Promotedlance,
        Role::Promotedknight,
        Role::Promotedsilver,
        Role::Horse,
        Role::Dragon,
    ];
    /// 自分の着手履歴として持つ直近の手数
    pub const HISTORY_MOVES: usize = 8;
    /// 鮮度の減衰（1手 = 1 move_number あたり）
    const DECAY: f32 = 0.95;
    /// 各駒種の総数（両者合計。`HAND_ROLES` 順）。相手の保有数の算出と正規化に使う
    pub(crate) const ROLE_TOTAL: [u32; 7] = crate::shogi::HAND_ROLE_TOTAL;

    // プレーンの配置
    pub(crate) const P_OWN: usize = 0; // 14
    pub(crate) const P_HIST: usize = P_OWN + 14; // 2 × HISTORY_MOVES（移動元, 移動先）
    pub(crate) const P_CAP: usize = P_HIST + 2 * HISTORY_MOVES; // 取ったマスの鮮度
    pub(crate) const P_CAP_ROLE: usize = P_CAP + 1; // 7: 取った駒種ごとの鮮度
    pub(crate) const P_LOST: usize = P_CAP_ROLE + 7; // 取られたマスの鮮度
    pub(crate) const P_LOST_COUNT: usize = P_LOST + 1; // 取られた回数
    // 反則は「今の手番」と「過去の手番」を同じ形で持つ（1組 = 移動元・移動先・成りの移動先・
    // 打ち先×駒種7）。打ちの駒種と成りを分けるのは、同じマスでも反則の原因（打ち歩詰め・
    // 成れない位置など）が違い、隠れた盤面への証拠が別物になるため
    pub(crate) const FOUL_GROUP: usize = 3 + 7;
    pub(crate) const P_FOUL: usize = P_LOST_COUNT + 1; // 今の手番の反則（値は 1）
    pub(crate) const P_PAST_FOUL: usize = P_FOUL + FOUL_GROUP; // 過去の手番の反則（値は鮮度）
    pub(crate) const F_FROM: usize = 0;
    pub(crate) const F_TO: usize = 1;
    pub(crate) const F_PROMO_TO: usize = 2;
    pub(crate) const F_DROP: usize = 3;
    pub(crate) const P_CHECKED_KING: usize = P_PAST_FOUL + FOUL_GROUP; // 王手されたときの自玉の位置（鮮度）
    pub(crate) const P_GAVE_CHECK: usize = P_CHECKED_KING + 1; // 王手をかけた手の移動先（鮮度）
    pub(crate) const P_UNMOVED: usize = P_GAVE_CHECK + 1; // 初期配置から一度も動いていない自駒
    pub(crate) const P_HAND: usize = P_UNMOVED + 1; // 7: 自分の持ち駒（全面に放送）
    pub(crate) const P_OPP_HOLD: usize = P_HAND + 7; // 7: 相手の駒種別保有数（盤上＋持ち駒。自分側から確定）
    pub(crate) const P_SCALAR: usize = P_OPP_HOLD + 7; // 以下のスカラー
    pub(crate) const S_MY_FOULS: usize = 0;
    pub(crate) const S_OPP_FOULS: usize = 1;
    pub(crate) const S_MY_LAST_FOUL: usize = 2; // あと1回で反則負け
    pub(crate) const S_OPP_LAST_FOUL: usize = 3;
    pub(crate) const S_FOULS_THIS_TURN: usize = 4;
    pub(crate) const S_IN_CHECK: usize = 5;
    pub(crate) const S_GAVE_CHECK_LAST: usize = 6; // 直前の自分の手で王手をかけた
    pub(crate) const S_MOVE_NUMBER: usize = 7;
    /// 観測ログから再構成した自駒・持ち駒が視界と食い違う（再起動で履歴を失った等）。
    /// 履歴由来の特徴（鮮度・未着手駒・王手履歴）が信用できないことをネットへ知らせる
    pub(crate) const S_HISTORY_BROKEN: usize = 8;
    pub(crate) const NUM_SCALARS: usize = 9;

    pub const NUM_PLANES: usize = P_SCALAR + NUM_SCALARS;
    pub const OBS_LEN: usize = NUM_PLANES * 81;

    // 確定事実のプレーン（エンコーダ v2、2026-10-16。docs/rl-deepnash-design.md の改善案2）。
    // v1（`NUM_PLANES`）の後ろに足すだけなので、v1 の重みの stem に 0 の列を足せば同じ関数から始められる。
    // どれも観測から論理的に確定する事実（真の盤面と矛盾しない）で、`action::deduced_facts` と `deduce` が作る
    pub(crate) const D_OCC: usize = NUM_PLANES; // 相手の駒がいると確定したマス
    pub(crate) const D_BLOCK: usize = D_OCC + 1; // 飛び越え反則の間のマス（どれかに相手の駒がいる。値は 1/マス数）
    pub(crate) const D_CHECKER: usize = D_BLOCK + 1; // 王手中: 王手駒がいうるマス（値は 1/仮説数）
    pub(crate) const D_KING: usize = D_CHECKER + 1; // 相手玉がいうるマス（deduce::opp_king_candidates）
    pub(crate) const D_PAWN: usize = D_KING + 1; // 動けないと確定した相手の歩（deduce::immobile_opponent_pawns）
    /// エンコーダ v2 のプレーン数（v1 + 確定事実 5枚）
    pub const NUM_PLANES_DEDUCED: usize = D_PAWN + 1;

    // エンコーダ v3（2026-11-08。docs/rl-deepnash-design.md の「エンコーダ v3」）。v2 の後ろに、観測と自分側の
    // 情報だけで決まる事実を2組足す。**並びは固定**で、学習でどの組を使うかはネットの側が持つ（tsuitate-nn の
    // `--opp-hand-planes` / `--opp-foul-planes`。使わない組は stem の重みの列が 0 に固定されるので、ここで書いた値は
    // 出力に効かない）。並びを固定にしておくと、組の選び方が違うネットどうし（錨・リーグ）が同じ観測を読める
    pub(crate) const E_HAND: usize = NUM_PLANES_DEDUCED; // 7: 相手の持ち駒の上界（駒種別。全面に放送）
    pub(crate) const E_FOUL: usize = E_HAND + 7; // 3: 相手の反則の時系列（全面に放送）
    pub(crate) const EF_LAST: usize = 0; // 相手の直前の手番の反則数
    pub(crate) const EF_CHECKED: usize = 1; // 王手を受けていた相手の手番の反則数（鮮度つきの合計）
    pub(crate) const EF_FREE: usize = 2; // 王手を受けていない相手の手番の反則数（鮮度つきの合計）
    /// エンコーダ v3 のプレーン数（v2 + 相手の持ち駒の上界 7枚 + 相手の反則の時系列 3枚）
    pub const NUM_PLANES_V3: usize = E_FOUL + 3;
    // エンコーダ v4（2026-11-08。設計 doc の「エンコーダ v3」の節の v4）。v3 の後ろに1枚足すだけなので、v3 の重みは
    // v4 の観測の先頭を読めばよい（v3 の実験の checkpoint をそのまま使える）。組の扱いは v3 と同じ（tsuitate-nn の
    // `--king-foul-planes`）
    pub(crate) const E_KING_FOUL: usize = NUM_PLANES_V3; // 1: 自玉の移動で反則したマス（鮮度は反則した手数から）
    /// エンコーダ v4 のプレーン数（v3 + 玉の移動で反則したマス 1枚）
    pub const NUM_PLANES_V4: usize = E_KING_FOUL + 1;
    // エンコーダ v5（2026-11-17。設計 doc の「エンコーダ v3」の節の v5）。v4 の後ろに「取った駒・取られた駒の一覧」を4組。
    // v1 の取った駒のプレーン（`P_CAP_ROLE`）は1手ごとに薄れ、同じマスの2回目で上書きされる。取られた駒は場所だけで
    // 駒種が無い（`P_LOST`）。ここでは薄れない回数と、取られた駒種・歩の切れた筋を持つ。組の扱いは v3 / v4 と同じ
    // （並びは固定で、使う組はネットが選ぶ。tsuitate-nn の `--lost-fresh-planes` / `--lost-role-planes` /
    // `--cap-role-planes` / `--pawn-file-planes`）
    pub(crate) const E_LOST_FRESH: usize = NUM_PLANES_V4; // 7: 取られた駒種ごとの、最後に取られた手数からの鮮度（全面に放送）
    pub(crate) const E_LOST_ROLE: usize = E_LOST_FRESH + 7; // 7: 取られた駒種ごとのマス（回数。薄れない）
    pub(crate) const E_CAP_ROLE: usize = E_LOST_ROLE + 7; // 7: 取った駒種ごとのマス（回数。薄れない）
    pub(crate) const E_PAWN_FILE: usize = E_CAP_ROLE + 7; // 1: 相手の歩を取った筋（筋の全マスに回数。薄れない）
    /// 回数のプレーンの1回ぶん（3回で 1 に張り付く。`P_LOST_COUNT` と同じ）
    const COUNT_STEP: f32 = 1.0 / 3.0;
    /// エンコーダ v5 のプレーン数（v4 + 取られた駒種の鮮度 7枚 + 取られた駒種のマス 7枚 + 取った駒種のマス 7枚 + 歩を取った筋 1枚）
    pub const NUM_PLANES_V5: usize = E_PAWN_FILE + 1;
    // エンコーダ v6（2026-11-23。設計 doc の「エンコーダ v3」の節の v6）。v5 の後ろに、安く作れる確定事実と集計を7組。
    // 1組ずつ効きを確かめるのは高くつく（本番の大きさで3日、CI の大きさでは v5 の差も再現しなかった。
    // docs/rl-input-ab.md）ので、まとめて足して版ごとに1回確かめる。組の扱いは v3〜v5 と同じ（並びは固定で、使う組は
    // ネットが選ぶ。tsuitate-nn の `--king-tight-planes` / `--check-clock-planes` / `--checker-memo-planes` /
    // `--own-attack-planes` / `--king-line-planes` / `--opp-count-planes` / `--exposed-planes`）
    pub(crate) const E_KING_TIGHT: usize = NUM_PLANES_V5; // 1: 相手玉がいうるマス（締めた版。deduce_king::opp_king_candidates_tight）
    pub(crate) const E_CHECK_CLOCK: usize = E_KING_TIGHT + 1; // 6: 王手からの手数（全面に放送）
    pub(crate) const CC_GAVE_EVER: usize = 0; // 王手をかけたことがある
    pub(crate) const CC_GAVE_MOVES: usize = 1; // 最後にかけてからの相手の手数（かけていなければ初手から）
    pub(crate) const CC_GAVE_FREE: usize = 2; // そのうち自由な手（王手への応手・駒取り・王手のどれでもない手）
    pub(crate) const CC_GOT_EVER: usize = 3; // 王手を受けたことがある
    pub(crate) const CC_GOT_MOVES: usize = 4; // 最後に受けてからの相手の手数（受けていなければ初手から）
    pub(crate) const CC_GOT_FREE: usize = 5; // そのうち自由な手
    /// 王手からの手数の上限（この手数で 1 に張り付く）
    pub(crate) const CLOCK_CAP: u32 = 12;
    pub(crate) const E_CHECKER_MEMO: usize = E_CHECK_CLOCK + 6; // 1: 取らずに逃げた王手駒がいたかもしれないマス（鮮度 ÷ 候補数）
    pub(crate) const E_OWN_ATTACK: usize = E_CHECKER_MEMO + 1; // 2: 自駒の利き（確実な利きの枚数, 遮られうる利きの枚数）
    pub(crate) const E_KING_LINE: usize = E_OWN_ATTACK + 2; // 2: 自玉の筋（最初の自駒の手前まで, ピンされうる自駒）
    pub(crate) const E_OPP_COUNT: usize = E_KING_LINE + 2; // 4: 相手の手数の合計（全面に放送）
    pub(crate) const OC_CHECKS: usize = 0; // 相手が王手をかけてきた手
    pub(crate) const OC_CAPTURES: usize = 1; // 相手が駒を取ってきた手
    pub(crate) const OC_BOTH: usize = 2; // 相手が王手をかけながら駒を取ってきた手
    pub(crate) const OC_MOVES: usize = 3; // 相手が指した手
    /// 相手の手数の合計の上限（この回数で 1 に張り付く。並びは `OC_*`）。本番 705局と自己対局 1,200局の手番ごとの分布から、
    /// 張り付くのが 1% に満たない値にした（王手 99% 点 10〜15・最大 37、駒取り 99% 点 16〜20・最大 33、
    /// 王手かつ駒取り 99% 点 3〜5・最大 15。指した手は手数の上限の半分）
    pub(crate) const OPP_COUNT_CAPS: [u32; 4] = [20, 30, 10, MAX_PLIES / 2];
    // 相手に位置が伝わっている自駒。相手は「自分の駒を取られたマス」と「王手を受けたこと」を審判から知らされるので、
    // 駒を取った自駒はそのマスにいると分かり、王手をかけた手で動かした駒も王手の筋から当たりを付けられる。その駒が
    // 動くか取られるまで印を残す。値はその手からの鮮度（0.95^手数。`P_CAP` / `P_GAVE_CHECK` の同じマスの値と同じ）
    pub(crate) const E_EXPOSED: usize = E_OPP_COUNT + 4; // 2: 駒を取ったまま動いていない自駒, 王手をかけた手で動かしたまま動いていない自駒
    /// エンコーダ v6 のプレーン数（v5 + 締めた相手玉の候補 1枚 + 王手からの手数 6枚 + 逃げた王手駒の候補 1枚 +
    /// 自駒の利き 2枚 + 自玉の筋 2枚 + 相手の手数の合計 4枚 + 相手に位置が伝わっている自駒 2枚）
    pub const NUM_PLANES_V6: usize = E_EXPOSED + 2;
    /// v3 以降で足したプレーンの組（名前, 先頭のプレーン, 枚数）。tsuitate-nn の `model.EXTRA_PLANES` と同じ
    pub const EXTRA_PLANE_GROUPS: [(&str, usize, usize); 14] = [
        ("opp_hand", E_HAND, 7),
        ("opp_fouls", E_FOUL, 3),
        ("king_fouls", E_KING_FOUL, 1),
        ("lost_fresh", E_LOST_FRESH, 7),
        ("lost_roles", E_LOST_ROLE, 7),
        ("cap_roles", E_CAP_ROLE, 7),
        ("pawn_files", E_PAWN_FILE, 1),
        ("king_tight", E_KING_TIGHT, 1),
        ("check_clock", E_CHECK_CLOCK, 6),
        ("checker_memo", E_CHECKER_MEMO, 1),
        ("own_attacks", E_OWN_ATTACK, 2),
        ("king_lines", E_KING_LINE, 2),
        ("opp_counts", E_OPP_COUNT, 4),
        ("exposed", E_EXPOSED, 2),
    ];

    /// プレーン数 → 観測の長さ。対応するエンコーダが無いプレーン数は None
    pub fn obs_len(planes: usize) -> Option<usize> {
        [NUM_PLANES, NUM_PLANES_DEDUCED, NUM_PLANES_V3, NUM_PLANES_V4, NUM_PLANES_V5, NUM_PLANES_V6]
            .contains(&planes)
            .then_some(planes * 81)
    }

    /// 反則1件を1組のプレーンへ書く（`base` は `P_FOUL` か `P_PAST_FOUL`）
    fn put_foul(p: &mut Planes, base: usize, usi: &str, v: f32) {
        match parse_usi(usi) {
            Some(ShogiMove::Board { from, to, promote }) => {
                p.max(base + F_FROM, from, v);
                p.max(base + F_TO, to, v);
                if promote {
                    p.max(base + F_PROMO_TO, to, v);
                }
            }
            Some(ShogiMove::Drop { role, to }) => {
                if let Some(i) = hand_index(role) {
                    p.max(base + F_DROP + i, to, v);
                }
            }
            None => {}
        }
    }

    /// 観測ログから再構成した自分側（盤上・持ち駒）が視界と一致するか
    fn history_consistent(model: &GameModel, view: &PlayerView) -> bool {
        if !model.consistent() {
            return false;
        }
        let mut a: Vec<(String, Role)> = view
            .your_pieces
            .iter()
            .map(|v| (v.square.clone(), v.role))
            .collect();
        let mut b: Vec<(String, Role)> = model
            .my_pieces()
            .into_iter()
            .map(|v| (v.square, v.role))
            .collect();
        a.sort();
        b.sort();
        let hand_a: Vec<u32> = HAND_ROLES
            .iter()
            .map(|r| view.your_hand.get(r).copied().unwrap_or(0))
            .collect();
        let model_hand = model.my_hand();
        let hand_b: Vec<u32> = HAND_ROLES
            .iter()
            .map(|r| model_hand.get(r).copied().unwrap_or(0))
            .collect();
        a == b && hand_a == hand_b
    }

    /// 受理手（自分の着手・相手の着手）が1つも欠けずに、今の手数 `now` まで並んでいるか。手数は初期局面で 1、受理の
    /// たびに +1 なので、欠けていなければ受理手の数は `now − 1`。受理手に付く手数の規約は2つある:
    /// 適用後の値（初手が 2、最後が `now`。審判・サーバー・Socket.IO 版、2026-11-24 からは webhook のセッションも）と、
    /// 適用前の値（初手が 1、最後が `now − 1`。〜2026-11-24 の webhook のセッション）。どちらの規約でも、手数が 1 ずつ
    /// 増えていて数が合えば欠けていない（片方の規約だけを受け付けると、もう片方では常に欠けている扱いになる）。
    /// なお鮮度のプレーンは手数そのものから測るので、適用前の規約のログでは1手ぶん古く出る
    pub(crate) fn moves_contiguous(log: &ObservationLog, now: u32) -> bool {
        let mut prev: Option<u32> = None;
        let mut count = 0u32;
        for event in log.events() {
            if let Observation::MyMove { move_number, .. } | Observation::OpponentMoved { move_number, .. } = event {
                if prev.is_some_and(|p| *move_number != p + 1) {
                    return false;
                }
                prev = Some(*move_number);
                count += 1;
            }
        }
        count + 1 == now
            && match prev {
                None => true,
                Some(last) => last == now || last + 1 == now,
            }
    }

    struct Planes<'a> {
        buf: &'a mut [f32],
        color: Color,
    }

    impl Planes<'_> {
        fn at(&mut self, plane: usize, c: Coord) -> &mut f32 {
            &mut self.buf[plane * 81 + square_index(normalize(c, self.color))]
        }
        fn set(&mut self, plane: usize, c: Coord, v: f32) {
            *self.at(plane, c) = v;
        }
        fn max(&mut self, plane: usize, c: Coord, v: f32) {
            let x = self.at(plane, c);
            *x = x.max(v);
        }
        fn add(&mut self, plane: usize, c: Coord, v: f32) {
            *self.at(plane, c) += v;
        }
        fn fill(&mut self, plane: usize, v: f32) {
            self.buf[plane * 81..(plane + 1) * 81].fill(v);
        }
    }

    fn move_squares(usi: &str) -> Option<(Option<Coord>, Coord)> {
        match parse_usi(usi)? {
            ShogiMove::Board { from, to, .. } => Some((Some(from), to)),
            ShogiMove::Drop { to, .. } => Some((None, to)),
        }
    }

    /// 確定事実のプレーンのうち、観測ログを先頭からたどって求めるもの（相手玉の候補・動けない歩）の
    /// 途中状態。同じ局・同じ側の観測を、ログを伸ばしながら何度も書くとき（RL の環境）に使い回すと、
    /// 伸びたぶんだけ処理すれば済む（毎回たどり直すと長い対局ほど重い）。結果は使い回さないときと同じ
    #[derive(Clone, Default)]
    pub struct DeducedCache {
        state: Option<(Color, KingCandidates, ImmobilePawns)>,
        /// 締めた相手玉の候補（v6 の観測を書くときだけ作る。`state` を作り直すときに一緒に捨てる）
        tight: Option<KingCandidatesTight>,
    }

    impl DeducedCache {
        /// `log` まで進めた途中状態。色が違う・ログが前回より短い（別の局）ときは作り直す
        fn advance(&mut self, color: Color, log: &ObservationLog) -> (&KingCandidates, &ImmobilePawns) {
            let stale = self
                .state
                .as_ref()
                .is_none_or(|(c, _, pawns)| *c != color || pawns.done() > log.events().len());
            if stale {
                self.state = Some((color, KingCandidates::new(color), ImmobilePawns::new(color)));
                self.tight = None;
            }
            let (_, king, pawns) = self.state.as_mut().expect("作った");
            king.advance(log);
            pawns.advance(log);
            (king, pawns)
        }

        /// `log` の時点の、締めた相手玉の候補（`deduce_king::opp_king_candidates_tight` と同じ。動けない歩のマスも除く）
        fn tight_candidates(&mut self, color: Color, log: &ObservationLog) -> std::collections::BTreeSet<Coord> {
            let locked = self.advance(color, log).1.result();
            let tight = self.tight.get_or_insert_with(|| KingCandidatesTight::new(color));
            tight.advance(log);
            tight.candidates(log, &locked)
        }
    }

    /// from から to への直線（縦横斜め）の、両端を除く途中に x があるか
    fn on_path(from: Coord, to: Coord, x: Coord) -> bool {
        let (df, dr) = (to.file - from.file, to.rank - from.rank);
        if !(df == 0 || dr == 0 || df.abs() == dr.abs()) {
            return false;
        }
        let (sf, sr) = (df.signum(), dr.signum());
        let mut c = Coord { file: from.file + sf, rank: from.rank + sr };
        while c != to {
            if c == x {
                return true;
            }
            c = Coord { file: c.file + sf, rank: c.rank + sr };
        }
        false
    }

    /// 縦横斜めの8方向（筋, 段）
    const LINE_DIRS: [(i8, i8); 8] = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1)];

    /// 観測をテンソルへ書く。`out` の長さでエンコーダの版を選ぶ: `OBS_LEN` は v1、
    /// `NUM_PLANES_DEDUCED * 81` は v2（v1 の後ろに確定事実のプレーン）、`NUM_PLANES_V3 * 81` は v3
    /// （v2 の後ろに相手の持ち駒の上界と相手の反則の時系列）、`NUM_PLANES_V4 * 81` は v4（v3 の後ろに玉の移動で反則したマス）、
    /// `NUM_PLANES_V5 * 81` は v5（v4 の後ろに取った駒・取られた駒の一覧）、`NUM_PLANES_V6 * 81` は v6
    /// （v5 の後ろに、締めた相手玉の候補・王手からの手数・逃げた王手駒の候補・自駒の利き・自玉の筋・相手の手数の合計・
    /// 相手に位置が伝わっている自駒）
    pub fn encode_into(
        view: &PlayerView,
        log: &ObservationLog,
        foul_tried: &HashSet<String>,
        out: &mut [f32],
    ) {
        encode_into_cached(view, log, foul_tried, out, &mut DeducedCache::default());
    }

    /// `encode_into` と同じ観測を、確定事実の途中状態 `cache` を使い回して書く。`cache` は同じ局・同じ側の
    /// 観測を書くときだけ使い回すこと（局が変わったら作り直す。色が違う・ログが短くなったときは中で作り直す）
    pub fn encode_into_cached(
        view: &PlayerView,
        log: &ObservationLog,
        foul_tried: &HashSet<String>,
        out: &mut [f32],
        cache: &mut DeducedCache,
    ) {
        assert!(
            out.len() % 81 == 0 && obs_len(out.len() / 81).is_some(),
            "観測の長さ {} に対応するエンコーダが無い",
            out.len()
        );
        let v3 = out.len() >= NUM_PLANES_V3 * 81;
        let v4 = out.len() >= NUM_PLANES_V4 * 81;
        let v5 = out.len() >= NUM_PLANES_V5 * 81;
        let v6 = out.len() == NUM_PLANES_V6 * 81;
        out.fill(0.0);
        if out.len() > OBS_LEN {
            encode_deduced(view, log, foul_tried, out, cache);
        }
        let color = view.your_color;
        let now = view.move_number;
        let fresh = |mn: u32| DECAY.powi(now.saturating_sub(mn) as i32);
        let mut p = Planes { buf: out, color };

        // 自駒
        for piece in &view.your_pieces {
            if let (Some(c), Some(r)) = (
                parse_usi_square(&piece.square),
                ROLES.iter().position(|&r| r == piece.role),
            ) {
                p.set(P_OWN + r, c, 1.0);
            }
        }

        // 履歴を先頭から辿る。自駒の配置（王手されたときの自玉の位置）は GameModel で追う
        let mut model = GameModel::new(color);
        let mut my_moves: Vec<&str> = vec![];
        let mut last_mn = 1u32;
        let mut gave_check_last = false;
        let initial: HashSet<Coord> = Position::initial()
            .pieces_of(color)
            .iter()
            .filter_map(|v| parse_usi_square(&v.square))
            .collect();
        let mut unmoved = initial.clone();
        // 相手の反則（v3）。反則は手番を変えないので、相手の手番の反則は `OpponentMoved` の手前に並ぶ。
        // 相手が王手を受けていた手番か = その手番の前の自分の手が王手（`gave_check_last`）
        let mut opp_fouls_pending = 0u32;
        let mut opp_fouls_last = 0u32;
        let (mut opp_fouls_checked, mut opp_fouls_free) = (0.0f32, 0.0f32);
        // 自玉の移動で反則したマス（v4）→ 最後に反則した手数。玉の手が反則になるのは「移動先に相手の利きがある」
        // ときだけ（自駒のマスへの手は候補に無い）なので、反則した時点でそのマスに相手の利きがあったことが確定する。
        // その後に玉がそのマスへ移動できたら（受理されたら）消す
        let mut king_sq = Position::initial().king_square(color);
        let mut king_fouls: Vec<(Coord, u32)> = vec![];
        // 取られた駒種ごとの、最後に取られた手数（v5）
        let mut lost_mn: [Option<u32>; 7] = [None; 7];
        // 王手からの手数（v6）。「自由な手」= 王手への応手でも駒取りでも王手でもない相手の手（行き先の手掛かりが無い）。
        // 王手かどうかは直後の `Check` で分かるので、いったん数えて王手だったら戻す
        let (mut gave_ever, mut got_ever) = (false, false);
        let (mut since_gave, mut free_gave, mut since_got, mut free_got) = (0u32, 0u32, 0u32, 0u32);
        let mut last_opp_free = false;
        // 逃げた王手駒の候補（v6）。`pending_check` = いま受けている王手の候補と手数（解消するまで）、
        // `checker_memo` = 最後に解消した王手の（残っている候補, 王手の時点の候補数, 手数）
        let mut last_opp_capture: Option<Coord> = None;
        let mut pending_check: Option<(Vec<Coord>, u32)> = None;
        let mut checker_memo: Option<(Vec<Coord>, usize, u32)> = None;
        // 相手の手数の合計（v6。並びは `OC_*`）。王手かどうかは直後の `Check` で分かる
        let mut opp_counts = [0u32; 4];
        // 相手に位置が伝わっている自駒（v6）: [駒を取った, 王手をかけた手で動かした] ごとの（マス, その手の手数）。
        // その駒が動いたら・取られたら外す
        let mut exposed: [Vec<(Coord, u32)>; 2] = Default::default();
        for event in log.events() {
            let lost_before = model.lost_pieces().len();
            match event {
                Observation::MyMove {
                    move_number,
                    usi,
                    captured,
                } => {
                    last_mn = *move_number;
                    my_moves.push(usi);
                    gave_check_last = false;
                    if let (true, Some((from, to))) = (v6, move_squares(usi)) {
                        // 王手を受けていた手番の受理手 = 王手を解消した。玉以外の駒で候補のマスの駒を取って解消したなら、
                        // 取ったのが王手駒（両王手は駒取りでは解消しない）なので何も残さない
                        if let Some((cands, mn)) = pending_check.take() {
                            let by_king = from.is_some() && from == king_sq;
                            let took_checker = captured.is_some() && !by_king && cands.contains(&to);
                            checker_memo = (!took_checker).then(|| {
                                let n = cands.len();
                                (cands, n, mn)
                            });
                        }
                        // 自駒が着地した・通り抜けたマスに、いまその駒はいない
                        if let Some((cands, _, _)) = &mut checker_memo {
                            cands.retain(|&c| c != to && !from.is_some_and(|f| on_path(f, to, c)));
                        }
                        // 動いた駒の印は外す。駒を取った駒は、取ったマスにいると相手に伝わる
                        for marks in &mut exposed {
                            marks.retain(|&(c, _)| Some(c) != from && c != to);
                        }
                        if captured.is_some() {
                            exposed[0].push((to, *move_number));
                        }
                    }
                    if let Some((from, to)) = move_squares(usi) {
                        if let Some(from) = from {
                            unmoved.remove(&from);
                            if Some(from) == king_sq {
                                king_sq = Some(to);
                                king_fouls.retain(|&(c, _)| c != to);
                            }
                        }
                        if let Some(role) = captured {
                            p.max(P_CAP, to, fresh(*move_number));
                            if let Some(i) = hand_index(*role) {
                                p.max(P_CAP_ROLE + i, to, fresh(*move_number));
                            }
                            if let (true, Some(i)) = (v5, hand_index(unpromote_role(*role))) {
                                p.add(E_CAP_ROLE + i, to, COUNT_STEP);
                                // 歩を取った筋: 相手の歩は初めに各筋1枚で、打ち直さない限りその筋から歩が消える
                                // （相手が歩を打てる筋 = 二歩にならない筋）。通知の駒種は成りを戻したものなので、
                                // と金を取ったときも区別できず歩として数える
                                if unpromote_role(*role) == Role::Pawn {
                                    for rank in 1..=9 {
                                        p.add(E_PAWN_FILE, Coord { file: to.file, rank }, COUNT_STEP);
                                    }
                                }
                            }
                        }
                    }
                }
                Observation::OpponentMoved {
                    move_number,
                    captured_my_piece_at,
                } => {
                    last_mn = *move_number;
                    if let Some(c) = captured_my_piece_at.as_deref().and_then(parse_usi_square) {
                        p.max(P_LOST, c, fresh(*move_number));
                        p.add(P_LOST_COUNT, c, 1.0 / 3.0);
                        unmoved.remove(&c);
                    }
                    // 王手からの手数（v6）
                    last_opp_capture = captured_my_piece_at.as_deref().and_then(parse_usi_square);
                    opp_counts[OC_MOVES] += 1;
                    opp_counts[OC_CAPTURES] += last_opp_capture.is_some() as u32;
                    if let Some(x) = last_opp_capture {
                        for marks in &mut exposed {
                            marks.retain(|&(c, _)| c != x);
                        }
                    }
                    since_gave += 1;
                    since_got += 1;
                    last_opp_free = !gave_check_last && last_opp_capture.is_none();
                    if last_opp_free {
                        free_gave += 1;
                        free_got += 1;
                    }
                    // 相手の手番がここで閉じる
                    let w = opp_fouls_pending as f32 * fresh(*move_number);
                    if gave_check_last {
                        opp_fouls_checked += w;
                    } else {
                        opp_fouls_free += w;
                    }
                    opp_fouls_last = opp_fouls_pending;
                    opp_fouls_pending = 0;
                }
                Observation::MyFoul { move_number, usi } => {
                    if let Some((Some(from), to)) = move_squares(usi) {
                        if Some(from) == king_sq {
                            king_fouls.retain(|&(c, _)| c != to);
                            king_fouls.push((to, *move_number));
                        }
                    }
                    // 今の手番の反則は下で foul_tried から載せる（反則は手番を変えないので
                    // move_number が今と同じものが今の手番）
                    if *move_number != now {
                        put_foul(&mut p, P_PAST_FOUL, usi, fresh(*move_number));
                    }
                }
                Observation::Check { in_check } => {
                    if *in_check == color {
                        // 直前の相手の手で王手された。自玉の位置は model が持つ（この後で apply）
                        let king = model
                            .my_pieces()
                            .into_iter()
                            .find(|v| v.role == Role::King)
                            .and_then(|v| parse_usi_square(&v.square));
                        if let Some(k) = king {
                            p.max(P_CHECKED_KING, k, fresh(last_mn));
                        }
                        got_ever = true;
                        since_got = 0;
                        free_got = 0;
                        opp_counts[OC_CHECKS] += 1;
                        opp_counts[OC_BOTH] += last_opp_capture.is_some() as u32;
                        if std::mem::take(&mut last_opp_free) {
                            free_gave = free_gave.saturating_sub(1); // 王手の手は相手玉の周りに使った手ではない
                        }
                        if let (true, Some(k)) = (v6, king) {
                            // 王手駒がいうるマス（この時点の自駒から。model は直前の駒取りを反映済み）
                            let own: HashSet<Coord> =
                                model.my_pieces().iter().filter_map(|v| parse_usi_square(&v.square)).collect();
                            pending_check = Some((checker_squares_at(&own, k, color, last_opp_capture), last_mn));
                        }
                    } else {
                        if let Some((_, to)) = my_moves.last().and_then(|u| move_squares(u)) {
                            p.max(P_GAVE_CHECK, to, fresh(last_mn));
                            exposed[1].push((to, last_mn));
                        }
                        gave_check_last = true;
                        gave_ever = true;
                        since_gave = 0;
                        free_gave = 0;
                    }
                }
                // 相手のまだ閉じていない手番の反則（手番側の観測を書くときは無い）は数えない
                Observation::OpponentFoul { .. } => opp_fouls_pending += 1,
            }
            model.apply(event);
            // 取られた自駒の駒種は、取られる前の自駒の配置（model）から決まる。model が追えていれば、いま
            // `lost_pieces` に1件足されている（成りは戻す = 相手の持ち駒に入る駒種）
            if let (true, Observation::OpponentMoved { move_number, .. }, Some(&(c, role))) =
                (v5, event, model.lost_pieces().get(lost_before))
            {
                if let Some(i) = hand_index(unpromote_role(role)) {
                    p.add(E_LOST_ROLE + i, c, COUNT_STEP);
                    lost_mn[i] = Some(*move_number);
                }
            }
        }
        for v in &mut p.buf[P_LOST_COUNT * 81..(P_LOST_COUNT + 1) * 81] {
            *v = v.min(1.0);
        }
        if v5 {
            for v in &mut p.buf[E_LOST_ROLE * 81..NUM_PLANES_V5 * 81] {
                *v = v.min(1.0);
            }
            for (i, mn) in lost_mn.iter().enumerate() {
                if let Some(mn) = *mn {
                    p.fill(E_LOST_FRESH + i, fresh(mn));
                }
            }
        }

        // 自分の着手履歴（新しい順）
        for (k, usi) in my_moves.iter().rev().take(HISTORY_MOVES).enumerate() {
            if let Some((from, to)) = move_squares(usi) {
                if let Some(from) = from {
                    p.set(P_HIST + 2 * k, from, 1.0);
                }
                p.set(P_HIST + 2 * k + 1, to, 1.0);
            }
        }

        // 今の手番の反則試行
        for usi in foul_tried {
            put_foul(&mut p, P_FOUL, usi, 1.0);
        }

        // 初期配置から動いていない自駒（今もそこに自駒がいるものだけ）
        for piece in &view.your_pieces {
            if let Some(c) = parse_usi_square(&piece.square) {
                if unmoved.contains(&c) {
                    p.set(P_UNMOVED, c, 1.0);
                }
            }
        }

        // 持ち駒と相手の保有数（総数 − 自駒 − 自分の持ち駒）。平手では駒の総数が一定なので
        // 視界だけから確定し、履歴が欠けていても正しい
        let mut mine = [0u32; 7];
        for piece in &view.your_pieces {
            if let Some(i) = hand_index(unpromote_role(piece.role)) {
                mine[i] += 1;
            }
        }
        let mut hold = [0u32; 7];
        for (i, role) in HAND_ROLES.iter().enumerate() {
            let n = view.your_hand.get(role).copied().unwrap_or(0);
            let total = ROLE_TOTAL[i] as f32;
            p.fill(P_HAND + i, n as f32 / total);
            hold[i] = ROLE_TOTAL[i].saturating_sub(mine[i] + n);
            p.fill(P_OPP_HOLD + i, hold[i] as f32 / total);
        }
        let history_broken = !history_consistent(&model, view);
        let max_fouls = MAX_FOULS as f32;

        if v3 {
            // 相手の持ち駒の上界。相手の持ち駒は取った自駒からしか増えないので、取られた自駒の駒種別の数
            // （成りは戻す）が上界になる。相手の打ちは見えないので取った数からは減らせない。
            // 相手の保有数（盤上＋持ち駒）でも抑える（こちらが相手の駒を取れば下がる）。着手履歴が欠けているときは
            // 取られた駒を数え漏らしうるので、保有数だけを使う（これも上界）。
            // 欠けの判定は自駒の配置の照合（`history_broken`）だけでは足りない: 取られた自駒を取り返して打ち直すと、
            // 欠けた区間の後でも自駒の配置と持ち駒が視界と一致する（再起動後の空ログで、相手は歩を1枚持つのに
            // 上界が0になる）。受理手の手数が欠けずに今の手数まで並んでいることも確かめる
            let lost_counted = !history_broken && moves_contiguous(log, now);
            let mut lost = [0u32; 7];
            for &(_, role) in model.lost_pieces() {
                if let Some(i) = hand_index(unpromote_role(role)) {
                    lost[i] += 1;
                }
            }
            for i in 0..7 {
                let bound = if lost_counted { lost[i].min(hold[i]) } else { hold[i] };
                p.fill(E_HAND + i, bound as f32 / ROLE_TOTAL[i] as f32);
            }
            // 相手の反則の時系列（鮮度は相手の手番を閉じた `OpponentMoved` の手数で測る。P_LOST と同じ）
            p.fill(E_FOUL + EF_LAST, opp_fouls_last as f32 / max_fouls);
            p.fill(E_FOUL + EF_CHECKED, opp_fouls_checked / max_fouls);
            p.fill(E_FOUL + EF_FREE, opp_fouls_free / max_fouls);
        }
        // 着手履歴が欠けていると自玉の位置を追えないので書かない（違うマスを確定事実として載せない）
        if v4 && !history_broken {
            for &(c, mn) in &king_fouls {
                p.set(E_KING_FOUL, c, fresh(mn));
            }
        }

        if v6 {
            let clock = |n: u32| n.min(CLOCK_CAP) as f32 / CLOCK_CAP as f32;
            for (k, v) in [
                (CC_GAVE_EVER, gave_ever as u8 as f32),
                (CC_GAVE_MOVES, clock(since_gave)),
                (CC_GAVE_FREE, clock(free_gave)),
                (CC_GOT_EVER, got_ever as u8 as f32),
                (CC_GOT_MOVES, clock(since_got)),
                (CC_GOT_FREE, clock(free_got)),
            ] {
                p.fill(E_CHECK_CLOCK + k, v);
            }
            // 自駒だけを置いた盤（相手の駒は見えないので、飛び駒の利きは自駒にしか遮られない上界になる）
            let mut board = Position::empty(color);
            for piece in &view.your_pieces {
                if let Some(c) = parse_usi_square(&piece.square) {
                    board.set(c, Some(Piece { color, role: piece.role }));
                }
            }
            let own: Vec<Coord> = board.pieces().map(|(c, _)| c).collect();
            // 締めた相手玉の候補は「相手のどの手で玉が動けたか」を1手ずつ絞るので、相手の手が1つでも欠けたログ
            // （再起動後の復帰対局・途中の取りこぼし）からは真の玉を落としうる。自駒の配置が追えていても、相手の
            // 駒を取らない手の欠けは分からないので、受理手の手数が欠けずに今の手数まで並んでいることも確かめる。
            // どちらかが崩れていたら上界（自駒のいないマス全部）を書く
            if !history_broken && moves_contiguous(log, now) {
                for c in cache.tight_candidates(color, log) {
                    p.set(E_KING_TIGHT, c, 1.0);
                }
            } else {
                for file in 1..=9 {
                    for rank in 1..=9 {
                        let c = Coord { file, rank };
                        if board.piece_at(c).is_none() {
                            p.set(E_KING_TIGHT, c, 1.0);
                        }
                    }
                }
            }
            // 逃げた王手駒の候補。値は 鮮度 ÷ 王手の時点の候補数（そのマスにまだいる見込みの目安）。いま自駒がいるマスは除く。
            // 着手履歴が欠けていると王手の時点の自駒が追えないので書かない
            if let (false, Some((cands, n, mn))) = (history_broken, &checker_memo) {
                for &c in cands {
                    if board.piece_at(c).is_none() {
                        p.set(E_CHECKER_MEMO, c, fresh(*mn) / *n as f32);
                    }
                }
            }
            // 自駒の利き。確実な利き = 隣接か桂の跳び（未知の駒に遮られ得ない）、遮られうる利き = 飛び駒の2マス目以降
            // （最初の自駒まで。自駒のいるマスも数える = 紐）。値は枚数 × 1/3 で 3枚以上は 1
            let mut attack = [[0u8; 81]; 2];
            let at = |c: Coord| ((c.file - 1) * 9 + (c.rank - 1)) as usize;
            for &from in &own {
                let role = board.piece_at(from).expect("自駒のマス").role;
                for &d in steps(role) {
                    let (df, dr) = orient(d, color);
                    let c = Coord { file: from.file + df, rank: from.rank + dr };
                    if on_board(c) {
                        attack[0][at(c)] += 1;
                    }
                }
                for &d in rays(role) {
                    let (df, dr) = orient(d, color);
                    let mut c = Coord { file: from.file + df, rank: from.rank + dr };
                    let mut k = 0; // 1マス目は確実、2マス目以降は遮られうる
                    while on_board(c) {
                        attack[k][at(c)] += 1;
                        k = 1;
                        if board.piece_at(c).is_some() {
                            break;
                        }
                        c = Coord { file: c.file + df, rank: c.rank + dr };
                    }
                }
            }
            for file in 1..=9 {
                for rank in 1..=9 {
                    let sq = Coord { file, rank };
                    for (k, counts) in attack.iter().enumerate() {
                        if counts[at(sq)] > 0 {
                            p.set(E_OWN_ATTACK + k, sq, (counts[at(sq)] as f32 * COUNT_STEP).min(1.0));
                        }
                    }
                }
            }
            // 自玉の筋: 8方向に最初の自駒の手前までのマス（相手の飛び駒がいれば王手になる筋）と、その最初の自駒の
            // うち先に自駒でないマスが続くもの（後ろに相手の飛び駒がいればピンされていて、筋から外す手は反則になる）
            let king = view.your_pieces.iter().find(|x| x.role == Role::King).and_then(|x| parse_usi_square(&x.square));
            if let Some(k) = king {
                for (df, dr) in LINE_DIRS {
                    let next = |c: Coord| Coord { file: c.file + df, rank: c.rank + dr };
                    let mut c = next(k);
                    while on_board(c) && board.piece_at(c).is_none() {
                        p.set(E_KING_LINE, c, 1.0);
                        c = next(c);
                    }
                    if on_board(c) && on_board(next(c)) && board.piece_at(next(c)).is_none() {
                        p.set(E_KING_LINE + 1, c, 1.0);
                    }
                }
            }
            // 相手の手数の合計
            for (k, &n) in opp_counts.iter().enumerate() {
                p.fill(E_OPP_COUNT + k, n.min(OPP_COUNT_CAPS[k]) as f32 / OPP_COUNT_CAPS[k] as f32);
            }
            // 相手に位置が伝わっている自駒。着手履歴が欠けていると、動いた・取られたを追えないので書かない
            if !history_broken {
                for (k, marks) in exposed.iter().enumerate() {
                    for &(c, mn) in marks {
                        if board.piece_at(c).is_some() {
                            p.set(E_EXPOSED + k, c, fresh(mn));
                        }
                    }
                }
            }
        }

        // スカラー
        let scalars = [
            (S_MY_FOULS, view.fouls.you as f32 / max_fouls),
            (S_OPP_FOULS, view.fouls.opponent as f32 / max_fouls),
            (S_MY_LAST_FOUL, (view.fouls.you + 1 >= MAX_FOULS) as u8 as f32),
            (S_OPP_LAST_FOUL, (view.fouls.opponent + 1 >= MAX_FOULS) as u8 as f32),
            (S_FOULS_THIS_TURN, foul_tried.len() as f32 / max_fouls),
            (S_IN_CHECK, view.you_in_check as u8 as f32),
            (S_GAVE_CHECK_LAST, gave_check_last as u8 as f32),
            (S_MOVE_NUMBER, now as f32 / MAX_PLIES as f32),
            (S_HISTORY_BROKEN, history_broken as u8 as f32),
        ];
        for (s, v) in scalars {
            p.fill(P_SCALAR + s, v);
        }
    }

    /// 確定事実のプレーン（v2 の後ろ5枚）を書く。`out` は 0 で埋めてあること
    fn encode_deduced(
        view: &PlayerView,
        log: &ObservationLog,
        foul_tried: &HashSet<String>,
        out: &mut [f32],
        cache: &mut DeducedCache,
    ) {
        let color = view.your_color;
        let facts = deduced_facts(view, log, foul_tried);
        let mut p = Planes { buf: out, color };
        for &c in &facts.occupied {
            p.set(D_OCC, c, 1.0);
        }
        for seg in &facts.blocked {
            let v = 1.0 / seg.len() as f32;
            for &c in seg {
                p.max(D_BLOCK, c, v);
            }
        }
        if !facts.checkers.is_empty() {
            let v = 1.0 / facts.checkers.len() as f32;
            for &c in &facts.checkers {
                p.set(D_CHECKER, c, v);
            }
        }
        // deduce::opp_king_candidates / immobile_opponent_pawns と同じ（途中状態を使い回すだけ）
        let (king, pawns) = cache.advance(color, log);
        let locked = pawns.result();
        for c in king.candidates(log, &locked) {
            p.set(D_KING, c, 1.0);
        }
        for c in locked {
            p.set(D_PAWN, c, 1.0);
        }
    }

    /// v1 の観測（長さ `OBS_LEN`）
    pub fn encode(view: &PlayerView, log: &ObservationLog, foul_tried: &HashSet<String>) -> Vec<f32> {
        encode_planes(NUM_PLANES, view, log, foul_tried)
    }

    /// プレーン数 `planes`（`NUM_PLANES` / `NUM_PLANES_DEDUCED` / `NUM_PLANES_V3` / `NUM_PLANES_V4` / `NUM_PLANES_V5` /
    /// `NUM_PLANES_V6`）の観測。
    /// 重みのヘッダのプレーン数を渡す
    pub fn encode_planes(
        planes: usize,
        view: &PlayerView,
        log: &ObservationLog,
        foul_tried: &HashSet<String>,
    ) -> Vec<f32> {
        let len = obs_len(planes).unwrap_or_else(|| panic!("プレーン数 {planes} に対応するエンコーダが無い"));
        let mut out = vec![0.0; len];
        encode_into(view, log, foul_tried, &mut out);
        out
    }

}

/// 手書きの推論（`src/rl/policy_net.rs` の凍結時点の固定コピー）
pub mod policy_net {
    // 方策・価値ネットの推論（手書きの forward pass。依存を増やさないため）。
    //
    // 学習は `~/Develop/tsuitate-nn/rnad/`（`model.py`）、重みは `export_weights.py` が
    // BatchNorm を畳み込みへ畳み込んで書き出す。形式は
    // `b"TSRLPV01" | u32 ヘッダ長 | ヘッダ JSON | f32 LE の並び`。
    //
    // 構成（`model.py` と同じ。層を変えたら両方と書き出しを直すこと）:
    // 3×3 畳み込み（入力 86 → C）→ 残差ブロック×B（3×3 畳み込み×2）→
    // 方策: 1×1（C→32）ReLU → 1×1（32→139）= [139, 9, 9] を平らにしたロジット、
    // 価値: 1×1（C→4）ReLU → 全結合 324→128 ReLU → 128→1 → tanh。
    // 盤は `[チャネル, 段, 筋]` の並び（`rl::encode` のプレーン・`rl::action` のマスと同じ）。

    use std::collections::HashMap;
    use std::io::Read;

    use super::action::{NUM_ACTION_KINDS, NUM_ACTIONS};
    use super::encode::{NUM_PLANES, obs_len};

    const MAGIC: &[u8; 8] = b"TSRLPV01";
    const SQ: usize = 81;

    #[derive(Debug)]
    pub struct PolicyNet {
        planes: usize,
        channels: usize,
        blocks: usize,
        tensors: HashMap<String, Vec<f32>>,
    }

    #[derive(serde::Deserialize)]
    struct Header {
        planes: usize,
        channels: usize,
        blocks: usize,
        tensors: Vec<TensorInfo>,
    }

    #[derive(serde::Deserialize)]
    struct TensorInfo {
        name: String,
        shape: Vec<usize>,
    }

    impl PolicyNet {
        pub fn load(path: &str) -> Result<Self, String> {
            let mut bytes = vec![];
            std::fs::File::open(path)
                .and_then(|mut f| f.read_to_end(&mut bytes))
                .map_err(|e| format!("{path}: {e}"))?;
            Self::from_bytes(&bytes)
        }

        pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
            if bytes.len() < 12 || &bytes[..8] != MAGIC {
                return Err("方策ネットの重みファイルではない（magic 不一致）".into());
            }
            let hlen = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
            let header: Header = serde_json::from_slice(
                bytes.get(12..12 + hlen).ok_or("ヘッダが切れている")?,
            )
            .map_err(|e| e.to_string())?;
            if obs_len(header.planes).is_none() {
                return Err(format!(
                    "入力プレーン数 {} に対応するエンコーダが無い（v1 = {NUM_PLANES}、v2 = {}、v3 = {}、v4 = {}、v5 = {}、v6 = {}）",
                    header.planes,
                    super::encode::NUM_PLANES_DEDUCED,
                    super::encode::NUM_PLANES_V3,
                    super::encode::NUM_PLANES_V4,
                    super::encode::NUM_PLANES_V5,
                    super::encode::NUM_PLANES_V6
                ));
            }
            let mut off = 12 + hlen;
            let mut tensors = HashMap::new();
            for t in &header.tensors {
                let n: usize = t.shape.iter().product();
                let raw = bytes
                    .get(off..off + 4 * n)
                    .ok_or_else(|| format!("{} のデータが切れている", t.name))?;
                let v: Vec<f32> = raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                tensors.insert(t.name.clone(), v);
                off += 4 * n;
            }
            if off != bytes.len() {
                return Err("末尾に余分なデータがある".into());
            }
            let net = PolicyNet {
                planes: header.planes,
                channels: header.channels,
                blocks: header.blocks,
                tensors,
            };
            net.check_shapes()?;
            Ok(net)
        }

        fn check_shapes(&self) -> Result<(), String> {
            let c = self.channels;
            let mut want: Vec<(String, usize)> = vec![
                ("stem.w".into(), c * self.planes * 9),
                ("stem.b".into(), c),
                ("p1.w".into(), 32 * c),
                ("p1.b".into(), 32),
                ("p2.w".into(), NUM_ACTION_KINDS * 32),
                ("p2.b".into(), NUM_ACTION_KINDS),
                ("v1.w".into(), 4 * c),
                ("v1.b".into(), 4),
                ("v_fc1.w".into(), 128 * 4 * SQ),
                ("v_fc1.b".into(), 128),
                ("v_fc2.w".into(), 128),
                ("v_fc2.b".into(), 1),
            ];
            for i in 0..self.blocks {
                for j in 1..=2 {
                    want.push((format!("block{i}.c{j}.w"), c * c * 9));
                    want.push((format!("block{i}.c{j}.b"), c));
                }
            }
            for (name, len) in want {
                match self.tensors.get(&name) {
                    Some(t) if t.len() == len => {}
                    Some(t) => return Err(format!("{name} の長さ {} が {len} と違う", t.len())),
                    None => return Err(format!("{name} が無い")),
                }
            }
            Ok(())
        }

        fn t(&self, name: &str) -> &[f32] {
            &self.tensors[name]
        }

        /// 入力プレーン数（エンコーダの版。`rl::encode::encode_planes` に渡す）
        pub fn planes(&self) -> usize {
            self.planes
        }

        /// 観測（長さ `planes() * 81`）→ (ロジット（長さ `NUM_ACTIONS`、マスク前）, 価値 ∈ [−1, 1])
        pub fn forward(&self, obs: &[f32]) -> (Vec<f32>, f32) {
            assert_eq!(obs.len(), self.planes * SQ);
            let c = self.channels;
            let mut x = conv3x3(obs, self.planes, c, self.t("stem.w"), self.t("stem.b"));
            relu(&mut x);
            for i in 0..self.blocks {
                let mut y = conv3x3(&x, c, c, self.t(&format!("block{i}.c1.w")), self.t(&format!("block{i}.c1.b")));
                relu(&mut y);
                let y = conv3x3(&y, c, c, self.t(&format!("block{i}.c2.w")), self.t(&format!("block{i}.c2.b")));
                for (a, b) in x.iter_mut().zip(&y) {
                    *a = (*a + b).max(0.0);
                }
            }
            let mut p = conv1x1(&x, c, 32, self.t("p1.w"), self.t("p1.b"));
            relu(&mut p);
            let logits = conv1x1(&p, 32, NUM_ACTION_KINDS, self.t("p2.w"), self.t("p2.b"));
            debug_assert_eq!(logits.len(), NUM_ACTIONS);

            let mut v = conv1x1(&x, c, 4, self.t("v1.w"), self.t("v1.b"));
            relu(&mut v);
            let mut h = linear(&v, 128, self.t("v_fc1.w"), self.t("v_fc1.b"));
            relu(&mut h);
            let value = linear(&h, 1, self.t("v_fc2.w"), self.t("v_fc2.b"))[0].tanh();
            (logits, value)
        }
    }

    fn relu(x: &mut [f32]) {
        for v in x {
            *v = v.max(0.0);
        }
    }

    /// 3×3 畳み込み（パディング1）。`w` は [出力, 入力, 3, 3]
    fn conv3x3(x: &[f32], cin: usize, cout: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; cout * SQ];
        for co in 0..cout {
            let o = &mut out[co * SQ..(co + 1) * SQ];
            o.fill(b[co]);
            for ci in 0..cin {
                let src = &x[ci * SQ..(ci + 1) * SQ];
                for ky in 0..3 {
                    for kx in 0..3 {
                        let wv = w[((co * cin + ci) * 3 + ky) * 3 + kx];
                        if wv == 0.0 {
                            continue;
                        }
                        let (dy, dx) = (ky as isize - 1, kx as isize - 1);
                        let (y0, y1) = ((-dy).max(0) as usize, (9 - dy).min(9) as usize);
                        let (x0, x1) = ((-dx).max(0) as usize, (9 - dx).min(9) as usize);
                        for y in y0..y1 {
                            let sy = (y as isize + dy) as usize;
                            let orow = &mut o[y * 9 + x0..y * 9 + x1];
                            let srow = &src[sy * 9 + (x0 as isize + dx) as usize..];
                            for (ov, sv) in orow.iter_mut().zip(srow) {
                                *ov += wv * sv;
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// 1×1 畳み込み。`w` は [出力, 入力]
    fn conv1x1(x: &[f32], cin: usize, cout: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; cout * SQ];
        for co in 0..cout {
            let o = &mut out[co * SQ..(co + 1) * SQ];
            o.fill(b[co]);
            for ci in 0..cin {
                let wv = w[co * cin + ci];
                for (ov, sv) in o.iter_mut().zip(&x[ci * SQ..(ci + 1) * SQ]) {
                    *ov += wv * sv;
                }
            }
        }
        out
    }

    /// 全結合。`w` は [出力, 入力]
    fn linear(x: &[f32], nout: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
        let nin = x.len();
        (0..nout)
            .map(|o| b[o] + w[o * nin..(o + 1) * nin].iter().zip(x).map(|(a, b)| a * b).sum::<f32>())
            .collect()
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 埋め込んだ定跡は凍結時点のもの() {
        use sha2::{Digest, Sha256};
        let got: String = Sha256::digest(OPENINGS.as_bytes()).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(got, OPENINGS_SHA256);
        assert!(book().line_lengths().iter().all(|&n| n > 0));
    }

    #[test]
    fn 埋め込んだ重みは凍結時点のもの() {
        use sha2::{Digest, Sha256};
        let got: String = Sha256::digest(WEIGHTS).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(got, WEIGHTS_SHA256);
        assert!(net().forward(&vec![0.0; net().planes() * 81]).0.len() == action::NUM_ACTIONS);
    }
}
