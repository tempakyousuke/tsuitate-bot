//! 相手局面の推定（determinization / パーティクルフィルタ）。
//!
//! ついたて将棋では相手の初期配置は既知（平手の初期局面）なので、
//! 「あり得る相手局面」= 観測と整合する相手の指し手列。厳密な情報集合は
//! 指数的に爆発するため、粒子（具体的なフル局面）の集合でモンテカルロ近似する。
//!
//! 使う観測（公平性: observation.rs にあるものだけ）:
//! - 自分の受理された手 … 粒子上でも合法で、取った駒種が一致しなければ棄却
//! - 自分の反則手 … 粒子上で合法だったら棄却（真の局面では非合法だったので）
//! - 相手の着手 … 粒子上の相手合法手から「取られたマス・王手宣言の有無」と
//!   整合する手をサンプルして進める。整合手がなければ棄却
//! - 王手宣言（の有無）… 手の直後の王手状態と一致しない粒子を棄却
//!
//! 粒子が枯渇したら、制約列を最初からリプレイして再生成する（回数上限つき）。
//!
//! 観測尤度の重み（SIR の重み更新）:
//! 相手手のサンプルは観測と整合するクラスへ絞ってから事前分布で正規化するため、
//! そのままでは「観測を相手が指しにくい手でしか説明できない粒子」も確率1で
//! 生き残ってしまう（例: 桂がいない粒子では角の飛び込み王手が強制される）。
//! そこで制約適用のたびに r = 整合クラスの事前質量 / 全合法手の事前質量 を
//! 対数重み logw へ累積し、評価側（strategy.rs の stratified_sample）が
//! 粒子間で正規化して乗じる。リプレイ生成粒子も全制約ぶん累積するので比較可能。
//!
//! ソフト粒子（POMCP の particle reinvigoration の変種）:
//! 厳密整合の生存粒子が target/4 を下回ったときは、棄却された粒子を
//! 「情報系の制約だけを緩和した」判定で救済し、info_miss を加算して残す。
//! 緩和するのは王手宣言の一致と自分の反則の説明のみで、物理的な制約
//! （自手の合法性・取った駒種・取られたマス）は緩和しない。救済時は
//! 観測尤度 EPS_INFO を logw へ課金する（C-7 P1: 従来の評価側 0.5^penalty
//! 減衰を連続重みへ統合）。info_miss は課金回数の別勘定カウンタで、
//! 上限管理（INFO_MISS_CAP）と評価側の較正（証拠数の勘定）にだけ使う。
//!
//! ESS リサンプリング（C-7 P1 / D2）:
//! logw の退化を ESS = (Σw)²/Σw² で監視し、閾値を割ったら systematic
//! resampling で質量を複製数へ実現して logw をリセットする。退化していないが
//! 頭数が目標に足りないときは、質量保存の分割複製（コピーと元で exp(logw) を
//! 等分）で埋める — 分布を変えずに、次の相手手サンプルで分岐する多様性の種を
//! 蒔く（従来の複製埋めの後継。評価側が multiplicity を畳み込むようになったため
//! 質量保存でない複製は事後分布を偏らせる）。info_miss はリサンプリングでも
//! リセットしない（嘘の昇格防止・較正はカウンタが担う）。

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::board::Coord;
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{Color, Role};
use crate::shogi::{Position, ShogiMove, parse_usi, unpromote_role};

/// 粒子の目標数。1手あたりの計算量はこれ*候補手数に比例する
const TARGET_PARTICLES: usize = 512;
/// 1回の update での再生成リプレイ試行の上限（時間予算の担保）。
/// 複製よりリプレイのほうが粒子の多様性を保てるので多めに取る。
/// v6: 相手モデルのフィット（2026-07-09）で提案分布の打率が上がったぶん
/// 試行回数の効果が大きくなったので、思考予算の余り（平均360ms/目安1〜2秒）を
/// リプレイに振る
const REGEN_ATTEMPTS: usize = 320;
/// リプレイ中バックトラックの1決定点あたりの再サンプル回数
const BACKTRACK_ATTEMPTS: u32 = 4;
/// ソフト救済の累積回数の上限。超えた粒子は棄却する
/// （観測と何度も矛盾した粒子は近似としても信用できない）。
/// ソフト救済の発動閾値は target/4（apply_constraint 参照）
const INFO_MISS_CAP: u8 = 3;
/// 情報系ソフト救済1回あたりの観測尤度（logw へ ln(EPS_INFO) を課金）。
/// 評価側の較正（証拠数の勘定）にも同じ値を使うため pub。
/// C-7 P1 で評価側の soft_decay^penalty（旧0.6753）を置き換えた。
/// フィルタ超パラメータなので調整は SPSA でなくグリッド＋シナリオ目的で行う
pub const EPS_INFO: f64 = 0.1;
/// ESS がこの割合（対 現粒子数）を下回ったら systematic resampling
const ESS_THRESHOLD: f64 = 0.5;

/// 観測列を推定に使える形に正規化した制約
#[derive(Debug, Clone)]
enum Constraint {
    /// 受理された自分の手（gives_check: 直後に相手玉へ王手宣言があったか）
    MyMove {
        mv: ShogiMove,
        captured: Option<Role>,
        gives_check: bool,
    },
    /// 反則になった自分の手（真の局面では非合法）
    MyFoul { mv: ShogiMove },
    /// 相手の着手（captured_at: 自駒が取られたマス、gives_check: 自玉への王手宣言）
    OppMove {
        captured_at: Option<Coord>,
        gives_check: bool,
    },
}

pub struct Estimator {
    my_color: Color,
    particles: Vec<Position>,
    /// particles と同じ並びのソフト救済回数（0 = 全制約と厳密整合）。
    /// 尤度の課金（EPS_INFO）は logw 側で行い、これは回数の別勘定
    /// （上限管理と評価側の証拠数較正用）。リサンプリングでもリセットしない
    info_miss: Vec<u8>,
    /// particles と同じ並びの観測尤度の対数重み（SIR の重み更新）。
    /// 相手手の制約適用ごとに log(整合クラスの事前質量 / 全合法手の事前質量) を
    /// 累積する。「観測と整合する手はあるが、それが相手として指しにくい手しか
    /// ない粒子」（例: 幻の角の飛び込み王手でしか王手を説明できない粒子）を
    /// 粒子間で相対的に軽くする。リプレイ生成粒子も全制約ぶん累積するので
    /// 生存粒子と比較可能。絶対値に意味はなく、評価側が max を引いて正規化する
    logw: Vec<f64>,
    /// 思考予算に応じた粒子の目標数（スケール1.0で TARGET_PARTICLES）
    target: usize,
    /// リプレイ試行回数の上限（スケール比例）
    regen_attempts: usize,
    /// 通常リプレイの時間打ち切り（ms、スケール比例）
    regen_deadline_ms: u64,
    /// 全滅時に粘る時間の上限（ms、スケール比例）
    empty_deadline_ms: u64,
    constraints: Vec<Constraint>,
    /// 自分が駒を取ったマス（= 相手は自駒がそこで死んだことを知っている）。
    /// 相手手の事前分布の threat_known 特徴量に使う。idx は制約列上の位置
    my_capture_idx: Vec<usize>,
    my_capture_sq: Vec<Coord>,
    /// 自分の手が触れたマス（from/to）。初期配置から動いていない自駒
    /// （相手が推論で狙ってくる = threat_home 特徴量）の判定に使う
    my_touched_idx: Vec<usize>,
    my_touched_sq: Vec<Coord>,
    /// ObservationLog の消化済みイベント数
    cursor: usize,
    /// 観測との矛盾（リプレイでも整合局面を作れない等）で信頼できなくなったら false
    healthy: bool,
    /// 直近の replenish で測った ESS（診断用）
    last_ess: f64,
    /// systematic resampling の累計回数（診断用）
    resamples: u64,
    rng: StdRng,
}

impl Estimator {
    pub fn new(my_color: Color) -> Self {
        Estimator::with_seed(my_color, rand::rng().random())
    }

    pub fn with_seed(my_color: Color, seed: u64) -> Self {
        Estimator::with_seed_and_scale(my_color, seed, 1.0)
    }

    /// 思考予算スケールつきで作る（1.0 = 従来基準。strategy.rs の
    /// TSUITATE_THINK_BUDGET_MS から渡される）。粒子数・リプレイ回数・
    /// 時間打ち切りがスケールに比例する
    pub fn with_scale(my_color: Color, scale: f64) -> Self {
        Estimator::with_seed_and_scale(my_color, rand::rng().random(), scale)
    }

    pub fn with_seed_and_scale(my_color: Color, seed: u64, scale: f64) -> Self {
        let scale = scale.clamp(0.25, 8.0);
        let target = ((TARGET_PARTICLES as f64 * scale) as usize).clamp(128, 4096);
        Estimator {
            my_color,
            particles: vec![Position::initial(); target],
            info_miss: vec![0; target],
            logw: vec![0.0; target],
            target,
            regen_attempts: (REGEN_ATTEMPTS as f64 * scale) as usize,
            regen_deadline_ms: (500.0 * scale) as u64,
            empty_deadline_ms: (900.0 * scale) as u64,
            constraints: vec![],
            my_capture_idx: vec![],
            my_capture_sq: vec![],
            my_touched_idx: vec![],
            my_touched_sq: vec![],
            cursor: 0,
            healthy: true,
            last_ess: target as f64,
            resamples: 0,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// 粒子の目標数（思考予算に応じてスケール済み）
    pub fn target(&self) -> usize {
        self.target
    }

    pub fn my_color(&self) -> Color {
        self.my_color
    }

    /// 現在の粒子集合。空なら推定は信頼できない（呼び出し側でフォールバック）
    pub fn particles(&self) -> &[Position] {
        &self.particles
    }

    /// particles() と同じ並びのソフト救済回数。評価側の証拠数較正に使う
    /// （尤度の減衰は logw 側に課金済みなので、重みには二重に掛けない）
    pub fn info_miss(&self) -> &[u8] {
        &self.info_miss
    }

    /// 直近の replenish で測った ESS（診断用）
    pub fn last_ess(&self) -> f64 {
        self.last_ess
    }

    /// systematic resampling の累計回数（診断用）
    pub fn resamples(&self) -> u64 {
        self.resamples
    }

    /// particles() と同じ並びの観測尤度の対数重み。粒子間の相対値だけに意味が
    /// ある（評価側で max を引いて exp し正規化する）。複製粒子は同じ値を持つ
    pub fn log_weights(&self) -> &[f64] {
        &self.logw
    }

    pub fn healthy(&self) -> bool {
        self.healthy && !self.particles.is_empty()
    }

    /// ログの未消化イベントを取り込み、粒子を前進・棄却・補充する
    pub fn update(&mut self, log: &ObservationLog) {
        let events = log.events();
        while self.cursor < events.len() {
            let (constraint, consumed) = self.normalize(&events[self.cursor..]);
            self.cursor += consumed;
            let Some(constraint) = constraint else {
                continue;
            };
            self.apply_constraint(&constraint);
            if let Constraint::MyMove { mv, captured, .. } = &constraint {
                let idx = self.constraints.len();
                let to = match *mv {
                    ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
                };
                if captured.is_some() {
                    self.my_capture_idx.push(idx);
                    self.my_capture_sq.push(to);
                }
                if let ShogiMove::Board { from, .. } = *mv {
                    self.my_touched_idx.push(idx);
                    self.my_touched_sq.push(from);
                }
                self.my_touched_idx.push(idx);
                self.my_touched_sq.push(to);
            }
            self.constraints.push(constraint);
        }
        self.replenish();
    }

    /// 先頭イベントを制約へ正規化する。直後の Check イベントも一緒に消化する
    fn normalize(&self, events: &[Observation]) -> (Option<Constraint>, usize) {
        let head = &events[0];
        // 手の直後に王手宣言が続いているか（同じ着手の結果として扱う）
        let followed_by_check = |on: Color| -> bool {
            matches!(events.get(1), Some(Observation::Check { in_check }) if *in_check == on)
        };
        match head {
            Observation::MyMove { usi, captured, .. } => {
                let Some(mv) = parse_usi(usi) else {
                    return (None, 1);
                };
                let gives_check = followed_by_check(self.my_color.other());
                let consumed = if gives_check { 2 } else { 1 };
                (
                    Some(Constraint::MyMove {
                        mv,
                        captured: *captured,
                        gives_check,
                    }),
                    consumed,
                )
            }
            Observation::MyFoul { usi, .. } => match parse_usi(usi) {
                Some(mv) => (Some(Constraint::MyFoul { mv }), 1),
                None => (None, 1),
            },
            Observation::OpponentMoved {
                captured_my_piece_at,
                ..
            } => {
                let captured_at = captured_my_piece_at
                    .as_deref()
                    .and_then(crate::board::parse_usi_square);
                let gives_check = followed_by_check(self.my_color);
                let consumed = if gives_check { 2 } else { 1 };
                (
                    Some(Constraint::OppMove {
                        captured_at,
                        gives_check,
                    }),
                    consumed,
                )
            }
            // 相手の反則は「相手が何か非合法手を試みた」ことしか分からないので使わない。
            // 単独で現れた Check（手と紐づかない）は情報としては手側で消化済みのはず
            Observation::OpponentFoul { .. } | Observation::Check { .. } => (None, 1),
        }
    }

    fn apply_constraint(&mut self, constraint: &Constraint) {
        let my_color = self.my_color;
        let particles = std::mem::take(&mut self.particles);
        let penalties = std::mem::take(&mut self.info_miss);
        let logws = std::mem::take(&mut self.logw);
        let mut surv_pos = Vec::with_capacity(particles.len());
        let mut surv_pen = Vec::with_capacity(particles.len());
        let mut surv_logw = Vec::with_capacity(particles.len());
        // 棄却された粒子は適用前の局面を保持しておく（ソフト救済のやり直し用。
        // apply_my_move / sample_opp_move は失敗時も局面を汚しうる）
        let mut failed: Vec<(Position, u8, f64)> = vec![];
        // 厳密生存者が今回の制約で得た対数重み増分（ソフト救済の課金基準に使う）
        let mut strict_dls: Vec<f64> = vec![];
        for ((mut pos, pen), lw) in particles.into_iter().zip(penalties).zip(logws) {
            let backup = pos.clone();
            // 自分の手・反則は決定的（尤度 0/1）なので重みは変えない。
            // 相手手は観測クラスの尤度 r を対数重みへ累積する
            let ok = match constraint {
                Constraint::MyMove {
                    mv,
                    captured,
                    gives_check,
                } => apply_my_move(&mut pos, my_color, mv, *captured, Some(*gives_check))
                    .then_some(0.0),
                Constraint::MyFoul { mv } => {
                    foul_consistent(&pos, my_color, mv).then_some(0.0)
                }
                Constraint::OppMove {
                    captured_at,
                    gives_check,
                } => sample_opp_move(
                    &mut pos,
                    my_color,
                    *captured_at,
                    Some(*gives_check),
                    &self.my_capture_sq,
                    &self.my_touched_sq,
                    &mut self.rng,
                )
                .map(f64::ln),
            };
            if let Some(dlw) = ok {
                surv_pos.push(pos);
                surv_pen.push(pen);
                surv_logw.push(lw + dlw);
                strict_dls.push(dlw);
            } else {
                failed.push((backup, pen, lw));
            }
        }
        // ソフト救済: 厳密整合の生存が少ないときだけ、情報系の制約を緩和して
        // 棄却粒子を penalty+1 で生かす（枯渇からの回復を初期局面リプレイに
        // 頼らない = POMCP の particle reinvigoration に相当）
        if surv_pos.len() < self.target / 4 {
            // ソフト粒子の観測尤度: 本当は P(観測|粒子)=0 だが近似として生かす
            // ので、「典型的な厳密生存者と同じ増分」（中央値）を課す。緩和クラスの
            // r（≈1）をそのまま使うと、観測を説明できない粒子のほうが正直に
            // 小さい r を払った厳密粒子より重くなってしまう。厳密生存者がいない
            // ときだけ緩和クラスの r で代用する（全員ソフトなら相対値として無害で、
            // 後からリプレイされる厳密粒子は正直な累積 r を持つので比較もできる）
            let strict_dlw_median = (!strict_dls.is_empty()).then(|| {
                strict_dls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                strict_dls[strict_dls.len() / 2]
            });
            for (mut pos, pen, lw) in failed {
                if pen >= INFO_MISS_CAP {
                    continue;
                }
                if let Some(dlw) = self.apply_soft(&mut pos, constraint) {
                    surv_pos.push(pos);
                    surv_pen.push(pen + 1);
                    // 観測を説明できなかった近似粒子の課金: 典型的な厳密生存者と
                    // 同じ増分（中央値）に加えて、情報系ソフトの尤度 EPS_INFO を払う
                    // （旧: 評価側の soft_decay^penalty。C-7 P1 で logw へ統合）
                    surv_logw.push(lw + strict_dlw_median.unwrap_or(dlw) + EPS_INFO.ln());
                }
            }
        }
        self.particles = surv_pos;
        self.info_miss = surv_pen;
        self.logw = surv_logw;
    }

    /// 情報系の制約（王手宣言の一致・自分の反則の説明）だけを緩和した適用。
    /// 物理的な制約（自手の合法性・取った駒種・取られたマス）は緩和しない。
    /// 成功時は対数重みの増分（緩和クラスでの観測尤度）を返す
    fn apply_soft(&mut self, pos: &mut Position, constraint: &Constraint) -> Option<f64> {
        match constraint {
            Constraint::MyMove { mv, captured, .. } => {
                apply_my_move(pos, self.my_color, mv, *captured, None).then_some(0.0)
            }
            // 粒子上では合法だった手が実際は反則だった: この粒子は反則を
            // 説明できないが、盤面自体は生かす（反則手は実行されていない）
            Constraint::MyFoul { .. } => Some(0.0),
            Constraint::OppMove { captured_at, .. } => sample_opp_move(
                pos,
                self.my_color,
                *captured_at,
                None,
                &self.my_capture_sq,
                &self.my_touched_sq,
                &mut self.rng,
            )
            .map(f64::ln),
        }
    }

    /// 粒子が減っていたら、制約列のリプレイ（多様性）と生存粒子の複製（安価）で補充。
    /// 枯渇時は時間予算いっぱいまでリプレイで粘る（観測が正しい限り整合局面は必ず存在する）。
    /// リプレイ1回のコストは手数に比例するため、回数と時間の両方で打ち切る
    fn replenish(&mut self) {
        let start = std::time::Instant::now();
        let regen_deadline = start + std::time::Duration::from_millis(self.regen_deadline_ms);
        // リプレイの目標は「厳密整合の粒子数」。ソフト粒子で頭数が足りていても
        // 厳密粒子が薄ければリプレイで置き換えにいく（ソフトはあくまで近似）
        let mut strict = self.info_miss.iter().filter(|&&p| p == 0).count();
        if strict < self.target {
            for _ in 0..self.regen_attempts {
                if strict >= self.target || std::time::Instant::now() > regen_deadline {
                    break;
                }
                if let Some((pos, lw)) = self.replay_once() {
                    self.particles.push(pos);
                    self.info_miss.push(0);
                    self.logw.push(lw);
                    strict += 1;
                }
            }
        }
        let deadline = start + std::time::Duration::from_millis(self.empty_deadline_ms);
        while self.particles.is_empty() && std::time::Instant::now() < deadline {
            if let Some((pos, lw)) = self.replay_once() {
                self.particles.push(pos);
                self.info_miss.push(0);
                self.logw.push(lw);
            }
        }
        // ラッチしない: 粒子が戻れば健全に戻る（呼び出し側は毎手 update する）
        self.healthy = !self.particles.is_empty();
        if self.particles.is_empty() {
            return;
        }
        // 溢れの整理: info_miss 昇順（厳密優先）→ logw 降順で target まで絞る
        if self.particles.len() > self.target {
            let mut triples: Vec<(u8, f64, Position)> = std::mem::take(&mut self.info_miss)
                .into_iter()
                .zip(std::mem::take(&mut self.logw))
                .zip(std::mem::take(&mut self.particles))
                .map(|((pen, lw), pos)| (pen, lw, pos))
                .collect();
            triples.sort_by(|a, b| {
                a.0.cmp(&b.0)
                    .then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
            });
            triples.truncate(self.target);
            for (pen, lw, pos) in triples {
                self.info_miss.push(pen);
                self.logw.push(lw);
                self.particles.push(pos);
            }
        }
        // ESS 監視（C-7 P1 / D2）: 重みが退化していたら systematic resampling で
        // 質量を複製数へ実現し logw をリセットする。退化していないが頭数が
        // 足りないときは質量保存の分割複製で埋める（logw の相対値 = 評価側の
        // 重み付けを崩さずに、次の相手手サンプルで分岐する多様性の種を蒔く）
        let m = self.particles.len();
        let max_lw = self.logw.iter().copied().fold(f64::MIN, f64::max);
        let ws: Vec<f64> = self.logw.iter().map(|&lw| (lw - max_lw).exp()).collect();
        let total: f64 = ws.iter().sum();
        let sum2: f64 = ws.iter().map(|w| w * w).sum();
        self.last_ess = if sum2 > 0.0 { total * total / sum2 } else { 0.0 };
        if self.last_ess < m as f64 * ESS_THRESHOLD {
            self.systematic_resample(&ws, total);
            self.resamples += 1;
        } else if m < self.target {
            self.split_fill(&ws, total);
        }
    }

    /// systematic resampling: 正規化重み比例で target 個へ複製し logw を
    /// リセットする（質量が複製数へ実現される）。低分散・O(n)。
    /// info_miss は各コピーへ引き継ぐ（較正・上限管理はカウンタが担う）
    fn systematic_resample(&mut self, ws: &[f64], total: f64) {
        let m = self.particles.len();
        let want = self.target;
        let step = total / want as f64;
        let mut u = self.rng.random_range(0.0..step);
        let mut new_pos = Vec::with_capacity(want);
        let mut new_miss = Vec::with_capacity(want);
        let mut i = 0usize;
        let mut cum = ws[0];
        for _ in 0..want {
            while cum < u && i + 1 < m {
                i += 1;
                cum += ws[i];
            }
            new_pos.push(self.particles[i].clone());
            new_miss.push(self.info_miss[i]);
            u += step;
        }
        self.particles = new_pos;
        self.info_miss = new_miss;
        self.logw = vec![0.0; want];
    }

    /// 質量保存の分割複製: 重み比例で複製先を選び、同一個体群（元+コピー）で
    /// exp(logw) を等分する。指紋ごとの合計質量が変わらないため、評価側の
    /// multiplicity 畳み込みと二重に効かない（旧複製埋めの後継）
    fn split_fill(&mut self, ws: &[f64], total: f64) {
        let m = self.particles.len();
        let mut cum = Vec::with_capacity(m);
        let mut acc = 0.0f64;
        for &w in ws {
            acc += w;
            cum.push(acc);
        }
        let mut copies = vec![0usize; m];
        for _ in m..self.target {
            let t = self.rng.random_range(0.0..total);
            let i = cum.partition_point(|&c| c < t).min(m - 1);
            copies[i] += 1;
        }
        for (i, &c) in copies.iter().enumerate() {
            if c == 0 {
                continue;
            }
            let share = self.logw[i] - ((c + 1) as f64).ln();
            self.logw[i] = share;
            for _ in 0..c {
                self.particles.push(self.particles[i].clone());
                self.info_miss.push(self.info_miss[i]);
                self.logw.push(share);
            }
        }
    }

    /// 制約列を最初からリプレイして整合する粒子を1つ作る。
    ///
    /// 相手手のサンプルは確率的なので、後続の制約（自分の手の合法性・反則・
    /// 取られたマス・王手宣言）と矛盾して失敗しうる。全部やり直すと手数に対して
    /// 成功率が指数的に落ちるため、失敗したら直近の決定点（相手手）まで戻って
    /// 引き直す限定バックトラックにする。ステップ予算で最悪時間を抑える
    fn replay_once(&mut self) -> Option<(Position, f64)> {
        let n = self.constraints.len();
        let step_budget = n * 4 + 32;
        let mut steps = 0usize;
        let mut pos = Position::initial();
        let mut lw = 0.0f64;
        // 決定点スタック: (制約index, 適用前の局面, 適用前の対数重み, 再試行回数)
        let mut stack: Vec<(usize, Position, f64, u32)> = vec![];
        let mut i = 0;
        while i < n {
            steps += 1;
            if steps > step_budget {
                return None;
            }
            let ok = match &self.constraints[i] {
                Constraint::MyMove {
                    mv,
                    captured,
                    gives_check,
                } => apply_my_move(&mut pos, self.my_color, mv, *captured, Some(*gives_check)),
                Constraint::MyFoul { mv } => foul_consistent(&pos, self.my_color, mv),
                Constraint::OppMove {
                    captured_at,
                    gives_check,
                } => {
                    // バックトラックで戻ってきた再訪なら積み直さない
                    let is_retry = stack.last().is_some_and(|(j, _, _, _)| *j == i);
                    if !is_retry {
                        stack.push((i, pos.clone(), lw, 0));
                    }
                    // この時点までに自分が駒を取ったマス／触れたマス
                    let k = self.my_capture_idx.partition_point(|&j| j < i);
                    let t = self.my_touched_idx.partition_point(|&j| j < i);
                    match sample_opp_move(
                        &mut pos,
                        self.my_color,
                        *captured_at,
                        Some(*gives_check),
                        &self.my_capture_sq[..k],
                        &self.my_touched_sq[..t],
                        &mut self.rng,
                    ) {
                        Some(r) => {
                            lw += r.ln();
                            true
                        }
                        None => false,
                    }
                }
            };
            if ok {
                i += 1;
                continue;
            }
            // 失敗: 直近の決定点に戻って引き直す。試行を使い切った点はさらに前へ
            loop {
                let Some((j, snapshot, snapshot_lw, attempts)) = stack.pop() else {
                    return None;
                };
                // 失敗した制約自身が決定点なら、同じ局面からの再試行は無意味
                // （整合候補ゼロは決定的）なのでさらに前へ戻る
                if j == i {
                    continue;
                }
                if attempts + 1 < BACKTRACK_ATTEMPTS {
                    pos = snapshot.clone();
                    lw = snapshot_lw;
                    stack.push((j, snapshot, snapshot_lw, attempts + 1));
                    i = j;
                    break;
                }
            }
        }
        Some((pos, lw))
    }
}

/// 受理された自分の手を粒子に適用する。粒子と観測が矛盾したら false。
/// gives_check が None のときは王手宣言との一致を検査しない（ソフト救済用）
fn apply_my_move(
    pos: &mut Position,
    my_color: Color,
    mv: &ShogiMove,
    captured: Option<Role>,
    gives_check: Option<bool>,
) -> bool {
    if pos.turn() != my_color || !pos.is_legal(mv) {
        return false;
    }
    let actual = pos.play_unchecked(mv).map(unpromote_role);
    if actual != captured {
        return false;
    }
    gives_check.is_none_or(|gc| pos.in_check(my_color.other()) == gc)
}

/// 反則になった手との整合: 粒子上でも非合法であること
fn foul_consistent(pos: &Position, my_color: Color, mv: &ShogiMove) -> bool {
    pos.turn() == my_color && !pos.is_legal(mv)
}

/// 動かした駒（着地点）が対象マスのどれかへ新たに利きを付けたか。
/// 「新たに」= 移動元からは利いていなかった（打ちは常に新規）。
/// **定義は bin/fit_opp の newly_threatens と一致させること**（学習と推論の整合）
fn newly_threatens(pos: &Position, next: &Position, mv: &ShogiMove, targets: &[Coord]) -> bool {
    let to = match *mv {
        ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
    };
    targets.iter().any(|&s| {
        if s == to || !next.attacks(to, s) {
            return false;
        }
        match *mv {
            ShogiMove::Board { from, .. } => !pos.attacks(from, s),
            ShogiMove::Drop { .. } => true,
        }
    })
}

/// 観測と整合する相手の合法手をサンプルして適用する。整合手がなければ None。
/// 成功時は観測尤度 r = 整合クラスの事前質量 / 全合法手の事前質量（0<r≤1）を
/// 返す（SIR の重み更新。呼び出し側が対数で累積する）。
/// - gives_check: None なら王手宣言との一致を検査しない（ソフト救済用）
/// - known_squares: 自分が駒を取ったマス（相手は自駒がそこで死んだことを知っている）
/// - my_touched: 自分の手が触れたマス（初期配置のまま動いていない自駒の判定用。
///   相手はそれらを推論で狙ってくる = 飛車頭への歩打ち等）
fn sample_opp_move(
    pos: &mut Position,
    my_color: Color,
    captured_at: Option<Coord>,
    gives_check: Option<bool>,
    known_squares: &[Coord],
    my_touched: &[Coord],
    rng: &mut StdRng,
) -> Option<f64> {
    let opp = my_color.other();
    if pos.turn() != opp {
        return None;
    }
    // 初期配置から動いていない自駒のマス（粒子内の実配置と突き合わせる）
    let initial = Position::initial();
    let homes: Vec<Coord> = initial
        .pieces()
        .filter(|(sq, p)| {
            p.color == my_color
                && !my_touched.contains(sq)
                && pos
                    .piece_at(*sq)
                    .is_some_and(|cur| cur.color == my_color && cur.role == p.role)
        })
        .map(|(sq, _)| sq)
        .collect();

    let mut candidates: Vec<(ShogiMove, f64)> = vec![];
    let mut total_mass = 0.0f64;
    for mv in pos.legal_moves() {
        // 取られたマスとの整合（取りがなかったなら自駒のあるマスへは来ていない）
        let to_capture = match mv {
            ShogiMove::Board { to, .. } => pos
                .piece_at(to)
                .filter(|p| p.color == my_color)
                .map(|p| (to, p.role)),
            ShogiMove::Drop { .. } => None,
        };
        let capture_ok = match (captured_at, to_capture) {
            (Some(at), Some((to, _))) => at == to,
            (None, None) => true,
            _ => false,
        };
        let mut next = pos.clone();
        next.play_unchecked(&mv);
        // 分母（total_mass）には全合法手の重みが要るが、王手判定はクラス判定に
        // しか使わないので capture_ok の短絡で省く（in_check は比較的重い）
        let consistent =
            capture_ok && gives_check.is_none_or(|gc| next.in_check(my_color) == gc);
        let threat_known = newly_threatens(pos, &next, &mv, known_squares);
        let threat_home = newly_threatens(pos, &next, &mv, &homes);
        let (is_king, flee) = match mv {
            ShogiMove::Board { from, to, .. } => {
                let is_king = pos.piece_at(from).is_some_and(|p| p.role == Role::King);
                (is_king, is_king && flees_danger(from, to, known_squares))
            }
            ShogiMove::Drop { .. } => (false, false),
        };
        let w = opp_move_weight(
            opp,
            &mv,
            threat_known,
            threat_home,
            is_king,
            flee,
            moved_is_minor(pos, &mv),
            deep_unsupported(&next, &mv, opp),
            hangs_on_landing(pos, &next, &mv, opp),
        );
        total_mass += w;
        if consistent {
            candidates.push((mv, w));
        }
    }
    let chosen = weighted_choice(&candidates, rng)?;
    let class_mass: f64 = candidates.iter().map(|(_, w)| w).sum();
    pos.play_unchecked(&chosen);
    // weighted_choice が成功した時点で class_mass > 0、total_mass ≥ class_mass
    Some((class_mass / total_mass).min(1.0))
}

/// 露見マス（自分が駒を取った=相手に通知されたマス）での取り返しブースト。
/// 事前分布のフィットでは駒取りは観測条件で絞られるため学習されていない。
/// 対人実戦では露見駒の回収はほぼ必ず実行されるので予測では強く優先する
const PREDICT_RECAPTURE_BOOST: f64 = 8.0;

/// 相手の応手を事前分布モデルで1手サンプルする（2手読み用の予測）。
/// sample_opp_move と同じ尤度モデルだが、これから指される手の予測なので
/// 観測（取られたマス・王手宣言）による絞り込みは行わない。
/// known_squares / my_touched の意味は sample_opp_move と同じ
pub fn predict_opp_reply<R: Rng>(
    pos: &Position,
    my_color: Color,
    known_squares: &[Coord],
    my_touched: &[Coord],
    rng: &mut R,
) -> Option<ShogiMove> {
    weighted_choice(
        &opp_reply_weights(pos, my_color, known_squares, my_touched),
        rng,
    )
}

/// 相手の全合法応手と方策重み（事前分布モデル＋露見マスの取り返しブースト）。
/// 2手読みの期待値評価用: サンプルせず重み付き平均を取れる
pub fn opp_reply_weights(
    pos: &Position,
    my_color: Color,
    known_squares: &[Coord],
    my_touched: &[Coord],
) -> Vec<(ShogiMove, f64)> {
    let opp = my_color.other();
    if pos.turn() != opp {
        return vec![];
    }
    let initial = Position::initial();
    let homes: Vec<Coord> = initial
        .pieces()
        .filter(|(sq, p)| {
            p.color == my_color
                && !my_touched.contains(sq)
                && pos
                    .piece_at(*sq)
                    .is_some_and(|cur| cur.color == my_color && cur.role == p.role)
        })
        .map(|(sq, _)| sq)
        .collect();
    let mut candidates: Vec<(ShogiMove, f64)> = vec![];
    for mv in pos.legal_moves() {
        let mut next = pos.clone();
        next.play_unchecked(&mv);
        let threat_known = newly_threatens(pos, &next, &mv, known_squares);
        let threat_home = newly_threatens(pos, &next, &mv, &homes);
        let (is_king, flee) = match mv {
            ShogiMove::Board { from, to, .. } => {
                let is_king = pos.piece_at(from).is_some_and(|p| p.role == Role::King);
                (is_king, is_king && flees_danger(from, to, known_squares))
            }
            ShogiMove::Drop { .. } => (false, false),
        };
        let mut w = opp_move_weight(
            opp,
            &mv,
            threat_known,
            threat_home,
            is_king,
            flee,
            moved_is_minor(pos, &mv),
            deep_unsupported(&next, &mv, opp),
            hangs_on_landing(pos, &next, &mv, opp),
        );
        if let ShogiMove::Board { to, .. } = mv {
            let captures_mine = pos.piece_at(to).is_some_and(|p| p.color == my_color);
            if captures_mine && known_squares.contains(&to) {
                w *= PREDICT_RECAPTURE_BOOST;
            }
        }
        candidates.push((mv, w));
    }
    candidates
}

/// 動かす駒種（移動前の役）が歩・香・桂の小駒か。
/// **定義は bin/fit_opp の moved_is_minor と一致させること**
fn moved_is_minor(pos: &Position, mv: &ShogiMove) -> bool {
    let role = match *mv {
        ShogiMove::Board { from, .. } => pos.piece_at(from).map(|p| p.role),
        ShogiMove::Drop { role, .. } => Some(role),
    };
    matches!(role, Some(Role::Pawn | Role::Lance | Role::Knight))
}

/// 相手の利きがあるマスへの紐なし着地か（取りは除く = 交換ではなく差し出し）。
/// 利き・紐とも着地後の盤面（next）で判定する（開き駒の利きを含む）。
/// 相手の玉の利きも数える（紐がなければ玉に取られる）。銀以上の駒での該当は
/// 実質タダの駒捨てで人間はほぼ指さない（馬@62 のような幻の飛び込み王手の
/// 過大評価を抑える）。**定義は bin/fit_opp の hangs_on_landing と一致させること**
fn hangs_on_landing(pos: &Position, next: &Position, mv: &ShogiMove, mover: Color) -> bool {
    let to = match *mv {
        ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
    };
    if pos.piece_at(to).is_some() {
        return false; // 取り（交換の文脈）は対象外
    }
    let opp = mover.other();
    let attacked = next
        .pieces()
        .any(|(sq, p)| p.color == opp && next.attacks(sq, to));
    attacked
        && !next
            .pieces()
            .any(|(sq, p)| p.color == mover && sq != to && next.attacks(sq, to))
}

/// 敵陣（成れる3段）への紐なし着地か。着地点に自分の別の駒の利きが無い。
/// **定義は bin/fit_opp の deep_unsupported と一致させること**
fn deep_unsupported(next: &Position, mv: &ShogiMove, mover: Color) -> bool {
    let to = match *mv {
        ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
    };
    let deep = match mover {
        Color::Sente => to.rank <= 3,
        Color::Gote => to.rank >= 7,
    };
    deep && !next
        .pieces()
        .any(|(sq, p)| p.color == mover && sq != to && next.attacks(sq, to))
}

/// チェビシェフ距離（玉の歩数）
fn dist(a: Coord, b: Coord) -> i8 {
    (a.file - b.file).abs().max((a.rank - b.rank).abs())
}

/// 玉の移動が危険地点集合（自分が駒を取ったマス = 相手にとっての露見地点）から
/// 遠ざかる手か。**定義は bin/fit_opp の flees_danger と一致させること**
fn flees_danger(from: Coord, to: Coord, danger: &[Coord]) -> bool {
    let near = |sq: Coord| danger.iter().map(|&d| dist(sq, d)).min();
    match (near(from), near(to)) {
        (Some(a), Some(b)) => b > a,
        _ => false,
    }
}

/// 相手の手の尤度づけ。対人59局の条件付き最尤推定（bin/fit_opp, 2026-07-17、
/// 成り・敵陣深入り・ハングの駒種分割）: パープレキシティ 28.2（旧手調整）→ 24.2。
/// 駒取り・王手の有無は観測との整合ですでに絞り込まれているため、
/// 事前分布には「観測クラス内で判別できる特徴量」だけが現れる。
/// king_flee がわずかに負なのは実測（守りを剥がされても玉は特に逃げない）。
/// 成り・深入り・ハングは小駒（歩香桂）と銀以上で分割: 垂れ歩・と金作りは
/// 好んで指されるが、大駒を相手の利きに紐なしで差し出す手（hang_major）は
/// 実質駒捨てで明確に避けられる（候補10.3%に対し選択3.6%）
fn opp_move_weight(
    opp: Color,
    mv: &ShogiMove,
    threat_known: bool,
    threat_home: bool,
    is_king_move: bool,
    king_flee: bool,
    moved_minor: bool,
    deep_unsup: bool,
    hang: bool,
) -> f64 {
    let mut s = 0.0;
    match *mv {
        ShogiMove::Board { from, to, promote } => {
            let advance = match opp {
                Color::Sente => (from.rank - to.rank) as f64,
                Color::Gote => (to.rank - from.rank) as f64,
            };
            s += 0.162 * advance;
            if promote {
                s += if moved_minor { 1.647 } else { 0.659 };
            }
        }
        ShogiMove::Drop { .. } => s += -1.451,
    }
    if threat_known {
        s += 0.561;
    }
    if threat_home {
        s += 0.670;
    }
    if is_king_move {
        s += 0.131;
    }
    if king_flee {
        s += -0.161;
    }
    if deep_unsup {
        s += if moved_minor { 0.320 } else { 0.026 };
    }
    if hang {
        s += if moved_minor { 0.433 } else { -0.839 };
    }
    s.exp()
}

fn weighted_choice<R: Rng>(candidates: &[(ShogiMove, f64)], rng: &mut R) -> Option<ShogiMove> {
    let total: f64 = candidates.iter().map(|(_, w)| w).sum();
    if candidates.is_empty() || total <= 0.0 {
        return None;
    }
    let mut t = rng.random_range(0.0..total);
    for (mv, w) in candidates {
        t -= w;
        if t <= 0.0 {
            return Some(*mv);
        }
    }
    candidates.last().map(|(mv, _)| *mv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Role;

    fn record_my_move(log: &mut ObservationLog, usi: &str, captured: Option<Role>) {
        log.record(Observation::MyMove {
            move_number: 0,
            usi: usi.into(),
            captured,
        });
    }

    fn record_opp_move(log: &mut ObservationLog, captured_at: Option<&str>) {
        log.record(Observation::OpponentMoved {
            move_number: 0,
            captured_my_piece_at: captured_at.map(String::from),
        });
    }

    #[test]
    fn particles_track_own_moves_exactly() {
        let mut est = Estimator::with_seed(Color::Sente, 42);
        let mut log = ObservationLog::default();
        record_my_move(&mut log, "7g7f", None);
        record_opp_move(&mut log, None);
        est.update(&log);
        assert!(est.healthy());
        assert_eq!(est.particles().len(), TARGET_PARTICLES);
        for pos in est.particles() {
            // 自分側は全粒子で正確
            assert_eq!(
                pos.piece_at(Coord { file: 7, rank: 6 }).map(|p| p.role),
                Some(Role::Pawn)
            );
            // 相手は20枚のまま（駒は取られていない）
            assert_eq!(pos.pieces_of(Color::Gote).len(), 20);
            assert_eq!(pos.turn(), Color::Sente);
        }
    }

    #[test]
    fn foul_reveals_blocking_piece() {
        // 初手 8h2b+（角道が開いていない）はどの粒子でも非合法…ではなく
        // 実戦なら反則観測により「経路に何かある」情報が得られる形をテストする。
        // 7g7f / 相手手 / 8h2b+ が反則 → 相手の角道（7c〜3g のどこか）に駒がある粒子だけが残る
        let mut est = Estimator::with_seed(Color::Sente, 7);
        let mut log = ObservationLog::default();
        record_my_move(&mut log, "7g7f", None);
        record_opp_move(&mut log, None);
        log.record(Observation::MyFoul {
            move_number: 0,
            usi: "8h2b+".into(),
        });
        est.update(&log);
        assert!(est.healthy());
        for pos in est.particles() {
            // 8h から 2b への斜線上（7g〜3c）のどこかに駒がある（=非合法の理由）。
            // 経路が通っていれば 2b への移動/駒取りは合法なので、その粒子は棄却されている
            let blocked = (3..=7).any(|i| {
                pos.piece_at(Coord { file: i, rank: i }).is_some()
            });
            assert!(blocked, "反則の説明がつかない粒子が残っている");
        }
    }

    #[test]
    fn capture_observation_pins_down_opponent_piece() {
        // 7g7f → 相手手 → 8h2b+ が受理され bishop を取った
        // → どの粒子でも「2b に角がいた」ことになり、相手の持ち駒推定も一致する
        let mut est = Estimator::with_seed(Color::Sente, 11);
        let mut log = ObservationLog::default();
        record_my_move(&mut log, "7g7f", None);
        record_opp_move(&mut log, None);
        record_my_move(&mut log, "8h2b+", Some(Role::Bishop));
        est.update(&log);
        assert!(est.healthy());
        assert!(!est.particles().is_empty());
        for pos in est.particles() {
            assert_eq!(
                pos.piece_at(Coord { file: 2, rank: 2 }).map(|p| p.role),
                Some(Role::Horse), // 自分の馬がいる
            );
            // 相手の盤上駒は19枚（角を取られた）
            assert_eq!(pos.pieces_of(Color::Gote).len(), 19);
        }
    }

    #[test]
    fn check_declaration_filters_particles() {
        // 7g7f → 相手手 → 8h3c+（3cの歩を取って馬に）。馬が 4b 越しに 5a の玉を
        // 睨むため、王手宣言があった場合は「4b が空いている」粒子だけが残る
        let mut est = Estimator::with_seed(Color::Sente, 13);
        let mut log = ObservationLog::default();
        record_my_move(&mut log, "7g7f", None);
        record_opp_move(&mut log, None);
        record_my_move(&mut log, "8h3c+", Some(Role::Pawn));
        log.record(Observation::Check {
            in_check: Color::Gote,
        });
        est.update(&log);
        assert!(est.healthy(), "王手と整合する粒子が残るはず");
        for pos in est.particles() {
            assert!(pos.in_check(Color::Gote));
        }
    }

    #[test]
    fn replay_backtracking_still_satisfies_all_constraints() {
        // 王手宣言つきの長め制約列でも、バックトラックで作った粒子が
        // 全制約と整合していること（check_declaration と同じ設定で枯渇→再生成）
        let mut est = Estimator::with_seed(Color::Sente, 23);
        let mut log = ObservationLog::default();
        record_my_move(&mut log, "7g7f", None);
        record_opp_move(&mut log, None);
        record_my_move(&mut log, "8h3c+", Some(Role::Pawn));
        log.record(Observation::Check {
            in_check: Color::Gote,
        });
        record_opp_move(&mut log, None);
        // 相手が金の合駒（4a4b）をした粒子だけが次の制約と整合する。
        // 4b の金を取ると馬が 5a の玉に再度王手になる
        record_my_move(&mut log, "3c4b", Some(Role::Gold));
        log.record(Observation::Check {
            in_check: Color::Gote,
        });
        est.update(&log);
        est.particles.clear();
        est.info_miss.clear();
        est.logw.clear();
        est.replenish();
        assert!(est.healthy(), "バックトラック付きリプレイで再生成できるはず");
        for pos in est.particles() {
            // 最終制約まで適用済み: 4b に自分の馬、相手の盤上駒は18枚（歩と金を取った）
            assert_eq!(
                pos.piece_at(Coord { file: 4, rank: 2 }).map(|p| p.role),
                Some(crate::protocol::Role::Horse)
            );
            assert_eq!(pos.pieces_of(Color::Gote).len(), 18);
        }
    }

    #[test]
    fn depleted_particles_regenerate_by_replay() {
        let mut est = Estimator::with_seed(Color::Sente, 17);
        let mut log = ObservationLog::default();
        record_my_move(&mut log, "7g7f", None);
        record_opp_move(&mut log, None);
        est.update(&log);
        // 人工的に枯渇させる
        est.particles.clear();
        est.info_miss.clear();
        est.logw.clear();
        est.replenish();
        assert!(est.healthy(), "リプレイで再生成できるはず");
        assert_eq!(est.particles().len(), TARGET_PARTICLES);
    }

    #[test]
    fn predict_opp_reply_returns_legal_move() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut pos = Position::initial();
        pos.play_unchecked(&parse_usi("7g7f").unwrap());
        let reply = predict_opp_reply(&pos, Color::Sente, &[], &[], &mut rng)
            .expect("初期局面の相手に応手がないはずはない");
        assert!(pos.is_legal(&reply));
        // 手番が自分側の局面では予測しない
        let initial = Position::initial();
        assert!(predict_opp_reply(&initial, Color::Sente, &[], &[], &mut rng).is_none());
    }

    #[test]
    fn reply_weights_apply_recapture_boost_deterministically() {
        // 7g7f / 3c3d / 8h2b+ の後、3a2b（取り返し）の重みは
        // 2b が既知地点のときだけ PREDICT_RECAPTURE_BOOST 倍される
        let mut pos = Position::initial();
        for usi in ["7g7f", "3c3d", "8h2b+"] {
            pos.play_unchecked(&parse_usi(usi).unwrap());
        }
        let recapture = parse_usi("3a2b").unwrap();
        let weight_of = |known: &[Coord]| -> f64 {
            opp_reply_weights(&pos, Color::Sente, known, &[])
                .iter()
                .find(|(mv, _)| *mv == recapture)
                .map(|(_, w)| *w)
                .expect("取り返しは合法応手のはず")
        };
        let with_boost = weight_of(&[Coord { file: 2, rank: 2 }]);
        let without = weight_of(&[]);
        assert!(
            (with_boost / without - PREDICT_RECAPTURE_BOOST).abs() < 1e-6,
            "with={with_boost} without={without}"
        );
    }

    #[test]
    fn recapture_boost_requires_known_square() {
        // 7g7f / 3c3d / 8h2b+（角で2bの角を取って馬に）。手番は後手で、
        // 3a銀による 2b の取り返しが合法。2b が既知地点なら取り返しが
        // 強くブーストされ、既知でなければ他の手と同程度の頻度に留まる
        let mut pos = Position::initial();
        for usi in ["7g7f", "3c3d", "8h2b+"] {
            pos.play_unchecked(&parse_usi(usi).unwrap());
        }
        let recapture = parse_usi("3a2b").unwrap();
        let freq = |known: &[Coord]| -> f64 {
            let mut rng = StdRng::seed_from_u64(99);
            let n = 400;
            let mut hits = 0;
            for _ in 0..n {
                if predict_opp_reply(&pos, Color::Sente, known, &[], &mut rng)
                    == Some(recapture)
                {
                    hits += 1;
                }
            }
            f64::from(hits) / f64::from(n)
        };
        let with_boost = freq(&[Coord { file: 2, rank: 2 }]);
        let without = freq(&[]);
        assert!(
            with_boost > without * 3.0,
            "既知地点の取り返しはブーストされるはず（with={with_boost:.3} without={without:.3}）"
        );
        assert!(with_boost > 0.10, "with={with_boost:.3}");
    }

    #[test]
    fn strict_survivors_keep_zero_penalty() {
        let mut est = Estimator::with_seed(Color::Sente, 42);
        let mut log = ObservationLog::default();
        record_my_move(&mut log, "7g7f", None);
        record_opp_move(&mut log, None);
        est.update(&log);
        assert!(est.info_miss().iter().all(|&p| p == 0));
        assert_eq!(est.particles().len(), est.info_miss().len());
        assert_eq!(est.particles().len(), est.log_weights().len());
    }

    #[test]
    fn soft_pass_rescues_check_declaration_mismatch() {
        // 初手 7g7f が王手になる粒子は存在しない → 厳密整合は全滅するが、
        // ソフト救済が王手宣言の一致を緩和して penalty=1 で全粒子を生かす
        let mut est = Estimator::with_seed(Color::Sente, 5);
        let c = Constraint::MyMove {
            mv: parse_usi("7g7f").unwrap(),
            captured: None,
            gives_check: true,
        };
        est.apply_constraint(&c);
        assert_eq!(est.particles.len(), TARGET_PARTICLES);
        assert!(est.info_miss.iter().all(|&p| p == 1));
        // 情報系ソフトの尤度 EPS_INFO が logw へ課金されている
        // （厳密生存者ゼロなので中央値課金はなく ln(EPS_INFO) のみ）
        assert!(
            est.logw.iter().all(|&lw| (lw - EPS_INFO.ln()).abs() < 1e-9),
            "ソフト救済の課金が logw に乗っていない"
        );
        // 物理的な適用（着手そのもの）は行われている
        for pos in est.particles() {
            assert_eq!(
                pos.piece_at(Coord { file: 7, rank: 6 }).map(|p| p.role),
                Some(Role::Pawn)
            );
        }
    }

    #[test]
    fn soft_pass_does_not_relax_physical_constraints() {
        // 初手で 5e の駒を取ることはどの粒子でも物理的に不可能
        // （5e への合法手自体がない）→ ソフト救済でも救えず全滅する
        let mut est = Estimator::with_seed(Color::Sente, 5);
        let c = Constraint::MyMove {
            mv: parse_usi("5g5e").unwrap(),
            captured: Some(Role::Pawn),
            gives_check: false,
        };
        est.apply_constraint(&c);
        assert!(est.particles.is_empty());
    }

    #[test]
    fn penalty_cap_culls_repeated_violators() {
        let mut est = Estimator::with_seed(Color::Sente, 5);
        for p in est.info_miss.iter_mut() {
            *p = INFO_MISS_CAP;
        }
        let c = Constraint::MyMove {
            mv: parse_usi("7g7f").unwrap(),
            captured: None,
            gives_check: true,
        };
        est.apply_constraint(&c);
        assert!(est.particles.is_empty(), "上限到達の粒子は救済されない");
    }

    /// 2種の局面を粒子に仕込むヘルパ（初期局面と 3c3d 後の局面は別指紋）
    fn two_kind_particles(est: &mut Estimator, n_a: usize, n_b: usize) -> (Position, Position) {
        let a = Position::initial();
        let mut b = Position::initial();
        b.play_unchecked(&parse_usi("7g7f").unwrap());
        b.play_unchecked(&parse_usi("3c3d").unwrap());
        est.particles.clear();
        est.info_miss.clear();
        est.logw.clear();
        for _ in 0..n_a {
            est.particles.push(a.clone());
            est.info_miss.push(0);
            est.logw.push(0.0);
        }
        for _ in 0..n_b {
            est.particles.push(b.clone());
            est.info_miss.push(0);
            est.logw.push(0.0);
        }
        (a, b)
    }

    #[test]
    fn ess_degeneracy_triggers_systematic_resample() {
        // 重みが1粒子へ退化した集合: ESS ≈ 1 → リサンプリングが発動し、
        // logw リセット・target への複製・質量は複製数へ実現される
        let mut est = Estimator::with_seed(Color::Sente, 31);
        let n = est.target;
        let (a, b) = two_kind_particles(&mut est, 1, n - 1);
        est.logw[0] = 0.0; // a 粒子が支配的
        for lw in est.logw.iter_mut().skip(1) {
            *lw = -20.0;
        }
        est.replenish();
        assert_eq!(est.resamples(), 1, "ESS退化でリサンプリングされるはず");
        assert_eq!(est.particles().len(), est.target());
        assert!(est.logw.iter().all(|&lw| lw == 0.0), "リサンプリング後は logw=0");
        let n_a = est
            .particles()
            .iter()
            .filter(|p| p.fingerprint() == a.fingerprint())
            .count();
        let n_b = est
            .particles()
            .iter()
            .filter(|p| p.fingerprint() == b.fingerprint())
            .count();
        assert!(
            n_a > est.target() * 9 / 10,
            "支配的粒子の質量が複製数に実現されていない: a={n_a} b={n_b}"
        );
    }

    #[test]
    fn split_fill_preserves_mass_per_fingerprint() {
        // 退化していない不足（2個体、重み比 1 : e^-1）は分割複製で埋まり、
        // 指紋ごとの合計質量 exp(logw) が保存される（multiplicity 畳み込みと
        // 二重に効かないための不変条件）。replenish のリプレイ充填と切り離すため
        // split_fill を直接呼ぶ
        let mut est = Estimator::with_seed(Color::Sente, 37);
        let (a, b) = two_kind_particles(&mut est, 1, 1);
        est.logw[1] = -1.0;
        let ws: Vec<f64> = est.logw.iter().map(|&lw| lw.exp()).collect();
        let total: f64 = ws.iter().sum();
        est.split_fill(&ws, total);
        assert_eq!(est.particles.len(), est.target());
        let mass_of = |est: &Estimator, fp: u64| -> f64 {
            est.particles
                .iter()
                .zip(&est.logw)
                .filter(|(p, _)| p.fingerprint() == fp)
                .map(|(_, &lw)| lw.exp())
                .sum()
        };
        let mass_a = mass_of(&est, a.fingerprint());
        let mass_b = mass_of(&est, b.fingerprint());
        assert!((mass_a - 1.0).abs() < 1e-9, "a の質量が保存されていない: {mass_a}");
        assert!(
            (mass_b - (-1.0f64).exp()).abs() < 1e-9,
            "b の質量が保存されていない: {mass_b}"
        );
    }

    #[test]
    fn resample_keeps_info_miss_counter() {
        // リサンプリングは logw をリセットするが info_miss は引き継ぐ
        // （嘘の昇格防止。較正・上限管理はカウンタが担う）
        let mut est = Estimator::with_seed(Color::Sente, 41);
        let n = est.target;
        two_kind_particles(&mut est, 1, n - 1);
        est.info_miss[0] = 2;
        est.logw[0] = 0.0;
        for lw in est.logw.iter_mut().skip(1) {
            *lw = -20.0;
        }
        let max_lw = est.logw.iter().copied().fold(f64::MIN, f64::max);
        let ws: Vec<f64> = est.logw.iter().map(|&lw| (lw - max_lw).exp()).collect();
        let total: f64 = ws.iter().sum();
        est.systematic_resample(&ws, total);
        assert_eq!(est.particles.len(), est.target());
        assert!(est.logw.iter().all(|&lw| lw == 0.0));
        assert!(
            est.info_miss.iter().filter(|&&m| m == 2).count() > est.target() * 9 / 10,
            "リサンプリングのコピーが info_miss を引き継いでいない"
        );
    }
}
