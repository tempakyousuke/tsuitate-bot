//! estimator の凍結版 v6（2026-07-14 凍結）。
//!
//! v5 からの差分:
//! - ソフト粒子（POMCP流reinvigoration）: 厳密整合の生存が target/4 未満のとき
//!   情報系制約（王手宣言・反則の説明）だけ緩和して penalty+1 で救済、
//!   評価は重み soft_decay^penalty。41手以降の推定生存率 13%→31%
//! - 2手読み: 上位候補だけ相手応手を事前分布からサンプルし、静的リスク項の
//!   depth2_replace 分を実測の期待損失に置き換える（gain再構築方式）。
//!   候補手の捕獲マスは応手予測の既知地点に加える（取り返しブースト）
//! - 交換価値 exchange_value =（盤上価値+持ち駒価値）/2（と金の反動是正）
//! - 利き被覆・と金プローブ・王探し情報利得・アンチドロー（endgame_push）
//! - 思考予算スケール（TSUITATE_THINK_BUDGET_MS、既定2000ms。凍結版も同じ
//!   環境変数を読むので「同一計算量での比較」が保たれる）
//! - 評価パラメータはSPSA第2ラウンドの収束点（共通乱数法・60反復×2×40局
//!   vs v5、tuning/tune-round2.jsonl、最終中心点の追加評価 score=0.675）
//! - シード注入（with_seed）: SPSA/アブレーションの共通乱数法用
//!
//! 凍結時の成績（200局×4基準・シャード並列・seed 20260714）:
//! vs v5 66.3%±7.1% / vs v4 78.2%±6.1% / vs v3 74.1%±6.7% / vs v2 83.5%±5.6%
//!
//! 凍結後は編集しない（シード注入等の挙動を変えない追加のみ許容）。

use std::collections::{HashMap, HashSet};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::board::{
    Coord, Promotion, drop_targets, make_usi_drop, make_usi_move, make_usi_square, move_targets,
    parse_usi_square, promotion_choice,
};
use crate::model::GameModel;
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{Color, PlayerView, Role, VisiblePiece};
use crate::shogi::{Position, ShogiMove, parse_usi, piece_value, promote_role, unpromote_role};
use crate::strategy::Strategy;

// ---------------------------------------------------------------------------
// 推定器（estimator.rs のコピー）
// ---------------------------------------------------------------------------

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
const PENALTY_CAP: u8 = 3;

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
    /// particles と同じ並びのソフト救済回数（0 = 全制約と厳密整合）
    penalties: Vec<u8>,
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
            penalties: vec![0; target],
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

    /// 現在の粒子集合。空なら推定は信頼できない（呼び出し側でフォールバック）。
    /// replenish 後は penalty 昇順（厳密整合が先頭側）に並んでいる
    pub fn particles(&self) -> &[Position] {
        &self.particles
    }

    /// particles() と同じ並びのソフト救済回数。評価側の重み付けに使う
    pub fn penalties(&self) -> &[u8] {
        &self.penalties
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
        let penalties = std::mem::take(&mut self.penalties);
        let mut surv_pos = Vec::with_capacity(particles.len());
        let mut surv_pen = Vec::with_capacity(particles.len());
        // 棄却された粒子は適用前の局面を保持しておく（ソフト救済のやり直し用。
        // apply_my_move / sample_opp_move は失敗時も局面を汚しうる）
        let mut failed: Vec<(Position, u8)> = vec![];
        for (mut pos, pen) in particles.into_iter().zip(penalties) {
            let backup = pos.clone();
            let ok = match constraint {
                Constraint::MyMove {
                    mv,
                    captured,
                    gives_check,
                } => apply_my_move(&mut pos, my_color, mv, *captured, Some(*gives_check)),
                Constraint::MyFoul { mv } => foul_consistent(&pos, my_color, mv),
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
                ),
            };
            if ok {
                surv_pos.push(pos);
                surv_pen.push(pen);
            } else {
                failed.push((backup, pen));
            }
        }
        // ソフト救済: 厳密整合の生存が少ないときだけ、情報系の制約を緩和して
        // 棄却粒子を penalty+1 で生かす（枯渇からの回復を初期局面リプレイに
        // 頼らない = POMCP の particle reinvigoration に相当）
        if surv_pos.len() < self.target / 4 {
            for (mut pos, pen) in failed {
                if pen >= PENALTY_CAP {
                    continue;
                }
                if self.apply_soft(&mut pos, constraint) {
                    surv_pos.push(pos);
                    surv_pen.push(pen + 1);
                }
            }
        }
        self.particles = surv_pos;
        self.penalties = surv_pen;
    }

    /// 情報系の制約（王手宣言の一致・自分の反則の説明）だけを緩和した適用。
    /// 物理的な制約（自手の合法性・取った駒種・取られたマス）は緩和しない
    fn apply_soft(&mut self, pos: &mut Position, constraint: &Constraint) -> bool {
        match constraint {
            Constraint::MyMove { mv, captured, .. } => {
                apply_my_move(pos, self.my_color, mv, *captured, None)
            }
            // 粒子上では合法だった手が実際は反則だった: この粒子は反則を
            // 説明できないが、盤面自体は生かす（反則手は実行されていない）
            Constraint::MyFoul { .. } => true,
            Constraint::OppMove { captured_at, .. } => sample_opp_move(
                pos,
                self.my_color,
                *captured_at,
                None,
                &self.my_capture_sq,
                &self.my_touched_sq,
                &mut self.rng,
            ),
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
        let mut strict = self.penalties.iter().filter(|&&p| p == 0).count();
        if strict < self.target {
            for _ in 0..self.regen_attempts {
                if strict >= self.target || std::time::Instant::now() > regen_deadline {
                    break;
                }
                if let Some(pos) = self.replay_once() {
                    self.particles.push(pos);
                    self.penalties.push(0);
                    strict += 1;
                }
            }
        }
        let deadline = start + std::time::Duration::from_millis(self.empty_deadline_ms);
        while self.particles.is_empty() && std::time::Instant::now() < deadline {
            if let Some(pos) = self.replay_once() {
                self.particles.push(pos);
                self.penalties.push(0);
            }
        }
        // ラッチしない: 粒子が戻れば健全に戻る（呼び出し側は毎手 update する）
        self.healthy = !self.particles.is_empty();
        if self.particles.is_empty() {
            return;
        }
        // penalty 昇順に並べ、厳密整合の粒子を優先して target まで絞る
        let mut pairs: Vec<(u8, Position)> = std::mem::take(&mut self.penalties)
            .into_iter()
            .zip(std::mem::take(&mut self.particles))
            .collect();
        pairs.sort_by_key(|(p, _)| *p);
        pairs.truncate(self.target);
        for (pen, pos) in pairs {
            self.penalties.push(pen);
            self.particles.push(pos);
        }
        // 不足分は複製で埋める（低ペナルティ粒子を重み 0.5^penalty で優先）
        let m = self.particles.len();
        if m < self.target {
            let mut cum = Vec::with_capacity(m);
            let mut total = 0.0f64;
            for &p in &self.penalties {
                total += 0.5f64.powi(i32::from(p));
                cum.push(total);
            }
            while self.particles.len() < self.target {
                let t = self.rng.random_range(0.0..total);
                let i = cum.partition_point(|&c| c < t).min(m - 1);
                self.particles.push(self.particles[i].clone());
                self.penalties.push(self.penalties[i]);
            }
        }
    }

    /// 制約列を最初からリプレイして整合する粒子を1つ作る。
    ///
    /// 相手手のサンプルは確率的なので、後続の制約（自分の手の合法性・反則・
    /// 取られたマス・王手宣言）と矛盾して失敗しうる。全部やり直すと手数に対して
    /// 成功率が指数的に落ちるため、失敗したら直近の決定点（相手手）まで戻って
    /// 引き直す限定バックトラックにする。ステップ予算で最悪時間を抑える
    fn replay_once(&mut self) -> Option<Position> {
        let n = self.constraints.len();
        let step_budget = n * 4 + 32;
        let mut steps = 0usize;
        let mut pos = Position::initial();
        // 決定点スタック: (制約index, 適用前の局面, これまでの再試行回数)
        let mut stack: Vec<(usize, Position, u32)> = vec![];
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
                    let is_retry = stack.last().is_some_and(|(j, _, _)| *j == i);
                    if !is_retry {
                        stack.push((i, pos.clone(), 0));
                    }
                    // この時点までに自分が駒を取ったマス／触れたマス
                    let k = self.my_capture_idx.partition_point(|&j| j < i);
                    let t = self.my_touched_idx.partition_point(|&j| j < i);
                    sample_opp_move(
                        &mut pos,
                        self.my_color,
                        *captured_at,
                        Some(*gives_check),
                        &self.my_capture_sq[..k],
                        &self.my_touched_sq[..t],
                        &mut self.rng,
                    )
                }
            };
            if ok {
                i += 1;
                continue;
            }
            // 失敗: 直近の決定点に戻って引き直す。試行を使い切った点はさらに前へ
            loop {
                let Some((j, snapshot, attempts)) = stack.pop() else {
                    return None;
                };
                // 失敗した制約自身が決定点なら、同じ局面からの再試行は無意味
                // （整合候補ゼロは決定的）なのでさらに前へ戻る
                if j == i {
                    continue;
                }
                if attempts + 1 < BACKTRACK_ATTEMPTS {
                    pos = snapshot.clone();
                    stack.push((j, snapshot, attempts + 1));
                    i = j;
                    break;
                }
            }
        }
        Some(pos)
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

/// 観測と整合する相手の合法手をサンプルして適用する。整合手がなければ false。
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
) -> bool {
    let opp = my_color.other();
    if pos.turn() != opp {
        return false;
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
    for mv in pos.legal_moves() {
        // 取られたマスとの整合（取りがなかったなら自駒のあるマスへは来ていない）
        let to_capture = match mv {
            ShogiMove::Board { to, .. } => pos
                .piece_at(to)
                .filter(|p| p.color == my_color)
                .map(|p| (to, p.role)),
            ShogiMove::Drop { .. } => None,
        };
        match (captured_at, to_capture) {
            (Some(at), Some((to, _))) if at == to => {}
            (None, None) => {}
            _ => continue,
        }
        let mut next = pos.clone();
        next.play_unchecked(&mv);
        if gives_check.is_some_and(|gc| next.in_check(my_color) != gc) {
            continue;
        }
        let threat_known = newly_threatens(pos, &next, &mv, known_squares);
        let threat_home = newly_threatens(pos, &next, &mv, &homes);
        let (is_king, flee) = match mv {
            ShogiMove::Board { from, to, .. } => {
                let is_king = pos.piece_at(from).is_some_and(|p| p.role == Role::King);
                (is_king, is_king && flees_danger(from, to, known_squares))
            }
            ShogiMove::Drop { .. } => (false, false),
        };
        candidates.push((
            mv,
            opp_move_weight(opp, &mv, threat_known, threat_home, is_king, flee),
        ));
    }
    let Some(chosen) = weighted_choice(&candidates, rng) else {
        return false;
    };
    pos.play_unchecked(&chosen);
    true
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
    let opp = my_color.other();
    if pos.turn() != opp {
        return None;
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
        let mut w = opp_move_weight(opp, &mv, threat_known, threat_home, is_king, flee);
        if let ShogiMove::Board { to, .. } = mv {
            let captures_mine = pos.piece_at(to).is_some_and(|p| p.color == my_color);
            if captures_mine && known_squares.contains(&to) {
                w *= PREDICT_RECAPTURE_BOOST;
            }
        }
        candidates.push((mv, w));
    }
    weighted_choice(&candidates, rng)
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

/// 相手の手の尤度づけ。対人57局の条件付き最尤推定（bin/fit_opp, 2026-07-10、
/// 駒単位threat定義）: パープレキシティ 28.2（旧手調整）→ 25.3。
/// 駒取り・王手の有無は観測との整合ですでに絞り込まれているため、
/// 事前分布には「観測クラス内で判別できる特徴量」だけが現れる。
/// king_flee がわずかに負なのは実測（守りを剥がされても玉は特に逃げない）
fn opp_move_weight(
    opp: Color,
    mv: &ShogiMove,
    threat_known: bool,
    threat_home: bool,
    is_king_move: bool,
    king_flee: bool,
) -> f64 {
    let mut s = 0.0;
    match *mv {
        ShogiMove::Board { from, to, promote } => {
            let advance = match opp {
                Color::Sente => (from.rank - to.rank) as f64,
                Color::Gote => (to.rank - from.rank) as f64,
            };
            s += 0.139 * advance;
            if promote {
                s += 1.422;
            }
        }
        ShogiMove::Drop { .. } => s += -1.437,
    }
    if threat_known {
        s += 0.507;
    }
    if threat_home {
        s += 0.574;
    }
    if is_king_move {
        s += 0.136;
    }
    if king_flee {
        s += -0.159;
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


// ---------------------------------------------------------------------------
// 王手ソルバー（check.rs のコピー）
// ---------------------------------------------------------------------------

/// 王手駒になりうる駒種（玉は王手できない）
const CHECKER_ROLES: [Role; 13] = [
    Role::Pawn,
    Role::Lance,
    Role::Knight,
    Role::Silver,
    Role::Gold,
    Role::Bishop,
    Role::Rook,
    Role::Tokin,
    Role::Promotedlance,
    Role::Promotedknight,
    Role::Promotedsilver,
    Role::Horse,
    Role::Dragon,
];

/// 反則が仮説で説明できない（仮説の下では合法だったはず）ときの減衰係数。
/// 0にしない: 反則の真因が別の隠れ駒（経路封鎖・別の利き）の可能性があるため
const UNEXPLAINED_FOUL_DECAY: f64 = 0.15;

/// 粒子投票の強さ（全粒子が一致した仮説は一様仮説の 1+PARTICLE_VOTE_W 倍）
const PARTICLE_VOTE_W: f64 = 8.0;

struct Hypothesis {
    square: Coord,
    role: Role,
    weight: f64,
}

pub struct CheckSolver {
    /// 自駒＋持ち駒だけを置いたスパース盤面（手番=自分）。仮説の駒を載せて使う
    base: Position,
    my_color: Color,
    hypotheses: Vec<Hypothesis>,
}

impl CheckSolver {
    /// 王手中の view から作る。自玉が見つからない等で推論できなければ None。
    /// particles はソフト救済の重みつき（strategy.rs の評価サンプルと同じ）
    pub fn new(
        view: &PlayerView,
        particles: &[(&Position, f64)],
        fouls_this_turn: &[ShogiMove],
        log: &ObservationLog,
    ) -> Option<CheckSolver> {
        let my_color = view.your_color;
        let mut base = Position::empty(my_color);
        for piece in &view.your_pieces {
            let sq = crate::board::parse_usi_square(&piece.square)?;
            base.set(
                sq,
                Some(crate::shogi::Piece {
                    color: my_color,
                    role: piece.role,
                }),
            );
        }
        for (&role, &count) in &view.your_hand {
            base.set_hand(my_color, role, count as u8);
        }
        base.king_square(my_color)?;

        // 位置が既知の敵駒（自駒が死んだマス = 敵駒がそこへ来た。取り返し済みは除く）を
        // 盤に載せる。回避先がこれらの利きに覆われているかを全仮説共通で判定できる
        // （対人実戦: 5三の既知の成駒が 4二/5二/6二 を覆っているのに順に試して4反則）。
        // **直近8手以内**の新鮮な情報に限定する: 古いマスは駒が動いて陳腐化しやすく、
        // 幻の駒が合法な逃げ場を塞ぐ害が実測で上回った（vs v5 アブレーション 2026-07-10）。
        // 駒種は不明なので粒子の多数決、なければ成駒の最頻・金動き（と金）で近似する
        for sq in known_enemy_squares(log, view.move_number.saturating_sub(8)) {
            if base.piece_at(sq).is_some() {
                continue;
            }
            let role = particle_majority_role(particles, my_color.other(), sq)
                .unwrap_or(Role::Tokin);
            base.set(
                sq,
                Some(crate::shogi::Piece {
                    color: my_color.other(),
                    role,
                }),
            );
            // 近似駒種が王を攻撃してしまう（本物の王手駒と区別できない）配置は
            // 仮説列挙を壊すので載せない
            if base.in_check(my_color) {
                base.set(sq, None);
            }
        }

        let mut solver = CheckSolver {
            base,
            my_color,
            hypotheses: vec![],
        };
        solver.enumerate(&opponent_role_counts(view, log));
        if solver.hypotheses.is_empty() {
            return None;
        }
        solver.vote_by_particles(particles);
        for foul in fouls_this_turn {
            solver.observe_foul(foul);
        }
        Some(solver)
    }

    /// 自玉を攻撃しうる（マス, 駒種）を全列挙する。
    /// 相手が1枚も持ちえない駒種（総数制約）は仮説から外す
    fn enumerate(&mut self, opp_counts: &HashMap<Role, i32>) {
        let opp = self.my_color.other();
        let king = self.base.king_square(self.my_color).expect("new で確認済み");
        for file in 1..=9i8 {
            for rank in 1..=9i8 {
                let sq = Coord { file, rank };
                if self.base.piece_at(sq).is_some() {
                    // 自駒・既知の敵駒のあるマスに（新たな）王手駒はいない
                    // （既知の敵駒が王手していたなら以前から王手宣言されているはず）
                    continue;
                }
                if sq == king {
                    continue;
                }
                for role in CHECKER_ROLES {
                    if opp_counts
                        .get(&unpromote_role(role))
                        .is_none_or(|&n| n <= 0)
                    {
                        continue;
                    }
                    self.base.set(
                        sq,
                        Some(crate::shogi::Piece { color: opp, role }),
                    );
                    let checks = self.base.in_check(self.my_color);
                    self.base.set(sq, None);
                    if checks {
                        self.hypotheses.push(Hypothesis {
                            square: sq,
                            role,
                            weight: 1.0,
                        });
                    }
                }
            }
        }
    }

    /// 粒子中の実際の王手駒に投票させる（粒子が健全なら仮説が鋭くなる）。
    /// ソフト救済の粒子は重みぶんだけ薄く投票する
    fn vote_by_particles(&mut self, particles: &[(&Position, f64)]) {
        let opp = self.my_color.other();
        let mut voters = 0.0f64;
        let mut votes: Vec<f64> = vec![0.0; self.hypotheses.len()];
        for (pos, w) in particles {
            if !pos.in_check(self.my_color) {
                continue; // 王手を反映していない粒子は情報にならない
            }
            voters += w;
            for (i, h) in self.hypotheses.iter().enumerate() {
                if pos.piece_at(h.square)
                    .is_some_and(|p| p.color == opp && p.role == h.role)
                {
                    // 粒子上でその駒が実際に王を攻撃しているかまでは見ない
                    // （enumerate 済みの仮説は自駒配置的に攻撃可能）
                    votes[i] += w;
                }
            }
        }
        if voters <= 0.0 {
            return;
        }
        for (h, &v) in self.hypotheses.iter_mut().zip(&votes) {
            h.weight *= 1.0 + PARTICLE_VOTE_W * (v / voters);
        }
    }

    /// この手番の反則を観測: 仮説の下で合法だったはずの手が反則になった
    /// → その仮説の重みを減衰させる
    fn observe_foul(&mut self, foul: &ShogiMove) {
        for i in 0..self.hypotheses.len() {
            if self.legal_under(i, foul) {
                self.hypotheses[i].weight *= UNEXPLAINED_FOUL_DECAY;
            }
        }
    }

    /// 仮説 i の下で（他の隠れ駒を無視して）mv が合法か = 王手を解消するか
    fn legal_under(&mut self, i: usize, mv: &ShogiMove) -> bool {
        let h = &self.hypotheses[i];
        let piece = crate::shogi::Piece {
            color: self.my_color.other(),
            role: h.role,
        };
        let sq = h.square;
        self.base.set(sq, Some(piece));
        let legal = self.base.is_legal(mv);
        self.base.set(sq, None);
        legal
    }

    /// 候補手が王手を解消する確率（仮説の重み付き割合）
    pub fn resolve_probability(&mut self, mv: &ShogiMove) -> f64 {
        let mut total = 0.0;
        let mut resolved = 0.0;
        for i in 0..self.hypotheses.len() {
            let w = self.hypotheses[i].weight;
            total += w;
            if self.legal_under(i, mv) {
                resolved += w;
            }
        }
        if total <= 0.0 {
            return 0.5; // 全仮説が死んだ（両王手など）: 情報なしに戻す
        }
        resolved / total
    }

    #[cfg(test)]
    fn hypothesis_count(&self) -> usize {
        self.hypotheses.len()
    }
}

/// 位置が既知の敵駒のマス: 自駒が取られたマス（敵駒がそこへ来た事実）のうち、
/// その後に自分が取り返しておらず、かつ since_move 手目以降の新しいもの
fn known_enemy_squares(log: &ObservationLog, since_move: u32) -> Vec<Coord> {
    let mut map: HashMap<Coord, u32> = HashMap::new();
    for e in log.events() {
        match e {
            crate::observation::Observation::OpponentMoved {
                move_number,
                captured_my_piece_at: Some(sq),
            } => {
                if let Some(c) = crate::board::parse_usi_square(sq) {
                    map.insert(c, *move_number);
                }
            }
            crate::observation::Observation::MyMove {
                usi,
                captured: Some(_),
                ..
            } => {
                if let Some(ShogiMove::Board { to, .. }) = crate::shogi::parse_usi(usi) {
                    map.remove(&to);
                }
            }
            _ => {}
        }
    }
    map.into_iter()
        .filter(|(_, mn)| *mn >= since_move)
        .map(|(c, _)| c)
        .collect()
}

/// 粒子の加重多数決でそのマスの敵駒の駒種を推定する（過半に満たなければ None）。
/// ソフト救済の粒子は重みぶんだけ薄く数える
fn particle_majority_role(particles: &[(&Position, f64)], opp: Color, sq: Coord) -> Option<Role> {
    if particles.is_empty() {
        return None;
    }
    let total: f64 = particles.iter().map(|(_, w)| w).sum();
    let mut counts: HashMap<Role, f64> = HashMap::new();
    for (pos, w) in particles {
        if let Some(p) = pos.piece_at(sq) {
            if p.color == opp {
                *counts.entry(p.role).or_default() += w;
            }
        }
    }
    let (role, n) = counts
        .into_iter()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
    if n * 2.0 > total {
        Some(role)
    } else {
        None
    }
}

/// 相手が盤上・持ち駒に持ちうる駒種の枚数（基本駒種で数える）。
/// = 初期枚数 + こちらが取られた枚数 − こちらが取った枚数（自分の持ち駒）
fn opponent_role_counts(view: &PlayerView, log: &ObservationLog) -> HashMap<Role, i32> {
    let mut counts: HashMap<Role, i32> = [
        (Role::Pawn, 9),
        (Role::Lance, 2),
        (Role::Knight, 2),
        (Role::Silver, 2),
        (Role::Gold, 2),
        (Role::Bishop, 1),
        (Role::Rook, 1),
    ]
    .into();
    for (_, role) in GameModel::from_log(view.your_color, log).lost_pieces() {
        *counts.entry(unpromote_role(*role)).or_default() += 1;
    }
    for (&role, &n) in &view.your_hand {
        *counts.entry(unpromote_role(role)).or_default() -= n as i32;
    }
    counts
}


// ---------------------------------------------------------------------------
// 序盤定跡ブック（opening.rs のコピー。ラインは凍結時点の joseki.json）
// ---------------------------------------------------------------------------

// 凍結時点の joseki.json を焼き込んだ定跡ライン（凍結版は挙動固定のためファイルを読まない）
const LINES: [&[&str]; 13] = [
    // 居飛車速攻
    &["2g2f", "2f2e", "2e2d", "2d2c+"],
    // 最速引き角居飛車速攻
    &["7i7h", "8h7i", "5g5f", "2g2f", "2f2e", "2e2d", "2d2c+"],
    // 引き角準居飛車速攻
    &["7i7h", "8h7i", "5i4h", "5g5f", "2g2f", "2f2e", "2e2d", "2d2c+"],
    // 引き角7六玉型居飛車
    &["7i7h", "8h7i", "5i5h", "6g6f", "5h6g", "6g7f", "5g5f", "2g2f", "2f2e", "2e2d", "2d2c+"],
    // 最速居飛車受け6八飛車端攻め
    &["6i7h", "7g7f", "7h7g", "7i7h", "2h6h", "3i3h", "4i5h", "5i6i", "6g6f", "5h6g", "9g9f", "8h9g", "6i7i", "1g1f", "1f1e", "1e1d", "1d1c+"],
    // 7七金向かい飛車
    &["7g7f", "8h6f", "6i7h", "7h7g", "2h8h", "8g8f", "8f8e", "8e8d", "8d8c+"],
    // 端角、7九玉、6八飛車
    &["9g9f", "8h9g", "7i7h", "5i6h", "6h7i", "2h6h", "3i3h", "4i5h", "6g6f", "5h6g", "7g7f", "7f7e", "6g7f", "8i7g", "7e7d", "7d7c+"],
    // 7七金 4八飛車
    &["7g7f", "8h6f", "6i7h", "7h7g", "5i6h", "6h7h", "2h4h", "3i3h", "3g3f", "4g4f", "3h4g", "2i3g", "4g5f", "4f4e", "5f5e", "4e4d", "4d4c+"],
    // 引き角　端攻め 1
    &["7i7h", "8h7i", "5i4h", "3i3h", "5g5f", "2g2f", "2h2g", "1g1f", "1f1e", "2i1g", "1g2e", "1e1d", "1d1c+"],
    // 引き角　端攻め 2
    &["7i7h", "8h7i", "5g5f", "1g1f", "1f1e", "2i1g", "1g2e", "1e1d", "1d1c+"],
    // 端攻め　飛車投球
    &["1g1f", "1f1e", "1i1f", "2h1h", "2i1g", "1g2e", "1e1d", "1d1c+"],
    // 左端ぜめ　通常
    &["7g7f", "8h6f", "8i7g", "9g9f", "9f9e", "9e9d", "7g8e", "9d9c+"],
    // 左端ぜめ　端角から
    &["9g9f", "8h9g", "9f9e", "9g7e", "8i9g", "9g8e", "9e9d", "8e9c"],
];

fn lines() -> &'static Vec<Vec<String>> {
    static CACHE: std::sync::OnceLock<Vec<Vec<String>>> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| {
        LINES
            .iter()
            .map(|l| l.iter().map(|s| s.to_string()).collect())
            .collect()
    })
}
/// USI手を点対称にミラーする（先手ライン → 後手用）
fn mirror_usi(usi: &str) -> Option<String> {
    let mv = parse_usi(usi)?;
    let flip = |c: crate::board::Coord| crate::board::Coord {
        file: 10 - c.file,
        rank: 10 - c.rank,
    };
    let mirrored = match mv {
        ShogiMove::Board { from, to, promote } => ShogiMove::Board {
            from: flip(from),
            to: flip(to),
            promote,
        },
        ShogiMove::Drop { role, to } => ShogiMove::Drop { role, to: flip(to) },
    };
    Some(mirrored.to_usi())
}

pub struct OpeningBook {
    /// 対局開始時に選んだライン（自色向けにミラー済み）
    line: Vec<String>,
    /// ブックから抜けたら true（以後戻らない）
    exited: bool,
}

impl OpeningBook {
    /// 指定インデックスのラインに固定したブック（定跡特化チューニング用）
    pub fn with_line(my_color: Color, index: usize) -> Self {
        let all = lines();
        let raw = &all[index % all.len()];
        let line = raw
            .iter()
            .filter_map(|usi| match my_color {
                Color::Sente => Some(usi.clone()),
                Color::Gote => mirror_usi(usi),
            })
            .collect();
        OpeningBook {
            line,
            exited: false,
        }
    }

    pub fn new(my_color: Color) -> Self {
        // ランダム選択（対局をまたいで人間に順番を読まれないため）。
        // SPSA（bin/tune）は with_seed で決定論的に選ぶ（共通乱数法）
        Self::with_line(my_color, rand::rng().random_range(0..lines().len()))
    }

    /// シードから決定論的にラインを選ぶ（SPSA の f+/f− 評価で
    /// 同じ対局番号に同じ定跡を割り当てるための共通乱数法用）
    pub fn with_seed(my_color: Color, seed: u64) -> Self {
        Self::with_line(my_color, (seed % lines().len() as u64) as usize)
    }

    /// ブックの次の一手。None ならブックを抜けた（通常思考へ）
    pub fn next(
        &mut self,
        view: &PlayerView,
        log: &ObservationLog,
        foul_tried: &HashSet<String>,
    ) -> Option<String> {
        if self.exited {
            return None;
        }
        // 静かな序盤でなくなったら抜ける
        let quiet = log.events().iter().all(|e| match e {
            Observation::MyMove { captured, .. } => captured.is_none(),
            Observation::OpponentMoved {
                captured_my_piece_at,
                ..
            } => captured_my_piece_at.is_none(),
            Observation::MyFoul { .. } | Observation::Check { .. } => false,
            Observation::OpponentFoul { .. } => true, // 相手の反則は情報にならない
        });
        if !quiet || view.you_in_check {
            self.exited = true;
            return None;
        }
        // 自分が何手指したか = ラインの進行位置
        let my_moves = log
            .events()
            .iter()
            .filter(|e| matches!(e, Observation::MyMove { .. }))
            .count();
        let Some(usi) = self.line.get(my_moves) else {
            self.exited = true; // ライン消化完了
            return None;
        };
        if foul_tried.contains(usi.as_str()) {
            self.exited = true;
            return None;
        }
        // 自分の駒が想定位置にいるか（自分に見える範囲の妥当性チェック）
        let playable = match parse_usi(usi) {
            Some(ShogiMove::Board { from, to, .. }) => {
                let from_ok = view
                    .your_pieces
                    .iter()
                    .any(|p| parse_usi_square(&p.square) == Some(from));
                let to_free = !view
                    .your_pieces
                    .iter()
                    .any(|p| parse_usi_square(&p.square) == Some(to));
                from_ok && to_free
            }
            _ => false, // 定跡ラインに打ちは想定しない
        };
        if !playable {
            self.exited = true;
            return None;
        }
        Some(usi.clone())
    }
}


// ---------------------------------------------------------------------------
// 戦略（strategy.rs の EstimatorV6 のコピー）
// ---------------------------------------------------------------------------

/// 評価に使う粒子数の基準値（スケール1.0時）。実際の値は思考予算に比例する
const EVAL_PARTICLES: usize = 192;

/// 1手の思考予算（ms）の既定値。TSUITATE_THINK_BUDGET_MS で上書きできる。
/// このリポジトリのアリーナは 1000秒+3秒 なので既定はやや厚めに使う。
/// 本番サイト（300秒+3秒）へのデプロイ時は環境変数で絞って
/// 思考時間（=強さ）を調整する（例: 900 で v5 相当の予算）
const DEFAULT_THINK_BUDGET_MS: u64 = 2000;
/// スケール1.0の基準予算。v5 までの暗黙の実測上限（p99 ≒ 900ms）
const REFERENCE_BUDGET_MS: f64 = 900.0;

/// 思考予算（ms）。環境変数 > 既定値
fn think_budget_ms() -> u64 {
    std::env::var("TSUITATE_THINK_BUDGET_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_THINK_BUDGET_MS)
}

/// 思考予算に比例して各種の粒子数・読み幅を決める
#[derive(Debug, Clone, Copy)]
struct SearchBudget {
    /// 推定器へ渡すスケール（粒子数・リプレイ予算）
    scale: f64,
    /// 評価に使うユニーク粒子数の上限
    eval_particles: usize,
    /// 王周辺圧力を測る粒子数
    pressure_samples: usize,
    /// 2手読みする上位候補数
    depth2_top_k: usize,
    /// 2手読みに使う粒子数
    depth2_particles: usize,
}

impl SearchBudget {
    fn from_ms(ms: u64) -> Self {
        let scale = (ms as f64 / REFERENCE_BUDGET_MS).clamp(0.25, 8.0);
        let f = |base: usize, lo: usize, hi: usize| {
            ((base as f64 * scale) as usize).clamp(lo, hi)
        };
        SearchBudget {
            scale,
            eval_particles: f(EVAL_PARTICLES, 48, 2048),
            pressure_samples: f(PRESSURE_SAMPLES, 8, 64),
            depth2_top_k: f(DEPTH2_TOP_K, 4, 32),
            depth2_particles: f(DEPTH2_PARTICLES, 16, 384),
        }
    }
}

/// 王周辺圧力を測る粒子数の基準値（スケール1.0時）
const PRESSURE_SAMPLES: usize = 16;

/// 2手読み（相手応手のサンプル再評価）を行う上位候補数の基準値（スケール1.0時）。
/// 1手読みの静的リスク項は近似なので、有望手だけ実際の応手分布で検算する
const DEPTH2_TOP_K: usize = 8;
/// 2手読みに使う粒子数の基準値（1候補あたり・スケール1.0時）
const DEPTH2_PARTICLES: usize = 48;
/// 応手で詰まされる場合のペナルティ（壊滅的なのでSPSA対象にしない）
const DEPTH2_MATE_PEN: f64 = 30.0;

/// 駒交換で動く価値: 盤上価値と持ち駒価値（基本駒種）の平均。
/// 素の駒は piece_value と一致し、成駒は取られても相手の持ち駒に入るのは
/// 基本駒種ぶんなので割り引かれる（と金を取り返された反動 = (6+1)/2 = 3.5）。
/// 逆に成駒を取る側の得も同じ理由で割り引く
fn exchange_value(role: Role) -> f64 {
    (piece_value(role) + piece_value(unpromote_role(role))) / 2.0
}

/// 着手後の自駒の利き被覆マス数（自分に見える盤面だけの近似）。
/// 相手の駒は見えないため飛び駒は自駒にだけ遮られる楽観値
fn coverage_after(view: &PlayerView, mv: &ShogiMove) -> f64 {
    let mut pieces: Vec<VisiblePiece> = view.your_pieces.clone();
    match *mv {
        ShogiMove::Board { from, to, promote } => {
            let from_usi = make_usi_square(from);
            let Some(p) = pieces.iter_mut().find(|p| p.square == from_usi) else {
                return 0.0;
            };
            if promote {
                if let Some(r) = promote_role(p.role) {
                    p.role = r;
                }
            }
            p.square = make_usi_square(to);
        }
        ShogiMove::Drop { role, to } => pieces.push(VisiblePiece {
            square: make_usi_square(to),
            role,
        }),
    }
    let mut covered: HashSet<Coord> = HashSet::new();
    for p in &pieces {
        covered.extend(move_targets(&pieces, p, view.your_color));
    }
    covered.len() as f64
}

/// 持ち駒の歩を成れる圏内（敵陣＋一段手前）へ打つ手か（1.0/0.0）。
/// 打った直後の利きは1マスだが、次に成れば利きが6マスへ広がる索敵ユニットになり、
/// 取り返されても相手に渡るのは歩1枚で反動が最小。重みは params.tokin_probe_w
fn tokin_probe(view: &PlayerView, mv: &ShogiMove) -> f64 {
    let ShogiMove::Drop {
        role: Role::Pawn,
        to,
    } = *mv
    else {
        return 0.0;
    };
    let depth_from_back = match view.your_color {
        Color::Sente => to.rank,
        Color::Gote => 10 - to.rank,
    };
    if depth_from_back <= 4 { 1.0 } else { 0.0 }
}

/// アンチドロー（終盤の寄せ）: 増幅を始める手数（plies）
const ANTI_DRAW_START: f64 = 60.0;
/// 増幅が最大になる手数。アリーナの手数上限200の手前で全開にする
const ANTI_DRAW_FULL: f64 = 160.0;
/// リードの正規化単位（歩換算。8 ≒ 飛車1枚のリードでほぼフル増幅）
const ANTI_DRAW_LEAD_UNIT: f64 = 8.0;

/// 終盤の攻め増幅係数。手数が進むほど・素材リードがあるほど大きくなる。
/// 互角でも弱く掛けて膠着を破りにいくが、負けているときは掛けない
/// （負けているときの引き分けは0.5勝ぶんの価値がある）
fn endgame_push(move_number: u32, lead: f64) -> f64 {
    let ramp = ((f64::from(move_number) - ANTI_DRAW_START) / (ANTI_DRAW_FULL - ANTI_DRAW_START))
        .clamp(0.0, 1.0);
    (ramp * (0.3 + (lead / ANTI_DRAW_LEAD_UNIT).clamp(-0.3, 1.2))).max(0.0)
}

/// 観測から確実に分かる素材リード（歩換算・相対値）。
/// 自分の駒の増減は取った駒（持ち駒に入る）と取られた駒を両方含み、
/// 相手側は鏡像（自分が+vなら相手は-v）なので、リード = 自分の変化×2。
/// 成りは基本駒種で数える（成駒を取った得は過小評価だが単調な信号としては十分）
fn material_lead(view: &PlayerView) -> f64 {
    let current: f64 = view
        .your_pieces
        .iter()
        .map(|p| piece_value(unpromote_role(p.role)))
        .sum::<f64>()
        + view
            .your_hand
            .iter()
            .map(|(r, n)| piece_value(*r) * f64::from(*n))
            .sum::<f64>();
    let initial: f64 = Position::initial()
        .pieces()
        .filter(|(_, p)| p.color == view.your_color)
        .map(|(_, p)| piece_value(p.role))
        .sum();
    2.0 * (current - initial)
}

/// evaluate() の結果。最終スコアでなく内訳を保持し、2手読みが
/// gain を組み替えた後に同じ最終式を適用し直せるようにする
/// （min形の非線形式に対して後から線形補正すると負のgainで壊れるため）
struct EvalOut {
    /// 期待値＋バイアス項（合法確率・反則コストを含まない）
    gain: f64,
    /// 静的な取られリスク項（mover/hidden の max）の粒子加重平均。
    /// 2手読みがこの分をサンプル実測で置き換える
    risk_mean: f64,
    p_legal: f64,
    foul_cost: f64,
}

impl EvalOut {
    fn score(&self) -> f64 {
        combine_score(self.gain, self.p_legal, self.foul_cost)
    }
}

/// 最終スコア: 期待値が負の手を p_legal で割り引かない（min の形）。
/// 割り引くと「合法確率が低いほどスコアが高い」= わざと反則に寄る手が
/// 選ばれてしまう。反則しても手番は残るので悪い局面からは逃げられず、
/// 反則の価値は「次善手の価値 − 反則コスト」でしかない
fn combine_score(gain: f64, p_legal: f64, foul_cost: f64) -> f64 {
    (p_legal * gain).min(gain) - (1.0 - p_legal) * foul_cost
}

/// evaluate() まわりの調整可能パラメータ。Default が現行の手調整値。
/// bin/tune.rs の SPSA がこれを最適化する（凍結版は各自のコピーを持ち依存しない）
#[derive(Debug, Clone)]
pub struct EvalParams {
    /// 王手ボーナスの基本値
    pub check_bonus: f64,
    /// 王手ボーナスの相手反則数スケール
    pub check_foul_scale: f64,
    /// 着手駒の取られリスク重み（駒を取った直後 = 位置がバレている）
    pub mover_w_captured: f64,
    /// 着手駒の取られリスク重み（静かな手）
    pub mover_w_quiet: f64,
    /// 着手駒の取られリスク重みへの加算（王手をかけた手）。王手宣言は「王を攻撃
    /// できる（マス,駒種）」まで仮説を絞らせるので、相手は反則覚悟の探り取りで
    /// 王手駒を高確率で回収できる（対人実戦: 竜の王手→2反則で竜を取られた）。
    /// 旧 mover_w_check は quiet/captured との max で不感帯があった
    /// （SPSAで勾配が立たない）ため、非負の加算に変更
    pub mover_check_extra: f64,
    /// 捕獲後の残留露見リスク（自駒価値に掛ける割合）。取ったマスは相手に
    /// 通知されるため、粒子に守り駒が見えなくても取り返しの下限リスクを敷く。
    /// 等価な取りなら安い駒で取る、というタイブレークにもなる
    /// （対人実戦: 成桂でも取れる角を竜で取って竜を回収された）
    pub capture_reveal_risk: f64,
    /// 敵陣リスク下限の「静かな進入」係数（捕獲時は 1.0）
    pub camp_known_quiet: f64,
    /// 敵陣の守られ事前確率のスケール（1.0 で 0.25/0.2/0.15）
    pub camp_scale: f64,
    /// 露出リスクの基本重み
    pub exposed_base: f64,
    /// 露出リスクの既知度係数
    pub exposed_known: f64,
    /// 初期配置から動いていない駒の既知度
    pub home_knownness: f64,
    /// 紐つき割引（着手駒）
    pub recapture_defended: f64,
    /// 紐つき割引（露出駒）
    pub exposed_defended: f64,
    /// 相手玉周辺への攻め圧力の重み
    pub attack_w: f64,
    /// 自玉周辺への相手圧力の重み
    pub pressure_w: f64,
    /// 反則コストの基本値
    pub foul_cost_base: f64,
    /// 反則コストの急峻さ（残り反則数に対する冪）
    pub foul_cost_pow: f64,
    /// 前進バイアス
    pub advance_w: f64,
    /// 成りバイアス
    pub promote_bias: f64,
    /// 打ちバイアス
    pub drop_bias: f64,
    /// p(合法) 事前確率の擬似観測数
    pub prior_weight: f64,
    /// 粒子退化時に prior_weight へ加算する上限（ユニーク粒子が減るほど事前を信じる。
    /// 少数の複製・偏った粒子への過信 = 「自信過剰な間違い」を防ぐ）
    pub prior_weight_degen: f64,
    /// 着手後に自分が当たりを付けている敵駒の価値への重み（露出リスクの鏡像）。
    /// 1手読みでは見えない「次の駒得」（飛車頭への歩打ち等）を作る手に価値を与える
    pub threat_w: f64,
    /// 探索ボーナス: 着地マスの敵駒有無について粒子が割れているほど加点。
    /// 取れても空振りでも観測が推定を絞る（情報の価値）
    pub info_bonus: f64,
    /// 大駒（飛・角）が初期位置に残っていることへのペナルティ（1枚あたり）。
    /// 初期位置の大駒は位置が予測可能で、開いた筋の背後を歩・桂で狙われる
    /// （対人50局で頻発）。展開を促す勾配を作り、動かせば消える
    pub big_home_penalty: f64,
    /// 相手の持ち駒による「打ち込み王手の受け入れ面積」への重み。
    /// 相手の持ち駒は既知（=取られた自駒）で、飛を持たれたら玉への開いた直線、
    /// 金銀なら玉の隣接空きマスがすべて王手打ちの入口になる。
    /// 持ち駒が空なら居玉でもコストゼロ（一律の玉移動推奨はしない）
    pub hand_drop_w: f64,
    /// 手戻り減点
    pub backtrack_penalty: f64,
    /// 直前に動かした駒をまた動かす手の減点（雑なシャッフルの抑制。
    /// 駒得や王手が絡む手は期待値側が勝つので実質影響しない）
    pub shuffle_penalty: f64,
    /// ソフト救済粒子の評価重み減衰（重み = soft_decay^penalty。厳密整合=1.0）
    pub soft_decay: f64,
    /// 王探しの情報利得: 粒子間で王手判定が割れる手への p(1-p) 加点
    pub king_probe_bonus: f64,
    /// 利き被覆1マスあたりの加点（自駒のみ考慮の近似被覆）
    pub coverage_w: f64,
    /// 成れる圏内への歩打ちのと金ポテンシャル加点
    pub tokin_probe_w: f64,
    /// 2手読みで静的リスク項をサンプル実測に置き換える割合（0=従来、1=全面置換）
    pub depth2_replace: f64,
    /// 2手読みで応手に王手を掛けられた場合のペナルティ
    pub depth2_check_pen: f64,
    /// 2手読みの取り返し補償の割引（取り返し自体への反撃リスクの近似）
    pub depth2_recap_discount: f64,
}

impl Default for EvalParams {
    fn default() -> Self {
        // SPSA第2ラウンドの収束点（2026-07-14、60反復×2×40局 vs estimator_v5、
        // 共通乱数法・tuning/tune-round2.jsonl、最終中心点の追加評価 score=0.675）。
        // 第1ラウンド（2026-07-11）からの主な動き: check_bonus 大幅減
        // （0.75→0.16。王手自体より check_foul_scale 側=相手の反則蓄積で加点）、
        // prior_weight_degen 増（4.7→8.0、退化時は事前をさらに信頼）、
        // threat_w 増（0.31→0.46）、coverage_w はほぼゼロへ
        // （利き被覆の一律加点は効かず、と金・王探しの個別項が残った）
        EvalParams {
            check_bonus: 0.1619,
            check_foul_scale: 0.0983,
            mover_w_captured: 0.8042,
            mover_w_quiet: 0.7312,
            mover_check_extra: 0.0622,
            capture_reveal_risk: 0.1313,
            camp_known_quiet: 0.4472,
            camp_scale: 0.1252,
            exposed_base: 0.4576,
            exposed_known: 0.1659,
            home_knownness: 0.0027,
            recapture_defended: 0.4692,
            exposed_defended: 0.3031,
            attack_w: 0.0434,
            pressure_w: 0.0918,
            foul_cost_base: 0.637,
            foul_cost_pow: 1.3331,
            advance_w: 0.0699,
            promote_bias: 0.1466,
            drop_bias: 0.2616,
            prior_weight: 4.9065,
            prior_weight_degen: 7.9515,
            threat_w: 0.4586,
            info_bonus: 0.64,
            big_home_penalty: 0.3156,
            hand_drop_w: 0.0757,
            backtrack_penalty: 0.3685,
            shuffle_penalty: 0.2996,
            soft_decay: 0.6753,
            king_probe_bonus: 0.2451,
            coverage_w: 0.0013,
            tokin_probe_w: 0.2025,
            depth2_replace: 0.6205,
            depth2_check_pen: 0.178,
            depth2_recap_discount: 0.7612,
        }
    }
}

/// SPSA用のパラメータ仕様（名前と探索範囲）。to_vec/from_vec と同じ順序
pub struct ParamSpec {
    pub name: &'static str,
    pub lo: f64,
    pub hi: f64,
}

impl EvalParams {
    pub const SPECS: [ParamSpec; 35] = [
        ParamSpec { name: "check_bonus", lo: 0.0, hi: 3.0 },
        ParamSpec { name: "check_foul_scale", lo: 0.0, hi: 0.5 },
        ParamSpec { name: "mover_w_captured", lo: 0.0, hi: 1.5 },
        ParamSpec { name: "mover_w_quiet", lo: 0.0, hi: 1.5 },
        ParamSpec { name: "mover_check_extra", lo: 0.0, hi: 1.0 },
        ParamSpec { name: "capture_reveal_risk", lo: 0.0, hi: 0.6 },
        ParamSpec { name: "camp_known_quiet", lo: 0.0, hi: 1.0 },
        ParamSpec { name: "camp_scale", lo: 0.0, hi: 3.0 },
        ParamSpec { name: "exposed_base", lo: 0.0, hi: 1.5 },
        ParamSpec { name: "exposed_known", lo: 0.0, hi: 1.5 },
        ParamSpec { name: "home_knownness", lo: 0.0, hi: 1.0 },
        ParamSpec { name: "recapture_defended", lo: 0.0, hi: 1.0 },
        ParamSpec { name: "exposed_defended", lo: 0.0, hi: 1.0 },
        ParamSpec { name: "attack_w", lo: 0.0, hi: 0.5 },
        ParamSpec { name: "pressure_w", lo: 0.0, hi: 0.6 },
        ParamSpec { name: "foul_cost_base", lo: 0.2, hi: 6.0 },
        ParamSpec { name: "foul_cost_pow", lo: 0.5, hi: 3.0 },
        ParamSpec { name: "advance_w", lo: -0.1, hi: 0.3 },
        ParamSpec { name: "promote_bias", lo: -0.2, hi: 0.6 },
        ParamSpec { name: "drop_bias", lo: -0.5, hi: 0.3 },
        ParamSpec { name: "prior_weight", lo: 0.5, hi: 16.0 },
        ParamSpec { name: "prior_weight_degen", lo: 0.0, hi: 32.0 },
        ParamSpec { name: "threat_w", lo: 0.0, hi: 1.0 },
        ParamSpec { name: "info_bonus", lo: 0.0, hi: 2.0 },
        ParamSpec { name: "big_home_penalty", lo: 0.0, hi: 1.5 },
        ParamSpec { name: "hand_drop_w", lo: 0.0, hi: 0.5 },
        ParamSpec { name: "backtrack_penalty", lo: 0.0, hi: 1.5 },
        ParamSpec { name: "shuffle_penalty", lo: 0.0, hi: 1.0 },
        ParamSpec { name: "soft_decay", lo: 0.05, hi: 1.0 },
        ParamSpec { name: "king_probe_bonus", lo: 0.0, hi: 1.5 },
        ParamSpec { name: "coverage_w", lo: 0.0, hi: 0.1 },
        ParamSpec { name: "tokin_probe_w", lo: 0.0, hi: 1.0 },
        ParamSpec { name: "depth2_replace", lo: 0.0, hi: 1.0 },
        ParamSpec { name: "depth2_check_pen", lo: 0.0, hi: 1.5 },
        ParamSpec { name: "depth2_recap_discount", lo: 0.0, hi: 1.0 },
    ];

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.check_bonus,
            self.check_foul_scale,
            self.mover_w_captured,
            self.mover_w_quiet,
            self.mover_check_extra,
            self.capture_reveal_risk,
            self.camp_known_quiet,
            self.camp_scale,
            self.exposed_base,
            self.exposed_known,
            self.home_knownness,
            self.recapture_defended,
            self.exposed_defended,
            self.attack_w,
            self.pressure_w,
            self.foul_cost_base,
            self.foul_cost_pow,
            self.advance_w,
            self.promote_bias,
            self.drop_bias,
            self.prior_weight,
            self.prior_weight_degen,
            self.threat_w,
            self.info_bonus,
            self.big_home_penalty,
            self.hand_drop_w,
            self.backtrack_penalty,
            self.shuffle_penalty,
            self.soft_decay,
            self.king_probe_bonus,
            self.coverage_w,
            self.tokin_probe_w,
            self.depth2_replace,
            self.depth2_check_pen,
            self.depth2_recap_discount,
        ]
    }

    pub fn from_vec(v: &[f64]) -> EvalParams {
        assert_eq!(v.len(), Self::SPECS.len());
        EvalParams {
            check_bonus: v[0],
            check_foul_scale: v[1],
            mover_w_captured: v[2],
            mover_w_quiet: v[3],
            mover_check_extra: v[4],
            capture_reveal_risk: v[5],
            camp_known_quiet: v[6],
            camp_scale: v[7],
            exposed_base: v[8],
            exposed_known: v[9],
            home_knownness: v[10],
            recapture_defended: v[11],
            exposed_defended: v[12],
            attack_w: v[13],
            pressure_w: v[14],
            foul_cost_base: v[15],
            foul_cost_pow: v[16],
            advance_w: v[17],
            promote_bias: v[18],
            drop_bias: v[19],
            prior_weight: v[20],
            prior_weight_degen: v[21],
            threat_w: v[22],
            info_bonus: v[23],
            big_home_penalty: v[24],
            hand_drop_w: v[25],
            backtrack_penalty: v[26],
            shuffle_penalty: v[27],
            soft_decay: v[28],
            king_probe_bonus: v[29],
            coverage_w: v[30],
            tokin_probe_w: v[31],
            depth2_replace: v[32],
            depth2_check_pen: v[33],
            depth2_recap_discount: v[34],
        }
    }
}

/// 観測履歴から相手局面を推定して指す戦略。
///
/// 候補手（自分に見える範囲の疑似合法手）を、推定粒子の平均で評価する:
/// - 駒得の期待値（その粒子でそのマスに相手駒がいるか）
/// - 反則確率（粒子上で非合法な割合）× 反則コスト（残り反則数が減るほど高い）
/// - 指した直後に取り返されるリスク（粒子上での相手の即時駒取り）
/// - 王手・詰みボーナス
pub struct EstimatorV6 {
    est: Option<Estimator>,
    book: Option<OpeningBook>,
    /// Some なら定跡をこのラインに固定する（定跡特化チューニング用）
    book_line: Option<usize>,
    params: EvalParams,
    /// 思考予算に応じた粒子数・読み幅（TSUITATE_THINK_BUDGET_MS 由来）
    budget: SearchBudget,
    /// Some なら推定器・定跡選択・タイブレークの乱数をこのシードから導出する
    /// （SPSA の共通乱数法用。None は従来どおりエントロピー由来）
    seed: Option<u64>,
    /// 評価タイブレーク用の乱数（seed があれば決定論的）
    rng: StdRng,
    /// 直近の choose 時点の内部状態（記録用）
    last_debug: Option<serde_json::Value>,
}

impl EstimatorV6 {
    pub fn new() -> Self {
        Self::with_params_line_seed(EvalParams::default(), None, None)
    }

    /// シードつきで作る（SPSA/アブレーションの共通乱数法用。挙動分布は new と同じ）
    pub fn with_seed(seed: u64) -> Self {
        Self::with_params_line_seed(EvalParams::default(), None, Some(seed))
    }

    fn with_params_line_seed(
        params: EvalParams,
        book_line: Option<usize>,
        seed: Option<u64>,
    ) -> Self {
        EstimatorV6 {
            est: None,
            book: None,
            book_line,
            params,
            budget: SearchBudget::from_ms(think_budget_ms()),
            seed,
            rng: match seed {
                Some(s) => StdRng::seed_from_u64(s ^ 0xA5A5_5A5A_DEAD_BEEF),
                None => StdRng::seed_from_u64(rand::rng().random()),
            },
            last_debug: None,
        }
    }
}

impl Default for EstimatorV6 {
    fn default() -> Self {
        Self::new()
    }
}

impl Strategy for EstimatorV6 {
    fn choose(
        &mut self,
        view: &PlayerView,
        log: &ObservationLog,
        foul_tried: &HashSet<String>,
    ) -> Option<String> {
        let budget = self.budget;
        let seed = self.seed;
        let est = self.est.get_or_insert_with(|| match seed {
            Some(s) => Estimator::with_seed_and_scale(view.your_color, s, budget.scale),
            None => Estimator::with_scale(view.your_color, budget.scale),
        });
        est.update(log);

        // 序盤定跡（静かな間だけ）。ブック中も推定器の update は回して粒子を保つ
        let book_line = self.book_line;
        let book = self.book.get_or_insert_with(|| match (book_line, seed) {
            (Some(idx), _) => OpeningBook::with_line(view.your_color, idx),
            (None, Some(s)) => OpeningBook::with_seed(view.your_color, s),
            (None, None) => OpeningBook::new(view.your_color),
        });
        if let Some(usi) = book.next(view, log, foul_tried) {
            return Some(usi);
        }

        let mut candidates = candidate_moves(view, foul_tried);
        if view.you_in_check {
            // 王手中: 解消しえない手は（王手駒がどこにいても）王手放置で必ず反則に
            // なるので候補から外す。全滅したら元の候補に戻す（投了よりは反則のほうが
            // 手番を失わないぶんまし。真に詰みならサーバー側で終局している）
            let filtered: Vec<_> = candidates
                .iter()
                .filter(|(_, mv)| may_resolve_check(view, mv))
                .cloned()
                .collect();
            if !filtered.is_empty() {
                candidates = filtered;
            }
        }
        if candidates.is_empty() {
            return None;
        }

        // 複製粒子を指紋で除いたユニーク粒子だけを評価に使う
        // （複製は独立な証拠ではないので p(合法) を過信させる）。
        // ソフト救済された粒子（penalty>0）は重み 0.5^penalty で薄く数える。
        // 粒子は penalty 昇順なので厳密整合の粒子から先に採用される。
        // 粒子が完全に枯渇していても、事前確率だけで安全側の評価が成り立つ
        let mut seen = HashSet::new();
        let mut sample: Vec<(&Position, f64)> = vec![];
        for (pos, pen) in est.particles().iter().zip(est.penalties()) {
            if sample.len() >= budget.eval_particles {
                break;
            }
            if seen.insert(pos.fingerprint()) {
                sample.push((pos, self.params.soft_decay.powi(i32::from(*pen))));
            }
        }

        // 相手の盤上駒数の概算（取った枚数ぶん減る。相手の打ちで戻る分は無視）
        let my_captures = log
            .events()
            .iter()
            .filter(|e| matches!(e, Observation::MyMove { captured: Some(_), .. }))
            .count();
        let opp_board_n = (20 - my_captures.min(19)) as f64;

        // 直前に受理された自分の手（手戻りシャッフルの抑制に使う）
        let last_my_move = log.events().iter().rev().find_map(|e| match e {
            Observation::MyMove { usi, .. } => parse_usi(usi),
            _ => None,
        });

        // 王手中は粒子に依存しない制約推論で「王手を解消する確率」を出す
        // （粒子が枯渇する終盤の反則バースト対策。check.rs 参照）
        let mut check_solver = if view.you_in_check {
            let fouls: Vec<ShogiMove> =
                foul_tried.iter().filter_map(|u| parse_usi(u)).collect();
            CheckSolver::new(view, &sample, &fouls, log)
        } else {
            None
        };

        // 相手が位置を知っている自駒（露出）の地図
        let known = knownness_map(view, log, self.params.home_knownness);

        // 2手読み用: 自分が駒を取ったマス（露見）と自分の手が触れたマス
        // （estimator の my_capture_sq / my_touched_sq と同じ定義）
        let mut my_capture_squares: Vec<Coord> = vec![];
        let mut my_touched_squares: Vec<Coord> = vec![];
        for e in log.events() {
            if let Observation::MyMove { usi, captured, .. } = e {
                if let Some(mv) = parse_usi(usi) {
                    let to = match mv {
                        ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
                    };
                    if captured.is_some() {
                        my_capture_squares.push(to);
                    }
                    if let ShogiMove::Board { from, .. } = mv {
                        my_touched_squares.push(from);
                    }
                    my_touched_squares.push(to);
                }
            }
        }

        // アンチドロー: 終盤にリードがあるほど攻め項を増幅して膠着を破る。
        // 手戻り/シャッフルの減点も同時に強めて「その場で回る」手を締め出す
        let push = endgame_push(view.move_number, material_lead(view));
        let params = {
            let mut p = self.params.clone();
            if push > 0.0 {
                p.check_bonus *= 1.0 + push;
                p.attack_w *= 1.0 + push;
                p.advance_w *= 1.0 + 0.5 * push;
                p.backtrack_penalty *= 1.0 + push;
                p.shuffle_penalty *= 1.0 + push;
            }
            p
        };

        let rng = &mut self.rng;
        // 1段目: 全候補を1手読み（静的リスク項つき）で評価する。
        // (usi, mv, 内訳, gain外の補正, 1段目スコア)
        let mut scored: Vec<(String, ShogiMove, EvalOut, f64, f64)> = vec![];
        for (usi, mv) in candidates {
            let mut prior = prior_legal(view, &mv, opp_board_n);
            if view.you_in_check {
                prior *= match check_solver.as_mut() {
                    Some(solver) => solver.resolve_probability(&mv).clamp(0.02, 1.0),
                    // ソルバーが作れないときは従来の粗い事前確率
                    // （玉移動 > 取り/合駒の順）に落とす
                    None => in_check_prior(view, &mv),
                };
            }
            let out = evaluate(view, &mv, &sample, prior, &known, &params, budget);
            // gain の外側の補正（タイブレーク乱数・手戻り/シャッフル減点）は
            // 2手読み後の再計算でも同じ値を使うので分離して持つ
            let mut adjust = rng.random_range(0.0..0.01);
            // 手戻り（直前の手をそのまま逆に戻す）は膠着の典型なので減点。
            // 直前に動かした駒をまた動かすだけの手も雑なシャッフルとして軽く減点
            if let (
                Some(ShogiMove::Board { from: pf, to: pt, .. }),
                ShogiMove::Board { from, to, .. },
            ) = (last_my_move, mv)
            {
                if from == pt && to == pf {
                    adjust -= params.backtrack_penalty;
                } else if from == pt {
                    adjust -= params.shuffle_penalty;
                }
            }
            let score = out.score() + adjust;
            scored.push((usi, mv, out, adjust, score));
        }

        // 2段目: 上位候補だけ相手の応手をサンプルして再評価。
        // gain 内の静的リスク項の depth2_replace 分を実測の期待損失で
        // 置き換えて（一致するなら無変化）、最終式を適用し直す
        scored.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
        let mut best: Option<(String, f64)> = None;
        for (i, (usi, mv, out, adjust, score)) in scored.into_iter().enumerate() {
            let final_score = if i < budget.depth2_top_k {
                let delta = depth2_delta(
                    view,
                    &mv,
                    &sample,
                    &known,
                    &my_capture_squares,
                    &my_touched_squares,
                    &params,
                    budget,
                    &mut *rng,
                );
                let gain2 = out.gain + params.depth2_replace * (out.risk_mean + delta);
                combine_score(gain2, out.p_legal, out.foul_cost) + adjust
            } else {
                score
            };
            if best.as_ref().is_none_or(|(_, s)| final_score > *s) {
                best = Some((usi, final_score));
            }
        }

        self.last_debug = Some(debug_summary(est, &sample, push));
        best.map(|(usi, _)| usi)
    }

    fn name(&self) -> &'static str {
        "estimator"
    }

    fn debug_state(&self) -> Option<serde_json::Value> {
        self.last_debug.clone()
    }
}

/// 記録用の推定サマリ: 粒子の健全性・ユニーク数・相手玉の位置分布（上位）。
/// 事後分析で「推定が外れていたのか、評価が悪かったのか」を切り分けるために残す
fn debug_summary(est: &Estimator, sample: &[(&Position, f64)], push: f64) -> serde_json::Value {
    let opp = est.my_color().other();
    let mut king_votes: HashMap<Coord, u32> = HashMap::new();
    for (pos, _) in sample {
        if let Some(sq) = pos.king_square(opp) {
            *king_votes.entry(sq).or_default() += 1;
        }
    }
    let mut top: Vec<(Coord, u32)> = king_votes.into_iter().collect();
    top.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    let n = sample.len().max(1) as f64;
    let opp_king_top: Vec<serde_json::Value> = top
        .iter()
        .take(3)
        .map(|(sq, votes)| {
            serde_json::json!({
                "sq": make_usi_square(*sq),
                "p": *votes as f64 / n,
            })
        })
        .collect();
    serde_json::json!({
        "healthy": est.healthy(),
        "unique_particles": sample.len(),
        "soft_particles": est.penalties().iter().filter(|&&p| p > 0).count(),
        "endgame_push": (push * 100.0).round() / 100.0,
        "opp_king_top": opp_king_top,
    })
}

/// 自分に見える範囲の候補手（foul_tried を除く）。bin/analyze の検証でも使う
pub fn candidate_moves(view: &PlayerView, foul_tried: &HashSet<String>) -> Vec<(String, ShogiMove)> {
    let color = view.your_color;
    let mut out = vec![];
    let push = |usi: String, out: &mut Vec<(String, ShogiMove)>| {
        if !foul_tried.contains(&usi) {
            if let Some(mv) = parse_usi(&usi) {
                out.push((usi, mv));
            }
        }
    };
    for piece in &view.your_pieces {
        let Some(from) = parse_usi_square(&piece.square) else {
            continue;
        };
        for to in move_targets(&view.your_pieces, piece, color) {
            match promotion_choice(piece.role, from, to, color) {
                Promotion::None => push(make_usi_move(from, to, false), &mut out),
                Promotion::Forced => push(make_usi_move(from, to, true), &mut out),
                Promotion::Optional => {
                    // 成れるなら成る（不成が有利な局面はまれなので候補を絞る）
                    push(make_usi_move(from, to, true), &mut out);
                }
            }
        }
    }
    for (&role, &count) in &view.your_hand {
        if count == 0 {
            continue;
        }
        for to in drop_targets(&view.your_pieces, role, color) {
            if let Some(usi) = make_usi_drop(role, to) {
                push(usi, &mut out);
            }
        }
    }
    out
}

/// 自玉のマス（PlayerView の自駒リストから引く）
fn king_square(view: &PlayerView) -> Option<Coord> {
    view.your_pieces
        .iter()
        .find(|p| p.role == Role::King)
        .and_then(|p| parse_usi_square(&p.square))
}

/// 王手されているとき、この手が王手を解消しうるか（自分に見える情報だけで判定）。
/// 解消手段は (a) 玉を動かす (b) 王手駒を取る (c) 合駒。王手駒の位置は不明でも
/// (b) の着地点は自玉に利きが通るマス（クイーンライン上か桂の利き元）、
/// (c) は玉と王手駒の間（クイーンライン上）に限られる。
/// どれにも該当しない手は王手放置で必ず反則になる
fn may_resolve_check(view: &PlayerView, mv: &ShogiMove) -> bool {
    let Some(king) = king_square(view) else {
        return true; // 玉が見つからないなら判定不能（除外しない）
    };
    let on_ray = |to: Coord| {
        let df = to.file - king.file;
        let dr = to.rank - king.rank;
        (df != 0 || dr != 0) && (df == 0 || dr == 0 || df.abs() == dr.abs())
    };
    // 相手の桂が自玉に利くマス（桂の王手は取るしかなく、合駒では防げない）
    let knight_source = |to: Coord| {
        let dr = match view.your_color {
            Color::Sente => -2, // 相手（後手）の桂は rank+2 へ利く → 利き元は rank-2 側
            Color::Gote => 2,
        };
        (to.file - king.file).abs() == 1 && to.rank - king.rank == dr
    };
    match *mv {
        ShogiMove::Board { from, to, .. } => {
            if from == king {
                return true; // 玉を動かす
            }
            on_ray(to) || knight_source(to)
        }
        // 打ちは駒を取れないので合駒（ライン上）のみ
        ShogiMove::Drop { to, .. } => on_ray(to),
    }
}

/// 王手中の p(合法) 補正係数。玉移動が最も解消しやすく、
/// 取り/合駒は王手駒の位置に当たっている必要があるので低め
fn in_check_prior(view: &PlayerView, mv: &ShogiMove) -> f64 {
    match *mv {
        ShogiMove::Board { from, .. } if Some(from) == king_square(view) => 0.5,
        _ => 0.25,
    }
}

/// 観測ゼロでも成り立つ p(合法) の事前確率。
/// 経路上の「中身の見えないマス」1つごとに空である確率 q を掛ける。
/// 打ちは着地点が空である確率 q（隠れた相手駒の上に打つのが典型的な反則源）
fn prior_legal(view: &PlayerView, mv: &ShogiMove, opp_board_n: f64) -> f64 {
    let my_n = view.your_pieces.len() as f64;
    let q = (1.0 - opp_board_n / (81.0 - my_n)).clamp(0.05, 1.0);
    match *mv {
        ShogiMove::Board { from, to, .. } => {
            let df = to.file - from.file;
            let dr = to.rank - from.rank;
            let aligned = df == 0 || dr == 0 || df.abs() == dr.abs();
            // 候補手は自駒には塞がれていないので、中間マスはすべて未知マス
            let unknown = if aligned {
                (df.abs().max(dr.abs()) - 1).max(0)
            } else {
                0 // 桂・1マス移動
            };
            q.powi(unknown as i32)
        }
        ShogiMove::Drop { .. } => q,
    }
}

/// 相手が位置を知っている自駒の地図（マス → 既知度 0.0〜1.0）。
///
/// 対人対局の分析（records/ 2026-07-08）より: 相手は (a) 自駒が死んだマス =
/// こちらの駒がいるマス、(b) 初期配置から動いていない駒、に当たりを付けて
/// 一方的に駒を回収してくる。ついたて将棋で相手に漏れる自駒の位置情報は
/// この2種類が主なので、露出リスクの重み付けに使う
/// - 1.0: 駒を取って位置が暴露し、以降動いていない駒
/// - home_knownness: 初期配置から一度も動いていない駒（相手は初期配置を知っている）
fn knownness_map(
    view: &PlayerView,
    log: &ObservationLog,
    home_knownness: f64,
) -> HashMap<Coord, f64> {
    let mut revealed: HashSet<Coord> = HashSet::new();
    let mut touched: HashSet<Coord> = HashSet::new();
    for e in log.events() {
        match e {
            Observation::MyMove { usi, captured, .. } => match parse_usi(usi) {
                Some(ShogiMove::Board { from, to, .. }) => {
                    revealed.remove(&from);
                    if captured.is_some() {
                        revealed.insert(to);
                    } else {
                        revealed.remove(&to);
                    }
                    touched.insert(from);
                    touched.insert(to);
                }
                Some(ShogiMove::Drop { to, .. }) => {
                    // 打った駒の位置は相手から見えない
                    revealed.remove(&to);
                    touched.insert(to);
                }
                None => {}
            },
            Observation::OpponentMoved {
                captured_my_piece_at: Some(sq),
                ..
            } => {
                if let Some(c) = parse_usi_square(sq) {
                    revealed.remove(&c);
                }
            }
            _ => {}
        }
    }

    let initial = Position::initial();
    let mut map = HashMap::new();
    for piece in &view.your_pieces {
        let Some(sq) = parse_usi_square(&piece.square) else {
            continue;
        };
        let k = if revealed.contains(&sq) {
            1.0
        } else if !touched.contains(&sq)
            && initial
                .piece_at(sq)
                .is_some_and(|p| p.color == view.your_color && p.role == piece.role)
        {
            home_knownness
        } else {
            0.0
        };
        if k > 0.0 {
            map.insert(sq, k);
        }
    }
    map
}

/// 敵陣のマスが（見えない駒に）守られている事前確率。
/// 粒子が枯渇・偏っていて守り駒を見落としていても、敵陣への単騎突入
/// （対人5局で歩→高価な駒の損な交換が9回）を抑えるための下限に使う
fn camp_defended_prior(to: Coord, me: Color, camp_scale: f64) -> f64 {
    let depth_from_back = match me {
        Color::Sente => to.rank,     // 相手（後手）の陣は rank 1..=3
        Color::Gote => 10 - to.rank, // 相手（先手）の陣は rank 7..=9
    };
    camp_scale
        * match depth_from_back {
            1 => 0.25,
            2 => 0.2,
            3 => 0.15,
            _ => 0.0,
        }
}

/// 候補手をユニーク粒子の加重平均で評価する（重み = ソフト救済の減衰）
fn evaluate(
    view: &PlayerView,
    mv: &ShogiMove,
    particles: &[(&Position, f64)],
    prior: f64,
    known: &HashMap<Coord, f64>,
    params: &EvalParams,
    budget: SearchBudget,
) -> EvalOut {
    let me = view.your_color;
    let opp = me.other();
    let mut legal = 0.0f64;
    let mut value_sum = 0.0;
    let mut risk_sum = 0.0;
    // 着地マスに敵駒がいた（=駒を取れた）粒子の重み。探索ボーナスの不一致度に使う
    let mut capture_hits = 0.0f64;
    // 王手になった粒子の重み。王探しの情報利得（判定が割れるほど価値）に使う
    let mut check_hits = 0.0f64;
    // 王周辺の圧力は粒子間の分散が小さいわりに計算が重い（9マス×利き走査）ので
    // 少数の粒子でだけ測って平均する（数は思考予算に比例）
    let pressure_samples = budget.pressure_samples;
    let mut pressure_sum = 0.0;
    let mut attack_sum = 0.0;
    let mut danger_sum = 0.0;
    let mut pressure_n = 0usize;
    // 圧力項もソフト粒子の重みで加重する（他の項と同じ扱い）
    let mut pressure_w_sum = 0.0f64;

    for (pos, w) in particles {
        let w = *w;
        if !pos.is_legal(mv) {
            continue;
        }
        legal += w;
        let mut v = 0.0;

        // 駒得（盤上価値で数える。成駒を取れば大きい）
        let mut captured_value = 0.0;
        if let ShogiMove::Board { to, .. } = *mv {
            if let Some(p) = pos.piece_at(to) {
                if p.color == opp {
                    captured_value = exchange_value(p.role);
                }
            }
        }
        v += captured_value;
        if captured_value > 0.0 {
            capture_hits += w;
        }

        let mut next = (*pos).clone();
        next.play_unchecked(mv);

        // 王手・詰み。ついたて将棋では王手された側は王手駒の位置が見えず
        // 手探りの反則をしやすい（反則10回で負け）ので、王手自体が得点源。
        // 相手の反則が溜まっているほど価値が上がる
        let gives_check = next.in_check(opp);
        if gives_check {
            v += params.check_bonus + params.check_foul_scale * f64::from(view.fouls.opponent);
            check_hits += w;
            if next.legal_moves().is_empty() {
                v += 1000.0; // 詰み（真の局面がこの粒子なら勝ち）
            }
        }

        // 取られリスクは「相手がこの駒の位置を知っているか」で重みを分ける。
        // 駒を取った直後は取られたマスが相手に通知される → 着手駒の位置は確実にバレて
        // いて、取り返しはほぼ実行される。それ以外の駒への当たりは相手から見えない
        // （推定はされうる）ので薄く見積もる
        let to = match *mv {
            ShogiMove::Board { to, .. } => to,
            ShogiMove::Drop { to, .. } => to,
        };
        // 相手が取れるのは1手で1枚なので、重み付きリスクの最大値だけを引く。
        // 敵陣への着手は「粒子には見えない守り駒がいる」事前確率を下限に敷く
        // （駒を取った直後は位置が確実にバレているので下限をフルに、静かな
        // 進入は相手からまだ見えないので薄く適用する）
        // 王手をかけた手は王手宣言で位置の仮説が絞られ、相手は反則覚悟の
        // 探り取りで回収に来る（人間の実証済み戦術）ので、露見扱いにする
        let mut mover_w = if captured_value > 0.0 {
            params.mover_w_captured
        } else {
            params.mover_w_quiet
        };
        if gives_check {
            mover_w += params.mover_check_extra;
        }
        let own_after = next
            .piece_at(to)
            .map(|p| exchange_value(p.role))
            .unwrap_or(0.0);
        let known_factor = if captured_value > 0.0 {
            1.0
        } else {
            params.camp_known_quiet
        };
        let mut floor = own_after * camp_defended_prior(to, me, params.camp_scale) * known_factor;
        if captured_value > 0.0 {
            // 取ったマスは相手に通知される。粒子に守りが見えなくても
            // 取り返しの残留リスクを敷く（= 等価な取りは安い駒で取る）
            floor = floor.max(own_after * params.capture_reveal_risk);
        }
        let mover_risk =
            mover_w * recapture_risk(&next, me, to, params.recapture_defended).max(floor);
        let hidden_risk = exposed_capture_risk(&next, me, Some(to), known, params);
        let risk = mover_risk.max(hidden_risk);
        v -= risk;
        risk_sum += w * risk;

        // 自分が敵駒に当たりを付けている価値（露出リスクの鏡像）。
        // 1手読みでは見えない「次の駒得」を作る手（大駒の頭への歩打ち等）に価値を与える
        v += params.threat_w * threat_value(&next, me);

        // 王の安全度と攻撃圧力（利き走査が重いので少数の粒子でだけ測って平均する）
        if pressure_n < pressure_samples {
            // 自玉の周囲に当たっている相手の利き（守り）
            pressure_sum += w * king_zone_pressure(&next, me, opp);
            // 相手玉の周囲に当たっている自分の利き（攻め）。王手にならない攻め駒の
            // 集結にも報酬を与える（王手/詰みボーナスだけだと攻めを組み立てない）
            attack_sum += w * king_zone_pressure(&next, opp, me);
            // 相手の持ち駒による王手打ちの受け入れ面積（対局実験の教訓:
            // 飛車を持たれた瞬間、玉への開いた直線はすべて即王手の入口になる）
            danger_sum += w * drop_check_danger(&next, me);
            pressure_w_sum += w;
            pressure_n += 1;
        }

        value_sum += w * v;
    }

    // 粒子の証拠と事前確率のブレンド（粒子ゼロなら事前そのもの）。
    // 粒子が退化している（実効重みが評価上限に届かない）ほど事前の重みを
    // 増やし、少数の偏った粒子への過信を防ぐ。ソフト粒子は重みぶんしか
    // 数えないので、退化度にも自然に反映される
    let n: f64 = particles.iter().map(|(_, w)| w).sum();
    let degen = 1.0 - (n / budget.eval_particles as f64).min(1.0);
    let w = params.prior_weight + params.prior_weight_degen * degen;
    let p_legal = (legal + prior * w) / (n + w);
    let expected = if legal > 0.0 {
        // 探索ボーナス: 着地マスの敵駒有無について粒子が割れているほど、
        // 指せば（取れても空でも）推定が絞れる。捕獲の期待値とは別の情報の価値
        let p_hit = capture_hits / legal;
        // 王探し: 王手判定が粒子間で割れる手は、指せば王手宣言の有無で
        // 玉位置仮説が絞れる（互角膠着で「玉が見つからない」を崩す勾配）
        let p_chk = check_hits / legal;
        // 攻め圧力は粒子の健全度でゲートする。退化した粒子は間違った玉位置に
        // 固まりやすく、「誰もいない場所への攻め」が加点され続ける
        // （対人実戦: 終盤の成桂の徘徊）。健全度が低いときは確実な項だけ残す
        let confidence = (n / budget.eval_particles as f64).min(1.0);
        value_sum / legal
            + params.info_bonus * p_hit * (1.0 - p_hit)
            + params.king_probe_bonus * p_chk * (1.0 - p_chk)
            + (params.attack_w * confidence * attack_sum
                - params.pressure_w * pressure_sum
                - params.hand_drop_w * danger_sum)
                / pressure_w_sum.max(1e-9)
    } else {
        0.0
    };

    // 反則コスト: 手番は失わないが反則数を消費する。残りが少ないほど急激に高価。
    // 序盤の「安い反則で情報を得る」は低コスト側で自然に許容される
    let fouls_left = (10u32.saturating_sub(view.fouls.you)).max(1) as f64;
    let foul_cost = params.foul_cost_base * (10.0 / fouls_left).powf(params.foul_cost_pow);

    // 前進の弱い事前バイアス（推定が薄い序盤に駒をぶつけに行くため）
    let advance_bias = match *mv {
        ShogiMove::Board { from, to, promote } => {
            let adv = match me {
                Color::Sente => (from.rank - to.rank) as f64,
                Color::Gote => (to.rank - from.rank) as f64,
            };
            params.advance_w * adv + if promote { params.promote_bias } else { 0.0 }
        }
        ShogiMove::Drop { .. } => params.drop_bias,
    };

    // 大駒を初期位置に置き続けるペナルティ（この手の後に残る枚数分）。
    // 動かす手だけペナルティが軽くなるので、展開への勾配になる
    let development = -params.big_home_penalty * big_home_after(view, mv);

    // 利き被覆（広い索敵網）と、成れる圏内への歩打ち（と金ポテンシャル）。
    // どちらも粒子に依存しない自明な情報だけで計算できる
    let coverage = params.coverage_w * coverage_after(view, mv);
    let probe = params.tokin_probe_w * tokin_probe(view, mv);

    let gain = expected + advance_bias + development + coverage + probe;
    EvalOut {
        gain,
        risk_mean: if legal > 0.0 { risk_sum / legal } else { 0.0 },
        p_legal,
        foul_cost,
    }
}

/// 2手読み: 候補手の後に相手の応手を粒子上でサンプルし、実測の期待損失
/// （露見度で割引した駒損 − 取り返し補償、被王手/被詰みペナルティ）を返す。
/// 静的リスク項（EvalOut::risk_mean）の置き換え先。値は「加点」方向（通常は負）
#[allow(clippy::too_many_arguments)]
fn depth2_delta(
    view: &PlayerView,
    mv: &ShogiMove,
    particles: &[(&Position, f64)],
    known: &HashMap<Coord, f64>,
    my_captures: &[Coord],
    my_touched: &[Coord],
    params: &EvalParams,
    budget: SearchBudget,
    rng: &mut impl rand::Rng,
) -> f64 {
    let me = view.your_color;
    let to = match *mv {
        ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
    };
    let mut sum = 0.0;
    let mut n = 0.0;
    for (pos, w) in particles.iter().take(budget.depth2_particles) {
        if !pos.is_legal(mv) {
            continue;
        }
        let mut next = (*pos).clone();
        let my_capture = next.play_unchecked(mv);
        let gives_check = next.in_check(me.other());
        n += w;
        // この候補手で駒を取った場合、捕獲通知でそのマスは相手に露見する。
        // 応手予測の既知地点に加えないと、最有力の応手である「即時の取り返し」に
        // PREDICT_RECAPTURE_BOOST が掛からず、捕獲手を過度に楽観視してしまう
        let extended;
        let known_for_reply: &[Coord] = if my_capture.is_some() {
            extended = [my_captures, &[to]].concat();
            &extended
        } else {
            my_captures
        };
        let Some(reply) = predict_opp_reply(&next, me, known_for_reply, my_touched, rng) else {
            continue; // 応手なし（詰み/ステイルメイト）は stage1 のボーナス側で評価済み
        };
        let reply_to = match reply {
            ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
        };
        let lost = next
            .piece_at(reply_to)
            .filter(|p| p.color == me)
            .map(|p| exchange_value(p.role))
            .unwrap_or(0.0);
        let mut next2 = next.clone();
        next2.play_unchecked(&reply);
        let mut d = 0.0;
        if lost > 0.0 {
            // 露見度スケール: 着手駒は stage1 の mover_w と同じ規則、
            // それ以外の駒は exposed_capture_risk と同じ knownness 重み。
            // 粒子上の応手はこちらの駒が全部見えてしまうので、実戦で相手が
            // その取りを狙える確率で割り引く（情報非対称の担保）
            let scale = if reply_to == to {
                let mut s = if my_capture.is_some() {
                    params.mover_w_captured
                } else {
                    params.mover_w_quiet
                };
                if gives_check {
                    s += params.mover_check_extra;
                }
                s
            } else {
                let knownness = known.get(&reply_to).copied().unwrap_or(0.0);
                params.exposed_base + params.exposed_known * knownness
            };
            // 取り返し補償: 応手の駒に自分の利きが残っていれば取り返せる
            let comp = if !next2.in_check(me) && next2.is_attacked(reply_to, me) {
                params.depth2_recap_discount
                    * next2
                        .piece_at(reply_to)
                        .map(|p| exchange_value(p.role))
                        .unwrap_or(0.0)
            } else {
                0.0
            };
            d -= scale * (lost - comp).max(0.0);
        }
        if next2.in_check(me) {
            d -= params.depth2_check_pen;
            if next2.legal_moves().is_empty() {
                d -= DEPTH2_MATE_PEN;
            }
        }
        sum += w * d;
    }
    if n > 0.0 { sum / n } else { 0.0 }
}

/// この手の後も初期位置に残っている自分の大駒（飛・角）の枚数
fn big_home_after(view: &PlayerView, mv: &ShogiMove) -> f64 {
    let (rook_home, bishop_home) = match view.your_color {
        Color::Sente => (Coord { file: 2, rank: 8 }, Coord { file: 8, rank: 8 }),
        Color::Gote => (Coord { file: 8, rank: 2 }, Coord { file: 2, rank: 2 }),
    };
    let from = match *mv {
        ShogiMove::Board { from, .. } => Some(from),
        ShogiMove::Drop { .. } => None,
    };
    let mut n = 0.0;
    for piece in &view.your_pieces {
        let Some(sq) = parse_usi_square(&piece.square) else {
            continue;
        };
        let home = (piece.role == Role::Rook && sq == rook_home)
            || (piece.role == Role::Bishop && sq == bishop_home);
        if home && from != Some(sq) {
            n += 1.0;
        }
    }
    n
}

/// 自分が当たりを付けている敵駒の最大価値（露出リスクの鏡像）。
/// 紐つき（相手が守っている）なら取ったときに取り返されるぶん割り引く。
/// 玉への当たりは王手であり合法性・王手ボーナス側で扱うので除く
fn threat_value(pos: &Position, me: Color) -> f64 {
    let opp = me.other();
    let mut best = 0.0f64;
    for (sq, piece) in pos.pieces() {
        if piece.color != opp || piece.role == Role::King {
            continue;
        }
        if !pos.is_attacked(sq, me) {
            continue;
        }
        let defended = pos.is_attacked(sq, opp);
        let gain = exchange_value(piece.role) * if defended { 0.45 } else { 1.0 };
        best = best.max(gain);
    }
    best
}

/// 着手駒（マス to にいる自駒）が次の相手番で取られるリスク。
/// 紐つきなら取り返せるぶん割り引く（相手のどの駒で取るかは不明なので近似）
fn recapture_risk(pos: &Position, me: Color, to: Coord, defended_discount: f64) -> f64 {
    let opp = me.other();
    let Some(piece) = pos.piece_at(to).filter(|p| p.color == me) else {
        return 0.0;
    };
    if piece.role == Role::King || !pos.is_attacked(to, opp) {
        return 0.0;
    }
    let defended = pos.is_attacked(to, me);
    exchange_value(piece.role) * if defended { defended_discount } else { 1.0 }
}

/// 次の相手番で失いうる駒の概算: 相手の利きが当たっている自駒の最大重み付き価値。
/// 自分の利きも当たっている（紐つき）なら取り返せるぶん割り引く。
/// 相手がその駒の位置を知っているほど（knownness_map）実際に取られやすいので
/// 重みを引き上げる。位置が漏れていない駒は従来通り薄く見積もる。
/// exclude（着手駒のマス）は recapture_risk 側で別の重みで数えるので除外する。
/// 合法手の完全列挙（ピン考慮など）はコストに見合わないので利きベースの近似
fn exposed_capture_risk(
    pos: &Position,
    me: Color,
    exclude: Option<Coord>,
    known: &HashMap<Coord, f64>,
    params: &EvalParams,
) -> f64 {
    let opp = me.other();
    let mut worst = 0.0f64;
    for (sq, piece) in pos.pieces() {
        if piece.color != me || piece.role == Role::King {
            continue; // 玉が当たっているなら王手なので合法性の側で処理される
        }
        if exclude == Some(sq) {
            continue;
        }
        if !pos.is_attacked(sq, opp) {
            continue;
        }
        let defended = pos.is_attacked(sq, me);
        let knownness = known.get(&sq).copied().unwrap_or(0.0);
        let weight = params.exposed_base + params.exposed_known * knownness;
        let loss = exchange_value(piece.role)
            * if defended { params.exposed_defended } else { 1.0 }
            * weight;
        worst = worst.max(loss);
    }
    worst
}

/// 相手の持ち駒による「王手打ちの受け入れ面積」。
/// 相手の持ち駒はこの粒子上で正確に分かる（取られた自駒 − 打たれた駒）。
/// - 飛: 玉からの縦横の空き直線の長さ（その各マスが王手打ちの入口）
/// - 角: 斜めの空き直線の長さ
/// - 香: 相手の香が王手できる側の1直線
/// - 金/銀: 玉の隣接空きマス（打てば即王手）
/// - 歩: 玉頭の1マス
/// 持ち駒が空ならゼロ = 居玉そのものは咎めない
fn drop_check_danger(pos: &Position, me: Color) -> f64 {
    let Some(king) = pos.king_square(me) else {
        return 0.0;
    };
    let opp = me.other();
    let on_board = |c: &Coord| (1..=9).contains(&c.file) && (1..=9).contains(&c.rank);
    let ray_len = |df: i8, dr: i8| -> f64 {
        let mut n = 0;
        let mut c = Coord { file: king.file + df, rank: king.rank + dr };
        while on_board(&c) && pos.piece_at(c).is_none() {
            n += 1;
            c = Coord { file: c.file + df, rank: c.rank + dr };
        }
        n as f64
    };

    let mut danger = 0.0;
    if pos.hand_count(opp, Role::Rook) > 0 {
        danger += ray_len(1, 0) + ray_len(-1, 0) + ray_len(0, 1) + ray_len(0, -1);
    }
    if pos.hand_count(opp, Role::Bishop) > 0 {
        danger += ray_len(1, 1) + ray_len(1, -1) + ray_len(-1, 1) + ray_len(-1, -1);
    }
    // 相手の香・歩は「相手から見て前へ」利くので、自玉側から見ると
    // 自分の陣の奥方向の直線・玉頭が入口になる
    let toward = if me == Color::Sente { -1 } else { 1 };
    if pos.hand_count(opp, Role::Lance) > 0 {
        danger += ray_len(0, toward);
    }
    if pos.hand_count(opp, Role::Pawn) > 0 {
        let head = Coord { file: king.file, rank: king.rank + toward };
        if on_board(&head) && pos.piece_at(head).is_none() {
            danger += 1.0;
        }
    }
    let generals =
        pos.hand_count(opp, Role::Gold) > 0 || pos.hand_count(opp, Role::Silver) > 0;
    if generals {
        let mut air = 0.0;
        for df in -1..=1i8 {
            for dr in -1..=1i8 {
                if df == 0 && dr == 0 {
                    continue;
                }
                let c = Coord { file: king.file + df, rank: king.rank + dr };
                if on_board(&c) && pos.piece_at(c).is_none() {
                    air += 0.5;
                }
            }
        }
        danger += air;
    }
    danger
}

/// owner 玉の周囲8マス（と玉のマス）に当たっている by 側の利きの数
fn king_zone_pressure(pos: &Position, owner: Color, by: Color) -> f64 {
    let Some(king) = pos.king_square(owner) else {
        return 0.0;
    };
    let mut pressure = 0;
    for df in -1..=1i8 {
        for dr in -1..=1i8 {
            let c = crate::board::Coord {
                file: king.file + df,
                rank: king.rank + dr,
            };
            if (1..=9).contains(&c.file)
                && (1..=9).contains(&c.rank)
                && pos.is_attacked(c, by)
            {
                pressure += 1;
            }
        }
    }
    pressure as f64
}

