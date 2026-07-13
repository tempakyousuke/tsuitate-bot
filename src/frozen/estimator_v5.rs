//! estimator の凍結版 v5（2026-07-10 凍結）。
//!
//! **このファイルは編集しない**（frozen/mod.rs の運用ルール参照）。
//! v4 との差分:
//! - 王手ソルバー: 王手駒の（マス,駒種）仮説列挙＋反則によるベイズ消去＋
//!   粒子投票で回避手の解消確率を出す。粒子枯渇時も機能する
//! - 評価式を min(期待値, p×期待値) − (1−p)×反則コスト に修正
//!   （期待値が負のとき低合法確率の手を選好する欠陥の修正）
//! - 相手手の事前分布を対人55局の条件付き最尤推定に置き換え
//!   （前進+0.144 / 成り+1.413 / 打ち−1.411 / 既知駒への当たり+0.488 /
//!   初期配置の駒への当たり+0.581）。41手以降の粒子健全性 12%→24%
//! - 思考予算増額（リプレイ320回・500/900ms・粒子512）と退化適応の事前重み
//! - 駒探し項（threat_w / info_bonus）と大駒の展開ペナルティ（big_home_penalty）
//! - 序盤定跡ブック（所有者登録の13ライン、凍結時点の joseki.json を焼き込み）
//! - 露出評価（knownness・敵陣下限）は実装したがアブレーションで有害と
//!   判明し既定無効（camp_scale=0, exposed_known=0, home_knownness=0）
//!
//! 確定強度（各200局、2026-07-09 CIガントレット、手数上限200）:
//! vs estimator_v4 70.5%±7.9% / vs estimator_v3 78.7%±7.1% / vs estimator_v2 68.7%±8.5%
//! 平均反則 3.7〜4.2（相手側 6.3〜6.5）
#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::board::{
    Coord, Promotion, drop_targets, make_usi_drop, make_usi_move, move_targets, parse_usi_square,
    promotion_choice,
};
use crate::model::GameModel;
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{Color, PlayerView, Role};
use crate::shogi::{Position, ShogiMove, parse_usi, piece_value, unpromote_role};
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

struct Estimator {
    my_color: Color,
    particles: Vec<Position>,
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
    fn new(my_color: Color) -> Self {
        Estimator::with_seed(my_color, rand::rng().random())
    }

    fn with_seed(my_color: Color, seed: u64) -> Self {
        Estimator {
            my_color,
            particles: vec![Position::initial(); TARGET_PARTICLES],
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

    fn my_color(&self) -> Color {
        self.my_color
    }

    /// 現在の粒子集合。空なら推定は信頼できない（呼び出し側でフォールバック）
    fn particles(&self) -> &[Position] {
        &self.particles
    }

    fn healthy(&self) -> bool {
        self.healthy && !self.particles.is_empty()
    }

    /// ログの未消化イベントを取り込み、粒子を前進・棄却・補充する
    fn update(&mut self, log: &ObservationLog) {
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
        let mut survivors = Vec::with_capacity(self.particles.len());
        let particles = std::mem::take(&mut self.particles);
        for mut pos in particles {
            let ok = match constraint {
                Constraint::MyMove {
                    mv,
                    captured,
                    gives_check,
                } => apply_my_move(&mut pos, my_color, mv, *captured, *gives_check),
                Constraint::MyFoul { mv } => foul_consistent(&pos, my_color, mv),
                Constraint::OppMove {
                    captured_at,
                    gives_check,
                } => sample_opp_move(
                    &mut pos,
                    my_color,
                    *captured_at,
                    *gives_check,
                    &self.my_capture_sq,
                    &self.my_touched_sq,
                    &mut self.rng,
                ),
            };
            if ok {
                survivors.push(pos);
            }
        }
        self.particles = survivors;
    }

    /// 粒子が減っていたら、制約列のリプレイ（多様性）と生存粒子の複製（安価）で補充。
    /// 枯渇時は時間予算いっぱいまでリプレイで粘る（観測が正しい限り整合局面は必ず存在する）。
    /// リプレイ1回のコストは手数に比例するため、回数と時間の両方で打ち切る
    fn replenish(&mut self) {
        let start = std::time::Instant::now();
        let regen_deadline = start + std::time::Duration::from_millis(500);
        if self.particles.len() < TARGET_PARTICLES {
            for _ in 0..REGEN_ATTEMPTS {
                if self.particles.len() >= TARGET_PARTICLES
                    || std::time::Instant::now() > regen_deadline
                {
                    break;
                }
                if let Some(pos) = self.replay_once() {
                    self.particles.push(pos);
                }
            }
        }
        let deadline = start + std::time::Duration::from_millis(900);
        while self.particles.is_empty() && std::time::Instant::now() < deadline {
            if let Some(pos) = self.replay_once() {
                self.particles.push(pos);
            }
        }
        // ラッチしない: 粒子が戻れば健全に戻る（呼び出し側は毎手 update する）
        self.healthy = !self.particles.is_empty();
        if self.particles.is_empty() {
            return;
        }
        while self.particles.len() < TARGET_PARTICLES {
            let i = self.rng.random_range(0..self.particles.len());
            let clone = self.particles[i].clone();
            self.particles.push(clone);
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
                } => apply_my_move(&mut pos, self.my_color, mv, *captured, *gives_check),
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
                        *gives_check,
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

/// 受理された自分の手を粒子に適用する。粒子と観測が矛盾したら false
fn apply_my_move(
    pos: &mut Position,
    my_color: Color,
    mv: &ShogiMove,
    captured: Option<Role>,
    gives_check: bool,
) -> bool {
    if pos.turn() != my_color || !pos.is_legal(mv) {
        return false;
    }
    let actual = pos.play_unchecked(mv).map(unpromote_role);
    if actual != captured {
        return false;
    }
    pos.in_check(my_color.other()) == gives_check
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
/// - known_squares: 自分が駒を取ったマス（相手は自駒がそこで死んだことを知っている）
/// - my_touched: 自分の手が触れたマス（初期配置のまま動いていない自駒の判定用。
///   相手はそれらを推論で狙ってくる = 飛車頭への歩打ち等）
fn sample_opp_move(
    pos: &mut Position,
    my_color: Color,
    captured_at: Option<Coord>,
    gives_check: bool,
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
        if next.in_check(my_color) != gives_check {
            continue;
        }
        let threat_known = newly_threatens(pos, &next, &mv, known_squares);
        let threat_home = newly_threatens(pos, &next, &mv, &homes);
        candidates.push((mv, opp_move_weight(opp, &mv, threat_known, threat_home)));
    }
    let Some(chosen) = weighted_choice(&candidates, rng) else {
        return false;
    };
    pos.play_unchecked(&chosen);
    true
}

/// 相手の手の尤度づけ。対人55局の条件付き最尤推定（bin/fit_opp, 2026-07-09、
/// 駒単位threat定義）: パープレキシティ 27.7（旧手調整）→ 24.9。
/// 駒取り・王手の有無は観測との整合ですでに絞り込まれているため、
/// 事前分布には「観測クラス内で判別できる特徴量」だけが現れる
fn opp_move_weight(opp: Color, mv: &ShogiMove, threat_known: bool, threat_home: bool) -> f64 {
    let mut s = 0.0;
    match *mv {
        ShogiMove::Board { from, to, promote } => {
            let advance = match opp {
                Color::Sente => (from.rank - to.rank) as f64,
                Color::Gote => (to.rank - from.rank) as f64,
            };
            s += 0.144 * advance;
            if promote {
                s += 1.413;
            }
        }
        ShogiMove::Drop { .. } => s += -1.411,
    }
    if threat_known {
        s += 0.488;
    }
    if threat_home {
        s += 0.581;
    }
    s.exp()
}

fn weighted_choice(candidates: &[(ShogiMove, f64)], rng: &mut StdRng) -> Option<ShogiMove> {
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

struct CheckSolver {
    /// 自駒＋持ち駒だけを置いたスパース盤面（手番=自分）。仮説の駒を載せて使う
    base: Position,
    my_color: Color,
    hypotheses: Vec<Hypothesis>,
}

impl CheckSolver {
    /// 王手中の view から作る。自玉が見つからない等で推論できなければ None
    fn new(
        view: &PlayerView,
        particles: &[&Position],
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
                    continue; // 自駒のあるマスに王手駒はいない
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

    /// 粒子中の実際の王手駒に投票させる（粒子が健全なら仮説が鋭くなる）
    fn vote_by_particles(&mut self, particles: &[&Position]) {
        let opp = self.my_color.other();
        let mut voters = 0usize;
        let mut votes: Vec<usize> = vec![0; self.hypotheses.len()];
        for pos in particles {
            if !pos.in_check(self.my_color) {
                continue; // 王手を反映していない粒子は情報にならない
            }
            voters += 1;
            for (i, h) in self.hypotheses.iter().enumerate() {
                if pos.piece_at(h.square)
                    .is_some_and(|p| p.color == opp && p.role == h.role)
                {
                    // 粒子上でその駒が実際に王を攻撃しているかまでは見ない
                    // （enumerate 済みの仮説は自駒配置的に攻撃可能）
                    votes[i] += 1;
                }
            }
        }
        if voters == 0 {
            return;
        }
        for (h, &v) in self.hypotheses.iter_mut().zip(&votes) {
            h.weight *= 1.0 + PARTICLE_VOTE_W * (v as f64 / voters as f64);
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
    fn resolve_probability(&mut self, mv: &ShogiMove) -> f64 {
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

/// 凍結時点の joseki.json を焼き込んだ定跡ライン（凍結版は挙動固定のため
/// ファイルを読まない）
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

struct OpeningBook {
    /// 対局開始時に選んだライン（自色向けにミラー済み）
    line: Vec<String>,
    /// ブックから抜けたら true（以後戻らない）
    exited: bool,
}

impl OpeningBook {
    fn new(my_color: Color) -> Self {
        let all = lines();
        Self::with_index(my_color, rand::rng().random_range(0..all.len()))
    }

    /// 指定インデックスのライン（共通乱数法用の追加。選択分布は new と同じ一様）
    fn with_index(my_color: Color, index: usize) -> Self {
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

    /// ブックの次の一手。None ならブックを抜けた（通常思考へ）
    fn next(
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
// 戦略（strategy.rs の EstimatorStrategy のコピー）
// ---------------------------------------------------------------------------

/// 評価に使う粒子数の上限（思考時間の予算。粒子は estimator 側で最大400）。
/// フィッシャー300秒+3秒に対し1手1〜2秒が目安。96粒子で平均370ms程度だったので
/// 精度側（反則率の低下）に予算を振る
const EVAL_PARTICLES: usize = 192;

/// evaluate() まわりの調整可能パラメータ。Default が現行の手調整値。
/// bin/tune.rs の SPSA がこれを最適化する（凍結版は各自のコピーを持ち依存しない）
#[derive(Debug, Clone)]
struct EvalParams {
    /// 王手ボーナスの基本値
    pub check_bonus: f64,
    /// 王手ボーナスの相手反則数スケール
    pub check_foul_scale: f64,
    /// 着手駒の取られリスク重み（駒を取った直後 = 位置がバレている）
    pub mover_w_captured: f64,
    /// 着手駒の取られリスク重み（静かな手）
    pub mover_w_quiet: f64,
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
    /// 手戻り減点
    pub backtrack_penalty: f64,
}

impl Default for EvalParams {
    fn default() -> Self {
        EvalParams {
            check_bonus: 0.9,
            check_foul_scale: 0.12,
            mover_w_captured: 0.9,
            mover_w_quiet: 0.45,
            camp_known_quiet: 0.35,
            // 露出評価（knownness）と敵陣リスク下限は既定で無効。
            // アブレーション（2026-07-09）で vs v4 40.7% → 無効化で 56.1% と
            // アリーナで明確に有害、対人50局でも只取られは改善しなかった。
            // 器は残すので SPSA（bin/tune）が非ゼロの最適値を探すことはできる
            camp_scale: 0.0,
            exposed_base: 0.35,
            exposed_known: 0.0,
            home_knownness: 0.0,
            recapture_defended: 0.45,
            exposed_defended: 0.4,
            attack_w: 0.12,
            pressure_w: 0.2,
            foul_cost_base: 1.5,
            foul_cost_pow: 1.5,
            advance_w: 0.05,
            promote_bias: 0.1,
            drop_bias: -0.05,
            prior_weight: 4.0,
            prior_weight_degen: 8.0,
            threat_w: 0.25,
            info_bonus: 0.6,
            big_home_penalty: 0.25,
            backtrack_penalty: 0.35,
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
pub struct EstimatorV5 {
    est: Option<Estimator>,
    book: Option<OpeningBook>,
    params: EvalParams,
    /// Some なら推定器・定跡選択・タイブレークをこのシードから決定論化する。
    /// 凍結後の唯一の追加（2026-07-13）: SPSA の共通乱数法（f+/f− の対局条件
    /// ペアリング）のためのシード注入で、挙動の分布は変えない
    seed: Option<u64>,
    tiebreak: Option<rand::rngs::StdRng>,
}

impl EstimatorV5 {
    pub fn new() -> Self {
        Self::with_params(EvalParams::default())
    }

    /// シードつきで作る（bin/tune の共通乱数法用。挙動分布は new と同じ）
    pub fn with_seed(seed: u64) -> Self {
        let mut s = Self::with_params(EvalParams::default());
        s.seed = Some(seed);
        s
    }

    /// パラメータを差し替えて作る（bin/tune.rs のSPSA評価用）
    fn with_params(params: EvalParams) -> Self {
        EstimatorV5 {
            est: None,
            book: None,
            params,
            seed: None,
            tiebreak: None,
        }
    }
}

impl Default for EstimatorV5 {
    fn default() -> Self {
        Self::new()
    }
}

impl Strategy for EstimatorV5 {
    fn choose(
        &mut self,
        view: &PlayerView,
        log: &ObservationLog,
        foul_tried: &HashSet<String>,
    ) -> Option<String> {
        let seed = self.seed;
        let est = self.est.get_or_insert_with(|| match seed {
            Some(s) => Estimator::with_seed(view.your_color, s),
            None => Estimator::new(view.your_color),
        });
        est.update(log);

        // 序盤定跡（静かな間だけ）。ブック中も推定器の update は回して粒子を保つ
        let book = self.book.get_or_insert_with(|| match seed {
            Some(s) => {
                OpeningBook::with_index(view.your_color, (s % lines().len() as u64) as usize)
            }
            None => OpeningBook::new(view.your_color),
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
        // 粒子が完全に枯渇していても、事前確率だけで安全側の評価が成り立つ
        let mut seen = HashSet::new();
        let mut sample: Vec<&Position> = vec![];
        for pos in est.particles() {
            if sample.len() >= EVAL_PARTICLES {
                break;
            }
            if seen.insert(pos.fingerprint()) {
                sample.push(pos);
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

        let rng = self.tiebreak.get_or_insert_with(|| {
            use rand::SeedableRng;
            match seed {
                Some(s) => StdRng::seed_from_u64(s ^ 0x7EA1_B00C),
                None => StdRng::seed_from_u64(rand::rng().random()),
            }
        });
        let mut best: Option<(String, f64)> = None;
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
            let mut score = evaluate(view, &mv, &sample, prior, &known, &self.params)
                + rng.random_range(0.0..0.01);
            // 手戻り（直前の手をそのまま逆に戻す）は膠着の典型なので減点。
            // 手数上限の引き分けを崩す側に倒す
            if let (
                Some(ShogiMove::Board { from: pf, to: pt, .. }),
                ShogiMove::Board { from, to, .. },
            ) = (last_my_move, mv)
            {
                if from == pt && to == pf {
                    score -= self.params.backtrack_penalty;
                }
            }
            if best.as_ref().is_none_or(|(_, s)| score > *s) {
                best = Some((usi, score));
            }
        }

        best.map(|(usi, _)| usi)
    }

    fn name(&self) -> &'static str {
        "estimator_v5"
    }
}

/// 自分に見える範囲の候補手（foul_tried を除く）。bin/analyze の検証でも使う
fn candidate_moves(view: &PlayerView, foul_tried: &HashSet<String>) -> Vec<(String, ShogiMove)> {
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

/// 候補手をユニーク粒子の平均で評価する
fn evaluate(
    view: &PlayerView,
    mv: &ShogiMove,
    particles: &[&Position],
    prior: f64,
    known: &HashMap<Coord, f64>,
    params: &EvalParams,
) -> f64 {
    let me = view.your_color;
    let opp = me.other();
    let mut legal = 0usize;
    let mut value_sum = 0.0;
    // 着地マスに敵駒がいた（=駒を取れた）粒子数。探索ボーナスの不一致度に使う
    let mut capture_hits = 0usize;
    // 王周辺の圧力は粒子間の分散が小さいわりに計算が重い（9マス×利き走査）ので
    // 少数の粒子でだけ測って平均する
    const PRESSURE_SAMPLES: usize = 16;
    let mut pressure_sum = 0.0;
    let mut attack_sum = 0.0;
    let mut pressure_n = 0usize;

    for pos in particles {
        if !pos.is_legal(mv) {
            continue;
        }
        legal += 1;
        let mut v = 0.0;

        // 駒得（盤上価値で数える。成駒を取れば大きい）
        let mut captured_value = 0.0;
        if let ShogiMove::Board { to, .. } = *mv {
            if let Some(p) = pos.piece_at(to) {
                if p.color == opp {
                    captured_value = piece_value(p.role);
                }
            }
        }
        v += captured_value;
        if captured_value > 0.0 {
            capture_hits += 1;
        }

        let mut next = (*pos).clone();
        next.play_unchecked(mv);

        // 王手・詰み。ついたて将棋では王手された側は王手駒の位置が見えず
        // 手探りの反則をしやすい（反則10回で負け）ので、王手自体が得点源。
        // 相手の反則が溜まっているほど価値が上がる
        if next.in_check(opp) {
            v += params.check_bonus + params.check_foul_scale * f64::from(view.fouls.opponent);
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
        let mover_w = if captured_value > 0.0 {
            params.mover_w_captured
        } else {
            params.mover_w_quiet
        };
        let own_after = next
            .piece_at(to)
            .map(|p| piece_value(p.role))
            .unwrap_or(0.0);
        let known_factor = if captured_value > 0.0 {
            1.0
        } else {
            params.camp_known_quiet
        };
        let floor = own_after * camp_defended_prior(to, me, params.camp_scale) * known_factor;
        let mover_risk =
            mover_w * recapture_risk(&next, me, to, params.recapture_defended).max(floor);
        let hidden_risk = exposed_capture_risk(&next, me, Some(to), known, params);
        v -= mover_risk.max(hidden_risk);

        // 自分が敵駒に当たりを付けている価値（露出リスクの鏡像）。
        // 1手読みでは見えない「次の駒得」を作る手（大駒の頭への歩打ち等）に価値を与える
        v += params.threat_w * threat_value(&next, me);

        // 王の安全度と攻撃圧力（利き走査が重いので少数の粒子でだけ測って平均する）
        if pressure_n < PRESSURE_SAMPLES {
            // 自玉の周囲に当たっている相手の利き（守り）
            pressure_sum += king_zone_pressure(&next, me, opp);
            // 相手玉の周囲に当たっている自分の利き（攻め）。王手にならない攻め駒の
            // 集結にも報酬を与える（王手/詰みボーナスだけだと攻めを組み立てない）
            attack_sum += king_zone_pressure(&next, opp, me);
            pressure_n += 1;
        }

        value_sum += v;
    }

    // 粒子の証拠と事前確率のブレンド（粒子ゼロなら事前そのもの）。
    // 粒子が退化している（ユニーク数が評価上限に届かない）ほど事前の重みを
    // 増やし、少数の偏った粒子への過信を防ぐ
    let n = particles.len() as f64;
    let degen = 1.0 - (n / EVAL_PARTICLES as f64).min(1.0);
    let w = params.prior_weight + params.prior_weight_degen * degen;
    let p_legal = (legal as f64 + prior * w) / (n + w);
    let expected = if legal > 0 {
        // 探索ボーナス: 着地マスの敵駒有無について粒子が割れているほど、
        // 指せば（取れても空でも）推定が絞れる。捕獲の期待値とは別の情報の価値
        let p_hit = capture_hits as f64 / legal as f64;
        value_sum / legal as f64
            + params.info_bonus * p_hit * (1.0 - p_hit)
            + (params.attack_w * attack_sum - params.pressure_w * pressure_sum)
                / pressure_n.max(1) as f64
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

    // 期待値が負の手を p_legal で割り引かない（min の形）。
    // 割り引くと「合法確率が低いほどスコアが高い」= わざと反則に寄る手が
    // 選ばれてしまう。反則しても手番は残るので悪い局面からは逃げられず、
    // 反則の価値は「次善手の価値 − 反則コスト」でしかない
    let gain = expected + advance_bias + development;
    (p_legal * gain).min(gain) - (1.0 - p_legal) * foul_cost
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
        let gain = piece_value(piece.role) * if defended { 0.45 } else { 1.0 };
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
    piece_value(piece.role) * if defended { defended_discount } else { 1.0 }
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
        let loss = piece_value(piece.role)
            * if defended { params.exposed_defended } else { 1.0 }
            * weight;
        worst = worst.max(loss);
    }
    worst
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

