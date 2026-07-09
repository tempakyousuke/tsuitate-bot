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

    pub fn my_color(&self) -> Color {
        self.my_color
    }

    /// 現在の粒子集合。空なら推定は信頼できない（呼び出し側でフォールバック）
    pub fn particles(&self) -> &[Position] {
        &self.particles
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
        est.replenish();
        assert!(est.healthy(), "リプレイで再生成できるはず");
        assert_eq!(est.particles().len(), TARGET_PARTICLES);
    }
}
