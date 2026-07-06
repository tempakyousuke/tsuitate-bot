//! estimator の凍結版 v2（2026-07-06 凍結）。
//!
//! **このファイルは編集しない**（frozen/mod.rs の運用ルール参照）。
//! v1 との差分: 王手回避の修正 — you_in_check のとき王手を解消しえない手
//! （玉移動・自玉ラインへの着地・桂の利き元への移動のどれでもない手）を
//! 候補から除外し、王手中の事前確率を玉移動優先に補正する。
//! 粒子枯渇時のフォールバックが王手を無視して反則を連発する問題への対処。
//!
//! 参考強度: vs estimator_v1 80.9%±5.7%（200局、平均反則 3.75 vs 8.38、2026-07-06）

use std::collections::HashSet;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::board::{
    Coord, Promotion, drop_targets, make_usi_drop, make_usi_move, move_targets, parse_usi_square,
    promotion_choice,
};
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{Color, PlayerView, Role};
use crate::shogi::{Position, ShogiMove, parse_usi, piece_value, unpromote_role};
use crate::strategy::Strategy;

// ---------------------------------------------------------------------------
// 推定器（estimator.rs のコピー）
// ---------------------------------------------------------------------------

/// 粒子の目標数。1手あたりの計算量はこれ*候補手数に比例する
const TARGET_PARTICLES: usize = 400;
/// 1回の update での再生成リプレイ試行の上限（時間予算の担保）。
const REGEN_ATTEMPTS: usize = 120;

/// 観測列を推定に使える形に正規化した制約
#[derive(Debug, Clone)]
enum Constraint {
    MyMove {
        mv: ShogiMove,
        captured: Option<Role>,
        gives_check: bool,
    },
    MyFoul {
        mv: ShogiMove,
    },
    OppMove {
        captured_at: Option<Coord>,
        gives_check: bool,
    },
}

struct Estimator {
    my_color: Color,
    particles: Vec<Position>,
    constraints: Vec<Constraint>,
    cursor: usize,
    healthy: bool,
    rng: StdRng,
}

impl Estimator {
    fn new(my_color: Color) -> Self {
        Estimator {
            my_color,
            particles: vec![Position::initial(); TARGET_PARTICLES],
            constraints: vec![],
            cursor: 0,
            healthy: true,
            rng: StdRng::seed_from_u64(rand::rng().random()),
        }
    }

    fn particles(&self) -> &[Position] {
        &self.particles
    }

    fn update(&mut self, log: &ObservationLog) {
        let events = log.events();
        while self.cursor < events.len() {
            let (constraint, consumed) = self.normalize(&events[self.cursor..]);
            self.cursor += consumed;
            let Some(constraint) = constraint else {
                continue;
            };
            self.apply_constraint(&constraint);
            self.constraints.push(constraint);
        }
        self.replenish();
    }

    fn normalize(&self, events: &[Observation]) -> (Option<Constraint>, usize) {
        let head = &events[0];
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
                } => sample_opp_move(&mut pos, my_color, *captured_at, *gives_check, &mut self.rng),
            };
            if ok {
                survivors.push(pos);
            }
        }
        self.particles = survivors;
    }

    fn replenish(&mut self) {
        let start = std::time::Instant::now();
        let regen_deadline = start + std::time::Duration::from_millis(250);
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
        let deadline = start + std::time::Duration::from_millis(450);
        while self.particles.is_empty() && std::time::Instant::now() < deadline {
            if let Some(pos) = self.replay_once() {
                self.particles.push(pos);
            }
        }
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

    fn replay_once(&mut self) -> Option<Position> {
        let mut pos = Position::initial();
        for constraint in &self.constraints {
            let ok = match constraint {
                Constraint::MyMove {
                    mv,
                    captured,
                    gives_check,
                } => apply_my_move(&mut pos, self.my_color, mv, *captured, *gives_check),
                Constraint::MyFoul { mv } => foul_consistent(&pos, self.my_color, mv),
                Constraint::OppMove {
                    captured_at,
                    gives_check,
                } => sample_opp_move(
                    &mut pos,
                    self.my_color,
                    *captured_at,
                    *gives_check,
                    &mut self.rng,
                ),
            };
            if !ok {
                return None;
            }
        }
        Some(pos)
    }
}

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

fn foul_consistent(pos: &Position, my_color: Color, mv: &ShogiMove) -> bool {
    pos.turn() == my_color && !pos.is_legal(mv)
}

fn sample_opp_move(
    pos: &mut Position,
    my_color: Color,
    captured_at: Option<Coord>,
    gives_check: bool,
    rng: &mut StdRng,
) -> bool {
    let opp = my_color.other();
    if pos.turn() != opp {
        return false;
    }
    let mut candidates: Vec<(ShogiMove, f64)> = vec![];
    for mv in pos.legal_moves() {
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
        candidates.push((mv, opp_move_weight(pos, opp, &mv, to_capture.map(|(_, r)| r))));
    }
    let Some(chosen) = weighted_choice(&candidates, rng) else {
        return false;
    };
    pos.play_unchecked(&chosen);
    true
}

fn opp_move_weight(_pos: &Position, opp: Color, mv: &ShogiMove, captured: Option<Role>) -> f64 {
    let mut w = 1.0;
    match *mv {
        ShogiMove::Board { from, to, promote } => {
            let advance = match opp {
                Color::Sente => (from.rank - to.rank) as f64,
                Color::Gote => (to.rank - from.rank) as f64,
            };
            w += 0.25 * advance.max(0.0);
            if promote {
                w += 1.0;
            }
        }
        ShogiMove::Drop { .. } => w *= 0.5,
    }
    if let Some(role) = captured {
        w += 0.8 * piece_value(role);
    }
    w.max(0.05)
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
// 戦略（strategy.rs の EstimatorStrategy のコピー）
// ---------------------------------------------------------------------------

/// 評価に使う粒子数の上限（思考時間の予算。粒子は推定器側で最大400）
const EVAL_PARTICLES: usize = 96;

/// 事前確率の重み（擬似観測数）。粒子が少ない・偏っているときほど事前が効く
const PRIOR_WEIGHT: f64 = 4.0;

/// estimator v2（凍結）。観測履歴から相手局面を推定し、候補手を粒子平均で評価する
pub struct EstimatorV2 {
    est: Option<Estimator>,
}

impl EstimatorV2 {
    pub fn new() -> Self {
        EstimatorV2 { est: None }
    }
}

impl Default for EstimatorV2 {
    fn default() -> Self {
        Self::new()
    }
}

impl Strategy for EstimatorV2 {
    fn choose(
        &mut self,
        view: &PlayerView,
        log: &ObservationLog,
        foul_tried: &HashSet<String>,
    ) -> Option<String> {
        let est = self
            .est
            .get_or_insert_with(|| Estimator::new(view.your_color));
        est.update(log);

        let mut candidates = candidate_moves(view, foul_tried);
        if view.you_in_check {
            // 王手中: 解消しえない手は（王手駒がどこにいても）王手放置で必ず反則に
            // なるので候補から外す。全滅したら元の候補に戻す
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

        let my_captures = log
            .events()
            .iter()
            .filter(|e| matches!(e, Observation::MyMove { captured: Some(_), .. }))
            .count();
        let opp_board_n = (20 - my_captures.min(19)) as f64;

        let mut rng = rand::rng();
        let mut best: Option<(String, f64)> = None;
        for (usi, mv) in candidates {
            let mut prior = prior_legal(view, &mv, opp_board_n);
            if view.you_in_check {
                prior *= in_check_prior(view, &mv);
            }
            let score = evaluate(view, &mv, &sample, prior) + rng.random_range(0.0..0.01);
            if best.as_ref().is_none_or(|(_, s)| score > *s) {
                best = Some((usi, score));
            }
        }
        best.map(|(usi, _)| usi)
    }

    fn name(&self) -> &'static str {
        "estimator_v2"
    }
}

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

/// 王手されているとき、この手が王手を解消しうるか（自分に見える情報だけで判定）
fn may_resolve_check(view: &PlayerView, mv: &ShogiMove) -> bool {
    let Some(king) = king_square(view) else {
        return true;
    };
    let on_ray = |to: Coord| {
        let df = to.file - king.file;
        let dr = to.rank - king.rank;
        (df != 0 || dr != 0) && (df == 0 || dr == 0 || df.abs() == dr.abs())
    };
    let knight_source = |to: Coord| {
        let dr = match view.your_color {
            Color::Sente => -2,
            Color::Gote => 2,
        };
        (to.file - king.file).abs() == 1 && to.rank - king.rank == dr
    };
    match *mv {
        ShogiMove::Board { from, to, .. } => {
            if from == king {
                return true;
            }
            on_ray(to) || knight_source(to)
        }
        ShogiMove::Drop { to, .. } => on_ray(to),
    }
}

/// 王手中の p(合法) 補正係数。玉移動が最も解消しやすい
fn in_check_prior(view: &PlayerView, mv: &ShogiMove) -> f64 {
    match *mv {
        ShogiMove::Board { from, .. } if Some(from) == king_square(view) => 0.5,
        _ => 0.25,
    }
}

fn prior_legal(view: &PlayerView, mv: &ShogiMove, opp_board_n: f64) -> f64 {
    let my_n = view.your_pieces.len() as f64;
    let q = (1.0 - opp_board_n / (81.0 - my_n)).clamp(0.05, 1.0);
    match *mv {
        ShogiMove::Board { from, to, .. } => {
            let df = to.file - from.file;
            let dr = to.rank - from.rank;
            let aligned = df == 0 || dr == 0 || df.abs() == dr.abs();
            let unknown = if aligned {
                (df.abs().max(dr.abs()) - 1).max(0)
            } else {
                0
            };
            q.powi(unknown as i32)
        }
        ShogiMove::Drop { .. } => q,
    }
}

fn evaluate(view: &PlayerView, mv: &ShogiMove, particles: &[&Position], prior: f64) -> f64 {
    let me = view.your_color;
    let opp = me.other();
    let mut legal = 0usize;
    let mut value_sum = 0.0;
    const PRESSURE_SAMPLES: usize = 16;
    let mut pressure_sum = 0.0;
    let mut pressure_n = 0usize;

    for pos in particles {
        if !pos.is_legal(mv) {
            continue;
        }
        legal += 1;
        let mut v = 0.0;

        if let ShogiMove::Board { to, .. } = *mv {
            if let Some(p) = pos.piece_at(to) {
                if p.color == opp {
                    v += piece_value(p.role);
                }
            }
        }

        let mut next = (*pos).clone();
        next.play_unchecked(mv);

        if next.in_check(opp) {
            v += 0.8;
            if next.legal_moves().is_empty() {
                v += 1000.0;
            }
        }

        v -= 0.6 * exposed_capture_risk(&next, me);

        if pressure_n < PRESSURE_SAMPLES {
            pressure_sum += king_zone_pressure(&next, me);
            pressure_n += 1;
        }

        value_sum += v;
    }

    let n = particles.len() as f64;
    let p_legal = (legal as f64 + prior * PRIOR_WEIGHT) / (n + PRIOR_WEIGHT);
    let expected = if legal > 0 {
        value_sum / legal as f64 - 0.2 * pressure_sum / pressure_n.max(1) as f64
    } else {
        0.0
    };

    let fouls_left = (10u32.saturating_sub(view.fouls.you)).max(1) as f64;
    let foul_cost = 1.5 * (10.0 / fouls_left).powf(1.5);

    let advance_bias = match *mv {
        ShogiMove::Board { from, to, promote } => {
            let adv = match me {
                Color::Sente => (from.rank - to.rank) as f64,
                Color::Gote => (to.rank - from.rank) as f64,
            };
            0.05 * adv + if promote { 0.1 } else { 0.0 }
        }
        ShogiMove::Drop { .. } => -0.05,
    };

    p_legal * (expected + advance_bias) - (1.0 - p_legal) * foul_cost
}

fn exposed_capture_risk(pos: &Position, me: Color) -> f64 {
    let opp = me.other();
    let mut worst = 0.0f64;
    for (sq, piece) in pos.pieces() {
        if piece.color != me || piece.role == Role::King {
            continue;
        }
        if !pos.is_attacked(sq, opp) {
            continue;
        }
        let defended = pos.is_attacked(sq, me);
        let loss = piece_value(piece.role) * if defended { 0.4 } else { 1.0 };
        worst = worst.max(loss);
    }
    worst
}

fn king_zone_pressure(pos: &Position, me: Color) -> f64 {
    let Some(king) = pos.king_square(me) else {
        return 0.0;
    };
    let opp = me.other();
    let mut pressure = 0;
    for df in -1..=1i8 {
        for dr in -1..=1i8 {
            let c = Coord {
                file: king.file + df,
                rank: king.rank + dr,
            };
            if (1..=9).contains(&c.file)
                && (1..=9).contains(&c.rank)
                && pos.is_attacked(c, opp)
            {
                pressure += 1;
            }
        }
    }
    pressure as f64
}
