//! 指し手の選択。
//!
//! `Strategy` trait の実装を差し替えて強さを比較する（bin/arena.rs で対戦できる）。
//! - `Heuristic`: サイト内蔵の簡易botと同じ「前進を好むヒューリスティック＋乱数」
//! - `EstimatorStrategy`: 観測履歴から相手局面の粒子集合を維持し（estimator.rs）、
//!   候補手を粒子平均で評価する

use std::collections::{HashMap, HashSet};

use rand::Rng;

use crate::board::{
    Coord, Promotion, drop_targets, make_usi_drop, make_usi_move, make_usi_square, move_targets,
    parse_usi_square, promotion_choice,
};
use crate::check::CheckSolver;
use crate::estimator::Estimator;
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{Color, PlayerView, Role};
use crate::shogi::{Position, ShogiMove, parse_usi, piece_value};

/// 1インスタンス = 1対局。対局開始ごとに `make` で作り直す。
pub trait Strategy {
    /// 自分の手番で呼ばれる。foul_tried の手は除外すること。
    /// None を返したら投了（指せる手がない）。
    fn choose(
        &mut self,
        view: &PlayerView,
        log: &ObservationLog,
        foul_tried: &HashSet<String>,
    ) -> Option<String>;

    fn name(&self) -> &'static str;

    /// 直近の choose 時点の内部状態（対局記録のデバッグ用）。推定系のみ実装する
    fn debug_state(&self) -> Option<serde_json::Value> {
        None
    }
}

pub const DEFAULT_STRATEGY: &str = "estimator";

/// 戦略名からインスタンスを作る。未知の名前は None。
/// `estimator_vN` はアリーナ比較用の凍結版（src/frozen/）
pub fn make(name: &str) -> Option<Box<dyn Strategy>> {
    match name {
        "heuristic" => Some(Box::new(Heuristic)),
        "estimator" => Some(Box::new(EstimatorStrategy::new())),
        "estimator_v2" => Some(Box::new(crate::frozen::estimator_v2::EstimatorV2::new())),
        "estimator_v3" => Some(Box::new(crate::frozen::estimator_v3::EstimatorV3::new())),
        "estimator_v4" => Some(Box::new(crate::frozen::estimator_v4::EstimatorV4::new())),
        _ => None,
    }
}

/// 前進を好むヒューリスティック＋乱数（従来実装）
pub struct Heuristic;

impl Strategy for Heuristic {
    fn choose(
        &mut self,
        view: &PlayerView,
        _log: &ObservationLog,
        foul_tried: &HashSet<String>,
    ) -> Option<String> {
        choose_move(view, foul_tried)
    }

    fn name(&self) -> &'static str {
        "heuristic"
    }
}

/// 候補手を生成してスコア最大の手を返す。foul_tried の手は除外。
/// 候補が尽きたら None（呼び出し側で投了する）。
pub fn choose_move(view: &PlayerView, foul_tried: &HashSet<String>) -> Option<String> {
    let mut rng = rand::rng();
    let mut best: Option<(String, f64)> = None;
    let consider = |usi: String, score: f64, best: &mut Option<(String, f64)>| {
        if foul_tried.contains(&usi) {
            return;
        }
        if best.as_ref().is_none_or(|(_, s)| score > *s) {
            *best = Some((usi, score));
        }
    };

    let color = view.your_color;
    for piece in &view.your_pieces {
        let Some(from) = parse_usi_square(&piece.square) else {
            continue;
        };
        for to in move_targets(&view.your_pieces, piece, color) {
            let promote = promotion_choice(piece.role, from, to, color) != Promotion::None;
            // 前進を好む（先手は rank 減少が前進）
            let advance = match color {
                Color::Sente => (from.rank - to.rank) as f64,
                Color::Gote => (to.rank - from.rank) as f64,
            };
            let mut score = advance + rng.random_range(0.0..4.0);
            if promote {
                score += 3.0;
            }
            if piece.role == Role::King {
                score -= 2.0; // 玉は無闇に動かさない
            }
            consider(make_usi_move(from, to, promote), score, &mut best);
        }
    }

    for (&role, &count) in &view.your_hand {
        if count == 0 {
            continue;
        }
        for to in drop_targets(&view.your_pieces, role, color) {
            if let Some(usi) = make_usi_drop(role, to) {
                // 打ちは控えめに（乱数のみ）
                consider(usi, rng.random_range(0.0..3.0), &mut best);
            }
        }
    }

    best.map(|(usi, _)| usi)
}

/// 評価に使う粒子数の上限（思考時間の予算。粒子は estimator 側で最大400）。
/// フィッシャー300秒+3秒に対し1手1〜2秒が目安。96粒子で平均370ms程度だったので
/// 精度側（反則率の低下）に予算を振る
const EVAL_PARTICLES: usize = 192;

/// 観測履歴から相手局面を推定して指す戦略。
///
/// 候補手（自分に見える範囲の疑似合法手）を、推定粒子の平均で評価する:
/// - 駒得の期待値（その粒子でそのマスに相手駒がいるか）
/// - 反則確率（粒子上で非合法な割合）× 反則コスト（残り反則数が減るほど高い）
/// - 指した直後に取り返されるリスク（粒子上での相手の即時駒取り）
/// - 王手・詰みボーナス
pub struct EstimatorStrategy {
    est: Option<Estimator>,
    /// 直近の choose 時点の内部状態（記録用）
    last_debug: Option<serde_json::Value>,
}

impl EstimatorStrategy {
    pub fn new() -> Self {
        EstimatorStrategy {
            est: None,
            last_debug: None,
        }
    }
}

impl Default for EstimatorStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl Strategy for EstimatorStrategy {
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
        let known = knownness_map(view, log);

        let mut rng = rand::rng();
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
            let mut score =
                evaluate(view, &mv, &sample, prior, &known) + rng.random_range(0.0..0.01);
            // 手戻り（直前の手をそのまま逆に戻す）は膠着の典型なので減点。
            // 手数上限の引き分けを崩す側に倒す
            if let (
                Some(ShogiMove::Board { from: pf, to: pt, .. }),
                ShogiMove::Board { from, to, .. },
            ) = (last_my_move, mv)
            {
                if from == pt && to == pf {
                    score -= 0.35;
                }
            }
            if best.as_ref().is_none_or(|(_, s)| score > *s) {
                best = Some((usi, score));
            }
        }

        self.last_debug = Some(debug_summary(est, &sample));
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
fn debug_summary(est: &Estimator, sample: &[&Position]) -> serde_json::Value {
    let opp = est.my_color().other();
    let mut king_votes: HashMap<Coord, u32> = HashMap::new();
    for pos in sample {
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
/// - 0.55: 初期配置から一度も動いていない駒（相手は初期配置を知っている）
fn knownness_map(view: &PlayerView, log: &ObservationLog) -> HashMap<Coord, f64> {
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
            0.4
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
fn camp_defended_prior(to: Coord, me: Color) -> f64 {
    let depth_from_back = match me {
        Color::Sente => to.rank,     // 相手（後手）の陣は rank 1..=3
        Color::Gote => 10 - to.rank, // 相手（先手）の陣は rank 7..=9
    };
    match depth_from_back {
        1 => 0.25,
        2 => 0.2,
        3 => 0.15,
        _ => 0.0,
    }
}

/// 事前確率の重み（擬似観測数）。粒子が少ない・偏っているときほど事前が効く
const PRIOR_WEIGHT: f64 = 4.0;

/// 候補手をユニーク粒子の平均で評価する
fn evaluate(
    view: &PlayerView,
    mv: &ShogiMove,
    particles: &[&Position],
    prior: f64,
    known: &HashMap<Coord, f64>,
) -> f64 {
    let me = view.your_color;
    let opp = me.other();
    let mut legal = 0usize;
    let mut value_sum = 0.0;
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

        let mut next = (*pos).clone();
        next.play_unchecked(mv);

        // 王手・詰み。ついたて将棋では王手された側は王手駒の位置が見えず
        // 手探りの反則をしやすい（反則10回で負け）ので、王手自体が得点源。
        // 相手の反則が溜まっているほど価値が上がる
        if next.in_check(opp) {
            v += 0.9 + 0.12 * f64::from(view.fouls.opponent);
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
        let mover_w = if captured_value > 0.0 { 0.9 } else { 0.45 };
        let own_after = next
            .piece_at(to)
            .map(|p| piece_value(p.role))
            .unwrap_or(0.0);
        let known_factor = if captured_value > 0.0 { 1.0 } else { 0.35 };
        let floor = own_after * camp_defended_prior(to, me) * known_factor;
        let mover_risk = mover_w * recapture_risk(&next, me, to).max(floor);
        let hidden_risk = exposed_capture_risk(&next, me, Some(to), known);
        v -= mover_risk.max(hidden_risk);

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

    // 粒子の証拠と事前確率のブレンド（粒子ゼロなら事前そのもの）
    let n = particles.len() as f64;
    let p_legal = (legal as f64 + prior * PRIOR_WEIGHT) / (n + PRIOR_WEIGHT);
    let expected = if legal > 0 {
        value_sum / legal as f64
            + (0.12 * attack_sum - 0.2 * pressure_sum) / pressure_n.max(1) as f64
    } else {
        0.0
    };

    // 反則コスト: 手番は失わないが反則数を消費する。残りが少ないほど急激に高価。
    // 序盤の「安い反則で情報を得る」は低コスト側で自然に許容される
    let fouls_left = (10u32.saturating_sub(view.fouls.you)).max(1) as f64;
    let foul_cost = 1.5 * (10.0 / fouls_left).powf(1.5);

    // 前進の弱い事前バイアス（推定が薄い序盤に駒をぶつけに行くため）
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

    // 期待値が負の手を p_legal で割り引かない（min の形）。
    // 割り引くと「合法確率が低いほどスコアが高い」= わざと反則に寄る手が
    // 選ばれてしまう。反則しても手番は残るので悪い局面からは逃げられず、
    // 反則の価値は「次善手の価値 − 反則コスト」でしかない
    let gain = expected + advance_bias;
    (p_legal * gain).min(gain) - (1.0 - p_legal) * foul_cost
}

/// 着手駒（マス to にいる自駒）が次の相手番で取られるリスク。
/// 紐つきなら取り返せるぶん割り引く（相手のどの駒で取るかは不明なので近似）
fn recapture_risk(pos: &Position, me: Color, to: Coord) -> f64 {
    let opp = me.other();
    let Some(piece) = pos.piece_at(to).filter(|p| p.color == me) else {
        return 0.0;
    };
    if piece.role == Role::King || !pos.is_attacked(to, opp) {
        return 0.0;
    }
    let defended = pos.is_attacked(to, me);
    piece_value(piece.role) * if defended { 0.45 } else { 1.0 }
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
        let weight = 0.35 + 0.3 * knownness;
        let loss = piece_value(piece.role) * if defended { 0.4 } else { 1.0 } * weight;
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

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::protocol::{ClockState, FoulCounts, GameStatus, VisiblePiece};

    pub(crate) fn minimal_view(pieces: Vec<VisiblePiece>, hand: HashMap<Role, u32>) -> PlayerView {
        PlayerView {
            game_id: "g".into(),
            your_color: Color::Sente,
            your_pieces: pieces,
            your_hand: hand,
            turn: Color::Sente,
            move_number: 1,
            clocks: ClockState {
                sente_ms: 300_000,
                gote_ms: 300_000,
                running: Some(Color::Sente),
                server_time: 0,
            },
            fouls: FoulCounts { you: 0, opponent: 0 },
            you_in_check: false,
            opponent_in_check: false,
            status: GameStatus::Playing,
        }
    }

    #[test]
    fn chooses_some_move() {
        let view = minimal_view(
            vec![VisiblePiece {
                square: "7g".into(),
                role: Role::Pawn,
            }],
            HashMap::new(),
        );
        assert_eq!(choose_move(&view, &HashSet::new()), Some("7g7f".to_string()));
    }

    #[test]
    fn skips_fouled_moves_and_resigns_when_exhausted() {
        let view = minimal_view(
            vec![VisiblePiece {
                square: "7g".into(),
                role: Role::Pawn,
            }],
            HashMap::new(),
        );
        let mut tried = HashSet::new();
        tried.insert("7g7f".to_string());
        assert_eq!(choose_move(&view, &tried), None);
    }

    #[test]
    fn may_resolve_check_filters_hopeless_moves() {
        // 先手玉 5i。ライン外への手・桂の利き元以外は王手を解消しえない
        let view = minimal_view(
            vec![
                VisiblePiece {
                    square: "5i".into(),
                    role: Role::King,
                },
                VisiblePiece {
                    square: "7g".into(),
                    role: Role::Pawn,
                },
            ],
            HashMap::new(),
        );
        let ok = |usi: &str| may_resolve_check(&view, &parse_usi(usi).unwrap());
        assert!(ok("5i5h"), "玉移動は常に候補");
        assert!(ok("7g5g"), "自玉と同段（ライン上）への移動は合駒/取りになりうる");
        assert!(ok("7g5e"), "架空の手でも判定対象はライン（5筋）上の着地点");
        assert!(!ok("7g7f"), "ライン外への移動は王手放置が確定");
    }

    #[test]
    fn may_resolve_check_knight_source_and_drops() {
        let view = minimal_view(
            vec![VisiblePiece {
                square: "5i".into(),
                role: Role::King,
            }],
            HashMap::new(),
        );
        let mv = |usi: &str| parse_usi(usi).unwrap();
        // 4g/6g は相手桂の利き元 → 盤上の駒での取りは候補
        assert!(may_resolve_check(&view, &mv("4f4g")));
        // 打ちは駒を取れないので桂の利き元でも解消しえない
        assert!(!may_resolve_check(&view, &mv("P*4g")));
        // ライン上への打ちは合駒
        assert!(may_resolve_check(&view, &mv("P*5e")));
        assert!(!may_resolve_check(&view, &mv("P*4e")));
    }

    #[test]
    fn estimator_in_check_prefers_resolving_moves() {
        // 粒子が王手を反映していなくても（空ログ = 初期局面粒子）、
        // you_in_check なら解消しうる手（ここでは玉移動のみ）しか指さない
        let mut view = minimal_view(
            vec![
                VisiblePiece {
                    square: "5i".into(),
                    role: Role::King,
                },
                VisiblePiece {
                    square: "7g".into(),
                    role: Role::Pawn,
                },
            ],
            HashMap::new(),
        );
        view.you_in_check = true;
        let mut strat = EstimatorStrategy::new();
        let log = ObservationLog::default();
        let usi = strat.choose(&view, &log, &HashSet::new()).unwrap();
        assert!(
            usi.starts_with("5i"),
            "王手中は玉移動を選ぶはず（選ばれた手: {usi}）"
        );
    }

    #[test]
    fn make_knows_heuristic() {
        assert!(make("heuristic").is_some());
        assert!(make("nonsense").is_none());
    }

    #[test]
    fn make_knows_frozen_versions() {
        assert!(make("estimator").is_some());
        assert!(make("estimator_v2").is_some());
        assert!(make("estimator_v3").is_some());
        assert!(make("estimator_v4").is_some());
    }
}
