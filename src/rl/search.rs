//! **実験**: 方策ネット＋粒子による手番局面の探索（Ataraxos の test-time search の簡約版）。
//!
//! Ataraxos（Stratego、arXiv 2511.07312）は信念ネットから隠れた局面をサンプルし、
//! 候補手×サンプルでロールアウトして、手番局面だけ磁気ミラー降下（MMD）を1ステップ回す。
//! ついたて将棋は駒の位置まで隠れるので、信念サンプルは既存の粒子フィルタ（`Estimator`）で代用し、
//! ロールアウトは2手（自分の手 → 相手の応手1手）で打ち切って方策ネットの価値ヘッドで評価する:
//!
//! 1. 方策ネットの分布 π から上位 `k` 手を候補にする
//! 2. 粒子を重みどおりに `n` 個引き、各候補 a × 粒子 s で次の末端を価値ヘッドで評価する。
//!    どちらの末端も自分の決定点なので、価値ヘッドの学習分布の中で評価できる:
//!    - a が s で非合法 → 反則として観測に積み、**同じ手番の決定点**の価値（反則上限なら −1）
//!    - 合法 → 適用し、相手の応手を `estimator::predict_opp_reply` で1手サンプルして、
//!      **次の自分の決定点**の価値（詰み・詰まされは ±1、200手の上限は 0）
//! 3. Q(a) = 粒子平均の価値。π'(a) ∝ π(a)·exp(η·Q(a))（磁石 = π の MMD 1ステップ）から引く
//!
//! 相手の観測履歴は粒子に無いので、相手の応手は方策ネットでなく推定器の相手モデル（opp_move NN）で
//! 代用する（露見マス・触ったマスは `EstimatorStrategy` の2手読みと同じ定義で渡す）。
//! 相手の反則は模擬しない。粒子は**物理的に整合するもの**を優先し、全滅していれば物理不整合の粒子で
//! 探索する（`taint=0` なら探索をやめて元の方策で指す。実測では落とすほうが大幅に強い）。
//!
//! 既知の近似: rl_v25 の価値ヘッドは R-NaD の正則化報酬（−η·log(π/π_reg)）込みの目標で学習している
//! ので、純粋な勝敗の期待値ではない（1〜2手先の候補間では差がほぼ打ち消し合う前提で使っている）。
//!
//! 戦略名: `rl_search:<重みのパス>[,k=8][,n=16][,eta=2][,greedy=1][,scale=2.2]`。
//! `eta=0` は「上位 k 手に切り詰めただけの方策」で、探索の効果を切り分ける対照に使う。

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use rand::{Rng, SeedableRng, rngs::StdRng};
use sha2::{Digest, Sha256};

use crate::board::Coord;
use crate::estimator::{Estimator, predict_opp_reply};
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{ClockState, Color, FoulCounts, GameStatus, PlayerView};
use crate::rl::action::{decode_action, decode_usi, legal_mask};
use crate::rl::encode::encode;
use crate::rl::policy_net::PolicyNet;
use crate::selfplay::{MAX_FOULS, MAX_PLIES};
use crate::shogi::{Outcome, Position, ShogiMove, parse_usi, unpromote_role};
use crate::strategy::Strategy;

pub const PREFIX: &str = "rl_search:";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchParams {
    pub k: usize,
    pub n: usize,
    pub eta: f64,
    pub greedy: bool,
    /// 推定器の思考予算スケール（estimator 戦略の既定 2000ms ÷ 900ms に合わせる）
    pub scale: f64,
    /// 物理的に整合する粒子が無い決定点で、物理不整合（phys_taint）の粒子で探索するか。
    /// false なら探索をやめて元の方策で指す
    pub taint: bool,
}

impl Default for SearchParams {
    fn default() -> Self {
        SearchParams {
            k: 8,
            n: 16,
            eta: 2.0,
            greedy: false,
            scale: 2000.0 / 900.0,
            taint: true,
        }
    }
}

/// `<パス>[,key=値...]` を分ける
fn parse_spec(spec: &str) -> Result<(String, SearchParams), String> {
    let mut parts = spec.split(',');
    let path = parts.next().unwrap_or("").to_string();
    let mut p = SearchParams::default();
    for kv in parts {
        let (k, v) = kv.split_once('=').ok_or_else(|| format!("{kv}: key=値 の形でない"))?;
        let bad = |e: String| format!("{kv}: {e}");
        match k {
            "k" => p.k = v.parse().map_err(|e: std::num::ParseIntError| bad(e.to_string()))?,
            "n" => p.n = v.parse().map_err(|e: std::num::ParseIntError| bad(e.to_string()))?,
            "eta" => p.eta = v.parse().map_err(|e: std::num::ParseFloatError| bad(e.to_string()))?,
            "scale" => p.scale = v.parse().map_err(|e: std::num::ParseFloatError| bad(e.to_string()))?,
            "greedy" => p.greedy = v == "1" || v == "true",
            "taint" => p.taint = v == "1" || v == "true",
            _ => return Err(format!("{k}: 未知のパラメータ")),
        }
    }
    if p.k == 0 || p.n == 0 {
        return Err("k と n は1以上".into());
    }
    Ok((path, p))
}

pub struct RlSearch {
    net: Arc<PolicyNet>,
    params: SearchParams,
    name: &'static str,
    est: Option<Estimator>,
    seed: u64,
    rng: StdRng,
    last: Option<serde_json::Value>,
}

impl RlSearch {
    /// 名前が違えば None、重みが読めない・パラメータが壊れていれば panic（arena の設定ミスを黙って
    /// 別の戦略にしないため。`RlPolicy::from_name` と同じ規約）
    pub fn from_name(name: &str, seed: Option<u64>) -> Option<Self> {
        let spec = name.strip_prefix(PREFIX)?;
        let (path, params) = parse_spec(spec).unwrap_or_else(|e| panic!("rl_search の指定が壊れている: {e}"));
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("方策ネットを読めない: {path}: {e}"));
        let sha: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
        let net = Arc::new(PolicyNet::from_bytes(&bytes).unwrap_or_else(|e| panic!("{path}: {e}")));
        let label = format!(
            "rl_search@{}[k{}n{}e{}{}]",
            &sha[..12],
            params.k,
            params.n,
            params.eta,
            format!("{}{}", if params.greedy { "g" } else { "" }, if params.taint { "" } else { "t0" })
        );
        let seed = seed.unwrap_or_else(|| rand::rng().random());
        Some(RlSearch {
            net,
            params,
            name: Box::leak(label.into_boxed_str()),
            est: None,
            seed,
            rng: StdRng::seed_from_u64(seed ^ 0x5eed_5eed),
            last: None,
        })
    }

    fn estimator(&mut self, color: Color) -> &mut Estimator {
        let (seed, scale) = (self.seed, self.params.scale);
        self.est
            .get_or_insert_with(|| Estimator::with_seed_and_scale(color, seed, scale))
    }
}

/// 局面と自分側の帳簿から、自分の決定点の視界を作る（`Referee::view` と同じ中身）
fn view_of(pos: &Position, me: Color, fouls: FoulCounts, game_id: &str) -> PlayerView {
    PlayerView {
        game_id: game_id.to_string(),
        your_color: me,
        your_pieces: pos.pieces_of(me),
        your_hand: pos.hand_map(me),
        turn: pos.turn(),
        move_number: pos.move_number(),
        clocks: ClockState {
            sente_ms: 0,
            gote_ms: 0,
            running: Some(pos.turn()),
            server_time: 0,
        },
        fouls,
        you_in_check: pos.in_check(me),
        opponent_in_check: pos.in_check(me.other()),
        status: GameStatus::Playing,
    }
}

fn copy_log(log: &ObservationLog) -> ObservationLog {
    let mut out = ObservationLog::default();
    for e in log.events() {
        out.record(e.clone());
    }
    out
}

/// 受理手の直後の終局判定（`Referee::accept` と同じ順: 詰み・ステイルメイト > 手数上限）。
/// 自分から見た価値を返す（勝ち 1・負け −1・引き分け 0）
fn terminal_value(pos: &Position, me: Color) -> Option<f32> {
    match pos.outcome() {
        Some(Outcome::Checkmate { winner }) | Some(Outcome::Stalemate { winner }) => {
            Some(if winner == me { 1.0 } else { -1.0 })
        }
        // 初期局面からの対局では 受理手数 = move_number − 1
        None if pos.move_number().saturating_sub(1) >= MAX_PLIES => Some(0.0),
        None => None,
    }
}

/// 1候補 × 1粒子の遷移の行き先
#[derive(Debug)]
pub(crate) enum Leaf {
    /// 終局（自分から見た価値）
    Terminal(f32),
    /// 次の自分の決定点（反則なら同じ手番、受理なら相手の応手の後）
    Decision {
        view: PlayerView,
        log: ObservationLog,
        foul_tried: HashSet<String>,
    },
}

/// 粒子 `pos`（自分の手番）で候補 `mv` を指し、受理なら `reply` が返す相手の応手を1手進めて、
/// 次の自分の決定点（または終局）まで観測を積む。観測の作り方は `Referee` と同じ
/// （反則は手番維持、王手宣言は受理手の後、終局判定は詰み > 手数上限）
pub(crate) fn step_leaf(
    view: &PlayerView,
    log: &ObservationLog,
    foul_tried: &HashSet<String>,
    pos: &Position,
    mv: &ShogiMove,
    reply: impl FnOnce(&Position) -> Option<ShogiMove>,
) -> Leaf {
    let me = view.your_color;
    let usi = mv.to_usi();
    if !pos.is_legal(mv) {
        // 反則: 手番はそのまま。反則上限なら負け
        let fouls = view.fouls.you + 1;
        if fouls >= MAX_FOULS {
            return Leaf::Terminal(-1.0);
        }
        let mut log2 = copy_log(log);
        log2.record(Observation::MyFoul {
            move_number: pos.move_number(),
            usi: usi.clone(),
        });
        let mut tried = foul_tried.clone();
        tried.insert(usi);
        let mut v = view.clone();
        v.fouls.you = fouls;
        return Leaf::Decision {
            view: v,
            log: log2,
            foul_tried: tried,
        };
    }
    let mut p = pos.clone();
    let captured = p.play_unchecked(mv);
    let mut log2 = copy_log(log);
    log2.record(Observation::MyMove {
        move_number: p.move_number(),
        usi,
        captured: captured.map(unpromote_role),
    });
    if p.in_check(me.other()) {
        log2.record(Observation::Check { in_check: me.other() });
    }
    if let Some(v) = terminal_value(&p, me) {
        return Leaf::Terminal(v);
    }
    // 相手の応手（相手の反則は模擬しない）
    let Some(r) = reply(&p) else {
        return Leaf::Terminal(0.0);
    };
    let lost_at = match r {
        ShogiMove::Board { to, .. } if p.piece_at(to).is_some_and(|q| q.color == me) => {
            Some(crate::board::make_usi_square(to))
        }
        _ => None,
    };
    p.play_unchecked(&r);
    log2.record(Observation::OpponentMoved {
        move_number: p.move_number(),
        captured_my_piece_at: lost_at,
    });
    if p.in_check(me) {
        log2.record(Observation::Check { in_check: me });
    }
    if let Some(v) = terminal_value(&p, me) {
        return Leaf::Terminal(v);
    }
    Leaf::Decision {
        view: view_of(&p, me, view.fouls.clone(), &view.game_id),
        log: log2,
        foul_tried: HashSet::new(),
    }
}

/// 観測ログから「自分が駒を取ったマス（相手に露見）」と「自分の手が触れたマス」を作る。
/// `EstimatorStrategy::choose` の2手読みと同じ定義（estimator の my_capture_sq / my_touched_sq）
fn my_squares(log: &ObservationLog) -> (Vec<Coord>, Vec<Coord>) {
    let mut captures = vec![];
    let mut touched = vec![];
    for e in log.events() {
        if let Observation::MyMove { usi, captured, .. } = e {
            if let Some(mv) = parse_usi(usi) {
                push_move_squares(&mv, captured.is_some(), &mut captures, &mut touched);
            }
        }
    }
    (captures, touched)
}

fn push_move_squares(mv: &ShogiMove, captured: bool, captures: &mut Vec<Coord>, touched: &mut Vec<Coord>) {
    let to = match *mv {
        ShogiMove::Board { to, .. } | ShogiMove::Drop { to, .. } => to,
    };
    if captured {
        captures.push(to);
    }
    if let ShogiMove::Board { from, .. } = *mv {
        touched.push(from);
    }
    touched.push(to);
}

/// 探索に使う粒子と重み。`allow_taint` が false なら**物理不整合（phys_taint）の粒子は使わない**
/// （推定器の約束どおり、幽霊取りなどで救済した盤面は合法性・詰みの根拠にならない）。
/// 情報制約だけを緩めたソフト救済の粒子は、推定器が logw へ課金済みなのでそのまま使う。
/// 重みは `weighted_unique_particles` と同じ規約（logw を max で正規化し、同一指紋を畳み込む）
fn search_pool(est: &Estimator, me: Color, allow_taint: bool) -> Vec<(Position, f64)> {
    let usable = |t: u8| allow_taint || t == 0;
    let max_logw = est
        .log_weights()
        .iter()
        .zip(est.phys_taint())
        .filter(|(_, t)| usable(**t))
        .map(|(w, _)| *w)
        .fold(f64::MIN, f64::max);
    let mut idx: HashMap<u64, usize> = HashMap::new();
    let mut out: Vec<(Position, f64)> = vec![];
    for ((p, &taint), &w) in est.particles().iter().zip(est.phys_taint()).zip(est.log_weights()) {
        if !usable(taint) || p.turn() != me {
            continue;
        }
        let m = (w - max_logw).exp();
        let fp = p.fingerprint();
        match idx.get(&fp) {
            Some(&i) => out[i].1 += m,
            None => {
                idx.insert(fp, out.len());
                out.push((p.clone(), m));
            }
        }
    }
    out
}

impl Strategy for RlSearch {
    fn prewarm(&mut self, view: &PlayerView, log: &ObservationLog) {
        self.estimator(view.your_color).update(log);
    }

    fn choose(
        &mut self,
        view: &PlayerView,
        log: &ObservationLog,
        foul_tried: &HashSet<String>,
    ) -> Option<String> {
        let t0 = std::time::Instant::now();
        let me = view.your_color;
        self.estimator(me).update(log);

        let mask = legal_mask(view, log, foul_tried);
        let legal: Vec<usize> = (0..mask.len()).filter(|&a| mask[a]).collect();
        if legal.is_empty() {
            return None;
        }
        let (logits, value) = self.net.forward(&encode(view, log, foul_tried));
        let max = legal.iter().map(|&a| logits[a]).fold(f32::MIN, f32::max);
        let total: f64 = legal.iter().map(|&a| f64::from(logits[a] - max).exp()).sum();
        let mut cands: Vec<(usize, f64)> = legal
            .iter()
            .map(|&a| (a, f64::from(logits[a] - max).exp() / total))
            .collect();
        cands.sort_by(|a, b| b.1.total_cmp(&a.1));
        // 上位 k 手に残った π の質量（切り詰めが η と無関係に分布を変える量の記録）
        let kept_mass: f64 = cands.iter().take(self.params.k).map(|c| c.1).sum();

        // 物理的に整合する粒子を重みどおりに引く（search_pool）。全滅していて `taint=1` なら
        // 物理不整合の粒子へ落とす（実測: 落とさずに元の方策へ戻すと η=30 で 80.5% → 66.5%。
        // 未探索の決定点を含む39局の勝率が 90% → 36% と、差のほぼ全部がそこに集中した）
        let est = self.est.as_ref().unwrap();
        let mut pool = search_pool(est, me, false);
        let mut tainted = false;
        if pool.is_empty() && self.params.taint {
            pool = search_pool(est, me, true);
            tainted = !pool.is_empty();
        }
        let pool_total: f64 = pool.iter().map(|x| x.1).sum();
        let can_search = !pool.is_empty() && pool_total > 0.0;
        let searched = self.params.eta != 0.0 && can_search && cands.len().min(self.params.k) > 1;
        // 探索できない決定点は元の方策（切り詰めなし）で指す。eta=0 の対照は常に上位 k 手
        if searched || self.params.eta == 0.0 {
            cands.truncate(self.params.k);
        }

        let mut q = vec![0.0f64; cands.len()];
        // Q の標準誤差（診断用。粒子と応手のサンプル数が足りているかを見る）
        let mut se = vec![0.0f64; cands.len()];
        if searched {
            // 相手の応手モデルの入力（露見マス・触ったマス）。候補自身の分は候補ごとに足す
            let (captures0, touched0) = my_squares(log);
            let my_fouls = foul_tried.len() as u32;
            let samples: Vec<&Position> = (0..self.params.n)
                .map(|_| {
                    let mut r = self.rng.random::<f64>() * pool_total;
                    for (p, w) in &pool {
                        if r < *w {
                            return p;
                        }
                        r -= w;
                    }
                    &pool.last().unwrap().0
                })
                .collect();
            for (ci, &(a, _)) in cands.iter().enumerate() {
                let mv = decode_action(a, me).expect("マスク内の行動は復号できる");
                let mut vals: Vec<f64> = Vec::with_capacity(samples.len());
                for s in &samples {
                    let rng = &mut self.rng;
                    let leaf = step_leaf(view, log, foul_tried, s, &mv, |next| {
                        // この候補で駒を取れば、捕獲通知でそのマスは相手に露見する
                        // （既知マスに入れないと即時の取り返しのブーストが掛からない）
                        let (mut captures, mut touched) = (captures0.clone(), touched0.clone());
                        let captured = match mv {
                            ShogiMove::Board { to, .. } => s.piece_at(to).is_some(),
                            ShogiMove::Drop { .. } => false,
                        };
                        push_move_squares(&mv, captured, &mut captures, &mut touched);
                        predict_opp_reply(next, me, &captures, &touched, my_fouls, rng)
                    });
                    vals.push(match leaf {
                        Leaf::Terminal(v) => f64::from(v),
                        Leaf::Decision { view, log, foul_tried } => {
                            f64::from(self.net.forward(&encode(&view, &log, &foul_tried)).1)
                        }
                    });
                }
                let m = vals.iter().sum::<f64>() / vals.len() as f64;
                q[ci] = m;
                let var = vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / vals.len() as f64;
                se[ci] = (var / vals.len() as f64).sqrt();
            }
        }

        // MMD 1ステップ（磁石 = π）: π'(a) ∝ π(a)·exp(η·Q(a))
        let qmax = q.iter().copied().fold(f64::MIN, f64::max);
        let w: Vec<f64> = cands
            .iter()
            .zip(&q)
            .map(|(&(_, p), &qa)| p * (self.params.eta * (qa - qmax)).exp())
            .collect();
        let wsum: f64 = w.iter().sum();
        let pick = if self.params.greedy {
            (0..cands.len()).max_by(|&i, &j| w[i].total_cmp(&w[j])).unwrap()
        } else {
            let mut r = self.rng.random::<f64>() * wsum;
            let mut chosen = cands.len() - 1;
            for (i, wi) in w.iter().enumerate() {
                if r < *wi {
                    chosen = i;
                    break;
                }
                r -= wi;
            }
            chosen
        };
        self.last = Some(serde_json::json!({
            "value": value,
            "searched": searched,
            "pool": pool.len(),
            "tainted": tainted,
            "kept_mass": kept_mass,
            "cands": cands.iter().zip(&q).zip(&w).zip(&se).map(|(((&(a, p), &qa), &wa), &sa)| serde_json::json!({
                "usi": decode_usi(a, me), "p": p, "q": qa, "se": sa, "p_new": wa / wsum,
            })).collect::<Vec<_>>(),
            "ms": t0.elapsed().as_millis() as u64,
        }));
        decode_usi(cands[pick].0, me)
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn debug_state(&self) -> Option<serde_json::Value> {
        self.last.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::referee::{Referee, StepResult};

    const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/rl/tiny_policy.bin");

    #[test]
    fn 指定を読める() {
        let (path, p) = parse_spec("a/b.bin,k=4,n=3,eta=0.5,greedy=1").unwrap();
        assert_eq!(path, "a/b.bin");
        assert_eq!((p.k, p.n, p.eta, p.greedy), (4, 3, 0.5, true));
        assert_eq!(parse_spec("x.bin").unwrap().1, SearchParams::default());
        assert!(parse_spec("x.bin,zz=1").is_err());
        assert!(parse_spec("x.bin,k=0").is_err());
    }

    /// 小さいネットでも、マスク内の手だけを指して1局を終えられる
    #[test]
    fn 探索つきで1局指せる() {
        let name = format!("{PREFIX}{FIXTURE},k=3,n=2,scale=0.25");
        let mut players = [
            RlSearch::from_name(&name, Some(1)).unwrap(),
            RlSearch::from_name(&name, Some(2)).unwrap(),
        ];
        let mut referee = Referee::new();
        for _ in 0..60 {
            let side = referee.to_move();
            let view = referee.view(side, [0, 0], 0);
            let p = &mut players[side as usize];
            let Some(usi) = p.choose(&view, referee.log(side), referee.foul_tried(side)) else {
                break;
            };
            let mask = legal_mask(&view, referee.log(side), referee.foul_tried(side));
            assert!(mask[crate::rl::action::encode_usi(&usi, side).unwrap()]);
            if let StepResult::Ended { .. } = referee.step(&usi, 0) {
                break;
            }
        }
        assert!(players[0].debug_state().unwrap()["searched"].as_bool().unwrap());
    }

    fn replay(hist: &[String]) -> Referee {
        let mut r = Referee::new();
        for usi in hist {
            r.step(usi, 0);
        }
        r
    }

    fn events_json(log: &ObservationLog) -> String {
        serde_json::to_string(log.events()).unwrap()
    }

    /// `step_leaf` の遷移（観測・視界・反則済みの手・終局）が審判と一致する。
    /// 真の局面を粒子とみなし、候補手（反則もありうる）と相手の応手を同じだけ審判に進めて突き合わせる。
    /// 200手の上限（自分の手で到達・相手の応手で到達）と反則負けの境界も通す
    #[test]
    fn 探索の遷移は審判と一致する() {
        use crate::selfplay::GameResult;
        let (mut hit_draw_mine, mut hit_draw_reply, mut hit_foul_limit, mut hit_foul) = (false, false, false, false);
        for seed in 0..8u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            // 実際の対局の反則率（高いと反則負けの境界、低いと200手の境界を通る）
            let foul_rate = if seed < 5 { 0.01 } else { 0.4 };
            let mut referee = Referee::new();
            let mut hist: Vec<String> = vec![];
            loop {
                let me = referee.to_move();
                let view = referee.view(me, [0, 0], 0);
                let (log, tried) = (referee.log(me), referee.foul_tried(me));
                let mask = legal_mask(&view, log, tried);
                let masked: Vec<usize> = (0..mask.len()).filter(|&a| mask[a]).collect();
                if masked.is_empty() {
                    break;
                }
                let pos = referee.position().clone();

                // 候補（マスク内から一様 = 反則もありうる）と相手の応手を突き合わせる
                let cand = decode_action(masked[rng.random_range(0..masked.len())], me).unwrap();
                let pick: u64 = rng.random();
                let mut used_reply: Option<String> = None;
                let leaf = step_leaf(&view, log, tried, &pos, &cand, |next| {
                    let moves = next.legal_moves();
                    let r = moves.get(pick as usize % moves.len().max(1)).copied();
                    used_reply = r.map(|m| m.to_usi());
                    r
                });
                let mut b = replay(&hist);
                let first = b.step(&cand.to_usi(), 0);
                if first == StepResult::Foul {
                    hit_foul = true;
                }
                if first == StepResult::Accepted {
                    b.step(used_reply.as_deref().expect("受理なら応手を引いている"), 0);
                }
                match (leaf, b.ended()) {
                    (Leaf::Terminal(v), Some((result, reason))) => {
                        let want = match result {
                            GameResult::Win(c) if c == me => 1.0,
                            GameResult::Win(_) => -1.0,
                            GameResult::Draw => 0.0,
                        };
                        assert_eq!(v, want, "seed {seed}: 終局 {reason} の価値");
                        match reason {
                            "max_plies" if used_reply.is_none() => hit_draw_mine = true,
                            "max_plies" => hit_draw_reply = true,
                            "foul_limit" => hit_foul_limit = true,
                            _ => {}
                        }
                    }
                    (Leaf::Decision { view: v2, log: l2, foul_tried: t2 }, None) => {
                        assert_eq!(b.to_move(), me, "seed {seed}: 次は自分の決定点");
                        let bv = b.view(me, [0, 0], 0);
                        assert_eq!(events_json(&l2), events_json(b.log(me)), "seed {seed}: 観測");
                        assert_eq!(&t2, b.foul_tried(me), "seed {seed}: 反則済みの手");
                        assert_eq!((v2.fouls.you, v2.fouls.opponent), (bv.fouls.you, bv.fouls.opponent));
                        assert_eq!(
                            encode(&v2, &l2, &t2),
                            encode(&bv, b.log(me), b.foul_tried(me)),
                            "seed {seed}: ネットの入力"
                        );
                    }
                    (leaf, ended) => panic!("seed {seed}: 食い違い {leaf:?} vs {ended:?}"),
                }

                // 実際の対局を1手進める（大半は合法手）
                let usi = if rng.random::<f64>() < foul_rate {
                    decode_usi(masked[rng.random_range(0..masked.len())], me).unwrap()
                } else {
                    let moves = pos.legal_moves();
                    moves[rng.random_range(0..moves.len())].to_usi()
                };
                hist.push(usi.clone());
                if let StepResult::Ended { .. } = referee.step(&usi, 0) {
                    break;
                }
            }
        }
        assert!(hit_foul, "反則の遷移を通っていない");
        assert!(hit_foul_limit, "反則負けの境界を通っていない");
        assert!(hit_draw_mine && hit_draw_reply, "200手の境界（自分の手 {hit_draw_mine} / 応手 {hit_draw_reply}）を通っていない");
    }
}
