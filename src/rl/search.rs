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
//!      **次の自分の決定点**の価値（詰み・詰まされは ±1）
//! 3. Q(a) = 粒子平均の価値。π'(a) ∝ π(a)·exp(η·Q(a))（磁石 = π の MMD 1ステップ）から引く
//!
//! 相手の観測履歴は粒子に無いので、相手の応手は方策ネットでなく推定器の相手モデル（opp_move NN）で
//! 代用する。相手の反則は模擬しない。
//!
//! 戦略名: `rl_search:<重みのパス>[,k=8][,n=16][,eta=2][,greedy=1][,scale=2.2]`。
//! `eta=0` は「上位 k 手に切り詰めただけの方策」で、探索の効果を切り分ける対照に使う。

use std::collections::HashSet;
use std::sync::Arc;

use rand::{Rng, SeedableRng, rngs::StdRng};
use sha2::{Digest, Sha256};

use crate::estimator::{Estimator, predict_opp_reply};
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{ClockState, Color, FoulCounts, GameStatus, PlayerView};
use crate::rl::action::{decode_action, decode_usi, legal_mask};
use crate::rl::encode::encode;
use crate::rl::policy_net::PolicyNet;
use crate::selfplay::MAX_FOULS;
use crate::shogi::{Outcome, Position, ShogiMove, unpromote_role};
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
}

impl Default for SearchParams {
    fn default() -> Self {
        SearchParams {
            k: 8,
            n: 16,
            eta: 2.0,
            greedy: false,
            scale: 2000.0 / 900.0,
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
            if params.greedy { "g" } else { "" }
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

fn mated_winner(pos: &Position) -> Option<Color> {
    match pos.outcome() {
        Some(Outcome::Checkmate { winner }) | Some(Outcome::Stalemate { winner }) => Some(winner),
        None => None,
    }
}

/// 粒子 `pos`（自分の手番）で候補 `mv` を指した結果を、次の自分の決定点の価値で評価する
#[allow(clippy::too_many_arguments)]
fn leaf_value(
    net: &PolicyNet,
    view: &PlayerView,
    log: &ObservationLog,
    foul_tried: &HashSet<String>,
    pos: &Position,
    mv: &ShogiMove,
    usi: &str,
    rng: &mut StdRng,
) -> f32 {
    let me = view.your_color;
    if !pos.is_legal(mv) {
        // 反則: 手番はそのまま。反則上限なら負け
        let fouls = view.fouls.you + 1;
        if fouls >= MAX_FOULS {
            return -1.0;
        }
        let mut log2 = copy_log(log);
        log2.record(Observation::MyFoul {
            move_number: pos.move_number(),
            usi: usi.to_string(),
        });
        let mut tried = foul_tried.clone();
        tried.insert(usi.to_string());
        let mut v = view.clone();
        v.fouls.you = fouls;
        return net.forward(&encode(&v, &log2, &tried)).1;
    }
    let mut p = pos.clone();
    let captured = p.play_unchecked(mv);
    let mut log2 = copy_log(log);
    log2.record(Observation::MyMove {
        move_number: p.move_number(),
        usi: usi.to_string(),
        captured: captured.map(unpromote_role),
    });
    if let Some(w) = mated_winner(&p) {
        return if w == me { 1.0 } else { -1.0 };
    }
    if p.in_check(me.other()) {
        log2.record(Observation::Check { in_check: me.other() });
    }
    // 相手の応手（推定器の相手モデル。相手の反則は模擬しない）
    let Some(reply) = predict_opp_reply(&p, me, &[], &[], foul_tried.len() as u32, rng) else {
        return 0.0;
    };
    let lost_at = match reply {
        ShogiMove::Board { to, .. } if p.piece_at(to).is_some_and(|q| q.color == me) => {
            Some(crate::board::make_usi_square(to))
        }
        _ => None,
    };
    p.play_unchecked(&reply);
    log2.record(Observation::OpponentMoved {
        move_number: p.move_number(),
        captured_my_piece_at: lost_at,
    });
    if let Some(w) = mated_winner(&p) {
        return if w == me { 1.0 } else { -1.0 };
    }
    if p.in_check(me) {
        log2.record(Observation::Check { in_check: me });
    }
    let v = view_of(&p, me, view.fouls.clone(), &view.game_id);
    net.forward(&encode(&v, &log2, &HashSet::new())).1
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
        cands.truncate(self.params.k);

        // 粒子を重みどおりに引く（厳密粒子が無ければ taint 込み）
        let est = self.est.as_ref().unwrap();
        let weighted = crate::scenario_core::weighted_unique_particles(est);
        let strict: Vec<(&Position, f64)> = weighted.iter().filter(|w| w.2).map(|w| (w.0, w.1)).collect();
        let pool: Vec<(&Position, f64)> = if strict.is_empty() {
            weighted.iter().map(|w| (w.0, w.1)).collect()
        } else {
            strict
        };
        // 粒子の手番が自分でないもの（推定器が壊れたとき）は使わない
        let pool: Vec<(Position, f64)> = pool
            .into_iter()
            .filter(|(p, _)| p.turn() == me)
            .map(|(p, w)| (p.clone(), w))
            .collect();
        let pool_total: f64 = pool.iter().map(|x| x.1).sum();

        let mut q = vec![0.0f64; cands.len()];
        // Q の標準誤差（診断用。粒子と応手のサンプル数が足りているかを見る）
        let mut se = vec![0.0f64; cands.len()];
        let searched = self.params.eta != 0.0 && !pool.is_empty() && pool_total > 0.0 && cands.len() > 1;
        if searched {
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
                let usi = mv.to_usi();
                let vals: Vec<f64> = samples
                    .iter()
                    .map(|s| f64::from(leaf_value(&self.net, view, log, foul_tried, s, &mv, &usi, &mut self.rng)))
                    .collect();
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
}
