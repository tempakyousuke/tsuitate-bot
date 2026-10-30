//! 診断: 探索（`rl_search`）が手番ごとの手を良くしているかを、**真の局面からの指し継ぎ**で直接測る
//! （runtime には何も入らない）。
//!
//! 方策 A（重み `net`）と相手 B（重み `opp`）を素の方策（温度 1 のサンプリング = `rl_policy:` と同じ）で指させ、
//! A の決定点を確率 `rate` で抜き出して、その時点の探索（`rl_search:<net>,eta=30`）が出す候補（上位 k 手）ごとに次を測る:
//!
//! - `q`: 探索の Q（推定器の粒子 × 相手モデルの応手 × 価値ヘッド。探索そのもの）
//! - `q_tp`: 粒子の代わりに**真の局面**を使った Q（相手モデルの応手 × 価値ヘッド）
//! - `q_tr`: 真の局面で、応手も**実際の相手 B の方策**から引いた Q（価値ヘッドだけが近似）
//! - `w`: 真の局面からその手を指し、以後は A・B とも素の方策で終局まで指し継いだ得点の平均（+1/0/−1。正解の代わり）
//!
//! 反則になる手は探索と同じ規約で扱う: `q_tp` / `q_tr` は同じ局面で合法な他候補の π 加重平均（反則して π で指し直す）、
//! `w` は実際に反則として審判へ渡し、A の方策が指し直す。指し継ぎは候補をまたいで同じ乱数（共通乱数法）を使う。
//! `w` は、どの Q と η ∈ {3, 10, 30} で引き直しても π' が `min_mass` に届かない候補では測らない（null）。
//! そういう候補しか無い決定点（探索が手を動かしようがない）は指し継ぎを丸ごと省く。
//!
//! ほかに、粒子の偏り（`belief`: 最大の π の手の後に相手モデルの応手で王手される・駒を取られる率と、
//! 相手の駒の前進を、探索と同じ規約で引いた粒子の上と真の局面の上で並べる）を記録する。
//! `alt=<重み>` を渡すと、`q_tr` と同じ末端を別のネットの価値ヘッドでも評価する（`q_tr_alt`。価値ヘッドの鋭さを
//! 同じ局面で比べる）。
//!
//! 出力は1行=1決定点の JSONL（`kind: probe`）と1行=1局（`kind: game`）。集計は `scripts/search_probe/report.py`。
//!
//! ```sh
//! cargo run --release --bin search_probe -- net=models/rl_mmd_v4.bin opp=models/rl_mmd_v4.bin games=40 seed=1 out=probe.jsonl
//! ```

use std::collections::HashSet;
use std::io::Write;
use std::sync::Arc;

use rand::{Rng, SeedableRng, rngs::StdRng};

use tsuitate_bot::board::Coord;
use tsuitate_bot::estimator::predict_opp_reply;
use tsuitate_bot::observation::{Observation, ObservationLog};
use tsuitate_bot::protocol::{Color, PlayerView};
use tsuitate_bot::referee::Referee;
use tsuitate_bot::rl::action::{decode_usi, legal_mask};
use tsuitate_bot::rl::encode::encode_planes;
use tsuitate_bot::rl::policy_net::PolicyNet;
use tsuitate_bot::rl::search::{Leaf, RlSearch, step_leaf};
use tsuitate_bot::selfplay::GameResult;
use tsuitate_bot::shogi::{Position, ShogiMove, parse_usi};
use tsuitate_bot::strategy::Strategy;

struct Args {
    net: String,
    opp: String,
    games: u32,
    seed: u64,
    rate: f64,
    skip: u32,
    rollouts: u32,
    replies: u32,
    k: usize,
    n: usize,
    min_mass: f64,
    /// 同じ末端（q_tr の局面）を別のネットの価値ヘッドでも評価する（価値ヘッドの鋭さを同じ局面で比べる）
    alt: String,
    out: String,
}

fn parse_args() -> Args {
    let mut a = Args {
        net: String::new(),
        opp: String::new(),
        games: 10,
        seed: 1,
        rate: 0.25,
        skip: 0,
        rollouts: 8,
        replies: 8,
        k: 8,
        n: 16,
        min_mass: 0.005,
        alt: String::new(),
        out: "search_probe.jsonl".into(),
    };
    for kv in std::env::args().skip(1) {
        let (k, v) = kv.split_once('=').unwrap_or_else(|| panic!("{kv}: key=値 の形で渡す"));
        match k {
            "net" => a.net = v.into(),
            "opp" => a.opp = v.into(),
            "games" => a.games = v.parse().unwrap(),
            "seed" => a.seed = v.parse().unwrap(),
            "rate" => a.rate = v.parse().unwrap(),
            "skip" => a.skip = v.parse().unwrap(),
            "rollouts" => a.rollouts = v.parse().unwrap(),
            "replies" => a.replies = v.parse().unwrap(),
            "k" => a.k = v.parse().unwrap(),
            "n" => a.n = v.parse().unwrap(),
            "min_mass" => a.min_mass = v.parse().unwrap(),
            "alt" => a.alt = v.into(),
            "out" => a.out = v.into(),
            _ => panic!("{k}: 未知の引数"),
        }
    }
    assert!(!a.net.is_empty() && !a.opp.is_empty(), "net= と opp=（どちらも重みのパス）は必須");
    a
}

fn load(path: &str) -> Arc<PolicyNet> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    Arc::new(PolicyNet::from_bytes(&bytes).unwrap_or_else(|e| panic!("{path}: {e}")))
}

/// 素の方策（`RlPolicy` の非 greedy と同じ: 新しいマスクの上で温度 1 のサンプリング）。
/// 重みを共有して乱数だけ持つので、指し継ぎのたびに安く作り直せる
struct NetPlayer {
    net: Arc<PolicyNet>,
    rng: StdRng,
}

impl NetPlayer {
    fn new(net: &Arc<PolicyNet>, seed: u64) -> Self {
        NetPlayer {
            net: net.clone(),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn choose(&mut self, view: &PlayerView, log: &ObservationLog, tried: &HashSet<String>) -> Option<String> {
        let mask = legal_mask(view, log, tried);
        let legal: Vec<usize> = (0..mask.len()).filter(|&a| mask[a]).collect();
        if legal.is_empty() {
            return None;
        }
        let (logits, _) = self.net.forward(&encode_planes(self.net.planes(), view, log, tried));
        let max = legal.iter().map(|&a| logits[a]).fold(f32::MIN, f32::max);
        let weights: Vec<f64> = legal.iter().map(|&a| f64::from(logits[a] - max).exp()).collect();
        let mut r = self.rng.random::<f64>() * weights.iter().sum::<f64>();
        let mut chosen = *legal.last().unwrap();
        for (&a, w) in legal.iter().zip(&weights) {
            if r < *w {
                chosen = a;
                break;
            }
            r -= w;
        }
        decode_usi(chosen, view.your_color)
    }
}

fn replay(hist: &[String]) -> Referee {
    let mut r = Referee::new();
    for usi in hist {
        r.step(usi, 0);
    }
    r
}

fn score(result: GameResult, me: Color) -> f64 {
    match result {
        GameResult::Win(c) if c == me => 1.0,
        GameResult::Win(_) => -1.0,
        GameResult::Draw => 0.0,
    }
}

/// `rl_search` の相手モデルへ渡す露見マス・触ったマス（search.rs の my_squares / push_move_squares と同じ定義）
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

fn my_squares(log: &ObservationLog) -> (Vec<Coord>, Vec<Coord>) {
    let (mut captures, mut touched) = (vec![], vec![]);
    for e in log.events() {
        if let Observation::MyMove { usi, captured, .. } = e {
            if let Some(mv) = parse_usi(usi) {
                push_move_squares(&mv, captured.is_some(), &mut captures, &mut touched);
            }
        }
    }
    (captures, touched)
}

/// 相手の駒（玉を除く）の前進の平均（相手の自陣の端 = 1 〜 こちらの自陣の端 = 9）と、こちらの自陣（3段）にいる相手の駒の数
fn opp_advance(pos: &Position, me: Color) -> (f64, f64) {
    let (mut sum, mut n, mut camp) = (0.0, 0.0, 0.0);
    for p in pos.pieces_of(me.other()) {
        if p.role == tsuitate_bot::protocol::Role::King {
            continue;
        }
        let r = f64::from(p.square.as_bytes()[1] - b'a' + 1);
        // 先手から見て段 a=1 が相手の自陣の端。後手から見ると逆
        let adv = if me == Color::Sente { r } else { 10.0 - r };
        sum += adv;
        n += 1.0;
        if adv >= 7.0 {
            camp += 1.0;
        }
    }
    (if n > 0.0 { sum / n } else { 0.0 }, camp)
}

/// 局面 `pos` で `mv` を指し、相手モデルの応手を1手引いて、(自分の駒が取られたか, 王手されたか)
#[allow(clippy::too_many_arguments)]
fn reply_events(
    pos: &Position,
    mv: &ShogiMove,
    me: Color,
    captures0: &[Coord],
    touched0: &[Coord],
    my_fouls: u32,
    rng: &mut StdRng,
) -> Option<(bool, bool)> {
    if !pos.is_legal(mv) {
        return None;
    }
    let mut p = pos.clone();
    let captured = match *mv {
        ShogiMove::Board { to, .. } => p.piece_at(to).is_some(),
        ShogiMove::Drop { .. } => false,
    };
    p.play_unchecked(mv);
    if p.outcome().is_some() {
        return Some((false, false));
    }
    let (mut captures, mut touched) = (captures0.to_vec(), touched0.to_vec());
    push_move_squares(mv, captured, &mut captures, &mut touched);
    let Some(r) = predict_opp_reply(&p, me, &captures, &touched, my_fouls, rng) else {
        return Some((false, false));
    };
    let lost = match r {
        ShogiMove::Board { to, .. } => p.piece_at(to).is_some_and(|q| q.color == me),
        ShogiMove::Drop { .. } => false,
    };
    p.play_unchecked(&r);
    Some((lost, p.in_check(me)))
}

fn value_at(net: &PolicyNet, view: &PlayerView, log: &ObservationLog, tried: &HashSet<String>) -> f64 {
    f64::from(net.forward(&encode_planes(net.planes(), view, log, tried)).1)
}

fn leaf_value(net: &PolicyNet, leaf: Leaf) -> f64 {
    match leaf {
        Leaf::Terminal(v) => f64::from(v),
        Leaf::Decision { view, log, foul_tried } => value_at(net, &view, &log, &foul_tried),
    }
}

/// 反則になる候補の値を、合法な他候補の値の π 加重平均で埋める（search.rs の redirect_foul_values と同じ規約）。
/// 合法な候補が無ければ `fallback`（反則の末端そのもの）を使う
fn redirect(probs: &[f64], vals: &mut [f64], legal: &[bool], fallback: &[f64]) {
    let den: f64 = (0..vals.len()).filter(|&i| legal[i]).map(|i| probs[i]).sum();
    let n_legal = legal.iter().filter(|&&l| l).count();
    let fill = if n_legal == 0 {
        None
    } else if den > 0.0 {
        Some((0..vals.len()).filter(|&i| legal[i]).map(|i| probs[i] * vals[i]).sum::<f64>() / den)
    } else {
        Some((0..vals.len()).filter(|&i| legal[i]).map(|i| vals[i]).sum::<f64>() / n_legal as f64)
    };
    for i in 0..vals.len() {
        if !legal[i] {
            vals[i] = fill.unwrap_or(fallback[i]);
        }
    }
}

/// π'(a) ∝ π(a)·exp(η·Q(a))（search.rs の選択と同じ式）
fn tilt(probs: &[f64], q: &[f64], eta: f64) -> Vec<f64> {
    let qmax = q.iter().copied().fold(f64::MIN, f64::max);
    let w: Vec<f64> = probs.iter().zip(q).map(|(p, qa)| p * (eta * (qa - qmax)).exp()).collect();
    let s: f64 = w.iter().sum();
    w.iter().map(|x| x / s).collect()
}

/// 1局を指し継ぐ（A・B とも素の方策）。A から見た得点
fn rollout(r: &mut Referee, me: Color, a: &mut NetPlayer, b: &mut NetPlayer) -> f64 {
    loop {
        if let Some((result, _)) = r.ended() {
            return score(result, me);
        }
        let side = r.to_move();
        let view = r.view(side, [0, 0], 0);
        let p = if side == me { &mut *a } else { &mut *b };
        let Some(u) = p.choose(&view, r.log(side), r.foul_tried(side)) else {
            return 0.0;
        };
        r.step(&u, 0);
    }
}

fn main() {
    let args = parse_args();
    let net = load(&args.net);
    let opp = load(&args.opp);
    let alt = (!args.alt.is_empty()).then(|| load(&args.alt));
    let search_name = format!("rl_search:{},k={},n={},eta=30,slow_ms=0", args.net, args.k, args.n);
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args.out).expect("out を作れない"));
    let mut rng = StdRng::seed_from_u64(args.seed ^ 0x9e37_79b9_7f4a_7c15);
    let t_all = std::time::Instant::now();
    let mut probes = 0u32;

    for g in 0..args.games {
        let gseed = args.seed * 1_000_003 + u64::from(g) * 7919;
        let me = if g % 2 == 0 { Color::Sente } else { Color::Gote };
        let mut a = NetPlayer::new(&net, gseed);
        let mut b = NetPlayer::new(&opp, gseed + 1);
        let mut search = RlSearch::from_name(&search_name, Some(gseed + 2)).expect("rl_search を作れない");
        let mut referee = Referee::new();
        let mut hist: Vec<String> = vec![];
        let mut game_probes = 0u32;
        let result = loop {
            if let Some((result, _)) = referee.ended() {
                break result;
            }
            let side = referee.to_move();
            let view = referee.view(side, [0, 0], g);
            let (log, tried) = (referee.log(side), referee.foul_tried(side));
            let usi = if side == me {
                search.prewarm(&view, log);
                if referee.plies() >= args.skip && rng.random::<f64>() < args.rate {
                    let t0 = std::time::Instant::now();
                    let mut rec = probe(&args, &net, &opp, alt.as_deref(), &mut search, &referee, &hist, &view, log, tried, &mut rng);
                    rec["game"] = g.into();
                    rec["probe_ms"] = (t0.elapsed().as_millis() as u64).into();
                    writeln!(out, "{rec}").unwrap();
                    out.flush().unwrap();
                    probes += 1;
                    game_probes += 1;
                }
                a.choose(&view, log, tried)
            } else {
                b.choose(&view, log, tried)
            };
            let Some(usi) = usi else {
                break GameResult::Draw;
            };
            hist.push(usi.clone());
            referee.step(&usi, 0);
        };
        let rec = serde_json::json!({
            "kind": "game", "game": g, "me": if me == Color::Sente { "sente" } else { "gote" },
            "score": score(result, me), "plies": referee.plies(), "probes": game_probes,
        });
        writeln!(out, "{rec}").unwrap();
        out.flush().unwrap();
        eprintln!(
            "game {g}: score {} plies {} probes {game_probes} (計 {probes}、{:.0}秒)",
            score(result, me),
            referee.plies(),
            t_all.elapsed().as_secs_f64()
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn probe(
    args: &Args,
    net: &Arc<PolicyNet>,
    opp: &Arc<PolicyNet>,
    alt: Option<&PolicyNet>,
    search: &mut RlSearch,
    referee: &Referee,
    hist: &[String],
    view: &PlayerView,
    log: &ObservationLog,
    tried: &HashSet<String>,
    rng: &mut StdRng,
) -> serde_json::Value {
    let me = view.your_color;
    search.choose(view, log, tried);
    let dbg = search.debug_state().expect("探索の記録");
    let mut rec = serde_json::json!({
        "kind": "probe", "ply": referee.plies(), "me": if me == Color::Sente { "sente" } else { "gote" },
        "fouls": view.fouls.you, "retry": tried.len(), "in_check": view.you_in_check,
        "value": dbg["value"], "searched": dbg["searched"], "pool": dbg["pool"], "tainted": dbg["tainted"],
        "kept_mass": dbg["kept_mass"], "search_ms": dbg["ms"],
    });
    if !dbg["searched"].as_bool().unwrap_or(false) {
        return rec;
    }
    let cands = dbg["cands"].as_array().unwrap().clone();
    let usis: Vec<String> = cands.iter().map(|c| c["usi"].as_str().unwrap().to_string()).collect();
    let probs: Vec<f64> = cands.iter().map(|c| c["p"].as_f64().unwrap()).collect();
    let q: Vec<f64> = cands.iter().map(|c| c["q"].as_f64().unwrap()).collect();
    let moves: Vec<ShogiMove> = usis.iter().map(|u| parse_usi(u).expect("候補は USI")).collect();
    let pos = referee.position().clone();
    let legal: Vec<bool> = moves.iter().map(|m| pos.is_legal(m)).collect();
    let kc = cands.len();

    // 反則の末端そのもの（合法な候補が1つも無いときの埋め草）
    let foul_leaf: Vec<f64> = moves
        .iter()
        .map(|m| leaf_value(net, step_leaf(view, log, tried, &pos, m, |_| None)))
        .collect();

    // q_tp: 真の局面 × 相手モデルの応手 × 価値ヘッド
    let (captures0, touched0) = my_squares(log);
    let my_fouls = tried.len() as u32;
    let mut q_tp = vec![0.0; kc];
    for ci in 0..kc {
        if !legal[ci] {
            continue;
        }
        let mv = moves[ci];
        let mut sum = 0.0;
        for _ in 0..args.replies {
            let leaf = step_leaf(view, log, tried, &pos, &mv, |next| {
                let (mut captures, mut touched) = (captures0.clone(), touched0.clone());
                let captured = match mv {
                    ShogiMove::Board { to, .. } => pos.piece_at(to).is_some(),
                    ShogiMove::Drop { .. } => false,
                };
                push_move_squares(&mv, captured, &mut captures, &mut touched);
                predict_opp_reply(next, me, &captures, &touched, my_fouls, rng)
            });
            sum += leaf_value(net, leaf);
        }
        q_tp[ci] = sum / f64::from(args.replies);
    }
    redirect(&probs, &mut q_tp, &legal, &foul_leaf);

    // 粒子の偏り: 相手の駒の前進と、最大の π の手の後に相手モデルの応手で駒を取られる・王手される率（粒子 vs 真の局面）
    if let Some(est) = search.estimator_ref() {
        let usable = |m: u8, t: u8| t == 0 && m < u8::MAX;
        let mut pool: Vec<(&Position, f64)> = vec![];
        for strict in [true, false] {
            let ws: Vec<(usize, f64)> = est
                .particles()
                .iter()
                .enumerate()
                .filter(|(i, p)| p.turn() == me && (!strict || usable(est.info_miss()[*i], est.phys_taint()[*i])))
                .map(|(i, _)| (i, est.log_weights()[i]))
                .collect();
            if let Some(mx) = ws.iter().map(|x| x.1).reduce(f64::max) {
                pool = ws.iter().map(|&(i, w)| (&est.particles()[i], (w - mx).exp())).collect();
                break;
            }
        }
        let tot: f64 = pool.iter().map(|x| x.1).sum();
        if tot > 0.0 {
            let (adv_t, camp_t) = opp_advance(&pos, me);
            let (mut adv_p, mut camp_p) = (0.0, 0.0);
            for (p, w) in &pool {
                let (a, c) = opp_advance(p, me);
                adv_p += a * w / tot;
                camp_p += c * w / tot;
            }
            let top = (0..kc).max_by(|&i, &j| probs[i].total_cmp(&probs[j])).unwrap();
            let mv = moves[top];
            const S: usize = 32;
            let (mut lost_t, mut chk_t, mut n_t) = (0.0, 0.0, 0.0);
            for _ in 0..S {
                if let Some((l, c)) = reply_events(&pos, &mv, me, &captures0, &touched0, my_fouls, rng) {
                    lost_t += f64::from(u8::from(l));
                    chk_t += f64::from(u8::from(c));
                    n_t += 1.0;
                }
            }
            let (mut lost_p, mut chk_p, mut n_p, mut foul_p) = (0.0, 0.0, 0.0, 0.0);
            for _ in 0..S {
                let mut r = rng.random::<f64>() * tot;
                let mut pick = pool[pool.len() - 1].0;
                for (p, w) in &pool {
                    if r < *w {
                        pick = p;
                        break;
                    }
                    r -= w;
                }
                match reply_events(pick, &mv, me, &captures0, &touched0, my_fouls, rng) {
                    Some((l, c)) => {
                        lost_p += f64::from(u8::from(l));
                        chk_p += f64::from(u8::from(c));
                        n_p += 1.0;
                    }
                    None => foul_p += 1.0,
                }
            }
            let rate = |x: f64, n: f64| if n > 0.0 { serde_json::json!(x / n) } else { serde_json::Value::Null };
            rec["belief"] = serde_json::json!({
                "adv_truth": adv_t, "adv_part": adv_p, "camp_truth": camp_t, "camp_part": camp_p,
                "top_usi": usis[top], "top_legal": legal[top],
                "lost_truth": rate(lost_t, n_t), "check_truth": rate(chk_t, n_t),
                "lost_part": rate(lost_p, n_p), "check_part": rate(chk_p, n_p), "foul_part": foul_p / S as f64,
            });
        }
    }

    // q_tr: 真の局面 × 実際の相手 B の応手 × 価値ヘッド（B が反則すれば B が指し直す）
    let mut q_tr = vec![0.0; kc];
    let mut q_tr_alt = vec![f64::NAN; kc];
    for ci in 0..kc {
        if !legal[ci] {
            continue;
        }
        let mut sum = 0.0;
        let mut sum_alt = 0.0;
        for j in 0..args.replies {
            let mut r = replay(hist);
            let mut bp = NetPlayer::new(opp, 0x5151 + u64::from(j));
            r.step(&usis[ci], 0);
            let (v, va) = loop {
                if let Some((result, _)) = r.ended() {
                    let s = score(result, me);
                    break (s, s);
                }
                let side = r.to_move();
                let v2 = r.view(side, [0, 0], 0);
                if side == me {
                    let (l2, t2) = (r.log(side), r.foul_tried(side));
                    break (value_at(net, &v2, l2, t2), alt.map_or(f64::NAN, |a| value_at(a, &v2, l2, t2)));
                }
                let Some(u) = bp.choose(&v2, r.log(side), r.foul_tried(side)) else {
                    break (0.0, 0.0);
                };
                r.step(&u, 0);
            };
            sum += v;
            sum_alt += va;
        }
        q_tr[ci] = sum / f64::from(args.replies);
        q_tr_alt[ci] = sum_alt / f64::from(args.replies);
    }
    redirect(&probs, &mut q_tr, &legal, &foul_leaf);
    if alt.is_some() {
        // 反則の候補は alt でも同じ規約（合法な他候補の π 加重平均）で埋める。埋め草は使わない（全候補が反則の決定点は稀）
        let fb = vec![f64::NAN; kc];
        redirect(&probs, &mut q_tr_alt, &legal, &fb);
    }

    // w を測る候補: どれかの Q・η ∈ {3, 10, 30} で π' が min_mass に届くもの（π 自身が届くものを含む）
    let mut relevant: Vec<bool> = probs.iter().map(|&p| p >= args.min_mass).collect();
    for qq in [&q, &q_tp, &q_tr] {
        for eta in [3.0, 10.0, 30.0] {
            for (ci, m) in tilt(&probs, qq, eta).iter().enumerate() {
                relevant[ci] |= *m >= args.min_mass;
            }
        }
    }
    let n_rel = relevant.iter().filter(|&&r| r).count();

    // w: 真の局面から指し継いだ得点（共通乱数法: 指し継ぎ j の乱数は候補によらない）
    let mut w: Vec<Option<f64>> = vec![None; kc];
    if n_rel > 1 {
        for ci in 0..kc {
            if !relevant[ci] {
                continue;
            }
            let mut sum = 0.0;
            for j in 0..args.rollouts {
                let base = 0xabcd_0000 + u64::from(j) * 2;
                let mut ap = NetPlayer::new(net, base);
                let mut bp = NetPlayer::new(opp, base + 1);
                let mut r = replay(hist);
                r.step(&usis[ci], 0);
                sum += rollout(&mut r, me, &mut ap, &mut bp);
            }
            w[ci] = Some(sum / f64::from(args.rollouts));
        }
    }

    rec["relevant"] = n_rel.into();
    rec["cands"] = cands
        .iter()
        .enumerate()
        .map(|(ci, c)| {
            serde_json::json!({
                "usi": usis[ci], "p": probs[ci], "q": c["q"], "se": c["se"], "foul": c["foul"],
                "legal": legal[ci], "q_tp": q_tp[ci], "q_tr": q_tr[ci], "w": w[ci],
                "q_tr_alt": if alt.is_some() && q_tr_alt[ci].is_finite() { serde_json::json!(q_tr_alt[ci]) } else { serde_json::Value::Null },
            })
        })
        .collect::<Vec<_>>()
        .into();
    rec
}
