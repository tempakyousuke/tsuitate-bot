//! **露見した歩の継続前進率**の実測（2026-08-22、quest_0809 36手目の 2三玉が発端）。
//!
//! 後手粒子は「1五で取った先手の歩が 1四→1三成と2手で来る」世界線に 2.0% しか
//! 置かなかった（人間は未説明2手＝最短手数ちょうどで五分と見る）。opp_move NN は
//! `hang_minor`（相手の利きがあるマスへの紐なし着地）で歩の前進を割り引くが、
//! 「その歩は直前に捕獲して位置が露見している＝意図を持って動いている駒」という
//! 情報は特徴量に無い。ここでは真実の棋譜を再生し、手番側 S の歩を類型に分けて
//! **その決定点で実際にその歩を突いた率**（即時）と **S の以後 k 手以内に突いた率**、
//! および同じ決定点で NN 事前（`opp_reply_weights`、観測制約なし）がその突きに置く
//! 質量を並べる。較正の3点測定（教師実測率 → 決定点内質量 → live probe）の前2点。
//!
//! 類型（ユーザー指摘で香・飛の後ろかどうかを分ける）:
//! - revealed: その歩が駒を取ったマスに立っている（相手は位置を知っている）
//! - advanced: 初期段から進んでいる
//! - backed_lance / backed_rook: 同じ筋の真後ろ（間は空き）に自分の香 / 飛・龍がいる
//! - ahead_attacked: 前のマスに相手の利きがある（NN の hang_minor 条件）
//! - zone_next: 次の1手で成れる
//!
//! usage: cargo run --release --bin pawn_push_probe -- [--k 4] <records/*.jsonl...>

use std::collections::{HashMap, HashSet};

use tsuitate_bot::board::Coord;
use tsuitate_bot::estimator::opp_reply_weights;
use tsuitate_bot::protocol::{Color, Role};
use tsuitate_bot::shogi::{Position, ShogiMove, parse_usi};
use tsuitate_bot::truth_replay::{for_each_decision, load_end};

#[derive(Clone, Copy, Default)]
struct Ev {
    /// 即時にその歩を突いたか
    now: bool,
    /// S の以後 k 手以内（即時含む。0 = 即時）に突いたか → 何手目か
    within: Option<u32>,
    /// 突いた手が成りだったか（within と同じ手）
    promoted: bool,
    /// NN 事前がその突き（成り・不成の合算）に置く質量
    nn_share: f64,
    /// 類型フラグ
    revealed: bool,
    advanced: bool,
    backed_lance: bool,
    backed_rook: bool,
    ahead_attacked: bool,
    ahead_empty: bool,
    zone_next: bool,
    human: bool,
}

#[derive(Default)]
struct Agg {
    n: u64,
    now: u64,
    within: u64,
    promoted: u64,
    nn_sum: f64,
}

impl Agg {
    fn add(&mut self, e: &Ev) {
        self.n += 1;
        self.now += u64::from(e.now);
        self.within += u64::from(e.within.is_some());
        self.promoted += u64::from(e.promoted);
        self.nn_sum += e.nn_share;
    }
}

fn pct(a: u64, n: u64) -> f64 {
    if n == 0 { 0.0 } else { 100.0 * a as f64 / n as f64 }
}

fn show(label: &str, a: &Agg, k: u32) {
    println!(
        "{label:<40} n={:>6}  即時={:>5.1}%  {k}手以内={:>5.1}%  うち成り={:>5.1}%  NN質量={:>5.1}%",
        a.n,
        pct(a.now, a.n),
        pct(a.within, a.n),
        pct(a.promoted, a.n),
        if a.n == 0 { 0.0 } else { 100.0 * a.nn_sum / a.n as f64 },
    );
}

fn ahead(sq: Coord, s: Color) -> Option<Coord> {
    let rank = match s {
        Color::Sente => sq.rank - 1,
        Color::Gote => sq.rank + 1,
    };
    (1..=9).contains(&rank).then_some(Coord { file: sq.file, rank })
}

fn behind(sq: Coord, s: Color) -> Option<Coord> {
    ahead(sq, s.other())
}

fn in_zone(sq: Coord, s: Color) -> bool {
    match s {
        Color::Sente => sq.rank <= 3,
        Color::Gote => sq.rank >= 7,
    }
}

fn home_rank(s: Color) -> i8 {
    match s {
        Color::Sente => 7,
        Color::Gote => 3,
    }
}

/// 真後ろの直線（間は空き）に自分の香 / 飛・龍がいるか
fn backed_by(pos: &Position, sq: Coord, s: Color) -> (bool, bool) {
    let mut cur = sq;
    while let Some(b) = behind(cur, s) {
        if let Some(p) = pos.piece_at(b) {
            if p.color == s {
                return match p.role {
                    Role::Lance => (true, false),
                    Role::Rook | Role::Dragon => (false, true),
                    _ => (false, false),
                };
            }
            return (false, false);
        }
        cur = b;
    }
    (false, false)
}

fn bot_color(path: &str) -> Option<Color> {
    let content = std::fs::read_to_string(path).ok()?;
    for line in content.lines() {
        let v: serde_json::Value = serde_json::from_str(line).ok()?;
        if v["type"] == "match" {
            return match v["your_color"].as_str()? {
                "sente" => Some(Color::Sente),
                "gote" => Some(Color::Gote),
                _ => None,
            };
        }
    }
    None
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut k = 4u32;
    if let Some(i) = args.iter().position(|a| a == "--k") {
        k = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(4);
        args.drain(i..=i + 1);
    }
    if args.is_empty() {
        eprintln!("usage: pawn_push_probe [--k 4] <records/*.jsonl...>");
        std::process::exit(2);
    }

    let mut events: Vec<Ev> = Vec::new();
    let mut games = 0u64;
    for path in &args {
        let Some(end) = load_end(path) else {
            continue;
        };
        let bot = bot_color(path);
        let moves = end.moves.clone();
        // revealed[c] = c の駒のうち「相手が位置を知っている」マス（そこで取った）
        let mut revealed: [HashSet<Coord>; 2] = [HashSet::new(), HashSet::new()];
        let mut touched: [HashSet<Coord>; 2] = [HashSet::new(), HashSet::new()];
        // (side, 歩のマス) -> まだ決着していないイベント（index, 作成時の手数カウンタ）
        let mut pending: HashMap<(usize, Coord), Vec<(usize, u32)>> = HashMap::new();
        let mut own_moves = [0u32; 2];
        let base = events.len();
        let ok = for_each_decision(&end, |pos: &Position, s: Color, _log, id| {
            let o = s.other();
            let si = s as usize;
            let actual = moves.get(id as usize).and_then(|m| parse_usi(&m.usi));
            // この決定点で実際に突かれた歩（from, 成り）
            let pushed = actual.and_then(|mv| match mv {
                ShogiMove::Board { from, to, promote } => pos
                    .piece_at(from)
                    .filter(|p| p.role == Role::Pawn && p.color == s)
                    .map(|_| (from, to, promote)),
                ShogiMove::Drop { .. } => None,
            });
            // 保留イベントの決着（同じマスの歩が動いた）
            if let Some((from, _, promote)) = pushed {
                if let Some(list) = pending.remove(&(si, from)) {
                    for (idx, created) in list {
                        let d = own_moves[si] - created;
                        if d <= k {
                            events[idx].within = Some(d);
                            events[idx].promoted = promote;
                        }
                    }
                }
            }
            // NN 事前（観測制約なし）。my_color = o 視点で S の手の重み
            let known: Vec<Coord> = revealed[o as usize].iter().copied().collect();
            let my_touched: Vec<Coord> = touched[o as usize].iter().copied().collect();
            let weights = opp_reply_weights(pos, o, &known, &my_touched, 0);
            let total: f64 = weights.iter().map(|(_, w)| w).sum();

            for (sq, piece) in pos.pieces() {
                if piece.color != s || piece.role != Role::Pawn {
                    continue;
                }
                let Some(a) = ahead(sq, s) else { continue };
                // 前が自駒なら突けない
                if pos.piece_at(a).is_some_and(|p| p.color == s) {
                    continue;
                }
                let (bl, br) = backed_by(pos, sq, s);
                let nn_share = if total > 0.0 {
                    weights
                        .iter()
                        .filter(|(mv, _)| matches!(mv, ShogiMove::Board { from, .. } if *from == sq))
                        .map(|(_, w)| w)
                        .sum::<f64>()
                        / total
                } else {
                    0.0
                };
                let now = pushed.is_some_and(|(from, _, _)| from == sq);
                let ev = Ev {
                    now,
                    within: if now { Some(0) } else { None },
                    promoted: now && pushed.is_some_and(|(_, _, p)| p),
                    nn_share,
                    revealed: revealed[si].contains(&sq),
                    advanced: sq.rank != home_rank(s),
                    backed_lance: bl,
                    backed_rook: br,
                    ahead_attacked: pos.is_attacked(a, o),
                    ahead_empty: pos.piece_at(a).is_none(),
                    zone_next: in_zone(a, s),
                    human: bot.is_some_and(|b| b != s),
                };
                events.push(ev);
                if !now {
                    pending
                        .entry((si, sq))
                        .or_default()
                        .push((events.len() - 1, own_moves[si]));
                }
            }
            own_moves[si] += 1;

            // revealed / touched の更新（collision_probe と同じ規則）
            if let Some(mv) = actual {
                let set = &mut revealed[si];
                match mv {
                    ShogiMove::Board { from, to, .. } => {
                        set.remove(&from);
                        if pos.piece_at(to).is_some() {
                            set.insert(to);
                        } else {
                            set.remove(&to);
                        }
                        // 取られた側の露見情報も消える
                        revealed[o as usize].remove(&to);
                        touched[si].insert(from);
                        touched[si].insert(to);
                    }
                    ShogiMove::Drop { to, .. } => {
                        set.remove(&to);
                        touched[si].insert(to);
                    }
                }
            }
        });
        if ok {
            games += 1;
        } else {
            events.truncate(base);
        }
    }

    let mut all = Agg::default();
    let mut rev = Agg::default();
    let mut rev_adv = Agg::default();
    let mut rev_empty = Agg::default();
    let mut rev_empty_att = Agg::default();
    let mut rev_empty_safe = Agg::default();
    let mut rev_lance = Agg::default();
    let mut rev_rook = Agg::default();
    let mut rev_unbacked = Agg::default();
    let mut rev_zone = Agg::default();
    let mut rev_human = Agg::default();
    let mut rev_bot = Agg::default();
    let mut nonrev_adv = Agg::default();
    let mut nonrev_adv_att = Agg::default();
    let mut nonrev_lance = Agg::default();
    let mut nonrev_rook = Agg::default();
    let mut nonrev_unbacked = Agg::default();
    let mut home = Agg::default();
    for e in &events {
        all.add(e);
        if e.revealed {
            rev.add(e);
            if e.advanced {
                rev_adv.add(e);
            }
            if e.ahead_empty {
                rev_empty.add(e);
                if e.ahead_attacked {
                    rev_empty_att.add(e);
                } else {
                    rev_empty_safe.add(e);
                }
            }
            if e.backed_lance {
                rev_lance.add(e);
            } else if e.backed_rook {
                rev_rook.add(e);
            } else {
                rev_unbacked.add(e);
            }
            if e.zone_next {
                rev_zone.add(e);
            }
            if e.human {
                rev_human.add(e);
            } else {
                rev_bot.add(e);
            }
        } else if e.advanced {
            nonrev_adv.add(e);
            if e.ahead_attacked {
                nonrev_adv_att.add(e);
            }
            if e.backed_lance {
                nonrev_lance.add(e);
            } else if e.backed_rook {
                nonrev_rook.add(e);
            } else {
                nonrev_unbacked.add(e);
            }
        } else {
            home.add(e);
        }
    }

    println!(
        "対局 {games} 局 / 決定点ごとに手番側の「突ける歩」を1行ずつ（計 {} 行）。\n\
         即時 = その決定点でその歩を突いた率、{k}手以内 = 手番側の以後 {k} 手以内に突いた率、\n\
         NN質量 = opp_reply_weights（観測制約なし）がその歩の突きに置く質量の平均\n",
        events.len()
    );
    show("全ての突ける歩", &all, k);
    show("  初期段の歩（未前進・未露見）", &home, k);
    show("  前進済み・未露見", &nonrev_adv, k);
    show("    うち 前のマスに相手の利き", &nonrev_adv_att, k);
    show("    うち 香の前", &nonrev_lance, k);
    show("    うち 飛の前", &nonrev_rook, k);
    show("    うち 後ろに飛び駒なし", &nonrev_unbacked, k);
    show("  露見歩（そこで駒を取った）", &rev, k);
    show("    うち 前進済み", &rev_adv, k);
    show("    うち 前が空き", &rev_empty, k);
    show("      うち 前のマスに相手の利き", &rev_empty_att, k);
    show("      うち 前のマスは安全", &rev_empty_safe, k);
    show("    うち 香の前", &rev_lance, k);
    show("    うち 飛の前", &rev_rook, k);
    show("    うち 後ろに飛び駒なし", &rev_unbacked, k);
    show("    うち 次で成れる", &rev_zone, k);
    show("    うち 人間の手番", &rev_human, k);
    show("    うち bot の手番", &rev_bot, k);
}
