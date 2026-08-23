//! **プローブ系施策の発火母数**の実測（2026-08-22）。
//!
//! 「この施策は全決定点の何%で発火しうるか」を**粒子を回さずに**数える。
//! 発火が 1% 未満なら、104局×2シード（同一設定でも 10pt 振れる）のアリーナでも
//! 188件の suite（同一コードの σ が 0.08）でも**原理的に判定できない**ので、
//! 実装する前にここで母数を測る（2026-08-22 の由来タグの敗因がまさにこれ）。
//!
//! 数えるのは真実の棋譜上の**上界**: 粒子は「本当にそこにいる駒」しか正しく
//! 当てられないので、真実で条件を満たす決定点の割合が、その施策が正しく
//! 発火しうる回数の上限になる。
//!
//! - **玉プローブ**（`TSUITATE_KING_PROBE_W`）: 玉の行き先 K に、**由来タグ付きの**
//!   相手駒 A（＝相手が自駒を取ったマスから追跡できている駒）が利いていて、
//!   かつ A を自分が玉以外の駒で当てている決定点。反則になれば A の存在が確定し、
//!   次の手で回収できる
//! - **経路プローブ**（`TSUITATE_PATH_PROBE_W`）: 飛び駒の移動経路を塞ぐ相手駒が
//!   由来タグ付きで、かつ自分が当てている決定点
//! - 比較用に**由来タグを要求しない**版（どの相手駒でもよい）も出す。両者の比が
//!   「粒子が幻に反応する余地」の大きさ
//!
//! usage: cargo run --release --bin anchor_probe -- <records/*.jsonl...>

use tsuitate_bot::board::Coord;
use tsuitate_bot::protocol::{Color, Role};
use tsuitate_bot::shogi::{Position, ShogiMove, parse_usi};
use tsuitate_bot::truth_replay::{for_each_decision, load_end};

/// 記録の bot 側（`your_color`）。相手（人間）側の決定点は数えない
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

/// 真実の上を走る由来タグ（`shogi::Anchors` と同じ意味論を棋譜再生で再現）。
/// side から見た「相手駒のうち、自駒を取ったマスから追跡できている駒」
#[derive(Default, Clone)]
struct TruthAnchors {
    /// (現在のマス, age = 観測確定後にその駒が動いた回数)
    slots: Vec<(Coord, u32)>,
}

impl TruthAnchors {
    fn pin(&mut self, sq: Coord) {
        self.slots.retain(|(s, _)| *s != sq);
        self.slots.push((sq, 0));
    }
    fn age_at(&self, sq: Coord) -> Option<u32> {
        self.slots.iter().find(|(s, _)| *s == sq).map(|(_, a)| *a)
    }
    /// 盤上移動の反映（どちらの側の手でも呼ぶ）
    fn on_move(&mut self, from: Coord, to: Coord) {
        self.slots.retain(|(s, _)| *s != to); // to にいた駒は取られた
        for (s, age) in self.slots.iter_mut() {
            if *s == from {
                *s = to;
                *age += 1;
            }
        }
    }
    fn on_drop(&mut self, to: Coord) {
        self.slots.retain(|(s, _)| *s != to);
    }
}

fn on_board(c: Coord) -> bool {
    (1..=9).contains(&c.file) && (1..=9).contains(&c.rank)
}

/// 自分（me）の玉以外の駒が sq へ利いているか
fn my_nonking_attacks(pos: &Position, sq: Coord, me: Color) -> bool {
    pos.pieces()
        .any(|(from, p)| p.color == me && p.role != Role::King && pos.attacks(from, sq))
}

#[derive(Default, Clone)]
struct Counts {
    decisions: u64,
    with_king_move: u64,
    king_any: u64,
    king_anchored: u64,
    path_any: u64,
    path_anchored: u64,
    /// 玉プローブが発火しうる決定点での、原因駒の age 分布
    age_hist: Vec<u64>,
}

impl Counts {
    fn bump_age(&mut self, age: u32) {
        let i = (age as usize).min(9);
        if self.age_hist.len() < 10 {
            self.age_hist.resize(10, 0);
        }
        self.age_hist[i] += 1;
    }
}

fn pct(a: u64, n: u64) -> f64 {
    if n == 0 { 0.0 } else { 100.0 * a as f64 / n as f64 }
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: anchor_probe <records/*.jsonl...>");
        std::process::exit(2);
    }
    let mut c = Counts::default();
    let mut games = 0u64;

    for path in &paths {
        let Some(end) = load_end(path) else { continue };
        // **bot 側の決定点だけ数える**（codex 指摘）。相手（人間）側の決定点を
        // 混ぜると母数が倍に膨らむ
        let Some(bot) = bot_color(path) else { continue };
        let before = c.clone();
        let moves = end.moves.clone();
        // 各手番側から見た「相手駒の由来タグ」
        let mut anchors = [TruthAnchors::default(), TruthAnchors::default()];
        let ok = for_each_decision(&end, |pos: &Position, side: Color, _log, id| {
            let me = side;
            let opp = me.other();
            let mine = &anchors[me as usize];
            // ProbeCtx は bot 側・王手中でないときだけ作られるので母数もそれに揃える
            let counted = me == bot && !pos.in_check(me);
            if counted {
                c.decisions += 1;
            }

            // --- 玉プローブの母数 ---
            if let Some(k) = pos.king_square(me).filter(|_| counted) {
                let mut has_king_move = false;
                let mut any = false;
                let mut anchored = false;
                let mut best_age = None;
                for df in -1i8..=1 {
                    for dr in -1i8..=1 {
                        if df == 0 && dr == 0 {
                            continue;
                        }
                        let dest = Coord { file: k.file + df, rank: k.rank + dr };
                        if !on_board(dest) {
                            continue;
                        }
                        if pos.piece_at(dest).is_some_and(|p| p.color == me) {
                            continue; // 自駒の上へは動けない
                        }
                        has_king_move = true;
                        // その行き先に利いている相手駒
                        for (sq, p) in pos.pieces() {
                            if p.color != opp || !pos.attacks(sq, dest) {
                                continue;
                            }
                            if !my_nonking_attacks(pos, sq, me) {
                                continue; // 回収できないなら情報を買っても使えない
                            }
                            any = true;
                            if let Some(age) = mine.age_at(sq) {
                                anchored = true;
                                best_age = Some(best_age.map_or(age, |a: u32| a.min(age)));
                            }
                        }
                    }
                }
                c.with_king_move += u64::from(has_king_move);
                c.king_any += u64::from(any);
                c.king_anchored += u64::from(anchored);
                if let Some(age) = best_age {
                    c.bump_age(age);
                }
            }

            // --- 経路プローブの母数（自分の飛び駒の移動で経路が塞がれる形）---
            let mut p_any = false;
            let mut p_anchored = false;
            if counted {
                for (from, p) in pos.pieces() {
                    if p.color != me {
                        continue;
                    }
                    // **駒種ごとの走り方向だけを見る**（codex 指摘: 飛の斜め・
                    // 角の直線・香の後方まで数えると母数が水増しになる）
                    let dirs: &[(i8, i8)] = match p.role {
                        Role::Rook | Role::Dragon => &[(0, 1), (0, -1), (1, 0), (-1, 0)],
                        Role::Bishop | Role::Horse => &[(1, 1), (1, -1), (-1, 1), (-1, -1)],
                        Role::Lance => match me {
                            Color::Sente => &[(0, -1)],
                            Color::Gote => &[(0, 1)],
                        },
                        _ => continue,
                    };
                    for &(df, dr) in dirs {
                        let mut cur = from;
                        loop {
                            cur = Coord { file: cur.file + df, rank: cur.rank + dr };
                            if !on_board(cur) {
                                break;
                            }
                            let Some(bp) = pos.piece_at(cur) else { continue };
                            // 経路上で最初に当たった駒だけが「経路を塞ぐ駒」
                            if bp.color == opp && my_nonking_attacks(pos, cur, me) {
                                p_any = true;
                                if mine.age_at(cur).is_some() {
                                    p_anchored = true;
                                }
                            }
                            break;
                        }
                    }
                }
            }
            c.path_any += u64::from(p_any);
            c.path_anchored += u64::from(p_anchored);

            // --- 真実の着手で由来タグを更新 ---
            if let Some(mv) = moves.get(id as usize).and_then(|m| parse_usi(&m.usi)) {
                let captured_sq = match mv {
                    ShogiMove::Board { to, .. } => pos
                        .piece_at(to)
                        .filter(|p| p.color != side)
                        .map(|_| to),
                    ShogiMove::Drop { .. } => None,
                };
                for (i, a) in anchors.iter_mut().enumerate() {
                    match mv {
                        ShogiMove::Board { from, to, .. } => a.on_move(from, to),
                        ShogiMove::Drop { to, .. } => a.on_drop(to),
                    }
                    // 「side が i 側の駒を取った」= i から見た相手駒の位置が確定
                    if let Some(sq) = captured_sq {
                        if i != side as usize {
                            a.pin(sq);
                        }
                    }
                }
            }
        });
        if ok {
            games += 1;
        } else {
            c = before; // 壊れた棋譜の途中集計は捨てる
        }
    }

    println!("対局 {games} 局 / 決定点 {} 個\n", c.decisions);
    println!(
        "玉の手がある決定点                    {:>6}  ({:.1}%)",
        c.with_king_move,
        pct(c.with_king_move, c.decisions)
    );
    println!(
        concat!(
            "  玉の行き先に「自分が当てている相手駒」がいる（由来タグ不問）\n",
            "　　　　　　　　　　　　　　　　　　  {:>6}  ({:.1}% / 全決定点)"
        ),
        c.king_any,
        pct(c.king_any, c.decisions)
    );
    println!(
        concat!(
            "  うち **由来タグ付き**（玉プローブが正しく発火しうる）\n",
            "　　　　　　　　　　　　　　　　　　  {:>6}  ({:.1}% / 全決定点、{:.1}% / 由来タグ不問)"
        ),
        c.king_anchored,
        pct(c.king_anchored, c.decisions),
        pct(c.king_anchored, c.king_any)
    );
    println!();
    println!(
        "経路プローブ（由来タグ不問）          {:>6}  ({:.1}%)",
        c.path_any,
        pct(c.path_any, c.decisions)
    );
    println!(
        "  うち **由来タグ付き**                {:>6}  ({:.1}%)",
        c.path_anchored,
        pct(c.path_anchored, c.decisions)
    );
    if !c.age_hist.is_empty() {
        println!("\n玉プローブが発火しうる決定点での原因駒の age（最小値）:");
        for (i, n) in c.age_hist.iter().enumerate() {
            if *n == 0 {
                continue;
            }
            let label = if i == 9 { "9以上".to_string() } else { i.to_string() };
            println!("  age {label}: {n} ({:.1}%)", pct(*n, c.king_anchored));
        }
    }
}
