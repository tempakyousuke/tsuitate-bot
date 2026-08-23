//! **鉢合わせリスク**の実測（2026-08-03、ユーザー指摘「4七の位置の歩は相手から
//! 取られやすい」が発端）。
//!
//! `strategy::exposed_capture_risk` は「相手がその駒の位置を知っているほど
//! 取られやすい」（`knownness_map`）という前提で当たりを重み付けるが、
//! ついたてには**相手が知らなくてもぶつかる**類型がある:
//! - 敵歩の正面に立っている駒は、相手が歩を突くという普通の手だけで取られる
//! - 敵の飛び駒の開き線上の駒は、相手がその線を走るだけで取られる
//!
//! 真実の棋譜（対局記録の `game:end`）を再生して、各決定点で手番側 S の
//! 相手 O の駒を類型に分け、**その直後の S の着手で実際に取られた割合**を数える。
//! `exposed_base`(0.4576) / `exposed_known`(0.1659) が妥当かの較正データになる。
//!
//! usage: cargo run --release --bin collision_probe -- records/*.jsonl

use std::collections::HashSet;

use tsuitate_bot::board::Coord;
use tsuitate_bot::protocol::{Color, Role};
use tsuitate_bot::shogi::{Position, ShogiMove, parse_usi};
use tsuitate_bot::truth_replay::{for_each_decision, load_end};

#[derive(Default, Clone, Copy)]
struct Tally {
    n: u64,
    captured: u64,
}

impl Tally {
    fn add(&mut self, captured: bool) {
        self.n += 1;
        self.captured += u64::from(captured);
    }
    fn rate(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.captured as f64 / self.n as f64
        }
    }
}

fn show(label: &str, t: Tally) {
    println!(
        "{label:<44} n={:>7}  取られた={:>6}  率={:>6.2}%",
        t.n,
        t.captured,
        100.0 * t.rate()
    );
}

/// マス sq の「S 側から見た正面手前」= S の歩がそこから sq へ進めるマス
fn pawn_from(sq: Coord, s: Color) -> Option<Coord> {
    let rank = match s {
        Color::Sente => sq.rank + 1,
        Color::Gote => sq.rank - 1,
    };
    (1..=9).contains(&rank).then_some(Coord {
        file: sq.file,
        rank,
    })
}

/// sq に S の飛び駒（飛角香・龍馬）の利きが**空き直線越しに**届いているか。
/// 隣接の利きは「ぶつかる」類型ではないので除く（歩の正面は別枠で数える）
fn slider_reaches(pos: &Position, sq: Coord, s: Color) -> bool {
    const DIRS: [(i8, i8); 8] = [
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ];
    for (df, dr) in DIRS {
        let mut d = 1i8;
        loop {
            let c = Coord {
                file: sq.file + df * d,
                rank: sq.rank + dr * d,
            };
            if !(1..=9).contains(&c.file) || !(1..=9).contains(&c.rank) {
                break;
            }
            let Some(p) = pos.piece_at(c) else {
                d += 1;
                continue;
            };
            if d >= 2 && p.color == s {
                let reaches = match p.role {
                    Role::Rook | Role::Dragon => df == 0 || dr == 0,
                    Role::Bishop | Role::Horse => df != 0 && dr != 0,
                    // 香は前方（sq から見て後ろ）の直線だけ
                    Role::Lance => {
                        df == 0
                            && match s {
                                Color::Sente => dr == 1,
                                Color::Gote => dr == -1,
                            }
                    }
                    _ => false,
                };
                if reaches {
                    return true;
                }
            }
            break;
        }
    }
    false
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: collision_probe <records/*.jsonl...>");
        std::process::exit(2);
    }

    let mut all = Tally::default();
    let mut attacked = Tally::default();
    let mut pawn_head = Tally::default();
    let mut pawn_head_undefended = Tally::default();
    let mut slider = Tally::default();
    let mut other_attacked = Tally::default();
    let mut quiet = Tally::default();
    // knownness_map と同じ3分類（相手がその駒の位置をどれだけ知っているか）
    let mut known_revealed = Tally::default();
    let mut known_home = Tally::default();
    let mut known_none = Tally::default();
    // 「知られていない」駒だけを類型で割る（鉢合わせは知識と独立か？）
    let mut unknown_pawn_head = Tally::default();
    let mut unknown_other = Tally::default();
    let mut games = 0u64;

    for path in &paths {
        let Some(end) = load_end(path) else {
            continue;
        };
        let moves = end.moves.clone();
        // revealed[c] = c 側の駒のうち「相手が位置を知っている」マス
        // （そこで c が相手の駒を取った = 相手は取られたマスを通知される）。
        // strategy::knownness_map と同じ更新規則
        let mut revealed: [HashSet<Coord>; 2] = [HashSet::new(), HashSet::new()];
        let mut touched: HashSet<Coord> = HashSet::new();
        let initial = Position::initial();
        let ok = for_each_decision(&end, |pos: &Position, s: Color, _log, id| {
            let o = s.other();
            // この決定点で S が実際に指した手（取ったマス）
            let captured_at = moves
                .get(id as usize)
                .and_then(|m| parse_usi(&m.usi))
                .and_then(|mv| match mv {
                    ShogiMove::Board { to, .. } => Some(to),
                    ShogiMove::Drop { .. } => None,
                })
                .filter(|to| pos.piece_at(*to).is_some_and(|p| p.color == o));

            for (sq, piece) in pos.pieces() {
                if piece.color != o || piece.role == Role::King {
                    continue;
                }
                let taken = captured_at == Some(sq);
                all.add(taken);
                let is_attacked = pos.is_attacked(sq, s);
                if !is_attacked {
                    quiet.add(taken);
                    continue;
                }
                attacked.add(taken);
                let head = pawn_from(sq, s).is_some_and(|c| {
                    pos.piece_at(c)
                        .is_some_and(|p| p.color == s && p.role == Role::Pawn)
                });
                if head {
                    pawn_head.add(taken);
                    if pos.attack_count(sq, o) == 0 {
                        pawn_head_undefended.add(taken);
                    }
                } else if slider_reaches(pos, sq, s) {
                    slider.add(taken);
                } else {
                    other_attacked.add(taken);
                }

                // 相手（S）がこの駒の位置をどれだけ知っているか
                let is_revealed = revealed[o as usize].contains(&sq);
                let is_home = !touched.contains(&sq)
                    && initial
                        .piece_at(sq)
                        .is_some_and(|p| p.color == o && p.role == piece.role);
                if is_revealed {
                    known_revealed.add(taken);
                } else if is_home {
                    known_home.add(taken);
                } else {
                    known_none.add(taken);
                    if head {
                        unknown_pawn_head.add(taken);
                    } else {
                        unknown_other.add(taken);
                    }
                }
            }

            // この決定点の着手で revealed / touched を更新する
            if let Some(mv) = moves.get(id as usize).and_then(|m| parse_usi(&m.usi)) {
                let set = &mut revealed[s as usize];
                match mv {
                    ShogiMove::Board { from, to, .. } => {
                        set.remove(&from);
                        if pos.piece_at(to).is_some() {
                            set.insert(to);
                        } else {
                            set.remove(&to);
                        }
                        touched.insert(from);
                        touched.insert(to);
                    }
                    ShogiMove::Drop { to, .. } => {
                        set.remove(&to);
                        touched.insert(to);
                    }
                }
            }
        });
        if ok {
            games += 1;
        }
    }

    println!("対局 {games} 局 / 決定点ごとに相手側の全駒を分類\n");
    show("全駒（玉を除く）", all);
    show("  当たっていない", quiet);
    show("  当たっている", attacked);
    show("    うち 敵歩の正面（鉢合わせ）", pawn_head);
    show("      うち 紐なし", pawn_head_undefended);
    show("    うち 敵の飛び駒の開き線上（隣接を除く）", slider);
    show("    うち それ以外の当たり", other_attacked);
    println!();
    println!("当たっている駒を knownness_map の3分類で割る:");
    show("  revealed（そこで取ったので相手は知っている）", known_revealed);
    show("  home（初期位置から動いていない）", known_home);
    show("  none（相手は位置を知らないはず）", known_none);
    show("    うち 敵歩の正面", unknown_pawn_head);
    show("    うち それ以外", unknown_other);
    println!(
        concat!(
            "\n参考: モデルの重みは exposed_base={:.4} + exposed_known={:.4}×knownness\n",
            "　　　= 未知 {:.4} 〜 既知 {:.4}（比 {:.2}倍）"
        ),
        0.4576,
        0.1659,
        0.4576,
        0.4576 + 0.1659,
        (0.4576 + 0.1659) / 0.4576
    );
}
