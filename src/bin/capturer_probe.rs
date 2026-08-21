//! **取った駒の駒種の実際の分布**（2026-08-22、quest_0809 26手目の「2六の歩を
//! 取った駒」が発端）。ついたてでは「どのマスで自駒が取られたか」だけが通知され、
//! 取った駒の駒種は分からない。人間は「歩を取ったのは普通は歩」と見るが、
//! 先手粒子は 27手目時点で 2六 に角 46% を置いていた。
//!
//! 真実の棋譜を再生し、各捕獲について (取られた駒種, 取った駒種) を数える。
//! 取った側が人間か bot か、取られた駒が取られた側の自陣（3段）か敵陣かでも割る。
//! 粒子側の対応する数字は `scenario <名> diag --diag <取られたマス>` の駒種ビリーフ。
//!
//! usage: cargo run --release --bin capturer_probe -- <records/*.jsonl...>

use std::collections::BTreeMap;

use tsuitate_bot::protocol::{Color, Role};
use tsuitate_bot::shogi::{Position, ShogiMove, parse_usi};
use tsuitate_bot::truth_replay::{for_each_decision, load_end};

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

fn role_name(r: Role) -> &'static str {
    match r {
        Role::Pawn => "歩",
        Role::Lance => "香",
        Role::Knight => "桂",
        Role::Silver => "銀",
        Role::Gold => "金",
        Role::Bishop => "角",
        Role::Rook => "飛",
        Role::King => "玉",
        Role::Tokin => "と",
        Role::Promotedlance => "成香",
        Role::Promotedknight => "成桂",
        Role::Promotedsilver => "成銀",
        Role::Horse => "馬",
        Role::Dragon => "龍",
    }
}

/// 成駒は元の駒種へ（「歩が取った」には と金 も含める。ビリーフ側の比較では別に出す）
fn base(r: Role) -> Role {
    match r {
        Role::Tokin => Role::Pawn,
        Role::Promotedlance => Role::Lance,
        Role::Promotedknight => Role::Knight,
        Role::Promotedsilver => Role::Silver,
        Role::Horse => Role::Bishop,
        Role::Dragon => Role::Rook,
        r => r,
    }
}

#[derive(Default)]
struct Tally {
    n: u64,
    by_capturer: BTreeMap<&'static str, u64>,
}

impl Tally {
    fn add(&mut self, capturer: Role) {
        self.n += 1;
        *self.by_capturer.entry(role_name(capturer)).or_default() += 1;
    }
    fn show(&self, label: &str) {
        let mut v: Vec<(&&str, &u64)> = self.by_capturer.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1));
        let parts: Vec<String> = v
            .iter()
            .map(|(k, c)| format!("{k} {:.1}%", 100.0 * **c as f64 / self.n.max(1) as f64))
            .collect();
        println!("{label:<44} n={:>5}  {}", self.n, parts.join(" / "));
    }
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: capturer_probe <records/*.jsonl...>");
        std::process::exit(2);
    }
    let mut all = Tally::default();
    let mut pawn_taken = Tally::default();
    let mut pawn_taken_human = Tally::default();
    let mut pawn_taken_bot = Tally::default();
    let mut pawn_taken_own_camp = Tally::default();
    let mut pawn_taken_mid = Tally::default();
    let mut pawn_taken_opp_camp = Tally::default();
    // 取った駒が歩のとき、その歩は初期段から何段進んでいたか（露見歩の前進の対）
    let mut pawn_by_pawn_adv: BTreeMap<i8, u64> = BTreeMap::new();
    let mut games = 0u64;
    for path in &paths {
        let Some(end) = load_end(path) else { continue };
        let bot = bot_color(path);
        let moves = end.moves.clone();
        let ok = for_each_decision(&end, |pos: &Position, s: Color, _log, id| {
            let Some(ShogiMove::Board { from, to, .. }) =
                moves.get(id as usize).and_then(|m| parse_usi(&m.usi))
            else {
                return;
            };
            let Some(victim) = pos.piece_at(to).filter(|p| p.color != s) else {
                return;
            };
            let Some(mover) = pos.piece_at(from) else { return };
            let capturer = base(mover.role);
            all.add(capturer);
            if base(victim.role) != Role::Pawn || victim.role != Role::Pawn {
                return;
            }
            pawn_taken.add(capturer);
            if bot.is_some_and(|b| b != s) {
                pawn_taken_human.add(capturer);
            } else {
                pawn_taken_bot.add(capturer);
            }
            // 取られた側（o）から見た段: 自陣 3段 / 中段 / 敵陣 3段
            let o = s.other();
            let rank_from_o = match o {
                Color::Sente => to.rank,
                Color::Gote => 10 - to.rank,
            };
            if rank_from_o >= 7 {
                pawn_taken_own_camp.add(capturer);
            } else if rank_from_o >= 4 {
                pawn_taken_mid.add(capturer);
            } else {
                pawn_taken_opp_camp.add(capturer);
            }
            if mover.role == Role::Pawn {
                let adv = match s {
                    Color::Sente => 7 - from.rank,
                    Color::Gote => from.rank - 3,
                };
                *pawn_by_pawn_adv.entry(adv).or_default() += 1;
            }
        });
        if ok {
            games += 1;
        }
    }
    println!("対局 {games} 局 / 取った駒の駒種（成駒は元の駒種へ畳む）\n");
    all.show("全捕獲");
    pawn_taken.show("取られた駒が（未成の）歩");
    pawn_taken_human.show("  うち 取った側が人間");
    pawn_taken_bot.show("  うち 取った側が bot");
    pawn_taken_own_camp.show("  うち 取られた側の自陣（3段）で");
    pawn_taken_mid.show("  うち 中段で");
    pawn_taken_opp_camp.show("  うち 取られた側の敵陣（3段）で");
    println!("\n歩が歩を取ったとき、取った歩の初期段からの前進量:");
    for (adv, c) in &pawn_by_pawn_adv {
        println!("  {adv}段進んだ歩: {c}");
    }
}
