//! 採点済み eval（`scenarios/quest31-*.kif` の `scores=`）を教師にして、
//! **粒子を一切使わない特徴量**と人間の点数の関係を出すオフライン診断。
//!
//! 「どの特徴が高得点手を説明するか」を先に測ってから評価項を足すためのもの
//! （手作り項を1つずつアリーナ/スイートで試すより桁違いに速い）。
//!
//! usage: cargo run --release --bin eval_features -- <kif...> > features.tsv

use std::collections::BTreeSet;

use tsuitate_bot::board::{Coord, defend_targets};
use tsuitate_bot::deduce::opp_king_candidates;
use tsuitate_bot::kifu::parse_kif;
use tsuitate_bot::protocol::{Color, Role, VisiblePiece};
use tsuitate_bot::scenario_core::{make_view, replay, side_idx};
use tsuitate_bot::shogi::{Position, ShogiMove, parse_usi};

fn cheb(a: Coord, b: Coord) -> i32 {
    (i32::from(a.file) - i32::from(b.file))
        .abs()
        .max((i32::from(a.rank) - i32::from(b.rank)).abs())
}

fn role_char(r: Role) -> &'static str {
    match r {
        Role::Pawn => "P",
        Role::Lance => "L",
        Role::Knight => "N",
        Role::Silver => "S",
        Role::Gold => "G",
        Role::Bishop => "B",
        Role::Rook => "R",
        Role::King => "K",
        Role::Tokin => "+P",
        Role::Promotedlance => "+L",
        Role::Promotedknight => "+N",
        Role::Promotedsilver => "+S",
        Role::Horse => "+B",
        Role::Dragon => "+R",
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    println!(
        "ply\tside\tusi\tscore\tdrop\trole\tpromote\tncand\tdist_to\tdist_from\tprox\tattacks_k\town_def\tenemy_camp\tadvance\tcapture"
    );
    for path in &args {
        let text = std::fs::read_to_string(path).expect("read kif");
        let head = text.lines().find(|l| l.starts_with("*scenario")).unwrap_or("");
        let Some(scores) = head
            .split_whitespace()
            .find(|t| t.starts_with("scores="))
            .map(|t| t.trim_start_matches("scores="))
        else {
            continue;
        };
        // fouls= 注入シナリオは局面が同じなので素の ply で近似する
        let ply: usize = head
            .split_whitespace()
            .find(|t| t.starts_with("ply="))
            .and_then(|t| t.trim_start_matches("ply=").parse().ok())
            .expect("ply=");
        let kifu = parse_kif(&text).expect("parse kif");
        let r = replay(&kifu, ply);
        let side = r.pos.turn();
        let log = &r.logs[side_idx(side)];
        let view = make_view(&r.pos, side, &r.fouls);
        let cands: BTreeSet<Coord> = opp_king_candidates(side, log);
        let ncand = cands.len();
        // 自駒の利き枚数
        let mut own = [0u8; 81];
        for p in &view.your_pieces {
            for s in defend_targets(&view.your_pieces, p, side) {
                let i = (usize::from(s.file as u8) - 1) * 9 + (usize::from(s.rank as u8) - 1);
                own[i] = own[i].saturating_add(1);
            }
        }
        // 自駒だけの盤面（利き判定用）
        let mut solo = Position::empty(side);
        for p in &view.your_pieces {
            if let Some(sq) = parse_sq(&p.square) {
                solo.set(
                    sq,
                    Some(tsuitate_bot::shogi::Piece {
                        color: side,
                        role: p.role,
                    }),
                );
            }
        }
        for (role, n) in &view.your_hand {
            solo.set_hand(side, *role, *n as u8);
        }
        for kv in scores.split(',') {
            let Some((usi, sv)) = kv.rsplit_once(':') else {
                continue;
            };
            let Ok(score) = sv.parse::<i32>() else { continue };
            let Some(mv) = parse_usi(usi) else { continue };
            let (to, from, role, promote, drop) = match mv {
                ShogiMove::Drop { role, to } => (to, None, role, false, true),
                ShogiMove::Board { from, to, promote } => {
                    let role = view
                        .your_pieces
                        .iter()
                        .find(|p: &&VisiblePiece| parse_sq(&p.square) == Some(from))
                        .map(|p| p.role);
                    match role {
                        Some(r) => (to, Some(from), r, promote, false),
                        None => continue, // 相手駒を動かす USI（採点表のブロック取り違え）
                    }
                }
            };
            let dist_to = cands.iter().map(|&k| cheb(to, k)).min().unwrap_or(9);
            let dist_from = from
                .map(|f| cands.iter().map(|&k| cheb(f, k)).min().unwrap_or(9))
                .unwrap_or(9);
            let prox: f64 = if ncand > 0 {
                cands.iter().map(|&k| 0.5f64.powi(cheb(to, k))).sum::<f64>() / ncand as f64
            } else {
                0.0
            };
            // その手を指した後、着地駒が玉候補マスに利くか（＝王手になりうるか）
            let mut after = solo.clone();
            let attacks_k = if after.is_pseudo_legal(&mv) {
                after.play_unchecked(&mv);
                cands.iter().any(|&k| after.attacks(to, k))
            } else {
                false
            };
            let idx = (usize::from(to.file as u8) - 1) * 9 + (usize::from(to.rank as u8) - 1);
            let own_def = own[idx];
            let enemy_camp = match side {
                Color::Sente => to.rank <= 3,
                Color::Gote => to.rank >= 7,
            };
            let advance = match (side, from) {
                (Color::Sente, Some(f)) => i32::from(f.rank) - i32::from(to.rank),
                (Color::Gote, Some(f)) => i32::from(to.rank) - i32::from(f.rank),
                _ => 0,
            };
            let capture = r.pos.piece_at(to).is_some_and(|p| p.color != side);
            println!(
                "{ply}\t{}\t{usi}\t{score}\t{}\t{}\t{}\t{ncand}\t{dist_to}\t{dist_from}\t{prox:.4}\t{}\t{own_def}\t{}\t{advance}\t{}",
                if side == Color::Sente { "S" } else { "G" },
                u8::from(drop),
                role_char(role),
                u8::from(promote),
                u8::from(attacks_k),
                u8::from(enemy_camp),
                u8::from(capture),
            );
        }
    }
}

fn parse_sq(s: &str) -> Option<Coord> {
    tsuitate_bot::board::parse_usi_square(s)
}
