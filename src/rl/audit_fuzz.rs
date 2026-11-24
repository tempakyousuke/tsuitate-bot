//! 入力と行動マスクの点検（調査用。どれも `--ignored` で回す。結果は docs/rl-deepnash-design.md の
//! 「入力と行動マスクの点検」）。
//!
//! 既存のテスト（`encode::tests` / `action::tests`）は一様にランダムな手の対局で検査している。こちらは凍結版の
//! 方策どうしの対局に、確率 ε でランダムな手（旧マスクの候補から）を混ぜて、王手・駒取り・打ち・成りが実戦の頻度で
//! 出る局面を作る。
//!
//! - `実戦に近い対局で入力とマスクが真の盤面と矛盾しない`: マスク v2〜v6 が真の合法手を落とさない・確定事実と
//!   相手玉の候補と持ち駒の上界が真の盤面と矛盾しない（矛盾があれば落ちる）
//! - `履歴が欠けたログでの入力とマスク`: 再起動後の復帰・再接続での取りこぼしに当たるログを渡したときの誤りの割合
//! - `履歴を失った側の勝率`: 片方だけ途中でログを失う対局での勝率
//! - `入力のずれへの感度`: 手数のスカラーの書き換え・空のログで、方策と価値がどのくらい動くか
//! - `改善の候補の頻度`: まだ使っていない確定の推論（ピン・通ったマス）が当たる頻度と、手数の分布
//!
//! 環境変数: `AUDIT_GAMES`（局数）・`AUDIT_THREADS`（既定 4）・`AUDIT_EPS`（ランダムな手の割合）・
//! `AUDIT_PLAYERS`（カンマ区切りの戦略名）・`AUDIT_SEED`・`AUDIT_WEIGHTS` / `AUDIT_WEIGHT`（重みのパス）・
//! `AUDIT_WIPE`（ログを失う手数の範囲 `lo:hi`）

use std::collections::HashSet;

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::board::{Coord, parse_usi_square};
use crate::observation::Observation;
use crate::deduce::immobile_opponent_pawns;
use crate::protocol::{Color, Role};
use crate::referee::{Referee, StepResult};
use crate::rl::action::{
    NUM_ACTIONS, basic_legal_mask, decode_usi, deduced_facts, encode_move, legal_mask_with_version, normalize,
    square_index,
};
use crate::rl::encode::{
    D_KING, DeducedCache, E_HAND, E_KING_TIGHT, NUM_PLANES_V6, P_SCALAR, ROLE_TOTAL, S_HISTORY_BROKEN,
    encode_into_cached, moves_contiguous,
};
use crate::shogi::{HAND_ROLES, ShogiMove, parse_usi};

#[derive(Default, Clone, Debug)]
struct Tally {
    games: u64,
    decisions: u64,
    in_check: u64,
    /// マスクの版ごと: [真の合法手を落とした件数, 落とした決定点]
    dropped: [[u64; 2]; 7],
    /// マスクの版ごと: 候補のうち反則の手・うちマスクが落とした手（王手中 / 王手でない）
    illegal: [[u64; 2]; 2],
    removed: [[[u64; 2]; 2]; 7],
    /// 確定事実の矛盾
    bad_occ: u64,
    bad_block: u64,
    bad_checker: u64,
    bad_king: u64,
    bad_king_tight: u64,
    bad_pawn: u64,
    bad_hand: u64,
    broken: u64,
    n_occ: u64,
    n_block: u64,
    n_checker: u64,
    n_pawn: u64,
    /// 王手中: 候補のマス数の合計（v2 の入力）と、マスク v3 の規則（開き王手の仮説を自駒の遮りで絞る）での合計
    checker_cands: u64,
    checker_cands_v3: u64,
    examples: Vec<String>,
}

impl Tally {
    fn merge(&mut self, o: Tally) {
        self.games += o.games;
        self.decisions += o.decisions;
        self.in_check += o.in_check;
        for v in 0..7 {
            for k in 0..2 {
                self.dropped[v][k] += o.dropped[v][k];
                for c in 0..2 {
                    self.removed[v][c][k] += o.removed[v][c][k];
                }
            }
        }
        for c in 0..2 {
            for k in 0..2 {
                self.illegal[c][k] += o.illegal[c][k];
            }
        }
        self.bad_occ += o.bad_occ;
        self.bad_block += o.bad_block;
        self.bad_checker += o.bad_checker;
        self.bad_king += o.bad_king;
        self.bad_king_tight += o.bad_king_tight;
        self.bad_pawn += o.bad_pawn;
        self.bad_hand += o.bad_hand;
        self.broken += o.broken;
        self.n_occ += o.n_occ;
        self.n_block += o.n_block;
        self.n_checker += o.n_checker;
        self.n_pawn += o.n_pawn;
        self.checker_cands += o.checker_cands;
        self.checker_cands_v3 += o.checker_cands_v3;
        for e in o.examples {
            if self.examples.len() < 40 {
                self.examples.push(e);
            }
        }
    }

    fn note(&mut self, s: String) {
        if self.examples.len() < 40 {
            self.examples.push(s);
        }
    }
}

fn play_games(seed: u64, games: u64, eps: f64, players: &[String]) -> Tally {
    let mut t = Tally::default();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut obs = vec![0.0f32; NUM_PLANES_V6 * 81];
    for g in 0..games {
        let names = [
            &players[rng.random_range(0..players.len())],
            &players[rng.random_range(0..players.len())],
        ];
        let mut strategies = [
            crate::strategy::make_seeded(names[0], seed * 1000 + g * 2).expect("戦略名"),
            crate::strategy::make_seeded(names[1], seed * 1000 + g * 2 + 1).expect("戦略名"),
        ];
        let mut referee = Referee::new();
        let mut caches: [DeducedCache; 2] = Default::default();
        let mut moves: Vec<String> = vec![];
        loop {
            let side = referee.to_move();
            let i = (side == Color::Gote) as usize;
            let view = referee.view(side, [0, 0], g as u32);
            let (log, tried) = (referee.log(side), referee.foul_tried(side));
            let pos = referee.position().clone();
            let opp = |c: Coord| pos.piece_at(c).filter(|p| p.color != side);
            let ctx = |what: &str| {
                format!("seed {seed} game {g} {side:?} 手数 {} 反則済み {:?}: {what} / 棋譜 {}", view.move_number, tried, moves.join(" "))
            };
            t.decisions += 1;
            t.in_check += view.you_in_check as u64;
            let ck = view.you_in_check as usize;

            // 行動マスク: 真の合法手を落とさない
            let legal: HashSet<usize> = pos.legal_moves().iter().filter_map(|m| encode_move(m, side)).collect();
            let basic = basic_legal_mask(&view, tried);
            let n_illegal = (0..NUM_ACTIONS).filter(|&a| basic[a] && !legal.contains(&a)).count() as u64;
            t.illegal[ck][0] += n_illegal;
            t.illegal[ck][1] += (0..NUM_ACTIONS).filter(|&a| basic[a]).count() as u64;
            for v in 2..=6u32 {
                let mask = legal_mask_with_version(&view, log, tried, v);
                let lost: Vec<usize> = legal
                    .iter()
                    .copied()
                    .filter(|&a| basic[a] && !mask[a])
                    .collect();
                if !lost.is_empty() {
                    t.dropped[v as usize][0] += lost.len() as u64;
                    t.dropped[v as usize][1] += 1;
                    let usis: Vec<String> = lost.iter().filter_map(|&a| decode_usi(a, side)).collect();
                    t.note(ctx(&format!("マスク v{v} が真の合法手 {usis:?} を落とした")));
                }
                let removed = (0..NUM_ACTIONS).filter(|&a| basic[a] && !mask[a]).count() as u64;
                t.removed[v as usize][ck][0] += removed;
                t.removed[v as usize][ck][1] += 1;
            }

            // 確定事実（入力の v2 の推論）
            let facts = deduced_facts(&view, log, tried);
            for &c in &facts.occupied {
                t.n_occ += 1;
                if opp(c).is_none() {
                    t.bad_occ += 1;
                    t.note(ctx(&format!("確定した占有 {c:?} に相手の駒がいない")));
                }
            }
            for seg in &facts.blocked {
                t.n_block += 1;
                if !seg.iter().any(|&c| opp(c).is_some()) {
                    t.bad_block += 1;
                    t.note(ctx(&format!("遮りの間 {seg:?} が空")));
                }
            }
            if view.you_in_check {
                let king = pos.king_square(side).unwrap();
                let real: Vec<Coord> =
                    pos.pieces().filter(|(c, p)| p.color != side && pos.attacks(*c, king)).map(|(c, _)| c).collect();
                t.n_checker += 1;
                t.checker_cands += facts.checkers.len() as u64;
                let own: HashSet<Coord> = view.your_pieces.iter().filter_map(|p| parse_usi_square(&p.square)).collect();
                let cap = log.events().iter().rev().find_map(|e| match e {
                    Observation::OpponentMoved { captured_my_piece_at, .. } => {
                        Some(captured_my_piece_at.as_deref().and_then(parse_usi_square))
                    }
                    _ => None,
                });
                let v3 = crate::rl::action::checker_squares_at(&own, king, side, cap.flatten());
                t.checker_cands_v3 += v3.len() as u64;
                if !real.iter().all(|c| v3.contains(c)) {
                    t.bad_checker += 1;
                    t.note(ctx(&format!("王手駒 {real:?} が v3 の候補 {v3:?} に無い")));
                }
                if !real.iter().all(|c| facts.checkers.contains(c)) {
                    t.bad_checker += 1;
                    t.note(ctx(&format!("王手駒 {real:?} が候補 {:?} に無い", facts.checkers)));
                }
            }
            for c in immobile_opponent_pawns(side, log) {
                t.n_pawn += 1;
                if opp(c).map(|p| p.role) != Some(Role::Pawn) {
                    t.bad_pawn += 1;
                    t.note(ctx(&format!("動けない歩 {c:?} が相手の歩でない")));
                }
            }

            // 両方の席の観測（相手玉の候補・持ち駒の上界・履歴の印）
            for s in [Color::Sente, Color::Gote] {
                let j = (s == Color::Gote) as usize;
                let v = referee.view(s, [0, 0], g as u32);
                encode_into_cached(&v, referee.log(s), referee.foul_tried(s), &mut obs, &mut caches[j]);
                let at = |k: usize, c: Coord| obs[k * 81 + square_index(normalize(c, s))];
                let ok = pos.king_square(s.other()).unwrap();
                if at(D_KING, ok) != 1.0 {
                    t.bad_king += 1;
                    t.note(format!("seed {seed} game {g} 席 {s:?} 手数 {}: 相手玉 {ok:?} が候補（89番）に無い / 棋譜 {}", v.move_number, moves.join(" ")));
                }
                if at(E_KING_TIGHT, ok) != 1.0 {
                    t.bad_king_tight += 1;
                    t.note(format!("seed {seed} game {g} 席 {s:?} 手数 {}: 相手玉 {ok:?} が締めた候補に無い / 棋譜 {}", v.move_number, moves.join(" ")));
                }
                for (k, role) in HAND_ROLES.iter().enumerate() {
                    let real = pos.hand_count(s.other(), *role) as f32 / ROLE_TOTAL[k] as f32;
                    if obs[(E_HAND + k) * 81] + 1e-6 < real {
                        t.bad_hand += 1;
                        t.note(format!("seed {seed} game {g} 席 {s:?} 手数 {}: 相手の持ち駒 {role:?} の上界 {} < 本当 {real}", v.move_number, obs[(E_HAND + k) * 81]));
                    }
                }
                if obs[(P_SCALAR + S_HISTORY_BROKEN) * 81] != 0.0 || !moves_contiguous(referee.log(s), v.move_number) {
                    t.broken += 1;
                    t.note(format!("seed {seed} game {g} 席 {s:?} 手数 {}: 審判のログなのに履歴が欠けた扱い", v.move_number));
                }
            }
            // 手を選ぶ: ε でランダム（旧マスクの候補から）、それ以外は戦略
            let usi = if rng.random::<f64>() < eps {
                let acts: Vec<usize> = (0..NUM_ACTIONS).filter(|&a| basic[a]).collect();
                if acts.is_empty() {
                    break;
                }
                decode_usi(acts[rng.random_range(0..acts.len())], side).unwrap()
            } else {
                match strategies[i].choose(&view, log, tried) {
                    Some(u) => u,
                    None => break,
                }
            };
            match referee.step(&usi, 0) {
                StepResult::Ended { .. } => break,
                StepResult::Accepted => moves.push(usi),
                StepResult::Foul => moves.push(format!("({usi})")),
            }
        }
        t.games += 1;
    }
    t
}

#[test]
#[ignore]
fn 実戦に近い対局で入力とマスクが真の盤面と矛盾しない() {
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
    let games: u64 = env("AUDIT_GAMES", "40").parse().unwrap();
    let threads: u64 = env("AUDIT_THREADS", "4").parse().unwrap();
    let eps: f64 = env("AUDIT_EPS", "0.15").parse().unwrap();
    let seed: u64 = env("AUDIT_SEED", "1").parse().unwrap();
    let players: Vec<String> =
        env("AUDIT_PLAYERS", "rl_scratch5_v2,rl_mmdhandex_v2,rl_scratch5free_v1,rl_mmdopen_v15")
            .split(',')
            .map(String::from)
            .collect();
    let mut total = Tally::default();
    std::thread::scope(|sc| {
        let handles: Vec<_> = (0..threads)
            .map(|k| {
                let players = &players;
                sc.spawn(move || play_games(seed * 100 + k, games.div_ceil(threads), eps, players))
            })
            .collect();
        for h in handles {
            total.merge(h.join().unwrap());
        }
    });
    let t = &total;
    println!("{} 局・{} 決定点（うち王手中 {}）", t.games, t.decisions, t.in_check);
    println!(
        "候補のうち反則の手: 王手でない {} / {}、王手中 {} / {}",
        t.illegal[0][0], t.illegal[0][1], t.illegal[1][0], t.illegal[1][1]
    );
    for v in 2..=6 {
        println!(
            "マスク v{v}: 真の合法手を落とした {} 件（{} 決定点）/ 落とした候補 王手でない {}・王手中 {}",
            t.dropped[v][0], t.dropped[v][1], t.removed[v][0][0], t.removed[v][1][0]
        );
    }
    println!(
        "確定事実: 占有 {}（矛盾 {}）/ 遮り {}（矛盾 {}）/ 王手駒の候補 {}（矛盾 {}、候補の平均 {:.1} マス。マスク v3 の規則なら {:.1} マス）/ 動けない歩 {}（矛盾 {}）",
        t.n_occ,
        t.bad_occ,
        t.n_block,
        t.bad_block,
        t.n_checker,
        t.bad_checker,
        t.checker_cands as f64 / t.n_checker.max(1) as f64,
        t.checker_cands_v3 as f64 / t.n_checker.max(1) as f64,
        t.n_pawn,
        t.bad_pawn
    );
    println!(
        "相手玉の候補: 89番の矛盾 {} / 締めた版の矛盾 {} / 持ち駒の上界の矛盾 {} / 履歴が欠けた扱い {}",
        t.bad_king, t.bad_king_tight, t.bad_hand, t.broken
    );
    for e in &t.examples {
        println!("  例: {e}");
    }
    let bad = t.dropped.iter().map(|d| d[0]).sum::<u64>()
        + t.bad_occ
        + t.bad_block
        + t.bad_checker
        + t.bad_king
        + t.bad_king_tight
        + t.bad_pawn
        + t.bad_hand
        + t.broken;
    assert_eq!(bad, 0, "矛盾があった");
}

/// 入力の一部を書き換えたときに、方策と価値がどのくらい動くか（調査用）。本番でだけ起きる入力のずれの大きさを測る。
/// `AUDIT_WEIGHTS`（カンマ区切りの重みのパス）・`AUDIT_GAMES`
#[test]
#[ignore]
fn 入力のずれへの感度() {
    use crate::rl::action::legal_mask;
    use crate::rl::encode::{D_KING, S_MOVE_NUMBER, encode_planes};
    use crate::rl::policy_net::PolicyNet;
    use crate::selfplay::MAX_PLIES;
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
    let games: u64 = env("AUDIT_GAMES", "60").parse().unwrap();
    let players: Vec<String> = env("AUDIT_PLAYERS", "rl_scratch5_v2,rl_mmdhandex_v2").split(',').map(String::from).collect();
    let weights = env("AUDIT_WEIGHTS", "models/rl_scratch5_v2.bin,models/rl_mmdhandex_v2.bin");
    let softmax = |logits: &[f32], legal: &[usize]| -> Vec<f64> {
        let m = legal.iter().map(|&a| logits[a]).fold(f32::MIN, f32::max);
        let w: Vec<f64> = legal.iter().map(|&a| f64::from(logits[a] - m).exp()).collect();
        let s: f64 = w.iter().sum();
        w.into_iter().map(|x| x / s).collect()
    };
    for path in weights.split(',') {
        let net = PolicyNet::load(path).expect("重み");
        let planes = net.planes();
        // 書き換え: (名前, 手数のスカラーへ足す手数)。None は別の書き換え
        let shifts: [(&str, i32); 5] = [("+10手", 10), ("+30手", 30), ("+60手", 60), ("+100手", 100), ("200手に固定", -1)];
        // [手数の帯][書き換え] → (件数, 最善手が変わった, 全変動の合計, 価値の差の合計)
        let mut acc = vec![vec![(0u64, 0u64, 0.0f64, 0.0f64); shifts.len() + 1]; 4];
        let mut vsum = vec![(0u64, 0.0f64, 0.0f64); 4];
        let mut rng = StdRng::seed_from_u64(7);
        for g in 0..games {
            let mut strategies = [
                crate::strategy::make_seeded(&players[rng.random_range(0..players.len())], g * 2).unwrap(),
                crate::strategy::make_seeded(&players[rng.random_range(0..players.len())], g * 2 + 1).unwrap(),
            ];
            let mut referee = Referee::new();
            loop {
                let side = referee.to_move();
                let view = referee.view(side, [0, 0], g as u32);
                let (log, tried) = (referee.log(side), referee.foul_tried(side));
                let mask = legal_mask(&view, log, tried);
                let legal: Vec<usize> = (0..NUM_ACTIONS).filter(|&a| mask[a]).collect();
                if legal.is_empty() {
                    break;
                }
                let obs = encode_planes(planes, &view, log, tried);
                let (l0, v0) = net.forward(&obs);
                let p0 = softmax(&l0, &legal);
                let top0 = (0..legal.len()).max_by(|&a, &b| p0[a].total_cmp(&p0[b])).unwrap();
                let band = ((view.move_number.saturating_sub(1)) / 50).min(3) as usize;
                vsum[band].0 += 1;
                vsum[band].1 += f64::from(v0);
                vsum[band].2 += f64::from(v0.abs());
                let mut compare = |k: usize, o: &[f32]| {
                    let (l1, v1) = net.forward(o);
                    let p1 = softmax(&l1, &legal);
                    let top1 = (0..legal.len()).max_by(|&a, &b| p1[a].total_cmp(&p1[b])).unwrap();
                    let tv: f64 = p0.iter().zip(&p1).map(|(a, b)| (a - b).abs()).sum::<f64>() / 2.0;
                    let e = &mut acc[band][k];
                    e.0 += 1;
                    e.1 += (top0 != top1) as u64;
                    e.2 += tv;
                    e.3 += f64::from((v1 - v0).abs());
                };
                for (k, &(_, add)) in shifts.iter().enumerate() {
                    let mut o = obs.clone();
                    let v = if add < 0 { 1.0 } else { (view.move_number as i32 + add) as f32 / MAX_PLIES as f32 };
                    o[(P_SCALAR + S_MOVE_NUMBER) * 81..(P_SCALAR + S_MOVE_NUMBER + 1) * 81].fill(v);
                    compare(k, &o);
                }
                // 再起動の直後に当たる入力: 履歴を失った空のログ（視界だけ残る）
                if planes > D_KING {
                    let o = encode_planes(planes, &view, &crate::observation::ObservationLog::default(), tried);
                    compare(shifts.len(), &o);
                }
                let usi = match strategies[(side == Color::Gote) as usize].choose(&view, log, tried) {
                    Some(u) => u,
                    None => break,
                };
                if let StepResult::Ended { .. } = referee.step(&usi, 0) {
                    break;
                }
            }
        }
        println!("== {path}（{planes} 枚）");
        for (b, name) in ["1〜50手", "51〜100手", "101〜150手", "151手〜"].iter().enumerate() {
            let (n, v, va) = vsum[b];
            if n == 0 {
                continue;
            }
            println!("  {name}: {n} 決定点、価値の平均 {:+.3}・|価値| の平均 {:.3}", v / n as f64, va / n as f64);
            for (k, e) in acc[b].iter().enumerate() {
                if e.0 == 0 {
                    continue;
                }
                let label = if k < shifts.len() { shifts[k].0 } else { "履歴なし（空のログ）" };
                println!(
                    "    {label}: 最善手が変わる {:.1}% / 全変動 {:.3} / |価値の差| {:.3}",
                    100.0 * e.1 as f64 / e.0 as f64,
                    e.2 / e.0 as f64,
                    e.3 / e.0 as f64
                );
            }
        }
    }
}

/// 改善の候補が、実戦に近い対局でどのくらい起きるか（調査用）。
/// - ピン: 王手を受けていない手番に、玉以外の駒の「1マスの手・桂の跳び」が反則になったら、原因はピンしかない
///   （移動先は自駒でないので擬似合法。途中のマスが無いので遮りも無い）。その駒の、玉の筋から外れる手は全部反則
/// - 通ったマス: 自分の受理された飛び駒の手が通ったマスは空だった = 相手玉はそこにいない
#[test]
#[ignore]
fn 改善の候補の頻度() {
    use crate::rl::action::{candidate_usis, legal_mask};
    use crate::rl::encode::encode_planes;
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
    let games: u64 = env("AUDIT_GAMES", "400").parse().unwrap();
    let threads: u64 = env("AUDIT_THREADS", "4").parse().unwrap();
    let eps: f64 = env("AUDIT_EPS", "0.0").parse().unwrap();
    let players: Vec<String> =
        env("AUDIT_PLAYERS", "rl_scratch5_v2,rl_mmdhandex_v2,rl_scratch5free_v1").split(',').map(String::from).collect();
    #[derive(Default)]
    struct T {
        games: u64,
        decisions: u64,
        fouls: u64,
        fouls_free: u64,
        // ピンの規則
        pin_claims: u64,
        pin_unsound: u64,
        pin_extra: u64,
        pin_followup: u64,
        pin_fouls_first: u64,
        board_fouls_free: u64,
        // 通ったマス
        slides: u64,
        slide_hits: u64,
        slide_squares: u64,
        slide_unsound: u64,
        lens: Vec<u32>,
        notes: Vec<String>,
    }
    let run = |seed: u64, n: u64| -> T {
        let mut t = T::default();
        let mut rng = StdRng::seed_from_u64(seed);
        for g in 0..n {
            let mut strategies = [
                crate::strategy::make_seeded(&players[rng.random_range(0..players.len())], seed * 1000 + g * 2).unwrap(),
                crate::strategy::make_seeded(&players[rng.random_range(0..players.len())], seed * 1000 + g * 2 + 1).unwrap(),
            ];
            let mut referee = Referee::new();
            loop {
                let side = referee.to_move();
                let view = referee.view(side, [0, 0], g as u32);
                let (log, tried) = (referee.log(side), referee.foul_tried(side));
                let pos = referee.position().clone();
                t.decisions += 1;
                let mask = legal_mask(&view, log, tried);
                let legal: HashSet<usize> = pos.legal_moves().iter().filter_map(|m| encode_move(m, side)).collect();
                let cands = candidate_usis(&view);
                let king = pos.king_square(side).unwrap();
                // ピンの規則: この手番の反則のうち、玉以外の駒の1マスの手・桂の跳び
                let mut pinned: Vec<(Coord, (i8, i8))> = vec![];
                if !view.you_in_check {
                    for usi in tried.iter().filter(|u| cands.contains(*u)) {
                        let Some(ShogiMove::Board { from, to, .. }) = parse_usi(usi) else { continue };
                        if from == king {
                            continue;
                        }
                        let (df, dr) = (to.file - from.file, to.rank - from.rank);
                        let short = df.abs().max(dr.abs()) == 1 || (df.abs() == 1 && dr.abs() == 2);
                        if !short {
                            continue;
                        }
                        // 玉から見た方向（玉の筋の上にいるはず）
                        let (kf, kr) = (from.file - king.file, from.rank - king.rank);
                        let on_line = kf == 0 || kr == 0 || kf.abs() == kr.abs();
                        if !on_line {
                            t.pin_unsound += 1;
                            t.notes.push(format!("seed {seed} game {g}: {usi} の反則なのに玉の筋の上にいない"));
                            continue;
                        }
                        let d = (kf.signum(), kr.signum());
                        if !pinned.iter().any(|&(f, _)| f == from) {
                            pinned.push((from, d));
                        }
                    }
                }
                let mut extra_actions: HashSet<usize> = HashSet::new();
                if !pinned.is_empty() {
                    t.pin_claims += 1;
                    for &(from, d) in &pinned {
                        for usi in &cands {
                            let Some(mv @ ShogiMove::Board { from: f, to, .. }) = parse_usi(usi) else { continue };
                            if f != from || tried.contains(usi) {
                                continue;
                            }
                            let (df, dr) = (to.file - from.file, to.rank - from.rank);
                            let along = (df.signum(), dr.signum()) == d || (df.signum(), dr.signum()) == (-d.0, -d.1);
                            let straight = df == 0 || dr == 0 || df.abs() == dr.abs();
                            if along && straight {
                                continue;
                            }
                            let Some(a) = encode_move(&mv, side) else { continue };
                            if legal.contains(&a) {
                                t.pin_unsound += 1;
                                t.notes.push(format!("seed {seed} game {g}: ピンと推論した {from:?} の手 {usi} が合法"));
                            }
                            if mask[a] {
                                t.pin_extra += 1;
                                extra_actions.insert(a);
                            }
                        }
                    }
                }
                let usi = if rng.random::<f64>() < eps {
                    let acts: Vec<usize> = (0..NUM_ACTIONS).filter(|&a| mask[a]).collect();
                    if acts.is_empty() {
                        break;
                    }
                    decode_usi(acts[rng.random_range(0..acts.len())], side).unwrap()
                } else {
                    match strategies[(side == Color::Gote) as usize].choose(&view, log, tried) {
                        Some(u) => u,
                        None => break,
                    }
                };
                let mv = parse_usi(&usi);
                if let Some(a) = mv.as_ref().and_then(|m| encode_move(m, side)) {
                    if extra_actions.contains(&a) {
                        t.pin_followup += 1;
                    }
                }
                // 通ったマス: 受理される飛び駒の手（2マス以上）の途中のマスが、締めた相手玉の候補に入っているか
                let slide_path: Vec<Coord> = match mv {
                    Some(ShogiMove::Board { from, to, .. }) => {
                        let (df, dr) = (to.file - from.file, to.rank - from.rank);
                        let n = df.abs().max(dr.abs());
                        if n >= 2 && (df == 0 || dr == 0 || df.abs() == dr.abs()) {
                            (1..n).map(|k| Coord { file: from.file + df.signum() * k, rank: from.rank + dr.signum() * k }).collect()
                        } else {
                            vec![]
                        }
                    }
                    _ => vec![],
                };
                let tight_before: Vec<f32> = if slide_path.is_empty() {
                    vec![]
                } else {
                    let o = encode_planes(NUM_PLANES_V6, &view, log, tried);
                    o[E_KING_TIGHT * 81..(E_KING_TIGHT + 1) * 81].to_vec()
                };
                let in_check = view.you_in_check;
                match referee.step(&usi, 0) {
                    StepResult::Ended { .. } => break,
                    StepResult::Accepted => {
                        if !slide_path.is_empty() {
                            t.slides += 1;
                            let hit = slide_path.iter().filter(|&&c| tight_before[square_index(normalize(c, side))] > 0.0).count();
                            t.slide_hits += (hit > 0) as u64;
                            t.slide_squares += hit as u64;
                            if slide_path.iter().any(|&c| pos.king_square(side.other()) == Some(c)) {
                                t.slide_unsound += 1;
                            }
                        }
                    }
                    StepResult::Foul => {
                        t.fouls += 1;
                        if !in_check {
                            t.fouls_free += 1;
                            if let Some(ShogiMove::Board { from, to, .. }) = mv {
                                if from != king {
                                    t.board_fouls_free += 1;
                                    let (df, dr) = (to.file - from.file, to.rank - from.rank);
                                    if df.abs().max(dr.abs()) == 1 || (df.abs() == 1 && dr.abs() == 2) {
                                        t.pin_fouls_first += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            t.games += 1;
            t.lens.push(referee.plies());
        }
        t
    };
    let mut all = T::default();
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads).map(|k| { let run = &run; sc.spawn(move || run(500 + k, games.div_ceil(threads))) }).collect();
        for h in hs {
            let t = h.join().unwrap();
            all.games += t.games;
            all.decisions += t.decisions;
            all.fouls += t.fouls;
            all.fouls_free += t.fouls_free;
            all.pin_claims += t.pin_claims;
            all.pin_unsound += t.pin_unsound;
            all.pin_extra += t.pin_extra;
            all.pin_followup += t.pin_followup;
            all.pin_fouls_first += t.pin_fouls_first;
            all.board_fouls_free += t.board_fouls_free;
            all.slides += t.slides;
            all.slide_hits += t.slide_hits;
            all.slide_squares += t.slide_squares;
            all.slide_unsound += t.slide_unsound;
            all.lens.extend(t.lens);
            all.notes.extend(t.notes);
        }
    });
    let t = &all;
    let g = t.games as f64;
    println!("{} 局・{} 決定点・反則 {}（1局 {:.2}回。うち王手を受けていない手番 {:.2}回）", t.games, t.decisions, t.fouls, t.fouls as f64 / g, t.fouls_free as f64 / g);
    println!(
        "ピン: 王手でない手番の玉以外の盤上の反則 {}（1局 {:.2}回）のうち 1マスの手・桂の跳び {}（1局 {:.2}回）",
        t.board_fouls_free, t.board_fouls_free as f64 / g, t.pin_fouls_first, t.pin_fouls_first as f64 / g
    );
    println!(
        "  ピンと推論できた決定点 {}（1局 {:.2}）/ 推論の誤り {} / 新しく落とせる候補 {}（決定点あたり {:.1}手）/ その候補を実際に選んだ（= 避けられた反則） {}（1局 {:.3}回）",
        t.pin_claims, t.pin_claims as f64 / g, t.pin_unsound, t.pin_extra, t.pin_extra as f64 / t.pin_claims.max(1) as f64, t.pin_followup, t.pin_followup as f64 / g
    );
    println!(
        "通ったマス: 受理された飛び駒の手 {}（1局 {:.1}）のうち、途中のマスが締めた相手玉の候補に入っていた {}（{:.1}%、のべ {} マス）/ 誤り {}",
        t.slides, t.slides as f64 / g, t.slide_hits, 100.0 * t.slide_hits as f64 / t.slides.max(1) as f64, t.slide_squares, t.slide_unsound
    );
    let mut lens = t.lens.clone();
    lens.sort();
    let q = |p: f64| lens[((lens.len() - 1) as f64 * p) as usize];
    println!(
        "手数（受理手）: 中央 {} / 90% 点 {} / 99% 点 {} / 最大 {} / 150手以上 {:.1}% / 180手以上 {:.1}% / 200手 {:.1}%",
        q(0.5), q(0.9), q(0.99), lens[lens.len() - 1],
        100.0 * lens.iter().filter(|&&x| x >= 150).count() as f64 / g,
        100.0 * lens.iter().filter(|&&x| x >= 180).count() as f64 / g,
        100.0 * lens.iter().filter(|&&x| x >= 200).count() as f64 / g
    );
    for n in t.notes.iter().take(10) {
        println!("  例: {n}");
    }
}

/// 履歴が欠けたログ（再起動後の復帰・再接続で取りこぼした通知）を渡したとき、入力とマスクがどうなるか（調査用）。
/// 視界（自駒・持ち駒・手数・王手中か）はサーバーから取り直せるので正しく、ログだけが欠ける
#[test]
#[ignore]
fn 履歴が欠けたログでの入力とマスク() {
    use crate::observation::{Observation, ObservationLog};
    use crate::rl::action::legal_mask;
    use crate::rl::encode::encode_planes;
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
    let games: u64 = env("AUDIT_GAMES", "300").parse().unwrap();
    let players: Vec<String> =
        env("AUDIT_PLAYERS", "rl_scratch5_v2,rl_mmdhandex_v2,rl_scratch5free_v1").split(',').map(String::from).collect();
    let names = ["空のログ（再起動の直後）", "相手の直前の手が欠けた", "自分の直前の手が欠けた", "自分と相手の直前の手が欠けた", "途中から後だけ（再起動後しばらく）"];
    // [欠け方] → [件数, マスクが真の合法手を落とした, マスクに真の合法手が残らない, 確定した占有の誤り, 89番が真の玉を落とした,
    //             締めた候補が真の玉を落とした, 履歴の食い違いの印が立った, 受理手の手数の欠けを検出]
    let mut acc = [[0u64; 9]; 5];
    let mut rng = StdRng::seed_from_u64(3);
    for g in 0..games {
        let mut strategies = [
            crate::strategy::make_seeded(&players[rng.random_range(0..players.len())], g * 2).unwrap(),
            crate::strategy::make_seeded(&players[rng.random_range(0..players.len())], g * 2 + 1).unwrap(),
        ];
        let mut referee = Referee::new();
        loop {
            let side = referee.to_move();
            let view = referee.view(side, [0, 0], g as u32);
            let (log, tried) = (referee.log(side), referee.foul_tried(side));
            let pos = referee.position().clone();
            let events = log.events();
            if view.move_number > 12 {
                let is_my = |e: &Observation| matches!(e, Observation::MyMove { .. });
                let is_opp = |e: &Observation| matches!(e, Observation::OpponentMoved { .. });
                let last_opp = events.iter().rposition(is_opp);
                let last_my = events.iter().rposition(is_my);
                let build = |keep: &dyn Fn(usize) -> bool| {
                    let mut l = ObservationLog::default();
                    for (i, e) in events.iter().enumerate() {
                        if keep(i) {
                            l.record(e.clone());
                        }
                    }
                    l
                };
                // 着手と、その直後の王手宣言をまとめて落とす
                let drop_move = |at: Option<usize>, i: usize| at.is_some_and(|a| i == a || (i == a + 1 && matches!(events.get(i), Some(Observation::Check { .. }))));
                let cut = rng.random_range(0..events.len().max(1));
                let logs = [
                    build(&|_| false),
                    build(&|i| !drop_move(last_opp, i)),
                    build(&|i| !drop_move(last_my, i)),
                    build(&|i| !drop_move(last_opp, i) && !drop_move(last_my, i)),
                    build(&|i| i >= cut),
                ];
                let legal: HashSet<usize> = pos.legal_moves().iter().filter_map(|m| encode_move(m, side)).collect();
                let basic = basic_legal_mask(&view, tried);
                let legal_cand: Vec<usize> = legal.iter().copied().filter(|&a| basic[a]).collect();
                let ok = pos.king_square(side.other()).unwrap();
                for (k, l) in logs.iter().enumerate() {
                    let a = &mut acc[k];
                    a[0] += 1;
                    let mask = legal_mask(&view, l, tried);
                    let lost = legal_cand.iter().filter(|&&x| !mask[x]).count();
                    a[1] += (lost > 0) as u64;
                    a[2] += (!legal_cand.is_empty() && lost == legal_cand.len()) as u64;
                    let facts = deduced_facts(&view, l, tried);
                    a[3] += facts.occupied.iter().any(|&c| pos.piece_at(c).is_none_or(|p| p.color == side)) as u64;
                    let o = encode_planes(NUM_PLANES_V6, &view, l, tried);
                    let at = |p: usize, c: Coord| o[p * 81 + square_index(normalize(c, side))];
                    a[4] += (at(D_KING, ok) != 1.0) as u64;
                    a[5] += (at(E_KING_TIGHT, ok) != 1.0) as u64;
                    a[6] += (o[(P_SCALAR + S_HISTORY_BROKEN) * 81] != 0.0) as u64;
                    a[7] += !moves_contiguous(l, view.move_number) as u64;
                    // 動けないと確定した相手の歩（90番）に、相手の歩がいないマスがある
                    let squares = (1..=9).flat_map(|file| (1..=9).map(move |rank| Coord { file, rank }));
                    a[8] += squares
                        .filter(|&c| at(crate::rl::encode::D_PAWN, c) > 0.0)
                        .any(|c| pos.piece_at(c).is_none_or(|p| p.color == side || p.role != Role::Pawn)) as u64;
                }
            }
            let usi = match strategies[(side == Color::Gote) as usize].choose(&view, log, tried) {
                Some(u) => u,
                None => break,
            };
            if let StepResult::Ended { .. } = referee.step(&usi, 0) {
                break;
            }
        }
    }
    for (k, name) in names.iter().enumerate() {
        let a = acc[k];
        let pc = |x: u64| 100.0 * x as f64 / a[0].max(1) as f64;
        println!(
            "{name}（{} 決定点）: マスクが真の合法手を落とす {:.2}%（合法手が1つも残らない {:.2}%）/ 確定した占有の誤り {:.2}% / \
             89番が真の玉を落とす {:.1}% / 締めた候補が落とす {:.2}% / 動けない歩（90番）の誤り {:.1}% / 食い違いの印 {:.1}% / 手数の欠けを検出 {:.1}%",
            a[0], pc(a[1]), pc(a[2]), pc(a[3]), pc(a[4]), pc(a[5]), pc(a[8]), pc(a[6]), pc(a[7])
        );
    }
}

/// 再起動で履歴を失った側が、その後どのくらい弱くなるか（調査用）。両者とも同じ重みを `rl_policy:` で指し（今の
/// エンコーダとマスクを通る）、片方だけ、ある手数でログを失う（それより前の観測が消え、以後の観測だけが積まれる）。
/// `AUDIT_WEIGHT`（重みのパス）・`AUDIT_GAMES`・`AUDIT_THREADS`・`AUDIT_WIPE`（失う手数の範囲 `lo:hi`。`0:0` で失わない）
#[test]
#[ignore]
fn 履歴を失った側の勝率() {
    use crate::observation::ObservationLog;
    use crate::referee::Referee;
    use crate::selfplay::GameResult;
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
    let games: u64 = env("AUDIT_GAMES", "400").parse().unwrap();
    let threads: u64 = env("AUDIT_THREADS", "4").parse().unwrap();
    let weight = env("AUDIT_WEIGHT", "models/rl_scratch5free_v1.bin");
    let wipe = env("AUDIT_WIPE", "20:60");
    let (lo, hi) = wipe.split_once(':').map(|(a, b)| (a.parse::<u32>().unwrap(), b.parse::<u32>().unwrap())).unwrap();
    let name = format!("rl_policy:{weight}");
    // [勝ち, 負け, 引き分け, 失った後の反則の合計, 失わない側の同じ区間の反則の合計]
    let run = |seed: u64, n: u64| -> [u64; 5] {
        let mut out = [0u64; 5];
        let mut rng = StdRng::seed_from_u64(seed);
        for g in 0..n {
            let mut players = [
                crate::strategy::make_seeded(&name, seed * 10_000 + g * 2).expect("重み"),
                crate::strategy::make_seeded(&name, seed * 10_000 + g * 2 + 1).expect("重み"),
            ];
            let victim = if g % 2 == 0 { Color::Sente } else { Color::Gote };
            let at = if hi == 0 { u32::MAX } else { rng.random_range(lo..=hi) };
            // 失った時点の、被害側のログの長さ（これより前の観測を渡さない）
            let mut cut: Option<usize> = None;
            let mut referee = Referee::new();
            let result = loop {
                let side = referee.to_move();
                let view = referee.view(side, [0, 0], g as u32);
                let (log, tried) = (referee.log(side), referee.foul_tried(side));
                if side == victim && cut.is_none() && view.move_number >= at {
                    cut = Some(log.events().len());
                }
                let usi = if let (true, Some(c)) = (side == victim, cut) {
                    let mut l = ObservationLog::default();
                    for e in &log.events()[c..] {
                        l.record(e.clone());
                    }
                    players[(side == Color::Gote) as usize].choose(&view, &l, tried)
                } else {
                    players[(side == Color::Gote) as usize].choose(&view, log, tried)
                };
                let Some(usi) = usi else { break None };
                match referee.step(&usi, 0) {
                    StepResult::Ended { result, .. } => break Some(result),
                    StepResult::Foul if view.move_number >= at => {
                        out[if side == victim { 3 } else { 4 }] += 1;
                    }
                    _ => {}
                }
            };
            match result {
                Some(GameResult::Win(w)) if w == victim => out[0] += 1,
                Some(GameResult::Win(_)) => out[1] += 1,
                _ => out[2] += 1,
            }
        }
        out
    };
    let mut total = [0u64; 5];
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads).map(|k| { let run = &run; sc.spawn(move || run(900 + k, games.div_ceil(threads))) }).collect();
        for h in hs {
            for (a, b) in total.iter_mut().zip(h.join().unwrap()) {
                *a += b;
            }
        }
    });
    let [w, l, d, fv, fo] = total;
    let n = (w + l) as f64;
    let p = w as f64 / n.max(1.0);
    println!(
        "{weight} / 失う手数 {wipe}: 失った側 {w}勝 {l}敗 {d}分 = {:.1}% ± {:.1}（引き分け除く）/ 失った後の反則 1局 {:.2}回（相手は {:.2}回）",
        100.0 * p,
        196.0 * (p * (1.0 - p) / n.max(1.0)).sqrt(),
        fv as f64 / (w + l + d) as f64,
        fo as f64 / (w + l + d) as f64
    );
}
