//! 観測から論理的に確定できる事実の演繹（C-8 の前段）。
//!
//! ユーザーの実践知見（kakunari 65手目 1一角成の確定根拠）を一般化する:
//! 「持ち駒会計で駒種の候補を絞り、各候補の移動経路を実際の移動規則で列挙し、
//! (a) 手数（テンポ）(b) 自玉の王手観測履歴との整合性 の2つで経路を刈り込む」
//! という消去法。負の証拠は「時刻Tに証明された事実」ではなく「自分の王手履歴
//! という常に厳密に分かる情報との矛盾」なので、C-8 設計で懸念された
//! 「負の証拠は時間減衰する」問題を回避できる — 自玉の位置と王手履歴は
//! 履歴ぜんぶを通じて exactly 分かるため、鮮度が腐らない。

use std::collections::{HashMap, HashSet, VecDeque};

use crate::board::{Coord, on_board, orient, steps};
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{Color, Role};
use crate::shogi::{Position, ShogiMove, promote_role};

/// **相手玉の位置候補**を観測履歴から健全に絞る（上の演繹の鏡像: あちらは
/// 「自玉の王手履歴で相手駒の経路を刈る」、こちらは「自分が掛けた王手宣言で
/// 相手玉の居場所を刈る」）。
///
/// 使う事実は全て自分側の完全既知情報だけ:
/// - 初期局面の相手玉のマスは分かっている（そこから出発する）
/// - 自分の手の直後に**王手宣言があった** → 相手玉は自駒の利きうるマスのどれか
/// - 自分の手の直後に**王手宣言が無かった** → 相手玉は自駒の確実な利きのマスには居ない
/// - 相手が指した直後も相手玉は自駒の確実な利きのマスには居ない
///   （自玉を王手に晒す手は指せない）
/// - 相手玉は自駒の居るマスには居ない
/// - 相手の1手で玉は最大1マスしか動かない（動かない場合もある）
///
/// **健全性が命**（真実を絶対に除外しない）なので、利きの数え方を2種類に分ける:
/// - 「確実に利いている」= 未知の駒に遮られ**得ない**利きだけ（隣接1マスと桂の跳び）。
///   除外（＝候補を減らす）にはこちらしか使えない
/// - 「利きうる」= 自駒だけの盤で走り駒のレイを最大まで伸ばしたもの（未知のマスは
///   空とみなす上界）。王手宣言による絞り込み（＝候補を残す条件）にはこちらを使う
///
/// 万一 候補が空になったら演繹のどこかが不健全なので、**制約を捨てて広い集合へ
/// 戻す**（玉位置の質量を潰す事故を避ける。ansatsu 回帰の教訓）。
pub fn opp_king_candidates(my_color: Color, log: &ObservationLog) -> std::collections::BTreeSet<Coord> {
    use std::collections::BTreeSet;
    let opp = my_color.other();
    let all_squares = || -> BTreeSet<Coord> {
        (1..=9)
            .flat_map(|file| (1..=9).map(move |rank| Coord { file, rank }))
            .collect()
    };
    let mut model = crate::model::GameModel::new(my_color);
    let mut cands: BTreeSet<Coord> = BTreeSet::new();
    if let Some(k) = Position::initial().king_square(opp) {
        cands.insert(k);
    } else {
        return all_squares();
    }

    let events = log.events();
    // 直前の自分の手で王手を宣言していたら、そのときの自駒盤を持ち越す
    // （相手の応手が「玉を動かすしかなかった」かの判定に使う）
    let mut checking_board: Option<Position> = None;
    for (i, e) in events.iter().enumerate() {
        model.apply(e);
        match e {
            Observation::MyMove { usi, captured, .. } => {
                let board = my_only_board(&model);
                let declared = matches!(
                    events.get(i + 1),
                    Some(Observation::Check { in_check }) if *in_check == opp
                );
                let before = cands.clone();
                if declared {
                    // 新たに利きが生じ得たマスに限る（王手は必ず自分の手が作る）
                    match crate::shogi::parse_usi(usi) {
                        Some(mv) => {
                            let (to, from) = match mv {
                                crate::shogi::ShogiMove::Board { from, to, .. } => (to, Some(from)),
                                crate::shogi::ShogiMove::Drop { to, .. } => (to, None),
                            };
                            let newly = newly_attacked_squares(
                                &board,
                                my_color,
                                to,
                                from,
                                captured.is_some(),
                            );
                            cands.retain(|k| newly.contains(k));
                        }
                        // USI が読めないときは緩い上界に落とす
                        None => cands.retain(|&k| possibly_attacked(&board, k, my_color)),
                    }
                } else {
                    cands.retain(|&k| !surely_attacked(&board, k, my_color));
                }
                cands.retain(|&k| board.piece_at(k).is_none());
                if cands.is_empty() {
                    // 演繹が不健全（か観測が欠けている）。制約を捨てて安全側へ
                    cands = before;
                    cands.retain(|&k| board.piece_at(k).is_none());
                    if cands.is_empty() {
                        return all_squares();
                    }
                }
                checking_board = declared.then_some(board);
            }
            Observation::OpponentMoved {
                captured_my_piece_at,
                ..
            } => {
                let captured_sq = captured_my_piece_at
                    .as_deref()
                    .and_then(crate::board::parse_usi_square);
                // 玉は最大1マス動く（動かないこともある）。ただし直前に王手を
                // 掛けていて、その王手駒が**隣接**していて（＝合駒で防げない）
                // **取られてもいない**なら、玉は動くしかなかった＝居残りは除外できる
                let mut next: BTreeSet<Coord> = BTreeSet::new();
                for &k in &cands {
                    let forced_to_move = checking_board.as_ref().is_some_and(|b| {
                        b.pieces().any(|(from, pc)| {
                            pc.color == my_color
                                && unblockable(from, k)
                                && b.attacks(from, k)
                                && Some(from) != captured_sq
                        })
                    });
                    if !forced_to_move {
                        next.insert(k);
                    }
                    for df in -1..=1i8 {
                        for dr in -1..=1i8 {
                            if df == 0 && dr == 0 {
                                continue;
                            }
                            let n = Coord { file: k.file + df, rank: k.rank + dr };
                            if on_board(n) {
                                next.insert(n);
                            }
                        }
                    }
                }
                let board = my_only_board(&model);
                // 相手は自玉を王手に晒す手を指せない
                next.retain(|&k| !surely_attacked(&board, k, my_color));
                next.retain(|&k| board.piece_at(k).is_none());
                cands = if next.is_empty() { all_squares() } else { next };
                checking_board = None;
            }
            _ => {}
        }
    }
    // **相手の歩が居ると確定しているマスに玉は居ない**（2026-08-03）。
    // 玉候補は `strategy::project_taint_kings` が taint 粒子の玉を移設する先に
    // なるが、その移設は移設先の駒と**入れ替え**るので、確定した歩のマスを
    // 候補に残すと歩が弾き出されて「自分の観測だけで否定できる粒子」が生まれる
    // （ユーザー指摘の 4七歩）。ここで落としておけば全ての利用箇所が守られる
    let locked = immobile_opponent_pawns(my_color, log);
    if !locked.is_empty() {
        let pruned: std::collections::BTreeSet<Coord> =
            cands.difference(&locked).copied().collect();
        // 空になったら演繹のどこかが不健全なので制約を捨てる（既存の安全弁と同じ）
        if !pruned.is_empty() {
            cands = pruned;
        }
    }
    cands
}

/// 自駒だけを置いた盤（未知のマスは空。走り駒のレイはここで最大まで伸びる）
fn my_only_board(model: &crate::model::GameModel) -> Position {
    let me = model.my_color();
    let mut pos = Position::empty(me);
    for p in model.my_pieces() {
        if let Some(sq) = crate::board::parse_usi_square(&p.square) {
            pos.set(sq, Some(crate::shogi::Piece { color: me, role: p.role }));
        }
    }
    pos
}

/// from から to への攻撃が未知の駒に遮られ**得ない**か（隣接1マス or 桂の跳び）
fn unblockable(from: Coord, to: Coord) -> bool {
    let df = (from.file - to.file).abs();
    let dr = (from.rank - to.rank).abs();
    (df <= 1 && dr <= 1) || (df == 1 && dr == 2)
}

/// sq が me の駒に**確実に**利かれているか（遮られ得ない利きだけを数える）
fn surely_attacked(board: &Position, sq: Coord, me: Color) -> bool {
    board
        .pieces()
        .any(|(from, pc)| pc.color == me && unblockable(from, sq) && board.attacks(from, sq))
}

/// sq が me の駒に利き**うる**か（未知のマスを空とみなした上界）
fn possibly_attacked(board: &Position, sq: Coord, me: Color) -> bool {
    board
        .pieces()
        .any(|(from, pc)| pc.color == me && board.attacks(from, sq))
}

/// from から to への直線上（両端を除く）に x があるか。走り駒の「線が開いたか」判定用
fn between(from: Coord, to: Coord, x: Coord) -> bool {
    let (df, dr) = (to.file - from.file, to.rank - from.rank);
    let step = |v: i8| v.signum();
    // 直線（縦横斜め）でなければ通り道は無い
    if !(df == 0 || dr == 0 || df.abs() == dr.abs()) {
        return false;
    }
    let (sf, sr) = (step(df), step(dr));
    let mut c = Coord { file: from.file + sf, rank: from.rank + sr };
    while c != to {
        if c == x {
            return true;
        }
        c = Coord { file: c.file + sf, rank: c.rank + sr };
    }
    false
}

/// 自分の着手によって**新たに**自駒の利きが生じ得るマスの上界。
/// 相手は自分から自玉を王手に晒す手を指せないので、新しい王手は必ず自分の手が作る。
/// 作り方は3通りしかない:
/// - (A) 着手した駒が着地点から利かせる
/// - (B) 出て行ったマス（from）を通る自分の走り駒の線が開く（開き王手）
/// - (C) 取った駒が居たマス（to）を通る自分の走り駒の線が開く
///
/// 集合の**差分**（after で利く ∧ before で利かない）では計算しない: 上界どうしの
/// 差は上界にならず（before 側が過大に主張すると真の新規王手マスを落とす）健全性が壊れる。
/// A/B/C を構成的に足し上げることで上界を保つ
fn newly_attacked_squares(
    after: &Position,
    me: Color,
    to: Coord,
    from: Option<Coord>,
    captured: bool,
) -> std::collections::BTreeSet<Coord> {
    let mut out = std::collections::BTreeSet::new();
    for file in 1..=9i8 {
        for rank in 1..=9i8 {
            let s = Coord { file, rank };
            if s == to {
                continue; // 着地点には自駒が居る
            }
            // (A) 着手した駒の利き
            if after.attacks(to, s) {
                out.insert(s);
                continue;
            }
            // (B)(C) 開いた線
            let opened = after.pieces().any(|(p, pc)| {
                pc.color == me
                    && p != to
                    && after.attacks(p, s)
                    && (from.is_some_and(|f| between(p, s, f)) || (captured && between(p, s, to)))
            });
            if opened {
                out.insert(s);
            }
        }
    }
    out
}

#[cfg(test)]
mod king_tests {
    use super::*;
    use crate::board::make_usi_square;
    use crate::scenario_core::{load_scenario, replay, scenarios_dir, side_idx};

    fn game() -> crate::kifu::Kifu {
        let path = scenarios_dir().join("archive/king-deduction.kif");
        load_scenario(&path.to_string_lossy(), Some(0), None, None)
            .expect("archive/king-deduction.kif")
            .kifu
    }

    /// **健全性**: 真の玉位置は必ず候補集合に含まれる（全 ply・両視点）。
    /// ここが破れると玉位置の質量を誤って潰すので、絶対に落としてはいけない
    #[test]
    fn 敵玉候補は真実を落とさない() {
        let kifu = game();
        for ply in 0..=kifu.plies.len() {
            let rep = replay(&kifu, ply);
            for me in [Color::Sente, Color::Gote] {
                let cands = opp_king_candidates(me, &rep.logs[side_idx(me)]);
                let truth = rep.pos.king_square(me.other()).expect("玉");
                assert!(
                    cands.contains(&truth),
                    "ply={ply} {me:?}視点: 真実 {} が候補{}件に無い",
                    make_usi_square(truth),
                    cands.len()
                );
            }
        }
    }

    /// **鋭さ**: 王手の直後は候補が実用的な数まで絞れる。
    /// 49手目 ▲3c4c（と金の王手）→ 50手目の逃げで7マス
    #[test]
    fn 王手の直後は敵玉候補が絞れる() {
        let kifu = game();
        let rep = replay(&kifu, 50);
        let cands = opp_king_candidates(Color::Sente, &rep.logs[side_idx(Color::Sente)]);
        // BTreeSet<Coord> は (file, rank) 昇順
        let got: Vec<String> = cands.iter().map(|c| make_usi_square(*c)).collect();
        assert_eq!(got, ["3d", "4e", "5a", "5e", "6a", "6b", "6c"], "50手目の候補");
        // 79手目まで進めると1マスまで確定する
        let rep = replay(&kifu, 79);
        let cands = opp_king_candidates(Color::Sente, &rep.logs[side_idx(Color::Sente)]);
        assert_eq!(cands.len(), 1, "79手目は1マスに確定するはず: {cands:?}");
    }

    /// **健全性**: 動けないと断言した相手の歩は、必ず真実でもそこに居る。
    /// 2局分の全局面・両者視点で検証する（真実を1つでも落としたら演繹が壊れている）
    #[test]
    fn 不動の歩の演繹は真実を落とさない() {
        for name in ["archive/king-deduction.kif", "archive/quest_20260731.kif"] {
            let path = scenarios_dir().join(name);
            let kifu = load_scenario(&path.to_string_lossy(), Some(0), None, None)
                .unwrap()
                .kifu;
            for ply in 0..=kifu.plies.len() {
                let rep = replay(&kifu, ply);
                for me in [Color::Sente, Color::Gote] {
                    let opp = me.other();
                    for sq in immobile_opponent_pawns(me, &rep.logs[side_idx(me)]) {
                        let truth = rep.pos.piece_at(sq);
                        assert!(
                            truth.is_some_and(|p| p.color == opp && p.role == Role::Pawn),
                            "{name} ply={ply} {me:?}視点: {} に相手の歩が居ると断言したが真実は {truth:?}",
                            make_usi_square(sq)
                        );
                    }
                }
            }
        }
    }

    /// **鋭さ**: ユーザー指摘の局面（quest_20260731 の38手目、後手番）で
    /// 4七が「動けない先手の歩」として確定する
    #[test]
    fn 塞がれた歩は不動と確定できる() {
        let path = scenarios_dir().join("archive/quest_20260731.kif");
        let kifu = load_scenario(&path.to_string_lossy(), Some(0), None, None)
            .unwrap()
            .kifu;
        let rep = replay(&kifu, 37);
        let locked = immobile_opponent_pawns(Color::Gote, &rep.logs[side_idx(Color::Gote)]);
        let got: Vec<String> = locked.iter().map(|c| make_usi_square(*c)).collect();
        assert!(
            got.contains(&"4g".to_string()),
            "38手目に 4g（先手の歩）が確定するはず: {got:?}"
        );
    }
}

/// **動けないことが確定している相手の歩**のマス（＝そこに相手の歩がいると
/// 断言できる集合）。
///
/// ユーザー指摘（2026-08-03、quest_20260731 の38手目 `S*4g`）の一般化:
/// 「先手の4七歩は前進先の4六に自分の歩がいるので動けない。動けば必ず取りが
/// 発生して観測されるはずで、観測が無い以上そこに居続けている」。
///
/// 使う事実は**全て自分側の完全既知情報**だけなので健全（真実を絶対に落とさない）:
/// - 初期局面の相手の歩の位置は分かっている（各筋の3段目/7段目）
/// - 歩の動きは前方1マスだけ
/// - 自分の駒の配置は常に厳密に分かる（自分の手と「取られたマス」の観測）
/// - 相手が指せるのは自分の手番のときだけ
///
/// 判定: **相手の手番が来るたびに前進先が自駒で塞がっていた**なら、その歩は
/// 一度も動く機会が無かった＝初期マスにいる。途中で前進先が空いた瞬間があれば
/// （そこで動いたかは分からないので）以後その筋は諦める。自分がその歩を
/// 取った観測があればもちろん除外する。
///
/// 用途は2つ: 粒子フィルタが**この事実に反する粒子を作らないようにする**
/// （変異救済の移設元から除く）ことと、打ちマスの反則を確実に避けること。
pub fn immobile_opponent_pawns(
    my_color: Color,
    log: &ObservationLog,
) -> std::collections::BTreeSet<Coord> {
    use std::collections::BTreeSet;
    let opp = my_color.other();
    // 相手の歩の初期段と、その前進先の段（相手から見た前方 = 自分に近づく向き）
    let (pawn_rank, front_rank) = match opp {
        Color::Sente => (7, 6),
        Color::Gote => (3, 4),
    };
    // 前進の向き（相手の歩が進む段の増分）と成り込み段
    let step: i8 = if opp == Color::Sente { -1 } else { 1 };
    let promo_zone = |r: i8| if opp == Color::Sente { r <= 3 } else { r >= 7 };
    // 自分の盤面を再構成する（自駒だけ。GameModel と同じ材料）
    let init = Position::initial();
    let mut mine: HashMap<Coord, Role> = init
        .pieces()
        .filter(|(_, p)| p.color == my_color)
        .map(|(sq, p)| (sq, p.role))
        .collect();
    // 筋ごとの「その筋の相手の歩がいられる段」の**上界集合**（1件に絞れたら確定）。
    // 歩は前進しかしないので段だけで表せる。give_up[f] は追跡を諦めた筋
    let mut possible: HashMap<i8, BTreeSet<i8>> =
        (1..=9).map(|f| (f, BTreeSet::from([pawn_rank]))).collect();
    let mut give_up: HashSet<i8> = HashSet::new();
    let _ = front_rank;

    // 自駒が居るマスに相手の歩は居ない（自駒の配置は常に厳密に分かる）
    let prune_by_mine = |possible: &mut HashMap<i8, BTreeSet<i8>>, mine: &HashMap<Coord, Role>| {
        for (&f, ranks) in possible.iter_mut() {
            ranks.retain(|&r| !mine.contains_key(&Coord { file: f, rank: r }));
        }
    };

    for e in log.events() {
        match e {
            Observation::MyMove { usi, captured, .. } => match crate::shogi::parse_usi(usi) {
                Some(ShogiMove::Board { from, to, promote }) => {
                    // その筋の歩がいられる段で駒を取ったなら、取ったのがその歩
                    // だった可能性がある（`captured` は成る前の駒種に潰されて
                    // 届くので と金 と歩を区別できない）→ その筋は諦める
                    if captured.is_some()
                        && possible.get(&to.file).is_some_and(|s| s.contains(&to.rank))
                    {
                        give_up.insert(to.file);
                    }
                    if let Some(role) = mine.remove(&from) {
                        let r = if promote {
                            promote_role(role).unwrap_or(role)
                        } else {
                            role
                        };
                        mine.insert(to, r);
                    }
                    prune_by_mine(&mut possible, &mine);
                }
                Some(ShogiMove::Drop { role, to }) => {
                    // 打てた = そのマスは空だった
                    mine.insert(to, role);
                    prune_by_mine(&mut possible, &mine);
                }
                None => {}
            },
            Observation::OpponentMoved {
                captured_my_piece_at,
                ..
            } => {
                let captured_sq = captured_my_piece_at
                    .as_deref()
                    .and_then(crate::board::parse_usi_square);
                // この手番で歩が1つ前進できた可能性を足す（上界なので広げる側）。
                // 前が自駒で塞がっていれば、前進 = その自駒を取る手なので
                // 「取られたマス」の観測と一致するときだけ可能
                for (&f, ranks) in possible.iter_mut() {
                    let mut next = ranks.clone();
                    for &r in ranks.iter() {
                        let front = Coord { file: f, rank: r + step };
                        if !on_board(front) {
                            continue;
                        }
                        let blocked = mine.contains_key(&front);
                        if !blocked || captured_sq == Some(front) {
                            next.insert(front.rank);
                        }
                    }
                    *ranks = next;
                }
                // 成り込みうる段に届いたら、以後は と金 として盤上を自由に動ける
                // ＝ 段だけでは追えないので、その筋は諦める
                for (&f, ranks) in possible.iter() {
                    if ranks.iter().any(|&r| promo_zone(r)) {
                        give_up.insert(f);
                    }
                }
                if let Some(c) = captured_sq {
                    mine.remove(&c);
                }
                prune_by_mine(&mut possible, &mine);
            }
            // 反則は盤面を動かさない。王手宣言も自駒の配置には影響しない
            Observation::MyFoul { .. }
            | Observation::OpponentFoul { .. }
            | Observation::Check { .. } => {}
        }
    }
    possible
        .into_iter()
        .filter(|(f, ranks)| !give_up.contains(f) && ranks.len() == 1)
        .map(|(f, ranks)| Coord {
            file: f,
            rank: *ranks.iter().next().expect("1件"),
        })
        .collect::<BTreeSet<_>>()
}

/// 相手の各着手（OpponentMoved）の直前時点での「自玉の位置」と、
/// その着手の直後に自玉への王手が観測されたか。自玉の位置は自分の手でしか
/// 動かないので、直前=直後で同じ（この着手中は動いていない）
#[derive(Debug, Clone, Copy)]
pub struct OppMoveContext {
    /// 相手の着手インデックス（0始まり）
    pub index: usize,
    /// この時点での自玉の位置（分からなければ None。通常は必ず分かる）
    pub my_king: Option<Coord>,
    /// この着手の直後に自玉への王手が観測されたか
    pub check_declared: bool,
}

/// 観測ログから相手の着手ごとの文脈（自玉位置・王手観測）を復元する。
/// 自分の手は自玉位置の更新にだけ使う（GameModel の再構成ロジックを踏襲）
pub fn opponent_move_contexts(my_color: Color, log: &ObservationLog) -> Vec<OppMoveContext> {
    let mut my_king: Option<Coord> = crate::shogi::Position::initial()
        .king_square(my_color);
    let mut out = vec![];
    let events = log.events();
    let mut i = 0;
    let mut opp_idx = 0usize;
    while i < events.len() {
        match &events[i] {
            Observation::MyMove { usi, .. } => {
                if let Some(mv) = crate::shogi::parse_usi(usi) {
                    if let crate::shogi::ShogiMove::Board { from, to, .. } = mv {
                        if Some(from) == my_king {
                            my_king = Some(to);
                        }
                    }
                }
            }
            Observation::OpponentMoved { .. } => {
                let check_declared = matches!(
                    events.get(i + 1),
                    Some(Observation::Check { in_check }) if *in_check == my_color
                );
                out.push(OppMoveContext {
                    index: opp_idx,
                    my_king,
                    check_declared,
                });
                opp_idx += 1;
                if check_declared {
                    i += 1; // 対になっている Check イベントも consume
                }
            }
            _ => {}
        }
        i += 1;
    }
    out
}

/// 駒 (role, color) が from に着地した瞬間、target に利くか
/// （ジャンプ系は遮蔽なし。スライド系は空盤前提 = 他の駒による遮蔽は考慮しない
/// 近似。桂馬・金型など近接駒の判定には影響しない）
pub fn piece_attacks_square(role: Role, color: Color, from: Coord, target: Coord) -> bool {
    let mut pos = Position::empty(color);
    pos.set(
        from,
        Some(crate::shogi::Piece { color, role }),
    );
    pos.attacks(from, target)
}

/// ある候補駒種が square に着地した場合、それが自玉への説明のつかない王手を
/// 作ってしまうか（= その着地が経路として不可能かどうか）を判定する。
/// contexts はその着地が起こり得る相手着手インデックスの範囲（絞り込めるなら
/// 絞ってよい。分からなければ全体を渡してよい — 少なくとも1つでも
/// 「着地時に王手だが観測なし」のインデックスがあれば矛盾とみなす保守的な版ではなく、
/// 「その正方形にその駒がいる限り常に王手になる」という時刻非依存の幾何的事実を
/// 使うので、対象窓に含まれるどの時点でも自玉が同じ位置なら判定は不変。
/// window 内の自玉位置がブレる場合は各時点で判定する
pub fn route_square_refuted_by_check_history(
    role: Role,
    color: Color,
    square: Coord,
    contexts: &[OppMoveContext],
    window: std::ops::Range<usize>,
) -> bool {
    for ctx in contexts.iter().filter(|c| window.contains(&c.index)) {
        let Some(king) = ctx.my_king else { continue };
        if piece_attacks_square(role, color, square, king) && !ctx.check_declared {
            return true; // この時点でこの駒がここにいれば王手のはずだが観測なし
        }
    }
    false
}

/// 空盤上での駒 role の到達可能マス（1手ぶん、ジャンプ・スライドとも空盤前提）。
/// テンポ収支の admissible な最短路探索に使う下請け
pub fn empty_board_targets(role: Role, color: Color, from: Coord) -> Vec<Coord> {
    let mut out = vec![];
    for &delta in steps(role) {
        let (df, dr) = orient(delta, color);
        let to = Coord {
            file: from.file + df,
            rank: from.rank + dr,
        };
        if on_board(to) {
            out.push(to);
        }
    }
    for &delta in crate::board::rays(role) {
        let (df, dr) = orient(delta, color);
        let mut c = Coord {
            file: from.file + df,
            rank: from.rank + dr,
        };
        while on_board(c) {
            out.push(c);
            c = Coord {
                file: c.file + df,
                rank: c.rank + dr,
            };
        }
    }
    out
}

/// 成れるゾーン（自分から見て奥3段）か
fn in_promotion_zone(color: Color, rank: i8) -> bool {
    match color {
        Color::Sente => rank <= 3,
        Color::Gote => rank >= 7,
    }
}

type DistanceMap = HashMap<(Coord, bool), u32>;

thread_local! {
    static EMPTY_BOARD_DISTANCE_CACHE: std::cell::RefCell<
        HashMap<(Role, Color, Coord), DistanceMap>,
    > = std::cell::RefCell::new(HashMap::new());
}

/// from を起点にした空盤上の最短手数を、到達しうる全マス(かつ成/不成の両状態)に
/// ついて1回のBFSでまとめて求める。target を1つずつ聞く min_moves_empty_board を
/// 何度も呼ぶより、候補が多い場面(守り駒の列挙など)で大幅に速い。
/// 結果は (role, color, from) だけで決まる純粋な値（盤面非依存）なので
/// スレッドローカルにメモ化する。guide_boost_factor 等、同じ from から
/// 何度も呼ばれるホットパスでの再BFSを避けるための最適化（挙動は変えない）
pub fn all_distances_empty_board(
    base_role: Role,
    color: Color,
    from: Coord,
) -> HashMap<(Coord, bool), u32> {
    let key = (base_role, color, from);
    if let Some(cached) = EMPTY_BOARD_DISTANCE_CACHE.with(|c| c.borrow().get(&key).cloned()) {
        return cached;
    }
    let dist_map = compute_distances_empty_board(base_role, color, from);
    EMPTY_BOARD_DISTANCE_CACHE.with(|c| c.borrow_mut().insert(key, dist_map.clone()));
    dist_map
}

pub fn distance_empty_board(
    base_role: Role,
    color: Color,
    from: Coord,
    target: Coord,
    promoted: bool,
) -> Option<u32> {
    let key = (base_role, color, from);
    EMPTY_BOARD_DISTANCE_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        let dist_map = cache
            .entry(key)
            .or_insert_with(|| compute_distances_empty_board(base_role, color, from));
        dist_map.get(&(target, promoted)).copied()
    })
}

fn compute_distances_empty_board(
    base_role: Role,
    color: Color,
    from: Coord,
) -> DistanceMap {
    let promoted_role = promote_role(base_role);
    let mut visited: HashSet<(Coord, bool)> = HashSet::new();
    let mut queue: VecDeque<((Coord, bool), u32)> = VecDeque::new();
    let mut dist_map = HashMap::new();
    queue.push_back(((from, false), 0));
    visited.insert((from, false));
    dist_map.insert((from, false), 0);
    while let Some(((sq, promoted), dist)) = queue.pop_front() {
        let role_now = if promoted {
            promoted_role.unwrap_or(base_role)
        } else {
            base_role
        };
        for next_sq in empty_board_targets(role_now, color, sq) {
            if !promoted && visited.insert((next_sq, false)) {
                dist_map.insert((next_sq, false), dist + 1);
                queue.push_back(((next_sq, false), dist + 1));
            }
            let can_promote_here = !promoted
                && promoted_role.is_some()
                && (in_promotion_zone(color, sq.rank) || in_promotion_zone(color, next_sq.rank));
            if can_promote_here && visited.insert((next_sq, true)) {
                dist_map.insert((next_sq, true), dist + 1);
                queue.push_back(((next_sq, true), dist + 1));
            }
            if promoted && visited.insert((next_sq, true)) {
                dist_map.insert((next_sq, true), dist + 1);
                queue.push_back(((next_sq, true), dist + 1));
            }
        }
    }
    dist_map
}

/// 空盤上での駒の最短到達手数（admissible な下限）。成りの状態遷移を含む
/// 拡張状態 (マス, 成っているか) の BFS — 生駒の移動グラフで進み、
/// 出発マスか到達マスのどちらかが成れるゾーンなら、その手で成る選択肢も生まれる
/// （将棋の成りルールどおり。成り自体は手数を消費しない）。
/// `want_promoted` が true なら「成った状態で to に到達する」最短手数を返す。
/// 金・王など成れない駒に true を渡した場合は None を返す。
/// 打ち歩詰め・二歩などの合法性、他の駒による遮蔽は考慮しない
/// （この関数は「これより少ない手数はあり得ない」という下限の計算専用）
pub fn min_moves_empty_board(
    base_role: Role,
    color: Color,
    from: Coord,
    to: Coord,
    want_promoted: bool,
) -> Option<u32> {
    let promoted_role = promote_role(base_role);
    if want_promoted && promoted_role.is_none() {
        // 成れない駒に「成った状態」を要求するのは無意味な問い
        return None;
    }
    if want_promoted {
        distance_empty_board(base_role, color, from, to, true)
    } else {
        // 成っていても不成でも構わない場合は両方のうち短い方
        let a = distance_empty_board(base_role, color, from, to, false);
        let b = distance_empty_board(base_role, color, from, to, true);
        a.into_iter().chain(b).min()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knight_landing_square_attacks_adjacent_diagonal_ranks() {
        // 先手桂馬が 2четыре(2d)相当の位置に着地すると 1二・3二 に利く
        // （kakunari 65手目 1一角成 確定根拠の核心部分）
        let landing = Coord { file: 2, rank: 4 };
        let king_a = Coord { file: 1, rank: 2 };
        let king_b = Coord { file: 3, rank: 2 };
        assert!(piece_attacks_square(Role::Knight, Color::Sente, landing, king_a));
        assert!(piece_attacks_square(Role::Knight, Color::Sente, landing, king_b));
        // 遠いマスには利かない
        let far = Coord { file: 5, rank: 5 };
        assert!(!piece_attacks_square(Role::Knight, Color::Sente, landing, far));
    }

    #[test]
    fn narikei_at_1b_does_not_attack_3b() {
        // 成桂（金型移動）は1マスしか動けないので 1二 から 3二 へは利かない
        let sq = Coord { file: 1, rank: 2 };
        let target = Coord { file: 3, rank: 2 };
        assert!(!piece_attacks_square(
            Role::Promotedknight,
            Color::Sente,
            sq,
            target
        ));
    }

    #[test]
    fn opponent_move_contexts_tracks_king_and_check() {
        let mut log = ObservationLog::default();
        log.record(Observation::OpponentMoved {
            move_number: 1,
            captured_my_piece_at: None,
        });
        log.record(Observation::Check {
            in_check: Color::Sente,
        });
        log.record(Observation::OpponentMoved {
            move_number: 3,
            captured_my_piece_at: None,
        });
        let ctxs = opponent_move_contexts(Color::Sente, &log);
        assert_eq!(ctxs.len(), 2);
        assert!(ctxs[0].check_declared);
        assert!(!ctxs[1].check_declared);
        // 初期玉位置（先手 5i）が両方の文脈で保持されている
        assert_eq!(ctxs[0].my_king, Some(Coord { file: 5, rank: 9 }));
        assert_eq!(ctxs[1].my_king, Some(Coord { file: 5, rank: 9 }));
    }

    #[test]
    fn knight_home_to_1a_promoted_takes_five_moves() {
        // 先手桂馬 2一(2i) → 1一(1a、成桂として) の最短手数。
        // 2一→1七or3七(1jump)→2五(2jump)→1三(3jump,成れるゾーン突入で成り可)→
        // 1二(4)→1一(5) の5手が下限のはず
        let home = Coord { file: 2, rank: 9 };
        let target = Coord { file: 1, rank: 1 };
        let dist = min_moves_empty_board(Role::Knight, Color::Sente, home, target, true);
        assert_eq!(dist, Some(5), "dist={dist:?}");
    }

    #[test]
    fn bishop_home_to_1a_promoted_is_short() {
        // 先手角 8八(8h) → 1一(1a、馬として) は対角線一直線で1手（成り込み）
        let home = Coord { file: 8, rank: 8 };
        let target = Coord { file: 1, rank: 1 };
        let dist = min_moves_empty_board(Role::Bishop, Color::Sente, home, target, true);
        assert_eq!(dist, Some(1), "dist={dist:?}");
    }

    #[test]
    fn unpromotable_role_ignores_want_promoted_mismatch() {
        // 成れない駒（金）に成った状態を要求すると None
        let dist = min_moves_empty_board(
            Role::Gold,
            Color::Sente,
            Coord { file: 5, rank: 9 },
            Coord { file: 5, rank: 8 },
            true,
        );
        assert_eq!(dist, None);
    }
}
