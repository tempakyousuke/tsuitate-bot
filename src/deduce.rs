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
use crate::shogi::{Position, promote_role};

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
