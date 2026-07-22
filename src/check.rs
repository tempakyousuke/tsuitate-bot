//! 王手中の回避手選択のための制約推論。
//!
//! 対人対局の分析（records/ 2026-07-08）で、反則の86%が「王手中に攻め駒の位置が
//! 分からず回避候補を手探りで試す」ことによるバーストだった。粒子が枯渇する終盤に
//! 集中するため、粒子に依存しない専用の推論を用意する:
//!
//! - 王手駒の仮説 = 自玉を攻撃しうる（マス, 駒種）の全列挙。自駒の配置は既知なので、
//!   自駒に遮られる仮説は除外できる
//! - この手番で出した反則1つごとに「その手が合法になるはずだった仮説」を減衰させる
//!   （硬い消去にしないのは、反則の原因が別の隠れ駒でもありうるため）
//! - 粒子が生きていれば、粒子中の実際の王手駒に投票させて仮説を重み付けする
//! - 各回避候補の「仮説の下で王手を解消する確率」を返し、評価側が事前確率に使う
//!
//! 両王手・王手駒が紐つきの場合は単一駒仮説では表せないが、反則のたびに減衰が
//! かかるので数手で正しい回避に収束する（従来は同型の反則を繰り返していた）。

use std::collections::HashMap;

use crate::board::Coord;
use crate::model::GameModel;
use crate::observation::ObservationLog;
use crate::protocol::{Color, PlayerView, Role};
use crate::shogi::{Position, ShogiMove, unpromote_role};

/// 王手駒になりうる駒種（玉は王手できない）
const CHECKER_ROLES: [Role; 13] = [
    Role::Pawn,
    Role::Lance,
    Role::Knight,
    Role::Silver,
    Role::Gold,
    Role::Bishop,
    Role::Rook,
    Role::Tokin,
    Role::Promotedlance,
    Role::Promotedknight,
    Role::Promotedsilver,
    Role::Horse,
    Role::Dragon,
];

/// 反則が仮説で説明できない（仮説の下では合法だったはず）ときの減衰係数。
/// 0にしない: 反則の真因が別の隠れ駒（経路封鎖・別の利き）の可能性があるため
const UNEXPLAINED_FOUL_DECAY: f64 = 0.15;

/// 粒子投票の強さ（全粒子が一致した仮説は一様仮説の 1+PARTICLE_VOTE_W 倍）
const PARTICLE_VOTE_W: f64 = 8.0;

struct Hypothesis {
    square: Coord,
    role: Role,
    weight: f64,
}

pub struct CheckSolver {
    /// 自駒＋持ち駒だけを置いたスパース盤面（手番=自分）。仮説の駒を載せて使う
    base: Position,
    my_color: Color,
    hypotheses: Vec<Hypothesis>,
}

impl CheckSolver {
    /// 王手中の view から作る。自玉が見つからない等で推論できなければ None。
    /// particles はソフト救済の重みつき（strategy.rs の評価サンプルと同じ）
    pub fn new(
        view: &PlayerView,
        particles: &[(&Position, f64)],
        fouls_this_turn: &[ShogiMove],
        log: &ObservationLog,
    ) -> Option<CheckSolver> {
        let my_color = view.your_color;
        let mut base = Position::empty(my_color);
        for piece in &view.your_pieces {
            let sq = crate::board::parse_usi_square(&piece.square)?;
            base.set(
                sq,
                Some(crate::shogi::Piece {
                    color: my_color,
                    role: piece.role,
                }),
            );
        }
        for (&role, &count) in &view.your_hand {
            base.set_hand(my_color, role, count as u8);
        }
        base.king_square(my_color)?;

        // 位置が既知の敵駒（自駒が死んだマス = 敵駒がそこへ来た。取り返し済みは除く）を
        // 盤に載せる。回避先がこれらの利きに覆われているかを全仮説共通で判定できる
        // （対人実戦: 5三の既知の成駒が 4二/5二/6二 を覆っているのに順に試して4反則）。
        // **直近8手以内**の新鮮な情報に限定する: 古いマスは駒が動いて陳腐化しやすく、
        // 幻の駒が合法な逃げ場を塞ぐ害が実測で上回った（vs v5 アブレーション 2026-07-10）。
        // 駒種は不明なので粒子の多数決、なければ成駒の最頻・金動き（と金）で近似する
        for sq in known_enemy_squares(log, view.move_number.saturating_sub(8)) {
            if base.piece_at(sq).is_some() {
                continue;
            }
            let role = particle_majority_role(particles, my_color.other(), sq)
                .unwrap_or(Role::Tokin);
            base.set(
                sq,
                Some(crate::shogi::Piece {
                    color: my_color.other(),
                    role,
                }),
            );
            // 近似駒種が王を攻撃してしまう（本物の王手駒と区別できない）配置は
            // 仮説列挙を壊すので載せない
            if base.in_check(my_color) {
                base.set(sq, None);
            }
        }

        let mut solver = CheckSolver {
            base,
            my_color,
            hypotheses: vec![],
        };
        solver.enumerate(&opponent_role_counts(view, log));
        if solver.hypotheses.is_empty() {
            return None;
        }
        solver.vote_by_particles(particles);
        for foul in fouls_this_turn {
            solver.observe_foul(foul);
        }
        Some(solver)
    }

    /// 自玉を攻撃しうる（マス, 駒種）を全列挙する。
    /// 相手が1枚も持ちえない駒種（総数制約）は仮説から外す
    fn enumerate(&mut self, opp_counts: &HashMap<Role, i32>) {
        let opp = self.my_color.other();
        let king = self.base.king_square(self.my_color).expect("new で確認済み");
        for file in 1..=9i8 {
            for rank in 1..=9i8 {
                let sq = Coord { file, rank };
                if self.base.piece_at(sq).is_some() {
                    // 自駒・既知の敵駒のあるマスに（新たな）王手駒はいない
                    // （既知の敵駒が王手していたなら以前から王手宣言されているはず）
                    continue;
                }
                if sq == king {
                    continue;
                }
                for role in CHECKER_ROLES {
                    if opp_counts
                        .get(&unpromote_role(role))
                        .is_none_or(|&n| n <= 0)
                    {
                        continue;
                    }
                    self.base.set(
                        sq,
                        Some(crate::shogi::Piece { color: opp, role }),
                    );
                    let checks = self.base.in_check(self.my_color);
                    self.base.set(sq, None);
                    if checks {
                        self.hypotheses.push(Hypothesis {
                            square: sq,
                            role,
                            weight: 1.0,
                        });
                    }
                }
            }
        }
    }

    /// 粒子中の実際の王手駒に投票させる（粒子が健全なら仮説が鋭くなる）。
    /// ソフト救済の粒子は重みぶんだけ薄く投票する
    fn vote_by_particles(&mut self, particles: &[(&Position, f64)]) {
        let opp = self.my_color.other();
        let mut voters = 0.0f64;
        let mut votes: Vec<f64> = vec![0.0; self.hypotheses.len()];
        for (pos, w) in particles {
            if !pos.in_check(self.my_color) {
                continue; // 王手を反映していない粒子は情報にならない
            }
            voters += w;
            for (i, h) in self.hypotheses.iter().enumerate() {
                if pos.piece_at(h.square)
                    .is_some_and(|p| p.color == opp && p.role == h.role)
                {
                    // 粒子上でその駒が実際に王を攻撃しているかまでは見ない
                    // （enumerate 済みの仮説は自駒配置的に攻撃可能）
                    votes[i] += w;
                }
            }
        }
        if voters <= 0.0 {
            return;
        }
        for (h, &v) in self.hypotheses.iter_mut().zip(&votes) {
            h.weight *= 1.0 + PARTICLE_VOTE_W * (v / voters);
        }
    }

    /// この手番の反則を観測: 仮説の下で合法だったはずの手が反則になった
    /// → その仮説の重みを減衰させる
    fn observe_foul(&mut self, foul: &ShogiMove) {
        for i in 0..self.hypotheses.len() {
            if self.legal_under(i, foul) {
                self.hypotheses[i].weight *= UNEXPLAINED_FOUL_DECAY;
            }
        }
    }

    /// 仮説 i の下で（他の隠れ駒を無視して）mv が合法か = 王手を解消するか
    fn legal_under(&mut self, i: usize, mv: &ShogiMove) -> bool {
        let h = &self.hypotheses[i];
        let piece = crate::shogi::Piece {
            color: self.my_color.other(),
            role: h.role,
        };
        let sq = h.square;
        self.base.set(sq, Some(piece));
        let legal = self.base.is_legal(mv);
        self.base.set(sq, None);
        legal
    }

    /// 候補手が王手を解消する確率（仮説の重み付き割合）
    pub fn resolve_probability(&mut self, mv: &ShogiMove) -> f64 {
        let mut total = 0.0;
        let mut resolved = 0.0;
        for i in 0..self.hypotheses.len() {
            let w = self.hypotheses[i].weight;
            total += w;
            if self.legal_under(i, mv) {
                resolved += w;
            }
        }
        if total <= 0.0 {
            return 0.5; // 全仮説が死んだ（両王手など）: 情報なしに戻す
        }
        resolved / total
    }

    /// mv が「王手駒仮説のマスへ、自玉以外の駒で移動して、その仮説の下で
    /// 王手が解消する」手か = 王手駒を捕獲しに行く手か。
    ///
    /// `resolve_probability`は仮説ごとの重みで平均するため、生存仮説が
    /// 多いと正しい捕獲でも確率が薄まってしまう（王手駒の粒子ビリーフが
    /// 誤っている局面では特に顕著。kakutori.kif参照）。捕獲そのものは
    /// 「当たれば王手駒を排除できる、外れても反則1回ぶんの探索コストで
    /// 済む」性質を持つ数少ない手なので、combine_score側でp_legalの
    /// フロアとして特別扱いする（strategy.rsのchoose参照）
    pub fn captures_checker(&mut self, mv: &ShogiMove) -> bool {
        let ShogiMove::Board { from, to, .. } = *mv else {
            return false;
        };
        if self.base.king_square(self.my_color) == Some(from) {
            return false;
        }
        for i in 0..self.hypotheses.len() {
            if self.hypotheses[i].square == to && self.legal_under(i, mv) {
                return true;
            }
        }
        false
    }

    #[cfg(test)]
    fn hypothesis_count(&self) -> usize {
        self.hypotheses.len()
    }
}

/// 位置が既知の敵駒のマス: 自駒が取られたマス（敵駒がそこへ来た事実）のうち、
/// その後に自分が取り返しておらず、かつ since_move 手目以降の新しいもの
fn known_enemy_squares(log: &ObservationLog, since_move: u32) -> Vec<Coord> {
    let mut map: HashMap<Coord, u32> = HashMap::new();
    for e in log.events() {
        match e {
            crate::observation::Observation::OpponentMoved {
                move_number,
                captured_my_piece_at: Some(sq),
            } => {
                if let Some(c) = crate::board::parse_usi_square(sq) {
                    map.insert(c, *move_number);
                }
            }
            crate::observation::Observation::MyMove {
                usi,
                captured: Some(_),
                ..
            } => {
                if let Some(ShogiMove::Board { to, .. }) = crate::shogi::parse_usi(usi) {
                    map.remove(&to);
                }
            }
            _ => {}
        }
    }
    let mut out: Vec<Coord> = map
        .into_iter()
        .filter(|(_, mn)| *mn >= since_move)
        .map(|(c, _)| c)
        .collect();
    out.sort_by_key(|c| (c.file, c.rank));
    out
}

/// 粒子の加重多数決でそのマスの敵駒の駒種を推定する（過半に満たなければ None）。
/// ソフト救済の粒子は重みぶんだけ薄く数える
fn particle_majority_role(particles: &[(&Position, f64)], opp: Color, sq: Coord) -> Option<Role> {
    if particles.is_empty() {
        return None;
    }
    let total: f64 = particles.iter().map(|(_, w)| w).sum();
    let mut counts: HashMap<Role, f64> = HashMap::new();
    for (pos, w) in particles {
        if let Some(p) = pos.piece_at(sq) {
            if p.color == opp {
                *counts.entry(p.role).or_default() += w;
            }
        }
    }
    let (role, n) = counts
        .into_iter()
        .max_by(|(ra, a), (rb, b)| {
            a.partial_cmp(b)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| role_order(*rb).cmp(&role_order(*ra)))
        })?;
    if n * 2.0 > total {
        Some(role)
    } else {
        None
    }
}

fn role_order(role: Role) -> u8 {
    match role {
        Role::Pawn => 0,
        Role::Lance => 1,
        Role::Knight => 2,
        Role::Silver => 3,
        Role::Gold => 4,
        Role::Bishop => 5,
        Role::Rook => 6,
        Role::King => 7,
        Role::Tokin => 8,
        Role::Promotedlance => 9,
        Role::Promotedknight => 10,
        Role::Promotedsilver => 11,
        Role::Horse => 12,
        Role::Dragon => 13,
    }
}

/// 相手が盤上・持ち駒に持ちうる駒種の枚数（基本駒種で数える）。
/// = 初期枚数 + こちらが取られた枚数 − こちらが取った枚数（自分の持ち駒）
fn opponent_role_counts(view: &PlayerView, log: &ObservationLog) -> HashMap<Role, i32> {
    let mut counts: HashMap<Role, i32> = [
        (Role::Pawn, 9),
        (Role::Lance, 2),
        (Role::Knight, 2),
        (Role::Silver, 2),
        (Role::Gold, 2),
        (Role::Bishop, 1),
        (Role::Rook, 1),
    ]
    .into();
    for (_, role) in GameModel::from_log(view.your_color, log).lost_pieces() {
        *counts.entry(unpromote_role(*role)).or_default() += 1;
    }
    for (&role, &n) in &view.your_hand {
        *counts.entry(unpromote_role(role)).or_default() -= n as i32;
    }
    counts
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::protocol::VisiblePiece;
    use crate::shogi::parse_usi;

    fn view_with(pieces: Vec<(&str, Role)>) -> PlayerView {
        let pieces = pieces
            .into_iter()
            .map(|(sq, role)| VisiblePiece {
                square: sq.into(),
                role,
            })
            .collect();
        let mut view = crate::strategy::tests::minimal_view(pieces, HashMap::new());
        view.you_in_check = true;
        view
    }

    fn mv(usi: &str) -> ShogiMove {
        parse_usi(usi).unwrap()
    }

    #[test]
    fn enumerates_plausible_checkers() {
        // 中央の裸玉: 隣接・桂・遠距離の攻撃仮説が多数列挙される
        let view = view_with(vec![("5e", Role::King)]);
        let solver = CheckSolver::new(&view, &[], &[], &ObservationLog::default()).unwrap();
        assert!(solver.hypothesis_count() > 50);
    }

    #[test]
    fn own_pieces_block_hypotheses() {
        // 5d に自分の歩がいれば「5c の香/飛が王手」仮説は成立しない
        let open = CheckSolver::new(
            &view_with(vec![("5e", Role::King)]),
            &[],
            &[],
            &ObservationLog::default(),
        )
        .unwrap();
        let blocked = CheckSolver::new(
            &view_with(vec![("5e", Role::King), ("5d", Role::Pawn)]),
            &[],
            &[],
            &ObservationLog::default(),
        )
        .unwrap();
        assert!(blocked.hypothesis_count() < open.hypothesis_count());
    }

    #[test]
    fn fouls_shift_probability_toward_consistent_evasions() {
        // 裸玉 5e。縦の移動 5e5d と 5e5f が両方反則
        // → 「5筋の縦利き（飛/香/竜）」仮説が支配的になり、横へ逃げる手の
        //    解消確率が縦に留まる手より高くなる
        let view = view_with(vec![("5e", Role::King)]);
        let fouls = [mv("5e5d"), mv("5e5f")];
        let mut solver = CheckSolver::new(&view, &[], &fouls, &ObservationLog::default()).unwrap();
        let side = solver.resolve_probability(&mv("5e4e"));
        assert!(
            side > 0.7,
            "縦筋仮説の下で横逃げは解消するはず（p={side:.2}）"
        );
    }

    #[test]
    fn captures_checker_detects_only_non_king_capture_on_hypothesis_square() {
        // 5d の後手歩は 5e 玉への王手駒仮説。4e 金で取る手だけを
        // 「王手駒捕獲」として扱い、玉捕獲・別マス移動・打ちは除外する
        let view = view_with(vec![
            ("5e", Role::King),
            ("4e", Role::Gold),
            ("1e", Role::Rook),
        ]);
        let mut solver = CheckSolver::new(&view, &[], &[], &ObservationLog::default()).unwrap();
        assert!(solver.captures_checker(&mv("4e5d")));
        assert!(!solver.captures_checker(&mv("5e5d")));
        assert!(!solver.captures_checker(&mv("1e1b")));
        assert!(!solver.captures_checker(&mv("P*5d")));
    }

    #[test]
    fn known_enemy_piece_rules_out_covered_escapes() {
        // 裸玉 5e。自駒が 5g で取られた → 敵駒（と金近似）が 5g にいると分かっている。
        // 後手のと金は 5g から 5f（後ろ）にも利くので、5e5f はどの王手駒仮説の
        // 下でも非合法（p=0）になるはず。4e への横逃げは生きている
        let mut view = view_with(vec![("5e", Role::King)]);
        view.move_number = 12; // 直近8手以内の捕獲情報として扱われる
        let mut log = ObservationLog::default();
        log.record(crate::observation::Observation::OpponentMoved {
            move_number: 10,
            captured_my_piece_at: Some("5g".into()),
        });
        let mut solver = CheckSolver::new(&view, &[], &[], &log).unwrap();
        let covered = solver.resolve_probability(&mv("5e5f"));
        let side = solver.resolve_probability(&mv("5e4e"));
        assert!(
            covered == 0.0,
            "既知の敵駒に覆われたマスへの逃げは全仮説で非合法のはず（p={covered:.2}）"
        );
        assert!(side > 0.0, "覆われていない逃げは生きているはず（p={side:.2}）");
    }

    #[test]
    fn particles_sharpen_hypotheses() {
        // 粒子が「5a の飛が王手」で一致していれば、5筋から外れる手の確率が上がり、
        // 5筋に留まる手（5d への合駒など）は下がる
        let view = view_with(vec![("5e", Role::King)]);
        let mut truth = Position::empty(Color::Sente);
        truth.set(
            Coord { file: 5, rank: 5 },
            Some(crate::shogi::Piece {
                color: Color::Sente,
                role: Role::King,
            }),
        );
        truth.set(
            Coord { file: 5, rank: 1 },
            Some(crate::shogi::Piece {
                color: Color::Gote,
                role: Role::Rook,
            }),
        );
        let particles: Vec<(&Position, f64)> = vec![(&truth, 1.0); 8];
        let mut solver =
            CheckSolver::new(&view, &particles, &[], &ObservationLog::default()).unwrap();
        let away = solver.resolve_probability(&mv("5e4d"));
        let stay = solver.resolve_probability(&mv("5e5d"));
        assert!(
            away > stay,
            "飛車仮説が支配的なら5筋から外れる手が有利（away={away:.2} stay={stay:.2}）"
        );
    }

    #[test]
    fn soft_particles_vote_with_reduced_weight() {
        // strict粒子（重み1.0）は「5aの飛」、soft粒子（重み0.1）は「2bの角」。
        // 重みが効いていれば飛仮説が支配的になり、5筋に留まる手の解消確率は
        // 重みを逆にしたソルバーより低くなる
        let view = view_with(vec![("5e", Role::King)]);
        let place = |sq: Coord, role: Role, pos: &mut Position| {
            pos.set(
                sq,
                Some(crate::shogi::Piece {
                    color: Color::Gote,
                    role,
                }),
            );
        };
        let mut rook = Position::empty(Color::Sente);
        place(Coord { file: 5, rank: 5 }, Role::King, &mut rook);
        rook.set(
            Coord { file: 5, rank: 5 },
            Some(crate::shogi::Piece {
                color: Color::Sente,
                role: Role::King,
            }),
        );
        place(Coord { file: 5, rank: 1 }, Role::Rook, &mut rook);
        let mut bishop = Position::empty(Color::Sente);
        bishop.set(
            Coord { file: 5, rank: 5 },
            Some(crate::shogi::Piece {
                color: Color::Sente,
                role: Role::King,
            }),
        );
        place(Coord { file: 2, rank: 2 }, Role::Bishop, &mut bishop);

        let rook_heavy: Vec<(&Position, f64)> = vec![(&rook, 1.0), (&bishop, 0.1)];
        let bishop_heavy: Vec<(&Position, f64)> = vec![(&rook, 0.1), (&bishop, 1.0)];
        let mut a =
            CheckSolver::new(&view, &rook_heavy, &[], &ObservationLog::default()).unwrap();
        let mut b =
            CheckSolver::new(&view, &bishop_heavy, &[], &ObservationLog::default()).unwrap();
        let stay_a = a.resolve_probability(&mv("5e5d"));
        let stay_b = b.resolve_probability(&mv("5e5d"));
        assert!(
            stay_a < stay_b,
            "重み付き投票なら飛重視側で5筋残留が不利（a={stay_a:.2} b={stay_b:.2}）"
        );
    }
}
