//! 信念ネット（NN段階②）の特徴量。**マスごと**に「そこに相手の駒がいるか」を
//! 予測するための入力を、観測履歴（`ObservationLog`）だけから作る。
//!
//! 設計の要点:
//! - **観測だけで作れること**が絶対条件。ここに真の局面の情報が混じると
//!   学習は良くなるのに実戦で使えない（value ネットとは逆の制約）
//! - 出力は「マスごとの独立確率」なので**盤面の直接合成には使わない**
//!   （駒数・駒種多重集合・二歩・行き所・テンポの同時制約が壊れる。
//!   docs/improvement-plan-2026-07-25.md 項目3の設計方針）。用途は
//!   粒子の**提案分布の prior**（リプレイ／若返りのガイド）
//! - 1決定点あたり1回だけ評価すれば足りる（粒子に依存しない）ので、
//!   81マス × 小さな MLP のコストはホットパスに載る
//!
//! 学習データ書き出し（`bin/export_belief_data`）と推論（`belief_nn.rs`）が
//! **この1ファイルを共有する**（opp_move_features.rs と同じ規約）。

use crate::board::{Coord, Delta, on_board, orient, parse_usi_square, steps};
use crate::model::GameModel;
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{Color, Role};
use crate::shogi::{Piece, Position, ShogiMove, parse_usi, promote_role, unpromote_role};

pub const BELIEF_FEATURES: usize = 35;

pub const BELIEF_FEATURE_NAMES: [&str; BELIEF_FEATURES] = [
    // --- 局面全体の量（マス間で共通だが、pointwise ネットが条件付けに使う） ---
    "prior_occ",   // 相手の盤上駒数（下限）÷ 未知マス数 = 基準占有率
    "opp_board_n", // 相手の盤上駒数の下限 ÷ 20（= 20 − 自分が取った枚数）
    "opp_taken_n", // 相手が自駒を取った枚数 ÷ 10（相手の持ち駒の上限。
    // 打ち戻しは観測できないので実際の持ち駒は不明 = 盤上駒数の上振れ幅でもある）
    "opp_moves_n", // 相手の着手数 ÷ 60（進行度）
    "in_check",    // 自玉が王手されている
    // --- マスの静的な属性 ---
    "depth_opp",    // 相手の底辺からの距離 ÷ 8（0 = 相手の1段目）
    "file_center",  // 中央度（|file-5| を 0..1 に）
    "init_opp",     // 初期配置で相手の駒があったマス
    "init_pawn",    // 初期配置の駒種（相手側）。「未観測の駒は初期配置のまま」の
    "init_lance",   // 事前分布が角・香で繰り返し欠けた（knight_bait / home_lance の
    "init_knight",  // 教訓）ため駒種まで持たせる
    "init_silver",
    "init_gold",
    "init_bishop",
    "init_rook",
    "init_king",
    // --- 観測から来る証拠 ---
    "opp_seen",       // 相手駒がそこに居たと確定したことがある
    "opp_seen_rec",   // その新しさ（0.85^経過相手手数）
    "empty_seen",     // 空だと確定したことがある（自分の手が通過・着地した）
    "empty_seen_rec", // その新しさ
    "untouched",      // 自分の手も取り合いも一度も触れていない
    "drop_foul",      // そのマスへの打ちが反則になった回数（歩以外・非王手中）
    "drop_foul_rec",  // その新しさ。**打ちマス反則の直接証拠**
    "to_foul",        // そのマスへの盤上移動が反則になった回数
    "path_foul",      // 反則になった走り手の経路上だった回数（経路封鎖の証拠）
    "king_foul",      // 自玉のそのマスへの移動が反則になった回数（= 相手の利き）
    "my_attack",      // 自駒が利かせている数 ÷ 3
    "my_adjacent",    // 隣接8マスの自駒数 ÷ 4
    "dist_my_king",   // 自玉からの距離 ÷ 8
    "neighbor_ev",    // 隣接8マスの opp_seen_rec の平均
    "dist_opp_last",  // 直近で自駒が取られたマスからの距離 ÷ 8（無ければ1）
    // --- 自玉との幾何（王手駒ビリーフ較正の入口） ---
    "king_line",  // 自駒に遮られず自玉へ利きうる走りのマス
    "king_step",  // 自玉に隣接 or 桂の跳び元
    "king_cand",  // deduce::opp_king_candidates に入っている（相手玉候補）
    "king_cand_frac", // 1 ÷ 候補数（候補集合の鋭さ）
];

/// 証拠の減衰率（相手の1手あたり）
const DECAY: f64 = 0.85;

#[derive(Clone, Copy, Default)]
struct SquareEvidence {
    /// 相手駒がそこに居たと確定した最後の時刻（相手手数。None = 未観測）
    opp_seen: Option<u32>,
    /// 空だと確定した最後の時刻
    empty_seen: Option<u32>,
    drop_foul: u16,
    drop_foul_at: Option<u32>,
    to_foul: u16,
    path_foul: u16,
    king_foul: u16,
    /// 自分の手・取り合いが一度でも触れたか
    touched: bool,
}

/// 観測履歴から作る「マスごとの証拠」＋局面全体の既知量。
/// `ObservationLog` だけで作れる（実戦でも学習でも同じものが得られる）。
pub struct BeliefContext {
    my_color: Color,
    /// 自駒だけを載せた盤（利き数の計算用。手番は自分）
    my_board: Position,
    /// 自駒の現配置（占有判定用）
    mine: [bool; 81],
    my_king: Option<Coord>,
    in_check: bool,
    opp_moves: u32,
    /// 相手の盤上駒数の**下限**（20 − 自分が取った枚数）。相手が自駒を取って
    /// 打ち直した分だけ増えうるが、打ちは観測できないので上限は不明
    opp_board_min: u32,
    /// 相手が自分から取った枚数（相手の持ち駒の上限。打ち戻した分は減る）
    opp_taken: u32,
    /// 直近で自駒が取られたマス
    opp_last_capture: Option<Coord>,
    ev: [SquareEvidence; 81],
    /// 相手玉候補（deduce）
    king_cands: [bool; 81],
    king_cand_count: u32,
    /// 自玉へ利きうる走りのマス
    king_line: [bool; 81],
    /// 自玉に隣接 or 桂の跳び元
    king_step: [bool; 81],
}

pub fn sq_index(c: Coord) -> usize {
    (c.file as usize - 1) * 9 + (c.rank as usize - 1)
}

pub fn all_squares() -> impl Iterator<Item = Coord> {
    (1..=9i8).flat_map(|file| (1..=9i8).map(move |rank| Coord { file, rank }))
}

/// 走り手の経路（from と to の間の中間マス）。直線（縦・横・斜め）に
/// 並んでいなければ空（桂の跳びには中間マスが無い）
fn ray_path(from: Coord, to: Coord) -> Vec<Coord> {
    let dfile = to.file - from.file;
    let drank = to.rank - from.rank;
    let n = dfile.abs().max(drank.abs());
    if n <= 1 || !(dfile == 0 || drank == 0 || dfile.abs() == drank.abs()) {
        return vec![];
    }
    let (df, dr) = (dfile.signum(), drank.signum());
    (1..n)
        .map(|i| Coord {
            file: from.file + df * i,
            rank: from.rank + dr * i,
        })
        .collect()
}

impl BeliefContext {
    pub fn from_log(my_color: Color, log: &ObservationLog) -> BeliefContext {
        let mut ev = [SquareEvidence::default(); 81];
        let mut opp_moves = 0u32;
        let mut my_captures = 0u32;
        let mut opp_captures = 0u32;
        let mut opp_last_capture = None;
        let mut in_check = false;
        // 自玉は自分の手でしか動かないので履歴中の各時点で厳密に分かる
        // （反則の理由分類に使う: 玉の移動反則 = 移動先が相手の利き）
        let mut king_now = Position::initial().king_square(my_color);

        // 自駒の現配置は GameModel の再構成をそのまま使う（ここでは経路・
        // 去ったマスの「空だった」証拠を追加で採るので独自に1周する）
        let model = GameModel::from_log(my_color, log);

        for e in log.events() {
            match e {
                Observation::MyMove { usi, captured, .. } => {
                    let Some(mv) = parse_usi(usi) else { continue };
                    match mv {
                        ShogiMove::Board { from, to, .. } => {
                            let f = &mut ev[sq_index(from)];
                            f.touched = true;
                            // 自駒が去ったマスは「この時点で空」
                            f.empty_seen = Some(opp_moves);
                            for p in ray_path(from, to) {
                                let s = &mut ev[sq_index(p)];
                                s.touched = true;
                                s.empty_seen = Some(opp_moves);
                            }
                            let t = &mut ev[sq_index(to)];
                            t.touched = true;
                            match captured {
                                Some(_) => t.opp_seen = Some(opp_moves),
                                None => t.empty_seen = Some(opp_moves),
                            }
                        }
                        ShogiMove::Drop { to, .. } => {
                            let t = &mut ev[sq_index(to)];
                            t.touched = true;
                            t.empty_seen = Some(opp_moves);
                        }
                    }
                    if captured.is_some() {
                        my_captures += 1;
                    }
                    if let ShogiMove::Board { from, to, .. } = mv
                        && Some(from) == king_now
                    {
                        king_now = Some(to);
                    }
                    in_check = false;
                }
                Observation::MyFoul { usi, .. } => {
                    let Some(mv) = parse_usi(usi) else { continue };
                    match mv {
                        ShogiMove::Drop { to, role } => {
                            let t = &mut ev[sq_index(to)];
                            t.touched = true;
                            // 歩打ちは打ち歩詰めという別の反則理由がありうる（自分からは
                            // 判定不能）ので占有の証拠にしない。王手中も「合駒のはずが
                            // 別ラインだった」という説明があるので採らない
                            // （estimator.rs の guide.occupies と同じ規約）
                            if role != Role::Pawn && !in_check {
                                t.drop_foul += 1;
                                t.drop_foul_at = Some(opp_moves);
                            }
                        }
                        ShogiMove::Board { from, to, .. } => {
                            let is_king = Some(from) == king_now;
                            let t = &mut ev[sq_index(to)];
                            t.touched = true;
                            if is_king {
                                // 自玉の移動は経路遮蔽が起きない（1マス）ので、
                                // 反則の理由は「移動先が相手の利きにある」で一意
                                t.king_foul += 1;
                            } else {
                                t.to_foul += 1;
                            }
                            for p in ray_path(from, to) {
                                ev[sq_index(p)].path_foul += 1;
                            }
                        }
                    }
                }
                Observation::OpponentMoved {
                    captured_my_piece_at,
                    ..
                } => {
                    opp_moves += 1;
                    if let Some(sq) = captured_my_piece_at.as_ref().and_then(|s| parse_usi_square(s)) {
                        let t = &mut ev[sq_index(sq)];
                        t.touched = true;
                        t.opp_seen = Some(opp_moves);
                        opp_last_capture = Some(sq);
                        opp_captures += 1;
                    }
                    in_check = false;
                }
                Observation::Check { in_check: side } => {
                    if *side == my_color {
                        in_check = true;
                    }
                }
                Observation::OpponentFoul { .. } => {}
            }
        }

        let mut my_board = Position::empty(my_color);
        let mut mine = [false; 81];
        let mut my_king = None;
        for p in model.my_pieces() {
            let Some(sq) = parse_usi_square(&p.square) else {
                continue;
            };
            mine[sq_index(sq)] = true;
            my_board.set(
                sq,
                Some(Piece {
                    color: my_color,
                    role: p.role,
                }),
            );
            if p.role == Role::King {
                my_king = Some(sq);
            }
        }

        let mut king_cands = [false; 81];
        let cands = crate::deduce::opp_king_candidates(my_color, log);
        for c in &cands {
            king_cands[sq_index(*c)] = true;
        }

        let (king_line, king_step) = king_geometry(&my_board, my_color, my_king);

        BeliefContext {
            my_color,
            my_board,
            mine,
            my_king,
            in_check,
            opp_moves,
            opp_board_min: 20u32.saturating_sub(my_captures),
            opp_taken: opp_captures,
            opp_last_capture,
            ev,
            king_cands,
            king_cand_count: cands.len() as u32,
            king_line,
            king_step,
        }
    }

    pub fn my_color(&self) -> Color {
        self.my_color
    }

    /// 自駒が居るマス（占有確率は自明に0なので学習・推論の対象外）
    pub fn is_mine(&self, sq: Coord) -> bool {
        self.mine[sq_index(sq)]
    }

    /// 相手駒が居ないと分かっている（自駒が居る）マスを除いた未知マス数
    fn unknown_squares(&self) -> u32 {
        81 - self.mine.iter().filter(|&&m| m).count() as u32
    }

    pub fn features(&self, sq: Coord) -> [f64; BELIEF_FEATURES] {
        let i = sq_index(sq);
        let e = &self.ev[i];
        let opp = self.my_color.other();
        let unknown = self.unknown_squares().max(1);

        let depth_opp = match opp {
            Color::Gote => (sq.rank - 1) as f64,
            Color::Sente => (9 - sq.rank) as f64,
        } / 8.0;

        let init = Position::initial();
        let init_piece = init.piece_at(sq).filter(|p| p.color == opp).map(|p| p.role);
        let is_init = |r: Role| f64::from(init_piece == Some(r));

        let rec = |t: Option<u32>| match t {
            Some(t) => DECAY.powi((self.opp_moves - t) as i32),
            None => 0.0,
        };

        let neighbor_ev = {
            let mut sum = 0.0;
            for (df, dr) in KING_DELTAS {
                let n = Coord {
                    file: sq.file + df,
                    rank: sq.rank + dr,
                };
                if on_board(n) {
                    sum += rec(self.ev[sq_index(n)].opp_seen);
                }
            }
            sum / 8.0
        };
        let my_adjacent = {
            let mut n = 0.0f64;
            for (df, dr) in KING_DELTAS {
                let c = Coord {
                    file: sq.file + df,
                    rank: sq.rank + dr,
                };
                if on_board(c) && self.mine[sq_index(c)] {
                    n += 1.0;
                }
            }
            (n / 4.0).min(2.0)
        };

        let dist = |a: Coord, b: Coord| (a.file - b.file).abs().max((a.rank - b.rank).abs()) as f64;

        [
            self.opp_board_min as f64 / unknown as f64,
            self.opp_board_min as f64 / 20.0,
            (self.opp_taken as f64 / 10.0).min(2.0),
            (self.opp_moves as f64 / 60.0).min(3.0),
            f64::from(self.in_check),
            depth_opp,
            (5 - (sq.file - 5).abs()) as f64 / 4.0,
            f64::from(init_piece.is_some()),
            is_init(Role::Pawn),
            is_init(Role::Lance),
            is_init(Role::Knight),
            is_init(Role::Silver),
            is_init(Role::Gold),
            is_init(Role::Bishop),
            is_init(Role::Rook),
            is_init(Role::King),
            f64::from(e.opp_seen.is_some()),
            rec(e.opp_seen),
            f64::from(e.empty_seen.is_some()),
            rec(e.empty_seen),
            f64::from(!e.touched),
            (e.drop_foul as f64 / 3.0).min(2.0),
            rec(e.drop_foul_at),
            (e.to_foul as f64 / 3.0).min(2.0),
            (e.path_foul as f64 / 3.0).min(2.0),
            (e.king_foul as f64 / 3.0).min(2.0),
            (self.my_board.attack_count(sq, self.my_color) as f64 / 3.0).min(2.0),
            my_adjacent,
            self.my_king.map_or(1.0, |k| dist(sq, k) / 8.0),
            neighbor_ev,
            self.opp_last_capture.map_or(1.0, |c| dist(sq, c) / 8.0),
            f64::from(self.king_line[i]),
            f64::from(self.king_step[i]),
            f64::from(self.king_cands[i]),
            if self.king_cand_count > 0 {
                1.0 / self.king_cand_count as f64
            } else {
                0.0
            },
        ]
    }
}

const KING_DELTAS: [Delta; 8] = [
    (0, -1),
    (1, -1),
    (-1, -1),
    (1, 0),
    (-1, 0),
    (0, 1),
    (1, 1),
    (-1, 1),
];

/// 自玉へ利きうるマスの幾何（未知マスは空とみなす上界）。
/// `king_line` = 走り駒なら自玉を攻撃しうるマス（自駒に遮られない範囲）、
/// `king_step` = 1マス駒・桂で自玉を攻撃しうるマス。
/// 王手駒ビリーフ（kakutori の根本原因の残り半分）の入口として、
/// 「その仮説位置に相手駒が居るか」をネットが学べるようにするための特徴量。
fn king_geometry(
    my_board: &Position,
    my_color: Color,
    my_king: Option<Coord>,
) -> ([bool; 81], [bool; 81]) {
    let mut line = [false; 81];
    let mut step = [false; 81];
    let Some(k) = my_king else {
        return (line, step);
    };
    // 走り: 玉から8方向に伸ばし、自駒に当たったら止める
    for (df, dr) in KING_DELTAS {
        let mut c = k;
        loop {
            c = Coord {
                file: c.file + df,
                rank: c.rank + dr,
            };
            if !on_board(c) {
                break;
            }
            if my_board.piece_at(c).is_some() {
                break;
            }
            line[sq_index(c)] = true;
        }
    }
    // 1マス駒: 玉の隣接マス（そこから玉を取れる駒種がある）
    for (df, dr) in KING_DELTAS {
        let c = Coord {
            file: k.file + df,
            rank: k.rank + dr,
        };
        if on_board(c) && my_board.piece_at(c).is_none() {
            step[sq_index(c)] = true;
        }
    }
    // 桂: 相手の桂が玉を取れる位置（相手視点の跳び先が玉になるマス）
    let opp = my_color.other();
    for d in steps(Role::Knight) {
        let (df, dr) = orient(*d, opp);
        let c = Coord {
            file: k.file - df,
            rank: k.rank - dr,
        };
        if on_board(c) && my_board.piece_at(c).is_none() {
            step[sq_index(c)] = true;
        }
    }
    (line, step)
}

/// 真の局面から取り出す教師ラベル。
/// `occ` = 相手の駒がそこに居るか、`role` = 居るならその駒種（成りは戻さない）
pub fn label(truth: &Position, my_color: Color, sq: Coord) -> (bool, Option<Role>) {
    match truth.piece_at(sq) {
        Some(p) if p.color == my_color.other() => (true, Some(p.role)),
        _ => (false, None),
    }
}

/// 駒種ラベルのクラス番号（0 = 空、1..=8 = 成る前の駒種、成りフラグは別列）。
/// 学習側で使う駒種ヘッドのための正規化
pub fn role_class(role: Option<Role>) -> (u8, u8) {
    let Some(role) = role else { return (0, 0) };
    let promoted = u8::from(promote_role(unpromote_role(role)) == Some(role));
    let base = match unpromote_role(role) {
        Role::Pawn => 1,
        Role::Lance => 2,
        Role::Knight => 3,
        Role::Silver => 4,
        Role::Gold => 5,
        Role::Bishop => 6,
        Role::Rook => 7,
        Role::King => 8,
        _ => 0,
    };
    (base, promoted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::make_usi_square;

    fn feat(ctx: &BeliefContext, sq: &str, name: &str) -> f64 {
        let i = BELIEF_FEATURE_NAMES.iter().position(|&n| n == name).unwrap();
        ctx.features(parse_usi_square(sq).unwrap())[i]
    }

    #[test]
    fn names_match_dimension() {
        assert_eq!(BELIEF_FEATURE_NAMES.len(), BELIEF_FEATURES);
        let ctx = BeliefContext::from_log(Color::Sente, &ObservationLog::default());
        assert_eq!(ctx.features(parse_usi_square("5e").unwrap()).len(), BELIEF_FEATURES);
    }

    #[test]
    fn initial_position_marks_opponent_home_squares() {
        let ctx = BeliefContext::from_log(Color::Sente, &ObservationLog::default());
        // 後手の飛車は 8二、角は 2二、玉は 5一
        assert_eq!(feat(&ctx, "8b", "init_rook"), 1.0);
        assert_eq!(feat(&ctx, "2b", "init_bishop"), 1.0);
        assert_eq!(feat(&ctx, "5a", "init_king"), 1.0);
        assert_eq!(feat(&ctx, "5e", "init_opp"), 0.0);
        // 自駒のマスは対象外
        assert!(ctx.is_mine(parse_usi_square("7g").unwrap()));
        assert!(!ctx.is_mine(parse_usi_square("5e").unwrap()));
    }

    #[test]
    fn accepted_slide_proves_path_empty() {
        let mut log = ObservationLog::default();
        // 7六歩（1マス）ではなく、角道を開けてから角を走らせる想定で
        // 8八→3三 の経路（7七・6六・5五・4四）が空だったことを刻む
        log.record(Observation::MyMove {
            move_number: 1,
            usi: "7g7f".into(),
            captured: None,
        });
        log.record(Observation::MyMove {
            move_number: 3,
            usi: "8h3c".into(),
            captured: Some(Role::Pawn),
        });
        let ctx = BeliefContext::from_log(Color::Sente, &log);
        for sq in ["7g", "6f", "5e", "4d"] {
            assert_eq!(feat(&ctx, sq, "empty_seen"), 1.0, "{sq} は空だと確定するはず");
        }
        // 着地マスは駒を取ったので「相手駒が居た」証拠
        assert_eq!(feat(&ctx, "3c", "opp_seen"), 1.0);
        assert_eq!(feat(&ctx, "3c", "empty_seen"), 0.0);
    }

    #[test]
    fn drop_foul_marks_occupancy_but_pawn_does_not() {
        let mut log = ObservationLog::default();
        log.record(Observation::MyFoul {
            move_number: 1,
            usi: "S*5e".into(),
        });
        log.record(Observation::MyFoul {
            move_number: 1,
            usi: "P*4e".into(),
        });
        let ctx = BeliefContext::from_log(Color::Sente, &log);
        assert!(feat(&ctx, "5e", "drop_foul") > 0.0);
        assert_eq!(feat(&ctx, "4e", "drop_foul"), 0.0, "歩打ちは打ち歩詰めの説明がある");
    }

    #[test]
    fn king_move_foul_is_separated_from_other_fouls() {
        let mut log = ObservationLog::default();
        log.record(Observation::MyFoul {
            move_number: 1,
            usi: "5i5h".into(), // 自玉
        });
        log.record(Observation::MyFoul {
            move_number: 1,
            usi: "2h5h".into(), // 飛車（経路 4八・3八）
        });
        let ctx = BeliefContext::from_log(Color::Sente, &log);
        assert!(feat(&ctx, "5h", "king_foul") > 0.0);
        assert!(feat(&ctx, "5h", "to_foul") > 0.0);
        assert!(feat(&ctx, "4h", "path_foul") > 0.0);
        assert!(feat(&ctx, "3h", "path_foul") > 0.0);
    }

    #[test]
    fn opponent_capture_marks_square_as_seen() {
        let mut log = ObservationLog::default();
        log.record(Observation::OpponentMoved {
            move_number: 2,
            captured_my_piece_at: Some("7f".into()),
        });
        let ctx = BeliefContext::from_log(Color::Sente, &log);
        assert_eq!(feat(&ctx, "7f", "opp_seen"), 1.0);
        assert!(feat(&ctx, "7f", "opp_seen_rec") > 0.9, "直近なので減衰していない");
        assert_eq!(feat(&ctx, "7f", "dist_opp_last"), 0.0);
    }

    #[test]
    fn king_geometry_stops_at_own_pieces() {
        let ctx = BeliefContext::from_log(Color::Sente, &ObservationLog::default());
        // 初期局面の先手玉は 5九。8段目は 2八の飛・8八の角だけなので、
        // 真上（5八）と斜め（4八・6八）までは伸びる
        assert_eq!(feat(&ctx, "5h", "king_line"), 1.0);
        assert_eq!(feat(&ctx, "4h", "king_line"), 1.0);
        // 7段目は歩が並んでいるので、そこで止まる（自駒のマス自体は入らない）
        assert_eq!(feat(&ctx, "5g", "king_line"), 0.0);
        assert_eq!(feat(&ctx, "3g", "king_line"), 0.0);
        assert_eq!(feat(&ctx, "5f", "king_line"), 0.0);
        assert_eq!(feat(&ctx, "2f", "king_line"), 0.0);
        // 隣接の空きマスは king_step、自駒（4九の金）はそうならない
        assert_eq!(feat(&ctx, "5h", "king_step"), 1.0);
        assert_eq!(feat(&ctx, "4i", "king_step"), 0.0);
        // 後手の桂が 5九の玉を取れるのは 4七・6七（どちらも先手の歩なので除外）
        assert_eq!(feat(&ctx, "4g", "king_step"), 0.0);
    }

    #[test]
    fn label_and_role_class_round_trip() {
        let pos = Position::initial();
        let (occ, role) = label(&pos, Color::Sente, parse_usi_square("8b").unwrap());
        assert!(occ);
        assert_eq!(role, Some(Role::Rook));
        assert_eq!(role_class(role), (7, 0));
        assert_eq!(role_class(Some(Role::Dragon)), (7, 1));
        assert_eq!(role_class(Some(Role::Tokin)), (1, 1));
        assert_eq!(role_class(None), (0, 0));
        let (occ, _) = label(&pos, Color::Sente, parse_usi_square("7g").unwrap());
        assert!(!occ, "自駒のマスは相手の占有ではない");
        let _ = make_usi_square(Coord { file: 1, rank: 1 });
    }
}
