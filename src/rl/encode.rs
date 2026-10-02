//! 観測 → 入力テンソル（docs/rl-deepnash-design.md の「2. 観測エンコード」）。
//!
//! 入力は `Strategy::choose` と同じ `(PlayerView, ObservationLog, foul_tried)` だけ。
//! 学習中の環境も arena・本番の方策 Strategy もこの関数を通るので、学習時と推論時で
//! 特徴量が食い違わない。observation.rs にない情報（相手駒の位置など）は受け取れない。
//!
//! 出力は `[NUM_PLANES, 9, 9]` を平らにした f32 列（プレーン優先、マス番号は
//! `action::square_index`）。盤は手番側を先手向きに正規化する。

use std::collections::HashSet;

use crate::board::{Coord, parse_usi_square};
use crate::model::GameModel;
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{Color, PlayerView, Role};
use crate::rl::action::{normalize, square_index};
use crate::selfplay::{MAX_FOULS, MAX_PLIES};
use crate::shogi::{HAND_ROLES, Position, ShogiMove, hand_index, parse_usi, unpromote_role};

/// 自駒の駒種（`Role` の宣言順）
const ROLES: [Role; 14] = [
    Role::Pawn,
    Role::Lance,
    Role::Knight,
    Role::Silver,
    Role::Gold,
    Role::Bishop,
    Role::Rook,
    Role::King,
    Role::Tokin,
    Role::Promotedlance,
    Role::Promotedknight,
    Role::Promotedsilver,
    Role::Horse,
    Role::Dragon,
];
/// 自分の着手履歴として持つ直近の手数
pub const HISTORY_MOVES: usize = 8;
/// 鮮度の減衰（1手 = 1 move_number あたり）
const DECAY: f32 = 0.95;
/// 各駒種の総数（両者合計。`HAND_ROLES` 順）。相手の保有数の算出と正規化に使う
const ROLE_TOTAL: [u32; 7] = [18, 4, 4, 4, 4, 2, 2];

// プレーンの配置
const P_OWN: usize = 0; // 14
const P_HIST: usize = P_OWN + 14; // 2 × HISTORY_MOVES（移動元, 移動先）
const P_CAP: usize = P_HIST + 2 * HISTORY_MOVES; // 取ったマスの鮮度
const P_CAP_ROLE: usize = P_CAP + 1; // 7: 取った駒種ごとの鮮度
const P_LOST: usize = P_CAP_ROLE + 7; // 取られたマスの鮮度
const P_LOST_COUNT: usize = P_LOST + 1; // 取られた回数
// 反則は「今の手番」と「過去の手番」を同じ形で持つ（1組 = 移動元・移動先・成りの移動先・
// 打ち先×駒種7）。打ちの駒種と成りを分けるのは、同じマスでも反則の原因（打ち歩詰め・
// 成れない位置など）が違い、隠れた盤面への証拠が別物になるため
const FOUL_GROUP: usize = 3 + 7;
const P_FOUL: usize = P_LOST_COUNT + 1; // 今の手番の反則（値は 1）
const P_PAST_FOUL: usize = P_FOUL + FOUL_GROUP; // 過去の手番の反則（値は鮮度）
const F_FROM: usize = 0;
const F_TO: usize = 1;
const F_PROMO_TO: usize = 2;
const F_DROP: usize = 3;
const P_CHECKED_KING: usize = P_PAST_FOUL + FOUL_GROUP; // 王手されたときの自玉の位置（鮮度）
const P_GAVE_CHECK: usize = P_CHECKED_KING + 1; // 王手をかけた手の移動先（鮮度）
const P_UNMOVED: usize = P_GAVE_CHECK + 1; // 初期配置から一度も動いていない自駒
const P_HAND: usize = P_UNMOVED + 1; // 7: 自分の持ち駒（全面に放送）
const P_OPP_HOLD: usize = P_HAND + 7; // 7: 相手の駒種別保有数（盤上＋持ち駒。自分側から確定）
const P_SCALAR: usize = P_OPP_HOLD + 7; // 以下のスカラー
const S_MY_FOULS: usize = 0;
const S_OPP_FOULS: usize = 1;
const S_MY_LAST_FOUL: usize = 2; // あと1回で反則負け
const S_OPP_LAST_FOUL: usize = 3;
const S_FOULS_THIS_TURN: usize = 4;
const S_IN_CHECK: usize = 5;
const S_GAVE_CHECK_LAST: usize = 6; // 直前の自分の手で王手をかけた
const S_MOVE_NUMBER: usize = 7;
/// 観測ログから再構成した自駒・持ち駒が視界と食い違う（再起動で履歴を失った等）。
/// 履歴由来の特徴（鮮度・未着手駒・王手履歴）が信用できないことをネットへ知らせる
const S_HISTORY_BROKEN: usize = 8;
const NUM_SCALARS: usize = 9;

pub const NUM_PLANES: usize = P_SCALAR + NUM_SCALARS;
pub const OBS_LEN: usize = NUM_PLANES * 81;

/// 反則1件を1組のプレーンへ書く（`base` は `P_FOUL` か `P_PAST_FOUL`）
fn put_foul(p: &mut Planes, base: usize, usi: &str, v: f32) {
    match parse_usi(usi) {
        Some(ShogiMove::Board { from, to, promote }) => {
            p.max(base + F_FROM, from, v);
            p.max(base + F_TO, to, v);
            if promote {
                p.max(base + F_PROMO_TO, to, v);
            }
        }
        Some(ShogiMove::Drop { role, to }) => {
            if let Some(i) = hand_index(role) {
                p.max(base + F_DROP + i, to, v);
            }
        }
        None => {}
    }
}

/// 観測ログから再構成した自分側（盤上・持ち駒）が視界と一致するか
fn history_consistent(model: &GameModel, view: &PlayerView) -> bool {
    if !model.consistent() {
        return false;
    }
    let mut a: Vec<(String, Role)> = view
        .your_pieces
        .iter()
        .map(|v| (v.square.clone(), v.role))
        .collect();
    let mut b: Vec<(String, Role)> = model
        .my_pieces()
        .into_iter()
        .map(|v| (v.square, v.role))
        .collect();
    a.sort();
    b.sort();
    let hand_a: Vec<u32> = HAND_ROLES
        .iter()
        .map(|r| view.your_hand.get(r).copied().unwrap_or(0))
        .collect();
    let model_hand = model.my_hand();
    let hand_b: Vec<u32> = HAND_ROLES
        .iter()
        .map(|r| model_hand.get(r).copied().unwrap_or(0))
        .collect();
    a == b && hand_a == hand_b
}

struct Planes<'a> {
    buf: &'a mut [f32],
    color: Color,
}

impl Planes<'_> {
    fn at(&mut self, plane: usize, c: Coord) -> &mut f32 {
        &mut self.buf[plane * 81 + square_index(normalize(c, self.color))]
    }
    fn set(&mut self, plane: usize, c: Coord, v: f32) {
        *self.at(plane, c) = v;
    }
    fn max(&mut self, plane: usize, c: Coord, v: f32) {
        let x = self.at(plane, c);
        *x = x.max(v);
    }
    fn add(&mut self, plane: usize, c: Coord, v: f32) {
        *self.at(plane, c) += v;
    }
    fn fill(&mut self, plane: usize, v: f32) {
        self.buf[plane * 81..(plane + 1) * 81].fill(v);
    }
}

fn move_squares(usi: &str) -> Option<(Option<Coord>, Coord)> {
    match parse_usi(usi)? {
        ShogiMove::Board { from, to, .. } => Some((Some(from), to)),
        ShogiMove::Drop { to, .. } => Some((None, to)),
    }
}

/// 観測をテンソルへ書く。`out` は長さ `OBS_LEN`
pub fn encode_into(
    view: &PlayerView,
    log: &ObservationLog,
    foul_tried: &HashSet<String>,
    out: &mut [f32],
) {
    assert_eq!(out.len(), OBS_LEN);
    out.fill(0.0);
    let color = view.your_color;
    let now = view.move_number;
    let fresh = |mn: u32| DECAY.powi(now.saturating_sub(mn) as i32);
    let mut p = Planes { buf: out, color };

    // 自駒
    for piece in &view.your_pieces {
        if let (Some(c), Some(r)) = (
            parse_usi_square(&piece.square),
            ROLES.iter().position(|&r| r == piece.role),
        ) {
            p.set(P_OWN + r, c, 1.0);
        }
    }

    // 履歴を先頭から辿る。自駒の配置（王手されたときの自玉の位置）は GameModel で追う
    let mut model = GameModel::new(color);
    let mut my_moves: Vec<&str> = vec![];
    let mut last_mn = 1u32;
    let mut gave_check_last = false;
    let initial: HashSet<Coord> = Position::initial()
        .pieces_of(color)
        .iter()
        .filter_map(|v| parse_usi_square(&v.square))
        .collect();
    let mut unmoved = initial.clone();
    for event in log.events() {
        match event {
            Observation::MyMove {
                move_number,
                usi,
                captured,
            } => {
                last_mn = *move_number;
                my_moves.push(usi);
                gave_check_last = false;
                if let Some((from, to)) = move_squares(usi) {
                    if let Some(from) = from {
                        unmoved.remove(&from);
                    }
                    if let Some(role) = captured {
                        p.max(P_CAP, to, fresh(*move_number));
                        if let Some(i) = hand_index(*role) {
                            p.max(P_CAP_ROLE + i, to, fresh(*move_number));
                        }
                    }
                }
            }
            Observation::OpponentMoved {
                move_number,
                captured_my_piece_at,
            } => {
                last_mn = *move_number;
                if let Some(c) = captured_my_piece_at.as_deref().and_then(parse_usi_square) {
                    p.max(P_LOST, c, fresh(*move_number));
                    p.add(P_LOST_COUNT, c, 1.0 / 3.0);
                    unmoved.remove(&c);
                }
            }
            Observation::MyFoul { move_number, usi } => {
                // 今の手番の反則は下で foul_tried から載せる（反則は手番を変えないので
                // move_number が今と同じものが今の手番）
                if *move_number != now {
                    put_foul(&mut p, P_PAST_FOUL, usi, fresh(*move_number));
                }
            }
            Observation::Check { in_check } => {
                if *in_check == color {
                    // 直前の相手の手で王手された。自玉の位置は model が持つ（この後で apply）
                    let king = model
                        .my_pieces()
                        .into_iter()
                        .find(|v| v.role == Role::King)
                        .and_then(|v| parse_usi_square(&v.square));
                    if let Some(k) = king {
                        p.max(P_CHECKED_KING, k, fresh(last_mn));
                    }
                } else {
                    if let Some((_, to)) = my_moves.last().and_then(|u| move_squares(u)) {
                        p.max(P_GAVE_CHECK, to, fresh(last_mn));
                    }
                    gave_check_last = true;
                }
            }
            Observation::OpponentFoul { .. } => {}
        }
        model.apply(event);
    }
    for v in &mut p.buf[P_LOST_COUNT * 81..(P_LOST_COUNT + 1) * 81] {
        *v = v.min(1.0);
    }

    // 自分の着手履歴（新しい順）
    for (k, usi) in my_moves.iter().rev().take(HISTORY_MOVES).enumerate() {
        if let Some((from, to)) = move_squares(usi) {
            if let Some(from) = from {
                p.set(P_HIST + 2 * k, from, 1.0);
            }
            p.set(P_HIST + 2 * k + 1, to, 1.0);
        }
    }

    // 今の手番の反則試行
    for usi in foul_tried {
        put_foul(&mut p, P_FOUL, usi, 1.0);
    }

    // 初期配置から動いていない自駒（今もそこに自駒がいるものだけ）
    for piece in &view.your_pieces {
        if let Some(c) = parse_usi_square(&piece.square) {
            if unmoved.contains(&c) {
                p.set(P_UNMOVED, c, 1.0);
            }
        }
    }

    // 持ち駒と相手の保有数（総数 − 自駒 − 自分の持ち駒）。平手では駒の総数が一定なので
    // 視界だけから確定し、履歴が欠けていても正しい
    let mut mine = [0u32; 7];
    for piece in &view.your_pieces {
        if let Some(i) = hand_index(unpromote_role(piece.role)) {
            mine[i] += 1;
        }
    }
    for (i, role) in HAND_ROLES.iter().enumerate() {
        let n = view.your_hand.get(role).copied().unwrap_or(0);
        let total = ROLE_TOTAL[i] as f32;
        p.fill(P_HAND + i, n as f32 / total);
        let hold = ROLE_TOTAL[i].saturating_sub(mine[i] + n) as f32;
        p.fill(P_OPP_HOLD + i, hold / total);
    }
    let history_broken = !history_consistent(&model, view);

    // スカラー
    let max_fouls = MAX_FOULS as f32;
    let scalars = [
        (S_MY_FOULS, view.fouls.you as f32 / max_fouls),
        (S_OPP_FOULS, view.fouls.opponent as f32 / max_fouls),
        (S_MY_LAST_FOUL, (view.fouls.you + 1 >= MAX_FOULS) as u8 as f32),
        (S_OPP_LAST_FOUL, (view.fouls.opponent + 1 >= MAX_FOULS) as u8 as f32),
        (S_FOULS_THIS_TURN, foul_tried.len() as f32 / max_fouls),
        (S_IN_CHECK, view.you_in_check as u8 as f32),
        (S_GAVE_CHECK_LAST, gave_check_last as u8 as f32),
        (S_MOVE_NUMBER, now as f32 / MAX_PLIES as f32),
        (S_HISTORY_BROKEN, history_broken as u8 as f32),
    ];
    for (s, v) in scalars {
        p.fill(P_SCALAR + s, v);
    }
}

pub fn encode(view: &PlayerView, log: &ObservationLog, foul_tried: &HashSet<String>) -> Vec<f32> {
    let mut out = vec![0.0; OBS_LEN];
    encode_into(view, log, foul_tried, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::referee::{Referee, StepResult};
    use crate::rl::action::{NUM_ACTIONS, decode_usi, index_square, legal_mask};
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn plane(obs: &[f32], k: usize) -> &[f32] {
        &obs[k * 81..(k + 1) * 81]
    }

    #[test]
    fn 初期局面は先後で同じテンソルになる() {
        let referee = Referee::new();
        let s = encode(
            &referee.view(Color::Sente, [0, 0], 0),
            referee.log(Color::Sente),
            referee.foul_tried(Color::Sente),
        );
        let g = encode(
            &referee.view(Color::Gote, [0, 0], 0),
            referee.log(Color::Gote),
            referee.foul_tried(Color::Gote),
        );
        // 手数スカラー以外は完全に一致（正規化で後手も「下から攻める」形になる）
        for k in 0..NUM_PLANES {
            if k == P_SCALAR + S_MOVE_NUMBER {
                continue;
            }
            assert_eq!(plane(&s, k), plane(&g, k), "plane {k}");
        }
        // 玉は正規化後の 5九（段9・筋5）
        let king = ROLES.iter().position(|&r| r == Role::King).unwrap();
        let sq = plane(&s, P_OWN + king).iter().position(|&v| v == 1.0).unwrap();
        assert_eq!(index_square(sq), Coord { file: 5, rank: 9 });
        // 相手の保有数は初期枚数
        assert!((plane(&s, P_OPP_HOLD)[0] - 9.0 / 18.0).abs() < 1e-6);
    }

    #[test]
    fn 反則と取られたマスと保有数が載る() {
        let mut referee = Referee::new();
        // 先手 7六歩、後手 3四歩、先手 2二角成（角交換）、後手 同銀
        for usi in ["7g7f", "3c3d", "8h2b+", "3a2b"] {
            assert!(matches!(referee.step(usi, 0), StepResult::Accepted));
        }
        // 先手が反則（1九香→1一 は擬似合法ですらない）
        assert!(matches!(referee.step("1i1a", 0), StepResult::Foul));
        let side = Color::Sente;
        let obs = encode(
            &referee.view(side, [0, 0], 0),
            referee.log(side),
            referee.foul_tried(side),
        );
        // 取ったマス（2二）と、取られたマス（2二で成った角を取られた）
        let sq22 = square_index(Coord { file: 2, rank: 2 });
        assert!(plane(&obs, P_CAP)[sq22] > 0.0);
        assert!(plane(&obs, P_LOST)[sq22] > 0.0);
        // 今の手番の反則
        let sq11 = square_index(Coord { file: 1, rank: 1 });
        assert_eq!(plane(&obs, P_FOUL + F_TO)[sq11], 1.0);
        assert_eq!(plane(&obs, P_PAST_FOUL + F_TO)[sq11], 0.0, "今の手番の反則は過去に載らない");
        assert!((plane(&obs, P_SCALAR + S_FOULS_THIS_TURN)[0] - 0.1).abs() < 1e-6);
        // 相手の角の保有数: 初期1 + 取られた1 − 取った1 = 1
        assert!((plane(&obs, P_OPP_HOLD + 5)[0] - 0.5).abs() < 1e-6);
        // 持ち駒: 角1枚
        assert!((plane(&obs, P_HAND + 5)[0] - 0.5).abs() < 1e-6);
    }

    fn encode_side(referee: &Referee, side: Color) -> Vec<f32> {
        encode(
            &referee.view(side, [0, 0], 0),
            referee.log(side),
            referee.foul_tried(side),
        )
    }

    #[test]
    fn 王手の履歴が両者に載る() {
        let mut referee = Referee::new();
        // 7六歩・3四歩のあと 3三角（不成）で 4二 を通して後手玉 5一 に王手
        for usi in ["7g7f", "3c3d", "8h3c"] {
            assert!(matches!(referee.step(usi, 0), StepResult::Accepted));
        }
        let gote = encode_side(&referee, Color::Gote);
        assert_eq!(plane(&gote, P_SCALAR + S_IN_CHECK)[0], 1.0);
        // 後手の自玉 5一 は正規化で 5九
        let king_sq = square_index(Coord { file: 5, rank: 9 });
        assert!(plane(&gote, P_CHECKED_KING)[king_sq] > 0.9);
        assert_eq!(plane(&gote, P_CHECKED_KING).iter().filter(|&&v| v > 0.0).count(), 1);

        let sente = encode_side(&referee, Color::Sente);
        let sq33 = square_index(Coord { file: 3, rank: 3 });
        assert!(plane(&sente, P_GAVE_CHECK)[sq33] > 0.9);
        assert_eq!(plane(&sente, P_SCALAR + S_GAVE_CHECK_LAST)[0], 1.0);
        assert_eq!(plane(&sente, P_SCALAR + S_HISTORY_BROKEN)[0], 0.0);
    }

    #[test]
    fn 手番が変わると反則は過去の反則へ移り打ちの駒種と成りが残る() {
        let mut referee = Referee::new();
        for usi in ["7g7f", "3c3d", "8h2b+", "3a2b"] {
            assert!(matches!(referee.step(usi, 0), StepResult::Accepted));
        }
        // 先手: 盤上の反則（1九香→1一）・打ちの反則（角を後手の金 4一 へ）・成りの反則
        // （2八飛→2三成 は 2七歩が塞ぐので反則）
        for usi in ["1i1a", "B*4a", "2h2c+"] {
            assert!(matches!(referee.step(usi, 0), StepResult::Foul), "{usi}");
        }
        let now = encode_side(&referee, Color::Sente);
        let sq41 = square_index(Coord { file: 4, rank: 1 });
        let sq23 = square_index(Coord { file: 2, rank: 3 });
        assert_eq!(plane(&now, P_FOUL + F_DROP + 5)[sq41], 1.0, "角の打ち反則");
        assert_eq!(plane(&now, P_FOUL + F_PROMO_TO)[sq23], 1.0, "成りの反則");
        assert_eq!(plane(&now, P_PAST_FOUL + F_DROP + 5)[sq41], 0.0);

        assert!(matches!(referee.step("6i7h", 0), StepResult::Accepted));
        assert!(matches!(referee.step("4a4b", 0), StepResult::Accepted));
        let later = encode_side(&referee, Color::Sente);
        assert_eq!(plane(&later, P_FOUL + F_DROP + 5)[sq41], 0.0);
        assert!(plane(&later, P_PAST_FOUL + F_DROP + 5)[sq41] > 0.8, "打ちの駒種が残る");
        assert!(plane(&later, P_PAST_FOUL + F_PROMO_TO)[sq23] > 0.8, "成りが残る");
        let sq11 = square_index(Coord { file: 1, rank: 1 });
        assert!(plane(&later, P_PAST_FOUL + F_TO)[sq11] > 0.8);
        assert_eq!(
            plane(&later, P_PAST_FOUL + F_DROP)[sq41],
            0.0,
            "歩の打ち反則とは区別される"
        );
    }

    #[test]
    fn 履歴が視界と食い違えば印が立ち保有数は視界から正しく出る() {
        use crate::board::parse_usi_square;
        use crate::selfplay::StartState;
        use crate::shogi::Piece;
        let sq = |s: &str| parse_usi_square(s).unwrap();
        // 履歴なし（空ログ）で途中局面から始める = 再起動で履歴を失った状態
        let mut pos = Position::empty(Color::Sente);
        pos.set(sq("5i"), Some(Piece { color: Color::Sente, role: Role::King }));
        pos.set(sq("5a"), Some(Piece { color: Color::Gote, role: Role::King }));
        pos.set(sq("2b"), Some(Piece { color: Color::Sente, role: Role::Horse }));
        pos.set_hand(Color::Sente, Role::Bishop, 1);
        let referee = Referee::from_start(StartState {
            pos,
            logs: [ObservationLog::default(), ObservationLog::default()],
            fouls: [0, 0],
            plies: 60,
        });
        let obs = encode_side(&referee, Color::Sente);
        assert_eq!(plane(&obs, P_SCALAR + S_HISTORY_BROKEN)[0], 1.0);
        // 角: 総数2 − 盤上の馬1 − 持ち駒1 = 0、歩: 18 − 0 − 0 = 18
        assert_eq!(plane(&obs, P_OPP_HOLD + 5)[0], 0.0);
        assert_eq!(plane(&obs, P_OPP_HOLD)[0], 1.0);
    }

    /// 1決定点あたりのエンコード＋マスクの所要時間（自己対局の速度の上限を決める）。
    /// `cargo test --release --lib エンコードの速度 -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn エンコードの速度() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut n = 0u32;
        let mut elapsed = std::time::Duration::ZERO;
        let mut buf = vec![0.0; OBS_LEN];
        for game in 0..20 {
            let mut referee = Referee::new();
            loop {
                let side = referee.to_move();
                let view = referee.view(side, [0, 0], game);
                let t = std::time::Instant::now();
                encode_into(&view, referee.log(side), referee.foul_tried(side), &mut buf);
                let mask = legal_mask(&view, referee.log(side), referee.foul_tried(side));
                elapsed += t.elapsed();
                n += 1;
                let actions: Vec<usize> = (0..NUM_ACTIONS).filter(|&a| mask[a]).collect();
                if actions.is_empty() {
                    break;
                }
                let usi = decode_usi(actions[rng.random_range(0..actions.len())], side).unwrap();
                if let StepResult::Ended { .. } = referee.step(&usi, 0) {
                    break;
                }
            }
        }
        println!(
            "{n} 決定点、平均 {:.1}µs/決定点",
            elapsed.as_secs_f64() * 1e6 / n as f64
        );
    }

    /// ランダム対局の全局面で値域が壊れない（NaN・無限大・負値が出ない）
    #[test]
    fn ランダム対局で値域が正常() {
        let mut rng = StdRng::seed_from_u64(7);
        for game in 0..10 {
            let mut referee = Referee::new();
            loop {
                let side = referee.to_move();
                let view = referee.view(side, [0, 0], game);
                let obs = encode(&view, referee.log(side), referee.foul_tried(side));
                assert!(obs.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 3.0));
                let mask = legal_mask(&view, referee.log(side), referee.foul_tried(side));
                let actions: Vec<usize> = (0..NUM_ACTIONS).filter(|&a| mask[a]).collect();
                if actions.is_empty() {
                    break;
                }
                let usi = decode_usi(actions[rng.random_range(0..actions.len())], side).unwrap();
                if let StepResult::Ended { .. } = referee.step(&usi, 0) {
                    break;
                }
            }
        }
    }
}
