//! **局の信号**: 報酬の整形の材料と、学習の監視値（docs/rl-fixed-opponent.md）。
//!
//! 勝敗（±1）以外に「学習に効く量」を探すための土台。env は各側の信号をそのまま出すだけで、
//! **どれにどんな重みを付けて報酬にするかは学習側（tsuitate-nn の `rnad/shaping.py`）が決める**。
//! 信号を1つ足すのに要るのはここの1行と計算だけで、報酬の式は Python の引数で振れる。
//!
//! 信号は2種類ある:
//! - **累積**（`moves`〜`evasions`）: 局の頭からの出来事の数。`EnvGame::play` が数える
//! - **状態**（`material`〜`king_holes`）: 今の真実の盤面から計算する量
//!
//! どちらも「その時点の局の状態の関数」なので、学習側は連続する2時点の差を報酬にできる
//! （終局で累計を差し戻せばポテンシャル型の整形になり、最適方策を変えない）。
//!
//! **真実の盤面を読む信号がある**（`king_attack` は相手玉の位置を、`mate_threat` は相手の全部の駒を使う）。報酬と監視は学習時にしか
//! 使わないので情報は漏れないが、**方策の入力には使わないこと**（`EnvGame::truth_into` と同じ扱い）。

use crate::board::{Coord, on_board};
use crate::protocol::{Color, Role};
use crate::shogi::{HAND_ROLES, Position, piece_value, unpromote_role};

pub const NUM_SIGNALS: usize = 23;

/// 信号の名前（`signals_into` の並び）。学習側は名前で引く（`tsuitate_rl.SIGNAL_NAMES`）
pub const SIGNAL_NAMES: [&str; NUM_SIGNALS] = [
    // ---- 累積（局の頭からの数）
    "moves",          // 受理された手
    "fouls",          // 反則
    "fouls_in_check", // 王手を受けている手番での反則
    "captures",       // 取った駒の数
    "capture_value",  // 取った駒の価値の合計（`capture_value`。歩 = 1）
    "promotions",     // 成った回数
    "checks",         // 王手をかけた回数
    "drops",          // 打った回数
    "evasions",       // 王手を受けている手番で受理された手（= 王手を外した回数）
    // ---- 状態（今の真実の盤面）
    "material",     // 自分の駒の価値の合計（盤上は成りを含む価値、持ち駒は元の駒の価値。玉は 0）
    "hand_value",   // 持ち駒の価値の合計
    "king_attack",  // 相手玉のマスと8近傍への、自駒の利きの数の合計
    "king_guard",   // 自玉の8近傍のうち自駒がいる割合（`env::guard_fraction` の半径1）
    "king_advance", // 自玉の前進段数（0〜8）
    "camp_pieces",  // 敵陣（相手側の3段）にいる玉以外の自駒の数
    "king_escape",  // 自玉の8近傍のうち、自駒がおらず相手の利きも無いマスの数（玉の逃げ場）
    "mate_threat",  // 一手詰めがあるか（0 / 1。自分の手番なら全合法手の一手詰め、相手の手番なら打ちの詰めろ。`mate_threat`）
    // ---- 状態: やねうら王の手作り評価の項（docs/yaneuraou-lessons.md の V1〜V4。2026-11-12）。
    // estimator では相手玉の位置と相手の利きが信念でしか分からなかったが、ここは真実の盤面から数える
    "effect_attack",  // 自駒の利きを**相手玉からの距離**で重み付けた総和（V2 の攻め側。利きの数は逓減 = V1）
    "effect_guard",   // 自駒の利きを**自玉からの距離**で重み付けた総和（V2 の守り側）
    "tied_value",     // 紐のついた（自駒の利きがある）玉以外の自駒の価値の合計（V3 の紐）
    "attacked_value", // 相手の利きがある玉以外の自駒の価値の合計（V3 の質）
    "hanging_value",  // そのうち紐のついていない自駒の価値の合計（ただ取られる形）
    "king_holes",     // 自玉の8近傍のうち、自駒がおらず、玉以外の自駒の利きも無いマスの数（V4 の穴）
];

const MOVES: usize = 0;
const FOULS: usize = 1;
const FOULS_IN_CHECK: usize = 2;
const CAPTURES: usize = 3;
const CAPTURE_VALUE: usize = 4;
const PROMOTIONS: usize = 5;
const CHECKS: usize = 6;
const DROPS: usize = 7;
const EVASIONS: usize = 8;
const NUM_EVENTS: usize = 9;
const MATERIAL: usize = 9;
const HAND_VALUE: usize = 10;
const KING_ATTACK: usize = 11;
const KING_GUARD: usize = 12;
const KING_ADVANCE: usize = 13;
const CAMP_PIECES: usize = 14;
const KING_ESCAPE: usize = 15;
const MATE_THREAT: usize = 16;
const EFFECT_ATTACK: usize = 17;
const EFFECT_GUARD: usize = 18;
const TIED_VALUE: usize = 19;
const ATTACKED_VALUE: usize = 20;
const HANGING_VALUE: usize = 21;
const KING_HOLES: usize = 22;

/// 盤の全マスへの利きの数（[先手の利き, 後手の利き]。添字は `sq_index`）
type AttackMap = [[u8; 81]; 2];

fn sq_index(c: Coord) -> usize {
    (c.file as usize - 1) * 9 + (c.rank as usize - 1)
}

fn all_squares() -> impl Iterator<Item = Coord> {
    (1..=9).flat_map(|file| (1..=9).map(move |rank| Coord { file, rank }))
}

fn attack_map(pos: &Position) -> AttackMap {
    let mut map = [[0u8; 81]; 2];
    for c in all_squares() {
        map[0][sq_index(c)] = pos.attack_count(c, Color::Sente);
        map[1][sq_index(c)] = pos.attack_count(c, Color::Gote);
    }
    map
}

/// 1マスに利きが `m` 本あるときの価値（1本 = 1）。やねうら王の連載の Lv4 の実測の式
/// `6365 − 0.8525^(m−1) × 5341`（1本目 1024）をそのまま使う: 2本で 1.77、3本で 2.43 と逓減するが飽和はしない
fn effect_multiplicity(m: u8) -> f64 {
    if m == 0 {
        0.0
    } else {
        (6365.0 - 0.8525f64.powi(i32::from(m) - 1) * 5341.0) / 1024.0
    }
}

/// 玉からの距離（チェビシェフ）での利きの重み。やねうら王の Lv3 の実測（1024, 496, 297, …）に近い
/// `1 / (1 + d)`（`strategy::king_dist_weight` と同じ式）
fn king_dist_weight(a: Coord, king: Coord) -> f64 {
    let d = (a.file - king.file).abs().max((a.rank - king.rank).abs());
    1.0 / (1.0 + f64::from(d))
}

/// 駒を1枚取ったときの価値 =（盤上での価値 + 持ち駒としての価値）÷ 2。
/// 取られた側は盤上の価値を失い、取った側は元の駒を持ち駒に得るので、その平均を1回の駒取りの大きさにする
/// （と金は (6 + 1) / 2 = 3.5。`strategy::exchange_value` と同じ式）
pub fn capture_value(role: Role) -> f32 {
    ((piece_value(role) + piece_value(unpromote_role(role))) / 2.0) as f32
}

/// 1つの側の出来事（`EnvGame::play` が1手ごとに渡す）
#[derive(Debug, Clone, Copy, Default)]
pub struct MoveEvent {
    /// 受理されたか（false = 反則）
    pub accepted: bool,
    /// 手番の頭で王手を受けていたか
    pub was_in_check: bool,
    /// 打ちか
    pub drop: bool,
    /// 成ったか
    pub promoted: bool,
    /// 取った駒（盤上での駒種）
    pub captured: Option<Role>,
    /// 相手に王手をかけたか
    pub gave_check: bool,
}

/// 累積の信号 [先手, 後手]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EventCounts([[f32; NUM_EVENTS]; 2]);

impl EventCounts {
    pub fn record(&mut self, side: Color, e: MoveEvent) {
        let c = &mut self.0[usize::from(side == Color::Gote)];
        if !e.accepted {
            c[FOULS] += 1.0;
            if e.was_in_check {
                c[FOULS_IN_CHECK] += 1.0;
            }
            return;
        }
        c[MOVES] += 1.0;
        if e.was_in_check {
            c[EVASIONS] += 1.0;
        }
        if e.drop {
            c[DROPS] += 1.0;
        }
        if e.promoted {
            c[PROMOTIONS] += 1.0;
        }
        if let Some(role) = e.captured {
            c[CAPTURES] += 1.0;
            c[CAPTURE_VALUE] += capture_value(role);
        }
        if e.gave_check {
            c[CHECKS] += 1.0;
        }
    }

    pub fn get(&self, side: Color, name: &str) -> Option<f32> {
        let k = SIGNAL_NAMES[..NUM_EVENTS].iter().position(|&n| n == name)?;
        Some(self.0[usize::from(side == Color::Gote)][k])
    }
}

/// 前進段数（先手は 9 段目が自陣の最下段、後手は 1 段目）
fn advance(c: Coord, color: Color) -> i8 {
    match color {
        Color::Sente => 9 - c.rank,
        Color::Gote => c.rank - 1,
    }
}

/// `color` の信号を `out`（長さ `NUM_SIGNALS`）へ書く
fn side_signals(pos: &Position, events: &EventCounts, attacks: &AttackMap, color: Color, out: &mut [f32]) {
    debug_assert_eq!(out.len(), NUM_SIGNALS);
    out[..NUM_EVENTS].copy_from_slice(&events.0[usize::from(color == Color::Gote)]);

    let mut material = 0.0f64;
    let mut camp = 0u32;
    for (c, p) in pos.pieces() {
        if p.color != color {
            continue;
        }
        material += piece_value(p.role);
        if p.role != Role::King && advance(c, color) >= 6 {
            camp += 1;
        }
    }
    let hand: f64 = HAND_ROLES
        .iter()
        .map(|&r| f64::from(pos.hand_count(color, r)) * piece_value(r))
        .sum();
    out[MATERIAL] = (material + hand) as f32;
    out[HAND_VALUE] = hand as f32;
    out[CAMP_PIECES] = camp as f32;

    // 相手玉のマスと8近傍への利きの数（盤の外は数えない）
    let mut attack = 0u32;
    if let Some(k) = pos.king_square(color.other()) {
        for df in -1..=1 {
            for dr in -1..=1 {
                let c = Coord { file: k.file + df, rank: k.rank + dr };
                if on_board(c) {
                    attack += u32::from(pos.attack_count(c, color));
                }
            }
        }
    }
    out[KING_ATTACK] = attack as f32;

    let (mut squares, mut own, mut escape) = (0u32, 0u32, 0u32);
    let mut king_adv = 0i8;
    if let Some(k) = pos.king_square(color) {
        king_adv = advance(k, color);
        for df in -1..=1 {
            for dr in -1..=1 {
                let c = Coord { file: k.file + df, rank: k.rank + dr };
                if (df, dr) == (0, 0) || !on_board(c) {
                    continue;
                }
                squares += 1;
                if pos.piece_at(c).is_some_and(|p| p.color == color) {
                    own += 1;
                } else if !pos.is_attacked(c, color.other()) {
                    // 玉の逃げ場（玉自身が遮っている利きは数えない近似。相手の駒がいるマスは「取って逃げる」先に数える）
                    escape += 1;
                }
            }
        }
    }
    out[KING_GUARD] = if squares == 0 { 0.0 } else { own as f32 / squares as f32 };
    out[KING_ADVANCE] = f32::from(king_adv);
    out[KING_ESCAPE] = escape as f32;
    out[MATE_THREAT] = if mate_threat(pos, color) { 1.0 } else { 0.0 };

    // やねうら王の手作り評価の項（V1〜V4）。利きの数は盤の全マスぶんを `attacks` に持っている
    let (own_att, opp_att) = match color {
        Color::Sente => (&attacks[0], &attacks[1]),
        Color::Gote => (&attacks[1], &attacks[0]),
    };
    let own_king = pos.king_square(color);
    let opp_king = pos.king_square(color.other());
    let (mut effect_attack, mut effect_guard) = (0.0f64, 0.0f64);
    for c in all_squares() {
        let v = effect_multiplicity(own_att[sq_index(c)]);
        if v == 0.0 {
            continue;
        }
        if let Some(k) = opp_king {
            effect_attack += v * king_dist_weight(c, k);
        }
        if let Some(k) = own_king {
            effect_guard += v * king_dist_weight(c, k);
        }
    }
    let (mut tied, mut attacked, mut hanging) = (0.0f64, 0.0f64, 0.0f64);
    for (c, p) in pos.pieces() {
        if p.color != color || p.role == Role::King {
            continue;
        }
        let v = piece_value(p.role);
        let i = sq_index(c);
        if own_att[i] > 0 {
            tied += v;
        }
        if opp_att[i] > 0 {
            attacked += v;
            if own_att[i] == 0 {
                hanging += v;
            }
        }
    }
    // 穴: 玉自身の利きは支えに数えない（玉の隣のマスには必ず玉の利きが1本あるので、それを引く）。
    // 自駒が乗っているマスは壁として働くので穴に数えない（`strategy` の king_holes と同じ決め方）
    let mut holes = 0u32;
    if let Some(k) = own_king {
        for df in -1..=1 {
            for dr in -1..=1 {
                let c = Coord { file: k.file + df, rank: k.rank + dr };
                if (df, dr) == (0, 0) || !on_board(c) {
                    continue;
                }
                let own_piece = pos.piece_at(c).is_some_and(|p| p.color == color);
                if !own_piece && own_att[sq_index(c)] <= 1 {
                    holes += 1;
                }
            }
        }
    }
    out[EFFECT_ATTACK] = effect_attack as f32;
    out[EFFECT_GUARD] = effect_guard as f32;
    out[TIED_VALUE] = tied as f32;
    out[ATTACKED_VALUE] = attacked as f32;
    out[HANGING_VALUE] = hanging as f32;
    out[KING_HOLES] = holes as f32;
}

/// `color` に**一手詰めがあるか**（詰みへの近さの信号）。
///
/// - `color` の手番: 全合法手の一手詰め（`mate::has_mate_in_1_fast`）
/// - 相手の手番: 相手が王手を受けていなければ、**持ち駒の打ちの詰めろ**（相手が何も受けなければ次に
///   打ちで詰む。`mate::drop_mate`）。相手が王手を受けている間は 0（もう1手続けて指せる前提の詰みは数えない）
///
/// 報酬にするときは状態の差分（かけた時に +、消えた時に −）なので、詰めろを保ったまま指さずに稼ぐことはできない
pub fn mate_threat(pos: &Position, color: Color) -> bool {
    if pos.turn() == color {
        crate::mate::has_mate_in_1_fast(pos)
    } else if pos.in_check(pos.turn()) {
        false
    } else {
        crate::mate::drop_mate(pos, color).is_some()
    }
}

/// 両者の信号を `out`（長さ `2 * NUM_SIGNALS`、[先手の信号…, 後手の信号…]）へ書く
pub fn signals_into(pos: &Position, events: &EventCounts, out: &mut [f32]) {
    debug_assert_eq!(out.len(), 2 * NUM_SIGNALS);
    let (s, g) = out.split_at_mut(NUM_SIGNALS);
    let attacks = attack_map(pos);
    side_signals(pos, events, &attacks, Color::Sente, s);
    side_signals(pos, events, &attacks, Color::Gote, g);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index(name: &str) -> usize {
        SIGNAL_NAMES.iter().position(|&n| n == name).unwrap()
    }

    #[test]
    fn 名前と添字がそろっている() {
        for (name, k) in [
            ("moves", MOVES),
            ("fouls", FOULS),
            ("fouls_in_check", FOULS_IN_CHECK),
            ("captures", CAPTURES),
            ("capture_value", CAPTURE_VALUE),
            ("promotions", PROMOTIONS),
            ("checks", CHECKS),
            ("drops", DROPS),
            ("evasions", EVASIONS),
            ("material", MATERIAL),
            ("hand_value", HAND_VALUE),
            ("king_attack", KING_ATTACK),
            ("king_guard", KING_GUARD),
            ("king_advance", KING_ADVANCE),
            ("camp_pieces", CAMP_PIECES),
            ("king_escape", KING_ESCAPE),
            ("mate_threat", MATE_THREAT),
            ("effect_attack", EFFECT_ATTACK),
            ("effect_guard", EFFECT_GUARD),
            ("tied_value", TIED_VALUE),
            ("attacked_value", ATTACKED_VALUE),
            ("hanging_value", HANGING_VALUE),
            ("king_holes", KING_HOLES),
        ] {
            assert_eq!(index(name), k, "{name}");
        }
        // 累積の信号が前半に並ぶ
        assert_eq!(NUM_EVENTS, MATERIAL);
        let mut names = SIGNAL_NAMES.to_vec();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), NUM_SIGNALS, "名前は重複しない");
    }

    #[test]
    fn 初期局面の状態の信号は先後で同じ() {
        let pos = Position::initial();
        let mut out = [0.0f32; 2 * NUM_SIGNALS];
        signals_into(&pos, &EventCounts::default(), &mut out);
        let (s, g) = out.split_at(NUM_SIGNALS);
        assert_eq!(s, g);
        // 歩9 + 香2 + 桂2 + 銀2 + 金2 + 角 + 飛 = 9 + 6 + 7 + 10 + 11 + 8 + 9.5
        assert!((s[MATERIAL] - 60.5).abs() < 1e-6);
        assert_eq!(s[HAND_VALUE], 0.0);
        assert_eq!(s[KING_ATTACK], 0.0);
        assert_eq!(s[KING_ADVANCE], 0.0);
        assert_eq!(s[CAMP_PIECES], 0.0);
        // 5九玉の8近傍のうち盤上は5マス、自駒は 4九金・6九金の2枚（4八・5八・6八は空き）
        assert!((s[KING_GUARD] - 0.4).abs() < 1e-6);
        // 空いている3マス（4八・5八・6八）に相手の利きは無い
        assert_eq!(s[KING_ESCAPE], 3.0);
        assert_eq!(s[MATE_THREAT], 0.0);
        assert!(s[..NUM_EVENTS].iter().all(|&v| v == 0.0));
        // やねうら王の項: 初期局面は相手の利きが自駒に届かず、玉の隣の空きマス（4八・5八・6八）には金銀飛の利きがある
        assert_eq!(s[ATTACKED_VALUE], 0.0);
        assert_eq!(s[HANGING_VALUE], 0.0);
        assert_eq!(s[KING_HOLES], 0.0);
        assert!(s[TIED_VALUE] > 0.0 && s[TIED_VALUE] < s[MATERIAL]);
        assert!(s[EFFECT_GUARD] > s[EFFECT_ATTACK] && s[EFFECT_ATTACK] > 0.0, "利きは自玉の近くに集まっている");
    }

    #[test]
    fn やねうら王の項は真実の盤面の利きから数える() {
        let put = |pos: &mut Position, sq: &str, color, role| {
            pos.set(crate::board::parse_usi_square(sq).unwrap(), Some(crate::shogi::Piece { color, role }));
        };
        let signals = |pos: &Position| {
            let mut out = [0.0f32; 2 * NUM_SIGNALS];
            signals_into(pos, &EventCounts::default(), &mut out);
            out
        };
        let gold = piece_value(Role::Gold) as f32;
        // 先手: 玉 5九・金 4八・歩 7七。後手: 玉 5一・飛 4二（4筋を通して 4八の金に当たっている）
        let mut pos = Position::empty(Color::Sente);
        put(&mut pos, "5i", Color::Sente, Role::King);
        put(&mut pos, "4h", Color::Sente, Role::Gold);
        put(&mut pos, "7g", Color::Sente, Role::Pawn);
        put(&mut pos, "5a", Color::Gote, Role::King);
        put(&mut pos, "4b", Color::Gote, Role::Rook);
        let out = signals(&pos);
        let s = &out[..NUM_SIGNALS];
        // 金は飛に当てられているが玉の紐がある。歩は当たっておらず紐も無い
        assert_eq!(s[ATTACKED_VALUE], gold);
        assert_eq!(s[TIED_VALUE], gold);
        assert_eq!(s[HANGING_VALUE], 0.0);
        // 玉の隣（4九・6九・4八・5八・6八）: 4八は自駒、4九・5八は金の利きがある。6九・6八は玉しか利いていない = 穴
        assert_eq!(s[KING_HOLES], 2.0);
        // 後手の飛には先手の利きが無く、後手玉の紐がある
        let g = &out[NUM_SIGNALS..];
        assert_eq!(g[ATTACKED_VALUE], 0.0);
        assert_eq!(g[TIED_VALUE], piece_value(Role::Rook) as f32);

        // 玉が離れると金はただ取られる形になる
        let mut far = pos.clone();
        far.set(crate::board::parse_usi_square("5i").unwrap(), None);
        put(&mut far, "9i", Color::Sente, Role::King);
        let out = signals(&far);
        assert_eq!(out[HANGING_VALUE], gold);
        assert_eq!(out[TIED_VALUE], 0.0);

        // 相手玉の近くに利きを作ると effect_attack が増える（同じ金でも遠くに置くより大きい）
        let base = signals(&pos)[EFFECT_ATTACK];
        let mut near = pos.clone();
        put(&mut near, "5c", Color::Sente, Role::Gold);
        let mut away = pos.clone();
        put(&mut away, "9g", Color::Sente, Role::Gold);
        let (near, away) = (signals(&near)[EFFECT_ATTACK], signals(&away)[EFFECT_ATTACK]);
        assert!(near > away && away > base, "{near} {away} {base}");
        // 利きの数は逓減する
        assert_eq!(effect_multiplicity(0), 0.0);
        assert!((effect_multiplicity(1) - 1.0).abs() < 1e-12);
        let (m2, m3) = (effect_multiplicity(2), effect_multiplicity(3));
        assert!((m2 - 1.769).abs() < 0.01 && m3 > m2 && m3 - m2 < m2 - 1.0, "{m2} {m3}");
    }

    #[test]
    fn 出来事は指した側にだけ数える() {
        let mut e = EventCounts::default();
        e.record(Color::Sente, MoveEvent { accepted: false, was_in_check: true, ..Default::default() });
        e.record(Color::Sente, MoveEvent { accepted: false, ..Default::default() });
        e.record(
            Color::Gote,
            MoveEvent {
                accepted: true,
                promoted: true,
                captured: Some(Role::Tokin),
                gave_check: true,
                ..Default::default()
            },
        );
        e.record(Color::Gote, MoveEvent { accepted: true, drop: true, was_in_check: true, ..Default::default() });
        assert_eq!(e.get(Color::Sente, "fouls"), Some(2.0));
        assert_eq!(e.get(Color::Sente, "fouls_in_check"), Some(1.0));
        assert_eq!(e.get(Color::Sente, "moves"), Some(0.0));
        assert_eq!(e.get(Color::Gote, "moves"), Some(2.0));
        assert_eq!(e.get(Color::Gote, "captures"), Some(1.0));
        assert_eq!(e.get(Color::Gote, "capture_value"), Some(3.5));
        assert_eq!(e.get(Color::Gote, "promotions"), Some(1.0));
        assert_eq!(e.get(Color::Gote, "checks"), Some(1.0));
        assert_eq!(e.get(Color::Gote, "drops"), Some(1.0));
        // 王手を受けている手番で受理された手だけが「王手を外した」（反則した試行は数えない）
        assert_eq!(e.get(Color::Gote, "evasions"), Some(1.0));
        assert_eq!(e.get(Color::Sente, "evasions"), Some(0.0));
        assert_eq!(e.get(Color::Gote, "fouls"), Some(0.0));
        assert_eq!(e.get(Color::Gote, "material"), None);
    }

    #[test]
    fn 一手詰めの信号は手番と王手で分ける() {
        use crate::shogi::parse_usi;
        // 頭金: 後手玉 5一、先手の歩 5三（金の支え）、先手の持ち駒に金。先手番
        let mut pos = Position::empty(Color::Sente);
        let put = |pos: &mut Position, sq: &str, color, role| {
            pos.set(crate::board::parse_usi_square(sq).unwrap(), Some(crate::shogi::Piece { color, role }));
        };
        put(&mut pos, "5a", Color::Gote, Role::King);
        put(&mut pos, "5c", Color::Sente, Role::Pawn);
        put(&mut pos, "5i", Color::Sente, Role::King);
        pos.set_hand(Color::Sente, Role::Gold, 1);
        pos.set_turn(Color::Sente);
        assert!(mate_threat(&pos, Color::Sente), "先手番: G*5b の一手詰め");
        assert!(!mate_threat(&pos, Color::Gote));
        // 後手番にしても、後手が何も受けなければ打ちで詰む（詰めろ）
        pos.set_turn(Color::Gote);
        assert!(mate_threat(&pos, Color::Sente), "後手番: 打ちの詰めろ");
        // 詰ませた後（後手玉が王手を受けている）は 0
        pos.set_turn(Color::Sente);
        pos.play_unchecked(&parse_usi("G*5b").unwrap());
        assert!(pos.in_check(Color::Gote));
        assert!(!mate_threat(&pos, Color::Sente));
        let mut out = [0.0f32; 2 * NUM_SIGNALS];
        signals_into(&pos, &EventCounts::default(), &mut out);
        // 後手玉 5一の逃げ場: 4一・6一は金の利き、4二・6二も金の利き、5二は金（歩の支えで取れない）= 0
        assert_eq!(out[NUM_SIGNALS + KING_ESCAPE], 0.0);
    }

    #[test]
    fn 駒を取る価値は成駒で割り引く() {
        assert_eq!(capture_value(Role::Pawn), 1.0);
        assert_eq!(capture_value(Role::Tokin), 3.5);
        assert_eq!(capture_value(Role::Rook), 9.5);
        assert!((capture_value(Role::Dragon) - 10.75).abs() < 1e-6);
        assert_eq!(capture_value(Role::King), 0.0);
    }
}
