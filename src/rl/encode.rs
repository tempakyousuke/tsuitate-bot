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
use crate::deduce::{ImmobilePawns, KingCandidates};
use crate::model::GameModel;
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{Color, PlayerView, Role};
use crate::rl::action::{deduced_facts, normalize, square_index};
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

// 確定事実のプレーン（エンコーダ v2、2026-10-16。docs/rl-deepnash-design.md の改善案2）。
// v1（`NUM_PLANES`）の後ろに足すだけなので、v1 の重みの stem に 0 の列を足せば同じ関数から始められる。
// どれも観測から論理的に確定する事実（真の盤面と矛盾しない）で、`action::deduced_facts` と `deduce` が作る
const D_OCC: usize = NUM_PLANES; // 相手の駒がいると確定したマス
const D_BLOCK: usize = D_OCC + 1; // 飛び越え反則の間のマス（どれかに相手の駒がいる。値は 1/マス数）
const D_CHECKER: usize = D_BLOCK + 1; // 王手中: 王手駒がいうるマス（値は 1/仮説数）
const D_KING: usize = D_CHECKER + 1; // 相手玉がいうるマス（deduce::opp_king_candidates）
const D_PAWN: usize = D_KING + 1; // 動けないと確定した相手の歩（deduce::immobile_opponent_pawns）
/// エンコーダ v2 のプレーン数（v1 + 確定事実 5枚）
pub const NUM_PLANES_DEDUCED: usize = D_PAWN + 1;

// エンコーダ v3（2026-11-08。docs/rl-deepnash-design.md の「エンコーダ v3」）。v2 の後ろに、観測と自分側の
// 情報だけで決まる事実を2組足す。**並びは固定**で、学習でどの組を使うかはネットの側が持つ（tsuitate-nn の
// `--opp-hand-planes` / `--opp-foul-planes`。使わない組は stem の重みの列が 0 に固定されるので、ここで書いた値は
// 出力に効かない）。並びを固定にしておくと、組の選び方が違うネットどうし（錨・リーグ）が同じ観測を読める
const E_HAND: usize = NUM_PLANES_DEDUCED; // 7: 相手の持ち駒の上界（駒種別。全面に放送）
const E_FOUL: usize = E_HAND + 7; // 3: 相手の反則の時系列（全面に放送）
const EF_LAST: usize = 0; // 相手の直前の手番の反則数
const EF_CHECKED: usize = 1; // 王手を受けていた相手の手番の反則数（鮮度つきの合計）
const EF_FREE: usize = 2; // 王手を受けていない相手の手番の反則数（鮮度つきの合計）
/// エンコーダ v3 のプレーン数（v2 + 相手の持ち駒の上界 7枚 + 相手の反則の時系列 3枚）
pub const NUM_PLANES_V3: usize = E_FOUL + 3;
/// v3 で足したプレーンの組（名前, 先頭のプレーン, 枚数）。tsuitate-nn の `model.EXTRA_PLANES` と同じ
pub const EXTRA_PLANE_GROUPS: [(&str, usize, usize); 2] = [("opp_hand", E_HAND, 7), ("opp_fouls", E_FOUL, 3)];

/// プレーン数 → 観測の長さ。対応するエンコーダが無いプレーン数は None
pub fn obs_len(planes: usize) -> Option<usize> {
    [NUM_PLANES, NUM_PLANES_DEDUCED, NUM_PLANES_V3]
        .contains(&planes)
        .then_some(planes * 81)
}

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

/// 確定事実のプレーンのうち、観測ログを先頭からたどって求めるもの（相手玉の候補・動けない歩）の
/// 途中状態。同じ局・同じ側の観測を、ログを伸ばしながら何度も書くとき（RL の環境）に使い回すと、
/// 伸びたぶんだけ処理すれば済む（毎回たどり直すと長い対局ほど重い）。結果は使い回さないときと同じ
#[derive(Clone, Default)]
pub struct DeducedCache {
    state: Option<(Color, KingCandidates, ImmobilePawns)>,
}

impl DeducedCache {
    /// `log` まで進めた途中状態。色が違う・ログが前回より短い（別の局）ときは作り直す
    fn advance(&mut self, color: Color, log: &ObservationLog) -> (&KingCandidates, &ImmobilePawns) {
        let stale = self
            .state
            .as_ref()
            .is_none_or(|(c, _, pawns)| *c != color || pawns.done() > log.events().len());
        if stale {
            self.state = Some((color, KingCandidates::new(color), ImmobilePawns::new(color)));
        }
        let (_, king, pawns) = self.state.as_mut().expect("作った");
        king.advance(log);
        pawns.advance(log);
        (king, pawns)
    }
}

/// 観測をテンソルへ書く。`out` の長さでエンコーダの版を選ぶ: `OBS_LEN` は v1、
/// `NUM_PLANES_DEDUCED * 81` は v2（v1 の後ろに確定事実のプレーン）、`NUM_PLANES_V3 * 81` は v3
/// （v2 の後ろに相手の持ち駒の上界と相手の反則の時系列）
pub fn encode_into(
    view: &PlayerView,
    log: &ObservationLog,
    foul_tried: &HashSet<String>,
    out: &mut [f32],
) {
    encode_into_cached(view, log, foul_tried, out, &mut DeducedCache::default());
}

/// `encode_into` と同じ観測を、確定事実の途中状態 `cache` を使い回して書く。`cache` は同じ局・同じ側の
/// 観測を書くときだけ使い回すこと（局が変わったら作り直す。色が違う・ログが短くなったときは中で作り直す）
pub fn encode_into_cached(
    view: &PlayerView,
    log: &ObservationLog,
    foul_tried: &HashSet<String>,
    out: &mut [f32],
    cache: &mut DeducedCache,
) {
    assert!(
        out.len() % 81 == 0 && obs_len(out.len() / 81).is_some(),
        "観測の長さ {} に対応するエンコーダが無い",
        out.len()
    );
    let v3 = out.len() == NUM_PLANES_V3 * 81;
    out.fill(0.0);
    if out.len() > OBS_LEN {
        encode_deduced(view, log, foul_tried, out, cache);
    }
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
    // 相手の反則（v3）。反則は手番を変えないので、相手の手番の反則は `OpponentMoved` の手前に並ぶ。
    // 相手が王手を受けていた手番か = その手番の前の自分の手が王手（`gave_check_last`）
    let mut opp_fouls_pending = 0u32;
    let mut opp_fouls_last = 0u32;
    let (mut opp_fouls_checked, mut opp_fouls_free) = (0.0f32, 0.0f32);
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
                // 相手の手番がここで閉じる
                let w = opp_fouls_pending as f32 * fresh(*move_number);
                if gave_check_last {
                    opp_fouls_checked += w;
                } else {
                    opp_fouls_free += w;
                }
                opp_fouls_last = opp_fouls_pending;
                opp_fouls_pending = 0;
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
            // 相手のまだ閉じていない手番の反則（手番側の観測を書くときは無い）は数えない
            Observation::OpponentFoul { .. } => opp_fouls_pending += 1,
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
    let mut hold = [0u32; 7];
    for (i, role) in HAND_ROLES.iter().enumerate() {
        let n = view.your_hand.get(role).copied().unwrap_or(0);
        let total = ROLE_TOTAL[i] as f32;
        p.fill(P_HAND + i, n as f32 / total);
        hold[i] = ROLE_TOTAL[i].saturating_sub(mine[i] + n);
        p.fill(P_OPP_HOLD + i, hold[i] as f32 / total);
    }
    let history_broken = !history_consistent(&model, view);
    let max_fouls = MAX_FOULS as f32;

    if v3 {
        // 相手の持ち駒の上界。相手の持ち駒は取った自駒からしか増えないので、取られた自駒の駒種別の数
        // （成りは戻す）が上界になる。相手の打ちは見えないので減らせない（上界は増える一方）。
        // 相手の保有数（盤上＋持ち駒）でも抑える。着手履歴が欠けているときは取られた駒を数え漏らしうるので、
        // 保有数だけを使う（これも上界）
        let mut lost = [0u32; 7];
        for &(_, role) in model.lost_pieces() {
            if let Some(i) = hand_index(unpromote_role(role)) {
                lost[i] += 1;
            }
        }
        for i in 0..7 {
            let bound = if history_broken { hold[i] } else { lost[i].min(hold[i]) };
            p.fill(E_HAND + i, bound as f32 / ROLE_TOTAL[i] as f32);
        }
        // 相手の反則の時系列（鮮度は相手の手番を閉じた `OpponentMoved` の手数で測る。P_LOST と同じ）
        p.fill(E_FOUL + EF_LAST, opp_fouls_last as f32 / max_fouls);
        p.fill(E_FOUL + EF_CHECKED, opp_fouls_checked / max_fouls);
        p.fill(E_FOUL + EF_FREE, opp_fouls_free / max_fouls);
    }

    // スカラー
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

/// 確定事実のプレーン（v2 の後ろ5枚）を書く。`out` は 0 で埋めてあること
fn encode_deduced(
    view: &PlayerView,
    log: &ObservationLog,
    foul_tried: &HashSet<String>,
    out: &mut [f32],
    cache: &mut DeducedCache,
) {
    let color = view.your_color;
    let facts = deduced_facts(view, log, foul_tried);
    let mut p = Planes { buf: out, color };
    for &c in &facts.occupied {
        p.set(D_OCC, c, 1.0);
    }
    for seg in &facts.blocked {
        let v = 1.0 / seg.len() as f32;
        for &c in seg {
            p.max(D_BLOCK, c, v);
        }
    }
    if !facts.checkers.is_empty() {
        let v = 1.0 / facts.checkers.len() as f32;
        for &c in &facts.checkers {
            p.set(D_CHECKER, c, v);
        }
    }
    // deduce::opp_king_candidates / immobile_opponent_pawns と同じ（途中状態を使い回すだけ）
    let (king, pawns) = cache.advance(color, log);
    let locked = pawns.result();
    for c in king.candidates(log, &locked) {
        p.set(D_KING, c, 1.0);
    }
    for c in locked {
        p.set(D_PAWN, c, 1.0);
    }
}

/// v1 の観測（長さ `OBS_LEN`）
pub fn encode(view: &PlayerView, log: &ObservationLog, foul_tried: &HashSet<String>) -> Vec<f32> {
    encode_planes(NUM_PLANES, view, log, foul_tried)
}

/// プレーン数 `planes`（`NUM_PLANES` / `NUM_PLANES_DEDUCED` / `NUM_PLANES_V3`）の観測。重みのヘッダのプレーン数を渡す
pub fn encode_planes(
    planes: usize,
    view: &PlayerView,
    log: &ObservationLog,
    foul_tried: &HashSet<String>,
) -> Vec<f32> {
    let len = obs_len(planes).unwrap_or_else(|| panic!("プレーン数 {planes} に対応するエンコーダが無い"));
    let mut out = vec![0.0; len];
    encode_into(view, log, foul_tried, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deduce::immobile_opponent_pawns;
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

    /// 確定事実の途中状態を使い回しても（`encode_into_cached`）、毎回たどり直すのと同じ観測になる。
    /// 両方の側を毎手番書く（手番でない側のログは自分の手で終わる = 王手宣言を待って保留する末尾の
    /// 自分の手も通る）。1つの途中状態を次の局へ持ち越すと、ログが短くなるので作り直される
    #[test]
    fn 確定事実の途中状態を使い回しても同じ観測になる() {
        let mut rng = StdRng::seed_from_u64(12);
        let mut caches = [DeducedCache::default(), DeducedCache::default()];
        // v3 は v2 を含むので、確定事実の途中状態も v3 の足したプレーンもまとめて検査する
        let mut fresh = vec![0.0; NUM_PLANES_V3 * 81];
        let mut cached = vec![0.0; NUM_PLANES_V3 * 81];
        let mut n = 0;
        for game in 0..12 {
            let mut referee = Referee::new();
            loop {
                let side = referee.to_move();
                for (i, s) in [Color::Sente, Color::Gote].into_iter().enumerate() {
                    let view = referee.view(s, [0, 0], game);
                    let (log, tried) = (referee.log(s), referee.foul_tried(s));
                    encode_into(&view, log, tried, &mut fresh);
                    encode_into_cached(&view, log, tried, &mut cached, &mut caches[i]);
                    assert_eq!(fresh, cached, "game {game} {s:?} ログ長 {}", log.events().len());
                    n += 1;
                }
                let view = referee.view(side, [0, 0], game);
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
        assert!(n > 1000, "局面が少なすぎる: {n}");
    }

    /// 1決定点あたりのエンコード＋マスクの所要時間（自己対局の速度の上限を決める）。
    /// `cargo test --release --lib エンコードの速度 -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn エンコードの速度() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut n = 0u32;
        let mut elapsed = std::time::Duration::ZERO;
        let mut elapsed_v2 = std::time::Duration::ZERO;
        let mut elapsed_v2_cached = std::time::Duration::ZERO;
        let mut elapsed_v3_cached = std::time::Duration::ZERO;
        let mut buf = vec![0.0; OBS_LEN];
        let mut buf_v2 = vec![0.0; NUM_PLANES_DEDUCED * 81];
        let mut buf_v3 = vec![0.0; NUM_PLANES_V3 * 81];
        for game in 0..20 {
            let mut referee = Referee::new();
            let mut caches = [DeducedCache::default(), DeducedCache::default()];
            let mut caches_v3 = [DeducedCache::default(), DeducedCache::default()];
            loop {
                let side = referee.to_move();
                let view = referee.view(side, [0, 0], game);
                let t = std::time::Instant::now();
                encode_into(&view, referee.log(side), referee.foul_tried(side), &mut buf);
                let mask = legal_mask(&view, referee.log(side), referee.foul_tried(side));
                elapsed += t.elapsed();
                let t = std::time::Instant::now();
                encode_into(&view, referee.log(side), referee.foul_tried(side), &mut buf_v2);
                elapsed_v2 += t.elapsed();
                let cache = &mut caches[(side == Color::Gote) as usize];
                let t = std::time::Instant::now();
                encode_into_cached(&view, referee.log(side), referee.foul_tried(side), &mut buf_v2, cache);
                elapsed_v2_cached += t.elapsed();
                let cache = &mut caches_v3[(side == Color::Gote) as usize];
                let t = std::time::Instant::now();
                encode_into_cached(&view, referee.log(side), referee.foul_tried(side), &mut buf_v3, cache);
                elapsed_v3_cached += t.elapsed();
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
            "{n} 決定点、v1 のエンコード＋マスク 平均 {:.1}µs/決定点、v2 のエンコード 平均 {:.1}µs/決定点\
             （途中状態を使い回すと {:.1}µs/決定点。v3 は使い回して {:.1}µs/決定点）",
            elapsed.as_secs_f64() * 1e6 / n as f64,
            elapsed_v2.as_secs_f64() * 1e6 / n as f64,
            elapsed_v2_cached.as_secs_f64() * 1e6 / n as f64,
            elapsed_v3_cached.as_secs_f64() * 1e6 / n as f64
        );
    }

    /// エンコーダ v2 の前半は v1 と同じで、確定事実のプレーンは真の盤面と矛盾しない
    /// （相手の駒がいると確定したマスには相手の駒がいる・飛び越え反則の間のどれかに相手の駒がいる・
    /// 王手駒の候補に本当の王手駒が入る・相手玉の候補に本当の相手玉が入る・動けない歩は本当に相手の歩）
    #[test]
    fn 確定事実のプレーンは真の盤面と矛盾しない() {
        let mut rng = StdRng::seed_from_u64(11);
        let (mut n, mut n_occ, mut n_block, mut n_check, mut n_pawn) = (0, 0, 0, 0, 0);
        for game in 0..30 {
            let mut referee = Referee::new();
            loop {
                let side = referee.to_move();
                let view = referee.view(side, [0, 0], game);
                let (log, tried) = (referee.log(side), referee.foul_tried(side));
                let v1 = encode(&view, log, tried);
                let v2 = encode_planes(NUM_PLANES_DEDUCED, &view, log, tried);
                assert_eq!(&v2[..OBS_LEN], &v1[..], "v2 の前半は v1 と同じ");
                assert!(v2.iter().all(|v| v.is_finite() && (0.0..=3.0).contains(v)));
                let pos = referee.position();
                let opp = |c: Coord| pos.piece_at(c).filter(|p| p.color != side);
                let at = |k: usize, c: Coord| v2[k * 81 + square_index(normalize(c, side))];
                let facts = deduced_facts(&view, log, tried);
                for &c in &facts.occupied {
                    assert!(opp(c).is_some(), "game {game}: 確定した占有 {c:?} に相手の駒がいない");
                    assert_eq!(at(D_OCC, c), 1.0);
                    n_occ += 1;
                }
                for seg in &facts.blocked {
                    assert!(seg.iter().any(|&c| opp(c).is_some()), "game {game}: 遮りの間 {seg:?} が空");
                    n_block += 1;
                }
                if view.you_in_check && !facts.checkers.is_empty() {
                    let king = pos.king_square(side).unwrap();
                    let real: Vec<Coord> = pos
                        .pieces()
                        .filter(|(c, p)| p.color != side && pos.attacks(*c, king))
                        .map(|(c, _)| c)
                        .collect();
                    assert!(
                        real.iter().any(|c| facts.checkers.contains(c)),
                        "game {game}: 王手駒 {real:?} が候補 {:?} に無い",
                        facts.checkers
                    );
                    assert!(real.iter().any(|&c| at(D_CHECKER, c) > 0.0));
                    n_check += 1;
                }
                let opp_king = pos.king_square(side.other()).unwrap();
                assert_eq!(at(D_KING, opp_king), 1.0, "game {game}: 相手玉のマスが候補に無い");
                for c in immobile_opponent_pawns(side, log) {
                    assert_eq!(opp(c).map(|p| p.role), Some(Role::Pawn), "game {game}: 動けない歩 {c:?}");
                    assert_eq!(at(D_PAWN, c), 1.0);
                    n_pawn += 1;
                }
                n += 1;
                let mask = legal_mask(&view, log, tried);
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
        println!("{n} 決定点: 占有 {n_occ} / 遮り {n_block} / 王手 {n_check} / 動けない歩 {n_pawn}");
        assert!(n_occ > 0 && n_block > 0 && n_check > 0, "検査が空振りしている");
    }

    /// エンコーダ v3 の前半は v2 と同じで、足した2組は審判の真実から数え直した値と一致する:
    /// - 相手の持ち駒の上界 = min(相手が取った自駒の数, 相手の保有数)。取った数は「相手の本当の持ち駒＋相手が打った数」
    ///   なので、上界は本当の持ち駒を下回らない
    /// - 相手の反則の時系列 = 相手の手番ごとの反則数（王手を受けていたかで分ける。鮮度は手番を閉じた手数で測る）
    #[test]
    fn v3の足したプレーンは真実から数え直した値と一致する() {
        #[derive(Default)]
        struct Turn {
            in_check: bool,
            fouls: u32,
        }
        let mut rng = StdRng::seed_from_u64(13);
        let (mut n, mut n_hand, mut n_tight, mut n_fouls, mut n_checked) = (0, 0, 0, 0, 0);
        for game in 0..12 {
            let mut referee = Referee::new();
            // 確定事実はログを先頭からたどり直すと長い局ほど重いので、途中状態を使い回す（結果は同じ）
            let mut caches = [DeducedCache::default(), DeducedCache::default()];
            // 側ごとの閉じた手番（反則数と、手番の頭で王手を受けていたか。手番を閉じた手数つき）と、打った駒の数
            let mut turns: [Vec<(Turn, u32)>; 2] = [vec![], vec![]];
            let mut dropped = [[0u32; 7]; 2];
            let mut cur = Turn::default();
            loop {
                let side = referee.to_move();
                let (me, opp) = (side as usize, side.other() as usize);
                let view = referee.view(side, [0, 0], game);
                let (log, tried) = (referee.log(side), referee.foul_tried(side));
                if tried.is_empty() {
                    cur = Turn { in_check: referee.position().in_check(side), fouls: 0 };
                }
                let mut v2 = vec![0.0; NUM_PLANES_DEDUCED * 81];
                let mut v3 = vec![0.0; NUM_PLANES_V3 * 81];
                encode_into_cached(&view, log, tried, &mut v2, &mut caches[me]);
                encode_into_cached(&view, log, tried, &mut v3, &mut caches[me]);
                assert_eq!(&v3[..v2.len()], &v2[..], "v3 の前半は v2 と同じ");
                assert!(v3.iter().all(|v| v.is_finite() && (0.0..=3.0).contains(v)));
                let pos = referee.position();
                for (i, &role) in HAND_ROLES.iter().enumerate() {
                    let got = plane(&v3, E_HAND + i)[0] * ROLE_TOTAL[i] as f32;
                    assert!(plane(&v3, E_HAND + i).iter().all(|&v| v == plane(&v3, E_HAND + i)[0]), "全面に放送");
                    let hand = pos.hand_count(side.other(), role) as u32;
                    let hold = hand
                        + pos
                            .pieces()
                            .filter(|(_, p)| p.color != side && unpromote_role(p.role) == role)
                            .count() as u32;
                    let want = (hand + dropped[opp][i]).min(hold);
                    assert!(
                        (got - want as f32).abs() < 1e-4,
                        "game {game} {side:?} {role:?}: 上界 {got} / 真実から {want}（持ち駒 {hand}、打った {}）",
                        dropped[opp][i]
                    );
                    assert!(got + 1e-4 >= hand as f32, "上界が本当の持ち駒を下回る");
                    n_tight += (want == hand) as u32;
                    n_hand += 1;
                }
                let now = view.move_number;
                let fresh = |mn: u32| DECAY.powi(now.saturating_sub(mn) as i32);
                let last = turns[opp].last().map_or(0, |(t, _)| t.fouls);
                let sum = |checked: bool| -> f32 {
                    turns[opp]
                        .iter()
                        .filter(|(t, _)| t.in_check == checked)
                        .map(|(t, mn)| t.fouls as f32 * fresh(*mn))
                        .sum()
                };
                let foul = |k: usize| plane(&v3, E_FOUL + k)[0] * MAX_FOULS as f32;
                assert!((foul(EF_LAST) - last as f32).abs() < 1e-4, "game {game}: 直前の手番の反則");
                assert!((foul(EF_CHECKED) - sum(true)).abs() < 1e-4, "game {game}: 王手を受けていた手番の反則");
                assert!((foul(EF_FREE) - sum(false)).abs() < 1e-4, "game {game}: 王手を受けていない手番の反則");
                n_fouls += (last > 0) as u32;
                n_checked += (sum(true) > 0.0) as u32;
                n += 1;

                let mask = legal_mask(&view, log, tried);
                let actions: Vec<usize> = (0..NUM_ACTIONS).filter(|&a| mask[a]).collect();
                if actions.is_empty() {
                    break;
                }
                let usi = decode_usi(actions[rng.random_range(0..actions.len())], side).unwrap();
                match referee.step(&usi, 0) {
                    StepResult::Ended { .. } => break,
                    StepResult::Foul => cur.fouls += 1,
                    _ => {
                        if let Some(ShogiMove::Drop { role, .. }) = parse_usi(&usi) {
                            dropped[me][hand_index(role).unwrap()] += 1;
                        }
                        let mn = referee.position().move_number();
                        turns[me].push((std::mem::take(&mut cur), mn));
                    }
                }
            }
        }
        println!(
            "{n} 決定点: 上界が本当の持ち駒と一致 {n_tight}/{n_hand}、直前の手番に相手の反則 {n_fouls}、\
             王手を受けた手番の反則あり {n_checked}"
        );
        assert!(n_fouls > 0 && n_checked > 0 && n_tight < n_hand, "検査が空振りしている");
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
