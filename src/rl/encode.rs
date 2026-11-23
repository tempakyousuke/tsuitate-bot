//! 観測 → 入力テンソル（docs/rl-deepnash-design.md の「2. 観測エンコード」）。
//!
//! 入力は `Strategy::choose` と同じ `(PlayerView, ObservationLog, foul_tried)` だけ。
//! 学習中の環境も arena・本番の方策 Strategy もこの関数を通るので、学習時と推論時で
//! 特徴量が食い違わない。observation.rs にない情報（相手駒の位置など）は受け取れない。
//!
//! 出力は `[NUM_PLANES, 9, 9]` を平らにした f32 列（プレーン優先、マス番号は
//! `action::square_index`）。盤は手番側を先手向きに正規化する。

use std::collections::HashSet;

use crate::board::{Coord, on_board, orient, parse_usi_square, rays, steps};
use crate::deduce::{ImmobilePawns, KingCandidates};
use crate::deduce_king::KingCandidatesTight;
use crate::model::GameModel;
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{Color, PlayerView, Role};
use crate::rl::action::{checker_squares_at, deduced_facts, normalize, square_index};
use crate::selfplay::{MAX_FOULS, MAX_PLIES};
use crate::shogi::{HAND_ROLES, Piece, Position, ShogiMove, hand_index, parse_usi, unpromote_role};

/// 自駒の駒種（`Role` の宣言順）
pub(crate) const ROLES: [Role; 14] = [
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
pub(crate) const ROLE_TOTAL: [u32; 7] = [18, 4, 4, 4, 4, 2, 2];

// プレーンの配置
pub(crate) const P_OWN: usize = 0; // 14
pub(crate) const P_HIST: usize = P_OWN + 14; // 2 × HISTORY_MOVES（移動元, 移動先）
pub(crate) const P_CAP: usize = P_HIST + 2 * HISTORY_MOVES; // 取ったマスの鮮度
pub(crate) const P_CAP_ROLE: usize = P_CAP + 1; // 7: 取った駒種ごとの鮮度
pub(crate) const P_LOST: usize = P_CAP_ROLE + 7; // 取られたマスの鮮度
pub(crate) const P_LOST_COUNT: usize = P_LOST + 1; // 取られた回数
// 反則は「今の手番」と「過去の手番」を同じ形で持つ（1組 = 移動元・移動先・成りの移動先・
// 打ち先×駒種7）。打ちの駒種と成りを分けるのは、同じマスでも反則の原因（打ち歩詰め・
// 成れない位置など）が違い、隠れた盤面への証拠が別物になるため
pub(crate) const FOUL_GROUP: usize = 3 + 7;
pub(crate) const P_FOUL: usize = P_LOST_COUNT + 1; // 今の手番の反則（値は 1）
pub(crate) const P_PAST_FOUL: usize = P_FOUL + FOUL_GROUP; // 過去の手番の反則（値は鮮度）
pub(crate) const F_FROM: usize = 0;
pub(crate) const F_TO: usize = 1;
pub(crate) const F_PROMO_TO: usize = 2;
pub(crate) const F_DROP: usize = 3;
pub(crate) const P_CHECKED_KING: usize = P_PAST_FOUL + FOUL_GROUP; // 王手されたときの自玉の位置（鮮度）
pub(crate) const P_GAVE_CHECK: usize = P_CHECKED_KING + 1; // 王手をかけた手の移動先（鮮度）
pub(crate) const P_UNMOVED: usize = P_GAVE_CHECK + 1; // 初期配置から一度も動いていない自駒
pub(crate) const P_HAND: usize = P_UNMOVED + 1; // 7: 自分の持ち駒（全面に放送）
pub(crate) const P_OPP_HOLD: usize = P_HAND + 7; // 7: 相手の駒種別保有数（盤上＋持ち駒。自分側から確定）
pub(crate) const P_SCALAR: usize = P_OPP_HOLD + 7; // 以下のスカラー
pub(crate) const S_MY_FOULS: usize = 0;
pub(crate) const S_OPP_FOULS: usize = 1;
pub(crate) const S_MY_LAST_FOUL: usize = 2; // あと1回で反則負け
pub(crate) const S_OPP_LAST_FOUL: usize = 3;
pub(crate) const S_FOULS_THIS_TURN: usize = 4;
pub(crate) const S_IN_CHECK: usize = 5;
pub(crate) const S_GAVE_CHECK_LAST: usize = 6; // 直前の自分の手で王手をかけた
pub(crate) const S_MOVE_NUMBER: usize = 7;
/// 観測ログから再構成した自駒・持ち駒が視界と食い違う（再起動で履歴を失った等）。
/// 履歴由来の特徴（鮮度・未着手駒・王手履歴）が信用できないことをネットへ知らせる
pub(crate) const S_HISTORY_BROKEN: usize = 8;
pub(crate) const NUM_SCALARS: usize = 9;

pub const NUM_PLANES: usize = P_SCALAR + NUM_SCALARS;
pub const OBS_LEN: usize = NUM_PLANES * 81;

// 確定事実のプレーン（エンコーダ v2、2026-10-16。docs/rl-deepnash-design.md の改善案2）。
// v1（`NUM_PLANES`）の後ろに足すだけなので、v1 の重みの stem に 0 の列を足せば同じ関数から始められる。
// どれも観測から論理的に確定する事実（真の盤面と矛盾しない）で、`action::deduced_facts` と `deduce` が作る
pub(crate) const D_OCC: usize = NUM_PLANES; // 相手の駒がいると確定したマス
pub(crate) const D_BLOCK: usize = D_OCC + 1; // 飛び越え反則の間のマス（どれかに相手の駒がいる。値は 1/マス数）
pub(crate) const D_CHECKER: usize = D_BLOCK + 1; // 王手中: 王手駒がいうるマス（値は 1/仮説数）
pub(crate) const D_KING: usize = D_CHECKER + 1; // 相手玉がいうるマス（deduce::opp_king_candidates）
pub(crate) const D_PAWN: usize = D_KING + 1; // 動けないと確定した相手の歩（deduce::immobile_opponent_pawns）
/// エンコーダ v2 のプレーン数（v1 + 確定事実 5枚）
pub const NUM_PLANES_DEDUCED: usize = D_PAWN + 1;

// エンコーダ v3（2026-11-08。docs/rl-deepnash-design.md の「エンコーダ v3」）。v2 の後ろに、観測と自分側の
// 情報だけで決まる事実を2組足す。**並びは固定**で、学習でどの組を使うかはネットの側が持つ（tsuitate-nn の
// `--opp-hand-planes` / `--opp-foul-planes`。使わない組は stem の重みの列が 0 に固定されるので、ここで書いた値は
// 出力に効かない）。並びを固定にしておくと、組の選び方が違うネットどうし（錨・リーグ）が同じ観測を読める
pub(crate) const E_HAND: usize = NUM_PLANES_DEDUCED; // 7: 相手の持ち駒の上界（駒種別。全面に放送）
pub(crate) const E_FOUL: usize = E_HAND + 7; // 3: 相手の反則の時系列（全面に放送）
pub(crate) const EF_LAST: usize = 0; // 相手の直前の手番の反則数
pub(crate) const EF_CHECKED: usize = 1; // 王手を受けていた相手の手番の反則数（鮮度つきの合計）
pub(crate) const EF_FREE: usize = 2; // 王手を受けていない相手の手番の反則数（鮮度つきの合計）
/// エンコーダ v3 のプレーン数（v2 + 相手の持ち駒の上界 7枚 + 相手の反則の時系列 3枚）
pub const NUM_PLANES_V3: usize = E_FOUL + 3;
// エンコーダ v4（2026-11-08。設計 doc の「エンコーダ v3」の節の v4）。v3 の後ろに1枚足すだけなので、v3 の重みは
// v4 の観測の先頭を読めばよい（v3 の実験の checkpoint をそのまま使える）。組の扱いは v3 と同じ（tsuitate-nn の
// `--king-foul-planes`）
pub(crate) const E_KING_FOUL: usize = NUM_PLANES_V3; // 1: 自玉の移動で反則したマス（鮮度は反則した手数から）
/// エンコーダ v4 のプレーン数（v3 + 玉の移動で反則したマス 1枚）
pub const NUM_PLANES_V4: usize = E_KING_FOUL + 1;
// エンコーダ v5（2026-11-17。設計 doc の「エンコーダ v3」の節の v5）。v4 の後ろに「取った駒・取られた駒の一覧」を4組。
// v1 の取った駒のプレーン（`P_CAP_ROLE`）は1手ごとに薄れ、同じマスの2回目で上書きされる。取られた駒は場所だけで
// 駒種が無い（`P_LOST`）。ここでは薄れない回数と、取られた駒種・歩の切れた筋を持つ。組の扱いは v3 / v4 と同じ
// （並びは固定で、使う組はネットが選ぶ。tsuitate-nn の `--lost-fresh-planes` / `--lost-role-planes` /
// `--cap-role-planes` / `--pawn-file-planes`）
pub(crate) const E_LOST_FRESH: usize = NUM_PLANES_V4; // 7: 取られた駒種ごとの、最後に取られた手数からの鮮度（全面に放送）
pub(crate) const E_LOST_ROLE: usize = E_LOST_FRESH + 7; // 7: 取られた駒種ごとのマス（回数。薄れない）
pub(crate) const E_CAP_ROLE: usize = E_LOST_ROLE + 7; // 7: 取った駒種ごとのマス（回数。薄れない）
pub(crate) const E_PAWN_FILE: usize = E_CAP_ROLE + 7; // 1: 相手の歩を取った筋（筋の全マスに回数。薄れない）
/// 回数のプレーンの1回ぶん（3回で 1 に張り付く。`P_LOST_COUNT` と同じ）
const COUNT_STEP: f32 = 1.0 / 3.0;
/// エンコーダ v5 のプレーン数（v4 + 取られた駒種の鮮度 7枚 + 取られた駒種のマス 7枚 + 取った駒種のマス 7枚 + 歩を取った筋 1枚）
pub const NUM_PLANES_V5: usize = E_PAWN_FILE + 1;
// エンコーダ v6（2026-11-23。設計 doc の「エンコーダ v3」の節の v6）。v5 の後ろに、安く作れる確定事実と集計を7組。
// 1組ずつ効きを確かめるのは高くつく（本番の大きさで3日、CI の大きさでは v5 の差も再現しなかった。
// docs/rl-input-ab.md）ので、まとめて足して版ごとに1回確かめる。組の扱いは v3〜v5 と同じ（並びは固定で、使う組は
// ネットが選ぶ。tsuitate-nn の `--king-tight-planes` / `--check-clock-planes` / `--checker-memo-planes` /
// `--own-attack-planes` / `--king-line-planes` / `--opp-count-planes` / `--exposed-planes`）
pub(crate) const E_KING_TIGHT: usize = NUM_PLANES_V5; // 1: 相手玉がいうるマス（締めた版。deduce_king::opp_king_candidates_tight）
pub(crate) const E_CHECK_CLOCK: usize = E_KING_TIGHT + 1; // 6: 王手からの手数（全面に放送）
pub(crate) const CC_GAVE_EVER: usize = 0; // 王手をかけたことがある
pub(crate) const CC_GAVE_MOVES: usize = 1; // 最後にかけてからの相手の手数（かけていなければ初手から）
pub(crate) const CC_GAVE_FREE: usize = 2; // そのうち自由な手（王手への応手・駒取り・王手のどれでもない手）
pub(crate) const CC_GOT_EVER: usize = 3; // 王手を受けたことがある
pub(crate) const CC_GOT_MOVES: usize = 4; // 最後に受けてからの相手の手数（受けていなければ初手から）
pub(crate) const CC_GOT_FREE: usize = 5; // そのうち自由な手
/// 王手からの手数の上限（この手数で 1 に張り付く）
pub(crate) const CLOCK_CAP: u32 = 12;
pub(crate) const E_CHECKER_MEMO: usize = E_CHECK_CLOCK + 6; // 1: 取らずに逃げた王手駒がいたかもしれないマス（鮮度 ÷ 候補数）
pub(crate) const E_OWN_ATTACK: usize = E_CHECKER_MEMO + 1; // 2: 自駒の利き（確実な利きの枚数, 遮られうる利きの枚数）
pub(crate) const E_KING_LINE: usize = E_OWN_ATTACK + 2; // 2: 自玉の筋（最初の自駒の手前まで, ピンされうる自駒）
pub(crate) const E_OPP_COUNT: usize = E_KING_LINE + 2; // 4: 相手の手数の合計（全面に放送）
pub(crate) const OC_CHECKS: usize = 0; // 相手が王手をかけてきた手
pub(crate) const OC_CAPTURES: usize = 1; // 相手が駒を取ってきた手
pub(crate) const OC_BOTH: usize = 2; // 相手が王手をかけながら駒を取ってきた手
pub(crate) const OC_MOVES: usize = 3; // 相手が指した手
/// 相手の手数の合計の上限（この回数で 1 に張り付く。並びは `OC_*`）。本番 705局と自己対局 1,200局の手番ごとの分布から、
/// 張り付くのが 1% に満たない値にした（王手 99% 点 10〜15・最大 37、駒取り 99% 点 16〜20・最大 33、
/// 王手かつ駒取り 99% 点 3〜5・最大 15。指した手は手数の上限の半分）
pub(crate) const OPP_COUNT_CAPS: [u32; 4] = [20, 30, 10, MAX_PLIES / 2];
// 相手に位置が伝わっている自駒。相手は「自分の駒を取られたマス」と「王手を受けたこと」を審判から知らされるので、
// 駒を取った自駒はそのマスにいると分かり、王手をかけた手で動かした駒も王手の筋から当たりを付けられる。その駒が
// 動くか取られるまで印を残す。値はその手からの鮮度（0.95^手数。`P_CAP` / `P_GAVE_CHECK` の同じマスの値と同じ）
pub(crate) const E_EXPOSED: usize = E_OPP_COUNT + 4; // 2: 駒を取ったまま動いていない自駒, 王手をかけた手で動かしたまま動いていない自駒
/// エンコーダ v6 のプレーン数（v5 + 締めた相手玉の候補 1枚 + 王手からの手数 6枚 + 逃げた王手駒の候補 1枚 +
/// 自駒の利き 2枚 + 自玉の筋 2枚 + 相手の手数の合計 4枚 + 相手に位置が伝わっている自駒 2枚）
pub const NUM_PLANES_V6: usize = E_EXPOSED + 2;
/// v3 以降で足したプレーンの組（名前, 先頭のプレーン, 枚数）。tsuitate-nn の `model.EXTRA_PLANES` と同じ
pub const EXTRA_PLANE_GROUPS: [(&str, usize, usize); 14] = [
    ("opp_hand", E_HAND, 7),
    ("opp_fouls", E_FOUL, 3),
    ("king_fouls", E_KING_FOUL, 1),
    ("lost_fresh", E_LOST_FRESH, 7),
    ("lost_roles", E_LOST_ROLE, 7),
    ("cap_roles", E_CAP_ROLE, 7),
    ("pawn_files", E_PAWN_FILE, 1),
    ("king_tight", E_KING_TIGHT, 1),
    ("check_clock", E_CHECK_CLOCK, 6),
    ("checker_memo", E_CHECKER_MEMO, 1),
    ("own_attacks", E_OWN_ATTACK, 2),
    ("king_lines", E_KING_LINE, 2),
    ("opp_counts", E_OPP_COUNT, 4),
    ("exposed", E_EXPOSED, 2),
];

/// プレーン数 → 観測の長さ。対応するエンコーダが無いプレーン数は None
pub fn obs_len(planes: usize) -> Option<usize> {
    [NUM_PLANES, NUM_PLANES_DEDUCED, NUM_PLANES_V3, NUM_PLANES_V4, NUM_PLANES_V5, NUM_PLANES_V6]
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

/// 受理手（自分の着手・相手の着手）が1つも欠けずに、今の手数 `now` まで並んでいるか。手数は初期局面で 1、受理の
/// たびに +1 なので、欠けていなければ受理手の数は `now − 1`。受理手に付く手数は、記録する側で規約が2つある:
/// 審判・サーバー（Socket.IO 版）は適用後の値（初手が 2、最後が `now`）、webhook のセッション
/// （`webhook_session.rs`）は適用前の値（初手が 1、最後が `now − 1`）。どちらの規約でも、手数が 1 ずつ増えていて
/// 数が合えば欠けていない（片方の規約だけを受け付けると、もう片方では常に欠けている扱いになる）
pub(crate) fn moves_contiguous(log: &ObservationLog, now: u32) -> bool {
    let mut prev: Option<u32> = None;
    let mut count = 0u32;
    for event in log.events() {
        if let Observation::MyMove { move_number, .. } | Observation::OpponentMoved { move_number, .. } = event {
            if prev.is_some_and(|p| *move_number != p + 1) {
                return false;
            }
            prev = Some(*move_number);
            count += 1;
        }
    }
    count + 1 == now
        && match prev {
            None => true,
            Some(last) => last == now || last + 1 == now,
        }
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
    /// 締めた相手玉の候補（v6 の観測を書くときだけ作る。`state` を作り直すときに一緒に捨てる）
    tight: Option<KingCandidatesTight>,
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
            self.tight = None;
        }
        let (_, king, pawns) = self.state.as_mut().expect("作った");
        king.advance(log);
        pawns.advance(log);
        (king, pawns)
    }

    /// `log` の時点の、締めた相手玉の候補（`deduce_king::opp_king_candidates_tight` と同じ。動けない歩のマスも除く）
    fn tight_candidates(&mut self, color: Color, log: &ObservationLog) -> std::collections::BTreeSet<Coord> {
        let locked = self.advance(color, log).1.result();
        let tight = self.tight.get_or_insert_with(|| KingCandidatesTight::new(color));
        tight.advance(log);
        tight.candidates(log, &locked)
    }
}

/// from から to への直線（縦横斜め）の、両端を除く途中に x があるか
fn on_path(from: Coord, to: Coord, x: Coord) -> bool {
    let (df, dr) = (to.file - from.file, to.rank - from.rank);
    if !(df == 0 || dr == 0 || df.abs() == dr.abs()) {
        return false;
    }
    let (sf, sr) = (df.signum(), dr.signum());
    let mut c = Coord { file: from.file + sf, rank: from.rank + sr };
    while c != to {
        if c == x {
            return true;
        }
        c = Coord { file: c.file + sf, rank: c.rank + sr };
    }
    false
}

/// 縦横斜めの8方向（筋, 段）
const LINE_DIRS: [(i8, i8); 8] = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1)];

/// 観測をテンソルへ書く。`out` の長さでエンコーダの版を選ぶ: `OBS_LEN` は v1、
/// `NUM_PLANES_DEDUCED * 81` は v2（v1 の後ろに確定事実のプレーン）、`NUM_PLANES_V3 * 81` は v3
/// （v2 の後ろに相手の持ち駒の上界と相手の反則の時系列）、`NUM_PLANES_V4 * 81` は v4（v3 の後ろに玉の移動で反則したマス）、
/// `NUM_PLANES_V5 * 81` は v5（v4 の後ろに取った駒・取られた駒の一覧）、`NUM_PLANES_V6 * 81` は v6
/// （v5 の後ろに、締めた相手玉の候補・王手からの手数・逃げた王手駒の候補・自駒の利き・自玉の筋・相手の手数の合計・
/// 相手に位置が伝わっている自駒）
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
    let v3 = out.len() >= NUM_PLANES_V3 * 81;
    let v4 = out.len() >= NUM_PLANES_V4 * 81;
    let v5 = out.len() >= NUM_PLANES_V5 * 81;
    let v6 = out.len() == NUM_PLANES_V6 * 81;
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
    // 自玉の移動で反則したマス（v4）→ 最後に反則した手数。玉の手が反則になるのは「移動先に相手の利きがある」
    // ときだけ（自駒のマスへの手は候補に無い）なので、反則した時点でそのマスに相手の利きがあったことが確定する。
    // その後に玉がそのマスへ移動できたら（受理されたら）消す
    let mut king_sq = Position::initial().king_square(color);
    let mut king_fouls: Vec<(Coord, u32)> = vec![];
    // 取られた駒種ごとの、最後に取られた手数（v5）
    let mut lost_mn: [Option<u32>; 7] = [None; 7];
    // 王手からの手数（v6）。「自由な手」= 王手への応手でも駒取りでも王手でもない相手の手（行き先の手掛かりが無い）。
    // 王手かどうかは直後の `Check` で分かるので、いったん数えて王手だったら戻す
    let (mut gave_ever, mut got_ever) = (false, false);
    let (mut since_gave, mut free_gave, mut since_got, mut free_got) = (0u32, 0u32, 0u32, 0u32);
    let mut last_opp_free = false;
    // 逃げた王手駒の候補（v6）。`pending_check` = いま受けている王手の候補と手数（解消するまで）、
    // `checker_memo` = 最後に解消した王手の（残っている候補, 王手の時点の候補数, 手数）
    let mut last_opp_capture: Option<Coord> = None;
    let mut pending_check: Option<(Vec<Coord>, u32)> = None;
    let mut checker_memo: Option<(Vec<Coord>, usize, u32)> = None;
    // 相手の手数の合計（v6。並びは `OC_*`）。王手かどうかは直後の `Check` で分かる
    let mut opp_counts = [0u32; 4];
    // 相手に位置が伝わっている自駒（v6）: [駒を取った, 王手をかけた手で動かした] ごとの（マス, その手の手数）。
    // その駒が動いたら・取られたら外す
    let mut exposed: [Vec<(Coord, u32)>; 2] = Default::default();
    for event in log.events() {
        let lost_before = model.lost_pieces().len();
        match event {
            Observation::MyMove {
                move_number,
                usi,
                captured,
            } => {
                last_mn = *move_number;
                my_moves.push(usi);
                gave_check_last = false;
                if let (true, Some((from, to))) = (v6, move_squares(usi)) {
                    // 王手を受けていた手番の受理手 = 王手を解消した。玉以外の駒で候補のマスの駒を取って解消したなら、
                    // 取ったのが王手駒（両王手は駒取りでは解消しない）なので何も残さない
                    if let Some((cands, mn)) = pending_check.take() {
                        let by_king = from.is_some() && from == king_sq;
                        let took_checker = captured.is_some() && !by_king && cands.contains(&to);
                        checker_memo = (!took_checker).then(|| {
                            let n = cands.len();
                            (cands, n, mn)
                        });
                    }
                    // 自駒が着地した・通り抜けたマスに、いまその駒はいない
                    if let Some((cands, _, _)) = &mut checker_memo {
                        cands.retain(|&c| c != to && !from.is_some_and(|f| on_path(f, to, c)));
                    }
                    // 動いた駒の印は外す。駒を取った駒は、取ったマスにいると相手に伝わる
                    for marks in &mut exposed {
                        marks.retain(|&(c, _)| Some(c) != from && c != to);
                    }
                    if captured.is_some() {
                        exposed[0].push((to, *move_number));
                    }
                }
                if let Some((from, to)) = move_squares(usi) {
                    if let Some(from) = from {
                        unmoved.remove(&from);
                        if Some(from) == king_sq {
                            king_sq = Some(to);
                            king_fouls.retain(|&(c, _)| c != to);
                        }
                    }
                    if let Some(role) = captured {
                        p.max(P_CAP, to, fresh(*move_number));
                        if let Some(i) = hand_index(*role) {
                            p.max(P_CAP_ROLE + i, to, fresh(*move_number));
                        }
                        if let (true, Some(i)) = (v5, hand_index(unpromote_role(*role))) {
                            p.add(E_CAP_ROLE + i, to, COUNT_STEP);
                            // 歩を取った筋: 相手の歩は初めに各筋1枚で、打ち直さない限りその筋から歩が消える
                            // （相手が歩を打てる筋 = 二歩にならない筋）。通知の駒種は成りを戻したものなので、
                            // と金を取ったときも区別できず歩として数える
                            if unpromote_role(*role) == Role::Pawn {
                                for rank in 1..=9 {
                                    p.add(E_PAWN_FILE, Coord { file: to.file, rank }, COUNT_STEP);
                                }
                            }
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
                // 王手からの手数（v6）
                last_opp_capture = captured_my_piece_at.as_deref().and_then(parse_usi_square);
                opp_counts[OC_MOVES] += 1;
                opp_counts[OC_CAPTURES] += last_opp_capture.is_some() as u32;
                if let Some(x) = last_opp_capture {
                    for marks in &mut exposed {
                        marks.retain(|&(c, _)| c != x);
                    }
                }
                since_gave += 1;
                since_got += 1;
                last_opp_free = !gave_check_last && last_opp_capture.is_none();
                if last_opp_free {
                    free_gave += 1;
                    free_got += 1;
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
                if let Some((Some(from), to)) = move_squares(usi) {
                    if Some(from) == king_sq {
                        king_fouls.retain(|&(c, _)| c != to);
                        king_fouls.push((to, *move_number));
                    }
                }
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
                    got_ever = true;
                    since_got = 0;
                    free_got = 0;
                    opp_counts[OC_CHECKS] += 1;
                    opp_counts[OC_BOTH] += last_opp_capture.is_some() as u32;
                    if std::mem::take(&mut last_opp_free) {
                        free_gave = free_gave.saturating_sub(1); // 王手の手は相手玉の周りに使った手ではない
                    }
                    if let (true, Some(k)) = (v6, king) {
                        // 王手駒がいうるマス（この時点の自駒から。model は直前の駒取りを反映済み）
                        let own: HashSet<Coord> =
                            model.my_pieces().iter().filter_map(|v| parse_usi_square(&v.square)).collect();
                        pending_check = Some((checker_squares_at(&own, k, color, last_opp_capture), last_mn));
                    }
                } else {
                    if let Some((_, to)) = my_moves.last().and_then(|u| move_squares(u)) {
                        p.max(P_GAVE_CHECK, to, fresh(last_mn));
                        exposed[1].push((to, last_mn));
                    }
                    gave_check_last = true;
                    gave_ever = true;
                    since_gave = 0;
                    free_gave = 0;
                }
            }
            // 相手のまだ閉じていない手番の反則（手番側の観測を書くときは無い）は数えない
            Observation::OpponentFoul { .. } => opp_fouls_pending += 1,
        }
        model.apply(event);
        // 取られた自駒の駒種は、取られる前の自駒の配置（model）から決まる。model が追えていれば、いま
        // `lost_pieces` に1件足されている（成りは戻す = 相手の持ち駒に入る駒種）
        if let (true, Observation::OpponentMoved { move_number, .. }, Some(&(c, role))) =
            (v5, event, model.lost_pieces().get(lost_before))
        {
            if let Some(i) = hand_index(unpromote_role(role)) {
                p.add(E_LOST_ROLE + i, c, COUNT_STEP);
                lost_mn[i] = Some(*move_number);
            }
        }
    }
    for v in &mut p.buf[P_LOST_COUNT * 81..(P_LOST_COUNT + 1) * 81] {
        *v = v.min(1.0);
    }
    if v5 {
        for v in &mut p.buf[E_LOST_ROLE * 81..NUM_PLANES_V5 * 81] {
            *v = v.min(1.0);
        }
        for (i, mn) in lost_mn.iter().enumerate() {
            if let Some(mn) = *mn {
                p.fill(E_LOST_FRESH + i, fresh(mn));
            }
        }
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
        // （成りは戻す）が上界になる。相手の打ちは見えないので取った数からは減らせない。
        // 相手の保有数（盤上＋持ち駒）でも抑える（こちらが相手の駒を取れば下がる）。着手履歴が欠けているときは
        // 取られた駒を数え漏らしうるので、保有数だけを使う（これも上界）。
        // 欠けの判定は自駒の配置の照合（`history_broken`）だけでは足りない: 取られた自駒を取り返して打ち直すと、
        // 欠けた区間の後でも自駒の配置と持ち駒が視界と一致する（再起動後の空ログで、相手は歩を1枚持つのに
        // 上界が0になる）。受理手の手数が欠けずに今の手数まで並んでいることも確かめる
        let lost_counted = !history_broken && moves_contiguous(log, now);
        let mut lost = [0u32; 7];
        for &(_, role) in model.lost_pieces() {
            if let Some(i) = hand_index(unpromote_role(role)) {
                lost[i] += 1;
            }
        }
        for i in 0..7 {
            let bound = if lost_counted { lost[i].min(hold[i]) } else { hold[i] };
            p.fill(E_HAND + i, bound as f32 / ROLE_TOTAL[i] as f32);
        }
        // 相手の反則の時系列（鮮度は相手の手番を閉じた `OpponentMoved` の手数で測る。P_LOST と同じ）
        p.fill(E_FOUL + EF_LAST, opp_fouls_last as f32 / max_fouls);
        p.fill(E_FOUL + EF_CHECKED, opp_fouls_checked / max_fouls);
        p.fill(E_FOUL + EF_FREE, opp_fouls_free / max_fouls);
    }
    // 着手履歴が欠けていると自玉の位置を追えないので書かない（違うマスを確定事実として載せない）
    if v4 && !history_broken {
        for &(c, mn) in &king_fouls {
            p.set(E_KING_FOUL, c, fresh(mn));
        }
    }

    if v6 {
        let clock = |n: u32| n.min(CLOCK_CAP) as f32 / CLOCK_CAP as f32;
        for (k, v) in [
            (CC_GAVE_EVER, gave_ever as u8 as f32),
            (CC_GAVE_MOVES, clock(since_gave)),
            (CC_GAVE_FREE, clock(free_gave)),
            (CC_GOT_EVER, got_ever as u8 as f32),
            (CC_GOT_MOVES, clock(since_got)),
            (CC_GOT_FREE, clock(free_got)),
        ] {
            p.fill(E_CHECK_CLOCK + k, v);
        }
        // 自駒だけを置いた盤（相手の駒は見えないので、飛び駒の利きは自駒にしか遮られない上界になる）
        let mut board = Position::empty(color);
        for piece in &view.your_pieces {
            if let Some(c) = parse_usi_square(&piece.square) {
                board.set(c, Some(Piece { color, role: piece.role }));
            }
        }
        let own: Vec<Coord> = board.pieces().map(|(c, _)| c).collect();
        // 締めた相手玉の候補は「相手のどの手で玉が動けたか」を1手ずつ絞るので、相手の手が1つでも欠けたログ
        // （再起動後の復帰対局・途中の取りこぼし）からは真の玉を落としうる。自駒の配置が追えていても、相手の
        // 駒を取らない手の欠けは分からないので、受理手の手数が欠けずに今の手数まで並んでいることも確かめる。
        // どちらかが崩れていたら上界（自駒のいないマス全部）を書く
        if !history_broken && moves_contiguous(log, now) {
            for c in cache.tight_candidates(color, log) {
                p.set(E_KING_TIGHT, c, 1.0);
            }
        } else {
            for file in 1..=9 {
                for rank in 1..=9 {
                    let c = Coord { file, rank };
                    if board.piece_at(c).is_none() {
                        p.set(E_KING_TIGHT, c, 1.0);
                    }
                }
            }
        }
        // 逃げた王手駒の候補。値は 鮮度 ÷ 王手の時点の候補数（そのマスにまだいる見込みの目安）。いま自駒がいるマスは除く。
        // 着手履歴が欠けていると王手の時点の自駒が追えないので書かない
        if let (false, Some((cands, n, mn))) = (history_broken, &checker_memo) {
            for &c in cands {
                if board.piece_at(c).is_none() {
                    p.set(E_CHECKER_MEMO, c, fresh(*mn) / *n as f32);
                }
            }
        }
        // 自駒の利き。確実な利き = 隣接か桂の跳び（未知の駒に遮られ得ない）、遮られうる利き = 飛び駒の2マス目以降
        // （最初の自駒まで。自駒のいるマスも数える = 紐）。値は枚数 × 1/3 で 3枚以上は 1
        let mut attack = [[0u8; 81]; 2];
        let at = |c: Coord| ((c.file - 1) * 9 + (c.rank - 1)) as usize;
        for &from in &own {
            let role = board.piece_at(from).expect("自駒のマス").role;
            for &d in steps(role) {
                let (df, dr) = orient(d, color);
                let c = Coord { file: from.file + df, rank: from.rank + dr };
                if on_board(c) {
                    attack[0][at(c)] += 1;
                }
            }
            for &d in rays(role) {
                let (df, dr) = orient(d, color);
                let mut c = Coord { file: from.file + df, rank: from.rank + dr };
                let mut k = 0; // 1マス目は確実、2マス目以降は遮られうる
                while on_board(c) {
                    attack[k][at(c)] += 1;
                    k = 1;
                    if board.piece_at(c).is_some() {
                        break;
                    }
                    c = Coord { file: c.file + df, rank: c.rank + dr };
                }
            }
        }
        for file in 1..=9 {
            for rank in 1..=9 {
                let sq = Coord { file, rank };
                for (k, counts) in attack.iter().enumerate() {
                    if counts[at(sq)] > 0 {
                        p.set(E_OWN_ATTACK + k, sq, (counts[at(sq)] as f32 * COUNT_STEP).min(1.0));
                    }
                }
            }
        }
        // 自玉の筋: 8方向に最初の自駒の手前までのマス（相手の飛び駒がいれば王手になる筋）と、その最初の自駒の
        // うち先に自駒でないマスが続くもの（後ろに相手の飛び駒がいればピンされていて、筋から外す手は反則になる）
        let king = view.your_pieces.iter().find(|x| x.role == Role::King).and_then(|x| parse_usi_square(&x.square));
        if let Some(k) = king {
            for (df, dr) in LINE_DIRS {
                let next = |c: Coord| Coord { file: c.file + df, rank: c.rank + dr };
                let mut c = next(k);
                while on_board(c) && board.piece_at(c).is_none() {
                    p.set(E_KING_LINE, c, 1.0);
                    c = next(c);
                }
                if on_board(c) && on_board(next(c)) && board.piece_at(next(c)).is_none() {
                    p.set(E_KING_LINE + 1, c, 1.0);
                }
            }
        }
        // 相手の手数の合計
        for (k, &n) in opp_counts.iter().enumerate() {
            p.fill(E_OPP_COUNT + k, n.min(OPP_COUNT_CAPS[k]) as f32 / OPP_COUNT_CAPS[k] as f32);
        }
        // 相手に位置が伝わっている自駒。着手履歴が欠けていると、動いた・取られたを追えないので書かない
        if !history_broken {
            for (k, marks) in exposed.iter().enumerate() {
                for &(c, mn) in marks {
                    if board.piece_at(c).is_some() {
                        p.set(E_EXPOSED + k, c, fresh(mn));
                    }
                }
            }
        }
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

/// プレーン数 `planes`（`NUM_PLANES` / `NUM_PLANES_DEDUCED` / `NUM_PLANES_V3` / `NUM_PLANES_V4` / `NUM_PLANES_V5` /
/// `NUM_PLANES_V6`）の観測。
/// 重みのヘッダのプレーン数を渡す
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

        // 履歴が欠けていると自玉の位置を追えないので、玉の移動で反則したマス（v4）は書かない。
        // ログには初期位置の玉（5九）の反則が残っているが、本当の自玉がそこにいたとは限らない
        let mut log = ObservationLog::default();
        log.record(Observation::MyFoul { move_number: 59, usi: "5i5h".into() });
        let view = referee.view(Color::Sente, [0, 0], 0);
        let v4 = encode_planes(NUM_PLANES_V4, &view, &log, &HashSet::new());
        assert_eq!(plane(&v4, P_SCALAR + S_HISTORY_BROKEN)[0], 1.0);
        assert!(plane(&v4, E_KING_FOUL).iter().all(|&v| v == 0.0), "履歴が欠けた局で玉の反則マスを書いた");
        // 同じ反則でも履歴が揃っていれば書く（初期局面から玉が 5八 へ行こうとして反則）
        let fresh_ref = Referee::new();
        let view = fresh_ref.view(Color::Sente, [0, 0], 0);
        let mut log = ObservationLog::default();
        log.record(Observation::MyFoul { move_number: 1, usi: "5i5h".into() });
        let v4 = encode_planes(NUM_PLANES_V4, &view, &log, &HashSet::new());
        assert_eq!(plane(&v4, P_SCALAR + S_HISTORY_BROKEN)[0], 0.0);
        assert_eq!(plane(&v4, E_KING_FOUL)[square_index(normalize(sq("5h"), Color::Sente))], 1.0);
    }

    /// 確定事実の途中状態を使い回しても（`encode_into_cached`）、毎回たどり直すのと同じ観測になる。
    /// 両方の側を毎手番書く（手番でない側のログは自分の手で終わる = 王手宣言を待って保留する末尾の
    /// 自分の手も通る）。1つの途中状態を次の局へ持ち越すと、ログが短くなるので作り直される
    #[test]
    fn 確定事実の途中状態を使い回しても同じ観測になる() {
        let mut rng = StdRng::seed_from_u64(12);
        let mut caches = [DeducedCache::default(), DeducedCache::default()];
        // v6 は v2〜v5 を含むので、確定事実の途中状態（締めた相手玉の候補も）と v3〜v6 の足したプレーンをまとめて検査する
        let mut fresh = vec![0.0; NUM_PLANES_V6 * 81];
        let mut cached = vec![0.0; NUM_PLANES_V6 * 81];
        let mut n = 0;
        // 毎回たどり直す側は手数の2乗で重い（v6 は締めた相手玉の候補も最初からたどる）ので、局数は少なめ
        for game in 0..5 {
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
        println!("{n} 観測");
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
        let mut elapsed_v5_cached = std::time::Duration::ZERO;
        let mut elapsed_v6_cached = std::time::Duration::ZERO;
        let mut buf = vec![0.0; OBS_LEN];
        let mut buf_v2 = vec![0.0; NUM_PLANES_DEDUCED * 81];
        let mut buf_v5 = vec![0.0; NUM_PLANES_V5 * 81];
        let mut buf_v6 = vec![0.0; NUM_PLANES_V6 * 81];
        for game in 0..20 {
            let mut referee = Referee::new();
            let mut caches = [DeducedCache::default(), DeducedCache::default()];
            let mut caches_v4 = [DeducedCache::default(), DeducedCache::default()];
            let mut caches_v6 = [DeducedCache::default(), DeducedCache::default()];
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
                let cache = &mut caches_v4[(side == Color::Gote) as usize];
                let t = std::time::Instant::now();
                encode_into_cached(&view, referee.log(side), referee.foul_tried(side), &mut buf_v5, cache);
                elapsed_v5_cached += t.elapsed();
                let cache = &mut caches_v6[(side == Color::Gote) as usize];
                let t = std::time::Instant::now();
                encode_into_cached(&view, referee.log(side), referee.foul_tried(side), &mut buf_v6, cache);
                elapsed_v6_cached += t.elapsed();
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
             （途中状態を使い回すと {:.1}µs/決定点。v5 は使い回して {:.1}µs/決定点、v6 は {:.1}µs/決定点）",
            elapsed.as_secs_f64() * 1e6 / n as f64,
            elapsed_v2.as_secs_f64() * 1e6 / n as f64,
            elapsed_v2_cached.as_secs_f64() * 1e6 / n as f64,
            elapsed_v5_cached.as_secs_f64() * 1e6 / n as f64,
            elapsed_v6_cached.as_secs_f64() * 1e6 / n as f64
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

    /// エンコーダ v6 の前半は v5 と同じで、足した7組は審判の真実と矛盾しない:
    /// - 締めた相手玉の候補は真の相手玉を含み、元の候補（`D_KING`）の中に収まる
    /// - 王手からの手数は、審判の盤で数え直した値（王手・駒取り・王手への応手の別）と一致する
    /// - 逃げた王手駒の候補: 王手を解消した直後（相手が指す前）、取っていない真の王手駒のマスの値は
    ///   鮮度 ÷ 王手の時点の候補数（候補数は真の盤から数える）。合計は 1 以下
    /// - 自駒の利き: 確実な利きのあるマスは真の盤でも自駒が利いていて、真の盤で自駒が利いているマスは
    ///   どちらかのプレーンに値がある。値は自駒だけを置いた盤で数え直した枚数 × 1/3（上限 1）と一致する
    /// - 自玉の筋: 筋の上に自駒はおらず、王手中の真の王手駒（桂以外）は筋の上にいる。ピンされうる自駒のマスには自駒がいる
    /// - 相手の手数の合計は、審判の盤で数え直した値（王手・駒取り・王手かつ駒取り・指した手）と一致する
    /// - 相手に位置が伝わっている自駒は、審判の盤で追った「駒を取ったまま／王手をかけた手で動かしたまま、動いても
    ///   取られてもいない自駒」のマスと一致し、値は印を付けた手の手数（審判の手数）からの鮮度
    /// - 審判のログは受理手の手数が欠けない（= 締めた相手玉の候補は上界へ戻らない）
    #[test]
    fn v6の足したプレーンは真の盤面と矛盾しない() {
        let mut rng = StdRng::seed_from_u64(16);
        let idx = |c: Color| (c == Color::Gote) as usize;
        let clock = |n: u32| n.min(CLOCK_CAP) as f32 / CLOCK_CAP as f32;
        let (mut n, mut n_tight, mut n_memo, mut n_checks, mut n_free, mut n_pin, mut n_both) = (0, 0, 0, 0, 0, 0, 0);
        let mut n_exposed = [0u32; 2];
        // 自駒の利きの枚数（0, 1, 2, 3以上）ごとのマス数（[確実, 遮られうる]）
        let mut n_attack = [[0u32; 4]; 2];
        let (mut v5, mut v6) = (vec![0.0; NUM_PLANES_V5 * 81], vec![0.0; NUM_PLANES_V6 * 81]);
        for game in 0..24 {
            let mut referee = Referee::new();
            // 側ごとの数え直し: ever = [かけたことがある, 受けたことがある]、cnt = [かけてから, その自由, 受けてから, その自由]
            let mut ever = [[false; 2]; 2];
            let mut cnt = [[0u32; 4]; 2];
            // 相手の手数の合計の数え直し（側ごと。並びは `OC_*`）
            let mut total = [[0u32; 4]; 2];
            // 相手に位置が伝わっている自駒の数え直し（側ごとの [駒を取った, 王手をかけた手で動かした] のマス）
            // （マス → 印を付けた手の手数）
            let mut marks: [[std::collections::HashMap<Coord, u32>; 2]; 2] = Default::default();
            // 途中状態を使い回して書く（毎回たどり直すのと同じ観測になることは上のテストが見ている）
            let mut caches: [[DeducedCache; 2]; 2] = Default::default();
            loop {
                let side = referee.to_move();
                let pos = referee.position().clone();
                let attackers = |target: Coord, by: Color| -> Vec<Coord> {
                    pos.pieces().filter(|(c, p)| p.color == by && pos.attacks(*c, target)).map(|(c, _)| c).collect()
                };
                for s in [Color::Sente, Color::Gote] {
                    let view = referee.view(s, [0, 0], game);
                    let (log, tried) = (referee.log(s), referee.foul_tried(s));
                    let [c5, c6] = &mut caches[idx(s)];
                    encode_into_cached(&view, log, tried, &mut v5, c5);
                    encode_into_cached(&view, log, tried, &mut v6, c6);
                    assert_eq!(&v6[..NUM_PLANES_V5 * 81], &v5[..], "v6 の前半は v5 と同じ");
                    assert!(v6[NUM_PLANES_V5 * 81..].iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
                    let at = |k: usize, c: Coord| v6[k * 81 + square_index(normalize(c, s))];
                    let squares = || (1..=9).flat_map(|file| (1..=9).map(move |rank| Coord { file, rank }));
                    // 締めた相手玉の候補
                    let opp_king = pos.king_square(s.other()).unwrap();
                    assert_eq!(at(E_KING_TIGHT, opp_king), 1.0, "game {game} {s:?}: 締めた候補に相手玉が無い");
                    assert!(squares().all(|c| at(E_KING_TIGHT, c) <= at(D_KING, c)), "締めた候補が元の候補の外にある");
                    n_tight += squares().any(|c| at(E_KING_TIGHT, c) < at(D_KING, c)) as u32;
                    // 王手からの手数
                    let (e, c4) = (ever[idx(s)], cnt[idx(s)]);
                    let want = [e[0] as u8 as f32, clock(c4[0]), clock(c4[1]), e[1] as u8 as f32, clock(c4[2]), clock(c4[3])];
                    for (k, w) in want.into_iter().enumerate() {
                        let got = plane(&v6, E_CHECK_CLOCK + k);
                        assert!(got.iter().all(|v| *v == w), "game {game} {s:?}: 手数の {k}番 {} ≠ {w}", got[0]);
                    }
                    // 逃げた王手駒の候補
                    assert!(plane(&v6, E_CHECKER_MEMO).iter().sum::<f32>() <= 1.0 + 1e-4);
                    // 相手に位置が伝わっている自駒（値は印を付けた手の手数から数えた鮮度。エンコーダの P_CAP / P_GAVE_CHECK
                    // からではなく、審判の手数から直接計算する）
                    let mine = |c: Coord| pos.piece_at(c).is_some_and(|pc| pc.color == s);
                    assert!(moves_contiguous(log, view.move_number), "審判のログは手数が欠けない");
                    for k in 0..2 {
                        for c in squares() {
                            let want = match marks[idx(s)][k].get(&c) {
                                Some(&mn) => DECAY.powi((view.move_number - mn) as i32),
                                None => 0.0,
                            };
                            assert!(
                                (at(E_EXPOSED + k, c) - want).abs() < 1e-6,
                                "game {game} {s:?}: 伝わっている自駒 {k} の {c:?} {} ≠ {want}",
                                at(E_EXPOSED + k, c)
                            );
                            assert!(want == 0.0 || mine(c), "印のマスに自駒がいない");
                            n_exposed[k] += (want > 0.0) as u32;
                        }
                    }
                    // 相手の手数の合計
                    for (k, &t) in total[idx(s)].iter().enumerate() {
                        let w = t.min(OPP_COUNT_CAPS[k]) as f32 / OPP_COUNT_CAPS[k] as f32;
                        let got = plane(&v6, E_OPP_COUNT + k);
                        assert!(got.iter().all(|v| *v == w), "game {game} {s:?}: 合計の {k}番 {} ≠ {w}", got[0]);
                    }
                    // 自駒の利き
                    for c in squares() {
                        let real = pos.pieces().any(|(f, pc)| pc.color == s && f != c && pos.attacks(f, c));
                        let (sure, maybe) = (at(E_OWN_ATTACK, c) > 0.0, at(E_OWN_ATTACK + 1, c) > 0.0);
                        assert!(!sure || real, "game {game} {s:?}: 確実な利き {c:?} が真の盤に無い");
                        assert!(!real || sure || maybe, "game {game} {s:?}: 真の利き {c:?} がどちらのプレーンにも無い");
                    }
                    // 自駒の利きの値: 自駒だけを置いた盤で数え直す。確実な利き = 隣接か桂の跳び（飛び駒の1マス目は
                    // 隣接）、遮られうる利き = それ以外（飛び駒の2マス目以降）。値は枚数 × 1/3 で 3枚以上は 1
                    let mut own_only = Position::empty(s);
                    for (f, pc) in pos.pieces().filter(|(_, pc)| pc.color == s) {
                        own_only.set(f, Some(pc));
                    }
                    for c in squares() {
                        let mut count = [0u32; 2];
                        for (f, _) in own_only.pieces() {
                            if f != c && own_only.attacks(f, c) {
                                let (df, dr) = ((f.file - c.file).abs(), (f.rank - c.rank).abs());
                                let sure = (df <= 1 && dr <= 1) || (df == 1 && dr == 2);
                                count[!sure as usize] += 1;
                            }
                        }
                        for k in 0..2 {
                            let want = (count[k] as f32 * COUNT_STEP).min(1.0);
                            assert!(
                                (at(E_OWN_ATTACK + k, c) - want).abs() < 1e-6,
                                "game {game} {s:?}: 利き {k} の {c:?} {} ≠ {want}（{} 枚）",
                                at(E_OWN_ATTACK + k, c),
                                count[k]
                            );
                            n_attack[k][count[k].min(3) as usize] += 1;
                        }
                    }
                    // 自玉の筋
                    for c in squares() {
                        assert!(at(E_KING_LINE, c) == 0.0 || !mine(c), "筋の上に自駒がいる");
                        if at(E_KING_LINE + 1, c) > 0.0 {
                            assert!(mine(c), "ピンされうる自駒のマスに自駒がいない");
                            n_pin += 1;
                        }
                    }
                    if view.you_in_check {
                        let king = pos.king_square(s).unwrap();
                        for c in attackers(king, s.other()) {
                            if pos.piece_at(c).unwrap().role != Role::Knight {
                                assert_eq!(at(E_KING_LINE, c), 1.0, "game {game} {s:?}: 王手駒 {c:?} が筋の上に無い");
                            }
                        }
                    }
                    n += 1;
                }
                let view = referee.view(side, [0, 0], game);
                let in_check = view.you_in_check;
                let checkers = if in_check { attackers(pos.king_square(side).unwrap(), side.other()) } else { vec![] };
                // 王手を受けた時点の候補数（真の盤の自駒と自玉・直前に取られたマスから。エンコーダは GameModel から作る）
                let n_cands = in_check.then(|| {
                    let own: HashSet<Coord> = pos.pieces().filter(|(_, pc)| pc.color == side).map(|(c, _)| c).collect();
                    let cap = referee.log(side).events().iter().rev().find_map(|e| match e {
                        Observation::OpponentMoved { captured_my_piece_at, .. } => {
                            Some(captured_my_piece_at.as_deref().and_then(parse_usi_square))
                        }
                        _ => None,
                    });
                    crate::rl::action::checker_squares_at(&own, pos.king_square(side).unwrap(), side, cap.flatten())
                        .len()
                });
                let check_mn = pos.move_number();
                let mask = legal_mask(&view, referee.log(side), referee.foul_tried(side));
                let actions: Vec<usize> = (0..NUM_ACTIONS).filter(|&a| mask[a]).collect();
                if actions.is_empty() {
                    break;
                }
                let usi = decode_usi(actions[rng.random_range(0..actions.len())], side).unwrap();
                match referee.step(&usi, 0) {
                    StepResult::Ended { .. } => break,
                    StepResult::Accepted => {}
                    _ => continue, // 反則は手番も数え直しも変えない
                }
                let other = side.other();
                let after = referee.position();
                let count = |q: &Position, c: Color| q.pieces().filter(|(_, pc)| pc.color == c).count();
                let captured = count(after, other) < count(&pos, other);
                let gives_check = referee.view(other, [0, 0], game).you_in_check;
                let (i, j) = (idx(side), idx(other));
                let (from, to) = match parse_usi(&usi) {
                    Some(ShogiMove::Board { from, to, .. }) => (Some(from), to),
                    Some(ShogiMove::Drop { to, .. }) => (None, to),
                    None => unreachable!("受理された手は読める"),
                };
                // 動いた駒・取られた駒の印を外し、駒を取った駒・王手をかけた手で動かした駒に印を付ける
                for k in 0..2 {
                    marks[j][k].remove(&to);
                    marks[i][k].remove(&to);
                    if let Some(f) = from {
                        marks[i][k].remove(&f);
                    }
                }
                let mn = after.move_number();
                if captured {
                    marks[i][0].insert(to, mn);
                }
                if gives_check {
                    marks[i][1].insert(to, mn);
                }
                total[j][OC_MOVES] += 1;
                total[j][OC_CAPTURES] += captured as u32;
                total[j][OC_CHECKS] += gives_check as u32;
                total[j][OC_BOTH] += (captured && gives_check) as u32;
                n_both += (captured && gives_check) as u32;
                cnt[j][0] += 1;
                cnt[j][2] += 1;
                if !in_check && !captured && !gives_check {
                    cnt[j][1] += 1;
                    cnt[j][3] += 1;
                    n_free += 1;
                }
                if gives_check {
                    (ever[i][0], ever[j][1]) = (true, true);
                    (cnt[i][0], cnt[i][1], cnt[j][2], cnt[j][3]) = (0, 0, 0, 0);
                    n_checks += 1;
                }
                // 王手を解消した直後: 取っていない真の王手駒のマスに、逃げた王手駒の候補の値がある
                if in_check {
                    let view = referee.view(side, [0, 0], game);
                    let v6 = encode_planes(NUM_PLANES_V6, &view, referee.log(side), referee.foul_tried(side));
                    // 値 = 王手の手数からの鮮度 ÷ 王手の時点の候補数（取っていない王手駒のマスには自駒が着地も通過もできない）
                    let want = DECAY.powi((view.move_number - check_mn) as i32) / n_cands.unwrap() as f32;
                    for c in checkers {
                        if after.piece_at(c).is_some_and(|pc| pc.color == other) {
                            let v = v6[E_CHECKER_MEMO * 81 + square_index(normalize(c, side))];
                            assert!(
                                (v - want).abs() < 1e-6,
                                "game {game} {side:?}: 逃げた王手駒 {c:?} の値 {v} ≠ {want}（候補 {:?} 個）",
                                n_cands
                            );
                            n_memo += 1;
                        }
                    }
                }
            }
        }
        println!(
            "{n} 観測: 締めた候補が元より狭い {n_tight} / 王手 {n_checks}（うち駒取り {n_both}）/ 自由な手 {n_free} / \
             逃げた王手駒 {n_memo} / ピンされうる自駒 {n_pin} / 伝わっている自駒 {n_exposed:?}"
        );
        assert!(
            n_tight > 0 && n_checks > 0 && n_both > 0 && n_free > 0 && n_memo > 0 && n_pin > 0,
            "v6 の検査が空振りしている"
        );
        assert!(n_exposed.iter().all(|&x| x > 0), "伝わっている自駒の検査が空振りしている");
        println!("自駒の利きの枚数ごとのマス数（0, 1, 2, 3以上）: 確実 {:?} / 遮られうる {:?}", n_attack[0], n_attack[1]);
        assert!(n_attack.iter().all(|k| k.iter().all(|&x| x > 0)), "利きの枚数の検査が空振りしている");
    }

    /// 履歴が欠けたログ（再起動後の復帰対局の空ログ・途中の相手の手の取りこぼし）では、締めた相手玉の候補は
    /// 真の相手玉を落とさない（上界 = 自駒のいないマス全部へ戻る）。自駒の配置が追えていても相手の駒を取らない手の
    /// 欠けは分からないので、受理手の手数の並びで見つける
    #[test]
    fn 履歴が欠けると締めた相手玉の候補は上界へ戻る() {
        let mut referee = Referee::new();
        // 先手 7六歩・後手 3四歩・先手 4八玉・後手 8四歩・先手 3八玉（先手玉は 5九 → 3八 と2手で動いた）
        for usi in ["7g7f", "3c3d", "5i4h", "8c8d", "4h3h"] {
            assert!(matches!(referee.step(usi, 0), StepResult::Accepted));
        }
        let side = Color::Gote;
        let view = referee.view(side, [0, 0], 0);
        let opp_king = referee.position().king_square(Color::Sente).unwrap();
        let full = referee.log(side);
        let tight_of = |log: &ObservationLog| -> Vec<f32> {
            plane(&encode_planes(NUM_PLANES_V6, &view, log, &HashSet::new()), E_KING_TIGHT).to_vec()
        };
        let upper: Vec<f32> = (0..81)
            .map(|i| {
                let c = normalize(index_square(i), side);
                view.your_pieces.iter().all(|v| parse_usi_square(&v.square) != Some(c)) as u8 as f32
            })
            .collect();
        let at_king = |t: &[f32]| t[square_index(normalize(opp_king, side))];
        // 欠けていないログ: 締めた候補を使う（上界より狭い）
        assert!(moves_contiguous(full, view.move_number));
        let t = tight_of(full);
        assert_eq!(at_king(&t), 1.0);
        assert!(t.iter().sum::<f32>() < upper.iter().sum::<f32>(), "欠けていないログは締めた候補を使う");
        // 途中の相手の手（先手 4八玉）が欠けたログ / 再起動後の空ログ
        let mut gap = ObservationLog::default();
        let mut skipped = false;
        for e in full.events() {
            if !skipped && matches!(e, Observation::OpponentMoved { move_number: 4, .. }) {
                skipped = true;
                continue;
            }
            gap.record(e.clone());
        }
        assert!(skipped);
        for (name, log) in [("途中の欠け", gap), ("空ログ", ObservationLog::default())] {
            assert!(!moves_contiguous(&log, view.move_number), "{name}");
            let t = tight_of(&log);
            assert_eq!(at_king(&t), 1.0, "{name}: 真の相手玉が候補に無い");
            assert_eq!(t, upper, "{name}: 上界へ戻る");
        }
        // 受理手の手数を適用前の値（初手が 1。webhook のセッションの規約）で付けたログは、欠けていない:
        // 締めた候補は審判の規約のログと同じ。途中が欠ければ、この規約でも上界へ戻る
        let renumber = |log: &ObservationLog| -> ObservationLog {
            let mut out = ObservationLog::default();
            for e in log.events() {
                out.record(match e.clone() {
                    Observation::MyMove { move_number, usi, captured } => {
                        Observation::MyMove { move_number: move_number - 1, usi, captured }
                    }
                    Observation::OpponentMoved { move_number, captured_my_piece_at } => {
                        Observation::OpponentMoved { move_number: move_number - 1, captured_my_piece_at }
                    }
                    other => other,
                });
            }
            out
        };
        let before = renumber(full);
        assert!(moves_contiguous(&before, view.move_number));
        assert_eq!(tight_of(&before), tight_of(full), "適用前の規約のログでも締めた候補を使う");
        let mut before_gap = ObservationLog::default();
        for e in before.events() {
            if !matches!(e, Observation::OpponentMoved { move_number: 3, .. }) {
                before_gap.record(e.clone());
            }
        }
        assert!(!moves_contiguous(&before_gap, view.move_number));
        assert_eq!(tight_of(&before_gap), upper, "適用前の規約でも、欠ければ上界へ戻る");
        // 最後の受理手だけが欠けたログ・先頭だけが欠けたログ（どちらの規約でも数が合わない）
        for log in [full, &before] {
            let events = log.events();
            let accepted = |e: &Observation| matches!(e, Observation::MyMove { .. } | Observation::OpponentMoved { .. });
            let (first, last) = (
                events.iter().position(accepted).unwrap(),
                events.iter().rposition(accepted).unwrap(),
            );
            for skip in [first, last] {
                let mut cut = ObservationLog::default();
                for (i, e) in events.iter().enumerate() {
                    if i != skip {
                        cut.record(e.clone());
                    }
                }
                assert!(!moves_contiguous(&cut, view.move_number), "受理手 {skip} が欠けたログ");
            }
        }
    }

    /// 履歴が欠けたログでは、相手の持ち駒の上界は取られた自駒の数を使わず保有数へ戻る。自駒の配置と持ち駒が
    /// 視界と一致していても（取られた自駒を取り返して打ち直すと、欠けた区間の後でも一致する）欠けは欠け
    #[test]
    fn 履歴が欠けると相手の持ち駒の上界は保有数へ戻る() {
        let mut referee = Referee::new();
        // 後手が先手の歩を2五で取り、先手が飛で取り返して飛を戻し、歩を2七へ打ち直す。先手の盤上と持ち駒は
        // 初期局面と同じだが、後手は歩を1枚持っている
        for usi in ["2g2f", "2c2d", "2f2e", "2d2e", "2h2e", "5a5b", "2e2h", "5b5a", "P*2g", "5a5b"] {
            assert!(matches!(referee.step(usi, 0), StepResult::Accepted), "{usi}");
        }
        let side = Color::Sente;
        let view = referee.view(side, [0, 0], 0);
        let pawn = hand_index(Role::Pawn).unwrap();
        assert_eq!(referee.position().hand_count(Color::Gote, Role::Pawn), 1);
        let full = referee.log(side);
        let bound_of = |log: &ObservationLog, planes: usize| -> (f32, f32) {
            let obs = encode_planes(planes, &view, log, &HashSet::new());
            (plane(&obs, E_HAND + pawn)[0], plane(&obs, P_SCALAR + S_HISTORY_BROKEN)[0])
        };
        let total = ROLE_TOTAL[pawn] as f32;
        let hold = 9.0 / total; // 歩の総数 18 − 先手の盤上 9 − 先手の持ち駒 0
        // 欠けていないログ: 取られた歩の数（1枚）が上界で、本当の持ち駒と一致する
        assert!(moves_contiguous(full, view.move_number));
        for planes in [NUM_PLANES_V3, NUM_PLANES_V6] {
            assert_eq!(bound_of(full, planes), (1.0 / total, 0.0));
        }

        // 再起動後の空ログ: 自駒の配置は初期局面から作ったモデルと一致する（欠けの印は立たない）が、上界は保有数
        let empty = ObservationLog::default();
        assert!(!moves_contiguous(&empty, view.move_number));
        let (bound, broken) = bound_of(&empty, NUM_PLANES_V3);
        assert_eq!(broken, 0.0, "自駒の配置の照合だけでは欠けが分からない例になっている");
        assert_eq!(bound, hold, "空ログで上界が本当の持ち駒（1枚）を下回った");

        // 途中の欠け: 後手が歩を取った手（2d2e）だけを取りこぼしたログ。両方の手数の規約で保有数へ戻る
        let renumber = |log: &ObservationLog, shift: u32| -> ObservationLog {
            let mut out = ObservationLog::default();
            for e in log.events() {
                out.record(match e.clone() {
                    Observation::MyMove { move_number, usi, captured } => {
                        Observation::MyMove { move_number: move_number - shift, usi, captured }
                    }
                    Observation::OpponentMoved { move_number, captured_my_piece_at } => {
                        Observation::OpponentMoved { move_number: move_number - shift, captured_my_piece_at }
                    }
                    other => other,
                });
            }
            out
        };
        for shift in [0, 1] {
            let log = renumber(full, shift);
            assert!(moves_contiguous(&log, view.move_number), "shift {shift}");
            assert_eq!(bound_of(&log, NUM_PLANES_V3).0, 1.0 / total, "shift {shift}: 欠けていない");
            let mut gap = ObservationLog::default();
            let mut skipped = false;
            for e in log.events() {
                if !skipped && matches!(e, Observation::OpponentMoved { captured_my_piece_at: Some(_), .. }) {
                    skipped = true;
                    continue;
                }
                gap.record(e.clone());
            }
            assert!(skipped);
            assert!(!moves_contiguous(&gap, view.move_number), "shift {shift}");
            assert_eq!(bound_of(&gap, NUM_PLANES_V3).0, hold, "shift {shift}: 途中の欠けで上界が保有数へ戻らない");
        }

        // 反則は受理手に数えない: 欠けていないログに今の手番の反則が付いても、取られた歩の数を使う
        let mut with_foul = renumber(full, 0);
        with_foul.record(Observation::MyFoul { move_number: view.move_number, usi: "2g2e".into() });
        assert!(moves_contiguous(&with_foul, view.move_number));
        assert_eq!(bound_of(&with_foul, NUM_PLANES_V3).0, 1.0 / total);
    }

    /// エンコーダ v3 の前半は v2 と同じ・v4 の前半は v3 と同じで、足した組は審判の真実から数え直した値と一致する:
    /// - 相手の持ち駒の上界 = min(相手が取った自駒の数, 相手の保有数)。取った数は「相手の本当の持ち駒＋相手が打った数」
    ///   なので、上界は本当の持ち駒を下回らない
    /// - 相手の反則の時系列 = 相手の手番ごとの反則数（王手を受けていたかで分ける。鮮度は手番を閉じた手数で測る）
    /// - 玉の移動で反則したマス（v4）= 審判で反則になった自玉の手の移動先（その後に玉がそこへ移動できたものは除く）。
    ///   反則した時点で、自玉を除いた盤でそのマスに相手の利きがある
    /// - 取った駒・取られた駒の一覧（v5）= 審判の盤で取られた駒（成りは戻す）のマスごとの回数・取られた駒種の
    ///   最後の手数からの鮮度・歩を取った筋
    #[test]
    fn v3の足したプレーンは真実から数え直した値と一致する() {
        #[derive(Default)]
        struct Turn {
            in_check: bool,
            fouls: u32,
        }
        let mut rng = StdRng::seed_from_u64(13);
        let (mut n, mut n_hand, mut n_tight, mut n_fouls, mut n_checked, mut n_king) = (0, 0, 0, 0, 0, 0);
        let (mut n_cap, mut n_lost, mut n_pawn_file, mut n_twice) = (0, 0, 0, 0);
        for game in 0..12 {
            let mut referee = Referee::new();
            // 確定事実はログを先頭からたどり直すと長い局ほど重いので、途中状態を使い回す（結果は同じ）
            let mut caches = [DeducedCache::default(), DeducedCache::default()];
            // 側ごとの閉じた手番（反則数と、手番の頭で王手を受けていたか。手番を閉じた手数つき）と、打った駒の数
            let mut turns: [Vec<(Turn, u32)>; 2] = [vec![], vec![]];
            let mut dropped = [[0u32; 7]; 2];
            // 側ごとの、自玉の移動で反則したマスと反則した手数
            let mut king_fouls: [Vec<(Coord, u32)>; 2] = [vec![], vec![]];
            // 側ごとの、取った駒（駒種の番号, マス）と取られた駒（駒種の番号, マス, 取られた手数）
            let mut caps: [Vec<(usize, Coord)>; 2] = [vec![], vec![]];
            let mut losts: [Vec<(usize, Coord, u32)>; 2] = [vec![], vec![]];
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
                let mut v4 = vec![0.0; NUM_PLANES_V4 * 81];
                let mut v5 = vec![0.0; NUM_PLANES_V5 * 81];
                encode_into_cached(&view, log, tried, &mut v2, &mut caches[me]);
                encode_into_cached(&view, log, tried, &mut v3, &mut caches[me]);
                encode_into_cached(&view, log, tried, &mut v4, &mut caches[me]);
                encode_into_cached(&view, log, tried, &mut v5, &mut caches[me]);
                assert_eq!(&v3[..v2.len()], &v2[..], "v3 の前半は v2 と同じ");
                assert_eq!(&v4[..v3.len()], &v3[..], "v4 の前半は v3 と同じ");
                assert_eq!(&v5[..v4.len()], &v4[..], "v5 の前半は v4 と同じ");
                assert!(v5.iter().all(|v| v.is_finite() && (0.0..=3.0).contains(v)));
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
                let mut want = vec![0.0f32; 81];
                for &(c, mn) in &king_fouls[me] {
                    want[square_index(normalize(c, side))] = fresh(mn);
                }
                let got = plane(&v4, E_KING_FOUL);
                assert!(
                    got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-6),
                    "game {game} {side:?}: 玉の反則マス {:?}",
                    king_fouls[me]
                );
                n_king += (!king_fouls[me].is_empty()) as u32;
                // v5: 取った駒・取られた駒の一覧
                let sq = |c: Coord| square_index(normalize(c, side));
                let mut want = vec![0.0f32; (NUM_PLANES_V5 - E_LOST_FRESH) * 81];
                let at = |k: usize| (k - E_LOST_FRESH) * 81;
                for &(i, c, mn) in &losts[me] {
                    want[at(E_LOST_ROLE + i) + sq(c)] += COUNT_STEP;
                    // 同じ駒種を何度も取られたら最後の手数（losts は古い順）
                    want[at(E_LOST_FRESH + i)..at(E_LOST_FRESH + i) + 81].fill(fresh(mn));
                }
                for &(i, c) in &caps[me] {
                    want[at(E_CAP_ROLE + i) + sq(c)] += COUNT_STEP;
                    if i == 0 {
                        for rank in 1..=9 {
                            want[at(E_PAWN_FILE) + sq(Coord { file: c.file, rank })] += COUNT_STEP;
                        }
                    }
                }
                for v in &mut want[at(E_LOST_ROLE)..] {
                    *v = v.min(1.0);
                }
                for k in E_LOST_FRESH..NUM_PLANES_V5 {
                    let (got, want) = (plane(&v5, k), &want[at(k)..at(k) + 81]);
                    assert!(
                        got.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-5),
                        "game {game} {side:?} plane {k}: 取った {:?} / 取られた {:?}",
                        caps[me],
                        losts[me]
                    );
                }
                n_cap += (!caps[me].is_empty()) as u32;
                n_lost += (!losts[me].is_empty()) as u32;
                n_pawn_file += caps[me].iter().any(|&(i, _)| i == 0) as u32;
                n_twice += caps[me].iter().enumerate().any(|(j, a)| caps[me][..j].contains(a)) as u32;
                n += 1;

                let mask = legal_mask(&view, log, tried);
                let actions: Vec<usize> = (0..NUM_ACTIONS).filter(|&a| mask[a]).collect();
                if actions.is_empty() {
                    break;
                }
                let usi = decode_usi(actions[rng.random_range(0..actions.len())], side).unwrap();
                let mn_before = referee.position().move_number();
                // この手で取る相手の駒（受理されたら一覧に足す。成りは戻した駒種）
                let victim = match parse_usi(&usi) {
                    Some(ShogiMove::Board { to, .. }) => referee
                        .position()
                        .piece_at(to)
                        .filter(|p| p.color != side)
                        .and_then(|p| hand_index(unpromote_role(p.role)))
                        .map(|i| (i, to)),
                    _ => None,
                };
                let king_to = match parse_usi(&usi) {
                    Some(ShogiMove::Board { from, to, .. })
                        if referee.position().piece_at(from).is_some_and(|p| p.role == Role::King) =>
                    {
                        Some((from, to))
                    }
                    _ => None,
                };
                match referee.step(&usi, 0) {
                    StepResult::Ended { .. } => break,
                    StepResult::Foul => {
                        cur.fouls += 1;
                        if let Some((from, to)) = king_to {
                            let mut wk = referee.position().clone();
                            wk.set(from, None);
                            assert!(wk.is_attacked(to, side.other()), "玉の反則なのに移動先に相手の利きが無い");
                            king_fouls[me].retain(|&(c, _)| c != to);
                            king_fouls[me].push((to, mn_before));
                        }
                    }
                    _ => {
                        if let Some((_, to)) = king_to {
                            king_fouls[me].retain(|&(c, _)| c != to);
                        }
                        if let Some(ShogiMove::Drop { role, .. }) = parse_usi(&usi) {
                            dropped[me][hand_index(role).unwrap()] += 1;
                        }
                        let mn = referee.position().move_number();
                        if let Some((i, to)) = victim {
                            caps[me].push((i, to));
                            losts[opp].push((i, to, mn));
                        }
                        turns[me].push((std::mem::take(&mut cur), mn));
                    }
                }
            }
        }
        println!(
            "{n} 決定点: 上界が本当の持ち駒と一致 {n_tight}/{n_hand}、直前の手番に相手の反則 {n_fouls}、\
             王手を受けた手番の反則あり {n_checked}、玉の反則マスあり {n_king}、駒を取った {n_cap}・取られた {n_lost}・\
             歩を取った筋あり {n_pawn_file}・同じマスで同じ駒種を2回取った {n_twice}"
        );
        assert!(n_fouls > 0 && n_checked > 0 && n_king > 0 && n_tight < n_hand, "検査が空振りしている");
        assert!(n_cap > 0 && n_lost > 0 && n_pawn_file > 0, "v5 の検査が空振りしている");
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
