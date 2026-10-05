//! **speed 系列の凍結版 v2**（2026-10-05）: 方策ネット（DeepNash 路線の R-NaD）。速攻特化 6,000更新（rl_speed_v1 の続き、λ=4・H=120、自己対局＋リーグを GPU で）。rl_speed_v1 に 83.5%、rl_v25 に 73.2%、勝ち局の手数 48.5手（v1 は 59.9手）
//!
//! `scripts/freeze_rl.py` が生成した。**編集しない**（改善は `src/rl/` で行う）。
//! 推論の一式（行動の符号化・観測のテンソル化・手書きの推論）は凍結時点の固定コピーで、
//! 重みは `models/rl_speed_v2.bin` をバイナリに埋め込む（sha256 `03aa2f50f923a72d03964f95d272105479b0faab64d3e252e28652fcba144bb9`）。
//! 探索なしで1手 約10ms。思考予算・実行時 env は持たない。

#![allow(clippy::all)]

use std::collections::HashSet;
use std::sync::OnceLock;

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::observation::ObservationLog;
use crate::protocol::PlayerView;
use crate::strategy::Strategy;

/// 埋め込んだ重み
const WEIGHTS: &[u8] = include_bytes!("../../models/rl_speed_v2.bin");
/// 埋め込んだ重みの sha256（凍結時点の値。テストで検査する）
pub const WEIGHTS_SHA256: &str = "03aa2f50f923a72d03964f95d272105479b0faab64d3e252e28652fcba144bb9";

fn net() -> &'static policy_net::PolicyNet {
    static NET: OnceLock<policy_net::PolicyNet> = OnceLock::new();
    NET.get_or_init(|| policy_net::PolicyNet::from_bytes(WEIGHTS).expect("凍結版の重みが読めない"))
}

/// speed 系列の凍結版 v2 の戦略（`strategy::make("rl_speed_v2")`）。softmax からサンプリングする
pub struct RlSpeedV2 {
    rng: StdRng,
    last: Option<serde_json::Value>,
}

impl RlSpeedV2 {
    pub fn new() -> Self {
        Self { rng: StdRng::from_rng(&mut rand::rng()), last: None }
    }

    pub fn with_seed(seed: u64) -> Self {
        Self { rng: StdRng::seed_from_u64(seed), last: None }
    }
}

impl Default for RlSpeedV2 {
    fn default() -> Self {
        Self::new()
    }
}

impl Strategy for RlSpeedV2 {
    fn choose(
        &mut self,
        view: &PlayerView,
        log: &ObservationLog,
        foul_tried: &HashSet<String>,
    ) -> Option<String> {
        // src/rl/policy.rs の rl_policy（サンプリング）と同じ手順
        let mask = action::legal_mask(view, log, foul_tried);
        let legal: Vec<usize> = (0..mask.len()).filter(|&a| mask[a]).collect();
        if legal.is_empty() {
            return None;
        }
        let (logits, value) = net().forward(&encode::encode(view, log, foul_tried));
        let max = legal.iter().map(|&a| logits[a]).fold(f32::MIN, f32::max);
        let weights: Vec<f64> = legal.iter().map(|&a| f64::from(logits[a] - max).exp()).collect();
        let total: f64 = weights.iter().sum();
        let mut r = self.rng.random::<f64>() * total;
        let mut pick = *legal.last().unwrap();
        for (&a, w) in legal.iter().zip(&weights) {
            if r < *w {
                pick = a;
                break;
            }
            r -= w;
        }
        let prob = weights[legal.iter().position(|&a| a == pick).unwrap()] / total;
        self.last = Some(serde_json::json!({ "p": prob, "value": value }));
        action::decode_usi(pick, view.your_color)
    }

    fn name(&self) -> &'static str {
        "rl_speed_v2"
    }

    fn debug_state(&self) -> Option<serde_json::Value> {
        self.last.clone()
    }
}

/// 行動の符号化とマスク（`src/rl/action.rs` の凍結時点の固定コピー）
pub mod action {
    // 行動の符号化（AlphaZero 将棋と同じ 139 種 × 81 マス）と行動マスク。
    //
    // 手番側を常に先手向きに正規化する（後手なら盤を180度回す）ので、
    // 「前」は常に段が減る方向になる。
    //
    // 行動番号 = 種類 × 81 + マス（方策ヘッドの出力 [139, 9, 9] を平らにした並び）。
    // - 種類 0..64: 8方向 × 距離 1〜8（不成）。マス = 移動元
    // - 種類 64, 65: 桂の2方向（不成）。マス = 移動元
    // - 種類 66..132: 上の 66 種の成り
    // - 種類 132..139: 打ち（`HAND_ROLES` 順）。マス = 打ち先
    //
    // マスクは**自駒だけを見た候補手**（`board.rs`）から、その手番で既に反則した手を除き、
    // さらに**観測から反則が確定する手**（`Deduction`）を除いたもの。
    // 見えている範囲で確定する反則（自駒のマス・自駒の飛び越え・二歩・行き所なし）と、
    // 観測から論理的に確定する反則（王手を解消し得ない手など）は落ち、
    // 見えない相手駒による反則のうち確定しないものは残る。**真の合法手はすべてマスクに含まれる**
    // （テストで常時検査。漏れた手は永久に指せなくなる）。
    //
    // 観測からの除外（`MASK_VERSION` 2、2026-10-02〜）を持たない旧マスクは `basic_legal_mask`。
    // 凍結版 rl_v15〜rl_v22 はこちらで学習・凍結されている。

    use std::collections::HashSet;

    use crate::board::{
        Coord, Promotion, drop_targets, make_usi_drop, make_usi_move, move_targets,
        parse_usi_square, promotion_choice,
    };
    use crate::observation::{Observation, ObservationLog};
    use crate::protocol::{Color, PlayerView, Role};
    use crate::shogi::{HAND_ROLES, ShogiMove, hand_index, parse_usi};

    /// マスクの版。1 = 自駒視点だけ（`basic_legal_mask`）、2 = 観測からの確定反則も除く。
    /// 凍結版の同一性テストは、凍結ファイルがこの定数を持つかでどちらのマスクと比べるかを決める
    pub const MASK_VERSION: u32 = 2;

    pub const NUM_SQUARES: usize = 81;
    /// 方向×距離 64 ＋ 桂 2
    const MOVE_KINDS: usize = 66;
    const DROP_BASE: usize = MOVE_KINDS * 2;
    pub const NUM_ACTION_KINDS: usize = DROP_BASE + HAND_ROLES.len();
    pub const NUM_ACTIONS: usize = NUM_ACTION_KINDS * NUM_SQUARES;

    /// 正規化後の8方向 (筋の差, 段の差)。段が減るのが「前」
    const DIRS: [(i8, i8); 8] = [
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
    ];
    const KNIGHT_DIRS: [(i8, i8); 2] = [(1, -2), (-1, -2)];

    /// 手番側から見た座標へ（後手なら180度回す。自分自身が逆変換）
    pub fn normalize(c: Coord, color: Color) -> Coord {
        match color {
            Color::Sente => c,
            Color::Gote => Coord {
                file: 10 - c.file,
                rank: 10 - c.rank,
            },
        }
    }

    /// 正規化済み座標 → マス番号（段優先: (段-1)×9 + (筋-1)）
    pub fn square_index(c: Coord) -> usize {
        (c.rank as usize - 1) * 9 + (c.file as usize - 1)
    }

    pub fn index_square(i: usize) -> Coord {
        Coord {
            file: (i % 9) as i8 + 1,
            rank: (i / 9) as i8 + 1,
        }
    }

    /// 移動の差分 → 種類（成りを含まない 0..66）
    fn move_kind(df: i8, dr: i8) -> Option<usize> {
        if let Some(k) = KNIGHT_DIRS.iter().position(|&d| d == (df, dr)) {
            return Some(64 + k);
        }
        let dist = df.abs().max(dr.abs());
        if dist == 0 || dist > 8 {
            return None;
        }
        // 直線（縦・横・斜め）でなければ符号化できない
        if df != 0 && dr != 0 && df.abs() != dr.abs() {
            return None;
        }
        let unit = (df.signum(), dr.signum());
        let d = DIRS.iter().position(|&x| x == unit)?;
        Some(d * 8 + (dist as usize - 1))
    }

    /// 指し手 → 行動番号。盤外・符号化できない差分は None
    pub fn encode_move(mv: &ShogiMove, color: Color) -> Option<usize> {
        match *mv {
            ShogiMove::Board { from, to, promote } => {
                let (f, t) = (normalize(from, color), normalize(to, color));
                let kind = move_kind(t.file - f.file, t.rank - f.rank)?;
                let kind = if promote { kind + MOVE_KINDS } else { kind };
                Some(kind * NUM_SQUARES + square_index(f))
            }
            ShogiMove::Drop { role, to } => {
                let kind = DROP_BASE + hand_index(role)?;
                Some(kind * NUM_SQUARES + square_index(normalize(to, color)))
            }
        }
    }

    pub fn encode_usi(usi: &str, color: Color) -> Option<usize> {
        encode_move(&parse_usi(usi)?, color)
    }

    /// 行動番号 → 指し手。盤外へ出る番号は None（マスクの外なので通常は起きない）
    pub fn decode_action(action: usize, color: Color) -> Option<ShogiMove> {
        if action >= NUM_ACTIONS {
            return None;
        }
        let (kind, sq) = (action / NUM_SQUARES, action % NUM_SQUARES);
        let at = index_square(sq);
        if kind >= DROP_BASE {
            return Some(ShogiMove::Drop {
                role: HAND_ROLES[kind - DROP_BASE],
                to: normalize(at, color),
            });
        }
        let promote = kind >= MOVE_KINDS;
        let base = kind % MOVE_KINDS;
        let (df, dr) = if base >= 64 {
            KNIGHT_DIRS[base - 64]
        } else {
            let (ux, uy) = DIRS[base / 8];
            let dist = (base % 8) as i8 + 1;
            (ux * dist, uy * dist)
        };
        let to = Coord {
            file: at.file + df,
            rank: at.rank + dr,
        };
        if !(1..=9).contains(&to.file) || !(1..=9).contains(&to.rank) {
            return None;
        }
        Some(ShogiMove::Board {
            from: normalize(at, color),
            to: normalize(to, color),
            promote,
        })
    }

    pub fn decode_usi(action: usize, color: Color) -> Option<String> {
        Some(decode_action(action, color)?.to_usi())
    }

    /// 自駒だけを見た候補手（USI）。成りが任意なら成・不成の両方、強制なら成りだけ
    pub fn candidate_usis(view: &PlayerView) -> Vec<String> {
        let color = view.your_color;
        let mut out = vec![];
        for piece in &view.your_pieces {
            let Some(from) = parse_usi_square(&piece.square) else {
                continue;
            };
            for to in move_targets(&view.your_pieces, piece, color) {
                match promotion_choice(piece.role, from, to, color) {
                    Promotion::None => out.push(make_usi_move(from, to, false)),
                    Promotion::Optional => {
                        out.push(make_usi_move(from, to, false));
                        out.push(make_usi_move(from, to, true));
                    }
                    Promotion::Forced => out.push(make_usi_move(from, to, true)),
                }
            }
        }
        for role in HAND_ROLES {
            if view.your_hand.get(&role).copied().unwrap_or(0) == 0 {
                continue;
            }
            for to in drop_targets(&view.your_pieces, role, color) {
                out.extend(make_usi_drop(role, to));
            }
        }
        out
    }

    /// 旧マスク（`MASK_VERSION` 1）: 自駒視点の候補手から、その手番で反則済みの手だけを除く
    pub fn basic_legal_mask(view: &PlayerView, foul_tried: &HashSet<String>) -> Vec<bool> {
        let mut mask = vec![false; NUM_ACTIONS];
        for usi in candidate_usis(view) {
            if foul_tried.contains(&usi) {
                continue;
            }
            if let Some(a) = encode_usi(&usi, view.your_color) {
                mask[a] = true;
            }
        }
        mask
    }

    /// 行動マスク（長さ `NUM_ACTIONS`）。反則済みの手と、観測から反則が確定する手（`Deduction`）を除く。
    ///
    /// 除外で候補が尽きたときは旧マスクへ戻す（推論が健全なら、真の局面に合法手がある限り
    /// 尽きることはない。尽きる＝推論の前提が崩れているので、指せなくなるより安全側に倒す）
    pub fn legal_mask(view: &PlayerView, log: &ObservationLog, foul_tried: &HashSet<String>) -> Vec<bool> {
        let candidates = candidate_usis(view);
        let deduction = Deduction::new(view, log, foul_tried, &candidates.iter().cloned().collect());
        let mut mask = vec![false; NUM_ACTIONS];
        let mut basic = vec![false; NUM_ACTIONS];
        let mut any = false;
        for usi in candidates {
            if foul_tried.contains(&usi) {
                continue;
            }
            let Some(mv) = parse_usi(&usi) else { continue };
            let Some(a) = encode_move(&mv, view.your_color) else {
                continue;
            };
            basic[a] = true;
            if !deduction.rules_out(&usi, &mv) {
                mask[a] = true;
                any = true;
            }
        }
        if any { mask } else { basic }
    }

    /// 王手駒がいうるマスの1仮説。`dir` は玉から見た方向（桂は None）、`dist` は玉からの距離
    #[derive(Debug, Clone, Copy)]
    struct CheckerHyp {
        square: Coord,
        dir: Option<(i8, i8)>,
        dist: i8,
    }

    /// 観測から論理的に確定する反則。どれも**真の合法手を落とさない**ことだけを条件にしている
    /// （「たぶん反則」は落とさない。それはネットの領分）。
    ///
    /// 1. **反則した手の成り／不成の片割れ**: 両者の違いは移動後の駒種だけで、自玉が取られるか
    ///    （= 合法か）は盤上の占有にしか依らない。行き所の有無は候補生成が自駒視点で落とし済み
    /// 2. **相手の駒がいると確定したマス**（`occupied`）への打ちと、そこを飛び越える移動。確定の根拠は
    ///    - 直前に相手が駒を取ったマス（相手の着手駒がいまそこにいる）
    ///    - 王手中でないときの**歩以外の打ちの反則**: 打ちで自玉が危なくなることは無く、二歩・
    ///      行き所は候補生成が落とし済みなので、原因は「打ち先に相手の駒がいる」しかない
    ///      （歩は打ち歩詰めがありうるので使わない）
    ///    - 王手中でないときの**間が1マスの飛び越え反則**（下の 3 で原因が遮りと確定したもの）:
    ///      遮っている駒は間の1マスにしかいられない
    /// 3. **飛び越え反則の先**（`blocked_beyond`）: 王手中でないときに2マス以上の直線移動 F→T が
    ///    反則で、F の駒がピンされえない（玉の筋で最初の自駒でない、または移動方向がその筋と平行）
    ///    なら、原因は「F と T の間に相手の駒がいる」しかない（T に相手の駒がいれば駒取りで合法）。
    ///    同じ F から同じ方向へ T より先へ進む手も必ず遮られる
    /// 4. **王手中に王手を解消し得ない手**: 王手駒がいうるマスの集合 H を作り、H のどの仮説に
    ///    対しても解消しない手を落とす。H は
    ///    - 相手の直前の手が駒を取っていなければ: 玉から8方向に最初の自駒の手前までのマスと、
    ///      桂の王手マス（自駒がいないもの）。王手駒はこのどこかにいる
    ///    - 駒を取っていれば（取られたマス X）: 直前の自分の手は合法だった＝相手の手の前は王手が
    ///      無かったので、王手駒は「X に来た着手駒」か「着手駒の元のマス O が空いて通った
    ///      飛び駒（開き王手）」しかない。O は玉の筋の上の空きマスで、その先に飛び駒が入る余地が
    ///      あり、X へ1手で行ける位置（直線か桂跳び）に限られる。開き王手の飛び駒は O の先の
    ///      筋の上のマス
    ///
    ///    仮説 h に対して手が王手を解消し得ない条件:
    ///    - 玉以外の手: 着地点が h（駒取り。打ちは除く）でも、玉と h の間（合駒）でもない
    ///    - 玉の手: h が2マス以上離れた筋の上（＝その方向へ利く飛び駒）で、移動先が同じ筋の上
    ///      （h の側へ1マス、または玉の真後ろ）。玉が動くと元のマスが空くので真後ろも利きの中。
    ///      h が隣接・桂のときは駒種が分からないので玉の手は落とさない
    struct Deduction {
        twins: HashSet<String>,
        /// 相手の駒がいると確定したマス（直前に取られたマスを含む）
        occupied: HashSet<Coord>,
        /// 飛び越え反則した (元のマス, 方向, 距離)。同じ元・方向でこれより遠い手は遮られる
        blocked_beyond: Vec<(Coord, (i8, i8), i8)>,
        check: Option<(Coord, Vec<CheckerHyp>)>,
    }

    const ALL_DIRS: [(i8, i8); 8] = [
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
    ];

    fn on_board(c: Coord) -> bool {
        (1..=9).contains(&c.file) && (1..=9).contains(&c.rank)
    }

    fn step(c: Coord, (df, dr): (i8, i8), k: i8) -> Coord {
        Coord {
            file: c.file + df * k,
            rank: c.rank + dr * k,
        }
    }

    /// from→to が直線の移動で、途中（両端を除く）に square を通るか
    fn passes_through(from: Coord, to: Coord, square: Coord) -> bool {
        let (df, dr) = (to.file - from.file, to.rank - from.rank);
        if (df == 0 && dr == 0) || (df != 0 && dr != 0 && df.abs() != dr.abs()) {
            return false;
        }
        let unit = (df.signum(), dr.signum());
        let n = df.abs().max(dr.abs());
        (1..n).any(|k| step(from, unit, k) == square)
    }

    /// 何かの駒が o から x へ1手で行けうるか（直線か桂跳び。駒種と遮りは見ない＝上界）
    fn reachable(o: Coord, x: Coord) -> bool {
        let (df, dr) = (x.file - o.file, x.rank - o.rank);
        if df == 0 && dr == 0 {
            return false;
        }
        df == 0 || dr == 0 || df.abs() == dr.abs() || (df.abs() == 1 && dr.abs() == 2)
    }

    impl Deduction {
        /// `candidates` は自駒視点の候補手（`candidate_usis`）。反則から推論するのは候補手の反則だけ
        /// （候補外の反則は原因が成りの選択や行き所でありうるので、占有・遮りの根拠にしない）
        fn new(
            view: &PlayerView,
            log: &ObservationLog,
            foul_tried: &HashSet<String>,
            candidates: &HashSet<String>,
        ) -> Self {
            let twins = foul_tried
                .iter()
                .filter_map(|usi| match parse_usi(usi)? {
                    ShogiMove::Board { from, to, promote } => Some(make_usi_move(from, to, !promote)),
                    ShogiMove::Drop { .. } => None,
                })
                .collect();

            // 直前の相手の手（自分の最後の受理手より後にあるもの）が取ったマス
            let mut known_opponent = None;
            for event in log.events().iter().rev() {
                match event {
                    Observation::OpponentMoved {
                        captured_my_piece_at,
                        ..
                    } => {
                        known_opponent = captured_my_piece_at.as_deref().and_then(parse_usi_square);
                        break;
                    }
                    Observation::MyMove { .. } => break,
                    _ => {}
                }
            }

            // 動けないと確定している相手の歩（deduce::immobile_opponent_pawns）は使わない:
            // ログを毎回先頭から辿るのでマスクのコストが倍になる割に、記録483局で反則2件にしか効かない
            let mut occupied: HashSet<Coord> = known_opponent.into_iter().collect();

            let mut blocked_beyond = vec![];
            if !view.you_in_check {
                let king = view
                    .your_pieces
                    .iter()
                    .find(|p| p.role == Role::King)
                    .and_then(|p| parse_usi_square(&p.square));
                let own: HashSet<Coord> = view
                    .your_pieces
                    .iter()
                    .filter_map(|p| parse_usi_square(&p.square))
                    .collect();
                // f が玉の筋で最初の自駒なら、その筋の方向（＝ピンされうる向き）
                let pin_dir = |f: Coord| -> Option<(i8, i8)> {
                    let k = king?;
                    ALL_DIRS.iter().copied().find(|&d| {
                        let mut c = step(k, d, 1);
                        while on_board(c) && !own.contains(&c) {
                            c = step(c, d, 1);
                        }
                        c == f
                    })
                };
                for usi in foul_tried.iter().filter(|u| candidates.contains(*u)) {
                    match parse_usi(usi) {
                        Some(ShogiMove::Drop { role, to }) if role != Role::Pawn => {
                            occupied.insert(to);
                        }
                        Some(ShogiMove::Board { from, to, .. }) if king.is_some() => {
                            let (df, dr) = (to.file - from.file, to.rank - from.rank);
                            let dist = df.abs().max(dr.abs());
                            if dist < 2 || (df != 0 && dr != 0 && df.abs() != dr.abs()) {
                                continue;
                            }
                            let unit = (df.signum(), dr.signum());
                            let pinnable = pin_dir(from)
                                .is_some_and(|d| d != unit && d != (-unit.0, -unit.1));
                            if pinnable {
                                continue;
                            }
                            blocked_beyond.push((from, unit, dist));
                            if dist == 2 {
                                occupied.insert(step(from, unit, 1));
                            }
                        }
                        _ => {}
                    }
                }
            }

            let check = if view.you_in_check {
                Self::checker_hyps(view, known_opponent)
            } else {
                None
            };
            Deduction {
                twins,
                occupied,
                blocked_beyond,
                check,
            }
        }

        /// 王手駒がいうるマスの集合（上の doc の H）。前提が崩れている（空になる）ときは None
        fn checker_hyps(view: &PlayerView, captured_at: Option<Coord>) -> Option<(Coord, Vec<CheckerHyp>)> {
            let own: HashSet<Coord> = view
                .your_pieces
                .iter()
                .filter_map(|p| parse_usi_square(&p.square))
                .collect();
            let king = view
                .your_pieces
                .iter()
                .find(|p| p.role == Role::King)
                .and_then(|p| parse_usi_square(&p.square))?;

            // 8方向の「最初の自駒の手前まで」
            let rays: Vec<((i8, i8), Vec<Coord>)> = ALL_DIRS
                .iter()
                .map(|&d| {
                    let mut squares = vec![];
                    let mut c = step(king, d, 1);
                    while on_board(c) && !own.contains(&c) {
                        squares.push(c);
                        c = step(c, d, 1);
                    }
                    (d, squares)
                })
                .collect();
            // 相手の桂が玉を狙える位置（相手から見て前へ跳んでくる）
            let forward = if view.your_color == Color::Sente { -2 } else { 2 };
            let knights: Vec<Coord> = [1, -1]
                .iter()
                .map(|&df| Coord {
                    file: king.file + df,
                    rank: king.rank + forward,
                })
                .filter(|&c| on_board(c) && !own.contains(&c))
                .collect();

            let ray_hyp = |d: (i8, i8), i: usize, c: Coord| CheckerHyp {
                square: c,
                dir: Some(d),
                dist: i as i8 + 1,
            };
            let knight_hyp = |c: Coord| CheckerHyp {
                square: c,
                dir: None,
                dist: 0,
            };

            let mut hyps = vec![];
            match captured_at {
                None => {
                    for (d, squares) in &rays {
                        hyps.extend(squares.iter().enumerate().map(|(i, &c)| ray_hyp(*d, i, c)));
                    }
                    hyps.extend(knights.iter().map(|&c| knight_hyp(c)));
                }
                Some(x) => {
                    // 着手駒そのものの王手
                    for (d, squares) in &rays {
                        if let Some(i) = squares.iter().position(|&c| c == x) {
                            hyps.push(ray_hyp(*d, i, x));
                        }
                    }
                    if knights.contains(&x) {
                        hyps.push(knight_hyp(x));
                    }
                    // 開き王手: 元のマス o の先の筋の上
                    for (d, squares) in &rays {
                        let opens = squares
                            .iter()
                            .enumerate()
                            .take(squares.len().saturating_sub(1))
                            .find(|&(_, &o)| o != x && reachable(o, x));
                        if let Some((oi, _)) = opens {
                            hyps.extend(
                                squares
                                    .iter()
                                    .enumerate()
                                    .skip(oi + 1)
                                    .map(|(i, &c)| ray_hyp(*d, i, c)),
                            );
                        }
                    }
                }
            }
            if hyps.is_empty() { None } else { Some((king, hyps)) }
        }

        fn rules_out(&self, usi: &str, mv: &ShogiMove) -> bool {
            if self.twins.contains(usi) {
                return true;
            }
            let on_occupied = self.occupied.iter().any(|&x| match *mv {
                ShogiMove::Drop { to, .. } => to == x,
                ShogiMove::Board { from, to, .. } => passes_through(from, to, x),
            });
            if on_occupied {
                return true;
            }
            if let ShogiMove::Board { from, to, .. } = *mv {
                let beyond = self.blocked_beyond.iter().any(|&(f, unit, dist)| {
                    f == from && (2..=8).any(|k| k > dist && step(f, unit, k) == to)
                });
                if beyond {
                    return true;
                }
            }
            let Some((king, hyps)) = &self.check else {
                return false;
            };
            let (to, is_king, is_drop) = match *mv {
                ShogiMove::Board { from, to, .. } => (to, from == *king, false),
                ShogiMove::Drop { to, .. } => (to, false, true),
            };
            hyps.iter().all(|h| {
                if is_king {
                    if to == h.square {
                        return false;
                    }
                    match h.dir {
                        Some(d) if h.dist >= 2 => to == step(*king, d, 1) || to == step(*king, d, -1),
                        _ => false,
                    }
                } else {
                    if !is_drop && to == h.square {
                        return false;
                    }
                    let blocks = h.dir.is_some_and(|d| (1..h.dist).any(|k| step(*king, d, k) == to));
                    !blocks
                }
            })
        }
    }

}

/// 観測のテンソル化（`src/rl/encode.rs` の凍結時点の固定コピー）
pub mod encode {
    // 観測 → 入力テンソル（docs/rl-deepnash-design.md の「2. 観測エンコード」）。
    //
    // 入力は `Strategy::choose` と同じ `(PlayerView, ObservationLog, foul_tried)` だけ。
    // 学習中の環境も arena・本番の方策 Strategy もこの関数を通るので、学習時と推論時で
    // 特徴量が食い違わない。observation.rs にない情報（相手駒の位置など）は受け取れない。
    //
    // 出力は `[NUM_PLANES, 9, 9]` を平らにした f32 列（プレーン優先、マス番号は
    // `action::square_index`）。盤は手番側を先手向きに正規化する。

    use std::collections::HashSet;

    use crate::board::{Coord, parse_usi_square};
    use crate::model::GameModel;
    use crate::observation::{Observation, ObservationLog};
    use crate::protocol::{Color, PlayerView, Role};
    use super::action::{normalize, square_index};
    // 凍結時点の手数・反則の上限（入力の正規化に使う）
    const MAX_FOULS: u32 = 10;
    const MAX_PLIES: u32 = 200;
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

}

/// 手書きの推論（`src/rl/policy_net.rs` の凍結時点の固定コピー）
pub mod policy_net {
    // 方策・価値ネットの推論（手書きの forward pass。依存を増やさないため）。
    //
    // 学習は `~/Develop/tsuitate-nn/rnad/`（`model.py`）、重みは `export_weights.py` が
    // BatchNorm を畳み込みへ畳み込んで書き出す。形式は
    // `b"TSRLPV01" | u32 ヘッダ長 | ヘッダ JSON | f32 LE の並び`。
    //
    // 構成（`model.py` と同じ。層を変えたら両方と書き出しを直すこと）:
    // 3×3 畳み込み（入力 86 → C）→ 残差ブロック×B（3×3 畳み込み×2）→
    // 方策: 1×1（C→32）ReLU → 1×1（32→139）= [139, 9, 9] を平らにしたロジット、
    // 価値: 1×1（C→4）ReLU → 全結合 324→128 ReLU → 128→1 → tanh。
    // 盤は `[チャネル, 段, 筋]` の並び（`rl::encode` のプレーン・`rl::action` のマスと同じ）。

    use std::collections::HashMap;
    use std::io::Read;

    use super::action::{NUM_ACTION_KINDS, NUM_ACTIONS};
    use super::encode::{NUM_PLANES, OBS_LEN};

    const MAGIC: &[u8; 8] = b"TSRLPV01";
    const SQ: usize = 81;

    #[derive(Debug)]
    pub struct PolicyNet {
        channels: usize,
        blocks: usize,
        tensors: HashMap<String, Vec<f32>>,
    }

    #[derive(serde::Deserialize)]
    struct Header {
        planes: usize,
        channels: usize,
        blocks: usize,
        tensors: Vec<TensorInfo>,
    }

    #[derive(serde::Deserialize)]
    struct TensorInfo {
        name: String,
        shape: Vec<usize>,
    }

    impl PolicyNet {
        pub fn load(path: &str) -> Result<Self, String> {
            let mut bytes = vec![];
            std::fs::File::open(path)
                .and_then(|mut f| f.read_to_end(&mut bytes))
                .map_err(|e| format!("{path}: {e}"))?;
            Self::from_bytes(&bytes)
        }

        pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
            if bytes.len() < 12 || &bytes[..8] != MAGIC {
                return Err("方策ネットの重みファイルではない（magic 不一致）".into());
            }
            let hlen = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
            let header: Header = serde_json::from_slice(
                bytes.get(12..12 + hlen).ok_or("ヘッダが切れている")?,
            )
            .map_err(|e| e.to_string())?;
            if header.planes != NUM_PLANES {
                return Err(format!(
                    "入力プレーン数 {} が現行のエンコーダ {NUM_PLANES} と違う",
                    header.planes
                ));
            }
            let mut off = 12 + hlen;
            let mut tensors = HashMap::new();
            for t in &header.tensors {
                let n: usize = t.shape.iter().product();
                let raw = bytes
                    .get(off..off + 4 * n)
                    .ok_or_else(|| format!("{} のデータが切れている", t.name))?;
                let v: Vec<f32> = raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                tensors.insert(t.name.clone(), v);
                off += 4 * n;
            }
            if off != bytes.len() {
                return Err("末尾に余分なデータがある".into());
            }
            let net = PolicyNet {
                channels: header.channels,
                blocks: header.blocks,
                tensors,
            };
            net.check_shapes()?;
            Ok(net)
        }

        fn check_shapes(&self) -> Result<(), String> {
            let c = self.channels;
            let mut want: Vec<(String, usize)> = vec![
                ("stem.w".into(), c * NUM_PLANES * 9),
                ("stem.b".into(), c),
                ("p1.w".into(), 32 * c),
                ("p1.b".into(), 32),
                ("p2.w".into(), NUM_ACTION_KINDS * 32),
                ("p2.b".into(), NUM_ACTION_KINDS),
                ("v1.w".into(), 4 * c),
                ("v1.b".into(), 4),
                ("v_fc1.w".into(), 128 * 4 * SQ),
                ("v_fc1.b".into(), 128),
                ("v_fc2.w".into(), 128),
                ("v_fc2.b".into(), 1),
            ];
            for i in 0..self.blocks {
                for j in 1..=2 {
                    want.push((format!("block{i}.c{j}.w"), c * c * 9));
                    want.push((format!("block{i}.c{j}.b"), c));
                }
            }
            for (name, len) in want {
                match self.tensors.get(&name) {
                    Some(t) if t.len() == len => {}
                    Some(t) => return Err(format!("{name} の長さ {} が {len} と違う", t.len())),
                    None => return Err(format!("{name} が無い")),
                }
            }
            Ok(())
        }

        fn t(&self, name: &str) -> &[f32] {
            &self.tensors[name]
        }

        /// 観測（長さ `OBS_LEN`）→ (ロジット（長さ `NUM_ACTIONS`、マスク前）, 価値 ∈ [−1, 1])
        pub fn forward(&self, obs: &[f32]) -> (Vec<f32>, f32) {
            assert_eq!(obs.len(), OBS_LEN);
            let c = self.channels;
            let mut x = conv3x3(obs, NUM_PLANES, c, self.t("stem.w"), self.t("stem.b"));
            relu(&mut x);
            for i in 0..self.blocks {
                let mut y = conv3x3(&x, c, c, self.t(&format!("block{i}.c1.w")), self.t(&format!("block{i}.c1.b")));
                relu(&mut y);
                let y = conv3x3(&y, c, c, self.t(&format!("block{i}.c2.w")), self.t(&format!("block{i}.c2.b")));
                for (a, b) in x.iter_mut().zip(&y) {
                    *a = (*a + b).max(0.0);
                }
            }
            let mut p = conv1x1(&x, c, 32, self.t("p1.w"), self.t("p1.b"));
            relu(&mut p);
            let logits = conv1x1(&p, 32, NUM_ACTION_KINDS, self.t("p2.w"), self.t("p2.b"));
            debug_assert_eq!(logits.len(), NUM_ACTIONS);

            let mut v = conv1x1(&x, c, 4, self.t("v1.w"), self.t("v1.b"));
            relu(&mut v);
            let mut h = linear(&v, 128, self.t("v_fc1.w"), self.t("v_fc1.b"));
            relu(&mut h);
            let value = linear(&h, 1, self.t("v_fc2.w"), self.t("v_fc2.b"))[0].tanh();
            (logits, value)
        }
    }

    fn relu(x: &mut [f32]) {
        for v in x {
            *v = v.max(0.0);
        }
    }

    /// 3×3 畳み込み（パディング1）。`w` は [出力, 入力, 3, 3]
    fn conv3x3(x: &[f32], cin: usize, cout: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; cout * SQ];
        for co in 0..cout {
            let o = &mut out[co * SQ..(co + 1) * SQ];
            o.fill(b[co]);
            for ci in 0..cin {
                let src = &x[ci * SQ..(ci + 1) * SQ];
                for ky in 0..3 {
                    for kx in 0..3 {
                        let wv = w[((co * cin + ci) * 3 + ky) * 3 + kx];
                        if wv == 0.0 {
                            continue;
                        }
                        let (dy, dx) = (ky as isize - 1, kx as isize - 1);
                        let (y0, y1) = ((-dy).max(0) as usize, (9 - dy).min(9) as usize);
                        let (x0, x1) = ((-dx).max(0) as usize, (9 - dx).min(9) as usize);
                        for y in y0..y1 {
                            let sy = (y as isize + dy) as usize;
                            let orow = &mut o[y * 9 + x0..y * 9 + x1];
                            let srow = &src[sy * 9 + (x0 as isize + dx) as usize..];
                            for (ov, sv) in orow.iter_mut().zip(srow) {
                                *ov += wv * sv;
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// 1×1 畳み込み。`w` は [出力, 入力]
    fn conv1x1(x: &[f32], cin: usize, cout: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; cout * SQ];
        for co in 0..cout {
            let o = &mut out[co * SQ..(co + 1) * SQ];
            o.fill(b[co]);
            for ci in 0..cin {
                let wv = w[co * cin + ci];
                for (ov, sv) in o.iter_mut().zip(&x[ci * SQ..(ci + 1) * SQ]) {
                    *ov += wv * sv;
                }
            }
        }
        out
    }

    /// 全結合。`w` は [出力, 入力]
    fn linear(x: &[f32], nout: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
        let nin = x.len();
        (0..nout)
            .map(|o| b[o] + w[o * nin..(o + 1) * nin].iter().zip(x).map(|(a, b)| a * b).sum::<f32>())
            .collect()
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 埋め込んだ重みは凍結時点のもの() {
        use sha2::{Digest, Sha256};
        let got: String = Sha256::digest(WEIGHTS).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(got, WEIGHTS_SHA256);
        assert!(net().forward(&vec![0.0; encode::OBS_LEN]).0.len() == action::NUM_ACTIONS);
    }
}
