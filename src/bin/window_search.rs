//! 構成的な観測駆動探索ハーネス(診断・ベンチマーク専用。本番の Estimator の
//! サンプリングループとは独立したスタンドアロン実験ツール)。
//!
//! v1(ply-by-ply の全合法手BFS)から設計を変えた。ユーザーの実際の手動推論
//! (kakunari 65手目 1一角成の確定)を観察すると、人間は「何も明かされていない
//! 自由な区間」については一切候補を作らず、**何かが明かされた瞬間
//! (自駒が取られた・反則になった・王手宣言があった)だけ**、その事実を
//! 説明できる駒の動かし方を、駒の性質(テンポ・利き)から直接構成している。
//! 全合法手を1手ずつ試すブルートフォースBFSは、v1の実測で「バイアスは無いが
//! 何も明かされていない区間でも律儀に分岐してしまい状態が爆発する」ことが
//! わかった。
//!
//! この v2 は:
//! - 沈黙した(何も明かされない)相手の自由手が連続する区間は分岐せず、
//!   区間末尾の1つの明かされた事実(自駒が取られたマス)だけを見て、
//!   「そのマスにテンポ内で到達できる駒(mover候補)」を
//!   deduce.rs の空盤テンポ下限で直接列挙する
//! - もし直後に自分の駒(主に自玉)によるその マスへの取り返しが反則になって
//!   いれば、「そのマスは他の相手駒に守られている」という追加の強い制約が
//!   ある。守り駒(defender候補)も同様にテンポで列挙する(ユーザーの
//!   手動例の「桂馬が守りに回る/角が守りに回る」という組み立てを再現する)
//! - 手そのもの(経路の実在)は空盤テンポ下限による必要条件チェックであり、
//!   他の駒による経路遮蔽までは検証していない(deduce.rs 既存関数と同じ
//!   近似)。反対に、王手宣言との整合(自玉の位置と履歴)は
//!   route_square_refuted_by_check_history で厳密にチェックする
//! - 王手のみで捕獲を伴わない明かされ方(対象マスが特定できない)は
//!   mover/defender構成が使えないため、その区間だけ v1 方式(全合法手BFS)に
//!   フォールバックする
//!
//! 使い方: cargo run --release --bin window_search -- <kif> <窓終了ply>
//! (常に ply0=初期局面 起点。初期局面は100%正しいので v1 で問題だった
//! 「Estimatorのprefixに真の局面が含まれない」バイアス問題がそもそも起きない)

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use tsuitate_bot::board::{Coord, make_usi_square};
use tsuitate_bot::deduce::{all_distances_empty_board, min_moves_empty_board, piece_attacks_square};
use tsuitate_bot::kifu::{Kifu, RawFoul, parse_kif};
use tsuitate_bot::protocol::{Color, Role};
use tsuitate_bot::shogi::{Piece, Position, ShogiMove, parse_usi, promote_role, unpromote_role};

/// 片側の最大駒数。相手側の駒配置をヒープ確保なしの固定長配列で持つための上限
const MAX_OPP_PIECES: usize = 20;
const PAD: (Coord, Role) = (Coord { file: 0, rank: 0 }, Role::Pawn);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct OppState {
    board: [(Coord, Role); MAX_OPP_PIECES],
    board_len: u8,
    hand: [u8; 7],
}

const HAND_ROLES: [Role; 7] = [
    Role::Pawn,
    Role::Lance,
    Role::Knight,
    Role::Silver,
    Role::Gold,
    Role::Bishop,
    Role::Rook,
];

fn to_opp_state(pos: &Position, opp: Color) -> OppState {
    let mut list: Vec<(Coord, Role)> = pos
        .pieces()
        .filter(|(_, p)| p.color == opp)
        .map(|(sq, p)| (sq, p.role))
        .collect();
    list.sort_by_key(|(sq, _)| (sq.file, sq.rank));
    assert!(list.len() <= MAX_OPP_PIECES, "相手駒数が上限を超えています");
    let mut board = [PAD; MAX_OPP_PIECES];
    board[..list.len()].copy_from_slice(&list);
    let mut hand = [0u8; 7];
    for (i, role) in HAND_ROLES.iter().enumerate() {
        hand[i] = pos.hand_count(opp, *role);
    }
    OppState {
        board,
        board_len: list.len() as u8,
        hand,
    }
}

fn combine(my_side: &Position, opp_state: &OppState, opp: Color, turn: Color) -> Position {
    let mut pos = my_side.clone();
    for &(sq, role) in &opp_state.board[..opp_state.board_len as usize] {
        pos.set(sq, Some(Piece { color: opp, role }));
    }
    for (i, role) in HAND_ROLES.iter().enumerate() {
        pos.set_hand(opp, *role, opp_state.hand[i]);
    }
    pos.set_turn(turn);
    pos
}

/// board[idx] を (new_sq, new_role) に差し替え、ソート順を保って返す
fn place_piece(opp_state: &OppState, idx: usize, new_sq: Coord, new_role: Role) -> OppState {
    let len = opp_state.board_len as usize;
    let mut list: Vec<(Coord, Role)> = opp_state.board[..len].to_vec();
    list[idx] = (new_sq, new_role);
    list.sort_by_key(|(sq, _)| (sq.file, sq.rank));
    let mut st = *opp_state;
    st.board[..len].copy_from_slice(&list);
    st
}

fn is_promoted_role(role: Role) -> bool {
    role != unpromote_role(role)
}

/// (未成/既成)を考慮したテンポ下限。既に成っている駒は、それ以上成れない
/// (= 遷移なしの素のBFS)ものとして扱う
fn tempo_to(role: Role, color: Color, from: Coord, to: Coord, arrive_promoted: bool) -> Option<u32> {
    if is_promoted_role(role) {
        if !arrive_promoted {
            return None; // 既に成っている駒が未成で終わることはない
        }
        min_moves_empty_board(role, color, from, to, false)
    } else {
        min_moves_empty_board(role, color, from, to, arrive_promoted)
    }
}

/// target に budget 手以内で到達できる相手駒(mover候補)を列挙する。
/// required_role が Some なら、着地後の駒種(成りを剥がした基本形)がそれと
/// 一致するものだけを候補にする(自分の捕獲は取った駒種まで厳密に分かるため。
/// 相手に取られた側は取った駒種が分からないので None を渡す)。
/// 戻り値は (差し替え後の OppState, その駒が着地までに使ったテンポ, 着地後の駒のindex)
fn candidate_movers(
    opp_state: &OppState,
    opp: Color,
    target: Coord,
    budget: u32,
    required_role: Option<Role>,
) -> Vec<(OppState, u32, usize)> {
    let mut out = vec![];
    let len = opp_state.board_len as usize;
    for i in 0..len {
        let (sq0, role0) = opp_state.board[i];
        // sq0 == target(0手で捕獲された = 自由手を1つも使わず元々そこにいた)は
        // 有効な候補。tempo=0 として自然に扱われる(除外しない)
        if required_role.is_none_or(|r| r == unpromote_role(role0)) {
            if let Some(d) = tempo_to(role0, opp, sq0, target, false) {
                if d <= budget {
                    let st = place_piece(opp_state, i, target, role0);
                    let idx = st.board[..len]
                        .iter()
                        .position(|&(sq, _)| sq == target)
                        .unwrap();
                    out.push((st, d, idx));
                }
            }
        }
        if !is_promoted_role(role0) && required_role.is_none_or(|r| r == role0) {
            if let Some(pr) = promote_role(role0) {
                if let Some(d) = tempo_to(role0, opp, sq0, target, true) {
                    if d <= budget {
                        let st = place_piece(opp_state, i, target, pr);
                        let idx = st.board[..len]
                            .iter()
                            .position(|&(sq, _)| sq == target)
                            .unwrap();
                        out.push((st, d, idx));
                    }
                }
            }
        }
    }
    out
}

/// attacked_sq に利く位置(budget手以内で到達でき、かつ着地点から attacked_sq に
/// 利く)へ移動できる相手駒を列挙する。exclude_idx があればその駒は候補から除く
/// (捕獲した駒自身を守る「別の」駒を探す defender 用途)。
/// 捕獲を伴わない王手(対象マスが定まらない)の場合は exclude_idx=None、
/// attacked_sq=自玉の位置 で呼べば「王手を作れる駒」の構成的列挙になる
/// 駒種ごとに「attacked_sq に利くマス集合」を求める(81マス全探索は駒種の
/// 組み合わせだけ = 高々14通りぶんで済む。状態にも駒の実位置にも依らないので
/// 呼び出し側で使い回せる)
fn attacking_squares_by_role(opp: Color, attacked_sq: Coord) -> HashMap<Role, Vec<Coord>> {
    const ALL_ROLES: [Role; 14] = [
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
    let mut map = HashMap::new();
    for role in ALL_ROLES {
        let mut v = vec![];
        for file in 1..=9i8 {
            for rank in 1..=9i8 {
                let sq = Coord { file, rank };
                if sq != attacked_sq && piece_attacks_square(role, opp, sq, attacked_sq) {
                    v.push(sq);
                }
            }
        }
        map.insert(role, v);
    }
    map
}

/// attacked_sq に利く位置(budget手以内で到達でき、かつ着地点から attacked_sq に
/// 利く)へ移動できる相手駒を列挙する。exclude_idx があればその駒は候補から除く
/// (捕獲した駒自身を守る「別の」駒を探す defender 用途)。
/// 捕獲を伴わない王手(対象マスが定まらない)の場合は exclude_idx=None、
/// attacked_sq=自玉の位置 で呼べば「王手を作れる駒」の構成的列挙になる。
/// attack_squares は attacking_squares_by_role の結果を使い回す(同じ
/// attacked_sq に対する複数呼び出しで81マス全探索を繰り返さないため)
fn candidate_attackers(
    opp_state: &OppState,
    exclude_idx: Option<usize>,
    opp: Color,
    budget: u32,
    attack_squares: &HashMap<Role, Vec<Coord>>,
) -> Vec<OppState> {
    let mut out = vec![];
    let len = opp_state.board_len as usize;
    for i in 0..len {
        if Some(i) == exclude_idx {
            continue;
        }
        let (sq0, role0) = opp_state.board[i];
        let Some(squares) = attack_squares.get(&role0) else {
            continue;
        };
        // 1回のBFSで全マスへの距離をまとめて求め、個別にBFSするより速くする。
        // 成りを考慮しない簡略化(role0 のまま移動する距離だけを見る。既に成っている
        // 駒は all_distances_empty_board 内部でそれ以上成れない扱いになり、
        // その移動は false 側のキーに入る)
        let dists = all_distances_empty_board(role0, opp, sq0);
        for &sq in squares {
            // sq==sq0(0手、動かず元から利いていた=発見王手で動いたのは別の駒)
            // も有効な候補として許す
            let Some(&d) = dists.get(&(sq, false)) else {
                continue;
            };
            if d <= budget {
                out.push(place_piece(opp_state, i, sq, role0));
            }
        }
    }
    out
}

/// opp_state から target の駒を取り除いた状態を作る(自分がそこを捕獲した後、
/// を組み立てるのに使う)
fn remove_piece_at(opp_state: &OppState, target: Coord) -> OppState {
    let len = opp_state.board_len as usize;
    let list: Vec<(Coord, Role)> = opp_state.board[..len]
        .iter()
        .copied()
        .filter(|&(sq, _)| sq != target)
        .collect();
    let mut st = *opp_state;
    st.board = [PAD; MAX_OPP_PIECES];
    st.board[..list.len()].copy_from_slice(&list);
    st.board_len = list.len() as u8;
    st
}

fn is_defended(pos: &Position, opp: Color, target: Coord) -> bool {
    pos.pieces()
        .any(|(sq, p)| p.color == opp && sq != target && pos.attacks(sq, target))
}

/// 1つの「沈黙した自由手の区間 + 末尾の捕獲イベント」を構成的に解決する。
/// (王手宣言との整合は呼び出し側が結果に対して事後フィルタする — 自分の捕獲/
/// 相手の捕獲で「王手後」の局面の組み立て方が違うため)
fn resolve_capture_stretch(
    states: &HashSet<OppState>,
    my_side: &Position,
    opp: Color,
    target: Coord,
    budget: u32,
    needs_defense: bool,
    required_role: Option<Role>,
) -> HashSet<OppState> {
    let mut result = HashSet::new();
    // target に利くマス集合は駒種ごとに1回だけ求める(全 mv_state で使い回す)
    let attack_squares = needs_defense.then(|| attacking_squares_by_role(opp, target));
    for st in states {
        for (mv_state, tempo_used, mover_idx) in
            candidate_movers(st, opp, target, budget, required_role)
        {
            if !needs_defense {
                result.insert(mv_state);
                continue;
            }
            let check_pos = combine(my_side, &mv_state, opp, opp);
            if is_defended(&check_pos, opp, target) {
                result.insert(mv_state);
            }
            let remaining = budget.saturating_sub(tempo_used);
            if remaining > 0 {
                for def_state in candidate_attackers(
                    &mv_state,
                    Some(mover_idx),
                    opp,
                    remaining,
                    attack_squares.as_ref().unwrap(),
                ) {
                    result.insert(def_state);
                }
            }
        }
    }
    result
}

fn resolve_foul(pos: &Position, f: &RawFoul) -> String {
    match f {
        RawFoul::Drop { role, to } => {
            tsuitate_bot::board::make_usi_drop(*role, *to).expect("打てない駒種の反則試行")
        }
        RawFoul::Board { from, to, role } => {
            let piece = pos.piece_at(*from).expect("反則試行の移動元に駒がない");
            let piece_promoted = piece.role != unpromote_role(piece.role);
            let code_promoted = *role != unpromote_role(*role);
            tsuitate_bot::board::make_usi_move(*from, *to, code_promoted && !piece_promoted)
        }
    }
}

struct RawFact {
    side: Color,
    foul_usis: Vec<String>,
    usi: String,
    captured: Option<Role>,
    captured_sq: Option<Coord>,
    gives_check: bool,
}

fn replay_raw(kifu: &Kifu) -> Vec<RawFact> {
    let mut pos = Position::initial();
    let mut facts = Vec::with_capacity(kifu.plies.len());
    for ply_data in &kifu.plies {
        let side = pos.turn();
        let foul_usis: Vec<String> = ply_data
            .fouls
            .iter()
            .map(|f| resolve_foul(&pos, f))
            .collect();
        let usi = ply_data.mv.to_usi();
        let mv = parse_usi(&usi).expect("USI解析失敗");
        let captured_role = pos.play_unchecked(&mv).map(unpromote_role);
        let captured_sq = match (captured_role, mv) {
            (Some(_), ShogiMove::Board { to, .. }) => Some(to),
            _ => None,
        };
        let gives_check = pos.in_check(pos.turn());
        facts.push(RawFact {
            side,
            foul_usis,
            usi,
            captured: captured_role,
            captured_sq,
            gives_check,
        });
    }
    facts
}

enum PlyFact {
    Mine {
        foul_usis: Vec<String>,
        usi: String,
        captured: Option<Role>,
        gives_check: bool,
    },
    Theirs {
        captured_at: Option<Coord>,
        gives_check: bool,
    },
}

fn classify(raw: &[RawFact], my_color: Color) -> Vec<PlyFact> {
    raw.iter()
        .map(|f| {
            if f.side == my_color {
                PlyFact::Mine {
                    foul_usis: f.foul_usis.clone(),
                    usi: f.usi.clone(),
                    captured: f.captured,
                    gives_check: f.gives_check,
                }
            } else {
                PlyFact::Theirs {
                    captured_at: f.captured_sq,
                    gives_check: f.gives_check,
                }
            }
        })
        .collect()
}

/// facts[..upto] だけから自分側(盤面+持ち駒)を再構成する
fn compute_my_side(facts: &[PlyFact], my_color: Color) -> Position {
    let mut pos = Position::initial();
    let opp = my_color.other();
    let opp_squares: Vec<Coord> = pos
        .pieces()
        .filter(|(_, p)| p.color == opp)
        .map(|(sq, _)| sq)
        .collect();
    for sq in opp_squares {
        pos.set(sq, None);
    }
    for fact in facts {
        match fact {
            PlyFact::Mine { usi, captured, .. } => {
                let mv = parse_usi(usi).expect("USI解析失敗");
                match mv {
                    ShogiMove::Board { from, to, promote } => {
                        let mut piece = pos.piece_at(from).expect("自分の駒がない");
                        if promote {
                            if let Some(pr) = promote_role(piece.role) {
                                piece.role = pr;
                            }
                        }
                        pos.set(from, None);
                        pos.set(to, Some(piece));
                    }
                    ShogiMove::Drop { role, to } => {
                        let h = pos.hand_count(my_color, role);
                        pos.set_hand(my_color, role, h.saturating_sub(1));
                        pos.set(
                            to,
                            Some(Piece {
                                color: my_color,
                                role,
                            }),
                        );
                    }
                }
                if let Some(r) = captured {
                    pos.set_hand(my_color, *r, pos.hand_count(my_color, *r) + 1);
                }
            }
            PlyFact::Theirs { captured_at, .. } => {
                if let Some(sq) = captured_at {
                    pos.set(*sq, None);
                }
            }
        }
    }
    pos
}

/// facts[reveal_idx] の直後(facts[reveal_idx+1])が、my_color の玉による
/// target への反則(取り返し試行が非合法)かどうか
fn detects_king_recapture_foul(
    facts: &[PlyFact],
    reveal_idx: usize,
    my_color: Color,
    my_side_at_reveal: &Position,
    target: Coord,
) -> bool {
    let Some(PlyFact::Mine { foul_usis, .. }) = facts.get(reveal_idx + 1) else {
        return false;
    };
    let king_sq = my_side_at_reveal.king_square(my_color);
    foul_usis.iter().any(|u| {
        matches!(parse_usi(u), Some(ShogiMove::Board { from, to, .. })
            if Some(from) == king_sq && to == target)
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args
        .first()
        .cloned()
        .unwrap_or_else(|| "scenarios/kakunari.kif".into());
    let win_end: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(65);

    let text = std::fs::read_to_string(&path).expect("kif が読めません");
    let kifu = parse_kif(&text).expect("kif のパース失敗");
    let raw = replay_raw(&kifu);
    assert!(win_end <= raw.len(), "窓終了plyが手数範囲を超えています");

    let decision_ply: usize = kifu
        .directives
        .get("ply")
        .and_then(|s| s.parse().ok())
        .unwrap_or(win_end);
    let my_color = raw
        .get(decision_ply)
        .map(|f| f.side)
        .unwrap_or_else(|| raw[win_end.min(raw.len() - 1)].side);
    let opp = my_color.other();
    let facts = classify(&raw, my_color);

    println!("{path}: ply0(初期局面)起点 → {win_end}手目 (my_color={my_color:?})");

    // ply0は初期局面そのもの: バイアスなしで100%正しい唯一の開始点
    let mut states: HashSet<OppState> = HashSet::new();
    states.insert(to_opp_state(&Position::initial(), opp));

    let t0 = Instant::now();
    let mut ply = 0usize;
    // 相手の総手数(リセットしない)。ある反応イベントで特定した駒以外の駒も、
    // それまでの全ての沈黙手を自由に使えたはず(「全体手数会計」を厳密にはやらない
    // 近似だが、区間ごとにリセットすると「特定した駒だけが全予算を使い、
    // 他の駒は一切動いていない」という誤った前提になってしまう。緩めの予算に
    // することで候補を取りこぼさない側に倒す。列挙は「駒の種類×成不成」の
    // 組み合わせなので、予算が緩くても状態数は経路探索のようには爆発しない)
    let mut opp_moves_so_far = 0u32;
    while ply < win_end {
        if std::env::var("WS_TRACE").is_ok() {
            eprintln!("    [trace] {}手目 処理開始 (状態数 {})", ply + 1, states.len());
        }
        match &facts[ply] {
            // 自分の手が相手駒を捕獲 → 取った駒種・マスとも厳密に分かる
            // 「明かされた事実」。相手の捕獲より強い制約(駒種まで確定)
            PlyFact::Mine {
                usi,
                captured: Some(captured_role),
                gives_check,
                ..
            } => {
                let mv = parse_usi(usi).expect("USI解析失敗");
                let ShogiMove::Board { to, .. } = mv else {
                    unreachable!("捕獲は打ちでは起きない")
                };
                let my_side_before = compute_my_side(&facts[..ply], my_color);
                let candidates = resolve_capture_stretch(
                    &states,
                    &my_side_before,
                    opp,
                    to,
                    opp_moves_so_far,
                    false,
                    Some(*captured_role),
                );
                // 王手宣言との整合を事後フィルタ: 捕獲後(相手の駒を除去し、
                // 自分の駒がtoへ移動済み)の局面で相手玉が王手されているか
                let my_side_after = compute_my_side(&facts[..ply + 1], my_color);
                let gc = *gives_check;
                states = candidates
                    .into_iter()
                    .map(|st| remove_piece_at(&st, to))
                    .filter(|st| combine(&my_side_after, st, opp, opp).in_check(opp) == gc)
                    .collect();
                println!(
                    "  {}手目(自分が{captured_role:?}を@{}で捕獲): 構成的候補 {}個 (相手の総手数{opp_moves_so_far})",
                    ply + 1,
                    make_usi_square(to),
                    states.len()
                );
                ply += 1;
            }
            // 捕獲を伴わない自分の手。近似として、相手側の状態には触れず
            // (経路遮蔽・反則整合の検証は行わない。既知の限界)そのまま進める
            PlyFact::Mine { .. } => {
                // 相手の総手数には影響しない(こちらの手番なので)
                ply += 1;
            }
            PlyFact::Theirs {
                captured_at: None,
                gives_check: false,
            } => {
                // 沈黙: まだ何も明かされていない。分岐せず数えるだけ
                opp_moves_so_far += 1;
                ply += 1;
            }
            PlyFact::Theirs {
                captured_at: Some(target),
                gives_check,
            } => {
                let target = *target;
                let gc = *gives_check;
                opp_moves_so_far += 1;
                let budget = opp_moves_so_far;
                // 捕獲後(自分の駒が取られた後)の自分側で組み立てる。取られた駒が
                // 自玉への他の駒の利きを遮っていた場合の王手判定にも影響するため
                let my_side_after = compute_my_side(&facts[..ply + 1], my_color);
                let needs_defense =
                    detects_king_recapture_foul(&facts, ply, my_color, &my_side_after, target);
                let candidates = resolve_capture_stretch(
                    &states,
                    &my_side_after,
                    opp,
                    target,
                    budget,
                    needs_defense,
                    None,
                );
                states = candidates
                    .into_iter()
                    .filter(|st| combine(&my_side_after, st, opp, my_color).in_check(my_color) == gc)
                    .collect();
                println!(
                    "  {}手目(相手が@{}で捕獲): 構成的候補 {}個 (相手の総手数{budget}, 守り要求={needs_defense})",
                    ply + 1,
                    make_usi_square(target),
                    states.len()
                );
                ply += 1;
            }
            PlyFact::Theirs { .. } => {
                // 捕獲を伴わない王手(対象マスは自玉の位置)。「自玉に利く位置へ
                // budget手以内で到達できる駒」を構成的に列挙する(discovered check
                // = 動いたのは別の駒、の場合も候補に含む。捕獲がないので
                // required_role は無し)
                opp_moves_so_far += 1;
                let budget = opp_moves_so_far;
                let my_side = compute_my_side(&facts[..ply + 1], my_color);
                let king_sq = my_side
                    .king_square(my_color)
                    .expect("自玉が盤上にいない");
                let attack_squares = attacking_squares_by_role(opp, king_sq);
                let mut next = HashSet::new();
                for st in &states {
                    // 既にその駒がいる位置から動かず利いている(発見王手)候補も
                    // candidate_attackers に含まれる(tempo=0)
                    for cand in candidate_attackers(st, None, opp, budget, &attack_squares) {
                        next.insert(cand);
                    }
                }
                states = next;
                println!(
                    "  {}手目(王手のみ、構成的候補): {}個 (相手の総手数{budget})",
                    ply + 1,
                    states.len()
                );
                ply += 1;
            }
        }
        if std::env::var("WS_TRACE").is_ok() {
            let kind = match &facts[ply - 1] {
                PlyFact::Mine { usi, captured, .. } => format!("Mine({usi}, cap={captured:?})"),
                PlyFact::Theirs {
                    captured_at,
                    gives_check,
                } => format!("Theirs(cap_at={captured_at:?}, chk={gives_check})"),
            };
            println!("    [trace] {}手目({kind})後: 状態数 {}", ply, states.len());
        }
        if states.is_empty() {
            println!("矛盾(0局面)で探索終了しました({}手目)。", ply);
            break;
        }
    }
    let elapsed = t0.elapsed();
    println!();
    println!(
        "探索完了: 終端局面 {}個, 所要時間 {:.2}秒",
        states.len(),
        elapsed.as_secs_f64()
    );

    let mut truth_pos = Position::initial();
    for ply_data in kifu.plies.iter().take(win_end) {
        let mv = parse_usi(&ply_data.mv.to_usi()).expect("USI解析失敗");
        truth_pos.play_unchecked(&mv);
    }
    let found_truth = states.contains(&to_opp_state(&truth_pos, opp));
    println!("真の局面が終端集合に含まれるか: {found_truth}");
}
