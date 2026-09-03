//! 戦力投資の限界効用の**共有定義**（issue #40）。
//!
//! 持ち駒の打ち・成りが「まだ価値が増える場所」へ向いているかを、着手前後の
//! **自駒だけ**の利き枚数マップの差分（しきい値横断）× 需要帯で数える。
//! `check_prep.rs` / `check_economy.rs` と同じ位置づけで、**runtime には一切
//! 入らない**（評価も推定も触らない。P0 の帯×遷移表は records からオフラインで
//! 計算する）。消費者は `bin/investment_probe`（P0-2 の頻度・分類）と、
//! P0 を通過した場合の P1 実装。
//!
//! # 設計の固定点（issue #40 本文の事前登録）
//!
//! - 増減は**しきい値横断の本数**で数える（`<1→≥1` = first / `<2→≥2` = second /
//!   `<3→≥3` = redundant）。`0→2` は first+second、`3→1` は
//!   redundant_loss+second_loss になり加法分解が一意になる。**利き枚数は3で飽和**
//!   （`3→4` は数えない）
//! - **帯マップは決定開始時（着手前）の状態から手番ごとに1回だけ作り**、その
//!   手番の全候補・全増減（gain 側も loss 側も）で共有する。着手した駒自身が
//!   需要帯を作って自分のペナルティを免れる構造をここで遮断する
//! - 帯の優先順位は own_king > opp_king > backed_target > active_own_piece >
//!   neutral で**排他的**に割る
//!
//! # 利き枚数 A(s) の規約
//!
//! - 数えるのは**玉を含む全自駒**の利き（issue #40 の事前登録は
//!   「自駒利き枚数」で、玉を除く但し書きは無い。runtime の
//!   `own_attack_counts`（`landing_def` 用）と同じ側。PR #41 レビューで
//!   玉除外の初版を訂正した — 玉が既に利くマスへの追加が first でなく
//!   second になるので、表現そのものが変わる）
//! - 利きは `board::defend_targets` と同じ規約: **自駒の乗ったマスも含み**、
//!   飛び利きはそこで止まる。自駒マスの枚数 = その駒の紐の本数
//! - 飛び利きは自駒にしか遮られない**楽観上限**（相手の駒は見えない）
//!
//! **「正の link を得る」の判定**（link-only drop の分類）は本数の近似ではなく
//! runtime の `link` 項と同じ値（`strategy::linked_value_of` = 働き重み込みの
//! `linked_value`。オフラインでは粒子由来の敵玉重みだけ None）の増分で行う。

use crate::belief_features::sq_index;
use crate::board::{Coord, defend_targets, parse_usi_square};
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{Color, Role, VisiblePiece};
use crate::shogi::{ShogiMove, parse_usi};

/// 需要帯。値は優先順位（小さいほど強い）そのもの
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub enum Band {
    /// 自玉のチェビシェフ距離2以内
    OwnKing = 0,
    /// `deduce::opp_king_candidates` が [`OPP_KING_GATE`] マス以下のときの
    /// 候補集合の和集合とそのチェビシェフ距離1以内
    OppKing = 1,
    /// 観測に裏付けられた相手占有マスそのもの（[`backed_targets`]）
    BackedTarget = 2,
    /// 決定開始時の自駒のうち、利きが帯1〜3へ届いている駒のマス
    ActiveOwnPiece = 3,
    /// 上のどれでもない
    Neutral = 4,
}

pub const N_BANDS: usize = 5;
pub const BAND_TAGS: [&str; N_BANDS] =
    ["own_king", "opp_king", "backed_target", "active_own_piece", "neutral"];
pub const BAND_LABELS: [&str; N_BANDS] = ["自玉圏", "敵玉圏", "裏付け標的", "働く自駒", "中立"];

/// しきい値横断は 1 / 2 / 3 の3段で**飽和**する（`3→4` は数えない。事前登録）
pub const N_TRANSITIONS: usize = 3;
pub const TRANSITION_TAGS: [&str; N_TRANSITIONS] = ["first", "second", "redundant"];

/// opp_king 帯の発火ゲート（候補マス数）。`king_cand_attack_w` の既定ゲートと
/// 同じ 20 だが、**config を読まない事前登録の定数**（診断が env で動くと
/// 「同じ記録から同じ表」が壊れる）
pub const OPP_KING_GATE: usize = 20;

/// **玉を含む全自駒**の利き枚数マップ（`sq_index` 順の 81 要素）。
///
/// `defend_targets` と同じ規約（自駒マスを含む・飛び利きはそこで止まる・
/// 楽観上限）。1枚の駒が同じマスへ複数経路で利いても1と数える。
pub fn attack_counts(pieces: &[VisiblePiece], color: Color) -> [u8; 81] {
    let mut n = [0u8; 81];
    for p in pieces.iter() {
        let mut seen = [false; 81];
        for s in defend_targets(pieces, p, color) {
            let i = sq_index(s);
            if !seen[i] {
                seen[i] = true;
                n[i] = n[i].saturating_add(1);
            }
        }
    }
    n
}

/// 1手を適用した後の自駒利き枚数マップ。
/// 着手駒だけでなく、ブロッカーの移動で開閉した飛び利きも含む盤全体を作り直す
pub fn attack_counts_after(pieces: &[VisiblePiece], mv: &ShogiMove, color: Color) -> [u8; 81] {
    attack_counts(&crate::check_prep::pieces_after(pieces, mv), color)
}

/// 観測に裏付けられた相手占有マス（needs帯3 = backed_target の由来）。
///
/// `own_zone_capture_w` v2 の観測裏付け（`strategy::opp_occupancy_evidence`）と
/// **同じ定義**で、runtime 側がこの関数へ委譲することで同一性を保つ:
/// - 相手が自駒を取ったマス（`OpponentMoved.captured_my_piece_at`）
/// - **今手番**の非歩の打ち反則マス（王手中は除く。歩は二歩の可能性があるので除外）
///
/// 自分がそのマスで取り返しても消えない点は `check_prep::known_enemy_squares`
/// と違う（runtime 側の実装をそのまま単一の真実とする）。
pub fn backed_targets(log: &ObservationLog, move_number: u32, you_in_check: bool) -> [bool; 81] {
    let mut backed = [false; 81];
    for e in log.events() {
        match e {
            Observation::OpponentMoved { captured_my_piece_at: Some(sq), .. } => {
                if let Some(c) = parse_usi_square(sq) {
                    backed[sq_index(c)] = true;
                }
            }
            Observation::MyFoul { move_number: mn, usi } if *mn == move_number => {
                if you_in_check {
                    continue;
                }
                if let Some(ShogiMove::Drop { role, to }) = parse_usi(usi) {
                    if role != Role::Pawn {
                        backed[sq_index(to)] = true;
                    }
                }
            }
            _ => {}
        }
    }
    backed
}

/// 決定開始時に1回だけ作る需要帯マップ（その手番の全候補で共有する）
#[derive(Clone, Debug)]
pub struct BandMap {
    pub band: [Band; 81],
    /// `opp_king_candidates` の候補マス数（監査用）
    pub opp_king_candidates: u32,
    /// 候補が 1〜[`OPP_KING_GATE`] マスで opp_king 帯が発火したか
    pub opp_king_gated: bool,
    /// 優先順位適用後に opp_king 帯へ入ったマス数（帯サイズの監査用。
    /// 候補数に応じて帯が広がるので P0 の表に併記する — issue #40）
    pub opp_king_band_size: u32,
}

/// 需要帯マップを**着手前**の状態から作る。
///
/// - `pieces` / `log` / `move_number` / `you_in_check` は決定開始時の値
///   （`PlayerView` 相当。真実盤面には依存しない）
/// - 帯1〜3は粒子を使わず決定的。帯4（active_own_piece）もその帰結として決定的
pub fn band_map(
    pieces: &[VisiblePiece],
    color: Color,
    log: &ObservationLog,
    move_number: u32,
    you_in_check: bool,
) -> BandMap {
    let own_king = pieces
        .iter()
        .find(|p| p.role == Role::King)
        .and_then(|p| parse_usi_square(&p.square));

    let cands = crate::deduce::opp_king_candidates(color, log);
    let opp_king_gated = !cands.is_empty() && cands.len() <= OPP_KING_GATE;
    let backed = backed_targets(log, move_number, you_in_check);

    // 帯1〜3のマスク（優先順位を掛ける前）
    let mut own_king_mask = [false; 81];
    if let Some(k) = own_king {
        for i in 0..81 {
            if crate::check_prep::cheb(coord_of(i), k) <= 2 {
                own_king_mask[i] = true;
            }
        }
    }
    let mut opp_king_mask = [false; 81];
    if opp_king_gated {
        // 和集合＋チェビシェフ距離1以内（どの候補に真の玉が居ても意味のある利き）
        for &c in &cands {
            for df in -1..=1i8 {
                for dr in -1..=1i8 {
                    let s = Coord { file: c.file + df, rank: c.rank + dr };
                    if (1..=9).contains(&s.file) && (1..=9).contains(&s.rank) {
                        opp_king_mask[sq_index(s)] = true;
                    }
                }
            }
        }
    }

    // 帯4: 決定開始時の自駒のうち、利きが帯1〜3へ1つ以上届いている駒のマス。
    // 帯1〜3は上のマスクで確定しているので、この判定も決定的
    let mut active_mask = [false; 81];
    let in_demand =
        |i: usize| -> bool { own_king_mask[i] || opp_king_mask[i] || backed[i] };
    for p in pieces {
        let Some(sq) = parse_usi_square(&p.square) else { continue };
        if defend_targets(pieces, p, color).iter().any(|&s| in_demand(sq_index(s))) {
            active_mask[sq_index(sq)] = true;
        }
    }

    let mut band = [Band::Neutral; 81];
    let mut opp_king_band_size = 0u32;
    for i in 0..81 {
        band[i] = if own_king_mask[i] {
            Band::OwnKing
        } else if opp_king_mask[i] {
            opp_king_band_size += 1;
            Band::OppKing
        } else if backed[i] {
            Band::BackedTarget
        } else if active_mask[i] {
            Band::ActiveOwnPiece
        } else {
            Band::Neutral
        };
    }
    BandMap {
        band,
        opp_king_candidates: cands.len() as u32,
        opp_king_gated,
        opp_king_band_size,
    }
}

fn coord_of(i: usize) -> Coord {
    Coord { file: (i / 9) as i8 + 1, rank: (i % 9) as i8 + 1 }
}

/// 帯 × しきい値横断 × gain/loss の表。
/// `gain[band][t]` = そのマスの利き枚数が `<t+1 → ≥t+1` を横断した本数
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MarginalWorkBreakdown {
    pub gain: [[u32; N_TRANSITIONS]; N_BANDS],
    pub loss: [[u32; N_TRANSITIONS]; N_BANDS],
}

impl MarginalWorkBreakdown {
    pub fn add(&mut self, other: &MarginalWorkBreakdown) {
        for b in 0..N_BANDS {
            for t in 0..N_TRANSITIONS {
                self.gain[b][t] += other.gain[b][t];
                self.loss[b][t] += other.loss[b][t];
            }
        }
    }

    /// 需要帯（own_king / opp_king / backed_target / active_own_piece）への
    /// first + second の本数。link-only drop の分類はこれが 0 であること
    pub fn demand_first_second_gain(&self) -> u32 {
        (0..N_BANDS - 1).map(|b| self.gain[b][0] + self.gain[b][1]).sum()
    }

    pub fn total_gain(&self) -> u32 {
        self.gain.iter().flatten().sum()
    }

    pub fn total_loss(&self) -> u32 {
        self.loss.iter().flatten().sum()
    }

    /// neutral × redundant の gain（制圧済み低需要マスへの3枚目以上）
    pub fn neutral_redundant_gain(&self) -> u32 {
        self.gain[Band::Neutral as usize][2]
    }

    /// 正の利き増分に占める neutral × redundant の割合（増分ゼロなら None）。
    /// saturated promotion の頻度記述（70% 閾値）に使う
    pub fn neutral_redundant_frac(&self) -> Option<f64> {
        let total = self.total_gain();
        (total > 0).then(|| f64::from(self.neutral_redundant_gain()) / f64::from(total))
    }
}

/// 着手前後の利き枚数マップから、しきい値横断を帯別に数える。
///
/// `0→2` は first+second、`3→1` は redundant_loss+second_loss として数えられ
/// （多段変化の一意な加法分解）、`3→4` はどこにも数えられない（飽和）。
pub fn breakdown(
    bands: &BandMap,
    before: &[u8; 81],
    after: &[u8; 81],
) -> MarginalWorkBreakdown {
    let mut out = MarginalWorkBreakdown::default();
    for i in 0..81 {
        let (a, b) = (before[i], after[i]);
        if a == b {
            continue;
        }
        let band = bands.band[i] as usize;
        for t in 1..=N_TRANSITIONS as u8 {
            if a < t && b >= t {
                out.gain[band][usize::from(t) - 1] += 1;
            }
            if a >= t && b < t {
                out.loss[band][usize::from(t) - 1] += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shogi::Position;

    fn pc(sq: &str, role: Role) -> VisiblePiece {
        VisiblePiece { square: sq.into(), role }
    }

    fn sq(s: &str) -> Coord {
        parse_usi_square(s).unwrap()
    }

    fn neutral_bands() -> BandMap {
        BandMap {
            band: [Band::Neutral; 81],
            opp_king_candidates: 81,
            opp_king_gated: false,
            opp_king_band_size: 0,
        }
    }

    #[test]
    fn 多段変化の加法性_飛び利きの開閉() {
        // 5九の飛の上方向は 5五の金で遮られている。金が 4五 へ横に避けると
        // 5四〜5一 が開く（着手駒でなくブロッカーの移動で変わる盤全体の差分）
        let before = vec![pc("5i", Role::Rook), pc("5e", Role::Gold)];
        let a = attack_counts(&before, Color::Sente);
        assert_eq!(a[sq_index(sq("5c"))], 0, "金の後ろは飛から遮蔽");
        assert_eq!(a[sq_index(sq("5e"))], 1, "金のマスまでは利く（紐）");

        let mv = ShogiMove::Board { from: sq("5e"), to: sq("4e"), promote: false };
        let b = attack_counts_after(&before, &mv, Color::Sente);
        assert_eq!(b[sq_index(sq("5c"))], 1, "ブロッカーが動いて飛の利きが開く");

        let bd = breakdown(&neutral_bands(), &a, &b);
        // 5三〜5一 の 0→1 = first ×3 を含む
        assert!(bd.gain[Band::Neutral as usize][0] >= 3);
    }

    #[test]
    fn しきい値横断は一意に加法分解される() {
        // 人工の before/after で 0→2 と 3→1 を確認する
        let bands = neutral_bands();
        let mut before = [0u8; 81];
        let mut after = [0u8; 81];
        before[0] = 0;
        after[0] = 2; // first + second
        before[1] = 3;
        after[1] = 1; // redundant_loss + second_loss
        let bd = breakdown(&bands, &before, &after);
        let n = Band::Neutral as usize;
        assert_eq!(bd.gain[n][0], 1, "0→2 は first");
        assert_eq!(bd.gain[n][1], 1, "0→2 は second も");
        assert_eq!(bd.gain[n][2], 0);
        assert_eq!(bd.loss[n][2], 1, "3→1 は redundant_loss");
        assert_eq!(bd.loss[n][1], 1, "3→1 は second_loss も");
        assert_eq!(bd.loss[n][0], 0);
    }

    #[test]
    fn 利き枚数は3で飽和する() {
        let bands = neutral_bands();
        let mut before = [0u8; 81];
        let mut after = [0u8; 81];
        before[0] = 3;
        after[0] = 4; // 3→4 はどこにも数えない
        before[1] = 4;
        after[1] = 3; // 4→3 も
        let bd = breakdown(&bands, &before, &after);
        assert_eq!(bd, MarginalWorkBreakdown::default());
    }

    #[test]
    fn 帯は排他的で優先順位どおり() {
        // 自玉 5九。相手玉候補は初期 5一 から OpponentMoved 2回ぶん広がった箱
        // （＋距離1）= opp_king 帯。裏付け標的を own_king 圏内（5八）・
        // opp_king 圏内（5二）・どちらの圏外（1五）に置いて優先順位を見る
        let pieces = vec![pc("5i", Role::King), pc("9i", Role::Lance)];
        let mut log = ObservationLog::default();
        log.record(Observation::OpponentMoved {
            move_number: 2,
            captured_my_piece_at: Some("5h".into()),
        });
        log.record(Observation::OpponentMoved {
            move_number: 4,
            captured_my_piece_at: Some("5b".into()),
        });
        // 両玉圏外の裏付けは今手番の非歩打ち反則で作る（backed_targets のもう一方の由来）
        log.record(Observation::MyFoul { move_number: 5, usi: "G*1e".into() });
        let bm = band_map(&pieces, Color::Sente, &log, 5, false);
        assert!(bm.opp_king_gated, "候補は 5一 の周囲に集中していてゲートを通る");
        assert_eq!(bm.band[sq_index(sq("5h"))], Band::OwnKing, "own_king が backed に勝つ");
        assert_eq!(bm.band[sq_index(sq("5b"))], Band::OppKing, "opp_king が backed に勝つ");
        assert_eq!(bm.band[sq_index(sq("1e"))], Band::BackedTarget, "両玉圏外の裏付け標的");
        assert_eq!(bm.band[sq_index(sq("4h"))], Band::OwnKing);
        // 9九の香は帯1〜3へ利いていない（9八へ利くだけ）ので中立
        assert_eq!(bm.band[sq_index(sq("9i"))], Band::Neutral, "働きのない駒のマスは中立");
    }

    #[test]
    fn 帯4は帯123へ利く駒のマスだけ() {
        // 自玉 5九、5二 に裏付け標的。5五の飛は5二へ（5四・5三 が空きなら）利く
        let pieces = vec![pc("5i", Role::King), pc("5e", Role::Rook), pc("1a", Role::Lance)];
        let mut log = ObservationLog::default();
        log.record(Observation::OpponentMoved {
            move_number: 2,
            captured_my_piece_at: Some("5b".into()),
        });
        let bm = band_map(&pieces, Color::Sente, &log, 3, false);
        assert_eq!(bm.band[sq_index(sq("5e"))], Band::ActiveOwnPiece, "裏付け標的へ利く飛");
        // 1一の香（先手なので上方向へは盤外）はどの帯へも利かない
        assert_eq!(bm.band[sq_index(sq("1a"))], Band::Neutral);
    }

    #[test]
    fn 帯マップは着手前の状態だけから決まる() {
        // band_map は pieces（着手前）しか受けない構造だが、着手後の駒が
        // 帯4を作らないことを明示的に確認する: 打つ前は neutral のマスに
        // 打っても、そのマスの帯は neutral のまま
        let pieces = vec![pc("5i", Role::King)];
        let log = ObservationLog::default();
        let bm = band_map(&pieces, Color::Sente, &log, 1, false);
        assert_eq!(bm.band[sq_index(sq("1c"))], Band::Neutral);
        let before = attack_counts(&pieces, Color::Sente);
        let mv = ShogiMove::Drop { role: Role::Gold, to: sq("1c") };
        let after = attack_counts_after(&pieces, &mv, Color::Sente);
        let bd = breakdown(&bm, &before, &after);
        // 金の利き5マス（盤内ぶん）は全部 neutral の first に入る
        assert_eq!(bd.demand_first_second_gain(), 0);
        assert!(bd.gain[Band::Neutral as usize][0] > 0);
    }

    #[test]
    fn opp_king帯は候補20以下のときだけ発火する() {
        // 観測ゼロなら相手玉は初期位置に確定（候補1）なので発火する
        let pieces = vec![pc("5i", Role::King)];
        let log = ObservationLog::default();
        let bm = band_map(&pieces, Color::Sente, &log, 1, false);
        assert!(bm.opp_king_gated);
        assert_eq!(bm.opp_king_candidates, 1);
        assert!(bm.opp_king_band_size >= 1);

        // 相手が4手動くと候補は 5一 の周囲へ radius 4 まで散る（>20）ので発火しない
        let mut log = ObservationLog::default();
        for i in 0..4u32 {
            log.record(Observation::OpponentMoved {
                move_number: 2 * i + 2,
                captured_my_piece_at: None,
            });
        }
        let bm = band_map(&pieces, Color::Sente, &log, 9, false);
        assert!(!bm.opp_king_gated);
        assert!(bm.opp_king_candidates > OPP_KING_GATE as u32);
        assert_eq!(bm.opp_king_band_size, 0);
    }

    #[test]
    fn 特徴量は相手の駒配置に依存しない() {
        // 真実盤面から相手の駒を全部消しても、自駒 view から作る表は不変
        let truth = Position::initial();
        let me = Color::Sente;
        let mine = truth.pieces_of(me);
        let mut cleared = truth.clone();
        let opp_squares: Vec<Coord> =
            truth.pieces().filter(|(_, p)| p.color != me).map(|(c, _)| c).collect();
        for c in opp_squares {
            cleared.set(c, None);
        }
        let key = |v: &[VisiblePiece]| -> Vec<(String, Role)> {
            let mut k: Vec<_> = v.iter().map(|p| (p.square.clone(), p.role)).collect();
            k.sort();
            k
        };
        assert_eq!(key(&mine), key(&cleared.pieces_of(me)));
        let log = ObservationLog::default();
        let a1 = attack_counts(&mine, me);
        let a2 = attack_counts(&cleared.pieces_of(me), me);
        assert_eq!(a1, a2);
        let b1 = band_map(&mine, me, &log, 1, false);
        let b2 = band_map(&cleared.pieces_of(me), me, &log, 1, false);
        assert_eq!(b1.band, b2.band);
    }

    #[test]
    fn 利き枚数マップは玉も数える() {
        // 事前登録の「自駒利き枚数」どおり、玉の利きも1枚として数える
        // （runtime の own_attack_counts と同じ側）
        let pieces = vec![pc("5i", Role::King), pc("5g", Role::Pawn)];
        let a = attack_counts(&pieces, Color::Sente);
        assert_eq!(a[sq_index(sq("5h"))], 1, "玉の利き");
        assert_eq!(a[sq_index(sq("5f"))], 1, "歩の利き");
        // 玉が既に利くマスへの利き足しは first でなく second になる
        let mv = ShogiMove::Drop { role: Role::Gold, to: sq("5g") };
        let _ = mv; // 5g は歩がいるので打てないが、しきい値の意味の確認は breakdown 側
        let mut before = [0u8; 81];
        before[sq_index(sq("5h"))] = 1;
        let mut after = before;
        after[sq_index(sq("5h"))] = 2;
        let bands = neutral_bands();
        let bd = breakdown(&bands, &before, &after);
        assert_eq!(bd.gain[Band::Neutral as usize][1], 1, "1→2 は second");
        assert_eq!(bd.gain[Band::Neutral as usize][0], 0);
    }
}
