//! **序盤の強制**（学習の自己対局で序盤を散らす。docs/rl-opening-diversity.md）。
//!
//! 方策ネットの序盤は約40,000更新から飛車先の歩の速攻1本に固まり、ほかの手の確率は 1e-18
//! まで落ちている（指されないので学ぶ機会もない）。そこで局ごと・**先後それぞれ独立に**確率 p で、
//! その側の最初の数手を定跡のラインから指させる。強制した手は env の中で指す（`EnvGame::settle`）
//! ので学習側の行には現れず、学習器から見ると「環境の手」（チャンス手）になる。
//!
//! - **独立に引く**のが要点: 強制されなかった側は、相手が強制されたかどうかを自分の観測から
//!   推せない（駒がぶつかるまで相手の手は観測に入らない）。両者をまとめて強制すると、
//!   「自分の序盤が定跡なら相手も定跡」という相関が漏れる
//! - **抜ける条件は `crate::opening::OpeningBook::next` と同じ**: どちらかの駒取り・王手
//!   （= 両者に見える接触）、自分の反則、ラインを指し終えた。ただし定跡の最後の**駒取りの手は
//!   強制しない**（初期配置の相手の駒に触れる手の手前で切る）。ぶつかった後の判断は方策が指す
//! - 定跡の手が実際には反則（見えない相手の駒に遮られた等）なら、**そのまま反則として指して**
//!   から抜ける（`opening.rs` と同じ）。真実の盤面で合法かを先に調べて黙って抜けると、
//!   方策が指し始める局面が見えない情報と相関する
//! - `opening.rs` と `joseki.json` は凍結版の共有 pin（`frozen::SHARED_MODEL_PINS`）なので
//!   触らない。ラインは呼び出し側（Python の学習）が読んで渡す

use std::sync::Arc;

use crate::board::{Coord, Promotion, make_usi_square, move_targets, parse_usi_square, promotion_choice};
use crate::protocol::Color;
use crate::rl::action::{decode_usi, encode_usi};
use crate::selfplay::mix;
use crate::shogi::{Position, ShogiMove, parse_usi, promote_role};

/// 序盤の乱数列の塩（相手戦略の seed `mix(seed ^ mix(game_no))` と混ざらないように）
const OPENING_SALT: u64 = 0x6f70_656e_696e_6721;

/// どの側を強制の対象にするか
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForcedSides {
    /// 先後どちらも（それぞれ独立に確率 p）
    Both,
    Sente,
    Gote,
    /// 評価モードの学習側だけ
    Learner,
    /// 評価モードの相手側だけ
    Opponent,
}

impl ForcedSides {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "both" => Some(Self::Both),
            "sente" => Some(Self::Sente),
            "gote" => Some(Self::Gote),
            "learner" => Some(Self::Learner),
            "opponent" => Some(Self::Opponent),
            _ => None,
        }
    }

    /// 学習側と相手側を区別する指定か（自己対局の env では使えない）
    pub fn needs_learner(self) -> bool {
        matches!(self, Self::Learner | Self::Opponent)
    }

    fn eligible(self, color: Color, learner: Option<Color>) -> bool {
        match self {
            Self::Both => true,
            Self::Sente => color == Color::Sente,
            Self::Gote => color == Color::Gote,
            Self::Learner => learner == Some(color),
            Self::Opponent => learner.is_some_and(|l| l != color),
        }
    }
}

/// 1局の1つの側に割り当てた強制の予定
#[derive(Debug, Clone)]
pub struct ForcedPlan {
    /// 先手視点で正規化した行動番号の並び（`action::encode_usi(_, Sente)`。後手は同じ番号を
    /// `decode_usi(_, Gote)` で読むと点対称にミラーした手になる）
    pub line: Arc<[usize]>,
    /// ブックの中のラインの添字（ラインの名前は一意でないので添字で数える）
    pub line_id: u32,
    /// 強制する手数の上限（≤ ラインの長さ）
    pub limit: u32,
}

/// 強制の材料（ラインの集まりと引き方）
#[derive(Debug, Clone)]
pub struct ForcedBook {
    lines: Vec<Arc<[usize]>>,
    /// 重みの累積（最後が 1）
    cdf: Vec<f64>,
    prob: f64,
    len: Option<(u32, u32)>,
    sides: ForcedSides,
}

impl ForcedBook {
    /// `lines` は先手視点の USI の並び（`joseki.json` の `moves`）。各ラインは自駒だけの盤で
    /// 指し切れるかを検査し、**初期配置の相手の駒に触れる手の手前で切る**（`quiet_prefix`）。
    /// `weights` は None なら一様。`len` は None ならライン（切った後）を最後まで、
    /// `Some((lo, hi))` なら手数を {lo..hi} から一様に引いてラインの長さで頭打ちにする
    pub fn new(
        lines: &[Vec<String>],
        weights: Option<&[f64]>,
        prob: f64,
        len: Option<(u32, u32)>,
        sides: ForcedSides,
    ) -> Result<Self, String> {
        if !(prob.is_finite() && (0.0..=1.0).contains(&prob)) {
            return Err(format!("opening_prob は 0〜1（{prob}）"));
        }
        if lines.is_empty() {
            return Err("openings が空".into());
        }
        if let Some((lo, hi)) = len {
            if lo == 0 || lo > hi {
                return Err(format!("opening_len は 1 ≤ lo ≤ hi（{lo}, {hi}）"));
            }
        }
        let mut encoded = Vec::with_capacity(lines.len());
        for (i, line) in lines.iter().enumerate() {
            let quiet = quiet_prefix(line).map_err(|e| format!("ライン {i}: {e}"))?;
            let actions: Vec<usize> = quiet
                .iter()
                .map(|u| {
                    let a = encode_usi(u, Color::Sente).ok_or_else(|| format!("ライン {i}: {u} を符号化できない"))?;
                    // 正規の表記か（成りの指定など）。記録・反則済みの手の照合は文字列で行うので、
                    // 方策が同じ手を選んだときと同じ文字列になることを保証する
                    if decode_usi(a, Color::Sente).as_deref() != Some(u.as_str()) {
                        return Err(format!("ライン {i}: {u} は正規の USI ではない"));
                    }
                    Ok(a)
                })
                .collect::<Result<_, String>>()?;
            encoded.push(Arc::from(actions));
        }
        let weights: Vec<f64> = match weights {
            None => vec![1.0; lines.len()],
            Some(w) if w.len() != lines.len() => {
                return Err(format!("opening_weights の長さ {} がラインの数 {} と違う", w.len(), lines.len()));
            }
            Some(w) => w.to_vec(),
        };
        if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err("opening_weights は有限で 0 以上".into());
        }
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Err("opening_weights の合計が 0".into());
        }
        let mut acc = 0.0;
        let mut cdf: Vec<f64> = weights
            .iter()
            .map(|w| {
                acc += w / total;
                acc
            })
            .collect();
        *cdf.last_mut().expect("ラインは1本以上") = 1.0;
        Ok(Self { lines: encoded, cdf, prob, len, sides })
    }

    pub fn sides(&self) -> ForcedSides {
        self.sides
    }

    /// 切った後の各ラインの手数（検査・表示用）
    pub fn line_lengths(&self) -> Vec<usize> {
        self.lines.iter().map(|l| l.len()).collect()
    }

    /// 局 `k`（`seed` の乱数列の中の通し番号）の [先手, 後手] の予定。`learner` は評価モードの
    /// 学習側の色（自己対局は None）。
    ///
    /// 乱数は側ごとに3つ（強制するか・ライン・手数）を**常に**引く: p を変えても、両方の p で
    /// 強制される側のラインと手数は同じになる（p の比較が対になる）
    pub fn plans(&self, seed: u64, k: u64, learner: Option<Color>) -> [Option<ForcedPlan>; 2] {
        let base = mix(mix(seed ^ OPENING_SALT) ^ mix(k));
        let unit = |j: u64| (mix(base.wrapping_add(j)) >> 11) as f64 / (1u64 << 53) as f64;
        let mut out = [None, None];
        for (ci, color) in [Color::Sente, Color::Gote].into_iter().enumerate() {
            let j = 3 * ci as u64;
            let (u_force, u_line, u_len) = (unit(j), unit(j + 1), unit(j + 2));
            if !self.sides.eligible(color, learner) || u_force >= self.prob {
                continue;
            }
            let line_id = self.cdf.iter().position(|&c| u_line < c).unwrap_or(self.cdf.len() - 1);
            let line = &self.lines[line_id];
            let n = line.len() as u32;
            let limit = match self.len {
                None => n,
                Some((lo, hi)) => (lo + ((u_len * f64::from(hi - lo + 1)) as u32).min(hi - lo)).min(n),
            };
            out[ci] = Some(ForcedPlan { line: Arc::clone(line), line_id: line_id as u32, limit });
        }
        out
    }
}

/// 先手視点のラインを自駒だけの盤で指し進めて検査し、**初期配置の後手の駒に触れる手**
/// （行き先か、飛び駒の通り道に後手の初期配置の駒がある手 = 相手が動いていなければ駒取りか
/// 反則になる手）の手前で切る。
///
/// 検査は `opening.rs` の `loaded_lines_replay_on_own_view` と同じ（移動元に自駒がある・
/// その駒がそこへ動ける・成りの指定が正しい）。打ちは扱わない。切った結果が空ならエラー
pub fn quiet_prefix(line: &[String]) -> Result<Vec<String>, String> {
    let initial = Position::initial();
    let mut pieces = initial.pieces_of(Color::Sente);
    let mut out = vec![];
    for usi in line {
        let Some(ShogiMove::Board { from, to, promote }) = parse_usi(usi) else {
            return Err(format!("{usi} は盤上の手ではない（打ちは使えない）"));
        };
        let idx = pieces
            .iter()
            .position(|p| parse_usi_square(&p.square) == Some(from))
            .ok_or_else(|| format!("{usi}: 移動元に自駒がない"))?;
        if !move_targets(&pieces, &pieces[idx], Color::Sente).contains(&to) {
            return Err(format!("{usi}: その駒はそこへ動けない"));
        }
        let choice = promotion_choice(pieces[idx].role, from, to, Color::Sente);
        match (promote, choice) {
            (true, Promotion::None) => return Err(format!("{usi}: 成れない手に + が付いている")),
            (false, Promotion::Forced) => return Err(format!("{usi}: 成りが強制の手なのに + がない")),
            _ => {}
        }
        if touches_initial_opponent(&initial, from, to) {
            break;
        }
        if promote {
            pieces[idx].role = promote_role(pieces[idx].role).ok_or_else(|| format!("{usi}: 成れない駒種"))?;
        }
        pieces[idx].square = make_usi_square(to);
        out.push(usi.clone());
    }
    if out.is_empty() {
        return Err("初手から相手の初期配置の駒に触れる（強制できる手が無い）".into());
    }
    Ok(out)
}

/// 行き先か、飛び駒の通り道（両端を除く）に後手の初期配置の駒があるか
fn touches_initial_opponent(initial: &Position, from: Coord, to: Coord) -> bool {
    let gote_at = |c: Coord| initial.piece_at(c).is_some_and(|p| p.color == Color::Gote);
    if gote_at(to) {
        return true;
    }
    let (df, dr) = (to.file - from.file, to.rank - from.rank);
    // 桂の跳びは通り道を持たない（直線・斜めの動きだけ途中のマスを見る）
    if df != 0 && dr != 0 && df.abs() != dr.abs() {
        return false;
    }
    let (sf, sr) = (df.signum(), dr.signum());
    let mut c = Coord { file: from.file + sf, rank: from.rank + sr };
    while c != to {
        if gote_at(c) {
            return true;
        }
        c = Coord { file: c.file + sf, rank: c.rank + sr };
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(moves: &[&str]) -> Vec<String> {
        moves.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn 初期配置の相手の駒に触れる手の手前で切る() {
        // 居飛車速攻: 2三歩成（2c の後手の歩を取る）の手前まで
        assert_eq!(quiet_prefix(&line(&["2g2f", "2f2e", "2e2d", "2d2c+"])).unwrap(), line(&["2g2f", "2f2e", "2e2d"]));
        // 中飛車の桂の跳び（行き先 5c に後手の歩）
        let nakabisha = line(&["7g7f", "2h5h", "5g5f", "8h6f", "8i7g", "6f7e", "5f5e", "7g6e", "5e5d", "6e5c"]);
        assert_eq!(quiet_prefix(&nakabisha).unwrap().len(), 9);
        // 飛び駒の通り道に後手の初期配置の駒（角道 8h→2b は 3c の歩に遮られる）
        assert_eq!(quiet_prefix(&line(&["7g7f", "8h2b+"])).unwrap(), line(&["7g7f"]));
        // 空いたマスへの角の移動は切らない
        assert_eq!(quiet_prefix(&line(&["7g7f", "8h6f"])).unwrap().len(), 2);
    }

    #[test]
    fn 不正なラインは理由つきで拒む() {
        assert!(quiet_prefix(&line(&["P*5e"])).unwrap_err().contains("打ち"));
        assert!(quiet_prefix(&line(&["5e5d"])).unwrap_err().contains("移動元"));
        assert!(quiet_prefix(&line(&["2h2c"])).unwrap_err().contains("動けない")); // 自分の歩に遮られる
        assert!(quiet_prefix(&line(&["7g7f+"])).unwrap_err().contains("成れない"));
        assert!(quiet_prefix(&line(&["2h2b+"])).unwrap_err().contains("動けない"));
        // 切る位置より後ろは読まない（解釈できない手があっても通る）
        assert_eq!(quiet_prefix(&line(&["2g2f", "2f2e", "2e2d", "2d2c+", "x"])).unwrap().len(), 3);
        assert!(ForcedBook::new(&[line(&["7g7f", "2d2c+"])], None, 0.5, None, ForcedSides::Both)
            .unwrap_err()
            .contains("ライン 0"));
        assert!(ForcedBook::new(&[line(&["2g2f"])], Some(&[1.0, 2.0]), 0.5, None, ForcedSides::Both).is_err());
        assert!(ForcedBook::new(&[line(&["2g2f"])], Some(&[0.0]), 0.5, None, ForcedSides::Both).is_err());
        assert!(ForcedBook::new(&[line(&["2g2f"])], None, 1.5, None, ForcedSides::Both).is_err());
        assert!(ForcedBook::new(&[line(&["2g2f"])], None, 0.5, Some((0, 3)), ForcedSides::Both).is_err());
        assert!(ForcedBook::new(&[line(&["2g2f"])], None, 0.5, Some((3, 2)), ForcedSides::Both).is_err());
        assert!(ForcedBook::new(&[], None, 0.5, None, ForcedSides::Both).is_err());
    }

    #[test]
    fn 後手は同じ行動番号を点対称にミラーした手として読む() {
        let book = ForcedBook::new(&[line(&["7g7f", "2h6h", "5i4h"])], None, 1.0, None, ForcedSides::Both).unwrap();
        let [Some(s), Some(g)] = book.plans(1, 0, None) else { panic!("p=1 なら両者とも強制") };
        let read = |p: &ForcedPlan, c| p.line.iter().map(|&a| decode_usi(a, c).unwrap()).collect::<Vec<_>>();
        assert_eq!(read(&s, Color::Sente), line(&["7g7f", "2h6h", "5i4h"]));
        assert_eq!(read(&g, Color::Gote), line(&["3c3d", "8b4b", "5a6b"]));
    }

    #[test]
    fn 予定は乱数列と通し番号で決まり_pを変えても強制される側の中身は同じ() {
        let lines: Vec<Vec<String>> = vec![
            line(&["7g7f", "6g6f", "2h6h", "5i4h"]),
            line(&["5g5f", "2h5h", "5f5e", "5i4h"]),
            line(&["2g2f", "2f2e", "2e2d"]),
        ];
        let lo = ForcedBook::new(&lines, None, 0.3, Some((1, 4)), ForcedSides::Both).unwrap();
        let hi = ForcedBook::new(&lines, None, 0.8, Some((1, 4)), ForcedSides::Both).unwrap();
        let key = |p: &Option<ForcedPlan>| p.as_ref().map(|p| (p.line_id, p.limit));
        let (mut n_lo, mut n_hi, mut lines_seen, mut limits_seen) = (0, 0, [0; 3], [0; 5]);
        for k in 0..2000u64 {
            let a = lo.plans(7, k, None);
            let b = hi.plans(7, k, None);
            assert_eq!(a.clone().map(|p| key(&p)), lo.plans(7, k, None).map(|p| key(&p)), "同じ番号なら同じ予定");
            for c in 0..2 {
                if let Some(p) = &a[c] {
                    n_lo += 1;
                    assert_eq!(key(&b[c]), Some((p.line_id, p.limit)), "p を上げても強制される側は同じ");
                }
                if let Some(p) = &b[c] {
                    n_hi += 1;
                    lines_seen[p.line_id as usize] += 1;
                    limits_seen[p.limit as usize] += 1;
                    assert!(p.limit >= 1 && p.limit as usize <= p.line.len());
                }
            }
        }
        // 側ごとに独立に p（4000 側）。二項分布の ±5σ に入る
        assert!((1000..1400).contains(&n_lo), "{n_lo}");
        assert!((3000..3400).contains(&n_hi), "{n_hi}");
        assert!(lines_seen.iter().all(|&n| n > 900), "{lines_seen:?}");
        // 3手のラインは 4 で頭打ちになるので limit 4 は残りの2本のぶんだけ
        assert!(limits_seen[1] > 700 && limits_seen[3] > 700 && limits_seen[4] > 300, "{limits_seen:?}");
        assert_ne!(
            lo.plans(7, 0, None).map(|p| key(&p)),
            lo.plans(8, 0, None).map(|p| key(&p)),
            "乱数列を変えると別の予定（この組はたまたま同じにならない）"
        );
    }

    #[test]
    fn 側の指定() {
        let lines = vec![line(&["7g7f"])];
        let only = |sides, learner| {
            let book = ForcedBook::new(&lines, None, 1.0, None, sides).unwrap();
            book.plans(3, 0, learner).map(|p| p.is_some())
        };
        assert_eq!(only(ForcedSides::Both, None), [true, true]);
        assert_eq!(only(ForcedSides::Sente, None), [true, false]);
        assert_eq!(only(ForcedSides::Gote, None), [false, true]);
        assert_eq!(only(ForcedSides::Learner, Some(Color::Gote)), [false, true]);
        assert_eq!(only(ForcedSides::Opponent, Some(Color::Gote)), [true, false]);
        assert_eq!(only(ForcedSides::Learner, None), [false, false]);
        let none = ForcedBook::new(&lines, None, 0.0, None, ForcedSides::Both).unwrap();
        assert!(none.plans(3, 0, None).iter().all(Option::is_none));
    }

    fn book_lines(name: &str, json: &str) -> Vec<Vec<String>> {
        let v: serde_json::Value = serde_json::from_str(json).unwrap_or_else(|e| panic!("{name} は JSON: {e}"));
        v["lines"]
            .as_array()
            .unwrap_or_else(|| panic!("{name} に lines が無い"))
            .iter()
            .map(|l| l["moves"].as_array().unwrap().iter().map(|m| m.as_str().unwrap().to_string()).collect())
            .collect()
    }

    #[test]
    fn 定跡ファイルの全ラインが通る() {
        let lines = book_lines("joseki.json", include_str!("../../joseki.json"));
        assert_eq!(lines.len(), 34);
        let book = ForcedBook::new(&lines, None, 1.0, None, ForcedSides::Both).unwrap();
        let lens = book.line_lengths();
        // 居飛車速攻（ライン 1）は 2三歩成の手前の3手
        assert_eq!(lens[1], 3);
        assert!(lens.iter().all(|&n| (1..=16).contains(&n)), "{lens:?}");
    }

    /// 学習用のブック（`rl_openings.json`。`train_rnad.py --open-book` に渡す）。joseki.json と違って
    /// 凍結版の pin ではないので自由に足してよいが、足したラインが env で拒まれないことはここで守る
    #[test]
    fn 学習用の定跡ファイルの全ラインが通る() {
        let json = include_str!("../../rl_openings.json");
        let lines = book_lines("rl_openings.json", json);
        assert!(!lines.is_empty());
        let book = ForcedBook::new(&lines, None, 1.0, None, ForcedSides::Both).unwrap();
        let lens = book.line_lengths();
        assert!(lens.iter().all(|&n| n >= 1), "接触の手前で 0 手になるライン: {lens:?}");
        // 系統ごとの重み（train_rnad.py の既定 --open-weight category）に使うので、系統と名前は必ず書く
        let v: serde_json::Value = serde_json::from_str(json).unwrap();
        for (i, l) in v["lines"].as_array().unwrap().iter().enumerate() {
            for key in ["name", "category"] {
                assert!(l[key].as_str().is_some_and(|s| !s.is_empty()), "ライン {i} に {key} が無い");
            }
            if let Some(w) = l.get("weight") {
                assert!(w.as_f64().is_some_and(|w| w.is_finite() && w >= 0.0), "ライン {i} の weight: {w}");
            }
        }
    }
}
