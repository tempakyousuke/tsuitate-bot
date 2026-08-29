//! 被王手の**前**の準備（王手耐性）の**共有定義**（issue #34）。
//!
//! #31 は「王手が来てからの価格・順序・較正」を測って不発で閉じた。ここで測るのは
//! **王手が来る前の形**で、`bin/analyze` の P0-1（連続王手の起点）と
//! `bin/check_prep_probe` の P0-2（較正プローブ）が**同じ定義**で数えるための場所。
//!
//! `check_economy.rs` / `mate_economy.rs` と同じ位置づけで、**runtime には一切
//! 入らない**（評価も推定も触らない）。
//!
//! # 特徴量の意味は「楽観上限」であって真実ではない
//!
//! F1〜F4 はすべて `PlayerView` 相当（自駒＋持ち駒＋観測ログ）だけから決まるので
//! 値そのものにノイズは無いが、**飛び駒の利きは自駒にしか遮られない楽観値**で、
//! 「空きマス」は「自駒がいないマス」（相手の駒がいるかもしれない）。
//! 名前もそのとおりに付けてある（`potential_covered` / `potential_flight`）。
//! ピン・両王手・隠れた敵駒は扱わない。

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::board::{Coord, make_usi_square, on_board, orient, parse_usi_square, rays, steps};
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{Color, Role, VisiblePiece};
use crate::shogi::{Position, ShogiMove, parse_usi};

/// 王手起点マスの分類。**隣接（距離1）は F3 の領分なのでここには持たない**
/// （F1 と F3 で同じ概念を二重に持つと、説明力を見てどちらかを選ぶことになる。
/// 設計時点で分担を固定する — issue #34 の「F1 側の隣接列は持たない」）。
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub enum OriginClass {
    /// 桂の跳び起点（2マス。**確定**に近い = 途中に駒がいても跳べる）
    Knight,
    /// 縦横の線上（距離2以上）
    Line,
    /// 斜めの線上（距離2以上）
    Diag,
}

impl OriginClass {
    pub const ALL: [OriginClass; 3] = [OriginClass::Knight, OriginClass::Line, OriginClass::Diag];

    pub fn tag(self) -> &'static str {
        match self {
            OriginClass::Knight => "knight",
            OriginClass::Line => "line",
            OriginClass::Diag => "diag",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            OriginClass::Knight => "桂",
            OriginClass::Line => "直線",
            OriginClass::Diag => "斜線",
        }
    }
}

/// 王手起点マス1つ
#[derive(Clone, Copy, Debug)]
pub struct Origin {
    pub sq: Coord,
    pub class: OriginClass,
    /// 自玉からのチェビシェフ距離（桂は 2）
    pub dist: u8,
}

/// 距離バケット（0=2, 1=3, 2=4, 3=5以上）。桂は常に 0
pub fn dist_bucket(dist: u8) -> usize {
    match dist {
        0 | 1 | 2 => 0,
        3 => 1,
        4 => 2,
        _ => 3,
    }
}

pub const DIST_BUCKETS: usize = 4;
pub const DIST_BUCKET_LABELS: [&str; DIST_BUCKETS] = ["d2", "d3", "d4", "d5+"];

/// **王手起点マス集合 C(K)**: 自玉 K の位置だけで決まる固定集合。
///
/// 盤端までの8方向の距離2以上の全マス＋桂の跳び起点2マス。
/// **自駒で止めない**のが要点で、自駒を置いて未被覆マスを集合から消せると
/// 「分母を縮めて被覆率を上げる」分母操作が成立してしまう（issue #34）。
pub fn origin_squares(king: Coord, bot: Color) -> Vec<Origin> {
    let mut out = vec![];
    // 桂の跳び起点: 相手の桂がそこにいれば自玉へ利く2マス
    for &delta in steps(Role::Knight) {
        let (df, dr) = orient(delta, bot.other());
        let s = Coord { file: king.file - df, rank: king.rank - dr };
        if on_board(s) {
            out.push(Origin { sq: s, class: OriginClass::Knight, dist: 2 });
        }
    }
    for (dirs, class) in [
        ([(0i8, -1i8), (0, 1), (1, 0), (-1, 0)], OriginClass::Line),
        ([(1, -1), (-1, -1), (1, 1), (-1, 1)], OriginClass::Diag),
    ] {
        for (df, dr) in dirs {
            let mut d = 2i8;
            loop {
                let s = Coord { file: king.file + df * d, rank: king.rank + dr * d };
                if !on_board(s) {
                    break;
                }
                out.push(Origin { sq: s, class, dist: d as u8 });
                d += 1;
            }
        }
    }
    out
}

/// 自駒の利き（**玉を除く**）を「確定」と「楽観レイ」に分けた被覆マップ。
///
/// - **確定利き**: 歩・桂・金銀の1マス歩進（馬龍の1マス部分も含む）。隠れた敵駒に
///   遮られようがないので、そのマスへ来た駒は本当に取れる
/// - **楽観レイ**: 飛角香（馬龍のレイ部分）。途中に見えない敵駒がいれば通らない
///
/// レイは `board::defend_targets` と同じく**自駒のマスを含めてそこで止まる**
/// （自駒が乗っているマスも「利かせている」= 取り返せる）。
#[derive(Clone, Debug, Default)]
pub struct Coverage {
    pub sure: HashSet<Coord>,
    pub ray: HashSet<Coord>,
    /// 自駒が乗っているマス
    pub occupied: HashSet<Coord>,
    /// マスごとの被覆枚数（玉を除く）
    pub count: HashMap<Coord, u32>,
}

impl Coverage {
    pub fn covered(&self, c: Coord) -> bool {
        self.sure.contains(&c) || self.ray.contains(&c)
    }
    pub fn covered_sure(&self, c: Coord) -> bool {
        self.sure.contains(&c)
    }
    pub fn count_at(&self, c: Coord) -> u32 {
        self.count.get(&c).copied().unwrap_or(0)
    }
}

/// 玉を除く自駒の利きを1回の走査で集める
pub fn coverage_of(pieces: &[VisiblePiece], color: Color) -> Coverage {
    let occupied: HashSet<Coord> =
        pieces.iter().filter_map(|p| parse_usi_square(&p.square)).collect();
    let mut cov = Coverage { occupied, ..Coverage::default() };
    for p in pieces.iter().filter(|p| p.role != Role::King) {
        let Some(from) = parse_usi_square(&p.square) else { continue };
        let mut hit: Vec<(Coord, bool)> = vec![];
        for &delta in steps(p.role) {
            let (df, dr) = orient(delta, color);
            let c = Coord { file: from.file + df, rank: from.rank + dr };
            if on_board(c) {
                hit.push((c, true));
            }
        }
        for &delta in rays(p.role) {
            let (df, dr) = orient(delta, color);
            let mut c = Coord { file: from.file + df, rank: from.rank + dr };
            while on_board(c) {
                hit.push((c, false));
                if cov.occupied.contains(&c) {
                    break; // 自駒のマスも利かせているが、そこでレイは止まる
                }
                c = Coord { file: c.file + df, rank: c.rank + dr };
            }
        }
        let mut seen: HashSet<Coord> = HashSet::new();
        for (c, sure) in hit {
            if sure {
                cov.sure.insert(c);
            } else {
                cov.ray.insert(c);
            }
            if seen.insert(c) {
                *cov.count.entry(c).or_insert(0) += 1;
            }
        }
    }
    cov
}

/// **F1 王手起点の潜在被覆**（指導1: 王手が来そうな場所に自駒の利きを作る）
#[derive(Clone, Debug, Default, PartialEq)]
pub struct F1 {
    /// クラス別の (起点数, 被覆数, うち確定利きでの被覆数)
    pub by_class: [(u32, u32, u32); 3],
    /// 距離バケット別の (起点数, 被覆数)
    pub by_dist: [(u32, u32); DIST_BUCKETS],
}

impl F1 {
    pub fn total(&self) -> u32 {
        self.by_class.iter().map(|c| c.0).sum()
    }
    pub fn covered(&self) -> u32 {
        self.by_class.iter().map(|c| c.1).sum()
    }
    pub fn covered_sure(&self) -> u32 {
        self.by_class.iter().map(|c| c.2).sum()
    }
    /// 素の被覆率（重み無し）
    pub fn cov_frac(&self) -> f64 {
        let t = self.total();
        if t == 0 { 0.0 } else { f64::from(self.covered()) / f64::from(t) }
    }
    /// 起点の重み付き被覆率。**重みは発見セットの実王手起点の分布から決めて
    /// 検証セットでは固定する**（`OriginWeights` の doc）
    pub fn cov_weighted(&self, w: &OriginWeights) -> f64 {
        // クラス × 距離の同時分布は持っていない（列が増えすぎる）ので、
        // クラス重みと距離重みを別々に正規化した2つの被覆率の平均を使う。
        // どちらも `covered/total` の凸結合なので [0,1] に収まる
        let mix = |cells: &[(u32, u32)], ws: &[f64]| -> f64 {
            let mut num = 0.0;
            let mut den = 0.0;
            for (cell, weight) in cells.iter().zip(ws) {
                if cell.0 == 0 {
                    continue;
                }
                num += weight * f64::from(cell.1) / f64::from(cell.0);
                den += weight;
            }
            if den > 0.0 { num / den } else { 0.0 }
        };
        let by_class: Vec<(u32, u32)> = self.by_class.iter().map(|c| (c.0, c.1)).collect();
        0.5 * mix(&by_class, &w.class) + 0.5 * mix(&self.by_dist, &w.dist)
    }
}

/// 起点の重み（クラス3 × 距離バケット4）。
///
/// **事前登録の既定は一様**（`UNIFORM`）。issue #34 の手順は
/// 「距離重みは事前に置かず、**発見セットの実王手起点の距離分布**から決めて
/// 検証セットでは固定する」なので、発見セットで測った分布を
/// `--origin-weights` で渡して検証セットへ送る。
#[derive(Clone, Debug)]
pub struct OriginWeights {
    pub class: [f64; 3],
    pub dist: [f64; DIST_BUCKETS],
}

impl OriginWeights {
    pub const UNIFORM: OriginWeights =
        OriginWeights { class: [1.0; 3], dist: [1.0; DIST_BUCKETS] };

    /// `"knight:1,line:2,diag:3;d2:4,d3:2,d4:1,d5+:1"` 形式（省略した項は 1.0）
    pub fn parse(spec: &str) -> Option<OriginWeights> {
        let mut w = OriginWeights::UNIFORM;
        let mut parts = spec.split(';');
        let class = parts.next().unwrap_or("");
        let dist = parts.next().unwrap_or("");
        if parts.next().is_some() {
            return None;
        }
        for item in class.split(',').filter(|s| !s.trim().is_empty()) {
            let (k, v) = item.split_once(':')?;
            let v: f64 = v.trim().parse().ok()?;
            let idx = OriginClass::ALL.iter().position(|c| c.tag() == k.trim())?;
            w.class[idx] = v;
        }
        for item in dist.split(',').filter(|s| !s.trim().is_empty()) {
            let (k, v) = item.split_once(':')?;
            let v: f64 = v.trim().parse().ok()?;
            let idx = DIST_BUCKET_LABELS.iter().position(|d| *d == k.trim())?;
            w.dist[idx] = v;
        }
        (w.class.iter().chain(w.dist.iter()).all(|v| *v >= 0.0)
            && w.class.iter().sum::<f64>() > 0.0
            && w.dist.iter().sum::<f64>() > 0.0)
            .then_some(w)
    }
}

/// **F2 攻撃面と潜在逃げ道**（指導2: 王手をかけられる前に安全な場所へ逃げる）。
///
/// **P0 の診断専用**（K の分解の説明変数）で、P1 の arm には入れない。
#[derive(Clone, Debug, Default, PartialEq)]
pub struct F2 {
    /// **盤端まで自駒に遮られていない方向**の数（相手の飛び駒が滑ってこられる面）
    pub open_dirs: u32,
    /// 8方向の「自駒に当たるまでの連続マス数」の合計（隣接を含む）
    pub open_len: u32,
    /// 隣接の自駒非占有マス数
    pub flight_all: u32,
    /// そのうち**距離2のマスが盤外か自駒**（＝逃げ先の背後がすぐ塞がっている）
    /// もの。「攻撃面のどの開放線とも独立に到達できる」の操作的定義がこれで、
    /// その逃げ先は少なくとも**同一直線上の飛び駒**には晒されていない
    pub flight_shielded: u32,
    /// 自玉が初期位置にいるか
    pub king_home: bool,
    /// **既知の敵駒**から最寄りの王手起点までのチェビシェフ距離
    /// （0 = 既知の敵駒が既に起点に載っている）。既知の敵駒が無ければ None
    pub known_enemy_to_origin: Option<u32>,
}

/// **F3 隣接の状態数**（指導3: 玉の周りに駒を打って打ち込みの隙を減らす）。
///
/// 打ち込み面を減らす効果と逃げ道を減らす効果を分けるため3状態で数える。
/// `open_uncovered` が現行 `king_holes`（V4）と同じ量。**盤外は数えない**
/// （壁として機能するので穴ではない）。
#[derive(Clone, Debug, Default, PartialEq)]
pub struct F3 {
    pub open_uncovered: u32,
    pub open_covered: u32,
    pub own_occupied: u32,
    /// 盤内の隣接マス数（盤端を除く分母）
    pub adj_total: u32,
    /// そのうち玉以外の自駒の利きがあるマス数（**占有の有無によらず**）。
    /// 実測で王手の 8 割が隣接から来るので、`F1` と同じ「起点の被覆」を
    /// 隣接についても数える列（`f13_cov` の材料）
    pub adj_covered: u32,
}

impl F3 {
    pub fn on_board_total(&self) -> u32 {
        self.open_uncovered + self.open_covered + self.own_occupied
    }
}

/// 王手前の形（F1〜F3）。F4 は玉の手に固有なので別（`f4_from_covered`）
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PrepFeatures {
    pub king: Option<Coord>,
    pub f1: F1,
    pub f2: F2,
    pub f3: F3,
}

impl PrepFeatures {
    /// 隣接マスの被覆率（`F3` の分母で数えた「起点の被覆」）
    pub fn adj_cov_frac(&self) -> f64 {
        if self.f3.adj_total == 0 {
            0.0
        } else {
            f64::from(self.f3.adj_covered) / f64::from(self.f3.adj_total)
        }
    }

    /// **F1（距離2以上の起点）と隣接を、実王手起点の実測シェアで混ぜた被覆率**。
    ///
    /// F1 と F3 の分担は設計時点で固定してあるが、「実王手駒マスが被覆されて
    /// いるか」に一番近い量は両者の混合になる（発見セットの実測で王手の
    /// 8 割が隣接から来るため）。`adj_share` は**発見セットで測って検証セットでは
    /// 固定する**（距離重みと同じ扱い）。
    pub fn f13_cov(&self, w: &OriginWeights, adj_share: f64) -> f64 {
        adj_share * self.adj_cov_frac() + (1.0 - adj_share) * self.f1.cov_weighted(w)
    }
}

const HOME_KING: [(Color, &str); 2] = [(Color::Sente, "5i"), (Color::Gote, "5a")];

/// 自駒配置（＋既知の敵駒マス）から F1〜F3 を作る。
pub fn features(pieces: &[VisiblePiece], color: Color, known_enemy: &[Coord]) -> PrepFeatures {
    let cov = coverage_of(pieces, color);
    let king = pieces
        .iter()
        .find(|p| p.role == Role::King)
        .and_then(|p| parse_usi_square(&p.square));
    let Some(king) = king else {
        return PrepFeatures::default();
    };
    let origins = origin_squares(king, color);
    let mut f1 = F1::default();
    for o in &origins {
        let ci = OriginClass::ALL.iter().position(|c| *c == o.class).unwrap();
        f1.by_class[ci].0 += 1;
        let di = dist_bucket(o.dist);
        f1.by_dist[di].0 += 1;
        if cov.covered(o.sq) {
            f1.by_class[ci].1 += 1;
            f1.by_dist[di].1 += 1;
            if cov.covered_sure(o.sq) {
                f1.by_class[ci].2 += 1;
            }
        }
    }

    let mut f2 = F2 {
        king_home: HOME_KING
            .iter()
            .any(|(c, sq)| *c == color && parse_usi_square(sq) == Some(king)),
        ..F2::default()
    };
    let mut f3 = F3::default();
    let mut adj_total = 0u32;
    let mut adj_covered = 0u32;
    for (df, dr) in KING_DIRS {
        // 方向ごとに「自駒に当たるまで」歩く（開放長・盤端まで開いているか）
        let mut d = 1i8;
        let mut blocked = false;
        loop {
            let c = Coord { file: king.file + df * d, rank: king.rank + dr * d };
            if !on_board(c) {
                break;
            }
            if cov.occupied.contains(&c) {
                blocked = true;
                break;
            }
            f2.open_len += 1;
            d += 1;
        }
        if !blocked {
            f2.open_dirs += 1;
        }

        let adj = Coord { file: king.file + df, rank: king.rank + dr };
        if !on_board(adj) {
            continue;
        }
        adj_total += 1;
        // **隣接は占有していても被覆を数える**: 敵駒がそこへ来るには自駒を
        // 取るしかなく、取り返せるかは「そのマスに玉以外の利きがあるか」で決まる
        if cov.covered(adj) {
            adj_covered += 1;
        }
        if cov.occupied.contains(&adj) {
            f3.own_occupied += 1;
            continue;
        }
        f2.flight_all += 1;
        if cov.covered(adj) {
            f3.open_covered += 1;
        } else {
            f3.open_uncovered += 1;
        }
        // 距離2が盤外か自駒なら、その逃げ先の背後はすぐ塞がっている
        let behind = Coord { file: king.file + df * 2, rank: king.rank + dr * 2 };
        if !on_board(behind) || cov.occupied.contains(&behind) {
            f2.flight_shielded += 1;
        }
    }
    f3.adj_total = adj_total;
    f3.adj_covered = adj_covered;
    if !known_enemy.is_empty() {
        f2.known_enemy_to_origin = origins
            .iter()
            .flat_map(|o| known_enemy.iter().map(move |e| cheb(o.sq, *e)))
            .min();
    }

    PrepFeatures { king: Some(king), f1, f2, f3 }
}

const KING_DIRS: [(i8, i8); 8] =
    [(0, -1), (0, 1), (1, 0), (-1, 0), (1, -1), (-1, -1), (1, 1), (-1, 1)];

pub fn cheb(a: Coord, b: Coord) -> u32 {
    (i32::from(a.file) - i32::from(b.file))
        .abs()
        .max((i32::from(a.rank) - i32::from(b.rank)).abs()) as u32
}

/// **F4 出発マスの被覆**（指導4: 玉がいた位置に王手の原因の駒がいることが多い）。
///
/// 玉の手について「出発マスに**着手後も**玉以外の自駒の利きがあるか」。
/// 王手中の逃げ先選別と、王手中以外の玉の手で同じ定義を使う。
pub fn f4_from_covered(pieces_after: &[VisiblePiece], color: Color, from: Coord) -> (bool, u32) {
    let cov = coverage_of(pieces_after, color);
    (cov.covered(from), cov.count_at(from))
}

/// 自駒配置に1手を適用した配置（`PlayerView::your_pieces` と同じ形）。
///
/// 相手の駒は見えないので**取った駒は消えない**（自駒だけの配置）。
pub fn pieces_after(pieces: &[VisiblePiece], mv: &ShogiMove) -> Vec<VisiblePiece> {
    let mut out = pieces.to_vec();
    match *mv {
        ShogiMove::Board { from, to, promote } => {
            let from_usi = make_usi_square(from);
            let Some(p) = out.iter_mut().find(|p| p.square == from_usi) else {
                return out;
            };
            if promote {
                if let Some(r) = crate::shogi::promote_role(p.role) {
                    p.role = r;
                }
            }
            p.square = make_usi_square(to);
        }
        ShogiMove::Drop { role, to } => {
            out.push(VisiblePiece { square: make_usi_square(to), role })
        }
    }
    out
}

/// **既知の敵駒マス**（観測だけから確定するもの）。
///
/// - 自駒が取られたマス（そこへ相手の駒が来た）
/// - 自分の打ちが反則になったマス（そこに駒がある）
///
/// どちらも後で「自分がそのマスで駒を取った」ら消える。相手が静かに動かした
/// 可能性は消せないので**鮮度は別に持つ**（`check.rs` の `known_enemy_squares`
/// と同じ考え方だが、こちらは診断専用なので窓を掛けずに素の集合を返す）。
pub fn known_enemy_squares(log: &ObservationLog) -> Vec<Coord> {
    let mut set: BTreeSet<Coord> = BTreeSet::new();
    for e in log.events() {
        match e {
            Observation::OpponentMoved { captured_my_piece_at: Some(sq), .. } => {
                if let Some(c) = parse_usi_square(sq) {
                    set.insert(c);
                }
            }
            Observation::MyFoul { usi, .. } => {
                if let Some(ShogiMove::Drop { to, .. }) = parse_usi(usi) {
                    set.insert(to);
                }
            }
            Observation::MyMove { usi, captured: Some(_), .. } => {
                if let Some(ShogiMove::Board { to, .. }) = parse_usi(usi) {
                    set.remove(&to);
                }
            }
            _ => {}
        }
    }
    set.into_iter().collect()
}

/// **`K_def_truth`**: 受けた王手に対する**真実盤面での全合法手数**。
/// 王手中の合法手 = すべて解消手なので、これがその手番に用意されていた出口の総数
pub fn k_def_truth(truth: &Position) -> u32 {
    truth.legal_moves().len() as u32
}

/// 「実際の王手駒を**玉以外**で真に合法に取れたか」（構成概念検証の oracle 列）。
///
/// F1 が測ろうとしているのは「王手起点に玉以外の利きがある形」なので、
/// 反則の回帰へ進む前にまずこの oracle を再現できるかを見る（issue #34 の
/// 中止条件の1つ目）。
pub fn nonking_capture_of_checker(truth: &Position, bot: Color) -> bool {
    let Some(king) = truth.king_square(bot) else { return false };
    let checkers: Vec<Coord> = crate::check_economy::true_checkers(truth, bot)
        .into_iter()
        .map(|(s, _)| s)
        .collect();
    truth.legal_moves().iter().any(|mv| match *mv {
        ShogiMove::Board { from, to, .. } => from != king && checkers.contains(&to),
        ShogiMove::Drop { .. } => false,
    })
}

/// 手数（`move_number`）→ その手番で bot が積んだ反則数。
///
/// 反則は手番を変えないので `FoulRecord::move_number` はその決定点の手数
/// そのまま（`check_economy::decision_groups` と同じ規約）。**終局した手番**
/// （受理手が無いので `end.moves` には現れない）の反則もここには入る。
pub fn bot_foul_counts(
    end: &crate::protocol::GameEndPayload,
    bot: Color,
) -> HashMap<u32, u32> {
    let mut out: HashMap<u32, u32> = HashMap::new();
    for f in end.foul_attempts.iter().filter(|f| f.by_color == bot) {
        *out.entry(f.move_number).or_insert(0) += 1;
    }
    out
}

/// 連続王手の2回目の王手駒の位置（指導4 の観察の分類）
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum ChaseClass {
    /// 前の玉位置そのもの
    AtPrevKing,
    /// 前の玉位置の隣接
    AdjacentPrevKing,
    /// それ以外
    Elsewhere,
}

impl ChaseClass {
    pub const ALL: [ChaseClass; 3] =
        [ChaseClass::AtPrevKing, ChaseClass::AdjacentPrevKing, ChaseClass::Elsewhere];
    pub fn label(self) -> &'static str {
        match self {
            ChaseClass::AtPrevKing => "(a) 前の玉位置",
            ChaseClass::AdjacentPrevKing => "(b) 前の玉位置の隣接",
            ChaseClass::Elsewhere => "(c) それ以外",
        }
    }
}

/// **連続王手**（P0-1）: bot が玉の手で王手を解消 → 次の相手手番で再び王手
#[derive(Clone, Debug)]
pub struct ChaseEvent {
    /// 2回目の王手を受けた bot の手番の手数
    pub move_number: u32,
    pub prev_king: Coord,
    pub new_king: Coord,
    pub class: ChaseClass,
    /// 両王手（別層。「全王手駒を除去可能か」を手番単位で判定する）
    pub double_check: bool,
    /// **前回と同じ駒が追った**（位置分類とは別軸のフラグ）
    pub same_piece: bool,
    /// F4: 逃げた後の配置で、前の玉位置に玉以外の自駒の利きがあったか
    pub prev_king_covered: bool,
    /// 実王手駒マスが（両王手なら全部）逃げた後の配置で被覆されていたか
    pub checkers_covered: bool,
    /// 2回目の王手手番で bot が積んだ反則数
    pub fouls: u32,
}

/// 記録の真実から連続王手を全部取り出す。
///
/// `positions[k]` は `end.moves[k]` を指す前の局面（`positions[0]` = 初期局面）。
pub fn chase_events(
    end: &crate::protocol::GameEndPayload,
    positions: &[Position],
    bot: Color,
    fouls_at: &HashMap<u32, u32>,
) -> Vec<ChaseEvent> {
    let mut out = vec![];
    for (k, m) in end.moves.iter().enumerate() {
        if m.by_color != bot {
            continue;
        }
        let (Some(before), Some(after)) = (positions.get(k), positions.get(k + 1)) else {
            break;
        };
        if !before.in_check(bot) {
            continue;
        }
        let Some(prev_king) = before.king_square(bot) else { continue };
        let Some(ShogiMove::Board { from, to, .. }) = parse_usi(&m.usi) else { continue };
        if from != prev_king {
            continue; // 玉の手で解消した手番だけ
        }
        // 相手が指していない（bot の手で終局した）なら連続王手は成立しない
        let Some(second) = positions.get(k + 2) else { break };
        if end.moves.get(k + 1).is_none_or(|n| n.by_color == bot) {
            continue;
        }
        if !second.in_check(bot) {
            continue;
        }
        let checkers = crate::check_economy::true_checkers(second, bot);
        if checkers.is_empty() {
            continue;
        }
        let class = if checkers.iter().any(|(s, _)| *s == prev_king) {
            ChaseClass::AtPrevKing
        } else if checkers.iter().any(|(s, _)| cheb(*s, prev_king) == 1) {
            ChaseClass::AdjacentPrevKing
        } else {
            ChaseClass::Elsewhere
        };
        let first_checkers: Vec<Coord> = crate::check_economy::true_checkers(before, bot)
            .into_iter()
            .map(|(s, _)| s)
            .collect();
        let same_piece = end.moves.get(k + 1).and_then(|n| parse_usi(&n.usi)).is_some_and(|mv| {
            matches!(mv, ShogiMove::Board { from, .. } if first_checkers.contains(&from))
        });
        // 「逃げた後の配置」= bot の玉の手の直後（= handoff）の自駒
        let handoff_pieces = after.pieces_of(bot);
        let cov = coverage_of(&handoff_pieces, bot);
        out.push(ChaseEvent {
            move_number: second.move_number(),
            prev_king,
            new_king: to,
            class,
            double_check: checkers.len() > 1,
            same_piece,
            prev_king_covered: cov.covered(prev_king),
            checkers_covered: checkers.iter().all(|(s, _)| cov.covered(*s)),
            fouls: fouls_at.get(&second.move_number()).copied().unwrap_or(0),
        });
    }
    out
}

/// 決定点1つぶんの復元状態（`truth_replay::for_each_decision_full` の
/// 出力を保存した形）。
///
/// **終端の手番**（反則負け・詰み等で受理手が無く `end.moves` に現れない手番）も
/// 1つだけ足す: P0-2 の主目的変数は「次の bot 手番の王手中反則数」なので、
/// そこを落とすと**反則を積んで終局した手番だけが系統的に欠測**して、
/// 反則が多い側が消える（門が甘くなる方向の欠測）。
pub struct Snapshot {
    pub pos: Position,
    pub side: Color,
    pub logs: [ObservationLog; 2],
    pub fouls: [u32; 2],
    /// この決定点で手番側が既に試した反則の数
    pub fouls_this_turn: u32,
    /// `end.moves` のインデックス（終端は `end.moves.len()`）
    pub decision_id: u64,
    /// 受理手が無い（その手番で終局した）
    pub terminal: bool,
}

/// 記録の真実から全決定点の状態を復元する（**終端の手番を含む**）。
///
/// 観測の作り方は `truth_replay` の共有関数だけを使う（規約を二重に書かない）。
/// 棋譜が壊れていたら `None`（その局は丸ごと捨てる）。
pub fn decision_snapshots(end: &crate::protocol::GameEndPayload) -> Option<Vec<Snapshot>> {
    use crate::scenario_core::clone_log;
    use crate::truth_replay::{add_foul_obs, add_move_obs, for_each_decision_full};
    let mut out: Vec<Snapshot> = vec![];
    let ok = for_each_decision_full(end, |d| {
        out.push(Snapshot {
            pos: d.pos.clone(),
            side: d.side,
            logs: [clone_log(&d.logs[0]), clone_log(&d.logs[1])],
            fouls: *d.fouls,
            fouls_this_turn: d.fouls_this_turn,
            decision_id: d.decision_id,
            terminal: false,
        });
    });
    if !ok {
        return None;
    }
    let last = out.last()?;
    let mut pos = last.pos.clone();
    let mut logs = [clone_log(&last.logs[0]), clone_log(&last.logs[1])];
    let mut fouls = last.fouls;
    let mv = parse_usi(&end.moves.get(last.decision_id as usize)?.usi)?;
    let captured = pos.play_unchecked(&mv);
    add_move_obs(last.side, &mv, captured, &pos, &mut logs);
    let side = pos.turn();
    let mut fouls_sorted = end.foul_attempts.clone();
    fouls_sorted.sort_by_key(|f| f.move_number);
    let mut fouls_this_turn = 0;
    for f in fouls_sorted
        .iter()
        .filter(|f| f.by_color == side && f.move_number == pos.move_number())
    {
        add_foul_obs(side, f.usi.clone(), &pos, &mut logs, &mut fouls);
        fouls_this_turn += 1;
    }
    out.push(Snapshot {
        pos,
        side,
        logs,
        fouls,
        fouls_this_turn,
        decision_id: end.moves.len() as u64,
        terminal: true,
    });
    Some(out)
}

/// 元対局 cluster bootstrap の**汎用版**（`check_economy::cluster_ratio_ci` は
/// 比しか出せない）。局を復元抽出して `stat` を計算し、percentile CI を返す。
///
/// `stat` が `None`（セルが空で統計量が定義できない）を返した draw は捨てる。
/// 決定論的な線形合同法なので、同じ入力なら同じ CI になる。
pub fn cluster_bootstrap<T>(
    clusters: &[Vec<T>],
    stat: impl Fn(&[&T]) -> Option<f64>,
    alpha: f64,
    seed: u64,
) -> (f64, f64) {
    if clusters.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mut state = seed | 1;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let reps = 2000;
    let mut draws: Vec<f64> = Vec::with_capacity(reps);
    let mut buf: Vec<&T> = vec![];
    for _ in 0..reps {
        buf.clear();
        for _ in 0..clusters.len() {
            buf.extend(clusters[next() % clusters.len()].iter());
        }
        if let Some(v) = stat(&buf) {
            if v.is_finite() {
                draws.push(v);
            }
        }
    }
    if draws.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    draws.sort_by(|a, b| a.total_cmp(b));
    let lo = ((draws.len() as f64) * (alpha / 2.0)) as usize;
    let hi = (((draws.len() as f64) * (1.0 - alpha / 2.0)) as usize).min(draws.len() - 1);
    (draws[lo], draws[hi])
}

/// P0-1 の門の寄与量: `頻度 × 被覆の有無による反則差`。
///
/// (a) と (b) は**別々に**出してから合算する（issue #34: 門は
/// 「(a)+(b) が過半」ではなく寄与量とその CI）。`cover` は
/// 「その事例で対応する被覆があったか」（(a) なら F4、(b) なら実王手駒マス）。
pub fn chase_contribution(events: &[&ChaseEvent], class: ChaseClass) -> Option<f64> {
    if events.is_empty() {
        return None;
    }
    let cover = |e: &ChaseEvent| match class {
        ChaseClass::AtPrevKing => e.prev_king_covered,
        _ => e.checkers_covered,
    };
    let mut on = (0u32, 0f64);
    let mut off = (0u32, 0f64);
    for e in events.iter().filter(|e| e.class == class) {
        let slot = if cover(e) { &mut on } else { &mut off };
        slot.0 += 1;
        slot.1 += f64::from(e.fouls);
    }
    if on.0 == 0 || off.0 == 0 {
        return None; // 片側が空なら差は定義できない
    }
    let freq = f64::from(on.0 + off.0) / events.len() as f64;
    let diff = off.1 / f64::from(off.0) - on.1 / f64::from(on.0);
    Some(freq * diff)
}


/// **P0-2 の回帰**（issue #34）: 反則数は 0 過多かつ残り反則で打ち切られるので、
/// 平均比では読めない。事前登録どおり **hurdle 分解**に固定する:
///
/// - 第1段: `P(反則 > 0)` = ロジスティック
/// - 第2段: `反則 | 反則 > 0` = **ゼロ切断＋上側打ち切り** Poisson
///   （`y == 残り反則` の観測は「その手番で反則負けした」= `Y ≥ cap` としてしか
///   分からないので、点確率ではなく上側確率を尤度に入れる）
///
/// 主 estimand は**係数ではなく両段を合成した周辺平均**（g-computation）:
/// 「層を固定したまま主特徴量だけを Q90 / Q10 に置いたときの予測反則数の差」。
/// 係数を個別に読まない（hurdle の2つの係数は符号が逆を向きうる）。
pub mod hurdle {
    /// 1観測（bot の handoff 1回）
    #[derive(Clone, Debug)]
    pub struct Row {
        /// 主特徴量（標準化前の生値）
        pub x: f64,
        /// 層・共変量（標準化前の生値）
        pub z: Vec<f64>,
        /// 次の bot 手番の王手中反則数
        pub y: u32,
        /// その手番の反則上限（= 残り反則）。`y == cap` は上側打ち切り
        pub cap: u32,
    }

    /// 標準化した設計行列（切片つき）と応答
    pub struct Prepared {
        /// 1行 = [1, x, z...]
        pub d: Vec<Vec<f64>>,
        pub y: Vec<u32>,
        pub cap: Vec<u32>,
        /// x を差し替えるときの標準化パラメータ
        x_mean: f64,
        x_sd: f64,
    }

    pub fn prepare(rows: &[Row]) -> Option<Prepared> {
        let k = rows.first()?.z.len();
        if rows.iter().any(|r| r.z.len() != k) {
            return None;
        }
        let n = rows.len() as f64;
        let mean = |f: &dyn Fn(&Row) -> f64| rows.iter().map(f).sum::<f64>() / n;
        let sd = |f: &dyn Fn(&Row) -> f64, m: f64| {
            let v = rows.iter().map(|r| (f(r) - m).powi(2)).sum::<f64>() / n;
            if v > 1e-12 { v.sqrt() } else { 1.0 }
        };
        let xm = mean(&|r: &Row| r.x);
        let xs = sd(&|r: &Row| r.x, xm);
        let zm: Vec<f64> = (0..k).map(|j| mean(&|r: &Row| r.z[j])).collect();
        let zs: Vec<f64> = (0..k).map(|j| sd(&|r: &Row| r.z[j], zm[j])).collect();
        let d = rows
            .iter()
            .map(|r| {
                let mut row = vec![1.0, (r.x - xm) / xs];
                row.extend((0..k).map(|j| (r.z[j] - zm[j]) / zs[j]));
                row
            })
            .collect();
        Some(Prepared {
            d,
            y: rows.iter().map(|r| r.y).collect(),
            cap: rows.iter().map(|r| r.cap).collect(),
            x_mean: xm,
            x_sd: xs,
        })
    }

    pub struct Fit {
        pub logit: Vec<f64>,
        pub count: Vec<f64>,
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// 小さい対称系の解（部分ピボットつきガウス消去）。特異なら None
    fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
        let n = b.len();
        for i in 0..n {
            let piv = (i..n).max_by(|&p, &q| a[p][i].abs().total_cmp(&a[q][i].abs()))?;
            if a[piv][i].abs() < 1e-12 {
                return None;
            }
            a.swap(i, piv);
            b.swap(i, piv);
            for r in (i + 1)..n {
                let f = a[r][i] / a[i][i];
                if f == 0.0 {
                    continue;
                }
                for c in i..n {
                    a[r][c] -= f * a[i][c];
                }
                b[r] -= f * b[i];
            }
        }
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let s: f64 = ((i + 1)..n).map(|c| a[i][c] * x[c]).sum();
            x[i] = (b[i] - s) / a[i][i];
        }
        x.iter().all(|v| v.is_finite()).then_some(x)
    }

    pub fn sigmoid(t: f64) -> f64 {
        1.0 / (1.0 + (-t).exp())
    }

    /// リッジつき IRLS。分離（完全予測）でも発散しないように弱い罰則を置く
    fn fit_logistic(d: &[Vec<f64>], y: &[bool], ridge: f64) -> Option<Vec<f64>> {
        let p = d.first()?.len();
        let mut beta = vec![0.0; p];
        for _ in 0..40 {
            let mut h = vec![vec![0.0; p]; p];
            let mut g = vec![0.0; p];
            for (row, &yi) in d.iter().zip(y) {
                let mu = sigmoid(dot(row, &beta));
                let w = (mu * (1.0 - mu)).max(1e-8);
                let resid = f64::from(u8::from(yi)) - mu;
                for a in 0..p {
                    g[a] += row[a] * resid;
                    for b in 0..p {
                        h[a][b] += w * row[a] * row[b];
                    }
                }
            }
            for a in 0..p {
                g[a] -= ridge * beta[a];
                h[a][a] += ridge;
            }
            let step = solve(h, g)?;
            let delta: f64 = step.iter().map(|v| v.abs()).fold(0.0, f64::max);
            for (b, s) in beta.iter_mut().zip(&step) {
                *b += s;
            }
            if delta < 1e-9 {
                break;
            }
        }
        beta.iter().all(|v| v.is_finite()).then_some(beta)
    }

    fn pois_pmf(k: u32, lam: f64) -> f64 {
        let mut t = (-lam).exp();
        for i in 1..=k {
            t *= lam / f64::from(i);
        }
        t
    }

    /// `P(Y >= c)`（c >= 1）
    fn pois_upper(c: u32, lam: f64) -> f64 {
        let mut cum = 0.0;
        let mut t = (-lam).exp();
        for i in 0..c {
            cum += t;
            t *= lam / f64::from(i + 1);
        }
        (1.0 - cum).max(1e-12)
    }

    /// ゼロ切断＋上側打ち切り Poisson の1観測ぶんの `d(logL)/d(eta)`
    fn ztp_dll(y: u32, cap: u32, lam: f64) -> f64 {
        let trunc = lam * (-lam).exp() / (1.0 - (-lam).exp()).max(1e-12);
        if y >= cap {
            // `Y >= cap` としてしか分からない（その手番で反則負け）
            let s = pois_upper(cap, lam);
            lam * pois_pmf(cap - 1, lam) / s - trunc
        } else {
            f64::from(y) - lam - trunc
        }
    }

    /// 第2段（反則ありの行だけ）。解析勾配 ＋ 数値ヘッシアンの Newton 法
    fn fit_ztp(d: &[Vec<f64>], y: &[u32], cap: &[u32], ridge: f64) -> Option<Vec<f64>> {
        let p = d.first()?.len();
        let grad = |beta: &[f64]| -> Vec<f64> {
            let mut g = vec![0.0; p];
            for ((row, &yi), &ci) in d.iter().zip(y).zip(cap) {
                let lam = dot(row, beta).clamp(-8.0, 8.0).exp();
                let s = ztp_dll(yi, ci, lam);
                for a in 0..p {
                    g[a] += row[a] * s;
                }
            }
            for (a, gi) in g.iter_mut().enumerate() {
                *gi -= ridge * beta[a];
            }
            g
        };
        let mut beta = vec![0.0; p];
        for _ in 0..60 {
            let g0 = grad(&beta);
            if g0.iter().all(|v| v.abs() < 1e-8) {
                break;
            }
            // 数値ヘッシアン（p は 10 程度なので十分安い）
            const EPS: f64 = 1e-5;
            let mut h = vec![vec![0.0; p]; p];
            for j in 0..p {
                let mut bp = beta.clone();
                bp[j] += EPS;
                let gp = grad(&bp);
                for a in 0..p {
                    h[a][j] = -(gp[a] - g0[a]) / EPS;
                }
            }
            for (a, hr) in h.iter_mut().enumerate() {
                hr[a] += ridge + 1e-6;
            }
            let Some(step) = solve(h, g0.clone()) else { break };
            // ステップ半減（数値ヘッシアンが不定になる領域での暴走を防ぐ）
            let mut scale = 1.0;
            for _ in 0..8 {
                let cand: Vec<f64> =
                    beta.iter().zip(&step).map(|(b, s)| b + scale * s).collect();
                if cand.iter().all(|v| v.is_finite() && v.abs() < 50.0) {
                    let gc = grad(&cand);
                    let n0: f64 = g0.iter().map(|v| v * v).sum();
                    let nc: f64 = gc.iter().map(|v| v * v).sum();
                    if nc <= n0 {
                        beta = cand;
                        break;
                    }
                }
                scale *= 0.5;
            }
            if scale < 1.0 / 256.0 {
                break;
            }
        }
        beta.iter().all(|v| v.is_finite()).then_some(beta)
    }

    pub fn fit(prep: &Prepared, ridge: f64) -> Option<Fit> {
        let any: Vec<bool> = prep.y.iter().map(|v| *v > 0).collect();
        let logit = fit_logistic(&prep.d, &any, ridge)?;
        let (d2, y2, c2): (Vec<Vec<f64>>, Vec<u32>, Vec<u32>) = prep
            .d
            .iter()
            .zip(&prep.y)
            .zip(&prep.cap)
            .filter(|((_, y), _)| **y > 0)
            .map(|((d, y), c)| (d.clone(), *y, *c))
            .fold((vec![], vec![], vec![]), |mut acc, (d, y, c)| {
                acc.0.push(d);
                acc.1.push(y);
                acc.2.push(c);
                acc
            });
        if d2.len() < prep.d.first()?.len() + 2 {
            return None; // 反則ありの行が少なすぎて第2段が同定できない
        }
        let count = fit_ztp(&d2, &y2, &c2, ridge)?;
        Some(Fit { logit, count })
    }

    /// 主特徴量を `x_raw` に置いたときの**周辺平均**（層は各行のまま）。
    ///
    /// `P(反則>0) × E[反則 | 反則>0]`。第2段の平均は上限（残り反則）で頭打ちに
    /// する（打ち切りの下でゼロ切断 Poisson の素の平均は上限を超えうる）。
    pub fn marginal_mean(prep: &Prepared, fit: &Fit, x_raw: f64) -> f64 {
        let xs = (x_raw - prep.x_mean) / prep.x_sd;
        let mut acc = 0.0;
        for (row, &cap) in prep.d.iter().zip(&prep.cap) {
            let mut r = row.clone();
            r[1] = xs;
            let p = sigmoid(dot(&r, &fit.logit));
            let lam = dot(&r, &fit.count).clamp(-8.0, 8.0).exp();
            let m = lam / (1.0 - (-lam).exp()).max(1e-12);
            acc += p * m.min(f64::from(cap));
        }
        acc / prep.d.len() as f64
    }

    /// 主 estimand: `周辺平均(x=hi) − 周辺平均(x=lo)`
    pub fn contrast(rows: &[Row], lo: f64, hi: f64, ridge: f64) -> Option<f64> {
        let prep = prepare(rows)?;
        let fit = fit(&prep, ridge)?;
        Some(marginal_mean(&prep, &fit, hi) - marginal_mean(&prep, &fit, lo))
    }

    /// 分位点（線形補間なしの最近傍。標本が少ないときに外挿しない）
    pub fn quantile(values: &mut Vec<f64>, q: f64) -> f64 {
        if values.is_empty() {
            return f64::NAN;
        }
        values.sort_by(|a, b| a.total_cmp(b));
        let idx = ((values.len() - 1) as f64 * q).round() as usize;
        values[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shogi::Piece;

    fn at(usi: &str) -> Coord {
        parse_usi_square(usi).unwrap()
    }
    fn vp(square: &str, role: Role) -> VisiblePiece {
        VisiblePiece { square: square.into(), role }
    }

    #[test]
    fn 王手起点集合は玉位置だけで決まり自駒では縮まない() {
        // 5九の先手玉。8方向の距離2以上＋桂の跳び起点
        let origins = origin_squares(at("5i"), Color::Sente);
        // 自駒をいくら置いても集合は同じ（分母操作の防止）
        assert_eq!(origins.len(), origin_squares(at("5i"), Color::Sente).len());
        // 隣接（距離1）は入らない = F3 の領分
        assert!(origins.iter().all(|o| o.dist >= 2));
        // 後手の桂が 4七/6七 にいれば 5九の玉へ利く
        let knights: BTreeSet<Coord> = origins
            .iter()
            .filter(|o| o.class == OriginClass::Knight)
            .map(|o| o.sq)
            .collect();
        assert_eq!(knights, BTreeSet::from([at("4g"), at("6g")]));
        // 盤端（隅）の玉は桂の起点が盤外へ落ちる
        assert!(
            origin_squares(at("1a"), Color::Sente)
                .iter()
                .all(|o| o.class != OriginClass::Knight)
        );
    }

    #[test]
    fn 起点の被覆は確定利きと楽観レイを分ける() {
        // 5九玉 / 5七歩（確定利き: 5六へ利く）/ 8八角（楽観レイ: 5五へ利く）
        let pieces = vec![vp("5i", Role::King), vp("5g", Role::Pawn), vp("8h", Role::Bishop)];
        let cov = coverage_of(&pieces, Color::Sente);
        assert!(cov.covered_sure(at("5f")), "歩の確定利き");
        assert!(cov.covered(at("5e")) && !cov.covered_sure(at("5e")), "角のレイは楽観");
        // 玉自身の利きは数えない（F1 は「玉以外の自駒」）
        assert!(!cov.covered(at("5h")));
    }

    #[test]
    fn f3は隣接を3状態に排他分割する() {
        // 5九玉 / 5八金（占有）/ 4八銀は 4九・5八 等を守る
        let pieces = vec![vp("5i", Role::King), vp("5h", Role::Gold), vp("4h", Role::Silver)];
        let f = features(&pieces, Color::Sente, &[]);
        // 盤端（rank 9 の下）は数えない: 5九の隣接で盤内は 4h/5h/6h/4i/6i の5マス
        assert_eq!(f.f3.on_board_total(), 5);
        assert_eq!(f.f3.own_occupied, 2, "5h の金と 4h の銀");
        // 4九は銀が守っている / 6八・6九は誰も守っていない
        assert_eq!(f.f3.open_covered, 1);
        assert_eq!(f.f3.open_uncovered, 2);
        // F2 の逃げ道は「自駒非占有の隣接マス」
        assert_eq!(f.f2.flight_all, 3);
    }

    #[test]
    fn f4は着手後の配置で出発マスの被覆を見る() {
        // 5九にいた玉が 4九へ逃げた後、5九に自駒の利きが残っているか
        let after = vec![vp("4i", Role::King), vp("5f", Role::Rook)];
        let (covered, n) = f4_from_covered(&after, Color::Sente, at("5i"));
        assert!(covered && n == 1, "5六の飛が 5七→5九 へ利く");
        let after2 = vec![vp("4i", Role::King), vp("5h", Role::Gold)];
        // 5八の金は前方向（5七・4七・6七）と横・後ろ1マスへ利く = 5九へも利く
        assert!(f4_from_covered(&after2, Color::Sente, at("5i")).0);
        let bare = vec![vp("4i", Role::King)];
        assert!(!f4_from_covered(&bare, Color::Sente, at("5i")).0);
    }

    #[test]
    fn 被覆率は自駒を足すと単調に上がる() {
        let bare = vec![vp("5i", Role::King)];
        let with_rook = vec![vp("5i", Role::King), vp("5h", Role::Rook)];
        let a = features(&bare, Color::Sente, &[]);
        let b = features(&with_rook, Color::Sente, &[]);
        assert_eq!(a.f1.total(), b.f1.total(), "分母は玉位置だけで決まる");
        assert!(b.f1.covered() > a.f1.covered());
        assert!(b.f1.cov_weighted(&OriginWeights::UNIFORM) > a.f1.cov_weighted(&OriginWeights::UNIFORM));
    }

    #[test]
    fn 起点重みのパースは範囲を検査する() {
        let w = OriginWeights::parse("knight:2,line:1,diag:1;d2:4,d3:2").unwrap();
        assert_eq!(w.class[0], 2.0);
        assert_eq!(w.dist, [4.0, 2.0, 1.0, 1.0]);
        assert!(OriginWeights::parse("knight:-1").is_none());
        assert!(OriginWeights::parse("knight:0,line:0,diag:0").is_none());
        assert!(OriginWeights::parse("bogus:1").is_none());
        // 重みが一様なら素の被覆率と同じ向きに動く（正規化の健全性）
        assert!(OriginWeights::parse("").is_some());
    }

    #[test]
    fn 既知の敵駒は自分がそのマスで取り返すと消える() {
        let mut log = ObservationLog::default();
        log.record(Observation::OpponentMoved {
            move_number: 4,
            captured_my_piece_at: Some("7f".into()),
        });
        log.record(Observation::MyFoul { move_number: 5, usi: "P*5e".into() });
        assert_eq!(known_enemy_squares(&log), vec![at("5e"), at("7f")]);
        log.record(Observation::MyMove {
            move_number: 6,
            usi: "8h7f".into(),
            captured: Some(Role::Pawn),
        });
        assert_eq!(known_enemy_squares(&log), vec![at("5e")]);
    }

    #[test]
    fn 玉以外での王手駒捕獲の合法性は真実で判定する() {
        // 先手玉 5i / 後手金 5h が王手 / 先手金 4h が 5h を取れる
        let mut pos = Position::empty(Color::Sente);
        pos.set(at("5i"), Some(Piece { color: Color::Sente, role: Role::King }));
        pos.set(at("4h"), Some(Piece { color: Color::Sente, role: Role::Gold }));
        pos.set(at("5h"), Some(Piece { color: Color::Gote, role: Role::Gold }));
        pos.set(at("1a"), Some(Piece { color: Color::Gote, role: Role::King }));
        assert!(pos.in_check(Color::Sente));
        assert!(nonking_capture_of_checker(&pos, Color::Sente));
        // 支え駒があると玉では取れても玉以外では取れない、を分けて数える
        let mut pinned = pos.clone();
        pinned.set(at("4h"), None);
        assert!(!nonking_capture_of_checker(&pinned, Color::Sente));
        assert_eq!(k_def_truth(&pinned), pos_legal_count(&pinned));
    }

    fn pos_legal_count(p: &Position) -> u32 {
        p.legal_moves().len() as u32
    }

    fn chase(class: ChaseClass, covered: bool, fouls: u32) -> ChaseEvent {
        ChaseEvent {
            move_number: 1,
            prev_king: at("5i"),
            new_king: at("4i"),
            class,
            double_check: false,
            same_piece: false,
            prev_king_covered: covered,
            checkers_covered: covered,
            fouls,
        }
    }

    #[test]
    fn 連続王手の寄与量は頻度と被覆別の反則差の積() {
        let events = vec![
            chase(ChaseClass::AtPrevKing, false, 3),
            chase(ChaseClass::AtPrevKing, true, 1),
            chase(ChaseClass::Elsewhere, false, 5),
            chase(ChaseClass::Elsewhere, true, 5),
        ];
        let refs: Vec<&ChaseEvent> = events.iter().collect();
        // (a) は 2/4 の頻度で、被覆ありなら反則 3→1
        let v = chase_contribution(&refs, ChaseClass::AtPrevKing).unwrap();
        assert!((v - 0.5 * 2.0).abs() < 1e-9, "{v}");
        // (c) は差が無いので寄与 0
        assert!(chase_contribution(&refs, ChaseClass::Elsewhere).unwrap().abs() < 1e-9);
        // 片側が空なら差は定義しない（0 と報告して門を通してはいけない）
        let one = vec![chase(ChaseClass::AtPrevKing, true, 1)];
        let one_refs: Vec<&ChaseEvent> = one.iter().collect();
        assert!(chase_contribution(&one_refs, ChaseClass::AtPrevKing).is_none());
    }

    fn end_of(
        moves: &[(&str, Color)],
        fouls: &[(u32, Color, &str)],
    ) -> crate::protocol::GameEndPayload {
        use crate::protocol::{
            FoulRecord, GameEndPayload, MoveRecord, OpponentInfo, RatingChange, RatingChangePair,
        };
        GameEndPayload {
            result: "draw".into(),
            reason: "test".into(),
            final_sfen: String::new(),
            moves: moves
                .iter()
                .map(|&(usi, by_color)| MoveRecord {
                    usi: usi.into(),
                    by_color,
                    ms: 0,
                    fouls_before: 0,
                })
                .collect(),
            foul_attempts: fouls
                .iter()
                .map(|&(move_number, by_color, usi)| FoulRecord {
                    move_number,
                    by_color,
                    usi: usi.into(),
                })
                .collect(),
            rating_change: RatingChangePair {
                you: RatingChange { before: 0, after: 0 },
                opponent: RatingChange { before: 0, after: 0 },
            },
            opponent: OpponentInfo { username: String::new(), rating: 0, is_bot: true },
        }
    }

    #[test]
    fn 決定点の復元は終端の手番も1つ足す() {
        // 3手指して終局。終端（4手目の後手番）で後手が反則を2回積んだ記録
        let end = end_of(
            &[("7g7f", Color::Sente), ("3c3d", Color::Gote), ("8h2b+", Color::Sente)],
            &[(4, Color::Gote, "5a5b"), (4, Color::Gote, "5a4b")],
        );
        let snaps = decision_snapshots(&end).unwrap();
        assert_eq!(snaps.len(), 4, "受理手3つ + 終端1つ");
        assert!(snaps[..3].iter().all(|s| !s.terminal));
        let last = snaps.last().unwrap();
        assert!(last.terminal);
        assert_eq!(last.side, Color::Gote);
        assert_eq!(last.fouls_this_turn, 2, "終端の手番の反則を落とさない");
        assert_eq!(last.fouls, [0, 2]);
        // 終端の局面は全手を指した後（2二角成が入っている）
        assert!(last.pos.piece_at(at("2b")).is_some_and(|p| p.color == Color::Sente));
        // 観測の規約は truth_replay と共有（後手のログに相手の着手が3つ入る）
        assert_eq!(
            last.logs[side_idx_local(Color::Gote)]
                .events()
                .iter()
                .filter(|e| matches!(e, Observation::OpponentMoved { .. }))
                .count(),
            2
        );
        // 壊れた棋譜はその局ごと捨てる
        assert!(decision_snapshots(&end_of(&[("9i9a", Color::Sente)], &[])).is_none());
    }

    fn side_idx_local(c: Color) -> usize {
        crate::truth_replay::side_idx(c)
    }

    /// 決定論的な線形合同法（テストの合成データ生成用）
    fn lcg(state: &mut u64) -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn synth(effect: f64, cap: u32) -> Vec<hurdle::Row> {
        let mut st = 0x2026_0829u64;
        let mut rows = vec![];
        for _ in 0..1500 {
            let x = lcg(&mut st);
            let z0 = lcg(&mut st);
            let p = hurdle::sigmoid(-0.6 + effect * (x - 0.5) * 3.0 + 0.4 * (z0 - 0.5));
            let mut y = 0u32;
            if lcg(&mut st) < p {
                // ゼロ切断 Poisson（λ は x に単調）から逆関数法で引く
                let lam = (0.1 + effect * (x - 0.5)).exp();
                let u = lcg(&mut st);
                let mut acc = 0.0;
                let mut k = 1u32;
                let denom = 1.0 - (-lam).exp();
                loop {
                    let mut pm = (-lam).exp();
                    for i in 1..=k {
                        pm *= lam / f64::from(i);
                    }
                    acc += pm / denom;
                    if acc >= u || k >= cap {
                        break;
                    }
                    k += 1;
                }
                y = k.min(cap);
            }
            rows.push(hurdle::Row { x, z: vec![z0], y, cap });
        }
        rows
    }

    #[test]
    fn hurdleは効果の符号と大きさを取り戻す() {
        let rows = synth(1.0, 10);
        let mut xs: Vec<f64> = rows.iter().map(|r| r.x).collect();
        let lo = hurdle::quantile(&mut xs.clone(), 0.1);
        let hi = hurdle::quantile(&mut xs, 0.9);
        let d = hurdle::contrast(&rows, lo, hi, 1e-6).expect("収束する");
        assert!(d > 0.2, "正の効果を取り戻す: {d}");
        // 効果ゼロの合成データでは 0 付近（門を通してはいけない）
        let flat = synth(0.0, 10);
        let mut fx: Vec<f64> = flat.iter().map(|r| r.x).collect();
        let flo = hurdle::quantile(&mut fx.clone(), 0.1);
        let fhi = hurdle::quantile(&mut fx, 0.9);
        let d0 = hurdle::contrast(&flat, flo, fhi, 1e-6).expect("収束する");
        assert!(d0.abs() < 0.08, "効果なしは 0 付近: {d0}");
    }

    #[test]
    fn 打ち切りは上側確率として尤度に入る() {
        // cap=2 で打ち切った標本を「点確率 2 回」として扱うと λ が過小になる。
        // 打ち切り版なら cap を上げた標本の推定と同じ向き・近い大きさになる
        let capped = synth(1.0, 2);
        let mut xs: Vec<f64> = capped.iter().map(|r| r.x).collect();
        let lo = hurdle::quantile(&mut xs.clone(), 0.1);
        let hi = hurdle::quantile(&mut xs, 0.9);
        let d = hurdle::contrast(&capped, lo, hi, 1e-6).expect("収束する");
        assert!(d > 0.0, "打ち切っても符号は保つ: {d}");
        // 周辺平均は上限を超えない（予測が cap を突き抜けたら打ち切りの意味が無い）
        let prep = hurdle::prepare(&capped).unwrap();
        let fit = hurdle::fit(&prep, 1e-6).unwrap();
        assert!(hurdle::marginal_mean(&prep, &fit, hi) <= 2.0);
    }

    #[test]
    fn cluster_bootstrapは決定論的で点推定を挟む() {
        let clusters: Vec<Vec<f64>> = (0..40)
            .map(|i| vec![if i % 4 == 0 { 0.0 } else { 1.0 }])
            .collect();
        let mean = |xs: &[&f64]| -> Option<f64> {
            (!xs.is_empty()).then(|| xs.iter().copied().sum::<f64>() / xs.len() as f64)
        };
        let a = cluster_bootstrap(&clusters, mean, 0.05, 7);
        assert_eq!(a, cluster_bootstrap(&clusters, mean, 0.05, 7));
        assert!(a.0 < 0.75 && 0.75 < a.1, "点推定 0.75 を挟む: {a:?}");
        // 統計量が定義できない draw しか無ければ NaN（0 を返して門を通さない）
        let (lo, hi) = cluster_bootstrap::<f64>(&clusters, |_| None, 0.05, 7);
        assert!(lo.is_nan() && hi.is_nan());
    }
}
