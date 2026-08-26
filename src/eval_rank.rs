//! **人手採点と候補手特徴の突き合わせ**（issue #24 P0 のデータ化部品）。
//!
//! `evals/*.eval.md`（0〜10 点の人手採点）の各ブロックを、元 KIF の決定点
//! （`ply` と、反則後サブ状態なら注入する `fouls` 列）へ**厳格に復元**し、
//! 現行 estimator の候補内訳（[`crate::strategy::CandidateScore`]）と結合して
//! 1行 = `(source_kif, decision_state, seed, usi, human_score, features...)`
//! のデータにする。`bin/export_eval_rank_data` が使う。
//!
//! ## 復元の規約（scenario_core / scripts/quest_review/foul_blocks.py と共有）
//!
//! - eval の見出し `## N手目` は「N 手目を考えさせる」= `ply = N-1` まで再生した
//!   決定点。`scenarios/*.kif` の `*scenario ply=` と同じ数え方
//! - `### N手目（和名(USI)の反則後）` は同じ ply で、元棋譜の `*illegal` 行を
//!   その USI まで注入した状態（`fouls=` ディレクティブと同じ規約）
//! - 元 KIF は `scenarios/archive/<eval名>.kif`（無ければ `scenarios/<eval名>.kif`）
//!
//! 復元した `(ply, fouls)` に対応するシナリオ kif があれば、その `scores=` が
//! eval の採点と**完全一致**することを検査する（`sync_eval.py` が同期している
//! はずのもの。ずれていたら「別の手目・別の反則後サブ状態へ接続した」ので停止する）。
//!
//! ## 特徴量の健全性
//!
//! [`feature_row`] が受け取るのは [`PlayerView`]（自分側の完全既知情報）と
//! [`CandidateScore`]（bot 自身が計算した内訳）だけで、**真の盤面
//! （`Position`）も実戦の正解手もコメント文も棋譜名も渡らない**。
//! 型で担保したうえで、相手駒を動かしても行が変わらないことをテストで検査する。
//! タイブレーク乱数は `CandidateScore::tiebreak` として分離済みなので、
//! 特徴量に載る `adjust` は乱数を引いた決定的な補正だけ。

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::board::{Coord, make_usi_square, parse_usi_square};
use crate::kifu::{Kifu, parse_kif};
use crate::protocol::{Color, PlayerView, Role};
use crate::scenario_core::{kifu_key, scenarios_dir};
use crate::shogi::{Position, ShogiMove, parse_usi};
use crate::strategy::CandidateScore;

/// eval のブロック（1決定状態ぶんの採点表）。
#[derive(Debug, Clone, PartialEq)]
pub struct EvalBlock {
    /// 見出しの「N手目」。決定点は `ply = num - 1`
    pub num: usize,
    /// 反則後サブ状態の見出しキー（`和名(USI)`）。通常ブロックは None
    pub sub: Option<String>,
    /// `(USI, 点)`。`?`（未採点）は None のまま持つ（**欠測であって負例ではない**）
    pub entries: Vec<(String, Option<u8>)>,
}

impl EvalBlock {
    /// 採点済みだけを `(USI, 点)` で（重複 USI は先勝ち。sync_eval.py と同じ規約）
    pub fn scored(&self) -> Vec<(String, u8)> {
        let mut seen = std::collections::HashSet::new();
        self.entries
            .iter()
            .filter_map(|(u, p)| p.map(|p| (u.clone(), p)))
            .filter(|(u, _)| seen.insert(u.clone()))
            .collect()
    }

    /// 反則後サブ状態の USI（見出しの括弧内）
    pub fn sub_usi(&self) -> Option<String> {
        let sub = self.sub.as_ref()?;
        let open = sub.rfind('(')?;
        let close = sub[open..].find(')')?;
        Some(sub[open + 1..open + close].to_string())
    }
}

/// `evals/<stem>.eval.md` を読む。見出しの順で返す。
pub fn parse_eval(path: &Path) -> Result<Vec<EvalBlock>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{} を読めません: {e}", path.display()))?;
    Ok(parse_eval_text(&text))
}

pub fn parse_eval_text(text: &str) -> Vec<EvalBlock> {
    let mut out: Vec<EvalBlock> = vec![];
    for line in text.split('\n') {
        if let Some(rest) = line.strip_prefix("### ") {
            if let (Some(num), Some(sub)) = (
                rest.split("手目").next().and_then(|s| s.trim().parse().ok()),
                rest.split('（').nth(1).and_then(|s| s.strip_suffix("の反則後）")),
            ) {
                out.push(EvalBlock { num, sub: Some(sub.to_string()), entries: vec![] });
                continue;
            }
        }
        if let Some(rest) = line.strip_prefix("## ") {
            if let Some(num) = rest.split("手目").next().and_then(|s| s.trim().parse().ok()) {
                out.push(EvalBlock { num, sub: None, entries: vec![] });
                continue;
            }
        }
        let Some(block) = out.last_mut() else { continue };
        if let Some(e) = parse_move_line(line.trim()) {
            block.entries.push(e);
        }
    }
    out
}

/// `和名(USI) 点 コメント…` の行（sync_eval.py の MOVE 正規表現と同じ規約）
fn parse_move_line(line: &str) -> Option<(String, Option<u8>)> {
    let open = line.find('(')?;
    let close = line[open..].find(')')? + open;
    let usi = &line[open + 1..close];
    parse_usi(usi)?;
    let rest = line[close + 1..].trim_start();
    let token = rest.split_whitespace().next()?;
    if token == "?" {
        return Some((usi.to_string(), None));
    }
    let pt: u8 = token.parse().ok()?;
    (pt <= 10).then(|| (usi.to_string(), Some(pt)))
}

/// eval の stem に対応する元 KIF（`scenarios/archive/<stem>.kif` 優先）。
pub fn source_kif_path(stem: &str) -> Option<PathBuf> {
    let archive = scenarios_dir().join("archive").join(format!("{stem}.kif"));
    if archive.exists() {
        return Some(archive);
    }
    let direct = scenarios_dir().join(format!("{stem}.kif"));
    direct.exists().then_some(direct)
}

/// `(ply, fouls列)` -> シナリオ（名前と採点表）。同じ元棋譜のシナリオだけを集める。
pub struct ScenarioIndex {
    by_key: HashMap<(usize, Vec<String>), (String, Vec<(String, u8)>)>,
}

impl ScenarioIndex {
    /// `scenarios/*.kif` のうち元棋譜が一致するものを (ply, fouls) で索引する。
    /// アーカイブ（`scenarios/archive/`）は決定点シナリオではないので見ない。
    pub fn build(source: &Kifu) -> Self {
        let key = kifu_key(source);
        let mut by_key = HashMap::new();
        let mut paths: Vec<PathBuf> = std::fs::read_dir(scenarios_dir())
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "kif"))
            .collect();
        paths.sort();
        for path in paths {
            let Ok(text) = std::fs::read_to_string(&path) else { continue };
            let Ok(kifu) = parse_kif(&text) else { continue };
            let Some(ply) = kifu.directives.get("ply").and_then(|s| s.parse::<usize>().ok()) else {
                continue;
            };
            if kifu_key(&kifu) != key {
                continue;
            }
            let fouls: Vec<String> = split_list(kifu.directives.get("fouls"));
            let scores: Vec<(String, u8)> = split_list(kifu.directives.get("scores"))
                .iter()
                .filter_map(|x| {
                    let (u, p) = x.split_once(':')?;
                    Some((u.to_string(), p.parse().ok()?))
                })
                .collect();
            let name = path.file_stem().unwrap().to_string_lossy().to_string();
            by_key.insert((ply, fouls), (name, scores));
        }
        ScenarioIndex { by_key }
    }

    pub fn get(&self, ply: usize, fouls: &[String]) -> Option<&(String, Vec<(String, u8)>)> {
        self.by_key.get(&(ply, fouls.to_vec()))
    }

    pub fn len(&self) -> usize {
        self.by_key.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_key.is_empty()
    }
}

fn split_list(v: Option<&String>) -> Vec<String> {
    v.map(|s| {
        s.split(',')
            .map(str::trim)
            .filter(|x| !x.is_empty())
            .map(str::to_string)
            .collect()
    })
    .unwrap_or_default()
}

/// 特徴量の列名（[`feature_row`] の並びと1対1）。
pub const FEATURE_COLUMNS: &[&str] = &[
    // --- 現行評価の内訳（bot が実際に計算した値）
    "score",
    "static_score",
    "gain",
    "static_gain",
    "p_legal",
    "foul_cost",
    "adjust",
    "foul_probe",
    "capture_value",
    "risk",
    "capture_bet_penalty",
    "checker_removal",
    "value_nn",
    "link",
    "promo",
    "hand_option",
    "own_zone",
    "mate_threat",
    "mate_risk",
    "king_holes",
    "board_discount",
    "depth2",
    // --- 候補集合の中での位置
    "rank",
    "rank_frac",
    "score_gap_top",
    "n_candidates",
    // --- 状態（bot 視点で得られるものだけ）
    "move_number",
    "in_check",
    "fouls_this_turn",
    "fouls_remaining",
    "own_hand_total",
    // --- 着手
    "is_drop",
    "is_promote",
    "mover_value",
    "role_pawn",
    "role_lance",
    "role_knight",
    "role_silver",
    "role_gold",
    "role_bishop",
    "role_rook",
    "role_king",
    "role_promoted",
    "to_rank_own",
    "advance",
    "dist_own_king",
];

/// 候補1件ぶんの特徴量。
///
/// **入力は自分側の完全既知情報（`view`）と bot 自身の内訳（`cand`）だけ**で、
/// 真の盤面・実戦の手・棋譜の識別子は渡らない（issue #24 の非目的）。
/// `ctx` は候補集合の統計（順位・首位との差・候補数）。
pub fn feature_row(view: &PlayerView, cand: &CandidateScore, ctx: &SetContext) -> Vec<f64> {
    let mv = parse_usi(&cand.usi);
    let (to, from, promote, role) = match mv {
        Some(ShogiMove::Board { from, to, promote }) => (
            Some(to),
            Some(from),
            promote,
            view.your_pieces
                .iter()
                .find(|p| p.square == make_usi_square(from))
                .map(|p| p.role),
        ),
        Some(ShogiMove::Drop { role, to }) => (Some(to), None, false, Some(role)),
        None => (None, None, false, None),
    };
    let own_king = view
        .your_pieces
        .iter()
        .find(|p| p.role == Role::King)
        .and_then(|p| parse_usi_square(&p.square));
    let own_rank = |c: Coord| -> f64 {
        // 自分から見た段（1 = 自陣の最奥、9 = 敵陣の最奥）
        match view.your_color {
            Color::Sente => f64::from(10 - i32::from(c.rank)),
            Color::Gote => f64::from(i32::from(c.rank)),
        }
    };
    let is_drop = from.is_none() && to.is_some();
    let mut row = vec![
        cand.score,
        cand.static_score,
        cand.gain,
        cand.static_gain,
        cand.p_legal,
        cand.foul_cost,
        // タイブレーク乱数を除いた決定的な補正だけを載せる
        cand.adjust - cand.tiebreak,
        cand.foul_probe,
        cand.capture_value,
        cand.risk,
        cand.capture_bet_penalty,
        cand.checker_removal,
        cand.value_nn,
        cand.link,
        cand.promo,
        cand.hand_option,
        cand.own_zone,
        cand.mate_threat,
        cand.mate_risk,
        cand.king_holes,
        cand.board_discount,
        f64::from(u8::from(cand.depth2)),
        ctx.rank as f64,
        if ctx.n_candidates > 1 {
            ctx.rank as f64 / (ctx.n_candidates - 1) as f64
        } else {
            0.0
        },
        ctx.top_score - cand.score,
        ctx.n_candidates as f64,
        f64::from(view.move_number),
        f64::from(u8::from(view.you_in_check)),
        f64::from(ctx.fouls_this_turn),
        f64::from(10u32.saturating_sub(view.fouls.you)),
        view.your_hand.values().map(|n| f64::from(*n)).sum(),
        f64::from(u8::from(is_drop)),
        f64::from(u8::from(promote)),
        role.map(exchange_value_of).unwrap_or(0.0),
    ];
    for r in [
        Role::Pawn,
        Role::Lance,
        Role::Knight,
        Role::Silver,
        Role::Gold,
        Role::Bishop,
        Role::Rook,
        Role::King,
    ] {
        row.push(f64::from(u8::from(role == Some(r))));
    }
    row.push(f64::from(u8::from(role.is_some_and(is_promoted_role))));
    row.push(to.map(own_rank).unwrap_or(0.0));
    row.push(match (from, to) {
        (Some(f), Some(t)) => own_rank(t) - own_rank(f),
        _ => 0.0,
    });
    row.push(match (own_king, to) {
        (Some(k), Some(t)) => f64::from(
            (i32::from(k.file) - i32::from(t.file))
                .abs()
                .max((i32::from(k.rank) - i32::from(t.rank)).abs()),
        ),
        _ => 0.0,
    });
    debug_assert_eq!(row.len(), FEATURE_COLUMNS.len());
    row
}

/// 候補集合ぜんたいの文脈（[`feature_row`] の順位系特徴に使う）。
pub struct SetContext {
    pub rank: usize,
    pub n_candidates: usize,
    pub top_score: f64,
    /// この決定状態までに（この手番で）既に消費した反則の数
    pub fouls_this_turn: u32,
}

fn is_promoted_role(r: Role) -> bool {
    matches!(
        r,
        Role::Tokin
            | Role::Promotedlance
            | Role::Promotedknight
            | Role::Promotedsilver
            | Role::Horse
            | Role::Dragon
    )
}

/// 交換価値（`strategy::exchange_value` はクレート内可視。同じ表を使う）
fn exchange_value_of(r: Role) -> f64 {
    crate::strategy::exchange_value(r)
}

/// 元 KIF の決定点で、この手番側が実戦で試みた反則列（USI）。
/// 棋譜末尾（反則負け等で `plies` に手が無い決定点）は `trailing_fouls`。
pub fn real_fouls_at(kifu: &Kifu, pos: &Position, side: Color, ply: usize) -> Vec<String> {
    let raw = if ply < kifu.plies.len() {
        &kifu.plies[ply].fouls
    } else {
        &kifu.trailing_fouls
    };
    raw.iter()
        .map(|f| crate::scenario_core::resolve_foul(pos, side, f))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{ClockState, FoulCounts, GameStatus};
    use crate::scenario_core::{make_view, replay};

    fn evals_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("evals")
    }

    fn eval_paths() -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = std::fs::read_dir(evals_dir())
            .expect("evals/ を読めません")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.to_string_lossy().ends_with(".eval.md"))
            .collect();
        v.sort();
        v
    }

    fn stem_of(p: &Path) -> String {
        p.file_name().unwrap().to_string_lossy().replace(".eval.md", "")
    }

    #[test]
    fn eval_の全ブロックが元kifの決定点へ復元できる() {
        let mut states = 0usize;
        let mut scored = 0usize;
        for path in eval_paths() {
            let stem = stem_of(&path);
            let src = source_kif_path(&stem)
                .unwrap_or_else(|| panic!("{stem}: 元 KIF が見つかりません"));
            let kifu = parse_kif(&std::fs::read_to_string(&src).unwrap()).unwrap();
            for block in parse_eval(&path).unwrap() {
                let ply = block.num - 1;
                assert!(
                    ply <= kifu.plies.len(),
                    "{stem}: {}手目が棋譜の手数 {} を超えています",
                    block.num,
                    kifu.plies.len()
                );
                let rep = replay(&kifu, ply);
                let side = rep.pos.turn();
                let real = real_fouls_at(&kifu, &rep.pos, side, ply);
                if let Some(usi) = block.sub_usi() {
                    let idx = real.iter().position(|u| *u == usi).unwrap_or_else(|| {
                        panic!(
                            "{stem}: {}手目の反則後ブロック {usi} が棋譜の反則列 {real:?} に無い",
                            block.num
                        )
                    });
                    assert!(idx < real.len());
                }
                states += 1;
                scored += block.scored().len();
            }
        }
        assert!(states > 250, "決定状態がほとんど読めていない（{states}）");
        assert!(scored > 4500, "採点がほとんど読めていない（{scored}）");
    }

    #[test]
    fn 復元したブロックの採点表がシナリオの採点表と一致する() {
        let mut checked = 0usize;
        for path in eval_paths() {
            let stem = stem_of(&path);
            let src = source_kif_path(&stem).unwrap();
            let kifu = parse_kif(&std::fs::read_to_string(&src).unwrap()).unwrap();
            let index = ScenarioIndex::build(&kifu);
            for block in parse_eval(&path).unwrap() {
                let ply = block.num - 1;
                let rep = replay(&kifu, ply);
                let side = rep.pos.turn();
                let real = real_fouls_at(&kifu, &rep.pos, side, ply);
                let fouls: Vec<String> = match block.sub_usi() {
                    None => vec![],
                    Some(usi) => {
                        let idx = real.iter().position(|u| *u == usi).unwrap();
                        real[..=idx].to_vec()
                    }
                };
                let Some((name, scores)) = index.get(ply, &fouls) else {
                    continue;
                };
                let mut want = block.scored();
                let mut got = scores.clone();
                want.sort();
                got.sort();
                assert_eq!(
                    want, got,
                    "{stem} の {}手目（{:?}）→ シナリオ {name} の採点表が一致しません",
                    block.num, block.sub
                );
                checked += 1;
            }
        }
        assert!(checked > 150, "照合できたシナリオが少なすぎます（{checked}）");
    }

    /// `scripts/quest_review/foul_blocks.py` の FOUL_MAP（Python 側の対応表）と、
    /// ここの `(ply, fouls)` 由来の対応が一致することを検査する。
    /// 表がずれると「反則後の採点が反則前のシナリオへ流れ込む」（PR #3 の事故）。
    #[test]
    fn 反則後ブロックの対応がpython側の表と一致する() {
        let py = std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("scripts/quest_review/foul_blocks.py"),
        )
        .expect("foul_blocks.py を読めません");
        // `(30, "1二飛(5b1b)"): "quest31-m030f1",` の行を拾う
        let re_line = |l: &str| -> Option<(usize, String, String)> {
            let l = l.trim();
            let rest = l.strip_prefix('(')?;
            let (num, rest) = rest.split_once(',')?;
            let num: usize = num.trim().parse().ok()?;
            let rest = rest.trim().strip_prefix('"')?;
            let (key, rest) = rest.split_once('"')?;
            let rest = rest.trim().strip_prefix("):")?;
            let name = rest.trim().trim_start_matches('"');
            let name = name.split('"').next()?;
            Some((num, key.to_string(), name.to_string()))
        };
        let mut py_map: Vec<(usize, String, String)> =
            py.lines().filter_map(re_line).collect();
        py_map.sort();
        assert!(py_map.len() >= 15, "python 側の表を読めていません（{}件）", py_map.len());

        // eval 側から同じ対応を作る
        let mut ours: Vec<(usize, String, String)> = vec![];
        for path in eval_paths() {
            let stem = stem_of(&path);
            let Some(src) = source_kif_path(&stem) else { continue };
            let kifu = parse_kif(&std::fs::read_to_string(&src).unwrap()).unwrap();
            let index = ScenarioIndex::build(&kifu);
            for block in parse_eval(&path).unwrap() {
                let Some(sub) = block.sub.clone() else { continue };
                let ply = block.num - 1;
                let rep = replay(&kifu, ply);
                let real = real_fouls_at(&kifu, &rep.pos, rep.pos.turn(), ply);
                let usi = block.sub_usi().unwrap();
                let Some(idx) = real.iter().position(|u| *u == usi) else { continue };
                if let Some((name, _)) = index.get(ply, &real[..=idx]) {
                    ours.push((block.num, sub, name.clone()));
                }
            }
        }
        ours.sort();
        // python の表は quest 系だけ（arena-check の単一決定点 eval は
        // sync_eval.py が kif ヘッダから引くので表に載らない）
        let ours_quest: Vec<(usize, String, String)> = ours
            .iter()
            .filter(|(_, _, n)| n.starts_with("quest"))
            .cloned()
            .collect();
        assert_eq!(
            py_map, ours_quest,
            "foul_blocks.py の FOUL_MAP と (ply, fouls) 由来の対応がずれています"
        );
    }

    /// **真実盤面が特徴量に漏れていない**ことの検査（issue #24 のガード）。
    /// 相手の駒を（王手関係の無いところで）動かしても行は変わらない。
    #[test]
    fn 特徴量は相手の駒配置に依存しない() {
        let kifu = parse_kif(
            &std::fs::read_to_string(source_kif_path("quest_20260731").unwrap()).unwrap(),
        )
        .unwrap();
        let rep = replay(&kifu, 60);
        let side = rep.pos.turn();
        let view = make_view(&rep.pos, side, &rep.fouls);
        let cand = CandidateScore {
            usi: "5g4g".into(),
            static_score: 1.0,
            static_gain: 2.0,
            score: 1.5,
            gain: 2.5,
            p_legal: 0.9,
            foul_cost: 5.0,
            adjust: 0.123,
            depth2: true,
            checker_removal: 0.0,
            capture_bet_penalty: 0.1,
            mate_threat: 0.0,
            mate_risk: 0.0,
            king_holes: 0.0,
            value_nn: 0.3,
            capture_value: 1.0,
            risk: 0.5,
            link: 0.2,
            promo: 0.1,
            hand_option: 0.0,
            board_discount: 0.0,
            own_zone: 0.0,
            probe_unit: 0.0,
            probe_mass: 0.0,
            probe_concentration: 0.0,
            foul_probe: 0.0,
            tiebreak: 0.003,
        };
        let ctx = SetContext { rank: 3, n_candidates: 40, top_score: 4.0, fouls_this_turn: 0 };
        let base = feature_row(&view, &cand, &ctx);

        // **別の真実**（相手の駒を全部消した盤面）から作った view でも行は同じ。
        // 特徴量が真の盤面を一切見ていないことの実測ガード
        let mut fake = rep.pos.clone();
        let opp = side.other();
        for (c, p) in rep.pos.pieces() {
            if p.color == opp && p.role != Role::King {
                fake.set(c, None);
            }
        }
        let fake_view = make_view(&fake, side, &rep.fouls);
        assert_ne!(
            format!("{:?}", rep.pos.pieces().collect::<Vec<_>>()),
            format!("{:?}", fake.pieces().collect::<Vec<_>>()),
            "テストの前提: 盤面は実際に変えている"
        );
        assert_eq!(base, feature_row(&fake_view, &cand, &ctx));

        // 自分側の情報（持ち駒）を変えれば行は変わる = 死んだ特徴ではない
        let mut mine = view.clone();
        mine.your_hand.insert(Role::Gold, 3);
        assert_ne!(base, feature_row(&mine, &cand, &ctx));

        // 乱数は載っていない
        let mut jittered = cand.clone();
        jittered.adjust = cand.adjust + 0.004;
        jittered.tiebreak = cand.tiebreak + 0.004;
        assert_eq!(base, feature_row(&view, &jittered, &ctx));

        // view の残りの項目（時計・状態）は行に載っていない: 変えても不変
        let mut noise = view.clone();
        noise.clocks = ClockState { sente_ms: 1, gote_ms: 2, running: None, server_time: 3 };
        noise.status = GameStatus::Playing;
        noise.opponent_in_check = !view.opponent_in_check;
        assert_eq!(base, feature_row(&noise, &cand, &ctx));

        // 反則の残数は自分側の観測なので効く
        let mut used = view.clone();
        used.fouls = FoulCounts { you: view.fouls.you + 3, opponent: view.fouls.opponent };
        assert_ne!(base, feature_row(&used, &cand, &ctx));
    }

    #[test]
    fn 特徴量の列数と列名が一致する() {
        assert_eq!(FEATURE_COLUMNS.len(), 46);
        let mut names = FEATURE_COLUMNS.to_vec();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), FEATURE_COLUMNS.len(), "列名が重複しています");
    }
}
