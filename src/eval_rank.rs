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
//!
//! **タイブレーク乱数はどの列にも載せない**。`adjust` だけでなく `score` /
//! `static_score` もこの乱数を含む（`score = combine_score(...) + foul_probe + adjust`）
//! ので、3つとも `CandidateScore::tiebreak` を引いて出す。順位系の
//! `rank` / `score_gap_top` も**乱数を除いたスコアで**付け直す
//! （エクスポータが `det_order` で並べ替える）。乱数込みの実際の着手順位は
//! 特徴量ではなく識別子列 `engine_rank` に置く（現行方策の baseline 再現用）。

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
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("{} を読めません: {e}", path.display()))?;
    parse_eval_text(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// eval の本文をブロックへ分解する。**壊れた行では止まる**（PR #25 レビュー指摘）:
/// `###` の見出しが1文字崩れただけでも、以降の候補行は**直前のブロックへ接続され**、
/// 反則後の採点が反則前の決定点へ流れ込む（PR #3 で実際に起きた事故と同じ形）。
/// 黙って無視すると `scores=` の照合も件数も合ってしまうので気づけない。
///
/// 止める条件（すべて行番号つき）:
///
/// - `##` で始まるのに `## N手目…` / `### N手目（…の反則後）` として読めない
/// - 括弧の中が正しい USI なのに、その後ろが点数（0〜10）でも `?` でもない
/// - 見出しの前に候補行がある
///
/// USI を含まない行は従来どおり自由記述として読み飛ばす。
pub fn parse_eval_text(text: &str) -> Result<Vec<EvalBlock>, String> {
    let mut out: Vec<EvalBlock> = vec![];
    for (no, line) in text.split('\n').enumerate() {
        let no = no + 1;
        let trimmed = line.trim();
        if trimmed.starts_with("###") {
            let rest = trimmed.strip_prefix("### ").ok_or_else(|| {
                format!("{no}行目: 反則後の見出しとして読めません: {trimmed}")
            })?;
            let num = rest
                .split("手目")
                .next()
                .and_then(|s| s.trim().parse().ok())
                .ok_or_else(|| format!("{no}行目: 手目を読めません: {trimmed}"))?;
            let sub = rest
                .split('（')
                .nth(1)
                .and_then(|s| s.strip_suffix("の反則後）"))
                .ok_or_else(|| {
                    format!("{no}行目: `（和名(USI)の反則後）` の形ではありません: {trimmed}")
                })?;
            out.push(EvalBlock { num, sub: Some(sub.to_string()), entries: vec![] });
            continue;
        }
        if trimmed.starts_with("##") {
            let num = trimmed
                .strip_prefix("## ")
                .and_then(|rest| rest.split("手目").next())
                .and_then(|s| s.trim().parse().ok())
                .ok_or_else(|| format!("{no}行目: `## N手目` として読めません: {trimmed}"))?;
            out.push(EvalBlock { num, sub: None, entries: vec![] });
            continue;
        }
        let Some(entry) = parse_move_line(trimmed, no)? else {
            continue;
        };
        let block = out
            .last_mut()
            .ok_or_else(|| format!("{no}行目: 見出しの前に候補行があります: {trimmed}"))?;
        block.entries.push(entry);
    }
    Ok(out)
}

/// `和名(USI) 点 コメント…` の行（sync_eval.py の MOVE 正規表現と同じ規約）。
///
/// **候補行らしい形（括弧の後ろが点数か `?`）なら、USI が不正でも止まる**
/// （PR #25 レビュー指摘 P2）。`4七金(5g4z) 2` のような打ち間違いを自由記述として
/// 読み飛ばすと、その手だけが教師からも指標からも黙って消える。
/// 括弧の後ろが点数でも `?` でもない行は自由記述（`Ok(None)`）。
fn parse_move_line(line: &str, no: usize) -> Result<Option<(String, Option<u8>)>, String> {
    let Some(open) = line.find('(') else {
        return Ok(None);
    };
    // 開き括弧があるのに閉じていない行は候補行の書き損じ（PR #25 レビュー指摘 P2）。
    // `4七金(5g4g 2` を自由記述として読み飛ばすと、その手だけが黙って消える
    let Some(close) = line[open..].find(')').map(|i| i + open) else {
        return Err(format!("{no}行目: 閉じ括弧がありません: {line}"));
    };
    let usi = &line[open + 1..close];
    let rest = line[close + 1..].trim_start();
    let token = rest.split_whitespace().next().unwrap_or("");
    let score = match token {
        "?" => None,
        t => match t.parse::<u8>() {
            Ok(pt) if pt <= 10 => Some(pt),
            // 点数が続かないなら候補行ではない。ただし USI として正しい括弧が
            // あるのに点数が無いのは「採点し忘れ」なので、そちらは止める
            // （採点は `?` と書く決まり）
            _ => {
                return if parse_usi(usi).is_some() {
                    Err(format!(
                        "{no}行目: {usi} の後ろが点数（0〜10）でも `?` でもありません: {line}"
                    ))
                } else {
                    Ok(None)
                };
            }
        },
    };
    if parse_usi(usi).is_none() {
        return Err(format!(
            "{no}行目: 候補行に見えますが {usi} を USI として読めません: {line}"
        ));
    }
    Ok(Some((usi.to_string(), score)))
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
    // タイブレーク乱数は score / static_score / adjust の3つに同じ量だけ載っている
    // （`adjust` を1回足した形なので、それぞれから引けば決定的な値になる）
    let mut row = vec![
        cand.score - cand.tiebreak,
        cand.static_score - cand.tiebreak,
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
        ctx.top_score - (cand.score - cand.tiebreak),
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
///
/// `rank` / `top_score` は**タイブレーク乱数を除いたスコア**で作ること
/// （[`det_order`] を使う）。乱数込みの順位を混ぜると、特徴量から乱数を外した
/// 意味が無くなる。
pub struct SetContext {
    /// 決定的スコアでの順位（0 始まり）
    pub rank: usize,
    pub n_candidates: usize,
    /// 決定的スコアの最大値
    pub top_score: f64,
    /// この決定状態までに（この手番で）既に消費した反則の数
    pub fouls_this_turn: u32,
}

/// タイブレーク乱数を除いたスコアの降順に並べた添字。
///
/// **同点は USI の辞書順で割る**（PR #25 レビュー指摘 P2）。`ranking` は乱数込みの
/// スコア順に並んでいるので、安定ソートで同点を放置すると**乱数の順序がそのまま
/// `rank` / `rank_frac` に残る**（実測: 決定的スコアが完全同点の候補は 8,145 件、
/// 1,088 決定状態×seed のうち 706 = 65% に存在する）。第2キーを決定的にすれば、
/// 同じ候補集合なら乱数の引き方によらず同じ順位になる。
pub fn det_order(ranking: &[CandidateScore]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..ranking.len()).collect();
    idx.sort_by(|&a, &b| {
        let (x, y) = (
            ranking[a].score - ranking[a].tiebreak,
            ranking[b].score - ranking[b].tiebreak,
        );
        y.partial_cmp(&x)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| ranking[a].usi.cmp(&ranking[b].usi))
    });
    idx
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
            promote_bias: 0.0,
            drop_bias: 0.0,
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

        // **タイブレーク乱数はどの列にも載っていない**。別の乱数を引くと
        // `adjust` と、それを含む `score` / `static_score` が同じ量だけずれる
        // （エンジンがそう作っている）ので、3つとも動かして不変を確かめる
        let d = 0.004;
        let jittered = CandidateScore {
            adjust: cand.adjust + d,
            score: cand.score + d,
            static_score: cand.static_score + d,
            tiebreak: cand.tiebreak + d,
            ..cand.clone()
        };
        assert_eq!(base, feature_row(&view, &jittered, &ctx));
        // 順位系も乱数を除いたスコアで付けること。生のスコア順と決定的スコア順が
        // **食い違う**組を作って、`det_order` が後者を返すことを確かめる
        let hi = CandidateScore { score: 1.000, tiebreak: 0.009, ..cand.clone() };
        let lo = CandidateScore { score: 0.995, tiebreak: 0.000, ..cand.clone() };
        assert!(hi.score > lo.score, "テストの前提: 生のスコアは hi が上");
        assert_eq!(
            det_order(&[hi, lo]),
            vec![1, 0],
            "乱数を除いたスコア（0.991 < 0.995）で並べ替えていない"
        );

        // **決定的スコアが完全同点**なら USI の辞書順で割る（乱数の順序を
        // 安定ソートで温存しない）。`ranking` は乱数込みのスコア順に並んで
        // 来るので、同じ集合を逆順で渡しても同じ順位になること
        let a = CandidateScore { usi: "1a1b".into(), score: 5.009, tiebreak: 0.009, ..cand.clone() };
        let b = CandidateScore { usi: "9i9h".into(), score: 5.000, tiebreak: 0.000, ..cand.clone() };
        assert_eq!(
            (a.score - a.tiebreak, b.score - b.tiebreak),
            (5.0, 5.0),
            "テストの前提: 決定的スコアは同点"
        );
        assert_eq!(det_order(&[a.clone(), b.clone()]), vec![0, 1]);
        assert_eq!(
            det_order(&[b, a]),
            vec![1, 0],
            "同点の順位が入力順（＝乱数の引き方）で変わっている"
        );

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

    /// 壊れた eval で**黙って続けない**こと（PR #25 レビュー指摘 P2）。
    /// とくに `###` の見出しが崩れると、以降の候補行が直前のブロックへ接続され、
    /// 反則後の採点が反則前の決定点へ流れ込む
    #[test]
    fn 壊れたevalは行番号つきで止まる() {
        let ok = [
            "## 61手目（先手番）",
            "4七金(5g4g) 2 コメント",
            "3六角打(B*3f) ?",
            "### 62手目（4七歩打(P*4g)の反則後）",
            "2二歩打(P*2b) 8",
        ]
        .join("\n");
        let ok = ok.as_str();
        let blocks = parse_eval_text(ok).expect("正しい eval が読めない");
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].entries.len(), 2);
        assert_eq!(blocks[1].sub.as_deref(), Some("4七歩打(P*4g)"));

        // 崩れた反則後の見出し（`の反則後）` が無い）は、続く候補行を
        // 直前のブロックへ流し込む代わりに止まる
        let broken = "## 61手目\n4七金(5g4g) 2\n### 62手目（4七歩打(P*4g)）\n2二歩打(P*2b) 8\n";
        let e = parse_eval_text(broken).unwrap_err();
        assert!(e.starts_with("3行目:"), "行番号が出ていない: {e}");

        // USI なのに点数でも ? でもない
        let e = parse_eval_text("## 61手目\n4七金(5g4g) いい手\n").unwrap_err();
        assert!(e.starts_with("2行目:"), "行番号が出ていない: {e}");
        let e = parse_eval_text("## 61手目\n4七金(5g4g) 11\n").unwrap_err();
        assert!(e.starts_with("2行目:"), "0〜10 の範囲外を通している: {e}");

        // **候補行らしいのに USI が不正**（打ち間違い）も止める。自由記述として
        // 読み飛ばすと、その手だけ教師からも指標からも黙って消える
        for bad in ["4七金(5g4z) 2", "4七金(5g4g+x) ?", "4七金(P*4z) 8"] {
            let e = parse_eval_text(&format!("## 61手目\n{bad}\n")).unwrap_err();
            assert!(e.starts_with("2行目:"), "{bad} を通している: {e}");
        }
        // 括弧の後ろが点数でない行は従来どおり自由記述（USI かどうかを問わない）
        assert_eq!(
            parse_eval_text("## 61手目\n出典 (2026-08-07) のメモ\n").unwrap()[0].entries.len(),
            0
        );

        // 閉じ括弧を欠いた候補行も止める（自由記述として消さない）
        let e = parse_eval_text("## 61手目\n4七金(5g4g 2\n").unwrap_err();
        assert!(e.starts_with("2行目:"), "閉じ括弧欠けを通している: {e}");
        // 括弧が一切無い行は従来どおり自由記述
        assert_eq!(
            parse_eval_text("## 61手目\n4七を守る唯一の駒\n").unwrap()[0].entries.len(),
            0
        );

        // 見出しの前の候補行
        let e = parse_eval_text("4七金(5g4g) 2\n").unwrap_err();
        assert!(e.starts_with("1行目:"), "行番号が出ていない: {e}");

        // USI を含まない行は従来どおり自由記述
        assert_eq!(
            parse_eval_text("## 61手目\n（4七を守る唯一の駒）\n").unwrap()[0].entries.len(),
            0
        );

        // インデントされた見出しも見出しとして読む（候補行として直前の
        // ブロックへ流し込まない = この関数が防ぎたい事故そのもの）
        let indented = parse_eval_text("## 61手目\n  ### 62手目（4七歩打(P*4g)の反則後）\n")
            .expect("インデントされた見出しで止まってしまう");
        assert_eq!(indented.len(), 2);
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
