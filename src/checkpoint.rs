//! checkpoint arena のデータ層: 固定チェックポイントデッキの抽出・保存・復元。
//!
//! 目的は「通常 arena へ送る前に**明確な悪化**を安価に除外する破滅検出器」で、
//! 小さな改善の証明ではない（issue #19）。最終採否は通常 arena のまま。
//!
//! ## v1 は手番境界のみ
//!
//! checkpoint は「直前の受理手が完了し、次の手番がまだ反則を一度も
//! 試していない時点」だけを採る。`scenario_core::replay(kifu, ply)` の
//! 自然な境界と一致するので、`foul_tried` は仕様上必ず空になり、注入済みの
//! 反則手を再試行できてしまう問題（`foul_tried.len()` を読む
//! `check_foul_prior_boost` が評価そのものを変える）を避けられる。
//! 同一手番内の反則後 checkpoint を扱うなら manifest へ `foul_tried` を
//! 明示的に保存し、MyFoul 観測・累積反則数との整合テストを足すこと。
//!
//! ## 保存形式は KIF ＋ manifest
//!
//! 抽出元は `truth_replay`（`game:end` の真実）だが、デッキには**元 KIF**を
//! 置き、復元は `scenario_core::replay` に任せる。回帰した checkpoint を
//! そのまま `bin/scenario <path.kif> diag` / `rank_probe` へ渡せるからで、
//! 「再現コマンドつき」の要件がこれでほぼ自動的に満たされる。
//! 両経路が同じ状態になることは本モジュールのテストで検査する。

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::kifu::{Kifu, parse_kif};
use crate::observation::ObservationLog;
use crate::protocol::{Color, GameEndPayload};
use crate::scenario_core::replay;
use crate::selfplay::{StartState, mix};
use crate::truth_replay::for_each_decision_full;

pub const DECK_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Deck {
    pub version: u32,
    /// 抽出元（例 "arena run 123456789" / "records/"）
    pub source: String,
    /// 継続対局で checkpoint 側と対戦させる凍結版。
    /// **凍結版を追加したら arena.yml の baselines 既定値と同じ運用で更新する**
    pub opponent: String,
    /// 元対局がその checkpoint から少なくとも何手続いたか（適格条件）
    pub min_remaining_plies: u32,
    pub entries: Vec<DeckEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeckEntry {
    /// 例 "run123-game017-ply72"
    pub id: String,
    /// manifest からの相対パス
    pub kif: String,
    /// 何手目まで再生するか（= その手を指す直前の手番境界）
    pub ply: usize,
    /// "dev" | "validation"
    pub split: String,
    /// 層化用のタグ（手番・進行度・被王手・反則・元対局の結末）
    pub tags: Vec<String>,
}

impl DeckEntry {
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }

    /// 開始手番（`sente` / `gote` タグ）
    pub fn side(&self) -> Color {
        if self.has_tag("gote") {
            Color::Gote
        } else {
            Color::Sente
        }
    }
}

impl Deck {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("{} を読めません: {e}", path.display()))?;
        let deck: Deck = serde_json::from_str(&text)
            .map_err(|e| format!("{} を解釈できません: {e}", path.display()))?;
        if deck.version != DECK_VERSION {
            return Err(format!(
                "デッキ version {} は未対応です（対応 {DECK_VERSION}）",
                deck.version
            ));
        }
        Ok(deck)
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{} を作れません: {e}", dir.display()))?;
        }
        let text = serde_json::to_string_pretty(self)
            .map_err(|e| format!("manifest を書けません: {e}"))?;
        std::fs::write(path, text + "\n")
            .map_err(|e| format!("{} を書けません: {e}", path.display()))
    }

    /// manifest の内容ハッシュ（JSONL へ記録して、比較する2本が同じデッキで
    /// 走ったことを後から確認できるようにする）
    pub fn hash(&self) -> String {
        use sha2::{Digest, Sha256};
        let canonical = serde_json::to_string(self).unwrap_or_default();
        let digest = Sha256::digest(canonical.as_bytes());
        digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
    }

    pub fn entry(&self, id: &str) -> Option<&DeckEntry> {
        self.entries.iter().find(|e| e.id == id)
    }
}

/// checkpoint の開始状態を復元する。`dir` は manifest の置いてあるディレクトリ。
///
/// **手番境界の検査つき**: `replay` は ply 手目の反則を適用しないので、
/// KIF 側にその手番の反則があれば「復元した状態」と「元対局の状態」が食い違う。
/// v1 のデッキはそういう ply を抽出しないので、ここで検出したら不整合を返す。
pub fn restore(dir: &Path, entry: &DeckEntry) -> Result<StartState, String> {
    let kifu = load_kifu(dir, entry)?;
    if entry.ply > kifu.plies.len() {
        return Err(format!(
            "{}: ply={} が棋譜の手数 {} を超えています",
            entry.id,
            entry.ply,
            kifu.plies.len()
        ));
    }
    if kifu.plies.get(entry.ply).is_some_and(|p| !p.fouls.is_empty()) {
        return Err(format!(
            "{}: ply={} は手番境界ではありません（その手番に反則試行あり）",
            entry.id, entry.ply
        ));
    }
    let rep = replay(&kifu, entry.ply);
    Ok(StartState {
        pos: rep.pos,
        logs: rep.logs,
        fouls: rep.fouls,
        plies: rep.plies,
    })
}

pub fn load_kifu(dir: &Path, entry: &DeckEntry) -> Result<Kifu, String> {
    let path = kif_path(dir, entry);
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("{} を読めません: {e}", path.display()))?;
    parse_kif(&text).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn kif_path(dir: &Path, entry: &DeckEntry) -> PathBuf {
    dir.join(&entry.kif)
}

// ---------------------------------------------------------------------------
// 抽出
// ---------------------------------------------------------------------------

/// 抽出候補（1決定点）
#[derive(Debug, Clone)]
pub struct Candidate {
    pub ply: usize,
    pub side: Color,
    pub in_check: bool,
    pub fouls: [u32; 2],
    /// この checkpoint 以降に元対局が続いた手数
    pub remaining_plies: u32,
    pub tags: Vec<String>,
}

/// 進行度の層。issue の「早中盤（ply 30〜50）/ 中盤 / 終盤」。
/// 早中盤が抜けると「序盤〜中盤に効く変更」が原理的に測れないので必ず層に入れる
pub fn phase_tag(ply: usize) -> &'static str {
    match ply {
        0..=49 => "early-middle",
        50..=89 => "middle",
        _ => "endgame",
    }
}

/// 元対局の結末を、開始手番側から見た粗い優劣の代理にする（issue の P0 方針）。
/// control 継続勝率で選別する方式に切り替えるなら、**選定 seed と計測 seed を
/// 分ける**こと（同じ結果で選んで同じ結果を評価しない）
fn eventual_tag(result: &str, side: Color) -> &'static str {
    let winner = match result {
        "sente_win" => Some(Color::Sente),
        "gote_win" => Some(Color::Gote),
        _ => None,
    };
    match winner {
        Some(w) if w == side => "eventual-win",
        Some(_) => "eventual-loss",
        None => "eventual-draw",
    }
}

/// 適格な checkpoint（手番境界・最低残り手数）を列挙する。
/// 棋譜が壊れていたら None（その局は丸ごと捨てる）
pub fn candidates(end: &GameEndPayload, min_remaining_plies: u32) -> Option<Vec<Candidate>> {
    let total = end.moves.len() as u32;
    let mut out = vec![];
    let ok = for_each_decision_full(end, |d| {
        // v1 は手番境界のみ（その手番でまだ反則を試していない時点）
        if d.fouls_this_turn != 0 {
            return;
        }
        // 初手は観測ゼロで prewarm も無く、実質「初期局面からの arena」なので採らない
        if d.plies == 0 {
            return;
        }
        let remaining = total - d.plies;
        if remaining < min_remaining_plies {
            return;
        }
        let in_check = d.pos.in_check(d.side);
        let tags = vec![
            (if d.side == Color::Sente { "sente" } else { "gote" }).to_string(),
            phase_tag(d.plies as usize).to_string(),
            (if in_check { "in-check" } else { "no-check" }).to_string(),
            (if d.fouls[0] + d.fouls[1] > 0 { "fouls-before" } else { "fouls-none" }).to_string(),
            eventual_tag(&end.result, d.side).to_string(),
        ];
        out.push(Candidate {
            ply: d.plies as usize,
            side: d.side,
            in_check,
            fouls: *d.fouls,
            remaining_plies: remaining,
            tags,
        });
    });
    if ok { Some(out) } else { None }
}

/// 文字列の決定論的ハッシュ（FNV-1a → SplitMix64。デッキ抽出の決定論化用）
pub fn stable_hash(seed: u64, s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    mix(seed ^ h)
}

/// 元対局1つぶんの抽出候補
pub struct GameCandidates {
    pub game_id: String,
    pub candidates: Vec<Candidate>,
}

/// 「原則1棋譜につき1checkpoint」で層化抽出する。
///
/// 既知の悪手局面を選ぶのではなく、適格局面から固定 seed で決定論的に採る。
/// 層のバランスは (進行度 × 手番) を主、被王手・結末を従で取る貪欲法
/// （少数の元対局でも層が偏らない）。`limit` は採る元対局の数（0 で全部）
pub fn stratified_pick(
    games: &[GameCandidates],
    limit: usize,
    seed: u64,
) -> Vec<(String, Candidate)> {
    let mut order: Vec<&GameCandidates> = games.iter().filter(|g| !g.candidates.is_empty()).collect();
    order.sort_by_key(|g| (stable_hash(seed, &g.game_id), g.game_id.clone()));

    let mut primary: HashMap<(String, String), usize> = HashMap::new();
    let mut secondary: HashMap<String, usize> = HashMap::new();
    let mut picked = vec![];
    for g in order {
        if limit > 0 && picked.len() >= limit {
            break;
        }
        let best = g
            .candidates
            .iter()
            .min_by_key(|c| {
                let phase = phase_tag(c.ply).to_string();
                let side = c.tags[0].clone();
                let p = *primary.get(&(phase, side)).unwrap_or(&0);
                let s: usize = c.tags[2..]
                    .iter()
                    .map(|t| *secondary.get(t).unwrap_or(&0))
                    .sum();
                (p, s, stable_hash(seed ^ 0x9E37, &format!("{}:{}", g.game_id, c.ply)))
            })
            .expect("空でない候補");
        *primary
            .entry((phase_tag(best.ply).to_string(), best.tags[0].clone()))
            .or_insert(0) += 1;
        for t in &best.tags[2..] {
            *secondary.entry(t.clone()).or_insert(0) += 1;
        }
        picked.push((g.game_id.clone(), best.clone()));
    }
    picked.sort_by(|a, b| a.0.cmp(&b.0));
    picked
}

/// dev / validation の割り当て（**元対局単位**。同じ棋譜が両方に出ない）
pub fn split_of(game_id: &str, dev_pct: u32, seed: u64) -> &'static str {
    if stable_hash(seed ^ 0x5F17_A9B3_u64, game_id) % 100 < u64::from(dev_pct) {
        "dev"
    } else {
        "validation"
    }
}

/// 終局理由から KIF の終局宣言を作る（`kifu::kif_body` の `ending`）
pub fn kif_ending(reason: &str) -> Option<&'static str> {
    match reason {
        "foul_limit" => Some("反則負け"),
        "resign" => Some("投了"),
        "timeout" => Some("時間切れ"),
        "max_plies" => Some("中断"),
        _ => None,
    }
}

/// 復元した開始状態のログ（両者）の長さ。テストとデバッグ表示用
pub fn log_lens(logs: &[ObservationLog; 2]) -> [usize; 2] {
    [logs[0].events().len(), logs[1].events().len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kifu::kif_body;
    use crate::protocol::{
        FoulRecord, MoveRecord, OpponentInfo, RatingChange, RatingChangePair,
    };
    use crate::truth_replay::side_idx;

    fn sample_end() -> GameEndPayload {
        let moves: Vec<(&str, Color)> = vec![
            ("7g7f", Color::Sente),
            ("3c3d", Color::Gote),
            ("2g2f", Color::Sente),
            ("8c8d", Color::Gote),
            ("2f2e", Color::Sente),
            ("8d8e", Color::Gote),
            ("8h2b+", Color::Sente),
            ("3a2b", Color::Gote),
            ("B*4e", Color::Sente),
        ];
        GameEndPayload {
            result: "sente_win".into(),
            reason: "foul_limit".into(),
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
            // 4手目（後手）と7手目（先手）の手番に反則を1つずつ
            foul_attempts: vec![
                FoulRecord {
                    move_number: 4,
                    by_color: Color::Gote,
                    usi: "P*3c".into(),
                },
                FoulRecord {
                    move_number: 7,
                    by_color: Color::Sente,
                    usi: "P*2c".into(),
                },
            ],
            rating_change: RatingChangePair {
                you: RatingChange { before: 0, after: 0 },
                opponent: RatingChange { before: 0, after: 0 },
            },
            opponent: OpponentInfo {
                username: "estimator_v14".into(),
                rating: 0,
                is_bot: true,
            },
        }
    }

    fn events_json(log: &ObservationLog) -> String {
        serde_json::to_string(log.events()).unwrap()
    }

    /// 抽出は手番境界だけを返す（反則を挟んだ決定点は候補にしない）
    #[test]
    fn candidates_are_turn_boundaries_only() {
        let end = sample_end();
        let cands = candidates(&end, 0).expect("棋譜は健全");
        assert!(!cands.is_empty());
        // 反則があったのは move_number 4（= ply 3 の決定点）と 7（= ply 6）
        assert!(cands.iter().all(|c| c.ply != 3 && c.ply != 6));
        assert!(cands.iter().any(|c| c.ply == 4));
        // 初手（観測ゼロ）は採らない
        assert!(cands.iter().all(|c| c.ply != 0));
    }

    /// 最低残り手数のフィルタ（決着済み局面は情報ゼロなので採らない）
    #[test]
    fn min_remaining_plies_filters_tail() {
        let end = sample_end();
        let cands = candidates(&end, 4).expect("棋譜は健全");
        assert!(cands.iter().all(|c| c.remaining_plies >= 4));
        assert!(cands.iter().all(|c| c.ply <= 5));
    }

    /// **KIF 経由の復元が truth_replay の状態と一致する**。
    /// デッキは KIF で持つ（`bin/scenario ... diag` へそのまま流せる）が、
    /// 抽出は truth_replay の真実から行うので、この2経路がズレると
    /// 「元対局の途中局面から指し継ぐ」という前提そのものが壊れる
    #[test]
    fn kif_roundtrip_matches_truth_replay() {
        let end = sample_end();
        let moves: Vec<String> = end.moves.iter().map(|m| m.usi.clone()).collect();
        let fouls: Vec<(u32, String)> = end
            .foul_attempts
            .iter()
            .map(|f| (f.move_number, f.usi.clone()))
            .collect();
        let body = kif_body(&moves, &fouls, kif_ending(&end.reason)).expect("KIF生成");
        let text = format!(
            "手合割：平手\n先手：先手\n後手：後手\n手数----指手---------消費時間--\n{body}"
        );
        let kifu = parse_kif(&text).expect("KIF解析");

        // truth_replay の各手番境界の状態を集める
        let mut truth: HashMap<usize, (u64, u32, String, String, [u32; 2], usize)> = HashMap::new();
        assert!(for_each_decision_full(&end, |d| {
            if d.fouls_this_turn != 0 {
                return;
            }
            truth.insert(
                d.plies as usize,
                (
                    d.pos.fingerprint(),
                    d.pos.move_number(),
                    events_json(&d.logs[0]),
                    events_json(&d.logs[1]),
                    *d.fouls,
                    side_idx(d.side),
                ),
            );
        }));
        assert!(truth.len() >= 5);

        for (&ply, expected) in &truth {
            let rep = replay(&kifu, ply);
            assert_eq!(rep.pos.fingerprint(), expected.0, "ply={ply} の盤面");
            assert_eq!(rep.pos.move_number(), expected.1, "ply={ply} の手数");
            assert_eq!(events_json(&rep.logs[0]), expected.2, "ply={ply} の先手ログ");
            assert_eq!(events_json(&rep.logs[1]), expected.3, "ply={ply} の後手ログ");
            assert_eq!(rep.fouls, expected.4, "ply={ply} の累積反則数");
            assert_eq!(rep.plies as usize, ply);
            assert_eq!(side_idx(rep.pos.turn()), expected.5, "ply={ply} の手番");
        }
    }

    /// 層化抽出は決定論的で、原則1棋譜1checkpoint
    #[test]
    fn stratified_pick_is_deterministic_and_one_per_game() {
        let end = sample_end();
        let games: Vec<GameCandidates> = (0..6)
            .map(|i| GameCandidates {
                game_id: format!("game{i:03}"),
                candidates: candidates(&end, 0).expect("健全"),
            })
            .collect();
        let a = stratified_pick(&games, 4, 20260823);
        let b = stratified_pick(&games, 4, 20260823);
        assert_eq!(a.len(), 4);
        assert_eq!(
            a.iter().map(|(g, c)| (g.clone(), c.ply)).collect::<Vec<_>>(),
            b.iter().map(|(g, c)| (g.clone(), c.ply)).collect::<Vec<_>>(),
            "同じ seed なら同じ抽出"
        );
        let mut ids: Vec<&String> = a.iter().map(|(g, _)| g).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 4, "1棋譜1checkpoint");
        // 層化が効いている: 先手・後手の両方が入る
        assert!(a.iter().any(|(_, c)| c.side == Color::Sente));
        assert!(a.iter().any(|(_, c)| c.side == Color::Gote));
    }

    /// dev / validation は元対局単位で決まる（同じ棋譜が両方に出ない）
    #[test]
    fn split_is_stable_per_game() {
        for i in 0..50 {
            let id = format!("game{i:03}");
            assert_eq!(split_of(&id, 50, 7), split_of(&id, 50, 7));
        }
        let dev = (0..200)
            .filter(|i| split_of(&format!("g{i}"), 50, 7) == "dev")
            .count();
        assert!((60..140).contains(&dev), "50%前後に割れる: {dev}/200");
    }
}
