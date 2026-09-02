//! **checkpoint arena**（issue #19 の P0）: 過去の実戦の途中局面から bot 同士で
//! 終局まで指し継ぎ、候補設定と対照設定を**同じ局面でブロック化**して比べる。
//!
//! 位置づけは「小さな改善を証明する短縮 arena」ではなく、**通常 arena へ送る前に
//! 明確な悪化を安価に除外する破滅検出器**。最終採否は通常 arena のままで、
//! ここは較正が済むまで informational。
//!
//! 分散低減の主因は「完全な共通乱数」ではなく**局面ブロッキング**である
//! （方策が分岐すれば乱数消費もずれるので共通乱数とは主張しない）。
//!
//! ## サブコマンド
//!
//! ```text
//! # デッキ抽出（元 KIF + manifest）
//! checkpoint_arena extract --records records --out checkpoint-arena \
//!     --opponent estimator_v14 --min-remaining 20 --limit 32
//!
//! # 2つの arm を同一 runner で交互実行（子プロセスなので env の OnceLock が分離される）
//! checkpoint_arena run checkpoint-arena/deck.json --split dev --seeds 3 \
//!     --experiment anchor_move --candidate-env TSUITATE_ANCHOR_MOVE_W=0.6 \
//!     --jsonl out/anchor_move.jsonl
//!
//! # ペア集計（cluster bootstrap / ICC / MDE / 安全性指標）
//! checkpoint_arena compare out/anchor_move.jsonl --known-arena-delta -8.5 \
//!     --markdown out/anchor_move.md --json out/anchor_move.summary.json
//!
//! # 実験横断（符号一致・順位相関）
//! checkpoint_arena report out/*.summary.json
//! ```
//!
//! `unit` / `pair` は `run` が内部で起動する1単位のワーカー（手で叩いてもよい）。

// json! のキー数が多いので再帰上限を上げる（既定 128 では unit の JSONL 行が通らない）
#![recursion_limit = "512"]

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use tsuitate_bot::config;
use tsuitate_bot::frozen;
use tsuitate_bot::checkpoint::{
    Deck, DeckEntry, GameCandidates, candidates, kif_ending, phase_tag, restore, stable_game_id,
    split_of, stable_hash, stratified_pick,
};
use tsuitate_bot::kifu::kif_body;
use tsuitate_bot::protocol::Color;
use tsuitate_bot::scenario_core::make_view;
use tsuitate_bot::selfplay::{GameResult, StartState, play_continuation};
use tsuitate_bot::strategy;
use tsuitate_bot::truth_replay::{load_end, side_idx};

/// JSONL の**行** schema。
///
/// **2 で `opponent_env` を必須にした**（PR #20 4回目レビュー指摘2）。
/// schema 1 は candidate env が固定相手にも効いていた時期の記録で、
/// `opponent_env` が無いため「相手の実効 env は両 arm とも空で一致」と
/// 誤って解釈できてしまう。schema 2 は arm 固有ノブをプロセス env で
/// 渡していた時期（＝相手にも効いていた）。どちらも集計から明示的に弾く
const ROW_SCHEMA: u32 = 3;

/// `compare` が出す **summary JSON** の schema。行 schema とは別に持つ
/// （両者は独立に進化しうる）。`report` もこれを検査して schema 1 を拒否する
/// —— さもないと、汚染済み JSONL から既に作られた summary を再 compare せずに
/// 横断表・符号一致・順位相関へ流し込めてしまう（PR #20 5回目レビュー指摘1）
const SUMMARY_SCHEMA: u32 = 3;

/// summary JSON の schema を検査する（`report` 用。テストできるよう関数に分けてある）
fn check_summary_schema(v: &serde_json::Value, where_: &str) -> Result<(), String> {
    let schema = v["schema"].as_u64().unwrap_or(0) as u32;
    if schema == 1 || schema == 2 {
        return Err(format!(
            "{where_}: schema {schema} の summary は使えません。\n             arm 固有の設定が固定相手にも効いていた時期の JSONL から作られたもので\n             （schema 1: 相手の実効 env が未記録 / schema 2: ノブをプロセス env で渡していた）、\n             撤回済みの delta・既知差が横断表へ再び混入します（PR #20 / issue #21）。\n             取り直した JSONL から compare し直してください"
        ));
    }
    if schema != SUMMARY_SCHEMA {
        return Err(format!(
            "{where_}: summary schema {schema} は未対応（対応 {SUMMARY_SCHEMA}）"
        ));
    }
    Ok(())
}

/// **共有モジュール**（凍結版も呼ぶ）。
///
/// 凍結版は `estimator.rs` / `check.rs` / `strategy.rs` を自己完結コピーにしている
/// （`scripts/freeze_estimator.py` の `SRC`）が、それ以外は共有モジュールを呼ぶ。
/// たとえば v14 は `crate::opening::OpeningBook` を作り、その `load()` が
/// `TSUITATE_JOSEKI` を読む。**凍結ファイル自体にその文字列は無い**ので、
/// 1ファイルだけを走査すると見落とす（PR #20 4回目レビュー指摘1）。
const SHARED_SOURCES: &[&str] = &[
    include_str!("../opening.rs"),
    include_str!("../likelihood.rs"),
    include_str!("../belief_nn.rs"),
    include_str!("../belief_features.rs"),
    include_str!("../king_belief_nn.rs"),
    include_str!("../opp_move_nn.rs"),
    include_str!("../opp_move_nn_v25.rs"),
    include_str!("../opp_move_features.rs"),
    include_str!("../value_nn.rs"),
    include_str!("../value_features.rs"),
    include_str!("../deduce.rs"),
    include_str!("../mate.rs"),
    include_str!("../model.rs"),
    include_str!("../board.rs"),
    include_str!("../shogi.rs"),
    include_str!("../hits.rs"),
    include_str!("../observation.rs"),
];

/// 戦略ごとの固有ソース（**compile-time に埋め込む**）。
///
/// `TSUITATE_*` はプロセス全体に効くので、candidate arm にだけ env を渡したつもりでも
/// **同じプロセスで作る相手（凍結版）にも効く**。凍結版は凍結時点で読んでいた env を
/// 今も読むので、env 実験のノブはほぼ全部 v14 が読む。
///
/// ソースを埋め込んで env 名を走査すれば、この食い違いを**実行前に**検出できる。
/// 凍結版のソースは `frozen::SOURCES`（**一箇所で管理**）から引くので、
/// 版を足したときにここを更新し忘れることはない。現行 estimator の3ファイルだけ
/// ここで埋め込む。バイナリは数 MB 太るが、これは開発用ツールなので許容する。
///
/// **ただしこの走査は「読まないことの証明」にはならない**（共有依存は定跡以外にもあり、
/// 動的な env 名の組み立ても原理的にありうる）。issue #21 以後、arm 固有ノブは
/// プロセス env を通らないのでこの走査に頼らずに済むが、**プロセス env に
/// arm 固有の値を置く経路が復活したときの関門**として残してある。
const STRATEGY_SOURCES: &[(&str, &[&str])] = &[("estimator", &[
    include_str!("../strategy.rs"),
    include_str!("../estimator.rs"),
    include_str!("../check.rs"),
])];

/// **プロセス env に置いてよい arm 固有キー**。
///
/// issue #21 以後、arm 固有のノブは**プロセス env ではなく config**
/// （`--control-env` / `--candidate-env` → `StrategyConfig`）で渡すので、
/// ここは空のままでよい。プロセス env に arm 固有の値を置くと、
/// env を読み続ける既存の凍結版 v6〜v14 が arm ごとに違う設定になる。
const CANDIDATE_ONLY_ENV: &[&str] = &[];

/// 戦略 `name` が env `key` を読むか。**未知の戦略名は「読む」と見なす**（安全側）。
/// 固有ソースに加えて**共有モジュールも走査する**
fn strategy_reads_env(name: &str, key: &str) -> bool {
    let own = match STRATEGY_SOURCES.iter().find(|(n, _)| *n == name) {
        Some((_, srcs)) => srcs.iter().any(|s| s.contains(key)),
        None if frozen::SOURCES.iter().any(|(_, n, _)| *n == name) => {
            // 凍結版は自分のファイルの中で env を読む（`frozen::SOURCES` が一次資料）。
            // 共有モジュール経由（定跡パス）も `env_keys_read_by` が含める
            return frozen::env_keys_read_by(name).iter().any(|k| k == key);
        }
        None => return true,
    };
    own || SHARED_SOURCES.iter().any(|s| s.contains(key))
}

/// 相手が読む env のうち、実効のプロセス env に設定されているものを列挙する。
/// JSONL へ残して「両 arm で相手の実効設定が同じか」を compare 側で検査できるようにする
fn opponent_effective_env(opponent: &str) -> BTreeMap<String, String> {
    std::env::vars()
        .filter(|(k, _)| k.starts_with("TSUITATE_") && strategy_reads_env(opponent, k))
        .collect()
}

/// **相手の実効挙動の指紋**（PR #22 レビュー指摘4）。
///
/// 現行 `StrategyConfig` の指紋をそのまま使うと、凍結版は各ファイル内の
/// 旧既定値・旧 env 読取規則で動くので**相手名に依らず同じ値**になり、
/// 「相手の実効設定」を表さない。凍結版は版のソース・その版が読む env の
/// 実効値・共有モデルの pin から作り、現行 estimator 系だけ config の指紋を使う。
fn opponent_fingerprint(opponent: &str, ambient: &config::StrategyConfig) -> String {
    let env: BTreeMap<String, String> = std::env::vars()
        .filter(|(k, _)| k.starts_with("TSUITATE_"))
        .collect();
    match frozen::behavior_fingerprint(opponent, &env) {
        Some(fp) => fp,
        // 現行 estimator 系は instance の config がそのまま実効設定
        None => ambient.fingerprint(),
    }
}

/// **arm 固有ノブが実際に効く戦略か**（issue #21）。
///
/// config を尊重するのは現行 estimator 系だけ。凍結版へノブを渡しても
/// 黙って無視される（凍結版は自分のコピーの中でプロセス env を読む）ので、
/// 「設定したつもり」で計測してしまう事故を起動時に止める。
fn assert_arm_knobs_apply(arm_label: &str, strategy: &str, knobs: &BTreeMap<String, String>) {
    if knobs.is_empty() {
        return;
    }
    // **綴り間違い・無効値の関門**（PR #22 レビュー指摘2）。「設定したつもり」で
    // 実効値が既定のまま完走するのを防ぐ
    let check = config::check_overrides(&config::EnvSource::from_process(), knobs);
    if !check.unknown.is_empty() {
        die(&format!(
            "arm={arm_label} のノブに戦略が読まないキーがあります（綴り間違い？）: {}",
            check.unknown.join(", ")
        ));
    }
    if !check.ineffective.is_empty() {
        eprintln!(
            "警告: arm={arm_label} の次のノブは実効値を変えませんでした\n                      （既定値と同じ値か、解釈できない・範囲外の値）: {}",
            check.ineffective.join(", ")
        );
    }
    if strategy::honors_config(strategy) {
        return;
    }
    die(&format!(
        "arm={arm_label} の戦略 {strategy} は config を尊重しないので、ノブ {} は無視されます。\n             \
         凍結版へ設定を渡すことはできません（凍結版は凍結時点の env を読む）。\n             \
         プロセス env で渡すと相手側にも効くため、それも安全ではありません（issue #21）。",
        knobs.keys().cloned().collect::<Vec<_>>().join(", ")
    ));
}

/// **arm 固有「env」は原則拒否**（PR #20 4回目レビュー指摘1）。
///
/// 走査による「相手が読む」検出は偽陰性がありうる（共有依存・動的な env 名）ので、
/// 「読まないと検出されたから許可」ではなく「監査済みリストに載っているから許可」にする。
/// 両 arm で同じ値の env（`--budget-ms` 等）は相手にも等しく効くので対象外。
fn assert_opponent_blind_to(opponent: &str, arm_env_keys: &[String], allow: bool) -> Vec<String> {
    let denied: Vec<String> = arm_env_keys
        .iter()
        .filter(|k| {
            // 監査済みでも、走査で相手が読むと出たら拒否（リストの陳腐化対策）
            !CANDIDATE_ONLY_ENV.contains(&k.as_str()) || strategy_reads_env(opponent, k)
        })
        .cloned()
        .collect();
    if !denied.is_empty() && !allow {
        let scanned: Vec<&str> = denied
            .iter()
            .filter(|k| strategy_reads_env(opponent, k))
            .map(String::as_str)
            .collect();
        die(&format!(
            "arm 固有の env は原則拒否です: {}\n\
             （うち相手 {opponent} が読むと検出できたもの: {}）\n\
             env はプロセス全体に効き、相手も同じプロセスで作られるので、arm ごとに\n\
             値が違うと candidate arm と control arm で「同じ固定相手」になりません。\n\
             走査は読まないことの証明にならない（共有モジュール経由の読取など）ため、\n\
             許可は監査済みの CANDIDATE_ONLY_ENV だけです（恒久対策は issue #21）。\n\
             承知のうえで続行するなら --allow-opponent-env",
            denied.join(", "),
            if scanned.is_empty() { "なし".to_string() } else { scanned.join(", ") }
        ));
    }
    denied
}


fn usage() -> &'static str {
    "usage: checkpoint_arena <extract|run|unit|pair|compare|report|arena-var|arena-balance> [options]\n\
     \n\
     extract --records <dir|file...> --out <dir> [--opponent NAME] [--min-remaining N]\n\
     \x20       [--limit N] [--seed N] [--dev-pct N]\n\
     run <deck.json> [--split dev] [--seeds N] [--seed-base N] [--jobs N]\n\
     \x20   [--experiment LABEL] [--opponent NAME]\n\
     \x20   [--control-strategy NAME] [--candidate-strategy NAME]\n\
     \x20   [--control-env K=V,...] [--candidate-env K=V,...]\n\
     \x20   [--control-bin PATH] [--candidate-bin PATH] [--shared-prewarm]\n\
     \x20   [--budget-ms N] [--jsonl OUT] [--limit N] [--shards N --shard I]\n\
     unit <deck.json> --id ID --seed N --arm NAME [--strategy NAME] [--opponent NAME]\n\
     pair <deck.json> --id ID --seed N [--control-strategy NAME] [--candidate-strategy NAME]\n\
     \x20    [--shared-prewarm] [--arm-order 0|1]\n\
     compare <arm.jsonl...> [--known-arena-delta PT] [--markdown OUT] [--json OUT]\n\
     \x20       [--boot N] [--label NAME]\n\
     report <summary.json...> [--markdown OUT]\n\
     arena-var --control <games.jsonl...> --candidate <games.jsonl...>\n\
     \x20         [--baseline NAME] [--label NAME] [--alpha 0.05] [--power 0.80] [--boot N]\n\
     \x20         [--allow-budget-diff]\n\
     \x20         [--markdown OUT] [--json OUT] [--allow-incomplete]\n\
     arena-balance --control <games.jsonl...> --candidate <games.jsonl...>\n\
     \x20             --manifest <path>（計測前に commit した validation manifest。\n\
     \x20               cand_knobs の期待値と行指紋の照合。無い集計は判定不能）\n\
     \x20             [--expect-cand-knobs \"K=V K=V\"（manifest 無しの診断用の期待ノブ）]\n\
     \x20             [--expect-games N] [--expect-shards N] [--expect-seeds \"相手=base,...\"]\n\
     \x20             [--expect-opponents \"...\"] [--alpha F] [--boot N]（既定はすべて\n\
     \x20               issue #40 の事前登録定数。外した指定は判定不能扱い）\n\
     \x20             [--label NAME] [--markdown OUT] [--json OUT] [--allow-incomplete]"
}

fn die(msg: &str) -> ! {
    eprintln!("{msg}");
    eprintln!("{}", usage());
    std::process::exit(1);
}

/// `--flag value` と `--flag=value` の両方を読む素朴なパーサ
struct Args {
    positional: Vec<String>,
    flags: HashMap<String, Vec<String>>,
}

impl Args {
    fn parse(raw: &[String]) -> Self {
        let mut positional = vec![];
        let mut flags: HashMap<String, Vec<String>> = HashMap::new();
        let mut i = 0;
        while i < raw.len() {
            let a = &raw[i];
            if let Some(rest) = a.strip_prefix("--") {
                if let Some((k, v)) = rest.split_once('=') {
                    flags.entry(k.to_string()).or_default().push(v.to_string());
                } else if raw.get(i + 1).is_some_and(|n| !n.starts_with("--")) {
                    flags
                        .entry(rest.to_string())
                        .or_default()
                        .push(raw[i + 1].clone());
                    i += 1;
                } else {
                    flags.entry(rest.to_string()).or_default().push("1".into());
                }
            } else {
                positional.push(a.clone());
            }
            i += 1;
        }
        Args { positional, flags }
    }

    fn get(&self, k: &str) -> Option<&str> {
        self.flags.get(k).and_then(|v| v.last()).map(String::as_str)
    }
    fn all(&self, k: &str) -> Vec<String> {
        self.flags.get(k).cloned().unwrap_or_default()
    }
    fn flag(&self, k: &str) -> bool {
        self.flags.contains_key(k)
    }
    fn num<T: std::str::FromStr>(&self, k: &str, default: T) -> T {
        self.get(k).and_then(|v| v.parse().ok()).unwrap_or(default)
    }
}

/// `--arm-env-keys a,b,c` を読む
fn split_keys(spec: Option<&str>) -> Vec<String> {
    spec.map(|s| {
        s.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    })
    .unwrap_or_default()
}

/// **実行時 cwd の HEAD** を返す（バイナリの revision ではない）。
/// 同一バイナリ内の比較では十分だが、P3 の `--control-bin` / `--candidate-bin`
/// で別 revision のバイナリを突き合わせるときは、これでは区別できない
/// （両方とも同じ cwd の HEAD になる）。P3 に入る前に build-time revision
/// （`env!` で埋め込む等）へ分ける必要がある（PR #20 追加レビュー指摘4）。
/// それまでは compare の commit 混在チェックが安全弁として働く
fn git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

/// `K=V,K=V` / `K=V K=V` の両方を受ける（arena.yml の `-f env=` と同じ気分で書ける）。
/// **`TSUITATE_*` だけ**を許可する（審判側の `ARENA_*` を arm ごとに変えると
/// 裁定が arm 間でズレる）。
///
/// issue #21 以後、ここで解釈した値は**プロセス env ではなく
/// `StrategyConfig`** として arm の戦略にだけ渡る
fn parse_env(spec: Option<&str>) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let Some(spec) = spec else { return out };
    for token in spec.split([',', ' ', '\n']).filter(|s| !s.trim().is_empty()) {
        let Some((k, v)) = token.trim().split_once('=') else {
            die(&format!("env の書式は K=V です: {token}"));
        };
        if !k.starts_with("TSUITATE_") {
            die(&format!("env は TSUITATE_* だけ許可されます: {k}"));
        }
        out.insert(k.to_string(), v.to_string());
    }
    out
}

/// `parse_env` の逆（子プロセスへ `--knobs` で渡す）。
fn fmt_env(env: &BTreeMap<String, String>) -> String {
    env.iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(",")
}

// ---------------------------------------------------------------------------
// extract
// ---------------------------------------------------------------------------

/// ディレクトリは**再帰的に**辿る（CI では artifact ごとのサブディレクトリへ
/// 展開されるので、1階層しか見ないと何も拾えない）
fn walk_jsonl(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let path = e.path();
        if path.is_dir() {
            walk_jsonl(&path, out);
        } else if path.extension().is_some_and(|x| x == "jsonl") {
            out.push(path);
        }
    }
}

/// (安定した元対局 ID, ファイルパス)。ID は入力ルートからの相対パス由来なので、
/// 複数ディレクトリに同名の JSONL があっても衝突しない（追加レビュー指摘5）
fn collect_record_files(specs: &[String]) -> Vec<(String, PathBuf)> {
    let mut files: Vec<(String, PathBuf)> = vec![];
    for spec in specs {
        let p = Path::new(spec);
        if p.is_dir() {
            let mut found = vec![];
            walk_jsonl(p, &mut found);
            for f in found {
                files.push((stable_game_id(p, &f), f));
            }
        } else {
            let root = p.parent().unwrap_or(Path::new("."));
            files.push((stable_game_id(root, p), p.to_path_buf()));
        }
    }
    files.sort();
    files.dedup();
    // それでも同じ ID が出たら（同じファイルを別スペックで2回渡した等）即エラー。
    // 黙って上書きすると別対局の状態が混ざる
    let mut seen: HashMap<&str, &PathBuf> = HashMap::new();
    for (id, path) in &files {
        if let Some(prev) = seen.insert(id.as_str(), path) {
            die(&format!(
                "元対局 ID が衝突しています: {id}\n  {}\n  {}",
                prev.display(),
                path.display()
            ));
        }
    }
    files
}

fn cmd_extract(args: &Args) {
    let records = {
        let mut v = args.all("records");
        v.extend(args.positional.iter().skip(1).cloned());
        if v.is_empty() {
            die("--records が必要です");
        }
        v
    };
    let out_dir = PathBuf::from(args.get("out").unwrap_or("checkpoint-arena"));
    let opponent = args.get("opponent").unwrap_or("estimator_v14").to_string();
    let min_remaining: u32 = args.num("min-remaining", 20);
    let limit: usize = args.num("limit", 32);
    let seed: u64 = args.num("seed", 20260823);
    let dev_pct: u32 = args.num("dev-pct", 50);
    let source = args
        .get("source")
        .map(str::to_string)
        .unwrap_or_else(|| records.join(" "));

    let files = collect_record_files(&records);
    if files.is_empty() {
        die("記録ファイルが見つかりません");
    }

    let mut games: Vec<GameCandidates> = vec![];
    let mut ends: HashMap<String, tsuitate_bot::protocol::GameEndPayload> = HashMap::new();
    let mut skipped = (0u32, 0u32, 0u32); // (記録不備, 棋譜破損, 適格ゼロ)
    for (stem, path) in &files {
        let stem = stem.clone();
        let Some(end) = load_end(&path.to_string_lossy()) else {
            skipped.0 += 1;
            continue;
        };
        // 記録不備・時間切れ局・終局していない局は除外
        if end.moves.is_empty() || end.reason == "timeout" || end.reason.is_empty() {
            skipped.0 += 1;
            continue;
        }
        let Some(cands) = candidates(&end, min_remaining) else {
            skipped.1 += 1;
            continue;
        };
        if cands.is_empty() {
            skipped.2 += 1;
            continue;
        }
        games.push(GameCandidates {
            game_id: stem.clone(),
            candidates: cands,
        });
        ends.insert(stem, end);
    }

    let picked = stratified_pick(&games, limit, seed);
    if picked.is_empty() {
        die("適格な checkpoint がありません（--min-remaining を下げるか記録を増やす）");
    }

    let games_dir = out_dir.join("games");
    std::fs::create_dir_all(&games_dir).unwrap_or_else(|e| die(&format!("{}: {e}", games_dir.display())));

    let mut entries = vec![];
    for (game_id, cand) in &picked {
        let end = &ends[game_id];
        let moves: Vec<String> = end.moves.iter().map(|m| m.usi.clone()).collect();
        let fouls: Vec<(u32, String)> = end
            .foul_attempts
            .iter()
            .map(|f| (f.move_number, f.usi.clone()))
            .collect();
        let body = match kif_body(&moves, &fouls, kif_ending(&end.reason)) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("{game_id}: KIF 生成に失敗（この局は捨てる）: {e}");
                continue;
            }
        };
        let id = format!("{game_id}-ply{}", cand.ply);
        let rel = format!("games/{id}.kif");
        let header = format!(
            "棋戦：checkpoint arena\n手合割：平手\n先手：先手\n後手：後手\n\
             手数----指手---------消費時間--\n\
             *scenario ply={} target={} desc=checkpoint {} ({}, 残り{}手)\n",
            cand.ply,
            end.moves
                .get(cand.ply)
                .map(|m| m.usi.clone())
                .unwrap_or_default(),
            id,
            cand.tags.join("/"),
            cand.remaining_plies,
        );
        std::fs::write(out_dir.join(&rel), header + &body)
            .unwrap_or_else(|e| die(&format!("{rel}: {e}")));
        entries.push(DeckEntry {
            id,
            game: game_id.clone(),
            kif: rel,
            ply: cand.ply,
            split: split_of(game_id, dev_pct, seed).to_string(),
            tags: cand.tags.clone(),
        });
    }

    let deck = Deck {
        version: tsuitate_bot::checkpoint::DECK_VERSION,
        source,
        opponent,
        min_remaining_plies: min_remaining,
        entries,
    };
    // 復元できることを抽出時に確かめる（手番境界の検査つき）
    for e in &deck.entries {
        if let Err(err) = restore(&out_dir, e) {
            die(&format!("復元できない checkpoint があります: {err}"));
        }
    }
    let manifest = out_dir.join("deck.json");
    deck.save(&manifest).unwrap_or_else(|e| die(&e));

    println!("記録 {} 件（不備 {} / 破損 {} / 適格ゼロ {}）", files.len(), skipped.0, skipped.1, skipped.2);
    println!("checkpoint {} 件 → {}", deck.entries.len(), manifest.display());
    println!("deck_hash {}", deck.hash(&out_dir).unwrap_or_else(|e| die(&e)));
    let mut tally: BTreeMap<&str, usize> = BTreeMap::new();
    for e in &deck.entries {
        for t in &e.tags {
            *tally.entry(t.as_str()).or_insert(0) += 1;
        }
        *tally.entry(if e.split == "dev" { "split:dev" } else { "split:validation" }).or_insert(0) += 1;
    }
    println!("層の内訳:");
    for (t, n) in tally {
        println!("  {t}: {n}");
    }
}

// ---------------------------------------------------------------------------
// 1単位の実行（unit / pair）
// ---------------------------------------------------------------------------

/// checkpoint 側（開始手番側）と相手側の seed。**両 arm で同一**にすることで
/// 「同じ局面・同じ相手」でブロック化する
fn unit_seeds(id: &str, seed: u64) -> (u64, u64) {
    let base = stable_hash(seed, id);
    (
        tsuitate_bot::selfplay::mix(base ^ 0x0C0F_FEE0),
        tsuitate_bot::selfplay::mix(base ^ 0x0BEE_F000),
    )
}

struct ArmSpec {
    label: String,
    strategy: String,
}

struct UnitResult {
    line: serde_json::Value,
}

/// prewarm 共有（opt-in）で作った戦略と、その実測コスト。
/// 共有時は 1回ぶんの prewarm を両 arm で折半して計上する
/// （「共有ありのコスト」を JSONL から直接読めるようにするため）
struct Prewarmed {
    me: Box<dyn strategy::Strategy>,
    opp: Box<dyn strategy::Strategy>,
    ms: [f64; 2],
}

fn think_summary(us: &[u64]) -> (f64, f64, f64, f64) {
    if us.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mut s = us.to_vec();
    s.sort_unstable();
    let mean = s.iter().sum::<u64>() as f64 / s.len() as f64 / 1000.0;
    let at = |q: usize| s[(s.len() * q / 100).min(s.len() - 1)] as f64 / 1000.0;
    (mean, at(95), at(99), *s.last().unwrap() as f64 / 1000.0)
}

/// 戦略を**設定を明示して**作る（issue #21）。config を尊重しない凍結版は
/// 従来どおり `make_seeded`（凍結時点の env を読む）で作る。
fn build_strategy(
    name: &str,
    seed: u64,
    cfg: &Arc<config::StrategyConfig>,
) -> Box<dyn strategy::Strategy> {
    let built = if strategy::honors_config(name) {
        strategy::make_seeded_with_config(name, seed, cfg.clone())
    } else {
        strategy::make_seeded(name, seed)
    };
    built.unwrap_or_else(|| die(&format!("未知の戦略名: {name}")))
}

/// 1 arm ぶんの継続対局。`prewarmed` があれば prewarm を省いて clone を使う
/// （opt-in の prewarm 共有。共有条件は run 側で判定する）
#[allow(clippy::too_many_arguments)]
fn play_one_arm(
    deck_dir: &Path,
    deck: &Deck,
    entry: &DeckEntry,
    seed: u64,
    arm: &ArmSpec,
    opponent: &str,
    arm_order: usize,
    prewarmed: Option<&Prewarmed>,
    experiment: &str,
    prewarm_reason: &str,
    arm_knobs: &BTreeMap<String, String>,
    arm_env_keys: &[String],
    allow_opponent_env: bool,
) -> UnitResult {
    // プロセス env に arm 固有の値が残っていないか（残っていたら固定相手が arm ごとに変わる）
    let leaked = assert_opponent_blind_to(opponent, arm_env_keys, allow_opponent_env);
    assert_arm_knobs_apply(&arm.label, &arm.strategy, arm_knobs);
    let opp_env = opponent_effective_env(opponent);
    // **設定の境界（issue #21）**: 相手（固定）はプロセス env だけ、
    // arm 側はそこへ arm 固有ノブを重ねたものを見る。プロセス env は
    // 両 arm で同一なので、env を読み続ける凍結相手は arm によらず同じ
    let base_source = config::EnvSource::from_process();
    let opp_config = Arc::new(config::StrategyConfig::from_source(base_source.clone()));
    let arm_config = Arc::new(config::StrategyConfig::from_source(
        base_source.with_overrides(arm_knobs.clone()),
    ));
    let start = restore(deck_dir, entry).unwrap_or_else(|e| die(&e));
    let me = start.pos.turn();
    let me_i = side_idx(me);
    let (seed_me, seed_opp) = unit_seeds(&entry.id, seed);

    let mut strats: [Option<Box<dyn strategy::Strategy>>; 2] = [None, None];
    let mut prewarm_ms = [0.0f64; 2];
    let shared = prewarmed.is_some();
    match prewarmed {
        Some(p) => {
            strats[me_i] = Some(
                p.me.clone_boxed()
                    .unwrap_or_else(|| die("prewarm 共有は clone_boxed 対応の戦略だけ")),
            );
            strats[1 - me_i] = Some(
                p.opp
                    .clone_boxed()
                    .unwrap_or_else(|| die("prewarm 共有は clone_boxed 対応の戦略だけ")),
            );
            // 1回の prewarm を2 arm で割る
            prewarm_ms[me_i] = p.ms[0] / 2.0;
            prewarm_ms[1 - me_i] = p.ms[1] / 2.0;
        }
        None => {
            for i in 0..2 {
                let color = if i == 0 { Color::Sente } else { Color::Gote };
                let (name, s) = if i == me_i {
                    (arm.strategy.as_str(), seed_me)
                } else {
                    (opponent, seed_opp)
                };
                let mut strat = build_strategy(
                    name,
                    s,
                    if i == me_i { &arm_config } else { &opp_config },
                );
                let view = make_view(&start.pos, color, &start.fouls);
                let t = Instant::now();
                strategy::prewarm_strategy(&mut *strat, &view, &start.logs[i]);
                prewarm_ms[i] = t.elapsed().as_secs_f64() * 1000.0;
                strats[i] = Some(strat);
            }
        }
    }
    let strategies = [strats[0].take().unwrap(), strats[1].take().unwrap()];

    let start_state = StartState {
        pos: start.pos.clone(),
        logs: start.logs,
        fouls: start.fouls,
        plies: start.plies,
    };
    let start_fouls = start.fouls;
    let t = Instant::now();
    let out = play_continuation(strategies, start_state, 0);
    let continuation_ms = t.elapsed().as_secs_f64() * 1000.0;

    let score = match out.result {
        GameResult::Win(w) if w == me => 1.0,
        GameResult::Win(_) => 0.0,
        GameResult::Draw => 0.5,
    };
    let first_move = out
        .truth
        .moves
        .iter()
        .find(|m| m.by_color == me)
        .map(|m| m.usi.clone())
        .unwrap_or_else(|| "-".into());
    let (t_mean, t_p95, t_p99, t_max) = think_summary(&out.think_us[me_i]);
    let (o_mean, _, _, _) = think_summary(&out.think_us[1 - me_i]);

    let env: BTreeMap<String, String> = std::env::vars()
        .filter(|(k, _)| k.starts_with("TSUITATE_"))
        .collect();
    let line = serde_json::json!({
        "schema": ROW_SCHEMA,
        "experiment": experiment,
        "arm": arm.label,
        "arm_order": arm_order,
        "checkpoint": entry.id,
        // 元対局（cluster bootstrap の統計単位）。1棋譜1checkpoint なら 1:1
        "source_game": if entry.game.is_empty() { entry.id.as_str() } else { entry.game.as_str() },
        "seed": seed,
        "split": entry.split,
        "tags": entry.tags,
        "side": if me == Color::Sente { "sente" } else { "gote" },
        "start_ply": start.plies,
        "start_fouls_me": start_fouls[me_i],
        "start_fouls_opp": start_fouls[1 - me_i],
        "opponent": opponent,
        "strategy": arm.strategy,
        "env": env,
        // arm 固有ノブ（config 経由。プロセス env は触っていない）と実効設定の指紋。
        // **指紋は解決後の値から作る**ので、「未知のキーを足しただけ」では変わらない
        "arm_knobs": arm_knobs,
        "arm_config": arm_config.fingerprint(),
        // 相手の**実効挙動**の指紋（凍結版は版のソース・読む env・共有モデル pin から作る）。
        // 両 arm で一致していないと「同じ固定相手」ではない
        "opponent_config": opponent_fingerprint(opponent, &opp_config),
        "commit": git_commit(),
        "deck_hash": deck.hash(deck_dir).unwrap_or_else(|e| die(&e)),
        "think_budget_ms": arm_config.think_budget_ms.to_string(),
        "prewarm_shared": shared,
        "prewarm_reason": prewarm_reason,
        // 相手が実際に読む env（両 arm で一致していないと「同じ固定相手」ではない）
        "opponent_env": opp_env,
        "arm_env_keys": arm_env_keys,
        "opponent_reads_arm_env": leaked,
        "score": score,
        "reason": out.reason,
        "plies": out.plies,
        "added_plies": out.added_plies,
        "fouls_me": out.added_fouls[me_i],
        "fouls_opp": out.added_fouls[1 - me_i],
        "fouls_in_check_me": out.added_fouls_in_check[me_i],
        "foul_limit_loss": out.reason == "foul_limit" && score == 0.0,
        "hit_max_plies": out.reason == "max_plies",
        "first_move": first_move,
        "moves_me": out.think_us[me_i].len(),
        "prewarm_ms_me": prewarm_ms[me_i],
        "prewarm_ms_opp": prewarm_ms[1 - me_i],
        "continuation_ms": continuation_ms,
        "total_ms": prewarm_ms[0] + prewarm_ms[1] + continuation_ms,
        "think_avg_ms_me": t_mean,
        "think_p95_ms_me": t_p95,
        "think_p99_ms_me": t_p99,
        "think_max_ms_me": t_max,
        "think_avg_ms_opp": o_mean,
        "threads": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
        "arena_threads": std::env::var("ARENA_THREADS").ok(),
    });
    UnitResult { line }
}

fn load_deck_arg(args: &Args) -> (PathBuf, Deck) {
    let path = PathBuf::from(
        args.positional
            .get(1)
            .unwrap_or_else(|| die("deck.json を指定してください")),
    );
    let deck = Deck::load(&path).unwrap_or_else(|e| die(&e));
    let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    (dir, deck)
}

fn cmd_unit(args: &Args) {
    let (dir, deck) = load_deck_arg(args);
    let id = args.get("id").unwrap_or_else(|| die("--id が必要です"));
    let entry = deck
        .entry(id)
        .unwrap_or_else(|| die(&format!("デッキにない checkpoint: {id}")))
        .clone();
    let seed: u64 = args.num("seed", 0);
    let arm = ArmSpec {
        label: args.get("arm").unwrap_or("candidate").to_string(),
        strategy: args.get("strategy").unwrap_or("estimator").to_string(),
    };
    let opponent = args.get("opponent").unwrap_or(&deck.opponent).to_string();
    let order: usize = args.num("arm-order", 0);
    let experiment = args.get("experiment").unwrap_or("adhoc").to_string();
    let reason = args.get("prewarm-reason").unwrap_or("").to_string();
    let arm_env_keys = split_keys(args.get("arm-env-keys"));
    let knobs = parse_env(args.get("knobs"));
    let r = play_one_arm(
        &dir, &deck, &entry, seed, &arm, &opponent, order, None, &experiment, &reason,
        &knobs, &arm_env_keys, args.flag("allow-opponent-env"),
    );
    println!("{}", r.line);
}

/// 両 arm を**同一プロセス**で走らせる（同一バイナリ・同一 env のときだけ使える）。
/// `--shared-prewarm` で prewarm 済み戦略を `clone_boxed` して両 arm へ配る。
///
/// 共有できるのは「推定器の実装と構築設定・観測ログと seed・prewarm 中に読む
/// 全設定」が両 arm で同一の場合に限る。評価だけが違う（= 戦略名が同じで
/// env も同じ）ケースは実質存在しないので、共有が使えるのは
/// **戦略名も同じ = A/A** のときと、`--shared-prewarm` を明示したときだけにする
fn cmd_pair(args: &Args) {
    let (dir, deck) = load_deck_arg(args);
    let id = args.get("id").unwrap_or_else(|| die("--id が必要です"));
    let entry = deck
        .entry(id)
        .unwrap_or_else(|| die(&format!("デッキにない checkpoint: {id}")))
        .clone();
    let seed: u64 = args.num("seed", 0);
    let opponent = args.get("opponent").unwrap_or(&deck.opponent).to_string();
    let experiment = args.get("experiment").unwrap_or("adhoc").to_string();
    let control = ArmSpec {
        label: "control".into(),
        strategy: args.get("control-strategy").unwrap_or("estimator").to_string(),
    };
    let candidate = ArmSpec {
        label: "candidate".into(),
        strategy: args.get("candidate-strategy").unwrap_or("estimator").to_string(),
    };
    let order: usize = args.num("arm-order", 0);
    let control_knobs = parse_env(args.get("control-knobs"));
    let candidate_knobs = parse_env(args.get("candidate-knobs"));
    assert_arm_knobs_apply("control", &control.strategy, &control_knobs);
    assert_arm_knobs_apply("candidate", &candidate.strategy, &candidate_knobs);

    // prewarm 共有（opt-in）: 両 arm の推定器と prewarm 設定が同じときだけ
    let shared = if args.flag("shared-prewarm") {
        if control.strategy != candidate.strategy {
            die("--shared-prewarm は両 arm の戦略が同じときだけ使えます（推定器が違うと共有できない）");
        }
        if control_knobs != candidate_knobs {
            die("--shared-prewarm は両 arm のノブが同じときだけ使えます（prewarm 中に読む設定が違う）");
        }
        let start = restore(&dir, &entry).unwrap_or_else(|e| die(&e));
        let me_i = side_idx(start.pos.turn());
        let (seed_me, seed_opp) = unit_seeds(&entry.id, seed);
        let mut built: Vec<Box<dyn strategy::Strategy>> = vec![];
        let mut ms = [0.0f64; 2];
        for i in 0..2 {
            let color = if i == 0 { Color::Sente } else { Color::Gote };
            let (name, s) = if i == me_i {
                (control.strategy.as_str(), seed_me)
            } else {
                (opponent.as_str(), seed_opp)
            };
            let base = config::EnvSource::from_process();
            let cfg = Arc::new(config::StrategyConfig::from_source(if i == me_i {
                base.with_overrides(control_knobs.clone())
            } else {
                base
            }));
            let mut strat = build_strategy(name, s, &cfg);
            let view = make_view(&start.pos, color, &start.fouls);
            let t = Instant::now();
            strategy::prewarm_strategy(&mut *strat, &view, &start.logs[i]);
            ms[if i == me_i { 0 } else { 1 }] = t.elapsed().as_secs_f64() * 1000.0;
            built.push(strat);
        }
        let second = built.pop().unwrap();
        let first = built.pop().unwrap();
        let (me, opp) = if me_i == 0 { (first, second) } else { (second, first) };
        Some(Prewarmed { me, opp, ms })
    } else {
        None
    };

    let arms: [&ArmSpec; 2] = if order == 0 {
        [&control, &candidate]
    } else {
        [&candidate, &control]
    };
    let reason = args.get("prewarm-reason").unwrap_or("").to_string();
    let arm_env_keys = split_keys(args.get("arm-env-keys"));
    let allow_opp_env = args.flag("allow-opponent-env");
    for (k, arm) in arms.iter().enumerate() {
        let knobs = if arm.label == "control" { &control_knobs } else { &candidate_knobs };
        let r = play_one_arm(
            &dir, &deck, &entry, seed, arm, &opponent, k, shared.as_ref(), &experiment, &reason,
            knobs, &arm_env_keys, allow_opp_env,
        );
        println!("{}", r.line);
    }
}

// ---------------------------------------------------------------------------
// run（同一 runner で arm を交互実行するスケジューラ）
// ---------------------------------------------------------------------------

struct RunArm {
    label: String,
    strategy: String,
    /// **arm 固有の設定ノブ**（issue #21）。プロセス env ではなく
    /// `--knobs` で子へ渡し、子は arm 側の `StrategyConfig` にだけ載せる。
    /// 凍結相手はプロセス env（両 arm 共通）しか見ないので影響を受けない
    knobs: BTreeMap<String, String>,
    bin: PathBuf,
}

fn run_child(
    bin: &Path,
    sub: &str,
    deck_path: &Path,
    extra: &[String],
    env: &BTreeMap<String, String>,
) -> Vec<String> {
    let mut cmd = std::process::Command::new(bin);
    cmd.arg(sub).arg(deck_path).args(extra);
    // **親から継承した TSUITATE_* を一度すべて落としてから、意図した env だけ設定する**
    // （PR #20 追加レビュー指摘1）。継承したままだと全 shard で同じ値になるので
    // arm 内一意性検査を通ってしまい、「親からの混入も捕まる」が成立しない。
    // ARENA_*（審判側の設定）は両 arm 同一なので継承したままでよい
    for (k, _) in std::env::vars().filter(|(k, _)| k.starts_with("TSUITATE_")) {
        cmd.env_remove(k);
    }
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd
        .output()
        .unwrap_or_else(|e| die(&format!("{} を起動できません: {e}", bin.display())));
    if !out.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&out.stderr));
        die(&format!("子プロセスが失敗しました（{sub}）"));
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| l.starts_with('{'))
        .map(str::to_string)
        .collect()
}

fn cmd_run(args: &Args) {
    let deck_path = PathBuf::from(
        args.positional
            .get(1)
            .unwrap_or_else(|| die("deck.json を指定してください")),
    );
    let deck = Deck::load(&deck_path).unwrap_or_else(|e| die(&e));
    let split = args.get("split").unwrap_or("dev").to_string();
    let seeds: u64 = args.num("seeds", 2);
    let seed_base: u64 = args.num("seed-base", 0);
    // arm 順は (checkpoint 番号 + seed) % 2 なので、2k と 2k+1 が対になって
    // 初めて AB/BA が cluster の内側で閉じる。奇数個だと実行順効果が
    // cluster 平均に残り、compare 側も replicate を作れない（追加レビュー指摘3）
    if seeds % 2 != 0 && !args.flag("allow-odd-seeds") {
        die(&format!(
            "--seeds は 2 の倍数にしてください（指定 {seeds}）。\n             承知のうえで奇数にするなら --allow-odd-seeds（compare は外挿表を出しません）"
        ));
    }
    if seed_base % 2 != 0 && !args.flag("allow-odd-seeds") {
        die(&format!("--seed-base は偶数にしてください（指定 {seed_base}）"));
    }
    let jobs: usize = args.num("jobs", 1);
    let limit: usize = args.num("limit", 0);
    let experiment = args.get("experiment").unwrap_or("adhoc").to_string();
    let opponent = args.get("opponent").unwrap_or(&deck.opponent).to_string();
    let budget = args.get("budget-ms").map(str::to_string);
    let shared_prewarm = args.flag("shared-prewarm");

    let self_bin = std::env::current_exe().unwrap_or_else(|e| die(&format!("自分の実行ファイルが分かりません: {e}")));
    // **arm 固有ノブは config で渡す**（issue #21）。子プロセスの env は
    // 両 arm で完全に同じにするので、env を読み続ける凍結相手は arm によらず同じ
    let control_knobs = parse_env(args.get("control-env"));
    let candidate_knobs = parse_env(args.get("candidate-env"));
    // 両 arm・両側に等しく効く共通設定だけを子の env に置く
    // （凍結相手も `TSUITATE_THINK_BUDGET_MS` を読むので、予算はここで渡す）
    let mut shared_env: BTreeMap<String, String> = BTreeMap::new();
    if let Some(b) = &budget {
        shared_env.insert("TSUITATE_THINK_BUDGET_MS".into(), b.clone());
    }
    // プロセス env に arm 固有の値は置かないので、この一覧は常に空。
    // 「置いてしまったら止める」検査だけ残す（回帰の関門）
    let arm_env_keys: Vec<String> = vec![];
    let allow_opponent_env = args.flag("allow-opponent-env");
    assert_opponent_blind_to(&opponent, &arm_env_keys, allow_opponent_env);

    let control = RunArm {
        label: "control".into(),
        strategy: args.get("control-strategy").unwrap_or("estimator").to_string(),
        knobs: control_knobs,
        bin: args.get("control-bin").map(PathBuf::from).unwrap_or_else(|| self_bin.clone()),
    };
    let candidate = RunArm {
        label: "candidate".into(),
        strategy: args.get("candidate-strategy").unwrap_or("estimator").to_string(),
        knobs: candidate_knobs,
        bin: args.get("candidate-bin").map(PathBuf::from).unwrap_or_else(|| self_bin.clone()),
    };
    assert_arm_knobs_apply("control", &control.strategy, &control.knobs);
    assert_arm_knobs_apply("candidate", &candidate.strategy, &candidate.knobs);

    // prewarm 共有の可否（issue の厳しい共有条件）。理由を JSONL 側へ残せるよう
    // ここで判定して表示する
    let same_bin = control.bin == candidate.bin;
    let same_knobs = control.knobs == candidate.knobs;
    let same_strategy = control.strategy == candidate.strategy;
    let share_reason = if !shared_prewarm {
        "opt-in されていない"
    } else if !same_bin {
        "base/target 別バイナリ"
    } else if !same_knobs {
        "arm ごとに異なる設定（prewarm 中に読む値が違う）"
    } else if !same_strategy {
        "推定器（戦略）が異なる"
    } else {
        ""
    };
    let share = shared_prewarm && share_reason.is_empty();
    if shared_prewarm && !share {
        eprintln!("prewarm 共有は無効です: {share_reason}");
    }

    let mut entries: Vec<DeckEntry> = deck
        .entries
        .iter()
        .filter(|e| split == "all" || e.split == split)
        .cloned()
        .collect();
    if limit > 0 {
        entries.truncate(limit);
    }
    // CI のシャーディング（checkpoint を n 等分して別ランナーへ）。
    // seed でなく checkpoint で割るので、1ランナー内でも AB/BA の均衡は保たれる
    let shards: usize = args.num("shards", 1);
    let shard: usize = args.num("shard", 0);
    if shards > 1 {
        entries = entries
            .into_iter()
            .enumerate()
            .filter(|(i, _)| i % shards == shard)
            .map(|(_, e)| e)
            .collect();
    }
    if entries.is_empty() {
        die(&format!("split={split} に checkpoint がありません"));
    }

    // 作業単位 (checkpoint, seed)。arm 順は checkpoint ごとに AB/BA を均衡させる
    let units: Vec<(usize, u64, usize)> = entries
        .iter()
        .enumerate()
        .flat_map(|(i, _)| {
            (0..seeds).map(move |s| {
                let seed = seed_base + s;
                (i, seed, (i as u64 + s) as usize % 2)
            })
        })
        .collect();
    let total = units.len();
    eprintln!(
        "checkpoint {} 件 × seed {} = {} 単位 / arm 2本 / jobs {} / 相手 {opponent}{}",
        entries.len(),
        seeds,
        total,
        jobs,
        if share { " / prewarm 共有" } else { "" }
    );

    let started = Instant::now();
    let done = std::sync::atomic::AtomicUsize::new(0);
    let lines: Vec<String> = std::thread::scope(|scope| {
        let entries = &entries;
        let units = &units;
        let handles: Vec<_> = (0..jobs.max(1))
            .map(|t| {
                let (control, candidate, deck_path, experiment, opponent, done) =
                    (&control, &candidate, &deck_path, &experiment, &opponent, &done);
                let shared_env = &shared_env;
                let arm_env_keys = &arm_env_keys;
                scope.spawn(move || {
                    let mut out: Vec<String> = vec![];
                    let mut k = t;
                    while k < units.len() {
                        let (ei, seed, order) = units[k];
                        let entry = &entries[ei];
                        let mut base = vec![
                            "--id".to_string(), entry.id.clone(),
                            "--seed".to_string(), seed.to_string(),
                            "--opponent".to_string(), opponent.clone(),
                            "--experiment".to_string(), experiment.clone(),
                            "--prewarm-reason".to_string(),
                            if share_reason.is_empty() { "共有条件を満たす".to_string() } else { share_reason.to_string() },
                            "--arm-env-keys".to_string(),
                            arm_env_keys.join(","),
                        ];
                        if allow_opponent_env {
                            base.push("--allow-opponent-env".to_string());
                            base.push("1".to_string());
                        }
                        if share {
                            let mut extra = base.clone();
                            extra.extend([
                                "--control-strategy".into(), control.strategy.clone(),
                                "--candidate-strategy".into(), candidate.strategy.clone(),
                                "--control-knobs".into(), fmt_env(&control.knobs),
                                "--candidate-knobs".into(), fmt_env(&candidate.knobs),
                                "--shared-prewarm".into(), "1".into(),
                                "--arm-order".into(), order.to_string(),
                            ]);
                            out.extend(run_child(&control.bin, "pair", deck_path, &extra, shared_env));
                        } else {
                            // 同一 runner・同じスロットで背中合わせに走らせる
                            // （キャッシュした control と比べない = 機械負荷を揃える）
                            let arms: [&RunArm; 2] = if order == 0 {
                                [control, candidate]
                            } else {
                                [candidate, control]
                            };
                            for (j, arm) in arms.iter().enumerate() {
                                let mut extra = base.clone();
                                extra.extend([
                                    "--arm".into(), arm.label.clone(),
                                    "--strategy".into(), arm.strategy.clone(),
                                    "--knobs".into(), fmt_env(&arm.knobs),
                                    "--arm-order".into(), j.to_string(),
                                ]);
                                out.extend(run_child(&arm.bin, "unit", deck_path, &extra, shared_env));
                            }
                        }
                        let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                        eprintln!("[{n}/{}] {} seed{seed}", units.len(), entry.id);
                        k += jobs.max(1);
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().expect("worker panic")).collect()
    });

    eprintln!(
        "壁時計 {:.1}分 / 行 {}",
        started.elapsed().as_secs_f64() / 60.0,
        lines.len()
    );
    match args.get("jsonl") {
        Some(path) => {
            if let Some(dir) = Path::new(path).parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            std::fs::write(path, lines.join("\n") + "\n")
                .unwrap_or_else(|e| die(&format!("{path}: {e}")));
            eprintln!("→ {path}");
        }
        None => {
            for l in &lines {
                println!("{l}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// compare（ペア集計）
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Row {
    arm: String,
    arm_order: usize,
    experiment: String,
    checkpoint: String,
    source_game: String,
    seed: u64,
    split: String,
    tags: Vec<String>,
    start_ply: u64,
    first_move: String,
    reason: String,
    commit: String,
    deck_hash: String,
    think_budget: String,
    opponent: String,
    strategy: String,
    /// 実効 env（`TSUITATE_*` のみ）。arm 内で一意であることを検査する
    env: BTreeMap<String, String>,
    /// **相手が実際に読む env**。両 arm で一致していないと「同じ固定相手」ではない
    opponent_env: BTreeMap<String, String>,
    /// arm 固有の設定ノブ（config 経由。issue #21）。arm 内で一意であることを検査する
    arm_knobs: BTreeMap<String, String>,
    /// arm の実効設定の指紋。arm 内で一意であること
    arm_config: String,
    /// **相手の実効設定の指紋**。両 arm で一致していないと「同じ固定相手」ではない
    opponent_config: String,
    prewarm_shared: bool,
    metrics: BTreeMap<&'static str, f64>,
    prewarm_ms: f64,
    continuation_ms: f64,
    total_ms: f64,
}

/// ペア比較する指標。score が主指標、残りは**安全性の共同指標**
/// （過去の arena 悪化で明確な署名を出した高頻度指標。反則減だけを
/// 「強くなった」とは判定しないが、重大な悪化を止める用途に使う）
const METRICS: &[(&str, &str)] = &[
    ("score", "スコア（勝1/分0.5/負0）"),
    ("fouls_me", "反則/局（候補側）"),
    ("fouls_in_check_me", "被王手中反則/局"),
    ("foul_limit_loss", "反則負け率"),
    ("added_plies", "継続手数"),
    ("hit_max_plies", "手数上限率"),
    ("fouls_opp", "反則/局（相手側）"),
    ("think_avg_ms_me", "思考平均ms"),
    ("think_p95_ms_me", "思考p95ms"),
    ("think_p99_ms_me", "思考p99ms"),
];

/// JSONL を読む。**必須フィールドの欠損は 0 で埋めずにエラーにする**
/// （PR #20 レビュー指摘4: 欠けた行が「もっともらしい」集計を作るのを防ぐ）
fn parse_rows(paths: &[String]) -> Vec<Row> {
    let mut rows = vec![];
    for p in paths {
        let text = std::fs::read_to_string(p).unwrap_or_else(|e| die(&format!("{p}: {e}")));
        for (ln, line) in text.lines().enumerate() {
            if !line.trim().starts_with('{') {
                continue;
            }
            let where_ = format!("{p}:{}", ln + 1);
            let v: serde_json::Value = serde_json::from_str(line)
                .unwrap_or_else(|e| die(&format!("{where_}: {e}")));
            let schema = v["schema"].as_u64().unwrap_or(0) as u32;
            if schema == 1 {
                die(&format!(
                    "{where_}: schema 1 の記録は集計に使えません。\n                     candidate arm の env が固定相手にも効いていた時期のもので、\n                     相手の実効 env が記録されていないため「両 arm とも空で一致」と\n                     誤って解釈されます（PR #20）。取り直してください"
                ));
            }
            if schema == 2 {
                die(&format!(
                    "{where_}: schema 2 の記録は集計に使えません。\n                     arm 固有ノブをプロセス env で渡していた時期のもので、\n                     同じプロセスの凍結相手にも効いていました（issue #21）。取り直してください"
                ));
            }
            if schema != ROW_SCHEMA {
                die(&format!("{where_}: schema {schema} は未対応（対応 {ROW_SCHEMA}）"));
            }
            let req_f = |k: &str| -> f64 {
                v[k].as_f64()
                    .unwrap_or_else(|| die(&format!("{where_}: 必須の数値 {k} がありません")))
            };
            let req_b = |k: &str| -> f64 {
                if v[k]
                    .as_bool()
                    .unwrap_or_else(|| die(&format!("{where_}: 必須の真偽値 {k} がありません")))
                {
                    1.0
                } else {
                    0.0
                }
            };
            let req_s = |k: &str| -> String {
                v[k].as_str()
                    .unwrap_or_else(|| die(&format!("{where_}: 必須の文字列 {k} がありません")))
                    .to_string()
            };
            let req_u = |k: &str| -> u64 {
                v[k].as_u64()
                    .unwrap_or_else(|| die(&format!("{where_}: 必須の整数 {k} がありません")))
            };
            let mut metrics: BTreeMap<&'static str, f64> = BTreeMap::new();
            metrics.insert("score", req_f("score"));
            metrics.insert("fouls_me", req_f("fouls_me"));
            metrics.insert("fouls_in_check_me", req_f("fouls_in_check_me"));
            metrics.insert("foul_limit_loss", req_b("foul_limit_loss"));
            metrics.insert("added_plies", req_f("added_plies"));
            metrics.insert("hit_max_plies", req_b("hit_max_plies"));
            metrics.insert("fouls_opp", req_f("fouls_opp"));
            metrics.insert("think_avg_ms_me", req_f("think_avg_ms_me"));
            metrics.insert("think_p95_ms_me", req_f("think_p95_ms_me"));
            metrics.insert("think_p99_ms_me", req_f("think_p99_ms_me"));
            let arm = req_s("arm");
            if arm != "control" && arm != "candidate" {
                die(&format!("{where_}: arm は control / candidate のみ（{arm}）"));
            }
            rows.push(Row {
                arm,
                arm_order: req_u("arm_order") as usize,
                experiment: req_s("experiment"),
                checkpoint: req_s("checkpoint"),
                source_game: req_s("source_game"),
                seed: req_u("seed"),
                split: req_s("split"),
                tags: v["tags"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|t| t.as_str().map(str::to_string)).collect())
                    .unwrap_or_default(),
                start_ply: req_u("start_ply"),
                first_move: req_s("first_move"),
                reason: req_s("reason"),
                commit: req_s("commit"),
                deck_hash: req_s("deck_hash"),
                think_budget: req_s("think_budget_ms"),
                opponent: req_s("opponent"),
                strategy: req_s("strategy"),
                arm_knobs: v["arm_knobs"]
                    .as_object()
                    .map(|o| {
                        o.iter()
                            .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                            .collect()
                    })
                    .unwrap_or_default(),
                arm_config: req_s("arm_config"),
                opponent_config: req_s("opponent_config"),
                opponent_env: v["opponent_env"]
                    .as_object()
                    .unwrap_or_else(|| {
                        die(&format!("{where_}: 必須の object opponent_env がありません"))
                    })
                    .iter()
                    .map(|(k, val)| (k.clone(), val.as_str().unwrap_or_default().to_string()))
                    .collect(),
                env: v["env"]
                    .as_object()
                    .map(|m| {
                        m.iter()
                            .map(|(k, val)| {
                                (k.clone(), val.as_str().unwrap_or_default().to_string())
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
                prewarm_shared: v["prewarm_shared"].as_bool().unwrap_or(false),
                metrics,
                prewarm_ms: req_f("prewarm_ms_me") + req_f("prewarm_ms_opp"),
                continuation_ms: req_f("continuation_ms"),
                total_ms: req_f("total_ms"),
            });
        }
    }
    rows
}

/// 比較して良い行の集合かを検査する（PR #20 レビュー指摘4）。
///
/// **別実験・別デッキ・別予算・別相手の JSONL を混ぜても、これまでは
/// もっともらしい summary が出てしまっていた**。実験条件の一意性・重複・
/// 期待件数を検査し、不一致や部分集計は失敗させる（`--allow-incomplete`
/// を明示したときだけ警告に落として続行する）。
fn validate_rows(
    rows: &[Row],
    allow_incomplete: bool,
    expected_checkpoints: Option<&HashSet<String>>,
) -> Vec<String> {
    let mut notes: Vec<String> = vec![];
    let fail = |msg: String, notes: &mut Vec<String>| {
        if allow_incomplete {
            notes.push(format!("**警告**: {msg}"));
        } else {
            die(&format!("{msg}（意図的なら --allow-incomplete）"));
        }
    };

    // 実験条件は1組に限る（arm ごとに変わってよいのは strategy と env だけ）
    for (name, vals) in [
        ("experiment", rows.iter().map(|r| r.experiment.clone()).collect::<HashSet<_>>()),
        ("deck_hash", rows.iter().map(|r| r.deck_hash.clone()).collect::<HashSet<_>>()),
        ("opponent", rows.iter().map(|r| r.opponent.clone()).collect::<HashSet<_>>()),
        ("think_budget_ms", rows.iter().map(|r| r.think_budget.clone()).collect::<HashSet<_>>()),
    ] {
        if vals.len() > 1 {
            let mut v: Vec<String> = vals.into_iter().collect();
            v.sort();
            fail(format!("{name} が混在しています: {}", v.join(", ")), &mut notes);
        }
    }
    // commit の混在は「同一コミットで対照と候補を取り直す」原則に反するので必ず出す
    let commits: HashSet<&str> = rows.iter().map(|r| r.commit.as_str()).collect();
    if commits.len() > 1 {
        let mut v: Vec<&str> = commits.into_iter().collect();
        v.sort_unstable();
        fail(format!("commit が混在しています: {}", v.join(", ")), &mut notes);
    }

    // **arm 内では strategy と正規化 env が一意**であること（追加レビュー指摘4）。
    // arm 間で違うのは正常だが、同じ arm のあるシャードだけ別 env / 別戦略で
    // 走っていたら別物の集計になる。親プロセスから継承した TSUITATE_* の
    // 混入もここで捕まる
    for arm in ["control", "candidate"] {
        let sigs: HashSet<String> = rows
            .iter()
            .filter(|r| r.arm == arm)
            .map(|r| {
                let env: Vec<String> =
                    r.env.iter().map(|(k, v)| format!("{k}={v}")).collect();
                let knobs: Vec<String> =
                    r.arm_knobs.iter().map(|(k, v)| format!("{k}={v}")).collect();
                format!(
                    "{}|{}|{}|{}",
                    r.strategy,
                    env.join(" "),
                    knobs.join(" "),
                    r.arm_config
                )
            })
            .collect();
        if sigs.len() > 1 {
            let mut v: Vec<String> = sigs.into_iter().collect();
            v.sort();
            fail(
                format!("arm={arm} 内で strategy / env / 設定が混在しています: {}", v.join(" || ")),
                &mut notes,
            );
        }
    }

    // **相手の実効 env が両 arm で同じ**であること（追加レビュー指摘1）。
    // env はプロセス全体に効くので、arm 固有のノブを相手が読むと
    // 「candidate 設定の v14 vs 既定の v14」を比べることになり、
    // 固定相手という前提が壊れる。JSONL 上は opponent 名が同じなので
    // 名前の一致だけでは捕まらない
    let opp_envs: HashSet<String> = rows
        .iter()
        .map(|r| {
            r.opponent_env
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect();
    if opp_envs.len() > 1 {
        let mut v: Vec<String> = opp_envs.into_iter().collect();
        v.sort();
        fail(
            format!(
                "相手の実効 env が arm 間で違います（同じ固定相手になっていません）: {}",
                v.join(" || ")
            ),
            &mut notes,
        );
    }
    // **ノブを渡したのに実効設定が動いていない**（PR #22 レビュー指摘2）。
    // arm ごとのノブが違うのに arm_config が同じなら、綴り間違いか無効値で
    // 「candidate == control」の実験を回したことになる
    let arm_sig = |arm: &str| -> Option<(String, String)> {
        rows.iter().find(|r| r.arm == arm).map(|r| {
            let knobs: Vec<String> =
                r.arm_knobs.iter().map(|(k, v)| format!("{k}={v}")).collect();
            (knobs.join(" "), r.arm_config.clone())
        })
    };
    if let (Some((ck, ccfg)), Some((tk, tcfg))) = (arm_sig("control"), arm_sig("candidate")) {
        if ck != tk && ccfg == tcfg {
            fail(
                format!(
                    "arm ごとのノブが違うのに実効設定が同じです（綴り間違い／無効値？）: \
                     control [{ck}] vs candidate [{tk}]"
                ),
                &mut notes,
            );
        }
    }

    // **相手の実効設定の指紋**が両 arm で同じであること（issue #21）。
    // 指紋は解決後の値から作るので、env の綴りが違っても実効が同じなら通る
    let opp_cfgs: HashSet<&str> = rows.iter().map(|r| r.opponent_config.as_str()).collect();
    if opp_cfgs.len() > 1 {
        let mut v: Vec<&str> = opp_cfgs.into_iter().collect();
        v.sort_unstable();
        fail(
            format!(
                "相手の実効設定が arm 間で違います（同じ固定相手になっていません）: {}",
                v.join(" || ")
            ),
            &mut notes,
        );
    }
    let leaked: HashSet<&str> = rows
        .iter()
        .flat_map(|r| r.opponent_env.keys().map(String::as_str))
        .filter(|k| {
            // arm 間で値が違う env を相手が読んでいるか
            let vals: HashSet<Option<&String>> = rows
                .iter()
                .map(|r| r.env.get(*k))
                .collect();
            vals.len() > 1
        })
        .collect();
    if !leaked.is_empty() {
        let mut v: Vec<&str> = leaked.into_iter().collect();
        v.sort_unstable();
        fail(
            format!("arm ごとに違う env を相手が読んでいます: {}", v.join(", ")),
            &mut notes,
        );
    }

    // 同じ (checkpoint, seed, arm) が2行あってはいけない（後勝ちで黙って上書きしない）
    let mut seen: HashSet<(String, u64, String)> = HashSet::new();
    for r in rows {
        let key = (r.checkpoint.clone(), r.seed, r.arm.clone());
        if !seen.insert(key) {
            fail(
                format!("重複行: {} seed{} arm={}", r.checkpoint, r.seed, r.arm),
                &mut notes,
            );
        }
    }

    // checkpoint ごとの seed 集合が揃っているか（欠けた shard は
    // 「片割れ欠損」にも出ないので、ここで検出する）
    let mut by_cp: BTreeMap<&str, BTreeSet<u64>> = BTreeMap::new();
    for r in rows {
        by_cp.entry(r.checkpoint.as_str()).or_default().insert(r.seed);
    }
    let expected: Option<&BTreeSet<u64>> = by_cp.values().max_by_key(|s| s.len());
    // **デッキ側の期待 checkpoint 集合と突き合わせる**（追加レビュー指摘1）。
    // 入力に現れた checkpoint 同士を比べるだけでは、shard の artifact が
    // 丸ごと欠けたときに「全部揃っている」ように見えてしまう
    if let Some(want) = expected_checkpoints {
        let got: HashSet<&str> = by_cp.keys().copied().collect();
        let missing: Vec<&String> = want.iter().filter(|c| !got.contains(c.as_str())).collect();
        if !missing.is_empty() {
            fail(
                format!(
                    "デッキにある checkpoint のうち {} 件が結果に出ていません（shard 欠落？）: {}",
                    missing.len(),
                    missing.iter().take(5).map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
                ),
                &mut notes,
            );
        }
        let extra: Vec<&str> = got.iter().filter(|c| !want.contains(**c)).copied().collect();
        if !extra.is_empty() {
            fail(
                format!(
                    "デッキに無い checkpoint が結果に含まれています: {}",
                    extra.iter().take(5).copied().collect::<Vec<_>>().join(", ")
                ),
                &mut notes,
            );
        }
    }
    if let Some(expected) = expected {
        let short: Vec<&str> = by_cp
            .iter()
            .filter(|(_, s)| s.len() < expected.len())
            .map(|(cp, _)| *cp)
            .collect();
        if !short.is_empty() {
            fail(
                format!(
                    "seed 数が揃っていない checkpoint が {} 件あります（期待 {} seed): {}",
                    short.len(),
                    expected.len(),
                    short.iter().take(5).copied().collect::<Vec<_>>().join(", ")
                ),
                &mut notes,
            );
        }
    }
    notes
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
}

/// 元対局（cluster）単位のブートストラップ。**seed を独立標本として数えない**:
/// 再標本化するのは cluster で、cluster 内の値はまとめて出入りする
fn cluster_bootstrap(clusters: &[Vec<f64>], b: usize, seed: u64, alpha: f64) -> (f64, f64) {
    if clusters.is_empty() {
        return (0.0, 0.0);
    }
    let means: Vec<f64> = clusters.iter().map(|c| mean(c)).collect();
    bootstrap_ci_of_means(&means, b, seed, alpha)
}

/// cluster 平均の列から percentile bootstrap CI を作る（本番の判定規則そのもの）。
/// **percentile は alpha に連動させる**（追加レビュー指摘2: `--alpha` を変えても
/// 2.5/97.5% 固定だと、MDE の alpha と主 CI の alpha が食い違う）
fn bootstrap_ci_of_means(means: &[f64], b: usize, seed: u64, alpha: f64) -> (f64, f64) {
    use rand::{Rng, SeedableRng};
    if means.is_empty() || b == 0 {
        return (0.0, 0.0);
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut samples: Vec<f64> = Vec::with_capacity(b);
    let mut buf = vec![0.0f64; means.len()];
    for _ in 0..b {
        for x in buf.iter_mut() {
            *x = means[rng.random_range(0..means.len())];
        }
        samples.push(mean(&buf));
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo_i = ((b as f64) * (alpha / 2.0)) as usize;
    let hi_i = (((b as f64) * (1.0 - alpha / 2.0)) as usize).min(b - 1);
    (samples[lo_i.min(b - 1)], samples[hi_i])
}

/// 標準正規の上側確率 p に対応する分位点（Acklam の有理近似。
/// 検出力計算に使うだけなので精度は 1e-9 で十分）
fn z_upper(p: f64) -> f64 {
    // 下側 1-p の分位点
    let q = 1.0 - p;
    let a = [
        -3.969_683_028_665_376e+01, 2.209_460_984_245_205e+02, -2.759_285_104_469_687e+02,
        1.383_577_518_672_690e+02, -3.066_479_806_614_716e+01, 2.506_628_277_459_239e+00,
    ];
    let b = [
        -5.447_609_879_822_406e+01, 1.615_858_368_580_409e+02, -1.556_989_798_598_866e+02,
        6.680_131_188_771_972e+01, -1.328_068_155_288_572e+01,
    ];
    let c = [
        -7.784_894_002_430_293e-03, -3.223_964_580_411_365e-01, -2.400_758_277_161_838e+00,
        -2.549_732_539_343_734e+00, 4.374_664_141_464_968e+00, 2.938_163_982_698_783e+00,
    ];
    let d = [
        7.784_695_709_041_462e-03, 3.224_671_290_700_398e-01, 2.445_134_137_142_996e+00,
        3.754_408_661_907_416e+00,
    ];
    let plow = 0.02425;
    if q <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if q >= 1.0 {
        return f64::INFINITY;
    }
    if q < plow {
        let t = (-2.0 * q.ln()).sqrt();
        return (((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5])
            / ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1.0);
    }
    if q > 1.0 - plow {
        let t = (-2.0 * (1.0 - q).ln()).sqrt();
        return -(((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5])
            / ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1.0);
    }
    let t = q - 0.5;
    let r = t * t;
    (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * t
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
}

/// 両側 alpha に対応する臨界値（alpha=0.05 → 1.96）
fn z_two_sided(alpha: f64) -> f64 {
    z_upper(alpha / 2.0)
}

/// **最小検出効果**（MDE）。CI の半幅（z_alpha·SE）ではなく、
/// 指定した検出力を持つ効果量 (z_alpha + z_beta)·SE。
/// alpha=0.05 / power=0.80 なら 2.80·SE で、CI 半幅の約1.43倍・必要 N は約2倍になる
/// （PR #20 レビュー指摘1）
fn mde(se: f64, z_alpha: f64, z_beta: f64) -> f64 {
    (z_alpha + z_beta) * se
}

/// **power simulation**: 本番と同じ判定規則（percentile cluster bootstrap CI が
/// 0 を跨がないか）を、経験分布から再標本化したデータへ当てて検出力を測る。
///
/// 追加レビュー指摘2 の対応。初版は `mean ± z_alpha·sample_SE` の z 検定を
/// 当てていたが、それは解析 MDE と同じ正規近似なので「独立の裏取り」にならず、
/// 「離散・同点多数・n=16 で bootstrap CI が怪しい」という肝心の点も確かめられない。
///
/// `centered` は中心化した cluster（または replicate 畳み込み後の）平均の列。
fn power_simulation_bootstrap(
    centered: &[f64],
    n: usize,
    delta: f64,
    alpha: f64,
    sims: usize,
    boot: usize,
    seed: u64,
) -> f64 {
    use rand::{Rng, SeedableRng};
    if centered.len() < 2 || n < 2 {
        return f64::NAN;
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut hits = 0usize;
    let mut sample = vec![0.0f64; n];
    for i in 0..sims {
        for x in sample.iter_mut() {
            *x = centered[rng.random_range(0..centered.len())] + delta;
        }
        let (lo, hi) = bootstrap_ci_of_means(&sample, boot, seed ^ (i as u64 + 1), alpha);
        let rejected = lo > 0.0 || hi < 0.0;
        if delta == 0.0 {
            // **delta=0 は type-I error（偽陽性率）を数える経路**。
            // 符号一致を要求すると構造上ゼロになり、検査にならない（追加レビュー指摘3）
            if rejected {
                hits += 1;
            }
        } else if (delta > 0.0 && lo > 0.0) || (delta < 0.0 && hi < 0.0) {
            // 「効果を検出した」= CI が 0 を跨がず、かつ符号が真の効果と同じ
            hits += 1;
        }
    }
    hits as f64 / sims as f64
}

/// cluster 平均の標本分散（= σ_b² + σ_w²/s の直接推定）。
/// cluster bootstrap が実際に再標本化しているのはこの分布なので、
/// 表に出す SE はここから作る
fn cluster_mean_var(clusters: &[Vec<f64>]) -> f64 {
    let n = clusters.len();
    if n < 2 {
        return 0.0;
    }
    let means: Vec<f64> = clusters.iter().map(|c| mean(c)).collect();
    let m = mean(&means);
    means.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n as f64 - 1.0)
}

/// 一元配置の分散成分（cluster 間 σ_b² と cluster 内 σ_w²）と ICC。
/// seed を増やしても cluster 内ノイズしか減らないことを数字で示すために出す
fn variance_components(clusters: &[Vec<f64>]) -> (f64, f64, f64) {
    let n = clusters.len();
    if n < 2 {
        return (0.0, 0.0, 0.0);
    }
    let s = mean(&clusters.iter().map(|c| c.len() as f64).collect::<Vec<_>>()).max(1.0);
    let grand = mean(&clusters.iter().flat_map(|c| c.iter().copied()).collect::<Vec<f64>>());
    let msb = clusters
        .iter()
        .map(|c| c.len() as f64 * (mean(c) - grand).powi(2))
        .sum::<f64>()
        / (n as f64 - 1.0);
    let within_df: f64 = clusters.iter().map(|c| (c.len() as f64 - 1.0).max(0.0)).sum();
    let msw = if within_df > 0.0 {
        clusters
            .iter()
            .map(|c| {
                let m = mean(c);
                c.iter().map(|x| (x - m).powi(2)).sum::<f64>()
            })
            .sum::<f64>()
            / within_df
    } else {
        0.0
    };
    let sb2 = ((msb - msw) / s).max(0.0);
    let sw2 = msw;
    let icc = if sb2 + sw2 > 0.0 { sb2 / (sb2 + sw2) } else { 0.0 };
    (sb2, sw2, icc)
}

fn pct(x: f64) -> String {
    format!("{:+.1}pt", x * 100.0)
}

/// 差ではなく水準（スコア率など）。符号を付けない
fn rate_pct(x: f64) -> String {
    format!("{:.1}%", x * 100.0)
}

/// 幅（SE・MDE・CI 半幅）。`±` を付けて出すので符号は付けない
fn width_pct(x: f64) -> String {
    format!("{:.1}pt", x.abs() * 100.0)
}

#[allow(clippy::too_many_lines)]
fn cmd_compare(args: &Args) {
    let paths: Vec<String> = {
        let mut v: Vec<String> = args.positional.iter().skip(1).cloned().collect();
        v.extend(args.all("jsonl"));
        if v.is_empty() {
            die("arm の JSONL を指定してください");
        }
        v
    };
    let boot: usize = args.num("boot", 10000);
    // 既知の通常 arena 差。**同じ candidate / control / opponent / 予算で
    // 測り直した値だけを渡すこと**（PR #20 レビュー指摘2）。別構成・別 matchup の
    // 記録を既知値に流用すると較正そのものが無意味になる
    let known: Option<f64> = args.get("known-arena-delta").and_then(|v| v.parse().ok());
    let label = args.get("label").map(str::to_string);
    let allow_incomplete = args.flag("allow-incomplete");
    // 検出力の設定（既定: 両側 5% / power 80%）
    let alpha: f64 = args.num("alpha", 0.05);
    let power: f64 = args.num("power", 0.80);
    let z_alpha = z_two_sided(alpha);
    let z_beta = z_upper(1.0 - power);

    let rows = parse_rows(&paths);
    if rows.is_empty() {
        die("行がありません");
    }
    // `--deck` を渡すと、デッキ側の期待 checkpoint 集合と突き合わせる
    // （shard 欠落の検出。CI は必ず渡す）
    let expected: Option<HashSet<String>> = args.get("deck").map(|p| {
        let path = PathBuf::from(p);
        let deck = Deck::load(&path).unwrap_or_else(|e| die(&e));
        let split = args.get("split").unwrap_or("all");
        deck.entries
            .iter()
            .filter(|e| split == "all" || e.split == split)
            .map(|e| e.id.clone())
            .collect()
    });
    let validation_notes = validate_rows(&rows, allow_incomplete, expected.as_ref());
    let label = label.unwrap_or_else(|| rows[0].experiment.clone());

    // (checkpoint, seed) でペアにする
    let mut pairs: BTreeMap<(String, u64), (Option<Row>, Option<Row>)> = BTreeMap::new();
    for r in &rows {
        let e = pairs.entry((r.checkpoint.clone(), r.seed)).or_insert((None, None));
        if r.arm == "control" {
            e.0 = Some(r.clone());
        } else {
            e.1 = Some(r.clone());
        }
    }
    let mut unpaired = 0usize;
    let mut paired: Vec<(Row, Row)> = vec![];
    for (_, (c, k)) in pairs {
        match (c, k) {
            (Some(c), Some(k)) => paired.push((c, k)),
            _ => unpaired += 1,
        }
    }
    if paired.is_empty() {
        die("control / candidate の対になった行がありません");
    }

    // cluster（元対局）ごとの delta
    let mut by_cluster: BTreeMap<String, Vec<&(Row, Row)>> = BTreeMap::new();
    for p in &paired {
        by_cluster.entry(p.0.source_game.clone()).or_default().push(p);
    }
    // cluster ごとの (seed, delta)。**seed は捨てない**: AB/BA を畳む replicate 化に要る
    let cluster_deltas_seeded = |metric: &str| -> Vec<Vec<(u64, f64)>> {
        by_cluster
            .values()
            .map(|v| {
                let mut d: Vec<(u64, f64)> = v
                    .iter()
                    .map(|(c, k)| (c.seed, k.metrics[metric] - c.metrics[metric]))
                    .collect();
                d.sort_by_key(|(seed, _)| *seed);
                d
            })
            .collect()
    };
    let cluster_deltas = |metric: &str| -> Vec<Vec<f64>> {
        cluster_deltas_seeded(metric)
            .into_iter()
            .map(|v| v.into_iter().map(|(_, d)| d).collect())
            .collect()
    };

    // **AB/BA を畳んだ replicate**（追加レビュー指摘3）。
    //
    // scheduler は arm 順を `(checkpoint 番号 + seed) % 2` で決めるので、同じ
    // checkpoint の seed 2k と 2k+1 は必ず逆の arm 順になる = 実行順成分について
    // **意図的に反相関**している。`σ_b² + σ_w²/s` は cluster 内残差が iid という
    // 前提なので、その反相関をモデル化できない（まさに今回見つけた差が消える）。
    // 連続する2 seed の平均を1 replicate として扱えば、replicate 間は iid と見なせる。
    let fold_replicates = |seeded: &[Vec<(u64, f64)>]| -> Option<Vec<Vec<f64>>> {
        let mut out = vec![];
        for cluster in seeded {
            // seed が 2k / 2k+1 の対で揃っていなければ畳めない
            let mut by_rep: BTreeMap<u64, Vec<f64>> = BTreeMap::new();
            for (seed, d) in cluster {
                by_rep.entry(seed / 2).or_default().push(*d);
            }
            if by_rep.values().any(|v| v.len() != 2) {
                return None;
            }
            out.push(by_rep.values().map(|v| mean(v)).collect::<Vec<f64>>());
        }
        Some(out)
    };

    let mut out = String::new();
    let commits: HashSet<&str> = rows.iter().map(|r| r.commit.as_str()).collect();
    let decks: HashSet<&str> = rows.iter().map(|r| r.deck_hash.as_str()).collect();
    let budgets: HashSet<&str> = rows.iter().map(|r| r.think_budget.as_str()).collect();
    let opponents: HashSet<&str> = rows.iter().map(|r| r.opponent.as_str()).collect();
    let strat = |arm: &str| -> String {
        let mut v: Vec<&str> = rows
            .iter()
            .filter(|r| r.arm == arm)
            .map(|r| r.strategy.as_str())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        v.sort_unstable();
        v.join(",")
    };
    out.push_str(&format!("## checkpoint arena: {label}\n\n"));
    out.push_str(&format!(
        "- commit `{}` / deck `{}` / 相手 {} / 思考予算 {} ms / prewarm 共有 {}\n",
        commits.iter().copied().collect::<Vec<_>>().join(","),
        decks.iter().copied().collect::<Vec<_>>().join(","),
        opponents.iter().copied().collect::<Vec<_>>().join(","),
        budgets.iter().copied().collect::<Vec<_>>().join(","),
        if rows.iter().any(|r| r.prewarm_shared) { "あり" } else { "なし" },
    ));
    out.push_str(&format!(
        "- control = `{}` / candidate = `{}`\n",
        strat("control"),
        strat("candidate")
    ));
    out.push_str(&format!(
        "- ペア {} 組 / 元対局 {} / seed {} / 片割れ欠損 {}\n",
        paired.len(),
        by_cluster.len(),
        paired.len() / by_cluster.len().max(1),
        unpaired
    ));
    for n in &validation_notes {
        out.push_str(&format!("- {n}\n"));
    }
    if unpaired > 0 {
        let msg = format!("**片割れの無い (checkpoint, seed) が {unpaired} 組あります**");
        if allow_incomplete {
            out.push_str(&format!("- 警告: {msg}\n"));
        } else {
            die(&format!("{msg}（意図的なら --allow-incomplete）"));
        }
    }

    // ---- 主指標 ----
    let score_clusters = cluster_deltas("score");
    let n = score_clusters.len();
    let s = mean(&score_clusters.iter().map(|c| c.len() as f64).collect::<Vec<_>>()).max(1.0);
    // 外挿に使う単位は **replicate**（AB/BA を畳んだもの）。畳めない設計
    // （seed が奇数個 / 2k と 2k+1 が揃っていない）では外挿表も best_seeds も出さない
    let score_reps = fold_replicates(&cluster_deltas_seeded("score"));
    let reps_per_cluster = score_reps
        .as_ref()
        .map(|r| mean(&r.iter().map(|c| c.len() as f64).collect::<Vec<_>>()).max(1.0));
    // **SE は cluster 平均の標本分散から直接取る**（= cluster bootstrap と整合する）。
    // 分散成分から σ_b²/n + σ_w²/(n·s) で組み立てると、σ_b² が 0 にクリップされた
    // ときに MSW をそのまま使うことになり SE を過大評価する（実測で bootstrap CI の
    // 半幅と2.5倍食い違った）。分散成分は (n, s) の外挿にだけ使い、
    // 「現在の s の列 = 直接推定の SE」になるよう σ_w² を総分散へ整合させる
    let total_var = cluster_mean_var(&score_clusters);
    let se = (total_var / n as f64).sqrt();
    let (sb2_raw, _msw, _icc_raw) = variance_components(&score_clusters);
    let sb2 = sb2_raw.min(total_var);
    let sw2 = s * (total_var - sb2);
    let icc = if total_var > 0.0 { sb2 / total_var } else { 0.0 };
    let d = mean(&score_clusters.iter().map(|c| mean(c)).collect::<Vec<f64>>());
    let (lo, hi) = cluster_bootstrap(&score_clusters, boot, 20260823, alpha);
    let ctrl_rate = mean(&paired.iter().map(|(c, _)| c.metrics["score"]).collect::<Vec<f64>>());
    let cand_rate = mean(&paired.iter().map(|(_, k)| k.metrics["score"]).collect::<Vec<f64>>());
    let improved = paired.iter().filter(|(c, k)| k.metrics["score"] > c.metrics["score"]).count();
    let worse = paired.iter().filter(|(c, k)| k.metrics["score"] < c.metrics["score"]).count();
    let same = paired.len() - improved - worse;

    out.push_str("\n### 勝敗（主指標）\n\n");
    out.push_str(&format!(
        "- candidate {:.1}% / control {:.1}% / **paired delta {} [{} , {}]**（元対局 cluster bootstrap {:.0}%）\n",
        cand_rate * 100.0,
        ctrl_rate * 100.0,
        pct(d),
        pct(lo),
        pct(hi),
        (1.0 - alpha) * 100.0
    ));
    out.push_str(&format!(
        "- ペア結果: 改善 {improved} / 同じ {same} / 悪化 {worse}\n"
    ));
    out.push_str(&format!(
        "- 分散成分: cluster間 σ_b² {sb2:.4} / cluster内 σ_w² {sw2:.4} / **ICC {icc:.3}** / empirical SE {} \n",
        pct(se)
    ));
    let mut split_tally: BTreeMap<&str, usize> = BTreeMap::new();
    for (c, _) in &paired {
        *split_tally.entry(c.split.as_str()).or_insert(0) += 1;
    }
    out.push_str(&format!(
        "- split: {}\n",
        split_tally
            .iter()
            .map(|(k, v)| format!("{k} {v}"))
            .collect::<Vec<_>>()
            .join(" / ")
    ));
    // 終局理由（安全性の共同指標。引き分け化・膠着への変質をここで見る）
    let mut reasons: BTreeMap<(&str, &str), usize> = BTreeMap::new();
    for r in &rows {
        *reasons.entry((r.arm.as_str(), r.reason.as_str())).or_insert(0) += 1;
    }
    out.push_str("- 終局理由: ");
    out.push_str(
        &reasons
            .iter()
            .map(|((arm, why), n)| format!("{arm}/{why} {n}"))
            .collect::<Vec<_>>()
            .join(" / "),
    );
    out.push('\n');
    let first_disagree = paired
        .iter()
        .filter(|(c, k)| c.first_move != k.first_move)
        .count();
    out.push_str(&format!(
        "- 最初の選択手の不一致率: {:.1}%（{first_disagree}/{}）\n",
        100.0 * first_disagree as f64 / paired.len() as f64,
        paired.len()
    ));
    // 実行順効果（AB/BA を均衡させたうえで、先に走った arm が有利かどうか）
    let order_delta = mean(
        &paired
            .iter()
            .map(|(c, k)| {
                let (first, second) = if c.arm_order == 0 { (c, k) } else { (k, c) };
                first.metrics["score"] - second.metrics["score"]
            })
            .collect::<Vec<f64>>(),
    );
    out.push_str(&format!(
        "- 実行順効果（先に走った arm − 後）: {}（0 付近なら交互実行が効いている）\n",
        pct(order_delta)
    ));

    // ---- 必要 N（CI 半幅と MDE を分けて出す） ----
    // 1.96·SE は **95% CI の半幅**であって MDE ではない。指定した検出力を持つ
    // 最小検出効果は (z_alpha + z_beta)·SE で、power 80% なら約1.43倍・
    // 必要 N は約2倍になる（PR #20 レビュー指摘1）
    out.push_str(&format!(
        "\n### 必要 N と最小検出効果（実測分散から、両側 alpha={alpha:.2} / power={:.0}%）\n\n",
        power * 100.0
    ));
    out.push_str(&format!(
        "係数: CI 半幅 = {z_alpha:.2}·SE / **MDE = {:.2}·SE**\n\n",
        z_alpha + z_beta
    ));
    match (&score_reps, reps_per_cluster) {
        (Some(reps), Some(r)) if r >= 1.0 => {
            // replicate（AB/BA を畳んだ 2 seed）単位で分散成分を取る。
            // replicate 間なら iid と見なせるので 1/r の外挿が正当化される
            let rep_total = cluster_mean_var(reps);
            let (rb2_raw, _, _) = variance_components(reps);
            let rb2 = rb2_raw.min(rep_total);
            let rw2 = r * (rep_total - rb2);
            out.push_str(&format!(
                "単位は **replicate = 連続する2 seed の AB/BA 平均**（実測 {r:.0} replicate/元対局 = {:.0} seed）。\n\n",
                r * 2.0
            ));
            if r < 1.5 {
                // **replicate が1つだと cluster 内自由度が 0 で σ_r² が同定できない**。
                // 未同定成分を 0 に置いたまま 4/6/10 seed へ外挿すると、
                // 「データから得た結論」ではなく「0 と置いた結果」を出すことになる
                // （PR #20 追加レビュー指摘2）
                out.push_str("**seed 数の外挿は出さない**: 1元対局あたり replicate が1つ（= 2 seed）しかないので replicate 間分散 σ_r² が同定できない（cluster 内自由度 0）。未同定成分を 0 に置いた外挿は data ではなく仮定なので抑止する。**seed 数の効果を測るには 4 seed 以上（2 replicate 以上）の実測が要る**。\n\n");
                out.push_str(&format!(
                    "この 2-seed 設計での実測値だけ: 元対局 {n} / SE ±{:.1}pt / MDE ±{:.1}pt / 元対局数を増やしたときの MDE は下表\n\n",
                    se * 100.0,
                    mde(se, z_alpha, z_beta) * 100.0
                ));
                out.push_str("| 元対局数 | 16 | 32 | 64 | 128 | 256 |\n|---|---|---|---|---|---|\n");
                let cells: Vec<String> = [16usize, 32, 64, 128, 256]
                    .iter()
                    .map(|&nn| {
                        let se_n = (rep_total / nn as f64).sqrt();
                        format!("±{:.1}pt", mde(se_n, z_alpha, z_beta) * 100.0)
                    })
                    .collect();
                out.push_str(&format!("| MDE | {} |\n", cells.join(" | ")));
            } else {
                out.push_str("replicate 間なら iid と見なせるので 1/r の外挿が正当化される（生の seed は arm 順で意図的に反相関しており、σ_w²/s の外挿は使えない）。\n\n");
                out.push_str("| 元対局数 \\ seed 数 | 2 | 4 | 6 | 10 |\n|---|---|---|---|---|\n");
                for nn in [16usize, 32, 64, 128, 256] {
                    let cells: Vec<String> = [1usize, 2, 3, 5]
                        .iter()
                        .map(|&rr| {
                            let se_c = (rb2 / nn as f64 + rw2 / (nn as f64 * rr as f64)).sqrt();
                            format!(
                                "MDE ±{:.1}pt<br>(CI半幅 ±{:.1}pt)",
                                mde(se_c, z_alpha, z_beta) * 100.0,
                                z_alpha * se_c * 100.0
                            )
                        })
                        .collect();
                    out.push_str(&format!("| {nn} | {} |\n", cells.join(" | ")));
                }
            }
        }
        _ => {
            out.push_str("**外挿表は出さない**: seed が AB/BA の対（2k と 2k+1）で揃っていないため replicate を作れない。生の seed は arm 順で意図的に反相関しているので `σ_b² + σ_w²/s` の外挿は前提を満たさない（追加レビュー指摘3）。**seed は 2 の倍数で取り直すこと**。\n\n");
            out.push_str(&format!(
                "この設計での実測値だけ: 元対局 {n} / seed {s:.0} / SE ±{:.1}pt / MDE ±{:.1}pt（{:.2}·SE）\n",
                se * 100.0,
                mde(se, z_alpha, z_beta) * 100.0,
                z_alpha + z_beta
            ));
        }
    }
    if s < 1.5 {
        out.push_str("\n**注意: seed 1 では cluster 内分散が同定できず、AB/BA も cluster 内で閉じない**。seed は 2 の倍数で取り直すこと。\n");
    }

    // ---- power simulation（本番の判定規則をそのまま当てる） ----
    // 主結果と「重大悪化の見逃し」は percentile cluster bootstrap CI で判定している。
    // simulation もその規則を使う（追加レビュー指摘2）。単位は外挿と同じ replicate、
    // 畳めなければ cluster 平均そのまま（その旨を書く）
    let (sim_units, sim_label): (Vec<f64>, &str) = match &score_reps {
        Some(reps) => (reps.iter().map(|c| mean(c)).collect(), "replicate 畳み込み後の元対局平均"),
        None => (
            score_clusters.iter().map(|c| mean(c)).collect(),
            "元対局平均（AB/BA 未畳み込み）",
        ),
    };
    let gm = mean(&sim_units);
    let centered: Vec<f64> = sim_units.iter().map(|x| x - gm).collect();
    let sims: usize = args.num("power-sims", 600);
    let sim_boot: usize = args.num("power-boot", 400);
    out.push_str(&format!(
        "\n#### power simulation（{sim_label}から再標本化、{sims} 回 × bootstrap {sim_boot} 回）\n\n"
    ));
    out.push_str(&format!(
        "判定規則は**本番と同じ** percentile cluster bootstrap CI（alpha={alpha:.2}）が 0 を跨がないこと。\n\n"
    ));
    out.push_str("| 効果量 \\ 元対局数 | 16 | 32 | 64 | 128 |\n|---|---|---|---|---|\n");
    // **偽陽性率（type-I）を必ず出す**。A/A が有意になったときに
    // 「判定規則が甘いのか、たまたま引いたのか」を切り分けられないと、
    // ゲートとして使えるかどうかが決まらない（2026-08-24 の A/A が +7.0pt
    // [+0.2, +14.1] で 0 を外したのが発端）
    {
        let cells: Vec<String> = [16usize, 32, 64, 128]
            .iter()
            .map(|&nn| {
                let fp = power_simulation_bootstrap(
                    &centered, nn, 0.0, alpha, sims, sim_boot, 20260823,
                );
                format!("{:.1}%", fp * 100.0)
            })
            .collect();
        out.push_str(&format!(
            "| **0pt（偽陽性率。名目 {:.0}%）** | {} |\n",
            alpha * 100.0,
            cells.join(" | ")
        ));
    }
    for pt in [5.0f64, 10.0, 15.0, 20.0, 25.0] {
        let cells: Vec<String> = [16usize, 32, 64, 128]
            .iter()
            .map(|&nn| {
                let pw = power_simulation_bootstrap(
                    &centered, nn, -pt / 100.0, alpha, sims, sim_boot, 20260823,
                );
                format!("{:.0}%", pw * 100.0)
            })
            .collect();
        out.push_str(&format!("| −{pt:.0}pt | {} |\n", cells.join(" | ")));
    }
    out.push_str(&format!(
        "\n（`power={:.0}%` を満たす最小の効果量が、その N での実効 MDE。解析式の MDE と食い違うなら、離散性・同点の多さで bootstrap CI が正規近似からずれているということ）\n",
        power * 100.0
    ));

    // ---- CPU あたりの情報量 ----
    // 「同じ検出力を得る CPU コスト」の比較。
    // **cluster 構造を無視してはいけない**（PR #20 レビュー指摘3）: CI と SE は
    // 元対局ごとの seed 平均を標本にしているので、効率も cluster 単位で数える。
    //   SE²(n クラスタ) = (σ_b² + σ_w²/s) / n、総コスト = n·s·(1ペアの CPU 秒)
    //   → SE²×コスト = (σ_b² + σ_w²/s)·s·(1ペアの CPU 秒)   … n に依らない
    // seed を増やすと σ_b² のぶんコストだけ増える（ICC>0 なら s は小さいほど良い）
    let cpu_per_pair = rows.iter().map(|r| r.total_ms).sum::<f64>() / 1000.0 / paired.len() as f64;
    // summary JSON 用（replicate が作れないときは None = 出さない）
    let var_cpu_sec: Option<f64> = match (&score_reps, reps_per_cluster) {
        (Some(reps), Some(r)) => {
            let rep_total = cluster_mean_var(reps);
            let (rb2_raw, _, _) = variance_components(reps);
            let rb2 = rb2_raw.min(rep_total);
            let rw2 = r * (rep_total - rb2);
            Some((rb2 + rw2 / r) * r * 2.0 * cpu_per_pair)
        }
        _ => None,
    };
    out.push_str("\n### CPU あたりの情報量（cluster 単位）\n\n");
    match (&score_reps, reps_per_cluster) {
        (Some(reps), Some(r)) => {
            // 単位は replicate（= 2 seed）。cost/replicate = 2 × 1ペアCPU秒
            let rep_total = cluster_mean_var(reps);
            let (rb2_raw, _, _) = variance_components(reps);
            let rb2 = rb2_raw.min(rep_total);
            let rw2 = r * (rep_total - rb2);
            let cpu_per_rep = 2.0 * cpu_per_pair;
            let var_cpu_at = |rr: f64| (rb2 + rw2 / rr) * rr * cpu_per_rep;
            let var_cpu = var_cpu_at(r);
            out.push_str(&format!(
                "1ペア（arm 2本）あたり {cpu_per_pair:.0} CPU秒 / 1 replicate（2 seed）あたり {cpu_per_rep:.0} CPU秒 / 実測 {r:.0} replicate\n\n"
            ));
            out.push_str(&format!(
                "**var·CPU秒 = (σ_b² + σ_r²/r)·r·(1 replicate の CPU秒) = {var_cpu:.0}**（小さいほど効率がよい。n に依らない）\n\n"
            ));
            if r < 1.5 {
                // replicate 1つでは σ_r² が同定できないので、replicate 数を変えた
                // var·CPU秒 は「未同定成分を 0 に置いた結果」でしかない（追加レビュー指摘2）
                out.push_str("**replicate 数を変えた比較は出さない**: replicate が1つ（= 2 seed）では σ_r² が同定できず、増減の結論はデータではなく仮定になる。4 seed 以上の実測が要る。\n");
            } else {
                out.push_str("| replicate 数（= seed÷2） | 1 | 2 | 3 | 5 |\n|---|---|---|---|---|\n");
                out.push_str(&format!(
                    "| var·CPU秒 | {:.0} | {:.0} | {:.0} | {:.0} |\n",
                    var_cpu_at(1.0),
                    var_cpu_at(2.0),
                    var_cpu_at(3.0),
                    var_cpu_at(5.0)
                ));
                if rb2 > 0.0 {
                    out.push_str("\nσ_b² > 0 なので replicate を増やすほど var·CPU秒 は悪化する（元対局数を増やすべき）。\n");
                } else {
                    out.push_str("\nこの実測では σ_b² = 0 なので replicate 数は var·CPU秒 に中立。\n");
                }
            }
        }
        _ => {
            out.push_str(&format!(
                "1ペア（arm 2本）あたり {cpu_per_pair:.0} CPU秒。\n\n"
            ));
            out.push_str("**var·CPU秒は出さない**: AB/BA を畳めない seed 構成なので、replicate 単位の分散が推定できない（追加レビュー指摘3）。\n");
        }
    }
    out.push_str("\n通常 arena と比べるときは、arena 側も同じ `match_seed` の2本から局ごとのペア差の分散を実測して並べること（現状の参考値 183 は Var=0.5 の仮定に乗っている）。\n");

    // ---- 安全性の共同指標 ----
    out.push_str("\n### 安全性の共同指標（元対局単位のペア差）\n\n");
    out.push_str(&format!(
        "| 指標 | control | candidate | delta | {:.0}% CI |\n|---|---:|---:|---:|---|\n",
        (1.0 - alpha) * 100.0
    ));
    for (key, name) in METRICS.iter().skip(1) {
        let cl = cluster_deltas(key);
        let dd = mean(&cl.iter().map(|c| mean(c)).collect::<Vec<f64>>());
        let (l, h) = cluster_bootstrap(&cl, boot.min(4000), 20260824, alpha);
        let cv = mean(&paired.iter().map(|(c, _)| c.metrics[key]).collect::<Vec<f64>>());
        let kv = mean(&paired.iter().map(|(_, k)| k.metrics[key]).collect::<Vec<f64>>());
        out.push_str(&format!(
            "| {name} | {cv:.3} | {kv:.3} | {dd:+.3} | [{l:+.3}, {h:+.3}] |\n"
        ));
    }

    // ---- 層別 delta ----
    out.push_str("\n### 層別 delta\n\n| 層 | 件 | delta |\n|---|---:|---:|\n");
    let mut tag_set: Vec<String> = paired.iter().flat_map(|(c, _)| c.tags.clone()).collect();
    tag_set.sort();
    tag_set.dedup();
    for tag in &tag_set {
        let sub: Vec<&(Row, Row)> = paired.iter().filter(|(c, _)| c.tags.contains(tag)).collect();
        if sub.is_empty() {
            continue;
        }
        let dd = mean(
            &sub.iter()
                .map(|(c, k)| k.metrics["score"] - c.metrics["score"])
                .collect::<Vec<f64>>(),
        );
        out.push_str(&format!("| {tag} | {} | {} |\n", sub.len(), pct(dd)));
    }

    // ---- コスト分解 ----
    out.push_str("\n### コスト分解（1 arm あたり、秒）\n\n");
    out.push_str("| 層 | 件 | prewarm(両側) | continuation | 合計 |\n|---|---:|---:|---:|---:|\n");
    let mut phases: Vec<&str> = rows
        .iter()
        .map(|r| phase_tag(r.start_ply as usize))
        .collect();
    phases.sort();
    phases.dedup();
    for ph in phases.iter().chain(std::iter::once(&"全体")) {
        let sub: Vec<&Row> = rows
            .iter()
            .filter(|r| *ph == "全体" || phase_tag(r.start_ply as usize) == *ph)
            .collect();
        if sub.is_empty() {
            continue;
        }
        out.push_str(&format!(
            "| {ph} | {} | {:.1} | {:.1} | {:.1} |\n",
            sub.len(),
            mean(&sub.iter().map(|r| r.prewarm_ms / 1000.0).collect::<Vec<f64>>()),
            mean(&sub.iter().map(|r| r.continuation_ms / 1000.0).collect::<Vec<f64>>()),
            mean(&sub.iter().map(|r| r.total_ms / 1000.0).collect::<Vec<f64>>()),
        ));
    }
    let cpu_h: f64 = rows.iter().map(|r| r.total_ms).sum::<f64>() / 3_600_000.0;
    out.push_str(&format!(
        "\n合計 CPU {cpu_h:.2} 時間（arm 2本ぶん / 全 {} 行）\n",
        rows.len()
    ));

    // ---- 大きな回帰 ----
    let mut per_cp: Vec<(String, f64, u64, String)> = by_cluster
        .iter()
        .map(|(g, v)| {
            (
                g.clone(),
                mean(&v.iter().map(|(c, k)| k.metrics["score"] - c.metrics["score"]).collect::<Vec<f64>>()),
                v[0].0.start_ply,
                v[0].0.checkpoint.clone(),
            )
        })
        .collect();
    per_cp.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    out.push_str("\n### 大きな回帰（再現コマンドつき）\n\n");
    for (_, dd, ply, cp) in per_cp.iter().take(5).filter(|x| x.1 < 0.0) {
        out.push_str(&format!(
            "- `{cp}` {}: `cargo run --release --bin rank_probe -- checkpoint-arena/games/{cp}.kif {ply} 3`\n",
            pct(*dd)
        ));
    }

    if let Some(k) = known {
        // **真値 0（A/A）は符号を持たない**。`f64::signum(0.0)` は +1 を返すので
        // 素直に比べると「正へ振れた A/A」まで「符号一致: はい」と出る
        // （PR #23 2回目レビュー指摘2）。report 側の扱いに揃え、A/A は
        // 「CI が 0 を含むか＝偽陽性か」で読む
        let verdict = if k.abs() <= 1e-9 {
            format!(
                "A/A（真値 0）: CI [{}, {}] が 0 を{}",
                pct(lo),
                pct(hi),
                if lo <= 0.0 && hi >= 0.0 {
                    "含む＝偽陽性なし"
                } else {
                    "含まない＝**偽陽性**"
                }
            )
        } else {
            format!(
                "符号一致: {}",
                if (d * 100.0).signum() == k.signum() { "はい" } else { "いいえ" }
            )
        };
        out.push_str(&format!(
            "\n### 較正\n\n既知の通常 arena 差 {k:+.1}pt に対し checkpoint delta {}（{verdict}）\n",
            pct(d),
        ));
    }

    print!("{out}");
    if let Some(p) = args.get("markdown") {
        if let Some(dir) = Path::new(p).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        std::fs::write(p, &out).unwrap_or_else(|e| die(&format!("{p}: {e}")));
    }
    if let Some(p) = args.get("json") {
        let summary = serde_json::json!({
            "schema": SUMMARY_SCHEMA,
            "kind": "checkpoint-compare",
            "label": label,
            "pairs": paired.len(),
            "clusters": by_cluster.len(),
            "seeds": s,
            "delta_pt": d * 100.0,
            "ci_lo_pt": lo * 100.0,
            "ci_hi_pt": hi * 100.0,
            "se_pt": se * 100.0,
            "ci_half_pt": z_alpha * se * 100.0,
            "mde_pt": mde(se, z_alpha, z_beta) * 100.0,
            "alpha": alpha,
            "power": power,
            "icc": icc,
            "sigma_b2": sb2,
            "sigma_w2": sw2,
            "control_rate": ctrl_rate,
            "candidate_rate": cand_rate,
            "improved": improved,
            "same": same,
            "worse": worse,
            "order_effect_pt": order_delta * 100.0,
            "first_move_disagree": first_disagree as f64 / paired.len() as f64,
            "fouls_delta": mean(&cluster_deltas("fouls_me").iter().map(|c| mean(c)).collect::<Vec<f64>>()),
            "cpu_hours": cpu_h,
            "cpu_sec_per_pair": cpu_per_pair,
            "var_cluster": sb2 + sw2 / s,
            "var_cpu_sec": var_cpu_sec,
            "replicates_per_cluster": reps_per_cluster,
            "think_budget_ms": budgets.iter().copied().collect::<Vec<_>>().join(","),
            "commit": commits.iter().copied().collect::<Vec<_>>().join(","),
            "deck_hash": decks.iter().copied().collect::<Vec<_>>().join(","),
            "known_arena_delta_pt": known,
        });
        if let Some(dir) = Path::new(p).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        std::fs::write(p, serde_json::to_string_pretty(&summary).unwrap() + "\n")
            .unwrap_or_else(|e| die(&format!("{p}: {e}")));
    }
}

// ---------------------------------------------------------------------------
// report（実験横断: 符号一致・順位相関・重大悪化の見逃し）
// ---------------------------------------------------------------------------

/// スピアマンの順位相関（同順位は平均順位）
fn spearman(x: &[f64], y: &[f64]) -> f64 {
    let rank = |v: &[f64]| -> Vec<f64> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
        let mut r = vec![0.0; v.len()];
        let mut i = 0;
        while i < idx.len() {
            let mut j = i;
            while j + 1 < idx.len() && (v[idx[j + 1]] - v[idx[i]]).abs() < 1e-12 {
                j += 1;
            }
            let avg = (i + j) as f64 / 2.0 + 1.0;
            for &k in &idx[i..=j] {
                r[k] = avg;
            }
            i = j + 1;
        }
        r
    };
    let (rx, ry) = (rank(x), rank(y));
    let (mx, my) = (mean(&rx), mean(&ry));
    let num: f64 = rx.iter().zip(&ry).map(|(a, b)| (a - mx) * (b - my)).sum();
    let dx: f64 = rx.iter().map(|a| (a - mx).powi(2)).sum::<f64>().sqrt();
    let dy: f64 = ry.iter().map(|b| (b - my).powi(2)).sum::<f64>().sqrt();
    if dx * dy == 0.0 { 0.0 } else { num / (dx * dy) }
}

fn cmd_report(args: &Args) {
    let paths: Vec<String> = args.positional.iter().skip(1).cloned().collect();
    if paths.is_empty() {
        die("summary.json を指定してください");
    }
    let mut rows: Vec<serde_json::Value> = vec![];
    for p in &paths {
        let text = std::fs::read_to_string(p).unwrap_or_else(|e| die(&format!("{p}: {e}")));
        let v: serde_json::Value =
            serde_json::from_str(&text).unwrap_or_else(|e| die(&format!("{p}: {e}")));
        // **summary にも schema の契約を適用する**（PR #20 5回目レビュー指摘1）
        check_summary_schema(&v, p).unwrap_or_else(|e| die(&e));
        // **arena-var の summary を混ぜない**。schema は同じ 3 でも中身の契約が
        // 違うので、通すと全列 NaN の行が横断表に並ぶ（欠損が「測った 0」に見える）。
        // `kind` が無いものは checkpoint 側（この欄より前に作られた summary）
        if let Some(kind) = v["kind"].as_str() {
            if kind != "checkpoint-compare" {
                die(&format!(
                    "{p}: kind={kind} は横断レポートの対象外です（checkpoint の compare が出した summary を渡してください）"
                ));
            }
        }
        rows.push(v);
    }
    let mut out = String::new();
    out.push_str("## checkpoint arena 横断レポート\n\n");
    out.push_str(
        "| 実験 | 予算ms | 元対局 | seed | 既知arena | checkpoint delta | CI | MDE | ICC | 反則delta | var·CPU秒 | CPU時間 |\n\
         |---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|\n",
    );
    let f = |v: &serde_json::Value, k: &str| v[k].as_f64().unwrap_or(f64::NAN);
    for v in &rows {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {:+.1}pt | [{:+.1}, {:+.1}] | ±{:.1}pt | {:.2} | {:+.2} | {:.0} | {:.2} |\n",
            v["label"].as_str().unwrap_or("-"),
            v["think_budget_ms"].as_str().unwrap_or("-"),
            f(v, "clusters"),
            f(v, "seeds"),
            v["known_arena_delta_pt"]
                .as_f64()
                .map(|x| format!("{x:+.1}pt"))
                .unwrap_or_else(|| "-".into()),
            f(v, "delta_pt"),
            f(v, "ci_lo_pt"),
            f(v, "ci_hi_pt"),
            f(v, "mde_pt"),
            f(v, "icc"),
            f(v, "fouls_delta"),
            f(v, "var_cpu_sec"),
            f(v, "cpu_hours"),
        ));
    }

    let known: Vec<&serde_json::Value> = rows
        .iter()
        .filter(|v| v["known_arena_delta_pt"].as_f64().is_some())
        .collect();
    if known.is_empty() {
        out.push_str(
            "\n**未較正**: 既知の通常 arena 差が1件も無い。\n             較正には、**同じ candidate / control / opponent / 予算で測り直した** arena の差を\n             `compare --known-arena-delta` で渡すこと（別構成・別 matchup の過去の記録は使えない。\n             PR #20 レビュー指摘2）。通常 arena 側は `arena.yml` の `pair_with` が\n             `checkpoint_arena arena-var` を回して出す。\n",
        );
    }
    if !known.is_empty() {
        let k: Vec<f64> = known.iter().map(|v| f(v, "known_arena_delta_pt")).collect();
        let c: Vec<f64> = known.iter().map(|v| f(v, "delta_pt")).collect();
        // **真値 0（A/A 等）は符号一致から外す**。`f64::signum(0.0)` は +1 を返すので、
        // 数えると「正に振れた A/A」が符号一致にカウントされて甘い数字になる
        let signed: Vec<(f64, f64)> = k
            .iter()
            .zip(&c)
            .filter(|(a, _)| a.abs() > 1e-9)
            .map(|(a, b)| (*a, *b))
            .collect();
        let zeros = k.len() - signed.len();
        if signed.len() >= 2 {
            let agree = signed.iter().filter(|(a, b)| a.signum() == b.signum()).count();
            out.push_str(&format!(
                "\n- 符号一致: {agree}/{}{}\n",
                signed.len(),
                if zeros > 0 {
                    format!("（真値 0 の {zeros} 件は符号を持たないので除外）")
                } else {
                    String::new()
                },
            ));
        }
        // **2点の順位相関は必ず ±1** になるので出さない（情報がゼロなのに
        // 「相関 −1.000」と書くと較正が済んだように読める）
        if signed.len() >= 3 {
            let (ks, cs): (Vec<f64>, Vec<f64>) = signed.iter().copied().unzip();
            out.push_str(&format!("- 順位相関（スピアマン）: {:.3}\n", spearman(&ks, &cs)));
        } else {
            out.push_str(&format!(
                "\n**較正は {} 点のみ**: 順位相関は3件以上でないと出さない（2点では必ず ±1 になる）。\n",
                signed.len()
            ));
        }
        // 重大悪化（既知 −8pt 以下）に何が起きたか。
        // **「検出できなかった」と「逆符号で有意になった」を分ける**:
        // 後者は「悪化していない」ではなく「改善している」と読める出力なので、
        // ゲートとしては前者より明確に悪い（2026-08-24 の drop_probe_gate がこれ）
        let severe: Vec<&&serde_json::Value> = known.iter().filter(|v| f(v, "known_arena_delta_pt") <= -8.0).collect();
        let opposite = severe.iter().filter(|v| f(v, "ci_lo_pt") > 0.0).count();
        let inconclusive = severe
            .iter()
            .filter(|v| f(v, "ci_lo_pt") <= 0.0 && f(v, "ci_hi_pt") >= 0.0)
            .count();
        out.push_str(&format!(
            "- 重大悪化（既知 −8pt 以下）{} 件: 検出 {} / 検出できず（CI が 0 を跨いだ）{inconclusive} / **逆符号で有意 {opposite}**\n",
            severe.len(),
            severe.len() - inconclusive - opposite,
        ));
    }
    print!("{out}");
    if let Some(p) = args.get("markdown") {
        if let Some(dir) = Path::new(p).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        std::fs::write(p, &out).unwrap_or_else(|e| die(&format!("{p}: {e}")));
    }
}


// ---------------------------------------------------------------------------
// arena-var: 通常 arena の**局ごとのペア差**の分散を実測する
// ---------------------------------------------------------------------------

/// `ARENA_GAMES_JSON` の1行（1対局）。
///
/// **なぜ要るか**（issue #19 の P0 の残り）: checkpoint arena の効率
/// （var·CPU秒）を「通常 arena の何倍か」と言うには、arena 側の
/// `Var(delta)` を実測した値が要る。従来の参考値 183 は `Var(delta)=0.5`
/// （＝ペアリングが全く効かない）の**仮定**に乗っていた。
#[derive(Debug, Clone)]
struct GameRow {
    candidate: String,
    baseline: String,
    match_seed: u64,
    /// **shard ずらしと基準ごとの XOR を掛ける前の base seed**。記録に残る
    /// `match_seed` は実効値（シャードごと・基準ごとに違う）なので、
    /// 「同じ run か」「事前登録した seed か」の検査はこちらで行う
    /// （PR #41 レビュー3巡目）
    match_seed_base: u64,
    /// この行を出したシャード番号。shard 集合の完全性検査に使う
    match_seed_shard: u64,
    /// **実行の識別子**（CI は GITHUB_RUN_ID、ローカルは ARENA_EXPERIMENT_ID）。
    /// `match_seed_base` は実験条件であって実行の識別子ではない: 同じ base で
    /// 取り直した2 run から shard を半分ずつ選んでも、base 1値・shard 集合完全・
    /// 局数一致で他の検査を全部通る（PR #41 レビュー4巡目）
    run_id: String,
    /// 同じ run の re-run attempt も別実行として区別する
    run_attempt: u64,
    /// 候補 run が対照に指した run（arena.yml の `-f pair_with=`）。あれば
    /// 対照の run_id との一致を検査する（対照の取り違えをここで閉じる）
    pair_with: Option<String>,
    /// 計測前に commit した validation manifest の指紋（無い run は null =
    /// #40 の採否では判定不能）
    balance_manifest: Option<String>,
    game_no: u64,
    a_is_sente: bool,
    score_a: f64,
    reason: String,
    plies: f64,
    fouls_a: f64,
    fouls_b: f64,
    fouls_in_check_a: f64,
    think_ms_a: f64,
    think_ms_b: f64,
    moves_a: f64,
    commit: String,
    cand_knobs: BTreeMap<String, String>,
    clock: String,
    /// 候補側の実効思考予算
    budget: String,
    /// **固定相手の実効思考予算**。片側だけ見ていると 700ms と 2000ms の
    /// 2 run が「同じ条件のペア」として通る（PR #23 レビュー指摘1・2）
    budget_opp: String,
    /// 候補の実効設定の指紋
    cand_config: String,
    /// **固定相手の実効挙動**の指紋。名前が同じでも共通 env・共有モデル pin が
    /// 違えば別物なので、ここを必須一致にする
    baseline_behavior: String,
    /// 両側に効くプロセス env（`-f env=`）
    shared_env: BTreeMap<String, String>,
}

/// schema 2 で実効条件の指紋を必須にした。schema 3 で `match_seed_base` /
/// `match_seed_shard` を必須にした（PR #41 レビュー3巡目: 記録の `match_seed` は
/// シャードずらし＋基準 XOR 済みの実効値なので、shard 完全性は base / shard 列で
/// しか検査できない）。schema 4 で `run_id` / `run_attempt`（実行の識別子）を
/// 必須にした（同4巡目: base は実験条件なので、同じ base で取り直した複数 run の
/// shard 混ぜを base の一意性では検出できない）。古い schema は**集計から弾く**:
/// 欠損を空値で埋めると検査を素通りする（PR #22 と同じ理由）
const GAME_ROW_SCHEMA: u64 = 4;

/// **ペア差の分散を同定するのに要る最小のペア数**。1ペアでは自由度が 0 で、
/// n−1 の標本分散が定義できない（`variance` は 0 を返す）
const MIN_ARENA_PAIRS: usize = 2;

fn arena_var_pair_error(n: usize) -> String {
    format!(
        "ペアになった対局が {n} 局しかありません（最低 {MIN_ARENA_PAIRS} 局）。\n            \
         1局では Var(ペア差) の自由度が 0 で、SE・MDE・CI が「完全に精密」に見えてしまいます。\n            \
         同じ match_seed の記録が両 arm に揃っているか確認してください\n            \
         （--allow-incomplete は欠損の許容であって、自由度は作れないので効きません）"
    )
}

/// 1行=1局の JSONL を読む。**失敗は `die` でなく `Result`** で返す:
/// プロセスごと落とすとパーサの抜け（欠損キーを空値で埋めていないか）を
/// テストで検出できない（PR #23 2回目レビュー指摘3）
fn parse_game_rows_text(text: &str, where_prefix: &str) -> Result<Vec<GameRow>, String> {
    let mut rows = vec![];
    for (ln, line) in text.lines().enumerate() {
        if !line.trim().starts_with('{') {
            continue;
        }
        let where_ = format!("{where_prefix}:{}", ln + 1);
        let v: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| format!("{where_}: JSON として読めません: {e}"))?;
        let schema = v.get("schema").and_then(|x| x.as_u64()).unwrap_or(0);
        if schema != GAME_ROW_SCHEMA {
            return Err(format!(
                "{where_}: schema {schema} は未対応（対応 {GAME_ROW_SCHEMA}）"
            ));
        }
        let miss = |k: &str| format!("{where_}: {k} がありません");
        let req_s = |k: &str| -> Result<String, String> {
            v.get(k)
                .and_then(|x| x.as_str())
                .map(|s| s.to_string())
                .ok_or_else(|| miss(k))
        };
        let req_f = |k: &str| -> Result<f64, String> {
            v.get(k).and_then(|x| x.as_f64()).ok_or_else(|| miss(k))
        };
        let req_u = |k: &str| -> Result<u64, String> {
            v.get(k).and_then(|x| x.as_u64()).ok_or_else(|| miss(k))
        };
        // **欠損は空値で埋めずにエラー**（PR #23 レビュー指摘2）。
        // null は「その戦略には概念が無い」の意味なので値として受ける
        let req_json = |k: &str| -> Result<String, String> {
            v.get(k).map(|x| x.to_string()).ok_or_else(|| miss(k))
        };
        // **map も欠損・object 以外はエラー**。`unwrap_or_default()` で空 map に
        // 落とすと「両 run とも env 指定なしで一致」と読めてしまい、
        // 実効条件の突き合わせが素通りする（PR #23 2回目レビュー指摘3）
        let req_map = |k: &str| -> Result<BTreeMap<String, String>, String> {
            let o = v
                .get(k)
                .ok_or_else(|| miss(k))?
                .as_object()
                .ok_or_else(|| format!("{where_}: {k} が object ではありません"))?;
            Ok(o.iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        v.as_str().map(|s| s.to_string()).unwrap_or_else(|| v.to_string()),
                    )
                })
                .collect())
        };
        // **キー自体は必須・値は null 可**の列（pair_with / balance_manifest）。
        // キーごと欠けた行を None に落とすと schema 契約の検査が素通りする
        let opt_s = |k: &str| -> Result<Option<String>, String> {
            let val = v.get(k).ok_or_else(|| miss(k))?;
            if val.is_null() {
                Ok(None)
            } else {
                val.as_str()
                    .map(|s| Some(s.to_string()))
                    .ok_or_else(|| format!("{where_}: {k} が文字列でも null でもありません"))
            }
        };
        rows.push(GameRow {
            candidate: req_s("candidate")?,
            baseline: req_s("baseline")?,
            match_seed: req_u("match_seed")?,
            match_seed_base: req_u("match_seed_base")?,
            match_seed_shard: req_u("match_seed_shard")?,
            run_id: req_s("run_id")?,
            run_attempt: req_u("run_attempt")?,
            pair_with: opt_s("pair_with")?,
            balance_manifest: opt_s("balance_manifest")?,
            game_no: req_u("game_no")?,
            a_is_sente: v
                .get("a_is_sente")
                .and_then(|x| x.as_bool())
                .ok_or_else(|| miss("a_is_sente"))?,
            score_a: req_f("score_a")?,
            reason: req_s("reason")?,
            plies: req_f("plies")?,
            fouls_a: req_f("fouls_a")?,
            fouls_b: req_f("fouls_b")?,
            fouls_in_check_a: req_f("fouls_in_check_a")?,
            think_ms_a: req_f("think_ms_a")?,
            think_ms_b: req_f("think_ms_b")?,
            moves_a: req_f("moves_a")?,
            commit: req_s("commit")?,
            cand_knobs: req_map("cand_knobs")?,
            clock: req_json("clock")?,
            budget: req_json("think_budget_ms_a")?,
            budget_opp: req_json("think_budget_ms_b")?,
            cand_config: req_json("cand_config")?,
            baseline_behavior: req_json("baseline_behavior")?,
            shared_env: req_map("shared_env")?,
        });
    }
    Ok(rows)
}

fn parse_game_rows(paths: &[String], arm: &str) -> Vec<GameRow> {
    let mut rows = vec![];
    for p in paths {
        let text = std::fs::read_to_string(p).unwrap_or_else(|e| die(&format!("{p}: {e}")));
        rows.extend(
            parse_game_rows_text(&text, &format!("{arm} {p}")).unwrap_or_else(|e| die(&e)),
        );
    }
    if rows.is_empty() {
        die(&format!("{arm}: 行が1件もありません"));
    }
    rows
}

/// 局ごとの指標（ペア差を取る前）。`METRICS` の arena 版
fn game_metrics(r: &GameRow) -> BTreeMap<&'static str, f64> {
    let mut m = BTreeMap::new();
    m.insert("score", r.score_a);
    m.insert("fouls_me", r.fouls_a);
    m.insert("fouls_in_check_me", r.fouls_in_check_a);
    m.insert(
        "foul_limit_loss",
        if r.reason == "foul_limit" && r.score_a == 0.0 { 1.0 } else { 0.0 },
    );
    m.insert("added_plies", r.plies);
    m.insert(
        "hit_max_plies",
        if r.reason == "max_plies" { 1.0 } else { 0.0 },
    );
    m.insert("fouls_opp", r.fouls_b);
    m.insert(
        "think_avg_ms_me",
        if r.moves_a > 0.0 { r.think_ms_a / r.moves_a } else { 0.0 },
    );
    m
}

/// arm 内で一意でなければならない属性を検査する
fn assert_uniform(arm: &str, rows: &[GameRow]) {
    let uniq = |name: &str, vals: BTreeSet<String>| {
        if vals.len() > 1 {
            die(&format!(
                "{arm}: {name} が混在しています（{}）。別条件の run を混ぜないでください",
                vals.into_iter().collect::<Vec<_>>().join(" / ")
            ));
        }
    };
    uniq("candidate", rows.iter().map(|r| r.candidate.clone()).collect());
    let baselines: BTreeSet<String> = rows.iter().map(|r| r.baseline.clone()).collect();
    if baselines.len() > 1 {
        die(&format!(
            "{arm}: 相手が混在しています（{}）。\n                         ガントレットの記録なら `--baseline <名前>` で1つに絞ってください\n                         （相手が違えば別のマッチアップなので、まとめてペアにはできません）",
            baselines.into_iter().collect::<Vec<_>>().join(" / ")
        ));
    }
    uniq("clock", rows.iter().map(|r| r.clock.clone()).collect());
    uniq("commit", rows.iter().map(|r| r.commit.clone()).collect());
    uniq("think_budget_ms_a", rows.iter().map(|r| r.budget.clone()).collect());
    uniq("think_budget_ms_b", rows.iter().map(|r| r.budget_opp.clone()).collect());
    uniq("cand_config", rows.iter().map(|r| r.cand_config.clone()).collect());
    uniq(
        "baseline_behavior",
        rows.iter().map(|r| r.baseline_behavior.clone()).collect(),
    );
    uniq("shared_env", rows.iter().map(|r| fmt_env(&r.shared_env)).collect());
    uniq(
        "cand_knobs",
        rows.iter().map(|r| fmt_env(&r.cand_knobs)).collect(),
    );
    let mut seen = BTreeSet::new();
    for r in rows {
        let key = (r.baseline.clone(), r.match_seed, r.game_no);
        if !seen.insert(key.clone()) {
            die(&format!(
                "{arm}: 同じ (baseline, match_seed, game_no) が2回あります: {key:?}\n            \
                 （同じ shard の artifact を二重に渡していませんか）"
            ));
        }
    }
}

fn cmd_arena_var(args: &Args) {
    let control_paths = args.all("control");
    let candidate_paths = args.all("candidate");
    if control_paths.is_empty() || candidate_paths.is_empty() {
        die("--control と --candidate に ARENA_GAMES_JSON の JSONL を指定してください");
    }
    let boot: usize = args.num("boot", 10000);
    let alpha: f64 = args.num("alpha", 0.05);
    let power: f64 = args.num("power", 0.80);
    let allow_incomplete = args.flag("allow-incomplete");
    let label = args.get("label").unwrap_or("arena-var").to_string();
    let z_alpha = z_two_sided(alpha);
    let z_beta = z_upper(1.0 - power);

    // ガントレットの記録（相手が複数）から1つのマッチアップだけを取り出す
    let only = args.get("baseline");
    let filter = |mut rows: Vec<GameRow>, arm: &str| -> Vec<GameRow> {
        if let Some(b) = only {
            rows.retain(|r| r.baseline == b);
            if rows.is_empty() {
                die(&format!("{arm}: 相手 {b} の対局がありません"));
            }
        }
        rows
    };
    let ctrl = filter(parse_game_rows(&control_paths, "control"), "control");
    let cand = filter(parse_game_rows(&candidate_paths, "candidate"), "candidate");
    assert_uniform("control", &ctrl);
    assert_uniform("candidate", &cand);

    // **同じ固定相手・同じ時計であること**。ここが違うと局面条件が揃わないので
    // ペア差にならない（checkpoint arena 側の opponent 指紋検査と同じ趣旨）
    if ctrl[0].baseline != cand[0].baseline {
        die(&format!(
            "相手が違います: control={} / candidate={}",
            ctrl[0].baseline, cand[0].baseline
        ));
    }
    if ctrl[0].clock != cand[0].clock {
        die(&format!(
            "時計が違います: control={} / candidate={}",
            ctrl[0].clock, cand[0].clock
        ));
    }
    // **固定相手が本当に同じか**は名前ではなく実効挙動の指紋で見る
    // （PR #22 / issue #21 で塞いだ「同名だが実効挙動が違う相手」の再発防止）。
    // 共通 env が違えば凍結相手の挙動も違うので、そちらも必須一致
    if ctrl[0].baseline_behavior != cand[0].baseline_behavior {
        die(&format!(
            "相手 {} の実効挙動が違います: control={} / candidate={}\n                         （名前が同じでも、共通 env・共有モデルの pin・実効予算が違えば別物です）",
            ctrl[0].baseline, ctrl[0].baseline_behavior, cand[0].baseline_behavior
        ));
    }
    if fmt_env(&ctrl[0].shared_env) != fmt_env(&cand[0].shared_env) {
        die(&format!(
            "両側に効く env が違います: control=[{}] / candidate=[{}]",
            fmt_env(&ctrl[0].shared_env),
            fmt_env(&cand[0].shared_env)
        ));
    }
    // **思考予算は必須一致**。ここを見ていなかったので、700ms の checkpoint と
    // 2000ms の arena を突き合わせる誤りを機械的に止められなかった
    // （PR #23 レビュー指摘1・2）。予算そのものを treatment にしたいときだけ
    // `--allow-budget-diff` で明示的に許可し、両方の値を出力へ残す
    let budget_diff = ctrl[0].budget != cand[0].budget || ctrl[0].budget_opp != cand[0].budget_opp;
    if budget_diff && !args.flag("allow-budget-diff") {
        die(&format!(
            "思考予算が違います: control=(候補 {} / 相手 {}) / candidate=(候補 {} / 相手 {})\n                         予算が違う2 run は別の estimand なので、そのままでは較正値に使えません。\n                         予算そのものを比べたいときだけ --allow-budget-diff（出力に両方の値を残します）",
            ctrl[0].budget, ctrl[0].budget_opp, cand[0].budget, cand[0].budget_opp
        ));
    }
    // **commit が違えば env アブレーションではない**（測っているのは
    // ノブの差 + revision の差）。P3 の base/target 比較では正常なので警告に留める
    if ctrl[0].commit != cand[0].commit {
        eprintln!(
            "警告: commit が違います（control={} / candidate={}）。\n                   ノブのアブレーションのつもりなら、同じ commit で取り直してください",
            ctrl[0].commit, cand[0].commit
        );
    }
    // **A/A かどうかは raw の指定でなく実効設定で判定する**（PR #23 2回目
    // レビュー指摘2）。名前・raw knobs・候補側予算だけを見ていた版は、
    // `cand_config` だけが違う2 run を「完全に同じ」と表示していた
    let same_effective = ctrl[0].candidate == cand[0].candidate
        && ctrl[0].cand_config == cand[0].cand_config
        && ctrl[0].budget == cand[0].budget
        && ctrl[0].budget_opp == cand[0].budget_opp
        && ctrl[0].commit == cand[0].commit;
    if same_effective {
        eprintln!(
            "注意: control と candidate の実効設定（config 指紋・予算・commit）が一致しています。\n                   A/A として読みます（差の期待値は 0）"
        );
    } else if fmt_env(&ctrl[0].cand_knobs) == fmt_env(&cand[0].cand_knobs)
        && ctrl[0].candidate == cand[0].candidate
    {
        // raw の指定が同じなのに実効設定が違う = commit か config 解決が違う
        eprintln!(
            "注意: raw の候補指定は同じですが実効設定が違います（config 指紋 {} vs {} / commit {} vs {}）",
            &ctrl[0].cand_config, &cand[0].cand_config, &ctrl[0].commit, &cand[0].commit
        );
    }

    let key = |r: &GameRow| (r.baseline.clone(), r.match_seed, r.game_no);
    let ctrl_by: BTreeMap<_, _> = ctrl.iter().map(|r| (key(r), r)).collect();
    let cand_by: BTreeMap<_, _> = cand.iter().map(|r| (key(r), r)).collect();
    let only_ctrl: Vec<_> = ctrl_by.keys().filter(|k| !cand_by.contains_key(*k)).collect();
    let only_cand: Vec<_> = cand_by.keys().filter(|k| !ctrl_by.contains_key(*k)).collect();
    if !only_ctrl.is_empty() || !only_cand.is_empty() {
        let msg = format!(
            "ペアにならない対局があります（control のみ {} 局 / candidate のみ {} 局）。\n            \
             同じ match_seed・同じ局数・同じ shard 構成で取り直してください",
            only_ctrl.len(),
            only_cand.len()
        );
        if allow_incomplete {
            eprintln!("警告: {msg}");
        } else {
            die(&msg);
        }
    }
    let keys: Vec<_> = ctrl_by
        .keys()
        .filter(|k| cand_by.contains_key(*k))
        .cloned()
        .collect();
    // **1ペアでは分散が同定できない**（PR #23 2回目レビュー指摘1）。
    // `variance` は n<2 で 0 を返すので、そのまま流すと Var=0 / SE=0 /
    // MDE=0 / CI=[観測値, 観測値] = 「完全に精密」と読める出力になる。
    // 自由度は `--allow-incomplete` では作れないので override 対象にしない
    if keys.len() < MIN_ARENA_PAIRS {
        die(&arena_var_pair_error(keys.len()));
    }
    // **先後が揃っているか**（game_no の偶奇で決まるので、揃わないなら
    // 別々の対局条件列を突き合わせている）
    for k in &keys {
        if ctrl_by[k].a_is_sente != cand_by[k].a_is_sente {
            die(&format!("{k:?}: 先後が食い違っています（別の条件列です）"));
        }
    }

    let n = keys.len();
    // 局ごとのペア差（cluster は無い = 1局1標本）
    let mut deltas: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    let mut arm_means: BTreeMap<&'static str, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for k in &keys {
        let mc = game_metrics(ctrl_by[k]);
        let mt = game_metrics(cand_by[k]);
        for (name, _) in METRICS.iter().filter(|(n, _)| mc.contains_key(n)) {
            let e = deltas.entry(name).or_default();
            e.push(mt[name] - mc[name]);
            let a = arm_means.entry(name).or_default();
            a.0.push(mc[name]);
            a.1.push(mt[name]);
        }
    }
    let score = &deltas["score"];
    let d_mean = mean(score);
    let var_paired = variance(score);
    let se = (var_paired / n as f64).sqrt();
    let (lo, hi) = bootstrap_ci_of_means(score, boot, 20260824, alpha);
    let mde_v = mde(se, z_alpha, z_beta);
    // ペアリングが効いているか: 独立に取ったときの Var(delta) と比べる
    let (ctrl_scores, cand_scores) = &arm_means["score"];
    let var_indep = variance(ctrl_scores) + variance(cand_scores);

    // CPU コスト。1ペア = 候補側1局 + 対照側1局。**思考時間の合計**で数える
    // （checkpoint 側の total_ms は壁時計なので厳密には同尺度ではない。
    //  思考が支配項なので比較には使えるが、注記つきで読むこと）
    let cpu_per_pair: f64 = keys
        .iter()
        .map(|k| {
            let c = ctrl_by[k];
            let t = cand_by[k];
            (c.think_ms_a + c.think_ms_b + t.think_ms_a + t.think_ms_b) / 1000.0
        })
        .sum::<f64>()
        / n as f64;
    let var_cpu = var_paired * cpu_per_pair;
    let var_cpu_indep = var_indep * cpu_per_pair;
    let var_cpu_assumed = 0.5 * cpu_per_pair;

    let mut out = String::new();
    out.push_str(&format!("## 通常 arena のペア差（{label}）\n\n"));
    out.push_str(&format!(
        "候補 `{}`{} / 対照 `{}`{} / 相手 `{}` / 時計 {} / 予算 {}\n\n",
        cand[0].candidate,
        if cand[0].cand_knobs.is_empty() {
            String::new()
        } else {
            format!("（{}）", fmt_env(&cand[0].cand_knobs))
        },
        ctrl[0].candidate,
        if ctrl[0].cand_knobs.is_empty() {
            String::new()
        } else {
            format!("（{}）", fmt_env(&ctrl[0].cand_knobs))
        },
        ctrl[0].baseline,
        ctrl[0].clock,
        if budget_diff {
            format!(
                "**不一致** control(候補 {} / 相手 {}) vs candidate(候補 {} / 相手 {})",
                ctrl[0].budget, ctrl[0].budget_opp, cand[0].budget, cand[0].budget_opp
            )
        } else {
            format!("候補 {} / 相手 {}", cand[0].budget, cand[0].budget_opp)
        },
    ));
    out.push_str(&format!(
        "相手の実効挙動 `{}` / 両側 env [{}]\n\n",
        &ctrl[0].baseline_behavior,
        fmt_env(&ctrl[0].shared_env)
    ));
    if budget_diff {
        out.push_str(
            "> **警告: 思考予算が違う2 run を突き合わせている**（`--allow-budget-diff`）。\
             これは別の estimand なので、checkpoint の `--known-arena-delta` には渡さないこと。\n\n",
        );
    }
    if ctrl[0].commit == cand[0].commit {
        out.push_str(&format!("commit `{}`\n\n", ctrl[0].commit));
    } else {
        out.push_str(&format!(
            "commit: control `{}` / candidate `{}`（**差分がエンジンに影響しないことは\
             ハッシュでは分からない**ので diff を見て確認すること）\n\n",
            ctrl[0].commit, cand[0].commit
        ));
    }
    out.push_str(&format!(
        "ペアになった対局 **{n} 局**（同じ `match_seed` の局ごとに突き合わせ）\n\n"
    ));
    out.push_str(&format!(
        "| 項目 | 値 |\n|---|---:|\n\
         | 候補のスコア率 | {} |\n\
         | 対照のスコア率 | {} |\n\
         | **ペア差（既知 arena 差）** | **{}** |\n\
         | {:.0}% CI | [{}, {}] |\n\
         | SE | ±{} |\n\
         | MDE（α={alpha:.2} / power={:.0}%） | ±{} |\n\
         | **Var(ペア差)** | **{var_paired:.4}** |\n\
         | 参考: 独立と仮定した Var | {var_indep:.4} |\n\
         | 参考: 従来の仮定 Var | 0.5000 |\n",
        rate_pct(mean(cand_scores)),
        rate_pct(mean(ctrl_scores)),
        pct(d_mean),
        (1.0 - alpha) * 100.0,
        pct(lo),
        pct(hi),
        width_pct(se),
        power * 100.0,
        width_pct(mde_v),
    ));
    out.push_str(&format!(
        "\n**同じ `match_seed` で局面条件を揃えるブロッキングの効き**: \
         Var(ペア差) {var_paired:.4} vs 独立 {var_indep:.4} = {:.2} 倍\
         （1.00 なら効いていない）。\n\n",
        if var_indep > 0.0 { var_paired / var_indep } else { f64::NAN }
    ));

    out.push_str("### 必要 N（この Var での MDE）\n\n");
    out.push_str("| 対局数 | 32 | 64 | 104 | 208 | 416 |\n|---|---|---|---|---|---|\n| MDE（±） |");
    for nn in [32usize, 64, 104, 208, 416] {
        out.push_str(&format!(
            " {} |",
            width_pct(mde((var_paired / nn as f64).sqrt(), z_alpha, z_beta))
        ));
    }
    out.push_str("\n| CI 半幅（±） |");
    for nn in [32usize, 64, 104, 208, 416] {
        out.push_str(&format!(
            " {} |",
            width_pct(z_alpha * (var_paired / nn as f64).sqrt())
        ));
    }
    out.push('\n');

    out.push_str("\n### CPU あたりの情報量\n\n");
    out.push_str(&format!(
        "1ペア（候補1局＋対照1局）あたり **{cpu_per_pair:.0} CPU秒**（思考時間の合計）\n\n\
         **var·CPU秒 = Var(ペア差) × 1ペアの CPU秒 = {var_cpu:.0}**\
         （ペアリング無しなら {var_cpu_indep:.0} / 従来の仮定 Var=0.5 なら {var_cpu_assumed:.0}）\n\n\
         checkpoint arena 側の `var·CPU秒` と直接比べる数字はこれ。\
         **ただし checkpoint 側の `total_ms` は壁時計、こちらは思考時間の合計**なので、\
         同尺度ではない（思考が支配項なので桁の比較には使えるが、\
         数%の差を読まないこと）。\n"
    ));

    out.push_str("\n### 安全性の共同指標（ペア差）\n\n");
    out.push_str("| 指標 | 対照 | 候補 | ペア差 | {} CI |\n|---|---:|---:|---:|---|\n");
    let ci_label = format!("{:.0}%", (1.0 - alpha) * 100.0);
    out = out.replace("{} CI", &format!("{ci_label} CI"));
    for (name, jp) in METRICS {
        let Some(d) = deltas.get(name) else { continue };
        let (c, t) = &arm_means[name];
        let (l, h) = bootstrap_ci_of_means(d, boot.min(4000), 20260824, alpha);
        out.push_str(&format!(
            "| {jp} | {:.3} | {:.3} | {:+.3} | [{:+.3}, {:+.3}] |\n",
            mean(c),
            mean(t),
            mean(d),
            l,
            h
        ));
    }
    out.push_str(
        "\n反則減だけで「強くなった」とは判定しない（反則は意図的な情報獲得にもなりうる）。\
         重大な悪化を止める用途に使う。\n\n\
         **水準は checkpoint 側と直接比べられない**: `継続手数` はここでは1局の総手数\
         （checkpoint では途中局面から先の手数）、反則もその差だけ長い区間の合計になる。\
         比べてよいのは**ペア差の向きと有意性**であって、絶対値ではない。\n",
    );

    println!("{out}");
    if let Some(p) = args.get("markdown") {
        std::fs::write(p, &out).unwrap_or_else(|e| die(&format!("{p}: {e}")));
    }
    if let Some(p) = args.get("json") {
        let v = serde_json::json!({
            "schema": SUMMARY_SCHEMA,
            "kind": "arena-var",
            "label": label,
            "candidate": cand[0].candidate,
            "control": ctrl[0].candidate,
            "candidate_knobs": cand[0].cand_knobs,
            "control_knobs": ctrl[0].cand_knobs,
            "opponent": ctrl[0].baseline,
            "control_commit": ctrl[0].commit,
            "candidate_commit": cand[0].commit,
            "clock": ctrl[0].clock,
            "think_budget_ms": cand[0].budget,
            "think_budget_ms_opponent": cand[0].budget_opp,
            "control_think_budget_ms": ctrl[0].budget,
            "control_think_budget_ms_opponent": ctrl[0].budget_opp,
            "budget_mismatch": budget_diff,
            "baseline_behavior": ctrl[0].baseline_behavior,
            "shared_env": ctrl[0].shared_env,
            "games": n,
            "alpha": alpha,
            "power": power,
            // **これを checkpoint arena の --known-arena-delta へ渡す**
            "arena_delta": d_mean,
            "ci_low": lo,
            "ci_high": hi,
            "se": se,
            "mde": mde_v,
            "var_paired": var_paired,
            "var_independent": var_indep,
            "cpu_sec_per_pair": cpu_per_pair,
            "var_cpu_sec": var_cpu,
            "metrics": METRICS.iter().filter(|(n,_)| deltas.contains_key(n)).map(|(n, _)| {
                (n.to_string(), serde_json::json!({
                    "control": mean(&arm_means[n].0),
                    "candidate": mean(&arm_means[n].1),
                    "delta": mean(&deltas[n]),
                }))
            }).collect::<serde_json::Map<_, _>>(),
        });
        std::fs::write(p, serde_json::to_string_pretty(&v).unwrap())
            .unwrap_or_else(|e| die(&format!("{p}: {e}")));
    }
}

// ---------------------------------------------------------------------------
// arena-balance: opponent-balanced 合算器（issue #40）
//
// 2相手ぶん（既定 v13 / v14）の対照・候補 arena-games.jsonl を読み、
// 相手ごとに局ペア差を作って `(Δv13 + Δv14) / 2` を**層化 bootstrap**
// （各相手の内側で局を引き直す）で判定する。`check_policy combined` と
// 同じ契約: 入力の同一性検査は fail-closed、判定は事前登録した門で、
// 不通過なら exit 3（`--allow-incomplete` で警告へ降格）。
// ---------------------------------------------------------------------------

/// issue #40 の held-out 判定の事前登録値（本文「検証 > Arena」の門）。
/// CLI から動かせない定数にする（門をノブにすると gate-shopping が成立する）
const BALANCE_MIN_DELTA: f64 = 0.04;
/// 安全性: 反則/局のペア差の上限
const BALANCE_MARGIN_FOULS: f64 = 0.3;
/// 安全性: 思考平均（ms/手）のペア差の上限
const BALANCE_MARGIN_THINK_MS: f64 = 100.0;
/// **事前登録した相手ごとの局数**（issue #40「held-out validation: 各600局。
/// N は事前固定し、門付近での増量・取り直しはしない」）。`--expect-games` が
/// この値でない入力は**判定不能**になり「通過」を出せない（別の N で回した
/// 標本に #40 の門を当てて通過を主張できてしまうため — PR #41 レビュー2巡目）
const BALANCE_EXPECT_GAMES: usize = 600;
/// 事前登録した shard 数（issue #40「held-out validation は shards=8」）
const BALANCE_EXPECT_SHARDS: usize = 8;
/// 事前登録した相手集合と base seed（issue #40「v14 は match_seed=20260909、
/// v13 は 20260910」）。相手集合の既定もここから作る
const BALANCE_EXPECT_SEEDS: [(&str, u64); 2] =
    [("estimator_v13", 20260910), ("estimator_v14", 20260909)];
/// 事前登録した信頼水準（95% CI）。`--alpha 0.5` の 50% CI で
/// 「CI 下限 > 0」を判定した結果は「通過」にしない
const BALANCE_ALPHA: f64 = 0.05;
/// 通過判定に使ってよい bootstrap 反復数の下限（既定値でもある）。
/// `--boot 1` は「1回引いた値」を CI と呼ぶだけで分解能が無い
const BALANCE_MIN_BOOT: usize = 10_000;

/// **事前登録との照合**（PR #41 レビュー4巡目 [P1]）。門の定数だけでなく、
/// 判定を変えられる呼び出し時パラメータ（局数・shard 数・base seed・相手集合・
/// 信頼水準・bootstrap 反復数）も事前登録値と比べ、外れたものは全部
/// **判定不能ノート**にする: informational な集計はできるが、別の seed・
/// shard 構成・相手・信頼水準で良く見えた結果を #40 の「通過」とは表示できない
fn balance_prereg_notes(
    expect_games: usize,
    expect_shards: usize,
    expect_seeds: &BTreeMap<String, u64>,
    expect_opps: &BTreeSet<String>,
    alpha: f64,
    boot: usize,
) -> Vec<String> {
    let mut notes = vec![];
    if expect_games != BALANCE_EXPECT_GAMES {
        notes.push(format!(
            "--expect-games {expect_games} ≠ 事前登録 {BALANCE_EXPECT_GAMES}（informational な集計はできるが #40 の採否は出せない）"
        ));
    }
    if expect_shards != BALANCE_EXPECT_SHARDS {
        notes.push(format!(
            "--expect-shards {expect_shards} ≠ 事前登録 {BALANCE_EXPECT_SHARDS}"
        ));
    }
    let want_seeds: BTreeMap<String, u64> = BALANCE_EXPECT_SEEDS
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();
    if *expect_seeds != want_seeds {
        notes.push(format!(
            "--expect-seeds {expect_seeds:?} ≠ 事前登録 {want_seeds:?}"
        ));
    }
    let want_opps: BTreeSet<String> = BALANCE_EXPECT_SEEDS
        .iter()
        .map(|(k, _)| k.to_string())
        .collect();
    if *expect_opps != want_opps {
        notes.push(format!(
            "--expect-opponents {expect_opps:?} ≠ 事前登録 {want_opps:?}"
        ));
    }
    if alpha != BALANCE_ALPHA {
        notes.push(format!(
            "--alpha {alpha} ≠ 事前登録 {BALANCE_ALPHA}（95% CI 以外で取った CI 下限は通過の根拠にしない）"
        ));
    }
    if boot < BALANCE_MIN_BOOT {
        notes.push(format!(
            "--boot {boot} < 下限 {BALANCE_MIN_BOOT}（CI の分解能が足りない）"
        ));
    }
    notes
}

/// validation manifest の指紋と行の指紋の照合。**指紋が違う行は Err**
/// （別の manifest の下で測った run か、計測後に manifest を書き換えている）。
/// **指紋の無い行（null）は本数を返す** = 呼び出し側が判定不能ノートへ回す
/// （「計測前に commit した manifest の下で測った」ことを機械検証できない）
fn check_manifest_rows(rows: &[GameRow], arm: &str, want_fp: &str) -> Result<usize, String> {
    let mut missing = 0usize;
    for r in rows {
        match &r.balance_manifest {
            None => missing += 1,
            Some(fp) if fp != want_fp => {
                return Err(format!(
                    "{arm}: 行の balance_manifest {fp} が --manifest の指紋 {want_fp} と一致しません\n            （別の manifest の下で測った run か、計測後に manifest を書き換えています）"
                ));
            }
            Some(_) => {}
        }
    }
    Ok(missing)
}

/// 最終判定: 入力が事前登録の条件を満たさない（判定不能の理由がある）とき、
/// 数値の門がどうであれ**「通過」を出さない**。返り値は (表示ラベル, pass)
fn balance_final(numeric_pass: bool, indeterminate: &[String]) -> (&'static str, bool) {
    if !indeterminate.is_empty() {
        ("判定不能", false)
    } else if numeric_pass {
        ("通過", true)
    } else {
        ("不通過", false)
    }
}

/// 層化 bootstrap: 各層（相手）の内側で標本を引き直し、層平均の平均を分布にする。
/// 返り値は (点推定, CI下限, CI上限)。点推定は各層の平均の単純平均 =
/// opponent-balanced な合算そのもの
fn stratified_boot_mean_ci(
    strata: &[Vec<f64>],
    boot: usize,
    seed: u64,
    alpha: f64,
) -> (f64, f64, f64) {
    if strata.iter().any(|v| v.is_empty()) {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let point = strata.iter().map(|v| mean(v)).sum::<f64>() / strata.len() as f64;
    let mut state = seed | 1;
    let mut draws: Vec<f64> = Vec::with_capacity(boot);
    for _ in 0..boot {
        let mut acc = 0.0;
        for v in strata {
            let mut s = 0.0;
            for _ in 0..v.len() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                s += v[(state >> 33) as usize % v.len()];
            }
            acc += s / v.len() as f64;
        }
        draws.push(acc / strata.len() as f64);
    }
    draws.sort_by(f64::total_cmp);
    let idx = |q: f64| -> f64 {
        let i = ((draws.len() - 1) as f64 * q).round() as usize;
        draws[i.min(draws.len() - 1)]
    };
    (point, idx(alpha / 2.0), idx(1.0 - alpha / 2.0))
}

/// 相手ごとに (対照, 候補) の局ペアを作る。ペアの鍵は (match_seed, game_no)
/// （実効 seed はシャードごとに違うので、shard をまたいだ衝突は起きない）。
/// - 相手集合が両 arm で一致しない・2相手未満は Err
/// - **arm × 相手ごとに `match_seed_base` は1つだけ**（常に Err、降格なし）。
///   実効 seed はシャードずらしで1 run でも複数値になる（PR #41 レビュー3巡目で
///   実効 seed の一意性検査を base へ付け替えた）が、base が2つある = 別々の
///   run（別の対局条件列）の混入で、「相手ごとに1つの未使用 seed」ではない
/// - **base seed が事前登録値と一致**（`expect_seeds`: 相手→base。不一致は Err）
/// - **shard 集合が 0..expect_shards の完全な集合**（欠け・重複・範囲外は Err。
///   実効 seed ＋ game_no のペア照合は末尾 shard の丸ごと欠落を素通しするため）
/// - **相手ごとのペア局数 == `expect_games`**（事前登録の各600局を呼び出し側が
///   明示する。`allow_incomplete` で警告へ降格するが、降格した事実は
///   返り値の**判定不能ノート**に残り、最終判定は「通過」を出せない）
/// - ペアにならない局は Err（`allow_incomplete` で捨てて続行 = 同じくノート行き）
/// - 先後の食い違いは常に Err（別の条件列）
/// - 相手ごとのペア数が [`MIN_ARENA_PAIRS`] 未満は Err（自由度は作れない）
///
/// 返り値は (相手→ペア列, 判定不能ノート)。ノートが空でない入力は
/// 「情報としての集計はできるが #40 の採否は出せない」
fn pair_by_opponent(
    ctrl: &[GameRow],
    cand: &[GameRow],
    expect_games: usize,
    expect_shards: usize,
    expect_seeds: &BTreeMap<String, u64>,
    allow_incomplete: bool,
) -> Result<(BTreeMap<String, Vec<(GameRow, GameRow)>>, Vec<String>), String> {
    let opps = |rows: &[GameRow]| -> BTreeSet<String> {
        rows.iter().map(|r| r.baseline.clone()).collect()
    };
    let (co, to) = (opps(ctrl), opps(cand));
    if co != to {
        return Err(format!(
            "相手集合が対照と候補で違います: control={co:?} / candidate={to:?}"
        ));
    }
    if co.len() < 2 {
        return Err(format!(
            "相手が {} 種類しかありません（opponent-balanced 合算には2相手以上が要ります。\n            1相手のペア差は arena-var を使ってください）",
            co.len()
        ));
    }
    let mut out: BTreeMap<String, Vec<(GameRow, GameRow)>> = BTreeMap::new();
    let mut notes: Vec<String> = vec![];
    for opp in &co {
        // run の一意性は **base seed** で検査する（実効 seed は shard ずらしで
        // 1 run でも複数値になるので、そちらを一意にすると正常な sharded run を
        // 必ず拒否する）。ここは降格しない: base が2つある = 別々の run の混入で、
        // 「未使用 seed 1本の held-out」という母集団の定義そのものが壊れる
        for (arm, rows) in [("control", ctrl), ("candidate", cand)] {
            let sub: Vec<&GameRow> = rows.iter().filter(|r| &r.baseline == opp).collect();
            let bases: BTreeSet<u64> = sub.iter().map(|r| r.match_seed_base).collect();
            if bases.len() > 1 {
                return Err(format!(
                    "{arm}({opp}): match_seed_base が複数あります（{bases:?}）。\n            複数の run の記録を混ぜないでください（相手ごとに1つの base seed の1 run）"
                ));
            }
            let base = *bases.iter().next().expect("相手集合の検査後なので空ではない");
            // 事前登録した相手別の base seed（held-out は v14=20260909 / v13=20260910）
            match expect_seeds.get(opp.as_str()) {
                None => {
                    return Err(format!(
                        "--expect-seeds に相手 {opp} の base seed がありません（\"相手=seed,...\" で全相手ぶん指定）"
                    ));
                }
                Some(&want) if want != base => {
                    return Err(format!(
                        "{arm}({opp}): match_seed_base {base} が事前登録 {want} と違います（別の run を渡していませんか）"
                    ));
                }
                Some(_) => {}
            }
            // shard 集合の完全性。実効 seed ＋ game_no のペア照合は「両 arm から
            // 同じ末尾 shard が丸ごと欠けた」を素通しするので、0..N の完全な
            // 集合であることをここで要求する
            let shards: BTreeSet<u64> = sub.iter().map(|r| r.match_seed_shard).collect();
            let want: BTreeSet<u64> = (0..expect_shards as u64).collect();
            if shards != want {
                return Err(format!(
                    "{arm}({opp}): shard 集合 {shards:?} が 0..{expect_shards} と一致しません（欠け・範囲外の shard があります）"
                ));
            }
            // **実行の識別子は arm × 相手で1つだけ**（PR #41 レビュー4巡目 [P1]、
            // 降格なし）。base は実験条件であって実行の識別子ではない: 同じ base で
            // 取り直した run A/B から shard を半分ずつ選んでも、base 1値・
            // shard 集合完全・局数一致・ペアキー重複なしでここまでの検査を全部
            // 通る。壁時計予算で同じ seed でも結果が揺れるので、それを許すのは
            // 「門付近での取り直し」を許すことと同じ
            let runs: BTreeSet<(&str, u64)> =
                sub.iter().map(|r| (r.run_id.as_str(), r.run_attempt)).collect();
            if runs.len() > 1 {
                return Err(format!(
                    "{arm}({opp}): 実行の識別子 (run_id, attempt) が複数あります（{runs:?}）。\n            同じ base seed でも別実行の記録は混ぜられません（取り直すなら run ごと差し替え）"
                ));
            }
        }
        // **対照の取り違え**: 候補行が `pair_with` で対照 run を指しているなら、
        // 実際に渡された対照の run_id と一致すること（`-f pair_with=` で対照を
        // 指して回した候補 run の記録がその紐を持っている）
        let ctrl_run = ctrl
            .iter()
            .find(|r| &r.baseline == opp)
            .map(|r| r.run_id.clone())
            .expect("相手集合の検査後なので空ではない");
        for r in cand.iter().filter(|r| &r.baseline == opp) {
            if let Some(pw) = &r.pair_with {
                if pw != &ctrl_run {
                    return Err(format!(
                        "candidate({opp}): pair_with={pw} が対照の run_id={ctrl_run} と一致しません\n            （候補 run が指した対照と、--control に渡した記録が別物です）"
                    ));
                }
            }
        }
        let key = |r: &GameRow| (r.match_seed, r.game_no);
        let cb: BTreeMap<_, _> = ctrl
            .iter()
            .filter(|r| &r.baseline == opp)
            .map(|r| (key(r), r))
            .collect();
        let tb: BTreeMap<_, _> = cand
            .iter()
            .filter(|r| &r.baseline == opp)
            .map(|r| (key(r), r))
            .collect();
        let unpaired =
            cb.keys().filter(|k| !tb.contains_key(*k)).count()
                + tb.keys().filter(|k| !cb.contains_key(*k)).count();
        if unpaired > 0 {
            let msg = format!(
                "{opp}: ペアにならない対局が {unpaired} 局あります。\n            同じ match_seed・同じ局数・同じ shard 構成で取り直してください"
            );
            if allow_incomplete {
                eprintln!("警告: {msg}");
                notes.push(format!("{opp}: ペアにならない対局 {unpaired} 局を捨てた"));
            } else {
                return Err(msg);
            }
        }
        let mut pairs: Vec<(GameRow, GameRow)> = vec![];
        for (k, c) in &cb {
            let Some(t) = tb.get(k) else { continue };
            if c.a_is_sente != t.a_is_sente {
                return Err(format!("{opp} {k:?}: 先後が食い違っています（別の条件列です）"));
            }
            pairs.push(((*c).clone(), (*t).clone()));
        }
        if pairs.len() < MIN_ARENA_PAIRS {
            return Err(format!("{opp}: {}", arena_var_pair_error(pairs.len())));
        }
        // **期待局数の強制**（fail-closed）: 事前登録した局数（検証セットは
        // 各600局）を呼び出し側が明示し、ここで照合する。「2ペア以上あればよい」
        // では 104局の smoke でも部分欠損した run でも判定を返せてしまう
        if pairs.len() != expect_games {
            let msg = format!(
                "{opp}: ペア局数 {} が期待 {expect_games} と違います（--expect-games は\n            事前登録した局数。検証セットは各600局）",
                pairs.len()
            );
            if allow_incomplete {
                eprintln!("警告: {msg}");
                notes.push(format!(
                    "{opp}: ペア局数 {} ≠ 期待 {expect_games}",
                    pairs.len()
                ));
            } else {
                return Err(msg);
            }
        }
        out.insert(opp.clone(), pairs);
    }
    Ok((out, notes))
}

/// 判定に使う量の入れ物（テスト対象）
#[derive(Clone)]
struct BalanceGate {
    combined: f64,
    ci_lo: f64,
    /// 相手ごとの Δ 点推定（符号 veto）
    per_opp: Vec<(String, f64)>,
    /// 反則/局（fouls_me）の合算ペア差
    fouls_delta: f64,
    /// 思考平均（ms/手）の合算ペア差
    think_delta: f64,
    /// 候補 arm の時間切れ負け局数
    cand_timeouts: u64,
}

/// 事前登録した門の判定。返り値は (通過, 落ちた理由の一覧)
fn balance_verdict(g: &BalanceGate) -> (bool, Vec<String>) {
    let mut reasons = vec![];
    if !(g.combined >= BALANCE_MIN_DELTA) {
        reasons.push(format!(
            "合算 Δ {:+.4} < +{BALANCE_MIN_DELTA}",
            g.combined
        ));
    }
    if !(g.ci_lo > 0.0) {
        reasons.push(format!("合算 CI 下限 {:+.4} ≤ 0", g.ci_lo));
    }
    for (opp, d) in &g.per_opp {
        if !(*d > 0.0) {
            reasons.push(format!("{opp} の Δ {d:+.4} ≤ 0（相手別符号 veto）"));
        }
    }
    if !(g.fouls_delta <= BALANCE_MARGIN_FOULS) {
        reasons.push(format!(
            "反則/局のペア差 {:+.3} > +{BALANCE_MARGIN_FOULS}",
            g.fouls_delta
        ));
    }
    if g.cand_timeouts > 0 {
        reasons.push(format!("候補の時間切れ負け {} 局 > 0", g.cand_timeouts));
    }
    if !(g.think_delta <= BALANCE_MARGIN_THINK_MS) {
        reasons.push(format!(
            "思考平均のペア差 {:+.1}ms > +{BALANCE_MARGIN_THINK_MS}ms",
            g.think_delta
        ));
    }
    (reasons.is_empty(), reasons)
}

fn cmd_arena_balance(args: &Args) {
    let control_paths = args.all("control");
    let candidate_paths = args.all("candidate");
    if control_paths.is_empty() || candidate_paths.is_empty() {
        die("--control と --candidate に ARENA_GAMES_JSON の JSONL を指定してください");
    }
    // **判定を変えられるパラメータの既定はすべて事前登録の定数**（PR #41
    // レビュー4巡目 [P1]）。CLI から動かすことはできる（他データの診断用）が、
    // 事前登録値から外れた指定は balance_prereg_notes が**判定不能**にする:
    // 別の N・shard 構成・seed・相手・信頼水準・反復数で良く見えた結果を
    // #40 の「通過」とは表示できない
    let boot: usize = args.num("boot", BALANCE_MIN_BOOT);
    let alpha: f64 = args.num("alpha", BALANCE_ALPHA);
    let allow_incomplete = args.flag("allow-incomplete");
    let label = args.get("label").unwrap_or("arena-balance").to_string();
    let parse_or_die = |flag: &str, v: &str| -> usize {
        v.parse()
            .unwrap_or_else(|_| die(&format!("--{flag} が数値ではありません: {v}")))
    };
    let expect_games: usize = match args.get("expect-games") {
        None => BALANCE_EXPECT_GAMES,
        Some(v) => parse_or_die("expect-games", v),
    };
    // shard 数と相手別 base seed の照合（PR #41 レビュー3巡目）。記録の
    // match_seed は shard ずらし＋基準 XOR 済みの実効値なので、run の同一性・
    // 完全性は base / shard 列で検査する
    let expect_shards: usize = match args.get("expect-shards") {
        None => BALANCE_EXPECT_SHARDS,
        Some(v) => parse_or_die("expect-shards", v),
    };
    let expect_seeds: BTreeMap<String, u64> = match args.get("expect-seeds") {
        None => BALANCE_EXPECT_SEEDS
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect(),
        Some(raw) => raw
            .split([',', ' '])
            .filter(|s| !s.is_empty())
            .map(|kv| match kv.split_once('=') {
                Some((k, v)) => match v.parse::<u64>() {
                    Ok(n) => (k.to_string(), n),
                    Err(_) => die(&format!("--expect-seeds の seed が数値ではありません: {kv}")),
                },
                None => die(&format!("--expect-seeds の項が 相手=seed 形式ではありません: {kv}")),
            })
            .collect(),
    };
    // **validation manifest**（PR #41 レビュー4巡目 [P1]）。処置ノブのような
    // 「P1 の後に決まる可変部分」は定数にできないので、**計測前に commit した
    // manifest ファイル**を一次資料にする: 期待ノブはここから読み、指紋
    // （ファイル bytes の sha256）を両 arm の全行（`ARENA_BALANCE_MANIFEST` が
    // 焼き込む）と照合する。--manifest 無しの集計は informational（判定不能）
    let manifest: Option<(String, serde_json::Value)> = args.get("manifest").map(|p| {
        let bytes = std::fs::read(p).unwrap_or_else(|e| die(&format!("--manifest {p}: {e}")));
        use sha2::Digest as _;
        let fp: String = sha2::Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let v: serde_json::Value = serde_json::from_slice(&bytes)
            .unwrap_or_else(|e| die(&format!("--manifest {p}: JSON として読めません: {e}")));
        (fp, v)
    });
    let manifest_knobs: Option<BTreeMap<String, String>> = manifest.as_ref().map(|(_, v)| {
        let o = v
            .get("cand_knobs")
            .and_then(|x| x.as_object())
            .unwrap_or_else(|| die("--manifest に cand_knobs（object）がありません"));
        o.iter()
            .map(|(k, val)| {
                (
                    k.clone(),
                    val.as_str().map(str::to_string).unwrap_or_else(|| val.to_string()),
                )
            })
            .collect()
    });
    // **候補側の処置ノブは事前登録値と完全一致**（PR #41 レビュー2巡目）。
    // 期待値は manifest（あれば）が一次資料。--expect-cand-knobs は manifest の
    // 無い診断用で、両方あれば一致を要求する（期待値の二重管理を許さない）
    let cli_knobs: Option<BTreeMap<String, String>> = args.get("expect-cand-knobs").map(|raw| {
        raw.split([',', ' '])
            .filter(|s| !s.is_empty())
            .map(|kv| match kv.split_once('=') {
                Some((k, v)) => (k.to_string(), v.to_string()),
                None => die(&format!("--expect-cand-knobs の項が K=V 形式ではありません: {kv}")),
            })
            .collect()
    });
    let expect_knobs: BTreeMap<String, String> = match (&manifest_knobs, &cli_knobs) {
        (Some(m), Some(c)) if m != c => die(&format!(
            "--expect-cand-knobs [{}] が --manifest の cand_knobs [{}] と一致しません",
            fmt_env(c),
            fmt_env(m)
        )),
        (Some(m), _) => m.clone(),
        (None, Some(c)) => c.clone(),
        (None, None) => die(
            "--manifest <path>（計測前に commit した validation manifest）か\n            --expect-cand-knobs \"K=V K=V\"（事前登録した処置ノブ）を指定してください",
        ),
    };
    // 期待する相手集合。**空にはできない**（片方の相手だけで通過を返せる穴と、
    // 「空文字で検査自体を無効化」の穴の両方を塞ぐ — PR #41 レビュー4巡目）
    let expect_opps: BTreeSet<String> = {
        let parsed: BTreeSet<String> = match args.get("expect-opponents") {
            None => BALANCE_EXPECT_SEEDS
                .iter()
                .map(|(k, _)| k.to_string())
                .collect(),
            Some(raw) => raw
                .split([',', ' '])
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect(),
        };
        if parsed.is_empty() {
            die("--expect-opponents は空にできません（相手集合の検査は無効化できない）");
        }
        parsed
    };

    let ctrl = parse_game_rows(&control_paths, "control");
    let cand = parse_game_rows(&candidate_paths, "candidate");

    // arm 内の一意性は**相手ごと**に検査する（assert_uniform は相手混在を
    // 拒否する設計なので、先に相手で割ってから掛ける）
    let split = |rows: &[GameRow]| -> BTreeMap<String, Vec<GameRow>> {
        let mut m: BTreeMap<String, Vec<GameRow>> = BTreeMap::new();
        for r in rows {
            m.entry(r.baseline.clone()).or_default().push(r.clone());
        }
        m
    };
    for (arm, rows) in [("control", &ctrl), ("candidate", &cand)] {
        for (opp, sub) in split(rows) {
            assert_uniform(&format!("{arm}({opp})"), &sub);
        }
        // **相手をまたいだ arm 設定の一致**（fail-closed）: baseline /
        // baseline_behavior / match_seed 以外はすべて一致していなければ、
        // 「2相手で同じ候補を測った」ことにならない
        let uniq_across = |name: &str, vals: BTreeSet<String>| {
            if vals.len() > 1 {
                die(&format!(
                    "{arm}: 相手をまたいで {name} が違います（{}）。同じ commit・同じ設定の run を渡してください",
                    vals.into_iter().collect::<Vec<_>>().join(" / ")
                ));
            }
        };
        uniq_across("candidate", rows.iter().map(|r| r.candidate.clone()).collect());
        uniq_across("clock", rows.iter().map(|r| r.clock.clone()).collect());
        uniq_across("commit", rows.iter().map(|r| r.commit.clone()).collect());
        uniq_across("think_budget_ms_a", rows.iter().map(|r| r.budget.clone()).collect());
        uniq_across("think_budget_ms_b", rows.iter().map(|r| r.budget_opp.clone()).collect());
        uniq_across("cand_config", rows.iter().map(|r| r.cand_config.clone()).collect());
        uniq_across("cand_knobs", rows.iter().map(|r| fmt_env(&r.cand_knobs)).collect());
        uniq_across("shared_env", rows.iter().map(|r| fmt_env(&r.shared_env)).collect());
    }

    // arm 間（対照 vs 候補）の実験条件。arena-var と同じ趣旨で fail-closed。
    // **予算の不一致に override は無い**（issue #40 の検証は予算 2000ms を
    // 4 run すべてで一致させる契約。予算そのものを比べる用途は arena-var の領分）
    //
    // **戦略名は同一**であること: issue #40 の比較は「同じ candidate 戦略の
    // W=0 vs 処置ノブ」であって、別戦略どうしの比較にこの門を使ってはいけない
    if ctrl[0].candidate != cand[0].candidate {
        die(&format!(
            "candidate 戦略が違います: control={} / candidate={}\n            （issue #40 の対照は同一戦略の W=0。別戦略の比較は arena-var の領分）",
            ctrl[0].candidate, cand[0].candidate
        ));
    }
    // **対照は W=0**（候補ノブなし = 既定 config）であること。対照側にノブが
    // 入っていると「処置ノブだけが違う」という比較の前提が崩れる
    if !ctrl[0].cand_knobs.is_empty() {
        die(&format!(
            "control に候補ノブが入っています（{}）。\n            issue #40 の対照は同一 commit・W=0（cand_env なし）の run です",
            fmt_env(&ctrl[0].cand_knobs)
        ));
    }
    if ctrl[0].clock != cand[0].clock {
        die(&format!(
            "時計が違います: control={} / candidate={}",
            ctrl[0].clock, cand[0].clock
        ));
    }
    if ctrl[0].budget != cand[0].budget || ctrl[0].budget_opp != cand[0].budget_opp {
        die(&format!(
            "思考予算が違います: control=(候補 {} / 相手 {}) / candidate=(候補 {} / 相手 {})",
            ctrl[0].budget, ctrl[0].budget_opp, cand[0].budget, cand[0].budget_opp
        ));
    }
    if fmt_env(&ctrl[0].shared_env) != fmt_env(&cand[0].shared_env) {
        die(&format!(
            "両側に効く env が違います: control=[{}] / candidate=[{}]",
            fmt_env(&ctrl[0].shared_env),
            fmt_env(&cand[0].shared_env)
        ));
    }
    if ctrl[0].commit != cand[0].commit {
        // issue #40 の対照は「同一 commit・W=0」。別 commit の main を指すと
        // arena-var は警告するだけなので、こちらは**常に止める**
        // （`--allow-incomplete` でも降格しない。別 commit の比較は
        // 「ノブの効果 + revision の差」で estimand が別物になる）
        die(&format!(
            "commit が違います: control={} / candidate={}。\n            issue #40 の対照は同一 commit・W=0 の run（別 commit の main は対照にできません。\n            この検査に override はありません）",
            ctrl[0].commit, cand[0].commit
        ));
    }
    // 相手の実効挙動は**相手ごとに** arm 間で一致していること
    {
        let by = |rows: &[GameRow]| -> BTreeMap<String, String> {
            rows.iter()
                .map(|r| (r.baseline.clone(), r.baseline_behavior.clone()))
                .collect()
        };
        let (cb, tb) = (by(&ctrl), by(&cand));
        for (opp, beh) in &cb {
            if tb.get(opp).is_some_and(|b| b != beh) {
                die(&format!(
                    "相手 {opp} の実効挙動が対照と候補で違います:\n            control={beh}\n            candidate={}",
                    tb[opp]
                ));
            }
        }
    }
    // 候補側の処置ノブは**事前登録値と完全一致**。違えば入力の取り違えなので die
    if fmt_env(&cand[0].cand_knobs) != fmt_env(&expect_knobs) {
        die(&format!(
            "candidate の処置ノブが事前登録と違います:\n            記録 [{}]\n            期待 [{}]",
            fmt_env(&cand[0].cand_knobs),
            fmt_env(&expect_knobs)
        ));
    }

    // **判定不能の理由**。1つでもあれば数値の門がどうであれ「通過」を出さない
    // （A/A・不完全入力・事前登録外のパラメータでの「通過」表示を塞ぐ）
    let mut indeterminate: Vec<String> =
        balance_prereg_notes(expect_games, expect_shards, &expect_seeds, &expect_opps, alpha, boot);
    if cand[0].cand_knobs.is_empty() || ctrl[0].cand_config == cand[0].cand_config {
        indeterminate.push(
            "候補と対照の実効設定が同一（A/A）= 処置が入っていないので採否の対象がない".into(),
        );
    }
    // manifest の照合: 指紋が違う行は die（別の manifest の下で測った run）、
    // 指紋の無い行と --manifest 未指定は判定不能。manifest が echo している
    // 既知の事前登録値（あれば）も定数と突き合わせる
    match &manifest {
        None => indeterminate.push(
            "--manifest 未指定 = 処置ノブが計測前に commit されたことを機械検証できない".into(),
        ),
        Some((fp, v)) => {
            for (arm, rows) in [("control", &ctrl), ("candidate", &cand)] {
                match check_manifest_rows(rows, arm, fp) {
                    Err(e) => die(&e),
                    Ok(0) => {}
                    Ok(n) => indeterminate.push(format!(
                        "{arm}: manifest 指紋の無い行が {n} 行（ARENA_BALANCE_MANIFEST 無しで測った run）"
                    )),
                }
            }
            if v.get("games").and_then(|x| x.as_u64()).is_some_and(|g| g != BALANCE_EXPECT_GAMES as u64) {
                indeterminate.push(format!(
                    "manifest の games {} ≠ 事前登録 {BALANCE_EXPECT_GAMES}",
                    v["games"]
                ));
            }
            if v.get("shards").and_then(|x| x.as_u64()).is_some_and(|s| s != BALANCE_EXPECT_SHARDS as u64) {
                indeterminate.push(format!(
                    "manifest の shards {} ≠ 事前登録 {BALANCE_EXPECT_SHARDS}",
                    v["shards"]
                ));
            }
            if let Some(o) = v.get("opponents").and_then(|x| x.as_object()) {
                let got: BTreeMap<String, u64> = o
                    .iter()
                    .filter_map(|(k, val)| val.as_u64().map(|n| (k.clone(), n)))
                    .collect();
                let want: BTreeMap<String, u64> = BALANCE_EXPECT_SEEDS
                    .iter()
                    .map(|(k, s)| (k.to_string(), *s))
                    .collect();
                if got != want {
                    indeterminate.push(format!(
                        "manifest の opponents {got:?} ≠ 事前登録 {want:?}"
                    ));
                }
            }
        }
    }

    let (paired, pairing_notes) = pair_by_opponent(
        &ctrl,
        &cand,
        expect_games,
        expect_shards,
        &expect_seeds,
        allow_incomplete,
    )
    .unwrap_or_else(|e| die(&e));
    indeterminate.extend(pairing_notes);
    let opps: Vec<String> = paired.keys().cloned().collect();
    if !expect_opps.is_empty() {
        let got: BTreeSet<String> = opps.iter().cloned().collect();
        if got != expect_opps {
            die(&format!(
                "相手集合が期待と違います: got={got:?} / expect={expect_opps:?}\n            （--expect-opponents で明示するか、空文字で検査を無効化）"
            ));
        }
    }

    // 相手ごと・指標ごとのペア差
    let mut strata_by_metric: BTreeMap<&'static str, Vec<Vec<f64>>> = BTreeMap::new();
    let mut arm_means: BTreeMap<&'static str, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    let mut cand_timeouts = 0u64;
    let mut ctrl_timeouts = 0u64;
    for opp in &opps {
        let pairs = &paired[opp];
        cand_timeouts += pairs
            .iter()
            .filter(|(_, t)| t.reason == "timeout" && t.score_a == 0.0)
            .count() as u64;
        ctrl_timeouts += pairs
            .iter()
            .filter(|(c, _)| c.reason == "timeout" && c.score_a == 0.0)
            .count() as u64;
        for (name, _) in METRICS {
            let mut deltas = vec![];
            for (c, t) in pairs {
                let (mc, mt) = (game_metrics(c), game_metrics(t));
                let (Some(a), Some(b)) = (mc.get(name), mt.get(name)) else { continue };
                deltas.push(b - a);
                let e = arm_means.entry(name).or_default();
                e.0.push(*a);
                e.1.push(*b);
            }
            if !deltas.is_empty() {
                strata_by_metric.entry(name).or_default().push(deltas);
            }
        }
    }

    let score_strata = &strata_by_metric["score"];
    let (combined, ci_lo, ci_hi) = stratified_boot_mean_ci(score_strata, boot, 0x40_2001, alpha);
    let per_opp: Vec<(String, f64)> = opps
        .iter()
        .zip(score_strata.iter())
        .map(|(o, v)| (o.clone(), mean(v)))
        .collect();
    let comb_of = |name: &str| -> f64 {
        strata_by_metric[name].iter().map(|v| mean(v)).sum::<f64>()
            / strata_by_metric[name].len() as f64
    };
    let gate = BalanceGate {
        combined,
        ci_lo,
        per_opp: per_opp.clone(),
        fouls_delta: comb_of("fouls_me"),
        think_delta: comb_of("think_avg_ms_me"),
        cand_timeouts,
    };
    let (numeric_pass, reasons) = balance_verdict(&gate);
    let (verdict_label, pass) = balance_final(numeric_pass, &indeterminate);

    let mut out = String::new();
    out.push_str(&format!("## opponent-balanced 合算（{label}、issue #40）\n\n"));
    out.push_str(&format!(
        "候補 `{}`{} / 対照 `{}`{} / commit `{}` / 時計 {} / 予算 候補 {} / 相手 {}\n\n",
        cand[0].candidate,
        if cand[0].cand_knobs.is_empty() {
            String::new()
        } else {
            format!("（{}）", fmt_env(&cand[0].cand_knobs))
        },
        ctrl[0].candidate,
        if ctrl[0].cand_knobs.is_empty() {
            String::new()
        } else {
            format!("（{}）", fmt_env(&ctrl[0].cand_knobs))
        },
        ctrl[0].commit,
        ctrl[0].clock,
        cand[0].budget,
        cand[0].budget_opp,
    ));
    out.push_str("| 相手 | ペア局数 | Δ（候補 − 対照） |\n|---|---:|---:|\n");
    for (opp, d) in &per_opp {
        out.push_str(&format!("| {} | {} | {} |\n", opp, paired[opp].len(), pct(*d)));
    }
    out.push_str(&format!(
        "\n**合算 (ΣΔ)/{} = {}**、{:.0}% CI [{}, {}]（層化 bootstrap {boot} 反復。各相手の内側で局を引き直す）\n\n",
        opps.len(),
        pct(combined),
        (1.0 - alpha) * 100.0,
        pct(ci_lo),
        pct(ci_hi),
    ));
    out.push_str("### 安全性・受動化監査（合算ペア差）\n\n");
    out.push_str("| 指標 | 対照 | 候補 | 合算ペア差 | CI |\n|---|---:|---:|---:|---|\n");
    for (name, jp) in METRICS {
        let Some(strata) = strata_by_metric.get(name) else { continue };
        let (d, l, h) = stratified_boot_mean_ci(strata, boot.min(4000), 0x40_2002, alpha);
        let (c, t) = &arm_means[name];
        out.push_str(&format!(
            "| {jp} | {:.3} | {:.3} | {:+.3} | [{:+.3}, {:+.3}] |\n",
            mean(c),
            mean(t),
            d,
            l,
            h
        ));
    }
    out.push_str(&format!(
        "\n時間切れ負け: 対照 {ctrl_timeouts} 局 / 候補 {cand_timeouts} 局。\n\
         持ち駒残数・王手率・詰み勝ち率と**平均評価粒子数**（対照比 −10% 超で中止）は\n\
         games.jsonl に無いので arena-records（`chose.debug`）から別途出すこと。\n\n"
    ));
    out.push_str(&format!(
        "### 判定（事前登録: 各{BALANCE_EXPECT_GAMES}局・合算 ≥ +{BALANCE_MIN_DELTA} かつ CI 下限 > 0・相手別符号・反則/局 +{BALANCE_MARGIN_FOULS} 以内・時間切れ0・思考平均 +{BALANCE_MARGIN_THINK_MS}ms 以内）\n\n"
    ));
    match verdict_label {
        "通過" => out.push_str("**通過**（v9〜v14 ガントレットへ進める）\n"),
        "判定不能" => {
            out.push_str(
                "**判定不能**（入力が事前登録の条件を満たさない。数値の門がどうであれ通過は出せない）:\n\n",
            );
            for r in &indeterminate {
                out.push_str(&format!("- {r}\n"));
            }
            if !numeric_pass {
                out.push_str("\n参考: 数値の門も次の理由で不通過:\n\n");
                for r in &reasons {
                    out.push_str(&format!("- {r}\n"));
                }
            }
        }
        _ => {
            out.push_str("**不通過**:\n\n");
            for r in &reasons {
                out.push_str(&format!("- {r}\n"));
            }
        }
    }

    println!("{out}");
    if let Some(p) = args.get("markdown") {
        std::fs::write(p, &out).unwrap_or_else(|e| die(&format!("{p}: {e}")));
    }
    if let Some(p) = args.get("json") {
        let v = serde_json::json!({
            "schema": SUMMARY_SCHEMA,
            "kind": "arena-balance",
            "label": label,
            "candidate": cand[0].candidate,
            "control": ctrl[0].candidate,
            "candidate_knobs": cand[0].cand_knobs,
            "control_knobs": ctrl[0].cand_knobs,
            "commit": ctrl[0].commit,
            "clock": ctrl[0].clock,
            "think_budget_ms": cand[0].budget,
            "think_budget_ms_opponent": cand[0].budget_opp,
            "alpha": alpha,
            "boot": boot,
            "expect_games": expect_games,
            "expect_shards": expect_shards,
            "expect_seeds": expect_seeds,
            "manifest_fingerprint": manifest.as_ref().map(|(fp, _)| fp.clone()),
            "opponents": per_opp.iter().map(|(o, d)| {
                let (c0, t0) = &paired[o][0];
                (o.clone(), serde_json::json!({
                    "pairs": paired[o].len(),
                    "delta": d,
                    // 監査用: どの実行の記録で判定したか
                    "control_run": format!("{}#{}", c0.run_id, c0.run_attempt),
                    "candidate_run": format!("{}#{}", t0.run_id, t0.run_attempt),
                }))
            }).collect::<serde_json::Map<_, _>>(),
            "combined_delta": combined,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "fouls_delta": gate.fouls_delta,
            "think_avg_ms_delta": gate.think_delta,
            "candidate_timeouts": cand_timeouts,
            "verdict": verdict_label,
            "pass": pass,
            "fail_reasons": reasons,
            "indeterminate_reasons": indeterminate,
            "expect_cand_knobs": expect_knobs,
            "metrics": METRICS.iter().filter(|(n, _)| strata_by_metric.contains_key(n)).map(|(n, _)| {
                (n.to_string(), serde_json::json!({
                    "control": mean(&arm_means[n].0),
                    "candidate": mean(&arm_means[n].1),
                    "delta": comb_of(n),
                }))
            }).collect::<serde_json::Map<_, _>>(),
        });
        std::fs::write(p, serde_json::to_string_pretty(&v).unwrap())
            .unwrap_or_else(|e| die(&format!("{p}: {e}")));
    }
    if !pass && !allow_incomplete {
        // fail-closed: 判定が通らない実験を緑で終わらせない（check_policy combined と同じ）
        std::process::exit(3);
    }
}

/// 標本分散（n−1）
fn variance(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v);
    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.is_empty() {
        die("サブコマンドを指定してください");
    }
    let args = Args::parse(&raw);
    match args.positional.first().map(String::as_str) {
        Some("extract") => cmd_extract(&args),
        Some("run") => cmd_run(&args),
        Some("unit") => cmd_unit(&args),
        Some("pair") => cmd_pair(&args),
        Some("compare") => cmd_compare(&args),
        Some("report") => cmd_report(&args),
        Some("arena-var") => cmd_arena_var(&args),
        Some("arena-balance") => cmd_arena_balance(&args),
        other => die(&format!("未知のサブコマンド: {}", other.unwrap_or("(なし)"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 正規分位点の近似（検出力計算の土台）
    #[test]
    fn normal_quantiles_are_accurate() {
        assert!((z_two_sided(0.05) - 1.959_964).abs() < 1e-4);
        assert!((z_upper(0.20) - 0.841_621).abs() < 1e-4);
        assert!((z_upper(0.025) - 1.959_964).abs() < 1e-4);
        assert!((z_upper(0.5) - 0.0).abs() < 1e-6);
    }

    /// **arena-var の summary を横断レポートへ混ぜない**（schema は同じでも
    /// 契約が違うので、通すと全列 NaN の行が並ぶ）
    #[test]
    fn report_rejects_foreign_summary_kinds() {
        let ck = serde_json::json!({ "schema": SUMMARY_SCHEMA, "kind": "checkpoint-compare" });
        let av = serde_json::json!({ "schema": SUMMARY_SCHEMA, "kind": "arena-var" });
        let legacy = serde_json::json!({ "schema": SUMMARY_SCHEMA });
        let accepted = |v: &serde_json::Value| {
            check_summary_schema(v, "x").is_ok()
                && v["kind"].as_str().is_none_or(|k| k == "checkpoint-compare")
        };
        assert!(accepted(&ck));
        assert!(accepted(&legacy), "kind が無い過去の summary は通す");
        assert!(!accepted(&av));
    }

    /// **実効条件の指紋が game row に必須**であること（PR #23 レビュー指摘2）。
    /// 名前と時計しか見ていないと、同じ `estimator_v14` でも共通 env・共有モデル pin・
    /// 予算が違う2 run を「ペア」として受理してしまう
    #[test]
    fn game_row_schema_requires_effective_conditions() {
        assert_eq!(
            GAME_ROW_SCHEMA, 4,
            "指紋（schema 2）・base/shard（schema 3）・実行の識別子（schema 4）を必須にした時点で schema を上げる"
        );
        let r = game_row("checkmate", 1.0);
        // 突き合わせの前提になる項目が型として存在すること（欠損は parse で die する）
        assert!(!r.baseline_behavior.is_empty());
        assert!(!r.cand_config.is_empty());
        assert!(!r.budget.is_empty() && !r.budget_opp.is_empty());
    }

    /// **予算が違えば別の estimand**。片側だけ見ていると 700ms と 2000ms が通る
    #[test]
    fn budget_mismatch_is_detected_on_both_sides() {
        let a = game_row("checkmate", 1.0);
        let mut b = game_row("checkmate", 1.0);
        assert!(!(a.budget != b.budget || a.budget_opp != b.budget_opp));
        b.budget = "700".into();
        assert!(a.budget != b.budget || a.budget_opp != b.budget_opp);
        let mut c = game_row("checkmate", 1.0);
        // 相手側だけ違う場合も捕まえる（候補側の予算だけ見ていると素通りする）
        c.budget_opp = "700".into();
        assert!(a.budget != c.budget || a.budget_opp != c.budget_opp);
    }

    /// arena-var の土台。`variance` は n−1 の標本分散
    #[test]
    fn sample_variance_matches_hand_calc() {
        assert!((variance(&[1.0, 0.0, 1.0, 0.0]) - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(variance(&[0.5]), 0.0);
        assert_eq!(variance(&[]), 0.0);
        // 勝1/負0 が半々なら Var = 0.5·(n/(n−1))。n→∞ で 0.25
        assert!((variance(&[1.0, 0.0]) - 0.5).abs() < 1e-12);
    }

    /// 同一入力どうしのペア差は厳密に 0（A/A の健全性）
    #[test]
    fn identical_arms_give_zero_paired_delta() {
        let scores = [1.0f64, 0.0, 0.5, 1.0, 0.0];
        let deltas: Vec<f64> = scores.iter().map(|s| s - s).collect();
        assert_eq!(mean(&deltas), 0.0);
        assert_eq!(variance(&deltas), 0.0);
    }

    fn game_row(reason: &str, score_a: f64) -> GameRow {
        GameRow {
            candidate: "estimator".into(),
            baseline: "estimator_v14".into(),
            match_seed: 1,
            match_seed_base: 1,
            match_seed_shard: 0,
            run_id: "run-1".into(),
            run_attempt: 1,
            pair_with: None,
            balance_manifest: None,
            game_no: 0,
            a_is_sente: true,
            score_a,
            reason: reason.into(),
            plies: 100.0,
            fouls_a: 6.0,
            fouls_b: 5.0,
            fouls_in_check_a: 1.0,
            think_ms_a: 40_000.0,
            think_ms_b: 38_000.0,
            moves_a: 40.0,
            commit: "deadbeef".into(),
            cand_knobs: BTreeMap::new(),
            clock: "[1000000,3000]".into(),
            budget: "2000".into(),
            budget_opp: "2000".into(),
            cand_config: "cfg".into(),
            baseline_behavior: "beh".into(),
            shared_env: BTreeMap::new(),
        }
    }

    fn balance_row(baseline: &str, game_no: u64, score_a: f64) -> GameRow {
        let mut r = game_row("checkmate", score_a);
        r.baseline = baseline.into();
        r.game_no = game_no;
        r.a_is_sente = game_no % 2 == 0;
        r
    }

    /// arena-balance のペアリング: 相手ごとに閉じ、集合の不一致・欠落・
    /// 1相手だけ・先後の食い違い・base seed / shard 集合のずれは fail-closed
    #[test]
    fn arena_balance_pairs_within_each_opponent() {
        let seeds: BTreeMap<String, u64> =
            [("estimator_v13".to_string(), 1u64), ("estimator_v14".to_string(), 1u64)]
                .into_iter()
                .collect();
        let ctrl: Vec<GameRow> = vec![
            balance_row("estimator_v13", 0, 0.0),
            balance_row("estimator_v13", 1, 1.0),
            balance_row("estimator_v14", 0, 0.5),
            balance_row("estimator_v14", 1, 0.5),
        ];
        let cand = ctrl.clone(); // A/A
        let (paired, notes) =
            pair_by_opponent(&ctrl, &cand, 2, 1, &seeds, false).expect("A/A はペアになる");
        assert_eq!(paired.len(), 2);
        assert!(paired.values().all(|v| v.len() == 2));
        assert!(notes.is_empty(), "完全な入力にノートは付かない");

        // **sharded run（同じ base・複数の実効 seed）は受理する**（PR #41
        // レビュー3巡目: 実効 seed の一意性検査は正常な shards=4 の run を
        // 必ず拒否していた。run の一意性は base で見る）
        let mut sh_ctrl = ctrl.clone();
        let mut sh_cand = cand.clone();
        for rows in [&mut sh_ctrl, &mut sh_cand] {
            for opp in ["estimator_v13", "estimator_v14"] {
                for g in 0..2u64 {
                    let mut r = balance_row(opp, g, 0.5);
                    r.match_seed = 1001; // shard ずらし後の別の実効 seed
                    r.match_seed_shard = 1;
                    rows.push(r);
                }
            }
        }
        let (paired, notes) =
            pair_by_opponent(&sh_ctrl, &sh_cand, 4, 2, &seeds, false).expect("sharded run は正常");
        assert!(paired.values().all(|v| v.len() == 4));
        assert!(notes.is_empty());
        // shard が欠けたら Err（末尾 shard の丸ごと欠落はペア照合を素通しする）
        assert!(pair_by_opponent(&ctrl, &cand, 2, 2, &seeds, false).is_err());

        // **期待局数は明示必須で照合される**（事前登録の各600局の強制。
        // 「2ペア以上あればよい」では smoke や欠損 run でも判定を返せてしまう）
        assert!(pair_by_opponent(&ctrl, &cand, 600, 1, &seeds, false).is_err());
        // --allow-incomplete では通るが**判定不能ノートが残る** = 「通過」を出せない
        let (_, notes) =
            pair_by_opponent(&ctrl, &cand, 600, 1, &seeds, true).expect("警告へ降格");
        assert!(!notes.is_empty());
        assert!(!balance_final(true, &notes).1, "ノート付き入力は数値の門が通っても pass しない");

        // **base seed は事前登録値との完全一致**（不一致・未登録の相手は Err）
        let wrong_seed: BTreeMap<String, u64> =
            [("estimator_v13".to_string(), 2u64), ("estimator_v14".to_string(), 1u64)]
                .into_iter()
                .collect();
        assert!(pair_by_opponent(&ctrl, &cand, 2, 1, &wrong_seed, false).is_err());
        let missing_seed: BTreeMap<String, u64> =
            [("estimator_v14".to_string(), 1u64)].into_iter().collect();
        assert!(pair_by_opponent(&ctrl, &cand, 2, 1, &missing_seed, false).is_err());

        // 1相手だけでは opponent-balanced にならない（arena-var の領分）
        let one: Vec<GameRow> = ctrl
            .iter()
            .filter(|r| r.baseline == "estimator_v13")
            .cloned()
            .collect();
        assert!(pair_by_opponent(&one, &one, 2, 1, &seeds, false).is_err());

        // 片 arm に無い局は fail-closed
        let mut missing = cand.clone();
        missing.pop();
        assert!(pair_by_opponent(&ctrl, &missing, 2, 1, &seeds, false).is_err());

        // 相手集合の不一致も fail-closed
        let mut wrong = cand.clone();
        wrong[3].baseline = "estimator_v12".into();
        assert!(pair_by_opponent(&ctrl, &wrong, 2, 1, &seeds, false).is_err());

        // 先後の食い違い = 別の条件列
        let mut flipped = cand.clone();
        flipped[0].a_is_sente = !flipped[0].a_is_sente;
        assert!(pair_by_opponent(&ctrl, &flipped, 2, 1, &seeds, false).is_err());

        // **同じ相手に複数の base seed を混ぜたら常に Err**（降格なし）。
        // 両 arm が同じ2つの base を持てばペア自体は成立してしまうので、
        // ペアリングとは別にここで止める必要がある
        let mut mixed_ctrl = ctrl.clone();
        let mut mixed_cand = cand.clone();
        for rows in [&mut mixed_ctrl, &mut mixed_cand] {
            for opp in ["estimator_v13", "estimator_v14"] {
                let mut extra = balance_row(opp, 2, 0.5);
                extra.match_seed = 999;
                extra.match_seed_base = 999;
                rows.push(extra);
            }
        }
        assert!(
            pair_by_opponent(&mixed_ctrl, &mixed_cand, 3, 1, &seeds, false).is_err(),
            "base 混在は期待局数が合っていても弾く"
        );
        assert!(
            pair_by_opponent(&mixed_ctrl, &mixed_cand, 3, 1, &seeds, true).is_err(),
            "--allow-incomplete でも base 混在は降格しない"
        );
    }

    /// **同じ base seed の複数 run の混入を実行の識別子で止める**（PR #41
    /// レビュー4巡目 [P1] の再現）: 同一 commit・同一設定・同一 base で取り直した
    /// run A/B から shard を半分ずつ選ぶと、base 1値・shard 集合 0..N 完全・
    /// 局数一致・ペアキー重複なしで従来の検査を全部通っていた
    #[test]
    fn arena_balance_rejects_shards_mixed_from_two_runs_with_same_base() {
        let seeds: BTreeMap<String, u64> =
            [("estimator_v13".to_string(), 1u64), ("estimator_v14".to_string(), 1u64)]
                .into_iter()
                .collect();
        // shard 0 = 実効 seed 1 / shard 1 = 実効 seed 1001（どちらも base 1）
        let mk = |opp: &str, run: &str, attempt: u64, shard: u64, g: u64| {
            let mut r = balance_row(opp, g, 0.5);
            r.run_id = run.into();
            r.run_attempt = attempt;
            r.match_seed_shard = shard;
            r.match_seed = if shard == 0 { 1 } else { 1001 };
            r
        };
        let arm = |run0: &str, a0: u64, run1: &str, a1: u64| -> Vec<GameRow> {
            ["estimator_v13", "estimator_v14"]
                .iter()
                .flat_map(|o| {
                    vec![mk(o, run0, a0, 0, 0), mk(o, run0, a0, 0, 1), mk(o, run1, a1, 1, 0), mk(o, run1, a1, 1, 1)]
                })
                .collect()
        };
        let clean = arm("A", 1, "A", 1);
        let cand = arm("C", 1, "C", 1);
        assert!(
            pair_by_opponent(&clean, &cand, 4, 2, &seeds, false).is_ok(),
            "1実行ずつの正常な入力は通る"
        );
        // 対照側が run A（shard 0）＋ run B（shard 1）の混ぜ物
        let mixed = arm("A", 1, "B", 1);
        assert!(pair_by_opponent(&mixed, &cand, 4, 2, &seeds, false).is_err());
        assert!(
            pair_by_opponent(&mixed, &cand, 4, 2, &seeds, true).is_err(),
            "--allow-incomplete でも run 混入は降格しない"
        );
        // 同じ run の re-run attempt 違いも別実行
        let reran = arm("A", 1, "A", 2);
        assert!(pair_by_opponent(&reran, &cand, 4, 2, &seeds, false).is_err());
        // 候補側の混入も同様
        assert!(pair_by_opponent(&clean, &arm("C", 1, "D", 1), 4, 2, &seeds, false).is_err());
    }

    /// **pair_with は対照の run_id と一致する**（対照の取り違えをここで閉じる）
    #[test]
    fn arena_balance_checks_pair_with_against_control_run() {
        let seeds: BTreeMap<String, u64> =
            [("estimator_v13".to_string(), 1u64), ("estimator_v14".to_string(), 1u64)]
                .into_iter()
                .collect();
        let arm = |run: &str, pw: Option<&str>| -> Vec<GameRow> {
            ["estimator_v13", "estimator_v14"]
                .iter()
                .flat_map(|o| {
                    (0..2u64)
                        .map(|g| {
                            let mut r = balance_row(o, g, 0.5);
                            r.run_id = run.into();
                            r.pair_with = pw.map(str::to_string);
                            r
                        })
                        .collect::<Vec<_>>()
                })
                .collect()
        };
        let ctrl = arm("ctrl-run", None);
        assert!(
            pair_by_opponent(&ctrl, &arm("cand-run", Some("ctrl-run")), 2, 1, &seeds, false).is_ok(),
            "対照を正しく指した候補は通る"
        );
        assert!(
            pair_by_opponent(&ctrl, &arm("cand-run", Some("other-run")), 2, 1, &seeds, false)
                .is_err(),
            "候補が指した対照と --control の記録が別物なら Err"
        );
        // pair_with の無い記録（旧 workflow・手動起動）は他の同一性検査に乗る
        assert!(pair_by_opponent(&ctrl, &arm("cand-run", None), 2, 1, &seeds, false).is_ok());
    }

    /// **判定を変えられるパラメータは事前登録定数と照合する**（PR #41 レビュー
    /// 4巡目 [P1]）: 50% CI・1反復 bootstrap・別の shard 数/seed/相手で取った
    /// 「良い数字」は #40 の通過にならない
    #[test]
    fn arena_balance_preregistration_deviations_are_indeterminate() {
        let seeds: BTreeMap<String, u64> = BALANCE_EXPECT_SEEDS
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect();
        let opps: BTreeSet<String> =
            BALANCE_EXPECT_SEEDS.iter().map(|(k, _)| k.to_string()).collect();
        assert!(
            balance_prereg_notes(600, 8, &seeds, &opps, 0.05, 10_000).is_empty(),
            "事前登録どおりの指定にはノートが付かない"
        );
        // 定数そのもの（issue #40 本文の事前登録値）
        assert_eq!(BALANCE_EXPECT_SHARDS, 8);
        assert_eq!(BALANCE_ALPHA, 0.05);
        assert_eq!(BALANCE_MIN_BOOT, 10_000);
        assert_eq!(seeds["estimator_v13"], 20260910);
        assert_eq!(seeds["estimator_v14"], 20260909);

        let mut wrong_seeds = seeds.clone();
        wrong_seeds.insert("estimator_v13".into(), 99);
        let wrong_opps: BTreeSet<String> = ["estimator_v14".to_string()].into_iter().collect();
        let deviations = [
            balance_prereg_notes(104, 8, &seeds, &opps, 0.05, 10_000),
            balance_prereg_notes(600, 4, &seeds, &opps, 0.05, 10_000),
            balance_prereg_notes(600, 8, &wrong_seeds, &opps, 0.05, 10_000),
            balance_prereg_notes(600, 8, &seeds, &wrong_opps, 0.05, 10_000),
            // レビューの再現: --alpha 0.5 = 50% CI で「CI 下限 > 0」を判定
            balance_prereg_notes(600, 8, &seeds, &opps, 0.5, 10_000),
            // レビューの再現: --boot 1 = 1回の再標本化を CI と呼ぶ
            balance_prereg_notes(600, 8, &seeds, &opps, 0.05, 1),
        ];
        for notes in deviations {
            assert!(!notes.is_empty(), "事前登録から外れた指定にはノートが付く");
            assert!(!balance_final(true, &notes).1, "数値の門が通っても pass しない");
        }
    }

    /// **manifest の指紋は行と照合する**: 別 manifest の指紋は Err、指紋の無い
    /// 行は本数が返る = 判定不能ノート行き（「計測前に commit した manifest の
    /// 下で測った」ことを機械検証できない）
    #[test]
    fn arena_balance_manifest_fingerprint_is_checked_on_rows() {
        let with_fp = |fp: Option<&str>| {
            let mut r = game_row("checkmate", 1.0);
            r.balance_manifest = fp.map(str::to_string);
            r
        };
        assert_eq!(
            check_manifest_rows(&[with_fp(Some("aa")), with_fp(Some("aa"))], "candidate", "aa"),
            Ok(0)
        );
        assert!(
            check_manifest_rows(&[with_fp(Some("bb"))], "candidate", "aa").is_err(),
            "別の manifest の下で測った run は Err"
        );
        assert_eq!(
            check_manifest_rows(&[with_fp(None), with_fp(Some("aa"))], "control", "aa"),
            Ok(1),
            "指紋の無い行は本数 = 判定不能ノート行き"
        );
    }

    /// 合算の点推定は「各相手の平均の単純平均」= 局数が偏っても 1/2 ずつ
    #[test]
    fn arena_balance_point_is_mean_of_stratum_means() {
        let strata = vec![vec![1.0, 0.0, 1.0, 0.0], vec![0.5, 0.5, 0.5, 0.5]];
        let (p, lo, hi) = stratified_boot_mean_ci(&strata, 2000, 7, 0.05);
        assert!((p - 0.5).abs() < 1e-12);
        assert!(lo <= p && p <= hi);
        let skewed = vec![vec![1.0; 100], vec![0.0; 4]];
        let (p, _, _) = stratified_boot_mean_ci(&skewed, 100, 7, 0.05);
        assert!((p - 0.5).abs() < 1e-12, "局数の多い相手が合算を支配しない");
        // 空の層は判定不能（NaN）: どの門も通らない側に落ちる
        let empty = vec![vec![1.0], vec![]];
        assert!(stratified_boot_mean_ci(&empty, 10, 7, 0.05).0.is_nan());
    }

    /// 事前登録した門: どの1条件が欠けても不通過。NaN も不通過側
    #[test]
    fn arena_balance_verdict_enforces_preregistered_gate() {
        let ok = BalanceGate {
            combined: 0.05,
            ci_lo: 0.01,
            per_opp: vec![("estimator_v13".into(), 0.04), ("estimator_v14".into(), 0.06)],
            fouls_delta: 0.1,
            think_delta: 20.0,
            cand_timeouts: 0,
        };
        assert!(balance_verdict(&ok).0);
        let fails = |g: BalanceGate| !balance_verdict(&g).0;
        assert!(fails(BalanceGate { combined: 0.039, ..ok.clone() }), "+0.04 未満");
        assert!(fails(BalanceGate { ci_lo: 0.0, ..ok.clone() }), "CI 下限 ≤ 0");
        assert!(fails(BalanceGate {
            per_opp: vec![("estimator_v13".into(), -0.01), ("estimator_v14".into(), 0.12)],
            combined: 0.055,
            ..ok.clone()
        }), "相手別符号 veto");
        assert!(fails(BalanceGate { fouls_delta: 0.31, ..ok.clone() }), "反則/局 +0.3 超");
        assert!(fails(BalanceGate { cand_timeouts: 1, ..ok.clone() }), "時間切れ");
        assert!(fails(BalanceGate { think_delta: 101.0, ..ok.clone() }), "思考平均 +100ms 超");
        assert!(fails(BalanceGate { combined: f64::NAN, ..ok.clone() }), "NaN は不通過側");
    }

    /// 判定不能の理由が1つでもあれば、数値の門が通っていても最終判定は
    /// 「通過」にならない（A/A・不完全入力・事前登録外の N の「通過」表示を塞ぐ）
    #[test]
    fn arena_balance_indeterminate_never_passes() {
        assert_eq!(balance_final(true, &[]), ("通過", true));
        assert_eq!(balance_final(false, &[]), ("不通過", false));
        let notes = vec!["--expect-games 104 ≠ 事前登録 600".to_string()];
        assert_eq!(balance_final(true, &notes), ("判定不能", false));
        assert_eq!(balance_final(false, &notes), ("判定不能", false));
        assert_eq!(BALANCE_EXPECT_GAMES, 600, "事前登録した各600局");
    }

    /// 実際の JSONL を通した parse の検査。**構造体を直接組み立てるテストでは
    /// パーサの抜けを検出できない**（PR #23 2回目レビュー指摘3）
    fn valid_row_json() -> serde_json::Value {
        serde_json::json!({
            "schema": GAME_ROW_SCHEMA,
            "candidate": "estimator",
            "baseline": "estimator_v14",
            "match_seed": 20260815,
            "match_seed_base": 20260815,
            "match_seed_shard": 0,
            "run_id": "33604671318",
            "run_attempt": 1,
            "pair_with": null,
            "balance_manifest": null,
            "game_no": 0,
            "a_is_sente": true,
            "score_a": 1.0,
            "reason": "checkmate",
            "plies": 100,
            "fouls_a": 6,
            "fouls_b": 5,
            "fouls_in_check_a": 1,
            "think_ms_a": 40000,
            "think_ms_b": 38000,
            "moves_a": 40,
            "commit": "deadbeef",
            "cand_knobs": { "TSUITATE_DROP_PROBE_REPEAT_GATE": "1" },
            "clock": [1000000, 3000],
            "think_budget_ms_a": 700,
            "think_budget_ms_b": 700,
            "cand_config": "abc123",
            "baseline_behavior": "def456",
            "shared_env": {},
        })
    }

    #[test]
    fn game_row_parser_requires_every_effective_condition() {
        let ok = parse_game_rows_text(&valid_row_json().to_string(), "t").expect("正常行");
        assert_eq!(ok.len(), 1);
        assert_eq!(ok[0].cand_knobs["TSUITATE_DROP_PROBE_REPEAT_GATE"], "1");
        assert_eq!(ok[0].budget_opp, "700");

        // **欠損は空値で埋めない**。map 系（`unwrap_or_default()` だった）も含めて
        // 全キーがエラーになること
        for k in [
            "candidate",
            "baseline",
            "match_seed",
            "match_seed_base",
            "match_seed_shard",
            "run_id",
            "run_attempt",
            "pair_with",
            "balance_manifest",
            "game_no",
            "a_is_sente",
            "score_a",
            "reason",
            "plies",
            "fouls_a",
            "fouls_b",
            "fouls_in_check_a",
            "think_ms_a",
            "think_ms_b",
            "moves_a",
            "commit",
            "cand_knobs",
            "clock",
            "think_budget_ms_a",
            "think_budget_ms_b",
            "cand_config",
            "baseline_behavior",
            "shared_env",
        ] {
            let mut v = valid_row_json();
            v.as_object_mut().unwrap().remove(k);
            let e = parse_game_rows_text(&v.to_string(), "t")
                .expect_err(&format!("{k} の欠損はエラーになるべき"));
            assert!(e.contains(k), "{k}: エラー文にキー名が無い: {e}");
        }

        // object でない env は「空 env」に落とさない
        for k in ["cand_knobs", "shared_env"] {
            let mut v = valid_row_json();
            v[k] = serde_json::json!("TSUITATE_X=1");
            let e = parse_game_rows_text(&v.to_string(), "t").expect_err("object 以外はエラー");
            assert!(e.contains("object"), "{e}");
        }

        // 古い schema は集計から弾く（schema 3 = 実行の識別子が無い時期も）
        for s in [1, 3] {
            let mut old = valid_row_json();
            old["schema"] = serde_json::json!(s);
            assert!(parse_game_rows_text(&old.to_string(), "t").is_err(), "schema {s}");
        }

        // pair_with / balance_manifest は null 可・文字列以外の値は弾く
        let mut v = valid_row_json();
        v["pair_with"] = serde_json::json!("33604671318");
        let ok = parse_game_rows_text(&v.to_string(), "t").expect("文字列は正常");
        assert_eq!(ok[0].pair_with.as_deref(), Some("33604671318"));
        v["pair_with"] = serde_json::json!(123);
        assert!(parse_game_rows_text(&v.to_string(), "t").is_err(), "数値の pair_with は弾く");
    }

    /// **1ペアでは分散が同定できない**ので明示的に失敗させる
    /// （Var=0 / SE=0 / MDE=0 を「完全に精密」と読ませない。2回目レビュー指摘1）
    #[test]
    fn arena_var_requires_at_least_two_pairs() {
        assert_eq!(MIN_ARENA_PAIRS, 2);
        assert_eq!(variance(&[0.5]), 0.0, "n<2 の分散は同定できない（0 は自由度 0 の産物）");
        for n in [0usize, 1] {
            let e = arena_var_pair_error(n);
            assert!(e.contains(&n.to_string()));
            assert!(e.contains("allow-incomplete"), "override できないことを書く");
        }
    }

    /// **反則負け率は「反則で負けた」ときだけ 1**（相手の反則負けで勝った局を
    /// 自分の反則負けに数えると、悪化と改善が同じ方向に出てゲートが壊れる）
    #[test]
    fn foul_limit_loss_counts_only_own_loss() {
        let lost = game_metrics(&game_row("foul_limit", 0.0));
        let won = game_metrics(&game_row("foul_limit", 1.0));
        let mated = game_metrics(&game_row("checkmate", 0.0));
        assert_eq!(lost["foul_limit_loss"], 1.0);
        assert_eq!(won["foul_limit_loss"], 0.0);
        assert_eq!(mated["foul_limit_loss"], 0.0);
        assert_eq!(mated["hit_max_plies"], 0.0);
        assert_eq!(game_metrics(&game_row("max_plies", 0.5))["hit_max_plies"], 1.0);
        // 思考平均は「1手あたり」（局の長さで割らないと長い局に引っ張られる）
        assert!((lost["think_avg_ms_me"] - 1000.0).abs() < 1e-9);
    }

    /// 局ごとの指標名は checkpoint 側（`METRICS`）の部分集合であること。
    /// 名前がずれると `report` の横断表で arena 側の列だけ空になる
    #[test]
    fn arena_metric_names_are_a_subset_of_checkpoint_metrics() {
        let m = game_metrics(&game_row("checkmate", 1.0));
        let known: BTreeSet<&str> = METRICS.iter().map(|(n, _)| *n).collect();
        for k in m.keys() {
            assert!(known.contains(k), "{k} が METRICS にありません");
        }
        assert!(m.contains_key("score"));
    }

    /// **MDE は CI 半幅ではない**（PR #20 レビュー指摘1）。
    /// alpha=0.05 / power=0.80 なら 2.80·SE で、CI 半幅 1.96·SE の約1.43倍。
    /// 同じ効果量を検出するのに必要な N は (2.80/1.96)² ≈ 2.04 倍になる
    #[test]
    fn mde_is_larger_than_ci_half_width() {
        let se = 0.05;
        let za = z_two_sided(0.05);
        let zb = z_upper(1.0 - 0.80);
        let ci_half = za * se;
        let m = mde(se, za, zb);
        assert!((m / ci_half - 1.43).abs() < 0.02, "MDE/CI半幅 = {}", m / ci_half);
        // 必要 N の比（SE ∝ 1/√N）
        let n_ratio = (m / ci_half).powi(2);
        assert!((n_ratio - 2.04).abs() < 0.05, "必要Nの比 = {n_ratio}");
    }

    /// power simulation は**本番と同じ percentile cluster bootstrap CI** を当てる。
    /// 効果量 0 のとき偽陽性率が alpha 付近、大きい効果量では検出力が上がること
    /// （解析式と同じ z 検定を当てていた初版は「独立の裏取り」になっていなかった）
    #[test]
    fn power_simulation_uses_the_real_decision_rule() {
        let centered: Vec<f64> = (0..64)
            .map(|i| match i % 4 {
                0 => -0.5,
                1 => 0.5,
                2 => -0.25,
                _ => 0.25,
            })
            .collect();
        // **効果ゼロは type-I error（偽陽性率）の経路**。符号一致を要求すると
        // 構造上ゼロになり検査にならないので、CI が 0 を外した割合を数える
        let fp = power_simulation_bootstrap(&centered, 64, 0.0, 0.05, 400, 400, 3);
        assert!(fp > 0.0, "偽陽性率が構造上ゼロになっている（検査が空回り）");
        assert!(fp < 0.15, "偽陽性率 {fp} が alpha=0.05 から離れすぎ");
        // 大きい効果 → ほぼ必ず検出
        let pw = power_simulation_bootstrap(&centered, 64, -0.25, 0.05, 300, 300, 3);
        assert!(pw > 0.9, "検出力 {pw} が低すぎる");
        // 単調性
        let mid = power_simulation_bootstrap(&centered, 64, -0.10, 0.05, 300, 300, 3);
        assert!(fp < mid && mid < pw, "fp {fp} / mid {mid} / pw {pw}");
    }

    /// 相手が arm 固有の env を読むかを、凍結版のソースから検出できる
    /// （PR #20 追加レビュー指摘1: env はプロセス全体に効くので、相手も
    ///  同じノブを読むと「同じ固定相手」で比べられない）
    #[test]
    fn detects_env_leaking_into_the_frozen_opponent() {
        // v14 は実際にこれらを読む（凍結時点の読み方が残っている）
        assert!(strategy_reads_env("estimator_v14", "TSUITATE_ANCHOR_MOVE_W"));
        assert!(strategy_reads_env("estimator_v14", "TSUITATE_DROP_PROBE_REPEAT_GATE"));
        assert!(strategy_reads_env("estimator_v14", "TSUITATE_HAND_ASSET_W"));
        // 未知の戦略名は安全側（読むと見なす）
        assert!(strategy_reads_env("estimator_rush", "TSUITATE_ANCHOR_MOVE_W"));
    }

    /// **共有モジュール経由の読取も検出する**（PR #20 4回目レビュー指摘1）。
    /// v14 の凍結ファイルに `TSUITATE_JOSEKI` の文字列は無いが、
    /// v14 が作る `crate::opening::OpeningBook` の `load()` が読む
    #[test]
    fn detects_env_read_through_shared_modules() {
        assert!(
            !include_str!("../frozen/estimator_v14.rs").contains("TSUITATE_JOSEKI"),
            "前提: 凍結ファイル自体にはこの文字列が無い（あるならテストの意味が変わる）"
        );
        assert!(
            // doc コメントの言及（backtick）ではなく、**文字列リテラル**が無いこと。
            // 走査が拾うのはリテラルだけなので、これが無ければ機械的には見えない
            !include_str!("../opening.rs").contains("\"TSUITATE_JOSEKI\""),
            "前提: opening.rs にリテラルは無い（config 経由で解決するので走査では拾えない）"
        );
        assert!(
            strategy_reads_env("estimator_v14", "TSUITATE_JOSEKI"),
            "共有 opening.rs 経由の読取を見落としている（frozen::SHARED_MODULE_ENV）"
        );
    }

    /// **プロセス env に arm 固有の値を置くのは今も原則拒否**。
    /// issue #21 でノブは config へ移したので、ここへ来るのは回帰したときだけ
    #[test]
    fn arm_specific_env_is_denied_by_default() {
        // どの凍結版も読まないであろう架空のキーでも拒否される
        let keys = vec!["TSUITATE_NOT_READ_BY_ANYONE_XYZ".to_string()];
        assert!(!strategy_reads_env("estimator_v14", "TSUITATE_NOT_READ_BY_ANYONE_XYZ"));
        // allow=true のときだけ通る（拒否リストは返る）
        let denied = assert_opponent_blind_to("estimator_v14", &keys, true);
        assert_eq!(denied, keys, "監査済みリストに無いので拒否対象として返る");
        assert!(CANDIDATE_ONLY_ENV.is_empty(), "arm 固有ノブは config で渡すので空でよい");
    }

    /// **issue #21 の要点**: arm 固有ノブを config で渡しても、
    /// 凍結相手の実効設定は動かない（プロセス env を触らないから）。
    /// PR #20 で見つかった「candidate 用の env が v14 にも効く」の再発防止。
    #[test]
    fn arm_knobs_do_not_change_the_frozen_opponent() {
        let key = "TSUITATE_HAND_ASSET_W";
        // 前提: v14 はこのキーを読む（読まないキーでは検査の意味が無い）
        assert!(strategy_reads_env("estimator_v14", key));

        // **base は空にする**（呼び出し元の shell に同じキーが残っていると
        //  上書きが no-op になってテストが env に左右される。PR #22 再レビュー P3）
        let base = config::EnvSource::empty();
        let opp = config::StrategyConfig::from_source(base.clone());
        let arm = config::StrategyConfig::from_source(
            base.with_overrides([(key.to_string(), "0.5".to_string())]),
        );
        assert_ne!(arm.fingerprint(), opp.fingerprint(), "候補側には効いている");
        assert_eq!(arm.strategy.hand_asset_w, 0.5);

        // **相手が読むのはプロセス env**で、それは arm config の構築で変わらない
        // （ここだけは実際の env を見る必要がある。値そのものではなく不変性を見る）
        let env_before = std::env::var(key).ok();
        let opp_before =
            config::StrategyConfig::from_source(config::EnvSource::from_process()).fingerprint();
        let _ = config::StrategyConfig::from_source(
            config::EnvSource::from_process()
                .with_overrides([(key.to_string(), "0.5".to_string())]),
        );
        assert_eq!(std::env::var(key).ok(), env_before);
        assert_eq!(
            config::StrategyConfig::from_source(config::EnvSource::from_process()).fingerprint(),
            opp_before,
            "arm 固有ノブでプロセス env が変わってはいけない"
        );
    }

    /// 凍結版は config を尊重しないので、ノブを渡す先として選べない
    /// （黙って無視されるのを防ぐ）
    #[test]
    fn frozen_strategies_do_not_honor_config() {
        assert!(strategy::honors_config("estimator"));
        assert!(strategy::honors_config("estimator_rush"));
        for (_, name, _) in frozen::SOURCES {
            assert!(!strategy::honors_config(name), "{name}");
            assert!(
                strategy::make_seeded_with_config(name, 0, std::sync::Arc::new(
                    config::StrategyConfig::defaults()
                ))
                .is_none(),
                "{name} へ config を渡せてはいけない"
            );
        }
    }

    /// bootstrap CI の percentile は alpha に連動する（alpha を広げれば CI は狭くなる）。
    /// 以前のレビューで実際に見つかった回帰（2.5/97.5% 固定）のテスト
    #[test]
    fn bootstrap_ci_follows_alpha() {
        let means: Vec<f64> = (0..40).map(|i| (i as f64 - 20.0) / 40.0).collect();
        let (lo95, hi95) = bootstrap_ci_of_means(&means, 2000, 1, 0.05);
        let (lo50, hi50) = bootstrap_ci_of_means(&means, 2000, 1, 0.50);
        assert!(hi95 - lo95 > hi50 - lo50, "alpha を広げれば CI は狭くなる");
    }

    /// **summary JSON にも schema の契約を適用する**（PR #20 5回目レビュー指摘1）。
    /// `compare` が schema 1 の JSONL を拒否しても、そこから既に作られた
    /// summary を `report` が受理してしまうと、撤回済みの数字が横断表へ戻る
    #[test]
    fn report_rejects_schema1_summary() {
        let legacy = serde_json::json!({ "schema": 1, "label": "old", "delta_pt": -6.2 });
        let err = check_summary_schema(&legacy, "old.summary.json").unwrap_err();
        assert!(err.contains("schema 1"), "{err}");
        // schema 2（ノブをプロセス env で渡していた時期）も拒否する（issue #21）
        let env_era = serde_json::json!({ "schema": 2, "label": "env", "delta_pt": 1.0 });
        let err = check_summary_schema(&env_era, "env.summary.json").unwrap_err();
        assert!(err.contains("schema 2"), "{err}");
        let current = serde_json::json!({ "schema": SUMMARY_SCHEMA, "label": "new" });
        assert!(check_summary_schema(&current, "new.summary.json").is_ok());
        // schema が無い / 未知の値も拒否
        assert!(check_summary_schema(&serde_json::json!({}), "x").is_err());
        assert!(check_summary_schema(&serde_json::json!({"schema": 99}), "x").is_err());
    }

    /// cluster 平均の分散は cluster bootstrap と同じ対象を見ている
    /// （分散成分から組み立てると σ_b² のクリップで過大評価される）
    #[test]
    fn cluster_mean_var_matches_direct_estimate() {
        let clusters = vec![vec![1.0, 1.0], vec![0.0, 0.0], vec![-1.0, -1.0], vec![0.0, 0.0]];
        // cluster 平均は 1, 0, -1, 0 → 標本分散 = 2/3
        assert!((cluster_mean_var(&clusters) - 2.0 / 3.0).abs() < 1e-9);
    }
}
