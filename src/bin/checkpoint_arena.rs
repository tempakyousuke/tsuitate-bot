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

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use tsuitate_bot::checkpoint::{
    Deck, DeckEntry, GameCandidates, candidates, kif_ending, phase_tag, restore,
    split_of, stable_hash, stratified_pick,
};
use tsuitate_bot::kifu::kif_body;
use tsuitate_bot::protocol::Color;
use tsuitate_bot::scenario_core::make_view;
use tsuitate_bot::selfplay::{GameResult, StartState, play_continuation};
use tsuitate_bot::strategy;
use tsuitate_bot::truth_replay::{load_end, side_idx};

const SCHEMA: u32 = 1;

fn usage() -> &'static str {
    "usage: checkpoint_arena <extract|run|unit|pair|compare|report> [options]\n\
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
     report <summary.json...> [--markdown OUT]"
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
/// 裁定が arm 間でズレる）
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

fn collect_record_files(specs: &[String]) -> Vec<PathBuf> {
    let mut out = vec![];
    for spec in specs {
        let p = Path::new(spec);
        if p.is_dir() {
            walk_jsonl(p, &mut out);
        } else {
            out.push(p.to_path_buf());
        }
    }
    out.sort();
    out.dedup();
    out
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
    for path in &files {
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
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
    println!("deck_hash {}", deck.hash());
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
) -> UnitResult {
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
                let mut strat = strategy::make_seeded(name, s)
                    .unwrap_or_else(|| die(&format!("未知の戦略名: {name}")));
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
        "schema": SCHEMA,
        "experiment": experiment,
        "arm": arm.label,
        "arm_order": arm_order,
        "checkpoint": entry.id,
        // 元対局（cluster bootstrap の統計単位）。1棋譜1checkpoint なら 1:1
        "source_game": entry.id.rsplit_once("-ply").map(|(g, _)| g).unwrap_or(&entry.id),
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
        "commit": git_commit(),
        "deck_hash": deck.hash(),
        "think_budget_ms": std::env::var("TSUITATE_CAND_THINK_BUDGET_MS")
            .or_else(|_| std::env::var("TSUITATE_THINK_BUDGET_MS"))
            .unwrap_or_else(|_| "2000".into()),
        "prewarm_shared": shared,
        "prewarm_reason": prewarm_reason,
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
    let r = play_one_arm(
        &dir, &deck, &entry, seed, &arm, &opponent, order, None, &experiment, &reason,
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

    // prewarm 共有（opt-in）: 両 arm の推定器と prewarm 設定が同じときだけ
    let shared = if args.flag("shared-prewarm") {
        if control.strategy != candidate.strategy {
            die("--shared-prewarm は両 arm の戦略が同じときだけ使えます（推定器が違うと共有できない）");
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
            let mut strat = strategy::make_seeded(name, s)
                .unwrap_or_else(|| die(&format!("未知の戦略名: {name}")));
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
    for (k, arm) in arms.iter().enumerate() {
        let r = play_one_arm(
            &dir, &deck, &entry, seed, arm, &opponent, k, shared.as_ref(), &experiment, &reason,
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
    env: BTreeMap<String, String>,
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
    // arm ごとの env は追加のみ（親の TSUITATE_* を消さない = 呼び出し側が
    // 両 arm 共通の予算を渡せる）。ARENA_* は審判側なので親のまま両 arm 同一
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
    let seeds: u64 = args.num("seeds", 1);
    let seed_base: u64 = args.num("seed-base", 0);
    let jobs: usize = args.num("jobs", 1);
    let limit: usize = args.num("limit", 0);
    let experiment = args.get("experiment").unwrap_or("adhoc").to_string();
    let opponent = args.get("opponent").unwrap_or(&deck.opponent).to_string();
    let budget = args.get("budget-ms").map(str::to_string);
    let shared_prewarm = args.flag("shared-prewarm");

    let self_bin = std::env::current_exe().unwrap_or_else(|e| die(&format!("自分の実行ファイルが分かりません: {e}")));
    let mut control_env = parse_env(args.get("control-env"));
    let mut candidate_env = parse_env(args.get("candidate-env"));
    if let Some(b) = &budget {
        // 両側・両 arm に同じ予算を渡す（凍結版も TSUITATE_THINK_BUDGET_MS を読む）
        control_env.insert("TSUITATE_THINK_BUDGET_MS".into(), b.clone());
        candidate_env.insert("TSUITATE_THINK_BUDGET_MS".into(), b.clone());
    }
    let control = RunArm {
        label: "control".into(),
        strategy: args.get("control-strategy").unwrap_or("estimator").to_string(),
        env: control_env,
        bin: args.get("control-bin").map(PathBuf::from).unwrap_or_else(|| self_bin.clone()),
    };
    let candidate = RunArm {
        label: "candidate".into(),
        strategy: args.get("candidate-strategy").unwrap_or("estimator").to_string(),
        env: candidate_env,
        bin: args.get("candidate-bin").map(PathBuf::from).unwrap_or_else(|| self_bin.clone()),
    };

    // prewarm 共有の可否（issue の厳しい共有条件）。理由を JSONL 側へ残せるよう
    // ここで判定して表示する
    let same_bin = control.bin == candidate.bin;
    let same_env = control.env == candidate.env;
    let same_strategy = control.strategy == candidate.strategy;
    let share_reason = if !shared_prewarm {
        "opt-in されていない"
    } else if !same_bin {
        "base/target 別バイナリ"
    } else if !same_env {
        "arm ごとに異なるグローバル env / OnceLock"
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
                scope.spawn(move || {
                    let mut out: Vec<String> = vec![];
                    let mut k = t;
                    while k < units.len() {
                        let (ei, seed, order) = units[k];
                        let entry = &entries[ei];
                        let base = vec![
                            "--id".to_string(), entry.id.clone(),
                            "--seed".to_string(), seed.to_string(),
                            "--opponent".to_string(), opponent.clone(),
                            "--experiment".to_string(), experiment.clone(),
                            "--prewarm-reason".to_string(),
                            if share_reason.is_empty() { "共有条件を満たす".to_string() } else { share_reason.to_string() },
                        ];
                        if share {
                            let mut extra = base.clone();
                            extra.extend([
                                "--control-strategy".into(), control.strategy.clone(),
                                "--candidate-strategy".into(), candidate.strategy.clone(),
                                "--shared-prewarm".into(), "1".into(),
                                "--arm-order".into(), order.to_string(),
                            ]);
                            out.extend(run_child(&control.bin, "pair", deck_path, &extra, &control.env));
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
                                    "--arm-order".into(), j.to_string(),
                                ]);
                                out.extend(run_child(&arm.bin, "unit", deck_path, &extra, &arm.env));
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

fn parse_rows(paths: &[String]) -> Vec<Row> {
    let mut rows = vec![];
    for p in paths {
        let text = std::fs::read_to_string(p).unwrap_or_else(|e| die(&format!("{p}: {e}")));
        for line in text.lines().filter(|l| l.trim().starts_with('{')) {
            let v: serde_json::Value =
                serde_json::from_str(line).unwrap_or_else(|e| die(&format!("{p}: {e}")));
            let f = |k: &str| v[k].as_f64().unwrap_or(0.0);
            let b = |k: &str| if v[k].as_bool().unwrap_or(false) { 1.0 } else { 0.0 };
            let mut metrics: BTreeMap<&'static str, f64> = BTreeMap::new();
            metrics.insert("score", f("score"));
            metrics.insert("fouls_me", f("fouls_me"));
            metrics.insert("fouls_in_check_me", f("fouls_in_check_me"));
            metrics.insert("foul_limit_loss", b("foul_limit_loss"));
            metrics.insert("added_plies", f("added_plies"));
            metrics.insert("hit_max_plies", b("hit_max_plies"));
            metrics.insert("fouls_opp", f("fouls_opp"));
            metrics.insert("think_avg_ms_me", f("think_avg_ms_me"));
            metrics.insert("think_p95_ms_me", f("think_p95_ms_me"));
            metrics.insert("think_p99_ms_me", f("think_p99_ms_me"));
            rows.push(Row {
                arm: v["arm"].as_str().unwrap_or("").to_string(),
                arm_order: v["arm_order"].as_u64().unwrap_or(0) as usize,
                experiment: v["experiment"].as_str().unwrap_or("adhoc").to_string(),
                checkpoint: v["checkpoint"].as_str().unwrap_or("").to_string(),
                source_game: v["source_game"].as_str().unwrap_or("").to_string(),
                seed: v["seed"].as_u64().unwrap_or(0),
                split: v["split"].as_str().unwrap_or("").to_string(),
                tags: v["tags"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|t| t.as_str().map(str::to_string)).collect())
                    .unwrap_or_default(),
                start_ply: v["start_ply"].as_u64().unwrap_or(0),
                first_move: v["first_move"].as_str().unwrap_or("-").to_string(),
                reason: v["reason"].as_str().unwrap_or("").to_string(),
                commit: v["commit"].as_str().unwrap_or("").to_string(),
                deck_hash: v["deck_hash"].as_str().unwrap_or("").to_string(),
                think_budget: v["think_budget_ms"].as_str().unwrap_or("").to_string(),
                prewarm_shared: v["prewarm_shared"].as_bool().unwrap_or(false),
                metrics,
                prewarm_ms: f("prewarm_ms_me") + f("prewarm_ms_opp"),
                continuation_ms: f("continuation_ms"),
                total_ms: f("total_ms"),
            });
        }
    }
    rows
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
}

/// 元対局（cluster）単位のブートストラップ。**seed を独立標本として数えない**:
/// 再標本化するのは cluster で、cluster 内の値はまとめて出入りする
fn cluster_bootstrap(clusters: &[Vec<f64>], b: usize, seed: u64) -> (f64, f64) {
    use rand::{Rng, SeedableRng};
    if clusters.is_empty() {
        return (0.0, 0.0);
    }
    let means: Vec<f64> = clusters.iter().map(|c| mean(c)).collect();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut samples: Vec<f64> = Vec::with_capacity(b);
    for _ in 0..b {
        let m = mean(
            &(0..means.len())
                .map(|_| means[rng.random_range(0..means.len())])
                .collect::<Vec<f64>>(),
        );
        samples.push(m);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo = samples[(b as f64 * 0.025) as usize];
    let hi = samples[((b as f64 * 0.975) as usize).min(b - 1)];
    (lo, hi)
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
    let known: Option<f64> = args.get("known-arena-delta").and_then(|v| v.parse().ok());
    let label = args.get("label").map(str::to_string);

    let rows = parse_rows(&paths);
    if rows.is_empty() {
        die("行がありません");
    }
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
    let cluster_deltas = |metric: &str| -> Vec<Vec<f64>> {
        by_cluster
            .values()
            .map(|v| {
                v.iter()
                    .map(|(c, k)| k.metrics[metric] - c.metrics[metric])
                    .collect()
            })
            .collect()
    };

    let mut out = String::new();
    let commits: HashSet<&str> = rows.iter().map(|r| r.commit.as_str()).collect();
    let decks: HashSet<&str> = rows.iter().map(|r| r.deck_hash.as_str()).collect();
    let budgets: HashSet<&str> = rows.iter().map(|r| r.think_budget.as_str()).collect();
    out.push_str(&format!("## checkpoint arena: {label}\n\n"));
    out.push_str(&format!(
        "- commit `{}` / deck `{}` / 思考予算 {} ms / prewarm 共有 {}\n",
        commits.iter().copied().collect::<Vec<_>>().join(","),
        decks.iter().copied().collect::<Vec<_>>().join(","),
        budgets.iter().copied().collect::<Vec<_>>().join(","),
        if rows.iter().any(|r| r.prewarm_shared) { "あり" } else { "なし" },
    ));
    out.push_str(&format!(
        "- ペア {} 組 / 元対局 {} / seed {} / 片割れ欠損 {}\n",
        paired.len(),
        by_cluster.len(),
        paired.len() / by_cluster.len().max(1),
        unpaired
    ));
    if commits.len() > 1 {
        out.push_str("- **注意: commit が混在しています**（同一コミットで取り直すこと）\n");
    }

    // ---- 主指標 ----
    let score_clusters = cluster_deltas("score");
    let (sb2, sw2, icc) = variance_components(&score_clusters);
    let n = score_clusters.len();
    let s = mean(&score_clusters.iter().map(|c| c.len() as f64).collect::<Vec<_>>()).max(1.0);
    let se = (sb2 / n as f64 + sw2 / (n as f64 * s)).sqrt();
    let d = mean(&score_clusters.iter().map(|c| mean(c)).collect::<Vec<f64>>());
    let (lo, hi) = cluster_bootstrap(&score_clusters, boot, 20260823);
    let ctrl_rate = mean(&paired.iter().map(|(c, _)| c.metrics["score"]).collect::<Vec<f64>>());
    let cand_rate = mean(&paired.iter().map(|(_, k)| k.metrics["score"]).collect::<Vec<f64>>());
    let improved = paired.iter().filter(|(c, k)| k.metrics["score"] > c.metrics["score"]).count();
    let worse = paired.iter().filter(|(c, k)| k.metrics["score"] < c.metrics["score"]).count();
    let same = paired.len() - improved - worse;

    out.push_str("\n### 勝敗（主指標）\n\n");
    out.push_str(&format!(
        "- candidate {:.1}% / control {:.1}% / **paired delta {} [{} , {}]**（元対局 cluster bootstrap 95%）\n",
        cand_rate * 100.0,
        ctrl_rate * 100.0,
        pct(d),
        pct(lo),
        pct(hi)
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

    // ---- MDE の power simulation ----
    out.push_str("\n### 必要 N と最小検出効果（実測分散から）\n\n");
    out.push_str("| 元対局数 \\ seed | 1 | 2 | 3 | 5 |\n|---|---|---|---|---|\n");
    for nn in [16usize, 32, 64, 128, 256] {
        let cells: Vec<String> = [1usize, 2, 3, 5]
            .iter()
            .map(|&ss| {
                let se = (sb2 / nn as f64 + sw2 / (nn as f64 * ss as f64)).sqrt();
                format!("±{:.1}pt", 1.96 * se * 100.0)
            })
            .collect();
        out.push_str(&format!("| {nn} | {} |\n", cells.join(" | ")));
    }
    out.push_str(
        "\nseed を増やしても減るのは cluster 内ノイズだけで、独立な元対局数は増えない。\n",
    );
    if s < 1.5 {
        out.push_str(
            "\n**注意: seed 1 では cluster 内分散 σ_w² が同定できない**（0 と推定される）ため、\n             上表の seed 列は全部同じ値になり、ICC も 1.000 に張り付く。\n             seed 次元の効果を見るには `--seeds 2` 以上で取り直すこと。\n",
        );
    }

    // ---- CPU あたりの情報量 ----
    // 「同じ検出力を得る CPU コスト」の比較用。Var(delta) × 1ペアあたり CPU 秒
    // （小さいほど効率がよい）。通常 arena と並べて撤退判断に使う
    let all_deltas: Vec<f64> = paired
        .iter()
        .map(|(c, k)| k.metrics["score"] - c.metrics["score"])
        .collect();
    let dm = mean(&all_deltas);
    let var_delta = if all_deltas.len() > 1 {
        all_deltas.iter().map(|x| (x - dm).powi(2)).sum::<f64>() / (all_deltas.len() as f64 - 1.0)
    } else {
        0.0
    };
    let cpu_per_pair = rows.iter().map(|r| r.total_ms).sum::<f64>() / 1000.0 / paired.len() as f64;
    out.push_str(&format!(
        "\n### CPU あたりの情報量\n\n         1ペア（arm 2本）あたり {cpu_per_pair:.0} CPU秒 / Var(delta) {var_delta:.3} /          **var·CPU秒 {:.0}**（小さいほど効率がよい）\n",
        var_delta * cpu_per_pair
    ));

    // ---- 安全性の共同指標 ----
    out.push_str("\n### 安全性の共同指標（元対局単位のペア差）\n\n");
    out.push_str("| 指標 | control | candidate | delta | 95% CI |\n|---|---:|---:|---:|---|\n");
    for (key, name) in METRICS.iter().skip(1) {
        let cl = cluster_deltas(key);
        let dd = mean(&cl.iter().map(|c| mean(c)).collect::<Vec<f64>>());
        let (l, h) = cluster_bootstrap(&cl, boot.min(4000), 20260824);
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
        out.push_str(&format!(
            "\n### 較正\n\n既知の通常 arena 差 {k:+.1}pt に対し checkpoint delta {}（符号一致: {}）\n",
            pct(d),
            if (d * 100.0).signum() == k.signum() { "はい" } else { "いいえ" }
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
            "schema": SCHEMA,
            "label": label,
            "pairs": paired.len(),
            "clusters": by_cluster.len(),
            "seeds": s,
            "delta_pt": d * 100.0,
            "ci_lo_pt": lo * 100.0,
            "ci_hi_pt": hi * 100.0,
            "se_pt": se * 100.0,
            "mde_pt": 1.96 * se * 100.0,
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
            "var_delta": var_delta,
            "var_cpu_sec": var_delta * cpu_per_pair,
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
        rows.push(serde_json::from_str(&text).unwrap_or_else(|e| die(&format!("{p}: {e}"))));
    }
    let mut out = String::new();
    out.push_str("## checkpoint arena 横断レポート\n\n");
    out.push_str(
        "| 実験 | 予算ms | 元対局 | seed | 既知arena | checkpoint delta | 95% CI | MDE | ICC | 反則delta | var·CPU秒 | CPU時間 |\n\
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
    if known.len() >= 2 {
        let k: Vec<f64> = known.iter().map(|v| f(v, "known_arena_delta_pt")).collect();
        let c: Vec<f64> = known.iter().map(|v| f(v, "delta_pt")).collect();
        let agree = k.iter().zip(&c).filter(|(a, b)| a.signum() == b.signum()).count();
        out.push_str(&format!(
            "\n- 符号一致: {agree}/{}\n- 順位相関（スピアマン）: {:.3}\n",
            k.len(),
            spearman(&k, &c)
        ));
        // 重大悪化（既知 −8pt 以下）の見逃し = CI の上端が 0 を跨いだもの
        let severe: Vec<&&serde_json::Value> = known.iter().filter(|v| f(v, "known_arena_delta_pt") <= -8.0).collect();
        let missed = severe.iter().filter(|v| f(v, "ci_hi_pt") >= 0.0).count();
        out.push_str(&format!(
            "- 重大悪化（既知 −8pt 以下）{} 件のうち、CI 上端が 0 を跨いで見逃したもの: {missed} 件\n",
            severe.len()
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
        other => die(&format!("未知のサブコマンド: {}", other.unwrap_or("(なし)"))),
    }
}
