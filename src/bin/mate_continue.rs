//! **一手強制の継続診断**（issue #28 P0-6）。
//!
//! 詰み負けした各局の「最後に受けられた決定点」から**一手だけ強制**して、
//! そのあとを通常方策で終局まで指し継ぎ、勝ち=1 / 引分=0.5 / 負け=0 で数える。
//! 受けが勝敗へ**どれだけ変換されうるか**の上限と、実現可能な推定値を出す:
//!
//! - `Δoracle` = Σ_局（候補生成に載る真実の安全手のうち最良の継続平均スコア）/ G
//! - `Δpolicy` = Σ_局（P0-3 のオフライン方策が選ぶ手の継続平均スコア）/ G
//!
//! G は**記録の全対局数**（受けの機会が無かった局は 0 として数える）。
//! `Δoracle < 0.04` なら issue #28 の P0 は即中止（受けても救われない）。
//!
//! 裁定は `selfplay::play_continuation` = 通常アリーナと同じ関数（MAX_PLIES・
//! 反則上限・王手/捕獲通知・終局判定を共有）。**時計は無効**（途中局面の残り
//! 時間は復元できない。本番相当で時間切れ0の実測があるので落としてよい）。
//!
//! **`Δoracle` は最良手を選ぶぶん楽観**（勝者の呪い）なので、seed を前半後半に
//! 割って「前半で選び後半で測る」正直版も併せて出す。中止判定は上限側で見る
//! ので楽観は保守的に働くが、`Δpolicy` との差を読むときは正直版を見ること。
//!
//! **replay 対照を必ず取る**: 元対局が負けだからといって「実戦の手を強制した
//! 継続」が 0 になるとは限らない（両者が指し直すので別の対局になる）。
//! `baseline` はその実測で、Δ の一部が「単に指し直したから」でないことを見る。
//!
//! usage:
//!   TSUITATE_THINK_BUDGET_MS=700 cargo run --release --bin mate_continue -- \
//!     [--seeds 2] [--max-safe 4] [--opponent estimator_v14] [--policy-w 2,4,8,16,30] \
//!     [--jobs N] [--shard i/n] [--out out.jsonl] <records/*.jsonl...>
//!   cargo run --release --bin mate_continue -- report <out-*.jsonl...>
//!
//! **思考予算はプロセス env**（`TSUITATE_THINK_BUDGET_MS`）。両側に同じ値が
//! 効く（凍結相手も同じ env を読む）ので、片側だけ厚くならない。

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tsuitate_bot::checkpoint::stable_hash;
use tsuitate_bot::mate::mate_moves_in_1_fast;
use tsuitate_bot::mate_economy::{analyze_game, force_move, rescored_usis};
use tsuitate_bot::observation::Observation;
use tsuitate_bot::protocol::{Color, GameEndPayload};
use tsuitate_bot::scenario_core::{
    Replayed, clone_log, make_view, ranking_and_particles, side_idx,
};
use tsuitate_bot::selfplay::{GameResult, StartState, mix, play_continuation};
use tsuitate_bot::shogi::{Position, ShogiMove};
use tsuitate_bot::strategy::{self, candidate_moves};
use tsuitate_bot::truth_replay::for_each_decision_full;

/// 今書き出す JSONL の schema。
///
/// - **schema 1 は撤回**（PR #30 レビュー1巡目の指摘②④の修正前に取った記録で、
///   q が別の粒子集合・継続の乱数が arm ごとに違う）
/// - **schema 2 は単一 w の記録**（arm 名が `policy_strict` / `policy_all`、
///   meta は `policy_w`）。run 33073983617 の値は w=4 の測定としては有効だが、
///   arm の語彙が違うので schema 3 以降とは混ぜられない（レビュー2巡目 [P1] で
///   「事前登録した有限個の w を全部回す」形になった）
/// - **schema 3 は行の内容が 4 と同じで、meta の実験キーだけが粗い**
///   （並列度が `jobs` 1本。レビュー3巡目 [P2] で `policy_jobs` /
///   `continuation_jobs` に分けた）。**撤回はしていない**ので `report` は
///   読めるが、4 と混ぜることはできない（下記 `READABLE_SCHEMAS`）
const ROW_SCHEMA: u32 = 4;

/// `report` が読める schema。**同じ集計の中で混ぜることはできない**
/// （3 と 4 は実験キーの粒度が違うので、同じ `jobs` でも別条件かもしれない）。
const READABLE_SCHEMAS: &[u32] = &[3, 4];

fn usage() -> &'static str {
    "usage: cargo run --release --bin mate_continue -- [--seeds 2] [--max-safe 4] \
     [--opponent estimator_v14] [--policy-w 2,4,8,16,30] [--jobs N] [--shard i/n] [--out out.jsonl] \
     [--allow-opponent-mismatch] <records/*.jsonl...>\n        または: mate_continue report [--allow-incomplete] <out-*.jsonl...>"
}

fn die(msg: &str) -> ! {
    eprintln!("{msg}");
    eprintln!("{}", usage());
    std::process::exit(1);
}

/// 強制する手の由来（arm のタグ）。
///
/// - `baseline` … 実戦で指した手（指し直しの対照）
/// - `oracle` … 真実の安全手
/// - `policy_w0` … 同じ経路で w=0（＝現行方策そのもの）。**再決定のぶんを
///   切り分ける対照**: `Δpolicy − Δpolicy(w=0)` が危険量ペナルティ自身の効果
/// - `policy_strict@w<W>` / `policy_all@w<W>` … P0-3 のオフライン方策。
///   **w は事前登録した有限集合を全部回す**（PR #30 レビュー2巡目 [P1]:
///   probe と continue は同じ run で同時に走るので、その run の P0-3 結果から
///   w を選ぶことはできない。1点だけ測ると「未計測の重みなら門を超えたかも
///   しれない」を否定できない）
fn policy_arm(taint: bool, w: f64) -> String {
    format!("policy_{}@w{}", if taint { "all" } else { "strict" }, fmt_w(w))
}

/// w の表示（arm タグと meta で同じ表記にする）
fn fmt_w(w: f64) -> String {
    if (w - w.round()).abs() < 1e-9 {
        format!("{}", w.round() as i64)
    } else {
        format!("{w}")
    }
}

struct Point {
    game: String,
    ply: usize,
    rep: Replayed,
    foul_tried: HashSet<String>,
    bot: Color,
    /// 真実の安全手のうち候補生成に載ったもの（オラクルの母集団）
    safe_covered: Vec<String>,
    /// そのうち今回強制したもの（`--max-safe` で絞った後）
    safe_used: Vec<String>,
    played: String,
}

fn walk_jsonl(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            walk_jsonl(&p, out);
        } else if p.extension().is_some_and(|e| e == "jsonl") {
            out.push(p);
        }
    }
}

fn collect_records(specs: &[String]) -> Vec<PathBuf> {
    let mut out = vec![];
    for s in specs {
        let p = PathBuf::from(s);
        if p.is_dir() {
            walk_jsonl(&p, &mut out);
        } else if p.exists() {
            out.push(p);
        } else {
            eprintln!("見つかりません: {s}");
        }
    }
    out.sort();
    out.dedup();
    out
}


/// 記録集合を**開始時に一度だけ**読む（名前と bytes）。
///
/// PR #30 レビュー3巡目 [P1]: 指紋を継続対局の**後**にディスクから取り直すと
/// TOCTOU になる（A を読んで30分継続した後にパスが B へ差し替わっていると、
/// 行は A 由来なのに meta は B の指紋になり、B 由来の別シャードと
/// 「同じ母集団」として集約を通ってしまう）。解析も指紋もこの bytes だけを見る。
fn read_records(files: &[PathBuf]) -> Vec<(PathBuf, Vec<u8>)> {
    files
        .iter()
        .map(|p| {
            let bytes = std::fs::read(p).unwrap_or_else(|e| die(&format!("{}: {e}", p.display())));
            (p.clone(), bytes)
        })
        .collect()
}

/// 記録集合の指紋（**ファイル名と中身の両方**）。シャードは同じ集合を見るので、
/// これが食い違う JSONL は別実験（`report` が弾く）。
///
/// PR #30 レビュー2巡目 [P2]: 名前だけを hash すると、**同名で中身の違う記録**
/// （別の Arena 実行から落とした同じ artifact 名など）が「同じ母集団」を
/// 通ってしまう。名前と bytes の境界も含めて hash する。
/// 引数は `read_records` が開始時に読んだ bytes（ディスクは読み直さない）。
fn records_fingerprint(records: &[(PathBuf, Vec<u8>)]) -> String {
    use sha2::Digest as _;
    let mut entries: Vec<(String, &[u8])> = records
        .iter()
        .map(|(p, bytes)| {
            let name = p
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            (name, bytes.as_slice())
        })
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
    let mut h = sha2::Sha256::new();
    h.update(entries.len().to_string().as_bytes());
    for (name, bytes) in &entries {
        // 長さを混ぜて名前と中身の境界を固定する（連結の曖昧さを避ける）
        h.update(name.len().to_string().as_bytes());
        h.update(b":");
        h.update(name.as_bytes());
        h.update(bytes.len().to_string().as_bytes());
        h.update(b":");
        h.update(bytes);
    }
    h.finalize().iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// `{相手: 局数}` を読みやすい1行にする
fn fmt_tally(t: &BTreeMap<String, u32>) -> String {
    if t.is_empty() {
        return "-".to_string();
    }
    t.iter()
        .map(|(k, n)| format!("{k} {n}局"))
        .collect::<Vec<_>>()
        .join(" / ")
}

/// 決定点の状態を復元する（`bin/mate_probe` と同じ規約: その手番の反則を
/// 消化した後の状態と、そこまでに試した `foul_tried`）
fn decision_state(end: &GameEndPayload, want: usize) -> Option<(Replayed, HashSet<String>)> {
    let mut out = None;
    for_each_decision_full(end, |d| {
        if d.decision_id as usize != want {
            return;
        }
        let mut foul_tried: HashSet<String> = HashSet::new();
        for e in d.logs[side_idx(d.side)].events().iter().rev() {
            match e {
                Observation::MyFoul { usi, .. } if foul_tried.len() < d.fouls_this_turn as usize => {
                    foul_tried.insert(usi.clone());
                }
                _ => break,
            }
        }
        out = Some((
            Replayed {
                pos: d.pos.clone(),
                logs: [clone_log(&d.logs[0]), clone_log(&d.logs[1])],
                fouls: *d.fouls,
                plies: d.plies,
                injected_fouls: vec![],
                oracle: None,
            },
            foul_tried,
        ));
    });
    out
}

/// 継続対局の乱数（bot 側・相手側）。**arm によらず (局, 決定点, seed) だけ**
/// から作る。
///
/// PR #30 レビュー指摘④: 強制した手を hash に混ぜると baseline / oracle /
/// policy_* が別の乱数列で継続することになり、`Δpolicy − Δpolicy(w=0)` が
/// 共通乱数のペア差にならない（強制手の効果に継続の乱数差が混ざる）。
fn continuation_seeds(game: &str, ply: usize, seed: u64) -> (u64, u64) {
    let base = stable_hash(seed, &format!("{game}#{ply}"));
    (mix(base ^ 0x0C0F_FEE0), mix(base ^ 0x0BEE_F000))
}

/// 一手強制して継続対局を回し、bot 側のスコア（勝1 / 分0.5 / 負0）を返す。
///
/// `order` は**優先順位つきの候補列**。先頭が真実で非合法なら、実対局と同じく
/// 反則を1つ積んで（手番は変わらない）次の候補へ進む。オフライン方策
/// （`Δpolicy`）は bot の信念で選ぶので**非合法な手を選びうる**ので、
/// ここを「非合法なら測定不能」にすると方策の実力を過大評価する。
/// 反則が上限に達したらその場で反則負け（スコア 0）。
///
/// 近似であることを明示しておく: 実対局なら反則の観測を食った推定器が
/// **選び直す**が、ここでは同じランキングの次点を取る（P0-3 のオフライン
/// ランキングは1回しか作らない）。
fn forced_continuation(
    p: &Point,
    order: &[String],
    seed: u64,
    opponent: &str,
) -> Option<serde_json::Value> {
    // 適用は共有規約（審判と同じ観測の記録・反則の積み方）
    let forced = force_move(&p.rep.pos, &p.rep.logs, p.rep.fouls, p.bot, order);
    let forced_fouls = forced.forced_fouls;
    let fouls = forced.fouls;
    if forced.foul_limit {
        return Some(serde_json::json!({
            "schema": ROW_SCHEMA,
            "game": p.game,
            "ply": p.ply + 1,
            "usi": "",
            "requested": order.first().cloned().unwrap_or_default(),
            "seed": seed,
            "opponent": opponent,
            "score": 0.0,
            "reason": "foul_limit",
            "plies": p.rep.plies,
            "added_plies": 0,
            "added_fouls_me": forced_fouls,
            "added_fouls_opp": 0,
            "forced_fouls": forced_fouls,
            "think_mean_ms": 0.0,
        }));
    }
    let usi = forced.played.clone()?;
    let usi = usi.as_str();
    let pos = forced.pos;
    let logs = forced.logs;

    let me_i = side_idx(p.bot);
    let (seed_me, seed_opp) = continuation_seeds(&p.game, p.ply, seed);

    let mut strats: [Option<Box<dyn strategy::Strategy>>; 2] = [None, None];
    for i in 0..2 {
        let color = if i == 0 { Color::Sente } else { Color::Gote };
        let (name, s) = if i == me_i {
            ("estimator", seed_me)
        } else {
            (opponent, seed_opp)
        };
        let mut strat = strategy::make_seeded(name, s)
            .unwrap_or_else(|| die(&format!("未知の戦略名: {name}")));
        let view = make_view(&pos, color, &fouls);
        strategy::prewarm_strategy(&mut *strat, &view, &logs[i]);
        strats[i] = Some(strat);
    }
    let start = StartState {
        pos,
        logs,
        fouls,
        plies: p.rep.plies + 1,
    };
    let strategies = [strats[0].take().unwrap(), strats[1].take().unwrap()];
    let out = play_continuation(strategies, start, 0);
    let score = match out.result {
        GameResult::Win(w) if w == p.bot => 1.0,
        GameResult::Win(_) => 0.0,
        GameResult::Draw => 0.5,
    };
    let think: Vec<u64> = out.think_us[me_i].clone();
    let mean_ms = if think.is_empty() {
        0.0
    } else {
        think.iter().sum::<u64>() as f64 / think.len() as f64 / 1000.0
    };
    Some(serde_json::json!({
        "schema": ROW_SCHEMA,
        "game": p.game,
        "ply": p.ply + 1,
        "usi": usi,
        "requested": order.first().cloned().unwrap_or_default(),
        "seed": seed,
        "opponent": opponent,
        "score": score,
        "reason": out.reason,
        "plies": out.plies,
        "added_plies": out.added_plies,
        "added_fouls_me": out.added_fouls[me_i] + forced_fouls,
        "added_fouls_opp": out.added_fouls[1 - me_i],
        "forced_fouls": forced_fouls,
        "think_mean_ms": mean_ms,
    }))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "report") {
        report_files(&args[1..]);
        return;
    }

    let mut seeds: u64 = 2;
    let mut max_safe: usize = 4;
    let mut opponent = "estimator_v14".to_string();
    // **事前登録する w の集合**（レビュー2巡目 [P1]）。単一値でも渡せる
    let mut policy_ws: Vec<f64> = vec![2.0, 4.0, 8.0, 16.0, 30.0];
    let mut jobs: usize =
        std::thread::available_parallelism().map_or(1, |n| n.get().saturating_sub(2).max(1));
    let mut shard = (0usize, 1usize);
    let mut out_path: Option<String> = None;
    let mut allow_opponent_mismatch = false;
    let mut allow_incomplete = false;
    let mut specs: Vec<String> = vec![];
    let mut i = 0;
    while i < args.len() {
        let need = |v: Option<&String>, what: &str| -> String {
            v.cloned().unwrap_or_else(|| die(&format!("{what} には値が必要です")))
        };
        match args[i].as_str() {
            "--seeds" => {
                seeds = need(args.get(i + 1), "--seeds").parse().unwrap_or_else(|_| die("--seeds は整数"));
                i += 2;
            }
            "--max-safe" => {
                max_safe = need(args.get(i + 1), "--max-safe").parse().unwrap_or_else(|_| die("--max-safe は整数"));
                i += 2;
            }
            "--opponent" => {
                opponent = need(args.get(i + 1), "--opponent");
                i += 2;
            }
            "--policy-w" => {
                policy_ws = need(args.get(i + 1), "--policy-w")
                    .split(',')
                    .filter(|t| !t.trim().is_empty())
                    .map(|t| t.trim().parse::<f64>().unwrap_or_else(|_| die("--policy-w は数値（カンマ区切りで複数可）")))
                    .collect();
                policy_ws.sort_by(|a, b| a.partial_cmp(b).unwrap());
                policy_ws.dedup();
                if policy_ws.is_empty() {
                    die("--policy-w には少なくとも1つの重みが要ります");
                }
                i += 2;
            }
            "--jobs" => {
                jobs = need(args.get(i + 1), "--jobs").parse::<usize>().unwrap_or_else(|_| die("--jobs は整数")).max(1);
                i += 2;
            }
            "--shard" => {
                let v = need(args.get(i + 1), "--shard");
                let (a, b) = v.split_once('/').unwrap_or_else(|| die("--shard は i/n の形式"));
                shard = (
                    a.parse().unwrap_or_else(|_| die("--shard は i/n の形式")),
                    b.parse().unwrap_or_else(|_| die("--shard は i/n の形式")),
                );
                if shard.1 == 0 || shard.0 >= shard.1 {
                    die("--shard は 0 <= i < n");
                }
                i += 2;
            }
            "--out" => {
                out_path = Some(need(args.get(i + 1), "--out"));
                i += 2;
            }
            "--allow-opponent-mismatch" => {
                allow_opponent_mismatch = true;
                i += 1;
            }
            "--allow-incomplete" => {
                allow_incomplete = true;
                i += 1;
            }
            s if s.starts_with("--") => die(&format!("未知のオプション: {s}")),
            s => {
                specs.push(s.to_string());
                i += 1;
            }
        }
    }
    if specs.is_empty() {
        die("記録ファイル（またはディレクトリ）を指定してください");
    }
    let files = collect_records(&specs);
    if files.is_empty() {
        die("記録ファイルが見つかりません");
    }
    // **開始時に読んだ bytes だけを使う**（解析も指紋も。PR #30 レビュー3巡目 [P1]）
    let records = read_records(&files);
    let records_fp = records_fingerprint(&records);
    if seeds % 2 != 0 {
        eprintln!("注意: --seeds が奇数だと「前半で選び後半で測る」正直版が偏ります");
    }

    let cfg = tsuitate_bot::config::ambient();
    let budget_ms = cfg.think_budget_ms;

    // ---- 対象の決定点 ----------------------------------------------------
    let mut points: Vec<Point> = vec![];
    let mut games_total = 0u32;
    let mut games_mated = 0u32;
    let mut games_with_defense = 0u32;
    let mut broken = 0u32;
    let mut dropped_safe = 0usize;
    // 記録の元相手の内訳（判定の前提を出力へ残す）と、不一致で捨てた局
    let mut record_opponents: BTreeMap<String, u32> = BTreeMap::new();
    let mut mismatched: Vec<(String, String)> = vec![];
    for (gi, (path, bytes)) in records.iter().enumerate() {
        let name = path.to_string_lossy().to_string();
        let Some((bot, end)) = std::str::from_utf8(bytes)
            .ok()
            .and_then(tsuitate_bot::truth_replay::parse_bot_and_end)
        else {
            broken += 1;
            continue;
        };
        // **元対局の相手が継続対局の相手と一致しているか**（PR #30 レビュー指摘①）。
        // ガントレットの Arena 実行は相手ごとに artifact が分かれるので、
        // `arena-records-*` をまとめて落とすと v13 相手の棋譜まで混ざり、
        // Δ が「受けの効果」と「相手が変わった効果」の混合になる
        *record_opponents.entry(end.opponent.username.clone()).or_insert(0) += 1;
        if !allow_opponent_mismatch && end.opponent.username != opponent {
            mismatched.push((name.clone(), end.opponent.username.clone()));
            continue;
        }
        let mut mates_of = |p: &Position, _ply: usize| mate_moves_in_1_fast(p);
        let Some(g) = analyze_game(&end, bot, &mut mates_of) else {
            broken += 1;
            continue;
        };
        games_total += 1;
        if !g.bot_mated {
            continue;
        }
        games_mated += 1;
        // シャードは**局単位**で割る（同じ局の arm は必ず同じシャードで走る）
        if shard.1 > 1 && gi % shard.1 != shard.0 {
            continue;
        }
        let Some(turn) = g.last_defense_point() else {
            continue;
        };
        let Some(decision_idx) = turn.decision_idx else {
            continue;
        };
        let Some((rep, foul_tried)) = decision_state(&end, decision_idx) else {
            continue;
        };
        let game = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| name.clone());
        // 漏斗の2段目: 真実の安全手のうち bot の候補生成に載るものだけが
        // オラクルの母集団（載らない手は方策が原理的に選べない）
        let view = make_view(&rep.pos, bot, &rep.fouls);
        let cands: HashSet<String> = candidate_moves(&view, &foul_tried)
            .into_iter()
            .map(|(usi, _)| usi)
            .collect();
        let mut safe_covered: Vec<String> = turn
            .safe
            .iter()
            .map(ShogiMove::to_usi)
            .filter(|u| cands.contains(u))
            .collect();
        safe_covered.sort();
        if safe_covered.is_empty() {
            continue;
        }
        games_with_defense += 1;
        // **黙って切らない**: 上限で落とした本数を記録して最後に出す。
        // 選び方は決定論的（USI の辞書順で等間隔）: 現行方策の順位で選ぶと
        // オラクルが現行方策に引きずられる
        let mut safe_used = safe_covered.clone();
        if safe_used.len() > max_safe && max_safe > 0 {
            let step = safe_used.len() as f64 / max_safe as f64;
            safe_used = (0..max_safe)
                .map(|k| safe_covered[((k as f64 * step) as usize).min(safe_covered.len() - 1)].clone())
                .collect();
            safe_used.dedup();
            dropped_safe += safe_covered.len() - safe_used.len();
        }
        let played = end
            .moves
            .get(decision_idx)
            .map(|m| m.usi.clone())
            .unwrap_or_default();
        points.push(Point {
            game,
            ply: decision_idx,
            rep,
            foul_tried,
            bot,
            safe_covered,
            safe_used,
            played,
        });
    }

    if !mismatched.is_empty() {
        eprintln!("元対局の相手が --opponent {opponent} と一致しない記録が {} 件あります:", mismatched.len());
        for (name, opp) in mismatched.iter().take(5) {
            eprintln!("  {name}: 相手 {opp}");
        }
        eprintln!("記録の相手の内訳: {}", fmt_tally(&record_opponents));
        die(
            "**Δ が「受けの効果」と「相手が変わった効果」の混合になる**ので中止しました。\n\
             records_pattern を相手で絞る（例 arena-records-estimator_v14-*）か、\n\
             承知のうえで混ぜるなら --allow-opponent-mismatch を付けてください",
        );
    }
    println!(
        "記録 {} 件（壊れ {broken}）/ 局 {games_total}（詰み負け {games_mated}・受けが候補にあった {games_with_defense}）\n\
         記録の相手: {}",
        records.len(),
        fmt_tally(&record_opponents),
    );
    println!(
        "対象の決定点 {}（--max-safe {max_safe} で落とした安全手 {dropped_safe}本）/ 相手 {opponent} / \
         思考予算 {budget_ms}ms / seeds {seeds} / jobs {jobs} / shard {}/{} / policy_w {}",
        points.len(),
        shard.0,
        shard.1,
        policy_ws.iter().map(|w| fmt_w(*w)).collect::<Vec<_>>().join(","),
    );
    if points.is_empty() {
        die("継続する決定点がありません");
    }

    // ---- P0-3 のオフライン方策が選ぶ手（seed ごと）------------------------
    // ランキングと粒子は `bin/mate_probe` と同じ構築（同じ seed・同じ scale）
    let policy_units: Vec<(usize, u64)> = (0..points.len())
        .flat_map(|p| (0..seeds).map(move |s| (p, s)))
        .collect();
    // 方策は argmax だけでなく**順序**を持つ（先頭が非合法なら次点へ = 実対局の反則）
    // (決定点, seed) → arm タグ → 強制する候補列
    let policy: Arc<Mutex<HashMap<(usize, u64), BTreeMap<String, Vec<String>>>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let points_ref = Arc::new(points);
    // ランキング/q 生成の**実効並列度**は継続対局とは別（決定点×seed の数で
    // clamp される）。壁時計予算なのでここも探索量を変える実験条件になる
    // （PR #30 レビュー3巡目 [P2]: 疎なシャードでは policy 2並列・継続 3並列の
    // ように食い違い、`jobs` 1つだと別条件のシャードが同じ値で集約を通る）
    let policy_jobs = jobs.min(policy_units.len().max(1)).max(1);
    run_parallel(jobs, policy_units.len(), |ui| {
        let (pi, seed) = policy_units[ui];
        let p = &points_ref[pi];
        // **ランキングを作ったのと同じ粒子**で q を測る（PR #30 レビュー指摘②）
        let Some((_, ranking, snapshot)) =
            ranking_and_particles(&p.rep, seed, "estimator", &p.foul_tried)
        else {
            return;
        };
        let particles = snapshot.entries();
        let rows = tsuitate_bot::mate_economy::build_rows(&p.rep.pos, &ranking, &particles);
        // 反則上限（10回）を超えて並べても意味が無いので先頭 12 本で足りる
        let order = |w: f64, taint: bool| {
            let mut v = rescored_usis(&rows, w, taint);
            v.truncate(12);
            v
        };
        let mut by_arm: BTreeMap<String, Vec<String>> = BTreeMap::new();
        by_arm.insert("policy_w0".to_string(), order(0.0, false));
        for &w in &policy_ws {
            by_arm.insert(policy_arm(false, w), order(w, false));
            by_arm.insert(policy_arm(true, w), order(w, true));
        }
        by_arm.retain(|_, v| !v.is_empty());
        policy.lock().unwrap().insert((pi, seed), by_arm);
    });
    let policy = Arc::try_unwrap(policy).ok().unwrap().into_inner().unwrap();

    // ---- 継続対局の unit ---------------------------------------------------
    // **同じ候補列は1回だけ走らせて、結果を該当する arm 全部へ配る**。
    // 継続の乱数は (局, 決定点, seed) だけから作るので、候補列が同じなら
    // 継続は完全に同一（近似ではなく厳密な重複排除）。w を5点に増やしても
    // 実際に走る継続数は「相異なる候補列の数」までしか増えない
    let mut units: Vec<(usize, u64, Vec<String>, Vec<String>)> = vec![];
    for (pi, p) in points_ref.iter().enumerate() {
        for seed in 0..seeds {
            let mut by_order: BTreeMap<Vec<String>, Vec<String>> = BTreeMap::new();
            by_order
                .entry(vec![p.played.clone()])
                .or_default()
                .push("baseline".to_string());
            for usi in &p.safe_used {
                by_order
                    .entry(vec![usi.clone()])
                    .or_default()
                    .push("oracle".to_string());
            }
            if let Some(by_arm) = policy.get(&(pi, seed)) {
                for (arm, order) in by_arm {
                    by_order.entry(order.clone()).or_default().push(arm.clone());
                }
            }
            for (order, arms) in by_order {
                units.push((pi, seed, order, arms));
            }
        }
    }
    let lines: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(vec![]));
    // **実効並列度**（壁時計予算なので、同時に走る unit 数は探索量を変える
    // = 実験条件。issue #24 の教訓と同じ）
    let continuation_jobs = jobs.min(units.len().max(1)).max(1);
    let started = std::time::Instant::now();
    {
        let units = &units;
        let points_ref = Arc::clone(&points_ref);
        let lines = Arc::clone(&lines);
        let opponent_ref = opponent.as_str();
        run_parallel(jobs, units.len(), move |ui| {
            let (pi, seed, order, arms) = &units[ui];
            let p = &points_ref[*pi];
            if let Some(v) = forced_continuation(p, order, *seed, opponent_ref) {
                let mut out = lines.lock().unwrap();
                for arm in arms {
                    // 同じ候補列 = 同じ継続なので、arm ごとに同じ結果を配る
                    let mut row = v.clone();
                    row["arm"] = serde_json::json!(arm);
                    row["safe_covered"] = serde_json::json!(p.safe_covered.len());
                    row["safe_used"] = serde_json::json!(p.safe_used.len());
                    out.push(row);
                }
            }
        });
    }
    let mut lines = Arc::try_unwrap(lines).ok().unwrap().into_inner().unwrap();
    lines.sort_by_key(|v| {
        (
            v["game"].as_str().unwrap_or("").to_string(),
            v["arm"].as_str().unwrap_or("").to_string(),
            v["usi"].as_str().unwrap_or("").to_string(),
            v["seed"].as_u64().unwrap_or(0),
        )
    });
    eprintln!(
        "継続 {} 局 / {:.1}分",
        lines.len(),
        started.elapsed().as_secs_f64() / 60.0
    );

    // **判定に効く実験キー**（PR #30 レビュー指摘③）。`report` は全シャードで
    // これが完全一致していることを要求する。局数だけ合っていれば混ざる、
    // という状態を無くすため相手・予算・seed 数・方策・実効並列度・
    // コード版・記録集合の指紋まで入れる（壁時計計測なので jobs も実験条件。
    // ランキング段と継続段は clamp 先が違うので**別々に**残す）
    let experiment = serde_json::json!({
        "opponent": opponent,
        "budget_ms": budget_ms,
        "seeds": seeds,
        "max_safe": max_safe,
        "policy_ws": policy_ws.iter().map(|w| fmt_w(*w)).collect::<Vec<_>>(),
        "policy_jobs": policy_jobs,
        "continuation_jobs": continuation_jobs,
        "games": games_total,
        "shard_total": shard.1,
        "config": cfg.fingerprint(),
        "source_fingerprint": env!("TSUITATE_SOURCE_FINGERPRINT"),
        "records": records_fp,
    });
    let meta = serde_json::json!({
        "schema": ROW_SCHEMA,
        "type": "meta",
        "experiment": experiment,
        "shard": shard.0,
        "games": games_total,
        "games_mated": games_mated,
        "games_with_defense": games_with_defense,
        "points": points_ref.len(),
        // **期待する行数を meta 自身に残す**（PR #30 レビュー2巡目 [P1]）。
        // 「入力に現れた (局, seed) の中で arm が揃っているか」だけを見ると、
        // ある seed の全 arm がまとめて欠けたときに検出できない
        "points_detail": points_ref
            .iter()
            .map(|p| serde_json::json!({
                "game": p.game,
                "ply": p.ply + 1,
                "safe_used": p.safe_used.len(),
                "safe_covered": p.safe_covered.len(),
            }))
            .collect::<Vec<_>>(),
        "dropped_safe": dropped_safe,
        "record_opponents": record_opponents,
    });
    if let Some(path) = &out_path {
        let mut s = format!("{meta}\n");
        for l in &lines {
            s.push_str(&format!("{l}\n"));
        }
        match std::fs::write(path, s) {
            Ok(()) => println!("JSONL: {path}"),
            Err(e) => eprintln!("{path}: 書けません: {e}"),
        }
    }
    report(&[meta], &lines, allow_incomplete);
}

/// `n` 個の unit を `jobs` 本のスレッドで回す（実効並列度は unit 数で clamp）
fn run_parallel(jobs: usize, n: usize, f: impl Fn(usize) + Sync + Send) {
    if n == 0 {
        return;
    }
    let next = Arc::new(Mutex::new(0usize));
    let effective = jobs.min(n).max(1);
    let f = &f;
    std::thread::scope(|scope| {
        for _ in 0..effective {
            let next = Arc::clone(&next);
            scope.spawn(move || {
                loop {
                    let ui = {
                        let mut g = next.lock().unwrap();
                        if *g >= n {
                            break;
                        }
                        let v = *g;
                        *g += 1;
                        v
                    };
                    f(ui);
                }
            });
        }
    });
}

fn report_files(args: &[String]) {
    let allow_incomplete = args.iter().any(|a| a == "--allow-incomplete");
    let paths: Vec<String> = args
        .iter()
        .filter(|a| !a.starts_with("--"))
        .cloned()
        .collect();
    if paths.is_empty() {
        die("report には JSONL を指定してください");
    }
    let mut metas = vec![];
    let mut rows = vec![];
    let mut seen_schema: Option<u32> = None;
    for p in &paths {
        let Ok(text) = std::fs::read_to_string(p) else {
            die(&format!("{p}: 読めません"));
        };
        for line in text.lines() {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
                die(&format!("{p}: JSON として読めない行があります"));
            };
            let schema = v["schema"].as_u64().unwrap_or(0) as u32;
            if !READABLE_SCHEMAS.contains(&schema) {
                die(&format!(
                    "{p}: schema {schema} は読めません（読めるのは {READABLE_SCHEMAS:?}。\
撤回済みの記録は混ぜない）"
                ));
            }
            match seen_schema {
                Some(s) if s != schema => die(&format!(
                    "{p}: schema {schema} と {s} を混ぜています（実験キーの粒度が違う）"
                )),
                _ => seen_schema = Some(schema),
            }
            if v["type"] == "meta" {
                metas.push(v);
            } else {
                rows.push(v);
            }
        }
    }
    report(&metas, &rows, allow_incomplete);
}

/// `report` の入力契約（PR #30 レビュー指摘③）。破っている点を全部返す。
///
/// - `experiment`（相手・予算・seed 数・max_safe・policy_w・実効並列度・
///   コード版・記録集合・全対局数）が全シャードで一致しているか。
///   **局数だけ合っていれば違う実験を混ぜられる**状態にしない
/// - 同じシャードの JSONL を2回渡していないか（行が二重に数えられる）
/// - 重複行（同じ 局・決定点・arm・手・seed）が無いか
/// - baseline がある (局, seed) に policy 3種も揃っているか
///   （欠けたぶんだけ `Δpolicy` が下振れする）
fn check_inputs(metas: &[serde_json::Value], rows: &[serde_json::Value]) -> Vec<String> {
    let mut out = vec![];
    let experiments: Vec<&serde_json::Value> = metas.iter().map(|m| &m["experiment"]).collect();
    if let Some(first) = experiments.first() {
        for (i, e) in experiments.iter().enumerate().skip(1) {
            if *e != *first {
                let diff: Vec<String> = first
                    .as_object()
                    .map(|o| {
                        o.keys()
                            .filter(|k| first[k.as_str()] != e[k.as_str()])
                            .map(|k| format!("{k}: {} vs {}", first[k.as_str()], e[k.as_str()]))
                            .collect()
                    })
                    .unwrap_or_default();
                out.push(format!(
                    "meta の実験キーが {i} 本目で食い違います（違う実験を混ぜている）: {}",
                    diff.join(" / ")
                ));
            }
        }
    }
    let mut shards: Vec<u64> = metas.iter().filter_map(|m| m["shard"].as_u64()).collect();
    let n = shards.len();
    shards.sort_unstable();
    shards.dedup();
    if shards.len() != n {
        out.push("同じシャードの JSONL を2回渡しています（行が二重に数えられる）".to_string());
    }

    let mut keys: HashSet<(String, u64, String, String, u64)> = HashSet::new();
    let mut dups = 0usize;
    let mut present: BTreeMap<(String, u64), HashSet<String>> = BTreeMap::new();
    for r in rows {
        let game = r["game"].as_str().unwrap_or("?").to_string();
        let arm = r["arm"].as_str().unwrap_or("?").to_string();
        let seed = r["seed"].as_u64().unwrap_or(0);
        let key = (
            game.clone(),
            r["ply"].as_u64().unwrap_or(0),
            arm.clone(),
            r["usi"].as_str().unwrap_or("").to_string(),
            seed,
        );
        if !keys.insert(key) {
            dups += 1;
        }
        present.entry((game, seed)).or_default().insert(arm);
    }
    if dups > 0 {
        out.push(format!(
            "重複行が {dups} 件あります（同じ arm・同じ seed の継続が二重）"
        ));
    }
    // 期待する arm は meta の `policy_ws` から作る（w を増やしても検査が追随する）
    let mut want: Vec<String> = vec!["baseline".to_string(), "policy_w0".to_string()];
    for w in metas
        .first()
        .and_then(|m| m["experiment"]["policy_ws"].as_array())
        .into_iter()
        .flatten()
    {
        let w = w.as_str().unwrap_or_default();
        want.push(format!("policy_strict@w{w}"));
        want.push(format!("policy_all@w{w}"));
    }
    let missing: Vec<String> = present
        .iter()
        .filter_map(|((g, s), arms)| {
            let lack: Vec<&str> = want
                .iter()
                .map(String::as_str)
                .filter(|a| !arms.contains(*a))
                .collect();
            (!lack.is_empty()).then(|| format!("{g}#seed{s}: {}", lack.join(",")))
        })
        .collect();
    if !missing.is_empty() {
        out.push(format!(
            "arm が欠けている (局, seed) が {} 件あります（Δpolicy がそのぶん下振れする）: {}{}",
            missing.len(),
            missing.iter().take(3).cloned().collect::<Vec<_>>().join(" / "),
            if missing.len() > 3 { " ..." } else { "" }
        ));
    }

    // **meta が宣言した期待行数と突き合わせる**（PR #30 レビュー2巡目 [P1]）。
    // 上の検査は「入力に現れた (局, seed)」しか見ないので、ある seed の全 arm が
    // まとめて欠けても素通りする（実測: 1局1 seed ぶん 8行を落とすと
    // Δoracle 0.1106 → 0.1058、Δpolicy 0.0240 → 0.0192 と動くのに通ってしまう）
    let seeds = metas
        .first()
        .and_then(|m| m["experiment"]["seeds"].as_u64())
        .unwrap_or(0);
    let mut want_points: BTreeMap<(String, u64), u64> = BTreeMap::new();
    for m in metas {
        for d in m["points_detail"].as_array().into_iter().flatten() {
            let key = (
                d["game"].as_str().unwrap_or("?").to_string(),
                d["ply"].as_u64().unwrap_or(0),
            );
            want_points.insert(key, d["safe_used"].as_u64().unwrap_or(0));
        }
    }
    if seeds > 0 && !want_points.is_empty() {
        // (局, ply, arm) → 見えた seed
        let mut seen_rows: BTreeMap<(String, u64, String), Vec<u64>> = BTreeMap::new();
        for r in rows {
            seen_rows
                .entry((
                    r["game"].as_str().unwrap_or("?").to_string(),
                    r["ply"].as_u64().unwrap_or(0),
                    r["arm"].as_str().unwrap_or("?").to_string(),
                ))
                .or_default()
                .push(r["seed"].as_u64().unwrap_or(0));
        }
        let mut lacks: Vec<String> = vec![];
        for ((game, ply), safe_used) in &want_points {
            for arm in &want {
                let got = seen_rows
                    .get(&(game.clone(), *ply, arm.clone()))
                    .map_or(0, |v| v.len() as u64);
                if got != seeds {
                    lacks.push(format!("{game}#{ply} {arm}: {got}/{seeds}"));
                }
            }
            // oracle は「強制した安全手の本数 × seed 数」だけあるはず
            let got = seen_rows
                .get(&(game.clone(), *ply, "oracle".to_string()))
                .map_or(0, |v| v.len() as u64);
            let expect = safe_used * seeds;
            if got != expect {
                lacks.push(format!("{game}#{ply} oracle: {got}/{expect}"));
            }
        }
        if !lacks.is_empty() {
            out.push(format!(
                "meta が宣言した決定点に対して行が {} 箇所欠けています（Δ の分子が欠ける）: {}{}",
                lacks.len(),
                lacks.iter().take(3).cloned().collect::<Vec<_>>().join(" / "),
                if lacks.len() > 3 { " ..." } else { "" }
            ));
        }
    }
    out
}

/// 局ごとの平均スコアを arm 別に畳み、Δ と cluster bootstrap CI を出す
fn report(metas: &[serde_json::Value], rows: &[serde_json::Value], allow_incomplete: bool) {
    if metas.is_empty() {
        eprintln!("meta 行がありません（Δ の分母が取れない）");
        return;
    }
    // 入力の契約（PR #30 レビュー指摘③）。破っていたら判定を出さない
    for msg in check_inputs(metas, rows) {
        if allow_incomplete {
            eprintln!("警告: {msg}");
        } else {
            die(&msg);
        }
    }
    let exp = metas
        .first()
        .map(|m| m["experiment"].clone())
        .unwrap_or_default();
    let games_total: u64 = exp["games"].as_u64().unwrap_or(0);
    let games_mated: u64 = metas
        .iter()
        .map(|m| m["games_mated"].as_u64().unwrap_or(0))
        .max()
        .unwrap_or(0);
    // シャードが揃っているか（Δ の分母は全対局数なので、1シャードだけ集計すると
    // 分母はそのままで分子だけが欠ける = **Δ が机上で小さくなる**）。
    // 揃っていなければ判定を出さない
    let shard_total = exp["shard_total"].as_u64().unwrap_or(1) as usize;
    let mut seen: Vec<usize> = metas
        .iter()
        .filter_map(|m| m["shard"].as_u64().map(|v| v as usize))
        .collect();
    seen.sort_unstable();
    seen.dedup();
    let shards_complete = seen.len() == shard_total;
    // 継続した局と落とした安全手はシャードで**割った**ぶんなので合計する
    let dropped: u64 = metas
        .iter()
        .map(|m| m["dropped_safe"].as_u64().unwrap_or(0))
        .sum();

    if games_total == 0 {
        eprintln!("meta の games が 0 です（Δ の分母が取れない）");
        return;
    }
    // game → arm → usi → seed → score
    type Bucket = BTreeMap<String, BTreeMap<String, BTreeMap<u64, f64>>>;
    let mut per_game: BTreeMap<String, Bucket> = BTreeMap::new();
    for r in rows {
        let game = r["game"].as_str().unwrap_or("?").to_string();
        let arm = r["arm"].as_str().unwrap_or("?").to_string();
        let usi = r["usi"].as_str().unwrap_or("?").to_string();
        let seed = r["seed"].as_u64().unwrap_or(0);
        let score = r["score"].as_f64().unwrap_or(0.0);
        per_game
            .entry(game)
            .or_default()
            .entry(arm)
            .or_default()
            .entry(usi)
            .or_default()
            .insert(seed, score);
    }

    let mean = |v: &[f64]| -> f64 {
        if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
    };
    // 局ごとの寄与（0 = 受けの機会が無かった局）
    let mut contrib: Vec<Contrib> = vec![];
    for (game, arms) in &per_game {
        let arm_mean = |tag: &str| -> f64 {
            arms.get(tag).map_or(0.0, |by_usi| {
                let all: Vec<f64> = by_usi.values().flat_map(|m| m.values().copied()).collect();
                mean(&all)
            })
        };
        let oracle = arms.get("oracle").map_or(0.0, |by_usi| {
            by_usi
                .values()
                .map(|m| mean(&m.values().copied().collect::<Vec<_>>()))
                .fold(0.0, f64::max)
        });
        // 正直版: seed を前半（選ぶ）/ 後半（測る）に割る
        let oracle_honest = arms.get("oracle").map_or(0.0, |by_usi| {
            let seeds: Vec<u64> = by_usi
                .values()
                .flat_map(|m| m.keys().copied())
                .collect::<HashSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            let mut seeds = seeds;
            seeds.sort_unstable();
            if seeds.len() < 2 {
                return f64::NAN;
            }
            let half = seeds.len() / 2;
            let (train, test) = seeds.split_at(half);
            let best = by_usi.iter().max_by(|a, b| {
                let sa = mean(&train.iter().filter_map(|s| a.1.get(s).copied()).collect::<Vec<_>>());
                let sb = mean(&train.iter().filter_map(|s| b.1.get(s).copied()).collect::<Vec<_>>());
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal).then_with(|| b.0.cmp(a.0))
            });
            best.map_or(f64::NAN, |(_, m)| {
                mean(&test.iter().filter_map(|s| m.get(s).copied()).collect::<Vec<_>>())
            })
        });
        // 方策 arm は w ごとに増えるので、名前で引ける形で持つ
        let policy: BTreeMap<String, f64> = arms
            .keys()
            .filter(|a| a.starts_with("policy"))
            .map(|a| (a.clone(), arm_mean(a)))
            .collect();
        contrib.push(Contrib {
            baseline: arm_mean("baseline"),
            oracle,
            oracle_honest,
            policy,
        });
        let _ = game;
    }

    let g = games_total as f64;
    let sum = |f: &dyn Fn(&Contrib) -> f64| -> f64 {
        contrib.iter().map(|c| f(c)).filter(|v| v.is_finite()).sum::<f64>() / g
    };
    let d_base = sum(&|c| c.baseline);
    let d_oracle = sum(&|c| c.oracle);
    let d_honest = sum(&|c| c.oracle_honest);
    let policy_of = |arm: &str| -> f64 {
        let arm = arm.to_string();
        sum(&move |c: &Contrib| c.policy.get(&arm).copied().unwrap_or(0.0))
    };
    let d_w0 = policy_of("policy_w0");

    println!("\n=== P0-6: 一手強制の継続診断 ===");
    // 並列度は**段ごと**に記録する（レビュー3巡目 [P2]）。段別の記録が入る前に
    // 取った JSONL は `jobs` 1本しか持たないので、そのときはそう明示して出す
    // （掲載済みの実測 artifact をそのまま再集計できるようにしておく）
    let jobs_disp = if exp["policy_jobs"].is_null() && exp["continuation_jobs"].is_null() {
        format!("{}（段別の記録なし）", exp["jobs"])
    } else {
        format!("{}/{}", exp["policy_jobs"], exp["continuation_jobs"])
    };
    println!(
        "  実験: 相手 {} / 予算 {}ms / seeds {} / max_safe {} / policy_w {} / \
jobs(ランキング/継続) {jobs_disp} / 記録 {} / code {}",
        exp["opponent"], exp["budget_ms"], exp["seeds"], exp["max_safe"], exp["policy_ws"],
        exp["records"], exp["source_fingerprint"],
    );
    println!(
        "全 {games_total}局（詰み負け {games_mated}・継続した局 {}）/ 継続 {} 局ぶん{}",
        contrib.len(),
        rows.len(),
        if dropped > 0 {
            format!("・--max-safe で落とした安全手 {dropped}本（Δoracle は下界）")
        } else {
            String::new()
        }
    );
    // 受けの機会が無かった局は寄与 0 の cluster として**必ず標本に入れる**
    // （再標本化を「受けがあった局」だけに限ると分散が過小になる）
    let ci = |f: &dyn Fn(&Contrib) -> f64| -> (f64, f64) {
        cluster_bootstrap(&contrib, f, games_total as usize)
    };
    let (lo_o, hi_o) = ci(&|c| c.oracle);
    let (lo_b, hi_b) = ci(&|c| c.baseline);
    let (lo_w0, hi_w0) = ci(&|c| c.policy.get("policy_w0").copied().unwrap_or(0.0));
    println!("  baseline（実戦の手を強制して指し直し）: {:+.4} [{lo_b:+.4}, {hi_b:+.4}]", d_base);
    println!("  Δoracle（最良の安全手・楽観）:          {:+.4} [{lo_o:+.4}, {hi_o:+.4}]", d_oracle);
    println!("  Δoracle（前半で選び後半で測る正直版）:  {:+.4}", d_honest);
    println!("  対照 w=0（同じ経路で再決定するだけ）:   {:+.4} [{lo_w0:+.4}, {hi_w0:+.4}]", d_w0);
    // **事前登録した w を全部出す**（レビュー2巡目 [P1]: 1点だけ測ると
    // 「未計測の重みなら門を超えたかもしれない」を否定できない）
    let mut policy_arms: Vec<String> = contrib
        .iter()
        .flat_map(|c| c.policy.keys().cloned())
        .filter(|a| a != "policy_w0")
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    policy_arms.sort_by_key(|a| {
        // policy_strict@w4 → (strict/all, w)
        let (kind, w) = a.split_once("@w").unwrap_or((a.as_str(), "0"));
        (
            kind.to_string(),
            (w.parse::<f64>().unwrap_or(0.0) * 1000.0) as i64,
        )
    });
    println!("  Δpolicy（事前登録した w を全部・w=0 対照との差つき）:");
    let mut best: Option<(String, f64)> = None;
    for arm in &policy_arms {
        let d = policy_of(arm);
        let (lo, hi) = {
            let a = arm.clone();
            ci(&move |c: &Contrib| c.policy.get(&a).copied().unwrap_or(0.0))
        };
        println!(
            "    {arm:<22} {d:+.4} [{lo:+.4}, {hi:+.4}]（w=0 対照との差 {:+.4}）",
            d - d_w0
        );
        if best.as_ref().is_none_or(|(_, b)| d > *b) {
            best = Some((arm.clone(), d));
        }
    }
    if shards_complete {
        println!(
            "  判定: Δoracle {} 0.04 → {}",
            if d_oracle < 0.04 { "<" } else { "≥" },
            if d_oracle < 0.04 {
                "**即中止**（受けても救われないので受け項は律速でない）"
            } else {
                "P1 へ進む必要条件のうち Δoracle は満たす"
            }
        );
        if let Some((arm, d)) = &best {
            println!(
                "        Δpolicy の最大は {arm} の {d:+.4} → {}",
                if *d < 0.04 {
                    "**事前登録した w のどれでも必要条件 0.04 に届かない**（この形の P1 は終了）"
                } else {
                    "門を超える w がある。**選定バイアスに注意**（未使用 seed で確認してから採否を決めること）"
                }
            );
        }
    } else {
        println!(
            "  判定: **出せない**（シャード {}/{} しか揃っていない。Δ の分母は全対局数なので、\
             欠けたシャードのぶんだけ分子だけが落ちて Δ が過小に出る）",
            seen.len(),
            shard_total
        );
    }
    println!(
        "  ※ baseline が 0 から離れているぶんは「指し直したから」の差。Δ から引いて読むこと"
    );

    // 終局理由の内訳（機構が壊れていないかの確認）
    let mut reasons: BTreeMap<(String, String), u32> = BTreeMap::new();
    let mut think: Vec<f64> = vec![];
    let mut fouls: Vec<f64> = vec![];
    for r in rows {
        let arm = r["arm"].as_str().unwrap_or("?").to_string();
        let reason = r["reason"].as_str().unwrap_or("?").to_string();
        *reasons.entry((arm, reason)).or_insert(0) += 1;
        think.push(r["think_mean_ms"].as_f64().unwrap_or(0.0));
        fouls.push(r["added_fouls_me"].as_f64().unwrap_or(0.0));
    }
    let mut arm_rows: BTreeMap<String, u32> = BTreeMap::new();
    for r in rows {
        *arm_rows
            .entry(r["arm"].as_str().unwrap_or("?").to_string())
            .or_insert(0) += 1;
    }
    println!(
        "  arm ごとの継続数: {}（方策が取れなかった決定点×seed は 0 として数えるので、\
         policy_* が baseline より少ないときは Δpolicy がそのぶん下振れする）",
        arm_rows
            .iter()
            .map(|(a, n)| format!("{a} {n}"))
            .collect::<Vec<_>>()
            .join(" / ")
    );
    println!("  終局理由:");
    for ((arm, reason), n) in &reasons {
        println!("    {arm} / {reason}: {n}");
    }
    let forced: Vec<f64> = rows
        .iter()
        .map(|r| r["forced_fouls"].as_f64().unwrap_or(0.0))
        .collect();
    println!(
        "  継続の平均: 思考 {:.0}ms / 追加反則 {:.2}回（うち強制手が非合法で焼いた反則 {:.2}回）",
        mean(&think),
        mean(&fouls),
        mean(&forced)
    );
}

/// 元対局単位の cluster bootstrap（percentile 95% CI）。
/// seed は独立標本として数えない（checkpoint arena と同じ規約）。
/// 局ごとの寄与（arm 別の平均スコア）。cluster bootstrap の統計単位
struct Contrib {
    baseline: f64,
    oracle: f64,
    oracle_honest: f64,
    /// arm 名（`policy_w0` / `policy_strict@w4` …）→ その局の平均スコア
    policy: BTreeMap<String, f64>,
}

fn cluster_bootstrap(
    contrib: &[Contrib],
    f: &dyn Fn(&Contrib) -> f64,
    games_total: usize,
) -> (f64, f64) {
    if contrib.is_empty() || games_total == 0 {
        return (f64::NAN, f64::NAN);
    }
    let mut vals: Vec<f64> = contrib
        .iter()
        .map(|c| f(c))
        .map(|v| if v.is_finite() { v } else { 0.0 })
        .collect();
    // 受けの機会が無かった局（寄与 0）も cluster として数える
    vals.resize(vals.len().max(games_total), 0.0);
    let denom = vals.len() as f64;
    // 決定論的な線形合同法（同じ入力なら同じ CI）
    let mut state: u64 = 0x2026_0827;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let reps = 2000;
    let mut draws: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let mut s = 0.0;
        for _ in 0..vals.len() {
            s += vals[next() % vals.len()];
        }
        draws.push(s / denom);
    }
    draws.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    (draws[(reps as f64 * 0.025) as usize], draws[(reps as f64 * 0.975) as usize])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(exp: serde_json::Value, shard: u64) -> serde_json::Value {
        serde_json::json!({
            "schema": ROW_SCHEMA, "type": "meta", "experiment": exp, "shard": shard,
            "games_mated": 2, "dropped_safe": 0, "points": 1,
            "points_detail": [{"game": "g1", "ply": 90, "safe_used": 1, "safe_covered": 3}],
        })
    }

    fn exp_of(opponent: &str) -> serde_json::Value {
        serde_json::json!({
            "opponent": opponent, "budget_ms": 700, "seeds": 2, "max_safe": 4,
            "policy_ws": ["4", "8"], "policy_jobs": 3, "continuation_jobs": 3,
            "games": 104, "shard_total": 2,
            "config": "cfg", "source_fingerprint": "src", "records": "rec",
        })
    }

    fn row(game: &str, arm: &str, usi: &str, seed: u64) -> serde_json::Value {
        serde_json::json!({
            "schema": ROW_SCHEMA, "game": game, "ply": 90, "arm": arm, "usi": usi,
            "seed": seed, "score": 0.0, "reason": "checkmate",
        })
    }

    /// meta の期待（決定点1つ・safe_used 1本）と整合する1 seed ぶんの行
    fn full_arms(game: &str, seed: u64) -> Vec<serde_json::Value> {
        [
            "baseline",
            "policy_w0",
            "policy_strict@w4",
            "policy_all@w4",
            "policy_strict@w8",
            "policy_all@w8",
            "oracle",
        ]
            .iter()
            .map(|a| row(game, a, "7g7f", seed))
            .collect()
    }

    /// seeds=2 ぶんの完全な入力
    fn full_input(game: &str) -> Vec<serde_json::Value> {
        let mut v = full_arms(game, 0);
        v.extend(full_arms(game, 1));
        v
    }

    /// 継続の乱数は arm（強制手）に依らない = 共通乱数でペアになる
    #[test]
    fn 継続の乱数は強制手に依存しない() {
        let a = continuation_seeds("game1", 90, 0);
        let b = continuation_seeds("game1", 90, 0);
        assert_eq!(a, b, "同じ (局, 決定点, seed) なら同じ乱数");
        assert_ne!(a, continuation_seeds("game1", 90, 1), "seed が違えば変わる");
        assert_ne!(a, continuation_seeds("game1", 92, 0), "決定点が違えば変わる");
        assert_ne!(a, continuation_seeds("game2", 90, 0), "局が違えば変わる");
        assert_ne!(a.0, a.1, "自分と相手のシードは別");
    }

    /// 実験キーが違う JSONL は混ぜられない（局数だけ合っていても止める）
    #[test]
    fn 違う実験のjsonlは混ぜられない() {
        let rows: Vec<serde_json::Value> = full_input("g1");
        // 2シャードぶんの meta は同じ決定点を宣言しない（各シャードは別の局を持つ）ので、
        // ここでは1シャードぶんの meta を2回渡さず、実験キーの一致だけを見る
        let ok = vec![meta(exp_of("estimator_v14"), 0)];
        assert!(check_inputs(&ok, &rows).is_empty(), "同じ実験なら通る");

        let mixed = vec![meta(exp_of("estimator_v14"), 0), meta(exp_of("estimator_v13"), 1)];
        let problems = check_inputs(&mixed, &rows);
        assert!(
            problems.iter().any(|p| p.contains("opponent")),
            "違うキーを名指しする: {problems:?}"
        );
    }

    /// 同じシャードを2回渡す・重複行・arm の欠落を検出する
    #[test]
    fn 重複と欠落を検出する() {
        let metas = vec![meta(exp_of("estimator_v14"), 0), meta(exp_of("estimator_v14"), 0)];
        let problems = check_inputs(&metas, &full_input("g1"));
        assert!(problems.iter().any(|p| p.contains("2回")), "{problems:?}");

        let metas = vec![meta(exp_of("estimator_v14"), 0)];
        let mut rows = full_input("g1");
        rows.push(row("g1", "baseline", "7g7f", 0));
        let problems = check_inputs(&metas, &rows);
        assert!(problems.iter().any(|p| p.contains("重複行が 1 件")), "{problems:?}");

        // w ごとの arm が欠けた (局, seed) も検出する
        let rows: Vec<serde_json::Value> = full_input("g1")
            .into_iter()
            .filter(|r| !(r["arm"] == "policy_all@w8" && r["seed"] == 0))
            .collect();
        let problems = check_inputs(&metas, &rows);
        assert!(problems.iter().any(|p| p.contains("policy_all@w8")), "{problems:?}");
    }

    /// **ある seed の全 arm がまとめて欠けても検出する**（レビュー2巡目 [P1]）。
    /// 「入力に現れた (局, seed) の中で arm が揃うか」だけでは素通りする穴
    #[test]
    fn seedごと欠けた行を検出する() {
        let metas = vec![meta(exp_of("estimator_v14"), 0)];
        assert!(
            check_inputs(&metas, &full_input("g1")).is_empty(),
            "揃っていれば通る"
        );

        // seed 0 の行を丸ごと落とす（(局, seed) のキー自体が現れない）
        let rows: Vec<serde_json::Value> = full_input("g1")
            .into_iter()
            .filter(|r| r["seed"] != 0)
            .collect();
        let problems = check_inputs(&metas, &rows);
        assert!(
            problems.iter().any(|p| p.contains("meta が宣言した決定点")),
            "{problems:?}"
        );

        // oracle は safe_used × seeds ぶん要る（強制した安全手が欠けても止める）
        let rows: Vec<serde_json::Value> = full_input("g1")
            .into_iter()
            .filter(|r| !(r["arm"] == "oracle" && r["seed"] == 1))
            .collect();
        let problems = check_inputs(&metas, &rows);
        assert!(
            problems.iter().any(|p| p.contains("oracle")),
            "{problems:?}"
        );
    }

    /// 記録集合の指紋はファイルの並び順に依らず、**中身**が違えば変わる
    #[test]
    fn 記録集合の指紋は名前と中身で決まる() {
        let dir = std::env::temp_dir().join(format!("mate-continue-fp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str, body: &str| {
            let p = dir.join(name);
            std::fs::write(&p, body).unwrap();
            p
        };
        let one = write("1.jsonl", "a");
        let two = write("2.jsonl", "b");

        let a = records_fingerprint(&read_records(&[one.clone(), two.clone()]));
        let b = records_fingerprint(&read_records(&[two.clone(), one.clone()]));
        assert_eq!(a, b, "並び順には依らない");
        assert_ne!(
            a,
            records_fingerprint(&read_records(&[one.clone()])),
            "集合が違えば変わる"
        );

        // **同名で中身が違う**記録は別集合として扱う（レビュー2巡目 [P2]）
        write("2.jsonl", "b2");
        assert_ne!(
            a,
            records_fingerprint(&read_records(&[one.clone(), two.clone()])),
            "中身が変われば指紋も変わる"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **指紋は解析に渡した bytes から作る**（レビュー3巡目 [P1]）。
    /// 開始時に読んだ集合を持ち回るので、継続対局の途中でパスの中身が
    /// 差し替わっても「行は A 由来・meta は B の指紋」にはならない。
    #[test]
    fn 指紋は開始時に読んだbytesから作る() {
        let dir = std::env::temp_dir().join(format!("mate-continue-toctou-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("g.jsonl");
        std::fs::write(&path, "A").unwrap();

        // 開始時に読む（この bytes だけを解析にも指紋にも使う）
        let records = read_records(&[path.clone()]);
        let fp = records_fingerprint(&records);

        // 継続対局の最中に同じパスが差し替わったとする
        std::fs::write(&path, "B").unwrap();

        assert_eq!(
            fp,
            records_fingerprint(&records),
            "終了時に読み直さないので指紋は動かない"
        );
        assert_eq!(records[0].1, b"A", "解析に渡る bytes も開始時のまま");
        assert_ne!(
            fp,
            records_fingerprint(&read_records(&[path.clone()])),
            "読み直せば別集合になる（＝差し替えを素通ししているわけではない）"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// ランキング段と継続段の実効並列度は別々に記録する（レビュー3巡目 [P2]）
    #[test]
    fn 実験キーはランキングと継続の並列度を分けて持つ() {
        let mut a = exp_of("estimator_v14");
        let b = a.clone();
        assert_eq!(a, b);
        // 疎なシャード（決定点が少なくランキングだけ 2 並列）は別実験
        a["policy_jobs"] = serde_json::json!(2);
        assert_ne!(a, b, "policy_jobs だけが違っても別実験として弾ける");
        let metas = vec![meta(b, 0), meta(a, 1)];
        let errs = check_inputs(&metas, &[]);
        assert!(
            errs.iter().any(|e| e.contains("policy_jobs")),
            "実験キーの食い違いとして報告される: {errs:?}"
        );
    }
}
