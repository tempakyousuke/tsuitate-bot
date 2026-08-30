//! **王手中の反則経済 P0-6: 継続の局所効果**（issue #31。**検証セットでだけ回す**）。
//!
//! 王手中の1手番を方策どおりに指させ（非合法なら実対局と同じく反則を積んで次候補へ）、
//! そのあとを通常方策で終局まで指し継いで 勝1 / 分0.5 / 負0 で数える。
//! P0-5 が出すのは「反則が減ったか・真実指標が良くなったか」で、**勝敗への変換**は
//! ここでしか測れない。
//!
//! **これは一手番の局所効果であって、全局反復適用の下界でも上界でもない**
//! （反復すれば節約が将来にも効いて大きくなることも、状態分布が変わって逆転する
//! こともある）。P1 の arena へ進むための局所的な有効性確認と位置づける。
//!
//! - **1元対局につき estimand ごとに最大1決定点**（同じ元対局の相互排他的な
//!   未来を足し合わせない）。候補が複数あるときは最初の対象手番
//! - estimand は2つ: `foul`（実戦で反則した王手手番＝改善側）と
//!   `nofoul`（反則0の王手手番＝**新しいプローブを足す害**の非劣性）
//! - 継続の乱数は **arm に依らず `(局, 決定点, seed)` だけ**から作る（共通乱数）
//! - Δ の分母は**全対局数**（対象の手番が無かった局は 0）。CI は元対局単位の
//!   cluster bootstrap。**シャードが欠けたら判定を出さない**（#28 の契約）
//!
//! 門（issue #31 で事前登録）:
//!
//! - 反則あり: 主 arm の **`Δpolicy − Δcurrent ≥ +0.04`（CI 下限 > 0）**
//!   かつ foul_limit・破滅率が悪化しない
//! - 反則0: **非劣性**（`≥ −0.01` かつ CI 下限 > −0.02）かつ即時反則・破滅率・
//!   foul_limit が悪化しない。ここで改善は要求しない
//! - **β-order は反則経済施策なので即時反則の非増加を必須**にする。
//!   **full β は反則増を許し**、勝率改善と foul_limit 非悪化で判定する
//!
//! **arm は `--policy` で外から固定する**（P0-4 / P0-5 を見てから主 arm を1本
//! 決める設計なので、水準をコードに埋めない）。発見セットを見てから検証セットで
//! 水準を変えてはいけない。
//!
//! usage:
//!   TSUITATE_THINK_BUDGET_MS=700 cargo run --release --bin check_continue -- \
//!     [--seeds 4] [--opponent estimator_v14] [--policy alpha@k2] \
//!     [--jobs N] [--shard i/n] [--out out.jsonl] <records/*.jsonl...>
//!   cargo run --release --bin check_continue -- report [--allow-incomplete] <out-*.jsonl...>
//!
//! **思考予算はプロセス env で両側・両段に同じ値が効く**（`bin/mate_continue` と
//! 同じ規約）。つまりここで測るのは「その予算での方策」の局所効果で、
//! ランキングも継続も同じ動作点に揃う。

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tsuitate_bot::check_belief::{self, ArmSpec, Attrition, decision_points};
use tsuitate_bot::check_policy::entry_setup;
use tsuitate_bot::checkpoint::stable_hash;
use tsuitate_bot::mate_economy::force_move;
use tsuitate_bot::protocol::Color;
use tsuitate_bot::scenario_core::{Replayed, clone_log, make_view, side_idx};
use tsuitate_bot::selfplay::{GameResult, StartState, mix, play_continuation};
use tsuitate_bot::shogi::Position;
use tsuitate_bot::strategy::{self, EvalParams};
use tsuitate_bot::truth_replay::parse_bot_and_end;

/// 行の契約の版。**schema 1 は集計から弾く**（PR #33 レビュー2巡目 [P1]）:
/// 1 には安全性の列（`foul_limit_loss` / `foul_limit_win` / `immediate_catastrophe`）と
/// `arm_order` が無く、しかも arm を別 unit として並走させていた。欠けた列を
/// `unwrap_or(false)` で 0 と読むと**反則負けも破滅も常に非悪化に見えて門を通せる**ので、
/// 版の拒否と**必須列の存在検査**（`REQUIRED_ROW_KEYS`）の両方で止める。
///
/// **3 …** PR #37 レビュー4〜5巡目で**母集団が変わった**: `check_belief::decision_points`
/// へ寄せて、反則だけ積んで受理手なしで終局した**終端手番**も含めるようになった
/// （それまでの `for_each_decision_full` は受理手を単位に回すので終端を返さない）。
/// 終端手番は改善対象の最悪ケースであり即時反則負けの分子でもあるので、
/// schema 2 の記録（終端が系統的に欠けている）を新しい gate へ混ぜると
/// オラクル効果も safety 指標も楽観側へ偏る。meta の `points_detail` に
/// `terminal` があることも要求する
///
/// **4 …** PR #37 レビュー8巡目で**実行順の生成規則が変わった**: schema 3 までは
/// arm 群を `(決定点番号 + seed) % グループ数` で **cyclic rotate** していたが、
/// 巡回は AB/BA にならない（固定の2 arm の前後は「切れ目がその間に入るか」だけで
/// 決まるので、g グループ・距離 d なら A が先になるのは g 回中 (g − d) 回）。
/// `schedule_groups` の**反転**へ変え、meta に `schedule_control` を残して
/// 集計側でも釣り合いを検査する。schema 3 の記録は主差に実行順効果を
/// 残したまま緑になるので弾く。
///
/// **5 …** レビュー9巡目 [P1] で、① 分離した unit の前後差を 1 まで許すと
/// 加法的な実行順効果が丸ごと主差へ残る（対の片方だけが分離した決定点）
/// ② 「畳まれた unit」を `arm_order` の一致で判定していたので、order を
/// 書き換えるだけで均衡検査を空集合にできた、の2点を直した。
/// 行に `continuation_group`（強制列の指紋）を足した。
///
/// **6 …** レビュー10巡目 [P1] で、① `continuation_group` の型を数値へ変えるだけで
/// `unit_index` から全行が消え、均衡検査も主判定も空集合のまま exit 0 になった
/// ② 5 で入れた「釣り合わない seed 対を事後に除外」は arm の選択結果で
/// 事前登録した母集団を条件づける post-treatment な操作で、全部落ちれば門を
/// 一度も評価せずに成功しうる、の2点を直した。**対照と treatment は強制列が
/// 同じでも畳まない**ようにして、全 unit を保持したまま順序が閉じるようにし、
/// `continuation_group` は arm も混ぜた指紋になったので schema 5 とは値が違う
const ROW_SCHEMA: u32 = 6;

/// estimand の全量。**集計が走査するのはこの2つだけ**なので、meta がこれ以外を
/// 宣言したら期待キーを作る前に拒否する（PR #33 レビュー7巡目 [P1]）。
/// meta と行が**同じ未知の値**で揃っていると、キーの厳密一致は通るのに集計の
/// ループから外れて、その決定点が層から無言で消える（= 任意の決定点を除外して
/// 点推定と安全性判定を動かせる）
const ESTIMANDS: [&str; 2] = ["foul", "nofoul"];

/// 継続1本の行に必ず入っていなければならない列。1つでも欠けたら集計しない
/// （欠測を「悪化なし」と読むのが一番危ない失敗の仕方なので、既定では降格させない）
const REQUIRED_ROW_KEYS: [&str; 11] = [
    "arm",
    "estimand",
    "score",
    "immediate_fouls",
    "foul_limit_loss",
    "foul_limit_win",
    "immediate_catastrophe",
    "seed",
    // 実行順の監査（arm 群の回転）ができなくなるので、これも欠測を許さない
    "arm_order",
    // 欠測を 0 と読むと**別々の実行が同じ replicate に潰れて**重複検査を素通りする
    "replicate",
    // 「畳まれた unit」の判定を `arm_order` の一致で代用させないための指紋
    "continuation_group",
];

fn die(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(2);
}

struct Point {
    game: String,
    move_number: u32,
    estimand: &'static str,
    /// 手番開始時（反則0）の状態
    entry: Replayed,
    /// 決定点の真実局面（裁定用。方策には渡さない）
    truth: Position,
    bot: Color,
    /// 実戦の反則列＋受理手（`baseline` arm が強制する列）。
    /// **終端手番では受理手が無い**ので反則列だけ（`baseline` は走らせない）
    record_order: Vec<String>,
    /// 終端手番（反則だけ積んで受理手なしで終局した手番）
    terminal: bool,
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

/// 継続の乱数（bot 側・相手側）。**arm に依らず (局, 決定点, seed) だけ**から作る。
///
/// 強制した手を hash に混ぜると arm ごとに別の乱数列で継続することになり、
/// `Δpolicy − Δcurrent` が共通乱数のペア差にならない（issue #28 の教訓④）。
fn continuation_seeds(game: &str, ply: u32, seed: u64) -> (u64, u64) {
    let base = stable_hash(seed, &format!("{game}#{ply}"));
    (mix(base ^ 0x31C0_FFEE), mix(base ^ 0x31BE_EF00))
}

/// 手番を `order` の順に強制して指させ、そのあとを終局まで指し継ぐ
/// **unit（決定点 × seed）の中での実行順**（PR #37 レビュー8巡目 [P1]）。
///
/// 以前は `(決定点番号 + seed) % グループ数` の **cyclic rotate** だったが、これは
/// AB/BA にならない: 巡回では固定の2 arm `A, B` の前後は「切れ目が A と B の間に
/// 入るか」だけで決まるので、グループ数 g・距離 d のとき A が先になるのは g 回中
/// (g − d) 回。3グループ・4 seed なら shift が `0,1,2,0` で、しかもグループ数と
/// 並びは seed ごとの強制列で変わり、`replicate` は shift に入らない。
///
/// **反転なら全ペアの前後が同時に入れ替わる**（`bin/check_policy` の実再決定
/// ブロックと同じ形）ので、`(決定点番号 + seed) % 2` で反転する。前向きの並びは
/// **arm の優先順位**（`baseline` → `policies` のタグ順）で決める: 強制列の
/// 辞書順で並べると seed ごとに前向きの並びそのものが変わり、反転しても
/// 決定点の中で釣り合わない。
///
/// 同じ強制列に畳まれた arm どうしは**同じ継続1本**を共有する（`arm_order` も
/// 同じ）ので、その組は実行順効果を持たない。したがって釣り合いを検査するのは
/// 「別グループに分かれた unit」だけでよく、それは `report` 側が数える。
fn schedule_groups(
    mut group: Vec<(GroupKey, Vec<String>)>,
    arm_rank: &BTreeMap<String, usize>,
    point_index: usize,
    seed: u64,
) -> Vec<(GroupKey, Vec<String>)> {
    group.sort_by_key(|(key, arms)| {
        (
            arms.iter().map(|a| arm_rank.get(a).copied().unwrap_or(usize::MAX)).min(),
            key.clone(),
        )
    });
    if (point_index + seed as usize) % 2 == 1 {
        group.reverse();
    }
    group
}

/// 継続1本を決めるキー: 強制列と、**主比較に関わる arm なら** その arm 名。
/// 対照と treatment は強制列が同じでも別々に走らせる（レビュー10巡目 [P1]）
type GroupKey = (Vec<String>, Option<String>);

/// `continuation_group`（16桁の小文字 hex）。同じ値 = 同じ継続1本を共有した組。
fn group_fingerprint(key: &GroupKey) -> String {
    use sha2::Digest as _;
    let mut h = sha2::Sha256::new();
    for m in &key.0 {
        sha2::Digest::update(&mut h, m.as_bytes());
        sha2::Digest::update(&mut h, b"\n");
    }
    sha2::Digest::update(&mut h, b"\0");
    sha2::Digest::update(&mut h, key.1.as_deref().unwrap_or("").as_bytes());
    h.finalize().iter().take(8).map(|b| format!("{b:02x}")).collect()
}

fn run_arm(p: &Point, order: &[String], seed: u64, opponent: &str) -> serde_json::Value {
    let me_i = side_idx(p.bot);
    let logs = [clone_log(&p.entry.logs[0]), clone_log(&p.entry.logs[1])];
    // 適用は共有規約（審判と同じ観測の記録・反則の積み方）
    let forced = force_move(&p.entry.pos, &logs, p.entry.fouls, p.bot, order);
    let forced_fouls = forced.forced_fouls;
    let base = |score: f64, reason: &str, plies: u32, added: u32, think: f64, extra: u32| {
        // **`reason == "foul_limit"` は「誰かが反則負けした」でしかない**
        // （PR #33 レビュー [P1]）。相手が反則負けして bot が勝った終局まで同じ 1 に
        // 数えると、相手の自滅を増やした方策を「反則負け悪化」で落としてしまうし、
        // 自分負けと相手負けの入れ替わりも見えない。**bot 側の負けだけ**を数える
        let limit = reason == "foul_limit";
        serde_json::json!({
            "schema": ROW_SCHEMA,
            "game": p.game, "move_number": p.move_number, "estimand": p.estimand,
            "seed": seed, "opponent": opponent,
            "score": score, "reason": reason, "plies": plies, "added_plies": added,
            // **即時反則** = その手番で積んだ反則（β-order の門はここを見る）
            "immediate_fouls": forced_fouls,
            // **即時破滅** = その手番だけで8反則以上（事前登録の破滅率）
            "immediate_catastrophe": forced_fouls >= 8,
            "added_fouls_me": forced_fouls + extra,
            "think_mean_ms": think,
            "foul_limit": limit,
            // bot が反則負けした / 相手が反則負けして bot が勝った
            "foul_limit_loss": limit && score < 0.5,
            "foul_limit_win": limit && score > 0.5,
        })
    };
    if forced.foul_limit || forced.played.is_none() {
        // その手番で反則上限に達した（＝その場で反則負け）／候補を出せなかった
        return base(0.0, "foul_limit", p.entry.plies, 0, 0.0, 0);
    }
    let pos = forced.pos;
    let logs = forced.logs;
    let fouls = forced.fouls;
    let (seed_me, seed_opp) = continuation_seeds(&p.game, p.move_number, seed);
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
    let start = StartState { pos, logs, fouls, plies: p.entry.plies + 1 };
    let out = play_continuation(
        [strats[0].take().unwrap(), strats[1].take().unwrap()],
        start,
        0,
    );
    let score = match out.result {
        GameResult::Win(w) if w == p.bot => 1.0,
        GameResult::Win(_) => 0.0,
        GameResult::Draw => 0.5,
    };
    let think = &out.think_us[me_i];
    let mean_ms = if think.is_empty() {
        0.0
    } else {
        think.iter().sum::<u64>() as f64 / think.len() as f64 / 1000.0
    };
    let mut row = base(
        score,
        out.reason,
        out.plies,
        out.added_plies,
        mean_ms,
        out.added_fouls[me_i],
    );
    row["played"] = serde_json::json!(forced.played);
    row
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "report") {
        run_report(&args[1..]);
        return;
    }
    if args.first().is_some_and(|a| a == "combined") {
        run_combined(&args[1..]);
        return;
    }
    let mut seeds: u64 = 4;
    let mut opponent = "estimator_v14".to_string();
    // **主 arm は外から固定する**（P0-4 / P0-5 を見てから1本決める）
    let mut policy_tags: Vec<String> = vec!["current".into()];
    let mut jobs: usize =
        std::thread::available_parallelism().map_or(1, |n| n.get().saturating_sub(2).max(1));
    let mut shard = (0usize, 1usize);
    // **2回実行して平均する**ときの実行番号（issue #31 の再実行規則）。
    // report は replicate ごとにシャードの完全性を検査し、全 replicate で
    // 実験キー（= 同じ build の `source_fingerprint`）が一致することを要求する
    let mut replicate: u64 = 0;
    let mut out_path: Option<String> = None;
    let mut allow_opponent_mismatch = false;
    let mut allow_incomplete = false;
    let mut specs: Vec<String> = vec![];
    let mut i = 0;
    let need = |v: Option<&String>, what: &str| -> String {
        v.cloned()
            .unwrap_or_else(|| die(&format!("{what} には値が必要です")))
    };
    while i < args.len() {
        match args[i].as_str() {
            "--seeds" => {
                seeds = need(args.get(i + 1), "--seeds")
                    .parse()
                    .unwrap_or_else(|_| die("--seeds は整数"));
                i += 2;
            }
            "--opponent" => {
                opponent = need(args.get(i + 1), "--opponent");
                i += 2;
            }
            "--policy" => {
                policy_tags = need(args.get(i + 1), "--policy")
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                i += 2;
            }
            "--jobs" => {
                jobs = need(args.get(i + 1), "--jobs")
                    .parse::<usize>()
                    .unwrap_or_else(|_| die("--jobs は整数"))
                    .max(1);
                i += 2;
            }
            "--replicate" => {
                replicate = need(args.get(i + 1), "--replicate")
                    .parse::<u64>()
                    .unwrap_or_else(|_| die("--replicate は整数"));
                i += 2;
            }
            "--shard" => {
                let v = need(args.get(i + 1), "--shard");
                let (a, b) = v.split_once('/').unwrap_or_else(|| die("--shard は i/n"));
                shard = (
                    a.parse().unwrap_or_else(|_| die("--shard は i/n")),
                    b.parse().unwrap_or_else(|_| die("--shard は i/n")),
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
    // **`--seeds 0` で空の集計を作れてはいけない**（issue #28 が塞いだ穴と同じ）
    if seeds == 0 {
        die("--seeds は 1 以上にしてください（0 だと継続を1局も走らせずに Δ が全部 0 になる）");
    }
    // `current` は必ず対照として要る（Δpolicy − Δcurrent が門なので）
    if !policy_tags.iter().any(|t| t == "current") {
        policy_tags.insert(0, "current".into());
    }
    // arm 名は `bin/check_policy` と共通の規約（`[belief|policy][@shadow|@real]`）。
    // issue #36 P0-2b の主 arm は `oracle@kinf@real`（= `oracle_full_score@real`）
    let mut policies: Vec<ArmSpec> = policy_tags
        .iter()
        .map(|t| ArmSpec::parse(t).unwrap_or_else(|| die(&format!("未知の arm: {t}"))))
        .collect();
    // **実再決定 arm を混ぜたら `current@real` も必ず取る**: shadow の `current` と
    // 比べると「オラクル」と「指し直したこと」の効果が混ざる（#31 P0-6 の教訓と同型）
    if policies.iter().any(|a| a.real) && !policies.iter().any(|a| a.tag == "current@real") {
        policies.push(ArmSpec::parse("current@real").expect("既定 arm"));
    }
    policies.sort_by(|a, b| a.tag.cmp(&b.tag));
    policies.dedup_by(|a, b| a.tag == b.tag);
    // **実再決定 arm を回すなら seed は偶数**（PR #37 レビュー8巡目 [P1]）。
    // 実行順は `schedule_groups` が seed の偶奇で**反転**させるので、奇数だと
    // 決定点ごとに AB/BA が閉じない。`check-belief.yml` の plan はこの契約を
    // 検査していたが、生成側にその規則が無ければ検査は空手形だった
    if policies.iter().any(|a| a.real) && seeds % 2 != 0 {
        die("--seeds は 2 以上の偶数にしてください（実再決定 arm の AB/BA は seed の偶奇で閉じるので、奇数だと実行順効果がペア差に残る）");
    }
    // **主比較の対照**（`report --baseline` と同じ規約で解決する）。スケジュールは
    // この arm と各 treatment の前後を決定点の中で反転させる
    let control_tag = if policies.iter().any(|a| a.real) { "current@real" } else { "current" };
    let files = collect_records(&specs);
    if files.is_empty() {
        die("記録ファイルが見つかりません");
    }
    let cfg = tsuitate_bot::config::ambient();
    let params = EvalParams::default();
    let eval_particles = strategy::eval_particles_for_budget(cfg.think_budget_ms);

    // ---- 決定点（1元対局につき estimand ごとに最大1つ）----------------------
    use sha2::Digest as _;
    let mut digest = sha2::Sha256::new();
    let mut points: Vec<Point> = vec![];
    let mut games = 0u32;
    let mut broken = 0u32;
    let mut attrition = Attrition::default();
    let mut record_opponents: BTreeMap<String, u32> = BTreeMap::new();
    let mut mismatched = 0u32;
    for path in &files {
        let name = path.to_string_lossy().to_string();
        let Ok(content) = std::fs::read_to_string(path) else {
            broken += 1;
            continue;
        };
        sha2::Digest::update(&mut digest, name.as_bytes());
        sha2::Digest::update(&mut digest, content.as_bytes());
        let Some((bot, end)) = parse_bot_and_end(&content) else {
            broken += 1;
            continue;
        };
        *record_opponents
            .entry(end.opponent.username.clone())
            .or_insert(0) += 1;
        if end.opponent.username != opponent {
            mismatched += 1;
            if !allow_opponent_mismatch {
                continue;
            }
        }
        let short = Path::new(&name)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or(name.clone());
        let mut found: Vec<Point> = vec![];
        // **母集団は共通の `check_belief::decision_points`**（PR #37 レビュー4巡目 [P1]）。
        // 終端手番（反則だけ積んで受理手なしで終局）は改善対象の最悪ケースなので
        // 落とさない。ただし `baseline`（実戦の反則列＋受理手を強制）は受理手が
        // 無いと組めないので、その arm だけ終端手番では走らせない
        let Some(raw) = decision_points(&end, bot, &mut attrition) else {
            broken += 1;
            continue;
        };
        for cp in raw {
            let estimand = cp.estimand();
            let mut record_order = cp.record_fouls.clone();
            let terminal = cp.record_accepted.is_none();
            if let Some(a) = &cp.record_accepted {
                record_order.push(a.clone());
            }
            found.push(Point {
                game: short.clone(),
                move_number: cp.move_number,
                estimand,
                truth: cp.truth,
                entry: cp.entry,
                bot,
                record_order,
                terminal,
            });
        }
        games += 1;
        // **estimand ごとに最初の1つだけ**（同じ元対局の相互排他的な未来を
        // 足し合わせない。issue #28 P0-6 の契約と同じ）
        for estimand in ESTIMANDS {
            if let Some(p) = found.iter().position(|p| p.estimand == estimand) {
                points.push(found.remove(p));
            }
        }
    }
    if mismatched > 0 && !allow_opponent_mismatch {
        eprintln!(
            "元対局の相手が --opponent {opponent} と一致しない記録が {mismatched} 件あります: {}",
            record_opponents
                .iter()
                .map(|(k, n)| format!("{k} {n}局"))
                .collect::<Vec<_>>()
                .join(" / ")
        );
        die("Δ が「受けの効果」と「相手が変わった効果」の混合になるので中止しました（--allow-opponent-mismatch で強行）");
    }
    points.sort_by(|a, b| (&a.game, a.move_number).cmp(&(&b.game, b.move_number)));
    // シャードは**局単位**（同じ局の arm は必ず同じシャードで走る）
    if shard.1 > 1 {
        let mut names: Vec<String> = points.iter().map(|p| p.game.clone()).collect();
        names.sort();
        names.dedup();
        let keep: HashSet<String> = names
            .into_iter()
            .enumerate()
            .filter(|(i, _)| i % shard.1 == shard.0)
            .map(|(_, n)| n)
            .collect();
        points.retain(|p| keep.contains(&p.game));
    }
    if points.is_empty() {
        die("対象の決定点がありません");
    }
    println!(
        "記録 {} 件（壊れ {broken}）/ 局 {games} / 決定点 {}（反則あり {} / 反則0 {}）",
        files.len(),
        points.len(),
        points.iter().filter(|p| p.estimand == "foul").count(),
        points.iter().filter(|p| p.estimand == "nofoul").count(),
    );
    println!(
        "相手 {opponent} / 方策 {} / 思考予算 {}ms / seeds {seeds} / jobs {jobs} / shard {}/{}",
        policies.iter().map(|a| a.tag.as_str()).collect::<Vec<_>>().join(","),
        cfg.think_budget_ms,
        shard.0,
        shard.1,
    );
    println!("source_fingerprint {}", env!("TSUITATE_SOURCE_FINGERPRINT"));

    // ---- 方策ごとの強制列（決定点 × seed）--------------------------------
    // P0-5 と**同じ配管**（`entry_setup` → `simulate`）で作る。simulate の
    // `sequence` は「反則した手…受理された手」なので、そのまま `force_move` へ
    // 渡せば実対局と同じ裁定を再現する
    let points = Arc::new(points);
    let orders: Arc<Mutex<BTreeMap<(usize, u64), BTreeMap<String, Vec<String>>>>> =
        Arc::new(Mutex::new(BTreeMap::new()));
    let policy_units: Vec<(usize, u64)> = (0..points.len())
        .flat_map(|p| (0..seeds).map(move |s| (p, s)))
        .collect();
    let policy_jobs = jobs.min(policy_units.len().max(1)).max(1);
    run_parallel(jobs, policy_units.len(), |ui| {
        let (pi, seed) = policy_units[ui];
        let p = &points[pi];
        let Some(setup) = entry_setup(&p.entry, &p.truth, seed, &params, eval_particles) else {
            return;
        };
        let mut by_arm: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for spec in &policies {
            // 配管は `bin/check_policy`（P0-2）と共有する
            let Some(run) = check_belief::run_arm(
                &setup,
                &p.entry,
                &p.truth,
                p.bot,
                spec,
                &params,
            ) else {
                continue;
            };
            if !run.out.sequence.is_empty() {
                by_arm.insert(spec.tag.clone(), run.out.sequence);
            }
        }
        orders.lock().unwrap().insert((pi, seed), by_arm);
    });
    let orders = Arc::try_unwrap(orders).ok().unwrap().into_inner().unwrap();

    // ---- 継続（同じ強制列は1回だけ走らせて該当 arm へ配る）-----------------
    // **1 unit = 1つの `(決定点, seed)` の全 arm**。同じ worker が背中合わせに
    // 走らせるので、壁時計予算のもとでの CPU 競合と開始順の差が arm 間の Δ へ
    // 混ざらない。並びは `schedule_groups` が決める（seed の偶奇で**反転**）。
    // arm の優先順位は「`baseline` → `policies` のタグ順」で固定する
    let arm_rank: BTreeMap<String, usize> = std::iter::once("baseline".to_string())
        .chain(policies.iter().map(|a| a.tag.clone()))
        .enumerate()
        .map(|(i, t)| (t, i))
        .collect();
    // **主比較に関わる arm どうしは畳まない**（PR #37 レビュー10巡目 [P1]）。
    // 強制列が一致したときに対照と treatment を1本の継続へ畳むと、その unit は
    // 「実行順を持たない」side へ落ちる。畳まれるかは seed ごとの選択手で変わるので、
    // 対の片方だけが畳まれた seed 対が生まれ、AB/BA が閉じない。8〜9巡目は
    // それを**事後に除外**して閉じたが、除外は arm の選択結果で母集団を条件づける
    // post-treatment な操作で、全部落ちれば門を一度も評価せずに成功しうる。
    // 対照と treatment は**強制列が同じでも別々に走らせる**（畳むのは
    // `baseline` と shadow arm どうしだけ）。こうすれば全 unit が常に分離するので、
    // 偶数 seed の反転だけで**除外なしに**順序が閉じる
    let compared: BTreeSet<&str> = std::iter::once(control_tag)
        .chain(policies.iter().map(|a| a.tag.as_str()).filter(|t| *t != control_tag))
        .collect();
    type Unit = (usize, u64, Vec<(GroupKey, Vec<String>)>);
    let mut units: Vec<Unit> = vec![];
    for (pi, p) in points.iter().enumerate() {
        for seed in 0..seeds {
            let mut by_order: BTreeMap<GroupKey, Vec<String>> = BTreeMap::new();
            // 終端手番は受理手が無いので `baseline` を組めない（反則列だけを
            // 強制すると、別の理由で終わった局を反則負け 0 点として数えてしまう）。
            // 方策 arm は自分で選ぶので終端手番でも走る
            if !p.terminal {
                by_order
                    .entry((p.record_order.clone(), None))
                    .or_default()
                    .push("baseline".to_string());
            }
            if let Some(by_arm) = orders.get(&(pi, seed)) {
                for (arm, order) in by_arm {
                    let key = if compared.contains(arm.as_str()) {
                        (order.clone(), Some(arm.clone()))
                    } else {
                        (order.clone(), None)
                    };
                    by_order.entry(key).or_default().push(arm.clone());
                }
            }
            let group = schedule_groups(by_order.into_iter().collect(), &arm_rank, pi, seed);
            units.push((pi, seed, group));
        }
    }
    let lines: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(vec![]));
    let continuation_jobs = jobs.min(units.len().max(1)).max(1);
    let started = std::time::Instant::now();
    {
        let lines = Arc::clone(&lines);
        let points = Arc::clone(&points);
        let units = &units;
        let opponent = opponent.as_str();
        run_parallel(jobs, units.len(), move |ui| {
            let (pi, seed, group) = &units[ui];
            let mut rows = vec![];
            for (k, (key, arms)) in group.iter().enumerate() {
                let row = run_arm(&points[*pi], &key.0, *seed, opponent);
                for arm in arms {
                    // 同じ強制列 = 同じ継続なので、arm ごとに同じ結果を配る
                    let mut r = row.clone();
                    r["arm"] = serde_json::json!(arm);
                    // 何番目に走ったか（実行順効果の監査用）
                    r["arm_order"] = serde_json::json!(k);
                    // **どの継続1本を共有したか**（PR #37 レビュー9巡目 [P1]）。
                    // 「畳まれた（＝実行順効果を持たない）」の判定を `arm_order` の
                    // 一致で代用すると、order を書き換えるだけで均衡検査を
                    // 空集合にできる。強制列そのものの指紋を行に残して照合する
                    r["continuation_group"] = serde_json::json!(group_fingerprint(key));
                    r["replicate"] = serde_json::json!(replicate);
                    rows.push(r);
                }
            }
            lines.lock().unwrap().extend(rows);
        });
    }
    let mut lines = Arc::try_unwrap(lines).ok().unwrap().into_inner().unwrap();
    lines.sort_by_key(|v| {
        (
            v["game"].as_str().unwrap_or("").to_string(),
            v["move_number"].as_u64().unwrap_or(0),
            v["arm"].as_str().unwrap_or("").to_string(),
            v["seed"].as_u64().unwrap_or(0),
        )
    });
    eprintln!(
        "継続 {} 局 / {:.1}分",
        lines.len(),
        started.elapsed().as_secs_f64() / 60.0
    );

    let records_fingerprint: String =
        digest.finalize().iter().map(|b| format!("{b:02x}")).collect();
    let experiment = serde_json::json!({
        "opponent": opponent,
        "budget_ms": cfg.think_budget_ms,
        "seeds": seeds,
        // **解決後の arm 名**を残す（`current@real` の自動追加を含む。
        // 宣言と実際がずれると完全性検査が空振りする）
        "policies": policies.iter().map(|a| a.tag.clone()).collect::<Vec<_>>(),
        // **スケジュールが前後を反転させた対照**（PR #37 レビュー8巡目 [P1]）。
        // `report --baseline` がこれと違う arm を指したら、AB/BA が閉じているのは
        // 別のペアなので集計側で気づけるようにしておく
        "schedule_control": control_tag,
        "policy_jobs": policy_jobs,
        "continuation_jobs": continuation_jobs,
        "games": games,
        // **読めなかった元対局は採否経路では致命**（PR #37 レビュー13巡目 [P1]）。
        // 黙って落とすと `games`（＝ Δ の分母）が縮んだ選択標本になり、
        // v13 / v14 で同数だけ失敗すれば相手間の `games` 一致検査も通ってしまう
        "broken": broken,
        "shard_total": shard.1,
        "config": cfg.fingerprint(),
        "source_fingerprint": env!("TSUITATE_SOURCE_FINGERPRINT"),
        "records": records_fingerprint,
    });
    let meta = serde_json::json!({
        "schema": ROW_SCHEMA,
        "type": "meta",
        "experiment": experiment,
        "shard": shard.0,
        "replicate": replicate,
        "games": games,
        "points": points.len(),
        // 母集団の attrition（終端手番が系統的に欠測すると門が甘くなる）
        "attrition": {
            "check_turns": attrition.turns,
            "terminal": attrition.terminal,
            "unreplayable": attrition.unreplayable,
        },
        "points_detail": points
            .iter()
            .map(|p| serde_json::json!({
                "game": p.game,
                "move_number": p.move_number,
                "estimand": p.estimand,
                // 終端手番は `baseline` を組めない（期待キーから外す）
                "terminal": p.terminal,
            }))
            .collect::<Vec<_>>(),
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

// ---- 集計 ------------------------------------------------------------------

/// 局ごとの寄与（arm 別の平均値）。cluster bootstrap の統計単位は元対局
fn contributions(
    rows: &[&serde_json::Value],
    key: &dyn Fn(&serde_json::Value) -> f64,
) -> BTreeMap<String, BTreeMap<String, f64>> {
    // game → arm → 値の平均
    let mut sums: BTreeMap<String, BTreeMap<String, (f64, f64)>> = BTreeMap::new();
    for r in rows {
        let e = sums
            .entry(r["game"].as_str().unwrap_or("?").to_string())
            .or_default()
            .entry(r["arm"].as_str().unwrap_or("?").to_string())
            .or_default();
        e.0 += key(r);
        e.1 += 1.0;
    }
    sums.into_iter()
        .map(|(g, m)| {
            (
                g,
                m.into_iter()
                    .map(|(a, (s, n))| (a, if n > 0.0 { s / n } else { 0.0 }))
                    .collect(),
            )
        })
        .collect()
}

/// Δ（= Σ_局 寄与 / 全対局数）と、元対局単位の cluster bootstrap CI
fn delta_ci(
    contrib: &BTreeMap<String, BTreeMap<String, f64>>,
    arm: &str,
    base: Option<&str>,
    games_total: usize,
) -> (f64, f64, f64) {
    if games_total == 0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let mut vals: Vec<f64> = contrib
        .values()
        .map(|m| {
            let a = m.get(arm).copied().unwrap_or(0.0);
            match base {
                Some(b) => a - m.get(b).copied().unwrap_or(0.0),
                None => a,
            }
        })
        .collect();
    // **対象の手番が無かった局も寄与 0 の cluster として標本に入れる**
    // （再標本化を「対象があった局」だけに限ると分散が過小になる）
    vals.resize(vals.len().max(games_total), 0.0);
    let point = vals.iter().sum::<f64>() / vals.len() as f64;
    let mut state: u64 = 0x3120_2606;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let reps = 2000;
    let mut draws: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let mut s = 0.0;
        for _ in 0..vals.len() {
            s += vals[next() % vals.len()];
        }
        draws.push(s / vals.len() as f64);
    }
    draws.sort_by(f64::total_cmp);
    (
        point,
        draws[(reps as f64 * 0.025) as usize],
        draws[(reps as f64 * 0.975) as usize],
    )
}

fn report(metas: &[serde_json::Value], rows: &[serde_json::Value], allow_incomplete: bool) {
    report_vs(metas, rows, allow_incomplete, "current", None)
}

/// 対照 arm を指定できる版（issue #36 P0-2b は `current@real` と比べる:
/// shadow の `current` と比べると「オラクル」と「指し直したこと」が混ざる）
fn report_vs(
    metas: &[serde_json::Value],
    rows: &[serde_json::Value],
    allow_incomplete: bool,
    baseline: &str,
    main_arm: Option<&str>,
) {
    for msg in check_inputs(metas, rows, baseline) {
        if allow_incomplete {
            eprintln!("警告: {msg}");
        } else {
            die(&msg);
        }
    }
    // **事後の除外はしない**（PR #37 レビュー10巡目 [P1]）。9巡目は釣り合わない
    // seed 対を主推定から落として順序を閉じたが、それは arm の選択結果で
    // 事前登録した母集団を条件づける post-treatment な操作で、全部落ちれば
    // 門を一度も評価せずに成功しうる。生成側で対照と treatment を畳まなく
    // したので、**全 unit を保持したまま**偶数 seed の反転だけで順序が閉じる。
    // 閉じていない入力は `check_inputs` が落とす（fail-closed）
    let exp = metas
        .first()
        .map(|m| m["experiment"].clone())
        .unwrap_or_default();
    let games_total = exp["games"].as_u64().unwrap_or(0) as usize;
    println!("\n=== P0-6: 継続の局所効果（issue #31。検証セット）===");
    println!(
        "  実験: 相手 {} / 予算 {}ms / seeds {} / 方策 {} / 全 {games_total}局 / 記録 {} / code {}",
        exp["opponent"], exp["budget_ms"], exp["seeds"], exp["policies"], exp["records"],
        exp["source_fingerprint"],
    );
    // シャードが揃っているか（Δ の分母は全対局数なので、欠けると分子だけが落ちる）。
    // **replicate ごとに**数える（2回実行して平均する経路。`check_inputs` が本判定）
    let shard_total = exp["shard_total"].as_u64().unwrap_or(1) as usize;
    let mut reps: Vec<u64> = metas
        .iter()
        .map(|m| m["replicate"].as_u64().unwrap_or(0))
        .collect();
    reps.sort_unstable();
    reps.dedup();
    let mut per_rep_ok = true;
    for rep in &reps {
        let mut seen: Vec<u64> = metas
            .iter()
            .filter(|m| m["replicate"].as_u64().unwrap_or(0) == *rep)
            .filter_map(|m| m["shard"].as_u64())
            .collect();
        seen.sort_unstable();
        seen.dedup();
        per_rep_ok &= seen.len() == shard_total;
    }
    let complete = per_rep_ok;
    if reps.len() > 1 {
        // **平均は集計器が取る**（表示値を手で平均しない。PR #33 レビュー3巡目 [P1]）。
        // 局ごとの寄与を replicate 間で平均してから cluster bootstrap するので、
        // 点推定は「各 replicate の Δ の平均」に厳密に一致し、CI も合成される
        println!(
            "  **{} replicate の平均**（replicate {:?}。実験キーが全 replicate で一致 = 同じ build・同じ設定）",
            reps.len(),
            reps
        );
    }
    if games_total == 0 {
        println!("  meta の games が 0（Δ の分母が取れない）");
        return;
    }

    // **meta が宣言した estimand は必ず判定する**（PR #37 レビュー10巡目 [P1]）。
    // 「行が無ければ黙って `continue`」だと、母集団が空になったときに +0.04 門も
    // 安全性 veto も一度も評価せずに exit 0 で終われる
    let declared: BTreeSet<String> = metas
        .iter()
        .flat_map(|m| m["points_detail"].as_array().cloned().unwrap_or_default())
        .filter_map(|d| d["estimand"].as_str().map(str::to_string))
        .collect();
    let mut failures: Vec<String> = vec![];
    let mut gated: BTreeSet<String> = BTreeSet::new();
    for estimand in ESTIMANDS {
        let sel: Vec<&serde_json::Value> =
            rows.iter().filter(|r| r["estimand"] == estimand).collect();
        if sel.is_empty() {
            if declared.contains(estimand) {
                die(&format!(
                    "estimand {estimand} は meta が宣言しているのに行が1つもありません（門を評価せずに成功できてしまう）"
                ));
            }
            continue;
        }
        let arms: BTreeSet<String> = sel
            .iter()
            .map(|r| r["arm"].as_str().unwrap_or("?").to_string())
            .collect();
        // 対照 arm が無いと `Δ − Δ対照` が黙って「対照 0」になる
        if !arms.contains(baseline) {
            die(&format!(
                "対照 arm {baseline} の行がありません（estimand {estimand}）: {arms:?}"
            ));
        }
        let score = contributions(&sel, &|r| r["score"].as_f64().unwrap_or(0.0));
        let fouls = contributions(&sel, &|r| r["immediate_fouls"].as_f64().unwrap_or(0.0));
        // **bot 側の反則負けだけ**を数える（相手の反則負け＝bot の勝ちを混ぜない）
        let loss = contributions(&sel, &|r| {
            f64::from(u8::from(r["foul_limit_loss"].as_bool().unwrap_or(false)))
        });
        let win = contributions(&sel, &|r| {
            f64::from(u8::from(r["foul_limit_win"].as_bool().unwrap_or(false)))
        });
        // 事前登録の**破滅率**（その手番だけで8反則以上）
        let cat = contributions(&sel, &|r| {
            f64::from(u8::from(r["immediate_catastrophe"].as_bool().unwrap_or(false)))
        });
        println!(
            "\n--- estimand {estimand}（決定点のある局 {} / 継続 {} 本）---",
            score.len(),
            sel.len()
        );
        println!(
            "  {:<18} {:>9} {:>26} {:>9} {:>9} {:>9} {:>9}",
            "arm", "Δ", "Δ − Δcurrent [CI]", "即時反則", "破滅%", "反則負け%", "相手反則負け%"
        );
        for arm in &arms {
            let (d, _, _) = delta_ci(&score, arm, None, games_total);
            let (dd, lo, hi) = delta_ci(&score, arm, Some(baseline), games_total);
            let (f, _, _) = delta_ci(&fouls, arm, None, games_total);
            let (c, _, _) = delta_ci(&cat, arm, None, games_total);
            let (fl, _, _) = delta_ci(&loss, arm, None, games_total);
            let (fw, _, _) = delta_ci(&win, arm, None, games_total);
            println!(
                "  {arm:<18} {d:>+9.4} {:>26} {f:>9.3} {:>9.1} {:>9.1} {:>9.1}",
                if arm.as_str() == baseline {
                    "—".to_string()
                } else {
                    format!("{dd:+.4} [{lo:+.4}, {hi:+.4}]")
                },
                100.0 * c,
                100.0 * fl,
                100.0 * fw,
            );
        }
        if !complete {
            continue;
        }
        // 門（issue #31 で事前登録）
        println!("  判定:");
        for arm in arms.iter().filter(|a| a.as_str() != baseline && *a != "baseline") {
            let (dd, lo, hi) = delta_ci(&score, arm, Some(baseline), games_total);
            let (df, _, _) = delta_ci(&fouls, arm, Some(baseline), games_total);
            let (dl, _, _) = delta_ci(&loss, arm, Some(baseline), games_total);
            let (dc, _, _) = delta_ci(&cat, arm, Some(baseline), games_total);
            // **full β だけが反則増を許される**（β-order は反則経済施策なので
            // 即時反則の非増加が必須。issue #31 の事前登録）
            let full_beta = arm.starts_with("beta@");
            let v = if estimand == "foul" {
                gate_foul(dd, lo, dl, dc)
            } else {
                gate_nofoul(dd, lo, df, dl, dc, full_beta)
            };
            // **主 arm を指定した実行は fail-closed**（レビュー11巡目 [P1]）
            if main_arm == Some(arm.as_str()) {
                gated.insert(estimand.to_string());
                if !v.pass {
                    failures.push(format!("estimand {estimand} / {arm}: {}", v.why));
                }
            }
            println!(
                "    {arm:<18} Δ差 {dd:+.4} [{lo:+.4}, {hi:+.4}] / 即時反則 {df:+.3} / \
破滅 {dc:+.4} / 反則負け {dl:+.4} → {}{}",
                if v.pass { "**" } else { "" },
                if v.pass { format!("{}**", v.why) } else { format!("不合格（{}）", v.why) }
            );
        }
    }
    if !complete {
        println!(
            "\n  **判定は出せない**（どれかの replicate でシャードが全 {shard_total} に\
揃っていない。Δ の分母は全対局数なので、欠けたシャードのぶんだけ分子だけが落ちて\
Δ が過小に出る）"
        );
    }
    println!(
        "\n  ※ baseline（実戦の反則列＋受理手を強制して指し直し）が 0 から離れているぶんは\
「単に指し直したから」の差。Δ から引いて読むこと"
    );
    println!(
        "  ※ **一手番の局所効果**であって全局反復適用の下界でも上界でもない\
（P1 の arena へ進むための有効性確認）"
    );
    // **主 arm を指定した実行は fail-closed**（PR #37 レビュー11巡目 [P1]）。
    // 表示するだけでは、不合格でも workflow は緑のまま終わる
    if let Some(main) = main_arm {
        // 事前登録は foul の +0.04 と nofoul の非劣性を**両方**門にしている。
        // 片方が標本に無いなら「通過」ではなく**判定不能**として落とす
        for estimand in ESTIMANDS {
            if !gated.contains(estimand) {
                failures.push(format!(
                    "estimand {estimand} で {main} の判定が出ていません（有効標本が無いか arm が欠けている = 判定不能。通過ではない）"
                ));
            }
        }
        if !complete {
            failures.push("シャードが揃っていません（Δ の分母が狂う）".into());
        }
        if failures.is_empty() {
            println!("\n  **判定: 通過**（主 arm {main}）");
        } else {
            println!("\n  **判定: 不通過**（主 arm {main}）");
            for f in &failures {
                println!("    - {f}");
            }
            std::process::exit(3);
        }
    }
    let mut reasons: BTreeMap<(String, String), u32> = BTreeMap::new();
    let mut think: Vec<f64> = vec![];
    for r in rows {
        *reasons
            .entry((
                r["arm"].as_str().unwrap_or("?").to_string(),
                r["reason"].as_str().unwrap_or("?").to_string(),
            ))
            .or_insert(0) += 1;
        think.push(r["think_mean_ms"].as_f64().unwrap_or(0.0));
    }
    println!("  終局理由:");
    for ((arm, reason), n) in &reasons {
        println!("    {arm} / {reason}: {n}");
    }
    println!(
        "  継続の平均思考: {:.0}ms",
        if think.is_empty() {
            0.0
        } else {
            think.iter().sum::<f64>() / think.len() as f64
        }
    );
}

/// 完全性検査の単位: `(replicate, game, move_number, estimand, seed, arm)`。
/// meta が宣言した集合と行の集合をこの粒度で厳密一致させる
type Key = (u64, String, u64, String, u64, String);

fn check_inputs(
    metas: &[serde_json::Value],
    rows: &[serde_json::Value],
    baseline: &str,
) -> Vec<String> {
    let mut out = vec![];
    if metas.is_empty() {
        return vec!["meta 行がありません（Δ の分母が取れない）".into()];
    }
    let first = &metas[0]["experiment"];
    if first["seeds"].as_u64().unwrap_or(0) == 0 {
        out.push("meta の seeds が 0 です（継続 0 局のまま Δ を 0 にできてしまう）".into());
    }
    // **読めなかった元対局があれば採否経路では落とす**（PR #37 レビュー13巡目 [P1]）。
    // `broken` は表示されるだけで meta にも失敗条件にも入っておらず、
    // 縮んだ `games` がそのまま Δ の分母になっていた（v13 / v14 で同数だけ
    // 失敗すれば相手間の `games` 一致検査も通る）。**欠測を「不明」で通さない**ので、
    // 列そのものが無い記録（schema 6 より前の生成）も落とす
    for m in metas {
        match m["experiment"]["broken"].as_u64() {
            Some(0) => {}
            Some(n) => out.push(format!(
                "読めなかった元対局が {n} 件あります（Δ の分母が縮んだ選択標本になる。記録を揃えてから測り直すこと）"
            )),
            None => out.push(
                "meta の experiment に broken がありません（壊れた元対局の本数が分からない記録です）"
                    .into(),
            ),
        }
    }
    // **必須列の欠落を「悪化なし」と読ませない**（schema の版だけに頼らない二重の関門）。
    // 安全性の列が欠けた行を 0 と読むと、反則負けも破滅も常に非悪化に見えて門を通せる
    {
        let mut missing: BTreeMap<&str, usize> = BTreeMap::new();
        for r in rows {
            for k in REQUIRED_ROW_KEYS {
                if r.get(k).is_none_or(serde_json::Value::is_null) {
                    *missing.entry(k).or_insert(0) += 1;
                }
            }
        }
        for (k, n) in missing {
            out.push(format!(
                "必須列 `{k}` が {n} 行で欠けています（欠測を 0 と読むと安全性の門が素通りする）"
            ));
        }
    }
    for (i, m) in metas.iter().enumerate().skip(1) {
        if m["experiment"] != *first {
            let diff: Vec<String> = first
                .as_object()
                .map(|o| {
                    o.keys()
                        .filter(|k| first[k.as_str()] != m["experiment"][k.as_str()])
                        .map(|k| {
                            format!("{k}: {} vs {}", first[k.as_str()], m["experiment"][k.as_str()])
                        })
                        .collect()
                })
                .unwrap_or_default();
            out.push(format!(
                "meta の実験キーが {i} 本目で食い違います（違う実験を混ぜている）: {}",
                diff.join(" / ")
            ));
        }
    }
    // **replicate ごとにシャードの完全性を検査する**（PR #33 レビュー3巡目 [P1]）。
    // 2回実行して平均するとき（issue #31 の「門付近なら全 arm を2回実行して平均」）、
    // 実験キーの一致検査が **同じ build**（`source_fingerprint`）であることを保証し、
    // ここが **各 replicate が全シャード揃っている**ことを保証する。
    // 手で report の表示値を平均すると、この2つがどちらも検証されない
    let total = first["shard_total"].as_u64().unwrap_or(1) as usize;
    let mut by_rep: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    for m in metas {
        by_rep
            .entry(m["replicate"].as_u64().unwrap_or(0))
            .or_default()
            .push(m["shard"].as_u64().unwrap_or(0));
    }
    for (rep, sh) in &mut by_rep {
        let n = sh.len();
        sh.sort_unstable();
        sh.dedup();
        if sh.len() != n {
            out.push(format!(
                "replicate {rep} に同じシャードの JSONL が2回あります（行が二重に数えられる）"
            ));
        }
        if sh.len() != total {
            out.push(format!(
                "replicate {rep} のシャードが欠けています（{sh:?} / 全 {total}）: Δ の分子だけが落ちる"
            ));
        }
    }
    // **meta が宣言する estimand を列挙として検査する**（PR #33 レビュー7巡目 [P1]）。
    // 期待キーは meta の値から作るので、meta と行が**同じ未知の値**で揃っていると
    // キーの厳密一致は通る。ところが集計が走査するのは `ESTIMANDS` の2つだけなので、
    // その決定点の全行が判定対象から無言で落ちる（実測: 1決定点を `unknown` に
    // 揃えると nofoul の層が 11局→10局・継続 352→320本 に減ったまま判定が出た）。
    // 任意の決定点を層から除外して点推定と安全性判定を動かせるので、ここで止める
    let mut bad_es: BTreeSet<String> = BTreeSet::new();
    for m in metas {
        for d in m["points_detail"].as_array().into_iter().flatten() {
            let es = d["estimand"].as_str().unwrap_or("?");
            if !ESTIMANDS.contains(&es) {
                bad_es.insert(es.to_string());
            }
        }
    }
    if !bad_es.is_empty() {
        out.push(format!(
            "meta が宣言した estimand に未知の値があります（{:?} / 既知は {ESTIMANDS:?}）: 集計の層から無言で落ちる",
            bad_es
        ));
    }
    // **replicate 間で決定点の母集団が同じか**（PR #33 レビュー5巡目 [P1]）。
    // 実験キーの一致検査が見るのは `experiment` だけで、`points_detail` はその外にある。
    // 期待キー集合は各 replicate が**自分の meta**から作るので、replicate ごとに行が
    // 完全でも、**別々の決定点母集団**の2本を「同じ実験の 2 replicate」として平均できる
    // （点推定も cluster bootstrap も「同じ標本を2回測った」前提なので契約が崩れる）。
    // 全シャードを合わせた `(game, move_number, estimand)` の集合が
    // replicate 間で完全一致することを要求する
    let mut pop: BTreeMap<u64, BTreeSet<(String, u64, String)>> = BTreeMap::new();
    for m in metas {
        let rep = m["replicate"].as_u64().unwrap_or(0);
        let e = pop.entry(rep).or_default();
        for d in m["points_detail"].as_array().into_iter().flatten() {
            e.insert((
                d["game"].as_str().unwrap_or("?").to_string(),
                d["move_number"].as_u64().unwrap_or(0),
                d["estimand"].as_str().unwrap_or("?").to_string(),
            ));
        }
    }
    if let Some((base_rep, base)) = pop.iter().next() {
        for (rep, s) in pop.iter().skip(1) {
            if s != base {
                let lack = base.difference(s).count();
                let more = s.difference(base).count();
                out.push(format!(
                    "replicate {rep} の決定点母集団が replicate {base_rep} と違います（欠け {lack} / 余分 {more}）: 別の標本を同じ実験の replicate として平均できない"
                ));
            }
        }
    }
    // 重複行（**replicate をキーに含める**。2回実行の同じ決定点は重複ではない）。
    // 重複の判定に `estimand` は入れない: 同じ unit の行が estimand 違いで2本あるのは
    // 重複（二重計上）であって別の決定点ではない
    let mut dup_keys: HashSet<(u64, String, u64, u64, String)> = HashSet::new();
    let mut keys: HashSet<Key> = HashSet::new();
    let mut dups = 0;
    for r in rows {
        let k = (
            r["replicate"].as_u64().unwrap_or(0),
            r["game"].as_str().unwrap_or("?").to_string(),
            r["move_number"].as_u64().unwrap_or(0),
            r["seed"].as_u64().unwrap_or(0),
            r["arm"].as_str().unwrap_or("?").to_string(),
        );
        // **`estimand` も実キーに入れる**（PR #33 レビュー6巡目 [P1]）。
        // `report` は**行側の** `estimand` で有効性側（foul）と非劣性側（nofoul）へ
        // 分けるので、meta と行で値が食い違うと片方の replicate が別の関門へ移り、
        // 結論が静かに変わる。未知の estimand も期待キーに無いので同時に弾ける
        let full = (
            k.0,
            k.1.clone(),
            k.2,
            r["estimand"].as_str().unwrap_or("?").to_string(),
            k.3,
            k.4.clone(),
        );
        keys.insert(full);
        if !dup_keys.insert(k) {
            dups += 1;
        }
    }
    if dups > 0 {
        out.push(format!("重複行が {dups} 件あります"));
    }
    // **期待キー集合と実キー集合を厳密に一致させる**（PR #33 レビュー4巡目 [P1]）。
    // 「(game, move_number, arm) の行数を**全 replicate 合算**で seeds × replicate 数と
    // 比べる」形だと、replicate 0 の欠測を replicate 1 の**範囲外 seed の余分行**が
    // 埋め合わせて通る（片方は欠測・片方は不正なのに「2 replicate の平均」が出る）。
    // 各 meta の `points_detail` × arm × seed 0..seeds × replicate を期待キー集合にして、
    // 欠測だけでなく**余分**（範囲外の seed・未宣言の決定点・meta に無い replicate
    // ラベル）も拒否する。シャードの完全性検査は meta の本数にしか掛からないので、
    // 行の完全性はここで replicate ごとに閉じる必要がある
    let seeds = first["seeds"].as_u64().unwrap_or(0);
    let mut want: Vec<String> = vec!["baseline".into()];
    for p in first["policies"].as_array().into_iter().flatten() {
        want.push(p.as_str().unwrap_or("?").to_string());
    }
    let mut want_keys: HashSet<Key> = HashSet::new();
    for m in metas {
        let rep = m["replicate"].as_u64().unwrap_or(0);
        for d in m["points_detail"].as_array().into_iter().flatten() {
            let g = d["game"].as_str().unwrap_or("?").to_string();
            let mn = d["move_number"].as_u64().unwrap_or(0);
            let es = d["estimand"].as_str().unwrap_or("?").to_string();
            // **`terminal` の欠測を false と読まない**（schema 3 の必須項目）。
            // 終端手番を「受理手のある手番」として扱うと `baseline` の行を
            // 期待してしまい、母集団の違いが欠落として現れて読み違える
            let Some(terminal) = d["terminal"].as_bool() else {
                die("meta の points_detail に terminal がありません（schema 3 未満の記録です）");
            };
            for arm in &want {
                // 終端手番は受理手が無いので `baseline` の行が存在しない
                if terminal && arm == "baseline" {
                    continue;
                }
                for seed in 0..seeds {
                    want_keys.insert((rep, g.clone(), mn, es.clone(), seed, arm.clone()));
                }
            }
        }
    }
    let show = |k: &Key| format!("rep{} {}#{}[{}] {} seed{}", k.0, k.1, k.2, k.3, k.5, k.4);
    let head = |v: &[String]| {
        format!(
            "{}{}",
            v.iter().take(3).cloned().collect::<Vec<_>>().join(" / "),
            if v.len() > 3 { " ..." } else { "" }
        )
    };
    let mut lacks: Vec<String> = want_keys.difference(&keys).map(show).collect();
    lacks.sort();
    if !lacks.is_empty() {
        out.push(format!(
            "meta が宣言した決定点に対して行が {} 件欠けています（Δ の分子が欠ける）: {}",
            lacks.len(),
            head(&lacks)
        ));
    }
    let mut extras: Vec<String> = keys.difference(&want_keys).map(show).collect();
    extras.sort();
    if !extras.is_empty() {
        out.push(format!(
            "meta が宣言していない行が {} 件あります（範囲外の seed・未宣言の決定点・meta と食い違う estimand・meta に無い replicate）: {}",
            extras.len(),
            head(&extras)
        ));
    }
    out.extend(check_arm_order_balance(metas, rows, baseline));
    out
}

/// **実行順が決定点の中で釣り合っているか**（PR #37 レビュー8〜9巡目 [P1]）。
///
/// `plan` の「seed が偶数なら AB/BA が閉じる」という契約は、生成規則
/// （`schedule_groups` の反転）と**この集計側の検査**が揃って初めて意味を持つ。
///
/// 数えるのは `(replicate, game, move_number, treatment)` ごとの
/// 「treatment 先 / 対照 先」で、**同じ強制列に畳まれた unit は数えない**
/// （同じ継続1本を共有するのでペア差が厳密に 0 = 実行順効果を持たない）。
/// 畳まれたかどうかは **`continuation_group` の一致**で見る:
/// `arm_order` の一致で代用すると、order を書き換えるだけで検査を空集合にできる
/// （レビュー9巡目 [P1]）。
///
/// **差は完全一致を要求する**（同 [P1]）。±1 を許すと、対の片方だけが分離した
/// 決定点で「先 1 / 後 0」が通り、ペア差が非ゼロになりうる唯一の unit が
/// treatment 先だけになるので、加法的な実行順効果が丸ごと主差へ残る。
/// 生成側で対照と treatment を畳まなくしたので（レビュー10巡目 [P1]）、
/// 全 unit が常に分離し、偶数 seed の反転だけで**除外なしに**閉じる。
/// 閉じていなければここで落とす（fail-closed。事後の除外はしない）。
fn check_arm_order_balance(
    metas: &[serde_json::Value],
    rows: &[serde_json::Value],
    baseline: &str,
) -> Vec<String> {
    let mut out = vec![];
    let Some(m0) = metas.first() else { return out };
    let first = &m0["experiment"];
    // スケジュールが反転させた対照と、集計が使う対照が違えば AB/BA が
    // 閉じているのは別のペア
    match first["schedule_control"].as_str() {
        Some(c) if c == baseline => {}
        Some(c) => out.push(format!(
            "スケジュールの対照は {c} ですが集計の対照は {baseline} です（AB/BA が閉じているのは別のペアなので、この主差には実行順効果が残ります）"
        )),
        None => out.push(
            "meta に schedule_control がありません（実行順の生成規則が分からない schema 3 以前の記録です）"
                .into(),
        ),
    }
    // ---- `arm_order` そのものの健全性（レビュー9巡目 [P1]）--------------------
    // 型と unit 内の構造を検査しない限り、order を文字列にする・treatment と
    // 対照へ同じ値を入れる、だけで下の均衡検査を空集合にできる
    let mut non_int = 0usize;
    let mut bad_group = 0usize;
    for r in rows {
        if !r["arm_order"].is_u64() {
            non_int += 1;
        }
        // **`continuation_group` の型と書式も検査する**（レビュー10巡目 [P1]）。
        // 非 null しか見ていなかったので、数値へ置き換えるだけで `unit_index` から
        // 全行が消え、均衡検査も主判定も空集合のまま exit 0 になった
        let ok = r["continuation_group"].as_str().is_some_and(|g| {
            g.len() == 16 && g.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        });
        if !ok {
            bad_group += 1;
        }
    }
    if non_int > 0 {
        out.push(format!(
            "arm_order が整数でない行が {non_int} 件あります（型を変えるだけで実行順の検査を素通りできる）"
        ));
    }
    if bad_group > 0 {
        out.push(format!(
            "continuation_group が16桁の小文字hexでない行が {bad_group} 件あります（型を変えるだけで実行順の検査と主判定を空集合にできる）"
        ));
    }
    let units = unit_index(rows);
    // **期待した行がすべて `unit_index` に入ったこと**まで照合する（同 [P1]）。
    // 索引に入らなかった行は均衡検査からも主判定からも静かに消える
    let indexed: usize = units.values().map(BTreeMap::len).sum();
    if indexed != rows.len() {
        out.push(format!(
            "実行順の索引へ入らなかった行が {} 件あります（arm_order / continuation_group が読めない行）",
            rows.len() - indexed
        ));
    }
    let mut struct_bad: Vec<String> = vec![];
    let mut shared_bad: Vec<String> = vec![];
    for (u, arms) in &units {
        // `arm_order` と `continuation_group` は1対1（同じ継続を共有した組だけが
        // 同じ order）。順位の集合は 0..グループ数 と厳密に一致する
        let mut by_group: BTreeMap<&str, BTreeSet<u64>> = BTreeMap::new();
        let mut by_order: BTreeMap<u64, BTreeSet<&str>> = BTreeMap::new();
        for a in arms.values() {
            by_group.entry(a.group.as_str()).or_default().insert(a.order);
            by_order.entry(a.order).or_default().insert(a.group.as_str());
        }
        let orders: BTreeSet<u64> = by_order.keys().copied().collect();
        let want: BTreeSet<u64> = (0..by_group.len() as u64).collect();
        if orders != want
            || by_group.values().any(|o| o.len() != 1)
            || by_order.values().any(|g| g.len() != 1)
        {
            struct_bad.push(format!("rep{} {}#{} seed{}", u.0, u.1, u.2, u.3));
        }
        // 同じ継続を共有したなら、継続の結果は arm によらず完全一致するはず
        let mut seen: BTreeMap<&str, &ArmRow> = BTreeMap::new();
        for a in arms.values() {
            match seen.get(a.group.as_str()) {
                None => {
                    seen.insert(a.group.as_str(), a);
                }
                Some(prev) if prev.outcome != a.outcome => {
                    shared_bad.push(format!("rep{} {}#{} seed{}", u.0, u.1, u.2, u.3));
                }
                Some(_) => {}
            }
        }
    }
    struct_bad.dedup();
    shared_bad.dedup();
    if !struct_bad.is_empty() {
        out.push(format!(
            "arm_order と continuation_group が1対1でない unit が {} 件あります（順位は 0..グループ数 と一致し、同じ order は同じ強制列でなければならない）: {}",
            struct_bad.len(),
            struct_bad.iter().take(3).cloned().collect::<Vec<_>>().join(" / ")
        ));
    }
    if !shared_bad.is_empty() {
        out.push(format!(
            "同じ continuation_group なのに継続の結果が違う unit が {} 件あります（畳まれた組は同じ継続1本を配ったはず）: {}",
            shared_bad.len(),
            shared_bad.iter().take(3).cloned().collect::<Vec<_>>().join(" / ")
        ));
    }
    // ---- 決定点の中で前後が完全に釣り合っているか ---------------------------
    let treatments: Vec<String> = first["policies"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|v| v.as_str())
        .filter(|t| *t != baseline)
        .map(|t| t.to_string())
        .collect();
    for treat in &treatments {
        let mut per_point: BTreeMap<(u64, String, u64), (usize, usize)> = BTreeMap::new();
        for (u, arms) in &units {
            let (Some(a), Some(b)) = (arms.get(treat.as_str()), arms.get(baseline)) else {
                continue;
            };
            if a.group == b.group {
                continue; // 同じ継続1本（ペア差は厳密に 0）
            }
            let e = per_point.entry((u.0, u.1.clone(), u.2)).or_default();
            if a.order < b.order {
                e.0 += 1
            } else {
                e.1 += 1
            }
        }
        let bad: Vec<String> = per_point
            .iter()
            .filter(|(_, (f, s))| f != s)
            .map(|((rep, g, mn), (f, s))| format!("rep{rep} {g}#{mn}（先 {f} / 後 {s}）"))
            .collect();
        if !bad.is_empty() {
            out.push(format!(
                "{treat} と {baseline} の実行順が決定点の中で均衡していません（{} 決定点）: {}{}。AB/BA が閉じていないと実行順効果がペア差に残ります",
                bad.len(),
                bad.iter().take(3).cloned().collect::<Vec<_>>().join(" / "),
                if bad.len() > 3 { " ..." } else { "" }
            ));
        }
    }
    out
}

/// 実行順の監査に要る1行ぶんの情報
struct ArmRow {
    order: u64,
    group: String,
    /// 継続の結果（同じ `group` なら arm によらず一致するはず）
    outcome: String,
}

type UnitKey = (u64, String, u64, u64);

/// `(replicate, game, move_number, seed)` → arm → 実行順・グループ・結果
fn unit_index(rows: &[serde_json::Value]) -> BTreeMap<UnitKey, BTreeMap<String, ArmRow>> {
    let mut units: BTreeMap<UnitKey, BTreeMap<String, ArmRow>> = BTreeMap::new();
    for r in rows {
        let Some(order) = r["arm_order"].as_u64() else { continue };
        let Some(group) = r["continuation_group"].as_str() else { continue };
        units
            .entry((
                r["replicate"].as_u64().unwrap_or(u64::MAX),
                r["game"].as_str().unwrap_or("?").to_string(),
                r["move_number"].as_u64().unwrap_or(0),
                r["seed"].as_u64().unwrap_or(u64::MAX),
            ))
            .or_default()
            .insert(
                r["arm"].as_str().unwrap_or("?").to_string(),
                ArmRow {
                    order,
                    group: group.to_string(),
                    outcome: [
                        "score",
                        "reason",
                        "plies",
                        "immediate_fouls",
                        "foul_limit_loss",
                        "foul_limit_win",
                        "immediate_catastrophe",
                    ]
                    .iter()
                    .map(|k| r[*k].to_string())
                    .collect::<Vec<_>>()
                    .join("|"),
                },
            );
    }
    units
}


/// 事前登録した門の判定結果（issue #36 P0-2b）。
#[derive(Debug, Clone, PartialEq)]
struct Verdict {
    pass: bool,
    why: String,
}

/// 反則あり estimand の門: `Δ差 ≥ +0.04` かつ CI 下限 > 0 かつ安全性が悪化しない
fn gate_foul(dd: f64, lo: f64, dl: f64, dc: f64) -> Verdict {
    let safe = dl <= 0.0 && dc <= 0.0;
    let effect = dd >= 0.04 && lo > 0.0;
    let mut why = vec![];
    if !effect {
        why.push("改善量か CI が門に届かない".to_string());
    }
    if dl > 0.0 {
        why.push("反則負けが悪化".to_string());
    }
    if dc > 0.0 {
        why.push("破滅率が悪化".to_string());
    }
    Verdict {
        pass: effect && safe,
        why: if why.is_empty() { "門を超える".into() } else { why.join("・") },
    }
}

/// 反則0 estimand の門: 非劣性 かつ 即時反則・安全性が悪化しない
fn gate_nofoul(dd: f64, lo: f64, df: f64, dl: f64, dc: f64, full_beta: bool) -> Verdict {
    let noninferior = dd >= -0.01 && lo > -0.02;
    let safe = dl <= 0.0 && dc <= 0.0;
    let foul_ok = df <= 0.0 || full_beta;
    let mut why = vec![];
    if !noninferior {
        why.push("非劣性を満たさない".to_string());
    }
    if df > 0.0 && !full_beta {
        why.push("即時反則が増えている".to_string());
    }
    if dl > 0.0 {
        why.push("反則負けが悪化".to_string());
    }
    if dc > 0.0 {
        why.push("破滅率が悪化".to_string());
    }
    Verdict {
        pass: noninferior && safe && foul_ok,
        why: if why.is_empty() { "非劣性を満たす".into() } else { why.join("・") },
    }
}

/// **相手をまたいだ最終判定**（issue #36 P0-2b の事前登録。`check_policy combined`
/// と同じ契約）。`Δcombined = (Δv13 + Δv14)/2` を**層化 cluster bootstrap**
/// （各相手の内側で元対局を引き直す）で出し、veto「両相手とも同符号」と
/// 安全性 veto を **fail-closed**（不通過なら exit 3）で判定する。
///
/// 相手ごとの `report` を並べるだけでは、主 CI 下限も veto も検査されない
/// （PR #37 レビュー11巡目 [P1]）。
fn run_combined(args: &[String]) {
    let mut allow_incomplete = false;
    let mut paths: Vec<String> = vec![];
    let mut baseline = "current@real".to_string();
    let mut main_arm = "oracle@kinf@real".to_string();
    let mut expect_opponents = "estimator_v13,estimator_v14".to_string();
    // **継続は常に2 replicate**（issue #36 の事前登録。同 [P1]）
    let mut expect_replicates: usize = 2;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut need = |k: &str| {
            it.next().unwrap_or_else(|| die(&format!("{k} に値がありません"))).clone()
        };
        match a.as_str() {
            "--allow-incomplete" => allow_incomplete = true,
            "--baseline" => baseline = need("--baseline"),
            "--main" => main_arm = need("--main"),
            "--expect-opponents" => expect_opponents = need("--expect-opponents"),
            "--expect-replicates" => {
                expect_replicates = need("--expect-replicates")
                    .parse()
                    .unwrap_or_else(|_| die("--expect-replicates は整数"))
            }
            x if x.starts_with("--") => die(&format!("未知のオプション: {x}")),
            x => paths.push(x.to_string()),
        }
    }
    if paths.is_empty() {
        die("combined には各相手の JSONL を指定してください");
    }
    // 相手ごとに meta / rows を分ける（1ファイル = 1シャード = 1相手）
    let mut by_opp: BTreeMap<String, (Vec<serde_json::Value>, Vec<serde_json::Value>)> =
        BTreeMap::new();
    for p in &paths {
        let text =
            std::fs::read_to_string(p).unwrap_or_else(|e| die(&format!("{p} を読めません: {e}")));
        let mut metas = vec![];
        let mut rows = vec![];
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let v: serde_json::Value = serde_json::from_str(line)
                .unwrap_or_else(|_| die(&format!("{p}: JSON として読めない行があります")));
            if v["schema"].as_u64() != Some(u64::from(ROW_SCHEMA)) {
                die(&format!(
                    "{p}: schema {} は集計できません（現行 {ROW_SCHEMA}）",
                    v["schema"]
                ));
            }
            if v["type"] == "meta" { metas.push(v) } else { rows.push(v) }
        }
        let Some(opp) = metas
            .first()
            .and_then(|m| m["experiment"]["opponent"].as_str())
            .map(str::to_string)
        else {
            die(&format!("{p}: meta の experiment.opponent がありません"));
        };
        let e = by_opp.entry(opp).or_default();
        e.0.extend(metas);
        e.1.extend(rows);
    }
    let want: BTreeSet<String> = expect_opponents
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    let got: BTreeSet<String> = by_opp.keys().cloned().collect();
    if !want.is_empty() && want != got {
        die(&format!("相手が契約と違います: 期待 {want:?} / 実際 {got:?}"));
    }
    // **相手以外の実験条件が相手間で同一であること**（PR #37 レビュー12巡目 [P1]）。
    // 契約上「相手だけが違う」ので、片方を別の build / 別の予算 / 別の seed 数 /
    // 別の実効並列度で測った JSONL を平均してはいけない。`opponent` と、相手ごとに
    // 必ず違う記録の指紋（`records`）だけを除いて比較する
    let mut failures: Vec<String> = vec![];
    {
        let key_of = |metas: &Vec<serde_json::Value>| -> serde_json::Value {
            let mut e = metas
                .first()
                .map(|m| m["experiment"].clone())
                .unwrap_or(serde_json::Value::Null);
            if let Some(o) = e.as_object_mut() {
                o.remove("opponent");
                o.remove("records");
            }
            e
        };
        let mut it = by_opp.iter();
        if let Some((first_opp, (first_metas, _))) = it.next() {
            let base = key_of(first_metas);
            for (opp, (metas, _)) in it {
                let k = key_of(metas);
                if k != base {
                    let diffs: Vec<String> = base
                        .as_object()
                        .into_iter()
                        .flatten()
                        .filter(|(key, v)| k.get(key.as_str()) != Some(*v))
                        .map(|(key, v)| {
                            format!("{key}: {first_opp}={v} vs {opp}={}", k[key.as_str()])
                        })
                        .collect();
                    failures.push(format!(
                        "相手間で実験条件が違います（相手以外は同一でなければならない）: {}",
                        diffs.join(" / ")
                    ));
                }
            }
        }
    }
    println!("== P0-2b 合算判定（主 arm {main_arm} vs 対照 {baseline}）==");
    // 相手ごとの検査（入力契約・replicate 数）
    for (opp, (metas, rows)) in &by_opp {
        for msg in check_inputs(metas, rows, &baseline) {
            if allow_incomplete {
                eprintln!("警告: [{opp}] {msg}");
            } else {
                failures.push(format!("[{opp}] {msg}"));
            }
        }
        let reps: BTreeSet<u64> = metas
            .iter()
            .map(|m| m["replicate"].as_u64().unwrap_or(0))
            .collect();
        if expect_replicates > 0 && reps.len() != expect_replicates {
            failures.push(format!(
                "[{opp}] replicate が {} 本しかありません（事前登録は {expect_replicates} 本: {reps:?}）",
                reps.len()
            ));
        }
    }
    // estimand ごとに層化 cluster bootstrap
    for estimand in ESTIMANDS {
        // **両 estimand とも有効標本が要る**（同 [P1]）。片方が空のまま
        // 「通過」にすると、foul の +0.04 か nofoul の非劣性のどちらかを
        // 一度も評価せずに成功できる
        let mut per_opp: Vec<(String, StratumContrib)> = vec![];
        for (opp, (metas, rows)) in &by_opp {
            let sel: Vec<&serde_json::Value> =
                rows.iter().filter(|r| r["estimand"] == estimand).collect();
            let games = metas
                .first()
                .and_then(|m| m["experiment"]["games"].as_u64())
                .unwrap_or(0) as usize;
            if sel.is_empty() || games == 0 {
                failures.push(format!(
                    "[{opp}] estimand {estimand} の有効標本がありません（判定不能。通過ではない）"
                ));
                continue;
            }
            // **対照と主 arm の存在を fail-closed で検査する**（同 [P1]）。
            // 欠落を `unwrap_or(0.0)` で補うと、対照の行が一貫して無い入力でも
            // 「暗黙のゼロ対照」と比べて合格しうる（対象の手番が無かった局を
            // ゼロ寄与の cluster として数える padding とは別物）
            let arms: BTreeSet<&str> =
                sel.iter().filter_map(|r| r["arm"].as_str()).collect();
            let mut missing = vec![];
            if !arms.contains(baseline.as_str()) {
                missing.push(baseline.clone());
            }
            if !arms.contains(main_arm.as_str()) {
                missing.push(main_arm.clone());
            }
            if !missing.is_empty() {
                failures.push(format!(
                    "[{opp}] estimand {estimand} に arm {missing:?} の行がありません（暗黙のゼロ対照と比べてはいけない）"
                ));
                continue;
            }
            // 決定点のある局は**両 arm とも**寄与を持つはず（片方だけの局を
            // 0 で埋めると差が水増しされる）
            let sc = contributions(&sel, &|r| r["score"].as_f64().unwrap_or(0.0));
            let lopsided: Vec<&String> = sc
                .iter()
                .filter(|(_, per_arm)| {
                    per_arm.contains_key(&baseline) != per_arm.contains_key(&main_arm)
                })
                .map(|(g, _)| g)
                .collect();
            if !lopsided.is_empty() {
                failures.push(format!(
                    "[{opp}] estimand {estimand}: 片方の arm しか寄与を持たない局が {} 件あります: {:?}",
                    lopsided.len(),
                    lopsided.iter().take(3).collect::<Vec<_>>()
                ));
                continue;
            }
            per_opp.push((
                opp.clone(),
                StratumContrib {
                    score: sc,
                    fouls: contributions(&sel, &|r| {
                        r["immediate_fouls"].as_f64().unwrap_or(0.0)
                    }),
                    loss: contributions(&sel, &|r| {
                        f64::from(u8::from(r["foul_limit_loss"].as_bool().unwrap_or(false)))
                    }),
                    cat: contributions(&sel, &|r| {
                        f64::from(u8::from(
                            r["immediate_catastrophe"].as_bool().unwrap_or(false),
                        ))
                    }),
                    games,
                },
            ));
        }
        if per_opp.len() != by_opp.len() {
            continue; // 上で failure を積んである
        }
        let (dd, lo, hi) =
            stratified_delta_ci(&per_opp, |c| &c.score, &main_arm, &baseline);
        let (df, _, _) = stratified_delta_ci(&per_opp, |c| &c.fouls, &main_arm, &baseline);
        let (dl, _, _) = stratified_delta_ci(&per_opp, |c| &c.loss, &main_arm, &baseline);
        let (dc, _, _) = stratified_delta_ci(&per_opp, |c| &c.cat, &main_arm, &baseline);
        // veto: 相手ごとの点推定が同符号（片方でも逆なら不通過）
        let signs: Vec<(String, f64)> = per_opp
            .iter()
            .map(|(o, c)| {
                (o.clone(), delta_ci(&c.score, &main_arm, Some(&baseline), c.games).0)
            })
            .collect();
        let v = if estimand == "foul" {
            gate_foul(dd, lo, dl, dc)
        } else {
            gate_nofoul(dd, lo, df, dl, dc, main_arm.starts_with("beta@"))
        };
        let veto_ok = if estimand == "foul" {
            signs.iter().all(|(_, d)| *d > 0.0)
        } else {
            true // 非劣性側は符号 veto を掛けない（門そのものが下側の制約）
        };
        println!(
            "  estimand {estimand}: Δ差 {dd:+.4} [{lo:+.4}, {hi:+.4}] / 即時反則 {df:+.3} / \
破滅 {dc:+.4} / 反則負け {dl:+.4} → {} {}",
            if v.pass && veto_ok { "通過" } else { "不通過" },
            v.why
        );
        for (o, d) in &signs {
            println!("    {o}: Δ差 {d:+.4}");
        }
        if !v.pass {
            failures.push(format!("estimand {estimand}: {}", v.why));
        }
        if !veto_ok {
            failures.push(format!(
                "estimand {estimand}: 相手ごとの符号 veto に掛かりました（{signs:?}）"
            ));
        }
    }
    if failures.is_empty() {
        println!("\n  **判定: 通過**");
    } else {
        println!("\n  **判定: 不通過**");
        for f in &failures {
            println!("    - {f}");
        }
        std::process::exit(3);
    }
}

/// 1相手ぶんの寄与（層化 bootstrap の1層）
struct StratumContrib {
    score: BTreeMap<String, BTreeMap<String, f64>>,
    fouls: BTreeMap<String, BTreeMap<String, f64>>,
    loss: BTreeMap<String, BTreeMap<String, f64>>,
    cat: BTreeMap<String, BTreeMap<String, f64>>,
    games: usize,
}

/// `Δcombined = 平均_相手(Δ相手)` の層化 cluster bootstrap
/// （**各相手の内側で元対局を引き直す**）
fn stratified_delta_ci(
    per_opp: &[(String, StratumContrib)],
    pick: impl Fn(&StratumContrib) -> &BTreeMap<String, BTreeMap<String, f64>>,
    arm: &str,
    baseline: &str,
) -> (f64, f64, f64) {
    let strata: Vec<Vec<f64>> = per_opp
        .iter()
        .map(|(_, c)| {
            let m = pick(c);
            let mut v: Vec<f64> = m
                .values()
                .map(|per_arm| {
                    per_arm.get(arm).copied().unwrap_or(0.0)
                        - per_arm.get(baseline).copied().unwrap_or(0.0)
                })
                .collect();
            // 対象の手番が無かった局も寄与 0 の cluster として入れる
            v.resize(v.len().max(c.games), 0.0);
            v
        })
        .collect();
    if strata.iter().any(Vec::is_empty) {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let mean = |vs: &[Vec<f64>]| -> f64 {
        vs.iter().map(|v| v.iter().sum::<f64>() / v.len() as f64).sum::<f64>() / vs.len() as f64
    };
    let point = mean(&strata);
    let mut state: u64 = 0x5f2b_9c31;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let reps = 2000;
    let mut draws: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let resampled: Vec<Vec<f64>> = strata
            .iter()
            .map(|v| (0..v.len()).map(|_| v[next() % v.len()]).collect())
            .collect();
        draws.push(mean(&resampled));
    }
    draws.sort_by(f64::total_cmp);
    (
        point,
        draws[(reps as f64 * 0.025) as usize],
        draws[(reps as f64 * 0.975) as usize],
    )
}

fn run_report(args: &[String]) {
    let allow_incomplete = args.iter().any(|a| a == "--allow-incomplete");
    // 対照 arm（既定 `current`）。issue #36 P0-2b は `--baseline current@real`
    let mut baseline = "current".to_string();
    // **主 arm を指定したら fail-closed**（PR #37 レビュー11巡目 [P1]）。
    // 指定しなければ情報表示だけ。
    //
    // **相手別の集計へ渡してはいけない**（同 12巡目 [P1]）: `--main` は
    // 「Δ ≥ +0.04 かつ**その標本の** CI 下限 > 0」を要求するが、issue #36 の
    // 事前登録は「**合算**で +0.04・合算 CI 下限 > 0」＋「相手別は**点推定の
    // 符号** veto だけ」で、104局×2 で相手別の CI 下限まで正にするのは強すぎると
    // 明記されている。**合否を執行するのは `combined` だけ**
    let mut main_arm: Option<String> = None;
    let mut paths: Vec<String> = vec![];
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--allow-incomplete" => i += 1,
            "--main" => {
                main_arm = Some(
                    args.get(i + 1).cloned().unwrap_or_else(|| die("--main には arm 名が必要です")),
                );
                i += 2;
            }
            "--baseline" => {
                baseline = args
                    .get(i + 1)
                    .cloned()
                    .unwrap_or_else(|| die("--baseline には arm 名が必要です"));
                i += 2;
            }
            a if a.starts_with("--") => die(&format!("未知のオプション: {a}")),
            a => {
                paths.push(a.to_string());
                i += 1;
            }
        }
    }
    if paths.is_empty() {
        die("report には JSONL を指定してください");
    }
    let mut metas = vec![];
    let mut rows = vec![];
    for p in &paths {
        let text =
            std::fs::read_to_string(p).unwrap_or_else(|e| die(&format!("{p} を読めません: {e}")));
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(line)
                .unwrap_or_else(|_| die(&format!("{p}: JSON として読めない行があります")));
            let schema = v["schema"].as_u64().unwrap_or(0) as u32;
            if schema != ROW_SCHEMA {
                die(&format!(
                    "{p}: schema {schema} は集計できません（現行 {ROW_SCHEMA}）"
                ));
            }
            if v["type"] == "meta" {
                metas.push(v);
            } else {
                rows.push(v);
            }
        }
    }
    println!("JSONL {} 本 / 行 {} / 対照 {baseline}", paths.len(), rows.len());
    report_vs(&metas, &rows, allow_incomplete, &baseline, main_arm.as_deref());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exp() -> serde_json::Value {
        serde_json::json!({
            "opponent": "estimator_v14", "budget_ms": 700, "seeds": 2,
            "policies": ["current", "alpha@k2"], "policy_jobs": 3, "continuation_jobs": 3,
            "schedule_control": "current",
            "games": 104, "broken": 0, "shard_total": 1, "config": "c",
            "source_fingerprint": "s", "records": "r",
        })
    }

    fn meta(e: serde_json::Value, shard: u64) -> serde_json::Value {
        serde_json::json!({
            "schema": ROW_SCHEMA, "type": "meta", "experiment": e, "shard": shard,
            "replicate": 0,
            "games": 104, "points": 1,
            "points_detail": [{"game": "g1", "move_number": 41, "estimand": "foul",
                               "terminal": false}],
        })
    }

    fn row(arm: &str, seed: u64, score: f64) -> serde_json::Value {
        serde_json::json!({
            "schema": ROW_SCHEMA, "game": "g1", "move_number": 41, "estimand": "foul",
            "seed": seed, "arm": arm, "score": score, "reason": "checkmate",
            "plies": 90, "added_plies": 10, "immediate_fouls": 1, "added_fouls_me": 1,
            "think_mean_ms": 700.0, "foul_limit": false,
            "foul_limit_loss": false, "foul_limit_win": false,
            "immediate_catastrophe": false, "replicate": 0,
            // 実行順とグループは arm ごとの slot に紐づける（強制列が違えば
            // グループも違う、という本番の関係を再現する）
            "arm_order": arm_slot(arm), "continuation_group": grp(arm_slot(arm)),
        })
    }

    /// 16桁 hex の `continuation_group`（本番と同じ書式）
    fn grp(slot: usize) -> String {
        format!("{slot:016x}")
    }

    /// 固定の arm 優先順位（`schedule_groups` の前向きの並び）
    fn arm_slot(arm: &str) -> usize {
        match arm {
            "baseline" => 0,
            "current" => 1,
            _ => 2,
        }
    }

    /// **2回実行の平均は集計器が取る**（PR #33 レビュー3巡目 [P1]）。
    /// 局ごとの寄与を replicate 間で平均してから Δ を作るので、点推定は
    /// 「各 replicate の Δ の平均」に厳密に一致する（手で表示値を平均しない）
    #[test]
    fn replicateの平均は各replicateのdeltaの平均に一致する() {
        let mk = |rep: u64, score: f64| -> Vec<serde_json::Value> {
            let mut v = vec![];
            for seed in 0..2u64 {
                for (arm, sc) in [("baseline", 0.5), ("current", 0.5), ("alpha@k2", score)] {
                    // 偶数 seed は前向き、奇数は反転（本番の `schedule_groups` と同じ）
                    let slot = arm_slot(arm);
                    let order = if seed % 2 == 0 { slot } else { 2 - slot };
                    let mut r = row_ord(arm, seed, sc, order);
                    r["replicate"] = serde_json::json!(rep);
                    v.push(r);
                }
            }
            v
        };
        // replicate 0 は alpha@k2 が 1.0、replicate 1 は 0.0 → 平均 0.5 = current と同じ
        let mut rows = mk(0, 1.0);
        rows.extend(mk(1, 0.0));
        let mut m0 = meta(exp(), 0);
        m0["replicate"] = serde_json::json!(0);
        let mut m1 = meta(exp(), 0);
        m1["replicate"] = serde_json::json!(1);
        assert!(
            check_inputs(&[m0, m1], &rows, "current").is_empty(),
            "同じ実験の2 replicate は契約を通る"
        );
        let sel: Vec<&serde_json::Value> = rows.iter().collect();
        let contrib = contributions(&sel, &|r| r["score"].as_f64().unwrap_or(0.0));
        let (d, _, _) = delta_ci(&contrib, "alpha@k2", Some("current"), 104);
        assert!(
            d.abs() < 1e-9,
            "1.0 と 0.0 の平均は current と同じ = Δ差 0 のはず: {d}"
        );
    }

    /// replicate ごとにシャードの完全性を検査する（片方だけ欠けても止まる）
    #[test]
    fn replicateごとにシャードの欠落を検出する() {
        let mut e = exp();
        e["shard_total"] = serde_json::json!(2);
        let mut m0a = meta(e.clone(), 0);
        m0a["replicate"] = serde_json::json!(0);
        let mut m0b = meta(e.clone(), 1);
        m0b["replicate"] = serde_json::json!(0);
        // replicate 1 はシャード 0 しか無い
        let mut m1 = meta(e, 0);
        m1["replicate"] = serde_json::json!(1);
        let problems = check_inputs(&[m0a, m0b, m1], &[], "current");
        assert!(
            problems.iter().any(|p| p.contains("replicate 1")),
            "replicate 1 の欠落を検出できていない: {problems:?}"
        );
    }

    /// **安全性の列を落とした行は集計させない**（PR #33 レビュー2巡目 [P1]）。
    /// 修正前（schema 1）の artifact には `foul_limit_loss` / `immediate_catastrophe` が
    /// 無く、`unwrap_or(false)` で 0 と読むと**反則負けも破滅も常に非悪化に見えて
    /// 門を通せる**。版の拒否と必須列の検査の両方で止まることを固定する
    #[test]
    fn 安全性の列が欠けた行は集計させない() {
        for key in [
            "foul_limit_loss",
            "foul_limit_win",
            "immediate_catastrophe",
            "score",
            "arm_order",
        ] {
            let rows: Vec<serde_json::Value> = full()
                .into_iter()
                .map(|mut r| {
                    if let Some(o) = r.as_object_mut() {
                        o.remove(key);
                    }
                    r
                })
                .collect();
            let problems = check_inputs(&[meta(exp(), 0)], &rows, "current");
            assert!(
                problems.iter().any(|p| p.contains(key)),
                "{key} が欠けても素通りした: {problems:?}"
            );
        }
    }

    /// 旧 schema は版の時点で弾く（`reaggregate_run_id` で修正前の artifact を
    /// 読み直したときに、欠けた列を 0 と読んで門を通すのを防ぐ二重の関門の片方）
    #[test]
    fn 旧schemaは現行の版と一致しない() {
        assert_eq!(
            ROW_SCHEMA, 6,
            "対照と treatment を畳まなくして continuation_group の指紋が変わったので版を上げてある"
        );
    }

    /// **`reason == "foul_limit"` は bot の負けとは限らない**（PR #33 レビュー [P1]）。
    /// 相手が反則負けして bot が勝った終局を「反則負け」に数えると、相手の自滅を
    /// 増やした方策を安全性ゲートで落としてしまう
    #[test]
    fn 反則負けは自分の負けだけを数える() {
        let win = serde_json::json!({
            "reason": "foul_limit", "score": 1.0,
            "foul_limit": true, "foul_limit_loss": false, "foul_limit_win": true,
        });
        let lose = serde_json::json!({
            "reason": "foul_limit", "score": 0.0,
            "foul_limit": true, "foul_limit_loss": true, "foul_limit_win": false,
        });
        // 素の `foul_limit` はどちらも真 = 区別できない
        assert_eq!(win["foul_limit"], lose["foul_limit"]);
        // 集計が見るのは `foul_limit_loss` の方
        assert_eq!(win["foul_limit_loss"], serde_json::json!(false));
        assert_eq!(lose["foul_limit_loss"], serde_json::json!(true));
    }

    /// 実行順つきの行（`schedule_groups` と同じ形: seed の偶奇で並びを反転する）。
    /// `continuation_group` は**強制列**の指紋なので反転では変わらない
    fn row_ord(arm: &str, seed: u64, score: f64, order: usize) -> serde_json::Value {
        let mut r = row(arm, seed, score);
        r["arm_order"] = serde_json::json!(order);
        r
    }

    fn full() -> Vec<serde_json::Value> {
        let mut v = vec![];
        for seed in 0..2u64 {
            // 偶数 seed は [baseline, current, alpha@k2]、奇数はその反転
            let fwd = seed % 2 == 0;
            let ord = |k: usize| if fwd { k } else { 2 - k };
            v.push(row_ord("baseline", seed, 0.0, ord(0)));
            v.push(row_ord("current", seed, 0.0, ord(1)));
            v.push(row_ord("alpha@k2", seed, 1.0, ord(2)));
        }
        v
    }

    #[test]
    fn 揃った入力は契約を通る() {
        let problems = check_inputs(&[meta(exp(), 0)], &full(), "current");
        assert!(problems.is_empty(), "{problems:?}");
    }

    #[test]
    fn 実行順が決定点の中で偏っていたら止まる() {
        // 全 seed で `alpha@k2` が `current` より後 = 反転が効いていない
        // （schema 3 までの cyclic rotate で実際に起きていた形）
        let rows: Vec<serde_json::Value> = full()
            .into_iter()
            .map(|mut r| {
                let o = match r["arm"].as_str().unwrap() {
                    "baseline" => 0,
                    "current" => 1,
                    _ => 2,
                };
                r["arm_order"] = serde_json::json!(o);
                r
            })
            .collect();
        let problems = check_inputs(&[meta(exp(), 0)], &rows, "current");
        assert!(
            problems.iter().any(|m| m.contains("決定点の中で均衡していません")),
            "固定順は検出されるべき: {problems:?}"
        );
    }

    /// **片方だけが強制列を共有した seed 対は主推定へ入れない**
    /// （PR #37 レビュー9巡目 [P1]）。±1 を許すと「先 1 / 後 0」が通り、
    /// ペア差が非ゼロになりうる唯一の unit が treatment 先だけになるので、
    /// 加法的な実行順効果が丸ごと主差へ残る
    fn 片側だけ畳まれた行() -> Vec<serde_json::Value> {
        let mut v = vec![];
        // seed 0: 3グループ（前向き）= alpha@k2 と current は別の継続
        for arm in ["baseline", "current", "alpha@k2"] {
            v.push(row_ord(arm, 0, if arm == "alpha@k2" { 1.0 } else { 0.0 }, arm_slot(arm)));
        }
        // seed 1: alpha@k2 が current と同じ強制列に畳まれた（＝ペア差 0）
        for arm in ["baseline", "current", "alpha@k2"] {
            let slot = if arm == "baseline" { 0 } else { 1 };
            let mut r = row_ord(arm, 1, 0.0, 1 - slot);
            r["continuation_group"] = serde_json::json!(grp(slot));
            v.push(r);
        }
        v
    }

    #[test]
    fn 片側だけ畳まれたseed対は入力ごと落とす() {
        // 9巡目は**事後に除外**して閉じたが、除外は arm の選択結果で母集団を
        // 条件づける post-treatment な操作で、全部落ちれば門を一度も評価せずに
        // 成功しうる（レビュー10巡目 [P1]）。生成側で対照と treatment を
        // 畳まなくしたのでこの形はもう出ないはずで、出たら**落とす**
        let rows = 片側だけ畳まれた行();
        let problems = check_inputs(&[meta(exp(), 0)], &rows, "current");
        assert!(
            problems.iter().any(|m| m.contains("均衡していません")),
            "閉じない入力は落とすべき（黙って除外しない）: {problems:?}"
        );
    }

    #[test]
    fn 継続グループが16桁hexでなければ止まる() {
        // 型を変えるだけで `unit_index` から全行が消え、均衡検査も主判定も
        // 空集合になっていた（レビュー10巡目 [P1]）
        for bad in [serde_json::json!(12345), serde_json::json!("zz"), serde_json::json!(null)] {
            let rows: Vec<serde_json::Value> = full()
                .into_iter()
                .map(|mut r| {
                    r["continuation_group"] = bad.clone();
                    r
                })
                .collect();
            let problems = check_inputs(&[meta(exp(), 0)], &rows, "current");
            assert!(
                problems.iter().any(|m| {
                    m.contains("continuation_group") || m.contains("必須列")
                }),
                "{bad} は拒否されるべき: {problems:?}"
            );
        }
    }

    #[test]
    fn 同じ順位なのに継続の結果が違えば止まる() {
        // treatment と対照の `arm_order` を同値にすると、以前は「畳まれた」と
        // 無条件に扱われて均衡検査が空集合になった（レビュー9巡目 [P1]）
        let rows: Vec<serde_json::Value> = full()
            .into_iter()
            .map(|mut r| {
                if r["arm"] == "alpha@k2" || r["arm"] == "current" {
                    r["arm_order"] = serde_json::json!(1);
                }
                r
            })
            .collect();
        let problems = check_inputs(&[meta(exp(), 0)], &rows, "current");
        assert!(
            problems.iter().any(|m| m.contains("1対1でない")),
            "order とグループの食い違いは検出されるべき: {problems:?}"
        );
    }

    #[test]
    fn 順位が整数でなければ止まる() {
        let rows: Vec<serde_json::Value> = full()
            .into_iter()
            .map(|mut r| {
                r["arm_order"] = serde_json::json!(r["arm_order"].to_string());
                r
            })
            .collect();
        let problems = check_inputs(&[meta(exp(), 0)], &rows, "current");
        assert!(
            problems.iter().any(|m| m.contains("整数でない")),
            "型を変えて検査を素通りできてはいけない: {problems:?}"
        );
    }

    #[test]
    fn 畳まれた組の継続結果が違えば止まる() {
        // 同じ `continuation_group` なら同じ継続1本を配ったはずなので、
        // score が違うのは配線が壊れている（または細工されている）印
        let mut rows = full();
        for r in &mut rows {
            if r["arm"] == "alpha@k2" {
                // current と同じグループ・同じ順位にするが score は違う
                r["continuation_group"] = serde_json::json!(grp(1));
                r["arm_order"] = serde_json::json!(1);
            }
        }
        let problems = check_inputs(&[meta(exp(), 0)], &rows, "current");
        assert!(
            problems.iter().any(|m| m.contains("継続の結果が違う")),
            "畳まれた組の結果不一致は検出されるべき: {problems:?}"
        );
    }

    #[test]
    fn 読めなかった元対局があれば止まる() {
        // 縮んだ `games` がそのまま Δ の分母になる（レビュー13巡目 [P1]）
        let mut e = exp();
        e["broken"] = serde_json::json!(3);
        let problems = check_inputs(&[meta(e, 0)], &full(), "current");
        assert!(
            problems.iter().any(|m| m.contains("読めなかった元対局")),
            "壊れた元対局は拒否されるべき: {problems:?}"
        );
        // 列そのものが無い記録も落とす（「不明」で通さない）
        let mut e2 = exp();
        e2.as_object_mut().unwrap().remove("broken");
        let problems = check_inputs(&[meta(e2, 0)], &full(), "current");
        assert!(
            problems.iter().any(|m| m.contains("broken がありません")),
            "欠測は拒否されるべき: {problems:?}"
        );
    }

    #[test]
    fn 門の判定は不合格を不合格として返す() {
        // 表示するだけでは workflow が緑のまま終わる（レビュー11巡目 [P1]）
        assert!(gate_foul(0.05, 0.01, 0.0, 0.0).pass, "門を超える");
        assert!(!gate_foul(-1.0, -1.0, 0.0, 0.0).pass, "改善量が届かない");
        assert!(!gate_foul(0.05, -0.01, 0.0, 0.0).pass, "CI 下限が 0 を跨ぐ");
        assert!(!gate_foul(0.05, 0.01, 0.01, 0.0).pass, "反則負けが悪化");
        assert!(!gate_foul(0.05, 0.01, 0.0, 0.01).pass, "破滅率が悪化");
        // 反則0 は非劣性＋即時反則の非増加（full β だけ増加を許す）
        assert!(gate_nofoul(0.0, -0.01, 0.0, 0.0, 0.0, false).pass);
        assert!(!gate_nofoul(-0.05, -0.09, 0.0, 0.0, 0.0, false).pass, "非劣性を満たさない");
        assert!(!gate_nofoul(0.0, -0.01, 0.1, 0.0, 0.0, false).pass, "即時反則が増えた");
        assert!(gate_nofoul(0.0, -0.01, 0.1, 0.0, 0.0, true).pass, "full β は許容");
    }

    #[test]
    fn スケジュールの対照と集計の対照が違えば止まる() {
        // AB/BA が閉じているのは `current` とのペアなので、`baseline` を対照に
        // すると主差には実行順効果が残る
        let problems = check_inputs(&[meta(exp(), 0)], &full(), "baseline");
        assert!(
            problems.iter().any(|m| m.contains("スケジュールの対照")),
            "対照の食い違いは検出されるべき: {problems:?}"
        );
    }

    /// **巡回では AB/BA にならない**（レビュー8巡目 [P1] の反例そのもの）:
    /// 3グループ・4 seed の shift は `0,1,2,0` で、固定ペアの前後は釣り合わない。
    /// 反転なら全ペアが同時に入れ替わる
    #[test]
    fn 反転は全ペアの前後を入れ替えるが巡回は入れ替えない() {
        let rank: BTreeMap<String, usize> = ["baseline", "current", "alpha@k2"]
            .iter()
            .enumerate()
            .map(|(i, a)| (a.to_string(), i))
            .collect();
        let groups: Vec<(GroupKey, Vec<String>)> = vec![
            ((vec!["m3".into()], None), vec!["alpha@k2".into()]),
            ((vec!["m1".into()], None), vec!["baseline".into()]),
            ((vec!["m2".into()], None), vec!["current".into()]),
        ];
        // 前向きの並びは**強制列の辞書順ではなく arm の優先順位**で決まる
        let fwd = schedule_groups(groups.clone(), &rank, 0, 0);
        let arms = |g: &[(GroupKey, Vec<String>)]| -> Vec<String> {
            g.iter().map(|(_, a)| a[0].clone()).collect()
        };
        assert_eq!(arms(&fwd), ["baseline", "current", "alpha@k2"]);
        // seed の偶奇で反転する
        let rev = schedule_groups(groups.clone(), &rank, 0, 1);
        assert_eq!(arms(&rev), ["alpha@k2", "current", "baseline"]);
        // 4 seed で `alpha@k2` と `current` の前後がちょうど半々になる
        let mut first = 0;
        for seed in 0..4u64 {
            let g = arms(&schedule_groups(groups.clone(), &rank, 0, seed));
            let ia = g.iter().position(|a| a == "alpha@k2").unwrap();
            let ic = g.iter().position(|a| a == "current").unwrap();
            if ia < ic {
                first += 1;
            }
        }
        assert_eq!(first, 2, "反転なら 4 seed で 2/2 になる");
        // 旧実装（巡回）は同じ条件で 1/3 にしかならない
        let mut cyc_first = 0;
        for seed in 0..4usize {
            let mut g: Vec<String> = vec!["baseline".into(), "current".into(), "alpha@k2".into()];
            g.rotate_left(seed % 3);
            let ia = g.iter().position(|a| a == "alpha@k2").unwrap();
            let ic = g.iter().position(|a| a == "current").unwrap();
            if ia < ic {
                cyc_first += 1;
            }
        }
        assert_eq!(cyc_first, 1, "巡回は 4 seed で 1/3 に偏る");
    }

    #[test]
    fn armが欠けた決定点を検出する() {
        let rows: Vec<serde_json::Value> = full()
            .into_iter()
            .filter(|r| !(r["arm"] == "alpha@k2" && r["seed"] == 1))
            .collect();
        let problems = check_inputs(&[meta(exp(), 0)], &rows, "current");
        assert!(problems.iter().any(|p| p.contains("alpha@k2")), "{problems:?}");
    }

    /// **欠測と余分を相殺させない**（PR #33 レビュー4巡目 [P1]）。
    /// replicate 0 から1行落とし、replicate 1 へ**範囲外の seed** の行を1本足すと、
    /// 「(game, move_number, arm) の行数を全 replicate 合算で seeds × replicate 数と
    /// 比べる」形では総数が合って素通りする（片方は欠測・片方は不正なのに
    /// 「2 replicate の平均」まで出る）。replicate ごと・seed ごとの集合で比べる
    #[test]
    fn 欠測を範囲外seedの余分行で埋め合わせられない() {
        let mk = |rep: u64| -> Vec<serde_json::Value> {
            full()
                .into_iter()
                .map(|mut r| {
                    r["replicate"] = serde_json::json!(rep);
                    r
                })
                .collect()
        };
        // replicate 0 は (seed 0, alpha@k2) が欠測
        let mut rows: Vec<serde_json::Value> = mk(0)
            .into_iter()
            .filter(|r| !(r["arm"] == "alpha@k2" && r["seed"] == 0))
            .collect();
        rows.extend(mk(1));
        // replicate 1 に範囲外 seed の行を1本（合算の行数だけは辻褄が合う）
        let mut extra = row("alpha@k2", 99, 1.0);
        extra["replicate"] = serde_json::json!(1);
        rows.push(extra);
        let mut m0 = meta(exp(), 0);
        m0["replicate"] = serde_json::json!(0);
        let mut m1 = meta(exp(), 0);
        m1["replicate"] = serde_json::json!(1);
        let problems = check_inputs(&[m0, m1], &rows, "current");
        assert!(
            problems.iter().any(|p| p.contains("欠けています")),
            "replicate 0 の欠測を検出できていない: {problems:?}"
        );
        assert!(
            problems.iter().any(|p| p.contains("宣言していない")),
            "範囲外 seed の余分行を検出できていない: {problems:?}"
        );
    }

    /// **別々の標本を同じ実験の replicate として平均できない**（PR #33 レビュー5巡目 [P1]）。
    #[test]
    fn 終端手番はbaselineの行を期待しない() {
        // 終端手番（受理手なし）は `baseline` を組めない。期待キーに入れると
        // 「欠測」で落ちてしまうし、逆に母集団から外すと最悪ケースが消える
        // （PR #37 レビュー4巡目 [P1]）
        let mut m = meta(exp(), 0);
        m["points_detail"] = serde_json::json!([
            {"game": "g1", "move_number": 41, "estimand": "foul", "terminal": true}
        ]);
        let mut rows = vec![];
        for seed in 0..2 {
            for arm in ["current", "alpha@k2"] {
                let mut r = row(arm, seed, 0.5);
                r["estimand"] = serde_json::json!("foul");
                rows.push(r);
            }
        }
        let problems = check_inputs(&[m], &rows, "current");
        assert!(
            !problems.iter().any(|p| p.contains("baseline")),
            "baseline の欠測を咎めない: {problems:?}"
        );
    }

    /// 実験キーの一致検査は `experiment` しか見ず `points_detail` はその外にあるので、
    /// replicate 1 の meta と全行の `game` に接頭辞を付けて母集団を完全に分離しても、
    /// **各 replicate 内の行は完全なまま**なので素通りしていた
    #[test]
    fn replicate間で決定点の母集団が違ったら止まる() {
        let mut m0 = meta(exp(), 0);
        m0["replicate"] = serde_json::json!(0);
        let mut m1 = meta(exp(), 0);
        m1["replicate"] = serde_json::json!(1);
        m1["points_detail"] =
            serde_json::json!([{"game": "other-g1", "move_number": 41, "estimand": "foul",
                               "terminal": false}]);
        let mut rows = full();
        for mut r in full() {
            r["replicate"] = serde_json::json!(1);
            r["game"] = serde_json::json!("other-g1");
            rows.push(r);
        }
        let problems = check_inputs(&[m0, m1], &rows, "current");
        assert!(
            problems.iter().any(|p| p.contains("決定点母集団")),
            "別の標本を 2 replicate として受理した: {problems:?}"
        );
    }

    /// **meta が宣言する estimand も列挙として検査する**（PR #33 レビュー7巡目 [P1]）。
    /// meta と行が**同じ未知の値**で揃っているとキーの厳密一致は通るが、集計は
    /// `ESTIMANDS` の2つしか走査しないので、その決定点が層から無言で消える
    #[test]
    fn metaの未知のestimandを拒否する() {
        let mut m = meta(exp(), 0);
        m["points_detail"] =
            serde_json::json!([{"game": "g1", "move_number": 41, "estimand": "unknown",
                               "terminal": false}]);
        let rows: Vec<serde_json::Value> = full()
            .into_iter()
            .map(|mut r| {
                r["estimand"] = serde_json::json!("unknown");
                r
            })
            .collect();
        // meta と行が揃っているのでキーの一致検査は通る = ここで止めるしかない
        let problems = check_inputs(&[m], &rows, "current");
        assert!(
            problems.iter().any(|p| p.contains("未知の値")),
            "meta ごと未知の estimand にした決定点が素通りした: {problems:?}"
        );
    }

    /// **行の `estimand` は meta と照合する**（PR #33 レビュー6巡目 [P1]）。
    /// `report` は**行側の** `estimand` で有効性側（foul）と非劣性側（nofoul）へ
    /// 分けるので、meta を変えずに行の estimand だけ入れ替えると、片方の replicate が
    /// 別の関門へ移って結論が静かに変わる。未知の estimand も同じ経路で弾く
    #[test]
    fn 行のestimandがmetaと違ったら止まる() {
        for swapped in ["nofoul", "unknown"] {
            let rows: Vec<serde_json::Value> = full()
                .into_iter()
                .map(|mut r| {
                    r["estimand"] = serde_json::json!(swapped);
                    r
                })
                .collect();
            let problems = check_inputs(&[meta(exp(), 0)], &rows, "current");
            assert!(
                problems.iter().any(|p| p.contains("宣言していない")),
                "estimand {swapped} への入れ替えが素通りした: {problems:?}"
            );
            assert!(
                problems.iter().any(|p| p.contains("欠けています")),
                "meta が宣言した estimand の行が欠けたことを検出できていない: {problems:?}"
            );
        }
    }

    /// 未宣言の決定点・meta に無い replicate ラベルの行も拒否する
    #[test]
    fn 未宣言の決定点とreplicateの行を拒否する() {
        for (label, mut bad) in [
            ("決定点", row("alpha@k2", 0, 1.0)),
            ("replicate", row("alpha@k2", 0, 1.0)),
        ] {
            if label == "決定点" {
                bad["move_number"] = serde_json::json!(99);
            } else {
                bad["replicate"] = serde_json::json!(7);
            }
            let mut rows = full();
            rows.push(bad);
            let problems = check_inputs(&[meta(exp(), 0)], &rows, "current");
            assert!(
                problems.iter().any(|p| p.contains("宣言していない")),
                "未宣言の{label}が素通りした: {problems:?}"
            );
        }
    }

    #[test]
    fn シャードが欠けたら止まる() {
        let mut e = exp();
        e["shard_total"] = serde_json::json!(2);
        let problems = check_inputs(&[meta(e, 0)], &full(), "current");
        assert!(
            problems.iter().any(|p| p.contains("シャードが欠けて")),
            "{problems:?}"
        );
    }

    /// **Δ の分母は全対局数**（対象の手番が無かった局は寄与 0）。
    /// 分母を「対象があった局」に取り替えると Δ が水増しされる
    #[test]
    fn deltaの分母は全対局数() {
        let rows = full();
        let refs: Vec<&serde_json::Value> = rows.iter().collect();
        let c = contributions(&refs, &|r| r["score"].as_f64().unwrap_or(0.0));
        assert_eq!(c.len(), 1, "1局ぶんの寄与");
        // 1局だけスコア 1.0 → 全104局で割るので Δ = 1/104
        let (d, lo, hi) = delta_ci(&c, "alpha@k2", None, 104);
        assert!((d - 1.0 / 104.0).abs() < 1e-9, "{d}");
        assert!(lo <= d && d <= hi);
        // ペア差（current は 0 なので同じ値）
        let (dd, _, _) = delta_ci(&c, "alpha@k2", Some("current"), 104);
        assert!((dd - 1.0 / 104.0).abs() < 1e-9, "{dd}");
        // 分母を1局にすると 1.0 になる = 分母の取り違えは Δ を100倍にする
        let (bad, _, _) = delta_ci(&c, "alpha@k2", None, 1);
        assert!((bad - 1.0).abs() < 1e-9);
    }

    #[test]
    fn 継続の乱数はarmに依らない() {
        assert_eq!(
            continuation_seeds("g1", 41, 0),
            continuation_seeds("g1", 41, 0)
        );
        assert_ne!(
            continuation_seeds("g1", 41, 0),
            continuation_seeds("g1", 41, 1)
        );
        assert_ne!(
            continuation_seeds("g1", 41, 0),
            continuation_seeds("g1", 43, 0)
        );
    }
}
