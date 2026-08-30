//! **王手中の反則経済 P0-5: オフライン方策シミュレーション**
//! （issue #31。runtime には何も入らない）。
//!
//! アリーナ記録の**王手中の手番**を復元し、候補ごとの gain / p_legal と
//! **ランキングを作ったのと同じ粒子**（`ParticleSnapshot`）を取り、方策を
//! 反則するたび更新しながら**真実で裁定**する。
//!
//! estimand は2つ（issue #31 の P0-6 の門と同じ割り方）:
//!
//! - `foul` … 実戦で反則した王手手番（改善側）
//! - `nofoul` … 反則0で受理された王手手番から**同数**を決定論的に抽出した対照。
//!   新しいプローブを足す害（即時反則が増えていないか）はここで見る
//!
//! 方策（すべて王手中限定。arm 名は `check_policy::Policy::tag`）:
//!
//! | arm | 内容 |
//! | --- | --- |
//! | `current` | 現行 `combine_score`（更新規則は下記の3本を別 arm で持つ） |
//! | `alpha@k*` | 反則価格 ×k（`c = max(k × base, 床)`。床は倍率化しない） |
//! | `beta@k*l*` | 動的継続価値（`c_eff = max(0.5c, c − λ·ΔV)`、残り1回では無効） |
//! | `beta_order@l*` | 現行がプローブを選ぶ決定に限りプローブ同士だけ並べ替え |
//! | `solver_greedy` | 解消確率の argmax（参考） |
//!
//! **γ（手種別の較正マップ）は測らない**: P0-2 の方向つき較正誤差が
//! v14 +0.025 [−0.008, +0.057] / v13 −0.009 [−0.055, +0.036] と門
//! （CI 下限 > 0.1・両相手で同方向）に届かず**符号すら逆**だったので、
//! H3 は棄却済み（`docs/check-economy-p0.md`）。落ちた枝に arm を足すと
//! 「有限個の arm の最大値」を無駄に広げるだけになる。
//!
//! 反則後の更新規則は issue の「4本の対照」に対応する:
//!
//! 1. 記録された実反則列 … 出力の `record_fouls`（再現の基準）
//! 2. `current@static` … 初回ランキングの次点を取る静的方策
//!    （= 現行 `combine_score` が暗黙に置く「次善手固定」の仮定そのもの）
//! 3. `current@real` … **実再決定**（`MyFoul` を注入して `Estimator::update` と
//!    ランキングを走らせる。正解基準・思考予算をまるごと使う）
//! 4. それ以外の arm … **p-only shadow update**（snapshot ベースの仮想更新）
//!
//! `current@shadow` と `current@real` の差が**近似通過条件**の材料。
//!
//! usage:
//!   TSUITATE_THINK_BUDGET_MS=2000 cargo run --release --bin check_policy -- \
//!     [--seeds 4] [--jobs N] [--limit N] [--shard i/n] [--opponent estimator_v14] \
//!     [--alpha-k 1.5,2,3] [--beta-lambda 0.5,1] [--beta-k 1] [--no-real] \
//!     [--out data/check_policy.jsonl] <records...>
//!   cargo run --release --bin check_policy -- report [--shards N] <out-*.jsonl...>
//!
//! **思考予算は記録を取ったアリーナと揃える**（違うと粒子数が変わり、
//! 「その決定点で実際に見えていたランキング」でなくなる）。

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tsuitate_bot::check::CheckSolver;
use tsuitate_bot::check_belief::{self, ArmSpec, Attrition, Belief, decision_points};
use tsuitate_bot::check_economy::{
    CheckMoveKind, classify_move_kind, cluster_ratio_ci, true_checkers,
};
use tsuitate_bot::check_policy::{
    CalibrationSums, EntrySetup, Policy, SimOutcome, UpdateRule,
    entry_setup as check_policy_entry, fmt_num, simulate, truth_after,
};
use tsuitate_bot::protocol::Color;
use tsuitate_bot::scenario_core::{Replayed, make_view, side_idx};
use tsuitate_bot::shogi::{Position, parse_usi};
use tsuitate_bot::strategy::{self, EvalParams};
use tsuitate_bot::truth_replay::parse_bot_and_end;

/// JSONL の契約バージョン。**古い schema は集計から弾く**（issue #28 / #31 の契約）。
///
/// - 1 … issue #31 の P0-5（オラクル arm と恒等対照の列が無い）
/// - 2 … issue #36 P0-2 の初版。**介入対象と被覆を「粒子なしのソルバー」で
///   決めていた**（実評価の集合と別物になりうる）ので集計から弾く
/// - 3 … PR #37 レビュー1巡目 [P1] を反映。倍率も被覆も**その arm が実際に列挙した
///   集合**に対して適用・記録する。ただし診断が **arm ごとでなく最後の1本で全行を
///   上書き**していた（`--belief-real` に複数指定すると別 arm の被覆を主 arm の
///   ものとして表示する）ので集計から弾く
/// - 4 … レビュー2〜3巡目。`deduce` / `oracle` は**その arm の行にだけ**入り、
///   `current@real` も他の実再決定 arm と同じ再ランキング経路を通る
///   （初期粒子サンプルを揃える）。`identity_err_real` はその実再決定側の恒等対照。
///   **欠けた列を「問題なし」と読むと恒等対照も演繹の健全性も素通りする**ので、
///   版の拒否と必須列の存在検査の両方で止める
/// - 5 … レビュー4〜5巡目。**母集団と重みの意味が変わった**（終端手番を含む全王手中
///   手番・自然頻度へ戻す `weight`）うえ、実再決定 arm を **AB/BA で背中合わせに**
///   走らせて `arm_order` を残すようになった。schema 4 の記録は
///   ①終端手番が欠けている ②`weight` が無いので「間引いた均衡標本」を
///   自然頻度として読んでしまう ③実行順が固定でしかも監査できない、の3点で
///   新しい gate へ通してはいけない（レビュー5巡目 [P1]: 実際に 102cc1a の
///   schema 4 JSONL が最新バイナリの `report` を exit 0 で通っていた）
/// schema 6 では **`double_check` を決定点の属性として meta に固定**した。
/// これは主 estimand の**分母**（両王手は介入が no-op なので除く）を決める列で、
/// schema 5 では行にしか無く meta と照合していなかったため、全 arm の行で
/// `false` へ書き換えると除外されていた手番が分母へ戻って主 CI が動いた
/// （PR #37 レビュー6巡目 [P1]: 実際に v13 の 6 シャードで再現した）。
const ROW_SCHEMA: u32 = 6;

/// schema 6 の行に必ずある列（欠測を既定値で埋めて門を通せてはいけない）。
///
/// `terminal` / `weight` / `estimand` は**母集団と分母**を決める列、
/// `arm_order` は**実行順の均衡**を検査する列なので、欠測を既定値で
/// 埋めると gate の意味が変わる
const REQUIRED_ROW_KEYS: [&str; 12] = [
    "identity_err",
    "identity_err_real",
    "identity_only_real",
    "deduce",
    "oracle",
    "double_check",
    "repro_err",
    "arm",
    "terminal",
    "weight",
    "estimand",
    "arm_order",
];

/// `--no-real` と `--belief-real` の解決（**起動時**の契約）。
///
/// `--no-real` は「実再決定 arm を1本も回さない」の意味なので、`beliefs_real` も
/// 空にする。`with_real` は `current@real` と恒等対照を自動で足すかしか見ていない
/// ので、これをやらないと既定の `beliefs_real`（`oracle@kinf`）が**対照も AB/BA も
/// 無いまま**走り、しかも偶数 seed の検査を素通りする（PR #37 レビュー7巡目 [P2]）。
/// 明示的な `--belief-real` との併用は矛盾なので `Err`。
///
/// 返り値は「解決後に実再決定 arm が1本でもあるか」＝偶数 seed を要求するか。
fn resolve_real_arms(
    with_real: bool,
    beliefs_real: &mut Vec<Belief>,
    explicit: bool,
) -> Result<bool, &'static str> {
    if !with_real {
        if explicit && !beliefs_real.is_empty() {
            return Err(
                "--no-real と --belief-real は同時に指定できません（実再決定を止めるのか回すのかが決まらない）",
            );
        }
        beliefs_real.clear();
    }
    Ok(with_real || !beliefs_real.is_empty())
}

fn die(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(2);
}

/// 監査対象の決定点（王手中の bot の手番1つ）
struct Point {
    game: String,
    move_number: u32,
    /// `foul`（実戦で反則した手番）/ `nofoul`（反則0の対照）
    estimand: &'static str,
    /// 型（P0-1 の粗い束）。反則0の手番は `no_foul`
    type_tag: String,
    /// 実戦の反則列（順番どおり）
    record_fouls: Vec<String>,
    /// 実戦で受理された手（復元の健全性検査用。方策には渡さない）
    record_accepted: String,
    /// 手番開始時（反則0）の状態
    entry: Replayed,
    /// 決定点の真実局面（裁定用。方策には渡さない）
    truth: Position,
    bot: Color,
    /// **終端手番**（反則だけ積んで受理手なしで終局した手番）。改善対象の最悪ケース
    terminal: bool,
    /// 自然頻度へ戻すための包含重み（間引かなければ 1.0）
    weight: f64,
    /// **両王手**（介入が no-op なので主 estimand の分母から除く）。
    /// 真実局面だけで決まるので決定点の属性として持ち、meta にも残す
    /// （行だけに書くと分母を決める列が meta と照合されない。
    /// PR #37 レビュー6巡目 [P1]）
    double_check: bool,
}

/// 1 arm の結果（決定点 × seed × arm）
struct ArmOut {
    arm: String,
    out: SimOutcome,
    truth: tsuitate_bot::check_policy::TruthAfter,
    /// `deduce_last_move` の健全性記録（その arm のものだけ。他は null）
    deduce: serde_json::Value,
    /// オラクル arm の被覆・fallback（**arm ごと**。他は null）
    oracle: serde_json::Value,
    /// この arm のシミュレーションに掛かった実時間（µs。P1 のコスト見積り）
    sim_us: u64,
    /// **この unit の中で何番目に走ったか**（実行順効果の監査用。
    /// PR #37 レビュー5巡目 [P1]）。実再決定 arm は AB/BA で反転するので、
    /// `report` がこの列で均衡を検査する
    arm_order: usize,
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

fn type_tag(first: CheckMoveKind, accepted: Option<CheckMoveKind>) -> String {
    use tsuitate_bot::check_economy::CoarseKind::*;
    let Some(acc) = accepted else {
        return "unfinished".to_string();
    };
    match (first.coarse(), acc.coarse()) {
        (Drop, _) => "drop",
        (NonKingBoard, King) => "nonking_king",
        (NonKingBoard, _) => "nonking_nonking",
        (King, King) => "king_king",
        (King, _) => "king_nonking",
    }
    .to_string()
}

/// belief から arm 名（`bin/check_continue` と共通の規約）を作る
fn spec_for(b: &Belief, real: bool) -> ArmSpec {
    let tag = format!("{}@{}", b.tag(), if real { "real" } else { "shadow" });
    ArmSpec::parse(&tag).expect("belief タグは ArmSpec でも読める")
}

/// `--belief` / `--belief-real` のタグ列（空文字は「なし」）
fn parse_beliefs(s: &str) -> Vec<Belief> {
    s.split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| Belief::parse(t).unwrap_or_else(|| die(&format!("未知の belief: {t}"))))
        .collect()
}

fn parse_list(s: &str, what: &str) -> Vec<f64> {
    let mut v: Vec<f64> = s
        .split(',')
        .filter(|t| !t.trim().is_empty())
        .map(|t| {
            t.trim()
                .parse()
                .unwrap_or_else(|_| die(&format!("{what} は数値のカンマ区切り")))
        })
        .collect();
    v.sort_by(f64::total_cmp);
    v.dedup();
    v
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
    let mut jobs: usize =
        std::thread::available_parallelism().map_or(1, |n| n.get().saturating_sub(2).max(1));
    let mut limit = usize::MAX;
    let mut shard = (0usize, 1usize);
    let mut opponent: Option<String> = None;
    let mut alpha_ks = vec![1.5, 2.0, 3.0];
    let mut beta_lambdas = vec![0.5, 1.0];
    let mut beta_k = 1.0f64;
    let mut with_real = true;
    // 反則0の手番を間引く上限（0 = 間引かない。主 estimand は自然頻度なので既定 0）
    let mut nofoul_cap: usize = 0;
    // **issue #36 P0-2 のオラクル arm**（既定で回す。issue #31 の arm はそのまま）。
    // `oracle@k1` は恒等対照（`current@shadow` と bit-exact でなければ配管が壊れている）
    let mut beliefs: Vec<Belief> = ["oracle@k1", "oracle@k2", "oracle@k4", "oracle@k8",
        "oracle@kinf", "oracle_misdirected@k4", "oracle_misdirected@kinf", "deduce_last_move"]
        .iter()
        .map(|t| Belief::parse(t).expect("既定の belief タグ"))
        .collect();
    // 実再決定まで回す belief（主 arm = `oracle@kinf@real` = issue の
    // `oracle_full_score@real`）。1本あたり思考予算をまるごと使うので既定は1つ
    let mut beliefs_real: Vec<Belief> = vec![Belief::parse("oracle@kinf").unwrap()];
    // `--belief-real` を**明示したか**（既定の `oracle@kinf` と区別する）。
    // `--no-real` との衝突を起動時に落とすのに要る
    let mut beliefs_real_explicit = false;
    let mut allow_incomplete = false;
    let mut out_path: Option<String> = None;
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
            "--jobs" => {
                jobs = need(args.get(i + 1), "--jobs")
                    .parse::<usize>()
                    .unwrap_or_else(|_| die("--jobs は整数"))
                    .max(1);
                i += 2;
            }
            "--limit" => {
                limit = need(args.get(i + 1), "--limit")
                    .parse()
                    .unwrap_or_else(|_| die("--limit は整数"));
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
            "--opponent" => {
                opponent = Some(need(args.get(i + 1), "--opponent"));
                i += 2;
            }
            "--alpha-k" => {
                alpha_ks = parse_list(&need(args.get(i + 1), "--alpha-k"), "--alpha-k");
                i += 2;
            }
            "--beta-lambda" => {
                beta_lambdas = parse_list(&need(args.get(i + 1), "--beta-lambda"), "--beta-lambda");
                i += 2;
            }
            "--beta-k" => {
                beta_k = need(args.get(i + 1), "--beta-k")
                    .parse()
                    .unwrap_or_else(|_| die("--beta-k は数値"));
                i += 2;
            }
            "--no-real" => {
                with_real = false;
                i += 1;
            }
            "--nofoul-cap" => {
                nofoul_cap = need(args.get(i + 1), "--nofoul-cap")
                    .parse()
                    .unwrap_or_else(|_| die("--nofoul-cap は整数（0 = 間引かない）"));
                i += 2;
            }
            "--belief" => {
                beliefs = parse_beliefs(&need(args.get(i + 1), "--belief"));
                i += 2;
            }
            "--belief-real" => {
                beliefs_real = parse_beliefs(&need(args.get(i + 1), "--belief-real"));
                beliefs_real_explicit = true;
                i += 2;
            }
            "--allow-incomplete" => {
                allow_incomplete = true;
                i += 1;
            }
            "--out" => {
                out_path = Some(need(args.get(i + 1), "--out"));
                i += 2;
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
    // **恒等対照は外せない**（PR #37 レビュー [P2]）。`report` 側でも検査するが、
    // 3時間走らせてから集計で落ちるより起動時に止めるほうがよい
    if (!beliefs.is_empty() || !beliefs_real.is_empty())
        && !beliefs.contains(&Belief::Oracle { k: 1.0 })
    {
        die(
            "--belief に恒等対照 oracle@k1 を入れてください\
             （オラクル arm が `current@shadow` と bit-exact であることを毎回確かめる契約）",
        );
    }
    // **`--seeds 0` で「対象なし」を成功終了にできてはいけない**
    // （issue #28 が `mate_continue` で塞いだ穴と同じ: 空の集計で判定を偽造できる）
    if seeds == 0 {
        die("--seeds は 1 以上にしてください（0 だと1本もシミュレーションせずに集計が空になる）");
    }
    let has_real = resolve_real_arms(with_real, &mut beliefs_real, beliefs_real_explicit)
        .unwrap_or_else(|m| die(m));
    // **実再決定 arm を回すなら seed は偶数**（PR #37 レビュー5巡目 [P1]）。
    // 実行順は `(決定点番号 + seed) % 2` で反転するので、奇数だと決定点ごとに
    // AB/BA が閉じず、実行順効果がペア差の平均に残る（#31 P0-7 と同じ理由）。
    // shadow だけの実験（`--no-real`）は決定論的なので偶数を要求しない。
    // **判定は解決後の real arm 集合で行う**（`with_real` だけを見ると、
    // `beliefs_real` 経由で入る real arm が検査を素通りする。レビュー7巡目 [P2]）
    if has_real && seeds % 2 != 0 {
        die("--seeds は 2 以上の偶数にしてください（実再決定 arm の AB/BA は seed の偶奇で閉じるので、奇数だと実行順効果がペア差に残る）");
    }
    let files = collect_records(&specs);
    if files.is_empty() {
        die("記録ファイルが見つかりません");
    }

    let cfg = tsuitate_bot::config::ambient();
    let params = EvalParams::default();
    let eval_particles = strategy::eval_particles_for_budget(cfg.think_budget_ms);

    // ---- 決定点の収集（粒子は回さない）-----------------------------------
    let mut fouled: Vec<Point> = vec![];
    let mut clean: Vec<Point> = vec![];
    let mut games = 0u32;
    let mut broken = 0u32;
    let mut mismatched = 0u32;
    let mut attrition = Attrition::default();
    let mut record_opponents: BTreeMap<String, u32> = BTreeMap::new();
    // **解析に渡したのと同じ bytes** から記録集合の指紋を作る（ディスクを読み直すと
    // TOCTOU。issue #28 PR #30 レビュー3巡目の教訓）
    use sha2::Digest as _;
    let mut digest = sha2::Sha256::new();
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
        if opponent.as_ref().is_some_and(|o| *o != end.opponent.username) {
            mismatched += 1;
            continue;
        }
        let short = Path::new(&name)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or(name.clone());
        // **母集団は共通の `check_belief::decision_points`**（PR #37 レビュー4巡目 [P1]）。
        // `for_each_decision_full` は受理手を単位に回すので**終端手番を返さない**:
        // 改善対象の最悪ケースであり即時反則負けの分子でもある手番が系統的に
        // 消えると、オラクル効果も safety 指標も楽観側へ偏る
        let Some(found_raw) = decision_points(&end, bot, &mut attrition) else {
            broken += 1;
            continue;
        };
        games += 1;
        for cp in found_raw {
            // 型は P0-1 と同じ規約（bot の意図 = `captures_checker` で分ける）
            let view = make_view(&cp.entry.pos, bot, &cp.entry.fouls);
            let log = &cp.entry.logs[side_idx(bot)];
            let mut solver = CheckSolver::new(&view, &[], &[], log);
            let acc_kind = cp
                .record_accepted
                .as_deref()
                .and_then(parse_usi)
                .map(|m| classify_move_kind(&m, &view, solver.as_mut()));
            let tag = match cp.record_fouls.first().and_then(|u| parse_usi(u)) {
                Some(first) => {
                    let k = classify_move_kind(&first, &view, solver.as_mut());
                    type_tag(k, acc_kind)
                }
                None => "no_foul".to_string(),
            };
            let estimand = cp.estimand();
            let double_check = true_checkers(&cp.truth, bot).len() > 1;
            let point = Point {
                game: short.clone(),
                move_number: cp.move_number,
                estimand,
                type_tag: tag,
                record_accepted: cp.record_accepted.clone().unwrap_or_default(),
                record_fouls: cp.record_fouls,
                truth: cp.truth,
                entry: cp.entry,
                bot,
                terminal: cp.terminal,
                weight: 1.0,
                double_check,
            };
            if estimand == "foul" { fouled.push(point) } else { clean.push(point) }
        }
    }
    // **既定は間引かない**（PR #37 レビュー4巡目 [P1]）。issue #36 の主 estimand は
    // 「全王手手番の自然頻度」なので、反則0の手番を foul と同数まで落とすと
    // 少数の foul 層を過大に重み付けしたまま合算することになる。
    // `--nofoul-cap N` で間引くときは**包含重み**（元の層頻度 ÷ 残した本数）を
    // 各点に残し、自然頻度の表はその重みで戻す
    fouled.sort_by(|a, b| (&a.game, a.move_number).cmp(&(&b.game, b.move_number)));
    clean.sort_by(|a, b| (&a.game, a.move_number).cmp(&(&b.game, b.move_number)));
    let clean_total = clean.len();
    let want = if nofoul_cap == 0 { clean_total } else { nofoul_cap.min(clean_total) };
    if clean_total > want && want > 0 {
        let step = clean_total as f64 / want as f64;
        let keep: BTreeSet<usize> = (0..want)
            .map(|k| ((k as f64 * step) as usize).min(clean_total - 1))
            .collect();
        clean = clean
            .into_iter()
            .enumerate()
            .filter(|(i, _)| keep.contains(i))
            .map(|(_, p)| p)
            .collect();
        let w = clean_total as f64 / clean.len().max(1) as f64;
        for p in &mut clean {
            p.weight = w;
        }
    }
    let mut points: Vec<Point> = fouled;
    points.extend(clean);
    points.sort_by(|a, b| (&a.game, a.move_number).cmp(&(&b.game, b.move_number)));
    // シャードは**局単位**（同じ局の決定点がシャードをまたがない）
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
    if points.len() > limit {
        points.truncate(limit);
    }
    if points.is_empty() {
        die("対象の決定点がありません（記録と --opponent を確認）");
    }

    // ---- arm の定義 --------------------------------------------------------
    let mut policies: Vec<(String, Policy)> = vec![("current".into(), Policy::Current)];
    for k in &alpha_ks {
        policies.push((Policy::Alpha { k: *k }.tag(), Policy::Alpha { k: *k }));
    }
    for l in &beta_lambdas {
        let b = Policy::Beta { k: beta_k, lambda: *l };
        policies.push((b.tag(), b));
        let o = Policy::BetaOrder { lambda: *l };
        policies.push((o.tag(), o));
    }
    policies.push(("solver_greedy".into(), Policy::SolverGreedy));

    println!(
        "記録 {} 件（壊れ {broken} / 相手不一致 {mismatched}）/ 局 {games}",
        files.len()
    );
    println!(
        "記録の相手: {}",
        record_opponents
            .iter()
            .map(|(k, n)| format!("{k} {n}局"))
            .collect::<Vec<_>>()
            .join(" / ")
    );
    println!(
        "王手中の bot の手番 {}（うち終端 {} / 復元できず {}）",
        attrition.turns, attrition.terminal, attrition.unreplayable
    );
    println!(
        "決定点 {}（反則あり {} / 反則0 {}）/ seeds {seeds} / jobs {jobs} / shard {}/{}",
        points.len(),
        points.iter().filter(|p| p.estimand == "foul").count(),
        points.iter().filter(|p| p.estimand == "nofoul").count(),
        shard.0,
        shard.1,
    );
    println!(
        "arm: {} + current@static{} / 思考予算 {}ms / eval_particles {eval_particles} / config {}",
        policies
            .iter()
            .map(|(t, _)| t.as_str())
            .collect::<Vec<_>>()
            .join(","),
        if with_real { " + current@real" } else { "（--no-real）" },
        cfg.think_budget_ms,
        cfg.fingerprint(),
    );
    println!(
        "belief（issue #36 P0-2）: {} / 実再決定まで回す: {}",
        if beliefs.is_empty() {
            "なし".to_string()
        } else {
            beliefs.iter().map(|b| b.tag()).collect::<Vec<_>>().join(",")
        },
        if beliefs_real.is_empty() {
            "なし".to_string()
        } else {
            beliefs_real.iter().map(|b| b.tag()).collect::<Vec<_>>().join(",")
        },
    );
    println!("source_fingerprint {}", env!("TSUITATE_SOURCE_FINGERPRINT"));

    // ---- 実行（決定点 × seed）---------------------------------------------
    let units: Vec<(usize, u64)> = (0..points.len())
        .flat_map(|p| (0..seeds).map(move |s| (p, s)))
        .collect();
    let effective_jobs = jobs.min(units.len()).max(1);
    let next = Arc::new(Mutex::new(0usize));
    let lines: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(vec![]));
    // ランキングが取れなかった決定点は**明示的に meta へ残す**（定跡手・候補ゼロ）。
    // 黙って行が欠けると `check_inputs` が「seed ごと欠けた」と区別できず、
    // 2時間走った集約が最後に失敗する。逆に記録しておけば、記録に無い欠落
    // （= 本当に落ちた行）は今までどおり検出できる
    let dropped: Arc<Mutex<BTreeSet<(String, u32)>>> = Arc::new(Mutex::new(BTreeSet::new()));
    let points = Arc::new(points);
    let policies = Arc::new(policies);
    let beliefs = Arc::new(beliefs);
    let beliefs_real = Arc::new(beliefs_real);
    let started = std::time::Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..effective_jobs {
            let next = Arc::clone(&next);
            let lines = Arc::clone(&lines);
            let dropped = Arc::clone(&dropped);
            let points = Arc::clone(&points);
            let policies = Arc::clone(&policies);
            let beliefs = Arc::clone(&beliefs);
            let beliefs_real = Arc::clone(&beliefs_real);
            let units = &units;
            let params = &params;
            scope.spawn(move || {
                loop {
                    let ui = {
                        let mut g = next.lock().unwrap();
                        if *g >= units.len() {
                            break;
                        }
                        let v = *g;
                        *g += 1;
                        v
                    };
                    let (pi, seed) = units[ui];
                    match run_unit(
                        &points[pi],
                        pi,
                        seed,
                        &policies,
                        &beliefs,
                        &beliefs_real,
                        params,
                        eval_particles,
                        with_real,
                    ) {
                        Some(rows) => lines.lock().unwrap().extend(rows),
                        None => {
                            dropped
                                .lock()
                                .unwrap()
                                .insert((points[pi].game.clone(), points[pi].move_number));
                        }
                    }
                }
            });
        }
    });
    let mut lines = Arc::try_unwrap(lines).ok().unwrap().into_inner().unwrap();
    lines.sort_by_key(|v| {
        (
            v["game"].as_str().unwrap_or("").to_string(),
            v["move_number"].as_u64().unwrap_or(0),
            v["seed"].as_u64().unwrap_or(0),
            v["arm"].as_str().unwrap_or("").to_string(),
        )
    });
    let dropped = Arc::try_unwrap(dropped).ok().unwrap().into_inner().unwrap();
    if !dropped.is_empty() {
        eprintln!(
            "ランキングが取れずに落とした決定点: {} 件（meta の dropped に残す）",
            dropped.len()
        );
    }
    eprintln!(
        "{} 行 / {:.1}分",
        lines.len(),
        started.elapsed().as_secs_f64() / 60.0
    );

    let records_fingerprint: String =
        digest.finalize().iter().map(|b| format!("{b:02x}")).collect();
    // **判定に効く実験キー**（全シャードで完全一致していることを `report` が要求する）
    let experiment = serde_json::json!({
        "opponent": opponent.clone().unwrap_or_default(),
        "budget_ms": cfg.think_budget_ms,
        "eval_particles": eval_particles,
        "seeds": seeds,
        "arms": policies.iter().map(|(t, _)| t.clone()).collect::<Vec<_>>(),
        "with_real": with_real,
        "beliefs": beliefs.iter().map(|b| b.tag()).collect::<Vec<_>>(),
        "beliefs_real": beliefs_real.iter().map(|b| b.tag()).collect::<Vec<_>>(),
        "nofoul_cap": nofoul_cap,
        "beta_k": fmt_num(beta_k),
        "jobs": effective_jobs,
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
        "games": games,
        "points": points.len(),
        // 母集団の attrition（終端手番が系統的に欠測すると門が甘くなる）
        "attrition": {
            "check_turns": attrition.turns,
            "terminal": attrition.terminal,
            "unreplayable": attrition.unreplayable,
        },
        // **期待する行の骨格を meta 自身に残す**（ある seed の全 arm がまとめて
        // 欠けても検出できるように。issue #28 PR #30 レビュー2巡目 [P1]）
        "points_detail": points
            .iter()
            .map(|p| serde_json::json!({
                "game": p.game,
                "move_number": p.move_number,
                "estimand": p.estimand,
                "type": p.type_tag,
                "terminal": p.terminal,
                "weight": p.weight,
                "double_check": p.double_check,
                "record_fouls": p.record_fouls.len(),
            }))
            .collect::<Vec<_>>(),
        // **ランキングが取れなかった決定点**（定跡手・候補ゼロ）。行が無いのが
        // 正常なので期待から外す。記録に無い欠落は今までどおり失敗させる
        "dropped": dropped
            .iter()
            .map(|(g, mn)| serde_json::json!({ "game": g, "move_number": mn }))
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
    if shard.1 > 1 {
        println!(
            "\n**シャード {}/{} の部分集計**（判定は `check_policy report` で全シャードを集めてから）",
            shard.0, shard.1
        );
    }
    report(&[meta], &lines, allow_incomplete);
}

/// 1 unit（決定点 × seed）を回して JSONL の行を返す
#[allow(clippy::too_many_arguments)]
fn run_unit(
    p: &Point,
    point_index: usize,
    seed: u64,
    policies: &[(String, Policy)],
    beliefs: &[Belief],
    beliefs_real: &[Belief],
    params: &EvalParams,
    eval_particles: usize,
    with_real: bool,
) -> Option<Vec<serde_json::Value>> {
    let side = p.entry.pos.turn();
    // **ランキングを作ったのと同じ粒子**で仮想更新する（issue #28 P0-3 の教訓）。
    // prewarm は1回だけで、実再決定はこの instance の clone に反則を食わせた継続
    let setup = check_policy_entry(&p.entry, &p.truth, seed, params, eval_particles)?;
    // `setup` は**丸ごと**持っておく（issue #36 の arm は `check_belief::run_arm`
    // が同じ instance から clone して実再決定するため）
    // `setup` を分解せず丸ごと持つ（実再決定 arm は `check_belief::run_arm` が
    // この instance から clone するので、ここでは読むだけ）
    let EntrySetup { moves, p0, updater, .. } = &setup;
    // **健全性**: 反則0での仮想更新は初回ランキングの p_legal を再現するはず。
    // ずれるなら `evaluate` の経路（min キャップや別のノブ）を取りこぼしている
    let p_repro = updater.p_after(&moves, &[]);
    let repro_err = p0
        .iter()
        .zip(&p_repro)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);

    let fouls_before = p.entry.fouls[side_idx(side)];
    let opp_fouls = p.entry.fouls[side_idx(side.other())];
    let mut cal = CalibrationSums::default();
    cal.add(&moves, &p0);

    let mut arms: Vec<ArmOut> = vec![];
    let mut run = |arm: String, policy: &Policy, rule: UpdateRule<'_>| {
        let t = std::time::Instant::now();
        let order = arms.len();
        let out = simulate(policy, &moves, &p0, params, fouls_before, opp_fouls, rule);
        let truth = truth_after(&p.truth, p.bot, out.accepted.as_deref());
        arms.push(ArmOut {
            arm,
            out,
            truth,
            deduce: serde_json::Value::Null,
            oracle: serde_json::Value::Null,
            sim_us: t.elapsed().as_micros() as u64,
            arm_order: order,
        });
    };
    // 2. 静的方策（現行 `combine_score` が暗黙に置く仮定そのもの）
    run("current@static".into(), &Policy::Current, UpdateRule::Static);
    // 4. p-only shadow update（全 arm 共通の更新規則）
    for (tag, policy) in policies {
        run(format!("{tag}@shadow"), policy, UpdateRule::Shadow(&updater));
    }
    // ---- issue #36 P0-2: 仮説重みへの介入 arm --------------------------------
    // 配管は `check_belief::run_arm` に一本化してある（`bin/check_continue` の
    // P0-2b と**同じ arm 名が同じ配管を指す**ようにするため）。
    // **介入対象・被覆・fallback はその arm が実際に列挙した集合から記録する**
    // （`check::DiagRecord`。外で作った倍率表を使うと、粒子投票の有無で列挙集合が
    // 変わったときに実評価側にしか無い誤仮説へ ×0 が付かない = PR #37 レビュー [P1]）
    let checkers = true_checkers(&p.truth, p.bot);
    let mut identity_err: Option<f64> = None;
    let mut belief_specs: Vec<(ArmSpec, bool)> =
        beliefs.iter().map(|b| (spec_for(b, false), *b == Belief::DeduceLastMove)).collect();
    // **実再決定 arm はすべて同じ `run_arm` 経路を通す**（PR #37 レビュー2巡目 [P1]）。
    // `entry_setup` は既に1回 `choose` しているので、その instance を clone して
    // もう一度 `choose` する arm と、`entry_setup` の初回 `moves/p0` をそのまま
    // 初手に使う arm では**別の粒子サンプル**を比べることになる。`current@real` も
    // 同じ再ランキング経路へ通せば、主差に残るのは介入だけになる
    if with_real {
        belief_specs.push((
            ArmSpec::parse("current@real").expect("既定 arm"),
            false,
        ));
    }
    belief_specs.extend(beliefs_real.iter().map(|b| (spec_for(b, true), false)));
    // 実再決定の**恒等対照**（`oracle@k1@real`）: shadow の k=1 は再ランキングを
    // 通らないので、この交絡を検出できない
    if beliefs_real.iter().any(|b| matches!(b, Belief::Oracle { .. })) && with_real {
        belief_specs.push((
            ArmSpec::parse("oracle@k1@real").expect("既定 arm"),
            false,
        ));
    }
    // **重複を正規化する**（PR #37 レビュー3巡目 [P2]）。`--belief-real oracle@k1` を
    // 明示すると自動追加と衝突し、同じ arm の行が2本出て**完走後**に
    // 「重複行があります」で落ちる。長時間実験の末尾で落とさない
    belief_specs.sort_by(|a, b| a.0.tag.cmp(&b.0.tag));
    belief_specs.dedup_by(|a, b| a.0.tag == b.0.tag);
    // **実再決定 arm は AB/BA で背中合わせに走らせる**（PR #37 レビュー5巡目 [P1]）。
    //
    // shadow arm は `ShadowUpdater` が初回ランキングから決定論的に計算するので
    // 実行順に依存しない（恒等対照が bit-exact なのがその証拠）。**実再決定だけが
    // 壁時計デッドラインまで粒子を回すので順序と負荷の影響を受ける**。タグ順に
    // 固定して並べると `current@real` は常に先・`oracle@kinf@real` は複数 arm を
    // 挟んだ後になり、主差に実行順効果が残る。
    //
    // そこで実再決定 arm だけを1つの連続ブロックにまとめ、**`current@real` を
    // その中央へ置いてから、`(決定点番号 + seed) % 2 == 1` のときブロックごと
    // 反転する**。反転は各 treatment の「対照から見た側」を入れ替えるので、
    // 偶数 seed の内側で主対 (`oracle@kinf@real` vs `current@real`) も
    // ノイズ床の対 (`oracle@k1@real` vs `current@real`) も AB/BA が閉じる
    // （#31 P0-7 の `check_price` と同じ設計。`--seeds` が偶数であることは
    // 起動時に要求する）。実行順は `arm_order` として全行に残し、`report` が
    // 均衡を検査する
    let (mut shadow_specs, mut real_specs): (Vec<_>, Vec<_>) =
        belief_specs.into_iter().partition(|(sp, _)| !sp.real);
    if !real_specs.is_empty() {
        if let Some(ci) = real_specs.iter().position(|(sp, _)| sp.tag == "current@real") {
            let cur = real_specs.remove(ci);
            real_specs.insert(real_specs.len() / 2, cur);
        }
        if (point_index + seed as usize) % 2 == 1 {
            real_specs.reverse();
        }
    }
    shadow_specs.append(&mut real_specs);
    let belief_specs = shadow_specs;
    let mut p_real_current: Option<Vec<(String, f64)>> = None;
    let mut p_real_identity: Option<Vec<(String, f64)>> = None;
    for (spec, is_deduce) in &belief_specs {
        let t = std::time::Instant::now();
        let Some(run) =
            check_belief::run_arm(&setup, &p.entry, &p.truth, p.bot, spec, params)
        else {
            continue;
        };
        // **その arm の全構築ぶん**の記録（1 arm = 1 scope）
        let rec = tsuitate_bot::check::take_diag_record();
        // **診断は arm ごとに持つ**（PR #37 レビュー2巡目 [P2]）。単一変数だと
        // `--belief-real` に複数指定したとき最後の1本が全行を上書きし、report が
        // 別 arm の被覆・fallback を主 arm のものとして表示する
        let mut deduce_note = serde_json::Value::Null;
        let mut oracle_note = serde_json::Value::Null;
        if *is_deduce {
            // **全滅 fallback の前**に「落とそうとした真仮説」を数える
            deduce_note = serde_json::json!({
                "dropped": rec.deduce_dropped.len(),
                "fallback": rec.deduce_fallback,
                "dropped_true": rec.deduce_dropped
                    .iter()
                    .any(|(ds, dr)| checkers.iter().any(|(s, r)| s == ds && r == dr)),
                "constructions": rec.constructions,
            });
        }
        // 実再決定のオラクル arm の被覆は**実評価の集合**で数える
        if spec.real && matches!(spec.belief, Belief::Oracle { .. }) {
            oracle_note = serde_json::json!({
                "arm": spec.tag,
                "constructions": rec.constructions,
                "truth_present": rec.truth_present,
                // 全構築で真仮説が列挙されていたか（1つでも欠ければ full oracle ではない）
                "covered_all": rec.constructions > 0 && rec.truth_present == rec.constructions,
                "weight_fallback": rec.weight_fallback,
            });
        }
        if spec.belief == (Belief::Oracle { k: 1.0 }) && !spec.real {
            // **恒等対照（shadow）**: `current@shadow` と bit-exact でなければ壊れている。
            // shadow は候補リストが `setup.moves` のままなので USI の並びも同じ
            let base: Vec<(String, f64)> = setup
                .moves
                .iter()
                .map(|m| m.usi.clone())
                .zip(p0.iter().copied())
                .collect();
            identity_err = Some(p_gap(&base, &run.p_entry).0);
        }
        // 実再決定側の恒等対照は `current@real` との差で見る（両方とも再ランキング）
        if spec.real && spec.belief == Belief::Current {
            p_real_current = Some(run.p_entry.clone());
        }
        if spec.real && spec.belief == (Belief::Oracle { k: 1.0 }) {
            p_real_identity = Some(run.p_entry.clone());
        }
        let truth = truth_after(&p.truth, p.bot, run.out.accepted.as_deref());
        let order = arms.len();
        arms.push(ArmOut {
            arm: spec.tag.clone(),
            out: run.out,
            truth,
            deduce: deduce_note,
            oracle: oracle_note,
            sim_us: t.elapsed().as_micros() as u64,
            arm_order: order,
        });
    }
    // **USI で突き合わせる**（PR #37 レビュー3巡目 [P2]）: 実再決定は候補リストごと
    // 作り直すので、添字で zip すると別の手の p を比べてしまう
    let (identity_err_real, identity_only_real) = match (&p_real_current, &p_real_identity) {
        (Some(a), Some(b)) => {
            let (d, only) = p_gap(a, b);
            (Some(d), Some(only))
        }
        _ => (None, None),
    };

    Some(
        arms.into_iter()
            .map(|a| {
                serde_json::json!({
                    "schema": ROW_SCHEMA,
                    "game": p.game,
                    "move_number": p.move_number,
                    "estimand": p.estimand,
                    "type": p.type_tag,
                    "terminal": p.terminal,
                    // 自然頻度へ戻す包含重み（間引かなければ 1.0）
                    "weight": p.weight,
                    "seed": seed,
                    "arm": a.arm,
                    "fouls": a.out.fouls,
                    "record_fouls": p.record_fouls.len(),
                    "foul_limit": a.out.foul_limit,
                    "accepted": a.out.accepted.clone().unwrap_or_default(),
                    "accepted_kind": a.out.accepted_kind.map(|k| k.tag()).unwrap_or(""),
                    // 復元の健全性（P0-4 と同じ読み方）: シミュレーションが
                    // 実戦の選択をどれだけ再現しているか。**判定の前にここを見る**
                    "record_accepted_matches":
                        a.out.accepted.as_deref() == Some(p.record_accepted.as_str()),
                    "record_first_matches": a.out.sequence.first().map(String::as_str)
                        == p.record_fouls.first().map(String::as_str)
                            .or(Some(p.record_accepted.as_str())),
                    "sequence": a.out.sequence,
                    "updates": a.out.updates,
                    "sim_us": a.sim_us,
                    "material": a.truth.material,
                    "mated_in_1": a.truth.mated_in_1,
                    "next_check": a.truth.next_check,
                    "truth_accepted": a.truth.accepted,
                    "candidates": moves.len(),
                    "repro_err": repro_err,
                    // issue #36 P0-2 の恒等対照（unit 単位）と、**arm ごと**の
                    // 演繹の健全性・オラクルの被覆（レビュー2巡目 [P2]）
                    "identity_err": identity_err,
                    "identity_err_real": identity_err_real,
                    // 片側にしか無い候補の本数（順位入れ替えでなく候補集合の差）
                    "identity_only_real": identity_only_real,
                    "deduce": a.deduce,
                    "oracle": a.oracle,
                    // 決定点の属性をそのまま出す（`check_inputs` が meta と照合する）
                    "double_check": p.double_check,
                    // **この unit の中での実行順**（実再決定 arm は AB/BA で反転）
                    "arm_order": a.arm_order,
                    "calibration": cal.to_json(),
                })
            })
            .collect(),
    )
}

// ---- 集計 ------------------------------------------------------------------

#[derive(Default, Clone)]
struct ArmStats {
    n: usize,
    fouls: Vec<f64>,
    foul_limit: usize,
    heavy: usize,
    accepted: usize,
    mated: usize,
    next_check: usize,
    material: Vec<f64>,
    sim_us: Vec<f64>,
    updates: Vec<f64>,
    /// 局ごとの (反則の和, 決定点数) — cluster bootstrap の統計単位
    by_game: BTreeMap<String, (f64, f64)>,
    /// 局ごとの (被一手詰めの数, 決定点数)
    mate_by_game: BTreeMap<String, (f64, f64)>,
}

impl ArmStats {
    fn push(&mut self, r: &serde_json::Value) {
        let game = r["game"].as_str().unwrap_or("?").to_string();
        let fouls = r["fouls"].as_f64().unwrap_or(0.0);
        self.n += 1;
        self.fouls.push(fouls);
        if r["foul_limit"].as_bool().unwrap_or(false) {
            self.foul_limit += 1;
        }
        if fouls >= 8.0 {
            self.heavy += 1;
        }
        if r["truth_accepted"].as_bool().unwrap_or(false) {
            self.accepted += 1;
            self.material.push(r["material"].as_f64().unwrap_or(0.0));
        }
        if r["mated_in_1"].as_bool().unwrap_or(false) {
            self.mated += 1;
        }
        if r["next_check"].as_bool().unwrap_or(false) {
            self.next_check += 1;
        }
        self.sim_us.push(r["sim_us"].as_f64().unwrap_or(0.0));
        self.updates.push(r["updates"].as_f64().unwrap_or(0.0));
        let e = self.by_game.entry(game.clone()).or_default();
        e.0 += fouls;
        e.1 += 1.0;
        let m = self.mate_by_game.entry(game).or_default();
        m.0 += f64::from(u8::from(r["mated_in_1"].as_bool().unwrap_or(false)));
        m.1 += 1.0;
    }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn pct(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[(((s.len() - 1) as f64) * q).round() as usize]
}

/// 局ごとの**ペア差**（arm − current）の cluster bootstrap CI。
///
/// 統計単位は元対局（seed も決定点も独立標本として数えない）。
fn paired_ci(
    rows: &[&serde_json::Value],
    arm: &str,
    base: &str,
    key: &dyn Fn(&serde_json::Value) -> f64,
) -> (f64, f64, f64) {
    paired_with(rows, arm, base, key, &|a, b| a - b)
}

/// **unit ごとに `|arm − base|` を作ってから**元対局 cluster で集約する。
///
/// 近似の良し悪しは**符号付き平均では測れない**（PR #33 レビュー [P1]）:
/// 「ある unit で +1 反則ずれ、別の unit で −1 反則ずれる」仮想更新は、
/// 符号付き平均だと誤差 0 に見えて門を通ってしまう。実測でも v13 は
/// 符号付き −0.015 に対し unit ごとの絶対誤差は 0.097 で、比率は
/// 0.33 → 2.15 と一桁違う。**門には必ずこちらを使う**。
fn paired_abs_ci(
    rows: &[&serde_json::Value],
    arm: &str,
    base: &str,
    key: &dyn Fn(&serde_json::Value) -> f64,
) -> (f64, f64, f64) {
    paired_with(rows, arm, base, key, &|a, b| (a - b).abs())
}

/// `combine(arm の値, base の値)` を unit ごとに作り、元対局 cluster で畳む
fn paired_with(
    rows: &[&serde_json::Value],
    arm: &str,
    base: &str,
    key: &dyn Fn(&serde_json::Value) -> f64,
    combine: &dyn Fn(f64, f64) -> f64,
) -> (f64, f64, f64) {
    // (局, 決定点, seed) → arm → 値
    let mut by: BTreeMap<(String, u64, u64), BTreeMap<String, f64>> = BTreeMap::new();
    for r in rows {
        by.entry((
            r["game"].as_str().unwrap_or("?").to_string(),
            r["move_number"].as_u64().unwrap_or(0),
            r["seed"].as_u64().unwrap_or(0),
        ))
        .or_default()
        .insert(r["arm"].as_str().unwrap_or("?").to_string(), key(r));
    }
    let mut clusters: BTreeMap<String, (f64, f64)> = BTreeMap::new();
    for ((game, _, _), m) in &by {
        if let (Some(a), Some(b)) = (m.get(arm), m.get(base)) {
            let e = clusters.entry(game.clone()).or_default();
            e.0 += combine(*a, *b);
            e.1 += 1.0;
        }
    }
    let v: Vec<(f64, f64)> = clusters.values().copied().collect();
    let num: f64 = v.iter().map(|(a, _)| a).sum();
    let den: f64 = v.iter().map(|(_, b)| b).sum();
    let point = if den > 0.0 { num / den } else { f64::NAN };
    let (lo, hi) = cluster_ratio_ci(&v, 0.05, 0x31_2026);
    (point, lo, hi)
}

/// **自然頻度の元対局 cluster**（`(局 → (重み付き差の和, 重みの和))`）。
///
/// issue #36 の主 estimand は「**全王手手番の自然頻度**」なので、
///
/// - `--nofoul-cap` で間引いたぶんは行の `weight`（元の層頻度 ÷ 残した本数）で戻す
/// - **両王手は除く**（介入が no-op なので効果量の分母に入れると薄まる）
///
/// 返り値をそのまま渡せば、相手をまたいだ層化 bootstrap（`combined`）も同じ
/// 定義の上で計算できる。
fn natural_clusters(
    rows: &[&serde_json::Value],
    arm: &str,
    base: &str,
    key: &dyn Fn(&serde_json::Value) -> f64,
) -> BTreeMap<String, (f64, f64)> {
    let mut by: BTreeMap<(String, u64, u64), (BTreeMap<String, f64>, f64)> = BTreeMap::new();
    for r in rows {
        if r["double_check"].as_bool().unwrap_or(false) {
            continue; // 両王手は別層（介入しない）
        }
        // **欠測を 1.0 で埋めない**（レビュー5巡目 [P1]）。`weight` は自然頻度へ
        // 戻す包含重みなので、無い行を 1.0 と読むと「間引いた均衡標本」が
        // そのまま自然頻度の表になる。必須列検査で弾いているが、ここでも止める
        let w = r["weight"]
            .as_f64()
            .unwrap_or_else(|| die("weight の無い行が自然頻度の集計に入りました（schema 5 未満の記録が混ざっています）"));
        let e = by
            .entry((
                r["game"].as_str().unwrap_or("?").to_string(),
                r["move_number"].as_u64().unwrap_or(0),
                r["seed"].as_u64().unwrap_or(0),
            ))
            .or_insert_with(|| (BTreeMap::new(), w));
        e.1 = w;
        e.0.insert(r["arm"].as_str().unwrap_or("?").to_string(), key(r));
    }
    let mut clusters: BTreeMap<String, (f64, f64)> = BTreeMap::new();
    for ((game, _, _), (m, w)) in &by {
        if let (Some(a), Some(b)) = (m.get(arm), m.get(base)) {
            let e = clusters.entry(game.clone()).or_default();
            e.0 += w * (a - b);
            e.1 += w;
        }
    }
    clusters
}

/// 層（相手）ごとの cluster を**層の内側で**再標本化し、層平均の percentile CI を返す。
///
/// issue #36 の契約: `Δcombined = (Δv13 + Δv14) / 2`、bootstrap は各相手の内側で
/// 元対局を別々に引き直す（opponent-balanced な層化 cluster bootstrap）。
fn stratified_mean_ci(strata: &[Vec<(f64, f64)>], alpha: f64, seed: u64) -> (f64, f64, f64) {
    let ratio = |v: &[(f64, f64)]| -> Option<f64> {
        let den: f64 = v.iter().map(|(_, d)| d).sum();
        (den > 0.0).then(|| v.iter().map(|(n, _)| n).sum::<f64>() / den)
    };
    let per: Vec<f64> = strata.iter().filter_map(|v| ratio(v)).collect();
    if per.len() != strata.len() || per.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let point = per.iter().sum::<f64>() / per.len() as f64;
    let mut state = seed | 1;
    let mut draws: Vec<f64> = vec![];
    for _ in 0..2000 {
        let mut acc = 0.0;
        let mut ok = true;
        for v in strata {
            if v.is_empty() {
                ok = false;
                break;
            }
            let mut boot: Vec<(f64, f64)> = Vec::with_capacity(v.len());
            for _ in 0..v.len() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                boot.push(v[(state >> 33) as usize % v.len()]);
            }
            match ratio(&boot) {
                Some(r) => acc += r,
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            draws.push(acc / strata.len() as f64);
        }
    }
    if draws.is_empty() {
        return (point, f64::NAN, f64::NAN);
    }
    draws.sort_by(f64::total_cmp);
    let idx = |q: f64| -> f64 {
        let i = ((draws.len() - 1) as f64 * q).round() as usize;
        draws[i.min(draws.len() - 1)]
    };
    (point, idx(alpha / 2.0), idx(1.0 - alpha / 2.0))
}

/// 事前登録した safety margin（issue #36 の P0-2）。
const MARGIN_CATASTROPHE: f64 = 0.005; // 破滅率（≥8 反則）+0.5pt まで
const MARGIN_ACCEPT: f64 = -0.01; // 受理率 −1pt まで
const MARGIN_FOUL_LIMIT: f64 = 0.005; // 即時反則負け +0.5pt まで

/// 主 arm の判定に使う量（1相手ぶん）。`combined` は同じ関数で層を作る
struct GateInput {
    /// `current@real − oracle@kinf@real` の反則/手番（**正 = 改善**）
    reduction: Vec<(f64, f64)>,
    /// arm − base（悪化が正）の安全性3種
    catastrophe: Vec<(f64, f64)>,
    accept: Vec<(f64, f64)>,
    foul_limit: Vec<(f64, f64)>,
}

fn gate_input(rows: &[&serde_json::Value], arm: &str, base: &str) -> GateInput {
    let vals = |k: &dyn Fn(&serde_json::Value) -> f64, flip: bool| -> Vec<(f64, f64)> {
        let (a, b) = if flip { (base, arm) } else { (arm, base) };
        natural_clusters(rows, a, b, k).into_values().collect()
    };
    GateInput {
        // R = current − treatment（正 = 反則が減った）
        reduction: vals(&|r| r["fouls"].as_f64().unwrap_or(0.0), true),
        catastrophe: vals(&|r| f64::from(u8::from(r["fouls"].as_f64().unwrap_or(0.0) >= 8.0)), false),
        accept: vals(
            &|r| f64::from(u8::from(r["truth_accepted"].as_bool().unwrap_or(false))),
            false,
        ),
        foul_limit: vals(
            &|r| f64::from(u8::from(r["foul_limit"].as_bool().unwrap_or(false))),
            false,
        ),
    }
}

fn report(metas: &[serde_json::Value], rows: &[serde_json::Value], allow_incomplete: bool) {
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
    println!("\n=== P0-5: オフライン方策シミュレーション（issue #31）===");
    println!(
        "  実験: 相手 {} / 予算 {}ms / seeds {} / arms {} / 記録 {} / code {}",
        exp["opponent"], exp["budget_ms"], exp["seeds"], exp["arms"], exp["records"],
        exp["source_fingerprint"],
    );
    // 健全性: 仮想更新が初回ランキングの p_legal を再現できているか
    let repro: Vec<f64> = rows
        .iter()
        .filter(|r| r["arm"] == "current@static")
        .map(|r| r["repro_err"].as_f64().unwrap_or(0.0))
        .collect();
    println!(
        "  健全性: 反則0での仮想更新と初回ランキングの p_legal の最大差 平均 {:.4} / p95 {:.4}",
        mean(&repro),
        pct(&repro, 0.95)
    );
    println!(
        "         （大きいなら `evaluate` の p_legal 経路を取りこぼしている。判定の前にここを見る）"
    );
    // **復元の健全性**（P0-4 と同じ読み方）: オフラインの再現が実戦の選択を
    // どれだけ再現できているか。低いと「実戦のその決定そのもの」ではなく
    // 「同じ状態から現行方策が引き直したときの分布」を見ていることになる
    for arm in ["current@static", "current@real"] {
        let sel: Vec<&serde_json::Value> = rows.iter().filter(|r| r["arm"] == arm).collect();
        if sel.is_empty() {
            continue;
        }
        let rate = |k: &str| -> f64 {
            let v: Vec<f64> = sel
                .iter()
                .map(|r| f64::from(u8::from(r[k].as_bool().unwrap_or(false))))
                .collect();
            100.0 * mean(&v)
        };
        println!(
            "  復元（{arm}）: 最初に試した手が実戦と一致 {:.1}% / 受理手が実戦と一致 {:.1}%",
            rate("record_first_matches"),
            rate("record_accepted_matches")
        );
    }

    // ---- issue #36 P0-2 の配管の関門 ---------------------------------------
    // **恒等対照が `current` と一致しないなら判定以前**（中止条件）
    let ident: Vec<f64> = rows
        .iter()
        .filter(|r| r["arm"] == "current@static")
        .filter_map(|r| r["identity_err"].as_f64())
        .collect();
    if !ident.is_empty() {
        let worst = ident.iter().cloned().fold(0.0f64, f64::max);
        println!(
            "  恒等対照（oracle@k1 vs current）の p の最大差 {worst:.6}（0 でなければ配管が壊れている）"
        );
        if worst > 0.0 && !allow_incomplete {
            die("恒等対照が current と一致しません（issue #36 の中止条件: 判定以前）");
        }
    }
    // **`deduce_last_move` が真仮説を落としたら中止**（fallback 前で数える）。
    // 診断は arm ごとの行に入るので、**その arm の行から**集める
    // （`current@static` で絞ると常に空振りする。PR #37 レビュー3巡目 [P1]）
    if let Some(d) = deduce_summary(rows) {
        println!(
            "  deduce_last_move: {} 行 / 落とした仮説 平均 {:.1} 本 / 全滅 fallback {} / **真仮説を落とした {}**",
            d.rows, d.dropped_mean, d.fallback, d.dropped_true
        );
        if d.dropped_true > 0 && !allow_incomplete {
            die("deduce_last_move が真の王手駒の仮説を落としました（健全性違反 = 中止条件）");
        }
    }
    // **オラクルが実際に介入できた行**（真の王手駒がその arm の列挙集合にある）。
    // 粒子なしのソルバーではなく `DiagRecord` の実測なので、「被覆ありと報告した
    // のに実評価の集合には無かった」が起きない（PR #37 レビュー1巡目 [P1]）。
    // **arm ごとに集計する**（同2巡目 [P2]: 単一変数だと最後の1本が全行を上書きする）
    let oracle_arms: BTreeSet<String> = rows
        .iter()
        .filter(|r| !r["oracle"].is_null())
        .map(|r| r["arm"].as_str().unwrap_or("?").to_string())
        .collect();
    let dbl = rows
        .iter()
        .filter(|r| r["arm"] == "current@static")
        .filter(|r| r["double_check"].as_bool().unwrap_or(false))
        .count();
    for arm in &oracle_arms {
        let cov: Vec<&serde_json::Value> =
            rows.iter().filter(|r| r["arm"].as_str() == Some(arm.as_str())).collect();
        let all = cov
            .iter()
            .filter(|r| r["oracle"]["covered_all"].as_bool().unwrap_or(false))
            .count();
        let part = cov
            .iter()
            .filter(|r| {
                let t = r["oracle"]["truth_present"].as_u64().unwrap_or(0);
                let n = r["oracle"]["constructions"].as_u64().unwrap_or(0);
                t > 0 && t < n
            })
            .count();
        let fb: u64 = cov
            .iter()
            .filter_map(|r| r["oracle"]["weight_fallback"].as_u64())
            .sum();
        println!(
            "  {arm}: 全構築で介入できた行 {all} / {}（一部の構築だけ {part} / \
             倍率が全滅して元へ戻した構築 {fb} / 両王手 {dbl} は別層・介入しない）",
            cov.len()
        );
        if part > 0 {
            println!(
                "    **一部の構築でしか真仮説が列挙されていない行がある**: その行の \
                 {arm} は full oracle ではない（判定の前にここを見る）"
            );
        }
    }
    // **実再決定側の恒等対照 = 再決定のノイズ床**（`oracle@k1@real` vs
    // `current@real`。PR #37 レビュー2巡目 [P1]）。
    //
    // 両 arm は同じ instance を clone して同じ再ランキング経路を通る（= 系統的な
    // 「別の粒子サンプルを比べる」交絡は無い）が、**bit-exact にはならない**:
    // `choose` は壁時計デッドラインまで粒子を若返らせるので、同じ状態から引き直しても
    // 集合が変わる（#31 P0-4 の「再決定そのもののノイズ床」）。実測でも同じ入力の
    // 4回の実行で 0.006 / 0.085 / 0.045 / 0.032 と揺れた。したがってここは
    // **門ではなく水準の報告**で、主 arm の差をこの床と並べて読むためにある。
    // 床が測れていない（`oracle@k1@real` が無い）ことのほうを入力契約で弾く
    let ident_real: Vec<f64> = rows
        .iter()
        .filter(|r| r["arm"] == "current@static")
        .filter_map(|r| r["identity_err_real"].as_f64())
        .collect();
    if !ident_real.is_empty() {
        println!(
            "  再決定のノイズ床（oracle@k1@real vs current@real の p の差）: \
             平均 {:.4} / p95 {:.4} / 最大 {:.4}",
            mean(&ident_real),
            pct(&ident_real, 0.95),
            ident_real.iter().cloned().fold(0.0f64, f64::max),
        );
        println!(
            "    （**門ではない**: 壁時計予算のせいで同じ状態から引き直しても集合が\
             変わる。主 arm の差はこの床と並べて読む）"
        );
    }

    // ---- 主 estimand: **全王手手番の自然頻度**（issue #36 の事前登録）----------
    // `foul` / `nofoul` の2表は層の記述で、合算に使うと少数の foul 層を
    // 過大に重み付けする（PR #37 レビュー4巡目 [P1]）
    let all: Vec<&serde_json::Value> = rows.iter().collect();
    let dbl_rows = rows
        .iter()
        .filter(|r| r["arm"] == "current@static" && r["double_check"].as_bool().unwrap_or(false))
        .count();
    let term_rows = rows
        .iter()
        .filter(|r| r["arm"] == "current@static" && r["terminal"].as_bool().unwrap_or(false))
        .count();
    let arms_all: BTreeSet<String> = rows
        .iter()
        .map(|r| r["arm"].as_str().unwrap_or("?").to_string())
        .collect();
    println!(
        "\n--- 主 estimand: 全王手手番の自然頻度（単王手のみ。両王手 {dbl_rows} 行は別層 / 終端手番 {term_rows} 行を含む）---"
    );
    println!("  {:<22} {:>10} {:>26}", "arm", "反則/手番差", "[元対局 cluster CI]");
    for arm in &arms_all {
        let base = baseline_for(arm);
        if arm == base {
            continue;
        }
        let v: Vec<(f64, f64)> =
            natural_clusters(&all, arm, base, &|r| r["fouls"].as_f64().unwrap_or(0.0))
                .into_values()
                .collect();
        let den: f64 = v.iter().map(|(_, d)| d).sum();
        if den <= 0.0 {
            continue;
        }
        let point = v.iter().map(|(n, _)| n).sum::<f64>() / den;
        let (lo, hi) = cluster_ratio_ci(&v, 0.05, 0x36_2026);
        println!("  {arm:<22} {point:>+10.3} {:>26}", format!("[{lo:+.3}, {hi:+.3}] vs {base}"));
    }
    // ---- P0-2 の事前登録した採否規則 ----------------------------------------
    let main_arm = "oracle@kinf@real";
    if arms_all.contains(main_arm) && arms_all.contains("current@real") {
        let g = gate_input(&all, main_arm, "current@real");
        let show = |label: &str, v: &[(f64, f64)], margin: f64, higher_is_bad: bool| -> bool {
            let den: f64 = v.iter().map(|(_, d)| d).sum();
            if den <= 0.0 {
                println!("    {label:<26} 判定不能（母数 0）");
                return false;
            }
            let point = v.iter().map(|(n, _)| n).sum::<f64>() / den;
            let (lo, hi) = cluster_ratio_ci(v, 0.05, 0x36_2027);
            let ok = if higher_is_bad { point <= margin } else { point >= margin };
            println!(
                "    {label:<26} {point:+.4} [{lo:+.4}, {hi:+.4}]  margin {margin:+.4}  {}",
                if ok { "OK" } else { "**NG**" }
            );
            ok
        };
        println!("\n--- P0-2 の採否判定（主 arm {main_arm} vs current@real、自然頻度）---");
        let rden: f64 = g.reduction.iter().map(|(_, d)| d).sum();
        let (rpoint, rlo, rhi) = if rden > 0.0 {
            let p = g.reduction.iter().map(|(n, _)| n).sum::<f64>() / rden;
            let (lo, hi) = cluster_ratio_ci(&g.reduction, 0.05, 0x36_2028);
            (p, lo, hi)
        } else {
            (f64::NAN, f64::NAN, f64::NAN)
        };
        println!(
            "    {:<26} {rpoint:+.4} [{rlo:+.4}, {rhi:+.4}]  門 CI 下限 > 0  {}",
            "R_foul（current − oracle）",
            if rlo > 0.0 { "OK" } else { "**NG**" }
        );
        let s1 = show("破滅率（≥8反則）", &g.catastrophe, MARGIN_CATASTROPHE, true);
        let s2 = show("受理率", &g.accept, MARGIN_ACCEPT, false);
        let s3 = show("即時反則負け", &g.foul_limit, MARGIN_FOUL_LIMIT, true);
        println!(
            "    → この相手だけの判定: {}（**採否は相手をまたいだ合算** = `check_policy combined`）",
            if rlo > 0.0 && s1 && s2 && s3 { "通過" } else { "不通過" }
        );
    } else {
        println!(
            "\n（主 arm {main_arm} か current@real が無いので P0-2 の採否判定は出さない）"
        );
    }

    for estimand in ["foul", "nofoul"] {
        let sel: Vec<&serde_json::Value> =
            rows.iter().filter(|r| r["estimand"] == estimand).collect();
        if sel.is_empty() {
            continue;
        }
        let mut by_arm: BTreeMap<String, ArmStats> = BTreeMap::new();
        for r in &sel {
            by_arm
                .entry(r["arm"].as_str().unwrap_or("?").to_string())
                .or_default()
                .push(r);
        }
        let points: BTreeSet<(String, u64)> = sel
            .iter()
            .map(|r| {
                (
                    r["game"].as_str().unwrap_or("?").to_string(),
                    r["move_number"].as_u64().unwrap_or(0),
                )
            })
            .collect();
        println!(
            "\n--- estimand {estimand}（決定点 {} / unit {}）---",
            points.len(),
            by_arm.values().map(|s| s.n).max().unwrap_or(0)
        );
        let record_fouls = mean(
            &sel.iter()
                .filter(|r| r["arm"] == "current@static")
                .map(|r| r["record_fouls"].as_f64().unwrap_or(0.0))
                .collect::<Vec<_>>(),
        );
        println!("  実戦の反則/手番（記録）: {record_fouls:.2}");
        println!(
            "  {:<22} {:>7} {:>8} {:>7} {:>7} {:>8} {:>8} {:>9} {:>9}",
            "arm", "反則/番", "Δ vs cur", "上限%", "≥8%", "受理%", "被詰1%", "材料差", "更新/番"
        );
        for (arm, st) in &by_arm {
            let n = st.n.max(1) as f64;
            // **実再決定 arm の対照は `current@real`**（PR #37 レビュー3巡目 [P1]）。
            // shadow を対照にすると、介入に「shadow → 実再決定」の差が混ざる
            let base = baseline_for(arm);
            let (d, lo, hi) = if arm == base {
                (0.0, 0.0, 0.0)
            } else {
                paired_ci(&sel, arm, base, &|r| r["fouls"].as_f64().unwrap_or(0.0))
            };
            println!(
                "  {:<22} {:>7.2} {:>+8.2} {:>7.1} {:>7.1} {:>8.1} {:>8.1} {:>9.2} {:>9.2}{}",
                arm,
                mean(&st.fouls),
                d,
                100.0 * st.foul_limit as f64 / n,
                100.0 * st.heavy as f64 / n,
                100.0 * st.accepted as f64 / n,
                100.0 * st.mated as f64 / n,
                mean(&st.material),
                mean(&st.updates),
                if arm == base {
                    String::new()
                } else {
                    format!("  [{lo:+.2}, {hi:+.2}] vs {base}")
                }
            );
        }
        // 受理直後の被一手詰め（真実ベース。受理手の gain は自己正当化するので主指標）。
        // 対照は arm ごと（実再決定 arm は `current@real`）
        println!("  被一手詰めのペア差（対照は arm ごと、元対局 cluster CI）:");
        for arm in by_arm.keys() {
            let base = baseline_for(arm);
            if arm == base {
                continue;
            }
            let (d, lo, hi) = paired_ci(&sel, arm, base, &|r| {
                f64::from(u8::from(r["mated_in_1"].as_bool().unwrap_or(false)))
            });
            println!("    {arm:<22} {d:+.4} [{lo:+.4}, {hi:+.4}] vs {base}");
        }
        // **ノイズ床**: 介入なしの実再決定 arm（`oracle@k1@real`）の同じ endpoint。
        // 主 arm の差はこれと並べて読む（再決定そのもののばらつき）
        if by_arm.contains_key("oracle@k1@real") {
            let (d, lo, hi) = paired_ci(&sel, "oracle@k1@real", "current@real", &|r| {
                r["fouls"].as_f64().unwrap_or(0.0)
            });
            let (m, mlo, mhi) = paired_ci(&sel, "oracle@k1@real", "current@real", &|r| {
                f64::from(u8::from(r["mated_in_1"].as_bool().unwrap_or(false)))
            });
            println!(
                "  再決定のノイズ床（oracle@k1@real − current@real）: 反則/番 {d:+.2} \
                 [{lo:+.2}, {hi:+.2}] / 被一手詰め {m:+.4} [{mlo:+.4}, {mhi:+.4}]"
            );
        }
        // 型別の反則（α を型で切らずに一律に掛けると型2 が沈む、を見る）
        let mut types: BTreeSet<String> = BTreeSet::new();
        for r in &sel {
            types.insert(r["type"].as_str().unwrap_or("?").to_string());
        }
        if types.len() > 1 {
            println!("  型別の反則/手番:");
            print!("    {:<22}", "arm");
            for t in &types {
                print!(" {t:>18}");
            }
            println!();
            for arm in by_arm.keys() {
                print!("    {arm:<22}");
                for t in &types {
                    let v: Vec<f64> = sel
                        .iter()
                        .filter(|r| r["arm"] == arm.as_str() && r["type"] == t.as_str())
                        .map(|r| r["fouls"].as_f64().unwrap_or(0.0))
                        .collect();
                    print!(" {:>18.2}", mean(&v));
                }
                println!();
            }
        }
    }

    // 近似通過条件（issue #31 の事前登録）
    approximation_gate(rows);
    calibration(rows);
    println!(
        "\n  コスト: 仮想更新 1回の実時間 p95 {:.0}µs（P1-β は候補×候補ぶん掛かる。思考 p95 の悪化 +200ms 以内が P1 の門）",
        pct(
            &rows
                .iter()
                .filter(|r| r["arm"] == "current@shadow" && r["updates"].as_f64().unwrap_or(0.0) > 0.0)
                .map(|r| r["sim_us"].as_f64().unwrap_or(0.0)
                    / r["updates"].as_f64().unwrap_or(1.0))
                .collect::<Vec<_>>(),
            0.95
        )
    );
}

/// **近似通過条件**（事前登録）: β の対現行改善量が正のときだけ比率を定義し、
/// 仮想更新と実再決定の「受理までの反則数/手番」の差がその **25% 以下**。
/// 改善量が 0 以下なら β は不採用。
fn approximation_gate(rows: &[serde_json::Value]) {
    let sel: Vec<&serde_json::Value> = rows.iter().filter(|r| r["estimand"] == "foul").collect();
    if sel.is_empty() {
        return;
    }
    let has_real = sel.iter().any(|r| r["arm"] == "current@real");
    println!("\n--- 近似通過条件（p-only shadow update vs 実再決定）---");
    if !has_real {
        println!("  実再決定の arm がありません（--no-real で回した）。判定は出せない");
        return;
    }
    let fouls = |r: &serde_json::Value| r["fouls"].as_f64().unwrap_or(0.0);
    // **門は unit ごとの絶対誤差**（符号付き平均だと正負が相殺して通ってしまう）
    let (gap, glo, ghi) = paired_abs_ci(&sel, "current@shadow", "current@real", &fouls);
    // 符号付きの差は「仮想更新に偏りがあるか」の情報として別に出す（門には使わない）
    let (bias, blo, bhi) = paired_ci(&sel, "current@shadow", "current@real", &fouls);
    let agree = {
        let mut by: BTreeMap<(String, u64, u64), BTreeMap<String, String>> = BTreeMap::new();
        for r in &sel {
            by.entry((
                r["game"].as_str().unwrap_or("?").to_string(),
                r["move_number"].as_u64().unwrap_or(0),
                r["seed"].as_u64().unwrap_or(0),
            ))
            .or_default()
            .insert(
                r["arm"].as_str().unwrap_or("?").to_string(),
                r["accepted"].as_str().unwrap_or("").to_string(),
            );
        }
        let pairs: Vec<bool> = by
            .values()
            .filter_map(|m| Some(m.get("current@shadow")? == m.get("current@real")?))
            .collect();
        if pairs.is_empty() {
            f64::NAN
        } else {
            pairs.iter().filter(|b| **b).count() as f64 / pairs.len() as f64
        }
    };
    println!(
        "  **受理までの反則数の絶対誤差 |shadow − real|（門はこちら）: {gap:.3} [{glo:.3}, {ghi:.3}]**"
    );
    println!(
        "  符号付きの差（shadow − real、偏りの情報。門には使わない）: {bias:+.3} [{blo:+.3}, {bhi:+.3}]"
    );
    println!("  受理手が一致した割合: {:.1}%", 100.0 * agree);
    // β の対現行改善量（反則が減る = 負の差なので符号を反転して「改善量」にする）
    let best_beta = rows
        .iter()
        .filter(|r| r["estimand"] == "foul")
        .filter_map(|r| r["arm"].as_str())
        .filter(|a| a.starts_with("beta"))
        .collect::<BTreeSet<_>>();
    let mut best: Option<(String, f64)> = None;
    for arm in best_beta {
        let (d, _, _) = paired_ci(&sel, arm, "current@shadow", &fouls);
        let improve = -d;
        if best.as_ref().is_none_or(|(_, b)| improve > *b) {
            best = Some((arm.to_string(), improve));
        }
    }
    match best {
        Some((arm, improve)) if improve > 0.0 => {
            let ratio = gap / improve;
            println!(
                "  β の対現行改善量の最大は {arm} の {improve:+.3}（反則/手番）→ 近似誤差 / 改善量 = {ratio:.2} → {}",
                if ratio <= 0.25 {
                    "**通過**（仮想更新で事前スコアを作ってよい）"
                } else {
                    "**不通過**（P1-β は候補ごとに推定器を clone して仮想 MyFoul を注入する設計が要る）"
                }
            );
        }
        Some((arm, improve)) => println!(
            "  β の対現行改善量が 0 以下（{arm} で {improve:+.3}）→ **β は不採用**（比率は定義しない）"
        ),
        None => println!("  β の arm がありません"),
    }
}

/// 候補分布上の較正（**現行方策が選んだ手ではなく候補全体**）。
///
/// P0-2 は「選ばれた手」の較正なので、順位を変えた後の候補分布へは外挿できない。
/// ここは初回ランキングの全候補を真実でラベル付けして手種ごとに出す。
fn calibration(rows: &[serde_json::Value]) {
    // 1決定点×seed につき1回だけ数える（arm ごとに同じ値が入っている）
    let mut by_kind: BTreeMap<String, BTreeMap<String, (f64, f64, f64)>> = BTreeMap::new();
    let mut seen: HashSet<(String, u64, u64)> = HashSet::new();
    for r in rows {
        let key = (
            r["game"].as_str().unwrap_or("?").to_string(),
            r["move_number"].as_u64().unwrap_or(0),
            r["seed"].as_u64().unwrap_or(0),
        );
        if !seen.insert(key.clone()) {
            continue;
        }
        let Some(obj) = r["calibration"].as_object() else {
            continue;
        };
        for (kind, v) in obj {
            let e = by_kind
                .entry(kind.clone())
                .or_default()
                .entry(key.0.clone())
                .or_default();
            e.0 += v["p_sum"].as_f64().unwrap_or(0.0);
            e.1 += v["legal"].as_f64().unwrap_or(0.0);
            e.2 += v["n"].as_f64().unwrap_or(0.0);
        }
    }
    if by_kind.is_empty() {
        return;
    }
    println!("\n--- 候補分布上の較正（真実ラベル・元対局 cluster CI）---");
    println!(
        "  {:<18} {:>9} {:>9} {:>9} {:>24}",
        "手種", "候補数", "平均予測", "合法率", "方向つき較正誤差"
    );
    for (kind, by_game) in &by_kind {
        let n: f64 = by_game.values().map(|(_, _, n)| n).sum();
        let psum: f64 = by_game.values().map(|(p, _, _)| p).sum();
        let legal: f64 = by_game.values().map(|(_, l, _)| l).sum();
        if n <= 0.0 {
            continue;
        }
        // 誤差の cluster: 局ごとの (予測の和 − 合法数, 候補数)
        let clusters: Vec<(f64, f64)> = by_game.values().map(|(p, l, n)| (p - l, *n)).collect();
        let (lo, hi) = cluster_ratio_ci(&clusters, 0.05, 0x31_2027);
        println!(
            "  {:<18} {:>9.0} {:>9.3} {:>9.3} {:>+11.3} [{lo:+.3}, {hi:+.3}]",
            kind,
            n,
            psum / n,
            legal / n,
            (psum - legal) / n
        );
    }
}

/// 2つの `(USI, p)` 列を**手で突き合わせて**最大差と「片側にしか無い候補の本数」を返す。
///
/// 実再決定は候補リストごと作り直すので、添字で `zip` すると順位が入れ替わった
/// ときに別の手の p を比べる（PR #37 レビュー3巡目 [P2]）。共通候補で差を取り、
/// 片側だけの候補は数えて明示する（0 でなければ候補集合そのものが違う）。
fn p_gap(a: &[(String, f64)], b: &[(String, f64)]) -> (f64, usize) {
    let ma: BTreeMap<&str, f64> = a.iter().map(|(u, p)| (u.as_str(), *p)).collect();
    let mb: BTreeMap<&str, f64> = b.iter().map(|(u, p)| (u.as_str(), *p)).collect();
    let mut worst = 0.0f64;
    let mut only = 0usize;
    for (u, pa) in &ma {
        match mb.get(u) {
            Some(pb) => worst = worst.max((pa - pb).abs()),
            None => only += 1,
        }
    }
    only += mb.keys().filter(|u| !ma.contains_key(*u)).count();
    (worst, only)
}

/// その arm の対照（**実再決定 arm は `current@real`**、それ以外は `current@shadow`）。
///
/// 実再決定は「shadow → 粒子を引き直しての再ランキング」の差を含むので、
/// shadow を対照にすると介入の効果と混ざる（PR #37 レビュー3巡目 [P1]）。
fn baseline_for(arm: &str) -> &'static str {
    if arm.ends_with("@real") { "current@real" } else { "current@shadow" }
}

/// `deduce_last_move` の健全性の集計（`report` の関門とテストが同じ経路を通る）。
#[derive(Debug, PartialEq)]
struct DeduceSummary {
    rows: usize,
    dropped_mean: f64,
    fallback: usize,
    /// **真の王手駒の仮説を落とした行数**（1つでもあれば中止）
    dropped_true: usize,
}

/// 診断は **`deduce_last_move` arm の行**に入る（`current@static` ではない）。
fn deduce_summary(rows: &[serde_json::Value]) -> Option<DeduceSummary> {
    let ded: Vec<&serde_json::Value> = rows.iter().filter(|r| !r["deduce"].is_null()).collect();
    if ded.is_empty() {
        return None;
    }
    let dropped: Vec<f64> = ded
        .iter()
        .filter_map(|r| r["deduce"]["dropped"].as_f64())
        .collect();
    Some(DeduceSummary {
        rows: ded.len(),
        dropped_mean: mean(&dropped),
        fallback: ded
            .iter()
            .filter(|r| r["deduce"]["fallback"].as_bool().unwrap_or(false))
            .count(),
        dropped_true: ded
            .iter()
            .filter(|r| r["deduce"]["dropped_true"].as_bool().unwrap_or(false))
            .count(),
    })
}

/// `report` の入力契約。破っている点を全部返す
fn check_inputs(metas: &[serde_json::Value], rows: &[serde_json::Value]) -> Vec<String> {
    let mut out = vec![];
    if metas.is_empty() {
        return vec!["meta 行がありません".into()];
    }
    let first = &metas[0]["experiment"];
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
    // シャードが揃っているか（欠けると決定点の分母が縮む）
    let total = first["shard_total"].as_u64().unwrap_or(1) as usize;
    let mut shards: Vec<u64> = metas.iter().filter_map(|m| m["shard"].as_u64()).collect();
    let n = shards.len();
    shards.sort_unstable();
    shards.dedup();
    if shards.len() != n {
        out.push("同じシャードの JSONL を2回渡しています（行が二重に数えられる）".into());
    }
    if shards.len() != total {
        out.push(format!(
            "シャードが欠けています（{:?} / 全 {total}）: 決定点の分母が縮むので判定は出せない",
            shards
        ));
    }
    // 重複行
    let mut keys: HashSet<(String, u64, u64, String)> = HashSet::new();
    let mut dups = 0;
    for r in rows {
        let k = (
            r["game"].as_str().unwrap_or("?").to_string(),
            r["move_number"].as_u64().unwrap_or(0),
            r["seed"].as_u64().unwrap_or(0),
            r["arm"].as_str().unwrap_or("?").to_string(),
        );
        if !keys.insert(k) {
            dups += 1;
        }
    }
    if dups > 0 {
        out.push(format!("重複行が {dups} 件あります"));
    }
    // **meta が宣言した決定点に対して行が揃っているか**（ある seed の全 arm が
    // まとめて欠けても検出する）
    let seeds = first["seeds"].as_u64().unwrap_or(0);
    let mut want_arms: Vec<String> = vec!["current@static".into()];
    for a in first["arms"].as_array().into_iter().flatten() {
        want_arms.push(format!("{}@shadow", a.as_str().unwrap_or("?")));
    }
    // issue #36 P0-2 のオラクル arm
    for b in first["beliefs"].as_array().into_iter().flatten() {
        want_arms.push(format!("{}@shadow", b.as_str().unwrap_or("?")));
    }
    let real_beliefs: Vec<&str> = first["beliefs_real"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|b| b.as_str())
        .collect();
    for b in &real_beliefs {
        want_arms.push(format!("{b}@real"));
    }
    if first["with_real"].as_bool().unwrap_or(false) {
        want_arms.push("current@real".into());
        // 実再決定のオラクル arm があれば、その恒等対照も必ず走る
        if real_beliefs.iter().any(|b| b.starts_with("oracle@k")) {
            want_arms.push("oracle@k1@real".into());
        }
    }
    want_arms.sort();
    want_arms.dedup();
    // **必須列の存在検査**（`identity_err` が無い行を「差 0」と読むと
    // 恒等対照の関門が素通りする）
    let mut missing: BTreeMap<&str, usize> = BTreeMap::new();
    for r in rows {
        for k in REQUIRED_ROW_KEYS {
            if r.get(k).is_none() {
                *missing.entry(k).or_default() += 1;
            }
        }
    }
    for (k, n) in missing {
        out.push(format!("必須列 {k} が {n} 行で欠けています（schema {ROW_SCHEMA} の契約違反）"));
    }
    // **恒等対照は外せない**（PR #37 レビュー [P2]）: `--belief` から `oracle@k1` を
    // 抜くと `identity_err` が null のままになり、`report` の関門が「数値が0件」で
    // 素通りする。オラクル arm を1つでも宣言した実験では恒等対照を必須にする
    let beliefs: Vec<String> = first["beliefs"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|b| b.as_str().map(str::to_string))
        .collect();
    let reals: Vec<String> = first["beliefs_real"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|b| b.as_str().map(str::to_string))
        .collect();
    if !beliefs.is_empty() || !reals.is_empty() {
        if !beliefs.iter().any(|b| b == "oracle@k1") {
            out.push(
                "オラクル arm を宣言した実験に恒等対照 oracle@k1 がありません                 （`current@shadow` と bit-exact であることを毎回確かめる契約）"
                    .into(),
            );
        }
        let has_ident = rows.iter().any(|r| r["identity_err"].as_f64().is_some());
        if !has_ident {
            out.push(
                "identity_err が全行 null です（恒等対照が1度も走っていない）".into(),
            );
        }
        // 実再決定のオラクル arm を回したなら、**再決定のノイズ床**も必ず測る
        // （主 arm の差をこの床と並べないと「介入の効果」に見えてしまう）
        if reals.iter().any(|b| b.starts_with("oracle@k"))
            && first["with_real"].as_bool().unwrap_or(false)
            && !rows.iter().any(|r| r["identity_err_real"].as_f64().is_some())
        {
            out.push(
                "identity_err_real が全行 null です（実再決定のノイズ床                  oracle@k1@real vs current@real が測れていない）"
                    .into(),
            );
        }
    }
    // **実キー集合と期待キー集合を厳密一致させる**（PR #37 レビュー5巡目 [P1]）。
    // 欠落しか見ないと、meta が宣言していない決定点や範囲外 seed の行を全 arm ぶん
    // 足しても検査を通り、自然頻度 gate は `rows` を全部読むので主 CI を動かせる。
    // seed もキーに入れる（`got != seeds` の本数比較では範囲外 seed が
    // 別の欠落と相殺しうる）
    let mut seen: BTreeMap<(String, u64, u64, String), usize> = BTreeMap::new();
    for r in rows {
        *seen
            .entry((
                r["game"].as_str().unwrap_or("?").to_string(),
                r["move_number"].as_u64().unwrap_or(0),
                r["seed"].as_u64().unwrap_or(u64::MAX),
                r["arm"].as_str().unwrap_or("?").to_string(),
            ))
            .or_default() += 1;
    }
    // 点属性（分母と対象集合を決める列）は meta の宣言と一致していること。
    // 片方だけ書き換えて分母を変えられてはいけない
    // **`double_check` も点属性**（PR #37 レビュー6巡目 [P1]）: 両王手は主 estimand の
    // 分母から外れるので、行だけに書いてあると全 arm ぶん `false` へ書き換えるだけで
    // 除外されていた手番が分母へ戻る。meta 側の欠測は既定値で埋めず**失敗**させる
    // （埋めると schema を上げた意味が消える）
    let mut point_attrs: BTreeMap<(String, u64), (String, bool, String, bool)> = BTreeMap::new();
    let mut meta_missing: Vec<String> = vec![];
    for m in metas {
        for d in m["points_detail"].as_array().into_iter().flatten() {
            let key = (
                d["game"].as_str().unwrap_or("?").to_string(),
                d["move_number"].as_u64().unwrap_or(0),
            );
            let (Some(terminal), Some(double_check)) =
                (d["terminal"].as_bool(), d["double_check"].as_bool())
            else {
                meta_missing.push(format!("{}#{}", key.0, key.1));
                continue;
            };
            point_attrs.insert(
                key,
                (
                    d["estimand"].as_str().unwrap_or("?").to_string(),
                    terminal,
                    fmt_num(d["weight"].as_f64().unwrap_or(f64::NAN)),
                    double_check,
                ),
            );
        }
    }
    if !meta_missing.is_empty() {
        out.push(format!(
            "meta の points_detail に terminal / double_check がありません（{} 件）: {}{}",
            meta_missing.len(),
            meta_missing.iter().take(3).cloned().collect::<Vec<_>>().join(" / "),
            if meta_missing.len() > 3 { " ..." } else { "" }
        ));
    }
    let mut attr_bad: Vec<String> = vec![];
    for r in rows {
        let key = (
            r["game"].as_str().unwrap_or("?").to_string(),
            r["move_number"].as_u64().unwrap_or(0),
        );
        let Some(want) = point_attrs.get(&key) else {
            continue; // 未宣言の点は下の厳密一致で捕まる
        };
        let got = (
            r["estimand"].as_str().unwrap_or("?").to_string(),
            r["terminal"].as_bool().unwrap_or(false),
            fmt_num(r["weight"].as_f64().unwrap_or(f64::NAN)),
            r["double_check"].as_bool().unwrap_or(false),
        );
        if got != *want {
            attr_bad.push(format!(
                "{}#{} {}: 行 {:?} vs meta {:?}",
                key.0,
                key.1,
                r["arm"].as_str().unwrap_or("?"),
                got,
                want
            ));
        }
    }
    if !attr_bad.is_empty() {
        out.push(format!(
            "行の点属性（estimand / terminal / weight / double_check）が meta の宣言と食い違います（{} 件）: {}{}",
            attr_bad.len(),
            attr_bad.iter().take(3).cloned().collect::<Vec<_>>().join(" / "),
            if attr_bad.len() > 3 { " ..." } else { "" }
        ));
    }
    let dropped: HashSet<(String, u64)> = metas
        .iter()
        .flat_map(|m| m["dropped"].as_array().cloned().unwrap_or_default())
        .map(|d| {
            (
                d["game"].as_str().unwrap_or("?").to_string(),
                d["move_number"].as_u64().unwrap_or(0),
            )
        })
        .collect();
    let mut want: BTreeSet<(String, u64, u64, String)> = BTreeSet::new();
    for m in metas {
        for d in m["points_detail"].as_array().into_iter().flatten() {
            let g = d["game"].as_str().unwrap_or("?").to_string();
            let mn = d["move_number"].as_u64().unwrap_or(0);
            // ランキングが取れなかった決定点は行が無いのが正常
            if dropped.contains(&(g.clone(), mn)) {
                continue;
            }
            for arm in &want_arms {
                for seed in 0..seeds {
                    want.insert((g.clone(), mn, seed, arm.clone()));
                }
            }
        }
    }
    let got: BTreeSet<(String, u64, u64, String)> = seen.keys().cloned().collect();
    let fmt = |k: &(String, u64, u64, String)| format!("{}#{} s{} {}", k.0, k.1, k.2, k.3);
    let lacks: Vec<String> = want.difference(&got).map(fmt).collect();
    if !lacks.is_empty() {
        out.push(format!(
            "meta が宣言した決定点に対して行が {} 箇所欠けています: {}{}",
            lacks.len(),
            lacks.iter().take(3).cloned().collect::<Vec<_>>().join(" / "),
            if lacks.len() > 3 { " ..." } else { "" }
        ));
    }
    // **余分な行も拒否する**（レビュー5巡目 [P1]）。自然頻度 gate は `rows` を
    // 全部読むので、未宣言の決定点や範囲外 seed を足せば主 CI を動かせる
    let extra: Vec<String> = got.difference(&want).map(fmt).collect();
    if !extra.is_empty() {
        out.push(format!(
            "meta が宣言していない行が {} 件あります（未宣言の決定点か範囲外の seed）: {}{}",
            extra.len(),
            extra.iter().take(3).cloned().collect::<Vec<_>>().join(" / "),
            if extra.len() > 3 { " ..." } else { "" }
        ));
    }
    // **実再決定 arm の AB/BA が閉じているか**（レビュー5巡目 [P1]、
    // 検査単位はレビュー6巡目 [P1] で決定点ごとへ）。
    // `arm_order` は unit 内の実行順なので、主対 (`oracle@kinf@real` /
    // `current@real`) が「先」になった seed の本数が釣り合っていること。
    // **全決定点の総数で見てはいけない**: 実装は
    // `treatment_first = (決定点番号 + seed) % 2` で cluster の内側に均衡を閉じるので、
    // 「決定点 A は全 seed が treatment 先・決定点 B は全 seed が対照 先」でも総数は
    // 釣り合ってしまい、決定点ごとの実行順効果がペア差に残る
    if want_arms.iter().any(|a| a == "current@real") {
        // unit（決定点 × seed）ごとの実 arm の実行順
        let mut order: BTreeMap<(String, u64, u64), BTreeMap<String, u64>> = BTreeMap::new();
        for r in rows {
            let arm = r["arm"].as_str().unwrap_or("?");
            if !arm.ends_with("@real") {
                continue;
            }
            let Some(o) = r["arm_order"].as_u64() else { continue };
            order
                .entry((
                    r["game"].as_str().unwrap_or("?").to_string(),
                    r["move_number"].as_u64().unwrap_or(0),
                    r["seed"].as_u64().unwrap_or(u64::MAX),
                ))
                .or_default()
                .insert(arm.to_string(), o);
        }
        // 同じ unit の実 arm が同じ `arm_order` を持っていたら前後関係が決まらない
        let dup: Vec<String> = order
            .iter()
            .filter(|(_, m)| m.values().collect::<BTreeSet<_>>().len() != m.len())
            .map(|((g, mn, s), _)| format!("{g}#{mn} seed{s}"))
            .collect();
        if !dup.is_empty() {
            out.push(format!(
                "実再決定 arm の arm_order が unit 内で重複しています（{} 件）: {}{}",
                dup.len(),
                dup.iter().take(3).cloned().collect::<Vec<_>>().join(" / "),
                if dup.len() > 3 { " ..." } else { "" }
            ));
        }
        for treat in want_arms.iter().filter(|a| a.ends_with("@real") && *a != "current@real") {
            // (game, move_number) ごとに treatment 先 / 対照 先 を数える
            let mut per_point: BTreeMap<(String, u64), (usize, usize)> = BTreeMap::new();
            for ((g, mn, _seed), m) in &order {
                if let (Some(a), Some(b)) = (m.get(treat), m.get("current@real")) {
                    let e = per_point.entry((g.clone(), *mn)).or_default();
                    if a < b {
                        e.0 += 1
                    } else {
                        e.1 += 1
                    }
                }
            }
            let bad: Vec<String> = per_point
                .iter()
                .filter(|(_, (f, s))| f != s)
                .map(|((g, mn), (f, s))| format!("{g}#{mn}（先 {f} / 後 {s}）"))
                .collect();
            if !bad.is_empty() {
                out.push(format!(
                    "{treat} と current@real の実行順が決定点の中で均衡していません（{} 決定点）: {}{}。AB/BA が閉じていないと実行順効果がペア差に残ります",
                    bad.len(),
                    bad.iter().take(3).cloned().collect::<Vec<_>>().join(" / "),
                    if bad.len() > 3 { " ..." } else { "" }
                ));
            }
        }
    }
    out
}

/// **相手をまたいだ最終集約**（issue #36 の契約）。
///
/// `Δcombined = (Δv13 + Δv14) / 2` を層化 cluster bootstrap（各相手の内側で
/// 元対局を引き直す）で出し、veto `Δv13 > 0 && Δv14 > 0` と安全性 margin を
/// **fail-closed** で判定する。相手ごとの report を並べるだけでは、主 CI 下限も
/// veto も検査されない（PR #37 レビュー4巡目 [P1]）。
///
/// 入力は各相手の JSONL（`experiment.opponent` で層に分ける）。
fn run_combined(args: &[String]) {
    let mut allow_incomplete = false;
    let mut paths: Vec<String> = vec![];
    // 契約の既定（issue #36 は v13 / v14 の2相手）。空文字で照合を切れる
    let mut expect_opponents = "estimator_v13,estimator_v14".to_string();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--allow-incomplete" => allow_incomplete = true,
            "--expect-opponents" => {
                expect_opponents = it
                    .next()
                    .unwrap_or_else(|| die("--expect-opponents に値がありません"))
                    .clone()
            }
            x if x.starts_with("--") => die(&format!("未知のオプション: {x}")),
            x => paths.push(x.to_string()),
        }
    }
    if paths.is_empty() {
        die("combined には各相手の JSONL を指定してください");
    }
    // 相手ごとに meta / rows を分ける
    let mut by_opp: BTreeMap<String, (Vec<serde_json::Value>, Vec<serde_json::Value>)> =
        BTreeMap::new();
    // **1ファイル = 1シャード = 1相手**（meta の `experiment.opponent` が層のラベル）。
    // 行そのものは相手を持たないので、同じファイルの meta から引く
    for p in &paths {
        let text =
            std::fs::read_to_string(p).unwrap_or_else(|e| die(&format!("{p} を読めません: {e}")));
        let mut metas = vec![];
        let mut rows = vec![];
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let v: serde_json::Value = serde_json::from_str(line)
                .unwrap_or_else(|_| die(&format!("{p}: JSON として読めない行があります")));
            if v["schema"].as_u64() != Some(ROW_SCHEMA as u64) {
                die(&format!(
                    "{p}: schema {} は集計できません（現行 {ROW_SCHEMA}）",
                    v["schema"]
                ));
            }
            if v["type"] == "meta" { metas.push(v) } else { rows.push(v) }
        }
        let opp = metas
            .first()
            .and_then(|m| m["experiment"]["opponent"].as_str())
            .unwrap_or_else(|| die(&format!("{p}: meta が無い（相手が分からない）")))
            .to_string();
        if opp.is_empty() {
            die(&format!("{p}: meta の opponent が空です（--opponent 無しで回した記録は層に分けられない）"));
        }
        let e = by_opp.entry(opp).or_insert_with(|| (vec![], vec![]));
        e.0.extend(metas);
        e.1.extend(rows);
    }
    if by_opp.len() < 2 {
        let msg = format!(
            "相手が {} 種類しかありません（opponent-balanced 合算には v13 / v14 の両方が要る）",
            by_opp.len()
        );
        if allow_incomplete { eprintln!("警告: {msg}") } else { die(&msg) }
    }
    // **期待する相手集合を明示的に固定する**（PR #37 レビュー5巡目 [P2]）。
    // 「2種類以上」だけでは任意の2相手・3相手以上でも表示が
    // `(Δv13+Δv14)/2` になってしまう
    let want_opps: BTreeSet<String> = expect_opponents
        .split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();
    if !want_opps.is_empty() {
        let got_opps: BTreeSet<String> = by_opp.keys().cloned().collect();
        if got_opps != want_opps {
            let msg = format!(
                "相手集合が期待と違います: 期待 {:?} / 実際 {:?}（--expect-opponents で変えられる）",
                want_opps, got_opps
            );
            if allow_incomplete { eprintln!("警告: {msg}") } else { die(&msg) }
        }
    }
    // **相手以外の treatment 定義が相手間で同一であること**（同 [P2]）。
    // 契約上「相手だけが違う」ので、片方を別の build / 別の予算 / 別の seed 数で
    // 測った JSONL を平均してはいけない。`opponent` と、相手ごとに必ず違う
    // 記録の指紋（`records`）だけを除いて比較する
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
                    let msg = format!(
                        "相手間で treatment の定義が違います（相手以外は同一でなければならない）: {}",
                        diffs.join(" / ")
                    );
                    if allow_incomplete { eprintln!("警告: {msg}") } else { die(&msg) }
                }
            }
        }
    }
    println!("\n=== P0-2 の最終判定（opponent-balanced 合算）===");
    let main_arm = "oracle@kinf@real";
    let mut strata_r: Vec<Vec<(f64, f64)>> = vec![];
    let mut strata_c: Vec<Vec<(f64, f64)>> = vec![];
    let mut strata_a: Vec<Vec<(f64, f64)>> = vec![];
    let mut strata_f: Vec<Vec<(f64, f64)>> = vec![];
    let mut per_opp: Vec<(String, f64)> = vec![];
    for (opp, (metas, rows)) in &by_opp {
        // 相手ごとに**入力契約を通す**（欠けたシャード・別実験の混入をここで弾く）
        for msg in check_inputs(metas, rows) {
            let m = format!("[{opp}] {msg}");
            if allow_incomplete { eprintln!("警告: {m}") } else { die(&m) }
        }
        let all: Vec<&serde_json::Value> = rows.iter().collect();
        let g = gate_input(&all, main_arm, "current@real");
        let den: f64 = g.reduction.iter().map(|(_, d)| d).sum();
        if den <= 0.0 {
            let m = format!("[{opp}] 主 arm {main_arm} の行がありません");
            if allow_incomplete { eprintln!("警告: {m}"); continue } else { die(&m) }
        }
        let point = g.reduction.iter().map(|(n, _)| n).sum::<f64>() / den;
        let (lo, hi) = cluster_ratio_ci(&g.reduction, 0.05, 0x36_2029);
        println!("  {opp}: R_foul {point:+.4} [{lo:+.4}, {hi:+.4}]（局 {}）", g.reduction.len());
        per_opp.push((opp.clone(), point));
        strata_r.push(g.reduction);
        strata_c.push(g.catastrophe);
        strata_a.push(g.accept);
        strata_f.push(g.foul_limit);
    }
    if strata_r.len() < 2 && !allow_incomplete {
        die("層が2つ揃っていないので合算しません");
    }
    let (r, rlo, rhi) = stratified_mean_ci(&strata_r, 0.05, 0x36_2030);
    let (c, _, chi) = stratified_mean_ci(&strata_c, 0.05, 0x36_2031);
    let (a, alo, _) = stratified_mean_ci(&strata_a, 0.05, 0x36_2032);
    let (f, _, fhi) = stratified_mean_ci(&strata_f, 0.05, 0x36_2033);
    println!("  合算 R_foul（(Δv13+Δv14)/2）: {r:+.4} [{rlo:+.4}, {rhi:+.4}]");
    println!("  破滅率 {c:+.4}（上限 {MARGIN_CATASTROPHE:+.4}、CI 上限 {chi:+.4}）");
    println!("  受理率 {a:+.4}（下限 {MARGIN_ACCEPT:+.4}、CI 下限 {alo:+.4}）");
    println!("  即時反則負け {f:+.4}（上限 {MARGIN_FOUL_LIMIT:+.4}、CI 上限 {fhi:+.4}）");
    let veto = per_opp.iter().all(|(_, p)| *p > 0.0);
    println!(
        "  veto（各相手で R_foul > 0）: {}",
        if veto { "OK" } else { "**NG**" }
    );
    let pass = rlo > 0.0
        && veto
        && c <= MARGIN_CATASTROPHE
        && a >= MARGIN_ACCEPT
        && f <= MARGIN_FOUL_LIMIT;
    println!("\n  **判定: {}**", if pass { "通過（P0-2b へ進む）" } else { "不通過" });
    if !pass && !allow_incomplete {
        // fail-closed: 判定が通らない実験を緑で終わらせない
        std::process::exit(3);
    }
}

fn run_report(args: &[String]) {
    let mut allow_incomplete = false;
    let mut want_shards: Option<usize> = None;
    let mut paths: Vec<String> = vec![];
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--allow-incomplete" => {
                allow_incomplete = true;
                i += 1;
            }
            "--shards" => {
                want_shards = Some(
                    args.get(i + 1)
                        .and_then(|s| s.parse().ok())
                        .unwrap_or_else(|| die("--shards には数値が必要です")),
                );
                i += 2;
            }
            s if s.starts_with("--") => die(&format!("未知のオプション: {s}")),
            s => {
                paths.push(s.to_string());
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
    // `--shards` は**上書きではなく突き合わせ**
    if let Some(want) = want_shards {
        let total = metas
            .first()
            .and_then(|m| m["experiment"]["shard_total"].as_u64())
            .unwrap_or(1) as usize;
        if want != total {
            die(&format!(
                "--shards {want} は JSONL の meta（shard_total {total}）と食い違います"
            ));
        }
    }
    println!("JSONL {} 本 / 行 {}", paths.len(), rows.len());
    report(&metas, &rows, allow_incomplete);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exp(opponent: &str) -> serde_json::Value {
        serde_json::json!({
            "opponent": opponent, "budget_ms": 2000, "eval_particles": 106, "seeds": 2,
            "arms": ["current", "alpha@k2"], "with_real": true, "beta_k": "1",
            "jobs": 4, "shard_total": 1, "config": "c", "source_fingerprint": "s", "records": "r",
        })
    }

    fn meta(e: serde_json::Value, shard: u64) -> serde_json::Value {
        serde_json::json!({
            "schema": ROW_SCHEMA, "type": "meta", "experiment": e, "shard": shard,
            "games": 10, "points": 1,
            "points_detail": [{"game": "g1", "move_number": 41, "estimand": "foul",
                               "type": "nonking_king", "record_fouls": 2,
                               "terminal": false, "weight": 1.0, "double_check": false}],
        })
    }

    fn row(arm: &str, seed: u64, fouls: u32) -> serde_json::Value {
        serde_json::json!({
            "schema": ROW_SCHEMA, "game": "g1", "move_number": 41, "estimand": "foul",
            "type": "nonking_king", "seed": seed, "arm": arm, "fouls": fouls,
            "record_fouls": 2, "foul_limit": false, "accepted": "5i4h",
            "truth_accepted": true, "mated_in_1": false, "next_check": false,
            "material": 0.0, "updates": 1, "sim_us": 100, "repro_err": 0.0,
            // schema 2 の必須列（issue #36 P0-2）
            "identity_err": 0.0, "deduce": serde_json::Value::Null,
            "oracle": serde_json::Value::Null, "double_check": false,
            "identity_err_real": 0.0, "identity_only_real": 0,
            "terminal": false, "weight": 1.0,
            // schema 5 の必須列（実行順の監査）
            "arm_order": 0,
        })
    }

    /// 実行順つきの行（AB/BA の均衡検査を通すため、実再決定 arm は seed の
    /// 偶奇で `current@real` との前後を入れ替える）
    fn row_ord(arm: &str, seed: u64, fouls: u32, order: usize) -> serde_json::Value {
        let mut r = row(arm, seed, fouls);
        r["arm_order"] = serde_json::json!(order);
        r
    }

    #[test]
    fn 実行順が固定なら均衡していないと分かる() {
        // 全 arm が同じ順（= レビュー5巡目 [P1] の修正前の姿）だと、
        // `oracle@kinf@real` は常に `current@real` の後ろに来る
        let mut rows = full_oracle();
        for r in &mut rows {
            if r["arm"] == "oracle@kinf@real" {
                r["arm_order"] = serde_json::json!(9);
            }
            if r["arm"] == "current@real" {
                r["arm_order"] = serde_json::json!(4);
            }
        }
        let problems = check_inputs(&[meta(oracle_exp(), 0)], &rows);
        assert!(
            problems.iter().any(|m| m.contains("均衡していません")),
            "固定順は検出されるべき: {problems:?}"
        );
    }

    #[test]
    fn metaが宣言していない行は拒否する() {
        // 未宣言の決定点を全 arm ぶん足すと、自然頻度 gate は読むのに
        // 欠落検査は素通りする（修正前の穴）
        let mut rows = full_oracle();
        let mut extra = row_ord("current@real", 0, 9, 4);
        extra["move_number"] = serde_json::json!(99);
        rows.push(extra);
        let problems = check_inputs(&[meta(oracle_exp(), 0)], &rows);
        assert!(
            problems.iter().any(|m| m.contains("meta が宣言していない行")),
            "未宣言の行は拒否されるべき: {problems:?}"
        );
    }

    #[test]
    fn 範囲外のseedの行も拒否する() {
        let mut rows = full_oracle();
        rows.push(row_ord("current@real", 7, 9, 4));
        let problems = check_inputs(&[meta(oracle_exp(), 0)], &rows);
        assert!(
            problems.iter().any(|m| m.contains("meta が宣言していない行")),
            "範囲外 seed は拒否されるべき: {problems:?}"
        );
    }

    #[test]
    fn 行の重みがmetaと違えば拒否する() {
        // `weight` は自然頻度の分母を決めるので、片方だけ書き換えて
        // 主 CI を動かせてはいけない
        let mut rows = full_oracle();
        rows[0]["weight"] = serde_json::json!(7.0);
        let problems = check_inputs(&[meta(oracle_exp(), 0)], &rows);
        assert!(
            problems.iter().any(|m| m.contains("点属性")),
            "meta と食い違う weight は拒否されるべき: {problems:?}"
        );
    }

    #[test]
    fn 終端フラグがmetaと違えば拒否する() {
        let mut rows = full_oracle();
        rows[0]["terminal"] = serde_json::json!(true);
        let problems = check_inputs(&[meta(oracle_exp(), 0)], &rows);
        assert!(
            problems.iter().any(|m| m.contains("点属性")),
            "meta と食い違う terminal は拒否されるべき: {problems:?}"
        );
    }

    #[test]
    fn no_realは実再決定armを1本も残さない() {
        // 既定の `beliefs_real`（`oracle@kinf`）が残っていると、対照も AB/BA も
        // 無い壁時計ベースの real arm が unit ごとに回る（レビュー7巡目 [P2]）
        let mut br = vec![Belief::parse("oracle@kinf").unwrap()];
        assert_eq!(resolve_real_arms(false, &mut br, false), Ok(false));
        assert!(br.is_empty(), "--no-real なら beliefs_real も空になるべき");
        // その結果、奇数 seed の検査対象からも外れる（shadow は決定論的）
        let mut br2 = vec![Belief::parse("oracle@kinf").unwrap()];
        assert_eq!(resolve_real_arms(true, &mut br2, false), Ok(true));
        assert_eq!(br2.len(), 1);
        // 明示指定との併用は矛盾なので起動時に落とす
        let mut br3 = vec![Belief::parse("oracle@kinf").unwrap()];
        assert!(resolve_real_arms(false, &mut br3, true).is_err());
        // `--belief-real` を明示していても空なら矛盾しない
        let mut br4: Vec<Belief> = vec![];
        assert_eq!(resolve_real_arms(false, &mut br4, true), Ok(false));
    }

    #[test]
    fn 両王手フラグがmetaと違えば拒否する() {
        // `double_check` は主 estimand の**分母**を決める（両王手は介入が no-op なので
        // 除く）。全 arm の行で `false` へ書き換えると、除外されていた手番が分母へ
        // 戻って主 CI が動く（PR #37 レビュー6巡目 [P1]）
        let mut m = meta(oracle_exp(), 0);
        m["points_detail"][0]["double_check"] = serde_json::json!(true);
        let problems = check_inputs(&[m], &full_oracle());
        assert!(
            problems.iter().any(|p| p.contains("点属性")),
            "meta と食い違う double_check は拒否されるべき: {problems:?}"
        );
    }

    #[test]
    fn metaに両王手の宣言が無ければ拒否する() {
        // schema を上げずに列だけ落とされたときに既定値で埋めない
        let mut m = meta(oracle_exp(), 0);
        m["points_detail"][0]
            .as_object_mut()
            .unwrap()
            .remove("double_check");
        let problems = check_inputs(&[m], &full_oracle());
        assert!(
            problems.iter().any(|p| p.contains("double_check")),
            "meta の欠測は拒否されるべき: {problems:?}"
        );
    }

    #[test]
    fn 実行順が決定点をまたいで相殺されても検出する() {
        // 決定点 A は全 seed が treatment 先・決定点 B は全 seed が対照 先。
        // **総数は釣り合う**ので全体集計の検査は通ってしまうが、決定点ごとの
        // 実行順効果はペア差に残る（PR #37 レビュー6巡目 [P1]）
        let mut m = meta(oracle_exp(), 0);
        let d1 = m["points_detail"][0].clone();
        let mut d2 = d1.clone();
        d2["move_number"] = serde_json::json!(51);
        m["points_detail"] = serde_json::json!([d1, d2]);
        m["points"] = serde_json::json!(2);

        let mut rows = vec![];
        for (mn, treat_first) in [(41u64, true), (51u64, false)] {
            for seed in 0..2u64 {
                for (i, arm) in
                    ["current@static", "current@shadow", "alpha@k2@shadow", "oracle@k1@shadow"]
                        .iter()
                        .enumerate()
                {
                    let mut r = row_ord(arm, seed, 1, i);
                    r["move_number"] = serde_json::json!(mn);
                    rows.push(r);
                }
                let block: [&str; 3] = if treat_first {
                    ["oracle@k1@real", "oracle@kinf@real", "current@real"]
                } else {
                    ["oracle@k1@real", "current@real", "oracle@kinf@real"]
                };
                for (i, arm) in block.iter().enumerate() {
                    let mut r = row_ord(arm, seed, 1, 4 + i);
                    r["move_number"] = serde_json::json!(mn);
                    rows.push(r);
                }
            }
        }
        let problems = check_inputs(&[m], &rows);
        assert!(
            problems
                .iter()
                .any(|p| p.contains("oracle@kinf@real") && p.contains("決定点の中で均衡していません")),
            "決定点ごとの偏りは検出されるべき: {problems:?}"
        );
    }

    fn full() -> Vec<serde_json::Value> {
        let mut v = vec![];
        for seed in 0..2 {
            for (i, arm) in
                ["current@static", "current@shadow", "alpha@k2@shadow", "current@real"]
                    .iter()
                    .enumerate()
            {
                v.push(row_ord(arm, seed, 1, i));
            }
        }
        v
    }

    /// オラクル arm 一式（恒等対照・実再決定・その床）を含む行。
    /// **実再決定 arm は seed の偶奇で `current@real` との前後を入れ替える**
    /// （本番の AB/BA と同じ形。均衡検査を通るのはこの形だけ）
    fn full_oracle() -> Vec<serde_json::Value> {
        let mut v = vec![];
        for seed in 0..2 {
            for (i, arm) in ["current@static", "current@shadow", "alpha@k2@shadow", "oracle@k1@shadow"]
                .iter()
                .enumerate()
            {
                v.push(row_ord(arm, seed, 1, i));
            }
            // 実再決定ブロック: 偶数 seed は [k1, current, kinf]、奇数は反転
            let block: [&str; 3] = if seed % 2 == 0 {
                ["oracle@k1@real", "current@real", "oracle@kinf@real"]
            } else {
                ["oracle@kinf@real", "current@real", "oracle@k1@real"]
            };
            for (i, arm) in block.iter().enumerate() {
                v.push(row_ord(arm, seed, 1, 4 + i));
            }
        }
        v
    }

    fn oracle_exp() -> serde_json::Value {
        let mut e = exp("estimator_v14");
        e["beliefs"] = serde_json::json!(["oracle@k1"]);
        e["beliefs_real"] = serde_json::json!(["oracle@kinf"]);
        e
    }

    /// 自然頻度の cluster を作る最小の行（`weight` と `double_check` を効かせる）
    fn nat_row(game: &str, mn: u64, arm: &str, fouls: f64, w: f64, dbl: bool) -> serde_json::Value {
        serde_json::json!({
            "schema": ROW_SCHEMA, "game": game, "move_number": mn, "seed": 0,
            "arm": arm, "fouls": fouls, "weight": w, "double_check": dbl,
            "truth_accepted": true, "foul_limit": false,
        })
    }

    #[test]
    fn 自然頻度は包含重みで戻し両王手を除く() {
        // 間引いた反則0の手番は `weight` で元の層頻度へ戻す。両王手は介入が
        // no-op なので分母に入れない（PR #37 レビュー4巡目 [P1]）
        let rows = vec![
            // 反則あり（weight 1）: arm は 1 少ない
            nat_row("g1", 10, "current@real", 3.0, 1.0, false),
            nat_row("g1", 10, "oracle@kinf@real", 2.0, 1.0, false),
            // 反則0を 3 本のうち 1 本に間引いた（weight 3）: 差 0
            nat_row("g1", 20, "current@real", 0.0, 3.0, false),
            nat_row("g1", 20, "oracle@kinf@real", 0.0, 3.0, false),
            // 両王手（除外されるので分母に入らない）
            nat_row("g1", 30, "current@real", 9.0, 1.0, true),
            nat_row("g1", 30, "oracle@kinf@real", 0.0, 1.0, true),
        ];
        let sel: Vec<&serde_json::Value> = rows.iter().collect();
        let c = natural_clusters(&sel, "oracle@kinf@real", "current@real", &|r| {
            r["fouls"].as_f64().unwrap_or(0.0)
        });
        let (num, den) = c["g1"];
        assert!((den - 4.0).abs() < 1e-9, "重み 1 + 3（両王手は除く）: {den}");
        assert!((num - (-1.0)).abs() < 1e-9, "差 −1 × 重み 1: {num}");
        // 均衡標本のまま合算すると −0.5 に見えるが、自然頻度では −0.25
        assert!((num / den - (-0.25)).abs() < 1e-9);
    }

    #[test]
    fn 合算は相手ごとの層の平均になる() {
        // `(Δv13 + Δv14) / 2`。層の内側で元対局を引き直す
        let a = vec![(-1.0, 1.0), (-1.0, 1.0)];
        let b = vec![(1.0, 1.0), (1.0, 1.0)];
        let (p, lo, hi) = stratified_mean_ci(&[a, b], 0.05, 7);
        assert!(p.abs() < 1e-9, "+1 と −1 の平均は 0: {p}");
        assert!(lo <= p && p <= hi);
        // 片方の層が空なら判定不能（NaN）にする
        let (p2, _, _) = stratified_mean_ci(&[vec![(-1.0, 1.0)], vec![]], 0.05, 7);
        assert!(p2.is_nan());
    }

    #[test]
    fn deduceの健全性はdeduce_armの行から数える() {
        // **診断は arm ごとの行に入る**ので、`current@static` で絞ると常に空振りし、
        // 「真仮説を落としたら中止」が発動しない（PR #37 レビュー3巡目 [P1]）
        let mut rows = full();
        assert_eq!(deduce_summary(&rows), None, "診断が無ければ None");
        for seed in 0..2 {
            let mut r = row("deduce_last_move@shadow", seed, 1);
            r["deduce"] = serde_json::json!({
                "dropped": 3, "fallback": false, "dropped_true": seed == 0,
                "constructions": 2,
            });
            rows.push(r);
        }
        let d = deduce_summary(&rows).expect("deduce arm の行から集める");
        assert_eq!(d.rows, 2);
        assert_eq!(d.dropped_true, 1, "真仮説を落とした行を数える = 中止条件");
        assert!((d.dropped_mean - 3.0).abs() < 1e-9);
    }

    #[test]
    fn 実再決定armの対照はcurrent_real() {
        // shadow を対照にすると介入に「shadow → 実再決定」の差が混ざる
        assert_eq!(baseline_for("oracle@kinf@real"), "current@real");
        assert_eq!(baseline_for("oracle@k1@real"), "current@real");
        assert_eq!(baseline_for("alpha@k2@shadow"), "current@shadow");
        assert_eq!(baseline_for("current@static"), "current@shadow");
    }

    #[test]
    fn pの差は添字でなく指し手で突き合わせる() {
        // 実再決定は候補リストごと作り直すので、順位が入れ替わると添字 zip は
        // 別の手の p を比べる（PR #37 レビュー3巡目 [P2]）
        let a = vec![("5i5h".to_string(), 0.9), ("5i4h".to_string(), 0.4)];
        let b = vec![("5i4h".to_string(), 0.4), ("5i5h".to_string(), 0.9)];
        assert_eq!(p_gap(&a, &b), (0.0, 0), "並びが違っても同じ手同士なら差 0");
        let c = vec![("5i4h".to_string(), 0.4), ("5i6h".to_string(), 0.9)];
        let (d, only) = p_gap(&a, &c);
        assert!((d - 0.0).abs() < 1e-9, "共通候補は 5i4h だけ");
        assert_eq!(only, 2, "片側にしか無い候補を数える");
    }

    #[test]
    fn 実再決定のオラクルには恒等対照とノイズ床が要る() {
        // `current@real` と同じ再ランキング経路を通る `oracle@k1@real` が無いと、
        // 主 arm の差を再決定のノイズ床と並べて読めない（PR #37 レビュー2巡目 [P1]）
        assert!(
            check_inputs(&[meta(oracle_exp(), 0)], &full_oracle()).is_empty(),
            "揃っていれば通る"
        );
        let mut rows = full_oracle();
        rows.retain(|r| r["arm"] != "oracle@k1@real");
        let problems = check_inputs(&[meta(oracle_exp(), 0)], &rows);
        assert!(
            problems.iter().any(|p| p.contains("oracle@k1@real")),
            "{problems:?}"
        );
    }

    #[test]
    fn ノイズ床が測れていない実験は集計させない() {
        let mut rows = full_oracle();
        for r in &mut rows {
            r["identity_err_real"] = serde_json::Value::Null;
        }
        let problems = check_inputs(&[meta(oracle_exp(), 0)], &rows);
        assert!(
            problems.iter().any(|p| p.contains("identity_err_real")),
            "{problems:?}"
        );
    }

    #[test]
    fn 揃った入力は契約を通る() {
        assert!(check_inputs(&[meta(exp("estimator_v14"), 0)], &full()).is_empty());
    }

    #[test]
    fn 必須列が欠けた行は集計させない() {
        // **欠けた列を既定値で埋めて門を通せてはいけない**: `identity_err` が
        // 無い行を「差 0」と読むと恒等対照の関門が素通りする（issue #36）
        let mut rows = full();
        for r in &mut rows {
            r.as_object_mut().unwrap().remove("identity_err");
        }
        let problems = check_inputs(&[meta(exp("estimator_v14"), 0)], &rows);
        assert!(
            problems.iter().any(|p| p.contains("identity_err")),
            "{problems:?}"
        );
    }

    #[test]
    fn beliefのarmも完全性検査の対象になる() {
        // `--belief` で足した arm の行が欠けたら失敗する（meta の宣言が根拠）
        let mut e = exp("estimator_v14");
        e["beliefs"] = serde_json::json!(["oracle@k1"]);
        e["beliefs_real"] = serde_json::json!(["oracle@kinf"]);
        let problems = check_inputs(&[meta(e, 0)], &full());
        assert!(
            problems.iter().any(|p| p.contains("oracle@k1@shadow")),
            "{problems:?}"
        );
    }

    #[test]
    fn 恒等対照なしのオラクル実験は集計させない() {
        // **`--belief` から `oracle@k1` を抜くと `identity_err` が null のままになり、
        // 関門が「数値が0件」で素通りする**（PR #37 レビュー [P2]）
        let mut e = exp("estimator_v14");
        e["beliefs"] = serde_json::json!(["oracle@kinf"]);
        e["beliefs_real"] = serde_json::json!([]);
        let problems = check_inputs(&[meta(e, 0)], &full());
        assert!(
            problems.iter().any(|p| p.contains("oracle@k1")),
            "{problems:?}"
        );
    }

    #[test]
    fn identity_errが全行nullなら集計させない() {
        let mut e = exp("estimator_v14");
        e["beliefs"] = serde_json::json!(["oracle@k1"]);
        let mut rows = full();
        for r in &mut rows {
            r["identity_err"] = serde_json::Value::Null;
        }
        // 恒等対照 arm の行も足しておく（欠測の指摘と区別するため）
        for seed in 0..2 {
            let mut x = row("oracle@k1@shadow", seed, 1);
            x["identity_err"] = serde_json::Value::Null;
            rows.push(x);
        }
        let problems = check_inputs(&[meta(e, 0)], &rows);
        assert!(
            problems.iter().any(|p| p.contains("identity_err が全行 null")),
            "{problems:?}"
        );
    }

    #[test]
    fn 実験キーが違うjsonlは混ぜられない() {
        let metas = vec![meta(exp("estimator_v14"), 0), meta(exp("estimator_v13"), 1)];
        let problems = check_inputs(&metas, &full());
        assert!(
            problems.iter().any(|p| p.contains("opponent")),
            "{problems:?}"
        );
    }

    #[test]
    fn armごと欠けた行を検出する() {
        // ある arm の seed が丸ごと欠けると Δ の分母が狂う
        let rows: Vec<serde_json::Value> = full()
            .into_iter()
            .filter(|r| !(r["arm"] == "alpha@k2@shadow" && r["seed"] == 1))
            .collect();
        let problems = check_inputs(&[meta(exp("estimator_v14"), 0)], &rows);
        assert!(
            problems.iter().any(|p| p.contains("alpha@k2@shadow")),
            "{problems:?}"
        );
    }

    /// ランキングが取れなかった決定点は meta の `dropped` に残し、期待から外す。
    /// **記録に無い欠落は今までどおり失敗させる**（黙って行が消えるのと区別する）
    #[test]
    fn 落とした決定点は期待から外れる() {
        let mut m = meta(exp("estimator_v14"), 0);
        // 行が1つも無い状態は、dropped に載っていなければ失敗
        assert!(!check_inputs(&[m.clone()], &[]).is_empty());
        m["dropped"] = serde_json::json!([{"game": "g1", "move_number": 41}]);
        assert!(
            check_inputs(&[m.clone()], &[]).is_empty(),
            "dropped に載っていれば行が無くても通る"
        );
        // 別の決定点の欠落は依然として失敗する
        m["dropped"] = serde_json::json!([{"game": "g1", "move_number": 99}]);
        assert!(!check_inputs(&[m], &[]).is_empty());
    }

    #[test]
    fn シャードが欠けたら判定を出さない() {
        let mut e = exp("estimator_v14");
        e["shard_total"] = serde_json::json!(2);
        let problems = check_inputs(&[meta(e, 0)], &full());
        assert!(
            problems.iter().any(|p| p.contains("シャードが欠けて")),
            "{problems:?}"
        );
    }

    #[test]
    fn ペア差は元対局単位で畳む() {
        // 同じ局の2 seed は1つの cluster。arm が片方しか無い unit は落とす
        let rows = full();
        let refs: Vec<&serde_json::Value> = rows.iter().collect();
        let (d, lo, hi) = paired_ci(&refs, "alpha@k2@shadow", "current@shadow", &|r| {
            r["fouls"].as_f64().unwrap_or(0.0)
        });
        assert_eq!(d, 0.0, "同じ反則数ならペア差は 0");
        assert_eq!((lo, hi), (0.0, 0.0));
    }

    /// **近似の門は符号付き平均では測れない**（PR #33 レビュー [P1]）。
    /// 「ある unit で +1、別の unit で −1」ずれる仮想更新は符号付きでは
    /// 誤差 0 に見えるが、unit ごとの絶対誤差なら 1.0 として現れる
    #[test]
    fn 近似誤差は符号付き平均でなくunitごとの絶対値で測る() {
        let mut rows = vec![];
        // 決定点 g1#41: shadow が real より +1 反則 / g2#41: −1 反則
        for (game, shadow, real) in [("g1", 2u32, 1u32), ("g2", 0, 1)] {
            for seed in 0..2 {
                let mut a = row("current@shadow", seed, shadow);
                a["game"] = serde_json::json!(game);
                let mut b = row("current@real", seed, real);
                b["game"] = serde_json::json!(game);
                rows.push(a);
                rows.push(b);
            }
        }
        let refs: Vec<&serde_json::Value> = rows.iter().collect();
        let fouls = |r: &serde_json::Value| r["fouls"].as_f64().unwrap_or(0.0);
        let (signed, _, _) = paired_ci(&refs, "current@shadow", "current@real", &fouls);
        let (abs, _, _) = paired_abs_ci(&refs, "current@shadow", "current@real", &fouls);
        assert!(signed.abs() < 1e-9, "符号付きでは相殺して 0 に見える: {signed}");
        assert!((abs - 1.0).abs() < 1e-9, "絶対誤差は 1.0: {abs}");
    }
}
