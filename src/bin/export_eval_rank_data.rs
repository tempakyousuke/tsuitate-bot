//! **人手採点 × 候補手特徴のエクスポータ**（issue #24 P0）。
//!
//! `evals/*.eval.md` の各ブロックを元 KIF の決定点（ply と反則後サブ状態）へ
//! 復元し、現行 estimator の候補ランキング内訳と結合して CSV を書く。
//! 1行 = `(source_kif, decision_state, seed, usi, human_score, features...)`。
//!
//! usage:
//!   cargo run --release --bin export_eval_rank_data -- \
//!     [--out data/eval_rank.csv] [--seeds 4] [--budget-ms 2000] [--jobs N] \
//!     [evals/xxx.eval.md ...]
//!
//! 省略時は `evals/*.eval.md` 全部・seed 0..3。**重い**（決定状態 × seed
//! ぶんの `choose`。同一棋譜・同一手番側は prewarm 済み戦略を ply 昇順で
//! 継ぎ足して共有する = `bin/rank_dump` / `choice_trials_batch` と同じ規約）。
//!
//! 出力の性質（issue #24 の非目的をコードで守る）:
//!
//! - 特徴量は `PlayerView` と `CandidateScore` だけから作る（`eval_rank::feature_row`）。
//!   真実盤面・実戦の正解手・コメント文・棋譜名は入らない
//! - 未採点（`?`）は `human_score` 空欄の**欠測**として出す（悪手として数えない）
//! - タイブレーク乱数はどの特徴量にも載らない（`score` / `static_score` / `adjust`
//!   から引き、順位系は乱数を除いたスコアで付け直す）。乱数込みの**実際のスコアと
//!   順位**は識別子列 `engine_score` / `engine_rank` に出す: 現行方策の baseline の
//!   再現と、P1 の統合形（`gain' = gain + 残差` → `combine_score(...)`）の数値ベースに
//!   要る（順位番号へ残差を足すと候補間の実スコア差が消えて別物の検証になる）
//! - eval の採点と、対応するシナリオ kif の `scores=` が一致しない場合は停止する

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tsuitate_bot::config::{EnvSource, StrategyConfig};
use tsuitate_bot::eval_rank::{
    EvalBlock, FEATURE_COLUMNS, ScenarioIndex, SetContext, det_order, feature_row, parse_eval,
    real_fouls_at, source_kif_path,
};
use tsuitate_bot::kifu::{Kifu, parse_kif};
use tsuitate_bot::observation::{Observation, ObservationLog};
use tsuitate_bot::protocol::Color;
use tsuitate_bot::scenario_core::{clone_log, make_view, replay, side_idx};
use tsuitate_bot::shogi::parse_usi;
use tsuitate_bot::strategy::{self, Strategy};

/// **実行時データの指紋**（PR #25 レビュー指摘 P1）。
///
/// コード版と実効ビルド条件は `build.rs` が焼き込む `TSUITATE_SOURCE_FINGERPRINT`。
/// こちらは**実行時に読むデータ**: 解決済みの定跡パス（`config.joseki_path` =
/// `TSUITATE_JOSEKI`。`config_fingerprint` にはパス文字列しか入らない）と、
/// 今回使った元 KIF。
///
/// **ディスクを読み直さない**: 元 KIF は `parse_kif` に渡した**その bytes**を
/// そのまま受け取る。長い実行の最後にファイルを読み直すと、A を読んで走った後に
/// 同じパスを B へ差し替えられたとき、**CSV は A 由来なのに B の指紋が付く**
/// （そして次の B 由来 export と検査を通ってしまう）。
/// 定跡は `opening.rs` が遅延ロードしてプロセス内にキャッシュするので bytes を
/// 取り出せない。代わりに**開始時と終了時に hash して不一致なら run を失敗させる**。
fn data_fingerprint(joseki_path: &str, joseki_bytes: &[u8], source_kifs: &[(PathBuf, String)]) -> String {
    use sha2::Digest as _;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut h = sha2::Sha256::new();
    h.update(b"joseki\0");
    h.update(joseki_path.as_bytes());
    h.update([0]);
    h.update(joseki_bytes);
    h.update([0]);
    let mut kifs: Vec<&(PathBuf, String)> = source_kifs.iter().collect();
    kifs.sort();
    kifs.dedup_by(|a, b| a.0 == b.0);
    for (path, text) in kifs {
        h.update(path.strip_prefix(root).unwrap_or(path).to_string_lossy().as_bytes());
        h.update([0]);
        h.update(text.as_bytes());
        h.update([0]);
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// 定跡ファイルの hash（開始時と終了時に取って突き合わせる）
fn joseki_bytes(path: &str) -> Vec<u8> {
    std::fs::read(path).unwrap_or_default()
}

/// 1決定状態（eval の1ブロック）の復元結果
struct State {
    /// 見出しの N手目
    num: usize,
    ply: usize,
    /// 注入する反則列（通常ブロックは空）
    fouls: Vec<String>,
    /// 対応するシナリオ名（無ければ空文字）
    scenario: String,
    /// eval の採点（未採点は None）
    scores: HashMap<String, Option<u8>>,
    /// 採点済みの件数
    n_scored: usize,
}

impl State {
    fn id(&self, stem: &str) -> String {
        if self.fouls.is_empty() {
            format!("{stem}@{}", self.ply)
        } else {
            format!("{stem}@{}+f{}", self.ply, self.fouls.len())
        }
    }
}

/// 1 eval ファイルぶんの復元済み仕事
struct Job {
    stem: String,
    kifu: Arc<Kifu>,
    states: Vec<State>,
}

struct Row {
    id: String,
    line: String,
}

/// 集計（summary JSON 用）
#[derive(Default)]
struct Summary {
    states: usize,
    states_with_scenario: usize,
    scored_entries: usize,
    unscored_entries: usize,
    rows: usize,
    /// 採点されているのに現行候補集合に無い手（seed ごとに数える）
    scored_not_in_candidates: usize,
    /// 現行候補集合にあるが eval に載っていない手（seed ごとに数える）
    candidates_not_in_eval: usize,
    /// ランキングが取れなかった決定状態（定跡・候補ゼロ）
    no_ranking: usize,
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let take_opt = |args: &mut Vec<String>, name: &str| -> Option<String> {
        args.iter().position(|a| a == name).map(|i| {
            let v = args
                .get(i + 1)
                .unwrap_or_else(|| panic!("{name} には値が必要です"))
                .clone();
            args.drain(i..i + 2);
            v
        })
    };
    let out = take_opt(&mut args, "--out").unwrap_or_else(|| "data/eval_rank.csv".into());
    // summary の場所は `--out` から一意に決まる（PR #25 レビュー指摘 P2）。
    // 分析側は CSV 名から導出するので、別名を許すと「正しく作った組を渡したのに
    // 存在しないパスを探して止まる」ことになる
    if args.iter().any(|a| a == "--summary") {
        eprintln!("--summary は廃止しました（summary は --out と同じ名前の .summary.json に出ます）");
        std::process::exit(2);
    }
    // **起動時に検査する**: `.csv` で終わらないと下の置換が `out` をそのまま返し、
    // 重いエクスポートの最後に summary が CSV を上書きして結果が消える
    let Some(stem) = out.strip_suffix(".csv") else {
        eprintln!("--out は `.csv` で終わる必要があります（summary を同名の .summary.json に出すため）: {out}");
        std::process::exit(2);
    };
    let summary_path = format!("{stem}.summary.json");
    let seeds: u64 = take_opt(&mut args, "--seeds")
        .map_or(4, |v| v.parse().expect("--seeds は整数"));
    let budget_ms: Option<u64> = take_opt(&mut args, "--budget-ms")
        .map(|v| v.parse().expect("--budget-ms は整数"));
    let jobs: usize = take_opt(&mut args, "--jobs").map_or_else(
        || std::thread::available_parallelism().map_or(1, |n| n.get()),
        |v| v.parse().expect("--jobs は整数"),
    );

    let eval_paths: Vec<PathBuf> = if args.is_empty() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("evals");
        let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
            .expect("evals/ を読めません")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.to_string_lossy().ends_with(".eval.md"))
            .collect();
        v.sort();
        v
    } else {
        args.iter().map(PathBuf::from).collect()
    };

    // **設定境界**（issue #21）: env はここで一度だけ解釈し、以後は config で渡す。
    let source = match budget_ms {
        Some(ms) => EnvSource::from_process()
            .with_overrides([("TSUITATE_CAND_THINK_BUDGET_MS", ms.to_string())]),
        None => EnvSource::from_process(),
    };
    let config = Arc::new(StrategyConfig::from_source(source));
    // **debug ビルドで計測させない**（PR #25 レビュー指摘 P1）。思考予算は壁時計
    // なので、同じ 2000ms でも debug では探索量が桁で違い、replicate として
    // 混ぜられない。ビルド条件は `source_fingerprint` にも入っているが、
    // そもそも取らせないほうが早い
    if env!("TSUITATE_BUILD_PROFILE") != "release" {
        eprintln!(
            "release ビルドで実行してください（現在: {}）。思考予算が壁時計なので\
             debug ビルドの計測は replicate として使えません",
            env!("TSUITATE_BUILD_PROFILE")
        );
        std::process::exit(2);
    }
    // 定跡は `opening.rs` が遅延ロードしてキャッシュするので bytes を取り出せない。
    // **開始時に取っておき、終了時に取り直して不一致なら run を失敗させる**
    let joseki_at_start = joseki_bytes(&config.joseki_path);
    eprintln!(
        "思考予算 {}ms / seeds {seeds} / jobs {jobs} / config {}",
        config.think_budget_ms,
        &config.fingerprint()[..12]
    );

    let mut summary = Summary::default();
    let mut jobs_list: Vec<Job> = vec![];
    // 教師（eval）の指紋。replicate をまたいで採点が変わっていないことの検査用
    let mut eval_hash = {
        use sha2::Digest as _;
        sha2::Sha256::new()
    };
    // 元 KIF は「parse に渡した bytes」をそのまま指紋へ回す（読み直さない）
    let mut source_kifs: Vec<(PathBuf, String)> = vec![];
    for path in &eval_paths {
        let stem = path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .replace(".eval.md", "");
        let src = source_kif_path(&stem)
            .unwrap_or_else(|| panic!("{stem}: 元 KIF（scenarios/archive/{stem}.kif）がありません"));
        let kif_text = std::fs::read_to_string(&src).expect("kif を読めません");
        let kifu = parse_kif(&kif_text).unwrap_or_else(|e| panic!("{}: {e}", src.display()));
        source_kifs.push((src.clone(), kif_text));
        let index = ScenarioIndex::build(&kifu);
        let blocks = parse_eval(path).unwrap_or_else(|e| panic!("{e}"));
        {
            use sha2::Digest as _;
            eval_hash.update(stem.as_bytes());
            for b in &blocks {
                eval_hash.update(b.num.to_le_bytes());
                eval_hash.update(b.sub.as_deref().unwrap_or("").as_bytes());
                for (u, p) in &b.entries {
                    eval_hash.update(u.as_bytes());
                    eval_hash.update([p.unwrap_or(255)]);
                }
            }
        }
        let mut states = vec![];
        for block in &blocks {
            let st = restore(&stem, &kifu, &index, block);
            summary.states += 1;
            if !st.scenario.is_empty() {
                summary.states_with_scenario += 1;
            }
            summary.scored_entries += st.n_scored;
            summary.unscored_entries += st.scores.len() - st.n_scored;
            states.push(st);
        }
        eprintln!(
            "{stem}: {} 決定状態（シナリオ照合 {} 件）",
            states.len(),
            states.iter().filter(|s| !s.scenario.is_empty()).count()
        );
        jobs_list.push(Job { stem, kifu: Arc::new(kifu), states });
    }

    // (eval, seed) を単位に並列化する。同一 eval・同一 seed の中では
    // prewarm 済み戦略を ply 昇順で継ぎ足して共有する
    let units: Vec<(usize, u64)> = (0..jobs_list.len())
        .flat_map(|i| (0..seeds).map(move |s| (i, s)))
        .collect();
    let next = Arc::new(Mutex::new(0usize));
    let out_rows: Arc<Mutex<Vec<Row>>> = Arc::new(Mutex::new(vec![]));
    let stats = Arc::new(Mutex::new((0usize, 0usize, 0usize, 0usize)));
    let jobs_list = Arc::new(jobs_list);
    let units = Arc::new(units);
    std::thread::scope(|scope| {
        for _ in 0..jobs.max(1) {
            let next = Arc::clone(&next);
            let out_rows = Arc::clone(&out_rows);
            let stats = Arc::clone(&stats);
            let jobs_list = Arc::clone(&jobs_list);
            let units = Arc::clone(&units);
            let config = Arc::clone(&config);
            scope.spawn(move || {
                loop {
                    let idx = {
                        let mut n = next.lock().unwrap();
                        if *n >= units.len() {
                            return;
                        }
                        let i = *n;
                        *n += 1;
                        i
                    };
                    let (ji, seed) = units[idx];
                    let job = &jobs_list[ji];
                    let (rows, s) = run_unit(job, seed, &config);
                    {
                        let mut acc = stats.lock().unwrap();
                        acc.0 += s.0;
                        acc.1 += s.1;
                        acc.2 += s.2;
                        acc.3 += rows.len();
                    }
                    out_rows.lock().unwrap().extend(rows);
                    eprintln!(
                        "[{}/{}] {} seed={seed} 完了",
                        idx + 1,
                        units.len(),
                        job.stem
                    );
                }
            });
        }
    });

    let mut rows = Arc::try_unwrap(out_rows).ok().unwrap().into_inner().unwrap();
    rows.sort_by(|a, b| a.id.cmp(&b.id));
    let acc = *stats.lock().unwrap();
    summary.scored_not_in_candidates = acc.0;
    summary.candidates_not_in_eval = acc.1;
    summary.no_ranking = acc.2;
    summary.rows = acc.3;

    if let Some(dir) = Path::new(&out).parent() {
        std::fs::create_dir_all(dir).expect("出力ディレクトリを作れません");
    }
    let mut csv = String::new();
    csv.push_str(
        "source_kif,decision_state,scenario,ply,side,seed,usi,human_score,in_candidates,\
engine_rank,engine_score,",
    );
    csv.push_str(&FEATURE_COLUMNS.join(","));
    csv.push('\n');
    for r in &rows {
        csv.push_str(&r.line);
        csv.push('\n');
    }
    std::fs::write(&out, csv).expect("CSV を書けません");

    let per_source = {
        let mut m: HashMap<&str, usize> = HashMap::new();
        for r in &rows {
            *m.entry(r.line.split(',').next().unwrap()).or_default() += 1;
        }
        let mut v: Vec<(&str, usize)> = m.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    };
    let total: usize = per_source.iter().map(|(_, n)| n).sum();
    let mut js = String::from("{\n");
    js.push_str(&format!("  \"budget_ms\": {},\n", config.think_budget_ms));
    js.push_str(&format!("  \"config_fingerprint\": \"{}\",\n", config.fingerprint()));
    js.push_str(&format!("  \"seeds\": {seeds},\n"));
    // コード版は build.rs がコンパイル時に焼き込んだ値（実行時に worktree を
    // 読むと、別コミットでビルドしたバイナリを今の worktree の指紋で記録してしまう）
    js.push_str(&format!(
        "  \"source_fingerprint\": \"{}\",\n",
        env!("TSUITATE_SOURCE_FINGERPRINT")
    ));
    // 実行中に定跡が差し替わっていないか（CSV は開始時の内容で走っている）
    if joseki_bytes(&config.joseki_path) != joseki_at_start {
        eprintln!(
            "定跡 {} が実行中に変わりました。この CSV がどの定跡で走ったか確定できないので中止します",
            config.joseki_path
        );
        std::process::exit(1);
    }
    js.push_str(&format!(
        "  \"data_fingerprint\": \"{}\",\n",
        data_fingerprint(&config.joseki_path, &joseki_at_start, &source_kifs)
    ));
    js.push_str(&format!("  \"eval_fingerprint\": \"{}\",\n", {
        use sha2::Digest as _;
        eval_hash
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    }));
    js.push_str(&format!("  \"decision_states\": {},\n", summary.states));
    js.push_str(&format!(
        "  \"decision_states_with_scenario\": {},\n",
        summary.states_with_scenario
    ));
    js.push_str(&format!("  \"scored_entries\": {},\n", summary.scored_entries));
    js.push_str(&format!("  \"unscored_entries\": {},\n", summary.unscored_entries));
    js.push_str(&format!("  \"rows\": {},\n", summary.rows));
    js.push_str(&format!(
        "  \"scored_not_in_candidates\": {},\n",
        summary.scored_not_in_candidates
    ));
    js.push_str(&format!(
        "  \"candidates_not_in_eval\": {},\n",
        summary.candidates_not_in_eval
    ));
    js.push_str(&format!("  \"no_ranking_states\": {},\n", summary.no_ranking));
    js.push_str("  \"rows_by_source_kif\": {\n");
    for (i, (k, n)) in per_source.iter().enumerate() {
        js.push_str(&format!(
            "    \"{k}\": {{ \"rows\": {n}, \"share\": {:.4} }}{}\n",
            *n as f64 / total.max(1) as f64,
            if i + 1 == per_source.len() { "" } else { "," }
        ));
    }
    js.push_str("  }\n}\n");
    std::fs::write(&summary_path, &js).expect("summary を書けません");
    println!("{out}: {} 行 / {}", summary.rows, summary_path);
    print!("{js}");
}

/// eval のブロックを `(ply, fouls)` へ復元し、シナリオがあれば採点表を照合する
fn restore(stem: &str, kifu: &Kifu, index: &ScenarioIndex, block: &EvalBlock) -> State {
    let ply = block.num - 1;
    assert!(
        ply <= kifu.plies.len(),
        "{stem}: {}手目が棋譜の手数 {} を超えています",
        block.num,
        kifu.plies.len()
    );
    let rep = replay(kifu, ply);
    let side = rep.pos.turn();
    let real = real_fouls_at(kifu, &rep.pos, side, ply);
    let fouls = match block.sub_usi() {
        None => vec![],
        Some(usi) => {
            let idx = real.iter().position(|u| *u == usi).unwrap_or_else(|| {
                panic!(
                    "{stem}: {}手目の反則後ブロック {usi} が棋譜の反則列 {real:?} に無い\
                     （別の手目・別のサブ状態へ接続しています）",
                    block.num
                )
            });
            real[..=idx].to_vec()
        }
    };
    let scored = block.scored();
    let scenario = match index.get(ply, &fouls) {
        None => String::new(),
        Some((name, scores)) => {
            let mut want = scored.clone();
            let mut got = scores.clone();
            want.sort();
            got.sort();
            assert_eq!(
                want, got,
                "{stem} の {}手目（{:?}）→ シナリオ {name} の scores= が eval と一致しません\
                 （sync_eval.py を掛け直してください）",
                block.num, block.sub
            );
            name.clone()
        }
    };
    let mut scores: HashMap<String, Option<u8>> = HashMap::new();
    for (u, p) in &block.entries {
        scores.entry(u.clone()).or_insert(*p);
    }
    State {
        num: block.num,
        ply,
        fouls,
        scenario,
        n_scored: scored.len(),
        scores,
    }
}

struct Chain {
    strat: Box<dyn Strategy>,
    running: ObservationLog,
    consumed: usize,
}

/// 1 (eval, seed) ぶん。戻り値は行と (採点あるが候補外, 候補だが未収載, ランキング無し)
fn run_unit(job: &Job, seed: u64, config: &Arc<StrategyConfig>) -> (Vec<Row>, (usize, usize, usize)) {
    let kifu = &job.kifu;
    // ply -> その ply の決定状態（通常＋反則後）。ply 昇順にまとめて処理する
    let mut by_ply: HashMap<usize, Vec<&State>> = HashMap::new();
    for st in &job.states {
        by_ply.entry(st.ply).or_default().push(st);
    }
    let mut plies: Vec<usize> = by_ply.keys().copied().collect();
    plies.sort_unstable();
    // その ply で手番になる側だけチェーンを進める（不要な側は prewarm しない）
    let mut chains: [Option<Chain>; 2] = [None, None];
    let mut rows = vec![];
    let mut miss = (0usize, 0usize, 0usize);
    for ply in plies {
        let rep = replay(kifu, ply);
        let side = rep.pos.turn();
        let idx = side_idx(side);
        let chain = chains[idx].get_or_insert_with(|| Chain {
            strat: strategy::make_seeded_with_config("estimator", seed, Arc::clone(config))
                .expect("estimator を作れません"),
            running: ObservationLog::default(),
            consumed: 0,
        });
        let base_view = make_view(&rep.pos, side, &rep.fouls);
        let log = &rep.logs[idx];
        let events = log.events();
        while chain.consumed < events.len() {
            let e = &events[chain.consumed];
            if matches!(e, Observation::MyMove { .. } | Observation::MyFoul { .. }) {
                chain.strat.prewarm(&base_view, &chain.running);
            }
            chain.running.record(e.clone());
            chain.consumed += 1;
        }
        let mut states = by_ply.remove(&ply).unwrap();
        states.sort_by_key(|s| s.fouls.len());
        for st in states {
            // 反則後サブ状態の再現（choice_trial_body / rank_dump と同じ規約）
            let mut log2 = clone_log(log);
            let mut fouls_arr = rep.fouls;
            let mut foul_tried: HashSet<String> = HashSet::new();
            for usi in &st.fouls {
                let mv = parse_usi(usi).unwrap_or_else(|| panic!("反則 USI を読めません: {usi}"));
                assert!(
                    !rep.pos.is_legal(&mv),
                    "{}: fouls に指定した手が合法です: {usi}",
                    job.stem
                );
                fouls_arr[idx] += 1;
                log2.record(Observation::MyFoul {
                    move_number: rep.pos.move_number(),
                    usi: usi.clone(),
                });
                foul_tried.insert(usi.clone());
            }
            let view = make_view(&rep.pos, side, &fouls_arr);
            let mut snap = chain
                .strat
                .clone_boxed()
                .expect("estimator は clone_boxed 対応のはず");
            let chosen = snap.choose(&view, &log2, &foul_tried);
            let Some(ranking) = snap.last_ranking().map(<[_]>::to_vec) else {
                eprintln!(
                    "{} {}手目（反則{}）: ランキングなし（定跡/候補ゼロ。choose={:?}）",
                    job.stem,
                    st.num,
                    st.fouls.len(),
                    chosen
                );
                miss.2 += 1;
                continue;
            };
            // 順位系の特徴量はタイブレーク乱数を除いたスコアで付け直す。
            // 乱数込みの実際の順位（= 現行方策の着手順）は engine_rank へ出す
            let order = det_order(&ranking);
            let mut det_rank = vec![0usize; ranking.len()];
            for (r, &i) in order.iter().enumerate() {
                det_rank[i] = r;
            }
            let top = order
                .first()
                .map(|&i| ranking[i].score - ranking[i].tiebreak)
                .unwrap_or(0.0);
            let n = ranking.len();
            let in_set: HashSet<&str> = ranking.iter().map(|c| c.usi.as_str()).collect();
            miss.0 += st
                .scores
                .iter()
                .filter(|(u, p)| p.is_some() && !in_set.contains(u.as_str()))
                .count();
            miss.1 += ranking
                .iter()
                .filter(|c| !st.scores.contains_key(&c.usi))
                .count();
            let id = st.id(&job.stem);
            for (engine_rank, cand) in ranking.iter().enumerate() {
                let ctx = SetContext {
                    rank: det_rank[engine_rank],
                    n_candidates: n,
                    top_score: top,
                    fouls_this_turn: st.fouls.len() as u32,
                };
                let feats = feature_row(&view, cand, &ctx);
                let score = st.scores.get(&cand.usi).copied().flatten();
                let mut line = format!(
                    "{},{},{},{},{},{seed},{},{},1,{engine_rank},{}",
                    job.stem,
                    id,
                    st.scenario,
                    st.ply,
                    if side == Color::Sente { "b" } else { "w" },
                    cand.usi,
                    score.map(|s| s.to_string()).unwrap_or_default(),
                    // 乱数込みの実スコア（特徴量ではない。P1 の数値合成と baseline 用）。
                    // **丸めない**（PR #25 レビュー指摘 P1）: `{:.6}` だと
                    // 5.0000004 と 5.0000003 が同値になり、現行方策では順位が
                    // ついている候補が CSV 上で同点になる。baseline の argmax と
                    // concordance がその偽の同点で変わってしまう。Rust の既定表示は
                    // round-trip する最短表現なので、これで実行時の値が復元できる
                    cand.score,
                );
                for f in feats {
                    // **丸めない**（PR #25 レビュー指摘 P2）。`gain` / `p_legal` /
                    // `foul_cost` は P1 の gain 側合成の成分でもあるので、6桁に
                    // 丸めると runtime の順位と一致しなくなる。列ごとに書式を
                    // 変えるより全部を round-trip 表現で出すほうが契約が単純
                    line.push_str(&format!(",{f}"));
                }
                rows.push(Row {
                    id: format!("{id}|{seed:03}|{engine_rank:04}"),
                    line,
                });
            }
            // **採点済みなのに現行候補集合に無い手**も行として残す（黙って
            // 落とすと candidate recall が測れず、「未採点候補の除外による
            // 見かけの改善」と同じ穴が分析側に開く）。特徴量は空欄 = 欠測、
            // `in_candidates=0` で学習からは外す
            let mut absent: Vec<(&String, u8)> = st
                .scores
                .iter()
                .filter_map(|(u, p)| p.map(|p| (u, p)))
                .filter(|(u, _)| !in_set.contains(u.as_str()))
                .collect();
            absent.sort();
            for (i, (usi, score)) in absent.iter().enumerate() {
                let mut line = format!(
                    "{},{},{},{},{},{seed},{usi},{score},0,,",
                    job.stem,
                    id,
                    st.scenario,
                    st.ply,
                    if side == Color::Sente { "b" } else { "w" },
                );
                line.push_str(&",".repeat(FEATURE_COLUMNS.len()));
                rows.push(Row {
                    id: format!("{id}|{seed:03}|9{i:03}"),
                    line,
                });
            }
        }
    }
    (rows, miss)
}
