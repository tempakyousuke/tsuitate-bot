//! scenario デバッグ GUI のバックエンド。
//!
//! 読み込み・リプレイ・一手選択の本体は tsuitate-bot の `scenario_core` を使う。
//! GUI 固有なのは「全 ply のスナップショット化」（フロントの瞬時ナビゲーション用）と
//! 「seed 並列の集計＋進捗イベント」だけ。

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;
use tauri::{AppHandle, Emitter, State};

use tsuitate_bot::board::make_usi_square;
use tsuitate_bot::kifu::{Kifu, parse_kif};
use tsuitate_bot::protocol::{Color, Role};
use tsuitate_bot::scenario_core::{
    Replayed, choice_trial_one, ranking_one, replay, resolve_foul, scenarios_dir, side_idx,
};
use tsuitate_bot::shogi::{Position, ShogiMove, parse_usi};
use tsuitate_bot::strategy::{self, CandidateScore};

/// GUI で選べるエンジン。凍結版は内部スコアを出せないので seed 集計のみ、
/// ランキング（内訳表示）は last_ranking を実装した現行 estimator だけ
const ENGINES: &[&str] = &[
    "estimator",
    "estimator_v10",
    "estimator_v9",
    "estimator_v8",
    "estimator_v7",
    "estimator_v6",
    "heuristic",
];

/// panic を Err(String) に変換して呼ぶ（scenario_core のリプレイ・裁定検証は
/// 棋譜不整合を assert で検出する設計なので、GUI ではメッセージとして返す）
fn catch<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    match std::panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => Ok(v),
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "不明なエラー".into());
            Err(format!("棋譜の検証エラー: {msg}"))
        }
    }
}

// ---------- list_scenarios ----------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ScenarioInfo {
    path: String,
    name: String,
    /// archive/ 配下なら true（suite 対象外の棋譜）
    archived: bool,
    total_plies: usize,
    directive_ply: Option<usize>,
    target: Option<String>,
    desc: Option<String>,
}

#[tauri::command]
fn list_scenarios() -> Vec<ScenarioInfo> {
    let dir = scenarios_dir();
    let mut out = vec![];
    for (sub, archived) in [(dir.clone(), false), (dir.join("archive"), true)] {
        let Ok(entries) = std::fs::read_dir(&sub) else {
            continue;
        };
        let mut paths: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "kif"))
            .collect();
        paths.sort();
        for path in paths {
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let Ok(kifu) = parse_kif(&text) else {
                eprintln!("{}: パース失敗（一覧から除外）", path.display());
                continue;
            };
            out.push(ScenarioInfo {
                path: path.to_string_lossy().into_owned(),
                name: path
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                archived,
                total_plies: kifu.plies.len(),
                directive_ply: kifu.directives.get("ply").and_then(|s| s.parse().ok()),
                target: kifu.directives.get("target").cloned(),
                desc: kifu.directives.get("desc").cloned(),
            });
        }
    }
    out
}

#[tauri::command]
fn engines() -> Vec<String> {
    ENGINES
        .iter()
        .filter(|n| strategy::make_seeded(n, 0).is_some())
        .map(|n| n.to_string())
        .collect()
}

// ---------- load_kifu ----------

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct PieceOut {
    role: Role,
    color: Color,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct LastMove {
    usi: String,
    /// 打ちのときは None
    from: Option<String>,
    to: String,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct Snapshot {
    /// USIマス（"7g"）→ 駒。空マスはキーなし
    board: HashMap<String, PieceOut>,
    hand_sente: HashMap<Role, u32>,
    hand_gote: HashMap<Role, u32>,
    turn: Color,
    move_number: u32,
    /// [先手, 後手] の反則累計（この局面に至るまでの試行）
    fouls: [u32; 2],
    /// [先手が王手されている, 後手が王手されている]
    in_check: [bool; 2],
    last_move: Option<LastMove>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct MoveRow {
    usi: String,
    /// この手の前に同じ手番側が試みた反則（USI）
    fouls_before: Vec<String>,
    side: Color,
    gives_check: bool,
    capture: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct KifuData {
    path: String,
    name: String,
    total_plies: usize,
    directive_ply: Option<usize>,
    /// *scenario target= があればそれ。無ければ directive_ply+1 手目の実際の手
    target: Option<String>,
    desc: Option<String>,
    /// snapshots[i] = i 手指した後の局面（i=0 は初期局面）
    snapshots: Vec<Snapshot>,
    moves: Vec<MoveRow>,
}

fn snapshot_of(pos: &Position, fouls: &[u32; 2], last_move: Option<LastMove>) -> Snapshot {
    let mut board = HashMap::new();
    for (sq, pc) in pos.pieces() {
        board.insert(
            make_usi_square(sq),
            PieceOut {
                role: pc.role,
                color: pc.color,
            },
        );
    }
    Snapshot {
        board,
        hand_sente: pos.hand_map(Color::Sente),
        hand_gote: pos.hand_map(Color::Gote),
        turn: pos.turn(),
        move_number: pos.move_number(),
        fouls: *fouls,
        in_check: [pos.in_check(Color::Sente), pos.in_check(Color::Gote)],
        last_move,
    }
}

/// 全手を裁定つきでリプレイして ply ごとのスナップショットを作る
/// （scenario_core::replay と同じ裁定・反則解決。観測ログは作らない）
fn build_snapshots(kifu: &Kifu) -> (Vec<Snapshot>, Vec<MoveRow>) {
    let mut pos = Position::initial();
    let mut fouls = [0u32; 2];
    let mut snapshots = vec![snapshot_of(&pos, &fouls, None)];
    let mut moves = vec![];
    for ply in &kifu.plies {
        let side = pos.turn();
        let mut fouls_before = vec![];
        for f in &ply.fouls {
            let usi = resolve_foul(&pos, side, f);
            let mv = parse_usi(&usi).expect("反則試行のUSI解析失敗");
            assert!(!pos.is_legal(&mv), "反則のはずの手が合法: {usi}");
            fouls[side_idx(side)] += 1;
            fouls_before.push(usi);
        }
        let usi = ply.mv.to_usi();
        let mv = parse_usi(&usi).unwrap_or_else(|| panic!("USI解析失敗: {usi}"));
        assert!(pos.is_legal(&mv), "棋譜の手が非合法: {usi}");
        let captured = pos.play_unchecked(&mv);
        let (from, to) = match &mv {
            ShogiMove::Board { from, to, .. } => {
                (Some(make_usi_square(*from)), make_usi_square(*to))
            }
            ShogiMove::Drop { to, .. } => (None, make_usi_square(*to)),
        };
        snapshots.push(snapshot_of(
            &pos,
            &fouls,
            Some(LastMove {
                usi: usi.clone(),
                from,
                to,
            }),
        ));
        moves.push(MoveRow {
            usi,
            fouls_before,
            side,
            gives_check: pos.in_check(pos.turn()),
            capture: captured.is_some(),
        });
    }
    (snapshots, moves)
}

#[tauri::command]
fn load_kifu(path: String) -> Result<KifuData, String> {
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{path} を読めません: {e}"))?;
    let kifu = parse_kif(&text).map_err(|e| format!("{path}: {e}"))?;
    let (snapshots, moves) = catch(|| build_snapshots(&kifu))?;
    let directive_ply: Option<usize> = kifu.directives.get("ply").and_then(|s| s.parse().ok());
    let target = kifu
        .directives
        .get("target")
        .cloned()
        .or_else(|| directive_ply.and_then(|p| kifu.plies.get(p).map(|x| x.mv.to_usi())));
    Ok(KifuData {
        name: PathBuf::from(&path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
        path,
        total_plies: kifu.plies.len(),
        directive_ply,
        target,
        desc: kifu.directives.get("desc").cloned(),
        snapshots,
        moves,
    })
}

// ---------- eval（集計・ランキング） ----------

#[derive(Default)]
struct EvalState {
    cancels: Mutex<HashMap<u32, Arc<AtomicBool>>>,
}

/// 思考予算（`TSUITATE_THINK_BUDGET_MS`）は各エンジンが**構築時に env から読む**
/// 設計（凍結版も同じ）なので、GUI からの指定は env の書き換えで実現する。
/// env はプロセス全域なので、eval 実行全体をこのロックで直列化して
/// 「実行Aの途中で実行Bが予算を書き換える」競合を防ぐ（UI 側も実行中は
/// ボタンを無効化しているが、防御はバックエンドで持つ）
static EVAL_LOCK: Mutex<()> = Mutex::new(());

fn with_budget<T>(budget_ms: u32, f: impl FnOnce() -> T) -> T {
    let _guard = EVAL_LOCK.lock().unwrap();
    // 100ms未満・60秒超は入力ミスとみなして丸める
    let ms = budget_ms.clamp(100, 60_000);
    std::env::set_var("TSUITATE_THINK_BUDGET_MS", ms.to_string());
    f()
}

/// path/ply から Replayed を作る（棋譜不整合・終局局面はエラー文字列に変換）
fn replayed_at(path: &str, ply: usize) -> Result<Replayed, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path} を読めません: {e}"))?;
    let kifu = parse_kif(&text).map_err(|e| format!("{path}: {e}"))?;
    if ply > kifu.plies.len() {
        return Err(format!(
            "ply={ply} が棋譜の手数 {} を超えています",
            kifu.plies.len()
        ));
    }
    let rep = catch(move || replay(&kifu, ply))?;
    if let Some(outcome) = rep.pos.outcome() {
        return Err(format!("ply={ply} の局面は終局しています（{outcome:?}）"));
    }
    Ok(rep)
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct TrialOutcome {
    seed: u64,
    accepted: String,
    fouls: Vec<String>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct ProgressEvent {
    run_id: u32,
    done: u64,
    total: u64,
    outcome: TrialOutcome,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TallyEntry {
    usi: String,
    count: u32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TallyResult {
    engine: String,
    /// 手番側（この側の視点で評価した）
    side: Color,
    tally: Vec<TallyEntry>,
    total_fouls: u32,
    trials: Vec<TrialOutcome>,
    cancelled: bool,
}

/// seed 0..trials を並列に試行して選択分布を集計する。
/// 1試行終わるごとに `eval-progress` イベントを emit する。
/// キャンセルは試行の切れ目でしか効かない（choose 中は中断できない）
#[tauri::command]
async fn eval_tally(
    app: AppHandle,
    state: State<'_, EvalState>,
    run_id: u32,
    path: String,
    ply: usize,
    engine: String,
    trials: u64,
    // 省略時は既定の2000ms（webview 側が古いままでも落とさない）
    budget_ms: Option<u32>,
) -> Result<TallyResult, String> {
    if strategy::make_seeded(&engine, 0).is_none() {
        return Err(format!("未知の戦略名です: {engine}"));
    }
    let cancel = Arc::new(AtomicBool::new(false));
    state.cancels.lock().unwrap().insert(run_id, cancel.clone());

    let result = tauri::async_runtime::spawn_blocking(move || {
        let rep = replayed_at(&path, ply)?;
        let side = rep.pos.turn();
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(4)
            .min(trials.max(1) as usize);
        let next_seed = AtomicU64::new(0);
        let done = AtomicU64::new(0);
        let outcomes: Mutex<Vec<TrialOutcome>> = Mutex::new(vec![]);
        with_budget(budget_ms.unwrap_or(2000), || std::thread::scope(|scope| {
            for _ in 0..n_threads {
                scope.spawn(|| {
                    loop {
                        if cancel.load(Ordering::Relaxed) {
                            break;
                        }
                        let seed = next_seed.fetch_add(1, Ordering::Relaxed);
                        if seed >= trials {
                            break;
                        }
                        let trial = catch(|| choice_trial_one(&rep, seed, &engine));
                        let (accepted, fouls) = match trial {
                            Ok(v) => v,
                            Err(e) => (format!("エラー: {e}"), vec![]),
                        };
                        let outcome = TrialOutcome {
                            seed,
                            accepted,
                            fouls,
                        };
                        let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                        let _ = app.emit(
                            "eval-progress",
                            ProgressEvent {
                                run_id,
                                done: d,
                                total: trials,
                                outcome: outcome.clone(),
                            },
                        );
                        outcomes.lock().unwrap().push(outcome);
                    }
                });
            }
        }));
        let mut trials_out = outcomes.into_inner().unwrap();
        trials_out.sort_by_key(|t| t.seed);
        let mut tally_map: HashMap<String, u32> = HashMap::new();
        let mut total_fouls = 0u32;
        for t in &trials_out {
            *tally_map.entry(t.accepted.clone()).or_insert(0) += 1;
            total_fouls += t.fouls.len() as u32;
        }
        let mut tally: Vec<TallyEntry> = tally_map
            .into_iter()
            .map(|(usi, count)| TallyEntry { usi, count })
            .collect();
        tally.sort_by(|a, b| b.count.cmp(&a.count).then(a.usi.cmp(&b.usi)));
        Ok(TallyResult {
            engine,
            side,
            tally,
            total_fouls,
            trials: trials_out,
            cancelled: cancel.load(Ordering::Relaxed),
        })
    })
    .await;

    state.cancels.lock().unwrap().remove(&run_id);
    result.map_err(|e| format!("実行スレッドの異常終了: {e}"))?
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RankingResult {
    engine: String,
    side: Color,
    seed: u64,
    chosen: String,
    ranking: Vec<CandidateScore>,
}

/// 現行 estimator の1回の choose を実行し、全候補の評価内訳（スコア降順）を返す
#[tauri::command]
async fn eval_ranking(
    path: String,
    ply: usize,
    engine: String,
    seed: u64,
    // 省略時は既定の2000ms（webview 側が古いままでも落とさない）
    budget_ms: Option<u32>,
) -> Result<RankingResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let rep = replayed_at(&path, ply)?;
        let side = rep.pos.turn();
        let result =
            with_budget(budget_ms.unwrap_or(2000), || catch(|| ranking_one(&rep, seed, &engine)))?;
        match result {
            Some((chosen, ranking)) => Ok(RankingResult {
                engine,
                side,
                seed,
                chosen,
                ranking,
            }),
            None => Err(
                "候補評価が取れませんでした（定跡で指した手番か、このエンジンは\
                 ランキング未対応です。seed を変えるか ply を進めてください）"
                    .into(),
            ),
        }
    })
    .await
    .map_err(|e| format!("実行スレッドの異常終了: {e}"))?
}

#[tauri::command]
fn cancel_eval(state: State<'_, EvalState>, run_id: u32) {
    if let Some(flag) = state.cancels.lock().unwrap().get(&run_id) {
        flag.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(name: &str) -> Kifu {
        let path = scenarios_dir().join(format!("{name}.kif"));
        let text = std::fs::read_to_string(&path).unwrap();
        parse_kif(&text).unwrap()
    }

    /// ブラウザ単体の UI 確認（src/mock.ts、`?mock=1`）用フィクスチャ生成。
    /// `FIXTURE_DIR=$(pwd)/public/fixtures cargo test --manifest-path src-tauri/Cargo.toml -- --ignored fixtures`
    #[test]
    #[ignore]
    fn dump_fixtures() {
        let out = std::env::var("FIXTURE_DIR").expect("FIXTURE_DIR を指定してください");
        std::fs::create_dir_all(&out).unwrap();
        let write = |name: &str, v: serde_json::Value| {
            std::fs::write(format!("{out}/{name}.json"), serde_json::to_string(&v).unwrap())
                .unwrap();
        };
        write(
            "scenarios",
            serde_json::to_value(list_scenarios()).unwrap(),
        );
        let path = scenarios_dir()
            .join("kakutori.kif")
            .to_string_lossy()
            .into_owned();
        let kifu = load_kifu(path.clone()).unwrap();
        let ply = kifu.directive_ply.unwrap();
        write("kakutori", serde_json::to_value(&kifu).unwrap());

        let rep = replayed_at(&path, ply).unwrap();
        let side = rep.pos.turn();
        let trials: Vec<TrialOutcome> = (0..5)
            .map(|seed| {
                let (accepted, fouls) = choice_trial_one(&rep, seed, "estimator");
                TrialOutcome {
                    seed,
                    accepted,
                    fouls,
                }
            })
            .collect();
        let mut tally_map: HashMap<String, u32> = HashMap::new();
        let mut total_fouls = 0;
        for t in &trials {
            *tally_map.entry(t.accepted.clone()).or_insert(0) += 1;
            total_fouls += t.fouls.len() as u32;
        }
        let mut tally: Vec<TallyEntry> = tally_map
            .into_iter()
            .map(|(usi, count)| TallyEntry { usi, count })
            .collect();
        tally.sort_by(|a, b| b.count.cmp(&a.count));
        write(
            "tally",
            serde_json::to_value(TallyResult {
                engine: "estimator".into(),
                side,
                tally,
                total_fouls,
                trials,
                cancelled: false,
            })
            .unwrap(),
        );

        let (chosen, ranking) = ranking_one(&rep, 0, "estimator").unwrap();
        write(
            "ranking",
            serde_json::to_value(RankingResult {
                engine: "estimator".into(),
                side,
                seed: 0,
                chosen,
                ranking,
            })
            .unwrap(),
        );
    }

    /// スナップショット列が scenario_core::replay と同じ裁定・反則会計になること
    #[test]
    fn snapshotsはreplayと整合する() {
        for name in ["keima", "kakunari", "ansatsu"] {
            let kifu = load(name);
            let (snapshots, moves) = build_snapshots(&kifu);
            assert_eq!(snapshots.len(), kifu.plies.len() + 1);
            assert_eq!(moves.len(), kifu.plies.len());
            // 初期局面: 40枚・持ち駒なし・先手番
            assert_eq!(snapshots[0].board.len(), 40);
            assert!(snapshots[0].hand_sente.is_empty());
            assert_eq!(snapshots[0].turn, Color::Sente);
            assert!(snapshots[0].last_move.is_none());
            // 任意の途中 ply で replay と盤面・反則数が一致する
            for ply in [1, kifu.plies.len() / 2, kifu.plies.len()] {
                let rep = replay(&kifu, ply);
                let snap = &snapshots[ply];
                assert_eq!(snap.fouls, rep.fouls, "{name} ply={ply} の反則数");
                assert_eq!(snap.turn, rep.pos.turn(), "{name} ply={ply} の手番");
                assert_eq!(snap.move_number, rep.pos.move_number());
                let n_pieces = rep.pos.pieces().count();
                assert_eq!(snap.board.len(), n_pieces, "{name} ply={ply} の駒数");
                for (sq, pc) in rep.pos.pieces() {
                    let out = &snap.board[&make_usi_square(sq)];
                    assert_eq!((out.role, out.color), (pc.role, pc.color));
                }
            }
        }
    }
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(EvalState::default())
        .invoke_handler(tauri::generate_handler![
            list_scenarios,
            engines,
            load_kifu,
            eval_tally,
            eval_ranking,
            cancel_eval,
        ])
        .run(tauri::generate_context!())
        .expect("tauri アプリの起動に失敗");
}
