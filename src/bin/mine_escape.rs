//! 自動対局の記録から「ついたて詰み逃れ」の問題を掘る。
//!
//! **ついたて詰み逃れ**はサイト（tsuitate）の問題モードで、ついたて詰将棋の
//! 裏返し。初期局面は全公開だが、以後の攻め方の王手は見えない。受け方（解き手）は
//! 自分の手の受理/反則・取った駒・取られたマス・王手の宣言だけを頼りに、
//! 許された反則回数のうちに「攻め方がもう王手をかけられない」状態まで逃げる。
//! 攻め方は受け方の観測を変えない範囲で途中の手を後から差し替えられる
//! （サイトの審判は観測と矛盾しない局面の集合を持ち、敵対的に分岐を選ぶ）。
//!
//! 使い方:
//!   cargo run --release --bin mine_escape -- <記録ファイル/ディレクトリ...> [オプション]
//!
//! オプション:
//!   --out <path>            取り込みJSONの出力先（既定: 標準出力）
//!   --solver <path>         ソルバーCLI のパス（既定: 環境変数 SOLVER_BIN）
//!   --source <name>         取り込み元識別子（既定: selfplay）
//!   --skip-plies N          この手数までの局面は無視する（既定: 20）
//!   --min-checks N          採用する「逃れるまでの攻め方の王手の回数」の下限（既定: 2）
//!   --max-checks N          同上限。ソルバーが攻め方の王手を何回先まで読むか（読みの深さ）も兼ねる（既定: 5）
//!   --max-fouls N           調べる反則回数の上限（既定: 3）
//!   --limit N               採用する問題数の上限（0 = 無制限）
//!   --max-per-game N        1対局・1受け方から採る問題数の上限（既定: 1、0 = 無制限）
//!   --max-candidates N      ソルバーに投げる候補数の上限（0 = 無制限）
//!   --jobs N                ソルバーの並列実行数（既定: 1）
//!   --timeout-secs N        ソルバーの1問あたりのタイムアウト（既定: 30）
//!   --node-limit N          ソルバーのノード上限（既定: 2,000,000）
//!   --allow-trivial         どの初手でも逃れられる問題も採用する（既定: 捨てる）
//!   --dump-candidates <path> 候補の問題JSONだけを書き出してソルバーを呼ばない
//!
//! ## 問題の作り方
//!
//! ある決定点の**実戦の局面をそのまま**問題にする。手番側が攻め方、相手が受け方。
//! 盤上の駒は両者とも全部残し、**持ち駒も両者とも実戦のまま明示する**
//! （詰将棋と違い「残り全部が玉方の持ち駒」にはしない。実戦の受け方が
//! 持っていた駒だけで受ける）。
//!
//! 受け方を**先手（盤の手前）**にそろえる。攻め方が先手の局面は盤面を180度回す。
//! 解き手は受け方なので、自分の玉が手前に来る向きのほうが自然に読める。
//!
//! ## 採用の条件
//!
//! - ソルバー（`--escape`）が、反則 `--max-fouls` 回以内・攻め方の王手を `--max-checks` 回
//!   先まで読んで逃れを強制できると証明した。サイトでは**王手の回数に上限は無い**が、
//!   この深さで逃れきる筋がある問題だけを採るので、正しく受ければ必ず逃れられる
//! - 問題の反則回数は**逃れに必要な最少の回数**にする（その読みの深さの範囲で、1回でも
//!   少なければ逃れられない）
//! - 逃れるまでの攻め方の王手が `--min-checks`〜`--max-checks` 回
//! - 受け方の初手に「逃れられない手」が1つ以上ある（`--allow-trivial` で無効）。
//!   どの手を指しても逃れられる問題は解き手が何も考えなくてよい
//!
//! **1対局・1受け方から採るのは1問だけ**（`--max-per-game`）。同じ対局の
//! 隣り合う局面はほとんど同じ問題になる（mine_tsume と同じ理由）。
//! 残すのは王手の回数が最も多い局面（同数なら反則回数が多い方、さらに手前）。

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tsuitate_bot::board::{Coord, parse_usi_square};
use tsuitate_bot::protocol::{Color, Role};
use tsuitate_bot::shogi::Position;
use tsuitate_bot::truth_replay::{for_each_decision, load_end};

/// 持ち駒の並び順（署名を安定させるため固定）
const HAND_ORDER: [Role; 7] = [
    Role::Rook,
    Role::Bishop,
    Role::Gold,
    Role::Silver,
    Role::Knight,
    Role::Lance,
    Role::Pawn,
];

fn role_kind(role: Role) -> &'static str {
    match role {
        Role::King => "king",
        Role::Rook => "rook",
        Role::Bishop => "bishop",
        Role::Gold => "gold",
        Role::Silver => "silver",
        Role::Knight => "knight",
        Role::Lance => "lance",
        Role::Pawn => "pawn",
        Role::Dragon => "promoted_rook",
        Role::Horse => "promoted_bishop",
        Role::Promotedsilver => "promoted_silver",
        Role::Promotedknight => "promoted_knight",
        Role::Promotedlance => "promoted_lance",
        Role::Tokin => "promoted_pawn",
    }
}

/// 受け方が後手（＝攻め方が先手）のときは盤面を180度回して受け方を先手にする
fn orient(c: Coord, flip: bool) -> (u8, u8) {
    if flip {
        ((10 - c.file) as u8, (10 - c.rank) as u8)
    } else {
        (c.file as u8, c.rank as u8)
    }
}

/// 手番側（攻め方）に受け方の玉への王手が1つでもあるか
fn attacker_can_check(pos: &Position, attacker: Color) -> bool {
    let defender = attacker.other();
    pos.legal_moves().iter().any(|mv| {
        let mut next = pos.clone();
        next.play_unchecked(mv);
        next.in_check(defender)
    })
}

struct Candidate {
    /// ソルバー入力そのもの（受け方 = 先手、攻め方 = 後手の手番）
    question: Value,
    /// 正規化JSONの SHA-256 先頭16桁。重複排除と取り込みの upsert キーを兼ねる
    signature: String,
    /// どの対局（記録ファイル名）のどちらの受け方から採ったか。`--max-per-game` の単位
    game: String,
    /// どの記録の何手目から採ったか（出典。デバッグ・再現用）
    origin: String,
}

fn hand_json(pos: &Position, color: Color) -> Vec<Value> {
    let hands = pos.hand_map(color);
    HAND_ORDER
        .iter()
        .filter_map(|&role| {
            let count = *hands.get(&role)?;
            (count > 0).then(|| json!({ "kind": role_kind(role), "count": count }))
        })
        .collect()
}

/// 決定点の真の局面から、詰み逃れの問題を作る（手番側 = 攻め方）。
///
/// 受け方の玉が無い・受け方が既に王手されている（攻め方の手番なのに王手 =
/// 不正な局面）・攻め方に王手が1つも無い（最初から逃れている）局面は None
fn question_from(pos: &Position, attacker: Color) -> Option<Value> {
    let defender = attacker.other();
    pos.king_square(defender)?;
    if pos.in_check(defender) {
        return None;
    }
    if !attacker_can_check(pos, attacker) {
        return None;
    }
    let flip = attacker == Color::Sente;

    let mut board: Vec<(u8, u8, &'static str, &'static str)> = Vec::new();
    for (color, label) in [(defender, "sente"), (attacker, "gote")] {
        for piece in pos.pieces_of(color) {
            let coord = parse_usi_square(&piece.square)?;
            let (file, rank) = orient(coord, flip);
            board.push((file, rank, label, role_kind(piece.role)));
        }
    }
    board.sort_by_key(|&(file, rank, color, _)| (file, rank, color));

    Some(json!({
        "board": board
            .into_iter()
            .map(|(file, rank, color, kind)| json!({
                "file": file, "rank": rank, "color": color, "kind": kind
            }))
            .collect::<Vec<_>>(),
        "sente_hand": hand_json(pos, defender),
        "gote_hand": hand_json(pos, attacker),
        "side_to_move": "gote",
    }))
}

fn signature_of(question: &Value) -> String {
    let canonical = serde_json::to_string(question).unwrap_or_default();
    let digest = Sha256::digest(canonical.as_bytes());
    digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

fn collect_record_files(inputs: &[String]) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for input in inputs {
        let path = Path::new(input);
        if path.is_dir() {
            let Ok(entries) = std::fs::read_dir(path) else {
                eprintln!("読めないディレクトリ: {input}");
                continue;
            };
            let mut dir_files: Vec<PathBuf> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|ext| ext == "jsonl"))
                .collect();
            dir_files.sort();
            files.extend(dir_files);
        } else {
            files.push(path.to_path_buf());
        }
    }
    files
}

struct Args {
    inputs: Vec<String>,
    out: Option<String>,
    solver: Option<String>,
    source: String,
    skip_plies: u32,
    min_checks: u64,
    max_checks: u64,
    max_fouls: u64,
    limit: usize,
    max_per_game: usize,
    max_candidates: usize,
    jobs: usize,
    timeout_secs: u64,
    node_limit: u64,
    allow_trivial: bool,
    dump_candidates: Option<String>,
}

fn default_args() -> Args {
    Args {
        inputs: vec![],
        out: None,
        solver: std::env::var("SOLVER_BIN").ok(),
        source: "selfplay".into(),
        skip_plies: 20,
        min_checks: 2,
        max_checks: 5,
        max_fouls: 3,
        limit: 0,
        max_per_game: 1,
        max_candidates: 0,
        jobs: 1,
        timeout_secs: 30,
        node_limit: 2_000_000,
        allow_trivial: false,
        dump_candidates: None,
    }
}

fn parse_args() -> Result<Args, String> {
    let mut args = default_args();
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        let mut value = |name: &str| -> Result<String, String> {
            iter.next().ok_or_else(|| format!("{name} には値が必要です"))
        };
        let num = |name: &str, v: String| -> Result<u64, String> {
            v.parse().map_err(|_| format!("{name} が数値ではありません"))
        };
        match arg.as_str() {
            "--out" => args.out = Some(value("--out")?),
            "--solver" => args.solver = Some(value("--solver")?),
            "--source" => args.source = value("--source")?,
            "--skip-plies" => args.skip_plies = num("--skip-plies", value("--skip-plies")?)? as u32,
            "--min-checks" => args.min_checks = num("--min-checks", value("--min-checks")?)?,
            "--max-checks" => args.max_checks = num("--max-checks", value("--max-checks")?)?,
            "--max-fouls" => args.max_fouls = num("--max-fouls", value("--max-fouls")?)?,
            "--limit" => args.limit = num("--limit", value("--limit")?)? as usize,
            "--max-per-game" => args.max_per_game = num("--max-per-game", value("--max-per-game")?)? as usize,
            "--max-candidates" => {
                args.max_candidates = num("--max-candidates", value("--max-candidates")?)? as usize
            }
            "--jobs" => args.jobs = num("--jobs", value("--jobs")?)? as usize,
            "--timeout-secs" => args.timeout_secs = num("--timeout-secs", value("--timeout-secs")?)?,
            "--node-limit" => args.node_limit = num("--node-limit", value("--node-limit")?)?,
            "--allow-trivial" => args.allow_trivial = true,
            "--dump-candidates" => args.dump_candidates = Some(value("--dump-candidates")?),
            _ if arg.starts_with("--") => return Err(format!("不明なオプション: {arg}")),
            _ => args.inputs.push(arg),
        }
    }
    if args.inputs.is_empty() {
        return Err("記録ファイル（またはディレクトリ）を指定してください".into());
    }
    if args.jobs == 0 {
        return Err("--jobs は1以上にしてください".into());
    }
    if args.min_checks > args.max_checks {
        return Err("--min-checks は --max-checks 以下にしてください".into());
    }
    Ok(args)
}

/// 記録から候補局面を集める（重複は署名で落とす）
fn collect_candidates(args: &Args) -> Vec<Candidate> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut candidates: Vec<Candidate> = Vec::new();
    let files = collect_record_files(&args.inputs);
    eprintln!("記録ファイル: {}件", files.len());

    for path in &files {
        let Some(end) = load_end(&path.to_string_lossy()) else {
            eprintln!("終局ペイロードが読めません: {}", path.display());
            continue;
        };
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let mut found: Vec<Candidate> = Vec::new();
        let ok = for_each_decision(&end, |pos, side, _log, decision_id| {
            if (decision_id as u32) < args.skip_plies {
                return;
            }
            let Some(question) = question_from(pos, side) else {
                return;
            };
            let signature = signature_of(&question);
            let defender = if side == Color::Sente { "gote" } else { "sente" };
            found.push(Candidate {
                question,
                signature,
                game: format!("{name}/{defender}"),
                origin: format!("{name}#{decision_id}"),
            });
        });
        if !ok {
            eprintln!("棋譜が壊れているので丸ごと捨てます: {}", path.display());
            continue;
        }
        for candidate in found {
            if seen.insert(candidate.signature.clone()) {
                candidates.push(candidate);
            }
        }
        if args.max_candidates > 0 && candidates.len() >= args.max_candidates {
            candidates.truncate(args.max_candidates);
            break;
        }
    }
    candidates
}

/// ソルバーCLI（`--escape`）を1問走らせて出力JSONを返す
fn run_solver(args: &Args, solver: &str, candidate: &Candidate, slot: usize) -> Option<Value> {
    let tmp = std::env::temp_dir().join(format!("mine-escape-{}-{}.json", std::process::id(), slot));
    if let Err(e) = std::fs::write(&tmp, serde_json::to_vec(&candidate.question).ok()?) {
        eprintln!("一時ファイルを書けません: {e}");
        return None;
    }
    let output = std::process::Command::new(solver)
        .arg("--escape")
        .arg(&tmp)
        .arg("--max-fouls")
        .arg(args.max_fouls.to_string())
        .arg("--max-checks")
        .arg(args.max_checks.to_string())
        .arg("--timeout-secs")
        .arg(args.timeout_secs.to_string())
        .arg("--node-limit")
        .arg(args.node_limit.to_string())
        .output();
    let _ = std::fs::remove_file(&tmp);
    let output = match output {
        Ok(o) => o,
        Err(e) => {
            eprintln!("ソルバーを起動できません（{solver}）: {e}");
            return None;
        }
    };
    serde_json::from_slice(&output.stdout).ok()
}

/// 採用判定
fn accept(args: &Args, result: &Value) -> bool {
    if result["verdict"] != json!("escape") {
        return false;
    }
    let (Some(checks), Some(_fouls)) = (result["checks"].as_u64(), result["minFouls"].as_u64()) else {
        return false;
    };
    if checks < args.min_checks || checks > args.max_checks {
        return false;
    }
    if !args.allow_trivial {
        // 初手の統計が取れない（打ち切り）ものは難しさが分からないので捨てる
        let (Some(tries), Some(solutions)) = (result["rootTries"].as_u64(), result["rootSolutions"].as_u64())
        else {
            return false;
        };
        if solutions >= tries {
            return false;
        }
    }
    true
}

/// 同じ対局・同じ受け方の問題を `--max-per-game` 件まで間引く。
/// 残すのは「王手の回数・反則回数が多い」順（同じなら手番順で手前）。出力は手番順のまま
fn take_per_game<T>(
    args: &Args,
    accepted: Vec<T>,
    game_of: impl Fn(&T) -> String,
    weight_of: impl Fn(&T) -> (u64, u64),
) -> Vec<T> {
    if args.max_per_game == 0 {
        return accepted;
    }
    let mut by_game: HashMap<String, Vec<((u64, u64), usize)>> = HashMap::new();
    for (idx, item) in accepted.iter().enumerate() {
        by_game.entry(game_of(item)).or_default().push((weight_of(item), idx));
    }
    let mut keep: HashSet<usize> = HashSet::new();
    for entries in by_game.values_mut() {
        entries.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        keep.extend(entries.iter().take(args.max_per_game).map(|(_, idx)| *idx));
    }
    accepted
        .into_iter()
        .enumerate()
        .filter_map(|(idx, item)| keep.contains(&idx).then_some(item))
        .collect()
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };

    let candidates = collect_candidates(&args);
    eprintln!("候補局面: {}件（重複除去後）", candidates.len());

    if let Some(path) = &args.dump_candidates {
        let dump: Vec<Value> = candidates
            .iter()
            .map(|c| json!({ "sourceId": c.signature, "origin": c.origin, "question": c.question }))
            .collect();
        write_out(Some(path), &json!({ "candidates": dump }));
        return;
    }

    let Some(solver) = args.solver.clone() else {
        eprintln!("ソルバーCLI のパスを --solver か SOLVER_BIN で指定してください");
        std::process::exit(2);
    };

    // 候補を並列に検証する。結果は元の順番に戻すので出力は決定的
    let next = Mutex::new(0usize);
    let results: Vec<Mutex<Option<Value>>> = (0..candidates.len()).map(|_| Mutex::new(None)).collect();
    let done = Mutex::new(0usize);
    let workers = args.jobs.min(candidates.len()).max(1);
    std::thread::scope(|scope| {
        for slot in 0..workers {
            let next = &next;
            let results = &results;
            let done = &done;
            let candidates = &candidates;
            let args = &args;
            let solver = solver.as_str();
            scope.spawn(move || {
                loop {
                    let idx = {
                        let mut guard = next.lock().unwrap();
                        let idx = *guard;
                        if idx >= candidates.len() {
                            break;
                        }
                        *guard += 1;
                        idx
                    };
                    let result = run_solver(args, solver, &candidates[idx], slot);
                    *results[idx].lock().unwrap() = result;
                    let mut count = done.lock().unwrap();
                    *count += 1;
                    if *count % 50 == 0 {
                        eprintln!("検証 {}/{}", *count, candidates.len());
                    }
                }
            });
        }
    });

    let mut verdicts: HashMap<String, usize> = HashMap::new();
    let accepted: Vec<(&Candidate, Value)> = candidates
        .iter()
        .zip(results)
        .filter_map(|(candidate, result)| {
            let result = result.into_inner().unwrap()?;
            let key = result["verdict"].as_str().unwrap_or("error").to_string();
            *verdicts.entry(key).or_default() += 1;
            accept(&args, &result).then_some((candidate, result))
        })
        .collect();
    let mut verdict_summary: Vec<_> = verdicts.into_iter().collect();
    verdict_summary.sort();
    eprintln!("ソルバーの判定: {verdict_summary:?}");
    let accepted_total = accepted.len();
    let accepted = take_per_game(
        &args,
        accepted,
        |(candidate, _)| candidate.game.clone(),
        |(_, result)| {
            (result["checks"].as_u64().unwrap_or(0), result["minFouls"].as_u64().unwrap_or(0))
        },
    );
    if accepted.len() < accepted_total {
        eprintln!(
            "同じ対局・同じ受け方の問題を間引き: {}件 → {}件（--max-per-game {}）",
            accepted_total,
            accepted.len(),
            args.max_per_game
        );
    }

    let mut puzzles: Vec<Value> = Vec::new();
    for (candidate, result) in accepted {
        puzzles.push(json!({
            "sourceId": candidate.signature,
            "origin": candidate.origin,
            "board": candidate.question["board"],
            "sente_hand": candidate.question["sente_hand"],
            "gote_hand": candidate.question["gote_hand"],
            "fouls": result["minFouls"],
            "depth": result["depth"],
            "features": {
                "checks": result["checks"],
                "firstCheckPositions": result["firstCheckPositions"],
                "firstCheckGroups": result["firstCheckGroups"],
                "rootTries": result["rootTries"],
                "rootSolutions": result["rootSolutions"],
                "nodes": result["nodes"],
            },
        }));
        if args.limit > 0 && puzzles.len() >= args.limit {
            break;
        }
    }

    eprintln!("採用: {}件 / 候補 {}件", puzzles.len(), candidates.len());
    write_out(
        args.out.as_deref(),
        &json!({ "source": args.source, "puzzles": puzzles }),
    );
}

fn write_out(path: Option<&str>, value: &Value) {
    let text = serde_json::to_string(value).unwrap_or_default();
    match path {
        Some(path) => {
            if let Err(e) = std::fs::write(path, text) {
                eprintln!("出力できません（{path}）: {e}");
                std::process::exit(1);
            }
            eprintln!("書き出しました: {path}");
        }
        None => {
            let mut out = std::io::stdout().lock();
            let _ = writeln!(out, "{text}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tsuitate_bot::shogi::parse_usi;

    fn play(pos: &mut Position, moves: &[&str]) {
        for usi in moves {
            pos.play_unchecked(&parse_usi(usi).unwrap());
        }
    }

    /// 王手の手段が無い局面は問題にしない（最初から逃れている）
    #[test]
    fn positions_without_any_check_are_skipped() {
        let pos = Position::initial();
        assert!(question_from(&pos, Color::Sente).is_none());
    }

    /// 受け方を先手（手前）にそろえ、持ち駒は両者とも実戦のまま明示する
    #[test]
    fn defender_is_sente_and_both_hands_are_explicit() {
        let mut pos = Position::initial();
        // ▲7六歩 △3四歩 ▲2二角成 △同銀 ▲4五角: 後手玉に王手できる形
        // （▲2二角成で先手が角を取り、△同銀で後手も角を取る）
        play(&mut pos, &["7g7f", "3c3d", "8h2b+", "3a2b"]);
        // 先手番 = 先手が攻め方、後手が受け方。持ち駒の角で王手できる
        let q = question_from(&pos, Color::Sente).expect("問題が作れる");
        assert_eq!(q["side_to_move"], json!("gote"));
        // 受け方（実戦の後手）が先手として手前に来る: 後手玉 5一 → 5九
        let board = q["board"].as_array().unwrap();
        assert!(board.iter().any(|p| {
            p["color"] == json!("sente")
                && p["kind"] == json!("king")
                && p["file"] == json!(5)
                && p["rank"] == json!(9)
        }));
        // 攻め方（実戦の先手）の玉 5九 → 5一
        assert!(board.iter().any(|p| {
            p["color"] == json!("gote")
                && p["kind"] == json!("king")
                && p["file"] == json!(5)
                && p["rank"] == json!(1)
        }));
        // 両者とも角を1枚ずつ持っている
        assert_eq!(q["sente_hand"], json!([{ "kind": "bishop", "count": 1 }]));
        assert_eq!(q["gote_hand"], json!([{ "kind": "bishop", "count": 1 }]));
        assert_eq!(board.len(), 38);
    }

    /// 攻め方が後手の局面は回さない（受け方 = 先手がそのまま手前）
    #[test]
    fn gote_attacker_is_not_rotated() {
        let mut pos = Position::initial();
        play(&mut pos, &["7g7f", "3c3d", "8h2b+", "3a2b", "6i7h"]);
        let q = question_from(&pos, Color::Gote).expect("問題が作れる");
        let board = q["board"].as_array().unwrap();
        assert!(board.iter().any(|p| {
            p["color"] == json!("sente")
                && p["kind"] == json!("king")
                && p["file"] == json!(5)
                && p["rank"] == json!(9)
        }));
    }

    #[test]
    fn accept_requires_escape_range_and_a_losing_first_move() {
        let args = default_args();
        let ok = json!({
            "verdict": "escape", "checks": 3, "minFouls": 1, "rootTries": 6, "rootSolutions": 2
        });
        assert!(accept(&args, &ok));
        let mut v = ok.clone();
        v["verdict"] = json!("caught");
        assert!(!accept(&args, &v));
        let mut v = ok.clone();
        v["checks"] = json!(1);
        assert!(!accept(&args, &v));
        let mut v = ok.clone();
        v["checks"] = json!(9);
        assert!(!accept(&args, &v));
        // どの初手でも逃れられる問題は既定では捨てる
        let mut v = ok.clone();
        v["rootSolutions"] = json!(6);
        assert!(!accept(&args, &v));
        let mut trivial = default_args();
        trivial.allow_trivial = true;
        assert!(accept(&trivial, &v));
        // 初手の統計が無い（打ち切り）ものも捨てる
        let mut v = ok.clone();
        v["rootTries"] = json!(null);
        assert!(!accept(&args, &v));
    }

    #[test]
    fn only_the_longest_escape_of_each_game_and_defender_is_taken() {
        // (対局/受け方, 手目, 王手の回数, 反則回数)
        let accepted = vec![
            ("game-a/sente", 40, 2, 0),
            ("game-a/sente", 42, 3, 0),
            ("game-a/sente", 44, 3, 1),
            ("game-b/gote", 51, 2, 1),
        ];
        let taken = take_per_game(
            &default_args(),
            accepted,
            |(game, _, _, _)| game.to_string(),
            |(_, _, checks, fouls)| (*checks, *fouls),
        );
        assert_eq!(taken, vec![("game-a/sente", 44, 3, 1), ("game-b/gote", 51, 2, 1)]);
    }
}
