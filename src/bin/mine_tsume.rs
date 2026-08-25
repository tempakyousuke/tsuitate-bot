//! 自動対局の記録から「ついたて詰将棋の問題」を掘る（詰めチャレの問題生成）。
//!
//! サイト（tsuitate）の**詰めチャレ**は、1問ずつレート連動で詰将棋を出題する
//! モード。その問題プールを人手の投稿だけで賄うのは無理なので、bot の自動対局
//! （アリーナ or 実戦の記録）に現れた局面のうち**実際に詰みがある**ものを
//! ソルバーで拾い出し、サイトの取り込みJSONに書き出す。
//!
//! 使い方:
//!   cargo run --release --bin mine_tsume -- <記録ファイル/ディレクトリ...> [オプション]
//!
//! オプション:
//!   --out <path>            取り込みJSONの出力先（既定: 標準出力）
//!   --solver <path>         ソルバーCLI のパス（既定: 環境変数 SOLVER_BIN）
//!   --source <name>         取り込み元識別子（既定: selfplay）
//!   --min-depth N           採用する詰み手数の下限（既定: 3）
//!   --max-depth N           採用する詰み手数の上限（既定: 15）
//!   --skip-plies N          この手数までの局面は無視する（既定: 20）
//!   --limit N               採用する問題数の上限（0 = 無制限）
//!   --max-per-game N        1対局から採る問題数の上限（既定: 1、0 = 無制限）
//!   --max-candidates N      ソルバーに投げる候補数の上限（0 = 無制限）
//!   --jobs N                ソルバーの並列実行数（既定: 1）
//!   --timeout-secs N        ソルバーの1問あたりのタイムアウト（既定: 60）
//!   --memory-limit-mb N     ソルバーのピークRSS上限（既定: 2000）
//!   --node-limit N          ソルバーのノード上限（既定: 20,000,000）
//!   --rating-node-limit N   レート推定の初手ごとのノード予算（既定: 200,000）
//!   --allow-second          余詰めのある問題も採用する（既定: 捨てる）
//!   --dump-candidates <path> 候補の問題JSONだけを書き出してソルバーを呼ばない
//!
//! ## 問題の作り方
//!
//! ある決定点（手番側 = 攻め方）の**実戦の局面をそのまま**問題にする。盤上の駒は
//! 両者とも全部残し、攻め方の持ち駒もそのまま渡す。玉方の持ち駒は
//! 「全駒 − 盤上 − 攻め方持ち駒」でソルバー側が自動算出するので、結果として
//! 実戦の玉方の持ち駒と一致する（＝局面が完全に復元される）。
//!
//! **攻め方の玉も落とさない**（双玉）。自玉があると駒が動かせない（ピン）・
//! 逆王手が受けの手段になる、といった実戦特有の制約がそのまま効くため、
//! 詰将棋の慣習で玉を外すより問題として味が出る。ソルバーもサイトも双玉を
//! 扱える（ソルバーの `generate_legal_moves` は自玉があれば自殺手を除外し、
//! 自玉がなければ擬似合法手をそのまま返す）。
//!
//! 玉方が攻め方になる局面（後手番）は、盤面を180度回して先手番に直す
//! （ソルバーは攻め方 = 先手を前提にしている）。駒の向きも回転で入れ替わるので、
//! 色のラベルを差し替えるだけでよい。
//!
//! **1対局・1攻め方から採るのは1問だけ**（`--max-per-game`、既定 1）。詰みのある
//! 局面は一度現れると、攻め方が見逃したまま数手続くことが多く、その各手番の局面を
//! 全部採ると「同じ対局の2手違い」のほとんど同じ問題が並んでしまう。盤面の署名
//! による重複排除では別局面なので落ちない。対局内で**最初に詰みが現れた局面**
//! （最も手前の決定点）を採る。攻め方が違えば（先手に詰みがあった後、見逃して
//! 後手に詰みが回った等）盤の向きも駒の構成も別なので、別問題として許容する。

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_json::{json, Value};
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

/// 攻め方が後手のときは盤面を180度回して先手番の問題にする
fn orient(c: Coord, flip: bool) -> (u8, u8) {
    if flip {
        ((10 - c.file) as u8, (10 - c.rank) as u8)
    } else {
        (c.file as u8, c.rank as u8)
    }
}

struct Candidate {
    /// ソルバー入力そのもの（{"board": [...], "sente_hand": [...]}）
    question: Value,
    /// 正規化JSONの SHA-256 先頭16桁。重複排除と取り込みの upsert キーを兼ねる
    signature: String,
    /// どの対局（記録ファイル名）のどちらの攻め方から採ったか。`--max-per-game` の単位
    game: String,
    /// どの記録の何手目から採ったか（出典。デバッグ・再現用）
    origin: String,
}

/// 決定点の真の局面から、攻め方（手番側）視点のついたて詰将棋の問題を作る。
///
/// 盤上の駒は両者とも全部残す（双玉）。玉方の持ち駒はソルバー側が
/// 「全駒 − 盤上 − 攻め方持ち駒」で自動算出するため JSON には入れない
fn question_from(pos: &Position, attacker: Color) -> Option<Value> {
    let defender = attacker.other();
    pos.king_square(defender)?;
    // 実戦の局面なら手番が回ってきた時点で相手玉が王手になっていることはない
    // （攻め方の手番なのに王手＝詰将棋として不正で、サイト側の盤面検証も
    // ERR_OPPOSITE_CHECK で弾く）。壊れた棋譜由来のデータへの保険
    if pos.in_check(defender) {
        return None;
    }
    let flip = attacker == Color::Gote;

    let mut board: Vec<(u8, u8, &'static str, &'static str)> = Vec::new();
    for (color, label) in [(attacker, "sente"), (defender, "gote")] {
        for piece in pos.pieces_of(color) {
            let coord = parse_usi_square(&piece.square)?;
            let (file, rank) = orient(coord, flip);
            board.push((file, rank, label, role_kind(piece.role)));
        }
    }
    board.sort_by_key(|&(file, rank, color, _)| (file, rank, color));

    let hands = pos.hand_map(attacker);
    let hand_json: Vec<Value> = HAND_ORDER
        .iter()
        .filter_map(|&role| {
            let count = *hands.get(&role)?;
            (count > 0).then(|| json!({ "kind": role_kind(role), "count": count }))
        })
        .collect();

    Some(json!({
        "board": board
            .into_iter()
            .map(|(file, rank, color, kind)| json!({
                "file": file, "rank": rank, "color": color, "kind": kind
            }))
            .collect::<Vec<_>>(),
        "sente_hand": hand_json,
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
    min_depth: u32,
    max_depth: u32,
    skip_plies: u32,
    limit: usize,
    max_per_game: usize,
    max_candidates: usize,
    jobs: usize,
    timeout_secs: u64,
    memory_limit_mb: u64,
    node_limit: u64,
    rating_node_limit: u64,
    allow_second: bool,
    dump_candidates: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        inputs: vec![],
        out: None,
        solver: std::env::var("SOLVER_BIN").ok(),
        source: "selfplay".into(),
        min_depth: 3,
        max_depth: 15,
        skip_plies: 20,
        limit: 0,
        max_per_game: 1,
        max_candidates: 0,
        jobs: 1,
        timeout_secs: 60,
        memory_limit_mb: 2000,
        node_limit: 20_000_000,
        rating_node_limit: 200_000,
        allow_second: false,
        dump_candidates: None,
    };
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        let mut value = |name: &str| -> Result<String, String> {
            iter.next().ok_or_else(|| format!("{name} には値が必要です"))
        };
        match arg.as_str() {
            "--out" => args.out = Some(value("--out")?),
            "--solver" => args.solver = Some(value("--solver")?),
            "--source" => args.source = value("--source")?,
            "--min-depth" => args.min_depth = value("--min-depth")?.parse().map_err(|_| "--min-depth が数値ではありません".to_string())?,
            "--max-depth" => args.max_depth = value("--max-depth")?.parse().map_err(|_| "--max-depth が数値ではありません".to_string())?,
            "--skip-plies" => args.skip_plies = value("--skip-plies")?.parse().map_err(|_| "--skip-plies が数値ではありません".to_string())?,
            "--limit" => args.limit = value("--limit")?.parse().map_err(|_| "--limit が数値ではありません".to_string())?,
            "--max-per-game" => args.max_per_game = value("--max-per-game")?.parse().map_err(|_| "--max-per-game が数値ではありません".to_string())?,
            "--max-candidates" => args.max_candidates = value("--max-candidates")?.parse().map_err(|_| "--max-candidates が数値ではありません".to_string())?,
            "--jobs" => args.jobs = value("--jobs")?.parse().map_err(|_| "--jobs が数値ではありません".to_string())?,
            "--timeout-secs" => args.timeout_secs = value("--timeout-secs")?.parse().map_err(|_| "--timeout-secs が数値ではありません".to_string())?,
            "--memory-limit-mb" => args.memory_limit_mb = value("--memory-limit-mb")?.parse().map_err(|_| "--memory-limit-mb が数値ではありません".to_string())?,
            "--node-limit" => args.node_limit = value("--node-limit")?.parse().map_err(|_| "--node-limit が数値ではありません".to_string())?,
            "--rating-node-limit" => args.rating_node_limit = value("--rating-node-limit")?.parse().map_err(|_| "--rating-node-limit が数値ではありません".to_string())?,
            "--allow-second" => args.allow_second = true,
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
            found.push(Candidate {
                question,
                signature,
                game: format!("{name}/{}", if side == Color::Sente { "sente" } else { "gote" }),
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

/// ソルバーCLI を1問走らせて出力JSONを返す
fn run_solver(args: &Args, solver: &str, candidate: &Candidate, slot: usize) -> Option<Value> {
    let tmp = std::env::temp_dir().join(format!("mine-tsume-{}-{}.json", std::process::id(), slot));
    if let Err(e) = std::fs::write(&tmp, serde_json::to_vec(&candidate.question).ok()?) {
        eprintln!("一時ファイルを書けません: {e}");
        return None;
    }
    let output = std::process::Command::new(solver)
        .arg(&tmp)
        .arg("--shortest")
        .arg("--find-second")
        .arg("--estimate-rating")
        .arg("--timeout-secs")
        .arg(args.timeout_secs.to_string())
        .arg("--memory-limit-mb")
        .arg(args.memory_limit_mb.to_string())
        .arg("--node-limit")
        .arg(args.node_limit.to_string())
        .arg("--rating-node-limit")
        .arg(args.rating_node_limit.to_string())
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

/// 採用判定（詰みがあり、手数が範囲内で、レート推定が付いていること）
fn accept(args: &Args, result: &Value) -> bool {
    if result["found"] != json!(true) {
        return false;
    }
    let Some(depth) = result["depth"].as_u64() else {
        return false;
    };
    if (depth as u32) < args.min_depth || (depth as u32) > args.max_depth {
        return false;
    }
    if !args.allow_second && result["hasSecondSolution"] == json!(true) {
        return false;
    }
    // レート推定が無いと詰めチャレの初期レートを決められない
    result["rating"]["value"].is_number()
}

/// 採用済みの問題から、対局×攻め方ごとの上限（`--max-per-game`）を超えるものを落とす。
///
/// 候補は対局内で手番順に並んでいるので、残るのは**最初に詰みが現れた局面**。
/// 詰みは一度現れると攻め方が見逃したまま数手続くことが多く、盤面署名の
/// 重複排除だけでは「同じ対局の2手違い」がほとんど同じ問題として並んでしまう。
/// 攻め方が違えば別問題として許容する（先手の詰みと後手の詰みは盤の向きも
/// 駒の構成も別物）
fn take_per_game<T>(args: &Args, accepted: Vec<T>, game_of: impl Fn(&T) -> String) -> Vec<T> {
    let mut per_game: HashMap<String, usize> = HashMap::new();
    accepted
        .into_iter()
        .filter(|item| {
            let count = per_game.entry(game_of(item)).or_insert(0);
            if args.max_per_game > 0 && *count >= args.max_per_game {
                return false;
            }
            *count += 1;
            true
        })
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

    let accepted: Vec<(&Candidate, Value)> = candidates
        .iter()
        .zip(results)
        .filter_map(|(candidate, result)| {
            let result = result.into_inner().unwrap()?;
            accept(&args, &result).then_some((candidate, result))
        })
        .collect();
    let accepted_total = accepted.len();
    let accepted = take_per_game(&args, accepted, |(candidate, _)| candidate.game.clone());
    if accepted.len() < accepted_total {
        eprintln!(
            "同じ対局・同じ攻め方の問題を間引き: {}件 → {}件（--max-per-game {}）",
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
            "depth": result["depth"],
            "hasSecond": result["hasSecondSolution"] == json!(true),
            "hasPieceSurplus": result["hasPieceSurplus"] == json!(true),
            "kizuCount": result["kizuCount"],
            "rating": result["rating"]["value"],
            "ratingFormula": result["rating"]["formula"],
            "ratingFeatures": result["rating"]["features"],
            "solution": result["tree"],
            "solverMessage": result["message"],
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

    /// 実戦の局面をそのまま問題にする（盤上の駒は両者とも全部残す＝双玉）
    #[test]
    fn question_keeps_the_whole_position() {
        let pos = Position::initial();
        let q = question_from(&pos, Color::Sente).expect("問題が作れる");
        let board = q["board"].as_array().unwrap();
        assert_eq!(board.len(), 40, "平手初期局面の全駒");
        assert_eq!(board.iter().filter(|p| p["color"] == json!("gote")).count(), 20);
        // 攻め方の玉も残す（双玉。ピンや逆王手といった実戦の制約がそのまま効く）
        assert!(board.iter().any(|p| p["color"] == json!("sente") && p["kind"] == json!("king")));
        assert!(board.iter().any(|p| p["color"] == json!("gote") && p["kind"] == json!("king")));
    }

    /// 玉方の持ち駒は JSON に入れない（ソルバーが残り駒から自動算出し、
    /// 実戦の玉方の持ち駒と一致する）
    #[test]
    fn defender_hand_is_left_to_the_solver() {
        let mut pos = Position::initial();
        // ▲7六歩 △3四歩 ▲2二角成（後手の角を取る）
        for usi in ["7g7f", "3c3d", "8h2b+"] {
            pos.play_unchecked(&tsuitate_bot::shogi::parse_usi(usi).unwrap());
        }
        // 後手番なので攻め方は後手。攻め方（後手）の持ち駒は空
        let q = question_from(&pos, Color::Gote).expect("問題が作れる");
        assert_eq!(q["sente_hand"].as_array().unwrap().len(), 0);

        // 先手を攻め方にすると、取った角が攻め方の持ち駒として入る
        pos.play_unchecked(&tsuitate_bot::shogi::parse_usi("3a2b").unwrap());
        let q = question_from(&pos, Color::Sente).expect("問題が作れる");
        assert_eq!(q["sente_hand"], json!([{ "kind": "bishop", "count": 1 }]));
    }

    /// 後手視点は盤面を180度回して「攻め方 = 先手」に直す
    #[test]
    fn gote_question_is_rotated() {
        let pos = Position::initial();
        let q = question_from(&pos, Color::Gote).expect("問題が作れる");
        let board = q["board"].as_array().unwrap();
        // 先手玉 5九 → 180度回して 5一の「玉方の玉」になる
        assert!(board.iter().any(|p| {
            p["color"] == json!("gote")
                && p["kind"] == json!("king")
                && p["file"] == json!(5)
                && p["rank"] == json!(1)
        }));
        // 後手の飛車 8二 → 2八の「攻め方の飛車」
        assert!(board.iter().any(|p| {
            p["color"] == json!("sente")
                && p["kind"] == json!("rook")
                && p["file"] == json!(2)
                && p["rank"] == json!(8)
        }));
    }

    /// 手番側の相手が既に王手を受けている局面は問題にしない
    /// （攻め方の手番なのに王手＝詰将棋として不正）
    #[test]
    fn positions_with_the_defender_already_in_check_are_rejected() {
        let mut pos = Position::initial();
        for usi in ["7g7f", "3c3d"] {
            pos.play_unchecked(&tsuitate_bot::shogi::parse_usi(usi).unwrap());
        }
        assert!(question_from(&pos, Color::Sente).is_some());

        // ▲8八角×3三成: 3三の馬が 4二（空きマス）を通って 5一玉に王手
        pos.play_unchecked(&tsuitate_bot::shogi::parse_usi("8h3c+").unwrap());
        assert!(question_from(&pos, Color::Sente).is_none());
        // 逆に後手を攻め方とみなす分には（先手玉は王手されていないので）問題になる
        assert!(question_from(&pos, Color::Gote).is_some());
    }

    #[test]
    fn signature_is_stable_and_distinguishes_positions() {
        let pos = Position::initial();
        let a = question_from(&pos, Color::Sente).unwrap();
        let b = question_from(&pos, Color::Sente).unwrap();
        assert_eq!(signature_of(&a), signature_of(&b));
        // 平手初期局面は点対称なので、後手視点を180度回すと先手視点と一致する
        assert_eq!(signature_of(&a), signature_of(&question_from(&pos, Color::Gote).unwrap()));

        // 局面をそのまま問題にするので、どちらの駒が動いても別問題になる
        let mut moved = pos.clone();
        moved.play_unchecked(&tsuitate_bot::shogi::parse_usi("7g7f").unwrap());
        assert_ne!(signature_of(&a), signature_of(&question_from(&moved, Color::Sente).unwrap()));
        assert_ne!(signature_of(&a), signature_of(&question_from(&moved, Color::Gote).unwrap()));
    }

    #[test]
    fn accept_requires_depth_range_and_rating() {
        let args = parse_args_for_test();
        let ok = json!({
            "found": true, "depth": 5, "hasSecondSolution": false,
            "rating": { "value": 1500 }
        });
        assert!(accept(&args, &ok));
        assert!(!accept(&args, &json!({ "found": false, "depth": 5, "rating": { "value": 1500 } })));
        assert!(!accept(&args, &json!({ "found": true, "depth": 1, "rating": { "value": 1500 } })));
        assert!(!accept(&args, &json!({ "found": true, "depth": 99, "rating": { "value": 1500 } })));
        assert!(!accept(&args, &json!({ "found": true, "depth": 5, "rating": null })));
        assert!(!accept(
            &args,
            &json!({ "found": true, "depth": 5, "hasSecondSolution": true, "rating": { "value": 1500 } })
        ));
    }

    /// 同じ対局・同じ攻め方からは最初に詰みが現れた局面だけを採る
    /// （詰みは見逃されたまま数手続くので、全部採るとほぼ同じ問題が並ぶ）。
    /// 攻め方が違えば別問題
    #[test]
    fn only_the_first_mate_of_each_game_and_attacker_is_taken() {
        let accepted = vec![
            ("game-a/sente", 40),
            ("game-a/sente", 42),
            ("game-b/gote", 51),
            ("game-a/sente", 44),
            ("game-a/gote", 45),
            ("game-b/gote", 53),
            ("game-a/gote", 47),
        ];
        let mut args = parse_args_for_test();
        let taken = take_per_game(&args, accepted.clone(), |(game, _)| game.to_string());
        assert_eq!(taken, vec![("game-a/sente", 40), ("game-b/gote", 51), ("game-a/gote", 45)]);

        // 上限を広げれば手番順にその数だけ採る
        args.max_per_game = 2;
        let taken = take_per_game(&args, accepted.clone(), |(game, _)| game.to_string());
        assert_eq!(
            taken,
            vec![
                ("game-a/sente", 40),
                ("game-a/sente", 42),
                ("game-b/gote", 51),
                ("game-a/gote", 45),
                ("game-b/gote", 53),
                ("game-a/gote", 47),
            ]
        );

        // 0 は無制限（従来の挙動）
        args.max_per_game = 0;
        let taken = take_per_game(&args, accepted.clone(), |(game, _)| game.to_string());
        assert_eq!(taken, accepted);
    }

    fn parse_args_for_test() -> Args {
        Args {
            inputs: vec![],
            out: None,
            solver: None,
            source: "test".into(),
            min_depth: 3,
            max_depth: 15,
            skip_plies: 20,
            limit: 0,
            max_per_game: 1,
            max_candidates: 0,
            jobs: 1,
            timeout_secs: 60,
            memory_limit_mb: 2000,
            node_limit: 20_000_000,
            rating_node_limit: 200_000,
            allow_second: false,
            dump_candidates: None,
        }
    }
}
