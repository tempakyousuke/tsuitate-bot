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
//! ある決定点（手番側 = 攻め方）で、盤上に残すのは**攻め方の駒と玉方の玉だけ**。
//! 玉方の残りの駒は、ついたて詰将棋の慣習どおり全部「玉方の持ち駒」になる
//! （ソルバー側が盤上と攻め方持ち駒の残りから自動算出する）。つまり実戦の
//! 局面そのものではなく、そこから作った独立した問題になる。攻め方の玉は
//! 詰将棋では置かないので落とす。
//!
//! 玉方が攻め方になる局面（後手番）は、盤面を180度回して先手番に直す
//! （ソルバーは攻め方 = 先手を前提にしている）。

use std::collections::HashSet;
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
    /// どの記録の何手目から採ったか（出典。デバッグ・再現用）
    origin: String,
}

/// 決定点の真の局面から、攻め方（手番側）視点のついたて詰将棋の問題を作る。
///
/// 玉方の駒を玉だけにすると、消えた駒が塞いでいた利きが通って**初形で王手**に
/// なることがある（攻め方の手番なのに王手 = 詰将棋として不正。サイト側の
/// 盤面検証も ERR_OPPOSITE_CHECK で弾く）。そういう局面は問題にしない
fn question_from(pos: &Position, attacker: Color) -> Option<Value> {
    let defender = attacker.other();
    let king = pos.king_square(defender)?;
    let flip = attacker == Color::Gote;

    // 攻め方の駒（玉を除く）と玉方の玉だけを残した盤で王手になっていないか
    let reduced = pos.retain_pieces(|_, piece| {
        if piece.color == attacker {
            piece.role != Role::King
        } else {
            piece.role == Role::King
        }
    });
    if reduced.in_check(defender) {
        return None;
    }

    let mut board: Vec<(u8, u8, &'static str, &'static str)> = Vec::new();
    for piece in pos.pieces_of(attacker) {
        // 攻め方の玉は詰将棋では盤に置かない
        if piece.role == Role::King {
            continue;
        }
        let coord = parse_usi_square(&piece.square)?;
        let (file, rank) = orient(coord, flip);
        board.push((file, rank, "sente", role_kind(piece.role)));
    }
    if board.is_empty() {
        return None;
    }
    let (kf, kr) = orient(king, flip);
    board.push((kf, kr, "gote", "king"));
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

    let mut puzzles: Vec<Value> = Vec::new();
    for (candidate, result) in candidates.iter().zip(results) {
        let Some(result) = result.into_inner().unwrap() else {
            continue;
        };
        if !accept(&args, &result) {
            continue;
        }
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

    /// 初期局面の先手視点: 攻め方の玉は落とし、玉方は玉だけが残る
    #[test]
    fn sente_question_keeps_own_pieces_and_enemy_king_only() {
        let pos = Position::initial();
        let q = question_from(&pos, Color::Sente).expect("問題が作れる");
        let board = q["board"].as_array().unwrap();
        let gote: Vec<&Value> = board.iter().filter(|p| p["color"] == json!("gote")).collect();
        assert_eq!(gote.len(), 1, "玉方は玉だけ");
        assert_eq!(gote[0]["kind"], json!("king"));
        assert_eq!(gote[0]["file"], json!(5));
        assert_eq!(gote[0]["rank"], json!(1));
        // 先手の駒 20 枚 - 玉 1 枚
        assert_eq!(board.len() - 1, 19);
        assert!(board.iter().all(|p| p["color"] == json!("gote") || p["kind"] != json!("king")));
    }

    /// 後手視点は盤面を180度回して「攻め方 = 先手」に直す
    #[test]
    fn gote_question_is_rotated() {
        let pos = Position::initial();
        let q = question_from(&pos, Color::Gote).expect("問題が作れる");
        let board = q["board"].as_array().unwrap();
        let gote: Vec<&Value> = board.iter().filter(|p| p["color"] == json!("gote")).collect();
        assert_eq!(gote.len(), 1);
        // 先手玉 5九 → 180度回して 5一
        assert_eq!(gote[0]["file"], json!(5));
        assert_eq!(gote[0]["rank"], json!(1));
        // 後手の飛車 8二 → 2八
        assert!(board.iter().any(|p| {
            p["color"] == json!("sente")
                && p["kind"] == json!("rook")
                && p["file"] == json!(2)
                && p["rank"] == json!(8)
        }));
    }

    /// 玉方の駒を落としたことで玉が王手にさらされる局面は問題にしない
    #[test]
    fn positions_already_in_check_after_reduction_are_rejected() {
        let mut pos = Position::initial();
        // ▲7六歩 △3四歩 ▲2二角成 …ではなく、後手の駒を落とすと角の利きが
        // 玉に通る形を直接作る: 先手角を5五へ、後手玉は5一のまま。
        // 平手初形では 5五角は 5一玉へ縦に利かないので、角道が通る斜めに置く
        for usi in ["7g7f", "3c3d"] {
            pos.play_unchecked(&tsuitate_bot::shogi::parse_usi(usi).unwrap());
        }
        // ここまでは角が 8八 にいて玉に利かないので問題になる
        assert!(question_from(&pos, Color::Sente).is_some());

        // ▲8八角×3三成: 3三の馬は 4二 を通って 5一玉へ利く。
        // 玉方の駒（4二金など）を落とすと初形で王手になってしまう
        pos.play_unchecked(&tsuitate_bot::shogi::parse_usi("8h3c+").unwrap());
        assert!(question_from(&pos, Color::Sente).is_none());
    }

    #[test]
    fn signature_is_stable_and_distinguishes_positions() {
        let pos = Position::initial();
        let a = question_from(&pos, Color::Sente).unwrap();
        let b = question_from(&pos, Color::Sente).unwrap();
        assert_eq!(signature_of(&a), signature_of(&b));
        // 平手初期局面は点対称なので、後手視点を180度回すと先手視点と一致する
        let mirrored = question_from(&pos, Color::Gote).unwrap();
        assert_eq!(signature_of(&a), signature_of(&mirrored));

        // 問題は「攻め方の駒 + 玉方の玉」だけなので、攻め方の駒が動けば別問題になる。
        // 逆に玉方の駒だけが動いた局面は同じ署名になり、重複として落ちる（狙いどおり）
        let mut moved = pos.clone();
        moved.play_unchecked(&tsuitate_bot::shogi::parse_usi("7g7f").unwrap());
        assert_ne!(signature_of(&a), signature_of(&question_from(&moved, Color::Sente).unwrap()));
        assert_eq!(signature_of(&a), signature_of(&question_from(&moved, Color::Gote).unwrap()));
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
