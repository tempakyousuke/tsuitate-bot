//! **被王手の決定点の採掘**: 対局記録（アリーナ `arena-records` / 本番 records）の
//! 真実を再生し、bot 側（記録の `match.your_color`）が王手を受けた決定点から
//! シナリオ候補を2種類抽出する（2026-08-19、ユーザー依頼）:
//!
//! - **foul**: 被王手の手番で反則を `--min-fouls`（既定3）回以上した決定点
//! - **flee**: 王手駒を**玉以外の駒で取れた**（真実の合法手に存在した）のに、
//!   取らずに玉を動かした決定点（取った場合の取り返しの有無も併記）
//!
//! 出力は1行1候補の TSV（標準出力）。`--emit <dir> <接頭辞>` を付けると候補ごとに
//! `scenarios/` 形式の kif（`*scenario ply=N target=... desc=...`）を書き出す。
//! 真実を見るので審判側のツール（bot の観測ではない）。
//!
//! usage:
//!   cargo run --release --bin mine_check -- [--min-fouls N] [--emit <dir> <接頭辞>]
//!       [--max N] <records/*.jsonl...>
//!
//! 書き出した kif は `bin/scenario <名前>` / `--oracle-lag 1` でそのまま使える。
//! flee の target は「王手駒を取る手」（複数あれば最も安い駒の手）、foul の
//! target は実際に受理された手。

use std::collections::HashMap;

use tsuitate_bot::board::make_usi_square;
use tsuitate_bot::kifu::kif_body;
use tsuitate_bot::protocol::Color;
use tsuitate_bot::shogi::{ShogiMove, piece_value, unpromote_role};
use tsuitate_bot::truth_replay::{for_each_decision, load_end};

fn usage() -> &'static str {
    "usage: cargo run --release --bin mine_check -- [--min-fouls N] [--emit <dir> <接頭辞>] [--max N] <records/*.jsonl...>"
}

fn exit_usage(msg: &str) -> ! {
    eprintln!("{msg}");
    eprintln!("{}", usage());
    std::process::exit(1);
}

/// 記録ファイルの bot 側の色（`match` 行の your_color）
fn bot_color(path: &str) -> Option<Color> {
    let content = std::fs::read_to_string(path).ok()?;
    for line in content.lines() {
        let v: serde_json::Value = serde_json::from_str(line).ok()?;
        if v["type"] == "match" {
            return match v["your_color"].as_str()? {
                "sente" => Some(Color::Sente),
                "gote" => Some(Color::Gote),
                _ => None,
            };
        }
    }
    None
}

struct Hit {
    kind: &'static str,
    file: String,
    /// 決定点の ply（= *scenario ply=）
    ply: usize,
    side: Color,
    /// 王手駒（マス・駒種）
    checkers: Vec<String>,
    /// この手番の反則列
    fouls: Vec<String>,
    /// 実際に受理された手
    actual: String,
    /// シナリオの target（flee は王手駒を取る手、foul は実際の手）
    target: String,
    /// 合法な解消手数 K
    k: usize,
    /// flee: 取った駒の価値と、取った後に取り返されるか
    note: String,
    moves: Vec<String>,
    foul_attempts: Vec<(u32, String)>,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut min_fouls = 3u32;
    let mut emit: Option<(String, String)> = None;
    let mut max_emit = usize::MAX;
    let mut files: Vec<String> = vec![];
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--min-fouls" => {
                min_fouls = args
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| exit_usage("--min-fouls には数値が必要です"));
                i += 2;
            }
            "--emit" => {
                let dir = args.get(i + 1).cloned().unwrap_or_else(|| exit_usage("--emit <dir> <接頭辞>"));
                let prefix = args.get(i + 2).cloned().unwrap_or_else(|| exit_usage("--emit <dir> <接頭辞>"));
                emit = Some((dir, prefix));
                i += 3;
            }
            "--max" => {
                max_emit = args
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| exit_usage("--max には数値が必要です"));
                i += 2;
            }
            _ => {
                files.push(args[i].clone());
                i += 1;
            }
        }
    }
    if files.is_empty() {
        exit_usage("記録ファイルを1つ以上指定してください");
    }

    let mut hits: Vec<Hit> = vec![];
    let mut games = 0usize;
    let mut check_decisions = 0usize;
    for path in &files {
        let Some(end) = load_end(path) else {
            eprintln!("{path}: end が無いので飛ばします");
            continue;
        };
        let Some(bot) = bot_color(path) else {
            eprintln!("{path}: match 行が無いので飛ばします");
            continue;
        };
        games += 1;
        let moves: Vec<String> = end.moves.iter().map(|m| m.usi.clone()).collect();
        let foul_attempts: Vec<(u32, String)> = end
            .foul_attempts
            .iter()
            .map(|f| (f.move_number, f.usi.clone()))
            .collect();
        let ok = for_each_decision(&end, |pos, side, _log, decision_id| {
            if side != bot || !pos.in_check(side) {
                return;
            }
            check_decisions += 1;
            let ply = decision_id as usize;
            let actual = moves[ply].clone();
            let fouls: Vec<String> = end
                .foul_attempts
                .iter()
                .filter(|f| f.by_color == side && f.move_number == pos.move_number())
                .map(|f| f.usi.clone())
                .collect();
            let my_king = pos.king_square(side).expect("自玉がない");
            let opp = side.other();
            let checkers: Vec<(tsuitate_bot::board::Coord, tsuitate_bot::protocol::Role)> = pos
                .pieces()
                .filter(|(c, p)| p.color == opp && pos.attacks(*c, my_king))
                .map(|(c, p)| (c, p.role))
                .collect();
            let checker_str: Vec<String> = checkers
                .iter()
                .map(|(c, r)| format!("{}{:?}", make_usi_square(*c), r))
                .collect();
            let legal = pos.legal_moves();
            let k = legal.len();
            let base = |kind: &'static str, target: String, note: String| Hit {
                kind,
                file: path.clone(),
                ply,
                side,
                checkers: checker_str.clone(),
                fouls: fouls.clone(),
                actual: actual.clone(),
                target,
                k,
                note,
                moves: moves.clone(),
                foul_attempts: foul_attempts.clone(),
            };
            if fouls.len() as u32 >= min_fouls {
                hits.push(base("foul", actual.clone(), String::new()));
            }
            // flee: 玉以外の駒で王手駒を取る合法手があったのに玉を動かした
            let actual_mv = tsuitate_bot::shogi::parse_usi(&actual).expect("USI");
            // 玉で王手駒を取った手は「逃げ」ではないので除く
            let actual_is_king_flee = matches!(
                actual_mv,
                ShogiMove::Board { from, to, .. }
                    if from == my_king && !checkers.iter().any(|(c, _)| *c == to)
            );
            if actual_is_king_flee && checkers.len() == 1 {
                let (csq, crole) = checkers[0];
                // 最も安い駒での捕獲を代表にする
                let mut best: Option<(f64, ShogiMove)> = None;
                for mv in &legal {
                    if let ShogiMove::Board { from, to, .. } = mv {
                        if *to == csq && *from != my_king {
                            let v = piece_value(unpromote_role(pos.piece_at(*from).unwrap().role));
                            if best.as_ref().is_none_or(|(bv, _)| v < *bv) {
                                best = Some((v, mv.clone()));
                            }
                        }
                    }
                }
                if let Some((v, mv)) = best {
                    // 取った後に取り返されるか（真実）
                    let mut after = pos.clone();
                    after.play_unchecked(&mv);
                    let recaptured = after.is_attacked(csq, opp);
                    let note = format!(
                        "王手駒 {:?}（価値{}）を価値{v}の駒で取れた・取り返し{}",
                        crole,
                        piece_value(unpromote_role(crole)),
                        if recaptured { "あり" } else { "なし" }
                    );
                    hits.push(base("flee", mv.to_usi(), note));
                }
            }
        });
        if !ok {
            eprintln!("{path}: 棋譜が壊れているので飛ばしました");
        }
    }

    eprintln!(
        "{games} 局 / bot 被王手決定点 {check_decisions} / 候補 foul {} 件・flee {} 件",
        hits.iter().filter(|h| h.kind == "foul").count(),
        hits.iter().filter(|h| h.kind == "flee").count()
    );
    println!("kind\tfile\tply\tside\tcheckers\tK\tfouls\tactual\ttarget\tnote");
    for h in &hits {
        println!(
            "{}\t{}\t{}\t{:?}\t{}\t{}\t{}\t{}\t{}\t{}",
            h.kind,
            h.file,
            h.ply,
            h.side,
            h.checkers.join(","),
            h.k,
            h.fouls.join(","),
            h.actual,
            h.target,
            h.note
        );
    }

    if let Some((dir, prefix)) = emit {
        std::fs::create_dir_all(&dir).expect("出力ディレクトリを作れません");
        let mut per_kind: HashMap<&str, usize> = HashMap::new();
        let mut written = 0usize;
        for h in &hits {
            if written >= max_emit {
                break;
            }
            let n = per_kind.entry(h.kind).or_insert(0);
            *n += 1;
            let name = format!("{prefix}-{}{:02}", h.kind, n);
            let game = std::path::Path::new(&h.file)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let desc = match h.kind {
                "foul" => format!(
                    "アリーナ自己対局 {game} の{}手目（bot={:?}）: 被王手（王手駒 {}）で反則 {} 回（{}）のあと {} を指した。K={}",
                    h.ply + 1,
                    h.side,
                    h.checkers.join("/"),
                    h.fouls.len(),
                    h.fouls.join(" "),
                    h.actual,
                    h.k
                ),
                _ => format!(
                    "アリーナ自己対局 {game} の{}手目（bot={:?}）: 被王手（王手駒 {}）で {}（{}）のに玉を {} と逃げた。反則 {} 回（{}）。K={}",
                    h.ply + 1,
                    h.side,
                    h.checkers.join("/"),
                    h.target,
                    h.note,
                    h.actual,
                    h.fouls.len(),
                    h.fouls.join(" "),
                    h.k
                ),
            };
            // *scenario 行は先頭（sync_eval.py は1行目をヘッダとして書き換える）
            let mut out = format!(
                "*scenario ply={} target={} desc={}\n",
                h.ply,
                h.target,
                desc.replace('\n', " ")
            );
            out.push_str("棋戦：arena\n手合割：平手\n先手：先手\n後手：後手\n手数----指手---------消費時間--\n");
            let ending = if h.foul_attempts.iter().any(|(mn, _)| *mn as usize > h.moves.len()) {
                Some("反則負け")
            } else {
                None
            };
            match kif_body(&h.moves, &h.foul_attempts, ending) {
                Ok(body) => out.push_str(&body),
                Err(e) => {
                    eprintln!("{name}: KIF生成に失敗: {e}");
                    continue;
                }
            }
            let path = std::path::Path::new(&dir).join(format!("{name}.kif"));
            std::fs::write(&path, out).expect("kif を書けません");
            eprintln!("書き出し: {}", path.display());
            written += 1;
        }
    }
}
