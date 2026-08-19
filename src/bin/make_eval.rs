//! 棋譜から採点式評価ファイル（evals/*.eval.md）のスケルトンを生成・追記する。
//!
//! usage:
//!   cargo run --release --bin make_eval -- [--ply N] <kifへのパス> [eval出力パス] [top_n=15] [seed=0]
//!
//! `--ply N` で決定点を1つ（N 手まで再生した局面 = N+1 手目。kif の
//! `*scenario ply=` と同じ数え方）に絞る（単一決定点のシナリオ kif、
//! 例 scenarios/arena-check-*.kif 用。推定器の構築はその ply までで済む）。
//! 省略時は従来どおり全決定点。
//!
//! 各決定点（反則後サブ状態を含む）について現行 estimator の候補上位 top_n と
//! 実戦手を列挙し、eval ファイルに `和名(USI) ?` の行として書く。**冪等**:
//! 既存の eval に対しては未収載の候補だけをブロック末尾へ追記し、採点・
//! コメント・自由記述はそのまま保つ。新しい棋譜のレビューを始めるときの
//! 入口（形式は evals/README.md）。
//!
//! 実行は重い（全決定点で choose を回す = 対局1局ぶん）。進捗は stderr。

use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::PathBuf;

use tsuitate_bot::kifu::parse_kif;
use tsuitate_bot::observation::{Observation, ObservationLog};
use tsuitate_bot::protocol::Role;
use tsuitate_bot::scenario_core::{clone_log, make_view, replay, resolve_foul, side_idx};
use tsuitate_bot::shogi::{Position, ShogiMove, parse_usi};
use tsuitate_bot::strategy::{self, Strategy};

fn kanji_rank(r: i8) -> char {
    "一二三四五六七八九".chars().nth(r as usize - 1).unwrap()
}

fn role_name(role: Role) -> &'static str {
    match role {
        Role::Pawn => "歩",
        Role::Lance => "香",
        Role::Knight => "桂",
        Role::Silver => "銀",
        Role::Gold => "金",
        Role::Bishop => "角",
        Role::Rook => "飛",
        Role::King => "玉",
        Role::Tokin => "と",
        Role::Promotedlance => "成香",
        Role::Promotedknight => "成桂",
        Role::Promotedsilver => "成銀",
        Role::Horse => "馬",
        Role::Dragon => "龍",
    }
}

/// doc の表記に合わせた日本語表記（rank_dump と同じ規約）
fn jp_move(pos: &Position, usi: &str) -> String {
    let Some(mv) = parse_usi(usi) else {
        return usi.to_string();
    };
    match mv {
        ShogiMove::Board { from, to, promote } => {
            let name = pos.piece_at(from).map(|p| role_name(p.role)).unwrap_or("?");
            format!(
                "{}{}{}{}",
                to.file,
                kanji_rank(to.rank),
                name,
                if promote { "成" } else { "" }
            )
        }
        ShogiMove::Drop { role, to } => {
            format!("{}{}{}打", to.file, kanji_rank(to.rank), role_name(role))
        }
    }
}

/// eval ファイルのブロック集合（挿入位置と収載済み USI）
struct EvalDoc {
    lines: Vec<String>,
}

/// ブロックキー: (手数, 反則後の見出し文字列 or None)
type Key = (usize, Option<String>);

impl EvalDoc {
    fn parse_key(line: &str) -> Option<Key> {
        if let Some(rest) = line.strip_prefix("### ") {
            let num: usize = rest.split("手目").next()?.parse().ok()?;
            let sub = rest.split('（').nth(1)?.strip_suffix("の反則後）")?;
            return Some((num, Some(sub.to_string())));
        }
        if let Some(rest) = line.strip_prefix("## ") {
            let num: usize = rest.split("手目").next()?.parse().ok()?;
            return Some((num, None));
        }
        None
    }

    /// key のブロック範囲（開始見出し行, 終了 = 次見出し行 or EOF）
    fn block_bounds(&self, key: &Key) -> Option<(usize, usize)> {
        let mut start = None;
        for (i, line) in self.lines.iter().enumerate() {
            match Self::parse_key(line) {
                Some(k) if start.is_none() && k == *key => start = Some(i),
                Some(_) if start.is_some() => return Some((start.unwrap(), i)),
                _ => {}
            }
        }
        start.map(|s| (s, self.lines.len()))
    }

    /// ブロック内に収載済みの USI 集合
    fn known_usis(&self, bounds: (usize, usize)) -> HashSet<String> {
        let mut out = HashSet::new();
        for line in &self.lines[bounds.0..bounds.1] {
            if let Some(open) = line.find('(') {
                if let Some(close) = line[open..].find(')') {
                    let usi = &line[open + 1..open + close];
                    if parse_usi(usi).is_some() {
                        out.insert(usi.to_string());
                    }
                }
            }
        }
        out
    }

    /// key のブロックへ候補行を追記（無いブロックは末尾に作る）。追記数を返す
    fn append(&mut self, key: &Key, header: &str, rows: &[String]) -> usize {
        match self.block_bounds(key) {
            Some(bounds) => {
                let known = self.known_usis(bounds);
                let fresh: Vec<String> = rows
                    .iter()
                    .filter(|r| {
                        // 行の (USI) を取り出して既知と照合
                        let open = r.find('(').unwrap();
                        let close = r[open..].find(')').unwrap();
                        !known.contains(&r[open + 1..open + close])
                    })
                    .cloned()
                    .collect();
                if fresh.is_empty() {
                    return 0;
                }
                // ブロック末尾の空行の直前へ挿入
                let mut p = bounds.1;
                while p > bounds.0 && self.lines[p - 1].trim().is_empty() {
                    p -= 1;
                }
                let n = fresh.len();
                self.lines.splice(p..p, fresh);
                n
            }
            None => {
                if !self.lines.last().is_none_or(|l| l.trim().is_empty()) {
                    self.lines.push(String::new());
                }
                self.lines.push(header.to_string());
                self.lines.extend(rows.iter().cloned());
                rows.len()
            }
        }
    }
}

struct Chain {
    strat: Box<dyn Strategy>,
    running: ObservationLog,
    consumed: usize,
}

fn candidates_of(snap: &dyn Strategy, chosen: &Option<String>, top_n: usize) -> Vec<String> {
    match snap.last_ranking() {
        Some(ranking) => ranking.iter().take(top_n).map(|c| c.usi.clone()).collect(),
        None => chosen.iter().cloned().collect(), // 定跡または候補ゼロ
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut only_ply: Option<usize> = None;
    if let Some(i) = args.iter().position(|a| a == "--ply") {
        let v = args.get(i + 1).and_then(|v| v.parse().ok());
        let Some(v) = v else {
            eprintln!("--ply には数値が必要です");
            std::process::exit(2);
        };
        only_ply = Some(v);
        args.drain(i..i + 2);
    }
    if args.is_empty() {
        eprintln!("usage: make_eval [--ply N] <kifへのパス> [eval出力パス] [top_n=15] [seed=0]");
        std::process::exit(2);
    }
    let kif_path = PathBuf::from(&args[0]);
    let eval_path = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let stem = kif_path.file_stem().unwrap().to_string_lossy();
            PathBuf::from("evals").join(format!("{stem}.eval.md"))
        });
    let top_n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15);
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);

    let text = std::fs::read_to_string(&kif_path).expect("kif を読めません");
    let kifu = parse_kif(&text).expect("kif をパースできません");
    let mut doc = EvalDoc {
        lines: match std::fs::read_to_string(&eval_path) {
            Ok(t) => t.split('\n').map(str::to_string).collect(),
            Err(_) => vec![
                format!(
                    "# eval: {}",
                    kif_path.file_stem().unwrap().to_string_lossy()
                ),
                "# 点数: 0〜10（10=最善級 8=あり 6=ぎりぎり 4=悪手だがマシ 2=悪手 0=論外）。未採点は ?".into(),
                String::new(),
            ],
        },
    };

    let mut chains: [Option<Chain>; 2] = [None, None];
    let mut added = 0usize;
    let last = kifu.plies.len();
    let (m_from, m_to) = match only_ply {
        Some(p) => {
            assert!(p < last, "--ply {p} が棋譜の手数 {last} 以上です");
            (p + 1, p + 1)
        }
        None => (1, last),
    };
    for m in m_from..=m_to {
        let ply = m - 1;
        let rep = replay(&kifu, ply);
        let side = rep.pos.turn();
        let idx = side_idx(side);
        let chain = chains[idx].get_or_insert_with(|| Chain {
            strat: strategy::make_seeded("estimator", seed).expect("estimator を作れません"),
            running: ObservationLog::default(),
            consumed: 0,
        });
        let view = make_view(&rep.pos, side, &rep.fouls);
        let log = &rep.logs[idx];
        let events = log.events();
        while chain.consumed < events.len() {
            let e = &events[chain.consumed];
            if matches!(e, Observation::MyMove { .. } | Observation::MyFoul { .. }) {
                chain.strat.prewarm(&view, &chain.running);
            }
            chain.running.record(e.clone());
            chain.consumed += 1;
        }
        let real_fouls = kifu.plies[ply].fouls.clone();
        let actual = kifu.plies[ply].mv.to_usi();

        // 反則前の決定
        {
            let mut snap = chain
                .strat
                .clone_boxed()
                .expect("estimator は clone_boxed 対応のはず");
            let chosen = snap.choose(&view, log, &HashSet::new());
            let mut usis = candidates_of(&*snap, &chosen, top_n);
            // 実戦手（と反則手）は必ず収載する
            for u in real_fouls
                .iter()
                .map(|raw| resolve_foul(&rep.pos, side, raw))
                .chain([actual.clone()])
            {
                if !usis.contains(&u) {
                    usis.push(u);
                }
            }
            let rows: Vec<String> = usis
                .iter()
                .map(|u| format!("{}({u}) ?", jp_move(&rep.pos, u)))
                .collect();
            let mut header = format!("## {m}手目");
            let _ = write!(header, "（{}番）", if idx == 0 { "先手" } else { "後手" });
            added += doc.append(&(m, None), &header, &rows);
        }

        // 反則後のサブ決定（rank_dump と同じ規約で再現）
        if !real_fouls.is_empty() {
            let mut log2 = clone_log(log);
            let mut fouls_arr = rep.fouls;
            let mut foul_tried: HashSet<String> = HashSet::new();
            for raw in &real_fouls {
                let usi = resolve_foul(&rep.pos, side, raw);
                fouls_arr[idx] += 1;
                log2.record(Observation::MyFoul {
                    move_number: rep.pos.move_number(),
                    usi: usi.clone(),
                });
                foul_tried.insert(usi.clone());
                let view2 = make_view(&rep.pos, side, &fouls_arr);
                let mut snap = chain
                    .strat
                    .clone_boxed()
                    .expect("estimator は clone_boxed 対応のはず");
                let chosen = snap.choose(&view2, &log2, &foul_tried);
                let mut usis = candidates_of(&*snap, &chosen, top_n);
                if !usis.contains(&actual) {
                    usis.push(actual.clone());
                }
                let name = jp_move(&rep.pos, &usi);
                let rows: Vec<String> = usis
                    .iter()
                    .map(|u| format!("{}({u}) ?", jp_move(&rep.pos, u)))
                    .collect();
                let header = format!("### {m}手目（{name}({usi})の反則後）");
                added += doc.append(&(m, Some(format!("{name}({usi})"))), &header, &rows);
            }
        }
        eprintln!("[{m}/{last}] 完了（追記 {added}）");
    }

    if let Some(dir) = eval_path.parent() {
        std::fs::create_dir_all(dir).expect("evals ディレクトリを作れません");
    }
    let mut out = doc.lines.join("\n");
    if !out.ends_with('\n') {
        out.push('\n');
    }
    std::fs::write(&eval_path, out).expect("eval を書けません");
    println!("{}: {added} 行追記", eval_path.display());
}
