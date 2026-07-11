//! 序盤定跡ブック。
//!
//! ついたて将棋では序盤の自分の手は相手から見えないため、駒がぶつかるまでの
//! 数手は読みではなく方針の質で決まる。一方、対局をまたいで挙動が固定だと
//! 人間の相手に学習・搾取される（対人50局で飛車先歩交換の型を突かれた実績）ので、
//! 複数ラインからゲームごとに無作為に選ぶ。
//!
//! ブックを抜ける条件（以後その対局では戻らない）:
//! - どちらかの駒取りが発生した / 王手宣言があった / 自分が反則した
//! - ブックの手が自分の候補手に存在しない（駒が想定位置にない）
//!
//! ラインは知識として手で登録する（所有者の定跡を反映する場所）。
//! 先手視点で書き、後手用は点対称にミラーする。

use std::collections::HashSet;

use rand::Rng;

use crate::board::parse_usi_square;
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{Color, PlayerView};
use crate::shogi::{ShogiMove, parse_usi};

/// 組み込みの定跡ライン（joseki.json が見つからないときのフォールバック）。
/// 正本は joseki.json（tools/joseki-editor.html で編集・エクスポートする）
const BUILTIN_LINES: [&[&str]; 4] = [
    // 居飛車速攻（所有者定跡: 基本中の基本）。2六歩〜2三歩成まで一直線。
    // 最後の歩成で駒取りが発生し、その観測でブックを抜けて通常思考に戻る
    &["2g2f", "2f2e", "2e2d", "2d2c+"],
    // 玉を右に逃がして金銀で蓋をする（仮ライン）
    &["5i4h", "4h3h", "7i6h", "5g5f"],
    // 中住まい風（仮ライン）
    &["5i5h", "3i4h", "7i6h", "5g5f"],
    // 左に囲う（仮ライン）
    &["5i6h", "7i7h", "6h7i", "5g5f"],
];

/// 定跡ラインの読み込み（プロセス内で1回だけ）。
/// TSUITATE_JOSEKI（既定 joseki.json）の {"lines":[{"name","moves":[usi...]}]} を読む。
/// パースできない手を含むラインは警告してスキップする
fn load() -> &'static (Vec<String>, Vec<Vec<String>>) {
    static LOADED: std::sync::OnceLock<(Vec<String>, Vec<Vec<String>>)> =
        std::sync::OnceLock::new();
    LOADED.get_or_init(|| {
        let path = std::env::var("TSUITATE_JOSEKI").unwrap_or_else(|_| "joseki.json".into());
        if let Ok(content) = std::fs::read_to_string(&path) {
            match serde_json::from_str::<serde_json::Value>(&content) {
                Ok(v) => {
                    let mut names = vec![];
                    let mut lines = vec![];
                    for line in v["lines"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
                        let moves: Vec<String> = line["moves"]
                            .as_array()
                            .map(|a| {
                                a.iter()
                                    .filter_map(|m| m.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default();
                        if moves.is_empty() || moves.iter().any(|u| parse_usi(u).is_none()) {
                            eprintln!("定跡ラインを解釈できずスキップ: {:?}", line["name"]);
                            continue;
                        }
                        names.push(line["name"].as_str().unwrap_or("?").to_string());
                        lines.push(moves);
                    }
                    if !lines.is_empty() {
                        return (names, lines);
                    }
                    eprintln!("{path} に有効なラインがないため組み込み定跡を使います");
                }
                Err(e) => eprintln!("{path} をパースできません（組み込み定跡を使用）: {e}"),
            }
        }
        (
            (1..=BUILTIN_LINES.len()).map(|i| format!("組み込み{i}")).collect(),
            BUILTIN_LINES
                .iter()
                .map(|l| l.iter().map(|s| s.to_string()).collect())
                .collect(),
        )
    })
}

fn lines() -> &'static Vec<Vec<String>> {
    &load().1
}

fn line_names() -> &'static Vec<String> {
    &load().0
}

/// USI手を点対称にミラーする（先手ライン → 後手用）
fn mirror_usi(usi: &str) -> Option<String> {
    let mv = parse_usi(usi)?;
    let flip = |c: crate::board::Coord| crate::board::Coord {
        file: 10 - c.file,
        rank: 10 - c.rank,
    };
    let mirrored = match mv {
        ShogiMove::Board { from, to, promote } => ShogiMove::Board {
            from: flip(from),
            to: flip(to),
            promote,
        },
        ShogiMove::Drop { role, to } => ShogiMove::Drop { role, to: flip(to) },
    };
    Some(mirrored.to_usi())
}

pub struct OpeningBook {
    /// 対局開始時に選んだライン（自色向けにミラー済み）
    line: Vec<String>,
    /// ブックから抜けたら true（以後戻らない）
    exited: bool,
}

impl OpeningBook {
    /// 指定インデックスのラインに固定したブック（定跡特化チューニング用）
    pub fn with_line(my_color: Color, index: usize) -> Self {
        let all = lines();
        let raw = &all[index % all.len()];
        let line = raw
            .iter()
            .filter_map(|usi| match my_color {
                Color::Sente => Some(usi.clone()),
                Color::Gote => mirror_usi(usi),
            })
            .collect();
        OpeningBook {
            line,
            exited: false,
        }
    }

    /// ライン名（joseki.json の name）からインデックスを引く
    pub fn line_index(name: &str) -> Option<usize> {
        line_names().iter().position(|n| n == name)
    }

    pub fn new(my_color: Color) -> Self {
        let all = lines();
        // 既定はランダム選択（対局をまたいで人間に順番を読まれないため）。
        // TSUITATE_BOOK_RR=1 のときは巡回選択にする: SPSA（bin/tune）の
        // f+/f− 評価で定跡分布を揃え、「どの定跡を引いたか」のノイズが
        // 勾配推定を汚さないようにする（共通乱数法）
        static RR: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let idx = if std::env::var("TSUITATE_BOOK_RR").as_deref() == Ok("1") {
            RR.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % all.len()
        } else {
            rand::rng().random_range(0..all.len())
        };
        let raw = &all[idx];
        let line = raw
            .iter()
            .filter_map(|usi| match my_color {
                Color::Sente => Some(usi.clone()),
                Color::Gote => mirror_usi(usi),
            })
            .collect();
        OpeningBook {
            line,
            exited: false,
        }
    }

    /// ブックの次の一手。None ならブックを抜けた（通常思考へ）
    pub fn next(
        &mut self,
        view: &PlayerView,
        log: &ObservationLog,
        foul_tried: &HashSet<String>,
    ) -> Option<String> {
        if self.exited {
            return None;
        }
        // 静かな序盤でなくなったら抜ける
        let quiet = log.events().iter().all(|e| match e {
            Observation::MyMove { captured, .. } => captured.is_none(),
            Observation::OpponentMoved {
                captured_my_piece_at,
                ..
            } => captured_my_piece_at.is_none(),
            Observation::MyFoul { .. } | Observation::Check { .. } => false,
            Observation::OpponentFoul { .. } => true, // 相手の反則は情報にならない
        });
        if !quiet || view.you_in_check {
            self.exited = true;
            return None;
        }
        // 自分が何手指したか = ラインの進行位置
        let my_moves = log
            .events()
            .iter()
            .filter(|e| matches!(e, Observation::MyMove { .. }))
            .count();
        let Some(usi) = self.line.get(my_moves) else {
            self.exited = true; // ライン消化完了
            return None;
        };
        if foul_tried.contains(usi.as_str()) {
            self.exited = true;
            return None;
        }
        // 自分の駒が想定位置にいるか（自分に見える範囲の妥当性チェック）
        let playable = match parse_usi(usi) {
            Some(ShogiMove::Board { from, to, .. }) => {
                let from_ok = view
                    .your_pieces
                    .iter()
                    .any(|p| parse_usi_square(&p.square) == Some(from));
                let to_free = !view
                    .your_pieces
                    .iter()
                    .any(|p| parse_usi_square(&p.square) == Some(to));
                from_ok && to_free
            }
            _ => false, // 定跡ラインに打ちは想定しない
        };
        if !playable {
            self.exited = true;
            return None;
        }
        Some(usi.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirror_flips_point_symmetric() {
        assert_eq!(mirror_usi("5i4h").unwrap(), "5a6b");
        assert_eq!(mirror_usi("7g7f").unwrap(), "3c3d");
    }

    #[test]
    fn book_plays_line_then_exits() {
        // フル初期配置（どのラインの初手も指せる）
        let view = crate::strategy::tests::minimal_view(
            crate::shogi::Position::initial().pieces_of(Color::Sente),
            std::collections::HashMap::new(),
        );
        let log = ObservationLog::default();
        let mut book = OpeningBook::new(Color::Sente);
        let mv = book.next(&view, &log, &HashSet::new());
        assert!(mv.is_some(), "初手はブックから出るはず");
        let usi = mv.unwrap();
        assert!(
            lines().iter().any(|l| l[0] == usi),
            "いずれかのラインの初手: {usi}"
        );
    }

    /// 読み込まれた全ライン（joseki.json があればそれ）が自視点の盤で
    /// 最後まで指し切れることを検証する。駒が想定位置にない・利きが通らない・
    /// 成りの指定が不正、のいずれかがあると失敗してライン名と手を報告する
    #[test]
    fn loaded_lines_replay_on_own_view() {
        use crate::board::{Promotion, make_usi_square, move_targets, promotion_choice};
        use crate::shogi::promote_role;

        for (li, line) in lines().iter().enumerate() {
            // 先手の初期配置（自駒のみ）
            let mut pieces = crate::shogi::Position::initial()
                .pieces_of(Color::Sente);
            for usi in line {
                let Some(ShogiMove::Board { from, to, promote }) = parse_usi(usi) else {
                    panic!("ライン{li} の {usi} が盤上の手ではない");
                };
                let idx = pieces
                    .iter()
                    .position(|p| parse_usi_square(&p.square) == Some(from))
                    .unwrap_or_else(|| panic!("ライン{li} の {usi}: 移動元に駒がない"));
                let targets = move_targets(&pieces, &pieces[idx], Color::Sente);
                assert!(
                    targets.contains(&to),
                    "ライン{li} の {usi}: その駒はそこへ動けない"
                );
                let choice = promotion_choice(pieces[idx].role, from, to, Color::Sente);
                if promote {
                    assert!(
                        choice != Promotion::None,
                        "ライン{li} の {usi}: 成れない手に + が付いている"
                    );
                    pieces[idx].role = promote_role(pieces[idx].role)
                        .unwrap_or_else(|| panic!("ライン{li} の {usi}: 成れない駒種"));
                } else {
                    assert!(
                        choice != Promotion::Forced,
                        "ライン{li} の {usi}: 成りが強制の手なのに + がない"
                    );
                }
                pieces[idx].square = make_usi_square(to);
            }
        }
    }

    #[test]
    fn book_exits_on_capture() {
        let view = crate::strategy::tests::minimal_view(vec![], std::collections::HashMap::new());
        let mut log = ObservationLog::default();
        log.record(Observation::OpponentMoved {
            move_number: 2,
            captured_my_piece_at: Some("5e".into()),
        });
        let mut book = OpeningBook::new(Color::Sente);
        assert!(book.next(&view, &log, &HashSet::new()).is_none());
    }
}
