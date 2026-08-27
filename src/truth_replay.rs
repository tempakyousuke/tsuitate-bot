//! 対局記録の**真実**（`game:end` の全手順＋反則試行）から、両者の観測列を
//! 再構成して決定点を1つずつ渡す共通部品。
//!
//! 観測の作り方（順序・move_number 規約・王手宣言の両者通知）は
//! `selfplay.rs` の審判と一致させること。ここがズレると「学習データの観測」と
//! 「実戦の観測」が食い違い、特徴量の意味が変わる。

use crate::board::make_usi_square;
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{Color, GameEndPayload, Role};
use crate::shogi::{Position, ShogiMove, parse_usi, unpromote_role};

pub fn side_idx(c: Color) -> usize {
    match c {
        Color::Sente => 0,
        Color::Gote => 1,
    }
}

/// 反則試行1回ぶんの観測を**両者**のログへ記録し、累積反則数を進める。
///
/// 手番は変わらない（サイトの judge.ts と同じ）。`bin/mate_continue` の
/// オフライン方策が真実で非合法な手を選んだときも、審判と同じ規約で
/// 反則を積むためにここを共有する。
pub fn add_foul_obs(
    side: Color,
    usi: String,
    pos: &Position,
    logs: &mut [ObservationLog; 2],
    fouls: &mut [u32; 2],
) {
    fouls[side_idx(side)] += 1;
    logs[side_idx(side)].record(Observation::MyFoul {
        move_number: pos.move_number(),
        usi,
    });
    logs[side_idx(side.other())].record(Observation::OpponentFoul {
        count: fouls[side_idx(side)],
    });
}

/// 受理された1手ぶんの観測を**両者**のログへ記録する（`pos` は着手後の局面）。
///
/// 順序と move_number 規約は `selfplay.rs` の審判と同じ。一手だけ強制して
/// 継続する診断（`bin/mate_continue`、issue #28 P0-6）が同じ規約で状態を
/// 作れるように公開してある（ここを別実装にすると観測が食い違う）。
pub fn add_move_obs(
    side: Color,
    mv: &ShogiMove,
    captured: Option<Role>,
    pos: &Position,
    logs: &mut [ObservationLog; 2],
) {
    let move_number = pos.move_number();
    let captured_sq = captured.map(|_| match *mv {
        ShogiMove::Board { to, .. } => make_usi_square(to),
        ShogiMove::Drop { .. } => unreachable!("打ちは駒を取れない"),
    });
    logs[side_idx(side)].record(Observation::MyMove {
        move_number,
        usi: mv.to_usi(),
        captured: captured.map(unpromote_role),
    });
    logs[side_idx(side.other())]
        .record(Observation::OpponentMoved {
            move_number,
            captured_my_piece_at: captured_sq,
        });
    if pos.in_check(pos.turn()) {
        let in_check = pos.turn();
        for log in logs.iter_mut() {
            log.record(Observation::Check { in_check });
        }
    }
}

/// 決定点の状態（`for_each_decision_full` が渡す）。
/// checkpoint arena の抽出はここから「開始盤面・両者ログ・累積反則数・絶対手数」を取る
pub struct DecisionState<'a> {
    /// 真の局面（手番側がこれから指す時点）
    pub pos: &'a Position,
    /// 手番側
    pub side: Color,
    /// **両者**の観測列 [0]=先手, [1]=後手
    pub logs: &'a [ObservationLog; 2],
    /// 累積反則数 [0]=先手, [1]=後手（この決定点までに試した反則を含む）
    pub fouls: &'a [u32; 2],
    /// この決定点で手番側が既に試した反則の数。
    /// **0 なら手番境界**（checkpoint arena v1 が採る唯一の checkpoint 種別）
    pub fouls_this_turn: u32,
    /// 絶対手数（受理された手の数）。`Position::move_number()` とは別に数える
    pub plies: u32,
    /// 決定点の通し番号（`end.moves` のインデックス）
    pub decision_id: u64,
}

/// 決定点（手番側がこれから指す時点）ごとにコールバックを呼ぶ。
/// 渡すのは (真の局面, 手番側, その側の観測列, 決定点の通し番号)。
/// 棋譜が壊れていたら false を返す（呼び出し側はその局を丸ごと捨てること）。
pub fn for_each_decision(
    end: &GameEndPayload,
    mut f: impl FnMut(&Position, Color, &ObservationLog, u64),
) -> bool {
    for_each_decision_full(end, |d| {
        f(d.pos, d.side, &d.logs[side_idx(d.side)], d.decision_id)
    })
}

/// `for_each_decision` の全量版: **両者**の観測列・累積反則数・絶対手数を渡す。
/// 片側のログしか要らない学習データ生成は `for_each_decision` を使う。
///
/// checkpoint arena はここから状態を取るが、**デッキには KIF を保存する**
/// （`scenario_core::replay` で復元でき、回帰局面をそのまま
/// `bin/scenario ... diag` / `rank_probe` に流せるため）。
/// 両者が同じ状態になることは checkpoint_arena 側のテストで検査する
pub fn for_each_decision_full(
    end: &GameEndPayload,
    mut f: impl FnMut(DecisionState<'_>),
) -> bool {
    let mut pos = Position::initial();
    let mut logs = [ObservationLog::default(), ObservationLog::default()];
    let mut fouls = [0u32; 2];
    let mut fouls_sorted = end.foul_attempts.clone();
    fouls_sorted.sort_by_key(|f| f.move_number);
    let mut foul_idx = 0usize;

    for (decision_id, m) in end.moves.iter().enumerate() {
        let side = pos.turn();
        // 反則は着手より前に起きている（手番は変わらない）
        let mut fouls_this_turn = 0u32;
        while foul_idx < fouls_sorted.len()
            && fouls_sorted[foul_idx].move_number == pos.move_number()
            && fouls_sorted[foul_idx].by_color == side
        {
            add_foul_obs(
                side,
                fouls_sorted[foul_idx].usi.clone(),
                &pos,
                &mut logs,
                &mut fouls,
            );
            foul_idx += 1;
            fouls_this_turn += 1;
        }

        f(DecisionState {
            pos: &pos,
            side,
            logs: &logs,
            fouls: &fouls,
            fouls_this_turn,
            plies: decision_id as u32,
            decision_id: decision_id as u64,
        });

        let Some(mv) = parse_usi(&m.usi) else {
            return false;
        };
        if m.by_color != side || !pos.is_legal(&mv) {
            return false;
        }
        let captured = pos.play_unchecked(&mv);
        add_move_obs(side, &mv, captured, &pos, &mut logs);
    }
    true
}

/// 記録ファイル（JSONL）から (bot 側の手番色, 終局ペイロード) を読む。
///
/// `bin/analyze` は同じ JSONL からもっと多くを読む（chose の debug 等）が、
/// 真実の再生だけが要る診断（`bin/mate_probe` / `bin/mate_continue`）は
/// この2つで足りる。どちらか欠けていたら None（その局は捨てる）。
pub fn load_bot_and_end(path: &str) -> Option<(Color, GameEndPayload)> {
    let content = std::fs::read_to_string(path).ok()?;
    let mut color = None;
    let mut end = None;
    for line in content.lines() {
        let v: serde_json::Value = serde_json::from_str(line).ok()?;
        match v["type"].as_str() {
            Some("match") => color = serde_json::from_value(v["your_color"].clone()).ok(),
            Some("end") => end = serde_json::from_value(v["payload"].clone()).ok(),
            _ => {}
        }
    }
    Some((color?, end?))
}

/// 記録ファイル（JSONL）から終局ペイロード（真実）を読む
pub fn load_end(path: &str) -> Option<GameEndPayload> {
    let content = std::fs::read_to_string(path).ok()?;
    let mut end = None;
    for line in content.lines() {
        let v: serde_json::Value = serde_json::from_str(line).ok()?;
        if v["type"] == "end" {
            end = serde_json::from_value(v["payload"].clone()).ok();
        }
    }
    end
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{FoulRecord, MoveRecord, OpponentInfo, RatingChange, RatingChangePair};

    fn end_of(moves: &[(&str, Color)], fouls: &[(u32, Color, &str)]) -> GameEndPayload {
        GameEndPayload {
            result: "draw".into(),
            reason: "test".into(),
            final_sfen: String::new(),
            moves: moves
                .iter()
                .map(|&(usi, by_color)| MoveRecord {
                    usi: usi.into(),
                    by_color,
                    ms: 0,
                    fouls_before: 0,
                })
                .collect(),
            foul_attempts: fouls
                .iter()
                .map(|&(move_number, by_color, usi)| FoulRecord {
                    move_number,
                    by_color,
                    usi: usi.into(),
                })
                .collect(),
            rating_change: RatingChangePair {
                you: RatingChange { before: 0, after: 0 },
                opponent: RatingChange { before: 0, after: 0 },
            },
            opponent: OpponentInfo {
                username: String::new(),
                rating: 0,
                is_bot: true,
            },
        }
    }

    #[test]
    fn decisions_alternate_and_logs_stay_side_local() {
        let end = end_of(
            &[
                ("7g7f", Color::Sente),
                ("3c3d", Color::Gote),
                ("8h2b+", Color::Sente),
            ],
            &[(2, Color::Gote, "5a5b")],
        );
        let mut seen: Vec<(Color, usize)> = vec![];
        assert!(for_each_decision(&end, |_pos, side, log, _id| {
            seen.push((side, log.events().len()));
        }));
        assert_eq!(seen.len(), 3);
        assert_eq!(seen[0].0, Color::Sente);
        assert_eq!(seen[0].1, 0, "初手の時点では観測ゼロ");
        assert_eq!(seen[1].0, Color::Gote);
        // 後手の視点: 相手の着手1つ + 自分の反則1つ
        assert_eq!(seen[1].1, 2);
        assert_eq!(seen[2].0, Color::Sente);
        // 先手の視点: 自分の着手 + 相手の反則宣言 + 相手の着手
        assert_eq!(seen[2].1, 3);
    }

    #[test]
    fn broken_kifu_is_rejected() {
        let end = end_of(&[("9i9a", Color::Sente)], &[]);
        assert!(!for_each_decision(&end, |_, _, _, _| {}));
    }
}
