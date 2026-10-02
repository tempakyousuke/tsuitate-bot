//! 対局記録（JSONL）の真実を `Referee` に流し直して、試行ごとの決定点を取り出す
//! （M2a の模倣学習の教師。docs/rl-deepnash-design.md）。
//!
//! `truth_replay::for_each_decision_full` は受理手ごとに1回しか呼ばないが、ここは
//! **反則した試行も1件ずつ**渡す（反則も教師がその情報で選んだ手なので教師になる）。
//! 観測は arena と同じ `Referee` が作るので、学習データの観測と実戦の観測は同じコードの出力になる。

use crate::observation::Observation;
use crate::protocol::{Color, GameEndPayload};
use crate::referee::{Referee, StepResult};
use crate::selfplay::GameResult;

/// 試行1回（手番側が指そうとした手）
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attempt {
    pub side: Color,
    pub usi: String,
    /// 反則だった試行か
    pub foul: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayError {
    /// 真実の手順が審判の裁定と食い違う（受理のはずが反則、など）
    Mismatch { index: usize, usi: String, got: String },
    /// 終局後に試行が残っている / 終局しないまま手順が尽きた
    Length,
    /// 記録の勝敗・終局理由が、再生した審判の裁定と食い違う（または読めない）
    Outcome { recorded: String, replayed: String },
}

/// 審判が自分で判定する終局理由（再生した審判の裁定と照合できるもの）
const REFEREE_REASONS: [&str; 4] = ["checkmate", "stalemate", "foul_limit", "max_plies"];

/// 記録の勝敗を、再生し終えた審判の裁定と照合して返す。
///
/// - 審判が判定する終局（詰み・ステイルメイト・反則負け・手数上限）: 勝敗と理由が
///   再生の裁定と完全に一致すること
/// - 審判の外の終局（時間切れ・投了など）: 再生した審判がまだ終局していないこと
///
/// 価値の教師は勝敗から作るので、ここを通らない記録は使わない
/// （勝者の取り違え・読めない結果文字列・途中で切れた記録が価値の教師を汚す）
pub fn verify_outcome(end: &GameEndPayload, replayed: &Referee) -> Result<GameResult, ReplayError> {
    let mismatch = || ReplayError::Outcome {
        recorded: format!("{} / {}", end.result, end.reason),
        replayed: format!("{:?}", replayed.ended()),
    };
    let result = parse_result(&end.result).ok_or_else(mismatch)?;
    let ok = if REFEREE_REASONS.contains(&end.reason.as_str()) {
        matches!(replayed.ended(), Some((r, why)) if r == result && why == end.reason)
    } else {
        replayed.ended().is_none()
    };
    if ok { Ok(result) } else { Err(mismatch()) }
}

/// 終局ペイロードの結果文字列 → 勝敗
pub fn parse_result(result: &str) -> Option<GameResult> {
    match result {
        "sente_win" => Some(GameResult::Win(Color::Sente)),
        "gote_win" => Some(GameResult::Win(Color::Gote)),
        "draw" => Some(GameResult::Draw),
        _ => None,
    }
}

/// 真実の試行を時系列に並べる。反則は同じ手数・同じ手番の着手より前に起きている
/// （反則は手番を変えない）。ファイル内の反則の順序は保つ
pub fn attempts(end: &GameEndPayload) -> Vec<Attempt> {
    let mut fouls = end.foul_attempts.clone();
    fouls.sort_by_key(|f| f.move_number); // 安定ソート
    let mut fouls = fouls.into_iter().peekable();
    let mut out = vec![];
    for (k, m) in end.moves.iter().enumerate() {
        let move_number = k as u32 + 1;
        while let Some(f) = fouls.next_if(|f| f.move_number == move_number) {
            out.push(Attempt {
                side: f.by_color,
                usi: f.usi,
                foul: true,
            });
        }
        out.push(Attempt {
            side: m.by_color,
            usi: m.usi.clone(),
            foul: false,
        });
    }
    // 反則負けで終わった局は最後の手数の反則が着手なしで残る
    out.extend(fouls.map(|f| Attempt {
        side: f.by_color,
        usi: f.usi,
        foul: true,
    }));
    out
}

/// 真実の試行を審判に流す。各試行の**直前**の審判（手番側の視界・観測・その手番の
/// 反則試行がそろった状態）で `f` を呼ぶ。裁定が真実と食い違ったら Err
pub fn replay(
    end: &GameEndPayload,
    mut f: impl FnMut(&Referee, &Attempt),
) -> Result<Referee, ReplayError> {
    let mut referee = Referee::new();
    let attempts = attempts(end);
    let last = attempts.len().saturating_sub(1);
    for (i, a) in attempts.iter().enumerate() {
        if referee.ended().is_some() {
            return Err(ReplayError::Length);
        }
        if referee.to_move() != a.side {
            return Err(ReplayError::Mismatch {
                index: i,
                usi: a.usi.clone(),
                got: "手番違い".into(),
            });
        }
        f(&referee, a);
        let r = referee.step(&a.usi, 0);
        let ok = match r {
            StepResult::Foul => a.foul,
            StepResult::Accepted => !a.foul,
            StepResult::Ended { move_accepted, .. } => move_accepted != a.foul && i == last,
        };
        if !ok {
            return Err(ReplayError::Mismatch {
                index: i,
                usi: a.usi.clone(),
                got: format!("{r:?}"),
            });
        }
    }
    Ok(referee)
}

/// 記録ファイルの中身から、記録した側（`your_color`）の観測イベントを読む
/// （再生した観測が記録と一致するかの検査用）
pub fn recorded_observations(content: &str) -> Vec<Observation> {
    content
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|v| v["type"] == "obs")
        .filter_map(|v| serde_json::from_value(v["event"].clone()).ok())
        .collect()
}

/// 再生した観測が記録と一致するか（serde の表現で比べる）
pub fn same_observations(a: &[Observation], b: &[Observation]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            serde_json::to_value(x).ok() == serde_json::to_value(y).ok()
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{OpponentInfo, RatingChange, RatingChangePair};
    use crate::rl::env::random_action;
    use crate::rl::action::{decode_usi, legal_mask};
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn end_payload(referee: &mut Referee, result: GameResult, reason: &str) -> GameEndPayload {
        let truth = referee.take_truth();
        let zero = RatingChange { before: 0, after: 0 };
        GameEndPayload {
            result: match result {
                GameResult::Win(Color::Sente) => "sente_win",
                GameResult::Win(Color::Gote) => "gote_win",
                GameResult::Draw => "draw",
            }
            .into(),
            reason: reason.into(),
            final_sfen: String::new(),
            moves: truth.moves,
            foul_attempts: truth.foul_attempts,
            rating_change: RatingChangePair {
                you: zero.clone(),
                opponent: zero,
            },
            opponent: OpponentInfo {
                username: "x".into(),
                rating: 0,
                is_bot: true,
            },
        }
    }

    /// ランダム対局（反則を多く含む）を審判で打ち、その真実を再生すると、両者の観測・
    /// 各試行直前の foul_tried・終局が元の対局と一致する
    #[test]
    fn 真実の再生は元の対局と同じ観測になる() {
        let mut rng = StdRng::seed_from_u64(11);
        for game in 0..30 {
            let mut referee = Referee::new();
            let mut seen: Vec<(Color, usize, Vec<String>)> = vec![];
            let (result, reason) = loop {
                let side = referee.to_move();
                let view = referee.view(side, [0, 0], game);
                let mask = legal_mask(&view, referee.log(side), referee.foul_tried(side));
                let a = random_action(&mask, rng.random()).unwrap();
                let mut tried: Vec<String> = referee.foul_tried(side).iter().cloned().collect();
                tried.sort();
                seen.push((side, referee.log(side).events().len(), tried));
                let usi = decode_usi(a, side).unwrap();
                if let StepResult::Ended { result, reason, .. } = referee.step(&usi, 0) {
                    break (result, reason);
                }
            };
            let logs = [
                referee.log(Color::Sente).events().to_vec(),
                referee.log(Color::Gote).events().to_vec(),
            ];
            let end = end_payload(&mut referee, result, reason);
            let mut replayed_seen = vec![];
            let replayed = replay(&end, |r, a| {
                let mut tried: Vec<String> = r.foul_tried(a.side).iter().cloned().collect();
                tried.sort();
                replayed_seen.push((a.side, r.log(a.side).events().len(), tried));
            })
            .unwrap();
            assert_eq!(replayed_seen, seen, "game {game}");
            assert_eq!(replayed.ended(), Some((result, reason)));
            for (c, log) in [(Color::Sente, &logs[0]), (Color::Gote, &logs[1])] {
                assert!(same_observations(replayed.log(c).events(), log), "game {game} {c:?}");
            }
            assert_eq!(parse_result(&end.result), Some(result));
        }
    }

    /// 実際の記録ファイルで、再生した観測が記録された観測と一致するか。
    /// `RL_RECORDS_DIR=<dir> cargo test --release --lib 記録ファイルの再生 -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn 記録ファイルの再生が記録された観測と一致する() {
        let dir = std::env::var("RL_RECORDS_DIR").expect("RL_RECORDS_DIR");
        let (mut ok, mut bad, mut attempts_n) = (0, 0, 0usize);
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let content = std::fs::read_to_string(&path).unwrap();
            let (color, end) = crate::truth_replay::parse_bot_and_end(&content).unwrap();
            let mut n = 0;
            match replay(&end, |_, _| n += 1) {
                Ok(r) if same_observations(r.log(color).events(), &recorded_observations(&content))
                    && verify_outcome(&end, &r).is_ok() =>
                {
                    ok += 1;
                    attempts_n += n;
                }
                other => {
                    bad += 1;
                    println!("不一致 {}: {:?}", path.display(), other.err());
                }
            }
        }
        println!("一致 {ok} 局（試行 {attempts_n}）/ 不一致 {bad} 局");
        assert_eq!(bad, 0);
    }

    /// 記録の勝敗が再生の裁定と食い違う・読めない・途中で切れている記録は拒否する
    #[test]
    fn 勝敗の食い違う記録は拒否する() {
        let mut rng = StdRng::seed_from_u64(12);
        let mut referee = Referee::new();
        let (result, reason) = loop {
            let side = referee.to_move();
            let view = referee.view(side, [0, 0], 0);
            let mask = legal_mask(&view, referee.log(side), referee.foul_tried(side));
            let usi = decode_usi(random_action(&mask, rng.random()).unwrap(), side).unwrap();
            if let StepResult::Ended { result, reason, .. } = referee.step(&usi, 0) {
                break (result, reason);
            }
        };
        let end = end_payload(&mut referee, result, reason);
        let replayed = replay(&end, |_, _| {}).unwrap();
        assert_eq!(verify_outcome(&end, &replayed), Ok(result));

        // 勝者の取り違え
        let mut flipped = end.clone();
        flipped.result = match result {
            GameResult::Win(Color::Sente) => "gote_win",
            _ => "sente_win",
        }
        .into();
        assert!(verify_outcome(&flipped, &replayed).is_err());
        // 読めない結果文字列
        let mut unknown = end.clone();
        unknown.result = "?".into();
        assert!(verify_outcome(&unknown, &replayed).is_err());
        // 途中で切れているのに審判の終局理由を名乗る記録
        let mut truncated = end.clone();
        truncated.moves.truncate(2);
        truncated.foul_attempts.retain(|f| f.move_number <= 2);
        let short = replay(&truncated, |_, _| {}).unwrap();
        assert!(verify_outcome(&truncated, &short).is_err());
        // 審判の外の終局（投了）は、再生が未終局なら受け付ける
        let mut resigned = truncated.clone();
        resigned.reason = "resign".into();
        assert!(verify_outcome(&resigned, &short).is_ok());
    }

    #[test]
    fn 真実と食い違う手順は拒否する() {
        let mut referee = Referee::new();
        for usi in ["7g7f", "3c3d"] {
            referee.step(usi, 0);
        }
        let mut end = end_payload(&mut referee, GameResult::Draw, "max_plies");
        // 受理された手を、反則になる手へ差し替える
        end.moves[1].usi = "1a1i".into();
        assert!(matches!(replay(&end, |_, _| {}), Err(ReplayError::Mismatch { .. })));
    }
}
