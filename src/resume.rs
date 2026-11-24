//! 復帰・再接続のときに、観測ログの末尾の欠けを埋める（Socket.IO 版の本番 bot 用）。
//!
//! サーバーが再接続で返すのは視界（`PlayerView` = 自駒・持ち駒・手数・反則数・王手中か）だけで、切断中に
//! 送られた通知の再送も履歴も無い（tsuitate の `game-room.ts` の `bindSocket` / `handleSync`）。一方、bot が
//! 止まっている間に進むのは高々「相手の1手」で、これは視界から完全に作り直せる:
//!
//! - 相手の手で取られた自駒のマス = ログから再構成した自駒のうち、視界に無い1枚
//! - 王手宣言 = 視界の「王手中か」
//! - 相手の反則 = 視界の相手の反則数との差
//!
//! 自分が送った手の結果（受理か反則か）を受け取る前に落ちた場合も、送った手（対局記録の `chose` 行・メモリ上の
//! 送信済みの手）と視界から決まる: 反則数が1つ増えていれば反則、手数が進んでいれば受理（取った駒は持ち駒の差）。
//!
//! **確かに決まるものだけ埋める**。決まらない欠け（自分の手の受理と相手の応手の両方が欠けた = 自分の手が王手
//! だったかが分からない、など）は埋めずにエラーを返す。王手宣言の有無は相手玉の候補の推論（`deduce`）が事実として
//! 使うので、当て推量で埋めると真の玉を落としうる。
//!
//! 履歴を失った側は、その局の勝率が 12〜17pt 下がる（docs/rl-deepnash-design.md の「入力と行動マスクの点検」）。

use std::collections::BTreeMap;

use crate::model::GameModel;
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{Color, PlayerView, Role, VisiblePiece};
use crate::shogi::HAND_ROLES;

/// 手数 `n`（1始まり。初手が 1）を指す側
fn mover(n: u32) -> Color {
    if n % 2 == 1 { Color::Sente } else { Color::Gote }
}

fn piece_map(pieces: &[VisiblePiece]) -> BTreeMap<String, Role> {
    pieces.iter().map(|p| (p.square.clone(), p.role)).collect()
}

fn hand_counts(hand: &std::collections::HashMap<Role, u32>) -> [u32; 7] {
    let mut out = [0u32; 7];
    for (i, role) in HAND_ROLES.iter().enumerate() {
        out[i] = hand.get(role).copied().unwrap_or(0);
    }
    out
}

/// ログの末尾（最後の受理手より後）の様子
struct Tail<'a> {
    /// 最後の受理手（自分の着手か相手の着手）
    last_accepted: Option<&'a Observation>,
    /// 最後の受理手の後に、その色への王手宣言があるか
    check_on: Option<Color>,
    /// 最後の受理手の後に自分の反則があるか
    my_foul: bool,
}

fn tail(log: &ObservationLog) -> Tail<'_> {
    let mut t = Tail { last_accepted: None, check_on: None, my_foul: false };
    for e in log.events() {
        match e {
            Observation::MyMove { .. } | Observation::OpponentMoved { .. } => {
                t = Tail { last_accepted: Some(e), check_on: None, my_foul: false };
            }
            Observation::Check { in_check } => t.check_on = Some(*in_check),
            Observation::MyFoul { .. } => t.my_foul = true,
            Observation::OpponentFoul { .. } => {}
        }
    }
    t
}

/// 視界 `view`（サーバーから取り直した今の状態）に合わせて、観測ログ `log` の末尾に足すべき観測を返す。
/// 欠けが無ければ空。`pending` は「送ったが結果をまだ受け取っていない手」の（選んだ時点の手数, USI）。
///
/// 足す観測の手数は審判・サーバーと同じ規約（受理手は適用後の値、反則は今の手数）。
/// 確かに決まらない欠けは `Err`（理由）で、そのときログは触らない
pub fn reconcile(
    color: Color,
    log: &ObservationLog,
    pending: Option<(u32, &str)>,
    view: &PlayerView,
) -> Result<Vec<Observation>, String> {
    let opp = color.other();
    let model = GameModel::from_log(color, log);
    if !model.consistent() {
        return Err("観測ログから自駒を再構成できない".into());
    }
    let accepted = log
        .events()
        .iter()
        .filter(|e| matches!(e, Observation::MyMove { .. } | Observation::OpponentMoved { .. }))
        .count() as u32;
    // ログが言う「今の手数」
    let expected = accepted + 1;
    if !crate::rl::encode::moves_contiguous(log, expected) {
        return Err("観測ログの受理手の手数がすでに欠けている".into());
    }
    let now = view.move_number;
    if now < expected {
        return Err(format!("視界（{now}手目）が観測ログ（{expected}手目）より古い"));
    }
    if view.turn != mover(now) {
        return Err(format!("視界の手番 {:?} が手数 {now} と合わない", view.turn));
    }
    let my_fouls = view
        .fouls
        .you
        .checked_sub(model.my_fouls())
        .ok_or("視界の自分の反則数が観測ログより少ない")?;
    let opp_fouls = view
        .fouls
        .opponent
        .checked_sub(model.opponent_fouls())
        .ok_or("視界の相手の反則数が観測ログより少ない")?;
    let push_opp_fouls = |out: &mut Vec<Observation>| {
        for count in model.opponent_fouls() + 1..=view.fouls.opponent {
            out.push(Observation::OpponentFoul { count });
        }
    };
    let t = tail(log);
    let mut out = vec![];
    match now - expected {
        0 if view.turn == color => {
            // 自分の手番。相手の手の通知は入っている。直後の王手宣言だけ取りこぼすことがある
            match (view.you_in_check, t.check_on == Some(color)) {
                (true, false) => {
                    if !matches!(t.last_accepted, Some(Observation::OpponentMoved { .. })) || t.my_foul {
                        return Err("王手中だが、王手宣言を足せる位置が無い".into());
                    }
                    out.push(Observation::Check { in_check: color });
                }
                (false, true) => return Err("観測ログは王手中だが、視界は王手中でない".into()),
                _ => {}
            }
            match my_fouls {
                0 => {}
                1 => {
                    // 送った手が反則だった（ack を受け取る前に落ちた）
                    let (_, usi) = pending
                        .filter(|&(n, _)| n == now)
                        .ok_or("自分の反則が1つ多いが、送った手が分からない")?;
                    out.push(Observation::MyFoul { move_number: now, usi: usi.to_string() });
                }
                n => return Err(format!("自分の反則が {n} 個多い")),
            }
            // 相手の反則の取りこぼし（相手の手番は閉じている）は埋めない: 末尾に足すと次の相手の手番の反則に
            // 見える。累計数は視界から入るので、失うのは時系列の1点だけ
        }
        0 => {
            // 相手の手番。自分の手の直後の王手宣言と、相手のこの手番の反則を取りこぼすことがある
            if my_fouls != 0 {
                return Err("相手の手番なのに自分の反則数が合わない".into());
            }
            match (view.opponent_in_check, t.check_on == Some(opp)) {
                (true, false) => {
                    if !matches!(t.last_accepted, Some(Observation::MyMove { .. })) {
                        return Err("相手が王手中だが、王手宣言を足せる位置が無い".into());
                    }
                    out.push(Observation::Check { in_check: opp });
                }
                (false, true) => return Err("観測ログは相手が王手中だが、視界は違う".into()),
                _ => {}
            }
            push_opp_fouls(&mut out);
        }
        1 if mover(expected) == opp => {
            // 相手の手が1つ欠けている（止まっている間に相手が指した）
            if my_fouls != 0 {
                return Err("相手の手が欠けているうえに、自分の反則数も合わない".into());
            }
            push_opp_fouls(&mut out);
            let (mine, seen) = (piece_map(&model.my_pieces()), piece_map(&view.your_pieces));
            if seen.iter().any(|(sq, role)| mine.get(sq) != Some(role)) {
                return Err("視界に、観測ログに無い自駒がある".into());
            }
            let lost: Vec<&String> = mine.keys().filter(|sq| !seen.contains_key(*sq)).collect();
            if lost.len() > 1 {
                return Err(format!("相手の1手で自駒が {} 枚減っている", lost.len()));
            }
            out.push(Observation::OpponentMoved {
                move_number: now,
                captured_my_piece_at: lost.first().map(|s| (*s).clone()),
            });
            if view.you_in_check {
                out.push(Observation::Check { in_check: color });
            }
        }
        1 => {
            // 自分の手が1つ欠けている（送った手が受理されたが、通知を受け取る前に落ちた）
            let (_, usi) = pending
                .filter(|&(n, _)| n == expected)
                .ok_or("自分の手が欠けているが、送った手が分からない")?;
            if my_fouls != 0 {
                return Err("自分の手が欠けているうえに、自分の反則数も合わない".into());
            }
            // 取った駒は持ち駒の差から決まる（相手の反則・王手宣言は持ち駒を変えない）
            let mut after = model.clone();
            after.apply(&Observation::MyMove { move_number: now, usi: usi.to_string(), captured: None });
            if !after.consistent() {
                return Err(format!("送った手 {usi} を観測ログの自駒に適用できない"));
            }
            let (had, has) = (hand_counts(&after.my_hand()), hand_counts(&view.your_hand));
            let mut captured = None;
            for (i, role) in HAND_ROLES.iter().enumerate() {
                match has[i] as i64 - had[i] as i64 {
                    0 => {}
                    1 if captured.is_none() => captured = Some(*role),
                    _ => return Err("送った手の後の持ち駒が視界と合わない".into()),
                }
            }
            out.push(Observation::MyMove { move_number: now, usi: usi.to_string(), captured });
            if view.opponent_in_check {
                out.push(Observation::Check { in_check: opp });
            }
            push_opp_fouls(&mut out);
        }
        2 => {
            return Err("自分の手と相手の応手の両方が欠けている（自分の手が王手だったかが分からない）".into());
        }
        n => return Err(format!("受理手が {n} 個欠けている")),
    }
    let _ = opp_fouls;

    // 検算: 足した観測を適用した自分側が視界と一致する
    let mut after = model;
    for e in &out {
        after.apply(e);
    }
    if !after.consistent()
        || piece_map(&after.my_pieces()) != piece_map(&view.your_pieces)
        || hand_counts(&after.my_hand()) != hand_counts(&view.your_hand)
        || after.my_fouls() != view.fouls.you
    {
        return Err("欠けを埋めても、自駒・持ち駒・反則数が視界と合わない".into());
    }
    Ok(out)
}

/// 今の手番で既に反則になった手（ログの末尾の、今の手数の `MyFoul`）。復帰のときに反則のメモを作り直す
pub fn fouls_this_turn(log: &ObservationLog, now: u32) -> std::collections::HashSet<String> {
    log.events()
        .iter()
        .filter_map(|e| match e {
            Observation::MyFoul { move_number, usi } if *move_number == now => Some(usi.clone()),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::referee::{Referee, StepResult};
    use crate::rl::action::{NUM_ACTIONS, basic_legal_mask, decode_usi};
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn json(events: &[Observation]) -> Vec<String> {
        events.iter().map(|e| serde_json::to_string(e).unwrap()).collect()
    }

    fn prefix(log: &ObservationLog, len: usize) -> ObservationLog {
        let mut out = ObservationLog::default();
        for e in &log.events()[..len] {
            out.record(e.clone());
        }
        out
    }

    /// `log` の先頭 `len` 個だけが残ったとき、`reconcile` が残りをそのまま作り直す
    fn assert_rebuilds(referee: &Referee, side: Color, len: usize, pending: Option<(u32, &str)>, what: &str) {
        let full = referee.log(side);
        let view = referee.view(side, [0, 0], 0);
        let got = reconcile(side, &prefix(full, len), pending, &view)
            .unwrap_or_else(|e| panic!("{what}: 埋められなかった: {e}"));
        assert_eq!(json(&got), json(&full.events()[len..]), "{what}");
    }

    /// ランダム対局のあらゆる時点で、末尾が欠けたログを視界から作り直せる（審判が積んだ観測と一致する）:
    /// 欠け無し / 相手の1手（その手番の相手の反則・王手宣言も）/ 王手宣言だけ / 送った手の反則 / 送った手の受理。
    /// 自分の手と相手の応手の両方が欠けたら埋めない
    #[test]
    fn 末尾の欠けを視界から作り直す() {
        let mut rng = StdRng::seed_from_u64(5);
        let mut n = [0u32; 6];
        for _game in 0..40 {
            let mut referee = Referee::new();
            // 各側の「自分の最後の受理手（と王手宣言）まで」のログの長さと、その1つ前
            let mut after_my_move = [0usize; 2];
            let mut before_my_move = [0usize; 2];
            loop {
                let side = referee.to_move();
                let i = (side == Color::Gote) as usize;
                let view = referee.view(side, [0, 0], 0);
                let tried = referee.foul_tried(side).clone();
                // 欠け無し（両方の席）
                for s in [Color::Sente, Color::Gote] {
                    let len = referee.log(s).events().len();
                    assert_rebuilds(&referee, s, len, None, "欠け無し");
                    n[0] += 1;
                }
                if tried.is_empty() && view.move_number > 1 {
                    // 相手の手が届いた直後: 相手の1手ぶん（相手の反則・着手・王手宣言）が欠けたログ
                    assert_rebuilds(&referee, side, after_my_move[i], None, "相手の1手が欠けた");
                    n[1] += 1;
                    if view.you_in_check {
                        let len = referee.log(side).events().len();
                        assert_rebuilds(&referee, side, len - 1, None, "王手宣言だけ欠けた");
                        n[2] += 1;
                    }
                    // 自分の手と相手の応手の両方が欠けたら埋めない
                    if view.move_number > 2 {
                        let short = prefix(referee.log(side), before_my_move[i]);
                        assert!(reconcile(side, &short, None, &view).is_err(), "2手の欠けを埋めた");
                        n[5] += 1;
                    }
                }
                let mask = basic_legal_mask(&view, &tried);
                let acts: Vec<usize> = (0..NUM_ACTIONS).filter(|&a| mask[a]).collect();
                if acts.is_empty() {
                    break;
                }
                let usi = decode_usi(acts[rng.random_range(0..acts.len())], side).unwrap();
                let len_before = referee.log(side).events().len();
                let pending = Some((view.move_number, usi.as_str()));
                match referee.step(&usi, 0) {
                    StepResult::Ended { .. } => break,
                    StepResult::Foul => {
                        // 送った手が反則だった（結果を受け取る前に落ちた）
                        assert_rebuilds(&referee, side, len_before, pending, "送った手の反則");
                        n[3] += 1;
                        // 送った手が分からなければ埋めない
                        let view = referee.view(side, [0, 0], 0);
                        assert!(reconcile(side, &prefix(referee.log(side), len_before), None, &view).is_err());
                    }
                    StepResult::Accepted => {
                        // 送った手が受理された（通知を受け取る前に落ちた）。いまは相手の手番
                        assert_rebuilds(&referee, side, len_before, pending, "送った手の受理");
                        n[4] += 1;
                        before_my_move[i] = len_before;
                        after_my_move[i] = referee.log(side).events().len();
                    }
                }
            }
        }
        println!("欠け無し {} / 相手の1手 {} / 王手宣言だけ {} / 送った手の反則 {} / 送った手の受理 {} / 2手の欠け {}", n[0], n[1], n[2], n[3], n[4], n[5]);
        assert!(n.iter().all(|&x| x > 0), "検査が空振りしている: {n:?}");
    }

    /// 実際の対局記録（復帰をまたいだもの）の観測が、終局時に公開された真実の手順から審判が作り直した観測と
    /// 一致する（本番の経路の確認用。`RESUME_RECORD=<記録のパス>` を付けて `--ignored` で回す）
    #[test]
    #[ignore]
    fn 対局記録の観測は真実から作り直した観測と一致する() {
        let path = std::env::var("RESUME_RECORD").expect("RESUME_RECORD に対局記録のパスを渡す");
        let loaded = crate::record::load_record(std::path::Path::new(&path)).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let mut resumed = 0;
        let mut end = None;
        for line in text.lines() {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
            match v["type"].as_str() {
                Some("resumed") => {
                    resumed += 1;
                    println!("復帰: 読み直した観測 {} 件・作り直した観測 {} 件", v["restored"], v["repaired"]);
                }
                Some("end") => {
                    end = Some(serde_json::from_value::<crate::protocol::GameEndPayload>(v["payload"].clone()).unwrap());
                }
                _ => {}
            }
        }
        let end = end.expect("終局の行が無い");
        let mut referee = Referee::new();
        for a in crate::rl::records::attempts(&end) {
            referee.step(&a.usi, 0);
        }
        let (got, want) = (json(&loaded.events), json(referee.log(loaded.your_color).events()));
        let common = got.len().min(want.len());
        for i in 0..common {
            assert_eq!(got[i], want[i], "{i} 番目の観測が違う");
        }
        println!(
            "{:?} 番・復帰 {resumed} 回・終局 {} / {}: 記録の観測 {} 件と真実からの観測 {} 件が、先頭 {common} 件で一致",
            loaded.your_color, end.result, end.reason, got.len(), want.len()
        );
        assert!(got.len().abs_diff(want.len()) <= 1, "観測の数が合わない");
    }

    #[test]
    fn 今の手番の反則のメモを作り直す() {
        let mut referee = Referee::new();
        assert!(matches!(referee.step("7g7f", 0), StepResult::Accepted));
        assert!(matches!(referee.step("3c3d", 0), StepResult::Accepted));
        // 先手: 1九香→1一 は反則
        assert!(matches!(referee.step("1i1a", 0), StepResult::Foul));
        let view = referee.view(Color::Sente, [0, 0], 0);
        let got = fouls_this_turn(referee.log(Color::Sente), view.move_number);
        assert_eq!(&got, referee.foul_tried(Color::Sente));
        assert!(matches!(referee.step("2g2f", 0), StepResult::Accepted));
        assert!(matches!(referee.step("8c8d", 0), StepResult::Accepted));
        let view = referee.view(Color::Sente, [0, 0], 0);
        assert!(fouls_this_turn(referee.log(Color::Sente), view.move_number).is_empty(), "前の手番の反則は入らない");
    }
}
