//! C-8（直接盤面合成）の検証用スタンドアロンハーネス。
//!
//! scenario.rs は bot に指し継ぎさせる（strategy 層）ためのツールなので、
//! 「観測列から盤面を直接合成する」機構だけを切り出して単体テストする。
//! kifu を読み、観測ログを組み立て、合成器を呼び、既知の確定事実
//! （ユーザーが手で検証した事実）とどれだけ一致するかを集計する。
//!
//! 使い方: cargo run --release --bin synth_check -- <名前.kif> <ply> [試行数]

use std::collections::HashMap;

use tsuitate_bot::board::make_usi_square;
use tsuitate_bot::estimator::synth_particle;
use tsuitate_bot::kifu::{RawFoul, parse_kif};
use tsuitate_bot::model::GameModel;
use tsuitate_bot::observation::{Observation, ObservationLog};
use tsuitate_bot::protocol::{Color, Role};
use tsuitate_bot::shogi::{Position, ShogiMove, parse_usi, unpromote_role};

fn resolve_foul(pos: &Position, f: &RawFoul) -> String {
    match f {
        RawFoul::Drop { role, to } => {
            tsuitate_bot::board::make_usi_drop(*role, *to).expect("打てない駒種の反則試行")
        }
        RawFoul::Board { from, to, role } => {
            let piece = pos.piece_at(*from).expect("反則試行の移動元に駒がない");
            let piece_promoted = piece.role != unpromote_role(piece.role);
            let code_promoted = *role != unpromote_role(*role);
            tsuitate_bot::board::make_usi_move(*from, *to, code_promoted && !piece_promoted)
        }
    }
}

/// kakunari.kif 等を ply 手目まで再生し、真実の局面と手番側の観測ログを返す
fn replay_to(path: &str, ply: usize) -> (Position, ObservationLog, Color) {
    let text = std::fs::read_to_string(path).expect("kif が読めません");
    let kifu = parse_kif(&text).expect("kif のパース失敗");
    let mut pos = Position::initial();
    let mut log_sente = ObservationLog::default();
    let mut log_gote = ObservationLog::default();
    for ply_data in &kifu.plies[..ply] {
        let side = pos.turn();
        for f in &ply_data.fouls {
            let usi = resolve_foul(&pos, f);
            let mv = parse_usi(&usi).expect("反則USI解析失敗");
            let log = if side == Color::Sente { &mut log_sente } else { &mut log_gote };
            log.record(Observation::MyFoul {
                move_number: pos.move_number(),
                usi,
            });
            let other = if side == Color::Sente { &mut log_gote } else { &mut log_sente };
            other.record(Observation::OpponentFoul { count: 0 });
            let _ = mv;
        }
        let usi = ply_data.mv.to_usi();
        let mv = parse_usi(&usi).expect("USI解析失敗");
        let captured = pos.play_unchecked(&mv);
        let move_number = pos.move_number();
        let captured_sq = captured.map(|_| match mv {
            ShogiMove::Board { to, .. } => make_usi_square(to),
            ShogiMove::Drop { .. } => unreachable!(),
        });
        let (mover_log, other_log) = if side == Color::Sente {
            (&mut log_sente, &mut log_gote)
        } else {
            (&mut log_gote, &mut log_sente)
        };
        mover_log.record(Observation::MyMove {
            move_number,
            usi,
            captured: captured.map(unpromote_role),
        });
        other_log.record(Observation::OpponentMoved {
            move_number,
            captured_my_piece_at: captured_sq,
        });
        if pos.in_check(pos.turn()) {
            let in_check = pos.turn();
            log_sente.record(Observation::Check { in_check });
            log_gote.record(Observation::Check { in_check });
        }
    }
    let side = pos.turn();
    let log = if side == Color::Sente { log_sente } else { log_gote };
    (pos, log, side)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().cloned().unwrap_or_else(|| "scenarios/kakunari.kif".into());
    let ply: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(69);
    let trials: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);

    let (truth, log, side) = replay_to(&path, ply);
    let opp = side.other();
    let model = GameModel::from_log(side, &log);
    assert!(model.consistent(), "観測ログの再構成が矛盾");

    println!("局面: {path} の {ply}手目まで再生（手番={side:?}）");
    println!(
        "真実: 玉={} / 相手持ち駒={:?}",
        truth
            .king_square(opp)
            .map(make_usi_square)
            .unwrap_or_else(|| "-".into()),
        model.opponent_hand(),
    );
    // 真実の盤上駒の役割別内訳（合成側が正しい駒数を作れているかの検算用）
    let mut truth_roles: HashMap<Role, u32> = HashMap::new();
    for (_, p) in truth.pieces() {
        if p.color == opp {
            *truth_roles.entry(p.role).or_insert(0) += 1;
        }
    }
    println!("真実の相手盤上駒（役割別）: {truth_roles:?}");

    // 合成を trials 回実行し、既知の確定事実との一致率を集計。
    // you_in_check（自玉が今まさに王手されているか）は観測から厳密に分かる
    let you_in_check = truth.in_check(side);
    println!("自玉が王手されているか（観測から厳密に既知）: {you_in_check}");
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    let mut king_tally: HashMap<String, u32> = HashMap::new();
    let mut king_hit_4g3g = 0usize;
    let mut horse_at_1a = 0usize; // 65手目 1一角成 の痕跡（馬が1一にいるか）
    let mut promoted_at_4a = 0usize; // 69手目 4一歩成/4一香成 の痕跡
    let mut role_count_ok = 0usize;
    let mut n = 0usize;
    let mut synth_fail = 0usize;
    use rand::SeedableRng;
    for _ in 0..trials {
        let Some(synth) = synth_particle(side, &model, you_in_check, &mut rng) else {
            synth_fail += 1;
            continue;
        };
        n += 1;
        if let Some(k) = synth.king_square(opp) {
            let s = make_usi_square(k);
            *king_tally.entry(s.clone()).or_insert(0) += 1;
            if s == "4g" || s == "3g" {
                king_hit_4g3g += 1;
            }
        }
        if synth
            .piece_at(tsuitate_bot::board::parse_usi_square("1a").unwrap())
            .is_some_and(|p| p.color == opp && p.role == Role::Horse)
        {
            horse_at_1a += 1;
        }
        if synth
            .piece_at(tsuitate_bot::board::parse_usi_square("4a").unwrap())
            .is_some_and(|p| p.color == opp && matches!(p.role, Role::Tokin | Role::Promotedlance))
        {
            promoted_at_4a += 1;
        }
        let mut roles: HashMap<Role, u32> = HashMap::new();
        for (_, p) in synth.pieces() {
            if p.color == opp {
                *roles.entry(p.role).or_insert(0) += 1;
            }
        }
        if roles == truth_roles {
            role_count_ok += 1;
        }
    }
    println!();
    println!("合成 {n}/{trials} 回成功（静的王手整合性で棄却: {synth_fail}）");
    let mut sorted_king: Vec<_> = king_tally.into_iter().collect();
    sorted_king.sort_by(|a, b| b.1.cmp(&a.1));
    println!("合成された玉位置の上位: {:?}", &sorted_king[..sorted_king.len().min(10)]);
    println!(
        "駒数の役割別内訳が真実と一致: {:.1}%",
        100.0 * role_count_ok as f64 / n.max(1) as f64
    );
    println!(
        "玉が4g/3gのどちらか（ユーザー確定事実）: {:.1}%",
        100.0 * king_hit_4g3g as f64 / n.max(1) as f64
    );
    println!(
        "1aに馬（65手目 1一角成 の痕跡）: {:.1}%",
        100.0 * horse_at_1a as f64 / n.max(1) as f64
    );
    println!(
        "4aに成駒（69手目 4一歩成/香成 の痕跡）: {:.1}%",
        100.0 * promoted_at_4a as f64 / n.max(1) as f64
    );

    // ユーザーの手動推論（1一角成の確定根拠）を計算で再現する検証。
    // 「先手桂馬が2四に着地すると1二・3二の両方に利く」という幾何的事実と、
    // 実際の王手観測履歴（自玉=後手玉への王手が一度も観測されていない）を
    // 突き合わせ、桂馬経由の経路が本当に矛盾するかを確認する
    println!();
    println!("=== 角成確定の演繹検証（ユーザーの手動推論を計算で再現） ===");
    let contexts = tsuitate_bot::deduce::opponent_move_contexts(side, &log);
    println!("相手（先手）の着手数: {}（うち自玉への王手が観測された回数: {}）",
        contexts.len(),
        contexts.iter().filter(|c| c.check_declared).count());
    let sq_2d = tsuitate_bot::board::parse_usi_square("2d").unwrap();
    let sq_1b = tsuitate_bot::board::parse_usi_square("1b").unwrap();
    let sq_1a = tsuitate_bot::board::parse_usi_square("1a").unwrap();
    let refuted_2d = tsuitate_bot::deduce::route_square_refuted_by_check_history(
        Role::Knight, Color::Sente, sq_2d, &contexts, 0..contexts.len(),
    );
    let refuted_1b = tsuitate_bot::deduce::route_square_refuted_by_check_history(
        Role::Promotedknight, Color::Sente, sq_1b, &contexts, 0..contexts.len(),
    );
    println!("桂馬が2四に着地する経路は自玉の王手履歴と矛盾するか: {refuted_2d}");
    println!("成桂が1二に着地する経路は自玉の王手履歴と矛盾するか: {refuted_1b}");
    let refuted_1a_horse = tsuitate_bot::deduce::route_square_refuted_by_check_history(
        Role::Horse, Color::Sente, sq_1a, &contexts, 0..contexts.len(),
    );
    println!("（対照）馬が1一に着地する経路は自玉の王手履歴と矛盾するか: {refuted_1a_horse}");

    // テンポ判定: 桂馬2本の本国・角の本国から1一（成り込み）への
    // 空盤上の最短手数と、実際に使えた先手の総手数を比較する
    println!();
    let n_sente_moves = contexts.len() as u32;
    let knight_homes = [
        tsuitate_bot::board::parse_usi_square("2i").unwrap(),
        tsuitate_bot::board::parse_usi_square("8i").unwrap(),
    ];
    for home in knight_homes {
        let d = tsuitate_bot::deduce::min_moves_empty_board(
            Role::Knight, Color::Sente, home, sq_1a, true,
        );
        println!(
            "桂馬({})→1一(成桂)の最短手数: {:?} / 使えた先手手数: {n_sente_moves}",
            tsuitate_bot::board::make_usi_square(home), d,
        );
    }
    let bishop_home = tsuitate_bot::board::parse_usi_square("8h").unwrap();
    let d_bishop = tsuitate_bot::deduce::min_moves_empty_board(
        Role::Bishop, Color::Sente, bishop_home, sq_1a, true,
    );
    println!(
        "角(8八)→1一(馬)の最短手数: {d_bishop:?} / 使えた先手手数: {n_sente_moves}"
    );
    println!();
    println!(
        "注意: 総手数（{n_sente_moves}）との比較だけでは桂馬の経路は却下できません\
        （最短5手 << 総手数）。ユーザーの手動推論は「総手数」ではなく、もっと\
        狭い窓（この桂馬がまだ本国にいたと分かる最後の地点からの手数）を\
        使っているはずです。その窓の起点（何手目時点か）を教えてもらえれば\
        同じ結論を自動で再現できます。"
    );
}
