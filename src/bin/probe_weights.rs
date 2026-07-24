//! 一時診断: 「初期配置の角に既知の当たり（8七と金）」局面で、相手手事前分布が
//! 角逃げ手群に与える質量を測る（tokin-bet の占有信念が動かない原因の切り分け用）
use tsuitate_bot::estimator::opp_reply_weights;
use tsuitate_bot::protocol::Color;
use tsuitate_bot::shogi::{Position, ShogiMove, parse_usi};

fn main() {
    let mut pos = Position::initial();
    // tokin-bet.kif の1〜12手目 + [13手目だけ 1六歩に差し替え（角を8八に残す）] + 8七歩成
    for usi in [
        "5i5h", "3c3d", "3i4h", "7c7d", "7i6h", "8a7c", "5g5f", "8c8d", "2h3h", "8d8e",
        "7g7f", "8e8f", "1g1f", "8f8g+",
    ] {
        let mv = parse_usi(usi).expect("usi");
        assert!(pos.is_legal(&mv), "非合法: {usi}");
        pos.play_unchecked(&mv);
    }
    // 手番=先手。後手(gote)視点の応手予測: known = 8七（と金の位置は先手に露見済み）
    let known = [tsuitate_bot::board::parse_usi_square("8g").expect("coord")];
    let weights = opp_reply_weights(&pos, Color::Gote, &known, &[], 0);
    let total: f64 = weights.iter().map(|(_, w)| w).sum();
    let bishop_from = tsuitate_bot::board::parse_usi_square("8h").expect("coord");
    let mut escape = 0.0;
    let mut escape_moves: Vec<(String, f64)> = vec![];
    for (mv, w) in &weights {
        if let ShogiMove::Board { from, .. } = mv {
            if *from == bishop_from {
                escape += w;
                escape_moves.push((format!("{mv:?}"), *w));
            }
        }
    }
    escape_moves.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("候補 {} 手 / 総質量 {total:.4}", weights.len());
    println!("角(8h)を動かす手の質量: {escape:.4} = {:.1}%", escape / total * 100.0);
    for (m, w) in escape_moves.iter().take(5) {
        println!("  {m} w={w:.5} ({:.2}%)", w / total * 100.0);
    }
}
