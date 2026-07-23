//! 「ついたて将棋ビューワー」webhook契約のCSA表記⇔内部表現の変換。
//!
//! 指し手は常に7文字固定（符号1 + 移動元2桁 + 移動先2桁 + 駒種2文字）。
//! マスは USI のような筋+段（例 "7g"）ではなく筋+段とも数字（例 "76"）。
//! 駒種2文字テーブル・マスクの書式（`+0076ZZ` 等）は運営者提供のサンプル
//! （tsuboshun/tsuitate-sample-bot）のREADMEに加え、実際のエンジン実装
//! （tsuboshun/tsuitate-shogi-crates の tsuitate_bindings/src/game_api.rs
//! `piece_kind_to_csa` / `last_move_csa`）をソースで確認して実装した。
//! `lastCapture` は実戦dispatcherではCSAの2文字コードで、エンジン直結の
//! サンプルでは1文字のUSIコードの場合もあるため、下記パーサーは両方を受理する。

use crate::board::Coord;
use crate::protocol::{Color, Role};
use crate::shogi::{ShogiMove, parse_usi, promote_role};

pub fn parse_csa_square(s: &str) -> Option<Coord> {
    let bytes = s.as_bytes();
    if bytes.len() != 2 {
        return None;
    }
    let file = (bytes[0] as char).to_digit(10)? as i8;
    let rank = (bytes[1] as char).to_digit(10)? as i8;
    if (1..=9).contains(&file) && (1..=9).contains(&rank) {
        Some(Coord { file, rank })
    } else {
        None
    }
}

pub fn to_csa_square(c: Coord) -> String {
    format!("{}{}", c.file, c.rank)
}

/// 指し手末尾の駒種2文字（成り後の駒種を表す。CSA標準と同一）
pub fn role_to_csa2(role: Role) -> &'static str {
    match role {
        Role::Pawn => "FU",
        Role::Lance => "KY",
        Role::Knight => "KE",
        Role::Silver => "GI",
        Role::Gold => "KI",
        Role::Bishop => "KA",
        Role::Rook => "HI",
        Role::King => "OU",
        Role::Tokin => "TO",
        Role::Promotedlance => "NY",
        Role::Promotedknight => "NK",
        Role::Promotedsilver => "NG",
        Role::Horse => "UM",
        Role::Dragon => "RY",
    }
}

fn csa2_to_role(code: &str) -> Option<Role> {
    Some(match code {
        "FU" => Role::Pawn,
        "KY" => Role::Lance,
        "KE" => Role::Knight,
        "GI" => Role::Silver,
        "KI" => Role::Gold,
        "KA" => Role::Bishop,
        "HI" => Role::Rook,
        "OU" => Role::King,
        "TO" => Role::Tokin,
        "NY" => Role::Promotedlance,
        "NK" => Role::Promotedknight,
        "NG" => Role::Promotedsilver,
        "UM" => Role::Horse,
        "RY" => Role::Dragon,
        _ => return None,
    })
}

/// `lastCapture` を内部の基本駒種へ変換する。
///
/// dispatcher の実payloadはCSAの2文字（`FU`など）だが、エンジン直結の
/// payloadではUSIの1文字（`P`など）も使われるため、両方を受け付ける。
/// 成駒表記が来ても、持ち駒へ入る時点では必ず不成へ戻す。
pub fn parse_capture_letter(s: &str) -> Option<Role> {
    let role = match s {
        "P" => Role::Pawn,
        "L" => Role::Lance,
        "N" => Role::Knight,
        "S" => Role::Silver,
        "G" => Role::Gold,
        "B" => Role::Bishop,
        "R" => Role::Rook,
        "K" => Role::King,
        _ => csa2_to_role(s)?,
    };
    Some(crate::shogi::unpromote_role(role))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsaMoveKind {
    /// 相手の手（内容は不明）。捕獲時のみ移動先マスが開示される
    Masked {
        to: Option<Coord>,
    },
    Drop {
        role: Role,
        to: Coord,
    },
    Board {
        from: Coord,
        to: Coord,
        /// 着手末尾の駒種2文字が表す着手後の駒種（成っていれば成駒コード）。
        /// `wasPromotion` が欠落した反則エントリでも、着手前の駒種と比較すれば
        /// 成りだったかを復元できる（webhook_session::advance 参照）
        role_after: Role,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CsaMove {
    pub mover: Color,
    pub kind: CsaMoveKind,
}

/// `lastMove` をパースする（例: "+7776FU" 自分の移動 / "-0076ZZ" 相手の捕獲を
/// 伴う手（マスクされ移動先だけ判明）/ "+0000ZZ" 相手の情報なしの手 /
/// "+0054FU" 自分の打ち）
pub fn parse_csa_move(csa: &str) -> Option<CsaMove> {
    if csa.len() != 7 || !csa.is_ascii() {
        return None;
    }
    let mover = match csa.as_bytes()[0] {
        b'+' => Color::Sente,
        b'-' => Color::Gote,
        _ => return None,
    };
    let from_str = &csa[1..3];
    let to_str = &csa[3..5];
    let code_str = &csa[5..7];

    if code_str == "ZZ" {
        let to = if to_str == "00" {
            None
        } else {
            Some(parse_csa_square(to_str)?)
        };
        return Some(CsaMove {
            mover,
            kind: CsaMoveKind::Masked { to },
        });
    }
    let to = parse_csa_square(to_str)?;
    if from_str == "00" {
        let role = csa2_to_role(code_str)?;
        return Some(CsaMove {
            mover,
            kind: CsaMoveKind::Drop { role, to },
        });
    }
    let from = parse_csa_square(from_str)?;
    let role_after = csa2_to_role(code_str)?;
    Some(CsaMove {
        mover,
        kind: CsaMoveKind::Board {
            from,
            to,
            role_after,
        },
    })
}

fn color_sign(color: Color) -> char {
    match color {
        Color::Sente => '+',
        Color::Gote => '-',
    }
}

/// 自分の指し手（USI文字列）を送信用CSA文字列に変換する。
/// 盤上移動の駒種はUSIに含まれないため、「移動前の自駒配置」から
/// 呼び出し側に引いてもらう（role_at: マス→自駒の種類）
pub fn usi_move_to_csa(
    color: Color,
    usi: &str,
    role_at: impl Fn(Coord) -> Option<Role>,
) -> Option<String> {
    match parse_usi(usi)? {
        ShogiMove::Drop { role, to } => Some(format!(
            "{}00{}{}",
            color_sign(color),
            to_csa_square(to),
            role_to_csa2(role)
        )),
        ShogiMove::Board { from, to, promote } => {
            let pre_role = role_at(from)?;
            let role_after = if promote {
                promote_role(pre_role).unwrap_or(pre_role)
            } else {
                pre_role
            };
            Some(format!(
                "{}{}{}{}",
                color_sign(color),
                to_csa_square(from),
                to_csa_square(to),
                role_to_csa2(role_after)
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_own_board_move() {
        let mv = parse_csa_move("+7776FU").unwrap();
        assert_eq!(mv.mover, Color::Sente);
        assert_eq!(
            mv.kind,
            CsaMoveKind::Board {
                from: Coord { file: 7, rank: 7 },
                to: Coord { file: 7, rank: 6 },
                role_after: Role::Pawn,
            }
        );
    }

    #[test]
    fn parses_own_board_promotion() {
        let mv = parse_csa_move("+8822UM").unwrap();
        assert_eq!(mv.mover, Color::Sente);
        assert_eq!(
            mv.kind,
            CsaMoveKind::Board {
                from: Coord { file: 8, rank: 8 },
                to: Coord { file: 2, rank: 2 },
                role_after: Role::Horse,
            }
        );
    }

    #[test]
    fn parses_own_drop() {
        let mv = parse_csa_move("+0054FU").unwrap();
        assert_eq!(mv.mover, Color::Sente);
        assert_eq!(
            mv.kind,
            CsaMoveKind::Drop {
                role: Role::Pawn,
                to: Coord { file: 5, rank: 4 }
            }
        );
    }

    #[test]
    fn parses_masked_opponent_move_with_capture() {
        let mv = parse_csa_move("+0076ZZ").unwrap();
        assert_eq!(mv.mover, Color::Sente);
        assert_eq!(
            mv.kind,
            CsaMoveKind::Masked {
                to: Some(Coord { file: 7, rank: 6 })
            }
        );
    }

    #[test]
    fn parses_masked_opponent_move_without_info() {
        let mv = parse_csa_move("-0000ZZ").unwrap();
        assert_eq!(mv.mover, Color::Gote);
        assert_eq!(mv.kind, CsaMoveKind::Masked { to: None });
    }

    #[test]
    fn parses_capture_from_sample_test_fixture() {
        // サンプルのエンジン直結形式（1文字USI）とdispatcher形式（2文字CSA）
        assert_eq!(parse_capture_letter("P"), Some(Role::Pawn));
        assert_eq!(parse_capture_letter("FU"), Some(Role::Pawn));
        assert_eq!(parse_capture_letter("KA"), Some(Role::Bishop));
        assert_eq!(parse_capture_letter("TO"), Some(Role::Pawn));
    }

    #[test]
    fn encodes_own_move_using_pre_move_role() {
        let csa = usi_move_to_csa(Color::Sente, "7g7f", |c| {
            if c == (Coord { file: 7, rank: 7 }) {
                Some(Role::Pawn)
            } else {
                None
            }
        })
        .unwrap();
        assert_eq!(csa, "+7776FU");
    }

    #[test]
    fn encodes_own_promotion_using_promoted_role() {
        let csa = usi_move_to_csa(Color::Sente, "8h2b+", |c| {
            if c == (Coord { file: 8, rank: 8 }) {
                Some(Role::Bishop)
            } else {
                None
            }
        })
        .unwrap();
        assert_eq!(csa, "+8822UM");
    }

    #[test]
    fn encodes_own_drop() {
        let csa = usi_move_to_csa(Color::Gote, "P*5e", |_| None).unwrap();
        assert_eq!(csa, "-0055FU");
    }
}
