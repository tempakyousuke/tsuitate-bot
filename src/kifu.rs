//! Shogi Quest がエクスポートする KIF 形式のパーサー（bin/scenario の実戦再現用）。
//!
//! 対応する形式（実際のエクスポート2件で確認済みの範囲）:
//! - ヘッダ行（棋戦：等）は読み飛ばす
//! - 指し手行: `30 同　歩(84)  ( 0:00/00:00:12)`
//!   - 移動元は常に `(筋段)` で付く。`打` は持ち駒、`成` は成り、`同` は直前の着地マス
//! - `*illegal:6465FU,0083FU` — **直前の指し手行の手番側が、その手を指す前に
//!   試みた反則**（Shogi Quest の出力規約。実棋譜2件の全12箇所で検証済み）。
//!   表記は 移動元2桁+移動先2桁+駒コード2字。移動元 `00` は打ち。
//!   駒コードは**移動後**の駒（`8389RY` = 飛が89へ成る試み）なので、
//!   成る手か成駒を動かす手かの判別には盤面が要る（RawFoul のまま返し、
//!   利用側が局面で解決する）
//! - 終局行（投了・反則負け等）以降の `*illegal:` は trailing_fouls に入れる
//! - `*scenario key=value ...` — このリポジトリ独自のシナリオ指定
//!   （ply=再生する手数 / target=注目手USI / diag=利き診断マス / desc=説明。
//!   desc は行末まで）

use std::collections::HashMap;

use crate::board::Coord;
use crate::protocol::Role;

/// 受理された1手。同・成・打は解決済み（KIF には移動元が常に付くので曖昧性がない）
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KifMove {
    Board { from: Coord, to: Coord, promote: bool },
    Drop { role: Role, to: Coord },
}

impl KifMove {
    pub fn to_usi(&self) -> String {
        match *self {
            KifMove::Board { from, to, promote } => {
                crate::board::make_usi_move(from, to, promote)
            }
            KifMove::Drop { role, to } => {
                crate::board::make_usi_drop(role, to).expect("打てない駒種")
            }
        }
    }
}

/// `*illegal` 行の1エントリ。駒コードが成駒でも「成る手」か「成駒を動かす手」かは
/// この時点では分からない（盤面の移動元の駒で解決する）
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RawFoul {
    Drop {
        role: Role,
        to: Coord,
    },
    Board {
        from: Coord,
        to: Coord,
        /// 駒コード（移動後の駒種）
        role: Role,
    },
}

#[derive(Debug, Clone)]
pub struct KifPly {
    pub mv: KifMove,
    /// この手を指す前の同じ手番側の反則試行
    pub fouls: Vec<RawFoul>,
}

#[derive(Debug, Default)]
pub struct Kifu {
    pub plies: Vec<KifPly>,
    /// 終局宣言（投了・反則負け等）の直前の反則試行
    pub trailing_fouls: Vec<RawFoul>,
    /// `*scenario` 行の key=value
    pub directives: HashMap<String, String>,
}

const KANJI_RANKS: [char; 9] = ['一', '二', '三', '四', '五', '六', '七', '八', '九'];

fn parse_file_digit(c: char) -> Option<i8> {
    match c {
        '1'..='9' => Some(c as i8 - b'0' as i8),
        '１'..='９' => Some((c as u32 - '１' as u32) as i8 + 1),
        _ => None,
    }
}

fn parse_kanji_rank(c: char) -> Option<i8> {
    KANJI_RANKS.iter().position(|&k| k == c).map(|i| i as i8 + 1)
}

fn coord(file: i8, rank: i8) -> Option<Coord> {
    if (1..=9).contains(&file) && (1..=9).contains(&rank) {
        Some(Coord { file, rank })
    } else {
        None
    }
}

/// 駒名（成香などの複合を先に試す）
fn parse_piece_name(chars: &[char]) -> Option<(Role, usize)> {
    let two: String = chars.iter().take(2).collect();
    match two.as_str() {
        "成香" => return Some((Role::Promotedlance, 2)),
        "成桂" => return Some((Role::Promotedknight, 2)),
        "成銀" => return Some((Role::Promotedsilver, 2)),
        "と金" => return Some((Role::Tokin, 2)),
        _ => {}
    }
    let role = match chars.first()? {
        '歩' => Role::Pawn,
        '香' => Role::Lance,
        '桂' => Role::Knight,
        '銀' => Role::Silver,
        '金' => Role::Gold,
        '角' => Role::Bishop,
        '飛' => Role::Rook,
        '玉' | '王' => Role::King,
        'と' => Role::Tokin,
        '馬' => Role::Horse,
        '龍' | '竜' => Role::Dragon,
        _ => return None,
    };
    Some((role, 1))
}

fn parse_foul_code(code: &str) -> Option<Role> {
    Some(match code {
        "FU" => Role::Pawn,
        "KY" => Role::Lance,
        "KE" => Role::Knight,
        "GI" => Role::Silver,
        "KI" => Role::Gold,
        "KA" => Role::Bishop,
        "HI" => Role::Rook,
        "OU" | "GY" => Role::King,
        "TO" => Role::Tokin,
        "NY" => Role::Promotedlance,
        "NK" => Role::Promotedknight,
        "NG" => Role::Promotedsilver,
        "UM" => Role::Horse,
        "RY" => Role::Dragon,
        _ => return None,
    })
}

fn parse_foul_entry(entry: &str) -> Result<RawFoul, String> {
    let e = entry.trim();
    if e.len() != 6 {
        return Err(format!("illegal エントリの長さが不正: {e}"));
    }
    let digits: Vec<i8> = e[..4]
        .chars()
        .map(|c| c.to_digit(10).map(|d| d as i8))
        .collect::<Option<_>>()
        .ok_or_else(|| format!("illegal エントリの座標が不正: {e}"))?;
    let role =
        parse_foul_code(&e[4..]).ok_or_else(|| format!("illegal エントリの駒コードが不正: {e}"))?;
    let to = coord(digits[2], digits[3]).ok_or_else(|| format!("illegal の移動先が不正: {e}"))?;
    if digits[0] == 0 && digits[1] == 0 {
        Ok(RawFoul::Drop { role, to })
    } else {
        let from =
            coord(digits[0], digits[1]).ok_or_else(|| format!("illegal の移動元が不正: {e}"))?;
        Ok(RawFoul::Board { from, to, role })
    }
}

/// 指し手行の本体（手数の後ろ、消費時間の前）をパースする。
/// 終局宣言なら None を返す
fn parse_move_body(body: &[char], prev_to: Option<Coord>) -> Result<Option<KifMove>, String> {
    let s: String = body.iter().collect();
    for term in ["投了", "反則負け", "時間切れ", "中断", "千日手", "持将棋"] {
        if s.starts_with(term) {
            return Ok(None);
        }
    }
    let mut i = 0;
    let to = if body.first() == Some(&'同') {
        i += 1;
        prev_to.ok_or("同 の前に指し手がありません")?
    } else {
        let f = body
            .get(i)
            .and_then(|&c| parse_file_digit(c))
            .ok_or_else(|| format!("移動先の筋が読めません: {s}"))?;
        let r = body
            .get(i + 1)
            .and_then(|&c| parse_kanji_rank(c))
            .ok_or_else(|| format!("移動先の段が読めません: {s}"))?;
        i += 2;
        coord(f, r).ok_or_else(|| format!("移動先が不正: {s}"))?
    };
    let (role, used) =
        parse_piece_name(&body[i..]).ok_or_else(|| format!("駒名が読めません: {s}"))?;
    i += used;
    let mut promote = false;
    let mut drop = false;
    loop {
        match body.get(i) {
            Some('成') => {
                promote = true;
                i += 1;
            }
            Some('不') if body.get(i + 1) == Some(&'成') => {
                i += 2;
            }
            Some('打') => {
                drop = true;
                i += 1;
            }
            // 相対位置の修飾語（移動元が付くので判別には不要。読み飛ばす）
            Some('右' | '左' | '直' | '引' | '寄' | '上' | '行') => {
                i += 1;
            }
            _ => break,
        }
    }
    if drop {
        return Ok(Some(KifMove::Drop { role, to }));
    }
    // 移動元 (筋段)
    if body.get(i) != Some(&'(') {
        // Shogi Quest は打ち以外に必ず移動元を付ける
        return Err(format!("移動元がありません: {s}"));
    }
    let f = body
        .get(i + 1)
        .and_then(|&c| parse_file_digit(c))
        .ok_or_else(|| format!("移動元の筋が読めません: {s}"))?;
    let r = body
        .get(i + 2)
        .and_then(|&c| parse_file_digit(c))
        .ok_or_else(|| format!("移動元の段が読めません: {s}"))?;
    let from = coord(f, r).ok_or_else(|| format!("移動元が不正: {s}"))?;
    Ok(Some(KifMove::Board { from, to, promote }))
}

pub fn parse_kif(text: &str) -> Result<Kifu, String> {
    let mut kifu = Kifu::default();
    let mut prev_to: Option<Coord> = None;
    let mut ended = false;
    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        let err = |msg: String| format!("{}行目: {msg}", lineno + 1);
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("*illegal:") {
            let mut fouls = vec![];
            for entry in rest.split(',') {
                fouls.push(parse_foul_entry(entry).map_err(err)?);
            }
            if ended {
                kifu.trailing_fouls.extend(fouls);
            } else if let Some(last) = kifu.plies.last_mut() {
                // 規約: 直前の指し手行の手番側が、その手を指す前に試みた反則
                last.fouls.extend(fouls);
            } else {
                return Err(err("最初の指し手の前に *illegal 行があります".into()));
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("*scenario") {
            let rest = rest.trim();
            let mut pos = 0;
            while pos < rest.len() {
                let seg = &rest[pos..];
                let token_end = seg.find(char::is_whitespace).unwrap_or(seg.len());
                let token = &seg[..token_end];
                if let Some(v) = token.strip_prefix("desc=") {
                    // desc は行末まで
                    let full = format!("{v}{}", &seg[token_end..]);
                    kifu.directives.insert("desc".into(), full);
                    break;
                }
                if let Some((k, v)) = token.split_once('=') {
                    kifu.directives.insert(k.into(), v.into());
                }
                pos += token_end;
                pos += rest[pos..].len() - rest[pos..].trim_start().len();
            }
            continue;
        }
        if line.starts_with('*') {
            continue; // その他のコメント
        }
        // 指し手行: 先頭が手数
        let digits: String = line.chars().take_while(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            continue; // ヘッダ行
        }
        let move_no: usize = digits
            .parse()
            .map_err(|_| err(format!("手数が読めません: {line}")))?;
        if ended {
            return Err(err(format!("終局後に指し手行があります: {line}")));
        }
        if move_no != kifu.plies.len() + 1 {
            return Err(err(format!(
                "手数が不連続です（期待 {} 実際 {move_no}）。棋譜の欠落を確認してください",
                kifu.plies.len() + 1
            )));
        }
        // 手数の後ろ全体を渡す。パーサーは1手ぶん読んだところで止まるので、
        // 後続の消費時間欄（空白の幅や形式に依存しない）は無視される
        let body: Vec<char> = line[digits.len()..]
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        match parse_move_body(&body, prev_to).map_err(err)? {
            Some(mv) => {
                prev_to = Some(match mv {
                    KifMove::Board { to, .. } | KifMove::Drop { to, .. } => to,
                });
                kifu.plies.push(KifPly { mv, fouls: vec![] });
            }
            None => ended = true,
        }
    }
    Ok(kifu)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usi_seq(kifu: &Kifu) -> Vec<String> {
        kifu.plies.iter().map(|p| p.mv.to_usi()).collect()
    }

    #[test]
    fn 指し手行の基本形をパースできる() {
        let kifu = parse_kif(
            "1 ７六歩(77)  ( 0:00/00:00:00)\n\
             2 ３二銀(31)  ( 0:01/00:00:01)\n\
             3 ２二角成(88)  ( 0:01/00:00:01)\n\
             4 同　銀(32)  ( 0:00/00:00:01)\n\
             5 ４五角打  ( 0:01/00:00:02)\n",
        )
        .unwrap();
        assert_eq!(usi_seq(&kifu), ["7g7f", "3a3b", "8h2b+", "3b2b", "B*4e"]);
    }

    #[test]
    fn 成駒の移動と不成をパースできる() {
        let kifu = parse_kif(
            "1 ７六歩(77)\n\
             2 ８八と(87)\n\
             3 ５五龍(59)\n\
             4 ４三歩不成(44)\n",
        )
        .unwrap();
        assert_eq!(usi_seq(&kifu), ["7g7f", "8g8h", "5i5e", "4d4c"]);
    }

    #[test]
    fn illegal行は直前の手の反則試行として付く() {
        let kifu = parse_kif(
            "1 ７六歩(77)\n\
             2 ８八と(87)\n\
             *illegal:8285HI,0083FU\n\
             3 投了\n\
             *illegal:5945OU\n",
        )
        .unwrap();
        assert_eq!(kifu.plies.len(), 2);
        assert_eq!(
            kifu.plies[1].fouls,
            vec![
                RawFoul::Board {
                    from: Coord { file: 8, rank: 2 },
                    to: Coord { file: 8, rank: 5 },
                    role: Role::Rook,
                },
                RawFoul::Drop {
                    role: Role::Pawn,
                    to: Coord { file: 8, rank: 3 },
                },
            ]
        );
        assert_eq!(kifu.trailing_fouls.len(), 1);
    }

    #[test]
    fn scenarioディレクティブを読める() {
        let kifu = parse_kif(
            "*scenario ply=69 diag=5g,4h desc=70手目 角成の実験\n\
             1 ７六歩(77)\n",
        )
        .unwrap();
        assert_eq!(kifu.directives.get("ply").unwrap(), "69");
        assert_eq!(kifu.directives.get("diag").unwrap(), "5g,4h");
        assert_eq!(kifu.directives.get("desc").unwrap(), "70手目 角成の実験");
    }

    #[test]
    fn 手数の欠落を検出する() {
        let err = parse_kif("1 ７六歩(77)\n3 ７五歩(76)\n").unwrap_err();
        assert!(err.contains("不連続"), "{err}");
    }

    #[test]
    fn 消費時間欄の空白幅に依存しない() {
        // 1スペース・タブ・時間欄なし、いずれも同じ結果になる
        for line in [
            "1 ７六歩(77) ( 0:00/00:00:00)",
            "1 ７六歩(77)\t( 0:00/00:00:00)",
            "1 ７六歩(77)",
        ] {
            let kifu = parse_kif(line).unwrap();
            assert_eq!(usi_seq(&kifu), ["7g7f"], "{line}");
        }
    }

    #[test]
    fn 相対位置の修飾語と全角の移動元を受け付ける() {
        let kifu = parse_kif(
            "1 ７六歩(77)\n\
             2 ３二金右(41)\n\
             3 ２二角成(８８)\n",
        )
        .unwrap();
        assert_eq!(usi_seq(&kifu), ["7g7f", "4a3b", "8h2b+"]);
    }

    #[test]
    fn 異常な手数はエラーになる() {
        let err = parse_kif("99999999999999999999999 ７六歩(77)\n").unwrap_err();
        assert!(err.contains("手数"), "{err}");
    }
}
