//! エンコーダの入力プレーンを、人が読める見出しつきで取り出す（scenario-gui の対局モードの「入力」表示用）。
//!
//! 値は `encode::encode_planes` が書いたものそのまま（学習・推論が読む数字と同じ）。変えるのは盤の向きだけで、
//! エンコーダの「手番側を先手向きに正規化した座標」から実際のマスへ戻す。並び（どの番号が何か）は
//! `encode.rs` の定数から引くので、プレーンを足したときは `layout` に1行足す（足し忘れはテストが落とす）

use std::collections::{BTreeMap, HashSet};

use serde::Serialize;

use crate::board::make_usi_square;
use crate::observation::ObservationLog;
use crate::protocol::PlayerView;
use crate::rl::action::{index_square, normalize};
use crate::rl::encode::{
    self as e, HISTORY_MOVES, NUM_PLANES, NUM_PLANES_DEDUCED, NUM_PLANES_V3, NUM_PLANES_V4, NUM_PLANES_V5, NUM_PLANES_V6,
    NUM_PLANES_V7,
};
use crate::selfplay::{MAX_FOULS, MAX_PLIES};

/// 駒種の見出し（`encode::ROLES` の順。先頭の7つは持ち駒の駒種 `HAND_ROLES` の順でもある）
const KANJI: [&str; 14] = [
    "歩", "香", "桂", "銀", "金", "角", "飛", "玉", "と", "成香", "成桂", "成銀", "馬", "龍",
];

/// 入力プレーン1枚ぶんの表示
#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PlaneView {
    /// プレーン番号（0 始まり）。いまは必ず Some（入力には無い参考の盤を並べるときのために Option のまま）
    pub index: Option<usize>,
    /// 組の見出し
    pub group: &'static str,
    pub label: String,
    /// 足したエンコーダの版（"v1"〜"v7"）
    pub version: &'static str,
    /// 全面に同じ値を入れるプレーンなら、その値
    pub scalar: Option<f32>,
    /// 値の読み方（「3枚」「2回」など。分かるものだけ）
    pub note: Option<String>,
    /// マスごとの値（USI のマス → 値。0 のマスは入れない）。全面のプレーンでは空
    pub cells: BTreeMap<String, f32>,
}

/// 全面のプレーンの値の読み方
#[derive(Clone, Copy)]
enum Unit {
    /// マスごとのプレーン
    Board,
    /// 値 × 総数 = 枚数
    Pieces(u32),
    /// 値 × 反則の上限 = 回数
    Fouls,
    /// 0 / 1
    Flag,
    /// 値 × 手数の上限 = 手数
    Plies,
    /// 値 × 上限 = 相手の手数（上限で張り付く）
    Moves(u32),
    /// 鮮度などそのままの値
    Raw,
}

struct Spec {
    index: usize,
    group: &'static str,
    label: String,
    unit: Unit,
}

const G_OWN: &str = "自分の駒";
const G_HIST: &str = "自分の手の履歴";
const G_TRADE: &str = "駒の取り合い";
const G_FOUL: &str = "反則";
const G_FACT: &str = "王手と確定事実";
const G_REACH: &str = "自駒の利きと自玉の筋";
const G_COUNT: &str = "持ち駒と相手の駒数（全面）";
const G_STATE: &str = "局面の数値（全面）";

/// 全プレーンの見出し（表示する順）
fn layout() -> Vec<Spec> {
    let mut out: Vec<Spec> = vec![];
    let mut add = |index: usize, group: &'static str, label: String, unit: Unit| {
        out.push(Spec { index, group, label, unit });
    };
    // 自分の駒
    for (i, k) in KANJI.iter().enumerate() {
        add(e::P_OWN + i, G_OWN, format!("自駒 {k}"), Unit::Board);
    }
    add(e::P_UNMOVED, G_OWN, "初期配置から動いていない自駒".into(), Unit::Board);
    // 自分の手の履歴
    for k in 0..HISTORY_MOVES {
        add(e::P_HIST + 2 * k, G_HIST, format!("{}手前の移動元", k + 1), Unit::Board);
        add(e::P_HIST + 2 * k + 1, G_HIST, format!("{}手前の移動先", k + 1), Unit::Board);
    }
    // 駒の取り合い
    add(e::P_CAP, G_TRADE, "駒を取ったマス（鮮度）".into(), Unit::Board);
    for (i, k) in KANJI[..7].iter().enumerate() {
        add(e::P_CAP_ROLE + i, G_TRADE, format!("{k}を取ったマス（鮮度）"), Unit::Board);
    }
    for (i, k) in KANJI[..7].iter().enumerate() {
        add(e::E_CAP_ROLE + i, G_TRADE, format!("{k}を取ったマス（回数）"), Unit::Board);
    }
    add(e::E_PAWN_FILE, G_TRADE, "相手の歩を取った筋（回数）".into(), Unit::Board);
    add(e::P_LOST, G_TRADE, "自駒を取られたマス（鮮度）".into(), Unit::Board);
    add(e::P_LOST_COUNT, G_TRADE, "自駒を取られたマス（回数）".into(), Unit::Board);
    for (i, k) in KANJI[..7].iter().enumerate() {
        add(e::E_LOST_ROLE + i, G_TRADE, format!("{k}を取られたマス（回数）"), Unit::Board);
    }
    // 反則
    let foul_parts = |base: usize, when: &str| -> Vec<(usize, String)> {
        let mut v = vec![
            (base + e::F_FROM, format!("{when}の反則: 移動元")),
            (base + e::F_TO, format!("{when}の反則: 移動先")),
            (base + e::F_PROMO_TO, format!("{when}の反則: 成りの移動先")),
        ];
        for (i, k) in KANJI[..7].iter().enumerate() {
            v.push((base + e::F_DROP + i, format!("{when}の反則: {k}の打ち先")));
        }
        v
    };
    for (i, label) in foul_parts(e::P_FOUL, "今の手番") {
        add(i, G_FOUL, label, Unit::Board);
    }
    for (i, label) in foul_parts(e::P_PAST_FOUL, "過去") {
        add(i, G_FOUL, format!("{label}（鮮度）"), Unit::Board);
    }
    add(e::E_KING_FOUL, G_FOUL, "自玉の移動で反則したマス（鮮度）".into(), Unit::Board);
    // 王手と確定事実
    add(e::P_CHECKED_KING, G_FACT, "王手されたときの自玉の位置（鮮度）".into(), Unit::Board);
    add(e::P_GAVE_CHECK, G_FACT, "王手をかけた手の移動先（鮮度）".into(), Unit::Board);
    add(e::D_OCC, G_FACT, "相手の駒がいると確定したマス".into(), Unit::Board);
    add(e::D_BLOCK, G_FACT, "飛び越え反則の間のマス（1/マス数）".into(), Unit::Board);
    add(e::D_CHECKER, G_FACT, "王手駒がいうるマス（1/候補数）".into(), Unit::Board);
    add(e::E_CHECKER_MEMO, G_FACT, "逃げた王手駒がいたかもしれないマス（鮮度÷候補数）".into(), Unit::Board);
    add(e::D_KING, G_FACT, "相手玉がいうるマス".into(), Unit::Board);
    add(e::E_KING_TIGHT, G_FACT, "相手玉がいうるマス（締めた版）".into(), Unit::Board);
    add(e::D_PAWN, G_FACT, "動けないと確定した相手の歩".into(), Unit::Board);
    // 相手に位置が伝わっている自駒
    add(e::E_EXPOSED, G_OWN, "相手に伝わっている自駒: 駒を取ったまま動いていない（鮮度）".into(), Unit::Board);
    add(e::E_EXPOSED + 1, G_OWN, "相手に伝わっている自駒: 王手をかけた手で動かしたまま（鮮度）".into(), Unit::Board);
    // 自駒の利きと自玉の筋
    add(e::E_OWN_ATTACK, G_REACH, "自駒の確実な利き（枚数×1/3）".into(), Unit::Board);
    add(e::E_OWN_ATTACK + 1, G_REACH, "自駒の遮られうる利き（飛び駒の2マス目以降。枚数×1/3）".into(), Unit::Board);
    add(e::E_KING_LINE, G_REACH, "自玉の筋（最初の自駒の手前まで）".into(), Unit::Board);
    add(e::E_KING_LINE + 1, G_REACH, "ピンされうる自駒（自玉の筋の最初の自駒）".into(), Unit::Board);
    // 飛び駒の線の先の確定駒（方向は手番側から見た向き。値 = (9 − 距離) / 8）
    for (i, dir) in ["前", "左前", "左", "左後ろ", "後ろ", "右後ろ", "右", "右前"].iter().enumerate() {
        add(e::E_RAY_HIT + i, G_REACH, format!("飛び駒の線の先の確定駒: {dir}（移動元に (9−距離)/8）"), Unit::Board);
    }
    // 持ち駒と相手の駒数
    for (i, k) in KANJI[..7].iter().enumerate() {
        add(e::P_HAND + i, G_COUNT, format!("自分の持ち駒 {k}"), Unit::Pieces(e::ROLE_TOTAL[i]));
    }
    for (i, k) in KANJI[..7].iter().enumerate() {
        add(e::P_OPP_HOLD + i, G_COUNT, format!("相手の保有数（盤上＋持ち駒）{k}"), Unit::Pieces(e::ROLE_TOTAL[i]));
    }
    for (i, k) in KANJI[..7].iter().enumerate() {
        add(e::E_HAND + i, G_COUNT, format!("相手の持ち駒の上界 {k}"), Unit::Pieces(e::ROLE_TOTAL[i]));
    }
    for (i, k) in KANJI[..7].iter().enumerate() {
        add(e::E_LOST_FRESH + i, G_COUNT, format!("{k}を最後に取られてからの鮮度"), Unit::Raw);
    }
    // 局面の数値
    let s = e::P_SCALAR;
    add(s + e::S_MY_FOULS, G_STATE, "自分の反則数".into(), Unit::Fouls);
    add(s + e::S_OPP_FOULS, G_STATE, "相手の反則数".into(), Unit::Fouls);
    add(s + e::S_MY_LAST_FOUL, G_STATE, "自分があと1回で反則負け".into(), Unit::Flag);
    add(s + e::S_OPP_LAST_FOUL, G_STATE, "相手があと1回で反則負け".into(), Unit::Flag);
    add(s + e::S_FOULS_THIS_TURN, G_STATE, "今の手番の反則数".into(), Unit::Fouls);
    add(s + e::S_IN_CHECK, G_STATE, "王手を受けている".into(), Unit::Flag);
    add(s + e::S_GAVE_CHECK_LAST, G_STATE, "直前の自分の手で王手をかけた".into(), Unit::Flag);
    add(s + e::S_MOVE_NUMBER, G_STATE, "手数".into(), Unit::Plies);
    add(s + e::S_HISTORY_BROKEN, G_STATE, "観測の履歴が欠けている".into(), Unit::Flag);
    add(e::E_FOUL + e::EF_LAST, G_STATE, "相手の直前の手番の反則数".into(), Unit::Fouls);
    add(e::E_FOUL + e::EF_CHECKED, G_STATE, "相手の反則: 王手を受けていた手番（鮮度つきの合計）".into(), Unit::Fouls);
    add(e::E_FOUL + e::EF_FREE, G_STATE, "相手の反則: 王手を受けていない手番（鮮度つきの合計）".into(), Unit::Fouls);
    let c = e::E_CHECK_CLOCK;
    add(c + e::CC_GAVE_EVER, G_STATE, "王手をかけたことがある".into(), Unit::Flag);
    add(c + e::CC_GAVE_MOVES, G_STATE, "最後に王手をかけてからの相手の手数（まだなら初手から）".into(), Unit::Moves(e::CLOCK_CAP));
    add(c + e::CC_GAVE_FREE, G_STATE, "　そのうち自由な手（応手・駒取り・王手でない手）".into(), Unit::Moves(e::CLOCK_CAP));
    add(c + e::CC_GOT_EVER, G_STATE, "王手を受けたことがある".into(), Unit::Flag);
    add(c + e::CC_GOT_MOVES, G_STATE, "最後に王手を受けてからの相手の手数（まだなら初手から）".into(), Unit::Moves(e::CLOCK_CAP));
    add(c + e::CC_GOT_FREE, G_STATE, "　そのうち自由な手（応手・駒取り・王手でない手）".into(), Unit::Moves(e::CLOCK_CAP));
    let (o, caps) = (e::E_OPP_COUNT, e::OPP_COUNT_CAPS);
    add(o + e::OC_MOVES, G_STATE, "相手が指した手の合計".into(), Unit::Moves(caps[e::OC_MOVES]));
    add(o + e::OC_CHECKS, G_STATE, "　うち王手をかけてきた手".into(), Unit::Moves(caps[e::OC_CHECKS]));
    add(o + e::OC_CAPTURES, G_STATE, "　うち駒を取ってきた手".into(), Unit::Moves(caps[e::OC_CAPTURES]));
    add(o + e::OC_BOTH, G_STATE, "　うち王手をかけながら駒を取ってきた手".into(), Unit::Moves(caps[e::OC_BOTH]));
    out
}

fn version_of(index: usize) -> &'static str {
    match index {
        i if i < NUM_PLANES => "v1",
        i if i < NUM_PLANES_DEDUCED => "v2",
        i if i < NUM_PLANES_V3 => "v3",
        i if i < NUM_PLANES_V4 => "v4",
        i if i < NUM_PLANES_V5 => "v5",
        i if i < NUM_PLANES_V6 => "v6",
        _ => "v7",
    }
}

fn note_of(unit: Unit, v: f32) -> Option<String> {
    // 鮮度つきの合計は整数にならないので、端数があればそのまま見せる
    let count = |x: f32, suffix: &str| {
        if (x - x.round()).abs() < 1e-3 {
            format!("{}{suffix}", x.round() as i64)
        } else {
            format!("{x:.2}{suffix}")
        }
    };
    match unit {
        Unit::Pieces(total) => Some(count(v * total as f32, "枚")),
        Unit::Fouls => Some(count(v * MAX_FOULS as f32, "回")),
        Unit::Plies => Some(count(v * MAX_PLIES as f32, "手目")),
        Unit::Flag => Some(if v > 0.5 { "はい" } else { "いいえ" }.to_string()),
        Unit::Moves(cap) if v >= 1.0 => Some(format!("{cap}手以上")),
        Unit::Moves(cap) => Some(count(v * cap as f32, "手")),
        Unit::Raw | Unit::Board => None,
    }
}

/// `view` の側の観測から作った入力（エンコーダ v7 の全プレーン）を、表示する順に返す
pub fn plane_views(view: &PlayerView, log: &ObservationLog, foul_tried: &HashSet<String>) -> Vec<PlaneView> {
    let color = view.your_color;
    let buf = e::encode_planes(NUM_PLANES_V7, view, log, foul_tried);
    // マス番号（正規化済み）→ 実際のマス。`normalize` は自分自身が逆変換
    let square = |i: usize| make_usi_square(normalize(index_square(i), color));
    layout()
        .into_iter()
        .map(|spec| {
            let plane = &buf[spec.index * 81..(spec.index + 1) * 81];
            let (scalar, cells) = match spec.unit {
                Unit::Board => (
                    None,
                    plane
                        .iter()
                        .enumerate()
                        .filter(|(_, v)| **v != 0.0)
                        .map(|(i, v)| (square(i), *v))
                        .collect(),
                ),
                _ => (Some(plane[0]), BTreeMap::new()),
            };
            PlaneView {
                index: Some(spec.index),
                group: spec.group,
                label: spec.label,
                version: version_of(spec.index),
                note: scalar.and_then(|v| note_of(spec.unit, v)),
                scalar,
                cells,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{Color, Role};
    use crate::rl::encode::EXTRA_PLANE_GROUPS;
    use crate::scenario_core::{load_scenario, make_view, replay, scenarios_dir, side_idx};

    #[test]
    fn 見出しは全部のプレーンをちょうど1回ずつ覆う() {
        let mut seen: Vec<usize> = layout().iter().map(|s| s.index).collect();
        seen.sort();
        assert_eq!(seen, (0..NUM_PLANES_V7).collect::<Vec<_>>());
        // 版の境目は v3 以降の組の並びと合う
        for (name, start, n) in EXTRA_PLANE_GROUPS {
            let want = match name {
                "opp_hand" | "opp_fouls" => "v3",
                "king_fouls" => "v4",
                "lost_fresh" | "lost_roles" | "cap_roles" | "pawn_files" => "v5",
                "ray_hits" => "v7",
                _ => "v6",
            };
            for i in start..start + n {
                assert_eq!(version_of(i), want, "{name} の {i}番");
            }
        }
        assert_eq!(version_of(NUM_PLANES - 1), "v1");
        assert_eq!(version_of(NUM_PLANES), "v2");
    }

    /// 実戦の棋譜の全手数・両視点で、「全面」とした組は本当に全マス同じ値で、マスごとのプレーンは
    /// 実際のマス（後手は180度戻したもの）に載っている
    #[test]
    fn 全面の見出しと盤の向きは実際の観測と合う() {
        let path = scenarios_dir().join("archive/king-deduction.kif");
        let kifu = load_scenario(&path.to_string_lossy(), Some(0), None, None).unwrap().kifu;
        let specs = layout();
        let none = HashSet::new();
        for ply in 0..=kifu.plies.len() {
            let rep = replay(&kifu, ply);
            for me in [Color::Sente, Color::Gote] {
                let view = make_view(&rep.pos, me, &rep.fouls);
                let log = &rep.logs[side_idx(me)];
                let buf = e::encode_planes(NUM_PLANES_V7, &view, log, &none);
                for spec in &specs {
                    let plane = &buf[spec.index * 81..(spec.index + 1) * 81];
                    if !matches!(spec.unit, Unit::Board) {
                        assert!(
                            plane.iter().all(|v| *v == plane[0]),
                            "ply={ply} {me:?}: {}番（{}）は全面のはず",
                            spec.index,
                            spec.label
                        );
                    }
                }
                let views = plane_views(&view, log, &none);
                assert_eq!(views.len(), NUM_PLANES_V7);
                let at = |index: usize| views.iter().find(|v| v.index == Some(index)).unwrap();
                // 自玉のプレーンは自玉のマス1つだけ
                let king_role = e::ROLES.iter().position(|r| *r == Role::King).unwrap();
                let my_king = make_usi_square(rep.pos.king_square(me).unwrap());
                let got: Vec<&String> = at(e::P_OWN + king_role).cells.keys().collect();
                assert_eq!(got, [&my_king], "ply={ply} {me:?}: 自玉のマス");
                // 相手玉の候補は真の玉を含み、締めた版はその中に収まる
                let opp_king = make_usi_square(rep.pos.king_square(me.other()).unwrap());
                let cands = &at(e::D_KING).cells;
                assert!(cands.contains_key(&opp_king), "ply={ply} {me:?}: 相手玉の候補");
                let pos = views.iter().position(|v| v.index == Some(e::E_KING_TIGHT)).unwrap();
                assert_eq!(views[pos - 1].index, Some(e::D_KING), "締めた版は元の候補のすぐ後ろ");
                let tight = &views[pos].cells;
                assert!(tight.contains_key(&opp_king), "ply={ply} {me:?}: 締めた候補");
                assert!(tight.keys().all(|k| cands.contains_key(k)), "ply={ply} {me:?}: 締めた候補は元の候補の中");
                // 手数は読める形で付く
                let n = at(e::P_SCALAR + e::S_MOVE_NUMBER);
                assert_eq!(n.note.as_deref(), Some(format!("{}手目", view.move_number).as_str()));
            }
        }
    }
}
