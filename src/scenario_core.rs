//! 局面再現シナリオの共通部品。
//!
//! `.kif`（Shogi Quest エクスポート + `*scenario` ディレクティブ）の読み込み・
//! 裁定つきリプレイ・一手選択の試行を、bin/scenario.rs（CLI）と
//! scenario-gui（Tauri デバッグGUI）が共有する。

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::board::{make_usi_drop, make_usi_move, make_usi_square, parse_usi_square};
use crate::kifu::{Kifu, RawFoul, parse_kif};
use crate::observation::{Observation, ObservationLog};
use crate::protocol::{ClockState, Color, FoulCounts, GameStatus, PlayerView};
use crate::shogi::{Position, ShogiMove, parse_usi, unpromote_role};
use crate::strategy;

pub struct Scenario {
    pub name: String,
    pub desc: String,
    /// 注目している手（一致したら出力に印をつける）。既定は棋譜の ply+1 手目
    pub target: String,
    /// 何手目まで再生するか（ply+1 手目を考えさせる）
    pub ply: usize,
    /// diag で相手駒の利き枚数分布を測るマス
    pub diag_squares: Vec<String>,
    /// continue の足切り手数（**通算**の手数。必勝局面の遂行実験で、これを
    /// 超えたら不合格 = 引き分け扱いで打ち切る）。既定 200
    pub limit: u32,
    pub kifu: Kifu,
}

pub fn scenarios_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("scenarios")
}

pub fn load_scenario(
    spec: &str,
    ply_flag: Option<usize>,
    target_flag: Option<String>,
    diag_flag: Option<String>,
) -> Result<Scenario, String> {
    let path = if spec.contains('/') || spec.ends_with(".kif") {
        PathBuf::from(spec)
    } else {
        scenarios_dir().join(format!("{spec}.kif"))
    };
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("{} を読めません: {e}", path.display()))?;
    let kifu = parse_kif(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let directive_ply: Option<usize> = kifu.directives.get("ply").and_then(|s| s.parse().ok());
    let ply = ply_flag
        .or(directive_ply)
        .ok_or("再生する手数が不明です（--ply か *scenario ply= を指定）")?;
    if ply > kifu.plies.len() {
        return Err(format!(
            "ply={ply} が棋譜の手数 {} を超えています",
            kifu.plies.len()
        ));
    }
    let target = target_flag
        .or_else(|| kifu.directives.get("target").cloned())
        .or_else(|| kifu.plies.get(ply).map(|p| p.mv.to_usi()))
        .unwrap_or_default();
    let diag_squares: Vec<String> = diag_flag
        .or_else(|| kifu.directives.get("diag").cloned())
        .map(|s| {
            s.split(',')
                .map(|x| x.trim())
                .filter(|x| !x.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    for sq in &diag_squares {
        parse_usi_square(sq).ok_or_else(|| format!("diag のマスを読めません: {sq}"))?;
    }
    let limit: u32 = kifu
        .directives
        .get("limit")
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| spec.to_string());
    // --ply で局面を変えたときは directive の説明（元の ply 前提）を使わない
    let desc = kifu
        .directives
        .get("desc")
        .filter(|_| Some(ply) == directive_ply)
        .cloned()
        .unwrap_or_else(|| format!("{ply}手目まで再生し、{}手目を考えさせる", ply + 1));
    Ok(Scenario {
        name,
        desc,
        target,
        ply,
        diag_squares,
        limit,
        kifu,
    })
}

/// リプレイ結果: 真実の局面と両者の観測ログ・反則数。[0]=先手, [1]=後手
pub struct Replayed {
    pub pos: Position,
    pub logs: [ObservationLog; 2],
    pub fouls: [u32; 2],
    pub plies: u32,
}

pub fn side_idx(c: Color) -> usize {
    if c == Color::Sente { 0 } else { 1 }
}

/// 反則試行を USI に解決する。駒コードは「移動後の駒」なので、盤上の移動元が
/// 生駒でコードが成駒なら成る手と判断する
pub fn resolve_foul(pos: &Position, side: Color, f: &RawFoul) -> String {
    match f {
        RawFoul::Drop { role, to } => {
            make_usi_drop(*role, *to).expect("打てない駒種の反則試行")
        }
        RawFoul::Board { from, to, role } => {
            let piece = pos
                .piece_at(*from)
                .expect("反則試行の移動元に駒がない（棋譜とKIFの不整合）");
            assert_eq!(piece.color, side, "反則試行の移動元が相手の駒");
            assert_eq!(
                unpromote_role(piece.role),
                unpromote_role(*role),
                "反則試行の駒コードと盤上の駒種が不一致"
            );
            // 駒コードは移動後の駒種: 盤上が生駒でコードが成駒なら成る手。
            // 盤上が成駒でコードが生駒に戻る組み合わせは存在しない（KIF不整合）
            let piece_promoted = piece.role != unpromote_role(piece.role);
            let code_promoted = *role != unpromote_role(*role);
            assert!(
                piece_promoted <= code_promoted,
                "反則試行のコードが生駒なのに盤上は成駒（KIF不整合）: {from:?}"
            );
            make_usi_move(*from, *to, code_promoted && !piece_promoted)
        }
    }
}

/// 棋譜（反則試行込み）を upto 手まで裁定つきでリプレイし、selfplay.rs と
/// 同じ規約で両者の観測ログを構築する
pub fn replay(kifu: &Kifu, upto: usize) -> Replayed {
    let mut pos = Position::initial();
    let mut logs = [ObservationLog::default(), ObservationLog::default()];
    let mut fouls = [0u32; 2];
    for ply in &kifu.plies[..upto] {
        let side = pos.turn();
        for f in &ply.fouls {
            let usi = resolve_foul(&pos, side, f);
            let mv = parse_usi(&usi).expect("反則試行のUSI解析失敗");
            assert!(!pos.is_legal(&mv), "反則のはずの手が合法: {usi}");
            fouls[side_idx(side)] += 1;
            logs[side_idx(side)].record(Observation::MyFoul {
                move_number: pos.move_number(),
                usi,
            });
            logs[side_idx(side.other())].record(Observation::OpponentFoul {
                count: fouls[side_idx(side)],
            });
        }
        let usi = ply.mv.to_usi();
        let mv = parse_usi(&usi).expect("USI解析失敗");
        assert!(pos.is_legal(&mv), "棋譜の手が非合法: {usi}");
        let captured = pos.play_unchecked(&mv);
        let move_number = pos.move_number();
        let captured_sq = captured.map(|_| match mv {
            ShogiMove::Board { to, .. } => make_usi_square(to),
            ShogiMove::Drop { .. } => unreachable!("打ちでは駒を取れない"),
        });
        logs[side_idx(side)].record(Observation::MyMove {
            move_number,
            usi,
            captured: captured.map(unpromote_role),
        });
        logs[side_idx(side.other())].record(Observation::OpponentMoved {
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
    Replayed {
        pos,
        logs,
        fouls,
        plies: upto as u32,
    }
}

pub fn clone_log(log: &ObservationLog) -> ObservationLog {
    let mut out = ObservationLog::default();
    for e in log.events() {
        out.record(e.clone());
    }
    out
}

pub fn make_view(pos: &Position, color: Color, fouls: &[u32; 2]) -> PlayerView {
    PlayerView {
        game_id: "scenario".into(),
        your_color: color,
        your_pieces: pos.pieces_of(color),
        your_hand: pos.hand_map(color),
        turn: pos.turn(),
        move_number: pos.move_number(),
        clocks: ClockState {
            sente_ms: 900_000,
            gote_ms: 900_000,
            running: Some(pos.turn()),
            server_time: 0,
        },
        fouls: FoulCounts {
            you: fouls[side_idx(color)],
            opponent: fouls[side_idx(color.other())],
        },
        you_in_check: pos.in_check(color),
        opponent_in_check: pos.in_check(color.other()),
        status: GameStatus::Playing,
    }
}

pub struct ChoiceStats {
    /// (受理された手, 回数) を回数降順で
    pub tally: Vec<(String, u32)>,
    pub total_fouls: u32,
}

impl ChoiceStats {
    pub fn target_hits(&self, target: &str) -> u32 {
        self.tally
            .iter()
            .find(|(usi, _)| usi == target)
            .map(|(_, n)| *n)
            .unwrap_or(0)
    }
}

/// 手番側の一手の選択を1シードぶん試行する。反則は観測として与えて指し直させる
/// （実対局と同じ）。返り値は (受理された手, その前に試みた反則列)。
/// 指せる手がなければ "resign"、反則累計10回で "foul_limit"
pub fn choice_trial_one(rep: &Replayed, seed: u64, name: &str) -> (String, Vec<String>) {
    let side = rep.pos.turn();
    let mut strat = strategy::make_seeded(name, seed).expect("未知の戦略名");
    let mut log = clone_log(&rep.logs[side_idx(side)]);
    strategy::prewarm_strategy(&mut *strat, &make_view(&rep.pos, side, &rep.fouls), &log);
    let mut foul_tried: HashSet<String> = HashSet::new();
    let mut fouls = rep.fouls;
    let mut foul_seq: Vec<String> = vec![];
    let accepted = loop {
        let view = make_view(&rep.pos, side, &fouls);
        let Some(usi) = strat.choose(&view, &log, &foul_tried) else {
            break "resign".to_string();
        };
        let legal = parse_usi(&usi).is_some_and(|mv| rep.pos.is_legal(&mv));
        if legal {
            break usi;
        }
        fouls[side_idx(side)] += 1;
        log.record(Observation::MyFoul {
            move_number: rep.pos.move_number(),
            usi: usi.clone(),
        });
        foul_tried.insert(usi.clone());
        foul_seq.push(usi);
        if fouls[side_idx(side)] >= 10 {
            break "foul_limit".to_string();
        }
    };
    (accepted, foul_seq)
}

/// 現行 estimator（last_ranking 実装済みの戦略）で1回だけ choose を実行し、
/// (選択した手, 全候補の評価内訳スコア降順) を返す。定跡で指した手番・
/// 候補ゼロ（投了）ではランキングが取れないので None
pub fn ranking_one(
    rep: &Replayed,
    seed: u64,
    name: &str,
) -> Option<(String, Vec<strategy::CandidateScore>)> {
    let side = rep.pos.turn();
    let mut strat = strategy::make_seeded(name, seed).expect("未知の戦略名");
    let log = clone_log(&rep.logs[side_idx(side)]);
    strategy::prewarm_strategy(&mut *strat, &make_view(&rep.pos, side, &rep.fouls), &log);
    let view = make_view(&rep.pos, side, &rep.fouls);
    let chosen = strat.choose(&view, &log, &HashSet::new())?;
    let ranking = strat.last_ranking()?.to_vec();
    Some((chosen, ranking))
}

/// 手番側の一手の選択を seed 0..trials で試行して集計する。
/// `on_trial(seed, 受理された手, 反則列)` が1試行終わるごとに呼ばれる
/// （CLI の逐次表示・GUI の進捗イベント用）
pub fn choice_trials(
    rep: &Replayed,
    trials: u64,
    name: &str,
    mut on_trial: impl FnMut(u64, &str, &[String]),
) -> ChoiceStats {
    let mut final_tally: HashMap<String, u32> = HashMap::new();
    let mut total_fouls = 0u32;
    for seed in 0..trials {
        let (accepted, foul_seq) = choice_trial_one(rep, seed, name);
        on_trial(seed, &accepted, &foul_seq);
        *final_tally.entry(accepted).or_insert(0) += 1;
        total_fouls += foul_seq.len() as u32;
    }
    let mut tally: Vec<_> = final_tally.into_iter().collect();
    tally.sort_by(|a, b| b.1.cmp(&a.1));
    ChoiceStats { tally, total_fouls }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(name: &str) -> Scenario {
        load_scenario(
            &scenarios_dir().join(format!("{name}.kif")).to_string_lossy(),
            None,
            None,
            None,
        )
        .unwrap()
    }

    /// 手動翻訳で検証済みだった USI 列とパーサーの出力が一致すること
    #[test]
    fn keimaの棋譜はUSI列と反則が既知の正解に一致する() {
        let sc = load("keima");
        let expected = [
            "7g7f", "3a3b", "5g5f", "2b3a", "5f5e", "5a6b", "2h5h", "5c5d", "5i4h",
            "7c7d", "7i6h", "8c8d", "6h5g", "6b7c", "5g5f", "6c6d", "4h3h", "9c9d",
            "6i6h", "9d9e", "6h5g", "9e9f", "4g4f", "9f9g+", "8h6f", "P*9h", "8i7g",
            "9h9i+", "7g8e", "8d8e",
        ];
        let usi: Vec<String> = sc.kifu.plies.iter().map(|p| p.mv.to_usi()).collect();
        assert_eq!(usi, expected);
        assert_eq!(sc.ply, 29);
        assert_eq!(sc.target, "8d8e"); // 30手目（同歩）が自動導出される
        // 30手目の前の反則試行 = 6465FU
        assert_eq!(sc.kifu.plies[29].fouls.len(), 1);
    }

    #[test]
    fn kakunariの棋譜はUSI列と反則が既知の正解に一致する() {
        let sc = load("kakunari");
        let expected = [
            "7g7f", "3a3b", "6i7h", "1c1d", "7h7g", "2b1c", "4i5h", "5a4b", "6g6f",
            "7c7d", "5h6g", "8a7c", "5g5f", "8c8d", "8g8f", "8d8e", "4g4f", "8e8f",
            "7i7h", "8f8g+", "4f4e", "8g8h", "4e4d", "8h8i", "4d4c+", "3b4c", "P*8c",
            "8b8c", "7h8g", "8i7i", "2h8h", "7i6i", "5i5h", "P*8f", "8h8i", "8f8g+",
            "7g8g", "8c8g+", "8i8g", "P*8e", "8g8i", "4c3d", "8i6i", "N*5g", "6i8i",
            "B*6i", "8i6i", "5g6i+", "5h6i", "1c7i", "6i5h", "R*6i", "P*4f", "P*4h",
            "R*4e", "4b3b", "4e8e", "7c8e", "N*5d", "8e7g+", "5h4g", "4h4i+", "B*2b",
            "6i5i+", "2b1a+", "R*5h", "L*4c", "G*4e", "4c4a+", "7i5g+", "6g5g",
            "5h5g+", "4g3h", "4i3i", "3h2h", "5i4h",
        ];
        let usi: Vec<String> = sc.kifu.plies.iter().map(|p| p.mv.to_usi()).collect();
        assert_eq!(usi, expected);
        assert_eq!(sc.ply, 69);
        assert_eq!(sc.target, "7i5g+"); // 70手目が自動導出される
        // 反則試行の総数（69手目まで7件 + 71/73/75手目の前に4件。終局後の4件は trailing）
        let n_fouls: usize = sc.kifu.plies.iter().map(|p| p.fouls.len()).sum();
        assert_eq!(n_fouls, 11);
        assert_eq!(sc.kifu.trailing_fouls.len(), 4);
    }

    /// last_ranking（scenario-gui のランキング表示）の結合検証。
    /// debug ビルドでは思考が遅すぎるので release で実行する:
    /// `cargo test --release -- --ignored ランキング`
    #[test]
    #[ignore]
    fn kakutoriのランキングは注目手の捕獲を候補に含む() {
        let sc = load("kakutori");
        let rep = replay(&sc.kifu, sc.ply);
        let (chosen, ranking) =
            ranking_one(&rep, 0, "estimator").expect("estimator はランキングを返す");
        assert!(!ranking.is_empty());
        for w in ranking.windows(2) {
            assert!(w[0].score >= w[1].score, "スコア降順でない");
        }
        assert_eq!(ranking[0].usi, chosen, "先頭候補と選択手が一致しない");
        assert!(
            ranking.iter().any(|c| c.usi == sc.target),
            "注目手 {} が候補にない",
            sc.target
        );
    }

    /// リプレイの裁定検証（合法手は合法・反則試行は非合法）が全編通ること
    #[test]
    fn 収録シナリオは裁定つきリプレイが通る() {
        for name in ["keima", "kakunari"] {
            let sc = load(name);
            let rep = replay(&sc.kifu, sc.kifu.plies.len());
            assert!(rep.plies > 0);
        }
        // kakunari は後手5反則・先手2反則で70手目を迎える
        let sc = load("kakunari");
        let rep = replay(&sc.kifu, sc.ply);
        assert_eq!(rep.fouls, [2, 5]);
        assert_eq!(rep.pos.turn(), Color::Gote);
    }
}
