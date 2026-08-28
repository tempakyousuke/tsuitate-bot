//! 局面再現シナリオの共通部品。
//!
//! `.kif`（Shogi Quest エクスポート + `*scenario` ディレクティブ）の読み込み・
//! 裁定つきリプレイ・一手選択の試行を、bin/scenario.rs（CLI）と
//! scenario-gui（Tauri デバッグGUI）が共有する。

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::board::{make_usi_drop, make_usi_move, make_usi_square, parse_usi_square};
use crate::estimator::Estimator;
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
    /// 不合格リスト（`bad=<USI,USI,...>`）: 選んだら悪手として数える手の全量。
    /// kakudo方式（target=悪手）のシナリオで「別の悪手へ逃げただけ」を検出する
    /// ためのもので、target が悪手ならこのリストにも重複して入れる（自己完結。
    /// suite の「不合格計」はこのリストだけで数える）。空なら従来どおり
    pub bad: Vec<String>,
    /// 採点表（`scores=<USI:点,...>`、0〜10）。evals/*.eval.md から
    /// `sync_eval.py` が同期する。SPSA の `TUNE_OBJECTIVE=scenario_score` と
    /// suite の「平均得点」が使う。空なら得点は出さない
    pub scores: Vec<(String, u8)>,
    /// 何手目まで再生するか（ply+1 手目を考えさせる）
    pub ply: usize,
    /// 決定の前に注入する反則列（`fouls=<USI,...>`）。実戦で反則を挟んだ
    /// 決定点の「反則後」の状態を再現する（例: quest_20260731 の30手目 =
    /// 1二飛・2一歩打の反則後に4二金を指した）。choice_trial_body と同じ規約で
    /// MyFoul 観測・反則カウント・foul_tried に反映される
    pub fouls: Vec<String>,
    /// **オラクル錨**（`oracle=<手数>`、bin/scenario の `--oracle N` /
    /// `--oracle-lag K` で上書き）: Some(N) なら手番側の推定器に「N 手目
    /// までの真実の局面」を与え、N+1 手目以降は観測だけで処理させる
    /// （`Estimator::oracle_anchor`）。例: ply=40 oracle=39 = 「39手目までの
    /// 盤面は分かっているが 40手目（相手の直前の手）だけ分からない」状態で
    /// 41手目を考えさせる。粒子生成を丸ごと省くので一瞬で終わり、被王手時の
    /// 悪手が信念（粒子）の問題か評価の問題かを切り分けられる。
    /// **真実を与えるので「確定粒子がなければ分かり得ない手」が出る**ことは
    /// 織り込んで読む。suite の集計上は通常シナリオと区別しないので、
    /// 常設 suite に oracle つきの kif を混ぜないこと（比較用に一時的に使う）
    pub oracle: Option<usize>,
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
    let fouls: Vec<String> = kifu
        .directives
        .get("fouls")
        .map(|s| {
            s.split(',')
                .map(|x| x.trim())
                .filter(|x| !x.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let bad: Vec<String> = kifu
        .directives
        .get("bad")
        .map(|s| {
            s.split(',')
                .map(|x| x.trim())
                .filter(|x| !x.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let scores: Vec<(String, u8)> = kifu
        .directives
        .get("scores")
        .map(|s| {
            s.split(',')
                .filter_map(|x| {
                    let (usi, pt) = x.trim().split_once(':')?;
                    Some((usi.to_string(), pt.parse().ok()?))
                })
                .collect()
        })
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
    let oracle: Option<usize> = match kifu.directives.get("oracle") {
        Some(v) => Some(
            v.parse::<usize>()
                .map_err(|_| format!("oracle を数値として読めません: {v}"))?,
        ),
        None => None,
    };
    if let Some(o) = oracle {
        if o > ply {
            return Err(format!("oracle={o} が ply={ply} を超えています"));
        }
    }
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
        bad,
        scores,
        ply,
        fouls,
        oracle,
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
    /// 決定の前に注入された反則列（`fouls=` ディレクティブ、apply_scenario_fouls が
    /// 設定）。choice_trial_body が foul_tried の初期値に使い、バッチの
    /// チェーン共有はこれが異なるシナリオを別グループにする（ログが
    /// プレフィックス拡張でなくなるため）
    pub injected_fouls: Vec<String>,
    /// オラクル錨（`apply_scenario_oracle` が設定）: `(手番側ログの接頭辞長,
    /// その時点の真実の局面)`。Some なら試行前の prewarm は接頭辞までを
    /// `Strategy::oracle_anchor` で一括に与え、残りだけを通常 prewarm する
    /// （`prewarm_for_trial`）。バッチのチェーン共有は使わない（粒子生成が
    /// 無いので共有する意味もない）
    pub oracle: Option<(usize, Position)>,
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
        injected_fouls: vec![],
        oracle: None,
    }
}

/// `oracle=` ディレクティブ（`Scenario::oracle`）を Replayed に反映する:
/// 同じ棋譜を `oracle_ply` 手まで再生した真実の局面と、手番側の観測ログの
/// その時点の長さを錨として持つ。`replay` の後に呼ぶ（`apply_scenario_fouls`
/// とは順不同: 注入反則はログ末尾に付くので接頭辞関係は崩れない）。
/// `oracle_ply == ply` は「直前の相手手まで全部知っている」全知条件
pub fn apply_scenario_oracle(rep: &mut Replayed, kifu: &Kifu, oracle_ply: usize) {
    assert!(
        oracle_ply as u32 <= rep.plies,
        "oracle={oracle_ply} が再生手数 {} を超えています",
        rep.plies
    );
    let side = rep.pos.turn();
    let anchor = replay(kifu, oracle_ply);
    let prefix_len = anchor.logs[side_idx(side)].events().len();
    // 接頭辞関係の検証（replay は決定的なので必ず成り立つはずだが、
    // 注入反則や将来の改修で崩れたら即座に気づけるように）
    let full = rep.logs[side_idx(side)].events();
    assert!(
        prefix_len <= full.len()
            && anchor.logs[side_idx(side)]
                .events()
                .iter()
                .zip(full)
                .all(|(a, b)| format!("{a:?}") == format!("{b:?}")),
        "oracle の接頭辞ログが手番側ログの接頭辞になっていません"
    );
    rep.oracle = Some((prefix_len, anchor.pos));
}

/// 試行前の戦略の温め: 通常は `prewarm_strategy`、オラクル錨つきなら
/// 接頭辞までを `Strategy::oracle_anchor` で一括に与えて残りだけを
/// 逐次 prewarm する。単発試行・ランキング・バッチが共有する
pub fn prewarm_for_trial(strat: &mut dyn strategy::Strategy, rep: &Replayed) {
    let side = rep.pos.turn();
    let log = clone_log(&rep.logs[side_idx(side)]);
    let view = make_view(&rep.pos, side, &rep.fouls);
    match &rep.oracle {
        None => strategy::prewarm_strategy(strat, &view, &log),
        Some((prefix_len, truth)) => {
            let mut prefix = ObservationLog::default();
            for e in &log.events()[..*prefix_len] {
                prefix.record(e.clone());
            }
            assert!(
                strat.oracle_anchor(&view, &prefix, truth),
                "戦略 {} はオラクル錨（oracle=）に対応していません（現行 estimator のみ）",
                strat.name()
            );
            strategy::prewarm_strategy_from(strat, &view, &log, *prefix_len);
        }
    }
}

/// `fouls=` ディレクティブの反則列を注入した状態にする（実戦で反則を挟んだ
/// 決定点の再現）。`replay` の後に呼ぶ。規約は choice_trial_body と同じ:
/// 手番側の MyFoul 観測・反則カウント・相手側への OpponentFoul 通知。
/// 反則のはずの手が合法なら panic（棋譜由来の真実データの検証を兼ねる）
pub fn apply_scenario_fouls(rep: &mut Replayed, fouls: &[String]) {
    let side = rep.pos.turn();
    for usi in fouls {
        let mv = parse_usi(usi).unwrap_or_else(|| panic!("fouls のUSIを読めません: {usi}"));
        assert!(
            !rep.pos.is_legal(&mv),
            "fouls に指定した手が合法です（反則になりません）: {usi}"
        );
        rep.fouls[side_idx(side)] += 1;
        rep.logs[side_idx(side)].record(Observation::MyFoul {
            move_number: rep.pos.move_number(),
            usi: usi.clone(),
        });
        rep.logs[side_idx(side.other())].record(Observation::OpponentFoul {
            count: rep.fouls[side_idx(side)],
        });
        rep.injected_fouls.push(usi.clone());
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
    /// (**最初に試みた手**, 回数) を回数降順で。反則プローブ（quest_0809 36手目の
    /// 2三玉 = 7点、32手目の 3一角 = 8点のような「反則で情報を買う手」）は
    /// 受理手ではなく最初の試みに意図があり、eval も反則前ブロックに
    /// 「実戦手（反則）」として採点している。受理手の採点（`tally`）だけだと
    /// 「2三玉（反則）→ 1三香で回収」が反則前ブロックの 1三香 = 3 点で
    /// 数えられてしまう。反則が無ければ `tally` と同じ。空なら旧データ
    /// （TRIAL 行に初手列が無い）= `tally` で代用する
    pub first_tally: Vec<(String, u32)>,
}

impl ChoiceStats {
    pub fn target_hits(&self, target: &str) -> u32 {
        self.tally
            .iter()
            .find(|(usi, _)| usi == target)
            .map(|(_, n)| *n)
            .unwrap_or(0)
    }

    /// bad リストに載った手を選んだ回数の合計（suite の「不合格計」。
    /// bin/scenario の表示と bin/tune のシナリオ目的関数が同じ数え方を共有する）
    pub fn bad_hits(&self, bad: &[String]) -> u32 {
        self.tally
            .iter()
            .filter(|(usi, _)| bad.iter().any(|b| b == usi))
            .map(|(_, n)| *n)
            .sum()
    }

    /// 採点表による平均得点（0〜10）と未採点の選択回数。
    /// 採点表に無い手（未採点）は `UNSCORED_DEFAULT` で数えて件数を別途返す
    /// （方策が評価済み候補の外へ出ると計測不能になる抜け穴 2026-08-06 への
    /// 対応: 未採点の選択は数として見える化し、evals へ追記して判定を受ける）。
    /// "resign" / "foul_limit" は 0 点
    pub fn mean_score(&self, scores: &[(String, u8)]) -> (f64, u32) {
        Self::mean_of(&self.tally, scores)
    }

    /// **初手得点**: 最初に試みた手（反則プローブを含む）の採点表による平均得点。
    /// `first_tally` が空（旧データ）なら `tally` で代用
    pub fn mean_score_first(&self, scores: &[(String, u8)]) -> (f64, u32) {
        if self.first_tally.is_empty() {
            Self::mean_of(&self.tally, scores)
        } else {
            Self::mean_of(&self.first_tally, scores)
        }
    }

    /// 初手が受理手と異なる（= 反則を挟んだ）試行の回数
    pub fn probe_trials(&self) -> u32 {
        if self.first_tally.is_empty() {
            return 0;
        }
        let total: u32 = self.first_tally.iter().map(|(_, n)| n).sum();
        let same: u32 = self
            .first_tally
            .iter()
            .map(|(u, n)| {
                let acc = self.tally.iter().find(|(a, _)| a == u).map(|(_, m)| *m).unwrap_or(0);
                (*n).min(acc)
            })
            .sum();
        total.saturating_sub(same)
    }

    fn mean_of(tally: &[(String, u32)], scores: &[(String, u8)]) -> (f64, u32) {
        let mut sum = 0.0f64;
        let mut n = 0u32;
        let mut unscored = 0u32;
        for (usi, count) in tally {
            let pt = if usi == "resign" || usi == "foul_limit" {
                Some(0.0)
            } else {
                scores
                    .iter()
                    .find(|(s, _)| s == usi)
                    .map(|(_, p)| f64::from(*p))
            };
            let pt = pt.unwrap_or_else(|| {
                unscored += count;
                UNSCORED_DEFAULT
            });
            sum += pt * f64::from(*count);
            n += count;
        }
        if n == 0 {
            (0.0, 0)
        } else {
            (sum / f64::from(n), unscored)
        }
    }
}

/// 未採点の手を選んだときの仮の点数（0〜10）。promo 価格改定の実測では
/// 新出手の約半分が「あり」判定だったので中間よりやや下の 4 とする
pub const UNSCORED_DEFAULT: f64 = 4.0;

/// シードから戦略を作るファクトリ。`choice_trial_one_with` / `choice_trials_batch_with`
/// が EvalParams を注入した戦略（bin/tune のシナリオ目的関数）で試行するための口。
/// Sync はグループ並列（bin/tune がパーティションごとにスレッドを立てる）で
/// 共有するため
pub type SeededFactory<'a> = &'a (dyn Fn(u64) -> Box<dyn strategy::Strategy + Send> + Sync);

/// 手番側の一手の選択を1シードぶん試行する。反則は観測として与えて指し直させる
/// （実対局と同じ）。返り値は (受理された手, その前に試みた反則列)。
/// 指せる手がなければ "resign"、反則累計10回で "foul_limit"
pub fn choice_trial_one(rep: &Replayed, seed: u64, name: &str) -> (String, Vec<String>) {
    choice_trial_one_with(rep, seed, &|s| {
        strategy::make_seeded(name, s).expect("未知の戦略名")
    })
}

/// choice_trial_one の戦略ファクトリ版（name 版はここへ委譲する）
pub fn choice_trial_one_with(
    rep: &Replayed,
    seed: u64,
    factory: SeededFactory,
) -> (String, Vec<String>) {
    let mut strat = factory(seed);
    prewarm_for_trial(&mut *strat, rep);
    choice_trial_body(&mut *strat, rep)
}

/// prewarm 済みの戦略で choice_trial_one の選択ループだけを実行する
/// （バッチ実行がスナップショットに対して呼ぶ共通部品）
fn choice_trial_body(strat: &mut dyn strategy::Strategy, rep: &Replayed) -> (String, Vec<String>) {
    let side = rep.pos.turn();
    let mut log = clone_log(&rep.logs[side_idx(side)]);
    // 注入済みの反則（fouls= ディレクティブ）は「もう試した手」なので
    // 実対局と同じく再提示しない
    let mut foul_tried: HashSet<String> = rep.injected_fouls.iter().cloned().collect();
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
    ranking_one_with(rep, seed, name, &HashSet::new())
}

/// [`ranking_one`] の「**評価が見た粒子ごと**」返す版（issue #28 P0-3 / P0-6）。
///
/// `build_estimator` で同じ seed からもう一度作り直した粒子は**同じ集合には
/// ならない**（`Estimator::update` は壁時計デッドラインまで若返らせるので、
/// 実行のたびに粒子数が揺れる）。危険質量 q を「ランキングを作ったのと同じ
/// 粒子」で測るには、`choose` が評価へ渡したプールをそのまま持ち出す。
pub fn ranking_and_particles(
    rep: &Replayed,
    seed: u64,
    name: &str,
    foul_tried: &HashSet<String>,
) -> Option<(String, Vec<strategy::CandidateScore>, strategy::ParticleSnapshot)> {
    let side = rep.pos.turn();
    let mut strat = strategy::make_seeded(name, seed).expect("未知の戦略名");
    strat.set_capture_particles(true);
    let log = clone_log(&rep.logs[side_idx(side)]);
    prewarm_for_trial(&mut *strat, rep);
    let view = make_view(&rep.pos, side, &rep.fouls);
    let chosen = strat.choose(&view, &log, foul_tried)?;
    let ranking = strat.last_ranking()?.to_vec();
    // 粒子スナップショットは現行 estimator だけが実装する（凍結版は None）
    let particles = strat.last_particles().cloned().unwrap_or_default();
    Some((chosen, ranking, particles))
}

/// [`ranking_one`] の `foul_tried` 指定版。
///
/// 実戦の決定点を復元する診断（`bin/mate_probe`）は、その手番で既に試した
/// 反則を渡さないと候補集合が実戦と変わる（反則した手が候補に残る）。
pub fn ranking_one_with(
    rep: &Replayed,
    seed: u64,
    name: &str,
    foul_tried: &HashSet<String>,
) -> Option<(String, Vec<strategy::CandidateScore>)> {
    let side = rep.pos.turn();
    let mut strat = strategy::make_seeded(name, seed).expect("未知の戦略名");
    let log = clone_log(&rep.logs[side_idx(side)]);
    prewarm_for_trial(&mut *strat, rep);
    let view = make_view(&rep.pos, side, &rep.fouls);
    let chosen = strat.choose(&view, &log, foul_tried)?;
    let ranking = strat.last_ranking()?.to_vec();
    Some((chosen, ranking))
}

/// 推定器を実対局と同じ**逐次 update** で構築する（`prewarm_strategy` と同じ理由:
/// 一括 update だと制約列の解き方が変わって粒子集合が実対局とずれる）。
/// `on_turn(&est, 手番番号)` が自分の手番ごとに呼ばれる（診断の進捗表示用）。
pub fn build_estimator(
    rep: &Replayed,
    seed: u64,
    scale: f64,
    mut on_turn: impl FnMut(&Estimator, usize),
) -> Estimator {
    let side = rep.pos.turn();
    let log = &rep.logs[side_idx(side)];
    let mut est = Estimator::with_seed_and_scale(side, seed, scale);
    let mut running = ObservationLog::default();
    let mut turn_no = 0;
    for e in log.events() {
        if matches!(e, Observation::MyMove { .. } | Observation::MyFoul { .. }) {
            est.update(&running);
            turn_no += 1;
            on_turn(&est, turn_no);
        }
        running.record(e.clone());
    }
    est.update(&running);
    est
}

/// 途中まで観測を食わせた推定器を保持し、**続きだけ**を食わせて進められる形。
///
/// `Estimator::update` は消化済みイベント数（cursor）を自分で覚えているので、
/// 同じ手順で観測を足していけば「ゼロから構築し直した場合と同じ状態」になる。
/// GUI で ply を進めながら信念を見るときに、毎回 1手目から粒子を作り直さずに済む
/// （`.kif` を再生した観測ログは ply が進んでも**前のログのプレフィックス拡張**に
/// なるので、続きから食わせるだけでよい）。
pub struct IncrementalEstimator {
    est: Estimator,
    running: ObservationLog,
    consumed: usize,
}

impl IncrementalEstimator {
    pub fn new(side: Color, seed: u64, scale: f64) -> Self {
        IncrementalEstimator {
            est: Estimator::with_seed_and_scale(side, seed, scale),
            running: ObservationLog::default(),
            consumed: 0,
        }
    }

    /// `log`（これまで食わせたログのプレフィックス拡張）の続きを食わせる。
    /// build_estimator と同じ順序（自分の手番の直前で update、最後にもう一度）
    pub fn feed(&mut self, log: &ObservationLog) {
        let events = log.events();
        while self.consumed < events.len() {
            let e = &events[self.consumed];
            if matches!(e, Observation::MyMove { .. } | Observation::MyFoul { .. }) {
                self.est.update(&self.running);
            }
            self.running.record(e.clone());
            self.consumed += 1;
        }
        self.est.update(&self.running);
    }

    pub fn est(&self) -> &Estimator {
        &self.est
    }

    /// 食わせ済みのイベント数（キャッシュが使えるかの判定用）
    pub fn consumed(&self) -> usize {
        self.consumed
    }
}

/// 推定器のユニーク粒子と**評価重み**。重みの規約は評価側 `stratified_sample`
/// と同じ: 推定器内の logw を max で正規化し、同一指紋の質量 Σexp(logw) を
/// 畳み込む（C-7 P1 の multiplicity 保持。ソフト減衰は logw に課金済み）。
/// `strict` は「情報制約も物理制約も緩和していない」粒子（phys_taint>0 は
/// info_miss を最低1として非厳密扱いにする、diag と同じ判定）。
///
/// 診断（bin/scenario の diag）と GUI の玉位置ビリーフが**同じ規約**で数える
/// ための共通部品。ここが食い違うと較正の数字が意味を失う
pub fn weighted_unique_particles(est: &Estimator) -> Vec<(&Position, f64, bool)> {
    let max_logw = est.log_weights().iter().copied().fold(f64::MIN, f64::max);
    let mut mass: HashMap<u64, (f64, u8)> = HashMap::new();
    for (((pp, &miss), &taint), &lw) in est
        .particles()
        .iter()
        .zip(est.info_miss())
        .zip(est.phys_taint())
        .zip(est.log_weights())
    {
        let miss_eff = if taint > 0 { miss.max(1) } else { miss };
        let e = mass.entry(pp.fingerprint()).or_insert((0.0, miss_eff));
        e.0 += (lw - max_logw).exp();
        e.1 = e.1.min(miss_eff);
    }
    let mut seen: HashSet<u64> = HashSet::new();
    let mut out = vec![];
    for pp in est.particles() {
        let fp = pp.fingerprint();
        if !seen.insert(fp) {
            continue;
        }
        let (w, penalty) = mass[&fp];
        out.push((pp, w, penalty == 0));
    }
    out
}

/// 相手玉の位置ごとの信念（重み付き割合）。
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KingBeliefSquare {
    /// USI マス（"7i"）
    pub sq: String,
    /// taint 込みの全粒子での割合 [0,1]
    pub all: f64,
    /// 厳密整合の粒子だけでの割合 [0,1]（厳密が全滅していれば 0）
    pub strict: f64,
}

/// 手番側から見た相手玉の位置ビリーフ（`bin/scenario diag` の「相手玉の位置分布」と
/// 同じ計算を、GUI から呼べる形で返す）。
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KingBelief {
    /// 信念を持っている側（= その局面の手番側）
    pub side: Color,
    /// 割合の降順
    pub squares: Vec<KingBeliefSquare>,
    /// 真実の相手玉のマス（GUI の答え合わせ表示用）
    pub truth: Option<String>,
    /// ユニーク粒子数と、そのうち厳密整合の数。**厳密が0なら評価は taint 頼り**で、
    /// strict 側の列は全ゼロになる（mate-net 57手目のような終盤で普通に起きる）
    pub unique: u32,
    pub strict_unique: u32,
    /// 集計に使った推定器（シード）の数
    pub seeds: u64,
    /// 王手宣言の履歴から**健全に**絞れる玉位置候補（deduce::opp_king_candidates）。
    /// ここに無いマスの信念は論理的にあり得ない = フィルタの綻び
    pub deduced: Vec<String>,
}

/// 推定器を seeds 個ぶん構築して相手玉の位置ビリーフを集計する
pub fn king_belief(rep: &Replayed, seeds: u64, scale: f64) -> KingBelief {
    let ests: Vec<Estimator> = (0..seeds.max(1))
        .map(|seed| build_estimator(rep, seed, scale, |_, _| {}))
        .collect();
    king_belief_from(rep, &ests, seeds.max(1))
}

/// 構築済みの推定器から相手玉の位置ビリーフを集計する
/// （GUI は推定器をキャッシュして ply を進めるので、構築と集計を分けておく）
pub fn king_belief_from(rep: &Replayed, ests: &[Estimator], seeds: u64) -> KingBelief {
    let refs: Vec<&Estimator> = ests.iter().collect();
    king_belief_from_refs(rep, &refs, seeds)
}

/// `king_belief_from` の参照版（キャッシュから借りたまま集計するため）
pub fn king_belief_from_refs(rep: &Replayed, ests: &[&Estimator], seeds: u64) -> KingBelief {
    let side = rep.pos.turn();
    let mut all_tally: HashMap<String, f64> = HashMap::new();
    let mut strict_tally: HashMap<String, f64> = HashMap::new();
    let mut all_mass = 0.0f64;
    let mut strict_mass = 0.0f64;
    let mut unique = 0u32;
    let mut strict_unique = 0u32;
    for est in ests {
        for (pp, w, strict) in weighted_unique_particles(est) {
            unique += 1;
            if strict {
                strict_unique += 1;
            }
            let Some(sq) = pp.king_square(side.other()) else {
                continue;
            };
            let key = make_usi_square(sq);
            *all_tally.entry(key.clone()).or_insert(0.0) += w;
            all_mass += w;
            if strict {
                *strict_tally.entry(key).or_insert(0.0) += w;
                strict_mass += w;
            }
        }
    }
    let mut squares: Vec<KingBeliefSquare> = all_tally
        .into_iter()
        .map(|(sq, m)| {
            let strict = strict_tally.get(&sq).copied().unwrap_or(0.0);
            KingBeliefSquare {
                all: m / all_mass.max(1e-12),
                strict: if strict_mass > 0.0 { strict / strict_mass } else { 0.0 },
                sq,
            }
        })
        .collect();
    squares.sort_by(|a, b| {
        b.all
            .partial_cmp(&a.all)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.sq.cmp(&b.sq))
    });
    let deduced = crate::deduce::opp_king_candidates(side, &rep.logs[side_idx(side)])
        .into_iter()
        .map(make_usi_square)
        .collect();
    KingBelief {
        side,
        squares,
        truth: rep.pos.king_square(side.other()).map(make_usi_square),
        unique,
        strict_unique,
        seeds,
        deduced,
    }
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
    let mut first_map: HashMap<String, u32> = HashMap::new();
    let mut total_fouls = 0u32;
    for seed in 0..trials {
        let (accepted, foul_seq) = choice_trial_one(rep, seed, name);
        on_trial(seed, &accepted, &foul_seq);
        let first = foul_seq.first().cloned().unwrap_or_else(|| accepted.clone());
        *first_map.entry(first).or_insert(0) += 1;
        *final_tally.entry(accepted).or_insert(0) += 1;
        total_fouls += foul_seq.len() as u32;
    }
    let mut tally: Vec<_> = final_tally.into_iter().collect();
    tally.sort_by(|a, b| b.1.cmp(&a.1));
    let mut first_tally: Vec<_> = first_map.into_iter().collect();
    first_tally.sort_by(|a, b| b.1.cmp(&a.1));
    ChoiceStats {
        tally,
        total_fouls,
        first_tally,
    }
}

/// 棋譜の同一性キー（バッチ実行のグループ化用）。指し手列と反則試行列が
/// 完全一致する棋譜だけが同じキーになる（＝観測ログがプレフィックス拡張の
/// 関係になることの保証）
pub fn kifu_key(kifu: &Kifu) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for p in &kifu.plies {
        p.mv.to_usi().hash(&mut h);
        format!("{:?}", p.fouls).hash(&mut h);
    }
    h.finish()
}

/// 複数シナリオの選択試行をまとめて実行する。**同一棋譜×同一手番側**の
/// シナリオはシードごとに prewarm 済み戦略を ply 昇順で継ぎ足して共有し、
/// 各決定点では `clone_boxed` のスナップショットに試行させる
/// （scenario-gui の IncrementalEstimator と同じ原理の Strategy 版。
/// 推定器の構築が ply に比例して重いので、同一棋譜から切り出した ply 違いの
/// シナリオ群では最深 ply 1本ぶんのコストに近づく）。
/// clone_boxed 非対応の戦略（凍結版）は従来どおり毎回作り直す。
/// 返り値は items と同順の ChoiceStats。on_trial(シナリオindex, seed, 受理手, 反則列)
pub fn choice_trials_batch(
    items: &[(&Scenario, &Replayed)],
    trials: u64,
    name: &str,
    on_trial: impl FnMut(usize, u64, &str, &[String]),
) -> Vec<ChoiceStats> {
    choice_trials_batch_with(
        items,
        trials,
        &|s| strategy::make_seeded(name, s).expect("未知の戦略名"),
        on_trial,
    )
}

/// choice_trials_batch の戦略ファクトリ版（name 版はここへ委譲する）
pub fn choice_trials_batch_with(
    items: &[(&Scenario, &Replayed)],
    trials: u64,
    factory: SeededFactory,
    on_trial: impl FnMut(usize, u64, &str, &[String]),
) -> Vec<ChoiceStats> {
    choice_trials_batch_range_with(items, 0..trials, factory, on_trial)
}

/// choice_trials_batch のシード範囲版。CI のシードシャード実行
/// （scenario.yml が seed ごとに1ランナーへ分割し、各ランナーは
/// `seed_base..seed_base+1` で全シナリオを1試行ずつ回す）用の口。
/// `0..trials` を渡せば従来の choice_trials_batch と同一
pub fn choice_trials_batch_range_with(
    items: &[(&Scenario, &Replayed)],
    seeds: std::ops::Range<u64>,
    factory: SeededFactory,
    mut on_trial: impl FnMut(usize, u64, &str, &[String]),
) -> Vec<ChoiceStats> {
    let mut tallies: Vec<HashMap<String, u32>> = vec![HashMap::new(); items.len()];
    let mut first_tallies: Vec<HashMap<String, u32>> = vec![HashMap::new(); items.len()];
    let mut fouls_total: Vec<u32> = vec![0; items.len()];

    // clone 対応か（戦略の構築だけなら安い。est は遅延構築なので空のまま）
    let supports_clone = factory(0).clone_boxed().is_some();

    // (棋譜キー, 手番側) でグループ化し、グループ内は ply 昇順。
    // 反則注入シナリオ（fouls=）も同じグループで共有する: 注入した MyFoul は
    // ログ末尾の既知個数（injected_fouls.len()）なので、共有チェーンは
    // その手前（= 素のリプレイのプレフィックス）まで進め、注入の尾は
    // スナップショット側だけに食わせる（下記）
    let mut groups: Vec<((u64, Color), Vec<usize>)> = vec![];
    for (i, (sc, rep)) in items.iter().enumerate() {
        let key = (kifu_key(&sc.kifu), rep.pos.turn());
        match groups.iter_mut().find(|(k, _)| *k == key) {
            Some((_, v)) => v.push(i),
            None => groups.push((key, vec![i])),
        }
    }
    for (_, idxs) in &mut groups {
        idxs.sort_by_key(|&i| items[i].0.ply);
    }

    for ((_, _side), idxs) in &groups {
        for seed in seeds.clone() {
            let mut record = |i: usize, accepted: String, foul_seq: Vec<String>| {
                on_trial(i, seed, &accepted, &foul_seq);
                let first = foul_seq.first().cloned().unwrap_or_else(|| accepted.clone());
                *first_tallies[i].entry(first).or_insert(0) += 1;
                *tallies[i].entry(accepted).or_insert(0) += 1;
                fouls_total[i] += foul_seq.len() as u32;
            };
            if !supports_clone || idxs.len() == 1 {
                for &i in idxs {
                    let (accepted, foul_seq) = choice_trial_one_with(items[i].1, seed, factory);
                    record(i, accepted, foul_seq);
                }
                continue;
            }
            // 継ぎ足し共有: prewarm_strategy と同じ規約（自分手番イベントの直前で
            // prewarm、最後の update は choose に任せる）を consumed から再開する
            let mut strat = factory(seed);
            let mut running = ObservationLog::default();
            let mut consumed = 0usize;
            for &i in idxs {
                let (_sc, rep) = items[i];
                if rep.oracle.is_some() {
                    // オラクル錨つきは粒子生成が無いので共有せず毎回作る
                    // （チェーンはこの項目を飛ばすだけ。ply 昇順は保たれる）
                    let (accepted, foul_seq) = choice_trial_one_with(rep, seed, factory);
                    record(i, accepted, foul_seq);
                    continue;
                }
                let side = rep.pos.turn();
                let view = make_view(&rep.pos, side, &rep.fouls);
                let log = &rep.logs[side_idx(side)];
                let events = log.events();
                // 注入反則（fouls=）はログ末尾に付くので、共有チェーンは
                // その手前 = 素のリプレイと同じプレフィックスまでしか進めない
                // （進めてしまうと後続シナリオのログが拡張にならない）
                let base_len = events.len() - rep.injected_fouls.len();
                while consumed < base_len {
                    let e = &events[consumed];
                    if matches!(e, Observation::MyMove { .. } | Observation::MyFoul { .. }) {
                        strat.prewarm(&view, &running);
                    }
                    running.record(e.clone());
                    consumed += 1;
                }
                let mut snap = strat
                    .clone_boxed()
                    .expect("clone_boxed 対応確認済みの戦略で clone に失敗");
                if base_len < events.len() {
                    // 注入反則の尾はスナップショット側だけに食わせる
                    // （prewarm_strategy と同じ規約で、共有チェーンは汚さない）
                    let mut tail_running = clone_log(&running);
                    for e in &events[base_len..] {
                        if matches!(e, Observation::MyMove { .. } | Observation::MyFoul { .. }) {
                            snap.prewarm(&view, &tail_running);
                        }
                        tail_running.record(e.clone());
                    }
                }
                let (accepted, foul_seq) = choice_trial_body(&mut *snap, rep);
                record(i, accepted, foul_seq);
            }
        }
    }

    tallies
        .into_iter()
        .zip(first_tallies)
        .zip(fouls_total)
        .map(|((t, ft), f)| {
            let mut tally: Vec<_> = t.into_iter().collect();
            tally.sort_by(|a, b| b.1.cmp(&a.1));
            let mut first_tally: Vec<_> = ft.into_iter().collect();
            first_tally.sort_by(|a, b| b.1.cmp(&a.1));
            ChoiceStats {
                tally,
                total_fouls: f,
                first_tally,
            }
        })
        .collect()
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

    /// `fouls=` の反則注入が実対局と同じ状態を作ること（quest31-m030f2 =
    /// 実戦で 1二飛・2一歩打 を反則してから 4二金 を指した決定点）
    #[test]
    fn fouls指定は反則観測と反則カウントを注入する() {
        let sc = load("quest31-m030f2");
        assert_eq!(sc.fouls, vec!["5b1b".to_string(), "P*2a".to_string()]);
        let mut rep = replay(&sc.kifu, sc.ply);
        let side = rep.pos.turn();
        let before = rep.logs[side_idx(side)].events().len();
        apply_scenario_fouls(&mut rep, &sc.fouls);
        // 手番側は MyFoul、相手側は OpponentFoul を反則の数だけ受け取る
        assert_eq!(rep.fouls[side_idx(side)], 2);
        assert_eq!(rep.fouls[side_idx(side.other())], 0);
        assert_eq!(rep.logs[side_idx(side)].events().len(), before + 2);
        assert!(matches!(
            rep.logs[side_idx(side)].events().last(),
            Some(Observation::MyFoul { usi, .. }) if usi == "P*2a"
        ));
        // 局面は進まない（反則は手番を失わない）
        assert_eq!(rep.pos.turn(), side);
        assert_eq!(rep.injected_fouls, sc.fouls);
    }

    /// `scores=` / `bad=` に「手番側が指せない手」が混ざっていないこと。
    ///
    /// eval の採点をブロックごと取り違えると（例: 26手目=後手番のブロックへ
    /// 27手目=先手番の手を書く）、相手の駒を動かす USI が採点表に入る。
    /// 到達しえない手なので平均得点には出てこず、`bad=` に入っても不合格計が
    /// 増えないため、**suite の数字を見ているかぎり永久に気づけない**。
    /// 2026-08-11 に PR #3 で実際に3件混入したので常設の関門にする。
    ///
    /// 合法性そのものは見ない（見えない相手駒に依存するし、`3三銀不成` のように
    /// `TSUITATE_GEN_NONPROMOTE=1` でしか生成されない手も採点対象にしてよい）。
    /// 「移動元に手番側の駒があるか」「打つ駒を持っているか」だけを見る。
    #[test]
    fn 採点表の手はすべて手番側の駒の手である() {
        let mut violations: Vec<String> = Vec::new();
        let mut checked = 0usize;
        let mut dir: Vec<PathBuf> = std::fs::read_dir(scenarios_dir())
            .expect("scenarios/ を読めません")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "kif"))
            .collect();
        dir.sort();
        for path in dir {
            let sc = match load_scenario(&path.to_string_lossy(), None, None, None) {
                Ok(sc) => sc,
                // ply= を持たない棋譜（アーカイブ等）はシナリオではない
                Err(_) => continue,
            };
            if sc.scores.is_empty() && sc.bad.is_empty() {
                continue;
            }
            let rep = replay(&sc.kifu, sc.ply);
            let side = rep.pos.turn();
            // bad= は scores= の部分集合なので同じ手を二重に数えない
            let listed: HashSet<&String> =
                sc.scores.iter().map(|(u, _)| u).chain(sc.bad.iter()).collect();
            let mut listed: Vec<&String> = listed.into_iter().collect();
            listed.sort();
            for usi in listed {
                checked += 1;
                let Some(mv) = parse_usi(usi) else {
                    violations.push(format!("{}: {usi} を USI として解析できない", sc.name));
                    continue;
                };
                match mv {
                    ShogiMove::Board { from, .. } => match rep.pos.piece_at(from) {
                        None => violations.push(format!(
                            "{}（{side:?}番）: {usi} の移動元が空きマス",
                            sc.name
                        )),
                        Some(p) if p.color != side => violations.push(format!(
                            "{}（{side:?}番）: {usi} の移動元は相手の {:?}",
                            sc.name, p.role
                        )),
                        Some(_) => {}
                    },
                    ShogiMove::Drop { role, .. } => {
                        if rep.pos.hand_count(side, role) == 0 {
                            violations.push(format!(
                                "{}（{side:?}番）: {usi} の {role:?} を持っていない",
                                sc.name
                            ));
                        }
                    }
                }
            }
        }
        assert!(checked > 1000, "採点表がほとんど読めていない（{checked} 件）");
        assert!(
            violations.is_empty(),
            "手番側の手でない採点が {} 件:\n{}",
            violations.len(),
            violations.join("\n")
        );
    }

    /// 反則にならない手を fouls= に書いたら気づけること（棋譜の写し間違い検出）
    #[test]
    #[should_panic(expected = "合法です")]
    fn fouls指定が合法手ならpanicする() {
        let sc = load("quest31-m030f2");
        let mut rep = replay(&sc.kifu, sc.ply);
        apply_scenario_fouls(&mut rep, &["4f4g+".to_string()]);
    }

    /// 手動翻訳で検証済みだった USI 列とパーサーの出力が一致すること
    #[test]
    fn keimaの棋譜はusi列と反則が既知の正解に一致する() {
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
    fn kakunariの棋譜はusi列と反則が既知の正解に一致する() {
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

    /// 玉位置ビリーフ（scenario-gui の盤オーバーレイ）の形の検証。
    /// 序盤（相手が玉をまだ動かしていない ply）なら初期マスに信念が集中するはず。
    /// 推定器の構築が重いので release で実行する:
    /// `cargo test --release -- --ignored ビリーフ`
    #[test]
    #[ignore]
    fn 玉位置ビリーフは序盤なら初期マスに集中する() {
        let sc = load("keima");
        // 4手目まで = 後手は 3a3b / 2b3a しか指しておらず玉は 5a のまま
        let rep = replay(&sc.kifu, 4);
        assert_eq!(rep.pos.turn(), Color::Sente);
        let b = king_belief(&rep, 2, 1.0);
        assert_eq!(b.side, Color::Sente);
        assert_eq!(b.truth.as_deref(), Some("5a"));
        // 割合は正規化されている（全粒子側）
        let sum: f64 = b.squares.iter().map(|s| s.all).sum();
        assert!((sum - 1.0).abs() < 1e-6, "全粒子の割合の和が1でない: {sum}");
        // 降順で返る
        for w in b.squares.windows(2) {
            assert!(w[0].all >= w[1].all, "割合の降順でない");
        }
        // 序盤は厳密粒子が生きていて、真実の 5a が最有力
        assert!(b.strict_unique > 0, "序盤なのに厳密粒子が全滅している");
        assert_eq!(b.squares[0].sq, "5a", "初期マスが最有力でない");
    }

    /// 粒子を引き継いで進めた推定器が、ゼロから作り直したものと同じになること
    /// （GUI の玉位置ビリーフのキャッシュが結果を変えないことの担保）。
    /// 粒子が潤沢な序盤で測る（終盤はリプレイの時間打ち切りが壁時計依存で揺れる）
    #[test]
    fn 推定器は途中から引き継いでも作り直しと一致する() {
        let sc = load("keima");
        let side = replay(&sc.kifu, 4).pos.turn();
        let mut inc = IncrementalEstimator::new(side, 0, 1.0);
        // 2手目まで食わせてから 4手目まで継ぎ足す
        inc.feed(&replay(&sc.kifu, 2).logs[side_idx(side)]);
        let mid = inc.consumed();
        let rep4 = replay(&sc.kifu, 4);
        inc.feed(&rep4.logs[side_idx(side)]);
        assert!(inc.consumed() > mid, "継ぎ足しでイベントが進んでいない");

        let fresh = build_estimator(&rep4, 0, 1.0, |_, _| {});
        let key = |est: &Estimator| -> Vec<(u64, u64)> {
            let mut v: Vec<(u64, u64)> = weighted_unique_particles(est)
                .iter()
                .map(|(p, w, _)| (p.fingerprint(), (w * 1e9) as u64))
                .collect();
            v.sort_unstable();
            v
        };
        assert_eq!(key(inc.est()), key(&fresh), "引き継ぎと作り直しで粒子集合が違う");
    }

    /// オラクル錨（oracle=）の担保: 錨を張った直後の粒子は全て真実の局面、
    /// 以後の相手手を観測だけで処理した後の粒子は「真実の後継局面かつ
    /// 観測と整合」で、真実の粒子（= 実際に指された手）がその中に残っている。
    /// 反則注入（fouls=）と併用しても接頭辞関係が崩れないことも見る
    #[test]
    fn オラクル錨は真実から出発して観測だけで分岐する() {
        // kakutori: 48手目の角の王手を49手目（先手）が受ける。錨 = 47手目まで
        let sc = load("kakutori");
        assert_eq!(sc.ply, 48);
        let mut rep = replay(&sc.kifu, sc.ply);
        apply_scenario_oracle(&mut rep, &sc.kifu, 47);
        let side = rep.pos.turn();
        let (prefix_len, truth47) = rep.oracle.clone().expect("錨が設定されていない");
        assert_eq!(truth47.turn(), side.other(), "47手目までなら相手番のはず");
        assert!(prefix_len < rep.logs[side_idx(side)].events().len());

        // 錨直後: 全粒子が真実
        let mut est = Estimator::with_seed_and_scale(side, 7, 1.0);
        let mut prefix = ObservationLog::default();
        for e in &rep.logs[side_idx(side)].events()[..prefix_len] {
            prefix.record(e.clone());
        }
        est.oracle_anchor(&prefix, &truth47);
        assert!(est.is_oracle_anchored());
        assert!(!est.particles().is_empty());
        assert!(
            est.particles().iter().all(|p| p.fingerprint() == truth47.fingerprint()),
            "錨直後の粒子が真実と一致しない"
        );

        // 48手目（相手の王手）を観測だけで処理: 粒子は真実の合法後継で、
        // 自玉に王手が掛かっていて、真実（実際の 1五角）も残っている
        est.update(&rep.logs[side_idx(side)]);
        let parts = est.particles();
        assert!(!parts.is_empty(), "観測処理後に粒子が消えた");
        let truth48 = &rep.pos;
        let mut has_truth = false;
        for p in parts {
            assert_eq!(p.turn(), side);
            assert!(p.in_check(side), "王手宣言と整合しない粒子");
            assert_eq!(
                p.pieces_of(side).len(),
                truth48.pieces_of(side).len(),
                "自駒が増減している（真実の後継でない）"
            );
            has_truth |= p.fingerprint() == truth48.fingerprint();
        }
        assert!(has_truth, "真実の局面が粒子集合から落ちている");

        // Strategy 経由（prewarm_for_trial）でも同じ錨が張られ、choose が動く
        let mut strat = strategy::EstimatorStrategy::with_params_line_seed(
            strategy::EvalParams::default(),
            None,
            Some(0),
        );
        prewarm_for_trial(&mut strat, &rep);
        assert!(strat.estimator().is_some_and(|e| e.is_oracle_anchored()));
        let (accepted, _) = choice_trial_body(&mut strat, &rep);
        assert!(parse_usi(&accepted).is_some(), "choose が手を返さない: {accepted}");

        // fouls= 併用: 注入反則はログ末尾なので接頭辞関係は崩れない（assert が通る）
        let sc2 = load("quest31-m030f2");
        let mut rep2 = replay(&sc2.kifu, sc2.ply);
        apply_scenario_fouls(&mut rep2, &sc2.fouls);
        apply_scenario_oracle(&mut rep2, &sc2.kifu, sc2.ply - 1);
        assert!(rep2.oracle.is_some());
    }

    /// バッチ実行の Strategy 版の担保: 「ply A まで prewarm → 続きを ply B まで
    /// 継ぎ足し」た戦略の推定器が、ゼロから ply B まで prewarm した戦略と一致する。
    /// スナップショット（clone_boxed）が元の状態を変えないことも見る
    #[test]
    fn 戦略のprewarm継ぎ足しは作り直しと一致する() {
        use crate::observation::Observation;
        let sc = load("keima");
        let rep2 = replay(&sc.kifu, 2);
        let rep4 = replay(&sc.kifu, 4);
        let side = rep4.pos.turn();
        assert_eq!(rep2.pos.turn(), side, "ply2 と ply4 で手番側が違う");

        let key = |est: &Estimator| -> Vec<(u64, u64)> {
            let mut v: Vec<(u64, u64)> = weighted_unique_particles(est)
                .iter()
                .map(|(p, w, _)| (p.fingerprint(), (w * 1e9) as u64))
                .collect();
            v.sort_unstable();
            v
        };

        // 作り直し: ゼロから ply4 まで
        let mut fresh = strategy::EstimatorStrategy::with_params_line_seed(
            strategy::EvalParams::default(),
            None,
            Some(0),
        );
        strategy::prewarm_strategy(
            &mut fresh,
            &make_view(&rep4.pos, side, &rep4.fouls),
            &rep4.logs[side_idx(side)],
        );

        // 継ぎ足し: ply2 まで → 続きを ply4 まで（choice_trials_batch と同じ手順）
        let mut inc = strategy::EstimatorStrategy::with_params_line_seed(
            strategy::EvalParams::default(),
            None,
            Some(0),
        );
        strategy::prewarm_strategy(
            &mut inc,
            &make_view(&rep2.pos, side, &rep2.fouls),
            &rep2.logs[side_idx(side)],
        );
        assert!(
            crate::strategy::Strategy::clone_boxed(&inc).is_some(),
            "EstimatorStrategy が clone_boxed 非対応"
        );
        let snap = inc.clone();
        let mut running = clone_log(&rep2.logs[side_idx(side)]);
        let mut consumed = running.events().len();
        let events4 = rep4.logs[side_idx(side)].events().to_vec();
        let view4 = make_view(&rep4.pos, side, &rep4.fouls);
        while consumed < events4.len() {
            let e = &events4[consumed];
            if matches!(e, Observation::MyMove { .. } | Observation::MyFoul { .. }) {
                crate::strategy::Strategy::prewarm(&mut inc, &view4, &running);
            }
            running.record(e.clone());
            consumed += 1;
        }
        // choose 相当の最終 update（prewarm_strategy はループ内でしか update
        // しないので、比較のため両者へ同じ最終消化を与える）
        crate::strategy::Strategy::prewarm(&mut fresh, &view4, &rep4.logs[side_idx(side)]);
        crate::strategy::Strategy::prewarm(&mut inc, &view4, &running);
        assert_eq!(
            key(fresh.estimator().unwrap()),
            key(inc.estimator().unwrap()),
            "継ぎ足しと作り直しで粒子集合が違う"
        );
        // スナップショットは ply2 時点のまま（元を ply4 へ進めても変わらない）
        let mut fresh_at2 = strategy::EstimatorStrategy::with_params_line_seed(
            strategy::EvalParams::default(),
            None,
            Some(0),
        );
        strategy::prewarm_strategy(
            &mut fresh_at2,
            &make_view(&rep2.pos, side, &rep2.fouls),
            &rep2.logs[side_idx(side)],
        );
        assert_eq!(
            key(snap.estimator().unwrap()),
            key(fresh_at2.estimator().unwrap()),
            "スナップショットが元の進行に引きずられている"
        );
    }

    /// 反則注入シナリオ（fouls=）のチェーン共有の担保: 「素のログまで prewarm →
    /// スナップショットに注入反則の尾だけ食わせる」（choice_trials_batch_range_with
    /// の方式）が、注入込みの全ログをゼロから prewarm した戦略と一致する
    #[test]
    fn 反則注入の尾をスナップショットへ食わせる継ぎ足しは作り直しと一致する() {
        use crate::observation::Observation;
        let sc = load("keima");
        let mut rep4f = replay(&sc.kifu, 4);
        let side = rep4f.pos.turn();
        apply_scenario_fouls(&mut rep4f, &["1g1e".to_string()]); // 2マス動く反則
        let rep4 = replay(&sc.kifu, 4); // 素のリプレイ（共有チェーン相当）

        let key = |est: &Estimator| -> Vec<(u64, u64)> {
            let mut v: Vec<(u64, u64)> = weighted_unique_particles(est)
                .iter()
                .map(|(p, w, _)| (p.fingerprint(), (w * 1e9) as u64))
                .collect();
            v.sort_unstable();
            v
        };

        // 作り直し: 注入込みの全ログをゼロから prewarm（fresh 経路と同じ）
        let mut fresh = strategy::EstimatorStrategy::with_params_line_seed(
            strategy::EvalParams::default(),
            None,
            Some(0),
        );
        let view_f = make_view(&rep4f.pos, side, &rep4f.fouls);
        strategy::prewarm_strategy(&mut fresh, &view_f, &rep4f.logs[side_idx(side)]);

        // 継ぎ足し: 素のログまで prewarm → clone → 注入の尾だけを食わせる
        let mut chain = strategy::EstimatorStrategy::with_params_line_seed(
            strategy::EvalParams::default(),
            None,
            Some(0),
        );
        strategy::prewarm_strategy(
            &mut chain,
            &view_f,
            &rep4.logs[side_idx(side)],
        );
        let mut snap = chain.clone();
        let mut tail_running = clone_log(&rep4.logs[side_idx(side)]);
        let events_f = rep4f.logs[side_idx(side)].events();
        let base_len = events_f.len() - rep4f.injected_fouls.len();
        assert!(base_len < events_f.len(), "注入反則がログに載っていない");
        for e in &events_f[base_len..] {
            if matches!(e, Observation::MyMove { .. } | Observation::MyFoul { .. }) {
                crate::strategy::Strategy::prewarm(&mut snap, &view_f, &tail_running);
            }
            tail_running.record(e.clone());
        }
        // choose 相当の最終 update を両者へ同じだけ与えてから比較
        crate::strategy::Strategy::prewarm(&mut fresh, &view_f, &rep4f.logs[side_idx(side)]);
        crate::strategy::Strategy::prewarm(&mut snap, &view_f, &tail_running);
        assert_eq!(
            key(fresh.estimator().unwrap()),
            key(snap.estimator().unwrap()),
            "反則注入の尾の継ぎ足しと作り直しで粒子集合が違う"
        );
    }

    #[test]
    fn 不合格計はbadリストの手の回数だけ数える() {
        let stats = ChoiceStats {
            first_tally: vec![],
            tally: vec![
                ("7f7e".into(), 12),
                ("P*9h".into(), 5),
                ("2h5h".into(), 3),
            ],
            total_fouls: 4,
        };
        assert_eq!(stats.bad_hits(&["7f7e".into(), "P*9h".into()]), 17);
        assert_eq!(stats.bad_hits(&["2h5h".into()]), 3);
        assert_eq!(stats.bad_hits(&["9i9h".into()]), 0);
        assert_eq!(stats.bad_hits(&[]), 0);
    }

    /// 決められた手を順に返すだけの戦略（ファクトリ版APIの配管テスト用）
    struct Scripted {
        moves: Vec<String>,
        i: usize,
    }

    impl strategy::Strategy for Scripted {
        fn choose(
            &mut self,
            _view: &PlayerView,
            _log: &ObservationLog,
            foul_tried: &HashSet<String>,
        ) -> Option<String> {
            while self.i < self.moves.len() {
                let m = self.moves[self.i].clone();
                self.i += 1;
                if !foul_tried.contains(&m) {
                    return Some(m);
                }
            }
            None
        }

        fn name(&self) -> &'static str {
            "scripted"
        }
    }

    /// ファクトリ版APIが「反則→観測に足して指し直し→受理」の試行ループを
    /// name 版と同じ規約で回すこと（bin/tune のシナリオ目的関数の配管の担保）
    #[test]
    fn ファクトリ版の試行は反則を経て受理手を返す() {
        let sc = load("keima");
        let rep = replay(&sc.kifu, 4); // 先手番。7f の歩が 7e へ進める
        let factory: &(dyn Fn(u64) -> Box<dyn strategy::Strategy + Send> + Sync) = &|_seed| {
            Box::new(Scripted {
                moves: vec!["1g1e".into(), "7f7e".into()], // 1g1e は2マス動く反則
                i: 0,
            })
        };
        let (accepted, foul_seq) = choice_trial_one_with(&rep, 0, factory);
        assert_eq!(accepted, "7f7e");
        assert_eq!(foul_seq, vec!["1g1e".to_string()]);

        let items = [(&sc, &rep)];
        let stats = choice_trials_batch_with(&items, 2, factory, |_, _, _, _| {});
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].tally, vec![("7f7e".to_string(), 2)]);
        assert_eq!(stats[0].total_fouls, 2);
        assert_eq!(stats[0].bad_hits(&["7f7e".into()]), 2);
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
