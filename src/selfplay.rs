//! 戦略同士のローカル対戦（bin/arena と bin/tune が共用する対局ループ）。
//!
//! サーバー（judge.ts / game-room.ts）と同じ裁定を再現する:
//! - 反則（フル盤面で非合法な手）は手番を変えずカウント。MAX_FOULS=10 で反則負け
//! - フィッシャー時計（1000秒+3秒。本番の300秒+3秒より厚い）をシミュレート。choose() の壁時計を消費し、
//!   時間切れは負け。加算は受理された手の後のみ（反則では加算しない）
//! - 各戦略に見えるのは自分の PlayerView 相当と観測イベントのみ（公平性の担保）
//! - 王手宣言は両者に、取った駒種は指した側に、取られたマスは相手側に通知
//! - 詰み・ステイルメイト・投了（choose が None）・手数上限で終局

use std::collections::HashSet;
use std::sync::OnceLock;
use std::time::Instant;

use rand::Rng;

use crate::observation::{Observation, ObservationLog};
use crate::protocol::{
    ClockState, Color, FoulCounts, FoulRecord, GameEndPayload, GameStatus, MoveRecord,
    OpponentInfo, PlayerView, RatingChange, RatingChangePair,
};
use crate::record::GameRecorder;
use crate::shogi::{Outcome, Position, parse_usi, unpromote_role};
use crate::strategy::Strategy;

pub const MAX_FOULS: u32 = 10;
/// これを超えたら引き分け扱い（千日手検出は未実装のため）。
/// 400手だと1ガントレットの実時間が長すぎるため200手に短縮
/// （200手を超える対局は膠着がほとんどで、勝敗の判別力への寄与が薄い）
pub const MAX_PLIES: u32 = 200;
/// フィッシャー時計の既定。本番サイトは 300秒+3秒 だが、このリポジトリの対戦は
/// 1000秒+3秒 で行う（思考予算を厚くして強さの上限を探るため。
/// 本番へのデプロイ時は TSUITATE_THINK_BUDGET_MS で思考時間を絞って調整する）
pub const DEFAULT_FISCHER_INITIAL_MS: i64 = 1_000_000;
pub const DEFAULT_FISCHER_INCREMENT_MS: i64 = 3_000;

/// 持ち時間（ms）。`ARENA_FISCHER_INITIAL_MS` で上書きできる。
/// **本番相当（300秒+3秒）でアリーナを回して時間配分を検証する**ためのノブ
/// （docs/improvement-plan-2026-07-26-yaneuraou.md 項目1）。
/// 名前を `ARENA_` にしているのは審判側の設定であり、戦略側が読む
/// `TSUITATE_*`（凍結版も読む = 両側に効く）と混ぜないため
pub fn fischer_initial_ms() -> i64 {
    static V: OnceLock<i64> = OnceLock::new();
    *V.get_or_init(|| env_i64("ARENA_FISCHER_INITIAL_MS", DEFAULT_FISCHER_INITIAL_MS))
}

/// 1手ごとの加算（ms）。`ARENA_FISCHER_INCREMENT_MS` で上書きできる。
/// 受理された手の後にだけ加算される（反則には付かない）ので、
/// 反則の多い戦略は同じ持ち時間でも実効の予算が薄くなる
pub fn fischer_increment_ms() -> i64 {
    static V: OnceLock<i64> = OnceLock::new();
    *V.get_or_init(|| env_i64("ARENA_FISCHER_INCREMENT_MS", DEFAULT_FISCHER_INCREMENT_MS))
}

/// 対局ループから毎手呼ばれるので値はキャッシュする（env::var はロックを取る）
fn env_i64(name: &str, default: i64) -> i64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .filter(|&v| v >= 0)
        .unwrap_or(default)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    Win(Color),
    Draw,
}

/// 1対局ぶんの結果（**局ごとのペア差の分散を実測する**ため）。
///
/// 通常 arena の集計は勝敗の合計しか残さないので、「同じ `ARENA_MATCH_SEED` の
/// 2本を局ごとに突き合わせたときの `Var(delta)`」が測れなかった。
/// checkpoint arena の効率比（var·CPU秒）は arena 側のこの実測が無いと
/// `Var(delta)=0.5` の仮定に乗ったままになる（issue #19 の P0 の残り）。
///
/// 突き合わせのキーは `(baseline, match_seed, game_no)`。`game_no` の偶奇で
/// 先後が決まるので、同じ `match_seed` の2本では対局条件列が揃う。
#[derive(Debug, Clone)]
pub struct GameOutcome {
    pub game_no: u32,
    /// A が先手だったか（`game_no % 2 == 0`）
    pub a_is_sente: bool,
    /// A 視点のスコア（勝1 / 分0.5 / 負0）
    pub score_a: f64,
    pub reason: String,
    pub plies: u32,
    pub fouls_a: u32,
    pub fouls_b: u32,
    pub fouls_in_check_a: u32,
    pub fouls_in_check_b: u32,
    /// この対局での思考時間の合計（ms）。CPU コストの実測に使う
    pub think_ms_a: f64,
    pub think_ms_b: f64,
    pub moves_a: u32,
    pub moves_b: u32,
}

#[derive(Debug, Default)]
pub struct MatchStats {
    pub games: u32,
    pub wins_a: u32,
    pub wins_b: u32,
    pub draws: u32,
    /// 終局理由ごとの回数（A勝ち, B勝ち問わず）
    pub checkmate: u32,
    pub stalemate: u32,
    pub foul_limit: u32,
    pub resign: u32,
    pub timeout: u32,
    pub max_plies: u32,
    pub total_plies: u64,
    pub fouls_a: u64,
    pub fouls_b: u64,
    /// うち王手を受けている局面での反則（王手ソルバーの効果測定用）
    pub fouls_in_check_a: u64,
    pub fouls_in_check_b: u64,
    /// 1手ごとの思考時間（マイクロ秒）。時間切れ検証と改善の副作用チェック用
    pub think_us_a: Vec<u64>,
    pub think_us_b: Vec<u64>,
    /// 支給された持ち時間の合計（初期持ち時間＋受理手ごとの加算）。
    /// 消費（= think_us の合計）との比が「クロックの何%を使っているか」で、
    /// 時間配分（docs/improvement-plan-2026-07-26-yaneuraou.md 項目A）の伸びしろそのもの
    pub clock_granted_ms_a: u64,
    pub clock_granted_ms_b: u64,
    /// 対局中に残り時間が最も減った時点の値（ms）。時間切れまでの余裕の実測。
    /// None は未計測（1手も指していない）
    pub clock_min_ms_a: Option<i64>,
    pub clock_min_ms_b: Option<i64>,
    /// 1局ごとの結果（`ARENA_GAMES_JSON` で書き出す）。スレッドに分散するので
    /// 順序は `game_no` 昇順ではない（書き出し側で整列する）
    pub per_game: Vec<GameOutcome>,
}

impl MatchStats {
    /// 引き分けを除いた A の勝率と、二項近似の95%信頼区間の半幅
    pub fn rate_and_ci(&self) -> (f64, f64) {
        let decisive = self.wins_a + self.wins_b;
        if decisive == 0 {
            return (0.5, 0.0);
        }
        let rate = self.wins_a as f64 / decisive as f64;
        let se = (rate * (1.0 - rate) / decisive as f64).sqrt();
        (rate, se * 1.96)
    }

    /// 引き分けを0.5勝と数えたスコア率（チューニングの目的関数用。
    /// 決着局だけの勝率より情報が密で、引き分け増加も正しく罰する/報いる）
    pub fn score_rate(&self) -> f64 {
        if self.games == 0 {
            return 0.5;
        }
        (self.wins_a as f64 + 0.5 * self.draws as f64) / self.games as f64
    }

    pub fn merge(&mut self, other: MatchStats) {
        self.wins_a += other.wins_a;
        self.wins_b += other.wins_b;
        self.draws += other.draws;
        self.checkmate += other.checkmate;
        self.stalemate += other.stalemate;
        self.foul_limit += other.foul_limit;
        self.resign += other.resign;
        self.timeout += other.timeout;
        self.max_plies += other.max_plies;
        self.total_plies += other.total_plies;
        self.fouls_a += other.fouls_a;
        self.fouls_b += other.fouls_b;
        self.fouls_in_check_a += other.fouls_in_check_a;
        self.fouls_in_check_b += other.fouls_in_check_b;
        self.think_us_a.extend(other.think_us_a);
        self.think_us_b.extend(other.think_us_b);
        self.clock_granted_ms_a += other.clock_granted_ms_a;
        self.clock_granted_ms_b += other.clock_granted_ms_b;
        self.clock_min_ms_a = min_opt(self.clock_min_ms_a, other.clock_min_ms_a);
        self.clock_min_ms_b = min_opt(self.clock_min_ms_b, other.clock_min_ms_b);
        self.per_game.extend(other.per_game);
        self.games += other.games;
    }
}

fn min_opt(a: Option<i64>, b: Option<i64>) -> Option<i64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (x, None) | (None, x) => x,
    }
}

struct PlayerState {
    strategy: Box<dyn Strategy>,
    log: ObservationLog,
    fouls: u32,
    fouls_in_check: u32,
    foul_tried: HashSet<String>,
    /// None = 時計なし（checkpoint arena の継続対局）。時間切れ終局が存在せず、
    /// フィッシャー加算もしない。**思考時間の記録（think_us）は時計の有無に
    /// 依らず必ず取る**（「遅くなったが勝率が上がった」偽の改善を弾くため）
    clock_ms: Option<i64>,
    /// この対局で支給された持ち時間（初期＋加算の累計）
    clock_granted_ms: u64,
    /// 残り時間の最小値（時間切れまでの余裕の実測）
    clock_min_ms: Option<i64>,
    think_us: Vec<u64>,
    /// 選択の履歴 (move_number, usi, debug)。記録の chose イベント用
    /// （p_legal 較正の測定に使う。MyMove/MyFoul 観測と1:1で対応する）
    chosen: Vec<(u32, String, Option<serde_json::Value>)>,
}

fn make_view(
    pos: &Position,
    color: Color,
    fouls: &[u32; 2],
    clocks_ms: &[i64; 2],
    game_no: u32,
) -> PlayerView {
    let idx = |c: Color| if c == Color::Sente { 0 } else { 1 };
    PlayerView {
        game_id: format!("arena-{game_no}"),
        your_color: color,
        your_pieces: pos.pieces_of(color),
        your_hand: pos.hand_map(color),
        turn: pos.turn(),
        move_number: pos.move_number(),
        clocks: ClockState {
            sente_ms: clocks_ms[0],
            gote_ms: clocks_ms[1],
            running: Some(pos.turn()),
            server_time: 0,
        },
        fouls: FoulCounts {
            you: fouls[idx(color)],
            opponent: fouls[idx(color.other())],
        },
        you_in_check: pos.in_check(color),
        opponent_in_check: pos.in_check(color.other()),
        status: GameStatus::Playing,
    }
}

/// 審判だけが知る対局の真実（全手順・反則試行）。
/// ARENA_RECORD_DIR 設定時に bin/analyze が読める記録として書き出す
pub struct GameTruth {
    pub moves: Vec<MoveRecord>,
    pub foul_attempts: Vec<FoulRecord>,
}

/// oracle[先手, 後手]: 診断用。Some の側は該当する反則手が「候補から外して
/// 指し直し」になり、反則カウント・観測は発生しない（OracleMode 参照）。
///
/// `start` は開始局面と**絶対手数**（通常の対局は `Position::initial()` と 0、
/// checkpoint arena は途中局面とそこまでの手数）。両者の観測ログ・反則数・
/// `foul_tried` は `players` 側が持っているので、ここでは受け取らない。
/// MAX_PLIES・反則上限・王手/捕獲通知・終局判定は開始点に依らず共通
fn play_game_with_oracle(
    players: &mut [PlayerState; 2],
    game_no: u32,
    oracle: [Option<OracleMode>; 2],
    start: (Position, u32),
) -> (GameResult, &'static str, u32, GameTruth) {
    let (mut pos, mut plies) = start;
    let idx = |c: Color| if c == Color::Sente { 0usize } else { 1usize };
    let mut truth = GameTruth {
        moves: vec![],
        foul_attempts: vec![],
    };

    loop {
        if plies >= MAX_PLIES {
            return (GameResult::Draw, "max_plies", plies, truth);
        }
        let side = pos.turn();
        let fouls = [players[0].fouls, players[1].fouls];
        // 時計なし（clock_ms = None）でも view の残り時間は必要なので初期値を見せる。
        // 戦略側は clocks を読まない（読むのは bridge.rs の表示だけ）
        let clocks_ms = [
            players[0].clock_ms.unwrap_or_else(fischer_initial_ms),
            players[1].clock_ms.unwrap_or_else(fischer_initial_ms),
        ];
        let view = make_view(&pos, side, &fouls, &clocks_ms, game_no);

        let mover = &mut players[idx(side)];
        let started = Instant::now();
        let choice = mover.strategy.choose(&view, &mover.log, &mover.foul_tried);
        let elapsed = started.elapsed();
        mover.think_us.push(elapsed.as_micros() as u64);
        if let Some(clock_ms) = mover.clock_ms.as_mut() {
            *clock_ms -= elapsed.as_millis() as i64;
            let now = *clock_ms;
            mover.clock_min_ms = Some(match mover.clock_min_ms {
                Some(m) => m.min(now),
                None => now,
            });
            if now <= 0 {
                return (GameResult::Win(side.other()), "timeout", plies, truth);
            }
        }
        let Some(usi) = choice else {
            return (GameResult::Win(side.other()), "resign", plies, truth);
        };

        let legal = parse_usi(&usi).is_some_and(|mv| pos.is_legal(&mv));
        if !legal {
            // 反則: 手番は変わらない（judge.ts と同じ）。フィッシャー加算もしない
            let in_check = pos.in_check(side);
            // 診断用オラクル: 該当する反則手は握りつぶして指し直させる
            // （反則カウントも観測も発生しない = 完全な合法性知識の上限測定）
            let skip = match oracle[idx(side)] {
                Some(OracleMode::NoFoul) => true,
                Some(OracleMode::CheckNoFoul) => in_check,
                None => false,
            };
            if skip {
                players[idx(side)].foul_tried.insert(usi);
                continue;
            }
            let mover = &mut players[idx(side)];
            mover.fouls += 1;
            if in_check {
                mover.fouls_in_check += 1;
            }
            mover.foul_tried.insert(usi.clone());
            let debug = mover.strategy.debug_state();
            mover.chosen.push((pos.move_number(), usi.clone(), debug));
            let foul_count = mover.fouls;
            truth.foul_attempts.push(FoulRecord {
                move_number: pos.move_number(),
                by_color: side,
                usi: usi.clone(),
            });
            mover.log.record(Observation::MyFoul {
                move_number: pos.move_number(),
                usi,
            });
            players[idx(side.other())]
                .log
                .record(Observation::OpponentFoul { count: foul_count });
            if foul_count >= MAX_FOULS {
                return (GameResult::Win(side.other()), "foul_limit", plies, truth);
            }
            continue;
        }

        let mv = parse_usi(&usi).unwrap();
        {
            let mover = &mut players[idx(side)];
            let debug = mover.strategy.debug_state();
            mover.chosen.push((pos.move_number(), usi.clone(), debug));
        }
        truth.moves.push(MoveRecord {
            usi: usi.clone(),
            by_color: side,
            ms: elapsed.as_millis() as u64,
            fouls_before: players[idx(side)].fouls,
        });
        let captured = pos.play_unchecked(&mv);
        plies += 1;
        players[idx(side)].foul_tried.clear();
        if let Some(clock_ms) = players[idx(side)].clock_ms.as_mut() {
            *clock_ms += fischer_increment_ms();
            players[idx(side)].clock_granted_ms += fischer_increment_ms() as u64;
        }

        // 通知（game-room.ts と同じ内容・同じ moveNumber 規約 = 適用後の値）
        let move_number = pos.move_number();
        let captured_square = captured.map(|_| match mv {
            crate::shogi::ShogiMove::Board { to, .. } => crate::board::make_usi_square(to),
            crate::shogi::ShogiMove::Drop { .. } => unreachable!("打ちでは駒を取れない"),
        });
        players[idx(side)].log.record(Observation::MyMove {
            move_number,
            usi,
            captured: captured.map(unpromote_role),
        });
        players[idx(side.other())].log.record(Observation::OpponentMoved {
            move_number,
            captured_my_piece_at: captured_square,
        });
        if pos.in_check(pos.turn()) {
            let in_check = pos.turn();
            for p in players.iter_mut() {
                p.log.record(Observation::Check { in_check });
            }
        }

        match pos.outcome() {
            Some(Outcome::Checkmate { winner }) => {
                return (GameResult::Win(winner), "checkmate", plies, truth);
            }
            Some(Outcome::Stalemate { winner }) => {
                return (GameResult::Win(winner), "stalemate", plies, truth);
            }
            None => {}
        }
    }
}

/// A視点の対局記録を ARENA_RECORD_DIR に書く（bin/analyze が読める形式）。
/// 実対局と違い審判が真実を持っているので、end ペイロードの全手順は正確
fn write_record(
    dir: &str,
    game_no: u32,
    a_is_sente: bool,
    players: &[PlayerState; 2],
    result: GameResult,
    reason: &str,
    truth: GameTruth,
) {
    let a_color = if a_is_sente { Color::Sente } else { Color::Gote };
    let idx = |c: Color| if c == Color::Sente { 0usize } else { 1usize };
    let pa = &players[idx(a_color)];
    let pb = &players[idx(a_color.other())];
    let mut rec = match GameRecorder::create(
        dir,
        &format!("arena-{game_no}"),
        a_color,
        pa.strategy.name(),
    ) {
        Ok(rec) => rec,
        Err(e) => {
            eprintln!("対局記録を作成できません（{dir}）: {e}");
            return;
        }
    };
    // chose イベントを対応する MyMove/MyFoul 観測の直前に挟む
    // （実対局の記録と同じ順序 = analyze の p_legal 突き合わせが機能する）
    let mut chosen_iter = pa.chosen.iter();
    for obs in pa.log.events() {
        if matches!(
            obs,
            Observation::MyMove { .. } | Observation::MyFoul { .. }
        ) {
            if let Some((mn, usi, debug)) = chosen_iter.next() {
                rec.chosen(*mn, usi, 0, debug.as_ref());
            }
        }
        rec.observation(obs);
    }
    let result_str = match result {
        GameResult::Draw => "draw",
        GameResult::Win(Color::Sente) => "sente_win",
        GameResult::Win(Color::Gote) => "gote_win",
    };
    let zero = RatingChange { before: 0, after: 0 };
    rec.end(
        &GameEndPayload {
            result: result_str.into(),
            reason: reason.into(),
            final_sfen: String::new(),
            moves: truth.moves,
            foul_attempts: truth.foul_attempts,
            rating_change: RatingChangePair {
                you: zero.clone(),
                opponent: zero,
            },
            opponent: OpponentInfo {
                username: pb.strategy.name().into(),
                rating: 0,
                is_bot: true,
            },
        },
        "arena",
    );
}

/// 1局ぶんの結果を stats に集計する
fn record_game(
    stats: &mut MatchStats,
    game_no: u32,
    a_is_sente: bool,
    players: [PlayerState; 2],
    result: GameResult,
    reason: &str,
    plies: u32,
) {
    let [sente, gote] = players;
    let (pa, pb) = if a_is_sente {
        (sente, gote)
    } else {
        (gote, sente)
    };
    // 局ごとの結果（ペア差の分散の実測用）。think_us は下で move されるので先に畳む
    let sum_ms = |us: &[u64]| us.iter().sum::<u64>() as f64 / 1000.0;
    stats.per_game.push(GameOutcome {
        game_no,
        a_is_sente,
        score_a: match result {
            GameResult::Draw => 0.5,
            GameResult::Win(winner) => {
                if (winner == Color::Sente) == a_is_sente {
                    1.0
                } else {
                    0.0
                }
            }
        },
        reason: reason.to_string(),
        plies,
        fouls_a: pa.fouls,
        fouls_b: pb.fouls,
        fouls_in_check_a: pa.fouls_in_check,
        fouls_in_check_b: pb.fouls_in_check,
        think_ms_a: sum_ms(&pa.think_us),
        think_ms_b: sum_ms(&pb.think_us),
        moves_a: pa.think_us.len() as u32,
        moves_b: pb.think_us.len() as u32,
    });
    stats.fouls_a += pa.fouls as u64;
    stats.fouls_b += pb.fouls as u64;
    stats.fouls_in_check_a += pa.fouls_in_check as u64;
    stats.fouls_in_check_b += pb.fouls_in_check as u64;
    stats.think_us_a.extend(pa.think_us);
    stats.think_us_b.extend(pb.think_us);
    stats.clock_granted_ms_a += pa.clock_granted_ms;
    stats.clock_granted_ms_b += pb.clock_granted_ms;
    stats.clock_min_ms_a = min_opt(stats.clock_min_ms_a, pa.clock_min_ms);
    stats.clock_min_ms_b = min_opt(stats.clock_min_ms_b, pb.clock_min_ms);
    stats.total_plies += plies as u64;
    match reason {
        "checkmate" => stats.checkmate += 1,
        "stalemate" => stats.stalemate += 1,
        "foul_limit" => stats.foul_limit += 1,
        "resign" => stats.resign += 1,
        "timeout" => stats.timeout += 1,
        _ => stats.max_plies += 1,
    }
    match result {
        GameResult::Draw => stats.draws += 1,
        GameResult::Win(winner) => {
            let a_won = (winner == Color::Sente) == a_is_sente;
            if a_won {
                stats.wins_a += 1;
            } else {
                stats.wins_b += 1;
            }
        }
    }
}

/// 診断用オラクル（ARENA_ORACLE_A）。候補(A)の反則手を審判が握りつぶして
/// 指し直させる = 「完全な合法性知識があったら」の上限を測る。
/// 反則経済が勝敗をどれだけ支配しているかのシーリング測定専用で、
/// 対局記録は書かない（観測列が実戦と異なり学習データを汚すため）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OracleMode {
    /// すべての反則手を候補から外して指し直させる（反則ゼロの上限）
    NoFoul,
    /// 王手中の反則だけ外す（王手解消の完全知識の上限）
    CheckNoFoul,
}

fn oracle_from_env() -> Option<OracleMode> {
    match std::env::var("ARENA_ORACLE_A").as_deref() {
        Ok("nofoul") => Some(OracleMode::NoFoul),
        Ok("check_nofoul") => Some(OracleMode::CheckNoFoul),
        _ => None,
    }
}

/// 1対局・1プレイヤーぶんの決定論的な乱数コンテキスト。
/// ランのシードと対局番号から導出され、スレッドのスケジューリングに依存しない。
/// SPSA（bin/tune）の f+/f− 評価で同じ対局条件列を再利用する（共通乱数法）ために使う。
/// 注意: 推定器の時間打ち切り（壁時計デッドライン）は決定論化できないため、
/// CPU負荷によるわずかな非決定性は残る（定跡・手のサンプリング等の主要な乱数は揃う）
#[derive(Debug, Clone, Copy)]
pub struct GameSeeds {
    pub game_no: u32,
    /// このプレイヤー用のシード（推定器・タイブレーク・定跡選択の乱数源）
    pub seed: u64,
}

/// SplitMix64。単純なXORや加算だとシード間に相関が残るため撹拌する
pub fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn player_seed(match_seed: u64, game_no: u32, is_a: bool) -> u64 {
    mix(match_seed ^ mix(u64::from(game_no) * 2 + u64::from(is_a)))
}

/// 対局の並列数。既定はコア数-2（推定器の時間予算・思考時間計測が
/// コア競合で歪みすぎないよう全コアは使わない）。ARENA_THREADS で上書き可
pub fn thread_count() -> usize {
    if let Some(n) = std::env::var("ARENA_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        return n.max(1);
    }
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(2))
        .unwrap_or(1)
        .max(1)
}

/// A と B を games 局（先後交代で）対戦させる。戦略はファクトリで対局ごとに作る。
/// シードは不要な用途向け（アリーナ等）。ファクトリはシードを受け取らない
pub fn run_match_with<FA, FB>(games: u32, make_a: &FA, make_b: &FB) -> MatchStats
where
    FA: Fn() -> Box<dyn Strategy> + Sync,
    FB: Fn() -> Box<dyn Strategy> + Sync,
{
    run_match_with_seeds(
        games,
        rand::rng().random(),
        &|_| make_a(),
        &|_| make_b(),
    )
}

/// A と B を games 局（先後交代で）対戦させる（シード付き）。
/// 各対局・各プレイヤーの GameSeeds は (match_seed, game_no) から決定論的に
/// 導出されるため、同じ match_seed で呼べば同じ対局条件列になる
/// （SPSA の f+/f− ペアリング用）。対局同士は独立なのでスレッドに分散する。
/// game_no の偶奇で先後を決めるためラウンドロビンに割っても先後バランスは保たれる。
/// 注意: 並列実行中の思考時間はコア競合ぶん逐次実行より長めに出る
pub fn run_match_with_seeds<FA, FB>(
    games: u32,
    match_seed: u64,
    make_a: &FA,
    make_b: &FB,
) -> MatchStats
where
    FA: Fn(GameSeeds) -> Box<dyn Strategy> + Sync,
    FB: Fn(GameSeeds) -> Box<dyn Strategy> + Sync,
{
    let threads = thread_count().min(games.max(1) as usize);
    // 診断用オラクル（A側のみ）。有効時は観測列が実戦と異なるため記録は書かない
    let oracle_a = oracle_from_env();
    // 設定時は A 視点の対局記録を書き出す（bin/analyze 用。空文字は無効）
    let record_dir = std::env::var("ARENA_RECORD_DIR")
        .ok()
        .filter(|s| !s.is_empty())
        .filter(|_| oracle_a.is_none());
    let record_dir = record_dir.as_deref();
    let mut stats = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    let mut local = MatchStats::default();
                    let mut game_no = t as u32;
                    while game_no < games {
                        // 偶数局は A が先手
                        let a_is_sente = game_no % 2 == 0;
                        let new_player = |strategy: Box<dyn Strategy>| PlayerState {
                            strategy,
                            log: ObservationLog::default(),
                            fouls: 0,
                            fouls_in_check: 0,
                            foul_tried: HashSet::new(),
                            clock_ms: Some(fischer_initial_ms()),
                            clock_granted_ms: fischer_initial_ms() as u64,
                            clock_min_ms: None,
                            think_us: Vec::new(),
                            chosen: Vec::new(),
                        };
                        let seeds_a = GameSeeds {
                            game_no,
                            seed: player_seed(match_seed, game_no, true),
                        };
                        let seeds_b = GameSeeds {
                            game_no,
                            seed: player_seed(match_seed, game_no, false),
                        };
                        let (sente, gote) = if a_is_sente {
                            (make_a(seeds_a), make_b(seeds_b))
                        } else {
                            (make_b(seeds_b), make_a(seeds_a))
                        };
                        let mut players = [new_player(sente), new_player(gote)];
                        let oracle = if a_is_sente {
                            [oracle_a, None]
                        } else {
                            [None, oracle_a]
                        };
                        let (result, reason, plies, truth) = play_game_with_oracle(
                            &mut players,
                            game_no,
                            oracle,
                            (Position::initial(), 0),
                        );
                        if let Some(dir) = record_dir {
                            write_record(dir, game_no, a_is_sente, &players, result, reason, truth);
                        }
                        record_game(&mut local, game_no, a_is_sente, players, result, reason, plies);
                        game_no += threads as u32;
                    }
                    local
                })
            })
            .collect();
        let mut merged = MatchStats::default();
        for h in handles {
            let local = h.join().expect("対局スレッドが panic した");
            merged.merge(local);
        }
        merged
    });
    stats.games = games;
    stats
}

/// 継続対局の開始状態（checkpoint arena）。
///
/// **v1 は手番境界のみ**を扱う: 「直前の受理手が完了し、次の手番がまだ反則を
/// 一度も試していない時点」。同一手番内の反則後 checkpoint を扱うようになったら
/// `foul_tried` をここへ明示的に足すこと（`check_foul_prior_boost` が
/// `foul_tried.len()` を読むので、復元漏れは評価そのものを変える）。
pub struct StartState {
    /// 真実の盤面と開始手番
    pub pos: Position,
    /// 両者の観測ログ [0]=先手, [1]=後手
    pub logs: [ObservationLog; 2],
    /// ここまでの累積反則数（反則負けの上限 MAX_FOULS は累計で判定される）
    pub fouls: [u32; 2],
    /// 絶対手数（MAX_PLIES の判定に使う）
    pub plies: u32,
}

/// 継続対局の結果。反則・思考時間は「継続で増えたぶん」と「累計」を分けて返す
pub struct ContinuationOutcome {
    pub result: GameResult,
    pub reason: &'static str,
    /// 終局時点の**絶対**手数
    pub plies: u32,
    /// 継続で増えた手数
    pub added_plies: u32,
    /// 終局時点の累積反則数
    pub fouls: [u32; 2],
    /// 継続で増えた反則数
    pub added_fouls: [u32; 2],
    /// 継続で増えた反則のうち王手を受けている局面だったもの
    pub added_fouls_in_check: [u32; 2],
    /// 継続中の1手ごとの思考時間（マイクロ秒）
    pub think_us: [Vec<u64>; 2],
    pub truth: GameTruth,
}

/// 途中局面から終局まで指し継ぐ（checkpoint arena）。
///
/// 裁定は通常の対局と同じ関数（`play_game_with_oracle`）を通るので、
/// MAX_PLIES・反則上限・王手/捕獲通知・終局判定は arena と共有される。
/// **時計は無効**（`clock_ms = None`）: 途中局面からの残り時間は復元できず、
/// 本番相当 300秒+3秒・100局で時間切れ0の実測があるため落としてよい。
/// ただし思考時間は必ず記録する。
///
/// `strategies` は **prewarm 済み**のものを渡すこと（prewarm のコストを
/// 継続対局のコストと分けて計測するため、ここでは prewarm しない）。
pub fn play_continuation(
    strategies: [Box<dyn Strategy>; 2],
    start: StartState,
    game_no: u32,
) -> ContinuationOutcome {
    let StartState {
        pos,
        logs,
        fouls,
        plies: start_plies,
    } = start;
    let [log_s, log_g] = logs;
    let new_player = |strategy: Box<dyn Strategy>, log: ObservationLog, fouls: u32| PlayerState {
        strategy,
        log,
        fouls,
        fouls_in_check: 0,
        foul_tried: HashSet::new(),
        clock_ms: None,
        clock_granted_ms: 0,
        clock_min_ms: None,
        think_us: Vec::new(),
        chosen: Vec::new(),
    };
    let [strat_s, strat_g] = strategies;
    let mut players = [
        new_player(strat_s, log_s, fouls[0]),
        new_player(strat_g, log_g, fouls[1]),
    ];
    let (result, reason, plies, truth) =
        play_game_with_oracle(&mut players, game_no, [None, None], (pos, start_plies));
    let [p0, p1] = players;
    ContinuationOutcome {
        result,
        reason,
        plies,
        added_plies: plies - start_plies,
        fouls: [p0.fouls, p1.fouls],
        added_fouls: [p0.fouls - fouls[0], p1.fouls - fouls[1]],
        added_fouls_in_check: [p0.fouls_in_check, p1.fouls_in_check],
        think_us: [p0.think_us, p1.think_us],
        truth,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use super::*;

    /// 即投了する戦略（シード配線のテスト用）
    struct Resigner;
    impl Strategy for Resigner {
        fn choose(
            &mut self,
            _view: &PlayerView,
            _log: &ObservationLog,
            _foul_tried: &HashSet<String>,
        ) -> Option<String> {
            None
        }
        fn name(&self) -> &'static str {
            "resigner"
        }
    }

    /// 初期局面を「開始状態」として `play_continuation` に渡した継続対局が、
    /// 通常の初期局面からの selfplay と同じ裁定になる（checkpoint arena が
    /// arena と裁定を共有していることの回帰テスト）。
    ///
    /// 推定器は壁時計デッドラインで打ち切るので手順のバイト一致は要求できない。
    /// ここでは裁定そのもの（決定論的な戦略なら同じ結果になること）を確かめる
    #[test]
    fn continuation_from_initial_matches_normal_selfplay() {
        let start = StartState {
            pos: Position::initial(),
            logs: [ObservationLog::default(), ObservationLog::default()],
            fouls: [0, 0],
            plies: 0,
        };
        let out = play_continuation([Box::new(Resigner), Box::new(Resigner)], start, 0);
        // 先手が即投了 = 後手の勝ち。通常の selfplay と同じ裁定
        assert_eq!(out.result, GameResult::Win(Color::Gote));
        assert_eq!(out.reason, "resign");
        assert_eq!(out.plies, 0);
        assert_eq!(out.added_plies, 0);

        let stats = run_match_with(1, &|| Box::new(Resigner), &|| Box::new(Resigner));
        assert_eq!(stats.resign, 1);
        assert_eq!(stats.wins_b, 1, "先手（A）が投了して B の勝ち");
    }

    /// 継続対局は開始時点の累積反則数を引き継ぎ、絶対手数で MAX_PLIES を判定する
    #[test]
    fn continuation_carries_absolute_counters() {
        let mut pos = Position::initial();
        for usi in ["7g7f", "3c3d"] {
            pos.play_unchecked(&parse_usi(usi).unwrap());
        }
        let start = StartState {
            pos,
            logs: [ObservationLog::default(), ObservationLog::default()],
            fouls: [3, 4],
            plies: 2,
        };
        let out = play_continuation([Box::new(Resigner), Box::new(Resigner)], start, 0);
        assert_eq!(out.fouls, [3, 4], "反則数は引き継ぐ");
        assert_eq!(out.added_fouls, [0, 0]);
        assert_eq!(out.plies, 2, "絶対手数は引き継ぐ");
        assert_eq!(out.added_plies, 0);
    }

    /// 同じ match_seed なら、スレッド数や実行順に関係なく
    /// 各対局・各プレイヤーに同じシードが割り当てられる（共通乱数法の土台）
    #[test]
    fn same_match_seed_reproduces_game_seeds() {
        let collect = |match_seed: u64| -> Vec<(u32, u64, u64)> {
            let a_seed: Mutex<HashMap<u32, u64>> = Mutex::new(HashMap::new());
            let b_seed: Mutex<HashMap<u32, u64>> = Mutex::new(HashMap::new());
            run_match_with_seeds(
                8,
                match_seed,
                &|gs: GameSeeds| {
                    a_seed.lock().unwrap().insert(gs.game_no, gs.seed);
                    Box::new(Resigner)
                },
                &|gs: GameSeeds| {
                    b_seed.lock().unwrap().insert(gs.game_no, gs.seed);
                    Box::new(Resigner)
                },
            );
            let a = a_seed.into_inner().unwrap();
            let b = b_seed.into_inner().unwrap();
            let mut v: Vec<(u32, u64, u64)> = (0..8u32)
                .map(|g| (g, a[&g], b[&g]))
                .collect();
            v.sort();
            v
        };
        let first = collect(42);
        let second = collect(42);
        assert_eq!(first, second, "同じシードなら同じ対局条件列");
        for (game_no, a, b) in &first {
            assert_ne!(a, b, "対局{game_no}でA/Bのシードが衝突");
        }
        let different = collect(43);
        assert_ne!(first, different, "違うシードなら違う条件列");
    }
}
