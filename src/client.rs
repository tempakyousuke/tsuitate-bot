//! Socket.IO 接続と対局ループ。
//!
//! コールバックスレッドからは Msg をチャネルに流すだけにして、
//! 状態（対局ID・反則済みの手・観測履歴）はメインループが一元管理する。

use std::collections::HashSet;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::{self, sleep};
use std::time::{Duration, Instant};

use rust_socketio::client::Client;
use rust_socketio::{ClientBuilder, Event, Payload, RawClient, TransportType};
use serde::de::DeserializeOwned;
use serde_json::json;

use crate::model::GameModel;
use crate::observation::{Observation, ObservationLog};
use crate::record::{self, GameRecorder};
use crate::protocol::{
    Ack, Color, GameEndPayload, GameStatus, MatchFoundPayload, MoveAck, MoveAcceptedPayload,
    OpponentMovedPayload, PlayerView, SyncAck,
};
use crate::strategy::{self, Strategy};

pub struct Config {
    pub url: String,
    pub token: String,
    /// 着手前の待ち時間（人間らしさ用。0でも動く）
    pub think_delay_ms: u64,
    /// 終局後に再度キューへ並ぶまでの待ち時間
    pub requeue_delay_ms: u64,
    /// queue:join 拒否（受付時間外など）・queue:closed 後に再試行するまでの待ち時間。
    /// 受付時間（平日21-22時 / 土日21-24時 JST）外は拒否され続けるので、
    /// この間隔でポーリングして開場を待つ
    pub queue_retry_ms: u64,
    /// 戦略名（strategy::make が知っている名前）
    pub strategy_name: String,
    /// 対局記録（JSONL）の出力先ディレクトリ。None で記録しない
    pub record_dir: Option<String>,
    /// 序盤の強制（`TSUITATE_OPEN_BOOK`。`rl::forced::bot_book_from_env`）。None でオフ
    pub opening_book: Option<crate::rl::forced::ForcedBook>,
}

#[derive(Debug)]
enum Msg {
    Connected,
    Closed,
    SocketError(String),
    QueueAck(Ack),
    RetryQueue,
    /// 受付時間の終了で待機列から外された（サーバーの queue:closed）
    QueueClosed(String),
    MatchFound(MatchFoundPayload),
    /// 接続時に届く「進行中の対局があるか」（プロセス再起動後の復帰用）
    ActiveGame(Option<String>),
    /// game:state / 反則リトライなど「考え直すべき」合図
    ThinkTrigger,
    MoveAccepted(MoveAcceptedPayload),
    OpponentMoved(OpponentMovedPayload),
    OpponentFoul(u32),
    Check(Color),
    GameEnd(GameEndPayload),
    /// game:sync の ack
    Sync(Option<PlayerView>),
    /// game:move の ack
    MoveResult { usi: String, ack: MoveAck },
}

const ACK_TIMEOUT: Duration = Duration::from_secs(10);

fn schedule_retry(tx: Sender<Msg>, delay_ms: u64) {
    thread::spawn(move || {
        sleep(Duration::from_millis(delay_ms));
        let _ = tx.send(Msg::RetryQueue);
    });
}

/// Payload の第1引数を型付きで取り出す。
/// 通常イベントは `Text([arg0, ...])`、ack コールバックは引数列がさらに
/// 配列に包まれて `Text([[arg0, ...]])` で届くため、両方を受ける。
fn parse_first<T: DeserializeOwned>(payload: &Payload) -> Option<T> {
    let parsed = match payload {
        Payload::Text(values) => values
            .first()
            .map(|v| match v.as_array() {
                Some(args) => args.first().cloned().unwrap_or(serde_json::Value::Null),
                None => v.clone(),
            })
            .and_then(|v| serde_json::from_value(v).ok()),
        _ => None,
    };
    if parsed.is_none() {
        eprintln!("payload を解釈できませんでした: {payload:?}");
    }
    parsed
}

fn forward<T, F>(tx: &Sender<Msg>, to_msg: F) -> impl FnMut(Payload, RawClient) + Send + 'static
where
    T: DeserializeOwned,
    F: Fn(T) -> Msg + Send + 'static,
{
    let tx = tx.clone();
    move |payload, _| {
        if let Some(parsed) = parse_first::<T>(&payload) {
            let _ = tx.send(to_msg(parsed));
        }
    }
}

fn connect(config: &Config, tx: &Sender<Msg>) -> Result<Client, rust_socketio::Error> {
    let tx_open = tx.clone();
    let tx_close = tx.clone();
    let tx_err = tx.clone();
    let tx_state = tx.clone();
    let tx_check = tx.clone();

    ClientBuilder::new(config.url.clone())
        .transport_type(TransportType::Websocket)
        .auth(json!({ "token": config.token }))
        .reconnect(true)
        .reconnect_on_disconnect(true)
        .on(Event::Connect, move |_, _| {
            let _ = tx_open.send(Msg::Connected);
        })
        .on(Event::Close, move |_, _| {
            let _ = tx_close.send(Msg::Closed);
        })
        .on(Event::Error, move |payload, _| {
            let _ = tx_err.send(Msg::SocketError(format!("{payload:?}")));
        })
        .on(
            "queue:closed",
            forward(tx, |v: serde_json::Value| {
                Msg::QueueClosed(v["reason"].as_str().unwrap_or("").to_string())
            }),
        )
        .on("match:found", forward(tx, Msg::MatchFound))
        .on(
            "game:active",
            forward(tx, |v: serde_json::Value| {
                Msg::ActiveGame(v["gameId"].as_str().map(String::from))
            }),
        )
        .on("game:state", move |_: Payload, _| {
            // 中身は game:sync で取り直すので合図だけ流す
            let _ = tx_state.send(Msg::ThinkTrigger);
        })
        .on("game:moveAccepted", forward(tx, Msg::MoveAccepted))
        .on("game:opponentMoved", forward(tx, Msg::OpponentMoved))
        .on(
            "game:opponentFoul",
            forward(tx, |v: serde_json::Value| {
                Msg::OpponentFoul(v["opponentFoulCount"].as_u64().unwrap_or(0) as u32)
            }),
        )
        .on(
            "game:check",
            move |payload: Payload, _| {
                if let Some(v) = parse_first::<serde_json::Value>(&payload) {
                    match serde_json::from_value(v["inCheck"].clone()) {
                        Ok(color) => {
                            let _ = tx_check.send(Msg::Check(color));
                        }
                        Err(e) => eprintln!("game:check の inCheck を解釈できませんでした: {e}"),
                    }
                }
            },
        )
        .on("game:end", forward(tx, Msg::GameEnd))
        .connect()
}

struct BotState {
    game_id: Option<String>,
    /// この手番中に反則になった手（同じ手を繰り返さない）
    foul_tried: HashSet<String>,
    last_move_number: u32,
    /// 着手を送信済みの手番番号。思考トリガが重複しても二重に指さないためのガード。
    /// 反則 ack で解除して同じ手番を指し直す
    pending_move_number: Option<u32>,
    /// queue:join の成功 ack を受けて待機列にいるか
    in_queue: bool,
    /// queue:join の保険タイマーと失敗時タイマーを重複して増やさないための予定時刻
    queue_retry_due: Option<Instant>,
    /// 直近に送った手（moveAccepted の記録用）
    last_sent: Option<String>,
    /// 送ったが結果（受理か反則）をまだ受け取っていない手（選んだ時点の手数, USI）。
    /// 再接続のとき、取りこぼした結果を視界から作り直すのに使う（`resume::reconcile`）
    awaiting: Option<(u32, String)>,
    /// プロセス再起動後の復帰対局: 次の sync で対局記録から観測ログを読み直す
    restore_pending: bool,
    /// 対局中の再接続: 次の sync で、切断中に取りこぼした観測を視界から作り直す
    reconcile_pending: bool,
    log: ObservationLog,
    /// 対局ごとに作り直す（推定系の戦略は対局内の内部状態を持つ）
    strategy: Box<dyn Strategy>,
    /// 対局記録（record_dir 指定時のみ、対局ごとに作る）
    recorder: Option<GameRecorder>,
}

impl BotState {
    /// 観測をメモリ上のログと対局記録の両方へ流す
    fn observe(&mut self, obs: Observation) {
        if let Some(rec) = &mut self.recorder {
            rec.observation(&obs);
        }
        self.log.record(obs);
    }
}

fn schedule_retry_once(state: &mut BotState, tx: &Sender<Msg>, delay_ms: u64) {
    let delay = Duration::from_millis(delay_ms);
    let due = Instant::now() + delay;
    if let Some(current) = state.queue_retry_due {
        if current <= due {
            return;
        }
    }
    state.queue_retry_due = Some(due);
    schedule_retry(tx.clone(), delay_ms);
}

/// 接続して対局し続ける。復帰不能なエラーでのみ返る。
pub fn run(config: Config) -> Result<(), rust_socketio::Error> {
    let (tx, rx): (Sender<Msg>, Receiver<Msg>) = channel();
    let socket = connect(&config, &tx)?;

    let make_strategy = || {
        strategy::make(&config.strategy_name)
            .expect("main で検証済みの戦略名") // main.rs が起動時に検証する
    };
    let mut state = BotState {
        game_id: None,
        foul_tried: HashSet::new(),
        last_move_number: 0,
        pending_move_number: None,
        in_queue: false,
        queue_retry_due: None,
        last_sent: None,
        awaiting: None,
        restore_pending: false,
        reconcile_pending: false,
        log: ObservationLog::default(),
        strategy: make_strategy(),
        recorder: None,
    };
    println!("戦略: {}", state.strategy.name());

    let join_queue = |socket: &Client, tx: &Sender<Msg>, state: &mut BotState| {
        let tx_ack = tx.clone();
        // queue:join はデータ引数なし（ack のみ）
        let result = socket.emit_with_ack(
            "queue:join",
            Payload::Text(vec![]),
            ACK_TIMEOUT,
            move |payload: Payload, _| {
                if let Some(ack) = parse_first::<Ack>(&payload) {
                    let _ = tx_ack.send(Msg::QueueAck(ack));
                }
            },
        );
        if let Err(e) = result {
            eprintln!("queue:join の送信に失敗: {e}");
        }
        schedule_retry_once(state, tx, config.queue_retry_ms);
    };

    let request_sync = |socket: &Client, tx: &Sender<Msg>, game_id: &str| {
        let tx = tx.clone();
        let result = socket.emit_with_ack(
            "game:sync",
            json!({ "gameId": game_id }),
            ACK_TIMEOUT,
            move |payload: Payload, _| {
                let state = parse_first::<SyncAck>(&payload).and_then(|a| a.state);
                let _ = tx.send(Msg::Sync(state));
            },
        );
        if let Err(e) = result {
            eprintln!("game:sync の送信に失敗: {e}");
        }
    };

    for msg in rx.iter() {
        match msg {
            Msg::Connected => {
                println!("接続しました: {}", config.url);
                state.pending_move_number = None;
                if let Some(game_id) = state.game_id.clone() {
                    // 再接続: 対局中なら状態を取り直す。切断中の通知は再送されないので、
                    // 取りこぼした観測は次の sync の視界から作り直す
                    state.reconcile_pending = true;
                    request_sync(&socket, &tx, &game_id);
                } else {
                    state.in_queue = false;
                    join_queue(&socket, &tx, &mut state);
                }
            }
            Msg::Closed => println!("切断されました（自動再接続します）"),
            Msg::SocketError(e) => eprintln!("socketエラー: {e}"),
            Msg::QueueAck(ack) => {
                if ack.ok {
                    state.in_queue = true;
                    state.queue_retry_due = None;
                    println!("キューに参加しました");
                } else if state.game_id.is_some() {
                    // 進行中の対局へ復帰済み（game:active）。再キューしない
                } else {
                    state.in_queue = false;
                    state.queue_retry_due = None;
                    // 受付時間外など。開場を待って再試行し続ける（常駐運用）
                    eprintln!(
                        "キュー参加失敗: {:?}（{}秒後に再試行）",
                        ack.error,
                        config.queue_retry_ms / 1000
                    );
                    schedule_retry_once(&mut state, &tx, config.queue_retry_ms);
                }
            }
            Msg::RetryQueue => {
                let Some(due) = state.queue_retry_due else {
                    continue;
                };
                let now = Instant::now();
                if now < due {
                    let remaining_ms = due.duration_since(now).as_millis().max(1) as u64;
                    schedule_retry(tx.clone(), remaining_ms);
                    continue;
                }
                state.queue_retry_due = None;
                if state.game_id.is_none() && !state.in_queue {
                    join_queue(&socket, &tx, &mut state);
                }
            }
            Msg::QueueClosed(reason) => {
                state.in_queue = false;
                state.queue_retry_due = None;
                println!(
                    "受付終了で待機列から外されました: {reason}（{}秒間隔で再試行）",
                    config.queue_retry_ms / 1000
                );
                schedule_retry_once(&mut state, &tx, config.queue_retry_ms);
            }
            Msg::MatchFound(m) => {
                println!("マッチ成立: {:?} 番（相手は終局まで匿名）", m.your_color);
                state.in_queue = false;
                state.queue_retry_due = None;
                state.foul_tried.clear();
                state.last_move_number = 0;
                state.pending_move_number = None;
                state.last_sent = None;
                state.awaiting = None;
                state.restore_pending = false;
                state.reconcile_pending = false;
                state.log.clear();
                // 序盤の強制は、新しい対局と、対局記録から観測ログを読み直せた復帰対局で付ける
                // （`restore_from_record`。読み直せなかった復帰対局は、ラインの何手目まで指したかを数えられない）
                if let Some(plan) =
                    config.opening_book.as_ref().and_then(|b| b.plan_for_game(&m.game_id, m.your_color))
                {
                    println!("序盤の強制: ライン #{} を最大 {} 手", plan.line_id, plan.limit);
                }
                state.strategy = crate::rl::forced::wrap_for_game(
                    make_strategy(),
                    config.opening_book.as_ref(),
                    &m.game_id,
                    m.your_color,
                );
                state.recorder = None; // 最初の sync で作る（復帰対局と共通の経路）
                state.game_id = Some(m.game_id);
            }
            Msg::ActiveGame(Some(game_id)) => {
                if state.game_id.is_none() {
                    // プロセス再起動などで対局IDを失った状態からの復帰。
                    // それまでの観測は、次の sync で対局記録から読み直す（`restore_from_record`。
                    // 読み直せなければ空のログのまま続き、sync とのズレ警告が出る）
                    println!("進行中の対局に復帰します: {game_id}");
                    state.in_queue = false;
                    state.queue_retry_due = None;
                    state.restore_pending = true;
                    state.reconcile_pending = false;
                    state.game_id = Some(game_id.clone());
                    request_sync(&socket, &tx, &game_id);
                }
            }
            Msg::ActiveGame(None) => {}
            Msg::ThinkTrigger => {
                if let Some(game_id) = state.game_id.clone() {
                    sleep(Duration::from_millis(config.think_delay_ms));
                    request_sync(&socket, &tx, &game_id);
                }
            }
            Msg::Sync(Some(view)) => handle_sync(&socket, &tx, &mut state, view, &config),
            Msg::Sync(None) => {
                // 対局中のはずなのにサーバーが対局を知らない
                // → サーバー再起動などで対局が消えた。キューへ戻る
                if state.game_id.take().is_some() {
                    println!("対局が見つかりませんでした（サーバー再起動？）。キューへ戻ります");
                    if let Some(mut rec) = state.recorder.take() {
                        rec.aborted("server_lost", &state.log.summary());
                    }
                    state.foul_tried.clear();
                    state.pending_move_number = None;
                    state.in_queue = false;
                    state.queue_retry_due = None;
                    state.last_sent = None;
                    state.awaiting = None;
                    state.restore_pending = false;
                    state.reconcile_pending = false;
                    schedule_retry_once(&mut state, &tx, config.requeue_delay_ms);
                }
            }
            Msg::MoveAccepted(p) => {
                state.awaiting = None;
                if let Some(usi) = state.last_sent.take() {
                    if let Some(role) = p.captured {
                        println!("着手 {usi} で {role:?} を取りました");
                    }
                    let captured = p.captured.map(crate::shogi::unpromote_role);
                    state.observe(Observation::MyMove {
                        move_number: p.move_number,
                        usi,
                        captured,
                    });
                }
            }
            Msg::OpponentMoved(p) => {
                if let Some(sq) = &p.captured_your_piece_at {
                    println!("{sq} の自駒が取られました");
                }
                state.observe(Observation::OpponentMoved {
                    move_number: p.move_number,
                    captured_my_piece_at: p.captured_your_piece_at.clone(),
                });
                let _ = tx.send(Msg::ThinkTrigger);
            }
            Msg::OpponentFoul(count) => {
                println!("相手が反則しました（{count}回目）");
                state.observe(Observation::OpponentFoul { count });
            }
            Msg::Check(color) => {
                state.observe(Observation::Check { in_check: color });
            }
            Msg::GameEnd(end) => {
                println!(
                    "終局: {} ({}) — vs {} (R{})",
                    end.result, end.reason, end.opponent.username, end.opponent.rating
                );
                println!("観測サマリ: {}", state.log.summary());
                if let Some(mut rec) = state.recorder.take() {
                    rec.end(&end, &state.log.summary());
                }
                state.game_id = None;
                state.foul_tried.clear();
                state.awaiting = None;
                state.restore_pending = false;
                state.reconcile_pending = false;
                state.in_queue = false;
                state.queue_retry_due = None;
                schedule_retry_once(&mut state, &tx, config.requeue_delay_ms);
            }
            Msg::MoveResult { usi, ack } => {
                if !ack.ok {
                    state.awaiting = None;
                }
                if !ack.ok && ack.reason.as_deref() == Some("foul") {
                    println!(
                        "反則: {usi}（{}回目）→ 指し直し",
                        ack.foul_count.unwrap_or(0)
                    );
                    // 再接続で視界から作り直した反則（`reconcile_after_reconnect`）は二重に記録しない
                    let fresh = state.foul_tried.insert(usi.clone());
                    state.pending_move_number = None; // 同じ手番を指し直す
                    if fresh {
                        state.observe(Observation::MyFoul {
                            move_number: state.last_move_number,
                            usi,
                        });
                    }
                    let _ = tx.send(Msg::ThinkTrigger);
                } else if !ack.ok {
                    eprintln!("着手エラー: {:?}", ack.error);
                    state.pending_move_number = None;
                    if let Some(game_id) = state.game_id.clone() {
                        request_sync(&socket, &tx, &game_id);
                    }
                }
            }
        }
    }
    Ok(())
}

/// 復帰対局で、読み直した履歴を戦略へ食わせる時間の上限の既定（webhook 版のコールドスタートと同じ
/// `TSUITATE_COLD_START_PREWARM_MS`）。残りは次の choose が通常の更新で処理する
const DEFAULT_RESUME_PREWARM_MS: u64 = 2_500;

/// プロセス再起動後の復帰対局: 対局記録（`TSUITATE_RECORD_DIR` の `<時刻>-<対局ID>.jsonl`）から観測ログを
/// 読み直し、止まっていた間の欠け（高々相手の1手と、送った手の結果）を視界から作り直す。
/// 履歴を失うと、その局の勝率が 12〜17pt 下がる（docs/rl-deepnash-design.md の「入力と行動マスクの点検」）。
/// 読み直せない・今の局面とつながらないときは、従来どおり空のログで続ける
fn restore_from_record(state: &mut BotState, view: &PlayerView, config: &Config) {
    let Some(dir) = config.record_dir.as_deref() else {
        println!("復帰: 対局記録を取っていないので、履歴なしで続けます");
        return;
    };
    let Some(path) = record::find_record(dir, &view.game_id) else {
        println!("復帰: 対局記録が見つからないので、履歴なしで続けます");
        return;
    };
    let loaded = match record::load_record(&path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("復帰: 対局記録を読めないので、履歴なしで続けます: {e}");
            return;
        }
    };
    if loaded.finished || loaded.game_id != view.game_id || loaded.your_color != view.your_color {
        eprintln!("復帰: 対局記録 {} はこの対局の続きではないので、履歴なしで続けます", path.display());
        return;
    }
    let mut log = ObservationLog::default();
    for e in &loaded.events {
        log.record(e.clone());
    }
    let pending = loaded.pending.as_ref().map(|(n, usi)| (*n, usi.as_str()));
    let repairs = match crate::resume::reconcile(view.your_color, &log, pending, view) {
        Ok(r) => r,
        Err(why) => {
            eprintln!(
                "復帰: 対局記録の観測 {} 件が今の局面とつながらないので、履歴なしで続けます（{why}）",
                loaded.events.len()
            );
            return;
        }
    };
    // 続きは同じファイルへ追記する（1対局 = 1ファイル）。開けなければ、後で新しいファイルを作る
    match GameRecorder::resume(&path) {
        Ok(mut rec) => {
            rec.resumed(loaded.events.len(), repairs.len());
            state.recorder = Some(rec);
        }
        Err(e) => eprintln!("復帰: 対局記録へ追記できません: {e}"),
    }
    state.log = log;
    let repaired = repairs.len();
    for e in repairs {
        state.observe(e);
    }
    state.foul_tried = if view.turn == view.your_color {
        crate::resume::fouls_this_turn(&state.log, view.move_number)
    } else {
        HashSet::new()
    };
    state.last_move_number = view.move_number;
    // 戦略は作り直して（序盤の強制は観測ログから何手目かを数え直す）、読み直した履歴を食わせる
    state.strategy = crate::rl::forced::wrap_for_game(
        strategy::make(&config.strategy_name).expect("main で検証済みの戦略名"),
        config.opening_book.as_ref(),
        &view.game_id,
        view.your_color,
    );
    let budget_ms = std::env::var("TSUITATE_COLD_START_PREWARM_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_RESUME_PREWARM_MS);
    strategy::prewarm_strategy_with_budget(
        &mut *state.strategy,
        view,
        &state.log,
        Some(Duration::from_millis(budget_ms)),
    );
    println!(
        "復帰: 対局記録から観測 {} 件を読み直しました（止まっていた間のぶんを視界から {} 件作り直しました）",
        loaded.events.len(),
        repaired
    );
}

/// 対局中の再接続: 切断中に送られた通知は再送されないので、取りこぼした観測（高々相手の1手と、送った手の
/// 結果）を視界から作り直してログへ足す。作り直せない欠けはそのまま残る（sync とのズレ警告が出る）
fn reconcile_after_reconnect(state: &mut BotState, view: &PlayerView) {
    let pending = state.awaiting.as_ref().map(|(n, usi)| (*n, usi.as_str()));
    match crate::resume::reconcile(view.your_color, &state.log, pending, view) {
        Ok(repairs) if repairs.is_empty() => {}
        Ok(repairs) => {
            println!("再接続: 取りこぼした観測 {} 件を視界から作り直しました", repairs.len());
            for e in repairs {
                match &e {
                    Observation::MyFoul { usi, .. } => {
                        state.foul_tried.insert(usi.clone());
                    }
                    Observation::MyMove { .. } => state.last_sent = None,
                    _ => {}
                }
                state.observe(e);
            }
            state.awaiting = None;
        }
        Err(why) => eprintln!("再接続: 取りこぼした観測を作り直せませんでした（{why}）"),
    }
}

fn handle_sync(
    socket: &Client,
    tx: &Sender<Msg>,
    state: &mut BotState,
    view: PlayerView,
    config: &Config,
) {
    if state.game_id.as_deref() != Some(view.game_id.as_str()) {
        return;
    }
    let record_dir = config.record_dir.as_deref();

    // 復帰・再接続の直後の最初の sync で、観測ログを今の局面までつなぐ（手番でなくても行う。
    // 以後の通知がつながった続きとして積まれるように）
    if view.status == GameStatus::Playing {
        if std::mem::take(&mut state.restore_pending) {
            state.reconcile_pending = false;
            restore_from_record(state, &view, config);
        } else if std::mem::take(&mut state.reconcile_pending) {
            reconcile_after_reconnect(state, &view);
        }
    }

    // 対局ごとの記録ファイルは最初の sync で作る（match:found 起点の新規対局と
    // game:active 起点の復帰対局の共通経路）
    if view.status == GameStatus::Playing && state.recorder.is_none() {
        if let Some(dir) = record_dir {
            match GameRecorder::create(dir, &view.game_id, view.your_color, state.strategy.name())
            {
                Ok(rec) => {
                    println!("対局記録: {}", rec.path().display());
                    state.recorder = Some(rec);
                }
                Err(e) => eprintln!("対局記録ファイルを作れませんでした: {e}"),
            }
        }
    }

    if view.status != GameStatus::Playing || view.turn != view.your_color {
        return;
    }
    if state.pending_move_number == Some(view.move_number) {
        return; // この手番はすでに着手済み（受理待ちを含む）
    }

    if view.move_number != state.last_move_number {
        state.last_move_number = view.move_number;
        state.foul_tried.clear();
    }

    // 観測履歴からの再構成と sync を照合（切断中の取りこぼし等の検出）
    if let Some(diff) = GameModel::from_log(view.your_color, &state.log).diff_view(&view) {
        eprintln!("観測モデルと sync がズレています（再接続などで観測が欠けた可能性）: {diff}");
    }

    let think_started = Instant::now();
    let chosen = state.strategy.choose(&view, &state.log, &state.foul_tried);
    let think_ms = think_started.elapsed().as_millis() as u64;
    match chosen {
        None => {
            // 候補が尽きた（すべて反則）→ 投了
            println!("指せる手がありません。投了します");
            if let Some(rec) = &mut state.recorder {
                rec.resigned(view.move_number);
            }
            let _ = socket.emit("game:resign", json!({ "gameId": view.game_id }));
        }
        Some(usi) => {
            if let Some(rec) = &mut state.recorder {
                let debug = state.strategy.debug_state();
                rec.chosen(view.move_number, &usi, think_ms, debug.as_ref());
            }
            state.pending_move_number = Some(view.move_number);
            state.last_sent = Some(usi.clone());
            state.awaiting = Some((view.move_number, usi.clone()));
            let tx = tx.clone();
            let sent = usi.clone();
            let result = socket.emit_with_ack(
                "game:move",
                json!({ "gameId": view.game_id, "usi": usi }),
                ACK_TIMEOUT,
                move |payload: Payload, _| {
                    if let Some(ack) = parse_first::<MoveAck>(&payload) {
                        let _ = tx.send(Msg::MoveResult {
                            usi: sent.clone(),
                            ack,
                        });
                    }
                },
            );
            if let Err(e) = result {
                eprintln!("game:move の送信に失敗: {e}");
            }
        }
    }
}
