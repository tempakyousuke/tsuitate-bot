//! Socket.IO 接続と対局ループ。
//!
//! コールバックスレッドからは Msg をチャネルに流すだけにして、
//! 状態（対局ID・反則済みの手・観測履歴）はメインループが一元管理する。

use std::collections::HashSet;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::sleep;
use std::time::{Duration, Instant};

use rust_socketio::client::Client;
use rust_socketio::{ClientBuilder, Event, Payload, RawClient, TransportType};
use serde::de::DeserializeOwned;
use serde_json::json;

use crate::model::GameModel;
use crate::observation::{Observation, ObservationLog};
use crate::record::GameRecorder;
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
}

#[derive(Debug)]
enum Msg {
    Connected,
    Closed,
    SocketError(String),
    QueueAck(Ack),
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
            forward(tx, |v: serde_json::Value| {
                let color = serde_json::from_value(v["inCheck"].clone()).unwrap_or(Color::Sente);
                Msg::Check(color)
            }),
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
    /// 直近に送った手（moveAccepted の記録用）
    last_sent: Option<String>,
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
        last_sent: None,
        log: ObservationLog::default(),
        strategy: make_strategy(),
        recorder: None,
    };
    println!("戦略: {}", state.strategy.name());

    let join_queue = |socket: &Client, tx: &Sender<Msg>| {
        let tx = tx.clone();
        // queue:join はデータ引数なし（ack のみ）
        let result = socket.emit_with_ack(
            "queue:join",
            Payload::Text(vec![]),
            ACK_TIMEOUT,
            move |payload: Payload, _| {
                if let Some(ack) = parse_first::<Ack>(&payload) {
                    let _ = tx.send(Msg::QueueAck(ack));
                }
            },
        );
        if let Err(e) = result {
            eprintln!("queue:join の送信に失敗: {e}");
        }
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
                if let Some(game_id) = state.game_id.clone() {
                    // 再接続: 対局中なら状態を取り直す
                    request_sync(&socket, &tx, &game_id);
                } else {
                    join_queue(&socket, &tx);
                }
            }
            Msg::Closed => println!("切断されました（自動再接続します）"),
            Msg::SocketError(e) => eprintln!("socketエラー: {e}"),
            Msg::QueueAck(ack) => {
                if ack.ok {
                    println!("キューに参加しました");
                } else if state.game_id.is_some() {
                    // 進行中の対局へ復帰済み（game:active）。再キューしない
                } else {
                    // 受付時間外など。開場を待って再試行し続ける（常駐運用）
                    eprintln!(
                        "キュー参加失敗: {:?}（{}秒後に再試行）",
                        ack.error,
                        config.queue_retry_ms / 1000
                    );
                    sleep(Duration::from_millis(config.queue_retry_ms));
                    join_queue(&socket, &tx);
                }
            }
            Msg::QueueClosed(reason) => {
                println!(
                    "受付終了で待機列から外されました: {reason}（{}秒間隔で再試行）",
                    config.queue_retry_ms / 1000
                );
                sleep(Duration::from_millis(config.queue_retry_ms));
                join_queue(&socket, &tx);
            }
            Msg::MatchFound(m) => {
                println!("マッチ成立: {:?} 番（相手は終局まで匿名）", m.your_color);
                state.foul_tried.clear();
                state.last_move_number = 0;
                state.pending_move_number = None;
                state.last_sent = None;
                state.log.clear();
                state.strategy = make_strategy();
                state.recorder = None; // 最初の sync で作る（復帰対局と共通の経路）
                state.game_id = Some(m.game_id);
            }
            Msg::ActiveGame(Some(game_id)) => {
                if state.game_id.is_none() {
                    // プロセス再起動などで対局IDを失った状態からの復帰。
                    // それまでの観測は失われている（sync とのズレ警告が出る）
                    println!("進行中の対局に復帰します: {game_id}");
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
            Msg::Sync(Some(view)) => {
                handle_sync(&socket, &tx, &mut state, view, config.record_dir.as_deref())
            }
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
                    state.last_sent = None;
                    sleep(Duration::from_millis(config.requeue_delay_ms));
                    join_queue(&socket, &tx);
                }
            }
            Msg::MoveAccepted(p) => {
                if let Some(usi) = state.last_sent.take() {
                    if let Some(role) = p.captured {
                        println!("着手 {usi} で {role:?} を取りました");
                    }
                    state.observe(Observation::MyMove {
                        move_number: p.move_number,
                        usi,
                        captured: p.captured,
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
                sleep(Duration::from_millis(config.requeue_delay_ms));
                join_queue(&socket, &tx);
            }
            Msg::MoveResult { usi, ack } => {
                if !ack.ok && ack.reason.as_deref() == Some("foul") {
                    println!(
                        "反則: {usi}（{}回目）→ 指し直し",
                        ack.foul_count.unwrap_or(0)
                    );
                    state.foul_tried.insert(usi.clone());
                    state.pending_move_number = None; // 同じ手番を指し直す
                    state.observe(Observation::MyFoul {
                        move_number: state.last_move_number,
                        usi,
                    });
                    let _ = tx.send(Msg::ThinkTrigger);
                } else if !ack.ok {
                    eprintln!("着手エラー: {:?}", ack.error);
                }
            }
        }
    }
    Ok(())
}

fn handle_sync(
    socket: &Client,
    tx: &Sender<Msg>,
    state: &mut BotState,
    view: PlayerView,
    record_dir: Option<&str>,
) {
    if state.game_id.as_deref() != Some(view.game_id.as_str()) {
        return;
    }

    // 対局ごとの記録ファイルは最初の sync で作る（match:found 起点の新規対局と
    // game:active 起点の復帰対局の共通経路）
    if state.recorder.is_none() {
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
                rec.chosen(view.move_number, &usi, think_ms);
            }
            state.pending_move_number = Some(view.move_number);
            state.last_sent = Some(usi.clone());
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
