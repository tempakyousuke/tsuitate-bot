//! 「ついたて将棋ビューワー」（tsuboshun氏運営の第三者サイト）向け webhook bot。
//!
//! 本体（main.rs、Socket.IO常駐）とは完全に独立したプロセス。サイトの
//! dispatcher が `your_turn` を毎手POSTしてくる同期HTTPサーバーとして動く。
//!
//! 環境変数:
//! - TSUITATE_WEBHOOK_SECRET（必須）: サイト登録時に発行されるWebhook Secret
//! - TSUITATE_WEBHOOK_BIND（既定 127.0.0.1:8787）: bind先（Caddy等でリバースプロキシする前提）
//! - TSUITATE_WEBHOOK_PATH（既定 /webhook）: 受け付けるパス。サイト登録時のエンドポイントURLと一致させる
//! - TSUITATE_WEBHOOK_STRATEGY（既定 estimator_v10）: 戦略名
//! - WEBHOOK_TIMESTAMP_TOLERANCE_SECONDS（既定 300）: HMAC timestampの許容秒数
//! - TSUITATE_THINK_BUDGET_MS: strategy.rs 側の思考予算（既定2000ms）。
//!   登録する「レスポンス時間」より十分小さい値に絞ること

use std::io::Read;
use std::process::exit;
use std::sync::Arc;

use tiny_http::{Header, Method, Response, Server};

use tsuitate_bot::strategy;
use tsuitate_bot::webhook_hmac;
use tsuitate_bot::webhook_protocol::BotTurnRequest;
use tsuitate_bot::webhook_session::{SessionStore, choose_move};

const MAX_BODY_BYTES: u64 = 128 * 1024;

fn main() {
    let secret = match std::env::var("TSUITATE_WEBHOOK_SECRET") {
        Ok(v) if !v.is_empty() => v,
        _ => {
            eprintln!("環境変数 TSUITATE_WEBHOOK_SECRET にWebhook Secretを設定してください");
            exit(1);
        }
    };
    let bind = std::env::var("TSUITATE_WEBHOOK_BIND").unwrap_or_else(|_| "127.0.0.1:8787".into());
    let path = std::env::var("TSUITATE_WEBHOOK_PATH").unwrap_or_else(|_| "/webhook".into());
    let strategy_name =
        std::env::var("TSUITATE_WEBHOOK_STRATEGY").unwrap_or_else(|_| "estimator_v10".into());
    if strategy::make(&strategy_name).is_none() {
        eprintln!("未知の戦略名です: {strategy_name}");
        exit(1);
    }
    let tolerance_secs: i64 = std::env::var("WEBHOOK_TIMESTAMP_TOLERANCE_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);

    let server = match Server::http(&bind) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("HTTPサーバーの起動に失敗しました ({bind}): {e}");
            exit(1);
        }
    };
    println!("webhook_bot listening on http://{bind}{path} (strategy={strategy_name})");

    let store = Arc::new(SessionStore::new(strategy_name));
    let secret = Arc::new(secret);
    let path = Arc::new(path);

    for request in server.incoming_requests() {
        let store = store.clone();
        let secret = secret.clone();
        let path = path.clone();
        std::thread::spawn(move || handle(request, &store, &secret, &path, tolerance_secs));
    }
}

fn handle(
    request: tiny_http::Request,
    store: &SessionStore,
    secret: &str,
    expected_path: &str,
    tolerance_secs: i64,
) {
    let mut request = request;
    if *request.method() != Method::Post {
        respond(
            request,
            405,
            &serde_json::json!({ "error": "method_not_allowed" }),
        );
        return;
    }
    if request.url() != expected_path {
        respond(request, 404, &serde_json::json!({ "error": "not_found" }));
        return;
    }

    let mut body = Vec::new();
    if let Err(e) = request
        .as_reader()
        .take(MAX_BODY_BYTES + 1)
        .read_to_end(&mut body)
    {
        eprintln!("リクエストボディの読み取りに失敗: {e}");
        respond(
            request,
            400,
            &serde_json::json!({ "error": "invalid_body" }),
        );
        return;
    }
    if body.len() as u64 > MAX_BODY_BYTES {
        respond(
            request,
            413,
            &serde_json::json!({ "error": "request_too_large" }),
        );
        return;
    }

    let timestamp = header_value(request.headers(), "X-Tsuitate-Timestamp");
    let signature = header_value(request.headers(), "X-Tsuitate-Signature");
    let verified = match (timestamp, signature) {
        (Some(ts), Some(sig)) => webhook_hmac::verify(
            secret.as_bytes(),
            &ts,
            &sig,
            &body,
            tolerance_secs,
            webhook_hmac::unix_now(),
        ),
        _ => false,
    };
    if !verified {
        respond(
            request,
            401,
            &serde_json::json!({ "error": "unauthorized" }),
        );
        return;
    }

    let payload: BotTurnRequest = match serde_json::from_slice(&body) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("payload解析に失敗: {e}");
            respond(
                request,
                400,
                &serde_json::json!({ "error": "invalid_json" }),
            );
            return;
        }
    };

    match choose_move(store, &payload) {
        Ok(mv) => {
            println!("[{}] ply={} -> {}", payload.game_id, payload.ply, mv);
            respond(request, 200, &serde_json::json!({ "move": mv }));
        }
        Err(e) => {
            eprintln!("[{}] ply={} エラー: {e}", payload.game_id, payload.ply);
            let status = e.status_code();
            respond(
                request,
                status,
                &serde_json::json!({ "error": e.to_string() }),
            );
        }
    }
}

fn header_value(headers: &[Header], name: &'static str) -> Option<String> {
    headers
        .iter()
        .find(|h| h.field.equiv(name))
        .map(|h| h.value.as_str().to_string())
}

fn respond(request: tiny_http::Request, status: u16, body: &serde_json::Value) {
    let json = serde_json::to_vec(body).unwrap_or_default();
    let header = Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
        .expect("static content-type header is valid");
    let response = Response::from_data(json)
        .with_status_code(status)
        .with_header(header);
    if let Err(e) = request.respond(response) {
        eprintln!("レスポンス送信に失敗: {e}");
    }
}
