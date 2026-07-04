//! ついたて将棋オンライン用の外部bot。
//!
//! 環境変数:
//! - TSUITATE_URL: 接続先（既定 http://localhost:5173）
//! - TSUITATE_BOT_TOKEN: マイページで発行したAPIトークン（必須）
//! - TSUITATE_THINK_MS: 着手前の待ち時間 ms（既定 600）

mod board;
mod client;
mod observation;
mod protocol;
mod strategy;

use std::process::exit;

fn main() {
    let url = std::env::var("TSUITATE_URL").unwrap_or_else(|_| "http://localhost:5173".into());
    let Ok(token) = std::env::var("TSUITATE_BOT_TOKEN") else {
        eprintln!("環境変数 TSUITATE_BOT_TOKEN にAPIトークンを設定してください");
        exit(1);
    };
    let think_delay_ms = std::env::var("TSUITATE_THINK_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(600);

    let config = client::Config {
        url,
        token,
        think_delay_ms,
        requeue_delay_ms: 3_000,
    };

    if let Err(e) = client::run(config) {
        eprintln!("接続に失敗しました: {e}");
        exit(1);
    }
}
