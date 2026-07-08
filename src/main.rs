//! 王様のかくれんぼ用の外部bot。
//!
//! 環境変数:
//! - TSUITATE_URL: 接続先（既定 http://localhost:5173）
//! - TSUITATE_BOT_TOKEN: マイページで発行したAPIトークン（必須）
//! - TSUITATE_THINK_MS: 着手前の待ち時間 ms（既定 600）
//! - TSUITATE_STRATEGY: 戦略名（既定は strategy::DEFAULT_STRATEGY）
//! - TSUITATE_QUEUE_RETRY_MS: キュー参加拒否（受付時間外）後の再試行間隔 ms（既定 60000）

use std::process::exit;

use tsuitate_bot::{client, strategy};

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
    let strategy_name =
        std::env::var("TSUITATE_STRATEGY").unwrap_or_else(|_| strategy::DEFAULT_STRATEGY.into());
    if strategy::make(&strategy_name).is_none() {
        eprintln!("未知の戦略名です: {strategy_name}");
        exit(1);
    }

    let queue_retry_ms = std::env::var("TSUITATE_QUEUE_RETRY_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60_000);

    let config = client::Config {
        url,
        token,
        think_delay_ms,
        requeue_delay_ms: 3_000,
        queue_retry_ms,
        strategy_name,
    };

    if let Err(e) = client::run(config) {
        eprintln!("接続に失敗しました: {e}");
        exit(1);
    }
}
