# tsuitate-bot

王様のかくれんぼ（tsuitate リポジトリ）用の外部bot。
Socket.IO クライアントとしてサイトに接続し、ランダムマッチに参加して人間と対戦する。

思考の既定は `estimator`（観測履歴から相手局面の粒子集合を維持し、候補手を評価する）。
旧来の前進ヒューリスティック＋乱数（`heuristic`）も残してある。

接続プロトコルの詳細は tsuitate リポジトリの `docs/bot-api.md` を参照。
開発・検証（アリーナ、シナリオ、採点、凍結版）の手順は `CLAUDE.md`（`AGENTS.md` は同一内容のシンボリックリンク）。

## 使い方

1. サイトにログインし、マイページの「bot管理」でbotを作成してAPIトークン（`tsb_...`）を取得
2. 実行:

```sh
TSUITATE_URL=http://localhost:5173 \
TSUITATE_BOT_TOKEN=tsb_... \
cargo run --release
```

キューに自動で並び、マッチしたら対局し、終局したらまた並ぶ。Ctrl-C で終了。

常駐運用を想定しており、以下は自動で処理される:

- ランダムマッチの受付時間外（`queue:join` 拒否・`queue:closed`）は
  `TSUITATE_QUEUE_RETRY_MS` 間隔で再試行して開場を待つ
- サーバー再起動などで対局が消えた場合（`game:sync` が対局を返さない）はキューへ戻る

### 環境変数

| 変数 | 既定値 | 説明 |
| --- | --- | --- |
| `TSUITATE_URL` | `http://localhost:5173` | 接続先サイト |
| `TSUITATE_BOT_TOKEN` | （必須） | マイページで発行したAPIトークン |
| `TSUITATE_THINK_MS` | `600` | 着手前の待ち時間 ms |
| `TSUITATE_THINK_BUDGET_MS` | `2000` | estimator の1手あたり思考予算 ms（アリーナ・本番ともこのまま） |
| `TSUITATE_STRATEGY` | `estimator` | 戦略名（`heuristic` や `estimator_v13` なども可） |
| `TSUITATE_QUEUE_RETRY_MS` | `60000` | キュー参加拒否（受付時間外など）後の再試行間隔 ms |
| `TSUITATE_RECORD_DIR` | `records` | 対局記録（JSONL）の出力先。空文字で無効 |

評価ノブ（`TSUITATE_*`）の一覧と採否の経緯は `CLAUDE.md`。

## 構成

- `protocol.rs` — サイト側イベント契約（`src/lib/shared/events.ts`）の Rust 版
- `board.rs` — 盤座標と「自分の駒だけを考慮した」候補手生成（`move-hints.ts` の移植）
- `observation.rs` — 観測履歴（自分の反則・取った/取られた駒・王手宣言）。bot が得る情報の全量で、推定器の入力
- `estimator.rs` / `strategy.rs` — 粒子フィルタと候補手評価。既定戦略はここ
- `client.rs` — Socket.IO 接続と対局ループ（コールバック→チャネル→単一メインループ）
- `frozen/` — アリーナ比較用の凍結版戦略（既定のガントレット対象は v9 以降）

## 設計メモ

- サーバーは相手の駒・指し手を一切教えてくれない（ついたて将棋の情報秘匿はサーバー側で保証）。
  botが得られる情報は `observation.rs` の観測のみで、人間のプレイヤーと完全に同等
- 反則（ack `reason: "foul"`）でも手番は変わらない。同じ手を繰り返さないよう記録して指し直す。
  候補が尽きたら投了。累計10回で反則負けになるので、反則は「情報」としても使う
- 時計はフィッシャー 300秒+3秒。思考予算の既定 2000ms なら時間切れは出ない（実測）

## 関連ツール

- `cargo run --release --bin arena -- [対局数] [候補] [基準...]` — 戦略同士のローカル対戦
- `cargo run --release --bin scenario -- suite` — 実戦棋譜の局面再現
- `webhook_bot` — 第三者サイト「ついたて将棋ビューワー」向け HTTP アダプタ（`docs/tsuitate-viewer-webhook-bot.md`）
