# tsuitate-bot

ついたて将棋オンライン（`~/Develop/tsuitate`）に外部bot APIで接続して対戦するRust製bot。
サイト・ソルバー（`~/Develop/tsuitate-resolver`）とは**意図的に独立**したプロジェクト（cargo依存もしない）。

## コマンド

- `cargo test` — ユニットテスト（候補手生成・プロトコルのデシリアライズ）
- `cargo build` / `cargo run --release` — 実行には環境変数が必要:
  - `TSUITATE_BOT_TOKEN`（必須）: サイトのマイページ「bot管理」で発行する `tsb_...` トークン
  - `TSUITATE_URL`（既定 `http://localhost:5173`）
  - `TSUITATE_THINK_MS`（既定 600）: 着手前の待ち時間

## アーキテクチャ

コールバック（Socket.IOスレッド）→ mpsc チャネル → 単一メインループ、の一方向。
状態（対局ID・反則済みの手・観測履歴）はメインループだけが触る。

- `protocol.rs` — サイト側イベント契約の serde 版。**真実は tsuitate リポジトリの
  `src/lib/shared/events.ts` / `game-types.ts`**。サイト側の契約が変わったらここを追随させる
- `board.rs` — 「自分の駒だけを考慮した」候補手生成。tsuitate の `src/lib/shared/move-hints.ts` の移植。
  実際の合法性はサーバーだけが判定する（相手の駒は見えない）
- `strategy.rs` — 指し手選択。現状は前進ヒューリスティック＋乱数。**強化はここの置き換え**
  （observation.rs の観測履歴から「あり得る相手局面の情報集合」を維持して探索する構想）
- `observation.rs` — 観測履歴。ついたて将棋で得られる情報はこれが全量:
  自分の手の受理/反則（理由は不明）・取った駒種・自駒が取られたマス・王手/反則宣言
- `client.rs` — 接続と対局ループ。反則リトライ（同じ手を繰り返さない）、
  `pending_move_number` による二重着手ガード、再接続時の `game:sync` 復帰、終局後の自動再キュー

## ルール上の前提（サイト側仕様）

- 反則しても手番は変わらない。ack が `reason: "foul"` なら別の手を指し直す。累計10回で反則負け
- 時計はフィッシャー 300秒+3秒。思考が遅いと時間切れ負け
- 同時1対局のみ。bot同士・所有者とはマッチしない
- 接続方法の一次資料は tsuitate リポジトリの `docs/bot-api.md`

## ハマりどころ

- **rust_socketio 0.6 の ack コールバックは引数列が配列に包まれて届く**
  （通常イベント `Text([arg0])`、ack `Text([[arg0, ...]])`）。`client.rs` の `parse_first` が両対応
- `queue:join` はデータ引数なし（ack のみ）で emit すること。余計な引数を付けると
  サーバー側で ack の位置がずれる

## E2E 検証手順

tsuitate リポジトリ側でスクラッチDBのdevサーバーを立てて1局打たせる:

1. シード（人間ユーザーのセッション + botトークンを発行）: tsuitate リポジトリで
   `createBotAccount` / `createSession` を呼ぶ小スクリプトを `DATABASE_URL=<scratch>.db npx tsx` で実行
2. `DATABASE_URL=<scratch>.db npm run dev -- --port 5175` でサーバー起動
3. このbotを `TSUITATE_URL=http://localhost:5175` で起動（バックグラウンド）
4. 人間役スクリプト（socket.io-client + クッキー認証。数手指して投了）を実行し、
   マッチ成立 → 交互着手 → 終局まで両者のログで確認

過疎判定（人間接続数 < 4）によりレート無視で即マッチするので、人間役1人で成立する。
