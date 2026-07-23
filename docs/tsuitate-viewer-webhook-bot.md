# ついたて将棋ビューワー向け webhook bot（`webhook_bot`）

tsuboshun氏運営の第三者サイト「ついたて将棋ビューワー」に estimator_v10 を
参加させるためのアダプタ。tsuitate リポジトリ本体（本番bot、`main.rs`/`client.rs`
のSocket.IO常駐接続）とは完全に独立したプロセス・プロトコル。

対応は**標準「ついたて」(9x9) のみ**。盤サイズ9x9はboard.rs/shogi.rs/estimator.rs/
strategy.rs/NN特徴量にモジュール横断でハードコードされており、「ついたて5五」や
カスタム盤には対応していない（`webhook_session::choose_move` が
`game.type != "ついたて"` または `requiredPlayers != {b:1,w:1}`（リレー形式）を
検出したら 400 を返すだけで、盤面計算は一切行わない）。

## プロトコル概要

サイトのdispatcherが `your_turn` を毎手POSTしてくるステートレスなHTTP webhook。
真実は運営者提供のサンプル
（<https://github.com/tsuboshun/tsuitate-sample-bot>）のREADME。

- 盤面は SFEN、指し手は CSA 形式（7文字固定: 符号1 + 移動元2桁 + 移動先2桁 + 駒種2文字）。
  マスは USI と違い筋・段とも数字（例 `"76"`）
- 相手の手は常にマスクされる: 捕獲時のみ `+00<to>ZZ` で移動先が開示され、
  それ以外は `+0000ZZ`。自分の手は常に全開示
- `lastCapture` は **1文字のUSI駒コード**（P/L/N/S/G/B/R/K）。`lastMove` に
  埋め込まれる2文字のCSA駒コード（FU/KY/...）とは別表記。
  **README中の `lastCapture` の例示（"FU"型）は誤りで、実際のエンジン実装
  （tsuboshun/tsuitate-shogi-crates の `tsuitate_bindings/src/game_api.rs`
  `last_capture()` が `PieceKind::to_usi()` を呼んでいる）で確認済み**
- `positions` はply（反則試行含む）をキーにした完全な履歴。SFENは使わず、
  各plyの `lastMove`/`lastInfo`/`lastCapture`/`wasPromotion` だけから
  `Observation` イベント列を組み立てる（詳細は `webhook_session.rs` 冒頭コメント）

## モジュール

- `webhook_protocol.rs` — `BotTurnRequest`/`PositionEntry` 等のserde型
- `webhook_hmac.rs` — HMAC-SHA256署名検証（`timestamp + "." + rawBody`）
- `webhook_csa.rs` — CSA⇔内部表現の変換。`parse_csa_move` が7文字固定のCSAを
  パースし、`usi_move_to_csa` が自分の選んだUSI手を送信用CSAへ変換する
  （盤上移動の駒種は「移動前の自駒配置」から解決する）
- `webhook_session.rs` — ply履歴から `ObservationLog`/`GameModel`/`PlayerView`
  を組み立て、gameIdごとに `Box<dyn Strategy>` をメモリ上にキャッシュする。
  キャッシュ済みなら新しいplyぶんだけ増分で読み進め、キャッシュを失った
  （プロセス再起動直後・老朽化したセッションの掃除後）場合は0手目から
  作り直す。老朽化したセッションはリクエストのたびに掃除する（TTL 2時間）
- `src/bin/webhook_bot.rs` — エントリポイント。`tiny_http` の同期HTTPサーバーで
  リクエストごとにスレッドを立てる（本体は非同期ランタイム未使用のため、
  tokio/axum一式ではなくこちらに合わせた）

## 環境変数

| 変数 | 既定値 | 説明 |
| --- | --- | --- |
| `TSUITATE_WEBHOOK_SECRET` | （必須） | サイト登録時に発行されるWebhook Secret |
| `TSUITATE_WEBHOOK_BIND` | `127.0.0.1:8787` | bind先。Caddy等でリバースプロキシする前提 |
| `TSUITATE_WEBHOOK_PATH` | `/webhook` | 受け付けるパス。サイト登録時のエンドポイントURLと一致させる |
| `TSUITATE_WEBHOOK_STRATEGY` | `estimator_v10` | 戦略名（`strategy::make` が認識する名前） |
| `WEBHOOK_TIMESTAMP_TOLERANCE_SECONDS` | `300` | HMAC timestampの許容秒数 |
| `TSUITATE_THINK_BUDGET_MS` | `2000`（strategy.rs既定） | 登録する「レスポンス時間」より十分小さい値に絞ること |

## デプロイ手順（既存VPS、`tsuitate/scripts/server/setup/07-bot.sh` と同じ思想）

tsuitate-bot本体の運営bot（`tsuitate-bot.service`）とは別サービスとして、
既存VPS（systemd常駐＋Caddy自動HTTPS）に相乗りする。AWSは不要。

1. ローカルで `openssl rand -hex 16` してランダムなwebhookパスを決める
2. VPSで `cargo build --release --bin webhook_bot`
3. 仮のSecretで `tsuitate-webhook-bot.service`（systemd、`tsuitate-bot.service`
   と同パターンで `EnvironmentFile` を分離）を用意し起動:

   ```ini
   [Unit]
   Description=tsuitate-viewer webhook bot (webhook_bot)
   After=network.target

   [Service]
   User=tsuitate
   WorkingDirectory=/home/tsuitate/tsuitate-bot
   EnvironmentFile=/home/tsuitate/tsuitate-webhook-bot.env
   ExecStart=/home/tsuitate/tsuitate-bot/target/release/webhook_bot
   Restart=always
   RestartSec=5
   MemoryMax=512M

   [Install]
   WantedBy=multi-user.target
   ```

   ```
   # /home/tsuitate/tsuitate-webhook-bot.env
   TSUITATE_WEBHOOK_SECRET=temporary-secret-before-registration
   TSUITATE_WEBHOOK_PATH=/webhook/<上で生成したランダム値>
   TSUITATE_WEBHOOK_STRATEGY=estimator_v10
   TSUITATE_THINK_BUDGET_MS=2000
   ```

4. 既存Caddyfile（`beta.tsuitate.info`）にパスベースの `handle` ブロックを
   1つ追記して `127.0.0.1:8787` へリバースプロキシする（新規ドメイン不要）:

   ```
   beta.tsuitate.info {
       handle /webhook/* {
           reverse_proxy 127.0.0.1:8787
       }
       reverse_proxy 127.0.0.1:3000
   }
   ```

   `caddy validate --config /etc/caddy/Caddyfile` → `systemctl reload caddy`。
   この編集は tsuitate リポジトリの自動化スクリプト（`05-https.sh` 等）には
   含めない。tsuitate-bot側の独立運用としてこの手順書で管理する
5. 「ついたて将棋ビューワー」の「Bot作成」フォームでBot名（`:` 始まり）と
   `https://beta.tsuitate.info/webhook/<path>` を登録 → 表示されるWebhook Secret
   を控える（**一度しか表示されない**）
6. env fileに本物のSecretを書いて `systemctl restart tsuitate-webhook-bot`

## 既知の制約

- **プロセス再起動直後、進行中対局への初回応答は0手目からのフルreplayになる**。
  合成した80ply（≒40手ずつ）の履歴で estimator_v10 のコールドスタートreplayを
  実測したところ約6.5秒かかった（`webhook_session::tests::
  long_synthetic_history_replays_cold_start_with_estimator_v10_within_deadline`、
  `cargo test --release -- --ignored` で再実行できる）。登録する「レスポンス時間」
  が5秒程度だとこのケースで間に合わない可能性がある。systemd `Restart=always`
  により再起動自体は稀想定だが、頻発するようならセッションの永続化
  （ply/カーソルの簡易ディスク保存）を検討する
- `sfen` フィールドを使った `GameModel::diff_view` 相当の整合性チェックは
  実装していない（観測ログ経路のみで自己完結）。ズレの実例が出たら追加検討する
- ゲームID使い回し等で色が食い違った場合はセッションを作り直す（`SessionStore::session_for`）

## ローカル動作確認

```sh
cargo build --release --bin webhook_bot
TSUITATE_WEBHOOK_SECRET=testsecret TSUITATE_WEBHOOK_BIND=127.0.0.1:8799 \
TSUITATE_WEBHOOK_STRATEGY=heuristic ./target/release/webhook_bot
```

別ターミナルで署名つきリクエストを送る（`openssl` で HMAC-SHA256 を計算）:

```sh
BODY='{"type":"your_turn","requestId":"r1","gameId":"g1","color":"b","number":0,"ply":0,"deadlineMs":5000,"game":{"type":"ついたて","requiredPlayers":{"b":1,"w":1}},"positions":{"0":{"sfen":"lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1","fouls":{"b":9,"w":9}}}}'
TS=$(date +%s)
SIG=$(printf '%s.%s' "$TS" "$BODY" | openssl dgst -sha256 -hmac testsecret | sed 's/^.* //')
curl -s -X POST "http://127.0.0.1:8799/webhook" \
  -H "Content-Type: application/json" \
  -H "X-Tsuitate-Timestamp: $TS" \
  -H "X-Tsuitate-Signature: sha256=$SIG" \
  --data "$BODY"
# => {"move":"+2726FU"}
```
