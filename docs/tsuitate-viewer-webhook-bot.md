# ついたて将棋ビューワー向け webhook bot（`webhook_bot`）

tsuboshun氏運営の第三者サイト「ついたて将棋ビューワー」向けの HTTP アダプタ。
既定の戦略は凍結版 `estimator_v10`（現行 `estimator` にしたいときは
`TSUITATE_WEBHOOK_STRATEGY=estimator`）。tsuitate リポジトリ本体（本番bot、
`main.rs`/`client.rs` のSocket.IO常駐接続）とは完全に独立したプロセス・プロトコル。

対応は**標準「ついたて」(9x9) のみ**。盤サイズ9x9はboard.rs/shogi.rs/estimator.rs/
strategy.rs/NN特徴量にモジュール横断でハードコードされており、「ついたて5五」や
カスタム盤には対応していない（`webhook_session::choose_move` が
`game.type != "ついたて"` または `requiredPlayers != {b:1,w:1}`（リレー形式）を
検出したら 400 を返すだけで、盤面計算は一切行わない）。

## プロトコル概要

サイトのdispatcherが `your_turn` を毎手POSTしてくるHTTP webhook。botプロセス内では
`gameId` ごとのセッションをTTL付きで保持し、前回までの履歴を再利用する。
真実は運営者提供のサンプル
（<https://github.com/tsuboshun/tsuitate-sample-bot>）のREADME。

- **差分プロトコル（2026-08 改訂、従量課金対策）**: 初回リクエストだけが
  手数0から現在までの全 `positions` と `game` を含み、2回目以降は
  `basePly`（botが保持済みと想定される最終手数）と `basePly+1..=ply` の差分
  `positions` だけが届く（`game` は来ない。README の型定義ではトップレベルの
  `type`/`deadlineMs` も2回目以降の型から消えているため、こちらの実装では
  どちらも任意フィールドとして扱う）。かつては毎回全履歴が届いており
  「キャッシュを失ったら受信履歴で0手目から作り直す」復旧ができたが、
  差分化で不可能になった（下記 `TSUITATE_WEBHOOK_SESSION_DIR` と「既知の制約」）
- 盤面は SFEN、指し手は CSA 形式（7文字固定: 符号1 + 移動元2桁 + 移動先2桁 + 駒種2文字）。
  マスは USI と違い筋・段とも数字（例 `"76"`）
- 相手の手は常にマスクされる: 捕獲時のみ `+00<to>ZZ` で移動先が開示され、
  それ以外は `+0000ZZ`。自分の手は常に全開示
- `lastCapture` は実戦dispatcherでは **2文字のCSA駒コード**（FU/KY/KE/GI/KI/KA/HI）。
  エンジン直結のサンプルでは1文字のUSI駒コード（P/L/N/S/G/B/R/K）が使われることも
  あるため、実装は両形式を受理する。
- `positions` はply（反則試行含む）をキーにした履歴（初回=全量、以降=差分）。
  SFENは使わず、各plyの `lastMove`/`lastInfo`/`lastCapture`/`wasPromotion` だけから
  `Observation` イベント列を組み立てる（詳細は `webhook_session.rs` 冒頭コメント）。
  差分に0手目が含まれない場合、開始時の反則残数は標準の初期値（9,9）として扱う
- 直前の相手正規手が自駒を取った場合、その捕獲升には相手の着手駒が確実にいる。
  駒打ちは捕獲できないため、その升への全ての駒打ちを候補から除外する。
  さらに、その升を飛び越える飛車・角・香などの長距離移動も除外する。
  王手回避の反則後も盤面は変わらないので、同じ手番中はこの除外を維持する
- **反則エントリの `wasPromotion` は信用しない**（2026-08-19、VPS ログ 1,302 反則行で
  確認）: 成り手の反則は `lastMove` が成駒コード付きの手そのもの（`+4741NY`）で
  届くのに `wasPromotion` は明示的に `false`（着手が適用されなかったので「成りは
  起きていない」扱いらしい。成駒コード付きの反則 202 件すべて）。`wasPromotion`
  だけを見ると `4g4a+` の反則を `4g4a` と記録してしまい、`foul_tried` の完全一致から
  漏れて**同じ成り手を反則し続ける**（実測: 同一成り手の連続反則 65 回、
  `+4741NY`×4・`-2629NY`×8 等。Socket.IO 版の `client.rs` は自分の送った USI を
  そのまま覚えるので起きない webhook 固有のバグ）。`advance` は「着手前に from に
  いた自駒の成り先 == 末尾の駒種2文字」なら `wasPromotion` に依らず成りとみなす
- `fouls` はプロトコル上任意。欠落時は標準ついたての初期残数（黒9・白9）として扱う
- `fouls` が履歴途中で欠落しても、観測した反則イベントから累計を補完する
- 同じ `requestId` を再送した場合は、セッションを進めず前回と同じCSA応答を返す
  （セッションごとに直近128件を保持）
- キャッシュから外れた古い `ply` のリクエストは、履歴を巻き戻さず409で拒否する
- `basePly` が保持済み（ディスク復元込み）のplyより先を指す差分（前回リクエストの
  取りこぼし・復元不能なキャッシュ喪失）は、セッションを壊さず 409
  `history_gap` で拒否する。保持済みが `basePly` より先行しているぶんには
  差分が重複するだけなので、処理済みplyを読み飛ばして正常に続行する
- 実戦payload（`TSUITATE_WEBHOOK_LOG_DIR`で取得、2026-07-23確認。**差分化前の
  旧仕様時点の観測**）で判明した点:
  - （旧仕様）`positions` は長い対局（実測82ply）でも常に0手目から現在plyまで
    欠番なく全量で届いていた（2026-08 の差分化でこの前提は廃止）
  - `deadlineMs` は相対時間ではなく**絶対epoch msタイムスタンプ**（リクエスト受信の
    約9〜11秒後を指す値を観測）。現状コードはこの値を参照していない
    （`webhook_protocol.rs` の `deadline_ms` は `#[allow(dead_code)]`）
  - `game` オブジェクトに `param`（URLクエリ文字列: `foul_limits=9.9&time_limits=180.180&
    byoyomi=60&promotion_rank=3&draw_move_count=150&enable_try_rule=false` 等）が
    含まれる。反則上限9（＝10回で反則負け）は実測と一致。時計は300秒+3秒のFischerではなく
    **180秒＋60秒秒読み**方式（`clocks`は戦略側で未使用のため動作への影響はない）
  - `sfen` は視点ごとにマスクされる（自分から見えない側の升は空で埋まる）。
    元々SFENには依存しない設計だったため実害はないが、実データで裏付けが取れた

## モジュール

- `webhook_protocol.rs` — `BotTurnRequest`/`PositionEntry` 等のserde型
- `webhook_hmac.rs` — HMAC-SHA256署名検証（`timestamp + "." + rawBody`）
- `webhook_csa.rs` — CSA⇔内部表現の変換。`parse_csa_move` が7文字固定のCSAを
  パースし、`usi_move_to_csa` が自分の選んだUSI手を送信用CSAへ変換する
  （盤上移動の駒種は「移動前の自駒配置」から解決する）
- `webhook_session.rs` — ply履歴から `ObservationLog`/`GameModel`/`PlayerView`
  を組み立て、gameIdごとに `Box<dyn Strategy>` をメモリ上にキャッシュする。
  標準の `game.param`（初期盤面・昇格段・反則上限・千日手手数・try rule）以外は
  初回リクエストで400拒否し（`game` は初回にしか来ない）、
  履歴から再構成した自駒局面に矛盾があれば着手せず400を返す。
  キャッシュ済みなら新しいplyぶんだけ増分で読み進める。キャッシュを失った
  （プロセス再起動直後・老朽化したセッションの掃除後）場合、
  `TSUITATE_WEBHOOK_SESSION_DIR` が設定されていればディスクの観測イベント列から
  復元し、無ければ以後の差分リクエストは 409 `history_gap` になる
  （リクエストに0手目からの全履歴が含まれる場合のみ従来どおり作り直す）。
  老朽化したセッションはリクエストのたびに掃除する（TTL 2時間、永続化ファイルも同じ）
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
| `TSUITATE_COLD_START_PREWARM_MS` | `2500` | 再起動後の履歴prewarmに使う上限。残りの履歴は通常updateで処理する |
| `TSUITATE_WEBHOOK_SESSION_DIR` | 未設定（無効） | 対局ごとの観測イベント列を `<dir>/<gameId>.jsonl` へ追記し、プロセス再起動・セッションTTL掃除の後もそこから復元する。**差分プロトコルではリクエストから全履歴を再構成できないため、本番運用では設定を強く推奨**（未設定だと再起動後、進行中の対局は 409 で継続不能になる） |
| `TSUITATE_WEBHOOK_LOG_DIR` | 未設定（無効） | 設定すると検証済みリクエストの生payload・応答・所要時間を `<dir>/<gameId>.jsonl` に1行1リクエストで追記する（本体の `TSUITATE_RECORD_DIR` と同じ思想。実戦での「弱く感じる」挙動を後から再現・分析するための診断用） |

## デプロイ手順（既存VPS、`tsuitate/scripts/server/setup/07-bot.sh` と同じ思想）

tsuitate-bot本体の運営bot（`tsuitate-bot.service`）とは別サービスとして、
既存VPS（systemd常駐＋Caddy自動HTTPS）に相乗りする。AWSは不要。

1. ローカルで `openssl rand -hex 16` してランダムなwebhookパスを決める
   （以下 `<path>` と書いた箇所に使う）
2. VPSにSSHし、`tsuitate` ユーザーに切り替えてビルドする（`root`でSSHしている
   場合、`su -` の `-` を忘れるとログインシェルにならずPATH等が引き継がれない
   ので注意）:

   ```bash
   su - tsuitate
   # ~/tsuitate-bot が無ければ（本体bot用に既にクローン済みのはず）:
   #   git clone https://github.com/tempakyousuke/tsuitate-bot.git ~/tsuitate-bot
   cd ~/tsuitate-bot
   git pull   # webhook_bot 関連のコードを取り込む
   cargo build --release --bin webhook_bot
   ```

3. env file とsystemdユニットの2つのファイルを作って起動する。
   仮のSecretを使うのは、この時点ではまだサイト側に登録しておらず
   本物のSecretが発行されていないため（登録後にstep 6で本物へ差し替える）:

   ```bash
   # --- 設定値（Secret・戦略名など）を env file に書く ---
   sudo -u tsuitate tee /home/tsuitate/tsuitate-webhook-bot.env > /dev/null <<'EOF'
   TSUITATE_WEBHOOK_SECRET=temporary-secret-before-registration
   TSUITATE_WEBHOOK_PATH=/webhook/<path>
   TSUITATE_WEBHOOK_STRATEGY=estimator_v10
   TSUITATE_THINK_BUDGET_MS=2000
   TSUITATE_COLD_START_PREWARM_MS=2500
   TSUITATE_WEBHOOK_SESSION_DIR=/home/tsuitate/webhook-sessions
   TSUITATE_WEBHOOK_LOG_DIR=/home/tsuitate/webhook-logs
   EOF
   sudo chmod 600 /home/tsuitate/tsuitate-webhook-bot.env

   # --- systemd のサービス定義を作る（07-bot.sh の tsuitate-bot.service と同パターン） ---
   sudo tee /etc/systemd/system/tsuitate-webhook-bot.service > /dev/null <<'EOF'
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
   EOF

   # --- systemd に認識させて起動 ---
   sudo systemctl daemon-reload
   sudo systemctl enable --now tsuitate-webhook-bot
   sudo systemctl status tsuitate-webhook-bot   # active (running) になっていればOK
   ```

   この時点では `TSUITATE_WEBHOOK_SECRET` が仮の値なので、外から本物の
   webhookが来てもHMAC検証に失敗して401を返す（＝まだ機能はしないが、
   プロセスとしては正常に起動している状態）。

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
6. env fileの `TSUITATE_WEBHOOK_SECRET` を本物のSecretに書き換えて再起動する:

   ```bash
   sudo -u tsuitate sed -i 's/^TSUITATE_WEBHOOK_SECRET=.*/TSUITATE_WEBHOOK_SECRET=<フォームで表示されたSecret>/' \
     /home/tsuitate/tsuitate-webhook-bot.env
   sudo systemctl restart tsuitate-webhook-bot
   sudo systemctl status tsuitate-webhook-bot
   ```

## 別バージョンのbotを並行稼働させる（例: v10を動かしたままv20を追加）

「1プロセス = 1戦略 = 1エンドポイント」という設計なので、既存の稼働中プロセスに
触れずに別プロセスとして追加できる。コード変更は不要（対象の戦略が
`strategy::make()` に登録済みであれば）で、以下をv10とは別名義で用意するだけ:

- 別のbind先ポート（例 `TSUITATE_WEBHOOK_BIND=127.0.0.1:8788`）
- 別のwebhookパス（`openssl rand -hex 16` で生成し直す）
- `TSUITATE_WEBHOOK_STRATEGY=estimator_v20`
- 別のsystemdユニット（`tsuitate-webhook-bot-v20.service`）＋別のenv file
- 既存Caddyfileに新しいパス→新しいポートへの `handle` ブロックを1つ追記
  （既存の `/webhook/<v10のpath>` ブロックはそのまま残す）
- サイトの「Bot作成」フォームで別のBot名（例 `:EstimatorV20`）として新規登録し、
  別のWebhook Secretを取得

v10・v20は別プロセス・別ポート・別Secretで完全に独立するため、片方の再起動や
停止がもう片方に影響することはない。

## 既知の制約

- **プロセス再起動後の進行中対局は `TSUITATE_WEBHOOK_SESSION_DIR` が無いと
  継続できない**（差分プロトコルではリクエストに過去の履歴が含まれないため、
  409 `history_gap` を返し続けることになる。dispatcher側に全履歴の再送
  フォールバックは無い）。設定してあれば、ディスクの観測イベント列から
  復元して0手目からのフルreplayを行う。
  当初は一括updateで粒子が完全枯渇するリスクがあったため、`bin/scenario.rs`と
  同じ「自分の手番ごとに逐次prewarmする」パターン（`strategy::prewarm_strategy_with_budget`）
  をコールドスタート経路に追加済み。HTTPではprewarmに既定2.5秒の上限を設け、
  残りの履歴は通常updateで処理する（80plyの合成履歴で約7.3秒）。
  合成した80ply（≒40手ずつ）の履歴で estimator_v10 のコールドスタートreplayを
  実測したもの（`webhook_session::tests::
  long_synthetic_history_replays_cold_start_with_estimator_v10_within_deadline`、
  `cargo test --release -- --ignored` で再実行できる）で、対局が長くなるほど
  この時間は伸びる。登録する「レスポンス時間」はデフォルトの5000msではなく
  10000ms程度に余裕を持たせることを推奨（`TSUITATE_THINK_BUDGET_MS`を下げれば
  さらに縮められる）。`deadlineMs`（リクエストで渡ってくる値）は現状コード側で
  参照していない（差分化後の型定義からは消えており、欠落も許容する）ため、
  内部で早期打ち切りはしない
- `sfen` フィールドを使った `GameModel::diff_view` 相当の整合性チェックは
  実装していない（観測ログ経路のみで自己完結）。ただし履歴モデル自身の矛盾は
  400で拒否する。SFENとの照合が必要なズレの実例が出たら追加検討する
- ゲームID使い回し等で色が食い違った場合はセッションを作り直す（`SessionStore::session_for`）

## 実戦の挙動を後から調べる（`TSUITATE_WEBHOOK_LOG_DIR`）

「弱く感じる」等、実戦での挙動を後から検証したい場合は `TSUITATE_WEBHOOK_LOG_DIR`
を設定しておく。gameIdごとに `<dir>/<gameId>.jsonl` へ、検証済みリクエストごとに
1行（生payload全文・返した手・所要時間ms）が追記される:

```bash
# 該当対局の各手を時系列で
jq -r '"\(.ply)\t\(.elapsed_ms)ms\t\(.request.positions[(.ply|tostring)].fouls)\t\(.response)"' \
  /home/tsuitate/webhook-logs/<gameId>.jsonl

# 特定plyの生リクエストだけ詳しく見る
jq 'select(.ply == 12)' /home/tsuitate/webhook-logs/<gameId>.jsonl
```

gameIdはサイトの対局URL等から控えておく。コールドスタート（プロセス再起動直後の
フルreplay）が疑わしい場合は `elapsed_ms` が跳ねている行を探すとよい
（`journalctl -u tsuitate-webhook-bot` の再起動ログと突き合わせる）。

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

2回目以降の差分リクエスト（`type`/`deadlineMs`/`game` なし・`basePly` あり・
positions は差分のみ）の形。ply 1 の `lastMove` には直前の応答で返した手を入れる:

```sh
BODY='{"requestId":"r2","gameId":"g1","color":"b","number":1,"ply":2,"basePly":0,"positions":{"1":{"sfen":"masked","lastMove":"+2726FU","lastInfo":0},"2":{"sfen":"masked","lastMove":"-0000ZZ","lastInfo":0}}}'
TS=$(date +%s)
SIG=$(printf '%s.%s' "$TS" "$BODY" | openssl dgst -sha256 -hmac testsecret | sed 's/^.* //')
curl -s -X POST "http://127.0.0.1:8799/webhook" \
  -H "Content-Type: application/json" \
  -H "X-Tsuitate-Timestamp: $TS" \
  -H "X-Tsuitate-Signature: sha256=$SIG" \
  --data "$BODY"
# => {"move":"+..."}（セッションが増分適用されて2手目を返す）
```
