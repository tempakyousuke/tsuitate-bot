# tsuitate-bot

王様のかくれんぼ（`~/Develop/tsuitate`）に外部bot APIで接続して対戦するRust製bot。
サイト・ソルバー（`~/Develop/tsuitate-resolver`）とは**意図的に独立**したプロジェクト（cargo依存もしない）。

## コマンド

- `cargo test` — ユニットテスト（候補手生成・プロトコル・エンジン・推定器）
- `cargo test --release -- --ignored` — 遅い検証（shogi.rs の perft depth 4/5）
- `cargo run --release --bin arena -- [対局数] [候補] [基準1] [基準2] ...` — 戦略同士の対戦。
  基準を複数並べるとガントレット（候補が各基準と対局数ずつ対戦）。
  戦略の変更は必ずこれで**全凍結版**（`src/frozen/` の `estimator_vN`）に有意に
  勝ち越すことを確認する。
  50%付近の信頼区間は 100局で±10pt / 200局で±7pt / 1000局で±3.1pt。当面（開発最初期）は
  100局を既定とし、結果が信頼区間内で判定できない僅差のときだけ局数を増やす。
  **実行はローカルでなく GitHub Actions で行う**（`.github/workflows/arena.yml`、手動起動のみ）:
  対象ブランチを push して
  `gh workflow run arena.yml --ref <ブランチ> -f games=100 -f candidate=estimator -f baselines="estimator_v2 estimator_v3 estimator_v4 estimator_v5"`。
  「基準 × シャード」の matrix に分割され（`-f shards=4` 既定。単一基準の
  200局も4ランナーに並列化される）、総合結果は **aggregate ジョブのサマリー**
  （および artifact `arena-combined`）に合算表で出る。シャード個別は
  `arena-result-<基準>-s<n>` / `arena-records-<基準>-s<n>`。
  `-f match_seed=<数>` で対局条件列を決定論化できる（アブレーション比較用。
  同じ入力なら版をまたいで同じ条件列。シャード間は自動で+shardずらし）。
  baselines の既定値は凍結版を追加したら手動で更新すること。
  **アリーナの時計は 1000秒+3秒**（本番サイトの300秒+3秒より厚い。思考予算を上げて
  強さの上限を探るため）。本番へのデプロイ時は `TSUITATE_THINK_BUDGET_MS` を絞って
  300秒+3秒に収める（強さと時間の調整ノブ）
- `cargo run --release --bin tune -- [反復数] [評価あたり対局数] [基準...]` — 評価パラメータ
  （`strategy::EvalParams`）のSPSA自動チューニング。目的関数はアリーナのスコア率
  （引き分け=0.5勝）。**f+/f− は共通乱数法でペアリングされる**: 同じ対局シード列
  （`TUNE_SEED` から決定論的に導出。定跡・推定器シード・タイブレークまで両陣営とも
  ペアになる）で評価し、評価順も反復ごとに入れ替える。境界クリップ時は実際に動いた
  距離を勾配の分母に使う。ログは `TUNE_LOG`（既定 `tune-log.jsonl`、gitignore済み）に
  追記し、中断後は同ファイルから自動再開（**start イベントの設定と不一致なら停止**。
  強行は `TUNE_FORCE_RESUME=1`）。eval イベントに対局内訳（勝敗・終局理由・max_plies・
  反則数・思考時間）が残るので、引き分け化や時間浪費でスコアが上がる変質を監視できる。
  ログの見方は `tuning/README.md`。
  `TUNE_CANDIDATE_LINE=<定跡名>` で候補側の定跡を固定できる（定跡特化チューニング。
  基準側の固定は `estimator_rush` を基準に指定）。完走時に最終中心点を追加評価して
  `done.final_score` に記録する。採用するときは `EvalParams::default` を書き換えて
  フルガントレットで確認する。
  対局ループは `selfplay.rs`（arena と共用）。**長時間ランはローカルでなくGCEで回す**（下記）
- `cargo run --release --bin scenario -- <名前|suite>` — 実戦棋譜の局面再現実験。
  `scenarios/*.kif`（Shogi Quest エクスポート + `*scenario ply=N` 行）を再生して
  特定局面での選択・粒子の信念（diag）・終局までの遂行（continue）を測る。
  追加手順は `scenarios/README.md`
- `cargo run --release --bin analyze -- records/*.jsonl` — 対局記録の事後分析。
  アリーナも `ARENA_RECORD_DIR` を設定すると候補(A)視点の記録を同形式で出力する
  （CIでは常時有効で artifact `arena-records` に上がる。真実の全手順つきなので
  そのまま analyze にかけられる）。
  game:end の全公開棋譜をリプレイし、反則の原因分類（王手解消失敗/飛び込み/経路封鎖/打ちマス）・
  駒得収支・只取られ・損な交換・取り返し逃し・詰み逃しを集計する
- `cargo build` / `cargo run --release` — 実行には環境変数が必要:
  - `TSUITATE_BOT_TOKEN`（必須）: サイトのマイページ「bot管理」で発行する `tsb_...` トークン
  - `TSUITATE_URL`（既定 `http://localhost:5173`）
  - `TSUITATE_THINK_MS`（既定 600）: 着手前の待ち時間
  - `TSUITATE_THINK_BUDGET_MS`（既定 2000）: estimator の1手あたり思考予算。
    粒子数（目標512×scale）・評価粒子数（192×scale）・2手読みの幅（上位8×scale/
    粒子48×scale）・リプレイ予算（500/900ms×scale）が scale = 予算÷900ms に比例する。
    アリーナ（1000秒+3秒）では既定のまま、本番（300秒+3秒）へは 900 前後に絞って
    デプロイする（強さの調整ノブでもある）
  - `TSUITATE_STRATEGY`（既定 `estimator`。旧来の単純botは `heuristic`）
  - `TSUITATE_QUEUE_RETRY_MS`（既定 60000）: キュー参加拒否・受付終了後の再試行間隔
  - `TSUITATE_RECORD_DIR`（既定 `records`。空文字で無効）: 対局記録（JSONL）の出力先。
    1対局1ファイルで、botの観測イベント全量・選択した手と思考時間・終局結果を記録する
    （`src/record.rs`）。相手の実際の手は含まれない。ローカルdevサーバー対局なら
    サーバーDBの `games.moves` に全手順（真実）が残るので、分析にはDBダンプと突き合わせる

## アーキテクチャ

コールバック（Socket.IOスレッド）→ mpsc チャネル → 単一メインループ、の一方向。
状態（対局ID・反則済みの手・観測履歴）はメインループだけが触る。

- `protocol.rs` — サイト側イベント契約の serde 版。**真実は tsuitate リポジトリの
  `src/lib/shared/events.ts` / `game-types.ts`**。サイト側の契約が変わったらここを追随させる
- `board.rs` — 「自分の駒だけを考慮した」候補手生成。tsuitate の `src/lib/shared/move-hints.ts` の移植。
  実際の合法性はサーバーだけが判定する（相手の駒は見えない）
- `shogi.rs` — フル盤面（両者可視）の通常将棋ルールエンジン。サーバーの judge.ts
  （shogiops の isLegal）と同じ合法性基準。初期局面 perft(1..5) で検証済み。
  アリーナの審判と推定器の局面シミュレーションの共通部品
- `observation.rs` — 観測履歴。ついたて将棋で得られる情報はこれが全量:
  自分の手の受理/反則（理由は不明）・取った駒種・自駒が取られたマス・王手/反則宣言
- `model.rs` — 観測履歴だけから自分側（自駒配置・持ち駒・相手手数・取られた駒）を
  再構成する GameModel。client.rs が sync の PlayerView と照合してズレを警告する
- `estimator.rs` — 相手局面のパーティクルフィルタ（determinization）。粒子=具体的なフル局面。
  観測と矛盾した粒子は棄却、相手手は観測（取られたマス・王手宣言の有無）と整合する
  合法手を弱い事前分布つきでサンプル。枯渇したら制約列をリプレイして再生成（時間予算つき）。
  厳密整合の生存粒子が target/4 を下回ると**ソフト救済**（POMCP の particle
  reinvigoration 相当）: 情報系の制約（王手宣言・反則の説明）だけ緩和して penalty+1 で
  生かし、評価側は重み 0.5^penalty で薄く数える。物理制約（合法性・駒種・取られたマス）は
  緩和しない。`predict_opp_reply` は観測フィルタなしの応手予測（2手読み用）。
  粒子数・リプレイ予算は思考予算スケール（`with_scale`）に比例
- `strategy.rs` — `Strategy` trait。`heuristic`（前進＋乱数の旧実装）と
  `estimator`（粒子加重平均で候補手を評価: 駒得期待値・反則確率×急峻な反則コスト・
  取られリスク・王周辺の利き圧力・王手/詰みボーナス・駒探し/王探しの情報利得・
  利き被覆・と金ポテンシャル）。
  粒子は複製で偏るので指紋でユニーク化し、経路上の未知マス数による事前確率とブレンドする。
  **2手読み**: 1手読みの上位候補だけ、粒子上で相手応手を `predict_opp_reply` から
  サンプルして静的リスク項の70%を実測の期待損失（露見度スケール×駒損−取り返し補償・
  被王手/被詰みペナルティ）に置き換える。
  駒交換の価値は `exchange_value` =（盤上価値+持ち駒価値）÷2（と金の反動は歩1枚ぶんに近い）。
  終盤は `endgame_push`（手数×素材リード）で攻め項を増幅して膠着を破る（劣勢時は掛けない）。
  粒子数・読み幅は `SearchBudget`（`TSUITATE_THINK_BUDGET_MS` 由来）に比例
- `frozen/` — アリーナ比較の基準となる凍結版戦略（`estimator_v2` = 王手回避修正、
  `estimator_v3` = リプレイ再生成の限定バックトラック（いずれも 2026-07-06 凍結）、
  `estimator_v4` = 評価関数の改善: 取られリスクの情報非対称・攻撃圧力・王手の
  反則数スケール・手戻り減点・評価粒子数192（2026-07-08 凍結）、
  `estimator_v5` = 王手ソルバー・評価式のmin修正・相手手事前分布の対人最尤推定・
  思考予算増額・駒探し項・定跡ブック（2026-07-10 凍結。200局×3で確定、
  vs v4 70.5%±7.9%）、
  `estimator_v6` = ソフト粒子（reinvigoration）・2手読み（応手サンプル・gain再構築）・
  交換価値是正・利き被覆/と金/王探し項・アンチドロー・思考予算スケール・
  SPSA第2ラウンド収束点（2026-07-14 凍結。200局×4基準で確定、
  vs v5 66.3%±7.1%。シード注入 with_seed 対応））。
  凍結後は編集しない。改善が確定したらその時点のコピーを `estimator_vN` として追加登録する。
  明らかに弱くなった古い凍結版は破棄してよい（v1 は王手放置癖が強すぎたため破棄済み）
- `client.rs` — 接続と対局ループ。反則リトライ（同じ手を繰り返さない）、
  `pending_move_number` による二重着手ガード、再接続時の `game:sync` 復帰、終局後の自動再キュー。
  常駐運用対応: 受付時間外の `queue:join` 拒否と `queue:closed` は `TSUITATE_QUEUE_RETRY_MS`
  間隔で再試行して開場を待ち、サーバー再起動で対局が消えた場合（sync が state=null）は
  キューへ戻る。本番（VPS）では systemd サービス `tsuitate-bot` として常駐
  （設置は tsuitate リポジトリの `scripts/server/setup/07-bot.sh`、更新は
  `npm run deploy -- --bot`）

## SPSAチューニング（GCE）

長時間のチューニングはローカルを熱くせず GCE の専用VMで回す（gcloud 認証済み前提）。

- **VM**: `tsuitate-tune`（プロジェクト `tsuitate-solver` / `asia-northeast1-b` /
  c2d-highcpu-16 **Spot**、約$0.1〜0.2/時）。使わないときは**停止**する（ディスクは残り課金は
  ほぼゼロ。次回は start するだけ）:
  `gcloud compute instances start|stop tsuitate-tune --project tsuitate-solver --zone asia-northeast1-b`
- **コード転送**（VMにgit認証は無いのでtarで送る）:
  `tar czf /tmp/tsuitate-bot.tar.gz -C ~/Develop --exclude tsuitate-bot/target --exclude tsuitate-bot/.git --exclude tsuitate-bot/records tsuitate-bot`
  → `gcloud compute scp` で `/tmp/` へ、`scripts/gce/setup-tune.sh` も一緒に送って実行
  （ビルド＋systemd常駐まで自動。引数と例はスクリプト冒頭参照）
- **並列度**: 単発ランは `ARENA_THREADS=14`、2実験並走は 7 ずつ
- **Spot停止への耐性**: 停止されたら `instances start` するだけ（systemd が tune を再起動し、
  `TUNE_LOG` から続きを自動再開）。在庫切れ（resources エラー）は時間をおいて再試行。
  監視するときは「復帰成功時のみ通知」の形にする
- **進捗確認**: `gcloud compute ssh ... --command "journalctl -u <サービス名> --no-pager | tail"`
- **回収**: 完走後（`最終パラメータ` 出力後は systemd が再起動ループになるので）
  `systemctl disable --now <サービス名>` → `gcloud compute scp` で `tune-*.jsonl` を
  `tuning/` へ回収 → VM停止。**採用判定は必ずCIの200局ガントレット**（100局は偽陽性事例あり）

## ルール上の前提（サイト側仕様）

- 反則しても手番は変わらない。ack が `reason: "foul"` なら別の手を指し直す。累計10回で反則負け
- 時計はフィッシャー 300秒+3秒。思考が遅いと時間切れ負け
- 同時1対局のみ。bot同士・所有者とはマッチしない
- 接続方法の一次資料は tsuitate リポジトリの `docs/bot-api.md`

## 強さの検証（アリーナ）

- `bin/arena.rs` がサーバーと同じ裁定（反則で手番維持・10回で反則負け・王手宣言を両者へ・
  詰み/ステイルメイト終局）をローカル再現し、戦略同士を対戦させる
- 各戦略に渡るのは PlayerView 相当と観測イベントのみ。**フル盤面は審判しか見ない**
  （observation.rs にない情報を使わない、という公平性の担保はこの構造で守る）
- 同一戦略同士は約50%になる（1000局で確認済み）。参考値: estimator vs heuristic は
  200局で勝率86.5%±4.7%、平均反則 2.2 vs 9.0（2026-07 時点）
- **比較の基準は heuristic ではなく凍結版**（`estimator_v2` 等）。heuristic への勝率は
  飽和していて改善の検出力がない。また非推移性（v2 に勝つが v1 に負ける）を検出するため、
  ガントレットで**全凍結版**に勝ち越すことを合格条件とする
- フィッシャー時計 1000秒+3秒 をシミュレートし時間切れは負け（本番サイトは300秒+3秒。
  デプロイ時は `TSUITATE_THINK_BUDGET_MS` を絞って収める）。`choose()` の壁時計を消費し、
  加算は受理された手の後のみ。思考時間の統計（平均/p99/最大）も出力するので、
  「遅くなったが勝率が上がった」偽の改善はここで検出する。戦略側は粒子数・
  リプレイ回数・時間予算で自ら打ち切る構造を保つこと（上限は思考予算 ≒ 既定2秒）

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
