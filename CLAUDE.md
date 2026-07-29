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
  `gh workflow run arena.yml --ref <ブランチ> -f games=100 -f candidate=estimator -f baselines="estimator_v6 estimator_v7"`。
  「基準 × シャード」の matrix に分割され（`-f shards=4` 既定。単一基準の
  200局も4ランナーに並列化される）、総合結果は **aggregate ジョブのサマリー**
  （および artifact `arena-combined`）に合算表で出る。シャード個別は
  `arena-result-<基準>-s<n>` / `arena-records-<基準>-s<n>`。
  `-f match_seed=<数>` で対局条件列を決定論化できる（アブレーション比較用。
  同じ入力なら版をまたいで同じ条件列。シャード間は自動で+shardずらし）。
  baselines の既定値は凍結版を追加したら手動で更新すること。
  **アリーナの時計は既定 1000秒+3秒**（本番サイトの300秒+3秒より厚い）。
  `-f clock="300+3"` で本番相当に切り替えられる（審判側の設定。
  `ARENA_FISCHER_INITIAL_MS` / `_INCREMENT_MS`）。
  **本番でも思考予算は既定の 2000ms のまま**でよい（2026-07-26 の実測で確定）:
  300秒+3秒・100局で**時間切れ0・クロック消費13.9%・残り最小は初期値の300秒**
  （加算3秒 > 平均消費1.2秒なので時計は減らない）。
  かつては 900ms 前後へ絞ってデプロイしていたが、**900ms は 2000ms より
  −14.5pt**（40.8% vs 55.3%、各100局 vs v11）なので絞ってはいけない。
  逆に予算を増やしても 8000ms で 55.8% と**2000msで飽和**しており、
  思考予算はもう「強さの調整ノブ」ではない
  （docs/improvement-plan-2026-07-26-yaneuraou.md 項目1）。
  `-f oracle=nofoul|check_nofoul` は診断用オラクル（候補側の反則を審判が握りつぶして
  指し直させる = 合法性知識の上限測定。記録は自動無効化）。
  `-f env="TSUITATE_MATE_RISK_W=3 TSUITATE_MATE_THREAT_W=4"` で評価ノブを渡せる
  （w スイープ用。`TSUITATE_*` のみ許可）。**env はプロセス全体に効くので
  「候補側だけ」になるのは、その凍結版が名前を知らないノブに限る**。
  凍結版は**凍結時点で読んでいた env を今も読む**（実測 2026-07-26）:
  - `TSUITATE_THINK_BUDGET_MS` — **v6〜v11 の全凍結版が読む**。思考予算の
    スイープを `-f env=` でやると両側の予算が動いて比較にならない。候補側だけ
    予算を変えたい場合は現行 `strategy.rs` だけが知る新しい名前を足すこと
  - `TSUITATE_JOSEKI` / `TSUITATE_EPS_PHYS` / `TSUITATE_DISABLE_DEFEND_GUIDE` /
    `TSUITATE_ENABLE_HANG_RISK` / `TSUITATE_FILTER_DEBUG` / `TSUITATE_DEBUG_CHECK`
    — v7 以降（版により差あり）
  - `TSUITATE_VALUE_NN_W` / `TSUITATE_CAPTURE_BET_VAR_W` / `TSUITATE_CHECKER_REMOVAL_W`
    — v11 のみ。**v11 を基準にした w スイープでは基準側も反応する**
  - `TSUITATE_MATE_*_W` / `TSUITATE_KING_HOLE_W` / `TSUITATE_TAINT_KING_FIX` は
    どの凍結版も読まない（＝候補側だけに効く）

  各シャードは `bin/analyze` も走らせ、**被詰めろオラクル**（相手に
  1手詰めを与えた局面数）・只取られ・反則内訳をジョブサマリーへ出す。
  実測 2026-07-16: vs v6 で nofoul 86.2%±4.8 / check_nofoul 59.5%±7.1 —
  反則経済に36ptの伸びしろが実在する（現状の評価構造のスカラー係数調整では
  届かないことが tune-round3/4 の不発で確定。tuning/README.md 参照）。
  C-7凍結後（2026-07-19）に再測定しても vs v7 で通常51.9%→nofoul 83.2%と
  約31ptのギャップが温存されており、反則マスを記憶して割り引く4種の対策
  （粒子ガイド2種・prior_legal/最終p_legal直接制約2種）はいずれも
  有意な改善に届かなかった（docs/c8-direct-synthesis.md 参照）。この
  ギャップは今のところ埋まっていない
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
  基準側の固定は `estimator_rush` を基準に指定）。
  `TUNE_PARAMS=<名前,...>` で調整対象をマスク（他は中心点固定）、
  `TUNE_SPAN=<0..1>` で有効範囲を既定値近傍へ局所化できる（中心が範囲端に近い
  項目の片側クリップ対策）。完走時に最終中心点を追加評価して
  `done.final_score` に記録する。採用するときは `EvalParams::default` を書き換えて
  フルガントレットで確認する。
  対局ループは `selfplay.rs`（arena と共用）。**長時間ランはローカルでなくGCEで回す**（下記）
- **粒子尤度のフィット**（`.github/workflows/fit.yml`、CIのみで実行する）:
  `gh workflow run fit.yml -f run_ids="<arena実行のrun ID...>" -f max_games=600`。
  過去アリーナの `arena-records`（観測列＋真実）を教師に、粒子群の中で真の局面を
  判別する条件付きMLE（`bin/fit_particles`）を回し、係数を `src/likelihood.rs` の
  `FITTED_THETA` へ手で反映する。評価側は `stratified_sample` が exp(θ·φ) を
  粒子重みに乗じる（fit_opp の局面版。2026-07-16 のフィットで実効候補数 59→33）
- `cargo run --release --bin scenario -- <名前|suite>` — 実戦棋譜の局面再現実験。
  `scenarios/*.kif`（Shogi Quest エクスポート + `*scenario ply=N` 行）を再生して
  特定局面での選択・粒子の信念（diag）・終局までの遂行（continue）を測る。
  追加手順は `scenarios/README.md`。
  **suite はローカル直列だと30分超かかるので CI で並列実行できる**
  （`.github/workflows/scenario.yml`、手動起動のみ）:
  `gh workflow run scenario.yml --ref <ブランチ> -f trials=20`。
  1シナリオ=1ランナーの matrix で壁時計は最遅1件分。
  `-f scenarios="kakutori keima"` で対象を絞れる。
  `-f env="TSUITATE_XXX=値"` で評価ノブを渡せる（arena.yml と同じ規約。
  既定0で入れた項の回帰確認に要る）。総合表は aggregate ジョブの
  サマリー（suite と同形式）。注意: 試行はシード同一でも壁時計ベースの
  予算スケールで揺れるため、10試行の±2〜3件差はノイズ。版比較は20試行×両版で
- `cd scenario-gui` → `npm run tauri dev` — シナリオデバッグ GUI（Tauri）。
  `.kif` の取り込み・任意 ply までの再生・先後視点切り替え・候補手分析
  （seed集計=全エンジン対応で回帰比較可 / ランキング=現行 estimator の
  スコア内訳 / **玉位置ビリーフ**=手番側の粒子が「相手玉はどこにいると
  思っているか」を盤に % で重ねる。赤枠=真実、厳密粒子のみ↔taint込みを切替可）。
  思考予算も指定可（900ms=本番相当）。
  リプレイ・選択試行の本体は `src/scenario_core.rs`
  （bin/scenario.rs と共有）。ランキングは `Strategy::last_ranking`
  （現行 estimator のみ実装。凍結版は編集しないため seed 集計のみ）。
  玉位置ビリーフは `scenario_core::king_belief`（推定器の構築 `build_estimator` と
  評価重み `weighted_unique_particles` は `bin/scenario ... diag` と共通。
  診断とGUIで重み規約が食い違うと較正の数字が無意味になるため）。
  粒子の構築は ply に比例して重い（実測: 83手の棋譜の50手目・seed10個・予算2000ms で
  **175秒**）ので、GUI は**推定器を手番側ごとにキャッシュして続きだけ食わせる**
  （`IncrementalEstimator`。`Estimator::update` が消化済みイベント数を持つので、
  作り直しと同じ結果になることをテストで担保）。実測 175秒 → 26秒（さらに seed 並列で
  1/4）。棋譜を読み直すと `clear_king_belief_cache` で捨てる
  **対局モード**（ヘッダの「対局」タブ）で GUI 内で直接対局できる。審判は
  selfplay.rs と同じ裁定（反則は手番維持でカウント・10回で反則負け・王手宣言）、
  時計はなし。パネル内に2つのサブモードがある:
  - **対人**（人間 vs bot）: 人間側は自駒のみ表示＋自駒だけ考慮の候補ハイライト
    （=実対局と同じ情報条件。「真実を表示」トグルあり）
  - **観戦**（bot vs bot）: 先後それぞれにエンジンと seed を指定し、「1手進める」
    または「自動再生」（間隔指定）で1局を眺める。観戦は隠す相手がいないので
    常に真実表示・視点切替つき。**arena の一括対戦とは用途が別**で、
    「1局を止めながら悪手を探す」ためのもの（気になる局面で止めて kif を
    書き出せば、そのままリプレイの候補手分析・玉位置ビリーフにかけられる）
  終局後（途中でも可）に `*illegal:` 行つき `.kif` を scenarios/ へ書き出して
  そのままリプレイ・`bin/scenario`・候補手分析へ流せる（スクラッチDBサーバーを
  立てる E2E 手順より手軽な実戦レビュー用。KIF 整形は `kifu::kif_body` =
  bin/make_scenario と共用）。対局セッションの本体は `src-tauri/src/play.rs`
  （対局者を `Player::{Human,Bot}` の2人組で持ち、**観測ログと反則試行メモは
  bot ごとに独立** = bot 同士でも互いの盤面は覗けない）
- `cargo run --release --bin analyze -- records/*.jsonl` — 対局記録の事後分析。
  アリーナも `ARENA_RECORD_DIR` を設定すると候補(A)視点の記録を同形式で出力する
  （CIでは常時有効で artifact `arena-records` に上がる。真実の全手順つきなので
  そのまま analyze にかけられる）。
  game:end の全公開棋譜をリプレイし、反則の原因分類（王手解消失敗/飛び込み/経路封鎖/打ちマス）・
  駒得収支・只取られ・損な交換・取り返し逃し・詰み逃しを集計する。
  **被詰めろオラクル**（2026-07-25）: 相手番の各局面で相手に一手詰めが存在したかを
  数える。相手が実際に詰みを見つけたかに依らないので、**相手が詰みを実行して
  こない環境（アリーナの凍結版同士は互いに玉位置が見えない）でも自玉の受けの
  改善を直接測れる**（nofoul オラクルと同じ趣旨）。実測の基準値: ローカルの
  対人記録83局で 232回/79局（うち打ち詰み166回・実際に詰まされた57回）。
  **王手の質の指標2種**（2026-07-29、ユーザーの指導に基づく設計）:
  ①「王手駒の即取られ」= 直接王手した駒を直後に取られた回数（弱い王手。
  実測ベースライン: 直接王手の53〜56%が即取られ・大半が取り返しなし）、
  ②「王手の強さ」= 王手直後の相手の合法解消手数 K・受け側の視界での選択肢 N
  （同じマスを別の駒で取る手は重複排除した「異なる確かめ」の数）・直後の実反則。
  実測: **K≤2 の王手は実反則 0.81〜1.17回/王手 vs K≥3 は 0.40〜0.46回（約2.4倍）**、
  N は平均79〜82手で N<10 はアリーナ800局で出現ゼロ（対人・終盤では効きうるので
  指標には残す）。王手の期待反則数は「手段クラス数 × 解消手の互いに素性 ÷ 解消手数」
  で決まる（詳細はメモリ strong-check-few-resolutions）
- `cargo run --release --bin belief_probe -- [--stride N] [--budget MS] [--seed S] <records/*.jsonl...>`
  — **粒子の信念の質を決定点ごとに測る**（NN段階②で作った診断。信念ネットとは
  独立に使える）。実戦と同じ順序で推定器を逐次 update し、マスごとの占有信念
  （`p_all` / `p_strict`）と一様事前を対数損失・Brier で並べる。
  厳密粒子ゼロの決定点の割合と、そこだけの信念の質も出る。
  **重い**（1局 ≒ 対局1局ぶんの思考）ので少数の記録に対して使うこと。
  出力CSVには `belief_features.rs` の特徴量も入るので、そのまま
  tsuitate-nn の `analyze_belief.py` でネットと突き合わせられる。
  **対照も必ず複数 seed 取ること**: seed だけで対数損失が 0.4380〜0.4796、
  ブラインド率が 11.8〜13.7% 動く（2026-07-27 実測。1本の対照と比べて
  「全条件が悪化」と誤判定した事故がある）
- `cargo run --release --bin king_probe -- [--stride N] [--budget MS] [--seed S] <records/*.jsonl...>`
  — **相手玉の位置ビリーフの質**を決定点ごとに測る（belief_probe の玉位置特化版。
  占有信念は全駒種で薄まるので、玉位置系の施策はこちらで測る）。真の相手玉マスへの
  信念質量と top1 一致率を、全決定点とブラインド（厳密粒子ゼロ）決定点で集計。
  belief_probe と同じく重い。**4局×3シードでは効果が出てもシード分散に埋もれる**
  （実測: ブラインドは1ランあたり17〜33決定点しかない）ので判定は 12局×5シード以上で
- `cargo run --release --bin export_belief_data -- [--stride N] <records/*.jsonl...>`
  — 信念ネットの学習データ（1行=1マス、両者ぶんの決定点）。学習は
  tsuitate-nn の `train_belief.py`、重み書き出しは `export_belief_weights.py`
- `cargo build` / `cargo run --release` — 実行には環境変数が必要:
  - `TSUITATE_BOT_TOKEN`（必須）: サイトのマイページ「bot管理」で発行する `tsb_...` トークン
  - `TSUITATE_URL`（既定 `http://localhost:5173`）
  - `TSUITATE_THINK_MS`（既定 600）: 着手前の待ち時間
  - `TSUITATE_THINK_BUDGET_MS`（既定 2000）: estimator の1手あたり思考予算。
    粒子数（目標512×scale）・評価粒子数（192×scale）・2手読みの幅（上位8×scale/
    粒子48×scale）・リプレイ予算（500/900ms×scale）が scale = 予算÷900ms に比例する。
    **アリーナも本番も既定 2000 のまま**でよい（2026-07-26 の実測。300秒+3秒で
    時間切れ0・消費13.9%・残り最小300秒）。900 へ絞ると −14.5pt、8000 へ増やしても
    +0.5pt（飽和）なので、**もう強さの調整ノブではない**。
    候補側だけ変えたいとき（版比較・スイープ）は `TSUITATE_CAND_THINK_BUDGET_MS`
    を使う（凍結版はこの名前を知らない）
  - `TSUITATE_LINK_WORK_W` / `_REF` / `TSUITATE_REPEAT_PENALTY_W` — 2026-07-28 に
    採用した項の切り戻し・スイープ用（既定は 1.0 / 2.0 / 0.3。0 にすると従来挙動）
  - `TSUITATE_CHECK_STRENGTH_W`（既定 0 = 無効）— **王手の強さ**（相手の合法解消手数 K）
    による値付け。g(K)=2.4/(1+K)−0.51 の中心化再配分で、強い王手（K≤2）を押し上げる。
    w=2 の実測（200局×3シード vs v12）: 強い王手 +14〜39%・詰み 148 vs 131 と機構は
    設計どおり動くが、**相手の反則総量は増えず勝率は中立**（51.5% vs 対照51.7%）—
    再配分だけでは反則総量が動かない（mate_threat/king_hole と同じ構図）ため既定0。
    SPSA対応（53個目）。王手の強さの理論と実測は CLAUDE.md 下記 analyze の節と
    メモリ strong-check-few-resolutions 参照
  - `TSUITATE_MUT_RESCUE`（既定 6、0 で無効 = 従来の force_apply のみ）— **変異救済**
    （2026-07-29 採用）: 完全全滅時の taint 粒子を「棄却粒子のコピー＋最小編集
    （相手駒の移設・持ち駒からの配置。駒は湧かせない）→ 通常の制約適用」で作る。
    王手駒・取り手が盤に実在する正規局面になり、相手玉は `deduce::opp_king_candidates`
    の健全な候補集合内へ補正される（project_taint_kings の生成時版）。実測:
    ブラインド決定点の玉位置 top1 一致 36.3%→44.4%（king_probe 12局×5シードのペア
    比較）、勝率は3シード×200局で中立（51.7% vs 対照51.1%）。凍結版はこの名前を
    知らないので `-f env=` は候補側にだけ効く
  - `TSUITATE_PLAN_W`（既定 0 = 無効）: **構想の読み**（自分の手 → 相手の応手 →
    自分の手）。`depth2_delta` の粒子ループに相乗りし、応手を3本サンプルして
    平均、3手目は駒得と取り返しだけ見る。**やねうら王が深く読めるのは1ノードの
    単価が3桁安いから**（NNUE の差分計算 vs 粒子集合上の評価）で、不完全情報では
    期待値を取るまで打ち切れず αβ カットも効かない ＝ 同じ深さを同じ値段では
    買えない。実測は中立（0.3 で 48.7%、0.6 で 51.0%、対照 51.5%）。
    応手を挟まない外付け版は −6pt で明確に負だった（＝楽観値をそのまま足すのは駄目）
  - `TSUITATE_EFFECT_OWN_W` / `_OPP_W`（既定 0 = 無効）: V2（玉距離重み付き利き）。
    実測は中立だが、**律速が玉位置ビリーフだと判明**した
    （docs/yaneuraou-lessons.md 1-2。打ち場所の序列化はできるのに、
    信念が古い玉位置を指していて攻め先が外れる）
  - `TSUITATE_BOARD_DISCOUNT_W`（既定 0 = 無効）: V5（盤上駒の減価）。
    w に単調悪化で**不採用**（記録は docs/yaneuraou-lessons.md 1-6）
  - `TSUITATE_BELIEF_*`（すべて既定 0 = 挙動不変）: 信念ネット（NN段階②）の
    供給チャネル。`_LIVE_W` / `_GUIDE_W` / `_SPAN` は粒子の提案分布側、
    `_GAIN_W` は厳密粒子ゼロの決定での gain 供給。**どれも単体では勝率中立**
    （実測は docs/nn-stage2-belief-net.md）。0 のときはネットを評価すらしない。
    凍結版はこれらの名前を知らないので `-f env=` は候補側にだけ効く
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
- `likelihood.rs` — 粒子の尤度モデル（アリーナ真実で教師あり学習した8特徴量の
  線形モデル）。評価側の粒子重みに exp(θ·φ) を乗じる。フィットは fit.yml/
  bin/fit_particles。C-7（尤度・ESSベースのフィルタ改修）の観測尤度の部品にもなる
- `belief_features.rs` / `belief_nn.rs` — **信念ネット（NN段階②）**。観測履歴だけから
  マスごとに「そこに相手の駒がいる確率」を予測する35特徴量 + pointwise MLP
  （35→64→32、手書き forward pass）。1決定点あたり81回しか呼ばないので安い。
  **既定では誰も呼ばない**（env ノブがすべて0）。
  詳細と全計測は docs/nn-stage2-belief-net.md。要点だけ:
  - 粒子より当たる（アリーナ・ホールドアウトで対数損失 0.6227 → 0.3777）。
    勝ちは**粒子が枯れる場所に集中**する（アリーナは決定点の 29.2% が厳密粒子ゼロで、
    そこの粒子は AUC 0.685 とほぼ無情報。ネットは 0.821）
  - **強さには変換できていない**。供給チャネルを2つ試して両方とも勝率は中立
    （提案分布 = `TSUITATE_BELIEF_LIVE_W` / `_GUIDE_W` / `_SPAN`、
    ブラインド時の gain = `TSUITATE_BELIEF_GAIN_W`）。
    後者は只取られ −7.4%・被詰めろ −20%・王手中の反則 −16% と機構は改善するが、
    打ちマス反則 +10% が相殺して反則/局・勝率は横ばい（king_hole_w と同じ構図）
  - **王手駒ビリーフ（kakutori の根本原因）は pointwise ネットでは埋まらない**。
    「今どの駒が王手しているか」は同時制約なので CheckSolver の領分のまま
  - 出力は**盤面の直接合成には使わない**（マスごとの独立確率からサンプルすると
    駒数・駒種多重集合・二歩・行き所・テンポの同時制約が壊れる）
- `truth_replay.rs` — 対局記録の真実（game:end の全手順＋反則試行）から
  **両者の観測列**を再構成して決定点を1つずつ渡す共通部品。観測の作り方
  （順序・move_number 規約・王手宣言の両者通知）は selfplay.rs の審判と一致させること
- `deduce.rs` — 観測から論理的に確定できる事実の演繹（C-8 の前段）。
  2方向ある: 「自玉の王手履歴で**相手駒の経路**を刈る」（`route_square_refuted_by_check_history`）と、
  その鏡像の「自分が掛けた王手宣言で**相手玉の居場所**を刈る」
  （`opp_king_candidates`、2026-07-25 追加）。後者が使う事実は全て自分側の完全既知情報:
  初期玉位置・王手宣言の有無・自駒の利き・「玉は1手で1マスまで」・「王手駒が隣接して
  いて取られてもいないなら玉は動くしかなかった」。
  **健全性（真実を絶対に落とさない）が命**なので利きを2種類に分ける:
  候補を**減らす**のは「確実な利き」（隣接・桂の跳び＝未知の駒に遮られ得ないもの）だけ、
  王手宣言で候補を**残す**条件には「その手で新たに利きが生じ得たマス」を使う。
  後者は集合の差分では計算しない（上界どうしの差は上界にならない）: 着手駒の利き・
  出て行ったマスを通る線が開く・取った駒のマスを通る線が開く、の3経路を構成的に足す。
  実測（`scenarios/archive/king-deduction.kif` 83手・168局面）: 健全性違反0、
  王手直後は 1〜7マスまで絞れる。`strategy.rs::project_taint_kings` と
  scenario-gui の玉位置ビリーフ表示が使う
- `mate.rs` — **持ち駒打ちに限定した一手詰めの検出**。玉のマスから8方向へ空きマスを
  辿って「そのマスに打てば王手になる (マス,駒種)」を逆引きし、詰み判定する。
  `drop_mate_threat` は成立度を2段階で返す: `Mate`（盤上の駒だけで詰み）と
  `IfSupported`（玉で打った駒を取る以外に受けがない = **打ちマスに支えが1枚
  あれば詰み**）。攻め側は自分の支え駒を正確に知っているので `Mate` を、受け側は
  相手の支え駒が不可視なので `IfSupported` も使う。全合法手からの一手詰め探索
  （`mate_in_1`、診断専用）はホットパスに載らないので打ちに限定している
  （実測: 対人83局の被詰めろ232件のうち72%が打ち詰み）。
  strategy.rs の `mate_threat_w`/`mate_risk_w` と bin/analyze の被詰めろ指標が使う
- `model.rs` — 観測履歴だけから自分側（自駒配置・持ち駒・相手手数・取られた駒）を
  再構成する GameModel。client.rs が sync の PlayerView と照合してズレを警告する
- `estimator.rs` — 相手局面のパーティクルフィルタ（determinization）。粒子=具体的なフル局面。
  観測と矛盾した粒子は棄却、相手手は観測（取られたマス・王手宣言の有無）と整合する
  合法手を弱い事前分布つきでサンプル。枯渇したら制約列をリプレイして再生成（時間予算つき）。
  厳密整合の生存粒子が target/4 を下回ると**ソフト救済**（POMCP の particle
  reinvigoration 相当）: 情報系の制約（王手宣言・反則の説明）だけ緩和して penalty+1 で
  生かし、評価側は重み 0.5^penalty で薄く数える。物理制約（合法性・駒種・取られたマス）は
  緩和しない。**taint 粒子は敵玉の位置まで嘘になる**ので、評価へ渡す前に
  `strategy.rs::project_taint_kings` が `deduce::opp_king_candidates` の候補集合へ
  玉を引き戻す（最も近い候補マスへ移動。相手駒が居れば入れ替え）。棄却でなく移動なのは
  玉位置の質量を潰さないため（ansatsu 回帰の教訓）。`TSUITATE_TAINT_KING_FIX=0` で無効化。
  `predict_opp_reply` は観測フィルタなしの応手予測（2手読み用）。
  粒子数・リプレイ予算は思考予算スケール（`with_scale`）に比例。
  相手手のサンプリング事前分布 `opp_move_weight` は v9 以降 NN
  （`opp_move_nn.rs`、駒種one-hot・from_home 等）。2026-07-25 に
  `en_prise_flee`（既知の当たりが付いた駒＝自分が駒を取ったマスの駒に
  狙われた駒を動かす手。king_flee の全駒種一般化）を加えて25特徴量。
  対象文脈の逃げ質量は教師実測にほぼ較正済み（41.7%に対し39.5%）だが、
  **教師（bot）自身が42%しか逃げないのが信念側の上限**で、人間的な
  「先回りの逃げ」（脅しが通る前に逃げる）は観測情報から原理的に
  モデル化できない。en_prise×from_home の交互作用列（26次元）は
  希少セルで逆に悪化して不採用（ブランチ opp-en-prise-flee の履歴参照）。
  相手が自分の駒位置を知る他の経路（相手自身の反則観測・王手宣言の逆推論・
  home駒の利き圏）は未カバー
- `check.rs` — 王手中の回避手選択のための制約推論（`CheckSolver`）。
  「自玉を攻撃しうる（マス,駒種）」を全列挙して仮説群とし、反則を観測する
  たびに「その仮説なら合法だったはず」の重みを減衰、粒子が健全なら
  実際の王手駒に投票させて鋭くする。`resolve_probability(mv)` = 仮説の
  重み付き割合で mv が王手を解消する確率。`strategy.rs` の `evaluate()`
  が `p_legal` の事前確率に使う。**弱点**: 仮説を単純平均するため、生存
  仮説が多い（＝粒子の王手駒ビリーフが誤っている）局面では、特定マスへの
  正しい捕獲でも確率が薄まってしまう（`scenarios/kakutori.kif` で発見:
  角の王手を無条件で捕獲できるのに0/20回しか選ばなかった）。対策の変遷:
  v8では `captures_checker`＋p_legal下限0.35（一律フロア）で捕獲選択率
  0/20→10-11/20としたが、フロアはシナリオ特化の一律税なので撤去し、
  `removal_term`（**仮説条件付きの王手駒除去期待値**）に置き換えた
  （2026-07-24、フロア撤去単体はアリーナ中立を確認済み）。受理を条件付けた
  仮説の事後分布で「王手駒のマスを玉以外の駒で取る手」に＋（交換価値＋
  回避される残存脅威）を与え、`strategy.rs::checker_removal_w`（既定1.0、
  SPSA対応、`TSUITATE_CHECKER_REMOVAL_W`で上書き可）でgainの内側に加算する。
  設計上の罠2つ: 対称形（王手駒を残す手に−脅威）は合法手全体が反則水位を
  割って反則爆発、玉での捕獲を加点すると隣接マスへの玉逃げ全部が仮説捕獲
  扱いになり相対差が消える（詳細は removal_term のdocコメント）。
  kakutori捕獲19/20・アリーナ中立（vs v10 52.9%±9.6、反則経済悪化なし）。
  挙動は「空振り捕獲プローブ→反則で仮説減衰→真の捕獲」の系列で人間の
  実戦手順と同型。根本原因はもう一つあり、王手駒についての粒子ビリーフ
  自体の較正不良（真の王手駒への信念1.7%等）は未着手。NN化ロードマップの
  段階①②（likelihood.rs/opp_move_weightのNN置換・信念ネット）が対象領域
  …**だったが、段階②の信念ネットでは埋まらないことが実測で判明した**
  （2026-07-27）: マスごとの独立確率を出す pointwise ネットは「今どの駒が
  王手しているか」という同時制約を表現できず、真の王手駒マスへの信念は
  粒子のほうが高い（0.55 vs 0.49）。ここは CheckSolver（制約推論）の領分のまま
- `strategy.rs` — `Strategy` trait。`heuristic`（前進＋乱数の旧実装）と
  `estimator`（粒子加重平均で候補手を評価: 駒得期待値・反則確率×急峻な反則コスト・
  取られリスク・王周辺の利き圧力・王手/詰みボーナス・駒探し/王探しの情報利得・
  利き被覆・と金ポテンシャル）。
  粒子は複製で偏るので指紋でユニーク化し、経路上の未知マス数による事前確率とブレンドする。
  **2手読み**: 1手読みの上位候補だけ、粒子上で相手応手を `predict_opp_reply` から
  サンプルして静的リスク項の70%を実測の期待損失（露見度スケール×駒損−取り返し補償・
  被王手/被詰みペナルティ）に置き換える。
  駒交換の価値は `exchange_value` =（盤上価値+持ち駒価値）÷2（と金の反動は歩1枚ぶんに近い）。
  **捕獲の賭け分散ペナルティ** `capture_bet_var_w`（既定2.5、2026-07-25、
  `TSUITATE_CAPTURE_BET_VAR_W` で上書き可）: p_hit(1−p_hit)×E[捕獲価値|hit] を
  gain から引く凹割引。占有が五分のマスへの大駒捕獲賭け（tokin-bet.kif 16手目
  「8八と」が発端）は空振り分岐の質を素の期待値が数えないことへの補正で、
  同じ1ビットを買うなら安い駒のプローブが浮く。**王手中は無効**（王手駒捕獲
  プローブは CheckSolver/removal_term の領分、kakutori 19/20 維持）。
  w=2 のほうがアリーナは強い（62.5% vs w=2.5 の51%、104局）が、「逃げ確率5割
  でも大駒賭けは沈むべき」の人間判断で 2.5 を採用（SPSA での再調整候補）。
  **詰めろ2項**（`mate.rs` 由来、2026-07-25、対人局 mate-net.kif が発端。
  既定はどちらも 0 = 従来と同一挙動。`TSUITATE_MATE_THREAT_W` /
  `TSUITATE_MATE_RISK_W` で上書き可、SPSA対応）:
  `mate_threat_w`（詰めろ生成 = この手の後に自分の打ち一手詰めが成立する確率）と
  `mate_risk_w`（被詰めろ = 相手の打ち一手詰めが残る確率。`IfSupported` は
  `MATE_RISK_IF_SUPPORTED=0.5` 倍）。**gain の内側**（p_legal 割引を受ける）だが
  `expected` の**外**に置く: 詰めろが決まる終盤ほど厳密粒子は枯れて
  `legal==0` になり expected が丸ごとゼロになるため。同じ理由で判定に使う
  粒子プール（`mate_pool`）は**厳密粒子が全滅していれば taint 粒子**に落とす
  （blind_king_attack / CheckSolver 投票と同じ規約。mate-net 57手目の実測で
  厳密0個・taint の玉位置信念は真実に71.6%）。王手中は両方とも無効。
  終盤は `endgame_push`（手数×素材リード）で攻め項を増幅して膠着を破る（劣勢時は掛けない）。
  `knight_bait_w`（桂馬の高跳び歩の餌食。敵桂馬の攻撃マスへ歩がBFS距離的に
  近づくほど加点。2026-07-19、人間レビュー指摘）は「安い駒で駒得を狙う」の
  唯一の駒種特化項。人間との対局レビューで「良い手を探す力が根本的に
  足りていない」という指摘も出ており（33手目5八四金: 位置が確定している
  敵駒の利きへ無防備に踏み込む手を20/20で選択）、個別の駒種特化項の
  積み増しでは埋まらない可能性がある。詳細は下記「NN方向」参照。
  `king_hole_w`（**自玉8近傍のうち「玉以外の自駒の利きが無いマス」1個あたりの減点**。
  既定 0 = 無効、`TSUITATE_KING_HOLE_W` で上書き可、SPSA対応。
  docs/yaneuraou-lessons.md の V4）は**自分の駒だけで決まる**ので粒子不要・
  ノイズゼロで測れる数少ない守りの指標。玉自身の利きは支えに数えない。
  実測 2026-07-26（600局、main 対照との比較）: 被詰めろは w に対して**単調に減る**
  （363回 → w=0.15 で339 → w=0.4 で300。うち打ち詰みは 276 → 208 → 186 = −33%）
  ので設計どおり効いているが、**`実際に詰まされた` は 75 → 69 → 72 で横ばい**、
  勝率も非単調（w=0.15 で v9 58.0/v10 59.5/v11 51.0、w=0.4 で 66.0/53.0/44.5）。
  理由は**相手botが自玉の位置を知らないので与えた詰めろを実行できない**こと。
  被詰めろオラクルの改善は**対人でのみ実害の減少に対応する**（en-prise 逃げが
  bot相手だと実被害2.7%だったのと同じ構図）ので、この指標だけで採否を決めないこと。
  既定0のままマージした（挙動不変・SPSAの対象・本番で試せる）。
  `link_work_w`（既定 **1.0**）/ `link_work_ref`（既定 **2.0**）= **紐を
  「守られる駒の働き」で重み付ける**（2026-07-28 採用）。素の交換価値だけで
  数えると「隅で何もしていないと金」に紐をつける手が、実戦的に価値のある紐と
  同じ重みになる（発端: watch-estimator-20260727-223458 の17手目 `L*1b`。
  自分のと金1一の直前に香を打ち、その香は最後まで動けない。と金と香が
  **相互に守り合う**ので単発の打ちで得られる紐の最大値になっていた）。
  働きは V2 の距離重み付き利き（`1/(1+d)`）を流用する。**可動性ではない**のが
  要点で、底歩は利き1マスでも自玉の隣なので働きを確保する（「動けない駒は
  価値なし」では底歩を取りこぼす、というユーザー指摘）。**王手中は適用しない**
  （v12 の実測で「紐は王手中もゲートしてはいけない」＝ゲートすると kakutori の
  反則が 7→51）。
  `repeat_penalty_w`（既定 **0.3**）= **自分側の配置が過去に出現した回数**に
  比例する減点（2026-07-28 採用）。既存の `backtrack_penalty`/`shuffle_penalty` は
  直前の1手しか見ず固定額なので、往復を繰り返しても減点が増えない
  （発端: watch-estimator-20260728-122107 の57〜67手目で 3四角↔2五角 を6手往復。
  手戻り減点 −0.369 に対しブラインド玉攻めが +1.8 で、角が動くたびに
  「信念上の敵玉マスへ利きを作る手」として加点され直していた）。
  自分側の配置は完全既知なので粒子不要・ノイズゼロ。取り合いが起きれば
  指紋が変わってリセットされる ＝「何も起きていないのに同じ形へ戻る」ときだけ効く。
  頻度は bot の着手の 2.9%（`bin/analyze` の「無意味な往復」）＝改善の天井。
  **この2件の実測**（200局×3シード vs v12）: 53.5 / 54.3 / 52.0%（対照 51.5 / 52.0）、
  反則/局 6.64→6.37、詰み 39→50。全凍結版へ勝ち越し（v6 76.8 / v7 69.0 /
  v8 67.3 / v9 68.0 / v10 56.5 / v11 56.5 / v12 52.0%）。
  ただし **vs v12 が 52.0%±6.9 で、過去の凍結時の相場（+7〜11pt）に届かないので
  v13 としては凍結していない**（凍結版を増やすと以後の改善判定が鈍るため）。
  粒子数・読み幅は `SearchBudget`（`TSUITATE_THINK_BUDGET_MS` 由来）に比例
- `frozen/` — アリーナ比較の基準となる凍結版戦略（
  `estimator_v6` = ソフト粒子（reinvigoration）・2手読み（応手サンプル・gain再構築）・
  交換価値是正・利き被覆/と金/王探し項・アンチドロー・思考予算スケール・
  SPSA第2ラウンド収束点（2026-07-14 凍結。200局×4基準で確定、
  vs v5 66.3%±7.1%。シード注入 with_seed 対応）、
  `estimator_v7` = C-7 連続重み粒子フィルタ P1+P2: logw統一・ESSリサンプリング・
  multiplicity畳み込み・窓付き若返り・制約後読みガイド・墓場復活
  （2026-07-18 凍結。設計と経緯は docs/c7-continuous-filter.md。200局×5基準で確定、
  vs v6 64.2%±6.8% / v5 83.8% / v4 82.7% / v3 88.0% / v2 89.3%）、
  `estimator_v8` = C-7 P3（ε_phys最後の砦・エポック正規化・ブラインド玉攻め勾配）・
  occupiesガイド・home_lance_move/knight_bait_w（相手手事前分布の駒種特化）・
  王手中の駒捕獲候補へのp_legal下限（`CheckSolver::captures_checker`、
  kakutori.kifの捕獲見逃しへの対応。今回の主変更）
  （2026-07-21 凍結。100局実測で vs v6 71.3%±8.8% / vs v7 62.5%±9.3%）、
  `estimator_v9` = NN段階①-a/①-b: opp_move_weight を23特徴量の1隠れ層MLP
  （手書きforward pass、`src/opp_move_nn.rs`）へ置き換え。①-a =
  bot自己対局データでのNN化＋反則回数特徴量 opp_foul_count_this_turn の
  ライブ配線、①-b = 駒種one-hot・成駒・移動距離・from_home（初期配置マス
  からの移動 = home_lance の全駒種一般化。別立て-1.3加点は廃止）。
  学習データは新定石1536局（定石更新は凍結版側の序盤分布も変えるため
  データ再生成が必要、が教訓）。kakutori 注目手2/10→10/10・
  真の王手駒信念1.0%→2.7%
  （2026-07-22 凍結。200局×3基準で確定、vs v6 78.0%±5.7 /
  vs v7 62.3%±6.7 / vs v8 57.0%±6.9 — 全凍結版に有意勝ち越し）、
  `estimator_v10` = NN段階③フェーズ2: 粒子上のvalueネットを `evaluate()` へ
  統合（state16+transition6 → 勝率相当、手書きforward pass `src/value_nn.rs`、
  `value_nn_w=6.0` で歩価値スケール化して gain 内へ加算 = p_legal割引の内側）。
  王手中（you_in_check）はNN項無効（王手回避プローブの反則増を実測して遮断、
  CheckSolverの領分）。学習は新定石1536局・pairwise補助loss w=20 m=0.1・
  4シード中 gold-check/kakudo 両正解の seed1。gold-check の悪手 17/20→1/20
  （2026-07-23 凍結。vs v6 77.0%±5.8 / vs v7 69.8%±6.4 /
  vs v8 56.9%±3.4（800局合算）/ vs v9 57.3%±6.9 — 全凍結版に有意勝ち越し）、
  `estimator_v11` = v10以降の小粒改善3件: 王手駒捕獲のp_legalフロアを撤去し、
  `checker_removal_w=1.0` の除去期待値項（仮説条件付きの王手駒除去EV）へ置換・
  `capture_bet_var_w=2.5`（捕獲賭け分散ペナルティ。p_hit(1−p_hit)×捕獲価値を
  gainから控除、王手中は無効）・opp_move NN `en_prise_flee`（25次元。既知当たり
  駒を動かす相手手特徴量）
  （2026-07-25 凍結。match_seed=20260725、各200局で vs v10 58.5%±6.8 /
  vs v9 63.0%±6.7 / vs v8 58.5%±6.8。反則・思考時間の悪化なし）、
  `estimator_v12` = **ブラインド取り返し**（`BLIND_RECAPTURE_W=1.0`）と
  **予防的な紐**（`link_w=0.06`）の2件:
  - ブラインド取り返しは「厳密粒子が全滅した決定（実測26%）では `expected` ごと
    駒得が消える」欠陥への対応。位置が確実に分かっている敵駒（直前に自駒を
    取られたマス）すら取らなくなっていた（実例 `scenarios/recap-dragon.kif`:
    タダの龍が gain 11〜13 → 0.045 で89位）。**taint 粒子は使わず観測だけ**を
    使うのが要点（taint 経由の一般フォールバックは別途200局で不採用）
  - 紐（docs/yaneuraou-lessons.md の V3）は自駒同士の連結が完全既知なので
    粒子不要。w スイープの最適は**やねうら王の比率の約10倍**（0.008→0.06）。
    0.25 以上では反則は減り続けるのに勝率が落ちる（守りが目的化して手数だけ
    伸びる）。**王手中もゲートしない**のが v10/v11 との差で、ゲートすると
    vs v10/v11 が約9pt落ちる（kakutori の反則が 7→51）
  （2026-07-27 凍結。match_seed=20260728、各200局で vs v11 61.5%±6.7 /
  vs v10 61.3%±6.8 / vs v9 60.3%±6.8。反則/局は減少、思考時間は横ばい）。
  凍結後は編集しない。改善が確定したら
  `python3 scripts/freeze_estimator.py <N> <日付> "<差分の要約>" > src/frozen/estimator_vN.rs`
  で生成し（estimator.rs/check.rs/strategy.rs を1ファイルへまとめ、テストと
  運用ノブの env を落とす）、`frozen/mod.rs`・`strategy::make`/`make_seeded`・
  `arena.yml` の baselines 既定値へ登録する。**生成後は
  `arena 20 estimator estimator_vN` が約50%になることを確認**すること
  （挙動が同一である担保）。
  明らかに弱くなった古い凍結版は破棄してよい（v1 は王手放置癖、v2〜v5 は v7 凍結時に
  全て80%超で上回ったため 2026-07-18 破棄。成績は git 履歴と tuning/README.md に残る）
- `client.rs` — 接続と対局ループ。反則リトライ（同じ手を繰り返さない）、
  `pending_move_number` による二重着手ガード、再接続時の `game:sync` 復帰、終局後の自動再キュー。
  常駐運用対応: 受付時間外の `queue:join` 拒否と `queue:closed` は `TSUITATE_QUEUE_RETRY_MS`
  間隔で再試行して開場を待ち、サーバー再起動で対局が消えた場合（sync が state=null）は
  キューへ戻る。本番（VPS）では systemd サービス `tsuitate-bot` として常駐
  （設置は tsuitate リポジトリの `scripts/server/setup/07-bot.sh`、更新は
  `npm run deploy -- --bot`）

## NN方向

**段階①（opp_move_weightのNN化）は本番統合済み**（estimator_v9、2026-07-22
凍結）。`src/opp_move_features.rs`（24特徴量の共有定義。学習エクスポートと
estimator推論の両方がここを参照）・`src/opp_move_nn.rs`（重み定数+手書き
forward pass。1手あたり最大10万回超のホットパスなのでONNX等は使わない）・
`bin/export_opp_move_data`（アリーナ記録→候補集合CSV）。学習は tsuitate-nn の
`train_opp_move.py`（候補集合内softmax条件付きNLL、`--feature-set base|piece`で
アブレーション）、重み書き出しは `export_opp_move_weights.py`。
再学習の手順: 記録用アリーナ（例 500局×3基準）→ artifact回収 →
`export_opp_move_data` → 学習（**4シードで頑健性確認**）→ export → cargo test
（numpyクロスチェックの期待値も再生成）→ `scenario suite`＋`kakutori diag` →
200局ガントレット。**定石（joseki.json）を更新したら学習データも再生成する
こと**（定石は実行時読み込みで凍結版＝教師側の序盤分布も変わる。①-bの
from_home等は序盤感度が高い）。NN出力のclamp(-15,15)は分布外入力への
安全弁なので外さない。
逆方向反則特徴量 my_foul_count_last_turn（24次元目、2026-07-23追加）は
現行のbot教師では寄与ゼロ（ゼロ化アブレーションでNLL差0.0002・
新1512局データの再学習も対v10互角400局）だが、人間対局データ等で
効く可能性を見て**当面残す**（ユーザー判断。「効かないから削る」提案は
しない）。同時に行った両NNの新データ再学習（opp_move seed0 /
value seed2）は強さ中立のままマージ済み（valueはオフライン較正が改善:
kakudo 正解シード1/4→3/4）。
**段階②（信念ネット）は 2026-07-27 に着手し、保留で決着**した:
オフライン較正は決定的に勝つ（粒子より当たる）が、供給チャネルを2つ試して
どちらも勝率中立。既定0のノブとして残してある（`belief_features.rs` の節と
docs/nn-stage2-belief-net.md）。**新しいパラメータと併用すれば効く可能性は
残る**ので捨てていない。段階④（フルRL）は未着手。

## NN方向 valueネット（段階③）: フェーズ2で本番統合済み（estimator_v10）

**フェーズ2（2026-07-22〜23、estimator_v10 凍結）で `strategy.rs::evaluate()` へ
統合済み**。粒子ごとに (state16+transition6) → 勝率相当[0,1] を
`src/value_nn.rs`（手書きforward pass 22→64→32→1、約0.6µs/回）で推論し、
重み付き平均の中心化値を `value_nn_w`（既定6.0、SPSA対応、
`TSUITATE_VALUE_NN_W` で上書き可＝切り戻しノブ。凍結版は反応しない）で
歩価値スケール化して gain 内（= combine_score の p_legal 割引の内側）へ加算。
設計上の要点:
- **王手中（you_in_check）はNN項無効**。w=6 で王手回避プローブの反則試行が
  増え dragon-check-drop で反則負け2/20が出た実測への対応。王手回避は
  CheckSolver（制約推論）の領分
- **重みは w 選定スイープで決める**（w=3 では gold-check の悪手 17/20 を
  変えられず、w=6 で 2/20。NNのスコア差 0.1〜0.2 × w が手作り項との綱引きに
  勝つ必要がある）
- state特徴量は候補間で共通なので粒子単位に遅延キャッシュ、粒子数は
  `nn_samples`（思考予算比例、基準48）で制限
- 再学習の手順は opp_move NN と同じ（記録用アリーナ → `export_value_data` +
  `export_pairwise_data` → tsuitate-nn/train.py を pairwise w=20 m=0.1 で
  **4シード** → gold-check/kakudo オフライン検証で両正解シードを選ぶ →
  `export_value_weights.py` → クロスチェック期待値再生成 → w スイープ →
  ガントレット）。**定石更新時はデータ再生成**（opp_move と同じ教訓）

限界の認識: valueネットは「明確な悪手の回避」（gold-check系）には効くが、
「無目的な安全手同士の序列」（54手目9二香のような横並び）はまだ判別しない。
以下はフェーズ1（データ基盤とオフライン学習、2026-07-20〜21）の記録。

## NN方向フェーズ1: valueネット（データ基盤とオフライン学習の記録）

`strategy.rs` の手作りヒューリスティックが「堅実な手 vs 派手だが危険な手」を
正しく評価できない事例（33手目5八四金、上記参照）を受け、以前見送っていた
NN化（4段階案の③「粒子上のvalueネット」）に2026-07-20着手した。
`src/value_features.rs`（真の局面のstate特徴量16 + 着手固有のtransition
特徴量6 = 22次元）・`bin/export_value_data`（対局記録→学習データCSV）・
`bin/export_pairwise_data`（同一局面内の手作り特徴量による優劣ペアを抽出、
補助hinge loss用）・`bin/eval_candidates`（候補手のオフライン評価、
`.kif`を直接読める）をtsuitate-bot側に実装し、学習パイプライン本体は
別リポジトリ `~/Develop/tsuitate-nn`（Python/PyTorch、まだリモート無し）に置く。

**現状（2026-07-21時点）: フェーズ1（データ基盤＋オフライン学習）は
一区切り、厳密な成功条件には一歩届かず**。既知シナリオ2件
（`scenarios/gold-check.kif`, `kakudo.kif`）のオフライン検証がgold-check
5/5・kakudo 4/5（5seed中）まで到達。経緯: 初回学習でオフライン検証「合格」
と一度報告したが、正規化統計の分割前計算・行単位train/val分割という
手法上の欠陥がcodexレビューで発覚し不合格に逆転（教訓: 良い結果が出た
ときこそ疑ってレビューすべき）。データ拡大（3000局）・過学習対策の後も
state特徴量だけでは候補手比較の信号がmax型特徴量に埋もれ、transition
特徴量を追加しても最終勝敗ラベルのみでは符号を学習できない
credit assignment問題に直面。pairwise補助loss（weight=20.0, margin=0.1）
導入で現在地まで改善した。詳細・全経緯は`docs/nn-value-phase1.md`参照。
（フェーズ2で本番統合済み、上記参照。）

副産物として、オフライン検証シナリオを増やす過程で`scenarios/kakutori.kif`
を追加したところ、NNとは無関係にestimator戦略本体のバグ（王手駒の
無条件捕獲を見逃す）を発見・修正し`estimator_v8`として凍結した
（`check.rs`節参照）。

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
- **比較の基準は heuristic ではなく凍結版**（`estimator_v6` 等）。heuristic への勝率は
  飽和していて改善の検出力がない。また非推移性（v2 に勝つが v1 に負ける）を検出するため、
  ガントレットで**全凍結版**に勝ち越すことを合格条件とする
- フィッシャー時計 1000秒+3秒 をシミュレートし時間切れは負け（本番サイトは300秒+3秒。
  `-f clock="300+3"` / `ARENA_FISCHER_*` で本番相当に切り替えて検証できる。
  既定予算2000msなら本番条件でも時間切れは出ない）。`choose()` の壁時計を消費し、
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
   `createBotAccount` / `createSession` を呼ぶ小スクリプトを `DATABASE_URL=<scratch>.db npx tsx` で実行。
   `$lib` エイリアス解決を避けたいだけなら `better-sqlite3` で users/sessions/bot_accounts に
   直接INSERTする生SQLスクリプトでもよい（`scripts/server/official-bot.mjs` と同じ手法。
   トークンは `tsb_` + 32byte base64url、ハッシュはSHA-256）
2. `DATABASE_URL=<scratch>.db npm run dev -- --port 5175` でサーバー起動。
   受付時間（平日21-22時/土日21-24時JST）外だと `queue:join` が拒否されるので、
   ローカル検証では `MATCH_OPEN_ALWAYS=1` を付けて常時開放にする
   （`src/lib/server/socket/handlers.ts` の `matchOpenNow` が参照。botには元々適用されない）
3. このbotを `TSUITATE_URL=http://localhost:5175` で起動（バックグラウンド）
4. 人間役はスクリプト（socket.io-client + クッキー認証。数手指して投了）でも、
   **実際にブラウザで対局してレビューする**のでもよい。後者は
   `document.cookie = "session=<生トークン>; path=/"` でログインできる
   （sessionクッキーはhttpOnlyだが、`document.cookie` での新規作成には効く。
   既存のhttpOnly cookieと衝突する場合は一度削除してから設定するか
   シークレットウィンドウを使う）

過疎判定（人間接続数 < 4）によりレート無視で即マッチするので、人間役1人で成立する。

**実戦対局レビューは有効な診断手段**（2026-07-19〜20の実績: 3件の評価関数バグ
knight_bait_w・home_lance_move・kakudo角道誤信を発見・修正）。真の局面を
`bin/dump_position <moves.json> [手数]` で確認し、`bin/make_scenario`で
`scenarios/*.kif` へ変換すれば `bin/scenario` の再現テスト・回帰テストとして
残せる（手順は `scenarios/README.md`）。学習データ化するなら
`bin/export_value_data`（NN方向、上記参照）。
