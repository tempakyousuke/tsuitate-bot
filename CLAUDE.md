# tsuitate-bot

王様のかくれんぼ（`~/Develop/tsuitate`）に外部bot APIで接続して対戦するRust製bot。
サイト・ソルバー（`~/Develop/tsuitate-resolver`）とは**意図的に独立**したプロジェクト（cargo依存もしない）。

**ここは現行の運用と規約の一次資料**。計測記録・実験の経緯は `docs/`（索引は
`docs/README.md`）に分けてある。よく引くもの:

| 知りたいこと | 見る場所 |
| --- | --- |
| 評価ノブ（`TSUITATE_*`）の既定値と、なぜその値なのか | `docs/knobs.md` |
| アリーナ・checkpoint arena・SPSA の詳しい回し方 | `docs/arena-ops.md` |
| 凍結版 v6〜v14 の内容と凍結時の成績 | `docs/frozen-versions.md` |
| 診断バイナリと P0 調査（#19/#24/#28/#31/#34/#36/#40）の判定 | `docs/diagnostics.md` |
| `bin/analyze` の各集計セクションの定義と実測 | `docs/analyze-metrics.md` |
| NN の段階①〜④の現状と再学習の手順 | `docs/nn-direction.md` |

## コマンド

- `cargo test` — ユニットテスト（候補手生成・プロトコル・エンジン・推定器）。
  **push のたびに CI でも走る**（`.github/workflows/test.yml`。`cargo test` と
  `RUSTFLAGS=-D warnings` のビルドだけの軽い門番で、強さの計測（arena/scenario）は
  従来どおり手動起動）。凍結版の未使用 import / 未使用コードは `src/lib.rs` の
  `pub mod frozen;` に付けた `#[allow(...)]` で抑止してあるので、警告が出るのは
  現行コードだけ。**CI のツールチェインは固定**（`-D warnings` と `@stable` の
  組み合わせは rustc が lint を足した日に無関係な PR を落とすため）なので、
  上げるときは手元で警告ゼロを確認してから test.yml のバージョンを書き換える。
  **NN の速度しきい値テスト**（`forward_pass_is_fast_enough_for_hot_loop`、
  value_nn / value_nn_v22 / opp_move_nn）は **debug の `cargo test` から
  `--skip` で外し、`cargo test --release --lib` の専用ステップで測る**
  （issue #26、2026-08-27）。debug 側のしきい値 2ms/回 は手元実測 514〜522µs に
  対して約3.9倍の余裕しかなく、GitHub のランナーはそれより約4.2倍遅い
  （実測 2.1〜2.3ms/回）ので断続的に落ちていた（PR #25 のブランチで CI 10回中2回。
  再実行で緑 = 回帰ではなくランナー差）。release 側は 100µs/回 に対して実測
  2.4〜2.8µs/回 = **36〜41倍の余裕**なので揺れず、**性能回帰の門はそのまま残る**
  （debug の assert を消すだけだと門が無くなる）。しきい値の定数は
  `src/value_nn.rs` 等にそのまま残してあるので、ローカルの `cargo test` では
  従来どおり debug 側も走る。**`src/value_nn_v22.rs` は触らない**:
  `frozen::SHARED_MODEL_PINS` がファイル全体を sha256 しているので、テストの
  定数だけを直しても pin 更新が要り、ランタイム挙動は不変なのに v12〜v14 の
  `behavior_fingerprint` が変わる（workflow 側で解決すればこの副作用はゼロ）
- `cargo test --release -- --ignored` — 遅い検証（shogi.rs の perft depth 4/5）
- `cargo run --release --bin arena -- [対局数] [候補] [基準1] [基準2] ...` — 戦略同士の対戦。
  基準を複数並べるとガントレット。**戦略の変更は必ずこれで凍結版ガントレットに
  有意に勝ち越すことを確認する**（既定の対象は v9 以降）。
  **実行はローカルでなく GitHub Actions**（`.github/workflows/arena.yml`。
  **通常のコード push では走らない**）:
  `gh workflow run arena.yml --ref <ブランチ> -f games=100 -f candidate=estimator -f baselines="estimator_v13 estimator_v14"`、
  `gh` が無ければ `.github/ci/arena.request.json` を書いて push（例は
  `arena.request.example.json`。削除の push は plan が全ジョブをスキップする）。
  - 局数は当面 **100 を既定**（50%付近の信頼区間は 100局で±10pt / 200局で±7pt /
    1000局で±3.1pt）。僅差で判定できないときだけ増やす
  - **時計は既定 1000秒+3秒**（`-f clock="300+3"` で本番相当）。**思考予算は本番も
    既定 2000ms のまま**でよい（2026-07-26 実測。300秒+3秒でも時間切れ0）。
    900ms へ絞ると −14.5pt、8000ms でも +0.5pt = **もう強さの調整ノブではない**
  - `-f match_seed=<数>` で対局条件列を決定論化（版比較は必ずこれでペアにする）。
    指定すると1行=1対局の `arena-games.jsonl` も出る
  - `-f env=` はプロセス全体（＝凍結相手にも効く）。**候補側だけに効かせるなら
    `-f cand_env=`**（`StrategyConfig` として候補 instance にだけ渡る）
  - `-f oracle=nofoul|check_nofoul` は診断用オラクル（記録は自動無効化）
  - 各シャードは `bin/analyze` も走らせてジョブサマリーへ集計を出す
  - **反則経済には約31〜36pt の伸びしろが実在する**（vs v7 で通常 51.9% に対し
    nofoul 83.2%）。ただし反則マスを記憶して割り引く系統は4種とも不発
    （`docs/c8-direct-synthesis.md`）
  - シャード分割・aggregate・`pair_with`・`arena-var`・issue #40 の
    opponent-balanced 合算器（`arena-balance`）の契約は **`docs/arena-ops.md`**
- `cargo run --release --bin checkpoint_arena -- <extract|run|compare|report|arena-var>` —
  checkpoint arena（issue #19）。**破滅検出器としては不合格**（2026-08-24 の実測で
  通常 arena と符号が逆・コスト優位も無し）。`arena-var` だけは通常 arena の相方として
  現役（1行=1対局の記録2本から局ごとのペア差の分散・MDE を出す）。
  詳細は `docs/arena-ops.md` と `docs/checkpoint-arena-p0.md`
- `cargo run --release --bin tune -- [反復数] [評価あたり対局数] [基準...]` — 評価パラメータ
  （`strategy::EvalParams`）の SPSA 自動チューニング。目的関数はアリーナのスコア率、
  `TUNE_OBJECTIVE=scenario` / `scenario_score` でシナリオ側にも切り替えられる。
  f+/f− は共通乱数法でペアリングし、ログ（`TUNE_LOG`）から自動再開する。
  **採用するときは `EvalParams::default` を書き換えてフルガントレットで確認する**。
  ログの見方は `tuning/README.md`、env の全量と GCE での回し方は `docs/arena-ops.md`
- **粒子尤度のフィット**（`.github/workflows/fit.yml`、CIのみで実行する）:
  `gh workflow run fit.yml -f run_ids="<arena実行のrun ID...>" -f max_games=600`。
  過去アリーナの `arena-records`（観測列＋真実）を教師に、粒子群の中で真の局面を
  判別する条件付きMLE（`bin/fit_particles`）を回し、係数を `src/likelihood.rs` の
  `FITTED_THETA` へ手で反映する。評価側は `stratified_sample` が exp(θ·φ) を
  粒子重みに乗じる（fit_opp の局面版。2026-07-16 のフィットで実効候補数 59→33）
- **採点式レビュー（evals/）**: 実戦棋譜の決定点ごとに候補手を 0〜10 点で
  採点し、SPSA の目的関数にする仕組み（2026-08-07。二値の bad= の一般化。
  形式とワークフローは `evals/README.md`）。
  `cargo run --release --bin make_eval -- <kif>` で全決定点のスケルトンを
  自動生成（冪等追記・反則後サブ状態対応）→ 採点 →
  `python3 scripts/quest_review/sync_eval.py` でシナリオの `scores=` / `bad=`
  （2点以下）へ同期 → suite に「平均得点」が出る。SPSA は
  `TUNE_OBJECTIVE=scenario_score`（score = 平均得点/10 のシナリオ平均 ∈ [0,1]）。
  未採点の手が選ばれたら仮4点＋件数表示（評価済み候補の外へ逃げる
  指標の抜け穴への対応）。
  quest_20260731 は md から移行済みで **eval が一次資料**。
  **ノブ検証の標準ループ**（2026-08-09〜10 に確立。指標穴で3度騙された結果）:
  ①候補ノブで suite → ②`python3 scripts/quest_review/append_unscored.py
  evals/<名前>.eval.md "<実験名>" <scenario-combined 展開先...>` で
  **選ばれたのに未収載の手**を `?` 追記（既定は計2回以上。1回だけの
  ロングテールは仮4点の有界ノイズなので追わない）→ ③ユーザーが採点 →
  ④`sync_eval.py` → ⑤**同一コミットで対照と候補の suite を取り直して確定判定**。
  **採点前の suite 差は信じない**（promo価格改定 C1・不成 combo・
  ブラインドhome の3件で「見かけの改善が未採点手への逃避だった」を実証。
  極端な例: taint フォールバックの −116 は全ブロックで 0〜2点の既知悪手
  7七桂成への逃避が主因だった）。
  **未収載候補の別経路**: `scripts/quest_review/rerank_eval.py` は
  「同じ USI の他の決定点での平均点」を事前値にして rank_dump を並べ替える
  オフライン診断（`prior + alpha×score`）。出力は TRIAL TSV なので
  ファイルへ落として append_unscored へ渡せる。局面を見ない事前値なので
  発見は少数の USI に集中し、stderr の平均は suite の平均得点とは
  比較できない（詳細は evals/README.md）。
  **初手得点**（2026-08-22、案2の玉プローブが発端）: suite の「得点」は**受理された
  手**を反則前ブロックの採点表で数えるので、「2三玉（反則プローブ、7点）→ 1三香で
  回収」は反則前ブロックの 1三香=3 点で数えられてしまう（回収の正解は反則後
  ブロックの領分）。`ChoiceStats.first_tally` / `mean_score_first` が**最初に試みた手**
  で採点し、suite 行・merge 表に「初手得点」列（反則を挟んだ試行が無い行は〃）、
  合計に「初手得点の全体平均」を出す。TRIAL 行は6列目に初手を持つ（旧 TSV は
  受理手で代用）。反則プローブ系の施策の採否は初手得点で見ること。
  **ブロック取り違えの関門**: eval の採点を別の手目・別の反則後サブ状態へ
  書くと、相手の駒を動かす USI が `scores=` に入る。到達しえないので
  平均得点にも不合格計にも出ず suite では永久に気づけないため、
  `cargo test 採点表` （`scenario_core` のテスト）で常時検査する。
  反則後ブロックとシナリオの対応表は
  `scripts/quest_review/foul_blocks.py` に一本化してある
- `cargo run --release --bin scenario -- <名前|suite|batch <名前...>>` — 実戦棋譜の
  局面再現実験。
  `scenarios/*.kif`（Shogi Quest エクスポート + `*scenario ply=N` 行）を再生して
  特定局面での選択・粒子の信念（diag）・終局までの遂行（continue）を測る。
  追加手順は `scenarios/README.md`。
  `target=` は注目手、`bad=` は**不合格リスト**（選んだら悪手として数える手の全量。
  kakudo方式で target が悪手なら重複して入れる。出力とsuite行に「不合格計」が出る）。
  **同一棋譜から切り出した ply 違いのシナリオ群のローカル計測は batch（suiteも同経路）
  で回す**: 同一棋譜×同一手番側はシードごとに prewarm 済み戦略を ply 昇順で継ぎ足して
  共有し、各決定点は `Strategy::clone_boxed` のスナップショットに試行させる
  （GUI の IncrementalEstimator の Strategy 版。等価性はテストで担保。
  v13 以前の凍結版は clone_boxed 非対応なので従来どおり毎回作り直し。
  v14 は対応する）。
  `fouls=` 注入シナリオも 2026-08-08 からチェーン共有する（共有チェーンは
  素のリプレイのプレフィックスまで進め、注入反則の尾はスナップショット側だけに
  食わせる。等価性テストあり）。
  suite は (棋譜,手番) グループ単位でコア数までスレッド並列
  （`SCENARIO_WORKERS` で上書き可）。
  **オラクル錨**（2026-08-19）: `--oracle N` / `--oracle-lag K`（kif の
  `oracle=N`）で手番側の推定器に「N 手目までの真実」を与え、以後は観測だけで
  処理させる（`Estimator::oracle_anchor`。`--oracle-lag 1` = 直前の相手手だけ
  不明）。粒子生成を省くので一瞬で終わり、悪手が「錨でも出る＝評価/CheckSolver
  側」「錨で消える＝信念側」の切り分けに使う。真実由来の手が出ることは織り込む。
  実対局・アリーナからは呼ばない（観測にない情報）。詳細は scenarios/README.md
  **全件 suite は CI のシードシャードで回す**
  （`.github/workflows/scenario.yml`。通常のコード push では走らない）:
  手動は `gh workflow run scenario.yml --ref <ブランチ> -f trials=20`。
  `gh` が使えないときは `.github/ci/scenario.request.json` を書いて push
  （例は `scenario.request.example.json`）。リクエストファイルの削除 push は
  全ジョブがスキップされる（arena.yml と同じ）。
  **1シード=1ランナー**の matrix（trials=N でシード 0..N-1 の N ランナー。
  各ランナーは `suite 1 --seed-base <シード> --tsv` で全シナリオ1試行）で、
  ランナー内はチェーン共有＋グループ並列が効く。総合表は aggregate ジョブが
  `scenario merge`（TRIAL 行の合算。集計は ChoiceStats 共有で suite と完全一致）
  で出す markdown 表（サマリーと artifact `scenario-combined` の merged.md）。
  旧方式（1シナリオ=1ランナー、〜2026-08-08）は全件で計30時間・壁時計4時間
  かかっていた（prewarm の重複が9割。143件×20シードでゼロから再生）。
  新方式の実測（run 31253093783、全143件×20試行）: **壁時計7分18秒・
  計算合計103分 = 旧比 1/33 と 1/18**。
  シード・試行数は同一なので統計は旧方式と等価
  （実測: 対照比の不合格計の差は平均−0.31±2.21/件で系統差なし）。
  `-f scenarios="kakutori keima"` で対象を絞れる。
  `-f env="TSUITATE_XXX=値"` で評価ノブを渡せる（arena.yml と同じ規約。
  既定0で入れた項の回帰確認に要る）。**プロセス env なので、同じプロセスで作る
  凍結版にも効く**（scenario は対照も候補も同じプロセスなので、版比較は
  `-f env=` ではなく同一コミットのペア計測で行う。設定境界は `src/config.rs`）。注意: 試行はシード同一でも壁時計ベースの
  予算スケールで揺れるため、10試行の±2〜3件差はノイズ。版比較は20試行×両版で
- `cd scenario-gui` → `npm run tauri dev` — シナリオデバッグ GUI（Tauri）。
  `.kif` の取り込み・任意 ply までの再生・先後視点切り替え・候補手分析
  （seed集計=全エンジン対応で回帰比較可 / ランキング=現行 estimator の
  スコア内訳 / **玉位置ビリーフ**=手番側の粒子が「相手玉はどこにいると
  思っているか」を盤に % で重ねる。赤枠=真実、厳密粒子のみ↔taint込みを切替可）。
  思考予算も指定可（既定 2000ms。900ms はスケール基準で、本番相当ではない）。
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
  （CI では常時有効で artifact `arena-records` に上がる）。反則の原因分類・駒得収支・
  只取られ・詰み逃しに加えて、**被詰めろオラクル**（相手が実行してこなくても
  自玉の受けを測れる）・**詰み経済**（#28）・**王手中の反則経済**（#31）・
  **連続王手の起点**（#34）のセクションを常設する。
  **analyze に集計行を足したら `arena.yml` と `check-economy.yml` のフィルタも直す**
  （集計行だけを拾う grep なので、足しただけでは CI に出ない）。
  各セクションの定義・門・実測ベースラインは **`docs/analyze-metrics.md`**
- **診断バイナリ（runtime には何も入らない計測専用）** — 一覧・契約・実測・判定は
  **`docs/diagnostics.md`**。専用ワークフローはどれも**通常のコード push では走らない**。
  - 詰み経済 #28: `mate_probe` / `mate_continue`（`mate-economy.yml`）
  - 王手中の反則経済 #31: `check_probe` / `check_policy` / `check_price` /
    `check_continue`（`check-economy.yml`）
  - 被王手の前の準備 #34: `check_prep_probe`（`check-prep.yml`）
  - 王手駒仮説の希釈 #36: `check_belief_probe` ＋ オラクル arm（`check-belief.yml`）
  - 人手採点の残差ランカー #24: `export_eval_rank_data` ＋
    `scripts/eval_rank/fit_residual.py`
  - 戦力投資 #40: `investment_probe`
  - 較正プローブ（粒子不要で一瞬）: `king_cands` / `eval_features` /
    `collision_probe` / `pawn_push_probe` / `anchor_probe` / `capturer_probe` /
    `mine_check`。**施策を実装する前にここで発火母数を測る**（発火が数%未満なら
    アリーナ 104局×2 でも suite でも原理的に判定できない）
  - 信念の質（重い・少数の記録に対してだけ）: `belief_probe` / `king_probe` /
    `export_belief_data`
  - **元 Arena 実行の実験条件は `scripts/ci/verify_arena_provenance.sh` で
    機械的に検査してから解析する**（ワークフローごとに書くと片方だけ緩くなる）。
    **シャードが欠けたら判定を出さない**
- `cargo run --release --bin mine_tsume -- <records/ or *.jsonl...> [--solver <path>] [--out <path>]`
  — **ついたて詰将棋の問題の採掘**（サイトの「詰めチャレ」の問題生成）。記録の
  真実を再生し、各決定点の**実戦の局面をそのまま**問題にして、ソルバーCLI
  （`--shortest --find-second --estimate-rating`）にかけて詰みがあるものだけを残す。
  出力はサイトの `/admin/tsume-challenge` に貼り付ける取り込みJSON
  （レート推定値と特徴量つき）。
  - 盤上の駒は両者とも全部残す。**攻め方の玉も落とさない**（双玉）。自玉があると
    ピンや逆王手といった実戦特有の制約がそのまま効くので、詰将棋の慣習で玉を
    外すより問題として味が出る（2026-08-20 ユーザー判断。将棋クエストの詰めチャレも
    実戦の局面をそのまま出している）。ソルバーの `generate_legal_moves` は自玉が
    あれば自殺手を除外し、無ければ擬似合法手を返すので双玉でも正しく解ける
  - 玉方の持ち駒は JSON に入れない。ソルバーが「全駒 − 盤上 − 攻め方持ち駒」で
    自動算出し、結果は実戦の玉方の持ち駒と一致する（＝局面が完全に復元される）
  - 後手番の局面は盤面を180度回して先手番に直す（ソルバーは攻め方＝先手が前提）。
    駒の向きも回転で入れ替わるので色のラベルを差し替えるだけでよい
  - 重複は署名（正規化JSONの SHA-256）で落とし、`sourceId` として取り込みの
    upsert キーにもする
  - **1対局・1攻め方から採るのは1問だけ**（`--max-per-game`、既定1。0で無制限）。
    詰みは一度現れると攻め方が見逃したまま数手続くことが多く、各手番の局面を全部
    採ると「同じ対局の2手違い」のほとんど同じ問題が並ぶ（署名の重複排除では
    別局面なので落ちない。2026-08-25 ユーザー指摘）。ソルバーにかけた後に
    対局×攻め方ごとに**詰み手数が最も長い局面**（同手数なら手前）だけ残す
    （2026-08-30 ユーザー要望。〜2026-08-29 は「最初に詰みが現れた局面」だった。
    全候補がソルバーを通った後の選別なので追加コストは無い）。攻め方が違えば別問題
    として許容する（ユーザー判断。先手の詰みと後手の詰みは盤の向きも駒も別物）。
    ワークフローは棋譜ファイル単位でシャードに分けるので、シャードをまたいで
    同じ対局が採られることはない。既に取り込み済みの重複は間引かない（放置で可）
  - 主なオプション: `--skip-plies N`（既定20。序盤は詰まないので飛ばす）/
    `--min-depth` `--max-depth`（既定 3〜15）/ `--jobs N`（ソルバーの並列数）/
    `--max-per-game N`（上記）/
    `--allow-second`（余詰めのある問題も採用。既定は捨てる）/
    `--dump-candidates <path>`（ソルバーを呼ばず候補だけ書き出す）
  - 実測（2026-08-20、`records/` の59局・`--skip-plies 40 --jobs 4 --timeout-secs 20`）:
    候補1823件・所要7分48秒で**34問**採用（3手詰14 / 5手詰11 / 9〜15手詰9、
    推定レート 992〜2244）。盤上26〜38枚
  - **本番の問題生成は GitHub Actions の `Mine tsume`**（`.github/workflows/mine-tsume.yml`）。
    ローカルの `records/` は開発用のサンプルで、運用では CI の自己対局から掘る:
    1. `arena_run_id` に既存の Arena 実行のIDを渡す = 強さの検証で回した棋譜を
       そのまま再利用する（**追加の対局コストゼロ**。ただし artifact の保持期間
       内の run に限る。`oracle` 有効の実行は記録を出さないので対象外）
    2. `arena_run_id` を空にすると、このワークフローが自分で自己対局してから掘る
       （問題を増やすことだけが目的のとき。時計の既定は本番相当の 300+3）
    採掘はソルバーを1問ずつ spawn するので対局より重く、`mine_shards` で棋譜を
    分割して並列に解き、`collect` ジョブが `sourceId`（盤面の署名）で重複を
    落として1つの取り込みJSONにまとめる。ソルバーは `tempakyousuke/tsuitate-solver`
    を checkout して `--no-default-features`（GUI 依存なし）でビルドする。
    成果物は artifact `tsume-puzzles` の `tsume-puzzles.json`。
    **毎日 09:00 UTC（18:00 JST）に自動実行される**（既定は自己対局160局。
    1対局1問になる前の実測は 0.62問/局 で概ね100問/日だったが、本番プールでは
    313問中145問が同じ対局の重複だったので、以降は概ね半分の 0.3問/局 前後になる
    見込み）。取り込みは**サイト側の日次 cron が artifact を
    取りにくる**（tsuitate の `docs/operations.md`「詰めチャレ問題の日次取り込み」）
    ので、このワークフローは artifact を置くところまで。手で確認したいときは
    artifact を落として `/admin/tsume-challenge` に貼り付ければ同じ経路を通る。
    **schedule では `workflow_dispatch` の `default:` が効かない**（`inputs.*` が
    空文字になる）ため、実効値はワークフロー冒頭の `env` ブロックで決めている。
    パラメータを足すときは `inputs` と `env` の両方に足すこと
  - `timeout_secs` は「実行時間」と「長手数の問題が採れるか」のトレードオフ。
    ほとんどの候補は詰みなしと一瞬で分かるが、**詰みのある長手数の問題ほど時間を食う**。
    実測（59局を4分割）: 20秒だと11〜15手詰の4問を取り逃して 34問→30問になった。
    難しい問題ほど詰めチャレでは価値が高いので CI の既定は60秒にしてある
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
    を使う。**ただし v12 / v13 はこの名前も読む**（凍結時に持ち込んだまま。
    2026-08-19 に判明、凍結後は編集しない方針で残置）ので、v12/v13 相手の
    予算スイープは両側が動く。v6〜v11 と v14 以降は読まない（v14 から
    `freeze_estimator.py` がこの名前を落とす）
  - **評価ノブ（`TSUITATE_*`）は約 150 個ある。全量・既定値・採否の実測は
    `docs/knobs.md`**。ここには運用に要るものだけ挙げる。押さえておく規約:
    - **既定値がそのまま現行挙動**。`0` / `off` のノブは実装だけ残した不採用・保留で、
      env から与えれば再試行できる
    - 設定の解釈は `config.rs` が構成境界で**一度だけ**行い、以後は
      `StrategyConfig` を見る。ノブを足したら `STRATEGY_ENV_KEYS` へ通すこと
      （通し忘れは `cargo test` が落とす）
    - **凍結版 v6〜v14 は凍結時点で読んでいた env を今も読む**（一覧は
      `frozen::env_keys_in_source(name)`）。`-f env=` は両側に効くので、
      候補側だけなら `-f cand_env=` / `ARENA_CAND_KNOBS`。**v15 以降は読まない**
    - 採否の基準は「**同一コミットのペア**で scenario suite ＋ アリーナ・
      ガントレット」。**採点前の suite 差は信じない**（未採点手への逃避で
      3度騙された。`docs/knobs.md` の該当節）
  - `TSUITATE_STRATEGY`（既定 `estimator`。旧来の単純botは `heuristic`）
  - `TSUITATE_QUEUE_RETRY_MS`（既定 60000）: キュー参加拒否・受付終了後の再試行間隔
  - `TSUITATE_RECORD_DIR`（既定 `records`。空文字で無効）: 対局記録（JSONL）の出力先。
    1対局1ファイルで、botの観測イベント全量・選択した手と思考時間・終局結果を記録する
    （`src/record.rs`）。相手の実際の手は含まれない。ローカルdevサーバー対局なら
    サーバーDBの `games.moves` に全手順（真実）が残るので、分析にはDBダンプと突き合わせる

## アーキテクチャ

コールバック（Socket.IOスレッド）→ mpsc チャネル → 単一メインループ、の一方向。
状態（対局ID・反則済みの手・観測履歴）はメインループだけが触る。

- `config.rs` — **設定境界**（issue #21、2026-08-24）。`TSUITATE_*` は
  CLI・サーバー等の構成境界で**一度だけ**解釈し、以後は strategy instance が持つ
  `StrategyConfig`（147 個のノブ＋思考予算＋定跡パス）だけを見る。
  評価・推定の実装は深い呼び出しの奥から定数を引くので、`config::scoped` で
  **スレッドローカルに設置**する（`EstimatorStrategy` の choose / prewarm /
  oracle_anchor の入口）。設置しない経路（診断バイナリ・GUI・凍結版）は
  `ambient()`＝プロセス env 由来に落ちるので移行前と同じ挙動。
  - **候補側だけノブを変えてもプロセス env は動かない**ので、env を読み続ける
    既存の凍結版 v6〜v14 は候補ノブに反応しない（PR #20 で見つかった事故の恒久対策）
  - `OnceLock` のプロセス全体キャッシュを使わないので、arena / scenario の
    スレッド並列でも arm ごとの値が混ざらない
  - `fingerprint()` は**解決後の値**の sha256（未知のキーや既定値と同じ指定では
    変わらない）。checkpoint arena が「相手の実効設定が両 arm で一致」を検査するのに使う
  - `STRATEGY_ENV_KEYS` が戦略が読むキーの全量。ソース走査との突き合わせを
    `cargo test` が常時検査するので、ノブを足して config を通し忘れると落ちる
  - 設計と凍結境界の分類は `docs/frozen-hermetic-boundary.md`。
    **既定挙動の同一性**は ①移行した96ノブの解決式が旧アクセサと文字列一致
    ②arena run 32697854659（104局×2・`match_seed=20260815`）で
    vs v13 56.7%±9.5 / vs v14 50.0%±9.6・反則/局 6.01〜6.24・
    思考平均 1204〜1232ms・時間切れ0 と直前の記録帯に収まること、の2段で確認
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
  scenario-gui の玉位置ビリーフ表示が使う。
  **3方向目（2026-08-03 追加）: `immobile_opponent_pawns`** =「動けないことが
  確定している相手の歩」。歩は前進1マスしか動けず、前が自駒で塞がっていれば
  進む＝取る＝観測される。相手の手番が来るたびにその可能性を検査して各筋の
  「歩がいられる段」の上界集合を更新し、1件に絞れた筋を確定として返す
  （自駒の配置は常に厳密に分かるので健全。成り込みうる段に届いた筋は
  と金として盤上を自由に動けるので諦める）。**用途は粒子生成の不変量**:
  変異救済（mut_rescue）の移設元から除き、`opp_king_candidates` からも落とす
  （確定歩のマスに玉は居ない = ここで落とせば投影・ネット・変異救済すべてが
  守られる）。健全性は2局×全局面×両視点で違反0（テスト常設）。
  **鋭さは平均 0.1〜0.3マスと限定的**（自分の歩が相手の初期歩と正面で
  向かい合う形は初期配置が3段離れているため稀）なので、単独で広く効く項では
  なく「論理的に確定した事実を粒子生成へ与える」枠組みの第一歩。
  診断は `bin/locked_pawns <kif...>`（確定数と健全性を再生検証）
- `mate.rs` — **持ち駒打ちに限定した一手詰めの検出**。玉のマスから8方向へ空きマスを
  辿って「そのマスに打てば王手になる (マス,駒種)」を逆引きし、詰み判定する。
  `drop_mate_threat` は成立度を2段階で返す: `Mate`（盤上の駒だけで詰み）と
  `IfSupported`（玉で打った駒を取る以外に受けがない = **打ちマスに支えが1枚
  あれば詰み**）。攻め側は自分の支え駒を正確に知っているので `Mate` を、受け側は
  相手の支え駒が不可視なので `IfSupported` も使う。全合法手からの一手詰め探索
  （`mate_in_1`、診断専用）はホットパスに載らないので打ちに限定している
  （実測: 対人83局の被詰めろ232件のうち72%が打ち詰み）。
  strategy.rs の `mate_threat_w`/`mate_risk_w` と bin/analyze の被詰めろ指標が使う
- `check_prep.rs` — **被王手の前の準備（王手耐性）の共有定義**（issue #34、2026-08-29。
  runtime には入らない診断だけ）。`bin/analyze` の P0-1（連続王手の起点）と
  `bin/check_prep_probe` の P0-2（較正プローブ）が**同じ定義**で数えるための場所。
  - **王手起点マス集合 C(K)** は**玉位置ごとの固定集合**（盤端までの8方向の距離2以上＋
    桂の跳び起点2マス）。**自駒で止めない**のが要点で、自駒を置いて未被覆マスを集合から
    消せると「分母を縮めて被覆率を上げる」分母操作が成立してしまう
  - 被覆は**確定利き**（歩・桂・金銀の歩進）と**楽観レイ**（飛角香）を分けて持つ。
    飛び駒の利きは自駒にしか遮られない楽観値で、「空きマス」は「自駒がいないマス」。
    名前もそのとおり（`potential_covered` / `potential_flight`）
  - **隣接8マスは F3 の領分で F1 には持たない**（説明力を見てどちらかを選ぶのではなく、
    設計時点で分担を固定する）。F3 は `自駒非占有・非被覆 / 自駒非占有・被覆 / 自駒占有` の
    3状態＋**占有の有無によらない隣接の被覆数**
  - `decision_snapshots` は**終端の手番**（受理手が無く `end.moves` に現れない手番）も
    1つ足す。観測の作り方は `truth_replay` の共有関数だけを使う
  - `hurdle` 部分モジュールが P0-2 の回帰（ロジスティック ＋ ゼロ切断・上側打ち切り
    Poisson ＋ g-computation の周辺平均差）を持つ。合成データで符号と大きさを取り戻すこと・
    打ち切りが上側確率として入ること・周辺平均が上限を超えないことを `cargo test` が検査する
- `check_economy.rs` — **王手中の反則経済の共有定義**（issue #31、2026-08-28。
  runtime には入らない診断だけ）。王手中の手番の取り出し（`check_turns`）・
  手数の対応の規約（`decision_groups`: 反則は手番を変えないので `MyFoul` は
  その手数のまま、`MyMove` は適用後の値なので −1）・手種（bot の意図で分ける
  `CheckMoveKind`）・型（`TurnType`）・反則の原因の真実からの分類
  （`classify_check_foul`）・元対局 cluster bootstrap（`cluster_ratio_ci`）。
  `bin/analyze`（P0-1〜P0-3）と、以後の P0-4/P0-5/P0-7 が**同じ定義**で
  数えるための場所。手番開始時（反則0）の状態の復元（`entry_replayed`。
  `for_each_decision_full` が渡すのは反則を食った後なので、両者のログ末尾から
  反則の観測を落とす）も P0-4 / P0-5 で共有する
- `check_belief.rs` — **王手駒仮説の希釈の共有定義**（issue #36、2026-08-29。
  runtime には入らない診断だけ）。仮説重みへの介入 arm（`Belief`: オラクル・誤誘導・
  直前手の演繹）と arm 名の規約（`ArmSpec` = `[belief|policy][@shadow|@real]`）・
  1 arm を回す配管（`run_arm`。p-only と実再決定の**両方が1か所**にある）・
  母集団の取り出し（`decision_points`。**終端手番を含む**全王手中手番と `Attrition`）・
  伝達の分解の注目手（`focus`）。`bin/check_belief_probe`（P0-1）・
  `bin/check_policy`（P0-2）・`bin/check_continue`（P0-2b）が**同じ arm 名で同じ配管**を
  指すための場所（別々に書くと、削減量と勝率差が別物の測定になる）
- `check_policy.rs` — **王手中の反則経済 P0-5 の共有定義**（issue #31、2026-08-28。
  runtime には入らない診断だけ）。1候補の最小の入力（`PolicyMove`。**p と価格の
  両方**を付け替えて score を引き直せる。`check_economy::PricedMove` は価格しか
  動かせない）・**p-only shadow update**（`ShadowUpdater`）・方策（`Policy`）と
  真実に対する裁定（`simulate`）・受理直後の真実指標（`truth_after`）・
  候補分布上の較正（`CalibrationSums`）。
  **p_legal のブレンドは `strategy::blend_p_legal` を呼ぶ**（`evaluate` と同じ関数。
  別々に書くと「仮想更新が実再決定とずれた」のか「式が食い違っていた」のかを
  分けられない）。`entry_setup`（手番開始時のランキング・粒子・prewarm 済み
  instance）は P0-5 と P0-6 が共有する
- `mate_economy.rs` — **詰み経済の共有定義**（issue #28、2026-08-27。runtime には
  入らない診断だけ）。詰め手の排他的分類（`drop_only / board_only / both`）・
  安全手（指した後に相手の一手詰めが消える手）・被詰めろのエピソード
  （連続する相手番を1つに畳む）・**最後に受けられた決定点**・同じ手数帯の
  対照決定点・粒子上の危険質量 q とオフライン再スコア。
  `bin/analyze`（P0-1/P0-2）・`bin/mate_probe`（P0-3）・`bin/mate_continue`（P0-6）が
  **同じ定義**で数えるための場所（別々に数えると較正の数字が意味を失う）
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
  粒子のほうが高い（0.55 vs 0.49）。ここは CheckSolver（制約推論）の領分のまま。
  **もう一つの盲点は 2026-07-29 に修正**: legal_under（仮説の王手駒1枚だけの
  盤面）は見えない支え駒を無視するため「支え付き王手駒を玉で取る手」を
  p=1.00 と断言して反則していた（過信は玉の手に集中: 玉捕獲 73%・玉逃げ
  75〜81% vs 玉以外の捕獲 97%。bin/analyze の「単独仮説合法→実際合法」で
  計測できる）。玉捕獲の寄与だけ実測 0.73 で割引＋反則減衰を緩和
  （`TSUITATE_CHECK_KING_PRIOR_W` の節参照。**玉の逃げまで割り引くと悪化**）。
  **v15 候補（2026-08-19）: 信念ネット占有を仮説の事前にする**
  （`TSUITATE_CHECK_BELIEF_OCC_W`、既定 0、作業点 1.0）。段階②が埋まらなかったのは
  「どの駒が王手しているか」の同時制約で、空きマスを落とす必要条件は別問題。
  仮説重み × 占有（床 0.05）。**粒子が投票できるときは掛けない**
  （乗算が投票を上書きする実測あり）。ブラインドだけ・捕獲マスは乗らない。
  既定 1.0 で PR #8 に入ったが vs v14 で負け越しが見えたため既定 0 へ戻した
  （機構は残す。凍結版は知らない）
- `strategy.rs` — `Strategy` trait。`heuristic`（前進＋乱数の旧実装）と
  `estimator`（粒子加重平均で候補手を評価: 駒得期待値・反則確率×急峻な反則コスト・
  取られリスク・王周辺の利き圧力・王手/詰みボーナス・駒探し/王探しの情報利得・
  利き被覆）。
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
  **駒種特化2項（knight_bait_w / tokin_probe_w）は 2026-07-30 に削除**
  （ユーザー方針: これらの手は一般の評価値から出てくるべきで、駒種特化の
  ハードコード項では補わない。v9 以前の凍結版には残っている）。実測:
  ペア3シード（match_seed=20260801〜03、vs v13）で 49.5% vs 対照 50.2% =
  中立、keima（knight_bait の発端）は 20/20 維持で既に冗長だった。ただし
  **tokin_probe が担っていた「歩を敵陣側へ働かせる勾配」の穴はシナリオに
  出ている**（tokin-bet P*8f 19/20 → P*8a 13/20・lance-selfdrop の垂れ歩
  P*9h 10/20 → 消滅・watch-…223458 の悪手 L*1b 5→10/20）。この穴は将来の
  一般項（成り価値・持ち駒経済）が埋める前提の既知ギャップ。
  人間との対局レビューで「良い手を探す力が根本的に足りていない」という
  指摘もあり（33手目5八四金: 位置が確定している敵駒の利きへ無防備に
  踏み込む手を20/20で選択）、個別の駒種特化項の積み増しでは埋まらない
  という認識と整合する。詳細は下記「NN方向」参照。
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
- `frozen/` — アリーナ比較の基準となる凍結版戦略。
  `SOURCES`（凍結版ソースの**一箇所管理**。監査とガードが引く）/
  `env_keys_in_source(name)`（その版が読む env）/ `versions_using(module)`
  （共有モジュールを呼ぶ版）/ `SHARED_MODEL_PINS`（共有モデルの sha256）を持つ。
  **共有モデルを更新すると、それを呼ぶ凍結版の挙動が変わる**:
  `likelihood.rs` / `value_features.rs` / `opp_move_features.rs` / `opening.rs` /
  `joseki.json` は v12〜v14、`belief_nn.rs` / `belief_features.rs` は v13/v14、
  `king_belief_nn.rs` は v14。**NN の重みは固定コピーへ移した**
  （`opp_move_nn_v25.rs` / `value_nn_v22.rs`。再学習しても凍結版は動かない）。
  実際に **2026-08-21 の value NN 再学習（commit 387f0ac）は v12〜v14 の挙動を
  変えている**（当時は検知の仕組みが無く、ガントレット値もこの前後で厳密には
  比較できない）。**2026-08-24 のユーザー判断で「再学習前へ戻さず、現在の
  v12〜v14 の挙動を基準として pin」**とし、`value_nn_v22.rs`（切り出し時点の
  `value_nn.rs` と数値完全一致）へ向けた。
  残りの共有依存は `SHARED_MODEL_PINS` の sha256 が変わるとテストが落ち、
  影響する凍結版を名指しするので、(a) 固定コピーを作る (b) 承知でハッシュを更新し
  再計測を記録する、のどちらかを必ず選ぶことになる。
  実績: **2026-08-26（PR #25）に `src/opening.rs` の pin を更新**した。`load()` を
  `parse_book()` へ切り出し、「与えた bytes からキャッシュへ入れる」`preload()` を
  足した変更で、**凍結版の挙動は変わらない**（読み取り規則とフォールバックは
  文字どおり同じ・`preload` を呼ぶのは `bin/export_eval_rank_data` だけ）ため
  基準の再計測はしていない。
  `versions_using(module)` / `env_keys_read_by(name)`（共有モジュール経由の env 込み）/
  `behavior_fingerprint(name, env)`（版・env・共有 pin から作る実効挙動の指紋）で
  機械可読に取れる。
  v9〜v11 は NN の重みを凍結ファイルへコピーしているので影響しない。
  **v15 以降は実行時 env を読まない**（`HERMETIC_FROM`）。
  **各版の内容と凍結時の成績は `docs/frozen-versions.md`**（現行の最新は
  `estimator_v14`、2026-08-19 凍結）。
  凍結後は編集しない。改善が確定したら
  `python3 scripts/freeze_estimator.py <N> <日付> "<差分の要約>" > src/frozen/estimator_vN.rs`
  で生成し（estimator.rs/check.rs/strategy.rs を1ファイルへまとめ、テストと
  診断フックを落とす。**v15 以降は実行時 env を一切読まない**（issue #21）:
  ノブのアクセサは凍結ファイル自身の `frozen_config()`（空の `EnvSource` から
  解決＝凍結時点の既定値）を引き、`ambient()` は `frozen_defaults()` へ差し替わる。
  生成物に `env::var(` が残ったら生成時に失敗し、`frozen/mod.rs` のテストも落ちる。
  思考予算は `EstimatorVN::with_budget_ms(seed, ms)` で明示的に渡す）、
  `frozen/mod.rs` の `pub mod` と **`SOURCES`**・`strategy::make`/`make_seeded`・
  `arena.yml` の baselines 既定値へ登録する（**`checkpoint-arena.yml` の
  `opponent` 既定値・`checkpoint-arena/deck.json` の `opponent` も同じ
  チェックリストで更新する**。`bin/checkpoint_arena` は `frozen::SOURCES` を
  引くので、そちらの手動更新はもう要らない。
  未使用 import / 未到達コードの警告は `lib.rs` の
  `pub mod frozen;` に付けた `#[allow(...)]` が版を問わず抑止するので、
  凍結ファイル側でもここでも何もしなくてよい）。**生成後は同一性確認**として、①スクリプトを再実行して
  凍結ファイルと diff がコメント以外ゼロであること、②CI で
  `-f games=100 -f baselines=estimator_vN` が 50%±10 に入ることを確認する
  （挙動が同一である担保。ローカル `arena 20` は ±22pt で検出力がなく、
  v14 では 7勝13敗と出た。同一コードでも 100 局で 44% まで振れる）。
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

| 段階 | 中身 | 状態 |
| --- | --- | --- |
| ① | `opp_move_weight` の NN 化（`opp_move_nn.rs`、26特徴量） | **本番統合済み**（v9 凍結）。2026-08-21 に教師データごと刷新 |
| ② | 信念ネット（`belief_nn.rs`、マスごとの占有確率） | **保留**。較正は粒子より当たるが勝率中立。既定 0 のノブとして残置 |
| ③ | 粒子上の value ネット（`value_nn.rs`） | **本番統合済み**（v10 凍結）。2026-08-21 に再学習 |
| ④ | フルRL | 未着手 |

玉位置ネット（`king_belief_nn.rs`）は②の枠外で、候補集合内 softmax として
2026-07-31 に採用済み（`TSUITATE_KING_NET_PROJ`）。

**運用ルール**: 定石（`joseki.json`）を更新したとき、および**方策が大きく変わったとき**は
opp_move / value の教師データを再生成する（教師データの鮮度が本体で、特徴量の
追加はそれ自体では効かない、というのが 2026-08-21 の刷新の結論）。凍結版が共有 NN を
呼ぶ問題は固定コピー（`opp_move_nn_v25.rs` / `value_nn_v22.rs`）で解消済みなので、
**再学習で凍結版は動かない**。

再学習の手順・各段階の全計測・限界の認識は **`docs/nn-direction.md`**
（段階②は `docs/nn-stage2-belief-net.md`、段階③フェーズ1は `docs/nn-value-phase1.md`）。
学習パイプライン本体は別リポジトリ `~/Develop/tsuitate-nn`。

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
- **比較の基準は heuristic ではなく凍結版**。heuristic への勝率は飽和していて
  改善の検出力がない。また非推移性（v2 に勝つが v1 に負ける）を検出するため、
  **ガントレットで凍結版に勝ち越すことを合格条件とする**（既定の対象は v9 以降）
- 同一戦略同士は約50%になる（1000局で確認済み）。**同一コードでも 100 局では
  44% まで振れる**ので、版の比較は必ず `match_seed` でペアにする
- 時間切れは負けとして数え、思考時間の統計（平均/p99/最大）も出す。
  「遅くなったが勝率が上がった」偽の改善はここで検出する
- **アリーナ勝率だけで評価項の良し悪しは測れない**（終局理由が偏るため）。
  合否はシナリオ suite ＋ 機構指標と併せて見る。詳しい運用・シャード分割・
  ペア差の分散推定・SPSA の GCE 運用は **`docs/arena-ops.md`**

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
