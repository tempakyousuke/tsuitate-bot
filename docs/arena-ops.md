# アリーナ運用（arena / checkpoint_arena / tune）

強さの検証と自動チューニングの詳細。CLAUDE.md から切り出した
（起動の要点だけは CLAUDE.md の「強さの検証」節にある）。

## 審判と公平性

- `bin/arena.rs` がサーバーと同じ裁定（反則で手番維持・10回で反則負け・王手宣言を両者へ・
  詰み/ステイルメイト終局）をローカル再現し、戦略同士を対戦させる
- 各戦略に渡るのは PlayerView 相当と観測イベントのみ。**フル盤面は審判しか見ない**
  （observation.rs にない情報を使わない、という公平性の担保はこの構造で守る）
- 同一戦略同士は約50%になる（1000局で確認済み）。参考値: estimator vs heuristic は
  200局で勝率86.5%±4.7%、平均反則 2.2 vs 9.0（2026-07 時点）
- **比較の基準は heuristic ではなく凍結版**。heuristic への勝率は
  飽和していて改善の検出力がない。また非推移性（v2 に勝つが v1 に負ける）を検出するため、
  ガントレットで凍結版に勝ち越すことを合格条件とする。対象は **v9 以降**
  （v6〜v8 は 2026-07-31 に既定から除外。v6 は 84〜90% で heuristic と同様に飽和）
- フィッシャー時計 1000秒+3秒 をシミュレートし時間切れは負け（本番サイトは300秒+3秒。
  `-f clock="300+3"` / `ARENA_FISCHER_*` で本番相当に切り替えて検証できる。
  既定予算2000msなら本番条件でも時間切れは出ない）。`choose()` の壁時計を消費し、
  加算は受理された手の後のみ。思考時間の統計（平均/p99/最大）も出力するので、
  「遅くなったが勝率が上がった」偽の改善はここで検出する。戦略側は粒子数・
  リプレイ回数・時間予算で自ら打ち切る構造を保つこと（上限は思考予算 ≒ 既定2秒）


## `bin/arena` と `arena.yml`

- `cargo run --release --bin arena -- [対局数] [候補] [基準1] [基準2] ...` — 戦略同士の対戦。
  基準を複数並べるとガントレット（候補が各基準と対局数ずつ対戦）。
  戦略の変更は必ずこれで**凍結版ガントレット**（`src/frozen/` の `estimator_vN`）に有意に
  勝ち越すことを確認する。**既定の対象は v9 以降**（2026-07-31 ユーザー判断:
  v6〜v8 は勝率が飽和して検出力がないため既定から除外。コードは残してあるので
  明示指定すれば対戦は可能）。
  50%付近の信頼区間は 100局で±10pt / 200局で±7pt / 1000局で±3.1pt。当面（開発最初期）は
  100局を既定とし、結果が信頼区間内で判定できない僅差のときだけ局数を増やす。
  **実行はローカルでなく GitHub Actions で行う**（`.github/workflows/arena.yml`）。
  起動は2通りで、**通常のコード push では走らない**:
  1. 手動: 対象ブランチを push して
  `gh workflow run arena.yml --ref <ブランチ> -f games=100 -f candidate=estimator -f baselines="estimator_v6 estimator_v7"`。
  2. リクエストファイル: `.github/ci/arena.request.json` を書いてそのファイルを含む
  commit を push する（例は `arena.request.example.json`。エージェントが
  `gh` 無しで回す用。JSON のキーは手動起動の inputs と同じ）。
  計測後にリクエストファイルを消す push も `paths` フィルタに一致して
  起動するが、**plan が「削除された push」と判定して全ジョブをスキップ**
  するので緑のまま終わる（後片付けで赤い run が残らない。scenario.yml も同様）。
  「基準 × シャード」の matrix に分割され（`-f shards=4` 既定。単一基準の
  200局も4ランナーに並列化される）、**分割は合計が正確に `games` になる偶数割り**
  （`scripts/ci/split_arena_games.py`、各シャード偶数 = 先後均衡・差≤2。
  奇数指定だけ全体 +1。〜2026-09-02 は ceil の偶数化で 100→104・600/8→608 に
  膨らんでいた = 過去記録の「各104局」はこの由来）、総合結果は **aggregate ジョブのサマリー**
  （および artifact `arena-combined`）に合算表で出る。シャード個別は
  `arena-result-<基準>-s<n>` / `arena-records-<基準>-s<n>`。
  `-f match_seed=<数>` で対局条件列を決定論化できる（アブレーション比較用。
  同じ入力なら版をまたいで同じ条件列。シャード間は自動で+shardずらし）。
  **`match_seed` 指定時は1行=1対局の記録 `arena-games.jsonl` も出る**
  （`ARENA_GAMES_JSON`。artifact `arena-result-*` と `arena-combined` の
  `all-games.jsonl`）。`-f pair_with=<対照のArena実行ID>` を付けると
  aggregate ジョブが対照の記録を取ってきて
  `checkpoint_arena arena-var` を回し、**局ごとのペア差**（Var・CI・MDE・
  var·CPU秒・安全性の共同指標）をサマリーへ出す（issue #19 の P0。
  対照と候補で match_seed / 局数 / shards / 相手 / 時計 をすべて揃え、
  違うのは `cand_env` や `candidate` だけにすること）。
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
  **候補側だけに効かせたいときは `-f cand_env="K=V K=V"`**（リクエストファイルなら
  `"cand_env"`。ローカルは env `ARENA_CAND_KNOBS`。issue #21、2026-08-24）:
  値はプロセス env でなく `StrategyConfig` として候補 instance にだけ渡るので、
  env を読み続ける凍結相手は必ず既定値のまま動く（凍結版を候補にしたときは
  config を尊重しないので起動時にエラー）。**綴り間違い（戦略が読まないキー）は
  起動時エラー、実効値が変わらなかったキーは警告**が出る（`config::check_overrides`）。
  seed の扱いはノブの有無で変わらない（`make_with_config` が `Option<u64>` を素通し。
  `Some(0)` へ落とすと候補だけ全対局が同じシードになる）。
  凍結版は**凍結時点で読んでいた env を今も読む**（実測 2026-07-26。
  一覧は `frozen::env_keys_in_source(name)` で機械的に取れる）:
  - `TSUITATE_THINK_BUDGET_MS` — **v6〜v11 の全凍結版が読む**。思考予算の
    スイープを `-f env=` でやると両側の予算が動いて比較にならない。候補側だけ
    予算を変えたい場合は現行 `strategy.rs` だけが知る新しい名前を足すこと
  - `TSUITATE_JOSEKI` / `TSUITATE_EPS_PHYS` / `TSUITATE_DISABLE_DEFEND_GUIDE` /
    `TSUITATE_ENABLE_HANG_RISK` / `TSUITATE_FILTER_DEBUG` / `TSUITATE_DEBUG_CHECK`
    — v7 以降（版により差あり）
  - `TSUITATE_VALUE_NN_W` / `TSUITATE_CAPTURE_BET_VAR_W` / `TSUITATE_CHECKER_REMOVAL_W`
    — v11 のみ。**v11 を基準にした w スイープでは基準側も反応する**
  - `TSUITATE_MATE_*_W` / `TSUITATE_KING_HOLE_W` / `TSUITATE_TAINT_KING_FIX` /
    `TSUITATE_KING_REPEAT_FOUL_W` はどの凍結版も読まない（＝候補側だけに効く）
  - `TSUITATE_KING_CAND_ATTACK_W` / `_CHECK_W` / `_ATTACK_GATE` /
    `TSUITATE_LANDING_SUPPORT_W` / `TSUITATE_KING_PROX_EXCLUDE_SELF` /
    `TSUITATE_KING_BELIEF_PROX_W` / `TSUITATE_CHECK_SAFE_RESOLVE` —
    **v14 が読む**。v14 を基準にした w スイープでは基準側も反応する

  各シャードは `bin/analyze` も走らせ、**被詰めろオラクル**（相手に
  1手詰めを与えた局面数）・**詰み経済セクション**（issue #28。終局の攻め／受け・
  手数調整した露出・詰め手の排他的分類・受けの漏斗・列挙コスト）・只取られ・
  反則内訳をジョブサマリーへ出す（集計行だけを拾う grep なので、
  **analyze に集計行を足したら arena.yml のフィルタも直す**）。
  実測 2026-07-16: vs v6 で nofoul 86.2%±4.8 / check_nofoul 59.5%±7.1 —
  反則経済に36ptの伸びしろが実在する（現状の評価構造のスカラー係数調整では
  届かないことが tune-round3/4 の不発で確定。tuning/README.md 参照）。
  C-7凍結後（2026-07-19）に再測定しても vs v7 で通常51.9%→nofoul 83.2%と
  約31ptのギャップが温存されており、反則マスを記憶して割り引く4種の対策
  （粒子ガイド2種・prior_legal/最終p_legal直接制約2種）はいずれも
  有意な改善に届かなかった（docs/c8-direct-synthesis.md 参照）。この
  ギャップは今のところ埋まっていない

## checkpoint arena（issue #19、破滅検出器として不合格）

- `cargo run --release --bin checkpoint_arena -- <extract|run|compare|report|arena-var>` —
  **checkpoint arena**（issue #19、2026-08-23 に P0 の計測経路として実装）。過去の
  実戦の途中局面から bot 同士で終局まで指し継ぎ、候補と対照を**同じ局面でブロック化**して
  比べる。位置づけは「通常 arena へ送る前に**明確な悪化**を安価に除外する破滅検出器」で、
  **採否は今までどおり arena.yml のガントレット**。較正が済むまで informational。
  設計・実測・撤退判断は `docs/checkpoint-arena-p0.md`。
  **2026-08-24 の CI 規模実測（700ms・64局面×4seed・合計43 CPU時間）で、
  この構成は破滅検出器として不合格**: 予算を 700ms へ揃えた通常 arena の
  **−14.4pt [−27.9, −1.9]** を、checkpoint は **+8.0pt [+0.2, +16.0]** と
  **逆向きに**報告した（反則/局も arena +1.37 の悪化 vs checkpoint −0.28 で逆）。
  統計・分解能・予算のどれでもない（A/A は2本で +7.0/+0.0pt、偽陽性率は
  n=64 で 4.5〜6.7% = 名目どおり、SE ±3.4〜4.7pt、700ms と 2000ms で
  arena 側は一致）。**コスト優位も無い**（var·CPU秒 42 vs 31〜57 で互角）ので
  issue #19 の撤退条件「通常 arena に明確なコスト優位が無い」に該当する。
  一方 **v14 vs v10 級の大差 +24.2pt は正しく分離できる**ので配管は壊れていない。
  較正点は1つなので「常に符号が逆」とは一般化しない。要点:
  - 裁定は `selfplay::play_continuation` = 通常 arena と**同じ関数**（MAX_PLIES・反則上限・
    王手/捕獲通知・終局判定を共有）。**時計は無効**（途中局面の残り時間は復元できない。
    本番相当で時間切れ0の実測があるので落としてよい）だが**思考時間は必ず記録する**
  - **v1 の checkpoint は手番境界のみ**（その手番でまだ反則を試していない時点）。
    `foul_tried` は仕様上必ず空になる。`foul_tried.len()` は `check_foul_prior_boost` が
    読むので、反則後 checkpoint を扱うなら manifest へ明示的に保存すること
  - デッキは **manifest（JSON）＋元 KIF**。復元は `scenario_core::replay`、抽出は
    `truth_replay::for_each_decision_full`。両経路が同じ状態を作ることは
    `cargo test kif_roundtrip` が常時検査する。KIF なので回帰した checkpoint を
    そのまま `bin/scenario <path>.kif diag` / `bin/rank_probe <path>.kif <ply>` へ流せる
  - 適格条件: 手番境界・初手を除く・元対局がそこから `--min-remaining`（既定20）手以上
    続いた・終局済み/記録不備/時間切れ局を除く・**原則1棋譜1checkpoint**。層化は
    先後 / 早中盤(ply 0〜49)・中盤(50〜89)・終盤(90〜) / 通常・被王手 / 反則0・あり /
    元対局の結末（優劣の粗い代理）。dev/validation は**元対局単位**で割る
  - **control をキャッシュしない**。両 arm は同じジョブ・同じスロットで背中合わせに走らせ、
    checkpoint ごとに AB/BA を均衡させる（`compare` が「先に走った arm − 後」の実行順効果を出す）。
    env は `OnceLock` なので arm ごとに**子プロセス**（`unit`）として起動する。
    `--control-bin` / `--candidate-bin` を分ければそのまま base/target 二重ビルドにも使える
  - `--shared-prewarm` は「同じバイナリ・同じ env・同じ推定器」のときだけ有効
    （それ以外は理由つきで無効化され、JSONL の `prewarm_shared` / `prewarm_reason` に残る）
  - `compare` は**元対局単位の cluster bootstrap**（seed を独立標本として数えない）・
    分散成分 σ_b²/σ_w² と ICC・「元対局数 × seed 数」の **MDE 表（α=0.05 / power=80% の
    2.80·SE。CI 半幅 1.96·SE とは別物）**・非パラメトリックな power simulation・層別 delta・
    コスト分解・大きな回帰の再現コマンドを出す。**安全性の共同指標**
    （反則/局・被王手中反則・反則負け率・継続手数・手数上限率・終局理由・思考時間）も
    同じペア差で出すが、**反則減だけで「強くなった」とは判定しない**。
    入力は厳格に検査する（schema・experiment・deck_hash・相手・予算・commit の一致、
    arm 名、重複、checkpoint ごとの seed 数。不一致は失敗。`--allow-incomplete` で警告に落とす）
  - **seed は 2 の倍数で取る**（`run` は既定 2 で奇数を拒否）。arm 順が
    `(checkpoint 番号 + seed) % 2` なので、同じ checkpoint の seed 2k/2k+1 は必ず逆順になり、
    **AB/BA の均衡が cluster の内側で閉じる**。s=1 だと実行順効果が cluster 平均に残り、
    実測で分散が 5〜8倍違った。**この反相関は意図した設計なので、統計側も
    「連続する2 seed の AB/BA 平均 = 1 replicate」を単位にする**（生の seed を
    独立標本として `σ_w²/s` で外挿してはいけない）。seed 数の効果を測るには
    **4 seed 以上**（2 replicate 以上）が要る
  - `compare` の power simulation は**本番と同じ percentile cluster bootstrap CI** を
    当てる（正規近似の z 検定では解析 MDE と同じ経路になり裏取りにならない）。
    主 CI の percentile も `--alpha` に連動する
  - **shard 欠落の検出には `compare --deck <manifest> --split <split>` が要る**
    （入力に現れた checkpoint 同士を比べるだけでは、artifact ごと欠けたときに
    「揃っている」ように見える）。CI は必ず渡し、label ごとの JSONL 本数と
    summary の数も検査して、欠けたら `INCOMPLETE.txt` を出してジョブを失敗させる
  - **既知の arena 差を較正に使うときは、同じ candidate / control / opponent / 予算で
    測り直した値だけを渡す**。CLAUDE.md / `docs/knobs.md` に残っている −12.8 / −8.5 / −7.4pt は
    当時の main・別の対戦条件で測った値なので流用できない（PR #20 レビュー指摘）
  - **arm 固有ノブは config で渡す**（issue #21、2026-08-24 に解消）。
    `--control-env` / `--candidate-env` は名前に反して**プロセス env を触らない**:
    値は `StrategyConfig` として arm の戦略にだけ渡るので、env を読み続ける
    凍結相手は arm によらず既定値のまま動く。候補戦略が凍結版のときはノブを
    渡せない（config を尊重せず黙って無視するので起動時に止まる）。
    両 arm に同じ値を渡す `--budget-ms` は従来どおり子プロセスの env。
    **較正が済むまで既定 matrix からは env 実験を外したまま**にしてある。
    JSONL に `arm_knobs` / `arm_config` / `opponent_config` を残し、`compare` は
    「相手の実効設定が両 arm で一致」を指紋で検査する。`opponent_config` は
    **凍結版なら `frozen::behavior_fingerprint`**（版のソース・その版が読む env の
    実効値・共有モデルの pin から作る）で、現行 config の指紋ではない。
    綴り間違いのノブは起動時エラー、「ノブが違うのに実効設定が同じ」は
    `compare` が止める
  - **seed は 4 以上の偶数**が P0 の条件（2 seed は AB/BA が閉じる最小構成だが
    replicate 間分散が同定できない）。`compare` は replicate が1つのとき
    seed 数の外挿を出さない
  - JSONL の schema は **3**。schema 1（相手の実効 env が未記録）と
    schema 2（arm 固有ノブをプロセス env で渡していた時期）は集計から明示的に弾く。
    **`compare` の summary JSON も同じ契約**で、`report` が schema 1/2 の summary を
    拒否する（撤回済みの数字が横断表へ戻らないように）
  - 実行は CI（`.github/workflows/checkpoint-arena.yml`、**通常のコード push では走らない**）。
    `gh workflow run checkpoint-arena.yml -f arena_run_id=<Arena実行ID> -f seeds=4`、
    `gh` が無ければ `.github/ci/checkpoint-arena.request.json` を置いて push（削除の push は
    全ジョブがスキップされる）。**デッキは arena 記録から作るのを既定にする**
    （局面分布が実際の対局分布と一致する。`arena_run_id` を渡せば追加の対局コストはゼロ）
  - **`arena-var` は通常 arena 側の相方**（2026-08-24）。`ARENA_GAMES_JSON` が出す
    1行=1対局の記録を2本ぶん受け取り、同じ `match_seed` の局ごとに
    `delta = 候補 − 対照` を取って **`Var(ペア差)`**・CI・MDE・必要 N・
    var·CPU秒・安全性の共同指標を出す。checkpoint 側の効率比はこの実測が
    無いと `Var(delta)=0.5` の仮定に乗ったままで、**同じ実行が
    `--known-arena-delta` に渡す既知値にもなる**。CI では `arena.yml` の
    `-f pair_with=<対照のArena実行ID>` が候補側 run の中でこれを回す。
    ガントレットの記録は `--baseline` で1マッチアップに絞る
  - **`arena-balance` は issue #40 の opponent-balanced 合算器**（2026-09-01 実装。
    まだ判定実績なし）。2相手ぶんの対照・候補 games.jsonl を受け取り、相手ごとに
    局ペア差を作って **`(Δv13 + Δv14) / 2` を層化 bootstrap**（各相手の内側で局を
    引き直す）で出し、**事前登録した門**（合算 ≥ +0.04・CI 下限 > 0・相手別符号
    veto・反則/局 +0.3 以内・時間切れ0・思考平均 +100ms 以内）を
    **fail-closed（不通過なら exit 3、`--allow-incomplete` で警告へ降格）** で
    判定する（`check_policy combined` と同じ契約）。
    **判定を変えられるパラメータは全部が事前登録の定数**（PR #41 レビュー4巡目）:
    局数 600（`BALANCE_EXPECT_GAMES`）・shards 8・base seed（v13=20260910 /
    v14=20260909）・相手集合 {v13, v14}・alpha 0.05・bootstrap 反復の下限 10000。
    `--expect-games` / `--expect-shards` / `--expect-seeds` / `--expect-opponents` /
    `--alpha` / `--boot` の既定はこの定数で、**外した指定は判定不能**（informational
    な集計はできるが「通過」を出せない。`--expect-opponents` を空にして検査を
    切ることもできない）。`--allow-incomplete` で降格した不完全入力・A/A =
    処置なしも判定不能。
    **処置ノブのような P1 後に決まる可変部分は validation manifest が一次資料**:
    計測前に commit した manifest（例は `.github/ci/balance-manifest.example.json`。
    `cand_knobs` に加えて `candidate` / `think_budget_ms` が必須）を指すと、
    `bin/arena` が起動時に「実効ノブ == manifest の cand_knobs（対照は空）・
    candidate 名と実効予算の一致・**オラクル無効**」を検査したうえで
    **games.jsonl の全行へ manifest の sha256 を焼き込み**、合算器は
    `--manifest <path>` で同じファイルの指紋と全行の一致（違えば die）・期待ノブの
    完全一致・必須フィールドを要求する。manifest 無しの集計は判定不能。
    **held-out の4 run（対照→候補 × v13/v14）は同一 commit が必須なので、
    request ファイルの書き換え push では作れない**（push ごとに commit が変わる。
    PR #41 レビュー5巡目）: manifest を commit した ref へ
    `gh workflow run arena.yml -f balance_manifest=<path> -f games=600 -f shards=8
    -f match_seed=<相手別seed> -f baselines=<相手>`（候補側は
    `-f cand_env="<ノブ>" -f pair_with=<対照のrun ID>` を追加）を4回起動する。
    plan が起動時に manifest との一致（games/shards/candidate/相手別 seed・
    1相手ずつ）と**合算器へ渡せる run であること**（cand arm は pair_with 必須・
    ctrl arm は禁止、oracle / 両側 env / 時計変更なし、cand_env == manifest の
    cand_knobs。判定不能になる run に claim を消費させない）を検査し、
    **取り直しは git タグの台帳で拒否する**（PR #41 レビュー6〜7巡目:
    `run_attempt==1` では「もう一度 dispatch した新しい run」を検出できない）:
    `balance-claim/<manifest指紋16>/<arm>-<相手>` タグを **plan が計測前に取得**
    （annotated tag の message に run_id / attempt を記録 = どの実行が slot を
    取ったかの不変記録。同時 dispatch の後着は ref 作成の 422 で対局前に落ちる
    ので、未claim の完走 artifact は生まれない）。**re-run attempt は plan が
    claim 取得前に拒否**（解放済み slot を再claimして1局も指さず埋める穴。
    レビュー8巡目）。**cand arm の claim は「対照 claim の owner run ==
    pair_with かつその run が success 完了かつ head_sha == この run の commit」を
    確認してから取得**（typo・実行中・失敗した・**別 commit** の対照を指した
    candidate が claim を消費しない = 「対照を先に完了・同一 commit」の機械保証。
    commit 不一致は合算器も die するので、claim だけ消費して判定に使えない
    記録を作らせない。レビュー9巡目）。**解放は「対局が完走しなかった attempt 1 の run」だけ**
    （release-claim ジョブ。claim 名は取得より先に output へ書くので、plan が
    取得後に落ちた経路も解放できる。aggregate の失敗では解放しない = 計測は
    存在しているため）。**合算器も台帳を照合する**: `--manifest` 指定時は
    `--repo`（既定 `.`）の git から claim タグを fetch して読み、arm × 相手の
    入力 run が台帳の owner と違えば die・タグが無ければ判定不能（台帳外の
    artifact で判定させない）。タグは永続なので事後監査もできる。
    **実験条件も事前登録と照合する**: 実効予算は両 arm・両側とも 2000ms
    （`BALANCE_BUDGET_MS`。arm 間の一致だけだと「両方 700ms」が通る）・両側 env
    なし・**診断オラクル（games.jsonl **schema 5** で `oracle` 列を必須記録）は
    非空なら die**（審判が候補の反則を握りつぶす nofoul 診断 run を held-out として
    通せてしまうため）。予算・env の逸脱は判定不能。
    入力の同一性検査も fail-closed: arm 内・相手内の一意性（`assert_uniform`）・
    **相手をまたいだ arm 設定の一致**（candidate / clock / commit / 予算 /
    cand_config / cand_knobs / shared_env）・**arm × 相手ごとに base は1つだけ・
    shard 集合は 0..N の完全な集合**（games.jsonl の `match_seed_base` /
    `match_seed_shard` 列。記録の `match_seed` はシャードずらし＋基準 XOR 済みの
    **実効値**なので一意性検査には使えない）・**arm × 相手ごとに実行の識別子
    `(run_id, run_attempt)` は1つだけ**（games.jsonl **schema 4** で必須化。
    **base は実験条件であって実行の識別子ではない**: 同じ base で取り直した
    2 run から shard を半分ずつ選んでも base 1値・shard 完全・局数一致で通って
    しまう。CI は `GITHUB_RUN_ID` / `GITHUB_RUN_ATTEMPT`、ローカルは
    `ARENA_EXPERIMENT_ID` が必須 — 無いと `ARENA_GAMES_JSON` 指定の run は起動時に
    落ちる）・**候補行の `pair_with`（`-f pair_with=` で記録）は対照の run_id と
    一致し、held-out 判定では必須**（紐の無い行は判定不能 = 後から別の同一 seed・
    同一 commit 対照へ差し替えられる穴を塞ぐ。不一致は die）・**両 arm とも
    `run_attempt == 1`**（re-run attempt = 取り直しなので判定不能。`pair_with` は
    run_id しか持たず attempt を区別できないため 1 に固定して組を一意にする。
    shard が落ちた run は re-run でなく新しい run として取り直す）・
    **arm 間は同一 candidate 名・対照は W=0
    （cand_knobs 空）・同一 commit**（commit 不一致に override は無い）。
    run 混入・pair_with 不一致・指紋不一致・oracle 非空は `--allow-incomplete` でも
    降格しない。
    予算不一致の override も無い。**平均評価粒子数の門（対照比 −10% 超で中止）は
    verdict に入っている**（PR #41 レビュー7〜8巡目）: 粒子数は games.jsonl に
    無いので、同じ run の `arena-records-*` を `--records-control` /
    `--records-candidate` で渡す（`chose.debug.unique_particles` の平均。
    展開先ディレクトリごと渡せる）。**record は由来 meta を持ち、games.jsonl と
    一対一で照合される**（`selfplay::write_record` が match ヘッダ直後へ
    run 識別・baseline・base/shard・game_no・manifest 指紋の meta 行を書く。
    (baseline, base, shard, game_no) の重複・鍵集合の不一致・別 run の record・
    manifest 指紋不一致・壊れた JSON 行・end の無い未完 record は **die**）。
    **records 入力の無い集計・meta の無い/不完全な record・`unique_particles` の
    無い chose 行は判定不能**（**定跡手は例外**: 評価を回さない正当な非評価手
    なので `chose.debug` に `{"joseki": true}` が入り、集計対象外 = 欠測に
    数えない。PR #41 レビュー9巡目）

## `bin/tune`（SPSA 自動チューニング）

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
  対局ループは `selfplay.rs`（arena と共用）。**長時間ランはローカルでなくGCEで回す**（下記）。
  **`TUNE_OBJECTIVE=scenario` でシナリオ不合格計を目的関数にできる**（2026-07-31 追加）:
  位置引数2がシナリオあたり試行数（既定10）になり、score = 1 − 不合格計/(件数×試行数)。
  対象は `TUNE_SCENARIOS`（既定は悪手8件。`bad=` 必須）。手動 env スイープ
  （hand_option_w の 39→7 のような w 選定）の自動化で、試行シードは 0..trials の
  固定列なので f+/f− は自動的に共通乱数ペアになる。(棋譜,手番) グループごとに
  スレッド並列＋prewarm 継ぎ足し共有。**悪手8件は実質1局の8局面なので
  TUNE_PARAMS で数次元に絞って使い、採用判定は従来どおり CI ガントレット**。
  運用手順は `scenarios/README.md` の「SPSAチューニング（シナリオ目的）」。
  両モード共通で、調整対象ノブの `TSUITATE_*` env が立っていると起動時にエラーになる
  （env が摂動を潰して勾配が死ぬ罠の検査）。SPSA は `EvalParams` を直接渡すので
  config を経由しない（`apply_param_overrides(params, &EnvSource)` が上書きを当てる）

## 長時間ランは GCE で回す

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

