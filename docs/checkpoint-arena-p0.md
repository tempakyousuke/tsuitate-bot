# checkpoint arena — P0（検出力・費用の実測と撤退判断）

2026-08-23。issue #19 のレビュー反映版に沿った P0 の実装と、その計測経路。

**この文書は進行中の記録**。P0 の結論（続行 / 安全性ゲートへ縮小 / 撤退）は
「撤退判断」節に、確定したところまでを書く。

## 位置づけ（変えないこと）

- 第一候補は「小さな改善を証明する短縮 arena」ではなく、**通常 arena へ送る前に
  明確な悪化を安価に除外する破滅検出器**
- **最終採否は引き続き通常 arena**（`arena.yml` のガントレット）。checkpoint arena は
  較正が済むまで informational
- 同じ検出力を得る CPU コストが通常 arena と大差なければ実装を打ち切る
- scenario suite・arena・凍結版ガントレットを置き換えない

分散低減の主因は「完全な共通乱数」ではなく **同じ局面でブロック化すること**。
方策が分岐すれば乱数消費も変わるので、共通乱数とは主張しない。

## 実装

| 部品 | 場所 | 役割 |
| --- | --- | --- |
| 継続対局 | `selfplay::play_continuation` / `StartState` | 「初期盤面・両者ログ・累積反則数・絶対手数」から指し継ぐ。裁定（MAX_PLIES・反則上限・王手/捕獲通知・終局判定）は通常 arena と**同じ関数**を通る |
| 両側ログの抽出 | `truth_replay::for_each_decision_full` | `game:end` の真実から両者の観測列・累積反則数・絶対手数・その手番の反則試行数を渡す |
| デッキ | `checkpoint.rs` | manifest（JSON）＋元 KIF。復元は `scenario_core::replay` |
| CLI | `bin/checkpoint_arena` | `extract` / `run` / `unit` / `pair` / `compare` / `report` |
| CI | `.github/workflows/checkpoint-arena.yml` | 手動起動のみ。通常の push では走らない |
| 通常 arena 側のペア差 | `bin/arena` の `ARENA_GAMES_JSON` ＋ `bin/checkpoint_arena arena-var` | 同じ `match_seed` の2本を**局ごとに**突き合わせて `Var(ペア差)`・既知 arena 差・var·CPU秒 を出す |

### v1 は手番境界に限定

checkpoint は「直前の受理手が完了し、次の手番がまだ反則を一度も試していない時点」だけ。
`scenario_core::replay(kifu, ply)` の自然な境界と一致するので `foul_tried` は仕様上必ず空になり、
注入済みの反則手を再試行できてしまう問題を避けられる。
`foul_tried.len()` は `check_foul_prior_boost`（2026-08-20 採用）が読むので、
復元漏れは評価そのものを変える。同一手番内の反則後 checkpoint を扱うときは
manifest へ `foul_tried` を明示的に保存し、MyFoul 観測・累積反則数との整合テストを足すこと。

`restore()` は ply が手番境界でなければエラーを返す（`extract` は抽出時にも全件検査する）。

### 保存形式は KIF ＋ manifest

抽出元は `truth_replay` の真実だが、デッキには元 KIF を置き、復元は
`scenario_core::replay` に任せる。回帰した checkpoint をそのまま
`cargo run --release --bin scenario -- checkpoint-arena/games/<id>.kif diag` /
`cargo run --release --bin rank_probe -- checkpoint-arena/games/<id>.kif <ply> 3` へ流せるので、
「再現コマンドつき」の要件が自動的に満たされる。
**両経路が同じ状態を作ること**は `checkpoint::tests::kif_roundtrip_matches_truth_replay` が検査する
（盤面 fingerprint・手数・両者ログ・累積反則数・手番）。

### 時計は落とす / 思考時間は必ず記録する

途中局面からの残り時間は復元できない。本番相当 300秒+3秒・100局で時間切れ0の実測があるので
`play_continuation` は時計を無効（`clock_ms = None`）にする。ただし
**1手ごとの思考時間は必ず記録する**（arena が「遅くなったが勝率が上がった」偽の改善を
弾いているのはこの統計なので、落とすと同じ穴が開く）。

### 決定論は要求しない

推定器のリプレイ・再生成・変異救済はすべて壁時計デッドラインで打ち切る（`Instant::now()`）。
同一入力でのバイト一致は原理的に書けないので、代わりに次を検査する。

- 状態復元の一致（KIF ↔ truth_replay、上記）
- 裁定の共有（`selfplay::tests::continuation_from_initial_matches_normal_selfplay` /
  `continuation_carries_absolute_counters`）
- 集計器の決定論（`compare` は入力 JSONL から決定論的、bootstrap も固定 seed）
- A/A の paired delta が 0 中心か（実測、下記）

**control のキャッシュはしない**。機械負荷が変わればキャッシュ済み control は
対照として弱くなる（負荷 → 粒子数 → 強さ、が直結する）ので、両 arm は
**同じジョブ・同じスロットで背中合わせに**走らせ、checkpoint ごとに AB/BA を均衡させる。
`compare` は「先に走った arm − 後」の実行順効果を出すので、交互実行が効いているかを毎回確認できる。

### arm の分離は子プロセス

`TSUITATE_*` は `OnceLock` で1プロセス1回しか読まれないので、env の違う2つの arm を
同一プロセスで切り替えられない。`run` は各 arm を**子プロセス**（`unit`）として
同一 runner 上で交互に起動する。`--control-bin` / `--candidate-bin` を分ければ
そのまま base SHA / target SHA の二重ビルド（P3）にも使える。

### prewarm 共有は opt-in

`--shared-prewarm` は次がすべて同一のときだけ有効になる（それ以外は理由つきで無効化される）。

- 同じバイナリ
- 同じ env（arm ごとに異なるグローバル `OnceLock` があると共有できない）
- 同じ推定器（戦略名）

有効時は `pair` サブコマンドが1プロセスで prewarm し、`Strategy::clone_boxed` の
スナップショットを両 arm へ配る。共有の有無は JSONL の `prewarm_shared` に必ず残る。

## 使い方

**いま実際に動く例だけを載せる**（PR #20 4回目レビュー指摘3）。
env 実験（`--candidate-env`）は issue #21 で安全に渡せるようになったが、
較正が済むまで既定 matrix からは外してある。

```bash
# デッキ抽出（arena 記録から。原則1棋譜1checkpoint・層化・手番境界のみ）
cargo run --release --bin checkpoint_arena -- extract \
  --records records --out checkpoint-arena \
  --opponent estimator_v14 --min-remaining 20 --limit 64

# A/A（同一設定の両 arm）。seed は 2 の倍数、seed 数の効果まで測るなら 4 以上
cargo run --release --bin checkpoint_arena -- run checkpoint-arena/deck.json \
  --split dev --seeds 4 --jobs 3 --experiment aa \
  --jsonl out/aa.jsonl

# 凍結版どうし（env を使わないので相手への漏れが起きない）。
# 測っているのは (v13 vs 相手) − (v12 vs 相手) であって v13 vs v12 ではない
cargo run --release --bin checkpoint_arena -- run checkpoint-arena/deck.json \
  --split dev --seeds 4 --jobs 3 --experiment v13_v12_vs_opp \
  --control-strategy estimator_v12 --candidate-strategy estimator_v13 \
  --jsonl out/v13_v12_vs_opp.jsonl

# ペア集計（--deck は shard 欠落の検出に必要）
cargo run --release --bin checkpoint_arena -- compare out/aa.jsonl \
  --deck checkpoint-arena/deck.json --split dev \
  --markdown out/aa.md --json out/aa.summary.json

# 実験横断（既知値が2件以上あるときだけ符号一致・順位相関を出す）
cargo run --release --bin checkpoint_arena -- report out/*.summary.json
```

`--known-arena-delta` は、**同じ candidate / control / opponent / 予算で arena を
取り直した差**にだけ付ける。CLAUDE.md に残っている過去の値は流用できない。

### 通常 arena 側の Var と既知差（1回のペアで両方そろう）

「checkpoint arena は通常 arena の何倍効率がよいか」は、arena 側の
`Var(ペア差)` を実測しないと決まらない（従来の参考値 183 は
`Var(delta)=0.5` = **ペアリングが全く効かない**という仮定に乗っていた）。
同じ実行が較正の既知値（`--known-arena-delta` に渡す値）にもなる。

`bin/arena` は `ARENA_GAMES_JSON` で**1行=1対局**の JSONL を書く
（突き合わせのキーは `(baseline, match_seed, game_no)` なので
`ARENA_MATCH_SEED` と併用しないと書かずに理由を出す）。

```bash
# 対照
ARENA_MATCH_SEED=20260824 ARENA_GAMES_JSON=control.jsonl \
  cargo run --release --bin arena -- 104 estimator estimator_v14
# 候補（違うのはノブだけ。相手・時計・局数・match_seed は揃える）
ARENA_MATCH_SEED=20260824 ARENA_GAMES_JSON=cand.jsonl \
  ARENA_CAND_KNOBS=TSUITATE_DROP_PROBE_REPEAT_GATE=1 \
  cargo run --release --bin arena -- 104 estimator estimator_v14
# 局ごとのペア集計
cargo run --release --bin checkpoint_arena -- arena-var \
  --control control.jsonl --candidate cand.jsonl --json arena-var.json
```

CI では **`arena.yml` の `pair_with`** に対照の実行IDを入れると、候補側の run の
aggregate ジョブが対照の `all-games.jsonl` を取ってきて `arena-var` を回す
（`match_seed` 必須。相手・局数・shards・時計はすべて揃えること）。
ガントレットの記録は `--baseline` で1マッチアップに絞る。

`arena-var` が止めるもの: ペアにならない対局（`--allow-incomplete` で警告に落とせる）・
先後の食い違い・相手や時計の混在・同じ `(baseline, match_seed, game_no)` の重複行。
安全性の共同指標は checkpoint 側の `METRICS` と**同じ名前**で出す
（名前がずれると横断表の列が空になるのでテストで検査する）。
反則負け率は「**自分が**反則で負けた」ときだけ数える。

CI は `gh workflow run checkpoint-arena.yml -f arena_run_id=<Arena実行ID> -f seeds=4`、
`gh` が無いときは `.github/ci/checkpoint-arena.request.json` を置いて push
（例は `checkpoint-arena.request.example.json`。削除の push は全ジョブがスキップされる）。

**デッキは arena 記録から作るのを既定にする**。checkpoint の局面分布が実際の対局分布と
一致するのが重要で、`arena_run_id` を渡せば強さの検証で回した棋譜をそのまま再利用できる
（追加の対局コストゼロ。artifact の保持期間内の run に限る）。

### 実行前に止まるもの

- **arm ごとに値が違う `TSUITATE_*`**（原則拒否。監査済みの `CANDIDATE_ONLY_ENV` は現在空。
  恒久対策は issue #21）
- **奇数の `--seeds` / `--seed-base`**（AB/BA が cluster 内で閉じない）
- **schema 1 / 2 の JSONL**（arm 固有の設定が固定相手にも効いていた時期の記録。
  schema 1 は相手の実効 env が未記録、schema 2 はノブをプロセス env で渡していた）
- **手番境界でない checkpoint**、**デッキと食い違う結果**、**arm 内での strategy / env の混在**

## 指標

### 勝敗（主指標）

`delta = candidate_score − control_score`（勝1 / 分0.5 / 負0）を同じ
`(checkpoint, seed)` で取る。CI は **元対局単位の cluster bootstrap**
（seed を独立標本として数えない）。分散成分 σ_b² / σ_w² と ICC、
そこから引いた empirical SE と「元対局数 × seed 数」の MDE 表も出す。

### 安全性の共同指標

過去の arena 悪化で明確な署名を出した高頻度指標を、表示項目ではなく共同指標として扱う。
反則/局・被王手中反則/局・反則負け率・継続手数・手数上限率・終局理由・思考時間（平均/p95/p99）。
統計単位は反則イベントではなく**元対局**。反則は意図的な情報獲得にもなりうるので、
**反則減だけで「強くなった」とは判定しない**（重大な悪化を止める用途に使う）。

### arm 固有ノブは config で渡す（issue #21 で解消）

**当初の制約**: `TSUITATE_*` は `OnceLock` でプロセス全体に効き、相手（凍結版）も
同じプロセスで作られるので、arm ごとに値が違う env があるとその比較は成立しなかった。
既定 matrix で使いたいノブは全部 `estimator_v14` が読む（確認済み）ため、
arm 固有 env は原則拒否にしていた。

**issue #21 以後**: `--control-env` / `--candidate-env` は**プロセス env を触らない**。
値は `crate::config::StrategyConfig` として arm の戦略にだけ渡り、
env を読み続ける凍結相手は arm によらず既定値のまま動く
（`run_child` は親から継承した `TSUITATE_*` も落とす）。設計は
`docs/frozen-hermetic-boundary.md`。

- **候補戦略が凍結版のときはノブを渡せない**（凍結版は config を尊重せず黙って
  無視するので、`assert_arm_knobs_apply` が起動時に止める）
- 両 arm に同じ値を渡す設定（`--budget-ms` = `TSUITATE_THINK_BUDGET_MS`）は
  相手にも等しく効かせたいので、従来どおり子プロセスの env で渡す
- JSONL には `arm_knobs`（arm 固有ノブ）・`arm_config`・`opponent_config`
  （実効設定の sha256 指紋）を残し、`compare` が
  「相手の実効設定が両 arm で一致」を**指紋で**検査する
- プロセス env に arm 固有の値を置く経路が復活したときのために、
  `assert_opponent_blind_to` と凍結版ソースの env 走査（`frozen::SOURCES` 経由）は
  関門として残してある

## デッキ

適格条件（`extract`）:

- 手番境界のみ（v1）
- 初手は除外（観測ゼロで prewarm も無く、実質「初期局面からの arena」なので）
- 元対局がその局面から `--min-remaining`（既定20）手以上続いたこと
  （決着済み局面は両 arm が同じ結果になり情報ゼロ）
- 終局済み・記録不備・時間切れ局を除外
- 原則1棋譜1checkpoint（1対局から多数採ると見かけの N が増える）
- 層化: 先後 / 早中盤（ply 0〜49）・中盤（50〜89）・終盤（90〜） / 通常・被王手 /
  累積反則0・あり / 元対局の結末（優劣の粗い代理）
- 既知の悪手局面を選ぶのではなく、適格局面から固定 seed で決定論的に抽出
- dev / validation は**元対局単位**で割る（validation を目的関数にしない）

優劣タグは P0 では元対局の最終勝敗を粗い代理にする。control 継続勝率で選別する方式へ
切り替えるなら、**選定 seed と計測 seed を分ける**こと。

## 実測

### 前提: 既定挙動の同一性確認（arena run 32650371131）

`selfplay.rs` の裁定ループを切り出したので、CLAUDE.md の慣例（PR #6 のときと同じ）どおり
`match_seed` 固定の arena を1本取った。104局 × 2基準、時計 1000+3、env なし、
commit `6db84e9`、`match_seed=20260815`。

| 基準 | 勝率 | 勝-負-分 | 詰み | 反則負け | 時間切れ | 手数上限 | 平均手数 | 反則/局(A) | think avg/p99max | クロック消費(A) | 残り最小(A) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| estimator_v13 | 61.5% ± 9.4% | 64-40-0 | 27 | 77 | 0 | 0 | 107.2 | 6.25 | 1258ms / 3116ms | 6.5% | 1000s |
| estimator_v14 | 51.0% ± 9.6% | 53-51-0 | 60 | 44 | 0 | 0 | 94.2 | 5.67 | 1195ms / 2743ms | 5.5% | 1000s |

**変質なしと判定した**。

- 勝率は CLAUDE.md の直近 main の記録帯に収まる（vs v13: 54.8〜58.2 の記録に対し 61.5%、
  vs v14: 51.9〜53.8 に対し 51.0%。どちらも単シード104局の CI ±9.4〜9.6 内）
- **時間切れ 0 / 手数上限 0**、平均手数 94〜107（記録帯 98〜105）、
  反則/局 5.67〜6.25（記録帯 5.7〜6.9）、思考平均 1195〜1258ms（記録帯 1.15〜1.25s）
- **クロックが生きていることの直接の証拠**: 消費率 5.5〜6.5% が出ている
  （= `clock_granted_ms` が加算されている = `Option` にした加算の分岐を通っている）。
  残り最小が初期値 1000秒 のままなのも期待どおりで、加算3秒 > 平均消費1.2秒なので
  時計は減らない（CLAUDE.md の 300+3 実測と同じ現象）

**留保**: MCP 経由の `workflow_dispatch` が 403 で main 側を起動できず、
リクエストファイル方式は push したブランチしか走らせられないため、
**同一 `match_seed` のペア対照は取れていない**。言えるのは「記録帯に収まる・変質なし」までで、
数ポイントのドリフトの有無ではない。arena 経路への差分は
`clock_ms` の `Option` 化（通常経路は常に `Some`）・開始状態の引数化（arena は `initial(), 0`）・
`for_each_decision` の委譲（arena は呼ばない）の3点だけで構造上 no-op なので、
ここでは band check で十分と判断した。

この実行の `arena-records-*`（8 artifact、保持期限 2026-11-21）は
**そのまま checkpoint デッキの抽出元にする**（`-f arena_run_id=32650371131`）。


### コスト分解（ローカル 4コア、2000ms、記録 `records/` の59局から16 checkpoint）

計測条件: `--jobs 3`（4コアのうち3スロット）、思考予算 2000ms、相手 `estimator_v14`、
デッキは `records/`（**人間対局の記録**。ここは配管の検証とコスト構造の把握が目的で、
本番の P0 は arena 記録から作ったデッキを CI で回す）。

| 層 | 開始ply（平均） | 継続手数（平均） | prewarm（両側） | continuation | 1 arm 合計 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 早中盤（ply 0〜49） | 26 | 74 | 12秒 | 133秒 | **145秒** |
| 中盤（ply 50〜89） | 55 | 32 | 47秒 | 65秒 | **112秒** |
| 全体 | 41 | 53 | 29秒 | 99秒 | **129秒** |

16 checkpoint × 1 seed × 2 arm = 32 本で **合計 1.14 CPU 時間・壁時計 28.9 分**。

**prewarm は支配項ではなかった**。レビューの見積もり（1 (局面,seed,arm) ≈ 90秒、うち
prewarm 50秒）に対し実測は 129秒・prewarm 29秒（23%）で、**継続対局のほうが重い**。
しかも層で構造が逆転する: 早中盤は prewarm が安く継続が長い、中盤はその逆。

**ここが撤退判断の中心**: 継続対局は「途中から」でも通常の1局とあまり変わらない。
同じ記録から測ると continuation は **1手あたり 1.96 秒**（両者ぶん）で、終局は平均 93手目。
つまりフル対局1局はこの機械で **約183 CPU秒**、checkpoint の 1 arm は **129 CPU秒**で、
**節約は 30%** にとどまる。

理由は構造的で、**コスト削減と情報量が ply 軸で逆を向く**ためである。

- 早い checkpoint（prewarm が安い）は継続が長く、フル対局とほぼ同じコストになる
- 遅い checkpoint（継続が短い）は prewarm が高く、しかも決着済みで情報が出にくい
  （そのために `--min-remaining` で弾く必要がある）

### A/A（同一設定の両 arm、交互実行）

**この節の数字は schema 1 の記録から取ったもの**。A/A は arm 固有 env を使わないので
env 漏れの影響は受けないが、現在の `compare` は schema 1 を集計に使えないよう弾くので、
**同じ JSONL からの再集計はできない**（取り直しが要る）。値は当時の出力として残す。


同一 env・同一戦略・同一 seed で、両 arm を同じスロットで背中合わせに、
checkpoint ごとに AB/BA を均衡させて実行した。

| 項目 | 2000ms（16 × seed1） | 700ms（16 × seed2） |
| --- | --- | --- |
| paired delta | **+18.8pt** [−6.2, +43.8] | **+3.1pt** [−6.2, +12.5] |
| ペア結果 | 改善 4 / 同じ 11 / 悪化 1 | 改善 5 / 同じ 23 / 悪化 4 |
| empirical SE | ±13.6pt | ±5.5pt |
| σ_b² / σ_w² / ICC | 0.296 / – / – | 0.000 / 0.098 / **0.000** |
| MDE（α=0.05 / power=80%） | ±38.1pt | ±15.5pt |
| 最初の選択手の不一致率 | 6.2% | **25.0%** |
| 実行順効果（先 − 後） | −18.8pt | +9.4pt |

**16 checkpoint では、差が無いはずの A/A が +18.8pt を出す**（2000ms・seed1）。
**この件数では捕まえたい −6〜−13pt 帯は原理的に見えない**ことが実測でも確認できた。

実測分散から引いた必要 N（**α=0.05 / power=80%、単位は replicate = 2 seed**）:

| 元対局数 | 16 | 32 | 64 | 128 | 256 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MDE | ±15.5pt | ±11.0pt | ±7.7pt | ±5.5pt | ±3.9pt |
| 参考: CI 半幅 | ±10.8pt | ±7.7pt | ±5.4pt | ±3.8pt | ±2.7pt |

**seed 数の効果はまだ測れていない**。1元対局あたり replicate が1つ（= 2 seed）しか
無いので replicate 間分散 σ_r² が同定できず、seed を増やしたときの外挿は degenerate。
**4 seed 以上（2 replicate 以上）で取り直す必要がある**。

本番と同じ percentile cluster bootstrap CI を当てた power simulation
（600回 × bootstrap 400回）も解析式と整合した: n=16 で −15pt が 70% / −20pt が 94%、
n=64 で −10pt が 93%。

### 統計と入力検査の直し（PR #20 レビュー 2回ぶん）

初版には統計・較正・入力整合性の誤りがあり、指摘を受けて作り直した。**下の数字は修正後**。

**1回目のレビュー**

1. **`1.96·SE` を「MDE」と呼んでいた**。これは 95% CI の半幅であって、
   指定した検出力を持つ最小検出効果ではない。α=0.05 / power=80% なら
   `(1.96+0.84)·SE = 2.80·SE` で、**約1.43倍・必要 N は約2倍**。
   `compare` は CI 半幅と MDE を分けて出し、α と power を明示する（`--alpha` / `--power`）
2. **SE を分散成分から組み立てていた**（`σ_b²/n + σ_w²/(n·s)`）が、
   σ_b² が 0 にクリップされると MSW をそのまま使うことになり **SE を過大評価する**。
   実測で cluster bootstrap の CI 半幅と 2.5倍 食い違っていた。
   SE は **cluster 平均の標本分散から直接**取るように直した
3. **`var·CPU秒` が cluster 構造を無視していた**。cluster 単位へ直した
4. **未較正の値を「既知の arena 差」として使っていた**（下記「既知差の較正」）
5. **`compare` が互換性のない行を黙ってペアにしていた** / **`deck_hash` が KIF の内容を
   含んでいなかった**

**2回目のレビュー**

6. **power simulation が本番の判定規則を模していなかった**。主結果は percentile
   cluster bootstrap CI で判定しているのに、simulation は `mean ± z·SE` の z 検定を
   当てていた。これは解析 MDE と同じ正規近似なので**独立の裏取りにならず**、
   「離散・同点多数・n=16 で bootstrap CI が怪しい」という肝心の点も確かめられない。
   simulation 内でも本番と同じ bootstrap CI を構築するようにし、
   **主 CI の percentile も `--alpha` に連動**させた（従来は 2.5/97.5% 固定）
7. **AB/BA で意図的に反相関させた seed を、独立 seed の `1/s` モデルで外挿していた**。
   scheduler は同一 checkpoint の seed 2k / 2k+1 を必ず逆の arm 順にするので、
   cluster 内残差は iid ではない（まさにそれが下の「5〜8倍」の正体）。
   **連続する2 seed の AB/BA 平均を1 replicate** として畳み、replicate 単位で
   分散成分・外挿・var·CPU秒 を出すようにした。畳めない seed 構成（奇数個）では
   **外挿表も var·CPU秒 も出さない**。`run` は既定 `--seeds 2` で奇数を拒否する
8. **shard が丸ごと欠けても検出できなかった**。`validate_rows` は入力に現れた
   checkpoint 同士しか比べられないので、shard の artifact ごと無いと
   「全部揃っている」ように見える。`compare --deck` でデッキ側の期待 checkpoint
   集合と突き合わせ、workflow 側でも label ごとの JSONL 本数 == shard 数・
   全 label に summary があることを検査して、欠けたら **`INCOMPLETE.txt` を出して
   ジョブを失敗**させる
9. **arm ごとの実効設定の混在を検査していなかった**。JSONL の `env` を読み、
   **同じ arm 内では strategy と正規化 env が一意**であることを検査する
   （親プロセスから継承した `TSUITATE_*` の混入もここで捕まる）
10. **extractor の元対局 ID が file stem だけ**で、複数ディレクトリに同名 JSONL が
    あると「前の対局の ply と後の対局の KIF」を混ぜた entry ができた。
    入力ルートからの相対パス由来の安定 ID にし、衝突は即エラーにした

**3回目のレビュー**

11. **candidate env が固定相手にも適用され、env 実験が全部別の比較になっていた**
    （上記「env 実験は無効」）
12. **replicate が1つしかない既定条件でも seed 数の外挿を出していた**。
    cluster 内自由度 0 で σ_r² が同定できず、未同定成分を 0 に置いた結果を
    「データから得た結論」として出していた。`r < 1.5` では
    **元対局数だけの MDE 表**に切り替え、seed 数の外挿と増減の結論を抑止する
13. **偽陽性率のテストが構造上必ず 0 になっていた**。`delta = 0` では
    「CI が 0 を跨がず、かつ符号が真の効果と同じ」が成立しえないため。
    `delta = 0` は type-I error 用の経路（CI が 0 を外した割合）へ分けた
14. **`--alpha` に CI 計算を連動させた一方、ラベルは「95% CI」固定だった**。
    `(1−alpha)×100` から生成するようにした

**4回目のレビュー**

15. **env 漏れ検査が共有モジュール経由の読取を見落としていた**。v14 の凍結ファイルに
    `TSUITATE_JOSEKI` の文字列は無いが、v14 が作る `crate::opening::OpeningBook` の
    `load()` が読む。共有モジュールも走査対象に加えたうえで、
    **arm 固有 env は原則拒否**（監査済みの `CANDIDATE_ONLY_ENV` は現在空）に変えた。
    走査は「読まないことの証明」にはならない（動的な env 名もありうる）ので、
    許可の根拠には使わない
16. **schema を上げずに `opponent_env` を足していた**。修正前の JSONL も `schema: 1` で、
    欠損時に空 map へフォールバックすると「相手 env は両 arm とも空で一致」と
    解釈されて新しい検査を通ってしまう。**schema 2** へ上げ、`opponent_env` を
    必須にし、schema 1 は集計から明示的に弾く
17. **docs の実行例が現在の実装で動かなかった**（`--seeds 1` は拒否、
    `--candidate-env TSUITATE_ANCHOR_MOVE_W` も拒否、`--known-arena-delta -8.5` は
    撤回済みの値）。実際に動く A/A と凍結版比較の例へ差し替え、
    既定 matrix の本数（7本 → 3本）と P0 の seed 条件（4 以上）も実装に合わせた

**5回目のレビュー**

18. **`compare` は schema 1 の JSONL を拒否するのに、`report` は schema 1 の
    summary を受理していた**。汚染済み JSONL から既に作られた `*.summary.json` を
    再 compare せずに `report` へ渡せるので、撤回済みの delta・既知差が
    横断表・符号一致・順位相関へ戻ってしまう。
    行 schema（`ROW_SCHEMA`）と summary schema（`SUMMARY_SCHEMA`）を分け、
    `report` も検査して schema 1 を拒否するようにした
19. **docs と workflow のコメントが旧方式（走査で止める）のままだった**。
    実装は「走査結果にかかわらず監査済み allowlist 以外は拒否」なので、
    一次資料を現行規約へ統一した。workflow の `candidate_env` / `control_env` は
    **現在すべて失敗する予約フィールド**である旨も明記した
20. **`--alpha` 連動の回帰テストを、env テスト追加時に消してしまっていた**。
    復元して env テスト2本と併存させた

### issue #21（恒久対策）で変わったこと

21. **arm 固有ノブがプロセス env でなく config で渡るようになった**（上記
    「arm 固有ノブは config で渡す」）。行 schema・summary schema は **3**。
    schema 2 の記録は集計から弾く（ノブをプロセス env で渡していた時期のもの）
22. **凍結版ソースの表が二重管理だった**。`frozen::SOURCES` へ一本化し、
    `checkpoint_arena` はそれを引くだけにした（版を足したときの更新漏れが起きない）

### ブロッキングの効き（CPU あたりの情報量）

「同じ検出力を得る CPU コスト」は **replicate（= AB/BA を畳んだ2 seed）単位**で数える
（`var·CPU秒 = (σ_b² + σ_r²/r)·r·(1 replicate の CPU秒)`、n に依らない。小さいほど効率がよい）。

| 実験 | 予算 | seed | 1ペアのコスト | SE | ICC | var·CPU秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aa | 2000ms | 1 | 257秒 | ±13.6pt | 1.00 | **出さない**（畳めない） |
| aa | 700ms | 2 | 90秒 | ±5.5pt | 0.00 | 9 |
| 通常 arena 2本（**粗い参考値**） | 2000ms | – | 366秒 | – | – | **183** |

（env 実験の2本は上記のとおり比較が成立していないので外した。）

**seed 1 の2本は var·CPU秒 を出さない**ようにした。arm 順は
`(checkpoint 番号 + seed) % 2` で決まるので、同じ checkpoint の seed 2k / 2k+1 は
必ず逆の arm 順になる = **実行順成分について意図的に反相関**しており、
`σ_b² + σ_w²/s` の iid 前提が成り立たない。seed 1 では AB/BA が cluster 内で閉じず、
実行順効果がそのまま cluster 平均に残る（初版で「s=1 と s=2 で分散が 5〜8倍違う」と
出たのはこれ）。**seed は 2 の倍数**で取り、replicate 単位で数えること。

通常 arena の参考値 183 は、「候補 vs 基準を1局・対照 vs 基準を1局」で1ペア、
1局 183 CPU秒（この機械での実測推定）、`Var(delta)` を 0.5（ペアリング無し）と
仮定したもの。**この仮定は未検証**で、`match_seed` を揃えた2本の実測 Var と
比べ直すまで「何倍効率がよいか」は確定しない（下記「P0 の残り」）。

分かっているのはコスト側だけである。

- 継続対局は 1 arm 129 CPU秒（2000ms）で、フル対局の推定 183 CPU秒に対し **節約は 30%**
- 局面ブロッキングで Var は下がるが、その量は arena 側の実測 Var と並べないと意味を持たない

なお `--shared-prewarm` はこの実測では使っていない（両 arm の env が違うケースが
本命なので共有できない）。prewarm は全体の 23% なので、共有が効く場面でも
上限は 12% 程度の削減にとどまる。

### env 実験は無効（PR #20 追加レビュー指摘1）

`TSUITATE_DROP_PROBE_REPEAT_GATE=1` を 700ms / 2000ms で回した結果を載せていたが、
**この比較は成立していなかった**ので撤回する。

`TSUITATE_*` はプロセス全体に効き、**相手（凍結版）も同じプロセスで作られる**。
つまり candidate arm は

```
候補設定の現行 estimator  vs  候補設定の estimator_v14
```

control arm は

```
既定の現行 estimator  vs  既定の estimator_v14
```

で、**固定相手が arm ごとに変わっている**。JSONL 上はどちらも `opponent=estimator_v14`
なので、名前の一致を見る検査では捕まらない。

しかも既定 matrix で使いたかったノブは**全部 `src/frozen/estimator_v14.rs` が読む**
（`DROP_PROBE_REPEAT_GATE` / `ANCHOR_MOVE_W` / `EXPOSED_PAWN_HEAD_W` / `HAND_ASSET_W` /
`KING_KNOWN_APPROACH_W` / `LINK_ENDGAME_DAMPEN` / `MATERIAL_DEGEN_Q0` /
`PROMOTE_FAR_W` / `GEN_NONPROMOTE`）。これは CLAUDE.md が arena.yml について
警告している「env はプロセス全体に効くので『候補側だけ』になるのは、その凍結版が
名前を知らないノブに限る」と同じ罠を、checkpoint arena 側で踏んだもの。

対策として入れたもの:

- **arm 固有 env を原則拒否**にした。通るのは監査済みの `CANDIDATE_ONLY_ENV`
  （**現在は空**）だけで、`--allow-opponent-env` を明示したときだけ続行できる。
  凍結版＋**共有モジュール**のソース走査は二次的な検査（リストの陳腐化対策）で、
  「読まないことの証明」には使わない —— v14 は `TSUITATE_JOSEKI` を
  凍結ファイルではなく共有 `opening.rs` 経由で読んでおり、
  1ファイル走査では見落としていた
- JSONL へ**相手の実効 env** を残し、`compare` が両 arm で一致するかを検査する
- 子プロセスの `TSUITATE_*` を**一度すべて落としてから**意図した env だけ設定する
  （親から継承した値は全 shard で同じなので、arm 内一意性検査では捕まらなかった）
- **workflow の既定 matrix から env 実験を外した**。残したのは `aa` と
  凍結版どうしの2本（env を使わないので影響を受けない）

候補側だけにノブを効かせる経路（現行 strategy だけが読む alias、または評価
パラメータのインスタンス注入）ができるまで、env 実験は P0 の較正に使えない。

なお `--budget-ms`（`TSUITATE_THINK_BUDGET_MS`）は**両 arm に同じ値**を渡すので、
相手にも等しく効き、この問題には当たらない（意図どおり両側同じ予算になる）。

### そのうえで残る観察

A/A（env なし）は成立しているので、次だけは言える。

| 実験 | 予算 | seed | paired delta | 95% CI | SE |
| --- | ---: | ---: | ---: | --- | ---: |
| aa | 2000ms | 1 | +18.8pt | [−6.2, +43.8] | ±13.6pt |
| aa | 700ms | 2 | +3.1pt | [−6.2, +12.5] | ±5.5pt |

**2000ms・seed1 は差が無いはずの A/A で +18.8pt を出す**。この構成はゲートに使えない
（元対局16・非ゼロのペアが数組では percentile cluster bootstrap 自体が信用できない）。
700ms・seed2 はほぼ 0 中心。

**較正（既知の arena 差との突き合わせ）は1件もできていない**。
`report` は既知値が2件未満なら符号一致・順位相関を出さず「未較正」と明示する。

### A/A の実行順効果について

実行順効果（先に走った arm − 後）は −18.8 / +9.4 / +6.2 / +0.0pt とばらつく。
ただし **s=2 では cluster の内側で AB/BA が閉じるので、cluster 平均からは
実行順効果が落ちる**。つまりこの指標は「s=1 のときに効く警報」であって、
s が偶数なら小さく出て当然。CI 規模では s を偶数に取ったうえで、
それでも 0 中心かを確認する。

なおこの反相関は**設計として意図したもの**なので、統計側も
replicate（2 seed の AB/BA 平均）を単位にして扱う。生の seed を独立標本として
`1/s` で外挿してはいけない。

### P0 の残り（CI で回す）

ローカル 4コアでは件数も実験数も足りない。**統計的に意味のある P0 は CI で回す**。

```bash
gh workflow run checkpoint-arena.yml \
  -f arena_run_id=32650371131 \
  -f checkpoints=64 -f seeds=4 -f split=all -f shards=8 \
  -f budgets="700 2000"
```

**`seeds` は 4 以上**（= 2 replicate 以上）。2 seed は AB/BA が cluster 内で閉じる
最小構成だが replicate 間分散が同定できないので、P0 の完了条件は満たせない
（workflow の既定も 4 にしてある）。

既定の実験セット（`plan` ジョブの `default_experiments`）は **3本**:
`aa` と、凍結版どうしの `v13_v12_vs_opp` / `v14_v10_vs_opp`。
**env 実験は既定から外してある**（相手も同じノブを読むので比較が成立しない。
恒久対策は issue #21）。

残っている項目:

- [ ] **通常 arena のペア差の実測 Var**。`match_seed` を揃えた2本
      （候補 / 対照）を取り、局ごとのペア差の分散を直接測る。これが無いと
      「通常 arena の何倍効率がよいか」は決まらない（現在の参考値 183 は
      `Var(delta)=0.5` の仮定に乗っている）。**同じ実行が較正の既知値にもなる**
      → 計測経路は実装済み（`ARENA_GAMES_JSON` / `arena-var` / `pair_with`）。
      対照 [run 32721832218](https://github.com/tempakyousuke/tsuitate-bot/actions/runs/32721832218)
      （104局 vs v14・`match_seed=20260824`・2000ms）を起動済み、候補側は
      `cand_env=TSUITATE_DROP_PROBE_REPEAT_GATE=1` で続けて回す
- [ ] 64 checkpoint 以上・**seed 4 以上**での empirical SE / ICC / MDE
      → 段階1（700ms）を
      [run 32723624182](https://github.com/tempakyousuke/tsuitate-bot/actions/runs/32723624182)
      で起動済み（64 checkpoint × 4 seed × 4実験 × 8シャード、
      デッキは arena run 32697854659 の記録から）
- [ ] 既知差の較正。**同じ candidate / control / opponent / 予算で arena を
      取り直した差**だけを `--known-arena-delta` に渡す。
      **issue #21 の完了（PR #22）で env ノブの較正が可能になった**
      （ノブはプロセス env でなく config で渡るので、固定相手は arm によらず既定値）。
      段階1では `drop_probe_gate`（`TSUITATE_DROP_PROBE_REPEAT_GATE=1`）を回している
- [ ] 700ms と 2000ms の符号・順位・副指標の方向の一致
      → **段階的に回す**（2026-08-24 ユーザー判断）: まず 700ms を回し、
      A/A が 0 中心か・符号と順位が筋の通るものかを見てから 2000ms を回す。
      無駄撃ちが少ない代わりに、700ms 単独では「2000ms の強さ」を主張できない
- [ ] **arena 記録から作ったデッキ**での再測定 → 段階1が
      [run 32697854659](https://github.com/tempakyousuke/tsuitate-bot/actions/runs/32697854659)
      の `arena-records-*`（保持期限 2026-11-22）から抽出したデッキで回っている

## 撤退判断

issue が置いた基準と、ローカル部分実測での現時点の読み。
**確定判断は CI 規模の計測（上記「P0 の残り」）を待つ**。

| 用途 | 基準 | 現時点の読み |
| --- | --- | --- |
| quick 破滅検出器 | −15〜−20pt 級を通常 arena の概ね 1/3 以下の CPU で検出 | **未判定**。CPU 効率の比が arena 側の実測 Var 待ち |
| validation ゲート | −8〜−13pt 級を概ね 1/2 以下の CPU で検出 | **未判定**（同上）。検出の条件も未証明 |
| 安全性ゲートへの縮小 | 既知の重大悪化を反則/局等が安定して検出でき、費用が小さい | **未判定**。根拠にしていた観察は env 実験に乗っており、比較が成立していなかった |
| 撤退 | v12→v14 級の大差の符号すら安定しない、または通常 arena に明確なコスト優位が無い | **今は該当しない**。ただし「コスト優位が実在する」と言い切れる根拠も無い |

### 現時点の結論（暫定）

1. **env 実験の経路を直すまで、env ノブの較正はできない**。相手が同じノブを
   読むので「同じ固定相手」で比べられない。候補側だけに効く経路
   （現行 strategy だけが読む alias、または評価パラメータのインスタンス注入）が要る。
   凍結版どうしの比較（`v13_v12_vs_opp` 等）は env を使わないので影響を受けない
2. **P1 以降へ進む前に CI 規模の P0 を回す**。ローカル16局面では判定できない
3. **その前に、通常 arena のペア差の実測 Var を取る**。効率比も較正の既知値も
   ここが起点で、1回の arena ペアで両方そろう
4. **較正が通るまで、この計測を採否に使わない**（CI は informational のまま）。
   特に **2000ms・seed1 の構成は使わない**
5. **seed は 2 の倍数、seed 数の効果まで測るなら 4 以上**。
   AB/BA が cluster 内で閉じるかどうかで cluster 平均の分散が変わり、
   replicate が1つでは replicate 間分散が同定できない
6. **勝敗より先に反則/局が動く**という観察は env 実験に乗っていたので、
   これも**保留**（比較が成立していなかったため）

### 変えていないこと

- 通常 arena の運用と採否基準は変更していない
- scenario suite・arena・凍結版ガントレットは置き換えていない
- checkpoint arena の CI は**通常のコード push では走らない**（手動起動のみ）
