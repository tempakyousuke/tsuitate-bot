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

```bash
# デッキ抽出（arena 記録から。原則1棋譜1checkpoint・層化・手番境界のみ）
cargo run --release --bin checkpoint_arena -- extract \
  --records records --out checkpoint-arena \
  --opponent estimator_v14 --min-remaining 20 --limit 64

# 2つの arm を同一 runner で交互実行
cargo run --release --bin checkpoint_arena -- run checkpoint-arena/deck.json \
  --split dev --seeds 1 --jobs 3 --experiment anchor_move \
  --candidate-env TSUITATE_ANCHOR_MOVE_W=0.6 \
  --jsonl out/anchor_move.jsonl

# ペア集計（cluster bootstrap / ICC / MDE / 安全性の共同指標）
cargo run --release --bin checkpoint_arena -- compare out/anchor_move.jsonl \
  --known-arena-delta -8.5 --markdown out/anchor_move.md --json out/anchor_move.summary.json

# 実験横断（符号一致・順位相関・重大悪化の見逃し）
cargo run --release --bin checkpoint_arena -- report out/*.summary.json
```

CI は `gh workflow run checkpoint-arena.yml -f arena_run_id=<Arena実行ID> -f budgets="700 2000"`、
`gh` が無いときは `.github/ci/checkpoint-arena.request.json` を置いて push
（例は `checkpoint-arena.request.example.json`。削除の push は全ジョブがスキップされる）。

**デッキは arena 記録から作るのを既定にする**。checkpoint の局面分布が実際の対局分布と
一致するのが重要で、`arena_run_id` を渡せば強さの検証で回した棋譜をそのまま再利用できる
（追加の対局コストゼロ。artifact の保持期間内の run に限る）。

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

実測分散から引いた必要 N（**α=0.05 / power=80%、`compare` の既定**）:

| 元対局数 | 16 | 32 | 64 | 128 | 256 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MDE（seed 2） | ±15.5pt | ±11.0pt | ±7.7pt | ±5.5pt | ±3.9pt |
| 参考: CI 半幅 | ±10.8pt | ±7.7pt | ±5.4pt | ±3.8pt | ±2.7pt |

非パラメトリックな power simulation（元対局 delta の経験分布から再標本化、4000回）も
解析式とほぼ一致した（n=16 で −15pt が 72% / −20pt が 93%、n=64 で −10pt が 94%）。

### 統計の直し（PR #20 レビュー指摘 1 / 3）

初版には統計上の誤りが3つあり、数字を作り直した。**下の数字は修正後**。

1. **`1.96·SE` を「MDE」と呼んでいた**。これは 95% CI の半幅であって、
   指定した検出力を持つ最小検出効果ではない。α=0.05 / power=80% なら
   `(1.96+0.84)·SE = 2.80·SE` で、**約1.43倍・必要 N は約2倍**。
   `compare` は CI 半幅と MDE を分けて出し、α と power を明示するようにした
   （`--alpha` / `--power`）。あわせて**実際の power simulation** も実装した
   （経験分布から再標本化して判定規則を当てる。解析式とは独立の裏取り）
2. **SE を分散成分から組み立てていた**（`σ_b²/n + σ_w²/(n·s)`）が、
   σ_b² が 0 にクリップされると MSW をそのまま使うことになり **SE を過大評価する**。
   実測で cluster bootstrap の CI 半幅と 2.5倍 食い違っていた（SE 12.1pt vs 実質 4.8pt）。
   SE は **cluster 平均の標本分散から直接**取るように直した（bootstrap と同じ対象）
3. **`var·CPU秒` が cluster 構造を無視していた**。CI と SE は元対局ごとの
   seed 平均を標本にしているのに、効率だけ全 pair の生 delta 分散を使っていた。
   `SE²(n) = (σ_b² + σ_w²/s)/n`・総コスト `n·s·(1ペアCPU秒)` から
   **`var·CPU秒 = (σ_b² + σ_w²/s)·s·(1ペアCPU秒)`**（n に依らない）へ直した

### ブロッキングの効き（CPU あたりの情報量）

「同じ検出力を得る CPU コスト」は **cluster 単位**で数える
（`var·CPU秒 = (σ_b² + σ_w²/s)·s·1ペアCPU秒`、n に依らない。小さいほど効率がよい）。

| 実験 | 予算 | seed | 1ペアのコスト | SE | ICC | var·CPU秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aa | 2000ms | 1 | 257秒 | ±13.6pt | 1.00 | **76** |
| repeat_gate | 2000ms | 1 | 267秒 | ±11.2pt | 1.00 | **53** |
| aa | 700ms | 2 | 90秒 | ±5.5pt | 0.00 | **9** |
| repeat_gate | 700ms | 2 | 89秒 | ±7.7pt | 0.00 | **17** |
| 通常 arena 2本（**粗い参考値**） | 2000ms | – | 366秒 | – | – | **183** |

**seed 1 と seed 2 で 5〜8倍も違うのは、標本数の差ではなく実行順の扱いの差**だった。
arm 順は `(checkpoint 番号 + seed) % 2` で決まるので、**同じ checkpoint の seed 0 と seed 1 は
必ず逆の arm 順になる**。つまり s=2 では AB/BA の均衡が cluster の内側で閉じ、
実行順効果が cluster 平均から落ちる。s=1 では落ちない。
→ **seed は 2 の倍数で取るべき**（ICC の議論だけから「seed 1 が最適」と読むのは誤り）。
この実測では予算と seed 数が交絡しているので、両者の寄与は分離できていない。

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

### 既知差の較正（**未較正**。PR #20 レビュー指摘2）

`TSUITATE_DROP_PROBE_REPEAT_GATE=1` を 700ms / 2000ms で回した。

| 実験 | 予算 | seed | checkpoint delta | 95% CI | SE | 反則/局 delta |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| aa | 2000ms | 1 | +18.8pt | [−6.2, +43.8] | ±13.6pt | −0.31 |
| aa | 700ms | 2 | +3.1pt | [−6.2, +12.5] | ±5.5pt | +0.66 |
| repeat_gate | 700ms | 2 | −6.2pt | [−21.9, +9.4] | ±7.7pt | +0.81 |
| repeat_gate | 2000ms | 1 | +25.0pt | [+6.2, +50.0] | ±11.2pt | +0.38 |

**この表を「較正」とは呼べない**。CLAUDE.md に残っている −12.8pt は

- 当時の main（その後たくさんの採用が入っている）を対照に、
- 別の対戦条件（vs v13 の3シード・初期局面からのフル対局）で

測った値で、ここで測っている量（固定相手 `estimator_v14`・途中局面からの継続・
この予算）とは**別物**。同様に、初版の workflow が付けていた

- `pr1combo = −7.4pt` … 実際の −7.4pt は5つの新ノブ＋`material_degen_q0`＋
  GEN/PREROLE/CAPTURE_RETREAT を全部入れた構成の値で、per-knob ablation は未実施
  （CLAUDE.md に明記）。2項だけに付ける根拠は無い
- `v13_vs_v12 = +9pt` / `v14_vs_v10 = +16pt` … このハーネスが測るのは
  `(v13 vs 相手) − (v12 vs 相手)` であって `v13 vs v12` ではない。
  非推移性のあるゲームで別 matchup の差を既知値にはできない

も根拠が無かったので、**workflow の既定から `known` を全部外した**。
`report` は既知値が2件未満なら符号一致・順位相関を出さず「未較正」と明示する。

そのうえで観察として残せるのは次だけ。

- 2000ms・seed1 は A/A で +18.8pt、repeat_gate で **+25.0pt（CI が 0 を跨がない）**。
  この構成は**自信を持って何かを言ってしまう**ので、ゲートには使えない
  （元対局16・非ゼロのペアが数組では percentile cluster bootstrap 自体が信用できない）
- 700ms・seed2 では A/A が +3.1pt（ほぼ 0 中心）、repeat_gate が −6.2pt。
  **符号は期待どおりだが CI は 0 を跨ぐ**
- **反則/局は両予算とも候補側で増える向き**（+0.81 / +0.38）。
  通常 arena で観測された署名（6.4〜6.9 → 7.8〜8.2）と同じ向きで、勝敗より先に動いた。
  ただし CI は両方とも 0 を跨ぐので「安定して検出できる」とはまだ言えない

### A/A の実行順効果について

実行順効果（先に走った arm − 後）は −18.8 / +9.4 / +6.2 / +0.0pt とばらつき、
系統的な順序バイアスの証拠にはならない。ただし**上で分かったとおり、s=2 では
cluster の内側で AB/BA が閉じるので、cluster 平均からは実行順効果が落ちる**。
つまりこの指標は「s=1 のときに効く警報」であって、s が偶数なら小さく出て当然。
CI 規模では s を偶数に取ったうえで、それでも 0 中心かを確認する。

### P0 の残り（CI で回す）

ローカル 4コアでは件数も実験数も足りない。**統計的に意味のある P0 は CI で回す**。

```bash
gh workflow run checkpoint-arena.yml \
  -f arena_run_id=<直近 Arena 実行のID> \
  -f checkpoints=64 -f seeds=2 -f split=all -f shards=8 \
  -f budgets="700 2000"
```

既定の実験セット（`plan` ジョブの `default_experiments`）は issue の表そのままで、
HEAD の env ノブだけで再現できる既知の負例4本と、同一バイナリ内の凍結版どうしの正例2本、
それに A/A を加えた7本。`budgets="700 2000"` で符号・順位が保たれるかの較正も同時に取れる。

残っている項目:

- [ ] **通常 arena のペア差の実測 Var**。`match_seed` を揃えた2本
      （候補 / 対照）を取り、局ごとのペア差の分散を直接測る。これが無いと
      「通常 arena の何倍効率がよいか」は決まらない（現在の参考値 183 は
      `Var(delta)=0.5` の仮定に乗っている）。**同じ実行が較正の既知値にもなる**
      ので、ここは1回で2つ片づく
- [ ] 64 checkpoint 以上・**seed は 2 の倍数**での empirical SE / ICC / MDE
      （seed 1 では σ_w² が同定できず、AB/BA も cluster 内で閉じない）
- [ ] 既知差の較正。**同じ candidate / control / opponent / 予算で arena を
      取り直した差**だけを `--known-arena-delta` に渡す
- [ ] 700ms と 2000ms の符号・順位・副指標の方向の一致
- [ ] **arena 記録から作ったデッキ**での再測定
      （[run 32650371131](https://github.com/tempakyousuke/tsuitate-bot/actions/runs/32650371131)
      の `arena-records-*` が使える）

## 撤退判断

issue が置いた基準と、ローカル部分実測での現時点の読み。
**確定判断は CI 規模の計測（上記「P0 の残り」）を待つ**。

| 用途 | 基準 | 現時点の読み |
| --- | --- | --- |
| quick 破滅検出器 | −15〜−20pt 級を通常 arena の概ね 1/3 以下の CPU で検出 | **未判定**。CPU 効率の比が arena 側の実測 Var 待ちで決まらない |
| validation ゲート | −8〜−13pt 級を概ね 1/2 以下の CPU で検出 | **未判定**（同上）。検出の条件も未証明で、2000ms・seed1 は既知の悪化に +25.0pt を返した |
| 安全性ゲートへの縮小 | 既知の重大悪化を反則/局等が安定して検出でき、費用が小さい | **最有望**。repeat_gate は両予算とも反則/局が正しい向きに動いた唯一の指標。ただし CI は 0 を跨ぐ |
| 撤退 | v12→v14 級の大差の符号すら安定しない、または通常 arena に明確なコスト優位が無い | **今は該当しない**。ただし「コスト優位が実在する」と言い切れる根拠もまだ無い（初版はここを言い過ぎていた） |

### 現時点の結論（暫定）

1. **P1 以降へ進む前に CI 規模の P0 を回す**。ローカル16局面では
   「使えるか」を判定できないことがはっきりした
2. **その前に、通常 arena のペア差の実測 Var を取る**。効率比も較正の既知値も
   ここが起点で、1回の arena ペアで両方そろう
3. **較正が通るまで、この計測を採否に使わない**（CI は informational のまま）。
   特に **2000ms・seed1 の構成は使わない**: 既知の重大悪化に対して
   CI が 0 を跨がずに逆符号を出した実例がある
4. **seed は 2 の倍数で取る**。AB/BA の均衡が cluster の内側で閉じるかどうかで
   cluster 平均の分散が 5〜8倍変わる
5. **700ms を quick preset にするのはまだ早い**。最初の選択手の不一致率が
   6% → 25〜31% に上がる（別の方策を測っている）
6. **勝敗より先に反則/局が動く**という観察は、issue が挙げた
   「安全性ゲートへの縮小」に具体的な根拠を与えた

### 変えていないこと

- 通常 arena の運用と採否基準は変更していない
- scenario suite・arena・凍結版ガントレットは置き換えていない
- checkpoint arena の CI は**通常のコード push では走らない**（手動起動のみ）
