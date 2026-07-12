# SPSAチューニングログ（bin/tune の出力）の見方

`cargo run --release --bin tune` が書き出す JSONL（1行=1イベント）。
ファイルは `TUNE_LOG` 環境変数で指定（既定 `tune-log.jsonl`、リポジトリ直下・gitignore済み）。
実験として残す価値があるものは、このディレクトリへコピーしてコミットする。

## 行の種類

**start** — ラン開始時に1行:

```json
{"type":"start","iterations":40,"games_per_eval":40,"baselines":["estimator_v5"],"initial":{...}}
```

- `initial`: 開始時点のパラメータ（= その時点の `EvalParams::default`）

**iter** — 各反復ごとに1行:

```json
{"type":"iter","k":12,"f_plus":0.55,"f_minus":0.475,"theta":{...}}
```

- `k`: 反復番号（再開してもラン内で通し）
- `f_plus` / `f_minus`: パラメータを正方向/負方向に揺らした2点のスコア率。
  **勝ち=1・引き分け=0.5・負け=0 の平均**なので、0.5 = 基準戦略と互角。
  引き分け除外の勝率ではないことに注意（アリーナ表示の「勝率」より低めに出る）
- `theta`: この反復の**更新後の中心点**（f± を測った摂動点そのものではない）。
  再開時はこの最後の `theta` から続行される

**done** — 完走時に1行:

```json
{"type":"done","final":{...},"best":{...},"best_score":0.7}
```

- `final`: 収束点（最後の反復の中心点）。**採用候補は基本こちら**
- `best` / `best_score`: ラン中に観測した最高スコアの摂動点。1回の評価の
  ノイズ（40局で±0.13程度）を含む**参考値**で、これ単体を採用しないこと

## 読むときの定石

- **スコアの推移**: `f_plus`/`f_minus` の移動平均が 0.5 を安定して超え始めたら
  「基準より強い領域」に入った証拠。1点の高値はノイズ
- **パラメータの動き**: `theta` の各項が反復をまたいで**一方向に動き続けているか**
  を見る（往復しているだけの項はまだ信号が取れていない）
- 抽出例:

```bash
# スコア推移
jq -r 'select(.type=="iter") | "\(.k)\t\(.f_plus)\t\(.f_minus)"' tuning/tune-rush.jsonl
# 特定パラメータの軌跡
jq -r 'select(.type=="iter") | "\(.k)\t\(.theta.camp_scale)"' tuning/tune-rush.jsonl
# 最終パラメータ
jq 'select(.type=="done") | .final' tuning/tune-rush.jsonl
```

- **採用の手順**: `done.final` を `EvalParams::default`（strategy.rs）へ書き写し、
  CIの200局ガントレットで確定させる（100局判定は偽陽性事例あり。
  メモリ/CLAUDE.md の検証ポリシー参照）

## このディレクトリのファイル

- `tune-antirush.jsonl` — 実験A: 対・居飛車速攻の受け特化（2026-07-12、40反復、
  基準 estimator_rush、best 0.70）
- `tune-rush.jsonl` — 実験B: 速攻側特化（同、候補の定跡を居飛車速攻に固定、
  基準 estimator_v5、best 0.80）
- 第1ラウンド（汎用、60反復 vs v5）のログはVM再構築時に喪失。
  収束点は commit 89a3612 の `EvalParams::default` に反映済み
