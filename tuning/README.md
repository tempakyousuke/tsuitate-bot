# SPSAチューニングログ（bin/tune の出力）の見方

`cargo run --release --bin tune` が書き出す JSONL（1行=1イベント）。
ファイルは `TUNE_LOG` 環境変数で指定（既定 `tune-log.jsonl`、リポジトリ直下・gitignore済み）。
実験として残す価値があるものは、このディレクトリへコピーしてコミットする。

## 行の種類（2026-07-13 の共通乱数法対応で拡張。旧形式ログも読める）

**start** — ラン開始時に1行:

```json
{"type":"start","iterations":40,"run_seed":123,"config":{...},"initial":{...}}
```

- `initial`: 開始時点のパラメータ（= その時点の `EvalParams::default`）
- `run_seed`: 対局条件（定跡・推定器シード等）を決める乱数の根。再開時に引き継がれる
- `config`: 基準・局数・定跡固定・思考予算・パラメータ空間の指紋。
  **再開時にこれと現在の設定が一致しないと停止する**（`TUNE_FORCE_RESUME=1` で強行）

**eval** — 各評価（f+ と f− それぞれ）ごとに1行:

```json
{"type":"eval","k":12,"which":"plus","u":[...],"score":0.55,"stats":[{...}]}
```

- `u`: 摂動点の正規化座標（完全精度）
- `stats`: 基準ごとの対局内訳（勝敗・終局理由・`max_plies` 数・平均手数・
  反則数・思考時間）。**引き分け化や時間浪費でスコアが上がる変質はここで検出する**

**iter** — 各反復ごとに1行:

```json
{"type":"iter","k":12,"f_plus":0.55,"f_minus":0.475,"plus_first":true,"u":[...],"theta":{...}}
```

- `k`: 反復番号（再開してもラン内で通し）
- `f_plus` / `f_minus`: 摂動2点のスコア率（勝ち=1・引き分け=0.5・負け=0 の平均。
  0.5 = 基準戦略と互角）。**同じ対局シード列で評価されている**ので、
  差分は主にパラメータの効果（旧形式は独立評価でノイズが大きい）
- `u` / `theta`: この反復の**更新後の中心点**（完全精度の正規化座標と表示用の実値）。
  再開は `u` を優先して使う

**done** — 完走時に1行:

```json
{"type":"done","final":{...},"final_score":0.55,"final_stats":[...],"best":{...},"best_score":0.7}
```

- `final` / `final_score`: 収束点とその追加評価スコア。**採用候補は基本こちら**
- `best` / `best_score`: ラン中に観測した最高スコアの摂動点。1回の評価の
  ノイズを含む**参考値**で、これ単体を採用しないこと

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
- `tune-round2.jsonl` — 第2ラウンド（2026-07-14、共通乱数法・全35次元・
  60反復×2×40局 vs v5）。**成功**: final_score 0.675、収束点は estimator_v6 の
  Default に反映。最大の発見は check_bonus↓/check_foul_scale↑
  （王手価値は相手の反則蓄積に比例）
- `tune-round3.jsonl` — 第3ラウンド（2026-07-15、機構系18次元マスク・span0.5・
  vs v6）。**不発**: final_score 0.438。平均スコア 0.513 で終始フラット
- `tune-round4.jsonl` — 第4ラウンド（2026-07-17、反則経済系9次元
  （foul_cost_base/pow・check_bonus/check_foul_scale・foul_diff_pow・
  check_limit_accel 等）・span1.0・vs v6）。**不発**: final_score 0.325、
  平均 0.512。オラクル上限（下記）に対しスカラー係数の調整では届かない、
  が確定した実験。
  **オラクル測定（2026-07-16、ARENA_ORACLE_A）**: 全反則回避で vs v6 86.2%±4.8 /
  王手中のみ回避で 59.5%±7.1 — 反則経済に36ptの伸びしろは実在する。
  次の本命は構造改修（C-7: 連続重み・観測尤度・ESSの粒子フィルタ）
- `tune-v10.jsonl` — 第5ラウンド（2026-07-23〜24、value_nn_w+攻め・圧力系
  9次元マスク（attack_w/pressure_w/advance_w/threat_w/info_bonus/
  king_probe_bonus/coverage_w/tokin_probe_w/value_nn_w）・span0.6・
  60反復×2×60局 vs v10、GCE Spot）。**不発**: final_score 0.442、
  評価スコアは前半0.474→終盤0.487でトレンドなし。value_nn_w は
  6.0→5.77 とほぼ不動（凍結時の w スイープが既に最適近傍だった証拠）、
  他の8次元も±25%以内の微動。「SPSAは未調整の新パラメータがあるときだけ
  効く」の追認で、EvalParams::default への反映なし。
  途中2回のSpot停止あり（TUNE_LOGから再開。1回は設定一致チェックが
  浮動小数点1ULP差で誤検知→TUNE_FORCE_RESUME=1のdrop-inで解消、
  メモリ tune-resume-float-ulp-trap 参照）
