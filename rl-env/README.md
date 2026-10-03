# rl-env（`tsuitate_rl`）

DeepNash（R-NaD）路線の RL 環境を Python から呼ぶための PyO3 の薄いラッパー。
設計は `docs/rl-deepnash-design.md`。

- 1局の進行（裁定・観測エンコード・行動マスク・相手戦略の自動着手）は本体
  （`src/referee.rs` / `src/rl/`）にあり、ここは **n 局の並列化（rayon、GIL を外す）と
  numpy 変換だけ**を持つ
- 本体の依存（pyo3 / numpy / rayon）を増やさないため別 crate。本体の `cargo build` / CI には
  含まれない

## ビルド

学習側の venv（`~/Develop/tsuitate-nn/.venv`、Python 3.12）へ入れる:

```sh
~/Develop/tsuitate-nn/.venv/bin/python -m pip install maturin   # 初回だけ
VIRTUAL_ENV=~/Develop/tsuitate-nn/.venv ~/Develop/tsuitate-nn/.venv/bin/maturin develop --release -m rl-env/Cargo.toml
~/Develop/tsuitate-nn/.venv/bin/python rl-env/smoke.py          # 通しと速度計測
```

本体（`src/rl/` 等）を変えたら `maturin develop` をやり直す。

## API

```python
import tsuitate_rl
env = tsuitate_rl.VecEnv(256, seed=0)                       # 自己対局
env = tsuitate_rl.VecEnv(64, opponent="heuristic", seed=0)  # 評価モード（strategy::make の名前）
obs, mask, player = env.observe()  # [n,86,9,9] f32 / [n,11259] bool / [n] i8（0=先手 1=後手）
rewards, done = env.step(actions)  # [n] i64 → [n,2] f32（先手,後手）/ [n] bool
env.learner_colors()               # 評価モードの学習側の色（自己対局は -1）
env.pop_finished()                 # 終局した局: game_no / winner / reason / plies / fouls / learner / opponent
                                   #   ＋玉の周りの集計: guard_r1 / guard_r2 / near_drops / near_drop_fouls / near_captures
env = tsuitate_rl.VecEnv(8, opponents=["rl_v15"] * 4 + ["rl_v16"] * 4, auto_reset=False)
                                   # リーグ学習: 局番号 g の局は opponents[g % len] と指す（下記）
env = tsuitate_rl.VecEnv(64, opponent="rl_policy:<重み>", guard_bonus=0.5, guard_radius=1)
                                   # 玉の周りの固めのボーナス（下記）
```

- 終局した局は `step` の中で新しい局へ差し替わる（`done` の局の次の `observe` は新しい局）。
  学習側では **`done` の局で価値のブートストラップと履歴の引き継ぎを切る**こと
- 報酬は終局時だけ ±1（引き分け 0）。反則などの途中報酬はない。自己対局では、終局の手を
  指さなかった側にも同じ `step` で報酬が返る（[先手, 後手] の両方が入る）ので、両者の
  直前の遷移へ割り当てる
- **自己対局では反則で同じプレイヤーが続けて指す**（反則は手番を変えない）。`player` は
  毎回 `observe` の値を使い、交互だと仮定しない
- 行動は**バッチ全体を先に検査**し、1つでも範囲外・マスク外があれば、どの局も進めずに
  `ValueError`
- 評価モードは偶数局で学習側が先手。相手の手番は内部で指し進め、`observe` は常に学習側の手番。
  相手の手で終局した場合も、その直前の学習側の `step` が報酬を返す
- **評価モードで報酬の列を選ぶ色は `step` の前に取る**（`step` の後の `learner_colors()` は
  差し替わった新しい局の色で、終局した局とは先後が逆のことがある）:

  ```python
  learner = env.learner_colors()
  rewards, done = env.step(actions)
  learner_rewards = rewards[np.arange(env.num_envs), learner]
  ```
- 相手の乱数は `seed` と局番号から決まる。heuristic は完全に再現する（同じ seed・同じ行動列 →
  同じ観測列）。estimator 系は壁時計で思考を打ち切るので、seed が同じでも完全には再現しない
- 自駒視点の候補が1つも残らない手番は `no_moves` で負け
- **`guard_bonus=λ`**（既定 0）: 終局の報酬に λ × 対局を通した玉の周りの占有率の平均を両者それぞれ
  足す（防御特化モデル用。`docs/rl-deepnash-design.md` の「防御特化（玉の周りの固め）」）。
  占有率 = 自玉から距離 `guard_radius`（1 = 8近傍 / 2）以内の盤上のマスのうち自駒がいる割合で、
  開始局面と受理手の直後ごとに標本を取る。**λ > 0 では報酬が零和でなくなる**ので、評価モードで
  学習側の列だけを使う形を想定している。`pop_finished()` の `guard_r1` / `guard_r2` は λ に
  関係なく常に出る（監視用）
- **`opponents=[名前, ...]`**（`opponent` と排他）: 1つの env に複数の相手を混ぜる。局番号 g の局は
  `opponents[g % len]` と指す。リーグ学習（tsuitate-nn の `--league`）で、苦手な相手ほど多く当てる配分を
  1つの env で回すためのもの。相手ごとに env を分けると、相手の推論（凍結版の方策ネットで1手約10ms）が
  env の数だけ直列になる（`VecEnv` は unsendable なので Python のスレッドでは並べられない）。
  学習側の色は局番号の偶奇で決まるので、相手ごとに**偶数個ずつ**並べると先後がそろう。
  作った直後に終局した局（相手の即投了など）は局番号を進めて作り直すので、その後ろの局の相手はずれうる。
  どの相手と指した局かは `pop_finished()` の `opponent` で分かる

### 模倣学習のデータセット（M2a）

```python
ds = tsuitate_rl.RecordDataset(paths)   # 対局記録（JSONL）のパスのリスト
ds.num_games, ds.num_attempts, ds.skipped   # 読めなかった局は (パス, 理由)
ds.signatures                            # 棋譜の署名（学習/検証の分割キー）
ds.game_attempts                         # 各局の試行数
(obs, mask, action, side, value, foul, game), outside = ds.encode_games([0, 1, 2])
```

- 読み込み時に真実を審判で再生し、**記録した側の観測が記録と一致し、勝敗が再生の裁定と
  一致する局だけ**を残す
- `encode_games` は指定した局の全試行（反則した試行を含む）を返す。`value` は手番側から見た
  終局の結果（勝ち +1 / 負け −1 / 引き分け 0）。1試行あたり約39KB なので、局は数十ずつ渡す

## 実測（2026-09-27、Apple Silicon、ランダム方策）

| n | 決定点/秒 | observe | step |
| --- | --- | --- | --- |
| 64 | 約10.3万 | 0.2ms | 0.4ms |
| 256 | 約11.5万 | 0.8ms | 1.5ms |
| 1024 | 約12.2万 | 3.1ms | 5.3ms |

ネットの推論を含まない環境側の上限。**マシンが空いているときの値**で、他の重い処理が
動いていると数分の1に落ちる（同じコードで 34µs → 96µs/決定点になった実測あり）。
速度を比べるときは同じ条件で対照も取り直すこと。
