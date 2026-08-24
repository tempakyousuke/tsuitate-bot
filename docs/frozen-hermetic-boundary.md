# 凍結境界と設定境界（issue #21）

2026-08-24。PR #20（checkpoint arena）のレビューで
「candidate arm 用の `TSUITATE_*` が同じプロセスの固定相手 `estimator_v14` にも効く」
ことが判明したのを受けた恒久対策。**何を共有し、何を凍結し、どこで検査するか**を
一箇所にまとめる。

## 1. 設定境界

`TSUITATE_*` は**構成境界で一度だけ**解釈し、以後は strategy instance が持つ
`crate::config::StrategyConfig` だけを見る（`src/config.rs`）。

| 部品 | 役割 |
| --- | --- |
| `EnvSource` | env の読み取り面。`std::env::var` と同じ戻り型なので既存の解釈式をそのまま移せる。`with_overrides` で arm 固有ノブを重ねる |
| `StrategyConfig` | 147 個のノブの解決値（`StrategyKnobs` / `EstimatorKnobs` / `CheckKnobs`）＋思考予算＋定跡パス |
| `StrategyConfig::fingerprint()` | **解決後の値**の sha256。未知のキーや既定値と同じ指定では変わらない |
| `config::scoped(&cfg)` | この config をこのスレッドへ設置（`EstimatorStrategy` の `choose` / `prewarm` / `oracle_anchor` の入口） |
| `config::ambient()` | プロセス env を一度だけ解釈した既定。config を設置しない経路（診断バイナリ・GUI）の互換 |
| `config::frozen_defaults()` | env を一切見ない既定。**v15 以降の凍結版**が使う |

設置がスレッドローカル・スコープつきなのは、arena / scenario が対局をスレッド並列に
回すから。`OnceLock` のプロセス全体キャッシュだと arm ごとの構築順で値が混ざる
（issue #21 方針1の「armごとにプロセスenvをset/removeする方式にはしない」）。

**候補側だけノブを変えても、プロセス env は動かない。** これが要点で、env を読み続ける
既存の凍結版 v6〜v14 は候補ノブに反応しない。

## 2. 誰が設定をどこから読むか

| 主体 | 読む先 | 備考 |
| --- | --- | --- |
| 現行 `estimator` / `estimator_rush` | instance の `StrategyConfig` | `make_seeded_with_config` / `EstimatorStrategy::with_config` |
| 凍結版 v6〜v14 | **プロセス env**（凍結時点の読み方） | 既知の負債。一覧は `frozen::env_keys_in_source` |
| 凍結版 v15 以降 | 自ファイルの `frozen_config()`（`EnvSource::empty()` から解決） | `scripts/freeze_estimator.py` が生成 |
| 共有モジュール（`opening.rs` の定跡パス） | 設置されている `StrategyConfig` | 設置が無ければ ambient |
| 審判（arena / checkpoint arena） | `ARENA_*` | 戦略設定とは別の名前空間のまま |

`strategy::honors_config(name)` が「config を尊重する戦略名か」を返す。
**凍結版へ config を渡すと `make_seeded_with_config` が `None` を返す**ので、
「設定したつもりで無視されていた」（PR #20 の事故）は起動時に止まる。

## 3. 凍結境界の分類

### 3.1 意図して共有し続けるもの

ルールのバグ修正は全バージョンへ反映されるべき、という従来方針のまま。

- `shogi.rs`（ルールエンジン）・`board.rs`（候補手生成）・`observation.rs`（観測）
- `protocol.rs`（サイト契約）・`model.rs`・`deduce.rs`・`mate.rs`

### 3.2 凍結すべきなのに共有しているもの（既知の負債）

凍結版が呼ぶモデル・特徴量は、更新するとその凍結版の**挙動が変わる**。

| ファイル | 依存する凍結版 |
| --- | --- |
| `src/likelihood.rs`（尤度係数 `FITTED_THETA`） | v12 / v13 / v14 |
| `src/value_nn.rs`（value ネットの重み） | v12 / v13 / v14 |
| `src/value_features.rs` | v12 / v13 / v14 |
| `src/belief_nn.rs` / `src/belief_features.rs` | v13 / v14 |
| `src/king_belief_nn.rs` | v14 |
| `src/opp_move_nn_v25.rs` / `src/opp_move_features.rs` | v12 / v13 / v14 |
| `joseki.json`（実行時に読むデータ） | v12 / v13 / v14 |

v9〜v11 は NN の重みを凍結ファイルへコピーしているので影響しない。
opp_move NN は 2026-08-21 の再学習時に `opp_move_nn_v25.rs` という固定コピーを
作って解決した（**先例**）。

**実際に起きた事故**: 2026-08-21 の value NN 再学習（commit `387f0ac`）は
v12〜v14 の挙動を変えている。当時は検知する仕組みが無く、CLAUDE.md の
ガントレット値もこの前後で厳密には比較できない。**この PR では検知だけ入れて
挙動は戻していない**（戻すと、再学習以後に測った基準値のほうが無効になる。
どちらを取るかは計測の運用判断なので勝手に決めない）。

### 3.3 検知の仕組み

`src/frozen/mod.rs`:

- `SHARED_MODEL_PINS` … 上表のファイルの sha256。**内容が変わるとテストが落ち**、
  「影響する凍結版」を名指しで出す。対応は (a) 固定コピーを作る (b) 承知でハッシュを
  更新し再計測を記録する、の2択
- `versions_using(module)` … 再学習の前に「何が動くか」を機械可読に出す
- `SOURCES` / `env_keys_in_source(name)` … 凍結版が読む env の一覧
  （checkpoint arena の実行前検査がこれを使う。表は一箇所だけ）

## 4. 既存 v6〜v14 の扱い

**一括編集しない。** 当時の計測結果と対応が取れなくなるため。

- 読む env の一覧は `frozen::env_keys_in_source` で機械的に取れる
- checkpoint arena / arena は arm 固有ノブを**プロセス env に置かない**ので、
  v6〜v14 は常に既定値で動く（`run_child` は継承した `TSUITATE_*` も落とす）
- 両側に等しく効かせたい設定（思考予算）は従来どおりプロセス env で渡す
- hermetic 規約は `frozen::HERMETIC_FROM`（= 15）以降に適用する

## 5. 生成規約（v15 以降）

`scripts/freeze_estimator.py` が次を行う。

1. `crate::config::current(|c| c.<節>.<名>)` → `frozen_config().<節>.<名>`
   （節ごとの Knobs 構造体と `from_source` は本文にコピーされるので、
   空の `EnvSource` から解決すれば**凍結時点の既定値**がそのまま出る）
2. `crate::config::ambient()` → `crate::config::frozen_defaults()`
   （共有モジュールが引く config も既定へ固定される）
3. 生成物に `env::var(` が残っていたら**生成時に失敗**

思考予算はプロセス env ではなく `EstimatorVN::with_budget_ms(seed, ms)` で明示的に渡す。

ガードは3本:

- `frozen::tests::新しい凍結版は戦略envを読まない` … v15 以降のファイルを走査
- `frozen::tests::freeze生成物は実行時envを読まない` … スクリプトを実際に走らせる
  （打ち切りの検出も兼ねる。`strip_file` が最初のテストモジュールでファイル末尾まで
  読み飛ばしていたバグをこの作業で発見した）
- `config::tests::現行戦略はプロセスenvを直接読まない` … 現行側の再発防止

## 6. ツールごとの設定規約

| ツール | 両側に効かせる | 候補側だけに効かせる |
| --- | --- | --- |
| `bin/arena` | `TSUITATE_*`（プロセス env。`-f env=`） | **`ARENA_CAND_KNOBS="K=V K=V"`**（config。凍結版が候補のときは使えない） |
| `bin/checkpoint_arena` | `--budget-ms`（子プロセスの env） | `--control-env` / `--candidate-env`（**config**。プロセス env は触らない） |
| `bin/scenario` | `TSUITATE_*`（プロセス env） | 対照も候補も同じプロセスで作るので、版比較は `-f env=` ではなく同一コミットのペア計測で |
| `bin/tune` | `TSUITATE_*` | SPSA は `EvalParams` を直接渡すので config を経由しない（調整対象ノブの env が立っていると起動時にエラー、は従来どおり） |

**`-f env=` はプロセス全体に効く**（凍結版も読む）ので、
「候補側だけ」を意図するときは上の右列を使うこと。

## 7. 残っている穴

- `joseki.json` は**パス**を config で固定できるが、**中身**は実行時に読む
  ファイルのまま。定跡を更新すると v12〜v14 の序盤分布が変わる
- 3.2 の共有モデルは検知するだけで、固定コピー化はしていない
- v6〜v14 は引き続きプロセス env に反応する。両側に等しく効くぶんには
  比較は壊れないが、**片側だけに効かせることはできない**
