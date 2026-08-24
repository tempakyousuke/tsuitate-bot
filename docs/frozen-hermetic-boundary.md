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
| `config::check_overrides()` | 明示ノブの検査。戦略が読まないキー（綴り間違い）と、実効値が変わらなかったキーを返す |

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

### 3.2 凍結版が依存する共有コード・モデル・データ

凍結版が呼ぶモデル・特徴量は、更新するとその凍結版の**挙動が変わる**。
すべて `SHARED_MODEL_PINS` で content hash を pin してある。

| ファイル | 依存する凍結版 | 種類 |
| --- | --- | --- |
| `src/opp_move_nn_v25.rs` | v12 / v13 / v14 | **固定コピー**（再学習しない） |
| `src/value_nn_v22.rs` | v12 / v13 / v14 | **固定コピー**（再学習しない） |
| `src/likelihood.rs`（尤度係数 `FITTED_THETA`） | v12 / v13 / v14 | 共有（pin のみ） |
| `src/value_features.rs` / `src/opp_move_features.rs` | v12 / v13 / v14 | 共有（pin のみ） |
| `src/belief_nn.rs` / `src/belief_features.rs` | v13 / v14 | 共有（pin のみ） |
| `src/king_belief_nn.rs` | v14 | 共有（pin のみ） |
| `src/opening.rs`（定跡の読み込み実装） | v12 / v13 / v14 | 共有（pin のみ） |
| `joseki.json`（実行時に読むデータ） | v12 / v13 / v14 | 共有（pin のみ） |

v9〜v11 は NN の重みを凍結ファイルへコピーしているので影響しない。

**実際に起きた事故**: 2026-08-21 の value NN 再学習（commit `387f0ac`）は
v12〜v14 の挙動を変えている（当時は検知する仕組みが無かった）。
opp_move NN は同じ再学習のときに `opp_move_nn_v25.rs` という固定コピーを作って
解決していたが、value NN は共有のままだった。

**対応（2026-08-24、ユーザー判断）**: 再学習前へは**戻さず**、
**現在の v12〜v14 の挙動を基準として pin する**。戻すと再学習以後に測った
基準値のほうが無効になるため。`src/value_nn_v22.rs` を切り出して v12〜v14 の
呼び先をそちらへ向けた。**この変更自体は挙動を変えない**（切り出し時点の
`value_nn.rs` と、数値 3629 個・forward の実装とも完全一致することを確認。
numpy クロスチェックのテストも固定コピー側で走る）。
**以後の value NN 再学習で v12〜v14 は動かない**。

### 3.3 検知の仕組み

`src/frozen/mod.rs`:

- `SHARED_MODEL_PINS` … 上表のファイルの sha256。**内容が変わるとテストが落ち**、
  「影響する凍結版」を名指しで出す。対応は (a) 固定コピーを作る (b) 承知でハッシュを
  更新し再計測を記録する、の2択
- `versions_using(module)` … 再学習の前に「何が動くか」を機械可読に出す
- `SOURCES` / `env_keys_in_source(name)` … 凍結版が読む env の一覧
  （checkpoint arena の実行前検査がこれを使う。表は一箇所だけ）
- `env_keys_read_by(name)` … **共有モジュール経由も含めた** env の一覧。
  `opening.rs` は config から定跡パスを引くだけでリテラルを持たないので、
  走査では拾えないぶんを `SHARED_MODULE_ENV` が明示する
- `behavior_fingerprint(name, env)` … 相手の**実効挙動**の指紋。
  版のソース sha256 ＋ その版が読む env の実効値 ＋ 共有モデルの pin から作る。
  現行 `StrategyConfig` の指紋を「相手の設定」として記録すると、凍結版は
  各ファイル内の旧既定値で動くので**相手名に依らず同じ値**になってしまう

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
2. `crate::config::ambient()` → 生成物が持つ `frozen_strategy_config()`。
   これは**凍結時点の定跡パスをリテラルで固定**した config
   （`const FROZEN_JOSEKI_PATH`）を設置する。`StrategyConfig::defaults()` を
   設置すると、将来この既定を変えたときに既存の凍結版まで追随してしまう
3. 生成物に `env::var(` が残っていたら**生成時に失敗**

定跡の**中身**（`joseki.json`）は `SHARED_MODEL_PINS` が content hash で見張る。
編集すると影響する凍結版を名指しでテストが落ちる。

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
| `bin/arena` | `TSUITATE_*`（プロセス env。`-f env=`） | **`-f cand_env=` / `ARENA_CAND_KNOBS="K=V K=V"`**（config。凍結版が候補のときは使えない） |
| `bin/checkpoint_arena` | `--budget-ms`（子プロセスの env） | `--control-env` / `--candidate-env`（**config**。プロセス env は触らない） |
| `bin/scenario` | `TSUITATE_*`（プロセス env） | 対照も候補も同じプロセスで作るので、版比較は `-f env=` ではなく同一コミットのペア計測で |
| `bin/tune` | `TSUITATE_*` | SPSA は `EvalParams` を直接渡すので config を経由しない（調整対象ノブの env が立っていると起動時にエラー、は従来どおり） |

**`-f env=` はプロセス全体に効く**（凍結版も読む）ので、
「候補側だけ」を意図するときは上の右列を使うこと。

明示的に渡したノブは `config::check_overrides` が検査する:

- 戦略が読まないキー（`TSUITATE_HAND_ASSET_WW` のような綴り間違い）は**起動時エラー**
- 実効値が変わらなかったキー（既定値と同じ値・解釈できない値・範囲外）は**警告**。
  checkpoint arena の `compare` は「arm ごとのノブが違うのに `arm_config` が同じ」を
  検出して止める（＝ candidate と control が実は同じ設定だった実験）

**seed の扱いはノブの有無で変えない**。config 付きの生成 API は seed を
`Option<u64>` で受ける（`make_with_config`）。`Some(0)` に落とすと
`ARENA_MATCH_SEED` 未指定の通常アリーナで候補だけ全対局が同じシードになり、
対照との比較がノブ以外の理由で崩れる。

## 7. 既定挙動の同一性確認

設定の解決を env から config へ移す変更なので、**既定（env なし）の挙動が
変わっていないこと**を2段で確かめた。

**(a) 解決式の一致（静的）**: 移行した 96 個のノブについて、config の初期化式が
旧アクセサの `OnceLock` 初期化式と（`std::env::var` → `src.var` の置換以外）
**文字列レベルで一致**することをスクリプトで確認した。残り4個
（`think_budget_ms` / `nonpromote_check_roles` / `regen_keep_is` /
`opp_pawn_intent_w`）は手で移して個別に照合。

**(b) アリーナ（動的）**: PR #20 と同じ条件で 104局×2基準、
`match_seed=20260815`、時計 1000+3、env なし
（[Arena run 32697854659](https://github.com/tempakyousuke/tsuitate-bot/actions/runs/32697854659)、
commit `886ef75`）。

| 基準 | 勝率 | 勝-負-分 | 詰み | 反則負け | 時間切れ | 手数上限 | 平均手数 | 反則/局(A) | think avg/p99max | クロック消費 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v13 | 56.7%±9.5 | 59-45-0 | 39 | 65 | 0 | 0 | 103.2 | 6.01 | 1232ms / 2858ms | 6.2% |
| v14 | 50.0%±9.6 | 52-52-0 | 53 | 51 | 0 | 0 | 98.5 | 6.24 | 1204ms / 2791ms | 5.8% |

同一 seed の直前の記録（PR #20、commit `6db84e9`、run 32650371131）は
vs v13 **61.5%±9.4**（64-40）/ vs v14 **51.0%±9.6**（53-51）、反則/局 5.67〜6.25、
思考平均 1195〜1258ms、時間切れ0、クロック消費 5.5〜6.5%。

差は vs v13 −4.8pt（5局ぶん）/ vs v14 −1.0pt（1局ぶん）で**どちらも CI の内側**。
反則/局・思考時間・クロック消費・時間切れ・手数上限はすべて直前の記録帯に収まる。

**これは決定論的な同一性の証明ではない**（壁時計ベースの思考予算で粒子数が揺れるので、
同一コード・同一 seed でも結果は動く。CLAUDE.md の凍結版同一性確認でも「同一コードで
100局 44% まで振れる」と記録がある）。同一性の直接の担保は (a) のほうで、
アリーナは「変質していないこと」の確認として読む。

## 8. 残っている穴

- **v6〜v14 は引き続きプロセス env に反応する**。両側に等しく効くぶんには
  比較は壊れないが、片側だけに効かせることはできない（hermetic 規約は v15 以降）
- 共有モデル（`likelihood.rs` / `belief_nn.rs` / `king_belief_nn.rs` /
  各 `*_features.rs` / `opening.rs` / `joseki.json`）は**検知するだけ**で、
  固定コピー化はしていない。更新するときに影響する凍結版を見て、
  固定コピーを作るか再計測を記録するかを選ぶ運用
- `joseki.json` は**パスを凍結版が固定し、中身は content hash で見張る**が、
  中身を変えたときに凍結版へ「凍結時点の定跡」を復元する仕組みは無い
  （必要になったら固定コピーを作る）
- `frozen::SOURCES` は凍結版ソース 2.6MB を `include_str!` でライブラリへ
  埋め込むので、本番 bot のバイナリもそのぶん太る（監査の一元化とのトレードオフ）
