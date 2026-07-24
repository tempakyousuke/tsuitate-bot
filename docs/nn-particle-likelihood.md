# 粒子尤度モデルのNN化（段階①の残り）: 実験記録

Status: 2026-07-23〜24 実施。**オフライン較正は大幅改善したがアリーナは
v10 と五分（強さ中立）**。ブランチ `nn-particle-likelihood` に完全な実装を
残して一区切り。マージ判断はユーザーに委ねる。

## 背景と動機

NNロードマップ段階①のうち `opp_move_weight` のNN化は v9 で完了済みだが、
もう半分の **`likelihood.rs`（粒子尤度、8特徴量線形 FITTED_THETA）のNN置換**は
未着手だった。粒子尤度は評価側 `strategy.rs::stratified_sample` が粒子重みに
乗じる exp(θ·φ) の θ を与えるモデルで、「王手駒ビリーフの較正不良」
（kakutori で真の王手駒への信念2.7%止まり）の対象領域とされていた。

## 実装（ブランチに完存、いつでも再開可能）

- `likelihood.rs::particle_nn_features` — 線形8特徴量を包含する26次元。
  駒種別home残存（from_home原則の駒種分解）・成駒/持ち駒・自分の利きとの
  相互作用（attacked_by_me/hanging/defended/王ゾーン攻撃数）・文脈特徴量
  （ply・my_dead・you_in_check。グループ内不変なのでsoftmaxでは相互作用と
  してだけ効く）。`ParticleCtx` に opp_moves/my_dead/you_in_check を追加
- `bin/export_particle_data` — fit_particles と同じ条件付きMLE抽出のCSV版
  （ベース対数重みオフセット付きグループsoftmax用）。game_id は1回の実行内
  でのみ一意（連結禁止）
- `particle_nn.rs` — 手書きforward pass（26→16→1）。特徴量抽出1.5µs/回
- tsuitate-nn: `train_particle.py`（線形 vs MLP、offset付きsoftmax NLL、
  対局単位分割、clamp(-15,15)を学習評価にも適用）・`export_particle_weights.py`
  （モデル生成直前のseedリセットで「比較したモデル=出荷モデル」を保証）・
  `scripts/crosscheck_particle.py`
- 統合: stratified_sample の logl を NN forward + clamp + 温度
  `PARTICLE_NN_SCALE`（env: TSUITATE_PARTICLE_NN_SCALE）に置換、
  **王手中は旧線形へフォールバック**

## 学習（データは再生成可能）

CI記録1512局（main の estimator vs v8/v9/v10 各500局余、run 30003482017）
→ 21,828決定点・128万行。4シードとも held-out val で頑健に大差:

| モデル | val NLL | 実効候補数 | 真実top-half |
|---|---|---|---|
| uniform | 4.00 | 55 | - |
| offset-only（現行ベース重み） | 5.05 | **156** | 48% |
| 線形（26特徴量） | 4.5 | 88-94 | 61-63% |
| **MLP（採用: seed0）** | **3.80** | **44.7** | 75.8% |

発見: ①現行のベース重み（フィルタ事後質量）は一様より悪い（較正不良の
定量化）。②線形は26特徴量でも一様に勝てず、**相互作用（NN）が必須**。

## 統合で判明した2つの落とし穴（本記録の主な価値）

1. **王手中の適用は有害**: 王手中グループはどの再重み付けも一様より悪い
   （NLL: 一様3.83 / offset-only 5.66 / 線形6.09 / NN 4.33）。NN適用だと
   kakutori 14/20→2〜8/20・keima 20/20→8〜15/20 と実測悪化。王手中の
   意思決定は CheckSolver の領分で、value_nn の you_in_check ゲートと同じ
   構図。→ 王手中は線形フォールバックで全シナリオ baseline 水準へ復帰
2. **ansatsu 回帰（scale=1.0）**: G*2e の1手詰め発見が 20/20→0/20。
   NNのグループ内 logit 振れ幅（中央値8.3）は線形（2.9）の約3倍で、
   「玉は深く進出していないはず」方向の割り引きが、観測を発生させない
   忍び込み玉の真実粒子の質量を潰す。温度応答は単調
   （1.0→0/20, 0.5→4/20, 0.35→8/20, 0.25→16/20, 0.15→19/20）で
   scale=0.25 でほぼ回復。**識別NLL最適 ≠ 意思決定最適**の実例

## アリーナ結果（採用判定）

- scale=1.0: vs v10 **51.5%±6.9** / v9 56.0±6.9 / v8 61.0±6.8
  （v8/v9への成績は v10 自身の凍結時と同水準 = v10 に上乗せなし）
- scale=0.25（ansatsu回復設定、match_seed=777）: vs v10 **51.8%±6.9**

**結論: 評価側の尤度再重み付けチャネルは飽和している。** オフラインの
判別を実効候補数156→45まで改善しても、温度によらずアリーナの強さに
変換されない。scenario suite は 0.25+ゲートで回帰なし（ansatsu 16/20 のみ
baseline 20/20 比でわずかに低い）。

## 解釈と次への示唆

- 評価（candidate ranking）は粒子重みの細部に対して頑健で、既存の線形
  尤度＋フィルタ事後質量で決まる順位を NN が覆すケースが勝敗に効くほど
  多くない、というのが最も素直な解釈
- 較正の改善が効くとしたら評価側でなく**フィルタ側**: リサンプリングの
  重み（C-7 logw）や再生成の提案分布（段階②信念ネットの本来の適用先）に
  尤度を入れれば、生き残る粒子集合そのものが変わる。ただし C-7 の
  再設計を伴うので慎重に
- 26特徴量セット・データパイプライン・「王手中ゲート」「玉位置質量の
  保護」の知見は段階②（信念ネット）にそのまま引き継げる
- 反則経済36ptギャップ（nofoulオラクル）は本変更でも未着手のまま

## 再開手順

1. ブランチ `nn-particle-likelihood` をチェックアウト（実装完備・全テスト通過）
2. データ再生成: 記録用アリーナ → `export_particle_data` → `train_particle.py`
   4シード → `export_particle_weights.py` → crosscheck期待値再生成
3. 検証: scenario 5件ローカル（王手中3件はゲートで不変のはず）＋
   ansatsu 温度チェック → CI suite → 200局ガントレット
