# NN方向 フェーズ1: 粒子上のvalueネット（データ基盤とオフライン学習）

Status: 2026-07-20 着手・同日いったん区切り（下記「区切り時点のまとめ」参照）。
オフライン検証は既知シナリオ2件でgold-check 5/5・kakudo 4/5まで到達したが、
厳密な成功条件（両シナリオとも5/5で安定）には未達。bot本体への推論統合は
未着手のまま。

## 背景

2026-07-19の対局レビューで、`evaluate()`の手作りヒューリスティック
（combine_score/gain）が「堅実な手 vs 派手だが危険な手」を正しく序列化
できないケースが見つかった（`scenarios/gold-check.kif`: 33手目 5八4八金。
後手の成銀の位置は100%確定情報だったにも関わらず、その利きへ無防備に
踏み込む手を20試行中16回選んだ）。粒子フィルタ自体の信念は正しかったので、
問題は「評価」側にある。

以前の議論（メモリ`nn-direction-deferred`、参照元セッションの記録）で、
NN化は4段階（①相手モデル→②信念ネット→③粒子上のvalueネット→④フルRL）に
分けて検討する合意があった。今回の症状は評価そのものの質が原因なので、
①②を飛ばして**③（粒子上のvalueネット）から着手する**。粒子生成
（estimator.rs）は一切変更せず、「仮説局面（真の情報を仮定した1粒子）の
良し悪しを判定する関数」を手作り式から学習済みモデルへ置き換える、
という限定されたスコープ。

**フェーズ1はデータ基盤＋オフライン学習まで**。bot本体への統合（推論を
evaluate()へ組み込む）はオフラインで学習済みモデルの判断が妥当だと
確認できてから、別セッションで着手する。

## リポジトリ境界

- **tsuitate-bot（このリポジトリ）**: 特徴量抽出定義・学習データ書き出し・
  （将来の）推論統合。学習/推論で特徴量定義がズレないよう、抽出コードは
  ここに一本化する
- **tsuitate-nn（`~/Develop/tsuitate-nn`）**: Python/PyTorch の学習パイプライン。
  tsuitate-botが書き出した特徴量ファイルを読み、ONNXモデルを書き出す。
  tsuitate-botには依存しない（データファイル経由の疎結合）

## 実装したもの（tsuitate-bot側）

- `src/value_features.rs` — 真の局面（両者視点）の12次元特徴量
  （`VALUE_FEATURES`/`VALUE_FEATURE_NAMES`/`value_features()`）。
  `likelihood.rs::particle_features`と同じ発想（手作り特徴量、名前付き配列）
  だが、あちらは粒子1個の尤もらしさ用の相手視点特徴、こちらは局面優劣用の
  両者視点特徴。`strategy.rs`の`king_zone_pressure`/`drop_check_danger`を
  `pub(crate)`化して再利用
- `src/bin/export_value_data.rs` — 対局記録（records/*.jsonl、
  ARENA_RECORD_DIR互換）から真の棋譜をreplayし、各手番の局面ごとに
  (value_features, ラベル) をCSVで出力する。ラベルはその手番側から見た
  対局結果（勝ち=1.0・負け=0.0・引き分け=0.5）
- データ生成は新規ドライバ不要: 既存の`ARENA_RECORD_DIR`（`selfplay.rs`の
  `GameTruth`）が両者の真の手順・反則試行をそのまま記録している。
  `cargo run --release --bin arena -- <局数> estimator estimator` を
  `ARENA_RECORD_DIR`付きで回すだけで自己対局データが作れる

## オフライン検証（フェーズ1のゴール）

学習したモデルを `scenarios/gold-check.kif`（33手目、真の局面が既知）に
適用し、「5八四金」より「地味な歩の前進」を高く評価するか確認する。
これがフェーズ1完了の合否基準。既存の`scenarios/*.kif`（kakudo等）でも
同様のオフラインチェックを行う。

## フェーズ1に含まないもの

- bot本体（strategy.rs）への推論統合
- ONNX推論クレート（ort/tract等）の選定・実装
- 段階①②（相手モデルNN・信念ネット）

## 結果（2026-07-20）

実装中に発見: 当初の12特徴量には「今動かした駒が取られる危険」を表す項が
一つも無く、動機となった局面を判別できないという欠落があった。
`max_hanging_value`（相手の利きが当たり自分の紐が無い駒の最大価値）を
自分側・相手側の両方に追加し14特徴量に拡張（`887831f`）。

自己対局600局（`estimator` vs `estimator`、既存のarena.yml + ARENA_RECORD_DIR
をそのまま利用、新規ドライバ不要）から44k行を書き出し、tsuitate-nn側で
小さいMLP（隠れ層64→32、300epoch、GPU不要・CPUで数秒）を学習。
val_mse=0.2256（ベースラインの分散0.25からわずかに改善、まだ弱い）。

**オフライン検証は初回「合格」だったが、これは誤りだった（訂正参照）**。
初回の val_mse=0.2256 / 検証結果（33手目5八四金 vs 歩の前進、kakudo局面
R*2d vs P*2h、いずれも堅実な手を正しく評価）は、**行単位のランダム分割
（同一対局の別手番が学習/検証の両方に混入）+ 正規化統計を分割前の
全データで計算**という2つの手法上の欠陥による見かけ上の結果だった。

## 訂正（codexレビュー、2026-07-20）

codex（gpt-5.5、read-only）にtsuitate-bot・tsuitate-nn双方をレビュー依頼し、
以下を指摘された（要点、Highのみ）:
- 検証データ漏洩: 正規化統計（mean/std）を`train_test_split`より前の
  全データで計算していた
- 同一対局の全手番を独立サンプルとして扱い、`game_id`が無いため対局単位の
  分割ができなかった（自己相関・データリーク）
- （Medium）300epoch固定で最終epochのモデルを保存していた
  （best val時点ではない）
- （Medium）`eval_candidates`の`me`指定に、実際の手番と一致するかの
  ガードが無かった
- （Medium）2件のシナリオの差（0.009〜0.015）はval RMSE（≈0.475）に
  対してかなり小さく、「正しく評価できた」を証明する強さの証拠ではない

**修正**（`export_value_data.rs`に`game_id`/`ply`列追加・全手replay成功時
のみ出力、`train.py`に`GroupShuffleSplit`（対局単位）・学習split行のみで
正規化統計計算・best val時点のstate_dictを保存、`eval_candidates.rs`に
`me`と実際の手番の一致チェックを追加）した上で**再学習・再検証したところ、
**両方のシナリオが逆転した**（5八四金がむしろ高評価0.408 vs 0.405/0.405、
R*2dもP*2hより高評価0.482 vs 0.465）。val_mseも学習が進むにつれ
epoch90付近から悪化する明確な過学習カーブが見えるようになった
（修正前は隠れていた）。

**結論**: 初回の「合格」は手法上の欠陥（データリーク）による見かけ上の
結果で、実際には600局・14特徴量・300epochのこの初回モデルは
まだ「堅実な手を正しく評価する」ところまで学習できていない。
これは想定内の結果（初回の最小規模実験としては妥当）だが、フェーズ1の
成功条件は未達のまま次のステップに進む必要がある。

**次のステップ**: データ量を大幅に増やす（1万局規模。600局・90対局の
held-outでは分散が大きすぎる）・過学習対策（正則化・epoch数の見直し・
early stopping）・自己対局の相手多様化（同一戦略同士だけでなく凍結版や
ノイズ入りの手も混ぜる。codexの指摘: 現状の自己対局は「現在の方策が
訪れる分布」の状態価値しか学習できず、実際に指されなかった対抗手の
価値を学習しにくいoff-policyの弱点がある）。bot本体への推論統合は
これらが解決してから検討する。

## 3000局・過学習対策後の再検証（2026-07-20）

**過学習対策**: `model.py`にdropout(0.2)、`train.py`にweight_decay(1e-4)・
ミニバッチ化(batch_size=256、元は全データ一括のフルバッチ勾配降下だった)・
early stopping(patience=30エポック)を追加。

**データ拡大**: `arena.yml`をcandidate=estimator、baselines=
"estimator estimator_v6 estimator_v7"（各1000局、計3000局）で実行し、
自己対局の相手を多様化（同一戦略同士だけでなく凍結版v6/v7とも対戦させる。
上記「次のステップ」の対抗手多様化を部分的に反映）。229,170行に拡大
（600局44k行の約5倍）。

**オフライン検証ツールの改善**: `eval_candidates`が`.kif`を直接読めるように
した（`parse_kif`を再利用。従来は`moves.json`への変換が別途必要だった）。

**結果**: seed=0で学習した1本だけでは両シナリオとも正しい順序
（gold-check: 歩の前進0.364 > 金打ち0.350、kakudo: P*2h 0.568 > R*2d 0.558）
だったが、**val_mseがまだ高く（0.22台、ベースライン分散0.25からわずかな
改善）差も小さいため、seed=0〜4の5本で頑健性を確認したところ結果が割れた**:

- **gold-check（歩の前進 vs 金打ち）**: 5シード全てで正しい順序
  （差+0.001〜+0.014、小さいが常に同じ向き）。頑健に学習できていそうな数少ない例
- **kakudo（P*2h vs R*2d）**: 5シード中1つしか正しい順序にならず、
  残り4つは**逆方向に頑健**（R*2dを一貫して高評価、差-0.003〜-0.038）。
  初回seed=0の「合格」はたまたまで、モデルは実際には
  「動いていない大駒の長い利きへ踏み込むリスク」を学習できていない、
  もしくは逆方向に学習してしまっている

**考察**: 14特徴量にはこの区別を表現する項が無い可能性がある。
`R*2d`/`P*2h`はどちらも真の局面上では`my_max_hanging=3.5`と同値
（新しく打った駒自体は「ハングしている（詰み取りされる）」判定には
掛からない＝別の駒で取り返せる）。しかし取り返しても駒種の損得
（飛車を切って角を得る／歩を切って角を得る、では損得が違う）までは
`max_hanging`（two値=攻撃側の最大評価額）では表現できない。
`strategy.rs`の`exchange_value`（盤上価値+持ち駒価値の平均）に相当する
「取り返された場合の交換損得」を表す特徴量が欠けている可能性が高い。

**結論**: フェーズ1の成功条件（複数シナリオを頑健に正しく評価する）は
依然未達。1本のseedだけでの検証は誤判定リスクが高い（今回がまさにその例）
ので、**以後のオフライン検証は複数seedでの頑健性チェックを必須とする**。
次の一手としては、データをさらに増やす前に、`exchange_value`相当の
特徴量を追加してkakudoの再学習で改善するか確認する方が費用対効果が
良さそうだが、方針はユーザーと相談してから決める。

## exchange_loss特徴量の追加とcodexレビュー2往復（2026-07-20 続報）

ユーザーの了承を得てcodex（gpt-5.5、read-only）に相談しながら進めた。

**1周目の指摘（distribution mismatch）**: `export_value_data.rs`は常に
`me == pos.turn()`（指す前の局面・指す側視点）で学習しているのに、
`eval_candidates.rs`は着手後（`me`は指した側のまま、`pos.turn()`は相手）を
評価しており分布がズレていた。`value_features(&pos, pos.turn())`（着手後・
相手視点）に修正して同じ5seedで再検証したところ、**gold-check 5/5→4/5、
kakudo 1/5→0/5と悪化**（後述の通りこの修正の方向自体が誤りだった）。

**exchange_loss特徴量の追加**: `strategy.rs::exchange_value`を`pub(crate)`化し、
`my_max_exchange_loss`/`opp_max_exchange_loss`（盤面全体で最悪の交換損失）を
16特徴量に追加。単体テストでは意図通り動いたが、**実際のkakudo局面では
無関係などこか別の駒の浮き駒(3.5)がmaxを支配し、R*2d/P*2hを全く区別できな
かった**（`max()`型特徴量が「盤面全体のworst-case」しか表せない構造的限界。
`my_max_hanging`も同じ限界を最初から抱えていた可能性）。

**2周目の指摘（着手固有のtransition特徴量）**: codexに相談したところ、
「盤面全体max」ではなく「直前に動いた/打たれた駒だけ」に絞った
transition特徴量を提案された。`value_features.rs`に
`transition_features(before, mv, after, mover)`を新設（
`moved_piece_value`/`moved_piece_hanging_value`/`moved_piece_exchange_loss`/
`captured_value`/`net_capture_then_recapture`/`gives_check`の6項目）。
**このタイミングで1周目の分布ズレ修正が誤りだったと判明**: 学習側は
「指す前・指す側視点」が一貫した規約なのに、1周目の修正は「指した後・
相手視点」に変えてしまっていた。正しい修正は`eval_candidates.rs`のstateを
`base`（着手前、me視点）に固定することだった。ここを直し、
`export_value_data.rs`にもtransition_featuresを追加（22特徴量）。

生特徴量レベルでは狙い通りの結果: gold-check（5i4h悪手 net=-2.0 vs 5g5f好手
net=0.0）・kakudo（R*2d悪手 net=-1.5 vs P*2h好手 net=-1.0）いずれも正しい
向き・明確な差が出た。

**しかし3000局・5seedで再学習すると、kakudoは0/5→3/5に改善した一方、
gold-checkは5/5→1/5に悪化した**。原因を切り分けるため学習データ全体で
transition特徴量とlabel（最終勝敗）の相関を見たところ、ほぼゼロ〜符号が
直感と逆だった:

```
moved_piece_exchange_loss  corr=+0.053  (直感的にはマイナスのはず)
net_capture_then_recapture corr=-0.013  (ほぼゼロ)
gives_check                corr=+0.087
```

**codexとの結論**: これは典型的なcredit assignment問題。自己対局
（estimator/v6/v7、いずれも一定以上強い）では壊滅的な一手損は稀で、
「駒を危険にさらす手」は同時に「攻めている・すでに優勢な局面」でもある
ことが多く、最終勝敗という1局70手超に1個しかないラベルでは、着手固有の
リスクの因果効果が交絡・ノイズに埋もれてしまう。データを1万局に増やしても
交絡の向きが同じ分布で増えるだけで符号は直りにくいと予想される。

**次の一手（codex推奨、費用対効果順）**: (a)1万局への単純拡大は優先度低い
（分散低減にはなるが交絡は直せない）。(b)TD的ブートストラップはまだ早い
（現モデルが未熟で自己参照バイアスを固定するリスク）。(c)**pairwise補助
loss**が最有力: 通常の勝敗回帰は残しつつ、`moved_piece_exchange_loss`等が
大きく異なり他条件が近い候補ペアを自己対局局面から自動抽出し、
`score(good) > score(bad) + margin`を補助lossとして加える
（gold-check/kakudoは固定回帰テスト兼アンカーとして使う）。(d)いったん
保留も選択肢。**codexの見立て: 「純粋なvalue回帰だけで候補手選択まで
行う」という設計は損切り寄りだが、transition特徴量自体の表現力は
シナリオレベルで実証済みなので、pairwise補助を足す方向でならもう一段
試す価値がある**。これを1本やって5seedでgold-check/kakudoが両方安定
しなければ、フェーズ1のこの路線は保留とする。

**教訓**: 「学習データの規約（この場合: 常に指す前・指す側視点）」を
最初にはっきり固定してから特徴量を足すべきだった。1周目の分布ズレ
修正は、規約を確認せず場当たり的に「pos.turn()と揃える」ことだけを
目的にしてしまい、結果的に規約と逆方向に直してしまった。

**bot本体への推論統合はまだしていない**（フェーズ1の成功条件は未達のまま）。

## pairwise補助lossの実装と結果（2026-07-20 続報3）

ユーザーの了承を得てcodex推奨案(c)を実装。`bin/export_pairwise_data.rs`を
新設し、同一局面（3000局・229,170局面）で全合法手を`legal_moves()`列挙、
`transition_features`の`net_capture_then_recapture`（それ自体は単体で
正しい向きを検証済みの手作り特徴量）が最大の手と最小の手のペアを抽出
（差0.5未満は除外）。210,881ペア。実行時間3000局で約10秒。

`train.py`に`--pairwise-data`オプションを追加。同じstate（局面共通なので
ペア内で不変）に対し`score(good) > score(bad) + margin`のhinge lossを
補助項として通常のMSE lossに加算（既定 weight=0.3, margin=0.05）。
pairwise側もmain dataと同じgame_id分割を流用し、held-out局のペアが
学習に混じらないようにした。

**結果**: pairwise検証精度（held-outペアでgood>badを正しく順序づけられた
割合）は学習開始直後から99%超に到達——手作り特徴量の向き自体は
容易に学習できることが確認された（credit assignment問題の診断が
正しかったことの傍証）。5seed再検証:

- **gold-check**: 1/5 → **4/5正解**に改善（seed4のみ逆転、-0.0087の僅差）
- **kakudo**: 3/5 → **3/5**（改善なし。seed1 -0.0348、seed4 -0.0283で逆転）

3段階の比較表:

| 設計 | gold-check | kakudo |
|---|---|---|
| state特徴量のみ（16次元） | 5/5 | 1/5 |
| +transition特徴量（22次元、補助lossなし） | 1/5 | 3/5 |
| +pairwise補助loss | **4/5** | **3/5** |

pairwise補助loss導入でここまでの最良の結果になったが、codexが事前に
置いた基準（両シナリオとも5/5で安定）には届いていない。フェーズ1の
成功条件は依然未達。次にやるとしたら候補は: pairwise_weight/marginの
調整、pairwise抽出基準の改善（現状はnet_capture_then_recaptureのみで
選んでいるが他のtransition特徴量も混ぜる）、データ拡大（1万局）など。
ここでいったんユーザーに結果を報告し、続行するか判断を仰ぐ。

## 実装コードのcodexレビュー（2026-07-20 続報4）

ユーザーから「design相談だけでなく実装コード自体のレビューは受けたか」と
指摘され、design相談とは別に実装済みコードのレビューを依頼した。
致命的な符号反転やリークは無かったが、Medium 3件・Low/Medium 1件の指摘:

- **Medium（修正済み）**: `transition_features`の`captured_value`が
  `piece_value`のままで`exchange_loss`側の`exchange_value`と基準が
  不整合だった（と金等の成駒捕獲を過大評価し`net_capture_then_recapture`が
  歪む）。`exchange_value`に統一
- **Medium（近似として許容、コメントで明記）**: `min_attacker_exchange_value`は
  `attacks()`（利きの有無）だけを見ており、ピンで動けない駒・取ると自玉が
  王手になる駒も攻撃駒に数える。既存の`max_hanging_value`と同じ近似方針を
  踏襲しているが、pairwiseの教師信号としてはノイズ源になりうる
- **Medium（修正済み）**: `train.py`のpairwiseデータのgame_id対応が暗黙で、
  主データと別コーパス/別順序で生成されたpairwise CSVを渡すと番号が
  偶然一致してリーク・誤分割になりうる。game_id集合の包含関係を検査する
  ガードを追加
- **Low/Medium（修正済み）**: モデル選択・early stoppingがval_mse単体で
  行われており、pairwiseの目的（候補手ランキング改善）を見ていなかった。
  `select_score = val_mse + pairwise_select_weight * (1 - pw_val_acc)`
  （既定weight=0.05）による合成スコアに変更

修正後、`captured_value`の変更を反映してデータ再生成・5seed再学習・
再検証したところ、**結果は変わらず安定**（gold-check 4/5、kakudo 3/5）。
このシナリオペアには捕獲を伴う候補が無かったため、captured_valueの修正が
直接効かなかったのは想定通り。

## pairwise_weightの引き上げで大幅改善（2026-07-20 続報5）

ユーザーから「残る不安定さ（4/5・3/5）はデータ不足ではなく綱引き
（MSE回帰とpairwiseが同じ重みを取り合っている）が原因では」という
仮説の妥当性を再度codexに確認。**codexの見立て: 仮説は妥当だが、もう1つ
「pairwise検証精度99%はpairwiseデータ自体が簡単すぎる（極端なペアばかりで
gold/kakudoのような僅差比較になっていない）ため」という可能性も示唆され、
最初に試すべきは高コストな対策（サブネット分離）ではなく安価な
`pairwise_weight`/`margin`のグリッド探索・pairwise-onlyアブレーション**、
との助言を得た。

`pairwise_weight`を0.3→20.0（margin 0.05→0.1、実質pairwise優位の
アブレーション）に引き上げて5seed再学習・再検証したところ、**劇的に改善**:

- **gold-check**: 4/5 → **5/5正解**。マージンも+0.16〜+0.21と大幅拡大
  （weight=0.3時の+0.001〜+0.013から一桁増）
- **kakudo**: 3/5 → **4/5正解**。唯一の不正解(seed2)も-0.0015とほぼ引き分け
  （weight=0.3時にはっきり逆転していたケースが僅差の際どい判定に変化）

val_mseはわずかに悪化（0.221〜0.227 → 0.222〜0.231）したが、候補手
ランキングの改善幅に対して軽微。**「綱引き」仮説がほぼ裏付けられた**:
pairwiseの重みを上げるだけでtransition特徴量の学習された符号・強度が
大きく安定した。

4段階通しの比較表:

| 設計 | gold-check | kakudo |
|---|---|---|
| state特徴量のみ(16次元) | 5/5 | 1/5 |
| +transition特徴量(22次元、補助lossなし) | 1/5 | 3/5 |
| +pairwise補助loss(weight=0.3) | 4/5 | 3/5 |
| +pairwise補助loss(weight=20.0) | **5/5** | **4/5** |

現状の`out/`はこの設定（weight=20.0, margin=0.1, seed=0）で再学習済み。
codexが事前に置いた基準（両シナリオとも5/5で安定）にほぼ到達
（kakudoが4/5、しかも不正解ケースはほぼ引き分け）。まだ厳密な5/5×5/5
ではないので、フェーズ1の成功条件としては「ほぼ達成」の位置づけ。

## weight/marginの微調整（2026-07-20 続報6）

ユーザー了承のもと`pairwise_weight`×`pairwise_margin`の小グリッド
（weight={5,10,20,40}×margin={0.05,0.1,0.2}、計12通り、まずseed=0のみ）を
実施。**gold-checkは12通り全てで正解**（かなり頑健）。kakudoは
margin=0.1の列（w5/10/20/40）だけ4通り全て正解で、margin=0.05/0.2は
不安定（8通り中4通りのみ正解）——**margin=0.1が明確に安定**という
パターンが出た。margin=0.1の中ではweight=40が最大マージン(+0.0457)
だったため5seedで確認したが、weight=20と同じ**kakudo 4/5**止まり
（不正解のseedが変わっただけ）。val_mseもweight=20よりわずかに悪化。

**結論**: weight=20〜40・margin=0.1あたりが局所的な最適域で、これ以上の
グリッド探索では頭打ち（gold-check 5/5・kakudo 4/5から動かない）。
**採用: weight=20.0, margin=0.1**（val_mseがわずかに良く、既にout/へ
反映済み）。kakudoを5/5まで詰めるには、codexが提案した他の対策
（僅差ペアを混ぜたhard pairwise mining、state/transition別サブネット
分離）が必要と判断し、weight/margin微調整はここで打ち止めとする。

## 区切り時点のまとめ（2026-07-20）

ユーザーの意向でここまでを一区切りとする。以後この作業に戻るときの
起点として現状を整理する。

**到達点**: state特徴量(16)+transition特徴量(6)=22次元、pairwise補助loss
（weight=20.0, margin=0.1）付きMLPで、既知シナリオ2件のオフライン検証が
gold-check 5/5・kakudo 4/5（5seed中）。厳密な成功条件（両方5/5）には
一歩届かないが、「地味な歩の前進 vs 危険な金打ち」「深い利きを警戒した
歩打ち vs 無警戒な飛車打ち」という、当初の動機（33手目5八四金）と
同種の判断は概ね学習できている。

**変更ファイル（tsuitate-bot、コミット済み dafb2d0）**:
`src/value_features.rs`（transition_features新設）・
`src/bin/export_value_data.rs`・`src/bin/eval_candidates.rs`・
`src/bin/export_pairwise_data.rs`（新規）・`src/strategy.rs`
（exchange_value を pub(crate) 化）・本ドキュメント・`.gitignore`

**変更ファイル（tsuitate-nn、コミット済み 691129a）**:
`train.py`（pairwise補助loss）・`model.py`（dropout・INPUT_DIM）・
`eval_scenario.py`。`out/`は`weight=20.0, margin=0.1, seed=0`で
再学習済みの現時点のベストモデル（gitignore対象、リポジトリには無い。
再学習手順は本ドキュメント参照）。`data/value_data_3000.csv`・
`data/pairwise_3000.csv`は3000局（estimator/v6/v7混合）から生成
（`.gitignore`でcsvは除外済み、再生成手順は本ドキュメント参照）

**追記（2026-07-21）**: 下記候補3（新規シナリオの追加）に着手し、
`scenarios/kakutori.kif`を追加したところ、NN検証とは別に**estimator戦略
本体（手作り評価、NN未統合）のバグ**を発見した（王手をかけてきた駒を
タダで取れるのに取らない）。原因は`check.rs::CheckSolver`の仮説平均化
（詳細はCLAUDE.mdの`check.rs`節）で、修正して`estimator_v8`として凍結
（vs v6 71.3%±8.8% / vs v7 62.5%±9.3%、100局）。フェーズ1（NN統合）
自体の進捗ではないが、シナリオ拡充が想定外の価値を生んだ例として記録。

**未着手（フェーズ1のスコープ外のまま）**: bot本体（strategy.rs）への
推論統合、ONNX推論クレートの選定。統合を検討するなら、まず今回の
gold-check/kakudo同様に既知シナリオでの頑健性を確認できるベンチマークを
もう数件増やしてからの方が安全（2件だけだと過学習/たまたま一致のリスクが
残る）。

**次に着手する場合の候補（優先度は未確定、ユーザーと要相談）**:
1. hard pairwise mining（僅差の候補ペアを混ぜる。現状はnet_capture_then_recapture
   最大/最小の極端ペアのみ）
2. state/transition別サブネット分離（MSE勾配とpairwise勾配の綱引きを構造的に解消）
3. 新規シナリオの追加（gold-check/kakudo以外の既知の弱点局面をシナリオ化し
   ベンチマーク数を増やす）
4. データ拡大（1万局。ただしcodex・実測ともに優先度は低いと判断済み）
