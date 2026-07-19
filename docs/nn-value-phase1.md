# NN方向 フェーズ1: 粒子上のvalueネット（データ基盤とオフライン学習）

Status: 実装中（2026-07-20 着手）

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
