# 実戦局面の再現シナリオ

Shogi Quest からエクスポートした棋譜をそのまま置くと、`bin/scenario` が
任意の局面を再現して bot の選択・信念・終盤遂行を測定できる。

## 追加手順

### Shogi Quest の実戦棋譜から

1. Shogi Quest の棋譜（`*illegal:` 行込み）を `scenarios/<名前>.kif` に保存
2. ファイル内のどこかに1行足す:
   ```
   *scenario ply=69 diag=5g,4h desc=何を測る局面か
   ```
   - `ply=N` — N手目まで再生して **N+1手目を考えさせる**
   - `target=<USI>` — 注目手。省略時は棋譜で実際に指された N+1手目
   - `diag=<マス,マス>` — diag モードで相手利き枚数を測るマス（省略可）
3. 実行:
   ```
   cargo run --release --bin scenario -- <名前>              # 選択実験（20試行）
   cargo run --release --bin scenario -- <名前> diag         # 粒子の信念分布
   cargo run --release --bin scenario -- <名前> continue 10  # bot同士で終局まで
   cargo run --release --bin scenario -- suite               # 全シナリオの注目手一致率
   ```
   `--ply N` で同じ棋譜の別の局面をアドホックに試せる。

### tsuitate の対局DB（自分でbotと対局した記録）から

E2E検証手順（CLAUDE.md）でスクラッチDBを使ってbotと対局した後、
`games.moves`/`games.foul_attempts`（JSON列）を `{moves, foulAttempts}` 形式で
書き出し、`bin/make_scenario`（診断専用の一時ツール）でKIFへ変換する:

```
cargo run --release --bin make_scenario -- <moves.json> <sente|gote> <ply> [diag=マス,マス] > scenarios/<名前>.kif
```

`<moves.json>` は `{"moves": games.moves, "foulAttempts": games.foul_attempts}` の
JSON（DBの列をそのままJSON.parseして詰め直すだけでよい）。`sente|gote` は bot 側の手番。
局面確認には `bin/dump_position <moves.json> [手数]` で盤面・持ち駒を表示できる
（`*scenario` の直前の局面を目で確認したいときに使う）。

リプレイ時に全手・全反則試行を裁定検証（合法手は合法・`*illegal` は非合法）するので、
棋譜の欠落・コピペミス・パース誤りは実行時に即 panic で検出される。

## 収録シナリオ

- `keima.kif` — 29手目▲８五桂（王手）。同歩で桂を取り返せるか。
  事前分布の課題（馬@62 過大評価）の回帰テスト。目標: 捕獲プローブ率95%
- `kakunari.kif` — 70手目△５七角成（馬捨ての決め手）。
  粒子フィルタの中盤死（46-48手目で全滅）の回帰テスト
- `ansatsu.kif` — 30手目△４八飛打（観測ゼロの忍び込み=暗殺の一手前）。
  実は先手に１手詰め G*2e があり bot は 16/20 で発見（continue 20/20 勝ち。
  2026-07-17 keima-recapture ブランチ時点）。観測を発生させない忍び込みが
  粒子に映らない盲点（48/49 の利き0枚を93〜99%で誤信）の記録も兼ねる
- `kakudo.kif` — 22手目 R*2d（自玉と同じ2筋への飛車打ち。ユーザーによる
  実戦対局のレビューで指摘）。20/20 で R*2d を選択。`diag` で確認すると、
  未動の先手角（7九、対角線が完全に開通）が2四に利いているのに、粒子は
  1枚利き=2.5%・0枚=97.5%と正反対に誤信していた（真実は1枚）。
  「動いていない大駒の長い利き」を軽視する事前分布の課題の回帰テスト
  （2026-07-19、tsuitateスクラッチDB対局より `bin/make_scenario` で変換）
- `gold-check.kif` — 33手目 5八4八金（後手の成銀の位置は100%確定情報だった
  にも関わらず、その利きへ無防備に踏み込む手を20/20で選択。ユーザーによる
  実戦対局のレビューで指摘）。粒子の信念は正しいのに評価（exposed_capture_risk
  相当）が甘い、事前分布ではなく評価関数側の課題。NN方向（粒子上のvalueネット）
  の着手理由になった局面で、オフライン検証のベンチマークとして使う
  （2026-07-19、`docs/nn-value-phase1.md`参照）
- `kakutori.kif` — 48手目１五角（王手）を49手目でと金が取れるか。実戦では
  先手が反則（2925HI、王手を解消しない飛車の動き）を1回試みた後に正しく
  1d1e で捕獲した局面。合法な応手（捕獲1d1e・玉の脱出3手・飛車の合駒2i2f）を
  比較すると捕獲が唯一の得（net_capture_then_recapture=+8、他は0または-1.5）。
  NN方向のオフライン検証（`docs/nn-value-phase1.md`）で5seed全て捕獲を
  最高評価した一方、**このシナリオで実際に`bin/scenario`を回したところ
  estimator戦略（NN未統合の手作り評価）は捕獲を0/20回しか選ばず常に
  玉を逃がしていたことが発覚**（2026-07-20）。原因は`check.rs`の
  `CheckSolver`が王手駒の仮説を単純平均するため、粒子の王手駒ビリーフが
  誤っている（真の王手駒への信念1.7%）局面で正しい捕獲のp_legalが
  潰れる構造的な弱点（詳細はCLAUDE.mdの`check.rs`節）。
  `CheckSolver::captures_checker`+p_legal下限で対応し、捕獲選択率
  0/20→10-11/20に改善、`estimator_v8`として凍結（2026-07-21）。
  「王手解消失敗」系の反則・CheckSolverの仮説希釈問題の回帰テストとして
  常設する

## 注意

- `*illegal:` 行は「**直前の指し手行の手番側が、その手を指す前に試みた反則**」
  （Shogi Quest の出力規約。実棋譜2件・全12箇所で検証済み）
- 棋譜のコピーはターミナル貼り付けだと欠落しやすい。ファイルで受け渡すこと
