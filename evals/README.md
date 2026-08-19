# evals/ — 採点式の対局レビュー

実戦棋譜の決定点ごとに bot 候補手を **0〜10 点**で採点するレビューの置き場。
従来の「悪手かどうか」の二値（`bad=`）を一般化したもので、SPSA の目的関数
（`TUNE_OBJECTIVE=scenario_score`）と suite の「平均得点」の教師になる。

## 形式（*.eval.md）

```
# eval: quest_20260731
# 点数: 0〜10（10=最善級 8=あり 6=ぎりぎり 4=悪手だがマシ 2=悪手 0=論外）。未採点は ?

## 61手目（先手番）
4七金(5g4g) 2 4七を守る唯一の駒自身を動かす手
3六角打(B*3f) 8 あり
2二歩打(P*2b) ?
（ブロック内の自由記述行はそのまま保全される）

### 62手目（4七歩打(P*4g)の反則後）
...
```

- 候補行: `和名(USI) 点数 コメント…`。点数は 0〜10 の整数か `?`（未採点）
- 見出し: `## N手目（…）` / 反則後は `### N手目（和名(USI)の反則後）`
- USI を含まない行は自由記述として保全される（採点の根拠メモに使う）

## 点数の目安

| 点 | 意味 | 旧語彙 |
|---|---|---|
| 10 | 最善級 | 本命 |
| 8 | 良い | あり・自然な手・当然の一手 |
| 6〜7 | 指せる | 悪くない・セーフ・ありだが留保つき |
| 4〜5 | 緩いが咎めにくい | 悪手だが比較的マシ・なくはない |
| 2 | 悪手 | 悪手 |
| 0 | 論外 | 論外・大悪手・王手放置 |

**未採点（?）の手が選ばれたときは仮に 4 点**で数え、件数が表示される
（`scenario_core::UNSCORED_DEFAULT`）。方策が評価済み候補の外へ出ると
計測不能になる抜け穴（2026-08-06 の実測）への対応なので、未採点の選択が
目立ったら `make_eval` を再実行して候補を追記し、採点すること。

## ワークフロー

1. **スケルトン生成**（新しい棋譜のレビュー開始・候補の追記）:
   `cargo run --release --bin make_eval -- [--ply N] <kif> [eval出力] [top_n=15] [seed=0]`
   全決定点（反則後サブ状態含む）の候補上位と実戦手を列挙する。**冪等**:
   既存の採点・コメントは保全し、未収載の候補だけ追記する（ランキングの
   揺れで再実行ごとに数行増えることはある）。重い（1棋譜 ≒ 対局1局ぶん）。
   `--ply N` で決定点を1つ（kif の `*scenario ply=` と同じ数え方）に絞れる
   = **単一決定点シナリオ**（`scenarios/arena-check-*.kif` のような
   1 kif = 1 決定点のもの）用。eval は `evals/<シナリオ名>.eval.md` に置き、
   sync_eval は eval の stem と同名の kif があれば `## N手目` を N == ply+1 の
   ときだけそのシナリオへ、反則後ブロックは `fouls=` 末尾が一致する
   `<名>f<k>.kif` へ同期する（foul_blocks の表は不要）。
   arena-check の eval には採点の参考として真実の盤面（`bin/scenario <名> board`
   の出力）を引用で入れてある（パーサは `和名(USI) 点` 行しか読まないので無害）
2. **採点**: `?` を 0〜10 に書き換える（コメント任意）
3. **同期**: `python3 scripts/quest_review/sync_eval.py [evalパス]`
   シナリオの `scores=`（採点全量）と `bad=`（**2点以下**）を更新する。
   反則後ブロックとシナリオの対応は `scripts/quest_review/foul_blocks.py` の
   FOUL_MAP に登録する（sync_eval / append_unscored / rerank_eval の共有表）
4. **計測**: `bin/scenario` が不合格計と並べて「平均得点 x.xx/10」を表示。
   SPSA は `TUNE_OBJECTIVE=scenario_score`（score = 平均得点/10 の
   シナリオ平均。試行シードは 0..trials の固定列で f+/f− は共通乱数ペア）

## 未収載候補の探索（rerank_eval.py）

`scripts/quest_review/rerank_eval.py` は「同じ USI の他の決定点での平均点」を
事前値として `prior + alpha × エンジンscore` で rank_dump を並べ替え、
現行方策とは違う手が上位に来る決定点を洗い出す**オフライン診断**。
出力は `scenario suite --tsv` と同じ TRIAL TSV なので、ファイルへ落として
そのまま `append_unscored.py` に食わせられる。

```
RANK_DUMP_SCORES=1 target/release/rank_dump <kif> <開始> <終了> > /tmp/rank.txt
mkdir -p /tmp/rerank
python3 scripts/quest_review/rerank_eval.py evals/quest_20260731.eval.md \
  /tmp/rank.txt --stop-unscored 10 > /tmp/rerank/result.txt
python3 scripts/quest_review/append_unscored.py \
  evals/quest_20260731.eval.md usi-prior-rerank /tmp/rerank --min 1
```

- 事前値は局面を見ないので、**高得点の USI がどの手目でも上位に来る**。
  発見される候補は少数の USI に集中する（網羅的な探索の道具ではない）
- stderr の「平均 x.xxx/10」は **suite の平均得点とは比較できない**
  （候補集合が違う。`--exclude-ply` や `--stop-unscored` を使えばなおさら）
- 反則後ブロックは親の手目と別の決定点として扱う。ここを混ぜると
  「反則後にしか存在しない候補」が反則前のシナリオ名で出てきて採点が
  ずれる（2026-08-11 に PR #3 で実際に起きた。`cargo test 採点表` が関門）

## quest_20260731 の移行について

`docs/quest_20260731.md` の判定は `scripts/quest_review/md_to_eval.py`
（語彙→点数の一括変換、2026-08-07）で `quest_20260731.eval.md` へ移行済み。
**以後はこの eval ファイルが一次資料**で、md は経緯の記録として残る。
語彙からの初期点数は目安なので、実感と違う手は個別に上書きしてよい
（同期はいつでも再実行できる）。
