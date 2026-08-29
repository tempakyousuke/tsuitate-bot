# 王手駒仮説の希釈（issue #36）P0 の設計と実装

2026-08-29。**runtime には何も入っていない**（`CheckSolver` に足したのは診断専用の
フックだけで、設置しなければ既定挙動と bit-exact）。

## 発端

#28（詰み経済）・#31（王手中の反則経済）・#34（被王手の前の準備）は3件とも
「機構は動く／反則は減る／勝率に変換されない」で閉じ、3件の docs の「次にやるなら」が
揃って**信念側**を指した。#31 P0-3 が出した破滅の署名は「知らない」ではなく**希釈**:

| | vs v14 | vs v13 |
| --- | ---: | ---: |
| 真の王手駒の仮説の重みシェア（反則あり / 反則0） | **0.037 / 0.078** | **0.035 / 0.084** |
| 仮説の正規化エントロピー（反則あり / 反則0） | 0.993 / 0.975 | 0.994 / 0.981 |
| ソルバー方策が8反則以上を積む手番のうち真の王手駒が仮説にある | 4/4 | 2/2 |

8反則以上を積んだ6手番の**すべてで真の王手駒は仮説集合に載っていた**のに、一様に
近い 30 本前後の仮説に埋もれて正しい捕獲の `p_legal` が薄まる（kakutori の構図）。

**ただし 0.035 はソルバー単体（粒子投票なし）の事前**で、runtime では

```
ソルバー q → 粒子投票後の q → prior_legal との積 → blend_p_legal → cap/min → score → rank
```

と6段を通って初めて選択に届く。**仮説シェアは選択への伝達量ではない**。

## 過去の失敗4件（すべて「較正されていない乗算」）

| 施策 | 結果 | 教訓 |
| --- | --- | --- |
| 駒種頻度事前 `freq^0.35`（2026-07-21） | 反則 270→277〜278（全強度で悪化） | 単純な事前の掛け合わせは筋が悪い |
| 信念ネット占有事前 `CHECK_BELIEF_OCC_W=1`（2026-08-19） | vs v14 44.2% → 既定 0 | 投票の前に乗算すると床×投票が占有1.0の幻仮説に負ける |
| walk-in 割引（2026-08-20） | foul02 42→47 | 割引は消えた質量の行き先を選べない |
| 打ち説明重み（標本化側、2026-08-20） | vs v14 −7.2 / −9.1pt | 説明分布を動かすと粒子が汚れる |

だから本 issue は**先に測る**（P0-1 の分解 → P0-2 のオラクル → P0-2b の継続上限）。
学習（P0-3）と方策 sim（P0-4）はその後。

## 実装（このブランチ）

### 診断フック（`src/check.rs`）

- `HypothesisDiag { factors, deduce_last_move }` を `scoped_hypothesis_diag` で
  **スレッドローカルに設置**する。`CheckSolver::new` は**列挙の直後**に見る
  （捕獲ブースト・粒子投票・反則減衰より前）。設置しなければ分岐の中身を1度も
  実行しないので、既定挙動は完全に不変（`cargo test 診断フックは既定では何もしない`）
- `factors` の乗算で総重みが 0 以下になったら**元へ戻す**（coverage failure の
  ときに `×0` で仮説を全滅させない。`cargo test 倍率ゼロで全滅するときは元へ戻す`）
- `deduce_last_move`（H1 ①）は「(a) 直前の相手手で q へ着地した駒」でも
  「(b) 元マスが空いて線が開いた静止飛び駒」でもない仮説だけを落とす。
  捕獲つきの王手は着地点が観測で分かるので既存の
  `prune_infeasible_discovered_checks` の領分（何もしない）。
  落とそうとした仮説は `take_deduce_dropped()` で**全滅 fallback の前に**取れる
  （fallback 後だけ見ると健全性違反を隠せる）

### 共有定義（`src/check_belief.rs`）

- `Belief`（`current` / `oracle@k*` / `oracle_misdirected@k*` / `deduce_last_move`）と
  `ArmSpec`（`[belief|policy][@shadow|@real]`）。**タグの規約を `bin/check_policy` と
  `bin/check_continue` で共通**にしてあるので、同じ arm 名が必ず同じ配管を指す
- `run_arm` … p-only（gain・`removal_term` は初回ランキングの値に固定）と
  実再決定（初回ランキングもオラクルの下で作り直し、反則ごとに `Estimator::update`）の
  2経路。**両方とも1か所**にある
- `decision_points` … 母集団の取り出し。**bot の全王手中手番**で、反則0も
  **終端手番**（反則だけ積んで受理手なしで終局した手番）も含む
  （`for_each_decision_full` は受理手を単位に回すので終端を返さない。#34 の
  `check_prep::decision_snapshots` を再利用）。復元できない手番は `Attrition` として
  本数を出す（**改善対象の最悪ケースが系統的に欠測すると門が甘くなる**）
- `focus` … 「真の王手駒を玉以外で取る手」と「誤仮説マスへの捕獲の最大値」の対

### P0-1（`bin/check_belief_probe`）

`ShadowUpdater::stages_after`（`p_after` の各段を残した版。`p_after` はこれの
`final_p` そのものなので分解と本番が食い違わない）で

- ソルバー単体 q / 粒子投票後 q（真の王手駒のシェアと正規化エントロピー）
- 注目2手の `prior_legal` → `solver_p` → 積 → 粒子合法率 → `blend_p_legal` → cap →
  最終 p → score → 順位
- 層: `particles_vote_check` の有無 × 厳密/taint × 反則あり/なし × 初反則の手種 ×
  王手駒の種別（打ち／盤上／捕獲つき）× 終端手番。**両王手は別層**
- 投票者（王手を説明する粒子）の重みシェアと、そのうち真の王手駒を当てている割合
- **coverage failure**（真の王手駒を玉以外で取る候補が無い手番）の本数

を JSONL で出す。**中止の門は置かない**（記述のみ。因果はオラクル arm でしか言えない）。

### P0-2（`bin/check_policy --belief`）

issue の arm 名との対応:

| issue | 実装の arm 名 |
| --- | --- |
| `oracle_p_only@k` | `oracle@k{2,4,8,inf}@shadow` |
| **`oracle_full_score@real`（主 arm）** | `oracle@kinf@real` |
| `oracle_misdirected@k` | `oracle_misdirected@k{4,inf}@shadow` |
| 恒等対照 | `oracle@k1@shadow`（`current@shadow` と **bit-exact**） |
| `deduce_last_move` | `deduce_last_move@shadow` |

`report` に3つの関門を足した（**どれも判定以前**）:

- 恒等対照の p の最大差が 0 でなければ `die`
- `deduce_last_move` が真仮説を落としたら `die`（fallback 前で数える）
- オラクルが介入できた行数（真の王手駒が仮説にある単王手）と両王手の本数を出す

`ROW_SCHEMA` は **2**（schema 1 = 恒等対照と演繹の列が無い時期の記録は集計から弾く。
**欠けた列を「問題なし」と読むと両方の関門が素通りする**ので、版の拒否と
`REQUIRED_ROW_KEYS` の存在検査の両方で止める）。

### P0-2b（`bin/check_continue --policy oracle@kinf@real`）

- 強制列は P0-2 と**同じ配管**（`check_belief::run_arm`）
- **対照は `current@real`**（`report --baseline current@real`）。shadow の `current` と
  比べると「オラクル」と「指し直したこと」の効果が混ざる。実再決定 arm を混ぜたら
  `current@real` を**自動で足す**ので取り忘れない
- 思考予算は **2000ms**（ランキング段と継続段の両方に効く。700ms は別の treatment）

### CI（`.github/workflows/check-belief.yml`）

**通常のコード push では走らない**。`gh workflow run check-belief.yml -f arena_run_id=...`
か `.github/ci/check-belief.request.json` を置いて push（削除の push は全ジョブがスキップ）。
相手ごとに別ジョブ・元対局単位のシャード・**欠けたら aggregate が失敗**。

元 Arena 実行の実験条件の検査は `scripts/ci/verify_arena_provenance.sh`（`check-prep.yml`
から切り出して**共通化**した。ワークフローごとに書くと片方だけ緩くなる）。

## 契約（先に固定した）

- 母集団は **bot の全王手中手番**（反則0も終端手番も含む）。落ちた本数は attrition
- **単王手が主 estimand**、両王手は別層で記述のみ（2仮説をともに ×k しても重みが
  半々になるだけで、一方しか解消しない非合法手に ≈0.5 が付く）
- 判定量の向き: 反則の削減は `R = current − treatment`、勝率は `Δ = treatment − current`
- 合算は **opponent-balanced**（`(Δv13 + Δv14) / 2`）。veto は `Δv13 > 0 && Δv14 > 0`
- 継続対局は**常に 2 replicate**、対象手番は estimand ごとに最初の1手番
- 発見セット = run 32697854659 / 33179939954、学習セット = 新規のアリーナ、
  検証セット = **未使用の `match_seed`**（`expect_match_seed` を必ず渡す）

## 中止条件（issue のまま）

- P0-2: `oracle@kinf@real` が最終 p / rank / 選択を動かさない、または即時反則・
  破滅率を改善しない（H0 = 希釈は破滅の結果であって原因ではない）
- P0-2b: オラクルの継続ペア差が +0.04 未満、または安全性 veto
- 恒等対照が `current` と一致しない、`deduce_last_move` が真仮説を落とす（配管が壊れている）

## 実測

（未実施。P0-1 を発見セットで回したらここへ表を足す）
