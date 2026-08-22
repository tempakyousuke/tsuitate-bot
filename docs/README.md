# docs/

実験記録・設計メモ・調査の置き場。**現行の運用手順・ノブの採否・凍結版の成績はリポジトリ直下の `CLAUDE.md`**（`AGENTS.md` は同一内容）。

ここにある日付つき計画は、書いた時点のスナップショットとして残す。後から本文を「今の結論」へ書き換えると経緯が消えるので、状態は下表と各ファイル冒頭の注記で示す。

| 文書 | 種類 | 状態（2026-08-18 時点） |
| --- | --- | --- |
| `c7-continuous-filter.md` | 設計 | P1+P2 は v7、P3 は v8 として実装・凍結済み |
| `c8-direct-synthesis.md` | 設計 | `synth_particle` の MVP はあるが主経路には未統合。反則マス記憶系は不発 |
| `nn-value-phase1.md` | 記録 | フェーズ2で推論統合済み。v10 凍結 |
| `nn-stage2-belief-net.md` | 記録 | 較正は粒子より当たるが勝率中立。既定0のノブとして保留 |
| `yaneuraou-lessons.md` | 調査 | 2026-07-26。V3（紐）はその後採用、思考予算は 2000ms で飽和。行番号は当時のもの |
| `improvement-plan-2026-07-25.md` | 計画 | v10 凍結直後のスナップショット。v11 はその後凍結済み |
| `improvement-plan-2026-07-26-yaneuraou.md` | 計画 | 上記調査の実行計画。項目1（思考予算）は実測で「ノブではない」と確定 |
| `improvement-plan-2026-08-02-quest31.md` | 計画 | F1（幻の詰み）は `mate_gate_q0` として採用。F2（未観測駒への捕獲賭け、m021）は未解決。m015 の 2二歩打（F2 とは別工事の反則の情報価値）は `drop_probe_w` として採用 |
| `quest_20260731.md` | 経緯 | 採点の一次資料は `evals/quest_20260731.eval.md`。本文は語彙判定の記録 |
| `improvement-plan-2026-08-22-probe-planning.md` | 計画 | ターン内の反則プローブ計画。プロトタイプ実装済み（既定0）・計測未実施 |
| `tsuitate-viewer-webhook-bot.md` | 運用 | `webhook_bot` の現行手順。既定戦略は凍結版 `estimator_v10` |
