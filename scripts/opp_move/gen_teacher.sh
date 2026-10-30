#!/bin/bash
# opp_move NN の教師の対局を集める（docs/nn-direction.md の「opp_move NN の学び直し」）。
# arena の記録は候補 A の視点で、export_opp_move_data が書き出すのは基準 B の手。
# 候補4体 × 基準（方策ネット13系統）× 40局 = 系統ごとに160局、estimator_v14 も同じく160局。
#
#   cargo build --release --bin arena && scripts/opp_move/gen_teacher.sh <出力ディレクトリ>
#
# 方策ネットどうしは4並列で約1時間、estimator_v14 は1手約1秒なので4並列で約30分（2026-10-30、4コア）。
# estimator は思考時間で読みが変わるので、estimator の対局中は ほかの重い処理を回さない。
set -eu
OUT=${1:?出力ディレクトリ}
cd "$(dirname "$0")/../.."
POOL="rl_mmd_v4 rl_mmd_v3 rl_mmdv3branch_v1 rl_deduce_v10 rl_deduce_v7 rl_selfplay_v4 rl_v30 rl_v25 rl_v22 rl_speed_v3 rl_speed_v4 rl_guard_v1 rl_nyugyoku_v2"
CANDS="rl_mmd_v4 rl_deduce_v10 rl_v25 rl_guard_v1"
for a in $CANDS; do
  mkdir -p "$OUT/pol_$a"
  ARENA_RECORD_DIR="$OUT/pol_$a" target/release/arena 40 "$a" $POOL > "$OUT/pol_$a.log" 2>&1 &
done
wait
for a in $CANDS; do
  mkdir -p "$OUT/est_$a"
  ARENA_RECORD_DIR="$OUT/est_$a" target/release/arena 40 "$a" estimator_v14 > "$OUT/est_$a.log" 2>&1 &
done
wait
