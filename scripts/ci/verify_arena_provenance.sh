#!/usr/bin/env bash
# **元 Arena 実行の実験条件を機械的に検査する**（PR #35 レビュー [P1]）。
#
# 記録（`arena-records-*`）の `match` / `end` にあるのは戦略名と相手名だけで、
# 解析側の `source_fingerprint` は解析バイナリの版でしかない。だから同じ run の
# `arena-result-*`（`arena-summary.json` = 実効設定の指紋・局数・時計・予算・
# `match_seed_base`、`arena-games.jsonl` = commit）と `arena-combined`（全シャード
# 合算の局数）も取り、**シャード集合が 0..n-1 で完備・実効設定が全シャード一致・
# 局数と記録本数が一致・合算局数と一致・run が success 終了**を確かめてから解析する。
#
# `check-prep.yml`（issue #34）と `check-belief.yml`（issue #36）が**同じ関門**を
# 通るように、ここに一本化してある（ワークフローごとに書くと片方だけ緩くなる）。
#
# 環境変数:
#   OPPONENT            記録の相手（artifact のラベル。空 = records/ を使う場合は呼ばない）
#   RUN_ID              元 Arena 実行の run ID
#   GH_TOKEN            API 用トークン
#   GITHUB_REPOSITORY   owner/repo
#   RECORDS_DIR         記録の展開先（既定 records-in）
#   SUMMARIES_DIR       arena-result-* の展開先（既定 summaries-in）
#   COMBINED_DIR        arena-combined の展開先（既定 combined-in）
#   OUT                 検証済みの実験条件の書き出し先（既定 provenance.env）
#   EXPECT_CANDIDATE / EXPECT_MATCH_SEED / EXPECT_COMMIT / EXPECT_GAMES / EXPECT_SHARDS
#                       期待値（空なら照合しない）
#   EXPECT_THINK_BUDGET_MS
#                       **この phase が実際に使う思考予算**。元 Arena の
#                       `think_budget_ms_a` と一致しなければ落とす。
#                       「記録時に実際に見えていたランキング」でない測定を
#                       採否に使わせないための関門（PR #37 レビュー [P1]）
#   ALLOW_LEGACY_THINK_BUDGET
#                       `1` のとき、**旧形式の `think_budget_ms_a: null`** を
#                       「その build の既定値」として解決する（下記）。既定は
#                       落とす（null を黙って通すと関門が意味を失う）
set -euo pipefail
shopt -s nullglob globstar

RECORDS_DIR=${RECORDS_DIR:-records-in}
SUMMARIES_DIR=${SUMMARIES_DIR:-summaries-in}
COMBINED_DIR=${COMBINED_DIR:-combined-in}
OUT=${OUT:-provenance.env}
EXPECT_CANDIDATE=${EXPECT_CANDIDATE:-}
EXPECT_MATCH_SEED=${EXPECT_MATCH_SEED:-}
EXPECT_COMMIT=${EXPECT_COMMIT:-}
EXPECT_GAMES=${EXPECT_GAMES:-}
EXPECT_SHARDS=${EXPECT_SHARDS:-}
EXPECT_THINK_BUDGET_MS=${EXPECT_THINK_BUDGET_MS:-}
ALLOW_LEGACY_THINK_BUDGET=${ALLOW_LEGACY_THINK_BUDGET:-}
# `think_budget_ms_a` が **null になりうる** のは commit 476483d より前の
# 記録だけで、そこでの定義は
#   env::var("TSUITATE_CAND_THINK_BUDGET_MS")
#     .or_else(|_| env::var("TSUITATE_THINK_BUDGET_MS")).ok()
# = 「env の生値、どちらも未設定なら null」。**null は「上書きなし = その
# build の既定値」という意味であって「不明」ではない**（476483d 以降は
# `budget_of(...)` で実効値を必ず入れるので null にならない）。
# 既定値は 886ef75 でも現在でも `strategy::DEFAULT_THINK_BUDGET_MS = 2000`
LEGACY_DEFAULT_THINK_BUDGET_MS=2000
: > "$OUT"

# --- 0. 元 run そのものの状態（commit と結末）を API から取る ---
# **artifact だけでは commit が分からない**（`arena-games.jsonl` が無い run もある）。
# 部分的に失敗した run の残骸で解析しないためにも `conclusion == success` を要求する
meta=$(curl -sSf -H "Authorization: Bearer $GH_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/${GITHUB_REPOSITORY}/actions/runs/${RUN_ID}")
run_sha=$(echo "$meta" | jq -r '.head_sha')
run_concl=$(echo "$meta" | jq -r '.conclusion')
run_name=$(echo "$meta" | jq -r '.name')
if [ "$run_name" != "Arena" ]; then
  echo "::error::run $RUN_ID は Arena ではありません（$run_name）"
  exit 1
fi
if [ "$run_concl" != "success" ]; then
  echo "::error::元 Arena 実行が success で終わっていません（$run_concl）。部分的に失敗した run の記録では解析しない"
  exit 1
fi
printf 'arena_run_id=%s\narena_head_sha=%s\n' "$RUN_ID" "$run_sha" >> "$OUT"
if [ -n "$EXPECT_COMMIT" ] && [ "$run_sha" != "$EXPECT_COMMIT" ]; then
  echo "::error::元 Arena 実行の commit が期待と違います（期待 $EXPECT_COMMIT / 実際 $run_sha）"
  exit 1
fi

# --- 1. シャード集合が 0..n-1 で完備か（途中が欠けた run を弾く）---
parts=("$SUMMARIES_DIR"/**/arena-summary.json)
if [ ${#parts[@]} -eq 0 ]; then
  echo "::error::arena-result-$OPPONENT-* が取得できません（実験条件を検査できないので解析しない）"
  exit 1
fi
# artifact 名のサブディレクトリから shard 番号を取る。download-artifact は
# 一致が1件だけだとサブディレクトリを作らないので、その場合は 0 とみなす
# （複数が番号なしになれば重複して完備検査で落ちる）
idx=$(for f in "${parts[@]}"; do
        i=$(echo "$f" | sed -n 's/.*-s\([0-9]\+\)\/.*/\1/p')
        [ -n "$i" ] || i=0
        echo "$i"
      done | sort -n)
n=$(echo "$idx" | wc -l)
want=$(seq 0 $((n - 1)))
if [ "$idx" != "$want" ]; then
  echo "::error::シャードが 0..$((n-1)) で完備ではありません（取得: $(echo $idx)）。**元対局が欠けると分母が狂う**ので解析しない"
  exit 1
fi
cat "${parts[@]}" > shard-summaries.jsonl

# --- 2. 実効設定・相手・時計・予算が全シャードで一致し、相手が一致するか ---
for k in candidate baseline cand_config baseline_behavior think_budget_ms_a \
         fischer_initial_ms fischer_increment_ms cand_knobs; do
  u=$(jq -r --arg k "$k" '.[$k] | tostring' shard-summaries.jsonl | sort -u)
  if [ "$(echo "$u" | wc -l)" -ne 1 ]; then
    echo "::error::シャード間で $k が食い違います: $(echo $u)"
    exit 1
  fi
  printf '%s=%s\n' "$k" "$u" >> "$OUT"
done
# **この phase の思考予算が元 Arena と一致するか**（PR #37 レビュー [P1]）。
# 粒子数は予算に比例するので、違う予算で引き直したランキングは
# 「その決定点で実際に見えていたランキング」ではない
if [ -n "$EXPECT_THINK_BUDGET_MS" ]; then
  budget=$(jq -r '.think_budget_ms_a | tostring' shard-summaries.jsonl | sort -u)
  budget_src=recorded
  if [ "$budget" = "null" ]; then
    # 旧形式（476483d より前）。null は「不明」ではなく「上書きなし = 既定値」
    if [ "$ALLOW_LEGACY_THINK_BUDGET" != "1" ]; then
      echo "::error::元 Arena は think_budget_ms_a を実効値で記録していません（commit 476483d より前の旧形式で、上書きが無いと null になる）。既定値 ${LEGACY_DEFAULT_THINK_BUDGET_MS}ms として解決してよいなら ALLOW_LEGACY_THINK_BUDGET=1 を渡してください"
      exit 1
    fi
    budget=$LEGACY_DEFAULT_THINK_BUDGET_MS
    budget_src=legacy_default
  fi
  if [ "$budget" != "$EXPECT_THINK_BUDGET_MS" ]; then
    echo "::error::この phase の思考予算 ${EXPECT_THINK_BUDGET_MS}ms が元 Arena の ${budget}ms（${budget_src}）と違います。粒子数が変わるので「記録時に実際に見えていたランキング」になりません"
    exit 1
  fi
  # 旧形式を既定値として解決したときは、**記録の思考時間で裏を取る**。
  # 予算より平均が大きいことは起こりえないので、「実際はもっと大きい予算
  # だったのに小さい予算を主張した」側は必ず捕まる（逆向き＝実際より大きい
  # 予算を主張した場合は捕まらない。**非対称なので単独の証拠にはしない**）
  if [ "$budget_src" = legacy_default ]; then
    # **`think_avg_ms_a` は実測値なのでシャードごとに違う**（設定値と違って
    # 「全シャードで一致」を要求してはいけない）。全シャードが主張した予算
    # 未満であることだけを見るので、**最大値**で判定する
    avgs=$(jq -r '.think_avg_ms_a | tostring' shard-summaries.jsonl)
    if [ -z "$avgs" ] || echo "$avgs" | grep -qx null; then
      echo "::error::旧形式の予算を裏づける think_avg_ms_a が記録にありません"
      exit 1
    fi
    avg_max=$(echo "$avgs" | sort -g | tail -1)
    if ! awk -v a="$avg_max" -v b="$EXPECT_THINK_BUDGET_MS" 'BEGIN{exit !(a+0 < b+0)}'; then
      echo "::error::記録の思考平均の最大 ${avg_max}ms が主張した予算 ${EXPECT_THINK_BUDGET_MS}ms 以上です。元 Arena は既定値では走っていません"
      exit 1
    fi
    printf 'think_avg_ms_a_max=%s\n' "$avg_max" >> "$OUT"
  fi
  printf 'phase_think_budget_ms=%s\n' "$EXPECT_THINK_BUDGET_MS" >> "$OUT"
  printf 'think_budget_ms_a_source=%s\n' "$budget_src" >> "$OUT"
fi
base=$(jq -r '.baseline' shard-summaries.jsonl | sort -u)
if [ "$base" != "$OPPONENT" ]; then
  echo "::error::記録の相手が一致しません（artifact=$OPPONENT / summary=$base）"
  exit 1
fi
cand=$(jq -r '.candidate' shard-summaries.jsonl | sort -u)
if [ -n "$EXPECT_CANDIDATE" ] && [ "$cand" != "$EXPECT_CANDIDATE" ]; then
  echo "::error::候補戦略が一致しません（期待 $EXPECT_CANDIDATE / 実際 $cand）"
  exit 1
fi

# --- 3. 局数と記録の本数が一致するか（部分的に失敗した run を弾く）---
games=$(jq -s 'map(.games) | add' shard-summaries.jsonl)
files=$(find "$RECORDS_DIR" -name '*.jsonl' | wc -l)
if [ "$games" -ne "$files" ]; then
  echo "::error::局数 $games と記録 $files 本が一致しません（記録が欠けています）"
  exit 1
fi
if [ -n "$EXPECT_GAMES" ] && [ "$games" -ne "$EXPECT_GAMES" ]; then
  echo "::error::局数が期待と違います（期待 $EXPECT_GAMES / 実際 $games）"
  exit 1
fi

# --- 4. aggregate が見た局数と一致するか（**末尾シャードの欠落を拾う唯一の経路**。
# シャード番号の連続性検査は末尾欠落を検出できない）---
if [ ! -f "$COMBINED_DIR/combined.json" ]; then
  echo "::error::arena-combined/combined.json がありません。末尾シャードの欠落を検査できないので解析しない"
  exit 1
fi
cg=$(jq -r --arg b "$OPPONENT" '.[] | select(.baseline == $b) | .games' "$COMBINED_DIR/combined.json")
if [ -z "$cg" ]; then
  echo "::error::arena-combined に相手 $OPPONENT の行がありません"
  exit 1
fi
if [ "$cg" != "$games" ]; then
  echo "::error::arena-combined の局数 $cg と取得した $games が違います（シャードの取り逃し）"
  exit 1
fi
if [ -n "$EXPECT_SHARDS" ] && [ "$n" -ne "$EXPECT_SHARDS" ]; then
  echo "::error::シャード数が期待と違います（期待 $EXPECT_SHARDS / 実際 $n）"
  exit 1
fi

# --- 5. commit（記録側）---
gj=("$SUMMARIES_DIR"/**/arena-games.jsonl)
if [ ${#gj[@]} -gt 0 ]; then
  commit=$(jq -r '.commit' "${gj[@]}" | sort -u)
  [ "$(echo "$commit" | wc -l)" -eq 1 ] || { echo "::error::記録の commit が単一でありません"; exit 1; }
  printf 'record_commit=%s\n' "$commit" >> "$OUT"
fi

# --- 6. match_seed の照合（**base seed で見る**）---------------------
# `arena-games.jsonl` / `arena-summary.json` の `match_seed` は「base + shard」を
# さらに**基準ごとに XOR した実効値**なので、シャードごとに違うし base とも
# 一致しない（PR #35 レビュー2巡目 [P1]）。照合は arena が明示的に残す
# `match_seed_base` で行う（式を複製しない）。
#
# **この段は jq で数値を触らない**（PR #35 レビュー4巡目 [P2]）: jq の数値演算は
# IEEE-754 倍精度なので、seed の型（`u64`）が 2^53 を超えると加算が壊れ、
# **正常な run を拒否する**。Python の json で整数のまま読んで任意精度で比べる。
# 自己検査を先に通す（この段の門はここにしか無いので、毎回走らせる）
python3 scripts/ci/verify_seed_provenance.py --self-test
seed_out=$(python3 scripts/ci/verify_seed_provenance.py \
  shard-summaries.jsonl "$EXPECT_MATCH_SEED" "$n")
case "$seed_out" in
  OK:*)
    rest=${seed_out#OK:}
    printf 'match_seed_base=%s\nmatch_seed_shards=%s\n' \
      "${rest%%:*}" "${rest#*:}" >> "$OUT"
    ;;
  ERR:*)
    echo "::error::${seed_out#ERR:}"
    exit 1
    ;;
  NOBASE)
    if [ -n "$EXPECT_MATCH_SEED" ]; then
      echo "::error::arena-summary.json に match_seed_base がありません（元 run の commit がこのフィールドより前か、match_seed 無しで回した run）。どちらにせよ検証セットの独立 seed を機械的に固定できません"
      exit 1
    fi
    echo "::warning::arena-summary.json に match_seed_base がありません（元 run が古いか match_seed 無し）。match_seed を記録から確かめられないので**検証セットには使えません**"
    ;;
  *)
    echo "::error::seed の検査が想定外の出力を返しました: $seed_out"
    exit 1
    ;;
esac
printf 'games=%s\nshards=%s\n' "$games" "$n" >> "$OUT"
echo "検証済みの実験条件:"; cat "$OUT"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "### 元 Arena 実行の検証（相手 $OPPONENT）"
    echo '```'
    cat "$OUT"
    echo '```'
  } >> "$GITHUB_STEP_SUMMARY"
fi
