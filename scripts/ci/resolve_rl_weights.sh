#!/usr/bin/env bash
# 方策ネットの戦略名に書いた GitHub Release の重みを取ってきて、手元のパスへ置き換える
# （arena.yml が候補・基準の両方に使う。docs/rl-deepnash-design.md の「重みの置き場所」）。
#
#   scripts/ci/resolve_rl_weights.sh 'rl_policy:release:<タグ>/<ファイル名>'
#     → Release <タグ> の <ファイル名> を rl-weights/<タグ>/ へ落として
#       'rl_policy:<絶対パス>' を標準出力へ出す（rl_policy_greedy も同じ）
#   それ以外の戦略名（estimator_v14 など）はそのまま出す
#
# 実験中の重みはリポジトリにコミットせず Release に置く（履歴が重みで膨らまないように）。
# 採用した版だけ models/ にコミットして `rl_policy:models/<ファイル>` で指す。
# 使った重みは記録上の戦略名（`rl_policy@<sha256 先頭12桁>`）で特定できる。
#
# --self-test: ネットワークを使わない分解規則の検査
set -euo pipefail

parse() {
  # 出力: prefix tag asset（分解できなければ何も出さない）
  local name="$1"
  case "$name" in
    rl_policy:release:* | rl_policy_greedy:release:*) ;;
    *) return 0 ;;
  esac
  local prefix="${name%%:release:*}"
  local spec="${name#*:release:}"
  local tag="${spec%/*}"
  local asset="${spec##*/}"
  if [ "$tag" = "$spec" ] || [ -z "$tag" ] || [ -z "$asset" ]; then
    echo "::error::重みの指定は release:<タグ>/<ファイル名> の形で書いてください: $name" >&2
    exit 1
  fi
  echo "$prefix $tag $asset"
}

if [ "${1:-}" = "--self-test" ]; then
  check() {
    local got
    got=$(parse "$1")
    if [ "$got" != "$2" ]; then
      echo "self-test 失敗: '$1' → '$got'（期待 '$2'）" >&2
      exit 1
    fi
  }
  check 'estimator_v14' ''
  check 'rl_policy:/abs/policy.bin' ''
  check 'rl_policy:release:rlw-bc-m2a/policy.bin' 'rl_policy rlw-bc-m2a policy.bin'
  check 'rl_policy_greedy:release:rlw/2026/a.bin' 'rl_policy_greedy rlw/2026 a.bin'
  if (parse 'rl_policy:release:policy.bin' 2>/dev/null); then
    echo "self-test 失敗: タグの無い指定を通した" >&2
    exit 1
  fi
  echo "resolve_rl_weights self-test OK"
  exit 0
fi

name="$1"
parsed=$(parse "$name")
if [ -z "$parsed" ]; then
  echo "$name"
  exit 0
fi
read -r prefix tag asset <<< "$parsed"
dir="rl-weights/$tag"
mkdir -p "$dir"
if [ ! -f "$dir/$asset" ]; then
  gh release download "$tag" --repo "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY が要る}" \
    --pattern "$asset" --dir "$dir" >&2
fi
if [ ! -s "$dir/$asset" ]; then
  echo "::error::Release $tag に $asset がありません" >&2
  exit 1
fi
echo "重み: $name → $dir/$asset（sha256 $(sha256sum "$dir/$asset" | cut -c1-12)）" >&2
echo "${prefix}:$PWD/$dir/$asset"
