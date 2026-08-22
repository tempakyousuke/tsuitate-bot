#!/usr/bin/env python3
"""採点済み候補から学んだ着手事前値で rank_dump を再順位付けする診断。

同じ USI の他の決定点での平均点（現在の手目は leave-one-ply-out）と、現行エンジンの
score を `prior + alpha * engine_score` で合成する。未知手の事前値は eval と同じ
4点（`scenario_core::UNSCORED_DEFAULT`）。これは本番方策ではなく、次に実装すべき
一般特徴と未採点候補を安価に発見するためのオフライン診断である。

usage:
  RANK_DUMP_SCORES=1 target/release/rank_dump <kif> <開始> <終了> > /tmp/rank.txt
  mkdir -p /tmp/rerank && python3 scripts/quest_review/rerank_eval.py \\
    evals/quest_20260731.eval.md /tmp/rank.txt --stop-unscored 10 \\
    > /tmp/rerank/result.txt
  python3 scripts/quest_review/append_unscored.py \\
    evals/quest_20260731.eval.md usi-prior-rerank /tmp/rerank --min 1

標準出力は `scenario suite --tsv` と同じ TRIAL TSV。append_unscored.py は
ディレクトリ配下の `**/result.txt` を読むので、**ファイルへリダイレクトする**こと。
診断は stderr に出す。

反則後サブブロック（`### N手目（…の反則後）`）は親の手目とは別の決定点として扱う。
rank_dump は既定でサブブロックも出力するので、ここを混ぜると「反則後にしか存在
しない候補」が反則前のシナリオ名で出てきて、点数も未採点カウントも壊れる
（PR #3 で実際に発生し、m026 に先手の手が混入した）。

限界: 事前値は「同じ USI の他の決定点の平均点」でしかなく、局面は考慮しない。
高得点の USI がどの手目でも上位に来るので、発見される未収載候補は少数の USI に
集中する（実測では 4g4f が6手目分・2c2f が5手目分）。網羅的な探索の道具ではない。
"""

import argparse
import collections
import pathlib
import re
import sys

from foul_blocks import DEFAULT_PREFIX, prefix_for_eval, scenario_for

ROOT = pathlib.Path(__file__).resolve().parents[2]

SECTION = re.compile(r"^## (\d+)手目")
SUBSECTION = re.compile(r"^### (\d+)手目（(.+?)の反則後）")
EVAL_MOVE = re.compile(r"^\S+?\(([^)]+)\)\s+(\d+|\?)")
RANK_MOVE = re.compile(r"^.+\(([^)]+)\) score=([-+.\d]+)\s")


def block_of(line):
    """見出し行ならブロックキーを返す（見出しでなければ None）"""
    m = SUBSECTION.match(line)
    if m:
        return (int(m.group(1)), m.group(2))
    m = SECTION.match(line)
    if m:
        return (int(m.group(1)), None)
    return None


def read_eval(path: pathlib.Path):
    """(ブロックキー -> {USI: 点}, USI -> [(手目, 点)])"""
    by_block = collections.defaultdict(dict)
    all_scores = collections.defaultdict(list)
    key = None
    for line in path.read_text(encoding="utf-8").splitlines():
        block = block_of(line)
        if block:
            key = block
            continue
        move = EVAL_MOVE.match(line)
        if key is None or not move or move.group(2) == "?":
            continue
        score = int(move.group(2))
        by_block[key][move.group(1)] = score
        all_scores[move.group(1)].append((key[0], score))
    return by_block, all_scores


def read_ranking(path: pathlib.Path):
    """ブロックキー -> [(USI, engine score)]"""
    rankings = collections.defaultdict(list)
    key = None
    for line in path.read_text(encoding="utf-8").splitlines():
        block = block_of(line)
        if block:
            key = block
            continue
        move = RANK_MOVE.match(line)
        if key is not None and move:
            rankings[key].append((move.group(1), float(move.group(2))))
    return rankings


def prior_for(usi, ply, all_scores, unknown_prior):
    """同じ USI の他の手目での平均点（同じ手目は反則後ブロックごと除外する）"""
    values = [score for source_ply, score in all_scores.get(usi, ()) if source_ply != ply]
    return sum(values) / len(values) if values else unknown_prior


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_path", type=pathlib.Path)
    parser.add_argument("rank_dump", type=pathlib.Path)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--unknown-prior", type=float, default=4.0)
    parser.add_argument("--stop-unscored", type=int, default=0)
    parser.add_argument(
        "--exclude-ply", default="",
        help="除外する手目のカンマ区切り（例 148。平均が変わるので stderr にも出る）")
    parser.add_argument("--scenarios", type=pathlib.Path, default=ROOT / "scenarios")
    parser.add_argument(
        "--prefix", default=None,
        help="シナリオ名プレフィックス（既定は eval の stem から foul_blocks.PREFIX_BY_EVAL で引く）")
    args = parser.parse_args()
    if args.prefix is None:
        args.prefix = prefix_for_eval(args.eval_path) or DEFAULT_PREFIX

    excluded = {int(x) for x in args.exclude_ply.split(",") if x.strip()}
    by_block, all_scores = read_eval(args.eval_path)
    rankings = read_ranking(args.rank_dump)
    score_sum = 0.0
    selected = 0
    unscored = 0
    no_scenario = []
    for key in sorted(rankings, key=lambda k: (k[0], k[1] or "")):
        if key[0] in excluded or not rankings[key]:
            continue
        name = scenario_for(key, args.prefix)
        if name is None or not (args.scenarios / f"{name}.kif").exists():
            no_scenario.append(key)
            continue
        usi, _ = max(
            rankings[key],
            key=lambda item: prior_for(item[0], key[0], all_scores, args.unknown_prior)
            + args.alpha * item[1],
        )
        score = by_block.get(key, {}).get(usi)
        if score is None:
            unscored += 1
            score = args.unknown_prior
        score_sum += score
        selected += 1
        print(f"TRIAL\t{name}\t0\t{usi}\t0")
        if args.stop_unscored and unscored >= args.stop_unscored:
            print(
                f"注意: 未採点 {unscored} 件で打ち切ったので平均は先頭 {selected} 決定点だけの値",
                file=sys.stderr,
            )
            break

    mean = score_sum / max(selected, 1)
    print(f"rerank: {selected}決定点 平均{mean:.3f}/10 未採点{unscored}", file=sys.stderr)
    if excluded:
        print(f"除外した手目: {sorted(excluded)}（suite の平均得点とは比較できない）",
              file=sys.stderr)
    if no_scenario:
        print(f"シナリオ未作成で飛ばした決定点 {len(no_scenario)} 件: "
              f"{[f'{p}手目' + (f'（{s}の反則後）' if s else '') for p, s in no_scenario]}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
