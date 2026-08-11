#!/usr/bin/env python3
"""採点済み候補から学んだ着手事前値で rank_dump を再順位付けする診断。

同じ USI の他局面での平均点（現在局面は leave-one-block-out）と、現行エンジンの
score を `prior + alpha * engine_score` で合成する。未知手の事前値は eval と同じ
4点。これは本番方策ではなく、次に実装すべき一般特徴と未採点候補を安価に発見する
ためのオフライン診断である。

usage:
  RANK_DUMP_SCORES=1 target/release/rank_dump ... > /tmp/rank.txt
  python3 scripts/quest_review/rerank_eval.py \
    evals/quest_20260731.eval.md /tmp/rank.txt --stop-unscored 10

標準出力は append_unscored.py が読める TRIAL TSV。診断は stderr に出す。
"""

import argparse
import collections
import pathlib
import re
import sys

SECTION = re.compile(r"^## (\d+)手目")
EVAL_MOVE = re.compile(r"^\S+?\(([^)]+)\)\s+(\d+|\?)")
RANK_MOVE = re.compile(r"^.+\(([^)]+)\) score=([-+.\d]+)\s")


def read_eval(path: pathlib.Path):
    by_ply = collections.defaultdict(dict)
    all_scores = collections.defaultdict(list)
    ply = None
    for line in path.read_text(encoding="utf-8").splitlines():
        section = SECTION.match(line)
        if section:
            ply = int(section.group(1))
            continue
        move = EVAL_MOVE.match(line)
        if ply is None or not move or move.group(2) == "?":
            continue
        score = int(move.group(2))
        by_ply[ply][move.group(1)] = score
        all_scores[move.group(1)].append((ply, score))
    return by_ply, all_scores


def read_ranking(path: pathlib.Path):
    rankings = collections.defaultdict(list)
    ply = None
    for line in path.read_text(encoding="utf-8").splitlines():
        section = SECTION.match(line)
        if section:
            ply = int(section.group(1))
            continue
        move = RANK_MOVE.match(line)
        if ply is not None and move:
            rankings[ply].append((move.group(1), float(move.group(2))))
    return rankings


def prior_for(usi, ply, all_scores, unknown_prior):
    values = [score for source_ply, score in all_scores.get(usi, ()) if source_ply != ply]
    return sum(values) / len(values) if values else unknown_prior


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_path", type=pathlib.Path)
    parser.add_argument("rank_dump", type=pathlib.Path)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--unknown-prior", type=float, default=4.0)
    parser.add_argument("--stop-unscored", type=int, default=0)
    args = parser.parse_args()

    by_ply, all_scores = read_eval(args.eval_path)
    rankings = read_ranking(args.rank_dump)
    score_sum = 0.0
    selected = 0
    unscored = 0
    for ply in sorted(rankings):
        if ply == 148 or not rankings[ply]:
            continue
        usi, _ = max(
            rankings[ply],
            key=lambda item: prior_for(item[0], ply, all_scores, args.unknown_prior)
            + args.alpha * item[1],
        )
        score = by_ply.get(ply, {}).get(usi)
        if score is None:
            unscored += 1
            score = args.unknown_prior
        score_sum += score
        selected += 1
        print(f"TRIAL\tquest31-m{ply:03d}\t0\t{usi}\t0")
        if args.stop_unscored and unscored >= args.stop_unscored:
            break

    mean = score_sum / max(selected, 1)
    print(
        f"rerank: {selected}局面 平均{mean:.3f}/10 未採点{unscored}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
