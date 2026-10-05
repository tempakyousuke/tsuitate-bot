#!/usr/bin/env python3
"""凍結版のレーティング表（Elo 尺度）。

対局結果は ratings/games.jsonl（1行=1ペア: {"a","b","wa","wb","d"}）にため、
そこから毎回**全員を計算し直す**（既存の版のレートも固定しない）。
最新の計算結果は ratings/ratings.json に書き出す。

  python3 scripts/rating.py add <版> [局数]   新しい版を記録済みの全版と対局させ（既定100局/ペア）、
                                              全員を計算し直す
  python3 scripts/rating.py show [--matrix]   全員を計算し直して表を出す（--matrix で勝率表も）

計算は2通り:
- 一括推定: 全対局の Bradley-Terry 最尤。平均1500
- サイト式: サイトの elo.ts と同じく全員1500から K=32 で1局ずつ更新する。
  対局順をランダムに変えて SIM_RUNS 回やり直した平均と 5〜95% の幅

対局は `target/release/arena`（先に `cargo build --release --bin arena`）。
"""
import json
import math
import os
import random
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMES = os.path.join(ROOT, "ratings", "games.jsonl")
RATINGS = os.path.join(ROOT, "ratings", "ratings.json")
ARENA = os.path.join(ROOT, "target", "release", "arena")
# 全勝・全敗で発散しないよう、各ペアに 0.5勝ずつの弱い事前を足す
PRIOR = 0.5
# サイト式（tsuitate の src/lib/server/game/elo.ts）
SITE_K = 32
SITE_INITIAL = 1500
SIM_RUNS = 200


def load_games():
    if not os.path.exists(GAMES):
        return []
    return [json.loads(l) for l in open(GAMES) if l.strip()]


def pair_scores(games):
    """(x, y) -> [x の得点, 対局数]（引き分け0.5）"""
    s = {}
    for g in games:
        n = g["wa"] + g["wb"] + g["d"]
        for x, y, w in ((g["a"], g["b"], g["wa"]), (g["b"], g["a"], g["wb"])):
            v = s.setdefault((x, y), [0.0, 0])
            v[0] += w + 0.5 * g["d"]
            v[1] += n
    return s


def expected(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def fit_all(games):
    """全員の Bradley-Terry 最尤（MM 反復）。平均1500。"""
    s = pair_scores(games)
    names = sorted({x for x, _ in s})
    g = {x: 1.0 for x in names}
    for _ in range(5000):
        ng = {}
        for i in names:
            wins = sum(v[0] + PRIOR for (x, _), v in s.items() if x == i)
            den = sum((v[1] + 2 * PRIOR) / (g[i] + g[y]) for (x, y), v in s.items() if x == i)
            ng[i] = wins / den
        m = sum(math.log(v) for v in ng.values()) / len(ng)
        g = {k: v / math.exp(m) for k, v in ng.items()}
    return {x: 1500 + 400 * math.log10(g[x]) for x in names}


def simulate_site(games):
    """サイト式の逐次 Elo を対局順を変えて SIM_RUNS 回。名前 -> 最終レートの昇順リスト"""
    seq = []
    for g in games:
        seq += [(g["a"], g["b"], 1)] * g["wa"] + [(g["a"], g["b"], 0)] * g["wb"] + [(g["a"], g["b"], 0.5)] * g["d"]
    names = {x for g in games for x in (g["a"], g["b"])}
    finals = {x: [] for x in names}
    rng = random.Random(0)
    for _ in range(SIM_RUNS):
        rng.shuffle(seq)
        r = {x: SITE_INITIAL for x in names}
        for a, b, score in seq:
            # elo.ts と同じく差分を整数に丸めて両者へ
            d = round(SITE_K * (score - expected(r[a], r[b])))
            r[a] += d
            r[b] -= d
        for x in names:
            finals[x].append(r[x])
    return {x: sorted(v) for x, v in finals.items()}


def run_pair(a, b, n):
    out = subprocess.run([ARENA, str(n), a, b], capture_output=True, text=True, cwd=ROOT, check=True).stdout
    m = re.search(r"A=(\S+): (\d+)勝 / B=(\S+): (\d+)勝 / 引き分け (\d+)", out)
    if not m:
        raise SystemExit(f"arena の出力を読めません: {a} vs {b}\n{out[-500:]}")
    return {"a": a, "b": b, "wa": int(m.group(2)), "wb": int(m.group(4)), "d": int(m.group(5))}


def cmd_add(name, n):
    if not os.path.exists(ARENA):
        raise SystemExit("先に cargo build --release --bin arena")
    games = load_games()
    rated = sorted({x for g in games for x in (g["a"], g["b"])} - {name})
    played = {(g["a"], g["b"]) for g in games} | {(g["b"], g["a"]) for g in games}
    for opp in rated:
        if (name, opp) in played:
            continue
        g = run_pair(name, opp, n)
        with open(GAMES, "a") as f:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
        games.append(g)
        print(f"  vs {opp:16s} {g['wa']:3d}-{g['wb']:3d}-{g['d']}", flush=True)
    print()
    cmd_show(matrix=False)


def cmd_show(matrix):
    games = load_games()
    if not games:
        raise SystemExit(f"{GAMES} に対局がありません")
    bt = fit_all(games)
    sim = simulate_site(games)
    s = pair_scores(games)
    total = {x: [0.0, 0] for x in bt}
    for (x, _), (w, k) in s.items():
        total[x][0] += w
        total[x][1] += k
    order = sorted(bt, key=lambda x: -bt[x])
    lo, hi = SIM_RUNS * 5 // 100, SIM_RUNS * 95 // 100 - 1

    out = {}
    print(f"| 順位 | 版 | 一括推定 | サイト式K={SITE_K}（平均 / 5〜95%） | 総合勝率 |")
    print("|---|---|---|---|---|")
    for i, x in enumerate(order, 1):
        v = sim[x]
        mean = sum(v) / len(v)
        rate = total[x][0] / total[x][1]
        print(f"| {i} | {x} | {bt[x]:.0f} | {mean:.0f}（{v[lo]:.0f}〜{v[hi]:.0f}） | {100 * rate:.1f}% |")
        out[x] = {"bt": round(bt[x], 1), "site_mean": round(mean, 1), "site_p5": v[lo], "site_p95": v[hi],
                  "score": round(rate, 4), "games": total[x][1]}
    with open(RATINGS, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")

    if matrix:
        short = lambda x: x.removeprefix("rl_").removeprefix("estimator_")
        print("\n勝率表（行の版の列の版に対する得点率%）\n")
        print("| | " + " | ".join(short(x) for x in order) + " |")
        print("|---" * (len(order) + 1) + "|")
        for x in order:
            cells = ["-" if (x, y) not in s else f"{100 * s[(x, y)][0] / s[(x, y)][1]:.0f}" for y in order]
            print(f"| {short(x)} | " + " | ".join(cells) + " |")


def main():
    args = sys.argv[1:]
    if args[:1] == ["add"] and len(args) in (2, 3):
        cmd_add(args[1], int(args[2]) if len(args) == 3 else 100)
    elif args[:1] == ["show"]:
        cmd_show(matrix="--matrix" in args)
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
