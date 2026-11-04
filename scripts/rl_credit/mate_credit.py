#!/usr/bin/env python3
"""詰まされて負けた局で、負けた側の最後の手番たちへ終局の δ がどれだけ届くか（MMD の GAE）を、λ の数え方ごとに比べる。
あわせて、詰まされた負けの率と、自分がかけた王手に王手駒を取り返された割合を出す（docs/rl-learning-speed.md 10.8）。

入力は arena の A 視点の対局記録（`ARENA_RECORD_DIR`、CI の artifact `arena-records`）のディレクトリ。複数可。

    ARENA_RECORD_DIR=rec ARENA_OPEN_BOOK=rl_openings.json ARENA_OPEN_P=0.25 ARENA_MATCH_SEED=20261103 \\
        cargo run --release --bin arena -- 800 rl_mmdopen_v6 rl_mmdopen_v6
    python3 scripts/rl_credit/mate_credit.py rec

行 = A の試行（my_move / my_foul）。行 j の重み = Π_{m=j}^{n-1} λ_m（n = 最後の行。GAE で終局の δ が行 j の A に入る係数）。
  attempt       : 全行 λ（今の既定）
  turn          : 反則の行だけ λ=1
  attempt+check : 自分が王手をかけた手（受理されて相手が王手になった手）だけ λ=1
  turn+check    : 両方
A の記録しか読まないので、自己対局（A と B が同じ方策）なら「負けた側」の代表になる。違う相手との対局なら A の負けだけを見ている。
`--side mating` なら、A が詰ませて勝った局の A（詰ませた側）で同じ表を出す（10.8 の「なぜ受けでなく攻めが育ったか」）。
"""
import argparse
import json
import statistics
from pathlib import Path

SCHEMES = ["attempt", "turn", "attempt+check", "turn+check"]


def load(path):
    """(A の色, 行の列, 観測の列, 終局) を返す。行は {kind: move|foul, check: 王手をかけた, in_check: 王手されていた, ev: 観測の番号}"""
    rows, events = [], []
    color = end = None
    in_check = False
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        t = d.get("type")
        if t == "match":
            color = d["your_color"]
        elif t == "obs":
            e = d["event"]
            events.append(e)
            k = e["kind"]
            if k == "my_move":
                rows.append({"kind": "move", "check": False, "in_check": in_check, "ev": len(events) - 1})
                in_check = False
            elif k == "my_foul":
                rows.append({"kind": "foul", "check": False, "in_check": in_check, "ev": len(events) - 1})
            elif k == "check":
                if e["in_check"] == color:
                    in_check = True
                elif rows and rows[-1]["kind"] == "move":
                    rows[-1]["check"] = True
        elif t == "end":
            end = d["payload"]
    return color, rows, events, end


def lam_of(row, scheme, lam):
    if row["kind"] == "foul" and scheme in ("turn", "turn+check"):
        return 1.0
    if row["kind"] == "move" and row["check"] and scheme in ("attempt+check", "turn+check"):
        return 1.0
    return lam


def checker_recaptured(events, row, color):
    """その王手に対する相手の応手が王手駒（着手の移動先）を取ったか。応手が無ければ None"""
    nxt = next((e for e in events[row["ev"] + 1:] if e["kind"] == "opponent_moved"), None)
    if nxt is None:
        return None
    return nxt["captured_my_piece_at"] == events[row["ev"]]["usi"][2:4]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="A 視点の対局記録（*.jsonl）のディレクトリ")
    ap.add_argument("--lam", type=float, default=0.5, help="GAE の λ（MMD の既定 0.5）")
    ap.add_argument("--k", type=int, default=12, help="後ろから何手番まで表にするか")
    ap.add_argument("--tail", type=int, default=6, help="王手駒を取り返された割合を「最後の何手番」で分けるか")
    ap.add_argument("--side", choices=["mated", "mating"], default="mated",
                    help="表にする側: mated = A が詰まされた局 / mating = A が詰ませた局")
    args = ap.parse_args()

    files = sorted(p for d in args.dirs for p in Path(d).glob("*.jsonl"))
    n_games, n_mated, n_foul_lost = 0, 0, 0
    lost = []
    recap = {}  # (勝ち/負け・終局理由, 最後/それより前) → [王手, 取り返された]
    for f in files:
        color, rows, events, end = load(f)
        if end is None or color is None:
            continue
        n_games += 1
        res, reason = end["result"], end["reason"]
        a_lost = (res == "sente_win" and color == "gote") or (res == "gote_win" and color == "sente")
        a_won = (res == "sente_win" and color == "sente") or (res == "gote_win" and color == "gote")
        if a_lost and reason == "checkmate":
            n_mated += 1
        if reason == "checkmate" and (a_lost if args.side == "mated" else a_won):
            if rows and rows[-1]["kind"] == "move":
                lost.append(rows)
        if a_lost and reason == "foul_limit":
            n_foul_lost += 1
        moves = [r for r in rows if r["kind"] == "move"]
        key = ("勝ち" if a_won else "負け" if a_lost else "分け") + "・" + reason
        for j, r in enumerate(moves):
            if not r["check"]:
                continue
            got = checker_recaptured(events, r, color)
            if got is None:
                continue
            s = recap.setdefault((key, "最後の手番" if j >= len(moves) - args.tail else "それより前"), [0, 0])
            s[0] += 1
            s[1] += got
    if not n_games:
        raise SystemExit("終局まである記録が無い")
    print(f"記録 {n_games}局（A 視点）: A が詰まされた {n_mated}局（{n_mated / n_games:.1%}）・"
          f"A の反則負け {n_foul_lost}局（{n_foul_lost / n_games:.1%}）")
    if not lost:
        return

    K = args.k
    table = {s: [[] for _ in range(K)] for s in SCHEMES}
    checks, fouls, inchk = ([[] for _ in range(K)] for _ in range(3))
    for rows in lost:
        n = len(rows)
        move_idx = [i for i, r in enumerate(rows) if r["kind"] == "move"]
        for s in SCHEMES:
            w = [0.0] * n
            acc = 1.0
            for j in range(n - 1, -1, -1):
                if j < n - 1:
                    acc *= lam_of(rows[j], s, args.lam)
                w[j] = acc
            for k in range(min(K, len(move_idx))):
                table[s][k].append(w[move_idx[-1 - k]])
        prev = -1
        turns = []
        for i in move_idx:
            turns.append((sum(1 for r in rows[prev + 1:i] if r["kind"] == "foul"), rows[i]["check"], rows[i]["in_check"]))
            prev = i
        for k in range(min(K, len(turns))):
            nf, ch, ic = turns[-1 - k]
            fouls[k].append(nf)
            checks[k].append(ch)
            inchk[k].append(ic)

    what = "詰まされた局" if args.side == "mated" else "詰ませた局"
    k0 = "詰まされる直前の自分の手" if args.side == "mated" else "詰ませた手"
    print(f"\n{what} {len(lost)}局。受理された手を後ろから K 手番目（K=0 = {k0}）へ届く終局の δ の係数"
          f"（λ={args.lam}。平均、括弧は中央値）と、その手番の中身")
    print("| K | " + " | ".join(SCHEMES) + " | 王手をかけた | 王手されていた | 反則/手番 | 局数 |")
    print("|" + "---|" * (len(SCHEMES) + 5))
    for k in range(K):
        if not checks[k]:
            break
        cells = [f"{statistics.mean(table[s][k]):.3f}（{statistics.median(table[s][k]):.3f}）" for s in SCHEMES]
        m = len(checks[k])
        cells += [f"{sum(checks[k]) / m:.0%}", f"{sum(inchk[k]) / m:.0%}", f"{statistics.mean(fouls[k]):.2f}", str(m)]
        print(f"| {k} | " + " | ".join(cells) + " |")
    with_check = sum(1 for rows in lost if any(r["check"] for r in [r for r in rows if r["kind"] == "move"][-5:]))
    print(f"\n最後の5手番に自分の王手が1回以上ある局: {with_check}/{len(lost)} = {with_check / len(lost):.0%}")

    print(f"\n自分の王手のうち、応手で王手駒を取り返された割合（最後の手番 = 最後の {args.tail} 手番）")
    for key in sorted(recap):
        a, b = recap[key]
        print(f"  {key[0]}・{key[1]}: 王手 {a}回・取り返された {b / a:.0%}")


if __name__ == "__main__":
    main()
