#!/usr/bin/env python3
"""負け方（詰み / 反則負け）ごとに、価値ヘッドの見込みがどこで下がり、GAE の責任（A）がどこに配られるかを測る
（docs/rl-learning-speed.md 10.8 の「反則負けの局の責任」）。

入力は arena の A 視点の対局記録のディレクトリ（`ARENA_RECORD_DIR`）。方策ネットの戦略は chose の debug に
価値ヘッドの出力（手番側から見た値）を書くので、それを V として使う（学習の v_actor は現在のネット、記録は書き出した
ターゲットなので近似）。序盤の強制で指した手（chose が無い）は学習と同じく行にしない。

    python3 scripts/rl_credit/loss_credit.py <記録のディレクトリ>

行 = A の試行。δ_i = V_{i+1} − V_i（最後の行は R − V_n）、A_i = Σ λ^l δ_{i+l}（λ=0.5、試行ごと = 今の既定）。
行の種類:
  entry   : 受理された手で、次の自分の決定点で王手されている（王手を許した手）
  foul_ck : 王手されている間の反則
  foul    : 王手されていない反則
  evade   : 王手されている間の受理された手
  other   : それ以外の受理された手
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

LAM = 0.5


def load(path):
    color = end = None
    rows = []
    pend = None
    in_check = False
    for line in open(path):
        d = json.loads(line)
        t = d["type"]
        if t == "match":
            color = d["your_color"]
        elif t == "chose":
            pend = (d.get("debug") or {}).get("value")
        elif t == "obs":
            e = d["event"]
            k = e["kind"]
            if k in ("my_move", "my_foul"):
                if pend is not None:
                    rows.append({"kind": "move" if k == "my_move" else "foul", "in_check": in_check, "v": pend})
                pend = None
                if k == "my_move":
                    in_check = False
            elif k == "check" and e["in_check"] == color:
                in_check = True
        elif t == "end":
            end = d["payload"]
    return color, rows, end


def classify(rows):
    for i, r in enumerate(rows):
        if r["kind"] == "foul":
            r["cls"] = "foul_ck" if r["in_check"] else "foul"
        elif r["in_check"]:
            r["cls"] = "evade"
        else:
            nxt = rows[i + 1] if i + 1 < len(rows) else None
            r["cls"] = "entry" if (nxt is not None and nxt["in_check"]) else "other"


def main(d):
    if not Path(d).is_dir():
        raise SystemExit(f"{d} はディレクトリでない")
    games = defaultdict(list)
    for f in Path(d).glob("*.jsonl"):
        color, rows, end = load(f)
        if not end or not rows:
            continue
        res = end["result"]
        won = (res == "sente_win") == (color == "sente") and res != "draw"
        R = 1.0 if won else (0.0 if res == "draw" else -1.0)
        key = ("勝ち" if won else "負け") + "・" + {"checkmate": "詰み", "foul_limit": "反則"}.get(end["reason"], end["reason"])
        classify(rows)
        n = len(rows)
        for i, r in enumerate(rows):
            r["delta"] = (rows[i + 1]["v"] if i + 1 < n else R) - r["v"]
        acc = 0.0
        for i in range(n - 1, -1, -1):
            acc = rows[i]["delta"] + (LAM * acc if i < n - 1 else 0.0)
            rows[i]["adv"] = acc
        fouls = 0
        for r in rows:
            r["fouls_before"] = fouls
            fouls += r["kind"] == "foul"
        # 何手番前か（受理された手の数で、最後の手番 = 0）
        turns_left = 0
        for r in reversed(rows):
            r["turns_left"] = turns_left
            if r["kind"] == "move":
                turns_left += 1
        games[key].append((R, rows))

    keys = sorted(games)
    print("局数: " + " / ".join(f"{k} {len(games[k])}" for k in keys))

    print("\n## 価値ヘッドは反則の数を見込みに入れているか（全局、手番側の累計反則数ごとの V の平均と、その局の実際の結果〔±1〕の平均）")
    byf = defaultdict(lambda: [[], []])
    for k in keys:
        for R, rows in games[k]:
            for r in rows:
                byf[r["fouls_before"]][0].append(r["v"])
                byf[r["fouls_before"]][1].append(R)
    print("| 累計反則 | 行 | V の平均 | 実際の結果の平均 |")
    print("|---|---|---|---|")
    for f in sorted(byf):
        vs, rs = byf[f]
        print(f"| {f} | {len(vs)} | {st.mean(vs):+.3f} | {st.mean(rs):+.3f} |")

    print("\n## 終局のときの見込み（最後の行の V）と、終局の δ = R − V_n")
    for k in keys:
        vn = [rows[-1]["v"] for R, rows in games[k]]
        dn = [rows[-1]["delta"] for R, rows in games[k]]
        print(f"  {k}: V_n 平均 {st.mean(vn):+.3f}・終局の δ 平均 {st.mean(dn):+.3f}")

    print("\n## 負けた局で、見込みの下がり（Σδ = R − V_0）がどの種類の行で起きたか（1局あたりの和）と、責任 A の配られ方")
    cls_names = ["entry", "foul_ck", "foul", "evade", "other"]
    for k in keys:
        if not k.startswith("負け"):
            continue
        G = len(games[k])
        sd = defaultdict(float)
        sa = defaultdict(float)
        cnt = defaultdict(int)
        last = 0.0
        for R, rows in games[k]:
            for i, r in enumerate(rows):
                dd = r["delta"] if i < len(rows) - 1 else 0.0
                sd[r["cls"]] += dd
                sa[r["cls"]] += r["adv"]
                cnt[r["cls"]] += 1
            last += rows[-1]["delta"]
        tot = sum(sd.values()) + last
        print(f"\n### {k}（{G}局、1局あたりの Σδ = {tot / G:+.3f}）")
        print("| 行の種類 | 1局あたりの行 | Σδ（1局あたり） | 行あたりの δ | Σ A（1局あたり） | 行あたりの A |")
        print("|---|---|---|---|---|---|")
        for c in cls_names:
            if cnt[c]:
                print(f"| {c} | {cnt[c] / G:.1f} | {sd[c] / G:+.3f} | {sd[c] / cnt[c]:+.4f} | {sa[c] / G:+.3f} | {sa[c] / cnt[c]:+.4f} |")
        print(f"| 終局の δ | 1 | {last / G:+.3f} | | | |")

    print("\n## 反則負けの局: 反則はいつ使ったか（最後の手番からの手番数）と、そのときの責任")
    for k in keys:
        if k != "負け・反則":
            continue
        bands = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 999)]
        print("| 最後から何手番前 | 反則（1局あたり） | うち王手中 | 反則の行の A の平均 | 同じ帯の受理された手の A の平均 |")
        print("|---|---|---|---|---|")
        G = len(games[k])
        for lo, hi in bands:
            fr = [r for R, rows in games[k] for r in rows if r["kind"] == "foul" and lo <= r["turns_left"] <= hi]
            mv = [r for R, rows in games[k] for r in rows if r["kind"] == "move" and lo <= r["turns_left"] <= hi]
            if not fr:
                continue
            ck = sum(r["in_check"] for r in fr)
            print(f"| {lo}〜{hi if hi < 999 else ''} | {len(fr) / G:.2f} | {ck / len(fr):.0%} | {st.mean(r['adv'] for r in fr):+.4f} | "
                  f"{st.mean(r['adv'] for r in mv) if mv else float('nan'):+.4f} |")

    print("\n## 王手中にまとめて反則した手番（反則3回以上）: その前の「王手を許した手」と、反則・逃げた手の責任")
    for k in keys:
        out = defaultdict(list)
        for R, rows in games[k]:
            i = 0
            while i < len(rows):
                if rows[i]["in_check"] and (i == 0 or not rows[i - 1]["in_check"] or rows[i - 1]["kind"] == "move"):
                    j = i
                    while j < len(rows) and rows[j]["kind"] == "foul":
                        j += 1
                    nf = j - i
                    if nf >= 3:
                        if i > 0 and rows[i - 1]["cls"] == "entry":
                            out["entry"].append(rows[i - 1]["adv"])
                            out["entry_delta"].append(rows[i - 1]["delta"])
                        out["fouls"].extend(r["adv"] for r in rows[i:j])
                        out["fouls_delta"].append(sum(r["delta"] for r in rows[i:j]))
                        if j < len(rows):
                            out["evade"].append(rows[j]["adv"])
                    i = max(j, i + 1)
                else:
                    i += 1
        if out["fouls"]:
            print(f"  {k}: {len(out['fouls_delta'])}回。王手を許した手 A {st.mean(out['entry']):+.3f}（δ {st.mean(out['entry_delta']):+.3f}）"
                  f"・反則の行 A {st.mean(out['fouls']):+.3f}（反則の間の Σδ {st.mean(out['fouls_delta']):+.3f}）"
                  f"・逃げた手 A {st.mean(out['evade']) if out['evade'] else float('nan'):+.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
