"""`bin/search_probe` の出力を集計する（docs/rl-deepnash-design.md の「探索が MMD の版で効かない理由」）。

    python3 scripts/search_probe/report.py NAME=a.jsonl,b.jsonl [NAME=...]

各決定点で π'_η(V) ∝ π·exp(η·V) を V ∈ {q（探索）, q_tp（真の局面）, q_tr（真の局面＋実際の応手）, w（指し継ぎの得点そのもの）}
で作り、指し継ぎの得点 w に対する改善 gain = Σ(π' − π)·w を測る（w を測った候補の中で π と π' を正規化し直す）。
V = w の行は同じ w で選んで同じ w で測るので上に偏る（上限の目安にだけ使う）。
"""

import json
import math
import sys
from collections import defaultdict

ETAS = [1, 3, 10, 30, 100]
VARS = ["q", "q_tp", "q_tr", "w"]


def tilt(p, q, eta):
    m = max(q)
    w = [pi * math.exp(eta * (qi - m)) for pi, qi in zip(p, q)]
    s = sum(w)
    return [x / s for x in w]


def mean_se(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(xs) / n
    if n < 2:
        return m, float("nan"), n
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(v / n), n


def spearman(a, b):
    def rank(x):
        o = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
                j += 1
            for t in range(i, j + 1):
                r[o[t]] = (i + j) / 2
            i = j + 1
        return r

    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den > 0 else None


def load(paths):
    probes, games = [], []
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            (probes if r["kind"] == "probe" else games).append(r)
    return probes, games


def band(ply):
    return "1-20" if ply <= 20 else "21-40" if ply <= 40 else "41-60" if ply <= 60 else "61+"


def report(name, probes, games):
    print(f"\n===== {name}: 局 {len(games)}（平均 {sum(g['plies'] for g in games) / max(len(games), 1):.1f}手）・決定点 {len(probes)}")
    searched = [p for p in probes if p["searched"]]
    print(f"探索した決定点 {len(searched)}（未探索 {len(probes) - len(searched)}）")
    # π の鋭さ
    top = [max(c["p"] for c in p["cands"]) for p in searched]
    print("π の最大: " + " / ".join(f">{t}: {sum(x > t for x in top) / len(top):.1%}" for t in (0.9, 0.99, 0.999)))
    # 探索が手を動かす量（全決定点。w は使わない）
    for eta in (10, 30):
        flip, tv = 0, []
        for p in searched:
            pr = [c["p"] for c in p["cands"]]
            pp = tilt(pr, [c["q"] for c in p["cands"]], eta)
            flip += max(range(len(pr)), key=lambda i: pr[i]) != max(range(len(pp)), key=lambda i: pp[i])
            tv.append(0.5 * sum(abs(a - b) for a, b in zip(pr, pp)))
        print(f"η={eta}: 最大の手が変わる {flip / len(searched):.1%}・π と π' の全変動距離の平均 {sum(tv) / len(tv):.3f}")
    # 粒子の質: 候補が反則になる確率（粒子のうち反則になる割合）の当たり具合。π が 1e-4 未満の候補は除く
    br, ill, on_ill, on_leg = [], 0, [], []
    for p in searched:
        for c in p["cands"]:
            if c["p"] < 1e-4:
                continue
            y = 0.0 if c["legal"] else 1.0
            br.append((c["foul"] - y) ** 2)
            ill += y
            (on_ill if y else on_leg).append(c["foul"])
    taint = sum(p["tainted"] for p in searched) / len(searched)
    print(f"粒子: 反則予測の Brier {sum(br) / len(br):.4f}（候補 {len(br)}、実際の反則率 {ill / len(br):.3f}）・"
          f"実際に反則の手への反則確率 {sum(on_ill) / max(len(on_ill), 1):.2f}・合法な手への反則確率 "
          f"{sum(on_leg) / max(len(on_leg), 1):.3f}・整合する粒子が全滅した決定点 {taint:.1%}")
    # 粒子の偏り（belief がある記録だけ）: 最大の π の手の後に相手モデルの応手で王手される・駒を取られる率と、相手の駒の前進
    bel = [p["belief"] for p in probes if p.get("belief")]
    if bel:
        def avg(key, rows):
            xs = [b[key] for b in rows if b.get(key) is not None]
            return mean_se(xs)
        both = [b for b in bel if b.get("check_truth") is not None and b.get("check_part") is not None]
        print(f"粒子の偏り（{len(bel)} 決定点、最大の π の手が真の局面でも粒子でも合法なのは {len(both)}）:")
        for key, label in [("check", "応手で王手される率"), ("lost", "応手で駒を取られる率")]:
            t, p_ = avg(f"{key}_truth", both), avg(f"{key}_part", both)
            print(f"  {label}: 真の局面 {t[0]:.3f}±{t[1]:.3f} / 粒子 {p_[0]:.3f}±{p_[1]:.3f}")
        for key, label in [("adv", "相手の駒（玉以外）の前進の平均"), ("camp", "自陣3段にいる相手の駒")]:
            t, p_ = avg(f"{key}_truth", bel), avg(f"{key}_part", bel)
            print(f"  {label}: 真の局面 {t[0]:.2f} / 粒子 {p_[0]:.2f}")
    rel = [p for p in searched if p.get("relevant", 0) > 1]
    print(f"指し継ぎを測った決定点（探索が手を動かしうる）{len(rel)}（{len(rel) / max(len(searched), 1):.1%}）")
    if not rel:
        return
    # 候補間の幅（w を測った候補の max − min）と、最大の π の手での Q の偏り
    spread = defaultdict(list)
    bias = defaultdict(list)
    for p in rel:
        cs = [c for c in p["cands"] if c["w"] is not None]
        for v in ["q", "q_tp", "q_tr", "w", "q_tr_alt"]:
            xs = [c.get(v) for c in cs]
            if all(x is not None for x in xs):
                spread[v].append(max(xs) - min(xs))
        top = max(p["cands"], key=lambda c: c["p"])
        bias["q-q_tp"].append(top["q"] - top["q_tp"])
        bias["q_tp-q_tr"].append(top["q_tp"] - top["q_tr"])
    print("候補間の幅の平均: " + " / ".join(f"{v} {sum(x) / len(x):.3f}" for v, x in spread.items() if x))
    print("最大の π の手で Q の偏り（正 = 楽観）: " + " / ".join(
        f"{k} {mean_se(x)[0]:+.3f}±{mean_se(x)[1]:.3f}" for k, x in bias.items()))
    # Q の誤差
    dq_belief, dq_model, se = [], [], []
    for p in rel:
        cs = [c for c in p["cands"] if c["w"] is not None]
        base = cs[0]
        for c in cs[1:]:
            # 候補間の差で比べる（一律のずれは選択に効かない）
            d = (c["q"] - base["q"]), (c["q_tp"] - base["q_tp"]), (c["q_tr"] - base["q_tr"])
            dq_belief.append(abs(d[0] - d[1]))
            dq_model.append(abs(d[1] - d[2]))
        se += [c["se"] for c in cs]
    print(f"候補間の Q 差のずれ: 粒子 vs 真の局面 {sum(dq_belief) / len(dq_belief):.3f}・相手モデル vs 実際の応手 "
          f"{sum(dq_model) / len(dq_model):.3f}・探索の Q の標準誤差の平均 {sum(se) / len(se):.3f}")
    # Q と w の順位相関
    for v in ["q", "q_tp", "q_tr", "q_tr_alt"]:
        rhos = []
        for p in rel:
            cs = [c for c in p["cands"] if c["w"] is not None]
            if len(cs) >= 3 and all(c.get(v) is not None for c in cs):
                r = spearman([c[v] for c in cs], [c["w"] for c in cs])
                if r is not None:
                    rhos.append(r)
        m, s, n = mean_se(rhos)
        if n:
            print(f"  {v:8s} と w の順位相関（候補3つ以上の {n} 決定点）: {m:+.3f} ± {s:.3f}")
    # gain
    print("改善 Σ(π'−π)·w（得点 ±1 の単位、決定点の平均 ± 標準誤差。全決定点あたりに直すと ×"
          f"{len(rel) / len(searched):.2f}）")
    print("  " + "V \\ η".ljust(8) + "".join(f"{e:>16}" for e in ETAS))
    rows = {}
    for v in VARS:
        cells = []
        for eta in ETAS:
            gains = []
            for p in rel:
                cs = [c for c in p["cands"] if c["w"] is not None]
                pr = [c["p"] for c in cs]
                s = sum(pr)
                pr = [x / s for x in pr]
                pp = tilt(pr, [c[v] for c in cs], eta)
                gains.append(sum((a - b) * c["w"] for a, b, c in zip(pp, pr, cs)))
            m, s, n = mean_se(gains)
            rows[(v, eta)] = gains
            cells.append(f"{m:+.4f}±{s:.4f}")
        print("  " + v.ljust(8) + "".join(f"{c:>16}" for c in cells))
    # 手数帯別（探索そのもの、η=30）
    by = defaultdict(list)
    for p, g in zip(rel, rows[("q", 30)]):
        by[band(p["ply"])].append(g)
    print("  探索（q, η=30）の改善を手数帯で: " + " / ".join(
        f"{b}: {mean_se(by[b])[0]:+.3f}±{mean_se(by[b])[1]:.3f}（{len(by[b])}）" for b in ["1-20", "21-40", "41-60", "61+"] if by[b]))
    by = defaultdict(list)
    for p, g in zip(rel, rows[("q", 30)]):
        key = "taint" if p["tainted"] else ("pool<50" if p["pool"] < 50 else "pool>=50")
        by[key].append(g)
    print("  粒子で: " + " / ".join(f"{k}: {mean_se(v)[0]:+.3f}±{mean_se(v)[1]:.3f}（{len(v)}）" for k, v in sorted(by.items())))


def main():
    for spec in sys.argv[1:]:
        name, _, paths = spec.partition("=")
        probes, games = load(paths.split(","))
        report(name, probes, games)


if __name__ == "__main__":
    main()
