#!/usr/bin/env python3
"""ガントレットの結果を相手の組ごとにまとめ、腕どうしの差を**相手でまとめた誤差**つきで出す
（docs/rl-learning-speed.md 10.7。MMD の追試の判定はこの出力で行う）。

入力は arena.yml の artifact `arena-combined` の `combined.json`（基準ごとの勝-負-分。1 run = 1ファイル）。
1つの腕は複数の run（ガントレットを相手で分けて回したもの）をまとめてよい。

    gh run download <run> -n arena-combined -D runs/<run>
    python3 scripts/ci/arena_groups.py --groups scripts/ci/groups_deduce_lineage.json \\
        --arm mmd_target=runs/1/combined.json,runs/2/combined.json \\
        --arm rnad108250=runs/3/combined.json,runs/4/combined.json \\
        --arm rnad108500=runs/5/combined.json,runs/6/combined.json \\
        --diff mmd_target-rnad108250+rnad108500 --detail 系列外

- 得点は (勝 + 分/2) / 局。差は相手ごとの得点の差を局数で重み付けした平均（pt）。`A-B+C` は右側を B と C の平均にする
  （腕の名前に `-` `+` `,` `=` は使えない）
- **相手でまとめた誤差**: 相手ごとの差 d_o を単位にした cluster-robust な標準誤差
  √(G/(G−1) · Σ (n_o/N)² (d_o − D)²)。同じ match_seed・局数・shards・時計で回した腕どうしなら（arena.yml の局の条件は
  相手・match_seed・シャード・局番号で決まる）d_o は局ごとに対にした差の平均なので、局の揺れ（対にした分だけ小さい）と
  相手による効き方の違いの両方が入る。z はこれで割った値（G − 1 の t で読む）。相手が8体未満の組は自由度が足りず
  誤差が小さく出すぎるので、相手でまとめた誤差と z は出さない
- 参考に、局を独立とみなした二項の誤差（対にしない上限）も出す
- 局数が腕どうしで違う相手は比べられないので止める。1つの腕の run は同じ candidate でなければ止め、違う腕が同じ candidate
  なら止める（run ID の貼り間違いを防ぐ）。腕ごとの candidate を出力の頭に書く
- 差ごとに、その差に出てくる腕のすべてにいる相手だけを数える

    python3 scripts/ci/arena_groups.py --self-test
"""

import argparse
import json
import math
import sys

MIN_CLUSTERS = 8  # これ未満の組は相手でまとめた誤差を出さない
TIE_PT = 1e-9  # 符号を数えるときに 0 とみなす差（pt。平均を取ると 1e-14 程度の丸めが出る）
BAD_NAME_CHARS = "-+,="


def load_arm(paths):
    """({相手: (局, 得点)}, candidate)。同じ相手が2つの run に出たら、candidate が run で違えば止める"""
    out = {}
    cands = set()
    for path in paths:
        with open(path) as f:
            rows = json.load(f)
        for r in rows:
            name = r["baseline"]
            if name in out:
                raise SystemExit(f"{path}: 相手 {name} が2つの run に出ている")
            n = r["wins"] + r["losses"] + r["draws"]
            out[name] = (n, (r["wins"] + 0.5 * r["draws"]) / n)
            if "candidate" in r:
                cands.add(r["candidate"])
    if len(cands) > 1:
        raise SystemExit(f"{', '.join(paths)}: 1つの腕に違う candidate の run が混ざっている: {sorted(cands)}")
    return out, (cands.pop() if cands else "")


def side_score(arms, names, opp):
    """名前の並びの得点の平均（右側の `B+C`）"""
    return sum(arms[a][opp][1] for a in names) / len(names)


def diff_stats(arms, left, right, opps):
    """(D, 相手でまとめた SE, 二項の SE, 相手ごとの差 {相手: pt})。opps が空なら止める"""
    if not opps:
        raise SystemExit("比べる相手が無い")
    ns = []
    ds = {}
    var_b = 0.0
    for o in opps:
        n_set = {arms[a][o][0] for a in left + right}
        if len(n_set) != 1:
            raise SystemExit(f"相手 {o} の局数が腕で違う: " + ", ".join(f"{a} {arms[a][o][0]}" for a in left + right))
        n = n_set.pop()
        ns.append(n)
        ds[o] = (side_score(arms, left, o) - side_score(arms, right, o)) * 100
        vl = sum(arms[a][o][1] * (1 - arms[a][o][1]) for a in left) / len(left) ** 2
        vr = sum(arms[a][o][1] * (1 - arms[a][o][1]) for a in right) / len(right) ** 2
        var_b += n * (vl + vr)  # Σ (n/N)² (vl + vr)/n の N² 倍
    N = sum(ns)
    D = sum(n * ds[o] for n, o in zip(ns, opps)) / N
    G = len(opps)
    se_c = math.sqrt(G / (G - 1) * sum((n / N) ** 2 * (ds[o] - D) ** 2 for n, o in zip(ns, opps))) if G > 1 else float("nan")
    se_b = math.sqrt(var_b) / N * 100
    return D, se_c, se_b, ds


def parse_diff(spec):
    left, sep, right = spec.partition("-")
    if not sep or not left or not right or "-" in right:
        raise SystemExit(f"--diff は A-B（右側は B+C で平均。腕の名前に - + は使えない）: {spec}")
    return left.split("+"), right.split("+")


def diff_opps(arms, names, group):
    """組の相手のうち、names の腕すべてにいる相手（と、いない相手の (相手, 腕)）"""
    have = [o for o in group if all(o in arms[a] for a in names)]
    missing = [(o, a) for o in group for a in names if o not in arms[a]]
    return have, missing


def fmt_cell(D, se_c, se_b, G):
    if abs(D) < 5e-10:
        D = 0.0  # 平均の丸めで −0.00 と出さない
    if G < MIN_CLUSTERS or not se_c > 0:
        why = f"{G}体" if G < MIN_CLUSTERS else "相手ごとの差がすべて同じ"
        return f"{D:+.2f}（{why}。二項 ± {se_b:.2f}）"
    return f"{D:+.2f} ± {se_c:.2f}（z {D / se_c:+.2f}。二項 ± {se_b:.2f}）"


def count_signs(ds):
    pos = sum(1 for x in ds.values() if x > TIE_PT)
    neg = sum(1 for x in ds.values() if x < -TIE_PT)
    return pos, neg, len(ds) - pos - neg


def report(arms, cands, groups, diffs, detail=None):
    lines = ["腕: " + "、".join(f"{a} = {cands.get(a) or '（candidate なし）'}" for a in arms)]
    every = set().union(*[set(v) for v in arms.values()])
    listed = {o for v in groups.values() for o in v}
    rest = sorted(every - listed)
    group_opps = dict(groups)
    if rest:
        lines.append(f"注意: どの組にも無い相手（「組の外」にまとめる）: {', '.join(rest)}")
        group_opps["組の外"] = rest
    if detail is not None and detail not in group_opps:
        raise SystemExit(f"--detail の組 {detail} が無い")
    parsed = [(spec, *parse_diff(spec)) for spec in diffs]
    notes = []
    head = ["| 差 |"]
    for g in group_opps:
        head.append(f" {g} |")
    lines += ["", "".join(head), "| --- |" + " --- |" * len(group_opps)]
    for spec, left, right in parsed:
        cells = []
        for g, members in group_opps.items():
            opps, missing = diff_opps(arms, left + right, members)
            for o, a in missing:
                notes.append(f"注意: {o} は {a} に無い（{spec} の {g} で数えない）")
            if not opps:
                cells.append("（相手なし）")
                continue
            D, se_c, se_b, _ = diff_stats(arms, left, right, opps)
            n = sum(arms[left[0]][o][0] for o in opps)
            cells.append(fmt_cell(D, se_c, se_b, len(opps)) + f"（{len(opps)}体・{n}局）")
        lines.append(f"| {spec} | " + " | ".join(cells) + " |")
    lines += sorted(set(notes))
    lines += ["", f"（± は 1σ。前は相手でまとめたもの〔cluster-robust、G − 1 の t で読む。{MIN_CLUSTERS}体未満の組は出さない〕、"
                  "括弧の中は局を独立とみなした二項）"]
    if detail is not None:
        members = group_opps[detail]
        per = []
        for spec, left, right in parsed:
            opps, _ = diff_opps(arms, left + right, members)
            if not opps:
                raise SystemExit(f"--detail の組 {detail} に {spec} で比べられる相手が無い")
            per.append((spec, left, right, opps, diff_stats(arms, left, right, opps)[3]))
        first = per[0][4]
        lines += ["", f"### {detail} の相手ごとの差（pt）", "", "| 相手 | 局 | " + " | ".join(p[0] for p in per) + " |",
                  "| --- | --- |" + " --- |" * len(per)]
        for o in sorted({o for p in per for o in p[3]}, key=lambda o: -first.get(o, float("-inf"))):
            n = next(arms[p[1][0]][o][0] for p in per if o in p[4])
            lines.append(f"| {o} | {n} | " + " | ".join(f"{p[4][o]:+.1f}" if o in p[4] else "" for p in per) + " |")
        for spec, left, right, opps, ds in per:
            pos, neg, tie = count_signs(ds)
            loo = [diff_stats(arms, left, right, [x for x in opps if x != o])[0] for o in opps] if len(opps) > 2 else []
            lines.append(f"- {spec}: 正 {pos}・負 {neg}・同じ {tie}" + (f"、1体抜きで {min(loo):+.2f}〜{max(loo):+.2f}" if loo else ""))
    return "\n".join(lines)


def self_test():
    def arm(scores, n=100):
        return {f"o{i}": (n, s) for i, s in enumerate(scores)}

    # 3体・各100局。A は B より 0 / +10 / +20pt → 平均 +10、相手でまとめた SE = √(3/2 · Σ (1/3)² (d − 10)²)
    arms = {"A": arm([0.5, 0.6, 0.7]), "B": arm([0.5, 0.5, 0.5]), "C": arm([0.5, 0.7, 0.5])}
    o3 = ["o0", "o1", "o2"]
    D, se_c, se_b, ds = diff_stats(arms, ["A"], ["B"], o3)
    assert abs(D - 10) < 1e-9 and abs(se_c - math.sqrt(1.5 * 200 / 9)) < 1e-9, (D, se_c)
    assert abs(ds["o2"] - 20) < 1e-9
    # 二項: Σ (1/3)² (s_A(1−s_A) + s_B(1−s_B)) / 100 の平方根
    want_b = math.sqrt(sum((1 / 3) ** 2 * (a * (1 - a) + 0.25) / 100 for a in (0.5, 0.6, 0.7))) * 100
    assert abs(se_b - want_b) < 1e-9, (se_b, want_b)
    # 右側の平均: A − (B+C)/2 = 0 / +0 / +20 → +6.67。二項は右側の分散が 1/4
    D2, se_c2, se_b2, _ = diff_stats(arms, ["A"], ["B", "C"], o3)
    assert abs(D2 - 20 / 3) < 1e-9, D2
    want_b2 = math.sqrt(sum((1 / 3) ** 2 * (a * (1 - a) + (0.25 + c * (1 - c)) / 4) / 100
                            for a, c in ((0.5, 0.5), (0.6, 0.7), (0.7, 0.5)))) * 100
    assert abs(se_b2 - want_b2) < 1e-9, (se_b2, want_b2)
    # 局数で重み付け: o0 だけ 300局なら D = (3·0 + 10 + 20) / 5 = 6、SE = √(3/2 · ((3/5)²·36 + (1/5)²·16 + (1/5)²·196))
    arms["A"]["o0"] = (300, 0.5)
    arms["B"]["o0"] = (300, 0.5)
    D3, se_c3, *_ = diff_stats(arms, ["A"], ["B"], o3)
    assert abs(D3 - 6) < 1e-9, D3
    assert abs(se_c3 - math.sqrt(1.5 * (0.36 * 36 + 0.04 * 16 + 0.04 * 196))) < 1e-9, se_c3
    # 局数が違えば止める
    try:
        diff_stats(arms, ["A"], ["C"], ["o0"])
    except SystemExit:
        pass
    else:
        raise AssertionError("局数の違いを見逃した")
    # --diff の読み方
    assert parse_diff("A+B-C+D") == (["A", "B"], ["C", "D"])
    for bad in ("A", "A-", "-B", "A-B-C"):
        try:
            parse_diff(bad)
        except SystemExit:
            continue
        raise AssertionError(f"--diff {bad} を通した")
    # report: 空の組・相手ごとの差が同じ組（SE 0）・ある腕にいない相手・8体未満の組は z を出さない
    many = {f"p{i}": (100, 0.5 + 0.01 * i) for i in range(10)}
    base = {f"p{i}": (100, 0.5) for i in range(10)}
    arms2 = {"X": dict(many, q0=(100, 0.6), q1=(100, 0.7)), "Y": dict(base, q0=(100, 0.5), q1=(100, 0.6)), "Z": dict(base)}
    groups = {"大": [f"p{i}" for i in range(10)], "同じ差": ["q0", "q1"], "空": [], "片方だけ": ["q0"]}
    out = report(arms2, {"X": "x", "Y": "y", "Z": "z"}, groups, ["X-Y", "X-Z"], detail="大")
    assert "+10.00（2体" in out and "（相手なし）" in out and "q0 は Z に無い" in out, out
    row = next(x for x in out.split("\n") if x.startswith("| X-Y |"))
    assert "z +" in row.split("|")[2], out  # 10体の組には z が出る
    assert "正 9・負 0・同じ 1" in out, out
    # 平均の丸めは符号に数えない
    assert count_signs({"a": 1e-14, "b": -1e-14, "c": 0.5}) == (1, 0, 2)
    # load_arm: 分は半分、candidate の混ざりは止める
    import os
    import tempfile
    paths = []
    try:
        for rows in ([{"baseline": "x", "wins": 1, "losses": 1, "draws": 2, "candidate": "c1"}],
                     [{"baseline": "y", "wins": 1, "losses": 0, "draws": 0, "candidate": "c2"}]):
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump(rows, f)
            paths.append(f.name)
        assert load_arm(paths[:1]) == ({"x": (4, 0.5)}, "c1")
        try:
            load_arm(paths)
        except SystemExit:
            pass
        else:
            raise AssertionError("candidate の混ざりを見逃した")
    finally:
        for p in paths:
            os.unlink(p)
    print("self-test ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--groups", help="{組名: [相手, ...]} の JSON")
    ap.add_argument("--arm", action="append", default=[], help="名前=combined.json[,combined.json ...]（名前に - + , = は使えない）")
    ap.add_argument("--diff", action="append", default=[], help="A-B（右側は B+C で平均）")
    ap.add_argument("--detail", default="", help="この組の相手ごとの差も出す")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.groups or not args.arm or not args.diff:
        ap.error("--groups・--arm・--diff が要る")
    with open(args.groups) as f:
        groups = json.load(f)
    arms, cands = {}, {}
    for spec in args.arm:
        name, sep, paths = spec.partition("=")
        if not sep or not name or not paths:
            raise SystemExit(f"--arm は 名前=combined.json[,...]: {spec}")
        if any(c in name for c in BAD_NAME_CHARS):
            raise SystemExit(f"腕の名前 {name} に {BAD_NAME_CHARS} のどれかが入っている")
        if name in arms:
            raise SystemExit(f"--arm {name} が2回ある（run は , でつなぐ）")
        arms[name], cands[name] = load_arm(paths.split(","))
    seen = {}
    for name, c in cands.items():
        if c and c in seen:
            raise SystemExit(f"腕 {seen[c]} と {name} が同じ candidate: {c}")
        seen[c] = name
    for spec in args.diff:
        for a in sum(parse_diff(spec), []):
            if a not in arms:
                raise SystemExit(f"--diff の腕 {a} が --arm に無い")
    print(report(arms, cands, groups, args.diff, args.detail or None))


if __name__ == "__main__":
    main()
