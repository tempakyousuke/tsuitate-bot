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
- 参考に、局を独立とみなした二項の誤差も出す（腕を1つずつ比べるなら対にしないときの上限。`B+C` の右側は B と C を独立とみなすので、
  ほぼ同じネットを平均すると小さく出る）
- 局数が腕どうしで違う相手は比べられないので止める。腕は (candidate, cand_knobs, cand_config) で見分け、1つの腕の run がそろって
  いなければ止め、違う腕が同じなら止める（run ID の貼り間違いを防ぐ。同じ腕を2回比べるときは --allow-same-arm）。腕ごとの candidate を
  出力の頭に書く
- combined.json と同じ場所に all.jsonl（同じ artifact に入っている、シャードごとの記録）があれば、差に出てくる腕どうしで相手ごとに
  match_seed_base・シャードの組・時計がそろっているかを確かめ、違えば止める（無ければ確かめられないと書く）
- 差ごとに、その差に出てくる腕のすべてにいる相手だけを数える

    python3 scripts/ci/arena_groups.py --self-test
"""

import argparse
import json
import math
import pathlib
import sys

MIN_CLUSTERS = 8  # これ未満の組は相手でまとめた誤差を出さない
TIE_PT = 1e-9  # 符号を数えるときに 0 とみなす差（pt。平均を取ると 1e-14 程度の丸めが出る）
BAD_NAME_CHARS = "-+,="


def load_arm(paths):
    """({相手: (局, 得点)}, 腕の見分け (candidate, cand_knobs, cand_config), {相手: 対局の条件} か None)。
    同じ相手が2つの run に出たら、腕の見分けが run で違えば止める。条件は combined.json の隣の all.jsonl から読む（無い run が
    あれば None）"""
    out = {}
    cands = set()  # combined.json の candidate
    ids = set()  # all.jsonl の (candidate, cand_knobs, cand_config)
    cond = {}
    have_all = True
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
        shards = pathlib.Path(path).with_name("all.jsonl")
        if not shards.exists():
            have_all = False
            continue
        for line in shards.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ids.add((r.get("candidate", ""), json.dumps(r.get("cand_knobs") or {}, sort_keys=True),
                     json.dumps(r.get("cand_config"), sort_keys=True)))
            c = cond.setdefault(r["baseline"], {"seed": set(), "shards": set(), "clock": set()})
            c["seed"].add(r.get("match_seed_base"))
            c["shards"].add(r.get("match_seed_shard"))
            c["clock"].add((r.get("fischer_initial_ms"), r.get("fischer_increment_ms")))
    if len(cands) > 1 or len(ids) > 1 or (cands and ids and {i[0] for i in ids} != cands):
        raise SystemExit(f"{', '.join(paths)}: 1つの腕に違う候補（candidate・cand_knobs・cand_config）の run が混ざっている: "
                         f"{sorted(cands)} / {sorted(ids)}")
    ident = ids.pop() if ids else ((cands.pop(), "", "") if cands else ("", "", ""))
    return out, ident, (cond if have_all else None)


def same_arm(a, b):
    """2つの腕の見分けが同じか。片方に all.jsonl の詳しさ（cand_knobs・cand_config）が無ければ candidate だけで比べる"""
    if not a[0] or not b[0]:
        return False
    if a[1:] == ("", "") or b[1:] == ("", ""):
        return a[0] == b[0]
    return a == b


def check_same_arm(idents, allow=False):
    """違う腕が同じ候補なら止める（run ID の貼り間違い）"""
    names = list(idents)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            if same_arm(idents[x], idents[y]) and not allow:
                raise SystemExit(f"腕 {x} と {y} が同じ候補（{idents[x][0]}）。同じ腕を2回比べるなら --allow-same-arm")


def check_conditions(conds, names, opps):
    """差に出てくる腕どうしで、相手ごとに match_seed_base・シャードの組・時計がそろっているか。そろっていなければ止める。
    "ok"・"unchecked"（all.jsonl の無い腕がある）・"unpaired"（match_seed の無い run がある = 局は対になっていない）を返す"""
    if any(conds.get(a) is None for a in names):
        return "unchecked"
    for o in opps:
        got = [(a, conds[a].get(o)) for a in names]
        if any(c is None for _, c in got):
            raise SystemExit(f"相手 {o} の all.jsonl の記録が " + ", ".join(a for a, c in got if c is None) + " に無い")
        first = got[0][1]
        for a, c in got[1:]:
            if c != first:
                raise SystemExit(f"相手 {o} の対局の条件（match_seed_base・シャード・時計）が {got[0][0]} と {a} で違う: "
                                 f"{sorted(map(str, first.items()))} / {sorted(map(str, c.items()))}")
    if any(None in conds[a][o]["seed"] for a in names for o in opps):
        return "unpaired"
    return "ok"


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
    if G < MIN_CLUSTERS or not se_c > TIE_PT:
        why = f"{G}体" if G < MIN_CLUSTERS else "相手ごとの差がすべて同じ"
        return f"{D:+.2f}（{why}。二項 ± {se_b:.2f}）"
    return f"{D:+.2f} ± {se_c:.2f}（z {D / se_c:+.2f}。二項 ± {se_b:.2f}）"


def count_signs(ds):
    pos = sum(1 for x in ds.values() if x > TIE_PT)
    neg = sum(1 for x in ds.values() if x < -TIE_PT)
    return pos, neg, len(ds) - pos - neg


def report(arms, cands, groups, diffs, detail=None, conds=None):
    lines = ["腕: " + "、".join(f"{a} = {cands.get(a) or '（candidate なし）'}" for a in arms)]
    if conds is not None:
        for spec in diffs:
            names = sum(parse_diff(spec), [])
            opps = sorted(set.intersection(*[set(arms[a]) for a in names]))
            status = check_conditions(conds, names, opps)
            if status == "unchecked":
                lines.append(f"注意: {spec} は all.jsonl の無い腕があり、match_seed・シャード・時計がそろっているかを確かめていない")
            elif status == "unpaired":
                lines.append(f"注意: {spec} は match_seed の無い run があり、局は対になっていない（相手でまとめた誤差はそのまま使える）")
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
    # z を出すのは 8体から（7体は出さない）。相手ごとの差がそろった 8体の組（丸めで SE が 1e-16 程度）も z を出さない
    for G, want_z in ((8, True), (7, False)):
        a = {f"r{i}": (100, 0.5 + 0.01 * i) for i in range(G)}
        b = {f"r{i}": (100, 0.5) for i in range(G)}
        out = report({"P": a, "Q": b}, {}, {"g": list(a)}, ["P-Q"])
        assert ("z +" in out) == want_z, (G, out)
    a = {f"r{i}": (100, 0.3 + 0.05 * i) for i in range(8)}
    b = {f"r{i}": (100, 0.3 + 0.05 * i - 0.07) for i in range(8)}
    out = report({"P": a, "Q": b}, {}, {"g": list(a)}, ["P-Q"])
    row = next(x for x in out.split("\n") if x.startswith("| P-Q |"))
    assert "相手ごとの差がすべて同じ" in row and "z " not in row, out
    try:
        report({"P": a, "Q": b}, {}, {"g": list(a)}, ["P-Q"], detail="無い組")
    except SystemExit:
        pass
    else:
        raise AssertionError("--detail の無い組を通した")
    # 左側の平均の二項: (A+B)/2 − C
    arms3 = {"A": arm([0.6]), "B": arm([0.8]), "C": arm([0.5])}
    _, _, se_b3, _ = diff_stats(arms3, ["A", "B"], ["C"], ["o0"])
    assert abs(se_b3 - math.sqrt((0.24 + 0.16) / 4 / 100 + 0.25 / 100) * 100) < 1e-9, se_b3
    # 1体抜きの幅
    out = report(arms2, {}, {"大": [f"p{i}" for i in range(10)]}, ["X-Y"], detail="大")
    loo = [diff_stats(arms2, ["X"], ["Y"], [f"p{j}" for j in range(10) if j != i])[0] for i in range(10)]
    assert f"1体抜きで {min(loo):+.2f}〜{max(loo):+.2f}" in out, out
    # 違う腕が同じ候補なら止める（--allow-same-arm で通す）。同じ candidate でも cand_knobs が違えば別の腕
    try:
        check_same_arm({"x": ("c", "{}", "1"), "y": ("c", "{}", "1")})
    except SystemExit:
        pass
    else:
        raise AssertionError("同じ候補の2つの腕を見逃した")
    check_same_arm({"x": ("c", "{}", "1"), "y": ("c", "{}", "1")}, allow=True)
    check_same_arm({"x": ("c", '{"K": "1"}', "1"), "y": ("c", "{}", "2")})
    try:  # 片方だけ all.jsonl がある（詳しさが違う）ときは candidate で比べる
        check_same_arm({"x": ("c", "{}", "null"), "y": ("c", "", "")})
    except SystemExit:
        pass
    else:
        raise AssertionError("詳しさの違う同じ候補を見逃した")
    # 対局の条件: そろっていれば通す、match_seed_base が違えば止める、all.jsonl の無い腕があれば確かめない
    c1 = {"o0": {"seed": {1}, "shards": {0, 1}, "clock": {(1000, 3)}}}
    c2 = {"o0": {"seed": {2}, "shards": {0, 1}, "clock": {(1000, 3)}}}
    assert check_conditions({"x": c1, "y": c1}, ["x", "y"], ["o0"]) == "ok"
    assert check_conditions({"x": c1, "y": None}, ["x", "y"], ["o0"]) == "unchecked"
    c_none = {"o0": {"seed": {None}, "shards": {None}, "clock": {(1000, 3)}}}
    assert check_conditions({"x": c_none, "y": c_none}, ["x", "y"], ["o0"]) == "unpaired"
    c_shard = {"o0": {"seed": {1}, "shards": {0, 1, 2, 3}, "clock": {(1000, 3)}}}
    c_clock = {"o0": {"seed": {1}, "shards": {0, 1}, "clock": {(300, 3)}}}
    for other in (c2, c_shard, c_clock, {}):
        try:
            check_conditions({"x": c1, "y": other}, ["x", "y"], ["o0"])
        except SystemExit:
            continue
        raise AssertionError(f"条件の違い {other} を見逃した")
    # report から条件を確かめる（違えば止める・確かめられなければ注意を書く）
    one = {"o0": (100, 0.6)}
    try:
        report({"x": one, "y": {"o0": (100, 0.5)}}, {}, {"g": ["o0"]}, ["x-y"], conds={"x": c1, "y": c2})
    except SystemExit:
        pass
    else:
        raise AssertionError("report が条件の違いを見逃した")
    out = report({"x": one, "y": {"o0": (100, 0.5)}}, {}, {"g": ["o0"]}, ["x-y"], conds={"x": c1, "y": None})
    assert "確かめていない" in out, out
    out = report({"x": one, "y": {"o0": (100, 0.5)}}, {}, {"g": ["o0"]}, ["x-y"], conds={"x": c_none, "y": c_none})
    assert "対になっていない" in out, out
    # load_arm: 分は半分、candidate の混ざり・1つの腕の中での相手の重なりは止める、all.jsonl から条件を読む
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        paths = []
        for i, rows in enumerate(([{"baseline": "x", "wins": 1, "losses": 1, "draws": 2, "candidate": "c1"}],
                                  [{"baseline": "y", "wins": 1, "losses": 0, "draws": 0, "candidate": "c2"}],
                                  [{"baseline": "x", "wins": 1, "losses": 0, "draws": 0, "candidate": "c1"}])):
            (tmp / f"r{i}").mkdir()
            path = tmp / f"r{i}" / "combined.json"
            path.write_text(json.dumps(rows))
            paths.append(str(path))
        got = load_arm(paths[:1])
        assert got[0] == {"x": (4, 0.5)} and got[1][0] == "c1" and got[2] is None, got
        for bad in ([paths[0], paths[1]], [paths[0], paths[2]]):
            try:
                load_arm(bad)
            except SystemExit:
                continue
            raise AssertionError(f"{bad} を通した")
        (tmp / "r0" / "all.jsonl").write_text("\n".join(json.dumps(
            {"baseline": "x", "candidate": "c1", "cand_knobs": {}, "cand_config": "h", "match_seed_base": 7,
             "match_seed_shard": sh, "fischer_initial_ms": 1000000, "fischer_increment_ms": 3000}) for sh in (0, 1)) + "\n")
        got = load_arm(paths[:1])
        assert got[1] == ("c1", "{}", '"h"') and got[2] == {"x": {"seed": {7}, "shards": {0, 1}, "clock": {(1000000, 3000)}}}, got
        # all.jsonl のある run と無い run が混ざっても、candidate が違えば順番によらず止め、同じなら通す（条件は確かめない）
        (tmp / "r3").mkdir()
        (tmp / "r3" / "combined.json").write_text(json.dumps([{"baseline": "z", "wins": 1, "losses": 0, "draws": 0, "candidate": "c1"}]))
        got = load_arm([paths[0], str(tmp / "r3" / "combined.json")])
        assert got[1][0] == "c1" and got[2] is None, got
        got = load_arm([str(tmp / "r3" / "combined.json"), paths[0]])
        assert got[1][0] == "c1" and got[2] is None, got
        for order in ([paths[0], paths[1]], [paths[1], paths[0]]):
            try:
                load_arm(order)
            except SystemExit:
                continue
            raise AssertionError(f"{order} を通した")
    print("self-test ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--groups", help="{組名: [相手, ...]} の JSON")
    ap.add_argument("--arm", action="append", default=[], help="名前=combined.json[,combined.json ...]（名前に - + , = は使えない）")
    ap.add_argument("--diff", action="append", default=[], help="A-B（右側は B+C で平均）")
    ap.add_argument("--detail", default="", help="この組の相手ごとの差も出す")
    ap.add_argument("--allow-same-arm", action="store_true", help="違う腕が同じ候補でも止めない（同じ腕どうしの揺れを見るとき）")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.groups or not args.arm or not args.diff:
        ap.error("--groups・--arm・--diff が要る")
    with open(args.groups) as f:
        groups = json.load(f)
    arms, cands, idents, conds = {}, {}, {}, {}
    for spec in args.arm:
        name, sep, paths = spec.partition("=")
        if not sep or not name or not paths:
            raise SystemExit(f"--arm は 名前=combined.json[,...]: {spec}")
        if any(c in name for c in BAD_NAME_CHARS):
            raise SystemExit(f"腕の名前 {name} に {BAD_NAME_CHARS} のどれかが入っている")
        if name in arms:
            raise SystemExit(f"--arm {name} が2回ある（run は , でつなぐ）")
        arms[name], idents[name], conds[name] = load_arm(paths.split(","))
        cands[name] = idents[name][0]
    check_same_arm(idents, args.allow_same_arm)
    for spec in args.diff:
        for a in sum(parse_diff(spec), []):
            if a not in arms:
                raise SystemExit(f"--diff の腕 {a} が --arm に無い")
    print(report(arms, cands, groups, args.diff, args.detail or None, conds))


if __name__ == "__main__":
    main()
