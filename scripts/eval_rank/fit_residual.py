#!/usr/bin/env python3
"""人手採点の**残差ランカー**が棋譜の外へ一般化するかを判定する（issue #24 P0）。

    python3 scripts/eval_rank/fit_residual.py data/eval_rank.csv [--out docs/xxx.md]

入力は `bin/export_eval_rank_data` の CSV（1行 = 候補手 × seed）。

判定の要点（issue #24「P0 の暫定合格条件」）:

- **分割単位は元 KIF だけ**。行・候補・決定点のランダム分割はリークするので実装しない。
  同じ棋譜の通常ブロック・反則後ブロック・全 seed は必ず同じ fold に入る
- 主評価は leave-one-source-KIF-out、ハイパーパラメータは fold 内の nested CV
- 指標は **source KIF ごとの macro 平均**。5,035行の micro 平均は主張に使わない
- 大きい quest 2局は個別に表示する（8クラスタしか無いので CI は過信しない）

ラベルは絶対値回帰でなく**同一決定状態内の順位**（点差2以上=明確な preference、
点差1=弱い重み、同点=対象外、`?`=欠測）。特徴量に USI・絶対マス・棋譜名・ply の
固有IDは入れない（exporter の時点で入っていない）。

numpy が要る（`pip install numpy`）。線形（ridge）だけを見る: 8クラスタしか
独立標本が無い段階で MLP を先に選ばない、というのが issue の方針。
"""

import argparse
import csv
import math
import pathlib
import random
import sys

try:
    import numpy as np
except ImportError:  # pragma: no cover
    sys.exit("numpy が要ります: pip install numpy")

# 点差ごとの pairwise 重み（issue #24「2. ラベルとモデル」）
W_STRONG, W_WEAK = 1.0, 0.3
BOUND_CAP = 2.0  # P1 の bounded residual の cap（人手採点の点数スケール）
# 未採点手を選んだ率の許容増分。これを超えたら**逃避した分が測れていない**ので、
# 他の条件が通っていても P0 は確定しない（PR #25 レビュー指摘 P1）
UNSCORED_TOL = 0.01
ALPHAS = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]

ID_COLS = 11  # source_kif..engine_score


class Group:
    """1決定状態 × 1 seed ぶんの候補集合"""

    def __init__(self, cluster, state, seed):
        self.cluster, self.state, self.seed = cluster, state, seed
        self.usi, self.score, self.rows = [], [], []
        # タイブレーク乱数込みの実際の順位とスコア（特徴量ではない）。
        # baseline の再現と、P1 の統合形の**数値**合成に使う
        self.engine_rank, self.engine_score = [], []
        self.absent = []  # 採点済みだが現行候補集合に無い手の点


def load(path):
    groups = {}
    feat_cols = None
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        feat_cols = header[ID_COLS:]
        for row in r:
            (src, state, _scenario, _ply, _side, seed, usi, human, in_cand,
             engine_rank, engine_score) = row[:ID_COLS]
            key = (state, seed)
            g = groups.get(key)
            if g is None:
                g = Group(src, state, seed)
                groups[key] = g
            if in_cand == "0":
                if human:
                    g.absent.append(float(human))
                continue
            g.usi.append(usi)
            g.score.append(float(human) if human else None)
            g.engine_rank.append(int(engine_rank))
            g.engine_score.append(float(engine_score))
            g.rows.append([float(x) for x in row[ID_COLS:]])
    out = []
    for g in groups.values():
        g.X = np.array(g.rows, dtype=float)
        g.engine_rank = np.array(g.engine_rank, dtype=float)
        g.engine_score = np.array(g.engine_score, dtype=float)
        g.rows = None
        # 同じ決定状態の中で move_number 等は定数なので、後で分散ゼロ列を落とす
        out.append(g)
    out.sort(key=lambda g: (g.cluster, g.state, g.seed))
    return out, feat_cols


def pairs_of(g):
    """(i, j, weight, delta): 点差つきの順序対（i が上位）"""
    idx = [k for k, s in enumerate(g.score) if s is not None]
    for a in range(len(idx)):
        for b in range(len(idx)):
            i, j = idx[a], idx[b]
            d = g.score[i] - g.score[j]
            if d <= 0:
                continue
            yield i, j, (W_STRONG if d >= 2 else W_WEAK), d


class ClusterStats:
    """1クラスタ（元 KIF）の**十分統計**。

    pairwise ridge は `(ΔXᵀWΔX + αI)β = ΔXᵀWΔy` なので、クラスタごとに
    生の `G = ΔXᵀWΔX` と `b = ΔXᵀWΔy` を持てば、任意の訓練集合の解は
    行列の**足し算**だけで出る（標準化は `D G D` / `D b`）。
    nested CV も学習曲線も対の作り直しなしで回せる。
    """

    def __init__(self, groups, n_feat):
        XD, Y, W = [], [], []
        for g in groups:
            for i, j, w, d in pairs_of(g):
                XD.append(g.X[i] - g.X[j])
                Y.append(d)
                W.append(w)
        self.pairs = len(XD)
        if XD:
            XD = np.array(XD)
            Wv = np.array(W)
            Xw = XD * Wv[:, None]
            self.G = XD.T @ Xw
            self.b = Xw.T @ np.array(Y)
        else:
            self.G = np.zeros((n_feat, n_feat))
            self.b = np.zeros(n_feat)
        X = np.vstack([g.X for g in groups])
        self.n = X.shape[0]
        self.sum = X.sum(axis=0)
        self.sumsq = (X * X).sum(axis=0)


def train_scale(stats, clusters):
    """訓練 fold の候補行だけから標準化統計を作る（**分割前に計算しない**。
    NN フェーズ1 で正規化統計を分割前に計算して不合格になった教訓）"""
    n = sum(stats[c].n for c in clusters)
    mean = sum(stats[c].sum for c in clusters) / n
    var = sum(stats[c].sumsq for c in clusters) / n - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))
    keep = [k for k in range(len(std)) if std[k] > 1e-9]
    return keep, std[keep]


def fit_ridge_from(stats, clusters, keep, scale, alpha):
    G = sum(stats[c].G for c in clusters)[np.ix_(keep, keep)]
    b = sum(stats[c].b for c in clusters)[keep]
    D = np.diag(1.0 / scale)
    Gs, bs = D @ G @ D, D @ b
    return np.linalg.solve(Gs + alpha * np.eye(len(keep)), bs)


def rank_scores(g, beta, cols_keep, scale):
    return (g.X[:, cols_keep] / scale) @ beta


def baseline_scores(g, feat_cols=None):
    """現行方策の**実スコア**（タイブレーク乱数込み）。

    特徴量の `score` 列は乱数を除いてあるので baseline はそちらを使わない。
    `engine_rank` でなく `engine_score` を返すのが要点で（PR #25 レビュー指摘 P1）、
    順位番号に残差を足すと候補間の実スコア差が消えて、P1 が提案する
    `score + W·cap·tanh(残差/cap)` とは別物の検証になる。
    argmax は `engine_rank` 昇順と一致するので baseline の順位付けは変わらない。
    """
    return g.engine_score


def group_metrics(g, pred):
    """1決定状態ぶんの指標。

    **未採点の top-1 を仮の点で埋めない**（PR #25 レビュー指摘）。`?` は欠測という
    契約どおり、得点系は「top-1 が採点済みの決定状態」だけで条件つきに数え、
    「未採点手を選んだ率」を別指標として並べる。仮 4 点で混ぜると、未採点選択が
    5.3% → 16.1% と増える側の腕の得点が、選ばれた手の実点次第で反転しうる。
    条件つきの比較は腕ごとに母集団が変わるので、合否は `paired_metrics` の
    **両腕とも採点済みの決定状態**で見ること。
    """
    order = np.argsort(-pred)
    top1 = order[0]
    scored = [s for s in g.score if s is not None]
    best = max(scored) if scored else None
    t1 = g.score[top1]
    m = {"top1_unscored": 1.0 if t1 is None else 0.0}
    if t1 is not None:
        m["top1_human"] = t1
        m["top1_bad"] = 1.0 if t1 <= 2 else 0.0
        if best is not None:
            m["regret"] = best - t1
    good = [k for k, s in enumerate(g.score) if s is not None and s >= 8]
    if good:
        top3 = set(order[:3].tolist())
        m["good_top3_recall"] = sum(1 for k in good if k in top3) / len(good)
    ok = wsum = ok2 = n2 = 0.0
    for i, j, w, d in pairs_of(g):
        wsum += w
        if pred[i] > pred[j]:
            ok += w
        if d >= 2:
            n2 += 1
            if pred[i] > pred[j]:
                ok2 += 1
    if wsum:
        m["concordance"] = ok / wsum
    if n2:
        m["concordance_strong"] = ok2 / n2
    # 候補生成 recall: 最高得点手が現在の候補集合にあるか
    if scored or g.absent:
        m["cand_recall"] = 1.0 if (scored and max(scored) >= max(g.absent, default=-1)) else 0.0
    return m


def macro(per_cluster, key):
    vals = [v[key] for v in per_cluster.values() if key in v]
    return sum(vals) / len(vals) if vals else float("nan")


def eval_fold(groups, pred_fn):
    """クラスタ内平均（決定状態ごとの指標を、まず状態で seed 平均してから平均）"""
    by_state = {}
    for g in groups:
        by_state.setdefault(g.state, []).append(group_metrics(g, pred_fn(g)))
    agg = {}
    keys = set(k for ms in by_state.values() for m in ms for k in m)
    for k in keys:
        per_state = []
        for ms in by_state.values():
            vals = [m[k] for m in ms if k in m]
            if vals:
                per_state.append(sum(vals) / len(vals))
        if per_state:
            agg[k] = sum(per_state) / len(per_state)
    agg["_states"] = len(by_state)
    return agg


def paired_metrics(groups, pred_a, pred_b):
    """**両腕とも top-1 が採点済み**の決定状態だけで比べる（合否と CI はこれで見る）。

    条件つきの top-1 平均は腕ごとに母集団が変わるので、そのままでは
    「未採点へ逃げた分だけ悪い手が母集団から消える」腕が有利になる。
    同じ決定状態の上で対にすれば、その偏りが入らない。
    得点系（top-1 得点・0〜2点率・regret）は**すべてこの経路で出す**
    （cluster bootstrap も含む。PR #25 レビュー指摘 P2）。

    戻り値は `{"top1_human": (a, b), "top1_bad": (a, b), "regret": (a, b),
    "paired_states": n, "states": 全体}`。対にできる状態が無ければ None。
    """
    by_state = {}
    for g in groups:
        ia, ib = int(np.argmax(pred_a(g))), int(np.argmax(pred_b(g)))
        best = max((s for s in g.score if s is not None), default=None)
        by_state.setdefault(g.state, []).append((g.score[ia], g.score[ib], best))
    acc = {k: ([], []) for k in ("top1_human", "top1_bad", "regret")}
    n = 0
    for ms in by_state.values():
        both = [(x, y, bt) for x, y, bt in ms if x is not None and y is not None]
        if not both:
            continue
        n += 1
        acc["top1_human"][0].append(sum(x for x, _, _ in both) / len(both))
        acc["top1_human"][1].append(sum(y for _, y, _ in both) / len(both))
        acc["top1_bad"][0].append(sum(x <= 2 for x, _, _ in both) / len(both))
        acc["top1_bad"][1].append(sum(y <= 2 for _, y, _ in both) / len(both))
        rg = [(bt - x, bt - y) for x, y, bt in both if bt is not None]
        if rg:
            acc["regret"][0].append(sum(x for x, _ in rg) / len(rg))
            acc["regret"][1].append(sum(y for _, y in rg) / len(rg))
    if not n:
        return None
    out = {
        k: (sum(v[0]) / len(v[0]), sum(v[1]) / len(v[1]))
        for k, v in acc.items()
        if v[0]
    }
    out["paired_states"] = n
    out["states"] = len(by_state)
    return out


def seed_agreement(groups, pred_fn):
    by_state = {}
    for g in groups:
        pred = pred_fn(g)
        by_state.setdefault(g.state, []).append(g.usi[int(np.argmax(pred))])
    tot = hit = 0
    for choices in by_state.values():
        for a in range(len(choices)):
            for b in range(a + 1, len(choices)):
                tot += 1
                hit += choices[a] == choices[b]
    return hit / tot if tot else float("nan")


def stratum_of(g):
    s = []
    s.append("被王手" if g.X[0][FEAT.index("in_check")] > 0 else "通常")
    s.append("反則後" if g.X[0][FEAT.index("fouls_this_turn")] > 0 else "反則前")
    mn = g.X[0][FEAT.index("move_number")]
    s.append("序中盤" if mn < 50 else ("中盤" if mn < 90 else "終盤"))
    s.append("quest" if g.cluster.startswith("quest") else "arena")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", help="markdown レポートの書き出し先")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    groups, feat_cols = load(args.csv)
    global FEAT
    FEAT = feat_cols
    clusters = sorted({g.cluster for g in groups})
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    n_rows = sum(len(g.usi) for g in groups)
    n_scored = sum(sum(1 for s in g.score if s is not None) for g in groups)
    n_absent = sum(len(g.absent) for g in groups)
    say("# eval 残差ランカー P0: 棋譜外への一般化")
    say()
    say(f"- 入力: `{args.csv}`")
    say(f"- 候補行 {n_rows}（うち採点済み {n_scored}）/ 決定状態×seed {len(groups)} 組")
    say(f"- 採点済みだが現行候補集合に無い手 {n_absent} 行")
    say(f"- クラスタ（元 KIF）{len(clusters)}: {', '.join(clusters)}")
    say()
    say("| source KIF | 決定状態 | 候補行 | 採点済み行 | 行シェア |")
    say("|---|---:|---:|---:|---:|")
    for c in clusters:
        gs = [g for g in groups if g.cluster == c]
        r = sum(len(g.usi) for g in gs)
        sc = sum(sum(1 for s in g.score if s is not None) for g in gs)
        say(f"| {c} | {len({g.state for g in gs})} | {r} | {sc} | {r / n_rows:.1%} |")
    say()

    # --- クラスタごとの十分統計（以後の fit は行列の足し算だけで済む）
    by_cluster = {c: [g for g in groups if g.cluster == c] for c in clusters}
    stats = {c: ClusterStats(by_cluster[c], len(feat_cols)) for c in clusters}
    say("- 学習に使う順序対（点差>0）の数: "
        + ", ".join(f"{c} {stats[c].pairs}" for c in clusters))
    say()

    def fitted(train_clusters, alpha):
        keep, scale = train_scale(stats, train_clusters)
        return fit_ridge_from(stats, train_clusters, keep, scale, alpha), keep, scale

    # --- 採点の被覆（**この節が P0 の効き方を決める**）。
    # eval は make_eval が出した「その時点の上位15候補＋実戦手」だけを列挙するので、
    # 候補集合の大半は未採点のまま。順位学習の教師も評価も採点済みの部分集合しか
    # 見られない一方、**argmax はいつでも未採点側へ逃げられる**
    n_states = len({g.state for g in groups})
    mean_cand = sum(len(g.usi) for g in groups) / len(groups)
    mean_scored = sum(sum(1 for x in g.score if x is not None) for g in groups) / len(groups)
    cov = {k: [0, 0] for k in (1, 3, 10)}
    for g in groups:
        order = np.argsort(-baseline_scores(g, feat_cols))
        for k in cov:
            for i in order[:k]:
                cov[k][1] += 1
                cov[k][0] += g.score[i] is not None
    say("## 採点の被覆")
    say()
    say(f"- 決定状態 {n_states}（× seed で {len(groups)} 組）")
    say(f"- 1決定状態あたりの候補 {mean_cand:.1f} 手、うち採点済み {mean_scored:.1f} 手"
        f"（{mean_scored / mean_cand:.1%}）")
    for k in sorted(cov):
        a, b = cov[k]
        say(f"- 現行 score の top-{k} が採点済みである割合: {a / b:.1%}")
    say()

    # --- leave-one-source-KIF-out（外側）+ 訓練 fold 内の nested CV（α 選択）
    base_res, model_res, chosen = {}, {}, {}
    for held in clusters:
        te = by_cluster[held]
        inner_clusters = [c for c in clusters if c != held]
        best_alpha, best_score = ALPHAS[len(ALPHAS) // 2], -1e9
        # 内側 CV も**元 KIF 単位**。訓練 fold が2クラスタ未満なら α は既定のまま
        for alpha in ALPHAS if len(inner_clusters) >= 2 else []:
            accs = []
            for inner in inner_clusters:
                itr = [c for c in inner_clusters if c != inner]
                beta, keep, scale = fitted(itr, alpha)
                m = eval_fold(by_cluster[inner], lambda g: rank_scores(g, beta, keep, scale))
                if "concordance" in m:
                    accs.append(m["concordance"])
            if accs and sum(accs) / len(accs) > best_score:
                best_score, best_alpha = sum(accs) / len(accs), alpha
        beta, keep, scale = fitted(inner_clusters, best_alpha)
        chosen[held] = (best_alpha, beta, keep, scale)
        base_res[held] = eval_fold(te, lambda g: baseline_scores(g, feat_cols))
        model_res[held] = eval_fold(te, lambda g: rank_scores(g, beta, keep, scale))

    keys = [
        ("top1_human", "top-1 平均得点（採点済み条件つき）", "+"),
        ("top1_bad", "0〜2点 top-1 率（採点済み条件つき）", "-"),
        ("good_top3_recall", "8〜10点 top-3 recall", "+"),
        ("regret", "regret（最高得点との差）", "-"),
        ("concordance", "pairwise concordance", "+"),
        ("concordance_strong", "concordance（点差2以上）", "+"),
        ("top1_unscored", "未採点手を選んだ率", "-"),
        ("cand_recall", "候補生成 recall", "+"),
    ]
    say("## leave-one-source-KIF-out（現行 score との比較）")
    say()
    say("| 指標 | 現行 score | 残差ランカー | 差 |")
    say("|---|---:|---:|---:|")
    for k, label, sign in keys:
        b, m = macro(base_res, k), macro(model_res, k)
        say(f"| {label} | {b:.3f} | {m:.3f} | {m - b:+.3f} |")
    say()
    say("条件つきの列は母集団が腕ごとに変わる（未採点へ逃げた分だけ悪手が母集団から"
        "消える）ので、**合否は下の対比較で見る**。")
    say()
    say("### 両腕とも top-1 が採点済みの決定状態だけの対比較")
    say()
    say("| holdout | 状態数（対 / 全体） | top-1 得点 現行→模型 | 0〜2点 top-1 率 現行→模型 |")
    say("|---|---:|---|---|")
    paired = {}
    for c in clusters:
        _, beta_c, keep_c, scale_c = chosen[c]
        r = paired_metrics(
            by_cluster[c],
            lambda g: baseline_scores(g),
            lambda g, b=beta_c, k=keep_c, sc=scale_c: rank_scores(g, b, k, sc),
        )
        paired[c] = r
        if r is None:
            say(f"| {c} | 0 | — | — |")
            continue
        say(f"| {c} | {r['paired_states']} / {r['states']} | "
            f"{r['top1_human'][0]:.2f} → {r['top1_human'][1]:.2f} | "
            f"{r['top1_bad'][0]:.3f} → {r['top1_bad'][1]:.3f} |")
    pv = [r for r in paired.values() if r]
    if pv:
        say(f"| **macro** | — | **{sum(r['top1_human'][0] for r in pv) / len(pv):.3f} → "
            f"{sum(r['top1_human'][1] for r in pv) / len(pv):.3f}** | "
            f"**{sum(r['top1_bad'][0] for r in pv) / len(pv):.3f} → "
            f"{sum(r['top1_bad'][1] for r in pv) / len(pv):.3f}** |")
    say()

    say("### fold（= 元 KIF）ごと")
    say()
    say("| holdout | α | 状態数 | top-1 得点 現行→模型 | concordance 現行→模型 | 0〜2点 top-1 率 |")
    say("|---|---:|---:|---|---|---|")
    for c in clusters:
        b, m = base_res[c], model_res[c]
        say(
            f"| {c} | {chosen[c][0]:g} | {b['_states']} | "
            f"{b.get('top1_human', float('nan')):.2f} → {m.get('top1_human', float('nan')):.2f} | "
            f"{b.get('concordance', float('nan')):.3f} → {m.get('concordance', float('nan')):.3f} | "
            f"{b.get('top1_bad', float('nan')):.3f} → {m.get('top1_bad', float('nan')):.3f} |"
        )
    say()

    # --- 元 KIF 単位の cluster bootstrap（8本しか無いので過信しない）
    say("### cluster bootstrap（元 KIF 単位・8クラスタなので参考値）")
    say()
    say("得点系（top-1 得点・0〜2点率・regret）は**対比較のクラスタ差**を resample する。"
        "腕ごとに条件つき母集団が違う平均どうしを引くと、未採点への逃避の偏りが CI にも残る。")
    say()
    rng = random.Random(20260826)

    def boot(deltas_by_cluster):
        """クラスタごとの差を元 KIF 単位で resample した中央値と 95% CI"""
        vals = []
        for _ in range(args.boot):
            pick = [rng.choice(clusters) for _ in clusters]
            d = [deltas_by_cluster[c] for c in pick if deltas_by_cluster.get(c) is not None]
            d = [x for x in d if not math.isnan(x)]
            if d:
                vals.append(sum(d) / len(d))
        if not vals:
            return None
        vals.sort()
        return (
            vals[len(vals) // 2],
            vals[int(0.025 * len(vals))],
            vals[int(0.975 * len(vals)) - 1],
        )

    say("| 指標 | 差の中央値 | 95% CI | 出どころ |")
    say("|---|---:|---|---|")
    for k, label, _ in keys:
        if k in ("top1_human", "top1_bad", "regret"):
            src = "対比較"
            deltas = {
                c: (paired[c][k][1] - paired[c][k][0])
                if paired[c] and k in paired[c]
                else None
                for c in clusters
            }
        else:
            src = "条件つき"
            deltas = {
                c: model_res[c].get(k, float("nan")) - base_res[c].get(k, float("nan"))
                for c in clusters
            }
        r = boot(deltas)
        if r:
            say(f"| {label} | {r[0]:+.3f} | [{r[1]:+.3f}, {r[2]:+.3f}] | {src} |")
    say()

    # --- 層別（holdout 予測を集めて層で切る）
    say("### 層別（holdout 予測のみ）")
    say()
    say("| 層 | 状態数 | top-1 得点 現行→模型 | concordance 現行→模型 |")
    say("|---|---:|---|---|")
    strata = {}
    for c in clusters:
        _, beta, keep, scale = chosen[c]
        for g in [x for x in groups if x.cluster == c]:
            for s in stratum_of(g):
                strata.setdefault(s, []).append((g, beta, keep, scale))
    for s in ["通常", "被王手", "反則前", "反則後", "序中盤", "中盤", "終盤", "quest", "arena"]:
        gs = strata.get(s)
        if not gs:
            continue
        b = eval_fold([g for g, *_ in gs], lambda g: baseline_scores(g, feat_cols))
        lut = {id(g): (be, k, sc) for g, be, k, sc in gs}
        m = eval_fold(
            [g for g, *_ in gs],
            lambda g: rank_scores(g, *lut[id(g)]),
        )
        say(
            f"| {s} | {b['_states']} | {b.get('top1_human', float('nan')):.2f} → "
            f"{m.get('top1_human', float('nan')):.2f} | "
            f"{b.get('concordance', float('nan')):.3f} → {m.get('concordance', float('nan')):.3f} |"
        )
    say()

    # --- seed 一致率（粒子揺れへの頑健性。seed を独立標本としては数えない）
    # --- fold ごとの beta を決定状態から引ける形にしておく（W スイープと seed 一致率で使う）
    lut_fold = {}
    for c in clusters:
        _, beta_c, keep_c, scale_c = chosen[c]
        for g in by_cluster[c]:
            lut_fold[id(g)] = (beta_c, keep_c, scale_c)

    say("### seed を変えたときの選択一致率")
    say()
    b = seed_agreement(groups, lambda g: baseline_scores(g))
    m = seed_agreement(groups, lambda g: rank_scores(g, *lut_fold[id(g)]))
    say(f"- 現行 score {b:.3f} / 残差ランカー {m:.3f}")
    say()

    # --- 「current_score の再現しかしていない」かの検査
    say("### 学習された係数（全クラスタで再フィット、α は fold の最頻値）")
    say()
    alpha = max(set(a for a, *_ in chosen.values()), key=lambda a: sum(1 for x, *_ in chosen.values() if x == a))
    beta, keep, scale = fitted(clusters, alpha)
    order = sorted(range(len(keep)), key=lambda k: -abs(beta[k]))
    say(f"- α={alpha:g} / 標準化後の係数（|w| 上位12）")
    say()
    say("| 特徴量 | 係数 |")
    say("|---|---:|")
    for k in order[:12]:
        say(f"| {feat_cols[keep[k]]} | {beta[k]:+.3f} |")
    sc_i = keep.index(feat_cols.index("score")) if feat_cols.index("score") in keep else None
    if sc_i is not None:
        rest = math.sqrt(float(sum(beta[k] ** 2 for k in range(len(keep)) if k != sc_i)))
        say()
        say(f"- `score` 係数 {beta[sc_i]:+.3f} / それ以外のノルム {rest:.3f}"
            "（後者が小さければ現行 score の再現でしかない）")
    say()

    # --- **P1 の統合形**（bounded residual）の W スイープ。
    # 素の残差ランカー（現行 score を捨てた形）が負けても、P1 が提案する
    # `score + W·cap·tanh(residual/cap)` なら現行 score が背骨として残るので、
    # 小さい W で改善する余地があるかを直接見る。W=0 は現行と完全一致
    say("### bounded residual（P1 の統合形）の W スイープ")
    say()
    say(f"`pred = score + W × cap × tanh((r − r̄) / cap)`（cap={BOUND_CAP:g} 点、r = 残差モデルの出力）。")
    say("W=0 は現行と完全一致。得点は**W=0 と対にした比較**（両腕とも top-1 が"
        "採点済みの決定状態だけ）で、母集団が W ごとに動かないようにしてある。")
    say()
    say("macro は1決定状態しかないクラスタも106状態のクラスタも同じ重みで平均するので、"
        "**大きい quest 2局の対差を並べて**改善が少数の状態で説明されないかを見る。")
    say()
    big2 = [c for c in clusters if c.startswith("quest")]
    say("| W | macro top-1 得点 W=0→W | macro 0〜2点 top-1 率 | "
        + " | ".join(f"{c} の差" for c in big2)
        + " | 未採点手を選んだ率 | concordance |")
    say("|---:|---|---|" + "---:|" * len(big2) + "---:|---:|")
    for w in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]:

        def pred_w(g, w=w):
            beta_w, keep_w, scale_w = lut_fold[id(g)]
            r = rank_scores(g, beta_w, keep_w, scale_w)
            return baseline_scores(g) + w * BOUND_CAP * np.tanh((r - r.mean()) / BOUND_CAP)

        res = {c: eval_fold(by_cluster[c], pred_w) for c in clusters}
        per = {
            c: paired_metrics(by_cluster[c], lambda g: baseline_scores(g), pred_w)
            for c in clusters
        }
        pr = [r for r in per.values() if r]
        big_cells = [
            f"{per[c]['top1_human'][1] - per[c]['top1_human'][0]:+.3f}" if per[c] else "—"
            for c in big2
        ]
        say(
            f"| {w:g} | {sum(r['top1_human'][0] for r in pr) / len(pr):.3f} → "
            f"{sum(r['top1_human'][1] for r in pr) / len(pr):.3f} | "
            f"{sum(r['top1_bad'][0] for r in pr) / len(pr):.3f} → "
            f"{sum(r['top1_bad'][1] for r in pr) / len(pr):.3f} | "
            + " | ".join(big_cells)
            + f" | {macro(res, 'top1_unscored'):.3f} | {macro(res, 'concordance'):.3f} |"
        )
    say()

    # --- 学習曲線: 訓練クラスタ数と holdout concordance の関係。
    # 「方向は良いが fold 間分散が大きい」ときに**あと何局要るか**を見積もる材料
    say("### 学習曲線（訓練に使った元 KIF の数 → holdout concordance）")
    say()
    rng2 = random.Random(20260826)
    say("| 訓練 KIF 数 | holdout concordance（現行 score との差） | 標本数 |")
    say("|---:|---|---:|")
    for k in range(1, len(clusters)):
        deltas = []
        for held in clusters:
            pool = [c for c in clusters if c != held]
            subsets = set()
            for _ in range(12):
                subsets.add(tuple(sorted(rng2.sample(pool, k))))
            for sub in subsets:
                beta_k, keep_k, scale_k = fitted(list(sub), chosen[held][0])
                m = eval_fold(by_cluster[held], lambda g: rank_scores(g, beta_k, keep_k, scale_k))
                if "concordance" in m and "concordance" in base_res[held]:
                    deltas.append(m["concordance"] - base_res[held]["concordance"])
        if deltas:
            mean = sum(deltas) / len(deltas)
            sd = math.sqrt(sum((x - mean) ** 2 for x in deltas) / max(len(deltas) - 1, 1))
            say(f"| {k} | {mean:+.3f} ± {sd:.3f} | {len(deltas)} |")
    say()

    # --- P0 の合格条件
    say("## P0 の暫定合格条件との突き合わせ")
    say()
    big = [c for c in clusters if c.startswith("quest")]
    checks = []
    quest_gate = []  # (得点条件, 未採点条件) を局ごとに
    for c in big:
        # **対比較で見る**（条件つきの片側平均は未採点への逃避で母集団が変わる）
        r = paired[c]
        if r is None:
            checks.append(False)
            quest_gate.append((False, True))
            say(f"- {c} 完全 holdout: 両腕とも採点済みの決定状態が無い → **否**")
            continue
        d_top1 = r["top1_human"][1] - r["top1_human"][0]
        b0, b1 = r["top1_bad"]
        rel_bad = (b0 - b1) / b0 if b0 > 0 else 0.0
        ok = d_top1 >= 0.5 or rel_bad >= 0.25
        # **その局の未採点率も見る**（PR #25 レビュー指摘 P1）。macro の関門だけだと、
        # この局で逃避が増えても他クラスタで減れば相殺されて通ってしまう。
        # 逃避した分は対比較の部分集合から抜けているので、この局の得点差は
        # 偏った母集団の上の値になる
        du = model_res[c].get("top1_unscored", 0.0) - base_res[c].get("top1_unscored", 0.0)
        ok_u = du <= UNSCORED_TOL
        checks.append(ok and ok_u)
        quest_gate.append((ok, ok_u))
        say(f"- {c} 完全 holdout（対 {r['paired_states']}/{r['states']} 状態）: "
            f"top-1 得点 {d_top1:+.3f}（要 +0.5）/ "
            f"0〜2点 top-1 率の相対減 {rel_bad:+.1%}（要 25%）→ **{'合' if ok else '否'}**"
            f"／この局の未採点率 {du:+.3f}（許容 +{UNSCORED_TOL:g}）"
            f"→ **{'合' if ok_u else '未確定'}**")
        if ok and not ok_u:
            say(f"  - 得点条件は通っているが、この局で逃避が増えているので"
                f"**{c} は incomplete**（追加採点まで確定しない）")
    d_conc = macro(model_res, "concordance") - macro(base_res, "concordance")
    ok_conc = d_conc >= 0.05
    checks.append(ok_conc)
    say(f"- macro pairwise concordance {d_conc:+.3f}（要 +0.05）→ **{'合' if ok_conc else '否'}**")
    small = [c for c in clusters if not c.startswith("quest")]
    worst, worst_c = 0.0, None
    for c in small:
        r = paired[c]
        if r is None:
            continue
        d = r["top1_human"][1] - r["top1_human"][0]
        if d < worst:
            worst, worst_c = d, c
    ok_small = worst > -0.5
    checks.append(ok_small)
    say(f"- 小さい eval 群（被王手・反則後）の最悪 top-1 差 {worst:+.3f}"
        f"{f'（{worst_c}）' if worst_c else ''}（要 > −0.5）→ **{'合' if ok_small else '否'}**")
    signs = [model_res[c].get("concordance", 0) - base_res[c].get("concordance", 0) for c in clusters]
    # **符号反転は中止条件そのもの**なので合否へ入れる（PR #25 レビュー指摘）。
    # 表示だけにしておくと、中止条件に該当していても他が通れば「合格」と出てしまう
    ok_sign = all(s >= 0 for s in signs) or all(s <= 0 for s in signs)
    checks.append(ok_sign)
    say(f"- fold 間で concordance 差の符号が反転しない: "
        f"{'一貫' if ok_sign else '**反転あり**'}（{', '.join(f'{s:+.3f}' for s in signs)}）"
        f"→ **{'合' if ok_sign else '否'}**")
    # **未採点への逃避は合否へ入れる**（PR #25 レビュー指摘 P1）。
    # 対比較は「両腕とも採点済み」の部分集合しか見ないので、逃避した分は
    # 測れていない = 他の条件が通っても P0 は**確定しない**
    d_unscored = macro(model_res, "top1_unscored") - macro(base_res, "top1_unscored")
    ok_unscored = d_unscored <= UNSCORED_TOL
    checks.append(ok_unscored)
    say(f"- 未採点手を選んだ率 {macro(base_res, 'top1_unscored'):.3f} → "
        f"{macro(model_res, 'top1_unscored'):.3f}（{d_unscored:+.3f}、許容 +{UNSCORED_TOL:g}）"
        f"→ **{'合' if ok_unscored else '未確定'}**")
    if not ok_unscored:
        say("  - 対比較は両腕とも採点済みの決定状態しか見ないので、**逃避した分は"
            "測れていない**。`append_unscored.py` で選ばれた未収載手を追記し、"
            "採点して `sync_eval.py` を掛けるまで得点差は確定しない")
    say()
    # 「得点・順位の条件は全部通っているが、未採点への逃避だけが引っかかる」
    # ときだけ incomplete（追加採点をすれば決まる）。それ以外は不合格
    scored_ok = all(
        [ok for ok, _ in quest_gate]  # 大きい quest の得点条件
        + [ok_conc, ok_small, ok_sign]
    )
    unscored_ok = all([ok_u for _, ok_u in quest_gate] + [ok_unscored])
    if scored_ok and unscored_ok:
        verdict = "P0 合格（P1 へ）"
    elif scored_ok:
        verdict = "**判定不能（incomplete）**: 未採点手への逃避を解消・追加採点してから再判定する"
    else:
        verdict = "P0 不合格（runtime 実装をしない）"
    say(f"**判定: {verdict}**")

    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n-> {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
