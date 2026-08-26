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
import hashlib
import json
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
# concordance の合否に要る replicate（別エクスポート）の本数。壁時計予算で粒子数が
# 揺れるため、同じコードでも macro concordance が門（+0.05）と同じ桁だけ動く。
# 1本で `d_conc >= 0.05` を合格にすると、そのノイズで P0 合格を出せてしまう
MIN_REPLICATES = 3
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
    violations = 0
    for g in groups.values():
        g.X = np.array(g.rows, dtype=float)
        g.engine_rank = np.array(g.engine_rank, dtype=float)
        g.engine_score = np.array(g.engine_score, dtype=float)
        g.rows = None
        # **契約の機械検査**（PR #25 レビュー指摘 P1）: `engine_score` の降順は
        # `engine_rank` の昇順と厳密に一致していなければならない。CSV 側で丸めて
        # 偽の同点ができると、baseline の argmax と concordance がそこで狂う
        order = np.argsort(g.engine_rank)
        sc = g.engine_score[order]
        if len(sc) > 1 and not np.all(sc[:-1] > sc[1:]):
            violations += 1
        # 同じ決定状態の中で move_number 等は定数なので、後で分散ゼロ列を落とす
        out.append(g)
    if violations:
        sys.exit(
            f"engine_score の降順が engine_rank と一致しない決定状態が {violations} 件"
            " あります（同点＝精度が落ちている疑い）。exporter が `cand.score` を"
            " 丸めずに出しているか確認してください"
        )
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


def combine_score(gain, p_legal, foul_cost):
    """`strategy::combine_score` と同じ式。

    期待値が負の手を `p_legal` で割り引かない min の形。割り引くと
    「合法確率が低いほどスコアが高い」= わざと反則に寄る手が選ばれる。
    """
    return np.minimum(p_legal * gain, gain) - (1.0 - p_legal) * foul_cost


def p1_composed(g, feat_cols, delta):
    """**issue #24 の P1 が定めた gain 側の統合形**を再現する（レビュー指摘 P1）。

        gain' = gain + delta
        score' = combine_score(gain', p_legal, foul_cost) + 既存の外側補正

    最終 score へ `delta` を直接足すのとは**等価でない**。`combine_score` は
    正の gain を `p_legal` で割り引く非線形式なので、外側へ足すと学習値が
    合法性の割引を迂回してしまう（issue が明示的に禁じている性質）。
    例: gain=2 / p_legal=0.5 / foul_cost=0 / delta=1 なら、現行 1.0・
    P1 案 1.5 に対し、外側加算は 2.0 まで上がる。

    外側補正（`adjust` / `foul_probe`）と丸めなし baseline を保つために、
    差分だけを `engine_score` へ乗せる:
    `engine_score + combine(gain+delta) − combine(gain)`。

    `gain` / `p_legal` / `foul_cost` は特徴量列なので 6 桁に丸めてあるが、
    ここで効くのは**差分**（典型的に 0.1〜8）なので相対誤差は 1e-5 以下。
    `engine_score` の丸めと違い**同点を作らない**（engine_score は全桁のまま
    連続値なので、この誤差で順位が変わるのは最終値が 1e-6 未満しか離れていない
    ときだけ）。
    """
    gain = g.X[:, feat_cols.index("gain")]
    p_legal = g.X[:, feat_cols.index("p_legal")]
    foul_cost = g.X[:, feat_cols.index("foul_cost")]
    return baseline_scores(g) + (
        combine_score(gain + delta, p_legal, foul_cost)
        - combine_score(gain, p_legal, foul_cost)
    )


def stratum_of(g):
    s = []
    s.append("被王手" if g.X[0][FEAT.index("in_check")] > 0 else "通常")
    s.append("反則後" if g.X[0][FEAT.index("fouls_this_turn")] > 0 else "反則前")
    mn = g.X[0][FEAT.index("move_number")]
    s.append("序中盤" if mn < 50 else ("中盤" if mn < 90 else "終盤"))
    s.append("quest" if g.cluster.startswith("quest") else "arena")
    return s


# replicate が満たすべき「同じ実験の別サンプル」の条件（PR #25 レビュー指摘 P1）。
# summary JSON のこれらが全 replicate で一致しなければ止める
EXPERIMENT_KEYS = ("budget_ms", "config_fingerprint", "seeds", "eval_fingerprint")


def experiment_fingerprint(csv_path):
    """CSV の隣の summary JSON から実験条件を読む（無ければ止める）"""
    p = pathlib.Path(csv_path)
    side = p.with_name(p.name.replace(".csv", ".summary.json"))
    if not side.exists():
        sys.exit(
            f"{csv_path}: summary JSON（{side.name}）がありません。"
            " replicate は同じ実験条件でしか混ぜられないので、exporter が出す"
            " summary と対で渡してください"
        )
    js = json.loads(side.read_text(encoding="utf-8"))
    missing = [k for k in EXPERIMENT_KEYS if k not in js]
    if missing:
        sys.exit(f"{side.name}: {', '.join(missing)} がありません（古い exporter の出力）")
    return {k: js[k] for k in EXPERIMENT_KEYS}


def content_hash(csv_path):
    h = hashlib.sha256()
    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def population_hash(groups):
    """`(decision_state, seed, usi, human_score, in_candidates)` の母集団。

    候補生成は自駒だけから決まるので replicate 間で同一になるはず。ここが違う
    replicate は別の母集団なので、macro を混ぜてはいけない。
    """
    h = hashlib.sha256()
    for g in sorted(groups, key=lambda g: (g.state, g.seed)):
        h.update(f"{g.state}|{g.seed}|".encode())
        for usi, sc in sorted(zip(g.usi, g.score)):
            h.update(f"{usi}:{'' if sc is None else int(sc)};".encode())
        for sc in sorted(g.absent):
            h.update(f"absent:{int(sc)};".encode())
    return h.hexdigest()


def fold_fit(groups, feat_cols):
    """leave-one-source-KIF-out（外側）＋ 訓練 fold 内の nested CV（α）を回す。

    replicate（別エクスポート）ごとに呼べるよう、記述的な表の出力から切り離してある。
    戻り値は `(clusters, by_cluster, stats, chosen, base_res, model_res, paired)`。
    """
    clusters = sorted({g.cluster for g in groups})
    by_cluster = {c: [g for g in groups if g.cluster == c] for c in clusters}
    stats = {c: ClusterStats(by_cluster[c], len(feat_cols)) for c in clusters}

    def fitted(train_clusters, alpha):
        keep, scale = train_scale(stats, train_clusters)
        return fit_ridge_from(stats, train_clusters, keep, scale, alpha), keep, scale

    base_res, model_res, chosen = {}, {}, {}
    for held in clusters:
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
        base_res[held] = eval_fold(by_cluster[held], lambda g: baseline_scores(g))
        model_res[held] = eval_fold(by_cluster[held], lambda g: rank_scores(g, beta, keep, scale))
    paired = {}
    for c in clusters:
        _, beta_c, keep_c, scale_c = chosen[c]
        paired[c] = paired_metrics(
            by_cluster[c],
            lambda g: baseline_scores(g),
            lambda g, b=beta_c, k=keep_c, sc=scale_c: rank_scores(g, b, k, sc),
        )
    return clusters, by_cluster, stats, chosen, base_res, model_res, paired


def gate_quantities(clusters, base_res, model_res, paired):
    """合否に使う量だけを replicate 横断で比べられる形にまとめる"""
    big = [c for c in clusters if c.startswith("quest")]
    out = {
        "d_conc": macro(model_res, "concordance") - macro(base_res, "concordance"),
        "d_unscored": macro(model_res, "top1_unscored") - macro(base_res, "top1_unscored"),
        "signs": [
            model_res[c].get("concordance", 0) - base_res[c].get("concordance", 0)
            for c in clusters
        ],
        "quest": {},
        "small_worst": 0.0,
        "small_worst_c": None,
    }
    for c in big:
        r = paired[c]
        if r is None:
            out["quest"][c] = None
            continue
        b0, b1 = r["top1_bad"]
        out["quest"][c] = {
            "d_top1": r["top1_human"][1] - r["top1_human"][0],
            "rel_bad": (b0 - b1) / b0 if b0 > 0 else 0.0,
            "d_unscored": model_res[c].get("top1_unscored", 0.0)
            - base_res[c].get("top1_unscored", 0.0),
            "paired_states": r["paired_states"],
            "states": r["states"],
        }
    for c in clusters:
        if c.startswith("quest") or paired[c] is None:
            continue
        d = paired[c]["top1_human"][1] - paired[c]["top1_human"][0]
        if d < out["small_worst"]:
            out["small_worst"], out["small_worst_c"] = d, c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "csv",
        nargs="+",
        help="bin/export_eval_rank_data の CSV。**複数渡すと replicate として扱う**"
        f"（concordance の合否には {MIN_REPLICATES} 本以上が要る）",
    )
    ap.add_argument("--out", help="markdown レポートの書き出し先")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    groups, feat_cols = load(args.csv[0])
    global FEAT
    FEAT = feat_cols
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    # 十分統計と fold fitting（`fold_fit`。replicate ごとに回せる形に切り出してある）
    clusters, by_cluster, stats, chosen, base_res, model_res, paired = fold_fit(groups, feat_cols)

    n_rows = sum(len(g.usi) for g in groups)
    n_scored = sum(sum(1 for s in g.score if s is not None) for g in groups)
    n_absent = sum(len(g.absent) for g in groups)
    say("# eval 残差ランカー P0: 棋譜外への一般化")
    say()
    say(f"- 入力: `{args.csv[0]}`"
        + (f"（記述的な表はこの1本。replicate 計 {len(args.csv)} 本）" if len(args.csv) > 1 else ""))
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
        order = np.argsort(-baseline_scores(g))
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
    for c in clusters:
        r = paired[c]
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
    say("issue #24 の P1 は **gain 側**へ足す形（学習値が合法性の割引を迂回しないため）:")
    say()
    say("```")
    say(f"gain' = gain + W × cap × tanh((r − r̄) / cap)   (cap={BOUND_CAP:g} 点)")
    say("score' = combine_score(gain', p_legal, foul_cost) + 既存の外側補正")
    say("```")
    say()
    say("最終 score へ直接足すのとは**等価でない**（`combine_score` は正の gain を"
        " `p_legal` で割り引く非線形式）。ここは差分だけを丸めなしの `engine_score` へ"
        "乗せて P1 案を再現する。"
        "W=0 は現行と完全一致。得点は**W=0 と対にした比較**（両腕とも top-1 が"
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
            delta = w * BOUND_CAP * np.tanh((r - r.mean()) / BOUND_CAP)
            return p1_composed(g, feat_cols, delta)

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

    # --- replicate（別エクスポート）の集約。**concordance の合否はここでしか出さない**
    # **同じ実験の独立サンプルだけを本数に数える**（PR #25 レビュー指摘 P1）。
    # 列名とクラスタ名しか見ないと、同じファイルを3回渡すだけで本数関門を通せる
    exp0 = experiment_fingerprint(args.csv[0])
    seen_hashes = {content_hash(args.csv[0]): args.csv[0]}
    pop0 = population_hash(groups)
    reps = [gate_quantities(clusters, base_res, model_res, paired)]
    for extra in args.csv[1:]:
        h = content_hash(extra)
        if h in seen_hashes:
            sys.exit(
                f"{extra}: {seen_hashes[h]} と中身が同一です。"
                " replicate は別々のエクスポートでなければノイズ対策になりません"
            )
        seen_hashes[h] = extra
        exp = experiment_fingerprint(extra)
        diff = [k for k in EXPERIMENT_KEYS if exp[k] != exp0[k]]
        if diff:
            sys.exit(
                f"{extra}: 実験条件が1本目と違います（{', '.join(diff)}）。"
                " 同じコミット・同じ budget/config・同じ seed 数・同じ採点でしか"
                " replicate として混ぜられません"
            )
        g2, f2 = load(extra)
        if f2 != feat_cols:
            sys.exit(f"{extra}: 特徴量の列が1本目と違います（同じ exporter で出すこと）")
        if population_hash(g2) != pop0:
            sys.exit(
                f"{extra}: 決定状態・候補・採点の母集団が1本目と違います"
                "（欠けた決定状態がある / 採点が変わった）。別母集団の macro は混ぜられません"
            )
        c2, by2, _, ch2, b2, m2, p2 = fold_fit(g2, f2)
        if c2 != clusters:
            sys.exit(f"{extra}: クラスタ（元 KIF）が1本目と違います")
        reps.append(gate_quantities(c2, b2, m2, p2))
    say("### replicate（別エクスポート）横断")
    say()
    if len(reps) < MIN_REPLICATES:
        say(f"**replicate {len(reps)} 本**（concordance の合否には {MIN_REPLICATES} 本以上が要る）。"
            "壁時計予算で粒子数が揺れるので、macro concordance は同じコードでも門"
            "（+0.05）と同じ桁だけ動く。**1本では合否を出さない**")
    else:
        say(f"**replicate {len(reps)} 本**。合否に使う量は**すべて平均**するので、"
            "引数の順序を入れ替えても判定は変わらない。")
    say()

    def rep_vals(fn):
        """replicate 横断の値（nan は落とす）"""
        return [v for v in (fn(r) for r in reps) if v is not None and not math.isnan(v)]

    def rep_mean(fn):
        vals = rep_vals(fn)
        return sum(vals) / len(vals) if vals else float("nan")

    big2 = [c for c in clusters if c.startswith("quest")]
    say("| 量 | " + " | ".join(f"#{i + 1}" for i in range(len(reps))) + " | 平均 |")
    say("|---|" + "---:|" * (len(reps) + 1))
    cross = [
        ("macro concordance の差", lambda r: r["d_conc"]),
        ("macro 未採点率の差", lambda r: r["d_unscored"]),
        ("小さい eval 群の最悪 top-1 差", lambda r: r["small_worst"]),
    ]
    for c in big2:
        cross.append(
            (f"{c} の対比較 top-1 差", lambda r, c=c: r["quest"][c]["d_top1"] if r["quest"].get(c) else None)
        )
        cross.append(
            (f"{c} の未採点率の差", lambda r, c=c: r["quest"][c]["d_unscored"] if r["quest"].get(c) else None)
        )
    for label, fn in cross:
        vals = [fn(r) for r in reps]
        say(f"| {label} | "
            + " | ".join("—" if v is None else f"{v:+.3f}" for v in vals)
            + f" | {rep_mean(fn):+.3f} |")
    say()

    # --- P0 の合格条件（**すべて replicate 集約**。引数順に依存させない）
    say("## P0 の暫定合格条件との突き合わせ")
    say()
    say(f"**合否に使う量はすべて {len(reps)} 本の平均**なので、この節は引数の順序に"
        "依存しない（記述的な表だけが1本目のもの）。")
    say()
    quest_gate = []
    for c in big2:
        if any(r["quest"].get(c) is None for r in reps):
            quest_gate.append((False, True))
            say(f"- {c} 完全 holdout: 両腕とも採点済みの決定状態が無い replicate がある → **否**")
            continue
        d_top1 = rep_mean(lambda r, c=c: r["quest"][c]["d_top1"])
        rel_bad = rep_mean(lambda r, c=c: r["quest"][c]["rel_bad"])
        du = rep_mean(lambda r, c=c: r["quest"][c]["d_unscored"])
        ok = d_top1 >= 0.5 or rel_bad >= 0.25
        # **その局の未採点率も見る**（macro の関門だけだと他クラスタで相殺される）
        ok_u = du <= UNSCORED_TOL
        quest_gate.append((ok, ok_u))
        # 状態数も replicate 平均で出す（1本目だけを出すと表示が引数順に依存する）
        ps = rep_mean(lambda r, c=c: float(r["quest"][c]["paired_states"]))
        ts = rep_mean(lambda r, c=c: float(r["quest"][c]["states"]))
        say(f"- {c} 完全 holdout（対 {ps:.0f}/{ts:.0f} 状態）: "
            f"top-1 得点 {d_top1:+.3f}（要 +0.5）/ "
            f"0〜2点 top-1 率の相対減 {rel_bad:+.1%}（要 25%）→ **{'合' if ok else '否'}**"
            f"／この局の未採点率 {du:+.3f}（許容 +{UNSCORED_TOL:g}）"
            f"→ **{'合' if ok_u else '未確定'}**")
        if ok and not ok_u:
            say(f"  - 得点条件は通っているが、この局で逃避が増えているので"
                f"**{c} は incomplete**（追加採点まで確定しない）")

    # **concordance は replicate 平均でしか合否を出さない**。1本での
    # `d_conc >= 0.05` を合格にすると、壁時計予算由来のノイズだけで P0 合格を出せる
    d_concs = rep_vals(lambda r: r["d_conc"])
    mean_conc = sum(d_concs) / len(d_concs)
    if len(reps) >= MIN_REPLICATES:
        conc_state = "合" if mean_conc >= 0.05 else "否"
    else:
        conc_state = "判定不能"
    say(f"- macro pairwise concordance {mean_conc:+.3f}"
        # 値は昇順で出す（引数順で表示が変わると差分検査ができない）
        f"（{len(reps)} 本: {', '.join(f'{v:+.3f}' for v in sorted(d_concs))}、要 +0.05）"
        f"→ **{conc_state}**")
    if conc_state == "判定不能":
        say(f"  - replicate が {MIN_REPLICATES} 本に満たないので、この条件では"
            "**合格を出さない**（ノイズが門と同じ桁）")

    worst = rep_mean(lambda r: r["small_worst"])
    worst_cs = sorted({r["small_worst_c"] for r in reps if r["small_worst_c"]})
    ok_small = worst > -0.5
    where = f"（{'、'.join(worst_cs)}）" if worst_cs else ""
    say(f"- 小さい eval 群（被王手・反則後）の最悪 top-1 差 {worst:+.3f}{where}"
        f"（要 > −0.5）→ **{'合' if ok_small else '否'}**")

    # fold ごとの concordance 差も replicate 平均にしてから符号の一貫性を見る
    sign_means = [
        rep_mean(lambda r, i=i: r["signs"][i]) for i in range(len(clusters))
    ]
    ok_sign = all(x >= 0 for x in sign_means) or all(x <= 0 for x in sign_means)
    say(f"- fold 間で concordance 差の符号が反転しない: "
        f"{'一貫' if ok_sign else '**反転あり**'}"
        f"（{', '.join(f'{x:+.3f}' for x in sign_means)}）"
        f"→ **{'合' if ok_sign else '否'}**")

    # **未採点への逃避は合否へ入れる**。対比較は「両腕とも採点済み」の部分集合しか
    # 見ないので、逃避した分は測れていない = 他の条件が通っても P0 は**確定しない**
    d_unscored = rep_mean(lambda r: r["d_unscored"])
    ok_unscored = d_unscored <= UNSCORED_TOL
    say(f"- 未採点手を選んだ率の差 {d_unscored:+.3f}（許容 +{UNSCORED_TOL:g}）"
        f"→ **{'合' if ok_unscored else '未確定'}**")
    if not ok_unscored:
        say("  - 対比較は両腕とも採点済みの決定状態しか見ないので、**逃避した分は"
            "測れていない**。`append_unscored.py` で選ばれた未収載手を追記し、"
            "採点して `sync_eval.py` を掛けるまで得点差は確定しない")
    say()

    # 得点・順位の条件が全部「合」で、未確定（未採点への逃避 / replicate 不足）
    # だけが残るときは incomplete。それ以外は不合格
    scored_ok = all([ok for ok, _ in quest_gate] + [ok_small, ok_sign]) and conc_state != "否"
    pending = conc_state == "判定不能" or not ok_unscored or not all(u for _, u in quest_gate)
    if scored_ok and not pending:
        verdict = "P0 合格（P1 へ）"
    elif scored_ok:
        verdict = ("**判定不能（incomplete）**: 未採点手への逃避の解消・追加採点、"
                   f"および replicate {MIN_REPLICATES} 本以上での再計測が要る")
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
