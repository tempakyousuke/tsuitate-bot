"""tsuitate_rl.VecEnv の通しと速度計測（ランダム方策）。

    VIRTUAL_ENV=~/Develop/tsuitate-nn/.venv maturin develop --release -m rl-env/Cargo.toml
    ~/Develop/tsuitate-nn/.venv/bin/python rl-env/smoke.py

マスク内の行動を一様に選んで回し、形・値域・終局・報酬の零和・評価モードの手番を検査して、
1秒あたりの決定点数を出す（ネットの推論を含まない環境側の上限）。
"""

import collections
import time

import numpy as np

import tsuitate_rl


def random_actions(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # 各行の True の中から一様に1つ（Gumbel で argmax）
    noise = rng.random(mask.shape)
    return np.where(mask, noise, -1.0).argmax(axis=1).astype(np.int64)


def check_selfplay(n: int, steps: int) -> None:
    env = tsuitate_rl.VecEnv(n, seed=1)
    rng = np.random.default_rng(1)
    reasons = collections.Counter()
    for _ in range(steps):
        obs, mask, player = env.observe()
        assert obs.shape == (n, tsuitate_rl.NUM_PLANES, 9, 9) and obs.dtype == np.float32
        assert mask.shape == (n, tsuitate_rl.NUM_ACTIONS) and mask.dtype == np.bool_
        assert np.isfinite(obs).all() and (obs >= 0).all()
        assert mask.any(axis=1).all(), "未終局の局はマスクが空でない"
        assert set(np.unique(player)) <= {0, 1}
        rewards, done = env.step(random_actions(mask, rng))
        assert (rewards.sum(axis=1) == 0).all(), "零和"
        assert (rewards[~done] == 0).all(), "報酬は終局時だけ"
        for f in env.pop_finished():
            reasons[f["reason"]] += 1
    assert sum(reasons.values()) > 0
    print(f"自己対局 {n}局並列 × {steps}手: 終局 {dict(reasons)}")


def check_versus() -> None:
    n = 8
    env = tsuitate_rl.VecEnv(n, opponent="heuristic", seed=2)
    rng = np.random.default_rng(2)
    learner = env.learner_colors()
    assert list(learner) == [0, 1] * (n // 2), "偶数局は学習側が先手"
    finished = []
    learner_return = 0.0
    while len(finished) < 16:
        obs, mask, player = env.observe()
        # 報酬の列を選ぶ色は step の**前**に取る（step 後は差し替わった新しい局の色）
        learner = env.learner_colors()
        assert (player == learner).all(), "評価モードは学習側の手番でだけ止まる"
        rewards, done = env.step(random_actions(mask, rng))
        learner_return += rewards[np.arange(n), learner].sum()
        finished += env.pop_finished()
    wins = sum(1 for f in finished if f["winner"] == f["learner"])
    losses = sum(1 for f in finished if f["winner"] is not None and f["winner"] != f["learner"])
    assert learner_return == wins - losses, "報酬の列選択と終局記録が一致する"
    print(f"評価モード（ランダム vs heuristic）: {len(finished)}局で学習側 {wins}勝")


def check_atomic_step() -> None:
    """1局でも不正な行動があれば、どの局も進まない"""
    n = 4
    env = tsuitate_rl.VecEnv(n, seed=4)
    rng = np.random.default_rng(4)
    obs, mask, player = env.observe()
    for bad in (-1, tsuitate_rl.NUM_ACTIONS, int(np.flatnonzero(~mask[1])[0])):
        actions = random_actions(mask, rng)
        actions[1] = bad
        try:
            env.step(actions)
        except ValueError:
            pass
        else:
            raise AssertionError(f"不正な行動 {bad} が通った")
        obs2, mask2, player2 = env.observe()
        assert (obs2 == obs).all() and (mask2 == mask).all() and (player2 == player).all(), (
            "拒否されたバッチで局が進んだ"
        )


def check_no_auto_reset() -> None:
    """auto_reset=False: 終局した局は止まり（マスク空・行動 -1）、reset() で全局が新しくなる"""
    n = 6
    env = tsuitate_rl.VecEnv(n, seed=5, auto_reset=False)
    rng = np.random.default_rng(5)
    ended = 0
    for _ in range(400):
        alive = env.alive()
        if not alive.any():
            break
        obs, mask, player = env.observe()
        assert (mask.any(axis=1) == alive).all(), "止まった局のマスクは空"
        actions = random_actions(mask, rng)
        actions[~alive] = -1
        _, done = env.step(actions)
        assert not (done & ~alive).any()
        ended += int(done.sum())
        # 止まった局に -1 以外を渡すと拒否（どの局も進まない）
        alive_now = env.alive()
        if (~alive_now).any() and alive_now.any():
            try:
                env.step(np.where(alive_now, random_actions(env.observe()[1], rng), 0))
            except ValueError:
                pass
            else:
                raise AssertionError("止まった局への行動が通った")
    assert ended == n and not env.alive().any(), "全局が1局ずつ終わって止まる"
    assert len(env.pop_finished()) == n
    env.reset()
    assert env.alive().all()


def check_seeded_opponent() -> None:
    """同じ seed・同じ行動列なら、heuristic 相手の対局は同じ観測列になる"""

    def run(seed: int) -> list[np.ndarray]:
        env = tsuitate_rl.VecEnv(4, opponent="heuristic", seed=seed)
        rng = np.random.default_rng(0)
        out = []
        for _ in range(60):
            obs, mask, _ = env.observe()
            out.append(obs.copy())
            env.step(random_actions(mask, rng))
        return out

    a, b, c = run(123), run(123), run(124)
    assert all((x == y).all() for x, y in zip(a, b)), "同じ seed で観測列が分岐した"
    assert any((x != y).any() for x, y in zip(a, c)), "seed を変えても同じ観測列"


def check_guard_bonus() -> None:
    """guard_bonus=λ の報酬は、勝敗の報酬 + λ × 玉の周りの占有率の対局平均（両者それぞれ）"""
    n, lam = 16, 0.5
    plain = tsuitate_rl.VecEnv(n, seed=6, auto_reset=False)
    bonus = tsuitate_rl.VecEnv(n, seed=6, auto_reset=False, guard_bonus=lam)
    rng = np.random.default_rng(6)
    got = np.zeros((n, 2), dtype=np.float32)
    base = np.zeros((n, 2), dtype=np.float32)
    while plain.alive().any():
        _, mask, _ = plain.observe()
        _, mask_b, _ = bonus.observe()
        assert (mask == mask_b).all(), "ボーナスは対局の進行を変えない"
        actions = random_actions(mask, rng)
        actions[~plain.alive()] = -1
        r0, _ = plain.step(actions)
        r1, _ = bonus.step(actions)
        base += r0
        got += r1
    finished = {f["game_no"]: f for f in bonus.pop_finished()}
    assert len(finished) == n
    for i in range(n):
        f = finished[i]  # 初期の n 局は局番号 = 並びの位置
        for key in ("guard_r1", "guard_r2"):
            assert all(0.0 <= v <= 1.0 for v in f[key])
        expected = base[i] + lam * np.array(f["guard_r1"], dtype=np.float32)
        assert np.allclose(got[i], expected, atol=1e-5), (got[i], expected)
    try:
        tsuitate_rl.VecEnv(1, guard_radius=3)
    except ValueError:
        pass
    else:
        raise AssertionError("未対応の半径が通った")
    mean = np.mean([f["guard_r1"] for f in finished.values()])
    print(f"玉の周りのボーナス: 報酬の加算が集計と一致（ランダム方策の8近傍占有率 平均 {mean:.3f}）")


def bench(n: int, steps: int) -> None:
    env = tsuitate_rl.VecEnv(n, seed=3)
    rng = np.random.default_rng(3)
    t_obs = t_act = t_step = 0.0
    for _ in range(steps):
        t0 = time.perf_counter()
        _, mask, _ = env.observe()
        t1 = time.perf_counter()
        actions = random_actions(mask, rng)
        t2 = time.perf_counter()
        env.step(actions)
        t3 = time.perf_counter()
        t_obs += t1 - t0
        t_act += t2 - t1
        t_step += t3 - t2
    total = t_obs + t_step
    print(
        f"n={n}: {n * steps / total:,.0f} 決定点/秒（observe {t_obs / steps * 1e3:.1f}ms・"
        f"step {t_step / steps * 1e3:.1f}ms /回、ランダム選択 {t_act / steps * 1e3:.1f}ms は除外）"
    )


if __name__ == "__main__":
    check_selfplay(64, 400)
    check_versus()
    check_atomic_step()
    check_no_auto_reset()
    check_seeded_opponent()
    check_guard_bonus()
    print("不正行動の一括拒否・seed つき相手の再現性: OK")
    for n in (64, 256, 1024):
        bench(n, 100)
