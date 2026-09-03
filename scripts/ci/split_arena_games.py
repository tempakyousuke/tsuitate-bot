#!/usr/bin/env python3
"""arena.yml のシャード分割: games を shards 個へ**合計が正確に games** になるよう割る。

従来の `ceil(games/shards)` を偶数化して全シャードへ配る方式は、
games=600 / shards=8 で 76×8 = **608局** を走らせていた（PR #41 レビュー6巡目
[P1]。issue #40 の held-out は各600局が事前登録で、合算器 arena-balance は
600ペアちょうどでなければ判定不能にする = 手順どおり回しても通過が出ない）。

規約:
- **各シャードは偶数局**（先後交代なので、奇数だとシャード内で先後が偏る）
- **合計は正確に games**。games が奇数のときだけ +1（全体の先後均衡）
- シャード間の差は最大2局（先頭側が2局多い）
- どのシャードも 2局未満にはしない（0局のシャードは games.jsonl を出さず、
  合算器の shard 完全性検査と矛盾する）

使い方:
  split_arena_games.py --self-test        # 既知ケースの検査（CI が毎回先に回す）
  split_arena_games.py <games> <shards>   # JSON 配列を stdout へ（例 [76,76,76,76,74,74,74,74]）
"""

import json
import sys


def split(games: int, shards: int) -> list[int]:
    if games <= 0 or shards <= 0:
        raise ValueError(f"games={games} / shards={shards} は正の整数で指定してください")
    total = games + (games % 2)
    pairs = total // 2
    base, rem = divmod(pairs, shards)
    if base == 0:
        raise ValueError(
            f"games={games} が shards={shards} に対して少なすぎます（1シャード2局未満になる）"
        )
    out = [(base + 1) * 2 if i < rem else base * 2 for i in range(shards)]
    # 事後条件（先後数の検査を含む: 各シャード偶数 ⟺ シャード内の先手番 == 後手番）
    assert sum(out) == total, (out, total)
    assert all(x % 2 == 0 and x >= 2 for x in out), out
    return out


def self_test() -> None:
    # 事前登録の held-out（各600局・8シャード）が**正確に600局**になること
    assert split(600, 8) == [76, 76, 76, 76, 74, 74, 74, 74]
    # 従来の慣用ケース: 104/4 は割り切れる
    assert split(104, 4) == [26, 26, 26, 26]
    # 100/4 は従来 26×4=104 に膨らんでいた。正確に100へ
    assert split(100, 4) == [26, 26, 24, 24]
    # 奇数 games は全体だけ +1（先後均衡）
    assert sum(split(99, 4)) == 100
    # 1シャード
    assert split(200, 1) == [200]
    # 少なすぎる分割は拒否
    for games, shards in [(2, 8), (0, 4), (10, -1)]:
        try:
            split(games, shards)
        except ValueError:
            continue
        raise AssertionError(f"split({games}, {shards}) は拒否されるべき")
    print("split_arena_games self-test OK", file=sys.stderr)


def main() -> int:
    args = sys.argv[1:]
    if args == ["--self-test"]:
        self_test()
        return 0
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        out = split(int(args[0]), int(args[1]))
    except ValueError as e:
        print(f"ERR: {e}", file=sys.stderr)
        return 1
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
