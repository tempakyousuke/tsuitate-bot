#!/usr/bin/env python3
"""元 Arena 実行の `match_seed_base` を検査する（check-prep.yml の provenance 関門）。

**jq を使わない**のが要点（PR #35 レビュー4巡目 [P2]）: jq の数値演算は
IEEE-754 倍精度なので、seed の型（Rust 側は `u64`）が 2^53 を超えると加算が壊れ、
**正常な run を拒否する**。実測では base=9007199254740993 / shard=1 /
env=9007199254740994 は `bin/arena` の起動時検査を正しく通り summary にも正確に
残るのに、jq の `.match_seed_base + .match_seed_shard` は 9007199254740992 になって
不一致を報告した。標準的な 64-bit 乱数 seed で検証セットへ進めなくなるので、
Python の `json` で**整数のまま**読んで任意精度で比べる（`json` は JSON の整数
リテラルを Python の int にするので桁落ちしない）。

出力は1行:
  - `OK:<base>:<shard 番号を空白区切りで>` — 検査通過
  - `NOBASE`                              — `match_seed_base` を持たない run（古い commit）
  - `ERR:<理由>`                          — 検査失敗

終了コードは常に 0（呼び出し側の shell が接頭辞で分岐する）。想定外の例外だけが
非ゼロで落ちる。
"""

import json
import sys


def verify(rows: list[dict], expect: str, shard_count: int) -> str:
    have = [r for r in rows if r.get("match_seed_base") is not None]
    if not have:
        return "NOBASE"
    bases = sorted({r["match_seed_base"] for r in have})
    if len(bases) != 1:
        return "ERR:シャード間で match_seed_base が食い違います: " + " ".join(
            str(b) for b in bases
        )
    base = bases[0]
    shards = sorted(
        r["match_seed_shard"] for r in have if r.get("match_seed_shard") is not None
    )
    # 記録された shard 番号の集合も 0..n-1 で完備か（artifact 名とは独立の裏取り）
    if shards != list(range(shard_count)):
        got = " ".join(str(x) for x in shards)
        return f"ERR:記録の match_seed_shard が 0..{shard_count - 1} で完備ではありません（{got}）"
    # **base seed で照合する**。記録の `match_seed` は「base + shard」をさらに
    # 基準ごとに XOR した実効値なので、シャードごとに違うし base とも一致しない
    if expect and str(base) != expect:
        return f"ERR:match_seed_base が期待と違います（期待 {expect} / 実際 {base}）"
    # **ラベルと実物の一致**（PR #35 レビュー3巡目 [P2]）。本来の関門は `bin/arena` の
    # 起動時検査（`check_seed_provenance`）で、そこを通った binary なら必ず
    # `match_seed_env == base + shard`。ここは**その検査より前の binary で回した run**
    # （base は残るが未検査）を弾くための裏取りで、XOR の式は複製していない
    bad = [
        f"shard {r['match_seed_shard']}: env={r['match_seed_env']} != "
        f"base={r['match_seed_base']}+shard={r['match_seed_shard']}"
        for r in have
        if r.get("match_seed_env") is not None
        and r.get("match_seed_shard") is not None
        and r["match_seed_env"] != r["match_seed_base"] + r["match_seed_shard"]
    ]
    if bad:
        return "ERR:match_seed_base のラベルが実際の対局 seed と一致しません: " + " / ".join(bad)
    return f"OK:{base}:{' '.join(str(x) for x in shards)}"


def self_test() -> None:
    """`cargo test` に相当する門が無いので、CI が毎回この自己検査を通す。"""
    # 2^53 超（jq なら壊れる。ここが今回の回帰そのもの）
    big = [
        {
            "match_seed_base": 9007199254740993,
            "match_seed_shard": s,
            "match_seed_env": 9007199254740993 + s,
        }
        for s in (0, 1)
    ]
    assert verify(big, "9007199254740993", 2) == "OK:9007199254740993:0 1", verify(
        big, "9007199254740993", 2
    )
    # u64 の上限近く（符号付き 64-bit の shell 演算でも壊れる領域）
    huge_base = 2**64 - 4
    huge = [
        {
            "match_seed_base": huge_base,
            "match_seed_shard": s,
            "match_seed_env": huge_base + s,
        }
        for s in (0, 1)
    ]
    assert verify(huge, str(huge_base), 2).startswith("OK:")
    # ラベルと実物の食い違いは 2^53 超でも捕まる
    bad = [
        {"match_seed_base": 9007199254740993, "match_seed_shard": 0, "match_seed_env": 7}
    ]
    assert verify(bad, "", 1).startswith("ERR:match_seed_base のラベル"), verify(bad, "", 1)
    # 通常の seed
    normal = [
        {"match_seed_base": 20260829, "match_seed_shard": s, "match_seed_env": 20260829 + s}
        for s in (0, 1)
    ]
    assert verify(normal, "20260829", 2) == "OK:20260829:0 1"
    assert verify(normal, "20260828", 2).startswith("ERR:match_seed_base が期待と違います")
    # シャード欠落・base 不一致・base 無し
    assert verify(normal, "", 3).startswith("ERR:記録の match_seed_shard")
    assert verify(
        [
            {"match_seed_base": 1, "match_seed_shard": 0, "match_seed_env": 1},
            {"match_seed_base": 2, "match_seed_shard": 1, "match_seed_env": 3},
        ],
        "",
        2,
    ).startswith("ERR:シャード間で match_seed_base")
    assert verify([{"candidate": "estimator"}], "", 1) == "NOBASE"
    print("self-test ok")


def main() -> None:
    if sys.argv[1:2] == ["--self-test"]:
        self_test()
        return
    path, expect, shard_count = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(verify(rows, expect, shard_count))


if __name__ == "__main__":
    main()
