#!/usr/bin/env python3
"""arena / check-prep の match_seed 契約を**任意精度の整数**で扱うための小道具。

用途は2つ:

- `--shard-seed <base> <shard>`（`arena.yml` が使う）: シャードずらし後の
  `ARENA_MATCH_SEED` を計算する。**Bash の `$(( ))` は符号付き 64-bit** なので
  `u64` の上半分で折り返し、`bin/arena` の `env_u64` が「非負整数でない」として
  run を止めてしまう（PR #35 レビュー5巡目 [P2]。実測: base=18446744073709551612 は
  shard 0/1 で `-4` / `-3` になる）。producer / checker / テストの3者が同じ
  `u64` 全域を扱えるように、ここで任意精度で足して範囲を検査する
- 引数3つ（`check-prep.yml` が使う）: 元 Arena 実行の `match_seed_base` を検査する

検査側で**jq を使わない**のが要点（PR #35 レビュー4巡目 [P2]）: jq の数値演算は
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

U64_MAX = 2**64 - 1


def _u64(value: object, what: str) -> int:
    """`u64` として妥当な JSON 整数か。**bool は int の派生なので明示的に弾く**。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what} が整数ではありません: {value!r}")
    if not 0 <= value <= U64_MAX:
        raise ValueError(f"{what} が u64 の範囲外です: {value}")
    return value


def shard_seed(base: int, shard: int) -> int:
    """`arena.yml` が渡す `ARENA_MATCH_SEED`（= base + shard）を任意精度で作る。

    `bin/arena` 側の `check_seed_provenance` は `base.checked_add(shard)` なので、
    **桁あふれは producer 側でも同じく拒否する**（折り返した値を渡して
    「3値は揃っているのに実際の対局条件が違う」を作らない）。
    """
    _u64(base, "match_seed")
    _u64(shard, "shard")
    total = base + shard
    if total > U64_MAX:
        raise ValueError(f"match_seed + shard が u64 の範囲外です: {base} + {shard}")
    return total


def verify(rows: list[dict], expect: str, shard_count: int) -> str:
    have = [r for r in rows if r.get("match_seed_base") is not None]
    if not have:
        return "NOBASE"
    # **base がある run では3値の欠測を黙って飛ばさない**（PR #35 レビュー5巡目 [P2]）。
    # `match_seed_env` が欠けた行を除外していたので、「base と shard は揃っているが
    # 実際の対局 seed との対応を証明できない」入力が OK になっていた。
    # base を持つ行と持たない行が混ざるのも「シャードごとに別の binary で回した」
    # 証拠なので通さない
    if len(have) != len(rows):
        return (
            f"ERR:match_seed_base を持つシャードと持たないシャードが混在しています"
            f"（{len(have)}/{len(rows)}）。別の binary で回したシャードが混ざっています"
        )
    try:
        for r in rows:
            for key in ("match_seed_base", "match_seed_shard", "match_seed_env"):
                _u64(r.get(key), key)
    except ValueError as e:
        return f"ERR:{e}"
    bases = sorted({r["match_seed_base"] for r in have})
    if len(bases) != 1:
        return "ERR:シャード間で match_seed_base が食い違います: " + " ".join(
            str(b) for b in bases
        )
    base = bases[0]
    shards = sorted(r["match_seed_shard"] for r in have)
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
        if r["match_seed_env"] != r["match_seed_base"] + r["match_seed_shard"]
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
    # u64 の上限近く（Bash の `$(( ))` は符号付き 64-bit なのでここで折り返す。
    # **producer 側も同じ範囲を扱えること**を `--shard-seed` の検査で担保する）
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
    for s in (0, 1):
        assert shard_seed(huge_base, s) == huge_base + s
    assert shard_seed(2**63, 1) == 2**63 + 1  # i64::MAX + 1 近辺
    assert shard_seed(20260829, 3) == 20260832
    # 桁あふれ・範囲外・型違いは producer 側でも拒否する
    for bad_args in ((U64_MAX, 1), (U64_MAX + 1, 0), (-1, 0), (0, -1)):
        try:
            shard_seed(*bad_args)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"shard_seed{bad_args} が通ってしまいました")
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
    # **欠測は fail-closed**（PR #35 レビュー5巡目 [P2]）: base と shard が揃っていても
    # `match_seed_env` が無ければ実際の対局 seed との対応を証明できない
    missing_env = [
        {"match_seed_base": 20260829, "match_seed_shard": s} for s in (0, 1)
    ]
    assert verify(missing_env, "20260829", 2).startswith(
        "ERR:match_seed_env が整数ではありません"
    ), verify(missing_env, "20260829", 2)
    null_env = [
        {"match_seed_base": 20260829, "match_seed_shard": s, "match_seed_env": None}
        for s in (0, 1)
    ]
    assert null_env and verify(null_env, "20260829", 2).startswith("ERR:match_seed_env")
    # 型違い（文字列・bool）と u64 範囲外
    assert verify(
        [{"match_seed_base": "20260829", "match_seed_shard": 0, "match_seed_env": 20260829}],
        "",
        1,
    ).startswith("ERR:match_seed_base が整数ではありません")
    assert verify(
        [{"match_seed_base": 20260829, "match_seed_shard": True, "match_seed_env": 20260830}],
        "",
        1,
    ).startswith("ERR:match_seed_shard が整数ではありません")
    assert verify(
        [{"match_seed_base": 2**64, "match_seed_shard": 0, "match_seed_env": 2**64}],
        "",
        1,
    ).startswith("ERR:match_seed_base が u64 の範囲外です")
    # base を持つ行と持たない行の混在（別 binary で回したシャード）
    assert verify(
        [
            {"match_seed_base": 20260829, "match_seed_shard": 0, "match_seed_env": 20260829},
            {"candidate": "estimator"},
        ],
        "",
        1,
    ).startswith("ERR:match_seed_base を持つシャードと持たないシャードが混在")
    print("self-test ok")


def main() -> None:
    if sys.argv[1:2] == ["--self-test"]:
        self_test()
        return
    if sys.argv[1:2] == ["--shard-seed"]:
        # arena.yml 用。**範囲外は非ゼロで落とす**（呼び出し側は素の代入で受けて
        # `set -e` に伝播させること。`export VAR=$(...)` は export の終了状態に
        # なるので失敗を握り潰す）
        try:
            print(shard_seed(int(sys.argv[2]), int(sys.argv[3])))
        except ValueError as e:
            print(f"{e}", file=sys.stderr)
            raise SystemExit(1) from None
        return
    path, expect, shard_count = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(verify(rows, expect, shard_count))


if __name__ == "__main__":
    main()
