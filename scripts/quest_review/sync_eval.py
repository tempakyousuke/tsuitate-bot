#!/usr/bin/env python3
"""採点式評価ファイル（evals/*.eval.md）をシナリオの `scores=` / `bad=` へ同期する。

    python3 scripts/quest_review/sync_eval.py [evals/quest_20260731.eval.md]

- `scores=` は採点済み（? 以外）の全候補 `USI:点` を列挙
- `bad=` は score <= BAD_THRESHOLD の手（従来の不合格計との互換表示用）
- シナリオ名のプレフィックスは eval の stem から foul_blocks.PREFIX_BY_EVAL で引く
  （quest_20260731 → quest31-m、quest_0809 → quest0809-m。未登録ならエラーで止まる。
  `EVAL_SCENARIO_PREFIX` env で上書き可）
- 反則後ブロック（`### N手目（…の反則後）`）は foul_blocks.FOUL_MAP のシナリオへ対応
- **単一決定点の eval**（eval の stem と同名の `scenarios/<stem>.kif` がある場合、
  例 `evals/arena-check-foul01.eval.md`）: `## N手目` は N == ply+1 のときだけ
  `<stem>` へ、反則後ブロックは `fouls=` の末尾がその USI に一致する
  `<stem>f*.kif` へ対応（foul_blocks の表は使わない）
- **冪等**。eval に点を書き足すたびに回せばよい
"""

import os
import pathlib
import re
import sys

from foul_blocks import PREFIX_BY_EVAL, eval_stem, prefix_for_eval, scenario_for

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCN = ROOT / "scenarios"
BAD_THRESHOLD = 2  # この点以下を bad= に入れる（0=論外, 2=悪手）

SEC = re.compile(r"^## (\d+)手目")
SUB = re.compile(r"^### (\d+)手目（(.+?)の反則後）")
MOVE = re.compile(
    r"^(\S+?)\((\d[a-i]\d[a-i]\+?|[PLNSGBR]\*\d[a-i])\)\s+(\d+|\?)(?:\s+(.*))?$"
)


def parse_eval(path: pathlib.Path):
    """(手目, サブキー|None) -> [(usi, 点|None)]"""
    blocks: dict[tuple[int, str | None], list[tuple[str, int | None]]] = {}
    key = None
    for line in path.read_text(encoding="utf-8").split("\n"):
        m = SEC.match(line)
        if m:
            key = (int(m.group(1)), None)
            blocks.setdefault(key, [])
            continue
        m = SUB.match(line)
        if m:
            key = (int(m.group(1)), m.group(2))
            blocks.setdefault(key, [])
            continue
        if key is None:
            continue
        m = MOVE.match(line.strip())
        if m:
            usi, pt = m.group(2), m.group(3)
            blocks[key].append((usi, None if pt == "?" else int(pt)))
    return blocks


def set_directive(header: str, name: str, value: str | None) -> str:
    """ヘッダ行の `name=...` を差し替え（value が None なら削除、無ければ挿入）"""
    pat = re.compile(rf" {name}=[^\s]+")
    header = pat.sub("", header)
    if value is None:
        return header
    # desc= の前（無ければ末尾）に挿入
    if " desc=" in header:
        return header.replace(" desc=", f" {name}={value} desc=", 1)
    return f"{header} {name}={value}"


def _header_directive(path: pathlib.Path, name: str) -> str | None:
    head = path.read_text(encoding="utf-8").split("\n", 1)[0]
    m = re.search(rf"(?:^|\s){name}=(\S+)", head)
    return m.group(1) if m else None


def single_decision_map(stem: str):
    """単一決定点シナリオ（`scenarios/<stem>.kif` が存在）のブロック→シナリオ対応。
    該当しなければ None を返す（従来の foul_blocks 経路）"""
    base = SCN / f"{stem}.kif"
    if not base.exists():
        return None
    ply = int(_header_directive(base, "ply") or -1)
    # 反則後サブ状態: <stem>f<k>.kif の fouls= 末尾 USI -> 名前
    subs: dict[str, str] = {}
    for p in SCN.glob(f"{stem}f*.kif"):
        fouls = _header_directive(p, "fouls")
        if fouls:
            subs[fouls.split(",")[-1]] = p.stem

    def lookup(key: tuple[int, str | None]) -> str | None:
        num, sub = key
        if num != ply + 1:
            return None
        if sub is None:
            return stem
        m = re.search(r"\(([^)]+)\)", sub)
        return subs.get(m.group(1)) if m else None

    return lookup


def main() -> None:
    eval_path = pathlib.Path(
        sys.argv[1] if len(sys.argv) > 1 else ROOT / "evals" / "quest_20260731.eval.md"
    )
    blocks = parse_eval(eval_path)
    single = single_decision_map(eval_stem(eval_path))
    # プレフィックスは eval の stem から引く（foul_blocks.PREFIX_BY_EVAL）。
    # 未登録の eval を黙って quest31-m へ流すと別棋譜のシナリオを上書きするので止める
    prefix = os.environ.get("EVAL_SCENARIO_PREFIX") or prefix_for_eval(eval_path)
    if single is None and prefix is None:
        sys.exit(
            f"{eval_path.name}: シナリオ名プレフィックスが未登録です。"
            f" scripts/quest_review/foul_blocks.py の PREFIX_BY_EVAL に"
            f" {eval_stem(eval_path)!r} を足してください（登録済み: {sorted(PREFIX_BY_EVAL)}）"
        )
    changed = 0
    for (num, sub), entries in sorted(blocks.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        name = single((num, sub)) if single else scenario_for((num, sub), prefix)
        if name is None:
            continue
        path = SCN / f"{name}.kif"
        if not path.exists():
            continue
        seen: set[str] = set()
        scored = [
            (u, p)
            for u, p in entries
            if p is not None and not (u in seen or seen.add(u))
        ]
        want_scores = ",".join(f"{u}:{p}" for u, p in scored)
        want_bad = ",".join(u for u, p in scored if p <= BAD_THRESHOLD)
        lines = path.read_text(encoding="utf-8").split("\n")
        new_header = set_directive(lines[0], "scores", want_scores or None)
        new_header = set_directive(new_header, "bad", want_bad or None)
        if new_header != lines[0]:
            lines[0] = new_header
            path.write_text("\n".join(lines), encoding="utf-8")
            changed += 1
            print(f"{name}: scores {len(scored)} 件 / bad {want_bad.count(',') + 1 if want_bad else 0} 件")
    print(f"--- {changed} ファイル更新")


if __name__ == "__main__":
    main()
