#!/usr/bin/env python3
"""採点式評価ファイル（evals/*.eval.md）をシナリオの `scores=` / `bad=` へ同期する。

    python3 scripts/quest_review/sync_eval.py [evals/quest_20260731.eval.md]

- `scores=` は採点済み（? 以外）の全候補 `USI:点` を列挙
- `bad=` は score <= BAD_THRESHOLD の手（従来の不合格計との互換表示用）
- 反則後ブロック（`### N手目（…の反則後）`）は FOUL_MAP のシナリオへ対応
- **冪等**。eval に点を書き足すたびに回せばよい
"""

import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCN = ROOT / "scenarios"
BAD_THRESHOLD = 2  # この点以下を bad= に入れる（0=論外, 2=悪手）

SEC = re.compile(r"^## (\d+)手目")
SUB = re.compile(r"^### (\d+)手目（(.+?)の反則後）")
MOVE = re.compile(
    r"^(\S+?)\((\d[a-i]\d[a-i]\+?|[PLNSGBR]\*\d[a-i])\)\s+(\d+|\?)(?:\s+(.*))?$"
)

# 反則後ブロックの見出しキー → シナリオ名（sync_bad_lists.py と同一に保つこと）
FOUL_MAP = {
    (30, "1二飛(5b1b)"): "quest31-m030f1",
    (30, "2一歩打(P*2a)"): "quest31-m030f2",
    (40, "5六銀打(S*5f)"): "quest31-m040f1",
    (41, "2一龍(2d2a)"): "quest31-m041f1",
    (50, "3五玉(4e3e)"): "quest31-m050f1",
    (58, "4七歩打(P*4g)"): "quest31-m058f1",
    (62, "4七歩打(P*4g)"): "quest31-m062f1",
    (66, "4六玉(4e4f)"): "quest31-m066f1",
    (75, "6二銀打(S*6b)"): "quest31-m075f1",
    (90, "7八歩打(P*7h)"): "quest31-m090f1",
    (99, "6五金(6d6e)"): "quest31-m099f1",
    (107, "7三歩打(P*7c)"): "quest31-m107f1",
    (120, "7四桂打(N*7d)"): "quest31-m120f1",
}


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


def main() -> None:
    eval_path = pathlib.Path(
        sys.argv[1] if len(sys.argv) > 1 else ROOT / "evals" / "quest_20260731.eval.md"
    )
    prefix = os.environ.get("EVAL_SCENARIO_PREFIX", "quest31-m")
    blocks = parse_eval(eval_path)
    changed = 0
    for (num, sub), entries in sorted(blocks.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        name = FOUL_MAP.get((num, sub)) if sub else f"{prefix}{num:03d}"
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
