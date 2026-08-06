#!/usr/bin/env python3
"""実戦レビュー doc の悪手判定を quest31-mNNN.kif の `bad=` へ同期する。

    python3 scripts/quest_review/sync_bad_lists.py

doc の各行は「4七歩成(4f4g+) 悪手。…」の形。評価語の判定:
- 「悪手」「大悪手」「論外」を含む → bad
- 「悪くない」「悪くはない」「悪手ではない」「悪手とは言えない」→ not bad
  （否定形を先に潰してから「悪手」を探す）
反則後ブロック（`### N手目（…の反則後）`）は `fouls=` 付きシナリオへ対応させる
（FOUL_MAP）。**冪等**なので、doc に判定を書き足すたびに回せばよい。

注意: 判定は**候補行に書く**こと（節見出しに書いても拾えない）。bot の候補に
上がらなかった手でも、行として足しておけば bad に入る
（例: 20手目の「5一金(4a5a) 悪手（実戦で人間が指した手。botの候補には
上がっていない）」）。
"""

import os
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOC = pathlib.Path(os.environ.get("QUEST_DOC", ROOT / "docs" / "quest_20260731.md"))
SCN = ROOT / "scenarios"
SEC_RE = re.compile(r"^## (\d+)手目")
SUB_RE = re.compile(r"^### (\d+)手目（(.+?)の反則後）")
USI_RE = re.compile(r"\((\d[a-i]\d[a-i]\+?|[PLNSGBR]\*\d[a-i])\)")

# 反則後ブロックの見出しキー → シナリオ名
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
}


def is_bad(text: str) -> bool:
    """行の評価コメントが『悪手』かどうか（否定形を先に潰す）"""
    t = text
    for ok in ("悪くない", "悪くはない", "悪手ではない", "悪手とは言えない"):
        t = t.replace(ok, "")
    return any(w in t for w in ("悪手", "論外"))


def main() -> None:
    # (手数, サブキー) -> [(usi, bad?), ...]
    blocks: dict[tuple[int, str | None], list[tuple[str, bool]]] = {}
    key: tuple[int, str | None] | None = None
    for line in DOC.read_text(encoding="utf-8").split("\n"):
        m = SEC_RE.match(line)
        if m:
            key = (int(m.group(1)), None)
            blocks.setdefault(key, [])
            continue
        m = SUB_RE.match(line)
        if m:
            key = (int(m.group(1)), m.group(2))
            blocks.setdefault(key, [])
            continue
        if key is None:
            continue
        u = USI_RE.search(line)
        if u:
            blocks[key].append((u.group(1), is_bad(line[u.end() :])))

    changed = 0
    for (num, sub), entries in sorted(blocks.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        name = FOUL_MAP.get((num, sub)) if sub else f"quest31-m{num:03d}"
        if name is None:
            continue
        path = SCN / f"{name}.kif"
        if not path.exists():
            continue
        want = [u for u, bad in entries if bad]
        # 重複除去（順序保存）
        seen: set[str] = set()
        want = [u for u in want if not (u in seen or seen.add(u))]
        lines = path.read_text(encoding="utf-8").split("\n")
        cur_m = re.search(r"bad=([^\s]+)", lines[0])
        cur = cur_m.group(1).split(",") if cur_m else []
        if cur == want:
            continue
        added = [u for u in want if u not in cur]
        removed = [u for u in cur if u not in want]
        new_bad = "bad=" + ",".join(want)
        if cur_m:
            lines[0] = lines[0].replace(f"bad={cur_m.group(1)}", new_bad)
        else:
            # target= の直後に差し込む（無ければ ply= の後ろ）
            lines[0] = re.sub(r"(ply=\d+(?: fouls=[^\s]+)?(?: target=[^\s]+)?)", rf"\1 {new_bad}", lines[0], count=1)
        path.write_text("\n".join(lines), encoding="utf-8")
        changed += 1
        print(f"{name}: {len(cur)} -> {len(want)} 件  追加{added} 削除{removed}")
    print(f"--- {changed} ファイル更新")


if __name__ == "__main__":
    main()
