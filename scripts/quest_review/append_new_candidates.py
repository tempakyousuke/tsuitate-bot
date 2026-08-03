#!/usr/bin/env python3
"""実戦レビュー doc へ、新ランキングに現れて doc に無い候補だけを追記する。

対人局のレビュー（docs/quest_20260731.md）は「## N手目」節と、反則を挟んだ
決定点の「### N手目（◯◯の反則後）」小節から成り、各行が候補手1つ。
評価ノブを変えたら候補の顔ぶれも変わるので、**差分だけ**を追記して
ユーザーが新しい手を評価できるようにする。

    # 候補ランキングを2シードぶん生成（反則後ブロックも同時に出る）
    cargo run --release --bin rank_dump -- scenarios/archive/quest_20260731.kif 14 148 7 0 > /tmp/s0.md
    cargo run --release --bin rank_dump -- scenarios/archive/quest_20260731.kif 14 148 7 1 > /tmp/s1.md
    python3 scripts/quest_review/append_new_candidates.py /tmp/s0.md /tmp/s1.md

**冪等**（同じ入力で再実行しても何も足さない）。重複判定は USI と
**日本語表記の正規化**の両方で行う: doc には手書きの USI 無し行
（「4七歩成 悪くない…」）が混在するため。表記ゆれは
桂馬→桂 / 香車→香 / 飛車→飛 / 王→玉、全角数字→半角 を吸収する。
"""

import re
import sys
from collections import OrderedDict

import os

DOC = os.environ.get(
    "QUEST_DOC",
    os.path.join(os.path.dirname(__file__), "..", "..", "docs", "quest_20260731.md"),
)
MARKER = os.environ.get(
    "QUEST_MARKER", "（以下 2026-08-03 の再計測で新しく上位7位内に入った候補）"
)
SEC_RE = re.compile(r"^## (\d+)手目")
SUB_RE = re.compile(r"^### (\d+)手目（(.+?)の反則後）")
USI_RE = re.compile(r"\((\d[a-i]\d[a-i]\+?|[PLNSGBR]\*\d[a-i])\)")
ZEN = str.maketrans("０１２３４５６７８９", "0123456789")


def norm_name(s):
    """行頭の指し手表記を正規化して返す（無ければ None）"""
    s = s.strip().translate(ZEN)
    m = re.match(r"^([0-9][一二三四五六七八九][^\s(（]*)", s)
    if not m:
        return None
    n = m.group(1)
    for a, b in (("桂馬", "桂"), ("香車", "香"), ("飛車", "飛"), ("王", "玉")):
        n = n.replace(a, b)
    return n


def parse_rank(path):
    out = {}
    key = None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        m = SEC_RE.match(line)
        if m:
            key = (int(m.group(1)), None)
            out.setdefault(key, OrderedDict())
            continue
        m = SUB_RE.match(line)
        if m:
            key = (int(m.group(1)), m.group(2))
            out.setdefault(key, OrderedDict())
            continue
        if key is None:
            continue
        u = USI_RE.search(line)
        if u and u.group(1) not in out[key]:
            out[key][u.group(1)] = line.strip()
    return out


def main():
    ranks = [parse_rank(p) for p in sys.argv[1:]]
    lines = open(DOC, encoding="utf-8").read().split("\n")

    blocks = []
    for i, line in enumerate(lines):
        m = SEC_RE.match(line) or SUB_RE.match(line)
        if m:
            if blocks:
                blocks[-1] = (blocks[-1][0], blocks[-1][1], i)
            key = (
                (int(m.group(1)), m.group(2))
                if SUB_RE.match(line)
                else (int(m.group(1)), None)
            )
            blocks.append((key, i, len(lines)))

    inserts = []
    n_blocks = n_moves = 0
    for key, start, end in blocks:
        seen_usi, seen_name = set(), set()
        for line in lines[start + 1 : end]:
            seen_usi.update(USI_RE.findall(line))
            nm = norm_name(line)
            if nm:
                seen_name.add(nm)
        new = OrderedDict()
        for r in ranks:
            for usi, disp in r.get(key, {}).items():
                nm = norm_name(disp)
                if usi in seen_usi or (nm and nm in seen_name) or usi in new:
                    continue
                new[usi] = disp
                if nm:
                    seen_name.add(nm)
        if not new:
            continue
        n_blocks += 1
        n_moves += len(new)
        at = end
        while at > start + 1 and lines[at - 1].strip() == "":
            at -= 1
        body = [f"{d} " for d in new.values()]
        has_marker = any(MARKER in lines[j] for j in range(start, end))
        inserts.append((at, body if has_marker else ["", MARKER] + body))

    for at, block in sorted(inserts, key=lambda x: -x[0]):
        lines[at:at] = block

    open(DOC, "w", encoding="utf-8").write("\n".join(lines))
    print(f"{n_blocks} ブロックへ計 {n_moves} 手を追記")


if __name__ == "__main__":
    main()
