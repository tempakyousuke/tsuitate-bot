#!/usr/bin/env python3
"""シナリオ実行で選ばれたのに eval に載っていない手を `?` 行として追記する。

    # scenario.yml の artifact（scenario-combined）を展開したディレクトリを渡す
    python3 scripts/quest_review/append_unscored.py \\
        evals/quest_20260731.eval.md 実験名 <artifactディレクトリ...> [--min N]

- 各 artifact ディレクトリ配下の `**/result.txt` の `TRIAL` 行を集計する
  （`scenario suite --tsv` の出力形式）
- eval に無い手を「計N回」つきでブロック末尾へ追記（`--min` 未満は無視。
  既定 2 = 1回だけのロングテールは仮4点の有界ノイズとして追わない）
- 冪等: 既に載っている手はスキップするので、同じ実験を何度流してもよい
- 追記後の手順: ユーザーが採点 → `sync_eval.py` → 同一コミットで
  対照と候補の suite を取り直して確定判定
  （**採点前の suite 差は信じない**。指標穴の実証3例あり）

反則後サブブロック（`### N手目（…の反則後）`）は FOUL_MAP で対応づける。
sync_eval.py と同じ表を持つので、シナリオを増やしたら両方に足すこと。
"""

import argparse
import collections
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]

SEC = re.compile(r"^## (\d+)手目")
SUB = re.compile(r"^### (\d+)手目（(.+?)の反則後）")
MOVE = re.compile(r"^(\S+?)\((\d[a-i]\d[a-i]\+?|[PLNSGBR]\*\d[a-i])\)\s+(\d+|\?)")

# sync_eval.py の FOUL_MAP と同一に保つこと（シナリオ名 → (手目, 見出しキー)）
FOUL_MAP = {
    "quest31-m030f1": (30, "1二飛(5b1b)"),
    "quest31-m030f2": (30, "2一歩打(P*2a)"),
    "quest31-m040f1": (40, "5六銀打(S*5f)"),
    "quest31-m041f1": (41, "2一龍(2d2a)"),
    "quest31-m050f1": (50, "3五玉(4e3e)"),
    "quest31-m058f1": (58, "4七歩打(P*4g)"),
    "quest31-m062f1": (62, "4七歩打(P*4g)"),
    "quest31-m066f1": (66, "4六玉(4e4f)"),
    "quest31-m075f1": (75, "6二銀打(S*6b)"),
    "quest31-m090f1": (90, "7八歩打(P*7h)"),
    "quest31-m099f1": (99, "6五金(6d6e)"),
    "quest31-m107f1": (107, "7三歩打(P*7c)"),
    "quest31-m120f1": (120, "7四桂打(N*7d)"),
}

KANJI = "一二三四五六七八九"
ROLE_JP = {
    "P": "歩", "L": "香", "N": "桂", "S": "銀", "G": "金", "B": "角", "R": "飛",
    "K": "玉", "+P": "と", "+L": "成香", "+N": "成桂", "+S": "成銀",
    "+B": "馬", "+R": "龍",
}


def parse_blocks(eval_path):
    """ブロックキー -> 収載済み USI 集合"""
    blocks = collections.defaultdict(set)
    key = None
    for line in eval_path.read_text(encoding="utf-8").split("\n"):
        m = SEC.match(line)
        if m:
            key = (int(m.group(1)), None)
            continue
        m = SUB.match(line)
        if m:
            key = (int(m.group(1)), m.group(2))
            continue
        m = MOVE.match(line)
        if m and key:
            blocks[key].add(m.group(2))
    return blocks


def block_key(scenario):
    if scenario in FOUL_MAP:
        return FOUL_MAP[scenario]
    m = re.match(r"quest31-m(\d+)$", scenario)
    return (int(m.group(1)), None) if m else None


_board_cache = {}


def board_of(scenario):
    """真実の盤面（着手駒の駒種を知るため。scenario board の出力を読む）"""
    if scenario not in _board_cache:
        out = subprocess.run(
            [str(ROOT / "target/release/scenario"), scenario, "board"],
            capture_output=True, text=True, cwd=ROOT,
        ).stdout
        pieces = {}
        for line in out.splitlines():
            m = re.match(r"^([1-9]): (.*)$", line)
            if not m:
                continue
            rank = int(m.group(1))
            for i in range(9):
                cell = m.group(2)[i * 4:(i + 1) * 4].strip()
                if cell and cell != ".":
                    pieces[(9 - i, rank)] = cell.lstrip("v")
        _board_cache[scenario] = pieces
    return _board_cache[scenario]


def jp_name(scenario, usi, moveno):
    """USI → 和名（成/不成を明示する。棋譜の慣習ではなく一意性を優先）"""
    if "*" in usi:
        role, sq = usi.split("*")
        return f"{sq[0]}{KANJI[ord(sq[1]) - 97]}{ROLE_JP[role]}打"
    ff, fr = int(usi[0]), ord(usi[1]) - 96
    tf, tr = int(usi[2]), ord(usi[3]) - 96
    cell = board_of(scenario).get((ff, fr), "?")
    jp = ROLE_JP.get(cell, cell)
    if usi.endswith("+"):
        return f"{tf}{KANJI[tr - 1]}{jp}成"
    sente = moveno % 2 == 1
    zone = (lambda r: r <= 3) if sente else (lambda r: r >= 7)
    optional = cell in ("P", "L", "N", "S", "B", "R") and (zone(fr) or zone(tr))
    return f"{tf}{KANJI[tr - 1]}{jp}{'不成' if optional else ''}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_path", type=pathlib.Path)
    ap.add_argument("label", help="追記マーカーに書く実験名")
    ap.add_argument("dirs", nargs="+", type=pathlib.Path)
    ap.add_argument("--min", type=int, default=2, help="この回数未満は追記しない")
    args = ap.parse_args()

    blocks = parse_blocks(args.eval_path)
    counts = collections.Counter()
    where = {}
    for d in args.dirs:
        for f in d.glob("**/result.txt"):
            for line in f.read_text(encoding="utf-8").splitlines():
                c = line.split("\t")
                if c[0] != "TRIAL" or len(c) < 4:
                    continue
                scenario, usi = c[1], c[3]
                if usi in ("resign", "foul_limit"):
                    continue
                key = block_key(scenario)
                if key is None or usi in blocks.get(key, set()):
                    continue
                counts[(key, usi)] += 1
                where[(key, usi)] = scenario

    add = collections.defaultdict(list)
    for (key, usi), n in sorted(counts.items(), key=lambda x: (x[0][0][0], x[0][1])):
        if n < args.min:
            continue
        add[key].append(f"{jp_name(where[(key, usi)], usi, key[0])}({usi}) ? 計{n}回")

    mark = f"（以下 {args.label} で選ばれた未収載候補。計N回 = 全実行の合算）"
    lines = args.eval_path.read_text(encoding="utf-8").split("\n")
    out, cur = [], None

    def flush(prev):
        if prev in add and add[prev]:
            while out and out[-1] == "":
                out.pop()
            out.append(mark)
            out.extend(add.pop(prev))
            out.append("")

    for line in lines:
        m, m2 = SEC.match(line), SUB.match(line)
        if m or m2:
            flush(cur)
            cur = (int(m.group(1)), None) if m else (int(m2.group(1)), m2.group(2))
        out.append(line)
    flush(cur)
    args.eval_path.write_text("\n".join(out), encoding="utf-8")

    added = sum(1 for n in counts.values() if n >= args.min)
    tail = sum(1 for n in counts.values() if n < args.min)
    print(f"追記 {added} 件（{args.min}回以上）/ 無視 {tail} 件（{args.min}回未満）")
    if add:
        print(f"警告: 対応ブロックが見つからず未追記のグループ {len(add)}")


if __name__ == "__main__":
    main()
