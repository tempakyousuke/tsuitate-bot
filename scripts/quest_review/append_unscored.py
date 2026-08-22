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

反則後サブブロック（`### N手目（…の反則後）`）は foul_blocks.FOUL_MAP で対応づける。
シナリオ名プレフィックスは eval の stem から foul_blocks.PREFIX_BY_EVAL で引き、
他の棋譜のシナリオ（quest31 と quest0809 は手目が重なる）の TRIAL 行は無視する。

**単一決定点の eval**（eval の stem と同名の `scenarios/<stem>.kif` がある場合、
例 `evals/arena-check-foul02.eval.md`）: TRIAL 行のうち scenario 名が `<stem>`
（→ `## ply+1手目`）と `<stem>f<k>`（→ `fouls=` 末尾 USI の反則後ブロック）の
ものだけを対応づける。quest31 の同じ手目のブロックへ流れ込む取り違え
（2026-08-20 に実際に起きた）を防ぐため、それ以外の scenario は無視する。
"""

import argparse
import collections
import pathlib
import re
import subprocess

from foul_blocks import PREFIX_BY_EVAL, block_key, eval_stem, prefix_for_eval

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCN = ROOT / "scenarios"


def _header_directive(path, name):
    head = path.read_text(encoding="utf-8").split("\n", 1)[0]
    m = re.search(rf"(?:^|\s){name}=(\S+)", head)
    return m.group(1) if m else None


def single_decision_block_key(stem):
    """単一決定点 eval のブロック対応（scenario 名 -> ブロックキー | None）。
    `scenarios/<stem>.kif` が無ければ None を返す（= 従来の foul_blocks 経路）。
    sync_eval.single_decision_map の逆向き（TRIAL の scenario 名から引く）"""
    base = SCN / f"{stem}.kif"
    if not base.exists():
        return None
    num = int(_header_directive(base, "ply")) + 1
    subs = {}
    for p in SCN.glob(f"{stem}f*.kif"):
        fouls = _header_directive(p, "fouls")
        if fouls:
            subs[p.stem] = fouls.split(",")[-1]

    def lookup(scenario, eval_blocks):
        if scenario == stem:
            return (num, None)
        usi = subs.get(scenario)
        if usi is None:
            return None
        # eval 側のサブブロック見出し（和名(USI)）を USI で探す
        for key in eval_blocks:
            if key[1] is not None and key[0] == num and f"({usi})" in key[1]:
                return key
        return None

    return lookup

SEC = re.compile(r"^## (\d+)手目")
SUB = re.compile(r"^### (\d+)手目（(.+?)の反則後）")
MOVE = re.compile(r"^(\S+?)\((\d[a-i]\d[a-i]\+?|[PLNSGBR]\*\d[a-i])\)\s+(\d+|\?)")

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
    # 単一決定点 eval（stem と同名の kif がある）なら専用の対応。取り違え防止
    single = single_decision_block_key(eval_stem(args.eval_path))
    # プレフィックスは eval の stem から引く。別棋譜の TRIAL 行は block_key が
    # None を返して無視される（quest31 と quest0809 は同じ手目を持つので必須）
    prefix = prefix_for_eval(args.eval_path)
    if single is None and prefix is None:
        raise SystemExit(
            f"{args.eval_path.name}: シナリオ名プレフィックスが未登録です。"
            f" foul_blocks.PREFIX_BY_EVAL に足してください（登録済み: {sorted(PREFIX_BY_EVAL)}）"
        )
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
                key = (
                    single(scenario, blocks.keys())
                    if single
                    else block_key(scenario, prefix)
                )
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
