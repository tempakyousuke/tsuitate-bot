#!/usr/bin/env python3
"""レビュー doc（quest_20260731.md 形式）を採点式評価ファイル（.eval.md）へ変換する。

    python3 scripts/quest_review/md_to_eval.py \
        docs/quest_20260731.md evals/quest_20260731.eval.md

一度きりの移行用。既存の評価語彙を 0〜10 の初期点数へ変換し、コメントは全文
保全する。評価の付いていない候補は `?`（未採点）。ブロック内の自由記述行
（USI を含まない行）もそのまま残す。

語彙 → 初期点数（evals/README.md の規約と一致させること）:
  論外・大悪手 → 0 / 悪手 → 2 / 悪手だが比較的マシ → 4 / なくはない → 5 /
  ありかもしれない・セーフ → 6 / 悪くない → 7 / あり → 8 / 本命 → 10
"""

import re
import sys
import pathlib

SEC = re.compile(r"^## (\d+)手目")
SUB = re.compile(r"^### (\d+)手目（(.+?)の反則後）")
MOVE = re.compile(r"^(\S+?)\((\d[a-i]\d[a-i]\+?|[PLNSGBR]\*\d[a-i])\)\s*(.*)$")


def score_of(text: str) -> str:
    """評価コメントから初期点数を決める。順序が重要（部分文字列の包含関係と
    逆接・否定形の処理。実測で誤変換した3パターンのテストが下にある）"""
    t = text
    if "論外" in t or "大悪手" in t:
        return "0"
    if "比較的マシ" in t or "比較的まし" in t:
        return "4"
    # 「それほど悪くない/悪手ではない」= 悪手だが軽い
    if "それほど悪くない" in t or "それほど悪手ではない" in t:
        return "4"
    # 逆接（「発想は悪くないが…悪手」）は肯定側の判定から除外する。
    # ただし逆接の後に悪手判定が続かない場合は下の 悪くはない → 7 が拾う
    if "悪手" in t:
        t = t.replace("悪くないが", "").replace("悪くはないが", "")
    if "悪手とは言えない" in t or "悪手ではない" in t or "悪い手ではない" in t:
        return "7"
    if "悪くない" in t or "悪くはない" in t:
        return "7"
    if "なくはない" in t:
        return "5"
    if "ありかもしれない" in t or "セーフ" in t:
        return "6"
    if "本命" in t:
        return "10"
    if "悪手" in t:
        return "2"
    if "あまりよくなかった" in t:
        return "4"
    if "あまり推奨しない" in t or "推奨しない" in t:
        return "5"
    if "推奨" in t:
        return "9"
    # 「ありだが〜」は留保つきのあり
    if "ありだが" in t:
        return "6"
    if "自然" in t or "当然" in t:
        return "8"
    if "人間の手と同じ" in t or "人間の手と一致" in t or "実戦の人間の手" in t or "指し手もこれ" in t:
        return "8"
    if "本質は同じ" in t:
        return "8"
    if "無難" in t or "普通" in t:
        return "7"
    if "ないより" in t:
        return "5"
    # 「あり」は「ありかも…」「ありだが」を先に処理した後で判定
    if re.search(r"あり[。、．.\s]|あり$", t):
        return "8"
    return "?"


def _self_test() -> None:
    assert score_of("悪手とは言えない。") == "7"
    assert score_of("悪手。それほど悪くないが、嬉しいことがない") == "4"
    assert score_of("発想は悪くないが、この局面では悪手。") == "2"
    assert score_of("悪手。他の手と比較するとそれほど悪手ではない。") == "4"
    assert score_of("悪手。比較的マシ。") == "4"
    assert score_of("あり。") == "8"
    assert score_of("あり.") == "8"
    assert score_of("ありかもしれない") == "6"
    assert score_of("ありだが、反則確率は高い。") == "6"
    assert score_of("本命。") == "10"
    assert score_of("自然な一手") == "8"
    assert score_of("当然の一手。") == "8"
    assert score_of("6一金に紐をつけつつ玉の逃げ場所が増える無難な手。") == "7"
    assert score_of("あまり推奨しないが、悪くはない") == "7"
    assert score_of("推奨") == "9"
    assert score_of("人間の手と同じ。") == "8"
    assert score_of("悪くはない") == "7"
    assert score_of("") == "?"


_self_test()


def main() -> None:
    src = pathlib.Path(sys.argv[1])
    dst = pathlib.Path(sys.argv[2])
    out: list[str] = [
        f"# eval: {src.stem}",
        "# 点数: 0〜10（10=最善級 8=あり 6=ぎりぎり 4=悪手だがマシ 2=悪手 0=論外）。未採点は ?",
        "",
    ]
    in_block = False
    for line in src.read_text(encoding="utf-8").split("\n"):
        m = SEC.match(line)
        if m:
            out.append("")
            out.append(f"## {m.group(1)}手目")
            in_block = True
            continue
        m = SUB.match(line)
        if m:
            out.append("")
            out.append(f"### {m.group(1)}手目（{m.group(2)}の反則後）")
            in_block = True
            continue
        if not in_block:
            continue
        m = MOVE.match(line.strip())
        if m:
            name, usi, comment = m.group(1), m.group(2), m.group(3).strip()
            out.append(f"{name}({usi}) {score_of(comment)} {comment}".rstrip())
        elif line.strip():
            # ブロック内の自由記述はそのまま保全
            out.append(line.rstrip())
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    scored = sum(
        1 for l in out if MOVE.match(l) and l.split(") ", 1)[1].split(" ")[0] != "?"
    )
    total = sum(1 for l in out if MOVE.match(l))
    print(f"{dst}: 候補 {total} 行（うち採点済み {scored}）")


if __name__ == "__main__":
    main()
