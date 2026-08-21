#!/usr/bin/env python3
"""採点式評価ファイル（evals/*.eval.md）からシナリオ .kif を生成する。

    python3 scripts/quest_review/eval_to_scenarios.py \
        evals/quest_20260731.eval.md scenarios/quest31-m061.kif quest31-m 94 121

    # 最初の1件目（流用元シナリオがまだ無い棋譜）は archive の元棋譜を渡す
    python3 scripts/quest_review/eval_to_scenarios.py \
        evals/quest_0809.eval.md scenarios/archive/quest_0809.kif quest0809-m 27 40

引数: <eval> <本体を流用する既存シナリオ|archive の元棋譜> <プレフィックス> <開始手目> <終了手目>

- 本体（2行目以降の棋譜）は既存シナリオから流用し、ヘッダだけ生成する。
  流用元の1行目が `*scenario` でなければ元棋譜とみなして全文を本体にする
  （`*illegal:` 行もそのまま写る）。出力先は流用元のディレクトリ、
  ただし `archive/` なら その親（= scenarios/）
- target= はブロック内の最高点の手（全て未採点なら target なし）
- 反則後サブブロック（### N手目（和名(USI)の反則後））は fouls= 付きの
  <プレフィックス>NNNf1, f2... として生成する
- 既存ファイルは上書きしない（ヘッダを更新したいときは削除してから再実行）
- scores= / bad= は付けない（sync_eval.py が同期する。FOUL_MAP への
  サブブロック登録を忘れないこと）
"""

import pathlib
import re
import sys

SEC = re.compile(r"^## (\d+)手目(?:（(先手|後手)番）)?")
SUB = re.compile(r"^### (\d+)手目（(.+?\((\S+?)\))の反則後）")
MOVE = re.compile(
    r"^(\S+?)\((\d[a-i]\d[a-i]\+?|[PLNSGBR]\*\d[a-i])\)\s+(\d+|\?)(?:\s+(.*))?$"
)


def main() -> None:
    eval_path, template, prefix, start, end = (
        pathlib.Path(sys.argv[1]),
        pathlib.Path(sys.argv[2]),
        sys.argv[3],
        int(sys.argv[4]),
        int(sys.argv[5]),
    )
    text = template.read_text(encoding="utf-8")
    body = text.split("\n", 1)[1] if text.startswith("*scenario") else text
    scn_dir = template.parent
    if scn_dir.name == "archive":
        scn_dir = scn_dir.parent

    # ブロック収集: (手目, サブ反則USI|None, サブ見出し名) -> [(name, usi, score)]
    blocks: list[tuple[int, str | None, str, list]] = []
    for line in eval_path.read_text(encoding="utf-8").split("\n"):
        m = SEC.match(line)
        if m:
            blocks.append((int(m.group(1)), None, "", []))
            continue
        m = SUB.match(line)
        if m:
            blocks.append((int(m.group(1)), m.group(3), m.group(2), []))
            continue
        if not blocks:
            continue
        m = MOVE.match(line.strip())
        if m:
            blocks[-1][3].append(
                (m.group(1), m.group(2), None if m.group(3) == "?" else int(m.group(3)))
            )

    made = 0
    foul_seq: dict[int, int] = {}
    for num, foul_usi, foul_name, moves in blocks:
        if not (start <= num <= end):
            continue
        if foul_usi is None:
            name = f"{prefix}{num:03d}"
        else:
            foul_seq[num] = foul_seq.get(num, 0) + 1
            name = f"{prefix}{num:03d}f{foul_seq[num]}"
        path = scn_dir / f"{name}.kif"
        if path.exists():
            print(f"{name}: 既存のためスキップ")
            continue
        side = "先手番" if num % 2 == 1 else "後手番"
        scored = [(n, u, s) for n, u, s in moves if s is not None]
        target = max(scored, key=lambda x: x[2]) if scored else None
        # 最高点でも 6 点未満なら「良い手が候補に無い」ブロック = target なし
        # （2点の手を注目手にすると一致率の意味が壊れる）
        if target is not None and target[2] < 6:
            target = None
        parts = [f"*scenario ply={num - 1}"]
        if foul_usi is not None:
            parts.append(f"fouls={foul_usi}")
        if target is not None:
            parts.append(f"target={target[1]}")
        desc = f"{num}手目（{side}"
        if foul_usi is not None:
            desc = f"{num}手目・{foul_name}の反則後（{side}"
        desc += "）。"
        if target is not None:
            desc += f"最高点は{target[0]}（{target[2]}点）。"
        desc += f"採点は evals/{eval_path.name} 参照"
        parts.append(f"desc={desc.replace(' ', '')}")
        path.write_text(" ".join(parts) + "\n" + body, encoding="utf-8")
        made += 1
        print(f"{name} 生成: {' '.join(parts[:3])}")
    print(f"--- {made} ファイル生成")


if __name__ == "__main__":
    main()
