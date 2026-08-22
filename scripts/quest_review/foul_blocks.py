#!/usr/bin/env python3
"""eval のブロックキーとシナリオ名の対応（quest_review スクリプトの共有部品）。

eval のブロックは `(手目, サブキー)` で表す。サブキーは通常ブロック（`## N手目`）が
None、反則後ブロック（`### N手目（和名(USI)の反則後）`）が見出しの `和名(USI)` 部分。

以前は sync_eval.py と append_unscored.py が互いに逆向きの同じ表を持っていて
「シナリオを増やしたら両方に足すこと」と注記していた。表がずれると
**反則後の採点が反則前のシナリオへ流れ込む**（実際に PR #3 で発生した）ので、
一箇所に集約してある。シナリオを増やすときはここだけに足す。

**表は eval（＝棋譜）ごとに分ける**（2026-08-22）。以前は手目だけをキーにした
1枚の表だったので、別棋譜の eval（quest_0809）へ sync_eval を掛けると同じ手目の
quest31-m シナリオを上書きする罠があった。eval の stem からプレフィックスを引き、
プレフィックスごとの表だけを見る。未登録の eval は各スクリプトがエラーで止まる
（黙って quest31 へ流さない）。
"""

import pathlib
import re

# eval の stem（ファイル名から .eval.md を除いたもの） -> シナリオ名プレフィックス
PREFIX_BY_EVAL: dict[str, str] = {
    "quest_20260731": "quest31-m",
    "quest_0809": "quest0809-m",
}

DEFAULT_PREFIX = "quest31-m"

# プレフィックス -> {(手目, 見出しキー) -> シナリオ名}
FOUL_MAP: dict[str, dict[tuple[int, str], str]] = {
    "quest31-m": {
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
        # 148手目の**反則後**（S*7a）は後手の反則10回目 = 即反則負けで到達不能
        # なのでシナリオ化しない（eval のブロックも削除済み）。反則**前**の
        # 148手目は「残り1回で合法な手を選べるか」を測る正当な決定点で、
        # quest31-m148 として計測対象に残す
    },
    # quest_0809（先手=お相手 1815 / 後手=ユーザー 1993）。27〜40手目を
    # 2026-08-22 にシナリオ化。41手目以降は採点校正が進んだら同名規則で追加する
    "quest0809-m": {
        (32, "3一角(1c3a)"): "quest0809-m032f1",
        (36, "2三玉(3b2c)"): "quest0809-m036f1",
    },
}

_BY_SCENARIO: dict[str, dict[str, tuple[int, str]]] = {
    prefix: {v: k for k, v in table.items()} for prefix, table in FOUL_MAP.items()
}


def eval_stem(path) -> str:
    """`evals/quest_0809.eval.md` -> `quest_0809`"""
    return pathlib.Path(path).name.split(".")[0]


def prefix_for_eval(path) -> str | None:
    """eval のパス -> シナリオ名プレフィックス（未登録なら None）"""
    return PREFIX_BY_EVAL.get(eval_stem(path))


def scenario_for(key: tuple[int, str | None], prefix: str = DEFAULT_PREFIX) -> str | None:
    """ブロックキー -> シナリオ名（対応が無ければ None）"""
    num, sub = key
    if sub is not None:
        return FOUL_MAP.get(prefix, {}).get((num, sub))
    return f"{prefix}{num:03d}"


def block_key(scenario: str, prefix: str = DEFAULT_PREFIX) -> tuple[int, str | None] | None:
    """シナリオ名 -> ブロックキー（そのプレフィックスのシナリオでなければ None）"""
    if scenario in _BY_SCENARIO.get(prefix, {}):
        return _BY_SCENARIO[prefix][scenario]
    m = re.match(rf"{re.escape(prefix)}(\d+)$", scenario)
    return (int(m.group(1)), None) if m else None
