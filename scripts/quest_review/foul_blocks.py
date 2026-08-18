#!/usr/bin/env python3
"""eval のブロックキーとシナリオ名の対応（quest_review スクリプトの共有部品）。

eval のブロックは `(手目, サブキー)` で表す。サブキーは通常ブロック（`## N手目`）が
None、反則後ブロック（`### N手目（和名(USI)の反則後）`）が見出しの `和名(USI)` 部分。

以前は sync_eval.py と append_unscored.py が互いに逆向きの同じ表を持っていて
「シナリオを増やしたら両方に足すこと」と注記していた。表がずれると
**反則後の採点が反則前のシナリオへ流れ込む**（実際に PR #3 で発生した）ので、
一箇所に集約してある。シナリオを増やすときはここだけに足す。
"""

import re

# (手目, 見出しキー) -> シナリオ名
FOUL_MAP: dict[tuple[int, str], str] = {
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
}

DEFAULT_PREFIX = "quest31-m"

_BY_SCENARIO = {v: k for k, v in FOUL_MAP.items()}


def scenario_for(key: tuple[int, str | None], prefix: str = DEFAULT_PREFIX) -> str | None:
    """ブロックキー -> シナリオ名（対応が無ければ None）"""
    num, sub = key
    if sub is not None:
        return FOUL_MAP.get((num, sub))
    return f"{prefix}{num:03d}"


def block_key(scenario: str, prefix: str = DEFAULT_PREFIX) -> tuple[int, str | None] | None:
    """シナリオ名 -> ブロックキー（対応が無ければ None）"""
    if scenario in _BY_SCENARIO:
        return _BY_SCENARIO[scenario]
    m = re.match(rf"{re.escape(prefix)}(\d+)$", scenario)
    return (int(m.group(1)), None) if m else None
