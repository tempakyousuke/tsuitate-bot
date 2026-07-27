#!/usr/bin/env python3
"""現行の estimator を凍結版 src/frozen/estimator_vN.rs として書き出す。

凍結版は「その時点の estimator の挙動」を丸ごと固定するための自己完結コピーで、
アリーナのガントレット基準になる（CLAUDE.md「強さの検証」）。ルールエンジン
（shogi.rs / board.rs）と観測（observation.rs）は共有し続けるので、
ここでまとめるのは estimator.rs / check.rs / strategy.rs の3つだけ。

使い方:
    python3 scripts/freeze_estimator.py 12 > src/frozen/estimator_v12.rs

やること:
- 3ファイルを連結し、モジュールドキュメント（//!）と行頭の `use` を落とす
  （import は凍結版のヘッダを1つだけ持つ。関数内の `use` はインデントが
  あるので残る）
- `EstimatorStrategy` → `EstimatorVN`、`fn name()` の戻り値 → "estimator_vN"
- **運用ノブ（TSUITATE_* の env）は既定値に固定する**。凍結版が候補側の
  w スイープに反応すると比較が壊れるため（2026-07-26 に判明した罠。
  TSUITATE_THINK_BUDGET_MS だけは既存の凍結版と揃えて残す）
"""

import re
import sys

SRC = ["src/estimator.rs", "src/check.rs", "src/strategy.rs"]

HEADER = """//! estimator の凍結版 v{n}（{date} 凍結）。
//!
//! {summary}
//!
//! 凍結後は編集しない（シード注入等の挙動を変えない追加のみ許容）。
//!
//! このファイルは `scripts/freeze_estimator.py` が
//! estimator.rs / check.rs / strategy.rs から生成したもの。
#![allow(dead_code)]

use std::collections::{{HashMap, HashSet, VecDeque}};
use std::time::{{Duration, Instant}};

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::board::{{
    Coord, Promotion, dead_end_rank, defend_targets, drop_targets, make_usi_drop, make_usi_move,
    make_usi_square, move_targets, parse_usi_square, promotion_choice,
}};
use crate::likelihood::{{FITTED_THETA, ParticleCtx, particle_features, particle_log_weight}};
use crate::model::GameModel;
use crate::observation::{{Observation, ObservationLog}};
use crate::opening::OpeningBook;
use crate::protocol::{{Color, PlayerView, Role, VisiblePiece}};
use crate::shogi::{{
    Piece, Position, ShogiMove, parse_usi, piece_value, promote_role, unpromote_role,
}};
use crate::strategy::Strategy;
"""

# 凍結版に含めない最上位アイテム（現行 src/ 側と共有する / 重複定義になるもの）。
# 各エントリはその行から、列0の閉じ括弧までを落とす
DROP_ITEMS = [
    "pub trait Strategy {",
    "pub fn prewarm_strategy(",
    "pub fn prewarm_strategy_with_budget(",
    "pub fn make_seeded(",
    "pub fn make(",
    "pub struct Heuristic;",
    "impl Strategy for Heuristic {",
    "pub fn choose_move(",
]

# 凍結版では既定値へ固定する運用ノブ（env を読ませない）。
# 値は「その env を設定していないときの挙動」と一致させること
PINNED = {
    "fn eval_taint_fallback() -> bool": "false",
    "fn v1_pressure_multiplicity() -> bool": "false",
    "fn v1_defended_by_count() -> bool": "false",
    "fn blind_recapture_w() -> f64": "BLIND_RECAPTURE_W",
}


def strip_file(path: str) -> str:
    """モジュールドキュメント・行頭の use・テストモジュールを落とす。

    テストは3ファイルとも `mod tests` という同じ名前なので、連結すると
    衝突する。凍結版は挙動を固定するためのコピーでテスト対象ではないので落とす
    （テストは現行の src/ 側に残る）
    """
    lines = open(path, encoding="utf-8").read().splitlines()
    out = []
    skipping_tests = False
    skipping_use = False
    skipping_item = False
    for line in lines:
        if skipping_tests:
            continue
        if skipping_item:
            # 列0の `}` でそのアイテムの終わり（1行アイテムは即座に抜ける）
            if line == "}" or line == "};":
                skipping_item = False
            continue
        if any(line.startswith(item) for item in DROP_ITEMS):
            # 直前のドキュメントコメントも一緒に落とす
            while out and (out[-1].startswith("///") or out[-1].startswith("#[")):
                out.pop()
            if not line.rstrip().endswith(";"):
                skipping_item = True
            continue
        if skipping_use:
            # 複数行の use（`use crate::board::{` … `};`）は閉じるまで読み飛ばす
            if line.rstrip().endswith(";"):
                skipping_use = False
            continue
        if re.match(r"#\[cfg\(test\)\]", line):
            skipping_tests = True  # 以降はファイル末尾までテスト（3ファイルとも末尾にある）
            continue
        if line.startswith("//!"):
            continue
        if line.startswith("use "):
            if not line.rstrip().endswith(";"):
                skipping_use = True
            continue
        out.append(line)
    return "\n".join(out).strip("\n")


def pin_env_knobs(body: str) -> str:
    """env を読む小さなノブ関数を、既定値を返すだけの関数へ置き換える"""
    for sig, value in PINNED.items():
        pattern = re.compile(
            re.escape(sig) + r" \{.*?\n\}\n", re.DOTALL
        )
        replacement = f"{sig} {{\n    {value}\n}}\n"
        body, n = pattern.subn(replacement, body)
        if n != 1:
            raise SystemExit(f"ノブの置換に失敗（{n}件）: {sig}")
    return body


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    n = sys.argv[1]
    date = sys.argv[2] if len(sys.argv) > 2 else "YYYY-MM-DD"
    summary = sys.argv[3] if len(sys.argv) > 3 else "TODO: 差分の要約を書く"

    body = "\n\n".join(strip_file(p) for p in SRC)
    body = pin_env_knobs(body)
    # 診断フック（発火率カウンタ）は凍結版では動かさない
    body = re.sub(
        r" *if crate::hits::enabled\(\) \{.*?\n *\}\n",
        "",
        body,
        flags=re.DOTALL,
    )
    # ランキングの公開は現行 estimator のみ（凍結版は Strategy trait の
    # CandidateScore と型が合わない。CLAUDE.md「凍結版は seed 集計のみ」）
    body = re.sub(
        r" *fn last_ranking\(&self\) -> Option<&\[CandidateScore\]> \{.*?\n    \}\n",
        "",
        body,
        flags=re.DOTALL,
    )
    body = body.replace("EstimatorStrategy", f"EstimatorV{n}")
    body = body.replace('        "estimator"\n', f'        "estimator_v{n}"\n')
    # アリーナの共通乱数法（match_seed）用のコンストラクタ
    body += f"""

impl EstimatorV{n} {{
    /// アリーナの共通乱数法用（挙動は with_params_line_seed と同じ）
    pub fn with_seed(seed: u64) -> Self {{
        EstimatorV{n}::with_params_line_seed(EvalParams::default(), None, Some(seed))
    }}
}}
"""

    print(HEADER.format(n=n, date=date, summary=summary))
    print(body)


if __name__ == "__main__":
    main()
