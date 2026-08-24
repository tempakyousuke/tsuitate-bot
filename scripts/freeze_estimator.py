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
- **実行時 env は一切読ませない**（issue #21、v15 以降の規約）。
  現行 estimator はノブを `crate::config::StrategyConfig` から引くので、
  凍結版では次の2つを差し替えれば足りる。
  - アクセサの `crate::config::current(|c| c.<節>.<名>)` を、この凍結ファイル
    自身が持つ `frozen_config()`（`EnvSource::empty()` から解決＝既定値）へ
  - `crate::config::ambient()`（プロセス env 由来）を
    `crate::config::frozen_defaults()`（env を見ない）へ
  共有モジュール（`opening.rs` の定跡パス）も、`Strategy` の入口で設置される
  `self.config` = 既定値を引くので固定される。生成物に `env::var` が残っていたら
  生成時に失敗し、`src/frozen/mod.rs` のテストも落ちる。
  思考予算はプロセス env ではなく `EstimatorVN::with_budget_ms` で明示的に渡す
  （渡さなければ凍結時点の既定 2000ms）
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
use std::sync::Arc;
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
use crate::observation::{{Observation, ObservationLog, stale_king_foul_dests}};
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
    "pub fn honors_config(",
    "pub fn make_seeded_with_config(",
    "pub fn apply_env_param_overrides(",
]

# 凍結版が使う実効設定（env を見ない）。節ごとの Knobs 構造体と `from_source` は
# 本文にコピーされているので、空の EnvSource から解決すれば凍結時点の既定値になる
FROZEN_CONFIG = """

/// **凍結時点の実効設定**（issue #21）。`EnvSource::empty()` から解決するので
/// 実行時のプロセス env・candidate 用ノブのどちらにも反応しない。
struct FrozenKnobs {
    strategy: StrategyKnobs,
    estimator: EstimatorKnobs,
    check: CheckKnobs,
}

fn frozen_config() -> &'static FrozenKnobs {
    static C: std::sync::OnceLock<FrozenKnobs> = std::sync::OnceLock::new();
    C.get_or_init(|| {
        let src = crate::config::EnvSource::empty();
        FrozenKnobs {
            strategy: StrategyKnobs::from_source(&src),
            estimator: EstimatorKnobs::from_source(&src),
            check: CheckKnobs::from_source(&src),
        }
    })
}

/// **凍結時点の定跡パス**（PR #22 レビュー指摘3）。
///
/// 共有 `opening.rs` は設置されている `StrategyConfig` からパスを引くので、
/// `crate::config::StrategyConfig::defaults()` を設置すると
/// **将来この既定を変えたときに凍結版まで追随してしまう**。ここで凍結時点の
/// 値をリテラルとして持ち、それを設置する。
/// 中身（`joseki.json` の内容）は `frozen::SHARED_MODEL_PINS` が content hash で
/// 見張っており、編集すると影響する凍結版を名指しでテストが落ちる。
const FROZEN_JOSEKI_PATH: &str = "{joseki}";

/// この凍結版が共有モジュールへ設置する `StrategyConfig`。
/// 実行時 env は見ず、定跡パスだけ凍結時点の値で上書きする。
fn frozen_strategy_config() -> std::sync::Arc<crate::config::StrategyConfig> {
    static C: std::sync::OnceLock<std::sync::Arc<crate::config::StrategyConfig>> =
        std::sync::OnceLock::new();
    C.get_or_init(|| {
        std::sync::Arc::new(crate::config::StrategyConfig::from_source(
            crate::config::EnvSource::from_pairs([("TSUITATE_JOSEKI", FROZEN_JOSEKI_PATH)]),
        ))
    })
    .clone()
}
"""



def strip_file(path: str) -> str:
    """モジュールドキュメント・行頭の use・テストモジュールを落とす。

    テストは3ファイルとも `mod tests` という同じ名前なので、連結すると
    衝突する。凍結版は挙動を固定するためのコピーでテスト対象ではないので落とす
    （テストは現行の src/ 側に残る）。

    **テストモジュールは1ファイルに複数あり、末尾とは限らない**ので、
    `#[cfg(test)]` から**波括弧の対応が閉じるまで**を落とす（以前は
    最初の1つでファイル末尾まで打ち切っていて、strategy.rs の 2523 行目以降が
    丸ごと欠ける状態だった。issue #21 で発見）
    """
    lines = open(path, encoding="utf-8").read().splitlines()
    out = []
    test_depth = None
    skipping_use = False
    skipping_item = False
    for line in lines:
        if test_depth is not None:
            test_depth += line.count("{") - line.count("}")
            if test_depth <= 0:
                test_depth = None
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
            test_depth = 0  # 次行以降、波括弧が閉じるまでがテストモジュール
            continue
        if line.startswith("//!"):
            continue
        if line.startswith("use "):
            if not line.rstrip().endswith(";"):
                skipping_use = True
            continue
        out.append(line)
    return "\n".join(out).strip("\n")


def freeze_config(body: str) -> str:
    """実効設定を凍結時点の既定値へ固定する（issue #21）。

    節ごとの Knobs 構造体と `from_source` は本文にコピーされているので、
    空の `EnvSource` から解決すれば凍結時点の既定値がそのまま出る。
    """
    body, n_knob = re.subn(
        r"crate::config::current\(\|c\| c\.(strategy|estimator|check)\.([a-z0-9_]+)"
        r"((?:\.clone\(\))?)\)",
        r"frozen_config().\1.\2\3",
        body,
    )
    if n_knob == 0:
        raise SystemExit("ノブのアクセサが見つからない（現行 strategy.rs の構造が変わった？）")
    body, n_amb = re.subn(r"crate::config::ambient\(\)", "frozen_strategy_config()", body)
    if n_amb != 1:
        raise SystemExit(f"ambient() の置換に失敗（{n_amb}件）")
    if "env::var(" in body:
        left = sorted(set(re.findall(r'"(TSUITATE_[A-Z0-9_]*)"', body)))
        raise SystemExit(f"凍結版に env::var が残っています（例: {left[:5]}）")
    return body + FROZEN_CONFIG.replace("{joseki}", frozen_joseki_path())


def frozen_joseki_path() -> str:
    """凍結時点の既定の定跡パス（src/config.rs から読む）。

    生成物へリテラルとして埋めるので、後で既定を変えても既存の凍結版は動かない。
    """
    cfg = open("src/config.rs", encoding="utf-8").read()
    m = re.search(r'\.var\("TSUITATE_JOSEKI"\)\s*\n?\s*\.unwrap_or_else\(\|_\| "([^"]+)"', cfg)
    if not m:
        raise SystemExit("src/config.rs から既定の定跡パスを読めません")
    return m.group(1)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    n = sys.argv[1]
    date = sys.argv[2] if len(sys.argv) > 2 else "YYYY-MM-DD"
    summary = sys.argv[3] if len(sys.argv) > 3 else "TODO: 差分の要約を書く"

    body = "\n\n".join(strip_file(p) for p in SRC)
    body = freeze_config(body)
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

    /// **思考予算を明示して作る**（issue #21）。凍結版はプロセス env を
    /// 読まないので、予算を変えたいときは呼び出し側が明示的に渡す
    pub fn with_budget_ms(seed: u64, budget_ms: u64) -> Self {{
        let mut s = EstimatorV{n}::with_seed(seed);
        s.budget = SearchBudget::from_ms(budget_ms);
        s
    }}
}}
"""

    print(HEADER.format(n=n, date=date, summary=summary))
    print(body)


if __name__ == "__main__":
    main()
