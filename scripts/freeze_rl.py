#!/usr/bin/env python3
"""方策ネット（DeepNash 路線の R-NaD）の凍結版を生成する。

    python3 scripts/freeze_rl.py <N> <日付> "<要約>" models/rl_vN.bin > src/frozen/rl_vN.rs

凍結版は **推論の一式を固定コピーとして持つ**（estimator の凍結と同じ考え方）:
- `src/rl/action.rs`（行動の符号化・マスク）・`src/rl/encode.rs`（観測のテンソル化）・
  `src/rl/policy_net.rs`（手書きの推論）をテストを落として部分モジュールへ入れる。
  以後 `src/rl/` を改良しても凍結版の挙動は変わらない
- テンソル化が使う手数・反則の上限（`selfplay::MAX_PLIES` / `MAX_FOULS`）は凍結時点の値で埋め込む
  （上限を変えると入力の正規化が変わるため）
- 重みは `include_bytes!` でバイナリに埋め込み、sha256 をテストで検査する
- サンプリング（softmax から seed つきで引く）は `src/rl/policy.rs` の `rl_policy:` と同じ手順
  （マスクは凍結時点の `action::legal_mask`。rl_v21 以前は旧マスク＝`rl_policy_basic:` と同じ）
- 共有のまま使うのはルールエンジン（board / shogi）・観測（observation）・自駒の再構成（model）。
  model は `frozen::SHARED_MODEL_PINS` で pin する（変えるとテンソルが変わるため）
- 実行時 env は読まない（`frozen::HERMETIC_FROM` 以降の規約）

同一性の確認: この スクリプトを再実行して凍結ファイルとの diff が無いこと、CI で
「元の重みの `rl_policy:release:...` vs `rl_vN`」が 50%±10 に入ること。
"""

import hashlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def body(path: str) -> str:
    src = (ROOT / path).read_text()
    # テストは末尾の #[cfg(test)] mod tests { ... } だけ（落とす）
    i = src.find("\n#[cfg(test)]\nmod tests")
    if i >= 0:
        src = src[:i] + "\n"
    # モジュール先頭の //! doc は部分モジュールでは // にする（内側の doc として残す）
    src = re.sub(r"^//!", "//", src, flags=re.M)
    src = src.replace("crate::rl::action::", "super::action::")
    src = src.replace("crate::rl::encode::", "super::encode::")
    src = src.replace("crate::rl::policy_net::", "super::policy_net::")
    return src


def indent(src: str) -> str:
    return "".join(("    " + line if line.strip() else line) for line in src.splitlines(True))


def main() -> None:
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    n, date, summary, weights = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    data = (ROOT / weights).read_bytes()
    sha = hashlib.sha256(data).hexdigest()

    selfplay = (ROOT / "src/selfplay.rs").read_text()
    max_fouls = re.search(r"pub const MAX_FOULS: u32 = (\d+);", selfplay).group(1)
    max_plies = re.search(r"pub const MAX_PLIES: u32 = (\d+);", selfplay).group(1)

    action = body("src/rl/action.rs")
    encode = body("src/rl/encode.rs").replace(
        "use crate::selfplay::{MAX_FOULS, MAX_PLIES};",
        f"// 凍結時点の手数・反則の上限（入力の正規化に使う）\n"
        f"const MAX_FOULS: u32 = {max_fouls};\nconst MAX_PLIES: u32 = {max_plies};",
    )
    policy_net = body("src/rl/policy_net.rs")
    for name, src in (("action", action), ("encode", encode), ("policy_net", policy_net)):
        if "env::var(" in src:
            sys.exit(f"{name} が実行時 env を読んでいる（凍結版は読まない）")
        if "crate::rl::" in src or "crate::selfplay::" in src:
            sys.exit(f"{name} に凍結されない依存が残っている")

    rel_weights = "../../" + weights
    print(f"""//! **凍結版 v{n}**（{date}）: 方策ネット（DeepNash 路線の R-NaD）。{summary}
//!
//! `scripts/freeze_rl.py` が生成した。**編集しない**（改善は `src/rl/` で行う）。
//! 推論の一式（行動の符号化・観測のテンソル化・手書きの推論）は凍結時点の固定コピーで、
//! 重みは `{weights}` をバイナリに埋め込む（sha256 `{sha}`）。
//! 探索なしで1手 約10ms。思考予算・実行時 env は持たない。

#![allow(clippy::all)]

use std::collections::HashSet;
use std::sync::OnceLock;

use rand::{{Rng, SeedableRng, rngs::StdRng}};

use crate::observation::ObservationLog;
use crate::protocol::PlayerView;
use crate::strategy::Strategy;

/// 埋め込んだ重み
const WEIGHTS: &[u8] = include_bytes!("{rel_weights}");
/// 埋め込んだ重みの sha256（凍結時点の値。テストで検査する）
pub const WEIGHTS_SHA256: &str = "{sha}";

fn net() -> &'static policy_net::PolicyNet {{
    static NET: OnceLock<policy_net::PolicyNet> = OnceLock::new();
    NET.get_or_init(|| policy_net::PolicyNet::from_bytes(WEIGHTS).expect("凍結版の重みが読めない"))
}}

/// 凍結版 v{n} の戦略（`strategy::make("rl_v{n}")`）。softmax からサンプリングする
pub struct RlV{n} {{
    rng: StdRng,
    last: Option<serde_json::Value>,
}}

impl RlV{n} {{
    pub fn new() -> Self {{
        Self {{ rng: StdRng::from_rng(&mut rand::rng()), last: None }}
    }}

    pub fn with_seed(seed: u64) -> Self {{
        Self {{ rng: StdRng::seed_from_u64(seed), last: None }}
    }}
}}

impl Default for RlV{n} {{
    fn default() -> Self {{
        Self::new()
    }}
}}

impl Strategy for RlV{n} {{
    fn choose(
        &mut self,
        view: &PlayerView,
        log: &ObservationLog,
        foul_tried: &HashSet<String>,
    ) -> Option<String> {{
        // src/rl/policy.rs の rl_policy（サンプリング）と同じ手順
        let mask = action::legal_mask(view, log, foul_tried);
        let legal: Vec<usize> = (0..mask.len()).filter(|&a| mask[a]).collect();
        if legal.is_empty() {{
            return None;
        }}
        let (logits, value) = net().forward(&encode::encode(view, log, foul_tried));
        let max = legal.iter().map(|&a| logits[a]).fold(f32::MIN, f32::max);
        let weights: Vec<f64> = legal.iter().map(|&a| f64::from(logits[a] - max).exp()).collect();
        let total: f64 = weights.iter().sum();
        let mut r = self.rng.random::<f64>() * total;
        let mut pick = *legal.last().unwrap();
        for (&a, w) in legal.iter().zip(&weights) {{
            if r < *w {{
                pick = a;
                break;
            }}
            r -= w;
        }}
        let prob = weights[legal.iter().position(|&a| a == pick).unwrap()] / total;
        self.last = Some(serde_json::json!({{ "p": prob, "value": value }}));
        action::decode_usi(pick, view.your_color)
    }}

    fn name(&self) -> &'static str {{
        "rl_v{n}"
    }}

    fn debug_state(&self) -> Option<serde_json::Value> {{
        self.last.clone()
    }}
}}

/// 行動の符号化とマスク（`src/rl/action.rs` の凍結時点の固定コピー）
pub mod action {{
{indent(action)}}}

/// 観測のテンソル化（`src/rl/encode.rs` の凍結時点の固定コピー）
pub mod encode {{
{indent(encode)}}}

/// 手書きの推論（`src/rl/policy_net.rs` の凍結時点の固定コピー）
pub mod policy_net {{
{indent(policy_net)}}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn 埋め込んだ重みは凍結時点のもの() {{
        use sha2::{{Digest, Sha256}};
        let got: String = Sha256::digest(WEIGHTS).iter().map(|b| format!("{{b:02x}}")).collect();
        assert_eq!(got, WEIGHTS_SHA256);
        assert!(net().forward(&vec![0.0; encode::OBS_LEN]).0.len() == action::NUM_ACTIONS);
    }}
}}""")


if __name__ == "__main__":
    main()
