//! DeepNash（R-NaD）路線の RL 部品（docs/rl-deepnash-design.md）。
//!
//! - `action`: 行動の符号化（139 × 81）と行動マスク
//! - `encode`: 観測 → 入力テンソル
//! - `env`: RL 環境の1局（自己対局・Rust 戦略を相手にする評価モード）。
//!   並列化と numpy 変換は PyO3 側の `rl-env/` crate
//!
//! どちらも入力は `Strategy::choose` と同じ `(PlayerView, ObservationLog, foul_tried)` で、
//! 学習環境（Python 経由）と arena・本番の方策 Strategy が同じ関数を通る。

pub mod action;
pub mod encode;
pub mod env;
