//! 利き走査の呼び出し回数カウンタ（`--features effect-profile` のときだけ有効）。
//!
//! docs/yaneuraou-lessons.md の E（利きの差分計算）は「まずコストを実測する。
//! 測らずにやらない」が着手条件なので、その測定用。差分更新は
//! バグると静かに壊れるので、投資に見合うかを先に数字で確認する。
//!
//! 既定ビルドでは `bump()` の呼び出し自体が消える（`#[cfg]` で囲んで呼ぶ）ので、
//! 対局・アリーナの速度には一切影響しない。
//! 使い方は `src/bin/effect_cost.rs` を参照。

use std::sync::atomic::{AtomicU64, Ordering};

/// `Position::is_attacked`（フル盤面の逆引き利き判定）の呼び出し回数
pub static IS_ATTACKED: AtomicU64 = AtomicU64::new(0);
/// `board::move_targets`（自駒だけを考慮した利き生成）の呼び出し回数
pub static MOVE_TARGETS: AtomicU64 = AtomicU64::new(0);
/// `Position::legal_moves`（合法手の全生成。粒子の相手手サンプルが使う）の回数
pub static LEGAL_MOVES: AtomicU64 = AtomicU64::new(0);
/// 相手手事前分布NN（`opp_move_nn_forward`）の回数
pub static OPP_MOVE_NN: AtomicU64 = AtomicU64::new(0);
/// valueネット（`value_nn_forward`）の回数
pub static VALUE_NN: AtomicU64 = AtomicU64::new(0);

#[inline]
pub fn bump(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

/// 現在値を読んでゼロに戻す
pub fn take(counter: &AtomicU64) -> u64 {
    counter.swap(0, Ordering::Relaxed)
}
