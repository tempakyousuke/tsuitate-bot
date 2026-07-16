//! ついたて将棋bot のライブラリ部分。
//!
//! main.rs（本番bot）と bin/arena.rs（戦略同士のローカル対戦）が共有する。

pub mod board;
pub mod bridge;
pub mod check;
pub mod client;
pub mod estimator;
pub mod frozen;
pub mod kifu;
pub mod likelihood;
pub mod model;
pub mod observation;
pub mod opening;
pub mod protocol;
pub mod record;
pub mod selfplay;
pub mod shogi;
pub mod strategy;
