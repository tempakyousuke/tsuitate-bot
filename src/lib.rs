//! ついたて将棋bot のライブラリ部分。
//!
//! main.rs（本番bot）と bin/arena.rs（戦略同士のローカル対戦）が共有する。

pub mod board;
pub mod client;
pub mod estimator;
pub mod frozen;
pub mod model;
pub mod observation;
pub mod protocol;
pub mod record;
pub mod shogi;
pub mod strategy;
