//! ついたて将棋bot のライブラリ部分。
//!
//! 本番bot（`main.rs`）・アリーナ（`bin/arena.rs`）・シナリオ（`bin/scenario`）・
//! webhook アダプタ（`bin/webhook_bot`）など、対局・診断ツールが共有する。

pub mod belief_features;
pub mod belief_nn;
pub mod board;
pub mod bridge;
pub mod check;
pub mod checkpoint;
pub mod client;
pub mod config;
pub mod deduce;
pub mod estimator;
pub mod eval_rank;
pub mod effect_profile;
// 凍結版は生成時点のコピーなので、現行コードから見ると未使用の import や
// 到達しない補助関数が残る（それを直さないのが「凍結後は編集しない」の意味）。
// lint 属性は入れ子のモジュールへ伝播するのでここ1箇所で全版に効く
// （版を足すたびに書き足す必要はない）。ビルドログは現行コードの警告だけになる
#[allow(unused_imports, unused_variables, dead_code)]
pub mod frozen;
pub mod hits;
pub mod kifu;
pub mod king_belief_nn;
pub mod likelihood;
pub mod mate;
pub mod model;
pub mod observation;
pub mod opening;
pub mod opp_move_features;
pub mod opp_move_nn;
pub mod opp_move_nn_v25;
pub mod protocol;
pub mod record;
pub mod scenario_core;
pub mod selfplay;
pub mod shogi;
pub mod strategy;
pub mod truth_replay;
pub mod value_features;
pub mod value_nn;
pub mod value_nn_v22;
pub mod webhook_csa;
pub mod webhook_hmac;
pub mod webhook_protocol;
pub mod webhook_session;
