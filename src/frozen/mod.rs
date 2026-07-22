//! 凍結した過去バージョンの戦略。
//!
//! アリーナ（bin/arena.rs）のガントレット比較の基準として挙動を固定する。
//! 新戦略は「直近の凍結版」だけでなく**過去の凍結版すべて**に勝ち越すことを確認する
//! （v2 には勝つが v1 には負ける、という非推移性を検出するため）。
//!
//! 運用ルール:
//! - 各ファイルは凍結後、ルールエンジンの追随を除いて編集しない
//! - 改善は src/estimator.rs / src/strategy.rs で行い、アリーナで確定したら
//!   その時点のコピーを estimator_vN.rs として追加し strategy::make に登録する
//! - ルールエンジン（shogi.rs / board.rs）と観測（observation.rs）は共有する
//!   （ルールのバグ修正は全バージョンに反映されるべきなので）

pub mod estimator_v6;
pub mod estimator_v7;
pub mod estimator_v8;
pub mod estimator_v9;
