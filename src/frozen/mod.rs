//! 凍結した過去バージョンの戦略。
//!
//! アリーナ（bin/arena.rs）のガントレット比較の基準として挙動を固定する。
//! 新戦略の合格条件は凍結版への勝ち越し。**既定の対象は v9 以降**
//! （v1〜v5 は破棄済み。v6〜v8 は勝率が飽和して検出力がないため既定から除外。
//! コードは残してあるので明示指定すれば対戦できる）。
//! 非推移性（新しい版に勝つが一つ前に負ける）を検出するため、対象は複数置く。
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
pub mod estimator_v10;
pub mod estimator_v11;
pub mod estimator_v12;
pub mod estimator_v13;
pub mod estimator_v14;
