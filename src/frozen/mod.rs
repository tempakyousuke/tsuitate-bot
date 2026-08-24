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

/// **hermetic 規約を適用する最初の版**（issue #21）。
///
/// v6〜v14 は凍結時点で `std::env::var("TSUITATE_...")` を持ち込んでおり、
/// 実行時のプロセス env に反応する。**過去の挙動を作り変えない**方針なので
/// 一括編集はしない（当時の計測結果と対応が取れなくなる）。
/// 代わりに、この版以降は `scripts/freeze_estimator.py` が env 読取を落とし、
/// 下のテストが実際に落ちていることを検査する。
pub const HERMETIC_FROM: u32 = 15;

/// 凍結版のソース（**compile-time に埋め込む**）。監査（どの env を読むか・
/// 共有依存が何か）と hermetic ガードが使う。版を足したらここにも足すこと。
pub const SOURCES: &[(u32, &str, &str)] = &[
    (6, "estimator_v6", include_str!("estimator_v6.rs")),
    (7, "estimator_v7", include_str!("estimator_v7.rs")),
    (8, "estimator_v8", include_str!("estimator_v8.rs")),
    (9, "estimator_v9", include_str!("estimator_v9.rs")),
    (10, "estimator_v10", include_str!("estimator_v10.rs")),
    (11, "estimator_v11", include_str!("estimator_v11.rs")),
    (12, "estimator_v12", include_str!("estimator_v12.rs")),
    (13, "estimator_v13", include_str!("estimator_v13.rs")),
    (14, "estimator_v14", include_str!("estimator_v14.rs")),
];

/// 凍結版 `name` が**自分のファイルの中で**読む `TSUITATE_*` の一覧。
/// 共有モジュール経由の読取は含まない（それは呼び出し側が別途走査する）。
///
/// `std::env::var(name)` のように**変数越しに**読む箇所（思考予算）があるので、
/// 呼び出し形ではなく**文字列リテラル**を拾う（安全側 = 多めに出る）。
/// doc コメントの中は backtick で書く規約なので誤検出しない。
pub fn env_keys_in_source(name: &str) -> Vec<String> {
    let Some((_, _, src)) = SOURCES.iter().find(|(_, n, _)| *n == name) else {
        return vec![];
    };
    env_literals(src)
}

/// ソース中の `"TSUITATE_*"` 文字列リテラルを拾う。
fn env_literals(src: &str) -> Vec<String> {
    let mut out: Vec<String> = vec![];
    for (i, _) in src.match_indices("\"TSUITATE_") {
        let rest = &src[i + 1..];
        if let Some(end) = rest.find('"') {
            out.push(rest[..end].to_string());
        }
    }
    out.sort();
    out.dedup();
    out
}

/// **共有モジュール経由で凍結版に効く env**。
///
/// 実装が `crate::config` を経由するので、モジュール本体には文字列リテラルが
/// 無い（`opening.rs` は `config` が解決した `joseki_path` を読むだけ）。
/// 走査では拾えないので明示する。
const SHARED_MODULE_ENV: &[(&str, &[&str])] = &[("opening", &["TSUITATE_JOSEKI"])];

/// 凍結版 `name` が**共有モジュール経由も含めて**読む env の一覧。
///
/// 凍結ファイル自身のリテラルだけを見ると、v12〜v14 が `opening.rs` 経由で
/// 読む `TSUITATE_JOSEKI` を取りこぼす（PR #20 4回目レビューと同じ穴）。
pub fn env_keys_read_by(name: &str) -> Vec<String> {
    let mut out = env_keys_in_source(name);
    for (_, _, module, src) in SHARED_MODEL_PINS {
        if versions_using(module).contains(&name) {
            out.extend(env_literals(src));
        }
    }
    for (module, keys) in SHARED_MODULE_ENV {
        if versions_using(module).contains(&name) {
            out.extend(keys.iter().map(|k| k.to_string()));
        }
    }
    out.sort();
    out.dedup();
    out
}

/// 凍結版の**実効挙動の指紋**（PR #22 レビュー指摘4）。
///
/// 「相手の設定」を現行 `StrategyConfig` の指紋で記録すると、v6〜v14 は
/// 各凍結ファイル内の旧既定値・旧 env 読取規則で動くので**相手名に依らず
/// 同じ値**になり、実効設定を表さない。ここでは名前だけでは表せないものを混ぜる:
///
/// - 凍結版ソースの sha256（その版そのもの）
/// - **その版が読む env の実効値**（共有モジュール経由を含む）
/// - その版が呼ぶ共有モデル・データの pin ハッシュ
///
/// 凍結版でない名前（現行 estimator 系）は `None`。呼び出し側が
/// instance の `StrategyConfig::fingerprint()` を使うこと。
pub fn behavior_fingerprint(
    name: &str,
    env: &std::collections::BTreeMap<String, String>,
) -> Option<String> {
    use sha2::{Digest, Sha256};
    let (_, _, src) = SOURCES.iter().find(|(_, n, _)| *n == name)?;
    let mut h = Sha256::new();
    h.update(name.as_bytes());
    h.update(b"\x00");
    h.update(sha256_hex(src).as_bytes());
    for k in env_keys_read_by(name) {
        h.update(b"\x00");
        h.update(k.as_bytes());
        h.update(b"=");
        h.update(env.get(&k).map(String::as_str).unwrap_or("").as_bytes());
    }
    for (path, _, module, shared) in SHARED_MODEL_PINS {
        if versions_using(module).contains(&name) {
            h.update(b"\x00");
            h.update(path.as_bytes());
            h.update(b"=");
            h.update(sha256_hex(shared).as_bytes());
        }
    }
    Some(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// **凍結版が呼ぶ共有モデル・特徴量モジュール**（issue #21 の依存監査）。
///
/// ここに載っているファイルを更新すると、依存している凍結版の**挙動が変わる**。
/// 実際に 2026-08-21 の value NN 再学習（`src/value_nn.rs`）は v12〜v14 の挙動を
/// 変えている（当時は検知する仕組みが無かった）。opp_move NN は同じ問題に対して
/// `opp_move_nn_v25.rs` という固定コピーを作って解決した先例。
///
/// 記録した sha256 と実ファイルが食い違うとテストが落ちるので、更新する人は
/// 「どの凍結版の挙動が変わるか」を必ず一度は見ることになる。対応は2択:
/// - 固定コピーを作って凍結版だけそちらを呼ばせる（`opp_move_nn_v25` 方式）
/// - 変わることを承知でハッシュを更新し、影響する基準の再計測を記録する
pub const SHARED_MODEL_PINS: &[(&str, &str, &str, &str)] = &[
    (
        "src/likelihood.rs",
        "296d5ce86f6b89d6b433d82bb29cedf5921f8a78738646fa41a90dbfa38342fb",
        "likelihood",
        include_str!("../likelihood.rs"),
    ),
    (
        "src/opp_move_nn_v25.rs",
        "3a1cf06b261f6253ecc95cae27b11d69eb05a3c8f345a82a3c09d530d82d2b90",
        "opp_move_nn_v25",
        include_str!("../opp_move_nn_v25.rs"),
    ),
    (
        "src/opp_move_features.rs",
        "ba6a075b78120105c3b36e0043b1ed08ab84c45f29978b1e0caa8a67ed2024d4",
        "opp_move_features",
        include_str!("../opp_move_features.rs"),
    ),
    (
        "src/value_nn_v22.rs",
        "b72953e2fd82900294c380bf6f1ea4e05752b8e6036ada8ebba63b7cab11d1ab",
        "value_nn_v22",
        include_str!("../value_nn_v22.rs"),
    ),
    (
        "src/value_features.rs",
        "b309c5778792b06dd96fe0366a658c789ae018e4a88a42d0537a45726f211597",
        "value_features",
        include_str!("../value_features.rs"),
    ),
    (
        "src/belief_nn.rs",
        "6a421d86c3881a0dcaee96d70ead71a32c9d2ce9a4ccdcaf9cd4d20b728562a9",
        "belief_nn",
        include_str!("../belief_nn.rs"),
    ),
    (
        "src/belief_features.rs",
        "78a64579e94658861312603b87bbe174f345fd04013fe7224e37eb2a38df38fd",
        "belief_features",
        include_str!("../belief_features.rs"),
    ),
    (
        // **定跡の読み込み実装**。パスの解決規則やフォールバックを変えると
        // 凍結版の序盤分布が変わる（この中の `TSUITATE_JOSEKI` が
        // `env_keys_read_by` で凍結版の実効 env として拾われる）
        "src/opening.rs",
        "7eedcb135bbd977b85811401702a3e43c1969a968dab442c3388ef8fd87cee09",
        "opening",
        include_str!("../opening.rs"),
    ),
    (
        // **定跡データ**。パスは凍結版が固定するが、中身は実行時に読むので
        // ここで内容を pin する（編集すると v12〜v14 と v15 以降の序盤分布が変わる）
        "joseki.json",
        "5076b863f0c68d001df2405e63692b8475bba0386d4b32af1f6b0a5e3e8acaa8",
        "opening",
        include_str!("../../joseki.json"),
    ),
    (
        "src/king_belief_nn.rs",
        "dc8e4165ec0c5a78c4a8c44d2dceb3547f48179b747834455186a44c7815499d",
        "king_belief_nn",
        include_str!("../king_belief_nn.rs"),
    ),
];

/// 共有モジュール `module`（`likelihood` 等のモジュール名）を呼ぶ凍結版の一覧。
/// 再学習・係数更新の前に「何が動くか」を機械的に出すために使う。
pub fn versions_using(module: &str) -> Vec<&'static str> {
    let needle = format!("crate::{module}::");
    SOURCES
        .iter()
        .filter(|(_, _, src)| src.contains(&needle))
        .map(|(_, n, _)| *n)
        .collect()
}

fn sha256_hex(src: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(src.as_bytes());
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **hermetic ガード**（issue #21）: v15 以降の凍結版は戦略 env を読まない。
    /// `scripts/freeze_estimator.py` が落とし忘れたらここで落ちる。
    #[test]
    fn 新しい凍結版は戦略envを読まない() {
        for (v, name, _) in SOURCES.iter().filter(|(v, _, _)| *v >= HERMETIC_FROM) {
            let keys = env_keys_in_source(name);
            assert!(
                keys.is_empty(),
                "凍結版 v{v} が実行時 env を読んでいます: {keys:?}\n                 \
                 scripts/freeze_estimator.py が落とすはずの経路です（issue #21）"
            );
        }
    }

    /// **共有モデル・データの更新を検知する**（issue #21）。
    /// 凍結版が呼ぶファイルの内容が変わったら、影響する基準を必ず見る。
    #[test]
    fn 共有モデルの更新は凍結版への影響つきで検知される() {
        for (path, want, module, src) in SHARED_MODEL_PINS {
            let got = sha256_hex(src);
            let affected = versions_using(module);
            assert_eq!(
                &got.as_str(),
                want,
                "\n{path} が凍結時点から変わっています。\n                 \
                 影響する凍結版: {affected:?}\n                 \
                 対応は2択です（issue #21 / docs/frozen-hermetic-boundary.md）:\n                 \
                 (a) 固定コピーを作って凍結版だけそちらを呼ばせる（opp_move_nn_v25 方式）\n                 \
                 (b) 変わることを承知でこの sha256 を {got} へ更新し、\n                     \
                     影響する基準の再計測を CLAUDE.md へ記録する"
            );
        }
    }

    /// 依存の一覧が機械可読に出せる（再学習前のチェックリスト用）。
    #[test]
    fn 共有モジュールから影響する凍結版を引ける() {
        assert_eq!(versions_using("king_belief_nn"), vec!["estimator_v14"]);
        // v9〜v11 は NN の重みを自分のファイルへコピーしているので影響しない
        assert!(versions_using("opp_move_nn").is_empty());
        for pinned in ["opp_move_nn_v25", "value_nn_v22"] {
            assert_eq!(
                versions_using(pinned),
                vec!["estimator_v12", "estimator_v13", "estimator_v14"],
                "{pinned}"
            );
        }
        // **共有の生モジュールはもう凍結版から呼ばれない**（固定コピーへ移した）
        for shared in ["value_nn", "opp_move_nn"] {
            assert!(
                versions_using(shared).is_empty(),
                "{shared} を凍結版が直接呼んでいる（固定コピーへ向けること）"
            );
        }
    }

    /// **生成側のガード**（issue #21 の完了条件「freeze 生成時に env 読取漏れを
    /// 検出する CI ガード」）。`scripts/freeze_estimator.py` を実際に走らせて、
    /// 生成物に実行時 env が残らないこと・打ち切られていないことを見る。
    ///
    /// 以前 `strip_file` が最初のテストモジュールでファイル末尾まで打ち切って
    /// いた（strategy.rs の 2523 行目以降が丸ごと欠ける）ので、行数の検査も入れる。
    #[test]
    fn freeze生成物は実行時envを読まない() {
        let out = match std::process::Command::new("python3")
            .args(["scripts/freeze_estimator.py", "99", "2026-01-01", "ガード"])
            .output()
        {
            Ok(o) => o,
            // python3 が無い環境ではスキップ（CI には必ずある）
            Err(_) => return,
        };
        assert!(
            out.status.success(),
            "freeze スクリプトが失敗しました:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let body = String::from_utf8_lossy(&out.stdout);
        assert!(
            !body.contains("env::var("),
            "生成した凍結版が実行時 env を読んでいます（issue #21）"
        );
        assert!(body.contains("fn frozen_config()"), "実効設定の固定が入っていない");
        assert!(
            body.contains("pub struct EstimatorV99"),
            "本体が生成されていない（strip_file の打ち切り？）"
        );
        // 3ファイル合計から test / drop 対象を引いた規模。半分以下なら打ち切り
        let lines = body.lines().count();
        assert!(lines > 12_000, "生成物が短すぎます（{lines} 行）");
    }

    /// **相手の実効挙動の指紋**は名前だけでは決まらない（PR #22 レビュー指摘4）。
    #[test]
    fn 凍結版の挙動指紋は版とenvと共有pinで決まる() {
        use std::collections::BTreeMap;
        let empty = BTreeMap::new();
        let v13 = behavior_fingerprint("estimator_v13", &empty).expect("v13");
        let v14 = behavior_fingerprint("estimator_v14", &empty).expect("v14");
        assert_ne!(v13, v14, "版が違えば違う（現行 config の指紋は同じ値になる）");

        // その版が読む env を変えると動く
        let mut env = BTreeMap::new();
        env.insert("TSUITATE_HAND_ASSET_W".to_string(), "0.5".to_string());
        assert!(env_keys_read_by("estimator_v14").contains(&"TSUITATE_HAND_ASSET_W".to_string()));
        assert_ne!(behavior_fingerprint("estimator_v14", &env).unwrap(), v14);

        // **共有モジュール経由の env（定跡パス）も拾う**
        assert!(
            env_keys_read_by("estimator_v14").contains(&"TSUITATE_JOSEKI".to_string()),
            "opening.rs 経由の TSUITATE_JOSEKI を取りこぼしている"
        );
        assert!(!env_keys_in_source("estimator_v14").contains(&"TSUITATE_JOSEKI".to_string()));
        let mut jos = BTreeMap::new();
        jos.insert("TSUITATE_JOSEKI".to_string(), "other.json".to_string());
        assert_ne!(behavior_fingerprint("estimator_v14", &jos).unwrap(), v14);

        // その版が読まない env では動かない
        let mut other = BTreeMap::new();
        other.insert("TSUITATE_KING_REPEAT_FOUL_W".to_string(), "0.8".to_string());
        assert!(!env_keys_read_by("estimator_v14").contains(&"TSUITATE_KING_REPEAT_FOUL_W".to_string()));
        assert_eq!(behavior_fingerprint("estimator_v14", &other).unwrap(), v14);

        // 凍結版でない名前は None（呼び出し側が instance config の指紋を使う）
        assert!(behavior_fingerprint("estimator", &empty).is_none());

        // 共有モジュール経由の env は STRATEGY_ENV_KEYS にも載っていること
        for (_, keys) in SHARED_MODULE_ENV {
            for k in *keys {
                assert!(crate::config::STRATEGY_ENV_KEYS.contains(k), "{k}");
            }
        }
    }

    /// v6〜v14 は env を読む「既知の負債」。一覧が取れることを担保する
    /// （checkpoint arena / arena の実行前検査がこれを使う）。
    #[test]
    fn 既存凍結版が読むenvを機械的に列挙できる() {
        assert!(SOURCES.iter().all(|(v, _, _)| *v < HERMETIC_FROM));
        let v14 = env_keys_in_source("estimator_v14");
        assert!(v14.contains(&"TSUITATE_HAND_ASSET_W".to_string()), "{v14:?}");
        assert!(v14.contains(&"TSUITATE_THINK_BUDGET_MS".to_string()), "{v14:?}");
        // 共有モジュール経由（定跡）はファイル内には出ない
        assert!(!v14.contains(&"TSUITATE_JOSEKI".to_string()));
        assert!(env_keys_in_source("estimator_v6").len() < v14.len());
    }
}
