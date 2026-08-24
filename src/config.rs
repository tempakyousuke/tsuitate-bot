//! **戦略設定の境界**（issue #21）。
//!
//! `TSUITATE_*` は CLI・サーバー等の構成境界で**一度だけ**解釈し、
//! 以後は strategy instance が持つ [`StrategyConfig`] だけを見る。
//! これで「同じプロセスの候補と凍結相手が同じ env を読む」という
//! checkpoint arena / arena の罠（PR #20 のレビュー指摘）を、
//! **プロセス env を触らずに**塞げる:
//!
//! - 候補へノブを渡すのは config であってプロセス env ではないので、
//!   env を読み続ける既存の凍結版 v6〜v14 は既定値のまま動く
//! - `OnceLock` のプロセス全体キャッシュを使わないので、
//!   arm ごとの構築順・並列実行で値が混ざらない
//!
//! ## 現在値の解決
//!
//! 評価・推定の実装は深い呼び出しの奥から定数を引くので、config は
//! [`scoped`] でスレッドローカルに**設置**する（`EstimatorStrategy` の
//! `choose` / `prewarm` / `oracle_anchor` が入口で設置する）。設置されて
//! いない経路（診断バイナリ・GUI・凍結版 v6〜v14）は [`ambient`]
//! （プロセス env を一度だけ解釈したもの）に落ちるので、**移行前と同じ挙動**
//! になる。v15 以降の凍結版は**凍結時点の値だけを持つ自前の config**
//! （定跡パスもリテラルで固定）を設置するので、共有モジュール経由の設定まで
//! env からも「将来の既定値の変更」からも切り離される。
//!
//! ## 監査
//!
//! [`STRATEGY_ENV_KEYS`] が「戦略の強さに関わる env キー」の全量。
//! ソース走査との突き合わせは `config.rs` のテストが常時検査するので、
//! ノブを足して config を通し忘れると `cargo test` が落ちる。

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

/// env の読み取り面。`std::env::var` と同じ戻り型なので、
/// 既存の解釈式（`.ok().and_then(...)` / `.is_ok_and(...)` / `map_or`）を
/// そのまま移設できる。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EnvSource {
    vars: BTreeMap<String, String>,
}

impl EnvSource {
    /// 何も与えない = すべて既定値。
    pub fn empty() -> Self {
        Self::default()
    }

    /// プロセス env の `TSUITATE_*` を取り込む（構成境界でだけ呼ぶ）。
    pub fn from_process() -> Self {
        Self {
            vars: std::env::vars()
                .filter(|(k, _)| k.starts_with("TSUITATE_"))
                .collect(),
        }
    }

    pub fn from_pairs<I, K, V>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        Self {
            vars: pairs.into_iter().map(|(k, v)| (k.into(), v.into())).collect(),
        }
    }

    /// `std::env::var` の置き換え。
    pub fn var(&self, key: &str) -> Result<String, std::env::VarError> {
        match self.vars.get(key) {
            Some(v) => Ok(v.clone()),
            None => Err(std::env::VarError::NotPresent),
        }
    }

    pub fn vars(&self) -> &BTreeMap<String, String> {
        &self.vars
    }

    /// 上書きを重ねた新しい source を返す（arm 固有ノブを共通設定へ載せる用）。
    pub fn with_overrides<I, K, V>(&self, pairs: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        let mut vars = self.vars.clone();
        for (k, v) in pairs {
            vars.insert(k.into(), v.into());
        }
        Self { vars }
    }

    /// 与えたキーのうち、戦略が実際に見るもの（[`STRATEGY_ENV_KEYS`]）だけ。
    /// 記録・検査用（未知のキーは戦略に影響しないので落とす）。
    pub fn strategy_keys(&self) -> BTreeMap<String, String> {
        self.vars
            .iter()
            .filter(|(k, _)| STRATEGY_ENV_KEYS.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

/// 戦略の強さに関わる設定の全量。**instance ごとに持つ**。
#[derive(Clone, Debug, PartialEq)]
pub struct StrategyConfig {
    /// 解決に使った生の値（記録・fingerprint 用）。
    pub source: EnvSource,
    /// 1手あたりの思考予算（粒子数・読み幅がこれに比例する）。
    pub think_budget_ms: u64,
    /// 定跡ファイル。
    pub joseki_path: String,
    pub strategy: crate::strategy::StrategyKnobs,
    pub estimator: crate::estimator::EstimatorKnobs,
    pub check: crate::check::CheckKnobs,
}

impl StrategyConfig {
    /// すべて既定値（env を一切見ない）。
    pub fn defaults() -> Self {
        Self::from_source(EnvSource::empty())
    }

    /// プロセス env から解決する（構成境界でだけ呼ぶ）。
    pub fn from_env() -> Self {
        Self::from_source(EnvSource::from_process())
    }

    pub fn from_source(source: EnvSource) -> Self {
        let think_budget_ms = ["TSUITATE_CAND_THINK_BUDGET_MS", "TSUITATE_THINK_BUDGET_MS"]
            .iter()
            .find_map(|name| source.var(name).ok().and_then(|v| v.parse().ok()))
            .unwrap_or(crate::strategy::DEFAULT_THINK_BUDGET_MS);
        let joseki_path = source
            .var("TSUITATE_JOSEKI")
            .unwrap_or_else(|_| "joseki.json".into());
        StrategyConfig {
            strategy: crate::strategy::StrategyKnobs::from_source(&source),
            estimator: crate::estimator::EstimatorKnobs::from_source(&source),
            check: crate::check::CheckKnobs::from_source(&source),
            think_budget_ms,
            joseki_path,
            source,
        }
    }

    /// 実効設定の指紋。**生の env ではなく解決後の値**から作るので、
    /// 「未知のキーを足しただけ」では変わらず、「既定値と同じ値を明示した」
    /// でも変わらない。凍結相手の実効設定が候補設定で動いていないことを
    /// 機械的に確かめるのに使う。
    pub fn fingerprint(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(self.think_budget_ms.to_string().as_bytes());
        h.update(b"\x00");
        h.update(self.joseki_path.as_bytes());
        h.update(b"\x00");
        // Debug は全フィールドを含み、フィールドを足せば必ず変わる
        h.update(format!("{:?}", self.strategy).as_bytes());
        h.update(b"\x00");
        h.update(format!("{:?}", self.estimator).as_bytes());
        h.update(b"\x00");
        h.update(format!("{:?}", self.check).as_bytes());
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }
}

thread_local! {
    static CURRENT: RefCell<Arc<StrategyConfig>> = RefCell::new(ambient());
}

/// プロセス env を一度だけ解釈した既定 config。config を設置しない経路
/// （診断バイナリ・GUI・凍結版）の互換のために残してある。
pub fn ambient() -> Arc<StrategyConfig> {
    static A: OnceLock<Arc<StrategyConfig>> = OnceLock::new();
    A.get_or_init(|| Arc::new(StrategyConfig::from_env())).clone()
}

/// 現在設置されている config を読む（評価・推定の実装から呼ぶ）。
#[inline]
pub fn current<R>(f: impl FnOnce(&StrategyConfig) -> R) -> R {
    CURRENT.with(|c| f(&c.borrow()))
}

/// 現在設置されている config そのもの（診断・記録用）。
pub fn current_config() -> Arc<StrategyConfig> {
    CURRENT.with(|c| c.borrow().clone())
}

/// [`scoped`] の戻り値。drop で元の config へ戻す。
pub struct Scope(Option<Arc<StrategyConfig>>);

impl Drop for Scope {
    fn drop(&mut self) {
        if let Some(prev) = self.0.take() {
            CURRENT.with(|c| *c.borrow_mut() = prev);
        }
    }
}

/// `cfg` をこのスレッドへ設置する。戻り値が生きている間だけ有効。
///
/// スレッドローカルなので、arena / scenario が対局をスレッド並列に回しても
/// arm 同士が混ざらない（`OnceLock` を使わない理由）。
#[must_use = "戻り値を保持している間だけ config が有効"]
pub fn scoped(cfg: &Arc<StrategyConfig>) -> Scope {
    let prev = CURRENT.with(|c| std::mem::replace(&mut *c.borrow_mut(), cfg.clone()));
    Scope(Some(prev))
}

/// 明示的に渡されたノブの検査結果（PR #22 レビュー指摘2）。
#[derive(Debug, Default, PartialEq, Eq)]
pub struct KnobCheck {
    /// 戦略が読まないキー（綴り間違い）。**エラーにすること**。
    pub unknown: Vec<String>,
    /// 与えたのに実効設定が変わらなかったキー（既定値と同じ値か、
    /// 解釈できない・範囲外の値で既定へ戻ったか）。**警告して記録すること**。
    pub ineffective: Vec<String>,
}

impl KnobCheck {
    pub fn is_clean(&self) -> bool {
        self.unknown.is_empty() && self.ineffective.is_empty()
    }
}

/// 明示的なノブ指定を検査する。
///
/// `TSUITATE_` の接頭辞しか見ないと、`TSUITATE_HAND_ASSET_WW=0.5` のような
/// 綴り間違いが黙って通り、**実効値は既定のまま**の実験が正常完走してしまう
/// （「凍結版へ渡して設定したつもり」と同じ事故の別経路）。既知キーでも
/// 解釈できない値は既定へ戻るので同じことが起きる。
pub fn check_overrides(base: &EnvSource, overrides: &BTreeMap<String, String>) -> KnobCheck {
    let mut out = KnobCheck::default();
    let base_fp = StrategyConfig::from_source(base.clone()).fingerprint();
    for (k, v) in overrides {
        if !STRATEGY_ENV_KEYS.contains(&k.as_str()) {
            out.unknown.push(k.clone());
            continue;
        }
        // 1件ずつ載せて実効設定が動くかを見る（他のキーと打ち消し合っても
        // 「このキーは効いた」と数えたいので、まとめてではなく個別に）
        let one = base.with_overrides([(k.clone(), v.clone())]);
        if StrategyConfig::from_source(one).fingerprint() == base_fp {
            out.ineffective.push(k.clone());
        }
    }
    out
}

/// 戦略の強さに関わる env キーの全量（監査・記録用）。
///
/// **ここに載っているキーは config 経由でしか読まない**（テストが検査する）。
/// 既存の凍結版 v6〜v14 は自分のコピーの中で直接 env を読むので、この一覧は
/// 「凍結相手が反応しうるキー」の上界でもある。
pub const STRATEGY_ENV_KEYS: &[&str] = &[
    "TSUITATE_ANCHOR_MOVE_W",
    "TSUITATE_BELIEF_GAIN_W",
    "TSUITATE_BELIEF_GUIDE_W",
    "TSUITATE_BELIEF_LIVE_W",
    "TSUITATE_BELIEF_OCC_CAP_W",
    "TSUITATE_BELIEF_SPAN",
    "TSUITATE_BISHOP_RETREAT_W",
    "TSUITATE_BLIND_ATTACK_SURVIVE_W",
    "TSUITATE_BLIND_HOME_DROP_OCC_W",
    "TSUITATE_BLIND_HOME_FLOOR",
    "TSUITATE_BLIND_HOME_LAMBDA",
    "TSUITATE_BLIND_HOME_RISK_W",
    "TSUITATE_BLIND_RECAPTURE_W",
    "TSUITATE_BOARD_DISCOUNT_W",
    "TSUITATE_CAND_THINK_BUDGET_MS",
    "TSUITATE_CAPTURE_BET_VAR_W",
    "TSUITATE_CAPTURE_PRIOR_LAMBDA",
    "TSUITATE_CAPTURE_RETREAT_W",
    "TSUITATE_CHECKER_REMOVAL_W",
    "TSUITATE_CHECK_BELIEF_OCC_W",
    "TSUITATE_CHECK_CAPTURE_BOOST",
    "TSUITATE_CHECK_CAPTURE_PRUNE",
    "TSUITATE_CHECK_DROP_EXPLAIN_W",
    "TSUITATE_CHECK_DROP_TARGET",
    "TSUITATE_CHECK_FOUL_PRIOR_BOOST",
    "TSUITATE_CHECK_KING_COVER_CAP",
    "TSUITATE_CHECK_KING_GAIN_MEAN",
    "TSUITATE_CHECK_KING_PRIOR_W",
    "TSUITATE_CHECK_KNOWN_UNACC",
    "TSUITATE_CHECK_SAFE_RESOLVE",
    "TSUITATE_CHECK_STRENGTH_W",
    "TSUITATE_CHECK_WALKIN_DISCOUNT",
    "TSUITATE_DEBUG_CHECK",
    "TSUITATE_DEBUG_REJUV",
    "TSUITATE_DEBUG_REJUV_SQ",
    "TSUITATE_DEFENDER_CAPTURE_W",
    "TSUITATE_DEPTH2_FOCAL_K",
    "TSUITATE_DEPTH2_OPTIMISM_CAP",
    "TSUITATE_DISABLE_DEFEND_GUIDE",
    "TSUITATE_DROP_HIT_ALL_RANKS",
    "TSUITATE_DROP_HIT_EVAC_W",
    "TSUITATE_DROP_PROBE_REPEAT_GATE",
    "TSUITATE_DROP_PROBE_W",
    "TSUITATE_EFFECT_OPP_W",
    "TSUITATE_EFFECT_OWN_W",
    "TSUITATE_ENABLE_HANG_RISK",
    "TSUITATE_ENDGAME_CAMP_GENERAL_W",
    "TSUITATE_EPS_PHYS",
    "TSUITATE_ESCAPE_COVER_W",
    "TSUITATE_EVAL_TAINT_ATTACK",
    "TSUITATE_EVAL_TAINT_FALLBACK",
    "TSUITATE_EVAL_WEIGHT_CAP",
    "TSUITATE_EXPOSED_KNOWN",
    "TSUITATE_EXPOSED_MULTI_W",
    "TSUITATE_EXPOSED_PAWN_HEAD_W",
    "TSUITATE_FAR_MAJOR_PROMO_CAPTURE_W",
    "TSUITATE_FILTER_DEBUG",
    "TSUITATE_FOUL_OCC_ATTACK_W",
    "TSUITATE_GEN_NONPROMOTE",
    "TSUITATE_GOLD_JOIN_KING_W",
    "TSUITATE_GOLD_KING_FILE_W",
    "TSUITATE_HAND_ASSET_W",
    "TSUITATE_HAND_OPTION_W",
    "TSUITATE_HOME_GOLD_ATTACK_W",
    "TSUITATE_JOSEKI",
    "TSUITATE_KING_ADJ_ENTRY_W",
    "TSUITATE_KING_ADJ_HEAVY_W",
    "TSUITATE_KING_BELIEF_PROX_W",
    "TSUITATE_KING_CAND_ATTACK_BLIND",
    "TSUITATE_KING_CAND_ATTACK_GATE",
    "TSUITATE_KING_CAND_ATTACK_W",
    "TSUITATE_KING_CAND_CHECK_W",
    "TSUITATE_KING_CAPTURE_REVEAL",
    "TSUITATE_KING_ENDGAME_FLEE_W",
    "TSUITATE_KING_FILE_GOLD_W",
    "TSUITATE_KING_FILE_PAWN_MID_W",
    "TSUITATE_KING_FILE_PAWN_W",
    "TSUITATE_KING_HOLE_W",
    "TSUITATE_KING_KNOWN_APPROACH_W",
    "TSUITATE_KING_NET_PROJ",
    "TSUITATE_KING_NET_W",
    "TSUITATE_KING_PROBE_W",
    "TSUITATE_KING_PROX_EXCLUDE_SELF",
    "TSUITATE_KING_REPEAT_FOUL_W",
    "TSUITATE_KING_SENSOR_W",
    "TSUITATE_KNIGHT_CAMP_EXIT_W",
    "TSUITATE_KNIGHT_ENDGAME_PROMO_W",
    "TSUITATE_KNIGHT_LATE_PROMO_W",
    "TSUITATE_LANDING_SUPPORT_W",
    "TSUITATE_LAST_FOUL_GUARD",
    "TSUITATE_LAST_FOUL_GUARD_2",
    "TSUITATE_LAST_FOUL_GUARD_3",
    "TSUITATE_LINK_ENDGAME_DAMPEN",
    "TSUITATE_LINK_W",
    "TSUITATE_LINK_WORK_REF",
    "TSUITATE_LINK_WORK_W",
    "TSUITATE_MAJOR_PROMO_PATH_W",
    "TSUITATE_MATERIAL_DEGEN_Q0",
    "TSUITATE_MATE_GATE_Q0",
    "TSUITATE_MATE_RISK_W",
    "TSUITATE_MATE_THREAT_W",
    "TSUITATE_MOVER_CHECK_EXTRA",
    "TSUITATE_MUT_RESCUE",
    "TSUITATE_NONPROMOTE_CHECK_P",
    "TSUITATE_NONPROMOTE_CHECK_ROLES",
    "TSUITATE_NONPROMOTE_CHECK_W",
    "TSUITATE_OPP_CAPTURE_REVEAL_W",
    "TSUITATE_OPP_MOVE_TEMP",
    "TSUITATE_OWN_CAMP_IDLE_W",
    "TSUITATE_OWN_CAMP_MINOR_PROMO_W",
    "TSUITATE_OWN_ZONE_CAPTURE_W",
    "TSUITATE_PATH_PROBE_W",
    "TSUITATE_PAWN_OFFFILE_W",
    "TSUITATE_PLAN_W",
    "TSUITATE_PROBE_ANCHOR_DECAY",
    "TSUITATE_PROBE_AUDIT",
    "TSUITATE_PROBE_THREAT_W",
    "TSUITATE_PROMOTE_BIAS",
    "TSUITATE_PROMOTE_CHECK_REVEAL_W",
    "TSUITATE_PROMOTE_FAR_ALL",
    "TSUITATE_PROMOTE_FAR_W",
    "TSUITATE_PROMO_DECAY",
    "TSUITATE_PROMO_KING_PROX",
    "TSUITATE_PROMO_POTENTIAL_W",
    "TSUITATE_PROMO_REALIZED_FLOOR",
    "TSUITATE_PROMO_RISK_PREROLE",
    "TSUITATE_REGEN_KEEP_IS",
    "TSUITATE_REJUV_KEEP_CAPTURER",
    "TSUITATE_REPEAT_PENALTY_W",
    "TSUITATE_SENSOR_P_PROMO",
    "TSUITATE_SENSOR_P_PUSH",
    "TSUITATE_SHUFFLE_PENALTY",
    "TSUITATE_SILVER_CAMP_EXIT_W",
    "TSUITATE_STALE_THREAT_W",
    "TSUITATE_TAINT_KING_EMPTY",
    "TSUITATE_TAINT_KING_FIX",
    "TSUITATE_TAINT_MULTISET_REPAIR",
    "TSUITATE_TAINT_OCC_LEGAL_W",
    "TSUITATE_THINK_BUDGET_MS",
    "TSUITATE_THREAT_BY_COUNT",
    "TSUITATE_TOKIN_APPROACH_W",
    "TSUITATE_TOKIN_FILE_DRIFT_W",
    "TSUITATE_UNBACKED_CAMP_W",
    "TSUITATE_UNBACKED_GS_CAPTURE_W",
    "TSUITATE_V1_DEFENDED",
    "TSUITATE_V1_PRESSURE",
    "TSUITATE_VALUE_NN_W",
];

#[cfg(test)]
mod tests {
    use super::*;

    /// `#[cfg(test)]` ブロックを落とした本体。行単位の素朴な走査だが、
    /// このリポジトリの書き方（属性が行頭・`mod tests {` が続く）には十分。
    fn non_test_body(src: &str) -> String {
        let lines: Vec<&str> = src.lines().collect();
        let mut out = String::new();
        let mut i = 0;
        while i < lines.len() {
            if lines[i].trim_start().starts_with("#[cfg(test)]") {
                let mut depth: i32 = 0;
                let mut started = false;
                while i < lines.len() {
                    depth += lines[i].matches('{').count() as i32;
                    depth -= lines[i].matches('}').count() as i32;
                    if lines[i].contains('{') {
                        started = true;
                    }
                    i += 1;
                    if started && depth <= 0 {
                        break;
                    }
                }
                continue;
            }
            out.push_str(lines[i]);
            out.push('\n');
            i += 1;
        }
        out
    }

    /// 戦略が読む env キーを本文から拾う（`src.var` / `source.var` /
    /// `probe_env(src, ..)` / `env_f64(src, ..)`）。
    fn config_keys(body: &str) -> Vec<String> {
        // 改行・インデントを畳んでから見る（`src\n    .var("...")` の書き方対策）
        let body: String = {
            let mut out = String::new();
            let mut ws = false;
            for ch in body.chars() {
                if ch.is_whitespace() {
                    ws = true;
                } else {
                    if ws && !out.is_empty() {
                        out.push(' ');
                    }
                    ws = false;
                    out.push(ch);
                }
            }
            out.replace(" .var(", ".var(")
        };
        let body = body.as_str();
        let mut out = vec![];
        for (i, _) in body.match_indices("\"TSUITATE_") {
            let rest = &body[i + 1..];
            let end = match rest.find('"') {
                Some(e) => e,
                None => continue,
            };
            let before = &body[..i];
            // 直前が config 経由の読み取りか（生の env::var なら別テストが落とす）
            if before.ends_with(".var(")
                || before.ends_with("probe_env(src, ")
                || before.ends_with("env_f64(src, ")
            {
                out.push(rest[..end].to_string());
            }
        }
        out
    }

    const SOURCES: &[(&str, &str)] = &[
        ("src/strategy.rs", include_str!("strategy.rs")),
        ("src/estimator.rs", include_str!("estimator.rs")),
        ("src/check.rs", include_str!("check.rs")),
        ("src/opening.rs", include_str!("opening.rs")),
        ("src/config.rs", include_str!("config.rs")),
    ];

    /// **issue #21 の中心的なガード**: 現行戦略はプロセス env を直接読まない。
    /// ノブを足すときに `std::env::var` を書くと必ずここで落ちるので、
    /// 「候補用の env が同じプロセスの凍結相手にも効く」問題が再発しない。
    #[test]
    fn 現行戦略はプロセスenvを直接読まない() {
        for (name, src) in SOURCES {
            let body = non_test_body(src);
            assert!(
                !body.contains("env::var(\"TSUITATE_"),
                "{name} が TSUITATE_* をプロセス env から直接読んでいる。\
                 crate::config::StrategyConfig 経由にすること（issue #21）"
            );
        }
    }

    /// config が読むキーと [`STRATEGY_ENV_KEYS`] が過不足なく一致する。
    /// 記録・検査（checkpoint arena の env 監査）がこの一覧に依存している。
    /// `config.rs` 自身は「キー名の一覧」を持つので、literal を全部拾う。
    fn literal_keys(body: &str) -> Vec<String> {
        let mut out = vec![];
        for (i, _) in body.match_indices("\"TSUITATE_") {
            let rest = &body[i + 1..];
            if let Some(end) = rest.find('"') {
                // `k.starts_with("TSUITATE_")` のような前方一致の判定は除く
                if end > "TSUITATE_".len() {
                    out.push(rest[..end].to_string());
                }
            }
        }
        out
    }

    #[test]
    fn 戦略envキーの一覧が実装と一致する() {
        let mut found: Vec<String> = SOURCES
            .iter()
            .flat_map(|(name, src)| {
                let body = non_test_body(src);
                if *name == "src/config.rs" {
                    literal_keys(&body)
                } else {
                    config_keys(&body)
                }
            })
            .collect();
        found.sort();
        found.dedup();
        let declared: Vec<String> = STRATEGY_ENV_KEYS.iter().map(|s| s.to_string()).collect();
        let missing: Vec<_> = found.iter().filter(|k| !declared.contains(k)).collect();
        let extra: Vec<_> = declared.iter().filter(|k| !found.contains(k)).collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "STRATEGY_ENV_KEYS の過不足: 未登録={missing:?} 余分={extra:?}"
        );
        assert_eq!(STRATEGY_ENV_KEYS.len(), declared.len());
    }

    #[test]
    fn 既定configはenvを一切見ない() {
        // `defaults()` は EnvSource::empty なので、プロセス env がどうであれ同じ
        let a = StrategyConfig::defaults();
        let b = StrategyConfig::defaults();
        assert_eq!(a.fingerprint(), b.fingerprint());
        assert_eq!(a.think_budget_ms, crate::strategy::DEFAULT_THINK_BUDGET_MS);
        assert_eq!(a.joseki_path, "joseki.json");
    }

    /// **候補設定はプロセス env を触らない**。同じプロセスで env を読み続ける
    /// 既存の凍結版 v6〜v14 が候補ノブに反応しないことの担保。
    #[test]
    fn 候補設定はプロセスenvを変えない() {
        let key = "TSUITATE_HAND_ASSET_W";
        let before = std::env::var(key);
        let cfg = StrategyConfig::from_source(EnvSource::from_pairs([(key, "0.5")]));
        assert_eq!(cfg.strategy.hand_asset_w, 0.5);
        assert_eq!(
            std::env::var(key).ok(),
            before.ok(),
            "config の構築でプロセス env が変わってはいけない"
        );
        // 既定 config は影響を受けない
        assert_eq!(StrategyConfig::defaults().strategy.hand_asset_w, 0.0);
    }

    /// 設置はスレッドローカル・スコープつきなので、入れ子・並列で混ざらない。
    #[test]
    fn 設置は入れ子で正しく戻る() {
        let base = std::sync::Arc::new(StrategyConfig::defaults());
        let cand = std::sync::Arc::new(StrategyConfig::from_source(EnvSource::from_pairs([(
            "TSUITATE_HAND_ASSET_W",
            "0.5",
        )])));
        assert_ne!(base.fingerprint(), cand.fingerprint());
        let _outer = scoped(&base);
        assert_eq!(current(|c| c.strategy.hand_asset_w), 0.0);
        {
            let _inner = scoped(&cand);
            assert_eq!(current(|c| c.strategy.hand_asset_w), 0.5);
        }
        assert_eq!(current(|c| c.strategy.hand_asset_w), 0.0);
    }

    /// 綴り間違い・解釈できない値を検出する（PR #22 レビュー指摘2）。
    #[test]
    fn ノブの綴り間違いと無効値を検出する() {
        let base = EnvSource::empty();
        let mk = |pairs: &[(&str, &str)]| -> BTreeMap<String, String> {
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect()
        };
        // 綴り間違い（戦略が読まないキー）
        let c = check_overrides(&base, &mk(&[("TSUITATE_HAND_ASSET_WW", "0.5")]));
        assert_eq!(c.unknown, vec!["TSUITATE_HAND_ASSET_WW".to_string()]);
        assert!(c.ineffective.is_empty());
        // 解釈できない値 → 既定へ戻るので実効値が動かない
        let c = check_overrides(&base, &mk(&[("TSUITATE_HAND_ASSET_W", "とても大きい")]));
        assert!(c.unknown.is_empty());
        assert_eq!(c.ineffective, vec!["TSUITATE_HAND_ASSET_W".to_string()]);
        // 既定値と同じ値も「効かなかった」側（実験として無意味なので気づけるように）
        let c = check_overrides(&base, &mk(&[("TSUITATE_HAND_ASSET_W", "0")]));
        assert_eq!(c.ineffective, vec!["TSUITATE_HAND_ASSET_W".to_string()]);
        // 効くノブは何も出ない
        assert!(check_overrides(&base, &mk(&[("TSUITATE_HAND_ASSET_W", "0.5")])).is_clean());
        // 範囲外（負値は filter で弾かれる）
        let c = check_overrides(&base, &mk(&[("TSUITATE_HAND_ASSET_W", "-1")]));
        assert_eq!(c.ineffective, vec!["TSUITATE_HAND_ASSET_W".to_string()]);
    }

    #[test]
    fn 指紋は解決後の値で決まる() {
        let a = StrategyConfig::defaults();
        // 既定値と同じ値を明示しても指紋は変わらない
        let b = StrategyConfig::from_source(EnvSource::from_pairs([(
            "TSUITATE_HAND_ASSET_W",
            "0",
        )]));
        assert_eq!(a.fingerprint(), b.fingerprint());
        // 未知のキーは戦略に影響しないので指紋も変わらない
        let c = StrategyConfig::from_source(EnvSource::from_pairs([("TSUITATE_NOT_A_KNOB", "9")]));
        assert_eq!(a.fingerprint(), c.fingerprint());
        // 範囲外の値は検証で落ちるので既定と同じ
        let d = StrategyConfig::from_source(EnvSource::from_pairs([(
            "TSUITATE_HAND_ASSET_W",
            "とても大きい",
        )]));
        assert_eq!(a.fingerprint(), d.fingerprint());
    }
}
