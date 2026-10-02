//! 方策ネットで指す戦略（arena・本番用）。
//!
//! 戦略名に重みファイルのパスを埋め込む（`strategy::make`）:
//! - `rl_policy:<path>` — マスク内の softmax から**サンプリング**（混合戦略として学んだ分布どおり）
//! - `rl_policy_greedy:<path>` — 最大の手（決定的）
//! - `rl_policy_basic:<path>` — サンプリングだが**旧マスク**（`action::basic_legal_mask`。観測からの
//!   確定反則を落とさない）。凍結版 rl_v15〜rl_v21 と同じ手順で、同一性テストとマスクの効果測定に使う
//!
//! 入力は `Strategy::choose` の引数だけで、学習環境と同じ `rl::encode` / `rl::action` を通る。
//!
//! **重みは内容の sha256 で識別する**: 戦略を作るたびにファイルを読んでハッシュを取り、同じ内容の
//! ネットはプロセス内で共有する（arena は対局ごとに戦略を作り直す）。`name()` は
//! `rl_policy@<sha256 先頭12桁>` を返すので、対局記録から使った重みを特定できる
//! （同じパスの重みを学習し直して上書きしても、記録上は別の戦略になる）。

use std::collections::{HashMap, HashSet};
use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex, OnceLock};

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::observation::ObservationLog;
use crate::protocol::PlayerView;
use crate::rl::action::{basic_legal_mask, decode_usi, legal_mask};
use crate::rl::encode::encode;
use crate::rl::policy_net::PolicyNet;
use crate::strategy::Strategy;

pub const PREFIX_SAMPLE: &str = "rl_policy:";
pub const PREFIX_GREEDY: &str = "rl_policy_greedy:";
pub const PREFIX_BASIC: &str = "rl_policy_basic:";

/// 読み込んだ重み（内容のハッシュで共有する）
#[derive(Clone)]
struct Loaded {
    net: Arc<PolicyNet>,
    /// `name()` が返す名前（サンプリング / greedy / 旧マスク）。ハッシュごとに1度だけ確保して使い回す
    names: (&'static str, &'static str, &'static str),
    sha256: String,
}

/// 重みを読む。内容の sha256 が同じならネットを共有する
fn load_cached(path: &str) -> Result<Loaded, String> {
    static CACHE: OnceLock<Mutex<HashMap<String, Loaded>>> = OnceLock::new();
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let sha256: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().unwrap();
    if let Some(l) = cache.get(&sha256) {
        return Ok(l.clone());
    }
    let short = &sha256[..12];
    let loaded = Loaded {
        net: Arc::new(PolicyNet::from_bytes(&bytes)?),
        names: (
            Box::leak(format!("rl_policy@{short}").into_boxed_str()),
            Box::leak(format!("rl_policy_greedy@{short}").into_boxed_str()),
            Box::leak(format!("rl_policy_basic@{short}").into_boxed_str()),
        ),
        sha256: sha256.clone(),
    };
    cache.insert(sha256, loaded.clone());
    Ok(loaded)
}

pub struct RlPolicy {
    net: Arc<PolicyNet>,
    names: (&'static str, &'static str, &'static str),
    sha256: String,
    greedy: bool,
    basic_mask: bool,
    rng: StdRng,
    last: Option<serde_json::Value>,
}

impl RlPolicy {
    /// 戦略名（`rl_policy:<path>` / `rl_policy_greedy:<path>` / `rl_policy_basic:<path>`）から作る。
    /// 名前が違えば None、重みが読めなければ panic（arena の設定ミスを黙って別の戦略にしないため）
    pub fn from_name(name: &str, seed: Option<u64>) -> Option<Self> {
        let (greedy, basic_mask, path) = if let Some(p) = name.strip_prefix(PREFIX_GREEDY) {
            (true, false, p)
        } else if let Some(p) = name.strip_prefix(PREFIX_BASIC) {
            (false, true, p)
        } else if let Some(p) = name.strip_prefix(PREFIX_SAMPLE) {
            (false, false, p)
        } else {
            return None;
        };
        let loaded = load_cached(path).unwrap_or_else(|e| panic!("方策ネットを読めない: {e}"));
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_rng(&mut rand::rng()),
        };
        Some(RlPolicy {
            net: loaded.net,
            names: loaded.names,
            sha256: loaded.sha256,
            greedy,
            basic_mask,
            rng,
            last: None,
        })
    }
}

impl Strategy for RlPolicy {
    fn choose(
        &mut self,
        view: &PlayerView,
        log: &ObservationLog,
        foul_tried: &HashSet<String>,
    ) -> Option<String> {
        let mask = if self.basic_mask {
            basic_legal_mask(view, foul_tried)
        } else {
            legal_mask(view, log, foul_tried)
        };
        let legal: Vec<usize> = (0..mask.len()).filter(|&a| mask[a]).collect();
        if legal.is_empty() {
            return None;
        }
        let (logits, value) = self.net.forward(&encode(view, log, foul_tried));
        let max = legal.iter().map(|&a| logits[a]).fold(f32::MIN, f32::max);
        let weights: Vec<f64> = legal.iter().map(|&a| f64::from(logits[a] - max).exp()).collect();
        let total: f64 = weights.iter().sum();
        let pick = if self.greedy {
            legal
                .iter()
                .copied()
                .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
                .unwrap()
        } else {
            let mut r = self.rng.random::<f64>() * total;
            let mut chosen = *legal.last().unwrap();
            for (&a, w) in legal.iter().zip(&weights) {
                if r < *w {
                    chosen = a;
                    break;
                }
                r -= w;
            }
            chosen
        };
        let prob = weights[legal.iter().position(|&a| a == pick).unwrap()] / total;
        self.last = Some(serde_json::json!({
            "p": prob,
            "value": value,
            "weights_sha256": self.sha256,
            "greedy": self.greedy,
            "mask_version": if self.basic_mask { 1 } else { crate::rl::action::MASK_VERSION },
        }));
        decode_usi(pick, view.your_color)
    }

    fn name(&self) -> &'static str {
        if self.greedy {
            self.names.1
        } else if self.basic_mask {
            self.names.2
        } else {
            self.names.0
        }
    }

    fn debug_state(&self) -> Option<serde_json::Value> {
        self.last.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::referee::{Referee, StepResult};

    const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/rl/tiny_policy.bin");

    /// 学習前の小さいネットでも、マスク内の手だけを指して1局を終えられる
    #[test]
    fn 方策ネットで1局指せる() {
        for name in [format!("{PREFIX_SAMPLE}{FIXTURE}"), format!("{PREFIX_GREEDY}{FIXTURE}")] {
            let mut players = [
                RlPolicy::from_name(&name, Some(1)).unwrap(),
                RlPolicy::from_name(&name, Some(2)).unwrap(),
            ];
            let mut referee = Referee::new();
            loop {
                let side = referee.to_move();
                let view = referee.view(side, [0, 0], 0);
                let p = &mut players[side as usize];
                let Some(usi) = p.choose(&view, referee.log(side), referee.foul_tried(side)) else {
                    break;
                };
                let mask = legal_mask(&view, referee.log(side), referee.foul_tried(side));
                assert!(mask[crate::rl::action::encode_usi(&usi, side).unwrap()]);
                if let StepResult::Ended { .. } = referee.step(&usi, 0) {
                    break;
                }
            }
        }
    }

    /// 名前に重みの内容のハッシュが入る（パスが同じでも中身が違えば別の名前）
    #[test]
    fn 名前は重みの内容で決まる() {
        let a = RlPolicy::from_name(&format!("{PREFIX_SAMPLE}{FIXTURE}"), Some(0)).unwrap();
        let g = RlPolicy::from_name(&format!("{PREFIX_GREEDY}{FIXTURE}"), Some(0)).unwrap();
        assert!(a.name().starts_with("rl_policy@") && a.name().len() == "rl_policy@".len() + 12);
        assert_eq!(g.name(), a.name().replace("rl_policy@", "rl_policy_greedy@"));
        assert!(Arc::ptr_eq(&a.net, &g.net), "同じ内容のネットは共有する");

        let dir = std::env::temp_dir().join(format!("rl_policy_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let copy = dir.join("w.bin");
        std::fs::copy(FIXTURE, &copy).unwrap();
        let path = format!("{PREFIX_SAMPLE}{}", copy.display());
        let before = RlPolicy::from_name(&path, Some(0)).unwrap().name();
        assert_eq!(before, a.name(), "同じ内容なら別のパスでも同じ名前");
        // 同じパスへ別の重み（末尾のバイアスを1つ変える）を上書きすると名前が変わる
        let mut bytes = std::fs::read(FIXTURE).unwrap();
        let n = bytes.len();
        bytes[n - 1] ^= 0x01;
        std::fs::write(&copy, &bytes).unwrap();
        let after = RlPolicy::from_name(&path, Some(0)).unwrap().name();
        assert_ne!(before, after);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn 名前が違えば作らない() {
        assert!(RlPolicy::from_name("heuristic", None).is_none());
    }
}
