//! 方策・価値ネットの推論（手書きの forward pass。依存を増やさないため）。
//!
//! 学習は `~/Develop/tsuitate-nn/rnad/`（`model.py`）、重みは `export_weights.py` が
//! BatchNorm を畳み込みへ畳み込んで書き出す。形式は
//! `b"TSRLPV01" | u32 ヘッダ長 | ヘッダ JSON | f32 LE の並び`。
//!
//! 構成（`model.py` と同じ。層を変えたら両方と書き出しを直すこと）:
//! 3×3 畳み込み（入力 86 → C）→ 残差ブロック×B（3×3 畳み込み×2）→
//! 方策: 1×1（C→32）ReLU → 1×1（32→139）= [139, 9, 9] を平らにしたロジット、
//! 価値: 1×1（C→4）ReLU → 全結合 324→128 ReLU → 128→1 → tanh。
//! 盤は `[チャネル, 段, 筋]` の並び（`rl::encode` のプレーン・`rl::action` のマスと同じ）。

use std::collections::HashMap;
use std::io::Read;

use crate::rl::action::{NUM_ACTION_KINDS, NUM_ACTIONS};
use crate::rl::encode::{NUM_PLANES, OBS_LEN};

const MAGIC: &[u8; 8] = b"TSRLPV01";
const SQ: usize = 81;

#[derive(Debug)]
pub struct PolicyNet {
    channels: usize,
    blocks: usize,
    tensors: HashMap<String, Vec<f32>>,
}

#[derive(serde::Deserialize)]
struct Header {
    planes: usize,
    channels: usize,
    blocks: usize,
    tensors: Vec<TensorInfo>,
}

#[derive(serde::Deserialize)]
struct TensorInfo {
    name: String,
    shape: Vec<usize>,
}

impl PolicyNet {
    pub fn load(path: &str) -> Result<Self, String> {
        let mut bytes = vec![];
        std::fs::File::open(path)
            .and_then(|mut f| f.read_to_end(&mut bytes))
            .map_err(|e| format!("{path}: {e}"))?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 12 || &bytes[..8] != MAGIC {
            return Err("方策ネットの重みファイルではない（magic 不一致）".into());
        }
        let hlen = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let header: Header = serde_json::from_slice(
            bytes.get(12..12 + hlen).ok_or("ヘッダが切れている")?,
        )
        .map_err(|e| e.to_string())?;
        if header.planes != NUM_PLANES {
            return Err(format!(
                "入力プレーン数 {} が現行のエンコーダ {NUM_PLANES} と違う",
                header.planes
            ));
        }
        let mut off = 12 + hlen;
        let mut tensors = HashMap::new();
        for t in &header.tensors {
            let n: usize = t.shape.iter().product();
            let raw = bytes
                .get(off..off + 4 * n)
                .ok_or_else(|| format!("{} のデータが切れている", t.name))?;
            let v: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            tensors.insert(t.name.clone(), v);
            off += 4 * n;
        }
        if off != bytes.len() {
            return Err("末尾に余分なデータがある".into());
        }
        let net = PolicyNet {
            channels: header.channels,
            blocks: header.blocks,
            tensors,
        };
        net.check_shapes()?;
        Ok(net)
    }

    fn check_shapes(&self) -> Result<(), String> {
        let c = self.channels;
        let mut want: Vec<(String, usize)> = vec![
            ("stem.w".into(), c * NUM_PLANES * 9),
            ("stem.b".into(), c),
            ("p1.w".into(), 32 * c),
            ("p1.b".into(), 32),
            ("p2.w".into(), NUM_ACTION_KINDS * 32),
            ("p2.b".into(), NUM_ACTION_KINDS),
            ("v1.w".into(), 4 * c),
            ("v1.b".into(), 4),
            ("v_fc1.w".into(), 128 * 4 * SQ),
            ("v_fc1.b".into(), 128),
            ("v_fc2.w".into(), 128),
            ("v_fc2.b".into(), 1),
        ];
        for i in 0..self.blocks {
            for j in 1..=2 {
                want.push((format!("block{i}.c{j}.w"), c * c * 9));
                want.push((format!("block{i}.c{j}.b"), c));
            }
        }
        for (name, len) in want {
            match self.tensors.get(&name) {
                Some(t) if t.len() == len => {}
                Some(t) => return Err(format!("{name} の長さ {} が {len} と違う", t.len())),
                None => return Err(format!("{name} が無い")),
            }
        }
        Ok(())
    }

    fn t(&self, name: &str) -> &[f32] {
        &self.tensors[name]
    }

    /// 観測（長さ `OBS_LEN`）→ (ロジット（長さ `NUM_ACTIONS`、マスク前）, 価値 ∈ [−1, 1])
    pub fn forward(&self, obs: &[f32]) -> (Vec<f32>, f32) {
        assert_eq!(obs.len(), OBS_LEN);
        let c = self.channels;
        let mut x = conv3x3(obs, NUM_PLANES, c, self.t("stem.w"), self.t("stem.b"));
        relu(&mut x);
        for i in 0..self.blocks {
            let mut y = conv3x3(&x, c, c, self.t(&format!("block{i}.c1.w")), self.t(&format!("block{i}.c1.b")));
            relu(&mut y);
            let y = conv3x3(&y, c, c, self.t(&format!("block{i}.c2.w")), self.t(&format!("block{i}.c2.b")));
            for (a, b) in x.iter_mut().zip(&y) {
                *a = (*a + b).max(0.0);
            }
        }
        let mut p = conv1x1(&x, c, 32, self.t("p1.w"), self.t("p1.b"));
        relu(&mut p);
        let logits = conv1x1(&p, 32, NUM_ACTION_KINDS, self.t("p2.w"), self.t("p2.b"));
        debug_assert_eq!(logits.len(), NUM_ACTIONS);

        let mut v = conv1x1(&x, c, 4, self.t("v1.w"), self.t("v1.b"));
        relu(&mut v);
        let mut h = linear(&v, 128, self.t("v_fc1.w"), self.t("v_fc1.b"));
        relu(&mut h);
        let value = linear(&h, 1, self.t("v_fc2.w"), self.t("v_fc2.b"))[0].tanh();
        (logits, value)
    }
}

fn relu(x: &mut [f32]) {
    for v in x {
        *v = v.max(0.0);
    }
}

/// 3×3 畳み込み（パディング1）。`w` は [出力, 入力, 3, 3]
fn conv3x3(x: &[f32], cin: usize, cout: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; cout * SQ];
    for co in 0..cout {
        let o = &mut out[co * SQ..(co + 1) * SQ];
        o.fill(b[co]);
        for ci in 0..cin {
            let src = &x[ci * SQ..(ci + 1) * SQ];
            for ky in 0..3 {
                for kx in 0..3 {
                    let wv = w[((co * cin + ci) * 3 + ky) * 3 + kx];
                    if wv == 0.0 {
                        continue;
                    }
                    let (dy, dx) = (ky as isize - 1, kx as isize - 1);
                    let (y0, y1) = ((-dy).max(0) as usize, (9 - dy).min(9) as usize);
                    let (x0, x1) = ((-dx).max(0) as usize, (9 - dx).min(9) as usize);
                    for y in y0..y1 {
                        let sy = (y as isize + dy) as usize;
                        let orow = &mut o[y * 9 + x0..y * 9 + x1];
                        let srow = &src[sy * 9 + (x0 as isize + dx) as usize..];
                        for (ov, sv) in orow.iter_mut().zip(srow) {
                            *ov += wv * sv;
                        }
                    }
                }
            }
        }
    }
    out
}

/// 1×1 畳み込み。`w` は [出力, 入力]
fn conv1x1(x: &[f32], cin: usize, cout: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; cout * SQ];
    for co in 0..cout {
        let o = &mut out[co * SQ..(co + 1) * SQ];
        o.fill(b[co]);
        for ci in 0..cin {
            let wv = w[co * cin + ci];
            for (ov, sv) in o.iter_mut().zip(&x[ci * SQ..(ci + 1) * SQ]) {
                *ov += wv * sv;
            }
        }
    }
    out
}

/// 全結合。`w` は [出力, 入力]
fn linear(x: &[f32], nout: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
    let nin = x.len();
    (0..nout)
        .map(|o| b[o] + w[o * nin..(o + 1) * nin].iter().zip(x).map(|(a, b)| a * b).sum::<f32>())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/rl/tiny_policy");

    /// PyTorch（eval モード、BatchNorm あり）の出力と一致する
    #[test]
    fn pytorch_と同じ出力になる() {
        let net = PolicyNet::load(&format!("{FIXTURE}.bin")).unwrap();
        let tv: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{FIXTURE}.test.json")).unwrap())
                .unwrap();
        let inputs = tv["input"].as_array().unwrap();
        for (i, input) in inputs.iter().enumerate() {
            let x: Vec<f32> = serde_json::from_value(input.clone()).unwrap();
            let want: Vec<f32> = serde_json::from_value(tv["logits"][i].clone()).unwrap();
            let want_v = tv["value"][i].as_f64().unwrap() as f32;
            let (logits, v) = net.forward(&x);
            let max_err = logits
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_err < 1e-3, "sample {i}: ロジットの最大誤差 {max_err}");
            assert!((v - want_v).abs() < 1e-4, "sample {i}: 価値 {v} vs {want_v}");
        }
    }

    /// 1回の forward の所要時間。
    /// `RL_POLICY_BENCH=<重み> cargo test --release --lib 方策ネットの速度 -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn 方策ネットの速度() {
        let net = PolicyNet::load(&std::env::var("RL_POLICY_BENCH").unwrap()).unwrap();
        let x = vec![0.5f32; OBS_LEN];
        let t = std::time::Instant::now();
        let n = 50;
        for _ in 0..n {
            std::hint::black_box(net.forward(std::hint::black_box(&x)));
        }
        println!(
            "{}ch×{}ブロック: {:.2}ms/回",
            net.channels,
            net.blocks,
            t.elapsed().as_secs_f64() * 1e3 / n as f64
        );
    }

    #[test]
    fn 壊れたファイルは拒否する() {
        assert!(PolicyNet::from_bytes(b"NOTMAGIC0000").is_err());
        let mut bytes = std::fs::read(format!("{FIXTURE}.bin")).unwrap();
        bytes.pop();
        assert!(PolicyNet::from_bytes(&bytes).is_err());
    }
}
