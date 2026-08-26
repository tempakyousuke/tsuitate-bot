//! **ビルド時のソース指紋**（issue #24 の replicate 検査。PR #25 レビュー指摘 P1）。
//!
//! `bin/export_eval_rank_data` が summary へ書く `source_fingerprint` は、
//! 「この CSV を作ったバイナリのコード版」でなければ意味がない。実行時に
//! worktree を読むと、commit A でビルドしたバイナリを worktree B の状態で
//! 走らせたときに **A の挙動を B の指紋で記録**してしまう（長い実行中に
//! 編集した場合も同じ TOCTOU）。ここで**コンパイルされるソースそのもの**を
//! ハッシュしてバイナリへ焼き込む。
//!
//! 対象は **`src/**/*.rs`（凍結版・共有モデルの重みを含む）＋ `Cargo.lock` ＋
//! `Cargo.toml` ＋ この `build.rs` 自身**に加えて、**実効ビルド条件**
//! （profile / target / features / `RUSTFLAGS` / rustc 版）。
//! ソースだけでは足りない（PR #25 レビュー指摘 P1）: 同じ head を debug と release で
//! ビルドすると指紋が一致してしまい、**壁時計予算のこのエクスポートでは探索量と
//! 出力分布が大きく変わるのに同じ実験として replicate 平均へ入る**。
//!
//! 定跡・元 KIF は実行時に読む**データ**なのでここには入れず、exporter が
//! `data_fingerprint` として別に出す。

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

fn main() {
    println!("cargo::rerun-if-changed=src");
    println!("cargo::rerun-if-changed=Cargo.lock");
    println!("cargo::rerun-if-changed=Cargo.toml");
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-env-changed=RUSTFLAGS");
    println!("cargo::rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");

    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut files = vec![];
    walk(&root.join("src"), &mut files);
    files.push(root.join("Cargo.lock"));
    files.push(root.join("Cargo.toml"));
    files.push(root.join("build.rs"));
    files.sort();

    let mut h = Sha256::new();
    for f in &files {
        h.update(f.strip_prefix(&root).unwrap_or(f).to_string_lossy().as_bytes());
        h.update([0]);
        h.update(std::fs::read(f).unwrap_or_default());
        h.update([0]);
    }

    // **実効ビルド条件**。同じソースでも profile が違えば挙動（探索量）が違う
    let profile = std::env::var("PROFILE").unwrap_or_default();
    let mut features: Vec<String> = std::env::vars()
        .filter(|(k, _)| k.starts_with("CARGO_FEATURE_"))
        .map(|(k, _)| k)
        .collect();
    features.sort();
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let rustc_version = std::process::Command::new(&rustc)
        .arg("--version")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    for (label, v) in [
        ("profile", profile.clone()),
        ("target", std::env::var("TARGET").unwrap_or_default()),
        ("opt-level", std::env::var("OPT_LEVEL").unwrap_or_default()),
        ("debug-assertions", std::env::var("DEBUG").unwrap_or_default()),
        ("features", features.join(",")),
        ("rustflags", std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default()),
        ("rustc", rustc_version),
    ] {
        h.update(label.as_bytes());
        h.update([0]);
        h.update(v.as_bytes());
        h.update([0]);
    }

    let hex: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    println!("cargo::rustc-env=TSUITATE_SOURCE_FINGERPRINT={hex}");
    // 起動時の profile 検査に使う（壁時計予算の計測を debug ビルドで取らせない）
    println!("cargo::rustc-env=TSUITATE_BUILD_PROFILE={profile}");
}
