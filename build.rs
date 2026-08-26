//! **ビルド時のソース指紋**（issue #24 の replicate 検査。PR #25 レビュー指摘 P1）。
//!
//! `bin/export_eval_rank_data` が summary へ書く `source_fingerprint` は、
//! 「この CSV を作ったバイナリのコード版」でなければ意味がない。実行時に
//! worktree を読むと、commit A でビルドしたバイナリを worktree B の状態で
//! 走らせたときに **A の挙動を B の指紋で記録**してしまう（長い実行中に
//! 編集した場合も同じ TOCTOU）。ここで**コンパイルされるソースそのもの**を
//! ハッシュしてバイナリへ焼き込む。
//!
//! 対象は `src/**/*.rs`（凍結版・共有モデルの重みを含む）と `Cargo.lock`
//! （依存の版も挙動に効く）。定跡・元 KIF は実行時に読む**データ**なので
//! ここには入れず、exporter が実効パスの中身を `data_fingerprint` として別に出す。

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
    println!("cargo::rerun-if-changed=build.rs");

    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut files = vec![];
    walk(&root.join("src"), &mut files);
    files.push(root.join("Cargo.lock"));
    files.sort();

    let mut h = Sha256::new();
    for f in &files {
        h.update(f.strip_prefix(&root).unwrap_or(f).to_string_lossy().as_bytes());
        h.update([0]);
        h.update(std::fs::read(f).unwrap_or_default());
        h.update([0]);
    }
    let hex: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    println!("cargo::rustc-env=TSUITATE_SOURCE_FINGERPRINT={hex}");
}
