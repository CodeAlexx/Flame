#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

//! Long-running model-level parity test for the TensorIterator silu
//! migration (HANDOFF_2026-04-22_TENSORITERATOR_PORT, gate 5).
//!
//! Spawns the `klein9b_infer_turbo` binary at seed 42 / "a cat in a hat" /
//! 1024² / 30 steps and compares the output PNG to the baseline fixture
//! checked in at
//! `inference-flame/tests/fixtures/klein_seed42_baseline.png`. Fails the
//! test if `cos_sim < 0.9999`.
//!
//! Marked `#[ignore]` because each run takes ~3 minutes and uses ~20 GB of
//! GPU VRAM. Run explicitly:
//!
//! ```text
//! cd flame-core && cargo test --release --test png_parity_klein \
//!     -- --ignored --nocapture
//! ```
//!
//! This test lives in flame-core/tests/ per the handoff, but it spawns an
//! inference-flame binary. The binary must be built first from inside
//! inference-flame:
//!
//! ```text
//! cd ../inference-flame && cargo build --release \
//!     --bin klein9b_infer_turbo --features turbo
//! ```
//!
//! The fixture was generated at flame-core HEAD 5bc9857 before the
//! `Tensor::silu` dispatch was flipped to route through
//! `ops::silu_iter::silu_bf16_iter`. Post-flip runs must match it —
//! `silu_bf16_iter` delegates contig inputs straight back to the pre-flip
//! kernel, so every current Klein caller should be bit-exact.

use std::path::PathBuf;
use std::process::Command;

/// Resolve paths relative to `flame-core`'s `CARGO_MANIFEST_DIR`. Assumes
/// the standard `EriDiffusion/{flame-core,inference-flame}` layout.
fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("flame-core dir has a parent")
        .to_path_buf()
}

fn klein_bin() -> PathBuf {
    workspace_root()
        .join("inference-flame")
        .join("target")
        .join("release")
        .join("klein9b_infer_turbo")
}

fn baseline_fixture() -> PathBuf {
    workspace_root()
        .join("inference-flame")
        .join("tests")
        .join("fixtures")
        .join("klein_seed42_baseline.png")
}

fn klein_output() -> PathBuf {
    workspace_root()
        .join("inference-flame")
        .join("output")
        .join("klein9b_rust_turbo.png")
}

/// Runtime cuDNN/CUDA library paths, matching the morning handoff's
/// `LD_LIBRARY_PATH` recipe so the bin can dlopen cuDNN 9.
fn ld_library_path() -> String {
    let extras = [
        "/home/alex/.local/lib/python3.12/site-packages/nvidia/cudnn/lib",
        "/home/alex/.serenity/cudnn/lib",
        "/usr/local/cuda/lib64",
    ];
    let existing = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
    let mut parts: Vec<&str> = extras.to_vec();
    if !existing.is_empty() {
        parts.push(&existing);
    }
    parts.join(":")
}

/// Decode a PNG at `path` to a flat RGB f64 buffer. Uses the `image` crate
/// already in flame-core's direct deps.
fn load_rgb_f64(path: &std::path::Path) -> Vec<f64> {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display()))
        .to_rgb8();
    img.as_raw().iter().map(|&b| b as f64).collect()
}

fn cos_sim(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "pixel count mismatch");
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[test]
#[ignore] // ~3 min, requires klein9b_infer_turbo built in sibling dir.
fn klein9b_silu_iter_dispatch_pngs_match_baseline() {
    let bin = klein_bin();
    let fixture = baseline_fixture();
    let output = klein_output();

    assert!(
        bin.exists(),
        "klein9b_infer_turbo not found at {}. \
         Build it first: cd inference-flame && cargo build --release \
         --bin klein9b_infer_turbo --features turbo",
        bin.display()
    );
    assert!(
        fixture.exists(),
        "baseline fixture not found at {}. Regenerate with \
         klein9b_infer_turbo at a pre-flip flame-core HEAD.",
        fixture.display()
    );

    let status = Command::new(&bin)
        .arg("a cat in a hat")
        .env("LD_LIBRARY_PATH", ld_library_path())
        .status()
        .expect("failed to spawn klein9b_infer_turbo");
    assert!(status.success(), "klein9b_infer_turbo exited {status:?}");

    let baseline = load_rgb_f64(&fixture);
    let produced = load_rgb_f64(&output);
    let cs = cos_sim(&baseline, &produced);
    println!("klein PNG cos_sim vs baseline: {cs:.10}");
    assert!(
        cs >= 0.9999,
        "Klein output cos_sim {cs} below 0.9999 threshold"
    );
}
