#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

//! FIXME: kernel bug at Sk > BKV=64 tile size (regardless of alignment).
//!
//! Minimal repro for an under-scaling bug in `flame_core::sdpa::forward_flash_bf16`
//! — the FA2 WMMA kernel at `src/cuda/flash_attention_fwd.cu` — surfaced by
//! the Stable Cascade AttnBlock parity test.
//!
//! ## What we observed
//!
//! Running Q `[1, 32, 64, 64]` and K/V `[1, 32, Sk, 64]` against a materialized
//! FP32 reference:
//!
//! ```text
//! Sk=64  cos_sim=0.999995  abs_mean(flash)=0.153  abs_mean(ref)=0.153    OK (1 tile)
//! Sk=71  cos_sim=0.997969  abs_mean(flash)=0.100  abs_mean(ref)=0.149    BAD (flash ≈ 67% of ref)
//! Sk=72  cos_sim=0.998328  abs_mean(flash)=0.099  abs_mean(ref)=0.147    BAD (flash ≈ 68% of ref)
//! ```
//!
//! The skeptic initially hypothesized "Sk not a multiple of 16", but Sk=72
//! (divisible by 16, 8, and 72) exhibits the same ~32% under-scale as Sk=71.
//! The actual pattern is **Sk > BKV=64 tile size** — i.e. any second K tile
//! collapses/under-contributes to the online softmax denominator, which
//! makes the output magnitude about 2/3 of correct.
//!
//! The per-element cos_sim ≈ 0.998 means the *direction* of the vectors is
//! close but the *magnitude* is off — consistent with a scale bug rather
//! than a full correctness failure.
//!
//! ## Why this is `#[ignore]`d
//!
//! The kernel bug is real and blocks Stable Cascade inference (CLIP seq=77
//! and arbitrary H*W resolutions regularly exceed 64 K tokens). This test
//! is kept as a *failing* test, gated by `#[ignore]`, so the bug remains
//! visible without breaking CI. Run explicitly to check repro:
//!
//! ```bash
//! cargo test --release -p flame-core --test sdpa_ragged_sk -- --ignored --nocapture
//! ```
//!
//! The follow-up work for the flame-core maintainer is to fix the online
//! softmax denominator accumulation across K tiles in
//! `src/cuda/flash_attention_fwd.cu`. Until then, any caller that needs
//! SDPA with `Sk > 64` must route through the materialized path by either:
//!   - setting `FLAME_NO_FLASH_ATTN=1` for the whole process, or
//!   - calling `sdpa::forward_with_bias(q, k, v, None, None)` explicitly,
//!     which always uses the FP32 materialized path.

use anyhow::Result;
use flame_core::{sdpa, DType, Device, Shape, Tensor};

/// Cosine similarity of two same-shaped tensors, both upcast to F32.
fn cos_sim(a: &Tensor, b: &Tensor) -> Result<f32> {
    assert_eq!(a.shape().dims(), b.shape().dims(), "shape mismatch");
    let av = a.to_dtype(DType::F32)?.to_vec_f32()?;
    let bv = b.to_dtype(DType::F32)?.to_vec_f32()?;
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in av.iter().zip(bv.iter()) {
        let xd = *x as f64;
        let yd = *y as f64;
        dot += xd * yd;
        na += xd * xd;
        nb += yd * yd;
    }
    Ok(if na > 0.0 && nb > 0.0 {
        (dot / (na.sqrt() * nb.sqrt())) as f32
    } else {
        0.0
    })
}

/// abs_mean(|t|).
fn abs_mean(t: &Tensor) -> Result<f32> {
    let v = t.to_dtype(DType::F32)?.to_vec_f32()?;
    if v.is_empty() {
        return Ok(0.0);
    }
    let s: f64 = v.iter().map(|x| x.abs() as f64).sum();
    Ok((s / v.len() as f64) as f32)
}

fn run_one(sk: usize, device: &std::sync::Arc<cudarc::driver::CudaDevice>) -> Result<(f32, f32, f32)> {
    const B: usize = 1;
    const H: usize = 32;
    const SQ: usize = 64;
    const D: usize = 64;

    // Deterministic per-sk seed.
    flame_core::rng::set_seed(0xCA5C4DE_u64 ^ (sk as u64))
        .map_err(|e| anyhow::anyhow!("set_seed: {e:?}"))?;

    let q = Tensor::randn(Shape::from_dims(&[B, H, SQ, D]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;
    let k = Tensor::randn(Shape::from_dims(&[B, H, sk, D]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;
    let v = Tensor::randn(Shape::from_dims(&[B, H, sk, D]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;

    // Path 1: public `forward` — dispatches to FA2 when conditions are met
    // (D ∈ {64, 96, 128}, mask=None, `FLAME_NO_FLASH_ATTN` unset). The test
    // harness is not permitted to mutate `FLAME_NO_FLASH_ATTN` mid-run
    // because `use_flash_attn()` caches the value in a OnceLock.
    let out_flash = sdpa::forward(&q, &k, &v, None)?;

    // Path 2: materialized FP32 reference via `forward_with_bias` with
    // `bias=None` and `scale=None` (defaults to 1/sqrt(D)). This path never
    // touches the FA2 kernel — it upcasts to F32, runs raw BMM/softmax/BMM.
    // It is the canonical correct-by-construction reference.
    let out_ref = sdpa::forward_with_bias(&q, &k, &v, None, None)?;

    let cs = cos_sim(&out_flash, &out_ref)?;
    let am_flash = abs_mean(&out_flash)?;
    let am_ref = abs_mean(&out_ref)?;
    Ok((cs, am_flash, am_ref))
}

/// Ratio of abs_means. If < 0.9 the flash kernel has materially under-scaled
/// the output relative to the reference.
fn mag_ratio(flash_abs_mean: f32, ref_abs_mean: f32) -> f32 {
    if ref_abs_mean.abs() < 1e-12 {
        1.0
    } else {
        flash_abs_mean / ref_abs_mean
    }
}

/// FIXME: kernel bug at Sk > BKV=64 tile size.
///
/// Explicitly `#[ignore]`d because the kernel is broken: Sk=71 and Sk=72
/// both fail, demonstrating the bug is NOT about tile alignment but about
/// any multi-tile K. This test is the canonical repro; it stays committed
/// so the bug remains visible until fixed in `src/cuda/flash_attention_fwd.cu`.
#[test]
#[ignore = "FIXME: kernel bug at non-aligned Sk (actually: any Sk > BKV=64 tile) — see module doc"]
fn sdpa_ragged_sk_minimal_repro() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev.cuda_device_arc(),
        Err(err) => {
            eprintln!("[skip] CUDA unavailable: {err}");
            return Ok(());
        }
    };

    // Sk=64 — fits in a single BKV=64 tile. Healthy.
    let (cs_64, am_f_64, am_r_64) = run_one(64, &device)?;
    println!(
        "[sdpa_ragged_sk] Sk=64  cos_sim={:.6}  abs_mean(flash)={:.4e}  abs_mean(ref)={:.4e}  mag_ratio={:.4}",
        cs_64, am_f_64, am_r_64, mag_ratio(am_f_64, am_r_64)
    );

    // Sk=72 — tile-aligned under the original "mod 16" hypothesis, but
    // requires a second K tile. This is where the bug first appears.
    let (cs_72, am_f_72, am_r_72) = run_one(72, &device)?;
    println!(
        "[sdpa_ragged_sk] Sk=72  cos_sim={:.6}  abs_mean(flash)={:.4e}  abs_mean(ref)={:.4e}  mag_ratio={:.4}",
        cs_72, am_f_72, am_r_72, mag_ratio(am_f_72, am_r_72)
    );

    // Sk=71 — genuinely ragged. Same bug as Sk=72, not worse. Proves the
    // issue is multi-tile K, not tile-alignment.
    let (cs_71, am_f_71, am_r_71) = run_one(71, &device)?;
    println!(
        "[sdpa_ragged_sk] Sk=71  cos_sim={:.6}  abs_mean(flash)={:.4e}  abs_mean(ref)={:.4e}  mag_ratio={:.4}",
        cs_71, am_f_71, am_r_71, mag_ratio(am_f_71, am_r_71)
    );

    // Single-tile control must pass. If this fails, the reference path
    // itself is broken; bail with a different message so we don't mis-
    // diagnose as a kernel bug.
    assert!(
        cs_64 >= 0.999,
        "Sk=64 (single-tile) cos_sim {:.6} < 0.999 — reference/flash disagree even in the \
         trivial case; test setup is wrong, not the kernel",
        cs_64
    );

    // These are the bug: multi-tile K produces ~32% magnitude shortfall.
    // We assert they should hold; this test is #[ignore]d so CI stays
    // green, but running it must demonstrate the failure.
    let ratio_72 = mag_ratio(am_f_72, am_r_72);
    let ratio_71 = mag_ratio(am_f_71, am_r_71);
    assert!(
        ratio_72 >= 0.95,
        "Sk=72 mag_ratio {:.4} — FA2 under-scales by ~{:.0}% (kernel bug)",
        ratio_72, (1.0 - ratio_72) * 100.0
    );
    assert!(
        ratio_71 >= 0.95,
        "Sk=71 mag_ratio {:.4} — FA2 under-scales by ~{:.0}% (kernel bug)",
        ratio_71, (1.0 - ratio_71) * 100.0
    );

    Ok(())
}
