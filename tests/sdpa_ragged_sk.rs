#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

//! Regression test for the multi-K-tile softmax-tail bug in
//! `flame_core::sdpa::forward_flash_bf16` (FA2 WMMA kernel at
//! `src/cuda/flash_attention_fwd.cu`).
//!
//! ## The bug (fixed 2026-04-19)
//!
//! The kernel zero-padded invalid tail columns in the second K tile with
//! `0.0f` before per-row softmax. `exp(0 - new_max)` then contributed ~56
//! spurious terms to the online-softmax denominator for `Sk=72`
//! (`kv_rows=8` in the second tile), inflating it by ~25-30% and
//! under-scaling output by a matching amount.
//!
//! Observed before fix:
//!
//! ```text
//! Sk=64  cos_sim=0.999995  mag_ratio=1.000   OK (1 tile, no tail)
//! Sk=71  cos_sim=0.997969  mag_ratio=0.672   BAD
//! Sk=72  cos_sim=0.998328  mag_ratio=0.678   BAD
//! ```
//!
//! Fix: mask invalid tail cells with `-INFINITY` instead of `0.0f`.
//! `exp(-inf - m) == 0` exactly, and `max(x, -inf) == x` so tile_max
//! stays correct. See `mask_tail_2d_float_neg_inf` in the kernel.
//!
//! After fix all three sizes return `mag_ratio=1.000` and
//! `cos_sim ≥ 0.9999`.

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

/// Regression: multi-K-tile softmax under-scale bug (fixed 2026-04-19).
///
/// Covers both Sk > BKV=64 with tail (Sk=71, 72) and Sk exactly at the
/// tile boundary (Sk=64). All three must return `mag_ratio ≥ 0.95` and
/// `cos_sim ≥ 0.999`.
#[test]
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

    // Sk=128 — two full tiles, no tail. Should have been healthy even
    // under the old bug, but locks the common case.
    let (cs_128, am_f_128, am_r_128) = run_one(128, &device)?;
    println!(
        "[sdpa_ragged_sk] Sk=128 cos_sim={:.6}  abs_mean(flash)={:.4e}  abs_mean(ref)={:.4e}  mag_ratio={:.4}",
        cs_128, am_f_128, am_r_128, mag_ratio(am_f_128, am_r_128)
    );

    // Sk=200 — three K tiles, last one ragged (kv_rows=8). Catches a
    // hypothetical off-by-one in the num_kv_tiles loop that two-tile
    // coverage alone could miss.
    let (cs_200, am_f_200, am_r_200) = run_one(200, &device)?;
    println!(
        "[sdpa_ragged_sk] Sk=200 cos_sim={:.6}  abs_mean(flash)={:.4e}  abs_mean(ref)={:.4e}  mag_ratio={:.4}",
        cs_200, am_f_200, am_r_200, mag_ratio(am_f_200, am_r_200)
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

    // Multi-tile K regressions. Each must be within BF16 precision of
    // the materialized reference.
    for (label, cs, ratio) in [
        ("Sk=72",  cs_72,  mag_ratio(am_f_72,  am_r_72)),
        ("Sk=71",  cs_71,  mag_ratio(am_f_71,  am_r_71)),
        ("Sk=128", cs_128, mag_ratio(am_f_128, am_r_128)),
        ("Sk=200", cs_200, mag_ratio(am_f_200, am_r_200)),
    ] {
        assert!(
            ratio >= 0.95,
            "{label} mag_ratio {:.4} — FA2 under-scales by ~{:.0}% (regression)",
            ratio, (1.0 - ratio) * 100.0
        );
        assert!(
            cs >= 0.999,
            "{label} cos_sim {:.6} < 0.999 — FA2 disagrees with reference (regression)",
            cs
        );
    }

    Ok(())
}
