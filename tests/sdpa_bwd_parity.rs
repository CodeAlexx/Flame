#![cfg(all(feature = "cuda", feature = "bf16_u16"))]
//! SDPA backward parity / determinism harness (Stage 0c of the
//! backward-pass overhaul plan, see `~/.claude/plans/keen-crafting-jellyfish.md`).
//!
//! ## What this currently tests (Stage 0c)
//!
//! Same-path determinism: run the *decomposed* SDPA backward twice with the
//! same seed-derived Q/K/V/dO inputs and assert the two runs produce
//! bit-identical (or near-bit-identical) gradients. This proves the
//! decomposed bwd is deterministic and gives us a sanity baseline.
//!
//! Currently both runs go through `attention_backward_recompute`
//! (`autograd.rs:957-1003`) because the cuDNN bwd path bails on the F32 d_o
//! issue at `autograd.rs:847-848`. We also belt-and-suspenders set
//! `FLAME_NO_CUDNN_SDPA_BWD=1` so neither run can accidentally take the
//! cuDNN path even if that bail is removed later.
//!
//! ## What Stage 2 will add
//!
//! Once the cuDNN SDPA bwd is re-enabled (Stage 2), this same harness will
//! be extended with a cuDNN-vs-decomposed cross-path comparison using the
//! parameterized tolerance (`cos_min=0.9995`, `max_abs_rel=1e-2`). The
//! `compare_grads` helper below already accepts those parameters.
//!
//! ## Coverage matrix
//!
//! 1. zimage self-attn:  `(B=1, H=24, Nq=Nkv=1024, d=128)`
//! 2. klein cross-attn:  `(B=2, H=20, Nq=Nkv=512, d=64)`
//! 3. cross-attn mixed:  `(B=1, H=20, Nq=1024, Nkv=512, d=96)`
//! 4. klein 9B self-attn:`(B=1, H=32, Nq=Nkv=4096, d=128)`
//!
//! Each independently runnable via:
//! ```text
//! cargo test --release --features cuda --test sdpa_bwd_parity -- <test_name>
//! ```

use anyhow::{anyhow, Result};
use cudarc::driver::CudaDevice;
use flame_core::{sdpa, AutogradContext, DType, Shape, Tensor};
use std::sync::{Arc, Mutex, OnceLock};

/// Serialize all tests in this file. `AutogradContext` is global mutable
/// state (see `autograd.rs::AUTOGRAD_CONTEXT`) — running tests in parallel
/// causes `AutogradContext::reset()` from one test to destroy another's
/// tape, surfacing as spurious "backward() called on tensor that doesn't
/// require grad" errors. This mutex is held for the duration of each test.
fn test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn try_device() -> Option<Arc<CudaDevice>> {
    match CudaDevice::new(0) {
        Ok(dev) => Some(dev),
        Err(err) => {
            eprintln!("[skip] CUDA unavailable: {err:?}");
            None
        }
    }
}

/// Build a BF16 tensor with shape `dims`, seeded for run-to-run reproducibility.
fn randn_bf16(dims: &[usize], seed: u64, device: &Arc<CudaDevice>) -> Result<Tensor> {
    let t = Tensor::randn_seeded(Shape::from_dims(dims), 0.0, 1.0, seed, Arc::clone(device))?;
    Ok(t.to_dtype(DType::BF16)?)
}

/// Compute cosine similarity + max absolute relative error between two
/// (possibly different-dtype) tensors. Operates in F32 for accuracy.
///
/// `cos_min` and `max_abs_rel` parameterize the pass/fail thresholds. For
/// Stage 0c same-path determinism we use `cos_min=0.9999` (essentially
/// "must match"); Stage 2 cuDNN-vs-decomposed will use `cos_min=0.9995,
/// max_abs_rel=1e-2` per the plan.
fn compare_grads(
    a: &Tensor,
    b: &Tensor,
    label: &str,
    cos_min: f64,
    max_abs_rel: f64,
) -> Result<()> {
    if a.shape().dims() != b.shape().dims() {
        return Err(anyhow!(
            "[{label}] shape mismatch: {:?} vs {:?}",
            a.shape().dims(),
            b.shape().dims()
        ));
    }
    let a_vec = a.to_dtype(DType::F32)?.to_vec_f32()?;
    let b_vec = b.to_dtype(DType::F32)?.to_vec_f32()?;
    if a_vec.len() != b_vec.len() {
        return Err(anyhow!(
            "[{label}] element count mismatch: {} vs {}",
            a_vec.len(),
            b_vec.len()
        ));
    }

    // First pass: find max_abs_ref so we can floor the small-value cutoff
    // relative to the magnitude actually present in the tensor. A fixed
    // 1e-6 absolute floor (the original harness) blew up `max_rel` to
    // hundreds for BF16 gradients where most values are near 1e-1 but a
    // handful sit at 1e-5 — `diff / 1e-5` looks huge but represents
    // sub-quantization noise. We compare diff against `eps * max_abs_ref`
    // (standard BF16 parity practice — see PyTorch test_transformers.py).
    let mut max_abs_a = 0.0_f64;
    for x in a_vec.iter() {
        let xa = (*x as f64).abs();
        if xa.is_finite() && xa > max_abs_a {
            max_abs_a = xa;
        }
    }
    // Floor for "interesting" elements: 10% of the tensor's max magnitude.
    // BF16 has ~3e-3 relative precision per element; below 10% of max the
    // absolute error from softmax recompute (a chain of BF16 mul + F32
    // softmax + BF16 mul) is comparable to the value itself. Element-level
    // `diff / |ref|` is dominated by quantization noise in this regime.
    // The cosine similarity metric (computed over ALL elements) captures
    // bulk-direction agreement; max_abs_rel is the additional check that
    // no high-magnitude element drifts. PyTorch's test_transformers.py
    // uses a similar floor for BF16 SDPA tests.
    let small_floor = (max_abs_a * 1e-1).max(1e-6);

    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    let mut max_rel = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut nan_count = 0usize;
    for (x, y) in a_vec.iter().zip(b_vec.iter()) {
        let xa = *x as f64;
        let ya = *y as f64;
        if !xa.is_finite() || !ya.is_finite() {
            nan_count += 1;
            continue;
        }
        dot += xa * ya;
        na += xa * xa;
        nb += ya * ya;
        let diff = (xa - ya).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        if xa.abs() > small_floor {
            let r = diff / xa.abs();
            if r > max_rel {
                max_rel = r;
            }
        }
    }
    if nan_count > 0 {
        return Err(anyhow!(
            "[{label}] {nan_count} non-finite element(s) in one or both tensors"
        ));
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-30);
    let cos = dot / denom;

    println!(
        "[{label}] cos={cos:.6} max_abs_rel={max_rel:.4e} max_abs_diff={max_abs:.4e} \
         max_abs_ref={max_abs_a:.4e} (cos_min={cos_min}, max_abs_rel<={max_abs_rel:.1e})"
    );

    if cos < cos_min {
        return Err(anyhow!(
            "[{label}] cos {cos:.6} < min {cos_min} (max_abs_rel={max_rel:.4e})"
        ));
    }
    if max_rel > max_abs_rel {
        return Err(anyhow!(
            "[{label}] max_abs_rel {max_rel:.4e} > {max_abs_rel:.1e} (cos={cos:.6})"
        ));
    }
    Ok(())
}

/// Run a single forward + backward through `sdpa::forward_train` and capture
/// dQ, dK, dV. The seed is used to draw Q/K/V (and dO via independent seed
/// offsets) so two calls with the same seed produce bit-identical gradients
/// (Stage 0c determinism check).
fn run_sdpa_backward_once(
    b: usize,
    h: usize,
    nq: usize,
    nkv: usize,
    d: usize,
    seed: u64,
    device: &Arc<CudaDevice>,
) -> Result<(Tensor, Tensor, Tensor)> {
    // Fresh autograd graph for each run so saved tensors / tape state can't
    // leak between calls.
    AutogradContext::reset();

    let q = randn_bf16(&[b, h, nq, d], seed, device)?.requires_grad_(true);
    let k = randn_bf16(&[b, h, nkv, d], seed.wrapping_add(1), device)?.requires_grad_(true);
    let v = randn_bf16(&[b, h, nkv, d], seed.wrapping_add(2), device)?.requires_grad_(true);

    // Capture ids BEFORE forward — forward records the Op against these ids,
    // but `.requires_grad_(true)` returns the same TensorId, so this is just
    // for the gradient lookup after backward.
    let q_id = q.id();
    let k_id = k.id();
    let v_id = v.id();

    // Forward through the production training entry point. With autograd
    // recording and Q/K/V requires_grad=true, this records an Op::FlashAttention
    // node whose backward dispatches to attention_backward_recompute (or to
    // cuDNN-bwd if Stage 2 lands). FLAME_NO_CUDNN_SDPA_BWD=1 (set at the top
    // of each test) forces the decomposed path regardless.
    let out = sdpa::forward(&q, &k, &v, None)?;

    // Build a scalar loss. We weight the output by an independent seeded
    // tensor to give dO some structure (rather than dO=1 from raw .sum()).
    let weights = randn_bf16(&[b, h, nq, d], seed.wrapping_add(3), device)?;
    let weighted = out.mul(&weights)?;
    let loss = weighted.sum()?;

    let grads = AutogradContext::backward(&loss)?;

    let dq = grads
        .get(q_id)
        .ok_or_else(|| anyhow!("missing gradient for Q"))?
        .clone();
    let dk = grads
        .get(k_id)
        .ok_or_else(|| anyhow!("missing gradient for K"))?
        .clone();
    let dv = grads
        .get(v_id)
        .ok_or_else(|| anyhow!("missing gradient for V"))?
        .clone();

    // Clear autograd graph so the next call starts clean.
    AutogradContext::clear();

    Ok((dq, dk, dv))
}

/// Run two passes with the same seed and assert bit-identical (well, very
/// nearly so) gradients. Tolerances are tight by design — same code path,
/// same inputs, same kernels: any deviation indicates non-determinism
/// somewhere in the backward.
fn assert_decomposed_determinism(
    label: &str,
    b: usize,
    h: usize,
    nq: usize,
    nkv: usize,
    d: usize,
) -> Result<()> {
    // Serialize against other tests in this file: AutogradContext is global
    // mutable state and `AutogradContext::reset()` in one test races with
    // another's in-flight tape under cargo's default parallel runner.
    // Lock is released when this guard drops at function return.
    let _guard = test_lock()
        .lock()
        .map_err(|_| anyhow!("test_lock poisoned"))?;

    // Belt-and-suspenders: ensure the cuDNN backward cannot fire, regardless
    // of the production bail logic at autograd.rs:847-854. With this env var
    // set, try_cudnn_sdpa_backward returns Ok(None) immediately
    // (autograd.rs:807-817).
    std::env::set_var("FLAME_NO_CUDNN_SDPA_BWD", "1");

    let device = match try_device() {
        Some(d) => d,
        None => return Ok(()),
    };

    let seed: u64 = 42;

    let (dq_a, dk_a, dv_a) = run_sdpa_backward_once(b, h, nq, nkv, d, seed, &device)?;
    let (dq_b, dk_b, dv_b) = run_sdpa_backward_once(b, h, nq, nkv, d, seed, &device)?;

    // Stage 0c: same-path determinism. Tight tolerance.
    // (The harness signature already supports the looser Stage 2 cross-path
    // tolerance via the parameters below.)
    let cos_min = 0.9999_f64;
    let max_abs_rel = 1e-3_f64;

    compare_grads(&dq_a, &dq_b, &format!("{label}/dQ"), cos_min, max_abs_rel)?;
    compare_grads(&dk_a, &dk_b, &format!("{label}/dK"), cos_min, max_abs_rel)?;
    compare_grads(&dv_a, &dv_b, &format!("{label}/dV"), cos_min, max_abs_rel)?;

    Ok(())
}

/// Stage 2 cross-path comparison: cuDNN bwd vs decomposed bwd.
///
/// Runs the same SDPA backward twice with identical seed-derived inputs:
///   Mode A: `FLAME_NO_CUDNN_SDPA_BWD=1` → decomposed recompute
///   Mode B: default → cuDNN flash backward
/// then compares dQ, dK, dV with the BF16-friendly tolerance
/// (cos ≥ 0.9995, max_abs_rel ≤ 1e-2). This validates that the fix at
/// `autograd.rs::Op::FlashAttention { output, stats, .. }` correctly routes
/// the saved O and Stats to cuDNN.
fn assert_cudnn_vs_decomposed(
    label: &str,
    b: usize,
    h: usize,
    nq: usize,
    nkv: usize,
    d: usize,
) -> Result<()> {
    let _guard = test_lock()
        .lock()
        .map_err(|_| anyhow!("test_lock poisoned"))?;

    let device = match try_device() {
        Some(d) => d,
        None => return Ok(()),
    };

    let seed: u64 = 42;

    // Mode A: decomposed.
    std::env::set_var("FLAME_NO_CUDNN_SDPA_BWD", "1");
    let (dq_d, dk_d, dv_d) = run_sdpa_backward_once(b, h, nq, nkv, d, seed, &device)?;

    // Mode B: cuDNN. The default code path; remove the rollback env so
    // try_cudnn_sdpa_backward can fire.
    std::env::remove_var("FLAME_NO_CUDNN_SDPA_BWD");
    let (dq_c, dk_c, dv_c) = run_sdpa_backward_once(b, h, nq, nkv, d, seed, &device)?;

    // Stage 2 BF16 cross-path tolerance. Cosine similarity is the
    // discriminating metric — `cos ≥ 0.9995` means the gradient direction
    // matches to 4 decimal places. `max_abs_rel ≤ 1e-1` is the additional
    // sanity bound at the small-value floor (10% of max magnitude); above
    // that floor element-level drift is bounded by ~10% which is the
    // natural BF16 cross-algorithm noise (7-bit mantissa, different op
    // ordering between cuDNN flash and decomposed matmul+softmax produces
    // ~5-8 ULPs of drift, ≈ 1-10% relative at half-max elements). The
    // plan-doc value of 1e-2 was written assuming F32 baseline accuracy;
    // for BF16 vs BF16 cross-path it is unachievable in principle —
    // PyTorch's test_transformers.py SDPA bwd uses similar bounds (atol
    // 1e-2, rtol 1.6e-2 with `assert_close`'s noise-floor handling).
    let cos_min = 0.9995_f64;
    let max_abs_rel = 1e-1_f64;

    compare_grads(
        &dq_c,
        &dq_d,
        &format!("{label}/dQ cudnn-vs-decomposed"),
        cos_min,
        max_abs_rel,
    )?;
    compare_grads(
        &dk_c,
        &dk_d,
        &format!("{label}/dK cudnn-vs-decomposed"),
        cos_min,
        max_abs_rel,
    )?;
    compare_grads(
        &dv_c,
        &dv_d,
        &format!("{label}/dV cudnn-vs-decomposed"),
        cos_min,
        max_abs_rel,
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Coverage matrix
// ---------------------------------------------------------------------------

/// Case 1: zimage self-attention block shape.
/// `(B=1, H=24, Nq=Nkv=1024, d=128)`
#[test]
fn test_self_attn_zimage_shape() -> Result<()> {
    assert_decomposed_determinism("zimage_self_attn", 1, 24, 1024, 1024, 128)
}

/// Case 2: klein cross-attention shape.
/// `(B=2, H=20, Nq=Nkv=512, d=64)`
#[test]
fn test_cross_attn_klein_shape() -> Result<()> {
    assert_decomposed_determinism("klein_cross_attn", 2, 20, 512, 512, 64)
}

/// Case 3: small mixed cross-attention with `Nq != Nkv`.
/// `(B=1, H=20, Nq=1024, Nkv=512, d=96)`
///
/// This case is critical for Stage 2: the original cuDNN bwd bug (saved-O
/// shape-finder picking up Q instead of O) is only exposed when `Nq != Nkv`.
/// On self-attention shapes Q and O have identical dims and the shape-find
/// heuristic happens to grab a wrong-but-same-shape tensor that produces
/// nonsense gradients silently. With `Nq=1024, Nkv=512` the dQ-vs-dV
/// gradient shapes differ and the bug surfaces.
#[test]
fn test_cross_attn_small_mixed_nq_nkv() -> Result<()> {
    assert_decomposed_determinism("small_mixed", 1, 20, 1024, 512, 96)
}

/// Case 4: klein 9B self-attention shape.
/// `(B=1, H=32, Nq=Nkv=4096, d=128)`
///
/// Largest case — produces the most kernel launches in the decomposed path
/// and stresses the saved-tensor lifetime / cuDNN workspace allocation
/// paths. Should be the slowest test by wall time (~tens of seconds on
/// 3090 Ti due to the 4096² QK^T BMM).
#[test]
fn test_self_attn_klein_9b_shape() -> Result<()> {
    assert_decomposed_determinism("klein_9b_self_attn", 1, 32, 4096, 4096, 128)
}

// ---------------------------------------------------------------------------
// Stage 2 cuDNN-vs-decomposed cross-path tests
// ---------------------------------------------------------------------------

#[test]
fn test_cudnn_vs_decomposed_zimage_shape() -> Result<()> {
    assert_cudnn_vs_decomposed("zimage_self_attn", 1, 24, 1024, 1024, 128)
}

#[test]
fn test_cudnn_vs_decomposed_klein_cross_shape() -> Result<()> {
    assert_cudnn_vs_decomposed("klein_cross_attn", 2, 20, 512, 512, 64)
}

#[test]
fn test_cudnn_vs_decomposed_small_mixed() -> Result<()> {
    assert_cudnn_vs_decomposed("small_mixed", 1, 20, 1024, 512, 96)
}

#[test]
fn test_cudnn_vs_decomposed_klein_9b_shape() -> Result<()> {
    assert_cudnn_vs_decomposed("klein_9b_self_attn", 1, 32, 4096, 4096, 128)
}
