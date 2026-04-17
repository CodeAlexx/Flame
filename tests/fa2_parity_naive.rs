#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

//! Phase-1.5 naive-reference parity test: the new FA2 forward kernel vs. a
//! pure-Rust materialized attention reference computed entirely in FP32.
//!
//! Why this exists: `fa2_parity.rs` compares the new FA2 against the legacy
//! BQ=32 WMMA kernel. Both share BKV=64 K-accumulation order, so a bit-exact
//! match there proves only that they do the same math in the same order —
//! if both had the same bug the test would pass and prove nothing.
//! `fa2_parity_torch.rs` brings in libtorch as a third reference, but
//! libtorch fails to load in this env due to a CUDA 12.8 symbol mismatch.
//!
//! This test is the FP32-independent reference. No tiling, no BF16 accum,
//! no shared code with FA2's online softmax. The algorithm is:
//!   1. Receive BF16 Q, K, V on CUDA (same inputs FA2 gets).
//!   2. Upcast Q, K, V to FP32 on CUDA.
//!   3. scores = Q @ K^T in FP32.
//!   4. Multiply by 1.0 / sqrt(head_dim) in FP32.
//!   5. attn = softmax(scores, dim=-1) in FP32.
//!   6. O = attn @ V in FP32.
//!   7. Downcast O to BF16 at the end.
//!
//! The whole point is that FP32 throughout means the reference does NOT
//! share BF16 rounding error with FA2. We gate on two metrics:
//!   * max_abs ≤ 1e-2 — lives at the BF16 noise floor (BF16 eps ≈ 2^-7 ≈ 7.8e-3).
//!   * cos_sim  ≥ 0.9999 — direction match of the flattened output vectors.
//! A per-element max_rel gate is intentionally NOT used: near-zero elements
//! one BF16 ULP apart give ~7.6e-3 relative error by construction, which
//! asks for finer-than-BF16 precision and fails spuriously. Cosine similarity
//! is the right direction-preservation check at this precision.
//!
//! Shape matrix: N ∈ {512, 4096}, HD ∈ {64, 128}, H=8, B=1 (4 configs).
//! The N=16384 config is in a separate `#[ignore]`d test because the
//! materialized FP32 scores tensor is ~4 GB at that size.

use anyhow::Result;
use flame_core::{DType, Device, Shape, Tensor};

/// Launch the new FA2 forward kernel directly via FFI. Identical to the
/// helpers in the sibling parity tests.
fn launch_fa2(q_3d: &Tensor, k_3d: &Tensor, v_3d: &Tensor) -> Result<Tensor> {
    let dims = q_3d.shape().dims();
    let bh = dims[0] as i32;
    let sq = dims[1] as i32;
    let hd = dims[2] as i32;
    let sk = k_3d.shape().dims()[1] as i32;

    let out = Tensor::empty_dtype(
        Shape::from_dims(&[dims[0], dims[1], dims[2]]),
        DType::BF16,
        q_3d.device().clone(),
    )?;

    let q_ptr = q_3d.as_device_ptr_bf16("fa2_parity_naive:q")? as *const core::ffi::c_void;
    let k_ptr = k_3d.as_device_ptr_bf16("fa2_parity_naive:k")? as *const core::ffi::c_void;
    let v_ptr = v_3d.as_device_ptr_bf16("fa2_parity_naive:v")? as *const core::ffi::c_void;
    let o_ptr = out.as_device_ptr_bf16("fa2_parity_naive:o")? as *mut core::ffi::c_void;

    let stream = flame_core::cuda::device_lt::stream_ptr(q_3d.device())
        .map_err(|e| anyhow::anyhow!("stream_ptr: {e:?}"))?;
    let ret = unsafe {
        flame_core::cuda::ffi::flame_flash_attention_bf16(
            q_ptr,
            k_ptr,
            v_ptr,
            o_ptr,
            core::ptr::null_mut(),
            bh,
            sq,
            sk,
            hd,
            stream,
        )
    };
    if ret != 0 {
        anyhow::bail!("FA2 kernel returned nonzero: {ret}");
    }
    q_3d.device()
        .synchronize()
        .map_err(|e| anyhow::anyhow!("device synchronize: {e:?}"))?;
    Ok(out)
}

/// Naive materialized attention reference. FP32 throughout. Input tensors
/// are BF16 `[bh, N, D]`; the output is BF16 `[bh, N, D]` after a final cast.
///
/// NOTE: this materializes a full `[bh, N, N]` FP32 scores tensor. At
/// `bh=8, N=16384` that's 8 GiB. This is intentional — no tiling is the
/// whole point: it shares nothing with FA2's online softmax.
fn naive_attn_fp32(
    q_bf16: &Tensor,
    k_bf16: &Tensor,
    v_bf16: &Tensor,
) -> Result<Tensor> {
    let dims = q_bf16.shape().dims();
    let hd = dims[2];
    let scale = 1.0f32 / (hd as f32).sqrt();

    // Upcast all three inputs to FP32.
    let q_f32 = q_bf16.to_dtype(DType::F32)?;
    let k_f32 = k_bf16.to_dtype(DType::F32)?;
    let v_f32 = v_bf16.to_dtype(DType::F32)?;

    // scores = Q @ K^T : [bh, N, D] @ [bh, D, N] -> [bh, N, N]
    // transpose_dims(1, 2) on a 3D tensor triggers the permute_021 GPU path,
    // which produces a contiguous F32 tensor (see tensor.rs permute()).
    let k_t = k_f32.transpose_dims(1, 2)?;
    let scores = q_f32.bmm(&k_t)?;

    // Multiply by scale in FP32.
    let scaled = scores.mul_scalar(scale)?;

    // Softmax along last dim, done MANUALLY in FP32 to avoid
    // `Tensor::softmax`'s default-dtype coercion at the end of the generic
    // path (which casts the F32 output back to BF16 when default_dtype is
    // BF16, as is typical in this repo). Pipeline:
    //   m  = max(scaled, dim=-1, keepdim=true)   (F32 -> F32)
    //   s' = scaled - m                          (F32, broadcast last dim)
    //   e  = exp(s')                             (F32 -> F32)
    //   z  = sum(e, dim=-1, keepdim=true)        (F32 -> F32)
    //   a  = e / z                               (F32, broadcast last dim)
    // All steps stay in F32 because `max_dim`/`sum_dim_keepdim`/`exp`/`div`
    // on F32 inputs preserve dtype.
    let last_dim = scaled.shape().dims().len() - 1;
    let max_keep = scaled.max_dim(last_dim, true)?;
    debug_assert_eq!(max_keep.dtype(), DType::F32);
    let shifted = scaled.sub(&max_keep)?;
    let exp_vals = shifted.exp()?;
    let sum_keep = exp_vals.sum_dim_keepdim(last_dim)?;
    debug_assert_eq!(sum_keep.dtype(), DType::F32);
    let attn = exp_vals.div(&sum_keep)?;
    debug_assert_eq!(attn.dtype(), DType::F32);

    // O = attn @ V : [bh, N, N] @ [bh, N, D] -> [bh, N, D]
    let out_f32 = attn.bmm(&v_f32)?;

    // Final downcast to BF16.
    let out_bf16 = out_f32.to_dtype(DType::BF16)?;
    Ok(out_bf16)
}

#[derive(Debug)]
struct DiffReport {
    seq_len: usize,
    head_dim: usize,
    max_abs: f32,
    /// Cosine similarity between the flattened FA2 and naive outputs, both
    /// upcast to F32 for the dot/norm computation. Gated at ≥ 0.9999.
    cos_sim: f32,
    passed: bool,
    skipped: Option<String>,
    naive_stats: Option<(f32, f32, f32, f32)>, // mean, std, min, max
}

/// Return the estimated bytes for the largest transient FP32 tensor we'll
/// hold live during the naive path: the `[bh, N, N]` scores tensor. We
/// actually hold several transients (q_f32, k_f32, v_f32, k_t, scores,
/// scaled, attn, out_f32) simultaneously; scores/scaled/attn dominate at
/// `4 * bh * N * N` bytes each.
fn estimate_naive_peak_bytes(bh: usize, n: usize, d: usize) -> usize {
    let scores_bytes = 4usize * bh * n * n;
    let qkv_bytes = 3 * 4 * bh * n * d;
    // scores + scaled + attn can coexist briefly; out_f32 is also f32 nd.
    scores_bytes * 3 + qkv_bytes + 4 * bh * n * d
}

fn run_case(
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    abs_tol: f32,
    cos_tol: f32,
    oom_budget_bytes: usize,
) -> Result<DiffReport> {
    let bh = num_heads;
    let est = estimate_naive_peak_bytes(bh, seq_len, head_dim);
    if est > oom_budget_bytes {
        return Ok(DiffReport {
            seq_len,
            head_dim,
            max_abs: 0.0,
            cos_sim: 0.0,
            passed: false,
            skipped: Some(format!(
                "estimated {} MiB FP32 transients > {} MiB budget",
                est >> 20,
                oom_budget_bytes >> 20
            )),
            naive_stats: None,
        });
    }

    // Deterministic inputs so reruns are reproducible.
    flame_core::rng::set_seed(0xFA2_0_0_0_CA_FEu64 ^ (seq_len as u64) ^ ((head_dim as u64) << 16))
        .map_err(|e| anyhow::anyhow!("set_seed: {e:?}"))?;

    // Build BF16 inputs. Both paths see the SAME bytes.
    let q = Tensor::randn(Shape::from_dims(&[bh, seq_len, head_dim]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;
    let k = Tensor::randn(Shape::from_dims(&[bh, seq_len, head_dim]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;
    let v = Tensor::randn(Shape::from_dims(&[bh, seq_len, head_dim]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;

    // FA2 forward.
    let out_fa2 = launch_fa2(&q, &k, &v)?;

    // Naive FP32 reference. If this OOMs the CUDA alloc path returns an
    // error; surface it as a skip rather than a failure for transparency.
    let out_naive = match naive_attn_fp32(&q, &k, &v) {
        Ok(t) => t,
        Err(e) => {
            let msg = format!("{e}");
            if msg.contains("out of memory")
                || msg.contains("OOM")
                || msg.contains("CUDA_ERROR_OUT_OF_MEMORY")
                || msg.contains("cudaErrorMemoryAllocation")
            {
                return Ok(DiffReport {
                    seq_len,
                    head_dim,
                    max_abs: 0.0,
                    cos_sim: 0.0,
                    passed: false,
                    skipped: Some(format!("CUDA OOM: {msg}")),
                    naive_stats: None,
                });
            }
            return Err(e);
        }
    };

    let fa2_f32 = out_fa2.to_dtype(DType::F32)?.to_vec_f32()?;
    let naive_f32 = out_naive.to_dtype(DType::F32)?.to_vec_f32()?;
    assert_eq!(
        fa2_f32.len(),
        naive_f32.len(),
        "FA2 vs naive output length mismatch: {} vs {}",
        fa2_f32.len(),
        naive_f32.len()
    );

    // Stats on the naive output itself — confirms it's not degenerate.
    let mut n_min = f32::INFINITY;
    let mut n_max = f32::NEG_INFINITY;
    let mut n_sum = 0f64;
    let mut n_sq = 0f64;
    for &x in &naive_f32 {
        if x < n_min {
            n_min = x;
        }
        if x > n_max {
            n_max = x;
        }
        n_sum += x as f64;
        n_sq += (x as f64) * (x as f64);
    }
    let count = naive_f32.len() as f64;
    let mean = (n_sum / count) as f32;
    let var = (n_sq / count - (n_sum / count).powi(2)).max(0.0) as f32;
    let std = var.sqrt();

    // We gate on two metrics:
    //   * max_abs — live at the BF16 noise floor. A per-element bound.
    //   * cos_sim — direction match of the full flattened output vectors.
    // A per-element max_rel was removed: near-zero elements one BF16 ULP
    // apart give ~7.6e-3 relative error by construction, which asks for
    // finer-than-BF16 precision. The f64 accumulators below keep the dot
    // and norms numerically stable across ~3e7 element products.
    let mut max_abs = 0f32;
    let mut nan_inf = 0usize;
    let mut dot_ab = 0f64;
    let mut norm_a_sq = 0f64;
    let mut norm_b_sq = 0f64;
    for (a, b) in fa2_f32.iter().zip(naive_f32.iter()) {
        if !a.is_finite() || !b.is_finite() {
            nan_inf += 1;
            continue;
        }
        let diff = (a - b).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        let af = *a as f64;
        let bf = *b as f64;
        dot_ab += af * bf;
        norm_a_sq += af * af;
        norm_b_sq += bf * bf;
    }
    assert_eq!(
        nan_inf, 0,
        "non-finite outputs: fa2 or naive produced NaN/Inf"
    );

    // cos_sim = dot(a, b) / (||a|| * ||b|| + eps). eps = 1e-12 guards only
    // against the degenerate all-zero case; for any real attention output
    // the norms are >> 1.
    let eps = 1e-12f64;
    let cos_sim = (dot_ab / (norm_a_sq.sqrt() * norm_b_sq.sqrt() + eps)) as f32;

    let passed = max_abs <= abs_tol && cos_sim >= cos_tol;
    Ok(DiffReport {
        seq_len,
        head_dim,
        max_abs,
        cos_sim,
        passed,
        skipped: None,
        naive_stats: Some((mean, std, n_min, n_max)),
    })
}

fn print_and_assert(reports: &[DiffReport], abs_tol: f32, cos_tol: f32) -> Result<()> {
    eprintln!("FA2 vs FP32-naive parity:");
    for r in reports {
        if let Some(reason) = &r.skipped {
            eprintln!(
                "  N={:<5} HD={:<3}  SKIP: {}",
                r.seq_len, r.head_dim, reason
            );
            continue;
        }
        let status = if r.passed { "PASS" } else { "FAIL" };
        eprintln!(
            "  N={:<5} HD={:<3}  max_abs={:.3e}  cos_sim={:.6}  {status}",
            r.seq_len,
            r.head_dim,
            r.max_abs,
            r.cos_sim,
            status = status,
        );
    }
    // Emit naive-output stats for the first non-skipped config so we have
    // evidence the reference isn't degenerate (all-zero, all-NaN, etc.).
    if let Some(r) = reports.iter().find(|r| r.skipped.is_none()) {
        if let Some((mean, std, mn, mx)) = r.naive_stats {
            eprintln!(
                "  naive-output stats @ N={} HD={}: mean={:.3e} std={:.3e} min={:.3e} max={:.3e}",
                r.seq_len, r.head_dim, mean, std, mn, mx
            );
        }
    }

    let any_fail = reports
        .iter()
        .any(|r| r.skipped.is_none() && !r.passed);
    if any_fail {
        anyhow::bail!(
            "FA2 vs FP32-naive parity failed (tol: abs ≤ {abs_tol:.1e}, cos_sim ≥ {cos_tol:.4}). \
             See table above for per-config results."
        );
    }
    Ok(())
}

/// Small-matrix configs: N ∈ {512, 4096} × HD ∈ {64, 128}. Runs by default.
#[test]
fn fa2_matches_fp32_naive_small() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(err) => {
            eprintln!("[skip] CUDA unavailable: {err}");
            return Ok(());
        }
    };
    let arc = device.cuda_device().clone();

    const ABS_TOL: f32 = 1e-2;
    const COS_TOL: f32 = 0.9999;
    // 6 GiB transient budget — fits N=4096, bh=8 (scores = 4*8*4096^2 = 512 MiB).
    const BUDGET: usize = 6 * 1024 * 1024 * 1024;

    let seq_lens = [512usize, 4096];
    let head_dims = [64usize, 128];
    let num_heads = 8usize;

    let mut reports: Vec<DiffReport> = Vec::with_capacity(seq_lens.len() * head_dims.len());
    for &n in &seq_lens {
        for &d in &head_dims {
            reports.push(run_case(&arc, num_heads, n, d, ABS_TOL, COS_TOL, BUDGET)?);
        }
    }

    print_and_assert(&reports, ABS_TOL, COS_TOL)
}

/// N=16384 configs. Materialized FP32 scores at N=16384, bh=8 is ~4 GiB
/// and the full peak (3× scores + q/k/v/out) is ~13 GiB. Ignored by
/// default — run manually with `--ignored` on boxes with ≥16 GiB free.
#[test]
#[ignore = "N=16384 naive FP32 attention needs ~13 GiB VRAM"]
fn fa2_matches_fp32_naive_n16384() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(err) => {
            eprintln!("[skip] CUDA unavailable: {err}");
            return Ok(());
        }
    };
    let arc = device.cuda_device().clone();

    const ABS_TOL: f32 = 1e-2;
    const COS_TOL: f32 = 0.9999;
    // 20 GiB — should fit on a 24 GiB card with headroom.
    const BUDGET: usize = 20 * 1024 * 1024 * 1024;

    let head_dims = [64usize, 128];
    let num_heads = 8usize;
    let n = 16384usize;

    let mut reports: Vec<DiffReport> = Vec::with_capacity(head_dims.len());
    for &d in &head_dims {
        reports.push(run_case(&arc, num_heads, n, d, ABS_TOL, COS_TOL, BUDGET)?);
    }

    print_and_assert(&reports, ABS_TOL, COS_TOL)
}
