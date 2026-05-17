//! Async global gradient L2 norm + clipping.
//!
//! Fusion Sprint Phase 5 production helper.
//!
//! ## Why this exists
//!
//! The naive per-tensor pattern in EriDiffusion-v2 trainers does
//!
//! ```ignore
//! let mut total_sq = 0.0_f32;
//! for g in &grads {
//!     let sq = g.to_dtype(F32)?.square()?.mean()?;
//!     total_sq += sq.to_vec()?[0] * n_elem;   // ← D2H sync per tensor
//! }
//! ```
//!
//! On Klein 9B LoRA with ~200 LoRA tensors that's 200 D2H syncs every
//! optimizer step. Each sync stalls the GPU for the next-already-launched
//! kernel and drives a 3-8% E2E penalty depending on what else is in flight.
//!
//! [`global_l2_norm`] keeps everything on device — one launch per gradient
//! to compute its sum-of-squares, a sequence of binary adds for the
//! global reduction, and a single `sqrt`. The result is a 1-element FP32
//! device tensor; the caller chooses when (if ever) to `.item()` for a
//! host-side norm value.
//!
//! ## Reference
//!
//! Apex's `csrc/multi_tensor_l2norm_kernel.cu` (BSD-3) implements this as
//! a two-stage reduction (per-tensor partial sums → single-block reduction).
//! That kernel is faster still — single launch over all tensors — but
//! requires a custom multi-tensor apply harness. This implementation uses
//! flame-core's existing per-tensor reduce ops, trading one launch per
//! tensor for "no new CUDA kernel needed". Most of the win (avoiding N
//! D2H syncs) lands here; the remaining latency from N launches can be
//! reclaimed with Apex's pattern in a follow-up phase.
//!
//! ## Numerical contract
//!
//! - Computation is FP32 throughout (square in FP32, sum in FP32, sqrt in
//!   FP32). BF16 grads are cast on the fly; F32 grads stay F32.
//! - Mixed-dtype slices are supported.
//! - Empty slice returns a 1-element FP32 zero tensor on the supplied
//!   device (caller must provide the device for the empty case to avoid
//!   a magic global lookup).

use crate::{global_cuda_device, DType, Error, Result, Shape, Tensor};
use std::sync::{Arc, Mutex};

// Phase 3: process-wide cache for the multi-tensor L2 norm primitives.
// Held behind a Mutex because `global_l2_norm` is a free function and
// callers may invoke it from any thread. Contention is per-step and
// trivial; allocator state is the only thing being protected.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
static MT_L2_CACHE: Mutex<crate::ops::multi_tensor::MultiTensorMetaCache> =
    Mutex::new(crate::ops::multi_tensor::MultiTensorMetaCache::new());

/// Compute the global L2 norm of a slice of gradient tensors.
///
/// Returns a 1-element FP32 device tensor. The caller can `.item()` the
/// result for a host f32 (one D2H sync), or chain it into further device-
/// side scaling without any sync.
///
/// All compute stays on device. Per-tensor work is `g.square().sum()`,
/// followed by a fold over the per-tensor scalars and a `sqrt`. There is
/// no `.to_vec()` / `.item()` inside this function.
///
/// Empty slice short-circuits to a 1-element FP32 zero tensor on the
/// global CUDA device.
pub fn global_l2_norm(grads: &[&Tensor]) -> Result<Tensor> {
    if grads.is_empty() {
        let dev = global_cuda_device();
        return Tensor::from_vec(vec![0.0_f32], Shape::from_dims(&[1]), dev);
    }

    // Multi-tensor fast path (Phase 3 of launch-storm refactor + Phase 4a
    // BF16 sibling). Collapses 2N+(N-1)+1 launches into 3 (stage1 +
    // stage2 + sqrt). Eligible iff every grad is contiguous AND every
    // grad shares a single dtype (all F32, or all BF16). Mixed-dtype
    // slices and non-contiguous tensors fall through to the per-tensor
    // path below.
    //
    // Phase 4a (Option A from `docs/BF16_GRAD_DECISION.md`): the BF16
    // sibling preserves BF16 grads BF16-through-clipping. F32 only
    // appears as `opmath_t` inside the kernel's sum-of-squares.
    //
    // Env override `FLAME_MT_L2NORM=0` forces legacy path for A/B testing.
    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    {
        let mt_disabled = std::env::var("FLAME_MT_L2NORM")
            .ok()
            .as_deref()
            .map(|v| matches!(v, "0" | "false" | "FALSE"))
            .unwrap_or(false);
        if !mt_disabled {
            let all_f32_contig = grads
                .iter()
                .all(|g| g.dtype() == DType::F32 && g.is_contiguous());
            let all_bf16_contig = grads
                .iter()
                .all(|g| g.dtype() == DType::BF16 && g.is_contiguous());
            if all_f32_contig {
                let mut cache = MT_L2_CACHE
                    .lock()
                    .map_err(|_| Error::Training("MT_L2_CACHE mutex poisoned".into()))?;
                let total_sq =
                    crate::ops::multi_tensor::multi_tensor_l2_norm_sq_f32(&mut cache, grads)?;
                return total_sq.sqrt();
            }
            if all_bf16_contig {
                let mut cache = MT_L2_CACHE
                    .lock()
                    .map_err(|_| Error::Training("MT_L2_CACHE mutex poisoned".into()))?;
                let total_sq =
                    crate::ops::multi_tensor::multi_tensor_l2_norm_sq_bf16(&mut cache, grads)?;
                return total_sq.sqrt();
            }
        }
    }

    let dev = grads[0].device().clone();

    // Per-tensor sum-of-squares in FP32. Each call is one or two kernel
    // launches (cast + square+sum or just square+sum); none touch the host.
    let mut sq_sums: Vec<Tensor> = Vec::with_capacity(grads.len());
    for g in grads {
        let g_f32 = if g.dtype() == DType::F32 {
            g.clone_result()?
        } else {
            g.to_dtype(DType::F32)?
        };
        // square() returns same shape; sum() reduces to a 1-element tensor.
        sq_sums.push(g_f32.square()?.sum()?);
    }

    // Reduce all per-tensor scalars on device. fold-add is N-1 launches
    // but each is a tiny scalar add — total latency is dominated by per-
    // tensor square+sum, not by the fold. Apex's two-stage kernel reduces
    // this to a single launch; multi-tensor fast path above does exactly
    // that.
    let zero = Tensor::from_vec(vec![0.0_f32], Shape::from_dims(&[1]), dev.clone())?;
    let total_sq = sq_sums.into_iter().try_fold(zero, |acc, s| {
        // s is shape [1] FP32; acc is shape [1] FP32. Direct add works.
        let s_flat = if s.shape().dims() == [1] {
            s
        } else {
            s.reshape(&[1])?
        };
        acc.add(&s_flat)
    })?;

    // sqrt → 1-element FP32. Stays on device.
    total_sq.sqrt()
}

/// Convenience: compute the global L2 norm and the clip-scale factor in
/// one pass. Returns `(norm_device_scalar, scale_device_scalar)`. The
/// scale is `min(max_norm / (norm + eps), 1.0)`.
///
/// Both returned tensors are 1-element FP32 device tensors. To get the
/// host-side norm for logging, do exactly one `.item()` call at the end.
pub fn global_l2_norm_with_scale(
    grads: &[&Tensor],
    max_norm: f32,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    let norm = global_l2_norm(grads)?;
    let dev = norm.device().clone();

    // scale = min(max_norm / (norm + eps), 1.0)
    let denom = norm.add_scalar(eps)?;
    // ratio = max_norm / denom = max_norm * (1/denom)
    let one_over_denom = denom.reciprocal()?;
    let ratio = one_over_denom.mul_scalar(max_norm)?;
    let one = Tensor::from_vec(vec![1.0_f32], Shape::from_dims(&[1]), dev)?;
    let scale = ratio.minimum(&one)?;

    Ok((norm, scale))
}

/// Re-export of `global_cuda_device` arc for callers that need it
/// when constructing the empty-slice fallback.
pub fn default_device() -> Arc<cudarc::driver::CudaDevice> {
    global_cuda_device()
}
