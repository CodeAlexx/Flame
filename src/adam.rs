//! Adam optimizer implementation
//!
//! # Decoupled weight decay — bug-prevention receipt
//!
//! All fused Adam kernels below implement DECOUPLED weight decay
//! (Loshchilov & Hutter, "Decoupled Weight Decay Regularization", 2017 —
//! this is what the rest of the world calls AdamW):
//!
//!     m = beta1 * m + (1 - beta1) * g
//!     v = beta2 * v + (1 - beta2) * g * g
//!     p = p - lr * m_hat / (sqrt(v_hat) + eps) - lr * wd * p
//!
//! Equivalent closed form: `param = (1 - lr*wd) * param - lr * m̂/(√v̂+ε)`.
//!
//! ## Why not L2 regularization into grad
//!
//! The pre-fused implementation did `grad += wd * param` and then ran the
//! usual Adam moments on that contaminated grad. For a param whose real
//! gradient signal is small compared to `wd*param` (which is EXACTLY the
//! case for a freshly-initialized LoRA A matrix whose B partner is still
//! zero), Adam's adaptive normalization `m̂ / (√v̂+ε)` collapses the step
//! to `~sign(param)`, and the update becomes a uniform `lr * sign(param)`
//! shrinkage per step regardless of element magnitude. Observed effect on
//! Klein 4B LoRA rank-16 at `lr=4e-4, wd=0.01`: `lora_A` total L2 dropped
//! from the ~50 Kaiming init to 0.85 at step 400 and 0.25 at step 800 —
//! the LoRA was being unlearned. Decoupled decay removes this runaway
//! because the term affects `param` directly, not the adaptive step.
//!
//! DO NOT fold `weight_decay` into the gradient before the moment updates
//! in any of the kernels below. The shape must stay:
//!   (1) moments from raw grad, (2) param step from moments, (3) wd on param.

use crate::{parameter::Parameter, DType, Error, Result, Tensor, TensorId};
use std::collections::HashMap;

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use cudarc::driver::{DevicePtr, DevicePtrMut};

// Re-exports for parity-test access. These are low-level launcher entry
// points for the fused Adam kernels — production code should use
// `Adam::step` / `AdamW::step` instead, which select the right path.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub use fused::{
    adam_fused_multi_tensor_step, adam_fused_step, adam_fused_step_f32, MultiTensorMetaCache,
};

// ---------------------------------------------------------------------------
// Fused Adam CUDA kernels (inline PTX, compiled on first use)
// ---------------------------------------------------------------------------

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
const CUDA_ADAM_FUSED_BF16: &str = r#"
#include <cuda_bf16.h>
// DECOUPLED weight decay (Loshchilov & Hutter, 2017 — AdamW):
//   m = beta1 * m + (1 - beta1) * g
//   v = beta2 * v + (1 - beta2) * g * g
//   p = p - lr * m_hat / (sqrt(v_hat) + eps) - lr * wd * p
// The weight-decay term is applied to `p` DIRECTLY after the Adam step.
// It must NOT be added into `g` before the moments — doing so makes
// wd contaminate the adaptive rate and causes a freshly-initialized
// LoRA `A` matrix (whose `B` partner is still zero, so the real grad
// is near-zero) to shrink at a uniform `lr * sign(p)` per step,
// regardless of magnitude. That bug destroyed klein-trainer LoRA_A
// training in April 2026.
extern "C" __global__ void adam_fused_bf16_kernel(
    __nv_bfloat16* __restrict__ param,
    const __nv_bfloat16* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = __bfloat162float(grad[idx]);
    float p = __bfloat162float(param[idx]);

    float mi = beta1 * m[idx] + (1.0f - beta1) * g;
    float vi = beta2 * v[idx] + (1.0f - beta2) * g * g;
    m[idx] = mi;
    v[idx] = vi;

    float m_hat = mi / bias_correction1;
    float v_hat = vi / bias_correction2;
    p -= lr * m_hat / (sqrtf(v_hat) + eps);
    if (weight_decay > 0.0f) {
        p -= lr * weight_decay * p;
    }

    param[idx] = __float2bfloat16(p);
}

extern "C" __global__ void adam_fused_f32grad_kernel(
    __nv_bfloat16* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = grad[idx];
    float p = __bfloat162float(param[idx]);

    float mi = beta1 * m[idx] + (1.0f - beta1) * g;
    float vi = beta2 * v[idx] + (1.0f - beta2) * g * g;
    m[idx] = mi;
    v[idx] = vi;

    float m_hat = mi / bias_correction1;
    float v_hat = vi / bias_correction2;
    p -= lr * m_hat / (sqrtf(v_hat) + eps);
    if (weight_decay > 0.0f) {
        p -= lr * weight_decay * p;
    }

    param[idx] = __float2bfloat16(p);
}
"#;

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
const CUDA_ADAM_FUSED_F32PARAM_F32GRAD: &str = r#"
// DECOUPLED weight decay (Loshchilov & Hutter, 2017 — AdamW):
//   m = beta1 * m + (1 - beta1) * g
//   v = beta2 * v + (1 - beta2) * g * g
//   p = p - lr * m_hat / (sqrt(v_hat) + eps) - lr * wd * p
// The weight-decay term is applied to `p` DIRECTLY after the Adam step.
// It must NOT be added into `g` before the moments — doing so makes
// wd contaminate the adaptive rate and causes a freshly-initialized
// LoRA `A` matrix (whose `B` partner is still zero, so the real grad
// is near-zero) to shrink at a uniform `lr * sign(p)` per step,
// regardless of magnitude. That bug destroyed klein-trainer LoRA_A
// training in April 2026.
extern "C" __global__ void adam_fused_f32param_f32grad_kernel(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = grad[idx];
    float p = param[idx];

    float mi = beta1 * m[idx] + (1.0f - beta1) * g;
    float vi = beta2 * v[idx] + (1.0f - beta2) * g * g;
    m[idx] = mi;
    v[idx] = vi;

    float m_hat = mi / bias_correction1;
    float v_hat = vi / bias_correction2;
    p -= lr * m_hat / (sqrtf(v_hat) + eps);
    if (weight_decay > 0.0f) {
        p -= lr * weight_decay * p;
    }

    param[idx] = p;
}
"#;

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
const CUDA_ADAM_FUSED_F32PARAM_BF16GRAD: &str = r#"
#include <cuda_bf16.h>
// DECOUPLED weight decay (Loshchilov & Hutter, 2017 — AdamW):
//   m = beta1 * m + (1 - beta1) * g
//   v = beta2 * v + (1 - beta2) * g * g
//   p = p - lr * m_hat / (sqrt(v_hat) + eps) - lr * wd * p
// The weight-decay term is applied to `p` DIRECTLY after the Adam step.
// It must NOT be added into `g` before the moments — doing so makes
// wd contaminate the adaptive rate and causes a freshly-initialized
// LoRA `A` matrix (whose `B` partner is still zero, so the real grad
// is near-zero) to shrink at a uniform `lr * sign(p)` per step,
// regardless of magnitude. That bug destroyed klein-trainer LoRA_A
// training in April 2026.
extern "C" __global__ void adam_fused_f32param_bf16grad_kernel(
    float* __restrict__ param,
    const __nv_bfloat16* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = __bfloat162float(grad[idx]);
    float p = param[idx];

    float mi = beta1 * m[idx] + (1.0f - beta1) * g;
    float vi = beta2 * v[idx] + (1.0f - beta2) * g * g;
    m[idx] = mi;
    v[idx] = vi;

    float m_hat = mi / bias_correction1;
    float v_hat = vi / bias_correction2;
    p -= lr * m_hat / (sqrtf(v_hat) + eps);
    if (weight_decay > 0.0f) {
        p -= lr * weight_decay * p;
    }

    param[idx] = p;
}
"#;

// Multi-tensor fused Adam (Fusion Sprint Phase 4 follow-up).
//
// Reference: NVIDIA Apex `csrc/multi_tensor_adam.cu` (BSD-3) and
// `csrc/multi_tensor_apply.cuh`. Apex's harness chunks tensors into ~110-per-launch
// blocks via a constant-memory metadata struct. We use a simpler "1 block per
// tensor + grid-strided loop within the block" model:
//
//   - gridDim.x = n_tensors
//   - blockDim.x = 256
//   - Each block handles its tensor's full element range via stride-256 walk.
//
// Pointer arrays + size array live on device — populated once per step from
// the host with a single H2D copy of `5 * n_tensors * 8` bytes (~8 KB for
// 200 LoRA tensors). The 5 arrays are packed contiguously into one allocation.
//
// This collapses N kernel launches (one per param) into 1, saving ~5 μs of
// launch overhead per param. For Klein 9B LoRA (~200 LoRA tensors per step)
// that's ~1 ms/step of pure overhead. Modest E2E impact (<0.2% on a 2 s step)
// but the helper is the right architecture going forward and matches the
// "multi-tensor reductions" rule in FLAME_CONVENTIONS.md.
//
// Same DECOUPLED-WD math as the single-tensor kernel — see the receipt at
// the top of the BF16 source above.
const CUDA_ADAM_FUSED_MULTI_BF16: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void adam_fused_multi_bf16_f32grad_kernel(
    __nv_bfloat16** const __restrict__ params,
    const float**     const __restrict__ grads,
    float**           const __restrict__ ms,
    float**           const __restrict__ vs,
    const long long*  __restrict__ sizes,
    int n_tensors,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2
) {
    int t = blockIdx.x;
    if (t >= n_tensors) return;

    long long n = sizes[t];
    __nv_bfloat16* p = params[t];
    const float*   g = grads[t];
    float*         m = ms[t];
    float*         v = vs[t];

    int tid    = threadIdx.x;
    int stride = blockDim.x;

    for (long long i = (long long)tid; i < n; i += (long long)stride) {
        float gv = g[i];
        float pv = __bfloat162float(p[i]);

        float mi = beta1 * m[i] + (1.0f - beta1) * gv;
        float vi = beta2 * v[i] + (1.0f - beta2) * gv * gv;
        m[i] = mi;
        v[i] = vi;

        float m_hat = mi / bias_correction1;
        float v_hat = vi / bias_correction2;
        pv -= lr * m_hat / (sqrtf(v_hat) + eps);
        if (weight_decay > 0.0f) {
            pv -= lr * weight_decay * pv;
        }
        p[i] = __float2bfloat16(pv);
    }
}

extern "C" __global__ void adam_fused_multi_f32param_f32grad_kernel(
    float**          const __restrict__ params,
    const float**    const __restrict__ grads,
    float**          const __restrict__ ms,
    float**          const __restrict__ vs,
    const long long* __restrict__ sizes,
    int n_tensors,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2
) {
    // Per-element math is identical to the single-tensor
    // `adam_fused_f32param_f32grad_kernel` (lines 147-180). The only
    // difference is the launch shape: this kernel uses one thread block
    // per tensor with a grid-stride inner loop, mirroring
    // `adam_fused_multi_bf16_f32grad_kernel` above. Since each thread
    // touches a distinct (t, i) pair, m[i] / v[i] / p[i] reads and writes
    // do not race, and the per-element ordering is identical to the
    // per-tensor kernel. Multi-tensor parity is therefore bit-identical
    // to single-tensor — see `tests/adam_multi_tensor_parity.rs`.
    int t = blockIdx.x;
    if (t >= n_tensors) return;

    long long n = sizes[t];
    float*       p = params[t];
    const float* g = grads[t];
    float*       m = ms[t];
    float*       v = vs[t];

    int tid    = threadIdx.x;
    int stride = blockDim.x;

    for (long long i = (long long)tid; i < n; i += (long long)stride) {
        float gv = g[i];
        float pv = p[i];

        float mi = beta1 * m[i] + (1.0f - beta1) * gv;
        float vi = beta2 * v[i] + (1.0f - beta2) * gv * gv;
        m[i] = mi;
        v[i] = vi;

        float m_hat = mi / bias_correction1;
        float v_hat = vi / bias_correction2;
        pv -= lr * m_hat / (sqrtf(v_hat) + eps);
        if (weight_decay > 0.0f) {
            pv -= lr * weight_decay * pv;
        }
        p[i] = pv;
    }
}

extern "C" __global__ void adam_fused_multi_bf16_bf16grad_kernel(
    __nv_bfloat16**         const __restrict__ params,
    const __nv_bfloat16**   const __restrict__ grads,
    float**                 const __restrict__ ms,
    float**                 const __restrict__ vs,
    const long long*        __restrict__ sizes,
    int n_tensors,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2
) {
    int t = blockIdx.x;
    if (t >= n_tensors) return;

    long long n = sizes[t];
    __nv_bfloat16*       p = params[t];
    const __nv_bfloat16* g = grads[t];
    float*               m = ms[t];
    float*               v = vs[t];

    int tid    = threadIdx.x;
    int stride = blockDim.x;

    for (long long i = (long long)tid; i < n; i += (long long)stride) {
        float gv = __bfloat162float(g[i]);
        float pv = __bfloat162float(p[i]);

        float mi = beta1 * m[i] + (1.0f - beta1) * gv;
        float vi = beta2 * v[i] + (1.0f - beta2) * gv * gv;
        m[i] = mi;
        v[i] = vi;

        float m_hat = mi / bias_correction1;
        float v_hat = vi / bias_correction2;
        pv -= lr * m_hat / (sqrtf(v_hat) + eps);
        if (weight_decay > 0.0f) {
            pv -= lr * weight_decay * pv;
        }
        p[i] = __float2bfloat16(pv);
    }
}
"#;

// Stochastic-rounding variants of the BF16 + F32-grad kernels (single +
// multi-tensor). Behaviorally identical to the round-to-nearest variants
// above, except the final F32 → BF16 store applies the same lower-16-bit
// hash-based stochastic rounding as the standalone
// `bf16_ops::bf16_stoch_round_kernel`. CPU reference:
// `bf16_convert::stochastic_round_to_bf16_cpu`.
//
// Per-element entropy is derived inside the kernel from `(seed, idx)` via
// splitmix64, so the caller only needs to pass a u64 seed (typically the
// step counter). For the multi-tensor kernel the seed is additionally mixed
// with the tensor index so two parameters at the same elementwise index do
// not stochastically round in lock-step.
//
// Stochastic rounding eliminates the systematic accumulator stalling that
// can occur in long-horizon BF16 training: when the post-Adam update for an
// element has magnitude smaller than `0.5 * ulp(BF16)`, round-to-nearest
// pins the result back to the same BF16 bucket every step and the live
// param never moves; stochastic rounding moves it with probability
// proportional to the F32 fractional remainder.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
const CUDA_ADAM_FUSED_BF16_STOCH: &str = r#"
#include <cuda_bf16.h>

__device__ inline unsigned short adam_stoch_round_f32_to_bf16(
    float p,
    unsigned long long seed,
    long long idx
) {
    unsigned int bits  = __float_as_uint(p);
    unsigned int lower = bits & 0xFFFFu;
    unsigned int upper = bits >> 16;
    // splitmix64 keyed on seed XOR (idx + golden ratio).
    unsigned long long s = seed ^ ((unsigned long long)idx + 0x9E3779B97F4A7C15ULL);
    s = (s ^ (s >> 30)) * 0xBF58476D1CE4E5B9ULL;
    s = (s ^ (s >> 27)) * 0x94D049BB133111EBULL;
    s = s ^ (s >> 31);
    unsigned int r = (unsigned int)(s & 0xFFFFu);
    if (r < lower) upper = (upper + 1u) & 0xFFFFu;
    return (unsigned short)upper;
}

extern "C" __global__ void adam_fused_bf16_f32grad_stoch_kernel(
    __nv_bfloat16* __restrict__ param,
    const float*   __restrict__ grad,
    float*         __restrict__ m,
    float*         __restrict__ v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    long long n,
    unsigned long long seed
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = grad[idx];
    float p = __bfloat162float(param[idx]);

    float mi = beta1 * m[idx] + (1.0f - beta1) * g;
    float vi = beta2 * v[idx] + (1.0f - beta2) * g * g;
    m[idx] = mi;
    v[idx] = vi;

    float m_hat = mi / bias_correction1;
    float v_hat = vi / bias_correction2;
    p -= lr * m_hat / (sqrtf(v_hat) + eps);
    if (weight_decay > 0.0f) {
        p -= lr * weight_decay * p;
    }

    unsigned short bf = adam_stoch_round_f32_to_bf16(p, seed, idx);
    ((unsigned short*)param)[idx] = bf;
}

extern "C" __global__ void adam_fused_multi_bf16_f32grad_stoch_kernel(
    __nv_bfloat16** const __restrict__ params,
    const float**     const __restrict__ grads,
    float**           const __restrict__ ms,
    float**           const __restrict__ vs,
    const long long*  __restrict__ sizes,
    int n_tensors,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    unsigned long long seed
) {
    int t = blockIdx.x;
    if (t >= n_tensors) return;
    long long n = sizes[t];
    __nv_bfloat16* p = params[t];
    const float*   g = grads[t];
    float*         m = ms[t];
    float*         v = vs[t];

    int tid    = threadIdx.x;
    int stride = blockDim.x;

    // Mix tensor index into the seed so two tensors do not stochastically
    // round in lock-step at the same elementwise idx.
    unsigned long long t_seed = seed ^ ((unsigned long long)t * 0x100000001B3ULL);

    for (long long i = (long long)tid; i < n; i += (long long)stride) {
        float gv = g[i];
        float pv = __bfloat162float(p[i]);

        float mi = beta1 * m[i] + (1.0f - beta1) * gv;
        float vi = beta2 * v[i] + (1.0f - beta2) * gv * gv;
        m[i] = mi;
        v[i] = vi;

        float m_hat = mi / bias_correction1;
        float v_hat = vi / bias_correction2;
        pv -= lr * m_hat / (sqrtf(v_hat) + eps);
        if (weight_decay > 0.0f) {
            pv -= lr * weight_decay * pv;
        }

        unsigned short bf = adam_stoch_round_f32_to_bf16(pv, t_seed, i);
        ((unsigned short*)p)[i] = bf;
    }
}
"#;

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
mod fused {
    use super::*;
    use cudarc::driver::{CudaDevice, DevicePtr, DevicePtrMut, LaunchAsync, LaunchConfig};
    use cudarc::nvrtc::CompileOptions;
    use std::sync::Arc;

    const MODULE_NAME: &str = "adam_fused";

    fn ensure_adam_kernels(device: &Arc<CudaDevice>) -> Result<()> {
        // Fast path: already loaded
        if device
            .get_func(MODULE_NAME, "adam_fused_bf16_kernel")
            .is_some()
        {
            return Ok(());
        }

        // Cold path: combine all single- and multi-tensor kernel sources into
        // a single translation unit and compile once. `<cuda_bf16.h>` has its
        // own include guard, so the repeated include across the bf16-touching
        // source constants is a no-op. Option A from the Phase 2 plan: one
        // NVRTC compile, one `load_ptx`, one `MODULE_NAME` — `get_func` is
        // O(1) regardless of which dtype combo hits at step time.
        let combined = format!(
            "{}\n{}\n{}\n{}\n{}",
            CUDA_ADAM_FUSED_BF16,
            CUDA_ADAM_FUSED_F32PARAM_F32GRAD,
            CUDA_ADAM_FUSED_F32PARAM_BF16GRAD,
            CUDA_ADAM_FUSED_MULTI_BF16,
            CUDA_ADAM_FUSED_BF16_STOCH,
        );

        let include_dir = std::env::var("CUDA_INCLUDE_DIR")
            .or_else(|_| std::env::var("CUDA_HOME").map(|home| format!("{home}/include")))
            .unwrap_or_else(|_| "/usr/local/cuda/include".into());
        let mut opts = CompileOptions::default();
        opts.include_paths.push(include_dir);
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(&combined, opts)
            .map_err(|e| Error::Cuda(format!("nvrtc adam_fused: {:?}", e)))?;
        device
            .load_ptx(
                ptx,
                MODULE_NAME,
                &[
                    "adam_fused_bf16_kernel",
                    "adam_fused_f32grad_kernel",
                    "adam_fused_f32param_f32grad_kernel",
                    "adam_fused_f32param_bf16grad_kernel",
                    "adam_fused_multi_bf16_f32grad_kernel",
                    "adam_fused_multi_f32param_f32grad_kernel",
                    "adam_fused_multi_bf16_bf16grad_kernel",
                    "adam_fused_bf16_f32grad_stoch_kernel",
                    "adam_fused_multi_bf16_f32grad_stoch_kernel",
                ],
            )
            .map_err(|e| Error::Cuda(format!("load adam_fused: {:?}", e)))?;
        Ok(())
    }

    /// Launch fused Adam update for BF16 parameters — single kernel, no temporaries.
    ///
    /// `param` must be BF16, `m` and `v` must be F32, `grad` can be BF16 or F32.
    /// All tensors are modified in-place (param, m, v).
    pub fn adam_fused_step(
        param: &mut Tensor,
        grad: &Tensor,
        m: &mut Tensor,
        v: &mut Tensor,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        bias_correction1: f32,
        bias_correction2: f32,
        stoch_seed: Option<u64>,
    ) -> Result<()> {
        // Validate dtypes once at entry
        debug_assert_eq!(param.dtype(), DType::BF16);
        debug_assert_eq!(m.dtype(), DType::F32);
        debug_assert_eq!(v.dtype(), DType::F32);

        let n = param.shape().elem_count();
        debug_assert_eq!(n, grad.shape().elem_count());
        debug_assert_eq!(n, m.shape().elem_count());
        debug_assert_eq!(n, v.shape().elem_count());

        let device = param.device.clone();
        ensure_adam_kernels(&device)?;

        let grad_is_bf16 = grad.dtype() == DType::BF16;
        // Stochastic-rounding BF16 store is supported only for the F32-grad
        // path. The BF16-grad kernel pipeline became reachable in Phase 4a
        // (commit 85d0542) via `Parameter::new_v2` with the `MatchParamDtype`
        // policy — v1 `Parameter::new` still casts grads to F32. When stoch is
        // requested with a BF16 grad we silently fall back to round-to-nearest;
        // it is a no-op in practice.
        let stoch_active = stoch_seed.is_some() && !grad_is_bf16;
        let kernel_name = if grad_is_bf16 {
            "adam_fused_bf16_kernel"
        } else if stoch_active {
            "adam_fused_bf16_f32grad_stoch_kernel"
        } else {
            "adam_fused_f32grad_kernel"
        };

        let f = device
            .get_func(MODULE_NAME, kernel_name)
            .ok_or_else(|| Error::Cuda(format!("missing kernel: {kernel_name}")))?;

        // Get raw pointers — minimal overhead, no format! tags
        let param_ptr = param.as_mut_device_ptr_bf16("adam:p")? as u64;
        let m_ptr = {
            let s = m.as_mut_slice_f32("adam:m")?;
            *s.device_ptr_mut()
        };
        let v_ptr = {
            let s = v.as_mut_slice_f32("adam:v")?;
            *s.device_ptr_mut()
        };

        let grad_ptr: u64 = if grad_is_bf16 {
            grad.as_device_ptr_bf16("adam:g")? as u64
        } else {
            let s = grad.as_slice_f32("adam:g")?;
            *s.device_ptr()
        };

        let n_i64 = n as i64;
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        let seed_u64: u64 = stoch_seed.unwrap_or(0);
        let mut params: Vec<*mut std::ffi::c_void> = Vec::with_capacity(13);
        params.push(&param_ptr as *const u64 as *mut std::ffi::c_void);
        params.push(&grad_ptr as *const u64 as *mut std::ffi::c_void);
        params.push(&m_ptr as *const u64 as *mut std::ffi::c_void);
        params.push(&v_ptr as *const u64 as *mut std::ffi::c_void);
        params.push(&lr as *const f32 as *mut std::ffi::c_void);
        params.push(&beta1 as *const f32 as *mut std::ffi::c_void);
        params.push(&beta2 as *const f32 as *mut std::ffi::c_void);
        params.push(&eps as *const f32 as *mut std::ffi::c_void);
        params.push(&weight_decay as *const f32 as *mut std::ffi::c_void);
        params.push(&bias_correction1 as *const f32 as *mut std::ffi::c_void);
        params.push(&bias_correction2 as *const f32 as *mut std::ffi::c_void);
        params.push(&n_i64 as *const i64 as *mut std::ffi::c_void);
        if stoch_active {
            params.push(&seed_u64 as *const u64 as *mut std::ffi::c_void);
        }

        unsafe {
            f.launch(cfg, &mut params)
                .map_err(|e| Error::Cuda(format!("adam_fused launch: {e:?}")))?;
        }
        // Autograd v2 prereq: bump version on in-place mutation.
        // Adam writes param (BF16), m (F32), and v (F32) in-place.
        param.storage_ref().bump_version();
        m.storage_ref().bump_version();
        v.storage_ref().bump_version();
        Ok(())
    }

    /// Launch fused Adam update for F32 parameters — single kernel, no temporaries.
    ///
    /// `param` must be F32, `m` and `v` must be F32, `grad` can be F32 or BF16.
    /// Unsupported `grad` dtypes (F16, I8, …) return an error — no silent cast.
    /// All tensors are modified in-place (param, m, v).
    pub fn adam_fused_step_f32(
        param: &mut Tensor,
        grad: &Tensor,
        m: &mut Tensor,
        v: &mut Tensor,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        bias_correction1: f32,
        bias_correction2: f32,
    ) -> Result<()> {
        // Validate dtypes once at entry
        debug_assert_eq!(param.dtype(), DType::F32);
        debug_assert_eq!(m.dtype(), DType::F32);
        debug_assert_eq!(v.dtype(), DType::F32);

        let n = param.shape().elem_count();
        debug_assert_eq!(n, grad.shape().elem_count());
        debug_assert_eq!(n, m.shape().elem_count());
        debug_assert_eq!(n, v.shape().elem_count());

        let device = param.device.clone();
        ensure_adam_kernels(&device)?;

        let kernel_name = match grad.dtype() {
            DType::F32 => "adam_fused_f32param_f32grad_kernel",
            DType::BF16 => "adam_fused_f32param_bf16grad_kernel",
            other => {
                return Err(Error::InvalidInput(format!(
                    "adam_fused_step_f32: unsupported grad dtype {:?} for F32 param \
                     (only F32 and BF16 grads are supported — convert upstream)",
                    other
                )))
            }
        };

        let f = device
            .get_func(MODULE_NAME, kernel_name)
            .ok_or_else(|| Error::Cuda(format!("missing kernel: {kernel_name}")))?;

        let param_ptr = {
            let s = param.as_mut_slice_f32("adam:p")?;
            *s.device_ptr_mut()
        };
        let m_ptr = {
            let s = m.as_mut_slice_f32("adam:m")?;
            *s.device_ptr_mut()
        };
        let v_ptr = {
            let s = v.as_mut_slice_f32("adam:v")?;
            *s.device_ptr_mut()
        };

        let grad_ptr: u64 = match grad.dtype() {
            DType::F32 => {
                let s = grad.as_slice_f32("adam:g")?;
                *s.device_ptr()
            }
            DType::BF16 => grad.as_device_ptr_bf16("adam:g")? as u64,
            _ => unreachable!("dtype was validated above"),
        };

        let n_i64 = n as i64;
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut params: Vec<*mut std::ffi::c_void> = Vec::with_capacity(12);
        params.push(&param_ptr as *const u64 as *mut std::ffi::c_void);
        params.push(&grad_ptr as *const u64 as *mut std::ffi::c_void);
        params.push(&m_ptr as *const u64 as *mut std::ffi::c_void);
        params.push(&v_ptr as *const u64 as *mut std::ffi::c_void);
        params.push(&lr as *const f32 as *mut std::ffi::c_void);
        params.push(&beta1 as *const f32 as *mut std::ffi::c_void);
        params.push(&beta2 as *const f32 as *mut std::ffi::c_void);
        params.push(&eps as *const f32 as *mut std::ffi::c_void);
        params.push(&weight_decay as *const f32 as *mut std::ffi::c_void);
        params.push(&bias_correction1 as *const f32 as *mut std::ffi::c_void);
        params.push(&bias_correction2 as *const f32 as *mut std::ffi::c_void);
        params.push(&n_i64 as *const i64 as *mut std::ffi::c_void);

        unsafe {
            f.launch(cfg, &mut params)
                .map_err(|e| Error::Cuda(format!("adam_fused_f32 launch: {e:?}")))?;
        }
        // Autograd v2 prereq: bump version on in-place mutation.
        param.storage_ref().bump_version();
        m.storage_ref().bump_version();
        v.storage_ref().bump_version();
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Multi-tensor fused Adam (one kernel launch covers many parameters).
    // -----------------------------------------------------------------------

    /// Cache of the device-resident metadata buffer used by the multi-tensor
    /// launcher. Held by `Adam` so we don't pay an alloc + free + sync per
    /// step. Capacity grows monotonically as the param count grows.
    pub struct MultiTensorMetaCache {
        buf: Option<cudarc::driver::CudaSlice<u64>>,
        capacity_n: usize,
    }

    impl MultiTensorMetaCache {
        pub const fn new() -> Self {
            Self {
                buf: None,
                capacity_n: 0,
            }
        }

        fn ensure(&mut self, dev: &Arc<CudaDevice>, n: usize) -> Result<()> {
            // cudarc 0.11 `htod_sync_copy_into` asserts src.len() == dst.len(),
            // so the cached buffer must be EXACTLY 5*n u64 slots — not "at
            // least". Re-allocate when n changes. For LoRA training where the
            // param count is fixed throughout, this is a one-time alloc on
            // step 0 and a no-op thereafter.
            let needed = 5 * n;
            if self.capacity_n == n && self.buf.is_some() {
                return Ok(());
            }
            let new_buf = dev
                .alloc_zeros::<u64>(needed)
                .map_err(|e| Error::Cuda(format!("multi_tensor meta alloc: {e:?}")))?;
            self.buf = Some(new_buf);
            self.capacity_n = n;
            Ok(())
        }
    }

    /// Launch fused multi-tensor AdamW update from a pre-built packed
    /// pointer + size buffer.
    ///
    /// `packed` must be exactly `5 * n` u64 entries laid out as five
    /// region-contiguous slabs: `[params(n) | grads(n) | ms(n) | vs(n) | sizes(n)]`.
    /// `m`/`v` storages are always F32. The (param, grad) dtype pair is
    /// selected by the two boolean discriminators:
    ///   `param_is_bf16=true,  grad_is_bf16=false` → `adam_fused_multi_bf16_f32grad_kernel`
    ///   `param_is_bf16=true,  grad_is_bf16=true`  → `adam_fused_multi_bf16_bf16grad_kernel`
    ///   `param_is_bf16=false, grad_is_bf16=false` → `adam_fused_multi_f32param_f32grad_kernel`
    ///   `param_is_bf16=false, grad_is_bf16=true`  → returns Err (no kernel; caller must
    ///                                                use per-param `adam_fused_step_f32`).
    /// The caller (typically `Adam::step`) is responsible for the
    /// pre-classification and for extracting the raw device pointers under
    /// whatever borrowing discipline the parameter storage requires.
    ///
    /// `cache` is held by the calling `Adam` instance so the device-side
    /// metadata buffer is allocated once and reused — without it, allocating
    /// a fresh buffer per step (plus the implicit cudaFree on drop) would
    /// erase the launch-overhead savings this kernel exists to capture.
    pub fn adam_fused_multi_tensor_step(
        cache: &mut MultiTensorMetaCache,
        device: &Arc<CudaDevice>,
        n: usize,
        param_is_bf16: bool,
        grad_is_bf16: bool,
        packed: &[u64],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        bias_correction1: f32,
        bias_correction2: f32,
        stoch_seed: Option<u64>,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        debug_assert_eq!(
            packed.len(),
            5 * n,
            "adam_fused_multi_tensor_step: packed buffer must be 5*n u64 entries"
        );
        ensure_adam_kernels(device)?;

        // Reuse / grow the device-side metadata buffer. cudarc 0.11
        // `htod_sync_copy_into` asserts equal lengths, so the cache holds
        // exactly `5 * n` slots and we re-allocate when n changes (a
        // one-time alloc on step 0 in steady-state training).
        cache.ensure(device, n)?;
        let dev_buf = cache
            .buf
            .as_mut()
            .expect("MultiTensorMetaCache::ensure post-condition: buf is Some");

        device
            .htod_sync_copy_into(packed, dev_buf)
            .map_err(|e| Error::Cuda(format!("adam_mt h2d: {e:?}")))?;

        // Compute device pointers for each region in the packed layout.
        let base = *dev_buf.device_ptr();
        let stride_bytes = (n * std::mem::size_of::<u64>()) as u64;
        let params_arr_ptr = base;
        let grads_arr_ptr = base + stride_bytes;
        let ms_arr_ptr = base + 2 * stride_bytes;
        let vs_arr_ptr = base + 3 * stride_bytes;
        let sizes_arr_ptr = base + 4 * stride_bytes;

        // Dispatch matrix:
        //   (BF16 param, F32 grad)          → BF16 fused multi-tensor
        //   (BF16 param, F32 grad, stoch)   → BF16 multi-tensor + stochastic round
        //   (BF16 param, BF16 grad)         → BF16 multi-tensor + BF16 grad cast
        //   (F32  param, F32 grad)          → F32  multi-tensor (no casts, no stoch)
        //   (F32  param, BF16 grad)         → ERROR — no kernel; classifier
        //                                     must route to per-param fallback
        //                                     via adam_fused_step_f32.
        //
        // Stoch rounding is only meaningful for BF16 param store; F32 params
        // don't benefit. The classifier in `Adam::step` clears the seed for
        // F32-param dispatch.
        let stoch_active = stoch_seed.is_some() && param_is_bf16 && !grad_is_bf16;
        let kernel_name = if !param_is_bf16 {
            if grad_is_bf16 {
                return Err(Error::Cuda(
                    "adam_fused_multi_tensor_step: (F32 param, BF16 grad) has no \
                     multi-tensor kernel; classifier must route to per-param fallback"
                        .into(),
                ));
            }
            "adam_fused_multi_f32param_f32grad_kernel"
        } else if grad_is_bf16 {
            "adam_fused_multi_bf16_bf16grad_kernel"
        } else if stoch_active {
            "adam_fused_multi_bf16_f32grad_stoch_kernel"
        } else {
            "adam_fused_multi_bf16_f32grad_kernel"
        };
        let f = device
            .get_func(MODULE_NAME, kernel_name)
            .ok_or_else(|| Error::Cuda(format!("missing kernel: {kernel_name}")))?;

        let n_tensors_i32 = n as i32;
        let block = 256u32;
        let grid = n as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        let seed_u64: u64 = stoch_seed.unwrap_or(0);
        let mut k_params: Vec<*mut std::ffi::c_void> = Vec::with_capacity(14);
        k_params.push(&params_arr_ptr as *const u64 as *mut std::ffi::c_void);
        k_params.push(&grads_arr_ptr as *const u64 as *mut std::ffi::c_void);
        k_params.push(&ms_arr_ptr as *const u64 as *mut std::ffi::c_void);
        k_params.push(&vs_arr_ptr as *const u64 as *mut std::ffi::c_void);
        k_params.push(&sizes_arr_ptr as *const u64 as *mut std::ffi::c_void);
        k_params.push(&n_tensors_i32 as *const i32 as *mut std::ffi::c_void);
        k_params.push(&lr as *const f32 as *mut std::ffi::c_void);
        k_params.push(&beta1 as *const f32 as *mut std::ffi::c_void);
        k_params.push(&beta2 as *const f32 as *mut std::ffi::c_void);
        k_params.push(&eps as *const f32 as *mut std::ffi::c_void);
        k_params.push(&weight_decay as *const f32 as *mut std::ffi::c_void);
        k_params.push(&bias_correction1 as *const f32 as *mut std::ffi::c_void);
        k_params.push(&bias_correction2 as *const f32 as *mut std::ffi::c_void);
        if stoch_active {
            k_params.push(&seed_u64 as *const u64 as *mut std::ffi::c_void);
        }

        unsafe {
            f.launch(cfg, &mut k_params)
                .map_err(|e| Error::Cuda(format!("adam_fused_multi launch: {e:?}")))?;
        }
        Ok(())
    }
}

/// Adam optimizer with momentum and adaptive learning rates
pub struct Adam {
    /// Learning rate
    lr: f32,
    /// Beta1 - exponential decay rate for first moment
    beta1: f32,
    /// Beta2 - exponential decay rate for second moment
    beta2: f32,
    /// Small constant for numerical stability
    eps: f32,
    /// Current timestep
    t: u32,
    /// First moment estimates
    m: HashMap<TensorId, Tensor>,
    /// Second moment estimates
    v: HashMap<TensorId, Tensor>,
    /// Weight decay coefficient
    weight_decay: f32,
    /// When `true`, the F32 → BF16 store at the end of each fused BF16-param
    /// step uses lower-16-bit hash-based stochastic rounding instead of
    /// round-to-nearest. Default `false` (byte-identical to prior behavior).
    /// Per-step entropy is derived from the step counter `t` so a run with a
    /// fixed top-level seed is reproducible across reruns.
    /// Affects only the F32-grad path; BF16-grad fallbacks (currently
    /// unreachable) keep round-to-nearest.
    stochastic_round: bool,
    /// Cached device-side metadata buffer for the multi-tensor fused-Adam
    /// launcher. Allocated lazily on the first multi-tensor-eligible step
    /// (all-BF16 params + all-F32 grads). Survives across steps so we don't
    /// pay an alloc + cudaFree-sync per step.
    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    multi_tensor_meta: fused::MultiTensorMetaCache,
}

impl Adam {
    /// Create a new Adam optimizer
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
            weight_decay,
            stochastic_round: false,
            #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
            multi_tensor_meta: fused::MultiTensorMetaCache::new(),
        }
    }

    /// Update the learning rate used for subsequent optimizer steps.
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    /// Toggle stochastic rounding on the BF16 param store. See
    /// [`Adam::stochastic_round`] field doc for semantics.
    pub fn set_stochastic_round(&mut self, on: bool) {
        self.stochastic_round = on;
    }

    /// Whether stochastic rounding is currently enabled for BF16 stores.
    pub fn is_stochastic_round(&self) -> bool {
        self.stochastic_round
    }

    /// Per-step seed for the stochastic-round kernel. Derived from the step
    /// counter so the schedule is reproducible across reruns. The hash mixes
    /// `t` with a large odd constant to avoid degenerate seeds at low step.
    fn stoch_seed(&self) -> Option<u64> {
        if self.stochastic_round {
            let t = self.t as u64;
            // splitmix64 finalizer keyed on the step count.
            let mut s = t.wrapping_add(0x9E3779B97F4A7C15);
            s = (s ^ (s >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            s = (s ^ (s >> 27)).wrapping_mul(0x94D049BB133111EB);
            s = s ^ (s >> 31);
            Some(s)
        } else {
            None
        }
    }

    fn validate_or_init_state_for_param(&mut self, param: &Parameter) -> Result<()> {
        let param_dtype = param.dtype()?;
        if param_dtype != DType::BF16 && param_dtype != DType::F32 {
            return Err(Error::InvalidInput(format!(
                "Adam::ensure_state_initialized: unsupported param dtype {:?} \
                 (only BF16 and F32 params are supported)",
                param_dtype
            )));
        }

        let id = param.id();
        let param_tensor = param.tensor()?;
        let shape = param_tensor.shape().clone();
        let device = param_tensor.device().clone();

        let validate_state = |name: &str, state: &Tensor| -> Result<()> {
            if state.dtype() != DType::F32 {
                return Err(Error::Training(format!(
                    "Adam optimizer {name} state for param {:?} has dtype {:?}; expected F32",
                    id,
                    state.dtype()
                )));
            }
            if state.shape() != &shape {
                return Err(Error::Training(format!(
                    "Adam optimizer {name} state for param {:?} has shape {:?}; expected {:?}",
                    id,
                    state.shape().dims(),
                    shape.dims()
                )));
            }
            Ok(())
        };

        if let Some(m) = self.m.get(&id) {
            validate_state("m", m)?;
        } else {
            self.m.insert(
                id,
                Tensor::zeros_dtype(shape.clone(), DType::F32, device.clone())?,
            );
        }

        if let Some(v) = self.v.get(&id) {
            validate_state("v", v)?;
        } else {
            self.v
                .insert(id, Tensor::zeros_dtype(shape.clone(), DType::F32, device)?);
        }

        Ok(())
    }

    /// Materialize persistent optimizer state for all parameters without
    /// advancing the step counter or mutating parameter values.
    ///
    /// Trainers that use a transient per-step allocator must call this before
    /// entering the step scope. Otherwise Adam's first `step()` lazily creates
    /// `m`/`v` tensors inside that transient scope and stores them on `self`,
    /// making them outlive the scope by design.
    pub fn ensure_state_initialized(&mut self, parameters: &[Parameter]) -> Result<()> {
        for param in parameters {
            self.validate_or_init_state_for_param(param)?;
        }
        Ok(())
    }

    /// Perform a single optimization step.
    ///
    /// All param dtypes except BF16 and F32 return an error — no silent
    /// fallback. Trainers using exotic dtypes (F16, I8, ...) must convert
    /// parameters to F32 or BF16 before calling this optimizer.
    pub fn step(&mut self, parameters: &[Parameter]) -> Result<()> {
        self.t += 1;

        // Bias correction factors
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        // ----------------------------------------------------------------
        // Multi-tensor fast path (Fusion Sprint Phase 4 follow-up).
        //
        // Eligible cases (mixed dtype across params falls back to per-param):
        //   1. All BF16 params + F32 grads — dominant LoRA training case
        //      (Parameter::set_grad casts incoming grads to F32).
        //   2. All F32 params + F32 grads — added Phase 1 of the launch-storm
        //      refactor: zimage LoRA uses F32 params (no-quantization rule),
        //      which previously fell through to the per-param F32 kernel
        //      (560 launches/step on zimage @ rank=8).
        //
        // Both collapse N kernel launches to 1 with bit-identical per-param
        // math (same DECOUPLED-WD receipt, same moment update order).
        // Verified by `tests/adam_multi_tensor_parity.rs`.
        // ----------------------------------------------------------------
        #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
        {
            let n = parameters.len();
            // Env override: `FLAME_ADAM_NO_MULTI_TENSOR=1` forces the per-param
            // fallback for A/B comparison. Production runs leave this unset.
            let multi_disabled = std::env::var("FLAME_ADAM_NO_MULTI_TENSOR")
                .ok()
                .as_deref()
                .map(|v| matches!(v, "1" | "true" | "TRUE"))
                .unwrap_or(false);

            // Four-way classifier (Phase 4a):
            //   - All BF16 params + all F32 grads — v1 / v3 dominant LoRA case
            //   - All BF16 params + all BF16 grads — autograd v2 / Option A
            //     (`adam_fused_multi_bf16_bf16grad_kernel` — see
            //     `BF16_GRAD_DECISION.md`).
            //   - All F32  params + all F32 grads — Phase 1 launch-storm.
            //   - Anything else (mixed dtype, F32-param+BF16-grad which has
            //     no multi-tensor kernel, etc.) → fall through to per-param.
            //
            // Returned tuple: `Some((param_is_bf16, grad_is_bf16))`.
            let mt_dtype: Option<(bool, bool)> = if multi_disabled || n == 0 {
                None
            } else if parameters.iter().all(|p| {
                p.dtype().ok() == Some(DType::BF16)
                    && p.grad().map_or(false, |g| g.dtype() == DType::F32)
            }) {
                Some((true, false))
            } else if parameters.iter().all(|p| {
                p.dtype().ok() == Some(DType::BF16)
                    && p.grad().map_or(false, |g| g.dtype() == DType::BF16)
            }) {
                Some((true, true))
            } else if parameters.iter().all(|p| {
                p.dtype().ok() == Some(DType::F32)
                    && p.grad().map_or(false, |g| g.dtype() == DType::F32)
            }) {
                Some((false, false))
            } else {
                // (F32 param, BF16 grad) intentionally NOT here — that
                // pair has no multi-tensor kernel (see
                // `adam_fused_multi_tensor_step` dispatch comment), so we
                // route through the per-param fused path which DOES have
                // an `adam_fused_f32param_bf16grad_kernel` arm.
                None
            };

            if let Some((param_is_bf16, grad_is_bf16)) = mt_dtype {
                // Ensure m/v exist for every param. Same shape + F32 dtype
                // contract as the per-param path so a switch between the
                // multi-tensor and per-param paths leaves state consistent.
                for param in parameters {
                    self.validate_or_init_state_for_param(param)?;
                }

                // Build the 5-region packed pointer + size buffer. Region
                // ordering must match the kernel's slab layout — see
                // `CUDA_ADAM_FUSED_MULTI_BF16` / `CUDA_ADAM_FUSED_MULTI_F32`
                // and `adam_fused_multi_tensor_step`.
                let mut packed: Vec<u64> = Vec::with_capacity(5 * n);

                // Region 0: param data pointers (dtype switch per param_is_bf16).
                if param_is_bf16 {
                    for param in parameters {
                        let p_ptr: u64 = param
                            .with_data_mut(|t| Ok(t.as_mut_device_ptr_bf16("adam_mt:p")? as u64))?;
                        packed.push(p_ptr);
                    }
                } else {
                    for param in parameters {
                        let p_ptr: u64 = param.with_data_mut(|t| {
                            let s = t.as_mut_slice_f32("adam_mt:p_f32")?;
                            Ok(*s.device_ptr_mut())
                        })?;
                        packed.push(p_ptr);
                    }
                }
                // Region 1: grad pointers (dtype switch per grad_is_bf16).
                // Phase 4a: the (BF16 param, BF16 grad) and
                // (F32 param, BF16 grad) arms route grad pointers through
                // `as_device_ptr_bf16` so the kernel reads BF16 storage
                // directly. The (F32 param, BF16 grad) case never reaches
                // here — it has no multi-tensor kernel and the classifier
                // routes it to per-param.
                if grad_is_bf16 {
                    for param in parameters {
                        let g = param
                            .grad()
                            .expect("grad presence checked in classifier above");
                        let p_ptr = g.as_device_ptr_bf16("adam_mt:g_bf16")? as u64;
                        packed.push(p_ptr);
                    }
                } else {
                    for param in parameters {
                        let g = param
                            .grad()
                            .expect("grad presence checked in classifier above");
                        let s = g.as_slice_f32("adam_mt:g")?;
                        packed.push(*s.device_ptr());
                    }
                }
                // Region 2: F32 m pointers. HashMap::get_mut once per id.
                for param in parameters {
                    let m_t = self
                        .m
                        .get_mut(&param.id())
                        .expect("m initialized in lazy-init pass above");
                    let s = m_t.as_mut_slice_f32("adam_mt:m")?;
                    packed.push(*s.device_ptr_mut());
                }
                // Region 3: F32 v pointers.
                for param in parameters {
                    let v_t = self
                        .v
                        .get_mut(&param.id())
                        .expect("v initialized in lazy-init pass above");
                    let s = v_t.as_mut_slice_f32("adam_mt:v")?;
                    packed.push(*s.device_ptr_mut());
                }
                // Region 4: per-tensor element counts (i64, stored as u64).
                for param in parameters {
                    packed.push(param.shape().elem_count() as u64);
                }

                let device = parameters[0]
                    .grad()
                    .expect("grad presence checked")
                    .device
                    .clone();

                // Stochastic rounding is only meaningful for BF16 param
                // store. F32 params write the full float — clear the seed
                // so the dispatcher doesn't even consider it. The BF16-grad
                // multi-tensor kernel itself currently keeps round-to-nearest
                // (see dispatch matrix comment in `adam_fused_multi_tensor_step`).
                let seed = if param_is_bf16 {
                    self.stoch_seed()
                } else {
                    None
                };
                fused::adam_fused_multi_tensor_step(
                    &mut self.multi_tensor_meta,
                    &device,
                    n,
                    param_is_bf16,
                    grad_is_bf16,
                    &packed,
                    self.lr,
                    self.beta1,
                    self.beta2,
                    self.eps,
                    self.weight_decay,
                    bias_correction1,
                    bias_correction2,
                    seed,
                )?;
                // Autograd v2 prereq: bump version on each in-place tensor
                // touched by the multi-tensor kernel (param, m, v per id).
                // The kernel sees raw u64 pointers and can't bump itself.
                for param in parameters {
                    param.with_data_mut(|t| {
                        t.storage_ref().bump_version();
                        Ok(())
                    })?;
                    if let Some(m_t) = self.m.get(&param.id()) {
                        m_t.storage_ref().bump_version();
                    }
                    if let Some(v_t) = self.v.get(&param.id()) {
                        v_t.storage_ref().bump_version();
                    }
                }
                return Ok(());
            }
        }

        // ----------------------------------------------------------------
        // Per-param fallback. Exercises mixed-dtype slices, F32 params, or
        // any case the multi-tensor classifier rejected.
        // ----------------------------------------------------------------
        for param in parameters {
            if let Some(grad) = param.grad() {
                let param_id = param.id();
                let param_dtype = param.dtype()?;

                // Fused path: BF16 param with F32 state — single kernel launch
                #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
                if param_dtype == DType::BF16 {
                    self.validate_or_init_state_for_param(param)?;

                    // Compute the per-step stoch seed BEFORE the &mut borrow
                    // of self.m / self.v below; the seed is a &self read of
                    // self.stochastic_round + self.t and would otherwise
                    // collide with the mutable map borrows.
                    let seed = self.stoch_seed();
                    let m = self
                        .m
                        .get_mut(&param_id)
                        .ok_or_else(|| Error::Training("optimizer m state missing".into()))?;
                    let v = self
                        .v
                        .get_mut(&param_id)
                        .ok_or_else(|| Error::Training("optimizer v state missing".into()))?;
                    // Single fused kernel: param, m, v updated in-place
                    param.with_data_mut(|param_tensor| {
                        fused::adam_fused_step(
                            param_tensor,
                            &grad,
                            m,
                            v,
                            self.lr,
                            self.beta1,
                            self.beta2,
                            self.eps,
                            self.weight_decay,
                            bias_correction1,
                            bias_correction2,
                            seed,
                        )
                    })?;
                    continue;
                }

                // Fused path: F32 param with F32 state.
                //
                // Dispatch below routes BF16 grads to `adam_fused_f32param_bf16grad_kernel`.
                // This path became reachable in Phase 4a (commit 85d0542) via
                // `Parameter::new_v2`, which uses the `MatchParamDtype` policy and
                // preserves BF16 grad dtype on `set_grad`. v1 `Parameter::new` (and
                // `set_grad` under `CastToF32`) still casts grads to F32, so v3
                // trainers continue to take the F32-grad arm. The BF16-grad kernel
                // is also exercised by `tests/autograd_v2_phase4a.rs::adam_step_f32_
                // param_bf16_grad_no_panic`.
                //
                // Hardcoded `state_dtype = F32` below: the fused F32-param kernel
                // requires F32 m/v. `config::select_optimizer_state_dtype` only ever
                // gates BF16 m/v for BF16 params — see debug_assert! below.
                #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
                if param_dtype == DType::F32 {
                    debug_assert_eq!(
                        crate::config::select_optimizer_state_dtype(DType::F32),
                        DType::F32,
                        "F32 params must have F32 optimizer state; \
                         select_optimizer_state_dtype changed surface — \
                         fused kernel requires F32 m/v"
                    );

                    self.validate_or_init_state_for_param(param)?;

                    let m = self
                        .m
                        .get_mut(&param_id)
                        .ok_or_else(|| Error::Training("optimizer m state missing".into()))?;
                    let v = self
                        .v
                        .get_mut(&param_id)
                        .ok_or_else(|| Error::Training("optimizer v state missing".into()))?;

                    param.with_data_mut(|param_tensor| {
                        fused::adam_fused_step_f32(
                            param_tensor,
                            &grad,
                            m,
                            v,
                            self.lr,
                            self.beta1,
                            self.beta2,
                            self.eps,
                            self.weight_decay,
                            bias_correction1,
                            bias_correction2,
                        )
                    })?;
                    continue;
                }

                // No silent fallback: every optimizer step goes through a fused
                // CUDA kernel. F16 / I8 / other param dtypes are the trainer's
                // responsibility to convert upstream.
                return Err(Error::InvalidInput(format!(
                    "Adam::step: unsupported param dtype {:?} \
                     (only BF16 and F32 params are supported — convert upstream)",
                    param_dtype
                )));
            }
        }

        Ok(())
    }

    /// Zero all gradients
    pub fn zero_grad(&self, parameters: &[Parameter]) {
        for param in parameters {
            param.zero_grad();
        }
    }
}

impl Default for Adam {
    fn default() -> Self {
        Self::new(0.001, 0.9, 0.999, 1e-8, 0.0)
    }
}

/// AdamW optimizer (Adam with decoupled weight decay)
pub struct AdamW {
    adam: Adam,
}

impl AdamW {
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            adam: Adam::new(lr, beta1, beta2, eps, weight_decay),
        }
    }

    /// Update the learning rate used for subsequent optimizer steps.
    pub fn set_lr(&mut self, lr: f32) {
        self.adam.set_lr(lr);
    }

    /// Toggle stochastic rounding on the F32 → BF16 store at the end of
    /// each fused BF16-param step. Default off (round-to-nearest, byte-
    /// identical to prior commits). When on, per-element rounding is driven
    /// by a step-counter-derived seed mixed with `(tensor_idx, elem_idx)`,
    /// matching the standalone `bf16_ops::stochastic_round_f32_to_bf16`
    /// kernel's CPU reference. Long horizon BF16-param training accumulates
    /// small grads correctly under stochastic rounding instead of stalling.
    pub fn set_stochastic_round(&mut self, on: bool) {
        self.adam.set_stochastic_round(on);
    }

    /// Whether stochastic BF16 rounding is currently on.
    pub fn is_stochastic_round(&self) -> bool {
        self.adam.is_stochastic_round()
    }

    pub fn step(&mut self, parameters: &[Parameter]) -> Result<()> {
        self.adam.step(parameters)
    }

    pub fn ensure_state_initialized(&mut self, parameters: &[Parameter]) -> Result<()> {
        self.adam.ensure_state_initialized(parameters)
    }

    pub fn zero_grad(&self, parameters: &[Parameter]) {
        self.adam.zero_grad(parameters)
    }

    // ── Checkpoint accessors ────────────────────────────────────────────
    // Used by trainers that save/restore full optimizer state across runs.
    // The TensorId-keyed m/v maps must be re-keyed by the caller using the
    // current run's Parameter ids (TensorIds are per-run unique, not stable
    // on reload).

    pub fn t(&self) -> u32 {
        self.adam.t
    }
    pub fn lr(&self) -> f32 {
        self.adam.lr
    }
    pub fn beta1(&self) -> f32 {
        self.adam.beta1
    }
    pub fn beta2(&self) -> f32 {
        self.adam.beta2
    }
    pub fn eps(&self) -> f32 {
        self.adam.eps
    }
    pub fn weight_decay(&self) -> f32 {
        self.adam.weight_decay
    }

    /// Read m and v for a given parameter, if state has been initialized
    /// (i.e. the parameter has had at least one step).
    pub fn state_for(&self, param: &Parameter) -> Option<(Tensor, Tensor)> {
        let id = param.id();
        let m = self.adam.m.get(&id)?.clone();
        let v = self.adam.v.get(&id)?.clone();
        Some((m, v))
    }

    /// Inject m/v state for a parameter and set the global step counter.
    /// Use after rebuilding the model on resume — TensorIds will be fresh,
    /// so the caller pairs each Parameter with its saved (m, v) by name.
    pub fn set_state(&mut self, param: &Parameter, m: Tensor, v: Tensor) {
        let id = param.id();
        self.adam.m.insert(id, m);
        self.adam.v.insert(id, v);
    }

    pub fn set_t(&mut self, t: u32) {
        self.adam.t = t;
    }
}

impl Default for AdamW {
    fn default() -> Self {
        Self::new(0.001, 0.9, 0.999, 1e-8, 0.01)
    }
}

impl Adam {
    fn state_dtype(&self, param_id: &TensorId) -> Option<(DType, DType)> {
        let m = self.m.get(param_id)?;
        let v = self.v.get(param_id)?;
        Some((m.dtype(), v.dtype()))
    }

    /// Return the total bytes consumed by optimizer state tensors.
    pub fn state_memory_bytes(&self) -> usize {
        let m_bytes: usize = self
            .m
            .values()
            .map(|tensor| tensor.shape().elem_count() * tensor.dtype().size_in_bytes())
            .sum();
        let v_bytes: usize = self
            .v
            .values()
            .map(|tensor| tensor.shape().elem_count() * tensor.dtype().size_in_bytes())
            .sum();
        m_bytes + v_bytes
    }

    /// Alias for compatibility with layout checks.
    pub fn state_bytes(&self) -> usize {
        self.state_memory_bytes()
    }
}

impl AdamW {
    /// Inspect the optimizer state tensor dtypes for a parameter.
    ///
    /// This is primarily intended for tests to ensure mixed-precision
    /// invariants (e.g. FP32 moment buffers) remain satisfied.
    pub fn debug_state_dtype(&self, param: &Parameter) -> Option<(DType, DType)> {
        self.adam.state_dtype(&param.id())
    }

    /// Return the total bytes consumed by optimizer state tensors.
    pub fn state_memory_bytes(&self) -> usize {
        self.adam.state_memory_bytes()
    }

    /// Alias matching the stabilization docs terminology.
    pub fn state_bytes(&self) -> usize {
        self.state_memory_bytes()
    }
}

#[cfg(all(test, feature = "legacy_full"))]
mod tests {
    use super::*;
    use crate::{Shape, Tensor};
    use cudarc::driver::CudaDevice;

    #[test]
    fn test_adam_step() -> Result<()> {
        let device = CudaDevice::new(0)?;

        // Create parameter
        let param = Parameter::randn(Shape::from_dims(&[10]), 0.0, 1.0, device)?;
        let before = param.tensor()?.to_vec()?;

        // Set a gradient
        let grad = Tensor::ones(Shape::from_dims(&[10]), param.tensor()?.device.clone())?;
        param.set_grad(grad)?;

        // Create optimizer and take a step
        let mut optimizer = Adam::default();
        optimizer.step(&[param.clone()])?;

        // Check that parameter was updated
        let new_value = param.tensor()?.to_vec()?;
        assert!(new_value[0] < before[0]);

        Ok(())
    }
}

// Kernel-level test for the F32-param / BF16-grad fused kernel.
//
// `Parameter::set_grad` casts all grads to F32, so the BF16-grad variant of
// `adam_fused_step_f32` is unreachable via the public `Adam::step` path.
// This test calls the fused launcher directly against a pair of Tensors,
// bypassing `Parameter`.
#[cfg(all(test, feature = "cuda", feature = "bf16_u16"))]
mod f32param_bf16grad_kernel_test {
    use super::*;
    use crate::{global_cuda_device, Shape, Tensor};

    #[test]
    fn adam_fused_f32param_bf16grad_matches_host() -> Result<()> {
        let device = global_cuda_device();
        let n = 1024usize;
        let shape = Shape::from_dims(&[n]);

        let init: Vec<f32> = (0..n).map(|i| (i as f32 * 0.0137).cos() * 0.1).collect();
        let grads: Vec<Vec<f32>> = (0..10)
            .map(|k| {
                (0..n)
                    .map(|i| ((i + k * 17) as f32 * 0.021).sin() * 0.01)
                    .collect()
            })
            .collect();

        let mut param = Tensor::from_vec(init.clone(), shape.clone(), device.clone())?;
        let mut m = Tensor::from_vec(vec![0.0; n], shape.clone(), device.clone())?;
        let mut v = Tensor::from_vec(vec![0.0; n], shape.clone(), device.clone())?;

        let lr = 1e-3f32;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        let wd = 0.01f32;

        // Reference (host, float32 grads)
        let mut ref_param = init.clone();
        let mut ref_m = vec![0.0f32; n];
        let mut ref_v = vec![0.0f32; n];

        for (t, gvec) in grads.iter().enumerate() {
            let t1 = (t + 1) as i32;
            let bc1 = 1.0f32 - beta1.powi(t1);
            let bc2 = 1.0f32 - beta2.powi(t1);

            // GPU fused step with BF16 grad
            let grad_bf16 = Tensor::from_vec(gvec.clone(), shape.clone(), device.clone())?
                .to_dtype(DType::BF16)?;
            fused::adam_fused_step_f32(
                &mut param, &grad_bf16, &mut m, &mut v, lr, beta1, beta2, eps, wd, bc1, bc2,
            )?;

            // Host reference: cast grad through BF16 round-trip to mirror kernel precision
            let grad_rounded: Vec<f32> = grad_bf16.to_vec_f32()?;
            for i in 0..n {
                let g = grad_rounded[i];
                ref_m[i] = beta1 * ref_m[i] + (1.0 - beta1) * g;
                ref_v[i] = beta2 * ref_v[i] + (1.0 - beta2) * g * g;
                let m_hat = ref_m[i] / bc1;
                let v_hat = ref_v[i] / bc2;
                let mut p = ref_param[i];
                p -= lr * m_hat / (v_hat.sqrt() + eps);
                if wd > 0.0 {
                    p -= lr * wd * p;
                }
                ref_param[i] = p;
            }
        }

        let fused_out = param.to_vec_f32()?;

        // BF16 grad introduces ~2^-7 relative rounding at each step; 10 steps,
        // wd-scaled. 5e-3 absolute tolerance per spec, with cos-sim ≥ 0.99999.
        let mut max_abs = 0f32;
        let mut dot = 0f64;
        let mut a2 = 0f64;
        let mut b2 = 0f64;
        for (a, b) in fused_out.iter().zip(ref_param.iter()) {
            let d = (a - b).abs();
            if d > max_abs {
                max_abs = d;
            }
            dot += (*a as f64) * (*b as f64);
            a2 += (*a as f64) * (*a as f64);
            b2 += (*b as f64) * (*b as f64);
        }
        let cos = dot / (a2.sqrt() * b2.sqrt() + 1e-30);
        assert!(
            max_abs <= 5e-3,
            "F32param/BF16grad fused diverges: max_abs={max_abs}"
        );
        assert!(
            cos >= 0.99999,
            "F32param/BF16grad cos similarity {cos} below 0.99999"
        );
        Ok(())
    }
}
