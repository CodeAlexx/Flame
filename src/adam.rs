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
use std::collections::{hash_map::Entry, HashMap};

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

        // Cold path: combine all four kernel sources into a single translation
        // unit and compile once. `<cuda_bf16.h>` has its own include guard, so
        // the repeated include across the bf16-touching source constants is a
        // no-op. Option A from the Phase 2 plan: one NVRTC compile, one
        // `load_ptx`, one `MODULE_NAME` — `get_func` is O(1) regardless of
        // which dtype combo hits at step time.
        let combined = format!(
            "{}\n{}\n{}",
            CUDA_ADAM_FUSED_BF16,
            CUDA_ADAM_FUSED_F32PARAM_F32GRAD,
            CUDA_ADAM_FUSED_F32PARAM_BF16GRAD,
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
        let kernel_name = if grad_is_bf16 {
            "adam_fused_bf16_kernel"
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
                .map_err(|e| Error::Cuda(format!("adam_fused launch: {e:?}")))?;
        }
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
        }
    }

    /// Update the learning rate used for subsequent optimizer steps.
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
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

        for param in parameters {
            if let Some(grad) = param.grad() {
                let param_id = param.id();
                let param_dtype = param.dtype()?;

                // Fused path: BF16 param with F32 state — single kernel launch
                #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
                if param_dtype == DType::BF16 {
                    let state_dtype = DType::F32;

                    // Initialize m/v on first step
                    if let Entry::Vacant(entry) = self.m.entry(param_id) {
                        entry.insert(grad.zeros_like_with_dtype(state_dtype)?);
                    }
                    if let Entry::Vacant(entry) = self.v.entry(param_id) {
                        entry.insert(grad.zeros_like_with_dtype(state_dtype)?);
                    }

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
                        )
                    })?;
                    continue;
                }

                // Fused path: F32 param with F32 state.
                //
                // Dispatch below routes BF16 grads to `adam_fused_f32param_bf16grad_kernel`,
                // but that path is currently UNREACHABLE via the public API:
                // `Parameter::set_grad` at `src/parameter.rs:90-94` unconditionally
                // casts incoming grads to F32, so any F32 param here always sees an
                // F32 grad. The BF16-grad kernel is validated in-crate by
                // `adam::f32param_bf16grad_kernel_test` and kept as dead-ready code
                // against a future change to `Parameter::set_grad` that preserves
                // BF16 grads end-to-end (part of any genuine BF16 training pipeline).
                //
                // Hardcoded `state_dtype = F32` below: the fused F32-param kernel
                // requires F32 m/v. `config::select_optimizer_state_dtype` only ever
                // gates BF16 m/v for BF16 params — see debug_assert! below.
                #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
                if param_dtype == DType::F32 {
                    let state_dtype = DType::F32;
                    debug_assert_eq!(
                        crate::config::select_optimizer_state_dtype(DType::F32),
                        DType::F32,
                        "F32 params must have F32 optimizer state; \
                         select_optimizer_state_dtype changed surface — \
                         fused kernel requires F32 m/v"
                    );

                    if let Entry::Vacant(entry) = self.m.entry(param_id) {
                        entry.insert(grad.zeros_like_with_dtype(state_dtype)?);
                    }
                    if let Entry::Vacant(entry) = self.v.entry(param_id) {
                        entry.insert(grad.zeros_like_with_dtype(state_dtype)?);
                    }

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

    pub fn step(&mut self, parameters: &[Parameter]) -> Result<()> {
        self.adam.step(parameters)
    }

    pub fn zero_grad(&self, parameters: &[Parameter]) {
        self.adam.zero_grad(parameters)
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
            .map(|k| (0..n).map(|i| ((i + k * 17) as f32 * 0.021).sin() * 0.01).collect())
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
            let grad_bf16 =
                Tensor::from_vec(gvec.clone(), shape.clone(), device.clone())?.to_dtype(DType::BF16)?;
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
