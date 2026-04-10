//! Adam optimizer implementation

use crate::{config, parameter::Parameter, DType, Error, Result, Tensor, TensorId};
use std::collections::{hash_map::Entry, HashMap};

// ---------------------------------------------------------------------------
// Fused Adam CUDA kernel (inline PTX, compiled on first use)
// ---------------------------------------------------------------------------

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
const CUDA_ADAM_FUSED_BF16: &str = r#"
#include <cuda_bf16.h>
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

    if (weight_decay > 0.0f) {
        g += weight_decay * p;
    }

    float mi = beta1 * m[idx] + (1.0f - beta1) * g;
    float vi = beta2 * v[idx] + (1.0f - beta2) * g * g;
    m[idx] = mi;
    v[idx] = vi;

    float m_hat = mi / bias_correction1;
    float v_hat = vi / bias_correction2;
    p -= lr * m_hat / (sqrtf(v_hat) + eps);

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

    if (weight_decay > 0.0f) {
        g += weight_decay * p;
    }

    float mi = beta1 * m[idx] + (1.0f - beta1) * g;
    float vi = beta2 * v[idx] + (1.0f - beta2) * g * g;
    m[idx] = mi;
    v[idx] = vi;

    float m_hat = mi / bias_correction1;
    float v_hat = vi / bias_correction2;
    p -= lr * m_hat / (sqrtf(v_hat) + eps);

    param[idx] = __float2bfloat16(p);
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
        if device.get_func(MODULE_NAME, "adam_fused_bf16_kernel").is_some() {
            return Ok(());
        }
        // Cold path: compile and load
        let include_dir = std::env::var("CUDA_INCLUDE_DIR")
            .or_else(|_| std::env::var("CUDA_HOME").map(|home| format!("{home}/include")))
            .unwrap_or_else(|_| "/usr/local/cuda/include".into());
        let mut opts = CompileOptions::default();
        opts.include_paths.push(include_dir);
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(CUDA_ADAM_FUSED_BF16, opts)
            .map_err(|e| Error::Cuda(format!("nvrtc adam_fused: {:?}", e)))?;
        device
            .load_ptx(
                ptx,
                MODULE_NAME,
                &["adam_fused_bf16_kernel", "adam_fused_f32grad_kernel"],
            )
            .map_err(|e| Error::Cuda(format!("load adam_fused: {:?}", e)))?;
        Ok(())
    }

    /// Launch fused Adam update — single kernel, no temporaries.
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
        use crate::device::CudaStreamRawPtrExt;

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

    /// Perform a single optimization step
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

                // Fallback path: non-BF16 params (F32 params, etc.)
                self.step_scalar_ops(param, grad, param_id, param_dtype, bias_correction1, bias_correction2)?;
            }
        }

        Ok(())
    }

    /// Scalar-op fallback for non-BF16 parameters (original 13-15 kernel path).
    fn step_scalar_ops(
        &mut self,
        param: &Parameter,
        mut grad: Tensor,
        param_id: TensorId,
        param_dtype: DType,
        bias_correction1: f32,
        bias_correction2: f32,
    ) -> Result<()> {
        let state_dtype = config::select_optimizer_state_dtype(param_dtype);

        if grad.dtype() != state_dtype {
            grad = grad.to_dtype(state_dtype)?;
        }
        if self.weight_decay > 0.0 {
            let param_tensor = param.tensor()?;
            let param_adjust = if param_tensor.dtype() == state_dtype {
                param_tensor
            } else {
                param_tensor.to_dtype(state_dtype)?
            };
            grad = grad.add(&param_adjust.mul_scalar(self.weight_decay)?)?;
        }

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

        *m = m
            .mul_scalar(self.beta1)?
            .add(&grad.mul_scalar(1.0 - self.beta1)?)?;

        let grad_sq = grad.mul(&grad)?;
        *v = v
            .mul_scalar(self.beta2)?
            .add(&grad_sq.mul_scalar(1.0 - self.beta2)?)?;

        let m_hat = m.div_scalar(bias_correction1)?;
        let v_hat = v.div_scalar(bias_correction2)?;

        let v_sqrt = v_hat.sqrt()?;
        let denominator = v_sqrt.add_scalar(self.eps)?;
        let update = m_hat.div(&denominator)?.mul_scalar(self.lr)?;

        param.apply_update(&update)?;
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
