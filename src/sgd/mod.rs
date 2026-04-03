//! Minimal GPU SGD: `param -= lr * grad` for F32/BF16 tensors.

use crate::{config, DType, Error, Result, Tensor};
use cudarc::{
    driver::{CudaDevice, LaunchAsync, LaunchConfig},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use std::collections::{hash_map::Entry, HashMap};
use std::sync::{Arc, OnceLock};

const CUDA_SRC: &str = r#"
extern "C" __global__
void sgd_f32(float* __restrict__ p, const float* __restrict__ g, size_t n, float lr){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){ p[i] -= lr * g[i]; }
}
#include <cuda_bf16.h>
extern "C" __global__
#ifdef __CUDA_ARCH__
void sgd_bf16(__nv_bfloat16* __restrict__ p, const float* __restrict__ g, size_t n, float lr){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        float pi = __bfloat162float(p[i]);
        pi -= lr * g[i];
        p[i] = __float2bfloat16_rn(pi);
    }
}
#endif
"#;

static MOD_ONCE: OnceLock<()> = OnceLock::new();

fn ensure_module(dev: &Arc<CudaDevice>) -> Result<()> {
    if dev.get_func("flame_sgd", "sgd_f32").is_some() {
        return Ok(());
    }
    if MOD_ONCE.get().is_none() {
        let include_dir = std::env::var("CUDA_INCLUDE_DIR")
            .or_else(|_| std::env::var("CUDA_HOME").map(|home| format!("{home}/include")))
            .unwrap_or_else(|_| "/usr/local/cuda/include".into());
        let mut opts = CompileOptions::default();
        opts.include_paths.push(include_dir.into());
        let ptx =
            compile_ptx_with_opts(CUDA_SRC, opts).map_err(|e| Error::KernelError(e.to_string()))?;
        #[cfg(feature = "bf16_u16")]
        let symbols: &[&str] = &["sgd_f32", "sgd_bf16"];
        #[cfg(not(feature = "bf16_u16"))]
        let symbols: &[&str] = &["sgd_f32"];
        dev.load_ptx(ptx, "flame_sgd", symbols)
            .map_err(|e| Error::KernelError(e.to_string()))?;
        let _ = MOD_ONCE.set(());
    }
    Ok(())
}

/// In-place SGD step: `param -= lr * grad`.
pub fn step_inplace(param: &mut Tensor, grad: &Tensor, lr: f32) -> Result<()> {
    if param.dtype() != DType::F32 && param.dtype() != DType::BF16 {
        return Err(Error::Unsupported("sgd: param dtype not supported".into()));
    }
    if grad.dtype() != DType::F32 {
        return Err(Error::InvalidInput("sgd: grad must be F32".into()));
    }
    if param.shape() != grad.shape() {
        return Err(Error::InvalidInput("sgd: shape mismatch".into()));
    }
    let param_dev = Arc::clone(param.device());
    let grad_dev = Arc::clone(grad.device());
    if Arc::as_ptr(&param_dev) != Arc::as_ptr(&grad_dev) {
        return Err(Error::InvalidInput(
            "sgd: param and grad must be on same device".into(),
        ));
    }
    ensure_module(&param_dev)?;
    let n = param.shape().elem_count();
    if n == 0 {
        return Ok(());
    }
    let cfg = LaunchConfig::for_num_elems(n as u32);

    unsafe {
        match param.dtype() {
            DType::F32 => {
                let p_slice = param.storage_mut().try_as_mut_slice_f32()?;
                let g_slice = grad.storage_ref().try_as_slice_f32()?;
                let func = param_dev
                    .get_func("flame_sgd", "sgd_f32")
                    .ok_or_else(|| Error::KernelError("sgd_f32 missing".into()))?;
                func.launch(cfg, (p_slice, g_slice, n as u64, lr))
                    .map_err(|e| Error::KernelError(e.to_string()))?;
            }
            DType::BF16 => {
                #[cfg(feature = "bf16_u16")]
                {
                    let p_slice = param.storage_mut().try_as_mut_slice_u16()?;
                    let g_slice = grad.storage_ref().try_as_slice_f32()?;
                    let func = param_dev
                        .get_func("flame_sgd", "sgd_bf16")
                        .ok_or_else(|| Error::KernelError("sgd_bf16 missing".into()))?;
                    func.launch(cfg, (p_slice, g_slice, n as u64, lr))
                        .map_err(|e| Error::KernelError(e.to_string()))?;
                }
                #[cfg(not(feature = "bf16_u16"))]
                {
                    return Err(Error::Unsupported(
                        "BF16 requires the bf16_u16 feature".into(),
                    ));
                }
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct SGDConfig {
    pub lr: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    pub nesterov: bool,
}

impl Default for SGDConfig {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        }
    }
}

pub struct SGD {
    cfg: SGDConfig,
    velocity: HashMap<usize, Tensor>, // FP32 momentum buffers
}

impl SGD {
    pub fn new(cfg: SGDConfig) -> Self {
        Self {
            cfg,
            velocity: HashMap::new(),
        }
    }

    pub fn step(&mut self, params: &[crate::parameter::Parameter]) -> Result<()> {
        for param in params {
            let grad = param
                .grad()
                .ok_or_else(|| Error::Training("sgd: gradient missing".into()))?;

            let param_dtype = param.dtype()?;
            let state_dtype = config::select_optimizer_state_dtype(param_dtype);

            if self.cfg.momentum > 0.0 && state_dtype != param_dtype {
                crate::log_once!(
                    "sgd_state_dtype_mismatch",
                    "SGD states use {state:?} while params use {param:?}",
                    state = state_dtype,
                    param = param_dtype
                );
            }

            let mut grad_state = if grad.dtype() == state_dtype {
                grad
            } else {
                grad.to_dtype(state_dtype)?
            };

            if self.cfg.weight_decay > 0.0 {
                let param_tensor = param.tensor()?;
                let param_adjust = if param_tensor.dtype() == state_dtype {
                    param_tensor
                } else {
                    param_tensor.to_dtype(state_dtype)?
                };
                grad_state = grad_state.add(&param_adjust.mul_scalar(self.cfg.weight_decay)?)?;
            }

            let update = if self.cfg.momentum > 0.0 {
                let entry = self.velocity.entry(param.id().0);
                let velocity = match entry {
                    Entry::Occupied(occ) => occ.into_mut(),
                    Entry::Vacant(vac) => {
                        vac.insert(grad_state.zeros_like_with_dtype(state_dtype)?)
                    }
                };

                let new_velocity = velocity.mul_scalar(self.cfg.momentum)?.add(&grad_state)?;
                *velocity = new_velocity.clone_result()?;

                if self.cfg.nesterov {
                    grad_state.add(&new_velocity.mul_scalar(self.cfg.momentum)?)?
                } else {
                    new_velocity
                }
            } else {
                grad_state
            };

            let update_scaled = update.mul_scalar(self.cfg.lr)?;
            param.apply_update(&update_scaled)?;
        }
        Ok(())
    }

    pub fn state_bytes(&self) -> usize {
        self.velocity
            .values()
            .map(|v| v.shape().elem_count() * v.dtype().size_in_bytes())
            .sum()
    }
}
