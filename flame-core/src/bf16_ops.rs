#![cfg(feature = "bf16_u16")]
use std::sync::Arc;
use cudarc::{driver::{CudaDevice, CudaSlice, LaunchConfig, DeviceSlice}, nvrtc::compile_ptx};
use crate::{Tensor, TensorId, Shape, Result, FlameError};
use crate::tensor_storage::TensorStorage;
use crate::dtype::DType;

const CUDA_GELU: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void gelu_bf16_kernel(const __nv_bfloat16* __restrict__ X,
                      __nv_bfloat16* __restrict__ Y,
                      long n) {
  long i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < n) {
    float x = __bfloat162float(X[i]);
    float c = 0.7978845608f*(x + 0.044715f*x*x*x);
    float y = 0.5f * x * (1.0f + tanhf(c));
    Y[i] = __float2bfloat16(y);
    i += gridDim.x * blockDim.x;
  }
}
"#;

const CUDA_SQUARE: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void square_bf16_kernel(const __nv_bfloat16* __restrict__ X,
                        __nv_bfloat16* __restrict__ Y,
                        long n) {
  long i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < n) {
    float x = __bfloat162float(X[i]);
    Y[i] = __float2bfloat16(x*x);
    i += gridDim.x * blockDim.x;
  }
}
"#;

fn ensure(dev: &Arc<CudaDevice>, name: &str, code: &str) -> Result<(), FlameError> {
    if dev.get_func(name, name).is_some() { return Ok(()); }
    let ptx = compile_ptx(code).map_err(|e| FlameError::Cuda(format!("nvrtc: {}", e)))?;
    dev.load_ptx(ptx, name, &[name]).map_err(|e| FlameError::Cuda(format!("load_ptx {}: {}", name, e)))?;
    Ok(())
}

#[inline]
fn lc(n: usize) -> LaunchConfig { LaunchConfig::for_num_elems(n as u32) }

pub fn gelu_bf16(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), DType::BF16);
    let n = x.shape().elem_count();
    let mut out = Tensor {
        storage: TensorStorage::BF16 { data: CudaSlice::<u16>::alloc(&x.device, n)?, numel: n },
        shape: x.shape().clone(),
        device: x.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
    };
    ensure(&x.device, "gelu_bf16_kernel", CUDA_GELU)?;
    let f = x.device.get_func("gelu_bf16_kernel", "gelu_bf16_kernel").ok_or_else(|| FlameError::Cuda("gelu_bf16_kernel missing".into()))?;
    let xs = match &x.storage { TensorStorage::BF16 { data, .. } => data, _ => return Err(FlameError::InvalidOperation("gelu_bf16 expects BF16".into()).into()) };
    let ys = match &mut out.storage { TensorStorage::BF16 { data, .. } => data, _ => unreachable!() };
    unsafe { f.launch(lc(n), (xs, ys, n as i64))?; }
    Ok(out)
}

pub fn square_bf16(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), DType::BF16);
    let n = x.shape().elem_count();
    let mut out = Tensor {
        storage: TensorStorage::BF16 { data: CudaSlice::<u16>::alloc(&x.device, n)?, numel: n },
        shape: x.shape().clone(),
        device: x.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
    };
    ensure(&x.device, "square_bf16_kernel", CUDA_SQUARE)?;
    let f = x.device.get_func("square_bf16_kernel", "square_bf16_kernel").ok_or_else(|| FlameError::Cuda("square_bf16_kernel missing".into()))?;
    let xs = match &x.storage { TensorStorage::BF16 { data, .. } => data, _ => return Err(FlameError::InvalidOperation("square_bf16 expects BF16".into()).into()) };
    let ys = match &mut out.storage { TensorStorage::BF16 { data, .. } => data, _ => unreachable!() };
    unsafe { f.launch(lc(n), (xs, ys, n as i64))?; }
    Ok(out)
}
