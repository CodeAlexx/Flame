#![cfg(feature = "bf16_u16")]
use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use crate::{Tensor, TensorId, Shape, Result, FlameError};
use crate::tensor_storage::TensorStorage;
use crate::dtype::DType;

#[inline]
fn lc(n: usize) -> LaunchConfig { LaunchConfig::for_num_elems(n as u32) }

const CUDA_CLAMP_BF16: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void clamp_bf16_kernel(const __nv_bfloat16* __restrict__ X,
                       __nv_bfloat16* __restrict__ Y,
                       long n, float lo, float hi) {
  long i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < n) {
    float x = __bfloat162float(X[i]);
    if (x < lo) x = lo;
    if (x > hi) x = hi;
    Y[i] = __float2bfloat16(x);
    i += gridDim.x * blockDim.x;
  }
}
"#;

pub fn clamp_bf16(x: &Tensor, lo: f32, hi: f32) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), DType::BF16);
    let n = x.shape().elem_count();
    let mut out = Tensor {
        storage: TensorStorage::BF16 { data: CudaSlice::<u16>::alloc(&x.device, n)?, numel: n },
        shape: x.shape().clone(),
        device: x.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
    };
    if x.device.get_func("clamp_bf16_kernel", "clamp_bf16_kernel").is_none() {
        let ptx = compile_ptx(CUDA_CLAMP_BF16)
            .map_err(|e| FlameError::Cuda(format!("nvrtc clamp_bf16: {}", e)))?;
        x.device
            .load_ptx(ptx, "clamp_bf16_kernel", &["clamp_bf16_kernel"])
            .map_err(|e| FlameError::Cuda(format!("load clamp_bf16: {}", e)))?;
    }
    let f = x
        .device
        .get_func("clamp_bf16_kernel", "clamp_bf16_kernel")
        .ok_or_else(|| FlameError::Cuda("clamp_bf16_kernel missing".into()))?;
    let xs = match &x.storage { TensorStorage::BF16 { data, .. } => data, _ => unreachable!() };
    let ys = match &out.storage { TensorStorage::BF16 { data, .. } => data, _ => unreachable!() };
    unsafe { f.launch(lc(n), (xs, ys, n as i64, lo, hi))?; }
    Ok(out)
}

