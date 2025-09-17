#![cfg(feature = "bf16_u16")]
use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use crate::{Tensor, TensorId, Shape, Result, FlameError};
use crate::tensor_storage::TensorStorage;
use crate::dtype::DType;

#[inline]
fn lc(n: usize) -> LaunchConfig { LaunchConfig::for_num_elems(n as u32) }

/// CUDA: Box–Muller using two xorshift32 uniforms; emits N(0,1) then scale/shift
const CUDA_NORMAL_BF16: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void normal_bf16_kernel(__nv_bfloat16* __restrict__ Y,
                        long n, unsigned long long seed,
                        float mean, float std) {
  long i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long s = seed;
  while (i < n) {
    // xorshift32 #1
    unsigned int x1 = (unsigned int)( (i<<1) ^ (s & 0xffffffffULL) ^ (s >> 32) );
    x1 ^= x1 << 13; x1 ^= x1 >> 17; x1 ^= x1 << 5;
    float u1 = (x1 * (1.0f / 4294967296.0f));
    u1 = fmaxf(u1, 1e-7f);

    // xorshift32 #2
    unsigned int x2 = (unsigned int)( (i<<1 | 1) ^ (s & 0xffffffffULL) ^ (s >> 32) );
    x2 ^= x2 << 13; x2 ^= x2 >> 17; x2 ^= x2 << 5;
    float u2 = (x2 * (1.0f / 4294967296.0f));

    float r   = sqrtf(-2.0f * logf(u1));
    float ang = 6.283185307179586f * u2; // 2π
    float z   = r * cosf(ang);           // one normal
    float y   = mean + std * z;
    Y[i] = __float2bfloat16(y);
    i += gridDim.x * blockDim.x;
  }
}
"#;

/// BF16 normal initializer: mean/std, no FP32 tensor materialized.
pub fn normal_bf16(
    shape: Shape,
    mean: f32,
    std: f32,
    seed: u64,
    device: Arc<CudaDevice>,
) -> Result<Tensor> {
    let n = shape.elem_count();
    if std::env::var("ALLOC_LOG").ok().as_deref() == Some("1") {
        let bytes = n * DType::BF16.size_in_bytes();
        if bytes >= (8 << 20) {
            eprintln!("[alloc] tag=normal_bf16 dtype=BF16 shape={:?} bytes={}", shape.dims(), bytes);
        }
    }
    let mut out = Tensor {
        storage: TensorStorage::BF16 { data: CudaSlice::<u16>::alloc(&device, n)?, numel: n },
        shape,
        device: device.clone(),
        id: TensorId::new(),
        requires_grad: false,
    };
    if device.get_func("normal_bf16_kernel", "normal_bf16_kernel").is_none() {
        let ptx = compile_ptx(CUDA_NORMAL_BF16)
            .map_err(|e| FlameError::Cuda(format!("nvrtc normal_bf16: {}", e)))?;
        device
            .load_ptx(ptx, "normal_bf16_kernel", &["normal_bf16_kernel"])
            .map_err(|e| FlameError::Cuda(format!("load normal_bf16: {}", e)))?;
    }
    let f = device
        .get_func("normal_bf16_kernel", "normal_bf16_kernel")
        .ok_or_else(|| FlameError::Cuda("normal_bf16_kernel missing".into()))?;
    let ys = match &out.storage { TensorStorage::BF16 { data, .. } => data, _ => unreachable!() };
    unsafe { f.launch(lc(n), (ys, n as i64, seed as u64, mean, std))?; }
    Ok(out)
}

