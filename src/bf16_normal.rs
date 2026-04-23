use std::sync::Arc;
// Legacy bf16 normalization helpers.
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use crate::dtype::DType;
use crate::tensor_storage::{slice_ref, TensorStorage};
use crate::{Error, Result, Shape, Tensor, TensorId};

#[inline]
fn lc(n: usize) -> LaunchConfig {
    LaunchConfig::for_num_elems(n as u32)
}

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
    if crate::env_flags::alloc_log_enabled() {
        let bytes = n * DType::BF16.size_in_bytes();
        if bytes >= (8 << 20) {
            eprintln!(
                "[alloc] tag=normal_bf16 dtype=BF16 shape={:?} bytes={}",
                shape.dims(),
                bytes
            );
        }
    }
    let data = unsafe { device.alloc::<u16>(n) }
        .map_err(|e| Error::Cuda(format!("alloc normal bf16: {:?}", e)))?;
    let mut out = Tensor {
        storage: TensorStorage::BF16 {
            data: data.into(),
            numel: n,
        },
        shape,
        device: device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,

    };
    if device
        .get_func("normal_bf16_kernel", "normal_bf16_kernel")
        .is_none()
    {
        let include_dir = std::env::var("CUDA_INCLUDE_DIR")
            .or_else(|_| std::env::var("CUDA_HOME").map(|home| format!("{home}/include")))
            .unwrap_or_else(|_| "/usr/local/cuda/include".into());
        let mut opts = CompileOptions::default();
        opts.include_paths.push(include_dir.into());
        let ptx = compile_ptx_with_opts(CUDA_NORMAL_BF16, opts)
            .map_err(|e| Error::Cuda(format!("nvrtc normal_bf16: {:?}", e)))?;
        device
            .load_ptx(ptx, "normal_bf16_kernel", &["normal_bf16_kernel"])
            .map_err(|e| Error::Cuda(format!("load normal_bf16: {:?}", e)))?;
    }
    let f = device
        .get_func("normal_bf16_kernel", "normal_bf16_kernel")
        .ok_or_else(|| Error::Cuda("normal_bf16_kernel missing".into()))?;
    let ys = match &out.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => unreachable!(),
    };
    unsafe {
        f.launch(lc(n), (slice_ref(ys), n as i64, seed, mean, std))?;
    }
    Ok(out)
}
