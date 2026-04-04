use std::sync::Arc;
// Legacy bf16 factory helpers.
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use crate::dtype::DType;
use crate::tensor_storage::{slice_ref, TensorStorage};
use crate::{Error, Result, Shape, Tensor, TensorId};

#[inline]
fn lc(n: usize) -> LaunchConfig {
    LaunchConfig::for_num_elems(n as u32)
}

/// Allocate a BF16 (u16) tensor of zeros (no FP32 materialization).
pub fn zeros_bf16(shape: Shape, device: Arc<CudaDevice>) -> Result<Tensor> {
    let n = shape.elem_count();
    if std::env::var("ALLOC_LOG").ok().as_deref() == Some("1") {
        let bytes = n * DType::BF16.size_in_bytes();
        if bytes >= (8 << 20) {
            eprintln!(
                "[alloc] tag=zeros_bf16 dtype=BF16 shape={:?} bytes={}",
                shape.dims(),
                bytes
            );
        }
    }
    let mut buf =
        unsafe { device.alloc::<u16>(n) }.map_err(|e| Error::Cuda(format!("alloc bf16: {:?}", e)))?;
    device.memset_zeros(&mut buf)?;
    Ok(Tensor {
        storage: TensorStorage::BF16 {
            data: buf.into(),
            numel: n,
        },
        shape,
        device,
        id: TensorId::new(),
        requires_grad: false,
    })
}

/// Simple counter-based RNG (xorshift32) → FP32 in (0,1), store as BF16.
/// Range: low..high applied in-kernel. No FP32 tensor allocation.
const CUDA_UNIFORM_BF16: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void uniform_bf16_kernel(__nv_bfloat16* __restrict__ Y,
                         long n, unsigned long long seed,
                         float low, float high) {
  long i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long s = seed;
  while (i < n) {
    // xorshift32 using different counter per idx
    unsigned int x = (unsigned int)(i ^ (s & 0xffffffffULL) ^ (s >> 32));
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    // [0,1)
    float u = (x * (1.0f / 4294967296.0f));
    float v = low + (high - low) * u;
    Y[i] = __float2bfloat16(v);
    i += gridDim.x * blockDim.x;
  }
}
"#;

/// Create BF16 tensor with uniform(low, high) values, RNG in-kernel (FP32 math in regs).
pub fn uniform_bf16(
    shape: Shape,
    low: f32,
    high: f32,
    seed: u64,
    device: Arc<CudaDevice>,
) -> Result<Tensor> {
    let n = shape.elem_count();
    if std::env::var("ALLOC_LOG").ok().as_deref() == Some("1") {
        let bytes = n * DType::BF16.size_in_bytes();
        if bytes >= (8 << 20) {
            eprintln!(
                "[alloc] tag=uniform_bf16 dtype=BF16 shape={:?} bytes={}",
                shape.dims(),
                bytes
            );
        }
    }
    // Output buffer (BF16)
    let data = unsafe { device.alloc::<u16>(n) }
        .map_err(|e| Error::Cuda(format!("alloc uniform bf16: {:?}", e)))?;
    let mut out = Tensor {
        storage: TensorStorage::BF16 {
            data: data.into(),
            numel: n,
        },
        shape,
        device: device.clone(),
        id: TensorId::new(),
        requires_grad: false,
    };

    // Compile once and launch
    if device
        .get_func("uniform_bf16_kernel", "uniform_bf16_kernel")
        .is_none()
    {
        let include_dir = std::env::var("CUDA_INCLUDE_DIR")
            .or_else(|_| std::env::var("CUDA_HOME").map(|home| format!("{home}/include")))
            .unwrap_or_else(|_| "/usr/local/cuda/include".into());
        let mut opts = CompileOptions::default();
        opts.include_paths.push(include_dir.into());
        let ptx = compile_ptx_with_opts(CUDA_UNIFORM_BF16, opts)
            .map_err(|e| Error::Cuda(format!("nvrtc uniform_bf16: {:?}", e)))?;
        device
            .load_ptx(ptx, "uniform_bf16_kernel", &["uniform_bf16_kernel"])
            .map_err(|e| Error::Cuda(format!("load uniform_bf16: {:?}", e)))?;
    }
    let f = device
        .get_func("uniform_bf16_kernel", "uniform_bf16_kernel")
        .ok_or_else(|| Error::Cuda("uniform_bf16_kernel missing".into()))?;

    let ys = match &out.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => unreachable!(),
    };

    unsafe {
        f.launch(lc(n), (slice_ref(ys), n as i64, seed, low, high))?;
    }
    Ok(out)
}
