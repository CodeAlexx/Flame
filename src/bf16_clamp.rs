// Legacy bf16 clamp helpers.
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use crate::dtype::DType;
use crate::tensor::contracts::trap_is_bf16;
use crate::tensor_storage::{slice_ref, TensorStorage};
use crate::{Error, Result, Shape, Tensor, TensorId};

#[inline]
fn lc(n: usize) -> LaunchConfig {
    LaunchConfig::for_num_elems(n as u32)
}

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
    trap_is_bf16("clamp_bf16::in", x)?;
    debug_assert_eq!(x.dtype(), DType::BF16);
    let n = x.shape().elem_count();
    let data = unsafe { x.device.alloc::<u16>(n) }
        .map_err(|e| Error::Cuda(format!("alloc clamp bf16: {:?}", e)))?;
    let mut out = Tensor {
        storage: TensorStorage::BF16 {
            data: data.into(),
            numel: n,
        },
        shape: x.shape().clone(),
        device: x.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
    };
    if x.device
        .get_func("clamp_bf16_kernel", "clamp_bf16_kernel")
        .is_none()
    {
        let include_dir = std::env::var("CUDA_INCLUDE_DIR")
            .or_else(|_| std::env::var("CUDA_HOME").map(|home| format!("{home}/include")))
            .unwrap_or_else(|_| "/usr/local/cuda/include".into());
        let mut opts = CompileOptions::default();
        opts.include_paths.push(include_dir.into());
        let ptx = compile_ptx_with_opts(CUDA_CLAMP_BF16, opts)
            .map_err(|e| Error::Cuda(format!("nvrtc clamp_bf16: {:?}", e)))?;
        x.device
            .load_ptx(ptx, "clamp_bf16_kernel", &["clamp_bf16_kernel"])
            .map_err(|e| Error::Cuda(format!("load clamp_bf16: {:?}", e)))?;
    }
    let f = x
        .device
        .get_func("clamp_bf16_kernel", "clamp_bf16_kernel")
        .ok_or_else(|| Error::Cuda("clamp_bf16_kernel missing".into()))?;
    let xs = match &x.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => unreachable!(),
    };
    let ys = match &out.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => unreachable!(),
    };
    unsafe {
        f.launch(lc(n), (slice_ref(xs), slice_ref(ys), n as i64, lo, hi))?;
    }
    trap_is_bf16("clamp_bf16::out", &out)?;
    Ok(out)
}
