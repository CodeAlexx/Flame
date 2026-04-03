use std::sync::Arc;
// Legacy bf16 ops retained for backwards compatibility.
use crate::dtype::DType;
use crate::tensor_storage::{ensure_unique_slice, slice_ref, TensorStorage};
use crate::{Error, Result, Shape, Tensor, TensorId};
use cudarc::{
    driver::{CudaDevice, LaunchAsync, LaunchConfig},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};

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

const CUDA_SILU: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void silu_bf16_kernel(const __nv_bfloat16* __restrict__ X,
                      __nv_bfloat16* __restrict__ Y,
                      long n) {
  long i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < n) {
    float x = __bfloat162float(X[i]);
    float y = x / (1.0f + expf(-x));
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

fn ensure(dev: &Arc<CudaDevice>, name: &'static str, code: &'static str) -> Result<()> {
    if dev.get_func(name, name).is_some() {
        return Ok(());
    }
    let include_dir = std::env::var("CUDA_INCLUDE_DIR")
        .or_else(|_| std::env::var("CUDA_HOME").map(|home| format!("{home}/include")))
        .unwrap_or_else(|_| "/usr/local/cuda/include".into());
    let mut opts = CompileOptions::default();
    opts.include_paths.push(include_dir.into());
    let ptx = compile_ptx_with_opts(code, opts)
        .map_err(|e| Error::Cuda(format!("nvrtc {}: {}", name, e)))?;
    dev.load_ptx(ptx, name, &[name])
        .map_err(|e| Error::Cuda(format!("load_ptx {}: {}", name, e)))?;
    Ok(())
}

#[inline]
fn lc(n: usize) -> LaunchConfig {
    LaunchConfig::for_num_elems(n as u32)
}

pub fn gelu_bf16(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), DType::BF16);
    let n = x.shape().elem_count();
    let data = unsafe { x.device.alloc::<u16>(n) }
        .map_err(|e| Error::Cuda(format!("alloc gelu bf16: {}", e)))?;
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
    ensure(&x.device, "gelu_bf16_kernel", CUDA_GELU)?;
    let f = x
        .device
        .get_func("gelu_bf16_kernel", "gelu_bf16_kernel")
        .ok_or_else(|| Error::Cuda("gelu_bf16_kernel missing".into()))?;
    let xs = match &x.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => return Err(Error::InvalidOperation("gelu_bf16 expects BF16".into())),
    };
    let ys = match &mut out.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => unreachable!(),
    };
    let ys = ensure_unique_slice(ys)?;
    unsafe {
        f.launch(lc(n), (slice_ref(xs), ys, n as i64))?;
    }
    Ok(out)
}

pub fn square_bf16(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), DType::BF16);
    let n = x.shape().elem_count();
    let data = unsafe { x.device.alloc::<u16>(n) }
        .map_err(|e| Error::Cuda(format!("alloc square bf16: {}", e)))?;
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
    ensure(&x.device, "square_bf16_kernel", CUDA_SQUARE)?;
    let f = x
        .device
        .get_func("square_bf16_kernel", "square_bf16_kernel")
        .ok_or_else(|| Error::Cuda("square_bf16_kernel missing".into()))?;
    let xs = match &x.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => return Err(Error::InvalidOperation("square_bf16 expects BF16".into())),
    };
    let ys = match &mut out.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => unreachable!(),
    };
    let ys = ensure_unique_slice(ys)?;
    unsafe {
        f.launch(lc(n), (slice_ref(xs), ys, n as i64))?;
    }
    Ok(out)
}

pub fn silu_bf16(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), DType::BF16);
    let n = x.shape().elem_count();
    let data = unsafe { x.device.alloc::<u16>(n) }
        .map_err(|e| Error::Cuda(format!("alloc silu bf16: {}", e)))?;
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
    ensure(&x.device, "silu_bf16_kernel", CUDA_SILU)?;
    let f = x
        .device
        .get_func("silu_bf16_kernel", "silu_bf16_kernel")
        .ok_or_else(|| Error::Cuda("silu_bf16_kernel missing".into()))?;
    let xs = match &x.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => return Err(Error::InvalidOperation("silu_bf16 expects BF16".into())),
    };
    let ys = match &mut out.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => unreachable!(),
    };
    let ys = ensure_unique_slice(ys)?;
    unsafe {
        f.launch(lc(n), (slice_ref(xs), ys, n as i64))?;
    }
    Ok(out)
}
