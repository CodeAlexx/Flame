use crate::{error::Error, Result};
// Legacy bf16 conversion kernels.
use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;

const CUDA_TO_F32: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void bf16_to_f32(const __nv_bfloat16* __restrict__ X,
                 float* __restrict__ Y, long n){
  long i = blockIdx.x * blockDim.x + threadIdx.x;
  while(i<n){ Y[i]=__bfloat162float(X[i]); i+=gridDim.x*blockDim.x; }
}
"#;

const CUDA_TO_BF16: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void f32_to_bf16(const float* __restrict__ X,
                 __nv_bfloat16* __restrict__ Y, long n){
  long i = blockIdx.x * blockDim.x + threadIdx.x;
  while(i<n){ Y[i]=__float2bfloat16(X[i]); i+=gridDim.x*blockDim.x; }
}
"#;

fn nvrtc_include_opt() -> String {
    std::env::var("CUDA_HOME")
        .map(|p| format!("{}/include", p))
        .unwrap_or_else(|_| "/usr/local/cuda/include".to_string())
}

fn ensure(dev: &Arc<CudaDevice>, name: &'static str, code: &'static str) -> Result<()> {
    if dev.get_func(name, name).is_some() {
        return Ok(());
    }
    let include = nvrtc_include_opt();
    let mut opts = CompileOptions::default();
    opts.include_paths.push(include);
    opts.use_fast_math = Some(false);
    opts.fmad = Some(true);
    let ptx =
        compile_ptx_with_opts(code, opts).map_err(|e| Error::Cuda(format!("nvrtc: {:?}", e)))?;
    dev.load_ptx(ptx, name, &[name])
        .map_err(|e| Error::Cuda(format!("load_ptx {}: {:?}", name, e)))?;
    Ok(())
}

#[inline]
fn lc_for(n: usize) -> LaunchConfig {
    LaunchConfig::for_num_elems(n as u32)
}

pub fn bf16_u16_to_f32(
    dev: Arc<CudaDevice>,
    src: u64,
    dst: &mut CudaSlice<f32>,
    n: usize,
) -> Result<()> {
    ensure(&dev, "bf16_to_f32", CUDA_TO_F32)?;
    let f = dev
        .get_func("bf16_to_f32", "bf16_to_f32")
        .ok_or_else(|| Error::Cuda("bf16_to_f32 missing".into()))?;
    unsafe {
        f.launch(lc_for(n), (src, dst, n as i64))?;
    }
    Ok(())
}

pub fn f32_to_bf16_u16(
    dev: Arc<CudaDevice>,
    src: &CudaSlice<f32>,
    dst: u64,
    n: usize,
) -> Result<()> {
    ensure(&dev, "f32_to_bf16", CUDA_TO_BF16)?;
    let f = dev
        .get_func("f32_to_bf16", "f32_to_bf16")
        .ok_or_else(|| Error::Cuda("f32_to_bf16 missing".into()))?;
    unsafe {
        f.launch(lc_for(n), (src, dst, n as i64))?;
    }
    Ok(())
}
