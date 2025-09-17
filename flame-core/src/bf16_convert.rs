#![cfg(feature = "bf16_u16")]
use crate::error::FlameError;
use cudarc::driver::{CudaDevice, CudaSlice, LaunchConfig, DeviceSlice};
use cudarc::nvrtc::compile_ptx;
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

fn ensure(dev: &Arc<CudaDevice>, name: &str, code: &str) -> Result<(), FlameError> {
    if dev.get_func(name, name).is_some() { return Ok(()); }
    let ptx = compile_ptx(code).map_err(|e| FlameError::Cuda(format!("nvrtc: {}", e)))?;
    dev.load_ptx(ptx, name, &[name]).map_err(|e| FlameError::Cuda(format!("load_ptx {}: {}", name, e)))?;
    Ok(())
}

#[inline]
fn lc_for(n: usize) -> LaunchConfig { LaunchConfig::for_num_elems(n as u32) }

pub fn bf16_u16_to_f32(dev: Arc<CudaDevice>, src: &CudaSlice<u16>, dst: &mut CudaSlice<f32>, n: usize) -> Result<(), FlameError> {
    ensure(&dev, "bf16_to_f32", CUDA_TO_F32)?;
    let f = dev.get_func("bf16_to_f32", "bf16_to_f32").ok_or_else(|| FlameError::Cuda("bf16_to_f32 missing".into()))?;
    unsafe { f.launch(lc_for(n), (src, dst, n as i64))?; }
    Ok(())
}

pub fn f32_to_bf16_u16(dev: Arc<CudaDevice>, src: &CudaSlice<f32>, dst: &mut CudaSlice<u16>, n: usize) -> Result<(), FlameError> {
    ensure(&dev, "f32_to_bf16", CUDA_TO_BF16)?;
    let f = dev.get_func("f32_to_bf16", "f32_to_bf16").ok_or_else(|| FlameError::Cuda("f32_to_bf16 missing".into()))?;
    unsafe { f.launch(lc_for(n), (src, dst, n as i64))?; }
    Ok(())
}

