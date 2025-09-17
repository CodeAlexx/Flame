#![cfg(feature = "bf16_u16")]
use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use crate::{Tensor, TensorId, Shape, Result, FlameError};
use crate::tensor_storage::TensorStorage;
use crate::dtype::DType;

#[derive(Debug, Clone)]
pub struct BcSpec {
    pub out_dims: Vec<i64>,
    pub a_strides: Vec<i64>,
    pub b_strides: Vec<i64>,
    pub total: i64,
}

/// Build broadcast spec (NumPy rules). Set stride=0 for broadcasted dims.
pub fn make_broadcast_spec(a_dims:&[i64], b_dims:&[i64]) -> BcSpec {
    let mut ad = a_dims.to_vec();
    let mut bd = b_dims.to_vec();
    let tr = ad.len().max(bd.len());
    while ad.len() < tr { ad.insert(0, 1); }
    while bd.len() < tr { bd.insert(0, 1); }

    let mut out = Vec::with_capacity(tr);
    for i in 0..tr {
        let (ai, bi) = (ad[i], bd[i]);
        assert!(ai==bi || ai==1 || bi==1, "broadcast mismatch at dim {}: {} vs {}", i, ai, bi);
        out.push(ai.max(bi));
    }

    fn compact_strides(dims:&[i64]) -> Vec<i64> {
        let mut st = vec![0; dims.len()];
        let mut s  = 1i64;
        for i in (0..dims.len()).rev() {
            st[i] = if dims[i]==1 { 0 } else { s };
            s *= dims[i].max(1);
        }
        st
    }
    let mut a_st = compact_strides(&ad);
    let mut b_st = compact_strides(&bd);
    for i in 0..tr { if ad[i]==1 { a_st[i]=0; } }
    for i in 0..tr { if bd[i]==1 { b_st[i]=0; } }

    let total = out.iter().product::<i64>();
    BcSpec{ out_dims: out, a_strides: a_st, b_strides: b_st, total }
}

const CUDA_ADD_MUL_BF16: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void add_bf16_kernel(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* O,
                     const long* out_dims, const long* a_strides, const long* b_strides,
                     int nd, long total) {
  long tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < total) {
    long rem = tid, a_off = 0, b_off = 0;
    #pragma unroll
    for (int i=nd-1; i>=0; --i) {
      long d = out_dims[i];
      long idx = rem % d; rem /= d;
      a_off += idx * a_strides[i];
      b_off += idx * b_strides[i];
    }
    float a = __bfloat162float(A[a_off]);
    float b = __bfloat162float(B[b_off]);
    O[tid]  = __float2bfloat16(a + b);
    tid += gridDim.x * blockDim.x;
  }
}
extern "C" __global__
void mul_bf16_kernel(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* O,
                     const long* out_dims, const long* a_strides, const long* b_strides,
                     int nd, long total) {
  long tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < total) {
    long rem = tid, a_off = 0, b_off = 0;
    #pragma unroll
    for (int i=nd-1; i>=0; --i) {
      long d = out_dims[i];
      long idx = rem % d; rem /= d;
      a_off += idx * a_strides[i];
      b_off += idx * b_strides[i];
    }
    float a = __bfloat162float(A[a_off]);
    float b = __bfloat162float(B[b_off]);
    O[tid]  = __float2bfloat16(a * b);
    tid += gridDim.x * blockDim.x;
  }
}
"#;

const CUDA_CMP_BF16: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void cmp_bf16_kernel(const __nv_bfloat16* A, const __nv_bfloat16* B, unsigned char* O,
                     const long* out_dims, const long* a_strides, const long* b_strides,
                     int nd, long total, int op) {
  // op: 0=ge,1=gt,2=le,3=lt,4=ne
  long tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < total) {
    long rem = tid, a_off = 0, b_off = 0;
    #pragma unroll
    for (int i=nd-1; i>=0; --i) {
      long d = out_dims[i];
      long idx = rem % d; rem /= d;
      a_off += idx * a_strides[i];
      b_off += idx * b_strides[i];
    }
    float a = __bfloat162float(A[a_off]);
    float b = __bfloat162float(B[b_off]);
    unsigned char m = 0;
    switch (op) {
      case 0: m = (a >= b); break;
      case 1: m = (a >  b); break;
      case 2: m = (a <= b); break;
      case 3: m = (a <  b); break;
      default: m = (a != b); break;
    }
    O[tid] = m;
    tid += gridDim.x * blockDim.x;
  }
}
"#;

#[inline]
fn lc(n: usize) -> LaunchConfig { LaunchConfig::for_num_elems(n as u32) }

fn ensure(dev:&Arc<CudaDevice>, nm:&str, code:&str)->Result<(),FlameError>{
    if dev.get_func(nm,nm).is_some(){ return Ok(()); }
    let ptx = compile_ptx(code).map_err(|e|FlameError::Cuda(format!("nvrtc {}: {}", nm, e)))?;
    dev.load_ptx(ptx, nm, &[nm]).map_err(|e|FlameError::Cuda(format!("load {}: {}", nm, e)))
}

pub fn add_bf16(a:&Tensor, b:&Tensor) -> Result<Tensor> {
    debug_assert_eq!(a.dtype(), DType::BF16);
    debug_assert_eq!(b.dtype(), DType::BF16);
    let spec = make_broadcast_spec(a.shape().dims(), b.shape().dims());
    let n = spec.total as usize;

    let out = Tensor {
        storage: TensorStorage::BF16 { data: CudaSlice::<u16>::alloc(&a.device, n)?, numel: n },
        shape: Shape::from_dims(&spec.out_dims),
        device: a.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
    };

    ensure(&a.device, "add_bf16_kernel", CUDA_ADD_MUL_BF16)?;
    let f = a.device
        .get_func("add_bf16_kernel", "add_bf16_kernel")
        .ok_or_else(|| FlameError::Cuda("missing kernel: add_bf16_kernel".into()))?;

    // Upload small metadata buffers
    let out_dims = a.device.htod_sync_copy(&spec.out_dims)?;
    let a_strides = a.device.htod_sync_copy(&spec.a_strides)?;
    let b_strides = a.device.htod_sync_copy(&spec.b_strides)?;

    let aslice = if let TensorStorage::BF16 { data, .. } = &a.storage { data } else { return Err(FlameError::InvalidOperation("expected BF16 storage for a".into())); };
    let bslice = if let TensorStorage::BF16 { data, .. } = &b.storage { data } else { return Err(FlameError::InvalidOperation("expected BF16 storage for b".into())); };
    let oslice = if let TensorStorage::BF16 { data, .. } = &out.storage { data } else { return Err(FlameError::InvalidOperation("expected BF16 storage for out".into())); };
    unsafe { f.launch(lc(n), (aslice, bslice, oslice, &out_dims, &a_strides, &b_strides, spec.out_dims.len() as i32, spec.total))
        .map_err(|e| FlameError::Training(e.to_string()))?; }
    Ok(out)
}

pub fn mul_bf16(a:&Tensor, b:&Tensor) -> Result<Tensor> {
    debug_assert_eq!(a.dtype(), DType::BF16);
    debug_assert_eq!(b.dtype(), DType::BF16);
    let spec = make_broadcast_spec(a.shape().dims(), b.shape().dims());
    let n = spec.total as usize;

    let out = Tensor {
        storage: TensorStorage::BF16 { data: CudaSlice::<u16>::alloc(&a.device, n)?, numel: n },
        shape: Shape::from_dims(&spec.out_dims),
        device: a.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
    };

    ensure(&a.device, "mul_bf16_kernel", CUDA_ADD_MUL_BF16)?;
    let f = a.device
        .get_func("mul_bf16_kernel", "mul_bf16_kernel")
        .ok_or_else(|| FlameError::Cuda("missing kernel: mul_bf16_kernel".into()))?;

    let out_dims = a.device.htod_sync_copy(&spec.out_dims)?;
    let a_strides = a.device.htod_sync_copy(&spec.a_strides)?;
    let b_strides = a.device.htod_sync_copy(&spec.b_strides)?;

    let aslice = if let TensorStorage::BF16 { data, .. } = &a.storage { data } else { return Err(FlameError::InvalidOperation("expected BF16 storage for a".into())); };
    let bslice = if let TensorStorage::BF16 { data, .. } = &b.storage { data } else { return Err(FlameError::InvalidOperation("expected BF16 storage for b".into())); };
    let oslice = if let TensorStorage::BF16 { data, .. } = &out.storage { data } else { return Err(FlameError::InvalidOperation("expected BF16 storage for out".into())); };
    unsafe { f.launch(lc(n), (aslice, bslice, oslice, &out_dims, &a_strides, &b_strides, spec.out_dims.len() as i32, spec.total))
        .map_err(|e| FlameError::Training(e.to_string()))?; }
    Ok(out)
}

fn cmp_bf16(a:&Tensor, b:&Tensor, op:i32)->Result<Tensor> {
    debug_assert_eq!(a.dtype(), DType::BF16);
    debug_assert_eq!(b.dtype(), DType::BF16);
    let spec = make_broadcast_spec(a.shape().dims(), b.shape().dims());
    let n = spec.total as usize;

    let out = Tensor {
        storage: TensorStorage::Bool { data: CudaSlice::<f32>::alloc(&a.device, n)?, numel: n },
        shape: Shape::from_dims(&spec.out_dims),
        device: a.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
    };

    ensure(&a.device, "cmp_bf16_kernel", CUDA_CMP_BF16)?;
    let f = a.device
        .get_func("cmp_bf16_kernel", "cmp_bf16_kernel")
        .ok_or_else(|| FlameError::Cuda("missing kernel: cmp_bf16_kernel".into()))?;

    let out_dims = a.device.htod_sync_copy(&spec.out_dims)?;
    let a_strides = a.device.htod_sync_copy(&spec.a_strides)?;
    let b_strides = a.device.htod_sync_copy(&spec.b_strides)?;

    let aslice = if let TensorStorage::BF16 { data, .. } = &a.storage { data } else { return Err(FlameError::InvalidOperation("expected BF16 storage for a".into())); };
    let bslice = if let TensorStorage::BF16 { data, .. } = &b.storage { data } else { return Err(FlameError::InvalidOperation("expected BF16 storage for b".into())); };
    let oslice = if let TensorStorage::Bool { data, .. } = &out.storage { data } else { return Err(FlameError::InvalidOperation("expected Bool storage for out".into())); };
    unsafe { f.launch(lc(n), (aslice, bslice, oslice, &out_dims, &a_strides, &b_strides, spec.out_dims.len() as i32, spec.total, op))
        .map_err(|e| FlameError::Training(e.to_string()))?; }
    Ok(out)
}

// Public compare helpers
pub fn ge_bf16(a:&Tensor,b:&Tensor)->Result<Tensor>{ cmp_bf16(a,b,0) }
pub fn gt_bf16(a:&Tensor,b:&Tensor)->Result<Tensor>{ cmp_bf16(a,b,1) }
pub fn le_bf16(a:&Tensor,b:&Tensor)->Result<Tensor>{ cmp_bf16(a,b,2) }
pub fn lt_bf16(a:&Tensor,b:&Tensor)->Result<Tensor>{ cmp_bf16(a,b,3) }
pub fn ne_bf16(a:&Tensor,b:&Tensor)->Result<Tensor>{ cmp_bf16(a,b,4) }
