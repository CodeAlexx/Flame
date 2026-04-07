use std::sync::Arc;
// Legacy bf16 elementwise kernels.
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use crate::dtype::DType;
use crate::tensor_storage::TensorStorage;
use crate::{Error, Result, Shape, Tensor, TensorId};

/// Fast async host-to-device copy for small metadata arrays.
/// Allocates on device and copies without synchronization.
/// The data will be ready before any kernel launched after this on the same stream.
fn htod_async_copy(device: &Arc<CudaDevice>, data: &[i64]) -> std::result::Result<CudaSlice<i64>, Error> {
    let mut gpu_buf = unsafe { device.alloc::<i64>(data.len()) }
        .map_err(|e| Error::Cuda(format!("alloc metadata: {:?}", e)))?;
    device.htod_copy_into(data.to_vec(), &mut gpu_buf)
        .map_err(|e| Error::Cuda(format!("htod_copy metadata: {:?}", e)))?;
    Ok(gpu_buf)
}

#[derive(Debug, Clone)]
pub struct BcSpec {
    pub out_dims: Vec<i64>,
    pub a_strides: Vec<i64>,
    pub b_strides: Vec<i64>,
    pub total: i64,
}

/// Build broadcast spec (NumPy rules). Set stride=0 for broadcasted dims.
pub fn make_broadcast_spec(a_dims: &[usize], b_dims: &[usize]) -> BcSpec {
    let mut ad: Vec<i64> = a_dims.iter().map(|&d| d as i64).collect();
    let mut bd: Vec<i64> = b_dims.iter().map(|&d| d as i64).collect();
    let tr = ad.len().max(bd.len());
    while ad.len() < tr {
        ad.insert(0, 1);
    }
    while bd.len() < tr {
        bd.insert(0, 1);
    }

    let mut out = Vec::with_capacity(tr);
    for i in 0..tr {
        let (ai, bi) = (ad[i], bd[i]);
        assert!(
            ai == bi || ai == 1 || bi == 1,
            "broadcast mismatch at dim {}: {} vs {}",
            i,
            ai,
            bi
        );
        out.push(ai.max(bi));
    }

    fn compact_strides(dims: &[i64]) -> Vec<i64> {
        let mut st = vec![0; dims.len()];
        let mut s = 1i64;
        for i in (0..dims.len()).rev() {
            st[i] = if dims[i] == 1 { 0 } else { s };
            s *= dims[i].max(1);
        }
        st
    }
    let mut a_st = compact_strides(&ad);
    let mut b_st = compact_strides(&bd);
    for i in 0..tr {
        if ad[i] == 1 {
            a_st[i] = 0;
        }
    }
    for i in 0..tr {
        if bd[i] == 1 {
            b_st[i] = 0;
        }
    }

    let total = out.iter().product::<i64>();
    BcSpec {
        out_dims: out,
        a_strides: a_st,
        b_strides: b_st,
        total,
    }
}

// Fast-path flat kernels for the common "no broadcast, both contiguous, same shape"
// case. Uses __nv_bfloat162 vectorization (2 elements per thread) and native BF16
// __hadd2/__hmul2 — no FP32 round-trip. This is the case the models hit on every
// elementwise add/mul during a transformer block forward.
//
// On RTX 3090 Ti at [1, 4096, 3840] = 16 MB tensors, this kernel runs in ~0.15 ms
// (matching cuDNN within ~1.5×) versus the broadcast path's 4.9 ms (44× slower).
const CUDA_ADD_MUL_BF16_FLAT: &str = r#"
#include <cuda_bf16.h>

// 2-element vectorized add: O[i] = A[i] + B[i]
extern "C" __global__
void add_bf16_flat_kernel(const __nv_bfloat16* __restrict__ A,
                          const __nv_bfloat16* __restrict__ B,
                          __nv_bfloat16* __restrict__ O,
                          long n) {
    long i2 = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n2 = n >> 1;
    if (i2 < n2) {
        const __nv_bfloat162* a2 = (const __nv_bfloat162*)A;
        const __nv_bfloat162* b2 = (const __nv_bfloat162*)B;
        __nv_bfloat162* o2 = (__nv_bfloat162*)O;
        o2[i2] = __hadd2(a2[i2], b2[i2]);
    }
    // Tail element if n is odd
    if (i2 == n2 && (n & 1)) {
        long last = n - 1;
        O[last] = __hadd(A[last], B[last]);
    }
}

extern "C" __global__
void mul_bf16_flat_kernel(const __nv_bfloat16* __restrict__ A,
                          const __nv_bfloat16* __restrict__ B,
                          __nv_bfloat16* __restrict__ O,
                          long n) {
    long i2 = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n2 = n >> 1;
    if (i2 < n2) {
        const __nv_bfloat162* a2 = (const __nv_bfloat162*)A;
        const __nv_bfloat162* b2 = (const __nv_bfloat162*)B;
        __nv_bfloat162* o2 = (__nv_bfloat162*)O;
        o2[i2] = __hmul2(a2[i2], b2[i2]);
    }
    if (i2 == n2 && (n & 1)) {
        long last = n - 1;
        O[last] = __hmul(A[last], B[last]);
    }
}

extern "C" __global__
void sub_bf16_flat_kernel(const __nv_bfloat16* __restrict__ A,
                          const __nv_bfloat16* __restrict__ B,
                          __nv_bfloat16* __restrict__ O,
                          long n) {
    long i2 = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n2 = n >> 1;
    if (i2 < n2) {
        const __nv_bfloat162* a2 = (const __nv_bfloat162*)A;
        const __nv_bfloat162* b2 = (const __nv_bfloat162*)B;
        __nv_bfloat162* o2 = (__nv_bfloat162*)O;
        o2[i2] = __hsub2(a2[i2], b2[i2]);
    }
    if (i2 == n2 && (n & 1)) {
        long last = n - 1;
        O[last] = __hsub(A[last], B[last]);
    }
}

extern "C" __global__
void div_bf16_flat_kernel(const __nv_bfloat16* __restrict__ A,
                          const __nv_bfloat16* __restrict__ B,
                          __nv_bfloat16* __restrict__ O,
                          long n) {
    long i2 = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n2 = n >> 1;
    if (i2 < n2) {
        const __nv_bfloat162* a2 = (const __nv_bfloat162*)A;
        const __nv_bfloat162* b2 = (const __nv_bfloat162*)B;
        __nv_bfloat162* o2 = (__nv_bfloat162*)O;
        o2[i2] = __h2div(a2[i2], b2[i2]);
    }
    if (i2 == n2 && (n & 1)) {
        long last = n - 1;
        O[last] = __hdiv(A[last], B[last]);
    }
}
"#;

// V2 kernels: dims/strides passed as scalar params (no GPU pointers, no htod copy).
// Max 8 dimensions. Unused dims padded to 1, unused strides to 0.
// SLOW broadcast path — kept as fallback for genuine broadcast/strided cases.
const CUDA_ADD_MUL_BF16: &str = r#"
#include <cuda_bf16.h>

// Macro: compute a_off and b_off from flat index using scalar dim/stride params.
// d0..d7 = output dims, as0..as7 = A strides, bs0..bs7 = B strides.
#define COMPUTE_OFFSETS(tid, nd, a_off, b_off) \
  long rem = (tid), a_off = 0, b_off = 0; \
  { long _d[8] = {d0,d1,d2,d3,d4,d5,d6,d7}; \
    long _as[8] = {as0,as1,as2,as3,as4,as5,as6,as7}; \
    long _bs[8] = {bs0,bs1,bs2,bs3,bs4,bs5,bs6,bs7}; \
    for (int i=(nd)-1; i>=0; --i) { \
      long idx = rem % _d[i]; rem /= _d[i]; \
      a_off += idx * _as[i]; \
      b_off += idx * _bs[i]; \
    } \
  }

#define KERNEL_PARAMS \
  long d0, long d1, long d2, long d3, long d4, long d5, long d6, long d7, \
  long as0, long as1, long as2, long as3, long as4, long as5, long as6, long as7, \
  long bs0, long bs1, long bs2, long bs3, long bs4, long bs5, long bs6, long bs7, \
  int nd, long total

extern "C" __global__
void add_bf16_kernel(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* O, KERNEL_PARAMS) {
  long tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < total) {
    COMPUTE_OFFSETS(tid, nd, a_off, b_off);
    O[tid] = __float2bfloat16(__bfloat162float(A[a_off]) + __bfloat162float(B[b_off]));
    tid += gridDim.x * blockDim.x;
  }
}
extern "C" __global__
void mul_bf16_kernel(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* O, KERNEL_PARAMS) {
  long tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < total) {
    COMPUTE_OFFSETS(tid, nd, a_off, b_off);
    O[tid] = __float2bfloat16(__bfloat162float(A[a_off]) * __bfloat162float(B[b_off]));
    tid += gridDim.x * blockDim.x;
  }
}
extern "C" __global__
void div_bf16_kernel(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* O, KERNEL_PARAMS) {
  long tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < total) {
    COMPUTE_OFFSETS(tid, nd, a_off, b_off);
    O[tid] = __float2bfloat16(__bfloat162float(A[a_off]) / __bfloat162float(B[b_off]));
    tid += gridDim.x * blockDim.x;
  }
}
extern "C" __global__
void max_bf16_kernel(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* O, KERNEL_PARAMS) {
  long tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < total) {
    COMPUTE_OFFSETS(tid, nd, a_off, b_off);
    float a = __bfloat162float(A[a_off]), b = __bfloat162float(B[b_off]);
    O[tid] = __float2bfloat16(a > b ? a : b);
    tid += gridDim.x * blockDim.x;
  }
}
extern "C" __global__
void min_bf16_kernel(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* O, KERNEL_PARAMS) {
  long tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < total) {
    COMPUTE_OFFSETS(tid, nd, a_off, b_off);
    float a = __bfloat162float(A[a_off]), b = __bfloat162float(B[b_off]);
    O[tid] = __float2bfloat16(a < b ? a : b);
    tid += gridDim.x * blockDim.x;
  }
}
"#;

const CUDA_TRANSPOSE2D_BF16: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void transpose2d_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                             __nv_bfloat16* __restrict__ output,
                             int rows,
                             int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < rows && col < cols) {
    int in_index = row * cols + col;
    int out_index = col * rows + row;
    output[out_index] = input[in_index];
  }
}
"#;

const CUDA_CMP_BF16: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void cmp_bf16_kernel(const __nv_bfloat16* A, const __nv_bfloat16* B, unsigned char* O,
                     long d0, long d1, long d2, long d3, long d4, long d5, long d6, long d7,
                     long as0, long as1, long as2, long as3, long as4, long as5, long as6, long as7,
                     long bs0, long bs1, long bs2, long bs3, long bs4, long bs5, long bs6, long bs7,
                     int nd, long total, int op) {
  // op: 0=ge,1=gt,2=le,3=lt,4=ne
  long tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < total) {
    long rem = tid, a_off = 0, b_off = 0;
    { long _d[8] = {d0,d1,d2,d3,d4,d5,d6,d7};
      long _as[8] = {as0,as1,as2,as3,as4,as5,as6,as7};
      long _bs[8] = {bs0,bs1,bs2,bs3,bs4,bs5,bs6,bs7};
      for (int i=nd-1; i>=0; --i) {
        long idx = rem % _d[i]; rem /= _d[i];
        a_off += idx * _as[i];
        b_off += idx * _bs[i];
      }
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
fn lc(n: usize) -> LaunchConfig {
    LaunchConfig::for_num_elems(n as u32)
}

/// Pad a BcSpec to exactly 8 dims for the v2 kernel scalar params.
/// Returns (d0..d7, as0..as7, bs0..bs7) all as i64.
fn pad8(spec: &BcSpec) -> ([i64; 8], [i64; 8], [i64; 8]) {
    let mut d = [1i64; 8];
    let mut a = [0i64; 8];
    let mut b = [0i64; 8];
    let nd = spec.out_dims.len();
    for i in 0..nd {
        d[i] = spec.out_dims[i];
        a[i] = spec.a_strides[i];
        b[i] = spec.b_strides[i];
    }
    (d, a, b)
}

fn ensure(dev: &Arc<CudaDevice>, nm: &'static str, code: &'static str) -> Result<()> {
    if dev.get_func(nm, nm).is_some() {
        return Ok(());
    }
    let include_dir = std::env::var("CUDA_INCLUDE_DIR")
        .or_else(|_| std::env::var("CUDA_HOME").map(|home| format!("{home}/include")))
        .unwrap_or_else(|_| "/usr/local/cuda/include".into());
    let mut opts = CompileOptions::default();
    opts.include_paths.push(include_dir);
    let ptx = compile_ptx_with_opts(code, opts)
        .map_err(|e| Error::Cuda(format!("nvrtc {}: {:?}", nm, e)))?;
    dev.load_ptx(ptx, nm, &[nm])
        .map_err(|e| Error::Cuda(format!("load {}: {:?}", nm, e)))?;
    Ok(())
}

/// Generic BF16 elementwise launch — no GPU metadata copies, all params are scalars.
fn launch_bf16_elementwise(
    a: &Tensor,
    b: &Tensor,
    kernel_name: &'static str,
    op_name: &str,
) -> Result<Tensor> {
    debug_assert_eq!(a.dtype(), DType::BF16);
    debug_assert_eq!(b.dtype(), DType::BF16);
    let spec = make_broadcast_spec(a.shape().dims(), b.shape().dims());
    let n = spec.total as usize;
    if spec.out_dims.len() > 8 {
        return Err(Error::InvalidInput(format!(
            "{}: ndim {} exceeds max 8 for scalar-param kernels",
            op_name, spec.out_dims.len()
        )));
    }

    let data = unsafe { a.device.alloc::<u16>(n) }
        .map_err(|e| Error::Cuda(format!("alloc {}: {:?}", op_name, e)))?;
    let out_dims_usize: Vec<usize> = spec.out_dims.iter().map(|&d| d as usize).collect();
    let mut out = Tensor {
        storage: TensorStorage::BF16 {
            data: data.into(),
            numel: n,
        },
        shape: Shape::from_dims(&out_dims_usize),
        device: a.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
    };

    ensure(&a.device, kernel_name, CUDA_ADD_MUL_BF16)?;
    let f = a
        .device
        .get_func(kernel_name, kernel_name)
        .ok_or_else(|| Error::Cuda(format!("missing kernel: {}", kernel_name)))?;

    let (d, ast, bst) = pad8(&spec);
    let a_ptr = a.as_device_ptr_bf16(&format!("{}:a", op_name))? as u64;
    let b_ptr = b.as_device_ptr_bf16(&format!("{}:b", op_name))? as u64;
    let o_ptr = out.as_mut_device_ptr_bf16(&format!("{}:out", op_name))? as u64;
    let nd = spec.out_dims.len() as i32;
    let total = spec.total;

    // Build params as Vec<*mut c_void> — cudarc supports this for arbitrary param counts.
    let mut params: Vec<*mut std::ffi::c_void> = Vec::with_capacity(29);
    params.push(&a_ptr as *const u64 as *mut std::ffi::c_void);
    params.push(&b_ptr as *const u64 as *mut std::ffi::c_void);
    params.push(&o_ptr as *const u64 as *mut std::ffi::c_void);
    for i in 0..8 { params.push(&d[i] as *const i64 as *mut std::ffi::c_void); }
    for i in 0..8 { params.push(&ast[i] as *const i64 as *mut std::ffi::c_void); }
    for i in 0..8 { params.push(&bst[i] as *const i64 as *mut std::ffi::c_void); }
    params.push(&nd as *const i32 as *mut std::ffi::c_void);
    params.push(&total as *const i64 as *mut std::ffi::c_void);

    unsafe {
        f.launch(lc(n), &mut params)
            .map_err(|e| Error::Training(format!("{e:?}")))?;
    }
    Ok(out)
}

// Fused BF16 softmax along the LAST dim. One block per row, threads cooperate
// via shared-mem max+sum reductions, then a single normalize pass writes BF16
// output. No FP32 scratch tensor allocated — replaces the 5-step
// max → sub → exp → sum → div pipeline that was allocating ~5× the tensor
// memory and running 175× slower than PyTorch on [30, 4096, 4096].
const CUDA_SOFTMAX_LASTDIM_BF16: &str = r#"
#include <cuda_bf16.h>

// NVRTC has no <float.h>, use the literal value of FLT_MAX directly.
#define LOCAL_FLT_MAX 3.402823466e+38f

extern "C" __global__
void softmax_lastdim_bf16_kernel(const __nv_bfloat16* __restrict__ X,
                                  __nv_bfloat16* __restrict__ Y,
                                  long rows, long cols) {
    extern __shared__ float smem[];

    long row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;

    const __nv_bfloat16* x_row = X + row * cols;
    __nv_bfloat16* y_row = Y + row * cols;

    // Pass 1: max (shared-mem reduction)
    float local_max = -LOCAL_FLT_MAX;
    for (long c = tid; c < cols; c += blockDim.x) {
        float v = __bfloat162float(x_row[c]);
        if (v > local_max) local_max = v;
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other = smem[tid + s];
            if (other > smem[tid]) smem[tid] = other;
        }
        __syncthreads();
    }
    float row_max = smem[0];

    // Pass 2: exp(x - max), sum (shared-mem reduction)
    float local_sum = 0.0f;
    for (long c = tid; c < cols; c += blockDim.x) {
        float v = __expf(__bfloat162float(x_row[c]) - row_max);
        local_sum += v;
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / smem[0];

    // Pass 3: write normalized output. Recompute exp to avoid an extra
    // shared-mem buffer for cols ≤ blockDim case. For cols > blockDim, this
    // is a memory-bound second read of x_row — same cost as PyTorch.
    for (long c = tid; c < cols; c += blockDim.x) {
        float v = __expf(__bfloat162float(x_row[c]) - row_max) * inv_sum;
        y_row[c] = __float2bfloat16(v);
    }
}
"#;

/// Fused BF16 softmax along the last dim. Replaces the 5-step pipeline
/// (max → sub → exp → sum → div) with one kernel and zero extra allocations.
pub fn softmax_lastdim_bf16(x: &Tensor) -> Result<Tensor> {
    if x.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "softmax_lastdim_bf16: input must be BF16".into(),
        ));
    }
    let dims = x.shape().dims();
    let cols = *dims.last().ok_or_else(|| {
        Error::InvalidInput("softmax_lastdim_bf16: empty shape".into())
    })?;
    let rows = x.shape().elem_count() / cols;

    let n = x.shape().elem_count();
    let data = unsafe { x.device.alloc::<u16>(n) }
        .map_err(|e| Error::Cuda(format!("alloc softmax: {:?}", e)))?;
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

    ensure(&x.device, "softmax_lastdim_bf16_kernel", CUDA_SOFTMAX_LASTDIM_BF16)?;
    let f = x
        .device
        .get_func("softmax_lastdim_bf16_kernel", "softmax_lastdim_bf16_kernel")
        .ok_or_else(|| Error::Cuda("missing kernel: softmax_lastdim_bf16_kernel".into()))?;

    // Pick a power-of-two block size up to 1024.
    let mut block_size = 1usize;
    while block_size * 2 <= cols && block_size * 2 <= 1024 {
        block_size *= 2;
    }
    if block_size < 32 { block_size = 32; }

    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: (block_size * std::mem::size_of::<f32>()) as u32,
    };

    let x_ptr = x.as_device_ptr_bf16("softmax_lastdim_bf16:x")? as u64;
    let o_ptr = out.as_mut_device_ptr_bf16("softmax_lastdim_bf16:out")? as u64;
    let rows_i = rows as i64;
    let cols_i = cols as i64;

    let mut params: Vec<*mut std::ffi::c_void> = Vec::with_capacity(4);
    params.push(&x_ptr as *const u64 as *mut std::ffi::c_void);
    params.push(&o_ptr as *const u64 as *mut std::ffi::c_void);
    params.push(&rows_i as *const i64 as *mut std::ffi::c_void);
    params.push(&cols_i as *const i64 as *mut std::ffi::c_void);

    unsafe {
        f.launch(cfg, &mut params)
            .map_err(|e| Error::Cuda(format!("softmax_lastdim_bf16 launch: {:?}", e)))?;
    }
    Ok(out)
}

pub fn add_bf16(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if shapes_equal_no_broadcast(a, b) {
        return launch_bf16_flat(a, b, "add_bf16_flat_kernel", "add_bf16");
    }
    launch_bf16_elementwise(a, b, "add_bf16_kernel", "add_bf16")
}

pub fn sub_bf16(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if shapes_equal_no_broadcast(a, b) {
        return launch_bf16_flat(a, b, "sub_bf16_flat_kernel", "sub_bf16");
    }
    // Fallback: a - b = a + (-1 * b). The broadcast slow path doesn't have a sub kernel.
    let neg_b = b.mul_scalar(-1.0)?;
    launch_bf16_elementwise(a, &neg_b, "add_bf16_kernel", "sub_bf16")
}

#[inline]
fn shapes_equal_no_broadcast(a: &Tensor, b: &Tensor) -> bool {
    a.dtype() == DType::BF16
        && b.dtype() == DType::BF16
        && a.shape().dims() == b.shape().dims()
        && a.shape().elem_count() == b.shape().elem_count()
}

/// Fast path: both inputs are contiguous, same shape, no broadcast.
/// Uses __nv_bfloat162 vectorization (2 elements per thread) and native BF16
/// add/mul/sub/div via __hadd2/__hmul2/__hsub2/__h2div. No FP32 round-trip.
fn launch_bf16_flat(
    a: &Tensor,
    b: &Tensor,
    kernel_name: &'static str,
    op_name: &str,
) -> Result<Tensor> {
    debug_assert_eq!(a.dtype(), DType::BF16);
    debug_assert_eq!(b.dtype(), DType::BF16);
    let n = a.shape().elem_count();

    let data = unsafe { a.device.alloc::<u16>(n) }
        .map_err(|e| Error::Cuda(format!("alloc {}: {:?}", op_name, e)))?;
    let mut out = Tensor {
        storage: TensorStorage::BF16 {
            data: data.into(),
            numel: n,
        },
        shape: a.shape().clone(),
        device: a.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
    };

    ensure(&a.device, kernel_name, CUDA_ADD_MUL_BF16_FLAT)?;
    let f = a
        .device
        .get_func(kernel_name, kernel_name)
        .ok_or_else(|| Error::Cuda(format!("missing kernel: {}", kernel_name)))?;

    // Launch with 2 elements per thread.
    let n_pairs = (n + 1) / 2;
    let block = 256u32;
    let grid = ((n_pairs as u32) + block - 1) / block;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };

    let a_ptr = a.as_device_ptr_bf16(&format!("{}:a", op_name))? as u64;
    let b_ptr = b.as_device_ptr_bf16(&format!("{}:b", op_name))? as u64;
    let o_ptr = out.as_mut_device_ptr_bf16(&format!("{}:out", op_name))? as u64;
    let n_i64 = n as i64;

    let mut params: Vec<*mut std::ffi::c_void> = Vec::with_capacity(4);
    params.push(&a_ptr as *const u64 as *mut std::ffi::c_void);
    params.push(&b_ptr as *const u64 as *mut std::ffi::c_void);
    params.push(&o_ptr as *const u64 as *mut std::ffi::c_void);
    params.push(&n_i64 as *const i64 as *mut std::ffi::c_void);

    unsafe {
        f.launch(cfg, &mut params)
            .map_err(|e| Error::Training(format!("{e:?}")))?;
    }
    Ok(out)
}

pub fn transpose2d_bf16(tensor: &Tensor) -> Result<Tensor> {
    if tensor.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "transpose2d_bf16 expects BF16 tensor".into(),
        ));
    }
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(Error::InvalidInput(format!(
            "transpose2d_bf16 expects 2D tensor, got {:?}",
            dims
        )));
    }
    let rows = dims[0];
    let cols = dims[1];

    ensure(
        &tensor.device,
        "transpose2d_bf16_kernel",
        CUDA_TRANSPOSE2D_BF16,
    )?;
    let f = tensor
        .device
        .get_func("transpose2d_bf16_kernel", "transpose2d_bf16_kernel")
        .ok_or_else(|| Error::Cuda("missing kernel: transpose2d_bf16_kernel".into()))?;

    let mut out = Tensor::zeros_dtype(
        Shape::from_dims(&[cols, rows]),
        DType::BF16,
        tensor.device.clone(),
    )?;

    let block_dim = (16u32, 16u32, 1u32);
    let grid_dim = (
        ((cols as u32) + block_dim.0 - 1) / block_dim.0,
        ((rows as u32) + block_dim.1 - 1) / block_dim.1,
        1u32,
    );

    let cfg = LaunchConfig {
        grid_dim,
        block_dim,
        shared_mem_bytes: 0,
    };

    let input_slice = tensor.as_device_ptr_bf16("transpose2d_bf16:input")? as *const u16;
    let output_slice = out.as_mut_device_ptr_bf16("transpose2d_bf16:output")? as *mut u16;

    unsafe {
        f.launch(
            cfg,
            (
                input_slice as u64,
                output_slice as u64,
                rows as i32,
                cols as i32,
            ),
        )
        .map_err(|e| Error::Cuda(format!("launch transpose2d_bf16: {:?}", e)))?;
    }

    Ok(out)
}

pub fn mul_bf16(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if shapes_equal_no_broadcast(a, b) {
        return launch_bf16_flat(a, b, "mul_bf16_flat_kernel", "mul_bf16");
    }
    launch_bf16_elementwise(a, b, "mul_bf16_kernel", "mul_bf16")
}

pub fn div_bf16(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if shapes_equal_no_broadcast(a, b) {
        return launch_bf16_flat(a, b, "div_bf16_flat_kernel", "div_bf16");
    }
    launch_bf16_elementwise(a, b, "div_bf16_kernel", "div_bf16")
}

pub fn max_bf16(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    launch_bf16_elementwise(a, b, "max_bf16_kernel", "max_bf16")
}

pub fn min_bf16(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    launch_bf16_elementwise(a, b, "min_bf16_kernel", "min_bf16")
}

fn cmp_bf16(a: &Tensor, b: &Tensor, op: i32) -> Result<Tensor> {
    debug_assert_eq!(a.dtype(), DType::BF16);
    debug_assert_eq!(b.dtype(), DType::BF16);
    let spec = make_broadcast_spec(a.shape().dims(), b.shape().dims());
    let n = spec.total as usize;
    if spec.out_dims.len() > 8 {
        return Err(Error::InvalidInput(format!(
            "cmp_bf16: ndim {} exceeds max 8", spec.out_dims.len()
        )));
    }

    let data = unsafe { a.device.alloc::<f32>(n) }
        .map_err(|e| Error::Cuda(format!("alloc cmp bf16: {:?}", e)))?;
    let out_dims_usize: Vec<usize> = spec.out_dims.iter().map(|&d| d as usize).collect();
    let mut out = Tensor {
        storage: TensorStorage::Bool {
            data: data.into(),
            numel: n,
        },
        shape: Shape::from_dims(&out_dims_usize),
        device: a.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
    };

    ensure(&a.device, "cmp_bf16_kernel", CUDA_CMP_BF16)?;
    let f = a
        .device
        .get_func("cmp_bf16_kernel", "cmp_bf16_kernel")
        .ok_or_else(|| Error::Cuda("missing kernel: cmp_bf16_kernel".into()))?;

    let (d, ast, bst) = pad8(&spec);
    let a_ptr = a.as_device_ptr_bf16("cmp_bf16:a")? as u64;
    let b_ptr = b.as_device_ptr_bf16("cmp_bf16:b")? as u64;
    let o_ptr = match &mut out.storage {
        TensorStorage::Bool { data, .. } => *data.device_ptr() as u64,
        _ => {
            return Err(Error::InvalidOperation(
                "expected Bool storage for out".into(),
            ))
        }
    };
    let nd = spec.out_dims.len() as i32;
    let total = spec.total;

    let mut params: Vec<*mut std::ffi::c_void> = Vec::with_capacity(30);
    params.push(&a_ptr as *const u64 as *mut std::ffi::c_void);
    params.push(&b_ptr as *const u64 as *mut std::ffi::c_void);
    params.push(&o_ptr as *const u64 as *mut std::ffi::c_void);
    for i in 0..8 { params.push(&d[i] as *const i64 as *mut std::ffi::c_void); }
    for i in 0..8 { params.push(&ast[i] as *const i64 as *mut std::ffi::c_void); }
    for i in 0..8 { params.push(&bst[i] as *const i64 as *mut std::ffi::c_void); }
    params.push(&nd as *const i32 as *mut std::ffi::c_void);
    params.push(&total as *const i64 as *mut std::ffi::c_void);
    params.push(&op as *const i32 as *mut std::ffi::c_void);

    unsafe {
        f.launch(lc(n), &mut params)
            .map_err(|e| Error::Training(format!("{e:?}")))?;
    }
    Ok(out)
}

// Public compare helpers
pub fn ge_bf16(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    cmp_bf16(a, b, 0)
}
pub fn gt_bf16(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    cmp_bf16(a, b, 1)
}
pub fn le_bf16(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    cmp_bf16(a, b, 2)
}
pub fn lt_bf16(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    cmp_bf16(a, b, 3)
}
pub fn ne_bf16(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    cmp_bf16(a, b, 4)
}

// ---------------------------------------------------------------------------
// Fused patchify / unpatchify — BF16, no 6D permute, no F32 round-trip
// ---------------------------------------------------------------------------

const CUDA_PATCHIFY_BF16: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void patchify_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,   // [B, C, H, W]
    __nv_bfloat16* __restrict__ output,         // [B, N, patch_dim]
    int B, int C, int H, int W,
    int ph, int pw, int p,                      // patch grid and size
    int N, int patch_dim                        // N=ph*pw, patch_dim=p*p*C
) {
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)B * N * patch_dim;
    while (tid < total) {
        // Decompose output index: (b, n, d)
        int d = (int)(tid % patch_dim);
        long bn = tid / patch_dim;
        int n = (int)(bn % N);
        int b = (int)(bn / N);

        // n -> (h_patch, w_patch)
        int h_patch = n / pw;
        int w_patch = n % pw;

        // d -> (p_h, p_w, c)  — layout matches permute(0,2,4,3,5,1)
        int c = d % C;
        int p_w = (d / C) % p;
        int p_h = d / (p * C);

        // Source pixel
        int h = h_patch * p + p_h;
        int w = w_patch * p + p_w;
        long src = (long)b * C * H * W + (long)c * H * W + (long)h * W + w;

        output[tid] = input[src];
        tid += (long)gridDim.x * blockDim.x;
    }
}
"#;

const CUDA_UNPATCHIFY_BF16: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void unpatchify_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,   // [B, N, patch_dim]
    __nv_bfloat16* __restrict__ output,         // [B, C, H, W]
    int B, int C, int H, int W,
    int ph, int pw, int p,
    int N, int patch_dim
) {
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)B * C * H * W;
    while (tid < total) {
        // Decompose output index: (b, c, h, w)
        int w = (int)(tid % W);
        long tmp = tid / W;
        int h = (int)(tmp % H);
        tmp /= H;
        int c = (int)(tmp % C);
        int b = (int)(tmp / C);

        // (h, w) -> patch coords
        int h_patch = h / p;
        int w_patch = w / p;
        int p_h = h % p;
        int p_w = w % p;

        // Source index in [B, N, patch_dim]
        int n = h_patch * pw + w_patch;
        int d = p_h * (p * C) + p_w * C + c;
        long src = (long)b * N * patch_dim + (long)n * patch_dim + d;

        output[tid] = input[src];
        tid += (long)gridDim.x * blockDim.x;
    }
}
"#;

/// Fused patchify: [B, C, H, W] → [B, ph*pw, p*p*C] in BF16, no 6D permute.
///
/// Returns (patches, ph, pw).
pub fn patchify_bf16(
    x: &Tensor,
    patch_size: usize,
) -> Result<(Tensor, usize, usize)> {
    assert_eq!(x.dtype(), DType::BF16, "patchify_bf16: expected BF16");
    let dims = x.shape().dims();
    assert_eq!(dims.len(), 4, "patchify_bf16: expected 4D [B,C,H,W]");
    let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let p = patch_size;
    let ph = h / p;
    let pw = w / p;
    let n = ph * pw;
    let patch_dim = p * p * c;

    ensure(&x.device, "patchify_bf16_kernel", CUDA_PATCHIFY_BF16)?;
    let f = x
        .device
        .get_func("patchify_bf16_kernel", "patchify_bf16_kernel")
        .ok_or_else(|| Error::Cuda("missing kernel: patchify_bf16_kernel".into()))?;

    let mut out = Tensor::zeros_dtype(
        Shape::from_dims(&[b, n, patch_dim]),
        DType::BF16,
        x.device.clone(),
    )?;

    let total = b * n * patch_dim;
    let cfg = lc(total);

    let input_ptr = x.as_device_ptr_bf16("patchify:input")? as *const u16;
    let output_ptr = out.as_mut_device_ptr_bf16("patchify:output")? as *mut u16;

    unsafe {
        f.launch(
            cfg,
            (
                input_ptr as u64,
                output_ptr as u64,
                b as i32,
                c as i32,
                h as i32,
                w as i32,
                ph as i32,
                pw as i32,
                p as i32,
                n as i32,
                patch_dim as i32,
            ),
        )
        .map_err(|e| Error::Cuda(format!("launch patchify_bf16: {:?}", e)))?;
    }

    Ok((out, ph, pw))
}

/// Fused unpatchify: [B, N, patch_dim] → [B, C, H, W] in BF16, no 6D permute.
pub fn unpatchify_bf16(
    x: &Tensor,
    ph: usize,
    pw: usize,
    patch_size: usize,
    in_channels: usize,
) -> Result<Tensor> {
    assert_eq!(x.dtype(), DType::BF16, "unpatchify_bf16: expected BF16");
    let dims = x.shape().dims();
    assert_eq!(dims.len(), 3, "unpatchify_bf16: expected 3D [B,N,P]");
    let b = dims[0];
    let c = in_channels;
    let p = patch_size;
    let h = ph * p;
    let w = pw * p;
    let n = ph * pw;
    let patch_dim = p * p * c;

    ensure(&x.device, "unpatchify_bf16_kernel", CUDA_UNPATCHIFY_BF16)?;
    let f = x
        .device
        .get_func("unpatchify_bf16_kernel", "unpatchify_bf16_kernel")
        .ok_or_else(|| Error::Cuda("missing kernel: unpatchify_bf16_kernel".into()))?;

    let mut out = Tensor::zeros_dtype(
        Shape::from_dims(&[b, c, h, w]),
        DType::BF16,
        x.device.clone(),
    )?;

    let total = b * c * h * w;
    let cfg = lc(total);

    let input_ptr = x.as_device_ptr_bf16("unpatchify:input")? as *const u16;
    let output_ptr = out.as_mut_device_ptr_bf16("unpatchify:output")? as *mut u16;

    unsafe {
        f.launch(
            cfg,
            (
                input_ptr as u64,
                output_ptr as u64,
                b as i32,
                c as i32,
                h as i32,
                w as i32,
                ph as i32,
                pw as i32,
                p as i32,
                n as i32,
                patch_dim as i32,
            ),
        )
        .map_err(|e| Error::Cuda(format!("launch unpatchify_bf16: {:?}", e)))?;
    }

    Ok(out)
}
