use std::sync::Arc;
// Legacy bf16 ops retained for backwards compatibility.
use crate::dtype::DType;
use crate::tensor_storage::{ensure_unique_slice, slice_ref, TensorStorage};
use crate::{Error, Result, Shape, Tensor, TensorId};
use cudarc::{
    driver::{CudaDevice, LaunchAsync, LaunchConfig},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};

// Vectorized BF16 unary kernels: 2 elements per thread via __nv_bfloat162.
// The previous scalar versions were running ~3.6 ms on [1,4096,15360] = 13×
// slower than PyTorch (0.29 ms). The vectorized versions should hit ~0.4 ms,
// matching PyTorch within 1.5×.
//
// Each kernel is launched with `(n+1)/2` threads (see `silu_bf16` /
// `gelu_bf16` below). The `if (i2 < n2)` guard handles the typical case;
// the `(n & 1)` tail handles odd `n`.
const CUDA_GELU: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void gelu_bf16_kernel(const __nv_bfloat16* __restrict__ X,
                      __nv_bfloat16* __restrict__ Y,
                      long n) {
  long i2 = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long n2 = n >> 1;
  if (i2 < n2) {
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(X);
    __nv_bfloat162* y2 = reinterpret_cast<__nv_bfloat162*>(Y);
    float2 v = __bfloat1622float2(x2[i2]);
    float c0 = 0.7978845608f * (v.x + 0.044715f * v.x * v.x * v.x);
    float c1 = 0.7978845608f * (v.y + 0.044715f * v.y * v.y * v.y);
    float g0 = 0.5f * v.x * (1.0f + tanhf(c0));
    float g1 = 0.5f * v.y * (1.0f + tanhf(c1));
    y2[i2] = __floats2bfloat162_rn(g0, g1);
  }
  if (i2 == n2 && (n & 1)) {
    long last = n - 1;
    float x = __bfloat162float(X[last]);
    float c = 0.7978845608f * (x + 0.044715f * x * x * x);
    Y[last] = __float2bfloat16(0.5f * x * (1.0f + tanhf(c)));
  }
}
"#;

const CUDA_SILU: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void silu_bf16_kernel(const __nv_bfloat16* __restrict__ X,
                      __nv_bfloat16* __restrict__ Y,
                      long n) {
  long i2 = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long n2 = n >> 1;
  if (i2 < n2) {
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(X);
    __nv_bfloat162* y2 = reinterpret_cast<__nv_bfloat162*>(Y);
    float2 v = __bfloat1622float2(x2[i2]);
    float s0 = v.x / (1.0f + __expf(-v.x));
    float s1 = v.y / (1.0f + __expf(-v.y));
    y2[i2] = __floats2bfloat162_rn(s0, s1);
  }
  if (i2 == n2 && (n & 1)) {
    long last = n - 1;
    float x = __bfloat162float(X[last]);
    Y[last] = __float2bfloat16(x / (1.0f + expf(-x)));
  }
}
"#;

const CUDA_SQUARE: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void square_bf16_kernel(const __nv_bfloat16* __restrict__ X,
                        __nv_bfloat16* __restrict__ Y,
                        long n) {
  long i2 = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long n2 = n >> 1;
  if (i2 < n2) {
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(X);
    __nv_bfloat162* y2 = reinterpret_cast<__nv_bfloat162*>(Y);
    float2 v = __bfloat1622float2(x2[i2]);
    y2[i2] = __floats2bfloat162_rn(v.x * v.x, v.y * v.y);
  }
  if (i2 == n2 && (n & 1)) {
    long last = n - 1;
    float x = __bfloat162float(X[last]);
    Y[last] = __float2bfloat16(x * x);
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
        .map_err(|e| Error::Cuda(format!("nvrtc {}: {:?}", name, e)))?;
    dev.load_ptx(ptx, name, &[name])
        .map_err(|e| Error::Cuda(format!("load_ptx {}: {:?}", name, e)))?;
    Ok(())
}

#[inline]
fn lc(n: usize) -> LaunchConfig {
    LaunchConfig::for_num_elems(n as u32)
}

/// Launch config for vectorized 2-element-per-thread kernels.
#[inline]
fn lc_pairs(n: usize) -> LaunchConfig {
    let pairs = (n + 1) / 2;
    LaunchConfig::for_num_elems(pairs as u32)
}

pub fn gelu_bf16(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), DType::BF16);
    let n = x.shape().elem_count();
    let data = crate::cuda_alloc_pool::pool_alloc_u16(&x.device, n)?;
    let mut out = Tensor {
        storage: TensorStorage::BF16 {
            data: data.into(),
            numel: n,
        },
        shape: x.shape().clone(),
        device: x.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,

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
        f.launch(lc_pairs(n), (slice_ref(xs), ys, n as i64))?;
    }
    Ok(out)
}

pub fn square_bf16(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), DType::BF16);
    let n = x.shape().elem_count();
    let data = crate::cuda_alloc_pool::pool_alloc_u16(&x.device, n)?;
    let mut out = Tensor {
        storage: TensorStorage::BF16 {
            data: data.into(),
            numel: n,
        },
        shape: x.shape().clone(),
        device: x.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,

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
        f.launch(lc_pairs(n), (slice_ref(xs), ys, n as i64))?;
    }
    Ok(out)
}

// ---- Fused softmax over last dimension (single kernel, no temporaries) ----

const CUDA_SOFTMAX_LAST_DIM: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void softmax_last_dim_bf16_kernel(const __nv_bfloat16* __restrict__ X,
                                  __nv_bfloat16* __restrict__ Y,
                                  int rows, int cols) {
    // One block per row. Shared memory used only for reductions (blockDim floats).
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16* src = X + (long)row * cols;
    __nv_bfloat16* dst       = Y + (long)row * cols;

    // 1) Find row max
    float local_max = -1e30f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(src[c]);
        if (v > local_max) local_max = v;
    }
    smem[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && smem[threadIdx.x + s] > smem[threadIdx.x])
            smem[threadIdx.x] = smem[threadIdx.x + s];
        __syncthreads();
    }
    float row_max = smem[0];
    __syncthreads();

    // 2) Compute sum of exp(x - max)
    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        local_sum += expf(__bfloat162float(src[c]) - row_max);
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / smem[0];
    __syncthreads();

    // 3) Write exp(x - max) / sum — recompute exp (3 reads total, no extra smem)
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = expf(__bfloat162float(src[c]) - row_max);
        dst[c] = __float2bfloat16(v * inv_sum);
    }
}
"#;

/// Fused softmax over the last dimension for BF16 tensors.
/// Single kernel launch instead of 5 (max, sub, exp, sum, div).
/// Input can be any shape — operates on the innermost dimension.
pub fn softmax_last_dim_bf16(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), DType::BF16);
    let dims = x.shape().dims();
    let cols = *dims.last().ok_or_else(|| Error::InvalidOperation("empty shape".into()))?;
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let n = rows * cols;

    let data = crate::cuda_alloc_pool::pool_alloc_u16(&x.device, n)?;
    let mut out = Tensor {
        storage: TensorStorage::BF16 {
            data: data.into(),
            numel: n,
        },
        shape: x.shape().clone(),
        device: x.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,

    };
    ensure(
        &x.device,
        "softmax_last_dim_bf16_kernel",
        CUDA_SOFTMAX_LAST_DIM,
    )?;
    let f = x
        .device
        .get_func(
            "softmax_last_dim_bf16_kernel",
            "softmax_last_dim_bf16_kernel",
        )
        .ok_or_else(|| Error::Cuda("softmax_last_dim_bf16_kernel missing".into()))?;
    let xs = match &x.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => return Err(Error::InvalidOperation("softmax_last_dim_bf16 expects BF16".into())),
    };
    let ys = match &mut out.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => unreachable!(),
    };
    let ys = ensure_unique_slice(ys)?;

    // One block per row, up to 256 threads
    let threads = cols.min(256) as u32;
    // Shared memory: blockDim floats for reduction
    let smem = threads as u32 * 4;
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem,
    };
    unsafe {
        f.launch(cfg, (slice_ref(xs), ys, rows as i32, cols as i32))?;
    }
    Ok(out)
}

pub fn silu_bf16(x: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), DType::BF16);
    let n = x.shape().elem_count();
    let data = crate::cuda_alloc_pool::pool_alloc_u16(&x.device, n)?;
    let mut out = Tensor {
        storage: TensorStorage::BF16 {
            data: data.into(),
            numel: n,
        },
        shape: x.shape().clone(),
        device: x.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,

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
        f.launch(lc_pairs(n), (slice_ref(xs), ys, n as i64))?;
    }
    Ok(out)
}

// ---- Fused RoPE: complex rotation in one kernel, no intermediates ----

const CUDA_ROPE_FUSED: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void rope_fused_bf16_kernel(
    const __nv_bfloat16* __restrict__ X,   // [BH, N, D]  (D = 2*half)
    const __nv_bfloat16* __restrict__ COS,  // [1, N, half]
    const __nv_bfloat16* __restrict__ SIN,  // [1, N, half]
    __nv_bfloat16* __restrict__ Y,          // [BH, N, D]
    long bh, long n, long half)
{
    // Interleaved (complex) RoPE: pairs adjacent elements (2d, 2d+1).
    // Used by Klein/Flux, LTX, HunyuanVideo.
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long total = bh * n * half;
    while (idx < total) {
        long tmp = idx;
        long d = tmp % half; tmp /= half;
        long seq = tmp % n;   tmp /= n;
        long b = tmp;

        long base = (b * n + seq) * (2 * half);
        long cos_idx = seq * half + d;

        float x_even = __bfloat162float(X[base + 2*d]);
        float x_odd  = __bfloat162float(X[base + 2*d + 1]);
        float c = __bfloat162float(COS[cos_idx]);
        float s = __bfloat162float(SIN[cos_idx]);

        Y[base + 2*d]     = __float2bfloat16(x_even * c - x_odd * s);
        Y[base + 2*d + 1] = __float2bfloat16(x_even * s + x_odd * c);

        idx += gridDim.x * blockDim.x;
    }
}

// FP32-COS/SIN variant: same interleaved RoPE math, but cos/sin tables are
// F32. Removes the ~4e-3 BF16 quantization floor on pe_cos/pe_sin. Used for
// Klein/Flux where the 1140 RoPE applications per inference accumulate that
// floor error into visible quality degradation. Table size is tiny (~1 MiB
// for FLUX at 1024²) so storing PE as F32 is free.
extern "C" __global__
void rope_fused_bf16_f32pe_kernel(
    const __nv_bfloat16* __restrict__ X,
    const float*         __restrict__ COS,  // [1, N, half] F32
    const float*         __restrict__ SIN,  // [1, N, half] F32
    __nv_bfloat16*       __restrict__ Y,
    long bh, long n, long half)
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long total = bh * n * half;
    while (idx < total) {
        long tmp = idx;
        long d = tmp % half; tmp /= half;
        long seq = tmp % n;   tmp /= n;
        long b = tmp;

        long base = (b * n + seq) * (2 * half);
        long cos_idx = seq * half + d;

        float x_even = __bfloat162float(X[base + 2*d]);
        float x_odd  = __bfloat162float(X[base + 2*d + 1]);
        float c = COS[cos_idx];
        float s = SIN[cos_idx];

        Y[base + 2*d]     = __float2bfloat16(x_even * c - x_odd * s);
        Y[base + 2*d + 1] = __float2bfloat16(x_even * s + x_odd * c);

        idx += gridDim.x * blockDim.x;
    }
}

extern "C" __global__
void rope_halfsplit_bf16_kernel(
    const __nv_bfloat16* __restrict__ X,   // [BH, N, D]  (D = 2*half)
    const __nv_bfloat16* __restrict__ COS,  // [cos_bh, N, half] — 1 or BH
    const __nv_bfloat16* __restrict__ SIN,  // [cos_bh, N, half]
    __nv_bfloat16* __restrict__ Y,          // [BH, N, D]
    long bh, long n, long half, long cos_bh)
{
    // Half-split RoPE (HuggingFace rotate_half convention):
    // Pairs elements across halves: (d, d+half).
    // cos_bh=1: broadcast same frequencies to all heads.
    // cos_bh=BH: per-head frequencies (LTX-2 style).
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long total = bh * n * half;
    while (idx < total) {
        long tmp = idx;
        long d = tmp % half; tmp /= half;
        long seq = tmp % n;   tmp /= n;
        long b = tmp;

        long base = (b * n + seq) * (2 * half);
        long cb = (cos_bh == 1) ? 0 : b;
        long cos_idx = (cb * n + seq) * half + d;

        float x_first  = __bfloat162float(X[base + d]);
        float x_second = __bfloat162float(X[base + half + d]);
        float c = __bfloat162float(COS[cos_idx]);
        float s = __bfloat162float(SIN[cos_idx]);

        Y[base + d]        = __float2bfloat16(x_first * c - x_second * s);
        Y[base + half + d] = __float2bfloat16(x_second * c + x_first * s);

        idx += gridDim.x * blockDim.x;
    }
}
"#;

/// Fused RoPE application — single kernel, no intermediate tensors.
///
/// `x`: [B, H, N, D] BF16 tensor (D must be even).
/// `cos`: [1, 1, N, D/2] BF16 cos table.
/// `sin`: [1, 1, N, D/2] BF16 sin table.
///
/// Returns [B, H, N, D] BF16 with rotation applied.
pub fn rope_fused_bf16(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), DType::BF16);
    let x_dims = x.shape().dims();
    if x_dims.len() != 4 {
        return Err(Error::InvalidOperation(format!(
            "rope_fused_bf16: expected 4D [B,H,N,D], got {:?}",
            x_dims
        )));
    }
    let (b, h, n, d) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
    let half = d / 2;
    let bh = b * h;

    // Reshape x to [BH, N, D] for the kernel
    let x_flat = x.reshape(&[bh, n, d])?;

    // cos/sin are [1, 1, N, half] — reshape to [1, N, half]
    let cos_flat = cos.reshape(&[1, n, half])?;
    let sin_flat = sin.reshape(&[1, n, half])?;

    let total = bh * n * half;
    let out_n = bh * n * d;

    let data = crate::cuda_alloc_pool::pool_alloc_u16(&x.device, out_n)?;
    let mut out = Tensor {
        storage: TensorStorage::BF16 {
            data: data.into(),
            numel: out_n,
        },
        shape: Shape::from_dims(&[bh, n, d]),
        device: x.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,

    };

    ensure(&x.device, "rope_fused_bf16_kernel", CUDA_ROPE_FUSED)?;
    let f = x
        .device
        .get_func("rope_fused_bf16_kernel", "rope_fused_bf16_kernel")
        .ok_or_else(|| Error::Cuda("rope_fused_bf16_kernel missing".into()))?;

    let xs = match &x_flat.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => return Err(Error::InvalidOperation("rope expects BF16".into())),
    };
    let cs = match &cos_flat.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => return Err(Error::InvalidOperation("rope cos expects BF16".into())),
    };
    let ss = match &sin_flat.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => return Err(Error::InvalidOperation("rope sin expects BF16".into())),
    };
    let ys = match &mut out.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => unreachable!(),
    };
    let ys = ensure_unique_slice(ys)?;

    unsafe {
        f.launch(
            lc(total),
            (
                slice_ref(xs),
                slice_ref(cs),
                slice_ref(ss),
                ys,
                bh as i64,
                n as i64,
                half as i64,
            ),
        )?;
    }

    out.reshape(&[b, h, n, d])
}

/// Interleaved-pairs RoPE with F32 cos/sin tables.
///
/// Identical math to `rope_fused_bf16` but keeps the position-embedding tables
/// in FP32, eliminating the ~4e-3 BF16 quantization floor on cos/sin that
/// accumulates across FLUX's 1140+ RoPE applications per inference.
///
/// `x`:   [B, H, N, D] BF16 (D even).
/// `cos`: [1, 1, N, D/2] **F32** cos table.
/// `sin`: [1, 1, N, D/2] **F32** sin table.
///
/// Returns [B, H, N, D] BF16.
pub fn rope_fused_bf16_f32pe(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), DType::BF16);
    if cos.dtype() != DType::F32 || sin.dtype() != DType::F32 {
        return Err(Error::InvalidOperation(format!(
            "rope_fused_bf16_f32pe: cos/sin must be F32, got cos={:?} sin={:?}",
            cos.dtype(),
            sin.dtype()
        )));
    }
    let x_dims = x.shape().dims();
    if x_dims.len() != 4 {
        return Err(Error::InvalidOperation(format!(
            "rope_fused_bf16_f32pe: expected 4D [B,H,N,D], got {:?}",
            x_dims
        )));
    }
    let (b, h, n, d) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
    let half = d / 2;
    let bh = b * h;

    let x_flat = x.reshape(&[bh, n, d])?;
    let cos_flat = cos.reshape(&[1, n, half])?;
    let sin_flat = sin.reshape(&[1, n, half])?;

    let total = bh * n * half;
    let out_n = bh * n * d;

    let data = crate::cuda_alloc_pool::pool_alloc_u16(&x.device, out_n)?;
    let mut out = Tensor {
        storage: TensorStorage::BF16 {
            data: data.into(),
            numel: out_n,
        },
        shape: Shape::from_dims(&[bh, n, d]),
        device: x.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,

    };

    ensure(&x.device, "rope_fused_bf16_f32pe_kernel", CUDA_ROPE_FUSED)?;
    let f = x
        .device
        .get_func("rope_fused_bf16_f32pe_kernel", "rope_fused_bf16_f32pe_kernel")
        .ok_or_else(|| Error::Cuda("rope_fused_bf16_f32pe_kernel missing".into()))?;

    let xs = match &x_flat.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => return Err(Error::InvalidOperation("rope expects BF16".into())),
    };
    let cs = match &cos_flat.storage {
        TensorStorage::F32 { data, .. } => data,
        _ => return Err(Error::InvalidOperation("rope_fused_bf16_f32pe cos expects F32".into())),
    };
    let ss = match &sin_flat.storage {
        TensorStorage::F32 { data, .. } => data,
        _ => return Err(Error::InvalidOperation("rope_fused_bf16_f32pe sin expects F32".into())),
    };
    let ys = match &mut out.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => unreachable!(),
    };
    let ys = ensure_unique_slice(ys)?;

    unsafe {
        f.launch(
            lc(total),
            (
                slice_ref(xs),
                slice_ref(cs),
                slice_ref(ss),
                ys,
                bh as i64,
                n as i64,
                half as i64,
            ),
        )?;
    }

    out.reshape(&[b, h, n, d])
}

/// Half-split RoPE (HuggingFace rotate_half convention).
///
/// Same interface as `rope_fused_bf16` but pairs elements across halves:
/// `(d, d+half)` instead of adjacent `(2d, 2d+1)`.
/// Used by Qwen3, LLaMA, Mistral.
pub fn rope_halfsplit_bf16(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), DType::BF16);
    let x_dims = x.shape().dims();
    if x_dims.len() != 4 {
        return Err(Error::InvalidOperation(format!(
            "rope_halfsplit_bf16: expected 4D [B,H,N,D], got {:?}",
            x_dims
        )));
    }
    let (b, h, n, d) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
    let half = d / 2;
    let bh = b * h;

    let x_flat = x.reshape(&[bh, n, d])?;
    // cos/sin can be [1, 1, N, half] (broadcast) or [B, H, N, half] (per-head).
    // Flatten to [cos_bh, N, half] where cos_bh is 1 or BH.
    let cos_elem = cos.shape().elem_count();
    let cos_bh = cos_elem / (n * half);
    let cos_flat = cos.reshape(&[cos_bh, n, half])?;
    let sin_flat = sin.reshape(&[cos_bh, n, half])?;

    let total = bh * n * half;
    let out_n = bh * n * d;

    let data = crate::cuda_alloc_pool::pool_alloc_u16(&x.device, out_n)?;
    let mut out = Tensor {
        storage: TensorStorage::BF16 {
            data: data.into(),
            numel: out_n,
        },
        shape: Shape::from_dims(&[bh, n, d]),
        device: x.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,

    };

    ensure(&x.device, "rope_halfsplit_bf16_kernel", CUDA_ROPE_FUSED)?;
    let f = x
        .device
        .get_func("rope_halfsplit_bf16_kernel", "rope_halfsplit_bf16_kernel")
        .ok_or_else(|| Error::Cuda("rope_halfsplit_bf16_kernel missing".into()))?;

    let xs = match &x_flat.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => return Err(Error::InvalidOperation("rope expects BF16".into())),
    };
    let cs = match &cos_flat.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => return Err(Error::InvalidOperation("rope cos expects BF16".into())),
    };
    let ss = match &sin_flat.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => return Err(Error::InvalidOperation("rope sin expects BF16".into())),
    };
    let ys = match &mut out.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => unreachable!(),
    };
    let ys = ensure_unique_slice(ys)?;

    unsafe {
        f.launch(
            lc(total),
            (
                slice_ref(xs),
                slice_ref(cs),
                slice_ref(ss),
                ys,
                bh as i64,
                n as i64,
                half as i64,
                cos_bh as i64,
            ),
        )?;
    }

    let mut result = out.reshape(&[b, h, n, d])?;

    // Record autograd op so gradients flow through RoPE during training.
    // Save the 3D-flattened cos/sin so backward's rope_fused_bf16 gets
    // the layout it expects ([cos_bh, N, half]).
    if x.requires_grad {
        result.requires_grad = true;
        if crate::autograd::AutogradContext::is_recording() {
            crate::autograd::AutogradContext::record_op(
                result.id,
                crate::autograd::Op::RoPePrecomputed {
                    input: x.id,
                    cos: cos_flat.id,
                    sin: sin_flat.id,
                },
                vec![
                    (x.id, x.clone()),
                    (cos_flat.id, cos_flat.clone()),
                    (sin_flat.id, sin_flat.clone()),
                ],
            );
        }
    }

    Ok(result)
}

// ---- Fused modulate_pre: LayerNorm + (1+scale)*x + shift, single kernel ----

const CUDA_MODULATE_PRE: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void modulate_pre_bf16_kernel(
    const __nv_bfloat16* __restrict__ X,     // [rows, dim]
    const __nv_bfloat16* __restrict__ SCALE,  // [B, dim] — row b uses SCALE[b*dim..]
    const __nv_bfloat16* __restrict__ SHIFT,  // [B, dim]
    __nv_bfloat16* __restrict__ Y,            // [rows, dim]
    int rows, int dim, int seq_len, float eps)
{
    // One block per row. Shared mem for reduction.
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int batch_idx = row / seq_len;  // which batch element

    const __nv_bfloat16* x_row = X + (long)row * dim;
    __nv_bfloat16* y_row = Y + (long)row * dim;
    const __nv_bfloat16* scale_row = SCALE + (long)batch_idx * dim;
    const __nv_bfloat16* shift_row = SHIFT + (long)batch_idx * dim;

    // 1) Compute mean
    float local_sum = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x)
        local_sum += __bfloat162float(x_row[d]);
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float mean = smem[0] / dim;
    __syncthreads();

    // 2) Compute variance
    float local_var = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float v = __bfloat162float(x_row[d]) - mean;
        local_var += v * v;
    }
    smem[threadIdx.x] = local_var;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(smem[0] / dim + eps);
    __syncthreads();

    // 3) Normalize + modulate: (1 + scale) * norm(x) + shift
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float normed = (__bfloat162float(x_row[d]) - mean) * inv_std;
        float sc = __bfloat162float(scale_row[d]);
        float sh = __bfloat162float(shift_row[d]);
        y_row[d] = __float2bfloat16((1.0f + sc) * normed + sh);
    }
}

// B.3 — same LN + modulate as above, but reads shift/scale from a
// COMBINED modulation tensor [B, mod_dim, dim] via shift_idx. Replaces
// two narrow(mod, idx) calls in the caller with a single kernel that
// indexes directly. Every DiT block's modulate_pre fires this pattern
// (shift_msa/scale_msa pair, shift_mlp/scale_mlp pair, etc.) — so every
// block now has 2 fewer slice_copy kernel launches.
extern "C" __global__
void modulate_pre_split_apply_bf16_kernel(
    const __nv_bfloat16* __restrict__ X,     // [rows, dim]
    const __nv_bfloat16* __restrict__ MOD,    // [B, mod_dim, dim]
    __nv_bfloat16* __restrict__ Y,            // [rows, dim]
    int rows, int dim, int seq_len, int mod_dim, int shift_idx, float eps)
{
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int batch_idx = row / seq_len;

    const __nv_bfloat16* x_row = X + (long)row * dim;
    __nv_bfloat16* y_row = Y + (long)row * dim;
    // Shift is at mod[batch_idx, shift_idx, :]; scale is the next entry
    // (shift_idx + 1). Keep matching the original call order where callers
    // narrowed (mod, shift_idx, 1) then (mod, shift_idx+1, 1).
    const __nv_bfloat16* shift_row =
        MOD + ((long)batch_idx * mod_dim + shift_idx)     * dim;
    const __nv_bfloat16* scale_row =
        MOD + ((long)batch_idx * mod_dim + shift_idx + 1) * dim;

    // 1) mean
    float local_sum = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x)
        local_sum += __bfloat162float(x_row[d]);
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float mean = smem[0] / dim;
    __syncthreads();

    // 2) variance
    float local_var = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float v = __bfloat162float(x_row[d]) - mean;
        local_var += v * v;
    }
    smem[threadIdx.x] = local_var;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(smem[0] / dim + eps);
    __syncthreads();

    // 3) normalize + modulate
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float normed = (__bfloat162float(x_row[d]) - mean) * inv_std;
        float sc = __bfloat162float(scale_row[d]);
        float sh = __bfloat162float(shift_row[d]);
        y_row[d] = __float2bfloat16((1.0f + sc) * normed + sh);
    }
}
"#;

/// Fused modulate_pre: LayerNorm + (1+scale)*x + shift in one kernel.
///
/// `x`: [B, N, dim] BF16.
/// `shift`: [B, dim] BF16.
/// `scale`: [B, dim] BF16.
///
/// Returns [B, N, dim] BF16.
pub fn modulate_pre_fused_bf16(
    x: &Tensor,
    shift: &Tensor,
    scale: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), DType::BF16);
    let x_dims = x.shape().dims();
    if x_dims.len() != 3 {
        return Err(Error::InvalidOperation(format!(
            "modulate_pre_fused: expected 3D [B,N,dim], got {:?}", x_dims
        )));
    }
    let (b, n, dim) = (x_dims[0], x_dims[1], x_dims[2]);
    let rows = b * n;
    let total = rows * dim;

    let data = crate::cuda_alloc_pool::pool_alloc_u16(&x.device, total)?;
    let mut out = Tensor {
        storage: TensorStorage::BF16 { data: data.into(), numel: total },
        shape: x.shape().clone(),
        device: x.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,

    };

    ensure(&x.device, "modulate_pre_bf16_kernel", CUDA_MODULATE_PRE)?;
    let f = x.device
        .get_func("modulate_pre_bf16_kernel", "modulate_pre_bf16_kernel")
        .ok_or_else(|| Error::Cuda("modulate_pre_bf16_kernel missing".into()))?;

    let xs = match &x.storage { TensorStorage::BF16 { data, .. } => data, _ => return Err(Error::InvalidOperation("expects BF16".into())) };
    let scs = match &scale.storage { TensorStorage::BF16 { data, .. } => data, _ => return Err(Error::InvalidOperation("expects BF16".into())) };
    let shs = match &shift.storage { TensorStorage::BF16 { data, .. } => data, _ => return Err(Error::InvalidOperation("expects BF16".into())) };
    let ys = match &mut out.storage { TensorStorage::BF16 { data, .. } => data, _ => unreachable!() };
    let ys = ensure_unique_slice(ys)?;

    let threads = dim.min(256) as u32;
    let smem = threads * 4;
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem,
    };
    unsafe {
        f.launch(cfg, (slice_ref(xs), slice_ref(scs), slice_ref(shs), ys, rows as i32, dim as i32, n as i32, eps))?;
    }
    Ok(out)
}

/// B.3 — fused LayerNorm + modulate with SPLIT-APPLY from combined mod tensor.
///
/// Semantically identical to `modulate_pre_fused_bf16` but takes the full
/// modulation tensor `[B, mod_dim, dim]` and a `shift_idx`, reading shift
/// and scale rows directly inside the kernel. Replaces two `Tensor::narrow`
/// calls + the LN-modulate call chain with a single kernel.
///
/// Convention (matches prior caller code): shift = mod[:, shift_idx, :],
/// scale = mod[:, shift_idx + 1, :]. For DiT blocks with 6-way modulation
/// packed as (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
/// pass shift_idx = 0 for MSA pair, shift_idx = 3 for MLP pair.
///
/// Requires shift_idx + 1 < mod_dim (kernel asserts before launch).
pub fn modulate_pre_split_apply_bf16(
    x: &Tensor,
    modulation: &Tensor,
    shift_idx: usize,
    eps: f32,
) -> Result<Tensor> {
    debug_assert_eq!(x.dtype(), DType::BF16);
    debug_assert_eq!(modulation.dtype(), DType::BF16);
    let x_dims = x.shape().dims();
    if x_dims.len() != 3 {
        return Err(Error::InvalidOperation(format!(
            "modulate_pre_split_apply: expected 3D x [B,N,dim], got {:?}", x_dims
        )));
    }
    let m_dims = modulation.shape().dims();
    if m_dims.len() != 3 {
        return Err(Error::InvalidOperation(format!(
            "modulate_pre_split_apply: expected 3D modulation [B,mod_dim,dim], got {:?}", m_dims
        )));
    }
    let (b, n, dim) = (x_dims[0], x_dims[1], x_dims[2]);
    if m_dims[0] != b || m_dims[2] != dim {
        return Err(Error::InvalidOperation(format!(
            "modulate_pre_split_apply: x {:?} / mod {:?} batch or dim mismatch",
            x_dims, m_dims
        )));
    }
    let mod_dim = m_dims[1];
    if shift_idx + 1 >= mod_dim {
        return Err(Error::InvalidOperation(format!(
            "modulate_pre_split_apply: shift_idx {} + 1 >= mod_dim {}",
            shift_idx, mod_dim
        )));
    }
    let rows = b * n;
    let total = rows * dim;

    // Modulation must be contiguous (kernel reads via linear offsets).
    let modulation = modulation.contiguous()?;

    let data = crate::cuda_alloc_pool::pool_alloc_u16(&x.device, total)?;
    let mut out = Tensor {
        storage: TensorStorage::BF16 { data: data.into(), numel: total },
        shape: x.shape().clone(),
        device: x.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,
    };

    ensure(&x.device, "modulate_pre_bf16_kernel", CUDA_MODULATE_PRE)?;
    let f = x.device
        .get_func("modulate_pre_bf16_kernel", "modulate_pre_split_apply_bf16_kernel")
        .ok_or_else(|| Error::Cuda("modulate_pre_split_apply_bf16_kernel missing".into()))?;

    let xs = match &x.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => return Err(Error::InvalidOperation("expects BF16 x".into())),
    };
    let ms = match &modulation.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => return Err(Error::InvalidOperation("expects BF16 modulation".into())),
    };
    let ys = match &mut out.storage {
        TensorStorage::BF16 { data, .. } => data,
        _ => unreachable!(),
    };
    let ys = ensure_unique_slice(ys)?;

    let threads = dim.min(256) as u32;
    let smem = threads * 4;
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem,
    };
    unsafe {
        f.launch(cfg, (
            slice_ref(xs),
            slice_ref(ms),
            ys,
            rows as i32,
            dim as i32,
            n as i32,
            mod_dim as i32,
            shift_idx as i32,
            eps,
        ))?;
    }
    Ok(out)
}

// ---- Fused gate+residual: residual + gate * x, single kernel ----

const CUDA_GATE_RESIDUAL: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void gate_residual_bf16_kernel(
    const __nv_bfloat16* __restrict__ RESIDUAL,  // [B, N, dim]
    const __nv_bfloat16* __restrict__ GATE,       // [B, dim]
    const __nv_bfloat16* __restrict__ X,          // [B, N, dim]
    __nv_bfloat16* __restrict__ Y,                // [B, N, dim]
    long total, int dim, int seq_len)
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < total) {
        int d = idx % dim;
        int row = idx / dim;
        int batch_idx = row / seq_len;

        float r = __bfloat162float(RESIDUAL[idx]);
        float g = __bfloat162float(GATE[(long)batch_idx * dim + d]);
        float x = __bfloat162float(X[idx]);
        Y[idx] = __float2bfloat16(r + g * x);

        idx += gridDim.x * blockDim.x;
    }
}
"#;

/// Fused gate+residual: residual + gate * x in one kernel.
///
/// `residual`: [B, N, dim] BF16.
/// `gate`: [B, dim] BF16 (broadcast over N).
/// `x`: [B, N, dim] BF16.
///
/// Returns [B, N, dim] BF16.
pub fn gate_residual_fused_bf16(
    residual: &Tensor,
    gate: &Tensor,
    x: &Tensor,
) -> Result<Tensor> {
    debug_assert_eq!(residual.dtype(), DType::BF16);
    let dims = residual.shape().dims();
    if dims.len() != 3 {
        return Err(Error::InvalidOperation(format!(
            "gate_residual_fused: expected 3D [B,N,dim], got {:?}", dims
        )));
    }
    let (b, n, dim) = (dims[0], dims[1], dims[2]);
    let total = b * n * dim;

    let data = crate::cuda_alloc_pool::pool_alloc_u16(&residual.device, total)?;
    let mut out = Tensor {
        storage: TensorStorage::BF16 { data: data.into(), numel: total },
        shape: residual.shape().clone(),
        device: residual.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,

    };

    ensure(&residual.device, "gate_residual_bf16_kernel", CUDA_GATE_RESIDUAL)?;
    let f = residual.device
        .get_func("gate_residual_bf16_kernel", "gate_residual_bf16_kernel")
        .ok_or_else(|| Error::Cuda("gate_residual_bf16_kernel missing".into()))?;

    let rs = match &residual.storage { TensorStorage::BF16 { data, .. } => data, _ => return Err(Error::InvalidOperation("expects BF16".into())) };
    let gs = match &gate.storage { TensorStorage::BF16 { data, .. } => data, _ => return Err(Error::InvalidOperation("expects BF16".into())) };
    let xs_data = match &x.storage { TensorStorage::BF16 { data, .. } => data, _ => return Err(Error::InvalidOperation("expects BF16".into())) };
    let ys = match &mut out.storage { TensorStorage::BF16 { data, .. } => data, _ => unreachable!() };
    let ys = ensure_unique_slice(ys)?;

    unsafe {
        f.launch(lc(total), (slice_ref(rs), slice_ref(gs), slice_ref(xs_data), ys, total as i64, dim as i32, n as i32))?;
    }
    Ok(out)
}

// ---- Fused SwiGLU: silu(gate) * up in one kernel ----

const CUDA_SWIGLU_FUSED: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void swiglu_fused_bf16_kernel(
    const __nv_bfloat16* __restrict__ GATE,  // [total]
    const __nv_bfloat16* __restrict__ UP,    // [total]
    __nv_bfloat16* __restrict__ Y,           // [total]
    long total)
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < total) {
        float g = __bfloat162float(GATE[idx]);
        float u = __bfloat162float(UP[idx]);
        float silu_g = g / (1.0f + expf(-g));
        Y[idx] = __float2bfloat16(silu_g * u);
        idx += gridDim.x * blockDim.x;
    }
}
"#;

/// Fused SwiGLU: silu(gate) * up in one kernel (no intermediate silu tensor).
pub fn swiglu_fused_bf16(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    debug_assert_eq!(gate.dtype(), DType::BF16);
    let total = gate.shape().elem_count();

    let data = crate::cuda_alloc_pool::pool_alloc_u16(&gate.device, total)?;
    let mut out = Tensor {
        storage: TensorStorage::BF16 { data: data.into(), numel: total },
        shape: gate.shape().clone(),
        device: gate.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,

    };

    ensure(&gate.device, "swiglu_fused_bf16_kernel", CUDA_SWIGLU_FUSED)?;
    let f = gate.device
        .get_func("swiglu_fused_bf16_kernel", "swiglu_fused_bf16_kernel")
        .ok_or_else(|| Error::Cuda("swiglu_fused_bf16_kernel missing".into()))?;

    let gs = match &gate.storage { TensorStorage::BF16 { data, .. } => data, _ => return Err(Error::InvalidOperation("expects BF16".into())) };
    let us = match &up.storage { TensorStorage::BF16 { data, .. } => data, _ => return Err(Error::InvalidOperation("expects BF16".into())) };
    let ys = match &mut out.storage { TensorStorage::BF16 { data, .. } => data, _ => unreachable!() };
    let ys = ensure_unique_slice(ys)?;

    unsafe {
        f.launch(lc(total), (slice_ref(gs), slice_ref(us), ys, total as i64))?;
    }
    Ok(out)
}

// NOTE: a larger "qkv_rmsnorm_rope_cat_bf16" single-shot joint-attention
// kernel was prototyped here but exceeded cudarc 0.11.9's 12-tuple
// LaunchAsync limit. The simpler `qkv_split_permute_bf16` below captures
// most of the win on its own; a packed-args variant of the full joint
// kernel is queued for a follow-up once the wins here are measured.

// ---- Fused post-SDPA split: [B,H,N_txt+N_img,D] -> txt[B,N_txt,H*D] + img[B,N_img,H*D]

const CUDA_ATTN_SPLIT_TXT_IMG: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void attn_split_txt_img_bf16_kernel(
    const __nv_bfloat16* __restrict__ ATTN,  // [B, H, N_total, D], N_total = N_txt + N_img
    __nv_bfloat16* __restrict__ TXT,          // [B, N_txt, H*D]
    __nv_bfloat16* __restrict__ IMG,          // [B, N_img, H*D]
    int B, int H, int N_txt, int N_img, int D)
{
    int N_total = N_txt + N_img;
    long total = (long)B * H * N_total * D;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int d = idx % D;
    long t = idx / D;
    int n_tot = t % N_total;
    t = t / N_total;
    int h = t % H;
    int b = t / H;

    __nv_bfloat16 val = ATTN[idx];

    if (n_tot < N_txt) {
        // TXT[b, n_txt, h*D + d]
        int n_txt_i = n_tot;
        long out_idx = (long)b * N_txt * H * D + (long)n_txt_i * H * D + (long)h * D + d;
        TXT[out_idx] = val;
    } else {
        int n_img_i = n_tot - N_txt;
        long out_idx = (long)b * N_img * H * D + (long)n_img_i * H * D + (long)h * D + d;
        IMG[out_idx] = val;
    }
}
"#;

/// Fused post-SDPA split: reads a joint-attention output `[B, H, N_txt+N_img, D]`
/// and writes two pre-permuted+flattened outputs `txt [B, N_txt, H*D]` and
/// `img [B, N_img, H*D]` in a single pass.
///
/// Replaces:
///   txt_attn = attn_out.narrow(2, 0, n_txt)?;                    // kernel
///   img_attn = attn_out.narrow(2, n_txt, n_img)?;                // kernel
///   txt_flat = txt_attn.permute([0,2,1,3]).reshape([B,N_txt,HD]); // kernel (view)
///   img_flat = img_attn.permute([0,2,1,3]).reshape([B,N_img,HD]); // kernel (view)
/// — 4 BF16 materializing ops → 1.
///
/// Layout convention: txt tokens occupy positions [0, N_txt) of the joint
/// sequence, img tokens occupy [N_txt, N_txt+N_img). Matches FLUX / Chroma /
/// Klein double-block concat order `cat([txt, img], seq_dim=2)`.
pub fn attn_split_txt_img_bf16(
    attn_out: &Tensor,
    n_txt: usize,
    n_img: usize,
) -> Result<(Tensor, Tensor)> {
    debug_assert_eq!(attn_out.dtype(), DType::BF16);
    let dims = attn_out.shape().dims();
    if dims.len() != 4 {
        return Err(Error::InvalidOperation(format!(
            "attn_split_txt_img_bf16: expected 4D [B,H,N_total,D], got {:?}", dims
        )));
    }
    let (b, h, n_total, d) = (dims[0], dims[1], dims[2], dims[3]);
    if n_total != n_txt + n_img {
        return Err(Error::InvalidOperation(format!(
            "attn_split_txt_img_bf16: N_total={} != N_txt({}) + N_img({})",
            n_total, n_txt, n_img
        )));
    }

    let txt_numel = b * n_txt * h * d;
    let img_numel = b * n_img * h * d;
    let txt_data = crate::cuda_alloc_pool::pool_alloc_u16(&attn_out.device, txt_numel)?;
    let img_data = crate::cuda_alloc_pool::pool_alloc_u16(&attn_out.device, img_numel)?;
    let mut txt = Tensor {
        storage: TensorStorage::BF16 { data: txt_data.into(), numel: txt_numel },
        shape: Shape::from_dims(&[b, n_txt, h * d]),
        device: attn_out.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,

    };
    let mut img = Tensor {
        storage: TensorStorage::BF16 { data: img_data.into(), numel: img_numel },
        shape: Shape::from_dims(&[b, n_img, h * d]),
        device: attn_out.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,

    };

    ensure(&attn_out.device, "attn_split_txt_img_bf16_kernel", CUDA_ATTN_SPLIT_TXT_IMG)?;
    let f = attn_out.device
        .get_func("attn_split_txt_img_bf16_kernel", "attn_split_txt_img_bf16_kernel")
        .ok_or_else(|| Error::Cuda("attn_split_txt_img_bf16_kernel missing".into()))?;

    let xs = match &attn_out.storage { TensorStorage::BF16 { data, .. } => data, _ => return Err(Error::InvalidOperation("attn_out BF16".into())) };
    let ts = match &mut txt.storage { TensorStorage::BF16 { data, .. } => data, _ => unreachable!() };
    let ts = ensure_unique_slice(ts)?;
    let is_ = match &mut img.storage { TensorStorage::BF16 { data, .. } => data, _ => unreachable!() };
    let is_ = ensure_unique_slice(is_)?;

    let total = b * h * n_total * d;
    unsafe {
        f.launch(
            lc(total),
            (slice_ref(xs), ts, is_, b as i32, h as i32, n_txt as i32, n_img as i32, d as i32),
        )?;
    }
    Ok((txt, img))
}

/*
const _CUDA_QKV_RMSNORM_ROPE_CAT_RESERVED: &str = r#"
#include <cuda_bf16.h>
#include <math_constants.h>

extern "C" __global__
void qkv_rmsnorm_rope_cat_bf16_kernel(
    const __nv_bfloat16* __restrict__ IMG_QKV,    // [B, N_img, 3*H*D]
    const __nv_bfloat16* __restrict__ TXT_QKV,    // [B, N_txt, 3*H*D]
    const __nv_bfloat16* __restrict__ IMG_Q_W,    // [D] (per-head query rms scale)
    const __nv_bfloat16* __restrict__ IMG_K_W,    // [D]
    const __nv_bfloat16* __restrict__ TXT_Q_W,    // [D]
    const __nv_bfloat16* __restrict__ TXT_K_W,    // [D]
    const __nv_bfloat16* __restrict__ COS,        // [N_total, D/2]
    const __nv_bfloat16* __restrict__ SIN,        // [N_total, D/2]
    __nv_bfloat16* __restrict__ Q_OUT,            // [B, H, N_total, D]
    __nv_bfloat16* __restrict__ K_OUT,            // [B, H, N_total, D]
    __nv_bfloat16* __restrict__ V_OUT,            // [B, H, N_total, D]
    int B, int H, int N_img, int N_txt, int D, float eps)
{
    // grid.y = (b, h) via blockIdx.y = b*H + h
    // grid.x = position in N_total = N_txt + N_img
    int bh = blockIdx.y;
    int h  = bh % H;
    int b  = bh / H;
    int n_tot = blockIdx.x;
    int N_total = N_txt + N_img;
    if (n_tot >= N_total) return;

    // Choose stream: txt first, then img
    bool is_txt = n_tot < N_txt;
    int n_local = is_txt ? n_tot : (n_tot - N_txt);
    const __nv_bfloat16* QKV = is_txt ? TXT_QKV : IMG_QKV;
    const __nv_bfloat16* QW  = is_txt ? TXT_Q_W : IMG_Q_W;
    const __nv_bfloat16* KW  = is_txt ? TXT_K_W : IMG_K_W;
    int N_local = is_txt ? N_txt : N_img;

    long HD = (long)H * D;
    long row_stride = 3L * HD;
    long qkv_row = (long)b * N_local * row_stride + (long)n_local * row_stride + (long)h * D;

    // Load Q, K, V head vectors (D each) into registers via shared memory staging.
    // Each thread handles one d.
    int tid = threadIdx.x;
    if (tid >= D) return;

    float q = __bfloat162float(QKV[qkv_row + 0 * HD + tid]);
    float k = __bfloat162float(QKV[qkv_row + 1 * HD + tid]);
    float v = __bfloat162float(QKV[qkv_row + 2 * HD + tid]);

    // RMSNorm on Q and K (head-dim).
    // sum(x^2) / D, reduce across threads in block (block = D).
    extern __shared__ float ssum[];  // size 2*D: ssum[0..D-1]=q^2, ssum[D..2D-1]=k^2
    ssum[tid]     = q * q;
    ssum[D + tid] = k * k;
    __syncthreads();

    // Tree reduction (D is power of 2 in practice; generic otherwise)
    for (int stride = D >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ssum[tid]     += ssum[tid + stride];
            ssum[D + tid] += ssum[D + tid + stride];
        }
        __syncthreads();
    }
    float q_inv = rsqrtf(ssum[0] / (float)D + eps);
    float k_inv = rsqrtf(ssum[D] / (float)D + eps);
    __syncthreads();

    float q_w = __bfloat162float(QW[tid]);
    float k_w = __bfloat162float(KW[tid]);
    q = q * q_inv * q_w;
    k = k * k_inv * k_w;

    // RoPE: (x0, x1) where x0=elem at 2i, x1=elem at 2i+1 within paired halves.
    // flame rope_fused_bf16 layout: D/2 = half; for each thread d in [0,half):
    //   out[d]      = x[d]      * cos[d] - x[d+half] * sin[d]
    //   out[d+half] = x[d+half] * cos[d] + x[d]      * sin[d]
    int half = D >> 1;
    // Exchange via shared so every thread can see both halves.
    __shared__ float q_vec[512];  // enough for D<=512
    __shared__ float k_vec[512];
    q_vec[tid] = q;
    k_vec[tid] = k;
    __syncthreads();

    float q_rot, k_rot;
    if (tid < half) {
        float c = __bfloat162float(COS[(long)n_tot * half + tid]);
        float s = __bfloat162float(SIN[(long)n_tot * half + tid]);
        q_rot = q_vec[tid] * c - q_vec[tid + half] * s;
        k_rot = k_vec[tid] * c - k_vec[tid + half] * s;
    } else {
        int d_l = tid - half;
        float c = __bfloat162float(COS[(long)n_tot * half + d_l]);
        float s = __bfloat162float(SIN[(long)n_tot * half + d_l]);
        q_rot = q_vec[tid] * c + q_vec[d_l] * s;
        k_rot = k_vec[tid] * c + k_vec[d_l] * s;
    }

    long out_idx = (long)b * H * N_total * D + (long)h * N_total * D + (long)n_tot * D + tid;
    Q_OUT[out_idx] = __float2bfloat16(q_rot);
    K_OUT[out_idx] = __float2bfloat16(k_rot);
    V_OUT[out_idx] = __float2bfloat16(v);
}
"#;

/// Joint-attention fused pre-SDPA kernel.
///
/// Input:
///   `img_qkv` [B, N_img, 3*H*D] BF16 — img QKV linear output (raw, unreshaped)
///   `txt_qkv` [B, N_txt, 3*H*D] BF16 — txt QKV linear output
///   `img_q_w, img_k_w` [D] BF16 — per-head RMSNorm scales for img stream
///   `txt_q_w, txt_k_w` [D] BF16 — per-head RMSNorm scales for txt stream
///   `cos, sin` [N_total, D/2] BF16 — RoPE tables indexed by concatenated position
/// Output:
///   `(q, k, v)` each [B, H, N_txt+N_img, D] BF16. Q and K have RMSNorm applied
///    then RoPE. V is the raw values permuted to [B,H,N,D]. txt tokens occupy
///    positions [0, N_txt), img tokens occupy [N_txt, N_total).
///
/// Replaces: 6 narrows + 6 permutes + 4 rms_norm + 3 cats + 2 rope calls
///         = ~21 separate BF16 kernels → 1.
#[allow(dead_code)]
fn _qkv_rmsnorm_rope_cat_bf16_reserved(
    img_qkv: &Tensor,
    txt_qkv: &Tensor,
    img_q_w: &Tensor,
    img_k_w: &Tensor,
    txt_q_w: &Tensor,
    txt_k_w: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    debug_assert_eq!(img_qkv.dtype(), DType::BF16);
    debug_assert_eq!(txt_qkv.dtype(), DType::BF16);
    let img_dims = img_qkv.shape().dims();
    let txt_dims = txt_qkv.shape().dims();
    if img_dims.len() != 3 || txt_dims.len() != 3 {
        return Err(Error::InvalidOperation(
            "qkv_rmsnorm_rope_cat_bf16: img/txt qkv must be 3D".into(),
        ));
    }
    let b = img_dims[0];
    if txt_dims[0] != b {
        return Err(Error::InvalidOperation(format!(
            "qkv_rmsnorm_rope_cat_bf16: batch mismatch img={} txt={}",
            b, txt_dims[0]
        )));
    }
    let n_img = img_dims[1];
    let n_txt = txt_dims[1];
    let hd = heads * head_dim;
    if img_dims[2] != 3 * hd || txt_dims[2] != 3 * hd {
        return Err(Error::InvalidOperation(format!(
            "qkv_rmsnorm_rope_cat_bf16: last-dim must be 3*H*D={} got img={} txt={}",
            3 * hd,
            img_dims[2],
            txt_dims[2]
        )));
    }
    if head_dim > 512 {
        return Err(Error::InvalidOperation(format!(
            "qkv_rmsnorm_rope_cat_bf16: head_dim {head_dim} exceeds shared-mem cap 512"
        )));
    }
    let n_total = n_txt + n_img;
    let out_numel = b * heads * n_total * head_dim;
    let out_shape = Shape::from_dims(&[b, heads, n_total, head_dim]);

    let q_data = crate::cuda_alloc_pool::pool_alloc_u16(&img_qkv.device, out_numel)?;
    let k_data = crate::cuda_alloc_pool::pool_alloc_u16(&img_qkv.device, out_numel)?;
    let v_data = crate::cuda_alloc_pool::pool_alloc_u16(&img_qkv.device, out_numel)?;
    let mut q = Tensor {
        storage: TensorStorage::BF16 { data: q_data.into(), numel: out_numel },
        shape: out_shape.clone(),
        device: img_qkv.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,

    };
    let mut k = Tensor {
        storage: TensorStorage::BF16 { data: k_data.into(), numel: out_numel },
        shape: out_shape.clone(),
        device: img_qkv.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,

    };
    let mut v = Tensor {
        storage: TensorStorage::BF16 { data: v_data.into(), numel: out_numel },
        shape: out_shape,
        device: img_qkv.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,

    };

    ensure(
        &img_qkv.device,
        "qkv_rmsnorm_rope_cat_bf16_kernel",
        CUDA_QKV_RMSNORM_ROPE_CAT,
    )?;
    let f = img_qkv
        .device
        .get_func(
            "qkv_rmsnorm_rope_cat_bf16_kernel",
            "qkv_rmsnorm_rope_cat_bf16_kernel",
        )
        .ok_or_else(|| Error::Cuda("qkv_rmsnorm_rope_cat_bf16_kernel missing".into()))?;

    // cos/sin: accept [1,1,N,D/2] or [N,D/2]. Reshape to [N,D/2] flat.
    let half = head_dim / 2;
    let cos_flat = cos.reshape(&[n_total, half])?;
    let sin_flat = sin.reshape(&[n_total, half])?;

    let ixs = match &img_qkv.storage { TensorStorage::BF16 { data, .. } => data, _ => return Err(Error::InvalidOperation("img_qkv BF16".into())) };
    let txs = match &txt_qkv.storage { TensorStorage::BF16 { data, .. } => data, _ => return Err(Error::InvalidOperation("txt_qkv BF16".into())) };
    let iqw = match &img_q_w.storage { TensorStorage::BF16 { data, .. } => data, _ => return Err(Error::InvalidOperation("img_q_w BF16".into())) };
    let ikw = match &img_k_w.storage { TensorStorage::BF16 { data, .. } => data, _ => return Err(Error::InvalidOperation("img_k_w BF16".into())) };
    let tqw = match &txt_q_w.storage { TensorStorage::BF16 { data, .. } => data, _ => return Err(Error::InvalidOperation("txt_q_w BF16".into())) };
    let tkw = match &txt_k_w.storage { TensorStorage::BF16 { data, .. } => data, _ => return Err(Error::InvalidOperation("txt_k_w BF16".into())) };
    let cs = match &cos_flat.storage { TensorStorage::BF16 { data, .. } => data, _ => return Err(Error::InvalidOperation("cos BF16".into())) };
    let ss = match &sin_flat.storage { TensorStorage::BF16 { data, .. } => data, _ => return Err(Error::InvalidOperation("sin BF16".into())) };
    let qs = match &mut q.storage { TensorStorage::BF16 { data, .. } => data, _ => unreachable!() };
    let qs = ensure_unique_slice(qs)?;
    let ks = match &mut k.storage { TensorStorage::BF16 { data, .. } => data, _ => unreachable!() };
    let ks = ensure_unique_slice(ks)?;
    let vs = match &mut v.storage { TensorStorage::BF16 { data, .. } => data, _ => unreachable!() };
    let vs = ensure_unique_slice(vs)?;

    let cfg = LaunchConfig {
        grid_dim: (n_total as u32, (b * heads) as u32, 1),
        block_dim: (head_dim as u32, 1, 1),
        shared_mem_bytes: (2 * head_dim * 4) as u32, // 2*D floats for q^2/k^2 reduction
    };
    unsafe {
        f.launch(
            cfg,
            (
                slice_ref(ixs),
                slice_ref(txs),
                slice_ref(iqw),
                slice_ref(ikw),
                slice_ref(tqw),
                slice_ref(tkw),
                slice_ref(cs),
                slice_ref(ss),
                qs,
                ks,
                vs,
                b as i32,
                heads as i32,
                n_img as i32,
                n_txt as i32,
                head_dim as i32,
                eps,
            ),
        )?;
    }
    Ok((q, k, v))
}
*/

// ---- Fused QKV split + permute: one kernel replaces 3 narrows + 3 permutes ----
//
// The attention layer pattern is:
//   qkv  = Linear(x)                              // [B, N, 3*H*D]
//   q,k,v = qkv.split(inner, dim=2)              // 3 × narrow  (materializes!)
//   q = q.view([B,N,H,D]).permute([0,2,1,3])     // materializes
//   k, v = same                                   // materializes
//
// That's 3 narrow kernels + 3 permute kernels per attention, each with its own
// HBM round-trip, per block, per forward, per step. This replaces the six ops
// with ONE kernel that reads the fused QKV tensor once and writes three
// pre-permuted outputs at final [B,H,N,D] layout in a single pass.

const CUDA_QKV_SPLIT_PERMUTE: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__
void qkv_split_permute_bf16_kernel(
    const __nv_bfloat16* __restrict__ QKV,  // [B, N, 3*H*D]
    __nv_bfloat16* __restrict__ Q,          // [B, H, N, D]
    __nv_bfloat16* __restrict__ K,          // [B, H, N, D]
    __nv_bfloat16* __restrict__ V,          // [B, H, N, D]
    int B, int H, int N, int D)
{
    long total = (long)B * H * N * D;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int d = idx % D;
    long t = idx / D;
    int n = t % N;
    t = t / N;
    int h = t % H;
    int b = t / H;

    long HD  = (long)H * D;
    long row_stride = 3 * HD;               // per-token stride in QKV
    long qkv_row = (long)b * N * row_stride + (long)n * row_stride;
    long head_off = (long)h * D + d;

    Q[idx] = QKV[qkv_row + 0 * HD + head_off];
    K[idx] = QKV[qkv_row + 1 * HD + head_off];
    V[idx] = QKV[qkv_row + 2 * HD + head_off];
}
"#;

/// Fused QKV split + permute: one kernel replaces 3 narrows + 3 permutes.
///
/// Input `qkv` shape `[B, N, 3*H*D]` BF16 — the raw linear output where the
/// last dim is `[q_head_0 .. q_head_{H-1}, k_head_0 .. k_head_{H-1}, v_head_0 ..]`.
///
/// Returns `(q, k, v)` each shape `[B, H, N, D]` BF16, ready for SDPA.
///
/// Semantics are bit-identical to the slow path:
///   let q = qkv.narrow(2, 0,     h*d)?.reshape([b,n,h,d])?.permute([0,2,1,3])?;
///   let k = qkv.narrow(2, h*d,   h*d)?.reshape([b,n,h,d])?.permute([0,2,1,3])?;
///   let v = qkv.narrow(2, 2*h*d, h*d)?.reshape([b,n,h,d])?.permute([0,2,1,3])?;
/// — but fused into a single kernel launch with one read of `qkv` and three
/// writes of contiguous outputs.
pub fn qkv_split_permute_bf16(
    qkv: &Tensor,
    heads: usize,
    head_dim: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    debug_assert_eq!(qkv.dtype(), DType::BF16);
    let dims = qkv.shape().dims();
    if dims.len() != 3 {
        return Err(Error::InvalidOperation(format!(
            "qkv_split_permute_bf16: expected 3D [B,N,3*H*D], got {:?}", dims
        )));
    }
    let (b, n, c) = (dims[0], dims[1], dims[2]);
    let hd = heads * head_dim;
    if c != 3 * hd {
        return Err(Error::InvalidOperation(format!(
            "qkv_split_permute_bf16: last dim {c} != 3 * H*D = 3*{heads}*{head_dim} = {}",
            3 * hd
        )));
    }
    let out_numel = b * heads * n * head_dim;
    let out_shape = Shape::from_dims(&[b, heads, n, head_dim]);

    let q_data = crate::cuda_alloc_pool::pool_alloc_u16(&qkv.device, out_numel)?;
    let k_data = crate::cuda_alloc_pool::pool_alloc_u16(&qkv.device, out_numel)?;
    let v_data = crate::cuda_alloc_pool::pool_alloc_u16(&qkv.device, out_numel)?;
    let mut q = Tensor {
        storage: TensorStorage::BF16 { data: q_data.into(), numel: out_numel },
        shape: out_shape.clone(),
        device: qkv.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,

    };
    let mut k = Tensor {
        storage: TensorStorage::BF16 { data: k_data.into(), numel: out_numel },
        shape: out_shape.clone(),
        device: qkv.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,

    };
    let mut v = Tensor {
        storage: TensorStorage::BF16 { data: v_data.into(), numel: out_numel },
        shape: out_shape,
        device: qkv.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
        custom_strides: None,
        view_offset: 0,

    };

    ensure(&qkv.device, "qkv_split_permute_bf16_kernel", CUDA_QKV_SPLIT_PERMUTE)?;
    let f = qkv.device
        .get_func("qkv_split_permute_bf16_kernel", "qkv_split_permute_bf16_kernel")
        .ok_or_else(|| Error::Cuda("qkv_split_permute_bf16_kernel missing".into()))?;

    let xs = match &qkv.storage { TensorStorage::BF16 { data, .. } => data, _ => return Err(Error::InvalidOperation("qkv_split_permute_bf16: input must be BF16".into())) };
    let qs = match &mut q.storage { TensorStorage::BF16 { data, .. } => data, _ => unreachable!() };
    let qs = ensure_unique_slice(qs)?;
    let ks = match &mut k.storage { TensorStorage::BF16 { data, .. } => data, _ => unreachable!() };
    let ks = ensure_unique_slice(ks)?;
    let vs = match &mut v.storage { TensorStorage::BF16 { data, .. } => data, _ => unreachable!() };
    let vs = ensure_unique_slice(vs)?;

    unsafe {
        f.launch(
            lc(out_numel),
            (slice_ref(xs), qs, ks, vs, b as i32, heads as i32, n as i32, head_dim as i32),
        )?;
    }
    Ok((q, k, v))
}
