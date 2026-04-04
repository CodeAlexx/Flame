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
        .map_err(|e| Error::Cuda(format!("nvrtc {}: {:?}", name, e)))?;
    dev.load_ptx(ptx, name, &[name])
        .map_err(|e| Error::Cuda(format!("load_ptx {}: {:?}", name, e)))?;
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
        .map_err(|e| Error::Cuda(format!("alloc gelu bf16: {:?}", e)))?;
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
        .map_err(|e| Error::Cuda(format!("alloc square bf16: {:?}", e)))?;
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

    let data = unsafe { x.device.alloc::<u16>(n) }
        .map_err(|e| Error::Cuda(format!("alloc softmax bf16: {:?}", e)))?;
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
    let data = unsafe { x.device.alloc::<u16>(n) }
        .map_err(|e| Error::Cuda(format!("alloc silu bf16: {:?}", e)))?;
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

extern "C" __global__
void rope_halfsplit_bf16_kernel(
    const __nv_bfloat16* __restrict__ X,   // [BH, N, D]  (D = 2*half)
    const __nv_bfloat16* __restrict__ COS,  // [1, N, half]
    const __nv_bfloat16* __restrict__ SIN,  // [1, N, half]
    __nv_bfloat16* __restrict__ Y,          // [BH, N, D]
    long bh, long n, long half)
{
    // Half-split RoPE (HuggingFace rotate_half convention):
    // Pairs elements across halves: (d, d+half).
    // Used by Qwen3, LLaMA, Mistral.
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long total = bh * n * half;
    while (idx < total) {
        long tmp = idx;
        long d = tmp % half; tmp /= half;
        long seq = tmp % n;   tmp /= n;
        long b = tmp;

        long base = (b * n + seq) * (2 * half);
        long cos_idx = seq * half + d;

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

    let data = unsafe { x.device.alloc::<u16>(out_n) }
        .map_err(|e| Error::Cuda(format!("alloc rope bf16: {:?}", e)))?;
    let mut out = Tensor {
        storage: TensorStorage::BF16 {
            data: data.into(),
            numel: out_n,
        },
        shape: Shape::from_dims(&[bh, n, d]),
        device: x.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
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
    let cos_flat = cos.reshape(&[1, n, half])?;
    let sin_flat = sin.reshape(&[1, n, half])?;

    let total = bh * n * half;
    let out_n = bh * n * d;

    let data = unsafe { x.device.alloc::<u16>(out_n) }
        .map_err(|e| Error::Cuda(format!("alloc rope halfsplit bf16: {:?}", e)))?;
    let mut out = Tensor {
        storage: TensorStorage::BF16 {
            data: data.into(),
            numel: out_n,
        },
        shape: Shape::from_dims(&[bh, n, d]),
        device: x.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
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
            ),
        )?;
    }

    out.reshape(&[b, h, n, d])
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

    let data = unsafe { x.device.alloc::<u16>(total) }
        .map_err(|e| Error::Cuda(format!("alloc modulate_pre: {:?}", e)))?;
    let mut out = Tensor {
        storage: TensorStorage::BF16 { data: data.into(), numel: total },
        shape: x.shape().clone(),
        device: x.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
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

    let data = unsafe { residual.device.alloc::<u16>(total) }
        .map_err(|e| Error::Cuda(format!("alloc gate_residual: {:?}", e)))?;
    let mut out = Tensor {
        storage: TensorStorage::BF16 { data: data.into(), numel: total },
        shape: residual.shape().clone(),
        device: residual.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
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

    let data = unsafe { gate.device.alloc::<u16>(total) }
        .map_err(|e| Error::Cuda(format!("alloc swiglu: {:?}", e)))?;
    let mut out = Tensor {
        storage: TensorStorage::BF16 { data: data.into(), numel: total },
        shape: gate.shape().clone(),
        device: gate.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
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
