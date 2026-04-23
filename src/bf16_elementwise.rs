//! Non-elementwise BF16 structured kernels that used to live alongside the
//! elementwise zoo.
//!
//! After the TensorIterator migration (Phases 4–11), this file holds only
//! the structurally-different survivors:
//!
//!   - `softmax_lastdim_bf16` — row-wise reduction (max + sum) + exp + div
//!   - `transpose2d_bf16`    — memory-layout-specific 2D transpose
//!   - `patchify_bf16` / `unpatchify_bf16` — DiT patch ops (reshape + gather)
//!
//! All pointwise ops previously in this file (add, mul, sub, div, max,
//! min, cmp, abs, the strided launchers and broadcast-spec helpers) have
//! been removed. Those ops now route through
//! `tensor_iterator::ops::{unary,binary,transcendentals,comparison}`.
//!
//! Per `PyTorch TensorIterator port plan` §7 do-not-touch list.

use std::sync::Arc;
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use crate::dtype::DType;
use crate::tensor_storage::TensorStorage;
use crate::{Error, Result, Shape, Tensor, TensorId};

#[inline]
fn lc(n: usize) -> LaunchConfig {
    LaunchConfig::for_num_elems(n as u32)
}

/// Load (if needed) and return a CUDA function in a single HashMap lookup on the hot path.
/// Cold path: compiles from source and loads into device cache.
fn ensure_and_get(
    dev: &Arc<CudaDevice>,
    nm: &'static str,
    code: &'static str,
) -> Result<cudarc::driver::CudaFunction> {
    if let Some(f) = dev.get_func(nm, nm) {
        return Ok(f);
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
    dev.get_func(nm, nm)
        .ok_or_else(|| Error::Cuda(format!("kernel {nm} not found after load")))
}

// ---------------------------------------------------------------------------
// softmax along last dim (row-wise reduction — NOT elementwise)
// ---------------------------------------------------------------------------

// Fused BF16 softmax kernel. Warp-shuffle reductions, vectorized BF16 loads.
// Each block handles one row. 256 threads (8 warps), elements in registers.
// 2-pass: (1) online max+sum, (2) normalize+write.
const CUDA_SOFTMAX_LASTDIM_BF16: &str = r#"
#include <cuda_bf16.h>

#define LOCAL_FLT_MAX 3.402823466e+38f
#define FULL_MASK 0xFFFFFFFF

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val, offset));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(FULL_MASK, val, offset);
    return val;
}

extern "C" __global__
void softmax_lastdim_bf16_kernel(const __nv_bfloat16* __restrict__ X,
                                  __nv_bfloat16* __restrict__ Y,
                                  long rows, long cols) {
    __shared__ float warp_smem[32];
    __shared__ float warp_smem2[32];

    long row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;

    const __nv_bfloat16* x_row = X + row * cols;
    __nv_bfloat16* y_row = Y + row * cols;

    // Pass 1: online softmax — compute max AND sum in a single pass.
    float local_max = -LOCAL_FLT_MAX;
    float local_sum = 0.0f;
    for (long c = tid; c < cols; c += blockDim.x) {
        float v = __bfloat162float(x_row[c]);
        if (v > local_max) {
            local_sum *= __expf(local_max - v);
            local_max = v;
        }
        local_sum += __expf(v - local_max);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_max = __shfl_xor_sync(FULL_MASK, local_max, offset);
        float other_sum = __shfl_xor_sync(FULL_MASK, local_sum, offset);
        float new_max = fmaxf(local_max, other_max);
        local_sum = local_sum * __expf(local_max - new_max)
                   + other_sum * __expf(other_max - new_max);
        local_max = new_max;
    }

    if (lane_id == 0) {
        warp_smem[warp_id] = local_max;
        warp_smem2[warp_id] = local_sum;
    }
    __syncthreads();
    if (warp_id == 0) {
        float m = (lane_id < num_warps) ? warp_smem[lane_id] : -LOCAL_FLT_MAX;
        float s = (lane_id < num_warps) ? warp_smem2[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float om = __shfl_xor_sync(FULL_MASK, m, offset);
            float os = __shfl_xor_sync(FULL_MASK, s, offset);
            float nm = fmaxf(m, om);
            s = s * __expf(m - nm) + os * __expf(om - nm);
            m = nm;
        }
        warp_smem[0] = m;
        warp_smem2[0] = s;
    }
    __syncthreads();
    float row_max = warp_smem[0];
    float inv_sum = 1.0f / warp_smem2[0];

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

    let f = ensure_and_get(&x.device, "softmax_lastdim_bf16_kernel", CUDA_SOFTMAX_LASTDIM_BF16)?;

    let block_size = if cols <= 128 { 128usize }
                     else if cols <= 256 { 256 }
                     else { 256 };

    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let x_ptr = x.as_device_ptr_bf16("softmax_lastdim_bf16:x")? as u64;
    let o_ptr = out.as_mut_device_ptr_bf16("softmax_lastdim_bf16:out")? as u64;
    let rows_i = rows as i64;
    let cols_i = cols as i64;

    let mut params: [*mut std::ffi::c_void; 4] = [
        &x_ptr as *const u64 as *mut std::ffi::c_void,
        &o_ptr as *const u64 as *mut std::ffi::c_void,
        &rows_i as *const i64 as *mut std::ffi::c_void,
        &cols_i as *const i64 as *mut std::ffi::c_void,
    ];

    unsafe {
        f.launch(cfg, &mut params[..])
            .map_err(|e| Error::Cuda(format!("softmax_lastdim_bf16 launch: {:?}", e)))?;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// 2D transpose (memory-layout-specific — NOT elementwise)
// ---------------------------------------------------------------------------

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

pub fn transpose2d_bf16(tensor: &Tensor) -> Result<Tensor> {
    if tensor.dtype() != DType::BF16 {
        return Err(Error::InvalidInput(
            "transpose2d_bf16 expects BF16 tensor".into(),
        ));
    }
    // Stride refactor Phase 2a safety net: the kernel walks storage linearly.
    let tensor_owned;
    let tensor = if tensor.is_contiguous() {
        tensor
    } else {
        tensor_owned = tensor.contiguous()?;
        &tensor_owned
    };
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(Error::InvalidInput(format!(
            "transpose2d_bf16 expects 2D tensor, got {:?}",
            dims
        )));
    }
    let rows = dims[0];
    let cols = dims[1];

    let f = ensure_and_get(&tensor.device, "transpose2d_bf16_kernel", CUDA_TRANSPOSE2D_BF16)?;

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

// ---------------------------------------------------------------------------
// Fused patchify / unpatchify — BF16, no 6D permute, no F32 round-trip.
// DiT patch ops — structured reshape + gather, NOT elementwise.
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
        int d = (int)(tid % patch_dim);
        long bn = tid / patch_dim;
        int n = (int)(bn % N);
        int b = (int)(bn / N);

        int h_patch = n / pw;
        int w_patch = n % pw;

        int c = d % C;
        int p_w = (d / C) % p;
        int p_h = d / (p * C);

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
        int w = (int)(tid % W);
        long tmp = tid / W;
        int h = (int)(tmp % H);
        tmp /= H;
        int c = (int)(tmp % C);
        int b = (int)(tmp / C);

        int h_patch = h / p;
        int w_patch = w / p;
        int p_h = h % p;
        int p_w = w % p;

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

    let f = ensure_and_get(&x.device, "patchify_bf16_kernel", CUDA_PATCHIFY_BF16)?;

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

    let f = ensure_and_get(&x.device, "unpatchify_bf16_kernel", CUDA_UNPATCHIFY_BF16)?;

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
