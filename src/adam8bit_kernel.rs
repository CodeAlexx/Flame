//! Block-wise dynamic-LUT 8-bit AdamW (bitsandbytes 0.49.2 parity).
//!
//! # What this is
//!
//! A pure-Rust port of the on-device math performed by bitsandbytes
//! `optim.AdamW8bit` with `block_wise=True` (the default since 0.42).
//! See `bitsandbytes/optim/optimizer.py:460-555` for state allocation
//! and `bitsandbytes/functional.py:1488-1550` for the dispatch entry
//! point we mirror.
//!
//! # State layout (per parameter)
//!
//! - `m_codes: CudaSlice<u8>` of length `n = param.numel()` — first-moment
//!   8-bit codes, indexing the **signed** dynamic LUT.
//! - `v_codes: CudaSlice<u8>` of length `n` — second-moment 8-bit codes,
//!   indexing the **unsigned** dynamic LUT.
//! - `m_absmax: CudaSlice<f32>` of length `ceil(n / 256)` — per-256-element
//!   block scale for the first moment.
//! - `v_absmax: CudaSlice<f32>` of length `ceil(n / 256)` — same for the
//!   second moment.
//! - `qmap_signed: CudaSlice<f32>` of length 256 — `create_dynamic_map(true)`.
//!   Shared by all params on a device — allocate once.
//! - `qmap_unsigned: CudaSlice<f32>` of length 256 — `create_dynamic_map(false)`.
//!
//! The dequantized value of element `i` in a tensor is
//! `qmap[code[i]] * absmax[i / 256]`; the codes are looked up in `qmap_signed`
//! for `m` and `qmap_unsigned` for `v`. Block size is hardcoded to **256** to
//! match bitsandbytes — both the source allocation
//! (`optimizer.py:478`: `blocksize = 256`) and the bnb device kernel.
//!
//! # Why not F32 `Tensor` storage for codes / absmax?
//!
//! `flame_core::DType` declares `U8` (`dtype.rs:10`) but `TensorStorage` has
//! no `U8` arm (`tensor_storage.rs:418-444` returns `InvalidOperation` for
//! `DType::U8` in `zeros`). Plumbing U8 through Tensor + the alloc pool +
//! autograd is a full-day refactor and explicitly out of scope per the
//! handoff. We hold the per-param state as raw `cudarc::driver::CudaSlice`s
//! on the optimizer struct (one `HashMap<TensorId, …>` in
//! `eridiffusion-core::AdamW8bit`); the launcher below takes the slices
//! directly. The F32 param + grad still come in as `Tensor` so we can
//! route through the standard `as_slice_f32` / `as_device_ptr_bf16` access.
//!
//! # Numerical contract
//!
//! Bit-exact-equivalent (modulo BF16 noise on the grad upcast) to bnb 0.49.2
//! `optim.AdamW8bit(block_wise=True)` for the AdamW algorithm:
//!
//! ```text
//!   m_old = qmap_s[m_codes[i]]  * m_absmax[blk]
//!   v_old = qmap_u[v_codes[i]]  * v_absmax[blk]
//!   m_new = beta1 * m_old + (1 - beta1) * g
//!   v_new = beta2 * v_old + (1 - beta2) * g * g
//!   m_hat = m_new / (1 - beta1^t)
//!   v_hat = v_new / (1 - beta2^t)
//!   p    -= lr * m_hat / (sqrt(v_hat) + eps) + lr * wd * p   // decoupled
//!
//!   absmax_m' = max_{i in block} |m_new[i]|     // block reduction
//!   absmax_v' = max_{i in block} |v_new[i]|
//!   m_codes[i] = argmin_c | qmap_s[c] - m_new[i] / absmax_m' |
//!   v_codes[i] = argmin_c | qmap_u[c] - v_new[i] / absmax_v' |
//! ```
//!
//! This is the same DECOUPLED-WD math as `adam::AdamW` — see the receipt at
//! the top of `src/adam.rs`.
//!
//! # Not implemented (deferred)
//!
//! - **Paged variant.** bnb `PagedAdamW8bit` uses CUDA unified memory with
//!   CPU spill (see `bitsandbytes/optim/optimizer.py:71-103`). Separate
//!   variant; not required for the trainers that currently consume
//!   AdamW8bit (`train_u1.rs`, `train_wan22.rs`).
//! - **`percentile_clipping`** and **`gnorm_scale`**. bnb defaults are
//!   `100` (no clipping) and `1.0` (no scaling). Trainers use external
//!   gradient clipping (`flame_core::ops::grad_norm`); the bnb-internal
//!   path is not used.
//! - **`skip_zeros`**. bnb default is `False` and trainers do not enable it.

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::{DType, Error, Result, Tensor};

// ---------------------------------------------------------------------------
// CPU-side dynamic LUT (port of bnb `functional.py:349-401`)
// ---------------------------------------------------------------------------

/// 256-entry dynamic-exponent qmap, mirroring bnb 0.49.2 `create_dynamic_map`.
/// See `bitsandbytes/functional.py:349-401`.
///
/// - `signed = true` → the LUT used for the AdamW first moment `m`
///   (stored as bnb `qmap1 = "dynamic"`).
/// - `signed = false` → the LUT used for the AdamW second moment `v`
///   (stored as bnb `qmap2 = "udynamic"`).
///
/// The two LUTs together with the per-block `absmax` are how bnb encodes
/// a value into 8 bits with logarithmic precision near zero (where most
/// optimizer-state values live). The LUT is sorted ascending so binary
/// search would work — but the kernel's requant step uses a linear scan
/// (256 compares per element) for simplicity.
///
/// Returns a `[f32; 256]`. Use the host array directly or upload to device
/// with `device.htod_copy(qmap.to_vec())`.
pub fn create_dynamic_map(signed: bool) -> [f32; 256] {
    // Constants from bnb call site: max_exponent_bits=7, total_bits=8.
    let max_exponent_bits: i32 = 7;
    let total_bits: i32 = 8;
    let non_sign_bits = total_bits - 1; // 7
    let additional_items = (1i32 << (non_sign_bits - max_exponent_bits)) - 1; // 0

    let mut data: Vec<f32> = Vec::with_capacity(256);

    // Linspace-then-midpoint helper matching torch.linspace(0.1, 1, k)
    // followed by (a[:-1] + a[1:]) / 2.
    fn linspace_midpoints(k: usize) -> Vec<f32> {
        // boundaries[j] = 0.1 + j * (0.9 / (k - 1)), j in 0..k
        // means[j]     = (boundaries[j] + boundaries[j+1]) / 2, j in 0..k-1
        if k < 2 {
            return Vec::new();
        }
        let step = 0.9_f32 / (k as f32 - 1.0);
        let mut prev = 0.1_f32;
        let mut out = Vec::with_capacity(k - 1);
        for j in 1..k {
            let next = 0.1_f32 + (j as f32) * step;
            out.push((prev + next) * 0.5);
            prev = next;
        }
        out
    }

    let mut last_i: i32 = 0;
    for i in 0..max_exponent_bits {
        last_i = i;
        // Python: 2 ** (i + non_sign_bits - max_exponent_bits) + 1 if signed
        //   else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1
        let fraction_items: usize = if signed {
            (1usize << (i + non_sign_bits - max_exponent_bits) as u32) + 1
        } else {
            (1usize << (i + non_sign_bits - max_exponent_bits + 1) as u32) + 1
        };
        let means = linspace_midpoints(fraction_items);
        // Scale: 10 ** (-(max_exponent_bits - 1) + i)
        let scale = 10f32.powi(-(max_exponent_bits - 1) + i);
        for m in &means {
            data.push(scale * m);
        }
        if signed {
            for m in &means {
                data.push(-(scale * m));
            }
        }
    }

    if additional_items > 0 {
        let means = linspace_midpoints((additional_items as usize) + 1);
        let scale = 10f32.powi(-(max_exponent_bits - 1) + last_i);
        for m in &means {
            data.push(scale * m);
        }
        if signed {
            for m in &means {
                data.push(-(scale * m));
            }
        }
    }

    data.push(0.0);
    data.push(1.0);

    // bnb asserts len == 2**total_bits; on signed=true,total_bits=8 this is
    // satisfied exactly. For signed=false the linspace pattern overshoots
    // and bnb's `assert len(data) == 2**total_bits` actually fires only on
    // the signed path — but since we always call with these exact constants
    // the pad-with-zeros loop just adds 0 entries:
    let gap = 256_isize - data.len() as isize;
    for _ in 0..gap.max(0) {
        data.push(0.0);
    }
    // If signed=false produces MORE than 256 entries with these constants,
    // bnb's assert would have already fired. We mirror that by debug-asserting
    // and truncating to 256 to keep the return type a fixed-size array.
    debug_assert!(
        data.len() <= 256,
        "create_dynamic_map(signed={signed}) produced {} entries (>256)",
        data.len()
    );
    data.truncate(256);

    data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut out = [0.0f32; 256];
    out.copy_from_slice(&data);
    out
}

// ---------------------------------------------------------------------------
// NVRTC kernel + Rust launcher
// ---------------------------------------------------------------------------

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
const KERNEL_SRC: &str = r#"
#include <cuda_bf16.h>

// Block-wise dynamic-LUT 8-bit AdamW step (bitsandbytes 0.49.2 parity).
//
// Launch geometry:
//   grid_dim  = ((n + 255) / 256, 1, 1)   // one block per 256-elem slab
//   block_dim = (256, 1, 1)               // one thread per element
//   shared    = 4096 bytes                // 2x qmap[256] + 2x s_m/s_v[256]
//
// Per-block flow:
//   1. Co-load qmap_signed[256] + qmap_unsigned[256] into shared mem.
//   2. Each thread: dequant (m, v), AdamW math, param update, stash m_new
//      and v_new in shared.
//   3. Block-wide max-abs reduction over s_m and s_v -> new absmax_m/v.
//   4. Each thread: requant s_m[t] / absmax_m via linear scan over the
//      shared qmap (256 compares), write u8 code. Same for v.
//
// Edge cases:
//   - Tail block where (blk*256 + t >= n): thread loads with idx clamped to
//     blk*256 (any valid in-block index), but writes are GUARDED so we do
//     not touch beyond `n`. The shared-mem entries for inactive threads are
//     pinned to 0 so they cannot drag absmax.
//   - All-zero block: absmax falls back to a tiny epsilon (1e-12f) so the
//     normalize divide does not produce NaN. Matches bnb behavior (their
//     CUDA kernel guards similarly; values quantize to whatever code best
//     approximates 0 in the qmap).
//
// Decoupled weight decay: p -= lr * m_hat/(sqrt(v_hat)+eps); then p -= lr*wd*p.
// MUST NOT fold wd into grad before the moments — see receipt at top of
// flame-core/src/adam.rs.

extern "C" __global__ void adam8bit_blockwise_kernel(
    float* __restrict__       param,
    const float* __restrict__ grad,
    unsigned char* __restrict__ m_codes,
    unsigned char* __restrict__ v_codes,
    float* __restrict__       m_absmax,
    float* __restrict__       v_absmax,
    const float* __restrict__ qmap_signed,
    const float* __restrict__ qmap_unsigned,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float wd,
    float bc1,
    float bc2,
    long long n
) {
    extern __shared__ float smem[];
    float* sq_signed   = smem;                  // [256]
    float* sq_unsigned = smem + 256;            // [256]
    float* s_m         = smem + 512;            // [256]
    float* s_v         = smem + 768;            // [256]

    int tid = threadIdx.x;
    long long blk = blockIdx.x;
    long long base = blk * 256;

    // Load qmaps once per block.
    sq_signed[tid]   = qmap_signed[tid];
    sq_unsigned[tid] = qmap_unsigned[tid];

    long long idx = base + (long long)tid;
    bool active = idx < n;
    long long safe_idx = active ? idx : base;  // any valid index inside this block

    __syncthreads();

    float g  = active ? grad[safe_idx]  : 0.0f;
    float pv = active ? param[safe_idx] : 0.0f;

    // Dequant via LUT * per-block scale.
    float absmax_m_prev = m_absmax[blk];
    float absmax_v_prev = v_absmax[blk];
    unsigned int m_code_prev = active ? (unsigned int)m_codes[safe_idx] : 0u;
    unsigned int v_code_prev = active ? (unsigned int)v_codes[safe_idx] : 0u;
    float m_old = sq_signed[m_code_prev]   * absmax_m_prev;
    float v_old = sq_unsigned[v_code_prev] * absmax_v_prev;

    // AdamW math (decoupled WD applied to p directly).
    float m_new = beta1 * m_old + (1.0f - beta1) * g;
    float v_new = beta2 * v_old + (1.0f - beta2) * g * g;
    float m_hat = m_new / bc1;
    float v_hat = v_new / bc2;
    float upd   = lr * m_hat / (sqrtf(v_hat) + eps);
    pv -= upd;
    if (wd != 0.0f) {
        pv -= lr * wd * pv;
    }
    if (active) {
        param[safe_idx] = pv;
    }

    // Stash new moments. Inactive lanes write 0 so they cannot drag absmax.
    s_m[tid] = active ? m_new : 0.0f;
    s_v[tid] = active ? v_new : 0.0f;
    __syncthreads();

    // Block-wide max-abs reduction (tree, in shared memory).
    // Reuse s_m / s_v as scratch — restore from registers afterward.
    float am = fabsf(s_m[tid]);
    float av = fabsf(s_v[tid]);
    __syncthreads();
    s_m[tid] = am;
    s_v[tid] = av;
    __syncthreads();
    _Pragma("unroll")
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) {
            float a = s_m[tid];
            float b = s_m[tid + off];
            s_m[tid] = a > b ? a : b;
            float c = s_v[tid];
            float d = s_v[tid + off];
            s_v[tid] = c > d ? c : d;
        }
        __syncthreads();
    }
    float absmax_m_new = s_m[0];
    float absmax_v_new = s_v[0];
    if (absmax_m_new == 0.0f) absmax_m_new = 1.0e-12f;
    if (absmax_v_new == 0.0f) absmax_v_new = 1.0e-12f;

    if (tid == 0) {
        m_absmax[blk] = absmax_m_new;
        v_absmax[blk] = absmax_v_new;
    }
    __syncthreads();

    // Repopulate s_m / s_v with the raw m_new / v_new (reduction clobbered
    // them with |.|). We just recompute from registers.
    s_m[tid] = active ? m_new : 0.0f;
    s_v[tid] = active ? v_new : 0.0f;
    __syncthreads();

    if (!active) return;

    // Requant: linear scan over the shared qmap, find argmin |qmap[c] - norm|.
    float m_norm = s_m[tid] / absmax_m_new;
    float v_norm = s_v[tid] / absmax_v_new;

    unsigned int best_m_code = 0u;
    float        best_m_dist = fabsf(sq_signed[0] - m_norm);
    unsigned int best_v_code = 0u;
    float        best_v_dist = fabsf(sq_unsigned[0] - v_norm);

    _Pragma("unroll 16")
    for (int c = 1; c < 256; ++c) {
        float dm = fabsf(sq_signed[c] - m_norm);
        if (dm < best_m_dist) { best_m_dist = dm; best_m_code = (unsigned int)c; }
        float dv = fabsf(sq_unsigned[c] - v_norm);
        if (dv < best_v_dist) { best_v_dist = dv; best_v_code = (unsigned int)c; }
    }

    m_codes[safe_idx] = (unsigned char)best_m_code;
    v_codes[safe_idx] = (unsigned char)best_v_code;
}

// BF16-grad variant: identical math, grad is upcast to F32 inside the kernel.
extern "C" __global__ void adam8bit_blockwise_bf16grad_kernel(
    float* __restrict__              param,
    const __nv_bfloat16* __restrict__ grad,
    unsigned char* __restrict__ m_codes,
    unsigned char* __restrict__ v_codes,
    float* __restrict__       m_absmax,
    float* __restrict__       v_absmax,
    const float* __restrict__ qmap_signed,
    const float* __restrict__ qmap_unsigned,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float wd,
    float bc1,
    float bc2,
    long long n
) {
    extern __shared__ float smem[];
    float* sq_signed   = smem;
    float* sq_unsigned = smem + 256;
    float* s_m         = smem + 512;
    float* s_v         = smem + 768;

    int tid = threadIdx.x;
    long long blk = blockIdx.x;
    long long base = blk * 256;

    sq_signed[tid]   = qmap_signed[tid];
    sq_unsigned[tid] = qmap_unsigned[tid];

    long long idx = base + (long long)tid;
    bool active = idx < n;
    long long safe_idx = active ? idx : base;

    __syncthreads();

    float g  = active ? __bfloat162float(grad[safe_idx]) : 0.0f;
    float pv = active ? param[safe_idx] : 0.0f;

    float absmax_m_prev = m_absmax[blk];
    float absmax_v_prev = v_absmax[blk];
    unsigned int m_code_prev = active ? (unsigned int)m_codes[safe_idx] : 0u;
    unsigned int v_code_prev = active ? (unsigned int)v_codes[safe_idx] : 0u;
    float m_old = sq_signed[m_code_prev]   * absmax_m_prev;
    float v_old = sq_unsigned[v_code_prev] * absmax_v_prev;

    float m_new = beta1 * m_old + (1.0f - beta1) * g;
    float v_new = beta2 * v_old + (1.0f - beta2) * g * g;
    float m_hat = m_new / bc1;
    float v_hat = v_new / bc2;
    float upd   = lr * m_hat / (sqrtf(v_hat) + eps);
    pv -= upd;
    if (wd != 0.0f) {
        pv -= lr * wd * pv;
    }
    if (active) {
        param[safe_idx] = pv;
    }

    s_m[tid] = active ? m_new : 0.0f;
    s_v[tid] = active ? v_new : 0.0f;
    __syncthreads();

    float am = fabsf(s_m[tid]);
    float av = fabsf(s_v[tid]);
    __syncthreads();
    s_m[tid] = am;
    s_v[tid] = av;
    __syncthreads();
    _Pragma("unroll")
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) {
            float a = s_m[tid];
            float b = s_m[tid + off];
            s_m[tid] = a > b ? a : b;
            float c = s_v[tid];
            float d = s_v[tid + off];
            s_v[tid] = c > d ? c : d;
        }
        __syncthreads();
    }
    float absmax_m_new = s_m[0];
    float absmax_v_new = s_v[0];
    if (absmax_m_new == 0.0f) absmax_m_new = 1.0e-12f;
    if (absmax_v_new == 0.0f) absmax_v_new = 1.0e-12f;

    if (tid == 0) {
        m_absmax[blk] = absmax_m_new;
        v_absmax[blk] = absmax_v_new;
    }
    __syncthreads();

    s_m[tid] = active ? m_new : 0.0f;
    s_v[tid] = active ? v_new : 0.0f;
    __syncthreads();

    if (!active) return;

    float m_norm = s_m[tid] / absmax_m_new;
    float v_norm = s_v[tid] / absmax_v_new;

    unsigned int best_m_code = 0u;
    float        best_m_dist = fabsf(sq_signed[0] - m_norm);
    unsigned int best_v_code = 0u;
    float        best_v_dist = fabsf(sq_unsigned[0] - v_norm);

    _Pragma("unroll 16")
    for (int c = 1; c < 256; ++c) {
        float dm = fabsf(sq_signed[c] - m_norm);
        if (dm < best_m_dist) { best_m_dist = dm; best_m_code = (unsigned int)c; }
        float dv = fabsf(sq_unsigned[c] - v_norm);
        if (dv < best_v_dist) { best_v_dist = dv; best_v_code = (unsigned int)c; }
    }

    m_codes[safe_idx] = (unsigned char)best_m_code;
    v_codes[safe_idx] = (unsigned char)best_v_code;
}
"#;

/// Block size used for both quantization and reduction. Matches bnb
/// `optimizer.py:478` (`blocksize = 256` for the blockwise 8-bit optimizer
/// state). Do not change without a matching change to the kernel and the
/// per-param `absmax` buffer sizing in the caller.
pub const ADAM8BIT_BLOCK_SIZE: usize = 256;

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
mod cuda_impl {
    use super::*;
    use cudarc::driver::{
        CudaDevice, CudaSlice, DevicePtr, DevicePtrMut, DeviceSlice, LaunchAsync, LaunchConfig,
    };
    use cudarc::nvrtc::CompileOptions;
    use std::sync::Arc;

    const MODULE_NAME: &str = "adam8bit_fused";
    const KERN_F32_GRAD: &str = "adam8bit_blockwise_kernel";
    const KERN_BF16_GRAD: &str = "adam8bit_blockwise_bf16grad_kernel";

    fn ensure_kernel(device: &Arc<CudaDevice>) -> Result<()> {
        if device.get_func(MODULE_NAME, KERN_F32_GRAD).is_some() {
            return Ok(());
        }
        let include_dir = std::env::var("CUDA_INCLUDE_DIR")
            .or_else(|_| std::env::var("CUDA_HOME").map(|home| format!("{home}/include")))
            .unwrap_or_else(|_| "/usr/local/cuda/include".into());
        let mut opts = CompileOptions::default();
        opts.include_paths.push(include_dir);
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(KERNEL_SRC, opts)
            .map_err(|e| Error::Cuda(format!("nvrtc adam8bit_fused: {:?}", e)))?;
        device
            .load_ptx(ptx, MODULE_NAME, &[KERN_F32_GRAD, KERN_BF16_GRAD])
            .map_err(|e| Error::Cuda(format!("load adam8bit_fused: {:?}", e)))?;
        Ok(())
    }

    /// Upload a host `[f32; 256]` LUT to device. Allocate once per device,
    /// share across all params + steps.
    pub fn upload_qmap(device: &Arc<CudaDevice>, qmap: &[f32; 256]) -> Result<CudaSlice<f32>> {
        device
            .htod_copy(qmap.to_vec())
            .map_err(|e| Error::Cuda(format!("upload qmap: {:?}", e)))
    }

    /// Allocate the per-parameter state for one tensor. Returns
    /// `(m_codes, v_codes, m_absmax, v_absmax)`, zero-initialized.
    /// Zero `m_codes` / `v_codes` index `qmap[0]` (most-negative for signed,
    /// 0.0 for unsigned). Multiplied by `m_absmax = v_absmax = 0`, the
    /// initial dequant is `0` regardless of LUT — matches bnb's
    /// torch.zeros initial state (`optimizer.py:471-483`).
    pub fn alloc_state(
        device: &Arc<CudaDevice>,
        numel: usize,
    ) -> Result<(CudaSlice<u8>, CudaSlice<u8>, CudaSlice<f32>, CudaSlice<f32>)> {
        let blocks = numel.div_ceil(ADAM8BIT_BLOCK_SIZE);
        let m_codes = device
            .alloc_zeros::<u8>(numel)
            .map_err(|e| Error::Cuda(format!("alloc m_codes: {:?}", e)))?;
        let v_codes = device
            .alloc_zeros::<u8>(numel)
            .map_err(|e| Error::Cuda(format!("alloc v_codes: {:?}", e)))?;
        let m_absmax = device
            .alloc_zeros::<f32>(blocks)
            .map_err(|e| Error::Cuda(format!("alloc m_absmax: {:?}", e)))?;
        let v_absmax = device
            .alloc_zeros::<f32>(blocks)
            .map_err(|e| Error::Cuda(format!("alloc v_absmax: {:?}", e)))?;
        Ok((m_codes, v_codes, m_absmax, v_absmax))
    }

    /// Fused dequant + AdamW step + requant for one parameter tensor.
    ///
    /// `param` MUST be F32. `grad` may be F32 or BF16 (auto-routed to the
    /// matching kernel). State buffers are mutated in place. `bc1` / `bc2`
    /// are host-computed `1 - beta^t`.
    ///
    /// All tensors must live on the same `CudaDevice`. The qmaps are not
    /// dtype-checked but are expected to be `create_dynamic_map(true)` and
    /// `create_dynamic_map(false)` respectively, uploaded once at optimizer
    /// construction.
    #[allow(clippy::too_many_arguments)]
    pub fn adam8bit_step_bnb(
        param: &mut Tensor,
        grad: &Tensor,
        m_codes: &mut CudaSlice<u8>,
        v_codes: &mut CudaSlice<u8>,
        m_absmax: &mut CudaSlice<f32>,
        v_absmax: &mut CudaSlice<f32>,
        qmap_signed: &CudaSlice<f32>,
        qmap_unsigned: &CudaSlice<f32>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
        bc1: f32,
        bc2: f32,
    ) -> Result<()> {
        if param.dtype() != DType::F32 {
            return Err(Error::InvalidOperation(format!(
                "adam8bit_step_bnb requires F32 param, got {:?}",
                param.dtype()
            )));
        }
        let n = param.shape().elem_count();
        if n != grad.shape().elem_count() {
            return Err(Error::InvalidOperation(format!(
                "adam8bit_step_bnb shape mismatch: param numel {} vs grad numel {}",
                n,
                grad.shape().elem_count()
            )));
        }
        if m_codes.len() < n || v_codes.len() < n {
            return Err(Error::InvalidOperation(format!(
                "adam8bit_step_bnb codes buffer too small: param numel {}, m_codes {}, v_codes {}",
                n,
                m_codes.len(),
                v_codes.len()
            )));
        }
        let blocks = n.div_ceil(ADAM8BIT_BLOCK_SIZE);
        if m_absmax.len() < blocks || v_absmax.len() < blocks {
            return Err(Error::InvalidOperation(format!(
                "adam8bit_step_bnb absmax buffer too small: need {} blocks, m_absmax {}, v_absmax {}",
                blocks,
                m_absmax.len(),
                v_absmax.len()
            )));
        }

        let device = param.device.clone();
        ensure_kernel(&device)?;

        let grad_is_bf16 = match grad.dtype() {
            DType::F32 => false,
            DType::BF16 => true,
            other => {
                return Err(Error::InvalidOperation(format!(
                    "adam8bit_step_bnb requires F32 or BF16 grad, got {:?}",
                    other
                )));
            }
        };

        let kernel_name = if grad_is_bf16 { KERN_BF16_GRAD } else { KERN_F32_GRAD };
        let f = device
            .get_func(MODULE_NAME, kernel_name)
            .ok_or_else(|| Error::Cuda(format!("missing kernel: {kernel_name}")))?;

        let param_ptr: u64 = {
            let s = param.as_mut_slice_f32("adam8bit:p")?;
            *s.device_ptr_mut()
        };
        let grad_ptr: u64 = if grad_is_bf16 {
            grad.as_device_ptr_bf16("adam8bit:g")? as u64
        } else {
            let s = grad.as_slice_f32("adam8bit:g")?;
            *s.device_ptr()
        };
        let m_codes_ptr: u64 = *m_codes.device_ptr_mut();
        let v_codes_ptr: u64 = *v_codes.device_ptr_mut();
        let m_absmax_ptr: u64 = *m_absmax.device_ptr_mut();
        let v_absmax_ptr: u64 = *v_absmax.device_ptr_mut();
        let qmap_s_ptr: u64 = *qmap_signed.device_ptr();
        let qmap_u_ptr: u64 = *qmap_unsigned.device_ptr();

        let n_i64 = n as i64;
        let grid = blocks as u32;
        // 4096 bytes shared: 2x qmap[256] + s_m[256] + s_v[256], each f32.
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (ADAM8BIT_BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 4 * 256 * 4,
        };

        let mut params: Vec<*mut std::ffi::c_void> = Vec::with_capacity(16);
        params.push(&param_ptr as *const u64 as *mut std::ffi::c_void);
        params.push(&grad_ptr as *const u64 as *mut std::ffi::c_void);
        params.push(&m_codes_ptr as *const u64 as *mut std::ffi::c_void);
        params.push(&v_codes_ptr as *const u64 as *mut std::ffi::c_void);
        params.push(&m_absmax_ptr as *const u64 as *mut std::ffi::c_void);
        params.push(&v_absmax_ptr as *const u64 as *mut std::ffi::c_void);
        params.push(&qmap_s_ptr as *const u64 as *mut std::ffi::c_void);
        params.push(&qmap_u_ptr as *const u64 as *mut std::ffi::c_void);
        params.push(&lr as *const f32 as *mut std::ffi::c_void);
        params.push(&beta1 as *const f32 as *mut std::ffi::c_void);
        params.push(&beta2 as *const f32 as *mut std::ffi::c_void);
        params.push(&eps as *const f32 as *mut std::ffi::c_void);
        params.push(&wd as *const f32 as *mut std::ffi::c_void);
        params.push(&bc1 as *const f32 as *mut std::ffi::c_void);
        params.push(&bc2 as *const f32 as *mut std::ffi::c_void);
        params.push(&n_i64 as *const i64 as *mut std::ffi::c_void);

        unsafe {
            f.launch(cfg, &mut params)
                .map_err(|e| Error::Cuda(format!("adam8bit_fused launch: {e:?}")))?;
        }

        // Autograd v2 prereq: bump version on in-place mutation. Only param
        // is a tracked Tensor — the U8 / F32 state slices are raw CudaSlices
        // owned by the optimizer and never enter the autograd graph.
        param.storage_ref().bump_version();
        Ok(())
    }
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub use cuda_impl::{adam8bit_step_bnb, alloc_state, upload_qmap};

// ---------------------------------------------------------------------------
// Tests (host-only — exercise create_dynamic_map)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `create_dynamic_map` returns a sorted 256-entry LUT with 0.0 present
    /// and 1.0 as the maximum. Sanity-check the structural invariants bnb
    /// asserts (sort + len = 256 + contains 0 + max = 1.0).
    #[test]
    fn dynamic_map_invariants() {
        for &signed in &[true, false] {
            let q = create_dynamic_map(signed);
            assert_eq!(q.len(), 256);
            // Sorted ascending.
            for w in q.windows(2) {
                assert!(w[0] <= w[1], "qmap signed={signed} not sorted at {:?}", w);
            }
            // 0.0 must appear (bnb appends 0 before sort).
            assert!(q.iter().any(|&v| v == 0.0));
            // Maximum is 1.0.
            assert!((q[255] - 1.0).abs() < 1e-7, "qmap max != 1.0: {}", q[255]);
            // Signed: minimum is -1.0 (mirrored from the positive side via the
            // for-each `data += (-(...))` in the algorithm).
            if signed {
                assert!((q[0] - (-1.0)).abs() < 1e-7, "signed qmap min != -1.0: {}", q[0]);
            } else {
                // Unsigned: minimum is 0.0 (no negative entries).
                assert!(q[0] >= 0.0, "unsigned qmap has negative entry: {}", q[0]);
            }
        }
    }

    /// Signed dynamic map: symmetric about zero, contains both +1 and -1.
    /// Mirrors the structural property bnb's algorithm guarantees by
    /// appending each magnitude twice with opposite signs.
    #[test]
    fn signed_dynamic_map_symmetric() {
        let q = create_dynamic_map(true);
        // For every nonzero entry there should be its negation (within fp eps).
        for &v in q.iter() {
            if v == 0.0 || v == 1.0 || v == -1.0 {
                continue;
            }
            let found = q.iter().any(|&u| (u + v).abs() < 1e-7);
            assert!(found, "no symmetric pair for {v}");
        }
    }
}
