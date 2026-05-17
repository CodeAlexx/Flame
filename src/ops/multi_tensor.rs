//! Multi-tensor (foreach-style) primitives for collapsing per-parameter
//! launch storms into single kernel launches.
//!
//! Phase 3 of the launch-storm refactor (plan in /home/alex/.claude/plans/
//! splendid-spinning-rabin.md). The Adam multi-tensor pattern already exists
//! in `flame_core::adam::adam_fused_multi_tensor_step` — this module
//! generalizes the packed-buffer + grid-per-tensor pattern to other ops
//! that fire per-parameter in the trainer hot path.
//!
//! ## Current entries
//!
//! - `multi_tensor_l2_norm_sq` — sum of squares across a list of tensors,
//!   returned as a 1-element F32 device scalar. Used by
//!   `flame_core::ops::grad_norm::global_l2_norm`.
//! - `multi_tensor_scale_inplace_packed` — in-place `x[i] *= scale` across
//!   a list of F32 or BF16 tensors. Collapses an N-launch per-parameter
//!   `mul_scalar` loop into a single grid-per-tensor kernel launch. Default
//!   off (`FLAME_MT_SCALE=1` enables in trainer call sites). See
//!   `EriDiffusion-v2/HANDOFF_2026-05-12_PHASE2_SCALE_FOLLOWUP.md`.
//!
//! ## Packed-buffer layout
//!
//! Each function builds its own packed pointer layout because the regions
//! differ (Adam has 5 regions: params/grads/m/v/sizes; L2 norm has 2:
//! ptrs/sizes). The pattern is the same: one H2D copy per call, the kernel
//! reads pointers from device memory via a single `MultiTensorMetaCache`
//! buffer (lives on the caller).

use crate::{Error, Result, Shape, Tensor};
use std::sync::Arc;

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DevicePtrMut, LaunchAsync, LaunchConfig};

/// Cache of the device-resident metadata buffer used by multi-tensor
/// kernels in this module. Identical pattern to
/// `flame_core::adam::MultiTensorMetaCache` but lives separately so the
/// L2-norm and Adam paths can grow their own caches independently.
///
/// Capacity grows monotonically with the tensor-list length. The buffer is
/// re-allocated when `n` changes — for LoRA training where the gradient
/// count is fixed per step, this is a one-time alloc on step 0.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub struct MultiTensorMetaCache {
    /// Device buffer holding pointers + sizes for the current call.
    buf: Option<CudaSlice<u64>>,
    /// Buffer length in u64 slots (NOT n_tensors — the caller controls the
    /// layout).
    capacity_slots: usize,
    /// Device buffer for per-tensor partial-sum outputs (1 f32 per tensor).
    /// Reused across calls to avoid per-step alloc/free churn.
    partials: Option<CudaSlice<f32>>,
    partials_capacity: usize,
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
impl MultiTensorMetaCache {
    pub const fn new() -> Self {
        Self {
            buf: None,
            capacity_slots: 0,
            partials: None,
            partials_capacity: 0,
        }
    }

    fn ensure_meta(&mut self, dev: &Arc<CudaDevice>, slots: usize) -> Result<()> {
        if self.capacity_slots == slots && self.buf.is_some() {
            return Ok(());
        }
        let new_buf = dev
            .alloc_zeros::<u64>(slots)
            .map_err(|e| Error::Cuda(format!("multi_tensor meta alloc: {e:?}")))?;
        self.buf = Some(new_buf);
        self.capacity_slots = slots;
        Ok(())
    }

    fn ensure_partials(&mut self, dev: &Arc<CudaDevice>, n: usize) -> Result<()> {
        if self.partials_capacity == n && self.partials.is_some() {
            return Ok(());
        }
        let new_buf = dev
            .alloc_zeros::<f32>(n)
            .map_err(|e| Error::Cuda(format!("multi_tensor partials alloc: {e:?}")))?;
        self.partials = Some(new_buf);
        self.partials_capacity = n;
        Ok(())
    }
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
impl Default for MultiTensorMetaCache {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CUDA kernels (NVRTC, compiled on first use).
// ---------------------------------------------------------------------------

/// Kernel sources for the L2-norm reduction primitives.
///
/// Reference: NVIDIA Apex `csrc/multi_tensor_l2norm_kernel.cu` (BSD-3). Apex
/// chunks tensors via constant-memory metadata and uses a fixed block count
/// per tensor for SM occupancy. We use the simpler "1 block per tensor +
/// grid-stride inner loop" pattern that adam_fused_multi_tensor_step uses,
/// trading some peak-SM utilization for a smaller, easier-to-verify kernel.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
const CUDA_MT_L2_NORM_F32: &str = r#"
extern "C" __global__ void multi_tensor_l2_norm_sq_stage1_f32_kernel(
    const float**    const __restrict__ tensors,
    const long long* __restrict__ sizes,
    float*           __restrict__ partials,   // length = n_tensors
    int n_tensors
) {
    int t = blockIdx.x;
    if (t >= n_tensors) return;

    long long n   = sizes[t];
    const float* x = tensors[t];

    int   tid    = threadIdx.x;
    int   stride = blockDim.x;
    float acc    = 0.0f;

    for (long long i = (long long)tid; i < n; i += (long long)stride) {
        float v = x[i];
        acc += v * v;
    }

    // Block-wide sum reduction in shared memory.
    __shared__ float sm[256];
    sm[tid] = acc;
    __syncthreads();

    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (tid < off) sm[tid] += sm[tid + off];
        __syncthreads();
    }

    if (tid == 0) partials[t] = sm[0];
}

// Stage 2: single-block reduction of `partials[n_tensors]` → `out[0]`.
// `n_tensors` can exceed 256; the loop strides until exhausted.
extern "C" __global__ void multi_tensor_l2_norm_sq_stage2_kernel(
    const float* __restrict__ partials,
    float*       __restrict__ out,
    int n_tensors
) {
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int i = tid; i < n_tensors; i += blockDim.x) {
        acc += partials[i];
    }

    __shared__ float sm[256];
    sm[tid] = acc;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (tid < off) sm[tid] += sm[tid + off];
        __syncthreads();
    }
    if (tid == 0) out[0] = sm[0];
}
"#;

/// Stage-1 kernel source for the BF16-input variant of the multi-tensor
/// L2-norm reduction. The stage-2 reducer is reused from the F32 module
/// (it works on F32 partials regardless of stage-1's input dtype).
///
/// Phase 4a addition: keeps BF16 gradients BF16 end-to-end through
/// gradient clipping under autograd v2's Option A (see
/// `docs/BF16_GRAD_DECISION.md`). The kernel reads BF16 storage, casts
/// to F32 inside the inner loop (`opmath_t`), accumulates in F32, and
/// writes an F32 partial per tensor — same numerical contract as the
/// F32 path.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
const CUDA_MT_L2_NORM_BF16: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void multi_tensor_l2_norm_sq_stage1_bf16_kernel(
    const __nv_bfloat16** const __restrict__ tensors,
    const long long*      __restrict__ sizes,
    float*                __restrict__ partials,   // length = n_tensors
    int n_tensors
) {
    int t = blockIdx.x;
    if (t >= n_tensors) return;

    long long n           = sizes[t];
    const __nv_bfloat16* x = tensors[t];

    int   tid    = threadIdx.x;
    int   stride = blockDim.x;
    float acc    = 0.0f;

    // F32 accumulation (opmath_t) — bit-equivalent fold ordering to the
    // F32 kernel above, just with a `__bfloat162float` widening at load.
    for (long long i = (long long)tid; i < n; i += (long long)stride) {
        float v = __bfloat162float(x[i]);
        acc += v * v;
    }

    __shared__ float sm[256];
    sm[tid] = acc;
    __syncthreads();

    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (tid < off) sm[tid] += sm[tid + off];
        __syncthreads();
    }

    if (tid == 0) partials[t] = sm[0];
}
"#;

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
const MT_L2_NORM_MODULE: &str = "multi_tensor_l2_norm";
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
const MT_L2_NORM_BF16_MODULE: &str = "multi_tensor_l2_norm_bf16";

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn ensure_l2_norm_kernels(device: &Arc<CudaDevice>) -> Result<()> {
    if device
        .get_func(
            MT_L2_NORM_MODULE,
            "multi_tensor_l2_norm_sq_stage1_f32_kernel",
        )
        .is_some()
    {
        return Ok(());
    }
    let ptx = cudarc::nvrtc::compile_ptx(CUDA_MT_L2_NORM_F32)
        .map_err(|e| Error::Cuda(format!("nvrtc multi_tensor_l2_norm: {:?}", e)))?;
    device
        .load_ptx(
            ptx,
            MT_L2_NORM_MODULE,
            &[
                "multi_tensor_l2_norm_sq_stage1_f32_kernel",
                "multi_tensor_l2_norm_sq_stage2_kernel",
            ],
        )
        .map_err(|e| Error::Cuda(format!("load multi_tensor_l2_norm: {:?}", e)))?;
    Ok(())
}

/// Load the BF16 stage-1 kernel under its own module. The stage-2 kernel
/// (F32 partials → F32 sum) is loaded by [`ensure_l2_norm_kernels`] in the
/// same call site since both BF16 and F32 callers need it.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn ensure_l2_norm_bf16_kernel(device: &Arc<CudaDevice>) -> Result<()> {
    if device
        .get_func(
            MT_L2_NORM_BF16_MODULE,
            "multi_tensor_l2_norm_sq_stage1_bf16_kernel",
        )
        .is_some()
    {
        return Ok(());
    }
    // BF16 stage-1 source needs `cuda_bf16.h` from the CUDA toolkit.
    // Match the include-path resolution pattern used by
    // `ensure_scale_kernel_bf16` below.
    let include_dir = std::env::var("CUDA_INCLUDE_DIR")
        .or_else(|_| std::env::var("CUDA_HOME").map(|home| format!("{home}/include")))
        .unwrap_or_else(|_| "/usr/local/cuda/include".into());
    let mut opts = cudarc::nvrtc::CompileOptions::default();
    opts.include_paths.push(include_dir);
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(CUDA_MT_L2_NORM_BF16, opts)
        .map_err(|e| Error::Cuda(format!("nvrtc multi_tensor_l2_norm bf16: {:?}", e)))?;
    device
        .load_ptx(
            ptx,
            MT_L2_NORM_BF16_MODULE,
            &["multi_tensor_l2_norm_sq_stage1_bf16_kernel"],
        )
        .map_err(|e| Error::Cuda(format!("load multi_tensor_l2_norm bf16: {:?}", e)))?;
    Ok(())
}

/// Sum of squares across a list of F32 tensors, returned as a 1-element
/// F32 device tensor.
///
/// **Dispatch constraints (current implementation):**
/// - All tensors in `grads` must be F32 logical & F32 storage.
/// - At least one tensor (empty slice returns Err — the caller should
///   short-circuit before calling).
/// - Tensors must be contiguous (no strided views).
///
/// Launch count: 2 kernel launches per call (stage 1 across all tensors,
/// stage 2 reduction). Compare to the per-tensor fold pattern in
/// `flame_core::ops::grad_norm::global_l2_norm`, which fires 2N+(N-1)+1
/// launches (square+sum per tensor, fold-add, sqrt).
///
/// **Numerical contract:** F32 accumulator throughout, block-wise tree
/// reduction. Bit-exact against single-tensor sum-of-squares is NOT
/// guaranteed: parallel-tree reduction sums in a different order than the
/// serial fold (`a + b + c + …`). Drift is bounded by F32 ULP × tree
/// depth and verified by `tests/multi_tensor_l2_norm_parity.rs` to be
/// well under 1e-5 relative for production gradient magnitudes.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn multi_tensor_l2_norm_sq_f32(
    cache: &mut MultiTensorMetaCache,
    grads: &[&Tensor],
) -> Result<Tensor> {
    use crate::DType;

    if grads.is_empty() {
        return Err(Error::InvalidInput(
            "multi_tensor_l2_norm_sq_f32: empty grads slice".into(),
        ));
    }
    let n = grads.len();
    let device = grads[0].device().clone();
    ensure_l2_norm_kernels(&device)?;

    for (i, g) in grads.iter().enumerate() {
        if g.dtype() != DType::F32 {
            return Err(Error::InvalidInput(format!(
                "multi_tensor_l2_norm_sq_f32: grads[{i}] is {:?}, expected F32",
                g.dtype()
            )));
        }
        if !g.is_contiguous() {
            return Err(Error::InvalidInput(format!(
                "multi_tensor_l2_norm_sq_f32: grads[{i}] is non-contiguous; \
                 callers must `.contiguous()` first"
            )));
        }
    }

    // Pack [ptrs(n) | sizes(n)] = 2n u64 entries.
    cache.ensure_meta(&device, 2 * n)?;
    cache.ensure_partials(&device, n)?;

    let mut packed: Vec<u64> = Vec::with_capacity(2 * n);
    for g in grads {
        let s = g.as_slice_f32("mt_l2:g")?;
        packed.push(*s.device_ptr());
    }
    for g in grads {
        packed.push(g.shape().elem_count() as u64);
    }

    let dev_buf = cache
        .buf
        .as_mut()
        .expect("ensure_meta post-condition: buf is Some");
    device
        .htod_sync_copy_into(&packed, dev_buf)
        .map_err(|e| Error::Cuda(format!("mt_l2 h2d: {e:?}")))?;

    let base = *dev_buf.device_ptr();
    let stride = (n * std::mem::size_of::<u64>()) as u64;
    let ptrs_arr_ptr = base;
    let sizes_arr_ptr = base + stride;

    let partials = cache
        .partials
        .as_mut()
        .expect("ensure_partials post-condition: partials is Some");
    let partials_ptr = *partials.device_ptr_mut();

    // Stage 1: per-tensor sum-of-squares → partials[n].
    let stage1 = device
        .get_func(
            MT_L2_NORM_MODULE,
            "multi_tensor_l2_norm_sq_stage1_f32_kernel",
        )
        .ok_or_else(|| Error::Cuda("missing kernel: stage1".into()))?;

    let n_i32 = n as i32;
    let stage1_cfg = LaunchConfig {
        grid_dim: (n as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut s1_params: Vec<*mut std::ffi::c_void> = Vec::with_capacity(4);
    s1_params.push(&ptrs_arr_ptr as *const u64 as *mut std::ffi::c_void);
    s1_params.push(&sizes_arr_ptr as *const u64 as *mut std::ffi::c_void);
    s1_params.push(&partials_ptr as *const u64 as *mut std::ffi::c_void);
    s1_params.push(&n_i32 as *const i32 as *mut std::ffi::c_void);

    unsafe {
        stage1
            .launch(stage1_cfg, &mut s1_params)
            .map_err(|e| Error::Cuda(format!("mt_l2 stage1 launch: {e:?}")))?;
    }

    // Stage 2: reduce partials[n] → out[1] (single-block).
    let mut out = Tensor::from_vec(vec![0.0_f32; 1], Shape::from_dims(&[1]), device.clone())?;
    let out_ptr = {
        let s = out.as_mut_slice_f32("mt_l2:out")?;
        *s.device_ptr_mut()
    };

    let stage2 = device
        .get_func(MT_L2_NORM_MODULE, "multi_tensor_l2_norm_sq_stage2_kernel")
        .ok_or_else(|| Error::Cuda("missing kernel: stage2".into()))?;

    let stage2_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut s2_params: Vec<*mut std::ffi::c_void> = Vec::with_capacity(3);
    s2_params.push(&partials_ptr as *const u64 as *mut std::ffi::c_void);
    s2_params.push(&out_ptr as *const u64 as *mut std::ffi::c_void);
    s2_params.push(&n_i32 as *const i32 as *mut std::ffi::c_void);

    unsafe {
        stage2
            .launch(stage2_cfg, &mut s2_params)
            .map_err(|e| Error::Cuda(format!("mt_l2 stage2 launch: {e:?}")))?;
    }

    Ok(out)
}

/// BF16 sibling of [`multi_tensor_l2_norm_sq_f32`].
///
/// Sum of squares across a list of BF16 tensors, returned as a 1-element
/// **F32** device tensor (partials are F32; the BF16 input is widened to
/// F32 inside the kernel — F32 only as `opmath_t`, never written back to
/// BF16).
///
/// Phase 4a addition. Used by [`crate::ops::grad_norm::global_l2_norm`]
/// when every grad in the slice is BF16-contiguous — keeps BF16 grads
/// BF16 through gradient clipping (no `to_dtype(F32)` per-tensor detour).
///
/// **Dispatch constraints:**
/// - All tensors must be BF16 logical & BF16 storage.
/// - Tensors must be contiguous.
/// - Non-empty slice (caller short-circuits empty).
///
/// **Numerical contract:** identical algebra to the F32 path (same
/// stage-1 block-wise reduction, same stage-2 reducer). Drift vs. a
/// reference `F32(grad).square().sum()` per-tensor fold is bounded by
/// the BF16→F32 widening (1 ULP at the input) plus the parallel-tree
/// reduction ULP × tree depth, the same parity bound as the F32 path.
///
/// **Launch count:** 2 kernel launches (stage 1 across all tensors,
/// stage 2 reduction). Replaces the legacy per-tensor pattern in
/// `global_l2_norm` which fired 2N + (N-1) + 1 launches.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn multi_tensor_l2_norm_sq_bf16(
    cache: &mut MultiTensorMetaCache,
    grads: &[&Tensor],
) -> Result<Tensor> {
    use crate::DType;

    if grads.is_empty() {
        return Err(Error::InvalidInput(
            "multi_tensor_l2_norm_sq_bf16: empty grads slice".into(),
        ));
    }
    let n = grads.len();
    let device = grads[0].device().clone();
    // Stage-2 reuses the F32 module's reducer kernel — ensure both
    // modules are loaded.
    ensure_l2_norm_kernels(&device)?;
    ensure_l2_norm_bf16_kernel(&device)?;

    for (i, g) in grads.iter().enumerate() {
        if g.dtype() != DType::BF16 {
            return Err(Error::InvalidInput(format!(
                "multi_tensor_l2_norm_sq_bf16: grads[{i}] is {:?}, expected BF16",
                g.dtype()
            )));
        }
        if !g.is_contiguous() {
            return Err(Error::InvalidInput(format!(
                "multi_tensor_l2_norm_sq_bf16: grads[{i}] is non-contiguous; \
                 callers must `.contiguous()` first"
            )));
        }
    }

    // Pack [ptrs(n) | sizes(n)] = 2n u64 entries. Same layout as the F32
    // path; only the pointer dtype changes (BF16 storage pointer).
    cache.ensure_meta(&device, 2 * n)?;
    cache.ensure_partials(&device, n)?;

    let mut packed: Vec<u64> = Vec::with_capacity(2 * n);
    for g in grads {
        let p = g.as_device_ptr_bf16("mt_l2_bf16:g")? as u64;
        packed.push(p);
    }
    for g in grads {
        packed.push(g.shape().elem_count() as u64);
    }

    let dev_buf = cache
        .buf
        .as_mut()
        .expect("ensure_meta post-condition: buf is Some");
    device
        .htod_sync_copy_into(&packed, dev_buf)
        .map_err(|e| Error::Cuda(format!("mt_l2_bf16 h2d: {e:?}")))?;

    let base = *dev_buf.device_ptr();
    let stride = (n * std::mem::size_of::<u64>()) as u64;
    let ptrs_arr_ptr = base;
    let sizes_arr_ptr = base + stride;

    let partials = cache
        .partials
        .as_mut()
        .expect("ensure_partials post-condition: partials is Some");
    let partials_ptr = *partials.device_ptr_mut();

    // Stage 1: per-tensor BF16 sum-of-squares → F32 partials[n].
    let stage1 = device
        .get_func(
            MT_L2_NORM_BF16_MODULE,
            "multi_tensor_l2_norm_sq_stage1_bf16_kernel",
        )
        .ok_or_else(|| Error::Cuda("missing kernel: stage1 bf16".into()))?;

    let n_i32 = n as i32;
    let stage1_cfg = LaunchConfig {
        grid_dim: (n as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut s1_params: Vec<*mut std::ffi::c_void> = Vec::with_capacity(4);
    s1_params.push(&ptrs_arr_ptr as *const u64 as *mut std::ffi::c_void);
    s1_params.push(&sizes_arr_ptr as *const u64 as *mut std::ffi::c_void);
    s1_params.push(&partials_ptr as *const u64 as *mut std::ffi::c_void);
    s1_params.push(&n_i32 as *const i32 as *mut std::ffi::c_void);

    unsafe {
        stage1
            .launch(stage1_cfg, &mut s1_params)
            .map_err(|e| Error::Cuda(format!("mt_l2_bf16 stage1 launch: {e:?}")))?;
    }

    // Stage 2: reduce partials[n] → out[1] (single-block) — reused
    // from the F32 module.
    let mut out = Tensor::from_vec(vec![0.0_f32; 1], Shape::from_dims(&[1]), device.clone())?;
    let out_ptr = {
        let s = out.as_mut_slice_f32("mt_l2_bf16:out")?;
        *s.device_ptr_mut()
    };

    let stage2 = device
        .get_func(MT_L2_NORM_MODULE, "multi_tensor_l2_norm_sq_stage2_kernel")
        .ok_or_else(|| Error::Cuda("missing kernel: stage2".into()))?;

    let stage2_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut s2_params: Vec<*mut std::ffi::c_void> = Vec::with_capacity(3);
    s2_params.push(&partials_ptr as *const u64 as *mut std::ffi::c_void);
    s2_params.push(&out_ptr as *const u64 as *mut std::ffi::c_void);
    s2_params.push(&n_i32 as *const i32 as *mut std::ffi::c_void);

    unsafe {
        stage2
            .launch(stage2_cfg, &mut s2_params)
            .map_err(|e| Error::Cuda(format!("mt_l2_bf16 stage2 launch: {e:?}")))?;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Multi-tensor in-place scale (`x[i] *= scale`).
//
// Targets the trainer hot path in `train_zimage.rs:880-888` and
// `train_klein.rs:1213-1226`: when `total_norm > CLIP_GRAD_NORM`, each
// parameter's gradient is multiplied by the clip-scale via a per-parameter
// `mul_scalar` kernel launch (280+ launches per step on zimage rank=8 LoRA).
//
// This module collapses that into a single launch (grid = n_tensors, one
// block per tensor + grid-stride inner loop). Bit-identical math: same
// `x[i] * scale`, same scalar, same element order. Multiplication is a
// pointwise op — no reduction-order drift.
//
// **Dispatch policy:** the trainer guards the call with `scale < 1.0` (no-op
// when the clip path doesn't fire) and an `FLAME_MT_SCALE` env gate. The
// kernel itself accepts any scale, including 1.0; the no-op guard is a
// caller responsibility.
// ---------------------------------------------------------------------------

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
const CUDA_MT_SCALE_F32: &str = r#"
extern "C" __global__ void multi_tensor_scale_inplace_f32_kernel(
    float**          const __restrict__ tensors,
    const long long* __restrict__ sizes,
    int   n_tensors,
    float scale
) {
    int t = blockIdx.x;
    if (t >= n_tensors) return;

    long long n = sizes[t];
    float* x    = tensors[t];

    int tid    = threadIdx.x;
    int stride = blockDim.x;

    for (long long i = (long long)tid; i < n; i += (long long)stride) {
        x[i] = x[i] * scale;
    }
}
"#;

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
const CUDA_MT_SCALE_BF16: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void multi_tensor_scale_inplace_bf16_kernel(
    __nv_bfloat16**  const __restrict__ tensors,
    const long long* __restrict__ sizes,
    int   n_tensors,
    float scale
) {
    int t = blockIdx.x;
    if (t >= n_tensors) return;

    long long       n = sizes[t];
    __nv_bfloat16* x  = tensors[t];

    int tid    = threadIdx.x;
    int stride = blockDim.x;

    for (long long i = (long long)tid; i < n; i += (long long)stride) {
        float v = __bfloat162float(x[i]);
        x[i]    = __float2bfloat16(v * scale);
    }
}
"#;

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
const MT_SCALE_MODULE: &str = "multi_tensor_scale";

/// Compile-on-first-use loaders. F32 has no header dependencies and uses
/// the plain `compile_ptx` path. BF16 requires `cuda_bf16.h` from the CUDA
/// toolkit, fetched via the `CUDA_INCLUDE_DIR` / `CUDA_HOME` env vars
/// (matching `bf16_elementwise::ensure_and_get`). Splitting the dispatch
/// means F32 callers never pay the BF16 toolkit lookup.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn ensure_scale_kernel_f32(device: &Arc<CudaDevice>) -> Result<()> {
    if device
        .get_func(MT_SCALE_MODULE, "multi_tensor_scale_inplace_f32_kernel")
        .is_some()
    {
        return Ok(());
    }
    let ptx = cudarc::nvrtc::compile_ptx(CUDA_MT_SCALE_F32)
        .map_err(|e| Error::Cuda(format!("nvrtc multi_tensor_scale f32: {:?}", e)))?;
    device
        .load_ptx(
            ptx,
            MT_SCALE_MODULE,
            &["multi_tensor_scale_inplace_f32_kernel"],
        )
        .map_err(|e| Error::Cuda(format!("load multi_tensor_scale f32: {:?}", e)))?;
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn ensure_scale_kernel_bf16(device: &Arc<CudaDevice>) -> Result<()> {
    if device
        .get_func(MT_SCALE_MODULE, "multi_tensor_scale_inplace_bf16_kernel")
        .is_some()
    {
        return Ok(());
    }
    let include_dir = std::env::var("CUDA_INCLUDE_DIR")
        .or_else(|_| std::env::var("CUDA_HOME").map(|home| format!("{home}/include")))
        .unwrap_or_else(|_| "/usr/local/cuda/include".into());
    let mut opts = cudarc::nvrtc::CompileOptions::default();
    opts.include_paths.push(include_dir);
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(CUDA_MT_SCALE_BF16, opts)
        .map_err(|e| Error::Cuda(format!("nvrtc multi_tensor_scale bf16: {:?}", e)))?;
    device
        .load_ptx(
            ptx,
            MT_SCALE_MODULE,
            &["multi_tensor_scale_inplace_bf16_kernel"],
        )
        .map_err(|e| Error::Cuda(format!("load multi_tensor_scale bf16: {:?}", e)))?;
    Ok(())
}

/// In-place `x[i] *= scale` across a packed list of device tensors.
///
/// **Packed buffer layout** (2 regions, 2n u64 entries total, written by
/// the caller before calling this function):
/// ```text
///   packed[0..n]      = device pointers (u64) to each tensor's first element
///   packed[n..2n]     = element counts (u64) for each tensor
/// ```
///
/// **Dtype dispatch:** the caller asserts homogeneous dtype across the list
/// and passes `is_bf16` to select the kernel. Pointers in `packed[0..n]`
/// must be valid for the selected dtype (BF16: each pointer addresses a
/// `__nv_bfloat16` array; F32: each pointer addresses a `float` array).
/// Mixing dtypes within a single call is undefined.
///
/// **Numerical contract:** bit-identical to a per-tensor `mul_scalar`
/// kernel for F32 (same `x * scale` math, same iteration order). BF16 path
/// does `__float2bfloat16(__bfloat162float(x) * scale)` — same as
/// flame-core's per-tensor BF16 `mul_scalar`. Parity to per-tensor BF16
/// `mul_scalar` is within 1 ULP and verified by
/// `tests/multi_tensor_scale_parity.rs`.
///
/// **Memory:** no allocations beyond the cache's metadata buffer (reused
/// across calls when `n` is stable). Tensors are modified in place; their
/// device pointers must remain valid for the kernel's lifetime.
///
/// **Autograd v2 prereq**: this function only sees packed raw device
/// pointers — it has no `Tensor` handles to bump. Callers that pack
/// trainable tensor storages into `packed` MUST call
/// `tensor.storage_ref().bump_version()` (or equivalent) on each tensor
/// after the kernel returns. See `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md`
/// §5.
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub fn multi_tensor_scale_inplace_packed(
    cache: &mut MultiTensorMetaCache,
    device: &Arc<CudaDevice>,
    n: usize,
    packed: &[u64],
    scale: f32,
    is_bf16: bool,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    if packed.len() != 2 * n {
        return Err(Error::InvalidInput(format!(
            "multi_tensor_scale_inplace_packed: packed.len() = {}, expected 2*n = {}",
            packed.len(),
            2 * n
        )));
    }
    if is_bf16 {
        ensure_scale_kernel_bf16(device)?;
    } else {
        ensure_scale_kernel_f32(device)?;
    }
    cache.ensure_meta(device, 2 * n)?;

    let dev_buf = cache
        .buf
        .as_mut()
        .expect("ensure_meta post-condition: buf is Some");
    device
        .htod_sync_copy_into(packed, dev_buf)
        .map_err(|e| Error::Cuda(format!("mt_scale h2d: {e:?}")))?;

    let base = *dev_buf.device_ptr();
    let stride = (n * std::mem::size_of::<u64>()) as u64;
    let ptrs_arr_ptr = base;
    let sizes_arr_ptr = base + stride;

    let kernel_name = if is_bf16 {
        "multi_tensor_scale_inplace_bf16_kernel"
    } else {
        "multi_tensor_scale_inplace_f32_kernel"
    };
    let func = device
        .get_func(MT_SCALE_MODULE, kernel_name)
        .ok_or_else(|| Error::Cuda(format!("missing kernel: {kernel_name}")))?;

    let cfg = LaunchConfig {
        grid_dim: (n as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i32 = n as i32;
    let mut params: Vec<*mut std::ffi::c_void> = Vec::with_capacity(4);
    params.push(&ptrs_arr_ptr as *const u64 as *mut std::ffi::c_void);
    params.push(&sizes_arr_ptr as *const u64 as *mut std::ffi::c_void);
    params.push(&n_i32 as *const i32 as *mut std::ffi::c_void);
    params.push(&scale as *const f32 as *mut std::ffi::c_void);

    unsafe {
        func.launch(cfg, &mut params)
            .map_err(|e| Error::Cuda(format!("mt_scale launch: {e:?}")))?;
    }

    Ok(())
}
