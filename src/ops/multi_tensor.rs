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

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
const MT_L2_NORM_MODULE: &str = "multi_tensor_l2_norm";

#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
fn ensure_l2_norm_kernels(device: &Arc<CudaDevice>) -> Result<()> {
    if device
        .get_func(MT_L2_NORM_MODULE, "multi_tensor_l2_norm_sq_stage1_f32_kernel")
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
