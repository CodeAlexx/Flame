pub mod contracts;

use crate::autograd::{AutogradContext, Op};
#[cfg(feature = "cuda")]
use crate::bf16_ops;
use crate::config::default_dtype;
use crate::cuda_memory_alignment::alloc_aligned_f32;
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::cuda_ops_bf16;
use crate::cuda_ops_ffi::CudaStream;
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::device::CudaStreamRawPtrExt;
use crate::gradient::{GradientMap, TensorGradExt};
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::staging::bf16_copy_async;
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
use crate::staging::ArenaLease;
use crate::tensor_ext::to_owning_fp32_strong;
use crate::tensor_storage::{ensure_unique_slice, slice_ref, wrap_slice, TensorStorage};
use crate::{DType, Error, Result, Shape};
use cudarc::driver::{
    CudaDevice, CudaSlice, DevicePtr, DevicePtrMut, DeviceSlice, LaunchAsync, LaunchConfig,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
// Fix cublas op imports to stable sys enums
use crate::cuda_kernels::CudaKernels;
use crate::cuda_ops::GpuOps;
use crate::rng;
use crate::tensor::contracts::assert_nhwc_bf16_public;
use std::fmt;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[cfg(feature = "bf16_u16")]
use half::bf16;

// Helper macro for kernel launches
macro_rules! launch_kernel {
    ($func:expr, $cfg:expr, $($args:expr),* $(,)?) => {{
        unsafe { $func.launch($cfg, ($($args,)*)) }
    }};
}

// ---------------------------------------------------------------------------
// Cached env-var flags (syscall-free hot-path reads)
// ---------------------------------------------------------------------------
//
// Every `std::env::var(...)` is a syscall (stat the environ table + copy the
// value to a `String`). flame-core used to read `ALLOC_LOG`, `FLAME_TRACE_DTYPE`,
// `FLAME_DTYPE_TRACE`, and `SDXL_DEBUG_SHAPES` on EVERY allocation / matmul /
// dtype cast / narrow, which is thousands of syscalls per denoise step even
// when nothing is set. These wrappers cache the reads once per process via
// `OnceLock` so the hot path only pays an atomic load.

#[inline]
fn env_flag_enabled(var: &'static str, cache: &'static std::sync::OnceLock<bool>) -> bool {
    *cache.get_or_init(|| std::env::var(var).ok().as_deref() == Some("1"))
}

#[inline]
fn alloc_log_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    env_flag_enabled("ALLOC_LOG", &CACHED)
}

#[inline]
fn trace_dtype_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    env_flag_enabled("FLAME_TRACE_DTYPE", &CACHED)
}

#[inline]
fn dtype_trace_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    env_flag_enabled("FLAME_DTYPE_TRACE", &CACHED)
}

#[inline]
fn sdxl_debug_shapes_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    env_flag_enabled("SDXL_DEBUG_SHAPES", &CACHED)
}

/// Global tensor ID counter
static TENSOR_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Unique tensor identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub usize);

impl TensorId {
    pub fn new() -> Self {
        TensorId(TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for TensorId {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to broadcast two shape arrays
#[allow(dead_code)]
fn broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> Result<Vec<usize>> {
    let max_len = shape1.len().max(shape2.len());
    let mut result = vec![1; max_len];

    // Right-align shapes
    let offset1 = max_len - shape1.len();
    let offset2 = max_len - shape2.len();

    for i in 0..max_len {
        let dim1 = if i >= offset1 { shape1[i - offset1] } else { 1 };
        let dim2 = if i >= offset2 { shape2[i - offset2] } else { 1 };

        if dim1 == dim2 {
            result[i] = dim1;
        } else if dim1 == 1 {
            result[i] = dim2;
        } else if dim2 == 1 {
            result[i] = dim1;
        } else {
            return Err(Error::InvalidOperation(format!(
                "Cannot broadcast dimensions {} and {}",
                dim1, dim2
            )));
        }
    }

    Ok(result)
}

/// The core tensor type with GPU-accelerated operations
#[derive(Clone)]
pub struct Tensor {
    /// GPU memory storage with dtype support
    pub(crate) storage: TensorStorage,

    /// Shape of this tensor
    pub(crate) shape: Shape,

    /// Device this tensor lives on
    pub(crate) device: Arc<CudaDevice>,

    /// Unique identifier for gradient tracking
    pub(crate) id: TensorId,

    /// Whether gradients should be computed
    pub(crate) requires_grad: bool,

    /// Custom strides. `None` = row-major contiguous (default).
    /// Phase-1 of the stride refactor: field added, every constructor
    /// initializes to `None`; behavior unchanged. Phase 2 wires view
    /// ops (permute/transpose/narrow/chunk/view) to populate it
    /// without materializing storage.
    ///
    /// Uses `Strides` (SmallVec<[usize;6]>) so views never heap-allocate
    /// to carry their stride metadata. See shape::Strides.
    pub(crate) custom_strides: Option<crate::shape::Strides>,

    /// Element offset into the underlying storage. Non-zero only for
    /// narrow/chunk views. Default 0.
    pub(crate) view_offset: usize,
}

impl AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Tensor {
        self
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor {{ shape: {:?}, dtype: {:?}, device: cuda:{}, id: {} }}",
            self.shape,
            self.storage.dtype(),
            self.device.ordinal(),
            self.id.0
        )
    }
}

// Note: We don't implement the standard Clone trait because we have a custom clone() method
// that returns Result<Tensor> for consistency with other operations

impl Tensor {
    pub(crate) fn storage_ref(&self) -> &TensorStorage {
        &self.storage
    }

    pub(crate) fn storage_mut(&mut self) -> &mut TensorStorage {
        &mut self.storage
    }

    // Mixed-precision policy (reminder):
    // - Parameters and activations: store as BF16; do math in FP32 inside kernels.
    // - Intermediates/temps/activations created here (zeros/new outputs): prefer BF16 storage directly.
    // - Masks: prefer Bool/u8, not F32 (convert to FP32 only at the last moment before math).
    // - Gradients/optimizer states/scalars/reductions: use FP32 for numerical stability.
    // - RNG/init: generate FP32 in registers inside kernels, write BF16 elements directly; avoid full FP32 tensors.
    /// Create a causal mask tensor
    pub fn causal_mask(seq_len: usize, device: &Arc<CudaDevice>) -> Result<Self> {
        // Create a lower triangular mask
        let mut mask_data = vec![0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                mask_data[i * seq_len + j] = 1.0;
            }
        }

        // Convert to tensor
        let shape = Shape::from_dims(&[seq_len, seq_len]);
        Self::from_vec(mask_data, shape, device.clone())
    }

    /// Apply a mask to tensor (set masked positions to value)
    pub fn masked_fill(&self, mask: &Tensor, value: f32) -> Result<Self> {
        if self.shape != mask.shape {
            return Err(Error::ShapeMismatch {
                expected: self.shape.clone(),
                got: mask.shape.clone(),
            });
        }

        // Use CUDA kernel for masked fill
        let kernel_code = r#"
extern "C" __global__ void masked_fill_kernel(
    const float* input,
    const float* mask,
    float* output,
    float value,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = mask[idx] > 0.5f ? value : input[idx];
    }
}
"#;

        crate::cuda_kernels::CudaKernels::ensure_kernel(
            &self.device,
            "masked_fill_kernel",
            kernel_code,
        )?;

        let f = self
            .device
            .get_func("masked_fill_kernel", "masked_fill_kernel")
            .ok_or_else(|| Error::Cuda("Failed to get masked_fill_kernel".into()))?;

        let n = self.shape.elem_count();
        let output_data = alloc_zeros_from_pool(&self.device, n)?;

        let cfg = cudarc::driver::LaunchConfig::for_num_elems(n as u32);

        crate::launch_kernel!(
            f,
            cfg,
            self.storage.try_as_slice_f32()?,
            mask.storage.try_as_slice_f32()?,
            &output_data,
            value,
            n as i32
        );

        // Preserve input dtype for output when feasible (F32/BF16); otherwise fall back to F32
        let storage = match self.dtype() {
            DType::BF16 => {
                #[cfg(not(feature = "bf16_u16"))]
                {
                    TensorStorage::BF16 {
                        data: output_data,
                        numel: n,
                    }
                }
                #[cfg(feature = "bf16_u16")]
                {
                    use cudarc::driver::DevicePtr;
                    let mut bf = unsafe { self.device.alloc::<u16>(n) }
                        .map_err(|e| Error::Cuda(format!("alloc masked_fill bf16: {:?}", e)))?;
                    crate::bf16_convert::f32_to_bf16_u16(
                        self.device.clone(),
                        &output_data,
                        *bf.device_ptr(),
                        n,
                    )?;
                    TensorStorage::BF16 {
                        data: bf.into(),
                        numel: n,
                    }
                }
            }
            DType::F32 => TensorStorage::F32 {
                data: output_data.into(),
                numel: n,
            },
            _ => TensorStorage::F32 {
                data: output_data.into(),
                numel: n,
            },
        };
        Ok(Self {
            storage,
            shape: self.shape.clone(),
            device: self.device.clone(),
            id: TensorId::new(),
            requires_grad: false,
            custom_strides: None,
            view_offset: 0,

        })
    }

    /// Create a new tensor filled with zeros (defaults to global default dtype)
    pub fn zeros(shape: Shape, device: Arc<CudaDevice>) -> Result<Self> {
        let dtype = default_dtype();
        let numel = shape.elem_count();
        if alloc_log_enabled() {
            let bytes = numel * dtype.size_in_bytes();
            if bytes >= (8 << 20) {
                eprintln!(
                    "[alloc] tag=zeros dtype={:?} shape={:?} bytes={}",
                    dtype,
                    shape.dims(),
                    bytes
                );
            }
        }
        Self::zeros_dtype(shape, dtype, device)
    }

    /// Create tensor with specific dtype
    pub fn zeros_dtype(shape: Shape, dtype: DType, device: Arc<CudaDevice>) -> Result<Self> {
        if alloc_log_enabled() {
            let bytes = shape.elem_count() * dtype.size_in_bytes();
            if bytes >= (8 << 20) {
                eprintln!(
                    "[alloc] tag=zeros_dtype dtype={:?} shape={:?} bytes={}",
                    dtype,
                    shape.dims(),
                    bytes
                );
            }
        }
        let storage = TensorStorage::zeros(&shape, dtype, &device)?;
        Ok(Self {
            storage,
            shape,
            device,
            id: TensorId::new(),
            requires_grad: false,
            custom_strides: None,
            view_offset: 0,

        })
    }

    /// Allocate an output tensor without zeroing. Safe ONLY when the
    /// caller immediately writes every element (e.g. as a cuBLASLt GEMM
    /// output or fused kernel output). The saved `memset_zeros` pass is
    /// ~2 ms per GB of BF16 activations at HBM bandwidth — small per
    /// call but adds up across a full VAE or DiT forward.
    pub fn empty_dtype(shape: Shape, dtype: DType, device: Arc<CudaDevice>) -> Result<Self> {
        if alloc_log_enabled() {
            let bytes = shape.elem_count() * dtype.size_in_bytes();
            if bytes >= (8 << 20) {
                eprintln!(
                    "[alloc] tag=empty_dtype dtype={:?} shape={:?} bytes={}",
                    dtype,
                    shape.dims(),
                    bytes
                );
            }
        }
        let storage = TensorStorage::empty(&shape, dtype, &device)?;
        Ok(Self {
            storage,
            shape,
            device,
            id: TensorId::new(),
            requires_grad: false,
            custom_strides: None,
            view_offset: 0,

        })
    }

    /// Update the tensor's logical shape without touching storage. Element count must match.
    pub fn reshape_inplace(&mut self, dims: &[usize]) -> Result<()> {
        let new_shape = Shape::from_dims(dims);
        if new_shape.elem_count() != self.shape.elem_count() {
            return Err(Error::ShapeMismatch {
                expected: new_shape,
                got: self.shape.clone(),
            });
        }
        self.shape = new_shape;
        Ok(())
    }

    /// Return a non-owning view that shares the underlying storage.
    ///
    /// IMPORTANT: preserves `custom_strides` and `view_offset`. Earlier
    /// versions zeroed these, which silently broke save-for-backward of
    /// strided permute views: `tensor.alias()` of a `[B,H,N,HD]` permute
    /// of a `[B,N,H,HD]` buffer would label itself as logical-shape
    /// contiguous with no strides, so a downstream backward kernel that
    /// did `is_contiguous()` returned true on the alias and read raw
    /// physical memory in the pre-permute order while interpreting it
    /// as post-permute shape — gradients land in wrong cells. Caught by
    /// `parity_klein_attn_chain_prod_diag` showing dscale cos_sim ~-0.1
    /// against PyTorch reference.
    #[inline]
    pub fn alias(&self) -> Tensor {
        Tensor {
            storage: self.storage.clone(),
            shape: self.shape.clone(),
            device: self.device.clone(),
            id: TensorId::new(),
            requires_grad: self.requires_grad,
            custom_strides: self.custom_strides.clone(),
            view_offset: self.view_offset,
        }
    }

    /// Create a zeros tensor that matches the receiver's shape/device but uses an explicit dtype.
    #[inline]
    pub fn zeros_like_with_dtype(&self, dtype: DType) -> Result<Self> {
        Self::zeros_dtype(self.shape.clone(), dtype, self.device.clone())
    }

    /// BF16 matmul helper for legacy callers.
    #[cfg(not(feature = "bf16_u16"))]
    pub fn matmul_bf16(&self, other: &Tensor) -> Result<Tensor> {
        let mut output = crate::ops::gemm::launch_gemm(self, other)?;
        if self.requires_grad || other.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::MatMul {
                        lhs: self.id,
                        rhs: other.id,
                    },
                    vec![
                        (self.id, self.clone()),
                        (other.id, other.clone()),
                    ],
                );
            }
        }
        Ok(output)
    }

    /// BF16 matmul helper for true BF16 storage.
    #[cfg(feature = "bf16_u16")]
    pub fn matmul_bf16(&self, other: &Tensor) -> Result<Tensor> {
        let mut output = crate::ops::gemm::launch_gemm(self, other)?;
        if self.requires_grad || other.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::MatMul {
                        lhs: self.id,
                        rhs: other.id,
                    },
                    vec![
                        (self.id, self.clone()),
                        (other.id, other.clone()),
                    ],
                );
            }
        }
        Ok(output)
    }

    /// Get dtype
    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    /// Get data as F32 (converting if necessary)
    /// This is for backward compatibility - prefer using storage directly
    pub fn data(&self) -> Result<Arc<CudaSlice<f32>>> {
        match &self.storage {
            TensorStorage::F32 { data, .. } => {
                #[cfg(feature = "shared_storage")]
                {
                    Ok(data.clone())
                }
                #[cfg(not(feature = "shared_storage"))]
                {
                    Ok(Arc::new(data.clone()))
                }
            }
            _ => {
                let f32_data = self.storage.to_f32(&self.device)?;
                Ok(Arc::new(f32_data))
            }
        }
    }

    /// Get raw CUDA pointer for cuDNN operations (read-only)
    pub fn cuda_ptr(&self) -> *const f32 {
        use cudarc::driver::DevicePtr;
        use std::ffi::c_void;
        match self.storage.try_as_slice_f32() {
            Ok(slice) => {
                let ptr_addr = *slice.device_ptr();
                ptr_addr as *const c_void as *const f32
            }
            Err(_) => std::ptr::null(),
        }
    }

    /// Get mutable raw CUDA pointer for cuDNN operations
    pub fn cuda_ptr_mut(&mut self) -> *mut f32 {
        use cudarc::driver::DevicePtr;
        use std::ffi::c_void;
        // We need to get a mutable reference to the storage
        // This is safe because we're the only owner of this tensor
        let ptr_addr = match &mut self.storage {
            TensorStorage::F32 { data, .. } => *data.device_ptr(),
            TensorStorage::F16 { data, .. } => *data.device_ptr(),
            #[cfg(not(feature = "bf16_u16"))]
            TensorStorage::BF16 { data, .. } => *data.device_ptr(),
            #[cfg(feature = "bf16_u16")]
            TensorStorage::BF16 { .. } => return std::ptr::null_mut(),
            #[cfg(feature = "bf16_u16")]
            TensorStorage::BF16Arena { .. } => return std::ptr::null_mut(),
            #[cfg(feature = "bf16_u16")]
            TensorStorage::BF16View { .. } => return std::ptr::null_mut(),
            TensorStorage::I32 { data, .. } => *data.device_ptr(),
            TensorStorage::Bool { data, .. } => *data.device_ptr(),
            TensorStorage::I8 { .. } => {
                return std::ptr::null_mut();
            }
        };
        // Cast u64 GPU address to pointer
        ptr_addr as *mut c_void as *mut f32
    }

    /// Returns the raw BF16 device pointer if BF16(u16) storage is enabled.
    ///
    /// Hot path: called 3x per elementwise launch. `dtype()` and
    /// `storage_dtype()` both return `self.storage.dtype()`, so the previous
    /// runtime double-check was pure overhead. It is now a `debug_assert!`
    /// that compiles out in release. The `BF16 { data }` arm is listed first
    /// so branch prediction favors the common case.
    #[cfg(feature = "bf16_u16")]
    #[inline]
    pub fn as_device_ptr_bf16(&self, tag: &str) -> Result<*const u16> {
        debug_assert_eq!(
            self.dtype(),
            DType::BF16,
            "[{tag}] as_device_ptr_bf16 expected BF16 tensor, got {:?}",
            self.dtype()
        );
        match self.storage_ref() {
            TensorStorage::BF16 { data, .. } => Ok((*data.device_ptr()) as *const u16),
            TensorStorage::BF16Arena { ptr, .. } => Ok(ptr.as_ptr()),
            TensorStorage::BF16View { ptr, .. } => Ok(ptr.as_ptr()),
            _ => Err(Error::InvalidOperation(format!(
                "[{tag}] expected BF16(u16) backing storage, got logical {:?} / storage {:?}",
                self.dtype(),
                self.storage_dtype()
            ))),
        }
    }

    /// Returns the read-only device slice for F32 tensors.
    pub fn as_slice_f32(&self, tag: &str) -> Result<&CudaSlice<f32>> {
        if self.dtype() != DType::F32 || self.storage_dtype() != DType::F32 {
            return Err(Error::InvalidOperation(format!(
                "[{tag}] expected F32 tensor, got logical {:?} / storage {:?}",
                self.dtype(),
                self.storage_dtype()
            )));
        }
        self.storage
            .try_as_slice_f32()
            .map_err(|_| Error::InvalidOperation(format!("[{tag}] expected F32 backing storage")))
    }

    /// Returns the writable device slice for F32 tensors.
    pub fn as_mut_slice_f32(&mut self, tag: &str) -> Result<&mut CudaSlice<f32>> {
        if self.dtype() != DType::F32 || self.storage_dtype() != DType::F32 {
            return Err(Error::InvalidOperation(format!(
                "[{tag}] expected F32 tensor, got logical {:?} / storage {:?}",
                self.dtype(),
                self.storage_dtype()
            )));
        }
        self.storage_mut()
            .try_as_mut_slice_f32()
            .map_err(|_| Error::InvalidOperation(format!("[{tag}] expected F32 backing storage")))
    }

    /// Copy full tensor contents from another F32 tensor of identical shape.
    pub fn copy_f32_from(&mut self, src: &Tensor) -> Result<()> {
        if self.shape != src.shape {
            return Err(Error::ShapeMismatch {
                expected: self.shape.clone(),
                got: src.shape.clone(),
            });
        }
        if self.dtype() != DType::F32 || self.storage_dtype() != DType::F32 {
            return Err(Error::InvalidOperation(
                "copy_f32_from destination must be F32".into(),
            ));
        }
        if src.dtype() != DType::F32 || src.storage_dtype() != DType::F32 {
            return Err(Error::InvalidOperation(
                "copy_f32_from source must be F32".into(),
            ));
        }

        let device = self.device.clone();
        let dst_slice = self.as_mut_slice_f32("copy_f32_from.dst")?;
        let src_slice = src.as_slice_f32("copy_f32_from.src")?;
        device
            .dtod_copy(src_slice, dst_slice)
            .map_err(|e| Error::Cuda(format!("copy_f32_from failed: {e:?}")))
    }

    /// Returns the mutable BF16 device pointer if BF16(u16) storage is enabled.
    ///
    /// Hot path: see `as_device_ptr_bf16` for rationale.
    #[cfg(feature = "bf16_u16")]
    #[inline]
    pub fn as_mut_device_ptr_bf16(&mut self, tag: &str) -> Result<*mut u16> {
        debug_assert_eq!(
            self.dtype(),
            DType::BF16,
            "[{tag}] as_mut_device_ptr_bf16 expected BF16 tensor, got {:?}",
            self.dtype()
        );
        match self.storage_mut() {
            TensorStorage::BF16 { ref mut data, .. } => {
                let data = ensure_unique_slice(data)?;
                Ok((*data.device_ptr_mut()) as *mut u16)
            }
            TensorStorage::BF16Arena { ptr, .. } => Ok(ptr.as_ptr()),
            TensorStorage::BF16View { ptr, .. } => Ok(ptr.as_ptr()),
            other => {
                let storage_dtype = other.dtype();
                Err(Error::InvalidOperation(format!(
                    "[{tag}] expected BF16(u16) backing storage, got storage {:?}",
                    storage_dtype
                )))
            }
        }
    }

    /// Copy a contiguous BF16 region (measured in BF16 elements) from `src` into `self`.
    /// Offsets and lengths are expressed in BF16 elements. Both tensors must be BF16 with BF16 backing storage.
    pub fn copy_bf16_region_from(
        &mut self,
        dst_offset_elems: usize,
        src: &Tensor,
        src_offset_elems: usize,
        elem_count: usize,
    ) -> Result<()> {
        if elem_count == 0 {
            return Ok(());
        }
        if self.dtype() != DType::BF16 || self.storage_dtype() != DType::BF16 {
            return Err(Error::InvalidOperation(
                "copy_bf16_region_from destination must be BF16".into(),
            ));
        }
        if src.dtype() != DType::BF16 || src.storage_dtype() != DType::BF16 {
            return Err(Error::InvalidOperation(
                "copy_bf16_region_from source must be BF16".into(),
            ));
        }

        let device = self.device.clone();
        let dst_slice = self
            .storage_mut()
            .try_as_mut_slice_u16()
            .map_err(|_| Error::InvalidOperation("destination storage is not BF16".into()))?;
        let src_slice = src
            .storage_ref()
            .try_as_slice_u16()
            .map_err(|_| Error::InvalidOperation("source storage is not BF16".into()))?;

        let dst_end = dst_offset_elems + elem_count;
        let src_end = src_offset_elems + elem_count;
        if dst_end > dst_slice.len() {
            return Err(Error::InvalidOperation(format!(
                "destination copy range {}..{} exceeds length {}",
                dst_offset_elems,
                dst_end,
                dst_slice.len()
            )));
        }
        if src_end > src_slice.len() {
            return Err(Error::InvalidOperation(format!(
                "source copy range {}..{} exceeds length {}",
                src_offset_elems,
                src_end,
                src_slice.len()
            )));
        }

        let mut dst_view = dst_slice.slice_mut(dst_offset_elems..dst_end);
        let src_view = src_slice.slice(src_offset_elems..src_end);
        device
            .dtod_copy(&src_view, &mut dst_view)
            .map_err(|e| Error::Cuda(format!("bf16 region copy failed: {e:?}")))?;
        Ok(())
    }

    /// BF16 pointer helpers are unavailable without the bf16_u16 feature.
    #[cfg(not(feature = "bf16_u16"))]
    #[allow(unused_variables)]
    pub fn as_device_ptr_bf16(&self, tag: &str) -> Result<*const u16> {
        Err(Error::InvalidOperation(
            "Tensor::as_device_ptr_bf16 requires the bf16_u16 feature".into(),
        ))
    }

    /// BF16 pointer helpers are unavailable without the bf16_u16 feature.
    #[cfg(not(feature = "bf16_u16"))]
    #[allow(unused_variables)]
    pub fn as_mut_device_ptr_bf16(&mut self, tag: &str) -> Result<*mut u16> {
        Err(Error::InvalidOperation(
            "Tensor::as_mut_device_ptr_bf16 requires the bf16_u16 feature".into(),
        ))
    }

    /// Cast to different dtype
    /// Cast dtype without recording on the autograd tape.
    /// Use for internal casts (gemm auto-cast, rms_norm input cast) where
    /// the gradient should flow through the original tensor, not through
    /// a Cast op that forces an extra backward allocation.
    pub fn to_dtype_no_grad(&self, dtype: DType) -> Result<Tensor> {
        if self.dtype() == dtype && self.storage.dtype() == dtype {
            return Ok(self.clone());
        }
        let _guard = AutogradContext::no_grad();
        self.to_dtype(dtype)
    }

    pub fn to_dtype(&self, dtype: DType) -> Result<Tensor> {
        if self.dtype() == dtype && self.storage.dtype() == dtype {
            return Ok(self.clone());
        }

        // Stride refactor Phase 2a safety net: to_dtype reads storage
        // linearly via `storage.to_f32`. A strided view would alias the
        // wrong memory.
        if !self.is_contiguous() {
            return self.contiguous()?.to_dtype(dtype);
        }

        // For now, we store everything as F32 but track the dtype
        // Use aligned allocation for the new tensor to avoid CUDA issues
        let numel = self.shape.elem_count();

        // Debug large allocations
        // Commented out for cleaner training output
        // if numel > 100000 {
        //     eprintln!("to_dtype: converting {} elements from {:?} to {:?}", numel, self.dtype(), dtype);
        // }

        // Use aligned allocation and convert via f32 staging
        let mut f32_data = alloc_aligned_f32(&self.device, numel)?;
        let src_f32 = self.storage.to_f32(&self.device)?;
        self.device.dtod_copy(&src_f32, &mut f32_data)?;

        let storage = match dtype {
            DType::F32 => TensorStorage::F32 {
                data: f32_data.into(),
                numel,
            },
            DType::F16 => TensorStorage::F16 {
                data: f32_data.into(),
                numel,
                scale: 1.0,
            },
            DType::BF16 => {
                #[cfg(not(feature = "bf16_u16"))]
                {
                    TensorStorage::BF16 {
                        data: f32_data,
                        numel,
                    }
                }
                #[cfg(feature = "bf16_u16")]
                {
                    use cudarc::driver::DevicePtr;
                    let mut bf_data = crate::cuda_alloc_pool::pool_alloc_u16(&self.device, numel)?;
                    crate::bf16_convert::f32_to_bf16_u16(
                        self.device.clone(),
                        &f32_data,
                        *bf_data.device_ptr(),
                        numel,
                    )?;
                    TensorStorage::BF16 {
                        data: bf_data.into(),
                        numel,
                    }
                }
            }
            DType::I32 => TensorStorage::I32 {
                data: f32_data.into(),
                numel,
            },
            DType::Bool => {
                Self::convert_f32_buffer_to_bool(&self.device, &mut f32_data, numel)?;
                TensorStorage::Bool {
                    data: f32_data.into(),
                    numel,
                }
            }
            _ => {
                return Err(Error::InvalidOperation(format!(
                    "Unsupported dtype: {:?}",
                    dtype
                )))
            }
        };

        let out = Tensor {
            storage,
            shape: self.shape.clone(),
            device: self.device.clone(),
            id: TensorId::new(),
            requires_grad: self.requires_grad,
            custom_strides: None,
            view_offset: 0,

        };
        if self.requires_grad && AutogradContext::is_recording() {
            // Record autograd Cast op so gradients flow across dtype boundaries
            AutogradContext::record_op(
                out.id,
                Op::Cast {
                    input: self.id,
                    from: self.dtype(),
                    to: dtype,
                },
                vec![(self.id, self.clone())],
            );
        }
        Ok(out)
    }

    /// Gather rows from a 2D table along axis 0 using I32 indices; supports ids with arbitrary leading dims
    pub fn index_select0(&self, ids: &Tensor) -> Result<Tensor> {
        if self.shape.dims().len() != 2 {
            return Err(Error::InvalidOperation(
                "index_select0 expects table [V,D]".into(),
            ));
        }
        if ids.dtype() != DType::I32 {
            return Err(Error::InvalidOperation(
                "index_select0 ids must be I32".into(),
            ));
        }
        let gathered = crate::cuda_kernels::gather_rows(self, ids, 0)?;

        if self.requires_grad && AutogradContext::is_recording() {
            let mut tracked = gathered.clone_result()?;
            tracked.requires_grad = true;
            AutogradContext::record_op(
                tracked.id,
                Op::IndexSelect {
                    input: self.id,
                    indices: ids.id,
                    dim: 0,
                },
                vec![
                    (self.id, self.clone()),
                    (ids.id, ids.clone()),
                ],
            );
            Ok(tracked)
        } else {
            Ok(gathered)
        }
    }

    // eq implemented in tensor_ops_extended.rs

    /// Where: out = mask ? a : b; mask Bool (0/1)
    pub fn where_mask(mask: &Tensor, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        crate::ops_ext::where_mask(mask, a, b)
    }

    /// Create a new tensor filled with ones (defaults to F32)
    pub fn ones(shape: Shape, device: Arc<CudaDevice>) -> Result<Self> {
        let size = shape.elem_count();
        let ones_vec = vec![1.0f32; size];
        Self::from_vec(ones_vec, shape, device)
    }

    /// Create a new tensor filled with ones with specific dtype
    pub fn ones_dtype(shape: Shape, dtype: DType, device: Arc<CudaDevice>) -> Result<Self> {
        let size = shape.elem_count();
        let ones_vec = vec![1.0f32; size];
        Self::from_vec_dtype(ones_vec, shape, device, dtype)
    }

    /// Create a new tensor from a Vec
    pub fn from_vec(data: Vec<f32>, shape: Shape, device: Arc<CudaDevice>) -> Result<Self> {
        if data.len() != shape.elem_count() {
            return Err(Error::ShapeMismatch {
                expected: shape.clone(),
                got: Shape::from_dims(&[data.len()]),
            });
        }
        // Allocate from memory pool
        let numel = data.len();
        let mut cuda_data = alloc_from_pool(&device, numel)?;

        // If the allocated size is larger than our data, we need to handle it carefully
        if cuda_data.len() > numel {
            // Pad the data to match the allocated size
            let mut padded_data = data;
            padded_data.resize(cuda_data.len(), 0.0);
            device
                .htod_copy_into(padded_data, &mut cuda_data)
                .map_err(|_| Error::CudaDriver)?;
        } else {
            // Normal case - sizes match
            device
                .htod_copy_into(data, &mut cuda_data)
                .map_err(|_| Error::CudaDriver)?;
        }
        Ok(Self {
            storage: TensorStorage::F32 {
                data: cuda_data.into(),
                numel,
            },
            shape,
            device,
            id: TensorId::new(),
            requires_grad: false,
            custom_strides: None,
            view_offset: 0,

        })
    }

    /// Create a new tensor from a Vec with specific dtype
    pub fn from_vec_dtype(
        data: Vec<f32>,
        shape: Shape,
        device: Arc<CudaDevice>,
        dtype: DType,
    ) -> Result<Self> {
        if data.len() != shape.elem_count() {
            return Err(Error::ShapeMismatch {
                expected: shape.clone(),
                got: Shape::from_dims(&[data.len()]),
            });
        }
        // Allocate from memory pool
        let numel = data.len();
        let mut cuda_data = alloc_from_pool(&device, numel)?;

        // If the allocated size is larger than our data, we need to handle it carefully
        if cuda_data.len() > numel {
            // Pad the data to match the allocated size
            let mut padded_data = data;
            padded_data.resize(cuda_data.len(), 0.0);
            device
                .htod_copy_into(padded_data, &mut cuda_data)
                .map_err(|_| Error::CudaDriver)?;
        } else {
            // Normal case - sizes match
            device
                .htod_copy_into(data, &mut cuda_data)
                .map_err(|_| Error::CudaDriver)?;
        }

        // Create storage with specified dtype
        let storage = match dtype {
            DType::F32 => TensorStorage::F32 {
                data: cuda_data.into(),
                numel,
            },
            DType::F16 => TensorStorage::F16 {
                data: cuda_data.into(),
                numel,
                scale: 1.0,
            },
            DType::BF16 => {
                #[cfg(not(feature = "bf16_u16"))]
                {
                    TensorStorage::BF16 {
                        data: cuda_data,
                        numel,
                    }
                }
                #[cfg(feature = "bf16_u16")]
                {
                    use cudarc::driver::DevicePtr;
                    let mut bf = unsafe { device.alloc::<u16>(numel) }
                        .map_err(|e| Error::Cuda(format!("alloc from_vec bf16: {:?}", e)))?;
                    crate::bf16_convert::f32_to_bf16_u16(
                        device.clone(),
                        &cuda_data,
                        *bf.device_ptr(),
                        numel,
                    )?;
                    TensorStorage::BF16 {
                        data: bf.into(),
                        numel,
                    }
                }
            }
            _ => {
                return Err(Error::InvalidOperation(format!(
                    "Unsupported dtype for from_vec_dtype: {:?}",
                    dtype
                )))
            }
        };

        Ok(Self {
            storage,
            shape,
            device,
            id: TensorId::new(),
            requires_grad: false,
            custom_strides: None,
            view_offset: 0,

        })
    }

    /// Create tensor from raw GPU data
    pub fn from_raw(
        data: Arc<CudaSlice<f32>>,
        shape: Shape,
        device: Arc<CudaDevice>,
        requires_grad: bool,
    ) -> Result<Self> {
        if data.len() != shape.elem_count() {
            return Err(Error::ShapeMismatch {
                expected: shape.clone(),
                got: Shape::from_dims(&[data.len()]),
            });
        }
        Ok(Self {
            storage: TensorStorage::F32 {
                data: (*data).clone().into(),
                numel: shape.elem_count(),
            },
            shape,
            device,
            id: TensorId::new(),
            requires_grad,
            custom_strides: None,
            view_offset: 0,

        })
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    pub(crate) fn from_bf16_arena(
        shape: Shape,
        device: Arc<CudaDevice>,
        ptr: NonNull<u16>,
        lease: ArenaLease,
    ) -> Result<Self> {
        let numel = shape.elem_count();
        if numel == 0 {
            return Err(Error::InvalidInput(
                "from_bf16_arena: zero-element tensors are not supported".into(),
            ));
        }
        Ok(Self {
            storage: TensorStorage::BF16Arena {
                ptr,
                numel,
                device: device.clone(),
                lease,
            },
            shape,
            device,
            id: TensorId::new(),
            requires_grad: false,
            custom_strides: None,
            view_offset: 0,

        })
    }

    /// Construct a BF16 tensor from a pre-populated GPU buffer.
    ///
    /// Takes ownership of the `CudaSlice<u16>`.  No copy — the slice IS
    /// the tensor's storage.  Used by flame-swap to wrap async-transferred
    /// block weights as tensors.
    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    pub fn from_bf16_slice_gpu(
        data: CudaSlice<u16>,
        shape: Shape,
        device: Arc<CudaDevice>,
    ) -> Self {
        let numel = shape.elem_count();
        debug_assert_eq!(
            data.len(),
            numel,
            "CudaSlice length {} != shape numel {}",
            data.len(),
            numel
        );
        Tensor {
            storage: TensorStorage::BF16 {
                data: wrap_slice(data),
                numel,
            },
            shape,
            device,
            id: TensorId::new(),
            requires_grad: false,
            custom_strides: None,
            view_offset: 0,

        }
    }

    /// Create a non-owning BF16 tensor that aliases existing GPU memory.
    ///
    /// Create a non-owning BF16 tensor that views into a shared GPU buffer.
    ///
    /// The returned tensor does NOT free the underlying memory on drop.
    /// The caller must guarantee that the backing buffer outlives all views
    /// created from it (e.g. the shared buffer lives for the entire forward
    /// pass, and views are dropped at block boundaries).
    ///
    /// # Safety
    /// - `ptr` must be a valid device pointer to `numel` BF16 elements.
    /// - The backing buffer must outlive this tensor.
    #[cfg(feature = "bf16_u16")]
    pub unsafe fn view_from_buffer(
        ptr: *mut u16,
        shape: Shape,
        device: Arc<CudaDevice>,
    ) -> Self {
        let numel = shape.elem_count();
        Tensor {
            storage: TensorStorage::BF16View {
                ptr: NonNull::new_unchecked(ptr),
                numel,
            },
            shape,
            device,
            id: TensorId::new(),
            requires_grad: false,
            custom_strides: None,
            view_offset: 0,

        }
    }

    /// Backwards-compatible alias for `view_from_buffer`.
    #[cfg(feature = "bf16_u16")]
    pub unsafe fn from_bf16_device_ptr_non_owning(
        ptr: u64,
        numel: usize,
        shape: Shape,
        device: Arc<CudaDevice>,
    ) -> Self {
        Self::view_from_buffer(ptr as *mut u16, shape, device)
    }

    /// Create random tensor
    pub fn randn(shape: Shape, mean: f32, std: f32, device: Arc<CudaDevice>) -> Result<Self> {
        let size = shape.elem_count();
        let cpu_data = rng::sample_normal(size, mean, std)?;
        let t = Self::from_vec(cpu_data, shape, device)?;
        let dd = default_dtype();
        if dd != DType::F32 {
            t.to_dtype(dd)
        } else {
            Ok(t)
        }
    }

    /// Create a random tensor whose samples are fully determined by `seed`.
    ///
    /// Unlike [`Tensor::randn`] (which consumes the global RNG stream set by
    /// [`crate::rng::set_seed`]), this uses a locally-seeded `StdRng` and
    /// CPU Box-Muller. Calling `randn_seeded(shape, mean, std, seed, device)`
    /// twice with identical arguments always yields bit-identical output —
    /// useful for element-wise parity against Python/torch references
    /// (LanPaint, diffusers, etc.) where reproducible noise is required.
    ///
    /// Output dtype matches [`Tensor::randn`]: F32 unless the workspace
    /// default dtype is otherwise.
    pub fn randn_seeded(
        shape: Shape,
        mean: f32,
        std: f32,
        seed: u64,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let size = shape.elem_count();
        let cpu_data = rng::sample_normal_seeded(size, mean, std, seed);
        let t = Self::from_vec(cpu_data, shape, device)?;
        let dd = default_dtype();
        if dd != DType::F32 {
            t.to_dtype(dd)
        } else {
            Ok(t)
        }
    }

    /// Create a tensor with random values like another tensor
    pub fn rand_like(tensor: &Tensor) -> Result<Self> {
        Self::randn(tensor.shape.clone(), 0.0, 1.0, tensor.device.clone())
    }

    /// Create a BF16 tensor from F32 data (stored as F32 internally)
    pub fn from_bf16_slice(
        data: CudaSlice<f32>,
        shape: Shape,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let numel = shape.elem_count();
        if data.len() != numel {
            return Err(Error::ShapeMismatch {
                expected: shape.clone(),
                got: Shape::from_dims(&[data.len()]),
            });
        }

        #[cfg(not(feature = "bf16_u16"))]
        {
            Ok(Self {
                storage: TensorStorage::BF16 { data, numel },
                shape,
                device,
                id: TensorId::new(),
                requires_grad: false,
                custom_strides: None,
                view_offset: 0,

            })
        }
        #[cfg(feature = "bf16_u16")]
        {
            use cudarc::driver::DevicePtr;
            let mut bf = unsafe { device.alloc::<u16>(numel) }
                .map_err(|e| Error::Cuda(format!("alloc from_bf16_slice: {:?}", e)))?;
            crate::bf16_convert::f32_to_bf16_u16(device.clone(), &data, *bf.device_ptr(), numel)?;
            Ok(Self {
                storage: TensorStorage::BF16 {
                    data: bf.into(),
                    numel,
                },
                shape,
                device,
                id: TensorId::new(),
                requires_grad: false,
                custom_strides: None,
                view_offset: 0,

            })
        }
    }

    /// Create a BF16 tensor from F32 data
    pub fn from_f32_to_bf16(data: Vec<f32>, shape: Shape, device: Arc<CudaDevice>) -> Result<Self> {
        use crate::bf16_support::BF16Ops;
        BF16Ops::from_f32(data, shape, device)
    }

    /// Get F32 slice for BF16 tensor (stored as F32 internally)
    pub fn as_bf16_slice(&self) -> Result<&CudaSlice<f32>> {
        if self.dtype() != DType::BF16 {
            return Err(Error::InvalidOperation("Not a BF16 tensor".to_string()));
        }
        #[cfg(not(feature = "bf16_u16"))]
        {
            self.storage.try_as_slice_f32()
        }
        #[cfg(feature = "bf16_u16")]
        {
            Err(Error::InvalidOperation(
                "BF16 storage uses u16 backing under bf16_u16".into(),
            ))
        }
    }

    /// Convert tensor to BF16
    pub fn to_bf16(&self) -> Result<Self> {
        if self.dtype() == DType::BF16 {
            return self.clone_result();
        }
        self.to_dtype(DType::BF16)
    }

    /// Create a BF16 tensor from CUDA slice (stored as F32)
    pub fn from_cuda_bf16(
        data: CudaSlice<f32>,
        shape: Shape,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        Self::from_bf16_slice(data, shape, device)
    }

    /// Enable gradient computation
    pub fn requires_grad_(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }

    /// Create tensor from slice
    pub fn from_slice(data: &[f32], shape: Shape, device: Arc<CudaDevice>) -> Result<Self> {
        if data.len() != shape.elem_count() {
            return Err(Error::ShapeMismatch {
                expected: shape.clone(),
                got: Shape::from_dims(&[data.len()]),
            });
        }
        Self::from_vec(data.to_vec(), shape, device)
    }

    pub fn from_slice_dtype(
        data: &[f32],
        shape: Shape,
        device: Arc<CudaDevice>,
        dtype: DType,
    ) -> Result<Self> {
        if data.len() != shape.elem_count() {
            return Err(Error::ShapeMismatch {
                expected: shape.clone(),
                got: Shape::from_dims(&[data.len()]),
            });
        }
        Self::from_vec_dtype(data.to_vec(), shape, device, dtype)
    }

    /// Compute gradients via automatic differentiation
    pub fn backward(&self) -> Result<GradientMap> {
        AutogradContext::backward(self)
    }

    /// Compute gradients with debug information
    pub fn backward_debug(&self) -> Result<GradientMap> {
        AutogradContext::backward_debug(self)
    }

    /// Set data from slice (useful for initialization)
    /// This creates a new tensor with the provided data
    pub fn set_data(&self, data: &[f32]) -> Result<Tensor> {
        if data.len() != self.shape.elem_count() {
            return Err(Error::ShapeMismatch {
                expected: self.shape.clone(),
                got: Shape::from_dims(&[data.len()]),
            });
        }
        // Create new tensor with the provided data
        Self::from_slice(data, self.shape.clone(), self.device.clone())
    }

    /// Functional weight update - returns new tensor
    pub fn update_weights(&self, gradient: &Tensor, lr: f32) -> Result<Tensor> {
        if self.shape != gradient.shape {
            return Err(Error::ShapeMismatch {
                expected: self.shape.clone(),
                got: gradient.shape.clone(),
            });
        }

        // w = w - lr * grad (functional style)
        let grad_scaled = gradient.mul_scalar(lr)?;
        self.sub(&grad_scaled)
    }

    /// Matrix multiplication routed through the shared GEMM backend.
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        if trace_dtype_enabled() {
            eprintln!(
                "[tensor.matmul] lhs dtype {:?} storage {:?} shape {:?}; rhs dtype {:?} storage {:?} shape {:?}",
                self.dtype(),
                self.storage_dtype(),
                self.shape().dims(),
                other.dtype(),
                other.storage_dtype(),
                other.shape().dims()
            );
        }
        let (lhs_ref, rhs_ref);
        let (lhs_owned, rhs_owned);
        let (lhs_tensor, rhs_tensor): (&Tensor, &Tensor) = if self.dtype() != other.dtype() {
            // Auto-cast: prefer BF16 (no autograd recording for the cast itself)
            let target = if self.dtype() == DType::BF16 || other.dtype() == DType::BF16 {
                DType::BF16
            } else {
                self.dtype()
            };
            if self.dtype() != target {
                lhs_owned = self.to_dtype_no_grad(target)?;
                lhs_ref = &lhs_owned;
            } else {
                lhs_ref = self;
            }
            if other.dtype() != target {
                rhs_owned = other.to_dtype_no_grad(target)?;
                rhs_ref = &rhs_owned;
            } else {
                rhs_ref = other;
            }
            (lhs_ref, rhs_ref)
        } else {
            (self, other)
        };

        let lhs_rank = lhs_tensor.shape().dims().len();
        let rhs_rank = rhs_tensor.shape().dims().len();

        let (mut output, lhs_for_grad, rhs_for_grad) = if lhs_rank <= 2 && rhs_rank <= 2 {
            let out = crate::ops::gemm::launch_gemm(lhs_tensor, rhs_tensor)?;
            (out, lhs_tensor.clone(), rhs_tensor.clone())
        } else if lhs_rank == 3 && rhs_rank == 2 {
            // [B, M, K] @ [K, N] → flatten to [B*M, K], gemm, reshape to [B, M, N]
            let ld = lhs_tensor.shape().dims();
            let (batch, m, k) = (ld[0], ld[1], ld[2]);
            let lhs_2d = lhs_tensor.reshape(&[batch * m, k])?;
            let out_2d = crate::ops::gemm::launch_gemm(&lhs_2d, rhs_tensor)?;
            let n = rhs_tensor.shape().dims()[1];
            let out = out_2d.reshape(&[batch, m, n])?;
            (out, lhs_tensor.clone(), rhs_tensor.clone())
        } else if lhs_rank == 3 && rhs_rank == 3 {
            // [B, M, K] @ [B, K, N] → batched GEMM
            let out = crate::ops::gemm::launch_bmm(lhs_tensor, rhs_tensor)?;
            (out, lhs_tensor.clone(), rhs_tensor.clone())
        } else {
            return Err(Error::InvalidInput(format!(
                "matmul: unsupported ranks lhs={lhs_rank}D rhs={rhs_rank}D (supported: 2D×2D, 3D×2D, 3D×3D)"
            )));
        };

        if self.requires_grad || other.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::MatMul {
                        lhs: lhs_for_grad.id,
                        rhs: rhs_for_grad.id,
                    },
                    vec![(lhs_for_grad.id, lhs_for_grad), (rhs_for_grad.id, rhs_for_grad)],
                );
            }
        }

        Ok(output)
    }

    /// Batch matrix multiplication
    /// For tensors with 3+ dimensions, applies matmul to the last 2 dimensions
    /// Broadcasting is supported for batch dimensions
    pub fn bmm(&self, other: &Tensor) -> Result<Tensor> {
        let self_shape = self.shape.dims();
        let other_shape = other.shape.dims();

        if self_shape.len() < 2 || other_shape.len() < 2 {
            return Err(Error::InvalidOperation(
                "bmm requires tensors with at least 2 dimensions".into(),
            ));
        }

        match (self_shape.len(), other_shape.len()) {
            (3, 3) => {
                let (batch, m, k1) = (self_shape[0], self_shape[1], self_shape[2]);
                let (batch2, k2, n) = (other_shape[0], other_shape[1], other_shape[2]);

                if batch != batch2 {
                    Err(Error::InvalidOperation(format!(
                        "bmm: batch size mismatch {} vs {}",
                        batch, batch2
                    )))
                } else if k1 != k2 {
                    Err(Error::InvalidOperation(format!(
                        "bmm: incompatible matrix dimensions {} vs {}",
                        k1, k2
                    )))
                } else {
                    self.bmm_3d(batch, m, k1, n, other)
                }
            }
            (4, 4) => {
                let (batch, heads, m, k1) =
                    (self_shape[0], self_shape[1], self_shape[2], self_shape[3]);
                let (batch2, heads2, k2, n) = (
                    other_shape[0],
                    other_shape[1],
                    other_shape[2],
                    other_shape[3],
                );

                if batch != batch2 || heads != heads2 {
                    Err(Error::InvalidOperation(format!(
                        "bmm: batch/heads mismatch: [{}, {}] vs [{}, {}]",
                        batch, heads, batch2, heads2
                    )))
                } else if k1 != k2 {
                    Err(Error::InvalidOperation(format!(
                        "bmm: incompatible matrix dimensions {} vs {}",
                        k1, k2
                    )))
                } else {
                    let total_batch = batch * heads;
                    let self_3d = self.reshape(&[total_batch, m, k1])?;
                    let other_3d = other.reshape(&[total_batch, k2, n])?;
                    let result_3d = self_3d.bmm_3d(total_batch, m, k1, n, &other_3d)?;
                    result_3d.reshape(&[batch, heads, m, n])
                }
            }
            _ => Err(Error::InvalidOperation(format!(
                "bmm: unsupported tensor shapes {:?} @ {:?}",
                self_shape, other_shape
            ))),
        }
    }

    /// Helper for 3D batch matrix multiplication
    fn bmm_3d(&self, batch: usize, m: usize, k: usize, n: usize, other: &Tensor) -> Result<Tensor> {
        let mut output = match self.dtype() {
            DType::F32 => {
                let mut out = Tensor::empty_dtype(
                    Shape::from_dims(&[batch, m, n]),
                    DType::F32,
                    self.device.clone(),
                )?;
                crate::ops::gemm::launch_gemm_strided_batched(self, other, &mut out)?;
                out
            }
            DType::BF16 => {
                let mut out = Tensor::empty_dtype(
                    Shape::from_dims(&[batch, m, n]),
                    DType::BF16,
                    self.device.clone(),
                )?;
                crate::ops::gemm_bf16::bmm_bf16_fp32acc_out(self, other, &mut out, false, false)?;
                out
            }
            _ => return self.bmm_3d_cpu(batch, m, k, n, other),
        };

        if self.requires_grad || other.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::BatchMatMul {
                        lhs: self.id,
                        rhs: other.id,
                    },
                    vec![
                        (self.id, self.clone()),
                        (other.id, other.clone()),
                    ],
                );
            }
        }

        Ok(output)
    }

    #[allow(clippy::manual_memcpy, clippy::needless_range_loop)]
    fn bmm_3d_cpu(
        &self,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        other: &Tensor,
    ) -> Result<Tensor> {
        // Prepare output shape [batch, m, n]
        let output_shape = vec![batch, m, n];
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        let mut output_data = vec![0.0f32; batch * m * n];

        for b in 0..batch {
            let self_offset = b * m * k;
            let other_offset = b * k * n;
            let output_offset = b * m * n;

            let self_slice = self_data[self_offset..self_offset + m * k].to_vec();
            let other_slice = other_data[other_offset..other_offset + k * n].to_vec();

            let self_batch =
                Tensor::from_vec(self_slice, Shape::from_dims(&[m, k]), self.device.clone())?;

            let other_batch =
                Tensor::from_vec(other_slice, Shape::from_dims(&[k, n]), self.device.clone())?;

            let batch_result = self_batch.matmul(&other_batch)?;
            let batch_result_data = batch_result.to_vec()?;
            output_data[output_offset..output_offset + m * n].copy_from_slice(&batch_result_data);
        }

        let mut output = Tensor::from_vec(
            output_data,
            Shape::from_dims(&output_shape),
            self.device.clone(),
        )?;

        // AUTOGRAD: Record operation if needed
        if self.requires_grad || other.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::BatchMatMul {
                        lhs: self.id,
                        rhs: other.id,
                    },
                    vec![
                        (self.id, self.clone()),
                        (other.id, other.clone()),
                    ],
                );
            }
        }

        Ok(output)
    }

    /// Helper to reshape tensor for batch matrix multiplication
    #[allow(dead_code)]
    fn reshape_for_bmm(&self, target_batch: &[usize], m: usize, n: usize) -> Result<Tensor> {
        let self_shape = self.shape.dims();
        let self_batch = &self_shape[..self_shape.len() - 2];

        // If shapes match, just flatten batch dimensions
        if self_batch == target_batch {
            let batch_size: usize = target_batch.iter().product();
            return self.reshape(&[batch_size, m, n]);
        }

        // Otherwise, need to broadcast
        // Check if we can broadcast
        let self_batch_prod: usize = self_batch.iter().product();
        let target_batch_prod: usize = target_batch.iter().product();

        if self_batch_prod == 1 {
            // Broadcast single batch to target batch size
            let expanded = self.broadcast_to(&Shape::from_dims(&[target_batch_prod, m, n]))?;
            Ok(expanded)
        } else if target_batch_prod % self_batch_prod == 0 {
            // Can broadcast if target is multiple of self
            let repeat_factor = target_batch_prod / self_batch_prod;
            let self_flat = self.reshape(&[self_batch_prod, m, n])?;

            match self.dtype() {
                DType::F32 => {
                    let mut repeated = Tensor::empty_dtype(
                        Shape::from_dims(&[target_batch_prod, m, n]),
                        DType::F32,
                        self.device.clone(),
                    )?;
                    let src = self_flat.storage_ref().try_as_slice_f32()?;
                    let dst = repeated.storage_mut().try_as_mut_slice_f32()?;
                    let chunk = src.len();
                    for r in 0..repeat_factor {
                        let start = r * chunk;
                        let end = start + chunk;
                        let mut dst_view = dst.slice_mut(start..end);
                        self.device
                            .dtod_copy(src, &mut dst_view)
                            .map_err(|e| Error::Cuda(format!("repeat copy failed: {e:?}")))?;
                    }
                    repeated.requires_grad = self.requires_grad;
                    Ok(repeated)
                }
                DType::BF16 => {
                    #[cfg(feature = "bf16_u16")]
                    {
                        let mut repeated = Tensor::empty_dtype(
                            Shape::from_dims(&[target_batch_prod, m, n]),
                            DType::BF16,
                            self.device.clone(),
                        )?;
                        let src = self_flat.storage_ref().try_as_slice_u16().map_err(|_| {
                            Error::InvalidOperation("repeat: expected BF16 storage".into())
                        })?;
                        let dst = repeated.storage_mut().try_as_mut_slice_u16().map_err(|_| {
                            Error::InvalidOperation("repeat: expected BF16 storage".into())
                        })?;
                        let chunk = src.len();
                        for r in 0..repeat_factor {
                            let start = r * chunk;
                            let end = start + chunk;
                            let mut dst_view = dst.slice_mut(start..end);
                            self.device
                                .dtod_copy(src, &mut dst_view)
                                .map_err(|e| Error::Cuda(format!("repeat copy failed: {e:?}")))?;
                        }
                        repeated.requires_grad = self.requires_grad;
                        Ok(repeated)
                    }
                    #[cfg(not(feature = "bf16_u16"))]
                    {
                        let mut repeated_data = Vec::new();
                        for _ in 0..repeat_factor {
                            let self_data = self_flat.to_vec()?;
                            repeated_data.extend_from_slice(&self_data);
                        }
                        let mut tensor = Tensor::from_vec(
                            repeated_data,
                            Shape::from_dims(&[target_batch_prod, m, n]),
                            self.device.clone(),
                        )?;
                        tensor.requires_grad = self.requires_grad;
                        Ok(tensor)
                    }
                }
                _ => {
                    // Fallback to CPU repeat for unsupported dtypes.
                    let mut repeated_data = Vec::new();
                    for _ in 0..repeat_factor {
                        let self_data = self_flat.to_vec()?;
                        repeated_data.extend_from_slice(&self_data);
                    }
                    let mut tensor = Tensor::from_vec(
                        repeated_data,
                        Shape::from_dims(&[target_batch_prod, m, n]),
                        self.device.clone(),
                    )?;
                    tensor.requires_grad = self.requires_grad;
                    Ok(tensor)
                }
            }
        } else {
            Err(Error::InvalidOperation(format!(
                "Cannot broadcast batch dimensions {:?} to {:?}",
                self_batch, target_batch
            )))
        }
    }

    /// Create a slice view of the tensor (internal use)
    #[allow(dead_code)]
    fn slice_internal(&self, start: usize, len: usize) -> Result<Tensor> {
        // Slice implementation for contiguous memory
        // Non-contiguous tensors would require stride handling
        if start + len > self.shape.elem_count() {
            return Err(Error::InvalidOperation("Slice out of bounds".into()));
        }

        // Now actually tries GPU-to-GPU slice copy
        let kernel_code = r#"
extern "C" __global__ void slice_kernel(
    float* output,
    const float* input,
    int start,
    int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        output[idx] = input[start + idx];
    }
}
"#;

        CudaKernels::ensure_kernel(&self.device, "slice_kernel", kernel_code)?;

        let f = self
            .device
            .get_func("slice_kernel", "slice_kernel")
            .ok_or_else(|| Error::Cuda("Failed to get slice_kernel".into()))?;

        let slice_data = alloc_zeros_from_pool(&self.device, len)?;

        let cfg = cudarc::driver::LaunchConfig::for_num_elems(len as u32);

        launch_kernel!(
            f,
            cfg,
            &slice_data,
            self.storage.try_as_slice_f32()?,
            start as i32,
            len as i32
        )?;

        Ok(Tensor {
            storage: TensorStorage::F32 {
                data: slice_data.into(),
                numel: len,
            },
            shape: Shape::from_dims(&[len]),
            device: self.device.clone(),
            id: TensorId::new(),
            requires_grad: false,
            custom_strides: None,
            view_offset: 0,

        })
    }

    fn convert_f32_buffer_to_bool(
        device: &Arc<CudaDevice>,
        buffer: &mut CudaSlice<f32>,
        numel: usize,
    ) -> Result<()> {
        const MODULE: &str = "f32_to_bool_module";
        const KERNEL: &str = "f32_to_bool_kernel";
        if device.get_func(MODULE, KERNEL).is_none() {
            let source = r#"
extern "C" __global__ void f32_to_bool_kernel(
    float* data,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = (data[idx] != 0.0f) ? 1.0f : 0.0f;
    }
}
"#;

            let include_path = std::env::var("CUDA_HOME")
                .map(|p| format!("{}/include", p))
                .unwrap_or_else(|_| "/usr/local/cuda/include".to_string());
            let mut opts = CompileOptions::default();
            opts.include_paths.push(include_path);

            let ptx = compile_ptx_with_opts(source, opts)
                .map_err(|e| Error::Cuda(format!("nvrtc f32_to_bool_kernel: {:?}", e)))?;
            device
                .load_ptx(ptx, MODULE, &[KERNEL])
                .map_err(|e| Error::Cuda(format!("load f32_to_bool_kernel: {:?}", e)))?;
        }

        let func = device
            .get_func(MODULE, KERNEL)
            .ok_or_else(|| Error::Cuda("f32_to_bool_kernel missing".into()))?;
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        unsafe {
            func.launch(cfg, (buffer, numel as i32))
                .map_err(|e| Error::Cuda(format!("launch f32_to_bool_kernel failed: {:?}", e)))?
        };
        Ok(())
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        // Phase 10: BF16+BF16 → TensorIterator, else → GpuOps::add. The
        // `*_bf16_iter` functions handle their own cuda-feature gating
        // (return InvalidOperation on non-cuda builds), so no extra cfg
        // scaffolding is needed here. Autograd tape block is unchanged.
        let mut output = crate::tensor_iterator::dispatch_binary_bf16(
            self,
            other,
            crate::tensor_iterator::ops::binary::add_bf16_iter,
            GpuOps::add,
        )?;
        if self.requires_grad || other.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::Add {
                        lhs: self.id,
                        rhs: other.id,
                        lhs_shape: self.shape.clone(),
                        rhs_shape: other.shape.clone(),
                    },
                    Vec::new(),
                );
            }
        }
        Ok(output)
    }

    /// Subtract another tensor
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        // Phase 10: BF16+BF16 → TensorIterator sub kernel; non-BF16 stays
        // on the add+neg composition (GpuOps has no `sub`; the compose
        // path is the canonical F32 fallback). Autograd records Sub
        // afterwards either way.
        let mut output = crate::tensor_iterator::dispatch_binary_bf16(
            self,
            other,
            crate::tensor_iterator::ops::binary::sub_bf16_iter,
            |a, b| {
                let neg_b = b.mul_scalar(-1.0)?;
                a.add(&neg_b)
            },
        )?;
        if self.requires_grad || other.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::Sub {
                        lhs: self.id,
                        rhs: other.id,
                    },
                    Vec::new(),
                );
            }
        }
        Ok(output)
    }

    /// Element-wise multiplication (Hadamard product)
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        // Phase 10: BF16+BF16 → TensorIterator, else → GpuOps::mul.
        let mut output = crate::tensor_iterator::dispatch_binary_bf16(
            self,
            other,
            crate::tensor_iterator::ops::binary::mul_bf16_iter,
            GpuOps::mul,
        )?;
        if self.requires_grad || other.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::Mul {
                        lhs: self.id,
                        rhs: other.id,
                    },
                    vec![(self.id, self.clone()), (other.id, other.clone())],
                );
            }
        }
        Ok(output)
    }

    /// Scalar multiplication
    pub fn mul_scalar(&self, scalar: f32) -> Result<Tensor> {
        // Phase 10: BF16 → TensorIterator (scalar captured in stateful
        // functor, mirrors PyTorch's opmath_gpu_kernel_with_scalars).
        // Other dtypes → GpuOps::mul_scalar (F32 path).
        let mut output = crate::tensor_iterator::dispatch_scalar_bf16(
            self,
            scalar,
            crate::tensor_iterator::ops::binary::mul_scalar_bf16_iter,
            GpuOps::mul_scalar,
        )?;
        if self.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::MulScalar {
                        input: self.id,
                        scalar,
                    },
                    Vec::new(),
                );
            }
        }
        Ok(output)
    }

    /// Scale tensor by a scalar (alias for mul_scalar)
    pub fn scale(&self, scalar: f32) -> Result<Tensor> {
        self.mul_scalar(scalar)
    }

    /// Add scalar to all elements
    pub fn add_scalar(&self, scalar: f32) -> Result<Tensor> {
        // Phase 10: BF16 → TensorIterator (scalar captured in functor);
        // other dtypes → GpuOps::add_scalar (F32 path).
        let mut output = crate::tensor_iterator::dispatch_scalar_bf16(
            self,
            scalar,
            crate::tensor_iterator::ops::binary::add_scalar_bf16_iter,
            GpuOps::add_scalar,
        )?;
        if self.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::AddScalar {
                        input: self.id,
                        scalar,
                    },
                    Vec::new(),
                );
            }
        }
        Ok(output)
    }

    /// ReLU activation
    pub fn relu(&self) -> Result<Tensor> {
        // Phase 10: BF16 → TensorIterator, else → GpuOps::relu.
        let mut output = crate::tensor_iterator::dispatch_unary_bf16(
            self,
            crate::tensor_iterator::ops::unary::relu_bf16_iter,
            GpuOps::relu,
        )?;
        if self.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::ReLU { input: self.id },
                    vec![(self.id, self.clone())],
                );
            }
        }
        Ok(output)
    }

    /// GELU activation
    pub fn gelu(&self) -> Result<Tensor> {
        // Phase 10: BF16 → TensorIterator, else → GpuOps::gelu.
        let mut output = crate::tensor_iterator::dispatch_unary_bf16(
            self,
            crate::tensor_iterator::ops::unary::gelu_bf16_iter,
            GpuOps::gelu,
        )?;
        if self.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::GELU { input: self.id },
                    vec![(self.id, self.clone())],
                );
            }
        }
        Ok(output)
    }

    /// SiLU (Swish) activation
    pub fn silu(&self) -> Result<Tensor> {
        // Phase 10: BF16 → TensorIterator, else → GpuOps::silu.
        let mut output = crate::tensor_iterator::dispatch_unary_bf16(
            self,
            crate::tensor_iterator::ops::unary::silu_bf16_iter,
            GpuOps::silu,
        )?;
        if self.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::SiLU { input: self.id },
                    vec![(self.id, self.clone())],
                );
            }
        }
        Ok(output)
    }

    /// Fused SwiGLU: silu(self) * up in a single kernel.
    /// Records Op::FusedSwiGLU for autograd backward.
    pub fn swiglu(&self, up: &Tensor) -> Result<Tensor> {
        let mut output = if self.dtype() == DType::BF16 && up.dtype() == DType::BF16 {
            crate::bf16_ops::swiglu_fused_bf16(self, up)?
        } else {
            // Fallback: separate silu + mul
            let s = self.silu()?;
            return s.mul(up);
        };

        if self.requires_grad || up.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::FusedSwiGLU {
                        gate: self.id,
                        up: up.id,
                    },
                    vec![(self.id, self.clone()), (up.id, up.clone())],
                );
            }
        }

        Ok(output)
    }

    /// Tanh activation
    pub fn tanh(&self) -> Result<Tensor> {
        // Phase 10: BF16 → TensorIterator, else → GpuOps::tanh.
        let mut output = crate::tensor_iterator::dispatch_unary_bf16(
            self,
            crate::tensor_iterator::ops::unary::tanh_bf16_iter,
            GpuOps::tanh,
        )?;
        if self.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::Tanh { input: self.id },
                    vec![(self.id, self.clone())],
                );
            }
        }
        Ok(output)
    }

    /// Sigmoid activation
    pub fn sigmoid(&self) -> Result<Tensor> {
        // Phase 10: BF16 → TensorIterator, else → GpuOps::sigmoid.
        let mut output = crate::tensor_iterator::dispatch_unary_bf16(
            self,
            crate::tensor_iterator::ops::unary::sigmoid_bf16_iter,
            GpuOps::sigmoid,
        )?;
        if self.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::Sigmoid { input: self.id },
                    vec![(self.id, self.clone())],
                );
            }
        }
        Ok(output)
    }

    /// Error function (erf) - needed for GELU
    pub fn erf(&self) -> Result<Tensor> {
        let data = self.to_vec()?;
        let result: Vec<f32> = data
            .iter()
            .map(|&x| {
                // Approximation of erf using a series expansion
                // This is good for |x| < 3.7
                let a1 = 0.254_829_6;
                let a2 = -0.284_496_72;
                let a3 = 1.421_413_8;
                let a4 = -1.453_152_1;
                let a5 = 1.061_405_4;
                let p = 0.3275911;

                let sign = if x < 0.0 { -1.0 } else { 1.0 };
                let x = x.abs();

                let t = 1.0 / (1.0 + p * x);
                let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

                sign * y
            })
            .collect();

        Tensor::from_vec(result, self.shape.clone(), self.device.clone())
    }

    /// Exponential function (GPU)
    pub fn exp(&self) -> Result<Tensor> {
        // Phase 10: BF16 → TensorIterator, else → GpuOps::exp. No autograd
        // record here — exp's autograd tape is recorded by callers that
        // need it (or via the composite Op::Exp in autograd_ops_complete).
        crate::tensor_iterator::dispatch_unary_bf16(
            self,
            crate::tensor_iterator::ops::transcendentals::exp_bf16_iter,
            GpuOps::exp,
        )
    }

    /// Square all elements
    pub fn square(&self) -> Result<Tensor> {
        // Phase 10: BF16 → TensorIterator; other dtypes → mul-self-self.
        let mut output = crate::tensor_iterator::dispatch_unary_bf16(
            self,
            crate::tensor_iterator::ops::unary::square_bf16_iter,
            |x| GpuOps::mul(x, x),
        )?;
        if self.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::Square { input: self.id },
                    vec![(self.id, self.clone())],
                );
            }
        }
        Ok(output)
    }

    /// Mean reduction (reduce in FP32 for stability, then downcast if needed)
    pub fn mean(&self) -> Result<Tensor> {
        // Upcast to F32 for numerically stable reduction
        let x32 = if matches!(self.dtype(), DType::F32) {
            self.clone_result()?
        } else {
            self.to_dtype(DType::F32)?
        };
        let sum32 = x32.sum()?;
        let count = self.shape.elem_count() as f32;
        let mut out32 = sum32.mul_scalar(1.0 / count)?;

        // Record mean operation if needed (attach grad to original input)
        if self.requires_grad {
            out32.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    out32.id,
                    Op::Mean {
                        input: self.id,
                        input_shape: self.shape.clone(),
                    },
                    vec![(self.id, self.clone())],
                );
            }
        }

        // Downcast to original dtype when appropriate
        if matches!(self.dtype(), DType::F32) {
            Ok(out32)
        } else {
            out32.to_dtype(self.dtype())
        }
    }

    /// Sum reduction
    pub fn sum(&self) -> Result<Tensor> {
        // Use CUDA kernel for GPU-accelerated sum reduction
        let mut output = GpuOps::sum(self)?;

        // AUTOGRAD: Record operation if needed
        if self.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::Sum {
                        input: self.id,
                        input_shape: self.shape.clone(),
                    },
                    vec![(self.id, self.clone())],
                );
            }
        }

        Ok(output)
    }

    /// Sum along specific dimensions
    pub fn sum_dims(&self, dims: &[usize]) -> Result<Tensor> {
        // Reduce iteratively in descending order to avoid index shifts
        if dims.is_empty() {
            return Ok(self.clone());
        }
        let mut dims_sorted: Vec<usize> = dims.to_vec();
        dims_sorted.sort_unstable();
        dims_sorted.dedup();

        // Perform reductions without recording intermediate ops
        let mut current = self.clone_result()?;
        for &d in dims_sorted.iter().rev() {
            let keep = GpuOps::sum_dim_keepdim(&current, d)?;
            // Build new shape without dimension d
            let mut new_shape = current.shape().dims().to_vec();
            new_shape.remove(d);
            current = keep.reshape(&new_shape)?;
        }

        // Record as a single multi-dim reduction for autograd clarity
        if self.requires_grad {
            let mut out = current;
            out.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    out.id,
                    Op::SumDims {
                        input: self.id,
                        dims: dims_sorted.clone(),
                    },
                    vec![(self.id, self.clone())],
                );
            }
            Ok(out)
        } else {
            Ok(current)
        }
    }

    /// Sum along a specific dimension
    pub fn sum_dim(&self, dim: usize) -> Result<Tensor> {
        let dims = self.shape.dims();
        if dim >= dims.len() {
            return Err(Error::InvalidOperation(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                dims.len()
            )));
        }

        // Compute sum keeping the dimension, then squeeze it away for the final shape
        let summed_keepdim = GpuOps::sum_dim_keepdim(self, dim)?;

        // Build output shape without the reduced dimension
        let mut out_dims = Vec::with_capacity(dims.len() - 1);
        for (i, &d) in dims.iter().enumerate() {
            if i != dim {
                out_dims.push(d);
            }
        }

        let mut output = summed_keepdim.reshape(&out_dims)?;

        // AUTOGRAD: Record operation if needed
        if self.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::SumDim {
                        input: self.id,
                        dim,
                    },
                    vec![(self.id, self.clone())],
                );
            }
        }

        Ok(output)
    }

    /// Get shape
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Return the tensor rank (number of dimensions).
    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Return true when the tensor adheres to NHWC semantics at the public boundary.
    #[inline]
    pub fn is_nhwc(&self) -> bool {
        if self.rank() != 4 {
            return false;
        }
        true
    }

    /// Borrow the raw shape dimensions as a slice.
    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Expect a 1D tensor and return its single dimension.
    pub fn dims1(&self) -> [usize; 1] {
        self.shape
            .dims()
            .try_into()
            .expect("Tensor::dims1 expects a 1D tensor")
    }

    /// Expect a 2D tensor and return its dimensions.
    pub fn dims2(&self) -> [usize; 2] {
        self.shape
            .dims()
            .try_into()
            .expect("Tensor::dims2 expects a 2D tensor")
    }

    /// Expect a 3D tensor and return its dimensions.
    pub fn dims3(&self) -> [usize; 3] {
        self.shape
            .dims()
            .try_into()
            .expect("Tensor::dims3 expects a 3D tensor")
    }

    /// Expect a 4D tensor and return its dimensions.
    pub fn dims4(&self) -> [usize; 4] {
        self.shape
            .dims()
            .try_into()
            .expect("Tensor::dims4 expects a 4D tensor")
    }

    /// Reshape tensor to new shape
    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor> {
        let new_shape = Shape::from_dims(shape);
        if self.shape.elem_count() != new_shape.elem_count() {
            return Err(Error::ShapeMismatch {
                expected: new_shape,
                got: self.shape.clone(),
            });
        }
        // Stride refactor Phase 2a: if self is a non-contiguous view
        // (from permute/transpose) we must materialize before reshape;
        // reshape assumes linear storage and would alias the wrong
        // elements otherwise.
        let base = if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous()?
        };
        // Create view tensor (shares storage) with new shape
        let out = Tensor {
            id: TensorId::new(),
            storage: base.storage.clone(),
            shape: new_shape.clone(),
            device: base.device.clone(),
            requires_grad: base.requires_grad,
            custom_strides: None,
            view_offset: 0,

        };
        // Record autograd op so gradients flow through reshape
        if self.requires_grad && AutogradContext::is_recording() {
            AutogradContext::record_op(
                out.id,
                Op::Reshape {
                    input: self.id,
                    new_shape: new_shape.dims().to_vec(),
                },
                vec![(self.id, self.clone())],
            );
        }
        Ok(out)
    }

    /// Get device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    pub fn slice_1d_device(&self, axis: usize, start: usize, len: usize) -> Result<Tensor> {
        cuda_ops_bf16::slice_axis_bf16(self, axis, start, len)
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    pub fn broadcast_to_device(&self, shape: &[usize]) -> Result<Tensor> {
        cuda_ops_bf16::broadcast_to_bf16(self, shape)
    }

    #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
    pub fn repeat_axis_device(&self, axis: usize, repeats: usize) -> Result<Tensor> {
        cuda_ops_bf16::repeat_axis_bf16(self, axis, repeats)
    }

    /// Get tensor ID for gradient tracking
    pub fn id(&self) -> TensorId {
        self.id
    }

    /// Check if this tensor requires gradients
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Copy to CPU
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        if dtype_trace_enabled() {
            eprintln!(
                "[dtype-trace] to_vec dtype={:?} shape={:?}",
                self.dtype(),
                self.shape.dims()
            );
        }
        // Stride refactor Phase 2a safety net: to_vec reads storage linearly.
        if !self.is_contiguous() {
            return self.contiguous()?.to_vec();
        }
        match self.storage.try_as_slice_f32() {
            Ok(slice) => self
                .device
                .dtoh_sync_copy(slice)
                .map_err(|_| Error::CudaDriver),
            Err(_) => {
                #[cfg(feature = "bf16_u16")]
                {
                    if self.dtype() == DType::BF16 && self.storage_dtype() == DType::BF16 {
                        let raw = self.to_vec_bf16()?;
                        let mut out = Vec::with_capacity(raw.len());
                        for bits in raw {
                            out.push(bf16::from_bits(bits).to_f32());
                        }
                        return Ok(out);
                    }
                }
                // Fallback: explicit cast to F32 when no direct path exists.
                let tmp = self.to_dtype(DType::F32)?;
                tmp.device
                    .dtoh_sync_copy(tmp.storage.try_as_slice_f32()?)
                    .map_err(|_| Error::CudaDriver)
            }
        }
    }

    /// Get single value
    pub fn item(&self) -> Result<f32> {
        if self.shape.elem_count() != 1 {
            return Err(Error::InvalidOperation(
                "item() requires tensor with single element".into(),
            ));
        }
        Ok(self.to_vec()?[0])
    }

    /// Check for exact tensor equality (shape, dtype, and element values)
    pub fn equal(&self, other: &Tensor) -> Result<bool> {
        if self.shape != other.shape {
            return Ok(false);
        }
        if self.dtype() != other.dtype() {
            return Ok(false);
        }

        let lhs = self.to_vec()?;
        let rhs = other.to_vec()?;

        Ok(lhs
            .iter()
            .zip(rhs.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()))
    }

    /// Transpose a 2D tensor.
    ///
    /// **Stride refactor Phase 2a**: metadata-only view (no kernel launch).
    /// Underlying storage is shared with `self`; swapped strides are
    /// recorded in `custom_strides`. Call `.contiguous()` on the result
    /// before feeding to a kernel that walks storage linearly.
    pub fn transpose(&self) -> Result<Tensor> {
        let dims = self.shape.dims();
        if dims.len() != 2 {
            return Err(Error::InvalidOperation(format!(
                "transpose() expects 2D tensor, got rank {}",
                dims.len()
            )));
        }
        let self_strides = self.strides();
        let new_shape = vec![dims[1], dims[0]];
        let new_strides: crate::shape::Strides =
            smallvec::smallvec![self_strides[1], self_strides[0]];

        let mut output = Tensor {
            storage: self.storage.clone(),
            shape: Shape::from_dims(&new_shape),
            device: self.device.clone(),
            id: TensorId::new(),
            requires_grad: self.requires_grad,
            custom_strides: Some(new_strides),
            view_offset: self.view_offset,
        };

        // AUTOGRAD: Record operation if needed
        if self.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::Transpose { input: self.id },
                    vec![(self.id, self.clone())],
                );
            }
        }

        Ok(output)
    }

    /// Transpose two dimensions of a tensor
    ///
    /// Broadcast tensor to a new shape
    pub fn broadcast_to(&self, target_shape: &Shape) -> Result<Tensor> {
        let src_shape = self.shape.dims();
        let dst_shape = target_shape.dims();

        // Check if broadcast is valid
        if src_shape.len() > dst_shape.len() {
            return Err(Error::InvalidOperation(format!(
                "Cannot broadcast from {:?} to {:?}: source has more dimensions",
                src_shape, dst_shape
            )));
        }

        // Pad source shape with ones on the left
        let mut padded_src: Vec<usize> = vec![1; dst_shape.len() - src_shape.len()];
        padded_src.extend_from_slice(src_shape);

        // Check compatibility
        for (i, (&src_dim, &dst_dim)) in padded_src.iter().zip(dst_shape.iter()).enumerate() {
            if src_dim != dst_dim && src_dim != 1 {
                return Err(Error::InvalidOperation(format!(
                    "Cannot broadcast dimension {} from {} to {}",
                    i, src_dim, dst_dim
                )));
            }
        }

        // If shapes are equal after padding, reshape if rank differs.
        if &padded_src[..] == dst_shape {
            if src_shape.len() != dst_shape.len() {
                return self.reshape(dst_shape);
            }
            return Ok(self.clone());
        }

        let target_i64: Vec<i64> = dst_shape.iter().map(|&d| d as i64).collect();
        crate::ops::broadcast::broadcast_to_impl(self, &target_i64)
    }

    /// Authoritative strides: returns the view strides if set, else row-major.
    ///
    /// Stride refactor Phase 2: view ops (permute/transpose/narrow/chunk)
    /// populate `custom_strides` with a reordered-strides vector; everything
    /// else produces contiguous tensors whose strides are row-major (computed
    /// on demand, same as before the refactor).
    ///
    /// Returns a `Strides` (SmallVec<[usize;6]>), not `Vec<usize>` — rank ≤ 6
    /// in every real DL tensor, so no heap allocation. Prior `Vec` return
    /// cost ~16 ns/call in the default allocator; kernel launchers hit this
    /// on every launch. See `benches/strides_alloc.rs` for the baseline.
    pub fn strides(&self) -> crate::shape::Strides {
        if let Some(s) = &self.custom_strides {
            return s.clone();
        }
        self.shape.stride_contiguous()
    }

    /// Fill a caller-provided buffer with this tensor's real strides, returning
    /// the rank. No heap allocation — callers on hot paths (FFI view setup,
    /// kernel-wrapper entry) should prefer this to `strides()`.
    ///
    /// Panics if `out.len() < rank`. Caller is responsible for sizing.
    #[inline]
    pub fn fill_strides_into(&self, out: &mut [usize]) -> usize {
        let rank = self.shape.rank();
        if let Some(s) = &self.custom_strides {
            out[..rank].copy_from_slice(&s[..rank]);
        } else {
            let dims = self.shape.dims();
            if rank == 0 {
                // nothing to write
            } else {
                out[rank - 1] = 1;
                for i in (0..rank - 1).rev() {
                    out[i] = out[i + 1] * dims[i + 1];
                }
            }
        }
        rank
    }

    /// Element offset into the underlying storage. Non-zero only for narrow/chunk views.
    #[inline]
    pub fn offset(&self) -> usize {
        self.view_offset
    }

    /// Is this tensor laid out contiguously row-major in its storage (no view)?
    ///
    /// True when no custom strides are set AND the view offset is zero. A
    /// row-major contiguous tensor is safe to read via a raw pointer that
    /// indexes linearly from the storage base. Strided views must be passed
    /// through `.contiguous()` (or a stride-aware kernel) before being fed to
    /// a kernel that assumes linear storage.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.custom_strides.is_none() && self.view_offset == 0
    }

    /// Return a contiguous, row-major copy if this tensor is a strided view;
    /// otherwise return a cheap clone.
    ///
    /// Dispatches to one of two materializing kernels:
    ///   * `permute_generic` — for permute/transpose views (custom_strides set,
    ///     view_offset == 0). Recovers the permutation by sorting strides.
    ///   * `materialize_view` — for narrow/chunk views (view_offset != 0) and
    ///     any composition thereof. Walks the source with (strides + offset).
    /// True iff `self.custom_strides` corresponds to some permutation of the
    /// row-major stride pattern of `self.shape`. Safe permute-recovery path.
    /// False for narrow-views where the stored strides correspond to the
    /// PARENT tensor's shape, not self's shape. Used to gate the
    /// permutation-recovery optimization in `contiguous()`.
    fn strides_match_permute_of_shape(&self) -> bool {
        let Some(strides) = self.custom_strides.as_ref() else {
            return true;
        };
        let dims = self.shape.dims();
        if strides.len() != dims.len() {
            return false;
        }
        // The multiset of (dim_i * stride_i)-product piece widths would only
        // tile the storage exactly when strides reflect a permutation. The
        // tightest sufficient check: sorting strides descending and
        // pairing each with the corresponding dim, the resulting
        // [prod_of_smaller_dims] sequence must equal the inner-to-outer
        // stride pattern of a contig shape. Equivalently: sort by stride,
        // the smallest stride must be 1, and each stride must equal the
        // product of all smaller dims.
        let mut pairs: Vec<(usize, usize)> =
            strides.iter().copied().zip(dims.iter().copied()).collect();
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        let mut expected: usize = 1;
        for (stride, dim) in &pairs {
            if *stride != expected {
                return false;
            }
            expected *= *dim;
        }
        true
    }

    pub fn contiguous(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }
        // Always route non-contig views through materialize_view. It handles
        // permute, narrow, and any compositions uniformly via a strided
        // gather. The old "recover permutation from strides" path below is
        // preserved behind a strict check: only viable for pure permute
        // views (view_offset == 0 AND strides match a known permutation of
        // the shape's row-major layout). narrow-at-start-0 has offset == 0
        // but strides from the LARGER parent, which the permute-recovery
        // path misinterprets as an identity permutation and reads the wrong
        // storage region. Fix 2026-04-23 Phase 2a.
        if self.view_offset != 0 || !self.strides_match_permute_of_shape() {
            let contig = crate::cuda_ops::GpuOps::materialize_view(self)?;
            return self.finalize_contiguous_autograd(contig);
        }
        // Recover the permutation from view strides vs row-major strides of
        // the current shape. For a view produced by `permute(dims)`:
        //   view_shape[i]   = orig_shape[dims[i]]
        //   view_strides[i] = orig_strides[dims[i]]
        //     = product(orig_shape[dims[i]+1 ..])
        //
        // We don't have `dims` recorded explicitly, but we can reconstruct it
        // by matching each view-stride to the unique row-major stride of the
        // corresponding original axis. The original shape = view_shape
        // permuted backwards, but we only need `dims` (the forward mapping)
        // to call `permute_generic` on the original-layout storage.
        //
        // The view's storage physically holds the ORIGINAL tensor. Its shape
        // is `orig_shape` where each axis sits at position `inv_perm[axis]`.
        // So if view_shape[i] = orig_shape[dims[i]], then
        // orig_shape[j] = view_shape[inv_perm[j]] where inv_perm[dims[i]]=i.
        let view_strides = self.custom_strides.as_ref().expect("custom_strides Some");
        let view_shape = self.shape.dims();
        let rank = view_shape.len();
        if view_strides.len() != rank {
            return Err(Error::InvalidOperation(format!(
                "custom_strides len {} != shape rank {}",
                view_strides.len(),
                rank
            )));
        }
        // Sort axes by descending stride: that recovers the original axis order.
        // (Row-major strides are strictly decreasing in original order.)
        let mut by_stride: Vec<(usize, usize)> = view_strides
            .iter()
            .copied()
            .enumerate()
            .collect(); // (view_axis, stride)
        by_stride.sort_by(|a, b| b.1.cmp(&a.1));
        // `dims[i] = orig_axis at view_axis i`.
        // After sorting by stride desc, by_stride[j].0 is the view-axis whose
        // original-axis index is j. So dims[by_stride[j].0] = j.
        let mut dims = vec![0usize; rank];
        for (j, (view_axis, _)) in by_stride.iter().enumerate() {
            dims[*view_axis] = j;
        }
        // Build a contiguous tensor with original shape (permuted back), then
        // permute it forward to recover view layout, materialized.
        let mut orig_shape = vec![0usize; rank];
        for i in 0..rank {
            orig_shape[dims[i]] = view_shape[i];
        }
        // Reinterpret `self` as a contiguous tensor with `orig_shape` (the
        // underlying storage IS contiguous in that layout — it was before the
        // permute view was taken).
        let as_orig = Tensor {
            storage: self.storage.clone(),
            shape: Shape::from_dims(&orig_shape),
            device: self.device.clone(),
            id: TensorId::new(),
            requires_grad: self.requires_grad,
            custom_strides: None,
            view_offset: 0,
        };
        // Now apply the recorded permutation to materialize.
        //
        // Dispatch to the hand-written specialized kernels when the
        // permutation matches a known fast path — same kernels the
        // pre-refactor `Tensor::permute` used. Falls back to the general
        // permute_generic kernel for anything else.
        let contig = if rank == 3 && dims == [0, 2, 1] {
            crate::cuda_ops::GpuOps::permute_021(&as_orig)?
        } else if rank == 4 && dims == [0, 2, 1, 3] {
            crate::cuda_ops::GpuOps::permute_0213(&as_orig)?
        } else {
            crate::cuda_ops::GpuOps::permute_generic(&as_orig, &dims)?
        };
        self.finalize_contiguous_autograd(contig)
    }

    /// Propagate `requires_grad` across a `contiguous()` materialization and
    /// record an autograd op so backward can route gradients back to the
    /// strided source. `contiguous()` is value-preserving and shape-
    /// preserving, so the backward is an identity (same-shape) reshape —
    /// we reuse `Op::Reshape` whose backward already handles that.
    ///
    /// Prior to 2026-04-23 this was a no-op because `narrow()` materialized
    /// eagerly. Phase 2a made `narrow()` a zero-copy view, so the first
    /// op forced through `contiguous()` (typically `to_dtype` on a narrow
    /// result) used to silently drop autograd. This finalizer restores
    /// the chain.
    fn finalize_contiguous_autograd(&self, mut contig: Tensor) -> Result<Tensor> {
        if self.requires_grad {
            contig.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    contig.id,
                    Op::Reshape {
                        input: self.id,
                        new_shape: contig.shape.dims().to_vec(),
                    },
                    vec![(self.id, self.alias())],
                );
            }
        }
        Ok(contig)
    }

    /// Get the strides of this tensor (compatibility shim — prefer `strides()`).
    /// Row-major unless the tensor is a view. Returns `Strides` (inline-6
    /// SmallVec), not `Vec<usize>` — see `Tensor::strides` for rationale.
    pub fn stride(&self) -> crate::shape::Strides {
        self.strides()
    }

    /// Debug helper: inspect underlying storage dtype (no guarantee about views).
    pub fn storage_dtype(&self) -> DType {
        self.storage.dtype()
    }

    /// Run the experimental autograd v4 backward pass.
    #[cfg(feature = "autograd_v4")]
    pub fn backward_v4(&self) -> crate::Result<crate::autograd_v4::Gradients> {
        crate::autograd_v4::backward_v4(self)
    }

    /// Convert tensor to a Vec<f32> on CPU
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        let tmp = self.storage.to_f32(&self.device)?;
        let cpu_data = self
            .device
            .dtoh_sync_copy(&tmp)
            .map_err(|_| Error::CudaDriver)?;
        Ok(cpu_data)
    }

    #[cfg(feature = "bf16_u16")]
    pub fn to_vec_bf16(&self) -> Result<Vec<u16>> {
        if self.dtype() != DType::BF16 {
            return Err(Error::InvalidOperation(
                "to_vec_bf16: tensor dtype is not BF16".into(),
            ));
        }
        self.storage.to_vec_bf16(&self.device)
    }

    #[cfg(not(feature = "bf16_u16"))]
    #[allow(unused_variables)]
    pub fn to_vec_bf16(&self) -> Result<Vec<u16>> {
        Err(Error::InvalidOperation(
            "to_vec_bf16 requires the bf16_u16 feature".into(),
        ))
    }

    /// Convert tensor to raw bytes (copies from GPU)
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        match self.dtype() {
            DType::F32 => {
                let vec = self.to_vec_f32()?;
                Ok(unsafe {
                    std::slice::from_raw_parts(
                        vec.as_ptr() as *const u8,
                        vec.len() * std::mem::size_of::<f32>(),
                    )
                }
                .to_vec())
            }
            DType::BF16 => {
                let vec = self.to_vec_bf16()?;
                Ok(unsafe {
                    std::slice::from_raw_parts(
                        vec.as_ptr() as *const u8,
                        vec.len() * std::mem::size_of::<u16>(),
                    )
                }
                .to_vec())
            }
            _ => Err(Error::InvalidOperation(
                "to_bytes not implemented for this dtype".into(),
            )),
        }
    }

    /// Clone the tensor (creates a new tensor with copied data)
    /// Fallible deep/device clone. Prefer `clone_result()` in internal code
    /// and the `Clone` trait (`.clone()`) for infallible API.
    pub fn clone_result(&self) -> Result<Tensor> {
        // For non-contiguous tensors (narrow / permute / transpose views),
        // we MUST materialize via `.contiguous()` first. The dtod_copy
        // paths below copy parent storage and label the result with the
        // view's logical shape, scrambling element addressing.
        if !self.is_contiguous() {
            return self.contiguous();
        }

        // Clone the storage while preserving dtype
        let storage = match &self.storage {
            TensorStorage::F32 { data, numel } => {
                let mut new_data = alloc_aligned_f32(&self.device, *numel)?;
                self.device
                    .dtod_copy(slice_ref(data), &mut new_data)
                    .map_err(|_| Error::CudaDriver)?;
                TensorStorage::F32 {
                    data: wrap_slice(new_data),
                    numel: *numel,
                }
            }
            TensorStorage::F16 { data, numel, scale } => {
                let mut new_data = alloc_aligned_f32(&self.device, *numel)?;
                self.device
                    .dtod_copy(slice_ref(data), &mut new_data)
                    .map_err(|_| Error::CudaDriver)?;
                TensorStorage::F16 {
                    data: wrap_slice(new_data),
                    numel: *numel,
                    scale: *scale,
                }
            }
            TensorStorage::BF16 { data, numel } => {
                #[cfg(not(feature = "bf16_u16"))]
                {
                    let mut new_data = alloc_aligned_f32(&self.device, *numel)?;
                    self.device
                        .dtod_copy(slice_ref(data), &mut new_data)
                        .map_err(|_| Error::CudaDriver)?;
                    TensorStorage::BF16 {
                        data: wrap_slice(new_data),
                        numel: *numel,
                    }
                }
                #[cfg(feature = "bf16_u16")]
                {
                    let mut new_data = unsafe { self.device.alloc::<u16>(*numel) }
                        .map_err(|_| Error::CudaDriver)?;
                    self.device
                        .dtod_copy(slice_ref(data), &mut new_data)
                        .map_err(|_| Error::CudaDriver)?;
                    TensorStorage::BF16 {
                        data: wrap_slice(new_data),
                        numel: *numel,
                    }
                }
            }
            #[cfg(feature = "bf16_u16")]
            TensorStorage::BF16Arena { ptr, numel, .. } => {
                use cudarc::driver::DevicePtrMut;
                let mut new_data =
                    unsafe { self.device.alloc::<u16>(*numel) }.map_err(|_| Error::CudaDriver)?;
                let stream = CudaStream::from_raw(self.device.cuda_stream_raw_ptr());
                bf16_copy_async(
                    (*new_data.device_ptr_mut()) as *mut std::ffi::c_void,
                    ptr.as_ptr() as *const std::ffi::c_void,
                    *numel,
                    &stream,
                )?;
                TensorStorage::BF16 {
                    data: wrap_slice(new_data),
                    numel: *numel,
                }
            }
            #[cfg(feature = "bf16_u16")]
            TensorStorage::BF16View { ptr, numel } => {
                use cudarc::driver::DevicePtrMut;
                let mut new_data =
                    unsafe { self.device.alloc::<u16>(*numel) }.map_err(|_| Error::CudaDriver)?;
                let stream = CudaStream::from_raw(self.device.cuda_stream_raw_ptr());
                bf16_copy_async(
                    (*new_data.device_ptr_mut()) as *mut std::ffi::c_void,
                    ptr.as_ptr() as *const std::ffi::c_void,
                    *numel,
                    &stream,
                )?;
                TensorStorage::BF16 {
                    data: wrap_slice(new_data),
                    numel: *numel,
                }
            }
            TensorStorage::I8 { data, numel } => {
                let mut new_data =
                    unsafe { self.device.alloc::<i8>(*numel) }.map_err(|_| Error::CudaDriver)?;
                self.device
                    .dtod_copy(slice_ref(data), &mut new_data)
                    .map_err(|_| Error::CudaDriver)?;
                TensorStorage::I8 {
                    data: wrap_slice(new_data),
                    numel: *numel,
                }
            }
            TensorStorage::I32 { data, numel } => {
                let mut new_data = alloc_aligned_f32(&self.device, *numel)?;
                self.device
                    .dtod_copy(slice_ref(data), &mut new_data)
                    .map_err(|_| Error::CudaDriver)?;
                TensorStorage::I32 {
                    data: wrap_slice(new_data),
                    numel: *numel,
                }
            }
            TensorStorage::Bool { data, numel } => {
                let mut new_data = alloc_aligned_f32(&self.device, *numel)?;
                self.device
                    .dtod_copy(slice_ref(data), &mut new_data)
                    .map_err(|_| Error::CudaDriver)?;
                TensorStorage::Bool {
                    data: wrap_slice(new_data),
                    numel: *numel,
                }
            }
        };

        Ok(Tensor {
            storage,
            shape: self.shape.clone(),
            device: self.device.clone(),
            id: TensorId::new(),
            requires_grad: self.requires_grad,
            custom_strides: None,
            view_offset: 0,

        })
    }

    // NOTE: Keep any additional tensor methods below.

    /// Detach from computation graph
    pub fn detach(&self) -> Result<Tensor> {
        let mut tensor = self.clone_result()?;
        tensor.requires_grad = false;
        Ok(tensor)
    }

    /// Detach and create a leaf variable for gradient checkpointing.
    ///
    /// Returns a tensor with:
    /// - Same storage (Arc bump, zero GPU copy)
    /// - **New** TensorId (disconnected from outer autograd graph)
    /// - requires_grad = true (so the local recompute graph tracks it)
    ///
    /// This is the flame-core equivalent of PyTorch's `detach_variable`:
    /// the returned tensor acts as a leaf in any local autograd scope,
    /// so backward() stops here and accumulates `.grad`.
    pub fn detach_leaf(&self) -> Tensor {
        let mut t = self.clone(); // Arc bump, same storage, same shape
        t.id = TensorId::new();   // Fresh ID — not in any tape
        t.requires_grad = true;   // Track in the new local graph
        t
    }

    /// 2D Convolution
    pub fn conv2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: usize,
        padding: usize,
    ) -> Result<Tensor> {
        assert_nhwc_bf16_public("Tensor::conv2d in", self)?;
        assert_nhwc_bf16_public("Tensor::conv2d in(weight)", weight)?;
        if let Some(b) = bias {
            assert_nhwc_bf16_public("Tensor::conv2d in(bias)", b)?;
        }

        let mut out = crate::cuda_conv2d::conv2d(self, weight, bias, stride, padding)?;
        if out.dtype() != DType::BF16 {
            out = out.to_dtype(DType::BF16)?;
        }
        assert_nhwc_bf16_public("Tensor::conv2d out", &out)?;
        Ok(out)
    }

    /// Create a view of the tensor with new shape (shares data)
    pub fn view(&self, new_shape: &[usize]) -> Result<Tensor> {
        // For now, view is the same as reshape since we clone the data pointer
        // In a full implementation, view would share the underlying data
        self.reshape(new_shape)
    }

    /// Flatten tensor to 2D: [batch_size, -1]
    pub fn flatten(&self, start_dim: usize) -> Result<Tensor> {
        let dims = self.shape.dims();
        if start_dim >= dims.len() {
            return Err(Error::InvalidOperation(format!(
                "start_dim {} out of range for tensor with {} dims",
                start_dim,
                dims.len()
            )));
        }

        let batch_size: usize = dims[..start_dim].iter().product();
        let feature_size: usize = dims[start_dim..].iter().product();

        self.reshape(&[batch_size, feature_size])
    }

    /// Permute/transpose dimensions.
    ///
    /// **Stride refactor Phase 2a**: metadata-only view — no kernel launch,
    /// no storage copy. The returned `Tensor` shares the underlying storage
    /// `Arc` with `self` and records the permuted strides in `custom_strides`.
    /// Callers that need row-major contiguous layout (any kernel that walks
    /// storage linearly) must call `.contiguous()` on the result; kernels
    /// that accept per-tensor strides can consume the view directly and
    /// skip the copy.
    ///
    /// Prior to this refactor `permute` materialized via `GpuOps::permute_*`
    /// kernels on every call. After the refactor materialization happens
    /// lazily only when a caller explicitly asks for contiguity.
    /// Build a zero-copy view over this tensor's storage with caller-supplied
    /// shape, strides, and element offset (PyTorch's `as_strided`). Used by
    /// `narrow`/`chunk` and by parity tests that need to construct views
    /// without going through a materializing op.
    ///
    /// The caller is responsible for ensuring every reachable coordinate falls
    /// inside the underlying storage; a debug-only bounds check is performed.
    /// No autograd op is recorded here — the higher-level op (narrow, chunk,
    /// etc.) is responsible for recording the appropriate backward op.
    pub fn as_strided(
        &self,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> Result<Tensor> {
        if shape.len() != strides.len() {
            return Err(Error::InvalidOperation(format!(
                "as_strided: shape rank {} != strides rank {}",
                shape.len(),
                strides.len()
            )));
        }
        #[cfg(debug_assertions)]
        {
            if !shape.is_empty() {
                let max_linear: usize = shape
                    .iter()
                    .zip(strides.iter())
                    .map(|(&d, &s)| if d == 0 { 0 } else { (d - 1) * s })
                    .sum();
                let storage_elems = self.storage.len();
                debug_assert!(
                    offset + max_linear < storage_elems || (shape.iter().any(|&d| d == 0)),
                    "as_strided: view [shape={:?}, strides={:?}, offset={}] exceeds \
                     storage ({} elements)",
                    shape,
                    strides,
                    offset,
                    storage_elems
                );
            }
        }
        Ok(Tensor {
            storage: self.storage.clone(),
            shape: Shape::from_dims(shape),
            device: self.device.clone(),
            id: TensorId::new(),
            requires_grad: false,
            custom_strides: Some(smallvec::SmallVec::from_slice(strides)),
            view_offset: offset,
        })
    }

    pub fn permute(&self, dims: &[usize]) -> Result<Tensor> {
        let shape = self.shape.dims();
        if dims.len() != shape.len() {
            return Err(Error::InvalidOperation(format!(
                "Permute dims {:?} doesn't match tensor dims {:?}",
                dims, shape
            )));
        }

        // Check for valid permutation
        let mut seen = vec![false; dims.len()];
        for &d in dims {
            if d >= dims.len() {
                return Err(Error::InvalidOperation(format!(
                    "Invalid permutation dimension: {}",
                    d
                )));
            }
            if seen[d] {
                return Err(Error::InvalidOperation(format!(
                    "Duplicate dimension in permutation: {}",
                    d
                )));
            }
            seen[d] = true;
        }

        // Identity permutation → cheap clone, no view needed.
        let is_identity = dims.iter().enumerate().all(|(i, &d)| i == d);

        let self_strides = self.strides();
        let new_shape: Vec<usize> = dims.iter().map(|&d| shape[d]).collect();
        let new_strides: crate::shape::Strides =
            dims.iter().map(|&d| self_strides[d]).collect();

        let mut output = if is_identity {
            self.clone()
        } else {
            Tensor {
                storage: self.storage.clone(),
                shape: Shape::from_dims(&new_shape),
                device: self.device.clone(),
                id: TensorId::new(),
                requires_grad: self.requires_grad,
                custom_strides: Some(new_strides),
                view_offset: self.view_offset,
            }
        };

        // AUTOGRAD: Record operation if needed
        if self.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::Permute {
                        input: self.id,
                        dims: dims.to_vec(),
                    },
                    vec![(self.id, self.clone())],
                );
            }
        }

        Ok(output)
    }

    /// Compute softmax along a dimension
    pub fn softmax(&self, dim: isize) -> Result<Tensor> {
        let shape = self.shape().dims();
        let ndim = shape.len() as isize;

        // Handle negative dimension
        let dim = if dim < 0 { ndim + dim } else { dim } as usize;

        if dim >= shape.len() {
            return Err(Error::InvalidOperation(format!(
                "Dimension {} out of bounds for tensor with {} dimensions",
                dim,
                shape.len()
            )));
        }

        // FAST PATH: BF16 + last-dim softmax → fused kernel, no scratch allocs.
        // Replaces the 5-step pipeline (max → sub → exp → sum → div) which was
        // catastrophically slow (175× PyTorch) and allocated 5× tensor memory.
        #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
        {
            if !self.requires_grad
                && self.dtype() == DType::BF16
                && dim == shape.len() - 1
            {
                let mut output = crate::bf16_elementwise::softmax_lastdim_bf16(self)?;
                if output.dtype() != crate::config::default_dtype() {
                    output = output.to_dtype(crate::config::default_dtype())?;
                }
                return Ok(output);
            }
        }

        // Compute max along dimension for numerical stability
        let mut max_vals = GpuOps::max_dim(self, dim, true)?;
        if max_vals.dtype() != self.dtype() {
            max_vals = max_vals.to_dtype(self.dtype())?;
        }
        let shifted = self.sub(&max_vals)?;

        // Compute exp
        let exp_vals = shifted.exp()?;

        // Sum along dimension
        let mut sum_exp = GpuOps::sum_dim_keepdim(&exp_vals, dim)?;
        if sum_exp.dtype() != exp_vals.dtype() {
            sum_exp = sum_exp.to_dtype(exp_vals.dtype())?;
        }

        // Divide by sum (compute in F32, cast back to default dtype if needed)
        let mut output = exp_vals.div(&sum_exp)?;

        // AUTOGRAD: Record operation if needed
        if self.requires_grad {
            output.requires_grad = true;
            if AutogradContext::is_recording() {
                AutogradContext::record_op(
                    output.id,
                    Op::Softmax {
                        input: self.id,
                        dim: dim as isize,
                    },
                    vec![(self.id, self.clone())],
                );
            }
        }

        // Cast to default dtype if not F32
        match crate::config::default_dtype() {
            DType::F32 => Ok(output),
            dt => output.to_dtype(dt),
        }
    }

    // NOTE: `permute_0132` (CPU scalar loop that downcast BF16 → F32) was
    // removed 2026-04-06. The `[0,1,3,2]` case now flows through
    // `Tensor::permute` → `GpuOps::permute_generic`, which is a real GPU
    // scatter that preserves dtype. See PERF_PERMUTE_FALLBACK_FIX.md.

    /// Add bias (broadcasting over batch dimensions)
    pub fn add_bias(&self, bias: &Tensor) -> Result<Tensor> {
        let shape = self.shape.dims();
        let bias_shape = bias.shape.dims();

        if bias_shape.len() != 1 || bias_shape[0] != shape[shape.len() - 1] {
            return Err(Error::InvalidOperation(format!(
                "Bias shape {:?} incompatible with tensor shape {:?}",
                bias_shape, shape
            )));
        }

        let target_dtype = self.dtype();
        let bias_cast = if bias.dtype() == target_dtype {
            bias.clone_result()?
        } else {
            bias.to_dtype(target_dtype)?
        };

        let bias_broadcast = bias_cast.broadcast_to(self.shape())?;
        self.add(&bias_broadcast)
    }

    /// Flatten tensor from a given dimension
    pub fn flatten_from(&self, from_dim: usize) -> Result<Tensor> {
        let dims = self.shape.dims();
        if from_dim >= dims.len() {
            return Err(Error::InvalidOperation(format!(
                "flatten_from: dimension {} out of range for tensor with {} dimensions",
                from_dim,
                dims.len()
            )));
        }

        // Calculate new shape
        let mut new_dims = dims[..from_dim].to_vec();
        let flattened_size: usize = dims[from_dim..].iter().product();
        new_dims.push(flattened_size);

        // Data remains the same, just reshape
        self.reshape(&new_dims)
    }

    /// Get a CUDA slice reference to the tensor data
    pub fn to_cuda_slice(&self) -> Result<&CudaSlice<f32>> {
        self.storage.try_as_slice_f32()
    }

    /// Transpose the last two dimensions (for batch operations)
    pub fn transpose_batch(&self) -> Result<Tensor> {
        let ndim = self.shape.dims().len();
        if ndim < 2 {
            return Err(Error::InvalidOperation(
                "Transpose batch requires at least 2 dimensions".into(),
            ));
        }

        // Swap last two dimensions
        let mut perm: Vec<usize> = (0..ndim).collect();
        perm[ndim - 2] = ndim - 1;
        perm[ndim - 1] = ndim - 2;

        self.permute(&perm)
    }

    /// Batch matrix multiplication
    pub fn batch_matmul(&self, other: &Tensor) -> Result<Tensor> {
        let self_dims = self.shape.dims();
        let other_dims = other.shape.dims();

        if self_dims.len() < 2 || other_dims.len() < 2 {
            return Err(Error::InvalidOperation(
                "Batch matmul requires at least 2D tensors".into(),
            ));
        }

        // Check batch dimensions match
        if self_dims[..self_dims.len() - 2] != other_dims[..other_dims.len() - 2] {
            return Err(Error::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }

        // Check matrix dimensions are compatible
        let m = self_dims[self_dims.len() - 2];
        let k1 = self_dims[self_dims.len() - 1];
        let k2 = other_dims[other_dims.len() - 2];
        let n = other_dims[other_dims.len() - 1];

        if k1 != k2 {
            return Err(Error::InvalidOperation(format!(
                "Matrix dimensions incompatible for matmul: ({}, {}) @ ({}, {})",
                m, k1, k2, n
            )));
        }

        // Implement using regular matmul on flattened batches
        // Fast-path: if effective batch_size == 1, collapse to 2D GEMM (cuBLAS)
        let batch_size: usize = self_dims[..self_dims.len() - 2].iter().product();
        if batch_size == 1 {
            let a2d = self.reshape(&[m, k1])?;
            let b2d = other.reshape(&[k2, n])?;
            let out2d = a2d.matmul(&b2d)?;
            let mut out_dims = self_dims[..self_dims.len() - 2].to_vec();
            if out_dims.is_empty() {
                out_dims.push(1);
            }
            out_dims.push(m);
            out_dims.push(n);
            return out2d.reshape(&out_dims);
        }

        // Reshape to [batch_size, m, k] and [batch_size, k, n]
        let self_3d = self.reshape(&[batch_size, m, k1])?;
        let other_3d = other.reshape(&[batch_size, k2, n])?;

        // General path (small batches): perform per-batch GEMM; TODO: switch to strided batched GEMM
        let mut results = Vec::new();
        for b in 0..batch_size {
            // Get slice for this batch
            let self_2d = self_3d
                .slice_1d(b * m * k1, (b + 1) * m * k1)?
                .reshape(&[m, k1])?;
            let other_2d = other_3d
                .slice_1d(b * k2 * n, (b + 1) * k2 * n)?
                .reshape(&[k2, n])?;

            let result = self_2d.matmul(&other_2d)?;
            results.push(result);
        }

        // Stack results
        let stacked = Self::stack(&results, 0)?;

        // Reshape back to original batch dimensions
        let mut output_shape = self_dims[..self_dims.len() - 2].to_vec();
        output_shape.push(m);
        output_shape.push(n);

        stacked.reshape(&output_shape)
    }

    /// Stack tensors along a new dimension
    pub fn stack(tensors: &[Tensor], axis: usize) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(Error::InvalidOperation(
                "Cannot stack empty tensor list".into(),
            ));
        }

        // Verify all tensors have the same shape
        let first_shape = &tensors[0].shape;
        for t in &tensors[1..] {
            if t.shape != *first_shape {
                return Err(Error::ShapeMismatch {
                    expected: first_shape.clone(),
                    got: t.shape.clone(),
                });
            }
        }
        let mut expanded = Vec::with_capacity(tensors.len());
        for t in tensors {
            expanded.push(t.unsqueeze(axis)?);
        }
        let refs: Vec<&Tensor> = expanded.iter().collect();
        Tensor::cat(&refs, axis)
    }

    /// Slice tensor data (renamed to avoid conflict)
    pub fn slice_1d(&self, start: usize, end: usize) -> Result<Tensor> {
        if end > self.shape.elem_count() || start > end {
            return Err(Error::InvalidOperation(format!(
                "Invalid slice range {}..{} for tensor with {} elements",
                start,
                end,
                self.shape.elem_count()
            )));
        }

        let data = self.to_vec()?;
        let slice_data = data[start..end].to_vec();

        // Calculate shape of slice
        let slice_len = end - start;
        Tensor::from_vec(
            slice_data,
            Shape::from_dims(&[slice_len]),
            self.device.clone(),
        )
    }

    /// Squeeze dimension (renamed to avoid conflict)
    pub fn squeeze_dim(&self, dim: usize) -> Result<Tensor> {
        let dims = self.shape.dims();
        if dim >= dims.len() {
            return Err(Error::InvalidOperation(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                dims.len()
            )));
        }

        if dims[dim] != 1 {
            return Err(Error::InvalidOperation(format!(
                "Cannot squeeze dimension {} with size {}",
                dim, dims[dim]
            )));
        }

        let mut new_dims = dims.to_vec();
        new_dims.remove(dim);

        self.reshape(&new_dims)
    }

    /// Narrow (slice) a tensor along a dimension - CUDA only implementation
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Tensor> {
        let dims = self.shape.dims();
        if dim >= dims.len() {
            return Err(Error::InvalidOperation(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                dims.len()
            )));
        }

        if start + length > dims[dim] {
            return Err(Error::InvalidOperation(format!(
                "Slice [{}, {}) out of range for dimension {} of size {}",
                start,
                start + length,
                dim,
                dims[dim]
            )));
        }

        // Create output shape
        let mut output_dims = dims.to_vec();
        output_dims[dim] = length;
        let output_shape = Shape::from_dims(&output_dims);

        #[cfg(all(feature = "cuda", feature = "bf16_u16"))]
        if self.dtype() == DType::BF16 && self.storage_dtype() == DType::BF16 {
            // Stride refactor Phase 2a: narrow is a zero-copy view.
            let self_strides = self.strides();
            let new_offset = self.view_offset + start * self_strides[dim];
            let mut out = Tensor {
                storage: self.storage.clone(),
                shape: output_shape.clone(),
                device: self.device.clone(),
                id: TensorId::new(),
                requires_grad: false,
                custom_strides: Some(self_strides.clone()),
                view_offset: new_offset,
            };
            if self.requires_grad {
                out.requires_grad = true;
                if AutogradContext::is_recording() {
                    let mut ranges: Vec<(usize, usize)> =
                        dims.iter().map(|&d| (0, d)).collect();
                    ranges[dim] = (start, start + length);
                    AutogradContext::record_op(
                        out.id,
                        Op::Slice {
                            input: self.id,
                            ranges,
                            input_shape: self.shape.clone(),
                        },
                        Vec::new(),
                    );
                }
            }
            return Ok(out);
        }

        // For ND > 4: collapse dims except `dim` into outer/inner, recurse, reshape back.
        if dims.len() > 4 {
            let outer: usize = dims[..dim].iter().product();
            let inner: usize = dims[dim + 1..].iter().product();
            let mid = dims[dim];
            let collapsed = self.reshape(&[outer, mid, inner])?;
            let narrowed = collapsed.narrow(1, start, length)?; // 3D path
            let mut new_dims = dims.to_vec();
            new_dims[dim] = length;
            return narrowed.reshape(&new_dims);
        }

        // Pad dimensions to 4D for kernel
        let mut input_size = [1, 1, 1, 1];
        let mut output_size = [1, 1, 1, 1];
        for (i, &d) in dims.iter().enumerate() {
            input_size[i] = d;
        }
        for (i, &d) in output_dims.iter().enumerate() {
            output_size[i] = d;
        }

        // Prepare F32 views for the CUDA kernel (kernels currently assume F32 storage)
        use crate::cuda_kernels::CudaKernels;
        let input_f32 = to_owning_fp32_strong(self)?;
        if sdxl_debug_shapes_enabled() {
            eprintln!(
                "[narrow] input logical={:?} storage={:?} dims={:?}",
                input_f32.dtype(),
                input_f32.storage_dtype(),
                input_f32.shape().dims()
            );
        }
        let cuda_kernels = CudaKernels::new(self.device.clone())?;

        // Allocate FP32 output buffer for the kernel
        let output = Tensor::empty_dtype(output_shape.clone(), DType::F32, self.device.clone())?;

        // Get the narrow kernel
        let kernel = cuda_kernels
            .kernels
            .get("narrow_kernel")
            .ok_or_else(|| Error::Cuda("narrow_kernel not found".into()))?
            .clone();

        // Launch kernel
        let n = output.shape.elem_count();
        let block_size = 256;
        let grid_size = n.div_ceil(block_size);

        let config = cudarc::driver::LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel.launch(
                config,
                (
                    input_f32.storage.try_as_slice_f32()?,
                    output.storage.try_as_slice_f32()?,
                    input_size[0] as i32,
                    input_size[1] as i32,
                    input_size[2] as i32,
                    input_size[3] as i32,
                    output_size[0] as i32,
                    output_size[1] as i32,
                    output_size[2] as i32,
                    output_size[3] as i32,
                    dim as i32,
                    start as i32,
                ),
            )?;
        }

        let mut out = if self.dtype() == DType::F32 {
            output
        } else {
            output.to_dtype(self.dtype())?
        };

        // Autograd recording for the F32 slow path. Mirrors the BF16 fast path
        // above. Without this, `Tensor::narrow` silently breaks training-time
        // gradient flow through fused-QKV splits.
        if self.requires_grad {
            out.requires_grad = true;
            if AutogradContext::is_recording() {
                let mut ranges: Vec<(usize, usize)> =
                    dims.iter().map(|&d| (0, d)).collect();
                ranges[dim] = (start, start + length);
                AutogradContext::record_op(
                    out.id,
                    Op::Slice {
                        input: self.id,
                        ranges,
                        input_shape: self.shape.clone(),
                    },
                    Vec::new(),
                );
            }
        }
        Ok(out)
    }

    /// Copy data from another tensor
    pub fn copy_(&mut self, other: &Tensor) -> Result<()> {
        if self.shape != other.shape {
            return Err(Error::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }
        self.storage = other.storage.clone();
        Ok(())
    }
}

/// Implementation of gradient access trait
impl TensorGradExt for Tensor {
    fn grad<'a>(&self, gradients: &'a GradientMap) -> Option<&'a Tensor> {
        gradients.get(self.id)
    }

    fn grad_mut<'a>(&self, gradients: &'a mut GradientMap) -> Option<&'a mut Tensor> {
        gradients.get_mut(self.id)
    }

    fn take_grad(&self, gradients: &mut GradientMap) -> Option<Tensor> {
        gradients.take(self.id)
    }

    fn has_grad(&self, gradients: &GradientMap) -> bool {
        gradients.contains(self.id)
    }
}

/// Allocate memory from the pool for internal use
const MAX_FP32_ALLOC_BYTES: usize = 4 * 1024 * 1024 * 1024;

pub fn guard_fp32_alloc(bytes: usize, tag: &str) {
    if bytes > MAX_FP32_ALLOC_BYTES {
        panic!("[alloc] {tag} requested {bytes} bytes (FP32) — exceeds 1GB cap");
    }
}

pub(crate) fn alloc_from_pool(device: &Arc<CudaDevice>, size: usize) -> Result<CudaSlice<f32>> {
    if alloc_log_enabled() {
        let bytes = size * std::mem::size_of::<f32>();
        if bytes >= (8 << 20) {
            eprintln!("[alloc] tag=pool_alloc_f32 size={} bytes={}", size, bytes);
            if bytes >= 9_437_184 {
                eprintln!("[sentry] FP32_ALLOC bytes={} shape_hint=[?,?]", bytes);
            }
        }
        guard_fp32_alloc(bytes, "pool_alloc_f32");
    } else {
        guard_fp32_alloc(size * std::mem::size_of::<f32>(), "pool_alloc_f32");
    }
    // Use aligned allocation to avoid CUDA alignment issues
    alloc_aligned_f32(device, size)
}

/// Allocate zeroed memory from the pool for internal use
pub(crate) fn alloc_zeros_from_pool(
    device: &Arc<CudaDevice>,
    size: usize,
) -> Result<CudaSlice<f32>> {
    if alloc_log_enabled() {
        let bytes = size * std::mem::size_of::<f32>();
        if bytes >= (8 << 20) {
            eprintln!("[alloc] tag=pool_zeros_f32 size={} bytes={}", size, bytes);
            if bytes >= 9_437_184 {
                eprintln!("[sentry] FP32_ALLOC bytes={} shape_hint=[?,?]", bytes);
            }
        }
        guard_fp32_alloc(bytes, "pool_zeros_f32");
    } else {
        guard_fp32_alloc(size * std::mem::size_of::<f32>(), "pool_zeros_f32");
    }
    let mut data = alloc_from_pool(device, size)?;
    device.memset_zeros(&mut data)?;
    Ok(data)
}

/// Drop: TensorStorage::drop handles returning GPU memory to the
/// caching allocator pool (see `cuda_alloc_pool` module).
impl Drop for Tensor {
    fn drop(&mut self) {
        // TensorStorage::drop returns the CudaSlice to the pool.
        // Nothing else needed here.
    }
}

// --- Unsafe external device pointer construction (placeholder) ---
impl Tensor {
    /// Unsafe constructor to wrap an external device pointer as a Tensor.
    /// Placeholder stub: returns an error until proper external storage support lands.
    ///
    /// # Safety
    /// - `ptr` must reference `shape.elem_count()` contiguous elements of `dtype`
    ///   on `device`.
    /// - The allocation must be correctly aligned for the target dtype and not
    ///   alias any other mutable reference.
    /// - Caller must ensure the pointer remains valid for the lifetime of the
    ///   returned tensor value.
    pub unsafe fn from_device_ptr_unsafe(
        _ptr: *mut u8,
        _shape: Shape,
        _dtype: DType,
        _device: Arc<CudaDevice>,
    ) -> Result<Self> {
        Err(Error::InvalidOperation(
            "from_device_ptr_unsafe not yet implemented".to_string(),
        ))
    }
}

// --- Missing BF16 helper methods for compatibility ---
impl Tensor {
    /// Copy data from a host BF16 slice (u16) into this tensor.
    /// The tensor must be BF16.
    pub fn copy_from_bf16_slice(&mut self, data: &[u16]) -> Result<()> {
        if self.dtype() != DType::BF16 {
            return Err(Error::InvalidOperation(
                "copy_from_bf16_slice requires BF16 tensor".into(),
            ));
        }
        if data.len() != self.shape.elem_count() {
            return Err(Error::ShapeMismatch {
                expected: self.shape.clone(),
                got: Shape::from_dims(&[data.len()]),
            });
        }

        let device = self.device.clone();
        match self.storage_mut() {
            #[cfg(feature = "bf16_u16")]
            TensorStorage::BF16 {
                data: device_data, ..
            } => {
                device
                    .htod_sync_copy_into(data, ensure_unique_slice(device_data)?)
                    .map_err(|_| Error::CudaDriver)?;
                Ok(())
            }
            #[cfg(feature = "bf16_u16")]
            TensorStorage::BF16Arena { ptr, .. } | TensorStorage::BF16View { ptr, .. } => {
                use cudarc::driver::DevicePtrMut;

                let mut staging = unsafe { device.alloc::<u16>(data.len()) }
                    .map_err(|e| Error::Cuda(format!("alloc staging: {:?}", e)))?;
                device
                    .htod_sync_copy_into(data, &mut staging)
                    .map_err(|_| Error::CudaDriver)?;

                let stream = CudaStream::from_raw(device.cuda_stream_raw_ptr());
                bf16_copy_async(
                    ptr.as_ptr() as *mut std::ffi::c_void,
                    (*staging.device_ptr()) as *const std::ffi::c_void,
                    data.len(),
                    &stream,
                )?;
                Ok(())
            }
            #[cfg(not(feature = "bf16_u16"))]
            TensorStorage::BF16 {
                data: device_data, ..
            } => {
                // Convert u16 to f32 on host, then copy
                let mut f32_data = Vec::with_capacity(data.len());
                for &bits in data {
                    let u = (bits as u32) << 16;
                    f32_data.push(f32::from_bits(u));
                }
                device
                    .htod_copy_into(f32_data, device_data)
                    .map_err(|_| Error::CudaDriver)?;
                Ok(())
            }
            _ => Err(Error::InvalidOperation(
                "copy_from_bf16_slice: storage mismatch".into(),
            )),
        }
    }

    /// Create a new BF16 tensor from a host slice of u16 (BF16 bits).
    /// Renamed to avoid conflict with existing from_bf16_slice(CudaSlice).
    pub fn from_bf16_u16_slice(
        data: &[u16],
        shape: Shape,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        if data.len() != shape.elem_count() {
            return Err(Error::ShapeMismatch {
                expected: shape.clone(),
                got: Shape::from_dims(&[data.len()]),
            });
        }

        // Allocate storage
        let mut tensor = Tensor::zeros_dtype(shape, DType::BF16, device)?;
        tensor.copy_from_bf16_slice(data)?;
        Ok(tensor)
    }

    /// Create a new BF16 tensor from a host slice of bytes (interpreted as BF16).
    pub fn from_bf16_bytes(data: &[u8], shape: Shape, device: Arc<CudaDevice>) -> Result<Self> {
        if data.len() % 2 != 0 {
            return Err(Error::InvalidInput(
                "from_bf16_bytes: data length must be even".into(),
            ));
        }
        let u16_len = data.len() / 2;
        if u16_len != shape.elem_count() {
            return Err(Error::ShapeMismatch {
                expected: shape.clone(),
                got: Shape::from_dims(&[u16_len]),
            });
        }

        // Cast bytes to u16
        let u16_data = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u16, u16_len) };

        Self::from_bf16_u16_slice(u16_data, shape, device)
    }

    /// Create a BF16 tensor by loading chunks via a closure.
    pub fn from_bf16_chunks<F>(shape: Shape, device: Arc<CudaDevice>, mut loader: F) -> Result<Self>
    where
        F: FnMut(usize, &mut [u16]) -> Result<()>,
    {
        let numel = shape.elem_count();
        let mut tensor = Tensor::zeros_dtype(shape, DType::BF16, device.clone())?;

        const CHUNK_SIZE: usize = 1024 * 1024 * 16; // 32MB chunks
        let mut host_buf = vec![0u16; CHUNK_SIZE.min(numel)];

        let mut offset = 0;
        while offset < numel {
            let len = (numel - offset).min(host_buf.len());
            let chunk_view = &mut host_buf[0..len];

            // Load data into host buffer
            loader(offset, chunk_view)?;

            // Clone device for closure/match scope
            let device_ref = device.clone();

            match tensor.storage_mut() {
                #[cfg(feature = "bf16_u16")]
                TensorStorage::BF16 { data, .. } => {
                    let mut sub_slice = ensure_unique_slice(data)?.slice_mut(offset..offset + len);
                    device_ref
                        .htod_sync_copy_into(chunk_view, &mut sub_slice)
                        .map_err(|_| Error::CudaDriver)?;
                }
                #[cfg(feature = "bf16_u16")]
                TensorStorage::BF16Arena { ptr, .. } | TensorStorage::BF16View { ptr, .. } => {
                    // Copy via staging
                    let mut staging = unsafe { device_ref.alloc::<u16>(len) }
                        .map_err(|e| Error::Cuda(format!("alloc staging: {:?}", e)))?;
                    device_ref
                        .htod_sync_copy_into(chunk_view, &mut staging)
                        .map_err(|_| Error::CudaDriver)?;

                    let stream = CudaStream::from_raw(device_ref.cuda_stream_raw_ptr());
                    let dst_ptr = unsafe { ptr.as_ptr().add(offset) };

                    bf16_copy_async(
                        dst_ptr as *mut std::ffi::c_void,
                        (*staging.device_ptr()) as *const std::ffi::c_void,
                        len,
                        &stream,
                    )?;
                }
                #[cfg(not(feature = "bf16_u16"))]
                TensorStorage::BF16 { data, .. } => {
                    // Convert and copy
                    let mut f32_chunk = Vec::with_capacity(len);
                    for &bits in chunk_view.iter() {
                        let u = (bits as u32) << 16;
                        f32_chunk.push(f32::from_bits(u));
                    }
                    let mut sub_slice = data.slice_mut(offset..offset + len);
                    device_ref
                        .htod_sync_copy_into(&f32_chunk, &mut sub_slice)
                        .map_err(|_| Error::CudaDriver)?;
                }
                _ => {
                    return Err(Error::InvalidOperation(
                        "from_bf16_chunks: storage mismatch".into(),
                    ))
                }
            }

            offset += len;
        }

        Ok(tensor)
    }
}
