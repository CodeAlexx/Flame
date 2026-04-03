use crate::cuda_memory_alignment::alloc_aligned_f32;
use crate::cuda_ops_ffi::CudaStream;
use crate::device::CudaStreamRawPtrExt;
use crate::staging::bf16_copy_async;
use crate::{DType, Error, Result, Shape};
#[cfg(feature = "shared_storage")]
use cudarc::driver::DeviceRepr;
use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

#[cfg(feature = "shared_storage")]
pub(crate) type StorageSlice<T> = Arc<CudaSlice<T>>;
#[cfg(not(feature = "shared_storage"))]
pub(crate) type StorageSlice<T> = CudaSlice<T>;

#[cfg(feature = "shared_storage")]
#[inline]
pub(crate) fn wrap_slice<T>(slice: CudaSlice<T>) -> StorageSlice<T> {
    Arc::new(slice)
}

#[cfg(not(feature = "shared_storage"))]
#[inline]
pub(crate) fn wrap_slice<T>(slice: CudaSlice<T>) -> StorageSlice<T> {
    slice
}

#[cfg(feature = "shared_storage")]
pub(crate) fn slice_ref<T>(slice: &StorageSlice<T>) -> &CudaSlice<T> {
    slice.as_ref()
}

#[cfg(not(feature = "shared_storage"))]
pub(crate) fn slice_ref<T>(slice: &StorageSlice<T>) -> &CudaSlice<T> {
    slice
}

#[cfg(feature = "shared_storage")]
pub(crate) fn ensure_unique_slice<T: DeviceRepr + Clone>(
    slice: &mut StorageSlice<T>,
) -> Result<&mut CudaSlice<T>> {
    Ok(Arc::make_mut(slice))
}

#[cfg(not(feature = "shared_storage"))]
pub(crate) fn ensure_unique_slice<T>(slice: &mut StorageSlice<T>) -> Result<&mut CudaSlice<T>> {
    Ok(slice)
}

/// Actual storage backend for tensors with proper dtype support
#[derive(Clone)]
pub enum TensorStorage {
    F32 {
        data: StorageSlice<f32>,
        numel: usize,
    },
    F16 {
        data: StorageSlice<f32>,
        numel: usize,
        scale: f32,
    },
    #[cfg(not(feature = "bf16_u16"))]
    BF16 {
        data: StorageSlice<f32>,
        numel: usize,
    },
    #[cfg(feature = "bf16_u16")]
    BF16 {
        data: StorageSlice<u16>,
        numel: usize,
    },
    #[cfg(feature = "bf16_u16")]
    BF16Arena {
        ptr: NonNull<u16>,
        numel: usize,
        device: Arc<CudaDevice>,
        lease: crate::staging::ArenaLease,
    },
    I8 {
        data: StorageSlice<i8>,
        numel: usize,
    },
    I32 {
        data: StorageSlice<f32>,
        numel: usize,
    },
    Bool {
        data: StorageSlice<f32>,
        numel: usize,
    },
}

impl TensorStorage {
    /// Get the dtype of this storage
    pub fn dtype(&self) -> DType {
        match self {
            TensorStorage::F32 { .. } => DType::F32,
            TensorStorage::F16 { .. } => DType::F16,
            TensorStorage::BF16 { .. } | TensorStorage::BF16Arena { .. } => DType::BF16,
            TensorStorage::I8 { .. } => DType::I8,
            TensorStorage::I32 { .. } => DType::I32,
            TensorStorage::Bool { .. } => DType::Bool,
        }
    }

    /// Get number of elements
    pub fn len(&self) -> usize {
        match self {
            TensorStorage::F32 { numel, .. } => *numel,
            TensorStorage::F16 { numel, .. } => *numel,
            TensorStorage::BF16 { numel, .. } | TensorStorage::BF16Arena { numel, .. } => *numel,
            TensorStorage::I8 { numel, .. } => *numel,
            TensorStorage::I32 { numel, .. } => *numel,
            TensorStorage::Bool { numel, .. } => *numel,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Allocate new storage using memory pool
    pub fn zeros(shape: &Shape, dtype: DType, device: &Arc<CudaDevice>) -> Result<Self> {
        let numel = shape.elem_count();

        match dtype {
            DType::F32 => {
                // Use aligned allocation for F32
                let mut data = alloc_aligned_f32(device, numel)?;
                device.memset_zeros(&mut data)?;
                Ok(TensorStorage::F32 {
                    data: wrap_slice(data),
                    numel,
                })
            }
            DType::F16 => {
                // F16 still uses F32 storage with scale
                let mut data = alloc_aligned_f32(device, numel)?;
                device.memset_zeros(&mut data)?;
                Ok(TensorStorage::F16 {
                    data: wrap_slice(data),
                    numel,
                    scale: 1.0,
                })
            }
            DType::BF16 => {
                #[cfg(not(feature = "bf16_u16"))]
                {
                    let mut data = alloc_aligned_f32(device, numel)?;
                    device.memset_zeros(&mut data)?;
                    Ok(TensorStorage::BF16 {
                        data: wrap_slice(data),
                        numel,
                    })
                }
                #[cfg(feature = "bf16_u16")]
                {
                    let mut data = unsafe { device.alloc::<u16>(numel) }
                        .map_err(|e| Error::Cuda(format!("alloc bf16 u16: {}", e)))?;
                    device.memset_zeros(&mut data)?;
                    Ok(TensorStorage::BF16 {
                        data: wrap_slice(data),
                        numel,
                    })
                }
            }
            DType::I8 => {
                // For I8, we need to allocate i8 storage
                Err(Error::InvalidOperation(
                    "I8 allocation not yet supported in zeros - use quantization functions".into(),
                ))
            }
            DType::I32 => {
                let mut data = alloc_aligned_f32(device, numel)?;
                device.memset_zeros(&mut data)?;
                Ok(TensorStorage::I32 {
                    data: wrap_slice(data),
                    numel,
                })
            }
            DType::Bool => {
                let mut data = alloc_aligned_f32(device, numel)?;
                device.memset_zeros(&mut data)?;
                Ok(TensorStorage::Bool {
                    data: wrap_slice(data),
                    numel,
                })
            }
            DType::F64 | DType::U8 | DType::U32 | DType::I64 => Err(Error::InvalidOperation(
                format!("Unsupported dtype in TensorStorage: {:?}", dtype),
            )),
        }
    }

    /// Convert to F32 (for operations that don't support F16/BF16)
    pub fn to_f32(&self, device: &Arc<CudaDevice>) -> Result<CudaSlice<f32>> {
        match self {
            TensorStorage::F32 { data, numel } | TensorStorage::F16 { data, numel, .. } => {
                // Use aligned allocation
                let mut out = alloc_aligned_f32(device, *numel)?;

                // If the allocation is larger, we need to handle it carefully
                if out.len() > *numel {
                    eprintln!(
                        "Warning: aligned allocation returned {} elements for {} requested",
                        out.len(),
                        *numel
                    );
                }

                // Copy data - dtod_copy should handle size mismatches gracefully
                device.dtod_copy(slice_ref(data), &mut out)?;
                Ok(out)
            }
            TensorStorage::BF16 { data, numel } => {
                #[cfg(not(feature = "bf16_u16"))]
                {
                    let mut out = alloc_aligned_f32(device, *numel)?;
                    if out.len() > *numel {
                        eprintln!(
                            "Warning: aligned allocation returned {} elements for {} requested",
                            out.len(),
                            *numel
                        );
                    }
                    device.dtod_copy(slice_ref(data), &mut out)?;
                    Ok(out)
                }
                #[cfg(feature = "bf16_u16")]
                {
                    use cudarc::driver::DevicePtr;
                    // Convert u16 BF16 → f32 via NVRTC kernel
                    let mut out = alloc_aligned_f32(device, *numel)?;
                    // Launch conversion kernel via helper
                    crate::bf16_convert::bf16_u16_to_f32(
                        device.clone(),
                        *slice_ref(data).device_ptr(),
                        &mut out,
                        *numel,
                    )?;
                    Ok(out)
                }
            }
            TensorStorage::BF16Arena {
                ptr,
                numel,
                device: arena_device,
                ..
            } => {
                #[cfg(not(feature = "bf16_u16"))]
                {
                    let mut out = alloc_aligned_f32(device, *numel)?;
                    device.memset_zeros(&mut out)?;
                    Ok(out)
                }
                #[cfg(feature = "bf16_u16")]
                {
                    use cudarc::driver::{DevicePtr, DevicePtrMut};
                    let mut staging = unsafe { arena_device.alloc::<u16>(*numel) }
                        .map_err(|e| Error::Cuda(format!("alloc bf16 arena staging: {}", e)))?;
                    let stream = CudaStream::from_raw(arena_device.cuda_stream_raw_ptr());
                    bf16_copy_async(
                        (*staging.device_ptr_mut()) as *mut c_void,
                        ptr.as_ptr() as *const c_void,
                        *numel,
                        &stream,
                    )?;
                    let mut out = alloc_aligned_f32(device, *numel)?;
                    crate::bf16_convert::bf16_u16_to_f32(
                        arena_device.clone(),
                        *staging.device_ptr(),
                        &mut out,
                        *numel,
                    )?;
                    Ok(out)
                }
            }
            TensorStorage::I8 { .. } => Err(Error::InvalidOperation(
                "I8 to F32 conversion not yet implemented".into(),
            )),
            TensorStorage::I32 { data, numel } | TensorStorage::Bool { data, numel } => {
                let mut out = alloc_aligned_f32(device, *numel)?;
                if out.len() > *numel {
                    eprintln!(
                        "Warning: aligned allocation returned {} elements for {} requested",
                        out.len(),
                        *numel
                    );
                }
                device.dtod_copy(slice_ref(data), &mut out)?;
                Ok(out)
            }
        }
    }

    /// Safe: get read-only f32 slice for f32-backed storage. Otherwise Err.
    #[track_caller]
    pub fn try_as_slice_f32(&self) -> Result<&CudaSlice<f32>> {
        match self {
            TensorStorage::F32 { data, .. } => Ok(slice_ref(data)),
            TensorStorage::F16 { data, .. } => Ok(slice_ref(data)),
            #[cfg(not(feature = "bf16_u16"))]
            TensorStorage::BF16 { data, .. } => Ok(slice_ref(data)),
            TensorStorage::I32 { data, .. } => Ok(slice_ref(data)),
            TensorStorage::Bool { data, .. } => Ok(slice_ref(data)),
            #[cfg(feature = "bf16_u16")]
            TensorStorage::BF16 { numel, .. } => {
                if std::env::var("SDXL_DEBUG_SHAPES").ok().as_deref() == Some("1") {
                    let bt = std::backtrace::Backtrace::capture();
                    eprintln!(
                        "[try_as_slice_f32] BF16 storage encountered len={numel}\n{:?}",
                        bt
                    );
                }
                Err(Error::InvalidInput(
                    format!("expected F32 slice, got BF16(u16) (len={})", numel).into(),
                ))
            }
            #[cfg(feature = "bf16_u16")]
            TensorStorage::BF16Arena { numel, .. } => Err(Error::InvalidInput(
                format!(
                    "expected F32 slice, got arena-backed BF16(u16) (len={})",
                    numel
                )
                .into(),
            )),
            TensorStorage::I8 { .. } => {
                Err(Error::InvalidInput("expected F32 slice, got I8".into()))
            }
        }
    }

    /// Safe: get mutable f32 slice for f32-backed storage. Otherwise Err.
    pub fn try_as_mut_slice_f32(&mut self) -> Result<&mut CudaSlice<f32>> {
        match self {
            TensorStorage::F32 { data, .. } => ensure_unique_slice(data),
            TensorStorage::F16 { data, .. } => ensure_unique_slice(data),
            #[cfg(not(feature = "bf16_u16"))]
            TensorStorage::BF16 { data, .. } => ensure_unique_slice(data),
            TensorStorage::I32 { data, .. } => ensure_unique_slice(data),
            TensorStorage::Bool { data, .. } => ensure_unique_slice(data),
            #[cfg(feature = "bf16_u16")]
            TensorStorage::BF16 { .. } => Err(Error::InvalidInput(
                "expected F32 slice, got BF16(u16)".into(),
            )),
            #[cfg(feature = "bf16_u16")]
            TensorStorage::BF16Arena { .. } => Err(Error::InvalidInput(
                "expected F32 slice, got arena-backed BF16(u16)".into(),
            )),
            TensorStorage::I8 { .. } => {
                Err(Error::InvalidInput("expected F32 slice, got I8".into()))
            }
        }
    }

    /// Deprecated: use try_as_slice_f32() and handle Result.
    #[allow(clippy::expect_used)]
    #[deprecated(note = "use try_as_slice_f32() and handle Result")]
    pub fn as_slice(&self) -> &CudaSlice<f32> {
        self.try_as_slice_f32()
            .expect("TensorStorage::as_slice() panicked; migrate to try_as_slice_f32()")
    }

    /// Safe: get read-only u16 slice for BF16(u16) storage. Otherwise Err.
    #[cfg(feature = "bf16_u16")]
    pub fn try_as_slice_u16(&self) -> Result<&CudaSlice<u16>> {
        match self {
            TensorStorage::BF16 { data, .. } => Ok(slice_ref(data)),
            TensorStorage::BF16Arena { .. } => Err(Error::InvalidOperation(
                "expected owning BF16 storage, got arena-backed BF16".into(),
            )),
            TensorStorage::F32 { .. } => Err(Error::InvalidInput(
                "expected BF16(u16) slice, got F32".into(),
            )),
            TensorStorage::F16 { .. } => Err(Error::InvalidInput(
                "expected BF16(u16) slice, got F16".into(),
            )),
            TensorStorage::I32 { .. } => Err(Error::InvalidInput(
                "expected BF16(u16) slice, got I32".into(),
            )),
            TensorStorage::Bool { .. } => Err(Error::InvalidInput(
                "expected BF16(u16) slice, got Bool".into(),
            )),
            TensorStorage::I8 { .. } => Err(Error::InvalidInput(
                "expected BF16(u16) slice, got I8".into(),
            )),
        }
    }

    /// Safe: get mutable u16 slice for BF16(u16) storage. Otherwise Err.
    #[cfg(feature = "bf16_u16")]
    pub fn try_as_mut_slice_u16(&mut self) -> Result<&mut CudaSlice<u16>> {
        match self {
            TensorStorage::BF16 { data, .. } => ensure_unique_slice(data),
            TensorStorage::BF16Arena { .. } => Err(Error::InvalidOperation(
                "expected owning BF16 storage, got arena-backed BF16".into(),
            )),
            TensorStorage::F32 { .. } => Err(Error::InvalidInput(
                "expected BF16(u16) slice, got F32".into(),
            )),
            TensorStorage::F16 { .. } => Err(Error::InvalidInput(
                "expected BF16(u16) slice, got F16".into(),
            )),
            TensorStorage::I32 { .. } => Err(Error::InvalidInput(
                "expected BF16(u16) slice, got I32".into(),
            )),
            TensorStorage::Bool { .. } => Err(Error::InvalidInput(
                "expected BF16(u16) slice, got Bool".into(),
            )),
            TensorStorage::I8 { .. } => Err(Error::InvalidInput(
                "expected BF16(u16) slice, got I8".into(),
            )),
        }
    }

    /// Deprecated: use try_as_slice_u16() and handle Result.
    #[allow(clippy::expect_used)]
    #[deprecated(note = "use try_as_slice_u16() and handle Result")]
    #[cfg(feature = "bf16_u16")]
    pub fn as_slice_u16(&self) -> &CudaSlice<u16> {
        self.try_as_slice_u16()
            .expect("TensorStorage::as_slice_u16() panicked; migrate to try_as_slice_u16()")
    }

    /// Get a reference to the underlying I8 CudaSlice
    pub fn as_i8_slice(&self) -> Result<&CudaSlice<i8>> {
        match self {
            TensorStorage::I8 { data, .. } => Ok(slice_ref(data)),
            _ => Err(Error::InvalidOperation("Not an I8 tensor".into())),
        }
    }

    /// Safe: get read-only i32 slice when storage uses I32.
    pub fn try_as_slice_i32(&self) -> Result<&CudaSlice<i32>> {
        match self {
            TensorStorage::I32 { data, .. } => {
                let ptr = slice_ref(data) as *const CudaSlice<f32> as *const CudaSlice<i32>;
                Ok(unsafe { &*ptr })
            }
            _ => Err(Error::InvalidInput("expected I32 slice".into())),
        }
    }

    /// Safe: get mutable i32 slice when storage uses I32.
    pub fn try_as_mut_slice_i32(&mut self) -> Result<&mut CudaSlice<i32>> {
        match self {
            TensorStorage::I32 { data, .. } => {
                let slice = ensure_unique_slice(data)?;
                let ptr = slice as *mut CudaSlice<f32> as *mut CudaSlice<i32>;
                Ok(unsafe { &mut *ptr })
            }
            _ => Err(Error::InvalidInput("expected I32 slice".into())),
        }
    }
}

#[cfg(feature = "bf16_u16")]
impl TensorStorage {
    pub fn to_vec_bf16(&self, device: &Arc<CudaDevice>) -> Result<Vec<u16>> {
        match self {
            TensorStorage::BF16 { data, .. } => device
                .dtoh_sync_copy(slice_ref(data))
                .map_err(|_| Error::CudaDriver),
            TensorStorage::BF16Arena {
                ptr,
                numel,
                device: arena_device,
                ..
            } => {
                use cudarc::driver::DevicePtrMut;
                let mut staging = unsafe { arena_device.alloc::<u16>(*numel) }
                    .map_err(|e| Error::Cuda(format!("alloc bf16 arena staging: {}", e)))?;
                let stream = CudaStream::from_raw(arena_device.cuda_stream_raw_ptr());
                bf16_copy_async(
                    (*staging.device_ptr_mut()) as *mut c_void,
                    ptr.as_ptr() as *const c_void,
                    *numel,
                    &stream,
                )?;
                arena_device
                    .dtoh_sync_copy(&staging)
                    .map_err(|_| Error::CudaDriver)
            }
            _ => Err(Error::InvalidOperation(
                "to_vec_bf16: tensor storage is not BF16".into(),
            )),
        }
    }
}

#[cfg(not(feature = "bf16_u16"))]
impl TensorStorage {
    pub fn to_vec_bf16(&self, _device: &Arc<CudaDevice>) -> Result<Vec<u16>> {
        Err(Error::InvalidOperation(
            "to_vec_bf16 requires the bf16_u16 feature".into(),
        ))
    }
}

unsafe impl Send for TensorStorage {}
unsafe impl Sync for TensorStorage {}

// Note: F16/BF16 conversion kernels can be specialized further; current path stores as F32-backed buffers.
// For now, we store everything as F32 but track the intended dtype for API compatibility

/// Clear the global BF16 memory pool.
/// This is a helper to manually release cached memory.
pub fn clear_bf16_pool() {
    crate::memory_pool::MEMORY_POOL.clear_all_caches();
}
