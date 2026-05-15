use crate::cuda_memory_alignment::alloc_aligned_f32;
use crate::cuda_ops_ffi::CudaStream;
use crate::device::CudaStreamRawPtrExt;
use crate::staging::bf16_copy_async_tagged;
use crate::{DType, Error, Result, Shape};
#[cfg(feature = "shared_storage")]
use cudarc::driver::DeviceRepr;
use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Phase 2 — version-counter side table
// ---------------------------------------------------------------------------
//
// PyTorch's `SavedVariable` snapshots a `version_counter_` on the underlying
// `StorageImpl` at save-time and re-checks at unpack-time; if the live counter
// has advanced, the saved tensor was modified in-place and backward would be
// silently wrong. flame-core's `TensorStorage` is an enum cloned by value
// (each `Tensor` clone makes a sibling enum holding the same inner `Arc`),
// so adding `AtomicU32` to the variants directly does NOT share the counter
// across clones. Instead we keep a global side-table keyed by the inner
// `Arc<CudaSlice<T>>` pointer (or raw `*const u16` for BF16Arena / BF16View).
//
// Pointer values are unique per live allocation: when the backing Arc is
// dropped and its pointer is later reused for a fresh allocation, the new
// allocation should start at version 0, which is automatic because the prior
// entry is removed in `unregister`. Caller responsibility:
//
//   * Storage construction sites call `register_version(...)` at the moment
//     a fresh backing allocation appears (real `cudaMalloc` / pool checkout /
//     arena lease) so version starts at 0.
//   * Storage *Drop* impl calls `unregister_version(...)` to release the
//     entry when the last clone goes away.
//   * In-place mutators call `TensorStorage::bump_version()` after writing.
//   * `TensorStorage::version()` returns the current counter (0 if not
//     registered — i.e. legacy callers that haven't been migrated).
//
// The side table is a `RwLock<HashMap>` keyed by `usize` (pointer cast). Most
// reads come from the autograd hot path; we use Relaxed atomics inside the
// `AtomicU32` and an `RwLock` for the map so concurrent reads don't contend.
// Inserts/removes are rare (per real allocation, not per clone) so the writer
// lock is acceptable.

use std::collections::HashMap;
use std::sync::RwLock;

/// Global version-counter table keyed by storage-pointer address.
static VERSION_TABLE: once_cell::sync::Lazy<RwLock<HashMap<usize, Arc<AtomicU32>>>> =
    once_cell::sync::Lazy::new(|| RwLock::new(HashMap::with_capacity(4096)));

/// Register a fresh allocation in the version table. Returns the new counter
/// (always starts at 0). Idempotent: re-registering an existing key returns
/// the existing counter (preserves cross-clone sharing semantics).
#[inline]
pub(crate) fn register_version(key: usize) -> Arc<AtomicU32> {
    if key == 0 {
        return Arc::new(AtomicU32::new(0));
    }
    if let Ok(table) = VERSION_TABLE.read() {
        if let Some(existing) = table.get(&key) {
            return existing.clone();
        }
    }
    if let Ok(mut table) = VERSION_TABLE.write() {
        // Re-check under the writer lock.
        return table
            .entry(key)
            .or_insert_with(|| Arc::new(AtomicU32::new(0)))
            .clone();
    }
    Arc::new(AtomicU32::new(0))
}

/// Remove a key from the version table. Called from `TensorStorage::Drop`
/// when the last clone of a storage releases the backing allocation. Safe
/// to call with an unknown key (no-op).
#[inline]
pub(crate) fn unregister_version(key: usize) {
    if key == 0 {
        return;
    }
    if let Ok(mut table) = VERSION_TABLE.write() {
        let _ = table.remove(&key);
    }
}

/// Clear the entire version-counter side table. Called by
/// `AutogradContext::clear()` at the end of each training step to avoid
/// unbounded growth. Any SavedRef captured before the clear keeps a private
/// `Arc<AtomicU32>` clone, so its version check remains valid — clearing
/// the table only affects future captures, which will lazily re-register.
pub fn clear_version_table() {
    if let Ok(mut table) = VERSION_TABLE.write() {
        table.clear();
    }
}

/// Number of entries currently in the version-counter table. Test/diag only.
pub fn version_table_len() -> usize {
    VERSION_TABLE.read().map(|t| t.len()).unwrap_or(0)
}

/// Look up the current version counter for a storage key. Returns `None` if
/// not registered.
#[inline]
pub(crate) fn lookup_version(key: usize) -> Option<Arc<AtomicU32>> {
    if key == 0 {
        return None;
    }
    if let Ok(table) = VERSION_TABLE.read() {
        return table.get(&key).cloned();
    }
    None
}

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

/// Pointer-derived key for the version-counter side-table.
/// With `shared_storage` (default): the Arc address (stable across clones).
/// Without: the CudaSlice's device pointer.
#[cfg(feature = "shared_storage")]
#[inline]
pub(crate) fn storage_key_t<T>(s: &StorageSlice<T>) -> usize {
    Arc::as_ptr(s) as *const () as usize
}

#[cfg(not(feature = "shared_storage"))]
#[inline]
pub(crate) fn storage_key_t<T>(s: &StorageSlice<T>) -> usize {
    use cudarc::driver::DevicePtr;
    *s.device_ptr() as usize
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
    /// Non-owning view into a shared GPU buffer. Does NOT free on drop.
    /// The caller must guarantee the backing buffer outlives all views.
    #[cfg(feature = "bf16_u16")]
    BF16View { ptr: NonNull<u16>, numel: usize },
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
    /// Return a unique-per-allocation key used to look up the version counter
    /// in the global side table. For owning variants this is the inner Arc
    /// pointer (cast to `usize`); for non-owning BF16View it's the raw u16
    /// pointer; for BF16Arena it's the lease pointer.
    ///
    /// Pointer reuse is correct because the Drop impl removes the entry
    /// before the allocation can be re-handed-out, so a fresh allocation
    /// starting at the same address sees no stale counter.
    #[inline]
    pub(crate) fn version_key(&self) -> usize {
        match self {
            TensorStorage::F32 { data, .. } => storage_key_t(data),
            TensorStorage::F16 { data, .. } => storage_key_t(data),
            #[cfg(not(feature = "bf16_u16"))]
            TensorStorage::BF16 { data, .. } => storage_key_t(data),
            #[cfg(feature = "bf16_u16")]
            TensorStorage::BF16 { data, .. } => storage_key_t(data),
            #[cfg(feature = "bf16_u16")]
            TensorStorage::BF16Arena { ptr, .. } => ptr.as_ptr() as usize,
            #[cfg(feature = "bf16_u16")]
            TensorStorage::BF16View { ptr, .. } => ptr.as_ptr() as usize,
            TensorStorage::I8 { data, .. } => storage_key_t(data),
            TensorStorage::I32 { data, .. } => storage_key_t(data),
            TensorStorage::Bool { data, .. } => storage_key_t(data),
        }
    }

    /// Current version counter for this storage. Lazily-registers if the
    /// allocation site missed the registration (returns 0 for unregistered
    /// storages, which is the legacy behavior — no in-place detection).
    #[inline]
    pub fn version(&self) -> u32 {
        let key = self.version_key();
        match lookup_version(key) {
            Some(arc) => arc.load(Ordering::Relaxed),
            None => 0,
        }
    }

    /// Lazily-fetch (or create) the version counter Arc for this storage.
    /// `SavedRef::capture` calls this so it can snapshot+share the counter
    /// without re-doing the side-table lookup at unpack time.
    #[inline]
    pub(crate) fn version_handle(&self) -> Arc<AtomicU32> {
        register_version(self.version_key())
    }

    /// Bump the storage's version counter. Called from in-place mutators.
    /// Safe to call before any registration — registers on first call.
    #[inline]
    pub fn bump_version(&self) {
        let key = self.version_key();
        let arc = register_version(key);
        arc.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the dtype of this storage
    pub fn dtype(&self) -> DType {
        match self {
            TensorStorage::F32 { .. } => DType::F32,
            TensorStorage::F16 { .. } => DType::F16,
            TensorStorage::BF16 { .. }
            | TensorStorage::BF16Arena { .. }
            | TensorStorage::BF16View { .. } => DType::BF16,
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
            TensorStorage::BF16 { numel, .. }
            | TensorStorage::BF16Arena { numel, .. }
            | TensorStorage::BF16View { numel, .. } => *numel,
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
    /// Allocate storage without zeroing, for use by kernels that fully
    /// overwrite the output. For the `bf16_u16` path this skips an
    /// explicit `memset_zeros` pass over the buffer — measurable savings
    /// on large activations (a 1 GB BF16 output saves ~2 ms per call at
    /// HBM bandwidth). For F32/F16/I32 paths the underlying allocator
    /// (`alloc_aligned_f32` → `device.alloc_zeros`) always zeros, so
    /// this is semantically identical to `zeros` on those dtypes and
    /// just saves the redundant second zero.
    ///
    /// **Contract:** only use for output tensors that are FULLY written
    /// by the kernel that follows. Never use for accumulator tensors or
    /// partial writes — the initial bytes are undefined.
    pub fn empty(shape: &Shape, dtype: DType, device: &Arc<CudaDevice>) -> Result<Self> {
        let numel = shape.elem_count();
        match dtype {
            DType::F32 => {
                let data = alloc_aligned_f32(device, numel)?;
                Ok(TensorStorage::F32 {
                    data: wrap_slice(data),
                    numel,
                })
            }
            DType::F16 => {
                let data = alloc_aligned_f32(device, numel)?;
                Ok(TensorStorage::F16 {
                    data: wrap_slice(data),
                    numel,
                    scale: 1.0,
                })
            }
            DType::BF16 => {
                #[cfg(not(feature = "bf16_u16"))]
                {
                    let data = alloc_aligned_f32(device, numel)?;
                    Ok(TensorStorage::BF16 {
                        data: wrap_slice(data),
                        numel,
                    })
                }
                #[cfg(feature = "bf16_u16")]
                {
                    // Raw alloc via caching pool, no memset — kernel must fully write.
                    let data = crate::cuda_alloc_pool::pool_alloc_u16(device, numel)?;
                    Ok(TensorStorage::BF16 {
                        data: wrap_slice(data),
                        numel,
                    })
                }
            }
            DType::I32 => {
                let data = alloc_aligned_f32(device, numel)?;
                Ok(TensorStorage::I32 {
                    data: wrap_slice(data),
                    numel,
                })
            }
            // Rare dtypes: fall through to zeros semantics.
            DType::I8 | DType::Bool | DType::F64 | DType::U8 | DType::U32 | DType::I64 => {
                Self::zeros(shape, dtype, device)
            }
        }
    }

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
                    let mut data = crate::cuda_alloc_pool::pool_alloc_u16(device, numel)?;
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
                        .map_err(|e| Error::Cuda(format!("alloc bf16 arena staging: {:?}", e)))?;
                    let stream = CudaStream::from_raw(arena_device.cuda_stream_raw_ptr());
                    bf16_copy_async_tagged(
                        (*staging.device_ptr_mut()) as *mut c_void,
                        ptr.as_ptr() as *const c_void,
                        *numel,
                        &stream,
                        "as_f32_slice:BF16Arena",
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
            #[cfg(feature = "bf16_u16")]
            TensorStorage::BF16View { ptr, numel } => {
                use cudarc::driver::DevicePtr;
                // BF16View: convert via direct bf16→f32 kernel from raw pointer
                let mut out = alloc_aligned_f32(device, *numel)?;
                crate::bf16_convert::bf16_u16_to_f32(
                    device.clone(),
                    ptr.as_ptr() as u64,
                    &mut out,
                    *numel,
                )?;
                Ok(out)
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
            #[cfg(feature = "bf16_u16")]
            TensorStorage::BF16View { numel, .. } => Err(Error::InvalidInput(
                format!(
                    "expected F32 slice, got view-backed BF16(u16) (len={})",
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
            #[cfg(feature = "bf16_u16")]
            TensorStorage::BF16View { .. } => Err(Error::InvalidInput(
                "expected F32 slice, got view-backed BF16(u16)".into(),
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
            TensorStorage::BF16View { .. } => Err(Error::InvalidOperation(
                "expected owning BF16 storage, got view-backed BF16".into(),
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
            TensorStorage::BF16View { .. } => Err(Error::InvalidOperation(
                "expected owning BF16 storage, got view-backed BF16".into(),
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
                .map_err(|e| Error::CudaDriver(format!("{e:?}"))),
            TensorStorage::BF16Arena {
                ptr,
                numel,
                device: arena_device,
                ..
            } => {
                use cudarc::driver::DevicePtrMut;
                let mut staging = unsafe { arena_device.alloc::<u16>(*numel) }
                    .map_err(|e| Error::Cuda(format!("alloc bf16 arena staging: {:?}", e)))?;
                let stream = CudaStream::from_raw(arena_device.cuda_stream_raw_ptr());
                bf16_copy_async_tagged(
                    (*staging.device_ptr_mut()) as *mut c_void,
                    ptr.as_ptr() as *const c_void,
                    *numel,
                    &stream,
                    "to_vec_bf16:BF16Arena",
                )?;
                arena_device
                    .dtoh_sync_copy(&staging)
                    .map_err(|e| Error::CudaDriver(format!("{e:?}")))
            }
            TensorStorage::BF16View { ptr, numel } => {
                use cudarc::driver::DevicePtrMut;
                // View: copy from raw ptr to staging, then dtoh
                let mut staging = unsafe { device.alloc::<u16>(*numel) }
                    .map_err(|e| Error::Cuda(format!("alloc bf16 view staging: {:?}", e)))?;
                let stream = CudaStream::from_raw(device.cuda_stream_raw_ptr());
                bf16_copy_async_tagged(
                    (*staging.device_ptr_mut()) as *mut c_void,
                    ptr.as_ptr() as *const c_void,
                    *numel,
                    &stream,
                    "to_vec_bf16:BF16View",
                )?;
                device
                    .dtoh_sync_copy(&staging)
                    .map_err(|e| Error::CudaDriver(format!("{e:?}")))
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

// ---------------------------------------------------------------------------
// Drop: return owned CudaSlice memory to the caching allocator pool
// ---------------------------------------------------------------------------
//
// When a TensorStorage drops, Rust calls Drop::drop and then drops each
// variant field.  We intercept by ptr::read-ing the CudaSlice out and
// handing it to the pool (which mem::forgets it to skip cudaFree), then
// writing a dummy CudaSlice back so the subsequent Rust field drop is
// harmless (cudaFree(0) is a documented CUDA no-op, and the dummy carries
// a valid Arc<CudaDevice> clone so Arc::drop won't segfault).
//
// Mirror struct matching cudarc 0.11.9 CudaSlice<T> layout.
// Must NOT be #[repr(C)] — must match CudaSlice's default Rust layout.
struct CudaSliceMirror<T> {
    cu_device_ptr: u64,
    len: usize,
    device: Arc<CudaDevice>,
    host_buf: Option<std::pin::Pin<Vec<T>>>,
}

/// Create a dummy CudaSlice<T> with ptr=0, len=0, and the given device.
/// When Rust drops this, cudarc calls cudaFree(0) which is a no-op.
///
/// # Safety
/// Only valid for the purpose of replacing a ptr::read-moved field so
/// Rust's field-drop doesn't touch freed memory.
unsafe fn make_dummy_slice<T>(device: Arc<CudaDevice>) -> CudaSlice<T> {
    let mirror = CudaSliceMirror::<T> {
        cu_device_ptr: 0,
        len: 0,
        device,
        host_buf: None,
    };
    std::mem::transmute(mirror)
}

/// R1b integration: check whether a slice's ptr is owned by a slab
/// (`static_slab_v2`). If so, decrement the slab's `live_count`, forget
/// the slice (no cudaFree), and return `true` so the caller skips the
/// rest of pool-return.
///
/// `is_u16` is used only for the BF16 live-ptr trap. The slab itself tracks
/// live_count irrespective of element type.
#[cfg_attr(not(feature = "shared_storage"), inline)]
fn slab_v2_try_claim<T>(slice: &CudaSlice<T>, is_u16: bool) -> bool {
    use cudarc::driver::DevicePtr;
    let ptr = *slice.device_ptr();
    let len = DeviceSlice::len(slice);
    let key = Arc::as_ptr(&slice.device()) as usize;
    let claimed = crate::static_slab_v2::slab_v2_return_if_owned(ptr, key);
    if claimed && is_u16 {
        crate::cuda_alloc_pool::trap_record_bf16_released(
            ptr,
            len,
            "static_slab_v2/TensorStorage::drop",
            0,
        );
    }
    claimed
}

impl Drop for TensorStorage {
    fn drop(&mut self) {
        // Phase 2: best-effort cleanup of the version-counter entry. We only
        // unregister when this is the LAST clone of an owning variant (Arc
        // strong_count == 1 inside the match arms below — implicitly the
        // case when `Arc::try_unwrap` succeeds for pool-return). For shared
        // storage where another clone is live we leave the entry in place
        // so other clones still see consistent versioning. Side table is
        // also flushed at AutogradContext::clear() (per training step).

        // R1b-bf: do NOT short-circuit on `pool_disabled()` at the top level.
        // The slab live-count decrement (`slab_v2_try_claim`) MUST run for
        // slab-owned F32/BF16 tensors regardless of pool state — otherwise
        // when `FLAME_ALLOC_POOL=0` is paired with `FLAME_USE_STATIC_SLAB=1`,
        // `live_count` leaks 1 per drop and `StaticSlabAllocator::reset()`
        // fails forever. The pool-return short-circuit is moved INTO each
        // arm below, after the slab check.
        let pool_off = crate::cuda_alloc_pool::pool_disabled();

        match self {
            TensorStorage::F32 { data, .. } => {
                #[cfg(feature = "shared_storage")]
                {
                    let arc: Arc<CudaSlice<f32>> = unsafe { std::ptr::read(data) };
                    let dev = arc.device();
                    match Arc::try_unwrap(arc) {
                        Ok(slice) => {
                            // R1b: slab claim ALWAYS runs (pool_off-agnostic).
                            if slab_v2_try_claim(&slice, false) {
                                std::mem::forget(slice);
                            } else if pool_off {
                                // Not slab-owned + pool disabled: let
                                // cudarc fire cudaFree naturally.
                                drop(slice);
                            } else {
                                crate::cuda_alloc_pool::pool_return_f32(slice);
                            }
                        }
                        Err(arc) => drop(arc),
                    }
                    unsafe { std::ptr::write(data, Arc::new(make_dummy_slice::<f32>(dev))) };
                }
                #[cfg(not(feature = "shared_storage"))]
                {
                    let slice: CudaSlice<f32> = unsafe { std::ptr::read(data) };
                    let dev = slice.device();
                    if slab_v2_try_claim(&slice, false) {
                        std::mem::forget(slice);
                    } else if pool_off {
                        drop(slice);
                    } else {
                        crate::cuda_alloc_pool::pool_return_f32(slice);
                    }
                    unsafe { std::ptr::write(data, make_dummy_slice::<f32>(dev)) };
                }
            }
            TensorStorage::F16 { data, .. } => {
                // F16 has no slab path — preserve legacy pool_off short-circuit.
                if pool_off {
                    return;
                }
                #[cfg(feature = "shared_storage")]
                {
                    let arc: Arc<CudaSlice<f32>> = unsafe { std::ptr::read(data) };
                    let dev = arc.device();
                    match Arc::try_unwrap(arc) {
                        Ok(slice) => crate::cuda_alloc_pool::pool_return_f32(slice),
                        Err(arc) => drop(arc),
                    }
                    unsafe { std::ptr::write(data, Arc::new(make_dummy_slice::<f32>(dev))) };
                }
                #[cfg(not(feature = "shared_storage"))]
                {
                    let slice: CudaSlice<f32> = unsafe { std::ptr::read(data) };
                    let dev = slice.device();
                    crate::cuda_alloc_pool::pool_return_f32(slice);
                    unsafe { std::ptr::write(data, make_dummy_slice::<f32>(dev)) };
                }
            }
            #[cfg(not(feature = "bf16_u16"))]
            TensorStorage::BF16 { data, .. } => {
                // Legacy F32-backed BF16: no slab path (slab is bf16_u16-only).
                if pool_off {
                    return;
                }
                #[cfg(feature = "shared_storage")]
                {
                    let arc: Arc<CudaSlice<f32>> = unsafe { std::ptr::read(data) };
                    let dev = arc.device();
                    match Arc::try_unwrap(arc) {
                        Ok(slice) => crate::cuda_alloc_pool::pool_return_f32(slice),
                        Err(arc) => drop(arc),
                    }
                    unsafe { std::ptr::write(data, Arc::new(make_dummy_slice::<f32>(dev))) };
                }
                #[cfg(not(feature = "shared_storage"))]
                {
                    let slice: CudaSlice<f32> = unsafe { std::ptr::read(data) };
                    let dev = slice.device();
                    crate::cuda_alloc_pool::pool_return_f32(slice);
                    unsafe { std::ptr::write(data, make_dummy_slice::<f32>(dev)) };
                }
            }
            #[cfg(feature = "bf16_u16")]
            TensorStorage::BF16 { data, .. } => {
                #[cfg(feature = "shared_storage")]
                {
                    let arc: Arc<CudaSlice<u16>> = unsafe { std::ptr::read(data) };
                    let dev = arc.device();
                    match Arc::try_unwrap(arc) {
                        Ok(slice) => {
                            // R1b: slab claim ALWAYS runs (pool_off-agnostic).
                            if slab_v2_try_claim(&slice, true) {
                                std::mem::forget(slice);
                            } else if pool_off {
                                drop(slice);
                            } else {
                                crate::cuda_alloc_pool::pool_return_u16(slice);
                            }
                        }
                        Err(arc) => drop(arc),
                    }
                    unsafe { std::ptr::write(data, Arc::new(make_dummy_slice::<u16>(dev))) };
                }
                #[cfg(not(feature = "shared_storage"))]
                {
                    let slice: CudaSlice<u16> = unsafe { std::ptr::read(data) };
                    let dev = slice.device();
                    if slab_v2_try_claim(&slice, true) {
                        std::mem::forget(slice);
                    } else if pool_off {
                        drop(slice);
                    } else {
                        crate::cuda_alloc_pool::pool_return_u16(slice);
                    }
                    unsafe { std::ptr::write(data, make_dummy_slice::<u16>(dev)) };
                }
            }
            TensorStorage::I32 { data, .. } => {
                if pool_off {
                    return;
                }
                #[cfg(feature = "shared_storage")]
                {
                    let arc: Arc<CudaSlice<f32>> = unsafe { std::ptr::read(data) };
                    let dev = arc.device();
                    match Arc::try_unwrap(arc) {
                        Ok(slice) => crate::cuda_alloc_pool::pool_return_f32(slice),
                        Err(arc) => drop(arc),
                    }
                    unsafe { std::ptr::write(data, Arc::new(make_dummy_slice::<f32>(dev))) };
                }
                #[cfg(not(feature = "shared_storage"))]
                {
                    let slice: CudaSlice<f32> = unsafe { std::ptr::read(data) };
                    let dev = slice.device();
                    crate::cuda_alloc_pool::pool_return_f32(slice);
                    unsafe { std::ptr::write(data, make_dummy_slice::<f32>(dev)) };
                }
            }
            TensorStorage::Bool { data, .. } => {
                if pool_off {
                    return;
                }
                #[cfg(feature = "shared_storage")]
                {
                    let arc: Arc<CudaSlice<f32>> = unsafe { std::ptr::read(data) };
                    let dev = arc.device();
                    match Arc::try_unwrap(arc) {
                        Ok(slice) => crate::cuda_alloc_pool::pool_return_f32(slice),
                        Err(arc) => drop(arc),
                    }
                    unsafe { std::ptr::write(data, Arc::new(make_dummy_slice::<f32>(dev))) };
                }
                #[cfg(not(feature = "shared_storage"))]
                {
                    let slice: CudaSlice<f32> = unsafe { std::ptr::read(data) };
                    let dev = slice.device();
                    crate::cuda_alloc_pool::pool_return_f32(slice);
                    unsafe { std::ptr::write(data, make_dummy_slice::<f32>(dev)) };
                }
            }
            // I8, BF16Arena, BF16View — let Rust drop them normally.
            _ => {}
        }
    }
}

unsafe impl Send for TensorStorage {}
unsafe impl Sync for TensorStorage {}

/// Test-only: wrap a pre-built `CudaSlice<u16>` (e.g. from
/// `StaticSlabAllocator::alloc_u16`) into a `TensorStorage::BF16`. Used by
/// the R1b Drop-wiring regression test. Production code constructs
/// `TensorStorage` via the safe `empty`/`zeros` constructors.
#[doc(hidden)]
#[cfg(feature = "bf16_u16")]
pub fn wrap_slab_slice_for_test(slice: CudaSlice<u16>, numel: usize) -> TensorStorage {
    TensorStorage::BF16 {
        data: wrap_slice(slice),
        numel,
    }
}

// Note: F16/BF16 conversion kernels can be specialized further; current path stores as F32-backed buffers.
// For now, we store everything as F32 but track the intended dtype for API compatibility

/// Clear the global BF16 memory pool.
/// This is a helper to manually release cached memory.
pub fn clear_bf16_pool() {
    crate::memory_pool::MEMORY_POOL.clear_all_caches();
}
