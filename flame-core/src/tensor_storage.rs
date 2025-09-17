use crate::{Result, FlameError, Shape, DType};
use crate::cuda_memory_alignment::alloc_aligned_f32;
use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use std::sync::Arc;
use half::bf16;

/// Actual storage backend for tensors with proper dtype support
#[derive(Clone)]
pub enum TensorStorage {
    F32 { data: CudaSlice<f32>, numel: usize },
    F16 { data: CudaSlice<f32>, numel: usize, scale: f32 },
    #[cfg(not(feature = "bf16_u16"))]
    BF16 { data: CudaSlice<f32>, numel: usize },
    #[cfg(feature = "bf16_u16")]
    BF16 { data: CudaSlice<u16>, numel: usize },
    I8 { data: CudaSlice<i8>, numel: usize },
    I32 { data: CudaSlice<f32>, numel: usize },
    Bool { data: CudaSlice<f32>, numel: usize },
}

impl TensorStorage {
    /// Get the dtype of this storage
    pub fn dtype(&self) -> DType {
        match self {
            TensorStorage::F32 { .. } => DType::F32,
            TensorStorage::F16 { .. } => DType::F16,
            TensorStorage::BF16 { .. } => DType::BF16,
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
            TensorStorage::BF16 { numel, .. } => *numel,
            TensorStorage::I8 { numel, .. } => *numel,
            TensorStorage::I32 { numel, .. } => *numel,
            TensorStorage::Bool { numel, .. } => *numel,
        }
    }
    
    /// Allocate new storage using memory pool
    pub fn zeros(shape: &Shape, dtype: DType, device: &Arc<CudaDevice>) -> Result<Self> {
        let numel = shape.elem_count();
        
        match dtype {
            DType::F32 => {
                // Use aligned allocation for F32
                let mut data = alloc_aligned_f32(device, numel)?;
                device.memset_zeros(&mut data)?;
                Ok(TensorStorage::F32 { data, numel })
            }
            DType::F16 => {
                // F16 still uses F32 storage with scale
                let mut data = alloc_aligned_f32(device, numel)?;
                device.memset_zeros(&mut data)?;
                Ok(TensorStorage::F16 { data, numel, scale: 1.0 })
            }
            DType::BF16 => {
                #[cfg(not(feature = "bf16_u16"))]
                {
                    let mut data = alloc_aligned_f32(device, numel)?;
                    device.memset_zeros(&mut data)?;
                    Ok(TensorStorage::BF16 { data, numel })
                }
                #[cfg(feature = "bf16_u16")]
                {
                    let mut data = CudaSlice::<u16>::alloc(device, numel)
                        .map_err(|e| FlameError::Cuda(format!("alloc bf16 u16: {}", e)))?;
                    device.memset_zeros(&mut data)?;
                    Ok(TensorStorage::BF16 { data, numel })
                }
            }
            DType::I8 => {
                // For I8, we need to allocate i8 storage
                Err(FlameError::InvalidOperation(
                    "I8 allocation not yet supported in zeros - use quantization functions".into()
                ))
            }
            DType::I32 => {
                let mut data = alloc_aligned_f32(device, numel)?;
                device.memset_zeros(&mut data)?;
                Ok(TensorStorage::I32 { data, numel })
            }
            DType::Bool => {
                let mut data = alloc_aligned_f32(device, numel)?;
                device.memset_zeros(&mut data)?;
                Ok(TensorStorage::Bool { data, numel })
            }
            DType::F64 | DType::U8 | DType::U32 | DType::I32 | DType::I64 => {
                Err(FlameError::InvalidOperation(
                    format!("Unsupported dtype in TensorStorage: {:?}", dtype)
                ))
            }
        }
    }
    
    /// Convert to F32 (for operations that don't support F16/BF16)
    pub fn to_f32(&self, device: &Arc<CudaDevice>) -> Result<CudaSlice<f32>> {
        match self {
            TensorStorage::F32 { data, numel } | 
            TensorStorage::F16 { data, numel, .. } => {
                // Use aligned allocation
                let mut out = alloc_aligned_f32(device, *numel)?;
                
                // If the allocation is larger, we need to handle it carefully
                if out.len() > *numel {
                    eprintln!("Warning: aligned allocation returned {} elements for {} requested", out.len(), *numel);
                }
                
                // Copy data - dtod_copy should handle size mismatches gracefully
                device.dtod_copy(data, &mut out)?;
                Ok(out)
            }
            TensorStorage::BF16 { data, numel } => {
                #[cfg(not(feature = "bf16_u16"))]
                {
                    let mut out = alloc_aligned_f32(device, *numel)?;
                    if out.len() > *numel { eprintln!("Warning: aligned allocation returned {} elements for {} requested", out.len(), *numel); }
                    device.dtod_copy(data, &mut out)?;
                    Ok(out)
                }
                #[cfg(feature = "bf16_u16")]
                {
                    // Convert u16 BF16 → f32 via NVRTC kernel
                    let mut out = alloc_aligned_f32(device, *numel)?;
                    // Launch conversion kernel via helper
                    crate::bf16_convert::bf16_u16_to_f32(device.clone(), data, &mut out, *numel)?;
                    Ok(out)
                }
            }
            TensorStorage::I8 { .. } => {
                Err(FlameError::InvalidOperation(
                    "I8 to F32 conversion not yet implemented".into()
                ))
            }
            TensorStorage::I32 { data, numel } | TensorStorage::Bool { data, numel } => {
                let mut out = alloc_aligned_f32(device, *numel)?;
                if out.len() > *numel { eprintln!("Warning: aligned allocation returned {} elements for {} requested", out.len(), *numel); }
                device.dtod_copy(data, &mut out)?;
                Ok(out)
            }
        }
    }
    
    /// Safe: get read-only f32 slice for f32-backed storage. Otherwise Err.
    pub fn try_as_slice_f32(&self) -> Result<&CudaSlice<f32>> {
        match self {
            TensorStorage::F32 { data, .. } => Ok(data),
            TensorStorage::F16 { data, .. } => Ok(data),
            #[cfg(not(feature = "bf16_u16"))]
            TensorStorage::BF16 { data, .. } => Ok(data),
            TensorStorage::I32 { data, .. } => Ok(data),
            TensorStorage::Bool { data, .. } => Ok(data),
            #[cfg(feature = "bf16_u16")]
            TensorStorage::BF16 { .. } => Err(FlameError::InvalidInput(
                "expected F32 slice, got BF16(u16)".into()
            )),
            TensorStorage::I8 { .. } => Err(FlameError::InvalidInput(
                "expected F32 slice, got I8".into()
            )),
        }
    }

    /// Safe: get mutable f32 slice for f32-backed storage. Otherwise Err.
    pub fn try_as_mut_slice_f32(&mut self) -> Result<&mut CudaSlice<f32>> {
        match self {
            TensorStorage::F32 { data, .. } => Ok(data),
            TensorStorage::F16 { data, .. } => Ok(data),
            #[cfg(not(feature = "bf16_u16"))]
            TensorStorage::BF16 { data, .. } => Ok(data),
            TensorStorage::I32 { data, .. } => Ok(data),
            TensorStorage::Bool { data, .. } => Ok(data),
            #[cfg(feature = "bf16_u16")]
            TensorStorage::BF16 { .. } => Err(FlameError::InvalidInput(
                "expected F32 slice, got BF16(u16)".into()
            )),
            TensorStorage::I8 { .. } => Err(FlameError::InvalidInput(
                "expected F32 slice, got I8".into()
            )),
        }
    }

    /// Deprecated: use try_as_slice_f32() and handle Result.
    #[allow(clippy::expect_used)]
    #[deprecated(note = "use try_as_slice_f32() and handle Result")]
    pub fn as_slice(&self) -> &CudaSlice<f32> {
        self
            .try_as_slice_f32()
            .expect("TensorStorage::as_slice() panicked; migrate to try_as_slice_f32()")
    }

    /// Safe: get read-only u16 slice for BF16(u16) storage. Otherwise Err.
    #[cfg(feature = "bf16_u16")]
    pub fn try_as_slice_u16(&self) -> Result<&CudaSlice<u16>> {
        match self {
            TensorStorage::BF16 { data, .. } => Ok(data),
            TensorStorage::F32 { .. } => Err(FlameError::InvalidInput(
                "expected BF16(u16) slice, got F32".into()
            )),
            TensorStorage::F16 { .. } => Err(FlameError::InvalidInput(
                "expected BF16(u16) slice, got F16".into()
            )),
            TensorStorage::I32 { .. } => Err(FlameError::InvalidInput(
                "expected BF16(u16) slice, got I32".into()
            )),
            TensorStorage::Bool { .. } => Err(FlameError::InvalidInput(
                "expected BF16(u16) slice, got Bool".into()
            )),
            TensorStorage::I8 { .. } => Err(FlameError::InvalidInput(
                "expected BF16(u16) slice, got I8".into()
            )),
        }
    }

    /// Safe: get mutable u16 slice for BF16(u16) storage. Otherwise Err.
    #[cfg(feature = "bf16_u16")]
    pub fn try_as_mut_slice_u16(&mut self) -> Result<&mut CudaSlice<u16>> {
        match self {
            TensorStorage::BF16 { data, .. } => Ok(data),
            TensorStorage::F32 { .. } => Err(FlameError::InvalidInput(
                "expected BF16(u16) slice, got F32".into()
            )),
            TensorStorage::F16 { .. } => Err(FlameError::InvalidInput(
                "expected BF16(u16) slice, got F16".into()
            )),
            TensorStorage::I32 { .. } => Err(FlameError::InvalidInput(
                "expected BF16(u16) slice, got I32".into()
            )),
            TensorStorage::Bool { .. } => Err(FlameError::InvalidInput(
                "expected BF16(u16) slice, got Bool".into()
            )),
            TensorStorage::I8 { .. } => Err(FlameError::InvalidInput(
                "expected BF16(u16) slice, got I8".into()
            )),
        }
    }

    /// Deprecated: use try_as_slice_u16() and handle Result.
    #[allow(clippy::expect_used)]
    #[deprecated(note = "use try_as_slice_u16() and handle Result")]
    #[cfg(feature = "bf16_u16")]
    pub fn as_slice_u16(&self) -> &CudaSlice<u16> {
        self
            .try_as_slice_u16()
            .expect("TensorStorage::as_slice_u16() panicked; migrate to try_as_slice_u16()")
    }
    
    /// Get a reference to the underlying I8 CudaSlice
    pub fn as_i8_slice(&self) -> Result<&CudaSlice<i8>> {
        match self {
            TensorStorage::I8 { data, .. } => Ok(data),
            _ => Err(FlameError::InvalidOperation("Not an I8 tensor".into())),
        }
    }
}

// Note: F16/BF16 conversion kernels can be specialized further; current path stores as F32-backed buffers.
// For now, we store everything as F32 but track the intended dtype for API compatibility
