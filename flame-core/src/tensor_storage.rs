use crate::{Result, FlameError, Shape, DType};
use crate::cuda_memory_alignment::alloc_aligned_f32;
use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use std::sync::Arc;
use half::bf16;

/// Actual storage backend for tensors with proper dtype support
#[derive(Clone)]
pub enum TensorStorage {
    F32 { data: CudaSlice<f32>, numel: usize },
    F16 { data: CudaSlice<f32>, numel: usize, scale: f32 },  // Store as F32 but with quantization scale
    BF16 { data: CudaSlice<f32>, numel: usize }, // BF16 stored as F32 for now (cudarc limitation)
    I8 { data: CudaSlice<i8>, numel: usize },  // INT8 support for Sage Attention
    I32 { data: CudaSlice<f32>, numel: usize }, // Stored as F32 but marked as I32
    Bool { data: CudaSlice<f32>, numel: usize }, // Stored as F32 (0.0/1.0) but marked as Bool
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
                // BF16 uses F32 storage for now (cudarc limitation)
                let mut data = alloc_aligned_f32(device, numel)?;
                device.memset_zeros(&mut data)?;
                Ok(TensorStorage::BF16 { data, numel })
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
                // BF16 is stored as F32 for now
                let mut out = alloc_aligned_f32(device, *numel)?;
                
                // If the allocation is larger, we need to handle it carefully
                if out.len() > *numel {
                    eprintln!("Warning: aligned allocation returned {} elements for {} requested", out.len(), *numel);
                }
                
                // Copy data - dtod_copy should handle size mismatches gracefully
                device.dtod_copy(data, &mut out)?;
                Ok(out)
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
    
    /// Get a reference to the underlying CudaSlice (for f32-backed storage)
    pub fn as_slice(&self) -> &CudaSlice<f32> {
        match self {
            TensorStorage::F32 { data, .. } |
            TensorStorage::F16 { data, .. } |
            TensorStorage::BF16 { data, .. } |
            TensorStorage::I32 { data, .. } |
            TensorStorage::Bool { data, .. } => data,  // stored as F32 for now
            TensorStorage::I8 { .. } => panic!("Cannot get f32 slice from I8 storage"),
        }
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
