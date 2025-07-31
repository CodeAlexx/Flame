//! CUDA-only narrow implementation using existing kernels

use crate::{Tensor, Result, FlameError, Shape};
use crate::tensor_storage::TensorStorage;
use crate::tensor::{alloc_zeros_from_pool, TensorId};
use cudarc::driver::{LaunchConfig, LaunchAsync};
use std::sync::Arc;

impl Tensor {
    /// Narrow (slice) a tensor along a dimension - CUDA only, no CPU fallback
    pub fn narrow_cuda(&self, dim: usize, start: usize, length: usize) -> Result<Tensor> {
        let dims = self.shape.dims();
        if dim >= dims.len() {
            return Err(FlameError::InvalidOperation(
                format!("Dimension {} out of range for tensor with {} dimensions", dim, dims.len())
            ));
        }
        
        if start + length > dims[dim] {
            return Err(FlameError::InvalidOperation(
                format!("Slice [{}, {}) out of range for dimension {} of size {}", 
                    start, start + length, dim, dims[dim])
            ));
        }
        
        // For batch dimension (dim 0) with single item, we can use existing slice operation
        if dim == 0 && length == 1 {
            // Calculate the offset in elements
            let mut stride = 1;
            for i in 1..dims.len() {
                stride *= dims[i];
            }
            let offset = start * stride;
            let slice_size = stride;
            
            // Use existing slice operation
            return self.slice(offset, slice_size);
        }
        
        // For other cases, we'll need to implement a proper narrow
        // For now, return an error to avoid CPU fallback
        Err(FlameError::InvalidOperation(
            "narrow operation only supported for batch dimension (dim=0) currently".to_string()
        ))
    }
}