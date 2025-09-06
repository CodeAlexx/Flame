// Basic cuDNN MatMul check for FLAME
// Since cudnnRNNMatrixMul is not available in older cuDNN versions,
// minimal shim to enable cuDNN-aware matmul operations

use crate::{Tensor, Result};

/// Check if tensors are suitable for cuDNN acceleration
pub fn is_cudnn_matmul_compatible(a: &Tensor, b: &Tensor) -> bool {
    // Check if cuDNN is available
    if !crate::cudnn::is_cudnn_available() {
        return false;
    }
    
    // Check tensor shapes
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    // Support 2D and 3D tensors
    match (a_shape.dims().len(), b_shape.dims().len()) {
        (2, 2) => true,      // Standard matrix multiply
        (3, 3) => true,      // Batch matrix multiply
        (3, 2) => true,      // Batch x matrix
        (2, 3) => true,      // Matrix x batch
        _ => false
    }
}

/// Basic cuDNN matmul - currently falls back to cuBLAS
/// Future versions can implement true cuDNN GEMM when available
pub fn cudnn_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // For now, just return an error indicating cuDNN matmul is not yet implemented
    // The tensor.matmul() method will fall back to cuBLAS automatically
    Err(crate::FlameError::InvalidOperation(
        "cuDNN matmul not yet implemented, use standard matmul".into()
    ))
}

/// Batch matrix multiply helper
pub fn cudnn_bmm(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    cudnn_matmul(a, b)
}
