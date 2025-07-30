use crate::shape::Shape;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, FlameError>;

#[derive(Error, Debug)]
pub enum FlameError {
    #[error("CUDA error: {0}")]
    Cuda(String),
    
    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: Shape, got: Shape },
    
    #[error("Broadcasting incompatible shapes: {lhs} and {rhs}")]
    BroadcastIncompatible { lhs: Shape, rhs: Shape },
    
    #[error("Unsupported dtype: {0}")]
    UnsupportedDType(String),
    
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
    
    #[error("CUDA driver error")]
    CudaDriver,
    
    #[error("CUBLAS error")]
    CuBlas,
    
    #[error("IO error: {0}")]
    Io(String),
    
    #[error("Kernel error: {0}")]
    KernelError(String),
    
    #[error("Out of memory: {0}")]
    OutOfMemory(String),
    
    #[error("Autograd error: {0}")]
    Autograd(String),
}

// Now actually converts cudarc errors
impl From<cudarc::driver::DriverError> for FlameError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        FlameError::Cuda(format!("{:?}", e))
    }
}

impl From<cudarc::cublas::result::CublasError> for FlameError {
    fn from(_: cudarc::cublas::result::CublasError) -> Self {
        FlameError::CuBlas
    }
}

impl From<std::io::Error> for FlameError {
    fn from(err: std::io::Error) -> Self {
        FlameError::Io(err.to_string())
    }
}

impl From<serde_json::Error> for FlameError {
    fn from(err: serde_json::Error) -> Self {
        FlameError::InvalidOperation(format!("JSON error: {}", err))
    }
}

impl From<image::ImageError> for FlameError {
    fn from(err: image::ImageError) -> Self {
        FlameError::InvalidOperation(format!("Image error: {}", err))
    }
}