use crate::shape::Shape;
use thiserror::Error as ThisError;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(ThisError, Debug)]
pub enum Error {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Unsupported: {0}")]
    Unsupported(String),

    #[error("Training/runtime error: {0}")]
    Training(String),
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

    #[error("Invalid index: {0}")]
    InvalidIndex(String),

    #[error("Invalid shape: {0}")]
    InvalidShape(String),

    #[error("CUDA error: {0}")]
    CudaError(String),

    /// Used by staged ports that intentionally delay implementing a code path
    /// (e.g. `TensorIteratorBase::build_unary_op` is stubbed in Phase 1 and
    /// filled in at Phase 3 of the TensorIterator migration). Carrying the
    /// phase name in `reason` lets the calling tests assert the stub without
    /// string-matching a generic message.
    #[error("Not implemented: {reason}")]
    NotImplemented { reason: String },
}

// Now actually converts cudarc errors
impl From<cudarc::driver::DriverError> for Error {
    fn from(e: cudarc::driver::DriverError) -> Self {
        Error::KernelError(format!("{e:?}"))
    }
}

impl From<cudarc::cublas::result::CublasError> for Error {
    fn from(err: cudarc::cublas::result::CublasError) -> Self {
        Error::KernelError(format!("cublas: {err:?}"))
    }
}

impl From<cudarc::cublaslt::result::CublasError> for Error {
    fn from(err: cudarc::cublaslt::result::CublasError) -> Self {
        Error::KernelError(format!("cublasLt: {err:?}"))
    }
}

impl From<cudarc::nvrtc::CompileError> for Error {
    fn from(err: cudarc::nvrtc::CompileError) -> Self {
        Error::KernelError(format!("nvrtc: {err:?}"))
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::InvalidOperation(format!("JSON error: {err}"))
    }
}

impl From<image::ImageError> for Error {
    fn from(err: image::ImageError) -> Self {
        Error::InvalidOperation(format!("Image error: {err}"))
    }
}

impl From<anyhow::Error> for Error {
    fn from(err: anyhow::Error) -> Self {
        Error::InvalidOperation(format!("Anyhow error: {err}"))
    }
}
