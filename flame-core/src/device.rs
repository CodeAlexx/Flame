use crate::{Result, FlameError};
use cudarc::driver::{CudaDevice as CudarcDevice};
use std::sync::Arc;

/// Device management for FLAME
#[derive(Clone)]
pub struct Device {
    inner: Arc<CudarcDevice>,
}

impl Device {
    /// Create a new device for the given GPU ordinal
    pub fn cuda(ordinal: usize) -> Result<Self> {
        let device = CudarcDevice::new(ordinal)
            .map_err(|e| FlameError::Cuda(format!("Failed to create CUDA device: {}", e)))?;
        Ok(Self { inner: device })
    }
    
    /// Get the underlying CUDA device
    pub fn cuda_device(&self) -> &Arc<CudarcDevice> {
        &self.inner
    }
    
    /// Get device ordinal
    pub fn ordinal(&self) -> usize {
        self.inner.ordinal()
    }
}

impl From<Arc<CudarcDevice>> for Device {
    fn from(device: Arc<CudarcDevice>) -> Self {
        Self { inner: device }
    }
}