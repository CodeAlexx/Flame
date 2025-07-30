use crate::{Result, FlameError};
use cudarc::driver::{CudaDevice as CudarcDevice};
use std::sync::Arc;

/// Device management for FLAME
#[derive(Clone)]
pub struct Device {
    inner: Arc<CudarcDevice>,
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("ordinal", &self.ordinal())
            .finish()
    }
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
    
    /// Set random seed for the device
    pub fn set_seed(&self, seed: u64) -> Result<()> {
        // For now, this is a no-op as FLAME doesn't have a global RNG state
        // In the future, we might want to integrate with cuRAND
        Ok(())
    }
    
    /// Create a CPU device (not supported in FLAME)
    pub fn cpu() -> Result<Self> {
        Err(FlameError::InvalidOperation("FLAME only supports CUDA devices".into()))
    }
    
    /// Check if this is a CPU device
    pub fn is_cpu(&self) -> bool {
        false // FLAME only supports CUDA
    }
    
    /// Check if this is a CUDA device
    pub fn is_cuda(&self) -> bool {
        true // FLAME only supports CUDA
    }
}

impl From<Arc<CudarcDevice>> for Device {
    fn from(device: Arc<CudarcDevice>) -> Self {
        Self { inner: device }
    }
}

/// Device enum for matching Candle API
#[derive(Clone, Debug)]
pub enum DeviceEnum {
    Cuda(Device),
}

impl DeviceEnum {
    /// Create a CUDA device
    pub fn cuda(ordinal: usize) -> Result<Self> {
        Ok(DeviceEnum::Cuda(Device::cuda(ordinal)?))
    }
    
    /// Check if CPU device
    pub fn is_cpu(&self) -> bool {
        false
    }
    
    /// Check if CUDA device  
    pub fn is_cuda(&self) -> bool {
        true
    }
}