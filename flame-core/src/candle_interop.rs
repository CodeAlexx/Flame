use crate::{Tensor, Shape, Result, FlameError};
use std::sync::Arc;
use candle_core::backend::BackendDevice;

/// Conversion utilities between Flame and Candle tensors
pub struct CandleInterop;

impl CandleInterop {
    /// Convert a Flame tensor to Candle tensor
    /// Note: This attempts zero-copy when possible
    pub fn flame_to_candle(flame_tensor: &Tensor) -> Result<candle_core::Tensor> {
        // Create Candle CUDA device with same ordinal as Flame device
        let device_id = flame_tensor.device.ordinal();
        let device = candle_core::Device::new_cuda(device_id)
            .map_err(|e| FlameError::Cuda(format!("Failed to create Candle CUDA device: {}", e)))?;
        
        // Get shape
        let shape = flame_tensor.shape().dims();
        let candle_shape = candle_core::Shape::from_dims(shape);
        
        // Get data - this creates a copy for now
        // TODO: Implement zero-copy when possible
        let data = flame_tensor.to_vec()?;
        
        // Create Candle tensor
        let candle_tensor = candle_core::Tensor::from_vec(
            data,
            candle_shape,
            &device
        ).map_err(|e| FlameError::Cuda(format!("Failed to create Candle tensor: {}", e)))?;
        
        Ok(candle_tensor)
    }
    
    /// Convert a Candle tensor to Flame tensor
    /// Note: This attempts zero-copy when possible
    pub fn candle_to_flame(
        candle_tensor: &candle_core::Tensor,
        flame_device: Arc<cudarc::driver::CudaDevice>
    ) -> Result<Tensor> {
        // Get shape
        let shape_dims = candle_tensor.dims();
        let flame_shape = Shape::from_dims(shape_dims);
        
        // Check device compatibility
        match candle_tensor.device() {
            candle_core::Device::Cuda(candle_cuda) => {
                // Get the GPU ID from Candle's device location
                let candle_gpu_id = match candle_cuda.location() {
                    candle_core::DeviceLocation::Cuda { gpu_id } => gpu_id,
                    _ => return Err(FlameError::InvalidOperation(
                        "Unexpected device location for CUDA device".into()
                    )),
                };
                
                if candle_gpu_id != flame_device.ordinal() {
                    return Err(FlameError::InvalidOperation(
                        format!("Device mismatch: Candle CUDA {} vs Flame CUDA {}", 
                            candle_gpu_id, flame_device.ordinal())
                    ));
                }
            }
            _ => {
                return Err(FlameError::InvalidOperation(
                    "Only CUDA tensors are supported".into()
                ));
            }
        }
        
        // Get data - this creates a copy for now
        // TODO: Implement zero-copy when possible
        let data: Vec<f32> = candle_tensor.flatten_all()
            .map_err(|e| FlameError::Cuda(format!("Failed to flatten tensor: {}", e)))?
            .to_vec1()
            .map_err(|e| FlameError::Cuda(format!("Failed to get tensor data: {}", e)))?;
        
        // Create Flame tensor
        Tensor::from_vec(data, flame_shape, flame_device)
    }
    
    /// Convert a batch of Flame tensors to Candle
    pub fn flame_batch_to_candle(tensors: &[Tensor]) -> Result<Vec<candle_core::Tensor>> {
        tensors.iter()
            .map(|t| Self::flame_to_candle(t))
            .collect::<Result<Vec<_>>>()
    }
    
    /// Convert a batch of Candle tensors to Flame
    pub fn candle_batch_to_flame(
        tensors: &[candle_core::Tensor],
        device: Arc<cudarc::driver::CudaDevice>
    ) -> Result<Vec<Tensor>> {
        tensors.iter()
            .map(|t| Self::candle_to_flame(t, device.clone()))
            .collect::<Result<Vec<_>>>()
    }
}

/// Trait for types that can be converted between Flame and Candle
pub trait InteropTensor {
    fn to_candle(&self) -> Result<candle_core::Tensor>;
    fn from_candle(tensor: &candle_core::Tensor, device: Arc<cudarc::driver::CudaDevice>) -> Result<Self>
    where 
        Self: Sized;
}

impl InteropTensor for Tensor {
    fn to_candle(&self) -> Result<candle_core::Tensor> {
        CandleInterop::flame_to_candle(self)
    }
    
    fn from_candle(tensor: &candle_core::Tensor, device: Arc<cudarc::driver::CudaDevice>) -> Result<Self> {
        CandleInterop::candle_to_flame(tensor, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CudaDevice;
    
    #[test]
    fn test_flame_to_candle_conversion() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        
        // Create Flame tensor
        let flame_tensor = Tensor::randn(
            Shape::from_dims(&[2, 3, 4]),
            0.0,
            1.0,
            device.clone()
        )?;
        
        // Convert to Candle
        let candle_tensor = flame_tensor.to_candle()?;
        
        // Check shape
        assert_eq!(candle_tensor.dims(), &[2, 3, 4]);
        
        // Convert back
        let flame_tensor2 = Tensor::from_candle(&candle_tensor, device)?;
        
        // Check shape preserved
        assert_eq!(flame_tensor2.shape().dims(), &[2, 3, 4]);
        
        // Check data preserved (approximately)
        let data1 = flame_tensor.to_vec()?;
        let data2 = flame_tensor2.to_vec()?;
        
        for (a, b) in data1.iter().zip(data2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_batch_conversion() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        
        // Create batch of Flame tensors
        let tensors: Vec<Tensor> = (0..3)
            .map(|_| Tensor::randn(Shape::from_dims(&[4, 5]), 0.0, 1.0, device.clone()))
            .collect::<Result<Vec<_>>>()?;
        
        // Convert batch to Candle
        let candle_tensors = CandleInterop::flame_batch_to_candle(&tensors)?;
        assert_eq!(candle_tensors.len(), 3);
        
        // Convert back
        let flame_tensors = CandleInterop::candle_batch_to_flame(&candle_tensors, device)?;
        assert_eq!(flame_tensors.len(), 3);
        
        Ok(())
    }
}