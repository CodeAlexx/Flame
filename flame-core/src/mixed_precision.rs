use crate::{Tensor, Shape, Result, FlameError, DType};
use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaSlice};
use half::{f16, bf16};
use std::collections::HashMap;

// Helper function for allocating and copying to GPU via memory pool
fn alloc_and_copy_to_pool<T: AsRef<[f32]>>(device: &Arc<CudaDevice>, data: T) -> Result<CudaSlice<f32>> {
    let slice = data.as_ref();
    let mut cuda_data = crate::tensor::alloc_from_pool(device, slice.len())?;
    device.htod_copy_into(slice.to_vec(), &mut cuda_data).map_err(|_| FlameError::CudaDriver)?;
    Ok(cuda_data)
}


/// Automatic Mixed Precision (AMP) context
pub struct AMPContext {
    /// Whether AMP is enabled
    pub enabled: bool,
    /// Precision to use for compute (f16 or bf16)
    pub compute_dtype: DType,
    /// Loss scale for gradient scaling
    pub loss_scale: f32,
    /// Dynamic loss scaling
    pub dynamic_scaling: bool,
    /// Growth factor for dynamic scaling
    pub growth_factor: f32,
    /// Backoff factor for dynamic scaling
    pub backoff_factor: f32,
    /// Growth interval
    pub growth_interval: usize,
    /// Current step count
    current_step: usize,
    /// Steps since last scale update
    steps_since_update: usize,
}

impl AMPContext {
    pub fn new(compute_dtype: DType) -> Self {
        Self {
            enabled: true,
            compute_dtype,
            loss_scale: 65536.0,  // 2^16
            dynamic_scaling: true,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            current_step: 0,
            steps_since_update: 0,
        }
    }
    
    /// Create a disabled AMP context (full precision)
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            compute_dtype: DType::F32,
            loss_scale: 1.0,
            dynamic_scaling: false,
            growth_factor: 1.0,
            backoff_factor: 1.0,
            growth_interval: 2000,
            current_step: 0,
            steps_since_update: 0,
        }
    }
    
    /// Update loss scale based on gradient overflow
    pub fn update_scale(&mut self, found_inf: bool) {
        if !self.dynamic_scaling {
            return;
        }
        
        if found_inf {
            // Gradient overflow detected, decrease scale
            self.loss_scale *= self.backoff_factor;
            self.steps_since_update = 0;
            println!("Gradient overflow detected. Decreasing loss scale to {}", self.loss_scale);
        } else {
            self.steps_since_update += 1;
            
            // Increase scale if we've gone long enough without overflow
            if self.steps_since_update >= self.growth_interval {
                self.loss_scale *= self.growth_factor;
                self.steps_since_update = 0;
                println!("Increasing loss scale to {}", self.loss_scale);
            }
        }
        
        self.current_step += 1;
    }
    
    /// Scale loss for backward pass
    pub fn scale_loss(&self, loss: &Tensor) -> Result<Tensor> {
        if self.enabled {
            loss.mul_scalar(self.loss_scale)
        } else {
            loss.clone()
        }
    }
    
    /// Unscale gradients after backward pass
    pub fn unscale_grads(&self, grad: &Tensor) -> Result<Tensor> {
        if self.enabled {
            grad.mul_scalar(1.0 / self.loss_scale)
        } else {
            grad.clone()
        }
    }
}

/// Mixed precision tensor wrapper
pub struct MixedPrecisionTensor {
    /// Full precision master weights (f32)
    pub master: Tensor,
    /// Half precision compute weights (f16/bf16)
    pub compute: Option<HalfTensor>,
    /// AMP context
    pub amp_context: Arc<AMPContext>,
}

impl MixedPrecisionTensor {
    /// Create from f32 tensor with AMP context
    pub fn new(tensor: Tensor, amp_context: Arc<AMPContext>) -> Result<Self> {
        let compute = if amp_context.enabled {
            Some(HalfTensor::from_f32(&tensor, amp_context.compute_dtype)?)
        } else {
            None
        };
        
        Ok(Self {
            master: tensor,
            compute,
            amp_context,
        })
    }
    
    /// Get tensor for forward pass (half precision if AMP enabled)
    pub fn forward_tensor(&self) -> &Tensor {
        if self.amp_context.enabled && self.compute.is_some() {
            // Return f32 view of half precision tensor
            // For now, we'll return master until we implement proper casting
            &self.master
        } else {
            &self.master
        }
    }
    
    /// Update master weights from optimizer
    pub fn update_master(&mut self, update: &Tensor) -> Result<()> {
        self.master = update.clone()?;
        
        // Update half precision copy if AMP enabled
        if self.amp_context.enabled {
            self.compute = Some(HalfTensor::from_f32(&self.master, self.amp_context.compute_dtype)?);
        }
        
        Ok(())
    }
}

/// Half precision tensor (f16 or bf16)
/// Note: Currently stores data as f32 on GPU due to cudarc limitations
pub struct HalfTensor {
    pub data: CudaSlice<f32>,  // Stores converted f16/bf16 data as f32
    pub shape: Shape,
    pub device: Arc<CudaDevice>,
    pub dtype: DType,
}

impl HalfTensor {
    /// Convert f32 tensor to half precision
    pub fn from_f32(tensor: &Tensor, dtype: DType) -> Result<Self> {
        let data = tensor.to_vec()?;
        let shape = tensor.shape().clone();
        let device = tensor.device().clone();
        
        match dtype {
            DType::F16 => {
                // Convert to f16 and store as bytes
                let half_data: Vec<f16> = data.iter()
                    .map(|&x| f16::from_f32(x))
                    .collect();
                // For now, store f16 data as f32 on GPU (2x memory usage)
                // Full implementation would use proper f16 CUDA support
                let f32_data: Vec<f32> = half_data.iter()
                    .map(|&x| x.to_f32())
                    .collect();
                let cuda_data = alloc_and_copy_to_pool(&device, &f32_data)
                    .map_err(|_| FlameError::CudaDriver)?;
                
                Ok(Self {
                    data: cuda_data,
                    shape,
                    device,
                    dtype: DType::F16,
                })
            }
            DType::BF16 => {
                // Convert to bf16 and store as bytes
                let half_data: Vec<bf16> = data.iter()
                    .map(|&x| bf16::from_f32(x))
                    .collect();
                // For now, store bf16 data as f32 on GPU (2x memory usage)
                // Full implementation would use proper bf16 CUDA support
                let f32_data: Vec<f32> = half_data.iter()
                    .map(|&x| x.to_f32())
                    .collect();
                let cuda_data = alloc_and_copy_to_pool(&device, &f32_data)
                    .map_err(|_| FlameError::CudaDriver)?;
                
                Ok(Self {
                    data: cuda_data,
                    shape,
                    device,
                    dtype: DType::BF16,
                })
            }
            _ => Err(FlameError::InvalidOperation(
                format!("Cannot create HalfTensor with dtype {:?}", dtype)
            )),
        }
    }
    
    /// Convert back to f32 tensor
    pub fn to_f32(&self) -> Result<Tensor> {
        // Since we're already storing as f32, just copy the data
        let f32_data = self.device.dtoh_sync_copy(&self.data)
            .map_err(|_| FlameError::CudaDriver)?;
        Tensor::from_vec(f32_data, self.shape.clone(), self.device.clone())
    }
}

/// Mixed precision optimizer wrapper
pub struct MixedPrecisionOptimizer<O> {
    /// Base optimizer
    pub optimizer: O,
    /// AMP context
    pub amp_context: Arc<AMPContext>,
    /// Master weight storage
    pub master_weights: HashMap<usize, MixedPrecisionTensor>,
}

impl<O> MixedPrecisionOptimizer<O> {
    pub fn new(optimizer: O, amp_context: Arc<AMPContext>) -> Self {
        Self {
            optimizer,
            amp_context,
            master_weights: HashMap::new(),
        }
    }
    
    /// Register a parameter for mixed precision training
    pub fn register_param(&mut self, id: usize, param: Tensor) -> Result<()> {
        let mp_tensor = MixedPrecisionTensor::new(param, self.amp_context.clone())?;
        self.master_weights.insert(id, mp_tensor);
        Ok(())
    }
    
    /// Check gradients for infinity/NaN
    pub fn check_grads(&self, grads: &[&Tensor]) -> Result<bool> {
        for grad in grads {
            let data = grad.to_vec()?;
            for &val in &data {
                if val.is_infinite() || val.is_nan() {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }
}

/// Loss scaler for gradient scaling
pub struct GradScaler {
    scale: f32,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: i32,
    _scale_update_count: i32,
}

impl GradScaler {
    pub fn new() -> Self {
        Self {
            scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            _scale_update_count: 0,
        }
    }
    
    /// Scale the loss
    pub fn scale(&self, loss: &Tensor) -> Result<Tensor> {
        loss.mul_scalar(self.scale)
    }
    
    /// Unscale gradients
    pub fn unscale(&self, optimizer_grads: &mut [Tensor]) -> Result<()> {
        let inv_scale = 1.0 / self.scale;
        for grad in optimizer_grads {
            *grad = grad.mul_scalar(inv_scale)?;
        }
        Ok(())
    }
    
    /// Update scale factor based on NaN/Inf in gradients
    pub fn update(&mut self, found_inf: bool) {
        if found_inf {
            self.scale *= self.backoff_factor;
            self._scale_update_count = 0;
        } else {
            self._scale_update_count += 1;
            if self._scale_update_count >= self.growth_interval {
                self.scale *= self.growth_factor;
                self._scale_update_count = 0;
            }
        }
    }
}

/// Utility functions for mixed precision
pub mod utils {
    use super::*;
    
    /// Check if operation should use mixed precision
    pub fn should_use_amp(op_name: &str) -> bool {
        // Whitelist of operations that benefit from mixed precision
        match op_name {
            "matmul" | "conv2d" | "linear" | "attention" => true,
            "softmax" | "layer_norm" | "batch_norm" => false,  // Keep in f32 for stability
            _ => false,
        }
    }
    
    /// Cast tensor to target dtype if needed
    pub fn auto_cast(tensor: &Tensor, _target_dtype: DType, _device: &Arc<CudaDevice>) -> Result<Tensor> {
        // For now, just return a clone since we're using f32 throughout
        // Full implementation would do actual dtype conversion
        tensor.clone()
    }
    
    /// Create a tensor directly in half precision
    pub fn zeros_half(shape: Shape, dtype: DType, device: Arc<CudaDevice>) -> Result<HalfTensor> {
        let f32_zeros = Tensor::zeros(shape.clone(), device.clone())?;
        HalfTensor::from_f32(&f32_zeros, dtype)
    }
    
    /// Create a random tensor in half precision
    pub fn randn_half(shape: Shape, mean: f32, std: f32, dtype: DType, device: Arc<CudaDevice>) -> Result<HalfTensor> {
        let f32_randn = Tensor::randn(shape.clone(), mean, std, device.clone())?;
        HalfTensor::from_f32(&f32_randn, dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_amp_context() -> Result<()> {
        let mut ctx = AMPContext::new(DType::F16);
        assert_eq!(ctx.loss_scale, 65536.0);
        
        // Simulate gradient overflow
        ctx.update_scale(true);
        assert_eq!(ctx.loss_scale, 32768.0);
        
        // Simulate successful steps
        for _ in 0..ctx.growth_interval {
            ctx.update_scale(false);
        }
        assert_eq!(ctx.loss_scale, 65536.0);
        
        Ok(())
    }
    
    #[test]
    fn test_half_tensor_conversion() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let f32_tensor = Tensor::randn(Shape::from_dims(&[2, 3]), 0.0, 1.0, device)?;
        
        // Test F16 conversion
        let f16_tensor = HalfTensor::from_f32(&f32_tensor, DType::F16)?;
        let f32_back = f16_tensor.to_f32()?;
        
        // Check shapes match
        assert_eq!(f32_tensor.shape().dims(), f32_back.shape().dims());
        
        // Test BF16 conversion
        let bf16_tensor = HalfTensor::from_f32(&f32_tensor, DType::BF16)?;
        let f32_back_bf16 = bf16_tensor.to_f32()?;
        
        assert_eq!(f32_tensor.shape().dims(), f32_back_bf16.shape().dims());
        
        Ok(())
    }
}