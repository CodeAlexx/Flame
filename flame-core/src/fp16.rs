//! FP16 and BF16 support for mixed precision training and inference
//! 
//! This module provides utilities for working with half-precision floating point
//! formats to reduce memory usage and increase performance.

use crate::{Tensor, Result, FlameError, CudaDevice, DType};
use std::sync::Arc;
use half::{f16, bf16};

/// Precision type for mixed precision operations
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// Full 32-bit precision
    FP32,
    /// 16-bit floating point
    FP16,
    /// Brain floating point (16-bit with FP32 range)
    BF16,
}

impl Precision {
    /// Get the DType for this precision
    pub fn dtype(&self) -> DType {
        match self {
            Precision::FP32 => DType::F32,
            Precision::FP16 => DType::F16,
            Precision::BF16 => DType::BF16,
        }
    }
    
    /// Get bytes per element
    pub fn bytes_per_elem(&self) -> usize {
        match self {
            Precision::FP32 => 4,
            Precision::FP16 | Precision::BF16 => 2,
        }
    }
}

/// Automatic mixed precision context
pub struct AutocastContext {
    enabled: bool,
    precision: Precision,
    device: Arc<CudaDevice>,
}

impl AutocastContext {
    pub fn new(device: Arc<CudaDevice>, precision: Precision) -> Self {
        Self {
            enabled: false,
            precision,
            device,
        }
    }
    
    /// Enable autocast
    pub fn enable(&mut self) {
        self.enabled = true;
    }
    
    /// Disable autocast
    pub fn disable(&mut self) {
        self.enabled = false;
    }
    
    /// Check if autocast is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    /// Get the current precision
    pub fn precision(&self) -> Precision {
        self.precision
    }
}

/// Cast tensor to different precision
pub fn cast_tensor(tensor: &Tensor, target_dtype: DType) -> Result<Tensor> {
    // Use CUDA kernels for efficient casting when available
    // Always try to convert since FLAME is GPU-only
    if true {
        return crate::cuda_kernels_gpu::cast_dtype(tensor, target_dtype);
    }
    
    // CPU casting implementation
    let data = tensor.to_vec()?;
    let shape = tensor.shape().clone();
    let device = tensor.device().clone();
    
    match target_dtype {
        DType::F16 => {
            let fp16_data: Vec<f16> = data.iter()
                .map(|&x| f16::from_f32(x))
                .collect();
            
            // Convert back to f32 for storage (until we have proper f16 support)
            let f32_data: Vec<f32> = fp16_data.iter()
                .map(|&x| x.to_f32())
                .collect();
            
            Tensor::from_vec(f32_data, shape, device)
        }
        DType::BF16 => {
            let bf16_data: Vec<bf16> = data.iter()
                .map(|&x| bf16::from_f32(x))
                .collect();
            
            // Convert back to f32 for storage
            let f32_data: Vec<f32> = bf16_data.iter()
                .map(|&x| x.to_f32())
                .collect();
            
            Tensor::from_vec(f32_data, shape, device)
        }
        DType::F32 => {
            // Already F32
            Ok(tensor.clone_result()?)
        }
        _ => Err(FlameError::InvalidOperation(
            format!("Unsupported dtype for casting: {:?}", target_dtype)
        )),
    }
}

/// Mixed precision training utilities
pub struct MixedPrecisionTraining {
    precision: Precision,
    loss_scale: f32,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: usize,
    device: Arc<CudaDevice>,
    
    // State
    step_count: usize,
    last_overflow_step: usize,
}

impl MixedPrecisionTraining {
    pub fn new(
        precision: Precision,
        initial_loss_scale: f32,
        device: Arc<CudaDevice>,
    ) -> Self {
        Self {
            precision,
            loss_scale: initial_loss_scale,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            device,
            step_count: 0,
            last_overflow_step: 0,
        }
    }
    
    /// Scale gradients before backward pass
    pub fn scale_loss(&self, loss: &Tensor) -> Result<Tensor> {
        loss.mul_scalar(self.loss_scale)
    }
    
    /// Unscale gradients after backward pass
    pub fn unscale_gradients(&self, gradients: &mut [Tensor]) -> Result<()> {
        let inv_scale = 1.0 / self.loss_scale;
        
        for grad in gradients {
            *grad = grad.mul_scalar(inv_scale)?;
        }
        
        Ok(())
    }
    
    /// Check for gradient overflow and update loss scale
    pub fn update_scale(&mut self, found_overflow: bool) -> f32 {
        self.step_count += 1;
        
        if found_overflow {
            self.loss_scale *= self.backoff_factor;
            self.last_overflow_step = self.step_count;
        } else if self.step_count - self.last_overflow_step >= self.growth_interval {
            self.loss_scale *= self.growth_factor;
        }
        
        self.loss_scale
    }
    
    /// Check if any gradient has inf or nan
    pub fn check_gradients(&self, gradients: &[Tensor]) -> Result<bool> {
        for grad in gradients {
            let data = grad.to_vec()?;
            for &val in &data {
                if val.is_nan() || val.is_infinite() {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }
}

/// FP16 optimizer wrapper
pub struct FP16Optimizer<O> {
    base_optimizer: O,
    fp32_params: Vec<Tensor>,
    fp16_params: Vec<Tensor>,
    loss_scaler: MixedPrecisionTraining,
}

impl<O> FP16Optimizer<O> {
    pub fn new(
        base_optimizer: O,
        params: Vec<Tensor>,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        // Create FP32 master copies
        let mut fp32_params = Vec::new();
        let mut fp16_params = Vec::new();
        
        for param in params {
            // Keep FP32 master copy
            fp32_params.push(param.clone_result()?);
            
            // Create FP16 version
            let fp16_param = cast_tensor(&param, DType::F16)?;
            fp16_params.push(fp16_param);
        }
        
        let loss_scaler = MixedPrecisionTraining::new(
            Precision::FP16,
            65536.0, // Initial loss scale
            device,
        );
        
        Ok(Self {
            base_optimizer,
            fp32_params,
            fp16_params,
            loss_scaler,
        })
    }
    
    /// Get FP16 parameters for forward pass
    pub fn fp16_params(&self) -> &[Tensor] {
        &self.fp16_params
    }
    
    /// Step optimizer with mixed precision
    pub fn step(&mut self, gradients: Vec<Tensor>) -> Result<()> {
        // Unscale gradients
        let mut unscaled_grads = gradients;
        self.loss_scaler.unscale_gradients(&mut unscaled_grads)?;
        
        // Check for overflow
        let found_overflow = self.loss_scaler.check_gradients(&unscaled_grads)?;
        
        if !found_overflow {
            // Update FP32 master weights with gradients
            // Now actually applies gradients to parameters
            for (i, (fp32_param, grad)) in self.fp32_params.iter_mut().zip(unscaled_grads.iter()).enumerate() {
                // Simple SGD update for now - real optimizers would use Adam/AdamW logic
                let learning_rate = 1e-4; // Default LR
                let updated = fp32_param.sub(&grad.mul_scalar(learning_rate)?)?;
                *fp32_param = updated;
            }
            
            // Copy updated FP32 weights to FP16
            for (fp32_param, fp16_param) in self.fp32_params.iter().zip(self.fp16_params.iter_mut()) {
                *fp16_param = cast_tensor(fp32_param, DType::F16)?;
            }
        }
        
        // Update loss scale
        self.loss_scaler.update_scale(found_overflow);
        
        Ok(())
    }
}

/// Utility functions for half precision operations
pub mod half_ops {
    use super::*;
    
    /// Convert f32 slice to f16
    pub fn f32_to_f16(data: &[f32]) -> Vec<f16> {
        data.iter().map(|&x| f16::from_f32(x)).collect()
    }
    
    /// Convert f16 slice to f32
    pub fn f16_to_f32(data: &[f16]) -> Vec<f32> {
        data.iter().map(|&x| x.to_f32()).collect()
    }
    
    /// Convert f32 slice to bf16
    pub fn f32_to_bf16(data: &[f32]) -> Vec<bf16> {
        data.iter().map(|&x| bf16::from_f32(x)).collect()
    }
    
    /// Convert bf16 slice to f32
    pub fn bf16_to_f32(data: &[bf16]) -> Vec<f32> {
        data.iter().map(|&x| x.to_f32()).collect()
    }
}

/// CUDA kernels for FP16 operations
pub mod cuda_fp16 {
    use super::*;
    
    /// Get FP16 cast kernel code
    pub fn get_fp16_cast_kernel() -> &'static str {
        r#"
#include <cuda_fp16.h>

extern "C" __global__ void cast_f32_to_f16(
    const float* input,
    __half* output,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = __float2half(input[idx]);
    }
}

extern "C" __global__ void cast_f16_to_f32(
    const __half* input,
    float* output,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = __half2float(input[idx]);
    }
}
"#
    }
    
    /// Get BF16 cast kernel code
    pub fn get_bf16_cast_kernel() -> &'static str {
        r#"
#include <cuda_bf16.h>

extern "C" __global__ void cast_f32_to_bf16(
    const float* input,
    __nv_bfloat16* output,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = __float2bfloat16(input[idx]);
    }
}

extern "C" __global__ void cast_bf16_to_f32(
    const __nv_bfloat16* input,
    float* output,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = __bfloat162float(input[idx]);
    }
}
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Shape;
    use std::sync::Arc;
    
    #[test]
    fn test_fp16_conversion() -> Result<()> {
        let device = CudaDevice::new(0)?;
        
        let tensor = Tensor::randn(
            Shape::from_dims(&[2, 3, 4]),
            0.0,
            1.0,
            device
        )?;
        
        // Cast to FP16
        let fp16_tensor = cast_tensor(&tensor, DType::F16)?;
        
        // Cast back to FP32
        let fp32_tensor = cast_tensor(&fp16_tensor, DType::F32)?;
        
        // Check shapes match
        assert_eq!(tensor.shape().dims(), fp32_tensor.shape().dims());
        
        // Check values are close (some precision loss expected)
        let orig_data = tensor.to_vec()?;
        let converted_data = fp32_tensor.to_vec()?;
        
        for (a, b) in orig_data.iter().zip(converted_data.iter()) {
            assert!((a - b).abs() < 0.01, "FP16 conversion error too large");
        }
        
        println!("FP16 conversion test passed!");
        Ok(())
    }
    
    #[test] 
    fn test_mixed_precision_scaler() -> Result<()> {
        let device = CudaDevice::new(0)?;
        
        let mut mp = MixedPrecisionTraining::new(
            Precision::FP16,
            1024.0,
            device.clone()
        );
        
        let loss = Tensor::from_vec(
            vec![0.5],
            Shape::from_dims(&[1]),
            device
        )?;
        
        // Scale loss
        let scaled_loss = mp.scale_loss(&loss)?;
        let scaled_val = scaled_loss.to_vec()?[0];
        assert_eq!(scaled_val, 0.5 * 1024.0);
        
        // Test scale update
        let new_scale = mp.update_scale(false);
        assert_eq!(new_scale, 1024.0); // No change without overflow
        
        let new_scale = mp.update_scale(true);
        assert_eq!(new_scale, 512.0); // Reduced on overflow
        
        println!("Mixed precision scaler test passed!");
        Ok(())
    }
}
