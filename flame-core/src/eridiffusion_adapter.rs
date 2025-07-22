//! EriDiffusion integration adapters
//! 
//! This module provides adapters to integrate Flame tensor operations
//! with EriDiffusion's training and inference infrastructure.

use crate::{
    Tensor, Shape, Result, FlameError,
    // candle_interop::CandleInterop,  // Temporarily disabled
};
use std::sync::Arc;
use std::collections::HashMap;

/// Adapter for converting between Flame and EriDiffusion tensor representations
pub struct TensorAdapter {
    device: Arc<cudarc::driver::CudaDevice>,
}

impl TensorAdapter {
    pub fn new(device: Arc<cudarc::driver::CudaDevice>) -> Self {
        Self { device }
    }
    
    /// Convert a batch of Flame tensors to format expected by EriDiffusion
    pub fn prepare_batch(&self, 
        images: &Tensor,
        latents: Option<&Tensor>,
        text_embeddings: &Tensor,
        timesteps: &Tensor,
    ) -> Result<BatchData> {
        // Validate shapes
        let batch_size = images.shape().dims()[0];
        
        if text_embeddings.shape().dims()[0] != batch_size {
            return Err(FlameError::InvalidOperation(
                format!("Batch size mismatch: images {} vs text_embeddings {}", 
                    batch_size, text_embeddings.shape().dims()[0])
            ));
        }
        
        // Clone tensors - Tensor doesn't implement Clone, so we need to copy data
        let images_copy = Tensor::from_vec(
            images.to_vec()?,
            images.shape().clone(),
            images.device.clone()
        )?;
        
        let latents_copy = if let Some(lat) = latents {
            Some(Tensor::from_vec(
                lat.to_vec()?,
                lat.shape().clone(),
                lat.device.clone()
            )?)
        } else {
            None
        };
        
        let text_embeddings_copy = Tensor::from_vec(
            text_embeddings.to_vec()?,
            text_embeddings.shape().clone(),
            text_embeddings.device.clone()
        )?;
        
        let timesteps_copy = Tensor::from_vec(
            timesteps.to_vec()?,
            timesteps.shape().clone(),
            timesteps.device.clone()
        )?;
        
        Ok(BatchData {
            images: images_copy,
            latents: latents_copy,
            text_embeddings: text_embeddings_copy,
            timesteps: timesteps_copy,
            batch_size,
        })
    }
    
    /// Convert EriDiffusion model outputs back to Flame tensors
    pub fn process_outputs(&self, outputs: HashMap<String, candle_core::Tensor>) -> Result<ModelOutputs> {
        let mut flame_outputs = HashMap::new();
        
        for (name, tensor) in outputs {
            let flame_tensor = CandleInterop::candle_to_flame(&tensor, self.device.clone())?;
            flame_outputs.insert(name, flame_tensor);
        }
        
        Ok(ModelOutputs {
            tensors: flame_outputs,
        })
    }
}

/// Batch data structure for training
pub struct BatchData {
    pub images: Tensor,
    pub latents: Option<Tensor>,
    pub text_embeddings: Tensor,
    pub timesteps: Tensor,
    pub batch_size: usize,
}

/// Model outputs structure
pub struct ModelOutputs {
    pub tensors: HashMap<String, Tensor>,
}

/// Adapter for model weight management
pub struct WeightAdapter {
    device: Arc<cudarc::driver::CudaDevice>,
}

impl WeightAdapter {
    pub fn new(device: Arc<cudarc::driver::CudaDevice>) -> Self {
        Self { device }
    }
    
    /// Load weights from EriDiffusion checkpoint into Flame tensors
    pub fn load_checkpoint(&self, path: &str) -> Result<HashMap<String, Tensor>> {
        use candle_core::safetensors;
        
        let device = candle_core::Device::new_cuda(0)
            .map_err(|e| FlameError::InvalidOperation(format!("Failed to create CUDA device: {}", e)))?;
        let tensors = safetensors::load(path, &device)
            .map_err(|e| FlameError::InvalidOperation(format!("Failed to load checkpoint: {}", e)))?;
        
        let mut flame_weights = HashMap::new();
        for (name, tensor) in tensors {
            let flame_tensor = CandleInterop::candle_to_flame(&tensor, self.device.clone())?;
            flame_weights.insert(name, flame_tensor);
        }
        
        Ok(flame_weights)
    }
    
    /// Save Flame tensors to EriDiffusion checkpoint format
    pub fn save_checkpoint(&self, weights: &HashMap<String, Tensor>, path: &str) -> Result<()> {
        use candle_core::safetensors;
        
        let mut candle_tensors = HashMap::new();
        for (name, tensor) in weights {
            let candle_tensor = CandleInterop::flame_to_candle(tensor)?;
            candle_tensors.insert(name.clone(), candle_tensor);
        }
        
        safetensors::save(&candle_tensors, path)
            .map_err(|e| FlameError::InvalidOperation(format!("Failed to save checkpoint: {}", e)))?;
        
        Ok(())
    }
    
    /// Apply LoRA weights to base model
    pub fn apply_lora(&self, 
        base_weights: &mut HashMap<String, Tensor>,
        lora_weights: &HashMap<String, Tensor>,
        alpha: f32,
    ) -> Result<()> {
        for (name, lora_weight) in lora_weights {
            if let Some(base_name) = name.strip_suffix(".lora_down") {
                let up_name = format!("{}.lora_up", base_name);
                
                if let (Some(lora_up), Some(base_weight)) = 
                    (lora_weights.get(&up_name), base_weights.get_mut(base_name)) {
                    
                    // Apply LoRA: base + alpha * (up @ down)
                    let lora_delta = lora_up.matmul(lora_weight)?;
                    let scaled_delta = lora_delta.scale(alpha)?;
                    let new_weight = base_weight.add(&scaled_delta)?;
                    
                    *base_weight = new_weight;
                }
            }
        }
        
        Ok(())
    }
}

/// Training state adapter
pub struct TrainingStateAdapter {
    device: Arc<cudarc::driver::CudaDevice>,
}

impl TrainingStateAdapter {
    pub fn new(device: Arc<cudarc::driver::CudaDevice>) -> Self {
        Self { device }
    }
    
    /// Create training state from configuration
    pub fn create_state(&self, config: &TrainingConfig) -> Result<TrainingState> {
        Ok(TrainingState {
            step: 0,
            epoch: 0,
            best_loss: f32::INFINITY,
            learning_rate: config.learning_rate,
            gradient_accumulation_steps: config.gradient_accumulation_steps,
        })
    }
    
    /// Update training state after step
    pub fn update_state(&self, state: &mut TrainingState, loss: f32, step: usize) {
        state.step = step;
        state.epoch = step / state.gradient_accumulation_steps;
        
        if loss < state.best_loss {
            state.best_loss = loss;
        }
    }
}

/// Training configuration
#[derive(Clone)]
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub gradient_accumulation_steps: usize,
    pub max_steps: usize,
    pub save_every: usize,
    pub validate_every: usize,
    pub use_ema: bool,
    pub ema_decay: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            gradient_accumulation_steps: 1,
            max_steps: 10000,
            save_every: 1000,
            validate_every: 100,
            use_ema: true,
            ema_decay: 0.9999,
        }
    }
}

/// Training state
pub struct TrainingState {
    pub step: usize,
    pub epoch: usize,
    pub best_loss: f32,
    pub learning_rate: f32,
    pub gradient_accumulation_steps: usize,
}

/// Loss adapter for different training objectives
pub struct LossAdapter {
    device: Arc<cudarc::driver::CudaDevice>,
}

impl LossAdapter {
    pub fn new(device: Arc<cudarc::driver::CudaDevice>) -> Self {
        Self { device }
    }
    
    /// Compute diffusion loss (noise prediction or v-prediction)
    pub fn compute_diffusion_loss(&self,
        model_output: &Tensor,
        target: &Tensor,
        timesteps: &Tensor,
        loss_type: LossType,
        snr_gamma: Option<f32>,
    ) -> Result<Tensor> {
        let mse_loss = self.mse_loss(model_output, target)?;
        
        // Apply SNR weighting if requested
        if let Some(gamma) = snr_gamma {
            let snr_weight = self.compute_snr_weight(timesteps, gamma)?;
            // For now, just average the SNR weights and apply uniformly
            let snr_data = snr_weight.to_vec()?;
            let avg_weight = snr_data.iter().sum::<f32>() / snr_data.len() as f32;
            mse_loss.scale(avg_weight)
        } else {
            Ok(mse_loss)
        }
    }
    
    /// Mean squared error loss
    fn mse_loss(&self, pred: &Tensor, target: &Tensor) -> Result<Tensor> {
        let diff = pred.sub(target)?;
        let squared = diff.mul(&diff)?;
        squared.mean()
    }
    
    /// Compute SNR-based loss weight
    fn compute_snr_weight(&self, timesteps: &Tensor, gamma: f32) -> Result<Tensor> {
        // Compute signal-to-noise ratio (SNR) based on DDPM formulation
        // For DDPM: alpha_t = prod(1 - beta_i) for i=1 to t
        // SNR(t) = alpha_t^2 / (1 - alpha_t^2)
        
        // Linear beta schedule (can be customized)
        let num_train_timesteps = 1000.0;
        let beta_start = 0.0001;
        let beta_end = 0.02;
        
        // Normalize timesteps to [0, 1]
        let normalized_timesteps = timesteps.div_scalar(num_train_timesteps)?;
        
        // Compute betas for continuous timesteps
        let betas = normalized_timesteps.mul_scalar(beta_end - beta_start)?
            .add_scalar(beta_start)?;
        
        // Compute alpha = 1 - beta
        let alphas = Tensor::ones_like(&betas)?
            .sub(&betas)?;
        
        // Compute cumulative product of alphas (alpha_bar)
        // For continuous time: alpha_bar(t) = exp(-0.5 * integral of log(1-beta))
        // Approximation: alpha_bar(t) ≈ exp(-0.5 * t * (beta_start + beta_end)/2)
        let avg_beta = (beta_start + beta_end) / 2.0;
        let log_alpha_bar = normalized_timesteps.mul_scalar(-avg_beta)?;
        let alpha_bar = log_alpha_bar.exp()?;
        
        // Compute SNR = alpha_bar^2 / (1 - alpha_bar^2)
        let alpha_bar_sq = alpha_bar.mul(&alpha_bar)?;
        let one_minus_alpha_bar_sq = Tensor::ones_like(&alpha_bar_sq)?
            .sub(&alpha_bar_sq)?;
        
        // Add small epsilon to avoid division by zero
        let snr = alpha_bar_sq.div(&one_minus_alpha_bar_sq.add_scalar(1e-8)?)?;
        
        // Apply Min-SNR weighting: min(snr, gamma) / snr
        // This prevents overweighting of easy timesteps
        let snr_clipped = snr.minimum_scalar(gamma)?;
        let weights = snr_clipped.div(&snr.add_scalar(1e-8)?)?;
        
        weights
    }
}

#[derive(Clone, Copy)]
pub enum LossType {
    L2,
    L1,
    Huber(f32),
}

/// Gradient scaler for mixed precision training
pub struct GradientScaler {
    pub scale: f32,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: usize,
    steps_since_update: usize,
}

impl GradientScaler {
    pub fn new() -> Self {
        Self {
            scale: 65536.0,  // 2^16
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            steps_since_update: 0,
        }
    }
    
    /// Scale gradients for mixed precision
    pub fn scale_gradients(&self, gradients: &mut HashMap<String, Tensor>) -> Result<()> {
        for (_, grad) in gradients.iter_mut() {
            *grad = grad.scale(self.scale)?;
        }
        Ok(())
    }
    
    /// Unscale gradients before optimizer step
    pub fn unscale_gradients(&self, gradients: &mut HashMap<String, Tensor>) -> Result<()> {
        let inv_scale = 1.0 / self.scale;
        for (_, grad) in gradients.iter_mut() {
            *grad = grad.scale(inv_scale)?;
        }
        Ok(())
    }
    
    /// Update scale based on gradient overflow
    pub fn update(&mut self, found_inf: bool) {
        if found_inf {
            self.scale *= self.backoff_factor;
            self.steps_since_update = 0;
        } else {
            self.steps_since_update += 1;
            if self.steps_since_update >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.steps_since_update = 0;
            }
        }
    }
}

/// Memory optimization utilities
pub struct MemoryOptimizer {
    device: Arc<cudarc::driver::CudaDevice>,
    cache: HashMap<String, Tensor>,
}

impl MemoryOptimizer {
    pub fn new(device: Arc<cudarc::driver::CudaDevice>) -> Self {
        Self {
            device,
            cache: HashMap::new(),
        }
    }
    
    /// Cache intermediate activations
    pub fn cache_activation(&mut self, name: &str, tensor: Tensor) {
        self.cache.insert(name.to_string(), tensor);
    }
    
    /// Retrieve cached activation
    pub fn get_activation(&self, name: &str) -> Option<&Tensor> {
        self.cache.get(name)
    }
    
    /// Clear activation cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
    
    /// Estimate memory usage
    pub fn estimate_memory_usage(&self, batch_size: usize, model_size: usize) -> usize {
        // Rough estimate: model weights + activations + gradients
        let weights_memory = model_size * 4;  // FP32
        let activation_memory = batch_size * model_size * 4;
        let gradient_memory = model_size * 4;
        
        weights_memory + activation_memory + gradient_memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CudaDevice;
    
    #[test]
    fn test_tensor_adapter() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let adapter = TensorAdapter::new(device.clone());
        
        let batch_size = 2;
        let images = Tensor::randn(
            Shape::from_dims(&[batch_size, 3, 512, 512]),
            0.0,
            1.0,
            device.clone()
        )?;
        
        let text_embeddings = Tensor::randn(
            Shape::from_dims(&[batch_size, 77, 768]),
            0.0,
            0.02,
            device.clone()
        )?;
        
        let timesteps = Tensor::from_vec(
            vec![0.1, 0.5],
            Shape::from_dims(&[batch_size]),
            device
        )?;
        
        let batch = adapter.prepare_batch(&images, None, &text_embeddings, &timesteps)?;
        assert_eq!(batch.batch_size, batch_size);
        
        Ok(())
    }
    
    #[test]
    fn test_loss_adapter() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let adapter = LossAdapter::new(device.clone());
        
        let pred = Tensor::randn(
            Shape::from_dims(&[2, 4, 64, 64]),
            0.0,
            0.1,
            device.clone()
        )?;
        
        let target = Tensor::randn(
            Shape::from_dims(&[2, 4, 64, 64]),
            0.0,
            0.1,
            device.clone()
        )?;
        
        let timesteps = Tensor::from_vec(
            vec![0.1, 0.5],
            Shape::from_dims(&[2]),
            device
        )?;
        
        let loss = adapter.compute_diffusion_loss(
            &pred,
            &target,
            &timesteps,
            LossType::L2,
            Some(5.0)
        )?;
        
        assert_eq!(loss.shape().dims(), &[]);  // Scalar loss
        
        Ok(())
    }
}