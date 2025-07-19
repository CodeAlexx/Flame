//! Diffusion model samplers for image generation
//! 
//! This module provides various sampling algorithms for diffusion models,
//! including Euler, DPM++, and other popular methods.

use crate::{Tensor, Result, FlameError};

/// Base trait for all diffusion samplers
pub trait DiffusionSampler {
    /// Single denoising step
    fn step(
        &self,
        model_output: &Tensor,
        timestep: f32,
        sample: &Tensor,
        generator: Option<&mut dyn FnMut() -> f32>,
    ) -> Result<Tensor>;
    
    /// Get the number of inference steps
    fn num_inference_steps(&self) -> usize;
    
    /// Set the number of inference steps
    fn set_timesteps(&mut self, num_inference_steps: usize);
    
    /// Get timesteps for sampling
    fn timesteps(&self) -> &[f32];
    
    /// Initialize noise
    fn init_noise_sigma(&self) -> f32 {
        1.0
    }
}

/// Euler sampler - simple and efficient first-order method
pub struct EulerSampler {
    num_train_timesteps: usize,
    num_inference_steps: usize,
    timesteps: Vec<f32>,
    sigmas: Vec<f32>,
    prediction_type: PredictionType,
}

/// Prediction type for model output
#[derive(Clone, Copy, Debug)]
pub enum PredictionType {
    Epsilon,     // Model predicts noise
    VPrediction, // Model predicts velocity
    Sample,      // Model predicts clean sample
}

impl EulerSampler {
    pub fn new(
        num_train_timesteps: usize,
        prediction_type: PredictionType,
    ) -> Self {
        Self {
            num_train_timesteps,
            num_inference_steps: 50,
            timesteps: Vec::new(),
            sigmas: Vec::new(),
            prediction_type,
        }
    }
    
    /// Calculate sigma from timestep
    fn timestep_to_sigma(&self, t: f32) -> f32 {
        // Simple linear schedule for now
        // TODO: Support different noise schedules
        let alpha = 1.0 - (t / self.num_train_timesteps as f32);
        ((1.0 - alpha.powi(2)) / alpha.powi(2)).sqrt()
    }
    
    /// Get predicted original sample from model output
    fn get_pred_original_sample(
        &self,
        model_output: &Tensor,
        timestep: f32,
        sample: &Tensor,
    ) -> Result<Tensor> {
        let sigma = self.timestep_to_sigma(timestep);
        
        match self.prediction_type {
            PredictionType::Epsilon => {
                // x0 = (x_t - sigma * eps) / alpha
                let alpha = (1.0 / (1.0 + sigma * sigma)).sqrt();
                let scaled_model_output = model_output.mul_scalar(sigma)?;
                let x0 = sample.sub(&scaled_model_output)?;
                x0.mul_scalar(1.0 / alpha)
            }
            PredictionType::VPrediction => {
                // x0 = alpha * x_t - sigma * v
                let alpha = (1.0 / (1.0 + sigma * sigma)).sqrt();
                let scaled_sample = sample.mul_scalar(alpha)?;
                let scaled_model_output = model_output.mul_scalar(sigma)?;
                scaled_sample.sub(&scaled_model_output)
            }
            PredictionType::Sample => {
                // Model directly predicts x0
                Ok(model_output.clone()?)
            }
        }
    }
}

impl DiffusionSampler for EulerSampler {
    fn step(
        &self,
        model_output: &Tensor,
        timestep: f32,
        sample: &Tensor,
        _generator: Option<&mut dyn FnMut() -> f32>,
    ) -> Result<Tensor> {
        // Get current timestep index
        let step_index = self.timesteps.iter()
            .position(|&t| (t - timestep).abs() < 1e-6)
            .ok_or_else(|| FlameError::InvalidOperation("Timestep not found".into()))?;
        
        let sigma = self.sigmas[step_index];
        
        // Get predicted original sample
        let pred_original_sample = self.get_pred_original_sample(model_output, timestep, sample)?;
        
        // Calculate derivative
        let derivative = sample.sub(&pred_original_sample)?.mul_scalar(1.0 / sigma)?;
        
        // Euler step
        let dt = if step_index < self.sigmas.len() - 1 {
            self.sigmas[step_index + 1] - sigma
        } else {
            0.0
        };
        
        let prev_sample = sample.add(&derivative.mul_scalar(dt)?)?;
        
        Ok(prev_sample)
    }
    
    fn num_inference_steps(&self) -> usize {
        self.num_inference_steps
    }
    
    fn set_timesteps(&mut self, num_inference_steps: usize) {
        self.num_inference_steps = num_inference_steps;
        
        // Linear spacing in timestep space
        let step_ratio = self.num_train_timesteps / num_inference_steps;
        self.timesteps = (0..num_inference_steps)
            .map(|i| ((num_inference_steps - 1 - i) * step_ratio) as f32)
            .collect();
        
        // Calculate corresponding sigmas
        self.sigmas = self.timesteps.iter()
            .map(|&t| self.timestep_to_sigma(t))
            .collect();
        
        // Add final sigma of 0
        self.sigmas.push(0.0);
    }
    
    fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }
}

/// DPM++ 2M sampler - second-order multistep method
pub struct DPMPlusPlus2MSampler {
    num_train_timesteps: usize,
    num_inference_steps: usize,
    timesteps: Vec<f32>,
    sigmas: Vec<f32>,
    prediction_type: PredictionType,
    sample_history: Option<Tensor>,
}

impl DPMPlusPlus2MSampler {
    pub fn new(
        num_train_timesteps: usize,
        prediction_type: PredictionType,
    ) -> Self {
        Self {
            num_train_timesteps,
            num_inference_steps: 25,
            timesteps: Vec::new(),
            sigmas: Vec::new(),
            prediction_type,
            sample_history: None,
        }
    }
    
    /// Calculate sigma from timestep
    fn timestep_to_sigma(&self, t: f32) -> f32 {
        // Simple linear schedule for now
        let alpha = 1.0 - (t / self.num_train_timesteps as f32);
        ((1.0 - alpha.powi(2)) / alpha.powi(2)).sqrt()
    }
    
    /// Convert between parameterizations
    fn get_pred_x0(&self, model_output: &Tensor, timestep: f32, sample: &Tensor) -> Result<Tensor> {
        let sigma = self.timestep_to_sigma(timestep);
        
        match self.prediction_type {
            PredictionType::Epsilon => {
                // x0 = (x_t - sigma * eps) / alpha
                let alpha = (1.0 / (1.0 + sigma * sigma)).sqrt();
                let scaled_noise = model_output.mul_scalar(sigma)?;
                sample.sub(&scaled_noise)?.mul_scalar(1.0 / alpha)
            }
            PredictionType::VPrediction => {
                // x0 = alpha * x_t - sigma * v
                let alpha = (1.0 / (1.0 + sigma * sigma)).sqrt();
                let scaled_sample = sample.mul_scalar(alpha)?;
                let scaled_velocity = model_output.mul_scalar(sigma)?;
                scaled_sample.sub(&scaled_velocity)
            }
            PredictionType::Sample => {
                Ok(model_output.clone()?)
            }
        }
    }
}

impl DiffusionSampler for DPMPlusPlus2MSampler {
    fn step(
        &self,
        model_output: &Tensor,
        timestep: f32,
        sample: &Tensor,
        _generator: Option<&mut dyn FnMut() -> f32>,
    ) -> Result<Tensor> {
        // Get current timestep index
        let step_index = self.timesteps.iter()
            .position(|&t| (t - timestep).abs() < 1e-6)
            .ok_or_else(|| FlameError::InvalidOperation("Timestep not found".into()))?;
        
        let sigma = self.sigmas[step_index];
        let sigma_next = self.sigmas[step_index + 1];
        
        // Get predicted x0
        let pred_x0 = self.get_pred_x0(model_output, timestep, sample)?;
        
        // DPM++ 2M uses a second-order method
        if step_index == 0 {
            // First step: use Euler
            let sigma_ratio = sigma_next / sigma;
            let sample_next = pred_x0.mul_scalar(sigma_ratio)?
                .add(&sample.mul_scalar(1.0 - sigma_ratio)?)?;
            Ok(sample_next)
        } else {
            // Subsequent steps: use second-order
            let h = (sigma_next - sigma).ln();
            let h_prev = if step_index > 1 {
                (sigma - self.sigmas[step_index - 1]).ln()
            } else {
                h
            };
            
            let r = h_prev / h;
            let d = pred_x0;
            
            // Linear multistep coefficients
            let coeff1 = (1.0 + 1.0 / (2.0 * r)).exp();
            let coeff2 = 1.0 / (2.0 * r);
            
            // Update sample
            let scaled_d = d.mul_scalar(coeff2)?;
            let sample_next = sample.mul_scalar(sigma_next / sigma)?
                .add(&scaled_d.mul_scalar(coeff1 - 1.0)?)?;
            
            Ok(sample_next)
        }
    }
    
    fn num_inference_steps(&self) -> usize {
        self.num_inference_steps
    }
    
    fn set_timesteps(&mut self, num_inference_steps: usize) {
        self.num_inference_steps = num_inference_steps;
        
        // Linear spacing
        let step_ratio = self.num_train_timesteps / num_inference_steps;
        self.timesteps = (0..num_inference_steps)
            .map(|i| ((num_inference_steps - 1 - i) * step_ratio) as f32)
            .collect();
        
        // Calculate sigmas
        self.sigmas = self.timesteps.iter()
            .map(|&t| self.timestep_to_sigma(t))
            .collect();
        
        // Add final sigma
        self.sigmas.push(0.0);
        
        // Reset history
        self.sample_history = None;
    }
    
    fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }
}

/// DDIM sampler - deterministic implicit method
pub struct DDIMSampler {
    num_train_timesteps: usize,
    num_inference_steps: usize,
    timesteps: Vec<f32>,
    alphas: Vec<f32>,
    prediction_type: PredictionType,
    eta: f32, // Controls stochasticity (0 = deterministic)
}

impl DDIMSampler {
    pub fn new(
        num_train_timesteps: usize,
        prediction_type: PredictionType,
        eta: f32,
    ) -> Self {
        Self {
            num_train_timesteps,
            num_inference_steps: 50,
            timesteps: Vec::new(),
            alphas: Vec::new(),
            prediction_type,
            eta,
        }
    }
}

impl DiffusionSampler for DDIMSampler {
    fn step(
        &self,
        model_output: &Tensor,
        timestep: f32,
        sample: &Tensor,
        generator: Option<&mut dyn FnMut() -> f32>,
    ) -> Result<Tensor> {
        // Get current timestep index
        let step_index = self.timesteps.iter()
            .position(|&t| (t - timestep).abs() < 1e-6)
            .ok_or_else(|| FlameError::InvalidOperation("Timestep not found".into()))?;
        
        let alpha_prod_t = self.alphas[step_index];
        let alpha_prod_t_prev = if step_index < self.alphas.len() - 1 {
            self.alphas[step_index + 1]
        } else {
            1.0
        };
        
        let beta_prod_t = 1.0 - alpha_prod_t;
        let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;
        
        // Get predicted x0
        let pred_x0 = match self.prediction_type {
            PredictionType::Epsilon => {
                // x0 = (x_t - sqrt(1 - alpha) * eps) / sqrt(alpha)
                let scaled_noise = model_output.mul_scalar(beta_prod_t.sqrt())?;
                sample.sub(&scaled_noise)?.mul_scalar(1.0 / alpha_prod_t.sqrt())
            }
            _ => {
                return Err(FlameError::InvalidOperation(
                    "DDIM only supports epsilon prediction".into()
                ));
            }
        }?;
        
        // Compute variance
        let variance = self.eta * self.eta * beta_prod_t_prev / beta_prod_t 
            * (1.0 - alpha_prod_t / alpha_prod_t_prev);
        let std_dev = variance.sqrt();
        
        // Direction pointing to x_t
        let pred_sample_direction = model_output.mul_scalar(
            (beta_prod_t_prev - std_dev * std_dev).sqrt()
        )?;
        
        // Compute previous sample
        let prev_sample = pred_x0.mul_scalar(alpha_prod_t_prev.sqrt())?
            .add(&pred_sample_direction)?;
        
        // Add noise if eta > 0
        if self.eta > 0.0 {
            let noise = if let Some(gen) = generator {
                // Generate noise using provided generator
                let mut noise_data = Vec::with_capacity(sample.shape().elem_count());
                for _ in 0..sample.shape().elem_count() {
                    noise_data.push(gen());
                }
                Tensor::from_vec(noise_data, sample.shape().clone(), sample.device().clone())?
            } else {
                // Use standard normal distribution
                Tensor::randn(sample.shape().clone(), 0.0, 1.0, sample.device().clone())?
            };
            
            prev_sample.add(&noise.mul_scalar(std_dev)?)
        } else {
            Ok(prev_sample)
        }
    }
    
    fn num_inference_steps(&self) -> usize {
        self.num_inference_steps
    }
    
    fn set_timesteps(&mut self, num_inference_steps: usize) {
        self.num_inference_steps = num_inference_steps;
        
        // Linear spacing
        let step_ratio = self.num_train_timesteps / num_inference_steps;
        self.timesteps = (0..num_inference_steps)
            .map(|i| ((num_inference_steps - 1 - i) * step_ratio) as f32)
            .collect();
        
        // Calculate alphas (cumulative product of 1 - beta)
        self.alphas = self.timesteps.iter()
            .map(|&t| {
                let beta = 0.0001 + (0.02 - 0.0001) * (t / self.num_train_timesteps as f32);
                (1.0 - beta).powf(t + 1.0)
            })
            .collect();
    }
    
    fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }
}

/// Factory function to create samplers
pub fn create_sampler(
    sampler_type: &str,
    num_train_timesteps: usize,
    prediction_type: PredictionType,
) -> Result<Box<dyn DiffusionSampler>> {
    match sampler_type.to_lowercase().as_str() {
        "euler" => Ok(Box::new(EulerSampler::new(num_train_timesteps, prediction_type))),
        "dpm++2m" | "dpmpp2m" => Ok(Box::new(DPMPlusPlus2MSampler::new(num_train_timesteps, prediction_type))),
        "ddim" => Ok(Box::new(DDIMSampler::new(num_train_timesteps, prediction_type, 0.0))),
        _ => Err(FlameError::InvalidOperation(
            format!("Unknown sampler type: {}", sampler_type)
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CudaDevice;
    
    #[test]
    fn test_euler_sampler() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        
        let mut sampler = EulerSampler::new(1000, PredictionType::Epsilon);
        sampler.set_timesteps(50);
        
        // Test shapes
        let batch_size = 2;
        let channels = 4;
        let height = 64;
        let width = 64;
        
        let sample = Tensor::randn(
            Shape::from_dims(&[batch_size, channels, height, width]),
            0.0, 1.0, device.clone()
        )?;
        
        let model_output = Tensor::randn(
            Shape::from_dims(&[batch_size, channels, height, width]),
            0.0, 0.1, device.clone()
        )?;
        
        let timestep = sampler.timesteps()[0];
        let next_sample = sampler.step(&model_output, timestep, &sample, None)?;
        
        assert_eq!(next_sample.shape().dims(), sample.shape().dims());
        
        println!("Euler sampler test passed!");
        Ok(())
    }
    
    #[test]
    fn test_dpm_sampler() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        
        let mut sampler = DPMPlusPlus2MSampler::new(1000, PredictionType::Epsilon);
        sampler.set_timesteps(25);
        
        assert_eq!(sampler.timesteps().len(), 25);
        assert_eq!(sampler.sigmas.len(), 26); // includes final 0
        
        println!("DPM++ sampler test passed!");
        Ok(())
    }
}