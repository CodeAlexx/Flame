use crate::{Tensor, Result, FlameError};
use std::collections::HashMap;

/// Adam optimizer configuration
pub struct AdamConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

/// Adam optimizer implementation
pub struct Adam {
    config: AdamConfig,
    step: usize,
    // Parameter ID -> (momentum, variance)
    states: HashMap<usize, (Tensor, Tensor)>,
}

impl Adam {
    pub fn new(config: AdamConfig) -> Self {
        Self {
            config,
            step: 0,
            states: HashMap::new(),
        }
    }
    
    /// Update parameters using Adam algorithm
    pub fn step(&mut self, params: &mut [(usize, &mut Tensor, &Tensor)]) -> Result<()> {
        self.step += 1;
        let step = self.step as f32;
        
        // Bias correction
        let bias_correction1 = 1.0 - self.config.beta1.powf(step);
        let bias_correction2 = 1.0 - self.config.beta2.powf(step);
        let step_size = self.config.lr * (bias_correction2.sqrt() / bias_correction1);
        
        for (param_id, param, grad) in params {
            // Initialize state if needed
            if !self.states.contains_key(param_id) {
                let zeros_m = Tensor::zeros(param.shape().clone(), param.device().clone())?;
                let zeros_v = Tensor::zeros(param.shape().clone(), param.device().clone())?;
                self.states.insert(*param_id, (zeros_m, zeros_v));
            }
            
            let (momentum, variance) = self.states.get_mut(param_id).unwrap();
            
            // Update biased first moment estimate
            // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            let momentum_update = momentum.mul_scalar(self.config.beta1)?
                .add(&grad.mul_scalar(1.0 - self.config.beta1)?)?;
            *momentum = momentum_update;
            
            // Update biased second raw moment estimate
            // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            let grad_squared = grad.square()?;
            let variance_update = variance.mul_scalar(self.config.beta2)?
                .add(&grad_squared.mul_scalar(1.0 - self.config.beta2)?)?;
            *variance = variance_update;
            
            // Compute update
            // theta_t = theta_{t-1} - step_size * m_t / (sqrt(v_t) + eps)
            let variance_sqrt = variance.sqrt()?;
            let denominator = variance_sqrt.add_scalar(self.config.eps)?;
            let update = momentum.div(&denominator)?;
            
            // Apply weight decay if needed
            if self.config.weight_decay > 0.0 {
                let decay = param.mul_scalar(self.config.weight_decay)?;
                param.update_weights(&decay.add(&update)?, step_size)?;
            } else {
                param.update_weights(&update, step_size)?;
            }
        }
        
        Ok(())
    }
    
    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.step = 0;
        self.states.clear();
    }
}

/// SGD optimizer configuration
pub struct SGDConfig {
    pub lr: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    pub nesterov: bool,
}

impl Default for SGDConfig {
    fn default() -> Self {
        Self {
            lr: 0.01,
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        }
    }
}

/// SGD optimizer implementation
pub struct SGD {
    config: SGDConfig,
    states: HashMap<usize, Tensor>,
}

impl SGD {
    pub fn new(config: SGDConfig) -> Self {
        Self {
            config,
            states: HashMap::new(),
        }
    }
    
    /// Update parameters using SGD algorithm
    pub fn step(&mut self, params: &mut [(usize, &mut Tensor, &Tensor)]) -> Result<()> {
        for (param_id, param, grad) in params {
            let mut update = grad.clone()?;
            
            // Apply weight decay
            if self.config.weight_decay > 0.0 {
                let decay = param.mul_scalar(self.config.weight_decay)?;
                update = update.add(&decay)?;
            }
            
            // Apply momentum if enabled
            if self.config.momentum > 0.0 {
                if !self.states.contains_key(param_id) {
                    let zeros = Tensor::zeros(param.shape().clone(), param.device().clone())?;
                    self.states.insert(*param_id, zeros);
                }
                
                let velocity = self.states.get_mut(param_id).unwrap();
                let new_velocity = velocity.mul_scalar(self.config.momentum)?
                    .add(&update)?;
                *velocity = new_velocity.clone()?;
                
                if self.config.nesterov {
                    // Nesterov momentum
                    update = grad.add(&new_velocity.mul_scalar(self.config.momentum)?)?;
                } else {
                    update = new_velocity;
                }
            }
            
            // Update parameters
            param.update_weights(&update, self.config.lr)?;
        }
        
        Ok(())
    }
    
    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.states.clear();
    }
}

// Helper trait for tensor operations needed by optimizers
impl Tensor {
    /// Element-wise square root
    pub fn sqrt(&self) -> Result<Tensor> {
        let data = self.to_vec()?;
        let result: Vec<f32> = data.iter().map(|&x| x.sqrt()).collect();
        Tensor::from_vec(result, self.shape.clone(), self.device.clone())
    }
    
    /// Element-wise division
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape != other.shape {
            return Err(FlameError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }
        
        let a_data = self.to_vec()?;
        let b_data = other.to_vec()?;
        let mut result = vec![0.0f32; a_data.len()];
        
        for i in 0..a_data.len() {
            result[i] = a_data[i] / b_data[i];
        }
        
        Tensor::from_vec(result, self.shape.clone(), self.device.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_adam_optimizer() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        
        // Create test parameters
        let mut param = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0],
            Shape::from_dims(&[2, 2]),
            device.clone()
        )?;
        
        let grad = Tensor::from_vec(
            vec![0.1, 0.2, 0.3, 0.4],
            Shape::from_dims(&[2, 2]),
            device
        )?;
        
        // Create optimizer
        let mut optimizer = Adam::new(AdamConfig::default());
        
        // Take a step
        let param_id = 0;
        optimizer.step(&mut vec![(param_id, &mut param, &grad)])?;
        
        // Check that parameters were updated
        let updated = param.to_vec()?;
        assert!(updated[0] < 1.0); // Should decrease
        assert!(updated[1] < 2.0);
        assert!(updated[2] < 3.0);
        assert!(updated[3] < 4.0);
        
        Ok(())
    }
}