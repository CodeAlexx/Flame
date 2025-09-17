//! Adam optimizer implementation

use crate::{Tensor, TensorId, Result, FlameError, parameter::Parameter};
use std::collections::HashMap;

/// Adam optimizer with momentum and adaptive learning rates
pub struct Adam {
    /// Learning rate
    lr: f32,
    /// Beta1 - exponential decay rate for first moment
    beta1: f32,
    /// Beta2 - exponential decay rate for second moment  
    beta2: f32,
    /// Small constant for numerical stability
    eps: f32,
    /// Current timestep
    t: u32,
    /// First moment estimates
    m: HashMap<TensorId, Tensor>,
    /// Second moment estimates
    v: HashMap<TensorId, Tensor>,
    /// Weight decay coefficient
    weight_decay: f32,
}

impl Adam {
    /// Create a new Adam optimizer
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
            weight_decay,
        }
    }
    
    /// Create with default parameters
    pub fn default() -> Self {
        Self::new(0.001, 0.9, 0.999, 1e-8, 0.0)
    }
    
    /// Perform a single optimization step
    pub fn step(&mut self, parameters: &[Parameter]) -> Result<()> {
        self.t += 1;
        
        // Bias correction factors
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);
        
        for param in parameters {
            if let Some(grad) = param.grad() {
                let param_id = param.id();
                
                // Apply weight decay if needed
                let grad = if self.weight_decay > 0.0 {
                    let param_tensor = param.tensor()?;
                    grad.add(&param_tensor.mul_scalar(self.weight_decay)?)?
                } else {
                    grad
                };
                
                // Initialize momentum buffers if needed
                if !self.m.contains_key(&param_id) {
                    self.m.insert(param_id, Tensor::zeros(grad.shape.clone(), grad.device.clone())?);
                    self.v.insert(param_id, Tensor::zeros(grad.shape.clone(), grad.device.clone())?);
                }
                
                // Get momentum buffers
                let m = self.m.get_mut(&param_id)
                    .ok_or_else(|| FlameError::Training("optimizer m state missing".into()))?;
                let v = self.v.get_mut(&param_id)
                    .ok_or_else(|| FlameError::Training("optimizer v state missing".into()))?;
                
                // Update biased first moment: m_t = β1 * m_{t-1} + (1 - β1) * g_t
                *m = m.mul_scalar(self.beta1)?.add(&grad.mul_scalar(1.0 - self.beta1)?)?;
                
                // Update biased second moment: v_t = β2 * v_{t-1} + (1 - β2) * g_t²
                let grad_sq = grad.mul(&grad)?;
                *v = v.mul_scalar(self.beta2)?.add(&grad_sq.mul_scalar(1.0 - self.beta2)?)?;
                
                // Compute bias-corrected moments
                let m_hat = m.div_scalar(bias_correction1)?;
                let v_hat = v.div_scalar(bias_correction2)?;
                
                // Compute update: lr * m_hat / (sqrt(v_hat) + eps)
                let v_sqrt = v_hat.sqrt()?;
                let denominator = v_sqrt.add_scalar(self.eps)?;
                let update = m_hat.div(&denominator)?.mul_scalar(self.lr)?;
                
                // Apply update
                param.apply_update(&update)?;
            }
        }
        
        Ok(())
    }
    
    /// Zero all gradients
    pub fn zero_grad(&self, parameters: &[Parameter]) {
        for param in parameters {
            param.zero_grad();
        }
    }
}

/// AdamW optimizer (Adam with decoupled weight decay)
pub struct AdamW {
    adam: Adam,
}

impl AdamW {
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            adam: Adam::new(lr, beta1, beta2, eps, weight_decay),
        }
    }
    
    pub fn default() -> Self {
        Self::new(0.001, 0.9, 0.999, 1e-8, 0.01)
    }
    
    pub fn step(&mut self, parameters: &[Parameter]) -> Result<()> {
        self.adam.step(parameters)
    }
    
    pub fn zero_grad(&self, parameters: &[Parameter]) {
        self.adam.zero_grad(parameters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Shape, Tensor};
    use cudarc::driver::CudaDevice;
    
    #[test]
    fn test_adam_step() -> Result<()> {
        let device = CudaDevice::new(0)?;
        
        // Create parameter
        let param = Parameter::randn(Shape::from_dims(&[10]), 0.0, 1.0, device)?;
        
        // Set a gradient
        let grad = Tensor::ones(Shape::from_dims(&[10]), param.tensor()?.device.clone())?;
        param.set_grad(grad)?;
        
        // Create optimizer and take a step
        let mut optimizer = Adam::default();
        optimizer.step(&[param.clone()])?;
        
        // Check that parameter was updated
        let new_value = param.tensor()?.to_vec()?;
        // After one step with gradient of 1.0 and lr=0.001, values should decrease
        assert!(new_value[0] < 0.0);
        
        Ok(())
    }
}
