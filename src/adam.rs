//! Adam optimizer implementation

use crate::{config, parameter::Parameter, DType, Error, Result, Tensor, TensorId};
use std::collections::{hash_map::Entry, HashMap};

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

    /// Perform a single optimization step
    pub fn step(&mut self, parameters: &[Parameter]) -> Result<()> {
        self.t += 1;

        // Bias correction factors
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for param in parameters {
            if let Some(mut grad) = param.grad() {
                let param_id = param.id();
                let param_dtype = param.dtype()?;
                let state_dtype = config::select_optimizer_state_dtype(param_dtype);

                if state_dtype != param_dtype {
                    crate::log_once!(
                        "adam_state_dtype_mismatch",
                        "Adam states use {state:?} while params use {param:?}",
                        state = state_dtype,
                        param = param_dtype
                    );
                }

                if grad.dtype() != state_dtype {
                    grad = grad.to_dtype(state_dtype)?;
                }
                if self.weight_decay > 0.0 {
                    let param_tensor = param.tensor()?;
                    let param_adjust = if param_tensor.dtype() == state_dtype {
                        param_tensor
                    } else {
                        param_tensor.to_dtype(state_dtype)?
                    };
                    grad = grad.add(&param_adjust.mul_scalar(self.weight_decay)?)?;
                }

                if let Entry::Vacant(entry) = self.m.entry(param_id) {
                    entry.insert(grad.zeros_like_with_dtype(state_dtype)?);
                }
                if let Entry::Vacant(entry) = self.v.entry(param_id) {
                    entry.insert(grad.zeros_like_with_dtype(state_dtype)?);
                }

                // Get momentum buffers
                let m = self
                    .m
                    .get_mut(&param_id)
                    .ok_or_else(|| Error::Training("optimizer m state missing".into()))?;
                let v = self
                    .v
                    .get_mut(&param_id)
                    .ok_or_else(|| Error::Training("optimizer v state missing".into()))?;

                // Update biased first moment: m_t = β1 * m_{t-1} + (1 - β1) * g_t
                *m = m
                    .mul_scalar(self.beta1)?
                    .add(&grad.mul_scalar(1.0 - self.beta1)?)?;

                // Update biased second moment: v_t = β2 * v_{t-1} + (1 - β2) * g_t²
                let grad_sq = grad.mul(&grad)?;
                *v = v
                    .mul_scalar(self.beta2)?
                    .add(&grad_sq.mul_scalar(1.0 - self.beta2)?)?;

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

impl Default for Adam {
    fn default() -> Self {
        Self::new(0.001, 0.9, 0.999, 1e-8, 0.0)
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

    pub fn step(&mut self, parameters: &[Parameter]) -> Result<()> {
        self.adam.step(parameters)
    }

    pub fn zero_grad(&self, parameters: &[Parameter]) {
        self.adam.zero_grad(parameters)
    }
}

impl Default for AdamW {
    fn default() -> Self {
        Self::new(0.001, 0.9, 0.999, 1e-8, 0.01)
    }
}

impl Adam {
    fn state_dtype(&self, param_id: &TensorId) -> Option<(DType, DType)> {
        let m = self.m.get(param_id)?;
        let v = self.v.get(param_id)?;
        Some((m.dtype(), v.dtype()))
    }

    /// Return the total bytes consumed by optimizer state tensors.
    pub fn state_memory_bytes(&self) -> usize {
        let m_bytes: usize = self
            .m
            .values()
            .map(|tensor| tensor.shape().elem_count() * tensor.dtype().size_in_bytes())
            .sum();
        let v_bytes: usize = self
            .v
            .values()
            .map(|tensor| tensor.shape().elem_count() * tensor.dtype().size_in_bytes())
            .sum();
        m_bytes + v_bytes
    }

    /// Alias for compatibility with layout checks.
    pub fn state_bytes(&self) -> usize {
        self.state_memory_bytes()
    }
}

impl AdamW {
    /// Inspect the optimizer state tensor dtypes for a parameter.
    ///
    /// This is primarily intended for tests to ensure mixed-precision
    /// invariants (e.g. FP32 moment buffers) remain satisfied.
    pub fn debug_state_dtype(&self, param: &Parameter) -> Option<(DType, DType)> {
        self.adam.state_dtype(&param.id())
    }

    /// Return the total bytes consumed by optimizer state tensors.
    pub fn state_memory_bytes(&self) -> usize {
        self.adam.state_memory_bytes()
    }

    /// Alias matching the stabilization docs terminology.
    pub fn state_bytes(&self) -> usize {
        self.state_memory_bytes()
    }
}

#[cfg(all(test, feature = "legacy_full"))]
mod tests {
    use super::*;
    use crate::{Shape, Tensor};
    use cudarc::driver::CudaDevice;

    #[test]
    fn test_adam_step() -> Result<()> {
        let device = CudaDevice::new(0)?;

        // Create parameter
        let param = Parameter::randn(Shape::from_dims(&[10]), 0.0, 1.0, device)?;
        let before = param.tensor()?.to_vec()?;

        // Set a gradient
        let grad = Tensor::ones(Shape::from_dims(&[10]), param.tensor()?.device.clone())?;
        param.set_grad(grad)?;

        // Create optimizer and take a step
        let mut optimizer = Adam::default();
        optimizer.step(&[param.clone()])?;

        // Check that parameter was updated
        let new_value = param.tensor()?.to_vec()?;
        assert!(new_value[0] < before[0]);

        Ok(())
    }
}
