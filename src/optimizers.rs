#![allow(unused_imports, dead_code)]
// Legacy optimizer scaffolding preserved for future rewrite.

use crate::{
    config,
    strict::{allow_clone, allow_f32_in_kernel_scoped, record_param_f32_store, scope, GuardMode},
    Error, Result, Shape, Tensor,
};
use cudarc::driver::CudaDevice;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

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
        scope("adam.step", GuardMode::env_default(), || {
            self.step += 1;
            let step = self.step as f32;

            let bias_correction1 = 1.0 - self.config.beta1.powf(step);
            let bias_correction2 = 1.0 - self.config.beta2.powf(step);
            let step_size = self.config.lr * (bias_correction2.sqrt() / bias_correction1);

            for (param_id, param, grad) in params {
                let param_dtype = param.dtype();
                let state_dtype = config::select_optimizer_state_dtype(param_dtype);

                if state_dtype != param_dtype {
                    crate::log_once!(
                        "adam_state_dtype_mismatch",
                        "Adam states use {state:?} while params use {param:?}",
                        state = state_dtype,
                        param = param_dtype
                    );
                }

                if !self.states.contains_key(param_id) {
                    let zeros_m = param.zeros_like_with_dtype(state_dtype)?;
                    let zeros_v = param.zeros_like_with_dtype(state_dtype)?;
                    self.states.insert(*param_id, (zeros_m, zeros_v));
                }

                let (momentum, variance) = self
                    .states
                    .get_mut(param_id)
                    .ok_or_else(|| Error::Training("adam: state missing for parameter".into()))?;

                let grad_cow = if grad.dtype() == state_dtype && grad.storage_dtype() == state_dtype
                {
                    Cow::Borrowed(*grad)
                } else {
                    Cow::Owned(allow_f32_in_kernel_scoped(|| grad.to_dtype(state_dtype))?)
                };
                let grad_state = grad_cow.as_ref();

                *momentum = momentum
                    .mul_scalar(self.config.beta1)?
                    .add(&grad_state.mul_scalar(1.0 - self.config.beta1)?)?;

                let grad_squared = grad_state.mul(grad_state)?;
                *variance = variance
                    .mul_scalar(self.config.beta2)?
                    .add(&grad_squared.mul_scalar(1.0 - self.config.beta2)?)?;

                let variance_sqrt = variance.sqrt()?;
                let denominator = variance_sqrt.add_scalar(self.config.eps)?;
                let mut update = momentum.div(&denominator)?;

                if self.config.weight_decay > 0.0 {
                    let param_cow =
                        if param_dtype == state_dtype && param.storage_dtype() == state_dtype {
                            Cow::Borrowed(&**param)
                        } else {
                            Cow::Owned(allow_f32_in_kernel_scoped(|| param.to_dtype(state_dtype))?)
                        };
                    let decay = param_cow.as_ref().mul_scalar(self.config.weight_decay)?;
                    update = decay.add(&update)?;
                }

                let update =
                    if update.dtype() == param_dtype && update.storage_dtype() == param_dtype {
                        update
                    } else {
                        allow_f32_in_kernel_scoped(|| update.to_dtype(param_dtype))?
                    };

                let mut new_param = param.update_weights(&update, step_size)?;
                if new_param.dtype() != param_dtype || new_param.storage_dtype() != param_dtype {
                    if new_param.dtype() != param_dtype {
                        record_param_f32_store("adam.step", param.shape());
                    }
                    new_param = allow_f32_in_kernel_scoped(|| new_param.to_dtype(param_dtype))?;
                }
                param.copy_(&new_param)?;
            }

            Ok(())
        })
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
        scope("sgd.step", GuardMode::env_default(), || {
            for (param_id, param, grad) in params {
                let param_dtype = param.dtype();
                let state_dtype = config::select_optimizer_state_dtype(param_dtype);

                if self.config.momentum > 0.0 && state_dtype != param_dtype {
                    crate::log_once!(
                        "sgd_state_dtype_mismatch",
                        "SGD states use {state:?} while params use {param:?}",
                        state = state_dtype,
                        param = param_dtype
                    );
                }

                let grad_cow = if grad.dtype() == state_dtype && grad.storage_dtype() == state_dtype
                {
                    Cow::Borrowed(*grad)
                } else {
                    Cow::Owned(allow_f32_in_kernel_scoped(|| grad.to_dtype(state_dtype))?)
                };
                let mut grad_state = match grad_cow {
                    Cow::Borrowed(t) => {
                        let _guard = allow_clone();
                        t.clone()
                    }
                    Cow::Owned(t) => t,
                };

                if self.config.weight_decay > 0.0 {
                    let param_cow =
                        if param_dtype == state_dtype && param.storage_dtype() == state_dtype {
                            Cow::Borrowed(&**param)
                        } else {
                            Cow::Owned(allow_f32_in_kernel_scoped(|| param.to_dtype(state_dtype))?)
                        };
                    grad_state = grad_state
                        .add(&param_cow.as_ref().mul_scalar(self.config.weight_decay)?)?;
                }

                let mut update = {
                    let _guard = allow_clone();
                    grad_state.clone()
                };

                if self.config.momentum > 0.0 {
                    if !self.states.contains_key(param_id) {
                        self.states
                            .insert(*param_id, grad_state.zeros_like_with_dtype(state_dtype)?);
                    }

                    let velocity = self
                        .states
                        .get_mut(param_id)
                        .ok_or_else(|| Error::Training("sgd: velocity state missing".into()))?;
                    let new_velocity = velocity
                        .mul_scalar(self.config.momentum)?
                        .add(&grad_state)?;
                    {
                        let _guard = allow_clone();
                        *velocity = new_velocity.clone();
                    }

                    update = if self.config.nesterov {
                        grad_state.add(&new_velocity.mul_scalar(self.config.momentum)?)?
                    } else {
                        new_velocity
                    };
                }

                let update =
                    if update.dtype() == param_dtype && update.storage_dtype() == param_dtype {
                        update
                    } else {
                        allow_f32_in_kernel_scoped(|| update.to_dtype(param_dtype))?
                    };

                let mut new_param = param.update_weights(&update, self.config.lr)?;
                if new_param.dtype() != param_dtype || new_param.storage_dtype() != param_dtype {
                    if new_param.dtype() != param_dtype {
                        record_param_f32_store("sgd.step", param.shape());
                    }
                    new_param = allow_f32_in_kernel_scoped(|| new_param.to_dtype(param_dtype))?;
                }
                param.copy_(&new_param)?;
            }

            Ok(())
        })
    }

    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.states.clear();
    }
}

// Helper operations are now defined in tensor_ops_extended.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam_optimizer() -> Result<()> {
        let device = CudaDevice::new(0)?;

        // Create test parameters
        let mut param = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0],
            Shape::from_dims(&[2, 2]),
            device.clone(),
        )?;

        let grad = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], Shape::from_dims(&[2, 2]), device)?;

        // Create optimizer
        let mut optimizer = Adam::new(AdamConfig::default());

        // Take a step
        let param_id = 0;
        let mut param_refs = [(param_id, &mut param, &grad)];
        optimizer.step(&mut param_refs)?;

        // Check that parameters were updated
        let updated = param.to_vec()?;
        assert!(updated[0] < 1.0); // Should decrease
        assert!(updated[1] < 2.0);
        assert!(updated[2] < 3.0);
        assert!(updated[3] < 4.0);

        Ok(())
    }
}
