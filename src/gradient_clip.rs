use crate::{Tensor, Result};
#[allow(unused_imports)]
use crate::Error;
use crate::cuda_gradient_ops::CudaGradientOps;
use crate::ops_ext::mean_all_f32;

/// Gradient clipping strategies
#[derive(Debug, Clone)]
pub enum GradientClipStrategy {
    /// Clip by global norm
    ClipByNorm { max_norm: f32 },
    /// Clip by value (element-wise)
    ClipByValue { min_value: f32, max_value: f32 },
    /// Adaptive gradient clipping
    AdaptiveClip { clip_factor: f32 },
}

/// Gradient clipper
pub struct GradientClipper {
    pub strategy: GradientClipStrategy,
}

impl GradientClipper {
    /// Create a new gradient clipper with specified strategy
    pub fn new(strategy: GradientClipStrategy) -> Self {
        Self { strategy }
    }
    
    /// Clip gradients by global norm
    pub fn clip_by_norm(max_norm: f32) -> Self {
        Self {
            strategy: GradientClipStrategy::ClipByNorm { max_norm },
        }
    }
    
    /// Clip gradients by value
    pub fn clip_by_value(min_value: f32, max_value: f32) -> Self {
        Self {
            strategy: GradientClipStrategy::ClipByValue { min_value, max_value },
        }
    }
    
    /// Adaptive gradient clipping
    pub fn adaptive(clip_factor: f32) -> Self {
        Self {
            strategy: GradientClipStrategy::AdaptiveClip { clip_factor },
        }
    }
    
    /// Apply gradient clipping to a set of gradients
    pub fn clip_grads(&self, grads: &mut [&mut Tensor]) -> Result<f32> {
        match &self.strategy {
            GradientClipStrategy::ClipByNorm { max_norm } => {
                self.clip_grads_by_norm(grads, *max_norm)
            }
            GradientClipStrategy::ClipByValue { min_value, max_value } => {
                self.clip_grads_by_value(grads, *min_value, *max_value)
            }
            GradientClipStrategy::AdaptiveClip { clip_factor } => {
                self.adaptive_clip_grads(grads, *clip_factor)
            }
        }
    }
    
    /// Clip gradients by global norm
    fn clip_grads_by_norm(&self, grads: &mut [&mut Tensor], max_norm: f32) -> Result<f32> {
        // Compute total norm
        let total_norm = self.compute_grad_norm(grads)?;
        
        if total_norm > max_norm {
            // Scale all gradients
            let scale_factor = max_norm / total_norm;
            
            for grad in grads {
                **grad = grad.mul_scalar(scale_factor)?;
            }
        }
        
        Ok(total_norm)
    }
    
    /// Clip gradients element-wise by value
    fn clip_grads_by_value(&self, grads: &mut [&mut Tensor], min_value: f32, max_value: f32) -> Result<f32> {
        let mut total_norm_sq = 0.0f32;

        for grad in grads {
            let grad_tensor: &mut Tensor = *grad;
            let mut ops = CudaGradientOps::new(grad_tensor.device().clone())?;
            let norm = ops.compute_l2_norm(grad_tensor)?;
            total_norm_sq += norm * norm;
            ops.clamp_tensor(grad_tensor, min_value, max_value)?;
        }
        Ok(total_norm_sq.sqrt())
    }
    
    /// Adaptive gradient clipping (based on parameter norm)
    fn adaptive_clip_grads(&self, grads: &mut [&mut Tensor], clip_factor: f32) -> Result<f32> {
        // For adaptive clipping, we need both gradients and parameters
        // Implement gradient clipping by global norm on FP32 grads
        // that clips based on gradient statistics

        let mut total_norm = 0.0f32;

        for grad in grads.iter_mut() {
            let grad_tensor: &mut Tensor = *grad;
            let mut ops = CudaGradientOps::new(grad_tensor.device().clone())?;
            let grad_norm = ops.compute_l2_norm(grad_tensor)?;

            if grad_norm > 0.0 {
                let mean = mean_all_f32(grad_tensor)?;
                let centered = grad_tensor.add_scalar(-mean)?;
                let centered_sq = centered.square()?;
                let variance = mean_all_f32(&centered_sq)?;
                let std_dev = variance.sqrt();

                let threshold = clip_factor * (std_dev + 1e-6);

                if grad_norm > threshold {
                    let scale = threshold / grad_norm;
                    ops.scale_gradient(grad_tensor, scale)?;
                }
            }

            total_norm += grad_norm * grad_norm;
        }

        Ok(total_norm.sqrt())
    }
    
    /// Compute the global norm of gradients
    pub fn compute_grad_norm(&self, grads: &[&mut Tensor]) -> Result<f32> {
        let mut total_norm_sq = 0.0f32;

        for grad in grads {
            let mut ops = CudaGradientOps::new(grad.device().clone())?;
            let norm = ops.compute_l2_norm(grad)?;
            total_norm_sq += norm * norm;
        }

        Ok(total_norm_sq.sqrt())
    }
}

/// Per-layer gradient clipping
pub struct LayerWiseGradientClipper {
    pub max_norm_per_layer: f32,
}

impl LayerWiseGradientClipper {
    pub fn new(max_norm_per_layer: f32) -> Self {
        Self { max_norm_per_layer }
    }
    
    /// Clip each gradient independently by its norm
    pub fn clip_grads(&self, grads: &mut [&mut Tensor]) -> Result<Vec<f32>> {
        let mut layer_norms = Vec::new();
        
        for grad in grads {
            let mut ops = CudaGradientOps::new(grad.device().clone())?;
            let norm = ops.compute_l2_norm(grad)?;
            layer_norms.push(norm);
            
            if norm > self.max_norm_per_layer {
                let scale = self.max_norm_per_layer / norm;
                **grad = grad.mul_scalar(scale)?;
            }
        }
        
        Ok(layer_norms)
    }
}

/// Gradient norm tracker for monitoring training stability
pub struct GradientNormTracker {
    pub history: Vec<f32>,
    pub window_size: usize,
}

impl GradientNormTracker {
    pub fn new(window_size: usize) -> Self {
        Self {
            history: Vec::new(),
            window_size,
        }
    }
    
    /// Record a gradient norm
    pub fn record(&mut self, norm: f32) {
        self.history.push(norm);
        
        // Keep only recent history
        if self.history.len() > self.window_size {
            self.history.remove(0);
        }
    }
    
    /// Get moving average of gradient norms
    pub fn moving_average(&self) -> f32 {
        if self.history.is_empty() {
            0.0
        } else {
            self.history.iter().sum::<f32>() / self.history.len() as f32
        }
    }
    
    /// Check if gradients are exploding
    pub fn is_exploding(&self, threshold: f32) -> bool {
        if let Some(&last_norm) = self.history.last() {
            last_norm > threshold
        } else {
            false
        }
    }
    
    /// Check if gradients are vanishing
    pub fn is_vanishing(&self, threshold: f32) -> bool {
        if let Some(&last_norm) = self.history.last() {
            last_norm < threshold
        } else {
            false
        }
    }
    
    /// Get gradient norm statistics
    pub fn stats(&self) -> GradientStats {
        if self.history.is_empty() {
            return GradientStats::default();
        }
        
        let mean = self.moving_average();
        let max = self.history.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min = self.history.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        
        let variance = self.history.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / self.history.len() as f32;
        let std_dev = variance.sqrt();
        
        GradientStats {
            mean,
            std_dev,
            max,
            min,
            current: *self.history.last().unwrap_or(&0.0),
        }
    }
}

/// Statistics about gradient norms
#[derive(Debug, Default)]
pub struct GradientStats {
    pub mean: f32,
    pub std_dev: f32,
    pub max: f32,
    pub min: f32,
    pub current: f32,
}

impl GradientStats {
    /// Check if gradients are healthy
    pub fn is_healthy(&self) -> bool {
        !self.current.is_nan() && 
        !self.current.is_infinite() &&
        self.current > 1e-7 &&  // Not vanishing
        self.current < 1e3      // Not exploding
    }
}

/// Utility functions for gradient analysis
pub mod utils {
    use super::*;
    
    /// Compute per-parameter gradient statistics
    pub fn compute_grad_stats(grads: &[&Tensor]) -> Result<Vec<(f32, f32, f32)>> {
        let mut stats = Vec::new();
        
        for grad in grads {
            let mean = mean_all_f32(grad)?;
            let centered = grad.add_scalar(-mean)?;
            let centered_sq = centered.square()?;
            let variance = mean_all_f32(&centered_sq)?;
            let norm_sq = grad.square()?.sum()?.item()?;

            stats.push((mean, variance.sqrt(), norm_sq.sqrt()));
        }
        
        Ok(stats)
    }
    
    /// Check for NaN or Inf in gradients
    pub fn check_grad_validity(grads: &[&Tensor]) -> Result<bool> {
        for grad in grads {
            if grad.shape().elem_count() == 0 {
                continue;
            }
            let mut ops = CudaGradientOps::new(grad.device().clone())?;
            if ops.has_invalid_values(grad)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
    
    /// Apply gradient value constraints
    pub fn constrain_grads(grads: &mut [&mut Tensor], max_value: f32) -> Result<()> {
        for grad in grads.iter_mut() {
            let grad_tensor: &mut Tensor = *grad;
            let mut ops = CudaGradientOps::new(grad_tensor.device().clone())?;
            ops.constrain_by_abs(grad_tensor, max_value)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Shape;
    use cudarc::driver::CudaDevice;
    
    #[test]
    fn test_gradient_clipping_by_norm() -> Result<()> {
        let device = CudaDevice::new(0)?;
        
        // Create gradients with large norm
        let mut grad1 = Tensor::from_vec(
            vec![10.0, 20.0, 30.0, 40.0],
            Shape::from_dims(&[2, 2]),
            device.clone()
        )?;
        
        let mut grad2 = Tensor::from_vec(
            vec![5.0, 10.0, 15.0, 20.0],
            Shape::from_dims(&[2, 2]),
            device
        )?;
        
        let clipper = GradientClipper::clip_by_norm(10.0);
        let mut grads = vec![&mut grad1, &mut grad2];
        
        let original_norm = clipper.compute_grad_norm(&grads)?;
        let clipped_norm = clipper.clip_grads(&mut grads)?;
        
        // Original norm should be large
        assert!(original_norm > 10.0);
        
        // After clipping, recompute norm
        let new_norm = clipper.compute_grad_norm(&grads)?;
        assert!((new_norm - 10.0).abs() < 1e-5);
        
        Ok(())
    }
    
    #[test]
    fn test_gradient_clipping_by_value() -> Result<()> {
        let device = CudaDevice::new(0)?;
        
        let mut grad = Tensor::from_vec(
            vec![-5.0, -2.0, 0.0, 2.0, 5.0],
            Shape::from_dims(&[5]),
            device
        )?;
        
        let clipper = GradientClipper::clip_by_value(-3.0, 3.0);
        let mut grads = vec![&mut grad];
        clipper.clip_grads(&mut grads)?;

        let within_bounds = grad.clamp(-3.0, 3.0)?;
        assert!(grad.equal(&within_bounds)?);
        
        Ok(())
    }
    
    #[test]
    fn test_gradient_norm_tracker() -> Result<()> {
        let mut tracker = GradientNormTracker::new(5);
        
        // Record some gradient norms
        tracker.record(1.0);
        tracker.record(2.0);
        tracker.record(3.0);
        tracker.record(100.0);  // Spike
        
        assert!(tracker.is_exploding(50.0));
        assert!(!tracker.is_vanishing(0.1));
        
        let stats = tracker.stats();
        assert_eq!(stats.max, 100.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.current, 100.0);
        
        Ok(())
    }
}
