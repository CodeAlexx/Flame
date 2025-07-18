use crate::{Tensor, Shape, Result, FlameError};
use std::sync::Arc;
use cudarc::driver::CudaDevice;
use rand::Rng;

/// Dropout layer - randomly zeros elements during training
pub struct Dropout {
    pub p: f32,  // probability of dropping an element
    pub training: bool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        assert!(p >= 0.0 && p <= 1.0, "Dropout probability must be in [0, 1]");
        Self { p, training: true }
    }
    
    /// Set training mode
    pub fn train(&mut self, mode: bool) {
        self.training = mode;
    }
    
    /// Apply dropout to input tensor
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.training || self.p == 0.0 {
            // During evaluation or with p=0, return input unchanged
            return input.clone();
        }
        
        if self.p == 1.0 {
            // Drop all elements
            return Tensor::zeros(input.shape().clone(), input.device().clone());
        }
        
        // Generate dropout mask
        let mask = self.generate_mask(input.shape(), input.device())?;
        
        // Apply mask and scale by 1/(1-p) to maintain expected value
        let scale = 1.0 / (1.0 - self.p);
        let output = input.mul(&mask)?;
        output.mul_scalar(scale)
    }
    
    /// Generate binary dropout mask
    fn generate_mask(&self, shape: &Shape, device: &Arc<CudaDevice>) -> Result<Tensor> {
        let mut rng = rand::thread_rng();
        let keep_prob = 1.0 - self.p;
        
        let mask_data: Vec<f32> = (0..shape.elem_count())
            .map(|_| if rng.gen::<f32>() < keep_prob { 1.0 } else { 0.0 })
            .collect();
        
        Tensor::from_vec(mask_data, shape.clone(), device.clone())
    }
}

/// Dropout2d - channel-wise dropout for Conv2d layers
pub struct Dropout2d {
    pub p: f32,
    pub training: bool,
}

impl Dropout2d {
    pub fn new(p: f32) -> Self {
        assert!(p >= 0.0 && p <= 1.0, "Dropout probability must be in [0, 1]");
        Self { p, training: true }
    }
    
    /// Set training mode
    pub fn train(&mut self, mode: bool) {
        self.training = mode;
    }
    
    /// Apply channel-wise dropout
    /// Input shape: [batch, channels, height, width]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.training || self.p == 0.0 {
            return input.clone();
        }
        
        let shape = input.shape().dims();
        if shape.len() != 4 {
            return Err(FlameError::InvalidOperation(
                format!("Dropout2d expects 4D input, got {:?}", shape)
            ));
        }
        
        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
        
        // Generate mask for each channel
        let mut rng = rand::thread_rng();
        let keep_prob = 1.0 - self.p;
        let scale = 1.0 / keep_prob;
        
        let mut output_data = vec![0.0f32; input.shape().elem_count()];
        let input_data = input.to_vec()?;
        
        for b in 0..batch {
            for c in 0..channels {
                let keep = rng.gen::<f32>() < keep_prob;
                let mask_value = if keep { scale } else { 0.0 };
                
                // Apply mask to entire channel
                for h in 0..height {
                    for w in 0..width {
                        let idx = b * channels * height * width + 
                                 c * height * width + 
                                 h * width + 
                                 w;
                        output_data[idx] = input_data[idx] * mask_value;
                    }
                }
            }
        }
        
        Tensor::from_vec(output_data, input.shape().clone(), input.device().clone())
    }
}

/// L2 regularization (weight decay)
pub struct L2Regularization {
    pub lambda: f32,  // regularization strength
}

impl L2Regularization {
    pub fn new(lambda: f32) -> Self {
        Self { lambda }
    }
    
    /// Compute L2 penalty: 0.5 * lambda * sum(w^2)
    pub fn penalty(&self, weights: &[&Tensor]) -> Result<Tensor> {
        if weights.is_empty() {
            return Err(FlameError::InvalidOperation("No weights provided".into()));
        }
        
        let device = weights[0].device().clone();
        let mut total_penalty = Tensor::zeros(Shape::from_dims(&[1]), device.clone())?;
        
        for weight in weights {
            let squared = weight.square()?;
            let sum = squared.sum()?;
            total_penalty = total_penalty.add(&sum)?;
        }
        
        total_penalty.mul_scalar(0.5 * self.lambda)
    }
    
    /// Apply L2 regularization to gradients (add lambda * w to gradients)
    pub fn apply_to_gradients(&self, weight: &Tensor, grad: &Tensor) -> Result<Tensor> {
        let penalty_grad = weight.mul_scalar(self.lambda)?;
        grad.add(&penalty_grad)
    }
}

/// L1 regularization (sparsity inducing)
pub struct L1Regularization {
    pub lambda: f32,
}

impl L1Regularization {
    pub fn new(lambda: f32) -> Self {
        Self { lambda }
    }
    
    /// Compute L1 penalty: lambda * sum(|w|)
    pub fn penalty(&self, weights: &[&Tensor]) -> Result<Tensor> {
        if weights.is_empty() {
            return Err(FlameError::InvalidOperation("No weights provided".into()));
        }
        
        let device = weights[0].device().clone();
        let mut total_penalty = Tensor::zeros(Shape::from_dims(&[1]), device.clone())?;
        
        for weight in weights {
            let abs_weight = self.abs(weight)?;
            let sum = abs_weight.sum()?;
            total_penalty = total_penalty.add(&sum)?;
        }
        
        total_penalty.mul_scalar(self.lambda)
    }
    
    /// Apply L1 regularization to gradients (add lambda * sign(w) to gradients)
    pub fn apply_to_gradients(&self, weight: &Tensor, grad: &Tensor) -> Result<Tensor> {
        let sign = self.sign(weight)?;
        let penalty_grad = sign.mul_scalar(self.lambda)?;
        grad.add(&penalty_grad)
    }
    
    /// Compute element-wise absolute value
    fn abs(&self, tensor: &Tensor) -> Result<Tensor> {
        let data = tensor.to_vec()?;
        let abs_data: Vec<f32> = data.iter().map(|&x| x.abs()).collect();
        Tensor::from_vec(abs_data, tensor.shape().clone(), tensor.device().clone())
    }
    
    /// Compute element-wise sign
    fn sign(&self, tensor: &Tensor) -> Result<Tensor> {
        let data = tensor.to_vec()?;
        let sign_data: Vec<f32> = data.iter().map(|&x| {
            if x > 0.0 { 1.0 }
            else if x < 0.0 { -1.0 }
            else { 0.0 }
        }).collect();
        Tensor::from_vec(sign_data, tensor.shape().clone(), tensor.device().clone())
    }
}

/// Alpha dropout - maintains mean and variance (for use with SELU activation)
pub struct AlphaDropout {
    pub p: f32,
    pub training: bool,
    alpha: f32,
    scale: f32,
    bias: f32,
}

impl AlphaDropout {
    pub fn new(p: f32) -> Self {
        // Constants for SELU
        let alpha = 1.6732632423543772848170429916717;
        let _scale = 1.0507009873554804934193349852946;
        
        // Compute affine transformation parameters
        let q = 1.0 - p;
        let a = ((q + alpha * alpha * p * q).sqrt() - q) / (alpha * p);
        let b = -a * alpha * p;
        
        Self {
            p,
            training: true,
            alpha,
            scale: a,
            bias: b,
        }
    }
    
    /// Set training mode
    pub fn train(&mut self, mode: bool) {
        self.training = mode;
    }
    
    /// Apply alpha dropout
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.training || self.p == 0.0 {
            return input.clone();
        }
        
        let mut rng = rand::thread_rng();
        let keep_prob = 1.0 - self.p;
        
        let data = input.to_vec()?;
        let output_data: Vec<f32> = data.iter().map(|&x| {
            if rng.gen::<f32>() < keep_prob {
                x * self.scale + self.bias
            } else {
                self.alpha * self.scale + self.bias
            }
        }).collect();
        
        Tensor::from_vec(output_data, input.shape().clone(), input.device().clone())
    }
}

/// Spectral normalization for weight matrices
pub struct SpectralNorm {
    pub n_power_iterations: usize,
    pub eps: f32,
    pub u: Option<Tensor>,  // Left singular vector
}

impl SpectralNorm {
    pub fn new(n_power_iterations: usize) -> Self {
        Self {
            n_power_iterations,
            eps: 1e-12,
            u: None,
        }
    }
    
    /// Apply spectral normalization to weight matrix
    /// Returns normalized weight
    pub fn forward(&mut self, weight: &Tensor) -> Result<Tensor> {
        let shape = weight.shape().dims();
        if shape.len() < 2 {
            return Err(FlameError::InvalidOperation(
                "SpectralNorm requires at least 2D weight".into()
            ));
        }
        
        // Reshape to 2D for SVD
        let (h, w) = if shape.len() == 2 {
            (shape[0], shape[1])
        } else {
            // Flatten all but last dimension
            let h: usize = shape[..shape.len()-1].iter().product();
            let w = shape[shape.len()-1];
            (h, w)
        };
        
        let weight_2d = weight.reshape(&[h, w])?;
        
        // Initialize u if needed
        if self.u.is_none() {
            let u = Tensor::randn(
                Shape::from_dims(&[h, 1]),
                0.0,
                1.0,
                weight.device().clone()
            )?;
            self.u = Some(u);
        }
        
        // Power iteration to find largest singular value
        for _ in 0..self.n_power_iterations {
            let u = self.u.as_ref().unwrap();
            // v = W^T @ u
            let v = weight_2d.transpose()?.matmul(u)?;
            let v_norm = self.vector_norm(&v)?;
            let v = v.mul_scalar(1.0 / (v_norm + self.eps))?;
            
            // u = W @ v
            let new_u = weight_2d.matmul(&v)?;
            let u_norm = self.vector_norm(&new_u)?;
            self.u = Some(new_u.mul_scalar(1.0 / (u_norm + self.eps))?);
        }
        
        // Compute spectral norm: sigma = u^T @ W @ v
        let u = self.u.as_ref().unwrap();
        let v = weight_2d.transpose()?.matmul(u)?;
        let v_norm = self.vector_norm(&v)?;
        let v = v.mul_scalar(1.0 / (v_norm + self.eps))?;
        
        let sigma = u.transpose()?.matmul(&weight_2d)?.matmul(&v)?;
        let sigma_val = sigma.item()?;
        
        // Normalize weight by spectral norm
        let normalized = weight.mul_scalar(1.0 / (sigma_val + self.eps))?;
        
        // Reshape back to original shape if needed
        if shape.len() > 2 {
            normalized.reshape(shape)
        } else {
            Ok(normalized)
        }
    }
    
    /// Compute L2 norm of a vector
    fn vector_norm(&self, v: &Tensor) -> Result<f32> {
        let squared = v.square()?;
        let sum = squared.sum()?;
        let norm = sum.item()?.sqrt();
        Ok(norm)
    }
}

/// Batch-wise weight standardization
pub struct WeightStandardization {
    pub eps: f32,
}

impl WeightStandardization {
    pub fn new() -> Self {
        Self { eps: 1e-5 }
    }
    
    /// Standardize weight tensor
    /// For Conv2d: [out_channels, in_channels, height, width]
    /// Standardize over in_channels, height, width dimensions
    pub fn forward(&self, weight: &Tensor) -> Result<Tensor> {
        let shape = weight.shape().dims();
        if shape.len() < 2 {
            return weight.clone();
        }
        
        let out_features = shape[0];
        let other_dims: usize = shape[1..].iter().product();
        
        let weight_2d = weight.reshape(&[out_features, other_dims])?;
        let data = weight_2d.to_vec()?;
        
        let mut standardized = vec![0.0f32; data.len()];
        
        // Standardize each output feature
        for i in 0..out_features {
            let start = i * other_dims;
            let end = start + other_dims;
            let feature_data = &data[start..end];
            
            // Compute mean
            let mean: f32 = feature_data.iter().sum::<f32>() / other_dims as f32;
            
            // Compute variance
            let variance: f32 = feature_data.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / other_dims as f32;
            
            let std = (variance + self.eps).sqrt();
            
            // Standardize
            for j in 0..other_dims {
                standardized[start + j] = (feature_data[j] - mean) / std;
            }
        }
        
        let standardized_2d = Tensor::from_vec(
            standardized,
            Shape::from_dims(&[out_features, other_dims]),
            weight.device().clone()
        )?;
        
        // Reshape back to original shape
        standardized_2d.reshape(shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dropout() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let input = Tensor::randn(Shape::from_dims(&[2, 4]), 0.0, 1.0, device)?;
        
        let mut dropout = Dropout::new(0.5);
        
        // Training mode - should apply dropout
        dropout.train(true);
        let output_train = dropout.forward(&input)?;
        
        // Eval mode - should return input unchanged
        dropout.train(false);
        let output_eval = dropout.forward(&input)?;
        
        // In eval mode, output should equal input
        let input_data = input.to_vec()?;
        let eval_data = output_eval.to_vec()?;
        for (a, b) in input_data.iter().zip(eval_data.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_l2_regularization() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let weight1 = Tensor::randn(Shape::from_dims(&[3, 4]), 0.0, 1.0, device.clone())?;
        let weight2 = Tensor::randn(Shape::from_dims(&[4, 5]), 0.0, 1.0, device)?;
        
        let l2_reg = L2Regularization::new(0.01);
        let penalty = l2_reg.penalty(&[&weight1, &weight2])?;
        
        // Penalty should be positive
        assert!(penalty.item()? > 0.0);
        
        Ok(())
    }
}