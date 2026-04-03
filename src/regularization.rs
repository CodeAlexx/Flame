use crate::rng;
use crate::tensor::contracts::assert_nhwc_bf16_public;
use crate::{DType, Error, Result, Shape, Tensor};
use cudarc::driver::CudaDevice;
use std::sync::Arc;

/// Dropout layer - randomly zeros elements during training
pub struct Dropout {
    pub p: f32, // probability of dropping an element
    pub training: bool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "Dropout probability must be in [0, 1]"
        );
        Self { p, training: true }
    }

    /// Set training mode
    pub fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    /// Apply dropout to input tensor
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        if self.p == 1.0 {
            return Tensor::zeros(input.shape().clone(), input.device().clone());
        }

        let mut mask = self.generate_mask(input.shape(), input.device(), input.dtype())?;
        if mask.dtype() != input.dtype() {
            mask = mask.to_dtype(input.dtype())?;
        }

        let scale = 1.0 / (1.0 - self.p);
        let output = input.mul(&mask)?;
        output.mul_scalar(scale)
    }

    /// Generate binary dropout mask
    fn generate_mask(
        &self,
        shape: &Shape,
        device: &Arc<CudaDevice>,
        target_dtype: DType,
    ) -> Result<Tensor> {
        let mut mask = rng::rand_on(device, shape.dims(), DType::F32, 0)?;
        let threshold = mask.full_like(self.p)?;
        mask = mask.ge(&threshold)?;
        if target_dtype != DType::F32 {
            mask = mask.to_dtype(target_dtype)?;
        }
        Ok(mask)
    }
}

/// Dropout2d - channel-wise dropout for Conv2d layers
pub struct Dropout2d {
    pub p: f32,
    pub training: bool,
}

impl Dropout2d {
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "Dropout probability must be in [0, 1]"
        );
        Self { p, training: true }
    }

    /// Set training mode
    pub fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    /// Apply channel-wise dropout
    /// Input shape: [batch, channels, height, width]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        assert_nhwc_bf16_public("Dropout2d::forward in", input)?;

        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        let shape = input.shape().dims();
        if shape.len() != 4 {
            return Err(Error::InvalidOperation(format!(
                "Dropout2d expects 4D input, got {:?}",
                shape
            )));
        }

        if self.p == 1.0 {
            return Tensor::zeros_dtype(input.shape().clone(), DType::BF16, input.device().clone());
        }

        let mut input_nchw = crate::cuda_ops::GpuOps::permute_nhwc_to_nchw(input)?;
        if input_nchw.dtype() != DType::F32 {
            input_nchw = input_nchw.to_dtype(DType::F32)?;
        }

        let dims = input_nchw.shape().dims();
        let (batch, channels, _height, _width) = (dims[0], dims[1], dims[2], dims[3]);

        let mask_shape = Shape::from_dims(&[batch, channels, 1, 1]);
        let mut mask = rng::rand_on(input.device(), mask_shape.dims(), DType::F32, 0)?;
        let threshold = mask.full_like(self.p)?;
        mask = mask.ge(&threshold)?;
        mask = mask.broadcast_to(input_nchw.shape())?;

        let scale = 1.0 / (1.0 - self.p);
        let output = input_nchw.mul(&mask)?.mul_scalar(scale)?;

        let mut output = crate::cuda_ops::GpuOps::permute_nchw_to_nhwc(&output)?;
        if output.dtype() != DType::BF16 {
            output = output.to_dtype(DType::BF16)?;
        }
        assert_nhwc_bf16_public("Dropout2d::forward out", &output)?;
        Ok(output)
    }
}

/// L2 regularization (weight decay)
pub struct L2Regularization {
    pub lambda: f32, // regularization strength
}

impl L2Regularization {
    pub fn new(lambda: f32) -> Self {
        Self { lambda }
    }

    /// Compute L2 penalty: 0.5 * lambda * sum(w^2)
    pub fn penalty(&self, weights: &[&Tensor]) -> Result<Tensor> {
        if weights.is_empty() {
            return Err(Error::InvalidOperation("No weights provided".into()));
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
            return Err(Error::InvalidOperation("No weights provided".into()));
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
        tensor.abs()
    }

    /// Compute element-wise sign
    fn sign(&self, tensor: &Tensor) -> Result<Tensor> {
        tensor.sign()
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
        let alpha = 1.673_263_2;
        let _scale = 1.050_700_987_355_480_5;

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
            return Ok(input.clone());
        }

        let mut mask = rng::rand_on(input.device(), input.shape().dims(), DType::F32, 0)?;
        let threshold = mask.full_like(self.p)?;
        mask = mask.ge(&threshold)?;
        if input.dtype() != DType::F32 {
            mask = mask.to_dtype(input.dtype())?;
        }

        let kept = input.mul_scalar(self.scale)?.add_scalar(self.bias)?;
        let dropped = input.full_like(self.alpha * self.scale + self.bias)?;

        let inv_mask = mask.clone_result()?.neg()?.add_scalar(1.0)?;
        let kept_part = kept.mul(&mask)?;
        let dropped_part = dropped.mul(&inv_mask)?;
        kept_part.add(&dropped_part)
    }
}

/// Spectral normalization for weight matrices
pub struct SpectralNorm {
    pub n_power_iterations: usize,
    pub eps: f32,
    pub u: Option<Tensor>, // Left singular vector
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
            return Err(Error::InvalidOperation(
                "SpectralNorm requires at least 2D weight".into(),
            ));
        }

        // Reshape to 2D for SVD
        let (h, w) = if shape.len() == 2 {
            (shape[0], shape[1])
        } else {
            // Flatten all but last dimension
            let h: usize = shape[..shape.len() - 1].iter().product();
            let w = shape[shape.len() - 1];
            (h, w)
        };

        let weight_2d = weight.reshape(&[h, w])?;

        // Initialize u if needed
        if self.u.is_none() {
            let u = Tensor::randn(Shape::from_dims(&[h, 1]), 0.0, 1.0, weight.device().clone())?;
            self.u = Some(u);
        }

        // Power iteration to find largest singular value
        for _ in 0..self.n_power_iterations {
            let u = match self.u.as_ref() {
                Some(u) => u,
                None => return Err(Error::Training("spectral norm state 'u' missing".into())),
            };
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
        let u = match self.u.as_ref() {
            Some(u) => u,
            None => return Err(Error::Training("spectral norm state 'u' missing".into())),
        };
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
            return Ok(weight.clone());
        }

        let out_features = shape[0];
        let other_dims: usize = shape[1..].iter().product();

        let weight_2d = weight.reshape(&[out_features, other_dims])?;
        let mean = weight_2d.mean_dim(&[1], true)?;
        let centered = weight_2d.sub(&mean)?;
        let variance = centered.square()?.mean_dim(&[1], true)?;
        let std = variance.add_scalar(self.eps)?.sqrt()?;
        let normalized = centered.div(&std)?;
        normalized.reshape(shape)
    }
}

impl Default for WeightStandardization {
    fn default() -> Self {
        Self::new()
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

        // Training mode - ensure we still produce correct shape and dtype
        dropout.train(true);
        let output_train = dropout.forward(&input)?;
        assert_eq!(output_train.shape(), input.shape());
        assert_eq!(output_train.dtype(), input.dtype());

        // Eval mode - should return input unchanged
        dropout.train(false);
        let output_eval = dropout.forward(&input)?;
        assert!(output_eval.equal(&input)?);

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
