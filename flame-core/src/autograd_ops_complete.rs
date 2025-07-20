//! Complete gradient implementations for all operations
//! This module provides the missing gradient computations with numerical stability

use crate::{Tensor, Result, FlameError, Shape, TensorId};
use crate::autograd::Op;
use std::sync::Arc;
use cudarc::driver::CudaDevice;

/// Compute gradients for LayerNorm operation
pub fn layer_norm_backward(
    grad_output: &Tensor,
    input: &Tensor,
    normalized: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    mean: &Tensor,
    var: &Tensor,
    normalized_shape: &[usize],
    eps: f32,
) -> Result<(Tensor, Option<Tensor>, Option<Tensor>)> {
    let device = input.device();
    
    // Get dimensions
    let shape = input.shape().dims();
    let ndim = shape.len();
    let axis = ndim - normalized_shape.len();
    
    // Calculate number of elements in normalized dimensions
    let norm_size: usize = normalized_shape.iter().product();
    let batch_size = input.shape().elem_count() / norm_size;
    
    // Compute gradients
    let grad_out_scaled = if let Some(w) = weight {
        grad_output.mul(w)?
    } else {
        grad_output.clone()?
    };
    
    // Compute grad w.r.t normalized input
    let std = var.add_scalar(eps)?.sqrt()?;
    let inv_std = std.reciprocal()?;
    
    // grad_normalized = grad_output * weight (if exists)
    let grad_normalized = &grad_out_scaled;
    
    // Compute intermediate values for input gradient
    // Sum over normalized dimensions
    let mut mean_grad_normalized = grad_normalized.clone()?;
    for i in 0..normalized_shape.len() {
        let dim = ndim - normalized_shape.len() + i;
        mean_grad_normalized = mean_grad_normalized.sum_keepdim(dim as isize)?;
    }
    mean_grad_normalized = mean_grad_normalized.div_scalar(norm_size as f32)?;
    
    let mut dot_p = normalized.mul(grad_normalized)?;
    for i in 0..normalized_shape.len() {
        let dim = ndim - normalized_shape.len() + i;
        dot_p = dot_p.sum_keepdim(dim as isize)?;
    }
    dot_p = dot_p.div_scalar(norm_size as f32)?;
    
    // Compute input gradient using the formula:
    // grad_input = (grad_normalized - mean(grad_normalized) - normalized * mean(normalized * grad_normalized)) / std
    let grad_input = grad_normalized
        .sub(&mean_grad_normalized)?
        .sub(&normalized.mul(&dot_p)?)?
        .mul(&inv_std)?;
    
    // Compute weight gradient if needed
    let grad_weight = if weight.is_some() {
        Some(
            grad_output.mul(&normalized)?
                .sum_dims(&(0..axis).collect::<Vec<_>>())?
        )
    } else {
        None
    };
    
    // Compute bias gradient if needed
    let grad_bias = if bias.is_some() {
        Some(
            grad_output.sum_dims(&(0..axis).collect::<Vec<_>>())?
        )
    } else {
        None
    };
    
    Ok((grad_input, grad_weight, grad_bias))
}

/// Compute gradients for BatchNorm operation
pub fn batch_norm_backward(
    grad_output: &Tensor,
    input: &Tensor,
    weight: Option<&Tensor>,
    running_mean: &Tensor,
    running_var: &Tensor,
    training: bool,
    eps: f32,
) -> Result<(Tensor, Option<Tensor>, Option<Tensor>)> {
    let device = input.device();
    let shape = input.shape().dims();
    
    if !training {
        // Inference mode: simpler gradients
        let std = running_var.add_scalar(eps)?.sqrt()?;
        let inv_std = std.reciprocal()?;
        
        let grad_input = if let Some(w) = weight {
            grad_output.mul(w)?.mul(&inv_std)?
        } else {
            grad_output.mul(&inv_std)?
        };
        
        let grad_weight = if weight.is_some() {
            let normalized = input.sub(running_mean)?.mul(&inv_std)?;
            Some(
                grad_output.mul(&normalized)?
                    .sum_dims(&[0, 2, 3])?  // Sum over batch and spatial dims
            )
        } else {
            None
        };
        
        let grad_bias = if weight.is_some() {
            Some(grad_output.sum_dims(&[0, 2, 3])?)
        } else {
            None
        };
        
        return Ok((grad_input, grad_weight, grad_bias));
    }
    
    // Training mode: full backward pass
    let batch_size = shape[0] as f32;
    let channels = shape[1];
    let spatial_size = shape[2..].iter().product::<usize>() as f32;
    let total_size = batch_size * spatial_size;
    
    // Compute saved statistics
    let mean = input.mean_dims(&[0, 2, 3], true)?;
    let var = input.var_dims(&[0, 2, 3], true, true)?;
    
    let std = var.add_scalar(eps)?.sqrt()?;
    let inv_std = std.reciprocal()?;
    let normalized = input.sub(&mean)?.mul(&inv_std)?;
    
    // Compute gradients
    let grad_out_scaled = if let Some(w) = weight {
        grad_output.mul(w)?
    } else {
        grad_output.clone()?
    };
    
    // Intermediate computations
    let mean_dy = grad_out_scaled.mean_dims(&[0, 2, 3], true)?;
    let mean_dy_xmu = grad_out_scaled.mul(&normalized)?
        .mean_dims(&[0, 2, 3], true)?;
    
    // Input gradient
    let grad_input = grad_out_scaled
        .sub(&mean_dy)?
        .sub(&normalized.mul(&mean_dy_xmu)?)?
        .mul(&inv_std)?
        .div_scalar(total_size)?;
    
    // Weight and bias gradients
    let grad_weight = if weight.is_some() {
        Some(
            grad_output.mul(&normalized)?
                .sum_dims(&[0, 2, 3])?
        )
    } else {
        None
    };
    
    let grad_bias = if weight.is_some() {
        Some(grad_output.sum_dims(&[0, 2, 3])?)
    } else {
        None
    };
    
    Ok((grad_input, grad_weight, grad_bias))
}

/// Compute gradients for Softmax operation
pub fn softmax_backward(
    grad_output: &Tensor,
    output: &Tensor,
    dim: isize,
) -> Result<Tensor> {
    // For softmax: grad_input = output * (grad_output - sum(grad_output * output))
    let sum_term = grad_output.mul(output)?
        .sum_keepdim(dim)?;
    
    let grad_input = output.mul(&grad_output.sub(&sum_term)?)?;
    Ok(grad_input)
}

/// Compute gradients for LogSoftmax operation
pub fn log_softmax_backward(
    grad_output: &Tensor,
    output: &Tensor,
    dim: isize,
) -> Result<Tensor> {
    // For log_softmax: grad_input = grad_output - softmax(output) * sum(grad_output)
    let softmax_output = output.exp()?;
    let sum_grad = grad_output.sum_keepdim(dim)?;
    let grad_input = grad_output.sub(&softmax_output.mul(&sum_grad)?)?;
    Ok(grad_input)
}

/// Compute gradients for Dropout operation
pub fn dropout_backward(
    grad_output: &Tensor,
    mask: &Tensor,
    dropout_prob: f32,
    training: bool,
) -> Result<Tensor> {
    if !training || dropout_prob == 0.0 {
        return Ok(grad_output.clone()?);
    }
    
    // Scale by keep probability and apply mask
    let scale = 1.0 / (1.0 - dropout_prob);
    grad_output.mul(mask)?.mul_scalar(scale)
}

/// Compute gradients for GELU operation with numerical stability
pub fn gelu_backward(
    grad_output: &Tensor,
    input: &Tensor,
) -> Result<Tensor> {
    // GELU(x) = 0.5 * x * (1 + erf(x/sqrt(2)))
    // GELU'(x) = 0.5 * (1 + erf(x/sqrt(2))) + x * pdf(x/sqrt(2)) / sqrt(2)
    
    let sqrt_2 = std::f32::consts::SQRT_2;
    let x_scaled = input.div_scalar(sqrt_2)?;
    
    // Compute erf term with clamping for stability
    let erf_term = x_scaled.erf()?;
    
    // Compute PDF term: exp(-0.5 * x^2) / sqrt(2*pi)
    let neg_half_x_sq = x_scaled.square()?.mul_scalar(-0.5)?;
    let pdf_term = neg_half_x_sq.exp()?.div_scalar((2.0 * std::f32::consts::PI).sqrt())?;
    
    // Compute derivative with numerical stability
    let deriv_term1 = erf_term.add_scalar(1.0)?.mul_scalar(0.5)?;
    let deriv_term2 = input.mul(&pdf_term)?.div_scalar(sqrt_2)?;
    let derivative = deriv_term1.add(&deriv_term2)?
        .clamp(1e-7, 1e7)?;  // Prevent numerical issues
    
    grad_output.mul(&derivative)
}

/// Compute gradients for SiLU (Swish) operation
pub fn silu_backward(
    grad_output: &Tensor,
    input: &Tensor,
) -> Result<Tensor> {
    // SiLU(x) = x * sigmoid(x)
    // SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    
    let sigmoid = input.sigmoid()?;
    let one_minus_sigmoid = sigmoid.neg()?.add_scalar(1.0)?;
    
    // Compute derivative with stability
    let term1 = sigmoid.clone()?;
    let term2 = input.mul(&sigmoid)?.mul(&one_minus_sigmoid)?;
    let derivative = term1.add(&term2)?
        .clamp(1e-7, 1e7)?;  // Prevent numerical issues
    
    grad_output.mul(&derivative)
}

/// Compute gradients for Tanh operation
pub fn tanh_backward(
    grad_output: &Tensor,
    output: &Tensor,
) -> Result<Tensor> {
    // tanh'(x) = 1 - tanh(x)^2
    let one_minus_tanh_sq = output.square()?.neg()?.add_scalar(1.0)?
        .clamp(1e-7, 1.0)?;  // Prevent vanishing gradients
    grad_output.mul(&one_minus_tanh_sq)
}

/// Compute gradients for Sigmoid operation with stability
pub fn sigmoid_backward(
    grad_output: &Tensor,
    output: &Tensor,
) -> Result<Tensor> {
    // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    let one_minus_sigmoid = output.neg()?.add_scalar(1.0)?;
    let derivative = output.mul(&one_minus_sigmoid)?
        .clamp(1e-7, 0.25)?;  // Max derivative of sigmoid is 0.25
    grad_output.mul(&derivative)
}

/// Compute gradients for general broadcasting operation
pub fn broadcast_backward(
    grad_output: &Tensor,
    original_shape: &[usize],
) -> Result<Tensor> {
    let output_shape = grad_output.shape().dims();
    
    // Handle dimension differences
    let ndim_diff = output_shape.len() - original_shape.len();
    
    // Sum over broadcasted dimensions
    let mut grad = grad_output.clone()?;
    
    // First, sum over extra leading dimensions
    for _ in 0..ndim_diff {
        grad = grad.sum_dim(0)?;
    }
    
    // Then, sum over dimensions that were size 1
    for (i, &orig_size) in original_shape.iter().enumerate() {
        if orig_size == 1 && grad.shape().dims()[i] > 1 {
            grad = grad.sum_keepdim(i as isize)?;
        }
    }
    
    Ok(grad)
}

/// Helper to compute gradients with proper shape handling
pub fn reduce_grad_to_shape(
    grad: &Tensor,
    target_shape: &[usize],
) -> Result<Tensor> {
    let grad_shape = grad.shape().dims();
    
    if grad_shape == target_shape {
        return Ok(grad.clone()?);
    }
    
    // Determine which dimensions to reduce
    let mut result = grad.clone()?;
    
    // Handle different number of dimensions
    let offset = grad_shape.len().saturating_sub(target_shape.len());
    
    // Sum over extra dimensions
    for i in 0..offset {
        result = result.sum_dim(0)?;
    }
    
    // Sum over broadcasted dimensions
    for (i, &target_dim) in target_shape.iter().enumerate() {
        if target_dim == 1 && result.shape().dims()[i] > 1 {
            result = result.sum_keepdim(i as isize)?;
        }
    }
    
    Ok(result)
}

/// Extension methods for Tensor to support missing operations
impl Tensor {
    /// Compute variance along dimensions
    pub fn var_dims(&self, dims: &[usize], keepdim: bool, unbiased: bool) -> Result<Tensor> {
        let mean = self.mean_dims(dims, keepdim)?;
        let centered = self.sub(&mean)?;
        let squared = centered.square()?;
        let sum_sq = squared.sum_dims(dims)?;
        
        // Calculate divisor
        let n: usize = dims.iter().map(|&d| self.shape().dims()[d]).product();
        let divisor = if unbiased && n > 1 {
            (n - 1) as f32
        } else {
            n as f32
        };
        
        let var = sum_sq.div_scalar(divisor)?;
        
        if keepdim {
            Ok(var)
        } else {
            // Squeeze dimensions
            let mut new_shape = vec![];
            for (i, &s) in self.shape().dims().iter().enumerate() {
                if !dims.contains(&i) {
                    new_shape.push(s);
                }
            }
            var.reshape(&new_shape)
        }
    }
    
    /// Compute mean along dimensions
    pub fn mean_dims(&self, dims: &[usize], keepdim: bool) -> Result<Tensor> {
        let sum = self.sum_dims(dims)?;
        let n: usize = dims.iter().map(|&d| self.shape().dims()[d]).product();
        let mean = sum.div_scalar(n as f32)?;
        
        if keepdim {
            // Reshape to keep dimensions as 1
            let mut new_shape = self.shape().dims().to_vec();
            for &d in dims {
                new_shape[d] = 1;
            }
            mean.reshape(&new_shape)
        } else {
            Ok(mean)
        }
    }
    
    
    /// Sum along dimension keeping the dimension
    pub fn sum_keepdim(&self, dim: isize) -> Result<Tensor> {
        let ndim = self.shape().dims().len() as isize;
        let dim = if dim < 0 { ndim + dim } else { dim } as usize;
        
        let sum = self.sum_dim(dim)?;
        
        // Reshape to keep dimension as 1
        let mut new_shape = self.shape().dims().to_vec();
        new_shape[dim] = 1;
        sum.reshape(&new_shape)
    }
    
    /// Compute reciprocal (1/x)
    pub fn reciprocal(&self) -> Result<Tensor> {
        let ones = Tensor::ones(self.shape().clone(), self.device.clone())?;
        ones.div(self)
    }
    
    /// Clamp tensor values with gradient preservation (internal implementation)
    pub fn clamp_with_grad(&self, min: f32, max: f32) -> Result<Tensor> {
        let data = self.to_vec()?;
        let clamped: Vec<f32> = data.iter()
            .map(|&x| x.clamp(min, max))
            .collect();
        
        let result = Tensor::from_slice(&clamped, self.shape.clone(), self.device.clone())?;
        
        // Preserve gradient tracking
        if self.requires_grad {
            // Record clamp operation for autograd
            let saved_tensors = vec![(self.id, self.clone()?)];
            crate::autograd::AutogradContext::record_op(
                result.id,
                Op::Clamp { input: self.id, min, max },
                saved_tensors,
            );
        }
        
        Ok(result)
    }
}

/// Clamp operation for autograd
#[derive(Debug, Clone)]
pub struct ClampOp {
    pub input: TensorId,
    pub min: f32,
    pub max: f32,
}

/// Compute gradient for clamp operation
pub fn clamp_backward(
    grad_output: &Tensor,
    input: &Tensor,
    min: f32,
    max: f32,
) -> Result<Tensor> {
    // Gradient passes through only where min < input < max
    let mask_data = input.to_vec()?;
    let mask: Vec<f32> = mask_data.iter()
        .map(|&x| if x > min && x < max { 1.0 } else { 0.0 })
        .collect();
    
    let mask_tensor = Tensor::from_slice(&mask, input.shape().clone(), input.device.clone())?;
    grad_output.mul(&mask_tensor)
}