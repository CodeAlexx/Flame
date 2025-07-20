//! Automatic differentiation engine for FLAME
//! 
//! This module provides a clean, integrated autograd system that works
//! seamlessly with the Tensor API.

use crate::{Tensor, Result, FlameError, Shape};
use crate::tensor::TensorId;
use crate::gradient::GradientMap;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use cudarc::driver::CudaDevice;

lazy_static::lazy_static! {
    /// Global autograd context - thread-safe
    static ref AUTOGRAD_CONTEXT: Mutex<AutogradContextInner> = Mutex::new(AutogradContextInner::new());
}

/// Operation types for autograd
#[derive(Debug, Clone)]
pub enum Op {
    Add { lhs: TensorId, rhs: TensorId },
    Sub { lhs: TensorId, rhs: TensorId },
    Mul { lhs: TensorId, rhs: TensorId },
    Div { lhs: TensorId, rhs: TensorId },
    MulScalar { input: TensorId, scalar: f32 },
    AddScalar { input: TensorId, scalar: f32 },
    MatMul { lhs: TensorId, rhs: TensorId },
    ReLU { input: TensorId },
    GELU { input: TensorId },
    SiLU { input: TensorId },
    Tanh { input: TensorId },
    Sigmoid { input: TensorId },
    Square { input: TensorId },
    Sum { input: TensorId, input_shape: Shape },
    Mean { input: TensorId, input_shape: Shape },
    Transpose { input: TensorId },
    Conv2d { input: TensorId, weight: TensorId, stride: usize, padding: usize },
    Linear { input: TensorId, weight: TensorId, bias: Option<TensorId> },
    LayerNorm { input: TensorId, normalized_shape: Vec<usize> },
    BatchMatMul { lhs: TensorId, rhs: TensorId },
    Reshape { input: TensorId, new_shape: Vec<usize> },
    Permute { input: TensorId, dims: Vec<usize> },
    AddBias { input: TensorId, bias: TensorId },
    SumDim { input: TensorId, dim: usize },
    SumDimKeepdim { input: TensorId, dim: usize },
    MaxDim { input: TensorId, dim: usize, keepdim: bool },
    Clamp { input: TensorId, min: f32, max: f32 },
}

/// Entry in the computation tape
struct TapeEntry {
    /// Output tensor ID
    output_id: TensorId,
    
    /// Operation that produced the output
    op: Op,
    
    /// Saved tensors needed for backward pass
    saved_tensors: HashMap<TensorId, Tensor>,
}

/// Internal autograd context
struct AutogradContextInner {
    /// Computation tape
    tape: Vec<TapeEntry>,
    
    /// Whether we're currently recording operations
    enabled: bool,
}

impl AutogradContextInner {
    fn new() -> Self {
        Self {
            tape: Vec::new(),
            enabled: true,
        }
    }
    
    fn record(&mut self, entry: TapeEntry) {
        if self.enabled {
            self.tape.push(entry);
        }
    }
    
    fn clear(&mut self) {
        self.tape.clear();
    }
}

/// Public API for autograd
pub struct AutogradContext;

impl AutogradContext {
    /// Record an operation in the computation graph
    pub fn record_op(
        output_id: TensorId,
        op: Op,
        saved_tensors: Vec<(TensorId, Tensor)>,
    ) {
        let mut ctx = AUTOGRAD_CONTEXT.lock().unwrap();
        
        let mut saved = HashMap::new();
        for (id, tensor) in saved_tensors {
            saved.insert(id, tensor);
        }
        
        ctx.record(TapeEntry {
            output_id,
            op,
            saved_tensors: saved,
        });
    }
    
    /// Clear the computation graph
    pub fn clear() {
        let mut ctx = AUTOGRAD_CONTEXT.lock().unwrap();
        ctx.clear();
    }
    
    /// Disable autograd (e.g., for inference)
    pub fn set_enabled(enabled: bool) {
        let mut ctx = AUTOGRAD_CONTEXT.lock().unwrap();
        ctx.enabled = enabled;
    }
    
    /// Context manager for no_grad mode
    pub fn no_grad() -> NoGradGuard {
        NoGradGuard::new()
    }
    
    /// Compute gradients via backpropagation
    pub fn backward(loss: &Tensor) -> Result<GradientMap> {
        if !loss.requires_grad {
            return Err(FlameError::InvalidOperation(
                "backward() called on tensor that doesn't require grad".into()
            ));
        }
        
        if loss.shape.elem_count() != 1 {
            return Err(FlameError::InvalidOperation(
                "backward() requires scalar loss tensor".into()
            ));
        }
        
        let device = loss.device.clone();
        
        // Initialize gradient storage
        let mut gradients = GradientMap::new(device.clone());
        gradients.set_ones(loss.id, loss.shape.clone())?;
        
        // Process tape in reverse under lock
        {
            let mut ctx = AUTOGRAD_CONTEXT.lock().unwrap();
            
            // Disable autograd during backward pass
            let prev_enabled = ctx.enabled;
            ctx.enabled = false;
            
            // Process tape in reverse
            for entry in ctx.tape.iter().rev() {
                if let Some(output_grad) = gradients.get(entry.output_id) {
                    let output_grad = output_grad.clone()?;
                    // Compute input gradients
                    let input_grads = compute_gradients(&entry, &output_grad, &device)?;
                    
                    // Accumulate gradients
                    for (tensor_id, grad) in input_grads {
                        gradients.accumulate(tensor_id, grad)?;
                    }
                }
            }
            
            // Re-enable autograd
            ctx.enabled = prev_enabled;
            
            // Clear tape after backward pass
            ctx.tape.clear();
        }
        
        Ok(gradients)
    }
}

/// RAII guard for no_grad mode
pub struct NoGradGuard {
    prev_state: bool,
}

impl NoGradGuard {
    fn new() -> Self {
        let mut ctx = AUTOGRAD_CONTEXT.lock().unwrap();
        let prev = ctx.enabled;
        ctx.enabled = false;
        Self { prev_state: prev }
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        let mut ctx = AUTOGRAD_CONTEXT.lock().unwrap();
        ctx.enabled = self.prev_state;
    }
}

/// Compute gradients for a single operation
fn compute_gradients(
    entry: &TapeEntry, 
    output_grad: &Tensor,
    device: &Arc<CudaDevice>
) -> Result<Vec<(TensorId, Tensor)>> {
    
    match &entry.op {
        Op::Add { lhs, rhs } => {
            // Gradient flows unchanged to both inputs
            Ok(vec![
                (*lhs, output_grad.clone()?),
                (*rhs, output_grad.clone()?),
            ])
        }
        
        Op::Sub { lhs, rhs } => {
            // d/dx(x-y) = 1, d/dy(x-y) = -1
            let neg_grad = output_grad.mul_scalar(-1.0)?;
            Ok(vec![
                (*lhs, output_grad.clone()?),
                (*rhs, neg_grad),
            ])
        }
        
        Op::Mul { lhs, rhs } => {
            // d/dx(x*y) = y, d/dy(x*y) = x
            let lhs_tensor = &entry.saved_tensors[lhs];
            let rhs_tensor = &entry.saved_tensors[rhs];
            
            let grad_lhs = output_grad.mul(rhs_tensor)?;
            let grad_rhs = output_grad.mul(lhs_tensor)?;
            
            Ok(vec![
                (*lhs, grad_lhs),
                (*rhs, grad_rhs),
            ])
        }
        
        Op::MulScalar { input, scalar } => {
            // d/dx(s*x) = s
            let grad = output_grad.mul_scalar(*scalar)?;
            Ok(vec![(*input, grad)])
        }
        
        Op::AddScalar { input, scalar: _ } => {
            // d/dx(x+s) = 1
            Ok(vec![(*input, output_grad.clone()?)])
        }
        
        Op::MatMul { lhs, rhs } => {
            // d/dA(A @ B) = grad @ B^T
            // d/dB(A @ B) = A^T @ grad
            let lhs_tensor = &entry.saved_tensors[lhs];
            let rhs_tensor = &entry.saved_tensors[rhs];
            
            // Gradient w.r.t. lhs
            let rhs_t = rhs_tensor.transpose()?;
            let grad_lhs = output_grad.matmul(&rhs_t)?;
            
            // Gradient w.r.t. rhs
            let lhs_t = lhs_tensor.transpose()?;
            let grad_rhs = lhs_t.matmul(output_grad)?;
            
            Ok(vec![
                (*lhs, grad_lhs),
                (*rhs, grad_rhs),
            ])
        }
        
        Op::ReLU { input } => {
            // d/dx ReLU(x) = 1 if x > 0, else 0
            let input_tensor = &entry.saved_tensors[input];
            let grad = relu_backward(output_grad, input_tensor)?;
            Ok(vec![(*input, grad)])
        }
        
        Op::GELU { input } => {
            // Use the complete GELU backward implementation
            let input_tensor = &entry.saved_tensors[input];
            let grad = crate::autograd_ops_complete::gelu_backward(output_grad, input_tensor)?;
            Ok(vec![(*input, grad)])
        }
        
        Op::SiLU { input } => {
            // Use the complete SiLU backward implementation
            let input_tensor = &entry.saved_tensors[input];
            let grad = crate::autograd_ops_complete::silu_backward(output_grad, input_tensor)?;
            Ok(vec![(*input, grad)])
        }
        
        Op::Tanh { input } => {
            // Use the complete Tanh backward implementation
            let input_tensor = &entry.saved_tensors[input];
            let output_tensor = input_tensor.tanh()?;
            let grad = crate::autograd_ops_complete::tanh_backward(output_grad, &output_tensor)?;
            Ok(vec![(*input, grad)])
        }
        
        Op::Sigmoid { input } => {
            // Use the complete Sigmoid backward implementation
            let input_tensor = &entry.saved_tensors[input];
            let output_tensor = input_tensor.sigmoid()?;
            let grad = crate::autograd_ops_complete::sigmoid_backward(output_grad, &output_tensor)?;
            Ok(vec![(*input, grad)])
        }
        
        Op::Square { input } => {
            // d/dx(x^2) = 2x
            let input_tensor = &entry.saved_tensors[input];
            let two_x = input_tensor.mul_scalar(2.0)?;
            let grad = output_grad.mul(&two_x)?;
            Ok(vec![(*input, grad)])
        }
        
        Op::Sum { input, input_shape } => {
            // Gradient of sum: broadcast grad to input shape
            let expanded = broadcast_to(output_grad, input_shape)?;
            Ok(vec![(*input, expanded)])
        }
        
        Op::Mean { input, input_shape } => {
            // d/dx mean(x) = 1/n for each element
            let n = input_shape.elem_count() as f32;
            let grad_scaled = output_grad.mul_scalar(1.0 / n)?;
            let expanded = broadcast_to(&grad_scaled, input_shape)?;
            Ok(vec![(*input, expanded)])
        }
        
        Op::Transpose { input } => {
            // Gradient of transpose is transpose of gradient
            let grad = output_grad.transpose()?;
            Ok(vec![(*input, grad)])
        }
        
        Op::Sigmoid { input } => {
            // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
            let input_tensor = &entry.saved_tensors[input];
            let sigmoid_out = input_tensor.sigmoid()?;
            let one_minus_sigmoid = sigmoid_out.mul_scalar(-1.0)?.add_scalar(1.0)?;
            let local_grad = sigmoid_out.mul(&one_minus_sigmoid)?;
            let grad = output_grad.mul(&local_grad)?;
            Ok(vec![(*input, grad)])
        }
        
        Op::Conv2d { input, weight, stride, padding } => {
            // Use CUDA Conv2D backward
            let input_tensor = &entry.saved_tensors[input];
            let weight_tensor = &entry.saved_tensors[weight];
            
            let (grad_input, grad_weight, grad_bias) = crate::cuda_conv2d::CudaConv2d::conv2d_backward(
                output_grad,
                input_tensor,
                weight_tensor,
                (*stride, *stride),
                (*padding, *padding),
            )?;
            
            let mut grads = vec![
                (*input, grad_input),
                (*weight, grad_weight),
            ];
            
            // Handle bias gradient if present
            if let Some(grad_bias) = grad_bias {
                // Check if bias was saved in the tape entry
                // The bias would be the third saved tensor if it exists
                if entry.saved_tensors.len() > 2 {
                    // Get the bias tensor ID from the saved tensors
                    let bias_id = entry.saved_tensors.keys()
                        .find(|&&id| id != *input && id != *weight)
                        .copied();
                    
                    if let Some(bias_id) = bias_id {
                        grads.push((bias_id, grad_bias));
                    }
                }
            }
            
            Ok(grads)
        }
        
        Op::LayerNorm { input, normalized_shape } => {
            // Use the complete LayerNorm backward implementation
            let input_tensor = &entry.saved_tensors[input];
            
            // For now, assume no weight/bias (pure normalization)
            // TODO: Add support for affine LayerNorm
            let mean = input_tensor.mean_dims(&normalized_shape, true)?;
            let var = input_tensor.var_dims(&normalized_shape, true, true)?;
            let normalized = input_tensor.sub(&mean)?.div(&var.add_scalar(1e-5)?.sqrt()?)?;
            
            let (grad_input, _, _) = crate::autograd_ops_complete::layer_norm_backward(
                output_grad,
                input_tensor,
                &normalized,
                None,  // weight
                None,  // bias
                &mean,
                &var,
                normalized_shape,
                1e-5,  // eps
            )?;
            
            Ok(vec![(*input, grad_input)])
        }
        
        Op::Linear { input, weight, bias } => {
            // d/dx(Wx + b) = W^T @ grad
            // d/dW(Wx + b) = grad @ x^T
            // d/db(Wx + b) = grad
            let input_tensor = &entry.saved_tensors[input];
            let weight_tensor = &entry.saved_tensors[weight];
            
            // Gradient w.r.t. input: W^T @ grad
            let weight_t = weight_tensor.transpose()?;
            let grad_input = output_grad.matmul(&weight_t)?;
            
            // Gradient w.r.t. weight: grad @ input^T
            let input_t = input_tensor.transpose()?;
            let grad_weight = output_grad.transpose()?.matmul(&input_t)?.transpose()?;
            
            let mut grads = vec![
                (*input, grad_input),
                (*weight, grad_weight),
            ];
            
            // Gradient w.r.t. bias (if present)
            if let Some(bias_id) = bias {
                // Sum over all dimensions except the last (features)
                let grad_bias = output_grad.sum_dims(&(0..output_grad.shape().dims().len()-1).collect::<Vec<_>>())?;
                grads.push((*bias_id, grad_bias));
            }
            
            Ok(grads)
        }
        
        Op::BatchMatMul { lhs, rhs } => {
            // Similar to MatMul but preserves batch dimension
            let lhs_tensor = &entry.saved_tensors[lhs];
            let rhs_tensor = &entry.saved_tensors[rhs];
            
            // Gradient w.r.t. lhs: grad @ rhs^T (batched)
            let rhs_t = rhs_tensor.transpose_batch()?;
            let grad_lhs = output_grad.batch_matmul(&rhs_t)?;
            
            // Gradient w.r.t. rhs: lhs^T @ grad (batched)
            let lhs_t = lhs_tensor.transpose_batch()?;
            let grad_rhs = lhs_t.batch_matmul(output_grad)?;
            
            Ok(vec![
                (*lhs, grad_lhs),
                (*rhs, grad_rhs),
            ])
        }
        
        Op::Reshape { input, .. } => {
            // Gradient of reshape is reshape of gradient back to original shape
            let input_tensor = &entry.saved_tensors[input];
            let grad = output_grad.reshape(input_tensor.shape().dims())?;
            Ok(vec![(*input, grad)])
        }
        
        Op::Permute { input, dims } => {
            // Gradient of permute is inverse permute
            let inverse_dims = inverse_permutation(dims);
            let grad = output_grad.permute(&inverse_dims)?;
            Ok(vec![(*input, grad)])
        }
        
        Op::AddBias { input, bias } => {
            // d/dx(x + b) = grad
            // d/db(x + b) = sum(grad) over batch and spatial dims
            let grad_input = output_grad.clone()?;
            
            // Sum over all dimensions except the bias dimension (usually channels)
            let ndims = output_grad.shape().dims().len();
            let mut sum_dims = vec![0]; // batch dimension
            if ndims > 2 {
                // Add spatial dimensions
                sum_dims.extend(2..ndims);
            }
            let grad_bias = output_grad.sum_dims(&sum_dims)?;
            
            Ok(vec![
                (*input, grad_input),
                (*bias, grad_bias),
            ])
        }
        
        Op::SumDim { input, dim } => {
            // Gradient of sum is broadcast back to original shape
            let input_tensor = &entry.saved_tensors[input];
            let mut grad_shape = input_tensor.shape().dims().to_vec();
            grad_shape[*dim] = 1;
            let grad_reshaped = output_grad.reshape(&grad_shape)?;
            let grad = grad_reshaped.broadcast_to(input_tensor.shape())?;
            Ok(vec![(*input, grad)])
        }
        
        Op::Clamp { input, min, max } => {
            // Use the clamp backward implementation
            let input_tensor = &entry.saved_tensors[input];
            let grad = crate::autograd_ops_complete::clamp_backward(
                output_grad, 
                input_tensor, 
                *min, 
                *max
            )?;
            Ok(vec![(*input, grad)])
        }
        
        Op::Div { lhs, rhs } => {
            // d/dx (x/y) = 1/y
            // d/dy (x/y) = -x/y^2
            let lhs_tensor = &entry.saved_tensors[lhs];
            let rhs_tensor = &entry.saved_tensors[rhs];
            
            // Gradient w.r.t. lhs: grad * (1/rhs)
            let grad_lhs = output_grad.div(rhs_tensor)?;
            
            // Gradient w.r.t. rhs: grad * (-lhs/rhs^2)
            let rhs_squared = rhs_tensor.square()?;
            let neg_lhs = lhs_tensor.mul_scalar(-1.0)?;
            let grad_rhs_term = neg_lhs.div(&rhs_squared)?;
            let grad_rhs = output_grad.mul(&grad_rhs_term)?;
            
            Ok(vec![(*lhs, grad_lhs), (*rhs, grad_rhs)])
        }
        
        Op::MaxDim { input, dim, keepdim } => {
            // For max reduction, gradient flows only through the max elements
            let input_tensor = &entry.saved_tensors[input];
            
            // Get the max values and indices
            let max_vals = input_tensor.max_dim(*dim, *keepdim)?;
            
            // Create a mask where input equals max (handling broadcasting)
            let max_broadcast = if *keepdim {
                max_vals.clone()?
            } else {
                max_vals.unsqueeze(*dim)?
            };
            
            // Create mask where input == max_broadcast
            let mask = input_tensor.eq(&max_broadcast)?;
            
            // Broadcast gradient if needed
            let grad_broadcast = if *keepdim {
                output_grad.clone()?
            } else {
                output_grad.unsqueeze(*dim)?
            };
            
            // Apply mask
            let grad = grad_broadcast.mul(&mask)?;
            
            Ok(vec![(*input, grad)])
        }
        
        Op::SumDimKeepdim { input, dim } => {
            // For sum with keepdim, gradient is broadcast back
            let input_tensor = &entry.saved_tensors[input];
            let input_shape = input_tensor.shape();
            
            // Broadcast gradient back to input shape
            let grad = output_grad.broadcast_to(input_shape)?;
            
            Ok(vec![(*input, grad)])
        }
        
        _ => {
            // This should not happen if all operations are implemented
            Err(FlameError::InvalidOperation(
                format!("Gradient not implemented for operation: {:?}", entry.op)
            ))
        }
    }
}

/// ReLU backward pass
fn relu_backward(grad_output: &Tensor, input: &Tensor) -> Result<Tensor> {
    // Create a mask where input > 0
    let zero = Tensor::zeros(input.shape.clone(), input.device.clone())?;
    let mask = input.gt(&zero)?;
    
    // Apply mask to gradient
    grad_output.mul(&mask)
}

/// Broadcast tensor to target shape
fn broadcast_to(tensor: &Tensor, target_shape: &Shape) -> Result<Tensor> {
    if tensor.shape == *target_shape {
        return Ok(tensor.clone()?);
    }
    
    // For now, handle simple case of scalar to tensor broadcast
    if tensor.shape.elem_count() == 1 {
        let val = tensor.to_vec()?[0];
        let expanded_data = vec![val; target_shape.elem_count()];
        return Tensor::from_vec(expanded_data, target_shape.clone(), tensor.device.clone());
    }
    
    // TODO: Implement general broadcasting
    Err(FlameError::InvalidOperation(
        format!("Broadcasting from {:?} to {:?} not yet implemented", 
                tensor.shape.dims(), target_shape.dims())
    ))
}

/// Helper function to compute inverse permutation
fn inverse_permutation(perm: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inverse[p] = i;
    }
    inverse
}

/// Comparison operations
impl Tensor {
    /// Element-wise greater than comparison
    pub fn gt(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape != other.shape {
            return Err(FlameError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }
        
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        
        let result: Vec<f32> = self_data.iter()
            .zip(other_data.iter())
            .map(|(a, b)| if a > b { 1.0 } else { 0.0 })
            .collect();
        
        Tensor::from_vec(result, self.shape.clone(), self.device.clone())
    }
}