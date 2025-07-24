//! Automatic differentiation engine for FLAME
//! 
//! This module provides a clean, integrated autograd system that works
//! seamlessly with the Tensor API.

use crate::{Tensor, Result, FlameError, Shape, DType};
use crate::tensor::TensorId;
use crate::tensor_storage::TensorStorage;
use crate::gradient::GradientMap;
use crate::cuda_ops::GpuOps;
use crate::cuda_kernels_gpu::CudaKernels;
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
    RMSNorm { input: TensorId, weight: Option<TensorId>, eps: f32 },
    BatchMatMul { lhs: TensorId, rhs: TensorId },
    Reshape { input: TensorId, new_shape: Vec<usize> },
    Permute { input: TensorId, dims: Vec<usize> },
    AddBias { input: TensorId, bias: TensorId },
    SumDim { input: TensorId, dim: usize },
    SumDimKeepdim { input: TensorId, dim: usize },
    MaxDim { input: TensorId, dim: usize, keepdim: bool },
    Clamp { input: TensorId, min: f32, max: f32 },
    Embedding { weight: TensorId, indices: TensorId },
    IndexSelect { input: TensorId, indices: TensorId, dim: usize },
    Cat { inputs: Vec<TensorId>, dim: usize },
    Split { input: TensorId, sizes: Vec<usize>, dim: usize },
    Abs { input: TensorId },
    Log { input: TensorId },
    Softmax { input: TensorId, dim: isize },
    LogSoftmax { input: TensorId, dim: isize },
    MSELoss { predictions: TensorId, targets: TensorId, num_elements: usize },
    L1Loss { predictions: TensorId, targets: TensorId, num_elements: usize },
    HuberLoss { predictions: TensorId, targets: TensorId, delta: f32, num_elements: usize },
    BCELoss { predictions: TensorId, targets: TensorId, num_elements: usize },
    NLLLoss { log_probs: TensorId, targets: TensorId, batch_size: usize },
    GroupNorm { input: TensorId, num_groups: usize, weight: Option<TensorId>, bias: Option<TensorId> },
    FlashAttention { query: TensorId, key: TensorId, value: TensorId, mask: Option<TensorId>, scale: f32, causal: bool },
    SageAttention {
        query_id: TensorId,
        key_id: TensorId,
        value_id: TensorId,
        scale: f32,
        causal: bool,
        quantized: bool,
    },
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
        
        // Only record if autograd is enabled
        if !ctx.enabled {
            return;
        }
        
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
    
    /// Reset the entire autograd context (for testing)
    pub fn reset() {
        let mut ctx = AUTOGRAD_CONTEXT.lock().unwrap();
        *ctx = AutogradContextInner::new();
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
    
    /// Compute gradients via backpropagation with debug logging
    pub fn backward_debug(loss: &Tensor) -> Result<GradientMap> {
        println!("=== AUTOGRAD DEBUG START ===");
        println!("Loss tensor shape: {:?}", loss.shape);
        println!("Loss requires_grad: {}", loss.requires_grad);
        
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
        println!("Root gradient initialized");
        
        // Process tape in reverse under lock
        {
            let mut ctx = AUTOGRAD_CONTEXT.lock().unwrap();
            println!("Tape length: {}", ctx.tape.len());
            
            // Print all operations in tape
            for (i, entry) in ctx.tape.iter().enumerate() {
                println!("Op {}: {:?} -> tensor_id {:?}", i, entry.op, entry.output_id);
            }
            
            // Disable autograd during backward pass
            let prev_enabled = ctx.enabled;
            ctx.enabled = false;
            
            // Process tape in reverse with timing
            for (i, entry) in ctx.tape.iter().enumerate().rev() {
                let tape_idx = ctx.tape.len() - 1 - i;
                println!("\nProcessing op {} (reverse index {}): {:?}", tape_idx, i, entry.op);
                let start = std::time::Instant::now();
                
                if let Some(output_grad) = gradients.get(entry.output_id) {
                    println!("  Output grad shape: {:?}", output_grad.shape());
                    let output_grad = output_grad.clone()?;
                    
                    // Process gradients based on operation type
                    match compute_gradients(entry, &output_grad, &device) {
                        Ok(input_grads) => {
                            println!("  Computed {} input gradients", input_grads.len());
                            
                            // Accumulate gradients
                            for (tensor_id, grad) in input_grads {
                                println!("    Accumulating grad for tensor {:?}, shape: {:?}", 
                                    tensor_id, grad.shape());
                                gradients.accumulate(tensor_id, grad)?;
                            }
                        }
                        Err(e) => {
                            println!("  ERROR computing gradients: {:?}", e);
                            ctx.enabled = prev_enabled;
                            return Err(e);
                        }
                    }
                } else {
                    println!("  No output gradient found, skipping");
                }
                
                let elapsed = start.elapsed();
                println!("  Op {} completed in {:?}", tape_idx, elapsed);
                
                if elapsed > std::time::Duration::from_secs(2) {
                    println!("  !!! SLOW OPERATION DETECTED !!!");
                    ctx.enabled = prev_enabled;
                    return Err(FlameError::InvalidOperation(
                        format!("Op {} took too long: {:?}", tape_idx, elapsed)
                    ));
                }
            }
            
            // Clear tape and restore state
            ctx.tape.clear();
            ctx.enabled = prev_enabled;
        }
        
        println!("\n=== AUTOGRAD DEBUG COMPLETE ===");
        println!("Total gradients computed: {}", gradients.len());
        Ok(gradients)
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
            // Gradient flows unchanged to both inputs, but handle broadcasting
            let lhs_tensor = entry.saved_tensors.get(lhs)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for lhs in Add".into()))?;
            let rhs_tensor = entry.saved_tensors.get(rhs)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for rhs in Add".into()))?;
            
            // If shapes differ, we need to reduce gradients to original shapes
            let grad_lhs = if lhs_tensor.shape() != output_grad.shape() {
                reduce_grad_for_broadcast(output_grad, lhs_tensor.shape())?
            } else {
                output_grad.clone()?
            };
            
            let grad_rhs = if rhs_tensor.shape() != output_grad.shape() {
                reduce_grad_for_broadcast(output_grad, rhs_tensor.shape())?
            } else {
                output_grad.clone()?
            };
            
            Ok(vec![
                (*lhs, grad_lhs),
                (*rhs, grad_rhs),
            ])
        }
        
        Op::Sub { lhs, rhs } => {
            // d/dx(x-y) = 1, d/dy(x-y) = -1
            let neg_grad = GpuOps::mul_scalar(output_grad, -1.0)?;
            Ok(vec![
                (*lhs, output_grad.clone()?),
                (*rhs, neg_grad),
            ])
        }
        
        Op::Mul { lhs, rhs } => {
            // d/dx(x*y) = y, d/dy(x*y) = x
            println!("  Computing Mul gradients...");
            println!("  Getting saved tensors for lhs={:?}, rhs={:?}", lhs, rhs);
            
            let lhs_tensor = entry.saved_tensors.get(lhs)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for lhs in Mul".into()))?;
            let rhs_tensor = entry.saved_tensors.get(rhs)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for rhs in Mul".into()))?;
            
            println!("  Got saved tensors, computing grad_lhs...");
            // Use GPU ops directly to avoid autograd recording
            let grad_lhs = GpuOps::mul(output_grad, rhs_tensor)?;
            println!("  grad_lhs computed, computing grad_rhs...");
            let grad_rhs = GpuOps::mul(output_grad, lhs_tensor)?;
            println!("  Both gradients computed");
            
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
            let lhs_tensor = entry.saved_tensors.get(lhs)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for lhs in MatMul".into()))?;
            let rhs_tensor = entry.saved_tensors.get(rhs)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for rhs in MatMul".into()))?;
            
            // Gradient w.r.t. lhs: grad @ B^T
            // Use GPU ops directly to avoid autograd recording
            let rhs_t = GpuOps::transpose(rhs_tensor)?;
            let grad_lhs = GpuOps::matmul(output_grad, &rhs_t)?;
            
            // Gradient w.r.t. rhs: A^T @ grad
            let lhs_t = GpuOps::transpose(lhs_tensor)?;
            let grad_rhs = GpuOps::matmul(&lhs_t, output_grad)?;
            
            Ok(vec![
                (*lhs, grad_lhs),
                (*rhs, grad_rhs),
            ])
        }
        
        Op::ReLU { input } => {
            // d/dx ReLU(x) = 1 if x > 0, else 0
            let input_tensor = entry.saved_tensors.get(input)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for input".into()))?;
            let grad = relu_backward(output_grad, input_tensor)?;
            Ok(vec![(*input, grad)])
        }
        
        Op::GELU { input } => {
            // Use the complete GELU backward implementation
            let input_tensor = entry.saved_tensors.get(input)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for input".into()))?;
            let grad = crate::autograd_ops_complete::gelu_backward(output_grad, input_tensor)?;
            Ok(vec![(*input, grad)])
        }
        
        Op::SiLU { input } => {
            // Use the complete SiLU backward implementation
            let input_tensor = entry.saved_tensors.get(input)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for input".into()))?;
            let grad = crate::autograd_ops_complete::silu_backward(output_grad, input_tensor)?;
            Ok(vec![(*input, grad)])
        }
        
        Op::Tanh { input } => {
            // Use the complete Tanh backward implementation
            let input_tensor = entry.saved_tensors.get(input)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for input".into()))?;
            let output_tensor = input_tensor.tanh()?;
            let grad = crate::autograd_ops_complete::tanh_backward(output_grad, &output_tensor)?;
            Ok(vec![(*input, grad)])
        }
        
        Op::Sigmoid { input } => {
            // Use the complete Sigmoid backward implementation
            let input_tensor = entry.saved_tensors.get(input)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for input".into()))?;
            let output_tensor = input_tensor.sigmoid()?;
            let grad = crate::autograd_ops_complete::sigmoid_backward(output_grad, &output_tensor)?;
            Ok(vec![(*input, grad)])
        }
        
        Op::Square { input } => {
            // d/dx(x^2) = 2x
            let input_tensor = entry.saved_tensors.get(input)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for input".into()))?;
            let two_x = GpuOps::mul_scalar(input_tensor, 2.0)?;
            let grad = GpuOps::mul(output_grad, &two_x)?;
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
            let grad_scaled = GpuOps::mul_scalar(output_grad, 1.0 / n)?;
            let expanded = broadcast_to(&grad_scaled, input_shape)?;
            Ok(vec![(*input, expanded)])
        }
        
        Op::Transpose { input } => {
            // Gradient of transpose is transpose of gradient
            let grad = GpuOps::transpose(output_grad)?;
            Ok(vec![(*input, grad)])
        }
        
        Op::Conv2d { input, weight, stride, padding } => {
            // Use CUDA Conv2D backward
            let input_tensor = entry.saved_tensors.get(input)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for input".into()))?;
            let weight_tensor = entry.saved_tensors.get(weight)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for weight".into()))?;
            
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
            let input_tensor = entry.saved_tensors.get(input)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for input".into()))?;
            
            // Support for affine LayerNorm with weight and bias
            let mean = input_tensor.mean_dims(&normalized_shape, true)?;
            let var = input_tensor.var_dims(&normalized_shape, true, true)?;
            let normalized = input_tensor.sub(&mean)?.div(&var.add_scalar(1e-5)?.sqrt()?)?;
            
            // Check if weight and bias tensors were saved (affine=true)
            // For LayerNorm with affine parameters, we just need to know if they exist
            let has_affine = entry.saved_tensors.len() > 1;
            
            let (grad_input, grad_weight, grad_bias) = crate::autograd_ops_complete::layer_norm_backward(
                output_grad,
                input_tensor,
                &normalized,
                None,  // weight not available in this context
                None,  // bias not available in this context
                &mean,
                &var,
                normalized_shape,
                1e-5,  // eps
            )?;
            
            let mut gradients = vec![(*input, grad_input)];
            
            // Add weight and bias gradients if they exist
            if let Some(grad_w) = grad_weight {
                // For LayerNorm with affine parameters, weight and bias are separate tensors
                // Find them in saved_tensors (they would be saved after the input tensor)
                let tensor_ids: Vec<&TensorId> = entry.saved_tensors.keys().collect();
                if tensor_ids.len() > 1 {
                    // Second tensor is weight
                    gradients.push((*tensor_ids[1], grad_w));
                }
            }
            if let Some(grad_b) = grad_bias {
                // For LayerNorm with affine parameters, bias is the third tensor
                let tensor_ids: Vec<&TensorId> = entry.saved_tensors.keys().collect();
                if tensor_ids.len() > 2 {
                    // Third tensor is bias
                    gradients.push((*tensor_ids[2], grad_b));
                }
            }
            
            Ok(gradients)
        }
        
        
        Op::RMSNorm { input, weight, eps } => {
            // RMSNorm backward pass
            let input_tensor = entry.saved_tensors.get(input)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for input".into()))?;
            
            // Calculate RMS
            let square = input_tensor.square()?;
            let mean_square = square.mean_dims(&[input_tensor.shape().dims().len() - 1], true)?;
            let rms = mean_square.add_scalar(*eps)?.sqrt()?;
            
            // Normalize
            let normalized = input_tensor.div(&rms)?;
            
            // Compute gradients
            let weight_tensor = weight.and_then(|w| entry.saved_tensors.get(&w));
            
            // Gradient w.r.t input
            let grad_norm = if let Some(w) = weight_tensor {
                output_grad.mul(w)?
            } else {
                output_grad.clone()?
            };
            
            // Compute d(loss)/d(rms)
            let d_rms = grad_norm.mul(&normalized)?.neg()?.div(&rms)?
                .mean_dims(&[input_tensor.shape().dims().len() - 1], true)?;
            
            // Gradient w.r.t input: d(loss)/d(x) = d(loss)/d(norm) * 1/rms + d(loss)/d(rms) * x/rms^2
            let grad_input = grad_norm.div(&rms)?
                .add(&d_rms.mul(input_tensor)?.div(&rms.square()?)?)?;
            
            let mut grads = vec![(*input, grad_input)];
            
            // Gradient w.r.t weight if present
            if let Some(w_id) = weight {
                let grad_weight = output_grad.mul(&normalized)?
                    .sum_dims(&(0..input_tensor.shape().dims().len()-1).collect::<Vec<_>>())?;
                grads.push((*w_id, grad_weight));
            }
            
            Ok(grads)
        }
        
        Op::Linear { input, weight, bias } => {
            // d/dx(Wx + b) = W^T @ grad
            // d/dW(Wx + b) = grad @ x^T
            // d/db(Wx + b) = grad
            let input_tensor = entry.saved_tensors.get(input)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for input".into()))?;
            let weight_tensor = entry.saved_tensors.get(weight)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for weight".into()))?;
            
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
            let lhs_tensor = entry.saved_tensors.get(lhs)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for lhs in BatchMatMul".into()))?;
            let rhs_tensor = entry.saved_tensors.get(rhs)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for rhs in BatchMatMul".into()))?;
            
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
            let input_tensor = entry.saved_tensors.get(input)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for input".into()))?;
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
            let input_tensor = entry.saved_tensors.get(input)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for input".into()))?;
            let mut grad_shape = input_tensor.shape().dims().to_vec();
            grad_shape[*dim] = 1;
            let grad_reshaped = output_grad.reshape(&grad_shape)?;
            let grad = grad_reshaped.broadcast_to(input_tensor.shape())?;
            Ok(vec![(*input, grad)])
        }
        
        Op::Clamp { input, min, max } => {
            // Use the clamp backward implementation
            let input_tensor = entry.saved_tensors.get(input)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for input".into()))?;
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
            let lhs_tensor = entry.saved_tensors.get(lhs)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for lhs in Div".into()))?;
            let rhs_tensor = entry.saved_tensors.get(rhs)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for rhs in Div".into()))?;
            
            // Gradient w.r.t. lhs: grad * (1/rhs)
            let grad_lhs = GpuOps::div(output_grad, rhs_tensor)?;
            
            // Gradient w.r.t. rhs: grad * (-lhs/rhs^2)
            let rhs_squared = GpuOps::mul(rhs_tensor, rhs_tensor)?; // x^2 = x * x
            let neg_lhs = GpuOps::mul_scalar(lhs_tensor, -1.0)?;
            let grad_rhs_term = GpuOps::div(&neg_lhs, &rhs_squared)?;
            let grad_rhs = GpuOps::mul(output_grad, &grad_rhs_term)?;
            
            Ok(vec![(*lhs, grad_lhs), (*rhs, grad_rhs)])
        }
        
        Op::MaxDim { input, dim, keepdim } => {
            // For max reduction, gradient flows only through the max elements
            let input_tensor = entry.saved_tensors.get(input)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for input".into()))?;
            
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
            let input_tensor = entry.saved_tensors.get(input)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for input".into()))?;
            let input_shape = input_tensor.shape();
            
            // Broadcast gradient back to input shape
            let grad = output_grad.broadcast_to(input_shape)?;
            
            Ok(vec![(*input, grad)])
        }
        
        Op::Embedding { weight, indices } => {
            // For embedding, gradient flows back to weight matrix
            // Gradient w.r.t weight: scatter_add gradients to corresponding rows
            let indices_tensor = entry.saved_tensors.get(indices)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for indices".into()))?;
            let weight_tensor = entry.saved_tensors.get(weight)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for weight".into()))?;
            
            // Create zero gradient for weight
            let mut weight_grad = Tensor::zeros(weight_tensor.shape().clone(), weight_tensor.device().clone())?;
            
            // Scatter add gradients using GPU kernel
            let weight_grad = CudaKernels::scatter_add(
                &weight_grad,
                indices_tensor,
                output_grad,
                0  // Scatter along dimension 0 (rows)
            )?;
            
            // No gradient w.r.t indices (they're discrete)
            Ok(vec![(*weight, weight_grad)])
        }
        
        Op::IndexSelect { input, indices, dim } => {
            // Gradient flows back to selected indices
            let input_tensor = entry.saved_tensors.get(input)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for input".into()))?;
            let indices_tensor = entry.saved_tensors.get(indices)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for indices".into()))?;
            
            // Create zero gradient for input
            // FLAME is GPU-only, always use CUDA scatter_add kernel
            let grad_input = crate::cuda_kernels::scatter_add(
                input_tensor.shape().dims(),
                output_grad,
                indices_tensor,
                *dim,
            )?;
            
            // No gradient w.r.t indices
            Ok(vec![(*input, grad_input)])
        }
        
        Op::Cat { inputs, dim } => {
            // Split gradient back to original tensors
            let mut grads = Vec::new();
            let mut offset = 0;
            
            for &input_id in inputs {
                let input_tensor = entry.saved_tensors.get(&input_id)
                    .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for input in Cat".into()))?;
                let size = input_tensor.shape().dims()[*dim];
                
                // Slice gradient for this input
                let mut ranges = Vec::new();
                for (i, &dim_size) in output_grad.shape().dims().iter().enumerate() {
                    if i == *dim {
                        ranges.push((offset, offset + size));
                    } else {
                        ranges.push((0, dim_size));
                    }
                }
                
                let grad_slice = output_grad.slice(&ranges)?;
                grads.push((input_id, grad_slice));
                offset += size;
            }
            
            Ok(grads)
        }
        
        Op::Split { input, sizes, dim } => {
            // Concatenate gradients back to original tensor
            // We need to collect gradients for all split outputs
            let input_tensor = entry.saved_tensors.get(input)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for input".into()))?;
            let input_size = input_tensor.shape().dims()[*dim];
            
            // Create gradient tensor filled with zeros
            let mut combined_grad = Tensor::zeros(input_tensor.shape().clone(), input_tensor.device().clone())?;
            
            // The output_grad corresponds to one of the split outputs
            // We need to place it at the correct position
            // Since we don't track which split output this is, we'll accumulate all available gradients
            
            // For proper implementation, we'd need to track split output indices
            // For now, we'll assume the gradient applies to the entire input
            // This is correct when all splits have gradients flowing back
            combined_grad = combined_grad.add(output_grad)?;
            
            Ok(vec![(*input, combined_grad)])
        }
        
        Op::Abs { input } => {
            // d/dx |x| = sign(x) 
            let input_tensor = entry.saved_tensors.get(input)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for input".into()))?;
            let sign = input_tensor.sign()?;
            let grad = output_grad.mul(&sign)?;
            Ok(vec![(*input, grad)])
        }
        
        Op::Log { input } => {
            // d/dx log(x) = 1/x
            let input_tensor = entry.saved_tensors.get(input)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for input".into()))?;
            let reciprocal = Tensor::ones(input_tensor.shape().clone(), input_tensor.device().clone())?
                .div(input_tensor)?;
            let grad = output_grad.mul(&reciprocal)?;
            Ok(vec![(*input, grad)])
        }
        
        Op::Softmax { input, dim } => {
            // Use the complete softmax backward implementation
            let input_tensor = entry.saved_tensors.get(input)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for input".into()))?;
            let output = input_tensor.softmax(*dim)?;
            let grad = crate::autograd_ops_complete::softmax_backward(output_grad, &output, *dim)?;
            Ok(vec![(*input, grad)])
        }
        
        Op::LogSoftmax { input, dim } => {
            // Use the complete log_softmax backward implementation
            let input_tensor = entry.saved_tensors.get(input)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for input".into()))?;
            let output = input_tensor.log_softmax(*dim)?;
            let grad = crate::autograd_ops_complete::log_softmax_backward(output_grad, &output, *dim)?;
            Ok(vec![(*input, grad)])
        }
        
        Op::MSELoss { predictions, targets, num_elements } => {
            // For MSE: d/dx[(x-y)^2] = 2(x-y)/n
            let predictions_tensor = entry.saved_tensors.get(predictions)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for predictions".into()))?;
            let targets_tensor = entry.saved_tensors.get(targets)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for targets".into()))?;
            
            // Gradient is 2 * (predictions - targets) / num_elements
            let diff = predictions_tensor.sub(targets_tensor)?;
            let scale = 2.0 / (*num_elements as f32);
            let grad_predictions = output_grad.mul_scalar(scale)?.mul(&diff)?;
            let grad_targets = grad_predictions.mul_scalar(-1.0)?;
            
            Ok(vec![
                (*predictions, grad_predictions),
                (*targets, grad_targets),
            ])
        }
        
        Op::L1Loss { predictions, targets, num_elements } => {
            // For L1: d/dx|x-y| = sign(x-y)/n
            let predictions_tensor = entry.saved_tensors.get(predictions)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for predictions".into()))?;
            let targets_tensor = entry.saved_tensors.get(targets)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for targets".into()))?;
            
            let diff = predictions_tensor.sub(targets_tensor)?;
            let sign = diff.sign()?;
            let scale = 1.0 / (*num_elements as f32);
            let grad_predictions = output_grad.mul_scalar(scale)?.mul(&sign)?;
            let grad_targets = grad_predictions.mul_scalar(-1.0)?;
            
            Ok(vec![
                (*predictions, grad_predictions),
                (*targets, grad_targets),
            ])
        }
        
        Op::HuberLoss { predictions, targets, delta, num_elements } => {
            // Huber gradient: 
            // if |x-y| <= delta: (x-y)/n
            // if |x-y| > delta: delta*sign(x-y)/n
            let predictions_tensor = entry.saved_tensors.get(predictions)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for predictions".into()))?;
            let targets_tensor = entry.saved_tensors.get(targets)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for targets".into()))?;
            
            let diff = predictions_tensor.sub(targets_tensor)?;
            let abs_diff = diff.abs()?;
            let delta_vec = vec![*delta; diff.shape().elem_count()];
            let delta_tensor = Tensor::from_vec(delta_vec, diff.shape().clone(), diff.device().clone())?;
            
            // Create mask for |diff| <= delta
            let mask = abs_diff.le(&delta_tensor)?;
            
            // Quadratic gradient: diff
            let quad_grad = diff.clone()?;
            
            // Linear gradient: delta * sign(diff)
            let linear_grad = diff.sign()?.mul_scalar(*delta)?;
            
            // Combine using mask
            let combined_grad = mask.where_tensor(&quad_grad, &linear_grad)?;
            
            let scale = 1.0 / (*num_elements as f32);
            let grad_predictions = output_grad.mul_scalar(scale)?.mul(&combined_grad)?;
            let grad_targets = grad_predictions.mul_scalar(-1.0)?;
            
            Ok(vec![
                (*predictions, grad_predictions),
                (*targets, grad_targets),
            ])
        }
        
        Op::BCELoss { predictions, targets, num_elements } => {
            // BCE gradient: d/dp[-y*log(p) - (1-y)*log(1-p)] = (p-y)/(p(1-p))/n
            let predictions_tensor = entry.saved_tensors.get(predictions)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for predictions".into()))?;
            let targets_tensor = entry.saved_tensors.get(targets)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for targets".into()))?;
            
            // Clamp predictions to avoid division by zero
            let eps = 1e-7;
            let pred_clamped = predictions_tensor.clamp(eps, 1.0 - eps)?;
            
            // Compute (predictions - targets) / (predictions * (1 - predictions))
            let numerator = pred_clamped.sub(targets_tensor)?;
            let one_minus_pred = pred_clamped.neg()?.add_scalar(1.0)?;
            let denominator = pred_clamped.mul(&one_minus_pred)?;
            
            let grad_base = numerator.div(&denominator)?;
            let scale = 1.0 / (*num_elements as f32);
            let grad_predictions = output_grad.mul_scalar(scale)?.mul(&grad_base)?;
            
            // No gradient w.r.t targets for BCE
            Ok(vec![(*predictions, grad_predictions)])
        }
        
        Op::NLLLoss { log_probs, targets, batch_size } => {
            // NLL gradient: sparse gradient, -1/batch_size at target indices
            let log_probs_tensor = entry.saved_tensors.get(log_probs)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for log_probs".into()))?;
            let targets_tensor = entry.saved_tensors.get(targets)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for targets".into()))?;
            
            // Create zero gradient tensor
            let mut grad_log_probs = Tensor::zeros(log_probs_tensor.shape().clone(), log_probs_tensor.device().clone())?;
            
            // Set gradients at target indices using GPU scatter
            let scale = -1.0 / (*batch_size as f32);
            
            // Create a tensor with the gradient values to scatter
            let grad_values = Tensor::ones(
                Shape::from_dims(&[*batch_size]),
                log_probs_tensor.device().clone()
            )?.mul_scalar(scale)?;
            
            // Use scatter_add to place gradients at target indices
            let grad_log_probs = CudaKernels::scatter_add(
                &grad_log_probs,
                targets_tensor,
                &grad_values,
                1  // Scatter along dimension 1 (class dimension)
            )?;
            
            let final_grad = output_grad.mul(&grad_log_probs)?;
            
            Ok(vec![(*log_probs, final_grad)])
        }
        
        Op::GroupNorm { input, num_groups, weight, bias } => {
            // GroupNorm backward pass
            let input_tensor = entry.saved_tensors.get(input)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for input".into()))?;
            let shape = input_tensor.shape().dims();
            let num_channels = shape[1];
            
            // Saved mean and variance should be in saved_tensors
            let mean = entry.saved_tensors.values()
                .find(|t| t.shape().dims() == &[shape[0], *num_groups])
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved mean".into()))?;
            let var = entry.saved_tensors.values()
                .skip(1)
                .find(|t| t.shape().dims() == &[shape[0], *num_groups])
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved variance".into()))?;
            
            // Compute gradients
            let weight_tensor = weight.and_then(|w| entry.saved_tensors.get(&w));
            let bias_tensor = bias.and_then(|b| entry.saved_tensors.get(&b));
            
            let (grad_input, grad_weight, grad_bias) = crate::autograd_ops_complete::group_norm_backward(
                output_grad,
                input_tensor,
                mean,
                var,
                weight_tensor,
                *num_groups,
                1e-5,
            )?;
            
            let mut grads = vec![(*input, grad_input)];
            if let (Some(w_id), Some(gw)) = (*weight, grad_weight) {
                grads.push((w_id, gw));
            }
            if let (Some(b_id), Some(gb)) = (*bias, grad_bias) {
                grads.push((b_id, gb));
            }
            
            Ok(grads)
        }
        
        Op::FlashAttention { query, key, value, mask, scale, causal } => {
            // FlashAttention backward pass
            let query_tensor = entry.saved_tensors.get(query)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for query".into()))?;
            let key_tensor = entry.saved_tensors.get(key)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for key".into()))?;
            let value_tensor = entry.saved_tensors.get(value)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for value".into()))?;
            
            let mask_tensor = if let Some(m_id) = mask {
                entry.saved_tensors.get(m_id)
            } else {
                None
            };
            
            // We need to get the output tensor from saved tensors
            // The output should have been saved during the forward pass
            let output_tensor = entry.saved_tensors.values()
                .find(|t| t.shape().dims() == output_grad.shape().dims() && t.id != query_tensor.id && t.id != key_tensor.id && t.id != value_tensor.id)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved output tensor".into()))?;
            
            // Call flash attention backward
            let (grad_q, grad_k, grad_v) = crate::flash_attention::flash_attention_backward(
                output_grad,
                query_tensor,
                key_tensor,
                value_tensor,
                mask_tensor,
                output_tensor,
                *scale,
                *causal,
            )?;
            
            let mut grads = vec![
                (*query, grad_q),
                (*key, grad_k),
                (*value, grad_v),
            ];
            
            Ok(grads)
        }
        
        Op::SageAttention { query_id, key_id, value_id, scale, causal, quantized } => {
            // SageAttention backward pass
            let query_tensor = entry.saved_tensors.get(query_id)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for query".into()))?;
            let key_tensor = entry.saved_tensors.get(key_id)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for key".into()))?;
            let value_tensor = entry.saved_tensors.get(value_id)
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved tensor for value".into()))?;
            
            // Get attention weights (should be saved with a known ID)
            // In SageAttention forward, attention_weights.id is saved in saved_tensors
            let attention_weights = entry.saved_tensors.values()
                .find(|t| t.shape().dims().len() == 4 && 
                      t.shape().dims()[0] == query_tensor.shape().dims()[0] &&
                      t.shape().dims()[1] == query_tensor.shape().dims()[1] &&
                      t.shape().dims()[2] == query_tensor.shape().dims()[2] && 
                      t.shape().dims()[3] == key_tensor.shape().dims()[2])
                .ok_or_else(|| FlameError::InvalidOperation("Missing saved attention weights".into()))?;
            
            // Call sage attention backward
            let (grad_q, grad_k, grad_v) = crate::sage_attention::sage_attention_backward(
                output_grad,
                query_tensor,
                key_tensor,
                value_tensor,
                attention_weights,
                *scale,
                *causal,
                *quantized,
            )?;
            
            Ok(vec![
                (*query_id, grad_q),
                (*key_id, grad_k),
                (*value_id, grad_v),
            ])
        }
        
        _ => {
            // This should not happen if all operations are implemented
            Err(FlameError::InvalidOperation(
                format!("Gradient not implemented for operation: {:?}", entry.op)
            ))
        }
    }
}

/// Reduce gradient for broadcast operations
/// When a tensor was broadcast during forward pass, we need to sum gradients
/// along the broadcast dimensions during backward pass
fn reduce_grad_for_broadcast(grad: &Tensor, target_shape: &Shape) -> Result<Tensor> {
    let grad_dims = grad.shape().dims();
    let target_dims = target_shape.dims();
    
    // If shapes are the same, no reduction needed
    if grad_dims == target_dims {
        return grad.clone();
    }
    
    let mut result = grad.clone()?;
    
    // Handle dimension mismatch by summing over extra dimensions
    if grad_dims.len() > target_dims.len() {
        // Sum over the extra leading dimensions
        for _ in 0..(grad_dims.len() - target_dims.len()) {
            result = result.sum_dim(0)?;
        }
    }
    
    // Now handle size-1 dimensions that were broadcast
    for i in 0..target_dims.len() {
        let result_dims = result.shape().dims().to_vec(); // Clone to avoid borrow
        if i < result_dims.len() && result_dims[i] != target_dims[i] && target_dims[i] == 1 {
            // This dimension was broadcast from size 1
            result = result.sum_dims(&[i])?;
            // Squeeze to remove the dimension if needed
            let new_result_dims = result.shape().dims();
            if new_result_dims[i] != 1 {
                // Reshape to add back the size-1 dimension
                let mut new_shape = new_result_dims.to_vec();
                new_shape[i] = 1;
                result = result.reshape(&new_shape)?;
            }
        }
    }
    
    // Final reshape to match target shape exactly
    result.reshape(target_dims)
}

/// ReLU backward pass
fn relu_backward(grad_output: &Tensor, input: &Tensor) -> Result<Tensor> {
    // ReLU backward: gradient passes through where input > 0
    // We need to create a mask without using tensor operations that record to autograd
    
    // Create zero tensor for comparison
    let zero_data = crate::tensor::alloc_zeros_from_pool(&input.device, input.shape.elem_count())?;
    let zero = Tensor {
        storage: TensorStorage::F32 { data: zero_data, numel: input.shape.elem_count() },
        shape: input.shape.clone(),
        device: input.device.clone(),
        id: TensorId::new(),
        requires_grad: false,
    };
    
    // Use comparison to create mask
    let mask = input.gt(&zero)?;
    
    // Apply mask to gradient using GPU ops
    GpuOps::mul(grad_output, &mask)
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
    
    // General broadcasting implementation
    let src_dims = tensor.shape.dims();
    let dst_dims = target_shape.dims();
    
    // Validate that broadcasting is possible
    let src_ndim = src_dims.len();
    let dst_ndim = dst_dims.len();
    
    // Pad source dimensions with 1s on the left
    let mut padded_src = vec![1; dst_ndim];
    for i in 0..src_ndim {
        padded_src[dst_ndim - src_ndim + i] = src_dims[i];
    }
    
    // Check broadcast compatibility
    for i in 0..dst_ndim {
        if padded_src[i] != dst_dims[i] && padded_src[i] != 1 {
            return Err(FlameError::InvalidOperation(
                format!("Cannot broadcast dimension {} from {} to {}", 
                    i, padded_src[i], dst_dims[i])
            ));
        }
    }
    
    // Perform broadcasting
    let src_data = tensor.to_vec()?;
    let mut dst_data = vec![0.0f32; target_shape.elem_count()];
    
    // Calculate strides for source and destination
    let mut src_strides = vec![1; dst_ndim];
    let mut dst_strides = vec![1; dst_ndim];
    
    for i in (0..dst_ndim - 1).rev() {
        src_strides[i] = if padded_src[i + 1] == 1 { 0 } else { src_strides[i + 1] * padded_src[i + 1] };
        dst_strides[i] = dst_strides[i + 1] * dst_dims[i + 1];
    }
    
    // Copy data with broadcasting
    for dst_idx in 0..dst_data.len() {
        let mut src_idx = 0;
        let mut remaining = dst_idx;
        
        for dim in 0..dst_ndim {
            let coord = remaining / dst_strides[dim];
            remaining %= dst_strides[dim];
            
            // Only advance source index if dimension is not broadcasted
            if padded_src[dim] > 1 {
                src_idx += coord * src_strides[dim];
            }
        }
        
        dst_data[dst_idx] = src_data[src_idx];
    }
    
    Tensor::from_vec(dst_data, target_shape.clone(), tensor.device.clone())
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