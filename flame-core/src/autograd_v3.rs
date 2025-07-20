//! Clean autograd implementation with separated gradients
//! 
//! This implementation avoids borrow checker issues by returning gradients
//! instead of mutating internal state.

use crate::{Tensor, Result, FlameError, Shape};
use crate::tensor::TensorId;
use crate::gradient::GradientMap;
use std::collections::HashMap;
use std::sync::Arc;
use cudarc::driver::LaunchAsync;

/// Operation types for autograd
#[derive(Debug, Clone)]
pub enum Op {
    Add { lhs: TensorId, rhs: TensorId },
    Sub { lhs: TensorId, rhs: TensorId },
    Mul { lhs: TensorId, rhs: TensorId },
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
}

/// Entry in the computation tape
pub struct TapeEntry {
    /// Output tensor ID
    pub output_id: TensorId,
    
    /// Operation that produced the output
    pub op: Op,
    
    /// Saved tensors needed for backward pass
    pub saved_tensors: HashMap<TensorId, Tensor>,
}

/// Autograd engine - no internal gradient storage
pub struct AutogradEngine {
    /// Computation tape
    tape: Vec<TapeEntry>,
    
    /// Device for this engine
    device: Arc<cudarc::driver::CudaDevice>,
}

impl AutogradEngine {
    /// Create a new autograd engine
    pub fn new(device: Arc<cudarc::driver::CudaDevice>) -> Self {
        Self {
            tape: Vec::new(),
            device,
        }
    }
    
    /// Record an operation
    pub fn record(&mut self, entry: TapeEntry) {
        self.tape.push(entry);
    }
    
    /// Record an operation (convenience method)
    pub fn record_op(
        &mut self,
        output_id: TensorId,
        op: Op,
        saved_tensors: Vec<(TensorId, Tensor)>,
    ) {
        let mut saved = HashMap::new();
        for (id, tensor) in saved_tensors {
            saved.insert(id, tensor);
        }
        
        self.record(TapeEntry {
            output_id,
            op,
            saved_tensors: saved,
        });
    }
    
    /// Clear the tape for next iteration
    pub fn clear(&mut self) {
        self.tape.clear();
    }
    
    /// Backward pass - returns gradients separately
    pub fn backward(&self, loss: &Tensor) -> Result<GradientMap> {
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
        
        // Initialize gradient storage
        let mut gradients = GradientMap::new(self.device.clone());
        gradients.set_ones(loss.id, loss.shape.clone())?;
        
        // Process tape in reverse - no mutation of self!
        for entry in self.tape.iter().rev() {
            if let Some(output_grad) = gradients.get(entry.output_id) {
                let output_grad = output_grad.clone()?;
                // Compute input gradients - pure functions
                let input_grads = self.compute_gradients(entry, &output_grad)?;
                
                // Accumulate gradients
                for (tensor_id, grad) in input_grads {
                    gradients.accumulate(tensor_id, grad)?;
                }
            }
        }
        
        Ok(gradients)
    }
    
    /// Pure gradient computation - no mutation
    fn compute_gradients(&self, entry: &TapeEntry, output_grad: &Tensor) 
        -> Result<Vec<(TensorId, Tensor)>> {
        
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
                // For element-wise multiplication: d/dx(x*y) = y, d/dy(x*y) = x
                if let (Some(x), Some(y)) = (entry.saved_tensors.get(lhs), entry.saved_tensors.get(rhs)) {
                    let grad_x = output_grad.mul(y)?;
                    let grad_y = output_grad.mul(x)?;
                    
                    Ok(vec![
                        (*lhs, grad_x),
                        (*rhs, grad_y),
                    ])
                } else {
                    Err(FlameError::InvalidOperation(
                        "Mul backward: missing saved tensors".into()
                    ))
                }
            }
            
            Op::MulScalar { input, scalar } => {
                // d/dx(x*c) = c
                let grad = output_grad.mul_scalar(*scalar)?;
                Ok(vec![(*input, grad)])
            }
            
            Op::MatMul { lhs, rhs } => {
                // For C = A @ B:
                // dA = dC @ B^T
                // dB = A^T @ dC
                if let (Some(a), Some(b)) = (entry.saved_tensors.get(lhs), entry.saved_tensors.get(rhs)) {
                    let grad_a = output_grad.matmul(&b.transpose()?)?;
                    let grad_b = a.transpose()?.matmul(output_grad)?;
                    
                    Ok(vec![
                        (*lhs, grad_a),
                        (*rhs, grad_b),
                    ])
                } else {
                    Err(FlameError::InvalidOperation(
                        "MatMul backward: missing saved tensors".into()
                    ))
                }
            }
            
            Op::ReLU { input } => {
                // ReLU gradient: 1 if x > 0, 0 otherwise
                if let Some(x) = entry.saved_tensors.get(input) {
                    let grad = self.relu_backward(output_grad, x)?;
                    Ok(vec![(*input, grad)])
                } else {
                    Err(FlameError::InvalidOperation(
                        "ReLU backward: missing saved input".into()
                    ))
                }
            }
            
            Op::Square { input } => {
                // d/dx(x^2) = 2x
                if let Some(x) = entry.saved_tensors.get(input) {
                    let grad = output_grad.mul(x)?.mul_scalar(2.0)?;
                    Ok(vec![(*input, grad)])
                } else {
                    Err(FlameError::InvalidOperation(
                        "Square backward: missing saved input".into()
                    ))
                }
            }
            
            Op::Sum { input, input_shape } => {
                // Broadcast scalar gradient to input shape
                let grad = output_grad.broadcast_to(input_shape)?;
                Ok(vec![(*input, grad)])
            }
            
            Op::Mean { input, input_shape } => {
                // Gradient is 1/n for each element
                let n = input_shape.elem_count() as f32;
                let grad = output_grad.mul_scalar(1.0 / n)?.broadcast_to(input_shape)?;
                Ok(vec![(*input, grad)])
            }
            
            Op::Transpose { input } => {
                // Gradient of transpose is transpose
                let grad = output_grad.transpose()?;
                Ok(vec![(*input, grad)])
            }
            
            Op::AddBias { input, bias } => {
                // Input gradient passes through unchanged
                let input_grad = output_grad.clone()?;
                
                // Bias gradient is sum over all dimensions except the last
                let grad_shape = output_grad.shape().dims();
                let mut grad_bias = output_grad.clone()?;
                
                // Sum over batch dimensions
                for i in 0..grad_shape.len() - 1 {
                    grad_bias = grad_bias.sum_dim(0)?;
                }
                
                Ok(vec![
                    (*input, input_grad),
                    (*bias, grad_bias),
                ])
            }
            
            Op::SumDim { input, dim } => {
                // Broadcast gradient along the summed dimension
                if let Some(input_tensor) = entry.saved_tensors.get(input) {
                    let input_shape = input_tensor.shape();
                    let grad = self.broadcast_sum_grad(output_grad, input_shape, *dim)?;
                    Ok(vec![(*input, grad)])
                } else {
                    Err(FlameError::InvalidOperation(
                        "SumDim backward: missing saved input".into()
                    ))
                }
            }
            
            _ => {
                // Add more operations as needed
                Err(FlameError::InvalidOperation(
                    format!("Backward not implemented for {:?}", entry.op)
                ))
            }
        }
    }
    
    // Helper methods for gradient computation
    
    fn relu_backward(&self, grad_out: &Tensor, input: &Tensor) -> Result<Tensor> {
        // ReLU backward: gradient passes through where input > 0
        // For now, implement using CUDA kernel
        let kernel_code = r#"
extern "C" __global__ void relu_backward_kernel(
    float *grad_in,
    const float *grad_out,
    const float *input,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        grad_in[idx] = input[idx] > 0.0f ? grad_out[idx] : 0.0f;
    }
}"#;
        
        crate::cuda_kernels::CudaKernels::ensure_kernel(&input.device, "relu_backward_kernel", kernel_code)?;
        
        let f = input.device.get_func("relu_backward_kernel", "relu_backward_kernel")
            .ok_or_else(|| crate::FlameError::Cuda("Failed to get relu_backward kernel".into()))?;
        
        let numel = input.shape().elem_count();
        let mut grad_in = unsafe { input.device.alloc::<f32>(numel) }
            .map_err(|_| crate::FlameError::CudaDriver)?;
        
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(numel as u32);
        unsafe {
            f.launch(
                cfg,
                (&mut grad_in, &*grad_out.data, &*input.data, numel as i32),
            ).map_err(|_| crate::FlameError::Cuda("Failed to launch relu_backward kernel".into()))?;
        }
        
        Ok(crate::cuda_kernels::create_output_tensor(
            grad_in,
            input.shape().clone(),
            input.device.clone()
        ))
    }
    
    fn broadcast_sum_grad(&self, grad: &Tensor, target_shape: &Shape, sum_dim: usize) -> Result<Tensor> {
        // Insert dimension that was summed
        let mut new_shape = grad.shape().dims().to_vec();
        new_shape.insert(sum_dim, 1);
        
        // Reshape and broadcast
        let reshaped = grad.reshape(&new_shape)?;
        reshaped.broadcast_to(target_shape)
    }
}

/// Thread-local autograd engine
thread_local! {
    static ENGINE: std::cell::RefCell<Option<AutogradEngine>> = std::cell::RefCell::new(None);
}

/// Record an operation in the thread-local engine
pub fn record_op(output_id: TensorId, op: Op, saved_tensors: Vec<(TensorId, Tensor)>) {
    ENGINE.with(|e| {
        if let Some(engine) = e.borrow_mut().as_mut() {
            engine.record_op(output_id, op, saved_tensors);
        }
    });
}

/// Set the thread-local engine
pub fn set_engine(engine: AutogradEngine) {
    ENGINE.with(|e| {
        *e.borrow_mut() = Some(engine);
    });
}

/// Clear the thread-local engine
pub fn clear_engine() {
    ENGINE.with(|e| {
        if let Some(engine) = e.borrow_mut().as_mut() {
            engine.clear();
        }
    });
}

/// Take the thread-local engine
pub fn take_engine() -> Option<AutogradEngine> {
    ENGINE.with(|e| {
        e.borrow_mut().take()
    })
}