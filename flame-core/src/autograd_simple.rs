/// Simple, working autograd implementation
use crate::{Tensor, Result, FlameError, Shape};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Global tensor ID counter
pub static NEXT_TENSOR_ID: AtomicUsize = AtomicUsize::new(1);

/// Tensor ID type
pub type TensorId = usize;

/// Operation types
#[derive(Debug, Clone)]
pub enum Op {
    Add { lhs: TensorId, rhs: TensorId },
    Sub { lhs: TensorId, rhs: TensorId },
    Mul { lhs: TensorId, rhs: TensorId },
    MulScalar { input: TensorId, scalar: f32 },
    MatMul { lhs: TensorId, rhs: TensorId },
    ReLU { input: TensorId },
    GELU { input: TensorId },
    SiLU { input: TensorId },
    Tanh { input: TensorId },
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

/// Tape entry for backward pass
struct TapeEntry {
    output_id: TensorId,
    op: Op,
    saved_tensors: HashMap<TensorId, Tensor>,
}

/// Simple autograd engine
pub struct AutogradEngine {
    pub tape: Vec<TapeEntry>,
    pub gradients: HashMap<TensorId, Tensor>,
}

impl AutogradEngine {
    pub fn new() -> Self {
        Self {
            tape: Vec::new(),
            gradients: HashMap::new(),
        }
    }
    
    /// Record an operation
    pub fn record_op(
        &mut self,
        output_id: TensorId,
        op: Op,
        inputs: Vec<(TensorId, Tensor)>,
    ) {
        let mut saved = HashMap::new();
        for (id, tensor) in inputs {
            saved.insert(id, tensor);
        }
        
        self.tape.push(TapeEntry {
            output_id,
            op,
            saved_tensors: saved,
        });
    }
    
    /// Get or create gradient for a tensor
    fn get_or_create_grad(&mut self, id: TensorId, shape: &Shape, device: &std::sync::Arc<cudarc::driver::CudaDevice>) -> Result<&mut Tensor> {
        use std::collections::hash_map::Entry;
        match self.gradients.entry(id) {
            Entry::Occupied(e) => Ok(e.into_mut()),
            Entry::Vacant(e) => {
                let zeros = Tensor::zeros(shape.clone(), device.clone())?;
                Ok(e.insert(zeros))
            }
        }
    }
    
    /// Perform backward pass
    pub fn backward(&mut self, loss_id: TensorId, loss_shape: &Shape, device: &std::sync::Arc<cudarc::driver::CudaDevice>) -> Result<()> {
        // Initialize loss gradient to 1.0
        let ones = Tensor::ones(loss_shape.clone(), device.clone())?;
        self.gradients.insert(loss_id, ones);
        
        // Process tape in reverse
        for entry in self.tape.iter().rev() {
            // Skip if no gradient flowing to this output
            let grad_output = match self.gradients.get(&entry.output_id) {
                Some(g) => g.clone_result()?,
                None => continue,
            };
            
            // Compute gradients based on operation
            match &entry.op {
                Op::Add { lhs, rhs } => {
                    // Gradient flows unchanged to both inputs
                    if let Some(lhs_tensor) = entry.saved_tensors.get(lhs) {
                        let lhs_grad = self.get_or_create_grad(*lhs, lhs_tensor.shape(), device)?;
                        *lhs_grad = lhs_grad.add(&grad_output)?;
                    }
                    if let Some(rhs_tensor) = entry.saved_tensors.get(rhs) {
                        let rhs_grad = self.get_or_create_grad(*rhs, rhs_tensor.shape(), device)?;
                        *rhs_grad = rhs_grad.add(&grad_output)?;
                    }
                }
                
                Op::Sub { lhs, rhs } => {
                    // d/dx (x - y) = 1, d/dy (x - y) = -1
                    if let Some(lhs_tensor) = entry.saved_tensors.get(lhs) {
                        let lhs_grad = self.get_or_create_grad(*lhs, lhs_tensor.shape(), device)?;
                        *lhs_grad = lhs_grad.add(&grad_output)?;
                    }
                    if let Some(rhs_tensor) = entry.saved_tensors.get(rhs) {
                        let neg_grad = grad_output.mul_scalar(-1.0)?;
                        let rhs_grad = self.get_or_create_grad(*rhs, rhs_tensor.shape(), device)?;
                        *rhs_grad = rhs_grad.add(&neg_grad)?;
                    }
                }
                
                Op::Mul { lhs, rhs } => {
                    // d/dx (x * y) = y, d/dy (x * y) = x
                    if let (Some(lhs_tensor), Some(rhs_tensor)) = 
                        (entry.saved_tensors.get(lhs), entry.saved_tensors.get(rhs)) {
                        
                        // Gradient w.r.t lhs
                        let grad_lhs = grad_output.mul(rhs_tensor)?;
                        let lhs_grad = self.get_or_create_grad(*lhs, lhs_tensor.shape(), device)?;
                        *lhs_grad = lhs_grad.add(&grad_lhs)?;
                        
                        // Gradient w.r.t rhs
                        let grad_rhs = grad_output.mul(lhs_tensor)?;
                        let rhs_grad = self.get_or_create_grad(*rhs, rhs_tensor.shape(), device)?;
                        *rhs_grad = rhs_grad.add(&grad_rhs)?;
                    }
                }
                
                Op::MulScalar { input, scalar } => {
                    // d/dx (s * x) = s
                    if let Some(input_tensor) = entry.saved_tensors.get(input) {
                        let grad_input = grad_output.mul_scalar(*scalar)?;
                        let input_grad = self.get_or_create_grad(*input, input_tensor.shape(), device)?;
                        *input_grad = input_grad.add(&grad_input)?;
                    }
                }
                
                Op::MatMul { lhs, rhs } => {
                    // d/dA (A @ B) = grad @ B^T
                    // d/dB (A @ B) = A^T @ grad
                    if let (Some(lhs_tensor), Some(rhs_tensor)) = 
                        (entry.saved_tensors.get(lhs), entry.saved_tensors.get(rhs)) {
                        
                        // Gradient w.r.t lhs
                        let rhs_t = rhs_tensor.transpose()?;
                        let grad_lhs = grad_output.matmul(&rhs_t)?;
                        let lhs_grad = self.get_or_create_grad(*lhs, lhs_tensor.shape(), device)?;
                        *lhs_grad = lhs_grad.add(&grad_lhs)?;
                        
                        // Gradient w.r.t rhs
                        let lhs_t = lhs_tensor.transpose()?;
                        let grad_rhs = lhs_t.matmul(&grad_output)?;
                        let rhs_grad = self.get_or_create_grad(*rhs, rhs_tensor.shape(), device)?;
                        *rhs_grad = rhs_grad.add(&grad_rhs)?;
                    }
                }
                
                Op::ReLU { input } => {
                    // d/dx ReLU(x) = 1 if x > 0, else 0
                    if let Some(input_tensor) = entry.saved_tensors.get(input) {
                        let grad_input = crate::autograd_ops::BackwardOps::relu_backward(&grad_output, input_tensor)?;
                        let input_grad = self.get_or_create_grad(*input, input_tensor.shape(), device)?;
                        *input_grad = input_grad.add(&grad_input)?;
                    }
                }
                
                Op::Square { input } => {
                    // d/dx (x^2) = 2x
                    if let Some(input_tensor) = entry.saved_tensors.get(input) {
                        let two_x = input_tensor.mul_scalar(2.0)?;
                        let grad_input = grad_output.mul(&two_x)?;
                        let input_grad = self.get_or_create_grad(*input, input_tensor.shape(), device)?;
                        *input_grad = input_grad.add(&grad_input)?;
                    }
                }
                
                Op::Sum { input, input_shape } => {
                    // Gradient of sum: broadcast grad_output to input shape
                    if entry.saved_tensors.contains_key(input) {
                        let grad_val = grad_output.item()?;
                        let expanded = Tensor::from_vec(
                            vec![grad_val; input_shape.elem_count()],
                            input_shape.clone(),
                            device.clone()
                        )?;
                        let input_grad = self.get_or_create_grad(*input, input_shape, device)?;
                        *input_grad = input_grad.add(&expanded)?;
                    }
                }
                
                Op::Mean { input, input_shape } => {
                    // d/dx mean(x) = 1/n for each element
                    if entry.saved_tensors.contains_key(input) {
                        let n = input_shape.elem_count() as f32;
                        let grad_val = grad_output.item()? / n;
                        let expanded = Tensor::from_vec(
                            vec![grad_val; input_shape.elem_count()],
                            input_shape.clone(),
                            device.clone()
                        )?;
                        let input_grad = self.get_or_create_grad(*input, input_shape, device)?;
                        *input_grad = input_grad.add(&expanded)?;
                    }
                }
                
                Op::Transpose { input } => {
                    // Gradient of transpose is transpose of gradient
                    if let Some(input_tensor) = entry.saved_tensors.get(input) {
                        let grad_input = grad_output.transpose()?;
                        let input_grad = self.get_or_create_grad(*input, input_tensor.shape(), device)?;
                        *input_grad = input_grad.add(&grad_input)?;
                    }
                }
                
                Op::GELU { input } => {
                    if let Some(input_tensor) = entry.saved_tensors.get(input) {
                        let grad_input = crate::autograd_ops::BackwardOps::gelu_backward(&grad_output, input_tensor)?;
                        let input_grad = self.get_or_create_grad(*input, input_tensor.shape(), device)?;
                        *input_grad = input_grad.add(&grad_input)?;
                    }
                }
                
                Op::SiLU { input } => {
                    if let Some(input_tensor) = entry.saved_tensors.get(input) {
                        let grad_input = crate::autograd_ops::BackwardOps::silu_backward(&grad_output, input_tensor)?;
                        let input_grad = self.get_or_create_grad(*input, input_tensor.shape(), device)?;
                        *input_grad = input_grad.add(&grad_input)?;
                    }
                }
                
                Op::Tanh { input } => {
                    if let Some(input_tensor) = entry.saved_tensors.get(input) {
                        let grad_input = crate::autograd_ops::BackwardOps::tanh_backward(&grad_output, input_tensor)?;
                        let input_grad = self.get_or_create_grad(*input, input_tensor.shape(), device)?;
                        *input_grad = input_grad.add(&grad_input)?;
                    }
                }
                
                Op::BatchMatMul { lhs, rhs } => {
                    // Same as MatMul but for batched tensors
                    if let (Some(lhs_tensor), Some(rhs_tensor)) = 
                        (entry.saved_tensors.get(lhs), entry.saved_tensors.get(rhs)) {
                        
                        // For batch dimensions, transpose last two dims
                        let rhs_shape = rhs_tensor.shape().dims();
                        let n_dims = rhs_shape.len();
                        let mut perm: Vec<usize> = (0..n_dims).collect();
                        perm[n_dims - 2] = n_dims - 1;
                        perm[n_dims - 1] = n_dims - 2;
                        
                        // Gradient w.r.t lhs: grad @ rhs.transpose(-2, -1)
                        let rhs_t = rhs_tensor.permute(&perm)?;
                        let grad_lhs = grad_output.bmm(&rhs_t)?;
                        let lhs_grad = self.get_or_create_grad(*lhs, lhs_tensor.shape(), device)?;
                        *lhs_grad = lhs_grad.add(&grad_lhs)?;
                        
                        // Gradient w.r.t rhs: lhs.transpose(-2, -1) @ grad
                        let lhs_shape = lhs_tensor.shape().dims();
                        let n_dims = lhs_shape.len();
                        let mut perm: Vec<usize> = (0..n_dims).collect();
                        perm[n_dims - 2] = n_dims - 1;
                        perm[n_dims - 1] = n_dims - 2;
                        
                        let lhs_t = lhs_tensor.permute(&perm)?;
                        let grad_rhs = lhs_t.bmm(&grad_output)?;
                        let rhs_grad = self.get_or_create_grad(*rhs, rhs_tensor.shape(), device)?;
                        *rhs_grad = rhs_grad.add(&grad_rhs)?;
                    }
                }
                
                Op::Reshape { input, new_shape: _ } => {
                    // Reshape gradient back to input shape
                    if let Some(input_tensor) = entry.saved_tensors.get(input) {
                        let input_shape = input_tensor.shape().dims();
                        let grad_input = grad_output.reshape(input_shape)?;
                        let input_grad = self.get_or_create_grad(*input, input_tensor.shape(), device)?;
                        *input_grad = input_grad.add(&grad_input)?;
                    }
                }
                
                Op::Permute { input, dims } => {
                    // Inverse permutation for gradient
                    if let Some(input_tensor) = entry.saved_tensors.get(input) {
                        // Create inverse permutation
                        let mut inv_perm = vec![0usize; dims.len()];
                        for (i, &d) in dims.iter().enumerate() {
                            inv_perm[d] = i;
                        }
                        
                        let grad_input = grad_output.permute(&inv_perm)?;
                        let input_grad = self.get_or_create_grad(*input, input_tensor.shape(), device)?;
                        *input_grad = input_grad.add(&grad_input)?;
                    }
                }
                
                Op::AddBias { input, bias } => {
                    // Gradient flows unchanged to input
                    if let Some(input_tensor) = entry.saved_tensors.get(input) {
                        let input_grad = self.get_or_create_grad(*input, input_tensor.shape(), device)?;
                        *input_grad = input_grad.add(&grad_output)?;
                    }
                    
                    // Gradient for bias is sum over all batch dimensions
                    if let Some(bias_tensor) = entry.saved_tensors.get(bias) {
                        // Sum over all dimensions except the last one
                        let grad_shape = grad_output.shape().dims();
                        let mut grad_bias = grad_output.clone_result()?;
                        
                        // Sum over batch dimensions
                        for i in 0..grad_shape.len() - 1 {
                            grad_bias = grad_bias.sum_dim(0)?;
                        }
                        
                        let bias_grad = self.get_or_create_grad(*bias, bias_tensor.shape(), device)?;
                        *bias_grad = bias_grad.add(&grad_bias)?;
                    }
                }
                
                Op::SumDim { input, dim } => {
                    // Broadcast gradient back along the summed dimension
                    if let Some(input_tensor) = entry.saved_tensors.get(input) {
                        let input_shape = input_tensor.shape().dims();
                        
                        // For now, only support dim=0
                        if *dim == 0 {
                            let batch_size = input_shape[0];
                            let mut expanded_grad = grad_output.clone_result()?;
                            
                            // Repeat the gradient batch_size times
                            for _ in 1..batch_size {
                                expanded_grad = expanded_grad.add(&grad_output)?;
                            }
                            
                            // Reshape to match input
                            let grad_input = expanded_grad.unsqueeze(0)?;
                            let mut final_grad = grad_input.clone_result()?;
                            
                            // Replicate along batch dimension
                            for _ in 1..batch_size {
                                final_grad = final_grad.add(&grad_input)?;
                            }
                            
                            let input_grad = self.get_or_create_grad(*input, input_tensor.shape(), device)?;
                            *input_grad = input_grad.add(&final_grad)?;
                        }
                    }
                }
                
                // Backward for Conv2d, Linear, LayerNorm is provided in the main autograd engine
                _ => {}
            }
        }
        
        Ok(())
    }
    
    /// Get gradient for a tensor ID
    pub fn get_grad(&self, id: TensorId) -> Option<&Tensor> {
        self.gradients.get(&id)
    }
    
    /// Take gradient for a tensor ID (removes it)
    pub fn take_grad(&mut self, id: TensorId) -> Option<Tensor> {
        self.gradients.remove(&id)
    }
    
    /// Clear the tape and gradients
    pub fn clear(&mut self) {
        self.tape.clear();
        self.gradients.clear();
    }
}

// Thread-local autograd engine
thread_local! {
    pub static ENGINE: std::cell::RefCell<AutogradEngine> = std::cell::RefCell::new(AutogradEngine::new());
}

/// Record an operation in the current engine
pub fn record_op(output_id: TensorId, op: Op, inputs: Vec<(TensorId, Tensor)>) {
    ENGINE.with(|engine| {
        engine.borrow_mut().record_op(output_id, op, inputs);
    });
}

/// Perform backward pass from a tensor
pub fn backward(tensor_id: TensorId, shape: &Shape, device: &std::sync::Arc<cudarc::driver::CudaDevice>) -> Result<()> {
    ENGINE.with(|engine| {
        engine.borrow_mut().backward(tensor_id, shape, device)
    })
}

/// Get gradient for a tensor
pub fn get_grad(tensor_id: TensorId) -> Option<Tensor> {
    ENGINE.with(|engine| {
        engine.borrow().get_grad(tensor_id).cloned()
    })
}

/// Take gradient for a tensor (removes it)
pub fn take_grad(tensor_id: TensorId) -> Option<Tensor> {
    ENGINE.with(|engine| {
        engine.borrow_mut().take_grad(tensor_id)
    })
}

/// Clear the computation graph
pub fn clear_graph() {
    ENGINE.with(|engine| {
        engine.borrow_mut().clear();
    });
}
