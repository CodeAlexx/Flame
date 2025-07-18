use crate::{Tensor, Result};
use crate::tensor::TensorId;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

/// Operation types in the computation graph
#[derive(Debug, Clone)]
pub enum Op {
    Add { lhs: TensorId, rhs: TensorId },
    Sub { lhs: TensorId, rhs: TensorId },
    Mul { lhs: TensorId, rhs: TensorId },
    MulScalar { input: TensorId, scalar: f32 },
    MatMul { lhs: TensorId, rhs: TensorId },
    ReLU { input: TensorId },
    Square { input: TensorId },
    Sum { input: TensorId },
    Mean { input: TensorId },
    Transpose { input: TensorId },
}

/// Node in the computation graph
#[derive(Debug)]
pub struct Node {
    pub tensor_id: TensorId,
    pub op: Option<Op>,
    pub shape: crate::Shape,
    pub requires_grad: bool,
}

/// Computation graph for automatic differentiation
pub struct ComputationGraph {
    nodes: HashMap<TensorId, Node>,
    next_id: TensorId,
    tape: Vec<(TensorId, Op)>, // Records operations in order
}

impl ComputationGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: TensorId(0),
            tape: Vec::new(),
        }
    }
    
    /// Register a new tensor in the graph
    pub fn register_tensor(&mut self, tensor: &Tensor, op: Option<Op>) -> TensorId {
        let id = self.next_id;
        self.next_id = TensorId(self.next_id.0 + 1);
        
        let node = Node {
            tensor_id: id,
            op: op.clone(),
            shape: tensor.shape.clone(),
            requires_grad: tensor.requires_grad,
        };
        
        self.nodes.insert(id, node);
        
        if let Some(op) = op {
            self.tape.push((id, op));
        }
        
        id
    }
    
    /// Compute gradients using backpropagation
    pub fn backward(&self, loss_id: TensorId, tensors: &HashMap<TensorId, Arc<Mutex<Tensor>>>) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients: HashMap<TensorId, Tensor> = HashMap::new();
        
        // Initialize gradient of loss w.r.t. itself as 1.0
        if let Some(loss_tensor) = tensors.get(&loss_id) {
            let loss_tensor = loss_tensor.lock().unwrap();
            let ones = Tensor::from_vec(
                vec![1.0f32; loss_tensor.shape.elem_count()],
                loss_tensor.shape.clone(),
                loss_tensor.device.clone()
            )?;
            gradients.insert(loss_id, ones);
        }
        
        // Traverse computation graph in reverse order
        for (output_id, op) in self.tape.iter().rev() {
            if let Some(grad_output) = gradients.get(output_id) {
                let grad_output = grad_output.clone()?;
                match op {
                    Op::Add { lhs, rhs } => {
                        // Gradient of add: grad flows through unchanged
                        gradients.entry(*lhs).and_modify(|g| {
                            *g = g.add(&grad_output).unwrap();
                        }).or_insert(grad_output.clone()?);
                        
                        gradients.entry(*rhs).and_modify(|g| {
                            *g = g.add(&grad_output).unwrap();
                        }).or_insert(grad_output.clone()?);
                    },
                    
                    Op::Sub { lhs, rhs } => {
                        // d/dx (x - y) = 1, d/dy (x - y) = -1
                        gradients.entry(*lhs).and_modify(|g| {
                            *g = g.add(&grad_output).unwrap();
                        }).or_insert(grad_output.clone()?);
                        
                        let neg_grad = grad_output.mul_scalar(-1.0)?;
                        gradients.entry(*rhs).and_modify(|g| {
                            *g = g.add(&neg_grad).unwrap();
                        }).or_insert(neg_grad);
                    },
                    
                    Op::Mul { lhs, rhs } => {
                        // d/dx (x * y) = y, d/dy (x * y) = x
                        if let (Some(lhs_tensor), Some(rhs_tensor)) = (tensors.get(lhs), tensors.get(rhs)) {
                            let lhs_tensor = lhs_tensor.lock().unwrap();
                            let rhs_tensor = rhs_tensor.lock().unwrap();
                            
                            let grad_lhs = grad_output.mul(&*rhs_tensor)?;
                            gradients.entry(*lhs).and_modify(|g| {
                                *g = g.add(&grad_lhs).unwrap();
                            }).or_insert(grad_lhs);
                            
                            let grad_rhs = grad_output.mul(&*lhs_tensor)?;
                            gradients.entry(*rhs).and_modify(|g| {
                                *g = g.add(&grad_rhs).unwrap();
                            }).or_insert(grad_rhs);
                        }
                    },
                    
                    Op::MulScalar { input, scalar } => {
                        // d/dx (s * x) = s
                        let grad_input = grad_output.mul_scalar(*scalar)?;
                        gradients.entry(*input).and_modify(|g| {
                            *g = g.add(&grad_input).unwrap();
                        }).or_insert(grad_input);
                    },
                    
                    Op::MatMul { lhs, rhs } => {
                        // d/dA (A @ B) = grad @ B^T
                        // d/dB (A @ B) = A^T @ grad
                        if let (Some(lhs_tensor), Some(rhs_tensor)) = (tensors.get(lhs), tensors.get(rhs)) {
                            let lhs_tensor = lhs_tensor.lock().unwrap();
                            let rhs_tensor = rhs_tensor.lock().unwrap();
                            
                            // Gradient w.r.t. lhs
                            let rhs_t = rhs_tensor.transpose()?;
                            let grad_lhs = grad_output.matmul(&rhs_t)?;
                            gradients.entry(*lhs).and_modify(|g| {
                                *g = g.add(&grad_lhs).unwrap();
                            }).or_insert(grad_lhs);
                            
                            // Gradient w.r.t. rhs
                            let lhs_t = lhs_tensor.transpose()?;
                            let grad_rhs = lhs_t.matmul(&grad_output)?;
                            gradients.entry(*rhs).and_modify(|g| {
                                *g = g.add(&grad_rhs).unwrap();
                            }).or_insert(grad_rhs);
                        }
                    },
                    
                    Op::ReLU { input } => {
                        // d/dx ReLU(x) = 1 if x > 0, else 0
                        if let Some(input_tensor) = tensors.get(input) {
                            let input_tensor = input_tensor.lock().unwrap();
                            let input_data = input_tensor.to_vec()?;
                            let grad_data = grad_output.to_vec()?;
                            
                            let mut result = vec![0.0f32; grad_data.len()];
                            for i in 0..result.len() {
                                result[i] = if input_data[i] > 0.0 { grad_data[i] } else { 0.0 };
                            }
                            
                            let grad_input = Tensor::from_vec(
                                result,
                                grad_output.shape.clone(),
                                grad_output.device.clone()
                            )?;
                            
                            gradients.entry(*input).and_modify(|g| {
                                *g = g.add(&grad_input).unwrap();
                            }).or_insert(grad_input);
                        }
                    },
                    
                    Op::Square { input } => {
                        // d/dx (x^2) = 2x
                        if let Some(input_tensor) = tensors.get(input) {
                            let input_tensor = input_tensor.lock().unwrap();
                            let two_x = input_tensor.mul_scalar(2.0)?;
                            let grad_input = grad_output.mul(&two_x)?;
                            
                            gradients.entry(*input).and_modify(|g| {
                                *g = g.add(&grad_input).unwrap();
                            }).or_insert(grad_input);
                        }
                    },
                    
                    Op::Sum { input } => {
                        // Gradient of sum: broadcast grad_output to input shape
                        if let Some(node) = self.nodes.get(input) {
                            let expanded = Tensor::from_vec(
                                vec![grad_output.to_vec()?[0]; node.shape.elem_count()],
                                node.shape.clone(),
                                grad_output.device.clone()
                            )?;
                            
                            gradients.entry(*input).and_modify(|g| {
                                *g = g.add(&expanded).unwrap();
                            }).or_insert(expanded);
                        }
                    },
                    
                    Op::Mean { input } => {
                        // d/dx mean(x) = 1/n for each element
                        if let Some(node) = self.nodes.get(input) {
                            let n = node.shape.elem_count() as f32;
                            let grad_val = grad_output.to_vec()?[0] / n;
                            
                            let expanded = Tensor::from_vec(
                                vec![grad_val; node.shape.elem_count()],
                                node.shape.clone(),
                                grad_output.device.clone()
                            )?;
                            
                            gradients.entry(*input).and_modify(|g| {
                                *g = g.add(&expanded).unwrap();
                            }).or_insert(expanded);
                        }
                    },
                    
                    Op::Transpose { input } => {
                        // Gradient of transpose is transpose of gradient
                        let grad_input = grad_output.transpose()?;
                        gradients.entry(*input).and_modify(|g| {
                            *g = g.add(&grad_input).unwrap();
                        }).or_insert(grad_input);
                    },
                }
            }
        }
        
        Ok(gradients)
    }
}

// Global computation graph (thread-local for simplicity)
thread_local! {
    pub static GRAPH: Arc<Mutex<ComputationGraph>> = Arc::new(Mutex::new(ComputationGraph::new()));
}