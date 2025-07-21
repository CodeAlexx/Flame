//! Debug version of autograd with detailed logging

use crate::{Tensor, Result, FlameError, Shape};
use crate::tensor::TensorId;
use crate::gradient::GradientMap;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use cudarc::driver::CudaDevice;

lazy_static::lazy_static! {
    static ref AUTOGRAD_CONTEXT: Mutex<AutogradContextInner> = Mutex::new(AutogradContextInner::new());
}

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
    Sum { input: TensorId, input_shape: Shape },
    Mean { input: TensorId, input_shape: Shape },
}

struct TapeEntry {
    output_id: TensorId,
    op: Op,
    saved_tensors: HashMap<TensorId, Tensor>,
}

struct AutogradContextInner {
    tape: Vec<TapeEntry>,
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
            println!("AUTOGRAD: Recording op {:?} -> tensor {:?}", entry.op, entry.output_id);
            self.tape.push(entry);
        }
    }
    
    fn clear(&mut self) {
        self.tape.clear();
    }
}

pub struct AutogradContext;

impl AutogradContext {
    pub fn record_op(
        output_id: TensorId,
        op: Op,
        saved_tensors: Vec<(TensorId, Tensor)>,
    ) {
        println!("AUTOGRAD: record_op called for {:?}", op);
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
    
    pub fn backward(loss: &Tensor) -> Result<GradientMap> {
        println!("AUTOGRAD: Starting backward pass");
        
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
        let mut gradients = GradientMap::new(device.clone());
        gradients.set_ones(loss.id, loss.shape.clone())?;
        
        println!("AUTOGRAD: Initialized gradients for loss tensor");
        
        {
            let mut ctx = AUTOGRAD_CONTEXT.lock().unwrap();
            println!("AUTOGRAD: Processing {} tape entries", ctx.tape.len());
            
            let prev_enabled = ctx.enabled;
            ctx.enabled = false;
            
            for (i, entry) in ctx.tape.iter().rev().enumerate() {
                println!("AUTOGRAD: Processing entry {} (op: {:?})", i, entry.op);
                
                if let Some(output_grad) = gradients.get(entry.output_id) {
                    println!("AUTOGRAD: Found gradient for output {:?}", entry.output_id);
                    let output_grad = output_grad.clone()?;
                    
                    // Compute input gradients
                    let input_grads = match &entry.op {
                        Op::Add { lhs, rhs } => {
                            vec![
                                (*lhs, output_grad.clone()?),
                                (*rhs, output_grad.clone()?),
                            ]
                        }
                        Op::Mul { lhs, rhs } => {
                            let lhs_tensor = &entry.saved_tensors[lhs];
                            let rhs_tensor = &entry.saved_tensors[rhs];
                            vec![
                                (*lhs, output_grad.mul(rhs_tensor)?),
                                (*rhs, output_grad.mul(lhs_tensor)?),
                            ]
                        }
                        Op::Sum { input, .. } => {
                            let input_shape = &entry.saved_tensors[input].shape;
                            let expanded = output_grad.broadcast_to(input_shape)?;
                            vec![(*input, expanded)]
                        }
                        _ => {
                            println!("AUTOGRAD: Op {:?} not implemented!", entry.op);
                            vec![]
                        }
                    };
                    
                    // Accumulate gradients
                    for (tensor_id, grad) in input_grads {
                        println!("AUTOGRAD: Accumulating gradient for tensor {:?}", tensor_id);
                        gradients.accumulate(tensor_id, grad)?;
                    }
                }
            }
            
            ctx.enabled = prev_enabled;
            ctx.tape.clear();
            println!("AUTOGRAD: Cleared tape");
        }
        
        println!("AUTOGRAD: Backward pass complete, returning {} gradients", gradients.len());
        Ok(gradients)
    }
}