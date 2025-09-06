/// Alternate autograd implementation that properly tracks tensors
use crate::{Tensor, Result, FlameError};
use std::rc::Rc;
use std::cell::RefCell;

/// Gradient function trait
pub trait GradFn {
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>>;
}

/// Addition backward
struct AddBackward {
    needs_grad: [bool; 2],
}

impl GradFn for AddBackward {
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        Ok(vec![
            if self.needs_grad[0] { Some(grad_output.clone()?) } else { None },
            if self.needs_grad[1] { Some(grad_output.clone()?) } else { None },
        ])
    }
}

/// Multiplication backward
struct MulBackward {
    lhs: Tensor,
    rhs: Tensor,
    needs_grad: [bool; 2],
}

impl GradFn for MulBackward {
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        Ok(vec![
            if self.needs_grad[0] { Some(grad_output.mul(&self.rhs)?) } else { None },
            if self.needs_grad[1] { Some(grad_output.mul(&self.lhs)?) } else { None },
        ])
    }
}

/// Mean backward
struct MeanBackward {
    input_shape: crate::Shape,
    needs_grad: bool,
}

impl GradFn for MeanBackward {
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if !self.needs_grad {
            return Ok(vec![None]);
        }
        
        let n = self.input_shape.elem_count() as f32;
        let grad_val = grad_output.item()? / n;
        
        let expanded = Tensor::from_vec(
            vec![grad_val; self.input_shape.elem_count()],
            self.input_shape.clone(),
            grad_output.device().clone()
        )?;
        
        Ok(vec![Some(expanded)])
    }
}

/// Gradient tape entry
struct TapeEntry {
    grad_fn: Box<dyn GradFn>,
    inputs: Vec<TensorData>,
    output: TensorData,
}

/// Shared tensor data with gradient accumulation
#[derive(Clone)]
pub struct TensorData {
    id: usize,
    data: Rc<RefCell<TensorDataInner>>,
}

struct TensorDataInner {
    tensor: Tensor,
    grad: Option<Tensor>,
    requires_grad: bool,
}

impl TensorData {
    pub fn new(tensor: Tensor, requires_grad: bool) -> Self {
        static NEXT_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        Self {
            id,
            data: Rc::new(RefCell::new(TensorDataInner {
                tensor,
                grad: None,
                requires_grad,
            })),
        }
    }
    
    pub fn tensor(&self) -> Tensor {
        self.data().borrow().tensor.clone().expect("tensor clone failed")
    }
    
    pub fn accumulate_grad(&self, grad: Tensor) -> Result<()> {
        let mut inner = self.data().borrow_mut();
        match &mut inner.grad {
            Some(existing) => {
                *existing = existing.add(&grad)?;
            }
            None => {
                inner.grad = Some(grad);
            }
        }
        Ok(())
    }
    
    pub fn grad(&self) -> Option<Tensor> {
        self.data().borrow().grad.as_ref().and_then(|g| g.clone().ok())
    }
    
    pub fn zero_grad(&self) {
        self.data().borrow_mut().grad = None;
    }
    
    pub fn requires_grad(&self) -> bool {
        self.data().borrow().requires_grad
    }
}

// Global gradient tape
thread_local! {
    static TAPE: RefCell<Vec<TapeEntry>> = RefCell::new(Vec::new());
}

/// Record an operation on the tape
fn record_op(grad_fn: Box<dyn GradFn>, inputs: Vec<TensorData>, output: TensorData) {
    TAPE.with(|tape| {
        tape.borrow_mut().push(TapeEntry {
            grad_fn,
            inputs,
            output,
        });
    });
}

/// Tracked tensor operations
pub mod tracked_ops {
    use super::*;
    
    pub fn add(lhs: &TensorData, rhs: &TensorData) -> Result<TensorData> {
        let result = lhs.tensor().add(&rhs.tensor())?;
        let output = TensorData::new(result, lhs.requires_grad() || rhs.requires_grad());
        
        if output.requires_grad() {
            record_op(
                Box::new(AddBackward {
                    needs_grad: [lhs.requires_grad(), rhs.requires_grad()],
                }),
                vec![lhs.clone(), rhs.clone()],
                output.clone(),
            );
        }
        
        Ok(output)
    }
    
    pub fn mul(lhs: &TensorData, rhs: &TensorData) -> Result<TensorData> {
        let result = lhs.tensor().mul(&rhs.tensor())?;
        let output = TensorData::new(result, lhs.requires_grad() || rhs.requires_grad());
        
        if output.requires_grad() {
            record_op(
                Box::new(MulBackward {
                    lhs: lhs.tensor(),
                    rhs: rhs.tensor(),
                    needs_grad: [lhs.requires_grad(), rhs.requires_grad()],
                }),
                vec![lhs.clone(), rhs.clone()],
                output.clone(),
            );
        }
        
        Ok(output)
    }
    
    pub fn mean(input: &TensorData) -> Result<TensorData> {
        let result = input.tensor().mean()?;
        let output = TensorData::new(result, input.requires_grad());
        
        if output.requires_grad() {
            record_op(
                Box::new(MeanBackward {
                    input_shape: input.tensor().shape().clone(),
                    needs_grad: input.requires_grad(),
                }),
                vec![input.clone()],
                output.clone(),
            );
        }
        
        Ok(output)
    }
}

/// Perform backward pass
pub fn backward(loss: &TensorData) -> Result<()> {
    if !loss.requires_grad() {
        return Err(FlameError::InvalidOperation(
            "backward() called on tensor that doesn't require grad".into()
        ));
    }
    
    if loss.tensor().shape().elem_count() != 1 {
        return Err(FlameError::InvalidOperation(
            "backward() can only be called on scalar tensors".into()
        ));
    }
    
    // Initialize gradient of loss as 1.0
    let ones = Tensor::from_vec(
        vec![1.0],
        loss.tensor().shape().clone(),
        loss.tensor().device().clone()
    )?;
    loss.accumulate_grad(ones)?;
    
    // Process tape in reverse order
    TAPE.with(|tape| {
        let tape_entries = tape.borrow();
        
        for entry in tape_entries.iter().rev() {
            if let Some(grad_output) = entry.output.grad() {
                let grads = entry.grad_fn.apply(&grad_output)?;
                
                for (input, grad) in entry.inputs.iter().zip(grads.iter()) {
                    if let Some(g) = grad {
                        input.accumulate_grad(g.clone()?)?;
                    }
                }
            }
        }
        
        Ok(())
    })
}

/// Clear the gradient tape
pub fn clear_tape() {
    TAPE.with(|tape| {
        tape.borrow_mut().clear();
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Shape, CudaDevice};
    
    #[test]
    fn test_simple_backward() -> Result<()> {
        let device = CudaDevice::new(0)?;
        
        // Create tensors
        let x = TensorData::new(
            Tensor::from_vec(vec![2.0, 3.0], Shape::from_dims(&[2]), device.clone())?,
            true
        );
        
        let y = TensorData::new(
            Tensor::from_vec(vec![4.0, 5.0], Shape::from_dims(&[2]), device)?,
            true
        );
        
        // z = x * y
        let z = tracked_ops::mul(&x, &y)?;
        
        // loss = mean(z)
        let loss = tracked_ops::mean(&z)?;
        
        // Backward
        backward(&loss)?;
        
        // Check gradients
        let x_grad = x.grad().unwrap();
        let y_grad = y.grad().unwrap();
        
        let x_grad_data = x_grad.to_vec()?;
        let y_grad_data = y_grad.to_vec()?;
        
        // d/dx mean(x*y) = y/n = [4,5]/2 = [2.0, 2.5]
        assert!((x_grad_data[0] - 2.0).abs() < 1e-5);
        assert!((x_grad_data[1] - 2.5).abs() < 1e-5);
        
        // d/dy mean(x*y) = x/n = [2,3]/2 = [1.0, 1.5]
        assert!((y_grad_data[0] - 1.0).abs() < 1e-5);
        assert!((y_grad_data[1] - 1.5).abs() < 1e-5);
        
        // Clear for next test
        clear_tape();
        
        Ok(())
    }
}
