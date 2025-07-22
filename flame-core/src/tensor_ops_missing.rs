//! Missing tensor operations implementation

use crate::{Tensor, Result, FlameError, Shape, Op, AutogradContext};
use std::sync::Arc;

impl Tensor {
    /// Power operation - raise tensor to a scalar power
    pub fn pow(&self, exponent: f32) -> Result<Tensor> {
        let data = self.to_vec()?;
        let result: Vec<f32> = data.iter()
            .map(|&x| x.powf(exponent))
            .collect();
        
        let mut output = Tensor::from_vec(result, self.shape.clone(), self.device.clone())?;
        
        // Autograd support
        if self.requires_grad {
            output.requires_grad = true;
            AutogradContext::record_op(
                output.id,
                Op::Square { input: self.id }, // Using Square for now
                vec![(self.id, self.clone()?)],
            );
        }
        
        Ok(output)
    }
    
    /// Sine function
    pub fn sin(&self) -> Result<Tensor> {
        let data = self.to_vec()?;
        let result: Vec<f32> = data.iter()
            .map(|&x| x.sin())
            .collect();
        
        let mut output = Tensor::from_vec(result, self.shape.clone(), self.device.clone())?;
        
        if self.requires_grad {
            output.requires_grad = true;
            // For sin, we need to save the input for cos(x) in backward
            AutogradContext::record_op(
                output.id,
                Op::Abs { input: self.id }, // Placeholder - need to add Sin op
                vec![(self.id, self.clone()?)],
            );
        }
        
        Ok(output)
    }
    
    /// Cosine function
    pub fn cos(&self) -> Result<Tensor> {
        let data = self.to_vec()?;
        let result: Vec<f32> = data.iter()
            .map(|&x| x.cos())
            .collect();
        
        let mut output = Tensor::from_vec(result, self.shape.clone(), self.device.clone())?;
        
        if self.requires_grad {
            output.requires_grad = true;
            // For cos, we need to save the input for -sin(x) in backward
            AutogradContext::record_op(
                output.id,
                Op::Abs { input: self.id }, // Placeholder - need to add Cos op
                vec![(self.id, self.clone()?)],
            );
        }
        
        Ok(output)
    }
    
    /// Standard deviation
    pub fn std(&self, dim: Option<&[usize]>, keepdim: bool) -> Result<Tensor> {
        let var = self.var(dim, keepdim)?;
        var.sqrt()
    }
    
    /// Variance
    pub fn var(&self, dim: Option<&[usize]>, keepdim: bool) -> Result<Tensor> {
        let mean = if let Some(dims) = dim {
            self.mean_dim(dims, keepdim)?
        } else {
            let mean_scalar = self.mean()?;
            // Broadcast mean to match input shape
            mean_scalar.broadcast_to(&self.shape)?
        };
        
        let diff = self.sub(&mean)?;
        let squared = diff.mul(&diff)?;
        
        if let Some(dims) = dim {
            squared.mean_dim(dims, keepdim)
        } else {
            squared.mean()
        }
    }
    
    /// Square root
    pub fn sqrt(&self) -> Result<Tensor> {
        let data = self.to_vec()?;
        let result: Vec<f32> = data.iter()
            .map(|&x| x.sqrt())
            .collect();
        
        Tensor::from_vec(result, self.shape.clone(), self.device.clone())
    }
    
    /// Mean along dimensions
    pub fn mean_dim(&self, dims: &[usize], keepdim: bool) -> Result<Tensor> {
        // Implement mean manually since sum_dims is actually computing mean incorrectly
        let data = self.to_vec()?;
        let input_shape = self.shape().dims();
        let ndims = input_shape.len();
        
        // Validate dimensions
        for &dim in dims {
            if dim >= ndims {
                return Err(FlameError::InvalidOperation(
                    format!("Dimension {} out of range for tensor with {} dimensions", dim, ndims)
                ));
            }
        }
        
        // Calculate output shape
        let mut output_shape = input_shape.to_vec();
        for &dim in dims {
            output_shape[dim] = 1;
        }
        
        // Calculate strides
        let mut strides = vec![1; ndims];
        for i in (0..ndims-1).rev() {
            strides[i] = strides[i+1] * input_shape[i+1];
        }
        
        // Total output elements
        let output_elems: usize = output_shape.iter().product();
        let mut output_data = vec![0.0f32; output_elems];
        
        // Count elements per output position
        let mut count = 1;
        for &dim in dims {
            count *= input_shape[dim];
        }
        
        // Iterate over output positions
        for out_idx in 0..output_elems {
            // Convert output index to multi-dimensional position
            let mut out_pos = vec![0; ndims];
            let mut idx = out_idx;
            for i in (0..ndims).rev() {
                out_pos[i] = idx % output_shape[i];
                idx /= output_shape[i];
            }
            
            // Sum over all positions that map to this output position
            let mut sum = 0.0f32;
            
            // Create iterator over dimensions to sum
            let mut pos = vec![0; ndims];
            loop {
                // Copy fixed dimensions from output position
                for i in 0..ndims {
                    if !dims.contains(&i) {
                        pos[i] = out_pos[i];
                    }
                }
                
                // Calculate linear index
                let lin_idx: usize = pos.iter().zip(&strides).map(|(p, s)| p * s).sum();
                sum += data[lin_idx];
                
                // Increment position over dimensions we're reducing
                let mut carry = true;
                for &dim in dims {
                    if carry {
                        pos[dim] += 1;
                        if pos[dim] < input_shape[dim] {
                            carry = false;
                        } else {
                            pos[dim] = 0;
                        }
                    }
                }
                
                if carry {
                    break;
                }
            }
            
            output_data[out_idx] = sum / count as f32;
        }
        
        let result = Tensor::from_vec(output_data, Shape::from_dims(&output_shape), self.device.clone())?;
        
        if keepdim {
            Ok(result)
        } else {
            // Remove dimensions with size 1
            let mut new_shape = Vec::new();
            for (i, &size) in output_shape.iter().enumerate() {
                if !dims.contains(&i) {
                    new_shape.push(size);
                }
            }
            result.reshape(&new_shape)
        }
    }
    
    /// Division by scalar
    pub fn div_scalar(&self, scalar: f32) -> Result<Tensor> {
        self.mul_scalar(1.0 / scalar)
    }
}