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
        let var = if let Some(dims) = dim {
            self.var(dims, false, keepdim)?  // Use the var() from tensor_ops_extended.rs
        } else {
            // For global std, compute variance over all dimensions
            let all_dims: Vec<usize> = (0..self.shape.dims().len()).collect();
            self.var(&all_dims, false, keepdim)?
        };
        var.sqrt()
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
    
    /// Flatten all dimensions into a single dimension
    pub fn flatten_all(&self) -> Result<Tensor> {
        let total_elements = self.shape.elem_count();
        self.reshape(&[total_elements])
    }
    
    /// Dropout operation - randomly zeros elements with probability p during training
    pub fn dropout(&self, p: f32) -> Result<Tensor> {
        if p <= 0.0 || p >= 1.0 {
            return Err(FlameError::InvalidOperation(format!("Dropout probability must be between 0 and 1, got {}", p)));
        }
        
        // During inference, dropout is identity
        // For now, we'll implement a simple version that always returns the input scaled
        // In a full implementation, this would check training mode and use random mask
        let scale = 1.0 / (1.0 - p);
        self.mul_scalar(scale)
    }
    
    /// Upsample using nearest neighbor interpolation
    pub fn upsample_nearest2d(&self, new_h: usize, new_w: usize) -> Result<Tensor> {
        // Assume input is [batch, channels, height, width]
        let shape = self.shape();
        if shape.rank() != 4 {
            return Err(FlameError::InvalidOperation(format!("Expected 4D tensor, got {}D", shape.rank())));
        }
        
        let batch = shape.dims()[0];
        let channels = shape.dims()[1];
        let old_h = shape.dims()[2];
        let old_w = shape.dims()[3];
        
        // Simple nearest neighbor upsampling implementation
        let data = self.to_vec()?;
        let mut output = vec![0.0f32; batch * channels * new_h * new_w];
        
        for b in 0..batch {
            for c in 0..channels {
                for h in 0..new_h {
                    for w in 0..new_w {
                        // Find nearest neighbor in original image
                        let src_h = (h * old_h) / new_h;
                        let src_w = (w * old_w) / new_w;
                        
                        let src_idx = ((b * channels + c) * old_h + src_h) * old_w + src_w;
                        let dst_idx = ((b * channels + c) * new_h + h) * new_w + w;
                        
                        output[dst_idx] = data[src_idx];
                    }
                }
            }
        }
        
        Tensor::from_vec(
            output,
            Shape::from_dims(&[batch, channels, new_h, new_w]),
            self.device.clone()
        )
    }
    
    /// Power operation with float exponent
    pub fn powf(&self, exponent: f32) -> Result<Tensor> {
        self.pow(exponent)
    }
    
    /// Variance along specific dimensions
    pub fn var_dim(&self, dims: &[usize], keepdim: bool) -> Result<Tensor> {
        self.var(dims, false, keepdim)  // Use unbiased=false by default
    }
    
    /// Element-wise maximum with a scalar
    pub fn maximum_scalar(&self, scalar: f32) -> Result<Tensor> {
        let data = self.to_vec()?;
        let result: Vec<f32> = data.iter()
            .map(|&x| x.max(scalar))
            .collect();
        
        Tensor::from_vec(result, self.shape.clone(), self.device.clone())
    }
    
    /// Element-wise minimum with a scalar
    pub fn minimum_scalar(&self, scalar: f32) -> Result<Tensor> {
        let data = self.to_vec()?;
        let result: Vec<f32> = data.iter()
            .map(|&x| x.min(scalar))
            .collect();
        
        Tensor::from_vec(result, self.shape.clone(), self.device.clone())
    }
    
    /// Get a single element from a 1D tensor
    pub fn get(&self, index: usize) -> Result<Tensor> {
        let dims = self.shape.dims();
        if dims.len() != 1 {
            return Err(FlameError::InvalidOperation(
                format!("get() only works on 1D tensors, got {}D tensor", dims.len())
            ));
        }
        
        if index >= dims[0] {
            return Err(FlameError::InvalidOperation(
                format!("Index {} out of bounds for tensor with size {}", index, dims[0])
            ));
        }
        
        // Extract single value and return as scalar tensor
        let data = self.to_vec()?;
        Tensor::from_vec(vec![data[index]], Shape::from_dims(&[]), self.device.clone())
    }
    
    /// Repeat tensor along specified dimensions
    pub fn repeat(&self, repeats: &[usize]) -> Result<Tensor> {
        let shape = self.shape();
        let dims = shape.dims();
        
        if repeats.len() != dims.len() {
            return Err(FlameError::InvalidOperation(
                format!("repeat dimensions mismatch: tensor has {} dims but got {} repeat values", 
                    dims.len(), repeats.len())
            ));
        }
        
        // Calculate new shape
        let mut new_dims = vec![];
        for (i, &dim) in dims.iter().enumerate() {
            new_dims.push(dim * repeats[i]);
        }
        
        // For now, use broadcast_to to repeat
        self.broadcast_to(&Shape::from_dims(&new_dims))
    }
}