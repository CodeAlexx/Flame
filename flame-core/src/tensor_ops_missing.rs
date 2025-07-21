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
        // For now, implement using sum and division
        let sum = self.sum_dims(dims)?;
        
        // Calculate number of elements summed
        let mut n = 1;
        for &dim in dims {
            n *= self.shape.dims()[dim];
        }
        
        let result = sum.div_scalar(n as f32)?;
        
        if keepdim {
            Ok(result)
        } else {
            // Remove dimensions
            let mut new_shape = self.shape.dims().to_vec();
            for &dim in dims.iter().rev() {
                new_shape.remove(dim);
            }
            result.reshape(&new_shape)
        }
    }
    
    /// Division by scalar
    pub fn div_scalar(&self, scalar: f32) -> Result<Tensor> {
        self.mul_scalar(1.0 / scalar)
    }
}