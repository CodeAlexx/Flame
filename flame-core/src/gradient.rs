//! Gradient storage and management
//! 
//! This module provides a separate gradient storage system that avoids
//! borrow checker issues and provides a cleaner API.

use std::collections::HashMap;
use std::sync::Arc;
use crate::{Tensor, Shape, Result};
use crate::tensor::TensorId;
use cudarc::driver::CudaDevice;

/// Gradient storage - completely separate from tensors
pub struct GradientMap {
    gradients: HashMap<TensorId, Tensor>,
    device: Arc<CudaDevice>,
}

impl GradientMap {
    /// Create a new gradient map
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            gradients: HashMap::new(),
            device,
        }
    }
    
    /// Set gradient to ones (for loss tensor)
    pub fn set_ones(&mut self, id: TensorId, shape: Shape) -> Result<()> {
        let ones = Tensor::ones(shape, self.device.clone())?;
        self.gradients.insert(id, ones);
        Ok(())
    }
    
    /// Get gradient for a tensor
    pub fn get(&self, id: TensorId) -> Option<&Tensor> {
        self.gradients.get(&id)
    }
    
    /// Get mutable gradient for a tensor
    pub fn get_mut(&mut self, id: TensorId) -> Option<&mut Tensor> {
        self.gradients.get_mut(&id)
    }
    
    /// Insert or replace gradient
    pub fn insert(&mut self, id: TensorId, grad: Tensor) {
        self.gradients.insert(id, grad);
    }
    
    /// Check if gradient exists
    pub fn contains(&self, id: TensorId) -> bool {
        self.gradients.contains_key(&id)
    }
    
    /// Accumulate gradient (in-place GPU addition)
    pub fn accumulate(&mut self, id: TensorId, grad: Tensor) -> Result<()> {
        match self.gradients.get_mut(&id) {
            Some(existing) => {
                // In-place addition on GPU
                *existing = existing.add(&grad)?;
            }
            None => {
                self.gradients.insert(id, grad);
            }
        }
        Ok(())
    }
    
    /// Get or create gradient initialized to zeros
    pub fn get_or_create(&mut self, id: TensorId, shape: Shape) -> Result<&mut Tensor> {
        if !self.gradients.contains_key(&id) {
            let zeros = Tensor::zeros(shape, self.device.clone())?;
            self.gradients.insert(id, zeros);
        }
        Ok(self.gradients.get_mut(&id).unwrap())
    }
    
    /// Take gradient (remove from map)
    pub fn take(&mut self, id: TensorId) -> Option<Tensor> {
        self.gradients.remove(&id)
    }
    
    /// Clear all gradients
    pub fn clear(&mut self) {
        self.gradients.clear();
    }
    
    /// Get number of stored gradients
    pub fn len(&self) -> usize {
        self.gradients.len()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.gradients.is_empty()
    }
    
    /// Iterate over gradients
    pub fn iter(&self) -> impl Iterator<Item = (&TensorId, &Tensor)> {
        self.gradients.iter()
    }
}

/// Extension trait for gradient access
pub trait TensorGradExt {
    /// Get gradient for this tensor
    fn grad<'a>(&self, gradients: &'a GradientMap) -> Option<&'a Tensor>;
    
    /// Get mutable gradient for this tensor
    fn grad_mut<'a>(&self, gradients: &'a mut GradientMap) -> Option<&'a mut Tensor>;
    
    /// Take gradient for this tensor (removes from map)
    fn take_grad(&self, gradients: &mut GradientMap) -> Option<Tensor>;
    
    /// Check if gradient exists
    fn has_grad(&self, gradients: &GradientMap) -> bool;
}