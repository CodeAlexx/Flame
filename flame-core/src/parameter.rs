//! Trainable parameters with mutable updates

use crate::{Tensor, TensorId, Result, FlameError, Shape};
use std::sync::{Arc, Mutex};
use cudarc::driver::CudaDevice;

/// A trainable parameter that supports in-place updates
#[derive(Clone)]
pub struct Parameter {
    /// The parameter data (wrapped in mutex for mutability)
    data: Arc<Mutex<Tensor>>,
    /// Current gradient (if any)
    grad: Arc<Mutex<Option<Tensor>>>,
    /// Whether this parameter requires gradients
    requires_grad: bool,
    /// Unique ID for this parameter
    id: TensorId,
}

impl Parameter {
    /// Create a new parameter from a tensor
    pub fn new(tensor: Tensor) -> Self {
        let requires_grad = tensor.requires_grad;
        let id = tensor.id;
        Self {
            data: Arc::new(Mutex::new(tensor)),
            grad: Arc::new(Mutex::new(None)),
            requires_grad,
            id,
        }
    }
    
    /// Create a parameter with specific initialization
    pub fn randn(shape: Shape, mean: f32, std: f32, device: Arc<CudaDevice>) -> Result<Self> {
        let tensor = Tensor::randn(shape, mean, std, device)?.requires_grad_(true);
        Ok(Self::new(tensor))
    }
    
    /// Create a parameter initialized with zeros
    pub fn zeros(shape: Shape, device: Arc<CudaDevice>) -> Result<Self> {
        let tensor = Tensor::zeros(shape, device)?.requires_grad_(true);
        Ok(Self::new(tensor))
    }
    
    /// Get the parameter ID
    pub fn id(&self) -> TensorId {
        self.id
    }
    
    /// Get a clone of the current tensor value
    pub fn tensor(&self) -> Result<Tensor> {
        Ok(self.data.lock().map_err(|_| FlameError::Training("parameter data mutex poisoned".into()))?.clone())
    }
    
    /// Get a reference to the tensor (as_tensor compatibility)
    pub fn as_tensor(&self) -> Result<Tensor> {
        self.tensor()
    }
    
    /// Set the parameter data directly
    pub fn set_data(&self, tensor: Tensor) -> Result<()> {
        let mut data_lock = self.data.lock().map_err(|_| FlameError::Training("parameter data mutex poisoned".into()))?;
        *data_lock = tensor;
        Ok(())
    }
    
    /// Set gradient for this parameter
    pub fn set_grad(&self, grad: Tensor) -> Result<()> {
        let mut grad_lock = self.grad.lock().map_err(|_| FlameError::Training("parameter grad mutex poisoned".into()))?;
        *grad_lock = Some(grad);
        Ok(())
    }
    
    /// Get current gradient (if any)
    pub fn grad(&self) -> Option<Tensor> {
        if let Ok(grad_lock) = self.grad.lock() {
            grad_lock.as_ref().map(|g| g.clone())
        } else {
            None
        }
    }
    
    /// Clear gradient
    pub fn zero_grad(&self) {
        if let Ok(mut grad_lock) = self.grad.lock() {
            *grad_lock = None;
        }
    }
    
    /// Update parameter in-place with gradient descent
    /// param = param - learning_rate * grad
    pub fn update(&self, learning_rate: f32) -> Result<()> {
        let grad_lock = self.grad.lock().map_err(|_| FlameError::Training("parameter grad mutex poisoned".into()))?;
        if let Some(grad) = grad_lock.as_ref() {
            let mut data_lock = self.data.lock().map_err(|_| FlameError::Training("parameter data mutex poisoned".into()))?;
            
            // Compute update: param = param - lr * grad
            let update = grad.mul_scalar(learning_rate)?;
            let new_data = data_lock.sub(&update)?;
            
            // Replace data
            *data_lock = new_data;
        }
        Ok(())
    }
    
    /// Apply an arbitrary update tensor
    pub fn apply_update(&self, update: &Tensor) -> Result<()> {
        let mut data_lock = self.data.lock().map_err(|_| FlameError::Training("parameter data mutex poisoned".into()))?;
        let new_data = data_lock.sub(update)?;
        *data_lock = new_data;
        Ok(())
    }
    
    /// Get shape of the parameter
    pub fn shape(&self) -> Shape {
        if let Ok(lock) = self.data.lock() {
            lock.shape.clone()
        } else {
            Shape::from_dims(&[])
        }
    }
    
    /// Check if parameter requires grad
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }
    
    /// Set requires_grad flag
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
        if let Ok(mut data_lock) = self.data.lock() {
            data_lock.requires_grad = requires_grad;
        }
    }
}

/// Collection of parameters for a module
pub struct ParameterDict {
    params: std::collections::HashMap<String, Parameter>,
}

impl ParameterDict {
    pub fn new() -> Self {
        Self {
            params: std::collections::HashMap::new(),
        }
    }
    
    pub fn insert(&mut self, name: String, param: Parameter) {
        self.params.insert(name, param);
    }
    
    pub fn get(&self, name: &str) -> Option<&Parameter> {
        self.params.get(name)
    }
    
    pub fn parameters(&self) -> Vec<&Parameter> {
        self.params.values().collect()
    }
    
    pub fn named_parameters(&self) -> impl Iterator<Item = (&String, &Parameter)> {
        self.params.iter()
    }
}
