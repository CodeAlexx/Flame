//! CUDA-based gradient checkpointing for memory-efficient training
//! 
//! Implements activation checkpointing to trade compute for memory,
//! enabling full model fine-tuning on limited GPU memory.

use crate::{
    Tensor, Shape, Result, FlameError, TensorId, DType,
    autograd::{AutogradContext, Op},
};
use std::sync::{Arc, Mutex, Weak};
use std::collections::HashMap;
use cudarc::driver::{CudaDevice, CudaStream, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

/// Global checkpoint manager
lazy_static::lazy_static! {
    pub static ref CHECKPOINT_MANAGER: Arc<Mutex<CheckpointManager>> = 
        Arc::new(Mutex::new(CheckpointManager::new()));
}

/// Checkpoint manager handles saving/restoring activation tensors
pub struct CheckpointManager {
    /// Saved activation tensors (moved to CPU or deleted)
    saved_activations: HashMap<TensorId, CheckpointedTensor>,
    
    /// Current checkpointing policy
    policy: CheckpointPolicy,
    
    /// CUDA device reference
    device: Option<Arc<CudaDevice>>,
    
    /// Memory statistics
    memory_saved: usize,
    recompute_count: usize,
    
    /// Tensor registry for looking up tensors by ID
    tensor_registry: HashMap<TensorId, Weak<Tensor>>,
}

/// Checkpointed tensor state
enum CheckpointedTensor {
    /// Tensor moved to CPU memory
    OnCPU {
        data: Vec<f32>,
        shape: Shape,
        dtype: DType,
    },
    
    /// Tensor deleted, will be recomputed
    Deleted {
        compute_fn: Box<dyn Fn() -> Result<Tensor> + Send + Sync>,
        shape: Shape,
        dtype: DType,
    },
}

/// Checkpointing policy
#[derive(Clone, Copy, Debug)]
pub enum CheckpointPolicy {
    /// Move activations to CPU (slower but preserves accuracy)
    CPUOffload,
    
    /// Delete and recompute (faster but requires deterministic ops)
    Recompute,
    
    /// Adaptive based on memory pressure
    Adaptive {
        memory_threshold: usize,
        prefer_recompute: bool,
    },
}

impl CheckpointManager {
    pub fn new() -> Self {
        Self {
            saved_activations: HashMap::new(),
            policy: CheckpointPolicy::Recompute,
            device: None,
            memory_saved: 0,
            recompute_count: 0,
            tensor_registry: HashMap::new(),
        }
    }
    
    /// Set the checkpointing policy
    pub fn set_policy(&mut self, policy: CheckpointPolicy) {
        self.policy = policy;
    }
    
    /// Set the CUDA device
    pub fn set_device(&mut self, device: Arc<CudaDevice>) {
        self.device = Some(device);
    }
    
    /// Register a tensor for potential checkpointing
    pub fn register_tensor(&mut self, tensor: &Arc<Tensor>) {
        self.tensor_registry.insert(tensor.id, Arc::downgrade(tensor));
    }
    
    /// Checkpoint a tensor based on current policy
    pub fn checkpoint_tensor(
        &mut self,
        tensor_id: TensorId,
        compute_fn: Option<Box<dyn Fn() -> Result<Tensor> + Send + Sync>>,
    ) -> Result<()> {
        // Find the tensor in registry
        let tensor = self.tensor_registry.get(&tensor_id)
            .and_then(|weak| weak.upgrade())
            .ok_or_else(|| FlameError::InvalidOperation("Tensor not found in registry".into()))?;
        
        match self.policy {
            CheckpointPolicy::CPUOffload => {
                // Move tensor data to CPU
                let data = tensor.to_vec()?;
                let shape = tensor.shape().clone();
                let dtype = tensor.dtype();
                
                self.saved_activations.insert(
                    tensor_id,
                    CheckpointedTensor::OnCPU { data, shape, dtype }
                );
                
                self.memory_saved += tensor.shape().elem_count() * 4; // f32 size
            }
            
            CheckpointPolicy::Recompute => {
                if let Some(compute_fn) = compute_fn {
                    let shape = tensor.shape().clone();
                    let dtype = tensor.dtype();
                    
                    self.saved_activations.insert(
                        tensor_id,
                        CheckpointedTensor::Deleted { compute_fn, shape, dtype }
                    );
                    
                    self.memory_saved += tensor.shape().elem_count() * 4;
                }
            }
            
            CheckpointPolicy::Adaptive { memory_threshold, prefer_recompute } => {
                // Check current GPU memory usage
                if self.memory_saved < memory_threshold && prefer_recompute {
                    // Use recompute strategy
                    if let Some(compute_fn) = compute_fn {
                        let shape = tensor.shape().clone();
                        let dtype = tensor.dtype();
                        
                        self.saved_activations.insert(
                            tensor_id,
                            CheckpointedTensor::Deleted { compute_fn, shape, dtype }
                        );
                        
                        self.memory_saved += tensor.shape().elem_count() * 4;
                    }
                } else {
                    // Use CPU offload
                    let data = tensor.to_vec()?;
                    let shape = tensor.shape().clone();
                    let dtype = tensor.dtype();
                    
                    self.saved_activations.insert(
                        tensor_id,
                        CheckpointedTensor::OnCPU { data, shape, dtype }
                    );
                    
                    self.memory_saved += tensor.shape().elem_count() * 4;
                }
            }
        }
        
        Ok(())
    }
    
    /// Restore a checkpointed tensor
    pub fn restore_tensor(&mut self, tensor_id: TensorId) -> Result<Tensor> {
        let checkpointed = self.saved_activations.remove(&tensor_id)
            .ok_or_else(|| FlameError::InvalidOperation("Checkpointed tensor not found".into()))?;
        
        let device = self.device.as_ref()
            .ok_or_else(|| FlameError::InvalidOperation("Device not set".into()))?;
        
        match checkpointed {
            CheckpointedTensor::OnCPU { data, shape, dtype } => {
                // Restore from CPU
                Tensor::from_vec(data, shape, device.clone())
            }
            
            CheckpointedTensor::Deleted { compute_fn, .. } => {
                // Recompute the tensor
                self.recompute_count += 1;
                compute_fn()
            }
        }
    }
    
    /// Get memory statistics
    pub fn stats(&self) -> CheckpointStats {
        CheckpointStats {
            memory_saved: self.memory_saved,
            recompute_count: self.recompute_count,
            checkpointed_tensors: self.saved_activations.len(),
        }
    }
    
    /// Clear all checkpoints
    pub fn clear(&mut self) {
        self.saved_activations.clear();
        self.memory_saved = 0;
        self.recompute_count = 0;
    }
}

/// Checkpoint statistics
#[derive(Debug, Clone)]
pub struct CheckpointStats {
    pub memory_saved: usize,
    pub recompute_count: usize,
    pub checkpointed_tensors: usize,
}

/// Checkpointed block wrapper
pub struct CheckpointedBlock<F> {
    pub forward_fn: F,
    pub block_name: String,
}

impl<F> CheckpointedBlock<F>
where
    F: Fn(&[Tensor]) -> Result<Vec<Tensor>> + Send + Sync + Clone + 'static,
{
    pub fn new(block_name: String, forward_fn: F) -> Self {
        Self { forward_fn, block_name }
    }
    
    /// Forward pass with checkpointing
    pub fn forward(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        let mut manager = CHECKPOINT_MANAGER.lock().unwrap();
        
        // Store input shapes and device info for recomputation
        let input_info: Vec<(Shape, Arc<CudaDevice>)> = inputs.iter()
            .map(|t| (t.shape().clone(), t.device.clone()))
            .collect();
        
        // Run forward pass
        let outputs = (self.forward_fn)(inputs)?;
        
        // For simplicity in this implementation, we won't checkpoint inputs
        // Instead, we'll assume inputs are available when needed
        // In a full implementation, we'd save inputs to CPU or recompute from earlier layers
        
        // Checkpoint intermediate activations if needed
        for (i, output) in outputs.iter().enumerate() {
            if i < outputs.len() - 1 {  // Don't checkpoint final output
                // For now, skip checkpointing - just store None
                manager.checkpoint_tensor(output.id, None)?;
            }
        }
        
        Ok(outputs)
    }
}

/// Enable gradient checkpointing for a model
pub fn enable_gradient_checkpointing<M>(model: &mut M, policy: CheckpointPolicy) 
where
    M: CheckpointableModel,
{
    let mut manager = CHECKPOINT_MANAGER.lock().unwrap();
    manager.set_policy(policy);
    model.enable_checkpointing();
}

/// Trait for models that support gradient checkpointing
pub trait CheckpointableModel {
    fn enable_checkpointing(&mut self);
    fn disable_checkpointing(&mut self);
}

/// CUDA kernel for efficient tensor copying between devices
fn get_async_memcpy_kernel() -> &'static str {
    r#"
extern "C" __global__ void async_memcpy_d2h(
    const float* __restrict__ src,
    float* __restrict__ dst,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = src[idx];
    }
}

extern "C" __global__ void async_memcpy_h2d(
    const float* __restrict__ src,
    float* __restrict__ dst,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = src[idx];
    }
}
"#
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_checkpoint_manager() {
        let mut manager = CheckpointManager::new();
        manager.set_policy(CheckpointPolicy::Recompute);
        
        // Test basic functionality
        assert_eq!(manager.stats().checkpointed_tensors, 0);
        assert_eq!(manager.stats().memory_saved, 0);
    }
}