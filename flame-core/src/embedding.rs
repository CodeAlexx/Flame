//! Embedding layer for FLAME

use crate::{Tensor, Shape, Result, FlameError};
use crate::tensor::TensorId;
use crate::autograd::{AutogradContext, Op};
use crate::cuda_memory_alignment::is_problematic_size;
use std::sync::Arc;
use cudarc::driver::CudaDevice;

/// Embedding layer that maps indices to dense vectors
pub struct Embedding {
    /// Weight matrix [vocab_size, embedding_dim]
    pub weight: Tensor,
    vocab_size: usize,
    embedding_dim: usize,
}

impl Embedding {
    /// Create a new embedding layer
    pub fn new(vocab_size: usize, embedding_dim: usize, device: Arc<CudaDevice>) -> Result<Self> {
        let weight = Tensor::randn(
            Shape::from_dims(&[vocab_size, embedding_dim]), 
            0.0, 
            0.02, 
            device
        )?.requires_grad_(true);
        
        Ok(Self {
            weight,
            vocab_size,
            embedding_dim,
        })
    }
    
    /// Forward pass: convert indices to embeddings
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Input should be integer indices
        let input_shape = input.shape().dims();
        let batch_size = input_shape[0];
        let seq_len = if input_shape.len() > 1 { input_shape[1] } else { 1 };
        
        // No efficient CUDA gather implemented here; signal not implemented for this backend
        Err(FlameError::InvalidOperation("Embedding gather not implemented on this backend".into()))
    }
}

/// Autograd operation for embedding
#[derive(Debug, Clone)]
pub struct EmbeddingOp {
    pub weight: TensorId,
    pub indices: TensorId,
}
