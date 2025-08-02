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
        
        // For now, implement as gather operation
        // Convert indices to one-hot and matmul with weight
        let indices = input.to_vec()?;
        let mut embeddings = Vec::new();
        
        for idx in indices {
            let idx = idx as usize;
            if idx >= self.vocab_size {
                return Err(FlameError::InvalidOperation(
                    format!("Index {} out of vocabulary size {}", idx, self.vocab_size)
                ));
            }
            
            // Get embedding vector for this index
            let start = idx * self.embedding_dim;
            let end = start + self.embedding_dim;
            
            let weight_data = self.weight.to_vec()?;
            embeddings.extend_from_slice(&weight_data[start..end]);
        }
        
        // Reshape to [batch_size, seq_len, embedding_dim]
        let output_shape = if input_shape.len() > 1 {
            vec![batch_size, seq_len, self.embedding_dim]
        } else {
            vec![batch_size, self.embedding_dim]
        };
        
        // Check if the output size would be problematic
        let total_elements: usize = output_shape.iter().product();
        if is_problematic_size(total_elements) {
            // For problematic sizes, pad the embeddings vector
            let padded_size = ((total_elements + 1023) / 1024) * 1024; // Round up to next 1024
            embeddings.resize(padded_size, 0.0);
            
            // Create tensor with padded size, then slice back
            let padded_shape = if output_shape.len() > 1 {
                vec![output_shape[0], output_shape[1], padded_size / (output_shape[0] * output_shape[1])]
            } else {
                vec![padded_size]
            };
            
            let padded_tensor = Tensor::from_vec(
                embeddings, 
                Shape::from_dims(&padded_shape), 
                self.weight.device().clone()
            )?;
            
            // Reshape back to original size
            return padded_tensor.narrow(output_shape.len() - 1, 0, output_shape[output_shape.len() - 1])
                .and_then(|t| t.reshape(&output_shape));
        }
        
        let mut output = Tensor::from_vec(
            embeddings, 
            Shape::from_dims(&output_shape), 
            self.weight.device().clone()
        )?;
        
        // Record operation for autograd if needed
        if self.weight.requires_grad() {
            output.requires_grad = true;
            
            AutogradContext::record_op(
                output.id(),
                Op::Embedding {
                    weight: self.weight.id(),
                    indices: input.id(),
                },
                vec![
                    (self.weight.id(), self.weight.clone()?),
                    (input.id(), input.clone()?)
                ]
            );
        }
        
        Ok(output)
    }
}

/// Autograd operation for embedding
#[derive(Debug, Clone)]
pub struct EmbeddingOp {
    pub weight: TensorId,
    pub indices: TensorId,
}