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
        
        println!("DEBUG Embedding forward: input shape = {:?}, batch_size = {}, seq_len = {}, embedding_dim = {}", 
                 input_shape, batch_size, seq_len, self.embedding_dim);
        
        // For now, implement as gather operation
        // Convert indices to one-hot and matmul with weight
        let indices = input.to_vec()?;
        println!("DEBUG Embedding: Got {} indices", indices.len());
        
        let mut embeddings = Vec::new();
        
        for (i, idx) in indices.iter().enumerate() {
            let idx = *idx as usize;
            if idx >= self.vocab_size {
                return Err(FlameError::InvalidOperation(
                    format!("Index {} out of vocabulary size {}", idx, self.vocab_size)
                ));
            }
            
            // Get embedding vector for this index
            let start = idx * self.embedding_dim;
            let end = start + self.embedding_dim;
            
            let weight_data = self.weight.to_vec()?;
            if i == 0 {
                println!("DEBUG Embedding: weight_data size = {}, extracting range {}..{}", 
                         weight_data.len(), start, end);
            }
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
        
        println!("DEBUG Embedding: Creating output tensor with shape {:?}, total elements: {}, embeddings.len() = {}", 
                 output_shape, total_elements, embeddings.len());
        
        // Verify we have the right amount of data
        if embeddings.len() != total_elements {
            return Err(FlameError::ShapeMismatch {
                expected: Shape::from_dims(&output_shape),
                got: Shape::from_dims(&[embeddings.len()]),
            });
        }
        
        // Create the tensor - use regular from_vec since the shape is what matters
        // The allocation will be handled internally
        let mut output = Tensor::from_vec(
            embeddings, 
            Shape::from_dims(&output_shape), 
            self.weight.device().clone()
        )?;
        
        println!("DEBUG Embedding: Created output tensor with shape {:?}", output.shape());
        
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