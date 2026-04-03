//! Embedding layer for FLAME

use crate::tensor::TensorId;
use crate::{Error, Result, Shape, Tensor};
use cudarc::driver::CudaDevice;
use std::sync::Arc;

/// Embedding layer that maps indices to dense vectors
pub struct Embedding {
    /// Weight matrix [vocab_size, embedding_dim]
    pub weight: Tensor,
    _vocab_size: usize,
    _embedding_dim: usize,
}

impl Embedding {
    /// Create a new embedding layer
    pub fn new(vocab_size: usize, embedding_dim: usize, device: Arc<CudaDevice>) -> Result<Self> {
        let weight = Tensor::randn(
            Shape::from_dims(&[vocab_size, embedding_dim]),
            0.0,
            0.02,
            device,
        )?
        .requires_grad_(true);

        Ok(Self {
            weight,
            _vocab_size: vocab_size,
            _embedding_dim: embedding_dim,
        })
    }

    /// Forward pass: convert indices to embeddings
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Input: [batch, seq_len] indices
        // Weight: [vocab, dim]
        // Output: [batch, seq_len, dim]

        // index_select0 expects I32 indices and handles the gather efficiently
        // It uses the robust gather_rows kernel which supports BF16
        if input.dtype() == crate::DType::I32 {
            self.weight.index_select0(input)
        } else {
            let indices = input.to_dtype(crate::DType::I32)?;
            self.weight.index_select0(&indices)
        }
    }
}

/// Autograd operation for embedding
#[derive(Debug, Clone)]
pub struct EmbeddingOp {
    pub weight: TensorId,
    pub indices: TensorId,
}
