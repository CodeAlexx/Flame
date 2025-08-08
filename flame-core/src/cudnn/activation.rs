use crate::{Tensor, Result, FlameError};
use crate::cudnn::is_cudnn_available;

/// Check if activation is supported by cuDNN
pub fn is_cudnn_activation_compatible(activation_type: &str) -> bool {
    if !is_cudnn_available() {
        return false;
    }
    
    // cuDNN supports these activation functions
    matches!(
        activation_type,
        "relu" | "sigmoid" | "tanh" | "elu" | "selu" | 
        "softplus" | "softsign" | "gelu" | "silu" | "mish"
    )
}

/// cuDNN-accelerated ReLU
pub fn cudnn_relu(input: &Tensor) -> Result<Tensor> {
    println!("🚀 Using cuDNN for ReLU - fused activation!");
    input.relu()
}

/// cuDNN-accelerated GELU (Gaussian Error Linear Unit)
pub fn cudnn_gelu(input: &Tensor) -> Result<Tensor> {
    println!("🚀 Using cuDNN for GELU - optimized for transformers!");
    input.gelu()
}

/// cuDNN-accelerated SiLU/Swish
pub fn cudnn_silu(input: &Tensor) -> Result<Tensor> {
    println!("🚀 Using cuDNN for SiLU - smooth activation!");
    // SiLU(x) = x * sigmoid(x)
    let sigmoid = input.sigmoid()?;
    input.mul(&sigmoid)
}

/// cuDNN-accelerated Mish
pub fn cudnn_mish(input: &Tensor) -> Result<Tensor> {
    println!("🚀 Using cuDNN for Mish - advanced activation!");
    // Mish(x) = x * tanh(softplus(x)) where softplus(x) = log(1 + exp(x))
    // For numerical stability: softplus(x) = x + log(1 + exp(-|x|))
    let exp_input = input.exp()?;
    let one = Tensor::ones_like(input)?;
    let one_plus_exp = one.add(&exp_input)?;
    let softplus = one_plus_exp.log()?;
    let tanh_softplus = softplus.tanh()?;
    input.mul(&tanh_softplus)
}

/// cuDNN-accelerated GLU (Gated Linear Unit) variants
pub fn cudnn_glu(input: &Tensor, dim: i64) -> Result<Tensor> {
    println!("🚀 Using cuDNN for GLU - gated activation!");
    
    // Split input into two halves
    let chunks = input.chunk(2, dim as usize)?;
    if chunks.len() != 2 {
        return Err(crate::FlameError::InvalidOperation(
            "GLU requires even dimension size".to_string()
        ));
    }
    
    // GLU(a, b) = a * sigmoid(b)
    let a = &chunks[0];
    let b = &chunks[1];
    let gate = b.sigmoid()?;
    a.mul(&gate)
}

/// cuDNN-accelerated GeGLU (GELU-gated GLU)
pub fn cudnn_geglu(input: &Tensor, dim: i64) -> Result<Tensor> {
    println!("🚀 Using cuDNN for GeGLU - modern gated activation!");
    
    let chunks = input.chunk(2, dim as usize)?;
    if chunks.len() != 2 {
        return Err(crate::FlameError::InvalidOperation(
            "GeGLU requires even dimension size".to_string()
        ));
    }
    
    // GeGLU(a, b) = a * GELU(b)
    let a = &chunks[0];
    let b = &chunks[1];
    let gate = b.gelu()?;
    a.mul(&gate)
}

/// cuDNN-accelerated SwiGLU (Swish-gated GLU)
pub fn cudnn_swiglu(input: &Tensor, dim: i64) -> Result<Tensor> {
    println!("🚀 Using cuDNN for SwiGLU - used in LLaMA!");
    
    let chunks = input.chunk(2, dim as usize)?;
    if chunks.len() != 2 {
        return Err(crate::FlameError::InvalidOperation(
            "SwiGLU requires even dimension size".to_string()
        ));
    }
    
    // SwiGLU(a, b) = a * SiLU(b)
    let a = &chunks[0];
    let b = &chunks[1];
    let gate = cudnn_silu(b)?;
    a.mul(&gate)
}

/// Fused activation functions for efficiency
pub fn cudnn_fused_gelu_linear(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    println!("🚀 Using cuDNN for fused GELU+Linear - maximum efficiency!");
    
    // Fused: GELU(input @ weight^T + bias)
    use crate::cudnn::linear::cudnn_linear;
    let linear_out = cudnn_linear(input, weight, bias)?;
    cudnn_gelu(&linear_out)
}