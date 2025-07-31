/// Macro for launching CUDA kernels
#[macro_export]
macro_rules! launch_kernel {
    ($func:expr, $cfg:expr, $($args:expr),* $(,)?) => {{
        unsafe { $func.launch($cfg, ($($args,)*)) }
    }};
}

pub mod dtype;
pub mod error;
pub mod shape;
pub mod tensor_storage;
pub mod tensor;
pub mod gradient;
pub mod memory_pool;
pub mod device;
pub mod linear;
pub mod conv;
pub mod cuda_conv2d;
pub mod layer_norm;
pub mod norm;
pub mod group_norm;
pub mod flash_attention;
pub mod embedding;
pub mod cuda_kernels;
pub mod cuda_kernels_gpu;
pub mod cuda_kernel_sources;
pub mod cuda_kernel_compiler;
pub mod cuda_ops;
// pub mod cuda_kernels_v2;  // Removed: experimental version
pub mod cuda_conv2d_kernels;
pub mod cuda_gradient_ops;
pub mod cuda_tensor_gpu;
pub mod autograd_ops_complete;
pub mod autograd;
// pub mod autograd_ops;  // Using new autograd
// pub mod autograd_engine;  // Using new autograd
// pub mod autograd_v2;  // Using new autograd
pub mod autograd_v3;  // This is used
// pub mod autograd_simple;  // Replaced by autograd
pub mod optimizers;
pub mod attention;
pub mod serialization;
pub mod regularization;
pub mod mixed_precision;
pub mod gradient_clip;
pub mod pooling;
pub mod pooling_impl;
pub mod upsampling;
pub mod activations;
// pub mod candle_interop;  // Temporarily disable due to cudarc version conflict
// Model-specific blocks removed - these belong in the application layer, not the framework
// pub mod modulated_blocks;  // Removed: model-specific
// pub mod flux_blocks;  // Removed: model-specific
// pub mod mmdit_blocks;  // Removed: model-specific
// pub mod eridiffusion_adapter;  // Temporarily disable due to candle_interop dependency
pub mod conv3d;
pub mod conv3d_simple;
pub mod fused_kernels;
pub mod samplers;
// pub mod tokenizer;  // Removed: model-specific
pub mod fp16;
pub mod kernel_launcher;
pub mod tensor_ops_extended;
pub mod tensor_ops_missing;
pub mod parameter;
pub mod adam;
pub mod lora;
// pub mod sdxl_attention;  // Removed: model-specific
// pub mod sdxl_unet_blocks;  // Removed: model-specific
pub mod loss;
pub mod gradient_checkpointing;
pub mod sage_attention;

pub use dtype::DType;
pub use error::{FlameError, Result};
pub use shape::{Shape, D};
pub use tensor::{Tensor, TensorId};
pub use autograd::{AutogradContext, Op};
pub use gradient::{GradientMap, TensorGradExt};
pub use group_norm::{group_norm, GroupNorm};
pub use flash_attention::{flash_attention_forward, FlashAttention};
pub use device::Device;
pub use parameter::Parameter as Var;

// Re-export cudarc types we use
pub use cudarc::driver::CudaDevice;

// Module trait for layers
pub trait Module {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}

// Backprop module for compatibility
pub mod backprop {
    pub use crate::gradient::GradientMap as GradStore;
}

// nn module for neural network layers
pub mod nn {
    pub use crate::linear::{Linear, linear};
    pub use crate::layer_norm::LayerNorm;
    pub use crate::embedding::Embedding;
    pub use crate::conv::Conv2d;
    pub use crate::cuda_conv2d::conv2d;
    pub use crate::adam::AdamW;
    
    // Re-export optimizer trait
    pub trait Optimizer {
        fn step(&mut self) -> Result<()>;
        fn zero_grad(&mut self);
    }
    
    use crate::Result;
}