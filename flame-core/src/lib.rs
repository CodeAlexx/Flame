
/// Macro for launching CUDA kernels
#[macro_export]
macro_rules! launch_kernel {
    ($func:expr, $cfg:expr, $($args:expr),* $(,)?) => {{
        unsafe { let _ = $func.launch($cfg, ($($args,)*)); }
    }};
}

pub mod config;
pub mod dtype;
pub mod error;
pub mod shape;
pub mod tensor_storage;
pub mod tensor;
pub mod ops_ext;
pub mod devtensor;
pub mod tensor_narrow;
pub mod gradient;
pub mod memory_pool;
pub mod device;
pub mod linear;
pub mod conv;
pub mod cuda_conv2d;
pub mod cuda;
pub mod cuda_conv2d_direct;
pub mod cuda_conv2d_fast;

// cuDNN integration module - separate and feature-gated
#[cfg(feature = "cudnn")]
pub mod cudnn;


pub mod layer_norm;
pub mod norm;
pub mod group_norm;
#[cfg(feature = "flash_attn")]
#[path = "flash_attention.rs"]
pub mod flash_attention;
pub mod embedding;
pub mod cuda_kernels;
pub mod cuda_kernels_gpu;
pub mod cuda_kernel_sources;
pub mod cuda_kernel_compiler;
pub mod cuda_ops;
pub mod blas;
pub mod logging;
// pub mod cuda_kernels_v2;  // Removed: experimental version
pub mod cuda_conv2d_kernels;
pub mod cuda_memory_alignment;
pub mod cuda_gradient_ops;
pub mod cuda_tensor_gpu;
pub mod autograd_ops_complete;
pub mod bf16_support;
pub mod autograd;
// pub mod autograd_ops;  // Using new autograd
// pub mod autograd_engine;  // Using new autograd
#[cfg(feature = "legacy")]
#[path = "legacy/autograd_v2.rs"]
// TEMP-REMOVE: pub mod autograd_v2;  // legacy
pub mod autograd_v3;  // This is used
#[cfg(feature = "legacy")]
#[path = "legacy/autograd_simple.rs"]
pub mod autograd_simple;  // legacy
pub mod optimizers;
pub mod attention;
pub mod sdpa;
pub mod serialization;
pub mod regularization;
pub mod mixed_precision;
#[cfg(feature = "legacy")]
#[path = "legacy/gradient_clip.rs"]
pub mod gradient_clip;
pub mod pooling;
pub mod pooling_impl;
pub mod upsampling;
pub mod activations;
pub mod image_ops_nhwc;
pub mod vae;
// candle_interop removed: no Candle interop in Flame
// Model-specific blocks removed - these belong in the application layer, not the framework
// pub mod modulated_blocks;  // Removed: model-specific
// pub mod flux_blocks;  // Removed: model-specific
// pub mod mmdit_blocks;  // Removed: model-specific
// eridiffusion_adapter removed: depended on Candle interop
#[cfg(feature = "legacy")]
#[path = "legacy/conv3d.rs"]
pub mod conv3d;
#[cfg(feature = "legacy")]
#[path = "legacy/conv3d_simple.rs"]
pub mod conv3d_simple;
pub mod fused_kernels;
pub mod samplers;
// pub mod tokenizer;  // Removed: model-specific
pub mod fp16;
#[cfg(feature = "bf16_u16")]
pub mod bf16_ops;
#[cfg(feature = "bf16_u16")]
pub mod bf16_convert;
#[cfg(feature = "bf16_u16")]
pub mod bf16_factories;
#[cfg(feature = "bf16_u16")]
pub mod bf16_clamp;
#[cfg(feature = "bf16_u16")]
pub mod bf16_normal;
#[cfg(feature = "bf16_u16")]
pub mod bf16_elementwise;
#[cfg(feature = "legacy")]
#[path = "legacy/kernel_launcher.rs"]
pub mod kernel_launcher;
pub mod tensor_ops_extended;
// Provide missing convenience ops (div_scalar, etc.) unconditionally
pub mod tensor_ops_missing;
pub mod parameter;
pub mod adam;
pub mod lora;
// pub mod sdxl_attention;  // Removed: model-specific
// pub mod sdxl_unet_blocks;  // Removed: model-specific
pub mod loss;
pub mod gradient_checkpointing;
pub mod sage_attention;

pub use config::{FlameConfig, should_use_cudnn, set_force_cudnn, default_dtype, set_default_dtype};
pub use dtype::DType;
pub use error::{FlameError, Result};
pub use error::FlameError as Error;
pub use shape::{Shape, D};
pub use tensor::{Tensor, TensorId};
pub use autograd::{AutogradContext, Op};
pub use gradient::{GradientMap, TensorGradExt};
pub use group_norm::{group_norm, GroupNorm};
#[cfg(feature = "flash_attn")]
pub use flash_attention::{flash_attention_forward, FlashAttention};
pub use device::Device;
pub use parameter::Parameter as Var;
pub use parameter::Parameter;

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

/// Initialize FLAME with sensible defaults.
/// Called automatically when FLAME is loaded.
pub fn init() {
    // Default: do not force cuDNN unless explicitly requested.
    // Use env var FLAME_FORCE_CUDNN=1 to force-enable at runtime when feature is present.
    let force = std::env::var("FLAME_FORCE_CUDNN").ok().as_deref() == Some("1");
    set_force_cudnn(force);
    // Default dtype: FLAME_DEFAULT_DTYPE=bf16|f16|f32 (default: bf16)
    if let Ok(v) = std::env::var("FLAME_DEFAULT_DTYPE") {
        let d = match v.to_lowercase().as_str() { "f32" => DType::F32, "f16" => DType::F16, _ => DType::BF16 };
        set_default_dtype(d);
    }
}

// Automatically initialize on load
#[ctor::ctor]
fn auto_init() { init(); }

// Optional C API surface (disabled by default)
#[cfg(feature = "capi")]
pub mod capi;
