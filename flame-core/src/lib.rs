pub mod dtype;
pub mod error;
pub mod shape;
pub mod tensor;
pub mod gradient;
pub mod device;
pub mod linear;
pub mod conv;
pub mod norm;
pub mod cuda_kernels;
pub mod cuda_kernel_sources;
pub mod cuda_ops;
pub mod cuda_kernels_v2;
pub mod autograd;
pub mod autograd_ops;
pub mod autograd_engine;
pub mod autograd_v2;
pub mod autograd_v3;
// pub mod autograd_simple;  // Replaced by autograd_v3
pub mod optimizers;
pub mod attention;
pub mod serialization;
pub mod regularization;
pub mod mixed_precision;
pub mod gradient_clip;
pub mod pooling;
pub mod upsampling;
pub mod activations;
// pub mod candle_interop;  // Temporarily disable due to cudarc version conflict
pub mod modulated_blocks;
pub mod flux_blocks;
pub mod mmdit_blocks;
// pub mod eridiffusion_adapter;  // Temporarily disable due to candle_interop dependency
pub mod conv3d;
pub mod flash_attention;
pub mod samplers;
pub mod tokenizer;
pub mod fp16;
pub mod kernel_launcher;

pub use dtype::DType;
pub use error::{FlameError, Result};
pub use shape::Shape;
pub use tensor::{Tensor, TensorId};

// Re-export cudarc types we use
pub use cudarc::driver::CudaDevice;