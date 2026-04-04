#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    unused_mut,
    unused_assignments,
    unreachable_code,
    unused_macros,
    unexpected_cfgs,
    non_camel_case_types,
    private_interfaces,
    elided_lifetimes_in_paths,
    clippy::identity_op,
    clippy::assign_op_pattern,
    clippy::derivable_impls,
    clippy::len_without_is_empty,
    clippy::let_and_return,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::manual_slice_size_calculation,
    clippy::missing_safety_doc,
    clippy::new_without_default,
    clippy::needless_borrow,
    clippy::needless_question_mark,
    clippy::needless_range_loop,
    clippy::needless_return,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::too_many_arguments,
    clippy::unnecessary_cast,
    clippy::useless_conversion
)]

/// Macro for launching CUDA kernels
#[macro_export]
macro_rules! launch_kernel {
    ($func:expr, $cfg:expr, $($args:expr),* $(,)?) => {{
        let func = $func;
        let cfg = $cfg;
        let args = ($($args,)*);
        unsafe { let _ = func.launch(cfg, args); }
    }};
}

#[cfg(feature = "legacy_cpu_autograd")]
compile_error!(
    "CPU autograd is disabled in this project. Remove the 'legacy_cpu_autograd' feature to build."
);

#[cfg(feature = "dtype_trace")]
#[macro_export]
macro_rules! dtype_trace {
    ($($arg:tt)*) => {
        eprintln!("[dtype-trace] {}", format!($($arg)*));
    };
}

#[cfg(not(feature = "dtype_trace"))]
#[macro_export]
macro_rules! dtype_trace {
    ($($arg:tt)*) => {{}};
}

pub mod trace {
    #[inline]
    pub fn trace_on() -> bool {
        std::env::var("FLAME_DTYPE_TRACE")
            .ok()
            .or_else(|| std::env::var("FLAME_TRACE_DTYPE").ok())
            .as_deref()
            == Some("1")
    }
}

pub mod config;
pub mod conv;
pub mod conv3d_bf16;
pub mod conv3d_simple;
pub mod cuda;
pub mod cuda_conv2d;
pub mod cuda_conv2d_direct;
pub mod cuda_conv2d_fast;
pub mod debug_device;
pub mod pinned;
pub use pinned::{
    memcpy_async_device_to_host, memcpy_async_host_to_device, register_slice_as_pinned,
    unregister_pinned, PinnedAllocFlags, PinnedHostBuffer, PinnedHostBufferView,
    PinnedHostBufferViewMut, StagingDeviceBuf,
};
pub mod device;
pub mod devtensor;
pub mod dtype;
pub mod error;
pub mod gradient;
pub mod linear;
pub mod memory_pool;
pub mod ops;
pub mod ops_ext;
pub mod rng;
pub mod sgd;
pub mod shape;
#[cfg(all(feature = "cuda", feature = "bf16_u16"))]
pub mod staging;
pub mod tensor;
pub mod tensor_compute;
pub mod tensor_ext;
pub mod tensor_narrow;
pub mod tensor_storage;
pub use tensor::contracts::*;
pub mod telemetry;

// cuDNN integration module - separate and feature-gated
#[cfg(feature = "cudnn")]
pub mod cudnn;

pub mod blas;
#[cfg(feature = "borrowed_weights")]
pub mod borrowed;
pub mod cuda_kernel_compiler;
pub mod cuda_kernel_sources;
pub mod cuda_kernels;
pub mod cuda_kernels_gpu;
pub mod cuda_ops;
pub mod cuda_ops_bf16;
pub mod cuda_ops_ffi;
pub mod embedding;
pub mod pinned_pool;
pub use pinned_pool::PinnedPool;
#[cfg(feature = "cuda")]
pub mod kernels {
    pub mod adaln;
}
#[cfg(feature = "flash_attn")]
#[path = "flash_attention.rs"]
pub mod flash_attention;
pub mod group_norm;
pub mod layer_norm;
pub mod logging;
pub mod norm;
// pub mod cuda_kernels_v2;  // Removed: experimental version
pub mod autograd;
pub mod autograd_ops_complete;
pub mod bf16_support;
pub mod cuda_conv2d_kernels;
pub mod cuda_gradient_ops;
pub mod cuda_memory_alignment;
pub mod cuda_tensor_gpu;
// pub mod autograd_ops;  // Using new autograd
// pub mod autograd_engine;  // Using new autograd
pub mod activations;
pub mod attention;
pub mod autograd_v3; // Primary autograd engine
#[cfg(feature = "autograd_v4")]
pub mod autograd_v4;
pub mod image_ops_nhwc;
pub mod mixed_precision;
pub mod optimizers;
pub mod perf_telemetry;
pub mod pooling;
pub mod pooling_impl;
pub mod regularization;
pub mod sdpa;
pub mod serialization;
pub mod strict;
pub mod upsampling;
pub mod vae;
// candle_interop removed: no Candle interop in Flame
// Model-specific blocks removed - these belong in the application layer, not the framework
// pub mod modulated_blocks;  // Removed: model-specific
// pub mod flux_blocks;  // Removed: model-specific
// pub mod mmdit_blocks;  // Removed: model-specific
// eridiffusion_adapter removed: depended on Candle interop
pub mod fused_kernels;
pub mod samplers;
// pub mod tokenizer;  // Removed: model-specific
#[cfg(feature = "bf16_u16")]
#[allow(dead_code, unused_imports, unused_variables, unused_mut)]
pub mod bf16_clamp;
#[cfg(feature = "bf16_u16")]
#[allow(dead_code, unused_imports, unused_variables, unused_mut)]
pub mod bf16_convert;
#[cfg(feature = "bf16_u16")]
#[allow(dead_code, unused_imports, unused_variables, unused_mut)]
pub mod bf16_elementwise;
#[cfg(feature = "bf16_u16")]
#[allow(dead_code, unused_imports, unused_variables, unused_mut)]
pub mod bf16_factories;
#[cfg(feature = "bf16_u16")]
#[allow(dead_code, unused_imports, unused_variables, unused_mut)]
pub mod bf16_normal;
#[cfg(feature = "bf16_u16")]
#[allow(dead_code, unused_imports, unused_variables, unused_mut)]
pub mod bf16_ops;
pub mod fp16;
pub mod tensor_ops_extended;
// Provide missing convenience ops (div_scalar, etc.) unconditionally
pub mod adam;
pub mod lora;
pub mod parameter;
pub mod tensor_ops_missing;
// pub mod sdxl_attention;  // Removed: model-specific
// pub mod sdxl_unet_blocks;  // Removed: model-specific
pub mod gradient_checkpointing;
pub mod loss;
pub mod sage_attention;

pub use autograd::{AutogradContext, Op};
pub use config::{
    default_dtype, set_default_dtype, set_force_cudnn, should_use_cudnn, FlameConfig,
};
pub use device::{global_cuda_device, Device};
pub use dtype::DType;
pub use error::{Error, Result};
pub type FlameError = Error;
#[cfg(feature = "flash_attn")]
pub use flash_attention::{flash_attention_forward, FlashAttention};
pub use gradient::{GradientMap, TensorGradExt};
pub use group_norm::{group_norm, GroupNorm};
pub use parameter::Parameter as Var;
pub use parameter::Parameter;
pub use rng::global_rng;
pub use shape::{Shape, D};
pub use strict::{
    allow_clone, allow_f32_in_kernel, allow_f32_in_kernel_scoped, scope, telemetry_snapshot,
    GuardMode, StrictTelemetry,
};
pub use telemetry::{
    record_dtype_trap, record_tensor_bytes, reset_counters as reset_telemetry,
    snapshot as telemetry_snapshot_full, TelemetrySnapshot,
};
pub use tensor::{Tensor, TensorId};
pub use tensor_ext::to_owning_fp32_strong;

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
    pub use crate::adam::AdamW;
    pub use crate::conv::Conv2d;
    pub use crate::cuda_conv2d::conv2d;
    pub use crate::embedding::Embedding;
    pub use crate::layer_norm::LayerNorm;
    pub use crate::linear::{linear, Linear};

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
        let d = match v.to_lowercase().as_str() {
            "f32" => DType::F32,
            "f16" => DType::F16,
            _ => DType::BF16,
        };
        set_default_dtype(d);
    }

    let strict_policy = if strict::is_enabled() {
        "bf16_policy=STRICT, f32_fallbacks=DISABLED, reshape_clone=DENY"
    } else {
        "bf16_policy=RELAXED, f32_fallbacks=WARN, reshape_clone=ALLOW"
    };
    log::info!("[flame-core] {strict_policy}");
}

// Automatically initialize on load
#[ctor::ctor]
fn auto_init() {
    init();
}

#[cfg(test)]
mod __flame_test_defaults {
    use super::{set_default_dtype, DType};

    #[ctor::ctor]
    fn set_test_default_dtype() {
        set_default_dtype(DType::F32);
    }
}

#[cfg(test)]
mod tests;

#[cfg(feature = "python")]
mod python;

// Optional C API surface (disabled by default)
#[cfg(feature = "capi")]
pub mod capi;
