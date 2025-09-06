/// FLAME Configuration
/// Controls global behavior and optimization settings

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use crate::DType;

/// Global flag to force cuDNN usage
static FORCE_CUDNN: AtomicBool = AtomicBool::new(false);  // Default to not force cuDNN
static DEFAULT_DTYPE: AtomicU8 = AtomicU8::new(2); // 0=F32,1=F16,2=BF16 (default BF16)

/// Check if cuDNN should be forced
pub fn should_use_cudnn() -> bool {
    FORCE_CUDNN.load(Ordering::Relaxed)
}

/// Set whether to force cuDNN usage (default: false)
pub fn set_force_cudnn(force: bool) {
    FORCE_CUDNN.store(force, Ordering::Relaxed);
    // Intentionally quiet by default; callers can log their own messages.
}

/// Get the global default dtype used for new tensors without explicit dtype
pub fn default_dtype() -> DType {
    match DEFAULT_DTYPE.load(Ordering::Relaxed) {
        0 => DType::F32,
        1 => DType::F16,
        _ => DType::BF16,
    }
}

/// Set the global default dtype
pub fn set_default_dtype(dtype: DType) {
    let v = match dtype { DType::F32 => 0, DType::F16 => 1, _ => 2 };
    DEFAULT_DTYPE.store(v, Ordering::Relaxed);
}

/// FLAME optimization settings
pub struct FlameConfig {
    /// Always use cuDNN when available (default: true)
    pub force_cudnn: bool,
    
    /// Enable memory pooling (default: true) 
    pub enable_memory_pool: bool,
    
    /// Enable kernel fusion (default: true)
    pub enable_fusion: bool,
    
    /// Maximum batch size for operations
    pub max_batch_size: usize,
}

impl Default for FlameConfig {
    fn default() -> Self {
        Self {
            force_cudnn: false,       // Do not force cuDNN by default
            enable_memory_pool: true,
            enable_fusion: true,
            max_batch_size: 1024,
        }
    }
}

impl FlameConfig {
    /// Apply configuration globally
    pub fn apply(&self) {
        set_force_cudnn(self.force_cudnn);
        println!("FLAME Configuration Applied:");
        println!("  - cuDNN: {}", if self.force_cudnn { "FORCED ON" } else { "auto" });
        println!("  - Memory Pool: {}", if self.enable_memory_pool { "enabled" } else { "disabled" });
        println!("  - Kernel Fusion: {}", if self.enable_fusion { "enabled" } else { "disabled" });
        println!("  - Max Batch Size: {}", self.max_batch_size);
    }
}
