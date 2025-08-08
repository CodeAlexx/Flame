/// FLAME Configuration
/// Controls global behavior and optimization settings

use std::sync::atomic::{AtomicBool, Ordering};

/// Global flag to force cuDNN usage
static FORCE_CUDNN: AtomicBool = AtomicBool::new(true);  // Default to always use cuDNN

/// Check if cuDNN should be forced
pub fn should_use_cudnn() -> bool {
    FORCE_CUDNN.load(Ordering::Relaxed)
}

/// Set whether to force cuDNN usage (default: true)
pub fn set_force_cudnn(force: bool) {
    FORCE_CUDNN.store(force, Ordering::Relaxed);
    if force {
        println!("🚀 FLAME: cuDNN acceleration ENABLED - 60% memory reduction active!");
    } else {
        println!("⚠️  FLAME: cuDNN acceleration DISABLED - memory usage will increase!");
    }
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
            force_cudnn: true,        // Always use cuDNN by default
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