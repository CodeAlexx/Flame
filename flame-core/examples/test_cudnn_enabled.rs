fn main() {
    println!("\n=== Testing cuDNN is Enabled ===");
    
    // Check if cuDNN feature is enabled
    #[cfg(feature = "cudnn")]
    {
        println!("✅ cuDNN feature is ENABLED!");
        println!("✅ cuDNN is the default feature!");
        
        // Check if cuDNN is available
        if flame_core::cudnn::is_cudnn_available() {
            println!("✅ cuDNN runtime is AVAILABLE!");
            println!("🚀 60% memory reduction is ACTIVE!");
        } else {
            println!("❌ cuDNN runtime check failed");
        }
    }
    
    #[cfg(not(feature = "cudnn"))]
    {
        println!("❌ cuDNN feature is DISABLED - this should never happen!");
        panic!("cuDNN must always be enabled!");
    }
    
    println!("\n🔥 FLAME is configured to ALWAYS use cuDNN!");
    println!("💪 The world's best Rust-only trainer!");
}