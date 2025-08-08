use std::env;
use std::path::PathBuf;

fn main() {
    // Always enable cuDNN since it's now default
    // Check if cudnn feature is enabled (which it always is now) and set up paths
    if env::var("CARGO_FEATURE_CUDNN").is_ok() || true {  // Always true now since cudnn is default
        // Check for CUDNN_PATH env var first, then fallback to extracted location
        let cudnn_lib_path = if let Ok(cudnn_path) = env::var("CUDNN_PATH") {
            format!("{}/usr/lib/x86_64-linux-gnu", cudnn_path)
        } else {
            // Use our extracted cuDNN 9.5.1
            "/home/alex/diffusers-rs/eridiffusion/cudnn_libs/usr/lib/x86_64-linux-gnu".to_string()
        };
        
        let cudnn_include_path = if let Ok(cudnn_path) = env::var("CUDNN_PATH") {
            format!("{}/usr/include", cudnn_path)
        } else {
            "/home/alex/diffusers-rs/eridiffusion/cudnn_libs/usr/include".to_string()
        };
        
        println!("cargo:warning=Building with cuDNN support");
        println!("cargo:warning=cuDNN lib path: {}", cudnn_lib_path);
        
        // Add library search path
        println!("cargo:rustc-link-search=native={}", cudnn_lib_path);
        
        // Link to cuDNN libraries
        println!("cargo:rustc-link-lib=cudnn");
        
        // Set rpath so the libraries can be found at runtime
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cudnn_lib_path);
        
        // Export include path for bindgen if needed
        println!("cargo:DEP_CUDNN_INCLUDE={}", cudnn_include_path);
    }
    
    // Only compile CUDA kernels if CUDA is available
    if let Ok(_cuda_path) = env::var("CUDA_HOME") {
        println!("cargo:rerun-if-changed=src/cuda/kernels.cu");
        
        // Set up CUDA compilation
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        
        // Compile CUDA kernels to PTX
        let status = std::process::Command::new("nvcc")
            .args(&[
                "-ptx",
                "-o",
                out_dir.join("kernels.ptx").to_str().unwrap(),
                "src/cuda/kernels.cu",
                "-arch=sm_86", // For RTX 3090, adjust as needed
                "-use_fast_math",
                "-O3",
            ])
            .status()
            .expect("Failed to execute nvcc");
            
        if !status.success() {
            panic!("Failed to compile CUDA kernels");
        }
        
        println!("cargo:rustc-env=FLAME_KERNELS_PTX={}", out_dir.join("kernels.ptx").display());
        
        // Link flash attention library if available
        if env::var("CARGO_FEATURE_FLASH_ATTN").is_ok() {
            // Look for pre-built flash attention library
            let flash_attn_dir = env::var("CANDLE_FLASH_ATTN_BUILD_DIR")
                .or_else(|_| env::var("FLASH_ATTN_LIB_DIR"));
                
            if let Ok(dir) = flash_attn_dir {
                println!("cargo:rustc-link-search={}", dir);
                println!("cargo:rustc-link-lib=flashattention");
                println!("cargo:rustc-link-lib=dylib=cudart");
                
                // Don't link stdc++ on Windows MSVC
                let target = env::var("TARGET").unwrap_or_default();
                if !target.contains("msvc") {
                    println!("cargo:rustc-link-lib=dylib=stdc++");
                }
            }
        }
    } else {
        println!("cargo:warning=CUDA_HOME not set, CUDA kernels will not be available");
    }
}