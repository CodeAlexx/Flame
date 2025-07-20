use std::env;
use std::path::PathBuf;

fn main() {
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