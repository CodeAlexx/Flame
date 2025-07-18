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
    } else {
        println!("cargo:warning=CUDA_HOME not set, CUDA kernels will not be available");
    }
}