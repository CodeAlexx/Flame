fn main() {
    // Generate C header when C API is enabled
    if std::env::var("CARGO_FEATURE_CAPI").is_ok() {
        if let (Ok(crate_dir), Ok(out_dir)) = (std::env::var("CARGO_MANIFEST_DIR"), std::env::var("OUT_DIR")) {
            let out = std::path::Path::new(&out_dir).join("flame.h");
            if let Ok(binding) = cbindgen::Builder::new().with_crate(&crate_dir).generate() {
                let _ = binding.write_to_file(out);
            } else {
                println!("cargo:warning=cbindgen generation failed");
            }
        } else {
            println!("cargo:warning=missing CARGO_MANIFEST_DIR or OUT_DIR for cbindgen");
        }
    }

    // Build CUDA narrow kernels
    let cuda_files = [
        "cuda/narrow_strided.cu",
        "cuda/narrow_strided_backward.cu",
    ];
    if cuda_files.iter().all(|p| std::path::Path::new(p).exists()) {
        let mut build = cc::Build::new();
        build.cuda(true);
        // Target architecture; adjust as needed or via env (e.g., CUDA_ARCH)
        let arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_80".to_string());
        build.flag(format!("-arch={}", arch));
        for f in &cuda_files {
            build.file(f);
            println!("cargo:rerun-if-changed={}", f);
        }
        build.compile("flame_core_cuda");
    }
}
