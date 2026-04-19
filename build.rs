#[cfg(feature = "cuda")]
fn main() {
    use std::env;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    // Generate C header when C API is enabled
    if env::var("CARGO_FEATURE_CAPI").is_ok() {
        if let (Ok(crate_dir), Ok(out_dir)) = (env::var("CARGO_MANIFEST_DIR"), env::var("OUT_DIR"))
        {
            let out = Path::new(&out_dir).join("flame.h");
            if let Ok(binding) = cbindgen::Builder::new().with_crate(&crate_dir).generate() {
                let _ = binding.write_to_file(out);
            } else {
                println!("cargo:warning=cbindgen generation failed");
            }
        } else {
            println!("cargo:warning=missing CARGO_MANIFEST_DIR or OUT_DIR for cbindgen");
        }
    }

    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=LD_LIBRARY_PATH");
    println!("cargo:rerun-if-changed=cuda");
    println!("cargo:rerun-if-changed=src/cuda");

    let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let nvcc = env::var("NVCC").unwrap_or_else(|_| format!("{cuda_home}/bin/nvcc"));

    if !Path::new(&nvcc).exists() {
        panic!(
            "CUDA GPU required. Set CUDA_HOME=/usr/local/cuda and export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH. NVCC not found at {nvcc}.",
        );
    }
    let cudart = format!("{cuda_home}/lib64/libcudart.so");
    if !Path::new(&cudart).exists() {
        panic!(
            "CUDA GPU required. Set CUDA_HOME=/usr/local/cuda and export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH. Missing {cudart}.",
        );
    }

    // Check for cuDNN in python venvs
    let cudnn_paths = [
        "/home/alex/serenity/venv/lib/python3.12/site-packages/nvidia/cudnn/lib",
        "/home/alex/SimpleTuner/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib",
        "/home/alex/SimpleTuner/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib",
    ];
    for venv_cudnn in &cudnn_paths {
        if Path::new(venv_cudnn).exists() {
            println!("cargo:rustc-link-search=native={}", venv_cudnn);
            // Set rpath so binary finds libcudnn at runtime
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", venv_cudnn);
            break;
        }
    }

    println!("cargo:warning=CUDA_HOME={cuda_home}");
    println!("cargo:warning=NVCC path={nvcc}");

    println!("cargo:warning=flame-core: compiling CUDA kernels");
    let mut cuda_sources = vec![
        "cuda/narrow_strided.cu",
        "cuda/permute0213.cu",
        "cuda/reduce_sum_bf16.cu",
        "cuda/gemm_bf16_fp32acc.cu",
        "cuda/gemm_bf16_cublaslt.cu",
        "cuda/conv2d_nhwc_bf16.cu",
        "cuda/sdpa_stream_bf16.cu",
        "cuda/add_inplace.cu",
        "cuda/add_same_shape.cu",
        "cuda/broadcast.cu",
        "cuda/tile_bc.cu",
        "cuda/bf16_slice_index.cu",
        "cuda/bf16_broadcast_repeat.cu",
        "cuda/repeat_bf16.cu",
        "cuda/streaming_attn_bf16.cu",
        "cuda/modulate_affine_bf16.cu",
        "cuda/gate_mul_bf16.cu",
        "src/cuda/pinned_host.cu",
    ];

    // BF16/NHWC CUDA ops surface (new implementation)
    cuda_sources.push("cuda/cuda_ops_common.cu");
    cuda_sources.push("cuda/cuda_ops.cu");
    cuda_sources.push("cuda/src/flame_cuda_common.cu");
    cuda_sources.push("cuda/src/flame_bf16_utils.cu");
    cuda_sources.push("cuda/src/flame_nhwc_adapters.cu");
    cuda_sources.push("cuda/src/flame_conv2d_stub.cu");
    cuda_sources.push("cuda/src/flame_sdpa_stub.cu");
    cuda_sources.push("cuda/src/flame_norm_bf16.cu");
    cuda_sources.push("cuda/upsample_nearest.cu");

    cuda_sources.push("kernels/adaln_layernorm_bf16.cu");
    cuda_sources.push("kernels/rope_kernels.cu");
    cuda_sources.push("kernels/sdpa_kernels.cu");
    cuda_sources.push("kernels/geglu_kernels.cu");
    cuda_sources.push("kernels/silu_backward.cu");
    cuda_sources.push("src/cuda/f32_to_bf16.cu");
    cuda_sources.push("kernels/swiglu_backward.cu");
    cuda_sources.push("kernels/relu_backward.cu");
    cuda_sources.push("kernels/gelu_backward.cu");
    cuda_sources.push("kernels/tanh_backward.cu");
    cuda_sources.push("kernels/sigmoid_backward.cu");

    // Fused inference kernels (flame-swap / LTX-2 perf)
    cuda_sources.push("src/cuda/fused_rms_norm.cu");
    cuda_sources.push("src/cuda/fused_modulate.cu");
    cuda_sources.push("src/cuda/fused_linear3d.cu");
    cuda_sources.push("src/cuda/flash_attention_fwd.cu");
    cuda_sources.push("src/cuda/flash_attention_bwd.cu");
    cuda_sources.push("src/cuda/fp8_dequant.cu");
    cuda_sources.push("src/cuda/fp8_quant.cu");
    cuda_sources.push("src/cuda/fp16_to_bf16.cu");
    cuda_sources.push("src/cuda/fused_norm_modulate.cu");
    cuda_sources.push("src/cuda/fused_residual_gate.cu");
    cuda_sources.push("src/cuda/fused_dequant_transpose.cu");
    cuda_sources.push("src/cuda/grouped_mm.cu");
    cuda_sources.push("src/cuda/fused_gated_scatter_add.cu");

    if !cuda_sources.iter().all(|p| Path::new(p).exists()) {
        panic!("CUDA sources missing; ensure submodules are synced");
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));

    // Clean out stale CUDA artifacts before rebuilding. When the archive already
    // contains objects with hashed names from a previous build, re-adding the
    // freshly compiled objects leads to duplicate symbol errors. Removing the
    // old `.o`/`.a` files keeps the archive deterministic.
    if let Ok(entries) = std::fs::read_dir(&out_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let should_remove = match path.extension().and_then(|s| s.to_str()) {
                Some("o") => true,
                Some("a") => path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .map(|name| name == "libflame_cuda_kernels.a")
                    .unwrap_or(false),
                _ => false,
            };
            if should_remove {
                let _ = std::fs::remove_file(&path);
            }
        }
    }

    let mut objects = Vec::new();
    for src in &cuda_sources {
        println!("cargo:rerun-if-changed={}", src);
        let obj_path = out_dir
            .join(Path::new(src).file_name().expect("cuda source filename"))
            .with_extension("o");

        let mut cmd = Command::new(&nvcc);
        cmd.arg("-std=c++17")
            .arg("-O3")
            .arg("--use_fast_math")
            .arg("-Xcompiler")
            .arg("-fPIC")
            .arg("-rdc=true")
            .arg("-c")
            .arg(src)
            .arg("-o")
            .arg(&obj_path)
            .arg("-gencode")
            .arg("arch=compute_80,code=sm_80")
            .arg("-gencode")
            .arg("arch=compute_86,code=sm_86")
            .arg("-gencode")
            .arg("arch=compute_89,code=sm_89")
            .arg(format!("-I{cuda_home}/include"));
        cmd.arg("-I").arg("cuda/include");
        if src.ends_with("streaming_attn_bf16.cu") {
            cmd.arg("-Xptxas").arg("-v");
        }

        let status = cmd.status().expect("failed to invoke nvcc");
        if !status.success() {
            panic!("nvcc failed for {src} with status {status:?}");
        }
        objects.push(obj_path);
    }

    // Device link step
    let dlink_obj = out_dir.join("flame_cuda_kernels_dlink.o");
    let mut dlink = Command::new(&nvcc);
    dlink
        .arg("-dlink")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("-Xcompiler")
        .arg("-fPIC")
        .arg("-rdc=true")
        .arg("-gencode")
        .arg("arch=compute_80,code=sm_80")
        .arg("-gencode")
        .arg("arch=compute_86,code=sm_86")
        .arg("-gencode")
        .arg("arch=compute_89,code=sm_89")
        .arg(format!("-I{cuda_home}/include"))
        .arg(format!("-L{cuda_home}/lib64"));
    for obj in &objects {
        dlink.arg(obj);
    }
    dlink.arg("-o").arg(&dlink_obj);
    let status = dlink.status().expect("nvcc device link failed");
    if !status.success() {
        panic!("nvcc device link failed with {status:?}");
    }
    objects.push(dlink_obj);

    // Archive objects into static library
    let lib_path = out_dir.join("libflame_cuda_kernels.a");
    let mut ar = Command::new("ar");
    ar.arg("crus").arg(&lib_path);
    for obj in &objects {
        ar.arg(obj);
    }
    let status = ar.status().expect("failed to invoke ar");
    if !status.success() {
        panic!("ar failed with {status:?}");
    }

    let cuda_lib = format!("{cuda_home}/lib64");
    println!("cargo:rustc-link-search=native={cuda_lib}");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=flame_cuda_kernels");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cudadevrt");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    println!("cargo:rerun-if-changed=src/ffi/cuda_ffi.c");
    cc::Build::new()
        .cpp(true)
        .file("src/ffi/cuda_ffi.c")
        .include("cuda/include")
        .include(format!("{cuda_home}/include"))
        .flag_if_supported("-std=c++17")
        .compile("flame_cuda_ffi");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    panic!(
        "CUDA feature disabled but required by default. Enable with --features=cuda or use the default build."
    );
}
