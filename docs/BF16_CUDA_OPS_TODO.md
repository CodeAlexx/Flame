# BF16 CUDA Ops TODO

- Implement CUDA kernels and wiring as per specification:
  - Workspace arena (`cuda/cuda_ops_common.cu`)
  - Elementwise BF16 ops (`cuda/elt_bf16.cu`)
  - Norms (layer/group/RMS) (`cuda/norm_bf16.cu`)
  - GEMM via cuBLASLt (`cuda/gemm_bf16_cublaslt.cu`)
  - Conv2d NHWC im2col (`cuda/conv2d_nhwc_bf16.cu`)
  - Streaming SDPA (`cuda/sdpa_stream_bf16.cu`)
- Integrate new object files into build.rs and link with cuBLAS/cuBLASLt.
- Add GPU unit tests for each op.
- Update Rust wrappers to call new CUDA implementations without FP32 fallbacks.
