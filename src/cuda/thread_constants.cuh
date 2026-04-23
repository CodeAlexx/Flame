// flame-core/src/cuda/thread_constants.cuh
//
// Port of `aten/src/ATen/native/cuda/thread_constants.h`. Three compile-time
// constants that every TensorIterator-driven kernel uses to shape its grid.
//
// Values match PyTorch's CUDA (non-ROCm) path, but we lock the thread-work
// size to 4 (not PyTorch's current vec8-era `thread_work_size() = 8`) because
// the Phase 2 legacy path mirrors `launch_legacy_kernel<128, 4>` —
// `gpu_kernel_impl_nocast`'s `launch_legacy_kernel<128, unroll_factor>` call
// at CUDALoops.cuh:667 picks `unroll_factor = sizeof(arg0_t) >= 4 ? 2 : 4`,
// i.e. 4 for BF16. The vectorized fast path (deferred to Phase 5) may use
// `thread_work_size() = 8` later, but that's a separate constant.
//
// num_threads × thread_work_size = block_work_size = 512 for these values.
// Trade-off vs flame-core's legacy 256-threads / 1-elem-per-thread: fewer
// blocks, more work per block, same total parallelism. Safe for purely
// elementwise ops (no reduction order concerns).

#pragma once

#ifndef GPU_LAMBDA
#define GPU_LAMBDA __host__ __device__ __forceinline__
#endif

namespace flame {
namespace iter {

// Threads per block (PyTorch default for CUDA: 4 warps × 32 = 128).
// __host__ __device__ so device-side kernels (vectorized path, Phase 5)
// can use these as compile-time constants inside the kernel body.
__host__ __device__ constexpr int num_threads() { return 128; }

// Elements per thread for the legacy (non-vectorized) kernel path. Matches
// the BF16 branch of `gpu_kernel_impl_nocast` (`unroll_factor = 4`).
__host__ __device__ constexpr int thread_work_size() { return 4; }

// Elements per block = threads × elements/thread.
__host__ __device__ constexpr int block_work_size() {
    return num_threads() * thread_work_size();
}

}  // namespace iter
}  // namespace flame
