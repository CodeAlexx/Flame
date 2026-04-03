#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "flame_cuda_status.h"

#define FLAME_CUDA_TRY(call)                                        \
  do {                                                              \
    cudaError_t _flame_cuda_err = (call);                           \
    if (_flame_cuda_err != cudaSuccess) {                           \
      return FLAME_CUDA_ERR_CUDA;                                   \
    }                                                               \
  } while (0)

typedef struct FlameStreamArena {
  int device;
  cudaStream_t stream;
  void* base;
  size_t capacity;
  size_t offset;
  cudaEvent_t fence;
  bool owns_base;
} FlameStreamArena;

#ifdef __cplusplus
extern "C" {
#endif

FlameCudaStatus flame_arena_create_impl(int device,
                                        cudaStream_t stream,
                                        size_t capacity_bytes,
                                        FlameStreamArena** out);
FlameCudaStatus flame_arena_reset_impl(FlameStreamArena* arena);
FlameCudaStatus flame_arena_alloc_impl(FlameStreamArena* arena,
                                       size_t bytes,
                                       size_t align,
                                       void** out_ptr);
FlameCudaStatus flame_arena_record_and_release_impl(FlameStreamArena* arena);
FlameCudaStatus flame_arena_destroy_impl(FlameStreamArena* arena);

FlameCudaStatus flame_h2d_async_impl(void* dst_device,
                                     const void* src_host,
                                     size_t bytes,
                                     cudaStream_t stream);
FlameCudaStatus flame_d2h_async_impl(void* dst_host,
                                     const void* src_device,
                                     size_t bytes,
                                     cudaStream_t stream);
FlameCudaStatus flame_d2d_async_impl(void* dst_device,
                                     const void* src_device,
                                     size_t bytes,
                                     cudaStream_t stream);

typedef struct FlameConv2dAutotuneStats {
  uint64_t cache_hits;
  uint64_t cache_misses;
  uint64_t tuned;
  uint64_t weak;
  uint64_t fallbacks;
  uint64_t workspace_skips;
  uint64_t errors;
  uint64_t reprobes;
} FlameConv2dAutotuneStats;

FlameCudaStatus flame_conv2d_autotune_get_stats_impl(FlameConv2dAutotuneStats* out);
FlameCudaStatus flame_conv2d_autotune_reset_stats_impl(void);

typedef struct FlameSdpaAutotuneStats {
  uint64_t env_forced;
  uint64_t clamped;
  uint64_t skipped;
  uint64_t fallback;
  uint64_t errors;
  uint64_t cache_hits;
  uint64_t cache_misses;
  uint64_t tuned;
  uint64_t last_q_chunk;
  uint64_t last_k_chunk;
  uint64_t cache_saved;
  uint64_t cache_loads;
  uint64_t cache_load_errors;
  uint64_t cache_entries;
  uint64_t last_candidate_count;
  uint64_t last_best_time_ns;
  uint64_t last_plan_source;
  uint64_t last_shape_b;
  uint64_t last_shape_h;
  uint64_t last_shape_q;
  uint64_t last_shape_k;
  uint64_t last_shape_dh;
  uint64_t last_shape_dv;
  uint64_t last_shape_mask_heads;
  uint64_t last_shape_causal;
} FlameSdpaAutotuneStats;

FlameCudaStatus flame_sdpa_autotune_get_stats_impl(FlameSdpaAutotuneStats* out);
FlameCudaStatus flame_sdpa_autotune_reset_stats_impl(void);
FlameCudaStatus flame_sdpa_autotune_flush_cache_impl(void);

FlameCudaStatus flame_sdpa_chunked_bf16_impl(const __nv_bfloat16* q,
                                             const __nv_bfloat16* k,
                                             const __nv_bfloat16* v,
                                             int B,
                                             int H,
                                             int Q,
                                             int K,
                                             int Dh,
                                             int Dv,
                                             float scale,
                                             int chunk,
                                             int causal,
                                             int mask_heads,
                                             const __nv_bfloat16* mask,
                                             __nv_bfloat16* out,
                                             void* workspace,
                                             size_t workspace_bytes,
                                             cudaStream_t stream);

#ifdef __cplusplus
}
#endif

static inline bool flame_is_aligned(const void* ptr, size_t align) {
  uintptr_t addr = (uintptr_t)ptr;
  return (addr % align) == 0;
}
