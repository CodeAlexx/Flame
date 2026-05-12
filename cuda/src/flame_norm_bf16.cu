#include "../cuda_ops.h"
#include "../include/flame_norm_bf16.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

#define FC_RETURN_IF_ERROR(stmt) \
  do { \
    fc_status_t _status = (stmt); \
    if (_status != FC_OK) { \
      return _status; \
    } \
  } while (0)

namespace {

__device__ inline float bf16_to_f32(const __nv_bfloat16& x) {
  return __bfloat162float(x);
}

__device__ inline __nv_bfloat16 f32_to_bf16(float x) {
  return __float2bfloat16_rn(x);
}

// 8-byte aligned struct for vectorized BF16 loads (4 bf16 = 8 bytes).
struct __align__(8) bf16x4_t {
  __nv_bfloat16 v[4];
};

// 2026-05-12 perf: vectorized LayerNorm backward writing dx only.
// Replaces the legacy single-thread-per-row kernel below. 256 threads/block,
// vec_size=4 BF16 loads, warp-shuffle + smem inter-warp reduction. dgamma /
// dbeta accumulation moved to a separate cross-row reduction kernel
// (layer_norm_grad_weight_bias_bf16_vec_kernel) — accumulating inline would
// require batch_size * norm_size atomicAdds (10M+ on production shapes).
//
// Caller must ensure norm_size % 4 == 0 (production shapes 2560/3072/4096
// all qualify).
__global__ void layer_norm_backward_bf16_vec_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ dy,
    const __nv_bfloat16* __restrict__ gamma,
    int64_t norm_size,
    float eps,
    bool has_gamma,
    __nv_bfloat16* __restrict__ dx) {
  const int VEC = 4;
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int n_threads = blockDim.x;
  const int64_t n_vec = norm_size / VEC;

  const bf16x4_t* X  = reinterpret_cast<const bf16x4_t*>(x  + row * norm_size);
  const bf16x4_t* DY = reinterpret_cast<const bf16x4_t*>(dy + row * norm_size);
  bf16x4_t*       DX = reinterpret_cast<bf16x4_t*>(dx + row * norm_size);
  const bf16x4_t* G  = (has_gamma && gamma != nullptr)
      ? reinterpret_cast<const bf16x4_t*>(gamma) : nullptr;

  // Pass 1: parallel sum + sum_sq for mean / var.
  float local_sum = 0.f;
  float local_sq  = 0.f;
  for (int64_t i = tid; i < n_vec; i += n_threads) {
    bf16x4_t d = X[i];
    _Pragma("unroll")
    for (int k = 0; k < VEC; ++k) {
      float v = __bfloat162float(d.v[k]);
      local_sum += v;
      local_sq  += v * v;
    }
  }

  // Intra-warp reduction (sum then sq) via shuffle.
  for (int off = 16; off > 0; off >>= 1) {
    local_sum += __shfl_xor_sync(0xffffffff, local_sum, off);
    local_sq  += __shfl_xor_sync(0xffffffff, local_sq,  off);
  }

  // Inter-warp reduction via shared memory.
  // Layout: [n_warps floats for sum] [n_warps floats for sq] (2 * n_warps total).
  extern __shared__ float smem[];  // sized 2*n_warps from launch
  const int warp_id = tid >> 5;
  const int lane    = tid & 31;
  const int n_warps = (n_threads + 31) >> 5;

  if (lane == 0) {
    smem[warp_id] = local_sum;
    smem[n_warps + warp_id] = local_sq;
  }
  __syncthreads();

  if (warp_id == 0) {
    float s = (lane < n_warps) ? smem[lane] : 0.f;
    float q = (lane < n_warps) ? smem[n_warps + lane] : 0.f;
    for (int off = 16; off > 0; off >>= 1) {
      s += __shfl_xor_sync(0xffffffff, s, off);
      q += __shfl_xor_sync(0xffffffff, q, off);
    }
    if (lane == 0) {
      smem[0] = s;
      smem[n_warps] = q;
    }
  }
  __syncthreads();

  const float total_sum = smem[0];
  const float total_sq  = smem[n_warps];
  const float inv_norm = 1.f / static_cast<float>(norm_size);
  const float mean = total_sum * inv_norm;
  const float var  = total_sq * inv_norm - mean * mean;
  const float inv_std = rsqrtf(var + eps);

  // Pass 2: parallel sum1 = sum(dy * g), sum2 = sum(dy * g * xn).
  float local_s1 = 0.f;
  float local_s2 = 0.f;
  for (int64_t i = tid; i < n_vec; i += n_threads) {
    bf16x4_t dyv = DY[i];
    bf16x4_t xv  = X[i];
    bf16x4_t gv;
    if (G != nullptr) gv = G[i];
    _Pragma("unroll")
    for (int k = 0; k < VEC; ++k) {
      float dy_v = __bfloat162float(dyv.v[k]);
      float x_v  = __bfloat162float(xv.v[k]);
      float g_v  = (G != nullptr) ? __bfloat162float(gv.v[k]) : 1.f;
      float xn   = (x_v - mean) * inv_std;
      local_s1 += dy_v * g_v;
      local_s2 += dy_v * g_v * xn;
    }
  }

  for (int off = 16; off > 0; off >>= 1) {
    local_s1 += __shfl_xor_sync(0xffffffff, local_s1, off);
    local_s2 += __shfl_xor_sync(0xffffffff, local_s2, off);
  }

  __syncthreads();
  if (lane == 0) {
    smem[warp_id] = local_s1;
    smem[n_warps + warp_id] = local_s2;
  }
  __syncthreads();

  if (warp_id == 0) {
    float s = (lane < n_warps) ? smem[lane] : 0.f;
    float q = (lane < n_warps) ? smem[n_warps + lane] : 0.f;
    for (int off = 16; off > 0; off >>= 1) {
      s += __shfl_xor_sync(0xffffffff, s, off);
      q += __shfl_xor_sync(0xffffffff, q, off);
    }
    if (lane == 0) {
      smem[0] = s;
      smem[n_warps] = q;
    }
  }
  __syncthreads();

  const float sum1 = smem[0];
  const float sum2 = smem[n_warps];

  // Pass 3: vectorized dx write (dgamma/dbeta done in separate kernel).
  for (int64_t i = tid; i < n_vec; i += n_threads) {
    bf16x4_t dyv = DY[i];
    bf16x4_t xv  = X[i];
    bf16x4_t gv;
    if (G != nullptr) gv = G[i];
    bf16x4_t out;
    _Pragma("unroll")
    for (int k = 0; k < VEC; ++k) {
      float dy_v = __bfloat162float(dyv.v[k]);
      float x_v  = __bfloat162float(xv.v[k]);
      float g_v  = (G != nullptr) ? __bfloat162float(gv.v[k]) : 1.f;
      float xn   = (x_v - mean) * inv_std;
      float dx_v = (dy_v * g_v - sum1 * inv_norm - xn * sum2 * inv_norm) * inv_std;
      out.v[k] = __float2bfloat16_rn(dx_v);
    }
    DX[i] = out;
  }
}

// 2026-05-12 perf: cross-row reduction for dgamma and dbeta (BF16 LayerNorm bwd).
//   dgamma[j] = sum_r (dy[r,j] * xn[r,j])    where xn[r,j] = (x[r,j]-mean[r])/std[r]
//   dbeta[j]  = sum_r (dy[r,j])
// Tile pattern: each block handles COLS_PER_BLOCK=64 contiguous columns × a
// ROWS_PER_BLOCK=128 row chunk; threads in the block collaboratively reduce
// each row's mean/var (so per-row stats are computed ONCE per block, not
// per column thread), then accumulate per-column dgamma/dbeta. One atomicAdd
// per column per row-chunk.
__global__ void layer_norm_grad_weight_bias_bf16_vec_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ dy,
    int64_t batch_size,
    int64_t norm_size,
    float eps,
    bool has_dgamma,
    bool has_dbeta,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta) {
  const int COLS_PER_BLOCK = 64;
  const int ROWS_PER_BLOCK = 128;
  const int tid = threadIdx.x;
  const int col_tile = blockIdx.x * COLS_PER_BLOCK;
  const int col = col_tile + tid;

  const int row_start = blockIdx.y * ROWS_PER_BLOCK;
  int row_end = row_start + ROWS_PER_BLOCK;
  if (row_end > batch_size) row_end = batch_size;

  const float inv_norm = 1.f / static_cast<float>(norm_size);

  // Shared memory for broadcasting per-row (mean, inv_std) from the
  // collaborative reduction to all column threads.
  __shared__ float s_mean;
  __shared__ float s_inv_std;
  // Per-warp partial sums for the row reduction (2 warps with 64 threads).
  __shared__ float s_partial_sum[2];
  __shared__ float s_partial_sq[2];

  float acc_g = 0.f;
  float acc_b = 0.f;

  for (int r = row_start; r < row_end; ++r) {
    const int64_t row_off = static_cast<int64_t>(r) * norm_size;

    // Pass A: collaborative sum / sum_sq across the WHOLE row using all 64
    // threads. Each thread reads norm_size/64 elements (e.g. 40 for 2560).
    float local_sum = 0.f;
    float local_sq  = 0.f;
    for (int64_t i = tid; i < norm_size; i += COLS_PER_BLOCK) {
      float v = __bfloat162float(x[row_off + i]);
      local_sum += v;
      local_sq  += v * v;
    }

    // Intra-warp reduction (warp shuffle).
    for (int off = 16; off > 0; off >>= 1) {
      local_sum += __shfl_xor_sync(0xffffffff, local_sum, off);
      local_sq  += __shfl_xor_sync(0xffffffff, local_sq,  off);
    }

    // Inter-warp (2 warps): lane 0 of each warp writes to smem, then warp 0
    // sums the 2 partials.
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    if (lane == 0) {
      s_partial_sum[warp_id] = local_sum;
      s_partial_sq[warp_id]  = local_sq;
    }
    __syncthreads();

    if (tid == 0) {
      float total_sum = s_partial_sum[0] + s_partial_sum[1];
      float total_sq  = s_partial_sq[0]  + s_partial_sq[1];
      float mean = total_sum * inv_norm;
      float var  = total_sq  * inv_norm - mean * mean;
      s_mean    = mean;
      s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    const float mean    = s_mean;
    const float inv_std = s_inv_std;

    // Pass B: each thread accumulates its column's contribution at row r.
    if (col < norm_size) {
      const float dy_v = __bfloat162float(dy[row_off + col]);
      const float x_v  = __bfloat162float(x[row_off + col]);
      const float xn   = (x_v - mean) * inv_std;
      if (has_dgamma) acc_g += dy_v * xn;
      if (has_dbeta)  acc_b += dy_v;
    }
    __syncthreads();  // protect smem before next iter
  }

  if (col < norm_size) {
    if (has_dgamma) atomicAdd(&dgamma[col], acc_g);
    if (has_dbeta)  atomicAdd(&dbeta[col],  acc_b);
  }
}

__global__ void layer_norm_backward_kernel(const __nv_bfloat16* x,
                                           const __nv_bfloat16* dy,
                                           const __nv_bfloat16* gamma,
                                           int64_t norm_size,
                                           float eps,
                                           bool has_gamma,
                                           bool has_dgamma,
                                           bool has_dbeta,
                                           __nv_bfloat16* dx,
                                           float* dgamma,
                                           float* dbeta) {
  const int64_t row = blockIdx.x;
  const int64_t offset = row * norm_size;

  float mean = 0.f;
  for (int64_t i = 0; i < norm_size; ++i) {
    mean += bf16_to_f32(x[offset + i]);
  }
  const float inv_norm = 1.f / static_cast<float>(norm_size);
  mean *= inv_norm;

  float var = 0.f;
  for (int64_t i = 0; i < norm_size; ++i) {
    float v = bf16_to_f32(x[offset + i]) - mean;
    var += v * v;
  }
  var *= inv_norm;
  const float inv_std = rsqrtf(var + eps);

  float sum1 = 0.f;
  float sum2 = 0.f;
  for (int64_t i = 0; i < norm_size; ++i) {
    const float g = has_gamma ? bf16_to_f32(gamma[i]) : 1.f;
    const float dy_val = bf16_to_f32(dy[offset + i]);
    const float xn = (bf16_to_f32(x[offset + i]) - mean) * inv_std;
    sum1 += dy_val * g;
    sum2 += dy_val * g * xn;
  }

  const float inv_N = inv_norm;
  for (int64_t i = 0; i < norm_size; ++i) {
    const float g = has_gamma ? bf16_to_f32(gamma[i]) : 1.f;
    const float dy_val = bf16_to_f32(dy[offset + i]);
    const float xn = (bf16_to_f32(x[offset + i]) - mean) * inv_std;
    const float dx_val = (dy_val * g - sum1 * inv_N - xn * sum2 * inv_N) * inv_std;
    dx[offset + i] = f32_to_bf16(dx_val);

    if (has_dgamma) {
      atomicAdd(dgamma + i, dy_val * xn);
    }
    if (has_dbeta) {
      atomicAdd(dbeta + i, dy_val);
    }
  }
}

__global__ void group_norm_backward_kernel(const __nv_bfloat16* x,
                                           const __nv_bfloat16* dy,
                                           const __nv_bfloat16* gamma,
                                           int64_t channels,
                                           int64_t spatial_size,
                                           int32_t groups,
                                           float eps,
                                           bool has_gamma,
                                           bool has_dgamma,
                                           bool has_dbeta,
                                           __nv_bfloat16* dx,
                                           float* dgamma,
                                           float* dbeta) {
  const int64_t group_index = blockIdx.x;
  const int64_t n = group_index / groups;
  const int32_t g = static_cast<int32_t>(group_index % groups);

  const int64_t channels_per_group = channels / groups;
  const int64_t group_elements = channels_per_group * spatial_size;
  const int64_t base =
      n * channels * spatial_size + static_cast<int64_t>(g) * channels_per_group * spatial_size;

  float mean = 0.f;
  for (int64_t c = 0; c < channels_per_group; ++c) {
    const int64_t channel_offset = base + c * spatial_size;
    for (int64_t s = 0; s < spatial_size; ++s) {
      mean += bf16_to_f32(x[channel_offset + s]);
    }
  }
  const float inv_count = 1.f / static_cast<float>(group_elements);
  mean *= inv_count;

  float var = 0.f;
  for (int64_t c = 0; c < channels_per_group; ++c) {
    const int64_t channel_offset = base + c * spatial_size;
    for (int64_t s = 0; s < spatial_size; ++s) {
      float val = bf16_to_f32(x[channel_offset + s]) - mean;
      var += val * val;
    }
  }
  var *= inv_count;
  const float inv_std = rsqrtf(var + eps);

  float sum1 = 0.f;
  float sum2 = 0.f;
  for (int64_t c = 0; c < channels_per_group; ++c) {
    const int64_t channel_index = static_cast<int64_t>(g) * channels_per_group + c;
    const int64_t channel_offset = base + c * spatial_size;
    const float gamma_val = has_gamma ? bf16_to_f32(gamma[channel_index]) : 1.f;
    for (int64_t s = 0; s < spatial_size; ++s) {
      const float dy_val = bf16_to_f32(dy[channel_offset + s]);
      const float xn = (bf16_to_f32(x[channel_offset + s]) - mean) * inv_std;
      sum1 += dy_val * gamma_val;
      sum2 += dy_val * gamma_val * xn;
    }
  }

  for (int64_t c = 0; c < channels_per_group; ++c) {
    const int64_t channel_index = static_cast<int64_t>(g) * channels_per_group + c;
    const float gamma_val = has_gamma ? bf16_to_f32(gamma[channel_index]) : 1.f;
    const int64_t channel_offset = base + c * spatial_size;
    for (int64_t s = 0; s < spatial_size; ++s) {
      const float dy_val = bf16_to_f32(dy[channel_offset + s]);
      const float xn = (bf16_to_f32(x[channel_offset + s]) - mean) * inv_std;
      const float dx_val =
          (dy_val * gamma_val - sum1 * inv_count - xn * sum2 * inv_count) * inv_std;
      dx[channel_offset + s] = f32_to_bf16(dx_val);
      if (has_dgamma) {
        atomicAdd(dgamma + channel_index, dy_val * xn);
      }
      if (has_dbeta) {
        atomicAdd(dbeta + channel_index, dy_val);
      }
    }
  }
}

inline fc_status_t memset_if_needed(void* ptr, size_t bytes, cudaStream_t stream) {
  if (ptr == nullptr || bytes == 0) {
    return FC_OK;
  }
  cudaError_t err = cudaMemsetAsync(ptr, 0, bytes, stream);
  if (err == cudaErrorMemoryAllocation) {
    return FC_ERR_OOM;
  }
  return err == cudaSuccess ? FC_OK : FC_ERR_LAUNCH;
}

}  // namespace

extern "C" fc_status_t fc_layer_norm_backward_bf16(const __nv_bfloat16* x,
                                                   const __nv_bfloat16* dy,
                                                   const __nv_bfloat16* gamma,
                                                   int64_t outer_size,
                                                   int64_t norm_size,
                                                   float eps,
                                                   __nv_bfloat16* dx,
                                                   float* dgamma,
                                                   float* dbeta,
                                                   cudaStream_t stream) {
  if (outer_size <= 0 || norm_size <= 0 || x == nullptr || dy == nullptr || dx == nullptr) {
    return FC_ERR_INVALID_ARGUMENT;
  }

  FC_RETURN_IF_ERROR(
      memset_if_needed(dgamma, sizeof(float) * static_cast<size_t>(norm_size), stream));
  FC_RETURN_IF_ERROR(memset_if_needed(dbeta, sizeof(float) * static_cast<size_t>(norm_size), stream));

  const bool has_gamma = gamma != nullptr;
  const bool has_dgamma = dgamma != nullptr;
  const bool has_dbeta = dbeta != nullptr;

  // 2026-05-12 perf: dispatch to vectorized backward when norm_size % 4 == 0
  // (production diffusion shapes 2560/3072/4096 all qualify). Env override
  // FLAME_LAYER_NORM_LEGACY=1 forces the legacy single-thread-per-row path
  // for A/B benchmarking. Same gating pattern as RMSNorm vec dispatch.
  const char* legacy_env = getenv("FLAME_LAYER_NORM_LEGACY");
  const bool force_legacy = (legacy_env != nullptr && legacy_env[0] != 0 && legacy_env[0] != '0');
  const bool use_vec = (norm_size % 4 == 0) && !force_legacy;

  if (use_vec) {
    // Vec backward writes only dx; dgamma/dbeta done in separate cross-row kernel.
    const int block_threads = 256;
    const int n_warps = (block_threads + 31) >> 5;
    // Smem layout: 2 * n_warps floats (sum + sq partials, reused for s1+s2).
    const size_t smem_bytes = 2 * static_cast<size_t>(n_warps) * sizeof(float);
    layer_norm_backward_bf16_vec_kernel<<<static_cast<int>(outer_size), block_threads, smem_bytes, stream>>>(
        x, dy, gamma, norm_size, eps, has_gamma, dx);
    if (cudaGetLastError() != cudaSuccess) return FC_ERR_LAUNCH;

    if (has_dgamma || has_dbeta) {
      const int COLS_PER_BLOCK = 64;
      const int ROWS_PER_BLOCK = 128;
      const int cols_blocks = static_cast<int>((norm_size + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK);
      const int rows_blocks = static_cast<int>((outer_size + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
      dim3 grid(cols_blocks, rows_blocks, 1);
      dim3 block(COLS_PER_BLOCK, 1, 1);
      layer_norm_grad_weight_bias_bf16_vec_kernel<<<grid, block, 0, stream>>>(
          x, dy, outer_size, norm_size, eps, has_dgamma, has_dbeta, dgamma, dbeta);
      if (cudaGetLastError() != cudaSuccess) return FC_ERR_LAUNCH;
    }
  } else {
    layer_norm_backward_kernel<<<static_cast<int>(outer_size), 1, 0, stream>>>(
        x, dy, gamma, norm_size, eps, has_gamma, has_dgamma, has_dbeta, dx, dgamma, dbeta);
    if (cudaGetLastError() != cudaSuccess) return FC_ERR_LAUNCH;
  }
  return FC_OK;
}

extern "C" fc_status_t fc_group_norm_backward_bf16(const __nv_bfloat16* x,
                                                   const __nv_bfloat16* dy,
                                                   const __nv_bfloat16* gamma,
                                                   int64_t batch_size,
                                                   int64_t channels,
                                                   int64_t spatial_size,
                                                   int32_t group_count,
                                                   float eps,
                                                   __nv_bfloat16* dx,
                                                   float* dgamma,
                                                   float* dbeta,
                                                   cudaStream_t stream) {
  if (batch_size <= 0 || channels <= 0 || spatial_size <= 0 || group_count <= 0 ||
      x == nullptr || dy == nullptr || dx == nullptr) {
    return FC_ERR_INVALID_ARGUMENT;
  }
  if (channels % group_count != 0) {
    return FC_ERR_INVALID_ARGUMENT;
  }

  FC_RETURN_IF_ERROR(
      memset_if_needed(dgamma, sizeof(float) * static_cast<size_t>(channels), stream));
  FC_RETURN_IF_ERROR(memset_if_needed(dbeta, sizeof(float) * static_cast<size_t>(channels), stream));

  const bool has_gamma = gamma != nullptr;
  const bool has_dgamma = dgamma != nullptr;
  const bool has_dbeta = dbeta != nullptr;

  const int64_t total_groups = batch_size * static_cast<int64_t>(group_count);
  group_norm_backward_kernel<<<static_cast<int>(total_groups), 1, 0, stream>>>(
      x,
      dy,
      gamma,
      channels,
      spatial_size,
      group_count,
      eps,
      has_gamma,
      has_dgamma,
      has_dbeta,
      dx,
      dgamma,
      dbeta);

  if (cudaGetLastError() != cudaSuccess) {
    return FC_ERR_LAUNCH;
  }
  return FC_OK;
}

