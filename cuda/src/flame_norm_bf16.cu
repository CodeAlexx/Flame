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

  layer_norm_backward_kernel<<<static_cast<int>(outer_size), 1, 0, stream>>>(
      x, dy, gamma, norm_size, eps, has_gamma, has_dgamma, has_dbeta, dx, dgamma, dbeta);

  if (cudaGetLastError() != cudaSuccess) {
    return FC_ERR_LAUNCH;
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

