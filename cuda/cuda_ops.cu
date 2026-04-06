#include "cuda_ops.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <math.h>
#include <limits>

#define FC_RETURN_IF_ERROR(stmt) \
  do { \
    fc_status_t _status = (stmt); \
    if (_status != FC_OK) { \
      return _status; \
    } \
  } while (0)

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

namespace {

__device__ inline __nv_bfloat16 bf16_from_f32(float x) {
  return __float2bfloat16(x);
}

__device__ inline float f32_from_bf16(__nv_bfloat16 h) {
  return __bfloat162float(h);
}

__constant__ float kGeluConst = 0.044715f;

struct TensorIter {
  const __nv_bfloat16* data;
  const int64_t* strides;
  int64_t index;
  int32_t rank;
  int64_t dims[8];

  __device__ float load(const int64_t* coords) const {
    int64_t offset = 0;
    for (int i = 0; i < rank; ++i) {
      offset += coords[i] * strides[i];
    }
    return f32_from_bf16(data[offset]);
  }
};

struct TensorWriter {
  __nv_bfloat16* data;
  const int64_t* strides;
  int32_t rank;

  __device__ void store(const int64_t* coords, float value) const {
    int64_t offset = 0;
    for (int i = 0; i < rank; ++i) {
      offset += coords[i] * strides[i];
    }
    data[offset] = bf16_from_f32(value);
  }
};

inline fc_status_t check_tensor(const fc_tensor_view_t* view) {
  if (!view || !view->data) {
    return FC_ERR_INVALID_ARGUMENT;
  }
  if (view->rank < 1 || view->rank > 8) {
    return FC_ERR_INVALID_ARGUMENT;
  }
  for (int32_t i = 0; i < view->rank; ++i) {
    if (view->dims[i] <= 0) {
      return FC_ERR_INVALID_ARGUMENT;
    }
  }
  return FC_OK;
}

inline fc_status_t launch_grid(size_t n, dim3* grid, dim3* block) {
  if (n == 0) {
    grid->x = grid->y = grid->z = 1;
    block->x = block->y = block->z = 1;
    return FC_OK;
  }
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  grid->x = blocks;
  grid->y = 1;
  grid->z = 1;
  block->x = threads;
  block->y = 1;
  block->z = 1;
  return FC_OK;
}

}  // namespace

// -----------------------------------------------------------------------------
// Elementwise kernels
// -----------------------------------------------------------------------------

namespace {

__global__ void relu_kernel(const __nv_bfloat16* x, __nv_bfloat16* y, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < n) {
    float v = f32_from_bf16(x[idx]);
    y[idx] = bf16_from_f32(v > 0.f ? v : 0.f);
    idx += gridDim.x * blockDim.x;
  }
}

__global__ void silu_kernel(const __nv_bfloat16* x, __nv_bfloat16* y, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < n) {
    float v = f32_from_bf16(x[idx]);
    float s = 1.f / (1.f + expf(-v));
    y[idx] = bf16_from_f32(v * s);
    idx += gridDim.x * blockDim.x;
  }
}

__global__ void gelu_kernel(const __nv_bfloat16* x, __nv_bfloat16* y, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < n) {
    float v = f32_from_bf16(x[idx]);
    float u = v + kGeluConst * v * v * v;
    float s = 0.5f * v * (1.f + tanhf(0.7978845608f * u));
    y[idx] = bf16_from_f32(s);
    idx += gridDim.x * blockDim.x;
  }
}

__global__ void axpby_kernel(const __nv_bfloat16* x, float a, __nv_bfloat16* y, float b, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < n) {
    float vx = f32_from_bf16(x[idx]);
    float vy = f32_from_bf16(y[idx]);
    y[idx] = bf16_from_f32(a * vx + b * vy);
    idx += gridDim.x * blockDim.x;
  }
}

}  // namespace

template <typename Kernel>
fc_status_t launch_unary_elementwise(const fc_tensor_view_t* x, fc_tensor_view_t* y, cudaStream_t stream, Kernel kernel) {
  FC_RETURN_IF_ERROR(check_tensor(x));
  FC_RETURN_IF_ERROR(check_tensor(y));
  size_t n = 1;
  for (int32_t i = 0; i < x->rank; ++i) {
    n *= static_cast<size_t>(x->dims[i]);
  }
  dim3 grid, block;
  FC_RETURN_IF_ERROR(launch_grid(n, &grid, &block));
  if (n == 0) {
    return FC_OK;
  }
  kernel<<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x->data),
      static_cast<__nv_bfloat16*>(y->data),
      n);
  if (cudaGetLastError() != cudaSuccess) {
    return FC_ERR_LAUNCH;
  }
  return FC_OK;
}

extern "C" fc_status_t fc_relu_bf16(const fc_tensor_view_t* x, fc_tensor_view_t* y, cudaStream_t stream) {
  return launch_unary_elementwise(x, y, stream, relu_kernel);
}

extern "C" fc_status_t fc_gelu_bf16(const fc_tensor_view_t* x, fc_tensor_view_t* y, cudaStream_t stream) {
  return launch_unary_elementwise(x, y, stream, gelu_kernel);
}

extern "C" fc_status_t fc_silu_bf16(const fc_tensor_view_t* x, fc_tensor_view_t* y, cudaStream_t stream) {
  return launch_unary_elementwise(x, y, stream, silu_kernel);
}

extern "C" fc_status_t fc_axpby_bf16(const fc_tensor_view_t* x, float a, fc_tensor_view_t* y, float b, cudaStream_t stream) {
  FC_RETURN_IF_ERROR(check_tensor(x));
  FC_RETURN_IF_ERROR(check_tensor(y));
  size_t n = 1;
  for (int32_t i = 0; i < x->rank; ++i) {
    n *= static_cast<size_t>(x->dims[i]);
  }
  dim3 grid, block;
  FC_RETURN_IF_ERROR(launch_grid(n, &grid, &block));
  if (n == 0) {
    return FC_OK;
  }
  axpby_kernel<<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x->data),
      a,
      static_cast<__nv_bfloat16*>(y->data),
      b,
      n);
  if (cudaGetLastError() != cudaSuccess) {
    return FC_ERR_LAUNCH;
  }
  return FC_OK;
}

// -----------------------------------------------------------------------------
// Norms (layer / RMS / group) — placeholder sum implementation
// -----------------------------------------------------------------------------

namespace {

__global__ void rms_norm_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight, __nv_bfloat16* y,
                                int64_t outer, int64_t channels, float eps) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < outer) {
    const __nv_bfloat16* x_ptr = x + idx * channels;
    __nv_bfloat16* y_ptr = y + idx * channels;
    float mean = 0.0f;
    float inv = 0.0f;
    for (int64_t c = 0; c < channels; ++c) {
      float v = f32_from_bf16(x_ptr[c]);
      mean += v * v;
    }
    mean /= static_cast<float>(channels);
    inv = rsqrtf(mean + eps);
    for (int64_t c = 0; c < channels; ++c) {
      float v = f32_from_bf16(x_ptr[c]);
      float scaled = v * inv;
      if (weight) {
        scaled *= f32_from_bf16(weight[c]);
      }
      y_ptr[c] = bf16_from_f32(scaled);
    }
    idx += gridDim.x * blockDim.x;
  }
}

__global__ void layer_norm_forward_bf16_kernel(const __nv_bfloat16* input,
                                                const __nv_bfloat16* weight,
                                                const __nv_bfloat16* bias,
                                                __nv_bfloat16* output,
                                                float* mean_out,
                                                float* rstd_out,
                                                int64_t norm_size,
                                                float eps) {
  extern __shared__ float shared[];
  float* shared_sum = shared;
  float* shared_sq = shared + blockDim.x;

  int row = blockIdx.x;
  int tid = threadIdx.x;
  int64_t offset = static_cast<int64_t>(row) * norm_size;

  float local_sum = 0.0f;
  float local_sq = 0.0f;
  for (int64_t i = tid; i < norm_size; i += blockDim.x) {
    float val = __bfloat162float(input[offset + i]);
    local_sum += val;
    local_sq += val * val;
  }
  shared_sum[tid] = local_sum;
  shared_sq[tid] = local_sq;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
      shared_sq[tid] += shared_sq[tid + stride];
    }
    __syncthreads();
  }

  float mean;
  float inv_std;
  if (tid == 0) {
    float total = shared_sum[0];
    float total_sq = shared_sq[0];
    float inv_norm = 1.0f / static_cast<float>(norm_size);
    mean = total * inv_norm;
    float var = total_sq * inv_norm - mean * mean;
    inv_std = rsqrtf(var + eps);
    mean_out[row] = mean;
    rstd_out[row] = inv_std;
    shared_sum[0] = mean;
    shared_sq[0] = inv_std;
  }
  __syncthreads();

  mean = shared_sum[0];
  inv_std = shared_sq[0];

  for (int64_t i = tid; i < norm_size; i += blockDim.x) {
    float val = (__bfloat162float(input[offset + i]) - mean) * inv_std;
    if (weight) {
      val *= __bfloat162float(weight[i]);
    }
    if (bias) {
      val += __bfloat162float(bias[i]);
    }
    output[offset + i] = __float2bfloat16_rn(val);
  }
}

__global__ void group_norm_compute_stats_bf16_kernel(const __nv_bfloat16* input,
                                                     float* mean_out,
                                                     float* var_out,
                                                     int batch_size,
                                                     int channels,
                                                     int groups,
                                                     int channels_per_group,
                                                     int spatial_size) {
  extern __shared__ float shared_data[];
  float* shared_sum = shared_data;
  float* shared_sum_sq = shared_data + blockDim.x;

  int tid = threadIdx.x;
  int group_id = blockIdx.x;
  int total_groups = batch_size * groups;
  if (group_id >= total_groups) {
    return;
  }

  int n = group_id / groups;
  int g = group_id % groups;
  int elements_per_group = channels_per_group * spatial_size;

  float local_sum = 0.0f;
  float local_sum_sq = 0.0f;

  for (int idx = tid; idx < elements_per_group; idx += blockDim.x) {
    int c_offset = idx / spatial_size;
    int hw_offset = idx % spatial_size;
    int c = g * channels_per_group + c_offset;
    int input_idx = ((n * channels + c) * spatial_size) + hw_offset;
    float val = __bfloat162float(input[input_idx]);
    local_sum += val;
    local_sum_sq += val * val;
  }

  shared_sum[tid] = local_sum;
  shared_sum_sq[tid] = local_sum_sq;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
      shared_sum_sq[tid] += shared_sum_sq[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    float count = static_cast<float>(elements_per_group);
    float mean = shared_sum[0] / count;
    float var = shared_sum_sq[0] / count - mean * mean;
    mean_out[group_id] = mean;
    var_out[group_id] = var;
  }
}

__global__ void group_norm_forward_bf16_kernel(const __nv_bfloat16* input,
                                               __nv_bfloat16* output,
                                               const __nv_bfloat16* weight,
                                               const __nv_bfloat16* bias,
                                               const float* mean,
                                               const float* var,
                                               int batch_size,
                                               int channels,
                                               int groups,
                                               int channels_per_group,
                                               int spatial_size,
                                               float eps,
                                               int has_weight,
                                               int has_bias) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(batch_size) * channels * spatial_size;
  if (static_cast<int64_t>(idx) >= total) {
    return;
  }

  int channel_stride = spatial_size;
  int batch_stride = channels * spatial_size;

  int n = idx / batch_stride;
  int c = (idx / channel_stride) % channels;
  int hw = idx % spatial_size;
  int g = c / channels_per_group;
  int mean_index = n * groups + g;

  float m = mean[mean_index];
  float v = var[mean_index];
  float inv_std = rsqrtf(v + eps);

  int offset = ((n * channels + c) * spatial_size) + hw;
  float value = __bfloat162float(input[offset]);
  float normalized = (value - m) * inv_std;

  if (has_weight) {
    normalized *= __bfloat162float(weight[c]);
  }
  if (has_bias) {
    normalized += __bfloat162float(bias[c]);
  }

  output[offset] = __float2bfloat16_rn(normalized);
}

}  // namespace

// RMS norm: BF16 input → F32 output (for Gemma3-style (1+w) multiply in F32)
__global__ void rms_norm_bf16_to_f32_kernel(const __nv_bfloat16* x, float* y,
                                             int64_t outer, int64_t channels, float eps) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < outer) {
    const __nv_bfloat16* x_ptr = x + idx * channels;
    float* y_ptr = y + idx * channels;
    float mean = 0.0f;
    for (int64_t c = 0; c < channels; ++c) {
      float v = f32_from_bf16(x_ptr[c]);
      mean += v * v;
    }
    mean /= static_cast<float>(channels);
    float inv = rsqrtf(mean + eps);
    for (int64_t c = 0; c < channels; ++c) {
      float v = f32_from_bf16(x_ptr[c]);
      y_ptr[c] = v * inv;  // F32 output — no BF16 truncation
    }
    idx += gridDim.x * blockDim.x;
  }
}

extern "C" fc_status_t fc_rms_norm_bf16_to_f32(const fc_tensor_view_t* x,
                                                 float eps,
                                                 fc_tensor_view_t* y,
                                                 cudaStream_t stream) {
  FC_RETURN_IF_ERROR(check_tensor(x));
  FC_RETURN_IF_ERROR(check_tensor(y));
  const int64_t channels = x->dims[x->rank - 1];
  int64_t outer = 1;
  for (int32_t i = 0; i < x->rank - 1; ++i) {
    outer *= x->dims[i];
  }
  dim3 grid, block;
  FC_RETURN_IF_ERROR(launch_grid(static_cast<size_t>(outer), &grid, &block));
  if (outer == 0) {
    return FC_OK;
  }
  rms_norm_bf16_to_f32_kernel<<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x->data),
      static_cast<float*>(y->data), outer, channels, eps);
  if (cudaGetLastError() != cudaSuccess) {
    return FC_ERR_LAUNCH;
  }
  return FC_OK;
}

extern "C" fc_status_t fc_rms_norm_bf16(const fc_tensor_view_t* x,
                                         const fc_tensor_view_t* weight,
                                         float eps,
                                         fc_tensor_view_t* y,
                                         cudaStream_t stream) {
  FC_RETURN_IF_ERROR(check_tensor(x));
  FC_RETURN_IF_ERROR(check_tensor(y));
  const int64_t channels = x->dims[x->rank - 1];
  int64_t outer = 1;
  for (int32_t i = 0; i < x->rank - 1; ++i) {
    outer *= x->dims[i];
  }
  dim3 grid, block;
  FC_RETURN_IF_ERROR(launch_grid(static_cast<size_t>(outer), &grid, &block));
  if (outer == 0) {
    return FC_OK;
  }
  rms_norm_kernel<<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x->data),
      weight ? static_cast<const __nv_bfloat16*>(weight->data) : nullptr,
      static_cast<__nv_bfloat16*>(y->data), outer, channels, eps);
  if (cudaGetLastError() != cudaSuccess) {
    return FC_ERR_LAUNCH;
  }
  return FC_OK;
}

extern "C" fc_status_t fc_layer_norm_bf16(const fc_tensor_view_t* x,
                                           const fc_tensor_view_t* gamma,
                                           const fc_tensor_view_t* beta,
                                           int64_t norm_size,
                                           float eps,
                                           fc_tensor_view_t* y,
                                           float* mean_out,
                                           float* rstd_out,
                                           cudaStream_t stream) {
  FC_RETURN_IF_ERROR(check_tensor(x));
  FC_RETURN_IF_ERROR(check_tensor(y));
  if (!x->data || !y->data || !mean_out || !rstd_out) {
    return FC_ERR_INVALID_ARGUMENT;
  }
  if (norm_size <= 0) {
    return FC_ERR_INVALID_ARGUMENT;
  }

  if (x->rank != y->rank) {
    return FC_ERR_INVALID_ARGUMENT;
  }
  for (int32_t i = 0; i < x->rank; ++i) {
    if (x->dims[i] != y->dims[i]) {
      return FC_ERR_INVALID_ARGUMENT;
    }
  }

  int64_t total_elems = 1;
  for (int32_t i = 0; i < x->rank; ++i) {
    if (x->dims[i] <= 0) {
      return FC_ERR_INVALID_ARGUMENT;
    }
    total_elems *= x->dims[i];
  }

  if (total_elems % norm_size != 0) {
    return FC_ERR_INVALID_ARGUMENT;
  }
  int64_t outer = total_elems / norm_size;
  if (outer <= 0 || outer > std::numeric_limits<int>::max()) {
    return FC_ERR_UNSUPPORTED;
  }

  if (gamma) {
    int64_t gamma_elems = 1;
    for (int32_t i = 0; i < gamma->rank; ++i) {
      if (gamma->dims[i] <= 0) {
        return FC_ERR_INVALID_ARGUMENT;
      }
      gamma_elems *= gamma->dims[i];
    }
    if (gamma_elems != norm_size) {
      return FC_ERR_INVALID_ARGUMENT;
    }
  }
  if (beta) {
    int64_t beta_elems = 1;
    for (int32_t i = 0; i < beta->rank; ++i) {
      if (beta->dims[i] <= 0) {
        return FC_ERR_INVALID_ARGUMENT;
      }
      beta_elems *= beta->dims[i];
    }
    if (beta_elems != norm_size) {
      return FC_ERR_INVALID_ARGUMENT;
    }
  }

  const __nv_bfloat16* input = static_cast<const __nv_bfloat16*>(x->data);
  __nv_bfloat16* output = static_cast<__nv_bfloat16*>(y->data);
  const __nv_bfloat16* weight =
      gamma ? static_cast<const __nv_bfloat16*>(gamma->data) : nullptr;
  const __nv_bfloat16* bias =
      beta ? static_cast<const __nv_bfloat16*>(beta->data) : nullptr;

  int block_threads = 1;
  if (norm_size >= 256) {
    block_threads = 256;
  } else if (norm_size >= 128) {
    block_threads = 128;
  } else if (norm_size >= 64) {
    block_threads = 64;
  } else if (norm_size >= 32) {
    block_threads = 32;
  } else if (norm_size >= 16) {
    block_threads = 16;
  } else if (norm_size >= 8) {
    block_threads = 8;
  } else if (norm_size >= 4) {
    block_threads = 4;
  } else if (norm_size >= 2) {
    block_threads = 2;
  }
  size_t shared_bytes =
      static_cast<size_t>(block_threads) * 2 * sizeof(float);

  layer_norm_forward_bf16_kernel<<<static_cast<int>(outer), block_threads, shared_bytes, stream>>>(
      input,
      weight,
      bias,
      output,
      mean_out,
      rstd_out,
      norm_size,
      eps);

  if (cudaGetLastError() != cudaSuccess) {
    return FC_ERR_LAUNCH;
  }
  return FC_OK;
}

extern "C" fc_status_t fc_group_norm_bf16(const fc_tensor_view_t* x,
                                           const fc_tensor_view_t* gamma,
                                           const fc_tensor_view_t* beta,
                                           int32_t groups,
                                           float eps,
                                           fc_tensor_view_t* y,
                                           float* mean_out,
                                           float* var_out,
                                           cudaStream_t stream) {
  FC_RETURN_IF_ERROR(check_tensor(x));
  FC_RETURN_IF_ERROR(check_tensor(y));
  if (x->rank != 4 || y->rank != 4) {
    return FC_ERR_INVALID_ARGUMENT;
  }
  if (!x->data || !y->data) {
    return FC_ERR_INVALID_ARGUMENT;
  }
  if (groups <= 0) {
    return FC_ERR_INVALID_ARGUMENT;
  }

  const int64_t batch = x->dims[0];
  const int64_t channels = x->dims[1];
  const int64_t height = x->dims[2];
  const int64_t width = x->dims[3];

  if (batch <= 0 || channels <= 0 || height <= 0 || width <= 0) {
    return FC_ERR_INVALID_ARGUMENT;
  }
  if (channels % groups != 0) {
    return FC_ERR_INVALID_ARGUMENT;
  }

  if (y->dims[0] != batch || y->dims[1] != channels || y->dims[2] != height || y->dims[3] != width) {
    return FC_ERR_INVALID_ARGUMENT;
  }

  auto is_contiguous = [](const fc_tensor_view_t* t) -> bool {
    if (!t) {
      return true;
    }
    int32_t rank = t->rank;
    if (rank <= 0) {
      return false;
    }
    int64_t stride = 1;
    for (int32_t i = rank - 1; i >= 0; --i) {
      if (t->strides[i] != stride) {
        return false;
      }
      stride *= t->dims[i];
    }
    return true;
  };

  if (!is_contiguous(x) || !is_contiguous(y)) {
    return FC_ERR_UNSUPPORTED;
  }
  if (gamma && (!is_contiguous(gamma) || gamma->rank != 1 || gamma->dims[0] != channels)) {
    return FC_ERR_INVALID_ARGUMENT;
  }
  if (beta && (!is_contiguous(beta) || beta->rank != 1 || beta->dims[0] != channels)) {
    return FC_ERR_INVALID_ARGUMENT;
  }

  const int64_t spatial = height * width;
  const int64_t channels_per_group = channels / groups;
  const int64_t stats_elems = batch * static_cast<int64_t>(groups);
  if (stats_elems <= 0) {
    return FC_ERR_INVALID_ARGUMENT;
  }
  if (stats_elems > std::numeric_limits<int>::max()) {
    return FC_ERR_UNSUPPORTED;
  }

  const int64_t total_elems = batch * channels * spatial;
  if (total_elems > std::numeric_limits<int>::max()) {
    return FC_ERR_UNSUPPORTED;
  }

  if (!mean_out || !var_out) {
    return FC_ERR_INVALID_ARGUMENT;
  }

  const __nv_bfloat16* input = static_cast<const __nv_bfloat16*>(x->data);
  __nv_bfloat16* output = static_cast<__nv_bfloat16*>(y->data);
  const __nv_bfloat16* weight = gamma ? static_cast<const __nv_bfloat16*>(gamma->data) : nullptr;
  const __nv_bfloat16* bias = beta ? static_cast<const __nv_bfloat16*>(beta->data) : nullptr;

  int stats_threads = (spatial > 65536) ? 512 : 256;
  int total_groups = static_cast<int>(stats_elems);
  dim3 stats_grid(total_groups, 1, 1);
  dim3 stats_block(stats_threads, 1, 1);
  size_t stats_shared = static_cast<size_t>(stats_threads) * 2 * sizeof(float);

  group_norm_compute_stats_bf16_kernel<<<stats_grid, stats_block, stats_shared, stream>>>(
      input,
      mean_out,
      var_out,
      static_cast<int>(batch),
      static_cast<int>(channels),
      groups,
      static_cast<int>(channels_per_group),
      static_cast<int>(spatial));

  cudaError_t launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess) {
    return FC_ERR_LAUNCH;
  }

  int norm_threads = (total_elems > 100000000) ? 512 : 256;
  int norm_blocks = static_cast<int>((total_elems + norm_threads - 1) / norm_threads);

  group_norm_forward_bf16_kernel<<<norm_blocks, norm_threads, 0, stream>>>(
      input,
      output,
      weight,
      bias,
      mean_out,
      var_out,
      static_cast<int>(batch),
      static_cast<int>(channels),
      groups,
      static_cast<int>(channels_per_group),
      static_cast<int>(spatial),
      eps,
      weight != nullptr,
      bias != nullptr);

  launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess) {
    return FC_ERR_LAUNCH;
  }
  return FC_OK;
}
