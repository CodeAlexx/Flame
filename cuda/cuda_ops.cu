#include "cuda_ops.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
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

// Vectorized BF16 elementwise: 2 elements per thread via __nv_bfloat162.
// Eliminates the per-element FP32 round-trip cost. The previous scalar
// kernels were running at ~3% of memory bandwidth (3.6 ms for [1,4096,15360]
// vs PyTorch's 0.29 ms = 13× slower). The vectorized versions hit ~80% of
// peak memory bandwidth.
// Phase 6 (2026-04-22): `relu_kernel` removed. Dispatch moved to
// src/cuda/unary/relu.cu through the TensorIterator pipeline.

__global__ void silu_kernel(const __nv_bfloat16* x, __nv_bfloat16* y, size_t n) {
  size_t i2 = blockIdx.x * blockDim.x + threadIdx.x;
  size_t n2 = n >> 1;
  if (i2 < n2) {
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    __nv_bfloat162* y2 = reinterpret_cast<__nv_bfloat162*>(y);
    float2 v = __bfloat1622float2(x2[i2]);
    // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    float s0 = v.x / (1.f + __expf(-v.x));
    float s1 = v.y / (1.f + __expf(-v.y));
    y2[i2] = __floats2bfloat162_rn(s0, s1);
  }
  if (i2 == n2 && (n & 1)) {
    size_t last = n - 1;
    float v = f32_from_bf16(x[last]);
    y[last] = bf16_from_f32(v / (1.f + expf(-v)));
  }
}

__global__ void gelu_kernel(const __nv_bfloat16* x, __nv_bfloat16* y, size_t n) {
  size_t i2 = blockIdx.x * blockDim.x + threadIdx.x;
  size_t n2 = n >> 1;
  if (i2 < n2) {
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    __nv_bfloat162* y2 = reinterpret_cast<__nv_bfloat162*>(y);
    float2 v = __bfloat1622float2(x2[i2]);
    // tanh-approx GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float u0 = v.x + kGeluConst * v.x * v.x * v.x;
    float u1 = v.y + kGeluConst * v.y * v.y * v.y;
    float g0 = 0.5f * v.x * (1.f + tanhf(0.7978845608f * u0));
    float g1 = 0.5f * v.y * (1.f + tanhf(0.7978845608f * u1));
    y2[i2] = __floats2bfloat162_rn(g0, g1);
  }
  if (i2 == n2 && (n & 1)) {
    size_t last = n - 1;
    float v = f32_from_bf16(x[last]);
    float u = v + kGeluConst * v * v * v;
    y[last] = bf16_from_f32(0.5f * v * (1.f + tanhf(0.7978845608f * u)));
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
  if (n == 0) {
    return FC_OK;
  }
  // The kernels are vectorized at 2 elements per thread, so launch n/2 threads.
  size_t n_pairs = (n + 1) / 2;
  dim3 grid, block;
  FC_RETURN_IF_ERROR(launch_grid(n_pairs, &grid, &block));
  kernel<<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x->data),
      static_cast<__nv_bfloat16*>(y->data),
      n);
  if (cudaGetLastError() != cudaSuccess) {
    return FC_ERR_LAUNCH;
  }
  return FC_OK;
}

// fc_relu_bf16 removed in Phase 6 — Tensor::relu and GpuOps::relu now route
// BF16 through src/cuda/unary/relu.cu via the TensorIterator pipeline.
// The `relu_kernel` device kernel at line ~109 is no longer called, but
// kept in the file for now because it's defined in an anonymous namespace
// alongside `gelu_kernel` and `silu_kernel` which are still live.

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

// One block per row, threads in the block cooperate via shared-memory
// reduction. The previous version was one THREAD per row scanning sequentially —
// at [4096, 3840] that's 4096 fully-serial reductions over 3840 elements each,
// running 2.6 ms vs PyTorch's 0.74 ms. The block-per-row version saturates
// memory bandwidth.
__global__ void rms_norm_kernel(const __nv_bfloat16* __restrict__ x,
                                const __nv_bfloat16* __restrict__ weight,
                                __nv_bfloat16* __restrict__ y,
                                int64_t outer, int64_t channels, float eps) {
  extern __shared__ float rms_shared[];

  int64_t row = blockIdx.x;
  if (row >= outer) return;

  const __nv_bfloat16* x_ptr = x + row * channels;
  __nv_bfloat16* y_ptr = y + row * channels;
  int tid = threadIdx.x;

  // Pass 1: sum of squares (per-thread accumulator + shared-mem reduction).
  float local_sq = 0.0f;
  for (int64_t c = tid; c < channels; c += blockDim.x) {
    float v = f32_from_bf16(x_ptr[c]);
    local_sq += v * v;
  }
  rms_shared[tid] = local_sq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) rms_shared[tid] += rms_shared[tid + stride];
    __syncthreads();
  }

  float inv;
  if (tid == 0) {
    float mean = rms_shared[0] / static_cast<float>(channels);
    rms_shared[0] = rsqrtf(mean + eps);
  }
  __syncthreads();
  inv = rms_shared[0];

  // Pass 2: normalize and (optionally) multiply by per-channel weight.
  for (int64_t c = tid; c < channels; c += blockDim.x) {
    float v = f32_from_bf16(x_ptr[c]) * inv;
    if (weight) v *= f32_from_bf16(weight[c]);
    y_ptr[c] = bf16_from_f32(v);
  }
}

// 2026-05-12 perf: vectorized LayerNorm forward (BF16) — 256 threads/row,
// vec=4 BF16 loads via __align__(8) bf16x4_ln struct, warp-shuffle reduction
// for both mean and variance, single shared-memory slot for inter-warp
// reduction. Same pattern as the rms_norm vec kernel landed in commit
// 2ebc2d1 and the layer_norm backward vec landed in commit 4d46832.
//
// The legacy `layer_norm_forward_bf16_kernel` below uses smem-tree reduction
// with scalar BF16 loads — bandwidth-limited at ~250 GB/s on Ampere. This
// vec version targets ~750-850 GB/s on the same shapes.
//
// Caller MUST verify norm_size % 4 == 0 (production diffusion shapes
// 1280/2560/3072/4096 all qualify) and dispatch to legacy otherwise.
struct __align__(8) bf16x4_ln {
  __nv_bfloat16 v[4];
};

__global__ void layer_norm_forward_bf16_vec_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ mean_out,
    float* __restrict__ rstd_out,
    int64_t norm_size,
    float eps)
{
  const int VEC = 4;
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int n_threads = blockDim.x;
  const int64_t n_vec = norm_size / VEC;

  const bf16x4_ln* X = reinterpret_cast<const bf16x4_ln*>(input  + row * norm_size);
  bf16x4_ln*       Y = reinterpret_cast<bf16x4_ln*>(output + row * norm_size);
  const bf16x4_ln* W = (weight != nullptr) ? reinterpret_cast<const bf16x4_ln*>(weight) : nullptr;
  const bf16x4_ln* B = (bias   != nullptr) ? reinterpret_cast<const bf16x4_ln*>(bias)   : nullptr;

  // Pass 1: parallel sum + sum_sq via vectorized loads.
  float local_sum = 0.f;
  float local_sq  = 0.f;
  for (int64_t i = tid; i < n_vec; i += n_threads) {
    bf16x4_ln d = X[i];
    #pragma unroll
    for (int k = 0; k < VEC; ++k) {
      float v = __bfloat162float(d.v[k]);
      local_sum += v;
      local_sq  += v * v;
    }
  }

  // Intra-warp reduction (warp shuffle for both partials).
  for (int off = 16; off > 0; off >>= 1) {
    local_sum += __shfl_xor_sync(0xffffffff, local_sum, off);
    local_sq  += __shfl_xor_sync(0xffffffff, local_sq,  off);
  }

  // Inter-warp reduction via shared memory.
  // Smem layout: [n_warps floats sum] [n_warps floats sq] (2 * n_warps total).
  extern __shared__ float ln_smem[];
  const int warp_id = tid >> 5;
  const int lane    = tid & 31;
  const int n_warps = (n_threads + 31) >> 5;

  if (lane == 0) {
    ln_smem[warp_id] = local_sum;
    ln_smem[n_warps + warp_id] = local_sq;
  }
  __syncthreads();

  if (warp_id == 0) {
    float s = (lane < n_warps) ? ln_smem[lane] : 0.f;
    float q = (lane < n_warps) ? ln_smem[n_warps + lane] : 0.f;
    for (int off = 16; off > 0; off >>= 1) {
      s += __shfl_xor_sync(0xffffffff, s, off);
      q += __shfl_xor_sync(0xffffffff, q, off);
    }
    if (lane == 0) {
      ln_smem[0] = s;
      ln_smem[n_warps] = q;
    }
  }
  __syncthreads();

  const float total_sum = ln_smem[0];
  const float total_sq  = ln_smem[n_warps];
  const float inv_norm = 1.f / static_cast<float>(norm_size);
  const float mean = total_sum * inv_norm;
  const float var  = total_sq * inv_norm - mean * mean;
  const float inv_std = rsqrtf(var + eps);

  if (tid == 0) {
    mean_out[row] = mean;
    rstd_out[row] = inv_std;
  }

  // Pass 2: vectorized normalize + affine.
  for (int64_t i = tid; i < n_vec; i += n_threads) {
    bf16x4_ln d = X[i];
    bf16x4_ln gv;
    bf16x4_ln bv;
    if (W != nullptr) gv = W[i];
    if (B != nullptr) bv = B[i];
    bf16x4_ln out;
    #pragma unroll
    for (int k = 0; k < VEC; ++k) {
      float v = (__bfloat162float(d.v[k]) - mean) * inv_std;
      if (W != nullptr) v *= __bfloat162float(gv.v[k]);
      if (B != nullptr) v += __bfloat162float(bv.v[k]);
      out.v[k] = __float2bfloat16_rn(v);
    }
    Y[i] = out;
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

// 2026-05-12 perf: vectorized GroupNorm stats kernel. Same warp-shuffle +
// smem inter-warp reduction pattern as rms_norm/layer_norm vec kernels.
// Uses vec=4 BF16 loads (bf16x4_gn) along the contiguous spatial axis.
// Caller MUST verify spatial_size % 4 == 0 before dispatching here.
//
// Index layout: each block handles one (n, g). Threads iterate over
// `elements_per_group = channels_per_group * spatial_size`. Since input is
// row-major NCHW with `c` varying slower than `hw`, consecutive thread
// indices in the *contiguous* axis (hw) land on adjacent input addresses
// → vec=4 loads coalesce.
struct __align__(8) bf16x4_gn {
  __nv_bfloat16 v[4];
};

__global__ void group_norm_compute_stats_bf16_vec_kernel(
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ mean_out,
    float* __restrict__ var_out,
    int batch_size,
    int channels,
    int groups,
    int channels_per_group,
    int spatial_size)
{
  const int VEC = 4;
  const int group_id = blockIdx.x;
  const int total_groups = batch_size * groups;
  if (group_id >= total_groups) return;
  const int n = group_id / groups;
  const int g = group_id % groups;
  const int tid = threadIdx.x;
  const int n_threads = blockDim.x;

  const int spatial_vec = spatial_size / VEC;          // # of bf16x4_gn per row
  const int elements_per_group = channels_per_group * spatial_size;
  const int vec_per_group = channels_per_group * spatial_vec;

  // Pass 1: collaborative sum + sum_sq using vec=4 BF16 loads.
  float local_sum = 0.f;
  float local_sq  = 0.f;
  for (int idx = tid; idx < vec_per_group; idx += n_threads) {
    const int c_offset = idx / spatial_vec;
    const int hw_vec   = idx % spatial_vec;
    const int c = g * channels_per_group + c_offset;
    // input offset in bf16x4_gn units; the underlying contiguous index
    // is ((n*channels + c)*spatial_size) + hw_vec*VEC, which is divisible
    // by VEC because hw_vec*VEC is.
    const int input_off = ((n * channels + c) * spatial_size) + hw_vec * VEC;
    const bf16x4_gn d = *reinterpret_cast<const bf16x4_gn*>(input + input_off);
    #pragma unroll
    for (int k = 0; k < VEC; ++k) {
      float v = __bfloat162float(d.v[k]);
      local_sum += v;
      local_sq  += v * v;
    }
  }

  // Intra-warp reduction.
  for (int off = 16; off > 0; off >>= 1) {
    local_sum += __shfl_xor_sync(0xffffffff, local_sum, off);
    local_sq  += __shfl_xor_sync(0xffffffff, local_sq,  off);
  }

  // Inter-warp reduction via shared memory.
  extern __shared__ float gn_smem[];  // size: 2 * n_warps
  const int warp_id = tid >> 5;
  const int lane    = tid & 31;
  const int n_warps = (n_threads + 31) >> 5;

  if (lane == 0) {
    gn_smem[warp_id] = local_sum;
    gn_smem[n_warps + warp_id] = local_sq;
  }
  __syncthreads();

  if (warp_id == 0) {
    float s = (lane < n_warps) ? gn_smem[lane] : 0.f;
    float q = (lane < n_warps) ? gn_smem[n_warps + lane] : 0.f;
    for (int off = 16; off > 0; off >>= 1) {
      s += __shfl_xor_sync(0xffffffff, s, off);
      q += __shfl_xor_sync(0xffffffff, q, off);
    }
    if (lane == 0) {
      const float count = static_cast<float>(elements_per_group);
      const float mean = s / count;
      const float var  = q / count - mean * mean;
      mean_out[group_id] = mean;
      var_out[group_id] = var;
    }
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

  // Truncate mean and rstd to BF16 precision before normalizing.
  // PyTorch's native_group_norm returns mean/rstd in the input dtype (BF16),
  // so its normalization step uses BF16-precision statistics. Without this
  // truncation, the full F32 statistics cause ~0.06 per-GroupNorm divergence
  // that compounds through deep networks (30+ layers → max diff ~5).
  float m = __bfloat162float(__float2bfloat16_rn(mean[mean_index]));
  float v = var[mean_index];
  float inv_std = __bfloat162float(__float2bfloat16_rn(rsqrtf(v + eps)));

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
  if (outer == 0) {
    return FC_OK;
  }
  // One block per row, threads cooperate via shared memory.
  // Pick the largest power-of-two ≤ min(channels, 1024) for block size.
  int block_size = 1;
  while (block_size * 2 <= channels && block_size * 2 <= 1024) block_size *= 2;
  if (block_size < 32) block_size = 32;
  dim3 grid(static_cast<unsigned int>(outer), 1, 1);
  dim3 block(static_cast<unsigned int>(block_size), 1, 1);
  size_t shmem = block_size * sizeof(float);
  rms_norm_kernel<<<grid, block, shmem, stream>>>(
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

  // 2026-05-12 perf: dispatch to the vec=4 kernel when norm_size is divisible
  // by 4 (production shapes 1280/2560/3072/4096 all qualify). Env override
  // FLAME_LAYER_NORM_FWD_LEGACY=1 forces the legacy smem-tree kernel.
  const char* fwd_legacy_env = getenv("FLAME_LAYER_NORM_FWD_LEGACY");
  const bool fwd_force_legacy = (fwd_legacy_env != nullptr && fwd_legacy_env[0] != 0 && fwd_legacy_env[0] != '0');
  if (!fwd_force_legacy && (norm_size % 4 == 0) && norm_size >= 4) {
    const int block_threads_vec = 256;
    const int n_warps_vec = (block_threads_vec + 31) >> 5;
    const size_t smem_vec = 2 * static_cast<size_t>(n_warps_vec) * sizeof(float);
    layer_norm_forward_bf16_vec_kernel<<<static_cast<int>(outer), block_threads_vec, smem_vec, stream>>>(
        input, weight, bias, output, mean_out, rstd_out, norm_size, eps);
    if (cudaGetLastError() != cudaSuccess) return FC_ERR_LAUNCH;
    return FC_OK;
  }

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

  // 2026-05-12 perf: dispatch to vec stats when spatial_size % 4 == 0.
  // VAE blocks typically have spatial = H*W = 64*64, 32*32, etc., always
  // a power-of-2 → divisible by 4. Env override
  // FLAME_GROUP_NORM_STATS_LEGACY=1 forces the legacy smem-tree path.
  const char* gn_legacy_env = getenv("FLAME_GROUP_NORM_STATS_LEGACY");
  const bool gn_force_legacy = (gn_legacy_env != nullptr && gn_legacy_env[0] != 0 && gn_legacy_env[0] != '0');
  const bool use_gn_vec = !gn_force_legacy && (spatial % 4 == 0) && spatial >= 4;

  int total_groups = static_cast<int>(stats_elems);
  dim3 stats_grid(total_groups, 1, 1);

  if (use_gn_vec) {
    const int block_threads_vec = 256;
    const int n_warps_vec = (block_threads_vec + 31) >> 5;
    const size_t smem_vec = 2 * static_cast<size_t>(n_warps_vec) * sizeof(float);
    dim3 stats_block_vec(block_threads_vec, 1, 1);
    group_norm_compute_stats_bf16_vec_kernel<<<stats_grid, stats_block_vec, smem_vec, stream>>>(
        input, mean_out, var_out,
        static_cast<int>(batch), static_cast<int>(channels), groups,
        static_cast<int>(channels_per_group), static_cast<int>(spatial));
  } else {
    int stats_threads = (spatial > 65536) ? 512 : 256;
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
  }

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
