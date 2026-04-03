#include "cuda_ops.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {

__device__ inline float f32_from_bf16(__nv_bfloat16 h) {
  return __bfloat162float(h);
}

__device__ inline __nv_bfloat16 bf16_from_f32(float x) {
  return __float2bfloat16(x);
}

template <typename T>
__global__ void upsample2d_nearest_nchw_kernel(
    const T* input,
    T* output,
    int batch,
    int channels,
    int h_in,
    int w_in,
    int h_out,
    int w_out,
    float scale_h,
    float scale_w) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * channels * h_out * w_out;
    
    if (idx >= total_elements) return;

    int w_out_idx = idx % w_out;
    int h_out_idx = (idx / w_out) % h_out;
    int c_idx = (idx / (w_out * h_out)) % channels;
    int b_idx = idx / (w_out * h_out * channels);

    // Nearest neighbor interpolation
    int h_in_idx = (int)(h_out_idx * scale_h);
    int w_in_idx = (int)(w_out_idx * scale_w);
    
    if (h_in_idx >= h_in) h_in_idx = h_in - 1;
    if (w_in_idx >= w_in) w_in_idx = w_in - 1;

    int input_idx = ((b_idx * channels + c_idx) * h_in + h_in_idx) * w_in + w_in_idx;
    output[idx] = input[input_idx];
}

} // namespace

extern "C" fc_status_t fc_upsample2d_nearest_bf16(
    const void* input,
    void* output,
    int batch,
    int channels,
    int h_in,
    int w_in,
    int h_out,
    int w_out,
    cudaStream_t stream) {
    
    if (!input || !output) {
        return FC_ERR_INVALID_ARGUMENT;
    }

    float scale_h = (float)h_in / (float)h_out;
    float scale_w = (float)w_in / (float)w_out;

    int total_elements = batch * channels * h_out * w_out;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    upsample2d_nearest_nchw_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<__nv_bfloat16*>(output),
        batch, channels, h_in, w_in, h_out, w_out, scale_h, scale_w
    );

    if (cudaGetLastError() != cudaSuccess) {
        return FC_ERR_LAUNCH;
    }

    return FC_OK;
}

extern "C" fc_status_t fc_upsample2d_nearest_f32(
    const void* input,
    void* output,
    int batch,
    int channels,
    int h_in,
    int w_in,
    int h_out,
    int w_out,
    cudaStream_t stream) {
    
    if (!input || !output) {
        return FC_ERR_INVALID_ARGUMENT;
    }

    float scale_h = (float)h_in / (float)h_out;
    float scale_w = (float)w_in / (float)w_out;

    int total_elements = batch * channels * h_out * w_out;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    upsample2d_nearest_nchw_kernel<float><<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(input),
        static_cast<float*>(output),
        batch, channels, h_in, w_in, h_out, w_out, scale_h, scale_w
    );

    if (cudaGetLastError() != cudaSuccess) {
        return FC_ERR_LAUNCH;
    }

    return FC_OK;
}
