// Bilinear 2D upsampling for NCHW tensors, BF16 and F32.
//
// Semantics match PyTorch's `F.interpolate(..., mode='bilinear')`:
//   align_corners=False → h_scale = h_in / h_out,
//                         h_idx   = (h_out_idx + 0.5) * h_scale - 0.5
//   align_corners=True  → h_scale = (h_in - 1) / (h_out - 1),
//                         h_idx   = h_out_idx * h_scale
//
// Template over element type; BF16 is upcast to F32 for the bilinear
// weighted sum (four taps × two frac multiplies) and rounded back to
// BF16 on store so we don't accumulate in a 7-bit mantissa.

#include "cuda_ops.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace {

__device__ inline float load_as_f32(const __nv_bfloat16& x) {
    return __bfloat162float(x);
}
__device__ inline float load_as_f32(const float& x) { return x; }

__device__ inline __nv_bfloat16 store_from_f32_bf16(float x) {
    return __float2bfloat16(x);
}
__device__ inline float store_from_f32_f32(float x) { return x; }

template <typename T>
__global__ void upsample2d_bilinear_nchw_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int batch,
    int channels,
    int h_in,
    int w_in,
    int h_out,
    int w_out,
    float h_scale,
    float w_scale,
    int align_corners
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * h_out * w_out;
    if (idx >= total) return;

    int w_o = idx % w_out;
    int h_o = (idx / w_out) % h_out;
    int c   = (idx / ((long long)w_out * h_out)) % channels;
    int b   = idx / ((long long)w_out * h_out * channels);

    float h_idx_f = align_corners
        ? (float)h_o * h_scale
        : ((float)h_o + 0.5f) * h_scale - 0.5f;
    float w_idx_f = align_corners
        ? (float)w_o * w_scale
        : ((float)w_o + 0.5f) * w_scale - 0.5f;

    int h0 = (int)floorf(h_idx_f);
    int w0 = (int)floorf(w_idx_f);
    int h1 = h0 + 1;
    int w1 = w0 + 1;

    float h_frac = h_idx_f - floorf(h_idx_f);
    float w_frac = w_idx_f - floorf(w_idx_f);

    // Edge clamp — matches PyTorch's edge-replication semantics.
    if (h0 < 0) h0 = 0;
    if (w0 < 0) w0 = 0;
    if (h1 >= h_in) h1 = h_in - 1;
    if (w1 >= w_in) w1 = w_in - 1;
    if (h0 >= h_in) h0 = h_in - 1;
    if (w0 >= w_in) w0 = w_in - 1;

    long long plane_stride = (long long)h_in * w_in;
    long long nc_offset = ((long long)b * channels + c) * plane_stride;

    float v00 = load_as_f32(input[nc_offset + (long long)h0 * w_in + w0]);
    float v01 = load_as_f32(input[nc_offset + (long long)h0 * w_in + w1]);
    float v10 = load_as_f32(input[nc_offset + (long long)h1 * w_in + w0]);
    float v11 = load_as_f32(input[nc_offset + (long long)h1 * w_in + w1]);

    float v0 = v00 * (1.0f - w_frac) + v01 * w_frac;
    float v1 = v10 * (1.0f - w_frac) + v11 * w_frac;
    float v  = v0  * (1.0f - h_frac) + v1  * h_frac;

    if constexpr (sizeof(T) == sizeof(__nv_bfloat16)) {
        output[idx] = store_from_f32_bf16(v);
    } else {
        output[idx] = store_from_f32_f32(v);
    }
}

// PyTorch's `(h_in) / h_out` vs `(h_in - 1) / (h_out - 1)` split.
__device__ __host__ inline float compute_scale(int in_size, int out_size, int align_corners) {
    if (align_corners && out_size > 1) {
        return (float)(in_size - 1) / (float)(out_size - 1);
    }
    return (float)in_size / (float)out_size;
}

} // namespace

extern "C" fc_status_t fc_upsample2d_bilinear_bf16(
    const void* input,
    void* output,
    int batch,
    int channels,
    int h_in,
    int w_in,
    int h_out,
    int w_out,
    int align_corners,
    cudaStream_t stream
) {
    if (!input || !output) return FC_ERR_INVALID_ARGUMENT;

    float h_scale = compute_scale(h_in, h_out, align_corners);
    float w_scale = compute_scale(w_in, w_out, align_corners);

    int total = batch * channels * h_out * w_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    upsample2d_bilinear_nchw_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<__nv_bfloat16*>(output),
        batch, channels, h_in, w_in, h_out, w_out,
        h_scale, w_scale, align_corners
    );

    if (cudaGetLastError() != cudaSuccess) return FC_ERR_LAUNCH;
    return FC_OK;
}

extern "C" fc_status_t fc_upsample2d_bilinear_f32(
    const void* input,
    void* output,
    int batch,
    int channels,
    int h_in,
    int w_in,
    int h_out,
    int w_out,
    int align_corners,
    cudaStream_t stream
) {
    if (!input || !output) return FC_ERR_INVALID_ARGUMENT;

    float h_scale = compute_scale(h_in, h_out, align_corners);
    float w_scale = compute_scale(w_in, w_out, align_corners);

    int total = batch * channels * h_out * w_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    upsample2d_bilinear_nchw_kernel<float><<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(input),
        static_cast<float*>(output),
        batch, channels, h_in, w_in, h_out, w_out,
        h_scale, w_scale, align_corners
    );

    if (cudaGetLastError() != cudaSuccess) return FC_ERR_LAUNCH;
    return FC_OK;
}
