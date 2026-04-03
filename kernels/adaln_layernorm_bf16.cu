// SPDX-License-Identifier: Apache-2.0
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ inline float bf16_to_f32(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ inline __nv_bfloat16 f32_to_bf16(const float v) { return __float2bfloat16_rn(v); }

extern "C" __global__ void layernorm_affine_bf16_nhwc_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
    const __nv_bfloat16* __restrict__ mod_s,
    const __nv_bfloat16* __restrict__ mod_b,
    int B, int H, int W, int C,
    float eps)
{
    int row = blockIdx.x;
    if (row >= B * H * W) {
        return;
    }

    int b = row / (H * W);
    size_t row_base = static_cast<size_t>(row) * C;

    float mean = 0.f;
    float m2   = 0.f;
    int   n    = 0;

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float v = bf16_to_f32(x[row_base + c]);
        n += 1;
        float delta = v - mean;
        mean += delta / n;
        float delta2 = v - mean;
        m2 += delta * delta2;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        mean += __shfl_down_sync(0xffffffff, mean, offset);
        m2   += __shfl_down_sync(0xffffffff, m2,   offset);
        n    += __shfl_down_sync(0xffffffff, n,    offset);
    }

    __shared__ float s_mean[32];
    __shared__ float s_m2[32];
    __shared__ int   s_n[32];

    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;

    if (lane == 0) {
        s_mean[warp_id] = mean;
        s_m2[warp_id]   = m2;
        s_n[warp_id]    = n;
    }
    __syncthreads();

    if (warp_id == 0) {
        float agg_mean = (lane < (blockDim.x >> 5)) ? s_mean[lane] : 0.f;
        float agg_m2   = (lane < (blockDim.x >> 5)) ? s_m2[lane]   : 0.f;
        int   agg_n    = (lane < (blockDim.x >> 5)) ? s_n[lane]    : 0;

        for (int offset = 16; offset > 0; offset >>= 1) {
            agg_mean += __shfl_down_sync(0xffffffff, agg_mean, offset);
            agg_m2   += __shfl_down_sync(0xffffffff, agg_m2,   offset);
            agg_n    += __shfl_down_sync(0xffffffff, agg_n,    offset);
        }

        if (lane == 0) {
            s_mean[0] = (agg_n > 0) ? agg_mean / agg_n : 0.f;
            s_m2[0]   = (agg_n > 0) ? agg_m2 / agg_n : 0.f;
        }
    }
    __syncthreads();

    float mu = s_mean[0];
    float var = s_m2[0];
    float inv_std = rsqrtf(var + eps);

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float v = bf16_to_f32(x[row_base + c]);
        float nrm = (v - mu) * inv_std;

        float g = gamma ? bf16_to_f32(gamma[c]) : 1.f;
        float be = beta ? bf16_to_f32(beta[c]) : 0.f;

        if (mod_s) {
            g *= (1.f + bf16_to_f32(mod_s[b * C + c]));
        }
        if (mod_b) {
            be += bf16_to_f32(mod_b[b * C + c]);
        }

        float out = nrm * g + be;
        y[row_base + c] = f32_to_bf16(out);
    }
}

extern "C" void layernorm_affine_bf16_nhwc_forward(
    const void* x,
    void* y,
    const void* gamma,
    const void* beta,
    int B,
    int H,
    int W,
    int C,
    float eps,
    cudaStream_t stream)
{
    const __nv_bfloat16* X = reinterpret_cast<const __nv_bfloat16*>(x);
    __nv_bfloat16* Y = reinterpret_cast<__nv_bfloat16*>(y);
    const __nv_bfloat16* G = reinterpret_cast<const __nv_bfloat16*>(gamma);
    const __nv_bfloat16* Bt = reinterpret_cast<const __nv_bfloat16*>(beta);

    int rows = B * H * W;
    dim3 grid(rows);
    dim3 block(256);
    layernorm_affine_bf16_nhwc_kernel<<<grid, block, 0, stream>>>(
        X, Y, G, Bt, nullptr, nullptr, B, H, W, C, eps);
}

extern "C" void adaln_modulate_bf16_nhwc_forward(
    const void* x,
    void* y,
    const void* gamma,
    const void* beta,
    const void* mod_s,
    const void* mod_b,
    int B,
    int H,
    int W,
    int C,
    float eps,
    cudaStream_t stream)
{
    const __nv_bfloat16* X = reinterpret_cast<const __nv_bfloat16*>(x);
    __nv_bfloat16* Y = reinterpret_cast<__nv_bfloat16*>(y);
    const __nv_bfloat16* G = reinterpret_cast<const __nv_bfloat16*>(gamma);
    const __nv_bfloat16* Bt = reinterpret_cast<const __nv_bfloat16*>(beta);
    const __nv_bfloat16* Ms = reinterpret_cast<const __nv_bfloat16*>(mod_s);
    const __nv_bfloat16* Mb = reinterpret_cast<const __nv_bfloat16*>(mod_b);

    int rows = B * H * W;
    dim3 grid(rows);
    dim3 block(256);
    layernorm_affine_bf16_nhwc_kernel<<<grid, block, 0, stream>>>(
        X, Y, G, Bt, Ms, Mb, B, H, W, C, eps);
}
