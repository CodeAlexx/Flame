// SPDX-License-Identifier: Apache-2.0
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ inline float bf16_to_f32(const __nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ inline __nv_bfloat16 f32_to_bf16(const float v) {
    return __float2bfloat16_rn(v);
}

__global__ void rope_apply_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B,
    int H,
    int S,
    int Dh,
    int rope_dim,
    float base_theta,
    int pos_offset)
{
    int token = blockIdx.x;
    int total_tokens = B * H * S;
    if (token >= total_tokens) {
        return;
    }

    int s_idx = token % S;
    int tmp = token / S;
    int h_idx = tmp % H;
    int b_idx = tmp / H;

    size_t base = (((size_t)b_idx * H + h_idx) * S + s_idx) * Dh;

    int rotary_pairs = rope_dim / 2;
    for (int pair = threadIdx.x; pair < rotary_pairs; pair += blockDim.x) {
        float freq = powf(base_theta, -2.0f * pair / static_cast<float>(rope_dim));
        int pos = pos_offset + s_idx;
        float angle = freq * static_cast<float>(pos);
        float sin_val;
        float cos_val;
        sincosf(angle, &sin_val, &cos_val);

        size_t idx0 = base + static_cast<size_t>(2 * pair);
        size_t idx1 = idx0 + 1;
        float x0 = bf16_to_f32(input[idx0]);
        float x1 = bf16_to_f32(input[idx1]);
        float out0 = x0 * cos_val - x1 * sin_val;
        float out1 = x0 * sin_val + x1 * cos_val;
        output[idx0] = f32_to_bf16(out0);
        output[idx1] = f32_to_bf16(out1);
    }

    for (int tail = threadIdx.x + rope_dim; tail < Dh; tail += blockDim.x) {
        output[base + tail] = input[base + tail];
    }
}

} // namespace

extern "C" int flame_rope_apply_bf16_fp32(
    const void* input_ptr,
    void* output_ptr,
    int B,
    int H,
    int S,
    int Dh,
    int rope_dim,
    float base_theta,
    int pos_offset,
    void* stream_ptr)
{
    if (rope_dim <= 0 || (rope_dim & 1) != 0 || rope_dim > Dh) {
        return 1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    dim3 grid(B * H * S);
    int rotary_pairs = rope_dim / 2;
    int threads = rotary_pairs;
    if (threads < 32) {
        threads = 32;
    }
    if (threads > 256) {
        threads = 256;
    }

    rope_apply_kernel<<<grid, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input_ptr),
        static_cast<__nv_bfloat16*>(output_ptr),
        B,
        H,
        S,
        Dh,
        rope_dim,
        base_theta,
        pos_offset);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}
