#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

#ifndef FLAME_SDPA_NEG_INF
#define FLAME_SDPA_NEG_INF -1.0e30f
#endif

namespace {

__device__ __forceinline__ int idx3(int b, int i, int j, int I, int J) {
    return (b * I + i) * J + j;
}

__device__ __forceinline__ uint32_t xorshift32(uint32_t x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

__device__ __forceinline__ float rng_uniform01(uint64_t seed, int b, int qi, int kj) {
    uint32_t s = static_cast<uint32_t>(seed) ^
        (1103515245u *
         (static_cast<uint32_t>(b) * 73856093u ^
          static_cast<uint32_t>(qi) * 19349663u ^
          static_cast<uint32_t>(kj) * 83492791u));
    return (xorshift32(s) & 0x00FFFFFF) / 16777216.0f;
}

__global__ void causal_mask_kernel(
    float* scores,
    int B,
    int H,
    int Q,
    int K,
    int q_offset,
    int k_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * Q * K;
    if (idx >= total) {
        return;
    }

    int k = idx % K;
    int q = (idx / K) % Q;
    int h = (idx / (K * Q)) % H;
    int b = idx / (K * Q * H);
    (void)h;
    (void)b;

    int global_q = q_offset + q;
    int global_k = k_offset + k;
    if (global_k > global_q) {
        scores[idx] = -1.0e9f;
    }
}

__global__ void attn_mask_kernel(
    float* scores,
    const uint8_t* mask,
    int B,
    int H,
    int Q,
    int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * Q * K;
    if (idx >= total) {
        return;
    }
    if (mask[idx] == 0) {
        scores[idx] = -1.0e9f;
    }
}

__global__ void add_mask_tile_fp32_kernel(
    float* logits,
    const uint8_t* user_bool,
    const float* user_add,
    int BH,
    int q_t,
    int k_t,
    int q_abs_start,
    int k_abs_start,
    int user_bool_rank,
    int user_add_rank,
    int k_total,
    bool bool_zero_is_mask,
    bool causal)
{
    int bh = blockIdx.z;
    int qi = blockIdx.y * blockDim.y + threadIdx.y;
    int kj = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= BH || qi >= q_t || kj >= k_t) {
        return;
    }

    int index = idx3(bh, qi, kj, q_t, k_t);
    float delta = 0.0f;

    if (user_add != nullptr) {
        if (user_add_rank == 3) {
            delta += user_add[index];
        } else if (user_add_rank == 2) {
            int k_abs = k_abs_start + kj;
            delta += user_add[bh * k_total + k_abs];
        }
    }

    if (user_bool != nullptr) {
        uint8_t mask_val = 1;
        if (user_bool_rank == 3) {
            mask_val = user_bool[index];
        } else if (user_bool_rank == 2) {
            int k_abs = k_abs_start + kj;
            mask_val = user_bool[bh * k_total + k_abs];
        }
        bool is_mask = bool_zero_is_mask ? (mask_val == 0) : (mask_val != 0);
        if (is_mask) {
            delta += FLAME_SDPA_NEG_INF;
        }
    }

    if (causal) {
        int q_abs = q_abs_start + qi;
        int k_abs = k_abs_start + kj;
        if (k_abs > q_abs) {
            delta += FLAME_SDPA_NEG_INF;
        }
    }
    logits[index] += delta;
}

__global__ void softmax_from_lse_tile_kernel(
    const float* logits,
    const float* lse_row,
    __nv_bfloat16* probs,
    int BH,
    int q_t,
    int k_t)
{
    int bh = blockIdx.z;
    int qi = blockIdx.y * blockDim.y + threadIdx.y;
    int kj = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= BH || qi >= q_t || kj >= k_t) {
        return;
    }
    int index = idx3(bh, qi, kj, q_t, k_t);
    float lse = lse_row[bh * q_t + qi];
    float p = __expf(logits[index] - lse);
    probs[index] = __float2bfloat16(p);
}

__global__ void lse_from_logits_tile_kernel(
    const float* logits,
    float* out_lse_row,
    int BH,
    int q_t,
    int k_t)
{
    int bh = blockIdx.z;
    int qi = blockIdx.y * blockDim.y + threadIdx.y;
    if (bh >= BH || qi >= q_t) {
        return;
    }

    float max_val = -FLT_MAX;
    for (int kj = threadIdx.x; kj < k_t; kj += blockDim.x) {
        float val = logits[idx3(bh, qi, kj, q_t, k_t)];
        max_val = fmaxf(max_val, val);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));
    }

    __shared__ float shared_max[32 * 8];
    if (threadIdx.x == 0) {
        shared_max[threadIdx.y] = max_val;
    }
    __syncthreads();
    max_val = shared_max[threadIdx.y];

    float sum = 0.0f;
    for (int kj = threadIdx.x; kj < k_t; kj += blockDim.x) {
        float val = logits[idx3(bh, qi, kj, q_t, k_t)];
        sum += __expf(val - max_val);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }
    if (threadIdx.x == 0) {
        out_lse_row[bh * q_t + qi] = logf(sum) + max_val;
    }
}

__global__ void lse_merge_rows_kernel(
    float* lse_row,
    const float* tile_lse_row,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    float a = lse_row[idx];
    float b = tile_lse_row[idx];
    float m = fmaxf(a, b);
    float out = m + logf(__expf(a - m) + __expf(b - m));
    lse_row[idx] = out;
}

__global__ void dropout_bf16_inplace_kernel(
    __nv_bfloat16* probs,
    float p,
    uint64_t seed,
    int BH,
    int q_t,
    int k_t)
{
    float keep = 1.0f - p;
    float scale = keep > 0.0f ? 1.0f / keep : 0.0f;
    int bh = blockIdx.z;
    int qi = blockIdx.y * blockDim.y + threadIdx.y;
    int kj = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= BH || qi >= q_t || kj >= k_t) {
        return;
    }
    int index = idx3(bh, qi, kj, q_t, k_t);
    float val = __bfloat162float(probs[index]);
    float u = rng_uniform01(seed, bh, qi, kj);
    if (u < p) {
        val = 0.0f;
    } else {
        val *= scale;
    }
    probs[index] = __float2bfloat16(val);
}

} // namespace

extern "C" int flame_apply_causal_mask_fp32(
    float* scores,
    int B,
    int H,
    int Q,
    int K,
    int q_offset,
    int k_offset,
    cudaStream_t stream)
{
    int total = B * H * Q * K;
    int block = 256;
    int grid = (total + block - 1) / block;
    if (grid > 0) {
        causal_mask_kernel<<<grid, block, 0, stream>>>(
            scores,
            B,
            H,
            Q,
            K,
            q_offset,
            k_offset);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int flame_apply_attn_mask_fp32(
    float* scores,
    const uint8_t* mask,
    int B,
    int H,
    int Q,
    int K,
    cudaStream_t stream)
{
    int total = B * H * Q * K;
    int block = 256;
    int grid = (total + block - 1) / block;
    if (grid > 0) {
        attn_mask_kernel<<<grid, block, 0, stream>>>(scores, mask, B, H, Q, K);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int flame_sdpa_add_mask_tile_fp32(
    float* logits,
    const uint8_t* user_bool,
    const float* user_add,
    int BH,
    int q_t,
    int k_t,
    int q_abs_start,
    int k_abs_start,
    int user_bool_rank,
    int user_add_rank,
    int k_total,
    int bool_zero_is_mask,
    int causal_flag,
    cudaStream_t stream)
{
    dim3 block(32, 8, 1);
    dim3 grid((k_t + block.x - 1) / block.x, (q_t + block.y - 1) / block.y, BH);
    add_mask_tile_fp32_kernel<<<grid, block, 0, stream>>>(
        logits,
        user_bool,
        user_add,
        BH,
        q_t,
        k_t,
        q_abs_start,
        k_abs_start,
        user_bool_rank,
        user_add_rank,
        k_total,
        bool_zero_is_mask != 0,
        causal_flag != 0);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int flame_sdpa_softmax_from_lse_tile(
    const float* logits,
    const float* lse_row,
    void* probs_bf16,
    int BH,
    int q_t,
    int k_t,
    cudaStream_t stream)
{
    dim3 block(32, 8, 1);
    dim3 grid((k_t + block.x - 1) / block.x, (q_t + block.y - 1) / block.y, BH);
    softmax_from_lse_tile_kernel<<<grid, block, 0, stream>>>(
        logits,
        lse_row,
        reinterpret_cast<__nv_bfloat16*>(probs_bf16),
        BH,
        q_t,
        k_t);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int flame_sdpa_lse_from_logits_tile(
    const float* logits,
    float* out_lse_row,
    int BH,
    int q_t,
    int k_t,
    cudaStream_t stream)
{
    dim3 block(32, 8, 1);
    dim3 grid(1, (q_t + block.y - 1) / block.y, BH);
    lse_from_logits_tile_kernel<<<grid, block, 0, stream>>>(
        logits,
        out_lse_row,
        BH,
        q_t,
        k_t);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int flame_sdpa_lse_merge_rows(
    float* lse_row,
    const float* tile_lse_row,
    int BH,
    int q_t,
    cudaStream_t stream)
{
    int N = BH * q_t;
    int block = 256;
    int grid = (N + block - 1) / block;
    if (grid > 0) {
        lse_merge_rows_kernel<<<grid, block, 0, stream>>>(
            lse_row,
            tile_lse_row,
            N);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int flame_sdpa_dropout_bf16_inplace(
    void* probs_bf16,
    float p,
    uint64_t seed,
    int BH,
    int q_t,
    int k_t,
    cudaStream_t stream)
{
    dim3 block(32, 8, 1);
    dim3 grid((k_t + block.x - 1) / block.x, (q_t + block.y - 1) / block.y, BH);
    dropout_bf16_inplace_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(probs_bf16),
        p,
        seed,
        BH,
        q_t,
        k_t);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
