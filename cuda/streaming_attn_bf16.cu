// streaming_attn_bf16_fp32.cu
// Correctness-first streaming attention kernel:
// - BF16 inputs/outputs
// - FP32 math (log-sum-exp streaming softmax, FP32 value accumulators)
// - No [S,S] allocations or intermediate tensors

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>
#include <stdio.h>

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 128
#endif

#ifndef ROWS_PER_BLOCK
#define ROWS_PER_BLOCK 8
#endif

#ifndef MAX_CHUNK_SIZE
#define MAX_CHUNK_SIZE 2048
#endif

#ifndef TILE_DV
#define TILE_DV 32
#endif

#if THREADS_PER_BLOCK & (THREADS_PER_BLOCK - 1)
#error "THREADS_PER_BLOCK must be a power of two"
#endif

__device__ __forceinline__ float bf16_to_f32(const __nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(const float x) {
    return __float2bfloat16(x);
}

__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    shared[threadIdx.x] = val;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] += shared[threadIdx.x + offset];
        }
        __syncthreads();
    }
    float result = shared[0];
    __syncthreads();
    return result;
}

__device__ __forceinline__ float block_reduce_max(float val, float* shared) {
    shared[threadIdx.x] = val;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            float rhs = shared[threadIdx.x + offset];
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], rhs);
        }
        __syncthreads();
    }
    float result = shared[0];
    __syncthreads();
    return result;
}

constexpr int MAX_KEYS_PER_THREAD =
    (MAX_CHUNK_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

extern "C" __global__
void streaming_attn_bf16_fp32_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    int B,
    int H,
    int S,
    int Dh,
    int Dv,
    long qB,
    long qH,
    long qS,
    long qD,
    long kB,
    long kH,
    long kS,
    long kD,
    long vB,
    long vH,
    long vS,
    long vD,
    __nv_bfloat16* __restrict__ O,
    long oB,
    long oH,
    long oS,
    long oD,
    float scale,
    int chunk_size,
    int causal,
    const uint8_t* __restrict__ k_pad_mask)
{
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int row_block = blockIdx.z;
    if (b >= B || h >= H) {
        return;
    }

    const bool has_mask = k_pad_mask != nullptr;
    const int tid = threadIdx.x;
    extern __shared__ float shmem[];
    float* q_cache = shmem; // Dh floats per row

    __shared__ float shared_reduce_a[THREADS_PER_BLOCK];
    __shared__ float shared_reduce_b[THREADS_PER_BLOCK];

    const int row_start = row_block * ROWS_PER_BLOCK;

    for (int i_rel = 0; i_rel < ROWS_PER_BLOCK; ++i_rel) {
        const int i = row_start + i_rel;
        if (i >= S) {
            break;
        }

        const long q_row_offset =
            (long)b * qB + (long)h * qH + (long)i * qS;
        const __nv_bfloat16* q_row_ptr = Q + q_row_offset;

        for (int d = tid; d < Dh; d += blockDim.x) {
            q_cache[d] = bf16_to_f32(q_row_ptr[(long)d * qD]);
        }
        __syncthreads();

        __shared__ float shared_m;
        __shared__ float shared_l;
        if (tid == 0) {
            shared_m = -CUDART_INF_F;
            shared_l = 0.f;
        }
        __syncthreads();

        for (int c0 = 0; c0 < S; c0 += chunk_size) {
            int C = chunk_size;
            if (c0 + C > S) {
                C = S - c0;
            }

            float scores[MAX_KEYS_PER_THREAD];
            int local_count = 0;
            float local_max = -CUDART_INF_F;

            for (int key = tid; key < C; key += blockDim.x) {
                const int seq_idx = c0 + key;
                if (seq_idx >= S) {
                    break;
                }

                const __nv_bfloat16* k_row_ptr =
                    K + (long)b * kB + (long)h * kH + (long)seq_idx * kS;

                bool masked = false;
                if (causal && seq_idx > i) masked = true;
                if (has_mask && !masked) {
                    size_t mask_index =
                        ((size_t)b * (size_t)H + (size_t)h) * (size_t)S + (size_t)seq_idx;
                    masked = (k_pad_mask[mask_index] == 0);
                }

                float score = -CUDART_INF_F;
                if (!masked) {
                    float dot = 0.f;
                    for (int d = 0; d < Dh; ++d) {
                        const float qv = q_cache[d];
                        const float kv = bf16_to_f32(k_row_ptr[(long)d * kD]);
                        dot += qv * kv;
                    }
                    score = dot * scale;
                }

                if (local_count < MAX_KEYS_PER_THREAD) {
                    scores[local_count] = score;
                    ++local_count;
                }
                local_max = fmaxf(local_max, score);
            }

            const float chunk_max = block_reduce_max(local_max, shared_reduce_a);
            if (chunk_max > -CUDART_INF_F) {
                float local_sum = 0.f;
                for (int t = 0; t < local_count; ++t) {
                    const float sc = scores[t];
                    if (sc > -CUDART_INF_F) {
                        local_sum += __expf(sc - chunk_max);
                    }
                }
                const float chunk_sum = block_reduce_sum(local_sum, shared_reduce_b);

                if (tid == 0) {
                    const float m_prev = shared_m;
                    const float l_prev = shared_l;
                    const float m_new = fmaxf(m_prev, chunk_max);
                    const float exp_prev =
                        (m_prev == -CUDART_INF_F) ? 0.f : __expf(m_prev - m_new);
                    const float l_new = l_prev * exp_prev + chunk_sum;
                    shared_m = m_new;
                    shared_l = l_new;
                }
            }
            __syncthreads();
        }

        float m_final = shared_m;
        float l_final = shared_l;
        if (tid == 0) {
            if (!isfinite(l_final) || l_final < 1e-20f) {
                shared_l = 1e-20f;
                l_final = 1e-20f;
            }
        }
        __syncthreads();
        m_final = shared_m;
        l_final = shared_l;
        const float inv_l = 1.f / l_final;

        const long out_row_offset =
            (long)b * oB + (long)h * oH + (long)i * oS;
        __nv_bfloat16* out_row_ptr = O + out_row_offset;

        for (int dv0 = 0; dv0 < Dv; dv0 += TILE_DV) {
            int dv_tile = TILE_DV;
            if (dv0 + dv_tile > Dv) {
                dv_tile = Dv - dv0;
            }

            float z_local[TILE_DV];
#pragma unroll
            for (int t = 0; t < TILE_DV; ++t) {
                z_local[t] = 0.f;
            }

            for (int c0 = 0; c0 < S; c0 += chunk_size) {
                int C = chunk_size;
                if (c0 + C > S) {
                    C = S - c0;
                }

                for (int key = tid; key < C; key += blockDim.x) {
                    const int seq_idx = c0 + key;
                    if (seq_idx >= S) {
                        break;
                    }

                    const __nv_bfloat16* k_row_ptr =
                        K + (long)b * kB + (long)h * kH + (long)seq_idx * kS;
                    const __nv_bfloat16* v_row_ptr =
                        V + (long)b * vB + (long)h * vH + (long)seq_idx * vS;

                    bool masked = false;
                    if (causal && seq_idx > i) masked = true;
                    if (has_mask && !masked) {
                        size_t mask_index =
                            ((size_t)b * (size_t)H + (size_t)h) * (size_t)S + (size_t)seq_idx;
                        masked = (k_pad_mask[mask_index] == 0);
                    }
                    if (masked) {
                        continue;
                    }

                    float dot = 0.f;
                    for (int d = 0; d < Dh; ++d) {
                        const float qv = q_cache[d];
                        const float kv = bf16_to_f32(k_row_ptr[(long)d * kD]);
                        dot += qv * kv;
                    }
                    const float score = dot * scale;
                    const float weight = __expf(score - m_final);

#pragma unroll
                    for (int t = 0; t < TILE_DV; ++t) {
                        if (t < dv_tile) {
                            const float vv = bf16_to_f32(v_row_ptr[((long)dv0 + t) * vD]);
                            z_local[t] += weight * vv;
                        }
                    }
                }
                __syncthreads();
            }

            for (int t = 0; t < dv_tile; ++t) {
                const float acc = block_reduce_sum(z_local[t], shared_reduce_a);
                if (tid == 0) {
                    out_row_ptr[((long)dv0 + t) * oD] = f32_to_bf16(acc * inv_l);
                }
            }
            __syncthreads();
        }

        __syncthreads();
    }
}

extern "C" cudaError_t streaming_attn_bf16_fp32_launch(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    int B,
    int H,
    int S,
    int Dh,
    int Dv,
    long qB,
    long qH,
    long qS,
    long qD,
    long kB,
    long kH,
    long kS,
    long kD,
    long vB,
    long vH,
    long vS,
    long vD,
    __nv_bfloat16* O,
    long oB,
    long oH,
    long oS,
    long oD,
    float scale,
    int chunk_size,
    int causal,
    const uint8_t* k_pad_mask,
    cudaStream_t stream)
{
    if (chunk_size <= 0) {
        return cudaErrorInvalidValue;
    }

    int chunk = chunk_size;
    const int seq_cap = (S > 0) ? S : 1;
    const int max_chunk = (seq_cap < MAX_CHUNK_SIZE) ? seq_cap : MAX_CHUNK_SIZE;
    const int min_chunk = (max_chunk < 32) ? max_chunk : 32;
    if (chunk < min_chunk) {
        chunk = min_chunk;
    }
    if (chunk > max_chunk) {
        chunk = max_chunk;
    }

    cudaError_t pre_err = cudaGetLastError();
    if (pre_err != cudaSuccess) {
        printf(
            "[streaming_attn_launch_debug] clearing prior CUDA error %d (%s)\n",
            static_cast<int>(pre_err),
            cudaGetErrorString(pre_err));
    }

    dim3 grid(
        static_cast<unsigned int>(B),
        static_cast<unsigned int>(H),
        static_cast<unsigned int>((S + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK));
    dim3 block(THREADS_PER_BLOCK, 1, 1);
    if (grid.z == 0) {
        grid.z = 1;
    }

    const size_t shmem_bytes = static_cast<size_t>(Dh) * sizeof(float);

    unsigned int threads_per_block = block.x * block.y * block.z;
    int device_id = 0;
    int sm = -1;
    if (cudaGetDevice(&device_id) == cudaSuccess) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
            sm = prop.major * 10 + prop.minor;
        }
    }

    printf(
        "[launch_dump] block=(%u,%u,%u) grid=(%u,%u,%u) threads=%u shmem=%zu coop=N sm=%d\n",
        block.x,
        block.y,
        block.z,
        grid.x,
        grid.y,
        grid.z,
        threads_per_block,
        shmem_bytes,
        sm);

    streaming_attn_bf16_fp32_kernel<<<grid, block, shmem_bytes, stream>>>(
        Q,
        K,
        V,
        B,
        H,
        S,
        Dh,
        Dv,
        qB,
        qH,
        qS,
        qD,
        kB,
        kH,
        kS,
        kD,
        vB,
        vH,
        vS,
        vD,
        O,
        oB,
        oH,
        oS,
        oD,
        scale,
        chunk,
        causal,
        k_pad_mask);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        printf(
            "[streaming_attn_launch_debug] launch failed with %d (%s)\n",
            static_cast<int>(launch_err),
            cudaGetErrorString(launch_err));
    }
    return launch_err;
}

extern "C" cudaError_t streaming_attn_bf16_fp32_attrs(
    int* max_threads_per_block,
    int* static_shared_bytes,
    int* binary_version)
{
    cudaFuncAttributes attr{};
    cudaError_t err = cudaFuncGetAttributes(
        &attr,
        reinterpret_cast<const void*>(streaming_attn_bf16_fp32_kernel));
    if (err != cudaSuccess) {
        return err;
    }
    if (max_threads_per_block) {
        *max_threads_per_block = attr.maxThreadsPerBlock;
    }
    if (static_shared_bytes) {
        *static_shared_bytes = attr.sharedSizeBytes;
    }
    if (binary_version) {
        *binary_version = attr.binaryVersion;
    }
    return cudaSuccess;
}
