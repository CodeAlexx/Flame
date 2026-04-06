// flash_attention_fwd.cu
// Flash Attention forward pass for FLAME inference.
// BF16 in/out, FP32 accumulation, online softmax (log-sum-exp trick).
// No backward pass needed — inference only.
//
// Supports head_dim = 64, 96, 128 via compile-time specialization.
// SD3 uses 64, Mistral 96, FLUX/LTX/Klein 128.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

extern "C" {

// Shared tile sizes (same for all HD variants)
#define BQ 32
#define BKV 32
#define THREADS 128

// Parameterized kernel body — generates one __global__ function per HD
#define DEFINE_FLASH_ATTN_KERNEL(HD)                                          \
__global__ void flash_attn_fwd_hd##HD(                                        \
    const __nv_bfloat16* __restrict__ Q,                                      \
    const __nv_bfloat16* __restrict__ K,                                      \
    const __nv_bfloat16* __restrict__ V,                                      \
    __nv_bfloat16* __restrict__ O,                                            \
    const int N_q,                                                            \
    const int N_kv,                                                           \
    const float scale                                                         \
) {                                                                           \
    const int bh = blockIdx.x;                                                \
    const int q_tile = blockIdx.y;                                            \
    const int tid = threadIdx.x;                                              \
                                                                              \
    const int q_start = q_tile * BQ;                                          \
    if (q_start >= N_q) return;                                               \
    const int q_rows = min(BQ, N_q - q_start);                                \
                                                                              \
    const __nv_bfloat16* Q_base = Q + (size_t)bh * N_q * HD + (size_t)q_start * HD; \
    const __nv_bfloat16* K_base = K + (size_t)bh * N_kv * HD;                 \
    const __nv_bfloat16* V_base = V + (size_t)bh * N_kv * HD;                 \
    __nv_bfloat16* O_base = O + (size_t)bh * N_q * HD + (size_t)q_start * HD; \
                                                                              \
    extern __shared__ float smem[];                                           \
    float* s_Q = smem;                                                        \
    float* s_K = s_Q + BQ * HD;                                               \
    float* s_V = s_K + BKV * HD;                                              \
    float* s_O = s_V + BKV * HD;                                              \
    float* s_S = s_O + BQ * HD;                                               \
    float* s_m = s_S + BQ * BKV;                                              \
    float* s_l = s_m + BQ;                                                    \
                                                                              \
    for (int i = tid; i < q_rows * HD; i += THREADS) {                        \
        s_Q[i] = __bfloat162float(Q_base[i]);                                 \
    }                                                                         \
    for (int i = tid; i < q_rows * HD; i += THREADS) {                        \
        s_O[i] = 0.0f;                                                        \
    }                                                                         \
    for (int i = tid; i < q_rows; i += THREADS) {                             \
        s_m[i] = -FLT_MAX;                                                    \
        s_l[i] = 0.0f;                                                        \
    }                                                                         \
    __syncthreads();                                                          \
                                                                              \
    const int num_kv_tiles = (N_kv + BKV - 1) / BKV;                          \
                                                                              \
    for (int kv_t = 0; kv_t < num_kv_tiles; kv_t++) {                         \
        const int kv_start = kv_t * BKV;                                      \
        const int kv_rows = min(BKV, N_kv - kv_start);                        \
                                                                              \
        const __nv_bfloat16* K_tile = K_base + (size_t)kv_start * HD;         \
        for (int i = tid; i < kv_rows * HD; i += THREADS) {                   \
            s_K[i] = __bfloat162float(K_tile[i]);                             \
        }                                                                     \
                                                                              \
        const __nv_bfloat16* V_tile = V_base + (size_t)kv_start * HD;         \
        for (int i = tid; i < kv_rows * HD; i += THREADS) {                   \
            s_V[i] = __bfloat162float(V_tile[i]);                             \
        }                                                                     \
        __syncthreads();                                                      \
                                                                              \
        for (int idx = tid; idx < q_rows * kv_rows; idx += THREADS) {         \
            const int qi = idx / kv_rows;                                     \
            const int kj = idx % kv_rows;                                     \
            float dot = 0.0f;                                                 \
            const float* q_row = s_Q + qi * HD;                               \
            const float* k_row = s_K + kj * HD;                               \
            _Pragma("unroll 16")                                              \
            for (int d = 0; d < HD; d++) {                                    \
                dot += q_row[d] * k_row[d];                                   \
            }                                                                 \
            s_S[qi * BKV + kj] = dot * scale;                                 \
        }                                                                     \
        __syncthreads();                                                      \
                                                                              \
        for (int qi = tid; qi < q_rows; qi += THREADS) {                      \
            float old_max = s_m[qi];                                          \
            float tile_max = -FLT_MAX;                                        \
            for (int j = 0; j < kv_rows; j++) {                               \
                tile_max = fmaxf(tile_max, s_S[qi * BKV + j]);                \
            }                                                                 \
            float new_max = fmaxf(old_max, tile_max);                         \
            float correction = expf(old_max - new_max);                       \
            s_l[qi] *= correction;                                            \
            float tile_sum = 0.0f;                                            \
            for (int j = 0; j < kv_rows; j++) {                               \
                float p = expf(s_S[qi * BKV + j] - new_max);                  \
                s_S[qi * BKV + j] = p;                                        \
                tile_sum += p;                                                \
            }                                                                 \
            s_l[qi] += tile_sum;                                              \
            s_m[qi] = new_max;                                                \
            for (int d = 0; d < HD; d++) {                                    \
                s_O[qi * HD + d] *= correction;                               \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
                                                                              \
        for (int idx = tid; idx < q_rows * HD; idx += THREADS) {              \
            const int qi = idx / HD;                                          \
            const int d = idx % HD;                                           \
            float sum = 0.0f;                                                 \
            for (int j = 0; j < kv_rows; j++) {                               \
                sum += s_S[qi * BKV + j] * s_V[j * HD + d];                   \
            }                                                                 \
            s_O[qi * HD + d] += sum;                                          \
        }                                                                     \
        __syncthreads();                                                      \
    }                                                                         \
                                                                              \
    for (int idx = tid; idx < q_rows * HD; idx += THREADS) {                  \
        const int qi = idx / HD;                                              \
        float inv_l = 1.0f / s_l[qi];                                         \
        s_O[idx] *= inv_l;                                                    \
    }                                                                         \
    __syncthreads();                                                          \
                                                                              \
    for (int i = tid; i < q_rows * HD; i += THREADS) {                        \
        O_base[i] = __float2bfloat16(s_O[i]);                                 \
    }                                                                         \
}

// Generate kernel specializations
DEFINE_FLASH_ATTN_KERNEL(64)
DEFINE_FLASH_ATTN_KERNEL(96)
DEFINE_FLASH_ATTN_KERNEL(128)


// Entry point for flame-core FFI
//
// Q, K, V: [B*H, N, D] BF16 — heads already split, contiguous per head
// O:       [B*H, N, D] BF16 output
//
// Supports head_dim = 64, 96, 128.
int flame_flash_attention_bf16(
    const void* Q,
    const void* K,
    const void* V,
    void* O,
    int batch_heads,     // B * H
    int seq_len_q,       // N_q
    int seq_len_kv,      // N_kv (can differ from N_q for cross-attention)
    int head_dim,        // 64, 96, or 128
    void* stream
) {
    dim3 grid(batch_heads, (seq_len_q + BQ - 1) / BQ);
    dim3 block(THREADS);
    cudaStream_t s = (cudaStream_t)stream;
    float scale = 1.0f / sqrtf((float)head_dim);

    #define LAUNCH_FLASH(HD) do {                                             \
        size_t smem_size = (BQ*HD + BKV*HD + BKV*HD + BQ*HD + BQ*BKV + BQ + BQ) * sizeof(float); \
        cudaFuncSetAttribute(                                                 \
            flash_attn_fwd_hd##HD,                                            \
            cudaFuncAttributeMaxDynamicSharedMemorySize,                      \
            (int)smem_size                                                    \
        );                                                                    \
        flash_attn_fwd_hd##HD<<<grid, block, smem_size, s>>>(                 \
            (const __nv_bfloat16*)Q,                                          \
            (const __nv_bfloat16*)K,                                          \
            (const __nv_bfloat16*)V,                                          \
            (__nv_bfloat16*)O,                                                \
            seq_len_q,                                                        \
            seq_len_kv,                                                       \
            scale                                                             \
        );                                                                    \
    } while (0)

    switch (head_dim) {
        case 64:  LAUNCH_FLASH(64);  break;
        case 96:  LAUNCH_FLASH(96);  break;
        case 128: LAUNCH_FLASH(128); break;
        default: return -1;  // unsupported head dim
    }

    #undef LAUNCH_FLASH
    return cudaGetLastError();
}

} // extern "C"
