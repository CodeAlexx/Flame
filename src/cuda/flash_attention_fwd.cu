// flash_attention_fwd_fixed.cu
//
// User-provided (2026-04-17) alternative FA2 forward kernel.
//
// Purpose: reduced KV tile (32) for HD=128 to fit shared-memory budget on
// 3090/4090-class cards; adds ragged-tail padding/zeroing and lifts the
// specialization into a single generic templated kernel with runtime
// head_dim dispatch (64/96/128).
//
// STATUS: saved as backup. Current in-tree kernel
// (`flash_attention_fwd.cu`, Phase 1.6) already passes
// `tests/fa2_parity_naive.rs` and runs clean end-to-end for FLUX.1-dev and
// ERNIE-Image in the inference-flame harness (see commit history
// 2026-04-17). No live swap performed — kept on disk for reference, future
// porting, or if later work exposes a shape that the current kernel can't
// service.
//
// DO NOT include this file in build.rs as-is. It defines the same
// extern-C symbol `flame_flash_attention_bf16` as the live kernel and will
// conflict at link time if both are compiled.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math.h>
#include <float.h>

using namespace nvcuda;

// Templates cannot be extern "C" — keep helpers/kernel at C++ linkage and
// expose only the final launcher through an extern "C" shim at the bottom.

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

template<int TILE_Q, int TILE_KV, int HD, int NUM_WARPS>
__device__ __forceinline__ void load_tile_bf16_zero_padded(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    int valid_rows,
    int cols,
    int src_stride,
    int tid,
    int num_threads
) {
    const int total = TILE_Q * cols;
    for (int i = tid; i < total; i += num_threads) {
        int r = i / cols;
        int c = i % cols;
        dst[i] = (r < valid_rows) ? src[r * src_stride + c] : __float2bfloat16(0.0f);
    }
}

template<int TILE_ROWS, int TILE_COLS>
__device__ __forceinline__ void zero_tail_2d_float(
    float* ptr,
    int valid_rows,
    int valid_cols,
    int tid,
    int num_threads
) {
    const int total = TILE_ROWS * TILE_COLS;
    for (int i = tid; i < total; i += num_threads) {
        int r = i / TILE_COLS;
        int c = i % TILE_COLS;
        if (r >= valid_rows || c >= valid_cols) ptr[i] = 0.0f;
    }
}

template<int TILE_ROWS, int TILE_COLS>
__device__ __forceinline__ void zero_tail_2d_bf16(
    __nv_bfloat16* ptr,
    int valid_rows,
    int valid_cols,
    int tid,
    int num_threads
) {
    const int total = TILE_ROWS * TILE_COLS;
    for (int i = tid; i < total; i += num_threads) {
        int r = i / TILE_COLS;
        int c = i % TILE_COLS;
        if (r >= valid_rows || c >= valid_cols) ptr[i] = __float2bfloat16(0.0f);
    }
}

template<int TILE_Q, int TILE_KV, int HD, int NUM_WARPS>
__global__ void flash_attn_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    int N_q,
    int N_kv,
    float scale
) {
    constexpr int THREADS = NUM_WARPS * 32;

    const int bh = blockIdx.x;
    const int q_tile = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int q_start = q_tile * TILE_Q;
    if (q_start >= N_q) return;
    const int q_rows = min(TILE_Q, N_q - q_start);

    const __nv_bfloat16* Q_base = Q + (size_t)bh * N_q  * HD + (size_t)q_start * HD;
    const __nv_bfloat16* K_base = K + (size_t)bh * N_kv * HD;
    const __nv_bfloat16* V_base = V + (size_t)bh * N_kv * HD;
    __nv_bfloat16* O_base       = O + (size_t)bh * N_q  * HD + (size_t)q_start * HD;

    // K/V-reuse layout (Phase 1.6 style): s_K and s_V share one buffer.
    // K is loaded, used for QK^T, then overwritten by V for PV. Saves
    // TILE_KV*HD*2 bytes of shared memory (16 KB at HD=128, TILE_KV=64)
    // so HD=128 can use TILE_KV=64 within the SM_86 100 KB opt-in budget.
    extern __shared__ char smem_raw[];
    __nv_bfloat16* s_Q = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* s_K = s_Q + TILE_Q * HD;
    __nv_bfloat16* s_V = s_K;  // alias — never alive simultaneously
    float* s_S = reinterpret_cast<float*>(s_K + TILE_KV * HD);
    __nv_bfloat16* s_P = reinterpret_cast<__nv_bfloat16*>(s_S + TILE_Q * TILE_KV);
    float* s_O = reinterpret_cast<float*>(s_P + TILE_Q * TILE_KV);
    float* s_m = s_O + TILE_Q * HD;
    float* s_l = s_m + TILE_Q;

    load_tile_bf16_zero_padded<TILE_Q, TILE_KV, HD, NUM_WARPS>(
        s_Q, Q_base, q_rows, HD, HD, tid, THREADS
    );

    for (int i = tid; i < TILE_Q * HD; i += THREADS) s_O[i] = 0.0f;
    for (int i = tid; i < TILE_Q; i += THREADS) {
        s_m[i] = -FLT_MAX;
        s_l[i] = 0.0f;
    }
    __syncthreads();

    constexpr int WARP_Q_TILES = TILE_Q / 16;
    constexpr int WARP_KV_TILES = TILE_KV / 16;

    const int warp_qi = (warp_id / WARP_KV_TILES) * 16;
    const int warp_kj = (warp_id % WARP_KV_TILES) * 16;

    const int num_kv_tiles = (N_kv + TILE_KV - 1) / TILE_KV;

    for (int kv_t = 0; kv_t < num_kv_tiles; kv_t++) {
        const int kv_start = kv_t * TILE_KV;
        const int kv_rows = min(TILE_KV, N_kv - kv_start);

        // Load K only; V comes after QK^T frees s_K.
        load_tile_bf16_zero_padded<TILE_KV, TILE_KV, HD, NUM_WARPS>(
            s_K, K_base + (size_t)kv_start * HD, kv_rows, HD, HD, tid, THREADS
        );
        __syncthreads();

        if (warp_qi < TILE_Q && warp_kj < TILE_KV) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);

            for (int hd = 0; hd < HD; hd += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> k_frag;

                wmma::load_matrix_sync(q_frag, s_Q + warp_qi * HD + hd, HD);
                wmma::load_matrix_sync(k_frag, s_K + warp_kj * HD + hd, HD);
                wmma::mma_sync(acc, q_frag, k_frag, acc);
            }

            for (int i = 0; i < acc.num_elements; i++) acc.x[i] *= scale;
            wmma::store_matrix_sync(s_S + warp_qi * TILE_KV + warp_kj, acc, TILE_KV, wmma::mem_row_major);
        }
        __syncthreads();

        // QK^T store is done for every warp; s_K is no longer needed.
        // Overwrite with V while softmax/P-write run in parallel on s_S.
        // The __syncthreads at the end of the zero_tail block covers V load.
        load_tile_bf16_zero_padded<TILE_KV, TILE_KV, HD, NUM_WARPS>(
            s_V, V_base + (size_t)kv_start * HD, kv_rows, HD, HD, tid, THREADS
        );

        zero_tail_2d_float<TILE_Q, TILE_KV>(s_S, q_rows, kv_rows, tid, THREADS);
        __syncthreads();

        for (int qi = tid; qi < q_rows; qi += THREADS) {
            float old_max = s_m[qi];
            float tile_max = -FLT_MAX;

            #pragma unroll
            for (int j = 0; j < TILE_KV; j++) tile_max = fmaxf(tile_max, s_S[qi * TILE_KV + j]);

            float new_max = fmaxf(old_max, tile_max);
            float corr = __expf(old_max - new_max);

            s_l[qi] *= corr;
            float tile_sum = 0.0f;

            #pragma unroll
            for (int j = 0; j < TILE_KV; j++) {
                float p = __expf(s_S[qi * TILE_KV + j] - new_max);
                s_S[qi * TILE_KV + j] = p;
                tile_sum += p;
            }

            s_l[qi] += tile_sum;
            s_m[qi] = new_max;

            for (int d = 0; d < HD; d++) s_O[qi * HD + d] *= corr;
        }
        __syncthreads();

        for (int i = tid; i < TILE_Q * TILE_KV; i += THREADS) {
            s_P[i] = __float2bfloat16(s_S[i]);
        }
        zero_tail_2d_bf16<TILE_Q, TILE_KV>(s_P, q_rows, kv_rows, tid, THREADS);
        __syncthreads();

        const int num_hd_tiles = HD / 16;
        const int kv_tile_per_rowgroup = WARP_KV_TILES;

        const int row_group = warp_id / kv_tile_per_rowgroup;
        const int col_group = warp_id % kv_tile_per_rowgroup;
        const int qi_base = row_group * 16;

        for (int ht = col_group; ht < num_hd_tiles; ht += kv_tile_per_rowgroup) {
            const int hd_base = ht * 16;

            wmma::fragment<wmma::accumulator, 16, 16, 16, float> pv_acc;
            wmma::fill_fragment(pv_acc, 0.0f);

            for (int kv = 0; kv < TILE_KV; kv += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> p_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> v_frag;

                wmma::load_matrix_sync(p_frag, s_P + qi_base * TILE_KV + kv, TILE_KV);
                wmma::load_matrix_sync(v_frag, s_V + kv * HD + hd_base, HD);
                wmma::mma_sync(pv_acc, p_frag, v_frag, pv_acc);
            }

            float* scratch = s_S + warp_id * 256;
            wmma::store_matrix_sync(scratch, pv_acc, 16, wmma::mem_row_major);

            for (int i = lane; i < 256; i += 32) {
                int r = i >> 4;
                int c = i & 15;
                if (qi_base + r < q_rows) {
                    atomicAdd(&s_O[(qi_base + r) * HD + hd_base + c], scratch[i]);
                }
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < q_rows * HD; i += THREADS) {
        int qi = i / HD;
        float denom = fmaxf(s_l[qi], 1e-20f);
        O_base[i] = __float2bfloat16(s_O[i] / denom);
    }
}

extern "C" {

static inline int launch_fwd(
    const void* Q, const void* K, const void* V, void* O,
    int batch_heads, int seq_len_q, int seq_len_kv, int head_dim, void* stream
) {
    cudaStream_t s = (cudaStream_t)stream;
    float scale = 1.0f / sqrtf((float)head_dim);

    if (head_dim == 64) {
        constexpr int TILE_Q = 64, TILE_KV = 64, HD = 64, NUM_WARPS = 16;
        dim3 grid(batch_heads, (seq_len_q + TILE_Q - 1) / TILE_Q);
        dim3 block(NUM_WARPS * 32);
        // K/V share one buffer — drop one TILE_KV*HD bf16 compared to the
        // pre-reuse layout.
        size_t smem =
            (TILE_Q*HD + TILE_KV*HD + TILE_Q*TILE_KV) * sizeof(__nv_bfloat16) +
            (TILE_Q*TILE_KV + TILE_Q*HD + TILE_Q + TILE_Q) * sizeof(float);
        cudaFuncSetAttribute(
            flash_attn_fwd_kernel<TILE_Q, TILE_KV, HD, NUM_WARPS>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem
        );
        flash_attn_fwd_kernel<TILE_Q, TILE_KV, HD, NUM_WARPS><<<grid, block, smem, s>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (__nv_bfloat16*)O, seq_len_q, seq_len_kv, scale
        );
    } else if (head_dim == 96) {
        constexpr int TILE_Q = 64, TILE_KV = 64, HD = 96, NUM_WARPS = 16;
        dim3 grid(batch_heads, (seq_len_q + TILE_Q - 1) / TILE_Q);
        dim3 block(NUM_WARPS * 32);
        // K/V share one buffer — drop one TILE_KV*HD bf16 compared to the
        // pre-reuse layout.
        size_t smem =
            (TILE_Q*HD + TILE_KV*HD + TILE_Q*TILE_KV) * sizeof(__nv_bfloat16) +
            (TILE_Q*TILE_KV + TILE_Q*HD + TILE_Q + TILE_Q) * sizeof(float);
        cudaFuncSetAttribute(
            flash_attn_fwd_kernel<TILE_Q, TILE_KV, HD, NUM_WARPS>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem
        );
        flash_attn_fwd_kernel<TILE_Q, TILE_KV, HD, NUM_WARPS><<<grid, block, smem, s>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (__nv_bfloat16*)O, seq_len_q, seq_len_kv, scale
        );
    } else if (head_dim == 128) {
        // K/V-reuse layout frees 16 KB shared, enabling TILE_KV=64 at HD=128
        // within the SM_86 100 KB opt-in budget.
        constexpr int TILE_Q = 64, TILE_KV = 64, HD = 128, NUM_WARPS = 16;
        dim3 grid(batch_heads, (seq_len_q + TILE_Q - 1) / TILE_Q);
        dim3 block(NUM_WARPS * 32);
        // K/V share one buffer — drop one TILE_KV*HD bf16 compared to the
        // pre-reuse layout.
        size_t smem =
            (TILE_Q*HD + TILE_KV*HD + TILE_Q*TILE_KV) * sizeof(__nv_bfloat16) +
            (TILE_Q*TILE_KV + TILE_Q*HD + TILE_Q + TILE_Q) * sizeof(float);
        cudaFuncSetAttribute(
            flash_attn_fwd_kernel<TILE_Q, TILE_KV, HD, NUM_WARPS>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem
        );
        flash_attn_fwd_kernel<TILE_Q, TILE_KV, HD, NUM_WARPS><<<grid, block, smem, s>>>(
            (const __nv_bfloat16*)Q, (const __nv_bfloat16*)K, (const __nv_bfloat16*)V,
            (__nv_bfloat16*)O, seq_len_q, seq_len_kv, scale
        );
    } else {
        return -1;
    }

    return (int)cudaGetLastError();
}

int flame_flash_attention_bf16(
    const void* Q,
    const void* K,
    const void* V,
    void* O,
    float* LSE,              // optional log-sum-exp per row; unused here
    int batch_heads,
    int seq_len_q,
    int seq_len_kv,
    int head_dim,
    void* stream
) {
    (void)LSE;  // inference-only kernel; backward path stores LSE separately
    return launch_fwd(Q, K, V, O, batch_heads, seq_len_q, seq_len_kv, head_dim, stream);
}

} // extern "C"
