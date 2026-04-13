// flash_attention_bwd.cu
// Flash Attention backward pass for FLAME training — WMMA tensor core version.
// BF16 in/out, FP32 accumulation via tensor cores.
//
// Computes dQ, dK, dV given Q, K, V, O (forward output), dO (upstream grad),
// and LSE (log-sum-exp from forward pass).
//
// Supports head_dim = 64, 96, 128 via compile-time specialization.
//
// Architecture: SM_80+ (Ampere, Ada, Hopper). Uses nvcuda::wmma BF16 fragments.
//
// Algorithm: FlashAttention-2 backward with KV-outer loop.
//   For each KV tile j (grid.y):
//     Load Kj, Vj; init dK/dV accumulators in REGISTERS (not shared mem)
//     For each Q tile i (inner loop):
//       Stage 1: S_ij = Qi @ Kj^T * scale          (recompute scores)
//       Stage 2: P = exp(S - LSE), D = rowsum(dO*O) (recompute attn weights)
//       Stage 3: dV_reg += P^T @ dO                 (register accumulation)
//       Stage 4: dP = dO @ V^T                      (scratch in s_S)
//       Stage 5: dS = P * (dP - D)                  (overwrites s_P)
//       Stage 6: dQ += dS @ K * scale               (atomicAdd to global FP32)
//       Stage 7: dK_reg += dS^T @ Q                 (register accumulation)
//     Write dK_reg, dV_reg to global BF16 (single write, no atomics)
//
// Only dQ uses FP32 global staging + atomicAdd (multiple KV-tile blocks write
// to the same Q rows). dK and dV use register-based accumulation — each
// KV-tile block is the sole writer for its chunk.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math.h>
#include <float.h>

using namespace nvcuda;

extern "C" {

#define BQ 32
#define BKV 64
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define NUM_WARPS 8
#define THREADS (NUM_WARPS * 32)

__device__ __forceinline__ void load_tile_bf16_bwd(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    int valid_rows, int cols, int buf_rows, int global_stride,
    int num_threads, int tid
) {
    const int total = buf_rows * cols;
    for (int i = tid; i < total; i += num_threads) {
        int r = i / cols;
        int c = i - r * cols;
        dst[i] = (r < valid_rows) ? src[r * global_stride + c] : __float2bfloat16(0.0f);
    }
}

__device__ __forceinline__ void load_tile_f32_bwd(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int valid_count, int buf_count,
    int num_threads, int tid
) {
    for (int i = tid; i < buf_count; i += num_threads) {
        dst[i] = (i < valid_count) ? src[i] : 0.0f;
    }
}

// ============================================================================
// Backward kernel macro. Grid: (batch_heads, ceil(N_kv/BKV)), Block: 256
//
// dK, dV: accumulated in wmma register fragments across Q tiles, written
//         once to global BF16 at end. No atomics, no staging buffer.
// dQ:     accumulated via atomicAdd to caller-provided FP32 staging buffer.
//
// Shared memory (~60KB for HD=128):
//   s_K:   [BKV, HD]  bf16   (persists)
//   s_V:   [BKV, HD]  bf16   (persists)
//   s_Q:   [BQ, HD]   bf16   (per Q tile)
//   s_dO:  [BQ, HD]   bf16   (per Q tile)
//   s_P:   [BQ, BKV]  bf16   (recomputed P / overwritten with dS)
//   s_S:   [BQ, BKV]  f32    (scratch: scores, dP, dS, wmma staging)
//   s_LSE: [BQ]       f32
//   s_D:   [BQ]       f32
// ============================================================================
#define DEFINE_FLASH_ATTN_BWD_KERNEL(HD)                                      \
__global__ void flash_attn_bwd_hd##HD(                                        \
    const __nv_bfloat16* __restrict__ Q,                                      \
    const __nv_bfloat16* __restrict__ K,                                      \
    const __nv_bfloat16* __restrict__ V,                                      \
    const __nv_bfloat16* __restrict__ O,                                      \
    const __nv_bfloat16* __restrict__ dO,                                     \
    const float* __restrict__ LSE,                                            \
    float* __restrict__ dQ_f32,                                               \
    __nv_bfloat16* __restrict__ dK_out,                                       \
    __nv_bfloat16* __restrict__ dV_out,                                       \
    const int N_q,                                                            \
    const int N_kv,                                                           \
    const float scale                                                         \
) {                                                                           \
    const int bh = blockIdx.x;                                                \
    const int kv_tile_idx = blockIdx.y;                                       \
    const int tid = threadIdx.x;                                              \
    const int warp_id = tid / 32;                                             \
    const int lane_id = tid % 32;                                             \
                                                                              \
    const int kv_start = kv_tile_idx * BKV;                                   \
    if (kv_start >= N_kv) return;                                             \
    const int kv_rows = min(BKV, N_kv - kv_start);                           \
                                                                              \
    const __nv_bfloat16* Q_bh  = Q  + (size_t)bh * N_q  * HD;               \
    const __nv_bfloat16* K_bh  = K  + (size_t)bh * N_kv * HD                 \
                                     + (size_t)kv_start * HD;                 \
    const __nv_bfloat16* V_bh  = V  + (size_t)bh * N_kv * HD                 \
                                     + (size_t)kv_start * HD;                 \
    const __nv_bfloat16* O_bh  = O  + (size_t)bh * N_q  * HD;               \
    const __nv_bfloat16* dO_bh = dO + (size_t)bh * N_q  * HD;               \
    const float* LSE_bh        = LSE + (size_t)bh * N_q;                     \
    float* dQ_bh               = dQ_f32 + (size_t)bh * N_q * HD;            \
    __nv_bfloat16* dK_bh = dK_out + (size_t)bh * N_kv * HD                   \
                                   + (size_t)kv_start * HD;                   \
    __nv_bfloat16* dV_bh = dV_out + (size_t)bh * N_kv * HD                   \
                                   + (size_t)kv_start * HD;                   \
                                                                              \
    extern __shared__ char smem_raw[];                                        \
    __nv_bfloat16* s_K  = (__nv_bfloat16*)smem_raw;                          \
    __nv_bfloat16* s_V  = s_K + BKV * HD;                                    \
    __nv_bfloat16* s_Q  = s_V + BKV * HD;                                    \
    __nv_bfloat16* s_dO = s_Q + BQ * HD;                                     \
    __nv_bfloat16* s_P  = s_dO + BQ * HD;                                    \
    float* s_S          = (float*)(s_P + BQ * BKV);                          \
    float* s_LSE        = s_S + BQ * BKV;                                    \
    float* s_D          = s_LSE + BQ;                                        \
                                                                              \
    load_tile_bf16_bwd(s_K, K_bh, kv_rows, HD, BKV, HD, THREADS, tid);       \
    load_tile_bf16_bwd(s_V, V_bh, kv_rows, HD, BKV, HD, THREADS, tid);       \
    __syncthreads();                                                          \
                                                                              \
    /* ---- dK/dV register accumulators (persist across Q loop) ---- */       \
    /* Each warp handles specific [BKV, HD] output tiles. For BKV=64,HD=128:*/\
    /* 4 KV-row groups × 8 HD-col groups = 32 tiles of 16x16.               */\
    /* 8 warps, each covers ceil(32/8)=4 tiles. Store as arrays of frags.   */\
    /*                                                                       */\
    /* Warp tile assignment for [BKV, HD] outputs:                           */\
    /* We iterate over row_iter (BKV rows) and ht (HD cols).                 */\
    /* Same pattern as the forward P@V stage.                                */\
    const int num_hd_tiles_kv = HD / WMMA_N;                                  \
    const int num_kv_row_groups = BKV / (2 * WMMA_M);                        \
    const int tiles_per_warp_kv = (num_hd_tiles_kv + 3) / 4;                 \
    /* Max tiles per warp: num_kv_row_groups * tiles_per_warp_kv */           \
    /* For HD=128: 2 * 2 = 4 tiles per warp */                               \
    /* For HD=64: 2 * 1 = 2 tiles per warp */                                \
    /* We'll use a fixed-size array: max 4 for HD=128 */                      \
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>         \
        dk_accum[2][2], dv_accum[2][2];                                       \
    for (int ri = 0; ri < 2; ri++)                                            \
        for (int hi = 0; hi < 2; hi++) {                                      \
            wmma::fill_fragment(dk_accum[ri][hi], 0.0f);                      \
            wmma::fill_fragment(dv_accum[ri][hi], 0.0f);                      \
        }                                                                     \
                                                                              \
    const int num_q_tiles = (N_q + BQ - 1) / BQ;                             \
    const int warp_qi = (warp_id / 4) * WMMA_M;                              \
    const int warp_kj = (warp_id % 4) * WMMA_N;                              \
                                                                              \
    for (int q_t = 0; q_t < num_q_tiles; q_t++) {                            \
        const int q_start = q_t * BQ;                                        \
        const int q_rows = min(BQ, N_q - q_start);                           \
                                                                              \
        load_tile_bf16_bwd(s_Q,  Q_bh  + (size_t)q_start * HD,               \
                           q_rows, HD, BQ, HD, THREADS, tid);                 \
        load_tile_bf16_bwd(s_dO, dO_bh + (size_t)q_start * HD,               \
                           q_rows, HD, BQ, HD, THREADS, tid);                 \
        load_tile_f32_bwd(s_LSE, LSE_bh + q_start, q_rows, BQ,               \
                          THREADS, tid);                                      \
        __syncthreads();                                                      \
                                                                              \
        /* ======== STAGE 1: S = Q @ K^T * scale ======== */                  \
        {                                                                     \
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc; \
            wmma::fill_fragment(acc, 0.0f);                                   \
            for (int hd = 0; hd < HD; hd += WMMA_K) {                        \
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,       \
                               __nv_bfloat16, wmma::row_major> q_frag;        \
                wmma::load_matrix_sync(q_frag, s_Q + warp_qi * HD + hd, HD);  \
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,       \
                               __nv_bfloat16, wmma::col_major> k_frag;        \
                wmma::load_matrix_sync(k_frag, s_K + warp_kj * HD + hd, HD);  \
                wmma::mma_sync(acc, q_frag, k_frag, acc);                     \
            }                                                                 \
            for (int i = 0; i < acc.num_elements; i++) acc.x[i] *= scale;     \
            wmma::store_matrix_sync(                                          \
                s_S + warp_qi * BKV + warp_kj, acc, BKV, wmma::mem_row_major); \
        }                                                                     \
        __syncthreads();                                                      \
                                                                              \
        /* ======== STAGE 2: P = exp(S-LSE), D = rowsum(dO*O) ======== */     \
        {                                                                     \
            for (int qi = warp_id; qi < BQ; qi += NUM_WARPS) {                \
                float d_sum = 0.0f;                                           \
                if (qi < q_rows) {                                            \
                    const __nv_bfloat16* O_row =                              \
                        O_bh + (size_t)(q_start + qi) * HD;                   \
                    for (int d = lane_id; d < HD; d += 32) {                  \
                        d_sum += __bfloat162float(s_dO[qi * HD + d])          \
                               * __bfloat162float(O_row[d]);                  \
                    }                                                         \
                    for (int off = 16; off > 0; off >>= 1)                    \
                        d_sum += __shfl_xor_sync(0xffffffff, d_sum, off);     \
                }                                                             \
                if (lane_id == 0) s_D[qi] = d_sum;                           \
                                                                              \
                float lse_val = s_LSE[qi];                                    \
                for (int c = lane_id; c < BKV; c += 32) {                     \
                    float p = 0.0f;                                           \
                    if (qi < q_rows && c < kv_rows)                           \
                        p = __expf(s_S[qi * BKV + c] - lse_val);             \
                    s_P[qi * BKV + c] = __float2bfloat16(p);                  \
                    s_S[qi * BKV + c] = p;                                    \
                }                                                             \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
                                                                              \
        /* ======== STAGE 3: dV_reg += P^T @ dO (register accum) ====== */   \
        {                                                                     \
            for (int ri = 0; ri < num_kv_row_groups; ri++) {                  \
                int ki_base = (warp_id / 4) * WMMA_M + ri * 2 * WMMA_M;     \
                if (ki_base >= BKV) break;                                    \
                for (int hi = 0; hi < tiles_per_warp_kv; hi++) {              \
                    int hd_tile = (warp_id % 4) + hi * 4;                    \
                    if (hd_tile >= num_hd_tiles_kv) break;                   \
                    int hd_base = hd_tile * WMMA_N;                          \
                                                                              \
                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> local_acc; \
                    wmma::fill_fragment(local_acc, 0.0f);                     \
                    for (int qq = 0; qq < BQ; qq += WMMA_K) {                \
                        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, \
                                       __nv_bfloat16, wmma::col_major> pt_frag; \
                        wmma::load_matrix_sync(pt_frag,                      \
                            s_P + qq * BKV + ki_base, BKV);                  \
                        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, \
                                       __nv_bfloat16, wmma::row_major> do_frag; \
                        wmma::load_matrix_sync(do_frag,                      \
                            s_dO + qq * HD + hd_base, HD);                   \
                        wmma::mma_sync(local_acc, pt_frag, do_frag, local_acc); \
                    }                                                         \
                    /* Accumulate into persistent register fragment */         \
                    for (int i = 0; i < local_acc.num_elements; i++)          \
                        dv_accum[ri][hi].x[i] += local_acc.x[i];             \
                }                                                             \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
                                                                              \
        /* ======== STAGE 4: dP = dO @ V^T (into s_S) ======== */            \
        {                                                                     \
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc; \
            wmma::fill_fragment(acc, 0.0f);                                   \
            for (int hd = 0; hd < HD; hd += WMMA_K) {                        \
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,       \
                               __nv_bfloat16, wmma::row_major> do_frag;       \
                wmma::load_matrix_sync(do_frag,                               \
                    s_dO + warp_qi * HD + hd, HD);                            \
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,       \
                               __nv_bfloat16, wmma::col_major> v_frag;        \
                wmma::load_matrix_sync(v_frag,                                \
                    s_V + warp_kj * HD + hd, HD);                             \
                wmma::mma_sync(acc, do_frag, v_frag, acc);                    \
            }                                                                 \
            wmma::store_matrix_sync(                                          \
                s_S + warp_qi * BKV + warp_kj, acc, BKV, wmma::mem_row_major); \
        }                                                                     \
        __syncthreads();                                                      \
                                                                              \
        /* ======== STAGE 5: dS = P * (dP - D), write BF16 to s_P ==== */    \
        {                                                                     \
            for (int i = tid; i < BQ * BKV; i += THREADS) {                   \
                int r = i / BKV;                                              \
                float p_val = __bfloat162float(s_P[i]);                       \
                float ds = p_val * (s_S[i] - s_D[r]);                         \
                s_S[i] = ds;                                                  \
                s_P[i] = __float2bfloat16(ds);                                \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
                                                                              \
        /* ======== STAGE 6: dQ += dS @ K * scale (atomicAdd global) == */    \
        {                                                                     \
            const int dq_qi = (warp_id / 4) * WMMA_M;                        \
            const int num_hd_tiles = HD / WMMA_N;                             \
            const int tiles_per_warp = (num_hd_tiles + 3) / 4;               \
            for (int ht = 0; ht < tiles_per_warp; ht++) {                     \
                int hd_tile = (warp_id % 4) + ht * 4;                        \
                if (hd_tile >= num_hd_tiles) break;                           \
                int hd_base = hd_tile * WMMA_N;                               \
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dq_acc; \
                wmma::fill_fragment(dq_acc, 0.0f);                            \
                for (int kv = 0; kv < BKV; kv += WMMA_K) {                    \
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,   \
                                   __nv_bfloat16, wmma::row_major> ds_frag;   \
                    wmma::load_matrix_sync(ds_frag,                           \
                        s_P + dq_qi * BKV + kv, BKV);                         \
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,   \
                                   __nv_bfloat16, wmma::row_major> k_frag;    \
                    wmma::load_matrix_sync(k_frag,                            \
                        s_K + kv * HD + hd_base, HD);                         \
                    wmma::mma_sync(dq_acc, ds_frag, k_frag, dq_acc);          \
                }                                                             \
                float* scratch = s_S + warp_id * WMMA_M * WMMA_N;             \
                wmma::store_matrix_sync(scratch, dq_acc, WMMA_N,              \
                                        wmma::mem_row_major);                 \
                for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {        \
                    int r = i >> 4;                                           \
                    int c = i & 15;                                           \
                    int gq = q_start + dq_qi + r;                             \
                    if (gq < N_q)                                             \
                        atomicAdd(&dQ_bh[gq * HD + hd_base + c],             \
                                  scratch[i] * scale);                        \
                }                                                             \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
                                                                              \
        /* ======== STAGE 7: dK_reg += dS^T @ Q (register accum) ==== */     \
        {                                                                     \
            for (int ri = 0; ri < num_kv_row_groups; ri++) {                  \
                int ki_base = (warp_id / 4) * WMMA_M + ri * 2 * WMMA_M;     \
                if (ki_base >= BKV) break;                                    \
                for (int hi = 0; hi < tiles_per_warp_kv; hi++) {              \
                    int hd_tile = (warp_id % 4) + hi * 4;                    \
                    if (hd_tile >= num_hd_tiles_kv) break;                   \
                    int hd_base = hd_tile * WMMA_N;                          \
                                                                              \
                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> local_acc; \
                    wmma::fill_fragment(local_acc, 0.0f);                     \
                    for (int qq = 0; qq < BQ; qq += WMMA_K) {                \
                        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, \
                                       __nv_bfloat16, wmma::col_major> dst_frag; \
                        wmma::load_matrix_sync(dst_frag,                      \
                            s_P + qq * BKV + ki_base, BKV);                   \
                        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, \
                                       __nv_bfloat16, wmma::row_major> q_frag; \
                        wmma::load_matrix_sync(q_frag,                        \
                            s_Q + qq * HD + hd_base, HD);                     \
                        wmma::mma_sync(local_acc, dst_frag, q_frag, local_acc); \
                    }                                                         \
                    for (int i = 0; i < local_acc.num_elements; i++)          \
                        dk_accum[ri][hi].x[i] += local_acc.x[i];             \
                }                                                             \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
    }  /* end Q tile loop */                                                  \
                                                                              \
    /* ---- Write dK, dV from register accumulators to global BF16 ---- */    \
    /* Use s_S as scratch for wmma store → BF16 conversion */                 \
    {                                                                         \
        for (int ri = 0; ri < num_kv_row_groups; ri++) {                      \
            int ki_base = (warp_id / 4) * WMMA_M + ri * 2 * WMMA_M;         \
            if (ki_base >= BKV) break;                                        \
            for (int hi = 0; hi < tiles_per_warp_kv; hi++) {                  \
                int hd_tile = (warp_id % 4) + hi * 4;                        \
                if (hd_tile >= num_hd_tiles_kv) break;                       \
                int hd_base = hd_tile * WMMA_N;                              \
                                                                              \
                /* Write dV tile */                                           \
                float* scratch = s_S + warp_id * WMMA_M * WMMA_N;            \
                wmma::store_matrix_sync(scratch, dv_accum[ri][hi], WMMA_N,   \
                                        wmma::mem_row_major);                 \
                for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {        \
                    int r = i >> 4;                                           \
                    int c = i & 15;                                           \
                    if (ki_base + r < kv_rows)                                \
                        dV_bh[(ki_base + r) * HD + hd_base + c] =            \
                            __float2bfloat16(scratch[i]);                     \
                }                                                             \
                                                                              \
                /* Write dK tile (with scale) */                              \
                wmma::store_matrix_sync(scratch, dk_accum[ri][hi], WMMA_N,   \
                                        wmma::mem_row_major);                 \
                for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {        \
                    int r = i >> 4;                                           \
                    int c = i & 15;                                           \
                    if (ki_base + r < kv_rows)                                \
                        dK_bh[(ki_base + r) * HD + hd_base + c] =            \
                            __float2bfloat16(scratch[i] * scale);             \
                }                                                             \
            }                                                                 \
        }                                                                     \
    }                                                                         \
}

DEFINE_FLASH_ATTN_BWD_KERNEL(64)
DEFINE_FLASH_ATTN_BWD_KERNEL(96)
DEFINE_FLASH_ATTN_BWD_KERNEL(128)

// FP32 → BF16 conversion kernel (for dQ staging buffer)
__global__ void convert_f32_to_bf16_bwd(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

// ============================================================================
// Entry point. dQ uses FP32 staging (caller-allocated, pre-zeroed).
// dK, dV written directly as BF16 by the kernel (no staging needed).
// ============================================================================
int flame_flash_attention_backward_bf16(
    const void* Q,
    const void* K,
    const void* V,
    const void* O,
    const void* dO,
    const void* LSE,
    void* dQ,
    void* dK,
    void* dV,
    int batch_heads,
    int seq_len_q,
    int seq_len_kv,
    int head_dim,
    void* stream,
    float* dQ_f32     // Pre-allocated FP32 staging [BH, N_q, HD], must be zeroed
) {
    cudaStream_t s = (cudaStream_t)stream;
    float scale_val = 1.0f / sqrtf((float)head_dim);

    dim3 grid(batch_heads, (seq_len_kv + BKV - 1) / BKV);
    dim3 block(THREADS);

    #define LAUNCH_FLASH_BWD(HD) do {                                         \
        size_t smem_bf16 = ((size_t)BKV*(HD) + BKV*(HD) + BQ*(HD) + BQ*(HD)  \
                            + BQ*BKV) * sizeof(__nv_bfloat16);                \
        size_t smem_f32  = ((size_t)BQ*BKV + BQ + BQ) * sizeof(float);       \
        size_t smem_size = smem_bf16 + smem_f32;                              \
        cudaError_t attr_err = cudaFuncSetAttribute(                          \
            flash_attn_bwd_hd##HD,                                            \
            cudaFuncAttributeMaxDynamicSharedMemorySize,                      \
            (int)smem_size                                                    \
        );                                                                    \
        if (attr_err != cudaSuccess) return (int)attr_err;                    \
        flash_attn_bwd_hd##HD<<<grid, block, smem_size, s>>>(                 \
            (const __nv_bfloat16*)Q,                                          \
            (const __nv_bfloat16*)K,                                          \
            (const __nv_bfloat16*)V,                                          \
            (const __nv_bfloat16*)O,                                          \
            (const __nv_bfloat16*)dO,                                         \
            (const float*)LSE,                                                \
            dQ_f32,                                                           \
            (__nv_bfloat16*)dK,                                               \
            (__nv_bfloat16*)dV,                                               \
            seq_len_q,                                                        \
            seq_len_kv,                                                       \
            scale_val                                                         \
        );                                                                    \
    } while (0)

    switch (head_dim) {
        case 64:  LAUNCH_FLASH_BWD(64);  break;
        case 96:  LAUNCH_FLASH_BWD(96);  break;
        case 128: LAUNCH_FLASH_BWD(128); break;
        default: return -1;
    }

    #undef LAUNCH_FLASH_BWD

    // Convert FP32 dQ staging buffer to BF16 output
    {
        int total = batch_heads * seq_len_q * head_dim;
        int conv_threads = 256;
        convert_f32_to_bf16_bwd<<<(total + conv_threads - 1) / conv_threads,
                                  conv_threads, 0, s>>>(
            dQ_f32, (__nv_bfloat16*)dQ, total);
    }

    return (int)cudaGetLastError();
}

} // extern "C"
