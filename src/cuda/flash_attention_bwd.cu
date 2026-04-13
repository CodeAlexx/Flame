// flash_attention_bwd.cu
// Flash Attention backward pass for FLAME training — WMMA tensor core version.
// BF16 in/out, FP32 accumulation via tensor cores.
//
// Computes dQ, dK, dV given Q, K, V, O (forward output), dO (upstream grad),
// and LSE (log-sum-exp from forward pass).
//
// Supports head_dim = 64, 96, 128 via compile-time specialization.
// SD3 uses 64, Mistral 96, FLUX/LTX/Klein/Z-Image 128.
//
// Architecture: SM_80+ (Ampere, Ada, Hopper). Uses nvcuda::wmma BF16 fragments.
// Compile with: -arch=sm_80 (or sm_86 for 3090, sm_89 for 4090)
//
// Algorithm: FlashAttention-2 backward with KV-outer loop.
//   For each KV tile j (grid.y):
//     Load Kj, Vj into shared memory
//     For each Q tile i (inner loop):
//       Load Qi, dOi, LSEi into shared memory
//       Stage 1: S_ij = Qi @ Kj^T * scale          (recompute scores via wmma)
//       Stage 2: P = exp(S - LSE), D = rowsum(dO*O) (recompute attn weights)
//       Stage 3: dV += P^T @ dO                     (before P is overwritten)
//       Stage 4: dP = dO @ V^T                      (gradient through V matmul)
//       Stage 5: dS = P * (dP - D)                  (softmax backward)
//       Stage 6: dQ += dS @ K * scale               (atomicAdd to global FP32)
//       Stage 7: dK += dS^T @ Q                     (atomicAdd to global FP32)
//
// All three gradient buffers (dQ, dK, dV) use FP32 global staging with atomicAdd.
// This avoids large shared-memory accumulators that would exceed the SM_86 100KB
// per-block limit. After the kernel, a conversion kernel writes BF16 outputs.
//
// Shared memory budget for HD=128:
//   s_K:  64*128*2=16KB, s_V: 16KB, s_Q: 32*128*2=8KB, s_dO: 8KB,
//   s_P:  32*64*2=4KB, s_S: 32*64*4=8KB, s_LSE+s_D: 256B  => ~60KB total

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math.h>
#include <float.h>

using namespace nvcuda;

extern "C" {

// Tile sizes — match forward kernel conventions
#define BQ 32
#define BKV 64
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define NUM_WARPS 8
#define THREADS (NUM_WARPS * 32)  // 256

// ============================================================================
// Helper: cooperative BF16 tile load from global to shared memory.
// Identical to forward kernel. Pads FULL buffer (buf_rows * cols).
// ============================================================================
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

// ============================================================================
// Helper: cooperative FP32 tile load with padding.
// ============================================================================
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
// Kernel: flash_attn_bwd_hdXX
// Grid:   (batch_heads, ceil(N_kv / BKV))
// Block:  THREADS (256)
//
// Q, K, V, O, dO: [BH, N, HD] BF16 contiguous
// LSE:            [BH, N_q] FP32 (log-sum-exp from forward)
// dQ_f32:         [BH, N_q, HD] FP32 (pre-zeroed, atomicAdd accumulation)
// dK_f32:         [BH, N_kv, HD] FP32 (pre-zeroed, atomicAdd accumulation)
// dV_f32:         [BH, N_kv, HD] FP32 (pre-zeroed, atomicAdd accumulation)
//
// Shared memory layout:
//   s_K:   [BKV, HD]  bf16   (persists across Q loop)
//   s_V:   [BKV, HD]  bf16   (persists across Q loop)
//   s_Q:   [BQ, HD]   bf16   (reloaded per Q tile)
//   s_dO:  [BQ, HD]   bf16   (reloaded per Q tile)
//   s_P:   [BQ, BKV]  bf16   (recomputed P, then overwritten with dS)
//   s_S:   [BQ, BKV]  f32    (scratch for scores/dP/dS)
//   s_LSE: [BQ]       f32
//   s_D:   [BQ]       f32    (rowsum of dO*O)
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
    float* __restrict__ dK_f32,                                               \
    float* __restrict__ dV_f32,                                               \
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
    /* Global memory bases for this batch-head */                             \
    const __nv_bfloat16* Q_bh  = Q  + (size_t)bh * N_q  * HD;               \
    const __nv_bfloat16* K_bh  = K  + (size_t)bh * N_kv * HD                 \
                                     + (size_t)kv_start * HD;                 \
    const __nv_bfloat16* V_bh  = V  + (size_t)bh * N_kv * HD                 \
                                     + (size_t)kv_start * HD;                 \
    const __nv_bfloat16* O_bh  = O  + (size_t)bh * N_q  * HD;               \
    const __nv_bfloat16* dO_bh = dO + (size_t)bh * N_q  * HD;               \
    const float* LSE_bh        = LSE + (size_t)bh * N_q;                     \
    float* dQ_bh               = dQ_f32 + (size_t)bh * N_q  * HD;           \
    float* dK_bh               = dK_f32 + (size_t)bh * N_kv * HD             \
                                         + (size_t)kv_start * HD;            \
    float* dV_bh               = dV_f32 + (size_t)bh * N_kv * HD             \
                                         + (size_t)kv_start * HD;            \
                                                                              \
    /* ---- Shared memory layout ---- */                                      \
    extern __shared__ char smem_raw[];                                        \
    __nv_bfloat16* s_K  = (__nv_bfloat16*)smem_raw;          /* [BKV, HD]  */\
    __nv_bfloat16* s_V  = s_K + BKV * HD;                    /* [BKV, HD]  */\
    __nv_bfloat16* s_Q  = s_V + BKV * HD;                    /* [BQ, HD]   */\
    __nv_bfloat16* s_dO = s_Q + BQ * HD;                     /* [BQ, HD]   */\
    __nv_bfloat16* s_P  = s_dO + BQ * HD;                    /* [BQ, BKV]  */\
    float* s_S          = (float*)(s_P + BQ * BKV);          /* [BQ, BKV]  */\
    float* s_LSE        = s_S + BQ * BKV;                    /* [BQ]       */\
    float* s_D          = s_LSE + BQ;                        /* [BQ]       */\
                                                                              \
    /* ---- Load K, V tiles (persist for entire Q loop) ---- */               \
    load_tile_bf16_bwd(s_K, K_bh, kv_rows, HD, BKV, HD, THREADS, tid);       \
    load_tile_bf16_bwd(s_V, V_bh, kv_rows, HD, BKV, HD, THREADS, tid);       \
    __syncthreads();                                                          \
                                                                              \
    /* ---- Main Q tile loop ---- */                                          \
    const int num_q_tiles = (N_q + BQ - 1) / BQ;                             \
                                                                              \
    /* Warp tile for [BQ, BKV] matmuls: 2x4 grid of 16x16 tiles */          \
    const int warp_qi = (warp_id / 4) * WMMA_M;                              \
    const int warp_kj = (warp_id % 4) * WMMA_N;                              \
                                                                              \
    for (int q_t = 0; q_t < num_q_tiles; q_t++) {                            \
        const int q_start = q_t * BQ;                                        \
        const int q_rows = min(BQ, N_q - q_start);                           \
                                                                              \
        /* Load Q, dO tiles for this Q chunk */                               \
        load_tile_bf16_bwd(s_Q,  Q_bh  + (size_t)q_start * HD,               \
                           q_rows, HD, BQ, HD, THREADS, tid);                 \
        load_tile_bf16_bwd(s_dO, dO_bh + (size_t)q_start * HD,               \
                           q_rows, HD, BQ, HD, THREADS, tid);                 \
        load_tile_f32_bwd(s_LSE, LSE_bh + q_start, q_rows, BQ,               \
                          THREADS, tid);                                      \
        __syncthreads();                                                      \
                                                                              \
        /* ======== STAGE 1: S = Q @ K^T * scale (recompute) ======== */      \
        /* Each warp computes one 16x16 tile of S[BQ, BKV] */                \
        {                                                                     \
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc; \
            wmma::fill_fragment(acc, 0.0f);                                   \
                                                                              \
            for (int hd = 0; hd < HD; hd += WMMA_K) {                        \
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,       \
                               __nv_bfloat16, wmma::row_major> q_frag;        \
                wmma::load_matrix_sync(q_frag,                                \
                    s_Q + warp_qi * HD + hd, HD);                             \
                                                                              \
                /* K^T: load K [BKV,HD] as col_major */                       \
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,       \
                               __nv_bfloat16, wmma::col_major> k_frag;        \
                wmma::load_matrix_sync(k_frag,                                \
                    s_K + warp_kj * HD + hd, HD);                             \
                                                                              \
                wmma::mma_sync(acc, q_frag, k_frag, acc);                     \
            }                                                                 \
                                                                              \
            for (int i = 0; i < acc.num_elements; i++) {                      \
                acc.x[i] *= scale;                                            \
            }                                                                 \
            wmma::store_matrix_sync(                                          \
                s_S + warp_qi * BKV + warp_kj,                                \
                acc, BKV, wmma::mem_row_major);                               \
        }                                                                     \
        __syncthreads();                                                      \
                                                                              \
        /* ======== STAGE 2: P = exp(S - LSE), D = rowsum(dO*O) ======== */   \
        /* Write P as BF16 to s_P. Keep P as FP32 in s_S (temporary). */      \
        /* Compute D reading O from global memory (saves shared memory). */   \
        {                                                                     \
            for (int qi = warp_id; qi < BQ; qi += NUM_WARPS) {                \
                /* D[qi] = sum_d(dO[qi][d] * O[qi][d]) */                    \
                float d_sum = 0.0f;                                           \
                if (qi < q_rows) {                                            \
                    const __nv_bfloat16* O_row =                              \
                        O_bh + (size_t)(q_start + qi) * HD;                   \
                    for (int d = lane_id; d < HD; d += 32) {                  \
                        float dO_val = __bfloat162float(s_dO[qi * HD + d]);   \
                        float O_val  = __bfloat162float(O_row[d]);            \
                        d_sum += dO_val * O_val;                              \
                    }                                                         \
                    _Pragma("unroll")                                         \
                    for (int off = 16; off > 0; off >>= 1) {                  \
                        d_sum += __shfl_xor_sync(0xffffffff, d_sum, off);     \
                    }                                                         \
                }                                                             \
                if (lane_id == 0) {                                           \
                    s_D[qi] = d_sum;                                          \
                }                                                             \
                                                                              \
                /* P[qi][c] = exp(S[qi][c] - LSE[qi]) */                     \
                float lse_val = s_LSE[qi];                                    \
                for (int c = lane_id; c < BKV; c += 32) {                     \
                    float p = 0.0f;                                           \
                    if (qi < q_rows && c < kv_rows) {                         \
                        p = __expf(s_S[qi * BKV + c] - lse_val);             \
                    }                                                         \
                    s_P[qi * BKV + c] = __float2bfloat16(p);                  \
                    s_S[qi * BKV + c] = p;  /* keep FP32 P in s_S */          \
                }                                                             \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
                                                                              \
        /* ======== STAGE 3: dV += P^T @ dO (atomicAdd to global) ===== */   \
        /* dV is [BKV, HD]. P^T is [BKV, BQ]. dO is [BQ, HD]. */            \
        /* Load P [BQ, BKV] as col_major for P^T. */                         \
        /* We must do this BEFORE stage 5 overwrites s_P with dS. */          \
        {                                                                     \
            const int num_hd_tiles = HD / WMMA_N;                             \
            /* 8 warps cover [BKV=64, HD] output. BKV/WMMA_M=4 row tiles, */ \
            /* HD/WMMA_N col tiles. With 2 warp-rows (warp_id/4 = 0 or 1), */\
            /* iterate row_iter to cover all 4 row groups. */                 \
            for (int row_iter = 0; row_iter < BKV / (2 * WMMA_M); row_iter++) { \
                int ki_base = (warp_id / 4) * WMMA_M + row_iter * 2 * WMMA_M; \
                if (ki_base >= BKV) break;                                    \
                                                                              \
                int tiles_per_warp = (num_hd_tiles + 3) / 4;                  \
                for (int ht = 0; ht < tiles_per_warp; ht++) {                 \
                    int hd_tile = (warp_id % 4) + ht * 4;                     \
                    if (hd_tile >= num_hd_tiles) break;                       \
                    int hd_base = hd_tile * WMMA_N;                           \
                                                                              \
                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dv_acc; \
                    wmma::fill_fragment(dv_acc, 0.0f);                        \
                                                                              \
                    for (int qq = 0; qq < BQ; qq += WMMA_K) {                 \
                        /* P^T: P [BQ,BKV] col_major at [qq, ki_base] */      \
                        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, \
                                       __nv_bfloat16, wmma::col_major> pt_frag; \
                        wmma::load_matrix_sync(pt_frag,                       \
                            s_P + qq * BKV + ki_base, BKV);                   \
                                                                              \
                        /* dO: [BQ, HD] row-major at [qq, hd_base] */         \
                        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, \
                                       __nv_bfloat16, wmma::row_major> do_frag; \
                        wmma::load_matrix_sync(do_frag,                       \
                            s_dO + qq * HD + hd_base, HD);                    \
                                                                              \
                        wmma::mma_sync(dv_acc, pt_frag, do_frag, dv_acc);     \
                    }                                                         \
                                                                              \
                    /* Store wmma result to scratch in s_S, then atomicAdd */ \
                    float* scratch = s_S + warp_id * WMMA_M * WMMA_N;         \
                    wmma::store_matrix_sync(scratch, dv_acc, WMMA_N,          \
                                            wmma::mem_row_major);             \
                                                                              \
                    for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {     \
                        int r = i >> 4;                                       \
                        int c = i & 15;                                       \
                        if (ki_base + r < kv_rows) {                          \
                            atomicAdd(&dV_bh[(ki_base + r) * HD + hd_base + c], \
                                      scratch[i]);                            \
                        }                                                     \
                    }                                                         \
                }                                                             \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
                                                                              \
        /* ======== STAGE 4: dP = dO @ V^T via wmma ======== */               \
        /* dP is [BQ, BKV]. dO is [BQ, HD]. V is [BKV, HD]. */               \
        /* Store dP to s_S (overwriting FP32 P — done with it). */            \
        {                                                                     \
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc; \
            wmma::fill_fragment(acc, 0.0f);                                   \
                                                                              \
            for (int hd = 0; hd < HD; hd += WMMA_K) {                        \
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,       \
                               __nv_bfloat16, wmma::row_major> do_frag;       \
                wmma::load_matrix_sync(do_frag,                               \
                    s_dO + warp_qi * HD + hd, HD);                            \
                                                                              \
                /* V^T: load V [BKV,HD] as col_major */                       \
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,       \
                               __nv_bfloat16, wmma::col_major> v_frag;        \
                wmma::load_matrix_sync(v_frag,                                \
                    s_V + warp_kj * HD + hd, HD);                             \
                                                                              \
                wmma::mma_sync(acc, do_frag, v_frag, acc);                    \
            }                                                                 \
                                                                              \
            wmma::store_matrix_sync(                                          \
                s_S + warp_qi * BKV + warp_kj,                                \
                acc, BKV, wmma::mem_row_major);                               \
        }                                                                     \
        __syncthreads();                                                      \
                                                                              \
        /* ======== STAGE 5: dS = P * (dP - D) (elementwise) ======== */      \
        /* P (BF16) is in s_P. dP (FP32) is in s_S. D is in s_D. */          \
        /* Write dS as BF16 to s_P (for wmma in stages 6,7). */              \
        /* Write dS as FP32 to s_S (for dQ scaling). */                       \
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
        /* ======== STAGE 6: dQ += dS @ K * scale (atomicAdd global) ==== */ \
        /* dQ is [BQ, HD]. dS is [BQ, BKV] BF16 in s_P. K is [BKV, HD]. */ \
        {                                                                     \
            const int dq_qi = (warp_id / 4) * WMMA_M;                        \
            const int num_hd_tiles = HD / WMMA_N;                             \
            const int tiles_per_warp = (num_hd_tiles + 3) / 4;               \
                                                                              \
            for (int ht = 0; ht < tiles_per_warp; ht++) {                     \
                int hd_tile = (warp_id % 4) + ht * 4;                        \
                if (hd_tile >= num_hd_tiles) break;                           \
                int hd_base = hd_tile * WMMA_N;                               \
                                                                              \
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dq_acc; \
                wmma::fill_fragment(dq_acc, 0.0f);                            \
                                                                              \
                for (int kv = 0; kv < BKV; kv += WMMA_K) {                    \
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,   \
                                   __nv_bfloat16, wmma::row_major> ds_frag;   \
                    wmma::load_matrix_sync(ds_frag,                           \
                        s_P + dq_qi * BKV + kv, BKV);                         \
                                                                              \
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,   \
                                   __nv_bfloat16, wmma::row_major> k_frag;    \
                    wmma::load_matrix_sync(k_frag,                            \
                        s_K + kv * HD + hd_base, HD);                         \
                                                                              \
                    wmma::mma_sync(dq_acc, ds_frag, k_frag, dq_acc);          \
                }                                                             \
                                                                              \
                /* Store to scratch, apply scale, atomicAdd to global */      \
                float* scratch = s_S + warp_id * WMMA_M * WMMA_N;             \
                wmma::store_matrix_sync(scratch, dq_acc, WMMA_N,              \
                                        wmma::mem_row_major);                 \
                                                                              \
                for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {        \
                    int r = i >> 4;                                           \
                    int c = i & 15;                                           \
                    int gq = q_start + dq_qi + r;                             \
                    if (gq < N_q) {                                           \
                        atomicAdd(&dQ_bh[gq * HD + hd_base + c],             \
                                  scratch[i] * scale);                        \
                    }                                                         \
                }                                                             \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
                                                                              \
        /* ======== STAGE 7: dK += dS^T @ Q (atomicAdd to global) ===== */   \
        /* dK is [BKV, HD]. dS^T is [BKV, BQ]. Q is [BQ, HD]. */            \
        /* Load dS [BQ, BKV] as col_major for dS^T semantics. */             \
        {                                                                     \
            const int num_hd_tiles = HD / WMMA_N;                             \
                                                                              \
            for (int row_iter = 0; row_iter < BKV / (2 * WMMA_M); row_iter++) { \
                int ki_base = (warp_id / 4) * WMMA_M + row_iter * 2 * WMMA_M; \
                if (ki_base >= BKV) break;                                    \
                                                                              \
                int tiles_per_warp = (num_hd_tiles + 3) / 4;                  \
                for (int ht = 0; ht < tiles_per_warp; ht++) {                 \
                    int hd_tile = (warp_id % 4) + ht * 4;                     \
                    if (hd_tile >= num_hd_tiles) break;                       \
                    int hd_base = hd_tile * WMMA_N;                           \
                                                                              \
                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dk_acc; \
                    wmma::fill_fragment(dk_acc, 0.0f);                        \
                                                                              \
                    for (int qq = 0; qq < BQ; qq += WMMA_K) {                 \
                        /* dS^T: dS [BQ,BKV] col_major at [qq, ki_base] */    \
                        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, \
                                       __nv_bfloat16, wmma::col_major> dst_frag; \
                        wmma::load_matrix_sync(dst_frag,                      \
                            s_P + qq * BKV + ki_base, BKV);                   \
                                                                              \
                        /* Q: [BQ, HD] row-major at [qq, hd_base] */          \
                        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, \
                                       __nv_bfloat16, wmma::row_major> q_frag; \
                        wmma::load_matrix_sync(q_frag,                        \
                            s_Q + qq * HD + hd_base, HD);                     \
                                                                              \
                        wmma::mma_sync(dk_acc, dst_frag, q_frag, dk_acc);     \
                    }                                                         \
                                                                              \
                    /* Store to scratch, then atomicAdd to global dK */        \
                    float* scratch = s_S + warp_id * WMMA_M * WMMA_N;         \
                    wmma::store_matrix_sync(scratch, dk_acc, WMMA_N,          \
                                            wmma::mem_row_major);             \
                                                                              \
                    for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {     \
                        int r = i >> 4;                                       \
                        int c = i & 15;                                       \
                        if (ki_base + r < kv_rows) {                          \
                            atomicAdd(&dK_bh[                                 \
                                (ki_base + r) * HD + hd_base + c],            \
                                scratch[i]);                                  \
                        }                                                     \
                    }                                                         \
                }                                                             \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
    }  /* end Q tile loop */                                                  \
}

// ============================================================================
// Generate kernel specializations for each head dimension
// ============================================================================
DEFINE_FLASH_ATTN_BWD_KERNEL(64)
DEFINE_FLASH_ATTN_BWD_KERNEL(96)
DEFINE_FLASH_ATTN_BWD_KERNEL(128)

// ============================================================================
// Helper kernel: convert FP32 buffer to BF16
// Grid: ceil(total_elements / 256), Block: 256
// ============================================================================
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
// Entry point
//
// All inputs/outputs are BF16 except LSE (FP32) and internal staging buffers.
//
// Gradient accumulation strategy:
//   dQ, dK, dV are each accumulated in FP32 global buffers via atomicAdd
//   (multiple blocks contribute to the same gradient locations). After the
//   backward kernel, a trivial conversion kernel writes BF16 to the caller's
//   output buffers.
//
// Memory overhead: 3x FP32 buffers ([BH,N_q,HD] + 2*[BH,N_kv,HD]).
//   For BH=40, N=4096, HD=128: 3 * 40 * 4096 * 128 * 4 = ~240 MB.
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
    float* dQ_f32,    // Pre-allocated FP32 staging [BH, N_q, HD], must be zeroed
    float* dK_f32,    // Pre-allocated FP32 staging [BH, N_kv, HD], must be zeroed
    float* dV_f32     // Pre-allocated FP32 staging [BH, N_kv, HD], must be zeroed
) {
    cudaStream_t s = (cudaStream_t)stream;
    float scale_val = 1.0f / sqrtf((float)head_dim);

    // Launch backward kernel
    dim3 grid(batch_heads, (seq_len_kv + BKV - 1) / BKV);
    dim3 block(THREADS);

    // Shared memory: s_K + s_V + s_Q + s_dO + s_P (bf16) + s_S + s_LSE + s_D (f32)
    //   bf16: (BKV*HD + BKV*HD + BQ*HD + BQ*HD + BQ*BKV) * 2
    //   f32:  (BQ*BKV + BQ + BQ) * 4
    //
    // For HD=128: bf16 = (8192+8192+4096+4096+2048)*2 = 53248
    //             f32  = (2048+32+32)*4 = 8448
    //             total = 61696 bytes (~60 KB) — well within SM_86 100KB limit

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
        if (attr_err != cudaSuccess) {                                        \
            return (int)attr_err;                                             \
        }                                                                     \
        flash_attn_bwd_hd##HD<<<grid, block, smem_size, s>>>(                 \
            (const __nv_bfloat16*)Q,                                          \
            (const __nv_bfloat16*)K,                                          \
            (const __nv_bfloat16*)V,                                          \
            (const __nv_bfloat16*)O,                                          \
            (const __nv_bfloat16*)dO,                                         \
            (const float*)LSE,                                                \
            dQ_f32,                                                           \
            dK_f32,                                                           \
            dV_f32,                                                           \
            seq_len_q,                                                        \
            seq_len_kv,                                                       \
            scale_val                                                         \
        );                                                                    \
    } while (0)

    switch (head_dim) {
        case 64:  LAUNCH_FLASH_BWD(64);  break;
        case 96:  LAUNCH_FLASH_BWD(96);  break;
        case 128: LAUNCH_FLASH_BWD(128); break;
        default:
            return -1;
    }

    #undef LAUNCH_FLASH_BWD

    // Convert all three FP32 staging buffers to BF16 output
    {
        int conv_threads = 256;

        int dQ_total = batch_heads * seq_len_q * head_dim;
        convert_f32_to_bf16_bwd<<<(dQ_total + conv_threads - 1) / conv_threads,
                                  conv_threads, 0, s>>>(
            dQ_f32, (__nv_bfloat16*)dQ, dQ_total);

        int dKV_total = batch_heads * seq_len_kv * head_dim;
        convert_f32_to_bf16_bwd<<<(dKV_total + conv_threads - 1) / conv_threads,
                                  conv_threads, 0, s>>>(
            dK_f32, (__nv_bfloat16*)dK, dKV_total);

        convert_f32_to_bf16_bwd<<<(dKV_total + conv_threads - 1) / conv_threads,
                                  conv_threads, 0, s>>>(
            dV_f32, (__nv_bfloat16*)dV, dKV_total);
    }

    // Staging buffers are caller-managed — no free here.
    return (int)cudaGetLastError();
}

} // extern "C"
