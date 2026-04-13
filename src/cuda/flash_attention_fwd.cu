// flash_attention_fwd.cu
// Flash Attention forward pass for FLAME inference — WMMA tensor core version.
// BF16 in/out, FP32 accumulation via tensor cores, online softmax.
// No backward pass needed — inference only.
//
// Supports head_dim = 64, 96, 128 via compile-time specialization.
// SD3 uses 64, Mistral 96, FLUX/LTX/Klein/Z-Image 128.
//
// Drop-in replacement: same extern "C" entry point as the scalar version.
// Same function signature, same semantics.
//
// Architecture: SM_80+ (Ampere, Ada, Hopper). Uses nvcuda::wmma BF16 fragments.
// Compile with: -arch=sm_80 (or sm_86 for 3090, sm_89 for 4090)
//
// Algorithm: FlashAttention-2 online softmax with tiled GEMM via tensor cores.
//   For each query tile Qi (BQ rows):
//     Load Qi into shared memory (BF16)
//     Initialize O_acc = 0 (FP32), m = -inf, l = 0
//     For each KV tile (BKV rows):
//       Load Kj, Vj into shared memory (BF16)
//       S_ij = Qi @ Kj^T * scale   (via wmma, output FP32 in shared mem)
//       Online softmax update: m_new, l_new, correction
//       P_ij = exp(S_ij - m_new)   (FP32, then convert to BF16 for PV wmma)
//       O_acc = O_acc * correction + P_ij @ Vj  (via wmma, FP32 accumulator)
//     O_final = O_acc / l           (normalize, write BF16)

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math.h>
#include <float.h>

using namespace nvcuda;

extern "C" {

// Tile sizes — tuned for Ampere tensor cores
// Each wmma fragment is 16x16x16 for BF16
//
// NOTE: original values were BQ=64, BKV=64, NUM_WARPS=16 (4x4 warp grid).
// That needs ~104.5 KB dynamic shared memory for HD=128, which exceeds
// the SM_86 (RTX 3090/Ti) per-block opt-in limit of 100 KB and causes
// the cudaFuncSetAttribute / kernel launch to fail. Cutting BQ in half
// brings shared memory to ~68 KB and the warp grid to 2x4 = 8 warps.
// Throughput per block is lower but the kernel runs correctly on
// SM_86; SM_89+ (Ada) and SM_80 (A100) have larger shared-memory
// budgets and could use the BQ=64 version.
#define BQ 32        // Query tile rows (was 64, halved for SM_86 shared-mem limit)
#define BKV 64       // Key/Value tile rows
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define NUM_WARPS 8  // 2x4 warp grid for QK^T (was 16; matches BQ=32)
#define THREADS (NUM_WARPS * 32)  // 256 threads (was 512)

// ============================================================================
// Helper: cooperative BF16 tile load from global to shared memory.
//
// CRITICAL: pads the FULL buffer (buf_rows * cols), not just valid rows.
// The previous version iterated `valid_rows * cols` and the (r < valid_rows)
// guard was dead code, leaving s_K/s_V padding rows holding uninitialized
// shared memory. WMMA loads then read garbage and the kernel was
// nondeterministic across runs whenever the allocator placed prior tensors
// at different addresses (e.g. when an extra unused tensor like
// cap_feats_uncond was loaded).
//
// `cols` MUST be a power of two for the bit shifts (HD ∈ {64, 96, 128} is
// always true at the call sites — for HD=96 we'd lose this opt, but the
// only callers in this file pass HD which is 128 or 64).
// ============================================================================
__device__ __forceinline__ void load_tile_bf16(
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
// Kernel: flash_attn_wmma_hdXX
// Grid:   (batch_heads, ceil(N_q / BQ))
// Block:  THREADS (512)
// Shared: see smem_size calculation in launch macro
//
// Q, K, V: [BH, N, HD] BF16 contiguous
// O:       [BH, N, HD] BF16 output
// ============================================================================
#define DEFINE_FLASH_ATTN_WMMA_KERNEL(HD)                                     \
__global__ void flash_attn_fwd_hd##HD(                                        \
    const __nv_bfloat16* __restrict__ Q,                                      \
    const __nv_bfloat16* __restrict__ K,                                      \
    const __nv_bfloat16* __restrict__ V,                                      \
    __nv_bfloat16* __restrict__ O,                                            \
    float* __restrict__ LSE,                                                  \
    const int N_q,                                                            \
    const int N_kv,                                                           \
    const float scale                                                         \
) {                                                                           \
    const int bh = blockIdx.x;                                                \
    const int q_tile_idx = blockIdx.y;                                        \
    const int tid = threadIdx.x;                                              \
    const int warp_id = tid / 32;                                             \
    const int lane_id = tid % 32;                                             \
                                                                              \
    const int q_start = q_tile_idx * BQ;                                      \
    if (q_start >= N_q) return;                                               \
    const int q_rows = min(BQ, N_q - q_start);                                \
                                                                              \
    /* Pointers into global memory for this batch-head */                     \
    const __nv_bfloat16* Q_base = Q + (size_t)bh * N_q  * HD + (size_t)q_start * HD; \
    const __nv_bfloat16* K_base = K + (size_t)bh * N_kv * HD;                \
    const __nv_bfloat16* V_base = V + (size_t)bh * N_kv * HD;                \
    __nv_bfloat16* O_base       = O + (size_t)bh * N_q  * HD + (size_t)q_start * HD; \
                                                                              \
    /* ---- Shared memory layout ---- */                                      \
    /* All BF16 tiles stored as __nv_bfloat16 (2 bytes each) */               \
    extern __shared__ char smem_raw[];                                        \
    __nv_bfloat16* s_Q = (__nv_bfloat16*)smem_raw;       /* [BQ, HD]    */    \
    __nv_bfloat16* s_K = s_Q + BQ * HD;                  /* [BKV, HD]   */    \
    __nv_bfloat16* s_V = s_K + BKV * HD;                 /* [BKV, HD]   */    \
    float* s_S = (float*)(s_V + BKV * HD);               /* [BQ, BKV]   */    \
    __nv_bfloat16* s_P = (__nv_bfloat16*)(s_S + BQ*BKV); /* [BQ, BKV]   */    \
    float* s_O = (float*)(s_P + BQ * BKV);               /* [BQ, HD]    */    \
    float* s_m = s_O + BQ * HD;                          /* [BQ]        */    \
    float* s_l = s_m + BQ;                               /* [BQ]        */    \
                                                                              \
    /* ---- Load Q tile (persists for entire KV loop) ---- */                 \
    /* Pad to BQ rows (not just q_rows) so wmma reads on the partial last  */ \
    /* query tile see zeros instead of stale shared memory.                */ \
    load_tile_bf16(s_Q, Q_base, q_rows, HD, BQ, HD, THREADS, tid);           \
                                                                              \
    /* ---- Initialize output accumulator and softmax state ---- */           \
    for (int i = tid; i < BQ * HD; i += THREADS) {                            \
        s_O[i] = 0.0f;                                                       \
    }                                                                         \
    for (int i = tid; i < BQ; i += THREADS) {                                 \
        s_m[i] = -FLT_MAX;                                                   \
        s_l[i] = 0.0f;                                                       \
    }                                                                         \
    __syncthreads();                                                          \
                                                                              \
    /* ---- Main KV tile loop ---- */                                         \
    const int num_kv_tiles = (N_kv + BKV - 1) / BKV;                         \
                                                                              \
    /* Warp tile assignment for QK^T: 4x4 grid of 16x16 tiles */             \
    const int warp_qi = (warp_id / 4) * WMMA_M; /* row offset in BQ */       \
    const int warp_kj = (warp_id % 4) * WMMA_N; /* col offset in BKV */      \
                                                                              \
    for (int kv_t = 0; kv_t < num_kv_tiles; kv_t++) {                        \
        const int kv_start = kv_t * BKV;                                     \
        const int kv_rows = min(BKV, N_kv - kv_start);                       \
                                                                              \
        /* Load K and V tiles. Pad to BKV rows (not just kv_rows) so       */ \
        /* wmma reads on the last KV tile see zeros instead of stale       */ \
        /* shared memory. This was the source of the run-to-run            */ \
        /* nondeterminism that depended on prior tensor allocations.       */ \
        load_tile_bf16(s_K, K_base + (size_t)kv_start * HD,                   \
                       kv_rows, HD, BKV, HD, THREADS, tid);                   \
        load_tile_bf16(s_V, V_base + (size_t)kv_start * HD,                   \
                       kv_rows, HD, BKV, HD, THREADS, tid);                   \
        __syncthreads();                                                      \
                                                                              \
        /* ======== STAGE 1: S = Q @ K^T via wmma ======== */                 \
        /* Each warp computes one 16x16 tile of S[BQ, BKV] */                 \
        /* S[warp_qi..+16, warp_kj..+16] = Q[warp_qi..+16, :] @ K[warp_kj..+16, :]^T */ \
        {                                                                     \
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc; \
            wmma::fill_fragment(acc, 0.0f);                                   \
                                                                              \
            /* Accumulate over HD dimension in chunks of WMMA_K=16 */         \
            for (int hd = 0; hd < HD; hd += WMMA_K) {                        \
                /* Q fragment: rows [warp_qi, warp_qi+16), cols [hd, hd+16) */\
                /* Q is [BQ, HD] row-major, stride = HD */                    \
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,       \
                               __nv_bfloat16, wmma::row_major> q_frag;        \
                wmma::load_matrix_sync(q_frag,                                \
                    s_Q + warp_qi * HD + hd, HD);                             \
                                                                              \
                /* K fragment: K is [BKV, HD] row-major.                   */ \
                /* We want K^T, so load K as col_major: reading K[BKV,HD]  */ \
                /* as col_major gives us K^T[HD, BKV] semantics.           */ \
                /* Fragment B is K×N = 16×16, load from K[warp_kj, hd]     */ \
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,       \
                               __nv_bfloat16, wmma::col_major> k_frag;        \
                wmma::load_matrix_sync(k_frag,                                \
                    s_K + warp_kj * HD + hd, HD);                             \
                                                                              \
                wmma::mma_sync(acc, q_frag, k_frag, acc);                     \
            }                                                                 \
                                                                              \
            /* Apply scale and store to s_S */                                \
            for (int i = 0; i < acc.num_elements; i++) {                      \
                acc.x[i] *= scale;                                            \
            }                                                                 \
            wmma::store_matrix_sync(                                          \
                s_S + warp_qi * BKV + warp_kj,                                \
                acc, BKV, wmma::mem_row_major);                               \
        }                                                                     \
        __syncthreads();                                                      \
                                                                              \
        /* ======== STAGE 2: Online softmax (warp-cooperative, fused) ====== \
         * One warp processes one row at a time; lanes cooperate via shfl   \
         * reductions over BKV. This version writes the BF16 P matrix       \
         * directly to s_P (skipping the FP32 s_S scratch + the separate    \
         * stage 2b conversion pass), saving one full BQ*BKV pass + one    \
         * __syncthreads per KV iteration.                                  \
         */                                                                   \
        {                                                                     \
            for (int qi = warp_id; qi < q_rows; qi += NUM_WARPS) {            \
                /* --- Reduce max over BKV elements with 32 lanes --- */      \
                float my_max = -FLT_MAX;                                      \
                for (int j = lane_id; j < kv_rows; j += 32) {                 \
                    my_max = fmaxf(my_max, s_S[qi * BKV + j]);               \
                }                                                             \
                _Pragma("unroll")                                             \
                for (int off = 16; off > 0; off >>= 1) {                      \
                    my_max = fmaxf(my_max, __shfl_xor_sync(0xffffffff, my_max, off)); \
                }                                                             \
                float old_max = s_m[qi];                                      \
                float new_max = fmaxf(old_max, my_max);                       \
                float correction = __expf(old_max - new_max);                 \
                                                                              \
                /* --- Compute exp(s - new_max), write directly to s_P,    */ \
                /*     accumulate sum.                                     */ \
                /*     Padding lanes (j >= kv_rows) get exp(0) and would   */ \
                /*     contribute 1.0 each — write zero to s_P[j] for them */ \
                /*     so the subsequent P@V wmma sees zeros, not stale.   */ \
                float my_sum = 0.0f;                                          \
                for (int j = lane_id; j < BKV; j += 32) {                     \
                    if (j < kv_rows) {                                        \
                        float p = __expf(s_S[qi * BKV + j] - new_max);       \
                        s_P[qi * BKV + j] = __float2bfloat16(p);             \
                        my_sum += p;                                          \
                    } else {                                                  \
                        s_P[qi * BKV + j] = __float2bfloat16(0.0f);          \
                    }                                                         \
                }                                                             \
                _Pragma("unroll")                                             \
                for (int off = 16; off > 0; off >>= 1) {                      \
                    my_sum += __shfl_xor_sync(0xffffffff, my_sum, off);       \
                }                                                             \
                                                                              \
                /* --- Lane 0 commits softmax state for this row --- */       \
                if (lane_id == 0) {                                           \
                    s_l[qi] = s_l[qi] * correction + my_sum;                  \
                    s_m[qi] = new_max;                                        \
                }                                                             \
                                                                              \
                /* --- Parallel-stride rescale of s_O[qi, 0..HD] --- */       \
                for (int d = lane_id; d < HD; d += 32) {                      \
                    s_O[qi * HD + d] *= correction;                           \
                }                                                             \
            }                                                                 \
            /* Pad rows qi >= q_rows so wmma sees zeros (not garbage) when */ \
            /* the last query tile is partial. Each warp zero-pads the     */ \
            /* rows it would have processed if q_rows == BQ.               */ \
            for (int qi = q_rows + warp_id; qi < BQ; qi += NUM_WARPS) {       \
                for (int j = lane_id; j < BKV; j += 32) {                     \
                    s_P[qi * BKV + j] = __float2bfloat16(0.0f);              \
                }                                                             \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
                                                                              \
        /* ======== STAGE 3: O += P @ V via wmma ======== */                  \
        /* O is [BQ, HD]. P is [BQ, BKV]. V is [BKV, HD]. */                 \
        /* With 16 warps and HD/16 = HD>>4 column tiles: */                   \
        /* Each warp handles multiple 16x16 output tiles. */                  \
        {                                                                     \
            const int qi_base = (warp_id / 4) * WMMA_M;                      \
            const int num_hd_tiles = HD / WMMA_N;                             \
            const int tiles_per_warp = (num_hd_tiles + 3) / 4;               \
                                                                              \
            for (int ht = 0; ht < tiles_per_warp; ht++) {                     \
                int hd_tile = (warp_id % 4) + ht * 4;                        \
                if (hd_tile >= num_hd_tiles) break;                           \
                int hd_base = hd_tile * WMMA_N;                               \
                                                                              \
                /* Accumulate P[qi..+16, :] @ V[:, hd..+16] */               \
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> pv_acc; \
                wmma::fill_fragment(pv_acc, 0.0f);                            \
                                                                              \
                for (int kv = 0; kv < BKV; kv += WMMA_K) {                    \
                    /* P fragment: [BQ, BKV] row-major, tile at [qi_base, kv] */ \
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,    \
                                   __nv_bfloat16, wmma::row_major> p_frag;     \
                    wmma::load_matrix_sync(p_frag,                             \
                        s_P + qi_base * BKV + kv, BKV);                        \
                                                                              \
                    /* V fragment: [BKV, HD] row-major, tile at [kv, hd_base] */ \
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,    \
                                   __nv_bfloat16, wmma::row_major> v_frag;     \
                    wmma::load_matrix_sync(v_frag,                             \
                        s_V + kv * HD + hd_base, HD);                          \
                                                                              \
                    wmma::mma_sync(pv_acc, p_frag, v_frag, pv_acc);            \
                }                                                             \
                                                                              \
                /* Add to running O accumulator in shared memory */            \
                /* Store wmma result to temp, then add to s_O */               \
                float temp_o[WMMA_M * WMMA_N];                                \
                /* Store to registers first via a small staging area */        \
                /* Use the warp's portion of s_S as scratch (safe — */         \
                /* softmax is done with s_S at this point) */                  \
                float* scratch = s_S + warp_id * WMMA_M * WMMA_N;             \
                wmma::store_matrix_sync(scratch, pv_acc, WMMA_N,              \
                                        wmma::mem_row_major);                 \
                                                                              \
                /* Cooperative add: each (warp, ht) writes a unique 16x16   \
                 * tile of s_O — no cross-warp collision possible (verified  \
                 * from warp tile map). The only RMW is the per-iteration    \
                 * accumulation across KV tiles, and each thread serializes  \
                 * its own location, so plain += is safe. No atomicAdd.      \
                 */                                                           \
                for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {         \
                    int r = i >> 4;          /* i / 16 */                     \
                    int c = i & 15;          /* i % 16 */                     \
                    int idx = (qi_base + r) * HD + hd_base + c;               \
                    s_O[idx] = s_O[idx] + scratch[i];                         \
                }                                                             \
            }                                                                 \
        }                                                                     \
        __syncthreads();                                                      \
    }  /* end KV tile loop */                                                 \
                                                                              \
    /* ---- Write LSE (log-sum-exp) per row for backward pass ---- */          \
    /* LSE[row] = m[row] + log(l[row])                            */          \
    /* Needed by the backward kernel to recompute softmax         */          \
    /* without storing the full attention matrix.                  */          \
    if (LSE != NULL) {                                                        \
        float* LSE_base = LSE + (size_t)bh * N_q + q_start;                  \
        for (int i = tid; i < q_rows; i += THREADS) {                        \
            LSE_base[i] = s_m[i] + logf(s_l[i]);                             \
        }                                                                     \
    }                                                                         \
                                                                              \
    /* ---- Normalize O by 1/l and write BF16 output ---- */                  \
    for (int i = tid; i < q_rows * HD; i += THREADS) {                        \
        int qi = i / HD;                                                      \
        float val = s_O[i] / s_l[qi];                                         \
        O_base[i] = __float2bfloat16(val);                                    \
    }                                                                         \
}

// ============================================================================
// Generate kernel specializations for each head dimension
// ============================================================================
DEFINE_FLASH_ATTN_WMMA_KERNEL(64)
DEFINE_FLASH_ATTN_WMMA_KERNEL(96)
DEFINE_FLASH_ATTN_WMMA_KERNEL(128)

// ============================================================================
// Entry point — same signature as the scalar version, drop-in replacement
// ============================================================================
int flame_flash_attention_bf16(
    const void* Q,
    const void* K,
    const void* V,
    void* O,
    float* LSE,
    int batch_heads,
    int seq_len_q,
    int seq_len_kv,
    int head_dim,
    void* stream
) {
    dim3 grid(batch_heads, (seq_len_q + BQ - 1) / BQ);
    dim3 block(THREADS);
    cudaStream_t s = (cudaStream_t)stream;
    float scale_val = 1.0f / sqrtf((float)head_dim);

    // Shared memory layout (BF16 tiles + FP32 accumulators):
    //   s_Q:  [BQ, HD]  bf16   = BQ * HD * 2
    //   s_K:  [BKV, HD] bf16   = BKV * HD * 2
    //   s_V:  [BKV, HD] bf16   = BKV * HD * 2
    //   s_S:  [BQ, BKV] f32    = BQ * BKV * 4
    //   s_P:  [BQ, BKV] bf16   = BQ * BKV * 2
    //   s_O:  [BQ, HD]  f32    = BQ * HD * 4
    //   s_m:  [BQ]      f32    = BQ * 4
    //   s_l:  [BQ]      f32    = BQ * 4

    #define LAUNCH_FLASH_WMMA(HD) do {                                        \
        size_t smem_bf16 = (BQ*(HD) + BKV*(HD) + BKV*(HD) + BQ*BKV)          \
                           * sizeof(__nv_bfloat16);                           \
        size_t smem_f32  = (BQ*BKV + BQ*(HD) + BQ + BQ)                      \
                           * sizeof(float);                                   \
        size_t smem_size = smem_bf16 + smem_f32;                              \
        cudaError_t attr_err = cudaFuncSetAttribute(                          \
            flash_attn_fwd_hd##HD,                                            \
            cudaFuncAttributeMaxDynamicSharedMemorySize,                      \
            (int)smem_size                                                    \
        );                                                                    \
        if (attr_err != cudaSuccess) return (int)attr_err;                    \
        flash_attn_fwd_hd##HD<<<grid, block, smem_size, s>>>(                 \
            (const __nv_bfloat16*)Q,                                          \
            (const __nv_bfloat16*)K,                                          \
            (const __nv_bfloat16*)V,                                          \
            (__nv_bfloat16*)O,                                                \
            LSE,                                                              \
            seq_len_q,                                                        \
            seq_len_kv,                                                       \
            scale_val                                                         \
        );                                                                    \
    } while (0)

    switch (head_dim) {
        case 64:  LAUNCH_FLASH_WMMA(64);  break;
        case 96:  LAUNCH_FLASH_WMMA(96);  break;
        case 128: LAUNCH_FLASH_WMMA(128); break;
        default: return -1;
    }

    #undef LAUNCH_FLASH_WMMA
    return (int)cudaGetLastError();
}

} // extern "C"
