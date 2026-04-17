// flash_attention_fwd.cu
// FA2-style tiled Flash Attention forward for FLAME inference — WMMA/BF16.
//
// Phase 1.6: cp.async pipelining with single-buffer KV slot + V-load/softmax
// overlap. Full KV double-buffering was investigated (see FA2_CP_ASYNC_DESIGN.md)
// and the SMEM reclaim needed to fit it (fold s_P into s_S) produced a subtle
// correctness regression that wasn't resolved in this phase. Ship what's green:
// cp.async V prefetch during softmax compute, hiding V's HBM read behind the
// online-softmax pass. This is a correctness-preserving strict subset of the
// full double-buffered plan, and still wins back meaningful latency on long
// seqs where softmax compute is non-trivial relative to a V-tile memcpy.
//
// Algorithm (online softmax with running max m, denominator l, correction
// alpha = exp(m_old - m_new)) is unchanged from Phase 1. Parity validated
// against an FP32 materialized reference in tests/fa2_parity_naive.rs.
//
// Supports head_dim = 64, 96, 128 via compile-time specialization. Architecture
// SM_80+. FFI symbol `flame_flash_attention_bf16` is unchanged.
//
// ---------------------------------------------------------------------------
// Shared-memory layout (BQ=64, BKV=64, HD=128 worst case)  — unchanged from
// Phase 1:
//
//   s_Q   [BQ, HD]   BF16   16 KB   (persists across KV iterations)
//   s_KV  [BKV, HD]  BF16   16 KB   (reused: holds K_j, then V_j per iter)
//   s_S   [BQ, BKV]  FP32   16 KB   (scores this iter; reset on QK^T store)
//   s_P   [BQ, BKV]  BF16    8 KB   (probs this iter; input to PV wmma)
//   s_O   [BQ, HD]   FP32   32 KB   (running O accumulator)
//   m, l  [BQ]       FP32  0.5 KB   (per-row running max + denom)
//   ---------------------------------------------------------------------
//   total                   88.5 KB  <= SM_86 opt-in 100 KB ✓
//
// What changed in 1.6: K_j and V_j loads use cp.async.cg instead of scalar
// uint4 stores. V_j's cp.async is ISSUED immediately after QK^T finishes
// reading K_j, before softmax begins. The softmax stage's compute then
// overlaps with V_j's HBM read, hiding that latency. cp.async.wait_group(0)
// + __syncthreads() gates PV on V_j visibility.
//
// ---------------------------------------------------------------------------
// Warp layout — NUM_WARPS=4:
//   BQ=64 / WMMA_M=16 = 4 row-tiles. Each warp owns 16 Q rows exclusively.
//   Per-row softmax is entirely within one warp (no cross-warp reduction).
//
// Per-warp fragment counts:
//   QK^T : FRAG_M=1, FRAG_N_S = BKV / WMMA_N = 4                  frags
//   PV   : FRAG_M=1, FRAG_N_O = HD / WMMA_N = {4, 6, 8} for {64,96,128}

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math.h>
#include <float.h>

using namespace nvcuda;

// ---------------------------------------------------------------------------
// cp.async PTX intrinsics (SM_80+).
// cp.async.cg: 16-byte aligned cached global → shared. One instruction moves
// 16 B per thread. Commit groups + wait_group provide pipelined prefetch.
//
// Outside `extern "C"` because templated wait_group cannot have C linkage.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned smem_int = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :
                 : "r"(smem_int), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most `N` groups are still in flight. PTX `wait_group`
// forces OLDEST groups to complete until ≤ N remain.
template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

extern "C" {

// ---------------------------------------------------------------------------
// Tile / warp / fragment geometry
// ---------------------------------------------------------------------------
#define BQ 64
#define BKV 64
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define NUM_WARPS 4
#define THREADS (NUM_WARPS * 32)   // 128

// ---------------------------------------------------------------------------
// Synchronous BF16 tile load (uint4-vectorized). Used for Q which persists
// across the whole KV loop.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void load_tile_bf16_v8(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    int valid_rows, int cols, int buf_rows, int global_stride,
    int num_threads, int tid
) {
    const int vec_per_row = cols >> 3;            // cols / 8
    const int vec_total   = buf_rows * vec_per_row;
    const uint4* src_v = reinterpret_cast<const uint4*>(src);
    uint4* dst_v = reinterpret_cast<uint4*>(dst);
    const int src_stride_v = global_stride >> 3;
    const uint4 zero_v = make_uint4(0u, 0u, 0u, 0u);

    for (int i = tid; i < vec_total; i += num_threads) {
        int r = i / vec_per_row;
        int c = i - r * vec_per_row;
        if (r < valid_rows) {
            dst_v[i] = src_v[r * src_stride_v + c];
        } else {
            dst_v[i] = zero_v;
        }
    }
}

// ---------------------------------------------------------------------------
// Issue cp.async loads for a [buf_rows, cols] BF16 tile. Each thread emits
// one cp.async.cg per 16 B stripe. Rows past `valid_rows` go through a
// regular SMEM zero store (cp.async has no masked form); these are the
// last-tile tail only. Caller pairs this with cp_async_commit() and an
// eventual cp_async_wait_group<N>() + __syncthreads().
// ---------------------------------------------------------------------------
__device__ __forceinline__ void issue_cp_async_tile_bf16(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    int valid_rows, int cols, int buf_rows, int global_stride,
    int num_threads, int tid
) {
    const int vec_per_row = cols >> 3;
    const int vec_total   = buf_rows * vec_per_row;
    const int src_stride_v = global_stride >> 3;

    const uint4* src_v = reinterpret_cast<const uint4*>(src);
    uint4* dst_v = reinterpret_cast<uint4*>(dst);
    const uint4 zero_v = make_uint4(0u, 0u, 0u, 0u);

    for (int i = tid; i < vec_total; i += num_threads) {
        const int r = i / vec_per_row;
        const int c = i - r * vec_per_row;
        if (r < valid_rows) {
            cp_async_cg_16(&dst_v[i], &src_v[r * src_stride_v + c]);
        } else {
            dst_v[i] = zero_v;   // not part of the cp.async group
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel — one specialization per head dim via macro expansion.
// Grid:  (batch_heads, ceil(N_q / BQ))
// Block: THREADS (128)
// ---------------------------------------------------------------------------
#define DEFINE_FA2_KERNEL(HD)                                                \
__global__ void fa2_fwd_hd##HD(                                              \
    const __nv_bfloat16* __restrict__ Q,                                     \
    const __nv_bfloat16* __restrict__ K,                                     \
    const __nv_bfloat16* __restrict__ V,                                     \
    __nv_bfloat16* __restrict__ O,                                           \
    float* __restrict__ LSE,                                                 \
    const int N_q,                                                           \
    const int N_kv,                                                          \
    const float scale                                                        \
) {                                                                          \
    const int bh         = blockIdx.x;                                       \
    const int q_tile_idx = blockIdx.y;                                       \
    const int tid        = threadIdx.x;                                      \
    const int warp_id    = tid >> 5;                                         \
    const int lane_id    = tid & 31;                                         \
                                                                             \
    const int q_start = q_tile_idx * BQ;                                     \
    if (q_start >= N_q) return;                                              \
    const int q_rows  = min(BQ, N_q - q_start);                              \
                                                                             \
    const __nv_bfloat16* Q_base = Q + (size_t)bh * N_q  * HD                 \
                                    + (size_t)q_start * HD;                  \
    const __nv_bfloat16* K_base = K + (size_t)bh * N_kv * HD;                \
    const __nv_bfloat16* V_base = V + (size_t)bh * N_kv * HD;                \
    __nv_bfloat16* O_base       = O + (size_t)bh * N_q  * HD                 \
                                    + (size_t)q_start * HD;                  \
                                                                             \
    /* ---- Shared-memory layout (Phase 1 layout, unchanged) ---- */         \
    extern __shared__ char smem_raw[];                                       \
    __nv_bfloat16* s_Q  = reinterpret_cast<__nv_bfloat16*>(smem_raw);        \
    __nv_bfloat16* s_KV = s_Q + BQ * HD;                                     \
    float* s_S          = reinterpret_cast<float*>(s_KV + BKV * HD);         \
    __nv_bfloat16* s_P  = reinterpret_cast<__nv_bfloat16*>(s_S + BQ * BKV);  \
    float* s_O          = reinterpret_cast<float*>(s_P + BQ * BKV);          \
    float* s_m          = s_O + BQ * HD;                                     \
    float* s_l          = s_m + BQ;                                          \
                                                                             \
    /* Each warp exclusively owns 16 Q rows: [warp_id*16, warp_id*16+16). */ \
    const int warp_row = warp_id * WMMA_M;                                   \
                                                                             \
    /* ---- Load Q tile (persists for the whole KV loop) ---- */             \
    load_tile_bf16_v8(s_Q, Q_base, q_rows, HD, BQ, HD, THREADS, tid);        \
                                                                             \
    /* ---- Init s_O = 0, m = -FLT_MAX, l = 0 ---- */                        \
    for (int i = tid; i < BQ * HD; i += THREADS) {                           \
        s_O[i] = 0.0f;                                                       \
    }                                                                        \
    for (int i = tid; i < BQ; i += THREADS) {                                \
        s_m[i] = -FLT_MAX;                                                   \
        s_l[i] = 0.0f;                                                       \
    }                                                                        \
    __syncthreads();                                                         \
                                                                             \
    const int num_kv_tiles    = (N_kv + BKV - 1) / BKV;                      \
    const int num_s_col_tiles = BKV / WMMA_N;         /* 4               */  \
    const int num_hd_tiles    = HD  / WMMA_N;         /* 4 / 6 / 8        */ \
    const int num_k_tiles     = BKV / WMMA_K;         /* 4                */ \
                                                                             \
    /* ---- Main KV loop ---- */                                             \
    for (int kv_t = 0; kv_t < num_kv_tiles; kv_t++) {                        \
        const int kv_start = kv_t * BKV;                                     \
        const int kv_rows  = min(BKV, N_kv - kv_start);                      \
                                                                             \
        /* ======== Load K_j into s_KV via cp.async ======== */              \
        issue_cp_async_tile_bf16(s_KV, K_base + (size_t)kv_start * HD,       \
                                 kv_rows, HD, BKV, HD, THREADS, tid);        \
        cp_async_commit();                                                   \
        cp_async_wait_group<0>();                                            \
        __syncthreads();                                                     \
                                                                             \
        /* ======== STAGE 1: S = (Q @ K^T) * scale → s_S ======== */         \
        {                                                                    \
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> \
                s_frag[BKV / WMMA_N];                                        \
            _Pragma("unroll")                                                \
            for (int j = 0; j < num_s_col_tiles; j++) {                      \
                wmma::fill_fragment(s_frag[j], 0.0f);                        \
            }                                                                \
                                                                             \
            _Pragma("unroll 1")                                              \
            for (int hd = 0; hd < HD; hd += WMMA_K) {                        \
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,       \
                               __nv_bfloat16, wmma::row_major> q_frag;       \
                wmma::load_matrix_sync(q_frag,                               \
                    s_Q + warp_row * HD + hd, HD);                           \
                                                                             \
                _Pragma("unroll")                                            \
                for (int j = 0; j < num_s_col_tiles; j++) {                  \
                    /* K[BKV,HD] row-major, col_major view → K^T[HD,BKV].*/  \
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,   \
                                   __nv_bfloat16, wmma::col_major> k_frag;   \
                    wmma::load_matrix_sync(k_frag,                           \
                        s_KV + j * WMMA_N * HD + hd, HD);                    \
                    wmma::mma_sync(s_frag[j], q_frag, k_frag, s_frag[j]);    \
                }                                                            \
            }                                                                \
                                                                             \
            _Pragma("unroll")                                                \
            for (int j = 0; j < num_s_col_tiles; j++) {                      \
                _Pragma("unroll")                                            \
                for (int k = 0; k < (int)s_frag[j].num_elements; k++) {      \
                    s_frag[j].x[k] *= scale;                                 \
                }                                                            \
                wmma::store_matrix_sync(                                     \
                    s_S + warp_row * BKV + j * WMMA_N,                       \
                    s_frag[j], BKV, wmma::mem_row_major);                    \
            }                                                                \
        }                                                                    \
        __syncthreads();                                                     \
                                                                             \
        /* ---- Issue V_j prefetch overlapped with softmax compute ---- */   \
        /* K_j is dead after QK^T, so V_j can overwrite s_KV. The cp.async */\
        /* runs in parallel with the softmax stage below, hiding V's HBM   */\
        /* read latency behind the softmax pass.                            */\
        issue_cp_async_tile_bf16(s_KV, V_base + (size_t)kv_start * HD,       \
                                 kv_rows, HD, BKV, HD, THREADS, tid);        \
        cp_async_commit();                                                   \
                                                                             \
        /* ======== STAGE 2: Online softmax + write P ======== */            \
        /* Each warp handles its 16 Q rows. For each row the warp's 32   */  \
        /* lanes cooperate via shuffle-butterfly reductions over 64 cols. */ \
        {                                                                    \
            _Pragma("unroll 1")                                              \
            for (int ri = 0; ri < WMMA_M; ri++) {                            \
                const int qi = warp_row + ri;                                \
                const bool row_active = (qi < q_rows);                       \
                                                                             \
                /* --- Max reduction over valid cols --- */                  \
                float lane_max = -FLT_MAX;                                   \
                if (row_active) {                                            \
                    for (int j = lane_id; j < kv_rows; j += 32) {            \
                        lane_max = fmaxf(lane_max, s_S[qi * BKV + j]);       \
                    }                                                        \
                }                                                            \
                _Pragma("unroll")                                            \
                for (int off = 16; off > 0; off >>= 1) {                     \
                    lane_max = fmaxf(lane_max,                               \
                                     __shfl_xor_sync(0xffffffff, lane_max, off));\
                }                                                            \
                                                                             \
                const float old_max = row_active ? s_m[qi] : -FLT_MAX;       \
                const float new_max = fmaxf(old_max, lane_max);              \
                /* exp(-FLT_MAX - x) underflows to 0 cleanly on sm_80+. */   \
                const float alpha   = (old_max > -FLT_MAX / 2.0f)            \
                                      ? __expf(old_max - new_max)            \
                                      : 0.0f;                                \
                                                                             \
                /* --- Write P = exp(s - new_max) to s_P, accumulate sum. */ \
                float lane_sum = 0.0f;                                       \
                if (row_active) {                                            \
                    for (int j = lane_id; j < BKV; j += 32) {                \
                        float p;                                             \
                        if (j < kv_rows) {                                   \
                            p = __expf(s_S[qi * BKV + j] - new_max);         \
                            lane_sum += p;                                   \
                        } else {                                             \
                            p = 0.0f;                                        \
                        }                                                    \
                        s_P[qi * BKV + j] = __float2bfloat16(p);             \
                    }                                                        \
                } else {                                                     \
                    /* Zero-pad P for inactive rows so PV wmma sees 0s. */   \
                    for (int j = lane_id; j < BKV; j += 32) {                \
                        s_P[qi * BKV + j] = __float2bfloat16(0.0f);          \
                    }                                                        \
                }                                                            \
                _Pragma("unroll")                                            \
                for (int off = 16; off > 0; off >>= 1) {                     \
                    lane_sum += __shfl_xor_sync(0xffffffff, lane_sum, off);  \
                }                                                            \
                                                                             \
                if (row_active) {                                            \
                    if (lane_id == 0) {                                      \
                        s_l[qi] = s_l[qi] * alpha + lane_sum;                \
                        s_m[qi] = new_max;                                   \
                    }                                                        \
                    /* Rescale s_O[qi, :] by alpha (parallel over HD). */    \
                    for (int d = lane_id; d < HD; d += 32) {                 \
                        s_O[qi * HD + d] *= alpha;                           \
                    }                                                        \
                }                                                            \
            }                                                                \
        }                                                                    \
                                                                             \
        /* Wait for V_j to complete (was issued before softmax above). */    \
        cp_async_wait_group<0>();                                            \
        __syncthreads();                                                     \
                                                                             \
        /* ======== STAGE 3: O += P @ V (via wmma) ======== */               \
        /* Each warp emits num_hd_tiles accumulator fragments over its 16 */ \
        /* Q rows; accumulators start at 0 and reduce over BKV, then add   */\
        /* into s_O through a per-warp FP32 scratch placed in s_S (s_S is  */\
        /* dead for this iteration after softmax consumed it).             */ \
        {                                                                    \
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> \
                o_frag[HD / WMMA_N];                                         \
            _Pragma("unroll")                                                \
            for (int j = 0; j < num_hd_tiles; j++) {                         \
                wmma::fill_fragment(o_frag[j], 0.0f);                        \
            }                                                                \
                                                                             \
            _Pragma("unroll 1")                                              \
            for (int kv = 0; kv < num_k_tiles; kv++) {                       \
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,       \
                               __nv_bfloat16, wmma::row_major> p_frag;       \
                wmma::load_matrix_sync(p_frag,                               \
                    s_P + warp_row * BKV + kv * WMMA_K, BKV);                \
                                                                             \
                _Pragma("unroll")                                            \
                for (int j = 0; j < num_hd_tiles; j++) {                     \
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,   \
                                   __nv_bfloat16, wmma::row_major> v_frag;   \
                    wmma::load_matrix_sync(v_frag,                           \
                        s_KV + kv * WMMA_K * HD + j * WMMA_N, HD);           \
                    wmma::mma_sync(o_frag[j], p_frag, v_frag, o_frag[j]);    \
                }                                                            \
            }                                                                \
                                                                             \
            /* Scratch for the fragment store: 4 warps × 16×16 FP32 = 4 KB, \
             * placed in s_S (dead for this iteration). s_S holds BQ*BKV =  \
             * 4096 floats = 16 KB, so 4 KB for scratch fits with room.     \
             */                                                              \
            float* scratch = s_S + warp_id * (WMMA_M * WMMA_N);              \
            _Pragma("unroll")                                                \
            for (int j = 0; j < num_hd_tiles; j++) {                         \
                wmma::store_matrix_sync(scratch, o_frag[j], WMMA_N,          \
                                        wmma::mem_row_major);                \
                /* 256 elements / 32 lanes = 8 elements per lane. Each    */ \
                /* (warp, j) writes a unique 16×16 slab of s_O — no       */ \
                /* cross-warp or cross-j collisions.                      */ \
                _Pragma("unroll")                                            \
                for (int e = 0; e < (WMMA_M * WMMA_N) / 32; e++) {           \
                    const int lin = lane_id + e * 32;                        \
                    const int r = lin >> 4;                                  \
                    const int c = lin & 15;                                  \
                    const int dst = (warp_row + r) * HD + j * WMMA_N + c;    \
                    s_O[dst] += scratch[lin];                                \
                }                                                            \
                __syncwarp();                                                \
            }                                                                \
        }                                                                    \
        __syncthreads();                                                     \
    }  /* end KV loop */                                                     \
                                                                             \
    /* ---- Optional LSE output (LSE[row] = m[row] + log l[row]) ---- */     \
    if (LSE != nullptr) {                                                    \
        float* LSE_base = LSE + (size_t)bh * N_q + q_start;                  \
        for (int i = tid; i < q_rows; i += THREADS) {                        \
            LSE_base[i] = s_m[i] + logf(s_l[i]);                             \
        }                                                                    \
    }                                                                        \
                                                                             \
    /* ---- Final normalize + cast to BF16 output ---- */                    \
    for (int i = tid; i < q_rows * HD; i += THREADS) {                       \
        const int qi = i / HD;                                               \
        const float inv_l = 1.0f / s_l[qi];                                  \
        O_base[i] = __float2bfloat16(s_O[i] * inv_l);                        \
    }                                                                        \
}

DEFINE_FA2_KERNEL(64)
DEFINE_FA2_KERNEL(96)
DEFINE_FA2_KERNEL(128)

// ---------------------------------------------------------------------------
// Host entry point — signature unchanged from the legacy kernel.
// Returns 0 on success, a CUDA error code or -1 on unsupported head_dim.
// ---------------------------------------------------------------------------
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
    const float scale_val = 1.0f / sqrtf((float)head_dim);

    // Shared memory size per block (Phase 1 layout, unchanged):
    //   s_Q  = BQ  * HD  * sizeof(bf16)
    //   s_KV = BKV * HD  * sizeof(bf16)          [K then V, reused]
    //   s_S  = BQ  * BKV * sizeof(float)         [scores this iter]
    //   s_P  = BQ  * BKV * sizeof(bf16)          [probs this iter]
    //   s_O  = BQ  * HD  * sizeof(float)         [running accumulator]
    //   m+l  = 2 * BQ    * sizeof(float)
    //
    // HD=128 worst case: 16K + 16K + 16K + 8K + 32K + 0.5K = 88.5 KB (fits
    // SM_86 100 KB opt-in).

    #define LAUNCH_FA2(HD) do {                                              \
        const size_t smem_Q    = (size_t)BQ  * (HD) * sizeof(__nv_bfloat16); \
        const size_t smem_KV   = (size_t)BKV * (HD) * sizeof(__nv_bfloat16); \
        const size_t smem_S    = (size_t)BQ  * BKV  * sizeof(float);         \
        const size_t smem_P    = (size_t)BQ  * BKV  * sizeof(__nv_bfloat16); \
        const size_t smem_O    = (size_t)BQ  * (HD) * sizeof(float);         \
        const size_t smem_ml   = (size_t)(2 * BQ)   * sizeof(float);         \
        const size_t smem_size =                                              \
            smem_Q + smem_KV + smem_S + smem_P + smem_O + smem_ml;           \
                                                                             \
        cudaError_t attr_err = cudaFuncSetAttribute(                         \
            fa2_fwd_hd##HD,                                                  \
            cudaFuncAttributeMaxDynamicSharedMemorySize,                     \
            (int)smem_size                                                   \
        );                                                                   \
        if (attr_err != cudaSuccess) return (int)attr_err;                   \
                                                                             \
        fa2_fwd_hd##HD<<<grid, block, smem_size, s>>>(                       \
            (const __nv_bfloat16*)Q,                                         \
            (const __nv_bfloat16*)K,                                         \
            (const __nv_bfloat16*)V,                                         \
            (__nv_bfloat16*)O,                                               \
            LSE,                                                             \
            seq_len_q, seq_len_kv,                                           \
            scale_val                                                        \
        );                                                                   \
    } while (0)

    switch (head_dim) {
        case 64:  LAUNCH_FA2(64);  break;
        case 96:  LAUNCH_FA2(96);  break;
        case 128: LAUNCH_FA2(128); break;
        default:  return -1;
    }

    #undef LAUNCH_FA2
    return (int)cudaGetLastError();
}

} // extern "C"
