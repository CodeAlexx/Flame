// flash_attention_fwd.cu
// Flash Attention forward pass for FLAME inference.
// BF16 in/out, FP32 accumulation, online softmax (log-sum-exp trick).
// No backward pass needed — inference only.
//
// Replaces: Q@K^T (bmm) + scale + softmax (cast+exp+sum+div) + S@V (bmm)
//           = 5+ kernel launches → 1
//
// Algorithm:
//   For each query block Qi (tile of Q rows):
//     Initialize: O_i = 0, l_i = 0, m_i = -inf
//     For each key/value block Kj, Vj:
//       S_ij = Qi @ Kj^T * scale        (in shared memory)
//       m_new = max(m_i, rowmax(S_ij))   (online max update)
//       P_ij = exp(S_ij - m_new)         (safe softmax numerator)
//       l_new = l_i * exp(m_i - m_new) + rowsum(P_ij)  (online denominator)
//       O_i = O_i * (l_i * exp(m_i - m_new) / l_new) + P_ij @ Vj / l_new
//       m_i = m_new, l_i = l_new
//     Write O_i to output
//
// Grid: (num_heads * batch_size, ceil(seq_len_q / BLOCK_Q))
// Each thread block handles one head, one tile of queries.
//
// Tuned for: head_dim=128 (LTX-2, FLUX, Klein, most modern models)
//            RTX 3090 (SM 8.6, 128KB shared memory)

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

extern "C" {

// Tile sizes — tuned for head_dim=128 on Ampere
// BLOCK_Q: query tile size (rows of Q per thread block)
// BLOCK_KV: key/value tile size (rows of K/V loaded per inner iteration)
// HEAD_DIM: must match model's head dimension
#define BLOCK_Q 64
#define BLOCK_KV 64
#define HEAD_DIM 128
#define WARP_SIZE 32
#define NUM_WARPS 4
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE)  // 128 threads

// Shared memory layout:
//   Q tile:  [BLOCK_Q, HEAD_DIM]   floats  = 64*128*4  = 32KB
//   K tile:  [BLOCK_KV, HEAD_DIM]  floats  = 64*128*4  = 32KB
//   V tile:  [BLOCK_KV, HEAD_DIM]  floats  = 64*128*4  = 32KB
//   S tile:  [BLOCK_Q, BLOCK_KV]   floats  = 64*64*4   = 16KB
//   Total:                                              = 112KB
//   (Fits in 128KB shared memory on Ampere/Ada)
//
// NOTE: If this exceeds shared memory, reduce BLOCK_Q or BLOCK_KV to 32.

__global__ void flash_attention_fwd_bf16(
    const __nv_bfloat16* __restrict__ Q,    // [B, H, N_q, D]
    const __nv_bfloat16* __restrict__ K,    // [B, H, N_kv, D]
    const __nv_bfloat16* __restrict__ V,    // [B, H, N_kv, D]
    __nv_bfloat16* __restrict__ O,          // [B, H, N_q, D]
    const int N_q,                           // query sequence length
    const int N_kv,                          // key/value sequence length
    const float scale                        // 1/sqrt(head_dim)
) {
    // Which head and which query tile
    const int head_batch_idx = blockIdx.x;   // flattened B*H
    const int q_tile_idx = blockIdx.y;       // which BLOCK_Q tile of queries

    const int tid = threadIdx.x;

    // Offsets into global memory
    const size_t qkv_head_offset = (size_t)head_batch_idx * N_q * HEAD_DIM;
    const size_t kv_head_offset = (size_t)head_batch_idx * N_kv * HEAD_DIM;

    const int q_start = q_tile_idx * BLOCK_Q;
    if (q_start >= N_q) return;
    const int q_end = min(q_start + BLOCK_Q, N_q);
    const int q_tile_rows = q_end - q_start;

    // Shared memory
    extern __shared__ float smem[];
    float* s_Q = smem;                                           // [BLOCK_Q, HEAD_DIM]
    float* s_K = s_Q + BLOCK_Q * HEAD_DIM;                      // [BLOCK_KV, HEAD_DIM]
    float* s_V = s_K + BLOCK_KV * HEAD_DIM;                     // [BLOCK_KV, HEAD_DIM]
    float* s_S = s_V + BLOCK_KV * HEAD_DIM;                     // [BLOCK_Q, BLOCK_KV]

    // ---- Load Q tile into shared memory ----
    // Each thread loads multiple elements
    const __nv_bfloat16* q_ptr = Q + qkv_head_offset + (size_t)q_start * HEAD_DIM;
    for (int i = tid; i < q_tile_rows * HEAD_DIM; i += BLOCK_SIZE) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        s_Q[row * HEAD_DIM + col] = __bfloat162float(q_ptr[row * HEAD_DIM + col]);
    }
    __syncthreads();

    // ---- Per-row running statistics ----
    // Each thread handles a subset of query rows
    // For BLOCK_Q=64 and BLOCK_SIZE=128, each thread handles ~0.5 rows
    // We assign threads to rows: thread t handles row (t % q_tile_rows)
    // But simpler: store per-row stats in registers for threads that "own" rows

    // Use shared memory for row stats
    // m_i: running max per query row
    // l_i: running sum of exp per query row
    // O_acc: running output accumulator [BLOCK_Q, HEAD_DIM] — reuse s_Q after loading
    // Actually, we need O_acc separately since we still need Q for recompute in some variants.
    // For forward-only, Q is used once per KV tile, so we can keep it.

    // Simpler approach: process in a more straightforward way.
    // Each thread is responsible for specific (row, col) pairs in the output.

    // Output accumulator in registers
    // Each thread handles ceil(BLOCK_Q * HEAD_DIM / BLOCK_SIZE) elements
    #define ELEMS_PER_THREAD ((BLOCK_Q * HEAD_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE)

    float o_acc[ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) o_acc[i] = 0.0f;

    // Row-wise running max and running sum — in shared memory
    // (One per query row in the tile)
    // Reuse s_S for this since we haven't used it yet
    float* row_max = s_S;                           // [BLOCK_Q]
    float* row_sum = s_S + BLOCK_Q;                 // [BLOCK_Q]
    float* row_max_prev = s_S + 2 * BLOCK_Q;       // [BLOCK_Q]

    // Initialize
    for (int i = tid; i < q_tile_rows; i += BLOCK_SIZE) {
        row_max[i] = -FLT_MAX;
        row_sum[i] = 0.0f;
    }
    __syncthreads();

    // ---- Iterate over KV tiles ----
    const int num_kv_tiles = (N_kv + BLOCK_KV - 1) / BLOCK_KV;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        const int kv_start = kv_tile * BLOCK_KV;
        const int kv_end = min(kv_start + BLOCK_KV, N_kv);
        const int kv_tile_rows = kv_end - kv_start;

        // Load K tile
        const __nv_bfloat16* k_ptr = K + kv_head_offset + (size_t)kv_start * HEAD_DIM;
        for (int i = tid; i < kv_tile_rows * HEAD_DIM; i += BLOCK_SIZE) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            s_K[row * HEAD_DIM + col] = __bfloat162float(k_ptr[row * HEAD_DIM + col]);
        }

        // Load V tile
        const __nv_bfloat16* v_ptr = V + kv_head_offset + (size_t)kv_start * HEAD_DIM;
        for (int i = tid; i < kv_tile_rows * HEAD_DIM; i += BLOCK_SIZE) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            s_V[row * HEAD_DIM + col] = __bfloat162float(v_ptr[row * HEAD_DIM + col]);
        }
        __syncthreads();

        // ---- Compute S = Q @ K^T * scale ----
        // S[i][j] = sum_d(Q[i][d] * K[j][d]) * scale
        // Store in s_S[BLOCK_Q][BLOCK_KV]
        for (int idx = tid; idx < q_tile_rows * kv_tile_rows; idx += BLOCK_SIZE) {
            int qi = idx / kv_tile_rows;
            int kj = idx % kv_tile_rows;
            float dot = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < HEAD_DIM; d++) {
                dot += s_Q[qi * HEAD_DIM + d] * s_K[kj * HEAD_DIM + d];
            }
            s_S[qi * BLOCK_KV + kj] = dot * scale;
        }
        __syncthreads();

        // ---- Online softmax: update row_max ----
        // Save previous max for rescaling
        for (int i = tid; i < q_tile_rows; i += BLOCK_SIZE) {
            row_max_prev[i] = row_max[i];
            float local_max = row_max[i];
            for (int j = 0; j < kv_tile_rows; j++) {
                local_max = fmaxf(local_max, s_S[i * BLOCK_KV + j]);
            }
            row_max[i] = local_max;
        }
        __syncthreads();

        // ---- Compute P = exp(S - row_max) and update row_sum ----
        // Also rescale previous row_sum by exp(old_max - new_max)
        for (int i = tid; i < q_tile_rows; i += BLOCK_SIZE) {
            float correction = expf(row_max_prev[i] - row_max[i]);
            row_sum[i] = row_sum[i] * correction;

            float local_sum = 0.0f;
            for (int j = 0; j < kv_tile_rows; j++) {
                float p = expf(s_S[i * BLOCK_KV + j] - row_max[i]);
                s_S[i * BLOCK_KV + j] = p;  // overwrite S with P
                local_sum += p;
            }
            row_sum[i] += local_sum;
        }
        __syncthreads();

        // ---- Rescale running output and add P @ V ----
        // O_acc[qi][d] = O_acc[qi][d] * correction + sum_j(P[qi][j] * V[j][d])
        for (int idx = tid; idx < q_tile_rows * HEAD_DIM; idx += BLOCK_SIZE) {
            int qi = idx / HEAD_DIM;
            int d = idx % HEAD_DIM;

            float correction = expf(row_max_prev[qi] - row_max[qi]);
            int reg_idx = idx / BLOCK_SIZE;
            if (reg_idx < ELEMS_PER_THREAD) {
                // Use register accumulator where possible
            }

            // Simpler: use shared memory for O accumulator
            // (register approach is complex with variable thread-to-element mapping)
        }

        // Actually, let's use a cleaner approach with shared memory for O
        // This is simpler and correct, even if slightly less optimal
        __syncthreads();
    }

    // This kernel structure is getting complex. Let me provide a cleaner version.
    // See flash_attention_fwd_v2 below.
}


// ============================================================================
// CLEAN VERSION: Simpler, correct, practical for inference
// ============================================================================
// Trade-off: slightly less register-optimal but much easier to verify correct.
// Uses shared memory for the output accumulator.
// For inference (not training), this is plenty fast.

// Reduced tile sizes to fit shared memory more conservatively
#define BQ 32       // query tile
#define BKV 32      // kv tile
#define HD 128      // head dim (compile-time constant for LTX-2)
#define THREADS 128

__global__ void flash_attn_fwd_v2(
    const __nv_bfloat16* __restrict__ Q,    // [B*H, N_q, HD]
    const __nv_bfloat16* __restrict__ K,    // [B*H, N_kv, HD]
    const __nv_bfloat16* __restrict__ V,    // [B*H, N_kv, HD]
    __nv_bfloat16* __restrict__ O,          // [B*H, N_q, HD]
    const int N_q,
    const int N_kv,
    const float scale
) {
    const int bh = blockIdx.x;         // which (batch, head)
    const int q_tile = blockIdx.y;     // which query tile
    const int tid = threadIdx.x;

    const int q_start = q_tile * BQ;
    if (q_start >= N_q) return;
    const int q_rows = min(BQ, N_q - q_start);

    // Global memory base pointers
    const __nv_bfloat16* Q_base = Q + (size_t)bh * N_q * HD + (size_t)q_start * HD;
    const __nv_bfloat16* K_base = K + (size_t)bh * N_kv * HD;
    const __nv_bfloat16* V_base = V + (size_t)bh * N_kv * HD;
    __nv_bfloat16* O_base = O + (size_t)bh * N_q * HD + (size_t)q_start * HD;

    // Shared memory:
    //   s_Q:    [BQ, HD]    = 32*128 = 4096 floats = 16KB
    //   s_K:    [BKV, HD]   = 32*128 = 4096 floats = 16KB
    //   s_V:    [BKV, HD]   = 32*128 = 4096 floats = 16KB
    //   s_O:    [BQ, HD]    = 32*128 = 4096 floats = 16KB
    //   s_S:    [BQ, BKV]   = 32*32  = 1024 floats = 4KB
    //   s_m:    [BQ]        = 32 floats             = 128B
    //   s_l:    [BQ]        = 32 floats             = 128B
    //   Total: ~68KB — fits comfortably
    extern __shared__ float smem[];
    float* s_Q = smem;
    float* s_K = s_Q + BQ * HD;
    float* s_V = s_K + BKV * HD;
    float* s_O = s_V + BKV * HD;
    float* s_S = s_O + BQ * HD;
    float* s_m = s_S + BQ * BKV;
    float* s_l = s_m + BQ;

    // Load Q tile into shared memory
    for (int i = tid; i < q_rows * HD; i += THREADS) {
        s_Q[i] = __bfloat162float(Q_base[i]);
    }

    // Initialize output accumulator and stats
    for (int i = tid; i < q_rows * HD; i += THREADS) {
        s_O[i] = 0.0f;
    }
    for (int i = tid; i < q_rows; i += THREADS) {
        s_m[i] = -FLT_MAX;
        s_l[i] = 0.0f;
    }
    __syncthreads();

    // Iterate over KV tiles
    const int num_kv_tiles = (N_kv + BKV - 1) / BKV;

    for (int kv_t = 0; kv_t < num_kv_tiles; kv_t++) {
        const int kv_start = kv_t * BKV;
        const int kv_rows = min(BKV, N_kv - kv_start);

        // Load K tile
        const __nv_bfloat16* K_tile = K_base + (size_t)kv_start * HD;
        for (int i = tid; i < kv_rows * HD; i += THREADS) {
            s_K[i] = __bfloat162float(K_tile[i]);
        }

        // Load V tile
        const __nv_bfloat16* V_tile = V_base + (size_t)kv_start * HD;
        for (int i = tid; i < kv_rows * HD; i += THREADS) {
            s_V[i] = __bfloat162float(V_tile[i]);
        }
        __syncthreads();

        // Compute S = Q @ K^T * scale
        for (int idx = tid; idx < q_rows * kv_rows; idx += THREADS) {
            const int qi = idx / kv_rows;
            const int kj = idx % kv_rows;
            float dot = 0.0f;
            const float* q_row = s_Q + qi * HD;
            const float* k_row = s_K + kj * HD;
            #pragma unroll 16
            for (int d = 0; d < HD; d++) {
                dot += q_row[d] * k_row[d];
            }
            s_S[qi * BKV + kj] = dot * scale;
        }
        __syncthreads();

        // Online softmax: find new row max, compute correction
        for (int qi = tid; qi < q_rows; qi += THREADS) {
            float old_max = s_m[qi];

            // Find max of this tile's scores
            float tile_max = -FLT_MAX;
            for (int j = 0; j < kv_rows; j++) {
                tile_max = fmaxf(tile_max, s_S[qi * BKV + j]);
            }

            float new_max = fmaxf(old_max, tile_max);
            float correction = expf(old_max - new_max);

            // Rescale running sum
            s_l[qi] *= correction;

            // Compute exp(s - new_max) and accumulate sum
            float tile_sum = 0.0f;
            for (int j = 0; j < kv_rows; j++) {
                float p = expf(s_S[qi * BKV + j] - new_max);
                s_S[qi * BKV + j] = p;  // store P for P@V below
                tile_sum += p;
            }
            s_l[qi] += tile_sum;

            // Store new max and correction for output rescaling
            s_m[qi] = new_max;

            // Rescale output accumulator for this row
            // O_i = O_i * correction
            for (int d = 0; d < HD; d++) {
                s_O[qi * HD + d] *= correction;
            }
        }
        __syncthreads();

        // Accumulate P @ V into output
        // O[qi][d] += sum_j(P[qi][j] * V[j][d])
        for (int idx = tid; idx < q_rows * HD; idx += THREADS) {
            const int qi = idx / HD;
            const int d = idx % HD;
            float sum = 0.0f;
            for (int j = 0; j < kv_rows; j++) {
                sum += s_S[qi * BKV + j] * s_V[j * HD + d];
            }
            s_O[qi * HD + d] += sum;
        }
        __syncthreads();
    }

    // Normalize output: O = O / l
    for (int idx = tid; idx < q_rows * HD; idx += THREADS) {
        const int qi = idx / HD;
        float inv_l = 1.0f / s_l[qi];
        s_O[idx] *= inv_l;
    }
    __syncthreads();

    // Write output to global memory as BF16
    for (int i = tid; i < q_rows * HD; i += THREADS) {
        O_base[i] = __float2bfloat16(s_O[i]);
    }
}


// Entry point for flame-core FFI
//
// Q, K, V: [B*H, N, D] BF16 — heads already split, contiguous per head
// O:       [B*H, N, D] BF16 output
//
// If your layout is [B, H, N, D], that's the same as [B*H, N, D] since H and N
// are the inner dimensions.
//
// head_dim must be 128 (compile-time constant). If you need other head dims,
// add #define variants or use NVRTC JIT.
int flame_flash_attention_bf16(
    const void* Q,
    const void* K,
    const void* V,
    void* O,
    int batch_heads,     // B * H
    int seq_len_q,       // N_q
    int seq_len_kv,      // N_kv (can differ from N_q for cross-attention)
    int head_dim,        // must be 128
    void* stream
) {
    if (head_dim != 128) {
        return -1;  // unsupported head dim
    }

    dim3 grid(batch_heads, (seq_len_q + BQ - 1) / BQ);
    dim3 block(THREADS);

    // Shared memory: s_Q + s_K + s_V + s_O + s_S + s_m + s_l
    size_t smem_size = (BQ * HD + BKV * HD + BKV * HD + BQ * HD + BQ * BKV + BQ + BQ) * sizeof(float);

    // Request increased shared memory if needed
    cudaFuncSetAttribute(
        flash_attn_fwd_v2,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_size
    );

    flash_attn_fwd_v2<<<grid, block, smem_size, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)Q,
        (const __nv_bfloat16*)K,
        (const __nv_bfloat16*)V,
        (__nv_bfloat16*)O,
        seq_len_q,
        seq_len_kv,
        1.0f / sqrtf((float)head_dim)
    );

    return cudaGetLastError();
}

} // extern "C"
