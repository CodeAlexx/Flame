// grouped_mm.cu
// Grouped BF16 matrix multiplication: one kernel, E experts.
//
// Semantics (matches PyTorch torch.nn.functional.grouped_mm(x, w, offs=offsets)):
//
//   x:       (T, K)   BF16, expert-major ordered tokens
//   w:       (E, K, N) BF16, stacked weights (PyTorch layout)
//   offsets: (E,)     i32, EXCLUSIVE cumulative end indices into x.
//                     Expert e's row range is [offsets[e-1] .. offsets[e]),
//                     with offsets[-1] := 0.
//   out:     (T, N)   BF16
//
// For each expert e:
//   out[offsets[e-1] : offsets[e], :] = x[offsets[e-1] : offsets[e], :] @ w[e]
//
// Accumulation is FP32 via WMMA tensor cores (SM80+).
//
// --------------------------------------------------------------------------
// Kernel design — WMMA tiled GEMM on SM80/86/89:
//   BM=128, BN=128, BK=32   (output tile; 128x128 BF16 per block)
//   Warp tile: 64x64        (2x2 grid of warp tiles per block → 4 warps)
//   WMMA frag: 16x16x16 BF16→FP32
//   Per warp: 4x4 grid of WMMA tiles (16 frag accumulators in FP32)
//   Threads per block = 4 * 32 = 128
//
// Grid:
//   grid.z = E (one expert per z)
//   grid.y = ceil(T_max / BM)  (blocks early-exit if past their expert's rows)
//   grid.x = ceil(N / BN)
//
// Each block loads K-tiles of A=x[BM,BK] and B=w[e][BK,BN] into shared
// memory via cp.async-friendly vectorized BF16 loads, then runs wmma::mma_sync
// over BK/WMMA_K iterations per warp tile row/col.
//
// Bounds: we bounds-check rows for partial T_e tiles; K and N are padded in
// shared memory so OOB K/N values are zero.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

namespace {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;

constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;

constexpr int WARPS_M = 2;                       // 2 warp tiles along M
constexpr int WARPS_N = 2;                       // 2 warp tiles along N
constexpr int WARP_TILE_M = BM / WARPS_M;        // 64
constexpr int WARP_TILE_N = BN / WARPS_N;        // 64
constexpr int NUM_WARPS = WARPS_M * WARPS_N;     // 4
constexpr int THREADS = NUM_WARPS * 32;          // 128

constexpr int FRAG_M = WARP_TILE_M / WM;         // 4
constexpr int FRAG_N = WARP_TILE_N / WN;         // 4
constexpr int FRAG_K = BK / WK;                  // 2

// Pad shared tiles slightly to reduce bank conflicts.
constexpr int A_STRIDE = BK + 8;                 // BM x (BK + pad) BF16
constexpr int B_STRIDE = BN + 8;                 // BK x (BN + pad) BF16

// Double-buffered shared memory (2 stages for overlap).
constexpr int STAGES = 2;

__device__ __forceinline__ int expert_start(const int* offsets, int e) {
    return (e == 0) ? 0 : offsets[e - 1];
}

// Load BM*BK worth of BF16 from global into shared, using vector loads where
// possible. `row_end` is the global row bound for the expert's slice.
__device__ __forceinline__ void load_A_tile(
    __nv_bfloat16* __restrict__ As,          // [BM][A_STRIDE]
    const __nv_bfloat16* __restrict__ X,     // expert's slice start
    int t_rows, int K,
    int block_row, int kk,
    int tid
) {
    // Each iteration loads 4 BF16 (8 bytes) per thread via uint2 when aligned.
    // Simpler: 1 element per thread per step; BM*BK / THREADS = 128*32/128 = 32.
    #pragma unroll
    for (int i = 0; i < (BM * BK) / THREADS; ++i) {
        int lin = tid + i * THREADS;
        int r = lin / BK;
        int c = lin - r * BK;
        int gr = block_row + r;
        int gc = kk + c;
        __nv_bfloat16 v = __float2bfloat16(0.0f);
        if (gr < t_rows && gc < K) {
            v = X[(size_t)gr * (size_t)K + (size_t)gc];
        }
        As[r * A_STRIDE + c] = v;
    }
}

__device__ __forceinline__ void load_B_tile(
    __nv_bfloat16* __restrict__ Bs,          // [BK][B_STRIDE]
    const __nv_bfloat16* __restrict__ W,     // expert's weight start
    int K, int N,
    int kk, int block_col,
    int tid
) {
    // BK*BN / THREADS = 32*128/128 = 32.
    #pragma unroll
    for (int i = 0; i < (BK * BN) / THREADS; ++i) {
        int lin = tid + i * THREADS;
        int r = lin / BN;
        int c = lin - r * BN;
        int gr = kk + r;
        int gc = block_col + c;
        __nv_bfloat16 v = __float2bfloat16(0.0f);
        if (gr < K && gc < N) {
            v = W[(size_t)gr * (size_t)N + (size_t)gc];
        }
        Bs[r * B_STRIDE + c] = v;
    }
}

__global__ void grouped_mm_bf16_kernel(
    const __nv_bfloat16* __restrict__ X,     // (T, K)
    const __nv_bfloat16* __restrict__ W,     // (E, K, N)
    const int* __restrict__ OFFS,            // (E,)
    __nv_bfloat16* __restrict__ Y,           // (T, N)
    int K,
    int N,
    int E
) {
    const int e = blockIdx.z;
    if (e >= E) return;

    const int row_start = expert_start(OFFS, e);
    const int row_end   = OFFS[e];
    const int T_e       = row_end - row_start;
    if (T_e <= 0) return;

    const int block_row = blockIdx.y * BM;
    if (block_row >= T_e) return;

    const int block_col = blockIdx.x * BN;
    if (block_col >= N) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m = warp_id / WARPS_N;   // 0..1
    const int warp_n = warp_id % WARPS_N;   // 0..1

    __shared__ __nv_bfloat16 As[BM * A_STRIDE];
    __shared__ __nv_bfloat16 Bs[BK * B_STRIDE];

    const __nv_bfloat16* Xptr = X + (size_t)row_start * (size_t)K;
    const __nv_bfloat16* Wptr = W + ((size_t)e * (size_t)K * (size_t)N);
    __nv_bfloat16* Yptr = Y + (size_t)row_start * (size_t)N;

    // FP32 accumulator fragments, FRAG_M × FRAG_N of them per warp.
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag[FRAG_M][FRAG_N];
    #pragma unroll
    for (int i = 0; i < FRAG_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAG_N; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    for (int kk = 0; kk < K; kk += BK) {
        load_A_tile(As, Xptr, T_e, K, block_row, kk, tid);
        load_B_tile(Bs, Wptr, K, N, kk, block_col, tid);
        __syncthreads();

        // WMMA accumulate: C[warp_m*64+i*16 : +16, warp_n*64+j*16 : +16]
        //   += sum over kk (2 WMMA_K tiles) of A @ B.
        #pragma unroll
        for (int kf = 0; kf < FRAG_K; ++kf) {
            wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag[FRAG_M];
            wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> b_frag[FRAG_N];

            // Load A fragments for this warp's row tiles
            #pragma unroll
            for (int i = 0; i < FRAG_M; ++i) {
                int row_off = warp_m * WARP_TILE_M + i * WM;
                const __nv_bfloat16* a_ptr = As + row_off * A_STRIDE + kf * WK;
                wmma::load_matrix_sync(a_frag[i], a_ptr, A_STRIDE);
            }
            // Load B fragments for this warp's col tiles
            #pragma unroll
            for (int j = 0; j < FRAG_N; ++j) {
                int col_off = warp_n * WARP_TILE_N + j * WN;
                const __nv_bfloat16* b_ptr = Bs + (kf * WK) * B_STRIDE + col_off;
                wmma::load_matrix_sync(b_frag[j], b_ptr, B_STRIDE);
            }
            // Outer product accumulate
            #pragma unroll
            for (int i = 0; i < FRAG_M; ++i) {
                #pragma unroll
                for (int j = 0; j < FRAG_N; ++j) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // Store accumulators. Convert FP32 → BF16 via a small shared scratch then
    // copy out with bounds checking.
    // Stage: use the As buffer as scratch for the warp's 64x64 BF16 tile.
    // Each warp writes its own 64x64 into a distinct region of As
    // (64*64 = 4096 BF16 per warp; As has BM*A_STRIDE = 128*40 = 5120 BF16,
    // enough for one warp's tile at a time, but we need all 4 warps in parallel
    // → use a dedicated [NUM_WARPS][64][64] buffer instead).
    __shared__ __nv_bfloat16 out_tile[NUM_WARPS * WARP_TILE_M * (WARP_TILE_N + 8)];
    constexpr int OT_STRIDE = WARP_TILE_N + 8;

    __nv_bfloat16* my_tile = out_tile + warp_id * WARP_TILE_M * OT_STRIDE;

    // store_matrix_sync expects FP32 dst for FP32 accumulator, so we first
    // store to a local FP32 register-backed buffer then convert. Simpler:
    // per-thread extract via fragment.x[] and write BF16 directly.
    {
        // The fragment layout of wmma::accumulator is implementation-defined,
        // but store_matrix_sync to an FP32 shared tile is portable. We allocate
        // a per-warp FP32 scratch of size 64x64 = 16 KB... that's 16 KB * 4 = 64 KB,
        // too much. Instead: write each 16x16 frag one at a time into the BF16
        // output tile via a two-step: FP32 scratch (one frag at a time) → BF16.
        __shared__ float frag_scratch[NUM_WARPS * WM * WN];
        float* my_scratch = frag_scratch + warp_id * WM * WN;

        #pragma unroll
        for (int i = 0; i < FRAG_M; ++i) {
            #pragma unroll
            for (int j = 0; j < FRAG_N; ++j) {
                wmma::store_matrix_sync(my_scratch, c_frag[i][j], WN, wmma::mem_row_major);
                // Convert FP32 → BF16 into out_tile
                // WM*WN = 256 elems; 32 lanes, 8 per lane.
                const int lane = tid & 31;
                #pragma unroll
                for (int t = 0; t < (WM * WN) / 32; ++t) {
                    int lin = lane * ((WM * WN) / 32) + t;
                    int rr = lin / WN;
                    int cc = lin - rr * WN;
                    int out_r = i * WM + rr;
                    int out_c = j * WN + cc;
                    my_tile[out_r * OT_STRIDE + out_c] =
                        __float2bfloat16(my_scratch[lin]);
                }
                __syncwarp();
            }
        }
    }
    __syncthreads();

    // Write the warp's 64x64 BF16 tile to global output with bounds checking.
    // 64*64 / 32 = 128 elements per thread (spread across warp).
    // Actually each warp writes its own 64x64 with 32 threads, so 128 elems/thread.
    {
        const int lane = tid & 31;
        const int base_r = warp_m * WARP_TILE_M;
        const int base_c = warp_n * WARP_TILE_N;

        #pragma unroll
        for (int t = 0; t < (WARP_TILE_M * WARP_TILE_N) / 32; ++t) {
            int lin = t * 32 + lane;
            int rr = lin / WARP_TILE_N;
            int cc = lin - rr * WARP_TILE_N;
            int out_r = block_row + base_r + rr;
            int out_c = block_col + base_c + cc;
            if (out_r < T_e && out_c < N) {
                Yptr[(size_t)out_r * (size_t)N + (size_t)out_c] =
                    my_tile[rr * OT_STRIDE + cc];
            }
        }
    }
}

} // anonymous namespace

extern "C" {

int flame_grouped_mm_bf16(
    const void* x,
    const void* w,
    const void* offsets,
    void* y,
    int T_max,
    int K,
    int N,
    int E,
    void* stream
) {
    if (E <= 0 || K <= 0 || N <= 0 || T_max <= 0) return 0;

    const int blocks_m = (T_max + BM - 1) / BM;
    const int blocks_n = (N     + BN - 1) / BN;

    dim3 grid((unsigned)blocks_n, (unsigned)blocks_m, (unsigned)E);
    dim3 block(THREADS, 1, 1);

    grouped_mm_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)x,
        (const __nv_bfloat16*)w,
        (const int*)offsets,
        (__nv_bfloat16*)y,
        K, N, E
    );
    cudaError_t err = cudaGetLastError();
    return (int)err;
}

} // extern "C"
