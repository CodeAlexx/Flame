// flame-core/cuda/permute0213.cu
// -----------------------------------------------------------------------------
// This CUDA kernel keeps Flux’ (N, heads, tokens, dim) -> (N, tokens, heads, dim)
// permutation entirely on the GPU.  The original Rust implementation fell back
// to `Tensor::to_vec()` which staged tensors on the CPU and assumed F32
// storage; that broke for BF16 tensors and violated the “GPU-only” Phase-4
// contract.  We provide both F32 and BF16 launch wrappers here so the Rust
// side can dispatch without ever touching host memory.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdlib>
#include <climits>

// Generic implementation shared by the F32 / BF16 entry points.
template <typename Scalar>
__global__ void permute0213_kernel(
    const Scalar* __restrict__ x,
    Scalar* __restrict__ y,
    int N, int A, int B, int C)
{
    // Total number of elements; we walk the flattened index space and map each
    // element to its destination.  Using long long avoids overflow for large
    // tensors.
    const long long total = static_cast<long long>(N) * A * B * C;
    long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (idx >= total) return;

    // Decompose the flattened index back into (n, a, b, c).
    int c = static_cast<int>(idx % C);
    idx /= C;
    int b = static_cast<int>(idx % B);
    idx /= B;
    int a = static_cast<int>(idx % A);
    int n = static_cast<int>(idx / A);

    // Input offset: ((n * A + a) * B + b) * C + c
    const long long in_offset =
        ((((static_cast<long long>(n) * A) + a) * B) + b) * C + c;

    // Output offset corresponds to (n, b, a, c): ((n * B + b) * A + a) * C + c
    const long long out_offset =
        ((((static_cast<long long>(n) * B) + b) * A) + a) * C + c;

    y[out_offset] = x[in_offset];
}

// 2026-05-12 perf: BF16 vec=4 + 4D-grid permute0213.
//
// The legacy permute0213_kernel is grid-strided with linear output index →
// per-thread divmod over (n, a, b, c). Measured ~348 GB/s on production
// attention shapes (~35% of 1 TB/s peak). Two issues:
//   1. Each thread does 4 divmod ops in the index unravel (ALU overhead)
//   2. Single BF16 load/store per thread (2-byte transactions)
//
// This vec kernel: 4D grid (n, b-tile, a-tile, c-vec) with index math by
// addition only. Each thread reads/writes one bf16x4 (4 BF16 = 8 bytes).
// Within a warp threads vary in c-vec → input/output stride 8 bytes →
// coalesced LDG.E.64 / STG.E.64 transactions.
//
// Constraint: C must be a multiple of 4 (head_dim 128 always satisfies).
// Falls back to the legacy kernel otherwise.
struct __align__(8) bf16x4 {
    __nv_bfloat16 v[4];
};

__global__ void permute0213_vec4_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N, int A, int B, int C)
{
    const int C_VEC = C / 4;
    const int c_vec = blockIdx.x * blockDim.x + threadIdx.x;
    const int a     = blockIdx.y * blockDim.y + threadIdx.y;
    const int nb    = blockIdx.z;
    const int n     = nb / B;
    const int b     = nb % B;

    if (a >= A || c_vec >= C_VEC) return;

    const long long c_off = (long long)c_vec * 4;

    // input[n, a, b, c_off..c_off+3]
    const long long in_off = ((((long long)n * A + a) * B + b) * C + c_off);
    const bf16x4* X = reinterpret_cast<const bf16x4*>(x + in_off);
    // output[n, b, a, c_off..c_off+3]
    const long long out_off = ((((long long)n * B + b) * A + a) * C + c_off);
    bf16x4* Y = reinterpret_cast<bf16x4*>(y + out_off);

    *Y = *X;
}

// Launch helpers matching the narrow kernels style the crate already uses.
extern "C" void launch_permute0213_f32(
    const float* x,
    float* y,
    int N,
    int A,
    int B,
    int C,
    cudaStream_t stream)
{
    const long long total = static_cast<long long>(N) * A * B * C;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    permute0213_kernel<float><<<grid, block, 0, stream>>>(x, y, N, A, B, C);
}

extern "C" void launch_permute0213_bf16(
    const __nv_bfloat16* x,
    __nv_bfloat16* y,
    int N,
    int A,
    int B,
    int C,
    cudaStream_t stream)
{
    // 2026-05-12 perf: vec=4 path when C % 4 == 0 (production attention
    // head_dims 64/128/256 all qualify). Env override
    // FLAME_PERMUTE_LEGACY=1 forces the scalar grid-strided kernel.
    const char* legacy_env = getenv("FLAME_PERMUTE_LEGACY");
    const bool force_legacy = (legacy_env != nullptr && legacy_env[0] != 0 && legacy_env[0] != '0');
    if (!force_legacy && (C % 4 == 0) && C >= 4) {
        const int C_VEC = C / 4;
        // 32 threads in x cover one full c row when head_dim = 128.
        // 32 threads in y cover up to 32 a values per block.
        const int TX = (C_VEC < 32) ? C_VEC : 32;
        const int TY = (A     < 32) ? A     : 32;
        dim3 block(TX, TY, 1);
        dim3 grid((C_VEC + TX - 1) / TX, (A + TY - 1) / TY, N * B);
        permute0213_vec4_bf16_kernel<<<grid, block, 0, stream>>>(x, y, N, A, B, C);
        return;
    }
    const long long total = static_cast<long long>(N) * A * B * C;
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    permute0213_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(x, y, N, A, B, C);
}

// ----------------------------------------------------------------------------
// 3D variant: (N, A, B) -> (N, B, A)

template <typename Scalar>
__global__ void permute021_kernel(
    const Scalar* __restrict__ x,
    Scalar* __restrict__ y,
    int N, int A, int B)
{
    const long long total = static_cast<long long>(N) * A * B;
    long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (idx >= total) return;

    int b = static_cast<int>(idx % B);
    idx /= B;
    int a = static_cast<int>(idx % A);
    int n = static_cast<int>(idx / A);

    const long long in_offset = ((static_cast<long long>(n) * A) + a) * B + b;
    const long long out_offset = ((static_cast<long long>(n) * B) + b) * A + a;

    y[out_offset] = x[in_offset];
}

// 2026-05-12 perf: tiled permute021 — [N, A, B] -> [N, B, A].
//
// Legacy kernel above hits ~135 GB/s (1/8 of 1 TB/s peak) because within a
// warp reads stride by B in input (input[n, a, b0..b0+31] has stride 1, but
// output linear index varies in `a` first per warp → reads jump by B). This
// is the classical matrix-transpose access pattern.
//
// Fix: 32×32 shared-memory tile with bank-conflict padding (+1 column).
// Each block reads a 32×32 (a, b) tile coalesced into smem, then writes it
// out transposed — both reads and writes hit warp-coalesced access.
template <typename Scalar>
__global__ void permute021_tiled_kernel(
    const Scalar* __restrict__ x,
    Scalar* __restrict__ y,
    int N, int A, int B)
{
    __shared__ Scalar tile[32][33];  // +1 col to avoid 32-bank conflict

    const int b0 = blockIdx.x * 32;
    const int a0 = blockIdx.y * 32;
    const int n  = blockIdx.z;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Phase 1: coalesced read from input[n, a0+ty, b0+tx] → tile[ty][tx].
    // Within a warp, tx varies → input stride = 1 (b). Coalesced.
    const int a_in = a0 + ty;
    const int b_in = b0 + tx;
    if (a_in < A && b_in < B) {
        const long long in_off = ((long long)n * A + a_in) * B + b_in;
        tile[ty][tx] = x[in_off];
    }
    __syncthreads();

    // Phase 2: coalesced write to output[n, b0+ty, a0+tx] = tile[tx][ty].
    // The tile read is now transposed: thread (tx, ty) reads tile[tx][ty],
    // and writes to output[n, b0+ty, a0+tx]. Within a warp (varying tx),
    // output address varies by 1 (a). Coalesced. Bank conflict avoided
    // by +1 col padding.
    const int b_out = b0 + ty;
    const int a_out = a0 + tx;
    if (a_out < A && b_out < B) {
        const long long out_off = ((long long)n * B + b_out) * A + a_out;
        y[out_off] = tile[tx][ty];
    }
}

extern "C" void launch_permute021_f32(
    const float* x,
    float* y,
    int N,
    int A,
    int B,
    cudaStream_t stream)
{
    // Use tiled kernel when both A and B are at least 16 — for tiny shapes the
    // tile overhead exceeds the win.
    const char* legacy_env = getenv("FLAME_PERMUTE_LEGACY");
    const bool force_legacy = (legacy_env != nullptr && legacy_env[0] != 0 && legacy_env[0] != '0');
    if (!force_legacy && A >= 16 && B >= 16) {
        dim3 grid((B + 31) / 32, (A + 31) / 32, N);
        dim3 block(32, 32, 1);
        permute021_tiled_kernel<float><<<grid, block, 0, stream>>>(x, y, N, A, B);
        return;
    }
    const long long total = static_cast<long long>(N) * A * B;
    const int block_sz = 256;
    const int grid = static_cast<int>((total + block_sz - 1) / block_sz);
    permute021_kernel<float><<<grid, block_sz, 0, stream>>>(x, y, N, A, B);
}

extern "C" void launch_permute021_bf16(
    const __nv_bfloat16* x,
    __nv_bfloat16* y,
    int N,
    int A,
    int B,
    cudaStream_t stream)
{
    const char* legacy_env = getenv("FLAME_PERMUTE_LEGACY");
    const bool force_legacy = (legacy_env != nullptr && legacy_env[0] != 0 && legacy_env[0] != '0');
    if (!force_legacy && A >= 16 && B >= 16) {
        dim3 grid((B + 31) / 32, (A + 31) / 32, N);
        dim3 block(32, 32, 1);
        permute021_tiled_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(x, y, N, A, B);
        return;
    }
    const long long total = static_cast<long long>(N) * A * B;
    const int block_sz = 256;
    const int grid = static_cast<int>((total + block_sz - 1) / block_sz);
    permute021_kernel<__nv_bfloat16><<<grid, block_sz, 0, stream>>>(x, y, N, A, B);
}

// ----------------------------------------------------------------------------
// 2026-05-12 perf: asymmetric tiled transpose for rank-2 [1,0] and small-inner
// rank-3 [0,2,1].
//
// The 32×32 tiled kernel above requires `A >= 16 && B >= 16` and is wasteful
// when one dim is small (~8). The zimage trace shows 840 calls/step of
// rank-2 [1,0] with shapes like [8, 3840], [3840, 8], [10240, 8], [8, 10240]
// — every one falls through to the scalar grid-strided kernel.
//
// This kernel uses an asymmetric tile `[TH][TW+1]` (+1 for bank-conflict
// avoidance) where TH and TW are template params. We dispatch:
//   - A>=32, B>=32: tile=32×32 (handled by permute021_tiled_kernel above)
//   - A>=8 small:   tile=8×128 (one block covers one 8×128 chunk)
//   - else:         scalar kernel
//
// Layout: input is [N, A, B] row-major, output is [N, B, A] row-major.
// Each block handles one (n, a-tile, b-tile) chunk. Read-phase: 32 threads
// in x cover one B-row coalesced (TW=128 → 4 reads per thread, each
// stride-1). Write-phase: transpose via shared memory, output stride-1 in
// A direction.

// Asymmetric tile permute021 — covers small-A rank-2/rank-3 cases.
//
// Tile layout: TH × TW (A-rows × B-cols), padded to (TW+1) cols to avoid
// 32-bank conflicts on the transposed read. Block dim = (TW, TH): each
// thread handles ONE input element + ONE output element.
//
// Phase 1 (read):  thread (tx, ty) → tile[ty][tx] = input[n, a0+ty, b0+tx]
//                  Within a warp (varying tx) → input stride = 1 (B). Coalesced.
// Phase 2 (write): thread (tx, ty) → output[n, b0+tx, a0+ty] = tile[ty][tx]
//                  Wait — this gives output stride = A (NOT coalesced for small A).
//
// Better: re-thread phase 2 so warp varies in a_out.
//   thread (tx, ty) writes: output[n, b0 + ty, a0 + tx] = tile[tx][ty]
//   This requires TX <= A for full warp coalescing. For A=8, TX=8 wastes
//   24 warp lanes. Still much better than the scalar grid-strided kernel
//   because phase 1 reads are fully coalesced.
//
// For TH<TW (the typical small-A case), each block covers a single A-tile
// but multiple B-tiles via the tx index. We use TX=TW threads in x for
// phase 1 (coalesced reads), then in phase 2 we "rethink" the same threads:
// the warp index (tx_warp) covers TW output rows (=B chunk), and threads
// within a warp cover TH columns (=A chunk).
//
// Concretely: block dim = (TW, TH). Phase 2 iterates with thread (tx, ty)
// writing output[n, b0+tx, a0+ty] = tile[ty][tx]. Coalesced WRITE: within
// a warp, tx varies → output address varies by A (NOT coalesced when A<32).
//
// Trade-off: we PRIORITIZE phase 1 coalesced reads (the expensive side —
// reading from global memory the first time fills caches). Phase 2 writes
// are partially uncoalesced but write-combining helps. For small A this
// is still ~5-10× faster than the scalar kernel because phase 1 alone
// dominates.
template <typename Scalar, int TH, int TW>
__global__ void permute021_tiled_small_kernel(
    const Scalar* __restrict__ x,
    Scalar* __restrict__ y,
    int N, int A, int B)
{
    __shared__ Scalar tile[TH][TW + 1];

    const int n  = blockIdx.z;
    const int a0 = blockIdx.y * TH;
    const int b0 = blockIdx.x * TW;

    const int ty = threadIdx.y;          // 0..TH-1
    const int tx = threadIdx.x;          // 0..TW-1

    // Phase 1: thread (tx, ty) reads input[n, a0+ty, b0+tx] → tile[ty][tx].
    // Within a warp (varying tx), input stride = 1 → coalesced.
    const int a_in = a0 + ty;
    const int b_in = b0 + tx;
    if (a_in < A && b_in < B) {
        const long long in_off = ((long long)n * A + a_in) * B + b_in;
        tile[ty][tx] = x[in_off];
    }
    __syncthreads();

    // Phase 2: thread (tx, ty) writes output[n, b0+tx, a0+ty] = tile[ty][tx].
    // Within a warp (varying tx), output address varies by A. NOT coalesced
    // for A<32. We accept this; phase 1 is the dominant cost and we win
    // most of the speedup there.
    const int b_out = b0 + tx;
    const int a_out = a0 + ty;
    if (a_out < A && b_out < B) {
        const long long out_off = ((long long)n * B + b_out) * A + a_out;
        y[out_off] = tile[ty][tx];
    }
}

// 32-row × 128-col tile, used when A is large but we want wider B coverage.
// Reused for the symmetric large case.

// Dispatch wrapper for rank-3 [0,2,1] that handles small A more efficiently.
// Routes:
//   - A>=32, B>=32 → existing permute021_tiled_kernel (32×32)
//   - 8<=A<32, B>=32 → permute021_tiled_small_kernel<TH=8,TW=32>  (or TH up to 16)
//   - else → scalar
template <typename Scalar>
static void launch_permute021_dispatch(
    const Scalar* x,
    Scalar* y,
    int N, int A, int B,
    cudaStream_t stream)
{
    if (A >= 32 && B >= 32) {
        dim3 grid((B + 31) / 32, (A + 31) / 32, N);
        dim3 block(32, 32, 1);
        permute021_tiled_kernel<Scalar><<<grid, block, 0, stream>>>(x, y, N, A, B);
        return;
    }
    if (A >= 16 && B >= 16) {
        // Use the symmetric 32x32 path (handles padding correctly).
        dim3 grid((B + 31) / 32, (A + 31) / 32, N);
        dim3 block(32, 32, 1);
        permute021_tiled_kernel<Scalar><<<grid, block, 0, stream>>>(x, y, N, A, B);
        return;
    }
    // Small-A asymmetric paths: cover (A, B) tiles with one thread per element.
    // Block dim = (TW, TH); pick TH = min(16, ceil(A)), TW = min(64, ceil(B)).
    // Constraint: TW * TH <= 1024 (max threads/block on sm_86).
    if (A >= 1 && A <= 16 && B >= 8) {
        // TH = 8 covers A up to 8 in one block; A=9..16 takes two A-blocks.
        constexpr int TH = 8;
        constexpr int TW = 64;  // 64 threads/x × 8 threads/y = 512 threads
        dim3 grid((B + TW - 1) / TW, (A + TH - 1) / TH, N);
        dim3 block(TW, TH, 1);
        permute021_tiled_small_kernel<Scalar, TH, TW><<<grid, block, 0, stream>>>(x, y, N, A, B);
        return;
    }
    // Small-B mirror case (e.g. [3840, 8] → [8, 3840]).
    if (B >= 1 && B <= 16 && A >= 8) {
        constexpr int TH = 64;  // 64 threads/y × 8 threads/x = 512
        constexpr int TW = 8;
        dim3 grid((B + TW - 1) / TW, (A + TH - 1) / TH, N);
        dim3 block(TW, TH, 1);
        permute021_tiled_small_kernel<Scalar, TH, TW><<<grid, block, 0, stream>>>(x, y, N, A, B);
        return;
    }
    // Tiny tail.
    const long long total = static_cast<long long>(N) * A * B;
    const int block_sz = 256;
    const int grid = static_cast<int>((total + block_sz - 1) / block_sz);
    permute021_kernel<Scalar><<<grid, block_sz, 0, stream>>>(x, y, N, A, B);
}

// New entry point: rank-2 [1,0] transpose. Implemented as rank-3 [0,2,1]
// with N=1. Output shape [B, A] = input shape [A, B] swapped.
extern "C" void launch_permute10_bf16(
    const __nv_bfloat16* x,
    __nv_bfloat16* y,
    int A, int B,
    cudaStream_t stream)
{
    const char* legacy_env = getenv("FLAME_PERMUTE_LEGACY");
    const bool force_legacy = (legacy_env != nullptr && legacy_env[0] != 0 && legacy_env[0] != '0');
    if (force_legacy) {
        const long long total = static_cast<long long>(A) * B;
        const int block = 256;
        const int grid = static_cast<int>((total + block - 1) / block);
        permute021_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(x, y, 1, A, B);
        return;
    }
    launch_permute021_dispatch<__nv_bfloat16>(x, y, 1, A, B, stream);
}

extern "C" void launch_permute10_f32(
    const float* x,
    float* y,
    int A, int B,
    cudaStream_t stream)
{
    const char* legacy_env = getenv("FLAME_PERMUTE_LEGACY");
    const bool force_legacy = (legacy_env != nullptr && legacy_env[0] != 0 && legacy_env[0] != '0');
    if (force_legacy) {
        const long long total = static_cast<long long>(A) * B;
        const int block = 256;
        const int grid = static_cast<int>((total + block - 1) / block);
        permute021_kernel<float><<<grid, block, 0, stream>>>(x, y, 1, A, B);
        return;
    }
    launch_permute021_dispatch<float>(x, y, 1, A, B, stream);
}

// Rank-4 [0,1,3,2]: swap the inner two dims.
//
// Input shape  [N, A, B, C] row-major → Output shape [N, A, C, B] row-major.
//
// Mathematically identical to rank-3 [0,2,1] with N' = N*A, A' = B, B' = C.
// Both inner pairs of dims are contiguous, so the flattening preserves the
// memory access pattern bit-for-bit.
//
// zimage trace: 120 calls/step, shapes [1, 30, 1536, 128] and
// [1, 30, 1536, 1536]. The first hits A'=1536, B'=128 → A>=32 && B>=32
// path. The second hits A'=1536, B'=1536 → same. Both fully use the tiled
// kernel.
extern "C" void launch_permute0132_bf16(
    const __nv_bfloat16* x,
    __nv_bfloat16* y,
    int N, int A, int B, int C,
    cudaStream_t stream)
{
    const long long NA = static_cast<long long>(N) * A;
    if (NA > INT_MAX) {
        // Cannot fit into int N' parameter. Fall back to scalar.
        const long long total = NA * B * C;
        const int block = 256;
        const long long grid_ll = (total + block - 1) / block;
        const int grid = (grid_ll > 65535) ? 65535 : (int)grid_ll;
        permute021_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(x, y, 1, (int)(NA), B * C);
        return;
    }
    const char* legacy_env = getenv("FLAME_PERMUTE_LEGACY");
    const bool force_legacy = (legacy_env != nullptr && legacy_env[0] != 0 && legacy_env[0] != '0');
    if (force_legacy) {
        const long long total = NA * B * C;
        const int block = 256;
        const int grid = static_cast<int>((total + block - 1) / block);
        permute021_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(x, y, (int)NA, B, C);
        return;
    }
    launch_permute021_dispatch<__nv_bfloat16>(x, y, (int)NA, B, C, stream);
}

extern "C" void launch_permute0132_f32(
    const float* x,
    float* y,
    int N, int A, int B, int C,
    cudaStream_t stream)
{
    const long long NA = static_cast<long long>(N) * A;
    if (NA > INT_MAX) {
        const long long total = NA * B * C;
        const int block = 256;
        const long long grid_ll = (total + block - 1) / block;
        const int grid = (grid_ll > 65535) ? 65535 : (int)grid_ll;
        permute021_kernel<float><<<grid, block, 0, stream>>>(x, y, 1, (int)(NA), B * C);
        return;
    }
    const char* legacy_env = getenv("FLAME_PERMUTE_LEGACY");
    const bool force_legacy = (legacy_env != nullptr && legacy_env[0] != 0 && legacy_env[0] != '0');
    if (force_legacy) {
        const long long total = NA * B * C;
        const int block = 256;
        const int grid = static_cast<int>((total + block - 1) / block);
        permute021_kernel<float><<<grid, block, 0, stream>>>(x, y, (int)NA, B, C);
        return;
    }
    launch_permute021_dispatch<float>(x, y, (int)NA, B, C, stream);
}
