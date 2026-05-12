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
