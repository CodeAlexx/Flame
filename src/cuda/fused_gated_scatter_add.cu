// fused_gated_scatter_add.cu
//
// Fused: accumulator[indices[t]] += expert_out[t] * gating[t]    (in-place)
//
// Shapes:
//   expert_out : (T, D)  BF16
//   gating     : (T,)    F32
//   indices    : (T,)    I32  (valid values in [0, N))
//   accum      : (N, D)  F32   (read-modify-write)
//
// Why fused: the unfused path does (a) cast expert_out BF16 -> F32, (b) broadcast
// multiply by gating[:, None], (c) scatter_add into accum. That's 3 full passes
// over (T, D) memory. We do it in one.
//
// Atomics: multiple t-indices can map to the same accumulator row (e.g. MoE
// top-K with K>1, each expert's output for the same token), so we use
// atomicAdd on F32. On Ampere+ F32 atomicAdd to global memory is a native
// instruction; throughput is effectively bandwidth-limited for sparse-enough
// writes, which is the realistic case.
//
// Grid: gridDim.y = T (one row per block), gridDim.x = ceil(D / BLOCK_D).
//        blockDim.x = BLOCK_D = 256 (one thread per D-lane).
// Each block loads gating[t] and indices[t] (warp-broadcast from lane 0).

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int BLOCK_D = 256;

__global__ void fused_gated_scatter_add_kernel(
    const __nv_bfloat16* __restrict__ expert_out,   // [T, D]
    const float*         __restrict__ gating,       // [T]
    const int*           __restrict__ indices,      // [T]
    float*               __restrict__ accum,        // [N, D]
    int T,
    int D,
    int N)
{
    int t = blockIdx.y;
    if (t >= T) return;

    // Load per-row scalars once. Every thread in the block needs them; the
    // simplest (and perfectly adequate) thing is to let every thread read the
    // same location — the L1 will coalesce that into one transaction.
    int row = indices[t];
    // Defensive bounds check: out-of-range index means "skip". This matches
    // the usual MoE convention where -1 or N flags dropped tokens.
    if (row < 0 || row >= N) return;
    float g = gating[t];

    const __nv_bfloat16* src = expert_out + (long)t * D;
    float* dst = accum + (long)row * D;

    // Grid-stride along D in case D > BLOCK_D * gridDim.x.
    int d = blockIdx.x * BLOCK_D + threadIdx.x;
    int stride = gridDim.x * BLOCK_D;
    for (; d < D; d += stride) {
        float v = __bfloat162float(src[d]) * g;
        atomicAdd(dst + d, v);
    }
}

} // namespace

extern "C" int flame_fused_gated_scatter_add_bf16(
    const void* expert_out,   // BF16 [T, D]
    const void* gating,       // F32  [T]
    const void* indices,      // I32  [T]
    void*       accum,        // F32  [N, D]  in-place
    int T,
    int D,
    int N,
    void* stream)
{
    if (T == 0 || D == 0) return 0;

    // One block per row, split D across x-dim blocks.
    int grid_x = (D + BLOCK_D - 1) / BLOCK_D;
    if (grid_x < 1) grid_x = 1;
    // Cap grid_x at a sane max; realistic D for MoE is 2048-8192, so this
    // stays small, but a guard doesn't hurt.
    if (grid_x > 65535) grid_x = 65535;

    dim3 grid(grid_x, T, 1);
    dim3 block(BLOCK_D, 1, 1);

    fused_gated_scatter_add_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)expert_out,
        (const float*)gating,
        (const int*)indices,
        (float*)accum,
        T, D, N
    );
    return (int)cudaGetLastError();
}
