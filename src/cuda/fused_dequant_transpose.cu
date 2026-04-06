// fused_dequant_transpose.cu
// GPU-side FP8 E4M3 -> BF16 dequantization with fused transpose.
// Reads [M, N] row-major FP8, writes [N, M] row-major BF16.
// One kernel launch, zero allocation (output pre-allocated by caller).

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define TILE 32
// +1 padding avoids bank conflicts on 32-wide shared memory access
#define TILE_PAD (TILE + 1)

extern "C" {

// Block: 32x8 threads, each thread processes 4 rows (loop of 4).
// Shared memory tile: 32x33 bf16 (33 = TILE + 1 padding).
__global__ void fp8_dequant_transpose_kernel(
    const unsigned char* __restrict__ input,   // [M, N] row-major FP8
    __nv_bfloat16* __restrict__ output,        // [N, M] row-major BF16
    const float scale,
    const int M,
    const int N
) {
    __shared__ __nv_bfloat16 tile[TILE][TILE_PAD];

    // Block covers a TILE x TILE region of the INPUT matrix.
    // gridDim.x tiles along N (columns), gridDim.y tiles along M (rows).
    const int bx = blockIdx.x * TILE;  // column offset in input
    const int by = blockIdx.y * TILE;  // row offset in input

    const int tx = threadIdx.x;  // 0..31
    const int ty = threadIdx.y;  // 0..7

    // ---- Phase 1: Read FP8 input in coalesced order, dequant, store to smem ----
    // Each thread reads 4 elements (ty, ty+8, ty+16, ty+24) from the tile.
    #pragma unroll
    for (int k = 0; k < TILE; k += 8) {
        int row = by + ty + k;
        int col = bx + tx;

        float val = 0.0f;
        if (row < M && col < N) {
            unsigned char bits = input[row * N + col];
            int sign = (bits >> 7) & 1;
            int exp  = (bits >> 3) & 0xF;
            int mant = bits & 0x7;

            if (exp == 0 && mant == 0) {
                val = 0.0f;
            } else if (exp == 0) {
                // Subnormal: value = (mant/8) * 2^(-6)
                val = ldexpf((float)mant / 8.0f, -6);
            } else if (exp == 15 && mant != 0) {
                // NaN in E4M3 (exp=15, mant!=0) -> NaN
                val = __int_as_float(0x7FC00000);  // quiet NaN
            } else if (exp == 15 && mant == 0) {
                // E4M3 has no Inf (exp=15,mant=0 is largest normal: 448.0)
                // But handle defensively as max value
                val = ldexpf(1.0f + 0.0f / 8.0f, 15 - 7);
            } else {
                // Normal: value = (1 + mant/8) * 2^(exp-7)
                val = ldexpf(1.0f + (float)mant / 8.0f, exp - 7);
            }

            if (sign) val = -val;
            val *= scale;
        }

        // Store to shared memory: tile[ty+k][tx] so reads along tx are coalesced
        tile[ty + k][tx] = __float2bfloat16(val);
    }

    __syncthreads();

    // ---- Phase 2: Write transposed from smem to output ----
    // Now we read tile[tx][ty+k] (transposed) and write to output[N, M].
    // Output row = bx + tx (was input column), output col = by + ty + k (was input row).
    #pragma unroll
    for (int k = 0; k < TILE; k += 8) {
        int out_row = bx + tx;    // transposed: input column -> output row
        int out_col = by + ty + k; // transposed: input row -> output column

        if (out_row < N && out_col < M) {
            // Read transposed from smem: tile[ty+k][tx] -> tile[row_in_tile][col_in_tile]
            // but we want the transpose, so we read tile[ty+k][tx] which is the
            // element that was at input position (by+ty+k, bx+tx).
            // For the output at (bx+tx, by+ty+k) we need input(by+ty+k, bx+tx).
            output[out_row * M + out_col] = tile[ty + k][tx];
        }
    }
}

int flame_fused_dequant_transpose_bf16(
    const void* input,      // FP8 E4M3 source [M, N] row-major
    void* output,           // BF16 dest [N, M] row-major (pre-allocated)
    float scale,            // dequant scale factor
    int M,                  // rows of input
    int N,                  // cols of input
    void* stream            // CUDA stream
) {
    if (M == 0 || N == 0) return 0;

    // Grid: each block handles one 32x32 tile of the input
    dim3 block(TILE, 8);  // 32x8 = 256 threads
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    fp8_dequant_transpose_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)input,
        (__nv_bfloat16*)output,
        scale,
        M,
        N
    );
    return cudaGetLastError();
}

} // extern "C"
