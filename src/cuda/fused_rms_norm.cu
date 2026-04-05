// fused_rms_norm.cu
// Fused RMS normalization: BF16 input → BF16 output with weight multiply.
// Replaces 6 kernel launches with 1.
//
// Algorithm per row:
//   mean_sq = mean(x^2)
//   rsqrt_val = rsqrt(mean_sq + eps)
//   out[i] = bf16(float(x[i]) * rsqrt_val * float(weight[i]))
//
// One block per row. Warp-level reduction for mean_sq.
// Supports up to 16384 hidden dim (4 warps x 4096 elements per warp).

#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" {

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void fused_rms_norm_bf16(
    const __nv_bfloat16* __restrict__ input,   // [rows, cols]
    const __nv_bfloat16* __restrict__ weight,  // [cols]
    __nv_bfloat16* __restrict__ output,        // [rows, cols]
    const int cols,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const __nv_bfloat16* row_in = input + (size_t)row * cols;
    __nv_bfloat16* row_out = output + (size_t)row * cols;

    // Pass 1: compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += block_size) {
        float val = __bfloat162float(row_in[i]);
        sum_sq += val * val;
    }

    // Warp reduction
    sum_sq = warp_reduce_sum(sum_sq);

    // Cross-warp reduction via shared memory
    __shared__ float shared[32]; // max 32 warps
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (lane_id == 0) {
        shared[warp_id] = sum_sq;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        float val = (lane_id < (block_size / 32)) ? shared[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();

    float mean_sq = shared[0] / (float)cols;
    float rsqrt_val = rsqrtf(mean_sq + eps);

    // Pass 2: normalize and write output
    for (int i = tid; i < cols; i += block_size) {
        float val = __bfloat162float(row_in[i]);
        float w = __bfloat162float(weight[i]);
        float normed = val * rsqrt_val * w;
        row_out[i] = __float2bfloat16(normed);
    }
}

// Entry point for flame-core FFI
//
// input:  [rows, cols] BF16
// weight: [cols] BF16
// output: [rows, cols] BF16
//
// Launched as: fused_rms_norm_bf16<<<rows, block_size, 0, stream>>>
int flame_fused_rms_norm_bf16(
    const void* input,
    const void* weight,
    void* output,
    int rows,
    int cols,
    float eps,
    void* stream
) {
    // Choose block size: round up to warp multiple, cap at 1024
    int block_size = ((cols + 31) / 32) * 32;
    if (block_size > 1024) block_size = 1024;
    if (block_size < 32) block_size = 32;

    fused_rms_norm_bf16<<<rows, block_size, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input,
        (const __nv_bfloat16*)weight,
        (__nv_bfloat16*)output,
        cols,
        eps
    );
    return cudaGetLastError();
}

} // extern "C"
