// fused_norm_modulate.cu
// Fused RMS norm + modulation: out = (x * rsqrt(mean(x²) + eps) * weight) * (1 + scale) + shift
// Replaces 2 separate kernels (fused_rms_norm + fused_modulate) with 1.
// Saves one full pass over the data.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" {

__device__ __forceinline__ float warp_reduce_sum_nm(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void fused_rms_norm_modulate_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ norm_weight,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ shift,
    __nv_bfloat16* __restrict__ output,
    const int cols,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const __nv_bfloat16* row_in = x + (size_t)row * cols;
    const __nv_bfloat16* row_scale = scale + (size_t)row * cols;
    const __nv_bfloat16* row_shift = shift + (size_t)row * cols;
    __nv_bfloat16* row_out = output + (size_t)row * cols;

    // Pass 1: sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += block_size) {
        float val = __bfloat162float(row_in[i]);
        sum_sq += val * val;
    }

    sum_sq = warp_reduce_sum_nm(sum_sq);

    __shared__ float shared[32];
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (lane_id == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (block_size / 32)) ? shared[lane_id] : 0.0f;
        val = warp_reduce_sum_nm(val);
        if (lane_id == 0) shared[0] = val;
    }
    __syncthreads();

    float mean_sq = shared[0] / (float)cols;
    float rsqrt_val = rsqrtf(mean_sq + eps);

    // Pass 2: normalize + modulate
    for (int i = tid; i < cols; i += block_size) {
        float val = __bfloat162float(row_in[i]);
        float w = __bfloat162float(norm_weight[i]);
        float s = __bfloat162float(row_scale[i]);
        float h = __bfloat162float(row_shift[i]);
        float normed = val * rsqrt_val * w;
        float result = normed * (1.0f + s) + h;
        row_out[i] = __float2bfloat16(result);
    }
}

int flame_fused_rms_norm_modulate_bf16(
    const void* x, const void* norm_weight,
    const void* scale, const void* shift,
    void* output, int rows, int cols, float eps, void* stream
) {
    int block_size = ((cols + 31) / 32) * 32;
    if (block_size > 1024) block_size = 1024;
    if (block_size < 32) block_size = 32;

    fused_rms_norm_modulate_bf16_kernel<<<rows, block_size, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)x, (const __nv_bfloat16*)norm_weight,
        (const __nv_bfloat16*)scale, (const __nv_bfloat16*)shift,
        (__nv_bfloat16*)output, cols, eps
    );
    return cudaGetLastError();
}

} // extern "C"
