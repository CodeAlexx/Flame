// fp16_to_bf16.cu
// GPU-side FP16 (IEEE half) → BF16 conversion.
// output[i] = bf16(fp16_to_f32(input[i]))
//
// In-place safe: input and output may point to the same buffer since both
// are 2 bytes per element and the kernel processes one element per thread.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern "C" {

__global__ void fp16_to_bf16_kernel(
    const __half* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    for (size_t i = idx; i < n; i += stride) {
        float val = __half2float(input[i]);
        output[i] = __float2bfloat16(val);
    }
}

int flame_fp16_to_bf16(
    const void* input,
    void* output,
    size_t n_elements,
    void* stream
) {
    if (n_elements == 0) return 0;
    const int block = 256;
    int grid = (int)((n_elements + block - 1) / block);
    if (grid > 65535) grid = 65535;

    fp16_to_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __half*)input,
        (__nv_bfloat16*)output,
        n_elements
    );
    return cudaGetLastError();
}

} // extern "C"
