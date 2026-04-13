// f32_to_bf16.cu
// GPU-side F32 → BF16 conversion.
// NOT in-place safe: input is 4 bytes, output is 2 bytes per element.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" {

__global__ void f32_to_bf16_kernel(
    const float* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    for (size_t i = idx; i < n; i += stride) {
        output[i] = __float2bfloat16(input[i]);
    }
}

int flame_f32_to_bf16(
    const void* input,
    void* output,
    size_t n_elements,
    void* stream
) {
    if (n_elements == 0) return 0;
    const int block = 256;
    int grid = (int)((n_elements + block - 1) / block);
    if (grid > 65535) grid = 65535;

    f32_to_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)input,
        (__nv_bfloat16*)output,
        n_elements
    );
    return cudaGetLastError();
}

} // extern "C"
