// fp8_dequant.cu
// GPU-side FP8 E4M3 → BF16 dequantization.
// output[i] = bf16(fp8_to_f32(input[i]) * scale)

#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" {

__global__ void fp8_to_bf16_kernel(
    const unsigned char* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const float scale,
    const size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    for (size_t i = idx; i < n; i += stride) {
        unsigned char bits = input[i];
        int sign = (bits >> 7) & 1;
        int exp = (bits >> 3) & 0xF;
        int mant = bits & 0x7;

        float val;
        if (exp == 0 && mant == 0) {
            val = 0.0f;
        } else if (exp == 0) {
            val = ldexpf((float)mant / 8.0f, -6);
        } else {
            val = ldexpf(1.0f + (float)mant / 8.0f, exp - 7);
        }

        if (sign) val = -val;
        val *= scale;
        output[i] = __float2bfloat16(val);
    }
}

int flame_fp8_to_bf16(
    const void* input,
    void* output,
    float scale,
    size_t n_elements,
    void* stream
) {
    if (n_elements == 0) return 0;
    const int block = 256;
    int grid = (int)((n_elements + block - 1) / block);
    if (grid > 65535) grid = 65535;

    fp8_to_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)input,
        (__nv_bfloat16*)output,
        scale,
        n_elements
    );
    return cudaGetLastError();
}

} // extern "C"
