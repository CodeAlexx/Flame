// fp8_quant.cu
// GPU-side BF16 → FP8 E4M3 quantization for activation offload.
//
// flame_bf16_to_fp8(input, output, inv_scale, n, stream)
//   output[i] = fp8_e4m3(bf16_to_f32(input[i]) * inv_scale)
//
// The caller provides inv_scale = 1.0 / scale where
// scale = absmax / 448.0 (E4M3 max representable). Scale computation
// is the caller's responsibility — it can use a fixed scale for known
// activation ranges or compute absmax separately.
//
// Pairs with fp8_dequant.cu::flame_fp8_to_bf16 for the reverse path.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" {

__device__ unsigned char f32_to_fp8_e4m3(float val) {
    unsigned char sign = 0;
    if (val < 0.0f) { sign = 0x80; val = -val; }

    // Clamp to E4M3 max representable: 448.0
    if (val > 448.0f) val = 448.0f;
    if (val == 0.0f) return sign;

    // E4M3: 1 sign, 4 exponent (bias=7), 3 mantissa, no inf/nan
    int exp;
    float frac = frexpf(val, &exp);
    // frexpf: val = frac * 2^exp, frac in [0.5, 1.0)
    // Rewrite: val = (2*frac) * 2^(exp-1) = (1 + 2*frac - 1) * 2^(exp-1)
    exp -= 1; // true exponent

    int biased = exp + 7; // biased exponent

    if (biased <= 0) {
        // Subnormal
        float subnorm = val * ldexpf(1.0f, 6); // val / 2^(-6)
        int mant = (int)(subnorm * 8.0f + 0.5f);
        if (mant > 7) mant = 7;
        if (mant < 0) mant = 0;
        return sign | (unsigned char)mant;
    }

    if (biased >= 15) {
        // Saturate to max: 0_1110_111 = 448.0
        return sign | 0x7F;
    }

    // Normal: extract 3-bit mantissa
    float mant_f = (2.0f * frac - 1.0f) * 8.0f;
    int mant = (int)(mant_f + 0.5f);
    if (mant > 7) { mant = 0; biased += 1; }
    if (biased >= 15) return sign | 0x7F;

    return sign | ((unsigned char)biased << 3) | (unsigned char)mant;
}

__global__ void bf16_to_fp8_kernel(
    const __nv_bfloat16* __restrict__ input,
    unsigned char* __restrict__ output,
    const float inv_scale,
    const size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    for (size_t i = idx; i < n; i += stride) {
        float val = __bfloat162float(input[i]) * inv_scale;
        output[i] = f32_to_fp8_e4m3(val);
    }
}

int flame_bf16_to_fp8(
    const void* input,
    void* output,
    float inv_scale,
    size_t n_elements,
    void* stream
) {
    if (n_elements == 0) return 0;
    const int block = 256;
    int grid = (int)((n_elements + block - 1) / block);
    if (grid > 65535) grid = 65535;

    bf16_to_fp8_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input,
        (unsigned char*)output,
        inv_scale,
        n_elements
    );
    return cudaGetLastError();
}

} // extern "C"
