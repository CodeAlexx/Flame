// mxfp4_dequant.cu
// GPU-side MXFP4 → BF16 dequantization.
//
// MXFP4 = 32 FP4 (E2M1) elements share one 8-bit E8M0 (exponent-only) scale.
// On-disk layout (matches HuggingFace transformers MXFP4 packing):
//   - blocks: uint8[..., G, 16]   — 16 bytes per block, 2 packed FP4 nibbles/byte
//   - scales: uint8[..., G]       — one E8M0 exponent byte per 32-element block
//
// The 16 representable FP4 magnitudes (signed by the high bit of the nibble):
//   index  : 0    1    2    3    4    5    6    7
//   magn   : 0.0  0.5  1.0  1.5  2.0  3.0  4.0  6.0
//   nibble high bit (0x8) = sign
//
// Nibble packing (matches transformers/integrations/mxfp4.py):
//   - low nibble  (byte & 0x0F) → even output index (0, 2, 4, …)
//   - high nibble (byte >> 4)   → odd  output index (1, 3, 5, …)
//
// Scale: out *= 2^(scale_byte - 127). 127 is the IEEE E8M0 bias.
//
// Reference: transformers/integrations/mxfp4.py::convert_moe_packed_tensors
//            FP4_VALUES list. Verified bit-exact at unit-test level.
//
// Output dtype is BF16 (matches transformers default and Lens use case).

#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" {

// FP4 LUT (constant memory). Indexed by 4-bit nibble (0-15).
// LUT[i & 7] gives magnitude; (i & 8) is the sign bit.
// Values from transformers FP4_VALUES exactly:
//   [+0.0,+0.5,+1.0,+1.5,+2.0,+3.0,+4.0,+6.0,
//    -0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0]
__device__ __constant__ float FP4_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// One thread = one MXFP4 block (32 output BF16 elements).
//
// Grid: 1-D, gridDim.x * blockDim.x = rows_total (total blocks).
// Each thread:
//   - loads 16 bytes from blocks[bid * 16 .. bid*16 + 16]
//   - loads 1 byte from scales[bid]
//   - emits 32 BF16 values to out[bid * 32 .. bid*32 + 32]
__global__ void flame_mxfp4_to_bf16_kernel(
    const unsigned char* __restrict__ blocks,
    const unsigned char* __restrict__ scales,
    __nv_bfloat16* __restrict__ out,
    const size_t rows_total
) {
    size_t bid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    for (size_t r = bid; r < rows_total; r += stride) {
        // E8M0: scale = 2^(scale_byte - 127). ldexpf takes an int exponent.
        int scale_exp = (int)scales[r] - 127;

        const unsigned char* blk_ptr = blocks + r * 16;
        __nv_bfloat16*       out_ptr = out    + r * 32;

        // 16 bytes → 32 FP4 nibbles → 32 BF16 outputs.
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            unsigned char byte = blk_ptr[i];
            unsigned int  lo   = byte & 0x0Fu;
            unsigned int  hi   = (byte >> 4) & 0x0Fu;

            float v_lo = FP4_LUT[lo];
            float v_hi = FP4_LUT[hi];

            // Apply E8M0 scale. ldexpf(x, e) = x * 2^e.
            v_lo = ldexpf(v_lo, scale_exp);
            v_hi = ldexpf(v_hi, scale_exp);

            out_ptr[2*i    ] = __float2bfloat16(v_lo);
            out_ptr[2*i + 1] = __float2bfloat16(v_hi);
        }
    }
}

// C entry. Returns 0 on success, cudaGetLastError() code otherwise.
//
// - blocks_ptr: device uint8 buffer, length = rows_total * 16 bytes
// - scales_ptr: device uint8 buffer, length = rows_total bytes
// - out_ptr:    device BF16 buffer, length = rows_total * 32 elements
// - rows_total: number of 32-element MXFP4 blocks
// - stream:     CUDA stream
int flame_mxfp4_to_bf16(
    const void* blocks_ptr,
    const void* scales_ptr,
    void*       out_ptr,
    size_t      rows_total,
    void*       stream
) {
    if (rows_total == 0) return 0;

    const int block = 256;
    long long grid_ll = (long long)((rows_total + (size_t)block - 1) / (size_t)block);
    if (grid_ll > 65535) grid_ll = 65535;
    int grid = (int)grid_ll;

    flame_mxfp4_to_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)blocks_ptr,
        (const unsigned char*)scales_ptr,
        (__nv_bfloat16*)out_ptr,
        rows_total
    );
    return cudaGetLastError();
}

} // extern "C"
