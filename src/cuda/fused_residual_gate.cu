// fused_residual_gate.cu
// Fused: out = x + gate * attn_out
// Replaces mul + add (2 kernels) with 1.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" {

__global__ void fused_residual_gate_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ attn_out,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ output,
    const size_t n_elements
) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;

    for (size_t i = idx; i < n_elements; i += stride) {
        float xv = __bfloat162float(x[i]);
        float av = __bfloat162float(attn_out[i]);
        float gv = __bfloat162float(gate[i]);
        output[i] = __float2bfloat16(xv + gv * av);
    }
}

int flame_fused_residual_gate_bf16(
    const void* x, const void* attn_out, const void* gate,
    void* output, size_t n_elements, void* stream
) {
    if (n_elements == 0) return 0;
    const int block_size = 256;
    int n_blocks = (int)((n_elements + block_size - 1) / block_size);
    if (n_blocks > 65535) n_blocks = 65535;

    fused_residual_gate_bf16_kernel<<<n_blocks, block_size, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)x, (const __nv_bfloat16*)attn_out,
        (const __nv_bfloat16*)gate, (__nv_bfloat16*)output, n_elements
    );
    return cudaGetLastError();
}

} // extern "C"
