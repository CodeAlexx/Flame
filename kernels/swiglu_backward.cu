#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Fused SwiGLU backward: out = silu(gate) * up
// d_up   = silu(gate) * grad_out
// d_gate = dsilu(gate) * up * grad_out
//        = (sig(gate) + gate*sig(gate)*(1-sig(gate))) * up * grad_out
// Single kernel replaces ~10 separate GpuOps calls.

__global__ void swiglu_backward_bf16_kernel(
    const __nv_bfloat16* __restrict__ grad_out,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ d_gate,
    __nv_bfloat16* __restrict__ d_up,
    int64_t n)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = __bfloat162float(grad_out[idx]);
    float x = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);

    // sig = sigmoid(gate)
    float sig = 1.0f / (1.0f + expf(-x));
    // silu(gate) = gate * sig
    float silu_x = x * sig;
    // dsilu(gate) = sig + gate*sig*(1-sig)
    float dsilu = sig + x * sig * (1.0f - sig);

    d_up[idx] = __float2bfloat16_rn(g * silu_x);
    d_gate[idx] = __float2bfloat16_rn(g * dsilu * u);
}

extern "C" int flame_swiglu_backward_bf16(
    const void* grad_out,
    const void* gate,
    const void* up,
    void* d_gate,
    void* d_up,
    int64_t n,
    cudaStream_t stream)
{
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    swiglu_backward_bf16_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)grad_out,
        (const __nv_bfloat16*)gate,
        (const __nv_bfloat16*)up,
        (__nv_bfloat16*)d_gate,
        (__nv_bfloat16*)d_up,
        n);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
