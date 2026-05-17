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

// Packed SwiGLU backward for an input whose last dimension is [gate | up].
// Produces one full packed gradient, avoiding the two narrow-backward
// zero+scatter passes required by separate gate/up views.
__global__ void swiglu_split_backward_bf16_kernel(
    const __nv_bfloat16* __restrict__ grad_out,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ grad_input,
    const int64_t* __restrict__ input_strides,
    int64_t input_offset,
    int ndim,
    int64_t out_n,
    int64_t half)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= out_n) return;

    int64_t row = idx / half;
    int64_t d = idx - row * half;
    int64_t rem = row;
    int64_t in_addr = input_offset;

    // Decode all non-last dimensions from the flattened row index.
    for (int axis = ndim - 2; axis >= 0; --axis) {
        int64_t dim = input_strides[ndim + axis];
        int64_t coord = rem % dim;
        rem /= dim;
        in_addr += coord * input_strides[axis];
    }

    int64_t last_stride = input_strides[ndim - 1];
    int64_t gate_addr = in_addr + d * last_stride;
    int64_t up_addr = in_addr + (half + d) * last_stride;

    float g = __bfloat162float(grad_out[idx]);
    float x = __bfloat162float(input[gate_addr]);
    float u = __bfloat162float(input[up_addr]);

    float sig = 1.0f / (1.0f + expf(-x));
    float silu_x = x * sig;
    float dsilu = sig + x * sig * (1.0f - sig);

    int64_t grad_base = row * (half * 2);
    grad_input[grad_base + d] = __float2bfloat16_rn(g * dsilu * u);
    grad_input[grad_base + half + d] = __float2bfloat16_rn(g * silu_x);
}

extern "C" int flame_swiglu_split_backward_bf16(
    const void* grad_out,
    const void* input,
    void* grad_input,
    const int64_t* input_meta,
    int64_t input_offset,
    int ndim,
    int64_t out_n,
    int64_t half,
    cudaStream_t stream)
{
    int block = 256;
    int grid = (int)((out_n + block - 1) / block);
    swiglu_split_backward_bf16_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)grad_out,
        (const __nv_bfloat16*)input,
        (__nv_bfloat16*)grad_input,
        input_meta,
        input_offset,
        ndim,
        out_n,
        half);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
