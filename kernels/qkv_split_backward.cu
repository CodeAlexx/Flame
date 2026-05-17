#include <cuda_bf16.h>
#include <cuda_runtime.h>

__global__ void qkv_split_permute_backward_bf16_kernel(
    const __nv_bfloat16* __restrict__ grad_out,
    __nv_bfloat16* __restrict__ grad_input,
    const int64_t* __restrict__ grad_strides,
    int64_t grad_offset,
    int part,
    int64_t B,
    int64_t H,
    int64_t N,
    int64_t D)
{
    int64_t hd = H * D;
    int64_t row = 3 * hd;
    int64_t total = B * N * row;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t step = (int64_t)blockDim.x * gridDim.x;
    while (idx < total) {
        int64_t c = idx % row;
        int64_t t = idx / row;
        int64_t n = t % N;
        int64_t b = t / N;
        int64_t this_part = c / hd;
        if (this_part == part) {
            int64_t within = c - this_part * hd;
            int64_t h = within / D;
            int64_t d = within - h * D;
            int64_t go = grad_offset
                + b * grad_strides[0]
                + h * grad_strides[1]
                + n * grad_strides[2]
                + d * grad_strides[3];
            grad_input[idx] = grad_out[go];
        } else {
            grad_input[idx] = __float2bfloat16_rn(0.0f);
        }
        idx += step;
    }
}

__global__ void qkv_split_permute_backward_f32_kernel(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_input,
    const int64_t* __restrict__ grad_strides,
    int64_t grad_offset,
    int part,
    int64_t B,
    int64_t H,
    int64_t N,
    int64_t D)
{
    int64_t hd = H * D;
    int64_t row = 3 * hd;
    int64_t total = B * N * row;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t step = (int64_t)blockDim.x * gridDim.x;
    while (idx < total) {
        int64_t c = idx % row;
        int64_t t = idx / row;
        int64_t n = t % N;
        int64_t b = t / N;
        int64_t this_part = c / hd;
        if (this_part == part) {
            int64_t within = c - this_part * hd;
            int64_t h = within / D;
            int64_t d = within - h * D;
            int64_t go = grad_offset
                + b * grad_strides[0]
                + h * grad_strides[1]
                + n * grad_strides[2]
                + d * grad_strides[3];
            grad_input[idx] = grad_out[go];
        } else {
            grad_input[idx] = 0.0f;
        }
        idx += step;
    }
}

extern "C" int flame_qkv_split_permute_backward_bf16(
    const void* grad_out,
    void* grad_input,
    const int64_t* grad_strides,
    int64_t grad_offset,
    int part,
    int64_t B,
    int64_t H,
    int64_t N,
    int64_t D,
    cudaStream_t stream)
{
    int64_t total = B * N * 3 * H * D;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    qkv_split_permute_backward_bf16_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)grad_out,
        (__nv_bfloat16*)grad_input,
        grad_strides,
        grad_offset,
        part,
        B,
        H,
        N,
        D);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int flame_qkv_split_permute_backward_f32(
    const void* grad_out,
    void* grad_input,
    const int64_t* grad_strides,
    int64_t grad_offset,
    int part,
    int64_t B,
    int64_t H,
    int64_t N,
    int64_t D,
    cudaStream_t stream)
{
    int64_t total = B * N * 3 * H * D;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    qkv_split_permute_backward_f32_kernel<<<grid, block, 0, stream>>>(
        (const float*)grad_out,
        (float*)grad_input,
        grad_strides,
        grad_offset,
        part,
        B,
        H,
        N,
        D);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
