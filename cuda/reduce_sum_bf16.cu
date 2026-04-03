#include <cuda_runtime.h>
#include <cuda_bf16.h>

template<int TPB>
__device__ float block_reduce_sum(float v) {
    __shared__ float smem[TPB];
    smem[threadIdx.x] = v;
    __syncthreads();

    if (TPB >= 512) { if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256]; __syncthreads(); }
    if (TPB >= 256) { if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128]; __syncthreads(); }
    if (TPB >= 128) { if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64]; __syncthreads(); }

    float val = smem[threadIdx.x];
    if (threadIdx.x < 32) {
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<int TPB, typename OutType, typename StoreFn>
__global__ void sum_last_keepdim_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    OutType* __restrict__ out,
    int B, int M, int K,
    StoreFn store)
{
    int b = blockIdx.y;
    int m = blockIdx.x;
    if (b >= B || m >= M) return;

    const __nv_bfloat16* row = x + ((b * M + m) * K);
    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += TPB) {
        acc += static_cast<float>(row[k]);
    }
    float sum = block_reduce_sum<TPB>(acc);
    if (threadIdx.x == 0) {
        store(out + (b * M + m), sum);
    }
}

struct StoreBF16 {
    __device__ void operator()(__nv_bfloat16* ptr, float value) const {
        *ptr = __float2bfloat16(value);
    }
};

struct StoreF32 {
    __device__ void operator()(float* ptr, float value) const {
        *ptr = value;
    }
};

extern "C" void launch_sum_last_keepdim_bf16(
    const void* x,
    void* y,
    int B, int M, int K,
    cudaStream_t stream)
{
    constexpr int TPB = 256;
    dim3 grid(M, B, 1);
    dim3 block(TPB, 1, 1);
    sum_last_keepdim_bf16_kernel<TPB><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<__nv_bfloat16*>(y),
        B, M, K,
        StoreBF16{});
}

extern "C" void launch_sum_last_keepdim_bf16_to_f32(
    const void* x,
    void* y,
    int B, int M, int K,
    cudaStream_t stream)
{
    constexpr int TPB = 256;
    dim3 grid(M, B, 1);
    dim3 block(TPB, 1, 1);
    sum_last_keepdim_bf16_kernel<TPB><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<float*>(y),
        B, M, K,
        StoreF32{});
}
