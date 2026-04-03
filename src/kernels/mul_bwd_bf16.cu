#include <cuda_bf16.h>

extern "C" __global__
void mul_backward_bf16_same_shape_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ go,
    __nv_bfloat16* __restrict__ gx,
    __nv_bfloat16* __restrict__ gy,
    size_t N)
{
    if (N == 0) {
        return;
    }

    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < N;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        float xf = __bfloat162float(x[idx]);
        float yf = __bfloat162float(y[idx]);
        float gof = __bfloat162float(go[idx]);

        float dxf = gof * yf;
        float dyf = gof * xf;

        gx[idx] = __float2bfloat16(dxf);
        gy[idx] = __float2bfloat16(dyf);
    }
}
