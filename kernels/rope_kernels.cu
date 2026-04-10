#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

__device__ inline __nv_bfloat16 bf16_from_fp32(float x) {
    return __float2bfloat16_rn(x);
}

__device__ inline float bf16_to_fp32(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__global__ void rope_apply_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B,
    int H,
    int S,
    int Dh,
    int rope_dim,
    float base_theta,
    int pos_offset)
{
    int elements_per_head = Dh;
    int half_dim = rope_dim / 2;
    int total_pairs = B * H * S * half_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) {
        return;
    }

    int pair = idx % half_dim;
    int tmp = idx / half_dim;
    int seq = tmp % S;
    tmp /= S;
    int head = tmp % H;
    int batch = tmp / H;

    int base_index = ((batch * H + head) * S + seq) * elements_per_head;
    int even_offset = pair * 2;
    int odd_offset = even_offset + 1;

    const __nv_bfloat16* in_ptr = input + base_index;
    __nv_bfloat16* out_ptr = output + base_index;

    float freq = powf(base_theta, -2.0f * pair / static_cast<float>(rope_dim));
    float angle = static_cast<float>(pos_offset + seq) * freq;
    float sin_angle;
    float cos_angle;
    sincosf(angle, &sin_angle, &cos_angle);

    float x_even = bf16_to_fp32(in_ptr[even_offset]);
    float x_odd = bf16_to_fp32(in_ptr[odd_offset]);

    float rotated_even = x_even * cos_angle - x_odd * sin_angle;
    float rotated_odd = x_even * sin_angle + x_odd * cos_angle;

    out_ptr[even_offset] = bf16_from_fp32(rotated_even);
    out_ptr[odd_offset] = bf16_from_fp32(rotated_odd);
}

__global__ void rope_copy_tail_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B,
    int H,
    int S,
    int Dh,
    int rope_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * S * Dh;
    if (idx >= total) {
        return;
    }

    int d = idx % Dh;
    if (d >= rope_dim) {
        output[idx] = input[idx];
    }
}

} // namespace

extern "C" int flame_rope_apply_bf16_fp32(
    const void* input,
    void* output,
    int B,
    int H,
    int S,
    int Dh,
    int rope_dim,
    float base_theta,
    int pos_offset,
    cudaStream_t stream)
{
    if (rope_dim <= 0 || rope_dim > Dh || (rope_dim % 2) != 0) {
        return 1;
    }
    const __nv_bfloat16* in_ptr = static_cast<const __nv_bfloat16*>(input);
    __nv_bfloat16* out_ptr = static_cast<__nv_bfloat16*>(output);

    int block = 256;
    int half_dim = rope_dim / 2;
    int total_pairs = B * H * S * half_dim;
    int grid = (total_pairs + block - 1) / block;
    if (grid > 0) {
        rope_apply_kernel<<<grid, block, 0, stream>>>(
            in_ptr,
            out_ptr,
            B,
            H,
            S,
            Dh,
            rope_dim,
            base_theta,
            pos_offset);
    }

    int total = B * H * S * Dh;
    int tail_grid = (total + block - 1) / block;
    if (rope_dim < Dh && tail_grid > 0) {
        rope_copy_tail_kernel<<<tail_grid, block, 0, stream>>>(
            in_ptr,
            out_ptr,
            B,
            H,
            S,
            Dh,
            rope_dim);
    }

    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

// ── Precomputed cos/sin variant ──────────────────────────────────────
// Input x: [B, H, S, D] BF16 (D = full head dim, rotation applies to all D)
// cos/sin: [1, H, S, D/2] BF16 (pre-expanded to H heads)
// Output: [B, H, S, D] BF16
// Layout: x[:, :, :, :half] * cos - x[:, :, :, half:] * sin  (first half)
//         x[:, :, :, :half] * sin + x[:, :, :, half:] * cos  (second half)

namespace {

__global__ void rope_precomputed_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos_buf,
    const __nv_bfloat16* __restrict__ sin_buf,
    __nv_bfloat16* __restrict__ out,
    int B, int H, int S, int D)
{
    int half = D / 2;
    int total = B * H * S * half;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int p = idx % half;       // pair index within head
    int tmp = idx / half;
    int s = tmp % S;
    tmp /= S;
    int h = tmp % H;
    int b = tmp / H;

    int base = ((b * H + h) * S + s) * D;
    int cs_base = (h * S + s) * half;  // cos/sin: [H, S, half] (batch dim broadcast)

    float x0 = __bfloat162float(x[base + p]);
    float x1 = __bfloat162float(x[base + half + p]);
    float c  = __bfloat162float(cos_buf[cs_base + p]);
    float sn = __bfloat162float(sin_buf[cs_base + p]);

    out[base + p]        = __float2bfloat16_rn(x0 * c - x1 * sn);
    out[base + half + p] = __float2bfloat16_rn(x0 * sn + x1 * c);
}

} // namespace

extern "C" int flame_rope_precomputed_bf16(
    const void* x,
    const void* cos_buf,
    const void* sin_buf,
    void* out,
    int B, int H, int S, int D,
    cudaStream_t stream)
{
    if (D <= 0 || (D % 2) != 0) return 1;

    int half = D / 2;
    int total = B * H * S * half;
    int block = 256;
    int grid = (total + block - 1) / block;

    rope_precomputed_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)x,
        (const __nv_bfloat16*)cos_buf,
        (const __nv_bfloat16*)sin_buf,
        (__nv_bfloat16*)out,
        B, H, S, D);

    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
