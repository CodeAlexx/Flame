#include "cuda_ops.h"
#include "include/flame_cuda_status.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <limits>

#define FC_RETURN_IF_ERROR(stmt) \
    do { \
        fc_status_t _status = (stmt); \
        if (_status != FC_OK) { \
            return _status; \
        } \
    } while (0)

namespace {

__global__ void im2col_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                                   __nv_bfloat16* __restrict__ output,
                                   int64_t total_rows,
                                   int64_t k_cols,
                                   int64_t H, int64_t W, int64_t C,
                                   int64_t KH, int64_t KW,
                                   int64_t Ho, int64_t Wo,
                                   int32_t stride_h, int32_t stride_w,
                                   int32_t pad_h, int32_t pad_w,
                                   int32_t dil_h, int32_t dil_w,
                                   int64_t in_stride_n,
                                   int64_t in_stride_h,
                                   int64_t in_stride_w,
                                   int64_t in_stride_c) {
    const int64_t total = total_rows * k_cols;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < total) {
        const int64_t row = idx / k_cols;
        const int64_t col = idx % k_cols;

        const int64_t ic = col % C;
        const int64_t filter_idx = col / C;
        const int64_t kw = filter_idx % KW;
        const int64_t kh = filter_idx / KW;

        const int64_t n = row / (Ho * Wo);
        const int64_t hw = row % (Ho * Wo);
        const int64_t oh = hw / Wo;
        const int64_t ow = hw % Wo;

        const int64_t ih = oh * stride_h - pad_h + kh * dil_h;
        const int64_t iw = ow * stride_w - pad_w + kw * dil_w;

        __nv_bfloat16 value = __float2bfloat16(0.0f);
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            const int64_t offset =
                n * in_stride_n +
                ih * in_stride_h +
                iw * in_stride_w +
                ic * in_stride_c;
            value = input[offset];
        }
        output[idx] = value;
        idx += static_cast<int64_t>(gridDim.x) * blockDim.x;
    }
}

inline fc_status_t check_tensor_rank(const fc_tensor_view_t* view, int32_t rank) {
    if (!view || !view->data || view->rank != rank) {
        return FC_ERR_INVALID_ARGUMENT;
    }
    for (int32_t i = 0; i < rank; ++i) {
        if (view->dims[i] < 0 || view->strides[i] <= 0) {
            return FC_ERR_INVALID_ARGUMENT;
        }
    }
    return FC_OK;
}

inline int64_t compute_out_dim(int64_t input,
                               int32_t pad,
                               int32_t dilation,
                               int64_t kernel,
                               int32_t stride) {
    const int64_t effective = (kernel - 1) * static_cast<int64_t>(dilation) + 1;
    const int64_t numer = input + 2 * static_cast<int64_t>(pad) - effective;
    if (numer < 0) {
        return 0;
    }
    return numer / stride + 1;
}

inline fc_status_t zero_tensor_async(fc_tensor_view_t* view, cudaStream_t stream) {
    const size_t nelems = static_cast<size_t>(std::max<int64_t>(view->dims[0], 0)) *
                          static_cast<size_t>(std::max<int64_t>(view->dims[1], 0)) *
                          static_cast<size_t>(std::max<int64_t>(view->dims[2], 0)) *
                          static_cast<size_t>(std::max<int64_t>(view->dims[3], 0));
    if (nelems == 0) {
        return FC_OK;
    }
    const size_t bytes = nelems * sizeof(__nv_bfloat16);
    cudaError_t err = cudaMemsetAsync(view->data, 0, bytes, stream);
    if (err != cudaSuccess) {
        return FC_ERR_LAUNCH;
    }
    return FC_OK;
}

}  // namespace

extern FlameCudaStatus flame_conv2d_nhwc_bf16_impl(const __nv_bfloat16* x,
                                                  const __nv_bfloat16* w,
                                                  const __nv_bfloat16* bias,
                                                  int N,
                                                  int H,
                                                  int W,
                                                  int Cin,
                                                  int Kh,
                                                  int Kw,
                                                  int stride_h,
                                                  int stride_w,
                                                  int pad_h,
                                                  int pad_w,
                                                  int dil_h,
                                                  int dil_w,
                                                  int Cout,
                                                  int activation,
                                                  int groups,
                                                  __nv_bfloat16* y,
                                                  void* workspace,
                                                  size_t workspace_bytes,
                                                  cudaStream_t stream);

static fc_status_t map_status(FlameCudaStatus status) {
    switch (status) {
        case FLAME_CUDA_OK:
            return FC_OK;
        case FLAME_CUDA_ERR_INVALID:
            return FC_ERR_INVALID_ARGUMENT;
        case FLAME_CUDA_ERR_UNSUPPORTED:
            return FC_ERR_UNSUPPORTED;
        case FLAME_CUDA_ERR_CUDA:
        default:
            return FC_ERR_LAUNCH;
    }
}

// Legacy entry point retained for backward compatibility. New code paths
// should call `flame_conv2d_nhwc_bf16_impl` (via the Rust FFI) directly.
extern "C" fc_status_t fc_conv2d_bf16(const fc_tensor_view_t* x,
                                      const fc_tensor_view_t* w,
                                      const fc_tensor_view_t* bias,
                                      int32_t stride_h, int32_t stride_w,
                                      int32_t pad_h, int32_t pad_w,
                                      int32_t dil_h, int32_t dil_w,
                                      fc_tensor_view_t* y,
                                      fc_workspace_t* ws,
                                      cudaStream_t stream) {
    FC_RETURN_IF_ERROR(check_tensor_rank(x, 4));
    FC_RETURN_IF_ERROR(check_tensor_rank(w, 4));
    FC_RETURN_IF_ERROR(check_tensor_rank(y, 4));
    if (!ws) {
        return FC_ERR_INVALID_ARGUMENT;
    }

    const int64_t N = x->dims[0];
    const int64_t H = x->dims[1];
    const int64_t W = x->dims[2];
    const int64_t C = x->dims[3];

    const int64_t Kh = w->dims[0];
    const int64_t Kw = w->dims[1];
    const int64_t Ic = w->dims[2];
    const int64_t Oc = w->dims[3];

    if (Ic != C) {
        return FC_ERR_INVALID_ARGUMENT;
    }

    const int64_t Ho = compute_out_dim(H, pad_h, dil_h, Kh, stride_h);
    const int64_t Wo = compute_out_dim(W, pad_w, dil_w, Kw, stride_w);
    if (Ho < 0 || Wo < 0) {
        return FC_ERR_INVALID_ARGUMENT;
    }
    if (y->dims[0] != N || y->dims[1] != Ho || y->dims[2] != Wo || y->dims[3] != Oc) {
        return FC_ERR_INVALID_ARGUMENT;
    }

    if (N == 0 || Ho == 0 || Wo == 0 || Oc == 0 || Kh == 0 || Kw == 0) {
        return zero_tensor_async(y, stream);
    }

    FlameCudaStatus status = flame_conv2d_nhwc_bf16_impl(
        static_cast<const __nv_bfloat16*>(x->data),
        static_cast<const __nv_bfloat16*>(w->data),
        bias ? static_cast<const __nv_bfloat16*>(bias->data) : nullptr,
        static_cast<int>(N),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(C),
        static_cast<int>(Kh),
        static_cast<int>(Kw),
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        static_cast<int>(Oc),
        /*activation=*/0,
        /*groups=*/1,
        static_cast<__nv_bfloat16*>(y->data),
        ws->ptr,
        ws->bytes,
        stream);

    return map_status(status);
}
