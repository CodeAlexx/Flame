#include "cuda_ops.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int32_t kMaxRank = 8;

struct RepeatMeta {
    int64_t in_dims[kMaxRank];
    int64_t in_strides[kMaxRank];
    int64_t repeats[kMaxRank];
    int64_t out_strides[kMaxRank];
    int32_t rank;
    int64_t total;
};

struct RepeatNhwcMeta {
    int64_t n, h, w, c;
    int64_t rn, rh, rw, rc;
    int64_t stride_n, stride_h, stride_w, stride_c;
    int64_t total;
};

__host__ __device__ inline void fill_contiguous_strides(fc_tensor_view_t* view) {
    int64_t stride = 1;
    for (int32_t i = view->rank - 1; i >= 0; --i) {
        view->strides[i] = stride;
        stride *= std::max<int64_t>(view->dims[i], 1);
    }
}

__global__ void repeat_nd_kernel(const __nv_bfloat16* __restrict__ src,
                                 __nv_bfloat16* __restrict__ dst,
                                 RepeatMeta meta) {
    int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) *
                           static_cast<int64_t>(gridDim.x);

    while (idx < meta.total) {
        int64_t tmp = idx;
        int64_t src_offset = 0;

        for (int32_t d = 0; d < meta.rank; ++d) {
            const int64_t dim = meta.in_dims[d];
            const int64_t rep = meta.repeats[d];
            const int64_t coord =
                (dim * rep == 0)
                    ? 0
                    : (tmp / meta.out_strides[d]) % (dim * rep);
            const int64_t in_coord = coord / rep;
            src_offset += in_coord * meta.in_strides[d];
        }

        dst[idx] = src[src_offset];
        idx += stride;
    }
}

__global__ void repeat_nhwc_kernel(const __nv_bfloat16* __restrict__ src,
                                   __nv_bfloat16* __restrict__ dst,
                                   RepeatNhwcMeta meta) {
    int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) *
                           static_cast<int64_t>(gridDim.x);

    const int64_t c_out = meta.c * meta.rc;
    const int64_t w_out = meta.w * meta.rw;
    const int64_t h_out = meta.h * meta.rh;
    const int64_t n_out = meta.n * meta.rn;

    while (idx < meta.total) {
        int64_t tmp = idx;
        const int64_t c_idx = tmp % c_out;
        tmp /= c_out;
        const int64_t w_idx = tmp % w_out;
        tmp /= w_out;
        const int64_t h_idx = tmp % h_out;
        tmp /= h_out;
        const int64_t n_idx = tmp % n_out;

        const int64_t in_c = c_idx / meta.rc;
        const int64_t in_w = w_idx / meta.rw;
        const int64_t in_h = h_idx / meta.rh;
        const int64_t in_n = n_idx / meta.rn;

        const int64_t src_offset = in_n * meta.stride_n +
                                   in_h * meta.stride_h +
                                   in_w * meta.stride_w +
                                   in_c * meta.stride_c;
        dst[idx] = src[src_offset];
        idx += stride;
    }
}

inline bool is_nhwc_contiguous(const fc_tensor_view_t* x) {
    if (x->rank != 4) {
        return false;
    }
    const int64_t c = x->dims[3];
    const int64_t w = x->dims[2];
    const int64_t h = x->dims[1];

    return x->strides[3] == 1 &&
           x->strides[2] == c &&
           x->strides[1] == w * c &&
           x->strides[0] == h * w * c;
}

inline int64_t compute_total(const fc_tensor_view_t* view) {
    int64_t total = 1;
    for (int32_t i = 0; i < view->rank; ++i) {
        total *= std::max<int64_t>(view->dims[i], 1);
    }
    return total;
}

}  // namespace

extern "C" fc_status_t fc_bf16_repeat_nd(const fc_tensor_view_t* x,
                                         const int64_t* repeats,
                                         int32_t rank,
                                         fc_tensor_view_t* y,
                                         cudaStream_t stream) {
    if (!x || !y || !repeats) {
        return FC_ERR_INVALID_ARGUMENT;
    }
    if (rank <= 0 || rank > kMaxRank) {
        return FC_ERR_INVALID_ARGUMENT;
    }
    if (x->rank != rank) {
        return FC_ERR_INVALID_ARGUMENT;
    }
    if (!x->data || !y->data) {
        return FC_ERR_INVALID_ARGUMENT;
    }

    y->rank = rank;
    for (int32_t i = 0; i < rank; ++i) {
        const int64_t dim = x->dims[i];
        const int64_t rep = repeats[i];
        if (dim <= 0 || rep <= 0) {
            return FC_ERR_INVALID_ARGUMENT;
        }

        const int64_t out_dim = dim * rep;
        if (out_dim / rep != dim) {
            return FC_ERR_INVALID_ARGUMENT;
        }
        y->dims[i] = out_dim;
    }

    const int64_t total = compute_total(y);
    fill_contiguous_strides(y);

    if (total == 0) {
        return FC_OK;
    }

    const int32_t threads = 256;
    const int32_t blocks =
        static_cast<int32_t>((total + threads - 1) / threads);

    if (is_nhwc_contiguous(x)) {
        RepeatNhwcMeta meta;
        meta.n = x->dims[0];
        meta.h = x->dims[1];
        meta.w = x->dims[2];
        meta.c = x->dims[3];
        meta.rn = repeats[0];
        meta.rh = repeats[1];
        meta.rw = repeats[2];
        meta.rc = repeats[3];
        meta.stride_n = x->strides[0];
        meta.stride_h = x->strides[1];
        meta.stride_w = x->strides[2];
        meta.stride_c = x->strides[3];
        meta.total = total;

        repeat_nhwc_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x->data),
            static_cast<__nv_bfloat16*>(y->data),
            meta);
    } else {
        RepeatMeta meta{};
        meta.rank = rank;
        meta.total = total;
        int64_t out_stride = 1;

        for (int32_t i = rank - 1; i >= 0; --i) {
            meta.in_dims[i] = x->dims[i];
            meta.in_strides[i] = x->strides[i];
            meta.repeats[i] = repeats[i];
            meta.out_strides[i] = out_stride;
            out_stride *= y->dims[i];
        }

        repeat_nd_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x->data),
            static_cast<__nv_bfloat16*>(y->data),
            meta);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return FC_ERR_LAUNCH;
    }
    return FC_OK;
}
