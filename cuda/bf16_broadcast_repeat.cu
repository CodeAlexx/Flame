#include "cuda_ops.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int32_t kMaxRank = 8;

struct TensorInfo {
    int64_t dims[kMaxRank];
    int64_t strides[kMaxRank];
    int32_t rank;
};

__host__ __device__ inline int64_t total_elems(const TensorInfo& info) {
    int64_t total = 1;
    for (int32_t i = 0; i < info.rank; ++i) {
        total *= info.dims[i];
    }
    return total;
}

__host__ TensorInfo make_tensor_info(const fc_tensor_view_t* view) {
    TensorInfo info{};
    info.rank = view->rank;
    for (int32_t i = 0; i < info.rank && i < kMaxRank; ++i) {
        info.dims[i] = view->dims[i];
        info.strides[i] = view->strides[i];
    }
    return info;
}

__host__ TensorInfo make_tensor_info_from_raw(const int64_t* dims,
                                              const int64_t* strides,
                                              int32_t rank) {
    TensorInfo info{};
    info.rank = rank;
    for (int32_t i = 0; i < rank && i < kMaxRank; ++i) {
        info.dims[i] = dims[i];
        info.strides[i] = strides[i];
    }
    return info;
}

__host__ TensorInfo make_contiguous_info(const int64_t* dims, int32_t rank) {
    TensorInfo info{};
    info.rank = rank;
    int64_t stride = 1;
    for (int32_t i = rank - 1; i >= 0; --i) {
        info.dims[i] = dims[i];
        info.strides[i] = stride;
        stride *= std::max<int64_t>(dims[i], 1);
    }
    return info;
}

__global__ void broadcast_kernel(const __nv_bfloat16* __restrict__ src,
                                 __nv_bfloat16* __restrict__ dst,
                                 TensorInfo src_info,
                                 TensorInfo dst_info) {
    const int64_t total = total_elems(dst_info);
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < total) {
        int64_t tmp = idx;
        int64_t src_offset = 0;
        for (int32_t d = 0; d < dst_info.rank; ++d) {
            const int64_t coord = (dst_info.dims[d] == 0)
                                      ? 0
                                      : (tmp / dst_info.strides[d]) % dst_info.dims[d];
            int64_t src_coord = coord;
            if (src_info.dims[d] == 1 || src_info.strides[d] == 0) {
                src_coord = 0;
            }
            src_offset += src_coord * src_info.strides[d];
        }
        dst[idx] = src[src_offset];
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void repeat_axis_kernel(const __nv_bfloat16* __restrict__ src,
                                   __nv_bfloat16* __restrict__ dst,
                                   TensorInfo src_info,
                                   TensorInfo dst_info,
                                   int32_t axis,
                                   int64_t repeats) {
    const int64_t total = total_elems(dst_info);
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < total) {
        int64_t tmp = idx;
        int64_t src_offset = 0;
        for (int32_t d = 0; d < dst_info.rank; ++d) {
            const int64_t coord = (dst_info.dims[d] == 0)
                                      ? 0
                                      : (tmp / dst_info.strides[d]) % dst_info.dims[d];
            int64_t src_coord = coord;
            if (d == axis) {
                src_coord /= repeats;
            }
            src_offset += src_coord * src_info.strides[d];
        }
        dst[idx] = src[src_offset];
        idx += blockDim.x * gridDim.x;
    }
}

inline void fill_contiguous_strides(fc_tensor_view_t* view) {
    int64_t stride = 1;
    for (int32_t i = view->rank - 1; i >= 0; --i) {
        view->strides[i] = stride;
        stride *= std::max<int64_t>(view->dims[i], 1);
    }
}

}  // namespace

extern "C" fc_status_t fc_bf16_broadcast(const fc_tensor_view_t* x,
                                         const int64_t* out_dims,
                                         const int64_t* out_strides,
                                         int32_t rank,
                                         fc_tensor_view_t* y,
                                         cudaStream_t stream) {
    if (!x || !y || !out_dims || !out_strides) {
        return FC_ERR_INVALID_ARGUMENT;
    }
    if (rank <= 0 || rank > kMaxRank) {
        return FC_ERR_INVALID_ARGUMENT;
    }
    if (x->rank != rank) {
        return FC_ERR_INVALID_ARGUMENT;
    }
    if (!y->data) {
        return FC_ERR_INVALID_ARGUMENT;
    }

    y->rank = rank;
    for (int32_t i = 0; i < rank; ++i) {
        if (out_dims[i] < 0) {
            return FC_ERR_INVALID_ARGUMENT;
        }
        y->dims[i] = out_dims[i];
        y->strides[i] = out_strides[i];

        const int64_t src_dim = x->dims[i];
        if (src_dim != 1 && src_dim != out_dims[i]) {
            return FC_ERR_INVALID_ARGUMENT;
        }
    }

    const TensorInfo src_info = make_tensor_info(x);
    const TensorInfo dst_info = make_tensor_info_from_raw(out_dims, out_strides, rank);

    const int64_t total = total_elems(dst_info);
    if (total == 0) {
        return FC_OK;
    }

    const int32_t threads = 256;
    const int32_t blocks = static_cast<int32_t>((total + threads - 1) / threads);

    broadcast_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x->data),
        static_cast<__nv_bfloat16*>(y->data),
        src_info,
        dst_info);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return FC_ERR_LAUNCH;
    }
    return FC_OK;
}

extern "C" fc_status_t fc_bf16_repeat_axis(const fc_tensor_view_t* x,
                                           int32_t axis,
                                           int64_t repeats,
                                           fc_tensor_view_t* y,
                                           cudaStream_t stream) {
    if (!x || !y) {
        return FC_ERR_INVALID_ARGUMENT;
    }
    if (x->rank <= 0 || x->rank > kMaxRank) {
        return FC_ERR_INVALID_ARGUMENT;
    }
    if (axis < 0 || axis >= x->rank) {
        return FC_ERR_INVALID_ARGUMENT;
    }
    if (repeats <= 0) {
        return FC_ERR_INVALID_ARGUMENT;
    }
    if (!y->data) {
        return FC_ERR_INVALID_ARGUMENT;
    }

    y->rank = x->rank;
    for (int32_t i = 0; i < x->rank; ++i) {
        y->dims[i] = x->dims[i];
    }
    if (y->dims[axis] < 0) {
        return FC_ERR_INVALID_ARGUMENT;
    }
    y->dims[axis] *= repeats;
    fill_contiguous_strides(y);

    const TensorInfo src_info = make_tensor_info(x);
    const TensorInfo dst_info = make_tensor_info(y);

    const int64_t total = total_elems(dst_info);
    if (total == 0) {
        return FC_OK;
    }

    const int32_t threads = 256;
    const int32_t blocks = static_cast<int32_t>((total + threads - 1) / threads);

    repeat_axis_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x->data),
        static_cast<__nv_bfloat16*>(y->data),
        src_info,
        dst_info,
        axis,
        repeats);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return FC_ERR_LAUNCH;
    }
    return FC_OK;
}

