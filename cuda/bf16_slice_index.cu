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

__global__ void slice_copy_kernel(const __nv_bfloat16* __restrict__ src,
                                  __nv_bfloat16* __restrict__ dst,
                                  TensorInfo src_info,
                                  TensorInfo dst_info,
                                  int32_t axis,
                                  int64_t start) {
    const int64_t total = total_elems(dst_info);
    const int64_t stride_axis = dst_info.strides[axis];
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
                src_coord += start;
            }
            src_offset += src_coord * src_info.strides[d];
        }
        dst[idx] = src[src_offset];
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void index_select_kernel(const __nv_bfloat16* __restrict__ src,
                                    __nv_bfloat16* __restrict__ dst,
                                    const float* __restrict__ indices,
                                    TensorInfo src_info,
                                    TensorInfo dst_info,
                                    int32_t axis) {
    const int64_t total = total_elems(dst_info);
    const int64_t axis_stride = dst_info.strides[axis];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < total) {
        int64_t tmp = idx;
        int64_t src_offset = 0;
        int64_t axis_coord = 0;
        for (int32_t d = 0; d < dst_info.rank; ++d) {
            const int64_t coord = (dst_info.dims[d] == 0)
                                      ? 0
                                      : (tmp / dst_info.strides[d]) % dst_info.dims[d];
            if (d == axis) {
                axis_coord = coord;
                continue;
            }
            src_offset += coord * src_info.strides[d];
        }

        int64_t gather_index = static_cast<int64_t>(indices[axis_coord]);
        if (gather_index < 0) {
            gather_index = 0;
        } else if (gather_index >= src_info.dims[axis]) {
            gather_index = src_info.dims[axis] - 1;
        }
        src_offset += gather_index * src_info.strides[axis];

        dst[idx] = src[src_offset];
        idx += blockDim.x * gridDim.x;
    }
}

inline void copy_dims(const fc_tensor_view_t* src, fc_tensor_view_t* dst, int32_t axis, int64_t len) {
    dst->rank = src->rank;
    for (int32_t i = 0; i < src->rank && i < kMaxRank; ++i) {
        dst->dims[i] = src->dims[i];
        dst->strides[i] = src->strides[i];
    }
    dst->dims[axis] = len;
}

inline void make_contiguous_strides(fc_tensor_view_t* view) {
    int64_t stride = 1;
    for (int32_t i = view->rank - 1; i >= 0; --i) {
        view->strides[i] = stride;
        stride *= std::max<int64_t>(view->dims[i], 1);
    }
}

inline bool axis_is_contiguous(const fc_tensor_view_t* x, int32_t axis) {
    int64_t expected = 1;
    for (int32_t d = x->rank - 1; d > axis; --d) {
        expected *= x->dims[d];
    }
    return x->strides[axis] == expected;
}

// 2026-05-12 perf: full row-major contiguity check. Used by the slice
// fast-path: when axis==0 AND x is fully contiguous, slicing reduces to a
// single `cudaMemcpyAsync` of `len * inner_elems * sizeof(bf16)` bytes.
inline bool is_row_major_contiguous(const fc_tensor_view_t* x) {
    int64_t expected = 1;
    for (int32_t d = x->rank - 1; d >= 0; --d) {
        if (x->strides[d] != expected) return false;
        expected *= x->dims[d];
    }
    return true;
}

}  // namespace

extern "C" fc_status_t fc_bf16_slice(const fc_tensor_view_t* x,
                                     int32_t axis,
                                     int64_t start,
                                     int64_t len,
                                     fc_tensor_view_t* y_view_or_buf,
                                     cudaStream_t stream) {
    if (!x || !y_view_or_buf || x->rank <= 0 || x->rank > kMaxRank) {
        return FC_ERR_INVALID_ARGUMENT;
    }
    if (axis < 0 || axis >= x->rank) {
        return FC_ERR_INVALID_ARGUMENT;
    }
    const int64_t dim = x->dims[axis];
    if (start < 0 || len < 0 || start + len > dim) {
        return FC_ERR_INVALID_ARGUMENT;
    }

    copy_dims(x, y_view_or_buf, axis, len);

    if (len == 0) {
        make_contiguous_strides(y_view_or_buf);
        return FC_OK;
    }

    if (!y_view_or_buf->data) {
        return FC_ERR_INVALID_ARGUMENT;
    }

    make_contiguous_strides(y_view_or_buf);

    const TensorInfo src_info = make_tensor_info(x);
    const TensorInfo dst_info = make_tensor_info(y_view_or_buf);

    const int64_t total = total_elems(dst_info);
    if (total == 0) {
        return FC_OK;
    }

    // 2026-05-12 perf: leading-axis fast path. When axis==0 AND x is row-major
    // contiguous, the slice [start..start+len] along the outermost dim is a
    // single contiguous chunk of memory — replace the kernel with one
    // cudaMemcpyAsync (the kernel's div/mod-per-element work is wasted in
    // this case). Env override FLAME_SLICE_COPY_LEGACY=1 forces the kernel.
    const char* legacy_env = getenv("FLAME_SLICE_COPY_LEGACY");
    const bool force_legacy = (legacy_env != nullptr && legacy_env[0] != 0 && legacy_env[0] != '0');
    if (!force_legacy && axis == 0 && is_row_major_contiguous(x)) {
        int64_t inner = 1;
        for (int32_t d = 1; d < x->rank; ++d) inner *= x->dims[d];
        const size_t bytes = static_cast<size_t>(len) * static_cast<size_t>(inner) * sizeof(__nv_bfloat16);
        const __nv_bfloat16* src_base = static_cast<const __nv_bfloat16*>(x->data);
        const __nv_bfloat16* src_chunk = src_base + start * inner;
        cudaError_t err = cudaMemcpyAsync(y_view_or_buf->data, src_chunk, bytes,
                                          cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) {
            return FC_ERR_LAUNCH;
        }
        return FC_OK;
    }

    const int32_t threads = 256;
    const int32_t blocks = static_cast<int32_t>((total + threads - 1) / threads);

    slice_copy_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x->data),
        static_cast<__nv_bfloat16*>(y_view_or_buf->data),
        src_info,
        dst_info,
        axis,
        start);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return FC_ERR_LAUNCH;
    }
    return FC_OK;
}

extern "C" fc_status_t fc_bf16_index_select(const fc_tensor_view_t* x,
                                            int32_t axis,
                                            const float* d_indices,
                                            int64_t nidx,
                                            fc_tensor_view_t* y,
                                            cudaStream_t stream) {
    if (!x || !y || !d_indices || x->rank <= 0 || x->rank > kMaxRank) {
        return FC_ERR_INVALID_ARGUMENT;
    }
    if (axis < 0 || axis >= x->rank) {
        return FC_ERR_INVALID_ARGUMENT;
    }
    if (nidx < 0) {
        return FC_ERR_INVALID_ARGUMENT;
    }
    if (!y->data) {
        return FC_ERR_INVALID_ARGUMENT;
    }

    copy_dims(x, y, axis, nidx);
    make_contiguous_strides(y);

    const TensorInfo src_info = make_tensor_info(x);
    const TensorInfo dst_info = make_tensor_info(y);

    const int64_t total = total_elems(dst_info);
    if (total == 0) {
        return FC_OK;
    }

    const int32_t threads = 256;
    const int32_t blocks = static_cast<int32_t>((total + threads - 1) / threads);

    index_select_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x->data),
        static_cast<__nv_bfloat16*>(y->data),
        d_indices,
        src_info,
        dst_info,
        axis);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return FC_ERR_LAUNCH;
    }
    return FC_OK;
}
