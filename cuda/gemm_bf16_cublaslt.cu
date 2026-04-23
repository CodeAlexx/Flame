#include "cuda_ops.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <limits>
#include <unordered_map>
#include <vector>

extern "C" cublasStatus_t gemm_bf16_fp32acc_stridedBatched(
    cublasLtHandle_t lt,
    cublasOperation_t opA,
    cublasOperation_t opB,
    int m,
    int n,
    int k,
    const __nv_bfloat16* A,
    long long lda,
    long long strideA,
    const __nv_bfloat16* B,
    long long ldb,
    long long strideB,
    __nv_bfloat16* C,
    long long ldc,
    long long strideC,
    int batchCount,
    float alpha,
    float beta,
    cudaStream_t stream);

namespace {

size_t linear_workspace_cap_bytes() {
    static size_t cap = []() -> size_t {
        const char* env = std::getenv("FLAME_LINEAR_WORKSPACE_BYTES");
        if (env && *env) {
            char* end = nullptr;
            unsigned long long value = std::strtoull(env, &end, 10);
            if (end != env && value > 0) {
                return static_cast<size_t>(value);
            }
        }
        // Default to 256 MiB per device/stream unless overridden.
        return 256ull * 1024ull * 1024ull;
    }();
    return cap;
}

struct WorkspaceEntry {
    int device = -1;
    void* ptr = nullptr;
    size_t size = 0;
};

void* acquire_workspace(size_t requested_bytes, size_t* granted_bytes) {
    if (requested_bytes == 0) {
        if (granted_bytes) {
            *granted_bytes = 0;
        }
        return nullptr;
    }

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        if (granted_bytes) {
            *granted_bytes = 0;
        }
        return nullptr;
    }

    static std::mutex workspace_mutex;
    static std::vector<WorkspaceEntry> workspaces;

    std::lock_guard<std::mutex> lock(workspace_mutex);
    WorkspaceEntry* entry = nullptr;
    for (auto& candidate : workspaces) {
        if (candidate.device == device) {
            entry = &candidate;
            break;
        }
    }
    if (!entry) {
        workspaces.push_back(WorkspaceEntry{device, nullptr, 0});
        entry = &workspaces.back();
    }

    if (entry->size < requested_bytes || entry->ptr == nullptr) {
        if (entry->ptr) {
            cudaFree(entry->ptr);
            entry->ptr = nullptr;
            entry->size = 0;
        }
        cudaError_t alloc_status = cudaMalloc(&entry->ptr, requested_bytes);
        if (alloc_status != cudaSuccess) {
            entry->ptr = nullptr;
            entry->size = 0;
            if (granted_bytes) {
                *granted_bytes = 0;
            }
            return nullptr;
        }
        entry->size = requested_bytes;
    }

    if (granted_bytes) {
        *granted_bytes = entry->size;
    }
    return entry->ptr;
}

fc_status_t check_tensor_2d(const fc_tensor_view_t* view) {
    if (!view || !view->data || view->rank != 2) {
        return FC_ERR_INVALID_ARGUMENT;
    }
    for (int32_t i = 0; i < view->rank; ++i) {
        if (view->dims[i] < 0 || view->strides[i] <= 0) {
            return FC_ERR_INVALID_ARGUMENT;
        }
    }
    return FC_OK;
}

fc_status_t check_tensor_3d(const fc_tensor_view_t* view) {
    if (!view || !view->data || view->rank != 3) {
        return FC_ERR_INVALID_ARGUMENT;
    }
    for (int32_t i = 0; i < view->rank; ++i) {
        if (view->dims[i] < 0 || view->strides[i] <= 0) {
            return FC_ERR_INVALID_ARGUMENT;
        }
    }
    return FC_OK;
}

fc_status_t status_from_cublas(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return FC_OK;
        case CUBLAS_STATUS_INVALID_VALUE:
            return FC_ERR_INVALID_ARGUMENT;
        case CUBLAS_STATUS_ALLOC_FAILED:
            return FC_ERR_OOM;
        case CUBLAS_STATUS_ARCH_MISMATCH:
        case CUBLAS_STATUS_NOT_SUPPORTED:
        case CUBLAS_STATUS_LICENSE_ERROR:
            return FC_ERR_UNSUPPORTED;
        default:
            return FC_ERR_LAUNCH;
    }
}

#define LT_CHECK(stmt)                                      \
    do {                                                    \
        cublasStatus_t _status = (stmt);                    \
        if (_status != CUBLAS_STATUS_SUCCESS) {             \
            return status_from_cublas(_status);             \
        }                                                   \
    } while (0)

cublasLtHandle_t get_lt_handle() {
    static cublasLtHandle_t handle = nullptr;
    static std::once_flag init_flag;
    static cublasStatus_t init_status = CUBLAS_STATUS_NOT_INITIALIZED;
    std::call_once(init_flag, []() {
        init_status = cublasLtCreate(&handle);
    });
    if (init_status != CUBLAS_STATUS_SUCCESS) {
        return nullptr;
    }
    return handle;
}

struct LtDescriptor {
    cublasLtMatmulDesc_t op = nullptr;
    LtDescriptor() = default;
    ~LtDescriptor() {
        if (op) {
            cublasLtMatmulDescDestroy(op);
        }
    }
};

struct LtLayout {
    cublasLtMatrixLayout_t layout = nullptr;
    LtLayout() = default;
    ~LtLayout() {
        if (layout) {
            cublasLtMatrixLayoutDestroy(layout);
        }
    }
};

struct LtPreference {
    cublasLtMatmulPreference_t pref = nullptr;
    LtPreference() = default;
    ~LtPreference() {
        if (pref) {
            cublasLtMatmulPreferenceDestroy(pref);
        }
    }
};

// Per-shape cache. Without this, every GEMM rebuilds 5 descriptors and
// runs cublasLtMatmulAlgoGetHeuristic (which internally issues tens of
// 64-byte D2D copies for algorithm-descriptor propagation). nsys on Klein
// 9B showed 147k of those tiny copies per step — more than the entire
// gap to PyTorch. PyTorch's ATen caches the winning algo per (M,N,K);
// this is the same treatment.
//
// The raw handles (op/layouts/pref) are leaked on process exit. That's
// fine — there's one entry per unique shape, bounded by the model's
// kernel roster. Each entry is ~few hundred bytes of host memory.
struct LtMatmulKey {
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t stride_a;
    int64_t stride_b;
    int64_t stride_c;
    int64_t bias_stride;
    int32_t batch_count;
    int32_t has_bias;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;

    bool operator==(const LtMatmulKey& o) const noexcept {
        return m == o.m && n == o.n && k == o.k &&
               stride_a == o.stride_a && stride_b == o.stride_b &&
               stride_c == o.stride_c && bias_stride == o.bias_stride &&
               batch_count == o.batch_count && has_bias == o.has_bias &&
               lda == o.lda && ldb == o.ldb && ldc == o.ldc;
    }
};

struct LtMatmulKeyHash {
    size_t operator()(const LtMatmulKey& k) const noexcept {
        size_t h = 0xcbf29ce484222325ULL;
        auto mix = [&](uint64_t v) {
            h ^= v;
            h *= 0x100000001b3ULL;
        };
        mix((uint64_t)k.m);
        mix((uint64_t)k.n);
        mix((uint64_t)k.k);
        mix((uint64_t)k.stride_a);
        mix((uint64_t)k.stride_b);
        mix((uint64_t)k.stride_c);
        mix((uint64_t)k.bias_stride);
        mix((uint64_t)k.batch_count);
        mix((uint64_t)k.has_bias);
        mix((uint64_t)k.lda);
        mix((uint64_t)k.ldb);
        mix((uint64_t)k.ldc);
        return h;
    }
};

struct LtMatmulEntry {
    cublasLtMatmulDesc_t     op         = nullptr;
    cublasLtMatrixLayout_t   a_layout   = nullptr;
    cublasLtMatrixLayout_t   b_layout   = nullptr;
    cublasLtMatrixLayout_t   c_layout   = nullptr;
    cublasLtMatmulPreference_t pref     = nullptr;
    cublasLtMatmulAlgo_t     algo;        // Winning algo from heuristic.
    size_t                   workspace_bytes = 0;
    bool                     algo_valid = false;
};

std::mutex g_lt_cache_mutex;
std::unordered_map<LtMatmulKey, LtMatmulEntry, LtMatmulKeyHash> g_lt_cache;

fc_status_t lt_matmul_run(const fc_tensor_view_t* a,
                          const fc_tensor_view_t* b,
                          const fc_tensor_view_t* bias,
                          fc_tensor_view_t* c,
                          int32_t batch_count,
                          const int64_t stride_a,
                          const int64_t stride_b,
                          const int64_t stride_c,
                          const int64_t bias_stride,
                          cudaStream_t stream) {
    cublasLtHandle_t lt = get_lt_handle();
    if (lt == nullptr) {
        return FC_ERR_LAUNCH;
    }

    const bool has_bias = bias && bias->data;

    const int64_t m64 = a->dims[a->rank - 2];
    const int64_t k64 = a->dims[a->rank - 1];
    const int64_t n64 = b->dims[b->rank - 1];
    if (m64 < 0 || k64 < 0 || n64 < 0 ||
        m64 > std::numeric_limits<int32_t>::max() ||
        k64 > std::numeric_limits<int32_t>::max() ||
        n64 > std::numeric_limits<int32_t>::max()) {
        return FC_ERR_INVALID_ARGUMENT;
    }

    const int64_t lda = static_cast<int64_t>(a->strides[a->rank - 2]);
    const int64_t ldb = static_cast<int64_t>(b->strides[b->rank - 2]);
    const int64_t ldc = static_cast<int64_t>(c->strides[c->rank - 2]);

    LtMatmulKey key{};
    key.m = m64;
    key.n = n64;
    key.k = k64;
    key.stride_a = stride_a;
    key.stride_b = stride_b;
    key.stride_c = stride_c;
    key.bias_stride = bias_stride;
    key.batch_count = batch_count;
    key.has_bias = has_bias ? 1 : 0;
    key.lda = lda;
    key.ldb = ldb;
    key.ldc = ldc;

    // Workspace: pulled from the same global pool the original path used.
    // Sized once (per cap_bytes); reused across all cache entries.
    size_t workspace_bytes = linear_workspace_cap_bytes();
    void* workspace_ptr = nullptr;
    if (workspace_bytes > 0) {
        workspace_ptr = acquire_workspace(workspace_bytes, &workspace_bytes);
        if (!workspace_ptr) {
            workspace_bytes = 0;
        }
    }

    // Cache hit → skip all descriptor/heuristic work.
    // Cache miss → build descriptors, run heuristic once, store the winner.
    LtMatmulEntry entry;
    {
        std::lock_guard<std::mutex> lock(g_lt_cache_mutex);
        auto it = g_lt_cache.find(key);
        if (it != g_lt_cache.end()) {
            entry = it->second;
        }
    }

    if (!entry.algo_valid) {
        // --- Build path: cache miss for this shape ---
        LtDescriptor op_desc_holder;
        LT_CHECK(cublasLtMatmulDescCreate(&op_desc_holder.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));

        cublasOperation_t op_n = CUBLAS_OP_N;
        LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc_holder.op, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n)));
        LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc_holder.op, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));

        if (has_bias) {
            cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
            LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc_holder.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
            // Note: BIAS_POINTER is set per-call below (pointer changes per
            // invocation), not here. Only the epilogue type is baked into
            // the cached descriptor.
            cublasDataType_t bias_type = CUDA_R_16BF;
            LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc_holder.op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type)));
            if (batch_count > 1 && bias_stride > 0) {
                LT_CHECK(cublasLtMatmulDescSetAttribute(
                    op_desc_holder.op,
                    CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE,
                    &bias_stride,
                    sizeof(bias_stride)));
            }
        }

        LtLayout a_layout_holder;
        LtLayout b_layout_holder;
        LtLayout c_layout_holder;

        const int32_t m = static_cast<int32_t>(m64);
        const int32_t k = static_cast<int32_t>(k64);
        const int32_t n = static_cast<int32_t>(n64);

        LT_CHECK(cublasLtMatrixLayoutCreate(&a_layout_holder.layout, CUDA_R_16BF, static_cast<uint64_t>(m), static_cast<uint64_t>(k), lda));
        LT_CHECK(cublasLtMatrixLayoutCreate(&b_layout_holder.layout, CUDA_R_16BF, static_cast<uint64_t>(k), static_cast<uint64_t>(n), ldb));
        LT_CHECK(cublasLtMatrixLayoutCreate(&c_layout_holder.layout, CUDA_R_16BF, static_cast<uint64_t>(m), static_cast<uint64_t>(n), ldc));

        cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
        LT_CHECK(cublasLtMatrixLayoutSetAttribute(a_layout_holder.layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
        LT_CHECK(cublasLtMatrixLayoutSetAttribute(b_layout_holder.layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
        LT_CHECK(cublasLtMatrixLayoutSetAttribute(c_layout_holder.layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

        if (batch_count > 1) {
            LT_CHECK(cublasLtMatrixLayoutSetAttribute(a_layout_holder.layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
            LT_CHECK(cublasLtMatrixLayoutSetAttribute(b_layout_holder.layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
            LT_CHECK(cublasLtMatrixLayoutSetAttribute(c_layout_holder.layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));

            LT_CHECK(cublasLtMatrixLayoutSetAttribute(a_layout_holder.layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a, sizeof(stride_a)));
            LT_CHECK(cublasLtMatrixLayoutSetAttribute(b_layout_holder.layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b, sizeof(stride_b)));
            LT_CHECK(cublasLtMatrixLayoutSetAttribute(c_layout_holder.layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c, sizeof(stride_c)));
        }

        LtPreference pref_holder;
        LT_CHECK(cublasLtMatmulPreferenceCreate(&pref_holder.pref));
        LT_CHECK(cublasLtMatmulPreferenceSetAttribute(
            pref_holder.pref,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_bytes,
            sizeof(workspace_bytes)));

        cublasLtMatmulHeuristicResult_t results[8];
        int returned_results = 0;
        cublasStatus_t stat = cublasLtMatmulAlgoGetHeuristic(
            lt,
            op_desc_holder.op,
            a_layout_holder.layout,
            b_layout_holder.layout,
            c_layout_holder.layout,
            c_layout_holder.layout,
            pref_holder.pref,
            sizeof(results) / sizeof(results[0]),
            results,
            &returned_results);
        if (stat != CUBLAS_STATUS_SUCCESS || returned_results == 0) {
            return FC_ERR_UNSUPPORTED;
        }

        // Release ownership from RAII wrappers — these now live in the cache.
        entry.op         = op_desc_holder.op;     op_desc_holder.op = nullptr;
        entry.a_layout   = a_layout_holder.layout; a_layout_holder.layout = nullptr;
        entry.b_layout   = b_layout_holder.layout; b_layout_holder.layout = nullptr;
        entry.c_layout   = c_layout_holder.layout; c_layout_holder.layout = nullptr;
        entry.pref       = pref_holder.pref;      pref_holder.pref = nullptr;
        entry.algo       = results[0].algo;       // first heuristic result wins
        entry.workspace_bytes = workspace_bytes;
        entry.algo_valid = true;

        {
            std::lock_guard<std::mutex> lock(g_lt_cache_mutex);
            // Another thread may have inserted this key between our lookup
            // and build — in that case, destroy our just-built handles and
            // use theirs. Avoids leaking.
            auto [it, inserted] = g_lt_cache.emplace(key, entry);
            if (!inserted) {
                if (entry.op)       cublasLtMatmulDescDestroy(entry.op);
                if (entry.a_layout) cublasLtMatrixLayoutDestroy(entry.a_layout);
                if (entry.b_layout) cublasLtMatrixLayoutDestroy(entry.b_layout);
                if (entry.c_layout) cublasLtMatrixLayoutDestroy(entry.c_layout);
                if (entry.pref)     cublasLtMatmulPreferenceDestroy(entry.pref);
                entry = it->second;
            }
        }
    }

    // Per-call-only: if bias is present, update the BIAS_POINTER on the
    // cached descriptor. The pointer changes per invocation (different
    // bias tensor), but the other descriptor state (epilogue type, dtype,
    // batch stride) is stable and stays cached.
    if (has_bias) {
        const void* bias_ptr = bias->data;
        LT_CHECK(cublasLtMatmulDescSetAttribute(
            entry.op,
            CUBLASLT_MATMUL_DESC_BIAS_POINTER,
            &bias_ptr,
            sizeof(bias_ptr)));
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const void* a_ptr = a->data;
    const void* b_ptr = b->data;
    void* c_ptr = c->data;

    cublasStatus_t launch_status = cublasLtMatmul(
        lt,
        entry.op,
        &alpha,
        a_ptr,
        entry.a_layout,
        b_ptr,
        entry.b_layout,
        &beta,
        c_ptr,
        entry.c_layout,
        c_ptr,
        entry.c_layout,
        &entry.algo,
        workspace_ptr,
        workspace_bytes,
        stream);
    if (launch_status == CUBLAS_STATUS_SUCCESS) {
        return FC_OK;
    }

    // Cached algo failed — likely the workspace shrank or something
    // hardware-level changed. Invalidate the cache entry and fall through
    // to re-heuristic on the next call.
    {
        std::lock_guard<std::mutex> lock(g_lt_cache_mutex);
        auto it = g_lt_cache.find(key);
        if (it != g_lt_cache.end()) {
            if (it->second.op)       cublasLtMatmulDescDestroy(it->second.op);
            if (it->second.a_layout) cublasLtMatrixLayoutDestroy(it->second.a_layout);
            if (it->second.b_layout) cublasLtMatrixLayoutDestroy(it->second.b_layout);
            if (it->second.c_layout) cublasLtMatrixLayoutDestroy(it->second.c_layout);
            if (it->second.pref)     cublasLtMatmulPreferenceDestroy(it->second.pref);
            g_lt_cache.erase(it);
        }
    }
    return status_from_cublas(launch_status);
}

}  // namespace

extern "C" fc_status_t fc_gemm_bf16(const fc_tensor_view_t* x,
                                    const fc_tensor_view_t* w,
                                    const fc_tensor_view_t* bias,
                                    fc_tensor_view_t* y,
                                    cudaStream_t stream) {
    bool force_fallback = false;
    if (const char* env = std::getenv("FLAME_CUBLASLT_FORCE_FALLBACK")) {
        if (env[0] != '\0' &&
            env[0] != '0' &&
            env[0] != 'f' && env[0] != 'F' &&
            env[0] != 'n' && env[0] != 'N') {
            force_fallback = true;
        }
    }

    fc_status_t st = check_tensor_2d(x);
    if (st != FC_OK) {
        return st;
    }
    st = check_tensor_2d(w);
    if (st != FC_OK) {
        return st;
    }
    st = check_tensor_2d(y);
    if (st != FC_OK) {
        return st;
    }
    const int64_t m = x->dims[0];
    const int64_t k = x->dims[1];
    if (w->dims[0] != k) {
        return FC_ERR_INVALID_ARGUMENT;
    }
    const int64_t n = w->dims[1];
    if (y->dims[0] != m || y->dims[1] != n) {
        return FC_ERR_INVALID_ARGUMENT;
    }
    if (m == 0 || n == 0 || k == 0) {
        const size_t elements = static_cast<size_t>(m) * static_cast<size_t>(n);
        if (elements > 0) {
            cudaError_t err = cudaMemsetAsync(y->data, 0, elements * sizeof(__nv_bfloat16), stream);
            if (err != cudaSuccess) {
                return FC_ERR_LAUNCH;
            }
        }
        return FC_OK;
    }

    int64_t bias_stride = 0;
    if (bias && bias->data) {
        if (bias->rank == 1) {
            if (bias->dims[0] != n) {
                return FC_ERR_INVALID_ARGUMENT;
            }
            if (bias->strides[0] <= 0) {
                return FC_ERR_INVALID_ARGUMENT;
            }
        } else if (bias->rank == 2) {
            if (bias->dims[0] != 1 || bias->dims[1] != n) {
                return FC_ERR_INVALID_ARGUMENT;
            }
            if (bias->strides[0] <= 0 || bias->strides[1] <= 0) {
                return FC_ERR_INVALID_ARGUMENT;
            }
            bias_stride = bias->strides[0];
        } else {
            return FC_ERR_INVALID_ARGUMENT;
        }
    }

    fc_status_t lt_status = FC_ERR_UNSUPPORTED;
    if (!force_fallback) {
        lt_status = lt_matmul_run(
            x,
            w,
            bias,
            y,
            /*batch_count=*/1,
            /*stride_a=*/0,
            /*stride_b=*/0,
            /*stride_c=*/0,
            bias_stride,
            stream);

        if (lt_status == FC_OK || lt_status != FC_ERR_UNSUPPORTED) {
            return lt_status;
        }
    }

    cublasLtHandle_t lt = get_lt_handle();
    if (lt == nullptr) {
        return FC_ERR_LAUNCH;
    }

    if (m <= 0 || n <= 0 || k <= 0 ||
        m > std::numeric_limits<int>::max() ||
        n > std::numeric_limits<int>::max() ||
        k > std::numeric_limits<int>::max()) {
        return FC_ERR_INVALID_ARGUMENT;
    }

    const long long lda = x->strides[0];
    const long long ldb = w->strides[0];
    const long long ldc = y->strides[0];

    cublasStatus_t fallback = gemm_bf16_fp32acc_stridedBatched(
        lt,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        static_cast<const __nv_bfloat16*>(x->data),
        lda,
        0,
        static_cast<const __nv_bfloat16*>(w->data),
        ldb,
        0,
        static_cast<__nv_bfloat16*>(y->data),
        ldc,
        0,
        1,
        1.0f,
        0.0f,
        stream);

    if (fallback != CUBLAS_STATUS_SUCCESS) {
        return status_from_cublas(fallback);
    }

    if (bias && bias->data) {
        fc_tensor_view_t bias_view = *bias;
        if (bias_view.rank == 2) {
            bias_view.rank = 1;
            bias_view.dims[0] = bias->dims[1];
            bias_view.strides[0] = bias->strides[1];
        }
        if (bias_view.strides[0] <= 0 || y->strides[1] <= 0) {
            return FC_ERR_INVALID_ARGUMENT;
        }

        fc_status_t add_status = FC_OK;
        for (int64_t row = 0; row < m; ++row) {
            fc_tensor_view_t row_view{};
            row_view.rank = 1;
            row_view.data = static_cast<void*>(
                static_cast<__nv_bfloat16*>(y->data) + row * y->strides[0]);
            row_view.dims[0] = n;
            row_view.strides[0] = y->strides[1];
            add_status = fc_axpby_bf16(&bias_view, 1.0f, &row_view, 1.0f, stream);
            if (add_status != FC_OK) {
                break;
            }
        }
        if (add_status != FC_OK) {
            return add_status;
        }
    }

    return FC_STATUS_LT_FALLBACK;
}

extern "C" fc_status_t fc_batched_gemm_bf16(const fc_tensor_view_t* a,
                                            const fc_tensor_view_t* b,
                                            const fc_tensor_view_t* bias,
                                            fc_tensor_view_t* c,
                                            cudaStream_t stream) {
    fc_status_t st = check_tensor_3d(a);
    if (st != FC_OK) {
        return st;
    }
    st = check_tensor_3d(b);
    if (st != FC_OK) {
        return st;
    }
    st = check_tensor_3d(c);
    if (st != FC_OK) {
        return st;
    }

    const int64_t batch = a->dims[0];
    const int64_t m = a->dims[1];
    const int64_t k = a->dims[2];

    if (b->dims[0] != batch || b->dims[1] != k) {
        return FC_ERR_INVALID_ARGUMENT;
    }
    const int64_t n = b->dims[2];
    if (c->dims[0] != batch || c->dims[1] != m || c->dims[2] != n) {
        return FC_ERR_INVALID_ARGUMENT;
    }

    if (batch == 0 || m == 0 || n == 0 || k == 0) {
        const size_t elements = static_cast<size_t>(batch) * static_cast<size_t>(m) * static_cast<size_t>(n);
        if (elements > 0) {
            cudaError_t err = cudaMemsetAsync(c->data, 0, elements * sizeof(__nv_bfloat16), stream);
            if (err != cudaSuccess) {
                return FC_ERR_LAUNCH;
            }
        }
        return FC_OK;
    }

    if (batch > std::numeric_limits<int32_t>::max()) {
        return FC_ERR_INVALID_ARGUMENT;
    }

    int64_t bias_stride = 0;
    if (bias && bias->data) {
        if (bias->rank == 1) {
            if (bias->dims[0] != n) {
                return FC_ERR_INVALID_ARGUMENT;
            }
            if (bias->strides[0] <= 0) {
                return FC_ERR_INVALID_ARGUMENT;
            }
        } else if (bias->rank == 2) {
            if (bias->dims[0] != batch || bias->dims[1] != n) {
                return FC_ERR_INVALID_ARGUMENT;
            }
            if (bias->strides[0] <= 0 || bias->strides[1] <= 0) {
                return FC_ERR_INVALID_ARGUMENT;
            }
            bias_stride = bias->strides[0];
        } else {
            return FC_ERR_INVALID_ARGUMENT;
        }
    }

    const int64_t stride_a = a->strides[0];
    const int64_t stride_b = b->strides[0];
    const int64_t stride_c = c->strides[0];

    return lt_matmul_run(
        a,
        b,
        bias,
        c,
        static_cast<int32_t>(batch),
        stride_a,
        stride_b,
        stride_c,
        bias_stride,
        stream);
}
