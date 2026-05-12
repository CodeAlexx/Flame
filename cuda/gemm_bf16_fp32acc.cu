// gemm_bf16_fp32acc.cu
//
// Public symbol: `gemm_bf16_fp32acc_stridedBatched`. This is the hot path
// for BF16 row-major strided-batched GEMMs in flame-core's training stack
// (Linear forward / backward, SDPA Q@K^T and P@V via `matmul_bf16_trans`
// and `bmm_bf16_fp32acc_out`).
//
// History: the original implementation here rebuilt one MatmulDesc, three
// MatrixLayouts, and one MatmulPreference per call, AND ran
// `cublasLtMatmulAlgoGetHeuristic` per call. The heuristic call alone
// issues tens of tiny D2D copies internally; an nsys trace on Klein 9B
// showed ~150k such micro-copies per step coming from this shim. PyTorch's
// ATen caches the winning algo per (M,N,K,...) — flame-core's sibling
// shim `gemm_bf16_cublaslt.cu::lt_matmul_run` (used by `fc_gemm_bf16`)
// already does the same, and was specifically called out in the perf
// investigation as the reason flame BEATS PyTorch on Lt GEMM dispatch.
//
// This file brings the same caching to `gemm_bf16_fp32acc_stridedBatched`:
//   - Per-shape cache of (op desc + 3 layouts + preference + winning algo)
//     keyed on (m,n,k,lda,ldb,ldc,strideA,strideB,strideC,batchCount,opA,opB).
//   - Persistent per-device workspace buffer (256 MiB default, env-tunable
//     via `FLAME_LINEAR_WORKSPACE_BYTES`).
//   - Rollback via `FLAME_HANDLE_TLS_DISABLE=1` — when set, falls back to
//     the original build-everything-per-call path.
//
// Workspace: pulled from the same in-file pool that `gemm_bf16_cublaslt.cu`
// uses (matching the symbol name `acquire_workspace`). Because both files
// link into the same shared library, we MUST keep the helper here in an
// anonymous namespace so it doesn't collide with the cublaslt.cu helper.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

#define CUBLASLT_CHECK(expr)                                \
    do {                                                    \
        cublasStatus_t status = (expr);                     \
        if (status != CUBLAS_STATUS_SUCCESS) return status; \
    } while (0)

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Rollback knob: when `FLAME_HANDLE_TLS_DISABLE=1` is set, skip the cache and
// rebuild descriptors per call (the pre-caching behavior). Probed once.
// ─────────────────────────────────────────────────────────────────────────────
bool cache_disabled() {
    static bool flag = []() -> bool {
        const char* env = std::getenv("FLAME_HANDLE_TLS_DISABLE");
        return env && env[0] == '1';
    }();
    return flag;
}

// ─────────────────────────────────────────────────────────────────────────────
// Workspace cap. Defaults to 0 so the heuristic picks workspace-free algos
// (matches the pre-cache behavior of this shim — we are not changing memory
// pressure as a side-effect of the cache rewrite). Bumping to a positive
// value can unlock faster Lt algos for some shapes; left as an opt-in via
// `FLAME_GEMM_BF16_WORKSPACE_BYTES`.
//
// NOTE: cuBLASLt requires the workspace pointer to be 256-byte aligned (per
// docs) — `cudaMalloc` satisfies that.
// ─────────────────────────────────────────────────────────────────────────────
size_t workspace_cap_bytes() {
    static size_t cap = []() -> size_t {
        const char* env = std::getenv("FLAME_GEMM_BF16_WORKSPACE_BYTES");
        if (env && *env) {
            char* end = nullptr;
            unsigned long long value = std::strtoull(env, &end, 10);
            if (end != env) {
                return static_cast<size_t>(value);
            }
        }
        // Default 0 → preserves pre-cache behavior (workspace-free algos).
        return 0;
    }();
    return cap;
}

struct WorkspaceEntry {
    int device = -1;
    void* ptr = nullptr;
    size_t size = 0;
};

// Per-device, process-lifetime cuBLASLt workspace. Allocated lazily on first
// use; never freed (matches PyTorch's `getCUDABlasLtWorkspace` behavior).
void* acquire_workspace(size_t requested_bytes, size_t* granted_bytes) {
    if (requested_bytes == 0) {
        if (granted_bytes) *granted_bytes = 0;
        return nullptr;
    }

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        if (granted_bytes) *granted_bytes = 0;
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
            if (granted_bytes) *granted_bytes = 0;
            return nullptr;
        }
        entry->size = requested_bytes;
    }

    if (granted_bytes) *granted_bytes = entry->size;
    return entry->ptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-shape cache.
// ─────────────────────────────────────────────────────────────────────────────
struct GemmKey {
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
    int64_t strideA;
    int64_t strideB;
    int64_t strideC;
    int32_t batch_count;
    int32_t op_a;  // 0=N, 1=T
    int32_t op_b;

    bool operator==(const GemmKey& o) const noexcept {
        return m == o.m && n == o.n && k == o.k &&
               lda == o.lda && ldb == o.ldb && ldc == o.ldc &&
               strideA == o.strideA && strideB == o.strideB && strideC == o.strideC &&
               batch_count == o.batch_count &&
               op_a == o.op_a && op_b == o.op_b;
    }
};

struct GemmKeyHash {
    size_t operator()(const GemmKey& k) const noexcept {
        size_t h = 0xcbf29ce484222325ULL;
        auto mix = [&](uint64_t v) {
            h ^= v;
            h *= 0x100000001b3ULL;
        };
        mix((uint64_t)k.m);
        mix((uint64_t)k.n);
        mix((uint64_t)k.k);
        mix((uint64_t)k.lda);
        mix((uint64_t)k.ldb);
        mix((uint64_t)k.ldc);
        mix((uint64_t)k.strideA);
        mix((uint64_t)k.strideB);
        mix((uint64_t)k.strideC);
        mix((uint64_t)k.batch_count);
        mix((uint64_t)k.op_a);
        mix((uint64_t)k.op_b);
        return h;
    }
};

struct GemmEntry {
    cublasLtMatmulDesc_t       op         = nullptr;
    cublasLtMatrixLayout_t     a_layout   = nullptr;
    cublasLtMatrixLayout_t     b_layout   = nullptr;
    cublasLtMatrixLayout_t     c_layout   = nullptr;
    cublasLtMatmulPreference_t pref       = nullptr;
    cublasLtMatmulAlgo_t       algo;
    bool                       algo_valid = false;
};

std::mutex                                                       g_gemm_cache_mutex;
std::unordered_map<GemmKey, GemmEntry, GemmKeyHash>             g_gemm_cache;

// Uncached path (also serves as rollback when FLAME_HANDLE_TLS_DISABLE=1).
cublasStatus_t run_uncached(
    cublasLtHandle_t lt,
    cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k,
    const __nv_bfloat16* A, long long lda, long long strideA,
    const __nv_bfloat16* B, long long ldb, long long strideB,
    __nv_bfloat16* C, long long ldc, long long strideC,
    int batchCount,
    float alpha, float beta,
    void* workspace_ptr, size_t workspace_size,
    cudaStream_t stream)
{
    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatrixLayout_t aDesc = nullptr, bDesc = nullptr, cDesc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;
    cublasStatus_t stat = CUBLAS_STATUS_SUCCESS;

#define CHK(x) do { cublasStatus_t _s = (x); if (_s != CUBLAS_STATUS_SUCCESS) { stat = _s; goto CLEANUP; } } while (0)

    CHK(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CHK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    CHK(cublasLtMatrixLayoutCreate(&aDesc, CUDA_R_16BF,
        (opA == CUBLAS_OP_N) ? m : k,
        (opA == CUBLAS_OP_N) ? k : m,
        lda));
    CHK(cublasLtMatrixLayoutSetAttribute(aDesc,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    CHK(cublasLtMatrixLayoutSetAttribute(aDesc,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
    CHK(cublasLtMatrixLayoutSetAttribute(aDesc,
        CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));

    CHK(cublasLtMatrixLayoutCreate(&bDesc, CUDA_R_16BF,
        (opB == CUBLAS_OP_N) ? k : n,
        (opB == CUBLAS_OP_N) ? n : k,
        ldb));
    CHK(cublasLtMatrixLayoutSetAttribute(bDesc,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    CHK(cublasLtMatrixLayoutSetAttribute(bDesc,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
    CHK(cublasLtMatrixLayoutSetAttribute(bDesc,
        CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));

    CHK(cublasLtMatrixLayoutCreate(&cDesc, CUDA_R_16BF, m, n, ldc));
    CHK(cublasLtMatrixLayoutSetAttribute(cDesc,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    CHK(cublasLtMatrixLayoutSetAttribute(cDesc,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(strideC)));
    CHK(cublasLtMatrixLayoutSetAttribute(cDesc,
        CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));

    CHK(cublasLtMatmulPreferenceCreate(&pref));
    CHK(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));

    {
        cublasLtMatmulHeuristicResult_t heuristics[8];
        int count = 0;
        stat = cublasLtMatmulAlgoGetHeuristic(
            lt, opDesc, aDesc, bDesc, cDesc, cDesc, pref,
            8, heuristics, &count);
        if (stat != CUBLAS_STATUS_SUCCESS || count == 0) {
            stat = CUBLAS_STATUS_NOT_SUPPORTED;
            goto CLEANUP;
        }

        stat = cublasLtMatmul(
            lt, opDesc,
            &alpha,
            A, aDesc,
            B, bDesc,
            &beta,
            C, cDesc,
            C, cDesc,
            &heuristics[0].algo,
            workspace_ptr, workspace_size,
            stream);
    }

CLEANUP:
#undef CHK
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (aDesc) cublasLtMatrixLayoutDestroy(aDesc);
    if (bDesc) cublasLtMatrixLayoutDestroy(bDesc);
    if (cDesc) cublasLtMatrixLayoutDestroy(cDesc);
    if (opDesc) cublasLtMatmulDescDestroy(opDesc);
    return stat;
}

// Build a cache entry for a new shape. Returns nullptr `entry.op` on failure.
cublasStatus_t build_entry(
    cublasLtHandle_t lt,
    const GemmKey& key,
    int m, int n, int k,
    long long lda, long long strideA,
    long long ldb, long long strideB,
    long long ldc, long long strideC,
    int batchCount,
    size_t workspace_size,
    GemmEntry& entry)
{
    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatrixLayout_t aDesc = nullptr, bDesc = nullptr, cDesc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;
    cublasStatus_t stat = CUBLAS_STATUS_SUCCESS;
    cublasOperation_t opA = static_cast<cublasOperation_t>(key.op_a);
    cublasOperation_t opB = static_cast<cublasOperation_t>(key.op_b);

#define CHK(x) do { cublasStatus_t _s = (x); if (_s != CUBLAS_STATUS_SUCCESS) { stat = _s; goto FAIL; } } while (0)

    CHK(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CHK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    CHK(cublasLtMatrixLayoutCreate(&aDesc, CUDA_R_16BF,
        (opA == CUBLAS_OP_N) ? m : k,
        (opA == CUBLAS_OP_N) ? k : m,
        lda));
    CHK(cublasLtMatrixLayoutSetAttribute(aDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    CHK(cublasLtMatrixLayoutSetAttribute(aDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
    CHK(cublasLtMatrixLayoutSetAttribute(aDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));

    CHK(cublasLtMatrixLayoutCreate(&bDesc, CUDA_R_16BF,
        (opB == CUBLAS_OP_N) ? k : n,
        (opB == CUBLAS_OP_N) ? n : k,
        ldb));
    CHK(cublasLtMatrixLayoutSetAttribute(bDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    CHK(cublasLtMatrixLayoutSetAttribute(bDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
    CHK(cublasLtMatrixLayoutSetAttribute(bDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));

    CHK(cublasLtMatrixLayoutCreate(&cDesc, CUDA_R_16BF, m, n, ldc));
    CHK(cublasLtMatrixLayoutSetAttribute(cDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    CHK(cublasLtMatrixLayoutSetAttribute(cDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(strideC)));
    CHK(cublasLtMatrixLayoutSetAttribute(cDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));

    CHK(cublasLtMatmulPreferenceCreate(&pref));
    CHK(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));

    {
        cublasLtMatmulHeuristicResult_t heuristics[8];
        int count = 0;
        stat = cublasLtMatmulAlgoGetHeuristic(
            lt, opDesc, aDesc, bDesc, cDesc, cDesc, pref,
            8, heuristics, &count);
        if (stat != CUBLAS_STATUS_SUCCESS || count == 0) {
            stat = (count == 0) ? CUBLAS_STATUS_NOT_SUPPORTED : stat;
            goto FAIL;
        }
        entry.algo = heuristics[0].algo;
        entry.algo_valid = true;
    }

    entry.op       = opDesc;
    entry.a_layout = aDesc;
    entry.b_layout = bDesc;
    entry.c_layout = cDesc;
    entry.pref     = pref;
    return CUBLAS_STATUS_SUCCESS;

FAIL:
#undef CHK
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (aDesc) cublasLtMatrixLayoutDestroy(aDesc);
    if (bDesc) cublasLtMatrixLayoutDestroy(bDesc);
    if (cDesc) cublasLtMatrixLayoutDestroy(cDesc);
    if (opDesc) cublasLtMatmulDescDestroy(opDesc);
    return stat;
}

void destroy_entry(GemmEntry& e) {
    if (e.pref)     { cublasLtMatmulPreferenceDestroy(e.pref);   e.pref = nullptr; }
    if (e.a_layout) { cublasLtMatrixLayoutDestroy(e.a_layout);   e.a_layout = nullptr; }
    if (e.b_layout) { cublasLtMatrixLayoutDestroy(e.b_layout);   e.b_layout = nullptr; }
    if (e.c_layout) { cublasLtMatrixLayoutDestroy(e.c_layout);   e.c_layout = nullptr; }
    if (e.op)       { cublasLtMatmulDescDestroy(e.op);           e.op = nullptr; }
    e.algo_valid = false;
}

}  // namespace

extern "C" cublasStatus_t gemm_bf16_fp32acc_stridedBatched(
    cublasLtHandle_t lt,
    cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k,
    const __nv_bfloat16* A, long long lda, long long strideA,
    const __nv_bfloat16* B, long long ldb, long long strideB,
    __nv_bfloat16* C, long long ldc, long long strideC,
    int batchCount,
    float alpha, float beta,
    cudaStream_t stream)
{
    // Persistent per-device workspace (matches PyTorch ATen behavior).
    size_t workspace_size = workspace_cap_bytes();
    void* workspace_ptr = nullptr;
    if (workspace_size > 0) {
        workspace_ptr = acquire_workspace(workspace_size, &workspace_size);
        if (!workspace_ptr) workspace_size = 0;
    }

    if (cache_disabled()) {
        return run_uncached(
            lt, opA, opB, m, n, k,
            A, lda, strideA, B, ldb, strideB, C, ldc, strideC,
            batchCount, alpha, beta,
            workspace_ptr, workspace_size, stream);
    }

    GemmKey key{};
    key.m = m;
    key.n = n;
    key.k = k;
    key.lda = lda;
    key.ldb = ldb;
    key.ldc = ldc;
    key.strideA = strideA;
    key.strideB = strideB;
    key.strideC = strideC;
    key.batch_count = batchCount;
    key.op_a = static_cast<int32_t>(opA);
    key.op_b = static_cast<int32_t>(opB);

    GemmEntry entry;
    bool hit = false;
    {
        std::lock_guard<std::mutex> lock(g_gemm_cache_mutex);
        auto it = g_gemm_cache.find(key);
        if (it != g_gemm_cache.end() && it->second.algo_valid) {
            entry = it->second;
            hit = true;
        }
    }

    if (!hit) {
        GemmEntry fresh;
        cublasStatus_t build_stat = build_entry(
            lt, key, m, n, k,
            lda, strideA, ldb, strideB, ldc, strideC,
            batchCount, workspace_size, fresh);
        if (build_stat != CUBLAS_STATUS_SUCCESS) {
            destroy_entry(fresh);
            // Heuristic refused this shape (rare with workspace=0 advertised).
            // Fall back to the uncached path so the call still succeeds.
            return run_uncached(
                lt, opA, opB, m, n, k,
                A, lda, strideA, B, ldb, strideB, C, ldc, strideC,
                batchCount, alpha, beta,
                workspace_ptr, workspace_size, stream);
        }
        {
            std::lock_guard<std::mutex> lock(g_gemm_cache_mutex);
            auto [it, inserted] = g_gemm_cache.emplace(key, fresh);
            if (!inserted) {
                // Another thread beat us; destroy ours, take theirs.
                destroy_entry(fresh);
                entry = it->second;
            } else {
                entry = it->second;
            }
        }
    }

    cublasStatus_t launch_status = cublasLtMatmul(
        lt, entry.op,
        &alpha,
        A, entry.a_layout,
        B, entry.b_layout,
        &beta,
        C, entry.c_layout,
        C, entry.c_layout,
        &entry.algo,
        workspace_ptr, workspace_size,
        stream);

    if (launch_status == CUBLAS_STATUS_SUCCESS) {
        return CUBLAS_STATUS_SUCCESS;
    }

    // Cached algo failed (rare — e.g., workspace shrank or driver state
    // changed). Invalidate this entry and retry through the uncached path
    // so the call still succeeds.
    {
        std::lock_guard<std::mutex> lock(g_gemm_cache_mutex);
        auto it = g_gemm_cache.find(key);
        if (it != g_gemm_cache.end()) {
            destroy_entry(it->second);
            g_gemm_cache.erase(it);
        }
    }
    return run_uncached(
        lt, opA, opB, m, n, k,
        A, lda, strideA, B, ldb, strideB, C, ldc, strideC,
        batchCount, alpha, beta,
        workspace_ptr, workspace_size, stream);
}
