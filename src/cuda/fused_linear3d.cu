// fused_linear3d.cu
// Single-call cublasLt GEMM for 3D linear: no reshape kernels.
//
// Row-major: output[B,N,Cout] = input[B,N,Cin] @ weight[Cin,Cout] + bias[Cout]
//
// Col-major trick: C^T = B^T @ A^T
//   A (first operand in cublasLt)  = weight^T = [Cout,Cin] col-major = weight[Cin,Cout] row-major
//   B (second operand in cublasLt) = input^T  = [Cin,N']   col-major = input[N',Cin]    row-major
//   C (result in cublasLt)         = output^T = [Cout,N']  col-major = output[N',Cout]  row-major
// where N' = B * N (batch folded into the sequence dim — see below).
//
// m=Cout, n=N', k=Cin
// lda=Cout (leading dim of weight in col-major), ldb=Cin (leading dim of input in col-major)
// ldc=Cout (leading dim of output in col-major)
//
// Batch handling: Linear is a per-position op and the [B, N, C] buffers are
// row-major contiguous, so [B, N, C] and [1, B*N, C] share bit-identical
// memory. We collapse the batch into the sequence dim (N' = B*N) and run as
// one non-batched GEMM. This avoids a cuBLASLt heuristic gap that returned
// CUBLAS_STATUS_INVALID_VALUE (err 7) at B>1 for the mixed-batch
// configuration (BATCH_COUNT=1 on weight, BATCH_COUNT=B on input/output).
//
// Bias fused via CUBLASLT_EPILOGUE_BIAS.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>

// Per-shape descriptor cache. Same pattern as gemm_bf16_cublaslt.cu's
// g_lt_cache: every `flame_linear3d_bf16_native` call used to build a
// fresh matmulDesc + 3 matrix layouts, then tear them down. That churn
// cost ~145k tiny (64-byte) D2D copies per Klein denoise step, enough
// to account for the entire 1.12× gap to PyTorch. Caching keyed on
// (Cout, Cin, N', has_bias, bias_stride_if_any) makes the descriptor
// setup a one-time-per-shape cost and the hot path pure
// cublasLtMatmul dispatch.
//
// BIAS_POINTER is set per-call on the cached descriptor because the
// pointer changes per invocation. Everything else (epilogue type,
// TRANSA/TRANSB, matrix layouts) is stable per shape.
//
// Workspace is NOT cached here — it comes in as a per-call argument
// from the Rust side (4 MiB allocation at every call site). That's the
// caller's concern, not the shim's.
struct LinearKey {
    int64_t m;  // Cout
    int64_t n;  // B*seq_len
    int64_t k;  // Cin
    int32_t has_bias;
};

struct LinearKeyHash {
    size_t operator()(const LinearKey& k) const noexcept {
        size_t h = 0xcbf29ce484222325ULL;
        auto mix = [&](uint64_t v) { h ^= v; h *= 0x100000001b3ULL; };
        mix((uint64_t)k.m);
        mix((uint64_t)k.n);
        mix((uint64_t)k.k);
        mix((uint64_t)k.has_bias);
        return h;
    }
};

struct LinearKeyEq {
    bool operator()(const LinearKey& a, const LinearKey& b) const noexcept {
        return a.m == b.m && a.n == b.n && a.k == b.k && a.has_bias == b.has_bias;
    }
};

struct LinearEntry {
    cublasLtMatmulDesc_t     op       = nullptr;
    cublasLtMatrixLayout_t   layoutA  = nullptr;
    cublasLtMatrixLayout_t   layoutB  = nullptr;
    cublasLtMatrixLayout_t   layoutC  = nullptr;
};

static std::mutex                                                    g_linear_cache_mutex;
static std::unordered_map<LinearKey, LinearEntry, LinearKeyHash, LinearKeyEq>
                                                                     g_linear_cache;

extern "C" {

int flame_linear3d_bf16(
    cublasLtHandle_t handle,
    const void* input,       // [B, N, Cin] BF16, row-major
    const void* weight,      // [Cin, Cout] BF16, row-major (pre-transposed)
    const void* bias,        // [Cout] BF16 or NULL
    void* output,            // [B, N, Cout] BF16, row-major
    int batch_size,
    int seq_len,
    int in_features,         // Cin
    int out_features,        // Cout
    void* workspace,
    size_t workspace_size,
    void* stream
) {
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    cublasStatus_t status;

    // Linear is a per-position op, and the [B, N, C] buffers are row-major
    // contiguous, so [B, N, C] and [1, B*N, C] are bit-identical in memory.
    // Collapse the batch into the sequence dim and run as a single non-batched
    // GEMM. This gives us BATCH_COUNT=1 on all three layouts, which sidesteps
    // the cuBLASLt heuristic gap that rejects mixed-batch (weight batch=1 vs
    // input/output batch=B) configurations for certain (M, N, K) at B>1.
    int n_eff = (batch_size > 0 ? batch_size : 1) * seq_len;

    // m=Cout, n=seq_len*batch, k=Cin (col-major trick: swap operands)
    int m = out_features;
    int n = n_eff;
    int k = in_features;

    // Create matmul descriptor: BF16 data, FP32 compute
    status = cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS) return (int)status;

    // No transpose needed: the col-major reinterpretation handles it
    cublasOperation_t opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    // Bias epilogue
    if (bias != NULL) {
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
        cudaDataType_t biasType = CUDA_R_16BF;
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &biasType, sizeof(biasType));
    }

    // All three layouts use BATCH_COUNT=1 (the batch has been folded into n).
    int batchOne = 1;
    int64_t strideZero = 0;

    // A = weight: [m=Cout, k=Cin] col-major, lda=Cout
    cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_16BF, m, k, m);
    cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchOne, sizeof(batchOne));
    cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideZero, sizeof(strideZero));

    // B = input: [k=Cin, n=B*seq_len] col-major, ldb=Cin
    cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_16BF, k, n, k);
    cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchOne, sizeof(batchOne));
    cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideZero, sizeof(strideZero));

    // C = output: [m=Cout, n=B*seq_len] col-major, ldc=Cout
    cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16BF, m, n, m);
    cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchOne, sizeof(batchOne));
    cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideZero, sizeof(strideZero));

    float alpha = 1.0f;
    float beta = 0.0f;

    // cublasLtMatmul: C = alpha * A @ B + beta * C
    // With our trick: output^T = weight^T @ input^T → output = input @ weight (row-major)
    status = cublasLtMatmul(
        handle,
        matmulDesc,
        &alpha,
        weight, layoutA,    // A = weight (first operand)
        input, layoutB,     // B = input (second operand)
        &beta,
        output, layoutC,
        output, layoutC,
        NULL,               // algo (default heuristic)
        workspace,
        workspace_size,
        (cudaStream_t)stream
    );

    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(matmulDesc);

    return (int)status;
}

// ─────────────────────────────────────────────────────────────────────────────
// flame_linear3d_bf16_native
// Same as flame_linear3d_bf16, but takes the weight in standard PyTorch
// `[Cout, Cin]` row-major layout (not pre-transposed). Uses cuBLASLt's
// TRANSA=T so the transpose happens inside the GEMM with no extra pass over
// memory. Bias is still fused via the epilogue.
//
// This eliminates the per-call `transpose2d_bf16` that the FLUX blocks were
// paying on every forward — for `single_blocks.linear1` (3072 → 21504) that
// alone was ~10–15 ms of pure memory copy per call.
//
// Layout reasoning (row-major notation throughout):
//   weight is [Cout, Cin] row-major = [Cin, Cout] col-major
//   Want: output[N, Cout] = input[N, Cin] @ weight[Cout, Cin]^T
//
// In cuBLASLt col-major terms with the standard `C^T = B^T @ A^T` trick:
//   output^T[Cout, N] = (weight as [Cout, Cin] col-major)^T_op @ input^T[Cin, N]
// The "weight as [Cout, Cin] col-major" interpretation is exactly TRANSA=T
// applied to the [Cin, Cout] col-major view of the stored buffer.
// ─────────────────────────────────────────────────────────────────────────────
int flame_linear3d_bf16_native(
    cublasLtHandle_t handle,
    const void* input,       // [B, N, Cin] BF16, row-major
    const void* weight,      // [Cout, Cin] BF16, row-major (PyTorch layout)
    const void* bias,        // [Cout] BF16 or NULL
    void* output,            // [B, N, Cout] BF16, row-major
    int batch_size,
    int seq_len,
    int in_features,         // Cin
    int out_features,        // Cout
    void* workspace,
    size_t workspace_size,
    void* stream
) {
    // Collapse batch into the sequence dim: linear is a per-position op and
    // the [B,N,C] buffer is row-major contig, so [B,N,C] === [1, B*N, C].
    int n_eff = (batch_size > 0 ? batch_size : 1) * seq_len;

    int m = out_features;
    int n = n_eff;
    int k = in_features;

    LinearKey key{};
    key.m = m;
    key.n = n;
    key.k = k;
    key.has_bias = (bias != NULL) ? 1 : 0;

    LinearEntry entry;
    bool hit = false;
    {
        std::lock_guard<std::mutex> lock(g_linear_cache_mutex);
        auto it = g_linear_cache.find(key);
        if (it != g_linear_cache.end()) {
            entry = it->second;
            hit = true;
        }
    }

    if (!hit) {
        // --- Build path: cache miss for this shape ---
        cublasLtMatmulDesc_t matmulDesc = nullptr;
        cublasLtMatrixLayout_t layoutA = nullptr, layoutB = nullptr, layoutC = nullptr;
        cublasStatus_t status;

        status = cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        if (status != CUBLAS_STATUS_SUCCESS) return (int)status;

        // Weight stored as [Cout, Cin] row-major == [Cin, Cout] col-major.
        // Set TRANSA=T so cuBLAS treats it as [Cout, Cin] col-major = [m, k].
        cublasOperation_t opT = CUBLAS_OP_T;
        cublasOperation_t opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

        if (bias != NULL) {
            cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
            cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
            // BIAS_POINTER set per-call below (pointer changes per invocation).
            cudaDataType_t biasType = CUDA_R_16BF;
            cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &biasType, sizeof(biasType));
        }

        int batchOne = 1;
        int64_t strideZero = 0;

        cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_16BF, k, m, k);
        cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchOne, sizeof(batchOne));
        cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideZero, sizeof(strideZero));

        cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_16BF, k, n, k);
        cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchOne, sizeof(batchOne));
        cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideZero, sizeof(strideZero));

        cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16BF, m, n, m);
        cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchOne, sizeof(batchOne));
        cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideZero, sizeof(strideZero));

        entry.op      = matmulDesc;
        entry.layoutA = layoutA;
        entry.layoutB = layoutB;
        entry.layoutC = layoutC;

        {
            std::lock_guard<std::mutex> lock(g_linear_cache_mutex);
            auto [it, inserted] = g_linear_cache.emplace(key, entry);
            if (!inserted) {
                // Another thread beat us; free ours, use theirs.
                if (entry.op)      cublasLtMatmulDescDestroy(entry.op);
                if (entry.layoutA) cublasLtMatrixLayoutDestroy(entry.layoutA);
                if (entry.layoutB) cublasLtMatrixLayoutDestroy(entry.layoutB);
                if (entry.layoutC) cublasLtMatrixLayoutDestroy(entry.layoutC);
                entry = it->second;
            }
        }
    }

    // Per-call: update bias pointer on the cached descriptor if bias present.
    // The pointer changes per invocation (different bias tensor); other
    // descriptor state (epilogue type, dtype) is stable and stays cached.
    if (bias != NULL) {
        cublasLtMatmulDescSetAttribute(
            entry.op,
            CUBLASLT_MATMUL_DESC_BIAS_POINTER,
            &bias,
            sizeof(bias));
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasStatus_t status = cublasLtMatmul(
        handle, entry.op, &alpha,
        weight, entry.layoutA,
        input,  entry.layoutB,
        &beta,
        output, entry.layoutC,
        output, entry.layoutC,
        NULL,
        workspace, workspace_size,
        (cudaStream_t)stream
    );
    return (int)status;
}

} // extern "C"
