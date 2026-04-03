#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>

#define CUBLASLT_CHECK(expr)                                \
    do {                                                    \
        cublasStatus_t status = (expr);                     \
        if (status != CUBLAS_STATUS_SUCCESS) return status; \
    } while (0)

static cublasStatus_t setup_matmul_desc(
    cublasLtMatmulDesc_t* desc,
    cublasOperation_t opA, cublasOperation_t opB) {
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(*desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(*desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
    return CUBLAS_STATUS_SUCCESS;
}

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
    cublasStatus_t stat = CUBLAS_STATUS_SUCCESS;
    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatrixLayout_t aDesc = nullptr, bDesc = nullptr, cDesc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    size_t workspace = 0;
    int count = 0;
    cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;

    stat = setup_matmul_desc(&opDesc, opA, opB);
    if (stat != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&aDesc, CUDA_R_16BF,
        (opA == CUBLAS_OP_N) ? m : k,
        (opA == CUBLAS_OP_N) ? k : m,
        lda));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(aDesc,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(aDesc,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(aDesc,
        CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));

    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&bDesc, CUDA_R_16BF,
        (opB == CUBLAS_OP_N) ? k : n,
        (opB == CUBLAS_OP_N) ? n : k,
        ldb));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(bDesc,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(bDesc,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(bDesc,
        CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));

    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&cDesc, CUDA_R_16BF,
        m, n, ldc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(cDesc,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(cDesc,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(strideC)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(cDesc,
        CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));

    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace)));

    cublasLtMatmulHeuristicResult_t heuristics[8];
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
        nullptr, 0,
        stream);

CLEANUP:
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (aDesc) cublasLtMatrixLayoutDestroy(aDesc);
    if (bDesc) cublasLtMatrixLayoutDestroy(bDesc);
    if (cDesc) cublasLtMatrixLayoutDestroy(cDesc);
    if (opDesc) cublasLtMatmulDescDestroy(opDesc);
    return stat;
}
