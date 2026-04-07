// fused_linear3d.cu
// Strided batched GEMM via cublasLt for 3D linear: no reshape kernels.
//
// Row-major: output[B,N,Cout] = input[B,N,Cin] @ weight[Cin,Cout] + bias[Cout]
//
// Col-major trick: C^T = B^T @ A^T
//   A (first operand in cublasLt)  = weight^T = [Cout,Cin] col-major = weight[Cin,Cout] row-major
//   B (second operand in cublasLt) = input^T  = [Cin,N]   col-major = input[N,Cin]      row-major
//   C (result in cublasLt)         = output^T = [Cout,N]  col-major = output[N,Cout]     row-major
//
// m=Cout, n=N, k=Cin
// lda=Cout (leading dim of weight in col-major), ldb=Cin (leading dim of input in col-major)
// ldc=Cout (leading dim of output in col-major)
//
// Weight is broadcast across batches (stride=0).
// Bias fused via CUBLASLT_EPILOGUE_BIAS.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

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

    // m=Cout, n=seq_len, k=Cin (col-major trick: swap operands)
    int m = out_features;
    int n = seq_len;
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

    // A = weight: [m=Cout, k=Cin] col-major, lda=Cout
    // No batch (broadcast): batch=1, stride=0
    int batchOne = 1;
    int64_t strideZero = 0;
    cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_16BF, m, k, m);
    cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchOne, sizeof(batchOne));
    cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideZero, sizeof(strideZero));

    // B = input: [k=Cin, n=seq_len] col-major, ldb=Cin
    // Batched: batch=batch_size, stride=seq_len*in_features
    int64_t strideB = (int64_t)n * k;
    cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_16BF, k, n, k);
    cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size));
    cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB));

    // C = output: [m=Cout, n=seq_len] col-major, ldc=Cout
    // Batched: batch=batch_size, stride=seq_len*out_features
    int64_t strideC = (int64_t)n * m;
    cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16BF, m, n, m);
    cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size));
    cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(strideC));

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
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    cublasStatus_t status;

    int m = out_features;
    int n = seq_len;
    int k = in_features;

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
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
        cudaDataType_t biasType = CUDA_R_16BF;
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &biasType, sizeof(biasType));
    }

    // A = weight: stored col-major as [k=Cin, m=Cout], lda = k
    // After TRANSA=T, cuBLAS sees A as [m, k].
    int batchOne = 1;
    int64_t strideZero = 0;
    cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_16BF, k, m, k);
    cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchOne, sizeof(batchOne));
    cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideZero, sizeof(strideZero));

    // B = input: [k=Cin, n=seq_len] col-major, ldb = k
    int64_t strideB = (int64_t)n * k;
    cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_16BF, k, n, k);
    cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size));
    cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB));

    // C = output: [m=Cout, n=seq_len] col-major, ldc = m
    int64_t strideC = (int64_t)n * m;
    cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16BF, m, n, m);
    cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size));
    cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(strideC));

    float alpha = 1.0f;
    float beta = 0.0f;

    status = cublasLtMatmul(
        handle, matmulDesc, &alpha,
        weight, layoutA,
        input,  layoutB,
        &beta,
        output, layoutC,
        output, layoutC,
        NULL,
        workspace, workspace_size,
        (cudaStream_t)stream
    );

    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(matmulDesc);
    return (int)status;
}

} // extern "C"
