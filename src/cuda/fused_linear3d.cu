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

} // extern "C"
