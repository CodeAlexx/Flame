// sdpa_stream_bf16.cu
// Streaming SDPA in BF16 with FP32 accumulations, without materializing [B*H,Q,K] logits.
// Algorithm: two-phase online softmax accumulation (FlashAttention-style) per (B,head_tile,Q_tile).
// Uses cuBLAS for GEMMs and small CUDA kernels for row-wise ops.
// MIT License.

#include "sdpa_stream_bf16.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(expr) do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) { \
        if (unsupported_reason && reason_buflen>0) { \
            snprintf(unsupported_reason, reason_buflen, "cuda error: %s", cudaGetErrorString(_err)); \
        } \
        return false; \
    } \
} while(0)
#endif

#ifndef CHECK_CUBLAS
#define CHECK_CUBLAS(expr) do { \
    cublasStatus_t _st = (expr); \
    if (_st != CUBLAS_STATUS_SUCCESS) { \
        if (unsupported_reason && reason_buflen>0) { \
            snprintf(unsupported_reason, reason_buflen, "cublas error: %d", int(_st)); \
        } \
        return false; \
    } \
} while(0)
#endif

// Small helpers

__global__ void ker_cast_bf16_to_f32(const __nv_bfloat16* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __bfloat162float(x[i]);
}

__global__ void ker_cast_f32_to_bf16(const float* __restrict__ x, __nv_bfloat16* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16(x[i]);
}

// Row-wise: apply scale, optional mask (1=mask), compute new row max m_ij over K_block.
// scores: [Q_t, K_b] FP32 in row-major contiguous (ld=K_b)
__global__ void ker_apply_scale_mask_rowmax(
    float* __restrict__ scores, int Q_t, int K_b, float scale,
    const __nv_bfloat16* __restrict__ mask, // may be NULL
    int64_t mask_row_stride,               // elements to jump to next row
    int64_t mask_col_stride,               // elements to jump to next col (usually 1)
    float* __restrict__ m_row              // out: [Q_t]
) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= Q_t) return;
    float m = -INFINITY;
    float* row = scores + q * K_b;
    if (!mask) {
        #pragma unroll 4
        for (int k=0;k<K_b;++k) {
            float v = row[k] * scale;
            m = fmaxf(m, v);
        }
        m_row[q] = m;
        for (int k=0;k<K_b;++k) {
            row[k] = row[k] * scale - m; // store shifted logits
        }
    } else {
        const __nv_bfloat16* mrow = mask + q*mask_row_stride;
        #pragma unroll 4
        for (int k=0;k<K_b;++k) {
            float mask_val = __bfloat162float(mrow[k*mask_col_stride]);
            float v = (mask_val > 0.5f) ? -INFINITY : (row[k] * scale);
            m = fmaxf(m, v);
        }
        m_row[q] = m;
        for (int k=0;k<K_b;++k) {
            float mask_val = __bfloat162float(mrow[k*mask_col_stride]);
            float v = (mask_val > 0.5f) ? -INFINITY : (row[k] * scale);
            row[k] = v - m;
        }
    }
}

// Row-wise: exp in place, compute row sums (sum_j e^{x_ij})
__global__ void ker_row_exp_and_sum(float* __restrict__ scores, int Q_t, int K_b, float* __restrict__ row_sums) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= Q_t) return;
    float* row = scores + q * K_b;
    float s = 0.f;
    for (int k=0;k<K_b;++k) {
        float v = row[k];
        float e = (v <= -80.f) ? 0.f : expf(v); // mild clamp
        row[k] = e;
        s += e;
    }
    row_sums[q] = s;
}

// Row-wise: update l_i and scaling factor alpha = l_prev * exp(m_prev - m_curr), and compute
// normalization factor inv_l_new = 1 / (alpha + row_sums). Also compute beta = alpha * inv_l_new.
__global__ void ker_update_l_and_factors(
    const float* __restrict__ m_prev,
    const float* __restrict__ l_prev,
    const float* __restrict__ m_curr,
    const float* __restrict__ row_sums,
    float* __restrict__ m_out,
    float* __restrict__ l_out,
    float* __restrict__ inv_l_new,
    float* __restrict__ beta,      // beta = alpha / l_new
    int Q_t
) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= Q_t) return;
    float mp = m_prev[q];
    float lp = l_prev[q];
    float mc = m_curr[q];
    float rs = row_sums[q];
    float alpha = (lp == 0.f) ? 0.f : (lp * expf(mp - mc));
    float lnew = alpha + rs;
    float inv = (lnew == 0.f) ? 0.f : (1.f / lnew);
    m_out[q] = mc;
    l_out[q] = lnew;
    inv_l_new[q] = inv;
    beta[q] = alpha * inv;
}

// Row-wise: normalize P in-place by inv_l_new (scores holds unnormalized p_tilde).
// scores: [Q_t, K_b]
__global__ void ker_row_normalize(float* __restrict__ scores, int Q_t, int K_b, const float* __restrict__ inv_l_new) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= Q_t) return;
    float* row = scores + q * K_b;
    float inv = inv_l_new[q];
    for (int k=0;k<K_b;++k) {
        row[k] *= inv;
    }
}

// Row-wise: scale existing O_accum by beta (per row scaling of each d vector)
// O_accum: [Q_t, d] FP32
__global__ void ker_scale_O_by_beta(float* __restrict__ O_accum, int Q_t, int d, const float* __restrict__ beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = Q_t * d;
    if (idx >= n) return;
    int q = idx / d;
    O_accum[idx] *= beta[q];
}

__global__ void ker_fill_constant(float* __restrict__ data, int n, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

// Compute strides for [B,H,Q_len,d] flattened pointers
static inline size_t offset_Q(int b, int h, int q, int d_idx, int H, int Q_len, int d) {
    // layout: ((b*H + h) * Q_len + q) * d + d_idx
    return (((size_t)b*H + h)*Q_len + q)*d + d_idx;
}
static inline size_t offset_KV(int b, int h, int k, int d_idx, int H, int K_len, int d) {
    return (((size_t)b*H + h)*K_len + k)*d + d_idx;
}

// Helper to write reason
static inline void write_reason(char* buf, int buflen, const char* msg) {
    if (buf && buflen > 0) {
        snprintf(buf, buflen, "%s", msg);
    }
}

bool sdpa_stream_bf16_launch(
    const void* dQv,
    const void* dKv,
    const void* dVv,
    void*       dOv,
    int B, int H, int Q_len, int K_len, int d,
    float scale,
    const __nv_bfloat16* attn_mask,
    int64_t mask_stride_ek,
    int64_t mask_stride_eq,
    int64_t mask_stride_eh,
    int64_t mask_stride_eb,
    int head_tile,
    int q_tile,
    int max_q_tile,
    void* cuda_stream_v,
    char* unsupported_reason,
    int  reason_buflen
) {
    const __nv_bfloat16* dQ = reinterpret_cast<const __nv_bfloat16*>(dQv);
    const __nv_bfloat16* dK = reinterpret_cast<const __nv_bfloat16*>(dKv);
    const __nv_bfloat16* dV = reinterpret_cast<const __nv_bfloat16*>(dVv);
    __nv_bfloat16* dO = reinterpret_cast<__nv_bfloat16*>(dOv);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream_v);

    if (d % 8 != 0) { write_reason(unsupported_reason, reason_buflen, "d not multiple of 8"); return false; }
    if (q_tile <= 0 || head_tile <= 0) { write_reason(unsupported_reason, reason_buflen, "invalid tile"); return false; }
    if (q_tile > max_q_tile) { write_reason(unsupported_reason, reason_buflen, "chunk > max_q_tile"); return false; }
    if (B <= 0 || H <= 0 || Q_len <= 0 || K_len <= 0) { write_reason(unsupported_reason, reason_buflen, "invalid shape"); return false; }

    // Use default stream if none provided
    if (stream == nullptr) stream = 0;

    // cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetStream(handle, stream));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // Workspace sizes
    const int Q_t = q_tile;
    const int K_b = K_len < 1024 ? K_len : 1024; // tuneable

    // Buffers
    float *scores = nullptr, *row_sums = nullptr;
    float *m_prev = nullptr, *l_prev = nullptr, *m_curr = nullptr, *l_curr = nullptr, *inv_l_new = nullptr, *beta = nullptr;
    float *O_accum = nullptr;
    float *scores_T = nullptr;
    __nv_bfloat16* scores_norm_bf16 = nullptr;
    CHECK_CUDA(cudaMalloc(&scores,   sizeof(float)*Q_t*K_b));
    CHECK_CUDA(cudaMalloc(&row_sums, sizeof(float)*Q_t));
    CHECK_CUDA(cudaMalloc(&m_prev,   sizeof(float)*Q_t));
    CHECK_CUDA(cudaMalloc(&l_prev,   sizeof(float)*Q_t));
    CHECK_CUDA(cudaMalloc(&m_curr,   sizeof(float)*Q_t));
    CHECK_CUDA(cudaMalloc(&l_curr,   sizeof(float)*Q_t));
    CHECK_CUDA(cudaMalloc(&inv_l_new,sizeof(float)*Q_t));
    CHECK_CUDA(cudaMalloc(&beta,     sizeof(float)*Q_t));
    CHECK_CUDA(cudaMalloc(&O_accum,  sizeof(float)*Q_t*d));
    CHECK_CUDA(cudaMalloc(&scores_T, sizeof(float)*Q_t*K_b));
    CHECK_CUDA(cudaMalloc(&scores_norm_bf16, sizeof(__nv_bfloat16)*Q_t*K_b));

    auto fill = [&](float* p, int n, float v){
        int tb = 128;
        int nb = (n + tb - 1) / tb;
        ker_fill_constant<<<nb, tb, 0, stream>>>(p, n, v);
        CHECK_CUDA(cudaGetLastError());
    };

    const float one_f = 1.0f, zero_f = 0.0f;
    auto nb1d = [](int n){ return dim3( (n+127)/128 ); };
    dim3 tb1d(128);

    auto report_gemm_failure = [&](const char* tag,
                                   cublasStatus_t status,
                                   int m, int n, int k,
                                   long long lda, long long ldb, long long ldc,
                                   int batch_idx, int head_idx, int q_offset, int k_offset) {
        if (unsupported_reason && reason_buflen > 0) {
            snprintf(
                unsupported_reason,
                reason_buflen,
                "%s gemm failed (status=%d, m=%d, n=%d, k=%d, lda=%lld, ldb=%lld, ldc=%lld, b=%d, h=%d, q0=%d, k0=%d)",
                tag,
                int(status),
                m,
                n,
                k,
                lda,
                ldb,
                ldc,
                batch_idx,
                head_idx,
                q_offset,
                k_offset);
        }
        fprintf(
            stderr,
            "sdpa_stream_bf16: %s gemm failed (status=%d, m=%d, n=%d, k=%d, lda=%lld, ldb=%lld, ldc=%lld, b=%d, h=%d, q0=%d, k0=%d)\n",
            tag,
            int(status),
            m,
            n,
            k,
            lda,
            ldb,
            ldc,
            batch_idx,
            head_idx,
            q_offset,
            k_offset);
    };

    // Main loops
    for (int b=0;b<B;++b) {
        for (int h=0; h<H; ++h) {
            for (int q0=0; q0<Q_len; q0+=Q_t) {
                int Qt = (q0+Q_t<=Q_len) ? Q_t : (Q_len - q0);

                fill(m_prev, Qt, -INFINITY);
                fill(l_prev, Qt, 0.f);
                CHECK_CUDA(cudaMemsetAsync(O_accum, 0, sizeof(float)*Qt*d, stream));

                for (int k0=0; k0<K_len; k0+=K_b) {
                    int Kb = (k0+K_b<=K_len) ? K_b : (K_len - k0);

                    // scores_colmajor = (Kb x d) * (d x Qt) -> [Kb,Qt]
                    float* scores_colmajor = scores;
                    int m = Kb, n = Qt, k = d;
                    long long lda = d, ldb = d, ldc = Kb;
                    cublasStatus_t gemm_qk = cublasGemmEx(
                        handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        m, n, k,
                        &one_f,
                        dK + offset_KV(b,h,k0,0,H,K_len,d), CUDA_R_16BF, /*lda=*/d,
                        dQ + offset_Q (b,h,q0,0,H,Q_len,d), CUDA_R_16BF, /*ldb=*/d,
                        &zero_f,
                        scores_colmajor, CUDA_R_32F, /*ldc=*/ldc,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                    if (gemm_qk != CUBLAS_STATUS_SUCCESS) {
                        report_gemm_failure("QK", gemm_qk, m, n, k, lda, ldb, ldc, b, h, q0, k0);
                        return false;
                    }

                    // Transpose to [Qt,Kb]
                    CHECK_CUDA(cudaMemcpyAsync(
                        scores_T,
                        scores_colmajor,
                        sizeof(float) * Qt * Kb,
                        cudaMemcpyDeviceToDevice,
                        stream));

                    // Mask base pointer for (b,h,q0,k0), if any
                    const __nv_bfloat16* mask_ptr = nullptr;
                    int64_t m_row_stride = 0, m_col_stride = 0;
                    if (attn_mask) {
                        mask_ptr = attn_mask
                            + b * mask_stride_eb
                            + h * mask_stride_eh
                            + q0 * mask_stride_eq
                            + k0 * mask_stride_ek;
                        m_row_stride = mask_stride_eq;
                        m_col_stride = mask_stride_ek;
                    }

                    // Apply scale/mask and get row max
                    ker_apply_scale_mask_rowmax<<<nb1d(Qt), tb1d, 0, stream>>>(
                        scores_T, Qt, Kb, scale, mask_ptr, m_row_stride, m_col_stride, m_curr);

                    // exp and sum
                    ker_row_exp_and_sum<<<nb1d(Qt), tb1d, 0, stream>>>(scores_T, Qt, Kb, row_sums);

                    // update l and factors
                    ker_update_l_and_factors<<<nb1d(Qt), tb1d, 0, stream>>>(
                        m_prev, l_prev, m_curr, row_sums, m_prev, l_prev, inv_l_new, beta, Qt);

                    // normalize
                    ker_row_normalize<<<nb1d(Qt), tb1d, 0, stream>>>(scores_T, Qt, Kb, inv_l_new);

                    int probs_elems = Qt * Kb;
                    ker_cast_f32_to_bf16<<< (probs_elems + 255) / 256, 256, 0, stream >>>(
                        scores_T,
                        scores_norm_bf16,
                        probs_elems);
                    CHECK_CUDA(cudaGetLastError());

                    // O_accum = beta * O_accum + scores_norm_bf16 @ V_block
                    ker_scale_O_by_beta<<< ((Qt*d)+255)/256, 256, 0, stream >>>(O_accum, Qt, d, beta);
                    {
                        int m2 = d, n2 = Qt, k2 = Kb;
                        long long lda2 = d, ldb2 = Kb, ldc2 = d;
                        cublasStatus_t gemm_pv = cublasGemmEx(
                            handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            m2, n2, k2,
                            &one_f,
                            dV + offset_KV(b,h,k0,0,H,K_len,d), CUDA_R_16BF, /*lda2=*/d,
                            scores_norm_bf16, CUDA_R_16BF, /*ldb2=*/Kb,
                            &one_f, // accumulate into O_accum (already scaled by beta)
                            O_accum, CUDA_R_32F, /*ldc2=*/ldc2,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                        if (gemm_pv != CUBLAS_STATUS_SUCCESS) {
                            report_gemm_failure("PV", gemm_pv, m2, n2, k2, lda2, ldb2, ldc2, b, h, q0, k0);
                            return false;
                        }
                    }
                } // k blocks

                // Cast and store to O
                {
                    int n = Qt * d;
                    ker_cast_f32_to_bf16<<< (n+255)/256, 256, 0, stream >>>(
                        O_accum, dO + offset_Q(b,h,q0,0,H,Q_len,d), n);
                }
            } // q tiles
        } // heads
    } // batch

    // Cleanup
    cudaFree(scores);
    cudaFree(row_sums);
    cudaFree(m_prev);
    cudaFree(l_prev);
    cudaFree(m_curr);
    cudaFree(l_curr);
    cudaFree(inv_l_new);
    cudaFree(beta);
    cudaFree(scores_norm_bf16);
    cudaFree(scores_T);
    cudaFree(O_accum);

    cublasDestroy(handle);
    return true;
}
