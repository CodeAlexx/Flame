#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include <mutex>

#include <cutlass/numeric_types.h>

#include "namespace_config.h"
#include "flash.h"

namespace {
std::mutex g_lse_mutex;
float* g_lse_scratch = nullptr;
size_t g_lse_scratch_elems = 0;

std::mutex g_bwd_mutex;
float* g_dsoftmax_scratch = nullptr;
size_t g_dsoftmax_scratch_elems = 0;
float* g_dq_accum_scratch = nullptr;
size_t g_dq_accum_scratch_elems = 0;
uint16_t* g_dk_expanded_scratch = nullptr;
size_t g_dk_expanded_scratch_elems = 0;
uint16_t* g_dv_expanded_scratch = nullptr;
size_t g_dv_expanded_scratch_elems = 0;

bool flash_bwd_deterministic() {
    static bool flag = []() -> bool {
        const char* env = std::getenv("FLAME_PT_FLASH_BWD_DETERMINISTIC");
        return env && env[0] == '1';
    }();
    return flag;
}

int sm_count() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return 1;
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, device) != cudaSuccess) return 1;
    return props.multiProcessorCount > 0 ? props.multiProcessorCount : 1;
}

int ensure_lse_scratch(float** ptr, size_t elems) {
    if (*ptr) return 0;
    std::lock_guard<std::mutex> lock(g_lse_mutex);
    if (g_lse_scratch_elems < elems) {
        if (g_lse_scratch) {
            cudaFree(g_lse_scratch);
            g_lse_scratch = nullptr;
            g_lse_scratch_elems = 0;
        }
        cudaError_t err = cudaMalloc(&g_lse_scratch, elems * sizeof(float));
        if (err != cudaSuccess) return static_cast<int>(err);
        g_lse_scratch_elems = elems;
    }
    *ptr = g_lse_scratch;
    return 0;
}

int ensure_f32_scratch(float** scratch, size_t* scratch_elems, size_t elems) {
    if (*scratch_elems >= elems && *scratch) return 0;
    if (*scratch) {
        cudaFree(*scratch);
        *scratch = nullptr;
        *scratch_elems = 0;
    }
    cudaError_t err = cudaMalloc(scratch, elems * sizeof(float));
    if (err != cudaSuccess) return static_cast<int>(err);
    *scratch_elems = elems;
    return 0;
}

int ensure_bf16_scratch(uint16_t** scratch, size_t* scratch_elems, size_t elems) {
    if (*scratch_elems >= elems && *scratch) return 0;
    if (*scratch) {
        cudaFree(*scratch);
        *scratch = nullptr;
        *scratch_elems = 0;
    }
    cudaError_t err = cudaMalloc(scratch, elems * sizeof(uint16_t));
    if (err != cudaSuccess) return static_cast<int>(err);
    *scratch_elems = elems;
    return 0;
}

void* advance_bf16(const void* ptr, long long offset) {
    return const_cast<void*>(
        static_cast<const void*>(static_cast<const char*>(ptr) + offset * 2));
}

__global__ void sum_gqa_bhnd_kernel(
    const __nv_bfloat16* __restrict__ expanded,
    __nv_bfloat16* __restrict__ out,
    int B,
    int Hkv,
    int groups,
    int N,
    int D,
    long long out_batch_stride,
    long long out_head_stride,
    long long out_row_stride
) {
    long long total = (long long)B * Hkv * N * D;
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int d = (int)(idx % D);
    long long tmp = idx / D;
    int n = (int)(tmp % N);
    tmp /= N;
    int hkv = (int)(tmp % Hkv);
    int b = (int)(tmp / Hkv);

    float acc = 0.0f;
    int h_base = hkv * groups;
    for (int r = 0; r < groups; ++r) {
        long long src_idx =
            (((long long)b * (Hkv * groups) + (h_base + r)) * N + n) * D + d;
        acc += __bfloat162float(expanded[src_idx]);
    }
    long long out_idx =
        (long long)b * out_batch_stride + (long long)hkv * out_head_stride
        + (long long)n * out_row_stride + d;
    out[out_idx] = __float2bfloat16(acc);
}
} // namespace

extern "C" int flame_pytorch_flash_attn_bf16_hd128(
    const void* q,
    const void* k,
    const void* v,
    void* o,
    float* lse,
    int B,
    int Hq,
    int Hkv,
    int Nq,
    int Nk,
    const long long* q_strides,
    const long long* k_strides,
    const long long* v_strides,
    const long long* o_strides,
    long long q_offset_elems,
    long long k_offset_elems,
    long long v_offset_elems,
    long long o_offset_elems,
    float softmax_scale,
    int causal,
    void* stream
) {
    if (!q || !k || !v || !o) return -1;
    if (!q_strides || !k_strides || !v_strides || !o_strides) return -2;
    if (B <= 0 || Hq <= 0 || Hkv <= 0 || Nq <= 0 || Nk <= 0) return -3;
    if (Hq % Hkv != 0) return -4;
    if (softmax_scale <= 0.0f) return -5;
    int lse_status = ensure_lse_scratch(&lse, static_cast<size_t>(B) * Hq * Nq);
    if (lse_status != 0) return lse_status;

    FLASH_NAMESPACE::Flash_fwd_params params{};
    params.q_ptr = advance_bf16(q, q_offset_elems);
    params.k_ptr = advance_bf16(k, k_offset_elems);
    params.v_ptr = advance_bf16(v, v_offset_elems);
    params.o_ptr = advance_bf16(o, o_offset_elems);

    params.q_batch_stride = q_strides[0];
    params.q_head_stride = q_strides[1];
    params.q_row_stride = q_strides[2];
    params.k_batch_stride = k_strides[0];
    params.k_head_stride = k_strides[1];
    params.k_row_stride = k_strides[2];
    params.v_batch_stride = v_strides[0];
    params.v_head_stride = v_strides[1];
    params.v_row_stride = v_strides[2];
    params.o_batch_stride = o_strides[0];
    params.o_head_stride = o_strides[1];
    params.o_row_stride = o_strides[2];

    params.b = B;
    params.h = Hq;
    params.h_k = Hkv;
    params.h_h_k_ratio = Hq / Hkv;
    params.seqlen_q = Nq;
    params.seqlen_k = Nk;
    params.seqlen_knew = 0;
    params.d = 128;
    params.d_rounded = 128;
    params.seqlen_q_rounded = ((Nq + 127) / 128) * 128;
    params.seqlen_k_rounded = ((Nk + 127) / 128) * 128;
    params.rotary_dim = 0;
    params.total_q = 0;

    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = params.scale_softmax * static_cast<float>(M_LOG2E);
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = params.scale_softmax;
    params.window_size_left = -1;
    params.window_size_right = causal ? 0 : -1;
    params.softmax_lse_ptr = lse;
    params.is_bf16 = true;
    params.is_causal = causal != 0;
    params.is_seqlens_k_cumulative = true;
    params.unpadded_lse = false;
    params.seqlenq_ngroups_swapped = false;

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (causal) {
        FLASH_NAMESPACE::run_mha_fwd_<cutlass::bfloat16_t, 128, true>(params, cuda_stream);
    } else {
        FLASH_NAMESPACE::run_mha_fwd_<cutlass::bfloat16_t, 128, false>(params, cuda_stream);
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" int flame_pytorch_flash_attn_bf16_hd128_bwd(
    const void* dout,
    const void* q,
    const void* k,
    const void* v,
    const void* o,
    const float* lse,
    void* dq,
    void* dk,
    void* dv,
    int B,
    int Hq,
    int Hkv,
    int Nq,
    int Nk,
    const long long* dout_strides,
    const long long* q_strides,
    const long long* k_strides,
    const long long* v_strides,
    const long long* o_strides,
    const long long* dq_strides,
    const long long* dk_strides,
    const long long* dv_strides,
    long long dout_offset_elems,
    long long q_offset_elems,
    long long k_offset_elems,
    long long v_offset_elems,
    long long o_offset_elems,
    long long dq_offset_elems,
    long long dk_offset_elems,
    long long dv_offset_elems,
    float softmax_scale,
    int causal,
    void* stream
) {
    if (!dout || !q || !k || !v || !o || !lse || !dq || !dk || !dv) return -1;
    if (!dout_strides || !q_strides || !k_strides || !v_strides || !o_strides
        || !dq_strides || !dk_strides || !dv_strides) return -2;
    if (B <= 0 || Hq <= 0 || Hkv <= 0 || Nq <= 0 || Nk <= 0) return -3;
    if (Hq % Hkv != 0) return -4;
    if (softmax_scale <= 0.0f) return -5;
    const int groups = Hq / Hkv;
    const bool deterministic = flash_bwd_deterministic();
    const int nsplits = deterministic
        ? (sm_count() + B * Hq - 1) / (B * Hq)
        : 1;

    const int Nq_rounded = ((Nq + 127) / 128) * 128;
    const int Nk_rounded = ((Nk + 127) / 128) * 128;
    const size_t dsoftmax_elems = static_cast<size_t>(B) * Hq * Nq_rounded;
    const size_t dq_accum_split_elems = static_cast<size_t>(B) * Nq_rounded * Hq * 128;
    const size_t dq_accum_elems =
        (deterministic ? static_cast<size_t>(nsplits) : 1) * dq_accum_split_elems;
    const size_t dkv_expanded_elems = static_cast<size_t>(B) * Hq * Nk * 128;

    {
        std::lock_guard<std::mutex> lock(g_bwd_mutex);
        int status = ensure_f32_scratch(
            &g_dsoftmax_scratch, &g_dsoftmax_scratch_elems, dsoftmax_elems);
        if (status != 0) return status;
        status = ensure_f32_scratch(
            &g_dq_accum_scratch, &g_dq_accum_scratch_elems, dq_accum_elems);
        if (status != 0) return status;
        if (groups != 1) {
            status = ensure_bf16_scratch(
                &g_dk_expanded_scratch, &g_dk_expanded_scratch_elems, dkv_expanded_elems);
            if (status != 0) return status;
            status = ensure_bf16_scratch(
                &g_dv_expanded_scratch, &g_dv_expanded_scratch_elems, dkv_expanded_elems);
            if (status != 0) return status;
        }
    }
    if (deterministic) {
        cudaError_t zero_status = cudaMemsetAsync(
            g_dq_accum_scratch,
            0,
            dq_accum_elems * sizeof(float),
            reinterpret_cast<cudaStream_t>(stream));
        if (zero_status != cudaSuccess) return static_cast<int>(zero_status);
    }

    FLASH_NAMESPACE::Flash_bwd_params params{};

    params.q_ptr = advance_bf16(q, q_offset_elems);
    params.k_ptr = advance_bf16(k, k_offset_elems);
    params.v_ptr = advance_bf16(v, v_offset_elems);
    params.o_ptr = advance_bf16(o, o_offset_elems);
    params.do_ptr = advance_bf16(dout, dout_offset_elems);
    params.dq_ptr = advance_bf16(dq, dq_offset_elems);
    params.dk_ptr = groups == 1
        ? advance_bf16(dk, dk_offset_elems)
        : static_cast<void*>(g_dk_expanded_scratch);
    params.dv_ptr = groups == 1
        ? advance_bf16(dv, dv_offset_elems)
        : static_cast<void*>(g_dv_expanded_scratch);

    params.q_batch_stride = q_strides[0];
    params.q_head_stride = q_strides[1];
    params.q_row_stride = q_strides[2];
    params.k_batch_stride = k_strides[0];
    params.k_head_stride = k_strides[1];
    params.k_row_stride = k_strides[2];
    params.v_batch_stride = v_strides[0];
    params.v_head_stride = v_strides[1];
    params.v_row_stride = v_strides[2];
    params.o_batch_stride = o_strides[0];
    params.o_head_stride = o_strides[1];
    params.o_row_stride = o_strides[2];
    params.do_batch_stride = dout_strides[0];
    params.do_head_stride = dout_strides[1];
    params.do_row_stride = dout_strides[2];
    params.dq_batch_stride = dq_strides[0];
    params.dq_head_stride = dq_strides[1];
    params.dq_row_stride = dq_strides[2];
    params.dk_batch_stride = groups == 1 ? dk_strides[0] : (long long)Hq * Nk * 128;
    params.dk_head_stride = groups == 1 ? dk_strides[1] : (long long)Nk * 128;
    params.dk_row_stride = groups == 1 ? dk_strides[2] : 128;
    params.dv_batch_stride = groups == 1 ? dv_strides[0] : (long long)Hq * Nk * 128;
    params.dv_head_stride = groups == 1 ? dv_strides[1] : (long long)Nk * 128;
    params.dv_row_stride = groups == 1 ? dv_strides[2] : 128;

    params.b = B;
    params.h = Hq;
    params.h_k = Hkv;
    params.h_h_k_ratio = groups;
    params.seqlen_q = Nq;
    params.seqlen_k = Nk;
    params.seqlen_knew = 0;
    params.d = 128;
    params.d_rounded = 128;
    params.seqlen_q_rounded = Nq_rounded;
    params.seqlen_k_rounded = Nk_rounded;
    params.rotary_dim = 0;
    params.total_q = 0;

    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * static_cast<float>(M_LOG2E);
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = softmax_scale;
    params.window_size_left = -1;
    params.window_size_right = causal ? 0 : -1;
    params.softmax_lse_ptr = const_cast<float*>(lse);
    params.dsoftmax_sum = g_dsoftmax_scratch;
    params.dq_accum_ptr = g_dq_accum_scratch;
    params.dk_accum_ptr = nullptr;
    params.dv_accum_ptr = nullptr;
    params.deterministic = deterministic;
    params.dq_accum_split_stride =
        deterministic ? static_cast<long long>(dq_accum_split_elems) : 0;
    params.is_bf16 = true;
    params.is_causal = causal != 0;
    params.is_seqlens_k_cumulative = true;
    params.unpadded_lse = false;
    params.seqlenq_ngroups_swapped = false;

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (causal) {
        FLASH_NAMESPACE::run_mha_bwd_<cutlass::bfloat16_t, 128, true>(params, cuda_stream);
    } else {
        FLASH_NAMESPACE::run_mha_bwd_<cutlass::bfloat16_t, 128, false>(params, cuda_stream);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int>(err);

    if (groups != 1) {
        int threads = 256;
        long long total = static_cast<long long>(B) * Hkv * Nk * 128;
        int blocks = static_cast<int>((total + threads - 1) / threads);
        auto dk_out = reinterpret_cast<__nv_bfloat16*>(advance_bf16(dk, dk_offset_elems));
        auto dv_out = reinterpret_cast<__nv_bfloat16*>(advance_bf16(dv, dv_offset_elems));
        sum_gqa_bhnd_kernel<<<blocks, threads, 0, cuda_stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(g_dk_expanded_scratch),
            dk_out,
            B,
            Hkv,
            groups,
            Nk,
            128,
            dk_strides[0],
            dk_strides[1],
            dk_strides[2]);
        sum_gqa_bhnd_kernel<<<blocks, threads, 0, cuda_stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(g_dv_expanded_scratch),
            dv_out,
            B,
            Hkv,
            groups,
            Nk,
            128,
            dv_strides[0],
            dv_strides[1],
            dv_strides[2]);
    }
    return static_cast<int>(cudaGetLastError());
}
