// cudnn_sdpa_bwd.cpp — cuDNN v9 Flash SDPA backward shim.
//
// Phase 2c: replaces the WMMA `flame_flash_attention_backward_bf16` kernel
// and the decomposed-recompute fallback for all training shapes the cuDNN
// frontend supports (head_dim ∈ {64, 96, 128}, unmasked full attention).
//
// Same layering as `cudnn_sdpa.cpp`:
//   - Host-only C++17, compiled via `cc::Build`, no device code here.
//   - Graph cached per (shape, per-tensor strides, scale). Offsets are
//     applied as pointer arithmetic at execute-time, not baked in.
//
// Entry point:
//
//   int flame_cudnn_sdpa_bwd_bf16(
//       const void* Q, const void* K, const void* V,
//       const void* O, const void* dO, const void* Stats,
//       void* dQ, void* dK, void* dV,
//       int B, int H, int N_q, int N_kv, int D,
//       float scale,
//       const int64_t* q_strides,  const int64_t* k_strides,
//       const int64_t* v_strides,  const int64_t* o_strides,
//       const int64_t* do_strides,
//       const int64_t* dq_strides, const int64_t* dk_strides,
//       const int64_t* dv_strides,
//       int64_t q_off,  int64_t k_off,  int64_t v_off,
//       int64_t o_off,  int64_t do_off, int64_t stats_off,
//       int64_t dq_off, int64_t dk_off, int64_t dv_off,
//       void* stream
//   );
//
// Layout: all BF16 tensors are [B, H, N, D] with caller-provided strides.
// Stats is FP32 with layout [B, H, N_q, 1] stride [H*N_q, N_q, 1, 1] —
// layout-equivalent to a contiguous [B*H, N_q] 2D tensor, which is what
// the train-forward shim in cudnn_sdpa.cpp writes.

#include <cudnn_frontend.h>
#include <cuda_runtime_api.h>
#include <cuda_bf16.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace fe = cudnn_frontend;

// Forward declaration of the handle from cudnn_sdpa.cpp. Both translation
// units are linked into `libflame_cudnn_sdpa.a`, but each TU has its own
// anonymous-namespace handle; we create our own rather than reach across.
namespace {

// Tensor UIDs. Order follows the SDPA_backward op input/output contract.
static constexpr int32_t Q_UID     = 1;
static constexpr int32_t K_UID     = 2;
static constexpr int32_t V_UID     = 3;
static constexpr int32_t O_UID     = 4;
static constexpr int32_t DO_UID    = 5;
static constexpr int32_t STATS_UID = 6;
static constexpr int32_t DQ_UID    = 7;
static constexpr int32_t DK_UID    = 8;
static constexpr int32_t DV_UID    = 9;

// Cache key: shape + scale + all 8 BF16 stride vectors. Stats stride is
// fixed by convention ([H*N_q, N_q, 1, 1]) so it's not in the key. Offsets
// are applied as pointer arithmetic, not in the graph.
struct SdpaBwdKey {
    int B;
    int H;
    int N_q;
    int N_kv;
    int D;
    float scale;
    int64_t q_s[4];
    int64_t k_s[4];
    int64_t v_s[4];
    int64_t o_s[4];
    int64_t do_s[4];
    int64_t dq_s[4];
    int64_t dk_s[4];
    int64_t dv_s[4];

    bool operator==(const SdpaBwdKey& o) const noexcept {
        if (!(B == o.B && H == o.H && N_q == o.N_q &&
              N_kv == o.N_kv && D == o.D && scale == o.scale))
            return false;
        for (int i = 0; i < 4; ++i) {
            if (q_s[i]  != o.q_s[i])  return false;
            if (k_s[i]  != o.k_s[i])  return false;
            if (v_s[i]  != o.v_s[i])  return false;
            if (o_s[i]  != o.o_s[i])  return false;
            if (do_s[i] != o.do_s[i]) return false;
            if (dq_s[i] != o.dq_s[i]) return false;
            if (dk_s[i] != o.dk_s[i]) return false;
            if (dv_s[i] != o.dv_s[i]) return false;
        }
        return true;
    }
};

struct SdpaBwdKeyHash {
    size_t operator()(const SdpaBwdKey& k) const noexcept {
        size_t h = 0xcbf29ce484222325ULL;
        auto mix = [&](uint64_t v) {
            h ^= v;
            h *= 0x100000001b3ULL;
        };
        mix((uint64_t)k.B);
        mix((uint64_t)k.H);
        mix((uint64_t)k.N_q);
        mix((uint64_t)k.N_kv);
        mix((uint64_t)k.D);
        uint32_t s_bits = 0;
        std::memcpy(&s_bits, &k.scale, sizeof(s_bits));
        mix((uint64_t)s_bits);
        for (int i = 0; i < 4; ++i) {
            mix((uint64_t)k.q_s[i]);
            mix((uint64_t)k.k_s[i]);
            mix((uint64_t)k.v_s[i]);
            mix((uint64_t)k.o_s[i]);
            mix((uint64_t)k.do_s[i]);
            mix((uint64_t)k.dq_s[i]);
            mix((uint64_t)k.dk_s[i]);
            mix((uint64_t)k.dv_s[i]);
        }
        return h;
    }
};

struct SdpaBwdEntry {
    std::shared_ptr<fe::graph::Graph> graph;
    int64_t                            workspace_size = 0;
    void*                              workspace_buf  = nullptr;
};

std::once_flag            g_handle_once;
cudnnHandle_t             g_handle        = nullptr;
std::mutex                g_cache_mutex;
std::unordered_map<SdpaBwdKey, SdpaBwdEntry, SdpaBwdKeyHash> g_cache;

int ensure_handle() {
    int ret = 0;
    std::call_once(g_handle_once, [&ret]() {
        cudnnStatus_t s = cudnnCreate(&g_handle);
        if (s != CUDNN_STATUS_SUCCESS) {
            fprintf(stderr, "[flame_cudnn_sdpa_bwd] cudnnCreate failed: %d\n", (int)s);
            ret = (int)s;
        }
    });
    if (!g_handle && ret == 0) return -999;
    return ret;
}

int build_graph(const SdpaBwdKey& k, SdpaBwdEntry& entry) {
    auto g = std::make_shared<fe::graph::Graph>();
    g->set_io_data_type(fe::DataType_t::BFLOAT16)
     .set_intermediate_data_type(fe::DataType_t::FLOAT)
     .set_compute_data_type(fe::DataType_t::FLOAT);

    const int64_t B   = k.B;
    const int64_t H   = k.H;
    const int64_t Nq  = k.N_q;
    const int64_t Nkv = k.N_kv;
    const int64_t D   = k.D;

    auto to_vec4 = [](const int64_t s[4]) {
        return std::vector<int64_t>{s[0], s[1], s[2], s[3]};
    };

    auto Q = g->tensor(fe::graph::Tensor_attributes()
                           .set_name("Q")
                           .set_uid(Q_UID)
                           .set_dim({B, H, Nq, D})
                           .set_stride(to_vec4(k.q_s)));

    auto K = g->tensor(fe::graph::Tensor_attributes()
                           .set_name("K")
                           .set_uid(K_UID)
                           .set_dim({B, H, Nkv, D})
                           .set_stride(to_vec4(k.k_s)));

    auto V = g->tensor(fe::graph::Tensor_attributes()
                           .set_name("V")
                           .set_uid(V_UID)
                           .set_dim({B, H, Nkv, D})
                           .set_stride(to_vec4(k.v_s)));

    auto O = g->tensor(fe::graph::Tensor_attributes()
                           .set_name("O")
                           .set_uid(O_UID)
                           .set_dim({B, H, Nq, D})
                           .set_stride(to_vec4(k.o_s)));

    auto dO = g->tensor(fe::graph::Tensor_attributes()
                           .set_name("dO")
                           .set_uid(DO_UID)
                           .set_dim({B, H, Nq, D})
                           .set_stride(to_vec4(k.do_s)));

    // Stats layout is fixed — matches what `flame_cudnn_sdpa_bf16_train_fwd`
    // writes. Contiguous [B*H, Nq] when viewed as 2D; stride collapses the
    // trailing size-1 dim.
    auto Stats = g->tensor(fe::graph::Tensor_attributes()
                              .set_name("Stats")
                              .set_uid(STATS_UID)
                              .set_data_type(fe::DataType_t::FLOAT)
                              .set_dim({B, H, Nq, 1})
                              .set_stride({H * Nq, Nq, 1, 1}));

    auto opts = fe::graph::SDPA_backward_attributes()
                    .set_name("flame_klein_sdpa_bwd")
                    .set_attn_scale(k.scale);
    // No causal, no bias, no dropout, no sliding window.

    auto [dQ, dK, dV] = g->sdpa_backward(Q, K, V, O, dO, Stats, opts);

    dQ->set_output(true)
      .set_dim({B, H, Nq, D})
      .set_stride(to_vec4(k.dq_s))
      .set_uid(DQ_UID);

    dK->set_output(true)
      .set_dim({B, H, Nkv, D})
      .set_stride(to_vec4(k.dk_s))
      .set_uid(DK_UID);

    dV->set_output(true)
      .set_dim({B, H, Nkv, D})
      .set_stride(to_vec4(k.dv_s))
      .set_uid(DV_UID);

    auto status = g->build(g_handle, {fe::HeurMode_t::A});
    if (!status.is_good()) {
        fprintf(stderr, "[flame_cudnn_sdpa_bwd] graph->build failed for (B=%d H=%d Nq=%d Nkv=%d D=%d): %s\n",
                k.B, k.H, k.N_q, k.N_kv, k.D, status.get_message().c_str());
        return -1;
    }

    int64_t ws = 0;
    status = g->get_workspace_size(ws);
    if (!status.is_good()) {
        fprintf(stderr, "[flame_cudnn_sdpa_bwd] get_workspace_size failed: %s\n",
                status.get_message().c_str());
        return -1;
    }

    void* ws_buf = nullptr;
    if (ws > 0) {
        cudaError_t e = cudaMalloc(&ws_buf, ws);
        if (e != cudaSuccess) {
            fprintf(stderr, "[flame_cudnn_sdpa_bwd] workspace cudaMalloc(%ld) failed: %s\n",
                    (long)ws, cudaGetErrorString(e));
            return -1;
        }
    }

    entry.graph          = g;
    entry.workspace_size = ws;
    entry.workspace_buf  = ws_buf;
    return 0;
}

} // namespace

extern "C" int flame_cudnn_sdpa_bwd_bf16(
    const void* Q, const void* K, const void* V,
    const void* O, const void* dO, const void* Stats,
    void* dQ, void* dK, void* dV,
    int B, int H, int N_q, int N_kv, int D,
    float scale,
    const int64_t* q_strides,
    const int64_t* k_strides,
    const int64_t* v_strides,
    const int64_t* o_strides,
    const int64_t* do_strides,
    const int64_t* dq_strides,
    const int64_t* dk_strides,
    const int64_t* dv_strides,
    int64_t q_offset_elems,  int64_t k_offset_elems,
    int64_t v_offset_elems,  int64_t o_offset_elems,
    int64_t do_offset_elems, int64_t stats_offset_elems,
    int64_t dq_offset_elems, int64_t dk_offset_elems,
    int64_t dv_offset_elems,
    void* stream
) {
    if (!Q || !K || !V || !O || !dO || !Stats) return -1;
    if (!dQ || !dK || !dV) return -1;
    if (B <= 0 || H <= 0 || N_q <= 0 || N_kv <= 0 || D <= 0) return -1;
    if (!q_strides || !k_strides || !v_strides || !o_strides ||
        !do_strides || !dq_strides || !dk_strides || !dv_strides) return -1;

    int rc = ensure_handle();
    if (rc != 0) return rc;

    SdpaBwdKey key{};
    key.B = B; key.H = H; key.N_q = N_q; key.N_kv = N_kv; key.D = D;
    key.scale = scale;
    for (int i = 0; i < 4; ++i) {
        key.q_s[i]  = q_strides[i];
        key.k_s[i]  = k_strides[i];
        key.v_s[i]  = v_strides[i];
        key.o_s[i]  = o_strides[i];
        key.do_s[i] = do_strides[i];
        key.dq_s[i] = dq_strides[i];
        key.dk_s[i] = dk_strides[i];
        key.dv_s[i] = dv_strides[i];
    }

    std::shared_ptr<fe::graph::Graph> graph;
    void*   ws_buf = nullptr;
    int64_t ws_sz  = 0;
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        auto it = g_cache.find(key);
        if (it == g_cache.end()) {
            SdpaBwdEntry entry{};
            rc = build_graph(key, entry);
            if (rc != 0) return rc;
            it = g_cache.emplace(key, std::move(entry)).first;
        }
        graph  = it->second.graph;
        ws_buf = it->second.workspace_buf;
        ws_sz  = it->second.workspace_size;
        (void)ws_sz;
    }

    cudnnStatus_t s = cudnnSetStream(g_handle, (cudaStream_t)stream);
    if (s != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "[flame_cudnn_sdpa_bwd] cudnnSetStream failed: %d\n", (int)s);
        return (int)s;
    }

    auto advance_bf16 = [&](const void* p, int64_t off_elems) -> void* {
        return const_cast<void*>(static_cast<const void*>(
            static_cast<const char*>(p) + off_elems * (int64_t)2));
    };
    auto advance_f32 = [&](const void* p, int64_t off_elems) -> void* {
        return const_cast<void*>(static_cast<const void*>(
            static_cast<const char*>(p) + off_elems * (int64_t)4));
    };

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> vp = {
        {Q_UID,     advance_bf16(Q,  q_offset_elems)},
        {K_UID,     advance_bf16(K,  k_offset_elems)},
        {V_UID,     advance_bf16(V,  v_offset_elems)},
        {O_UID,     advance_bf16(O,  o_offset_elems)},
        {DO_UID,    advance_bf16(dO, do_offset_elems)},
        {STATS_UID, advance_f32(Stats, stats_offset_elems)},
        {DQ_UID,    advance_bf16(dQ, dq_offset_elems)},
        {DK_UID,    advance_bf16(dK, dk_offset_elems)},
        {DV_UID,    advance_bf16(dV, dv_offset_elems)},
    };

    auto status = graph->execute(g_handle, vp, ws_buf);
    if (!status.is_good()) {
        fprintf(stderr, "[flame_cudnn_sdpa_bwd] execute failed: %s\n",
                status.get_message().c_str());
        return -1;
    }
    return 0;
}
