// cudnn_sdpa.cpp — real cuDNN v9 Flash SDPA shim for flame-core.
//
// Replaces the deleted stub `src/cudnn/attention.rs` that never touched cuDNN.
// This file is compiled as host-only C++17 via `cc::Build` (see build.rs); it
// does NOT participate in the `-rdc=true` device-link pipeline used for the
// `.cu` kernels, because there's no device code here — only host-side cuDNN
// graph API calls.
//
// Entry point exposed to Rust:
//
//   int flame_cudnn_sdpa_bf16(
//       const void* Q, const void* K, const void* V, void* O,
//       int B, int H, int N_q, int N_kv, int D,
//       float scale, void* stream
//   );
//
// Layout: Q,K,V,O are [B, H, N, D] contiguous BF16 on GPU. O is written.
// Returns 0 on success, non-zero on error (cuDNN status or -1 for bad args).
//
// Graphs are cached keyed on (B, H, N_q, N_kv, D). For a Klein 9B run, every
// FA call uses the same shape, so the graph builds once and is reused for
// 64 calls × 50 steps × 2 CFG = 6400 executions per generate.
//
// Phase C standalone test (`/tmp/cudnn_sdpa_test.cpp`) measured 3.24 ms/call
// at the canonical Klein shape (B=1, H=24, N=4608, D=128), 12.11× faster
// than the in-tree WMMA kernel (39.26 ms/call).

#include <cudnn_frontend.h>
#include <cuda_runtime_api.h>
#include <cuda_bf16.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace fe = cudnn_frontend;

// Fixed tensor UIDs — the graph has exactly four bound tensors.
static constexpr int32_t Q_UID = 1;
static constexpr int32_t K_UID = 2;
static constexpr int32_t V_UID = 3;
static constexpr int32_t O_UID = 4;

// Cache key. Stride refactor Phase 2b: strides are part of the graph
// definition (cuDNN bakes them into the operation signature), so they must
// be part of the cache key. Offsets are pointer arithmetic applied at
// execute-time only — not part of the graph — so they are NOT in the key.
struct SdpaKey {
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

    bool operator==(const SdpaKey& o) const noexcept {
        if (!(B == o.B && H == o.H && N_q == o.N_q &&
              N_kv == o.N_kv && D == o.D && scale == o.scale))
            return false;
        for (int i = 0; i < 4; ++i) {
            if (q_s[i] != o.q_s[i]) return false;
            if (k_s[i] != o.k_s[i]) return false;
            if (v_s[i] != o.v_s[i]) return false;
            if (o_s[i] != o.o_s[i]) return false;
        }
        return true;
    }
};

struct SdpaKeyHash {
    size_t operator()(const SdpaKey& k) const noexcept {
        // FNV-ish mix; collisions are a miss not a correctness issue.
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
        }
        return h;
    }
};

struct SdpaEntry {
    std::shared_ptr<fe::graph::Graph> graph;
    int64_t                            workspace_size = 0;
    // Per-workspace buffer (allocated lazily on first execute with this key).
    // Reused across executions at this shape.
    void* workspace_buf = nullptr;
};

namespace {

std::once_flag            g_handle_once;
cudnnHandle_t             g_handle        = nullptr;
std::mutex                g_cache_mutex;
std::unordered_map<SdpaKey, SdpaEntry, SdpaKeyHash> g_cache;

int ensure_handle() {
    int ret = 0;
    std::call_once(g_handle_once, [&ret]() {
        cudnnStatus_t s = cudnnCreate(&g_handle);
        if (s != CUDNN_STATUS_SUCCESS) {
            fprintf(stderr, "[flame_cudnn_sdpa] cudnnCreate failed: %d\n", (int)s);
            ret = (int)s;
        }
    });
    if (!g_handle && ret == 0) return -999; // should not happen
    return ret;
}

// Build and finalize a graph for this shape. Called under g_cache_mutex.
int build_graph(const SdpaKey& k, SdpaEntry& entry) {
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

    auto opts = fe::graph::SDPA_attributes()
                    .set_name("flame_klein_sdpa")
                    .set_generate_stats(false)
                    .set_attn_scale(k.scale);
    // No causal / padding mask — Klein, FLUX, Chroma, Qwen-Edit joint attn
    // is full self-attention. Callers that need masking use the WMMA path.

    auto [O, Stats] = g->sdpa(Q, K, V, opts);
    (void)Stats; // generate_stats=false, Stats is nullptr

    O->set_output(true)
     .set_dim({B, H, Nq, D})
     .set_stride(to_vec4(k.o_s))
     .set_uid(O_UID);

    auto status = g->build(g_handle, {fe::HeurMode_t::A});
    if (!status.is_good()) {
        fprintf(stderr, "[flame_cudnn_sdpa] graph->build failed for (B=%d H=%d Nq=%d Nkv=%d D=%d): %s\n",
                k.B, k.H, k.N_q, k.N_kv, k.D, status.get_message().c_str());
        return -1;
    }

    int64_t ws = 0;
    status = g->get_workspace_size(ws);
    if (!status.is_good()) {
        fprintf(stderr, "[flame_cudnn_sdpa] get_workspace_size failed: %s\n",
                status.get_message().c_str());
        return -1;
    }

    void* ws_buf = nullptr;
    if (ws > 0) {
        cudaError_t e = cudaMalloc(&ws_buf, ws);
        if (e != cudaSuccess) {
            fprintf(stderr, "[flame_cudnn_sdpa] workspace cudaMalloc(%ld) failed: %s\n",
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

extern "C" int flame_cudnn_sdpa_bf16(
    const void* Q, const void* K, const void* V, void* O,
    int B, int H, int N_q, int N_kv, int D,
    float scale,
    const int64_t* q_strides,
    const int64_t* k_strides,
    const int64_t* v_strides,
    const int64_t* o_strides,
    int64_t q_offset_elems, int64_t k_offset_elems,
    int64_t v_offset_elems, int64_t o_offset_elems,
    void* stream
) {
    if (!Q || !K || !V || !O) return -1;
    if (B <= 0 || H <= 0 || N_q <= 0 || N_kv <= 0 || D <= 0) return -1;
    if (!q_strides || !k_strides || !v_strides || !o_strides) return -1;

    int rc = ensure_handle();
    if (rc != 0) return rc;

    SdpaKey key{};
    key.B = B; key.H = H; key.N_q = N_q; key.N_kv = N_kv; key.D = D;
    key.scale = scale;
    for (int i = 0; i < 4; ++i) {
        key.q_s[i] = q_strides[i];
        key.k_s[i] = k_strides[i];
        key.v_s[i] = v_strides[i];
        key.o_s[i] = o_strides[i];
    }

    // Serialize cache access. Graph execution itself is thread-safe per cuDNN
    // docs once the graph is finalized, so we hold the lock only across
    // lookup/build, not across execute.
    std::shared_ptr<fe::graph::Graph> graph;
    void*   ws_buf = nullptr;
    int64_t ws_sz  = 0;
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        auto it = g_cache.find(key);
        if (it == g_cache.end()) {
            SdpaEntry entry{};
            rc = build_graph(key, entry);
            if (rc != 0) return rc;
            it = g_cache.emplace(key, std::move(entry)).first;
        }
        graph  = it->second.graph;
        ws_buf = it->second.workspace_buf;
        ws_sz  = it->second.workspace_size;
        (void)ws_sz;
    }

    // Per-exec: set the caller's stream on the handle. The handle itself is
    // shared, but cudnnSetStream is cheap and the downstream work runs on
    // whatever stream we set.
    cudnnStatus_t s = cudnnSetStream(g_handle, (cudaStream_t)stream);
    if (s != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "[flame_cudnn_sdpa] cudnnSetStream failed: %d\n", (int)s);
        return (int)s;
    }

    // BF16 element size = 2 bytes. Offsets are applied here (not in the
    // graph) so we don't need a fresh graph for every slice.
    const size_t elem_bytes = 2;
    auto advance = [&](const void* p, int64_t off_elems) -> void* {
        return const_cast<void*>(static_cast<const void*>(
            static_cast<const char*>(p) + off_elems * (int64_t)elem_bytes));
    };

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> vp = {
        {Q_UID, advance(Q, q_offset_elems)},
        {K_UID, advance(K, k_offset_elems)},
        {V_UID, advance(V, v_offset_elems)},
        {O_UID, advance(O, o_offset_elems)},
    };

    auto status = graph->execute(g_handle, vp, ws_buf);
    if (!status.is_good()) {
        fprintf(stderr, "[flame_cudnn_sdpa] execute failed: %s\n",
                status.get_message().c_str());
        return -1;
    }
    return 0;
}
