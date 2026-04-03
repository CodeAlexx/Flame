#include "flame_cuda_common.cuh"

#include "../cuda_ops.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <cublasLt.h>
#include <functional>
#include <iomanip>
#include <limits>
#include <math_constants.h>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdlib>

namespace {

constexpr int kIm2ColBlockSize = 256;
constexpr int kAutotuneVersion = 1;
constexpr int kKernelBuildId = 1;
const std::array<int64_t, 9> kAutotuneRowCandidates = {32, 48, 64, 96, 128, 192, 256, 384, 512};

inline size_t hash_combine(size_t seed, int value) {
  size_t result = seed;
  result ^= static_cast<size_t>(value) + 0x9e3779b97f4a7c15ULL + (result << 6) + (result >> 2);
  return result;
}

struct ConvKey {
  int N;
  int Ho;
  int Wo;
  int Kh;
  int Kw;
  int Cin;
  int Cout;
  int groups;
  int stride_h;
  int stride_w;
  int dil_h;
  int dil_w;

  bool operator==(const ConvKey& other) const {
    return N == other.N && Ho == other.Ho && Wo == other.Wo && Kh == other.Kh &&
           Kw == other.Kw && Cin == other.Cin && Cout == other.Cout &&
           groups == other.groups && stride_h == other.stride_h &&
           stride_w == other.stride_w && dil_h == other.dil_h &&
           dil_w == other.dil_w;
  }
};

struct ConvKeyHash {
  size_t operator()(const ConvKey& key) const {
    size_t seed = 0;
    seed = hash_combine(seed, key.N);
    seed = hash_combine(seed, key.Ho);
    seed = hash_combine(seed, key.Wo);
    seed = hash_combine(seed, key.Kh);
    seed = hash_combine(seed, key.Kw);
    seed = hash_combine(seed, key.Cin);
    seed = hash_combine(seed, key.Cout);
    seed = hash_combine(seed, key.groups);
    seed = hash_combine(seed, key.stride_h);
    seed = hash_combine(seed, key.stride_w);
    seed = hash_combine(seed, key.dil_h);
    seed = hash_combine(seed, key.dil_w);
    return seed;
  }
};

struct DeviceSignature {
  int device = -1;
  int sm_major = 0;
  int sm_minor = 0;
  int driver_version = 0;
  int runtime_version = 0;
  long long cublaslt_version = 0;
  int kernel_build_id = 0;
  std::array<unsigned char, 16> uuid{};

  bool operator==(const DeviceSignature& other) const {
    return device == other.device && sm_major == other.sm_major &&
           sm_minor == other.sm_minor && driver_version == other.driver_version &&
           runtime_version == other.runtime_version &&
           cublaslt_version == other.cublaslt_version &&
           kernel_build_id == other.kernel_build_id && uuid == other.uuid;
  }
};

struct PlanCacheKey {
  ConvKey conv;
  DeviceSignature device;
  int autotune_version = 0;

  bool operator==(const PlanCacheKey& other) const {
    return conv == other.conv && device == other.device &&
           autotune_version == other.autotune_version;
  }
};

struct PlanCacheKeyHash {
  size_t operator()(const PlanCacheKey& key) const {
    size_t seed = ConvKeyHash{}(key.conv);
    seed = hash_combine(seed, key.device.device);
    seed = hash_combine(seed, key.device.sm_major);
    seed = hash_combine(seed, key.device.sm_minor);
    seed = hash_combine(seed, key.device.driver_version);
    seed = hash_combine(seed, key.device.runtime_version);
    seed = hash_combine(seed, static_cast<int>(key.device.cublaslt_version & 0xffffffff));
    seed = hash_combine(seed, static_cast<int>((key.device.cublaslt_version >> 32) & 0xffffffff));
    seed = hash_combine(seed, key.device.kernel_build_id);
    for (unsigned char b : key.device.uuid) {
      seed = hash_combine(seed, static_cast<int>(b));
    }
    seed = hash_combine(seed, key.autotune_version);
    return seed;
  }
};

struct AutotuneConfig {
  bool enabled = false;
  int warmups = 2;
  int repeats = 5;
  int max_candidates = 6;
  double tie_epsilon = 0.02;
  bool verbose = false;
  int64_t min_rows = 64;
};

struct CachedPlan {
  int64_t rows = 0;
  int reprobe_remaining = 0;
  bool weak = false;
};

struct AutotuneStats {
  std::atomic<int64_t> cache_hits{0};
  std::atomic<int64_t> cache_misses{0};
  std::atomic<int64_t> tuned{0};
  std::atomic<int64_t> weak{0};
  std::atomic<int64_t> fallbacks{0};
  std::atomic<int64_t> workspace_skips{0};
  std::atomic<int64_t> errors{0};
  std::atomic<int64_t> reprobes{0};
};

static std::mutex g_conv_plan_mutex;
static std::unordered_map<PlanCacheKey, CachedPlan, PlanCacheKeyHash> g_conv_plan_cache;
static AutotuneStats g_autotune_stats;
static std::mutex g_autotune_log_mutex;

constexpr int kWeakReprobeDefault = 10;

int parse_env_int(const char* name, int default_value) {
  const char* value = std::getenv(name);
  if (!value || *value == '\0') {
    return default_value;
  }
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value) {
    return default_value;
  }
  return static_cast<int>(parsed);
}

bool parse_env_bool(const char* name, bool default_value) {
  const char* value = std::getenv(name);
  if (!value || *value == '\0') {
    return default_value;
  }
  std::string lowered(value);
  for (char& ch : lowered) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  if (lowered == "0" || lowered == "false" || lowered == "off" || lowered == "no") {
    return false;
  }
  if (lowered == "1" || lowered == "true" || lowered == "on" || lowered == "yes") {
    return true;
  }
  return default_value;
}

const AutotuneConfig& get_autotune_config() {
  static AutotuneConfig config = [] {
    AutotuneConfig cfg;
    cfg.enabled = parse_env_bool("FLAME_CONV2D_AUTOTUNE", true);
    cfg.warmups = std::max(0, parse_env_int("FLAME_CONV2D_AUTOTUNE_WARMUPS", cfg.warmups));
    cfg.repeats = std::max(1, parse_env_int("FLAME_CONV2D_AUTOTUNE_REPEATS", cfg.repeats));
    cfg.max_candidates =
        std::max(1, parse_env_int("FLAME_CONV2D_AUTOTUNE_MAX_CANDS", cfg.max_candidates));
    cfg.verbose = parse_env_bool("FLAME_CONV2D_AUTOTUNE_VERBOSE", false);
    cfg.min_rows = std::max<int64_t>(
        1, static_cast<int64_t>(parse_env_int("FLAME_CONV2D_AUTOTUNE_MIN_ROWS", 64)));
    int tie_eps_basis =
        parse_env_int("FLAME_CONV2D_AUTOTUNE_TIE_EPS_PCT", static_cast<int>(cfg.tie_epsilon * 100));
    cfg.tie_epsilon = std::max(0.0, static_cast<double>(tie_eps_basis) / 100.0);
    return cfg;
  }();
  return config;
}

DeviceSignature query_device_signature(int device) {
  DeviceSignature sig{};
  sig.device = device;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
    sig.sm_major = prop.major;
    sig.sm_minor = prop.minor;
#if CUDART_VERSION >= 11000
    for (size_t i = 0; i < sig.uuid.size(); ++i) {
      sig.uuid[i] = prop.uuid.bytes[i];
    }
#endif
  }
  cudaRuntimeGetVersion(&sig.runtime_version);
  cudaDriverGetVersion(&sig.driver_version);
  sig.cublaslt_version = static_cast<long long>(cublasLtGetVersion());
  sig.kernel_build_id = kKernelBuildId;
  return sig;
}

PlanCacheKey make_plan_cache_key(const ConvKey& conv, int device) {
  PlanCacheKey key;
  key.conv = conv;
  key.device = query_device_signature(device);
  key.autotune_version = kAutotuneVersion;
  return key;
}

std::string conv_key_to_string(const ConvKey& key) {
  char buffer[256];
  std::snprintf(buffer,
                sizeof(buffer),
                "N=%d Ho=%d Wo=%d Kh=%d Kw=%d Cin=%d Cout=%d groups=%d stride=(%d,%d) "
                "dil=(%d,%d)",
                key.N,
                key.Ho,
                key.Wo,
                key.Kh,
                key.Kw,
                key.Cin,
                key.Cout,
                key.groups,
                key.stride_h,
                key.stride_w,
                key.dil_h,
                key.dil_w);
  return std::string(buffer);
}

std::string plan_key_to_string(const PlanCacheKey& key) {
  std::ostringstream oss;
  oss << conv_key_to_string(key.conv)
      << " dev=" << key.device.device << " sm=" << key.device.sm_major << key.device.sm_minor
      << " drv=" << key.device.driver_version << " rt=" << key.device.runtime_version
      << " lt=" << key.device.cublaslt_version
      << " build=" << key.device.kernel_build_id
      << " ver=" << key.autotune_version;
  return oss.str();
}

struct AutotuneDecision {
  bool success = false;
  bool weak = false;
  int64_t rows = 0;
};

double compute_trimmed_median(std::vector<float> samples) {
  if (samples.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  std::sort(samples.begin(), samples.end());
  size_t drop = static_cast<size_t>(std::ceil(samples.size() * 0.2));
  if (drop >= samples.size()) {
    drop = samples.size() > 1 ? samples.size() - 1 : 0;
  }
  samples.resize(samples.size() - drop);
  const size_t n = samples.size();
  if (n == 0) {
    return std::numeric_limits<double>::infinity();
  }
  if (n % 2 == 1) {
    return static_cast<double>(samples[n / 2]);
  }
  return 0.5 * (static_cast<double>(samples[n / 2 - 1]) +
                static_cast<double>(samples[n / 2]));
}

AutotuneDecision autotune_rows(const PlanCacheKey& plan_key,
                               const AutotuneConfig& config,
                               int64_t total_rows,
                               int64_t workspace_rows_cap,
                               int64_t heuristic_rows,
                               int64_t spatial,
                               const std::vector<int64_t>& base_candidates,
                               cudaStream_t stream,
                               const std::function<FlameCudaStatus(int64_t)>& run_with_rows) {
  AutotuneDecision decision;
  if (!config.enabled) {
    return decision;
  }
  if (workspace_rows_cap <= 0) {
    g_autotune_stats.workspace_skips.fetch_add(1, std::memory_order_relaxed);
    return decision;
  }
  if (total_rows < config.min_rows || spatial < config.min_rows) {
    g_autotune_stats.workspace_skips.fetch_add(1, std::memory_order_relaxed);
    return decision;
  }

  std::vector<int64_t> candidates;
  candidates.reserve(base_candidates.size() + 1);
  for (int64_t cand : base_candidates) {
    if (cand <= 0) {
      continue;
    }
    int64_t rows = std::max<int64_t>(1, std::min<int64_t>(cand, total_rows));
    if (rows > workspace_rows_cap) {
      continue;
    }
    if (rows < config.min_rows && workspace_rows_cap >= config.min_rows) {
      continue;
    }
    if (std::find(candidates.begin(), candidates.end(), rows) == candidates.end()) {
      candidates.push_back(rows);
    }
  }
  if (heuristic_rows > 0 && heuristic_rows <= workspace_rows_cap &&
      std::find(candidates.begin(), candidates.end(), heuristic_rows) == candidates.end()) {
    candidates.push_back(heuristic_rows);
  }
  if (candidates.empty()) {
    g_autotune_stats.workspace_skips.fetch_add(1, std::memory_order_relaxed);
    return decision;
  }
  if (static_cast<int>(candidates.size()) > config.max_candidates) {
    candidates.resize(config.max_candidates);
  }

  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;
  if (cudaEventCreate(&start_event) != cudaSuccess ||
      cudaEventCreate(&stop_event) != cudaSuccess) {
    if (start_event) {
      cudaEventDestroy(start_event);
    }
    if (stop_event) {
      cudaEventDestroy(stop_event);
    }
    g_autotune_stats.errors.fetch_add(1, std::memory_order_relaxed);
    return decision;
  }

  struct CandidateMetrics {
    int64_t rows = 0;
    double median_ms = std::numeric_limits<double>::infinity();
    bool valid = false;
  };

  std::vector<CandidateMetrics> metrics;
  metrics.reserve(candidates.size());

  for (int64_t rows : candidates) {
    CandidateMetrics metric;
    metric.rows = rows;
    bool valid = true;

    for (int w = 0; w < config.warmups; ++w) {
      if (run_with_rows(rows) != FLAME_CUDA_OK) {
        valid = false;
        break;
      }
    }
    if (!valid) {
      metrics.push_back(metric);
      continue;
    }

    std::vector<float> timings;
    timings.reserve(config.repeats);
    for (int r = 0; r < config.repeats; ++r) {
      if (cudaEventRecord(start_event, stream) != cudaSuccess) {
        valid = false;
        break;
      }
      FlameCudaStatus run_status = run_with_rows(rows);
      if (run_status != FLAME_CUDA_OK) {
        valid = false;
        break;
      }
      if (cudaEventRecord(stop_event, stream) != cudaSuccess) {
        valid = false;
        break;
      }
      if (cudaEventSynchronize(stop_event) != cudaSuccess) {
        valid = false;
        break;
      }
      float elapsed_ms = 0.0f;
      if (cudaEventElapsedTime(&elapsed_ms, start_event, stop_event) != cudaSuccess) {
        valid = false;
        break;
      }
      timings.push_back(elapsed_ms);
    }

    if (valid && !timings.empty()) {
      metric.median_ms = compute_trimmed_median(timings);
      metric.valid = std::isfinite(metric.median_ms);
    }
    metrics.push_back(metric);
  }

  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);

  std::vector<CandidateMetrics> valid_metrics;
  for (const auto& metric : metrics) {
    if (metric.valid) {
      valid_metrics.push_back(metric);
    }
  }

  if (valid_metrics.empty()) {
    g_autotune_stats.errors.fetch_add(1, std::memory_order_relaxed);
    return decision;
  }

  std::sort(valid_metrics.begin(),
            valid_metrics.end(),
            [&](const CandidateMetrics& a, const CandidateMetrics& b) {
              if (std::fabs(a.median_ms - b.median_ms) <= b.median_ms * config.tie_epsilon) {
                return a.rows < b.rows;
              }
              return a.median_ms < b.median_ms;
            });

  const CandidateMetrics& winner = valid_metrics.front();
  double second_best_ms = std::numeric_limits<double>::infinity();
  if (valid_metrics.size() > 1) {
    second_best_ms = valid_metrics[1].median_ms;
  }

  decision.success = true;
  decision.rows = winner.rows;
  if (std::isfinite(second_best_ms) &&
      winner.median_ms > second_best_ms * 0.98) {
    decision.weak = true;
    g_autotune_stats.weak.fetch_add(1, std::memory_order_relaxed);
  }

  if (config.verbose) {
    std::lock_guard<std::mutex> log_lock(g_autotune_log_mutex);
    std::ostringstream cand_rows;
    std::ostringstream cand_ms;
    bool first = true;
    for (const auto& metric : valid_metrics) {
      if (!first) {
        cand_rows << ",";
        cand_ms << ",";
      }
      first = false;
      cand_rows << metric.rows;
      cand_ms << std::fixed << std::setprecision(4) << metric.median_ms;
    }
    std::fprintf(stdout,
                 "conv2d_autotune: key=%s candidates=[%s] ms=[%s] winner=%lld (%.4f ms)%s\n",
                 plan_key_to_string(plan_key).c_str(),
                 cand_rows.str().c_str(),
                 cand_ms.str().c_str(),
                 static_cast<long long>(winner.rows),
                 winner.median_ms,
                 decision.weak ? " weak" : "");
  }

  return decision;
}

inline int64_t align_down_multiple(int64_t value, int64_t multiple) {
  if (multiple <= 0) {
    return value;
  }
  if (value < multiple) {
    return 0;
  }
  return (value / multiple) * multiple;
}

inline int64_t plan_tile_rows(int64_t total_rows, int64_t max_rows_available) {
  if (max_rows_available <= 0) {
    return 0;
  }
  if (total_rows <= max_rows_available) {
    return std::max<int64_t>(1, total_rows);
  }
  if (max_rows_available < 32) {
    return std::max<int64_t>(1, max_rows_available);
  }

  const int64_t capped_rows =
      std::min<int64_t>(max_rows_available, static_cast<int64_t>(512));
  constexpr int64_t kPreferred[] = {
      512, 384, 320, 256, 224, 192, 160, 128, 96, 80, 64, 48, 32,
  };
  for (int64_t candidate : kPreferred) {
    if (candidate > capped_rows) {
      continue;
    }
    int64_t aligned = align_down_multiple(candidate, 32);
    if (aligned == 0) {
      aligned = candidate;
    }
    if (aligned <= 0 || aligned > capped_rows) {
      continue;
    }
    if (aligned <= total_rows) {
      return aligned;
    }
  }

  int64_t fallback = align_down_multiple(capped_rows, 32);
  if (fallback <= 0) {
    fallback = std::min<int64_t>(capped_rows, total_rows);
  }
  if (fallback <= 0) {
    fallback = 1;
  }
  return std::min<int64_t>(fallback, total_rows);
}

int64_t heuristic_select_rows(const ConvKey& key,
                              int64_t total_rows,
                              int64_t K_group,
                              int group_out) {
  (void)key;
  const int64_t max_rows = std::max<int64_t>(
      static_cast<int64_t>(1), std::min<int64_t>(total_rows, static_cast<int64_t>(512)));
  const int64_t preferred_bases[] = {512, 384, 320, 256, 224, 192, 160,
                                     128, 96,  80,  64,  48,  32,  16,  8};

  std::vector<int64_t> candidates;
  candidates.reserve(sizeof(preferred_bases) / sizeof(int64_t) + 3);
  for (int64_t base : preferred_bases) {
    if (base <= 0) {
      continue;
    }
    int64_t cand = std::min<int64_t>(base, max_rows);
    if (cand <= 0) {
      continue;
    }
    if (cand > total_rows) {
      cand = total_rows;
    }
    if (cand <= 0) {
      continue;
    }
    if (std::find(candidates.begin(), candidates.end(), cand) == candidates.end()) {
      candidates.push_back(cand);
    }
  }

  int64_t aligned_full = align_down_multiple(max_rows, 32);
  if (aligned_full <= 0) {
    aligned_full = max_rows;
  }
  if (aligned_full > 0 && aligned_full <= total_rows &&
      std::find(candidates.begin(), candidates.end(), aligned_full) == candidates.end()) {
    candidates.push_back(aligned_full);
  }

  if (std::find(candidates.begin(), candidates.end(), total_rows) == candidates.end()) {
    candidates.push_back(std::max<int64_t>(1, total_rows));
  }

  double best_score = std::numeric_limits<double>::infinity();
  int64_t best_rows = std::max<int64_t>(1, std::min<int64_t>(max_rows, total_rows));
  for (int64_t rows : candidates) {
    if (rows <= 0) {
      continue;
    }
    rows = std::max<int64_t>(1, std::min<int64_t>(rows, total_rows));
    const int64_t tiles = (total_rows + rows - 1) / rows;
    double tile_factor = static_cast<double>(tiles);
    double row_factor = static_cast<double>(rows);

    const double im2col_cost = tile_factor * row_factor * static_cast<double>(K_group);
    const double gemm_cost =
        tile_factor * static_cast<double>(K_group) * static_cast<double>(group_out);
    double cost = im2col_cost + gemm_cost;

    if (rows < 64) {
      cost *= 1.15;
    }
    if (rows > 256) {
      cost *= 1.05;
    }

    if (cost < best_score) {
      best_score = cost;
      best_rows = rows;
    }
  }

  return best_rows;
}

__global__ void depthwise_conv2d_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                                             const __nv_bfloat16* __restrict__ filter,
                                             const __nv_bfloat16* __restrict__ bias,
                                             __nv_bfloat16* __restrict__ output,
                                             int N,
                                             int H,
                                             int W,
                                             int Ho,
                                             int Wo,
                                             int Kh,
                                             int Kw,
                                             int stride_h,
                                             int stride_w,
                                             int pad_h,
                                             int pad_w,
                                             int dil_h,
                                             int dil_w,
                                             int channels) {
  const int64_t total = static_cast<int64_t>(N) * Ho * Wo * channels;
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  while (idx < total) {
    int channel = static_cast<int>(idx % channels);
    int64_t tmp = idx / channels;
    int ow = static_cast<int>(tmp % Wo);
    tmp /= Wo;
    int oh = static_cast<int>(tmp % Ho);
    int n = static_cast<int>(tmp / Ho);

    float acc = 0.0f;
    if (bias != nullptr) {
      acc = __bfloat162float(bias[channel]);
    }

    for (int kh_idx = 0; kh_idx < Kh; ++kh_idx) {
      const int ih = oh * stride_h - pad_h + kh_idx * dil_h;
      if (ih < 0 || ih >= H) {
        continue;
      }
      for (int kw_idx = 0; kw_idx < Kw; ++kw_idx) {
        const int iw = ow * stride_w - pad_w + kw_idx * dil_w;
        if (iw < 0 || iw >= W) {
          continue;
        }

        const int64_t input_index =
            ((static_cast<int64_t>(n) * H + ih) * W + iw) * channels + channel;
        const int64_t filter_index =
            ((static_cast<int64_t>(kh_idx) * Kw + kw_idx) * channels) + channel;

        const float x_val = __bfloat162float(input[input_index]);
        const float w_val = __bfloat162float(filter[filter_index]);
        acc += x_val * w_val;
      }
    }

    const int64_t out_index =
        ((static_cast<int64_t>(n) * Ho + oh) * Wo + ow) * channels + channel;
    output[out_index] = __float2bfloat16(acc);
    idx += static_cast<int64_t>(gridDim.x) * blockDim.x;
  }
}

__device__ __forceinline__ float activation_apply(float x, int activation) {
  switch (activation) {
    case 1:
      return x > 0.0f ? x : 0.0f;
    case 2:
      return x / (1.0f + expf(-x));
    case 3: {
      const float c = 0.044715f;
      const float inner = CUDART_SQRT_2_OVER_PI_F * (x + c * x * x * x);
      return 0.5f * x * (1.0f + tanhf(inner));
    }
    default:
      return x;
  }
}

__global__ void apply_activation_kernel(__nv_bfloat16* data,
                                        int64_t total,
                                        int activation) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  while (idx < total) {
    float value = __bfloat162float(data[idx]);
    value = activation_apply(value, activation);
    data[idx] = __float2bfloat16(value);
    idx += static_cast<int64_t>(gridDim.x) * blockDim.x;
  }
}

__global__ void bf16_matmul_bias_kernel(const __nv_bfloat16* __restrict__ A,
                                        const __nv_bfloat16* __restrict__ B,
                                        const __nv_bfloat16* __restrict__ bias,
                                        __nv_bfloat16* __restrict__ C,
                                        int M,
                                        int K,
                                        int N,
                                        int ldb,
                                        int ldc) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) {
    return;
  }

  float acc = 0.0f;
  if (bias != nullptr) {
    acc = __bfloat162float(bias[col]);
  }

  const int64_t a_row_offset = static_cast<int64_t>(row) * K;
  const int64_t b_col_offset = col;
  for (int k = 0; k < K; ++k) {
    const float a_val = __bfloat162float(A[a_row_offset + k]);
    const float b_val = __bfloat162float(B[static_cast<int64_t>(k) * ldb + b_col_offset]);
    acc += a_val * b_val;
  }

  const int64_t c_index = static_cast<int64_t>(row) * ldc + col;
  C[c_index] = __float2bfloat16(acc);
}

inline FlameCudaStatus from_fc_status(fc_status_t status) {
  switch (status) {
    case FC_OK:
      return FLAME_CUDA_OK;
    case FC_ERR_INVALID_ARGUMENT:
      return FLAME_CUDA_ERR_INVALID;
    case FC_ERR_UNSUPPORTED:
      return FLAME_CUDA_ERR_UNSUPPORTED;
    case FC_ERR_OOM:
    case FC_ERR_LAUNCH:
    default:
      return FLAME_CUDA_ERR_CUDA;
  }
}

__global__ void im2col_bf16_tile(const __nv_bfloat16* __restrict__ input,
                                 __nv_bfloat16* __restrict__ output,
                                 int64_t row_offset,
                                 int64_t rows,
                                 int64_t k_cols,
                                 int64_t H,
                                 int64_t W,
                                 int64_t C,
                                 int64_t KH,
                                 int64_t KW,
                                 int64_t Ho,
                                 int64_t Wo,
                                 int32_t stride_h,
                                 int32_t stride_w,
                                 int32_t pad_h,
                                 int32_t pad_w,
                                 int32_t dil_h,
                                 int32_t dil_w,
                                 int64_t stride_n,
                                 int64_t stride_h_in,
                                 int64_t stride_w_in,
                                 int64_t stride_c_in) {
  const int64_t total = rows * k_cols;
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < total) {
    const int64_t local_row = idx / k_cols;
    const int64_t col = idx % k_cols;
    const int64_t row = row_offset + local_row;

    const int64_t ic = col % C;
    const int64_t filter_idx = col / C;
    const int64_t kw = filter_idx % KW;
    const int64_t kh = filter_idx / KW;

    const int64_t n = row / (Ho * Wo);
    const int64_t hw = row % (Ho * Wo);
    const int64_t oh = hw / Wo;
    const int64_t ow = hw % Wo;

    const int64_t ih = oh * stride_h - pad_h + kh * dil_h;
    const int64_t iw = ow * stride_w - pad_w + kw * dil_w;

    __nv_bfloat16 value = __float2bfloat16(0.0f);
    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
      const int64_t offset = n * stride_n + ih * stride_h_in + iw * stride_w_in + ic * stride_c_in;
      value = input[offset];
    }
    output[idx] = value;
    idx += static_cast<int64_t>(blockDim.x) * gridDim.x;
  }
}

inline bool validate_positive(int value) {
  return value > 0;
}

}  // namespace

FlameCudaStatus flame_conv2d_nhwc_bf16_impl(const __nv_bfloat16* x,
                                            const __nv_bfloat16* w,
                                            const __nv_bfloat16* bias,
                                            int N,
                                            int H,
                                            int W,
                                            int Cin,
                                            int Kh,
                                            int Kw,
                                            int stride_h,
                                            int stride_w,
                                            int pad_h,
                                            int pad_w,
                                            int dil_h,
                                            int dil_w,
                                            int Cout,
                                            int activation,
                                            int groups,
                                            __nv_bfloat16* y,
                                            void* workspace,
                                            size_t workspace_bytes,
                                            cudaStream_t stream) {
  if (!x || !w || !y || !validate_positive(N) || !validate_positive(H) ||
      !validate_positive(W) || !validate_positive(Cin) || !validate_positive(Kh) ||
      !validate_positive(Kw) || !validate_positive(Cout) || stride_h <= 0 ||
      stride_w <= 0 || dil_h <= 0 || dil_w <= 0) {
    return FLAME_CUDA_ERR_INVALID;
  }

  const int64_t Ho =
      ((static_cast<int64_t>(H) + 2 * pad_h -
        (static_cast<int64_t>(Kh) - 1) * dil_h - 1) / stride_h) +
      1;
  const int64_t Wo =
      ((static_cast<int64_t>(W) + 2 * pad_w -
        (static_cast<int64_t>(Kw) - 1) * dil_w - 1) / stride_w) +
      1;

  if (Ho <= 0 || Wo <= 0) {
    return FLAME_CUDA_OK;
  }

  if (groups <= 0 || Cin % groups != 0 || Cout % groups != 0) {
    return FLAME_CUDA_ERR_INVALID;
  }
  const int group_count = groups;
  const int group_in = Cin / group_count;
  const int group_out = Cout / group_count;
  const bool depthwise = (group_count == Cin) && (Cout == Cin);

  const int64_t M = static_cast<int64_t>(N) * Ho * Wo;
  if (M <= 0) {
    return FLAME_CUDA_OK;
  }

  if (depthwise) {
    const int64_t total = static_cast<int64_t>(N) * Ho * Wo * Cout;
    if (total > 0) {
      const int blocks = static_cast<int>(std::max<int64_t>(
          1, (total + kIm2ColBlockSize - 1) / kIm2ColBlockSize));
      depthwise_conv2d_bf16_kernel<<<blocks, kIm2ColBlockSize, 0, stream>>>(
          x,
          w,
          bias,
          y,
          N,
          H,
          W,
          Ho,
          Wo,
          Kh,
          Kw,
          stride_h,
          stride_w,
          pad_h,
          pad_w,
          dil_h,
          dil_w,
          Cout);
      cudaError_t depth_err = cudaGetLastError();
      if (depth_err != cudaSuccess) {
        return FLAME_CUDA_ERR_CUDA;
      }
    }
  } else {
    const int64_t K_group = static_cast<int64_t>(Kh) * Kw * group_in;
    if (K_group <= 0) {
      return FLAME_CUDA_OK;
    }
    if (group_count <= 0 || group_in <= 0 || group_out <= 0) {
      return FLAME_CUDA_ERR_INVALID;
    }

    const size_t elem_size = sizeof(__nv_bfloat16);
    const size_t bytes_per_row = static_cast<size_t>(K_group) * elem_size;
    if (bytes_per_row == 0) {
      return FLAME_CUDA_ERR_INVALID;
    }

    ConvKey key{};
    key.N = N;
    key.Ho = static_cast<int>(Ho);
    key.Wo = static_cast<int>(Wo);
    key.Kh = Kh;
    key.Kw = Kw;
    key.Cin = Cin;
    key.Cout = Cout;
    key.groups = group_count;
    key.stride_h = stride_h;
    key.stride_w = stride_w;
    key.dil_h = dil_h;
    key.dil_w = dil_w;

    auto* workspace_ptr = static_cast<__nv_bfloat16*>(workspace);
    size_t capacity_bytes = workspace_bytes;
    bool owns_workspace = false;

    if (workspace_ptr != nullptr && capacity_bytes < bytes_per_row) {
      workspace_ptr = nullptr;
      capacity_bytes = 0;
    }

    const int64_t stride_c = 1;
    const int64_t stride_w_in = static_cast<int64_t>(Cin);
    const int64_t stride_h_in = static_cast<int64_t>(W) * stride_w_in;
    const int64_t stride_n = static_cast<int64_t>(H) * stride_h_in;
    int current_device = 0;
    cudaGetDevice(&current_device);
    PlanCacheKey plan_key = make_plan_cache_key(key, current_device);

    int64_t heuristic_rows = heuristic_select_rows(key, M, K_group, group_out);
    if (heuristic_rows <= 0) {
      heuristic_rows =
          std::max<int64_t>(1, std::min<int64_t>(M, static_cast<int64_t>(512)));
    }

    int64_t rows_per_tile = heuristic_rows;
    if (workspace_ptr != nullptr && capacity_bytes >= bytes_per_row) {
      int64_t max_rows_from_workspace =
          static_cast<int64_t>(capacity_bytes / bytes_per_row);
      if (max_rows_from_workspace > 0) {
        rows_per_tile =
            std::max<int64_t>(1, std::min<int64_t>(rows_per_tile, max_rows_from_workspace));
        rows_per_tile = std::min<int64_t>(
            rows_per_tile, plan_tile_rows(M, max_rows_from_workspace));
      } else {
        rows_per_tile = 1;
      }
    } else if (workspace_ptr == nullptr) {
      capacity_bytes =
          static_cast<size_t>(std::max<int64_t>(1, rows_per_tile)) * bytes_per_row;
      if (capacity_bytes < bytes_per_row) {
        capacity_bytes = bytes_per_row;
        rows_per_tile = 1;
      }
      cudaError_t alloc = cudaMallocAsync(
          reinterpret_cast<void**>(&workspace_ptr), capacity_bytes, stream);
      if (alloc != cudaSuccess) {
        fprintf(stderr,
                "[conv2d_stub] cudaMallocAsync failed: %d (%s) bytes=%zu\n",
                static_cast<int>(alloc),
                cudaGetErrorString(alloc),
                capacity_bytes);
        return FLAME_CUDA_ERR_CUDA;
      }
      owns_workspace = true;
    } else {
      rows_per_tile = 1;
    }

    rows_per_tile = std::max<int64_t>(1, std::min<int64_t>(rows_per_tile, M));
    capacity_bytes =
        static_cast<size_t>(std::max<int64_t>(1, rows_per_tile)) * bytes_per_row;

    auto run_with_rows = [&](int64_t tile_rows) -> FlameCudaStatus {
      if (tile_rows <= 0 ||
          tile_rows > static_cast<int64_t>(capacity_bytes / bytes_per_row)) {
        return FLAME_CUDA_ERR_INVALID;
      }
      tile_rows = std::max<int64_t>(1, std::min<int64_t>(tile_rows, M));

      for (int64_t row_offset = 0; row_offset < M; row_offset += tile_rows) {
        const int64_t current_tile_rows = std::min(tile_rows, M - row_offset);

        for (int g = 0; g < group_count; ++g) {
          const __nv_bfloat16* x_group = x + g * group_in;
          const __nv_bfloat16* w_group = w + g * group_out;
          const __nv_bfloat16* bias_group = bias ? bias + g * group_out : nullptr;
          __nv_bfloat16* y_group = y + row_offset * Cout + g * group_out;

          const int64_t total = current_tile_rows * K_group;
          const int blocks = static_cast<int>(std::max<int64_t>(
              1, (total + kIm2ColBlockSize - 1) / kIm2ColBlockSize));

          im2col_bf16_tile<<<blocks, kIm2ColBlockSize, 0, stream>>>(
              x_group,
              workspace_ptr,
              row_offset,
              current_tile_rows,
              K_group,
              H,
              W,
              group_in,
              Kh,
              Kw,
              Ho,
              Wo,
              stride_h,
              stride_w,
              pad_h,
              pad_w,
              dil_h,
              dil_w,
              stride_n,
              stride_h_in,
              stride_w_in,
              stride_c);
          cudaError_t launch_err = cudaGetLastError();
          if (launch_err != cudaSuccess) {
            fprintf(stderr,
                    "[conv2d_stub] im2col_bf16_tile launch failed: %d (%s) tile_rows=%lld K=%lld\n",
                    static_cast<int>(launch_err),
                    cudaGetErrorString(launch_err),
                    static_cast<long long>(current_tile_rows),
                    static_cast<long long>(K_group));
            return FLAME_CUDA_ERR_CUDA;
          }

          fc_tensor_view_t a_view{};
          a_view.rank = 2;
          a_view.data = workspace_ptr;
          a_view.dims[0] = current_tile_rows;
          a_view.dims[1] = K_group;
          a_view.strides[0] = K_group;
          a_view.strides[1] = 1;

          fc_tensor_view_t w_view{};
          w_view.rank = 2;
          w_view.data = const_cast<void*>(reinterpret_cast<const void*>(w_group));
          w_view.dims[0] = K_group;
          w_view.dims[1] = group_out;
          w_view.strides[0] = Cout;
          w_view.strides[1] = 1;

          fc_tensor_view_t y_view{};
          y_view.rank = 2;
          y_view.data = y_group;
          y_view.dims[0] = current_tile_rows;
          y_view.dims[1] = group_out;
          y_view.strides[0] = Cout;
          y_view.strides[1] = 1;

          fc_tensor_view_t bias_view{};
          const fc_tensor_view_t* bias_ptr_view = nullptr;
          if (bias_group != nullptr) {
            bias_view.rank = 1;
            bias_view.data = const_cast<void*>(reinterpret_cast<const void*>(bias_group));
            bias_view.dims[0] = group_out;
            bias_view.strides[0] = 1;
            bias_ptr_view = &bias_view;
          }

          fc_status_t gemm_status =
              fc_gemm_bf16(&a_view, &w_view, bias_ptr_view, &y_view, stream);
          if (gemm_status == FC_STATUS_LT_FALLBACK) {
            gemm_status = FC_OK;
          }
          if (gemm_status == FC_ERR_UNSUPPORTED) {
            const dim3 block(16, 16, 1);
            const dim3 grid((group_out + block.x - 1) / block.x,
                            (current_tile_rows + block.y - 1) / block.y,
                            1);
            bf16_matmul_bias_kernel<<<grid, block, 0, stream>>>(
                workspace_ptr,
                w_group,
                bias_group,
                y_group,
                static_cast<int>(current_tile_rows),
                static_cast<int>(K_group),
                group_out,
                Cout,
                Cout);
            cudaError_t matmul_err = cudaGetLastError();
            if (matmul_err != cudaSuccess) {
              fprintf(stderr,
                      "[conv2d_stub] fallback bf16_matmul_bias kernel failed: %d (%s) tile_rows=%lld K=%lld Cout=%d\n",
                      static_cast<int>(matmul_err),
                      cudaGetErrorString(matmul_err),
                      static_cast<long long>(current_tile_rows),
                      static_cast<long long>(K_group),
                      group_out);
              return FLAME_CUDA_ERR_CUDA;
            }
          } else if (gemm_status != FC_OK) {
            fprintf(stderr,
                    "[conv2d_stub] fc_gemm_bf16 failed: status=%d tile_rows=%lld K=%lld Cout=%d\n",
                    static_cast<int>(gemm_status),
                    static_cast<long long>(current_tile_rows),
                    static_cast<long long>(K_group),
                    group_out);
            return from_fc_status(gemm_status);
          }
        }
      }

      return FLAME_CUDA_OK;
    };

    const AutotuneConfig& autotune_cfg = get_autotune_config();
    const int64_t workspace_rows_cap =
        capacity_bytes > 0 ? static_cast<int64_t>(capacity_bytes / bytes_per_row) : 0;
    const int64_t spatial = static_cast<int64_t>(Ho) * static_cast<int64_t>(Wo);

    auto store_plan = [&](int64_t rows, bool weak) {
      CachedPlan plan;
      plan.rows = rows;
      plan.weak = weak;
      plan.reprobe_remaining = weak ? kWeakReprobeDefault : 0;
      std::lock_guard<std::mutex> lock(g_conv_plan_mutex);
      g_conv_plan_cache[plan_key] = plan;
    };

    bool use_cached_plan = false;
    int64_t cached_rows = 0;
    {
      std::lock_guard<std::mutex> lock(g_conv_plan_mutex);
      auto it = g_conv_plan_cache.find(plan_key);
      if (it != g_conv_plan_cache.end()) {
        use_cached_plan = true;
        cached_rows = it->second.rows;
        g_autotune_stats.cache_hits.fetch_add(1, std::memory_order_relaxed);
        if (it->second.weak && it->second.reprobe_remaining > 0) {
          it->second.reprobe_remaining--;
          if (it->second.reprobe_remaining == 0) {
            g_autotune_stats.reprobes.fetch_add(1, std::memory_order_relaxed);
            g_conv_plan_cache.erase(it);
          }
        }
      } else {
        g_autotune_stats.cache_misses.fetch_add(1, std::memory_order_relaxed);
      }
    }

    if (use_cached_plan) {
      rows_per_tile = std::max<int64_t>(1, std::min<int64_t>(cached_rows, workspace_rows_cap));
    } else if (autotune_cfg.enabled) {
      std::vector<int64_t> candidate_rows(
          kAutotuneRowCandidates.begin(), kAutotuneRowCandidates.end());
      AutotuneDecision decision = autotune_rows(plan_key,
                                                autotune_cfg,
                                                M,
                                                workspace_rows_cap,
                                                rows_per_tile,
                                                spatial,
                                                candidate_rows,
                                                stream,
                                                run_with_rows);
      if (decision.success) {
        rows_per_tile =
            std::max<int64_t>(1, std::min<int64_t>(decision.rows, workspace_rows_cap));
        g_autotune_stats.tuned.fetch_add(1, std::memory_order_relaxed);
        store_plan(rows_per_tile, decision.weak);
      } else {
        g_autotune_stats.fallbacks.fetch_add(1, std::memory_order_relaxed);
        store_plan(rows_per_tile, false);
      }
    } else {
      g_autotune_stats.fallbacks.fetch_add(1, std::memory_order_relaxed);
      store_plan(rows_per_tile, false);
    }

    rows_per_tile =
        std::max<int64_t>(1, std::min<int64_t>(rows_per_tile, workspace_rows_cap));
    rows_per_tile = std::min<int64_t>(rows_per_tile, M);

    FlameCudaStatus run_status = run_with_rows(rows_per_tile);
    if (run_status != FLAME_CUDA_OK) {
      if (owns_workspace) {
        FLAME_CUDA_TRY(cudaFreeAsync(workspace_ptr, stream));
      }
      fprintf(stderr,
              "[conv2d_stub] run_with_rows returned status=%d rows_per_tile=%lld\n",
              static_cast<int>(run_status),
              static_cast<long long>(rows_per_tile));
      return run_status;
    }

    if (owns_workspace) {
      FLAME_CUDA_TRY(cudaFreeAsync(workspace_ptr, stream));
    }
  }

  const int64_t total_elements = static_cast<int64_t>(N) * Ho * Wo * Cout;
  if (activation != 0 && total_elements > 0) {
    const int blocks = static_cast<int>(std::max<int64_t>(
        1, (total_elements + kIm2ColBlockSize - 1) / kIm2ColBlockSize));
    apply_activation_kernel<<<blocks, kIm2ColBlockSize, 0, stream>>>(y, total_elements, activation);
    cudaError_t act_err = cudaGetLastError();
  if (act_err != cudaSuccess) {
    fprintf(stderr,
            "[conv2d_stub] activation kernel failed: %d (%s) total=%lld\n",
            static_cast<int>(act_err),
            cudaGetErrorString(act_err),
            static_cast<long long>(total_elements));
    return FLAME_CUDA_ERR_CUDA;
  }
  }

  return FLAME_CUDA_OK;
}

extern "C" FlameCudaStatus flame_conv2d_autotune_get_stats_impl(FlameConv2dAutotuneStats* out) {
  if (out == nullptr) {
    return FLAME_CUDA_ERR_INVALID;
  }
  out->cache_hits = static_cast<uint64_t>(g_autotune_stats.cache_hits.load(std::memory_order_relaxed));
  out->cache_misses =
      static_cast<uint64_t>(g_autotune_stats.cache_misses.load(std::memory_order_relaxed));
  out->tuned = static_cast<uint64_t>(g_autotune_stats.tuned.load(std::memory_order_relaxed));
  out->weak = static_cast<uint64_t>(g_autotune_stats.weak.load(std::memory_order_relaxed));
  out->fallbacks = static_cast<uint64_t>(g_autotune_stats.fallbacks.load(std::memory_order_relaxed));
  out->workspace_skips =
      static_cast<uint64_t>(g_autotune_stats.workspace_skips.load(std::memory_order_relaxed));
  out->errors = static_cast<uint64_t>(g_autotune_stats.errors.load(std::memory_order_relaxed));
  out->reprobes = static_cast<uint64_t>(g_autotune_stats.reprobes.load(std::memory_order_relaxed));
  return FLAME_CUDA_OK;
}

extern "C" FlameCudaStatus flame_conv2d_autotune_reset_stats_impl(void) {
  g_autotune_stats.cache_hits.store(0, std::memory_order_relaxed);
  g_autotune_stats.cache_misses.store(0, std::memory_order_relaxed);
  g_autotune_stats.tuned.store(0, std::memory_order_relaxed);
  g_autotune_stats.weak.store(0, std::memory_order_relaxed);
  g_autotune_stats.fallbacks.store(0, std::memory_order_relaxed);
  g_autotune_stats.workspace_skips.store(0, std::memory_order_relaxed);
  g_autotune_stats.errors.store(0, std::memory_order_relaxed);
  g_autotune_stats.reprobes.store(0, std::memory_order_relaxed);
  return FLAME_CUDA_OK;
}
