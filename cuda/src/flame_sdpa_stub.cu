#include "flame_cuda_common.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <list>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <unistd.h>
#include <vector>

struct SdpaAutotuneStatsAtomic {
  uint64_t env_forced = 0;
  uint64_t clamped = 0;
  uint64_t skipped = 0;
  uint64_t fallback = 0;
  uint64_t errors = 0;
  uint64_t cache_hits = 0;
  uint64_t cache_misses = 0;
  uint64_t tuned = 0;
  uint64_t last_q_chunk = 0;
  uint64_t last_k_chunk = 0;
  uint64_t cache_saved = 0;
  uint64_t cache_loads = 0;
  uint64_t cache_load_errors = 0;
  uint64_t cache_entries = 0;
  uint64_t last_candidate_count = 0;
  uint64_t last_best_time_ns = 0;
  uint64_t last_plan_source = 0;
  uint64_t last_shape_b = 0;
  uint64_t last_shape_h = 0;
  uint64_t last_shape_q = 0;
  uint64_t last_shape_k = 0;
  uint64_t last_shape_dh = 0;
  uint64_t last_shape_dv = 0;
  uint64_t last_shape_mask_heads = 0;
  uint64_t last_shape_causal = 0;
};

struct SdpaAutotuneConfig {
  bool enabled = true;
  int qchunk_override = 0;
  bool verbose = false;
  int warmups = 1;
  int repeats = 3;
  int max_candidates = 4;
  double tie_epsilon = 0.02;
  bool persist_enabled = false;
  int flush_inserts = 200;
  int flush_interval_s = 120;
  bool flush_verbose = false;
  size_t cache_limit = 4096;
  std::string cache_path;
};

static SdpaAutotuneStatsAtomic g_sdpa_stats;
static constexpr int kSdpaAutotuneCacheVersion = 1;
struct SdpaPlanKey {
  int B;
  int H;
  int Q;
  int K;
  int Dh;
  int Dv;
  int mask_heads;
  int causal;

  bool operator==(const SdpaPlanKey& other) const {
    return B == other.B && H == other.H && Q == other.Q && K == other.K &&
           Dh == other.Dh && Dv == other.Dv && mask_heads == other.mask_heads &&
           causal == other.causal;
  }
};

struct SdpaPlanKeyHash {
  std::size_t operator()(const SdpaPlanKey& key) const {
    std::size_t h = 0;
    auto hash_combine = [&h](int value) {
      std::size_t v = static_cast<std::size_t>(value);
      h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    };
    hash_combine(key.B);
    hash_combine(key.H);
    hash_combine(key.Q);
    hash_combine(key.K);
    hash_combine(key.Dh);
    hash_combine(key.Dv);
    hash_combine(key.mask_heads);
    hash_combine(key.causal);
    return h;
  }
};

struct SdpaPlanDecision {
  int q_chunk = 0;
  int k_chunk = 0;
};

struct SdpaPlanEntry {
  SdpaPlanDecision decision;
  std::list<SdpaPlanKey>::iterator lru_it;
};

static const SdpaAutotuneConfig& get_sdpa_autotune_config();

static std::mutex g_sdpa_plan_mutex;
static std::unordered_map<SdpaPlanKey, SdpaPlanEntry, SdpaPlanKeyHash> g_sdpa_plan_cache;
static std::list<SdpaPlanKey> g_sdpa_plan_lru;
static bool g_sdpa_cache_loaded = false;
static bool g_sdpa_cache_dirty = false;
static std::once_flag g_sdpa_cache_flush_once;
static std::string g_sdpa_cache_path;
static size_t g_sdpa_new_entries_since_flush = 0;
static std::chrono::steady_clock::time_point g_sdpa_last_flush_time =
    std::chrono::steady_clock::time_point::min();

static void update_cache_entry_metrics_locked() {
  g_sdpa_stats.cache_entries = static_cast<uint64_t>(g_sdpa_plan_cache.size());
}

static std::string default_sdpa_cache_path() {
  const char* xdg_cache = std::getenv("XDG_CACHE_HOME");
  std::string base;
  if (xdg_cache && *xdg_cache) {
    base = xdg_cache;
  } else {
    const char* home = std::getenv("HOME");
    if (home && *home) {
      base = std::string(home) + "/.cache";
    }
  }
  if (base.empty()) {
    return std::string();
  }
  return base + "/flame/autotune_sdpa.json";
}

static bool ensure_parent_dir(const std::string& file_path) {
  auto slash = file_path.find_last_of('/');
  if (slash == std::string::npos) {
    return true;
  }
  std::string dir = file_path.substr(0, slash);
  if (dir.empty()) {
    return true;
  }

  size_t pos = 0;
  while (pos < dir.size()) {
    size_t next = dir.find('/', pos);
    std::string subdir = dir.substr(0, next);
    if (!subdir.empty()) {
      if (::mkdir(subdir.c_str(), 0755) != 0 && errno != EEXIST) {
        return false;
      }
    }
    if (next == std::string::npos) {
      break;
    }
    pos = next + 1;
  }
  return true;
}

static void record_last_plan(int B,
                             int H,
                             int Q,
                             int K,
                             int Dh,
                             int Dv,
                             int mask_heads,
                             int causal,
                             uint64_t candidate_count,
                             uint64_t best_time_ns,
                             uint64_t plan_source) {
  g_sdpa_stats.last_shape_b = static_cast<uint64_t>(std::max(B, 0));
  g_sdpa_stats.last_shape_h = static_cast<uint64_t>(std::max(H, 0));
  g_sdpa_stats.last_shape_q = static_cast<uint64_t>(std::max(Q, 0));
  g_sdpa_stats.last_shape_k = static_cast<uint64_t>(std::max(K, 0));
  g_sdpa_stats.last_shape_dh = static_cast<uint64_t>(std::max(Dh, 0));
  g_sdpa_stats.last_shape_dv = static_cast<uint64_t>(std::max(Dv, 0));
  g_sdpa_stats.last_shape_mask_heads = static_cast<uint64_t>(std::max(mask_heads, 0));
  g_sdpa_stats.last_shape_causal = static_cast<uint64_t>(std::max(causal, 0));
  g_sdpa_stats.last_candidate_count = candidate_count;
  g_sdpa_stats.last_best_time_ns = best_time_ns;
  g_sdpa_stats.last_plan_source = plan_source;
}

static void touch_plan_locked(const SdpaPlanKey& key) {
  auto it = g_sdpa_plan_cache.find(key);
  if (it == g_sdpa_plan_cache.end()) {
    return;
  }
  g_sdpa_plan_lru.splice(g_sdpa_plan_lru.end(), g_sdpa_plan_lru, it->second.lru_it);
  it->second.lru_it = std::prev(g_sdpa_plan_lru.end());
}

static void evict_over_limit_locked(size_t limit) {
  if (limit == 0) {
    return;
  }
  while (g_sdpa_plan_cache.size() > limit && !g_sdpa_plan_lru.empty()) {
    const SdpaPlanKey key = g_sdpa_plan_lru.front();
    g_sdpa_plan_cache.erase(key);
    g_sdpa_plan_lru.pop_front();
  }
  update_cache_entry_metrics_locked();
}

static bool insert_or_assign_plan_locked(const SdpaPlanKey& key,
                                         const SdpaPlanDecision& decision,
                                         const SdpaAutotuneConfig& config,
                                         bool count_new_entry) {
  bool inserted = false;
  auto it = g_sdpa_plan_cache.find(key);
  if (it != g_sdpa_plan_cache.end()) {
    it->second.decision = decision;
    g_sdpa_plan_lru.splice(g_sdpa_plan_lru.end(), g_sdpa_plan_lru, it->second.lru_it);
    it->second.lru_it = std::prev(g_sdpa_plan_lru.end());
  } else {
    g_sdpa_plan_lru.push_back(key);
    auto lru_it = std::prev(g_sdpa_plan_lru.end());
    g_sdpa_plan_cache.emplace(key, SdpaPlanEntry{decision, lru_it});
    inserted = true;
  }
  update_cache_entry_metrics_locked();
  evict_over_limit_locked(config.cache_limit);
  if (inserted && count_new_entry && config.persist_enabled && !g_sdpa_cache_path.empty()) {
    g_sdpa_new_entries_since_flush += 1;
  }
  return inserted;
}

static bool should_trigger_opportunistic_flush_locked(const SdpaAutotuneConfig& config) {
  if (!config.persist_enabled || g_sdpa_cache_path.empty()) {
    return false;
  }
  if (config.flush_inserts <= 0) {
    return false;
  }
  if (g_sdpa_new_entries_since_flush < static_cast<size_t>(config.flush_inserts)) {
    return false;
  }
  if (config.flush_interval_s <= 0) {
    return true;
  }
  auto now = std::chrono::steady_clock::now();
  if (g_sdpa_last_flush_time == std::chrono::steady_clock::time_point::min()) {
    return true;
  }
  auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - g_sdpa_last_flush_time);
  return elapsed.count() >= config.flush_interval_s;
}

static std::string compact_json(const std::string& line) {
  std::string out;
  out.reserve(line.size());
  for (char c : line) {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      continue;
    }
    out.push_back(c);
  }
  return out;
}

static long extract_json_long(const std::string& sanitized, const char* key) {
  const char* buffer = sanitized.c_str();
  const char* location = std::strstr(buffer, key);
  if (!location) {
    return std::numeric_limits<long>::min();
  }
  location += std::strlen(key);
  char* end = nullptr;
  long value = std::strtol(location, &end, 10);
  if (location == end) {
    return std::numeric_limits<long>::min();
  }
  return value;
}

static bool parse_sdpa_cache_json_line(const std::string& sanitized,
                                       SdpaPlanKey* key,
                                       SdpaPlanDecision* decision) {
  SdpaPlanKey parsed{};
  SdpaPlanDecision plan{};
  if (std::sscanf(
          sanitized.c_str(),
          "{\"key\":{\"B\":%d,\"H\":%d,\"Q\":%d,\"K\":%d,\"Dh\":%d,\"Dv\":%d,\"mask_heads\":%d,"
          "\"causal\":%d},\"plan\":{\"q\":%d,\"k\":%d}}",
          &parsed.B,
          &parsed.H,
          &parsed.Q,
          &parsed.K,
          &parsed.Dh,
          &parsed.Dv,
          &parsed.mask_heads,
          &parsed.causal,
          &plan.q_chunk,
          &plan.k_chunk) == 10) {
    *key = parsed;
    *decision = plan;
    return true;
  }
  return false;
}

static bool parse_sdpa_cache_line(const std::string& line,
                                  SdpaPlanKey* key,
                                  SdpaPlanDecision* decision) {
  const std::string sanitized = compact_json(line);
  if (!sanitized.empty() && sanitized[0] == '{' && sanitized.rfind("{\"meta\"", 0) != 0) {
    if (parse_sdpa_cache_json_line(sanitized, key, decision)) {
      return true;
    }
  }
  if (!line.empty() && line[0] == '#') {
    return false;
  }
  std::istringstream iss(line);
  SdpaPlanKey parsed{};
  SdpaPlanDecision plan{};
  if (iss >> parsed.B >> parsed.H >> parsed.Q >> parsed.K >> parsed.Dh >> parsed.Dv >>
      parsed.mask_heads >> parsed.causal >> plan.q_chunk >> plan.k_chunk) {
    *key = parsed;
    *decision = plan;
    return true;
  }
  return false;
}

static std::string json_escape(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  for (char c : value) {
    switch (c) {
      case '\"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += c;
        break;
    }
  }
  return out;
}

static std::string make_cache_meta_json(size_t entry_count) {
  int device = -1;
  int driver_version = 0;
  int runtime_version = 0;
  cudaGetDevice(&device);
  cudaDriverGetVersion(&driver_version);
  cudaRuntimeGetVersion(&runtime_version);

  int sm = 0;
  std::string device_name = "unknown";
  if (device >= 0) {
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, device) == cudaSuccess) {
      sm = props.major * 100 + props.minor;
      device_name = props.name ? props.name : "unknown";
    }
  }

  std::ostringstream oss;
  oss << "{\"meta\":{\"version\":" << kSdpaAutotuneCacheVersion << ",\"entries\":"
      << static_cast<unsigned long long>(entry_count) << ",\"device\":" << device
      << ",\"sm\":" << sm << ",\"driver\":" << driver_version << ",\"runtime\":"
      << runtime_version << ",\"name\":\"" << json_escape(device_name) << "\"}}\n";
  return oss.str();
}

static void read_sdpa_cache_locked(bool track_stats) {
  if (g_sdpa_cache_path.empty()) {
    return;
  }
  const SdpaAutotuneConfig& cfg = get_sdpa_autotune_config();
  std::ifstream in(g_sdpa_cache_path);
  if (!in.is_open()) {
    return;
  }
  if (track_stats) {
    g_sdpa_stats.cache_loads += 1;
  }

  std::string line;
  bool version_ok = true;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    std::string sanitized = compact_json(line);
    if (sanitized.empty()) {
      continue;
    }
    if (sanitized.rfind("{\"meta\"", 0) == 0) {
      long file_version = extract_json_long(sanitized, "\"version\":");
      if (file_version != std::numeric_limits<long>::min() &&
          file_version != kSdpaAutotuneCacheVersion) {
        version_ok = false;
        if (track_stats) {
          g_sdpa_stats.cache_load_errors += 1;
        }
      }
      continue;
    }
    if (!version_ok) {
      continue;
    }
    SdpaPlanKey key{};
    SdpaPlanDecision decision{};
    if (parse_sdpa_cache_line(line, &key, &decision) && decision.q_chunk > 0 &&
        decision.k_chunk > 0) {
      insert_or_assign_plan_locked(key, decision, cfg, false);
    } else if (track_stats) {
      g_sdpa_stats.cache_load_errors += 1;
    }
  }
  update_cache_entry_metrics_locked();
}

static void merge_sdpa_cache_from_disk_locked() {
  read_sdpa_cache_locked(false);
}

static void flush_sdpa_cache();

static void register_sdpa_cache_flush() {
  std::call_once(g_sdpa_cache_flush_once, []() { std::atexit(flush_sdpa_cache); });
}

static void load_sdpa_cache() {
  std::lock_guard<std::mutex> guard(g_sdpa_plan_mutex);
  if (g_sdpa_cache_loaded) {
    return;
  }
  g_sdpa_cache_loaded = true;

  const SdpaAutotuneConfig& cfg = get_sdpa_autotune_config();
  const char* explicit_path = std::getenv("FLAME_SDPA_AUTOTUNE_CACHE");
  if (explicit_path && *explicit_path) {
    g_sdpa_cache_path = explicit_path;
  } else if (cfg.persist_enabled) {
    g_sdpa_cache_path = cfg.cache_path;
  }

  if (cfg.persist_enabled && !g_sdpa_cache_path.empty()) {
    register_sdpa_cache_flush();
  }

  if (g_sdpa_cache_path.empty()) {
    return;
  }

  read_sdpa_cache_locked(cfg.persist_enabled);
}

static void flush_sdpa_cache() {
  const SdpaAutotuneConfig& cfg = get_sdpa_autotune_config();
  std::unique_lock<std::mutex> guard(g_sdpa_plan_mutex);
  if (!cfg.persist_enabled || g_sdpa_cache_path.empty() || !g_sdpa_cache_dirty) {
    return;
  }

  merge_sdpa_cache_from_disk_locked();
  update_cache_entry_metrics_locked();

  if (!ensure_parent_dir(g_sdpa_cache_path)) {
    return;
  }

  const std::string tmp_path = g_sdpa_cache_path + ".tmp";

  int lock_fd = ::open(g_sdpa_cache_path.c_str(), O_RDWR | O_CREAT, 0644);
  if (lock_fd < 0) {
    return;
  }
  if (::flock(lock_fd, LOCK_EX | LOCK_NB) != 0) {
    ::close(lock_fd);
    return;
  }

  int tmp_fd = ::open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (tmp_fd < 0) {
    ::flock(lock_fd, LOCK_UN);
    ::close(lock_fd);
    return;
  }

  auto write_all = [&](const std::string& payload) -> bool {
    const char* data = payload.data();
    size_t remaining = payload.size();
    while (remaining > 0) {
      ssize_t written = ::write(tmp_fd, data, remaining);
      if (written < 0) {
        if (errno == EINTR) {
          continue;
        }
        return false;
      }
      data += written;
      remaining -= static_cast<size_t>(written);
    }
    return true;
  };

  bool ok = write_all(make_cache_meta_json(g_sdpa_plan_cache.size()));
  if (ok) {
    for (const auto& entry : g_sdpa_plan_cache) {
      const auto& key = entry.first;
      const auto& decision = entry.second.decision;
      std::ostringstream line;
      line << "{\"key\":{\"B\":" << key.B << ",\"H\":" << key.H << ",\"Q\":" << key.Q << ",\"K\":"
           << key.K << ",\"Dh\":" << key.Dh << ",\"Dv\":" << key.Dv << ",\"mask_heads\":"
           << key.mask_heads << ",\"causal\":" << key.causal << "},\"plan\":{\"q\":"
           << decision.q_chunk << ",\"k\":" << decision.k_chunk << "}}\n";
      if (!write_all(line.str())) {
        ok = false;
        break;
      }
    }
  }

  if (ok && ::fsync(tmp_fd) != 0) {
    ok = false;
  }

  if (::close(tmp_fd) != 0) {
    ok = false;
  }

  if (!ok) {
    ::unlink(tmp_path.c_str());
    ::flock(lock_fd, LOCK_UN);
    ::close(lock_fd);
    return;
  }

  if (::rename(tmp_path.c_str(), g_sdpa_cache_path.c_str()) != 0) {
    ::unlink(tmp_path.c_str());
    ::flock(lock_fd, LOCK_UN);
    ::close(lock_fd);
    return;
  }

  ::fsync(lock_fd);
  ::flock(lock_fd, LOCK_UN);
  ::close(lock_fd);

  g_sdpa_cache_dirty = false;
  g_sdpa_new_entries_since_flush = 0;
  g_sdpa_last_flush_time = std::chrono::steady_clock::now();
  g_sdpa_stats.cache_saved += 1;
  update_cache_entry_metrics_locked();
}

static int parse_env_int(const char* name, int default_value) {
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

static bool parse_env_bool(const char* name, bool default_value) {
  const char* value = std::getenv(name);
  if (!value || *value == '\0') {
    return default_value;
  }
  char first = static_cast<char>(std::tolower(static_cast<unsigned char>(value[0])));
  if (first == '0' || first == 'f' || first == 'n') {
    return false;
  }
  if (first == '1' || first == 't' || first == 'y') {
    return true;
  }
  return default_value;
}

static const SdpaAutotuneConfig& get_sdpa_autotune_config() {
  static SdpaAutotuneConfig config = [] {
    SdpaAutotuneConfig cfg;
    cfg.enabled = parse_env_bool("FLAME_SDPA_AUTOTUNE", true);
    cfg.verbose = parse_env_bool("FLAME_SDPA_AUTOTUNE_VERBOSE", false);
    cfg.qchunk_override = std::max(0, parse_env_int("FLAME_SDPA_QCHUNK", 0));
    cfg.warmups = std::max(0, parse_env_int("FLAME_SDPA_AUTOTUNE_WARMUPS", cfg.warmups));
    cfg.repeats = std::max(1, parse_env_int("FLAME_SDPA_AUTOTUNE_REPEATS", cfg.repeats));
  cfg.max_candidates =
      std::max(1, parse_env_int("FLAME_SDPA_AUTOTUNE_MAX_CANDS", cfg.max_candidates));
  int tie_eps_pct = parse_env_int(
      "FLAME_SDPA_AUTOTUNE_TIE_EPS_PCT", static_cast<int>(cfg.tie_epsilon * 100));
  cfg.tie_epsilon = std::max(0.0, static_cast<double>(tie_eps_pct) / 100.0);
  cfg.persist_enabled = parse_env_bool("FLAME_SDPA_AUTOTUNE_PERSIST", false);
  cfg.flush_inserts = std::max(1, parse_env_int("FLAME_SDPA_FLUSH_INSERTS", cfg.flush_inserts));
  cfg.flush_interval_s =
      std::max(1, parse_env_int("FLAME_SDPA_FLUSH_INTERVAL_S", cfg.flush_interval_s));
  cfg.flush_verbose = parse_env_bool("FLAME_SDPA_FLUSH_VERBOSE", false);
  int cache_limit = parse_env_int(
      "FLAME_SDPA_AUTOTUNE_CACHE_LIMIT", static_cast<int>(cfg.cache_limit));
  cfg.cache_limit = static_cast<size_t>(std::max(1, cache_limit));
  const char* cache_env = std::getenv("FLAME_SDPA_AUTOTUNE_CACHE");
  if (cache_env && *cache_env) {
    cfg.cache_path = cache_env;
  } else if (cfg.persist_enabled) {
    cfg.cache_path = default_sdpa_cache_path();
  }
  if (cfg.persist_enabled && cfg.cache_path.empty()) {
    cfg.persist_enabled = false;
  }
  return cfg;
}();
return config;
}

struct ChunkSelection {
  int chunk_rows = 0;
  bool forced = false;
  bool clamped = false;
};

static ChunkSelection select_chunk_rows(int requested, int q_len, const SdpaAutotuneConfig& config) {
  ChunkSelection sel;
  int clamped = requested;
  if (clamped > q_len) {
    clamped = q_len;
  }
  if (clamped < 1) {
    clamped = 1;
  }
  sel.clamped = clamped != requested;
  int chosen = clamped;
  if (config.qchunk_override > 0) {
    int override_val = config.qchunk_override;
    if (override_val > requested) {
      override_val = requested;
    }
    if (override_val > q_len) {
      override_val = q_len;
    }
    if (override_val < 1) {
      override_val = 1;
    }
    chosen = override_val;
    sel.forced = true;
    sel.clamped = sel.clamped || chosen != requested;
  }
  sel.chunk_rows = chosen;
  return sel;
}

__device__ inline float bf16_to_f32(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

__device__ inline __nv_bfloat16 f32_to_bf16(float v) {
  return __float2bfloat16(v);
}

__global__ void qk_matmul_bf16_kernel(const __nv_bfloat16* __restrict__ q,
                                      const __nv_bfloat16* __restrict__ k,
                                      float* __restrict__ scores,
                                      int64_t rows,
                                      int64_t k_len,
                                      int64_t dh,
                                      int64_t q_stride_row,
                                      int64_t q_stride_d,
                                      int64_t k_stride_seq,
                                      int64_t k_stride_d,
                                      int64_t score_stride,
                                      int64_t k_offset) {
  const int64_t total = rows * k_len;
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < total) {
    const int64_t row = idx / k_len;
    const int64_t col = idx % k_len;

    const __nv_bfloat16* q_row = q + row * q_stride_row;
    const __nv_bfloat16* k_col = k + (k_offset + col) * k_stride_seq;

    float acc = 0.0f;
    for (int64_t d = 0; d < dh; ++d) {
      const float q_val = bf16_to_f32(q_row[d * q_stride_d]);
      const float k_val = bf16_to_f32(k_col[d * k_stride_d]);
      acc += q_val * k_val;
    }
    scores[row * score_stride + col] = acc;
    idx += static_cast<int64_t>(gridDim.x) * blockDim.x;
  }
}

__global__ void sdpa_reset_kernel(float* row_max,
                                  float* row_sum,
                                  float* row_out,
                                  int64_t rows,
                                  int64_t dv) {
  int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  row_max[row] = -FLT_MAX;
  row_sum[row] = 0.0f;
  float* out_row = row_out + row * dv;
  for (int64_t d = 0; d < dv; ++d) {
    out_row[d] = 0.0f;
  }
}

__global__ void sdpa_block_accumulate_kernel(float* scores,
                                             int64_t score_stride,
                                             int64_t rows,
                                             int64_t block_k,
                                             float scale,
                                             float* row_max,
                                             float* row_sum,
                                             float* row_out,
                                             int64_t dv,
                                             const __nv_bfloat16* v_head,
                                             int64_t v_stride_seq,
                                             int64_t v_stride_d,
                                             const __nv_bfloat16* attn_mask,
                                             int has_mask,
                                             int64_t mask_stride_b,
                                             int64_t mask_stride_head,
                                             int64_t mask_stride_q,
                                             int64_t mask_stride_k,
                                             int mask_heads,
                                             int causal,
                                             int64_t b_idx,
                                             int64_t h_idx,
                                             int64_t q_start,
                                             int64_t k_offset,
                                             int64_t Q,
                                             int64_t K) {
  int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  float* row_scores = scores + row * score_stride;
  float prev_max = row_max[row];
  float prev_sum = row_sum[row];
  float block_max = -CUDART_INF_F;

  const int64_t global_q = q_start + row;
  const __nv_bfloat16* mask_row = nullptr;
  if (has_mask) {
    const int mask_head = (mask_heads == 1) ? 0 : static_cast<int>(h_idx);
    mask_row = attn_mask + b_idx * mask_stride_b +
               mask_head * mask_stride_head +
               global_q * mask_stride_q +
               k_offset * mask_stride_k;
  }

  for (int64_t col = 0; col < block_k; ++col) {
    float val = row_scores[col] * scale;
    bool masked = false;
    if (causal && (k_offset + col) > global_q) {
      masked = true;
    }
    if (!masked && has_mask) {
      float m = bf16_to_f32(mask_row[col * mask_stride_k]);
      if (m < 0.5f) {
        masked = true;
      }
    }
    if (masked) {
      val = -CUDART_INF_F;
    }
    row_scores[col] = val;
    if (val > block_max) {
      block_max = val;
    }
  }

  if (block_k == 0) {
    return;
  }

  float new_max = fmaxf(prev_max, block_max);
  float prev_scale = (prev_sum > 0.0f) ? expf(prev_max - new_max) : 0.0f;
  float new_sum = prev_sum * prev_scale;

  for (int64_t col = 0; col < block_k; ++col) {
    float weight = __expf(row_scores[col] - new_max);
    row_scores[col] = weight;
    new_sum += weight;
  }

  float* out_row = row_out + row * dv;
  for (int64_t d = 0; d < dv; ++d) {
    out_row[d] *= prev_scale;
  }

  for (int64_t col = 0; col < block_k; ++col) {
    float weight = row_scores[col];
    const __nv_bfloat16* v_vec =
        v_head + (k_offset + col) * v_stride_seq;
    for (int64_t d = 0; d < dv; ++d) {
      float v_val = bf16_to_f32(v_vec[d * v_stride_d]);
      out_row[d] = fmaf(weight, v_val, out_row[d]);
    }
  }

  row_max[row] = new_max;
  row_sum[row] = new_sum;
}

__global__ void sdpa_finalize_kernel(const float* row_sum,
                                     const float* row_out,
                                     int64_t rows,
                                     int64_t dv,
                                     __nv_bfloat16* o_chunk,
                                     int64_t o_stride_q,
                                     int64_t o_stride_d) {
  int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  float sum = row_sum[row];
  if (!isfinite(sum) || sum <= 0.0f) {
    sum = 0.0f;
  }
  float norm = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
  const float* in_row = row_out + row * dv;
  __nv_bfloat16* out_row = o_chunk + row * o_stride_q;
  for (int64_t d = 0; d < dv; ++d) {
    out_row[d * o_stride_d] = f32_to_bf16(in_row[d] * norm);
  }
}


FlameCudaStatus flame_sdpa_chunked_bf16_impl(const __nv_bfloat16* q,
                                             const __nv_bfloat16* k,
                                             const __nv_bfloat16* v,
                                             int B,
                                             int H,
                                             int Q,
                                             int K,
                                             int Dh,
                                             int Dv,
                                             float scale,
                                             int chunk,
                                             int causal,
                                             int mask_heads,
                                             const __nv_bfloat16* attn_mask,
                                             __nv_bfloat16* out,
                                             void* workspace,
                                             size_t workspace_bytes,
                                             cudaStream_t stream) {
  if (!q || !k || !v || !out) {
    return FLAME_CUDA_ERR_INVALID;
  }
  if (B <= 0 || H <= 0 || Q < 0 || K < 0 || Dh <= 0 || Dv <= 0) {
    return FLAME_CUDA_ERR_INVALID;
  }
  if (Q == 0 || K == 0) {
    return FLAME_CUDA_OK;
  }
  if (chunk <= 0) {
    return FLAME_CUDA_ERR_INVALID;
  }

  const SdpaAutotuneConfig& autotune_cfg = get_sdpa_autotune_config();
  ChunkSelection chunk_sel = select_chunk_rows(chunk, Q, autotune_cfg);
  const bool forced_chunk = chunk_sel.forced;
  if (forced_chunk) {
    g_sdpa_stats.env_forced += 1;
  } else if (chunk_sel.clamped) {
    g_sdpa_stats.clamped += 1;
  }

  if (workspace == nullptr || !flame_is_aligned(workspace, alignof(float))) {
    g_sdpa_stats.errors += 1;
    return FLAME_CUDA_ERR_INVALID;
  }

  auto compute_max_cols_for_rows = [&](int64_t rows) -> int64_t {
    if (rows <= 0) {
      return 0;
    }
    int64_t total_floats = static_cast<int64_t>(workspace_bytes / sizeof(float));
    int64_t row_overhead = rows * (Dv + 2);
    if (total_floats <= row_overhead) {
      return 0;
    }
    int64_t capacity = (total_floats - row_overhead) / rows;
    if (capacity <= 0) {
      return 0;
    }
    return std::min<int64_t>(capacity, static_cast<int64_t>(K));
  };

  int64_t requested_rows =
      std::min<int64_t>(chunk_sel.chunk_rows, static_cast<int64_t>(Q));
  if (requested_rows <= 0) {
    g_sdpa_stats.errors += 1;
    return FLAME_CUDA_ERR_INVALID;
  }

  int64_t max_cols_global = compute_max_cols_for_rows(requested_rows);
  if (max_cols_global <= 0) {
    g_sdpa_stats.errors += 1;
    return FLAME_CUDA_ERR_INVALID;
  }

  const int64_t q_stride_d = 1;
  const int64_t q_stride_q = Dh;
  const int64_t q_stride_h = static_cast<int64_t>(Q) * Dh;
  const int64_t q_stride_b = static_cast<int64_t>(H) * q_stride_h;

  const int64_t k_stride_d = 1;
  const int64_t k_stride_seq = Dh;
  const int64_t k_stride_h = static_cast<int64_t>(K) * Dh;
  const int64_t k_stride_b = static_cast<int64_t>(H) * k_stride_h;

  const int64_t v_stride_d = 1;
  const int64_t v_stride_seq = Dv;
  const int64_t v_stride_h = static_cast<int64_t>(K) * Dv;
  const int64_t v_stride_b = static_cast<int64_t>(H) * v_stride_h;

  const int64_t o_stride_d = 1;
  const int64_t o_stride_q = Dv;
  const int64_t o_stride_h = static_cast<int64_t>(Q) * Dv;
  const int64_t o_stride_b = static_cast<int64_t>(H) * o_stride_h;

  const int has_mask = (attn_mask && mask_heads > 0) ? 1 : 0;
  const int64_t mask_stride_k = 1;
  const int64_t mask_stride_q = static_cast<int64_t>(K);
  const int64_t mask_stride_head = static_cast<int64_t>(Q) * K;
  const int64_t mask_stride_b = static_cast<int64_t>(mask_heads) * mask_stride_head;

  auto run_configuration = [&](int64_t q_chunk_rows, int64_t k_chunk_cols) -> FlameCudaStatus {
    if (q_chunk_rows <= 0 || k_chunk_cols <= 0) {
      return FLAME_CUDA_ERR_INVALID;
    }
    int64_t total_floats = static_cast<int64_t>(workspace_bytes / sizeof(float));
    int64_t required = q_chunk_rows * k_chunk_cols + q_chunk_rows * (Dv + 2);
    if (required > total_floats) {
      return FLAME_CUDA_ERR_INVALID;
    }

    float* scores = static_cast<float*>(workspace);
    float* row_max = scores + q_chunk_rows * k_chunk_cols;
    float* row_sum = row_max + q_chunk_rows;
    float* row_out = row_sum + q_chunk_rows;

    const int threads_reset = 128;

    for (int64_t b_idx = 0; b_idx < B; ++b_idx) {
      for (int64_t h_idx = 0; h_idx < H; ++h_idx) {
        const __nv_bfloat16* q_head = q + b_idx * q_stride_b + h_idx * q_stride_h;
        const __nv_bfloat16* k_head = k + b_idx * k_stride_b + h_idx * k_stride_h;
        const __nv_bfloat16* v_head = v + b_idx * v_stride_b + h_idx * v_stride_h;
        __nv_bfloat16* o_head = out + b_idx * o_stride_b + h_idx * o_stride_h;

        for (int64_t q_start = 0; q_start < Q; q_start += q_chunk_rows) {
          int64_t rows = std::min<int64_t>(q_chunk_rows, static_cast<int64_t>(Q) - q_start);
          if (rows <= 0) {
            break;
          }

          int64_t blocks_reset = (rows + threads_reset - 1) / threads_reset;
          sdpa_reset_kernel<<<blocks_reset, threads_reset, 0, stream>>>(
              row_max,
              row_sum,
              row_out,
              rows,
              Dv);
          FLAME_CUDA_TRY(cudaGetLastError());

          const __nv_bfloat16* q_chunk = q_head + q_start * q_stride_q;
          __nv_bfloat16* o_chunk = o_head + q_start * o_stride_q;

          for (int64_t k_offset = 0; k_offset < K; k_offset += k_chunk_cols) {
            int64_t block_k = std::min<int64_t>(k_chunk_cols, static_cast<int64_t>(K) - k_offset);
            if (block_k <= 0) {
              break;
            }

            const __nv_bfloat16* k_block = k_head + k_offset * k_stride_seq;

            int64_t total_qk = rows * block_k;
            int blocks_qk = static_cast<int>((total_qk + 127) / 128);
            blocks_qk = std::max(1, std::min(blocks_qk, 65535));

            qk_matmul_bf16_kernel<<<blocks_qk, 128, 0, stream>>>(
                q_chunk,
                k_block,
                scores,
                rows,
                block_k,
                Dh,
                q_stride_q,
                q_stride_d,
                k_stride_seq,
                k_stride_d,
                k_chunk_cols,
                k_offset);
            FLAME_CUDA_TRY(cudaGetLastError());

            sdpa_block_accumulate_kernel<<<rows, 1, 0, stream>>>(
                scores,
                k_chunk_cols,
                rows,
                block_k,
                scale,
                row_max,
                row_sum,
                row_out,
                Dv,
                v_head,
                v_stride_seq,
                v_stride_d,
                attn_mask,
                has_mask,
                mask_stride_b,
                mask_stride_head,
                mask_stride_q,
                mask_stride_k,
                mask_heads,
                causal,
                b_idx,
                h_idx,
                q_start,
                k_offset,
                Q,
                K);
            FLAME_CUDA_TRY(cudaGetLastError());
          }

          const int threads_final = 64;
          int64_t blocks_final = (rows + threads_final - 1) / threads_final;
          sdpa_finalize_kernel<<<blocks_final, threads_final, 0, stream>>>(
              row_sum,
              row_out,
              rows,
              Dv,
              o_chunk,
              o_stride_q,
              o_stride_d);
          FLAME_CUDA_TRY(cudaGetLastError());
        }
      }
    }

    return FLAME_CUDA_OK;
  };

  load_sdpa_cache();

  SdpaPlanKey plan_key{B, H, Q, K, Dh, Dv, mask_heads, causal};
  SdpaPlanDecision cached_decision{};
  bool cached_hit = false;
  if (autotune_cfg.enabled && !forced_chunk) {
    std::lock_guard<std::mutex> guard(g_sdpa_plan_mutex);
    auto it = g_sdpa_plan_cache.find(plan_key);
    if (it != g_sdpa_plan_cache.end()) {
      cached_decision = it->second.decision;
      touch_plan_locked(plan_key);
      cached_hit = true;
    }
  }
  if (autotune_cfg.enabled && !forced_chunk) {
    if (cached_hit) {
      g_sdpa_stats.cache_hits += 1;
    } else {
      g_sdpa_stats.cache_misses += 1;
    }
  }

  auto validate_decision = [&](SdpaPlanDecision decision) -> bool {
    if (decision.q_chunk <= 0 || decision.k_chunk <= 0) {
      return false;
    }
    decision.q_chunk = std::min<int>(decision.q_chunk, static_cast<int>(chunk_sel.chunk_rows));
    int64_t max_cols = compute_max_cols_for_rows(decision.q_chunk);
    if (max_cols <= 0) {
      return false;
    }
    if (decision.k_chunk > max_cols) {
      decision.k_chunk = static_cast<int>(max_cols);
    }
    cached_decision = decision;
    return decision.q_chunk > 0 && decision.k_chunk > 0;
  };

  if (cached_hit && !validate_decision(cached_decision)) {
    cached_hit = false;
  }

  int64_t best_q = requested_rows;
  int64_t best_k = max_cols_global;
  bool tuned = false;
  uint64_t candidate_count = 0;
  uint64_t best_time_ns = 0;
  uint64_t plan_source = cached_hit ? 1 : 3;

  if (cached_hit) {
    best_q = cached_decision.q_chunk;
    best_k = std::min<int64_t>(cached_decision.k_chunk, compute_max_cols_for_rows(best_q));
    plan_source = 1;  // cache hit
  } else if (!forced_chunk && autotune_cfg.enabled) {
    struct Candidate {
      int64_t q;
      int64_t k;
    };
    std::vector<Candidate> candidates;

    std::vector<int64_t> preferred = {512, 384, 256, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 2, 1};

    auto push_candidate = [&](int64_t q_val, int64_t k_val) {
      if (q_val <= 0 || q_val > chunk_sel.chunk_rows) {
        return;
      }
      int64_t max_cols = compute_max_cols_for_rows(q_val);
      if (max_cols <= 0) {
        return;
      }
      if (k_val > max_cols) {
        k_val = max_cols;
      }
      if (k_val <= 0) {
        return;
      }
      for (const auto& cand : candidates) {
        if (cand.q == q_val && cand.k == k_val) {
          return;
        }
      }
      candidates.push_back({q_val, k_val});
    };

    std::vector<int64_t> q_candidates;
    auto push_q = [&](int64_t value) {
      if (value <= 0 || value > chunk_sel.chunk_rows) {
        return;
      }
      if (std::find(q_candidates.begin(), q_candidates.end(), value) != q_candidates.end()) {
        return;
      }
      q_candidates.push_back(value);
    };
    push_q(requested_rows);
    int64_t current_q = requested_rows;
    while (current_q > 1 &&
           q_candidates.size() < static_cast<size_t>(autotune_cfg.max_candidates)) {
      current_q = std::max<int64_t>(1, current_q / 2);
      push_q(current_q);
      if (current_q == 1) {
        break;
      }
    }
    for (int64_t val : preferred) {
      if (q_candidates.size() >= static_cast<size_t>(autotune_cfg.max_candidates)) {
        break;
      }
      if (val < requested_rows) {
        push_q(val);
      }
    }

    for (int64_t q_candidate : q_candidates) {
      int64_t max_cols = compute_max_cols_for_rows(q_candidate);
      if (max_cols <= 0) {
        continue;
      }
      std::vector<int64_t> k_candidates;
      auto push_k = [&](int64_t value) {
        if (value <= 0 || value > max_cols) {
          return;
        }
        if (std::find(k_candidates.begin(), k_candidates.end(), value) != k_candidates.end()) {
          return;
        }
        k_candidates.push_back(value);
      };
      push_k(max_cols);
      int64_t current_k = max_cols;
      while (current_k > 1 &&
             k_candidates.size() < static_cast<size_t>(autotune_cfg.max_candidates)) {
        current_k = std::max<int64_t>(1, current_k / 2);
        push_k(current_k);
        if (current_k == 1) {
          break;
        }
      }
      for (int64_t val : preferred) {
        if (k_candidates.size() >= static_cast<size_t>(autotune_cfg.max_candidates)) {
          break;
        }
        if (val < max_cols) {
          push_k(val);
        }
      }
      for (int64_t k_candidate : k_candidates) {
        push_candidate(q_candidate, k_candidate);
        if (candidates.size() >= static_cast<size_t>(autotune_cfg.max_candidates)) {
          break;
        }
      }
      if (candidates.size() >= static_cast<size_t>(autotune_cfg.max_candidates)) {
        break;
      }
    }

    if (candidates.empty()) {
      candidates.push_back({requested_rows, max_cols_global});
    }

    candidate_count = static_cast<uint64_t>(candidates.size());
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    if (cudaEventCreate(&start_event) == cudaSuccess &&
        cudaEventCreate(&stop_event) == cudaSuccess) {
      float best_time_ms = std::numeric_limits<float>::infinity();
      for (const auto& cand : candidates) {
        bool failed = false;
        for (int w = 0; w < autotune_cfg.warmups; ++w) {
          if (run_configuration(cand.q, cand.k) != FLAME_CUDA_OK) {
            g_sdpa_stats.errors += 1;
            failed = true;
            break;
          }
        }
        if (failed) {
          continue;
        }
        if (cudaEventRecord(start_event, stream) != cudaSuccess) {
          g_sdpa_stats.errors += 1;
          continue;
        }
        FlameCudaStatus status = FLAME_CUDA_OK;
        for (int r = 0; r < autotune_cfg.repeats; ++r) {
          status = run_configuration(cand.q, cand.k);
          if (status != FLAME_CUDA_OK) {
            g_sdpa_stats.errors += 1;
            failed = true;
            break;
          }
        }
        if (failed) {
          continue;
        }
        if (cudaEventRecord(stop_event, stream) != cudaSuccess) {
          g_sdpa_stats.errors += 1;
          continue;
        }
        if (cudaEventSynchronize(stop_event) != cudaSuccess) {
          g_sdpa_stats.errors += 1;
          continue;
        }
        float elapsed_ms = 0.0f;
        if (cudaEventElapsedTime(&elapsed_ms, start_event, stop_event) != cudaSuccess) {
          g_sdpa_stats.errors += 1;
          continue;
        }
        float avg_ms = elapsed_ms / static_cast<float>(std::max(1, autotune_cfg.repeats));
        const float tie_eps = static_cast<float>(autotune_cfg.tie_epsilon);
        if (avg_ms + tie_eps < best_time_ms) {
          tuned = true;
          best_time_ms = avg_ms;
          best_q = cand.q;
          best_k = cand.k;
        } else if (std::fabs(avg_ms - best_time_ms) <= tie_eps &&
                   (cand.q > best_q || (cand.q == best_q && cand.k > best_k))) {
          tuned = true;
          best_q = cand.q;
          best_k = cand.k;
        }
      }
      cudaEventDestroy(start_event);
      cudaEventDestroy(stop_event);
      if (!tuned) {
        best_q = requested_rows;
        best_k = max_cols_global;
      } else {
        best_time_ns =
            static_cast<uint64_t>(std::max(0.0f, best_time_ms) * 1.0e6f);  // ms -> ns
      }
    }

    if (tuned && autotune_cfg.enabled) {
      bool should_flush = false;
      {
        std::lock_guard<std::mutex> guard(g_sdpa_plan_mutex);
        g_sdpa_stats.tuned += 1;
        insert_or_assign_plan_locked(
            plan_key,
            SdpaPlanDecision{static_cast<int>(best_q), static_cast<int>(best_k)},
            autotune_cfg,
            true);
        g_sdpa_cache_dirty = true;
        if (autotune_cfg.persist_enabled && !g_sdpa_cache_path.empty()) {
          register_sdpa_cache_flush();
        }
        should_flush = should_trigger_opportunistic_flush_locked(autotune_cfg);
      }
      if (should_flush) {
        flush_sdpa_cache();
      }
      plan_source = 2;  // freshly tuned
    }
  }

  if (!cached_hit && autotune_cfg.enabled && !forced_chunk && !tuned) {
    bool should_flush = false;
    {
      std::lock_guard<std::mutex> guard(g_sdpa_plan_mutex);
      insert_or_assign_plan_locked(
          plan_key,
          SdpaPlanDecision{static_cast<int>(best_q), static_cast<int>(best_k)},
          autotune_cfg,
          true);
      g_sdpa_cache_dirty = true;
      if (autotune_cfg.persist_enabled && !g_sdpa_cache_path.empty()) {
        register_sdpa_cache_flush();
      }
      should_flush = should_trigger_opportunistic_flush_locked(autotune_cfg);
    }
    if (should_flush) {
      flush_sdpa_cache();
    }
    plan_source = 3;  // heuristic / fallback
  }

  if (forced_chunk) {
    plan_source = 4;  // forced via env override
  }

  FlameCudaStatus final_status = run_configuration(best_q, best_k);
  if (final_status != FLAME_CUDA_OK) {
    g_sdpa_stats.errors += 1;
    return final_status;
  }

  g_sdpa_stats.last_q_chunk = static_cast<uint64_t>(best_q);
  g_sdpa_stats.last_k_chunk = static_cast<uint64_t>(best_k);
  record_last_plan(
      B, H, Q, K, Dh, Dv, mask_heads, causal, candidate_count, best_time_ns, plan_source);

  return FLAME_CUDA_OK;
}


FlameCudaStatus flame_sdpa_autotune_get_stats_impl(FlameSdpaAutotuneStats* out) {
  if (out == nullptr) {
    return FLAME_CUDA_ERR_INVALID;
  }
  out->env_forced = g_sdpa_stats.env_forced;
  out->clamped = g_sdpa_stats.clamped;
  out->skipped = g_sdpa_stats.skipped;
  out->fallback = g_sdpa_stats.fallback;
  out->errors = g_sdpa_stats.errors;
  out->cache_hits = g_sdpa_stats.cache_hits;
  out->cache_misses = g_sdpa_stats.cache_misses;
  out->tuned = g_sdpa_stats.tuned;
  out->last_q_chunk = g_sdpa_stats.last_q_chunk;
  out->last_k_chunk = g_sdpa_stats.last_k_chunk;
  out->cache_saved = g_sdpa_stats.cache_saved;
  out->cache_loads = g_sdpa_stats.cache_loads;
  out->cache_load_errors = g_sdpa_stats.cache_load_errors;
  out->cache_entries = g_sdpa_stats.cache_entries;
  out->last_candidate_count = g_sdpa_stats.last_candidate_count;
  out->last_best_time_ns = g_sdpa_stats.last_best_time_ns;
  out->last_plan_source = g_sdpa_stats.last_plan_source;
  out->last_shape_b = g_sdpa_stats.last_shape_b;
  out->last_shape_h = g_sdpa_stats.last_shape_h;
  out->last_shape_q = g_sdpa_stats.last_shape_q;
  out->last_shape_k = g_sdpa_stats.last_shape_k;
  out->last_shape_dh = g_sdpa_stats.last_shape_dh;
  out->last_shape_dv = g_sdpa_stats.last_shape_dv;
  out->last_shape_mask_heads = g_sdpa_stats.last_shape_mask_heads;
  out->last_shape_causal = g_sdpa_stats.last_shape_causal;
  return FLAME_CUDA_OK;
}

FlameCudaStatus flame_sdpa_autotune_reset_stats_impl(void) {
  g_sdpa_stats.env_forced = 0;
  g_sdpa_stats.clamped = 0;
  g_sdpa_stats.skipped = 0;
  g_sdpa_stats.fallback = 0;
  g_sdpa_stats.errors = 0;
  g_sdpa_stats.cache_hits = 0;
  g_sdpa_stats.cache_misses = 0;
  g_sdpa_stats.tuned = 0;
  g_sdpa_stats.last_q_chunk = 0;
  g_sdpa_stats.last_k_chunk = 0;
  g_sdpa_stats.cache_saved = 0;
  g_sdpa_stats.cache_loads = 0;
  g_sdpa_stats.cache_load_errors = 0;
  g_sdpa_stats.cache_entries = 0;
  g_sdpa_stats.last_candidate_count = 0;
  g_sdpa_stats.last_best_time_ns = 0;
  g_sdpa_stats.last_plan_source = 0;
  g_sdpa_stats.last_shape_b = 0;
  g_sdpa_stats.last_shape_h = 0;
  g_sdpa_stats.last_shape_q = 0;
  g_sdpa_stats.last_shape_k = 0;
  g_sdpa_stats.last_shape_dh = 0;
  g_sdpa_stats.last_shape_dv = 0;
  g_sdpa_stats.last_shape_mask_heads = 0;
  g_sdpa_stats.last_shape_causal = 0;
  return FLAME_CUDA_OK;
}

FlameCudaStatus flame_sdpa_autotune_flush_cache_impl(void) {
  flush_sdpa_cache();
  return FLAME_CUDA_OK;
}
