// Origin: flame-core dispatch refactor — Phase 1 (TensorIterator geometry cache).
// Reference: plan `/home/alex/.claude/plans/same-as-pytorch-way-nifty-engelbart.md`.
//
// Purpose: skip the broadcast / reorder / coalesce / 32bit-indexing pipeline in
// `TensorIteratorConfig::build` when the geometry is fully determined by the
// (operand shapes, operand strides, operand dtypes, pending-output mask,
// config-flag tuple). Step 1 (populate_operands) and step 5
// (allocate_or_resize_outputs) still run per call because they reference live
// tensor pointers and per-call output allocation; everything that's a pure
// function of the cache key is replayed from the cached `CachedIterGeometry`.
//
// Cache shape pattern mirrors `flame-core/src/cudnn/conv3d.rs`'s `ALGO_CACHE`:
// `Lazy<Mutex<HashMap<Key, Value>>>` with `once_cell`. Concurrency is not the
// Phase 1 bottleneck — Mutex<HashMap> is sufficient.
//
// Rollback knob: `FLAME_TI_CACHE_DISABLE=1` env var short-circuits the lookup
// in `config.rs::build`, restoring the legacy full-pipeline behavior. Used by
// the skeptic + bug-fixer to validate.

use crate::DType;
use once_cell::sync::Lazy;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use super::base::FastSetupType;
use super::dim_vec::{DimVec, I64StrideVec};

/// Bound on cache size. Production workloads have ~50-200 unique
/// (shape, op) combos per training step; 4096 is a generous headroom.
/// When exceeded, the cache is cleared wholesale.
pub(crate) const CACHE_CAPACITY: usize = 4096;

/// Per-operand shape vector used as part of the cache key. SmallVec<[usize; 6]>
/// matches DimVec / Strides element types so we don't need to convert at lookup.
pub(crate) type KeyShape = SmallVec<[usize; 6]>;

/// Per-operand stride vector (element strides, as returned by `Tensor::strides()`).
/// Distinct alias from `KeyShape` only for documentation at call sites.
pub(crate) type KeyStrides = SmallVec<[usize; 6]>;

/// Cache key. Captures everything the geometry pipeline depends on:
///
///   - Operand shapes (post-broadcast input source shapes).
///   - Operand element-strides (Tensor::strides() values; broadcast dims
///     keep their natural source strides).
///   - Operand dtypes (encoded as `DType as u8` to keep the key compact + Hash).
///   - Pending-output mask: bit `i` is set iff operand `i` is a pending
///     output (i.e. `OperandSrc == None`). For pending slots we don't carry
///     shape/strides (they're empty SmallVecs in the per-operand fields).
///   - Flag bundle: relevant `TensorIteratorConfig` booleans packed into a u32.
///     Only flags that influence shape/stride computation belong here. Flags
///     that gate Step 1 (dtype check) or Step 5 (allocation) are still
///     replayed per-call, so they don't need to participate in the key.
///   - num_outputs, static_dtype, static_shape — all directly affect geometry.
#[derive(Hash, PartialEq, Eq, Clone)]
pub(crate) struct IterCacheKey {
    pub operand_shapes: SmallVec<[KeyShape; 4]>,
    pub operand_strides: SmallVec<[KeyStrides; 4]>,
    pub operand_dtypes: SmallVec<[u8; 4]>,
    pub pending_output_mask: u8,
    pub num_outputs: u8,
    pub static_dtype: Option<u8>,
    pub static_shape: Option<KeyShape>,
    pub flags: u32,
}

/// Bit positions for `IterCacheKey::flags`. Only the flags that influence
/// shape/stride geometry are tracked here (resize_outputs gates step 2's
/// "skip outputs in shape inference" branch; is_reduction skips the empty-
/// operands error in compute_shape). `check_all_same_dtype` does NOT belong
/// here because it gates a step-1 validation that re-runs per call from the
/// live tensors anyway.
pub(crate) mod flag_bits {
    pub const RESIZE_OUTPUTS: u32 = 1 << 0;
    pub const IS_REDUCTION: u32 = 1 << 1;
    pub const PROMOTE_INPUTS: u32 = 1 << 2;
    pub const STATIC_DEVICE_DECLARED: u32 = 1 << 3;
    pub const CAST_COMMON_DTYPE_TO_OUTPUTS: u32 = 1 << 4;
    pub const ENFORCE_SAFE_CASTING: u32 = 1 << 5;
}

/// Cached geometry — everything `TensorIteratorBase::build` computes that's a
/// pure function of the cache key.
///
/// On hit, `TensorIteratorConfig::build` skips Steps 2-4 + 6-7 (broadcast /
/// strides / reorder / coalesce / 32bit-indexing) and replays this snapshot.
/// Step 1 (populate_operands) still runs per call to wire `src` slots into
/// live tensors; Step 5 (allocate_or_resize_outputs) still runs per call to
/// allocate fresh output tensors against `logical_output_shape`.
///
/// Fields are post-coalesce except `logical_output_shape` which is the
/// shape allocate-or-resize verifies / allocates against (PyTorch's
/// `inverted_shape` = `invert_perm(post-reorder shape)`). Caching it
/// separately avoids having to retain the pre-coalesce shape just for
/// the allocator.
#[derive(Clone)]
pub(crate) struct CachedIterGeometry {
    /// Post-coalesce iteration shape (`base.shape_`).
    pub shape: DimVec,
    /// Permutation from reorder. Invalid after coalesce per PyTorch
    /// convention; cached as-was-at-reorder for any consumer that still
    /// inspects it (build_iter_metadata + a few accessors don't).
    pub perm: DimVec,
    /// Logical (pre-reorder, pre-coalesce) output shape, used by step 5
    /// to allocate / verify outputs. Equivalent to `invert_perm(shape@reorder)`.
    pub logical_output_shape: DimVec,
    /// Per-operand byte-strides, post-coalesce, in iterator (post-reorder)
    /// frame.
    pub stride_bytes: SmallVec<[I64StrideVec; 4]>,
    pub has_coalesced_dimensions: bool,
    pub all_ops_same_shape: bool,
    pub requires_32bit_indexing: bool,
    pub common_dtype: Option<DType>,
    pub static_dtype: Option<DType>,
    pub fast_setup: FastSetupType,
    pub target_dtypes: SmallVec<[DType; 4]>,
}

/// The global cache. `Lazy<Mutex<HashMap<...>>>` per the conv3d AlgoKey pattern.
pub(crate) fn cache() -> &'static Mutex<HashMap<IterCacheKey, CachedIterGeometry>> {
    static CACHE: Lazy<Mutex<HashMap<IterCacheKey, CachedIterGeometry>>> =
        Lazy::new(|| Mutex::new(HashMap::new()));
    &CACHE
}

/// Hit/miss counters for `log::trace!` observability. Atomic so we don't take
/// the mutex on the counter update path.
pub(crate) static CACHE_HITS: AtomicU64 = AtomicU64::new(0);
pub(crate) static CACHE_MISSES: AtomicU64 = AtomicU64::new(0);

/// Returns true if `FLAME_TI_CACHE_DISABLE` is set in the environment.
/// Checked once per process via `OnceLock` to avoid the syscall on every build.
pub(crate) fn cache_disabled() -> bool {
    use std::sync::OnceLock;
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var("FLAME_TI_CACHE_DISABLE").is_ok())
}

/// Bump the hits counter and (rate-limited) emit a trace log. Called on cache hit.
#[inline]
pub(crate) fn note_hit() {
    let hits = CACHE_HITS.fetch_add(1, Ordering::Relaxed) + 1;
    // Rate-limit the log: every 4096 hits emit a line. The hot path is the
    // training inner loop, we don't want a log call per op.
    if hits & 0xFFF == 0 {
        let misses = CACHE_MISSES.load(Ordering::Relaxed);
        let entries = cache().lock().map(|c| c.len()).unwrap_or(0);
        log::trace!(
            "tensor_iterator cache: hits={} misses={} entries={}",
            hits,
            misses,
            entries
        );
    }
}

/// Insert a freshly-computed geometry into the cache. Caller has already
/// computed the key (cheap to construct, expensive to recompute on hit).
/// Bumps the misses counter and (rate-limited) emits a trace log.
pub(crate) fn insert(key: IterCacheKey, value: CachedIterGeometry) {
    let misses = CACHE_MISSES.fetch_add(1, Ordering::Relaxed) + 1;
    if let Ok(mut cache) = cache().lock() {
        if cache.len() > CACHE_CAPACITY {
            // Drop the entire cache when oversized. A more sophisticated LRU
            // is not warranted at Phase 1 — production runs settle into a
            // working-set well below 4096.
            cache.clear();
            log::trace!(
                "tensor_iterator cache: capacity {} exceeded, cleared",
                CACHE_CAPACITY
            );
        }
        cache.insert(key, value);
    }
    if misses & 0x3FF == 0 {
        let hits = CACHE_HITS.load(Ordering::Relaxed);
        let entries = cache().lock().map(|c| c.len()).unwrap_or(0);
        log::trace!(
            "tensor_iterator cache: hits={} misses={} entries={}",
            hits,
            misses,
            entries
        );
    }
}

/// Look up a key in the cache. Clones the cached geometry on hit (cheap —
/// SmallVecs on the stack for typical operand counts).
pub(crate) fn lookup(key: &IterCacheKey) -> Option<CachedIterGeometry> {
    let cache = cache().lock().ok()?;
    cache.get(key).cloned()
}
