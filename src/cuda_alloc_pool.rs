//! CUDA caching allocator for flame-core.
//!
//! Eliminates per-op `cudaMalloc`/`cudaFree` during backward by maintaining
//! power-of-2 bucketed free lists of GPU memory. Same strategy as PyTorch's
//! `CUDACachingAllocator`, simplified for single-device use.
//!
//! Integration: [`alloc_aligned_f32`](crate::cuda_memory_alignment::alloc_aligned_f32)
//! routes through [`pool_alloc_f32`], and [`Tensor::drop`](crate::tensor::Tensor)
//! returns slices via [`pool_return_f32`].

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DeviceSlice};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

// ---------------------------------------------------------------------------
// Mirror struct matching cudarc 0.11.x CudaSlice<T> layout.
//
// CudaSlice<T> is:
//   cu_device_ptr: CUdeviceptr (u64),
//   len: usize,
//   device: Arc<CudaDevice>,
//   host_buf: Option<Pin<Vec<T>>>,
//
// We reconstruct CudaSlice from raw parts via transmute.  This is safe as
// long as the struct layout hasn't changed (pinned to cudarc 0.11.9).
// ---------------------------------------------------------------------------
// Must NOT be #[repr(C)] — must match CudaSlice's default Rust layout.
struct CudaSliceMirror<T> {
    cu_device_ptr: u64,
    len: usize,
    device: Arc<CudaDevice>,
    host_buf: Option<std::pin::Pin<Vec<T>>>,
}

/// Entry stored in the free list — raw device pointer + metadata.
struct FreeEntry {
    ptr: u64,
    len: usize, // element count (f32 elements, not bytes)
    device: Arc<CudaDevice>,
    /// True if this entry's backing memory is owned by an external allocator
    /// (e.g. `ring_alloc::RingAllocator`) installed via
    /// [`install_miss_allocator`]. External entries skip `cudaFree` on
    /// `clear_cache` / pool drop; their lifecycle is the external allocator's
    /// responsibility. See "Ring allocator as pool backend" in
    /// `docs/FLAME_CONVENTIONS.md` for the lifecycle contract.
    is_external: bool,
}

/// Cached env check for FLAME_PROFILE=1.
#[inline]
fn profiling_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("FLAME_PROFILE").ok().as_deref() == Some("1"))
}

/// Cached env check for FLAME_ALLOC_POOL=0 (disable pool).
#[inline]
pub fn pool_disabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("FLAME_ALLOC_POOL").ok().as_deref() == Some("0"))
}

/// Cached env check for FLAME_F32_ZERO_INIT=1 (opt-in: restore legacy
/// behavior where pool_alloc_f32 zero-initializes the buffer on cache miss).
///
/// **Default = OFF (uninitialized).** This matches the BF16 path
/// (`pool_alloc_u16` already returns uninitialized memory) and PyTorch's
/// BFCAllocator semantics: callers are responsible for initialization.
///
/// Audit (2026-05-12): every caller of `alloc_aligned_f32` /
/// `pool_alloc_f32` in flame-core either (a) explicitly memsets afterward
/// via `alloc_zeros_from_pool` / `TensorStorage::zeros` (their previous
/// implicit zero was redundant), or (b) fully overwrites the buffer via
/// `dtod_copy` / `htod_copy_into` / a kernel that writes every element.
/// The implicit zero-init was wasted work in every case.
///
/// Set `FLAME_F32_ZERO_INIT=1` to revert to legacy behavior if a hidden
/// caller is discovered.
#[inline]
fn f32_zero_init_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("FLAME_F32_ZERO_INIT").ok().as_deref() == Some("1"))
}

/// Cached env check for `FLAME_F32_POOL_CACHE=1` (opt-in: enable the F32
/// free-list caching path in `pool_alloc_f32`).
///
/// **Default = OFF.** The F32 free-list has a documented stale-ptr re-use
/// bug across alloc generations that crashes Klein 9B + --offload at step
/// 2-13 with `CUDA_ERROR_INVALID_VALUE` (see Skeptic Phase 2c diagnosis).
/// Disabling the cache routes F32 allocs directly to cudart `device.alloc`
/// and `cudaFree` on drop, bypassing the free-list entirely. The BF16
/// path (`pool_alloc_u16`) is gated independently by
/// [`bf16_pool_cache_enabled`].
#[inline]
fn f32_pool_cache_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("FLAME_F32_POOL_CACHE").ok().as_deref() == Some("1"))
}

/// Cached env check for `FLAME_BF16_POOL_CACHE=1` (opt-in: enable the BF16
/// free-list caching path in `pool_alloc_u16`).
///
/// **Default = OFF.** Phase 2 smoke gate (2026-05-15) showed that the
/// Klein 9B step-13 `cuMemcpy2DAsync_v2 (cat) failed: CUDA_ERROR_INVALID_VALUE`
/// crash reproduces with `FLAME_F32_POOL_CACHE=0` (F32 cache OFF), proving
/// the bug surface is NOT exclusive to F32. The BF16 free-list has the
/// identical code shape and the same stale-ptr re-use class. Routing BF16
/// directly to cudart eliminates the crash class entirely.
///
/// Cost: BF16 transient allocations (Tensor::cat outputs, intermediate
/// casts, scratch buffers) lose free-list reuse. cudart's own mempool
/// still provides allocator-level reuse, so the perf cost is bounded by
/// the cudart driver's allocation churn — measured in Phase 2 smoke.
#[inline]
fn bf16_pool_cache_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("FLAME_BF16_POOL_CACHE").ok().as_deref() == Some("1"))
}

/// Cached env check for `FLAME_POOL_TRACE_BF16=1` — emits a per-event
/// trace line to stderr (op kind, ptr, bucket, size) for every BF16
/// alloc, return, push, and try_pop. Used to forensically reconstruct
/// the alloc history of a failing pointer.
#[inline]
fn pool_trace_bf16_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("FLAME_POOL_TRACE_BF16").ok().as_deref() == Some("1"))
}

/// Per-event log line: `[BF16-POOL] op=<kind> ptr=0x<hex> bucket=<u> size=<u> ext=<bool> tag=<...>`
/// Cheap (uses Atomic counters + eprintln). Stripped when env unset.
#[inline]
fn trace_bf16(op: &str, ptr: u64, bucket: usize, size: usize, ext: bool, tag: &str) {
    if pool_trace_bf16_enabled() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        eprintln!(
            "[BF16-POOL] seq={} op={} ptr=0x{:x} bucket={} size={} ext={} tag={}",
            n, op, ptr, bucket, size, ext, tag
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// LIVE-PTR TRAP (2026-05-15)
//
// Records every BF16 alloc / return / cache-clear event with a sequence
// number and last-known state. `trap_validate_bf16_ptr(ptr, site)` queries
// the trap right before passing a ptr to a CUDA API like cuMemcpy2D —
// when the trap is on and the ptr is non-Live, panics with provenance
// instead of getting an opaque `CUDA_ERROR_INVALID_VALUE`. This is the
// soul.md-pattern trap for the Klein 9B step-13 crash.
//
// Activated via `FLAME_POOL_TRAP_BF16=1`. Default OFF (per-op cost is a
// single OnceLock atomic load).
// ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PtrState {
    Live,
    InCache,
    Freed,
}

#[derive(Clone, Debug)]
struct PtrEvent {
    seq: u64,
    state: PtrState,
    op: &'static str,
    bucket: usize,
    /// Backtrace of the call site for this event. Captured when
    /// `FLAME_POOL_TRAP_BACKTRACE=1` and the event marks the ptr non-live.
    /// Cheap when off (None).
    bt: Option<std::sync::Arc<String>>,
}

#[derive(Clone, Debug)]
struct PtrHistory {
    state: PtrState,
    last_seq: u64,
    last_op: &'static str,
    bucket: usize,
    event_count: u32,
    /// Last N events for forensics (newest at end).
    events: Vec<PtrEvent>,
}

const TRAP_HISTORY_LEN: usize = 16;

#[derive(Clone, Debug)]
struct LiveRange {
    start: u64,
    end: u64,
}

#[inline]
fn pool_trap_bf16_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("FLAME_POOL_TRAP_BF16").ok().as_deref() == Some("1"))
}

/// Backtrace capture is OPT-IN (extra slow) via `FLAME_POOL_TRAP_BACKTRACE=1`.
#[inline]
fn pool_trap_backtrace_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("FLAME_POOL_TRAP_BACKTRACE").ok().as_deref() == Some("1"))
}

fn trap_history() -> &'static Mutex<HashMap<u64, PtrHistory>> {
    static H: OnceLock<Mutex<HashMap<u64, PtrHistory>>> = OnceLock::new();
    H.get_or_init(|| Mutex::new(HashMap::new()))
}

fn trap_live_ranges() -> &'static Mutex<Vec<LiveRange>> {
    static R: OnceLock<Mutex<Vec<LiveRange>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(Vec::new()))
}

fn trap_seq() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static S: AtomicU64 = AtomicU64::new(0);
    S.fetch_add(1, Ordering::Relaxed)
}

#[inline]
fn trap_record(ptr: u64, state: PtrState, op: &'static str, bucket: usize) {
    if !pool_trap_bf16_enabled() {
        return;
    }
    // Backtrace only for non-live transitions — capture is slow.
    let bt = if pool_trap_backtrace_enabled() && state != PtrState::Live {
        Some(std::sync::Arc::new(
            std::backtrace::Backtrace::force_capture().to_string(),
        ))
    } else {
        None
    };
    if let Ok(mut h) = trap_history().lock() {
        let seq = trap_seq();
        let entry = h.entry(ptr).or_insert(PtrHistory {
            state,
            last_seq: seq,
            last_op: op,
            bucket,
            event_count: 0,
            events: Vec::with_capacity(TRAP_HISTORY_LEN),
        });
        entry.state = state;
        entry.last_seq = seq;
        entry.last_op = op;
        entry.bucket = bucket;
        entry.event_count = entry.event_count.saturating_add(1);
        if entry.events.len() == TRAP_HISTORY_LEN {
            entry.events.remove(0);
        }
        entry.events.push(PtrEvent {
            seq,
            state,
            op,
            bucket,
            bt,
        });
    }
}

#[inline]
fn trap_record_range(ptr: u64, state: PtrState, op: &'static str, bucket: usize, elems: usize) {
    trap_record(ptr, state, op, bucket);
    if !pool_trap_bf16_enabled() || elems == 0 {
        return;
    }
    let bytes = (elems as u64).saturating_mul(std::mem::size_of::<u16>() as u64);
    if bytes == 0 {
        return;
    }
    if let Ok(mut ranges) = trap_live_ranges().lock() {
        ranges.retain(|r| r.start != ptr);
        if state == PtrState::Live {
            ranges.push(LiveRange {
                start: ptr,
                end: ptr.saturating_add(bytes),
            });
        }
    }
}

#[inline]
fn trap_live_range_contains(ptr: u64) -> bool {
    if !pool_trap_bf16_enabled() {
        return false;
    }
    trap_live_ranges()
        .lock()
        .map(|ranges| ranges.iter().any(|r| ptr >= r.start && ptr < r.end))
        .unwrap_or(false)
}

/// Record an externally-sourced BF16 ptr being wrapped into a Tensor
/// (e.g. via `Tensor::from_bf16_slice_gpu` from a `BlockOffloader`
/// alloc or ring slice). Sets state to `Live` so subsequent
/// `pool_return_u16` is the legitimate disposition for THIS Arc — and
/// the trap can detect if the SAME ptr later gets a SECOND wrap (which
/// would indicate two independent Arcs for the same memory, the actual
/// bug pattern).
pub fn trap_record_external(ptr: u64, call_site: &'static str) {
    trap_record(ptr, PtrState::Live, call_site, 0);
}

/// Range-aware variant of [`trap_record_external`] for callers that know the
/// live allocation length. This is needed because tensor views often pass a
/// mid-allocation pointer into CUDA APIs.
pub fn trap_record_external_range(ptr: u64, elems: usize, call_site: &'static str) {
    trap_record_range(ptr, PtrState::Live, call_site, elems, elems);
}

/// Record a BF16 allocation as live. Used by allocator paths that do not pass
/// through the BF16 free-list checkout path (direct cudart alloc, external miss
/// allocators, and the static slab).
pub(crate) fn trap_record_bf16_live(
    ptr: u64,
    elems: usize,
    call_site: &'static str,
    bucket: usize,
) {
    trap_record_range(ptr, PtrState::Live, call_site, bucket, elems);
}

/// Record a BF16 allocation as no longer live without implying it was returned
/// to the legacy free-list. Static slab releases use this when the final owning
/// TensorStorage drops.
pub(crate) fn trap_record_bf16_released(
    ptr: u64,
    elems: usize,
    call_site: &'static str,
    bucket: usize,
) {
    trap_record_range(ptr, PtrState::Freed, call_site, bucket, elems);
}

/// Validate a BF16 ptr right before passing it to a CUDA API. Panics
/// with provenance when the trap is on AND the ptr's last recorded
/// state is not `Live`. When the trap is off, no-op. Foreign ptrs
/// (never seen by the pool) are tolerated silently.
pub fn trap_validate_bf16_ptr(ptr: u64, call_site: &str) {
    if !pool_trap_bf16_enabled() {
        return;
    }
    if let Ok(h) = trap_history().lock() {
        if let Some(hist) = h.get(&ptr) {
            if hist.state != PtrState::Live {
                if trap_live_range_contains(ptr) {
                    return;
                }
                let state = hist.state;
                let last_seq = hist.last_seq;
                let last_op = hist.last_op;
                let bucket = hist.bucket;
                let event_count = hist.event_count;
                let events = hist.events.clone();
                drop(h);
                eprintln!(
                    "[BF16-POOL TRAP] ====== FORENSIC HISTORY (last {} events) ======",
                    events.len()
                );
                for (i, ev) in events.iter().enumerate() {
                    eprintln!(
                        "[BF16-POOL TRAP] [{}] seq={} state={:?} op={} bucket={}",
                        i, ev.seq, ev.state, ev.op, ev.bucket
                    );
                    if let Some(ref bt) = ev.bt {
                        eprintln!("[BF16-POOL TRAP]      backtrace:");
                        for line in bt.lines().take(40) {
                            eprintln!("[BF16-POOL TRAP]        {}", line);
                        }
                    }
                }
                eprintln!("[BF16-POOL TRAP] ====== END HISTORY ======");
                panic!(
                    "[BF16-POOL TRAP] call_site={} ptr=0x{:x} state={:?} \
                     last_seq={} last_op={} bucket={} event_count={}",
                    call_site, ptr, state, last_seq, last_op, bucket, event_count
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pool statistics
// ---------------------------------------------------------------------------
#[derive(Debug)]
pub struct PoolStats {
    pub alloc_count: usize,
    pub reuse_count: usize,
    pub return_count: usize,
    pub peak_bytes: usize,
    pub current_cached_bytes: usize,
    pub current_cached_entries: usize,
}

// ---------------------------------------------------------------------------
// CudaAllocPool — the global caching allocator
// ---------------------------------------------------------------------------

/// Maximum bucket size: 2 GiB (2^31 bytes = 536_870_912 f32 elements).
const MAX_POOL_BYTES: usize = 2 * 1024 * 1024 * 1024;
/// Maximum elements per size-class free list to prevent unbounded growth.
const MAX_FREE_PER_SIZE: usize = 32;

// ---------------------------------------------------------------------------
// PyTorch-style size-bucket rounding.
//
// Mirrors `/home/alex/pytorch/c10/core/AllocatorConfig.h:13-24` and
// `c10/cuda/CUDACachingAllocator.cpp::round_size` / `::get_allocation_size`.
// Without rounding, every unique-size temporary (broadcast scratch, narrow
// scratch, KV intermediate, seq-len-dependent shapes) misses the pool and
// hits cudaMalloc — observed as 370 cudaMalloc/free per step on plain-LoRA
// zimage. Rounding maps slight shape variations to a shared bucket key.
//
// Constants are PyTorch's defaults; do not change without re-benchmarking.
// ---------------------------------------------------------------------------
/// All sizes are rounded to at least this many bytes (PT `kMinBlockSize`).
const K_MIN_BLOCK_BYTES: usize = 512;
/// Largest "small" allocation in bytes (PT `kSmallSize`). Requests at or
/// below this round to multiples of `K_MIN_BLOCK_BYTES`.
const K_SMALL_SIZE_BYTES: usize = 1 * 1024 * 1024;
/// Threshold above which we use the larger 2 MiB bucket (PT `kMinLargeAlloc`).
const K_MIN_LARGE_ALLOC_BYTES: usize = 10 * 1024 * 1024;
/// Bucket granularity for large allocations in bytes (PT `kRoundLarge` and
/// `kSmallBuffer`).
const K_ROUND_LARGE_BYTES: usize = 2 * 1024 * 1024;

/// Round `bytes` up to the next bucket boundary using PyTorch's three-tier
/// strategy. Returns the rounded byte count.
#[inline]
fn round_bytes_up(bytes: usize) -> usize {
    if bytes <= K_SMALL_SIZE_BYTES {
        // Round to next multiple of K_MIN_BLOCK_BYTES (= 512).
        if bytes < K_MIN_BLOCK_BYTES {
            K_MIN_BLOCK_BYTES
        } else {
            (bytes + K_MIN_BLOCK_BYTES - 1) & !(K_MIN_BLOCK_BYTES - 1)
        }
    } else if bytes < K_MIN_LARGE_ALLOC_BYTES {
        // 1 MiB < x < 10 MiB → round to 2 MiB.
        (bytes + K_ROUND_LARGE_BYTES - 1) & !(K_ROUND_LARGE_BYTES - 1)
    } else {
        // x >= 10 MiB → round to next 2 MiB.
        (bytes + K_ROUND_LARGE_BYTES - 1) & !(K_ROUND_LARGE_BYTES - 1)
    }
}

/// Round element count up to the next bucket boundary for `T`-sized elements.
#[inline]
fn round_elems_up<T>(n: usize) -> usize {
    let elem_size = std::mem::size_of::<T>();
    debug_assert!(elem_size > 0);
    if n == 0 {
        return 0;
    }
    let req_bytes = n.saturating_mul(elem_size);
    let bucket_bytes = round_bytes_up(req_bytes);
    // Convert back to element count. Buckets are multiples of 512 bytes
    // or 2 MiB; both are multiples of every type alignment we use
    // (sizeof(f32)=4, sizeof(u16)=2), so this divides evenly.
    bucket_bytes / elem_size
}

/// Cache key. The device pointer disambiguates `CudaDevice` instances —
/// tests sometimes construct fresh `CudaDevice::new(0)` Arcs which carry
/// distinct streams; memory pooled from one device cannot be safely
/// reused on another without an event-sync handshake. Production code
/// uses a single global Arc (`global_cuda_device`) so this discriminator
/// has no cost (same key always).
#[derive(Eq, PartialEq, Hash, Clone, Copy)]
struct CacheKey {
    device_ptr: usize,
    bucket: usize,
    is_u16: bool,
}

pub struct CudaAllocPool {
    /// Free lists keyed by `(device, bucket-rounded element count, dtype)`.
    /// Bucket rounding lets slightly-different sizes share a free list,
    /// matching PyTorch's BFC allocator strategy.
    free_lists: Mutex<HashMap<CacheKey, Vec<FreeEntry>>>,
    /// Whether the pool is accepting returns (set false during shutdown).
    active: AtomicBool,
    // --- stats (only updated when profiling_enabled()) ---
    alloc_count: AtomicUsize,
    reuse_count: AtomicUsize,
    return_count: AtomicUsize,
    peak_bytes: AtomicUsize,
    current_bytes: AtomicUsize,
    // --- hit/miss/bucket counters (always live, lock-free; minimal cost) ---
    hits: AtomicU64,
    misses: AtomicU64,
    bucket_saves: AtomicU64,
    /// External cache-miss allocator (Phase 2a). When `Some`, cache-MISS
    /// paths in `pool_alloc_u16` / `pool_alloc_f32` route through this
    /// instead of `device.alloc::<T>`. See `PoolMissAllocator` docs.
    miss_alloc: Mutex<Option<Arc<dyn PoolMissAllocator>>>,
    /// **R1a:** the underlying refcount storage moved to
    /// [`crate::external_memory::ExternalMemoryRegistry`]. This crate's
    /// `register_external_ptr` / `is_external_ptr` / `unregister_external_ptr`
    /// stay as the public surface (the `BlockOffloader` ring path,
    /// `pool_return_*` external-guards, and `PoolMissAllocator` callers all
    /// hit these by name) but route through the unified registry so a
    /// future range-based slab allocator can protect mid-slab offsets the
    /// exact-ptr map could never see. See `external_memory.rs` for the
    /// rationale and `RangeHandle` API.
    ///
    /// **Why a refcount, not a set** (2026-05-14 Phase 2 round 2 fix): the
    /// `RingAllocator` cyclically reuses slab offsets — when the forward
    /// cursor wraps, the SAME `device_ptr` is handed out for a fresh
    /// allocation while a prior tensor with the same ptr may still be
    /// alive. With a `HashSet`, the first drop unregistered the ptr and
    /// the second drop saw `is_external_ptr=false`, tagged its FreeEntry
    /// non-external, and then `clear_cache` called `free_async` on a
    /// ring-slab offset → `CUDA_ERROR_INVALID_VALUE` panic. The refcount
    /// keeps the ptr marked external until ALL live tensors sharing it
    /// have been forgotten. The R1a registry preserves that semantics
    /// exactly. See `tests` in this module and the Klein 9B Phase 2 gate
    /// (`/tmp/k9_p2r2_*` logs).
    /// (No local storage — see `external_memory::ExternalMemoryRegistry`.)
    /// Lock-free counter of cache misses that routed through the external
    /// allocator. Useful for verifying the miss-route is firing as
    /// expected from a smoke-test harness.
    external_misses: AtomicU64,
}

impl CudaAllocPool {
    fn new() -> Self {
        Self {
            free_lists: Mutex::new(HashMap::new()),
            active: AtomicBool::new(true),
            alloc_count: AtomicUsize::new(0),
            reuse_count: AtomicUsize::new(0),
            return_count: AtomicUsize::new(0),
            peak_bytes: AtomicUsize::new(0),
            current_bytes: AtomicUsize::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            bucket_saves: AtomicU64::new(0),
            miss_alloc: Mutex::new(None),
            external_misses: AtomicU64::new(0),
        }
    }

    /// Test if `ptr` is tagged as external (Phase 2a). Public so external
    /// allocators (e.g. `BlockOffloader::alloc_bf16_via_ring`) can verify
    /// their pointers are tracked.
    ///
    /// **R1a:** delegates to
    /// [`crate::external_memory::ExternalMemoryRegistry::should_skip_free_any_device`].
    /// Returns true iff any range covers `ptr` OR an exact entry exists
    /// with non-zero refcount (any device).
    #[inline]
    pub fn is_external_ptr(&self, ptr: u64) -> bool {
        crate::external_memory::ExternalMemoryRegistry::global().should_skip_free_any_device(ptr)
    }

    /// Register `ptr` as external. Public so direct external-allocator
    /// callers (Phase 2: `BlockOffloader` ring-backed allocs) can mark
    /// their pointers without going through the `PoolMissAllocator`
    /// trait.
    ///
    /// Each call increments the ptr's refcount by 1. Must be balanced by
    /// a matching `unregister_external_ptr`. The ring allocator may hand
    /// out the same physical address for multiple concurrent allocations
    /// after the forward cursor wraps, in which case both lifetimes are
    /// tracked here as count=2.
    ///
    /// **R1a:** delegates to the unified
    /// [`crate::external_memory::ExternalMemoryRegistry`] under the
    /// device-agnostic [`crate::external_memory::DEVICE_KEY_ANY`] key
    /// (the back-compat API has no device parameter; range entries
    /// keyed by real devices remain reachable via the
    /// `should_skip_free_any_device` path).
    pub fn register_external_ptr(&self, ptr: u64) {
        crate::external_memory::ExternalMemoryRegistry::global().register_exact(
            ptr,
            crate::external_memory::DEVICE_KEY_ANY,
            crate::external_memory::ExternalOwner::PoolExact,
        );
    }

    /// Decrement `ptr`'s external refcount. Removes the entry when the
    /// count reaches zero. Called from the `push_*` guards when an
    /// external entry has been reconstructed-and-forgotten (its
    /// lifecycle is the external allocator's, not the pool's), so the
    /// map doesn't grow unbounded across step boundaries.
    ///
    /// No-op if `ptr` is not registered. Saturates at 0 (extra unregisters
    /// after the count is already 0 are silently ignored).
    ///
    /// **R1a:** delegates to
    /// [`crate::external_memory::ExternalMemoryRegistry::unregister_exact`].
    pub fn unregister_external_ptr(&self, ptr: u64) {
        let _ = crate::external_memory::ExternalMemoryRegistry::global().unregister_exact(ptr);
    }

    /// Inspection: total number of external ptr entries currently tracked
    /// (sum across all distinct ptrs, not summed refcounts). Used by tests.
    #[doc(hidden)]
    pub fn external_ptr_count(&self) -> usize {
        crate::external_memory::ExternalMemoryRegistry::global().exact_entry_count()
    }

    /// Inspection: current refcount for `ptr`. Returns 0 if not tracked.
    /// Used by tests.
    #[doc(hidden)]
    pub fn external_ptr_refcount(&self, ptr: u64) -> u32 {
        crate::external_memory::ExternalMemoryRegistry::global()
            .exact_refcount(ptr, crate::external_memory::DEVICE_KEY_ANY)
    }

    /// Lock-free counter of cache misses that were served by an installed
    /// `PoolMissAllocator` (Phase 2a diagnostic). Returns 0 when no
    /// external allocator is installed.
    #[inline]
    pub fn external_miss_count(&self) -> u64 {
        self.external_misses.load(Ordering::Relaxed)
    }

    /// Snapshot of hit/miss/bucket-save counters. `bucket_saves` is incremented
    /// every time a request maps to a different bucket than its exact size
    /// (i.e., would have been a miss without rounding but became a hit). The
    /// counters are best-effort (`Relaxed` ordering) and always-on.
    pub fn hit_miss_counts(&self) -> (u64, u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.bucket_saves.load(Ordering::Relaxed),
        )
    }

    /// Round up to next power of 2 (element count). **Unused in production;**
    /// kept for the existing unit test. The real bucketing is `round_elems_up`.
    #[inline]
    fn bucket_size(n: usize) -> usize {
        if n == 0 {
            return 1;
        }
        n.next_power_of_two()
    }

    /// Try to pop a cached f32 allocation matching `(device, bucket)`.
    /// The popped entry's `len` is the actual underlying allocation size
    /// (= `bucket`).
    fn try_pop(&self, device: &Arc<CudaDevice>, bucket: usize) -> Option<FreeEntry> {
        let key = CacheKey {
            device_ptr: Arc::as_ptr(device) as *const () as usize,
            bucket,
            is_u16: false,
        };
        let mut lists = self.free_lists.lock().ok()?;
        let list = lists.get_mut(&key)?;
        let entry = list.pop();
        if entry.is_some() && profiling_enabled() {
            self.reuse_count.fetch_add(1, Ordering::Relaxed);
            let bytes = bucket * std::mem::size_of::<f32>();
            self.current_bytes.fetch_sub(bytes, Ordering::Relaxed);
        }
        entry
    }

    /// Push a freed f32 allocation back into the pool, keyed by
    /// `(device, bucket-rounded element count)`.
    fn push_f32(&self, entry: FreeEntry) {
        if !self.active.load(Ordering::Relaxed) {
            // Skip cudaFree if the backing memory is owned by the
            // external miss-allocator (Phase 2a) — the slab Arc owns it.
            if entry.is_external {
                unsafe { reconstruct_and_forget::<f32>(entry.ptr, entry.len, entry.device) };
            } else {
                unsafe { reconstruct_and_drop::<f32>(entry.ptr, entry.len, entry.device) };
            }
            return;
        }

        let size = entry.len; // already bucket-sized (allocator invariant)
        let bytes = size * std::mem::size_of::<f32>();

        // Don't cache huge allocations (>2 GiB).
        if bytes > MAX_POOL_BYTES {
            if entry.is_external {
                unsafe { reconstruct_and_forget::<f32>(entry.ptr, entry.len, entry.device) };
            } else {
                unsafe { reconstruct_and_drop::<f32>(entry.ptr, entry.len, entry.device) };
            }
            return;
        }

        let key = CacheKey {
            device_ptr: Arc::as_ptr(&entry.device) as *const () as usize,
            bucket: size,
            is_u16: false,
        };

        if let Ok(mut lists) = self.free_lists.lock() {
            let list = lists.entry(key).or_insert_with(Vec::new);
            if list.len() >= MAX_FREE_PER_SIZE {
                drop(lists);
                if entry.is_external {
                    unsafe { reconstruct_and_forget::<f32>(entry.ptr, entry.len, entry.device) };
                } else {
                    unsafe { reconstruct_and_drop::<f32>(entry.ptr, entry.len, entry.device) };
                }
                return;
            }
            // Phase 2 guard: external entries (ring-backed) must NEVER enter the
            // active free list. The ring slab's bytes will be handed out again
            // on the next allocation; if a stale entry sat in the free list,
            // `try_pop` would return a slice aliased to live ring memory and
            // the caller would silently corrupt training. Reconstruct-and-
            // forget here (the slab Arc still owns the memory) and untag.
            if entry.is_external {
                let ptr = entry.ptr;
                unsafe { reconstruct_and_forget::<f32>(entry.ptr, entry.len, entry.device) };
                drop(lists);
                self.unregister_external_ptr(ptr);
                return;
            }
            list.push(entry);

            if profiling_enabled() {
                self.return_count.fetch_add(1, Ordering::Relaxed);
                let cur = self.current_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
                let mut peak = self.peak_bytes.load(Ordering::Relaxed);
                while cur > peak {
                    match self.peak_bytes.compare_exchange_weak(
                        peak,
                        cur,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(p) => peak = p,
                    }
                }
            }
        } else {
            if entry.is_external {
                unsafe { reconstruct_and_forget::<f32>(entry.ptr, entry.len, entry.device) };
            } else {
                unsafe { reconstruct_and_drop::<f32>(entry.ptr, entry.len, entry.device) };
            }
        }
    }

    /// Push a u16 (BF16) allocation back into the pool.
    fn push_u16(&self, entry: FreeEntry) {
        if !self.active.load(Ordering::Relaxed) {
            if entry.is_external {
                unsafe { reconstruct_and_forget::<u16>(entry.ptr, entry.len, entry.device) };
            } else {
                unsafe { reconstruct_and_drop::<u16>(entry.ptr, entry.len, entry.device) };
            }
            return;
        }

        let size = entry.len; // already bucket-sized
        let bytes = size * std::mem::size_of::<u16>();
        if bytes > MAX_POOL_BYTES {
            if entry.is_external {
                unsafe { reconstruct_and_forget::<u16>(entry.ptr, entry.len, entry.device) };
            } else {
                unsafe { reconstruct_and_drop::<u16>(entry.ptr, entry.len, entry.device) };
            }
            return;
        }

        let key = CacheKey {
            device_ptr: Arc::as_ptr(&entry.device) as *const () as usize,
            bucket: size,
            is_u16: true,
        };

        if let Ok(mut lists) = self.free_lists.lock() {
            let list = lists.entry(key).or_insert_with(Vec::new);
            if list.len() >= MAX_FREE_PER_SIZE {
                drop(lists);
                if entry.is_external {
                    unsafe { reconstruct_and_forget::<u16>(entry.ptr, entry.len, entry.device) };
                } else {
                    unsafe { reconstruct_and_drop::<u16>(entry.ptr, entry.len, entry.device) };
                }
                return;
            }
            // Phase 2 guard: external (ring-backed) entries must never enter
            // the active free list. See identical comment in `push_f32`.
            if entry.is_external {
                let ptr = entry.ptr;
                unsafe { reconstruct_and_forget::<u16>(entry.ptr, entry.len, entry.device) };
                drop(lists);
                self.unregister_external_ptr(ptr);
                return;
            }
            // 2026-05-15 DEFENSIVE FIX: if the same ptr is ALREADY in the
            // free-list (or in ANY bucket's list), this is a duplicate-Arc
            // bug — two independent `CudaSlice<u16>` objects with the same
            // underlying ptr. Pushing again would queue a double-cudaFree
            // for `clear_cache`. Instead, MEM::FORGET this entry (don't
            // cudaFree, don't push). cudart sees only one allocation at
            // this ptr; the first cudaFree will free it correctly. This
            // trades a small leak for crash-class elimination.
            //
            // Detection cost: O(N) scan of the matching bucket's list.
            // Bucket lists are bounded at MAX_FREE_PER_SIZE=32, so per-
            // call cost is ≤32 ptr comparisons. Worth it.
            let ptr = entry.ptr;
            if list.iter().any(|e| e.ptr == ptr) {
                drop(lists);
                // Forget the duplicate. The original entry will free it.
                unsafe { reconstruct_and_forget::<u16>(entry.ptr, entry.len, entry.device) };
                log::warn!(
                    "pool: u16 duplicate-return detected ptr=0x{:x} bucket={} — forgetting (defensive fix)",
                    ptr, size
                );
                if pool_trap_bf16_enabled() {
                    eprintln!(
                        "[BF16-POOL] DUPLICATE-RETURN DETECTED ptr=0x{:x} bucket={} — forgotten",
                        ptr, size
                    );
                }
                return;
            }
            list.push(entry);

            if profiling_enabled() {
                self.return_count.fetch_add(1, Ordering::Relaxed);
                let cur = self.current_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
                let mut peak = self.peak_bytes.load(Ordering::Relaxed);
                while cur > peak {
                    match self.peak_bytes.compare_exchange_weak(
                        peak,
                        cur,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(p) => peak = p,
                    }
                }
            }
        } else {
            if entry.is_external {
                unsafe { reconstruct_and_forget::<u16>(entry.ptr, entry.len, entry.device) };
            } else {
                unsafe { reconstruct_and_drop::<u16>(entry.ptr, entry.len, entry.device) };
            }
        }
    }

    /// Try to pop a cached u16 allocation matching `(device, bucket)`.
    fn try_pop_u16(&self, device: &Arc<CudaDevice>, bucket: usize) -> Option<FreeEntry> {
        let key = CacheKey {
            device_ptr: Arc::as_ptr(device) as *const () as usize,
            bucket,
            is_u16: true,
        };
        let mut lists = self.free_lists.lock().ok()?;
        let list = lists.get_mut(&key)?;
        let entry = list.pop();
        if entry.is_some() && profiling_enabled() {
            self.reuse_count.fetch_add(1, Ordering::Relaxed);
            let bytes = bucket * std::mem::size_of::<u16>();
            self.current_bytes.fetch_sub(bytes, Ordering::Relaxed);
        }
        entry
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStats {
        let (cached_bytes, cached_entries) = if let Ok(lists) = self.free_lists.lock() {
            let mut bytes = 0usize;
            let mut entries = 0usize;
            for (key, list) in lists.iter() {
                let elem_bytes = if key.is_u16 { 2 } else { 4 };
                bytes += key.bucket * elem_bytes * list.len();
                entries += list.len();
            }
            (bytes, entries)
        } else {
            (0, 0)
        };

        PoolStats {
            alloc_count: self.alloc_count.load(Ordering::Relaxed),
            reuse_count: self.reuse_count.load(Ordering::Relaxed),
            return_count: self.return_count.load(Ordering::Relaxed),
            peak_bytes: self.peak_bytes.load(Ordering::Relaxed),
            current_cached_bytes: cached_bytes,
            current_cached_entries: cached_entries,
        }
    }

    /// Free all cached memory. Call between training steps or on OOM retry.
    ///
    /// **Phase 2a — external entries**: any entry tagged `is_external` (its
    /// backing memory came from an installed `PoolMissAllocator`) is dropped
    /// via `reconstruct_and_forget` so that the external allocator's slab
    /// retains ownership. Non-external entries `cudaFree` as before.
    ///
    /// **Diagnostic mode**: set `FLAME_POOL_CLEAR_DEBUG=1` to enable
    /// per-entry `eprintln!` before each drop, plus `catch_unwind`
    /// around each drop so a panic in one entry is logged with its
    /// `(ptr,bucket,is_u16,tagged_ext,hook_ext)` provenance instead of
    /// unwinding the whole process. Used to pinpoint exact failing
    /// entries when Phase 2 ring + pool integration trips
    /// `CUDA_ERROR_INVALID_VALUE`. The diagnostic was the tool that
    /// localized the ring-wrap double-free fixed in 2026-05-14 (see
    /// `external_ptrs` doc and `test_external_ptr_refcount_under_ring_wrap`).
    pub fn clear_cache(&self) {
        let entries: Vec<(CacheKey, Vec<FreeEntry>)> = {
            let mut lists = match self.free_lists.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            lists.drain().collect()
        };
        let debug = std::env::var("FLAME_POOL_CLEAR_DEBUG").ok().as_deref() == Some("1");
        if debug {
            self.clear_cache_debug(entries);
        } else {
            // Default fast path — direct drops, no env check or counters.
            for (key, list) in entries {
                for entry in list {
                    let trap_ptr = entry.ptr;
                    let trap_bucket = key.bucket;
                    let trap_is_u16 = key.is_u16;
                    let trap_is_ext = entry.is_external;
                    unsafe {
                        if entry.is_external {
                            if key.is_u16 {
                                reconstruct_and_forget::<u16>(entry.ptr, entry.len, entry.device);
                            } else {
                                reconstruct_and_forget::<f32>(entry.ptr, entry.len, entry.device);
                            }
                        } else {
                            if key.is_u16 {
                                reconstruct_and_drop::<u16>(entry.ptr, entry.len, entry.device);
                            } else {
                                reconstruct_and_drop::<f32>(entry.ptr, entry.len, entry.device);
                            }
                        }
                    }
                    if trap_is_u16 {
                        let tag = if trap_is_ext {
                            "clear_cache/external_forget"
                        } else {
                            "clear_cache/cudaFree"
                        };
                        trap_record(trap_ptr, PtrState::Freed, tag, trap_bucket);
                    }
                }
            }
        }
        self.current_bytes.store(0, Ordering::Relaxed);
        // NOTE: do NOT clear `external_ptrs` here. Live tensors (whose
        // storage holds CudaSlice<T> values not yet returned to the pool)
        // still carry external pointers. When those tensors later drop, the
        // cudarc-pinctx Drop hook MUST see their ptr in `external_ptrs` to
        // skip `cudaFree` on the ring slab. Clearing the set on
        // `clear_cache` orphans every in-flight external tensor and causes
        // `CUDA_ERROR_INVALID_VALUE` panics inside `CudaSlice::drop` (see
        // Phase 2a 5-step smoke crash at backward-graph teardown).
        //
        // External entries removed FROM the free list above have already
        // been `reconstruct_and_forget`-ed and will never drop again, so
        // their ptr need never appear in the map again — but the map is
        // a HashMap<u64, u32> refcount, so leaving stale entries is
        // harmless (the matching slab is also gone, so no live tensor
        // can collide).
    }

    /// `FLAME_POOL_CLEAR_DEBUG=1` slow path: per-entry log + `catch_unwind`
    /// so a panic in one drop is captured with provenance rather than
    /// terminating the process. Used to localize ring/pool integration
    /// bugs (e.g., the 2026-05-14 round-2 ring-wrap double-free).
    #[cold]
    fn clear_cache_debug(&self, entries: Vec<(CacheKey, Vec<FreeEntry>)>) {
        let mut total_entries = 0usize;
        let mut total_fail = 0usize;
        for (key, list) in entries {
            for entry in list {
                total_entries += 1;
                let ptr = entry.ptr;
                let len = entry.len;
                let tagged_external = entry.is_external;
                let hook_says_external = self.is_external_ptr(ptr);
                eprintln!(
                    "[pool.clear_cache] #{total_entries} ptr=0x{ptr:x} bucket={} u16={} tagged_ext={} hook_ext={}",
                    len, key.is_u16, tagged_external, hook_says_external,
                );
                let do_drop = move || unsafe {
                    if tagged_external {
                        if key.is_u16 {
                            reconstruct_and_forget::<u16>(ptr, len, entry.device);
                        } else {
                            reconstruct_and_forget::<f32>(ptr, len, entry.device);
                        }
                    } else {
                        if key.is_u16 {
                            reconstruct_and_drop::<u16>(ptr, len, entry.device);
                        } else {
                            reconstruct_and_drop::<f32>(ptr, len, entry.device);
                        }
                    }
                };
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(do_drop));
                if r.is_err() {
                    total_fail += 1;
                    eprintln!(
                        "[pool.clear_cache] PANIC at #{total_entries} ptr=0x{ptr:x} bucket={} u16={} tagged_ext={} hook_ext={}",
                        len, key.is_u16, tagged_external, hook_says_external,
                    );
                }
            }
        }
        eprintln!("[pool.clear_cache] done: entries={total_entries} failed={total_fail}",);
    }
}

impl Drop for CudaAllocPool {
    fn drop(&mut self) {
        self.active.store(false, Ordering::SeqCst);
        self.clear_cache();
    }
}

// ---------------------------------------------------------------------------
// Unsafe helpers — reconstruct / decompose CudaSlice<T>
// ---------------------------------------------------------------------------

/// Reconstruct a `CudaSlice<T>` from raw parts and let it drop (calling cudaFree).
///
/// # Safety
/// `ptr` must be a valid device pointer allocated by the same `device`,
/// with `len` elements of type T.
unsafe fn reconstruct_and_drop<T>(ptr: u64, len: usize, device: Arc<CudaDevice>) {
    let mirror = CudaSliceMirror::<T> {
        cu_device_ptr: ptr,
        len,
        device,
        host_buf: None,
    };
    let slice: CudaSlice<T> = std::mem::transmute(mirror);
    drop(slice); // runs cudaFree
}

/// Reconstruct a `CudaSlice<T>` mirror from raw parts and immediately
/// `mem::forget` it — does NOT run `cudaFree`. Used for Phase 2a external
/// allocations whose backing memory is owned by a `PoolMissAllocator`
/// (e.g., a `ring_alloc::RingAllocator` slab).
///
/// The `device: Arc<CudaDevice>` clone increments the Arc refcount when
/// the mirror is built; `mem::forget` then leaks that one ref. The
/// caller path always passes an Arc obtained from a fresh `clone()` /
/// move on the input entry, so the leak is bounded to the lifecycle of
/// the in-flight pool return.
///
/// # Safety
/// `ptr` MUST point into memory owned by the installed
/// `PoolMissAllocator` (i.e., the slab Arc keeps it alive). Calling this
/// on a `device.alloc`-sourced pointer leaks GPU memory.
unsafe fn reconstruct_and_forget<T>(ptr: u64, len: usize, device: Arc<CudaDevice>) {
    let mirror = CudaSliceMirror::<T> {
        cu_device_ptr: ptr,
        len,
        device,
        host_buf: None,
    };
    let slice: CudaSlice<T> = std::mem::transmute(mirror);
    std::mem::forget(slice);
}

/// Reconstruct a `CudaSlice<T>` from raw parts WITHOUT dropping.
///
/// # Safety
/// Same preconditions as `reconstruct_and_drop`.
unsafe fn reconstruct_slice<T>(ptr: u64, len: usize, device: Arc<CudaDevice>) -> CudaSlice<T> {
    let mirror = CudaSliceMirror::<T> {
        cu_device_ptr: ptr,
        len,
        device,
        host_buf: None,
    };
    std::mem::transmute(mirror)
}

/// Decompose a `CudaSlice<T>` into raw parts, consuming it without cudaFree.
///
/// # Safety
/// Caller must eventually either reconstruct the slice or manually free the ptr.
unsafe fn decompose_slice<T>(slice: CudaSlice<T>) -> (u64, usize, Arc<CudaDevice>) {
    let ptr = *slice.device_ptr();
    let len = DeviceSlice::len(&slice);
    // We need the device Arc. Read it from the mirror layout.
    let mirror: CudaSliceMirror<T> = std::mem::transmute(slice);
    // mirror won't drop (no Drop impl), so ptr stays live.
    let device = mirror.device.clone();
    // Forget mirror to prevent any implicit cleanup.
    // (CudaSliceMirror has no Drop, but Arc<CudaDevice> clone keeps it alive.)
    std::mem::forget(mirror);
    (ptr, len, device)
}

// ---------------------------------------------------------------------------
// Miss-allocator plugin (Phase 2a — ring-alloc backend opt-in)
//
// When installed, ALL cache-MISS allocations of `pool_alloc_u16` /
// `pool_alloc_f32` route through the installed allocator instead of
// calling `device.alloc::<T>(bucket)`. The returned `CudaSlice<T>` is
// expected to be a transmuted "borrowed view" onto memory the external
// allocator owns; the pool tags free-list entries from this path with
// `is_external: true` so that subsequent `clear_cache` / `pool.drop`
// skips `cudaFree` on those entries.
//
// Failure mode: if the external allocator returns `Err`, the cache miss
// falls back to `device.alloc::<T>` (identical to the no-allocator path).
//
// Design rationale: this exists to test the hypothesis (per
// `OFFLOAD_GAPS_vs_ONETRAINER.md` Gap 1) that routing cache-miss
// allocations through a `ring_alloc::RingAllocator` reduces the count of
// `cudaMallocAsync` / driver-mempool interactions across step boundaries
// and resolves the Klein 9B + `--offload` step-2 `CUDA_ERROR_INVALID_VALUE`
// crash without the `FLAME_ALLOC_POOL=0` workaround. See
// `OFFLOAD_NEXT_GEN_DESIGN.md` Phase 4-restart Phase 2 status.
// ---------------------------------------------------------------------------

/// External allocator that the pool can route cache misses through.
///
/// Implementors must return memory that is **NOT** owned by the returned
/// `CudaSlice` (i.e., dropping the slice MUST NOT call `cudaFree`). The
/// pool tags free-list entries from this path so that on `clear_cache` /
/// pool drop it skips the `cudaFree` step.
///
/// The element-count argument is the bucket-rounded size (already > 0).
/// Return `Err` to fall back to the default `device.alloc::<T>` path.
pub trait PoolMissAllocator: Send + Sync {
    /// Allocate `bucket_elems` u16 elements. Returned slice's `len()` MUST
    /// be `bucket_elems`. Underlying memory MUST NOT be freed when the
    /// slice is dropped.
    fn alloc_u16(
        &self,
        device: &Arc<CudaDevice>,
        bucket_elems: usize,
    ) -> crate::Result<CudaSlice<u16>>;

    /// Allocate `bucket_elems` f32 elements. Same contract as `alloc_u16`.
    fn alloc_f32(
        &self,
        device: &Arc<CudaDevice>,
        bucket_elems: usize,
    ) -> crate::Result<CudaSlice<f32>>;
}

/// Install a miss-allocator. All subsequent `pool_alloc_*` cache MISSES
/// route through this allocator instead of calling `device.alloc::<T>`.
///
/// Replaces any previously installed allocator. The previous allocator
/// (if any) is returned so the caller can restore it.
///
/// **Lifecycle contract** — the caller is responsible for:
/// 1. Calling [`clear_pool_cache`] BEFORE invalidating the backing
///    memory of any slice this allocator handed out. The pool's free
///    list may still hold ring-sourced entries; clearing the cache
///    drops the (no-op-cudaFree) mirror handles.
/// 2. Calling [`uninstall_miss_allocator`] when done; otherwise the
///    pool keeps trying to allocate through an allocator whose
///    backing storage may already be reset.
pub fn install_miss_allocator(
    allocator: Arc<dyn PoolMissAllocator>,
) -> Option<Arc<dyn PoolMissAllocator>> {
    let mut guard = global_pool().miss_alloc.lock().unwrap();
    let prev = guard.take();
    *guard = Some(allocator);
    // Install the cudarc-pinctx Drop hook so direct `CudaSlice::drop` calls
    // on ring-backed slices (which can leak out of `pool_alloc_*` through
    // helpers that don't call `pool_return_*`) skip the `cudaFree`. Without
    // this, dropping a slice whose ptr is an offset into a ring slab
    // panics with `CUDA_ERROR_INVALID_VALUE`.
    //
    // **R1a:** the hook closure is owned by
    // [`crate::external_memory::ExternalMemoryRegistry`] and consults the
    // unified registry (ranges + exact). Idempotent: first install wins,
    // subsequent calls are no-ops.
    crate::external_memory::ExternalMemoryRegistry::ensure_hook_installed();
    prev
}

/// Remove the installed miss-allocator (if any). The pool reverts to
/// calling `device.alloc::<T>` on cache miss. Returns the previously
/// installed allocator (if any) so the caller can decide whether to
/// drop it.
///
/// You should normally call [`clear_pool_cache`] immediately before this
/// to drain any free-list entries that point into the external
/// allocator's memory.
pub fn uninstall_miss_allocator() -> Option<Arc<dyn PoolMissAllocator>> {
    let mut guard = global_pool().miss_alloc.lock().unwrap();
    guard.take()
}

// ---------------------------------------------------------------------------
// Global singleton
// ---------------------------------------------------------------------------

static POOL: OnceLock<CudaAllocPool> = OnceLock::new();

/// Get the global allocation pool.
#[inline]
pub fn global_pool() -> &'static CudaAllocPool {
    POOL.get_or_init(CudaAllocPool::new)
}

// ---------------------------------------------------------------------------
// Public API — f32
// ---------------------------------------------------------------------------

/// Allocate at least `size` f32 elements from the caching pool.
///
/// **Bucket rounding (PT BFC allocator parity):** the request is rounded
/// up to the next bucket boundary; allocations within the same bucket
/// share a free list. The returned `CudaSlice<f32>` has
/// `len() == round_elems_up::<f32>(size)` (>= `size`) — callers must use
/// their original requested element count to compute kernel grids, or
/// honor `slice.len()` directly. The `TensorStorage::*` paths track
/// `numel` (= request) separately from the slice len, so they are
/// unaffected.
///
/// **Initialization:** the slice is **not** zeroed on either hit or miss.
/// This matches the BF16 path (`pool_alloc_u16`) and PT's BFCAllocator
/// semantics: callers are responsible for initializing the buffer. Set
/// `FLAME_F32_ZERO_INIT=1` to revert to the legacy zero-on-miss behavior
/// if a hidden caller is discovered.
pub fn pool_alloc_f32(device: &Arc<CudaDevice>, size: usize) -> crate::Result<CudaSlice<f32>> {
    // R2a: transient-scope slab dispatch. Routes to `StaticSlabAllocator`
    // when `FLAME_USE_STATIC_SLAB=1` AND a `StepSlabGuard` is active on the
    // current thread. Outside any guard or with the env unset, falls
    // through to the existing pool path unchanged. The slab helper returns
    // `None` (not Err) on any failure mode (overflow, poisoned mutex, no
    // active guard) — the legacy path is the safety net.
    if let Some(slice) = crate::static_slab_v2::pool_alloc_f32_via_slab(size) {
        return Ok(slice);
    }

    if pool_disabled() || size == 0 {
        // Non-pool path unchanged: legacy callers expect zero-init.
        return device
            .alloc_zeros::<f32>(size)
            .map_err(|e| crate::Error::CudaDriver(format!("{e:?}")));
    }

    // 2026-05-15: F32 free-list caching is OPT-IN via FLAME_F32_POOL_CACHE=1.
    //
    // The F32 caching path has a stale-ptr re-use bug across alloc generations
    // (see HANDOFF_2026-05-14_F32_MEMPOOL_BUG_OPEN.md + Skeptic Phase 2c). The
    // crash manifests as CUDA_ERROR_INVALID_VALUE on a later cuMemcpy/cuMemset
    // call targeting a ptr the cudart driver later rejects as invalid.
    //
    // The BF16 caching path (`pool_alloc_u16`) is unaffected — same code shape,
    // different free-list, no reported crash. BF16 is also the dominant
    // allocation volume (model weights, activations) so the perf delta from
    // disabling F32 caching is small.
    //
    // Default OFF removes the workaround tax (~0.7-1.0 s/step) by keeping
    // `FLAME_ALLOC_POOL=1` viable: BF16 caching delivers most of the perf,
    // F32 goes direct to cudart (same as the old `FLAME_ALLOC_POOL=0`
    // workaround did for ALL allocations).
    //
    // Set `FLAME_F32_POOL_CACHE=1` to re-enable the (buggy) F32 free-list
    // for experimentation. Default OFF.
    if !f32_pool_cache_enabled() {
        // Zero-init to match the `pool_disabled` path's behavior. This is
        // the path that's hit by default once the trainer-side
        // `FLAME_ALLOC_POOL=0` workaround is removed in Phase 3 — the
        // visible behavior must match (per Phase 3 safety constraint).
        // Per the 2026-05-12 audit, no caller observably depends on
        // zero-init, but the `pool_disabled` path zeros defensively for
        // "legacy callers expect zero-init"; mirror that here.
        return device
            .alloc_zeros::<f32>(size)
            .map_err(|e| crate::Error::CudaDriver(format!("alloc_zeros::<f32>({size}): {e:?}")));
    }

    let pool = global_pool();
    let bucket = round_elems_up::<f32>(size);

    if profiling_enabled() {
        pool.alloc_count.fetch_add(1, Ordering::Relaxed);
    }

    // Try cache hit at the bucket size.
    if let Some(entry) = pool.try_pop(device, bucket) {
        pool.hits.fetch_add(1, Ordering::Relaxed);
        if bucket != size {
            pool.bucket_saves.fetch_add(1, Ordering::Relaxed);
        }
        // Reconstruct slice with len = original request. The underlying
        // memory is `bucket` elements; cudaFree only needs the pointer.
        // Reporting `len = size` preserves the historical caller contract
        // that `slice.len() == requested_size` (callers use this for grid
        // math + cudarc's dtod_copy asserts src.len()==dst.len()).
        let slice = unsafe { reconstruct_slice::<f32>(entry.ptr, size, entry.device) };
        log::trace!(
            "pool: f32 hit size={} bucket={} hits={} misses={} bucket_saves={}",
            size,
            bucket,
            pool.hits.load(Ordering::Relaxed),
            pool.misses.load(Ordering::Relaxed),
            pool.bucket_saves.load(Ordering::Relaxed),
        );
        return Ok(slice);
    }

    // Cache miss — fresh allocation at the bucket size.
    pool.misses.fetch_add(1, Ordering::Relaxed);
    log::trace!(
        "pool: f32 miss size={} bucket={} hits={} misses={}",
        size,
        bucket,
        pool.hits.load(Ordering::Relaxed),
        pool.misses.load(Ordering::Relaxed),
    );

    // Phase 2a: if a miss-allocator is installed, route through it.
    // Returns a slice whose backing memory is owned by the external
    // allocator (e.g., a `ring_alloc::RingAllocator` slab). Drop on that
    // slice MUST NOT call cudaFree — see `reconstruct_and_forget`.
    if let Some(ext) = pool
        .miss_alloc
        .lock()
        .ok()
        .and_then(|g| g.as_ref().cloned())
    {
        match ext.alloc_f32(device, bucket) {
            Ok(slice) => {
                pool.external_misses.fetch_add(1, Ordering::Relaxed);
                let (ptr, _, dev) = unsafe { decompose_slice(slice) };
                // Register the pointer as external so the eventual pool_return
                // tags its FreeEntry correctly.
                pool.register_external_ptr(ptr);
                // Hand back a slice with len = original request.
                return Ok(unsafe { reconstruct_slice::<f32>(ptr, size, dev) });
            }
            Err(e) => {
                log::warn!(
                    "pool: external miss-allocator alloc_f32({bucket}) failed: {e:?} — falling back to device.alloc"
                );
                // Fall through to legacy path.
            }
        }
    }

    let zero_init = f32_zero_init_enabled();

    let alloc_once = |dev: &Arc<CudaDevice>, n: usize| -> crate::Result<CudaSlice<f32>> {
        if zero_init {
            dev.alloc_zeros::<f32>(n)
                .map_err(|e| crate::Error::CudaDriver(format!("alloc_zeros::<f32>({n}): {e:?}")))
        } else {
            unsafe { dev.alloc::<f32>(n) }
                .map_err(|e| crate::Error::CudaDriver(format!("alloc::<f32>({n}): {e:?}")))
        }
    };

    // Allocate `bucket` elements of underlying memory, but hand back a
    // CudaSlice with len=size. The slice's len affects cudarc's
    // dtod_copy/htod_copy assertions; the underlying capacity stays at
    // `bucket` for free-list reuse purposes.
    let result = match alloc_once(device, bucket) {
        Ok(s) => s,
        Err(_) => {
            pool.clear_cache();
            alloc_once(device, bucket).map_err(|e| {
                crate::Error::CudaDriver(format!(
                    "f32 alloc({bucket}) after pool.clear_cache: {e:?}"
                ))
            })?
        }
    };
    if bucket == size {
        Ok(result)
    } else {
        // Reconstruct with truncated len. The underlying memory is `bucket`
        // elements; only the slice header changes. cudaFree on drop just
        // uses the pointer.
        let (ptr, _, dev) = unsafe { decompose_slice(result) };
        Ok(unsafe { reconstruct_slice::<f32>(ptr, size, dev) })
    }
}

/// Return a `CudaSlice<f32>` to the caching pool instead of freeing it.
///
/// The slice's `len()` is the originally-requested element count. The
/// underlying allocation is at the bucket size; we re-round to recover
/// the bucket key for the free list.
///
/// # Safety
/// The slice must have been allocated by `pool_alloc_f32` or cudarc's
/// `device.alloc_zeros`. After this call, `slice` is consumed and the
/// caller must not use it.
pub fn pool_return_f32(slice: CudaSlice<f32>) {
    // 2026-05-15: when pool is disabled OR F32 caching is opt-out (default),
    // drop directly. Symmetric with `pool_alloc_f32`'s opt-out check.
    if pool_disabled() || !f32_pool_cache_enabled() {
        // If the slice's pointer is external (slab-owned or ring-owned), do
        // NOT cudaFree it. The owning allocator keeps it alive.
        let ptr = *slice.device_ptr();
        if global_pool().is_external_ptr(ptr) {
            let len = DeviceSlice::len(&slice);
            let (p, _, dev) = unsafe { decompose_slice(slice) };
            unsafe { reconstruct_and_forget::<f32>(p, len, dev) };
            global_pool().unregister_external_ptr(ptr);
        } else {
            drop(slice); // normal cudaFree
        }
        return;
    }

    let len = DeviceSlice::len(&slice);
    if len == 0 {
        drop(slice);
        return;
    }

    let (ptr, elem_len, device) = unsafe { decompose_slice(slice) };
    // Bucket key = round of the user-visible len. Same function applied to
    // the same input always yields the same bucket, so alloc and free see
    // matching keys.
    let bucket = round_elems_up::<f32>(elem_len);
    let is_external = global_pool().is_external_ptr(ptr);

    global_pool().push_f32(FreeEntry {
        ptr,
        len: bucket,
        device,
        is_external,
    });
}

// ---------------------------------------------------------------------------
// Public API — u16 (BF16)
// ---------------------------------------------------------------------------

/// Allocate at least `size` u16 (BF16) elements from the caching pool.
///
/// **Bucket rounding (PT BFC allocator parity):** the request is rounded
/// up to the next bucket boundary; allocations within the same bucket
/// share a free list. The returned `CudaSlice<u16>` has
/// `len() == round_elems_up::<u16>(size)` (>= `size`).
///
/// Always returns uninitialized memory (matches PT BFCAllocator semantics).
pub fn pool_alloc_u16(device: &Arc<CudaDevice>, size: usize) -> crate::Result<CudaSlice<u16>> {
    // R2a: transient-scope slab dispatch. See `pool_alloc_f32` for rationale.
    if let Some(slice) = crate::static_slab_v2::pool_alloc_u16_via_slab(size) {
        return Ok(slice);
    }

    if size == 0 {
        return unsafe {
            device
                .alloc::<u16>(size)
                .map_err(|e| crate::Error::CudaDriver(format!("{e:?}")))
        };
    }

    if pool_disabled() {
        return unsafe {
            device
                .alloc::<u16>(size)
                .map_err(|e| crate::Error::CudaDriver(format!("{e:?}")))
        };
    }

    // 2026-05-15: BF16 free-list caching is OPT-IN via FLAME_BF16_POOL_CACHE=1.
    //
    // Symmetric to the F32 opt-out. Phase 2 smoke proved the crash class
    // is NOT F32-exclusive — the BF16 free-list has the same stale-ptr
    // re-use bug. Routes BF16 allocs directly to cudart `device.alloc`
    // and cudaFree on drop, bypassing the free-list.
    //
    // BF16 returns uninitialized (matches the cached path's contract).
    if !bf16_pool_cache_enabled() {
        let slice = unsafe { device.alloc::<u16>(size) }
            .map_err(|e| crate::Error::CudaDriver(format!("alloc::<u16>({size}): {e:?}")))?;
        let ptr = *DevicePtr::device_ptr(&slice);
        trace_bf16(
            "alloc_direct",
            ptr,
            size,
            size,
            false,
            "pool_alloc_u16/direct",
        );
        trap_record_bf16_live(ptr, size, "pool_alloc_u16/direct", size);
        return Ok(slice);
    }

    let pool = global_pool();
    let bucket = round_elems_up::<u16>(size);

    if profiling_enabled() {
        pool.alloc_count.fetch_add(1, Ordering::Relaxed);
    }

    if let Some(entry) = pool.try_pop_u16(device, bucket) {
        pool.hits.fetch_add(1, Ordering::Relaxed);
        if bucket != size {
            pool.bucket_saves.fetch_add(1, Ordering::Relaxed);
        }
        trace_bf16(
            "alloc_hit",
            entry.ptr,
            bucket,
            size,
            entry.is_external,
            "pool_alloc_u16",
        );
        trap_record_bf16_live(entry.ptr, size, "pool_alloc_u16/hit", bucket);
        // Reconstruct with len = requested size, not bucket (see f32 path).
        let slice = unsafe { reconstruct_slice::<u16>(entry.ptr, size, entry.device) };
        log::trace!(
            "pool: u16 hit size={} bucket={} hits={} misses={} bucket_saves={}",
            size,
            bucket,
            pool.hits.load(Ordering::Relaxed),
            pool.misses.load(Ordering::Relaxed),
            pool.bucket_saves.load(Ordering::Relaxed),
        );
        return Ok(slice);
    }

    pool.misses.fetch_add(1, Ordering::Relaxed);
    log::trace!(
        "pool: u16 miss size={} bucket={} hits={} misses={}",
        size,
        bucket,
        pool.hits.load(Ordering::Relaxed),
        pool.misses.load(Ordering::Relaxed),
    );

    // Phase 2a: if a miss-allocator is installed, route through it.
    if let Some(ext) = pool
        .miss_alloc
        .lock()
        .ok()
        .and_then(|g| g.as_ref().cloned())
    {
        match ext.alloc_u16(device, bucket) {
            Ok(slice) => {
                pool.external_misses.fetch_add(1, Ordering::Relaxed);
                let (ptr, _, dev) = unsafe { decompose_slice(slice) };
                pool.register_external_ptr(ptr);
                trace_bf16(
                    "alloc_external_miss",
                    ptr,
                    bucket,
                    size,
                    true,
                    "pool_alloc_u16/external_miss",
                );
                trap_record_bf16_live(ptr, size, "pool_alloc_u16/external_miss", bucket);
                return Ok(unsafe { reconstruct_slice::<u16>(ptr, size, dev) });
            }
            Err(e) => {
                log::warn!(
                    "pool: external miss-allocator alloc_u16({bucket}) failed: {e:?} — falling back to device.alloc"
                );
                // Fall through.
            }
        }
    }

    // Fresh allocation at the bucket size (uninitialized).
    let result = unsafe {
        device
            .alloc::<u16>(bucket)
            .map_err(|e| crate::Error::CudaDriver(format!("alloc::<u16>({bucket}): {e:?}")))
    };

    let allocated = result.or_else(|_| {
        pool.clear_cache();
        unsafe {
            device.alloc::<u16>(bucket).map_err(|e| {
                crate::Error::CudaDriver(format!(
                    "alloc::<u16>({bucket}) after pool.clear_cache: {e:?}"
                ))
            })
        }
    })?;

    if bucket == size {
        let ptr = *DevicePtr::device_ptr(&allocated);
        trace_bf16(
            "alloc_miss",
            ptr,
            bucket,
            size,
            false,
            "pool_alloc_u16/fresh",
        );
        trap_record_bf16_live(ptr, size, "pool_alloc_u16/miss", bucket);
        Ok(allocated)
    } else {
        let (ptr, _, dev) = unsafe { decompose_slice(allocated) };
        trace_bf16(
            "alloc_miss",
            ptr,
            bucket,
            size,
            false,
            "pool_alloc_u16/fresh-shrunk",
        );
        trap_record_bf16_live(ptr, size, "pool_alloc_u16/miss-shrunk", bucket);
        Ok(unsafe { reconstruct_slice::<u16>(ptr, size, dev) })
    }
}

/// Return a `CudaSlice<u16>` to the caching pool.
pub fn pool_return_u16(slice: CudaSlice<u16>) {
    let trace_ptr = *slice.device_ptr();
    let trace_len = DeviceSlice::len(&slice);
    trace_bf16(
        "return_enter",
        trace_ptr,
        0,
        trace_len,
        false,
        "pool_return_u16",
    );
    // 2026-05-15: symmetric BF16 opt-out — when pool is disabled OR BF16
    // caching is opt-out (default), drop directly. Mirrors `pool_alloc_u16`
    // / `pool_return_f32` opt-outs.
    if pool_disabled() || !bf16_pool_cache_enabled() {
        let ptr = *slice.device_ptr();
        if global_pool().is_external_ptr(ptr) {
            let len = DeviceSlice::len(&slice);
            let (p, _, dev) = unsafe { decompose_slice(slice) };
            unsafe { reconstruct_and_forget::<u16>(p, len, dev) };
            global_pool().unregister_external_ptr(ptr);
            trap_record_bf16_released(p, len, "pool_return_u16/external_forget", 0);
        } else {
            drop(slice);
            trap_record_bf16_released(ptr, trace_len, "pool_return_u16/drop", 0);
        }
        return;
    }

    let len = DeviceSlice::len(&slice);
    if len == 0 {
        drop(slice);
        return;
    }

    let (ptr, elem_len, device) = unsafe { decompose_slice(slice) };
    let bucket = round_elems_up::<u16>(elem_len);
    let is_external = global_pool().is_external_ptr(ptr);
    trace_bf16(
        "return_push",
        ptr,
        bucket,
        elem_len,
        is_external,
        "pool_return_u16/push",
    );
    trap_record_range(
        ptr,
        PtrState::InCache,
        "pool_return_u16/push",
        bucket,
        elem_len,
    );

    global_pool().push_u16(FreeEntry {
        ptr,
        len: bucket,
        device,
        is_external,
    });
}

// ---------------------------------------------------------------------------
// Convenience: print stats summary
// ---------------------------------------------------------------------------

/// Print pool stats to stderr (gated on FLAME_PROFILE=1).
pub fn print_pool_stats() {
    if !profiling_enabled() {
        return;
    }
    let pool = global_pool();
    let s = pool.stats();
    let (hits, misses, bucket_saves) = pool.hit_miss_counts();
    let reuse_pct = if s.alloc_count > 0 {
        (s.reuse_count as f64) / (s.alloc_count as f64) * 100.0
    } else {
        0.0
    };
    let hit_pct = if hits + misses > 0 {
        (hits as f64) / ((hits + misses) as f64) * 100.0
    } else {
        0.0
    };
    eprintln!(
        "[alloc_pool] allocs={} reuses={} ({:.1}%) returns={} peak_cached={:.1}MB current_cached={:.1}MB entries={}",
        s.alloc_count,
        s.reuse_count,
        reuse_pct,
        s.return_count,
        s.peak_bytes as f64 / (1024.0 * 1024.0),
        s.current_cached_bytes as f64 / (1024.0 * 1024.0),
        s.current_cached_entries,
    );
    eprintln!(
        "[alloc_pool] hits={} misses={} ({:.1}%) bucket_saves={}",
        hits, misses, hit_pct, bucket_saves,
    );
}

/// Clear all cached GPU memory. Call on OOM or between phases.
pub fn clear_pool_cache() {
    global_pool().clear_cache();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_size() {
        assert_eq!(CudaAllocPool::bucket_size(0), 1);
        assert_eq!(CudaAllocPool::bucket_size(1), 1);
        assert_eq!(CudaAllocPool::bucket_size(2), 2);
        assert_eq!(CudaAllocPool::bucket_size(3), 4);
        assert_eq!(CudaAllocPool::bucket_size(5), 8);
        assert_eq!(CudaAllocPool::bucket_size(1000), 1024);
        assert_eq!(CudaAllocPool::bucket_size(1024), 1024);
        assert_eq!(CudaAllocPool::bucket_size(1025), 2048);
    }

    #[test]
    fn test_pool_disabled_env() {
        // Just verify the function doesn't panic.
        let _ = pool_disabled();
        let _ = profiling_enabled();
    }

    #[test]
    fn test_alloc_return_reuse() -> crate::Result<()> {
        // Allocate, return, allocate again — should get the same pointer.
        let device = CudaDevice::new(0)?;
        let size = 1024usize;

        let slice1 = pool_alloc_f32(&device, size)?;
        let ptr1 = *slice1.device_ptr();
        assert_eq!(DeviceSlice::len(&slice1), size);

        // Return to pool (this consumes slice1 without cudaFree).
        pool_return_f32(slice1);

        // Allocate again — should reuse the cached entry.
        let slice2 = pool_alloc_f32(&device, size)?;
        let ptr2 = *slice2.device_ptr();
        assert_eq!(ptr1, ptr2, "expected pool reuse — same device pointer");
        assert_eq!(DeviceSlice::len(&slice2), size);

        // Clean up — return and then clear cache (which does cudaFree).
        pool_return_f32(slice2);
        global_pool().clear_cache();

        Ok(())
    }

    #[test]
    fn test_round_bytes_up() {
        // Small allocs round to 512 bytes.
        assert_eq!(round_bytes_up(1), 512);
        assert_eq!(round_bytes_up(511), 512);
        assert_eq!(round_bytes_up(512), 512);
        assert_eq!(round_bytes_up(513), 1024);
        assert_eq!(round_bytes_up(4097), 4608);

        // At kSmallSize boundary (1 MiB), still rounds to 512.
        assert_eq!(round_bytes_up(K_SMALL_SIZE_BYTES), K_SMALL_SIZE_BYTES);

        // Mid range: 1 MiB < x < 10 MiB rounds to 2 MiB.
        assert_eq!(round_bytes_up(K_SMALL_SIZE_BYTES + 1), K_ROUND_LARGE_BYTES);
        assert_eq!(round_bytes_up(2 * 1024 * 1024), 2 * 1024 * 1024);
        assert_eq!(round_bytes_up(3_142_727), 4 * 1024 * 1024);
        assert_eq!(round_bytes_up(9 * 1024 * 1024), 10 * 1024 * 1024);

        // Large: >= 10 MiB rounds to 2 MiB.
        assert_eq!(
            round_bytes_up(K_MIN_LARGE_ALLOC_BYTES),
            K_MIN_LARGE_ALLOC_BYTES
        );
        assert_eq!(
            round_bytes_up(K_MIN_LARGE_ALLOC_BYTES + 1),
            K_MIN_LARGE_ALLOC_BYTES + K_ROUND_LARGE_BYTES
        );
    }

    #[test]
    fn test_round_elems_up_f32() {
        // 1 elem * 4 bytes = 4 bytes → rounds to 512 bytes = 128 f32 elems.
        assert_eq!(round_elems_up::<f32>(1), 128);
        // 1024 elems * 4 = 4096 bytes → 4096 bytes → 1024 elems (no change).
        assert_eq!(round_elems_up::<f32>(1024), 1024);
        // 0 → 0.
        assert_eq!(round_elems_up::<f32>(0), 0);
        // 3 MiB worth of f32 (786_433 elems) → rounds to 4 MiB.
        assert_eq!(round_elems_up::<f32>(786_433), 4 * 1024 * 1024 / 4);
    }

    #[test]
    fn test_round_elems_up_u16() {
        assert_eq!(round_elems_up::<u16>(1), 256);
        assert_eq!(round_elems_up::<u16>(0), 0);
        // 2048 elems * 2 = 4096 bytes → 2048 elems.
        assert_eq!(round_elems_up::<u16>(2048), 2048);
    }

    #[test]
    fn test_bucket_reuse_different_request_sizes() -> crate::Result<()> {
        // Two slightly-different-size requests should map to the same bucket
        // and reuse memory. This is the core mechanism Fix #A2 enables.
        let device = CudaDevice::new(0)?;
        // Pick two sizes that fall in the same 2 MiB bucket.
        // 600_000 f32 = 2.4 MiB → rounds to 4 MiB → bucket of 1_048_576 elems.
        // 700_000 f32 = 2.8 MiB → rounds to 4 MiB → same bucket.
        let size_a = 600_000usize;
        let size_b = 700_000usize;
        let bucket_a = round_elems_up::<f32>(size_a);
        let bucket_b = round_elems_up::<f32>(size_b);
        assert_eq!(bucket_a, bucket_b, "test setup: pick sizes in same bucket");

        let slice_a = pool_alloc_f32(&device, size_a)?;
        let ptr_a = *slice_a.device_ptr();
        assert_eq!(DeviceSlice::len(&slice_a), size_a);
        pool_return_f32(slice_a);

        let slice_b = pool_alloc_f32(&device, size_b)?;
        let ptr_b = *slice_b.device_ptr();
        assert_eq!(ptr_a, ptr_b, "expected bucket reuse — same device pointer");
        assert_eq!(DeviceSlice::len(&slice_b), size_b);

        pool_return_f32(slice_b);
        global_pool().clear_cache();
        Ok(())
    }

    /// Regression test for the Phase 2 round-2 ring-wrap double-free.
    ///
    /// Pre-fix (`HashSet<u64>`): registering the same ptr twice was a
    /// no-op; the first `unregister_external_ptr` cleared the entry,
    /// so the second `is_external_ptr(ptr)` returned `false`. Under
    /// `BlockOffloader` + ring wrap, the second return path tagged
    /// the FreeEntry non-external and `clear_cache` later called
    /// `free_async` on a ring-slab offset → CUDA_ERROR_INVALID_VALUE.
    ///
    /// Post-fix (`HashMap<u64, u32>` refcount): two registrations
    /// produce count=2; the first unregister leaves count=1 (ptr
    /// stays external); the second unregister removes the entry.
    /// `is_external_ptr` correctly returns true for the duration of
    /// either tensor's lifetime.
    #[test]
    fn test_external_ptr_refcount_under_ring_wrap() {
        let pool = global_pool();
        let fake_ptr: u64 = 0xdeadbeef_cafef00d;

        // Clean any prior state from other tests / threads.
        while pool.external_ptr_refcount(fake_ptr) > 0 {
            pool.unregister_external_ptr(fake_ptr);
        }
        assert_eq!(pool.external_ptr_refcount(fake_ptr), 0);
        assert!(!pool.is_external_ptr(fake_ptr));

        // Simulate ring wrap: alloc_bf16_via_ring is called twice with
        // the same physical address before either tensor is dropped.
        pool.register_external_ptr(fake_ptr);
        pool.register_external_ptr(fake_ptr);
        assert_eq!(pool.external_ptr_refcount(fake_ptr), 2);
        assert!(pool.is_external_ptr(fake_ptr));

        // First tensor drops → push_u16 external-guard unregisters.
        pool.unregister_external_ptr(fake_ptr);
        // Critical: ptr MUST still be tagged external for the second
        // tensor. This is the regression — pre-fix, this was false.
        assert!(
            pool.is_external_ptr(fake_ptr),
            "ptr must stay tagged external while a second live tensor \
             still holds it (ring-wrap double-allocation case)"
        );
        assert_eq!(pool.external_ptr_refcount(fake_ptr), 1);

        // Second tensor drops → final unregister.
        pool.unregister_external_ptr(fake_ptr);
        assert!(!pool.is_external_ptr(fake_ptr));
        assert_eq!(pool.external_ptr_refcount(fake_ptr), 0);

        // Extra unregister after count=0 must be a no-op (saturate).
        pool.unregister_external_ptr(fake_ptr);
        assert_eq!(pool.external_ptr_refcount(fake_ptr), 0);
    }
}
