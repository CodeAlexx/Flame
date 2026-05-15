//! Per-prefetch / per-step telemetry for [`BlockOffloader`](super::BlockOffloader).
//!
//! Ports the **measurement-and-observation** parts of FlexTensor's
//! `instrumentation/` package (registry + dumper) into a flame-core shape:
//! a process-global counter bag plus a bounded ring buffer of per-event
//! records. The strategic / state-machine parts of FlexTensor stay out of
//! scope for Phase 1 (see HANDOFF).
//!
//! Design goals (tenets §2, §4):
//!
//! * **Cheap by default.** Every counter increment is one atomic add.
//!   Disabled-mode adds a single `load(Relaxed)` on the path; the existing
//!   offloader hot path does not need to take a Mutex.
//! * **Opt-in for verbose tracing.** The bounded per-event ring buffer is
//!   only populated when `set_event_log_capacity(N)` was called with N > 0
//!   (or `FLAME_OFFLOAD_TELEMETRY=trace` is set in the environment at
//!   first-use). Counters always update — they are the bandwidth-bound
//!   measurement, not the per-event traces.
//! * **No `cudaStreamSynchronize` introduced.** Wall-time measurements use
//!   `std::time::Instant`; this records when the *host* observed the call
//!   start/finish, which is what telemetry should reflect for clauses 1 + 5
//!   of `SPEED_CONTRACT.md`. GPU-side timing belongs in
//!   `transfer_benchmark.rs`, not here.
//!
//! Hook points in [`BlockOffloader`](super::BlockOffloader):
//!
//! | Method | Hook |
//! |---|---|
//! | `prefetch_block` start | [`Telemetry::record_prefetch_begin`] |
//! | `prefetch_block` finish | [`Telemetry::record_prefetch_end`] (with bytes) |
//! | `await_block` start | [`Telemetry::record_await_begin`] |
//! | `await_block` end (slot hit) | [`Telemetry::record_await_end_hit`] |
//! | `await_block` end (slot miss) | [`Telemetry::record_await_end_miss`] |
//!
//! The hooks accept `&self` only; they take no offloader lock and do not
//! interact with any CUDA stream.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};

/// Aggregate counters covering every offloader call since process start.
///
/// All fields are bandwidth-bound (single atomic add per update). They are
/// safe to read concurrently with updates; callers that want a coherent
/// snapshot use [`Telemetry::snapshot`], which reads all fields under a
/// single ordering fence.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct TelemetryCounters {
    /// Total H2D bytes the offloader has issued via prefetch.
    pub h2d_bytes_total: u64,
    /// Total wall time the host thread spent inside `prefetch_block`, in
    /// nanoseconds. This is *issue cost* — the time to queue copies, not
    /// the GPU H2D wall time.
    pub prefetch_wall_ns: u64,
    /// Total wall time the host thread spent inside `await_block`, in
    /// nanoseconds. Includes the GPU-side `cudaStreamWaitEvent` gating,
    /// not a host sync.
    pub await_wall_ns: u64,
    /// Count of `await_block` calls that found the slot already prepared
    /// (no H2D issue needed). Higher is better — it means prefetch is
    /// landing in time.
    pub await_hits: u64,
    /// Count of `await_block` calls that had to issue an H2D themselves
    /// (no prior `prefetch_block` for the requested block).
    pub await_misses: u64,
    /// Count of `prefetch_block` calls accepted (i.e. not a same-slot
    /// no-op).
    pub prefetch_issued: u64,
    /// Count of `prefetch_block` calls that were short-circuited because
    /// the block was already resident on one of the slots.
    pub prefetch_already_resident: u64,

    // ──────────────────────────────────────────────────────────────
    // Phase 2 (strategy) counters. All default to zero when no
    // strategy is attached.
    // ──────────────────────────────────────────────────────────────
    /// Count of `Strategy::plan()` calls served (always one per
    /// non-resident `prefetch_block` when a strategy is attached).
    pub strategy_plans: u64,
    /// Total eviction decisions strategies have issued. Each
    /// `plan.evict.len()` accumulates here.
    pub strategy_eviction_decisions: u64,
    /// Sum of `plan.keep.len()` across every plan — the running total
    /// of "resident-set size after plan". Divide by `strategy_plans`
    /// for the average.
    pub strategy_keep_total: u64,
    /// Last reported `target_resident_bytes`. Strategies' adaptive
    /// behavior shows up as this value moving over time.
    pub strategy_last_target_resident_bytes: u64,
}

impl TelemetryCounters {
    /// Aggregate effective H2D bandwidth across the lifetime of the
    /// offloader, in bytes/sec. Returns `0.0` if no prefetch wall time
    /// has been recorded — there is no measurement to base a rate on.
    pub fn effective_h2d_bps(&self) -> f64 {
        if self.prefetch_wall_ns == 0 {
            return 0.0;
        }
        (self.h2d_bytes_total as f64) / (self.prefetch_wall_ns as f64 * 1e-9)
    }

    /// Fraction of `await_block` calls that landed on a pre-prepared slot.
    /// In `[0.0, 1.0]`. Returns `0.0` when no awaits have been recorded.
    pub fn await_hit_ratio(&self) -> f64 {
        let total = self.await_hits + self.await_misses;
        if total == 0 {
            return 0.0;
        }
        (self.await_hits as f64) / (total as f64)
    }
}

/// One per-event trace record. Populated only when the ring buffer is
/// enabled (see [`Telemetry::set_event_log_capacity`]).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TelemetryEvent {
    pub kind: TelemetryEventKind,
    /// Block index this event refers to.
    pub block_idx: u32,
    /// Bytes moved by this event (0 for await events).
    pub bytes: u64,
    /// Wall-clock duration the call took, in nanoseconds.
    pub duration_ns: u64,
}

/// Kind of telemetry event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TelemetryEventKind {
    PrefetchIssued,
    PrefetchAlreadyResident,
    AwaitHit,
    AwaitMiss,
}

/// Internal — mutable state for the event ring buffer. Kept inside a
/// `Mutex` because `TelemetryEvent` is a `Copy` struct but the bounded
/// ring needs both push and rotate, which is awkward to do lock-free
/// without `crossbeam`. The mutex is touched only on the trace path
/// (disabled by default).
#[derive(Default)]
struct EventLog {
    capacity: usize,
    events: Vec<TelemetryEvent>,
    /// Monotonically increasing total events seen — capacity-limited
    /// `events` is a window over the *tail* of this stream.
    total_seen: u64,
}

impl EventLog {
    fn push(&mut self, ev: TelemetryEvent) {
        if self.capacity == 0 {
            return;
        }
        self.total_seen = self.total_seen.saturating_add(1);
        if self.events.len() < self.capacity {
            self.events.push(ev);
        } else {
            // Rotate: drop oldest, push newest. With capacity ≤ a few
            // thousand this is cheap relative to a prefetch + H2D.
            let last_idx = (self.total_seen as usize - 1) % self.capacity;
            self.events[last_idx] = ev;
        }
    }
}

/// Per-process telemetry sink for offloader events.
///
/// Use [`global`] to access the shared instance. The struct is `Sync` and
/// can be cloned cheaply via `Arc` if a caller needs to install a sink in
/// multiple places.
pub struct Telemetry {
    enabled: AtomicUsize, // 0 = disabled, 1 = counters only, 2 = counters + event log

    h2d_bytes_total: AtomicU64,
    prefetch_wall_ns: AtomicU64,
    await_wall_ns: AtomicU64,
    await_hits: AtomicU64,
    await_misses: AtomicU64,
    prefetch_issued: AtomicU64,
    prefetch_already_resident: AtomicU64,

    // Phase 2: strategy decision counters.
    strategy_plans: AtomicU64,
    strategy_eviction_decisions: AtomicU64,
    strategy_keep_total: AtomicU64,
    strategy_last_target_resident_bytes: AtomicU64,

    event_log: Mutex<EventLog>,

    // Phase 4 (telemetry export): periodic dump bookkeeping.
    /// Cached value of `FLAME_OFFLOAD_TELEMETRY_DUMP_INTERVAL_EVENTS` (or
    /// the legacy alias `_STEPS`), read once at first-use. `0` disables
    /// periodic dumps. Counts EVENTS — every record_prefetch_end +
    /// record_await_end_{hit,miss} ticks the counter, not training steps.
    periodic_interval: AtomicU64,
    /// Cumulative count of events seen across record_prefetch_end /
    /// record_await_end_{hit,miss}. Used to decide when the next periodic
    /// dump fires. Increments unconditionally (cheap) but the dump only
    /// fires when interval > 0 and `events_since_dump % interval == 0`.
    events_since_dump: AtomicU64,
}

impl Telemetry {
    fn new() -> Self {
        let initial = match std::env::var("FLAME_OFFLOAD_TELEMETRY").ok().as_deref() {
            Some("off") | Some("0") | None => 0usize,
            Some("trace") => 2usize,
            // Any other non-empty value enables counters but not trace.
            _ => 1usize,
        };
        let capacity = std::env::var("FLAME_OFFLOAD_TELEMETRY_RING")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(if initial >= 2 { 4096 } else { 0 });
        // Periodic dump interval. New name is `_EVENTS` because the counter
        // ticks per record_prefetch_end / record_await_end_{hit,miss}, NOT
        // per training step. The legacy `_STEPS` alias is kept so existing
        // scripts and the OFFLOAD_GETTING_STARTED tutorial in `cde77a4`
        // keep working. New name takes precedence if both are set.
        let periodic_interval = std::env::var("FLAME_OFFLOAD_TELEMETRY_DUMP_INTERVAL_EVENTS")
            .ok()
            .or_else(|| std::env::var("FLAME_OFFLOAD_TELEMETRY_DUMP_INTERVAL_STEPS").ok())
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);
        Self {
            enabled: AtomicUsize::new(initial),
            h2d_bytes_total: AtomicU64::new(0),
            prefetch_wall_ns: AtomicU64::new(0),
            await_wall_ns: AtomicU64::new(0),
            await_hits: AtomicU64::new(0),
            await_misses: AtomicU64::new(0),
            prefetch_issued: AtomicU64::new(0),
            prefetch_already_resident: AtomicU64::new(0),
            strategy_plans: AtomicU64::new(0),
            strategy_eviction_decisions: AtomicU64::new(0),
            strategy_keep_total: AtomicU64::new(0),
            strategy_last_target_resident_bytes: AtomicU64::new(0),
            event_log: Mutex::new(EventLog {
                capacity,
                events: Vec::with_capacity(capacity),
                total_seen: 0,
            }),
            periodic_interval: AtomicU64::new(periodic_interval),
            events_since_dump: AtomicU64::new(0),
        }
    }

    /// Configure periodic dump interval (in events). `0` disables. When
    /// non-zero, `record_prefetch_end` / `record_await_end_*` will every
    /// `N` events trigger an atomic JSON dump of the current snapshot +
    /// event-log to the directory named by
    /// `FLAME_OFFLOAD_TELEMETRY_DUMP_DIR` (or the platform tmpdir if
    /// unset). The dump path itself is host-only I/O — no CUDA calls.
    pub fn set_periodic_dump_interval(&self, every_n_events: u64) {
        self.periodic_interval
            .store(every_n_events, Ordering::Release);
        self.events_since_dump.store(0, Ordering::Release);
    }

    /// Current periodic-dump interval. `0` means disabled.
    pub fn periodic_dump_interval(&self) -> u64 {
        self.periodic_interval.load(Ordering::Relaxed)
    }

    /// Check whether a periodic dump is due, and if so, perform it.
    /// Cheap-by-default: one relaxed atomic load checks if the feature
    /// is enabled at all; only on the every-N-events boundary does the
    /// host actually write JSON to disk. Errors are swallowed (telemetry
    /// is observe-only — a missing log file must not break training).
    #[inline]
    fn maybe_periodic_dump(&self) {
        let interval = self.periodic_interval.load(Ordering::Relaxed);
        if interval == 0 {
            return;
        }
        // fetch_add returns the prior value; the (n+1)-th call (1-indexed)
        // triggers at prior+1 == interval, then prior+1 == 2*interval, etc.
        let prior = self.events_since_dump.fetch_add(1, Ordering::AcqRel);
        let count = prior.wrapping_add(1);
        if count == 0 || count % interval != 0 {
            return;
        }
        // Best-effort dump — failures are logged via eprintln so a missing
        // directory doesn't go silent, but never propagated.
        if let Err(e) = dump_all_inner(self, None) {
            eprintln!("[flame-core][telemetry] periodic dump failed: {e}");
        }
    }

    /// Are counters currently being updated? Cheap (single relaxed load).
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed) > 0
    }

    /// Are per-event traces currently being recorded?
    #[inline]
    pub fn is_trace_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed) >= 2
    }

    /// Enable / disable telemetry capture entirely.
    pub fn set_enabled(&self, on: bool) {
        let new = if on { 1 } else { 0 };
        // Preserve trace mode if already higher.
        let cur = self.enabled.load(Ordering::Relaxed);
        if on && cur >= 2 {
            return;
        }
        self.enabled.store(new, Ordering::Release);
    }

    /// Resize the per-event trace ring buffer. Pass `0` to disable trace
    /// recording. A non-zero value enables it (and bumps the counter mode
    /// to at least 2).
    pub fn set_event_log_capacity(&self, capacity: usize) {
        let mut log = self.event_log.lock().unwrap();
        log.capacity = capacity;
        log.events.clear();
        log.total_seen = 0;
        if capacity > 0 {
            log.events.reserve(capacity);
            self.enabled.store(2, Ordering::Release);
        } else if self.enabled.load(Ordering::Relaxed) >= 2 {
            self.enabled.store(1, Ordering::Release);
        }
    }

    /// Take a coherent counter snapshot. Cheap: 11 atomic loads.
    pub fn snapshot(&self) -> TelemetryCounters {
        TelemetryCounters {
            h2d_bytes_total: self.h2d_bytes_total.load(Ordering::Acquire),
            prefetch_wall_ns: self.prefetch_wall_ns.load(Ordering::Acquire),
            await_wall_ns: self.await_wall_ns.load(Ordering::Acquire),
            await_hits: self.await_hits.load(Ordering::Acquire),
            await_misses: self.await_misses.load(Ordering::Acquire),
            prefetch_issued: self.prefetch_issued.load(Ordering::Acquire),
            prefetch_already_resident: self.prefetch_already_resident.load(Ordering::Acquire),
            strategy_plans: self.strategy_plans.load(Ordering::Acquire),
            strategy_eviction_decisions: self.strategy_eviction_decisions.load(Ordering::Acquire),
            strategy_keep_total: self.strategy_keep_total.load(Ordering::Acquire),
            strategy_last_target_resident_bytes: self
                .strategy_last_target_resident_bytes
                .load(Ordering::Acquire),
        }
    }

    /// Reset all counters and clear the event ring buffer. Does not change
    /// the enabled/trace mode.
    pub fn reset(&self) {
        self.h2d_bytes_total.store(0, Ordering::Release);
        self.prefetch_wall_ns.store(0, Ordering::Release);
        self.await_wall_ns.store(0, Ordering::Release);
        self.await_hits.store(0, Ordering::Release);
        self.await_misses.store(0, Ordering::Release);
        self.prefetch_issued.store(0, Ordering::Release);
        self.prefetch_already_resident.store(0, Ordering::Release);
        self.strategy_plans.store(0, Ordering::Release);
        self.strategy_eviction_decisions.store(0, Ordering::Release);
        self.strategy_keep_total.store(0, Ordering::Release);
        self.strategy_last_target_resident_bytes
            .store(0, Ordering::Release);
        let mut log = self.event_log.lock().unwrap();
        log.events.clear();
        log.total_seen = 0;
    }

    /// Hook: a [`Strategy`](super::strategy::Strategy) emitted a plan.
    /// Cheap when telemetry is disabled (single relaxed load + early
    /// return). Records an aggregate snapshot (no per-plan event in
    /// the ring buffer — strategy plans run inside the offloader lock
    /// and burn the event-log mutex unnecessarily).
    ///
    /// `_name` is the strategy's stable name; reserved for future
    /// per-strategy counter splits.
    pub fn record_strategy_decision(
        &self,
        _name: &'static str,
        evicted: u64,
        kept: u64,
        target_bytes: u64,
    ) {
        if !self.is_enabled() {
            return;
        }
        self.strategy_plans.fetch_add(1, Ordering::AcqRel);
        self.strategy_eviction_decisions
            .fetch_add(evicted, Ordering::AcqRel);
        self.strategy_keep_total.fetch_add(kept, Ordering::AcqRel);
        self.strategy_last_target_resident_bytes
            .store(target_bytes, Ordering::Release);
    }

    /// Copy the current contents of the per-event ring buffer. Empty when
    /// trace mode is off. The order is undefined relative to insertion
    /// once the ring wraps — callers that need ordering should sample
    /// before `capacity` events.
    pub fn event_log(&self) -> Vec<TelemetryEvent> {
        let log = self.event_log.lock().unwrap();
        log.events.clone()
    }

    /// Number of events the ring buffer has *observed*, even if rotated
    /// out. Always ≥ `event_log().len()`.
    pub fn total_events_seen(&self) -> u64 {
        let log = self.event_log.lock().unwrap();
        log.total_seen
    }

    // ------------------------------------------------------------------
    // Hooks called by BlockOffloader. Each is cheap when telemetry is
    // off (single relaxed load + early return).
    // ------------------------------------------------------------------

    /// Hook: start of `prefetch_block`. Returns an opaque timer the
    /// caller passes back to [`Self::record_prefetch_end`] /
    /// [`Self::record_prefetch_already_resident`].
    #[inline]
    pub fn record_prefetch_begin(&self) -> TelemetryTimer {
        TelemetryTimer::start(self.is_enabled())
    }

    /// Hook: end of `prefetch_block` after a real H2D issue.
    pub fn record_prefetch_end(&self, timer: TelemetryTimer, block_idx: usize, bytes: u64) {
        if !self.is_enabled() {
            return;
        }
        let dur_ns = timer.elapsed_ns();
        self.h2d_bytes_total.fetch_add(bytes, Ordering::AcqRel);
        self.prefetch_wall_ns.fetch_add(dur_ns, Ordering::AcqRel);
        self.prefetch_issued.fetch_add(1, Ordering::AcqRel);
        if self.is_trace_enabled() {
            self.event_log.lock().unwrap().push(TelemetryEvent {
                kind: TelemetryEventKind::PrefetchIssued,
                block_idx: block_idx as u32,
                bytes,
                duration_ns: dur_ns,
            });
        }
        self.maybe_periodic_dump();
    }

    /// Hook: end of `prefetch_block` when the block was already on a slot.
    pub fn record_prefetch_already_resident(&self, timer: TelemetryTimer, block_idx: usize) {
        if !self.is_enabled() {
            return;
        }
        let dur_ns = timer.elapsed_ns();
        self.prefetch_already_resident
            .fetch_add(1, Ordering::AcqRel);
        if self.is_trace_enabled() {
            self.event_log.lock().unwrap().push(TelemetryEvent {
                kind: TelemetryEventKind::PrefetchAlreadyResident,
                block_idx: block_idx as u32,
                bytes: 0,
                duration_ns: dur_ns,
            });
        }
    }

    /// Hook: start of `await_block`.
    #[inline]
    pub fn record_await_begin(&self) -> TelemetryTimer {
        TelemetryTimer::start(self.is_enabled())
    }

    /// Hook: end of `await_block`, slot was already prepared (hit).
    pub fn record_await_end_hit(&self, timer: TelemetryTimer, block_idx: usize) {
        if !self.is_enabled() {
            return;
        }
        let dur_ns = timer.elapsed_ns();
        self.await_wall_ns.fetch_add(dur_ns, Ordering::AcqRel);
        self.await_hits.fetch_add(1, Ordering::AcqRel);
        if self.is_trace_enabled() {
            self.event_log.lock().unwrap().push(TelemetryEvent {
                kind: TelemetryEventKind::AwaitHit,
                block_idx: block_idx as u32,
                bytes: 0,
                duration_ns: dur_ns,
            });
        }
        self.maybe_periodic_dump();
    }

    /// Hook: end of `await_block`, had to issue H2D internally (miss).
    pub fn record_await_end_miss(&self, timer: TelemetryTimer, block_idx: usize) {
        if !self.is_enabled() {
            return;
        }
        let dur_ns = timer.elapsed_ns();
        self.await_wall_ns.fetch_add(dur_ns, Ordering::AcqRel);
        self.await_misses.fetch_add(1, Ordering::AcqRel);
        if self.is_trace_enabled() {
            self.event_log.lock().unwrap().push(TelemetryEvent {
                kind: TelemetryEventKind::AwaitMiss,
                block_idx: block_idx as u32,
                bytes: 0,
                duration_ns: dur_ns,
            });
        }
        self.maybe_periodic_dump();
    }
}

/// Opaque host-time timer threaded through the begin/end pair. `start`
/// captures `Instant::now()` when telemetry is enabled, otherwise it
/// stores nothing and `elapsed_ns()` returns 0.
#[derive(Clone, Copy)]
pub struct TelemetryTimer {
    start: Option<Instant>,
}

impl TelemetryTimer {
    #[inline]
    fn start(enabled: bool) -> Self {
        Self {
            start: if enabled { Some(Instant::now()) } else { None },
        }
    }

    #[inline]
    fn elapsed_ns(&self) -> u64 {
        match self.start {
            Some(s) => {
                let d: Duration = s.elapsed();
                let secs = d.as_secs().saturating_mul(1_000_000_000);
                secs.saturating_add(u64::from(d.subsec_nanos()))
            }
            None => 0,
        }
    }
}

/// Access the process-global telemetry sink. Lazily initialized on first
/// call. Environment-controlled defaults (see module docs) apply at
/// first init.
pub fn global() -> &'static Telemetry {
    static GLOBAL: OnceCell<Telemetry> = OnceCell::new();
    GLOBAL.get_or_init(Telemetry::new)
}

/// Dump current counters to a `String` in a stable diagnostic format.
/// Intended for `eprintln!` / log output, not machine parsing.
pub fn format_counters(counters: &TelemetryCounters) -> String {
    let mb = (counters.h2d_bytes_total as f64) / (1024.0 * 1024.0);
    let prefetch_ms = counters.prefetch_wall_ns as f64 * 1e-6;
    let await_ms = counters.await_wall_ns as f64 * 1e-6;
    let bps = counters.effective_h2d_bps();
    let gbps = bps / 1e9;
    let hit_ratio = counters.await_hit_ratio();
    format!(
        "[offload-telemetry] h2d_total={:.1} MiB prefetch_wall={:.2} ms \
         await_wall={:.2} ms eff_h2d≈{:.3} GB/s await_hit_ratio={:.3} \
         (issued={}, resident={}, hits={}, misses={})",
        mb,
        prefetch_ms,
        await_ms,
        gbps,
        hit_ratio,
        counters.prefetch_issued,
        counters.prefetch_already_resident,
        counters.await_hits,
        counters.await_misses,
    )
}

// ----------------------------------------------------------------------------
// Telemetry export (Phase 4, 2026-05-12)
// ----------------------------------------------------------------------------
//
// Make counters and event traces visible from outside the process without
// requiring source edits in every trainer:
//
//  * `snapshot_to_file(path)`     — single JSON document, the counter snapshot.
//  * `ring_buffer_to_file(path)`  — JSON-lines, one event per line.
//  * `dump_all(dir)`              — convenience pair into one directory.
//
// All three are atomic: write-to-tmp-file + rename. A SIGKILL mid-write
// leaves the previous file intact. No CUDA calls anywhere on the export
// path (clause 1 of SPEED_CONTRACT — host I/O only).

/// File name written by [`dump_all`] for the counter snapshot.
pub const DUMP_SNAPSHOT_FILENAME: &str = "flame_offload_telemetry_snapshot.json";

/// File name written by [`dump_all`] for the per-event ring buffer.
pub const DUMP_EVENTS_FILENAME: &str = "flame_offload_telemetry_events.jsonl";

/// Environment variable read by [`dump_all`] when the explicit directory is
/// `None`. Falls back to the platform temp directory if also unset.
pub const DUMP_DIR_ENV: &str = "FLAME_OFFLOAD_TELEMETRY_DUMP_DIR";

/// Environment variable for the periodic-dump interval. When set to a
/// positive integer, an end-of-event hook every Nth event will write
/// `dump_all` to the configured directory. Read once at [`global`]
/// initialization; runtime overrides go via
/// [`Telemetry::set_periodic_dump_interval`].
/// Primary env var name for the periodic-dump interval (counts events, not
/// training steps — see [`Telemetry::set_periodic_dump_interval`]). The
/// legacy alias `FLAME_OFFLOAD_TELEMETRY_DUMP_INTERVAL_STEPS` is also
/// recognized for back-compat with the v1 release of this module.
pub const DUMP_INTERVAL_ENV: &str = "FLAME_OFFLOAD_TELEMETRY_DUMP_INTERVAL_EVENTS";
/// Back-compat alias for [`DUMP_INTERVAL_ENV`]. Recognized but misleadingly
/// named — counts events, not steps. New code should use `DUMP_INTERVAL_ENV`.
#[deprecated(note = "use DUMP_INTERVAL_ENV instead; STEPS was a misnomer (counter is per-event)")]
pub const DUMP_INTERVAL_ENV_LEGACY: &str = "FLAME_OFFLOAD_TELEMETRY_DUMP_INTERVAL_STEPS";

/// Write `path` atomically. Serializes `value` to JSON in a sibling tmp
/// file, then `rename`s into place. The rename is atomic on POSIX
/// filesystems; on Windows it's a "best-effort replace".
fn write_json_atomic<P: AsRef<Path>, T: Serialize>(path: P, value: &T) -> std::io::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let tmp = path.with_extension("tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        let json = serde_json::to_vec_pretty(value)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("serde: {e}")))?;
        f.write_all(&json)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Serialize the global telemetry counter snapshot to `path`. Atomic
/// (tmp file + rename). Cheap — one snapshot read + one short JSON
/// document. No GPU work.
pub fn snapshot_to_file(path: &Path) -> std::io::Result<()> {
    let snap = global().snapshot();
    write_json_atomic(path, &snap)
}

/// Drain the per-event ring buffer to `path` as JSON-lines. Each line is
/// one [`TelemetryEvent`]. Returns the number of events written; `0` when
/// trace mode is off or the buffer is empty.
///
/// Atomic (tmp file + rename). The buffer itself is not cleared; the same
/// events will appear again on the next call if no new events landed.
pub fn ring_buffer_to_file(path: &Path) -> std::io::Result<usize> {
    let events = global().event_log();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let tmp = path.with_extension("tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        for ev in &events {
            let line = serde_json::to_string(ev).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::Other, format!("serde: {e}"))
            })?;
            f.write_all(line.as_bytes())?;
            f.write_all(b"\n")?;
        }
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(events.len())
}

/// Convenience end-of-run dump. Writes the counter snapshot
/// (`flame_offload_telemetry_snapshot.json`) and the per-event ring
/// buffer (`flame_offload_telemetry_events.jsonl`) into `dir`. When
/// `dir` is `None`, falls back to `$FLAME_OFFLOAD_TELEMETRY_DUMP_DIR`,
/// or the platform `std::env::temp_dir()` if that's unset too.
///
/// Returns the directory that was actually used. The directory is
/// created if it does not exist.
pub fn dump_all(dir: Option<&Path>) -> std::io::Result<PathBuf> {
    dump_all_inner(global(), dir)
}

fn dump_all_inner(t: &Telemetry, dir: Option<&Path>) -> std::io::Result<PathBuf> {
    let chosen: PathBuf = match dir {
        Some(p) => p.to_path_buf(),
        None => match std::env::var_os(DUMP_DIR_ENV) {
            Some(v) => PathBuf::from(v),
            None => std::env::temp_dir(),
        },
    };
    std::fs::create_dir_all(&chosen)?;

    let snap_path = chosen.join(DUMP_SNAPSHOT_FILENAME);
    let snap = t.snapshot();
    write_json_atomic(&snap_path, &snap)?;

    let events_path = chosen.join(DUMP_EVENTS_FILENAME);
    let events = t.event_log();
    let tmp = events_path.with_extension("tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        for ev in &events {
            let line = serde_json::to_string(ev).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::Other, format!("serde: {e}"))
            })?;
            f.write_all(line.as_bytes())?;
            f.write_all(b"\n")?;
        }
        f.sync_all()?;
    }
    std::fs::rename(&tmp, &events_path)?;

    Ok(chosen)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Counters update correctly when enabled.
    #[test]
    fn counters_basic_lifecycle() {
        let t = Telemetry::new();
        t.set_enabled(true);
        let timer = t.record_prefetch_begin();
        std::thread::sleep(std::time::Duration::from_micros(10));
        t.record_prefetch_end(timer, 7, 4 * 1024 * 1024);

        let snap = t.snapshot();
        assert_eq!(snap.prefetch_issued, 1);
        assert_eq!(snap.h2d_bytes_total, 4 * 1024 * 1024);
        assert!(snap.prefetch_wall_ns > 0);
        assert!(snap.effective_h2d_bps() > 0.0);
    }

    /// Disabled mode is a true no-op for counters.
    #[test]
    fn disabled_mode_no_counters() {
        let t = Telemetry::new();
        t.set_enabled(false);
        let timer = t.record_prefetch_begin();
        t.record_prefetch_end(timer, 0, 1024 * 1024);
        let snap = t.snapshot();
        assert_eq!(snap.prefetch_issued, 0);
        assert_eq!(snap.h2d_bytes_total, 0);
    }

    /// Event log only populates in trace mode.
    #[test]
    fn event_log_only_in_trace_mode() {
        let t = Telemetry::new();
        t.set_enabled(true);

        let timer = t.record_await_begin();
        t.record_await_end_hit(timer, 3);
        assert!(t.event_log().is_empty(), "trace off → no events captured");
        assert_eq!(t.snapshot().await_hits, 1);

        t.set_event_log_capacity(16);
        let timer = t.record_await_begin();
        t.record_await_end_miss(timer, 9);
        let log = t.event_log();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].kind, TelemetryEventKind::AwaitMiss);
        assert_eq!(log[0].block_idx, 9);
    }

    /// Hit-ratio math.
    #[test]
    fn hit_ratio_math() {
        let t = Telemetry::new();
        t.set_enabled(true);
        for _ in 0..3 {
            let timer = t.record_await_begin();
            t.record_await_end_hit(timer, 0);
        }
        for _ in 0..1 {
            let timer = t.record_await_begin();
            t.record_await_end_miss(timer, 0);
        }
        let snap = t.snapshot();
        assert!((snap.await_hit_ratio() - 0.75).abs() < 1e-9);
    }

    /// `format_counters` produces non-empty output.
    #[test]
    fn format_counters_non_empty() {
        let mut c = TelemetryCounters::default();
        c.h2d_bytes_total = 1024;
        c.prefetch_wall_ns = 1_000_000;
        c.await_hits = 1;
        let s = format_counters(&c);
        assert!(s.contains("offload-telemetry"));
        assert!(s.contains("h2d_total"));
    }
}
