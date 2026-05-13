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

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use once_cell::sync::OnceCell;

/// Aggregate counters covering every offloader call since process start.
///
/// All fields are bandwidth-bound (single atomic add per update). They are
/// safe to read concurrently with updates; callers that want a coherent
/// snapshot use [`Telemetry::snapshot`], which reads all fields under a
/// single ordering fence.
#[derive(Debug, Default, Clone)]
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
#[derive(Debug, Clone, Copy)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}

impl Telemetry {
    fn new() -> Self {
        let initial = match std::env::var("FLAME_OFFLOAD_TELEMETRY")
            .ok()
            .as_deref()
        {
            Some("off") | Some("0") | None => 0usize,
            Some("trace") => 2usize,
            // Any other non-empty value enables counters but not trace.
            _ => 1usize,
        };
        let capacity = std::env::var("FLAME_OFFLOAD_TELEMETRY_RING")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(if initial >= 2 { 4096 } else { 0 });
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
            prefetch_already_resident: self
                .prefetch_already_resident
                .load(Ordering::Acquire),
            strategy_plans: self.strategy_plans.load(Ordering::Acquire),
            strategy_eviction_decisions: self
                .strategy_eviction_decisions
                .load(Ordering::Acquire),
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
