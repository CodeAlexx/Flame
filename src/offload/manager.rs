//! `OffloadManager` — Phase 3 of the FlexTensor port.
//!
//! Wraps a [`BlockOffloader`](super::BlockOffloader) with a four-state
//! lifecycle (Discovery → Profiling → Active) and an opt-in
//! auto-strategy selector. This is the *autonomous policy* layer for
//! heavy memory-pressure workloads (sensenova_u1 @ 2048², hidream-o1 32B,
//! ltx2/wan22 video DiTs) — trainers ask the manager to "figure it out"
//! instead of hand-configuring `set_strategy` per model.
//!
//! ## Scope
//!
//! ### What this does
//!
//! * Owns and exposes a `BlockOffloader`. Existing
//!   `BlockOffloader::set_strategy` and direct construction are
//!   unaffected — the manager is the opt-in convenience layer.
//! * State machine: `NotInitialized → Discovery → Profiling → Active`.
//!   Each transition is explicit (`discover`, `profile`, `activate`)
//!   and records into the global [`telemetry`](super::telemetry) sink
//!   so the lifecycle is visible in the event ring.
//! * **Discovery** — snapshots the offloader's block geometry (sizes
//!   from `BlockOffloader::block_sizes`). Flame-core has no
//!   `__torch_function__` tensor crawler; block IDs are stable inputs
//!   declared via [`BlockFacilitator`](super::BlockFacilitator).
//! * **Profiling** — runs (or loads from cache) the
//!   [`transfer_benchmark`](super::transfer_benchmark) PCIe sweep and
//!   stores a [`TransferBandwidthProfile`].
//! * **Active** — picks a [`Strategy`](super::strategy::Strategy) given
//!   current VRAM headroom and block geometry, and attaches it to the
//!   underlying offloader. The default selector is:
//!     * If `2 × max_block_bytes < 0.3 × free_VRAM` → [`TwoSlot`].
//!     * Otherwise → [`Adaptive`].
//!
//! ### What it deliberately does NOT do (out of FlexTensor scope)
//!
//! * Walk PyTorch's tensor graph (`tensor_discovery.py`,
//!   `trap_tensor_mode.py`) — flame-core declares geometry explicitly.
//! * Multi-process / shared-memory coordination (`shm/`, `_ShmCoordinator`)
//!   — flame-core trainers are single-process.
//! * Sub-iteration metering that auto-transitions between phases. The
//!   manager's transitions are caller-driven, not iter-counter-driven.
//! * Modify the existing `BlockOffloader` public API.
//!
//! ## Tenet alignment
//!
//! * **Sync (clause 1).** State transitions are pure host work. The
//!   single CUDA touchpoint is the optional VRAM probe via
//!   [`crate::cuda::utils::cuda_mem_get_info`] (a non-blocking driver
//!   query) at the [`activate`](OffloadManager::activate) decision
//!   point. No `cudaStreamSynchronize`.
//! * **API (tenet §2).** Trainers opt in by constructing an
//!   `OffloadManager`. Existing `BlockOffloader` callers are
//!   bit-identical to pre-Phase-3 behavior.
//! * **Dispatcher (tenet §3).** The manager is the dispatcher for
//!   strategy selection — fix the policy here, every model with a
//!   manager gets the new policy.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use cudarc::driver::CudaDevice;

use super::strategy::{Adaptive, Strategy, TwoSlot};
use super::telemetry;
use super::transfer_benchmark::{self, BenchmarkConfig, TransferBandwidthProfile};
use super::{state, BlockFacilitator, BlockOffloader};

// ───────────────────────────────────────────────────────────────────────
// Phase enum
// ───────────────────────────────────────────────────────────────────────

/// Lifecycle phase of an [`OffloadManager`].
///
/// FlexTensor's upstream `OffloadPhase` enum has `INFERENCE` as the final
/// phase. flame-core does not distinguish inference vs training at this
/// layer — both run through the same `BlockOffloader` slot mechanic.
/// `Active` subsumes both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffloadPhase {
    /// Initial state. No discovery has been performed; the offloader is
    /// available via [`OffloadManager::offloader_mut`] but the manager
    /// has no opinion about it.
    NotInitialized,
    /// Discovery has completed; block sizes have been snapshotted from
    /// the underlying offloader. Profiling has not yet been done.
    Discovery,
    /// PCIe bandwidth profiling has completed. The
    /// [`TransferBandwidthProfile`] is available via
    /// [`OffloadManager::profile`].
    Profiling,
    /// A strategy has been auto-selected and attached to the underlying
    /// offloader. The manager's job is done; the trainer can pull
    /// `offloader_mut` and proceed.
    Active,
}

impl OffloadPhase {
    /// Short stable label for telemetry / log lines.
    pub fn as_str(self) -> &'static str {
        match self {
            OffloadPhase::NotInitialized => "NotInitialized",
            OffloadPhase::Discovery => "Discovery",
            OffloadPhase::Profiling => "Profiling",
            OffloadPhase::Active => "Active",
        }
    }
}

// ───────────────────────────────────────────────────────────────────────
// Config knobs
// ───────────────────────────────────────────────────────────────────────

/// Construction-time knobs for [`OffloadManager`].
#[derive(Debug, Clone)]
pub struct ManagerConfig {
    /// If `Some`, the manager will try to load this profile cache during
    /// [`OffloadManager::profile`] before running a fresh sweep. On load
    /// failure (missing, corrupt, schema mismatch) the bench runs and
    /// the result overwrites the file. `None` disables the cache —
    /// every `profile()` call runs the sweep.
    pub profile_cache_path: Option<PathBuf>,
    /// Knobs forwarded to [`transfer_benchmark::run_benchmark`] when
    /// a sweep runs. Default is `BenchmarkConfig::default()`.
    pub benchmark: BenchmarkConfig,
    /// Bytes of VRAM the caller wants kept clear *after* the
    /// offloader's residency. Used by the auto-strategy selector to
    /// compute the effective free budget. Defaults to 256 MiB
    /// (CUDA driver overhead + small workspace bumps).
    pub vram_headroom_bytes: u64,
    /// If `Some`, override the auto-strategy decision and always
    /// install this strategy on [`OffloadManager::activate`]. The
    /// auto-selector is bypassed entirely.
    ///
    /// Use for tests / regression gates that want a deterministic
    /// strategy regardless of host state.
    pub force_strategy: Option<ForcedStrategy>,
}

impl Default for ManagerConfig {
    fn default() -> Self {
        Self {
            profile_cache_path: Some(state::default_profile_path()),
            benchmark: BenchmarkConfig::default(),
            vram_headroom_bytes: 256 * 1024 * 1024,
            force_strategy: None,
        }
    }
}

/// Selector for an explicit strategy choice in [`ManagerConfig`].
/// Trait objects can't impl `Clone`, so the manager carries an enum and
/// constructs the trait object inside [`OffloadManager::activate`].
#[derive(Debug, Clone, Copy)]
pub enum ForcedStrategy {
    TwoSlot,
    Adaptive,
}

// ───────────────────────────────────────────────────────────────────────
// Manager
// ───────────────────────────────────────────────────────────────────────

/// State-machine wrapper around a [`BlockOffloader`].
///
/// Construct via [`OffloadManager::new`]; drive through the phases by
/// calling [`discover`](OffloadManager::discover),
/// [`profile`](OffloadManager::profile),
/// [`activate`](OffloadManager::activate) in order. After `activate`
/// the trainer pulls [`offloader_mut`](OffloadManager::offloader_mut)
/// and uses the offloader exactly as before.
pub struct OffloadManager {
    config: ManagerConfig,
    device: Arc<CudaDevice>,
    offloader: BlockOffloader,
    phase: OffloadPhase,

    /// Cached per-block sizes captured during [`discover`]. Empty until
    /// then. Lives here (not just on the offloader) so the manager's
    /// strategy selector can reason about block geometry without
    /// re-querying the offloader.
    block_sizes: Vec<usize>,

    /// Bandwidth profile from [`profile`]. `None` until profiling runs.
    profile: Option<TransferBandwidthProfile>,

    /// Name of the strategy installed in [`activate`]. `"none"` before
    /// that.
    active_strategy_name: &'static str,
}

impl OffloadManager {
    /// Construct a manager around an *already-loaded* [`BlockOffloader`].
    ///
    /// The manager starts in [`OffloadPhase::NotInitialized`]; the
    /// caller must drive `discover → profile → activate` before the
    /// auto-strategy is installed. During that window, the offloader is
    /// fully usable via [`Self::offloader_mut`] — the manager's only
    /// invariant is that it does not touch the offloader's strategy
    /// surface until `activate`.
    pub fn new(device: Arc<CudaDevice>, offloader: BlockOffloader) -> Self {
        Self::with_config(device, offloader, ManagerConfig::default())
    }

    /// Like [`Self::new`] but with explicit config.
    pub fn with_config(
        device: Arc<CudaDevice>,
        offloader: BlockOffloader,
        config: ManagerConfig,
    ) -> Self {
        Self {
            config,
            device,
            offloader,
            phase: OffloadPhase::NotInitialized,
            block_sizes: Vec::new(),
            profile: None,
            active_strategy_name: "none",
        }
    }

    /// Convenience: load + wrap in one call. Mirrors
    /// [`BlockOffloader::load`] with default [`ManagerConfig`].
    pub fn load(
        paths: &[&str],
        facilitator: &dyn BlockFacilitator,
        device: Arc<CudaDevice>,
    ) -> anyhow::Result<Self> {
        let offloader = BlockOffloader::load(paths, facilitator, device.clone())?;
        Ok(Self::new(device, offloader))
    }

    // ─────────────────────────── Accessors ──────────────────────────────

    /// Current lifecycle phase.
    pub fn phase(&self) -> OffloadPhase {
        self.phase
    }

    /// Borrow the underlying offloader immutably.
    pub fn offloader(&self) -> &BlockOffloader {
        &self.offloader
    }

    /// Borrow the underlying offloader mutably. Trainers call
    /// `prefetch_block` / `await_block` / `ensure_block` through this
    /// handle exactly as they would on a raw `BlockOffloader`.
    pub fn offloader_mut(&mut self) -> &mut BlockOffloader {
        &mut self.offloader
    }

    /// Cached block sizes captured at [`discover`]. Empty before
    /// discovery has run.
    pub fn block_sizes(&self) -> &[usize] {
        &self.block_sizes
    }

    /// PCIe bandwidth profile produced (or loaded) by [`Self::run_profile`].
    /// `None` before profiling has run.
    pub fn bandwidth_profile(&self) -> Option<&TransferBandwidthProfile> {
        self.profile.as_ref()
    }

    /// Name of the currently-installed strategy. Returns `"none"` until
    /// [`activate`] runs.
    pub fn active_strategy_name(&self) -> &'static str {
        self.active_strategy_name
    }

    /// Consume the manager and return the inner offloader. Use when the
    /// trainer no longer needs the manager (e.g. discovery/profile/
    /// activate have run and the strategy is now wired in).
    pub fn into_offloader(self) -> BlockOffloader {
        self.offloader
    }

    // ─────────────────────────── Transitions ────────────────────────────

    /// **Phase 1: discovery.** Snapshots the offloader's block geometry
    /// (per-block byte sizes) so the manager's strategy selector can
    /// reason about it without re-querying the offloader. Equivalent of
    /// FlexTensor's `tensor_discovery` for flame-core's
    /// explicit-block-ID model.
    ///
    /// Idempotent: safe to call multiple times; later calls re-snapshot.
    /// Always sets `phase = Discovery` on return.
    pub fn discover(&mut self) -> anyhow::Result<()> {
        self.block_sizes = self.offloader.block_sizes().to_vec();
        self.phase = OffloadPhase::Discovery;
        log::info!(
            "[offload-manager] discover: {} blocks, max_block={} bytes, total={} bytes",
            self.block_sizes.len(),
            self.block_sizes.iter().copied().max().unwrap_or(0),
            self.block_sizes.iter().sum::<usize>(),
        );
        record_transition_event("discover", self.block_sizes.len());
        Ok(())
    }

    /// **Phase 2: profiling.** Try to load a cached
    /// [`TransferBandwidthProfile`] from `config.profile_cache_path`.
    /// On load failure (or when the cache is disabled), run the
    /// `transfer_benchmark` sweep and persist the result back to the
    /// cache.
    ///
    /// Requires that [`Self::discover`] has run.
    pub fn run_profile(&mut self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.phase >= OffloadPhase::Discovery,
            "OffloadManager::profile requires discover() first (phase={:?})",
            self.phase
        );

        // 1) Try the cache.
        if let Some(path) = self.config.profile_cache_path.clone() {
            match state::load_profile(&path) {
                Ok(p) => {
                    log::info!(
                        "[offload-manager] profile: loaded cache from {} (peak h2d={:.2} GB/s)",
                        path.display(),
                        p.peak_h2d_bps / 1e9
                    );
                    self.profile = Some(p);
                    self.phase = OffloadPhase::Profiling;
                    record_transition_event("profile_cached", 0);
                    return Ok(());
                }
                Err(e) => {
                    log::info!(
                        "[offload-manager] profile cache miss ({}): running fresh bench",
                        e
                    );
                }
            }
        }

        // 2) Run the bench.
        let p = transfer_benchmark::run_benchmark(&self.device, &self.config.benchmark)
            .context("transfer_benchmark::run_benchmark")?;
        log::info!(
            "[offload-manager] profile: bench done — peak h2d={:.2} GB/s, peak d2h={:.2} GB/s",
            p.peak_h2d_bps / 1e9,
            p.peak_d2h_bps / 1e9,
        );

        // 3) Persist (best effort — IO failure is logged but not fatal).
        if let Some(path) = self.config.profile_cache_path.clone() {
            if let Err(e) = state::save_profile(&path, &p, "flame-core/offload-manager") {
                log::warn!(
                    "[offload-manager] failed to persist profile to {}: {e}",
                    path.display()
                );
            } else {
                log::info!("[offload-manager] persisted profile to {}", path.display());
            }
        }

        self.profile = Some(p);
        self.phase = OffloadPhase::Profiling;
        record_transition_event("profile_measured", 0);
        Ok(())
    }

    /// **Phase 3: activate.** Pick a [`Strategy`] given the current
    /// VRAM headroom and block geometry, and attach it to the
    /// underlying offloader via
    /// [`BlockOffloader::set_strategy`](super::BlockOffloader::set_strategy).
    ///
    /// Decision logic (overridden when `config.force_strategy` is `Some`):
    ///
    /// * If `2 × max_block_bytes < 0.3 × free_VRAM_after_headroom` →
    ///   [`TwoSlot`]. The 2-slot static pipeline is fine when capacity
    ///   is comfortable.
    /// * Otherwise → [`Adaptive`]. Headroom is tight at the current
    ///   model + activation shape; let the strategy track pressure and
    ///   shrink resident-set when needed.
    ///
    /// Requires that [`Self::discover`] and [`Self::run_profile`] have run.
    pub fn activate(&mut self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.phase >= OffloadPhase::Profiling,
            "OffloadManager::activate requires run_profile() first (phase={:?})",
            self.phase
        );

        let choice = match self.config.force_strategy {
            Some(ForcedStrategy::TwoSlot) => ForcedStrategy::TwoSlot,
            Some(ForcedStrategy::Adaptive) => ForcedStrategy::Adaptive,
            None => self.auto_select_strategy(),
        };

        match choice {
            ForcedStrategy::TwoSlot => {
                let s = Box::new(TwoSlot::new());
                self.active_strategy_name = s.name();
                self.offloader.set_strategy(s);
            }
            ForcedStrategy::Adaptive => {
                let s = Box::new(Adaptive::new());
                self.active_strategy_name = s.name();
                self.offloader.set_strategy(s);
            }
        }
        self.phase = OffloadPhase::Active;
        log::info!(
            "[offload-manager] activate: strategy={} blocks={} max_block={} bytes",
            self.active_strategy_name,
            self.block_sizes.len(),
            self.block_sizes.iter().copied().max().unwrap_or(0),
        );
        record_transition_event("activate", 0);
        Ok(())
    }

    /// One-shot convenience: discover → run_profile → activate.
    pub fn discover_profile_activate(&mut self) -> anyhow::Result<()> {
        self.discover()?;
        self.run_profile()?;
        self.activate()?;
        Ok(())
    }

    // ───────────────────────── Internal: auto-select ────────────────────

    /// Decide which strategy to install based on VRAM headroom vs
    /// max block size. See [`Self::activate`] for the rule.
    ///
    /// Falls back to `TwoSlot` when the VRAM probe fails (e.g.
    /// non-CUDA build, driver error) — preserves pre-Phase-3 behavior
    /// for callers in degraded environments.
    fn auto_select_strategy(&self) -> ForcedStrategy {
        let max_block_bytes = self.block_sizes.iter().copied().max().unwrap_or(0) as u64;

        // No blocks (e.g. an empty model) → strategy is moot, default
        // to TwoSlot (cheapest). Bit-identical to no strategy.
        if max_block_bytes == 0 {
            return ForcedStrategy::TwoSlot;
        }

        let (free_bytes, _total_bytes) = match crate::cuda::utils::cuda_mem_get_info() {
            Ok(t) => (t.0 as u64, t.1 as u64),
            Err(_e) => {
                log::warn!("[offload-manager] cuda_mem_get_info failed — defaulting to TwoSlot");
                return ForcedStrategy::TwoSlot;
            }
        };
        let effective_free = free_bytes.saturating_sub(self.config.vram_headroom_bytes);

        // Decision: if the two largest slots' static footprint fits
        // comfortably (< 30%) inside effective free VRAM, the 2-slot
        // pipeline doesn't need adaptive sizing. Otherwise pressure is
        // real and `Adaptive` earns its keep.
        let static_two_slot_bytes = max_block_bytes.saturating_mul(2);
        let comfort_budget = (effective_free as f64 * 0.30) as u64;
        if static_two_slot_bytes < comfort_budget {
            ForcedStrategy::TwoSlot
        } else {
            ForcedStrategy::Adaptive
        }
    }
}

impl PartialOrd for OffloadPhase {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.rank().cmp(&other.rank()))
    }
}
impl Ord for OffloadPhase {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.rank().cmp(&other.rank())
    }
}
impl OffloadPhase {
    fn rank(self) -> u8 {
        match self {
            OffloadPhase::NotInitialized => 0,
            OffloadPhase::Discovery => 1,
            OffloadPhase::Profiling => 2,
            OffloadPhase::Active => 3,
        }
    }
}

// ───────────────────────────────────────────────────────────────────────
// Telemetry plumbing
// ───────────────────────────────────────────────────────────────────────

/// Push a one-shot transition event into the telemetry sink.
///
/// We reuse the existing `record_strategy_decision` hook because (a) it's
/// the only Phase-3-friendly sink already in `telemetry` and (b) we don't
/// want to expand the telemetry surface for one-call lifecycle events.
/// The fields are repurposed:
///   - `evicted` carries `0` (no eviction implied)
///   - `kept` carries the block count for `discover`, `0` otherwise
///   - `target_resident_bytes` carries `0` (the manager has no opinion
///     about resident bytes at transition time)
///
/// This is intentionally lossy at the type level — the transition is
/// logged via `log::info!` for the human-readable trail, while the
/// counter bump gives a programmatic confirmation.
fn record_transition_event(name: &'static str, block_count: usize) {
    let t = telemetry::global();
    if !t.is_enabled() {
        return;
    }
    t.record_strategy_decision(name, 0, block_count as u64, 0);
}

// ───────────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Phase ordering is total and matches the lifecycle.
    #[test]
    fn phase_ordering_matches_lifecycle() {
        assert!(OffloadPhase::NotInitialized < OffloadPhase::Discovery);
        assert!(OffloadPhase::Discovery < OffloadPhase::Profiling);
        assert!(OffloadPhase::Profiling < OffloadPhase::Active);
    }

    /// Phase string labels are stable.
    #[test]
    fn phase_as_str_is_stable() {
        assert_eq!(OffloadPhase::NotInitialized.as_str(), "NotInitialized");
        assert_eq!(OffloadPhase::Discovery.as_str(), "Discovery");
        assert_eq!(OffloadPhase::Profiling.as_str(), "Profiling");
        assert_eq!(OffloadPhase::Active.as_str(), "Active");
    }

    /// Default config plumbs a non-None profile cache path.
    #[test]
    fn default_config_has_cache_path() {
        let c = ManagerConfig::default();
        assert!(c.profile_cache_path.is_some());
        assert!(c.vram_headroom_bytes > 0);
    }
}
