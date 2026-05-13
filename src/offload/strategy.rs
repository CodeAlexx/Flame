//! Offloader resident-set strategies.
//!
//! Phase 2 of the FlexTensor port. The "strategy" layer lets callers
//! change *which* blocks are resident on the GPU and *how many* at once,
//! bounded by measured VRAM headroom and access pattern. The 2-slot
//! double-buffer in [`BlockOffloader`](super::BlockOffloader) is the
//! default; strategies are an opt-in upgrade for heavy memory-pressure
//! cases (sensenova_u1 @ 2048², hidream-o1 32B mixed-precision experts,
//! ltx2/wan22 video DiTs, etc).
//!
//! ## What this is — and isn't
//!
//! This is the **algorithmic** half of FlexTensor's strategy layer:
//!
//! * A trait — [`Strategy`] — that turns offloader state into a plan.
//! * Three implementations:
//!     - [`two_slot::TwoSlot`] — the current 2-slot behavior reformulated
//!       as a [`Strategy`] impl, kept as the default. Bit-identical to
//!       the pre-Phase-2 hardcoded path.
//!     - [`knapsack::Knapsack`] — value-based resident-set selection.
//!     - [`adaptive::Adaptive`] — VRAM-pressure-driven sizing.
//! * Telemetry hooks — strategy decisions emit through
//!   [`super::telemetry`] under a small extension surface so
//!   per-strategy effectiveness can be measured without per-PR vigilance.
//!
//! It is **not** FlexTensor's `OffloadManager` state machine
//! (deferred to Phase 3), `state_handler.py` persistence (Phase 3),
//! `tensor_discovery` / `trap_tensor_mode` (rely on Python
//! `__torch_function__`; flame-core knows its block geometry through
//! [`BlockFacilitator`](super::BlockFacilitator)), or the `shm/`
//! cross-process plumbing (single-process trainers only).
//!
//! ## Tenet alignment
//!
//! * **Sync (clause 1).** Strategy `plan()` runs pure host logic — no
//!   CUDA launches, no `cudaStreamSynchronize`, no event work. The
//!   single CUDA touchpoint is the optional VRAM probe in
//!   [`adaptive::Adaptive`] via
//!   [`crate::cuda::utils::cuda_mem_get_info`], which is itself a
//!   non-blocking driver query.
//! * **Dtype (clause 2).** No tensor ops; no F32 detours.
//! * **API (tenet §2).** When no strategy is set, the offloader runs the
//!   pre-Phase-2 code paths unchanged. Strategies are opt-in via
//!   [`super::BlockOffloader::set_strategy`].
//!
//! ## Strategy interface
//!
//! A [`Strategy`] is invoked with [`OffloaderState`] — a cheap snapshot of
//! the current slot/access state — and returns a [`ResidentPlan`] saying
//! which block IDs the offloader should ideally evict / keep / prefetch.
//! The offloader treats the plan as advisory: when it disagrees with the
//! 2-slot ping-pong (e.g. tells us to evict the active block), the
//! offloader ignores that suggestion. Strategies steer; they do not
//! commandeer.

use std::collections::VecDeque;

pub mod adaptive;
pub mod knapsack;
pub mod two_slot;

// Re-exports so call sites can `use flame_core::offload::strategy::{Strategy, TwoSlot, ...}`
pub use adaptive::Adaptive;
pub use knapsack::Knapsack;
pub use two_slot::TwoSlot;

/// Maximum length of the access-pattern ring kept inside
/// [`OffloaderState`]. Longer histories give knapsack better
/// "recency × frequency" signal at negligible cost — the ring is a
/// `VecDeque<u32>` of `<= 64` entries.
pub const ACCESS_HISTORY_CAP: usize = 64;

/// Cheap snapshot of the offloader's state, passed to
/// [`Strategy::plan`] every time a plan is requested.
///
/// All fields are `Copy` or owned by reference into the strategy; the
/// strategy must not retain references across `plan()` calls.
#[derive(Debug, Clone)]
pub struct OffloaderState<'a> {
    /// Total number of blocks the offloader knows about.
    pub block_count: usize,

    /// Per-block bytes (BF16 footprint). Used by [`Knapsack`] and
    /// [`Adaptive`] to budget the resident set.
    ///
    /// `block_sizes[i] = 0` is legal and means "empty block / no data" —
    /// strategies treat zero-byte blocks as free to keep resident.
    pub block_sizes: &'a [usize],

    /// Block IDs currently sitting on a GPU slot. Length is 0, 1, or 2
    /// for the default 2-slot offloader; longer when a richer
    /// strategy-driven slot ring lands (Phase 3).
    pub resident: &'a [usize],

    /// The block ID the caller is asking the strategy to plan around
    /// (typically the next prefetch target).
    pub requested: usize,

    /// Most-recent-first access history. Bounded to
    /// [`ACCESS_HISTORY_CAP`]. The strategy uses this for recency /
    /// frequency scoring.
    pub access_history: &'a VecDeque<u32>,

    /// Free VRAM bytes at plan time. Populated when the offloader has
    /// enabled VRAM probing; `0` means the field is not authoritative
    /// (the strategy must use [`AccessHints::vram_authoritative`] to
    /// gate VRAM-pressure logic).
    pub free_vram_bytes: u64,

    /// Total VRAM bytes on the device. Same authoritativeness gate as
    /// [`Self::free_vram_bytes`].
    pub total_vram_bytes: u64,

    /// Auxiliary hints — telemetry counters, opt-in flags. Strategies
    /// may ignore.
    pub hints: AccessHints,
}

/// Auxiliary hints alongside [`OffloaderState`].
#[derive(Debug, Clone, Copy, Default)]
pub struct AccessHints {
    /// `true` when [`OffloaderState::free_vram_bytes`] /
    /// `total_vram_bytes` reflect a real cudaMemGetInfo call. `false`
    /// means the offloader chose not to probe (e.g. probe disabled, or
    /// last probe was too recent).
    pub vram_authoritative: bool,

    /// Number of `prefetch_block` calls observed since the last
    /// `plan()` — strategies use this to throttle expensive replanning.
    pub prefetches_since_last_plan: u32,
}

/// What the strategy thinks the offloader should do.
///
/// Strategies emit *advice*, not commands. The offloader inspects the
/// plan and applies what is safe (evictions of non-active slots) and
/// what fits its current double-buffer mechanic. Future Phase 3 work
/// may widen the offloader's slot ring to honor a strategy's
/// `desired_resident` exactly.
#[derive(Debug, Clone, Default)]
pub struct ResidentPlan {
    /// Block IDs the strategy says are safe / desirable to evict.
    /// Ordered by descending preference (front = first to evict).
    pub evict: Vec<usize>,

    /// Block IDs the strategy wants to keep resident.
    pub keep: Vec<usize>,

    /// Block IDs the strategy suggests prefetching next. Ordered by
    /// descending priority.
    pub prefetch: Vec<usize>,

    /// Strategy's target for total resident bytes. The offloader does
    /// not currently enforce this; it is reported into telemetry so
    /// adaptive strategies can be measured.
    pub target_resident_bytes: u64,
}

impl ResidentPlan {
    /// Total resident bytes the plan implies, given the offloader's
    /// `block_sizes`. Sums `keep` entries.
    pub fn keep_bytes(&self, block_sizes: &[usize]) -> u64 {
        self.keep
            .iter()
            .filter_map(|&b| block_sizes.get(b).copied())
            .map(|s| s as u64)
            .sum()
    }
}

/// Per-strategy planning trait.
///
/// `plan` is called by the offloader at strategic points (e.g. at the
/// top of `prefetch_block`). It MUST be cheap — microseconds at most.
/// Concretely:
///
/// * No CUDA launches.
/// * No allocations of more than a few hundred bytes; reuse internal
///   `Vec`s if possible.
/// * No I/O, no syscalls, no locking on contended global state.
///
/// `name()` is used in telemetry output so a tracing reader can tell
/// which strategy made a decision. Keep it short and stable.
pub trait Strategy: Send + Sync {
    /// Compute the resident plan for the current state.
    fn plan(&mut self, state: &OffloaderState) -> ResidentPlan;

    /// Stable short name (alphanumeric, no spaces) for telemetry.
    fn name(&self) -> &'static str;
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `keep_bytes` correctly sums against `block_sizes`.
    #[test]
    fn keep_bytes_sums_with_block_sizes() {
        let sizes = vec![100usize, 200, 300, 400];
        let plan = ResidentPlan {
            keep: vec![0, 2],
            ..Default::default()
        };
        assert_eq!(plan.keep_bytes(&sizes), 400);
    }

    /// `keep_bytes` survives out-of-bounds block IDs (planner can be
    /// stale relative to the offloader). Out-of-bounds entries
    /// contribute zero.
    #[test]
    fn keep_bytes_ignores_oob_blocks() {
        let sizes = vec![10usize, 20];
        let plan = ResidentPlan {
            keep: vec![0, 5, 1, 99],
            ..Default::default()
        };
        assert_eq!(plan.keep_bytes(&sizes), 30);
    }
}
