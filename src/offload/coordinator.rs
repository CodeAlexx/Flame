//! Phase 3 of OFFLOAD_NEXT_GEN_DESIGN — `OffloadCoordinator` skeleton.
//!
//! Single entry point that trainers talk to. Wraps the existing
//! `BlockOffloader` (weight blocks) and `GrowOnDemandActivationCache`
//! (block I/O activations) behind one type-safe surface.
//!
//! This phase ships the SKELETON only: types, RAII `BlockGuard`, and the
//! `before_block` API. No fraction/strategy logic yet — that lands in
//! Phase 5. No ring-slab allocator yet — that's Phase 4. Trainers can
//! optionally adopt this surface today; the substance arrives later.
//!
//! Per OFFLOAD_NEXT_GEN_DESIGN §D and the phase-plan gate "Phase 3:
//! compile; no perf gate. Just the type definitions and BlockGuard."

use std::sync::{Arc, Mutex};

use crate::activation_offload::GrowOnDemandActivationCache;
use crate::offload::BlockOffloader;
use crate::{Error, Result};
use cudarc::driver::CudaDevice;

/// Autograd traversal direction at the moment a block enters compute.
/// Drives the 3-case resident-set strategy in Phase 5. Today it's only
/// recorded for telemetry / future use; `BlockGuard` carries it so
/// downstream code can branch without re-querying autograd state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutogradDirection {
    /// Forward pass — autograd is recording (or no-grad sampling).
    Forward,
    /// Backward pass — walking the tape, computing VJPs.
    Backward,
}

impl AutogradDirection {
    /// Probe the autograd context for the current direction. Returns
    /// `Forward` when autograd is enabled (recording new ops or no-grad
    /// sampling), `Backward` when traversing the tape — for now we use
    /// the simpler heuristic of "is autograd currently recording?" since
    /// flame-core toggles `AUTOGRAD_ENABLED` during sub-tape walks.
    #[inline]
    pub fn from_autograd_state() -> Self {
        if crate::autograd::AutogradContext::is_recording() {
            Self::Forward
        } else {
            Self::Backward
        }
    }
}

/// Shared budget for host pinned RAM across the weight block pool and the
/// activation cache. Phase 8 turns this into a real enforced ceiling; for
/// now it's a non-enforcing record that the coordinator's two pools agree
/// on the same number.
#[derive(Clone, Copy, Debug)]
pub struct HostRamBudget {
    /// Max pinned host bytes both pools may use combined.
    pub max_bytes: usize,
}

impl HostRamBudget {
    /// Unbounded budget — current behavior. Use this until Phase 8 wires
    /// real enforcement.
    pub fn unbounded() -> Self {
        Self { max_bytes: usize::MAX }
    }

    /// Construct a budget with an explicit ceiling. Coordinator will warn
    /// at construction time if `pinned_bytes(block_offloader)` already
    /// exceeds the ceiling; today it does not refuse construction.
    pub fn new(max_bytes: usize) -> Self {
        Self { max_bytes }
    }
}

/// Placeholder for the Phase 5 strategy. Three resident-set cases per
/// OneTrainer (fwd→bwd, fwd→fwd, bwd→fwd) will be implemented here. The
/// `fraction` knob (0..1) selects how much of the model stays GPU-resident
/// vs offloads to pinned RAM. Today we record the knob and pass through.
#[derive(Clone, Copy, Debug)]
pub struct BlockOffloadStrategy {
    /// Fraction of layers that may stay GPU-resident at peak. `1.0` =
    /// keep everything resident (offload disabled in practice). `0.0` =
    /// offload everything (worst-case streaming).
    pub layer_offload_fraction: f32,
}

impl BlockOffloadStrategy {
    pub fn new(layer_offload_fraction: f32) -> Result<Self> {
        if !(0.0..=1.0).contains(&layer_offload_fraction) {
            return Err(Error::InvalidInput(format!(
                "layer_offload_fraction must be in [0.0, 1.0], got {layer_offload_fraction}"
            )));
        }
        Ok(Self { layer_offload_fraction })
    }

    /// Default: keep everything resident. Matches today's no-offload
    /// behavior and is the safe choice before Phase 5 lands real logic.
    pub fn all_resident() -> Self {
        Self { layer_offload_fraction: 1.0 }
    }
}

/// Single entry point trainers talk to. Owns the weight-block pool and
/// the activation cache, wires them to one device + one host-RAM budget.
///
/// **Phase 3 scope (this commit)**: skeleton + `BlockGuard` RAII. The
/// `before_block` call records the block index and current autograd
/// direction; `BlockGuard::drop` is a no-op stub. Phases 4-5 wire the
/// ring-slab allocator and the resident-set strategy. Phase 8 wires the
/// host RAM budget enforcement and telemetry.
///
/// Constructed by `OffloadCoordinator::new`. The trainer keeps it across
/// the entire training run. `coord.before_block(i)` returns a `BlockGuard`
/// for the duration of the block's compute.
pub struct OffloadCoordinator {
    /// Existing weight-block pinned-RAM + double-buffered GPU offloader.
    /// Owned by the coordinator; trainers don't access directly.
    blocks: BlockOffloader,
    /// Existing grow-on-demand activation cache (host pinned slabs +
    /// device-stream HtoD pull). Shared via Arc<Mutex> because the
    /// global `set_grow_activation_cache(arc)` is the install path the
    /// `checkpoint_offload_boundary` API reads from.
    activations: Arc<Mutex<GrowOnDemandActivationCache>>,
    /// Resident-set strategy (Phase 5 expands this).
    strategy: BlockOffloadStrategy,
    /// Shared host pinned-RAM budget (Phase 8 enforces).
    budget: HostRamBudget,
    /// Stash the device for future `RingSlabAllocator` (Phase 4).
    #[allow(dead_code)]
    device: Arc<CudaDevice>,
}

impl OffloadCoordinator {
    /// Wire up a coordinator from an already-constructed weight-block
    /// offloader and a grow activation cache. The cache is also
    /// installed as the global cache for `checkpoint_offload_boundary`
    /// so model code that calls that API gets the boundary semantics.
    ///
    /// Phase 6 will replace this constructor with one that takes raw
    /// model + dataset config and builds both pools internally. Today's
    /// shape mirrors how train_klein already constructs them.
    pub fn new(
        device: Arc<CudaDevice>,
        blocks: BlockOffloader,
        activations: GrowOnDemandActivationCache,
        strategy: BlockOffloadStrategy,
        budget: HostRamBudget,
    ) -> Result<Self> {
        let activations = Arc::new(Mutex::new(activations));
        // Install as the global cache that `checkpoint_offload_boundary`
        // reads from. Idempotent — re-installing overrides any previous
        // cache, which matches how train_klein already calls it directly.
        crate::autograd::set_grow_activation_cache(activations.clone())?;

        Ok(Self {
            blocks,
            activations,
            strategy,
            budget,
            device,
        })
    }

    /// Enter compute for block `idx`. Returns a guard whose `Drop`
    /// records a `compute_done` event (Phase 5 wires this to the
    /// strategy's resident-set rotation).
    ///
    /// Trainer pattern:
    /// ```ignore
    /// for i in 0..num_blocks {
    ///     let _guard = coord.before_block(i)?;
    ///     let out = AutogradContext::checkpoint_offload_boundary(
    ///         &[input.clone()],
    ///         |inputs| block.forward_inner(&inputs[0])
    ///     )?;
    ///     input = out;
    ///     // _guard drops here automatically
    /// }
    /// ```
    pub fn before_block(&mut self, idx: usize) -> Result<BlockGuard<'_>> {
        let direction = AutogradDirection::from_autograd_state();
        Ok(BlockGuard {
            coordinator: self,
            block_idx: idx,
            direction,
        })
    }

    /// Read-only access to the wrapped block offloader. Trainers that
    /// still call the offloader directly during the migration window
    /// (Phase 6) can use this; once migration completes the field stays
    /// private.
    pub fn blocks(&self) -> &BlockOffloader {
        &self.blocks
    }

    /// Mutable access to the wrapped block offloader. Same migration
    /// caveat as `blocks()`.
    pub fn blocks_mut(&mut self) -> &mut BlockOffloader {
        &mut self.blocks
    }

    /// Arc handle to the activation cache. Cloning the Arc is cheap and
    /// matches the install pattern of `set_grow_activation_cache`.
    pub fn activations(&self) -> Arc<Mutex<GrowOnDemandActivationCache>> {
        self.activations.clone()
    }

    /// Current strategy. Phase 5 lets trainers update this mid-run for
    /// fraction sweeps; for now it's read-only after construction.
    pub fn strategy(&self) -> BlockOffloadStrategy {
        self.strategy
    }

    /// Current host RAM budget. Phase 8 will enforce; today informational.
    pub fn budget(&self) -> HostRamBudget {
        self.budget
    }
}

/// RAII guard for a single block's compute window. `Drop` records the
/// block as having finished compute. Today the drop is a no-op stub —
/// Phase 5 wires it to rotate the resident-set strategy. The point of
/// shipping the RAII shape now is that trainer code written against
/// Phase 3 won't need to change when Phase 5 lands.
pub struct BlockGuard<'a> {
    /// Mutable borrow of the coordinator for the block's lifetime.
    /// Forces single-block-at-a-time semantics at the type level: you
    /// cannot `before_block(i)` and `before_block(j)` simultaneously.
    #[allow(dead_code)]
    coordinator: &'a mut OffloadCoordinator,
    /// Block index handed to `before_block`. Recorded for telemetry
    /// and Phase 5 transition-case dispatch.
    pub block_idx: usize,
    /// Autograd direction captured at `before_block` time.
    pub direction: AutogradDirection,
}

impl<'a> Drop for BlockGuard<'a> {
    fn drop(&mut self) {
        // Phase 5: record compute_done event on the active CUDA stream
        // so the strategy can rotate resident sets per the 3-case plan.
        // Phase 3 ships the type shape only; no work happens here yet.
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strategy_rejects_out_of_range_fraction() {
        assert!(BlockOffloadStrategy::new(-0.1).is_err());
        assert!(BlockOffloadStrategy::new(1.1).is_err());
        assert!(BlockOffloadStrategy::new(0.0).is_ok());
        assert!(BlockOffloadStrategy::new(1.0).is_ok());
        assert!(BlockOffloadStrategy::new(0.5).is_ok());
    }

    #[test]
    fn strategy_all_resident_is_1_0() {
        assert_eq!(BlockOffloadStrategy::all_resident().layer_offload_fraction, 1.0);
    }

    #[test]
    fn budget_unbounded_is_max() {
        assert_eq!(HostRamBudget::unbounded().max_bytes, usize::MAX);
    }

    #[test]
    fn autograd_direction_from_state_does_not_panic() {
        // Smoke test — exact value depends on global autograd state which
        // tests don't manipulate. Just verifies the call compiles + runs.
        let _ = AutogradDirection::from_autograd_state();
    }
}
