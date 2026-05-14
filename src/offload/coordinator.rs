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

/// Which transition the trainer is in. Drives the resident-set precompute
/// per OneTrainer's 3-case model:
///
/// - **ForwardBackward**: forward pass that will be followed by a
///   backward — don't offload the LAST layers because they're needed
///   first in backward (LIFO consumption).
/// - **ForwardForward**: forward pass followed by another forward (next
///   step or microbatch) — cyclic; start loading first layers while
///   executing last ones.
/// - **BackwardForward**: backward pass — mirror of ForwardBackward,
///   walking in reverse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransitionCase {
    ForwardBackward,
    ForwardForward,
    BackwardForward,
}

/// Phase 5 strategy — fraction knob + 3-case resident-set precompute.
///
/// The trainer registers per-layer GPU-resident byte sizes once via
/// [`BlockOffloadStrategy::with_layer_bytes`]. The strategy precomputes
/// resident-layer indices for each (block_idx, transition_case) cell so
/// the per-block hot path is a single Vec lookup.
///
/// Ported from
/// `OneTrainer/modules/util/LayerOffloadConductor.py::LayerOffloadStrategy`.
#[derive(Clone, Debug)]
pub struct BlockOffloadStrategy {
    /// Fraction of layers (by bytes) to keep offloaded at peak.
    /// `0.0` = everything resident (offload off in practice).
    /// `1.0` = offload everything (extreme streaming).
    /// OneTrainer convention: `target_loaded_bytes = total * (1 - fraction)`.
    pub layer_offload_fraction: f32,
    /// Per-block GPU-resident byte sizes. Empty until
    /// [`with_layer_bytes`] is called.
    layer_bytes: Vec<usize>,
    /// `forward_backward[i]` = layer indices resident before executing
    /// layer `i` when the next thing after the forward pass is a
    /// backward pass. Precomputed at `with_layer_bytes` time.
    forward_backward: Vec<Vec<usize>>,
    /// `forward_forward[i]` = same but for the cyclic case (forward
    /// followed by another forward).
    forward_forward: Vec<Vec<usize>>,
    /// `backward_forward[i]` = same but for the backward direction.
    backward_forward: Vec<Vec<usize>>,
}

impl BlockOffloadStrategy {
    pub fn new(layer_offload_fraction: f32) -> Result<Self> {
        if !(0.0..=1.0).contains(&layer_offload_fraction) {
            return Err(Error::InvalidInput(format!(
                "layer_offload_fraction must be in [0.0, 1.0], got {layer_offload_fraction}"
            )));
        }
        Ok(Self {
            layer_offload_fraction,
            layer_bytes: Vec::new(),
            forward_backward: Vec::new(),
            forward_forward: Vec::new(),
            backward_forward: Vec::new(),
        })
    }

    /// Default: keep everything resident. Matches today's no-offload
    /// behavior and is the safe choice before the trainer registers
    /// real layer-byte data.
    pub fn all_resident() -> Self {
        Self {
            layer_offload_fraction: 0.0,
            layer_bytes: Vec::new(),
            forward_backward: Vec::new(),
            forward_forward: Vec::new(),
            backward_forward: Vec::new(),
        }
    }

    /// Register per-block byte sizes. Precomputes the resident-set
    /// tables for all 3 transition cases. Idempotent — recalling with
    /// the same data is fine, just wastes a few ms.
    pub fn with_layer_bytes(mut self, layer_bytes: Vec<usize>) -> Self {
        let total_bytes: usize = layer_bytes.iter().sum();
        let target_loaded_bytes =
            (total_bytes as f32 * (1.0 - self.layer_offload_fraction)) as usize;
        let n = layer_bytes.len();

        self.forward_backward = (0..n)
            .map(|i| Self::layers_below(&layer_bytes, i, target_loaded_bytes, true, false))
            .collect();
        self.forward_forward = (0..n)
            .map(|i| Self::layers_below(&layer_bytes, i, target_loaded_bytes, true, true))
            .collect();
        self.backward_forward = (0..n)
            .map(|i| Self::layers_below(&layer_bytes, i, target_loaded_bytes, false, false))
            .collect();

        self.layer_bytes = layer_bytes;
        self
    }

    /// Total bytes registered via `with_layer_bytes` (`0` if not registered).
    pub fn total_bytes(&self) -> usize {
        self.layer_bytes.iter().sum()
    }

    /// Target GPU-resident bytes (`total * (1 - fraction)`).
    pub fn target_loaded_bytes(&self) -> usize {
        let total = self.total_bytes() as f32;
        (total * (1.0 - self.layer_offload_fraction)) as usize
    }

    /// Resident-layer indices before executing `block_idx` in the
    /// given transition case. Returns empty vec if `with_layer_bytes`
    /// hasn't been called yet (interpret as "everything resident").
    pub fn resident_layers(&self, block_idx: usize, case: TransitionCase) -> &[usize] {
        let table = match case {
            TransitionCase::ForwardBackward => &self.forward_backward,
            TransitionCase::ForwardForward => &self.forward_forward,
            TransitionCase::BackwardForward => &self.backward_forward,
        };
        table.get(block_idx).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Max GPU-resident bytes observed across all (block, case) cells —
    /// the strategy's peak memory budget guarantee. `0` if not registered.
    pub fn max_loaded_bytes(&self) -> usize {
        if self.layer_bytes.is_empty() {
            return 0;
        }
        let mut max = 0usize;
        for table in [&self.forward_backward, &self.forward_forward, &self.backward_forward] {
            for resident in table {
                let sum: usize = resident.iter().map(|&i| self.layer_bytes[i]).sum();
                if sum > max {
                    max = sum;
                }
            }
        }
        max
    }

    /// OneTrainer's `__get_layers_below`. Returns the list of layer
    /// indices that, summed by bytes, just exceed `max_bytes` (with a
    /// minimum of 2 layers always loaded). `is_forward` controls walk
    /// direction; `is_cyclic` controls whether the walk wraps at the
    /// model's end.
    fn layers_below(
        layer_bytes: &[usize],
        start_layer: usize,
        max_bytes: usize,
        is_forward: bool,
        is_cyclic: bool,
    ) -> Vec<usize> {
        let n = layer_bytes.len();
        let mut accumulator = 0usize;
        let mut layers = Vec::new();

        let mut push = |i: usize, acc: &mut usize, lyrs: &mut Vec<usize>| -> bool {
            *acc += layer_bytes[i];
            if *acc > max_bytes && lyrs.len() >= 2 {
                return true; // stop
            }
            lyrs.push(i);
            false
        };

        if is_forward && is_cyclic {
            for i in start_layer..n {
                if push(i, &mut accumulator, &mut layers) { return layers; }
            }
            for i in 0..start_layer {
                if push(i, &mut accumulator, &mut layers) { return layers; }
            }
        } else if is_forward && !is_cyclic {
            for i in start_layer..n {
                if push(i, &mut accumulator, &mut layers) { return layers; }
            }
            for i in (0..start_layer).rev() {
                if push(i, &mut accumulator, &mut layers) { return layers; }
            }
        } else {
            // backward
            for i in (0..=start_layer).rev() {
                if push(i, &mut accumulator, &mut layers) { return layers; }
            }
            for i in (start_layer + 1)..n {
                if push(i, &mut accumulator, &mut layers) { return layers; }
            }
        }

        layers
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
    /// Owned by the coordinator; trainers don't access directly. `None`
    /// when the trainer creates its own BlockOffloader and the
    /// coordinator manages only the activation cache — a temporary state
    /// during the Phase 6/7 migration. Phase 7b moves block ownership
    /// into the coordinator across all trainers.
    blocks: Option<BlockOffloader>,
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
    pub fn new(
        device: Arc<CudaDevice>,
        blocks: BlockOffloader,
        activations: GrowOnDemandActivationCache,
        strategy: BlockOffloadStrategy,
        budget: HostRamBudget,
    ) -> Result<Self> {
        let activations = Arc::new(Mutex::new(activations));
        crate::autograd::set_grow_activation_cache(activations.clone())?;

        Ok(Self {
            blocks: Some(blocks),
            activations,
            strategy,
            budget,
            device,
        })
    }

    /// Construct with only the activation cache. Used by trainers whose
    /// model already owns its own `BlockOffloader` (Klein today) — the
    /// coordinator manages activation offload while the model continues
    /// to drive its own block streaming. Phase 7b will collapse this
    /// path into [`Self::new`] once trainers hand block ownership over.
    pub fn with_activation_cache_only(
        device: Arc<CudaDevice>,
        activations: GrowOnDemandActivationCache,
        strategy: BlockOffloadStrategy,
        budget: HostRamBudget,
    ) -> Result<Self> {
        let activations = Arc::new(Mutex::new(activations));
        crate::autograd::set_grow_activation_cache(activations.clone())?;

        Ok(Self {
            blocks: None,
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

    /// Read-only access to the wrapped block offloader, if the
    /// coordinator owns one. Returns `None` when constructed via
    /// [`Self::with_activation_cache_only`].
    pub fn blocks(&self) -> Option<&BlockOffloader> {
        self.blocks.as_ref()
    }

    /// Mutable access to the wrapped block offloader.
    pub fn blocks_mut(&mut self) -> Option<&mut BlockOffloader> {
        self.blocks.as_mut()
    }

    /// Arc handle to the activation cache. Cloning the Arc is cheap and
    /// matches the install pattern of `set_grow_activation_cache`.
    pub fn activations(&self) -> Arc<Mutex<GrowOnDemandActivationCache>> {
        self.activations.clone()
    }

    /// Current strategy. Phase 5 lets trainers update this mid-run for
    /// fraction sweeps; for now it's read-only after construction.
    pub fn strategy(&self) -> &BlockOffloadStrategy {
        &self.strategy
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
    fn strategy_all_resident_offloads_nothing() {
        // `all_resident` = fraction 0 = target_loaded = total.
        let s = BlockOffloadStrategy::all_resident()
            .with_layer_bytes(vec![100, 100, 100, 100]);
        assert_eq!(s.target_loaded_bytes(), 400);
        // Every block-case cell contains all 4 layers.
        for i in 0..4 {
            for case in [
                TransitionCase::ForwardBackward,
                TransitionCase::ForwardForward,
                TransitionCase::BackwardForward,
            ] {
                let r = s.resident_layers(i, case);
                assert_eq!(r.len(), 4, "block {} case {:?}", i, case);
            }
        }
    }

    #[test]
    fn strategy_full_offload_keeps_min_two_layers() {
        // fraction 1.0 → target_loaded = 0; algorithm guarantees >= 2 resident.
        let s = BlockOffloadStrategy::new(1.0)
            .unwrap()
            .with_layer_bytes(vec![100, 100, 100, 100]);
        for i in 0..4 {
            for case in [
                TransitionCase::ForwardBackward,
                TransitionCase::ForwardForward,
                TransitionCase::BackwardForward,
            ] {
                let r = s.resident_layers(i, case);
                assert!(r.len() >= 2, "block {} case {:?}", i, case);
            }
        }
    }

    #[test]
    fn strategy_resident_starts_at_block_index() {
        // For forward at block 0, resident should start at 0.
        let s = BlockOffloadStrategy::new(0.5)
            .unwrap()
            .with_layer_bytes(vec![100; 10]);
        let r = s.resident_layers(0, TransitionCase::ForwardBackward);
        assert_eq!(r[0], 0);
        // For backward at block 9, resident should start at 9.
        let r = s.resident_layers(9, TransitionCase::BackwardForward);
        assert_eq!(r[0], 9);
    }

    #[test]
    fn strategy_max_loaded_bytes_bounds_under_target_plus_one_layer() {
        // The algorithm exceeds max_bytes by AT MOST one layer's worth
        // (the one that pushed over the limit; min-2-resident exception
        // applies otherwise). Verify the bound.
        let s = BlockOffloadStrategy::new(0.5)
            .unwrap()
            .with_layer_bytes(vec![100; 10]);
        let target = s.target_loaded_bytes(); // 500
        let max_layer = *s.layer_bytes.iter().max().unwrap(); // 100
        // max_loaded_bytes >= target (we always include the boundary layer)
        // and <= target + max_layer (algorithm doesn't double-overshoot).
        // Account for the min-2 floor which can sit above target when
        // target is small. Here target=500, fits 5 layers, plenty of room.
        assert!(s.max_loaded_bytes() >= target);
        assert!(s.max_loaded_bytes() <= target + max_layer);
    }

    #[test]
    fn strategy_resident_layers_without_register_returns_empty() {
        // Until with_layer_bytes is called, every cell is empty.
        let s = BlockOffloadStrategy::new(0.5).unwrap();
        assert_eq!(s.resident_layers(0, TransitionCase::ForwardBackward).len(), 0);
        assert_eq!(s.resident_layers(99, TransitionCase::BackwardForward).len(), 0);
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
