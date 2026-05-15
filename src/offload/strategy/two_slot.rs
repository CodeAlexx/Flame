//! `TwoSlot` — the current 2-slot ping-pong policy as a [`Strategy`] impl.
//!
//! This is the default. The offloader's pre-Phase-2 hardcoded path keeps
//! exactly two slots: one "active" (compute) and one "prefetch target"
//! (the other slot). Reformulating that policy as a [`Strategy`] lets us
//! prove the reformulation matches the hardcoded path bit-for-bit (the
//! regression gate in `tests/offload_strategy_smoke.rs::two_slot_matches_hardcoded`).
//!
//! Behavior:
//!
//! * `keep` always contains the resident block IDs (whatever the
//!   offloader currently has on slots).
//! * `evict` is the non-requested resident block ID — i.e. the slot we
//!   are about to overwrite. Empty when both slots already hold the
//!   requested block (no-op prefetch).
//! * `prefetch` contains just `[requested]` when not already resident,
//!   else empty.
//! * `target_resident_bytes` is the sum of the two largest blocks
//!   (capacity bound) — diagnostic only.

use super::{OffloaderState, ResidentPlan, Strategy};

/// 2-slot ping-pong policy. Stateless: every call to `plan()` is a
/// pure function of `state`.
#[derive(Debug, Default, Clone, Copy)]
pub struct TwoSlot;

impl TwoSlot {
    pub fn new() -> Self {
        Self
    }
}

impl Strategy for TwoSlot {
    fn plan(&mut self, state: &OffloaderState) -> ResidentPlan {
        let requested = state.requested;
        let resident = state.resident;

        // Already on a slot? → no-op (no evict, no prefetch).
        if resident.contains(&requested) {
            return ResidentPlan {
                evict: Vec::new(),
                keep: resident.to_vec(),
                prefetch: Vec::new(),
                target_resident_bytes: capacity_bound_bytes(state.block_sizes),
            };
        }

        // Pick the eviction target: the non-active resident slot. For
        // the 2-slot offloader this is the unique resident block that
        // is *not* the most recently accessed one. We fall back to
        // "the first resident" when history is empty.
        let last_accessed = state.access_history.front().map(|&b| b as usize);
        let mut evict: Vec<usize> = Vec::new();
        for &b in resident {
            if Some(b) != last_accessed {
                evict.push(b);
            }
        }
        // If both resident blocks are the last-accessed (e.g. duplicate)
        // or no history, evict the first resident slot regardless.
        if evict.is_empty() && !resident.is_empty() {
            evict.push(resident[0]);
        }

        ResidentPlan {
            evict,
            keep: resident.to_vec(),
            prefetch: vec![requested],
            target_resident_bytes: capacity_bound_bytes(state.block_sizes),
        }
    }

    fn name(&self) -> &'static str {
        "TwoSlot"
    }
}

/// Sum of the two largest block sizes — the static capacity bound the
/// 2-slot offloader works to. Diagnostic / telemetry only.
fn capacity_bound_bytes(block_sizes: &[usize]) -> u64 {
    if block_sizes.is_empty() {
        return 0;
    }
    let mut sorted: Vec<usize> = block_sizes.to_vec();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    let top2: usize = sorted.iter().take(2).sum();
    top2 as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    fn state<'a>(
        resident: &'a [usize],
        requested: usize,
        history: &'a VecDeque<u32>,
        sizes: &'a [usize],
    ) -> OffloaderState<'a> {
        OffloaderState {
            block_count: sizes.len(),
            block_sizes: sizes,
            resident,
            requested,
            access_history: history,
            free_vram_bytes: 0,
            total_vram_bytes: 0,
            hints: Default::default(),
        }
    }

    #[test]
    fn already_resident_is_noop() {
        let sizes = vec![10usize, 20, 30];
        let resident = vec![1, 2];
        let history: VecDeque<u32> = VecDeque::from(vec![2, 1]);
        let mut s = TwoSlot::new();
        let plan = s.plan(&state(&resident, 1, &history, &sizes));
        assert!(plan.evict.is_empty(), "no eviction when block is resident");
        assert!(plan.prefetch.is_empty(), "no prefetch when resident");
        assert_eq!(plan.keep, vec![1, 2]);
    }

    #[test]
    fn evicts_non_last_accessed() {
        let sizes = vec![10usize, 20, 30, 40];
        // Slot order [1, 2] with 1 last accessed → evict 2.
        let resident = vec![1, 2];
        let history: VecDeque<u32> = VecDeque::from(vec![1, 2]);
        let mut s = TwoSlot::new();
        let plan = s.plan(&state(&resident, 3, &history, &sizes));
        assert_eq!(plan.evict, vec![2], "should evict the older resident block");
        assert_eq!(plan.prefetch, vec![3]);
    }

    #[test]
    fn evicts_first_resident_when_no_history() {
        let sizes = vec![10usize, 20, 30];
        let resident = vec![1, 2];
        let history: VecDeque<u32> = VecDeque::new();
        let mut s = TwoSlot::new();
        let plan = s.plan(&state(&resident, 0, &history, &sizes));
        assert_eq!(plan.evict, vec![1, 2], "evict any resident without history");
        assert_eq!(plan.prefetch, vec![0]);
    }

    #[test]
    fn name_is_stable() {
        let mut s = TwoSlot::new();
        assert_eq!(s.name(), "TwoSlot");
        let history: VecDeque<u32> = VecDeque::new();
        let _ = s.plan(&state(&[], 0, &history, &[]));
        assert_eq!(s.name(), "TwoSlot", "name must not change after plan()");
    }
}
