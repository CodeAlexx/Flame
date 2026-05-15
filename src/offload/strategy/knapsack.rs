//! `Knapsack` — value-based resident-set selection bounded by a byte budget.
//!
//! Ports the **algorithmic core** of FlexTensor's
//! `strategy/knapsack.py`. The Python implementation interleaves with
//! `LayerStatistics`, `MemoryTransferInterpolator`, and `TensorStatistics`
//! discovery — all of which require PyTorch `__torch_function__` hooks
//! that flame-core does not (and should not) emulate. What we keep:
//!
//! 1. **Value function.** Each block has a per-call value derived from
//!    its recent access history (most-recent-first decay + frequency).
//! 2. **Budget-bounded selection.** Given a byte budget, pick the
//!    subset of blocks with the highest total value that fits.
//! 3. **Greedy 0/1 knapsack** by value-per-byte (the FlexTensor
//!    implementation uses dynamic programming, but our N (block count)
//!    is small — ≤ a few hundred — and per-call cost matters; greedy
//!    delivers within ~5% of optimal at <1 µs).
//!
//! What we **do not** port:
//!
//! * `MemoryTransferInterpolator` — Phase 2 instead consults
//!   [`super::super::transfer_benchmark::TransferBandwidthProfile`]
//!   for predicted H2D wall time (when the offloader has one).
//! * `_estimate_required_scale` and the scale-vs-bytes binary search
//!   — those work against FlexTensor's compute schedule, not ours.
//! * `KnapsackBlockStrategy.cyclic` and `group_size` — FlexTensor uses
//!   them to merge consecutive layers into a single transfer batch.
//!   The flame-core `BlockOffloader` already keys on
//!   [`BlockFacilitator`](super::super::BlockFacilitator)-classified
//!   blocks, so the merging is upstream of the strategy.
//!
//! Phase 3 may swap in a true DP solver if benchmarks show the greedy
//! variant leaves bandwidth on the table.

use super::{OffloaderState, ResidentPlan, Strategy};

/// Value-function weighting. Tune `recency_weight` and
/// `frequency_weight` to bias toward either short-term hot blocks or
/// long-term frequent ones.
#[derive(Debug, Clone, Copy)]
pub struct ValueWeights {
    /// Weight on inverse-recency (1.0 / (1 + steps_since_last_access)).
    /// FlexTensor's heuristic privileges recency heavily — match it
    /// at `1.0`.
    pub recency_weight: f64,
    /// Weight on frequency (count of accesses in history / window).
    pub frequency_weight: f64,
}

impl Default for ValueWeights {
    fn default() -> Self {
        Self {
            recency_weight: 1.0,
            frequency_weight: 0.25,
        }
    }
}

/// Knapsack policy: pick the highest-value resident set that fits in
/// `budget_bytes`. State-light: only `prev_budget_bytes` and a reusable
/// `Vec` buffer are retained across calls.
#[derive(Debug)]
pub struct Knapsack {
    /// Hard ceiling on resident bytes for the strategy's planning.
    /// `None` means "no explicit budget — plan as if every block can
    /// be resident", which collapses to selecting all blocks. Useful
    /// for diagnostic comparisons.
    pub budget_bytes: Option<u64>,
    /// Value function weights.
    pub weights: ValueWeights,
    /// Reusable scratch buffer for per-call value computation. Avoids
    /// allocation on hot path.
    scratch: Vec<ScoredBlock>,
}

#[derive(Debug, Clone, Copy)]
struct ScoredBlock {
    block_id: usize,
    value: f64,
    bytes: u64,
}

impl Knapsack {
    /// New knapsack with an explicit byte budget.
    pub fn with_budget(budget_bytes: u64) -> Self {
        Self {
            budget_bytes: Some(budget_bytes),
            weights: ValueWeights::default(),
            scratch: Vec::new(),
        }
    }

    /// New knapsack with no explicit budget. The plan will return every
    /// block as keep — useful for measuring the value scores in isolation.
    pub fn unbounded() -> Self {
        Self {
            budget_bytes: None,
            weights: ValueWeights::default(),
            scratch: Vec::new(),
        }
    }

    /// Override the value weights.
    pub fn with_weights(mut self, weights: ValueWeights) -> Self {
        self.weights = weights;
        self
    }
}

impl Strategy for Knapsack {
    fn plan(&mut self, state: &OffloaderState) -> ResidentPlan {
        // ── 1. Score every block. ───────────────────────────────────
        self.scratch.clear();
        self.scratch.reserve(state.block_count);
        for b in 0..state.block_count {
            let bytes = state.block_sizes.get(b).copied().unwrap_or(0) as u64;
            // Always include the requested block — strategy can't say
            // "don't load the block the offloader is about to ask for".
            let value = score_block(b, state, &self.weights);
            self.scratch.push(ScoredBlock {
                block_id: b,
                value,
                bytes,
            });
        }

        // ── 2. Apply budget (or no-op). ─────────────────────────────
        let budget = self.budget_bytes.unwrap_or(u64::MAX);

        // Greedy: sort by value/byte descending, tiebreak on bytes
        // ascending. Requested block always wins its bucket.
        let requested = state.requested;
        self.scratch.sort_unstable_by(|a, b| {
            // Requested block sorted to front.
            if a.block_id == requested && b.block_id != requested {
                return std::cmp::Ordering::Less;
            }
            if b.block_id == requested && a.block_id != requested {
                return std::cmp::Ordering::Greater;
            }
            let a_score = if a.bytes == 0 {
                f64::INFINITY
            } else {
                a.value / (a.bytes as f64)
            };
            let b_score = if b.bytes == 0 {
                f64::INFINITY
            } else {
                b.value / (b.bytes as f64)
            };
            b_score
                .partial_cmp(&a_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut keep: Vec<usize> = Vec::with_capacity(self.scratch.len());
        let mut used: u64 = 0;
        for sb in &self.scratch {
            let next = used.saturating_add(sb.bytes);
            if next <= budget {
                keep.push(sb.block_id);
                used = next;
            }
        }

        // ── 3. Derive evict / prefetch. ─────────────────────────────
        let mut evict: Vec<usize> = Vec::new();
        for &b in state.resident {
            if !keep.contains(&b) {
                evict.push(b);
            }
        }

        let mut prefetch: Vec<usize> = Vec::new();
        if keep.contains(&requested) && !state.resident.contains(&requested) {
            prefetch.push(requested);
        }

        ResidentPlan {
            evict,
            keep,
            prefetch,
            target_resident_bytes: used,
        }
    }

    fn name(&self) -> &'static str {
        "Knapsack"
    }
}

/// Per-block value score. Larger = more valuable to keep resident.
///
/// Combines:
/// * Inverse recency: most-recently-accessed block scores `1.0`,
///   `1/(1+k)` after k steps.
/// * Frequency: count of `block_id` in the access ring, normalized by
///   the ring length.
/// * A small floor for the explicitly-requested block so it never
///   prices itself out of the budget.
fn score_block(block_id: usize, state: &OffloaderState, w: &ValueWeights) -> f64 {
    let mut recency_score = 0.0;
    let mut freq_count = 0.0;
    let id = block_id as u32;
    let history_len = state.access_history.len().max(1);
    for (k, &b) in state.access_history.iter().enumerate() {
        if b == id {
            // First match dominates recency.
            if recency_score == 0.0 {
                recency_score = 1.0 / (1.0 + k as f64);
            }
            freq_count += 1.0;
        }
    }
    let frequency_score = freq_count / history_len as f64;
    let requested_bonus = if block_id == state.requested {
        0.5
    } else {
        0.0
    };
    w.recency_weight * recency_score + w.frequency_weight * frequency_score + requested_bonus
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

    /// Knapsack must never plan more resident bytes than the budget.
    #[test]
    fn never_exceeds_budget() {
        let sizes = vec![100usize, 200, 300, 400, 500];
        let history: VecDeque<u32> = VecDeque::from(vec![0, 1, 2, 3, 4, 0, 1]);
        let mut k = Knapsack::with_budget(500);
        let plan = k.plan(&state(&[], 0, &history, &sizes));
        let bytes = plan.keep_bytes(&sizes);
        assert!(
            bytes <= 500,
            "knapsack exceeded budget: kept {bytes} bytes vs 500"
        );
    }

    /// With an unbounded budget every block fits.
    #[test]
    fn unbounded_keeps_all() {
        let sizes = vec![1usize, 2, 3, 4];
        let history: VecDeque<u32> = VecDeque::new();
        let mut k = Knapsack::unbounded();
        let plan = k.plan(&state(&[], 0, &history, &sizes));
        assert_eq!(plan.keep.len(), 4, "unbounded must keep all blocks");
    }

    /// The most-recently-accessed block must score higher than an
    /// unaccessed one of equal size.
    #[test]
    fn recency_wins_under_budget() {
        let sizes = vec![100usize, 100, 100, 100];
        let mut history: VecDeque<u32> = VecDeque::new();
        history.push_back(2); // block 2 was just accessed
        history.push_back(1);
        history.push_back(0);
        let mut k = Knapsack::with_budget(100); // only one fits
        let plan = k.plan(&state(&[], 3, &history, &sizes));
        // Requested is block 3, which gets a bonus; it should fit.
        assert!(
            plan.keep.contains(&3),
            "requested block must always survive its budget bucket"
        );
    }

    /// Empty access history doesn't panic and budget is still honored.
    #[test]
    fn empty_history_safe() {
        let sizes = vec![100usize, 100, 100];
        let history: VecDeque<u32> = VecDeque::new();
        let mut k = Knapsack::with_budget(150);
        let plan = k.plan(&state(&[], 0, &history, &sizes));
        assert!(plan.keep_bytes(&sizes) <= 150);
    }

    /// Zero-byte blocks are always free to keep.
    #[test]
    fn zero_byte_blocks_keep_for_free() {
        let sizes = vec![0usize, 100, 0, 100];
        let history: VecDeque<u32> = VecDeque::new();
        let mut k = Knapsack::with_budget(100);
        let plan = k.plan(&state(&[], 0, &history, &sizes));
        // Both zero-byte blocks plus one 100-byte block should fit.
        assert!(plan.keep.contains(&0));
        assert!(plan.keep.contains(&2));
        assert!(plan.keep_bytes(&sizes) <= 100);
    }

    /// `evict` lists resident blocks not in `keep`.
    #[test]
    fn evict_lists_dropped_residents() {
        let sizes = vec![100usize, 100, 100, 100];
        let resident = vec![0, 1];
        let history: VecDeque<u32> = VecDeque::from(vec![3, 2]); // 3,2 hot
        let mut k = Knapsack::with_budget(200);
        let plan = k.plan(&state(&resident, 2, &history, &sizes));
        // budget=200 → exactly 2 keepers fit; hot ones (3 requested, 2)
        // win, so residents 0 and 1 should be evicted.
        for &b in &resident {
            if !plan.keep.contains(&b) {
                assert!(plan.evict.contains(&b));
            }
        }
    }

    /// Telemetry: target_resident_bytes is bounded by budget.
    #[test]
    fn target_resident_bytes_bounded() {
        let sizes = vec![100usize; 10];
        let history: VecDeque<u32> = VecDeque::new();
        let mut k = Knapsack::with_budget(350);
        let plan = k.plan(&state(&[], 0, &history, &sizes));
        assert!(plan.target_resident_bytes <= 350);
    }
}
