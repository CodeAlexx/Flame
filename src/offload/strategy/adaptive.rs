//! `Adaptive` — VRAM-pressure-driven resident-set sizing.
//!
//! Ports the **observe-and-adapt** spirit of FlexTensor's
//! `strategy/adaptive.py`. The Python implementation tries every
//! candidate strategy and picks the one that scored best against a
//! cost model — it pre-supposes a corpus of measurements and
//! `evaluate_strategy_result`. flame-core's `BlockOffloader` doesn't
//! have that corpus, so `Adaptive` here is simpler and aligned with
//! the user's strategic framing (Phase 2):
//!
//! **Watch VRAM pressure; shrink the resident-set when pressure is
//! high; grow it when pressure releases.**
//!
//! The decision flow at each `plan()` is:
//!
//! 1. Read `free_vram_bytes` from [`OffloaderState`] (the offloader
//!    fills this from `cuda_mem_get_info`; non-authoritative reads
//!    are honored conservatively).
//! 2. Compute "pressure" = `1 - free/total`. Higher = tighter.
//! 3. If pressure > `high_watermark`, target ~50% of the unconstrained
//!    resident-set bytes. Hand the (smaller) budget to an internal
//!    [`super::Knapsack`].
//! 4. If pressure < `low_watermark`, target the unconstrained set
//!    (no budget). Resident set grows up to whatever fits.
//! 5. Otherwise (between watermarks) hold the previous step's target
//!    — hysteresis to avoid thrashing under jitter.
//!
//! On platforms where `vram_authoritative == false`, `Adaptive`
//! degrades to "pretend pressure is medium" and keeps the previous
//! target. That makes the strategy safe on CI / headless setups where
//! the offloader chose not to call the driver.

use super::knapsack::{Knapsack, ValueWeights};
use super::{OffloaderState, ResidentPlan, Strategy};

/// VRAM-pressure-driven adaptive strategy.
#[derive(Debug)]
pub struct Adaptive {
    /// Pressure above this fraction → shrink.
    pub high_watermark: f64,
    /// Pressure below this fraction → grow.
    pub low_watermark: f64,
    /// Shrink target as a fraction of total block bytes (sum over all
    /// blocks). `0.5` = "keep at most half of all block bytes resident".
    pub shrink_fraction: f64,
    /// Internal knapsack used to actually pick blocks given the
    /// derived budget.
    inner: Knapsack,
    /// Most recent target_bytes the strategy committed to. Persisted
    /// so the medium-pressure band can hold steady.
    last_target_bytes: u64,
    /// Most recent pressure observation (for telemetry / debugging).
    /// `f64::NAN` until the first authoritative read.
    last_pressure: f64,
}

impl Adaptive {
    /// New adaptive with default watermarks: shrink above 85% used,
    /// grow below 60% used.
    pub fn new() -> Self {
        Self {
            high_watermark: 0.85,
            low_watermark: 0.60,
            shrink_fraction: 0.50,
            inner: Knapsack::unbounded(),
            last_target_bytes: u64::MAX,
            last_pressure: f64::NAN,
        }
    }

    /// Override the watermarks. `high_watermark` must be ≥ `low_watermark`.
    pub fn with_watermarks(mut self, low: f64, high: f64) -> Self {
        assert!(
            high >= low,
            "Adaptive::with_watermarks: high ({high}) must be >= low ({low})"
        );
        self.low_watermark = low;
        self.high_watermark = high;
        self
    }

    /// Override the shrink fraction (default 0.5).
    pub fn with_shrink_fraction(mut self, fraction: f64) -> Self {
        self.shrink_fraction = fraction;
        self
    }

    /// Override knapsack value weights used internally.
    pub fn with_value_weights(mut self, weights: ValueWeights) -> Self {
        self.inner = Knapsack::unbounded().with_weights(weights);
        self
    }

    /// Last observed pressure, in `[0,1]`. NaN if no authoritative
    /// VRAM read has happened. Exposed for telemetry; not part of the
    /// trait surface.
    pub fn last_observed_pressure(&self) -> f64 {
        self.last_pressure
    }

    /// Last target bytes the strategy committed to. Useful for tests
    /// that simulate pressure swings.
    pub fn last_target_bytes(&self) -> u64 {
        self.last_target_bytes
    }
}

impl Default for Adaptive {
    fn default() -> Self {
        Self::new()
    }
}

impl Strategy for Adaptive {
    fn plan(&mut self, state: &OffloaderState) -> ResidentPlan {
        let total_block_bytes: u64 = state.block_sizes.iter().map(|&s| s as u64).sum::<u64>();

        // Decide a target budget.
        let target = if state.hints.vram_authoritative && state.total_vram_bytes > 0 {
            let used = state.total_vram_bytes.saturating_sub(state.free_vram_bytes);
            let pressure = used as f64 / state.total_vram_bytes as f64;
            self.last_pressure = pressure;

            if pressure >= self.high_watermark {
                // Shrink.
                let shrunk = (total_block_bytes as f64 * self.shrink_fraction) as u64;
                self.last_target_bytes = shrunk;
                shrunk
            } else if pressure <= self.low_watermark {
                // Grow.
                self.last_target_bytes = u64::MAX;
                u64::MAX
            } else {
                // Hysteresis: hold the previous target.
                self.last_target_bytes
            }
        } else {
            // Non-authoritative VRAM read — hold last target. On the
            // very first call (no last target), use unbounded so we
            // don't accidentally evict useful blocks based on bad data.
            self.last_target_bytes
        };

        // Drive the inner knapsack with the computed budget.
        if target == u64::MAX {
            self.inner.budget_bytes = None;
        } else {
            self.inner.budget_bytes = Some(target);
        }
        let mut plan = self.inner.plan(state);
        // Force the telemetry target to reflect *our* target, not the
        // inner knapsack's "kept bytes" sum (which can be smaller when
        // a block doesn't fit the remaining budget).
        plan.target_resident_bytes = match self.inner.budget_bytes {
            Some(b) => b,
            None => plan.target_resident_bytes,
        };
        plan
    }

    fn name(&self) -> &'static str {
        "Adaptive"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offload::strategy::AccessHints;
    use std::collections::VecDeque;

    fn pressured_state<'a>(
        sizes: &'a [usize],
        history: &'a VecDeque<u32>,
        free: u64,
        total: u64,
    ) -> OffloaderState<'a> {
        OffloaderState {
            block_count: sizes.len(),
            block_sizes: sizes,
            resident: &[],
            requested: 0,
            access_history: history,
            free_vram_bytes: free,
            total_vram_bytes: total,
            hints: AccessHints {
                vram_authoritative: true,
                prefetches_since_last_plan: 0,
            },
        }
    }

    /// Under high pressure the strategy shrinks the budget.
    #[test]
    fn shrinks_under_high_pressure() {
        let sizes = vec![1024usize; 10]; // 10 KiB total
        let history: VecDeque<u32> = VecDeque::new();
        let mut a = Adaptive::new();
        // free=10%, total=100% → pressure 0.9 → above default 0.85.
        let plan = a.plan(&pressured_state(&sizes, &history, 10, 100));
        // With shrink_fraction=0.5 and total=10240 → target ≈ 5120.
        assert!(a.last_target_bytes < 10_000);
        assert!(plan.target_resident_bytes < 10_000);
    }

    /// Releasing pressure grows the budget back.
    #[test]
    fn grows_under_low_pressure() {
        let sizes = vec![1024usize; 10];
        let history: VecDeque<u32> = VecDeque::new();
        let mut a = Adaptive::new();

        // First: high pressure → shrink.
        let _ = a.plan(&pressured_state(&sizes, &history, 5, 100));
        let shrunk = a.last_target_bytes;
        assert!(shrunk < 10_000);

        // Then: low pressure → unbounded.
        let _ = a.plan(&pressured_state(&sizes, &history, 80, 100));
        assert_eq!(
            a.last_target_bytes,
            u64::MAX,
            "low pressure must grow back to unbounded"
        );
    }

    /// Medium pressure holds the previous target (hysteresis).
    #[test]
    fn holds_in_medium_band() {
        let sizes = vec![1024usize; 10];
        let history: VecDeque<u32> = VecDeque::new();
        let mut a = Adaptive::new();

        let _ = a.plan(&pressured_state(&sizes, &history, 5, 100)); // high
        let after_shrink = a.last_target_bytes;
        // pressure = 0.7, between 0.6 and 0.85 → hold.
        let _ = a.plan(&pressured_state(&sizes, &history, 30, 100));
        assert_eq!(
            a.last_target_bytes, after_shrink,
            "medium pressure must hold the previous target"
        );
    }

    /// Non-authoritative VRAM read → hold previous target, don't crash.
    #[test]
    fn non_authoritative_holds() {
        let sizes = vec![1024usize; 4];
        let history: VecDeque<u32> = VecDeque::new();
        let mut a = Adaptive::new();
        let state = OffloaderState {
            block_count: sizes.len(),
            block_sizes: &sizes,
            resident: &[],
            requested: 0,
            access_history: &history,
            free_vram_bytes: 0,
            total_vram_bytes: 0,
            hints: AccessHints {
                vram_authoritative: false,
                prefetches_since_last_plan: 0,
            },
        };
        let plan = a.plan(&state);
        // Should not panic; should produce a valid plan.
        assert!(plan.keep.len() <= sizes.len());
        assert!(a.last_pressure.is_nan());
    }
}
