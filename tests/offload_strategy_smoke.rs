//! Smoke tests for `flame_core::offload::strategy` (Phase 2 FlexTensor port).
//!
//! These tests exercise the strategy trait + its three implementations
//! (`TwoSlot`, `Knapsack`, `Adaptive`) at the unit level. They do NOT
//! require a CUDA device — strategies are pure host logic per the
//! design (sync clause 1).
//!
//! Per tenets §4 (measurement beats assertion) the assertions are
//! quantitative: the regression gate proves that `TwoSlot` mirrors the
//! pre-Phase-2 hardcoded path's eviction picks, knapsack stays inside
//! its byte budget, and adaptive shrinks under simulated pressure.

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use std::collections::VecDeque;

use flame_core::offload::strategy::{
    AccessHints, Adaptive, Knapsack, OffloaderState, Strategy, TwoSlot,
};

// ---------------------------------------------------------------------------
// Helper: build a synthetic OffloaderState.
// ---------------------------------------------------------------------------

fn state<'a>(
    resident: &'a [usize],
    requested: usize,
    history: &'a VecDeque<u32>,
    sizes: &'a [usize],
    free_vram: u64,
    total_vram: u64,
    vram_authoritative: bool,
) -> OffloaderState<'a> {
    OffloaderState {
        block_count: sizes.len(),
        block_sizes: sizes,
        resident,
        requested,
        access_history: history,
        free_vram_bytes: free_vram,
        total_vram_bytes: total_vram,
        hints: AccessHints {
            vram_authoritative,
            prefetches_since_last_plan: 0,
        },
    }
}

/// Reference implementation of the pre-Phase-2 2-slot eviction rule.
///
/// The pre-Phase-2 `BlockOffloader::prefetch_block` picked the
/// non-active slot — i.e. `1 - active`. With history available, the
/// "active" slot holds the most-recently-awaited block. So:
///
/// * If both slots already hold the requested block → no eviction
///   (no-op prefetch).
/// * Otherwise → evict the slot whose `block_idx` is NOT the most
///   recent in `history`.
fn pre_phase2_eviction_pick(
    resident: &[usize],
    requested: usize,
    history: &VecDeque<u32>,
) -> Vec<usize> {
    if resident.contains(&requested) {
        return Vec::new();
    }
    let last = history.front().map(|&b| b as usize);
    let mut evict: Vec<usize> = Vec::new();
    for &b in resident {
        if Some(b) != last {
            evict.push(b);
        }
    }
    if evict.is_empty() && !resident.is_empty() {
        evict.push(resident[0]);
    }
    evict
}

// ---------------------------------------------------------------------------
// TwoSlot regression gate
// ---------------------------------------------------------------------------

/// `TwoSlot` reformulated as a `Strategy` must produce the same
/// eviction picks as the pre-Phase-2 hardcoded logic on a synthetic
/// block schedule. This is the bit-identical-behavior gate for the
/// klein 9B loss invariant.
#[test]
fn two_slot_matches_hardcoded_pre_phase2() {
    let sizes = vec![10usize; 12]; // 12 uniform blocks
    let mut ts = TwoSlot::new();

    // Walk a sliding 2-slot pattern: at each step, the requested block
    // advances by 1, the "resident" pair is the previous two requests.
    let mut history: VecDeque<u32> = VecDeque::new();
    let mut resident: Vec<usize> = Vec::new();

    for step in 0..12 {
        let requested = step;

        // Strategy decision.
        let plan = ts.plan(&state(&resident, requested, &history, &sizes, 0, 0, false));

        // Reference decision.
        let ref_evict = pre_phase2_eviction_pick(&resident, requested, &history);

        assert_eq!(
            plan.evict, ref_evict,
            "step {step}: TwoSlot eviction picks must match pre-Phase-2 hardcoded path\n\
             resident={resident:?} requested={requested} history={history:?}"
        );

        // Update simulated resident set and history.
        if resident.contains(&requested) {
            // No-op; history still rolls forward.
        } else {
            // Evict whatever the strategy said to evict; insert
            // requested.
            for e in &plan.evict {
                resident.retain(|x| x != e);
            }
            resident.push(requested);
            // Cap at 2 (the offloader's hardcoded slot count).
            while resident.len() > 2 {
                resident.remove(0);
            }
        }
        history.push_front(requested as u32);
        if history.len() > 64 {
            history.pop_back();
        }
    }
}

/// `TwoSlot` never errors on edge cases (empty state, single block).
#[test]
fn two_slot_handles_edge_cases() {
    let mut ts = TwoSlot::new();

    // Empty everything.
    let history = VecDeque::new();
    let _ = ts.plan(&state(&[], 0, &history, &[], 0, 0, false));

    // Single block, nothing resident.
    let sizes = vec![100usize];
    let plan = ts.plan(&state(&[], 0, &history, &sizes, 0, 0, false));
    assert!(plan.evict.is_empty(), "no eviction with empty residency");
    assert_eq!(plan.prefetch, vec![0]);
}

// ---------------------------------------------------------------------------
// Knapsack budget invariant
// ---------------------------------------------------------------------------

/// Knapsack must never plan more resident bytes than the VRAM budget
/// allows — across a sweep of budgets and block size distributions.
#[test]
fn knapsack_never_exceeds_budget() {
    let configs: &[(Vec<usize>, u64)] = &[
        // (block_sizes, budget)
        (vec![100; 8], 250),
        (vec![1024, 2048, 4096, 8192, 16384], 7000),
        (vec![1; 50], 25),
        (
            (0..32)
                .map(|i| (1 << (i % 12)) as usize)
                .collect::<Vec<_>>(),
            1 << 14,
        ),
    ];

    for (sizes, budget) in configs {
        let history: VecDeque<u32> = VecDeque::from_iter((0..16).map(|i| i % sizes.len() as u32));
        let mut k = Knapsack::with_budget(*budget);
        for requested in 0..sizes.len().min(8) {
            let plan = k.plan(&state(&[], requested, &history, sizes, 0, 0, false));
            let bytes = plan.keep_bytes(sizes);
            assert!(
                bytes <= *budget,
                "knapsack exceeded budget: kept {} bytes > budget {} (sizes={:?}, req={})",
                bytes,
                budget,
                sizes,
                requested
            );
        }
    }
}

/// Knapsack honors `target_resident_bytes` as a real cap (not just
/// informational). The reported target must be <= budget.
#[test]
fn knapsack_target_resident_bounded_by_budget() {
    let sizes = vec![100usize; 50];
    let history: VecDeque<u32> = VecDeque::new();
    for budget in [50u64, 99, 100, 101, 500, 1234] {
        let mut k = Knapsack::with_budget(budget);
        let plan = k.plan(&state(&[], 0, &history, &sizes, 0, 0, false));
        assert!(
            plan.target_resident_bytes <= budget,
            "knapsack target_resident_bytes={} > budget {}",
            plan.target_resident_bytes,
            budget
        );
    }
}

// ---------------------------------------------------------------------------
// Adaptive pressure response
// ---------------------------------------------------------------------------

/// Adaptive shrinks the resident-set when VRAM pressure crosses the
/// high watermark, and grows back when pressure releases.
#[test]
fn adaptive_shrinks_under_pressure_grows_back() {
    let sizes = vec![1024usize * 1024; 20]; // 20 × 1 MiB blocks = 20 MiB total
    let history: VecDeque<u32> = VecDeque::from_iter((0..16).map(|i| i % 20));
    let mut a = Adaptive::new();

    let total_block_bytes: u64 = sizes.iter().map(|&s| s as u64).sum();

    // Phase 1: very high pressure (98% used) → shrink.
    let plan_shrunk = a.plan(&state(
        &[],
        0,
        &history,
        &sizes,
        /* free */ 2 * 1024 * 1024,
        /* total */ 100 * 1024 * 1024,
        true,
    ));
    let shrunk_target = a.last_target_bytes();
    assert!(
        shrunk_target < total_block_bytes,
        "shrunk target ({shrunk_target}) must be < total block bytes ({total_block_bytes})"
    );
    assert!(
        plan_shrunk.target_resident_bytes < total_block_bytes,
        "plan target_resident_bytes ({}) must be < total ({})",
        plan_shrunk.target_resident_bytes,
        total_block_bytes
    );
    assert!(
        a.last_observed_pressure() > 0.95,
        "expected high pressure observation, got {}",
        a.last_observed_pressure()
    );

    // Phase 2: very low pressure (10% used) → grow back to unbounded.
    let _ = a.plan(&state(
        &[],
        0,
        &history,
        &sizes,
        /* free */ 90 * 1024 * 1024,
        /* total */ 100 * 1024 * 1024,
        true,
    ));
    assert_eq!(
        a.last_target_bytes(),
        u64::MAX,
        "low pressure must grow target back to unbounded"
    );
}

/// Adaptive must not crash or change state when `vram_authoritative`
/// is false — it should hold its previous target.
#[test]
fn adaptive_stable_when_vram_read_non_authoritative() {
    let sizes = vec![1024usize; 4];
    let history: VecDeque<u32> = VecDeque::new();
    let mut a = Adaptive::new();

    // Force a shrink first.
    let _ = a.plan(&state(&[], 0, &history, &sizes, 1, 100, true));
    let prior = a.last_target_bytes();

    // Now serve a non-authoritative state — target should not change.
    let _ = a.plan(&state(&[], 0, &history, &sizes, 0, 0, false));
    assert_eq!(
        a.last_target_bytes(),
        prior,
        "non-authoritative VRAM read must hold previous target"
    );
}

// ---------------------------------------------------------------------------
// Planner sanity
// ---------------------------------------------------------------------------

#[test]
fn planner_smallest_and_largest_first_respect_budget() {
    use flame_core::offload::planner::{plan_largest_first, plan_smallest_first, PlanRequest};

    let sizes = vec![100usize, 50, 200, 25, 75, 1000];
    let budget = 200;
    for plan_fn in [plan_smallest_first, plan_largest_first] {
        let res = plan_fn(&PlanRequest {
            block_sizes: &sizes,
            budget_bytes: Some(budget),
        });
        assert!(res.total_bytes <= budget);
    }
}

// ---------------------------------------------------------------------------
// Telemetry plumbing
// ---------------------------------------------------------------------------

/// The strategy decision hook in telemetry increments counters under
/// `set_enabled(true)`.
#[test]
fn telemetry_strategy_counters_record() {
    use flame_core::offload::telemetry;

    let t = telemetry::global();
    t.set_enabled(true);

    let before = t.snapshot();
    t.record_strategy_decision(
        "TwoSlot", /* evicted */ 1, /* kept */ 2, /* target */ 4096,
    );
    t.record_strategy_decision("Knapsack", 3, 5, 8192);
    let after = t.snapshot();

    assert_eq!(after.strategy_plans - before.strategy_plans, 2);
    assert_eq!(
        after.strategy_eviction_decisions - before.strategy_eviction_decisions,
        4
    );
    assert_eq!(after.strategy_keep_total - before.strategy_keep_total, 7);
    assert_eq!(after.strategy_last_target_resident_bytes, 8192);
}
