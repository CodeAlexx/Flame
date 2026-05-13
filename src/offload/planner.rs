//! Block-resident planner.
//!
//! Given per-block sizes and a VRAM budget, decide which blocks fit
//! resident. The Python original (`memory_block_planner.py`) solves a
//! richer problem: assign tensors into N adjacency-respecting *buckets*
//! that PyTorch's hook-based capture has to discover. flame-core's
//! `BlockOffloader` already knows its block geometry through
//! [`BlockFacilitator`](super::BlockFacilitator) — the block indices
//! are stable inputs, not outputs of a coloring algorithm.
//!
//! What we keep from `memory_block_planner.py`:
//!
//! * **Capacity-bounded selection** (`allocate_blocks(..., block_capacity)`)
//!   — pick a subset of blocks whose total size fits a budget.
//! * **Minimum-block-bytes summary** (`find_minimum_block_sizes`) —
//!   the maximum block size among a selection (the offloader needs a
//!   slot at least this big to host any of them at once).
//!
//! What we drop:
//!
//! * Adjacency graph coloring — flame-core's blocks are flat IDs and
//!   the offloader never needs to "place" a block in some other
//!   block's slot. The 2-slot ping-pong handles ordering directly.
//! * Recursive backtracking allocator — overkill for the
//!   "does this fit?" question on a flat array.
//!
//! The planner is consumed by [`super::strategy::Knapsack`] and
//! [`super::strategy::Adaptive`] when they need to translate a byte
//! budget into a concrete subset of block IDs.

/// A planning request: per-block bytes + a VRAM budget.
#[derive(Debug, Clone)]
pub struct PlanRequest<'a> {
    /// Bytes per block. Length = block_count. Zero-byte entries mean
    /// "empty block / free to keep resident".
    pub block_sizes: &'a [usize],
    /// Hard ceiling on total resident bytes. `None` means "no limit"
    /// (every block fits).
    pub budget_bytes: Option<u64>,
}

/// A planning result.
#[derive(Debug, Clone, Default)]
pub struct PlanResult {
    /// Block IDs selected as resident, in ascending order.
    pub resident: Vec<usize>,
    /// Sum of `block_sizes[i]` for `i in resident`.
    pub total_bytes: u64,
    /// Maximum `block_sizes[i]` among the selected residents. The
    /// offloader needs at least this much slot capacity to host any
    /// of these blocks at once.
    pub max_block_bytes: u64,
}

/// Greedy size-first selection: keep the smallest blocks first until
/// the budget is exhausted. The intent is to maximize *count* of
/// resident blocks (hit-rate proxy) rather than total bytes.
///
/// For a value-aware selection use [`super::strategy::Knapsack`]
/// directly; this is a "no opinion, just fit" baseline.
pub fn plan_smallest_first(req: &PlanRequest) -> PlanResult {
    let mut indexed: Vec<(usize, u64)> = req
        .block_sizes
        .iter()
        .enumerate()
        .map(|(i, &s)| (i, s as u64))
        .collect();
    indexed.sort_unstable_by_key(|&(_, s)| s);

    let mut chosen: Vec<usize> = Vec::new();
    let mut used: u64 = 0;
    let mut max_block: u64 = 0;
    let budget = req.budget_bytes.unwrap_or(u64::MAX);

    for (idx, bytes) in indexed {
        let next = used.saturating_add(bytes);
        if next <= budget {
            chosen.push(idx);
            used = next;
            max_block = max_block.max(bytes);
        }
    }
    chosen.sort_unstable();
    PlanResult {
        resident: chosen,
        total_bytes: used,
        max_block_bytes: max_block,
    }
}

/// Largest-first variant. Fills the budget with the smallest possible
/// count of large blocks. Useful when "fewer transfers" is more
/// important than "more resident blocks" (e.g. video DiT with large
/// stack-shared activations).
pub fn plan_largest_first(req: &PlanRequest) -> PlanResult {
    let mut indexed: Vec<(usize, u64)> = req
        .block_sizes
        .iter()
        .enumerate()
        .map(|(i, &s)| (i, s as u64))
        .collect();
    indexed.sort_unstable_by(|a, b| b.1.cmp(&a.1));

    let mut chosen: Vec<usize> = Vec::new();
    let mut used: u64 = 0;
    let mut max_block: u64 = 0;
    let budget = req.budget_bytes.unwrap_or(u64::MAX);

    for (idx, bytes) in indexed {
        let next = used.saturating_add(bytes);
        if next <= budget {
            chosen.push(idx);
            used = next;
            max_block = max_block.max(bytes);
        }
    }
    chosen.sort_unstable();
    PlanResult {
        resident: chosen,
        total_bytes: used,
        max_block_bytes: max_block,
    }
}

/// Helper: max block size — useful when sizing a slot's capacity to
/// guarantee any selected block fits.
pub fn max_block_bytes(block_sizes: &[usize]) -> u64 {
    block_sizes.iter().map(|&s| s as u64).max().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smallest_first_budget_respected() {
        let sizes = vec![100usize, 50, 200, 25, 75];
        let req = PlanRequest {
            block_sizes: &sizes,
            budget_bytes: Some(150),
        };
        let res = plan_smallest_first(&req);
        assert!(res.total_bytes <= 150);
        // smallest 25, 50, 75 = 150 fits.
        assert!(res.resident.contains(&3)); // 25
        assert!(res.resident.contains(&1)); // 50
        assert!(res.resident.contains(&4)); // 75
        assert!(!res.resident.contains(&2)); // 200 — too big
    }

    #[test]
    fn largest_first_budget_respected() {
        let sizes = vec![100usize, 50, 200, 25, 75];
        let req = PlanRequest {
            block_sizes: &sizes,
            budget_bytes: Some(250),
        };
        let res = plan_largest_first(&req);
        assert!(res.total_bytes <= 250);
        // largest 200 fits; next 100 → 300 > 250, skip; next 75 → 275 > 250, skip;
        // next 50 → 250 fits.
        assert!(res.resident.contains(&2)); // 200
    }

    #[test]
    fn no_budget_keeps_all() {
        let sizes = vec![100usize, 50, 200, 25];
        let req = PlanRequest {
            block_sizes: &sizes,
            budget_bytes: None,
        };
        let res = plan_smallest_first(&req);
        assert_eq!(res.resident.len(), 4);
        assert_eq!(res.total_bytes, 375);
    }

    #[test]
    fn empty_inputs_safe() {
        let req = PlanRequest {
            block_sizes: &[],
            budget_bytes: Some(100),
        };
        let res = plan_smallest_first(&req);
        assert!(res.resident.is_empty());
        assert_eq!(res.total_bytes, 0);
        assert_eq!(res.max_block_bytes, 0);
    }

    #[test]
    fn max_block_bytes_basic() {
        assert_eq!(max_block_bytes(&[10, 50, 30]), 50);
        assert_eq!(max_block_bytes(&[]), 0);
    }
}
