//! MoE expert-choice routing primitives.
//!
//! Given a token sequence and a router weight, decide which expert sees
//! which tokens. This module implements **expert-choice** routing as used
//! by Nucleus-Image (and most modern Mixture-of-Experts diffusion models):
//! each expert independently picks its top-C most-affinity-scored tokens,
//! where C = `ceil(capacity_factor * S / E)`. Per-token K is therefore
//! variable (a token may be picked by 0, 1, or many experts).
//!
//! Reference (Nucleus diffusers):
//! `transformer_nucleusmoe_image.py::NucleusMoELayer.forward` (lines 548-604):
//!
//! ```text
//! logits   = router(router_input)          # (B, S, E)
//! scores   = softmax(logits, dim=-1)       # over experts
//! affinity = scores.transpose(1, 2)        # (B, E, S)
//! capacity = ceil(capacity_factor * S / E)
//! top_w, top_idx = topk(affinity, k=capacity, dim=-1)   # (B, E, C)
//! ```
//!
//! Then the gating values are renormalised per-token (a token's gates
//! across all experts that picked it must sum to 1) and scaled by
//! `route_scale`. Tokens not picked by any expert get gating 0 from MoE
//! and pass through only the shared expert.
//!
//! The output of this module is a packed flat plan suitable for direct
//! consumption by `flame_core::ops::grouped_mm::grouped_mm_bf16` (offsets)
//! and `flame_core::ops::fused_gated_scatter_add::fused_gated_scatter_add_bf16`
//! (indices, gating). All host slices, since both downstream wrappers take
//! `&[i32]` / `Vec<f32>` host slices.
//!
//! ## Where the work happens
//!
//! Top-K is computed **host-side** (download → partial sort per row →
//! return). Rationale: the downstream MoE primitives already accept host
//! `&[i32]` for offsets/indices because flame-core's `DType::I32` is
//! f32-bytes-relabeled; producing the indices on the GPU would require
//! either a real-i32 dtype path or a custom CUDA top-K kernel. For
//! batch=1 inference at S~1024 / E~64, the affinity matrix is ~64K f32s
//! per layer per step — sub-millisecond to download and partial-sort on
//! a modern host. If profiling shows it dominates, swap in a CUDA top-K
//! later (see HANDOFF_2026-04-29_MOE_KERNELS_AND_NUCLEUS.md, Phase 2 fast
//! path).

use crate::{DType, Error, Result, Shape, Tensor};

/// Output of `expert_choice_route`: everything the downstream MoE
/// primitives need, in the layout they expect.
#[derive(Debug, Clone)]
pub struct ExpertRoutingPlan {
    /// Capacity per expert (tokens each expert picks, uniform).
    pub capacity: usize,
    /// Number of experts (E).
    pub num_experts: usize,
    /// Batch size (B).
    pub batch_size: usize,
    /// Sequence length per batch (S).
    pub seq_len: usize,

    /// Constant offsets vector for `grouped_mm_bf16`. Length = `E`,
    /// values `[B*C, 2*B*C, ..., E*B*C]`. Exclusive cumulative.
    /// For expert-choice routing this is fixed per (B, E, C) — does not
    /// depend on which tokens were picked.
    pub offsets: Vec<i32>,

    /// Global token indices for `fused_gated_scatter_add_bf16`.
    /// Length = `E*B*C`. Each entry is in `[0, B*S)`.
    /// Layout: expert-major, then within an expert, batch-major, then
    /// the C picks for that (expert, batch) row.
    pub global_token_indices: Vec<i32>,

    /// Per-pick gating weights. Length = `E*B*C`. Same layout as
    /// `global_token_indices`. **Renormalised** per-token so a token's
    /// gates across all experts that picked it sum to 1, and scaled by
    /// `route_scale`.
    pub gating_flat: Vec<f32>,
}

/// Run expert-choice routing.
///
/// # Arguments
/// - `affinity`: `(B, E, S)` F32 tensor on GPU. The caller is responsible
///   for: building the (B, S, E) router logits via a matmul, applying
///   softmax (or sigmoid) over the last dim, then transposing axes 1 & 2.
///   See module-level doc for the full reference recipe.
/// - `capacity`: top-C per expert. Caller computes
///   `ceil(capacity_factor * S / E)` from the layer config (Nucleus uses
///   `capacity_factor` ∈ {2, 4} per layer).
/// - `route_scale`: multiplier applied to the renormalised gating values.
///   Nucleus's `route_scale=2.5`. Pass `1.0` if your model doesn't have
///   an explicit scale.
///
/// # Returns
/// An `ExpertRoutingPlan`. The `offsets` field plugs straight into
/// `grouped_mm_bf16(.., offsets: &offsets, t_max: B*C)`. The
/// `global_token_indices` and `gating_flat` plug into the
/// `fused_gated_scatter_add_bf16` post-FFN combine.
pub fn expert_choice_route(
    affinity: &Tensor,
    capacity: usize,
    route_scale: f32,
) -> Result<ExpertRoutingPlan> {
    if capacity == 0 {
        return Err(Error::InvalidOperation(
            "expert_choice_route: capacity must be > 0".into(),
        ));
    }

    // We do the math in F32 host-side. `affinity` may live as BF16 or F32
    // on GPU; cast through to_dtype if needed and download.
    let aff_dims = affinity.shape().dims().to_vec();
    if aff_dims.len() != 3 {
        return Err(Error::InvalidOperation(format!(
            "expert_choice_route: affinity must be 3-D (B, E, S), got {aff_dims:?}"
        )));
    }
    let (b, e, s) = (aff_dims[0], aff_dims[1], aff_dims[2]);

    if capacity > s {
        return Err(Error::InvalidOperation(format!(
            "expert_choice_route: capacity={capacity} > S={s}"
        )));
    }

    let aff_f32 = if affinity.dtype() == DType::F32 {
        affinity.clone()
    } else {
        affinity.to_dtype(DType::F32)?
    };
    let aff_host: Vec<f32> = aff_f32.to_vec()?;
    debug_assert_eq!(aff_host.len(), b * e * s, "affinity host len mismatch");

    // Per-row top-K via a partial sort. We keep (value, index) pairs in
    // expert-major order for flat output.
    //
    // Shape of intermediate buffers:
    //   top_idx : (B, E, C) i64    — local-to-batch token positions in [0, S)
    //   top_w   : (B, E, C) f32    — affinity values
    //
    // Storage layout: we flatten (B, E, C) row-major. Index =
    //   b * (E * C) + e * C + c.
    let mut top_idx: Vec<i32> = vec![0; b * e * capacity];
    let mut top_w: Vec<f32> = vec![0.0; b * e * capacity];

    let mut row: Vec<(f32, i32)> = Vec::with_capacity(s);
    for bi in 0..b {
        for ei in 0..e {
            let row_start = (bi * e + ei) * s;
            row.clear();
            for si in 0..s {
                row.push((aff_host[row_start + si], si as i32));
            }
            // Partial sort: select_nth_unstable_by + sort the prefix.
            // For C ≤ 256 and S ≤ 1024, this is fast enough on host.
            // Sort descending by affinity; ties broken by lower index.
            row.select_nth_unstable_by(capacity - 1, |a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.1.cmp(&b.1))
            });
            row[..capacity].sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.1.cmp(&b.1))
            });
            let dst = (bi * e + ei) * capacity;
            for (k, &(v, i)) in row[..capacity].iter().enumerate() {
                top_w[dst + k] = v;
                top_idx[dst + k] = i;
            }
        }
    }

    // Build the flat global-token-indices and flat gating.
    //
    // Reference recipe (transformer_nucleusmoe_image.py:576-578):
    //   batch_offsets = arange(B) * S        # (B,)
    //   global_token_indices =
    //       (batch_offsets[:,None,None] + top_idx)   # (B, E, C)
    //       .transpose(0, 1)                          # (E, B, C)
    //       .reshape(E, -1).reshape(-1)               # (E*B*C,)
    //   gating_flat = top_w.transpose(0, 1).reshape(E, -1).reshape(-1)
    //
    // After transpose-then-flatten the layout is expert-major, then within
    // an expert it's batch-major, then the C picks for that (e, b) row.
    let n_global = e * b * capacity;
    let mut global_token_indices: Vec<i32> = vec![0; n_global];
    let mut gating_packed: Vec<f32> = vec![0.0; n_global];

    for ei in 0..e {
        for bi in 0..b {
            let src = (bi * e + ei) * capacity; // (B, E, C) row-major
            let dst = (ei * b + bi) * capacity; // (E, B, C) row-major
            let batch_off = (bi * s) as i32;
            for k in 0..capacity {
                global_token_indices[dst + k] = batch_off + top_idx[src + k];
                gating_packed[dst + k] = top_w[src + k];
            }
        }
    }

    // Per-token renormalisation: a token may be picked by 0, 1, or many
    // experts. Sum the gates per global token, then divide each pick by
    // its token's sum. Tokens with sum == 0 (picked by no expert) keep
    // gating 0; the token's final contribution then comes only from the
    // shared expert path.
    let total_tokens = b * s;
    let mut token_score_sums: Vec<f32> = vec![0.0; total_tokens];
    for k in 0..n_global {
        let idx = global_token_indices[k] as usize;
        if idx < total_tokens {
            token_score_sums[idx] += gating_packed[k];
        }
    }
    for k in 0..n_global {
        let idx = global_token_indices[k] as usize;
        if idx < total_tokens {
            let s_sum = token_score_sums[idx];
            // Numerical floor matches the reference (`+ 1e-12`).
            let denom = s_sum + 1e-12;
            gating_packed[k] = (gating_packed[k] / denom) * route_scale;
        } else {
            gating_packed[k] = 0.0;
        }
    }

    // Constant offsets for the grouped_mm: every expert is assigned
    // exactly B*C rows of the permuted input.
    let bc = (b * capacity) as i32;
    let offsets: Vec<i32> = (1..=e as i32).map(|i| i * bc).collect();

    Ok(ExpertRoutingPlan {
        capacity,
        num_experts: e,
        batch_size: b,
        seq_len: s,
        offsets,
        global_token_indices,
        gating_flat: gating_packed,
    })
}

/// Permute a flat `(B*S, D)` token tensor into expert-major
/// `(E*B*C, D)` order using the global token indices from a routing plan.
///
/// This is the "scatter to experts" step that precedes
/// `grouped_mm_bf16` in the MoE forward. The output rows are arranged
/// expert-major (block 0 of `B*C` rows is for expert 0, etc.) so the
/// constant `plan.offsets` slice partitions them cleanly.
///
/// Uses `Tensor::index_select0` under the hood. The wrapper builds an
/// I32 index tensor from `plan.global_token_indices`. flame-core's I32
/// storage is f32-bytes-relabeled, but `gather_rows` happens to read its
/// indices via an f32-→int cast (`static_cast<int>(idx[i])`), so an
/// f32-valued index tensor that we then `.to_dtype(I32)` works correctly
/// — every index in our routing plan is an exact f32 integer (≤ 2²⁴).
///
/// # Arguments
/// - `x`: `(B*S, D)` BF16 (or F32) hidden states. The caller is
///   responsible for flattening from `(B, S, D)` first.
/// - `plan`: routing plan from `expert_choice_route`.
///
/// # Returns
/// `(E*B*C, D)` tensor of the same dtype as `x`, in expert-major order.
pub fn permute_tokens(x: &Tensor, plan: &ExpertRoutingPlan) -> Result<Tensor> {
    let x_dims = x.shape().dims();
    if x_dims.len() != 2 {
        return Err(Error::InvalidOperation(format!(
            "permute_tokens: x must be 2-D (B*S, D), got {x_dims:?}"
        )));
    }
    let total_tokens = plan.batch_size * plan.seq_len;
    if x_dims[0] != total_tokens {
        return Err(Error::InvalidOperation(format!(
            "permute_tokens: x rows = {} but B*S = {}",
            x_dims[0], total_tokens
        )));
    }

    // Build the index tensor on the device. We keep f32 values (1.0, 2.0, ...)
    // and label as I32 so `gather_rows` round-trips them back to int via
    // its `static_cast<int>(idx[i])`. Every value in our routing plan is
    // a non-negative integer ≤ B*S, well within exact-f32 range.
    let n_idx = plan.global_token_indices.len();
    if n_idx == 0 {
        return Err(Error::InvalidOperation(
            "permute_tokens: routing plan has no indices".into(),
        ));
    }
    let idx_f32: Vec<f32> = plan
        .global_token_indices
        .iter()
        .map(|&v| v as f32)
        .collect();
    let idx_t = Tensor::from_vec_dtype(
        idx_f32,
        Shape::from_dims(&[n_idx]),
        x.device().clone(),
        DType::F32,
    )?
    .to_dtype(DType::I32)?;

    x.index_select0(&idx_t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaDevice;

    #[test]
    fn expert_choice_route_basic() -> Result<()> {
        // (B=1, E=4, S=8), capacity=2 → each expert picks its top 2 tokens.
        let device = CudaDevice::new(0).map_err(|e| Error::Cuda(format!("{e:?}")))?;
        let b = 1usize;
        let e = 4usize;
        let s = 8usize;
        let c = 2usize;

        // Hand-crafted affinity: each expert has obvious top-2 picks.
        // affinity[b, e, s] — expert e prefers token (e*2) and (e*2+1).
        let mut aff = vec![0.0f32; b * e * s];
        for ei in 0..e {
            for si in 0..s {
                let row = (0 * e + ei) * s + si;
                aff[row] = if si == ei * 2 || si == ei * 2 + 1 {
                    1.0
                } else {
                    0.1
                };
            }
        }
        let aff_t = Tensor::from_vec_dtype(
            aff.clone(),
            Shape::from_dims(&[b, e, s]),
            device.clone(),
            DType::F32,
        )?;

        let plan = expert_choice_route(&aff_t, c, 1.0)?;

        assert_eq!(plan.capacity, c);
        assert_eq!(plan.num_experts, e);
        assert_eq!(plan.offsets, vec![2, 4, 6, 8]);

        // global_token_indices laid out (E, B, C):
        //   expert 0 → (b=0, c={0,1}) → token 0, token 1
        //   expert 1 → (b=0, c={0,1}) → token 2, token 3
        //   ...
        let expected: Vec<i32> = (0..(e * c) as i32).collect();
        assert_eq!(plan.global_token_indices, expected);

        // Each token in this contrived input is picked by exactly ONE
        // expert, so the per-token sum is the original value (1.0).
        // Renormalised value = 1.0 / (1.0 + 1e-12) * 1.0 ≈ 1.0.
        for &g in &plan.gating_flat {
            assert!((g - 1.0).abs() < 1e-6, "expected gating ≈ 1.0, got {g}");
        }

        Ok(())
    }

    #[test]
    fn expert_choice_route_collisions_renormalize() -> Result<()> {
        // (B=1, E=2, S=4), capacity=4. Both experts can pick all 4 tokens.
        // Affinity is uniform 0.5 → every (E, S) slot is 0.5 → after
        // renormalisation each token's TWO gates (one from each expert)
        // should each be 0.5 (so they sum to 1.0).
        let device = CudaDevice::new(0).map_err(|e| Error::Cuda(format!("{e:?}")))?;
        let b = 1usize;
        let e = 2usize;
        let s = 4usize;
        let c = 4usize;

        let aff = vec![0.5f32; b * e * s];
        let aff_t = Tensor::from_vec_dtype(
            aff,
            Shape::from_dims(&[b, e, s]),
            device.clone(),
            DType::F32,
        )?;

        let plan = expert_choice_route(&aff_t, c, 1.0)?;

        // Each pick's gate before renorm = 0.5, token score sum = 0.5+0.5=1.0,
        // so renormalised = 0.5/1.0 = 0.5.
        for &g in &plan.gating_flat {
            assert!((g - 0.5).abs() < 1e-6, "expected gating ≈ 0.5, got {g}");
        }

        // offsets = [B*C, 2*B*C] = [4, 8]
        assert_eq!(plan.offsets, vec![4, 8]);
        Ok(())
    }

    #[test]
    fn permute_tokens_round_trip() -> Result<()> {
        // (B=1, E=4, S=8, D=3), capacity=2.
        // Build the same toy plan as `expert_choice_route_basic`:
        // expert e picks tokens (e*2, e*2+1).
        // After permute, output rows must be x[0], x[1], x[2], x[3], ...,
        // x[6], x[7] in order.
        let device = CudaDevice::new(0).map_err(|e| Error::Cuda(format!("{e:?}")))?;
        let b = 1usize;
        let e = 4usize;
        let s = 8usize;
        let d = 3usize;
        let c = 2usize;

        // x: row i has values [i*10, i*10+1, i*10+2] so we can spot-check by
        // value after permute.
        let x_data: Vec<f32> = (0..(b * s) * d)
            .map(|i| (i / d) as f32 * 10.0 + (i % d) as f32)
            .collect();
        let x = Tensor::from_vec_dtype(
            x_data.clone(),
            Shape::from_dims(&[b * s, d]),
            device.clone(),
            DType::F32,
        )?;

        // Build the affinity that produces the toy plan, then route.
        let mut aff = vec![0.1f32; b * e * s];
        for ei in 0..e {
            aff[ei * s + ei * 2] = 1.0;
            aff[ei * s + ei * 2 + 1] = 1.0;
        }
        let aff_t = Tensor::from_vec_dtype(
            aff,
            Shape::from_dims(&[b, e, s]),
            device.clone(),
            DType::F32,
        )?;
        let plan = expert_choice_route(&aff_t, c, 1.0)?;

        let x_perm = permute_tokens(&x, &plan)?;

        let got = x_perm.to_vec()?;
        // Expected: rows 0..7 in order (each expert took its 2 tokens
        // contiguously, and the experts together cover all 8 tokens).
        assert_eq!(got.len(), e * c * d);
        for row in 0..(e * c) {
            for col in 0..d {
                let expected = row as f32 * 10.0 + col as f32;
                let v = got[row * d + col];
                assert!(
                    (v - expected).abs() < 1e-5,
                    "permute_tokens mismatch at row={row} col={col}: got {v}, expected {expected}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn expert_choice_route_route_scale_applied() -> Result<()> {
        // Same as basic but route_scale=2.5 → expected gating ≈ 2.5.
        let device = CudaDevice::new(0).map_err(|e| Error::Cuda(format!("{e:?}")))?;
        let b = 1usize;
        let e = 4usize;
        let s = 8usize;
        let c = 2usize;
        let mut aff = vec![0.1f32; b * e * s];
        for ei in 0..e {
            aff[ei * s + ei * 2] = 1.0;
            aff[ei * s + ei * 2 + 1] = 1.0;
        }
        let aff_t = Tensor::from_vec_dtype(
            aff,
            Shape::from_dims(&[b, e, s]),
            device.clone(),
            DType::F32,
        )?;
        let plan = expert_choice_route(&aff_t, c, 2.5)?;
        for &g in &plan.gating_flat {
            assert!(
                (g - 2.5).abs() < 1e-5,
                "expected route_scale=2.5 gating, got {g}"
            );
        }
        Ok(())
    }
}
