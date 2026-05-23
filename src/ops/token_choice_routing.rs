//! MoE token-choice top-K routing primitives.
//!
//! Dual of `moe_routing::expert_choice_route`. In token-choice routing,
//! **each token picks its top-K experts** (rather than each expert picking
//! its top-C tokens). This is the routing scheme used by GPT-OSS, Mixtral,
//! Qwen-MoE, and most decoder-style MoE models.
//!
//! ## Reference recipe (transformers `GptOssTopKRouter.forward`)
//!
//! ```text
//! router_logits = router.weight @ x + router.bias          # (T, E)
//! top_val, top_idx = topk(router_logits, K, dim=-1)        # (T, K)
//! top_w  = softmax(top_val, dim=-1)                        # (T, K)
//! ```
//!
//! Note that GPT-OSS does **topk-then-softmax**, not softmax-then-topk.
//! These two are mathematically equivalent up to a constant denominator:
//!
//! ```text
//! softmax(topk(l))_i           = exp(l_i) / sum_{j∈K}(exp(l_j))
//! renorm(topk(softmax(l)))_i   = [exp(l_i)/Z] / sum_{j∈K}[exp(l_j)/Z]
//!                              = exp(l_i) / sum_{j∈K}(exp(l_j))
//! ```
//!
//! so we expose a `score_mode` parameter that picks between the two paths.
//! GPT-OSS uses `ScoreMode::TopKSoftmax`. Models that do
//! softmax-over-E-then-topk-then-renorm (Nucleus-style, Mixtral) use
//! `ScoreMode::SoftmaxRenorm`.
//!
//! ## What this returns
//!
//! A `TokenChoiceRoutingPlan` packed in the exact layout the downstream
//! MoE primitives consume:
//!
//! - `offsets`: `&[i32]` host slice (length `E+1`, the conventional
//!   prefix-sum layout including the leading 0). The wrapper exposes
//!   `offsets_for_grouped_mm()` to drop the leading 0 when calling
//!   `grouped_mm_bf16` (which expects exclusive cumulative end indices
//!   without the leading 0, length `E`).
//! - `permuted_token_indices`: `&[i32]` host slice (length `T*K`). For each
//!   expert-major row `r`, `permuted_token_indices[r]` is the source token
//!   index in `[0, T)` that produced this row.
//! - `expert_weights_flat`: `Vec<f32>` host slice (length `T*K`), aligned
//!   with `permuted_token_indices` (expert-major). Feeds straight into
//!   `fused_gated_scatter_add_bf16`.
//! - `expert_indices`: `(T, K)` I32 device tensor (which K experts each
//!   token picked; in original token-major order). Mostly informational for
//!   diagnostics; the routing pipeline doesn't strictly need it.
//! - `expert_weights`: `(T, K)` F32 device tensor (the K weights per token;
//!   in original token-major order). Same: informational.
//!
//! ## Where the work happens
//!
//! Top-K + softmax are computed **host-side** for the same reason as
//! `expert_choice_route`: the downstream `grouped_mm_bf16` and
//! `fused_gated_scatter_add_bf16` already accept host `&[i32]` slices, and
//! for T~1024 and E~32 the host-side cost is microseconds.

use crate::{DType, Error, Result, Shape, Tensor};

/// Scoring strategy for token-choice routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreMode {
    /// GPT-OSS style: `topk(logits)` then `softmax` over the K selected
    /// values. The K weights sum to 1 exactly.
    TopKSoftmax,
    /// Mixtral/Nucleus style: `softmax(logits)` over E, then `topk`, then
    /// renormalise so the K weights sum to 1.
    SoftmaxRenorm,
}

/// Token-choice routing plan: each of the T tokens picks its top-K experts.
#[derive(Debug, Clone)]
pub struct TokenChoiceRoutingPlan {
    /// Number of tokens (T).
    pub num_tokens: usize,
    /// Number of experts (E).
    pub num_experts: usize,
    /// Top-K per token (K).
    pub top_k: usize,

    /// Per-token expert ids. `(T, K)` I32 device tensor, token-major order.
    /// Each row gives the K expert ids that token picked, sorted by
    /// descending logit (stable, ties broken by lower expert id).
    pub expert_indices: Tensor,

    /// Per-token expert weights. `(T, K)` F32 device tensor, token-major
    /// order, aligned with `expert_indices`. Each row sums to 1.
    pub expert_weights: Tensor,

    /// Prefix-sum offsets, host. Length `E+1`. `offsets[0] == 0`,
    /// `offsets[e+1] - offsets[e]` is the number of (token, k) picks that
    /// fell on expert `e`. `offsets[E] == T*K`.
    pub offsets: Vec<i32>,

    /// Expert-major source token indices, host. Length `T*K`. For the i-th
    /// row of the permuted (expert-major) layout,
    /// `permuted_token_indices[i]` is the source token in `[0, T)`.
    pub permuted_token_indices: Vec<i32>,

    /// Expert-major weights, host. Length `T*K`. Aligned with
    /// `permuted_token_indices`. Feeds `fused_gated_scatter_add_bf16` as
    /// the per-row gating value.
    pub expert_weights_flat: Vec<f32>,
}

impl TokenChoiceRoutingPlan {
    /// Offsets in the layout `grouped_mm_bf16` expects: length `E`,
    /// EXCLUSIVE cumulative end indices (i.e. `self.offsets[1..]`).
    pub fn offsets_for_grouped_mm(&self) -> Vec<i32> {
        self.offsets[1..].to_vec()
    }
}

/// Run token-choice top-K routing.
///
/// # Arguments
/// - `router_logits`: `(T, E)` F32 or BF16 tensor on GPU. The caller is
///   responsible for the router matmul (and bias) that produced these
///   logits.
/// - `top_k`: K. Must satisfy `1 <= top_k <= E`.
/// - `score_mode`: how to convert logits → weights. See `ScoreMode`.
///
/// # Returns
/// A `TokenChoiceRoutingPlan` ready for `permute_tokens_for_token_choice`,
/// `grouped_mm_bf16`, and `fused_gated_scatter_add_bf16`.
pub fn token_choice_route(
    router_logits: &Tensor,
    top_k: usize,
    score_mode: ScoreMode,
) -> Result<TokenChoiceRoutingPlan> {
    if top_k == 0 {
        return Err(Error::InvalidOperation(
            "token_choice_route: top_k must be > 0".into(),
        ));
    }

    let dims = router_logits.shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(Error::InvalidOperation(format!(
            "token_choice_route: router_logits must be 2-D (T, E), got {dims:?}"
        )));
    }
    let (t, e) = (dims[0], dims[1]);
    if top_k > e {
        return Err(Error::InvalidOperation(format!(
            "token_choice_route: top_k={top_k} > E={e}"
        )));
    }

    // Host F32 logits.
    let logits_f32 = if router_logits.dtype() == DType::F32 {
        router_logits.clone()
    } else {
        router_logits.to_dtype(DType::F32)?
    };
    let logits_host: Vec<f32> = logits_f32.to_vec()?;
    debug_assert_eq!(
        logits_host.len(),
        t * e,
        "router_logits host len mismatch"
    );

    // (T, K) outputs in token-major order, host.
    let mut top_idx_host: Vec<i32> = vec![0; t * top_k];
    let mut top_w_host: Vec<f32> = vec![0.0; t * top_k];

    let mut row: Vec<(f32, i32)> = Vec::with_capacity(e);

    for ti in 0..t {
        let row_start = ti * e;

        match score_mode {
            ScoreMode::TopKSoftmax => {
                // Top-K of raw logits, then softmax over the K values.
                row.clear();
                for ei in 0..e {
                    row.push((logits_host[row_start + ei], ei as i32));
                }
                top_k_descending(&mut row, top_k);

                // Numerically-stable softmax over the K logits.
                let mut maxv = f32::NEG_INFINITY;
                for &(v, _) in &row[..top_k] {
                    if v > maxv {
                        maxv = v;
                    }
                }
                let mut sumexp = 0.0f32;
                let mut exps: [f32; 64] = [0.0; 64]; // K typically <= 8; cap at 64.
                if top_k > 64 {
                    // Fall back to a vec for the rare large-K case.
                    let mut exps_v: Vec<f32> = Vec::with_capacity(top_k);
                    for &(v, _) in &row[..top_k] {
                        let ev = (v - maxv).exp();
                        exps_v.push(ev);
                        sumexp += ev;
                    }
                    let inv = 1.0 / (sumexp + 1e-20);
                    for k in 0..top_k {
                        let (_, idx) = row[k];
                        top_idx_host[ti * top_k + k] = idx;
                        top_w_host[ti * top_k + k] = exps_v[k] * inv;
                    }
                } else {
                    for k in 0..top_k {
                        let ev = (row[k].0 - maxv).exp();
                        exps[k] = ev;
                        sumexp += ev;
                    }
                    let inv = 1.0 / (sumexp + 1e-20);
                    for k in 0..top_k {
                        let (_, idx) = row[k];
                        top_idx_host[ti * top_k + k] = idx;
                        top_w_host[ti * top_k + k] = exps[k] * inv;
                    }
                }
            }
            ScoreMode::SoftmaxRenorm => {
                // Softmax over all E logits, then top-K, then renormalise.
                let mut maxv = f32::NEG_INFINITY;
                for ei in 0..e {
                    let v = logits_host[row_start + ei];
                    if v > maxv {
                        maxv = v;
                    }
                }
                let mut sumexp = 0.0f32;
                let mut probs: Vec<f32> = Vec::with_capacity(e);
                for ei in 0..e {
                    let ev = (logits_host[row_start + ei] - maxv).exp();
                    probs.push(ev);
                    sumexp += ev;
                }
                let inv_z = 1.0 / (sumexp + 1e-20);
                for p in probs.iter_mut() {
                    *p *= inv_z;
                }
                row.clear();
                for ei in 0..e {
                    row.push((probs[ei], ei as i32));
                }
                top_k_descending(&mut row, top_k);

                // Renormalise top-K.
                let mut s = 0.0f32;
                for k in 0..top_k {
                    s += row[k].0;
                }
                let inv = 1.0 / (s + 1e-20);
                for k in 0..top_k {
                    let (v, idx) = row[k];
                    top_idx_host[ti * top_k + k] = idx;
                    top_w_host[ti * top_k + k] = v * inv;
                }
            }
        }
    }

    // Build expert-major permutation: for each expert e, collect all
    // (token, k) picks that landed on e. We pass through (T, K) in
    // token-major order so within an expert the picks are sorted by source
    // token id ascending — that gives a deterministic, contiguous layout.
    //
    // First count per expert; then bucket fill via running write cursors.
    let n_picks = t * top_k;
    let mut counts: Vec<i32> = vec![0; e];
    for ti in 0..t {
        for k in 0..top_k {
            let exp_id = top_idx_host[ti * top_k + k] as usize;
            debug_assert!(exp_id < e, "top_idx_host out of range");
            counts[exp_id] += 1;
        }
    }
    let mut offsets: Vec<i32> = vec![0; e + 1];
    for ei in 0..e {
        offsets[ei + 1] = offsets[ei] + counts[ei];
    }
    debug_assert_eq!(offsets[e] as usize, n_picks, "offsets[E] != T*K");

    let mut write_cursor: Vec<i32> = offsets[..e].to_vec();
    let mut permuted_token_indices: Vec<i32> = vec![0; n_picks];
    let mut expert_weights_flat: Vec<f32> = vec![0.0; n_picks];
    for ti in 0..t {
        for k in 0..top_k {
            let exp_id = top_idx_host[ti * top_k + k] as usize;
            let w = top_w_host[ti * top_k + k];
            let dst = write_cursor[exp_id] as usize;
            permuted_token_indices[dst] = ti as i32;
            expert_weights_flat[dst] = w;
            write_cursor[exp_id] += 1;
        }
    }

    // Device tensors for diagnostics / model code that wants the
    // token-major view.
    let device = router_logits.device().clone();
    // Build expert_indices as F32 then cast to I32 (flame-core's I32 stores
    // f32-bytes-relabeled; same trick `expert_choice_route` uses for index
    // tensors).
    let exp_idx_f32: Vec<f32> = top_idx_host.iter().map(|&v| v as f32).collect();
    let expert_indices = Tensor::from_vec_dtype(
        exp_idx_f32,
        Shape::from_dims(&[t, top_k]),
        device.clone(),
        DType::F32,
    )?
    .to_dtype(DType::I32)?;
    let expert_weights = Tensor::from_vec_dtype(
        top_w_host.clone(),
        Shape::from_dims(&[t, top_k]),
        device,
        DType::F32,
    )?;

    Ok(TokenChoiceRoutingPlan {
        num_tokens: t,
        num_experts: e,
        top_k,
        expert_indices,
        expert_weights,
        offsets,
        permuted_token_indices,
        expert_weights_flat,
    })
}

/// Partial sort: arrange the first `top_k` entries of `row` to be the top-K
/// largest values, in descending order, ties broken by lower index.
fn top_k_descending(row: &mut Vec<(f32, i32)>, top_k: usize) {
    if top_k == 0 || row.len() <= top_k {
        // Sort the full row.
        row.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        return;
    }
    row.select_nth_unstable_by(top_k - 1, |a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    row[..top_k].sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
}

/// Permute a `(T, D)` token tensor into expert-major
/// `(T*K, D)` order using a `TokenChoiceRoutingPlan`.
///
/// Each source token appears `K` times in the output, once per expert it
/// picked. The order matches `plan.permuted_token_indices` (expert-major,
/// then by source token id ascending within an expert).
///
/// Pairs with `grouped_mm_bf16(&permuted, &expert_w, &plan.offsets_for_grouped_mm(), t_max)`.
pub fn permute_tokens_for_token_choice(
    x: &Tensor,
    plan: &TokenChoiceRoutingPlan,
) -> Result<Tensor> {
    let x_dims = x.shape().dims();
    if x_dims.len() != 2 {
        return Err(Error::InvalidOperation(format!(
            "permute_tokens_for_token_choice: x must be 2-D (T, D), got {x_dims:?}"
        )));
    }
    if x_dims[0] != plan.num_tokens {
        return Err(Error::InvalidOperation(format!(
            "permute_tokens_for_token_choice: x rows = {} but plan.num_tokens = {}",
            x_dims[0], plan.num_tokens
        )));
    }

    let n_idx = plan.permuted_token_indices.len();
    if n_idx == 0 {
        return Err(Error::InvalidOperation(
            "permute_tokens_for_token_choice: routing plan has no indices".into(),
        ));
    }
    // Same f32-bytes-as-i32 trick as moe_routing::permute_tokens: build the
    // index tensor as F32 then relabel via to_dtype(I32). index_select0
    // reads its indices via static_cast<int>, so the round-trip is exact
    // for the small integers (< 2^24) in routing plans.
    let idx_f32: Vec<f32> = plan
        .permuted_token_indices
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
    fn token_choice_route_plan_structure() -> Result<()> {
        // T=4 tokens, E=4 experts, K=2.
        // Hand-crafted logits so token t picks experts {t, (t+1)%E}.
        let device = CudaDevice::new(0).map_err(|e| Error::Cuda(format!("{e:?}")))?;
        let t = 4usize;
        let e = 4usize;
        let k = 2usize;

        let mut logits = vec![-1.0f32; t * e];
        for ti in 0..t {
            logits[ti * e + ti] = 2.0; // top pick
            logits[ti * e + (ti + 1) % e] = 1.0; // second pick
        }
        let logits_t = Tensor::from_vec_dtype(
            logits,
            Shape::from_dims(&[t, e]),
            device.clone(),
            DType::F32,
        )?;
        let plan = token_choice_route(&logits_t, k, ScoreMode::TopKSoftmax)?;

        assert_eq!(plan.num_tokens, t);
        assert_eq!(plan.num_experts, e);
        assert_eq!(plan.top_k, k);

        // expert_indices in token-major order, descending by logit.
        let exp_idx_f32 = plan.expert_indices.to_dtype(DType::F32)?.to_vec()?;
        for ti in 0..t {
            assert_eq!(exp_idx_f32[ti * k] as i32, ti as i32);
            assert_eq!(exp_idx_f32[ti * k + 1] as i32, ((ti + 1) % e) as i32);
        }

        // Each expert was picked exactly twice (once as top pick, once as
        // second pick by the previous token), so counts = [2, 2, 2, 2].
        assert_eq!(plan.offsets, vec![0, 2, 4, 6, 8]);
        assert_eq!(plan.offsets_for_grouped_mm(), vec![2, 4, 6, 8]);

        // permuted_token_indices: for expert e, the sources are token e
        // (as its top pick) and token (e-1) mod E (as its second pick).
        // Within an expert we ordered picks in source-token ascending
        // order, so:
        //   expert 0 : tokens {0 (top), 3 (second from token 3)} -> [0, 3]
        //   expert 1 : tokens {1, 0} -> sorted [0, 1]
        //   expert 2 : tokens {2, 1} -> sorted [1, 2]
        //   expert 3 : tokens {3, 2} -> sorted [2, 3]
        assert_eq!(plan.permuted_token_indices, vec![0, 3, 0, 1, 1, 2, 2, 3]);

        // expert_weights_flat aligned with permuted_token_indices.
        // softmax over (2.0, 1.0) gives w_top ≈ 0.7311, w_second ≈ 0.2689.
        let w_top = 0.7310585786300049_f32;
        let w_sec = 0.2689414213699951_f32;
        // For each entry, decide whether it was a "top" or "second" pick by
        // matching against expert ids. We re-derive from the structure:
        //   expert 0: [(token 0, top), (token 3, second)]      -> [w_top, w_sec]
        //   expert 1: [(token 0, second), (token 1, top)]      -> [w_sec, w_top]
        //   expert 2: [(token 1, second), (token 2, top)]      -> [w_sec, w_top]
        //   expert 3: [(token 2, second), (token 3, top)]      -> [w_sec, w_top]
        let expected_w = vec![w_top, w_sec, w_sec, w_top, w_sec, w_top, w_sec, w_top];
        for (got, exp) in plan.expert_weights_flat.iter().zip(expected_w.iter()) {
            assert!((got - exp).abs() < 1e-5, "weight mismatch: got={got} exp={exp}");
        }
        Ok(())
    }

    #[test]
    fn token_choice_route_weights_sum_to_one_topk_softmax() -> Result<()> {
        let device = CudaDevice::new(0).map_err(|e| Error::Cuda(format!("{e:?}")))?;
        let t = 6usize;
        let e = 8usize;
        let k = 3usize;

        // Pseudo-random logits, deterministic.
        let mut state = 0xC0FFEE_u64;
        let mut next = || -> f32 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let logits: Vec<f32> = (0..t * e).map(|_| next()).collect();
        let logits_t = Tensor::from_vec_dtype(
            logits,
            Shape::from_dims(&[t, e]),
            device.clone(),
            DType::F32,
        )?;
        let plan = token_choice_route(&logits_t, k, ScoreMode::TopKSoftmax)?;

        // Token-major weights, each row sums to 1.
        let w = plan.expert_weights.to_vec()?;
        for ti in 0..t {
            let s: f32 = w[ti * k..(ti + 1) * k].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row {ti} sum {s} != 1.0");
        }

        // Flat layout total == T*K, also sums to T (each row sums to 1).
        assert_eq!(plan.expert_weights_flat.len(), t * k);
        let total: f32 = plan.expert_weights_flat.iter().sum();
        assert!(
            (total - t as f32).abs() < 1e-4,
            "flat total {total} != T={t}"
        );
        Ok(())
    }

    #[test]
    fn token_choice_route_weights_sum_to_one_softmax_renorm() -> Result<()> {
        let device = CudaDevice::new(0).map_err(|e| Error::Cuda(format!("{e:?}")))?;
        let t = 5usize;
        let e = 6usize;
        let k = 2usize;

        let mut state = 0xDEAD_BEEF_u64;
        let mut next = || -> f32 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let logits: Vec<f32> = (0..t * e).map(|_| next()).collect();
        let logits_t = Tensor::from_vec_dtype(
            logits,
            Shape::from_dims(&[t, e]),
            device.clone(),
            DType::F32,
        )?;
        let plan = token_choice_route(&logits_t, k, ScoreMode::SoftmaxRenorm)?;

        let w = plan.expert_weights.to_vec()?;
        for ti in 0..t {
            let s: f32 = w[ti * k..(ti + 1) * k].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row {ti} sum {s} != 1.0");
        }
        Ok(())
    }

    #[test]
    fn token_choice_route_offsets_correct() -> Result<()> {
        let device = CudaDevice::new(0).map_err(|e| Error::Cuda(format!("{e:?}")))?;
        // T=8, E=4, K=2. Random logits.
        let t = 8usize;
        let e = 4usize;
        let k = 2usize;
        let mut state = 0x12345678_u64;
        let mut next = || -> f32 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let logits: Vec<f32> = (0..t * e).map(|_| next()).collect();
        let logits_t = Tensor::from_vec_dtype(
            logits,
            Shape::from_dims(&[t, e]),
            device,
            DType::F32,
        )?;
        let plan = token_choice_route(&logits_t, k, ScoreMode::TopKSoftmax)?;

        // offsets has length E+1, starts at 0, ends at T*K, monotone non-dec.
        assert_eq!(plan.offsets.len(), e + 1);
        assert_eq!(plan.offsets[0], 0);
        assert_eq!(plan.offsets[e] as usize, t * k);
        for w in plan.offsets.windows(2) {
            assert!(w[0] <= w[1], "offsets not monotone: {w:?}");
        }

        // permuted_token_indices.len() == T*K
        assert_eq!(plan.permuted_token_indices.len(), t * k);
        // every entry in [0, T)
        for &v in &plan.permuted_token_indices {
            assert!((0..t as i32).contains(&v), "token idx out of range: {v}");
        }

        // Each source token appears exactly K times in permuted_token_indices.
        let mut hits = vec![0u32; t];
        for &v in &plan.permuted_token_indices {
            hits[v as usize] += 1;
        }
        for (ti, &h) in hits.iter().enumerate() {
            assert_eq!(h as usize, k, "token {ti} hit count {h} != K={k}");
        }
        Ok(())
    }

    #[test]
    fn token_choice_route_matches_pytorch_recipe() -> Result<()> {
        // Hand-computed reference: T=2, E=4, K=2.
        // Token 0 logits: [0.0, 1.0, 0.5, -1.0]   → top-2 = [(1.0, e=1), (0.5, e=2)]
        //   softmax([1.0, 0.5]) = exp1/(exp1+exp0.5), exp0.5/(exp1+exp0.5)
        //                       ≈ [0.62246, 0.37754]
        // Token 1 logits: [2.0, 0.0, -1.0, 1.0]   → top-2 = [(2.0, e=0), (1.0, e=3)]
        //   softmax([2.0, 1.0]) ≈ [0.73106, 0.26894]
        let device = CudaDevice::new(0).map_err(|e| Error::Cuda(format!("{e:?}")))?;
        let t = 2usize;
        let e = 4usize;
        let k = 2usize;

        let logits = vec![
            0.0_f32, 1.0, 0.5, -1.0, //
            2.0, 0.0, -1.0, 1.0,
        ];
        let logits_t = Tensor::from_vec_dtype(
            logits,
            Shape::from_dims(&[t, e]),
            device.clone(),
            DType::F32,
        )?;
        let plan = token_choice_route(&logits_t, k, ScoreMode::TopKSoftmax)?;

        let idx = plan.expert_indices.to_dtype(DType::F32)?.to_vec()?;
        let w = plan.expert_weights.to_vec()?;

        // Token 0: top experts are 1, 2 (in that order).
        assert_eq!(idx[0] as i32, 1);
        assert_eq!(idx[1] as i32, 2);
        assert!((w[0] - 0.62245935_f32).abs() < 1e-5, "got {}", w[0]);
        assert!((w[1] - 0.37754065_f32).abs() < 1e-5, "got {}", w[1]);

        // Token 1: top experts are 0, 3 (in that order).
        assert_eq!(idx[2] as i32, 0);
        assert_eq!(idx[3] as i32, 3);
        assert!((w[2] - 0.7310586_f32).abs() < 1e-5, "got {}", w[2]);
        assert!((w[3] - 0.26894143_f32).abs() < 1e-5, "got {}", w[3]);

        // Offsets:
        //   expert 0: 1 pick (token 1's top)
        //   expert 1: 1 pick (token 0's top)
        //   expert 2: 1 pick (token 0's second)
        //   expert 3: 1 pick (token 1's second)
        // -> counts = [1, 1, 1, 1], offsets = [0, 1, 2, 3, 4].
        assert_eq!(plan.offsets, vec![0, 1, 2, 3, 4]);

        // permuted_token_indices, expert-major:
        //   expert 0: [1]
        //   expert 1: [0]
        //   expert 2: [0]
        //   expert 3: [1]
        assert_eq!(plan.permuted_token_indices, vec![1, 0, 0, 1]);
        // expert_weights_flat aligned:
        //   expert 0: token 1 picked it as top -> w = 0.73106
        //   expert 1: token 0 picked it as top -> w = 0.62246
        //   expert 2: token 0 picked it as second -> w = 0.37754
        //   expert 3: token 1 picked it as second -> w = 0.26894
        let expected = [0.7310586_f32, 0.62245935, 0.37754065, 0.26894143];
        for (got, exp) in plan.expert_weights_flat.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-5, "got {got} exp {exp}");
        }
        Ok(())
    }

    #[test]
    fn permute_tokens_for_token_choice_basic() -> Result<()> {
        // T=4, D=2, E=4, K=2 — reuse the plan from plan_structure.
        let device = CudaDevice::new(0).map_err(|e| Error::Cuda(format!("{e:?}")))?;
        let t = 4usize;
        let e = 4usize;
        let k = 2usize;
        let d = 2usize;

        let mut logits = vec![-1.0f32; t * e];
        for ti in 0..t {
            logits[ti * e + ti] = 2.0;
            logits[ti * e + (ti + 1) % e] = 1.0;
        }
        let logits_t = Tensor::from_vec_dtype(
            logits,
            Shape::from_dims(&[t, e]),
            device.clone(),
            DType::F32,
        )?;
        let plan = token_choice_route(&logits_t, k, ScoreMode::TopKSoftmax)?;

        // x[i] = [i*10, i*10+1]
        let x_data: Vec<f32> = (0..t * d)
            .map(|i| (i / d) as f32 * 10.0 + (i % d) as f32)
            .collect();
        let x = Tensor::from_vec_dtype(
            x_data.clone(),
            Shape::from_dims(&[t, d]),
            device.clone(),
            DType::F32,
        )?;
        let x_perm = permute_tokens_for_token_choice(&x, &plan)?;
        let got = x_perm.to_vec()?;
        // permuted_token_indices = [0, 3, 0, 1, 1, 2, 2, 3]
        let expected_sources = [0, 3, 0, 1, 1, 2, 2, 3];
        assert_eq!(got.len(), t * k * d);
        for (row, &src) in expected_sources.iter().enumerate() {
            for col in 0..d {
                let v = got[row * d + col];
                let expected = src as f32 * 10.0 + col as f32;
                assert!(
                    (v - expected).abs() < 1e-5,
                    "permute mismatch at row={row} col={col} src={src}: got {v} expected {expected}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn token_choice_route_top_k_equal_one() -> Result<()> {
        // K=1 corner case: weights are all 1.0.
        let device = CudaDevice::new(0).map_err(|e| Error::Cuda(format!("{e:?}")))?;
        let t = 3usize;
        let e = 4usize;
        let logits = vec![
            0.0_f32, 1.0, 0.5, -1.0, //
            2.0, 0.0, -1.0, 1.0, //
            -0.5, -0.7, 0.3, 0.2,
        ];
        let logits_t = Tensor::from_vec_dtype(
            logits,
            Shape::from_dims(&[t, e]),
            device,
            DType::F32,
        )?;
        let plan = token_choice_route(&logits_t, 1, ScoreMode::TopKSoftmax)?;
        let w = plan.expert_weights.to_vec()?;
        for &v in &w {
            assert!((v - 1.0).abs() < 1e-6);
        }
        let idx = plan.expert_indices.to_dtype(DType::F32)?.to_vec()?;
        assert_eq!(idx[0] as i32, 1); // token 0 top = e1
        assert_eq!(idx[1] as i32, 0); // token 1 top = e0
        assert_eq!(idx[2] as i32, 2); // token 2 top = e2
        Ok(())
    }

    #[test]
    fn token_choice_route_top_k_equal_e_softmax_renorm_matches_full_softmax() -> Result<()> {
        // K = E + SoftmaxRenorm: top-K covers all experts, the renormalised
        // weights collapse to the full E-softmax distribution.
        let device = CudaDevice::new(0).map_err(|e| Error::Cuda(format!("{e:?}")))?;
        let t = 2usize;
        let e = 4usize;
        let logits = vec![0.0_f32, 1.0, 0.5, -1.0, 2.0, 0.0, -1.0, 1.0];
        let logits_t = Tensor::from_vec_dtype(
            logits.clone(),
            Shape::from_dims(&[t, e]),
            device.clone(),
            DType::F32,
        )?;
        let plan = token_choice_route(&logits_t, e, ScoreMode::SoftmaxRenorm)?;
        let w_row_t = plan.expert_weights.to_vec()?;
        // expert_indices is per-token DESCENDING by score; pull the
        // permutation back to expert-natural order before comparing.
        let idx = plan.expert_indices.to_dtype(DType::F32)?.to_vec()?;

        for ti in 0..t {
            // Build natural-order weights for this token.
            let mut natural = vec![0.0_f32; e];
            for k in 0..e {
                let exp_id = idx[ti * e + k] as usize;
                natural[exp_id] = w_row_t[ti * e + k];
            }
            // Compute full softmax in F32 for comparison.
            let row = &logits[ti * e..(ti + 1) * e];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sumexp = 0.0;
            let mut probs = vec![0.0; e];
            for ei in 0..e {
                let v = (row[ei] - m).exp();
                probs[ei] = v;
                sumexp += v;
            }
            for v in probs.iter_mut() {
                *v /= sumexp;
            }
            for ei in 0..e {
                assert!(
                    (natural[ei] - probs[ei]).abs() < 1e-5,
                    "token {ti} expert {ei}: got {} expected {}",
                    natural[ei],
                    probs[ei]
                );
            }
        }
        Ok(())
    }
}
