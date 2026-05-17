//! Nucleus-Image MoE expert FFN composite (Phase 4 of the MoE kernel plan).
//!
//! Chains the Phase 1-3 primitives into a single SwiGLU MoE expert forward:
//!
//! 1. Route via `expert_choice_route` → `(offsets, indices, gating_flat)`
//! 2. Permute tokens to expert-major via `permute_tokens`
//! 3. `grouped_mm_bf16(x_perm, gate_up_w)` → `(E*B*C, 2*inter)` BF16
//! 4. SwiGLU: `silu(gate) * up` via `Tensor::swiglu` → `(E*B*C, inter)` BF16
//! 5. `grouped_mm_bf16(act, down_w)` → `(E*B*C, D)` BF16
//! 6. `fused_gated_scatter_add_bf16` weighted unpermute → `(B*S, D)` F32
//! 7. Cast back to BF16
//!
//! Mirrors `SwiGLUExperts._run_experts_grouped_mm` + the surrounding
//! routing in `NucleusMoELayer.forward`
//! (`diffusers/models/transformers/transformer_nucleusmoe_image.py`).
//!
//! Out of scope here:
//! - Router (`x @ gate_w`, softmax, transpose) — caller produces the
//!   `affinity` tensor and passes it in. Lets the test exercise the kernel
//!   chain on hand-crafted affinity values for parity rigor.
//! - Shared expert FFN (`shared_expert(hidden_states)`). The Nucleus
//!   reference adds it post-MoE; equivalent here to `y_moe + shared(x)`,
//!   so we leave the addition to the caller.
//! - Modulation. `x_flat` is the *already-modulated* hidden state. The
//!   *unmodulated* hidden state feeds the router — the caller does the
//!   split.

use crate::ops::fused_gated_scatter_add::fused_gated_scatter_add_bf16;
use crate::ops::grouped_mm::grouped_mm_bf16;
use crate::ops::moe_routing::{expert_choice_route, permute_tokens};
use crate::{DType, Error, Result, Shape, Tensor};

/// SwiGLU MoE expert forward.
///
/// # Arguments
/// - `x_flat`: `(B*S, D)` BF16 — modulated hidden states, flattened from `(B, S, D)`.
/// - `affinity`: `(B, E, S)` F32 — the post-`softmax(router_logits).transpose(1,2)`
///   tensor. Caller is responsible for the matmul + softmax.
/// - `gate_up_w`: `(E, D, 2*inter)` BF16 — stacked gate-up projections.
///   The first `inter` output cols of each expert's matmul become `gate`,
///   the next `inter` become `up`.
/// - `down_w`: `(E, inter, D)` BF16 — stacked down projections.
/// - `capacity`: per-expert capacity `C` for expert-choice routing. Caller
///   computes from `ceil(capacity_factor * S / E)`.
/// - `route_scale`: gating multiplier (Nucleus uses 2.5).
///
/// # Returns
/// `(B*S, D)` BF16 — sum of weighted expert outputs over picks. Tokens
/// not picked by any expert get zero contribution from this function and
/// must receive their value from the shared expert that the caller adds.
pub fn nucleus_moe_expert_forward(
    x_flat: &Tensor,
    affinity: &Tensor,
    gate_up_w: &Tensor,
    down_w: &Tensor,
    capacity: usize,
    route_scale: f32,
) -> Result<Tensor> {
    // ---- shape sanity --------------------------------------------------
    if x_flat.dtype() != DType::BF16 {
        return Err(Error::InvalidOperation(format!(
            "nucleus_moe_expert_forward: x_flat must be BF16, got {:?}",
            x_flat.dtype()
        )));
    }
    if gate_up_w.dtype() != DType::BF16 {
        return Err(Error::InvalidOperation(format!(
            "nucleus_moe_expert_forward: gate_up_w must be BF16, got {:?}",
            gate_up_w.dtype()
        )));
    }
    if down_w.dtype() != DType::BF16 {
        return Err(Error::InvalidOperation(format!(
            "nucleus_moe_expert_forward: down_w must be BF16, got {:?}",
            down_w.dtype()
        )));
    }
    let xd = x_flat.shape().dims();
    if xd.len() != 2 {
        return Err(Error::InvalidOperation(format!(
            "nucleus_moe_expert_forward: x_flat must be 2-D (B*S, D), got {xd:?}"
        )));
    }
    let d = xd[1];
    let gud = gate_up_w.shape().dims();
    if gud.len() != 3 || gud[1] != d {
        return Err(Error::InvalidOperation(format!(
            "nucleus_moe_expert_forward: gate_up_w must be (E, D={d}, 2*inter), got {gud:?}"
        )));
    }
    let two_inter = gud[2];
    if two_inter % 2 != 0 {
        return Err(Error::InvalidOperation(format!(
            "nucleus_moe_expert_forward: gate_up_w last dim {two_inter} must be even (= 2*inter)"
        )));
    }
    let inter = two_inter / 2;
    let dwd = down_w.shape().dims();
    if dwd.len() != 3 || dwd[0] != gud[0] || dwd[1] != inter || dwd[2] != d {
        return Err(Error::InvalidOperation(format!(
            "nucleus_moe_expert_forward: down_w must be (E={}, inter={inter}, D={d}), got {dwd:?}",
            gud[0]
        )));
    }

    let device = x_flat.device().clone();

    // ---- 1. Routing plan (host-side top-K + renorm) -------------------
    let plan = expert_choice_route(affinity, capacity, route_scale)?;
    let t_max = plan.batch_size * plan.capacity; // B*C uniform per expert

    // ---- 2. Permute tokens to expert-major ----------------------------
    let x_perm = permute_tokens(x_flat, &plan)?; // (E*B*C, D) BF16

    // ---- 3. Grouped GEMM: gate-up projection --------------------------
    let gate_up = grouped_mm_bf16(&x_perm, gate_up_w, &plan.offsets, t_max)?; // (E*B*C, 2*inter) BF16

    // ---- 4. SwiGLU split + activation ---------------------------------
    let gate = gate_up.narrow(1, 0, inter)?;
    let up = gate_up.narrow(1, inter, inter)?;
    let act = gate.swiglu(&up)?; // (E*B*C, inter) BF16

    // ---- 5. Grouped GEMM: down projection -----------------------------
    let down = grouped_mm_bf16(&act, down_w, &plan.offsets, t_max)?; // (E*B*C, D) BF16

    // ---- 6. Weighted scatter-add unpermute ----------------------------
    let n_picks = plan.global_token_indices.len();
    let gating_t = Tensor::from_vec_dtype(
        plan.gating_flat.clone(),
        Shape::from_dims(&[n_picks]),
        device.clone(),
        DType::F32,
    )?;
    let total_tokens = plan.batch_size * plan.seq_len;
    let mut accum = Tensor::zeros_dtype(
        Shape::from_dims(&[total_tokens, d]),
        DType::F32,
        device.clone(),
    )?;
    fused_gated_scatter_add_bf16(&down, &gating_t, &plan.global_token_indices, &mut accum)?;

    // ---- 7. Cast back to BF16 -----------------------------------------
    accum.to_dtype(DType::BF16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaDevice;

    /// Hand-rolled scalar Rust reference for the full SwiGLU MoE expert forward.
    /// Mirrors the diffusers recipe step for step on host f32 math, with BF16
    /// rounding applied at every cast site to match the GPU pipeline.
    #[allow(clippy::too_many_arguments)]
    fn scalar_ref(
        x_flat_bf16: &[f32],  // (B*S, D), pre-rounded to BF16
        affinity: &[f32],     // (B, E, S) F32
        gate_up_bf16: &[f32], // (E, D, 2*inter), pre-rounded BF16
        down_bf16: &[f32],    // (E, inter, D), pre-rounded BF16
        b: usize,
        e: usize,
        s: usize,
        d: usize,
        inter: usize,
        capacity: usize,
        route_scale: f32,
    ) -> Vec<f32> {
        // Top-C per (batch, expert) — same as expert_choice_route.
        // top_idx: (B, E, C) and top_w: (B, E, C).
        let mut top_idx = vec![0i32; b * e * capacity];
        let mut top_w = vec![0.0f32; b * e * capacity];
        let mut row: Vec<(f32, i32)> = Vec::with_capacity(s);
        for bi in 0..b {
            for ei in 0..e {
                row.clear();
                for si in 0..s {
                    row.push((affinity[(bi * e + ei) * s + si], si as i32));
                }
                row.sort_by(|a, b| {
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

        // Flatten to expert-major (E, B, C).
        let n_picks = e * b * capacity;
        let mut global_idx = vec![0i32; n_picks];
        let mut gating = vec![0.0f32; n_picks];
        for ei in 0..e {
            for bi in 0..b {
                let src = (bi * e + ei) * capacity;
                let dst = (ei * b + bi) * capacity;
                let batch_off = (bi * s) as i32;
                for k in 0..capacity {
                    global_idx[dst + k] = batch_off + top_idx[src + k];
                    gating[dst + k] = top_w[src + k];
                }
            }
        }
        // Renormalise + scale.
        let total_tokens = b * s;
        let mut tsum = vec![0.0f32; total_tokens];
        for k in 0..n_picks {
            let i = global_idx[k] as usize;
            if i < total_tokens {
                tsum[i] += gating[k];
            }
        }
        for k in 0..n_picks {
            let i = global_idx[k] as usize;
            gating[k] = (gating[k] / (tsum[i] + 1e-12)) * route_scale;
        }

        // Permute tokens.
        let mut x_perm = vec![0.0f32; n_picks * d];
        for k in 0..n_picks {
            let src = (global_idx[k] as usize) * d;
            for di in 0..d {
                x_perm[k * d + di] = x_flat_bf16[src + di];
            }
        }

        // Grouped GEMM gate-up. x_perm is expert-major: each block of B*C
        // rows belongs to one expert. Then BF16-round the output.
        let bc = b * capacity;
        let mut gate_up = vec![0.0f32; n_picks * (2 * inter)];
        for ei in 0..e {
            for ti in 0..bc {
                let row = ei * bc + ti;
                for ni in 0..(2 * inter) {
                    let mut acc = 0.0f32;
                    for ki in 0..d {
                        acc += x_perm[row * d + ki]
                            * gate_up_bf16[ei * d * (2 * inter) + ki * (2 * inter) + ni];
                    }
                    gate_up[row * (2 * inter) + ni] = bf16_round(acc);
                }
            }
        }

        // SwiGLU: act[r, i] = silu(gate_up[r, i]) * gate_up[r, inter + i].
        // Match Tensor::swiglu's exact convention: FP32 sigmoid math with a
        // BF16 round on `silu(gate)` between sigmoid and the multiply.
        let mut act = vec![0.0f32; n_picks * inter];
        for r in 0..n_picks {
            for i in 0..inter {
                let g = gate_up[r * (2 * inter) + i];
                let u = gate_up[r * (2 * inter) + inter + i];
                let s = g * sigmoid_f32(g); // silu(g) in f32
                let s_bf16 = bf16_round(s); // intermediate BF16 round
                act[r * inter + i] = bf16_round(s_bf16 * u);
            }
        }

        // Grouped GEMM down.
        let mut down = vec![0.0f32; n_picks * d];
        for ei in 0..e {
            for ti in 0..bc {
                let row = ei * bc + ti;
                for di in 0..d {
                    let mut acc = 0.0f32;
                    for ki in 0..inter {
                        acc += act[row * inter + ki] * down_bf16[ei * inter * d + ki * d + di];
                    }
                    down[row * d + di] = bf16_round(acc);
                }
            }
        }

        // Scatter-add weighted: accum (F32) collects, then BF16-round at end.
        let mut accum = vec![0.0f32; total_tokens * d];
        for k in 0..n_picks {
            let dst = (global_idx[k] as usize) * d;
            let g = gating[k];
            for di in 0..d {
                accum[dst + di] += down[k * d + di] * g;
            }
        }
        for v in accum.iter_mut() {
            *v = bf16_round(*v);
        }
        accum
    }

    #[inline]
    fn bf16_round(v: f32) -> f32 {
        // Round-to-nearest-even BF16 truncation. Matches what flame-core's
        // bf16_round does internally for the swiglu reference.
        let bits = v.to_bits();
        let lsb = (bits >> 16) & 1;
        let bias = 0x7FFF + lsb;
        let rounded = bits.saturating_add(bias) & 0xFFFF_0000;
        f32::from_bits(rounded)
    }

    #[inline]
    fn sigmoid_f32(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// PyTorch parity test (Phase 5). Loads the fixture generated by
    /// `scripts/generate_nucleus_moe_parity.py` (which runs the same MoE
    /// expert forward through `F.grouped_mm` + `F.silu` on real CUDA),
    /// runs flame-core's composite on the identical weights, and asserts
    /// agreement within BF16 tolerance.
    ///
    /// Run via `cargo test --release --lib nucleus_moe_parity_vs_pytorch`.
    /// If the fixture is missing the test soft-fails with a clear
    /// instruction (so a fresh checkout doesn't fail this hard before
    /// regenerating).
    #[test]
    fn nucleus_moe_parity_vs_pytorch() -> Result<()> {
        use std::path::Path;
        let fixture_path = Path::new("tests/pytorch_fixtures/moe/nucleus_moe_parity.safetensors");
        if !fixture_path.exists() {
            eprintln!(
                "[nucleus_moe_parity_vs_pytorch] fixture not found at {:?}; \
                 regenerate with `python3 scripts/generate_nucleus_moe_parity.py`. \
                 Skipping.",
                fixture_path
            );
            return Ok(());
        }

        let device = CudaDevice::new(0).map_err(|e| Error::Cuda(format!("{e:?}")))?;
        let fix = crate::serialization::load_file(fixture_path, &device)?;

        let x_flat = fix.get("x_flat").expect("fixture missing x_flat").clone();
        let affinity = fix
            .get("affinity")
            .expect("fixture missing affinity")
            .clone();
        let gate_up_w = fix
            .get("gate_up_w")
            .expect("fixture missing gate_up_w")
            .clone();
        let down_w = fix.get("down_w").expect("fixture missing down_w").clone();
        let expected = fix
            .get("expected_y")
            .expect("fixture missing expected_y")
            .clone();

        // Pull capacity + route_scale out of 0-D metadata tensors. They
        // were saved as int32 / float32 so to_vec() gives us f32 with the
        // right value (route_scale is exact; capacity is a small positive
        // integer that round-trips losslessly through f32).
        let capacity_v = fix
            .get("meta_capacity")
            .expect("fixture missing meta_capacity")
            .to_vec()?;
        let route_scale_v = fix
            .get("meta_route_scale")
            .expect("fixture missing meta_route_scale")
            .to_vec()?;
        let capacity = capacity_v[0] as usize;
        let route_scale = route_scale_v[0];

        let y = nucleus_moe_expert_forward(
            &x_flat,
            &affinity,
            &gate_up_w,
            &down_w,
            capacity,
            route_scale,
        )?;

        let got = y.to_vec()?;
        let exp = expected.to_vec()?;
        assert_eq!(got.len(), exp.len(), "shape mismatch");

        let mut max_abs: f32 = 0.0;
        let mut max_rel: f32 = 0.0;
        let mut mean_abs: f32 = 0.0;
        for (g, e) in got.iter().zip(exp.iter()) {
            let abs = (g - e).abs();
            mean_abs += abs;
            if abs > max_abs {
                max_abs = abs;
            }
            let rel = if e.abs() > 1e-4 { abs / e.abs() } else { 0.0 };
            if rel > max_rel {
                max_rel = rel;
            }
        }
        mean_abs /= got.len() as f32;
        eprintln!(
            "nucleus_moe_parity_vs_pytorch: \
             max_abs={max_abs:.2e} max_rel={max_rel:.2e} mean_abs={mean_abs:.2e}"
        );

        // The two stacks share `F.grouped_mm` semantics (we mirror it
        // bit-for-bit per the kernel doc) and the same SwiGLU formula. The
        // residual error comes from BF16 quantisation differences between
        // the WMMA accumulator path and PyTorch's grouped_mm dispatch
        // (different SM-specific code paths, different tile schedules).
        // For values clustered near ±1e-3 the absolute diff per element
        // is dominated by 1-ULP BF16 rounding (~4e-6 at this magnitude).
        // We allow a generous 1e-3 absolute floor with a tighter mean
        // gate.
        assert!(
            max_abs < 1e-3,
            "PyTorch parity max_abs={max_abs} exceeds tolerance"
        );
        assert!(
            mean_abs < 1e-4,
            "PyTorch parity mean_abs={mean_abs} exceeds tolerance"
        );
        Ok(())
    }

    #[test]
    fn nucleus_moe_expert_forward_matches_scalar_ref() -> Result<()> {
        let device = CudaDevice::new(0).map_err(|e| Error::Cuda(format!("{e:?}")))?;

        // Toy shapes — small enough to be fast, large enough that BF16
        // accumulation stays sane.
        let b = 1usize;
        let e = 4usize;
        let s = 8usize;
        let d = 64usize; // hidden — multiple of 16 for WMMA
        let inter = 64usize; // expert intermediate
        let capacity = 4usize; // each expert picks 4 of 8 tokens

        // Seed-deterministic input/weights, BF16-rounded so the reference
        // sees the same values the kernel will see.
        let mut rng_state = 12345u64;
        let mut next = || -> f32 {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = (rng_state >> 33) as u32;
            // Map u32 → roughly Uniform(-0.5, 0.5)
            (v as f32 / u32::MAX as f32) - 0.5
        };

        let x_data: Vec<f32> = (0..b * s * d).map(|_| bf16_round(next() * 0.5)).collect();
        let aff_data: Vec<f32> = (0..b * e * s).map(|_| next() * 2.0).collect(); // F32 affinity
        let gate_up_data: Vec<f32> = (0..e * d * (2 * inter))
            .map(|_| bf16_round(next() * 0.1))
            .collect();
        let down_data: Vec<f32> = (0..e * inter * d)
            .map(|_| bf16_round(next() * 0.1))
            .collect();

        let x = Tensor::from_vec_dtype(
            x_data.clone(),
            Shape::from_dims(&[b * s, d]),
            device.clone(),
            DType::BF16,
        )?;
        let aff = Tensor::from_vec_dtype(
            aff_data.clone(),
            Shape::from_dims(&[b, e, s]),
            device.clone(),
            DType::F32,
        )?;
        let gate_up_w = Tensor::from_vec_dtype(
            gate_up_data.clone(),
            Shape::from_dims(&[e, d, 2 * inter]),
            device.clone(),
            DType::BF16,
        )?;
        let down_w = Tensor::from_vec_dtype(
            down_data.clone(),
            Shape::from_dims(&[e, inter, d]),
            device.clone(),
            DType::BF16,
        )?;

        let route_scale = 2.5f32; // Nucleus default
        let y = nucleus_moe_expert_forward(&x, &aff, &gate_up_w, &down_w, capacity, route_scale)?;
        let got = y.to_vec()?;

        let expected = scalar_ref(
            &x_data,
            &aff_data,
            &gate_up_data,
            &down_data,
            b,
            e,
            s,
            d,
            inter,
            capacity,
            route_scale,
        );

        assert_eq!(got.len(), expected.len());

        // Tolerance: BF16 multiply-add accumulates errors roughly per the
        // square root of the chained matmul widths. Two grouped GEMMs (D
        // and inter wide) + a SwiGLU + a weighted accumulate. Empirical
        // tolerance for the toy at D=inter=64: a few percent absolute.
        let mut max_abs: f32 = 0.0;
        let mut max_rel: f32 = 0.0;
        for (g, r) in got.iter().zip(expected.iter()) {
            let abs = (g - r).abs();
            let rel = if r.abs() > 1e-3 { abs / r.abs() } else { 0.0 };
            if abs > max_abs {
                max_abs = abs;
            }
            if rel > max_rel {
                max_rel = rel;
            }
        }
        assert!(
            max_abs < 0.10,
            "nucleus_moe_expert_forward diverged: max_abs={max_abs} max_rel={max_rel}"
        );
        Ok(())
    }
}
