//! Matmul backward parity — frozen rhs (requires_grad=false) + live lhs.
//!
//! Experiment A, CRITICAL GAP TEST from LOCALIZE_FINDINGS_2026-05-24.
//!
//! Tests Op::MatMul backward when rhs has requires_grad=False (frozen weight)
//! but lhs has requires_grad=True (activation flowing from LoRA).
//!
//! This exercises the `need_rhs_grad=false` branch introduced in commit 2143bcb:
//!   if need_lhs_grad { compute grad_lhs } else { skip }
//!   if need_rhs_grad { compute grad_rhs } else { skip }
//!
//! In a LoRA-over-frozen-base-model setup, frozen base linear weights have
//! requires_grad=False. The activation gradient MUST still be computed for
//! the backward chain to reach earlier LoRA parameters.
//!
//! PASS criterion: cos_sim ≥ 0.999 AND rel_L2 ≤ 1% for grad_lhs (dA).
//!
//! Run with:
//!   cargo test --test frozen_rhs_matmul_bwd_parity -- --test-threads=1 --nocapture

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::{global_cuda_device, AutogradContext, DType, Result, Shape, Tensor};
use std::path::PathBuf;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("parity")
        .join("frozen_rhs_matmul_bwd_ref.json")
}

#[derive(serde::Deserialize)]
struct Case {
    batch: usize,
    in_dim: usize,
    out_dim: usize,
    seed: u32,
    #[serde(rename = "A_f32")]
    a_f32: Vec<f32>,
    #[serde(rename = "B_f32")]
    b_f32: Vec<f32>,
    #[serde(rename = "dA_f32")]
    da_f32: Vec<f32>,
    // dB is None in PyTorch (frozen weight) — just checking dA here
}

#[derive(serde::Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

fn cos_and_rel_l2(got: &[f32], reference: &[f32]) -> (f64, f64) {
    assert_eq!(got.len(), reference.len());
    let mut dot = 0.0f64;
    let mut got_n2 = 0.0f64;
    let mut ref_n2 = 0.0f64;
    let mut diff_sq = 0.0f64;
    let mut ref_sq = 0.0f64;
    for (&a, &b) in got.iter().zip(reference.iter()) {
        let (a, b) = (a as f64, b as f64);
        dot += a * b;
        got_n2 += a * a;
        ref_n2 += b * b;
        diff_sq += (a - b) * (a - b);
        ref_sq += b * b;
    }
    let cos = dot / (got_n2.sqrt() * ref_n2.sqrt() + 1e-20);
    let rel_l2 = diff_sq.sqrt() / (ref_sq.sqrt() + 1e-20);
    (cos, rel_l2)
}

#[test]
fn frozen_rhs_matmul_backward_matches_pytorch() -> Result<()> {
    let fixture_path = fixture_path();
    let bytes = std::fs::read(&fixture_path).unwrap_or_else(|e| {
        panic!(
            "missing fixture {}: {} — regenerate via `python3 /tmp/test_frozen_rhs_matmul.py`",
            fixture_path.display(),
            e
        )
    });
    let fx: Fixture =
        serde_json::from_slice(&bytes).expect("malformed frozen_rhs_matmul_bwd_ref.json");

    let device = global_cuda_device();
    let mut any_failed = false;
    let mut report = String::new();

    let cos_thresh = 0.999;
    let rel_l2_thresh = 0.01;

    for (ci, case) in fx.cases.iter().enumerate() {
        let (b, i, o) = (case.batch, case.in_dim, case.out_dim);
        let tag = format!("case#{ci} ({b},{i},{o}) seed={}", case.seed);

        // A [batch, in_dim] BF16, requires_grad=TRUE (activation)
        let a_tensor = Tensor::from_vec(
            case.a_f32.clone(),
            Shape::from_dims(&[b, i]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);

        // B [out_dim, in_dim] BF16, requires_grad=FALSE (frozen weight)
        // Do NOT call requires_grad_(true) — this is the frozen path
        let b_tensor = Tensor::from_vec(
            case.b_f32.clone(),
            Shape::from_dims(&[o, i]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;
        // Verify B is truly not requiring grad
        assert!(!b_tensor.requires_grad(), "{tag}: B should have requires_grad=false");

        AutogradContext::clear();

        // Forward: out = A @ B^T [batch, out_dim]
        let b_t = b_tensor.transpose()?;
        // B.transpose() — since B.requires_grad=false, b_t.requires_grad=false too
        assert!(!b_t.requires_grad(), "{tag}: B^T should have requires_grad=false");

        let out = a_tensor.matmul(&b_t)?;
        // out.requires_grad should be TRUE because a_tensor.requires_grad=true
        assert!(out.requires_grad(), "{tag}: out should inherit requires_grad=true from A");

        let out_f32 = out.to_dtype(DType::F32)?;
        let loss = out_f32.sum()?;

        let grads = loss.backward()?;

        // Grad for A MUST be present — it's required for upstream backprop
        let grad_a = grads.get(a_tensor.id()).unwrap_or_else(|| {
            panic!(
                "{tag}: CRITICAL — no grad for A despite requires_grad=True! \
                 The requires_grad gate in Op::MatMul backward may be incorrectly \
                 dropping grad_lhs when rhs.requires_grad=False."
            )
        });

        // B grad should NOT be present (it's frozen)
        assert!(
            grads.get(b_tensor.id()).is_none(),
            "{tag}: B should NOT have gradient (frozen weight)"
        );

        let got_da: Vec<f32> = grad_a.to_dtype(DType::F32)?.to_vec()?;
        assert_eq!(got_da.len(), case.da_f32.len(), "{tag}: dA shape mismatch");

        let (cos_a, rel_a) = cos_and_rel_l2(&got_da, &case.da_f32);
        let pass_a = cos_a >= cos_thresh && rel_a <= rel_l2_thresh;
        let status_a = if pass_a { "PASS" } else { "FAIL" };

        let line_a = format!(
            "  {tag}: dA cos={:.6} rel_L2={:.4e} [{status_a}] (rhs frozen, lhs live)",
            cos_a, rel_a
        );
        println!("{line_a}");

        if !pass_a {
            any_failed = true;
            report.push_str(&format!("{line_a} <-- CONFIRMED BUG: 2143bcb frozen-weight gate\n"));
        }
    }

    if any_failed {
        panic!(
            "CONFIRMED BUG: Op::MatMul backward with frozen rhs diverges from PyTorch.\n\
             Commit 2143bcb introduced requires_grad gating that silently drops grad_lhs\n\
             when rhs.requires_grad=False — breaking backward through frozen base-model\n\
             weights in LoRA training. This is the shared regression in all 5 trainers.\n\
             {}", report
        );
    }

    println!("\n  All cases PASS. Op::MatMul backward with frozen rhs is CLEAN.");
    println!("  If all 7 Experiment A tests pass including this one, the bug is NOT");
    println!("  in the frozen-rhs gate but elsewhere (checkpoint backward, multi-step).\n");

    Ok(())
}
