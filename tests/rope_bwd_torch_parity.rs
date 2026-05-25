//! RoPE backward parity vs PyTorch — interleaved + halfsplit layouts.
//!
//! Experiment A, test #5 from LOCALIZE_FINDINGS_2026-05-24.
//!
//! For each case from `tests/parity/rope_bwd_ref.json`, applies flame-core's
//! rope forward via the autograd-recording path and compares dX gradients.
//!
//! Z-Image uses Halfsplit layout; HiDream-O1 used Interleaved (previously
//! buggy due to shape-sniff at autograd.rs:4159, fixed with layout tag).
//! This test verifies both layouts at trainer shapes.
//!
//! PASS criterion: cos_sim ≥ 0.999 AND rel_L2 ≤ 1%.
//!
//! Run with:
//!   cargo test --test rope_bwd_torch_parity -- --test-threads=1 --nocapture

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::{global_cuda_device, AutogradContext, DType, Result, Shape, Tensor};
use std::path::PathBuf;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("parity")
        .join("rope_bwd_ref.json")
}

#[derive(serde::Deserialize)]
struct Case {
    layout: String,
    b: usize,
    h: usize,
    n: usize,
    d_head: usize,
    seed: u32,
    x_bf16_as_f32: Vec<f32>,      // flat [b, h, n, d_head]
    cos_bf16_as_f32: Vec<f32>,    // flat [1, 1, n, d_head/2]
    sin_bf16_as_f32: Vec<f32>,    // flat [1, 1, n, d_head/2]
    dx_f32: Vec<f32>,             // flat [b, h, n, d_head]
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
fn rope_backward_matches_pytorch() -> Result<()> {
    let fixture_path = fixture_path();
    let bytes = std::fs::read(&fixture_path).unwrap_or_else(|e| {
        panic!(
            "missing fixture {}: {} — regenerate via \
             `python3 tests/parity/rope_bwd_ref.py`",
            fixture_path.display(),
            e
        )
    });
    let fx: Fixture = serde_json::from_slice(&bytes).expect("malformed rope_bwd_ref.json");

    let device = global_cuda_device();
    let mut any_failed = false;
    let mut report = String::new();

    let cos_thresh = 0.999;
    let rel_l2_thresh = 0.01;

    for (ci, case) in fx.cases.iter().enumerate() {
        let (b, h, n, d) = (case.b, case.h, case.n, case.d_head);
        let half = d / 2;
        let tag = format!(
            "case#{ci} {} ({b},{h},{n},{d}) seed={}",
            case.layout, case.seed
        );

        assert_eq!(case.x_bf16_as_f32.len(), b * h * n * d, "{tag}: x len");
        assert_eq!(case.cos_bf16_as_f32.len(), n * half, "{tag}: cos len");
        assert_eq!(case.sin_bf16_as_f32.len(), n * half, "{tag}: sin len");
        assert_eq!(case.dx_f32.len(), b * h * n * d, "{tag}: dx len");

        // x [b, h, n, d_head] BF16, requires_grad
        let x = Tensor::from_vec(
            case.x_bf16_as_f32.clone(),
            Shape::from_dims(&[b, h, n, d]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);

        // cos/sin [1, 1, n, half] BF16 (no grad)
        let cos = Tensor::from_vec(
            case.cos_bf16_as_f32.clone(),
            Shape::from_dims(&[1, 1, n, half]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;
        let sin = Tensor::from_vec(
            case.sin_bf16_as_f32.clone(),
            Shape::from_dims(&[1, 1, n, half]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;

        AutogradContext::clear();

        // Apply RoPE via the autograd-recording bf16_ops functions.
        // Layout dispatch matches autograd.rs Op::RoPePrecomputed backward.
        let y = match case.layout.as_str() {
            "halfsplit" => {
                // Z-Image / Klein path: RopeLayout::Halfsplit or HalfsplitPytorch.
                // Use rope_halfsplit_bf16 directly — this is what the model calls.
                // For the purpose of this parity test we use the functional path
                // that records Op::RoPePrecomputed with the correct layout tag.
                apply_rope_autograd_halfsplit(&x, &cos, &sin)?
            }
            "interleaved" => {
                apply_rope_autograd_interleaved(&x, &cos, &sin)?
            }
            other => panic!("unknown layout {other}"),
        };

        // Loss = sum(y.to_f32())
        let loss = y.to_dtype(DType::F32)?.sum()?;
        let grads = loss.backward()?;

        let grad_x = grads.get(x.id()).unwrap_or_else(|| {
            panic!("{tag}: no grad for x — autograd broken")
        });
        let got_dx: Vec<f32> = grad_x.to_dtype(DType::F32)?.to_vec()?;

        assert_eq!(got_dx.len(), case.dx_f32.len(), "{tag}: dx len");
        let (cos_v, rel) = cos_and_rel_l2(&got_dx, &case.dx_f32);
        let pass = cos_v >= cos_thresh && rel <= rel_l2_thresh;
        let s = if pass { "PASS" } else { "FAIL" };
        let line = format!("  {tag}: dx cos={:.6} rel_L2={:.4e} [{s}]", cos_v, rel);
        println!("{line}");
        if !pass {
            any_failed = true;
            report.push_str(&format!("{line} <-- BACKWARD BUG\n"));
        }
    }

    if any_failed {
        panic!(
            "RoPE backward diverges from PyTorch — \
             this is a candidate shared backward regression:\n{}",
            report
        );
    }

    Ok(())
}

/// Apply halfsplit RoPE with autograd recording (Op::RoPePrecomputed with
/// RopeLayout::Halfsplit). Calls the same function the model forward uses.
fn apply_rope_autograd_halfsplit(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Tensor> {
    // flame_core::bf16_ops::rope_halfsplit_bf16 records the op when autograd is active.
    // The autograd.rs backward handler reads layout from Op::RoPePrecomputed.layout.
    // This is the real Z-Image / Klein model path.
    flame_core::bf16_ops::rope_halfsplit_bf16(x, cos, sin)
}

fn apply_rope_autograd_interleaved(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Tensor> {
    flame_core::bf16_ops::rope_fused_bf16(x, cos, sin)
}
