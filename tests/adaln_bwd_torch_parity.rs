//! adaLN modulation backward parity vs PyTorch.
//!
//! Experiment A, test #7 from LOCALIZE_FINDINGS_2026-05-24.
//!
//! adaLN: out = x * (1 + scale[:, None, :]) + shift[:, None, :]
//! where x: [B, N, D] BF16, scale/shift: [B, D] BF16 (broadcast over N)
//!
//! This is the adaLN modulation pattern used in Z-Image and Klein:
//!   x_mod = rms_norm(x)
//!   out   = x_mod * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
//!
//! Tests: Op::Mul + Op::Add + Op::AddScalar + Op::Broadcast backward chain.
//!
//! PASS criterion: cos_sim ≥ 0.999 AND rel_L2 ≤ 1%.
//!
//! Run with:
//!   cargo test --test adaln_bwd_torch_parity -- --test-threads=1 --nocapture

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::{global_cuda_device, AutogradContext, DType, Result, Shape, Tensor};
use std::path::PathBuf;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("parity")
        .join("adaln_bwd_ref.json")
}

#[derive(serde::Deserialize)]
struct Case {
    b: usize,
    n: usize,
    d: usize,
    seed: u32,
    x_f32: Vec<f32>,
    scale_f32: Vec<f32>,
    shift_f32: Vec<f32>,
    dx_f32: Vec<f32>,
    dscale_f32: Vec<f32>,
    dshift_f32: Vec<f32>,
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

fn check(tag: &str, name: &str, got: &[f32], reference: &[f32], any_failed: &mut bool, report: &mut String) {
    let (cos_v, rel) = cos_and_rel_l2(got, reference);
    let pass = cos_v >= 0.999 && rel <= 0.01;
    let s = if pass { "PASS" } else { "FAIL" };
    let line = format!("  {tag}: {name} cos={:.6} rel_L2={:.4e} [{s}]", cos_v, rel);
    println!("{line}");
    if !pass {
        *any_failed = true;
        report.push_str(&format!("{line} <-- BACKWARD BUG\n"));
    }
}

#[test]
fn adaln_backward_matches_pytorch() -> Result<()> {
    let fixture_path = fixture_path();
    let bytes = std::fs::read(&fixture_path).unwrap_or_else(|e| {
        panic!(
            "missing fixture {}: {} — regenerate via `python3 /tmp/test_adaln_bwd.py`",
            fixture_path.display(),
            e
        )
    });
    let fx: Fixture = serde_json::from_slice(&bytes).expect("malformed adaln_bwd_ref.json");

    let device = global_cuda_device();
    let mut any_failed = false;
    let mut report = String::new();

    for (ci, case) in fx.cases.iter().enumerate() {
        let (b, n, d) = (case.b, case.n, case.d);
        let tag = format!("case#{ci} adaln ({b},{n},{d}) seed={}", case.seed);

        assert_eq!(case.x_f32.len(), b * n * d);
        assert_eq!(case.scale_f32.len(), b * d);
        assert_eq!(case.shift_f32.len(), b * d);

        // x [b, n, d] BF16, requires_grad
        let x = Tensor::from_vec(
            case.x_f32.clone(),
            Shape::from_dims(&[b, n, d]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);

        // scale [b, d] BF16, requires_grad
        let scale = Tensor::from_vec(
            case.scale_f32.clone(),
            Shape::from_dims(&[b, d]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);

        // shift [b, d] BF16, requires_grad
        let shift = Tensor::from_vec(
            case.shift_f32.clone(),
            Shape::from_dims(&[b, d]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);

        AutogradContext::clear();

        // Replicate adaLN: out = x * (1 + scale[:, None, :]) + shift[:, None, :]
        // scale: [b, d] → [b, 1, d] via unsqueeze/reshape, then broadcast
        // shift: [b, d] → [b, 1, d] via unsqueeze/reshape, then broadcast
        let scale_3d = scale.reshape(&[b, 1, d])?;   // [b, 1, d]
        let shift_3d = shift.reshape(&[b, 1, d])?;   // [b, 1, d]

        // (1 + scale_3d) via add_scalar
        let one_plus_scale = scale_3d.add_scalar(1.0)?;   // [b, 1, d]
        // Broadcast to [b, n, d]
        let one_plus_scale_bcast = one_plus_scale.broadcast_to(&Shape::from_dims(&[b, n, d]))?;
        let shift_bcast = shift_3d.broadcast_to(&Shape::from_dims(&[b, n, d]))?;

        // out = x * (1 + scale) + shift
        let modulated = x.mul(&one_plus_scale_bcast)?;
        let out = modulated.add(&shift_bcast)?;

        // Loss: sum(out.to_f32())
        let loss = out.to_dtype(DType::F32)?.sum()?;
        let grads = loss.backward()?;

        let grad_x = grads.get(x.id()).unwrap_or_else(|| panic!("{tag}: no grad for x"));
        let grad_scale = grads.get(scale.id()).unwrap_or_else(|| panic!("{tag}: no grad for scale"));
        let grad_shift = grads.get(shift.id()).unwrap_or_else(|| panic!("{tag}: no grad for shift"));

        let got_dx: Vec<f32> = grad_x.to_dtype(DType::F32)?.to_vec()?;
        let got_dscale: Vec<f32> = grad_scale.to_dtype(DType::F32)?.to_vec()?;
        let got_dshift: Vec<f32> = grad_shift.to_dtype(DType::F32)?.to_vec()?;

        assert_eq!(got_dx.len(), case.dx_f32.len());
        assert_eq!(got_dscale.len(), case.dscale_f32.len());
        assert_eq!(got_dshift.len(), case.dshift_f32.len());

        check(&tag, "dx",     &got_dx,     &case.dx_f32,     &mut any_failed, &mut report);
        check(&tag, "dscale", &got_dscale, &case.dscale_f32, &mut any_failed, &mut report);
        check(&tag, "dshift", &got_dshift, &case.dshift_f32, &mut any_failed, &mut report);
    }

    if any_failed {
        panic!(
            "adaLN backward diverges from PyTorch:\n{}",
            report
        );
    }

    Ok(())
}
