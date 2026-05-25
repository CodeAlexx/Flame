//! SwiGLU and GateResidual backward parity vs PyTorch.
//!
//! Experiment A, test #6 from LOCALIZE_FINDINGS_2026-05-24.
//!
//! SwiGLU: out = silu(gate) * up
//!   backward: dgate = dup * silu'(gate), dup_out = grad_out * silu(gate)
//!
//! GateResidual: out = residual + gate[:,None,:] * x   (NO tanh — direct multiply)
//!   backward:
//!     dresidual = grad_out
//!     dx        = grad_out * gate[:,None,:]
//!     dgate     = (grad_out * x).sum(dim=1)
//!
//! PASS criterion: cos_sim ≥ 0.999 AND rel_L2 ≤ 1%.
//!
//! Run with:
//!   cargo test --test swiglu_gate_residual_bwd_torch_parity -- --test-threads=1 --nocapture

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::{global_cuda_device, AutogradContext, DType, Result, Shape, Tensor};
use std::path::PathBuf;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("parity")
        .join("swiglu_gate_residual_bwd_ref.json")
}

#[derive(serde::Deserialize)]
#[serde(tag = "op")]
enum Case {
    #[serde(rename = "swiglu")]
    SwiGLU {
        b: usize,
        n: usize,
        d: usize,
        seed: u32,
        gate_f32: Vec<f32>,
        up_f32: Vec<f32>,
        dgate_f32: Vec<f32>,
        dup_f32: Vec<f32>,
    },
    #[serde(rename = "gate_residual")]
    GateResidual {
        b: usize,
        n: usize,
        d: usize,
        seed: u32,
        residual_f32: Vec<f32>,
        gate_f32: Vec<f32>,
        x_f32: Vec<f32>,
        dresidual_f32: Vec<f32>,
        dgate_f32: Vec<f32>,
        dx_f32: Vec<f32>,
    },
}

#[derive(serde::Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

fn cos_and_rel_l2(got: &[f32], reference: &[f32]) -> (f64, f64) {
    assert_eq!(got.len(), reference.len(), "length mismatch");
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
fn swiglu_gate_residual_backward_matches_pytorch() -> Result<()> {
    let fixture_path = fixture_path();
    let bytes = std::fs::read(&fixture_path).unwrap_or_else(|e| {
        panic!(
            "missing fixture {}: {} — regenerate via \
             `python3 tests/parity/swiglu_gate_residual_bwd_ref.py`",
            fixture_path.display(),
            e
        )
    });
    let fx: Fixture =
        serde_json::from_slice(&bytes).expect("malformed swiglu_gate_residual_bwd_ref.json");

    let device = global_cuda_device();
    let mut any_failed = false;
    let mut report = String::new();

    for (ci, case) in fx.cases.iter().enumerate() {
        match case {
            Case::SwiGLU { b, n, d, seed, gate_f32, up_f32, dgate_f32, dup_f32 } => {
                let (b, n, d) = (*b, *n, *d);
                let tag = format!("case#{ci} swiglu ({b},{n},{d}) seed={seed}");

                assert_eq!(gate_f32.len(), b * n * d, "{tag}: gate len");
                assert_eq!(up_f32.len(), b * n * d, "{tag}: up len");

                // gate [b, n, d] BF16, requires_grad
                let gate = Tensor::from_vec(
                    gate_f32.clone(),
                    Shape::from_dims(&[b, n, d]),
                    device.clone(),
                )?
                .to_dtype(DType::BF16)?
                .requires_grad_(true);

                // up [b, n, d] BF16, requires_grad
                let up = Tensor::from_vec(
                    up_f32.clone(),
                    Shape::from_dims(&[b, n, d]),
                    device.clone(),
                )?
                .to_dtype(DType::BF16)?
                .requires_grad_(true);

                AutogradContext::clear();

                // Forward via the autograd-recording fused kernel
                let out = flame_core::bf16_ops::swiglu_fused_bf16(&gate, &up)?;
                // Loss = sum(out.to_f32())
                let loss = out.to_dtype(DType::F32)?.sum()?;
                let grads = loss.backward()?;

                let grad_gate = grads.get(gate.id()).unwrap_or_else(|| {
                    panic!("{tag}: no grad for gate — autograd broken")
                });
                let grad_up = grads.get(up.id()).unwrap_or_else(|| {
                    panic!("{tag}: no grad for up — autograd broken")
                });

                let got_dgate: Vec<f32> = grad_gate.to_dtype(DType::F32)?.to_vec()?;
                let got_dup: Vec<f32> = grad_up.to_dtype(DType::F32)?.to_vec()?;

                assert_eq!(got_dgate.len(), dgate_f32.len(), "{tag}: dgate len");
                assert_eq!(got_dup.len(), dup_f32.len(), "{tag}: dup len");

                check(&tag, "dgate", &got_dgate, dgate_f32, &mut any_failed, &mut report);
                check(&tag, "dup",   &got_dup,   dup_f32,   &mut any_failed, &mut report);
            }

            Case::GateResidual { b, n, d, seed, residual_f32, gate_f32, x_f32, dresidual_f32, dgate_f32, dx_f32 } => {
                let (b, n, d) = (*b, *n, *d);
                let tag = format!("case#{ci} gate_residual ({b},{n},{d}) seed={seed}");

                assert_eq!(residual_f32.len(), b * n * d, "{tag}: residual len");
                assert_eq!(gate_f32.len(), b * d, "{tag}: gate len");
                assert_eq!(x_f32.len(), b * n * d, "{tag}: x len");

                // residual [b, n, d] BF16, requires_grad
                let residual = Tensor::from_vec(
                    residual_f32.clone(),
                    Shape::from_dims(&[b, n, d]),
                    device.clone(),
                )?
                .to_dtype(DType::BF16)?
                .requires_grad_(true);

                // gate [b, d] BF16, requires_grad
                let gate = Tensor::from_vec(
                    gate_f32.clone(),
                    Shape::from_dims(&[b, d]),
                    device.clone(),
                )?
                .to_dtype(DType::BF16)?
                .requires_grad_(true);

                // x [b, n, d] BF16, requires_grad
                let x = Tensor::from_vec(
                    x_f32.clone(),
                    Shape::from_dims(&[b, n, d]),
                    device.clone(),
                )?
                .to_dtype(DType::BF16)?
                .requires_grad_(true);

                AutogradContext::clear();

                // Forward via the autograd-recording fused kernel
                // gate_residual_fused_bf16(residual, gate, x) → out = residual + gate[:,None,:] * x
                let out = flame_core::bf16_ops::gate_residual_fused_bf16(&residual, &gate, &x)?;
                // Loss = sum(out.to_f32())
                let loss = out.to_dtype(DType::F32)?.sum()?;
                let grads = loss.backward()?;

                let grad_residual = grads.get(residual.id()).unwrap_or_else(|| {
                    panic!("{tag}: no grad for residual — autograd broken")
                });
                let grad_gate = grads.get(gate.id()).unwrap_or_else(|| {
                    panic!("{tag}: no grad for gate — autograd broken")
                });
                let grad_x = grads.get(x.id()).unwrap_or_else(|| {
                    panic!("{tag}: no grad for x — autograd broken")
                });

                let got_dresidual: Vec<f32> = grad_residual.to_dtype(DType::F32)?.to_vec()?;
                let got_dgate:    Vec<f32> = grad_gate.to_dtype(DType::F32)?.to_vec()?;
                let got_dx:       Vec<f32> = grad_x.to_dtype(DType::F32)?.to_vec()?;

                assert_eq!(got_dresidual.len(), dresidual_f32.len(), "{tag}: dresidual len");
                assert_eq!(got_dgate.len(), dgate_f32.len(), "{tag}: dgate len");
                assert_eq!(got_dx.len(), dx_f32.len(), "{tag}: dx len");

                check(&tag, "dresidual", &got_dresidual, dresidual_f32, &mut any_failed, &mut report);
                check(&tag, "dgate",     &got_dgate,     dgate_f32,     &mut any_failed, &mut report);
                check(&tag, "dx",        &got_dx,        dx_f32,        &mut any_failed, &mut report);
            }
        }
    }

    if any_failed {
        panic!(
            "SwiGLU/GateResidual backward diverges from PyTorch — \
             candidate shared backward regression:\n{}",
            report
        );
    }

    Ok(())
}
