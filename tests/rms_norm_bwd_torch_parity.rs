//! RMSNorm backward parity vs PyTorch — trainer shapes.
//!
//! Experiment A, test #3 from LOCALIZE_FINDINGS_2026-05-24.
//!
//! For each (batch, norm_size) shape from `tests/parity/rms_norm_bwd_ref.json`,
//! runs RMSNorm forward via flame-core's `norm::rms_norm` and compares the
//! resulting gradients to PyTorch's `torch.autograd.grad` output.
//!
//! Tests both no-affine and with-affine weight variants.
//!
//! PASS criterion: cos_sim ≥ 0.999 AND rel_L2 ≤ 1%.
//!
//! Run with:
//!   cargo test --test rms_norm_bwd_torch_parity -- --test-threads=1 --nocapture

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::{global_cuda_device, AutogradContext, DType, Result, Shape, Tensor};
use std::path::PathBuf;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("parity")
        .join("rms_norm_bwd_ref.json")
}

#[derive(serde::Deserialize)]
struct Case {
    batch: usize,
    norm_size: usize,
    with_weight: bool,
    seed: u32,
    x_bf16_as_f32: Vec<Vec<f32>>,
    dx_f32: Vec<Vec<f32>>,
    w_bf16_as_f32: Option<Vec<f32>>,
    dw_f32: Option<Vec<f32>>,
}

#[derive(serde::Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

fn flat2(v: &[Vec<f32>]) -> Vec<f32> {
    v.iter().flatten().copied().collect()
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
    let rel_l2 = (diff_sq.sqrt()) / (ref_sq.sqrt() + 1e-20);
    (cos, rel_l2)
}

#[test]
fn rms_norm_backward_matches_pytorch() -> Result<()> {
    let fixture_path = fixture_path();
    let bytes = std::fs::read(&fixture_path).unwrap_or_else(|e| {
        panic!(
            "missing fixture {}: {} — regenerate via \
             `python3 tests/parity/rms_norm_bwd_ref.py`",
            fixture_path.display(),
            e
        )
    });
    let fx: Fixture = serde_json::from_slice(&bytes).expect("malformed rms_norm_bwd_ref.json");

    let device = global_cuda_device();
    let mut any_failed = false;
    let mut report = String::new();

    let cos_thresh = 0.999;
    let rel_l2_thresh = 0.01;

    for (ci, case) in fx.cases.iter().enumerate() {
        let (batch, ns) = (case.batch, case.norm_size);
        let w_tag = if case.with_weight { "w+" } else { "no-w" };
        let tag = format!("case#{ci} ({batch},{ns}) {w_tag} seed={}", case.seed);

        let x_flat = flat2(&case.x_bf16_as_f32);
        assert_eq!(x_flat.len(), batch * ns, "{tag}: x length");

        // Build x as BF16 tensor with requires_grad=true
        let x_tensor = Tensor::from_vec(x_flat, Shape::from_dims(&[batch, ns]), device.clone())?
            .to_dtype(DType::BF16)?
            .requires_grad_(true);
        let x_id = x_tensor.id();

        // Build optional weight (also BF16, requires_grad=true)
        let w_tensor: Option<Tensor> = if case.with_weight {
            let w_flat = case.w_bf16_as_f32.as_ref().expect("w missing").clone();
            assert_eq!(w_flat.len(), ns, "{tag}: w length");
            let t = Tensor::from_vec(w_flat, Shape::from_dims(&[ns]), device.clone())?
                .to_dtype(DType::BF16)?
                .requires_grad_(true);
            Some(t)
        } else {
            None
        };
        let w_id = w_tensor.as_ref().map(|t| t.id());

        AutogradContext::clear();

        let y = flame_core::norm::rms_norm(
            &x_tensor,
            &[ns],
            w_tensor.as_ref(),
            1e-5,
        )?;
        // Loss = sum(y.to_f32())
        let loss = y.to_dtype(DType::F32)?.sum()?;
        let grads = loss.backward()?;

        // grad for x
        let grad_x_tensor = grads.get(x_id).unwrap_or_else(|| {
            panic!("{tag}: no grad for x")
        });
        let got_dx: Vec<f32> = grad_x_tensor.to_dtype(DType::F32)?.to_vec()?;
        let ref_dx = flat2(&case.dx_f32);
        assert_eq!(got_dx.len(), ref_dx.len(), "{tag}: dx shape");

        let (cos_x, rel_x) = cos_and_rel_l2(&got_dx, &ref_dx);
        let pass_x = cos_x >= cos_thresh && rel_x <= rel_l2_thresh;
        let sx = if pass_x { "PASS" } else { "FAIL" };
        let lx = format!("  {tag}: dx cos={:.6} rel_L2={:.4e} [{sx}]", cos_x, rel_x);
        println!("{lx}");
        if !pass_x {
            any_failed = true;
            report.push_str(&format!("{lx} <-- BACKWARD BUG dx\n"));
        }

        // grad for weight (optional)
        if let (Some(w_id_val), Some(ref_dw_flat)) = (w_id, &case.dw_f32) {
            let grad_w_tensor = grads.get(w_id_val).unwrap_or_else(|| {
                panic!("{tag}: no grad for weight")
            });
            let got_dw: Vec<f32> = grad_w_tensor.to_dtype(DType::F32)?.to_vec()?;
            assert_eq!(got_dw.len(), ref_dw_flat.len(), "{tag}: dw shape");

            let (cos_w, rel_w) = cos_and_rel_l2(&got_dw, ref_dw_flat);
            let pass_w = cos_w >= cos_thresh && rel_w <= rel_l2_thresh;
            let sw = if pass_w { "PASS" } else { "FAIL" };
            let lw = format!("  {tag}: dw cos={:.6} rel_L2={:.4e} [{sw}]", cos_w, rel_w);
            println!("{lw}");
            if !pass_w {
                any_failed = true;
                report.push_str(&format!("{lw} <-- BACKWARD BUG dw\n"));
            }
        }
    }

    if any_failed {
        panic!(
            "RMSNorm backward diverges from PyTorch — \
             potential shared backward regression:\n{}",
            report
        );
    }

    Ok(())
}
