//! BF16 matmul backward parity vs PyTorch — trainer shapes.
//!
//! Experiment A, test #1 from LOCALIZE_FINDINGS_2026-05-24.
//!
//! For each shape (batch, in_dim, out_dim) from `tests/parity/bf16_matmul_bwd_ref.json`,
//! reconstructs the same forward graph using autograd and compares:
//!   - dA (gradient w.r.t. input A, BF16 [batch, in_dim])
//!   - dB (gradient w.r.t. weight B, BF16 [out_dim, in_dim])
//!
//! Forward:  out = A @ B^T   (Op::MatMul: A [batch,in] @ B^T [in,out])
//! Loss:     loss = out.to_f32().sum()   (matches fixture generator)
//! Backward: via flame-core's autograd tape
//!
//! PASS criterion: cos_sim ≥ 0.999 AND rel_L2 ≤ 1% for each gradient.
//! A failure here isolates BF16 matmul backward (Op::MatMul path) as the
//! regression source shared by all 5 trainers.
//!
//! Run with:
//!   cargo test --test bf16_matmul_bwd_torch_parity -- --test-threads=1 --nocapture

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::{global_cuda_device, AutogradContext, DType, Result, Shape, Tensor};
use std::path::PathBuf;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("parity")
        .join("bf16_matmul_bwd_ref.json")
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
    #[serde(rename = "dB_f32")]
    db_f32: Vec<f32>,
}

#[derive(serde::Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

fn cos_and_rel_l2(got: &[f32], reference: &[f32]) -> (f64, f64) {
    assert_eq!(got.len(), reference.len(), "length mismatch in cos_and_rel_l2");
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
fn bf16_matmul_backward_matches_pytorch() -> Result<()> {
    let fixture_path = fixture_path();
    let bytes = std::fs::read(&fixture_path).unwrap_or_else(|e| {
        panic!(
            "missing fixture {}: {} — regenerate via \
             `python3 tests/parity/bf16_matmul_bwd_ref.py`",
            fixture_path.display(),
            e
        )
    });
    let fx: Fixture = serde_json::from_slice(&bytes).expect("malformed bf16_matmul_bwd_ref.json");

    let device = global_cuda_device();
    let mut any_failed = false;
    let mut report = String::new();

    // PASS thresholds
    let cos_thresh = 0.999;
    let rel_l2_thresh = 0.01; // 1%

    for (ci, case) in fx.cases.iter().enumerate() {
        let (b, i, o) = (case.batch, case.in_dim, case.out_dim);
        let tag = format!("case#{ci} ({b},{i},{o}) seed={}", case.seed);

        assert_eq!(case.a_f32.len(), b * i, "{tag}: A_f32 length mismatch");
        assert_eq!(case.b_f32.len(), o * i, "{tag}: B_f32 length mismatch");

        // Build A [batch, in_dim] BF16, requires_grad=true
        let a_tensor = Tensor::from_vec(case.a_f32.clone(), Shape::from_dims(&[b, i]), device.clone())?
            .to_dtype(DType::BF16)?
            .requires_grad_(true);
        // Build B [out_dim, in_dim] BF16, requires_grad=true
        let b_tensor = Tensor::from_vec(case.b_f32.clone(), Shape::from_dims(&[o, i]), device.clone())?
            .to_dtype(DType::BF16)?
            .requires_grad_(true);

        AutogradContext::clear();

        // Forward: out = A @ B^T  [batch, out_dim]
        let b_t = b_tensor.transpose()?;
        let out = a_tensor.matmul(&b_t)?;
        // Loss: sum as F32 scalar (Op::Sum or cast+sum path)
        let out_f32 = out.to_dtype(DType::F32)?;
        let loss = out_f32.sum()?;

        let grads = loss.backward()?;

        let grad_a = grads.get(a_tensor.id()).unwrap_or_else(|| {
            panic!("{tag}: no grad for A — autograd broken")
        });
        let grad_b = grads.get(b_tensor.id()).unwrap_or_else(|| {
            panic!("{tag}: no grad for B — autograd broken")
        });

        let got_da: Vec<f32> = grad_a.to_dtype(DType::F32)?.to_vec()?;
        let got_db: Vec<f32> = grad_b.to_dtype(DType::F32)?.to_vec()?;
        let ref_da = &case.da_f32;
        let ref_db = &case.db_f32;

        assert_eq!(got_da.len(), ref_da.len(), "{tag}: dA shape mismatch (got {} expected {})", got_da.len(), ref_da.len());
        assert_eq!(got_db.len(), ref_db.len(), "{tag}: dB shape mismatch (got {} expected {})", got_db.len(), ref_db.len());

        let (cos_a, rel_a) = cos_and_rel_l2(&got_da, ref_da);
        let (cos_b, rel_b) = cos_and_rel_l2(&got_db, ref_db);

        let pass_a = cos_a >= cos_thresh && rel_a <= rel_l2_thresh;
        let pass_b = cos_b >= cos_thresh && rel_b <= rel_l2_thresh;

        let status_a = if pass_a { "PASS" } else { "FAIL" };
        let status_b = if pass_b { "PASS" } else { "FAIL" };

        let line_a = format!(
            "  {tag}: dA cos={:.6} rel_L2={:.4e} [{status_a}]",
            cos_a, rel_a
        );
        let line_b = format!(
            "  {tag}: dB cos={:.6} rel_L2={:.4e} [{status_b}]",
            cos_b, rel_b
        );
        println!("{line_a}");
        println!("{line_b}");

        if !pass_a {
            any_failed = true;
            report.push_str(&format!("{line_a} <-- BACKWARD BUG dA\n"));
        }
        if !pass_b {
            any_failed = true;
            report.push_str(&format!("{line_b} <-- BACKWARD BUG dB\n"));
        }
    }

    if any_failed {
        panic!(
            "BF16 matmul backward diverges from PyTorch reference — \
             this is the shared backward regression causing ALL 5 trainers to fail:\n\
             {}",
            report
        );
    }

    Ok(())
}
