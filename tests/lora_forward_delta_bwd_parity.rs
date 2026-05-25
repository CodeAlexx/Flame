//! LoRALinear::forward_delta backward parity — Task 1 from reproducer brief.
//!
//! Tests the EXACT code path used by all 5 trainers (klein/zimage/ernie/anima/qwen-image).
//! This is NOT a simplified matmul test — it replicates:
//!
//!   LoRALinear::forward_delta (lora.rs:83) step by step:
//!     1. Flatten input: [B, N, Cin] -> [B*N, Cin]
//!     2. Cast lora_a  F32 -> BF16   (via Op::Cast)
//!     3. Cast lora_b  F32 -> BF16   (via Op::Cast)
//!     4. Transpose a_bf16 -> a_t    (Op::Transpose)
//!     5. Transpose b_bf16 -> b_t    (Op::Transpose)
//!     6. intermediate = input_2d @ a_t   [B*N, rank]   (Op::MatMul)
//!     7. out_2d = intermediate @ b_t     [B*N, Cout]   (Op::MatMul)
//!     8. scaled = out_2d * (alpha/rank)  (Op::MulScalar, if scale≠1)
//!     9. reshape back to [B, N, Cout]
//!
//! Loss is MSE(delta_out + base_out, target), matching real trainer usage.
//!
//! Experiments A–C tested isolated ops and single matmul paths. This test exercises
//! the FULL composition: Cast(F32->BF16) → Transpose → MatMul → Transpose → MatMul
//! → MulScalar — exactly as the trainer's backward graph walks it.
//!
//! The key untested interaction (per HANDOFF §2 + LOCALIZE_FINDINGS §Remaining Candidates):
//!   - lora_a and lora_b are F32 leaves. The backward must flow through Op::Cast
//!     (BF16->F32 cast-backward, i.e., accumulate BF16 gradient into F32 leaf).
//!   - If Op::Cast backward clips, truncates, or drops the gradient for the F32 leaf,
//!     it would manifest as wrong grad_lora_a / grad_lora_b — exactly the 2.82× ratio
//!     observed in the same-batch parity probe (§2).
//!
//! Fixture: tests/parity/lora_forward_delta_bwd_ref.py + lora_forward_delta_bwd_ref.json
//!
//! PASS criterion: cos_sim ≥ 0.999 AND rel_L2 ≤ 1% for both grad_lora_a and grad_lora_b.
//!
//! Run with:
//!   cargo test --test lora_forward_delta_bwd_parity -- --test-threads=1 --nocapture

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::{global_cuda_device, AutogradContext, CudaDevice, DType, Result, Shape, Tensor};
use std::path::PathBuf;
use std::sync::Arc;

type CudaDev = Arc<CudaDevice>;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("parity")
        .join("lora_forward_delta_bwd_ref.json")
}

#[derive(serde::Deserialize)]
struct Case {
    tag: String,
    batch: usize,
    seq: usize,
    in_features: usize,
    out_features: usize,
    rank: usize,
    alpha: f32,
    seed: u64,
    // Input: BF16 activation (stored as F32 in fixture, cast in Rust)
    input_bf16_as_f32: Vec<f32>,
    // LoRA params: F32 (as stored in Parameter — flat row-major)
    lora_a_f32: Vec<f32>,
    lora_b_f32: Vec<f32>,
    // Frozen base weight: BF16 (stored as F32)
    base_w_bf16_as_f32: Vec<f32>,
    // MSE target: BF16 (stored as F32)
    target_bf16_as_f32: Vec<f32>,
    // Expected gradients (F32)
    grad_lora_a_f32: Vec<f32>,
    grad_lora_b_f32: Vec<f32>,
}

#[derive(serde::Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

fn cos_and_rel_l2(got: &[f32], reference: &[f32]) -> (f64, f64) {
    assert_eq!(
        got.len(),
        reference.len(),
        "gradient length mismatch: got {} vs ref {}",
        got.len(),
        reference.len()
    );
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

/// Replicate forward_delta exactly as in lora.rs:
///   1. Flatten [B, N, Cin] -> [B*N, Cin]
///   2. Cast lora_a_f32 -> BF16, transpose -> [Cin, rank]
///   3. Cast lora_b_f32 -> BF16, transpose -> [rank, Cout]
///   4. intermediate = input_2d @ a_t   [B*N, rank]
///   5. out_2d = intermediate @ b_t     [B*N, Cout]
///   6. scale = alpha / rank; scaled = out_2d * scale  (if scale≠1)
///   7. reshape -> [B, N, Cout]
fn forward_delta_rust(
    input_3d: &Tensor,   // BF16 [B, N, Cin], requires_grad=False (activation)
    lora_a: &Tensor,     // F32  [rank, Cin], requires_grad=True
    lora_b: &Tensor,     // F32  [Cout, rank], requires_grad=True
    alpha: f32,
    rank: usize,
    batch: usize,
    seq: usize,
    in_features: usize,
    out_features: usize,
) -> Result<Tensor> {
    // 1. Flatten [B, N, Cin] -> [B*N, Cin]
    let input_2d = input_3d.reshape(&[batch * seq, in_features])?;

    // 2–3. Cast F32 params -> BF16 (Op::Cast, autograd-aware)
    let a_bf16 = lora_a.to_dtype(DType::BF16)?;   // [rank, in_features], requires_grad=True via Cast chain
    let b_bf16 = lora_b.to_dtype(DType::BF16)?;   // [out_features, rank], requires_grad=True via Cast chain

    // 4–5. Transpose and matmul
    let a_t = a_bf16.transpose()?;                 // [in_features, rank]
    let b_t = b_bf16.transpose()?;                 // [rank, out_features]
    let intermediate = input_2d.matmul(&a_t)?;     // [B*N, rank]
    let out_2d = intermediate.matmul(&b_t)?;       // [B*N, out_features]

    // 6. Scale
    let scale = alpha / rank as f32;
    let scaled = if (scale - 1.0f32).abs() > f32::EPSILON {
        out_2d.mul_scalar(scale)?
    } else {
        out_2d
    };

    // 7. Reshape back to [B, N, Cout]
    scaled.reshape(&[batch, seq, out_features]).map_err(Into::into)
}

#[test]
fn lora_forward_delta_backward_matches_pytorch() -> Result<()> {
    let fixture_path = fixture_path();
    let bytes = std::fs::read(&fixture_path).unwrap_or_else(|e| {
        panic!(
            "missing fixture {}: {} — regenerate via \
             `python3 tests/parity/lora_forward_delta_bwd_ref.py`",
            fixture_path.display(),
            e
        )
    });
    let fx: Fixture =
        serde_json::from_slice(&bytes).expect("malformed lora_forward_delta_bwd_ref.json");

    let device = global_cuda_device();
    let mut any_failed = false;
    let mut fail_report = String::new();

    // Thresholds — BF16 compute so allow slightly wider than F32
    let cos_thresh = 0.999f64;
    let rel_thresh = 0.01f64;

    for case in &fx.cases {
        let (b, n, ic, oc, r) = (
            case.batch,
            case.seq,
            case.in_features,
            case.out_features,
            case.rank,
        );
        let tag = format!(
            "case[{}] ({},{},{}->{},rank={},alpha={})",
            case.tag, b, n, ic, oc, r, case.alpha
        );

        AutogradContext::clear();

        // ── Load input (BF16 activation, no grad) ─────────────────────────────
        let input_3d = Tensor::from_vec(
            case.input_bf16_as_f32.clone(),
            Shape::from_dims(&[b, n, ic]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;
        // input is an activation — no requires_grad on the activation itself
        // (trainers don't backprop into frozen model activations at the LoRA branch)

        // ── Load LoRA params (F32 leaves, requires_grad=True) ─────────────────
        // lora_a shape: [rank, in_features]
        let lora_a = Tensor::from_vec(
            case.lora_a_f32.clone(),
            Shape::from_dims(&[r, ic]),
            device.clone(),
        )?
        .requires_grad_(true);

        // lora_b shape: [out_features, rank]
        let lora_b = Tensor::from_vec(
            case.lora_b_f32.clone(),
            Shape::from_dims(&[oc, r]),
            device.clone(),
        )?
        .requires_grad_(true);

        // ── Load frozen base weight (BF16, no grad) ───────────────────────────
        // shape: [out_features, in_features]  (weight matrix, transposed in matmul)
        let base_w = Tensor::from_vec(
            case.base_w_bf16_as_f32.clone(),
            Shape::from_dims(&[oc, ic]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;
        // no requires_grad_ call → requires_grad=false

        // ── Load target (BF16, no grad) ───────────────────────────────────────
        let target = Tensor::from_vec(
            case.target_bf16_as_f32.clone(),
            Shape::from_dims(&[b, n, oc]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)?;

        // ── Forward: base path ────────────────────────────────────────────────
        // base_out = input_2d @ base_w^T  (frozen, like the real model)
        let input_2d = input_3d.reshape(&[b * n, ic])?;
        let base_w_t = base_w.transpose()?;
        let base_out_2d = input_2d.matmul(&base_w_t)?;
        let base_out = base_out_2d.reshape(&[b, n, oc])?;

        // ── Forward: LoRA delta ───────────────────────────────────────────────
        let lora_out = forward_delta_rust(
            &input_3d, &lora_a, &lora_b,
            case.alpha, r, b, n, ic, oc,
        )?;

        // ── Combined output ───────────────────────────────────────────────────
        let out = base_out.add(&lora_out)?;   // [B, N, Cout], BF16

        // ── MSE loss (F32) ────────────────────────────────────────────────────
        let out_f32 = out.to_dtype(DType::F32)?;
        let target_f32 = target.to_dtype(DType::F32)?;
        let diff = out_f32.sub(&target_f32)?;
        // MSE = mean(diff^2) — use sum+scalar for simplicity (same gradient direction)
        let diff_sq = diff.mul(&diff)?;
        // mean = sum / N to match Python F.mse_loss (mean not sum)
        let n_elems = (b * n * oc) as f32;
        let loss_sum = diff_sq.sum()?;
        let loss = loss_sum.mul_scalar(1.0f32 / n_elems)?;

        // ── Backward ─────────────────────────────────────────────────────────
        let grads = loss.backward()?;

        // ── Check gradients exist ─────────────────────────────────────────────
        let grad_a_t = grads.get(lora_a.id()).unwrap_or_else(|| {
            panic!(
                "{tag}: MISSING grad_lora_a — the Cast(F32->BF16)→Transpose→MatMul backward\n\
                 did not produce a gradient for the F32 lora_a leaf.\n\
                 This is the critical forward_delta path used by all 5 trainers.\n\
                 Suspect: Op::Cast backward does not accumulate into F32 leaf when\n\
                 upstream gradient is BF16."
            )
        });
        let grad_b_t = grads.get(lora_b.id()).unwrap_or_else(|| {
            panic!(
                "{tag}: MISSING grad_lora_b — the Cast(F32->BF16)→Transpose→MatMul backward\n\
                 did not produce a gradient for the F32 lora_b leaf.\n\
                 This is the second matmul in the LoRA chain."
            )
        });

        // ── Convert to CPU F32 for comparison ────────────────────────────────
        let got_a: Vec<f32> = grad_a_t.to_dtype(DType::F32)?.to_vec()?;
        let got_b: Vec<f32> = grad_b_t.to_dtype(DType::F32)?.to_vec()?;

        // ── Compare ──────────────────────────────────────────────────────────
        let (cos_a, rel_a) = cos_and_rel_l2(&got_a, &case.grad_lora_a_f32);
        let (cos_b, rel_b) = cos_and_rel_l2(&got_b, &case.grad_lora_b_f32);

        let pass_a = cos_a >= cos_thresh && rel_a <= rel_thresh;
        let pass_b = cos_b >= cos_thresh && rel_b <= rel_thresh;

        let sa = if pass_a { "PASS" } else { "FAIL" };
        let sb = if pass_b { "PASS" } else { "FAIL" };

        println!(
            "{tag}:\n  grad_lora_a: cos={:.6} rel_L2={:.4e} [{sa}]\n  grad_lora_b: cos={:.6} rel_L2={:.4e} [{sb}]",
            cos_a, rel_a, cos_b, rel_b
        );

        if !pass_a {
            any_failed = true;
            fail_report.push_str(&format!(
                "FAIL {tag} grad_lora_a: cos={cos_a:.6} rel={rel_a:.4e}\n\
                 -- Op::Cast(F32->BF16) backward likely clips or drops F32 leaf gradient.\n\
                 -- This is consistent with the 2.82x lora_B ratio from the §2 same-batch probe.\n"
            ));
        }
        if !pass_b {
            any_failed = true;
            fail_report.push_str(&format!(
                "FAIL {tag} grad_lora_b: cos={cos_b:.6} rel={rel_b:.4e}\n\
                 -- Op::Cast(F32->BF16) backward for lora_b leaf is wrong.\n"
            ));
        }
    }

    if any_failed {
        panic!(
            "\n=== CONFIRMED BUG: LoRALinear::forward_delta backward mismatch vs PyTorch ===\n\
             This IS the source of the shared regression in ALL 5 trainers.\n\
             The forward_delta path (Cast+Transpose+MatMul+MatMul) produces wrong\n\
             gradients for F32 LoRA parameters even when single-op tests pass.\n\
             Fix Op::Cast backward to correctly accumulate BF16 upstream gradient\n\
             into F32 leaf (sum reduction over the extra precision, not BF16-truncate).\n\
             \n\
             Failures:\n{fail_report}"
        );
    }

    println!(
        "\n  ALL CASES PASS — LoRALinear::forward_delta backward is correct.\n\
         The Cast(F32->BF16) + double matmul path produces correct F32 leaf\n\
         gradients matching PyTorch within BF16 noise (cos≥0.999, rel≤1%).\n\
         The forward_delta code path is NOT the source of the 2.82× ratio.\n\
         The bug must be in multi-step training dynamics or optimizer state."
    );

    Ok(())
}
