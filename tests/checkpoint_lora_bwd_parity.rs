//! Checkpoint LoRA backward parity — Experiment C from LOCALIZE_FINDINGS_2026-05-24.
//!
//! Tests that `AutogradContext::checkpoint(...)` correctly propagates gradients
//! to LoRA parameters (requires_grad=True) captured inside the closure, even
//! when the closure also uses frozen weights (requires_grad=False).
//!
//! This is the UNTESTED gap identified after Experiment A: ALL 7 single-op
//! backward tests passed, but none used gradient checkpointing. ALL 5 trainers
//! (klein/zimage/ernie/anima/qwen-image) use Op::Checkpoint on every attention
//! block.
//!
//! Setup for each case:
//!   lhs_activation [batch, in_dim]    BF16, requires_grad=True  (activation)
//!   lora_a         [rank, in_dim]     BF16, requires_grad=True  (LoRA-A)
//!   lora_b         [out_dim, rank]    BF16, requires_grad=True  (LoRA-B)
//!   frozen_w       [out_dim, in_dim]  BF16, requires_grad=False (frozen base weight)
//!
//! Forward inside checkpoint:
//!   x_lora = (lhs @ lora_a^T) @ lora_b^T    [batch, out_dim]
//!   x_base = lhs @ frozen_w^T                [batch, out_dim]
//!   out    = x_lora + x_base                 [batch, out_dim]
//!
//! Assertions:
//!   1. grad_lhs is present (activation flows back)
//!   2. grad_lora_a is present (LoRA-A gets gradient through checkpoint)
//!   3. grad_lora_b is present (LoRA-B gets gradient through checkpoint)
//!   4. grad_frozen_w is absent (frozen weight, no grad expected)
//!   5. All three gradients match PyTorch reference within BF16 noise (cos≥0.999)
//!   6. Checkpointed version matches plain (non-checkpointed) version
//!
//! PASS criterion: cos_sim ≥ 0.999 AND rel_L2 ≤ 1% for all three grads.
//!
//! Run with:
//!   cargo test --test checkpoint_lora_bwd_parity -- --test-threads=1 --nocapture

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::{global_cuda_device, AutogradContext, CudaDevice, DType, Result, Shape, Tensor};
use std::path::PathBuf;
use std::sync::Arc;

type CudaDev = Arc<CudaDevice>;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("parity")
        .join("checkpoint_lora_bwd_ref.json")
}

#[derive(serde::Deserialize)]
struct Case {
    batch: usize,
    in_dim: usize,
    rank: usize,
    out_dim: usize,
    seed: u32,
    lhs_f32: Vec<f32>,
    lora_a_f32: Vec<f32>,
    lora_b_f32: Vec<f32>,
    frozen_w_f32: Vec<f32>,
    // Plain (non-checkpointed) reference
    grad_lhs_plain_f32: Vec<f32>,
    grad_lora_a_plain_f32: Vec<f32>,
    grad_lora_b_plain_f32: Vec<f32>,
    // Checkpointed reference (should match plain within BF16 noise)
    grad_lhs_ckpt_f32: Vec<f32>,
    grad_lora_a_ckpt_f32: Vec<f32>,
    grad_lora_b_ckpt_f32: Vec<f32>,
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

fn check(
    tag: &str,
    name: &str,
    got: &[f32],
    reference: &[f32],
    any_failed: &mut bool,
    report: &mut String,
) {
    let (cos_v, rel) = cos_and_rel_l2(got, reference);
    let pass = cos_v >= 0.999 && rel <= 0.01;
    let s = if pass { "PASS" } else { "FAIL" };
    let line = format!(
        "  {tag}: {name} cos={:.6} rel_L2={:.4e} [{s}]",
        cos_v, rel
    );
    println!("{line}");
    if !pass {
        *any_failed = true;
        report.push_str(&format!("{line} <-- CHECKPOINT BACKWARD BUG\n"));
    }
}

/// Run the LoRA-over-frozen-base forward WITHOUT checkpointing.
///
/// Returns (out, lhs, lora_a, lora_b, frozen_w) as leaf tensors.
fn run_plain(
    lhs_data: &[f32],
    lora_a_data: &[f32],
    lora_b_data: &[f32],
    frozen_w_data: &[f32],
    b: usize,
    i: usize,
    r: usize,
    o: usize,
    device: &CudaDev,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let lhs = Tensor::from_vec(lhs_data.to_vec(), Shape::from_dims(&[b, i]), device.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);

    let lora_a = Tensor::from_vec(lora_a_data.to_vec(), Shape::from_dims(&[r, i]), device.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);

    let lora_b = Tensor::from_vec(lora_b_data.to_vec(), Shape::from_dims(&[o, r]), device.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);

    let frozen_w =
        Tensor::from_vec(frozen_w_data.to_vec(), Shape::from_dims(&[o, i]), device.clone())?
            .to_dtype(DType::BF16)?;
    // frozen_w: requires_grad=false (no call to requires_grad_)

    // x_lora_a = lhs @ lora_a^T  [b, r]
    let lora_a_t = lora_a.transpose()?;
    let x_lora_a = lhs.matmul(&lora_a_t)?;
    // x_lora = x_lora_a @ lora_b^T  [b, o]
    let lora_b_t = lora_b.transpose()?;
    let x_lora = x_lora_a.matmul(&lora_b_t)?;
    // x_base = lhs @ frozen_w^T  [b, o]
    let frozen_w_t = frozen_w.transpose()?;
    let x_base = lhs.matmul(&frozen_w_t)?;
    // out = x_lora + x_base
    let out = x_lora.add(&x_base)?;

    Ok((out, lhs, lora_a, lora_b, frozen_w))
}

#[test]
fn checkpoint_lora_backward_matches_pytorch() -> Result<()> {
    let fixture_path = fixture_path();
    let bytes = std::fs::read(&fixture_path).unwrap_or_else(|e| {
        panic!(
            "missing fixture {}: {} — regenerate via \
             `python3 tests/parity/checkpoint_lora_bwd_ref.py`",
            fixture_path.display(),
            e
        )
    });
    let fx: Fixture =
        serde_json::from_slice(&bytes).expect("malformed checkpoint_lora_bwd_ref.json");

    let device = global_cuda_device();
    let mut any_failed = false;
    let mut report = String::new();

    let cos_thresh = 0.999;
    let rel_l2_thresh = 0.01;

    for (ci, case) in fx.cases.iter().enumerate() {
        let (b, i, r, o) = (case.batch, case.in_dim, case.rank, case.out_dim);
        let tag_plain = format!("case#{ci} plain ({b},{i},{r},{o}) seed={}", case.seed);
        let tag_ckpt = format!("case#{ci} ckpt  ({b},{i},{r},{o}) seed={}", case.seed);

        // ────────────────────────────────────────────────────────────
        // Part 1: Plain backward (no checkpoint)
        // ────────────────────────────────────────────────────────────
        AutogradContext::clear();
        let (out_plain, lhs_plain, lora_a_plain, lora_b_plain, frozen_plain) = run_plain(
            &case.lhs_f32,
            &case.lora_a_f32,
            &case.lora_b_f32,
            &case.frozen_w_f32,
            b, i, r, o,
            &device,
        )?;

        assert!(
            frozen_plain.requires_grad() == false,
            "{tag_plain}: frozen_w should have requires_grad=false"
        );

        let loss_plain = out_plain.to_dtype(DType::F32)?.sum()?;
        let grads_plain = loss_plain.backward()?;

        // Verify grad presence
        let grad_lhs_plain_t = grads_plain.get(lhs_plain.id()).unwrap_or_else(|| {
            panic!("{tag_plain}: MISSING grad_lhs — activation grad lost in plain backward")
        });
        let grad_lora_a_plain_t = grads_plain.get(lora_a_plain.id()).unwrap_or_else(|| {
            panic!("{tag_plain}: MISSING grad_lora_a — LoRA-A grad lost in plain backward")
        });
        let grad_lora_b_plain_t = grads_plain.get(lora_b_plain.id()).unwrap_or_else(|| {
            panic!("{tag_plain}: MISSING grad_lora_b — LoRA-B grad lost in plain backward")
        });
        assert!(
            grads_plain.get(frozen_plain.id()).is_none(),
            "{tag_plain}: frozen_w should NOT have gradient"
        );

        let got_lhs_plain: Vec<f32> = grad_lhs_plain_t.to_dtype(DType::F32)?.to_vec()?;
        let got_lora_a_plain: Vec<f32> = grad_lora_a_plain_t.to_dtype(DType::F32)?.to_vec()?;
        let got_lora_b_plain: Vec<f32> = grad_lora_b_plain_t.to_dtype(DType::F32)?.to_vec()?;

        // Compare plain Rust vs plain PyTorch
        check(
            &tag_plain, "grad_lhs",
            &got_lhs_plain, &case.grad_lhs_plain_f32,
            &mut any_failed, &mut report,
        );
        check(
            &tag_plain, "grad_lora_a",
            &got_lora_a_plain, &case.grad_lora_a_plain_f32,
            &mut any_failed, &mut report,
        );
        check(
            &tag_plain, "grad_lora_b",
            &got_lora_b_plain, &case.grad_lora_b_plain_f32,
            &mut any_failed, &mut report,
        );

        // ────────────────────────────────────────────────────────────
        // Part 2: Checkpointed backward
        //
        // Wrap the same computation in AutogradContext::checkpoint().
        // This is the pattern used by all 5 trainers:
        //   out = AutogradContext::checkpoint(&[activation], || { ... block ... })?;
        //
        // The LoRA params (lora_a, lora_b) are CAPTURED by the closure —
        // they are not passed in the `inputs` slice (that's the activation).
        //
        // The Op::Checkpoint backward must:
        //   a) Re-run the closure with autograd enabled (builds sub-tape)
        //   b) Walk sub-tape backward, accumulating grads for lora_a + lora_b
        //      (both in trainable_ids via requires_grad=true on saved_tensors)
        //   c) Return gradients for ALL trainable_ids via Step 5
        //   d) Outer backward accumulates those grads (is_checkpoint=true path)
        // ────────────────────────────────────────────────────────────
        AutogradContext::clear();

        let lhs_c =
            Tensor::from_vec(case.lhs_f32.clone(), Shape::from_dims(&[b, i]), device.clone())?
                .to_dtype(DType::BF16)?
                .requires_grad_(true);

        let lora_a_c =
            Tensor::from_vec(case.lora_a_f32.clone(), Shape::from_dims(&[r, i]), device.clone())?
                .to_dtype(DType::BF16)?
                .requires_grad_(true);

        let lora_b_c =
            Tensor::from_vec(case.lora_b_f32.clone(), Shape::from_dims(&[o, r]), device.clone())?
                .to_dtype(DType::BF16)?
                .requires_grad_(true);

        let frozen_c =
            Tensor::from_vec(case.frozen_w_f32.clone(), Shape::from_dims(&[o, i]), device.clone())?
                .to_dtype(DType::BF16)?;
        // frozen_c: requires_grad=false

        // The inputs slice contains only the activation — LoRA params are captured.
        // This mirrors how trainers use checkpoint: the activation flows in, but
        // LoRA parameter clones are captured by the closure.
        let lhs_c_clone = lhs_c.clone();
        let lora_a_c_clone = lora_a_c.clone();
        let lora_b_c_clone = lora_b_c.clone();
        let frozen_c_clone = frozen_c.clone();

        let out_ckpt = AutogradContext::checkpoint(
            &[lhs_c.clone()],
            move || {
                let lora_a_t = lora_a_c_clone.transpose()?;
                let x_lora_a = lhs_c_clone.matmul(&lora_a_t)?;
                let lora_b_t = lora_b_c_clone.transpose()?;
                let x_lora = x_lora_a.matmul(&lora_b_t)?;
                let frozen_w_t = frozen_c_clone.transpose()?;
                let x_base = lhs_c_clone.matmul(&frozen_w_t)?;
                x_lora.add(&x_base)
            },
        )?;

        // out_ckpt.requires_grad should be true (set by checkpoint())
        assert!(
            out_ckpt.requires_grad(),
            "{tag_ckpt}: checkpoint output should have requires_grad=true"
        );

        let loss_ckpt = out_ckpt.to_dtype(DType::F32)?.sum()?;
        let grads_ckpt = loss_ckpt.backward()?;

        // CRITICAL ASSERTIONS — these are the experiment's hypothesis tests:
        let grad_lhs_ckpt_t = grads_ckpt.get(lhs_c.id()).unwrap_or_else(|| {
            panic!(
                "{tag_ckpt}: CRITICAL — MISSING grad_lhs after checkpoint backward!\n\
                 The activation gradient was dropped. This breaks backward through\n\
                 any computation BEFORE the checkpointed block."
            )
        });
        let grad_lora_a_ckpt_t = grads_ckpt.get(lora_a_c.id()).unwrap_or_else(|| {
            panic!(
                "{tag_ckpt}: CRITICAL — MISSING grad_lora_a after checkpoint backward!\n\
                 LoRA-A gradient was NOT returned by checkpoint step 5.\n\
                 Check trainable_ids construction in Op::Checkpoint backward.\n\
                 Possible cause: lora_a.requires_grad() returned false during recompute\n\
                 (e.g. captured clone lost requires_grad flag)."
            )
        });
        let grad_lora_b_ckpt_t = grads_ckpt.get(lora_b_c.id()).unwrap_or_else(|| {
            panic!(
                "{tag_ckpt}: CRITICAL — MISSING grad_lora_b after checkpoint backward!\n\
                 LoRA-B gradient was NOT returned by checkpoint step 5.\n\
                 Check trainable_ids construction in Op::Checkpoint backward."
            )
        });
        assert!(
            grads_ckpt.get(frozen_c.id()).is_none(),
            "{tag_ckpt}: frozen_w should NOT have gradient (frozen weight)"
        );

        let got_lhs_ckpt: Vec<f32> = grad_lhs_ckpt_t.to_dtype(DType::F32)?.to_vec()?;
        let got_lora_a_ckpt: Vec<f32> = grad_lora_a_ckpt_t.to_dtype(DType::F32)?.to_vec()?;
        let got_lora_b_ckpt: Vec<f32> = grad_lora_b_ckpt_t.to_dtype(DType::F32)?.to_vec()?;

        // Compare checkpointed Rust vs checkpointed PyTorch (primary parity check)
        check(
            &tag_ckpt, "grad_lhs vs PyTorch-ckpt",
            &got_lhs_ckpt, &case.grad_lhs_ckpt_f32,
            &mut any_failed, &mut report,
        );
        check(
            &tag_ckpt, "grad_lora_a vs PyTorch-ckpt",
            &got_lora_a_ckpt, &case.grad_lora_a_ckpt_f32,
            &mut any_failed, &mut report,
        );
        check(
            &tag_ckpt, "grad_lora_b vs PyTorch-ckpt",
            &got_lora_b_ckpt, &case.grad_lora_b_ckpt_f32,
            &mut any_failed, &mut report,
        );

        // Cross-check: plain Rust vs checkpointed Rust (should be bit-exact or within BF16 noise)
        let (cos_lhs_cross, rel_lhs_cross) = cos_and_rel_l2(&got_lhs_ckpt, &got_lhs_plain);
        let (cos_a_cross, rel_a_cross) = cos_and_rel_l2(&got_lora_a_ckpt, &got_lora_a_plain);
        let (cos_b_cross, rel_b_cross) = cos_and_rel_l2(&got_lora_b_ckpt, &got_lora_b_plain);
        let pass_cross_lhs = cos_lhs_cross >= cos_thresh && rel_lhs_cross <= rel_l2_thresh;
        let pass_cross_a = cos_a_cross >= cos_thresh && rel_a_cross <= rel_l2_thresh;
        let pass_cross_b = cos_b_cross >= cos_thresh && rel_b_cross <= rel_l2_thresh;
        println!(
            "  {tag_ckpt}: plain-vs-ckpt lhs cos={cos_lhs_cross:.6} rel={rel_lhs_cross:.4e} [{}]",
            if pass_cross_lhs { "PASS" } else { "FAIL" }
        );
        println!(
            "  {tag_ckpt}: plain-vs-ckpt loraA cos={cos_a_cross:.6} rel={rel_a_cross:.4e} [{}]",
            if pass_cross_a { "PASS" } else { "FAIL" }
        );
        println!(
            "  {tag_ckpt}: plain-vs-ckpt loraB cos={cos_b_cross:.6} rel={rel_b_cross:.4e} [{}]",
            if pass_cross_b { "PASS" } else { "FAIL" }
        );
        if !pass_cross_lhs || !pass_cross_a || !pass_cross_b {
            any_failed = true;
            report.push_str(&format!(
                "  {tag_ckpt}: plain-vs-ckpt MISMATCH — checkpoint backward produces different \
                 gradient than plain backward. This is a composition bug in Op::Checkpoint.\n"
            ));
        }

        println!();
    }

    if any_failed {
        panic!(
            "CONFIRMED BUG: Op::Checkpoint backward drops or corrupts LoRA gradients.\n\
             This is the source of the shared regression in ALL 5 trainers.\n\
             {}", report
        );
    }

    println!("\n  All cases PASS. AutogradContext::checkpoint backward correctly");
    println!("  propagates gradients to LoRA-A and LoRA-B parameters captured");
    println!("  by the recompute closure. The checkpoint backward is CLEAN.");
    println!("  If trainers still diverge, the bug must be in multi-step or");
    println!("  optimizer-state accumulation — not in single-step checkpoint.\n");

    Ok(())
}
