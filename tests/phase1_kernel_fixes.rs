//! Phase 1 kernel-fix parity tests.
//!
//! Tier-1 audit punch-list (HANDOFF_2026-05-03):
//!   1.1 Tensor::cat returns contiguous output
//!   1.2 Conv1d k=1 fast path on permuted-view inputs
//!   1.3 Fused norm/linear backward BF16 accumulation (separate test file)
//!
//! These tests EXIST TO VERIFY THE FIXES. If a test passes against the
//! current code (no kernel change), the bug is already defused — we keep
//! the test as a regression guard. If it fails, the kernel needs work.

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use anyhow::Result;
use flame_core::device::Device;
use flame_core::tensor::Tensor;
use flame_core::{conv1d::conv1d, DType, Shape};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn seeded_f32(numel: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..numel).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect()
}

use std::sync::OnceLock;
fn dev() -> std::sync::Arc<cudarc::driver::CudaDevice> {
    static D: OnceLock<std::sync::Arc<cudarc::driver::CudaDevice>> = OnceLock::new();
    D.get_or_init(|| Device::cuda(0).unwrap().cuda_device_arc())
        .clone()
}

fn make_bf16(data: Vec<f32>, shape: &[usize]) -> Result<Tensor> {
    let t = Tensor::from_vec(data, Shape::from_dims(shape), dev())?;
    Ok(t.to_dtype(DType::BF16)?)
}

fn make_f32(data: Vec<f32>, shape: &[usize]) -> Result<Tensor> {
    Ok(Tensor::from_vec(data, Shape::from_dims(shape), dev())?)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ---------------------------------------------------------------------------
// 1.1 — Tensor::cat returns contiguous output (BF16 + F32, with permuted inputs)
// ---------------------------------------------------------------------------

#[test]
fn cat_bf16_output_is_contiguous() -> Result<()> {
    let a = make_bf16(seeded_f32(2 * 3 * 4, 1), &[2, 3, 4])?;
    let b = make_bf16(seeded_f32(2 * 3 * 4, 2), &[2, 3, 4])?;
    let c = make_bf16(seeded_f32(2 * 3 * 4, 3), &[2, 3, 4])?;
    let out = Tensor::cat(&[&a, &b, &c], 2)?;
    assert!(out.is_contiguous(), "cat BF16 output not contiguous");
    assert_eq!(out.shape().dims(), &[2, 3, 12]);
    Ok(())
}

#[test]
fn cat_f32_output_is_contiguous() -> Result<()> {
    let a = make_f32(seeded_f32(2 * 3 * 4, 1), &[2, 3, 4])?;
    let b = make_f32(seeded_f32(2 * 3 * 4, 2), &[2, 3, 4])?;
    let out = Tensor::cat(&[&a, &b], 2)?;
    assert!(out.is_contiguous(), "cat F32 output not contiguous");
    assert_eq!(out.shape().dims(), &[2, 3, 8]);
    Ok(())
}

#[test]
fn cat_with_permuted_input_materializes_and_is_contiguous() -> Result<()> {
    // Per `tensor_ops_extended.rs:313-316`, cat must materialize non-contig
    // inputs internally. Output must be contiguous.
    let raw = make_bf16(seeded_f32(2 * 3 * 4, 7), &[2, 3, 4])?;
    let permuted = raw.permute(&[0, 2, 1])?; // now [2, 4, 3], non-contig
    assert!(!permuted.is_contiguous());

    let extra = make_bf16(seeded_f32(2 * 4 * 3, 8), &[2, 4, 3])?;
    let out = Tensor::cat(&[&permuted, &extra], 1)?;
    assert!(
        out.is_contiguous(),
        "cat with permuted input lost contiguity"
    );
    assert_eq!(out.shape().dims(), &[2, 8, 3]);

    // Spot-check values.
    let out_vec = out.to_dtype(DType::F32)?.to_vec()?;
    let perm_vec = permuted.contiguous()?.to_dtype(DType::F32)?.to_vec()?;
    let extra_vec = extra.to_dtype(DType::F32)?.to_vec()?;
    for batch in 0..2 {
        for r in 0..4 {
            for c in 0..3 {
                let oi = batch * 8 * 3 + r * 3 + c;
                let pi = batch * 4 * 3 + r * 3 + c;
                let d = (out_vec[oi] - perm_vec[pi]).abs();
                assert!(
                    d < 5e-3,
                    "permuted half mismatch at b{batch} r{r} c{c}: {} vs {}",
                    out_vec[oi],
                    perm_vec[pi]
                );
            }
        }
        for r in 0..4 {
            for c in 0..3 {
                let oi = batch * 8 * 3 + (4 + r) * 3 + c;
                let ei = batch * 4 * 3 + r * 3 + c;
                let d = (out_vec[oi] - extra_vec[ei]).abs();
                assert!(d < 5e-3, "extra half mismatch");
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 1.2 — Conv1d k=1 fast path on permuted-view inputs
// ---------------------------------------------------------------------------

#[test]
fn conv1d_k1_fast_path_matches_unfused_path() -> Result<()> {
    let b = 2;
    let c_in = 6;
    let c_out = 4;
    let l = 5;
    let x = make_bf16(seeded_f32(b * c_in * l, 11), &[b, c_in, l])?;
    let w = make_bf16(seeded_f32(c_out * c_in, 13), &[c_out, c_in, 1])?;
    let bias = make_bf16(seeded_f32(c_out, 17), &[c_out])?;

    // Path A: conv1d k=1 fast path (current code).
    let out_fast = conv1d(&x, &w, Some(&bias), 1, 0, 1, 1)?;

    // Path B: hand-roll the same op with explicit materialization.
    let x_contig = x.contiguous()?;
    let x_t = x_contig.permute(&[0, 2, 1])?.contiguous()?; // [B, L, C_in]
    let w_2d = w.reshape(&[c_out, c_in])?;
    let w_t = w_2d.permute(&[1, 0])?.contiguous()?; // [C_in, C_out]
    let out_2d = x_t.matmul(&w_t)?; // [B, L, C_out]
    let out_chl = out_2d.permute(&[0, 2, 1])?.contiguous()?; // [B, C_out, L]
    let bias_3d = bias.reshape(&[1, c_out, 1])?;
    let out_ref = out_chl.add(&bias_3d)?;

    let af = out_fast.to_dtype(DType::F32)?.to_vec()?;
    let bf = out_ref.to_dtype(DType::F32)?.to_vec()?;
    let max_diff = max_abs_diff(&af, &bf);
    assert!(
        max_diff < 1e-2,
        "conv1d k=1 fast path differs from contiguous reference: max_abs={max_diff}"
    );
    Ok(())
}

#[test]
fn conv1d_k1_with_permuted_input_view_matches_contig() -> Result<()> {
    // Caller hands us a permuted view BEFORE conv1d.
    let b = 2;
    let l = 5;
    let c_in = 6;
    let c_out = 4;
    let raw = make_bf16(seeded_f32(b * l * c_in, 23), &[b, l, c_in])?;
    let x_view = raw.permute(&[0, 2, 1])?; // non-contig [B, C_in, L]
    assert!(!x_view.is_contiguous());

    let w = make_bf16(seeded_f32(c_out * c_in, 29), &[c_out, c_in, 1])?;
    let bias = make_bf16(seeded_f32(c_out, 31), &[c_out])?;

    let out_view = conv1d(&x_view, &w, Some(&bias), 1, 0, 1, 1)?;

    // Reference: same conv over contiguous-materialized input.
    let x_contig = x_view.contiguous()?;
    let out_contig = conv1d(&x_contig, &w, Some(&bias), 1, 0, 1, 1)?;

    let av = out_view.to_dtype(DType::F32)?.to_vec()?;
    let bv = out_contig.to_dtype(DType::F32)?.to_vec()?;
    let max_diff = max_abs_diff(&av, &bv);
    assert!(
        max_diff < 1e-2,
        "conv1d k=1 with permuted-view input differs from contig input: max_abs={max_diff}"
    );
    Ok(())
}
