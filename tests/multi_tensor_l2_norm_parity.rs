//! Multi-tensor L2 norm vs per-tensor fold parity.
//!
//! `flame_core::ops::grad_norm::global_l2_norm` has a multi-tensor fast
//! path that collapses N tensors' sum-of-squares + fold into 2 kernel
//! launches (stage1 + stage2). The legacy per-tensor fold remains as the
//! fallback for mixed-dtype slices. Both paths must produce the same L2
//! norm within a documented F32 tolerance.
//!
//! Parallel-tree reduction does NOT preserve floating-point associativity:
//! `(a + b) + c` and `a + (b + c)` can differ in the low bits of F32. The
//! multi-tensor stage1 sums elements in a shared-memory tree (block-wide
//! reduction), then stage2 sums per-tensor partials in another tree. The
//! legacy path sums tensor-by-tensor with a serial fold-add. Both finish
//! with `sqrt` on the same scalar register, so the only drift source is
//! reduction order.
//!
//! Expected drift bound for production gradient magnitudes (~1e-2 RMS,
//! ~1e5 elements per tensor, ~280 tensors): F32 ULP × log2(elems) ≈ 1e-7 ×
//! 17 ≈ 1.7e-6 per partial, accumulated tree depth 9 across partials ≈
//! 1.5e-5 worst case. We assert 1e-5 absolute / 1e-6 relative.

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::ops::grad_norm::global_l2_norm;
use flame_core::{global_cuda_device, DType, Shape, Tensor};

fn make_shapes() -> Vec<Shape> {
    // Mix of LoRA-style shapes — what zimage / klein training feeds in.
    let mut out = Vec::with_capacity(60);
    for i in 0..60usize {
        let r = 8 + (i % 3) * 4;
        let dim = 1024 + (i % 5) * 256;
        if i % 2 == 0 {
            out.push(Shape::from_dims(&[r, dim]));
        } else {
            out.push(Shape::from_dims(&[dim, r]));
        }
    }
    out
}

fn deterministic_data(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut x = seed
        .wrapping_mul(2862933555777941757)
        .wrapping_add(3037000493);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (x >> 32) as u32;
            let normalized = (bits as f32 / u32::MAX as f32) * 2.0 - 1.0;
            normalized * scale
        })
        .collect()
}

fn build_f32_grads(shapes: &[Shape]) -> Vec<Tensor> {
    let dev = global_cuda_device();
    shapes
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let n = s.elem_count();
            let data = deterministic_data(n, 3000 + i as u64, 0.01);
            Tensor::from_vec(data, s.clone(), dev.clone()).unwrap()
        })
        .collect()
}

#[test]
fn multi_tensor_l2_norm_matches_legacy_within_tolerance() {
    let shapes = make_shapes();
    let grads_owned = build_f32_grads(&shapes);
    let grads: Vec<&Tensor> = grads_owned.iter().collect();

    // Multi-tensor path (default).
    std::env::remove_var("FLAME_MT_L2NORM");
    let norm_mt = global_l2_norm(&grads).expect("mt path");
    let norm_mt_val = norm_mt.item().expect("item mt") as f32;

    // Legacy per-tensor fold path.
    std::env::set_var("FLAME_MT_L2NORM", "0");
    let norm_legacy = global_l2_norm(&grads).expect("legacy path");
    let norm_legacy_val = norm_legacy.item().expect("item legacy") as f32;
    std::env::remove_var("FLAME_MT_L2NORM");

    let abs_diff = (norm_mt_val - norm_legacy_val).abs();
    let rel_diff = abs_diff / norm_legacy_val.abs().max(1e-6);

    println!(
        "L2 norm: mt={norm_mt_val:.9}  legacy={norm_legacy_val:.9}  \
         abs={abs_diff:.3e}  rel={rel_diff:.3e}"
    );

    assert!(
        abs_diff <= 1e-5,
        "mt vs legacy abs drift = {abs_diff:.3e} exceeds 1e-5 tolerance"
    );
    assert!(
        rel_diff <= 1e-6,
        "mt vs legacy rel drift = {rel_diff:.3e} exceeds 1e-6 tolerance"
    );
}

#[test]
fn multi_tensor_l2_norm_single_tensor_matches_legacy() {
    // Edge case: single-tensor list. Both paths should agree closely.
    let dev = global_cuda_device();
    let n = 4096usize;
    let data = deterministic_data(n, 12345, 0.05);
    let t = Tensor::from_vec(data, Shape::from_dims(&[64, 64]), dev).unwrap();
    let grads: Vec<&Tensor> = vec![&t];

    std::env::remove_var("FLAME_MT_L2NORM");
    let mt = global_l2_norm(&grads).unwrap().item().unwrap() as f32;
    std::env::set_var("FLAME_MT_L2NORM", "0");
    let legacy = global_l2_norm(&grads).unwrap().item().unwrap() as f32;
    std::env::remove_var("FLAME_MT_L2NORM");

    let abs_diff = (mt - legacy).abs();
    assert!(
        abs_diff <= 1e-5,
        "single-tensor mt={mt} legacy={legacy} abs={abs_diff:.3e}"
    );
}

#[test]
fn multi_tensor_l2_norm_empty_returns_zero() {
    let grads: Vec<&Tensor> = vec![];
    let norm = global_l2_norm(&grads).unwrap();
    let val = norm.item().unwrap() as f32;
    assert_eq!(val, 0.0, "empty slice must yield exact zero, got {val}");
}

#[test]
fn multi_tensor_l2_norm_bf16_falls_through_to_legacy() {
    // BF16 grads: classifier rejects multi-tensor path (F32-only for now),
    // legacy path handles via cast. Both should agree within BF16 tolerance.
    // This test is mainly a smoke check that the dispatch doesn't error.
    let dev = global_cuda_device();
    let shapes: Vec<Shape> = (0..8)
        .map(|i| Shape::from_dims(&[16, 256 + i * 64]))
        .collect();
    let owned: Vec<Tensor> = shapes
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let data = deterministic_data(s.elem_count(), 5000 + i as u64, 0.02);
            Tensor::from_vec(data, s.clone(), dev.clone())
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        })
        .collect();
    let grads: Vec<&Tensor> = owned.iter().collect();

    let norm = global_l2_norm(&grads).expect("bf16 fallback");
    let val = norm.item().unwrap() as f32;
    assert!(val.is_finite() && val > 0.0, "got bogus norm {val}");
}
