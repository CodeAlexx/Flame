//! Phase 5b — autograd v2 `backward_v2()` bridge tests.
//!
//! Validates the loss.backward_v2() bridge that constructs a v2-policy
//! `GradientMap` (MatchInsertedDtype) and routes v3 backward through it.
//! This is Route (ii) of Deliverable C in
//! `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md` §Phase 5.
//!
//! Coverage (all gated on `autograd_v2` so the default v3 build is
//! unaffected):
//!
//! - `backward_v2` on a 2-op graph (mul + add) with BF16 inputs yields
//!   BF16-stored grads in the returned `GradientMap`.
//! - `backward_v2` and `backward` on an F32 graph produce identical
//!   numerical results — F32 path is the parity-preserving case.
//! - `backward_v2` on a BF16 graph matches `backward` v3 within the
//!   documented BF16 tolerance (cos >= 0.999, max_abs_ratio <= 5e-3),
//!   per `BF16_GRAD_DECISION.md`.
//!
//! The bridge keeps v3 op-dispatch in place; the only thing different is
//! the grad-map's policy + dtype-unification cast at accumulation time.
//!
//! NOTE on serialization: `AutogradContext` is process-global state
//! (single `AUTOGRAD_CONTEXT: Mutex<...>` shared across threads). Running
//! these tests in cargo's default parallel mode causes one test's
//! `reset()` to wipe another test's tape entries mid-flight, producing
//! empty grad maps and downstream `.expect()`/`.unwrap()` panics.
//! As of 2026-05-13 the test suite uses `serial_test` to gate each test
//! via `#[serial]` so per-scenario CI granularity is preserved.

#![cfg(all(feature = "cuda", feature = "bf16_u16", feature = "autograd_v2"))]

use std::sync::Arc;

use cudarc::driver::CudaDevice;
use serial_test::serial;

use flame_core::autograd::policy::GradStorePolicy;
use flame_core::autograd::AutogradContext;
use flame_core::{global_cuda_device, DType, Shape, Tensor};

fn dev() -> Arc<CudaDevice> {
    global_cuda_device()
}

fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-30);
    (dot / denom) as f32
}

fn max_abs_ratio(a: &[f32], b: &[f32]) -> f32 {
    let mut max_diff = 0f32;
    let mut max_mag = 1e-12f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > max_diff {
            max_diff = d;
        }
        let m = x.abs().max(y.abs());
        if m > max_mag {
            max_mag = m;
        }
    }
    max_diff / max_mag
}

fn mk_bf16(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    let mut s = seed;
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = (((s >> 32) as f32) / (u32::MAX as f32) - 0.5) * scale;
        v.push(x);
    }
    v
}

/// 2-op graph (mul + mul) over BF16 inputs. After `backward_v2`, all
/// stored gradients must be BF16. Mul saves both operands so leaf grads
/// stay in `needed_grad_ids`.
#[test]
#[serial]
fn backward_v2_yields_bf16_grads_for_bf16_loss() {
    AutogradContext::reset();
    AutogradContext::set_enabled(true);
    let device = dev();
    let shape = Shape::from_dims(&[4]);

    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![0.5, 0.25, -1.0, 2.0];

    let a = Tensor::from_vec_dtype(a_data, shape.clone(), device.clone(), DType::BF16)
        .unwrap()
        .requires_grad_(true);
    let b = Tensor::from_vec_dtype(b_data, shape.clone(), device.clone(), DType::BF16)
        .unwrap()
        .requires_grad_(true);

    let aid = a.id();
    let bid = b.id();

    // loss = mean(a * b * a) — scalar BF16; a and b are both saved by Mul.
    let prod1 = a.mul(&b).unwrap();
    let prod2 = prod1.mul(&a).unwrap();
    let loss = prod2.mean().unwrap();
    assert_eq!(loss.dtype(), DType::BF16, "loss must be BF16 for this test");

    let grads = AutogradContext::backward_v2(&loss).unwrap();
    assert_eq!(grads.policy(), GradStorePolicy::MatchInsertedDtype);

    let g_a = grads.get(aid).expect("a has grad");
    let g_b = grads.get(bid).expect("b has grad");
    assert_eq!(g_a.dtype(), DType::BF16, "grad(a) must be BF16 end-to-end");
    assert_eq!(g_b.dtype(), DType::BF16, "grad(b) must be BF16 end-to-end");

    assert_eq!(g_a.shape().dims(), shape.dims());
    assert_eq!(g_b.shape().dims(), shape.dims());
    AutogradContext::set_enabled(false);
}

/// F32 graph: backward_v2 result equals backward v3 result numerically.
#[test]
#[serial]
fn backward_v2_matches_v3_at_f32() {
    let device = dev();
    let shape = Shape::from_dims(&[3, 4]);
    let n = shape.elem_count();

    // ---- v3 ----
    let (gv3_a, gv3_b) = {
        AutogradContext::reset();
        AutogradContext::set_enabled(true);
        let a = Tensor::from_vec_dtype(mk_bf16(11, n, 1.0), shape.clone(), device.clone(), DType::F32)
            .unwrap()
            .requires_grad_(true);
        let b = Tensor::from_vec_dtype(mk_bf16(13, n, 1.0), shape.clone(), device.clone(), DType::F32)
            .unwrap()
            .requires_grad_(true);
        let aid = a.id();
        let bid = b.id();
        let prod = a.mul(&b).unwrap();
        let loss = prod.mean().unwrap();
        let g = AutogradContext::backward(&loss).unwrap();
        let ga = g.get(aid).unwrap().to_dtype(DType::F32).unwrap().to_vec_f32().unwrap();
        let gb = g.get(bid).unwrap().to_dtype(DType::F32).unwrap().to_vec_f32().unwrap();
        AutogradContext::set_enabled(false);
        (ga, gb)
    };

    // ---- v2 ----
    let (gv2_a, gv2_b) = {
        AutogradContext::reset();
        AutogradContext::set_enabled(true);
        let a = Tensor::from_vec_dtype(mk_bf16(11, n, 1.0), shape.clone(), device.clone(), DType::F32)
            .unwrap()
            .requires_grad_(true);
        let b = Tensor::from_vec_dtype(mk_bf16(13, n, 1.0), shape.clone(), device.clone(), DType::F32)
            .unwrap()
            .requires_grad_(true);
        let aid = a.id();
        let bid = b.id();
        let prod = a.mul(&b).unwrap();
        let loss = prod.mean().unwrap();
        let g = AutogradContext::backward_v2(&loss).unwrap();
        let ga = g.get(aid).unwrap().to_dtype(DType::F32).unwrap().to_vec_f32().unwrap();
        let gb = g.get(bid).unwrap().to_dtype(DType::F32).unwrap().to_vec_f32().unwrap();
        AutogradContext::set_enabled(false);
        (ga, gb)
    };

    let cos_a = cos_sim(&gv3_a, &gv2_a);
    let cos_b = cos_sim(&gv3_b, &gv2_b);
    let r_a = max_abs_ratio(&gv3_a, &gv2_a);
    let r_b = max_abs_ratio(&gv3_b, &gv2_b);
    assert!(cos_a >= 1.0 - 1e-6, "F32 grad(a) cos {} < 1-1e-6", cos_a);
    assert!(cos_b >= 1.0 - 1e-6, "F32 grad(b) cos {} < 1-1e-6", cos_b);
    assert!(r_a <= 1e-5, "F32 grad(a) max_abs_ratio {} > 1e-5", r_a);
    assert!(r_b <= 1e-5, "F32 grad(b) max_abs_ratio {} > 1e-5", r_b);
}

/// BF16 MLP-like graph: backward_v2 vs backward must match within BF16
/// tolerance per BF16_GRAD_DECISION.md.
#[test]
#[serial]
fn backward_v2_within_tolerance_at_bf16() {
    let device = dev();
    let shape = Shape::from_dims(&[2, 8]);
    let n = shape.elem_count();

    let run = |use_v2: bool| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        AutogradContext::reset();
        AutogradContext::set_enabled(true);
        let x = Tensor::from_vec_dtype(mk_bf16(101, n, 1.0), shape.clone(), device.clone(), DType::BF16)
            .unwrap()
            .requires_grad_(true);
        let w1 = Tensor::from_vec_dtype(mk_bf16(103, n, 0.5), shape.clone(), device.clone(), DType::BF16)
            .unwrap()
            .requires_grad_(true);
        let w2 = Tensor::from_vec_dtype(mk_bf16(107, n, 0.5), shape.clone(), device.clone(), DType::BF16)
            .unwrap()
            .requires_grad_(true);
        let bias = Tensor::from_vec_dtype(mk_bf16(109, n, 0.1), shape.clone(), device.clone(), DType::BF16)
            .unwrap()
            .requires_grad_(true);

        let xid = x.id();
        let w1id = w1.id();
        let w2id = w2.id();

        let h1 = x.mul(&w1).unwrap();
        let h2 = h1.mul(&w2).unwrap();
        let h3 = h2.add(&bias).unwrap();
        let loss = h3.mean().unwrap();

        let grads = if use_v2 {
            AutogradContext::backward_v2(&loss).unwrap()
        } else {
            AutogradContext::backward(&loss).unwrap()
        };
        let g_x = grads.get_public_grad(xid).unwrap();
        let g_w1 = grads.get_public_grad(w1id).unwrap();
        let g_w2 = grads.get_public_grad(w2id).unwrap();
        AutogradContext::set_enabled(false);
        (
            g_x.to_dtype(DType::F32).unwrap().to_vec_f32().unwrap(),
            g_w1.to_dtype(DType::F32).unwrap().to_vec_f32().unwrap(),
            g_w2.to_dtype(DType::F32).unwrap().to_vec_f32().unwrap(),
        )
    };

    let (gv3_x, gv3_w1, gv3_w2) = run(false);
    let (gv2_x, gv2_w1, gv2_w2) = run(true);

    for (name, a, b) in [
        ("x", &gv3_x, &gv2_x),
        ("w1", &gv3_w1, &gv2_w1),
        ("w2", &gv3_w2, &gv2_w2),
    ] {
        let cos = cos_sim(a, b);
        let r = max_abs_ratio(a, b);
        assert!(
            cos >= 0.999,
            "grad({}) cos {} < 0.999 (BF16 tolerance per BF16_GRAD_DECISION.md)",
            name,
            cos
        );
        assert!(
            r <= 5e-3,
            "grad({}) max_abs_ratio {} > 5e-3 (BF16 tolerance per BF16_GRAD_DECISION.md)",
            name,
            r
        );
    }
}
