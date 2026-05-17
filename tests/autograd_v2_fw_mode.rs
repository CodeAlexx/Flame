//! Phase 3c2 — autograd v2 per-op forward-mode AD (JVP) tests.
//!
//! Each test installs a `fw_grad` (tangent) on one or more inputs,
//! invokes the v2 op forward wrapper, and verifies the output's
//! `fw_grad` matches the analytical JVP formula applied to the
//! primals + tangents.
//!
//! These tests do NOT verify against PyTorch — that's Phase 5's job.
//! They verify the JVP plumbing is wired correctly through the v2
//! ops and matches the analytical formula documented in each
//! wrapper's doc comment.
//!
//! Coverage (12 tests, one per JVP-supporting op):
//!
//! 1.  `add_v2_forward_mode_jvp`
//! 2.  `mul_v2_forward_mode_jvp_product_rule`
//! 3.  `sum_v2_forward_mode_jvp`
//! 4.  `matmul_v2_forward_mode_jvp_product_rule`
//! 5.  `silu_v2_forward_mode_jvp`
//! 6.  `reshape_v2_forward_mode_jvp`
//! 7.  `view_v2_forward_mode_jvp`
//! 8.  `transpose_v2_forward_mode_jvp`
//! 9.  `narrow_v2_forward_mode_jvp`
//! 10. `squeeze_v2_forward_mode_jvp`
//! 11. `unsqueeze_v2_forward_mode_jvp`
//! 12. `permute_v2_forward_mode_jvp`
//!
//! Plus one regression test:
//!
//! 13. `set_fw_grad_implicitly_allocates_meta_on_untracked_tensor`

#![cfg(all(feature = "cuda", feature = "bf16_u16", feature = "autograd_v2"))]

use flame_core::autograd_v2::DispatchCtx;
use flame_core::{global_cuda_device, Device, Shape, Tensor};

fn make_f32(values: Vec<f32>, dims: &[usize]) -> Tensor {
    let device = global_cuda_device().clone();
    Tensor::from_vec(values, Shape::from_dims(dims), device).expect("from_vec")
}

fn default_ctx() -> DispatchCtx {
    let dev = Device::from_arc(global_cuda_device().clone());
    DispatchCtx::default_for(dev)
}

fn approx_eq(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(
        got.len(),
        want.len(),
        "length mismatch: {} vs {}",
        got.len(),
        want.len()
    );
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < tol,
            "elem {}: got {} want {} (tol {})",
            i,
            g,
            w,
            tol
        );
    }
}

// ---------------------------------------------------------------------------
// 1. add_v2 forward-mode JVP
// JVP: out_fw = a_dot + b_dot
// ---------------------------------------------------------------------------

#[test]
fn add_v2_forward_mode_jvp() {
    let mut a = make_f32(vec![1.0, 2.0, 3.0], &[3]);
    let mut b = make_f32(vec![10.0, 20.0, 30.0], &[3]);
    let a_dot = make_f32(vec![0.1, 0.2, 0.3], &[3]);
    let b_dot = make_f32(vec![1.0, 2.0, 3.0], &[3]);
    a.set_fw_grad(a_dot);
    b.set_fw_grad(b_dot);

    let ctx = default_ctx();
    let out = flame_core::autograd_v2::ops::add::add_v2(&a, &b, &ctx).expect("add_v2");

    let out_fw = out
        .fw_grad()
        .expect("fw_grad must be set after add_v2 with tangents");
    let got = out_fw.to_vec1::<f32>().expect("to_vec1");
    // Expected JVP: a_dot + b_dot = [1.1, 2.2, 3.3]
    approx_eq(&got, &[1.1, 2.2, 3.3], 1e-5);
}

// ---------------------------------------------------------------------------
// 2. mul_v2 forward-mode JVP (product rule)
// JVP: out_fw = a_dot * b + a * b_dot
// ---------------------------------------------------------------------------

#[test]
fn mul_v2_forward_mode_jvp_product_rule() {
    let mut a = make_f32(vec![1.0, 2.0, 3.0], &[3]);
    let mut b = make_f32(vec![10.0, 20.0, 30.0], &[3]);
    let a_dot = make_f32(vec![0.1, 0.2, 0.3], &[3]);
    let b_dot = make_f32(vec![1.0, 1.0, 1.0], &[3]);
    a.set_fw_grad(a_dot);
    b.set_fw_grad(b_dot);

    let ctx = default_ctx();
    let out = flame_core::autograd_v2::ops::mul::mul_v2(&a, &b, &ctx).expect("mul_v2");

    let out_fw = out.fw_grad().expect("fw_grad must be set");
    let got = out_fw.to_vec1::<f32>().expect("to_vec1");
    // Expected: a_dot * b + a * b_dot
    //   = [0.1*10 + 1*1, 0.2*20 + 2*1, 0.3*30 + 3*1]
    //   = [2.0, 6.0, 12.0]
    approx_eq(&got, &[2.0, 6.0, 12.0], 1e-5);
}

// ---------------------------------------------------------------------------
// 3. sum_v2 forward-mode JVP
// JVP: out_fw = sum(a_dot)
// ---------------------------------------------------------------------------

#[test]
fn sum_v2_forward_mode_jvp() {
    let mut a = make_f32(vec![1.0, 2.0, 3.0, 4.0], &[4]);
    let a_dot = make_f32(vec![0.5, 0.25, 0.125, 0.0625], &[4]);
    a.set_fw_grad(a_dot);

    let ctx = default_ctx();
    let out = flame_core::autograd_v2::ops::sum::sum_v2(&a, &ctx).expect("sum_v2");

    let out_fw = out.fw_grad().expect("fw_grad must be set");
    let got = out_fw.to_vec1::<f32>().expect("to_vec1");
    // Expected: sum(a_dot) = 0.5+0.25+0.125+0.0625 = 0.9375
    approx_eq(&got, &[0.9375], 1e-5);
}

// ---------------------------------------------------------------------------
// 4. matmul_v2 forward-mode JVP (product rule)
// JVP: out_fw = a_dot @ b + a @ b_dot
// ---------------------------------------------------------------------------

#[test]
fn matmul_v2_forward_mode_jvp_product_rule() {
    // 2x2 @ 2x2 case. Use identity-like a so the math is easy to verify.
    // a = [[1, 0], [0, 1]]    b = [[2, 3], [4, 5]]
    // a_dot = [[0.1, 0.0], [0.0, 0.1]]
    // b_dot = [[1, 0], [0, 1]]
    let mut a = make_f32(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let mut b = make_f32(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let a_dot = make_f32(vec![0.1, 0.0, 0.0, 0.1], &[2, 2]);
    let b_dot = make_f32(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    a.set_fw_grad(a_dot);
    b.set_fw_grad(b_dot);

    let ctx = default_ctx();
    let out = flame_core::autograd_v2::ops::matmul::matmul_v2(&a, &b, &ctx).expect("matmul_v2");

    let out_fw = out.fw_grad().expect("fw_grad must be set");
    let got = out_fw.to_vec2::<f32>().expect("to_vec2");
    let flat: Vec<f32> = got.into_iter().flatten().collect();
    // a_dot @ b = 0.1 * b = [[0.2, 0.3], [0.4, 0.5]]
    // a @ b_dot = I @ I = I = [[1, 0], [0, 1]]
    // sum = [[1.2, 0.3], [0.4, 1.5]]
    approx_eq(&flat, &[1.2, 0.3, 0.4, 1.5], 1e-5);
}

// ---------------------------------------------------------------------------
// 5. silu_v2 forward-mode JVP
// JVP: out_fw = x_dot * silu_deriv(x)
// where silu_deriv(x) = sigmoid(x) * (1 + x*(1 - sigmoid(x)))
// At x=0: sigmoid(0) = 0.5, so silu_deriv(0) = 0.5 * (1 + 0 * 0.5) = 0.5
// ---------------------------------------------------------------------------

#[test]
fn silu_v2_forward_mode_jvp() {
    let mut x = make_f32(vec![0.0, 0.0, 0.0], &[3]);
    let x_dot = make_f32(vec![1.0, 2.0, 4.0], &[3]);
    x.set_fw_grad(x_dot);

    let ctx = default_ctx();
    let out = flame_core::autograd_v2::ops::silu::silu_v2(&x, &ctx).expect("silu_v2");

    let out_fw = out.fw_grad().expect("fw_grad must be set");
    let got = out_fw.to_vec1::<f32>().expect("to_vec1");
    // At x=0: silu_deriv = 0.5. out_fw = x_dot * 0.5 = [0.5, 1.0, 2.0]
    approx_eq(&got, &[0.5, 1.0, 2.0], 1e-5);
}

// ---------------------------------------------------------------------------
// 6. reshape_v2 forward-mode JVP
// JVP: out_fw = a_dot.reshape(new_shape)
// ---------------------------------------------------------------------------

#[test]
fn reshape_v2_forward_mode_jvp() {
    let mut a = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let a_dot = make_f32(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3]);
    a.set_fw_grad(a_dot);

    let ctx = default_ctx();
    let out =
        flame_core::autograd_v2::ops::reshape::reshape_v2(&a, &[3, 2], &ctx).expect("reshape_v2");

    let out_fw = out.fw_grad().expect("fw_grad must be set");
    assert_eq!(out_fw.shape().dims(), &[3, 2]);
    let got = out_fw.to_vec2::<f32>().expect("to_vec2");
    let flat: Vec<f32> = got.into_iter().flatten().collect();
    // Reshape preserves storage order: same data, new shape.
    approx_eq(&flat, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], 1e-5);
}

// ---------------------------------------------------------------------------
// 7. view_v2 forward-mode JVP (alias of reshape)
// ---------------------------------------------------------------------------

#[test]
fn view_v2_forward_mode_jvp() {
    let mut a = make_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let a_dot = make_f32(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]);
    a.set_fw_grad(a_dot);

    let ctx = default_ctx();
    let out = flame_core::autograd_v2::ops::reshape::view_v2(&a, &[4], &ctx).expect("view_v2");

    let out_fw = out.fw_grad().expect("fw_grad must be set");
    assert_eq!(out_fw.shape().dims(), &[4]);
    let got = out_fw.to_vec1::<f32>().expect("to_vec1");
    approx_eq(&got, &[0.1, 0.2, 0.3, 0.4], 1e-5);
}

// ---------------------------------------------------------------------------
// 8. transpose_v2 forward-mode JVP
// JVP: out_fw = a_dot.transpose().contiguous()
// ---------------------------------------------------------------------------

#[test]
fn transpose_v2_forward_mode_jvp() {
    // 2x3 -> 3x2.
    let mut a = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let a_dot = make_f32(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3]);
    a.set_fw_grad(a_dot);

    let ctx = default_ctx();
    let out =
        flame_core::autograd_v2::ops::transpose::transpose_v2(&a, &ctx).expect("transpose_v2");

    let out_fw = out.fw_grad().expect("fw_grad must be set");
    assert_eq!(out_fw.shape().dims(), &[3, 2]);
    let got = out_fw.to_vec2::<f32>().expect("to_vec2");
    let flat: Vec<f32> = got.into_iter().flatten().collect();
    // Transpose of [[10,20,30],[40,50,60]] = [[10,40],[20,50],[30,60]]
    approx_eq(&flat, &[10.0, 40.0, 20.0, 50.0, 30.0, 60.0], 1e-5);
}

// ---------------------------------------------------------------------------
// 9. narrow_v2 forward-mode JVP
// JVP: out_fw = a_dot.narrow(dim, start, length)
// ---------------------------------------------------------------------------

#[test]
fn narrow_v2_forward_mode_jvp() {
    // Shape [4]; narrow dim=0 start=1 length=2.
    let mut a = make_f32(vec![1.0, 2.0, 3.0, 4.0], &[4]);
    let a_dot = make_f32(vec![10.0, 20.0, 30.0, 40.0], &[4]);
    a.set_fw_grad(a_dot);

    let ctx = default_ctx();
    let out =
        flame_core::autograd_v2::ops::narrow::narrow_v2(&a, 0, 1, 2, &ctx).expect("narrow_v2");

    let out_fw = out.fw_grad().expect("fw_grad must be set");
    assert_eq!(out_fw.shape().dims(), &[2]);
    // Narrow returns a view; materialise contiguous before reading.
    let got = out_fw
        .contiguous()
        .expect("contiguous")
        .to_vec1::<f32>()
        .expect("to_vec1");
    // a_dot.narrow(0, 1, 2) = [20.0, 30.0]
    approx_eq(&got, &[20.0, 30.0], 1e-5);
}

// ---------------------------------------------------------------------------
// 10. squeeze_v2 forward-mode JVP
// JVP: out_fw = a_dot.squeeze(Some(dim))
// ---------------------------------------------------------------------------

#[test]
fn squeeze_v2_forward_mode_jvp() {
    // Shape [1, 3] → squeeze dim 0 → [3].
    let mut a = make_f32(vec![1.0, 2.0, 3.0], &[1, 3]);
    let a_dot = make_f32(vec![10.0, 20.0, 30.0], &[1, 3]);
    a.set_fw_grad(a_dot);

    let ctx = default_ctx();
    let out = flame_core::autograd_v2::ops::squeeze::squeeze_v2(&a, 0, &ctx).expect("squeeze_v2");

    let out_fw = out.fw_grad().expect("fw_grad must be set");
    assert_eq!(out_fw.shape().dims(), &[3]);
    let got = out_fw.to_vec1::<f32>().expect("to_vec1");
    approx_eq(&got, &[10.0, 20.0, 30.0], 1e-5);
}

// ---------------------------------------------------------------------------
// 11. unsqueeze_v2 forward-mode JVP
// JVP: out_fw = a_dot.unsqueeze(dim)
// ---------------------------------------------------------------------------

#[test]
fn unsqueeze_v2_forward_mode_jvp() {
    // Shape [3] → unsqueeze dim 0 → [1, 3].
    let mut a = make_f32(vec![1.0, 2.0, 3.0], &[3]);
    let a_dot = make_f32(vec![10.0, 20.0, 30.0], &[3]);
    a.set_fw_grad(a_dot);

    let ctx = default_ctx();
    let out =
        flame_core::autograd_v2::ops::unsqueeze::unsqueeze_v2(&a, 0, &ctx).expect("unsqueeze_v2");

    let out_fw = out.fw_grad().expect("fw_grad must be set");
    assert_eq!(out_fw.shape().dims(), &[1, 3]);
    let got = out_fw.to_vec2::<f32>().expect("to_vec2");
    let flat: Vec<f32> = got.into_iter().flatten().collect();
    approx_eq(&flat, &[10.0, 20.0, 30.0], 1e-5);
}

// ---------------------------------------------------------------------------
// 12. permute_v2 forward-mode JVP
// JVP: out_fw = a_dot.permute(perm).contiguous()
// ---------------------------------------------------------------------------

#[test]
fn permute_v2_forward_mode_jvp() {
    // Shape [2, 3] → permute [1, 0] → [3, 2].
    let mut a = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let a_dot = make_f32(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3]);
    a.set_fw_grad(a_dot);

    let ctx = default_ctx();
    let out =
        flame_core::autograd_v2::ops::permute::permute_v2(&a, &[1, 0], &ctx).expect("permute_v2");

    let out_fw = out.fw_grad().expect("fw_grad must be set");
    assert_eq!(out_fw.shape().dims(), &[3, 2]);
    let got = out_fw.to_vec2::<f32>().expect("to_vec2");
    let flat: Vec<f32> = got.into_iter().flatten().collect();
    // permute([1,0]) on a 2x3 == transpose: [[10,40],[20,50],[30,60]].
    approx_eq(&flat, &[10.0, 40.0, 20.0, 50.0, 30.0, 60.0], 1e-5);
}

// ---------------------------------------------------------------------------
// 13. set_fw_grad implicitly allocates meta on untracked tensor
// PyTorch parity: setting _fw_grad on a tensor without autograd_meta
// allocates one (with requires_grad=false — backward-mode tracking is
// NOT silently enabled).
// ---------------------------------------------------------------------------

#[test]
fn set_fw_grad_implicitly_allocates_meta_on_untracked_tensor() {
    let mut a = make_f32(vec![1.0, 2.0], &[2]);
    assert!(a.autograd_meta().is_none(), "fresh tensor has no meta");
    assert!(a.fw_grad().is_none(), "fresh tensor has no fw_grad");

    let a_dot = make_f32(vec![0.1, 0.2], &[2]);
    a.set_fw_grad(a_dot);

    // After set_fw_grad, meta is allocated; fw_grad is set.
    assert!(
        a.autograd_meta().is_some(),
        "set_fw_grad must implicitly allocate meta"
    );
    let g = a.fw_grad().expect("fw_grad is set");
    let got = g.to_vec1::<f32>().expect("to_vec1");
    approx_eq(&got, &[0.1, 0.2], 1e-5);

    // But requires_grad MUST remain false — setting fw_grad does not
    // silently enable backward-mode recording.
    let m = a.autograd_meta().unwrap().lock().unwrap();
    assert!(
        !m.requires_grad,
        "set_fw_grad on an untracked tensor must NOT enable requires_grad"
    );
}
