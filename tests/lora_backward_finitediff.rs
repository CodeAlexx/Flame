//! Minimal finite-diff check on a LoRA-linear micro-computation.
//!
//! Builds: `out = x @ (A^T) @ (B^T) * scale`; loss = (out - target)^2.mean().
//! Computes analytical grads via `loss.backward()`, then perturbs A[0,0]
//! and B[0,0] by ±eps and computes central finite-diff grads. Prints
//! relative error; test fails if any probe disagrees by > 5%.
//!
//! Everything here runs in F32 on tiny shapes to eliminate BF16 noise
//! and fused-kernel issues as explanations. If THIS test fails, the
//! core autograd wiring (matmul backward, saved-tensors, Cast paths, or
//! Op::MatMul gradient formulas) is broken.
#![cfg(feature = "cuda")]

use flame_core::{
    autograd::AutogradContext, global_cuda_device, parameter::Parameter, DType, Result, Shape,
    Tensor,
};

fn rel_err(a: f32, b: f32) -> f32 {
    let m = a.abs().max(b.abs()).max(1e-8);
    (a - b).abs() / m
}

fn loss_fn(x: &Tensor, a: &Tensor, b: &Tensor, target: &Tensor, scale: f32) -> Result<Tensor> {
    // x: [N, in]; A: [rank, in]; B: [out, rank]; target: [N, out]
    //
    // .transpose() returns a strided VIEW. Test without caller-side
    // contiguify — relies on flame-core Op::MatMul backward to handle
    // strided saved tensors correctly.
    let at = a.transpose()?;
    let bt = b.transpose()?;
    let mid = x.matmul(&at)?;
    let out = mid.matmul(&bt)?.mul_scalar(scale)?;
    let diff = out.sub(target)?;
    diff.mul(&diff)?.mean()
}

fn eval_loss_no_grad(
    x: &Tensor,
    a_data: &Tensor,
    b_data: &Tensor,
    target: &Tensor,
    scale: f32,
) -> Result<f32> {
    let _g = AutogradContext::no_grad();
    let l = loss_fn(x, a_data, b_data, target, scale)?;
    Ok(l.to_vec()?[0])
}

#[test]
fn lora_backward_matches_finite_diff_f32() -> Result<()> {
    let device = global_cuda_device();
    // Small shape, F32 throughout — no BF16, no fused kernels.
    let n = 8usize;
    let in_feat = 16usize;
    let rank = 4usize;
    let out_feat = 12usize;
    let scale = 0.5f32;

    // Deterministic inputs via from_vec + hashing.
    let mk = |shape: &[usize], seed: u64, req: bool| -> Result<Tensor> {
        let numel: usize = shape.iter().product();
        let mut v = Vec::with_capacity(numel);
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
        for _ in 0..numel {
            s = s.wrapping_mul(0x5851F42D4C957F2D).wrapping_add(1);
            let f = ((s >> 32) as i32 as f32) / (i32::MAX as f32);
            v.push(f * 0.3);
        }
        let t = Tensor::from_vec(v, Shape::from_dims(shape), device.clone())?;
        Ok(if req { t.requires_grad_(true) } else { t })
    };

    let x = mk(&[n, in_feat], 1, false)?;
    let target = mk(&[n, out_feat], 2, false)?;

    let a_param = Parameter::new(mk(&[rank, in_feat], 3, true)?);
    let b_param = Parameter::new(mk(&[out_feat, rank], 4, true)?);

    // Analytical
    let a_t = a_param.tensor()?;
    let b_t = b_param.tensor()?;
    let loss_val = loss_fn(&x, &a_t, &b_t, &target, scale)?;
    let baseline = loss_val.to_vec()?[0];
    println!("baseline loss = {baseline:.6}");

    let grads = loss_val.backward()?;
    let grad_a = grads.get(a_t.id()).expect("grad_a missing");
    let grad_b = grads.get(b_t.id()).expect("grad_b missing");
    let grad_a_data = grad_a.to_vec()?;
    let grad_b_data = grad_b.to_vec()?;

    let eps = 1e-3f32;
    let perturb = |orig: &Tensor, idx: usize, delta: f32| -> Result<Tensor> {
        let mut d = orig.to_vec()?;
        d[idx] += delta;
        Tensor::from_vec(d, orig.shape().clone(), device.clone())
    };

    // Probe several elements
    let probes_a = [(0, 0), (1, 3), (rank - 1, in_feat - 1)];
    let probes_b = [(0, 0), (3, 1), (out_feat - 1, rank - 1)];

    println!("\n=== A probes ===");
    let mut worst: f32 = 0.0;
    for (i, j) in probes_a {
        let idx = i * in_feat + j;
        let analytical = grad_a_data[idx];
        let a_plus = perturb(&a_t, idx, eps)?;
        let a_minus = perturb(&a_t, idx, -eps)?;
        let lp = eval_loss_no_grad(&x, &a_plus, &b_t, &target, scale)?;
        let lm = eval_loss_no_grad(&x, &a_minus, &b_t, &target, scale)?;
        let fd = (lp - lm) / (2.0 * eps);
        let re = rel_err(analytical, fd);
        worst = worst.max(re);
        let tag = if re > 0.05 { "BAD" } else { "ok" };
        println!("  A[{i},{j}] analytical={analytical:+.6e} fd={fd:+.6e} rel_err={re:.4} {tag}");
    }

    println!("\n=== B probes ===");
    for (i, j) in probes_b {
        let idx = i * rank + j;
        let analytical = grad_b_data[idx];
        let b_plus = perturb(&b_t, idx, eps)?;
        let b_minus = perturb(&b_t, idx, -eps)?;
        let lp = eval_loss_no_grad(&x, &a_t, &b_plus, &target, scale)?;
        let lm = eval_loss_no_grad(&x, &a_t, &b_minus, &target, scale)?;
        let fd = (lp - lm) / (2.0 * eps);
        let re = rel_err(analytical, fd);
        worst = worst.max(re);
        let tag = if re > 0.05 { "BAD" } else { "ok" };
        println!("  B[{i},{j}] analytical={analytical:+.6e} fd={fd:+.6e} rel_err={re:.4} {tag}");
    }

    println!("\nworst relative error = {worst:.4}");
    assert!(worst < 0.05, "autograd produces wrong LoRA gradients (worst rel_err={worst:.4})");
    Ok(())
}
