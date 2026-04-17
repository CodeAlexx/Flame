//! Phase 3 numerical parity: `Linear::forward` / `Op::Linear` backward now
//! route through `ops::gemm_bf16::matmul_bf16_trans` (fused cuBLASLt with
//! `TRANSB=T`), replacing the old `transpose2d_bf16` + `matmul` path.
//!
//! This test constructs a reference output using the old unfused sequence
//! (`transpose2d_bf16(weight_2d)` → `input_2d.matmul(weight_t)` → bias add)
//! and compares it against `Linear::forward` / autograd backward.
//!
//! Gate (same shape as FA2 parity tests):
//!   * max_abs ≤ 1e-2 — BF16 noise floor (BF16 eps ≈ 2^-7 ≈ 7.8e-3).
//!   * cos_sim  ≥ 0.9999 — direction match of the flattened vectors.
//!
//! `max_rel` is deliberately NOT checked: relative error is meaningless for
//! BF16 outputs near zero.

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use cudarc::driver::CudaDevice;
use flame_core::bf16_elementwise::transpose2d_bf16;
use flame_core::{linear::Linear, AutogradContext, DType, Shape, Tensor};
use std::sync::Arc;

const ABS_TOL: f32 = 1e-2;
const COS_TOL: f32 = 0.9999;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect(
        "CUDA GPU required. Set CUDA_HOME=/usr/local/cuda and export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH.",
    )
}

/// Deterministic pseudo-random `f32` stream. Tiny LCG; only needs to avoid
/// trivial patterns, not be statistically nice.
fn pseudo_stream(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map to roughly [-0.5, 0.5).
        let u = ((state >> 32) as u32) as f32 / (u32::MAX as f32);
        out.push(u - 0.5);
    }
    out
}

fn bf16_tensor_from_vec(dev: &Arc<CudaDevice>, shape: &[usize], data: Vec<f32>) -> Tensor {
    Tensor::from_vec_dtype(
        data,
        Shape::from_dims(shape),
        dev.clone(),
        DType::F32,
    )
    .expect("from_vec_dtype f32")
    .to_dtype(DType::BF16)
    .expect("cast to bf16")
}

fn max_abs_and_cos(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let mut max_abs = 0f32;
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    let eps = 1e-12;
    let cos = (dot / (na.sqrt() * nb.sqrt() + eps)) as f32;
    (max_abs, cos)
}

fn report(tag: &str, a: &Tensor, b: &Tensor) {
    let av = a.to_vec_f32().expect("a to_vec_f32");
    let bv = b.to_vec_f32().expect("b to_vec_f32");
    let (max_abs, cos) = max_abs_and_cos(&av, &bv);
    assert!(
        max_abs <= ABS_TOL && cos >= COS_TOL,
        "{tag}: max_abs={:.3e} cos={:.6} (need abs ≤ {:.1e}, cos ≥ {:.4})",
        max_abs,
        cos,
        ABS_TOL,
        COS_TOL
    );
    eprintln!("  {tag}: max_abs={:.3e} cos={:.6} OK", max_abs, cos);
}

/// Reference forward: `output = input @ weight^T + bias` using the old
/// `transpose2d_bf16` + `matmul` sequence. This is what `Linear::forward`
/// used to do before Phase 3.
fn reference_forward(
    input: &Tensor,
    weight_2d: &Tensor,
    bias: Option<&Tensor>,
) -> Tensor {
    let input_shape = input.shape().dims().to_vec();
    let in_features = input_shape[input_shape.len() - 1];
    let out_features = weight_2d.shape().dims()[0];
    let batch: usize = input_shape[..input_shape.len() - 1].iter().product();
    let input_2d = input.reshape(&[batch, in_features]).unwrap();
    let weight_t = transpose2d_bf16(weight_2d).expect("transpose2d_bf16 ref");
    let mut out = input_2d.matmul(&weight_t).expect("matmul ref");
    if let Some(b) = bias {
        let bv = b.reshape(&[1, out_features]).unwrap();
        out = out.add(&bv).expect("bias add ref");
    }
    let mut out_shape = input_shape[..input_shape.len() - 1].to_vec();
    out_shape.push(out_features);
    out.reshape(&out_shape).unwrap()
}

#[test]
fn linear_forward_matmul_bf16_trans_parity() {
    let dev = cuda_device();
    let in_features = 32;
    let out_features = 48;
    let batch = 8;

    let mut linear = Linear::new(in_features, out_features, true, &dev).expect("linear");

    // Deterministic weights + bias via copy_weight_from / copy_bias_from.
    let w_data = pseudo_stream(0xC0FFEE, out_features * in_features);
    let weight = bf16_tensor_from_vec(&dev, &[out_features, in_features], w_data);
    linear.copy_weight_from(&weight).expect("copy weight");
    let b_data = pseudo_stream(0xBADF00D, out_features);
    let bias = bf16_tensor_from_vec(&dev, &[out_features], b_data);
    linear.copy_bias_from(&bias).expect("copy bias");

    // Deterministic input.
    let x_data = pseudo_stream(0x1234_5678, batch * in_features);
    let input = bf16_tensor_from_vec(&dev, &[batch, in_features], x_data);

    // Reference: old unfused path directly against the un-transposed weight.
    let weight_2d = linear
        .weight
        .reshape(&[out_features, in_features])
        .expect("reshape w");
    let y_ref = reference_forward(&input, &weight_2d, Some(linear.bias.as_ref().unwrap()));

    // New path: Linear::forward (goes through matmul_bf16_trans).
    let y_new = linear.forward(&input).expect("forward");
    assert_eq!(y_new.shape().dims(), &[batch, out_features]);

    report("forward_2d", &y_new, &y_ref);
}

#[test]
fn linear_forward_matmul_bf16_trans_parity_3d() {
    let dev = cuda_device();
    let in_features = 24;
    let out_features = 16;
    let batch = 2;
    let seq = 5;

    let mut linear = Linear::new(in_features, out_features, false, &dev).expect("linear");
    let w_data = pseudo_stream(0xDEAD_BEEF, out_features * in_features);
    let weight = bf16_tensor_from_vec(&dev, &[out_features, in_features], w_data);
    linear.copy_weight_from(&weight).expect("copy weight");

    let x_data = pseudo_stream(0xFEED, batch * seq * in_features);
    let input = bf16_tensor_from_vec(&dev, &[batch, seq, in_features], x_data);

    let weight_2d = linear
        .weight
        .reshape(&[out_features, in_features])
        .expect("reshape w");
    let y_ref = reference_forward(&input, &weight_2d, None);
    let y_new = linear.forward(&input).expect("forward 3d");
    assert_eq!(y_new.shape().dims(), &[batch, seq, out_features]);

    report("forward_3d", &y_new, &y_ref);
}

#[test]
fn linear_backward_matmul_bf16_trans_parity() {
    let dev = cuda_device();
    let in_features = 32;
    let out_features = 48;
    let batch = 8;

    // --- New path: build an autograd-enabled Linear and backprop through it.
    AutogradContext::reset();

    let mut linear = Linear::new(in_features, out_features, true, &dev).expect("linear");
    let w_data = pseudo_stream(0xC0FFEE, out_features * in_features);
    let weight = bf16_tensor_from_vec(&dev, &[out_features, in_features], w_data);
    linear.copy_weight_from(&weight).expect("copy weight");
    let b_data = pseudo_stream(0xBADF00D, out_features);
    let bias = bf16_tensor_from_vec(&dev, &[out_features], b_data);
    linear.copy_bias_from(&bias).expect("copy bias");

    let x_data = pseudo_stream(0x1234_5678, batch * in_features);
    let input = bf16_tensor_from_vec(&dev, &[batch, in_features], x_data).requires_grad_(true);

    // Deterministic "upstream" weights for the loss scalar so grad_output is non-trivial.
    let tgt_data = pseudo_stream(0xCAFE, batch * out_features);
    let target = bf16_tensor_from_vec(&dev, &[batch, out_features], tgt_data);

    // loss = sum(output * target). grad_output = target (elementwise).
    let output = linear.forward(&input).expect("forward bwd");
    let loss = output.mul(&target).expect("mul").sum().expect("sum");

    let weight_id = linear.weight.id();
    let input_id = input.id();
    let gradients = AutogradContext::backward(&loss).expect("backward");
    let grad_input_new = gradients.get_public_grad(input_id).expect("grad input");
    let grad_weight_new = gradients.get_public_grad(weight_id).expect("grad weight");
    AutogradContext::clear();

    // --- Reference: compute the same grads manually with the OLD unfused path.
    //   grad_input  = grad_output @ weight
    //   grad_weight = grad_output^T @ input  (then NO extra transpose — desired layout)
    //
    // In the old backward, this was expressed as
    //   grad_weight = (grad_output^T @ input^T)^T
    //                = transpose(transpose(grad_output) @ transpose(input))
    // which we avoid here; the simpler equivalent is:
    //   grad_weight = transpose(input^T @ grad_output)
    //                = transpose(grad_output^T @ input)^T   [commutes]
    // Easiest: compute (batch x out)^T @ (batch x in) via materialized transposes.
    let weight_2d_ref = linear
        .weight
        .reshape(&[out_features, in_features])
        .expect("reshape w ref");
    let grad_output_bf16 = target.clone(); // grad of sum(y * t) wrt y is t

    // grad_input_ref = grad_output @ weight  (matmul: [B,out] @ [out,in] = [B,in])
    let grad_input_ref = grad_output_bf16
        .matmul(&weight_2d_ref)
        .expect("ref grad_input");

    // grad_weight_ref: [out, in] via materialized transpose fallback.
    //   grad_output.transpose() -> [out, B]
    //   input.transpose()       -> [in, B]
    //   (grad^T @ input^T)^T    -> [out, in]
    //     or simpler: grad^T @ input  -> [out, in]  directly (matmul [out,B] @ [B,in])
    let input_tensor = Tensor::from_vec_dtype(
        pseudo_stream(0x1234_5678, batch * in_features),
        Shape::from_dims(&[batch, in_features]),
        dev.clone(),
        DType::F32,
    )
    .unwrap()
    .to_dtype(DType::BF16)
    .unwrap();
    let grad_out_t = grad_output_bf16.transpose().expect("grad_out^T");
    let grad_weight_ref = grad_out_t.matmul(&input_tensor).expect("ref grad_weight");

    report("backward_grad_input", &grad_input_new, &grad_input_ref);
    report("backward_grad_weight", &grad_weight_new, &grad_weight_ref);
}
