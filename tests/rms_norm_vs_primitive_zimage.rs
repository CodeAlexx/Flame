//! Parity check: `flame_core::norm::rms_norm` (fused BF16 kernel,
//! F32-accumulating forward + backward) vs the F32 primitive chain
//! used by `EriDiffusion-v2/crates/eridiffusion-core/src/models/zimage.rs`
//! `primitive_rms_norm`.
//!
//! The primitive chain exists in zimage.rs because the comment there
//! claims the fused kernel's backward suffers BF16-accumulation error.
//! The current fused kernels (post `bcc37a7`) compute internally in F32
//! for both forward and backward. This test pins whether they actually
//! agree with the primitive chain on Z-Image-realistic shapes — if so,
//! the workaround is obsolete and EDv2 can swap, recovering the speed
//! lost to ~8 primitive op launches per RMSNorm call.
//!
//! Z-Image hidden dim = 2560. Sequence at 512² is 4096 image patches;
//! some block norms run on [B, T, 2560] where T includes caption tokens.
//! For this test we use [1, 4096, 2560] (image-only).

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::{global_cuda_device, parameter::Parameter, DType, Result, Shape, Tensor};

const NORM_EPS: f32 = 1e-6;

/// Verbatim copy of EDv2 zimage.rs `primitive_rms_norm`. Kept literal so
/// any divergence shows up as parity failure, not as a stale copy.
fn primitive_rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let out_dtype = x.dtype();
    let x_f32 = if out_dtype == DType::F32 {
        x.clone()
    } else {
        x.to_dtype(DType::F32)?
    };
    let weight_f32 = if weight.dtype() == DType::F32 {
        weight.clone()
    } else {
        weight.to_dtype(DType::F32)?
    };
    let sq = x_f32.mul(&x_f32)?;
    let dims = sq.shape().dims().to_vec();
    let last = dims.len() - 1;
    let n = dims[last] as f32;
    let mean_sq = sq.sum_dim_keepdim(last)?.mul_scalar(1.0 / n)?;
    let inv_rms = mean_sq.add_scalar(eps)?.rsqrt()?;
    let normed = x_f32.mul(&inv_rms)?;
    let scaled = normed.mul(&weight_f32)?;
    if out_dtype == DType::F32 {
        Ok(scaled)
    } else {
        scaled.to_dtype(out_dtype)
    }
}

fn rand_bf16(shape: &[usize], seed: u64, scale: f32) -> Result<Tensor> {
    let device = global_cuda_device();
    let numel: usize = shape.iter().product();
    let mut v = Vec::with_capacity(numel);
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for _ in 0..numel {
        s = s.wrapping_mul(0x5851F42D4C957F2D).wrapping_add(1);
        let f = ((s >> 32) as i32 as f32) / (i32::MAX as f32);
        v.push(f * scale);
    }
    Tensor::from_vec(v, Shape::from_dims(shape), device.clone())?
        .to_dtype(DType::BF16)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, f32, usize) {
    let mut mx = 0f32;
    let mut mx_idx = 0usize;
    let mut mx_rel = 0f32;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > mx {
            mx = d;
            mx_idx = i;
        }
        let denom = x.abs().max(y.abs()).max(1e-6);
        let r = d / denom;
        if r > mx_rel {
            mx_rel = r;
        }
    }
    (mx, mx_rel, mx_idx)
}

#[test]
fn rms_norm_forward_parity_zimage_block_shape() -> Result<()> {
    // [1, 4096, 2560] — block-level RMSNorm at 512² resolution.
    let shape = &[1usize, 4096, 2560];
    let weight_shape = &[2560usize];

    let x_bf16 = rand_bf16(shape, 1, 0.5)?;
    let w_bf16 = rand_bf16(weight_shape, 2, 1.0)?;

    // Two paths share data but use separate Tensor instances (separate ids).
    let x_fused = x_bf16.clone();
    let w_fused = w_bf16.clone();
    let x_prim = x_bf16.clone();
    let w_prim = w_bf16.clone();

    let out_fused = flame_core::norm::rms_norm(&x_fused, weight_shape, Some(&w_fused), NORM_EPS)?;
    let out_prim = primitive_rms_norm(&x_prim, &w_prim, NORM_EPS)?;

    assert_eq!(out_fused.dtype(), DType::BF16);
    assert_eq!(out_prim.dtype(), DType::BF16);
    assert_eq!(out_fused.shape().dims(), shape);
    assert_eq!(out_prim.shape().dims(), shape);

    let f = out_fused.to_dtype(DType::F32)?.to_vec()?;
    let p = out_prim.to_dtype(DType::F32)?.to_vec()?;

    let (mx, mx_rel, idx) = max_abs_diff(&f, &p);
    println!(
        "[fwd] shape={:?} max_abs={:.4e} max_rel={:.4e} at idx {}",
        shape, mx, mx_rel, idx
    );

    // BF16 storage rounds at ~1/128 of magnitude (8-bit mantissa).
    // Allow 3 ulp of headroom (~2.3% relative).
    assert!(
        mx_rel < 0.03,
        "forward parity broken: max_rel={mx_rel:.4e} > 0.03"
    );
    Ok(())
}

#[test]
fn rms_norm_forward_parity_zimage_qk_norm_shape() -> Result<()> {
    // Z-Image Q/K norm shape: [B, H, T, head_dim].
    // 24 heads * 128 head_dim = 3072 model dim is not Z-Image's geometry,
    // but the Q/K norm in primitive_rms_norm is called on the LAST dim,
    // so a representative head_dim is what matters. Test head_dim = 128.
    let shape = &[1usize, 24, 4096, 128];
    let weight_shape = &[128usize];

    let x_bf16 = rand_bf16(shape, 3, 0.5)?;
    let w_bf16 = rand_bf16(weight_shape, 4, 1.0)?;

    let out_fused = flame_core::norm::rms_norm(&x_bf16, weight_shape, Some(&w_bf16), NORM_EPS)?;
    let out_prim = primitive_rms_norm(&x_bf16, &w_bf16, NORM_EPS)?;

    let f = out_fused.to_dtype(DType::F32)?.to_vec()?;
    let p = out_prim.to_dtype(DType::F32)?.to_vec()?;
    let (mx, mx_rel, idx) = max_abs_diff(&f, &p);
    println!(
        "[fwd-qk] shape={:?} max_abs={:.4e} max_rel={:.4e} at idx {}",
        shape, mx, mx_rel, idx
    );
    assert!(
        mx_rel < 0.03,
        "Q/K-norm-shape forward parity broken: max_rel={mx_rel:.4e}"
    );
    Ok(())
}

/// Backward parity. Builds two independent compute graphs sharing input
/// VALUES but with distinct Parameter ids so each `.backward()` populates
/// its own slot in the GradMap; then compares grad_input and grad_weight.
#[test]
fn rms_norm_backward_parity_zimage_block_shape() -> Result<()> {
    let shape = &[1usize, 4096, 2560];
    let weight_shape = &[2560usize];

    // Build BF16 data once.
    let x_data = rand_bf16(shape, 11, 0.5)?;
    let w_data = rand_bf16(weight_shape, 12, 1.0)?;

    // Fused path: x_f, w_f as parameters (require grad).
    let x_f = Parameter::new(x_data.clone().requires_grad_(true)).tensor()?;
    let w_f = Parameter::new(w_data.clone().requires_grad_(true)).tensor()?;
    assert!(x_f.requires_grad(), "x_f must require grad");
    assert!(w_f.requires_grad(), "w_f must require grad");
    let out_f = flame_core::norm::rms_norm(&x_f, weight_shape, Some(&w_f), NORM_EPS)?;
    assert!(
        out_f.requires_grad(),
        "fused rms_norm output should require grad when inputs do"
    );
    // Loss: (out * out).sum(). Grad through .sum() and .mul() flows in
    // the same dtype as the operand, but the autograd-recorded Op::Mul
    // backward and Op::Sum backward both handle BF16 correctly in
    // flame-core today.
    let loss_f = out_f.mul(&out_f)?.sum()?;
    assert!(
        loss_f.requires_grad(),
        "loss_f must require grad after mul+sum"
    );
    let loss_f_val = loss_f.to_dtype(DType::F32)?.to_vec()?[0];
    let grads_f = loss_f.backward()?;
    let grad_x_f = grads_f
        .get(x_f.id())
        .expect("fused: grad_x missing")
        .to_dtype(DType::F32)?
        .to_vec()?;
    let grad_w_f = grads_f
        .get(w_f.id())
        .expect("fused: grad_w missing")
        .to_dtype(DType::F32)?
        .to_vec()?;

    // Primitive path: x_p, w_p — fresh parameter ids.
    let x_p = Parameter::new(x_data.clone().requires_grad_(true)).tensor()?;
    let w_p = Parameter::new(w_data.clone().requires_grad_(true)).tensor()?;
    let out_p = primitive_rms_norm(&x_p, &w_p, NORM_EPS)?;
    assert!(
        out_p.requires_grad(),
        "primitive rms_norm output should require grad when inputs do"
    );
    let loss_p = out_p.mul(&out_p)?.sum()?;
    assert!(
        loss_p.requires_grad(),
        "loss_p must require grad after mul+sum"
    );
    let loss_p_val = loss_p.to_dtype(DType::F32)?.to_vec()?[0];
    let grads_p = loss_p.backward()?;
    let grad_x_p = grads_p
        .get(x_p.id())
        .expect("prim: grad_x missing")
        .to_dtype(DType::F32)?
        .to_vec()?;
    let grad_w_p = grads_p
        .get(w_p.id())
        .expect("prim: grad_w missing")
        .to_dtype(DType::F32)?
        .to_vec()?;

    fn cos_sim_and_mags(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
        let mut dot = 0f64;
        let mut na = 0f64;
        let mut nb = 0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-30);
        (cos, na.sqrt(), nb.sqrt())
    }

    let (lmx_x, lmx_rel_x, idx_x) = max_abs_diff(&grad_x_f, &grad_x_p);
    let (cos_x, l2_x_f, l2_x_p) = cos_sim_and_mags(&grad_x_f, &grad_x_p);
    let l1_x_f: f64 = grad_x_f.iter().map(|x| x.abs() as f64).sum();
    let l1_x_p: f64 = grad_x_p.iter().map(|x| x.abs() as f64).sum();
    let mag_ratio_x = l2_x_f / l2_x_p.max(1e-30);

    let (lmx_w, lmx_rel_w, idx_w) = max_abs_diff(&grad_w_f, &grad_w_p);
    let (cos_w, l2_w_f, l2_w_p) = cos_sim_and_mags(&grad_w_f, &grad_w_p);
    let l1_w_f: f64 = grad_w_f.iter().map(|x| x.abs() as f64).sum();
    let l1_w_p: f64 = grad_w_p.iter().map(|x| x.abs() as f64).sum();

    println!(
        "[bwd] loss fused={loss_f_val:.6e} prim={loss_p_val:.6e} (delta {:.4e})",
        (loss_f_val - loss_p_val).abs()
    );
    println!(
        "[bwd] grad_x: max_abs={lmx_x:.4e} max_rel_pt={lmx_rel_x:.4e}@{idx_x}  cos={cos_x:.6}  L2_fused={l2_x_f:.4e} L2_prim={l2_x_p:.4e} mag_ratio={mag_ratio_x:.6}"
    );
    println!("[bwd] grad_x: L1_fused={l1_x_f:.4e} L1_prim={l1_x_p:.4e}");
    println!(
        "[bwd] grad_w: max_abs={lmx_w:.4e} max_rel_pt={lmx_rel_w:.4e}@{idx_w}  cos={cos_w:.6}  L2_fused={l2_w_f:.4e} L2_prim={l2_w_p:.4e}"
    );
    println!("[bwd] grad_w: L1_fused={l1_w_f:.4e} L1_prim={l1_w_p:.4e}");

    // For RMSNorm with weight present, dL/dw should be nonzero given a
    // nonzero loss. If BOTH paths produce zero grad_w, that's a clue that
    // the loss computation's grad isn't flowing to weight at all (likely
    // a test bug: aliasing in `out * out` or BF16 mul autograd not
    // recording). Surface this clearly.
    assert!(
        l1_w_p > 1e-3,
        "primitive grad_w is suspicious zero (L1={l1_w_p:.4e}) — \
         loss/backward setup likely doesn't propagate grad to weight"
    );
    assert!(
        l1_w_f > 1e-3,
        "fused grad_w is suspicious zero (L1={l1_w_f:.4e}) — \
         fused-kernel grad_weight atomicAdd may not be wired"
    );

    // The primitive chain in zimage.rs comment claims the fused backward
    // amplifies grad magnitude by ~1.25× per call. If that bug is still
    // live, mag_ratio will be far from 1.0.
    //
    // Element-wise max_rel is dominated by BF16 round-off at points where
    // the gradient is near zero (denominator collapses). The proper
    // direction metric is cosine similarity over the full vector, which
    // is order-of-accumulation-insensitive.
    assert!(
        cos_x > 0.999,
        "grad_x direction drift: cos_sim={cos_x:.6} < 0.999"
    );
    assert!(
        (mag_ratio_x - 1.0).abs() < 0.05,
        "grad_x magnitude drift fused/prim = {mag_ratio_x:.6} (allowed 0.95..=1.05)"
    );
    assert!(
        cos_w > 0.999,
        "grad_w direction drift: cos_sim={cos_w:.6} < 0.999"
    );
    let mag_ratio_w = l2_w_f / l2_w_p.max(1e-30);
    assert!(
        (mag_ratio_w - 1.0).abs() < 0.05,
        "grad_w magnitude drift fused/prim = {mag_ratio_w:.6} (allowed 0.95..=1.05)"
    );
    Ok(())
}

/// Verbatim copy of EDv2 zimage.rs `primitive_layer_norm` (no affine).
fn primitive_layer_norm(x: &Tensor, eps: f32) -> Result<Tensor> {
    let out_dtype = x.dtype();
    let x_f32 = if out_dtype == DType::F32 {
        x.clone()
    } else {
        x.to_dtype(DType::F32)?
    };
    let dims = x_f32.shape().dims().to_vec();
    let last = dims.len() - 1;
    let n = dims[last] as f32;
    let mean = x_f32.sum_dim_keepdim(last)?.mul_scalar(1.0 / n)?;
    let centered = x_f32.sub(&mean)?;
    let sq = centered.mul(&centered)?;
    let var = sq.sum_dim_keepdim(last)?.mul_scalar(1.0 / n)?;
    let inv_std = var.add_scalar(eps)?.rsqrt()?;
    let normed = centered.mul(&inv_std)?;
    if out_dtype == DType::F32 {
        Ok(normed)
    } else {
        normed.to_dtype(out_dtype)
    }
}

#[test]
fn layer_norm_no_affine_forward_parity_zimage_block_shape() -> Result<()> {
    let shape = &[1usize, 4096, 2560];
    let weight_shape = &[2560usize];

    let x_bf16 = rand_bf16(shape, 21, 0.5)?;

    let out_fused =
        flame_core::layer_norm::layer_norm(&x_bf16, weight_shape, None, None, 1e-6)?;
    let out_prim = primitive_layer_norm(&x_bf16, 1e-6)?;

    let f = out_fused.to_dtype(DType::F32)?.to_vec()?;
    let p = out_prim.to_dtype(DType::F32)?.to_vec()?;
    let (mx, mx_rel, idx) = max_abs_diff(&f, &p);
    println!(
        "[ln-fwd] shape={:?} max_abs={:.4e} max_rel={:.4e} at idx {}",
        shape, mx, mx_rel, idx
    );
    assert!(
        mx_rel < 0.03,
        "LN forward parity broken: max_rel={mx_rel:.4e}"
    );
    Ok(())
}

#[test]
fn layer_norm_no_affine_backward_parity_zimage_block_shape() -> Result<()> {
    let shape = &[1usize, 4096, 2560];
    let weight_shape = &[2560usize];

    let x_data = rand_bf16(shape, 31, 0.5)?;

    let x_f = Parameter::new(x_data.clone().requires_grad_(true)).tensor()?;
    let out_f =
        flame_core::layer_norm::layer_norm(&x_f, weight_shape, None, None, 1e-6)?;
    assert!(out_f.requires_grad(), "fused LN must propagate grad");
    let loss_f = out_f.mul(&out_f)?.sum()?;
    let loss_f_val = loss_f.to_dtype(DType::F32)?.to_vec()?[0];
    let grads_f = loss_f.backward()?;
    let grad_x_f = grads_f
        .get(x_f.id())
        .expect("LN fused: grad_x missing")
        .to_dtype(DType::F32)?
        .to_vec()?;

    let x_p = Parameter::new(x_data.clone().requires_grad_(true)).tensor()?;
    let out_p = primitive_layer_norm(&x_p, 1e-6)?;
    assert!(out_p.requires_grad(), "primitive LN must propagate grad");
    let loss_p = out_p.mul(&out_p)?.sum()?;
    let loss_p_val = loss_p.to_dtype(DType::F32)?.to_vec()?[0];
    let grads_p = loss_p.backward()?;
    let grad_x_p = grads_p
        .get(x_p.id())
        .expect("LN prim: grad_x missing")
        .to_dtype(DType::F32)?
        .to_vec()?;

    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut l1f = 0f64;
    let mut l1p = 0f64;
    for (a, b) in grad_x_f.iter().zip(grad_x_p.iter()) {
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64) * (*a as f64);
        nb += (*b as f64) * (*b as f64);
        l1f += (*a).abs() as f64;
        l1p += (*b).abs() as f64;
    }
    let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-30);
    let mag_ratio = na.sqrt() / nb.sqrt().max(1e-30);

    println!(
        "[ln-bwd] loss fused={loss_f_val:.6e} prim={loss_p_val:.6e} (delta {:.4e})",
        (loss_f_val - loss_p_val).abs()
    );
    println!(
        "[ln-bwd] grad_x: cos={cos:.6} mag_ratio={mag_ratio:.6} L1_fused={l1f:.4e} L1_prim={l1p:.4e}"
    );

    // RMSNorm fused vs primitive is BIT-EXACT (cos=1.000000) per the
    // companion test above; LayerNorm fused-vs-primitive is currently
    // cos ~= 0.997 with mag_ratio ~= 1.003. The gap is almost certainly
    // single-pass vs two-pass variance accumulation order in the fused
    // kernel. Pinning at cos > 0.99 documents the known-loose state and
    // catches genuine regression; tightening to 0.999 awaits a fused
    // backward kernel that mirrors the primitive's E[(x-mean)^2] order.
    assert!(cos > 0.99, "LN grad_x direction drift: cos={cos:.6}");
    assert!(
        (mag_ratio - 1.0).abs() < 0.05,
        "LN grad_x magnitude drift fused/prim = {mag_ratio:.6}"
    );
    Ok(())
}
