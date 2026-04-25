//! Layer-by-layer autograd bisect: minimal forward paths that exercise
//! ONE op at a time, each verified against central finite-diff.
//!
//! Strategy: start with matmul-only (known good), then add one op per
//! test. Whichever test fails first narrows the bug to the newly-added op.
#![cfg(feature = "cuda")]

use flame_core::{
    autograd::AutogradContext, global_cuda_device, parameter::Parameter, DType, Result, Shape,
    Tensor,
};

fn rel_err(a: f32, b: f32) -> f32 {
    let m = a.abs().max(b.abs()).max(1e-8);
    (a - b).abs() / m
}

fn seeded(shape: &[usize], seed: u64, device: &std::sync::Arc<cudarc::driver::CudaDevice>) -> Result<Tensor> {
    let numel: usize = shape.iter().product();
    let mut v = Vec::with_capacity(numel);
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for _ in 0..numel {
        s = s.wrapping_mul(0x5851F42D4C957F2D).wrapping_add(1);
        let f = ((s >> 32) as i32 as f32) / (i32::MAX as f32);
        v.push(f * 0.3);
    }
    Tensor::from_vec(v, Shape::from_dims(shape), device.clone())
}

fn mse_mean(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
    pred.sub(target)?.square()?.mean()
}

fn check_probes<F>(
    a_param: &Parameter,
    loss_fn: F,
    label: &str,
) -> Result<f32>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let a = a_param.tensor()?;
    let loss = loss_fn(&a)?;
    let baseline = loss.to_vec()?[0];
    let grads = loss.backward()?;
    let grad_a = grads
        .get(a.id())
        .ok_or_else(|| flame_core::Error::Training("grad_a missing".into()))?;
    let grad_data = grad_a.to_vec()?;

    let dims = a.shape().dims().to_vec();
    let shape = Shape::from_dims(&dims);
    let device = a.device().clone();

    // BF16 has ~3 decimal digits of precision; loss values around 1.0 have
    // ULP ~5e-4, so eps must be large enough that ε * ||g||² lands well
    // above that floor. After norm ops (layer_norm, rms_norm), single-
    // element FD perturbations are heavily attenuated, requiring even
    // larger eps. F32 is fine with eps=1e-3.
    let eps = if a.dtype() == DType::BF16 { 0.25f32 } else { 1.0e-3f32 };
    let a_dtype = a.dtype();
    let numel: usize = dims.iter().product();
    let probes = [
        0,
        numel / 7,
        numel / 2,
        numel - 1,
    ];
    let mut worst = 0.0f32;
    println!("  {label}: baseline loss = {baseline:.6}");
    for &idx in &probes {
        let analytical = grad_data[idx];
        let mut data = a.to_vec()?;
        data[idx] += eps;
        let a_plus = Tensor::from_vec(data, shape.clone(), device.clone())?
            .to_dtype(a_dtype)?;
        let lp = {
            let _g = AutogradContext::no_grad();
            loss_fn(&a_plus)?.to_vec()?[0]
        };
        let mut data = a.to_vec()?;
        data[idx] -= eps;
        let a_minus = Tensor::from_vec(data, shape.clone(), device.clone())?
            .to_dtype(a_dtype)?;
        let lm = {
            let _g = AutogradContext::no_grad();
            loss_fn(&a_minus)?.to_vec()?[0]
        };
        let fd = (lp - lm) / (2.0 * eps);
        let re = rel_err(analytical, fd);
        // Skip noise-floor probes: when |analytical| AND |fd| are both
        // below BF16 / loss-precision threshold, the FD is dominated by
        // rounding noise and rel_err is meaningless. Only count probes
        // where at least one side is well above noise.
        let noise_floor = if a.dtype() == DType::BF16 { 5e-3f32 } else { 1e-5f32 };
        let signal = analytical.abs().max(fd.abs());
        let counted = signal > noise_floor;
        let tag = if !counted { "noise" } else if re > 0.05 { "BAD" } else { "ok" };
        if counted {
            worst = worst.max(re);
        }
        println!("    idx={idx} analytical={analytical:+.3e} fd={fd:+.3e} rel={re:.4} {tag}");
    }
    Ok(worst)
}

/// Test 1: y = A @ target -> loss = mse(y, target2)  (pure matmul)
#[test]
fn test_1_pure_matmul() -> Result<()> {
    let device = global_cuda_device();
    let a_param = Parameter::new(seeded(&[16, 32], 1, &device)?.requires_grad_(true));
    let x = seeded(&[32, 8], 2, &device)?;
    let target = seeded(&[16, 8], 3, &device)?;
    let worst = check_probes(&a_param, |a| a.matmul(&x)?.sub(&target).and_then(|d| d.mul(&d)?.mean()), "pure_matmul")?;
    assert!(worst < 0.05, "worst={worst:.4}");
    Ok(())
}

/// Test 2: y = (A @ x) transposed then summed -> tests transpose backward
#[test]
fn test_2_matmul_then_transpose() -> Result<()> {
    let device = global_cuda_device();
    let a_param = Parameter::new(seeded(&[16, 32], 1, &device)?.requires_grad_(true));
    let x = seeded(&[32, 8], 2, &device)?;
    let target = seeded(&[8, 16], 3, &device)?;
    let worst = check_probes(&a_param, |a| {
        let y = a.matmul(&x)?;       // [16, 8]
        let yt = y.transpose()?;      // [8, 16] strided view
        mse_mean(&yt, &target)
    }, "matmul_then_transpose")?;
    assert!(worst < 0.05, "worst={worst:.4}");
    Ok(())
}

/// Test 3: y = layer_norm(A @ x) -> tests layer_norm backward
///
/// BF16 precision-limited: layer_norm normalizes mean and variance, so
/// single-element FD perturbations of A produce tiny ΔL that often round
/// to zero in BF16. Threshold relaxed to 25%; the test still catches
/// gross direction errors (sign flips, factor-of-2 magnitude bugs).
#[test]
fn test_3_matmul_layernorm() -> Result<()> {
    let device = global_cuda_device();
    let a_param = Parameter::new(seeded(&[8, 16], 1, &device)?.to_dtype(DType::BF16)?.requires_grad_(true));
    let x = seeded(&[16, 32], 2, &device)?.to_dtype(DType::BF16)?;
    let target = seeded(&[8, 32], 3, &device)?.to_dtype(DType::BF16)?;
    let worst = check_probes(&a_param, |a| {
        let y = a.matmul(&x)?;
        let normed = flame_core::layer_norm::layer_norm(&y, &[32], None, None, 1e-5)?;
        mse_mean(&normed, &target)
    }, "matmul_layernorm")?;
    assert!(worst < 0.30, "worst={worst:.4}");
    Ok(())
}

/// Test 4: y = rms_norm(A @ x, scale) -> tests rms_norm backward
///
/// BF16 precision-limited; same caveat as test_3.
#[test]
fn test_4_matmul_rmsnorm() -> Result<()> {
    let device = global_cuda_device();
    let a_param = Parameter::new(seeded(&[8, 16], 1, &device)?.to_dtype(DType::BF16)?.requires_grad_(true));
    let x = seeded(&[16, 32], 2, &device)?.to_dtype(DType::BF16)?;
    let target = seeded(&[8, 32], 3, &device)?.to_dtype(DType::BF16)?;
    let scale = seeded(&[32], 4, &device)?.to_dtype(DType::BF16)?;
    let worst = check_probes(&a_param, |a| {
        let y = a.matmul(&x)?;
        let normed = flame_core::norm::rms_norm(&y, &[32], Some(&scale), 1e-5)?;
        mse_mean(&normed, &target)
    }, "matmul_rmsnorm")?;
    assert!(worst < 0.30, "worst={worst:.4}");
    Ok(())
}

/// Test 5: pure interleaved RoPE with autograd.
/// If this fails, the RoPe forward/backward layout match is broken.
#[test]
fn test_5_rope_interleaved() -> Result<()> {
    use flame_core::autograd::{AutogradContext, Op};
    let device = global_cuda_device();
    let bh = 2usize;
    let n = 8usize;
    let d = 16usize; // even
    let half = d / 2;

    // Input param (strided via permute to exercise kernel path).
    let x_plain = seeded(&[bh, n, d], 7, &device)?.to_dtype(DType::BF16)?;
    let x_param = Parameter::new(x_plain.requires_grad_(true));
    let cos = seeded(&[1, n, half], 11, &device)?.to_dtype(DType::BF16)?;
    let sin = seeded(&[1, n, half], 13, &device)?.to_dtype(DType::BF16)?;

    let make_loss = |x: &Tensor, cos: &Tensor, sin: &Tensor| -> Result<Tensor> {
        let x4 = x.reshape(&[1, bh, n, d])?;
        let cos4 = cos.reshape(&[1, 1, n, half])?;
        let sin4 = sin.reshape(&[1, 1, n, half])?;
        let y = flame_core::bf16_ops::rope_fused_bf16(&x4, &cos4, &sin4)?;
        // Record Op::RoPePrecomputed manually — same as klein-trainer.
        let mut y = y;
        if x.requires_grad() && AutogradContext::is_recording() {
            y = y.requires_grad_(true);
            AutogradContext::record_op(
                y.id(),
                Op::RoPePrecomputed {
                    input: x4.id(),
                    cos: cos4.id(),
                    sin: sin4.id(),
                },
                vec![
                    (x4.id(), x4.clone()),
                    (cos4.id(), cos4.clone()),
                    (sin4.id(), sin4.clone()),
                ],
            );
        }
        // Reshape causes autograd — avoid by staying in 4D for sum.
        // Use to_dtype(F32) for stable FD; square+sum gives clean signal.
        let y32 = y.to_dtype(DType::F32)?;
        y32.mul(&y32)?.sum()
    };

    let x_t = x_param.tensor()?;
    let loss = make_loss(&x_t, &cos, &sin)?;
    let baseline = loss.to_vec()?[0];
    println!("  rope_interleaved: baseline = {baseline:.6}");
    let grads = loss.backward()?;
    let grad_x = grads.get(x_t.id()).expect("grad_x missing").to_dtype(DType::F32)?.to_vec()?;

    let eps = 5.0e-2f32;
    let probes = [0usize, 17, bh*n*d/2, bh*n*d - 1];
    let mut worst = 0.0f32;
    let x_vec_bf16 = x_t.to_dtype(DType::F32)?.to_vec()?;
    for &idx in &probes {
        let mut data = x_vec_bf16.clone(); data[idx] += eps;
        let lp = {
            let _g = AutogradContext::no_grad();
            let xp = Tensor::from_vec(data, Shape::from_dims(&[bh, n, d]), device.clone())?.to_dtype(DType::BF16)?;
            make_loss(&xp, &cos, &sin)?.to_vec()?[0]
        };
        let mut data = x_vec_bf16.clone(); data[idx] -= eps;
        let lm = {
            let _g = AutogradContext::no_grad();
            let xm = Tensor::from_vec(data, Shape::from_dims(&[bh, n, d]), device.clone())?.to_dtype(DType::BF16)?;
            make_loss(&xm, &cos, &sin)?.to_vec()?[0]
        };
        let fd = (lp - lm) / (2.0 * eps);
        let analytical = grad_x[idx];
        let re = rel_err(analytical, fd);
        // Skip noise-floor probes (BF16 precision limit).
        let signal = analytical.abs().max(fd.abs());
        let counted = signal > 1e-3f32;
        let tag = if !counted { "noise" } else if re > 0.05 { "BAD" } else { "ok" };
        if counted {
            worst = worst.max(re);
        }
        println!("    idx={idx} analytical={analytical:+.3e} fd={fd:+.3e} rel={re:.4} {tag}");
    }
    assert!(worst < 0.1, "rope interleaved backward disagrees with FD: worst={worst:.4}");
    Ok(())
}
