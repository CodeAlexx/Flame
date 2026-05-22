#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

//! Autograd backward parity for `pooling::MaxPool2d`.
//!
//! Wave 0a of the L2P port: `Op::MaxPool2D` was registered in
//! `autograd.rs` so that `pooling::MaxPool2d::forward` records on the
//! tape and `loss.backward()` reaches the backward kernel. Before this
//! change, `MaxPool2d::backward` existed as a function but was not
//! reachable from `Tensor::backward()`.
//!
//! The structural assertions here are:
//! 1. Forward propagates `requires_grad` onto the output.
//! 2. `loss.backward()` returns a non-`None` gradient for the input.
//! 3. The grad has the right shape (same as input).
//! 4. The grad is sparse — only the max-arg position in each pooling
//!    window receives the upstream value; other positions are zero.
//! 5. On a hand-crafted input where the max position is known a priori,
//!    the grad value at that position equals the upstream grad.

use anyhow::Result;
use flame_core::{
    autograd::AutogradContext,
    pooling::{MaxPool2d, MaxPool2dConfig},
    DType, Device, Shape, Tensor,
};
use serial_test::serial;

#[test]
#[serial]
fn maxpool2d_backward_smoke() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(d) => d.cuda_device_arc(),
        Err(_) => return Ok(()),
    };
    AutogradContext::reset();
    AutogradContext::set_enabled(true);

    // NHWC BF16 input, small enough to reason about. [1, 8, 8, 4].
    flame_core::rng::set_seed(0x4A9_u64).ok();
    let input = Tensor::randn(Shape::from_dims(&[1, 8, 8, 4]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);

    let cfg = MaxPool2dConfig::new((2, 2));
    let pool = MaxPool2d::new(cfg);
    let (output, _indices) = pool.forward(&input)?;

    assert_eq!(
        output.shape().dims(),
        &[1, 4, 4, 4],
        "MaxPool2d 2x2/2 [1,8,8,4] should produce [1,4,4,4]"
    );
    assert!(
        output.requires_grad(),
        "forward must propagate requires_grad onto its output"
    );

    let loss = output.sum()?;
    let grads = loss.backward()?;

    let g = grads
        .get(input.id())
        .ok_or_else(|| anyhow::anyhow!("no gradient recorded for MaxPool2d input"))?;
    assert_eq!(
        g.shape().dims(),
        input.shape().dims(),
        "MaxPool2d input grad shape mismatch"
    );

    // The grad should be sparse: 1 nonzero per 2x2 window. With
    // [1,8,8,4]=256 elements and 16 windows * 4 channels = 64 max
    // positions, exactly 64 elements should be 1.0 (sum-grad upstream
    // is all 1s) and the remainder 0.
    let g_vec = g.to_dtype(DType::F32)?.to_vec_f32()?;
    let nonzero = g_vec.iter().filter(|v| v.abs() > 1e-6).count();
    let nan_count = g_vec.iter().filter(|v| v.is_nan()).count();
    assert_eq!(nan_count, 0, "MaxPool2d backward produced NaNs");
    assert!(
        nonzero > 0,
        "MaxPool2d backward produced an all-zero input grad"
    );
    println!(
        "[maxpool2d_bwd] [1,8,8,4] kernel=2x2 stride=2: \
         nonzero {nonzero}/256, expected ≈ 64 (one per pool window per channel)"
    );
    // Upstream grad is 1.0 from sum() — round-trips bit-exact in BF16.
    // Backward recomputes argmax with the same kernel as forward, and the
    // kernel uses strict `>` for tie-breaking (first-occurrence wins), so
    // each of the 16 windows * 4 channels = 64 max positions receives
    // exactly one nonzero grad. No slop is needed.
    assert_eq!(
        nonzero, 64,
        "MaxPool2d backward nonzero count {nonzero} != expected 64"
    );

    Ok(())
}

#[test]
#[serial]
fn maxpool2d_backward_known_argmax() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(d) => d.cuda_device_arc(),
        Err(_) => return Ok(()),
    };
    AutogradContext::reset();
    AutogradContext::set_enabled(true);

    // Construct a [1, 2, 2, 1] NHWC input where the max in the single
    // 2x2 window is at NHWC index (n=0, h=1, w=0, c=0). Layout (NHWC
    // memory order) is [v(0,0), v(0,1), v(1,0), v(1,1)] for c=0.
    let raw: Vec<f32> = vec![0.1, 0.2, 0.9, 0.3];
    let input = Tensor::from_vec(raw.clone(), Shape::from_dims(&[1, 2, 2, 1]), device.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);

    let pool = MaxPool2d::new(MaxPool2dConfig::new((2, 2)));
    let (output, _) = pool.forward(&input)?;
    assert_eq!(output.shape().dims(), &[1, 1, 1, 1]);
    assert!(
        output.requires_grad(),
        "forward must propagate requires_grad onto its output"
    );

    let out_vec = output.to_dtype(DType::F32)?.to_vec_f32()?;
    assert!(
        (out_vec[0] - 0.9).abs() < 1e-2,
        "expected forward max 0.9, got {}",
        out_vec[0]
    );

    let loss = output.sum()?; // upstream grad = 1.0
    let grads = loss.backward()?;
    let g = grads
        .get(input.id())
        .ok_or_else(|| anyhow::anyhow!("no gradient recorded for input"))?;
    let g_vec = g.to_dtype(DType::F32)?.to_vec_f32()?;
    assert_eq!(g_vec.len(), 4);

    // The max is at NHWC index 2 (i.e. raw[2]=0.9). All other positions
    // must be 0; that one must be ≈1.0.
    assert!(
        (g_vec[2] - 1.0).abs() < 1e-2,
        "grad at argmax position should be ≈1, got {}",
        g_vec[2]
    );
    for (i, v) in g_vec.iter().enumerate() {
        if i != 2 {
            assert!(
                v.abs() < 1e-3,
                "grad at non-argmax position {i} should be ≈0, got {v}"
            );
        }
    }
    println!("[maxpool2d_bwd] known argmax test: grad = {:?}", g_vec);
    Ok(())
}

#[test]
#[serial]
fn maxpool2d_no_record_when_requires_grad_false() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(d) => d.cuda_device_arc(),
        Err(_) => return Ok(()),
    };
    AutogradContext::reset();
    AutogradContext::set_enabled(true);

    // Same shape as the smoke test, but DO NOT request grad on the input.
    flame_core::rng::set_seed(0x4A9_u64).ok();
    let input = Tensor::randn(Shape::from_dims(&[1, 8, 8, 4]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;
    assert!(
        !input.requires_grad(),
        "test precondition: input must not require grad"
    );

    let pool = MaxPool2d::new(MaxPool2dConfig::new((2, 2)));
    let (output, _) = pool.forward(&input)?;

    assert!(
        !output.requires_grad(),
        "MaxPool2d::forward must NOT propagate requires_grad when input.requires_grad=false"
    );
    Ok(())
}
