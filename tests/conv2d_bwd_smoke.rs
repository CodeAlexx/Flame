#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

//! Wave 0b: smoke test that `conv::Conv2d::forward` records on the
//! autograd tape so `loss.backward()` can flow gradients back through it.
//!
//! Why this matters: `conv::Conv2d::forward` calls into either
//! `cudnn::cudnn_conv2d_bf16` (cuDNN fast path) or
//! `cuda_ops_bf16::conv2d_bf16` (NHWC BF16 kernel). Neither of these
//! records on the tape — recording lives in
//! `ops::conv2d::conv2d_forward` (`ops/conv2d.rs:846-869`). The L2P
//! U-Net has 11 Conv2d calls; if the recording entry isn't reached when
//! training, all those grads silently vanish.
//!
//! This test is INTENTIONALLY a smoke test, not a fix. If it fails,
//! the recommended action is to surface the gap; fixing the routing in
//! `conv::Conv2d::forward` is a separate, larger change.

use anyhow::Result;
use flame_core::{
    autograd::AutogradContext,
    cuda_conv2d::CudaConv2d,
    conv::{Conv2d, Conv2dConfig},
    DType, Device, Shape, Tensor,
};
use serial_test::serial;

#[test]
#[serial]
fn conv2d_forward_records_on_tape() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(d) => d.cuda_device_arc(),
        Err(_) => return Ok(()),
    };
    AutogradContext::reset();
    AutogradContext::set_enabled(true);

    // NCHW BF16 input. Conv2d::forward expects NCHW (per source comment at conv.rs:321).
    let input = Tensor::randn(Shape::from_dims(&[1, 4, 16, 16]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);

    // 3x3 stride-1 pad-1, 4 -> 8, with bias (L2P U-Net uses bias-enabled convs).
    let config = Conv2dConfig {
        in_channels: 4,
        out_channels: 8,
        kernel_size: (3, 3),
        stride: (1, 1),
        padding: (1, 1),
        groups: 1,
    };
    let conv = Conv2d::from_config_with_bias(config, device.clone(), true)?;

    let output = conv.forward(&input)?;
    assert_eq!(
        output.shape().dims(),
        &[1, 8, 16, 16],
        "Conv2d(4->8, 3x3, stride=1, pad=1) on [1,4,16,16] should produce [1,8,16,16]"
    );

    // The diagnostic that catches the gap: did requires_grad propagate?
    println!(
        "[conv2d_bwd_smoke] output.requires_grad={} (input.requires_grad={}, weight.requires_grad={}, bias.requires_grad={})",
        output.requires_grad(),
        input.requires_grad(),
        conv.weight.requires_grad(),
        conv.bias.as_ref().map(|b| b.requires_grad()).unwrap_or(false),
    );
    assert!(
        output.requires_grad(),
        "FINDING: conv::Conv2d::forward did not propagate requires_grad. \
         This means cuDNN/conv2d_bf16 fast path was taken and the autograd \
         tape was bypassed. Backward through this Conv2d will fail. \
         See ops/conv2d.rs:735 — `ops::conv2d::conv2d_forward` gates the cuDNN \
         fast path on `!is_recording()`, but `conv::Conv2d::forward` does not."
    );

    let loss = output.sum()?;
    let grads = loss.backward()?;

    let g_input = grads
        .get(input.id())
        .ok_or_else(|| anyhow::anyhow!(
            "FINDING: no input grad after Conv2d backward — recording was skipped"
        ))?;
    assert_eq!(
        g_input.shape().dims(),
        input.shape().dims(),
        "Conv2d input grad shape mismatch"
    );
    let g_in_vec = g_input.to_dtype(DType::F32)?.to_vec_f32()?;
    let nz = g_in_vec.iter().filter(|v| v.abs() > 1e-6).count();
    assert!(nz > 0, "Conv2d input grad is all-zero");
    assert_eq!(
        g_in_vec.iter().filter(|v| v.is_nan()).count(),
        0,
        "Conv2d input grad has NaNs"
    );

    let g_weight = grads.get(conv.weight.id()).ok_or_else(|| {
        anyhow::anyhow!("FINDING: no weight grad after Conv2d backward — recording was skipped")
    })?;
    assert_eq!(
        g_weight.shape().dims(),
        conv.weight.shape().dims(),
        "Conv2d weight grad shape mismatch"
    );

    if let Some(bias) = &conv.bias {
        let g_bias = grads.get(bias.id()).ok_or_else(|| {
            anyhow::anyhow!("FINDING: no bias grad after Conv2d backward — recording was skipped")
        })?;
        assert_eq!(
            g_bias.shape().dims(),
            bias.shape().dims(),
            "Conv2d bias grad shape mismatch"
        );
    }

    println!("[conv2d_bwd_smoke] PASS — Conv2d records and backwards cleanly.");
    Ok(())
}

#[test]
#[serial]
fn conv2d_backward_matches_cpu_layout() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(d) => d.cuda_device_arc(),
        Err(_) => return Ok(()),
    };

    let n = 1usize;
    let ic = 2usize;
    let oc = 3usize;
    let h = 3usize;
    let w = 4usize;
    let kh = 2usize;
    let kw = 2usize;
    let oh = h - kh + 1;
    let ow = w - kw + 1;

    let input_v: Vec<f32> = (0..n * ic * h * w)
        .map(|i| ((i as i32 % 13) as f32 - 6.0) * 0.125)
        .collect();
    let weight_v: Vec<f32> = (0..oc * ic * kh * kw)
        .map(|i| ((i as i32 % 11) as f32 - 5.0) * 0.0625)
        .collect();
    let grad_out_v: Vec<f32> = (0..n * oc * oh * ow)
        .map(|i| ((i as i32 % 7) as f32 - 3.0) * 0.25)
        .collect();

    let input = Tensor::from_vec(input_v.clone(), Shape::from_dims(&[n, ic, h, w]), device.clone())?;
    let weight = Tensor::from_vec(
        weight_v.clone(),
        Shape::from_dims(&[oc, ic, kh, kw]),
        device.clone(),
    )?;
    let grad_out = Tensor::from_vec(
        grad_out_v.clone(),
        Shape::from_dims(&[n, oc, oh, ow]),
        device.clone(),
    )?;

    let (grad_in, grad_w, grad_b) =
        CudaConv2d::conv2d_backward(&grad_out, &input, &weight, (1, 1), (0, 0))?;

    let mut exp_in = vec![0.0f32; n * ic * h * w];
    let mut exp_w = vec![0.0f32; oc * ic * kh * kw];
    let mut exp_b = vec![0.0f32; oc];

    let input_idx = |b: usize, c: usize, y: usize, x: usize| ((b * ic + c) * h + y) * w + x;
    let weight_idx =
        |o: usize, c: usize, ky: usize, kx: usize| ((o * ic + c) * kh + ky) * kw + kx;
    let go_idx = |b: usize, o: usize, y: usize, x: usize| ((b * oc + o) * oh + y) * ow + x;

    for b in 0..n {
        for o in 0..oc {
            for oy in 0..oh {
                for ox in 0..ow {
                    let go = grad_out_v[go_idx(b, o, oy, ox)];
                    exp_b[o] += go;
                    for c in 0..ic {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = oy + ky;
                                let ix = ox + kx;
                                exp_in[input_idx(b, c, iy, ix)] +=
                                    go * weight_v[weight_idx(o, c, ky, kx)];
                                exp_w[weight_idx(o, c, ky, kx)] +=
                                    go * input_v[input_idx(b, c, iy, ix)];
                            }
                        }
                    }
                }
            }
        }
    }

    fn assert_close(name: &str, got: &[f32], exp: &[f32]) {
        assert_eq!(got.len(), exp.len(), "{name} len mismatch");
        let mut max_diff = 0.0f32;
        for (g, e) in got.iter().zip(exp.iter()) {
            max_diff = max_diff.max((g - e).abs());
        }
        assert!(
            max_diff < 1e-4,
            "{name} max diff {max_diff:.6e} exceeds tolerance"
        );
    }

    assert_close("grad_input", &grad_in.to_vec_f32()?, &exp_in);
    assert_close("grad_weight", &grad_w.to_vec_f32()?, &exp_w);
    assert_close(
        "grad_bias",
        &grad_b.expect("bias grad").to_vec_f32()?,
        &exp_b,
    );

    Ok(())
}
