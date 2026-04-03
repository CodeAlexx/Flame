#![cfg(all(feature = "cuda", feature = "heavy_kernels", feature = "bf16_u16"))]

mod testutil;

use anyhow::Result;
use flame_core::{
    cuda_ops_bf16, cuda_ops_bf16::ConvActivation, device::Device, DType, Shape, Tensor,
};
use half::bf16;

#[cfg(feature = "strict_bf16")]
use once_cell::sync::Lazy;
#[cfg(feature = "strict_bf16")]
use std::sync::Mutex;
#[cfg(feature = "strict_bf16")]
static STRICT_BF16_TEST_GUARD: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));
#[cfg(feature = "strict_bf16")]
use flame_core::strict;

fn zeros_on(device: &Device, dims: &[usize], dtype: DType) -> Result<Tensor> {
    Tensor::zeros_dtype(Shape::from_dims(dims), dtype, device.cuda_device_arc())
        .map_err(|e| e.into())
}

#[test]
fn conv2d_f32_ok() {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return,
    };
    let x = zeros_on(&device, &[1, 3, 8, 8], DType::F32).expect("F32 tensor");
    let w = zeros_on(&device, &[4, 3, 3, 3], DType::F32).expect("F32 tensor");
    let y = x.conv2d(&w, None, 1, 1).expect("F32 conv2d");
    assert_eq!(y.dtype(), DType::F32);
}

#[test]
fn conv2d_bf16_ok() {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return,
    };
    let x = zeros_on(&device, &[1, 3, 8, 8], DType::BF16).expect("BF16 tensor");
    let w = zeros_on(&device, &[4, 3, 3, 3], DType::BF16).expect("BF16 tensor");
    let y = x.conv2d(&w, None, 1, 1).expect("BF16 conv2d");
    assert_eq!(
        y.dtype(),
        DType::BF16,
        "BF16 conv2d must preserve storage dtype"
    );
}

#[test]
fn conv2d_mixed_rejected() {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return,
    };
    let x = zeros_on(&device, &[1, 3, 8, 8], DType::F32).expect("F32 tensor");
    let w = zeros_on(&device, &[4, 3, 3, 3], DType::BF16).expect("BF16 tensor");
    let err = x
        .conv2d(&w, None, 1, 1)
        .expect_err("mixed conv2d should error");
    testutil::assert_mixed_dtype_err(&err);
}

#[test]
fn conv2d_bf16_depthwise_matches_reference() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let cuda = device.cuda_device_arc();

    let n = 1usize;
    let h = 4usize;
    let w = 4usize;
    let c = 2usize;
    let kh = 3usize;
    let kw = 3usize;
    let groups = c as i32;

    let input_data: Vec<f32> = (0..(n * h * w * c)).map(|i| (i as f32) * 0.1).collect();
    let weight_data: Vec<f32> = vec![1.0f32; kh * kw * 1 * c];

    let input = Tensor::from_vec(
        input_data.clone(),
        Shape::from_dims(&[n, h, w, c]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let weights = Tensor::from_vec(
        weight_data.clone(),
        Shape::from_dims(&[kh, kw, 1, c]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;

    let output = cuda_ops_bf16::conv2d_bf16(
        &input,
        &weights,
        None,
        (1, 1),
        (0, 0),
        (1, 1),
        groups,
        ConvActivation::None,
    )?;

    let output_host = output.to_dtype(DType::F32)?.to_vec_f32()?;

    let ho = h - kh + 1;
    let wo = w - kw + 1;
    let mut expected = vec![0f32; ho * wo * c];
    for oh in 0..ho {
        for ow in 0..wo {
            for ch in 0..c {
                let mut acc = 0f32;
                for kh_idx in 0..kh {
                    for kw_idx in 0..kw {
                        let ih = oh + kh_idx;
                        let iw = ow + kw_idx;
                        let input_index = ((ih * w + iw) * c + ch) as usize;
                        let weight_index = ((kh_idx * kw + kw_idx) * c + ch) as usize;
                        acc += input_data[input_index] * weight_data[weight_index];
                    }
                }
                let out_index = ((oh * wo + ow) * c + ch) as usize;
                expected[out_index] = acc;
            }
        }
    }

    for (idx, (expected_val, actual_val)) in expected.iter().zip(output_host.iter()).enumerate() {
        let diff = (expected_val - actual_val).abs();
        assert!(
            diff < 1e-1,
            "depthwise conv mismatch at {}: expected {}, got {}, diff {}",
            idx,
            expected_val,
            actual_val,
            diff
        );
    }

    Ok(())
}

#[test]
fn conv2d_bf16_relu_activation() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let cuda = device.cuda_device_arc();

    let input_data = vec![-1.0f32, 2.0, -3.0, 4.0];
    let weight_data = vec![1.0f32];

    let input = Tensor::from_vec(
        input_data.clone(),
        Shape::from_dims(&[1, 2, 2, 1]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let weight = Tensor::from_vec(weight_data, Shape::from_dims(&[1, 1, 1, 1]), cuda.clone())?
        .to_dtype(DType::BF16)?;

    let output = cuda_ops_bf16::conv2d_bf16(
        &input,
        &weight,
        None,
        (1, 1),
        (0, 0),
        (1, 1),
        1,
        ConvActivation::Relu,
    )?;

    let output_host = output.to_dtype(DType::F32)?.to_vec_f32()?;
    let expected: Vec<f32> = input_data.into_iter().map(|v| v.max(0.0)).collect();
    for (idx, (exp, got)) in expected.iter().zip(output_host.iter()).enumerate() {
        assert!(
            (exp - got).abs() < 2e-3,
            "relu activation mismatch at {}: expected {}, got {}",
            idx,
            exp,
            got
        );
    }

    Ok(())
}

#[test]
fn reshape_slice_reshape_bf16_device_path() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let cuda = device.cuda_device_arc();

    let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let tensor = Tensor::from_vec(data.clone(), Shape::from_dims(&[1, 4, 4, 4]), cuda.clone())?
        .to_dtype(DType::BF16)?;

    let reshaped = tensor.reshape(&[1, 16, 4])?;
    let sliced = reshaped.slice_1d_device(1, 4, 8)?;
    let final_view = sliced.reshape(&[1, 2, 4, 4])?;

    assert_eq!(final_view.dtype(), DType::BF16);
    assert_eq!(final_view.storage_dtype(), DType::BF16);

    let final_host = final_view.to_dtype(DType::F32)?.to_vec_f32()?;
    let mut expected = Vec::with_capacity(32);
    for idx in 4..12 {
        let h = idx / 4;
        let w = idx % 4;
        for c in 0..4 {
            let offset = ((h * 4 + w) * 4 + c) as usize;
            expected.push(data[offset]);
        }
    }

    assert_eq!(expected.len(), final_host.len());
    for (i, (exp, got)) in expected.iter().zip(final_host.iter()).enumerate() {
        let diff = (exp - got).abs();
        assert!(
            diff < 1e-2,
            "reshape→slice→reshape mismatch at {i}: expected {exp}, got {got}, diff {diff}"
        );
    }

    Ok(())
}

#[test]
fn conv2d_bf16_grouped_matches_reference() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let cuda = device.cuda_device_arc();

    let n = 1usize;
    let h = 3usize;
    let w = 3usize;
    let cin = 8usize;
    let cout = 8usize;
    let groups = 2usize;
    let kh = 1usize;
    let kw = 1usize;

    let mut input_data = Vec::new();
    for i in 0..(n * h * w * cin) {
        input_data.push((i % 7) as f32 - 3.0);
    }
    let mut weight_data = Vec::new();
    for i in 0..(kh * kw * (cin / groups) * cout) {
        weight_data.push(((i % 5) as f32 - 2.0) * 0.5);
    }
    let bias_data: Vec<f32> = (0..cout).map(|i| (i as f32) * 0.1).collect();

    let input = Tensor::from_vec(
        input_data.clone(),
        Shape::from_dims(&[n, h, w, cin]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let weights = Tensor::from_vec(
        weight_data.clone(),
        Shape::from_dims(&[kh, kw, cin / groups, cout]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let bias = Tensor::from_vec(bias_data.clone(), Shape::from_dims(&[cout]), cuda.clone())?
        .to_dtype(DType::BF16)?;

    let output = cuda_ops_bf16::conv2d_bf16(
        &input,
        &weights,
        Some(&bias),
        (1, 1),
        (0, 0),
        (1, 1),
        groups as i32,
        ConvActivation::None,
    )?;

    let output_host = output.to_dtype(DType::F32)?.to_vec_f32()?;

    let ho = h - kh + 1;
    let wo = w - kw + 1;
    let mut expected = vec![0f32; ho * wo * cout];
    let cin_per_group = cin / groups;
    let cout_per_group = cout / groups;
    for n_idx in 0..n {
        for oh in 0..ho {
            for ow in 0..wo {
                for g in 0..groups {
                    for co in 0..cout_per_group {
                        let mut acc = bias_data[g * cout_per_group + co];
                        for kh_idx in 0..kh {
                            for kw_idx in 0..kw {
                                let ih = oh + kh_idx;
                                let iw = ow + kw_idx;
                                for ci in 0..cin_per_group {
                                    let channel_in = g * cin_per_group + ci;
                                    let channel_out = g * cout_per_group + co;
                                    let input_index =
                                        (((n_idx * h + ih) * w + iw) * cin + channel_in) as usize;
                                    let weight_index =
                                        ((((kh_idx * kw) + kw_idx) * cin_per_group + ci) * cout
                                            + channel_out)
                                            as usize;
                                    acc += input_data[input_index] * weight_data[weight_index];
                                }
                            }
                        }
                        let out_index = (((n_idx * ho + oh) * wo + ow) * cout
                            + g * cout_per_group
                            + co) as usize;
                        expected[out_index] = acc;
                    }
                }
            }
        }
    }

    for (idx, (exp, got)) in expected.iter().zip(output_host.iter()).enumerate() {
        assert!(
            (exp - got).abs() < 5e-2,
            "grouped conv mismatch at {}: expected {}, got {}",
            idx,
            exp,
            got
        );
    }

    Ok(())
}

#[test]
fn conv2d_autotune_stats_report_activity() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let cuda = device.cuda_device_arc();

    cuda_ops_bf16::reset_conv2d_autotune_stats()?;
    let before = cuda_ops_bf16::conv2d_autotune_stats()?;
    assert_eq!(before.tuned, 0, "stats should reset tuned count to zero");

    let x = Tensor::zeros_dtype(Shape::from_dims(&[1, 16, 16, 8]), DType::BF16, cuda.clone())?;
    let w = Tensor::zeros_dtype(Shape::from_dims(&[3, 3, 8, 8]), DType::BF16, cuda.clone())?;
    let _ = cuda_ops_bf16::conv2d_bf16(
        &x,
        &w,
        None,
        (1, 1),
        (1, 1),
        (1, 1),
        1,
        ConvActivation::None,
    )?;

    let after = cuda_ops_bf16::conv2d_autotune_stats()?;
    assert!(
        after.tuned >= 1,
        "expected autotune to run at least once; stats={:?}",
        after
    );
    Ok(())
}

#[test]
fn conv2d_autotune_skip_and_tune_paths() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let cuda = device.cuda_device_arc();

    cuda_ops_bf16::reset_conv2d_autotune_stats()?;

    // Small spatial footprint (Ho*Wo = 16) should skip tuning and increment workspace_skips.
    let small_input =
        Tensor::zeros_dtype(Shape::from_dims(&[1, 4, 4, 8]), DType::BF16, cuda.clone())?;
    let small_weight =
        Tensor::zeros_dtype(Shape::from_dims(&[3, 3, 8, 8]), DType::BF16, cuda.clone())?;
    let _ = cuda_ops_bf16::conv2d_bf16(
        &small_input,
        &small_weight,
        None,
        (1, 1),
        (1, 1),
        (1, 1),
        1,
        ConvActivation::None,
    )?;

    let stats_after_small = cuda_ops_bf16::conv2d_autotune_stats()?;
    assert!(
        stats_after_small.workspace_skips >= 1,
        "expected autotune to skip due to small spatial tiles; stats={:?}",
        stats_after_small
    );
    assert_eq!(
        stats_after_small.tuned, 0,
        "small conv should not trigger tuning"
    );

    // Larger shape should autotune and update tuned counter.
    let large_input =
        Tensor::zeros_dtype(Shape::from_dims(&[1, 32, 32, 8]), DType::BF16, cuda.clone())?;
    let large_weight =
        Tensor::zeros_dtype(Shape::from_dims(&[3, 3, 8, 8]), DType::BF16, cuda.clone())?;
    let _ = cuda_ops_bf16::conv2d_bf16(
        &large_input,
        &large_weight,
        None,
        (1, 1),
        (1, 1),
        (1, 1),
        1,
        ConvActivation::None,
    )?;

    let stats_after_large = cuda_ops_bf16::conv2d_autotune_stats()?;
    assert!(
        stats_after_large.tuned >= 1,
        "expected autotune to run on large conv; stats={:?}",
        stats_after_large
    );
    Ok(())
}

fn gelu_tanh_reference(x: f32) -> f32 {
    const C: f32 = 0.044715;
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654; // sqrt(2 / pi)
    let inner = SQRT_2_OVER_PI * (x + C * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

fn conv2d_reference_nhwc(
    input: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    n: usize,
    h: usize,
    w: usize,
    cin: usize,
    kh: usize,
    kw: usize,
    groups: usize,
    stride: (usize, usize),
    pad: (usize, usize),
    dilation: (usize, usize),
    cout: usize,
) -> Vec<f32> {
    let (stride_h, stride_w) = (stride.0 as isize, stride.1 as isize);
    let (pad_h, pad_w) = (pad.0 as isize, pad.1 as isize);
    let (dil_h, dil_w) = (dilation.0 as isize, dilation.1 as isize);
    let ho = ((h as isize + 2 * pad_h - (kh as isize - 1) * dil_h - 1) / stride_h + 1) as usize;
    let wo = ((w as isize + 2 * pad_w - (kw as isize - 1) * dil_w - 1) / stride_w + 1) as usize;
    let mut out = vec![0f32; n * ho * wo * cout];
    let group_in = cin / groups;
    let group_out = cout / groups;
    for batch in 0..n {
        for oh in 0..ho {
            for ow in 0..wo {
                for co in 0..cout {
                    let g = co / group_out;
                    let mut acc = bias.map(|b| b[co]).unwrap_or(0.0);
                    for kh_idx in 0..kh {
                        let ih = oh as isize * stride_h - pad_h + kh_idx as isize * dil_h;
                        if ih < 0 || ih >= h as isize {
                            continue;
                        }
                        for kw_idx in 0..kw {
                            let iw = ow as isize * stride_w - pad_w + kw_idx as isize * dil_w;
                            if iw < 0 || iw >= w as isize {
                                continue;
                            }
                            for ci_group in 0..group_in {
                                let ci = g * group_in + ci_group;
                                let input_index =
                                    ((((batch * h + ih as usize) * w + iw as usize) * cin) + ci)
                                        as usize;
                                let weight_index =
                                    ((((kh_idx * kw + kw_idx) * group_in + ci_group) * cout) + co)
                                        as usize;
                                acc += input[input_index] * weights[weight_index];
                            }
                        }
                    }
                    let out_index = ((((batch * ho + oh) * wo + ow) * cout) + co) as usize;
                    out[out_index] = acc;
                }
            }
        }
    }
    out
}

#[test]
fn conv2d_bf16_bias_matches_reference() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let cuda = device.cuda_device_arc();

    let n = 1usize;
    let h = 3usize;
    let w = 3usize;
    let ic = 2usize;
    let oc = 2usize;
    let kh = 1usize;
    let kw = 1usize;

    let input_values: Vec<f32> = (0..(n * h * w * ic))
        .map(|idx| (idx as f32) * 0.25 - 1.0)
        .collect();
    let weight_values: Vec<f32> = vec![0.5, -0.25, 0.75, 1.0];
    let bias_values: Vec<f32> = vec![0.5, -1.5];

    let input = Tensor::from_vec(
        input_values.clone(),
        Shape::from_dims(&[n, h, w, ic]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let weights = Tensor::from_vec(
        weight_values.clone(),
        Shape::from_dims(&[kh, kw, ic, oc]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let bias = Tensor::from_vec(bias_values.clone(), Shape::from_dims(&[oc]), cuda.clone())?
        .to_dtype(DType::BF16)?;

    let output = cuda_ops_bf16::conv2d_bf16(
        &input,
        &weights,
        Some(&bias),
        (1, 1),
        (0, 0),
        (1, 1),
        1,
        ConvActivation::None,
    )?;

    let host = output.to_dtype(DType::F32)?.to_vec_f32()?;
    let mut expected = vec![0f32; host.len()];
    for oh in 0..h {
        for ow in 0..w {
            for co in 0..oc {
                let mut acc = bias_values[co];
                for ci in 0..ic {
                    let x_idx = (((oh * w) + ow) * ic + ci) as usize;
                    let w_idx = (ci * oc + co) as usize;
                    acc += input_values[x_idx] * weight_values[w_idx];
                }
                expected[((oh * w + ow) * oc + co) as usize] = acc;
            }
        }
    }

    for (idx, (exp, got)) in expected.iter().zip(host.iter()).enumerate() {
        let diff = (exp - got).abs();
        assert!(
            diff < 2e-2,
            "bias conv mismatch at {}: expected {}, got {}, diff {}",
            idx,
            exp,
            got,
            diff
        );
    }

    Ok(())
}

#[test]
fn conv2d_bf16_gelu_activation_matches_reference() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let cuda = device.cuda_device_arc();

    let input_values = vec![-2.0f32, -0.5, 0.5, 2.0];
    let weights = vec![1.25f32];

    let input = Tensor::from_vec(
        input_values.clone(),
        Shape::from_dims(&[1, 2, 2, 1]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let weight = Tensor::from_vec(weights, Shape::from_dims(&[1, 1, 1, 1]), cuda.clone())?
        .to_dtype(DType::BF16)?;

    let output = cuda_ops_bf16::conv2d_bf16(
        &input,
        &weight,
        None,
        (1, 1),
        (0, 0),
        (1, 1),
        1,
        ConvActivation::Gelu,
    )?;

    let host = output.to_dtype(DType::F32)?.to_vec_f32()?;
    let expected: Vec<f32> = input_values
        .iter()
        .map(|&v| gelu_tanh_reference(v * 1.25))
        .collect();
    for (idx, (exp, got)) in expected.iter().zip(host.iter()).enumerate() {
        let diff = (exp - got).abs();
        assert!(
            diff < 2e-2,
            "gelu fusion mismatch at {}: expected {}, got {}, diff {}",
            idx,
            exp,
            got,
            diff
        );
    }
    Ok(())
}

#[test]
fn conv2d_autotune_disable_env_forces_fallback() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let cuda = device.cuda_device_arc();

    let prev = std::env::var("FLAME_CONV2D_AUTOTUNE").ok();
    std::env::set_var("FLAME_CONV2D_AUTOTUNE", "0");
    cuda_ops_bf16::reset_conv2d_autotune_stats()?;

    let input = Tensor::zeros_dtype(Shape::from_dims(&[1, 32, 32, 8]), DType::BF16, cuda.clone())?;
    let weight = Tensor::zeros_dtype(Shape::from_dims(&[3, 3, 8, 16]), DType::BF16, cuda.clone())?;
    let _ = cuda_ops_bf16::conv2d_bf16(
        &input,
        &weight,
        None,
        (1, 1),
        (1, 1),
        (1, 1),
        1,
        ConvActivation::None,
    )?;

    let stats = cuda_ops_bf16::conv2d_autotune_stats()?;
    if let Some(prev) = prev {
        std::env::set_var("FLAME_CONV2D_AUTOTUNE", prev);
    } else {
        std::env::remove_var("FLAME_CONV2D_AUTOTUNE");
    }

    assert_eq!(stats.tuned, 0, "autotune disabled should not tune");
    assert!(
        stats.fallbacks >= 1,
        "expect fallback count increment when autotune disabled; stats={:?}",
        stats
    );
    Ok(())
}

#[test]
fn conv2d_bf16_stride_gt_one_matches_reference() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let cuda = device.cuda_device_arc();

    let n = 1usize;
    let h = 7usize;
    let w = 6usize;
    let cin = 4usize;
    let cout = 3usize;
    let kh = 3usize;
    let kw = 2usize;
    let groups = 1usize;
    let stride = (2usize, 3usize);
    let pad = (1usize, 0usize);
    let dilation = (1usize, 1usize);

    let input_values: Vec<f32> = (0..(n * h * w * cin))
        .map(|i| ((i * 7 % 19) as f32) * 0.125 - 1.0)
        .collect();
    let weight_values: Vec<f32> = (0..(kh * kw * cin * cout))
        .map(|i| ((i * 5 % 23) as f32) * 0.05 - 0.5)
        .collect();

    let input = Tensor::from_vec(
        input_values.clone(),
        Shape::from_dims(&[n, h, w, cin]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let weight = Tensor::from_vec(
        weight_values.clone(),
        Shape::from_dims(&[kh, kw, cin / groups, cout]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;

    let output = cuda_ops_bf16::conv2d_bf16(
        &input,
        &weight,
        None,
        (stride.0 as i32, stride.1 as i32),
        (pad.0 as i32, pad.1 as i32),
        (dilation.0 as i32, dilation.1 as i32),
        groups as i32,
        ConvActivation::None,
    )?;

    let host = output.to_dtype(DType::F32)?.to_vec_f32()?;
    let expected = conv2d_reference_nhwc(
        &input_values,
        &weight_values,
        None,
        n,
        h,
        w,
        cin,
        kh,
        kw,
        groups,
        stride,
        pad,
        dilation,
        cout,
    );

    for (idx, (exp, got)) in expected.iter().zip(host.iter()).enumerate() {
        assert!(
            (exp - got).abs() < 3e-2,
            "stride conv mismatch at {}: expected {}, got {}",
            idx,
            exp,
            got
        );
    }
    Ok(())
}

#[test]
fn conv2d_bf16_dilation_gt_one_matches_reference() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let cuda = device.cuda_device_arc();

    let n = 1usize;
    let h = 6usize;
    let w = 6usize;
    let cin = 2usize;
    let cout = 2usize;
    let kh = 3usize;
    let kw = 3usize;
    let groups = 1usize;
    let stride = (1usize, 1usize);
    let pad = (2usize, 2usize);
    let dilation = (2usize, 2usize);

    let input_values: Vec<f32> = (0..(n * h * w * cin))
        .map(|i| ((i * 11 % 17) as f32) * 0.08 - 0.6)
        .collect();
    let weight_values: Vec<f32> = (0..(kh * kw * cin * cout))
        .map(|i| ((i * 13 % 29) as f32) * 0.04 - 0.3)
        .collect();

    let input = Tensor::from_vec(
        input_values.clone(),
        Shape::from_dims(&[n, h, w, cin]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let weight = Tensor::from_vec(
        weight_values.clone(),
        Shape::from_dims(&[kh, kw, cin / groups, cout]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;

    let output = cuda_ops_bf16::conv2d_bf16(
        &input,
        &weight,
        None,
        (stride.0 as i32, stride.1 as i32),
        (pad.0 as i32, pad.1 as i32),
        (dilation.0 as i32, dilation.1 as i32),
        groups as i32,
        ConvActivation::None,
    )?;

    let host = output.to_dtype(DType::F32)?.to_vec_f32()?;
    let expected = conv2d_reference_nhwc(
        &input_values,
        &weight_values,
        None,
        n,
        h,
        w,
        cin,
        kh,
        kw,
        groups,
        stride,
        pad,
        dilation,
        cout,
    );

    for (idx, (exp, got)) in expected.iter().zip(host.iter()).enumerate() {
        assert!(
            (exp - got).abs() < 3e-2,
            "dilation conv mismatch at {}: expected {}, got {}",
            idx,
            exp,
            got
        );
    }
    Ok(())
}

#[test]
fn conv2d_bf16_groups_four_matches_reference() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let cuda = device.cuda_device_arc();

    let n = 1usize;
    let h = 4usize;
    let w = 5usize;
    let cin = 8usize;
    let cout = 8usize;
    let groups = 4usize;
    let kh = 1usize;
    let kw = 3usize;
    let stride = (1usize, 1usize);
    let pad = (0usize, 1usize);
    let dilation = (1usize, 1usize);

    let input_values: Vec<f32> = (0..(n * h * w * cin))
        .map(|i| ((i * 3 % 37) as f32) * 0.06 - 0.4)
        .collect();
    let weight_values: Vec<f32> = (0..(kh * kw * (cin / groups) * cout))
        .map(|i| ((i * 19 % 31) as f32) * 0.03 - 0.25)
        .collect();

    let input = Tensor::from_vec(
        input_values.clone(),
        Shape::from_dims(&[n, h, w, cin]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let weight = Tensor::from_vec(
        weight_values.clone(),
        Shape::from_dims(&[kh, kw, cin / groups, cout]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;

    let output = cuda_ops_bf16::conv2d_bf16(
        &input,
        &weight,
        None,
        (stride.0 as i32, stride.1 as i32),
        (pad.0 as i32, pad.1 as i32),
        (dilation.0 as i32, dilation.1 as i32),
        groups as i32,
        ConvActivation::None,
    )?;

    let host = output.to_dtype(DType::F32)?.to_vec_f32()?;
    let expected = conv2d_reference_nhwc(
        &input_values,
        &weight_values,
        None,
        n,
        h,
        w,
        cin,
        kh,
        kw,
        groups,
        stride,
        pad,
        dilation,
        cout,
    );

    for (idx, (exp, got)) in expected.iter().zip(host.iter()).enumerate() {
        assert!(
            (exp - got).abs() < 3e-2,
            "grouped conv mismatch at {}: expected {}, got {}",
            idx,
            exp,
            got
        );
    }
    Ok(())
}

fn sdpa_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    b: usize,
    h: usize,
    q_len: usize,
    k_len: usize,
    dh: usize,
    dv: usize,
    scale: f32,
    causal: bool,
    mask_heads: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; b * h * q_len * dv];
    for batch in 0..b {
        for head in 0..h {
            let mask_head = if mask_heads == 1 { 0 } else { head };
            for q_row in 0..q_len {
                let mut logits = vec![f32::NEG_INFINITY; k_len];
                for k_col in 0..k_len {
                    if causal && k_col > q_row {
                        continue;
                    }
                    if let Some(m) = mask {
                        let mask_idx = (((batch * mask_heads + mask_head) * q_len + q_row) * k_len
                            + k_col) as usize;
                        if m[mask_idx] < 0.5 {
                            continue;
                        }
                    }
                    let mut dot = 0f32;
                    for d in 0..dh {
                        let q_idx = (((batch * h + head) * q_len + q_row) * dh + d) as usize;
                        let k_idx = (((batch * h + head) * k_len + k_col) * dh + d) as usize;
                        dot += q[q_idx] * k[k_idx];
                    }
                    logits[k_col] = dot * scale;
                }

                let mut max_val = f32::NEG_INFINITY;
                for &val in &logits {
                    if val > max_val {
                        max_val = val;
                    }
                }
                let mut weights = vec![0f32; k_len];
                let mut sum = 0f32;
                for (idx, &val) in logits.iter().enumerate() {
                    if val == f32::NEG_INFINITY {
                        continue;
                    }
                    let w = (val - max_val).exp();
                    weights[idx] = w;
                    sum += w;
                }
                if sum > 0.0 {
                    let inv_sum = 1.0 / sum;
                    for w in &mut weights {
                        *w *= inv_sum;
                    }
                }

                for d in 0..dv {
                    let mut acc = 0f32;
                    for k_col in 0..k_len {
                        let v_idx = (((batch * h + head) * k_len + k_col) * dv + d) as usize;
                        acc += weights[k_col] * v[v_idx];
                    }
                    let out_idx = (((batch * h + head) * q_len + q_row) * dv + d) as usize;
                    out[out_idx] = acc;
                }
            }
        }
    }
    out
}

fn sdpa_reference_chunked_bf16(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    b: usize,
    h: usize,
    q_len: usize,
    k_len: usize,
    dh: usize,
    dv: usize,
    scale: f32,
    causal: bool,
    mask_heads: usize,
    q_chunk: usize,
    k_chunk: usize,
) -> Vec<f32> {
    let q_bf16: Vec<bf16> = q.iter().copied().map(bf16::from_f32).collect();
    let k_bf16: Vec<bf16> = k.iter().copied().map(bf16::from_f32).collect();
    let v_bf16: Vec<bf16> = v.iter().copied().map(bf16::from_f32).collect();

    let mut out = vec![0f32; b * h * q_len * dv];
    for batch in 0..b {
        for head in 0..h {
            let mask_head = if mask_heads == 1 { 0 } else { head };
            let mut q_row = 0;
            while q_row < q_len {
                let rows = q_chunk.min(q_len - q_row);
                for local_row in 0..rows {
                    let global_q = q_row + local_row;
                    let mut prev_max = f32::NEG_INFINITY;
                    let mut prev_sum = 0f32;
                    let mut row_out = vec![0f32; dv];

                    let mut k_offset = 0;
                    while k_offset < k_len {
                        let block_k = k_chunk.min(k_len - k_offset);
                        if block_k == 0 {
                            break;
                        }
                        let mut block_logits = vec![f32::NEG_INFINITY; block_k];
                        for col in 0..block_k {
                            if causal && (k_offset + col) > global_q {
                                continue;
                            }
                            if let Some(mask_vals) = mask {
                                let mask_idx =
                                    (((batch * mask_heads + mask_head) * q_len + global_q) * k_len
                                        + (k_offset + col))
                                        as usize;
                                if mask_vals[mask_idx] < 0.5 {
                                    continue;
                                }
                            }
                            let mut dot = 0f32;
                            for d in 0..dh {
                                let q_idx =
                                    (((batch * h + head) * q_len + global_q) * dh + d) as usize;
                                let k_idx = (((batch * h + head) * k_len + (k_offset + col)) * dh
                                    + d) as usize;
                                let q_val = f32::from(q_bf16[q_idx]);
                                let k_val = f32::from(k_bf16[k_idx]);
                                dot += q_val * k_val;
                            }
                            block_logits[col] = dot * scale;
                        }

                        let block_max = block_logits
                            .iter()
                            .copied()
                            .fold(f32::NEG_INFINITY, f32::max);
                        let new_max = prev_max.max(block_max);
                        let prev_scale = if prev_sum > 0.0 {
                            (prev_max - new_max).exp()
                        } else {
                            0.0
                        };
                        prev_sum *= prev_scale;
                        for val in &mut row_out {
                            *val *= prev_scale;
                        }

                        for col in 0..block_k {
                            let val = block_logits[col];
                            if !val.is_finite() {
                                continue;
                            }
                            let weight = (val - new_max).exp();
                            prev_sum += weight;
                            let v_base =
                                (((batch * h + head) * k_len + (k_offset + col)) * dv) as usize;
                            for d in 0..dv {
                                let v_idx = v_base + d;
                                let v_val = f32::from(v_bf16[v_idx]);
                                row_out[d] += weight * v_val;
                            }
                        }

                        prev_max = new_max;
                        k_offset += block_k;
                    }

                    let norm = if prev_sum > 0.0 { 1.0 / prev_sum } else { 0.0 };
                    let out_idx = (((batch * h + head) * q_len + global_q) * dv) as usize;
                    for d in 0..dv {
                        out[out_idx + d] = row_out[d] * norm;
                    }
                }
                q_row += rows;
            }
        }
    }
    out
}

#[test]
fn sdpa_stream_bf16_matches_reference() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let cuda = device.cuda_device_arc();

    let _ = cuda_ops_bf16::flush_sdpa_autotune_cache();
    cuda_ops_bf16::reset_sdpa_autotune_stats()?;

    let b = 1usize;
    let h = 2usize;
    let q_len = 5usize;
    let k_len = 6usize;
    let dh = 4usize;
    let dv = dh;
    let chunk = 3usize;
    let scale = 1.0f32 / (dh as f32).sqrt();

    let q_vals: Vec<f32> = (0..(b * h * q_len * dh))
        .map(|i| ((i * 11 % 23) as f32) * 0.05 - 0.8)
        .collect();
    let k_vals: Vec<f32> = (0..(b * h * k_len * dh))
        .map(|i| ((i * 7 % 29) as f32) * 0.04 - 0.6)
        .collect();
    let v_vals: Vec<f32> = (0..(b * h * k_len * dv))
        .map(|i| ((i * 13 % 31) as f32) * 0.06 - 0.5)
        .collect();

    let q = Tensor::from_vec(
        q_vals.clone(),
        Shape::from_dims(&[b, h, q_len, dh]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let k = Tensor::from_vec(
        k_vals.clone(),
        Shape::from_dims(&[b, h, k_len, dh]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let v = Tensor::from_vec(
        v_vals.clone(),
        Shape::from_dims(&[b, h, k_len, dv]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;

    let out = cuda_ops_bf16::sdpa_stream_bf16(&q, &k, &v, None, chunk, false, Some(scale))?;
    let out_host = out.to_dtype(DType::F32)?.to_vec_f32()?;
    let expected = sdpa_reference(
        &q_vals, &k_vals, &v_vals, None, b, h, q_len, k_len, dh, dv, scale, false, h,
    );

    let stats = cuda_ops_bf16::sdpa_autotune_stats()?;
    let q_chunk = std::cmp::max(stats.last_q_chunk as usize, 1).min(q_len);
    let k_chunk = std::cmp::max(stats.last_k_chunk as usize, 1).min(k_len);
    let chunked_expected = sdpa_reference_chunked_bf16(
        &q_vals, &k_vals, &v_vals, None, b, h, q_len, k_len, dh, dv, scale, false, h, q_chunk,
        k_chunk,
    );

    for (idx, got) in out_host.iter().enumerate() {
        let chunked = chunked_expected[idx];
        assert!(
            (chunked - got).abs() < 1e-2,
            "sdpa chunked mismatch at {}: chunked {}, got {}",
            idx,
            chunked,
            got
        );
    }

    for (idx, (exp, chunked)) in expected.iter().zip(chunked_expected.iter()).enumerate() {
        assert!(
            (exp - chunked).abs() < 3e-2,
            "sdpa reference drift at {}: expected {}, chunked {}",
            idx,
            exp,
            chunked
        );
    }
    Ok(())
}

#[test]
fn bf16_permute_patchify_roundtrip() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let cuda = device.cuda_device_arc();

    let b = 1usize;
    let c = 4usize;
    let h = 8usize;
    let w = 8usize;
    let numel = b * c * h * w;
    let values: Vec<f32> = (0..numel).map(|i| i as f32).collect();
    let tensor = Tensor::from_vec(
        values.clone(),
        Shape::from_dims(&[b, c, h, w]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;

    let patchified = tensor
        .reshape(&[b, c, h / 2, 2, w / 2, 2])?
        .permute(&[0, 2, 4, 1, 3, 5])?
        .reshape(&[b, (h / 2) * (w / 2), c * 4])?;

    let unpatchified = patchified
        .reshape(&[b, h / 2, w / 2, c, 2, 2])?
        .permute(&[0, 3, 1, 4, 2, 5])?
        .reshape(&[b, c, h, w])?;

    assert_eq!(patchified.dtype(), DType::BF16);
    assert_eq!(unpatchified.dtype(), DType::BF16);

    let restored = unpatchified.to_dtype(DType::F32)?.to_vec_f32()?;
    assert_eq!(restored, values);
    Ok(())
}

#[test]
fn bf16_slice_matches_reference() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let cuda = device.cuda_device_arc();

    let shape = [2usize, 3usize, 4usize];
    let numel = shape.iter().product();
    let values: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5).collect();
    let tensor = Tensor::from_vec(values.clone(), Shape::from_dims(&shape), cuda.clone())?
        .to_dtype(DType::BF16)?;

    let slice = tensor.slice(&[(0, 2), (1, 3), (1, 4)])?;
    assert_eq!(slice.dtype(), DType::BF16);

    let slice_host = slice.to_dtype(DType::F32)?.to_vec_f32()?;
    let mut expected = Vec::new();
    for b in 0..2 {
        for c in 1..3 {
            for x in 1..4 {
                let idx = ((b * shape[1] + c) * shape[2] + x) as usize;
                expected.push(values[idx]);
            }
        }
    }
    assert_eq!(slice_host, expected);
    Ok(())
}

#[test]
fn sdpa_autotune_stats_updates() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let cuda = device.cuda_device_arc();

    cuda_ops_bf16::reset_sdpa_autotune_stats()?;

    let b = 1usize;
    let h = 1usize;
    let q_len = 4usize;
    let k_len = 5usize;
    let dh = 2usize;
    let chunk = 2usize;

    let q_vals: Vec<f32> = (0..(b * h * q_len * dh))
        .map(|i| ((i * 5 % 17) as f32) * 0.07 - 0.3)
        .collect();
    let k_vals: Vec<f32> = (0..(b * h * k_len * dh))
        .map(|i| ((i * 7 % 19) as f32) * 0.05 - 0.25)
        .collect();
    let v_vals: Vec<f32> = (0..(b * h * k_len * dh))
        .map(|i| ((i * 11 % 23) as f32) * 0.04 - 0.2)
        .collect();

    let q = Tensor::from_vec(
        q_vals.clone(),
        Shape::from_dims(&[b, h, q_len, dh]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let k = Tensor::from_vec(
        k_vals.clone(),
        Shape::from_dims(&[b, h, k_len, dh]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let v = Tensor::from_vec(
        v_vals.clone(),
        Shape::from_dims(&[b, h, k_len, dh]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;

    let _ = cuda_ops_bf16::sdpa_stream_bf16(&q, &k, &v, None, chunk, false, None)?;
    let stats_first = cuda_ops_bf16::sdpa_autotune_stats()?;
    assert!(stats_first.last_q_chunk > 0);
    assert!(stats_first.last_k_chunk > 0);
    assert!(stats_first.cache_entries >= 1);

    let _ = cuda_ops_bf16::sdpa_stream_bf16(&q, &k, &v, None, chunk, false, None)?;
    let stats_second = cuda_ops_bf16::sdpa_autotune_stats()?;
    assert!(stats_second.cache_hits >= stats_first.cache_hits);
    assert!(stats_second.last_q_chunk == stats_first.last_q_chunk);
    assert!(stats_second.last_k_chunk == stats_first.last_k_chunk);
    assert!(stats_second.cache_entries == stats_first.cache_entries);
    assert!(stats_second.last_plan_source > 0);
    assert_eq!(stats_second.last_shape_q, q_len as u64);
    assert_eq!(stats_second.last_shape_k, k_len as u64);

    Ok(())
}

#[cfg(feature = "strict_bf16")]
#[test]
#[ignore = "Run manually with STRICT_BF16 enabled to verify storage guardrails"]
fn sdpa_stream_bf16_preserves_bf16_storage_under_strict_guard() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let _lock = STRICT_BF16_TEST_GUARD
        .lock()
        .expect("strict guard mutex poisoned");
    strict::reset_counters();

    let cuda = device.cuda_device_arc();

    let b = 1usize;
    let h = 1usize;
    let q_len = 4usize;
    let k_len = 4usize;
    let dh = 8usize;
    let dv = dh;
    let chunk = 2usize;
    let scale = 1.0f32 / (dh as f32).sqrt();

    let q_vals: Vec<f32> = (0..(b * h * q_len * dh))
        .map(|i| ((i * 5 % 13) as f32) * 0.05 - 0.4)
        .collect();
    let k_vals: Vec<f32> = (0..(b * h * k_len * dh))
        .map(|i| ((i * 7 % 17) as f32) * 0.04 - 0.35)
        .collect();
    let v_vals: Vec<f32> = (0..(b * h * k_len * dv))
        .map(|i| ((i * 11 % 19) as f32) * 0.03 - 0.25)
        .collect();

    let q = Tensor::from_vec(
        q_vals.clone(),
        Shape::from_dims(&[b, h, q_len, dh]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let k = Tensor::from_vec(
        k_vals.clone(),
        Shape::from_dims(&[b, h, k_len, dh]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let v = Tensor::from_vec(
        v_vals.clone(),
        Shape::from_dims(&[b, h, k_len, dv]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;

    let out = cuda_ops_bf16::sdpa_stream_bf16(&q, &k, &v, None, chunk, false, Some(scale))?;

    assert_eq!(
        out.dtype(),
        DType::BF16,
        "sdpa_stream_bf16 must preserve BF16 logical dtype"
    );
    assert_eq!(
        out.storage_dtype(),
        DType::BF16,
        "sdpa_stream_bf16 must preserve BF16 storage dtype"
    );

    let telemetry = strict::telemetry_snapshot();
    assert_eq!(
        telemetry.f32_graph_casts, 0,
        "sdpa_stream_bf16 should not promote tensors to FP32 storage under STRICT_BF16"
    );
    assert_eq!(
        telemetry.param_f32_store, 0,
        "sdpa_stream_bf16 should not persist parameters as FP32 under STRICT_BF16"
    );

    strict::reset_counters();
    Ok(())
}

#[test]
fn sdpa_stream_bf16_causal_matches_reference() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let cuda = device.cuda_device_arc();

    let b = 1usize;
    let h = 1usize;
    let q_len = 6usize;
    let k_len = 6usize;
    let dh = 3usize;
    let dv = dh;
    let chunk = 4usize;
    let scale = 1.0f32 / (dh as f32).sqrt();

    let q_vals: Vec<f32> = (0..(b * h * q_len * dh))
        .map(|i| ((i * 5 % 17) as f32) * 0.07 - 0.3)
        .collect();
    let k_vals: Vec<f32> = (0..(b * h * k_len * dh))
        .map(|i| ((i * 7 % 19) as f32) * 0.05 - 0.25)
        .collect();
    let v_vals: Vec<f32> = (0..(b * h * k_len * dv))
        .map(|i| ((i * 11 % 23) as f32) * 0.04 - 0.2)
        .collect();

    let q = Tensor::from_vec(
        q_vals.clone(),
        Shape::from_dims(&[b, h, q_len, dh]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let k = Tensor::from_vec(
        k_vals.clone(),
        Shape::from_dims(&[b, h, k_len, dh]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let v = Tensor::from_vec(
        v_vals.clone(),
        Shape::from_dims(&[b, h, k_len, dv]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;

    let out = cuda_ops_bf16::sdpa_stream_bf16(&q, &k, &v, None, chunk, true, Some(scale))?;
    let out_host = out.to_dtype(DType::F32)?.to_vec_f32()?;
    let expected = sdpa_reference(
        &q_vals, &k_vals, &v_vals, None, b, h, q_len, k_len, dh, dv, scale, true, h,
    );

    for (idx, (exp, got)) in expected.iter().zip(out_host.iter()).enumerate() {
        assert!(
            (exp - got).abs() < 3e-2,
            "causal sdpa mismatch at {}: expected {}, got {}",
            idx,
            exp,
            got
        );
    }
    Ok(())
}

#[test]
fn sdpa_stream_bf16_mask_matches_reference() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let cuda = device.cuda_device_arc();

    let b = 2usize;
    let h = 1usize;
    let q_len = 4usize;
    let k_len = 5usize;
    let dh = 2usize;
    let dv = dh;
    let chunk = 2usize;
    let scale = 1.0f32 / (dh as f32).sqrt();

    let q_vals: Vec<f32> = (0..(b * h * q_len * dh))
        .map(|i| ((i * 3 % 13) as f32) * 0.09 - 0.45)
        .collect();
    let k_vals: Vec<f32> = (0..(b * h * k_len * dh))
        .map(|i| ((i * 5 % 17) as f32) * 0.04 - 0.35)
        .collect();
    let v_vals: Vec<f32> = (0..(b * h * k_len * dv))
        .map(|i| ((i * 7 % 23) as f32) * 0.05 - 0.3)
        .collect();

    let mask_vals: Vec<f32> = (0..(b * 1 * q_len * k_len))
        .map(|i| if i % 3 == 0 { 0.0 } else { 1.0 })
        .collect();

    let q = Tensor::from_vec(
        q_vals.clone(),
        Shape::from_dims(&[b, h, q_len, dh]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let k = Tensor::from_vec(
        k_vals.clone(),
        Shape::from_dims(&[b, h, k_len, dh]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let v = Tensor::from_vec(
        v_vals.clone(),
        Shape::from_dims(&[b, h, k_len, dv]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let mask_tensor = Tensor::from_vec(
        mask_vals.clone(),
        Shape::from_dims(&[b, 1, q_len, k_len]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;

    let out =
        cuda_ops_bf16::sdpa_stream_bf16(&q, &k, &v, Some(&mask_tensor), chunk, false, Some(scale))?;
    let out_host = out.to_dtype(DType::F32)?.to_vec_f32()?;
    let expected = sdpa_reference(
        &q_vals,
        &k_vals,
        &v_vals,
        Some(&mask_vals),
        b,
        h,
        q_len,
        k_len,
        dh,
        dv,
        scale,
        false,
        1,
    );

    for (idx, (exp, got)) in expected.iter().zip(out_host.iter()).enumerate() {
        assert!(
            (exp - got).abs() < 3e-2,
            "mask sdpa mismatch at {}: expected {}, got {}",
            idx,
            exp,
            got
        );
    }
    Ok(())
}

#[test]
fn sdpa_stream_bf16_extreme_scale_no_nan() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let cuda = device.cuda_device_arc();

    let b = 1usize;
    let h = 1usize;
    let q_len = 256usize;
    let k_len = 256usize;
    let dh = 64usize;
    let dv = 64usize;
    let chunk = 128usize;
    let scale = 1.0e4f32;

    let total_q = b * h * q_len * dh;
    let total_k = b * h * k_len * dh;
    let total_v = b * h * k_len * dv;

    let q_vals: Vec<f32> = (0..total_q)
        .map(|i| if i % 2 == 0 { 6.0e4f32 } else { -6.0e4f32 })
        .collect();
    let k_vals: Vec<f32> = (0..total_k)
        .map(|i| {
            if (i / dh) % 3 == 0 {
                5.0e4f32
            } else {
                -5.0e4f32
            }
        })
        .collect();
    let v_vals: Vec<f32> = (0..total_v)
        .map(|i| ((i * 13 % 97) as f32) * 1e-1 - 4.0)
        .collect();

    let q = Tensor::from_vec(q_vals, Shape::from_dims(&[b, h, q_len, dh]), cuda.clone())?
        .to_dtype(DType::BF16)?;
    let k = Tensor::from_vec(k_vals, Shape::from_dims(&[b, h, k_len, dh]), cuda.clone())?
        .to_dtype(DType::BF16)?;
    let v = Tensor::from_vec(v_vals, Shape::from_dims(&[b, h, k_len, dv]), cuda.clone())?
        .to_dtype(DType::BF16)?;

    let out = cuda_ops_bf16::sdpa_stream_bf16(&q, &k, &v, None, chunk, false, Some(scale))?;
    let host = out.to_dtype(DType::F32)?.to_vec_f32()?;

    for (idx, value) in host.iter().enumerate() {
        assert!(
            value.is_finite(),
            "expected finite attention output at index {idx}, found {value}"
        );
    }

    Ok(())
}

#[test]
fn sdpa_stream_bf16_value_dim_mismatch_matches_reference() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let cuda = device.cuda_device_arc();

    let b = 1usize;
    let h = 2usize;
    let q_len = 6usize;
    let k_len = 6usize;
    let dh = 8usize;
    let dv = 12usize;
    let chunk = 4usize;
    let scale = 1.0f32 / (dh as f32).sqrt();

    let q_vals: Vec<f32> = (0..(b * h * q_len * dh))
        .map(|i| ((i * 19 % 37) as f32) * 0.045 - 0.55)
        .collect();
    let k_vals: Vec<f32> = (0..(b * h * k_len * dh))
        .map(|i| ((i * 23 % 41) as f32) * 0.038 - 0.4)
        .collect();
    let v_vals: Vec<f32> = (0..(b * h * k_len * dv))
        .map(|i| ((i * 29 % 53) as f32) * 0.05 - 0.35)
        .collect();

    let q = Tensor::from_vec(
        q_vals.clone(),
        Shape::from_dims(&[b, h, q_len, dh]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let k = Tensor::from_vec(
        k_vals.clone(),
        Shape::from_dims(&[b, h, k_len, dh]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let v = Tensor::from_vec(
        v_vals.clone(),
        Shape::from_dims(&[b, h, k_len, dv]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;

    let out = cuda_ops_bf16::sdpa_stream_bf16(&q, &k, &v, None, chunk, false, Some(scale))?;
    let out_host = out.to_dtype(DType::F32)?.to_vec_f32()?;
    let expected = sdpa_reference(
        &q_vals, &k_vals, &v_vals, None, b, h, q_len, k_len, dh, dv, scale, false, h,
    );

    for (idx, (exp, got)) in expected.iter().zip(out_host.iter()).enumerate() {
        assert!(
            (exp - got).abs() < 5e-2,
            "dv != dh sdpa mismatch at {}: expected {}, got {}",
            idx,
            exp,
            got
        );
    }

    Ok(())
}

#[test]
fn sdpa_stream_bf16_long_sequence_autotune_exercise() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let cuda = device.cuda_device_arc();

    cuda_ops_bf16::reset_sdpa_autotune_stats()?;

    let b = 2usize;
    let h = 3usize;
    let q_len = 96usize;
    let k_len = 128usize;
    let dh = 32usize;
    let dv = 32usize;
    let chunk = 48usize;
    let scale = 1.0f32 / (dh as f32).sqrt();

    let total_q = b * h * q_len * dh;
    let total_kv = b * h * k_len * dv;

    let q_vals: Vec<f32> = (0..total_q)
        .map(|i| ((i * 37 % 101) as f32) * 0.01 - 0.5)
        .collect();
    let k_vals: Vec<f32> = (0..total_kv)
        .map(|i| ((i * 29 % 89) as f32) * 0.012 - 0.45)
        .collect();
    let v_vals: Vec<f32> = (0..total_kv)
        .map(|i| ((i * 41 % 97) as f32) * 0.015 - 0.4)
        .collect();

    let q = Tensor::from_vec(
        q_vals.clone(),
        Shape::from_dims(&[b, h, q_len, dh]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let k = Tensor::from_vec(
        k_vals.clone(),
        Shape::from_dims(&[b, h, k_len, dh]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let v = Tensor::from_vec(
        v_vals.clone(),
        Shape::from_dims(&[b, h, k_len, dv]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;

    let _ = cuda_ops_bf16::sdpa_stream_bf16(&q, &k, &v, None, chunk, false, Some(scale))?;
    let stats_first = cuda_ops_bf16::sdpa_autotune_stats()?;
    assert!(stats_first.last_q_chunk > 0);
    assert!(stats_first.last_k_chunk > 0);

    let out_second = cuda_ops_bf16::sdpa_stream_bf16(&q, &k, &v, None, chunk, false, Some(scale))?;
    let stats_second = cuda_ops_bf16::sdpa_autotune_stats()?;
    assert!(stats_second.cache_hits >= stats_first.cache_hits);
    assert_eq!(stats_second.last_q_chunk, stats_first.last_q_chunk);
    assert_eq!(stats_second.last_k_chunk, stats_first.last_k_chunk);

    let out_host = out_second.to_dtype(DType::F32)?.to_vec_f32()?;
    let q_chunk = std::cmp::max(stats_second.last_q_chunk as usize, 1).min(q_len);
    let k_chunk = std::cmp::max(stats_second.last_k_chunk as usize, 1).min(k_len);
    let chunked_expected = sdpa_reference_chunked_bf16(
        &q_vals, &k_vals, &v_vals, None, b, h, q_len, k_len, dh, dv, scale, false, h, q_chunk,
        k_chunk,
    );
    let expected = sdpa_reference(
        &q_vals, &k_vals, &v_vals, None, b, h, q_len, k_len, dh, dv, scale, false, h,
    );

    for (idx, got) in out_host.iter().enumerate() {
        assert!(
            (chunked_expected[idx] - got).abs() < 1e-2,
            "long sequence chunked mismatch at {}: expected {}, got {}",
            idx,
            chunked_expected[idx],
            got
        );
    }

    for (idx, (exp, chunked)) in expected.iter().zip(chunked_expected.iter()).enumerate() {
        assert!(
            (exp - chunked).abs() < 5e-2,
            "long sequence reference drift at {}: expected {}, chunked {}",
            idx,
            exp,
            chunked
        );
    }

    Ok(())
}

#[test]
fn sdpa_stream_bf16_long_sequence_causal_matches_chunked_reference() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let cuda = device.cuda_device_arc();

    let b = 1usize;
    let h = 2usize;
    let q_len = 512usize;
    let k_len = 512usize;
    let dh = 64usize;
    let dv = 64usize;
    let chunk = 256usize;
    let scale = 1.0f32 / (dh as f32).sqrt();

    let total_q = b * h * q_len * dh;
    let total_k = b * h * k_len * dh;
    let total_v = b * h * k_len * dv;

    let q_vals: Vec<f32> = (0..total_q)
        .map(|i| ((i * 23 % 127) as f32) * 0.007 - 0.25)
        .collect();
    let k_vals: Vec<f32> = (0..total_k)
        .map(|i| ((i * 31 % 131) as f32) * 0.006 - 0.2)
        .collect();
    let v_vals: Vec<f32> = (0..total_v)
        .map(|i| ((i * 17 % 113) as f32) * 0.008 - 0.22)
        .collect();

    let q = Tensor::from_vec(
        q_vals.clone(),
        Shape::from_dims(&[b, h, q_len, dh]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let k = Tensor::from_vec(
        k_vals.clone(),
        Shape::from_dims(&[b, h, k_len, dh]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let v = Tensor::from_vec(
        v_vals.clone(),
        Shape::from_dims(&[b, h, k_len, dv]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;

    let out = cuda_ops_bf16::sdpa_stream_bf16(&q, &k, &v, None, chunk, true, Some(scale))?;
    let out_host = out.to_dtype(DType::F32)?.to_vec_f32()?;

    let expected = sdpa_reference_chunked_bf16(
        &q_vals, &k_vals, &v_vals, None, b, h, q_len, k_len, dh, dv, scale, true, h, chunk, chunk,
    );

    for (idx, got) in out_host.iter().enumerate() {
        let exp = expected[idx];
        assert!(
            (exp - *got).abs() < 4e-2,
            "sdpa_stream_bf16 causal long sequence mismatch at {}: expected {} got {}",
            idx,
            exp,
            got
        );
    }

    Ok(())
}

#[test]
fn sdpa_stream_bf16_long_sequence_causal_matches_reference() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let cuda = device.cuda_device_arc();

    let b = 1usize;
    let h = 2usize;
    let q_len = 512usize;
    let k_len = 512usize;
    let dh = 64usize;
    let dv = 64usize;
    let chunk = 256usize;
    let scale = 1.0f32 / (dh as f32).sqrt();

    let total_q = b * h * q_len * dh;
    let total_k = b * h * k_len * dh;
    let total_v = b * h * k_len * dv;

    let q_vals: Vec<f32> = (0..total_q)
        .map(|i| ((i * 23 % 127) as f32) * 0.007 - 0.25)
        .collect();
    let k_vals: Vec<f32> = (0..total_k)
        .map(|i| ((i * 31 % 131) as f32) * 0.006 - 0.2)
        .collect();
    let v_vals: Vec<f32> = (0..total_v)
        .map(|i| ((i * 17 % 113) as f32) * 0.008 - 0.22)
        .collect();

    let q = Tensor::from_vec(
        q_vals.clone(),
        Shape::from_dims(&[b, h, q_len, dh]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let k = Tensor::from_vec(
        k_vals.clone(),
        Shape::from_dims(&[b, h, k_len, dh]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let v = Tensor::from_vec(
        v_vals.clone(),
        Shape::from_dims(&[b, h, k_len, dv]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;

    let out = cuda_ops_bf16::sdpa_stream_bf16(&q, &k, &v, None, chunk, true, Some(scale))?;
    let out_host = out.to_dtype(DType::F32)?.to_vec_f32()?;

    let expected = sdpa_reference_chunked_bf16(
        &q_vals, &k_vals, &v_vals, None, b, h, q_len, k_len, dh, dv, scale, true, h, chunk, chunk,
    );

    for (idx, got) in out_host.iter().enumerate() {
        let exp = expected[idx];
        assert!(
            (exp - *got).abs() < 5e-2,
            "causal long sequence mismatch at {}: expected {} got {}",
            idx,
            exp,
            got
        );
        assert!(
            got.is_finite(),
            "causal long sequence produced non-finite output at {}: {}",
            idx,
            got
        );
    }

    Ok(())
}

#[test]
fn sdpa_stream_bf16_long_sequence_mask_broadcast() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    let cuda = device.cuda_device_arc();

    cuda_ops_bf16::reset_sdpa_autotune_stats()?;

    let b = 1usize;
    let h = 2usize;
    let q_len = 80usize;
    let k_len = 96usize;
    let dh = 16usize;
    let dv = 20usize;
    let chunk = 40usize;
    let scale = 1.0f32 / (dh as f32).sqrt();

    let total_q = b * h * q_len * dh;
    let total_kv = b * h * k_len * dv;

    let q_vals: Vec<f32> = (0..total_q)
        .map(|i| ((i * 17 % 67) as f32) * 0.013 - 0.42)
        .collect();
    let k_vals: Vec<f32> = (0..(b * h * k_len * dh))
        .map(|i| ((i * 23 % 71) as f32) * 0.011 - 0.38)
        .collect();
    let v_vals: Vec<f32> = (0..total_kv)
        .map(|i| ((i * 31 % 73) as f32) * 0.015 - 0.35)
        .collect();

    let mask_elems = b * 1 * q_len * k_len;
    let mask_vals: Vec<f32> = (0..mask_elems)
        .map(|i| if (i + (i / 7)) % 4 == 0 { 0.0 } else { 1.0 })
        .collect();

    let q = Tensor::from_vec(
        q_vals.clone(),
        Shape::from_dims(&[b, h, q_len, dh]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let k = Tensor::from_vec(
        k_vals.clone(),
        Shape::from_dims(&[b, h, k_len, dh]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let v = Tensor::from_vec(
        v_vals.clone(),
        Shape::from_dims(&[b, h, k_len, dv]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;
    let mask = Tensor::from_vec(
        mask_vals.clone(),
        Shape::from_dims(&[b, 1, q_len, k_len]),
        cuda.clone(),
    )?
    .to_dtype(DType::BF16)?;

    let _ = cuda_ops_bf16::sdpa_stream_bf16(&q, &k, &v, Some(&mask), chunk, false, Some(scale))?;
    let stats_first = cuda_ops_bf16::sdpa_autotune_stats()?;
    assert!(stats_first.last_q_chunk > 0);
    assert!(stats_first.last_k_chunk > 0);

    let out_second =
        cuda_ops_bf16::sdpa_stream_bf16(&q, &k, &v, Some(&mask), chunk, false, Some(scale))?;
    let stats_second = cuda_ops_bf16::sdpa_autotune_stats()?;
    assert!(stats_second.cache_hits >= stats_first.cache_hits);
    assert_eq!(stats_second.last_q_chunk, stats_first.last_q_chunk);
    assert_eq!(stats_second.last_k_chunk, stats_first.last_k_chunk);

    let out_host = out_second.to_dtype(DType::F32)?.to_vec_f32()?;
    let expected = sdpa_reference(
        &q_vals,
        &k_vals,
        &v_vals,
        Some(&mask_vals),
        b,
        h,
        q_len,
        k_len,
        dh,
        dv,
        scale,
        false,
        1,
    );

    for (idx, (exp, got)) in expected.iter().zip(out_host.iter()).enumerate() {
        assert!(
            (exp - got).abs() < 5e-2,
            "long sequence mask sdpa mismatch at {}: expected {}, got {}",
            idx,
            exp,
            got
        );
    }

    Ok(())
}
