#![cfg(feature = "cuda")]

//! Smoke + numerical-sanity tests for the ConvTranspose2d forward path that
//! reuses conv2d machinery (zero-insert + pad + kernel-flip + transpose).
//!
//! Coverage:
//!   - stride=1, padding=0 (trivial pass-through of a conv)
//!   - stride=2, padding=1 (the common up-sampling case)
//!   - groups/dilation unsupported paths return clean errors
//!   - output shape matches PyTorch formula

use flame_core::{
    cuda_ops::GpuOps, global_cuda_device, DType, Result, Shape, Tensor,
};

fn ones_bf16(dims: &[usize]) -> Result<Tensor> {
    let dev = global_cuda_device();
    Tensor::ones_dtype(Shape::from_dims(dims), DType::BF16, dev)
}

fn zeros_like_bf16(dims: &[usize]) -> Result<Tensor> {
    let dev = global_cuda_device();
    Tensor::zeros_dtype(Shape::from_dims(dims), DType::BF16, dev)
}

/// Expected output size for ConvTranspose2d (PyTorch formula, dilation=1):
/// out = (in - 1) * stride - 2*padding + kernel + output_padding
fn expected_hw(in_hw: usize, kernel: usize, stride: usize, padding: usize, out_pad: usize) -> usize {
    (in_hw - 1) * stride + kernel - 2 * padding + out_pad
}

#[test]
fn conv_transpose2d_stride1_pad0_shape_and_nonzero() -> Result<()> {
    // Input [1, 2, 3, 3], weight [2, 4, 2, 2]: out_channels=4 with kernel 2x2.
    // Expected output HW = (3-1)*1 - 0 + 2 + 0 = 4.
    let input = ones_bf16(&[1, 2, 3, 3])?;
    let weight = ones_bf16(&[2, 4, 2, 2])?;

    let out = GpuOps::conv_transpose2d_forward(
        &input, &weight, None, (1, 1), (0, 0), (0, 0), 1, (1, 1),
    )?;

    let dims = out.shape().dims();
    assert_eq!(dims, &[1, 4, 4, 4], "stride1 shape mismatch: got {:?}", dims);

    // All inputs and weights are 1.0, so every output cell is a positive sum.
    // Going through bf16 we accept tiny precision slack but cell value must be
    // well above zero.
    let out_f32 = out.to_dtype(DType::F32)?.to_vec()?;
    assert!(!out_f32.iter().any(|v| v.is_nan()), "output contains NaN");
    assert!(out_f32.iter().all(|v| *v >= 0.0), "output has negative cells");
    assert!(out_f32.iter().any(|v| *v > 0.0), "output is all zeros");
    Ok(())
}

#[test]
fn conv_transpose2d_stride2_pad1_upsample_shape() -> Result<()> {
    // Common U-Net upsampling: [B=1, Cin=4, H=4, W=4] → [B=1, Cout=2, H*2, W*2]
    // via kernel 4x4, stride 2, padding 1, output_padding 0.
    // Expected HW = (4-1)*2 - 2 + 4 + 0 = 8.
    let input = ones_bf16(&[1, 4, 4, 4])?;
    let weight = ones_bf16(&[4, 2, 4, 4])?;
    let bias = Some(zeros_like_bf16(&[2])?);

    let out = GpuOps::conv_transpose2d_forward(
        &input,
        &weight,
        bias.as_ref(),
        (2, 2),
        (1, 1),
        (0, 0),
        1,
        (1, 1),
    )?;

    let dims = out.shape().dims();
    let h_expected = expected_hw(4, 4, 2, 1, 0);
    let w_expected = expected_hw(4, 4, 2, 1, 0);
    assert_eq!(dims, &[1, 2, h_expected, w_expected], "stride2 shape mismatch");

    let out_f32 = out.to_dtype(DType::F32)?.to_vec()?;
    assert!(!out_f32.iter().any(|v| v.is_nan()), "NaN in stride2 output");
    assert!(out_f32.iter().any(|v| *v > 0.0), "stride2 output all zero");
    Ok(())
}

#[test]
fn conv_transpose2d_rejects_groups_gt_1() -> Result<()> {
    let input = ones_bf16(&[1, 4, 2, 2])?;
    let weight = ones_bf16(&[4, 2, 2, 2])?;
    let err = GpuOps::conv_transpose2d_forward(
        &input, &weight, None, (1, 1), (0, 0), (0, 0), 2, (1, 1),
    );
    assert!(err.is_err(), "groups>1 should fail");
    let msg = format!("{}", err.unwrap_err());
    assert!(
        msg.contains("groups=1"),
        "expected explicit groups error, got {msg}"
    );
    Ok(())
}

#[test]
fn conv_transpose2d_rejects_dilation_ne_1() -> Result<()> {
    let input = ones_bf16(&[1, 2, 2, 2])?;
    let weight = ones_bf16(&[2, 2, 2, 2])?;
    let err = GpuOps::conv_transpose2d_forward(
        &input, &weight, None, (1, 1), (0, 0), (0, 0), 1, (2, 2),
    );
    assert!(err.is_err(), "dilation!=(1,1) should fail");
    let msg = format!("{}", err.unwrap_err());
    assert!(
        msg.contains("dilation"),
        "expected dilation error, got {msg}"
    );
    Ok(())
}
