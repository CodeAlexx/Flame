#![cfg(all(feature = "cuda", feature = "heavy_kernels", feature = "bf16_u16"))]

use flame_core::{
    cuda_ops_bf16, default_dtype, device::Device, ops::cast::cast_bf16_to_f32, ops::conv2d, rng,
    set_default_dtype, DType, Error, Result, Shape, Tensor,
};

fn cuda_device() -> Device {
    Device::cuda(0).expect(
        "CUDA GPU required. Set CUDA_HOME=/usr/local/cuda and export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH.",
    )
}

fn assert_close(a: &[f32], b: &[f32], rtol: f32, atol: f32) {
    assert_eq!(a.len(), b.len());
    for (idx, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * y.abs();
        assert!(
            diff <= tol,
            "mismatch at index {idx}: {x} vs {y} (diff {diff} > tol {tol})"
        );
    }
}

fn gemm_case(m: usize, n: usize, k: usize) -> Result<()> {
    let device = cuda_device().cuda_device().clone();

    let _ = rng::set_seed(1234);
    let a = Tensor::randn(Shape::from_dims(&[m, k]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;
    let b = Tensor::randn(Shape::from_dims(&[k, n]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;

    let y_bf16 = a.matmul(&b)?;
    assert_eq!(y_bf16.dtype(), DType::BF16);

    let a32 = cast_bf16_to_f32(&a)?;
    let b32 = cast_bf16_to_f32(&b)?;
    let y_ref = a32.matmul(&b32)?;

    let y_bf16_f32 = cast_bf16_to_f32(&y_bf16)?;
    let ref_host = y_ref.to_vec_f32()?;
    let test_host = y_bf16_f32.to_vec_f32()?;
    assert_close(&test_host, &ref_host, 1.5e-2, 1.5e-2);

    // Mixed dtype guard
    let err = a.matmul(&b32).expect_err("mixed dtype gemm should error");
    match err {
        Error::InvalidInput(msg) => assert!(msg.contains("dtype mismatch")),
        other => panic!("unexpected error: {other:?}"),
    }

    Ok(())
}

fn conv_case(stride: (usize, usize), padding: (usize, usize)) -> Result<()> {
    let device = cuda_device().cuda_device().clone();

    let _ = rng::set_seed(1234);
    let input = Tensor::randn(Shape::from_dims(&[1, 3, 32, 32]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;
    let weight = Tensor::randn(Shape::from_dims(&[4, 3, 3, 3]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;

    let y_bf16 = conv2d::conv2d_forward(&input, &weight, None, stride, padding, 1)?;
    assert_eq!(y_bf16.dtype(), DType::BF16);

    let input_f32 = cast_bf16_to_f32(&input)?;
    let weight_f32 = cast_bf16_to_f32(&weight)?;
    let y_ref = conv2d::conv2d_forward(&input_f32, &weight_f32, None, stride, padding, 1)?;

    let y_bf16_f32 = cast_bf16_to_f32(&y_bf16)?;
    let ref_host = y_ref.to_vec_f32()?;
    let test_host = y_bf16_f32.to_vec_f32()?;
    assert_close(&test_host, &ref_host, 1.5e-2, 1.5e-2);

    // Mixed dtype guard
    let err = conv2d::conv2d_forward(&input, &weight_f32, None, stride, padding, 1)
        .expect_err("mixed dtype conv2d should error");
    match err {
        Error::InvalidInput(msg) => assert!(msg.contains("dtype mismatch")),
        other => panic!("unexpected error: {other:?}"),
    }

    Ok(())
}

fn gemm_bias_case(m: usize, n: usize, k: usize) -> Result<()> {
    let device = cuda_device().cuda_device().clone();

    let _ = rng::set_seed(4321);
    let a = Tensor::randn(Shape::from_dims(&[m, k]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;
    let b = Tensor::randn(Shape::from_dims(&[k, n]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;
    let bias =
        Tensor::randn(Shape::from_dims(&[n]), 0.0, 1.0, device.clone())?.to_dtype(DType::BF16)?;

    let y_bf16 = cuda_ops_bf16::gemm_bf16(&a, &b, Some(&bias))?;
    assert_eq!(y_bf16.dtype(), DType::BF16);
    assert_eq!(y_bf16.storage_dtype(), DType::BF16);

    let a32 = cast_bf16_to_f32(&a)?;
    let b32 = cast_bf16_to_f32(&b)?;
    let bias32 = cast_bf16_to_f32(&bias)?;
    let y_ref = a32.matmul(&b32)?.add(&bias32.reshape(&[1, n])?)?;

    let y_bf16_f32 = cast_bf16_to_f32(&y_bf16)?;
    let ref_host = y_ref.to_vec_f32()?;
    let test_host = y_bf16_f32.to_vec_f32()?;
    assert_close(&test_host, &ref_host, 1.5e-2, 1.5e-2);

    Ok(())
}

#[test]
fn gemm_bf16_matches_fp32_oracle() -> Result<()> {
    let prev_dtype = default_dtype();
    set_default_dtype(DType::BF16);
    for &(m, n, k) in &[
        (64, 64, 64),
        (128, 64, 96),
        (256, 256, 128),
        (64, 512, 48),
        (320, 96, 192),
    ] {
        gemm_case(m, n, k)?;
    }
    set_default_dtype(prev_dtype);
    Ok(())
}

#[test]
fn gemm_bf16_bias_matches_fp32_oracle() -> Result<()> {
    let prev_dtype = default_dtype();
    set_default_dtype(DType::BF16);
    for &(m, n, k) in &[
        (32, 48, 40),
        (96, 128, 80),
        (192, 64, 256),
        (48, 96, 320),
        (128, 32, 512),
    ] {
        gemm_bias_case(m, n, k)?;
    }
    set_default_dtype(prev_dtype);
    Ok(())
}

#[test]
fn conv_bf16_matches_fp32_oracle() -> Result<()> {
    let prev_dtype = default_dtype();
    set_default_dtype(DType::BF16);
    for &(stride, pad) in &[((1, 1), (1, 1)), ((2, 2), (1, 1)), ((1, 1), (0, 0))] {
        conv_case(stride, pad)?;
    }
    set_default_dtype(prev_dtype);
    Ok(())
}
