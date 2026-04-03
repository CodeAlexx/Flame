#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use anyhow::Result;
use flame_core::{sdpa, DType, Device, Shape, Tensor};

fn randn(shape: &[usize], device: &Device) -> Result<Tensor> {
    let arc = device.cuda_device().clone();
    Ok(Tensor::randn(Shape::from_dims(shape), 0.0, 1.0, arc)?.to_dtype(DType::F32)?)
}

#[test]
fn sdpa_bf16_matches_f32() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(err) => {
            eprintln!("[skip] CUDA unavailable: {err}");
            return Ok(());
        }
    };

    let (b, h, q_len, k_len, d) = (2usize, 4usize, 16usize, 20usize, 64usize);

    let q_f32 = randn(&[b, h, q_len, d], &device)?;
    let k_f32 = randn(&[b, h, k_len, d], &device)?;
    let v_f32 = randn(&[b, h, k_len, d], &device)?;

    let q_bf16 = q_f32.to_dtype(DType::BF16)?;
    let k_bf16 = k_f32.to_dtype(DType::BF16)?;
    let v_bf16 = v_f32.to_dtype(DType::BF16)?;

    let bf16_out = sdpa::forward(&q_bf16, &k_bf16, &v_bf16, None)?;
    let f32_out = sdpa::forward(&q_f32, &k_f32, &v_f32, None)?;

    let bf16_vec = bf16_out.to_dtype(DType::F32)?.to_vec_f32()?;
    let f32_vec = f32_out.to_vec_f32()?;

    assert_eq!(bf16_vec.len(), f32_vec.len());
    let mut max_diff = 0f32;
    for (a, b) in bf16_vec.iter().zip(f32_vec.iter()) {
        max_diff = max_diff.max((a - b).abs());
    }

    assert!(
        max_diff < 5e-2,
        "max difference too high: {max_diff}. BF16 path diverges from F32"
    );

    Ok(())
}
