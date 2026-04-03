#![cfg(feature = "autograd_v4")]

use crate::{
    autograd_v4::{sdpa_forward, SdpaConfig, SdpaSave},
    config, rng, sdpa, DType, Result, Shape, Tensor,
};

fn test_device() -> Option<std::sync::Arc<cudarc::driver::CudaDevice>> {
    crate::device::Device::cuda(0)
        .ok()
        .map(|d| d.cuda_device_arc())
}

#[test]
fn sdpa_v4_forward_runs() -> Result<()> {
    let Some(device) = test_device() else {
        return Ok(());
    };

    config::set_default_dtype(DType::BF16);

    let shape = Shape::from_dims(&[1, 2, 4, 8]);
    let q = Tensor::randn(shape.clone(), 0.0, 0.01, device.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);
    let k = Tensor::randn(shape.clone(), 0.0, 0.01, device.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);
    let v = Tensor::randn(shape, 0.0, 0.01, device.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);

    let out = sdpa::forward_v4(&q, &k, &v, None, false, None, None)?;
    assert_eq!(out.shape().dims(), &[1, 2, 4, 8]);
    Ok(())
}

#[test]
fn sdpa_v4_supports_causal_masks() -> Result<()> {
    let Some(device) = test_device() else {
        return Ok(());
    };

    config::set_default_dtype(DType::BF16);

    let shape = Shape::from_dims(&[1, 2, 4, 8]);
    let q = Tensor::randn(shape.clone(), 0.0, 0.01, device.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);
    let k = Tensor::randn(shape.clone(), 0.0, 0.01, device.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);
    let v = Tensor::randn(shape, 0.0, 0.01, device.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);

    let out = sdpa::forward_v4(&q, &k, &v, None, true, None, None)?;
    let stats = out.to_dtype(DType::F32)?.to_vec_f32()?;
    assert!(stats.iter().all(|v| v.is_finite()));
    Ok(())
}

#[test]
fn sdpa_v4_chunked_matches_full() -> Result<()> {
    let Some(device) = test_device() else {
        return Ok(());
    };

    config::set_default_dtype(DType::BF16);

    let shape = Shape::from_dims(&[1, 2, 6, 8]);
    let q = Tensor::randn(shape.clone(), 0.0, 0.01, device.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);
    let k = Tensor::randn(shape.clone(), 0.0, 0.01, device.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);
    let v = Tensor::randn(shape, 0.0, 0.01, device.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);

    let full = sdpa::forward_v4(&q, &k, &v, None, false, None, None)?;
    let chunked = sdpa::forward_v4(&q, &k, &v, None, false, None, Some((3, 4)))?;

    let diff = chunked.sub(&full)?.to_dtype(DType::F32)?.to_vec_f32()?;
    let max_diff = diff.into_iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    assert!(
        max_diff < 1e-3,
        "chunked SDPA output differs from full path"
    );

    Ok(())
}

#[test]
fn sdpa_v4_chunked_matches_full_with_mask() -> Result<()> {
    let Some(device) = test_device() else {
        return Ok(());
    };

    config::set_default_dtype(DType::BF16);

    let shape = Shape::from_dims(&[1, 2, 6, 8]);
    let q = Tensor::randn(shape.clone(), 0.0, 0.01, device.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);
    let k = Tensor::randn(shape.clone(), 0.0, 0.01, device.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);
    let v = Tensor::randn(shape.clone(), 0.0, 0.01, device.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);

    let mask_shape = Shape::from_dims(&[1, 2, shape.dims()[2], shape.dims()[2]]);
    let mask_rand = Tensor::randn(mask_shape.clone(), 0.0, 1.0, device.clone())?;
    let threshold = mask_rand.full_like(-0.25)?;
    let mask = mask_rand.gt(&threshold)?;

    let full = sdpa::forward_v4(&q, &k, &v, Some(&mask), false, None, None)?;
    let chunked = sdpa::forward_v4(&q, &k, &v, Some(&mask), false, None, Some((3, 4)))?;

    let diff = chunked.sub(&full)?.to_dtype(DType::F32)?.to_vec_f32()?;
    let max_diff = diff.into_iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    assert!(
        max_diff < 1e-3,
        "chunked SDPA (with mask) output differs from full path"
    );

    Ok(())
}

#[test]
fn sdpa_v4_chunked_matches_full_with_dropout() -> Result<()> {
    let Some(device) = test_device() else {
        return Ok(());
    };

    config::set_default_dtype(DType::BF16);
    rng::set_seed(1234)?;

    let shape = Shape::from_dims(&[1, 2, 6, 8]);
    let q = Tensor::randn(shape.clone(), 0.0, 0.01, device.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);
    let k = Tensor::randn(shape.clone(), 0.0, 0.01, device.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);
    let v = Tensor::randn(shape.clone(), 0.0, 0.01, device.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);

    let mut cfg_full = SdpaConfig::default();
    cfg_full.save = SdpaSave::SaveLSE;
    cfg_full.dropout_p = 0.1;
    cfg_full.dropout_seed = Some(98765);

    let (full, _) = sdpa_forward(&q, &k, &v, None, &cfg_full)?;

    let mut cfg_chunk = cfg_full;
    cfg_chunk.chunk_q = Some(3);
    cfg_chunk.chunk_kv = Some(4);

    let (chunked, _) = sdpa_forward(&q, &k, &v, None, &cfg_chunk)?;

    let diff = chunked.sub(&full)?.to_dtype(DType::F32)?.to_vec_f32()?;
    let max_diff = diff.into_iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    assert!(
        max_diff < 1e-3,
        "chunked SDPA (with dropout) output differs from full path"
    );

    Ok(())
}
