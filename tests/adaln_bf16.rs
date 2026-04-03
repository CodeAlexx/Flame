#![cfg(all(feature = "cuda", feature = "bf16_u16", feature = "bf16_kernels"))]

use anyhow::Result;
use cudarc::driver::CudaDevice;
use flame_core::kernels::adaln::adaln_modulate_bf16_inplace;
use flame_core::{DType, Shape, Tensor};
use std::sync::Arc;

const EPS: f32 = 1e-5;

#[test]
fn adaln_modulate_matches_fp32_reference() -> Result<()> {
    let device = match CudaDevice::new(0) {
        Ok(dev) => Arc::new(dev),
        Err(err) => {
            eprintln!("[skip] CUDA unavailable: {err}");
            return Ok(());
        }
    };

    let (b, h, w, c) = (2, 2, 2, 4);
    let target_shape = Shape::from_dims(&[b, h, w, c]);

    let x = Tensor::randn(target_shape.clone(), 0.0, 1.0, Arc::clone(&device))?
        .to_dtype(DType::BF16)?;
    let gamma = Tensor::randn(Shape::from_dims(&[c]), 0.0, 0.5, Arc::clone(&device))?
        .to_dtype(DType::BF16)?;
    let beta = Tensor::randn(Shape::from_dims(&[c]), 0.0, 0.5, Arc::clone(&device))?
        .to_dtype(DType::BF16)?;
    let mod_scale = Tensor::randn(Shape::from_dims(&[b, c]), 0.0, 0.1, Arc::clone(&device))?
        .to_dtype(DType::BF16)?;
    let mod_shift = Tensor::randn(Shape::from_dims(&[b, c]), 0.0, 0.1, Arc::clone(&device))?
        .to_dtype(DType::BF16)?;

    let mut fused = x.clone_result()?;
    adaln_modulate_bf16_inplace(
        &mut fused,
        Some(&gamma),
        Some(&beta),
        Some(&mod_scale),
        Some(&mod_shift),
        b as i32,
        h as i32,
        w as i32,
        c as i32,
        EPS,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    let x32 = x.to_dtype(DType::F32)?;
    let gamma32 = gamma.to_dtype(DType::F32)?.reshape(&[1, 1, 1, c])?;
    let beta32 = beta.to_dtype(DType::F32)?.reshape(&[1, 1, 1, c])?;
    let mod_scale32 = mod_scale.to_dtype(DType::F32)?.reshape(&[b, 1, 1, c])?;
    let mod_shift32 = mod_shift.to_dtype(DType::F32)?.reshape(&[b, 1, 1, c])?;

    let gamma_bc = gamma32.broadcast_to(&target_shape)?;
    let beta_bc = beta32.broadcast_to(&target_shape)?;
    let mod_scale_bc = mod_scale32.broadcast_to(&target_shape)?;
    let mod_shift_bc = mod_shift32.broadcast_to(&target_shape)?;

    let hidden_f = c as f32;
    let mean = x32.sum_dim_keepdim(3)?.div_scalar(hidden_f)?;
    let centered = x32.sub(&mean)?;
    let var = centered
        .mul(&centered)?
        .sum_dim_keepdim(3)?
        .div_scalar(hidden_f)?;
    let inv_std = var.add_scalar(EPS)?.rsqrt()?;
    let norm = centered.mul(&inv_std)?;

    let scale_term = gamma_bc.mul(&mod_scale_bc.add_scalar(1.0f32)?)?;
    let shift_term = beta_bc.add(&mod_shift_bc)?;
    let y_ref = norm.mul(&scale_term)?.add(&shift_term)?;

    let fused32 = fused.to_dtype(DType::F32)?;
    let fused_host = fused32.to_vec()?;
    let ref_host = y_ref.to_vec()?;

    let max_err = fused_host
        .iter()
        .zip(ref_host.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(max_err < 5e-3, "max abs diff {max_err}");
    Ok(())
}
