#![allow(clippy::too_many_arguments)]

use crate::Shape;
use crate::{DType, Result, Tensor};

pub fn layernorm_affine_bf16_inplace(
    x: &mut Tensor,
    gamma: Option<&Tensor>,
    beta: Option<&Tensor>,
    _b: i32,
    _h: i32,
    _w: i32,
    c: i32,
    eps: f32,
) -> Result<()> {
    ensure_bf16_tensor(x, "layernorm_affine_bf16_inplace.x")?;

    let normalized_shape = [c as usize];
    let out = crate::layer_norm::layer_norm(x, &normalized_shape, gamma, beta, eps)?;
    *x = out;
    Ok(())
}

pub fn adaln_modulate_bf16_inplace(
    x: &mut Tensor,
    gamma: Option<&Tensor>,
    beta: Option<&Tensor>,
    mod_scale: Option<&Tensor>,
    mod_shift: Option<&Tensor>,
    b: i32,
    h: i32,
    w: i32,
    c: i32,
    eps: f32,
) -> Result<()> {
    ensure_bf16_tensor(x, "adaln_modulate_bf16_inplace.x")?;

    let normalized_shape = [c as usize];
    let mut norm = crate::layer_norm::layer_norm(x, &normalized_shape, gamma, beta, eps)?;

    let target_shape = Shape::from_dims(&[b as usize, h as usize, w as usize, c as usize]);

    if let Some(scale) = mod_scale {
        ensure_bf16_tensor(scale, "adaln_modulate_bf16_inplace.mod_scale")?;
        let scale_bc = scale
            .reshape(&[b as usize, 1, 1, c as usize])?
            .broadcast_to(&target_shape)?;
        norm = norm.mul(&scale_bc)?;
    }

    if let Some(shift) = mod_shift {
        ensure_bf16_tensor(shift, "adaln_modulate_bf16_inplace.mod_shift")?;
        let shift_bc = shift
            .reshape(&[b as usize, 1, 1, c as usize])?
            .broadcast_to(&target_shape)?;
        norm = norm.add(&shift_bc)?;
    }

    *x = norm;
    Ok(())
}

fn ensure_bf16_tensor(t: &Tensor, tag: &str) -> Result<()> {
    if t.dtype() != DType::BF16 || t.storage_dtype() != DType::BF16 {
        Err(crate::Error::InvalidOperation(format!(
            "{tag} expects BF16 storage"
        )))
    } else {
        Ok(())
    }
}
