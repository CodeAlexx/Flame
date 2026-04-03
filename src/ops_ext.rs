use crate::{Error, Tensor};
use std::result::Result as StdResult;

type OpResult<T> = StdResult<T, Error>;

/// Return the 4D shape tuple if the tensor has exactly four dimensions.
pub fn shape4(t: &Tensor) -> OpResult<(usize, usize, usize, usize)> {
    let dims = t.shape().dims();
    if dims.len() != 4 {
        return Err(Error::InvalidInput(format!(
            "expected 4D tensor, got shape {:?}",
            dims
        )));
    }
    Ok((dims[0], dims[1], dims[2], dims[3]))
}

/// Transpose the last two dimensions of a tensor.
pub fn transpose_last2(t: &Tensor) -> OpResult<Tensor> {
    let ndim = t.shape().dims().len();
    if ndim < 2 {
        return Err(Error::InvalidInput(
            "transpose_last2 requires >=2 dims".into(),
        ));
    }
    if ndim == 2 {
        return t.transpose();
    }
    let mut perm: Vec<usize> = (0..ndim).collect();
    perm[ndim - 2] = ndim - 1;
    perm[ndim - 1] = ndim - 2;
    t.permute(&perm)
}

/// Batched matrix multiplication with the second operand transposed on its last two dims.
pub fn matmul_tt(a: &Tensor, b: &Tensor) -> OpResult<Tensor> {
    let bt = transpose_last2(b)?;
    a.bmm(&bt)
}

/// Allocate a zero tensor with the same shape/dtype/device as the input tensor.
pub fn zeros_like(t: &Tensor) -> OpResult<Tensor> {
    let zeros = t.zeros_like()?;
    if zeros.dtype() == t.dtype() {
        Ok(zeros)
    } else {
        zeros.to_dtype(t.dtype())
    }
}

/// Allocate a tensor filled with `value` matching the input tensor layout.
pub fn full_like(t: &Tensor, value: f32) -> OpResult<Tensor> {
    let filled = t.full_like(value)?;
    if filled.dtype() == t.dtype() {
        Ok(filled)
    } else {
        filled.to_dtype(t.dtype())
    }
}

/// Elementwise selection using a mask (mask != 0 selects `a`).
pub fn where_mask(mask: &Tensor, a: &Tensor, b: &Tensor) -> OpResult<Tensor> {
    if a.shape() != b.shape() {
        return Err(Error::InvalidInput(format!(
            "where_mask expects matching tensors, got {:?} and {:?}",
            a.shape(),
            b.shape()
        )));
    }

    let target_shape = a.shape().clone();
    let mask_cast = if mask.shape() != &target_shape {
        mask.broadcast_to(&target_shape)?
    } else {
        mask.clone_result()?
    };

    let dtype = a.dtype();
    let mask_typed = if mask_cast.dtype() == dtype {
        mask_cast
    } else {
        mask_cast.to_dtype(dtype)?
    };

    let ones = full_like(&mask_typed, 1.0)?;
    let inv_mask = ones.sub(&mask_typed)?;
    let a_term = mask_typed.mul(&a.clone_result()?)?;
    let b_term = inv_mask.mul(&b.clone_result()?)?;
    a_term.add(&b_term)
}

/// Compute the mean over all elements in FP32.
pub fn mean_all_f32(t: &Tensor) -> OpResult<f32> {
    let count = t.shape().elem_count();
    if count == 0 {
        return Err(Error::InvalidInput("mean_all_f32 on empty tensor".into()));
    }
    let sum = t.sum_all()?;
    let mean = sum.div_scalar(count as f32)?;
    mean.to_scalar::<f32>()
}
