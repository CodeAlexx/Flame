use crate::{Tensor, DType, Error, Result, Shape};

/// Return the 4D shape tuple if the tensor has exactly four dimensions.
pub fn shape4(t: &Tensor) -> Result<(usize, usize, usize, usize)> {
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
pub fn transpose_last2(t: &Tensor) -> Result<Tensor> {
    let ndim = t.shape().dims().len();
    if ndim < 2 {
        return Err(Error::InvalidInput("transpose_last2 requires >=2 dims".into()));
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
pub fn matmul_tt(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let bt = transpose_last2(b)?;
    a.bmm(&bt)
}

/// Allocate a zero tensor with the same shape/dtype/device as the input tensor.
pub fn zeros_like(t: &Tensor) -> Result<Tensor> {
    let zeros = t.zeros_like()?;
    if zeros.dtype() == t.dtype() {
        Ok(zeros)
    } else {
        zeros.to_dtype(t.dtype())
    }
}

/// Allocate a tensor filled with `value` matching the input tensor layout.
pub fn full_like(t: &Tensor, value: f32) -> Result<Tensor> {
    let filled = t.full_like(value)?;
    if filled.dtype() == t.dtype() {
        Ok(filled)
    } else {
        filled.to_dtype(t.dtype())
    }
}

/// Elementwise selection using a mask (mask != 0 selects `a`).
pub fn where_mask(mask: &Tensor, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.shape() != b.shape() {
        return Err(Error::InvalidInput(format!(
            "where_mask expects matching tensors, got {:?} and {:?}",
            a.shape(),
            b.shape()
        )));
    }
    let target_shape = a.shape().clone();
    let mask_broadcast = if mask.shape() != &target_shape {
        mask.broadcast_to(&target_shape)?
    } else {
        mask.clone_result()?
    };
    let mask_f32 = mask_broadcast.to_dtype(DType::F32)?;
    mask_f32.where_tensor(&a.clone_result()?, &b.clone_result()?)
}

/// Compute the mean over all elements in FP32.
pub fn mean_all_f32(t: &Tensor) -> Result<f32> {
    let count = t.shape().elem_count();
    if count == 0 {
        return Err(Error::InvalidInput("mean_all_f32 on empty tensor".into()));
    }
    let data = t.to_dtype(DType::F32)?.to_vec()?;
    Ok(data.iter().sum::<f32>() / count as f32)
}
