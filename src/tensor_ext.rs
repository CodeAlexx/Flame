use crate::{error::Error, shape::Shape, tensor::guard_fp32_alloc, tensor::Tensor, DType, Result};

/// Convert a tensor to a true owning FP32 buffer on the current device.
/// Guarantees logical dtype == storage dtype == FP32.
pub fn to_owning_fp32_strong(x: &Tensor) -> Result<Tensor> {
    let mut out = if x.dtype() == DType::F32 && x.storage_dtype() == DType::F32 {
        x.clone_result()?
    } else {
        let numel = x.shape().elem_count();
        guard_fp32_alloc(numel * std::mem::size_of::<f32>(), "to_owning_fp32_strong");
        if std::env::var("SDXL_DEBUG_SHAPES").ok().as_deref() == Some("1") {
            eprintln!(
                "[to_owning_fp32_strong] cast dtype {:?} storage {:?} shape {:?}",
                x.dtype(),
                x.storage_dtype(),
                x.shape().dims()
            );
        }
        x.to_dtype(DType::F32)?
    };

    if out.storage_dtype() != DType::F32 {
        let numel = out.shape().elem_count();
        guard_fp32_alloc(
            numel * std::mem::size_of::<f32>(),
            "to_owning_fp32_strong-storage",
        );
        if std::env::var("SDXL_DEBUG_SHAPES").ok().as_deref() == Some("1") {
            eprintln!(
                "[to_owning_fp32_strong] fixing storage {:?} -> F32 shape {:?}",
                out.storage_dtype(),
                out.shape().dims()
            );
        }
        out = out.to_dtype(DType::F32)?;
    }

    debug_assert_eq!(out.dtype(), DType::F32, "logical dtype must be FP32");
    debug_assert_eq!(
        out.storage_dtype(),
        DType::F32,
        "storage dtype must be FP32"
    );
    Ok(out)
}

fn ensure_nhwc(tag: &str, x: &Tensor) -> Result<(usize, usize, usize, usize)> {
    let dims = x.shape().dims().to_vec();
    if dims.len() != 4 {
        return Err(Error::InvalidInput(format!(
            "{tag}: expected NHWC rank-4 tensor, got {:?}",
            dims
        )));
    }
    Ok((dims[0], dims[1], dims[2], dims[3]))
}

/// Slice the channel dimension (NHWC) down to `out_c`, returning an owning tensor on the same device.
pub fn slice_channels(x: &Tensor, out_c: usize) -> Result<Tensor> {
    let (_b, _h, _w, c) = ensure_nhwc("slice_channels", x)?;
    if out_c > c {
        return Err(Error::InvalidInput(format!(
            "slice_channels: requested channels {} > input channels {}",
            out_c, c
        )));
    }
    if out_c == c {
        return x.clone_result();
    }
    let view = x.narrow(3, 0, out_c)?;
    view.clone_result()
}

/// Pad the channel dimension (NHWC) with zeros up to `out_c`, returning an owning tensor on the same device.
pub fn pad_channels(x: &Tensor, out_c: usize) -> Result<Tensor> {
    let (b, h, w, c) = ensure_nhwc("pad_channels", x)?;
    if out_c < c {
        return Err(Error::InvalidInput(format!(
            "pad_channels: requested channels {} < input channels {}",
            out_c, c
        )));
    }
    if out_c == c {
        return x.clone_result();
    }
    let pad_c = out_c - c;
    let zeros = Tensor::zeros_dtype(
        Shape::from_dims(&[b, h, w, pad_c]),
        x.dtype(),
        x.device().clone(),
    )?;
    let concatenated = Tensor::cat(&[x, &zeros], 3)?;
    concatenated.clone_result()
}
