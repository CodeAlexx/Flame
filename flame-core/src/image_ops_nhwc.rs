use crate::{Tensor, Result, FlameError};
use crate::dtype::DType;
use crate::cuda_ops::GpuOps;

/// Resize an NHWC tensor with bilinear interpolation
pub fn resize_bilinear_nhwc(x: &Tensor, out_h: usize, out_w: usize, align_corners: bool) -> Result<Tensor> {
    // Rank check (NHWC)
    let dims = x.shape().dims();
    if dims.len() != 4 {
        return Err(FlameError::InvalidOperation(
            "resize_bilinear_nhwc expects 4D NHWC".into(),
        ));
    }
    // Output dims must be > 0
    if out_h == 0 || out_w == 0 {
        return Err(FlameError::InvalidOperation(
            "resize_bilinear_nhwc: output dims must be > 0".into(),
        ));
    }
    // DType guard
    let dt = x.dtype();
    if dt != DType::F32 && dt != DType::BF16 {
        return Err(FlameError::InvalidOperation(
            format!("resize_bilinear_nhwc: unsupported dtype {:?} (expect F32/BF16)", dt),
        ));
    }
    GpuOps::resize_bilinear_nhwc(x, out_h, out_w, align_corners)
}

/// Center crop an NHWC tensor to the target size
pub fn center_crop_nhwc(x: &Tensor, tgt_h: usize, tgt_w: usize) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() != 4 {
        return Err(FlameError::InvalidOperation(
            "center_crop_nhwc expects 4D NHWC".into(),
        ));
    }
    if tgt_h == 0 || tgt_w == 0 {
        return Err(FlameError::InvalidOperation(
            "center_crop_nhwc: target dims must be > 0".into(),
        ));
    }
    GpuOps::center_crop_nhwc(x, tgt_h, tgt_w)
}

/// Normalize an NHWC tensor with per-channel mean and std (y = (x - mean[c]) * 1/std[c])
pub fn normalize_nhwc(x: &Tensor, mean: &[f32], std: &[f32]) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() != 4 {
        return Err(FlameError::InvalidOperation(
            "normalize_nhwc expects 4D NHWC".into(),
        ));
    }
    let c = dims[3];
    if mean.len() != c || std.len() != c {
        return Err(FlameError::InvalidOperation(
            format!("normalize_nhwc: mean/std must each have C={} elements", c),
        ));
    }
    let dt = x.dtype();
    if dt != DType::F32 && dt != DType::BF16 {
        return Err(FlameError::InvalidOperation(
            format!("normalize_nhwc: unsupported dtype {:?} (expect F32/BF16)", dt),
        ));
    }
    GpuOps::normalize_nhwc(x, mean, std)
}
