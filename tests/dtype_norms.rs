#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use anyhow::Result;
use cudarc::driver::CudaDevice;
use flame_core::{
    cuda_ops::GpuOps,
    device::Device,
    norm::{GroupNorm, LayerNorm, RMSNorm},
    reset_telemetry,
    shape::Shape,
    telemetry_snapshot_full, DType, Tensor,
};

fn cuda_device() -> Result<(Device, std::sync::Arc<CudaDevice>)> {
    let device = Device::cuda(0)?;
    let cuda = device.cuda_device_arc();
    Ok((device, cuda))
}

#[test]
fn layer_norm_bf16_contract() -> Result<()> {
    let (_dev, cuda) = cuda_device()?;
    let ln = LayerNorm::new_with_affine(vec![8], 1e-5, true, cuda.clone())?;
    let input = Tensor::zeros_dtype(Shape::from_dims(&[4, 8]), DType::BF16, cuda.clone())?;

    reset_telemetry();
    let output = ln.forward(&input)?;
    let snap = telemetry_snapshot_full();

    assert_eq!(output.dtype(), DType::BF16);
    assert_eq!(output.storage_dtype(), DType::BF16);
    assert_eq!(snap.dtype_traps, 0);
    Ok(())
}

#[test]
fn group_norm_bf16_contract() -> Result<()> {
    let (_dev, cuda) = cuda_device()?;
    let gn = GroupNorm::new_with_affine(2, 8, 1e-5, true, cuda.clone())?;
    let input = Tensor::zeros_dtype(Shape::from_dims(&[2, 4, 4, 8]), DType::BF16, cuda.clone())?;

    reset_telemetry();
    let output = gn.forward(&input)?;
    let snap = telemetry_snapshot_full();

    assert_eq!(output.dtype(), DType::BF16);
    assert_eq!(output.storage_dtype(), DType::BF16);
    assert_eq!(snap.dtype_traps, 0);
    Ok(())
}

#[test]
fn rms_norm_bf16_contract() -> Result<()> {
    let (_dev, cuda) = cuda_device()?;
    let rms = RMSNorm::new(vec![16], 1e-6, true, cuda.clone())?;
    let input = Tensor::zeros_dtype(Shape::from_dims(&[3, 5, 16]), DType::BF16, cuda.clone())?;

    reset_telemetry();
    let output = rms.forward(&input)?;
    let snap = telemetry_snapshot_full();

    assert_eq!(output.dtype(), DType::BF16);
    assert_eq!(output.storage_dtype(), DType::BF16);
    assert_eq!(snap.dtype_traps, 0);
    Ok(())
}

#[test]
fn permute_bf16_layout_roundtrip() -> Result<()> {
    let (_dev, cuda) = cuda_device()?;
    let input = Tensor::zeros_dtype(Shape::from_dims(&[1, 2, 2, 3]), DType::BF16, cuda.clone())?;

    let nchw = GpuOps::permute_nhwc_to_nchw(&input)?;
    assert_eq!(nchw.dtype(), DType::BF16);
    assert_eq!(nchw.storage_dtype(), DType::BF16);

    let nhwc = GpuOps::permute_nchw_to_nhwc(&nchw)?;
    assert_eq!(nhwc.dtype(), DType::BF16);
    assert_eq!(nhwc.storage_dtype(), DType::BF16);
    assert_eq!(nhwc.shape().dims(), input.shape().dims());
    Ok(())
}
