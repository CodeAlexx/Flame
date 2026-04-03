#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use anyhow::Result;
use cudarc::driver::CudaDevice;
use flame_core::{
    device::Device, reset_telemetry, sdpa, shape::Shape, telemetry_snapshot_full, DType, Tensor,
};

fn cuda_device() -> Result<(Device, std::sync::Arc<CudaDevice>)> {
    let device = Device::cuda(0)?;
    let cuda = device.cuda_device_arc();
    Ok((device, cuda))
}

#[test]
fn sdpa_bf16_contract() -> Result<()> {
    let (_dev, cuda) = cuda_device()?;
    let shape = Shape::from_dims(&[1, 2, 4, 8]);
    let q = Tensor::zeros_dtype(shape.clone(), DType::BF16, cuda.clone())?;
    let k = Tensor::zeros_dtype(shape.clone(), DType::BF16, cuda.clone())?;
    let v = Tensor::zeros_dtype(shape, DType::BF16, cuda.clone())?;

    reset_telemetry();
    let output = sdpa::forward(&q, &k, &v, None)?;
    let snap = telemetry_snapshot_full();

    assert_eq!(output.dtype(), DType::BF16);
    assert_eq!(output.storage_dtype(), DType::BF16);
    assert_eq!(snap.dtype_traps, 0);
    Ok(())
}
