#![cfg(feature = "cuda")]

use cudarc::driver::CudaDevice;
use flame_core::device::Device;
use flame_core::layer_norm::LayerNorm;
use flame_core::linear::Linear;
use flame_core::tensor::contracts::assert_nhwc_bf16_public;
use flame_core::{set_default_dtype, DType, Result, Shape, Tensor};

#[test]
fn conv2d_layernorm_linear_boundary() -> Result<()> {
    let device = CudaDevice::new(0)?;
    set_default_dtype(DType::BF16);

    // Conv2d public entry
    // Current conv2d implementation still expects NCHW layout internally while the
    // contract guard only enforces rank-4 BF16 storage, so we feed a tensor whose
    // dimensions align with those assumptions.
    let x =
        Tensor::zeros(Shape::from_dims(&[1, 4, 8, 8]), device.clone())?.to_dtype(DType::BF16)?;
    let weight =
        Tensor::zeros(Shape::from_dims(&[3, 4, 3, 3]), device.clone())?.to_dtype(DType::BF16)?;
    let y = x.conv2d(&weight, None, 1, 1)?;
    assert_eq!(y.dtype(), DType::BF16);
    assert_eq!(y.shape().rank(), 4);

    // LayerNorm (2D path)
    let layer_input =
        Tensor::zeros(Shape::from_dims(&[2, 4]), device.clone())?.to_dtype(DType::BF16)?;
    let mut ln = LayerNorm::new(vec![4], 1e-5, device.clone())?;
    if let Some(w) = ln.weight.take() {
        ln.weight = Some(w.to_dtype(DType::BF16)?.requires_grad_(true));
    }
    if let Some(b) = ln.bias.take() {
        ln.bias = Some(b.to_dtype(DType::BF16)?.requires_grad_(true));
    }
    let ln_out = ln.forward(&layer_input)?;
    assert_eq!(ln_out.dtype(), DType::BF16);

    // Linear (2D input, BF16 enforced)
    let linear = Linear::new(4, 2, true, &device)?;
    let lin_input =
        Tensor::zeros(Shape::from_dims(&[3, 4]), device.clone())?.to_dtype(DType::BF16)?;
    let lin_out = linear.forward(&lin_input)?;
    assert_eq!(lin_out.dtype(), DType::BF16);

    Ok(())
}

#[test]
fn nhwc_bf16_guard_accepts_latents() {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => {
            eprintln!("skipping nhwc_bf16_guard_accepts_latents: cuda device unavailable");
            return;
        }
    };

    let shape = Shape::from_dims(&[1, 4, 4, 16]);
    let tensor =
        Tensor::zeros_dtype(shape, DType::BF16, device.cuda_device_arc()).expect("create tensor");
    assert!(tensor.is_nhwc());
    assert_nhwc_bf16_public("test::latents", &tensor).expect("guard passes");
}
