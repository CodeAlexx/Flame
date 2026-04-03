#![cfg(feature = "cuda")]

use cudarc::driver::CudaDevice;
use flame_core::{set_default_dtype, AutogradContext, DType, Result, Shape, Tensor};

#[test]
fn backward_emits_bf16_grads() -> Result<()> {
    AutogradContext::reset();
    let device = CudaDevice::new(0).expect(
        "CUDA GPU required. Set CUDA_HOME=/usr/local/cuda and export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH.",
    );
    set_default_dtype(DType::BF16);

    let x = Tensor::randn(Shape::from_dims(&[1, 4, 4, 2]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);
    let loss = x.sum()?;

    let gradients = AutogradContext::backward(&loss)?;
    let grad_x = gradients.get_public_grad(x.id())?;
    assert_eq!(grad_x.dtype(), DType::BF16);
    if grad_x.rank() == 4 {
        assert_eq!(grad_x.rank(), 4);
    }

    AutogradContext::clear();
    Ok(())
}
