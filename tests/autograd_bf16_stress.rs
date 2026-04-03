#![cfg(feature = "cuda")]

use cudarc::driver::CudaDevice;
use flame_core::{set_default_dtype, AutogradContext, DType, Result, Shape, Tensor};
use std::sync::Arc;

fn assert_finite(t: &Tensor) {
    let host = t.to_dtype(DType::F32).unwrap();
    let values = host.to_vec_f32().unwrap();
    assert!(
        values.iter().all(|v| v.is_finite()),
        "tensor contained non-finite values"
    );
}

#[test]
fn gradient_map_fp32_public_bf16_consistency() -> Result<()> {
    AutogradContext::reset();
    set_default_dtype(DType::BF16);
    let device: Arc<CudaDevice> = CudaDevice::new(0).expect(
        "CUDA GPU required. Set CUDA_HOME=/usr/local/cuda and export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH.",
    );

    let x = Tensor::randn(Shape::from_dims(&[2, 4]), 0.0, 1.0, Arc::clone(&device))?
        .to_dtype(DType::BF16)?
        .requires_grad_(true);
    let loss = x.mul(&x)?.mean()?; // scalar BF16 loss

    let gradients = AutogradContext::backward(&loss)?;
    let public = gradients.take_public_grads()?;
    assert!(!public.is_empty());
    for grad in public.values() {
        assert_eq!(grad.dtype(), DType::BF16);
        assert_finite(grad);
    }
    Ok(())
}
