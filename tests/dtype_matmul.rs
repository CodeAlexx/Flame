#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

mod testutil;

use cudarc::driver::CudaDevice;
use flame_core::{linear::Linear, DType, Shape, Tensor};
use std::sync::Arc;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect(
        "CUDA GPU required. Set CUDA_HOME=/usr/local/cuda and export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH.",
    )
}

fn bf16_tensor(dev: &Arc<CudaDevice>, dims: &[usize]) -> Tensor {
    Tensor::randn(Shape::from_dims(dims), 0.0, 1.0, dev.clone())
        .expect("randn")
        .to_dtype(DType::BF16)
        .expect("cast to bf16")
}

#[test]
fn bf16_linear_io_preserved() {
    let dev = cuda_device();
    let layer = Linear::new(16, 8, true, &dev).expect("make linear");

    assert_eq!(layer.weight.dtype(), DType::BF16);
    assert_eq!(layer.weight.storage_dtype(), DType::BF16);
    if let Some(bias) = &layer.bias {
        assert_eq!(bias.dtype(), DType::BF16);
        assert_eq!(bias.storage_dtype(), DType::BF16);
    }

    let input = bf16_tensor(&dev, &[4, 16]).requires_grad_(false);
    let output = layer.forward(&input).expect("forward bf16");

    assert_eq!(output.dtype(), DType::BF16);
    assert_eq!(output.storage_dtype(), DType::BF16);
    assert_eq!(output.shape().dims(), &[4, 8]);
}

#[test]
fn bf16_linear_bias_act_bf16_out() {
    let dev = cuda_device();
    let layer = Linear::new(32, 12, true, &dev).expect("make linear");

    let input = bf16_tensor(&dev, &[2, 3, 1, 32]);
    let output = layer.forward(&input).expect("forward bf16 nhwc");

    assert_eq!(output.dtype(), DType::BF16);
    assert_eq!(output.storage_dtype(), DType::BF16);
    assert_eq!(output.shape().dims(), &[2, 3, 1, 12]);
}
