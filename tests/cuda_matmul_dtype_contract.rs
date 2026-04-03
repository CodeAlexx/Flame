#![cfg(all(feature = "cuda", feature = "heavy_kernels", feature = "bf16_u16"))]

mod testutil;

use flame_core::{DType, Shape, Tensor};
use std::sync::Arc;

use cudarc::driver::CudaDevice;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect(
        "CUDA GPU required. Set CUDA_HOME=/usr/local/cuda and export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH.",
    )
}

fn mk_on_cuda(dev: &Arc<CudaDevice>, rows: i64, cols: i64, dtype: DType) -> Tensor {
    let shape = Shape::from_dims(&[rows as usize, cols as usize]);
    Tensor::zeros_dtype(shape, dtype, dev.clone()).expect("allocate tensor")
}

#[test]
fn matmul_f32_ok() {
    let dev = cuda_device();
    let a = mk_on_cuda(&dev, 8, 16, DType::F32);
    let b = mk_on_cuda(&dev, 16, 4, DType::F32);
    let y = a.matmul(&b).expect("F32 matmul");
    assert_eq!(y.dtype(), DType::F32);
    assert_eq!(
        y.storage_dtype(),
        DType::F32,
        "F32 matmul must allocate F32 storage"
    );
}

#[test]
fn matmul_bf16_ok() {
    let dev = cuda_device();
    let a = mk_on_cuda(&dev, 8, 16, DType::BF16);
    let b = mk_on_cuda(&dev, 16, 4, DType::BF16);
    let y = a.matmul(&b).expect("BF16 matmul");
    assert_eq!(
        y.dtype(),
        DType::BF16,
        "BF16 matmul must preserve storage dtype"
    );
    assert_eq!(
        y.storage_dtype(),
        DType::BF16,
        "BF16 matmul must preserve storage dtype"
    );
}

#[test]
fn matmul_mixed_rejected() {
    let dev = cuda_device();
    let a = mk_on_cuda(&dev, 8, 16, DType::BF16);
    let b = mk_on_cuda(&dev, 16, 4, DType::F32);
    let err = a.matmul(&b).expect_err("mixed matmul should error");
    testutil::assert_mixed_dtype_err(&err);
}
