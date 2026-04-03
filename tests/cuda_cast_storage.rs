#![cfg(feature = "cuda")]

use cudarc::driver::CudaDevice;
use flame_core::{tensor_ext::to_owning_fp32_strong, DType, Shape, Tensor};

#[test]
fn cast_bf16_to_f32_produces_fp32_storage() {
    let dev = CudaDevice::new(0).expect(
        "CUDA GPU required. Set CUDA_HOME=/usr/local/cuda and export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH.",
    );

    let shape = Shape::from_dims(&[4, 8]);
    let bf16 = Tensor::zeros_dtype(shape.clone(), DType::BF16, dev.clone()).unwrap();
    let f32 = bf16.to_dtype(DType::F32).unwrap();
    assert_eq!(f32.dtype(), DType::F32);
    assert_eq!(
        f32.storage_dtype(),
        DType::F32,
        "to_dtype must allocate F32 storage"
    );

    let strong = to_owning_fp32_strong(&bf16).unwrap();
    assert_eq!(strong.dtype(), DType::F32);
    assert_eq!(strong.storage_dtype(), DType::F32);
    assert_eq!(strong.shape(), &shape);
}
