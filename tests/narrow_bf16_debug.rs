#![cfg(all(feature = "cuda", feature = "heavy_kernels", feature = "bf16_u16"))]

use flame_core::{DType, Result, Shape, Tensor};
use std::sync::Arc;

use cudarc::driver::CudaDevice;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect(
        "CUDA GPU required. Set CUDA_HOME=/usr/local/cuda and export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH.",
    )
}

#[test]
fn reproduce_narrow_bf16_backward_illegal_address() -> Result<()> {
    let dev = cuda_device();

    let shape = Shape::from_dims(&[1, 4, 2]);
    let data: Vec<f32> = (0..shape.elem_count()).map(|v| v as f32).collect();
    let tensor = Tensor::from_vec_dtype(data.clone(), shape.clone(), dev.clone(), DType::BF16)?;

    let _sliced = tensor.narrow_general_cuda(1, 2, 2)?;

    let mut grad_in = Tensor::zeros_dtype(shape.clone(), DType::BF16, dev.clone())?;
    let grad_vals: Vec<f32> = (0..4).map(|v| (v as f32) - 1.25).collect();
    let grad_out = Tensor::from_vec_dtype(
        grad_vals,
        Shape::from_dims(&[1, 2, 2]),
        dev.clone(),
        DType::BF16,
    )?;

    Tensor::narrow_backward_scatter_add_cuda(&grad_out, &mut grad_in, 1, 2, 2)?;
    Ok(())
}
