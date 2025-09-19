use crate::{device::Device, Tensor, DType, Error, Shape};
use std::result::Result as StdResult;
use std::sync::Arc;

type DevResult<T> = StdResult<T, Error>;

fn cuda_arc(dev: &Device) -> DevResult<Arc<cudarc::driver::CudaDevice>> {
    if dev.is_cuda() {
        Ok(dev.cuda_device().clone())
    } else {
        Err(Error::Unsupported("only CUDA devices are supported in FLAME".into()))
    }
}

pub fn zeros_on(shape: &[usize], dtype: DType, dev: &Device) -> DevResult<Tensor> {
    let arc = cuda_arc(dev)?;
    Tensor::zeros_dtype(Shape::from_dims(shape), dtype, arc)
}

pub fn ones_on(shape: &[usize], dtype: DType, dev: &Device) -> DevResult<Tensor> {
    let arc = cuda_arc(dev)?;
    Tensor::ones_dtype(Shape::from_dims(shape), dtype, arc)
}

pub fn randn_on(shape: &[usize], dtype: DType, dev: &Device) -> DevResult<Tensor> {
    let arc = cuda_arc(dev)?;
    let base = Tensor::randn(Shape::from_dims(shape), 0.0, 1.0, arc)?;
    if base.dtype() == dtype {
        Ok(base)
    } else {
        base.to_dtype(dtype)
    }
}

pub fn from_vec_on(data: Vec<f32>, shape: &[usize], dtype: DType, dev: &Device) -> DevResult<Tensor> {
    let arc = cuda_arc(dev)?;
    Tensor::from_vec_dtype(data, Shape::from_dims(shape), arc, dtype)
}
