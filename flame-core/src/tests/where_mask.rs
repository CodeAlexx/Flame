#[cfg(test)]
mod tests {
    use crate::{Tensor, Shape, Device, DType};

    #[test]
    fn where_mask_basic() -> crate::Result<()> {
        let dev = Device::cuda(0)?;
        let a = Tensor::from_vec(vec![1.0,2.0,3.0,4.0], Shape::from_dims(&[2,2]), dev.cuda_device().clone())?;
        let b = Tensor::from_vec(vec![5.0,6.0,7.0,8.0], Shape::from_dims(&[2,2]), dev.cuda_device().clone())?;
        let m = Tensor::from_vec(vec![0.0,1.0,0.0,1.0], Shape::from_dims(&[2,2]), dev.cuda_device().clone())?.to_dtype(DType::Bool)?;
        let out = crate::Tensor::where_mask(&m, &a, &b)?;
        let v = out.to_vec()?;
        assert_eq!(v, vec![5.0,2.0,7.0,4.0]);
        Ok(())
    }
}
