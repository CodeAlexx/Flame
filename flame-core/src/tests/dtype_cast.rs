#[cfg(test)]
mod tests {
    use crate::{Tensor, Shape, DType, Device};

    #[test]
    fn cast_i32_roundtrip() -> crate::Result<()> {
        let dev = Device::cuda(0)?;
        let data: Vec<f32> = (0..12).map(|i| (i as f32)*3.0).collect();
        let t = Tensor::from_vec(data.clone(), Shape::from_dims(&[3,4]), dev.cuda_device().clone())?;
        let i32t = t.to_dtype(DType::I32)?;
        let back = i32t.to_dtype(DType::F32)?;
        let v = back.to_vec()?;
        assert_eq!(v.len(), 12);
        for i in 0..12 { assert!((v[i] - data[i]).abs() < 1e-6); }
        Ok(())
    }

    #[test]
    fn cast_bool_roundtrip() -> crate::Result<()> {
        let dev = Device::cuda(0)?;
        let data: Vec<f32> = vec![0.0, 1.0, -2.0, 3.5];
        let t = Tensor::from_vec(data, Shape::from_dims(&[2,2]), dev.cuda_device().clone())?;
        let bt = t.to_dtype(DType::Bool)?;
        let bf = bt.to_dtype(DType::F32)?;
        let v = bf.to_vec()?;
        assert_eq!(v, vec![0.0, 1.0, 1.0, 1.0]);
        Ok(())
    }
}
