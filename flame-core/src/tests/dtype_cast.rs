#[cfg(test)]
mod tests {
    use crate::{Tensor, Shape, DType, Device};

    #[test]
    fn cast_i32_roundtrip() {
        let dev = Device::cuda(0).unwrap();
        let data: Vec<f32> = (0..12).map(|i| (i as f32)*3.0).collect();
        let t = Tensor::from_vec(data.clone(), Shape::from_dims(&[3,4]), dev.cuda_device().clone()).unwrap();
        let i32t = t.to_dtype(DType::I32).unwrap();
        let back = i32t.to_dtype(DType::F32).unwrap();
        let v = back.to_vec().unwrap();
        assert_eq!(v.len(), 12);
        for i in 0..12 { assert!((v[i] - data[i]).abs() < 1e-6); }
    }

    #[test]
    fn cast_bool_roundtrip() {
        let dev = Device::cuda(0).unwrap();
        let data: Vec<f32> = vec![0.0, 1.0, -2.0, 3.5];
        let t = Tensor::from_vec(data, Shape::from_dims(&[2,2]), dev.cuda_device().clone()).unwrap();
        let bt = t.to_dtype(DType::Bool).unwrap();
        let bf = bt.to_dtype(DType::F32).unwrap();
        let v = bf.to_vec().unwrap();
        assert_eq!(v, vec![0.0, 1.0, 1.0, 1.0]);
    }
}

