#[cfg(test)]
mod tests {
    use crate::{Tensor, Shape, Device, DType};

    #[test]
    fn gather_rows_basic() -> crate::Result<()> {
        let dev = Device::cuda(0)?;
        let v=7usize; let d=5usize; let b=2usize; let s=3usize;
        let table: Vec<f32> = (0..v*d).map(|i| (i as f32)/10.0).collect();
        let tab_t = Tensor::from_vec(table.clone(), Shape::from_dims(&[v,d]), dev.cuda_device().clone())?;
        let ids = Tensor::from_vec(vec![1.0,3.0,6.0, 0.0,2.0,4.0], Shape::from_dims(&[b,s]), dev.cuda_device().clone())?
            .to_dtype(DType::I32)?;
        let out = tab_t.index_select0(&ids)?.to_dtype(DType::F32)?;
        let o = out.to_vec()?;
        assert_eq!(out.shape().dims(), &[b,s,d]);
        // spot check a couple values
        // (0,0)-> row1 col0 => 1*5+0 =5 -> 0.5
        assert!((o[0] - 0.5).abs() < 1e-6);
        // (1,2)-> row4 col0 => 4*5+0 =20 -> 2.0
        let idx = ((1*3 + 2)*5) as usize;
        assert!((o[idx] - 2.0).abs() < 1e-6);
        Ok(())
    }
}
