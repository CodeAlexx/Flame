use flame_core::*;

fn allclose(a: &Tensor, b: &Tensor, rtol: f32, atol: f32) -> Result<bool> {
    let diff = a.sub(b)?.abs()?;
    let tol = b.abs()?.mul_scalar(rtol)?.add_scalar(atol)?;
    let cmp = diff.le(&tol)?; // 1 where close, 0 otherwise
    let v = cmp.to_vec()?;
    Ok(v.into_iter().all(|x| x >= 0.5))
}

#[test]
fn reduce_max_keepdim_matches_cpu_single_axis() -> Result<()> {
    let dev = CudaDevice::new(0).unwrap();
    set_default_dtype(DType::F32);
    let x = Tensor::randn(Shape::from_dims(&[3,5,7]), 0.0, 1.0, dev.clone())?;
    let axes = [0usize, 1usize, 2usize];

    for &axis in &axes {
        // GPU path
        let gpu = crate::cuda_ops::GpuOps::max_dim(&x, axis, true)?;
        // CPU reference via Tensor method (uses to_vec internally)
        let cpu = x.max_dim(axis, true)?;
        assert!(allclose(&gpu, &cpu, 1e-5, 1e-6)?);
    }
    Ok(())
}

