use flame_core::*;

#[test]
fn maximum_broadcast_and_ties() -> Result<()> {
    let dev = CudaDevice::new(0).unwrap();
    set_default_dtype(DType::F32);

    // a: [2,3], b: [1,3] -> broadcast b across dim 0
    let a = Tensor::from_vec(vec![0.0, 2.0, 5.0, 5.0, 1.0, 7.0], Shape::from_dims(&[2,3]), dev.clone())?
        .requires_grad_(true);
    let b = Tensor::from_vec(vec![1.0, 2.0, 5.0], Shape::from_dims(&[1,3]), dev.clone())?
        .requires_grad_(true);

    // y = max(a,b) ; loss = mean(y)
    let y = a.maximum(&b)?;
    let loss = y.mean()?;
    let grads = AutogradContext::backward(&loss)?;

    let ga = grads.get(a.id()).unwrap().to_vec()?;
    let gb = grads.get(b.id()).unwrap().to_vec()?;

    // Each element contributes 1/(2*3) to grad because of mean over 6 elements.
    let g = 1.0f32 / 6.0;

    // Expected grads for lhs-wins-ties policy (FLAME uses >= mask for lhs):
    // y = [[1,2,5],[5,2,7]]
    // row 0: [b wins, tie->a, tie->a] -> a grads [0, g, g]
    // row 1: [a wins, b wins, a wins] -> a grads [g, 0, g]
    assert!((ga[0] - 0.0).abs() < 1e-6);
    assert!((ga[1] - g).abs() < 1e-6);
    assert!((ga[2] - g).abs() < 1e-6);
    assert!((ga[3] - g).abs() < 1e-6);
    assert!((ga[4] - 0.0).abs() < 1e-6);
    assert!((ga[5] - g).abs() < 1e-6);

    // b receives grads only where b strictly wins; broadcasting sums grads across rows
    // b wins at (0,0) and (1,1)
    let expected_gb = [g, g, 0.0];
    for i in 0..3 {
        assert!((expected_gb[i] - gb[i]).abs() < 1e-6, "gb[{}] = {}, expected {}", i, gb[i], expected_gb[i]);
    }

    Ok(())
}

