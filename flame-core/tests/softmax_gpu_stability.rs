use flame_core::*;

fn row_sums_close_to_one(x: &Tensor) -> Result<bool> {
    // Sum over classes dimension (dim=1), keepdim -> shape [N,1]
    let sums = x.sum_dim_keepdim(1)?;
    let ones = Tensor::ones(sums.shape().clone(), sums.device().clone())?;
    let diff = sums.sub(&ones)?.abs()?;
    let v = diff.to_vec()?;
    let max_abs = v.into_iter().fold(0f32, |m, d| if d > m { d } else { m });
    Ok(max_abs < 1e-4)
}

#[test]
fn softmax_forward_backward_stability() -> Result<()> {
    let dev = CudaDevice::new(0).unwrap();
    set_default_dtype(DType::F32);
    let (n, classes) = (4usize, 17usize);

    // logits: [N, C]
    let logits = Tensor::randn(Shape::from_dims(&[n, classes]), 0.0, 1.0, dev.clone())?
        .requires_grad_(true);

    // Softmax along dim=1 (classes)
    let probs = logits.softmax(1)?;
    assert!(row_sums_close_to_one(&probs)?);

    // Simple scalar loss to trigger backward: mean of KL(probs || uniform)
    let uni = Tensor::full(Shape::from_dims(&[n, classes]), 1.0f32 / classes as f32, dev.clone())?;
    // KL = sum p * (log p - log u)
    let loss = probs.clone_result()?.mul(&probs.log()?)?.sub(&probs.mul(&uni.log()?)?)?.mean()?;
    let grads = AutogradContext::backward(&loss)?;

    // Grads finite?
    let g = grads.get(logits.id()).unwrap();
    let vv = g.to_vec()?;
    assert!(vv.iter().all(|v| v.is_finite()));
    Ok(())
}
