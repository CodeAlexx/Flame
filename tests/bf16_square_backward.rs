use env_logger;
use flame_core::Result;

#[cfg(feature = "cuda")]
fn run_square_backward_bf16_smoke() -> Result<()> {
    use flame_core::{device::Device as FlameDevice, DType, Shape, Tensor};

    let _ = env_logger::builder().is_test(true).try_init();
    let device = FlameDevice::cuda(0)?;
    let cuda = device.cuda_device_arc();

    let data: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4];
    let tensor =
        Tensor::from_vec_dtype(data, Shape::from_dims(&[2, 2]), cuda.clone(), DType::BF16)?
            .requires_grad_(true);

    let loss = tensor.square()?.mean()?.requires_grad_(true);
    let grads = loss.backward()?;

    let grad_tensor = grads
        .get(tensor.id())
        .expect("missing gradient for input tensor");
    let grads_vec = grad_tensor.to_dtype(DType::F32)?.to_vec()?;

    let expected = vec![
        0.1 * 2.0 / 4.0,
        -0.2 * 2.0 / 4.0,
        0.3 * 2.0 / 4.0,
        -0.4 * 2.0 / 4.0,
    ];
    for (got, exp) in grads_vec.iter().zip(expected.iter()) {
        assert!(
            (got - exp).abs() < 5e-3,
            "grad mismatch: got={} expected={}",
            got,
            exp
        );
    }
    Ok(())
}

#[test]
fn square_backward_bf16_smoke() -> Result<()> {
    if cfg!(feature = "cuda") {
        #[cfg(feature = "cuda")]
        {
            return run_square_backward_bf16_smoke();
        }
    }
    Ok(())
}
