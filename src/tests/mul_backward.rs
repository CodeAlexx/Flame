#[cfg(test)]
mod tests {
    use crate::autograd::AutogradContext;
    use crate::{DType, Device, Shape, Tensor};
    use lazy_static::lazy_static;
    use std::sync::Mutex;

    lazy_static! {
        static ref AUTOGRAD_TEST_LOCK: Mutex<()> = Mutex::new(());
    }

    fn make_tensor(data: Vec<f32>, shape: &[usize], device: &Device) -> crate::Result<Tensor> {
        let cuda = device.cuda_device().clone();
        Tensor::from_vec_dtype(data, Shape::from_dims(shape), cuda, DType::BF16)
    }

    #[test]
    #[cfg_attr(not(feature = "cuda"), ignore)]
    fn bwd_mul_bf16_tiny_ok() -> crate::Result<()> {
        let _guard = AUTOGRAD_TEST_LOCK.lock().unwrap();
        AutogradContext::reset();

        let result = (|| {
            let device = Device::cuda(0)?;
            let x = make_tensor(vec![0.1, -0.2, 0.3, -0.4], &[2, 2], &device)?.requires_grad_(true);
            let y = make_tensor(vec![0.5, 0.6, -0.7, 0.8], &[2, 2], &device)?.requires_grad_(true);

            let loss = x.mul(&y)?.mean()?;
            let grads = loss.backward()?;
            device.cuda_device().synchronize()?;

            assert!(grads.get(x.id).is_some());
            assert!(grads.get(y.id).is_some());
            Ok(())
        })();

        AutogradContext::reset();
        result
    }

    #[test]
    #[cfg_attr(not(feature = "cuda"), ignore)]
    fn bwd_mul_bf16_various_n_ok() -> crate::Result<()> {
        let _guard = AUTOGRAD_TEST_LOCK.lock().unwrap();
        AutogradContext::reset();

        let result = (|| {
            let device = Device::cuda(0)?;
            let sizes = [1usize, 2, 3, 4, 7, 32, 1024];

            for &n in &sizes {
                let data_x: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();
                let data_y: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * -0.05).collect();
                let x = make_tensor(data_x, &[n], &device)?.requires_grad_(true);
                let y = make_tensor(data_y, &[n], &device)?.requires_grad_(true);

                let loss = x.mul(&y)?.sum()?;
                let grads = loss.backward()?;
                device.cuda_device().synchronize()?;

                assert!(grads.get(x.id).is_some(), "missing grad for size {}", n);
                assert!(grads.get(y.id).is_some(), "missing grad for size {}", n);
            }

            Ok(())
        })();

        AutogradContext::reset();
        result
    }
}
