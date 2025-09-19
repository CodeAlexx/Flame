use flame_core::{attention, Device, DType, Shape, Tensor};

fn randn_on(shape: &[usize], device: &Device) -> Tensor {
    let arc = device.cuda_device().clone();
    Tensor::randn(Shape::from_dims(shape), 0.0, 1.0, arc)
        .and_then(|t| t.to_dtype(DType::F32))
        .expect("randn should succeed")
}

#[test]
fn sdpa_smoke_runs() {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(_) => {
            // No CUDA available in this environment; skip.
            return;
        }
    };

    let (b, h, q_len, k_len, d) = (2usize, 2usize, 8usize, 10usize, 32usize);
    let q = randn_on(&[b, h, q_len, d], &device);
    let k = randn_on(&[b, h, k_len, d], &device);
    let v = randn_on(&[b, h, k_len, d], &device);

    let out = attention::attention(&q, &k, &v, None).expect("sdpa attention should run");
    assert_eq!(out.shape().dims(), &vec![b, h, q_len, d]);
}
