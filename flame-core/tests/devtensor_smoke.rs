use flame_core::{devtensor, DType, Device};

fn cuda_device() -> Option<Device> {
    Device::cuda(0).ok()
}

#[test]
fn device_constructors_create_expected_shapes() {
    let device = match cuda_device() {
        Some(dev) => dev,
        None => return, // skip if CUDA unavailable
    };

    let shape = [2usize, 3, 4];

    let zeros = devtensor::zeros_on(&shape, DType::F32, &device).expect("zeros_on should succeed");
    assert_eq!(zeros.shape().dims(), &shape);
    assert_eq!(zeros.dtype(), DType::F32);

    let ones = devtensor::ones_on(&shape, DType::F32, &device).expect("ones_on should succeed");
    let host = ones.to_vec().expect("to_vec works");
    assert!(host.iter().all(|&v| (v - 1.0).abs() < 1e-6));

    let bf16_tensor = devtensor::zeros_on(&shape, DType::BF16, &device).expect("bf16 zeros");
    assert_eq!(bf16_tensor.dtype(), DType::BF16);

    let random = devtensor::randn_on(&shape, DType::F32, &device).expect("randn_on should succeed");
    assert_eq!(random.shape().dims(), &shape);

    let from_vec = devtensor::from_vec_on(vec![0.0f32; 6], &[2, 3], DType::F32, &device)
        .expect("from_vec_on should succeed");
    assert_eq!(from_vec.shape().dims(), &[2, 3]);
}
