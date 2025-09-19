use flame_core::{ops_ext, Device, DType, Shape, Tensor};
use std::sync::Arc;

fn cuda_device() -> Option<Arc<cudarc::driver::CudaDevice>> {
    Device::cuda(0).ok().map(|dev| dev.cuda_device().clone())
}

fn tensor_from(data: Vec<f32>, dims: &[usize], device: &Arc<cudarc::driver::CudaDevice>) -> Tensor {
    Tensor::from_vec_dtype(data, Shape::from_dims(dims), device.clone(), DType::F32)
        .expect("tensor creation")
}

#[test]
fn ops_ext_helpers_behave() {
    let device = match cuda_device() {
        Some(dev) => dev,
        None => return, // skip without CUDA
    };

    let q = tensor_from((0..24).map(|v| v as f32).collect(), &[1, 2, 3, 4], &device);
    let k = tensor_from((0..24).map(|v| (v * 2) as f32).collect(), &[1, 2, 3, 4], &device);

    let shape = ops_ext::shape4(&q).expect("shape4");
    assert_eq!(shape, (1, 2, 3, 4));

    let qt = ops_ext::transpose_last2(&q).expect("transpose_last2");
    assert_eq!(qt.shape().dims(), &[1, 2, 4, 3]);

    let prod = ops_ext::matmul_tt(&q, &k).expect("matmul_tt");
    assert_eq!(prod.shape().dims(), &[1, 2, 3, 3]);

    let zeros = ops_ext::zeros_like(&q).expect("zeros_like");
    assert_eq!(zeros.dtype(), q.dtype());

    let filled = ops_ext::full_like(&q, 3.5).expect("full_like");
    let host = filled.to_vec().expect("copy to host");
    assert!(host.iter().all(|&v| (v - 3.5).abs() < 1e-6));

    let mask = tensor_from(vec![1.0, 0.0, 1.0, 0.0], &[1, 2, 1, 2], &device);
    let a = tensor_from(vec![10.0; 4], &[1, 2, 1, 2], &device);
    let b = tensor_from(vec![1.0; 4], &[1, 2, 1, 2], &device);
    let mixed = ops_ext::where_mask(&mask, &a, &b).expect("where_mask");
    let mixed_host = mixed.to_vec().expect("host");
    assert_eq!(mixed_host, vec![10.0, 1.0, 10.0, 1.0]);

    let mean = ops_ext::mean_all_f32(&q).expect("mean_all_f32");
    assert!((mean - 11.5).abs() < 1e-6);
}
