use std::sync::Arc;
use flame_core::{Shape, Tensor, DType};
use cudarc::driver::CudaDevice;

fn init_device_once() {
    let _ = CudaDevice::new(0);
}
use cudarc::driver::CudaDevice;

fn host_check_forward(input: &Tensor, out: &Tensor, dim: usize, start: usize, len: usize) {
    let in_dims = input.shape().dims().to_vec();
    let out_dims = out.shape().dims().to_vec();
    assert_eq!(in_dims.len(), out_dims.len());
    assert_eq!(out_dims[dim], len);
    let a = input.to_vec().expect("input to host");
    let b = out.to_vec().expect("out to host");
    let rank = in_dims.len();
    let out_numel: usize = out_dims.iter().product();
    // Compute strides
    let mut in_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() { in_strides[i] = in_strides[i + 1] * in_dims[i + 1]; }
    let mut out_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() { out_strides[i] = out_strides[i + 1] * out_dims[i + 1]; }

    for lin in 0..out_numel {
        // unravel linear index into out multi-index
        let mut rem = lin;
        let mut idx = vec![0usize; rank];
        for i in 0..rank {
            let stride = out_strides[i];
            let val = rem / stride;
            rem -= val * stride;
            idx[i] = val;
        }
        // map to input index
        let mut in_lin = 0usize;
        for i in 0..rank {
            let coord = if i == dim { idx[i] + start } else { idx[i] };
            in_lin += coord * in_strides[i];
        }
        assert!( (b[lin] - a[in_lin]).abs() < 1e-5, "mismatch at {}: {} vs {}", lin, b[lin], a[in_lin]);
    }
}

fn test_forward_dim(dtype: DType, dim: usize) {
    let dev = Arc::new(CudaDevice::new(0).expect("cuda"));
    let in_dims = [2usize, 3usize, 4usize, 5usize];
    let numel: usize = in_dims.iter().product();
    let data: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.01 - 1.0).collect();
    let t = Tensor::from_slice_dtype(&data, Shape::from_dims(&in_dims), dev.clone(), dtype).expect("from");
    let (start, len) = (1usize, match dim { 0 => 1, 1 => 2, 2 => 2, _ => 3 });
    let out = t.narrow_general_cuda(dim, start, len).expect("narrow");
    host_check_forward(&t, &out, dim, start, len);
}

#[test]
fn slice_forward_cuda_fp32_all_dims() {
    init_device_once();
    for dim in 0..4 { test_forward_dim(DType::F32, dim); }
}

#[test]
fn slice_forward_cuda_bf16_all_dims() {
    init_device_once();
    for dim in 0..4 { test_forward_dim(DType::BF16, dim); }
}

fn test_backward_dim(dtype: DType, dim: usize) {
    let dev = Arc::new(CudaDevice::new(0).expect("cuda"));
    let in_dims = [2usize, 3usize, 4usize, 5usize];
    let numel_in: usize = in_dims.iter().product();
    // Build grad_out shape
    let (start, len) = (1usize, match dim { 0 => 1, 1 => 2, 2 => 2, _ => 3 });
    let mut out_dims = in_dims.clone();
    out_dims[dim] = len;
    let numel_out: usize = out_dims.iter().product();
    // grad_out fill
    let go_data: Vec<f32> = (0..numel_out).map(|i| (i as f32) * 0.02 + 0.5).collect();
    let grad_out = Tensor::from_slice_dtype(&go_data, Shape::from_dims(&out_dims), dev.clone(), dtype).expect("go");
    let mut grad_in = Tensor::zeros_dtype(Shape::from_dims(&in_dims), dtype, dev.clone()).expect("gi");
    grad_in.narrow_backward_scatter_add_cuda(&grad_out, &mut grad_in, dim, start, len).expect("backward");
    let gi = grad_in.to_vec().expect("gi vec");
    // Compute expected: zero everywhere except window along dim where it equals grad_out
    // Build strides
    let rank = in_dims.len();
    let mut in_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() { in_strides[i] = in_strides[i + 1] * in_dims[i + 1]; }
    let mut out_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() { out_strides[i] = out_strides[i + 1] * out_dims[i + 1]; }

    // Iterate over all in positions; ensure correct window
    for lin_in in 0..numel_in {
        // unravel in index
        let mut rem = lin_in;
        let mut idx = vec![0usize; rank];
        for i in 0..rank {
            let stride = in_strides[i];
            let val = rem / stride;
            rem -= val * stride;
            idx[i] = val;
        }
        let val = gi[lin_in];
        if idx[dim] >= start && idx[dim] < start + len {
            // map to out index
            let mut lin_out = 0usize;
            for i in 0..rank {
                let coord = if i == dim { idx[i] - start } else { idx[i] };
                lin_out += coord * out_strides[i];
            }
            let expected = go_data[lin_out];
            assert!( (val - expected).abs() < 1e-5, "grad_in mismatch at {}: {} vs {}", lin_in, val, expected);
        } else {
            assert!(val.abs() < 1e-6, "grad_in outside window not zero at {}: {}", lin_in, val);
        }
    }
}

#[test]
fn slice_backward_cuda_fp32_all_dims() {
    init_device_once();
    for dim in 0..4 { test_backward_dim(DType::F32, dim); }
}

#[test]
fn slice_backward_cuda_bf16_all_dims() {
    init_device_once();
    for dim in 0..4 { test_backward_dim(DType::BF16, dim); }
}
