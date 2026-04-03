#![cfg(all(feature = "cuda", feature = "heavy_kernels", feature = "bf16_u16"))]

use flame_core::{
    cuda_ops::GpuOps,
    ops::{elt, reduce, tile},
    tensor_ext::to_owning_fp32_strong,
    DType, Result, Shape, Tensor,
};
use std::sync::Arc;

use cudarc::driver::CudaDevice;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect(
        "CUDA GPU required. Set CUDA_HOME=/usr/local/cuda and export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH.",
    )
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (idx, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (got - want).abs();
        assert!(
            diff <= tol,
            "value mismatch at {idx}: got {got}, expected {want}, diff {diff}"
        );
    }
}

#[test]
fn narrow_strided_forward_backward_fp32() -> Result<()> {
    let dev = cuda_device();

    let shape = Shape::from_dims(&[2, 3, 4]);
    let data: Vec<f32> = (0..shape.elem_count()).map(|v| v as f32).collect();
    let tensor = Tensor::from_vec_dtype(data.clone(), shape.clone(), dev.clone(), DType::F32)?;

    let sliced = tensor.narrow_general_cuda(1, 1, 1)?;
    assert_eq!(sliced.shape().dims(), &[2, 1, 4]);
    assert_eq!(sliced.dtype(), DType::F32);
    assert_eq!(sliced.storage_dtype(), DType::F32);

    let host = sliced.to_vec_f32()?;
    let expected: Vec<f32> = (0..2)
        .flat_map(|b| (0..4).map(move |c| ((b * 3 + 1) * 4 + c) as f32))
        .collect();
    assert_eq!(host, expected);

    let mut grad_in = Tensor::zeros_dtype(shape.clone(), DType::F32, dev.clone())?;
    let grad_vals: Vec<f32> = (0..8).map(|v| (v as f32) * 0.5).collect();
    let grad_out = Tensor::from_vec_dtype(
        grad_vals.clone(),
        Shape::from_dims(&[2, 1, 4]),
        dev.clone(),
        DType::F32,
    )?;
    Tensor::narrow_backward_scatter_add_cuda(&grad_out, &mut grad_in, 1, 1, 1)?;
    let grad_host = grad_in.to_vec_f32()?;

    let mut expected_grad = vec![0.0f32; shape.elem_count()];
    for b in 0..2 {
        for c in 0..4 {
            let src_idx = b * 4 + c;
            let dst_idx = (b * 3 + 1) * 4 + c;
            expected_grad[dst_idx] = grad_vals[src_idx];
        }
    }
    assert_eq!(grad_host, expected_grad);

    Ok(())
}

#[test]
fn narrow_strided_forward_backward_bf16() -> Result<()> {
    let dev = cuda_device();

    let shape = Shape::from_dims(&[1, 4, 2]);
    let data: Vec<f32> = (0..shape.elem_count()).map(|v| v as f32).collect();
    let tensor = Tensor::from_vec_dtype(data.clone(), shape.clone(), dev.clone(), DType::BF16)?;

    let sliced = tensor.narrow_general_cuda(1, 2, 2)?;
    assert_eq!(sliced.shape().dims(), &[1, 2, 2]);
    assert_eq!(sliced.dtype(), DType::BF16);
    assert_eq!(sliced.storage_dtype(), DType::BF16);

    let host = sliced.to_dtype(DType::F32)?.to_vec_f32()?;
    let expected: Vec<f32> = (0..2)
        .flat_map(|row| (0..2).map(move |c| ((row + 2) * 2 + c) as f32))
        .collect();
    assert_eq!(host, expected);

    let mut grad_in = Tensor::zeros_dtype(shape.clone(), DType::BF16, dev.clone())?;
    let grad_vals: Vec<f32> = (0..4).map(|v| (v as f32) - 1.25).collect();
    let grad_out = Tensor::from_vec_dtype(
        grad_vals.clone(),
        Shape::from_dims(&[1, 2, 2]),
        dev.clone(),
        DType::BF16,
    )?;
    Tensor::narrow_backward_scatter_add_cuda(&grad_out, &mut grad_in, 1, 2, 2)?;
    let grad_host = grad_in.to_dtype(DType::F32)?.to_vec_f32()?;

    let mut expected_grad = vec![0.0f32; shape.elem_count()];
    for row in 0..2 {
        for c in 0..2 {
            let src_idx = row * 2 + c;
            let dst_idx = (row + 2) * 2 + c;
            expected_grad[dst_idx] = grad_vals[src_idx];
        }
    }
    assert_eq!(grad_host, expected_grad);

    Ok(())
}

#[test]
fn permute_0213_matches_reference() -> Result<()> {
    let dev = cuda_device();

    let shape = Shape::from_dims(&[2, 3, 4, 2]);
    let data: Vec<f32> = (0..shape.elem_count()).map(|v| v as f32 + 0.5).collect();
    let f32_tensor = Tensor::from_vec_dtype(data.clone(), shape.clone(), dev.clone(), DType::F32)?;
    let bf16_tensor =
        Tensor::from_vec_dtype(data.clone(), shape.clone(), dev.clone(), DType::BF16)?;

    let perm_f32 = GpuOps::permute_0213(&f32_tensor)?;
    let perm_bf16 = GpuOps::permute_0213(&bf16_tensor)?;

    assert_eq!(perm_f32.shape().dims(), &[2, 4, 3, 2]);
    assert_eq!(perm_bf16.shape().dims(), &[2, 4, 3, 2]);
    assert_eq!(perm_f32.dtype(), DType::F32);
    assert_eq!(perm_bf16.dtype(), DType::BF16);

    let host_f32 = perm_f32.to_vec_f32()?;
    let host_bf16 = perm_bf16.to_dtype(DType::F32)?.to_vec_f32()?;

    let mut expected = Vec::with_capacity(shape.elem_count());
    for n in 0..2 {
        for b in 0..4 {
            for a in 0..3 {
                for c in 0..2 {
                    let idx = (((n * 3 + a) * 4 + b) * 2 + c) as usize;
                    expected.push(data[idx]);
                }
            }
        }
    }
    assert_close(&host_f32, &expected, 1e-6);
    assert_close(&host_bf16, &expected, 5e-3);

    Ok(())
}

#[test]
fn sum_last_keepdim_matches_cpu() -> Result<()> {
    let dev = cuda_device();

    let b = 2usize;
    let m = 3usize;
    let k = 5usize;
    let shape = Shape::from_dims(&[b, m, k]);
    let data: Vec<f32> = (0..shape.elem_count()).map(|v| v as f32 * 0.25).collect();
    let tensor = Tensor::from_vec_dtype(data.clone(), shape.clone(), dev.clone(), DType::BF16)?;

    let reduced = reduce::sum_dim_keepdim_as(&tensor, 2, DType::BF16)?;
    assert_eq!(reduced.shape().dims(), &[b, m, 1]);
    assert_eq!(reduced.dtype(), DType::BF16);

    let host = reduced.to_dtype(DType::F32)?.to_vec_f32()?;
    let mut expected = Vec::with_capacity(b * m);
    for batch in 0..b {
        for row in 0..m {
            let mut acc = 0.0f32;
            for col in 0..k {
                let idx = ((batch * m + row) * k) + col;
                acc += data[idx];
            }
            expected.push(acc);
        }
    }
    assert_close(&host, &expected, 1e-2);

    Ok(())
}

#[test]
fn add_inplace_same_dtype_works() -> Result<()> {
    let dev = cuda_device();

    let shape = Shape::from_dims(&[4, 4]);
    let left_data: Vec<f32> = (0..shape.elem_count()).map(|v| v as f32).collect();
    let right_data: Vec<f32> = (0..shape.elem_count()).map(|v| (v as f32) * 0.1).collect();

    let mut lhs_f32 =
        Tensor::from_vec_dtype(left_data.clone(), shape.clone(), dev.clone(), DType::F32)?;
    let rhs_f32 =
        Tensor::from_vec_dtype(right_data.clone(), shape.clone(), dev.clone(), DType::F32)?;
    elt::add_inplace_same_dtype(&mut lhs_f32, &rhs_f32)?;
    let host_f32 = lhs_f32.to_vec_f32()?;
    let expected_f32: Vec<f32> = left_data
        .iter()
        .zip(&right_data)
        .map(|(a, b)| a + b)
        .collect();
    assert_close(&host_f32, &expected_f32, 1e-6);

    let mut lhs_bf16 =
        Tensor::from_vec_dtype(left_data.clone(), shape.clone(), dev.clone(), DType::BF16)?;
    let rhs_bf16 =
        Tensor::from_vec_dtype(right_data.clone(), shape.clone(), dev.clone(), DType::BF16)?;
    elt::add_inplace_same_dtype(&mut lhs_bf16, &rhs_bf16)?;
    let host_bf16 = lhs_bf16.to_dtype(DType::F32)?.to_vec_f32()?;
    assert_close(&host_bf16, &expected_f32, 3e-2);

    Ok(())
}

#[test]
fn tile_bc_to_bhwc_replicates_rows() -> Result<()> {
    let dev = cuda_device();

    let b = 2usize;
    let c = 3usize;
    let h = 2usize;
    let w = 4usize;

    let shape = Shape::from_dims(&[b, c]);
    let data: Vec<f32> = (0..shape.elem_count()).map(|v| v as f32 + 1.0).collect();
    let tensor = Tensor::from_vec_dtype(data.clone(), shape.clone(), dev.clone(), DType::F32)?;

    let tiled = tile::tile_bc_to_bhwc_f32(&tensor, b, h, w, c)?;
    assert_eq!(tiled.shape().dims(), &[b, h, w, c]);
    assert_eq!(tiled.dtype(), DType::F32);
    assert_eq!(tiled.storage_dtype(), DType::F32);

    let host = tiled.to_vec_f32()?;
    let mut expected = Vec::with_capacity(b * h * w * c);
    for batch in 0..b {
        for _yy in 0..h {
            for _xx in 0..w {
                let offset = batch * c;
                expected.extend_from_slice(&data[offset..offset + c]);
            }
        }
    }
    assert_eq!(host, expected);

    Ok(())
}

#[test]
fn gemm_bf16_fp32acc_matches_reference() -> Result<()> {
    let dev = cuda_device();

    let a = Tensor::from_vec_dtype(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Shape::from_dims(&[2, 3]),
        dev.clone(),
        DType::BF16,
    )?;
    let b = Tensor::from_vec_dtype(
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        Shape::from_dims(&[3, 2]),
        dev.clone(),
        DType::BF16,
    )?;

    let y = a.matmul(&b)?;
    assert_eq!(y.dtype(), DType::BF16);
    assert_eq!(y.storage_dtype(), DType::BF16);
    let host = y.to_dtype(DType::F32)?.to_vec_f32()?;
    let expected = vec![
        1.0 * 7.0 + 2.0 * 9.0 + 3.0 * 11.0,
        1.0 * 8.0 + 2.0 * 10.0 + 3.0 * 12.0,
        4.0 * 7.0 + 5.0 * 9.0 + 6.0 * 11.0,
        4.0 * 8.0 + 5.0 * 10.0 + 6.0 * 12.0,
    ];
    assert_close(&host, &expected, 5e-2);

    Ok(())
}

#[test]
fn broadcast_kernel_fp32_and_bf16_match_reference() -> Result<()> {
    let dev = cuda_device();

    let base = Shape::from_dims(&[3]);
    let broad = Shape::from_dims(&[2, 3, 3]);
    let data: Vec<f32> = vec![1.0, -2.0, 3.5];

    let x_f32 = Tensor::from_vec_dtype(data.clone(), base.clone(), dev.clone(), DType::F32)?;
    let y_f32 = x_f32.broadcast_to(&broad)?;
    let host_f32 = y_f32.to_vec_f32()?;
    let expected_f32: Vec<f32> = (0..(2 * 3)).flat_map(|_| data.clone()).collect();
    assert_eq!(host_f32, expected_f32);

    let x_bf16 = Tensor::from_vec_dtype(data.clone(), base.clone(), dev.clone(), DType::BF16)?;
    let y_bf16 = x_bf16.broadcast_to(&broad)?;
    let host_bf16 = y_bf16.to_dtype(DType::F32)?.to_vec_f32()?;
    assert_eq!(host_bf16, expected_f32);

    Ok(())
}

#[test]
fn clone_result_preserves_true_fp32_storage() -> Result<()> {
    let dev = cuda_device();

    let shape = Shape::from_dims(&[4, 4]);
    let tensor = Tensor::zeros_dtype(shape.clone(), DType::BF16, dev.clone())?;
    let upcast = to_owning_fp32_strong(&tensor)?;
    assert_eq!(upcast.dtype(), DType::F32);
    assert_eq!(upcast.storage_dtype(), DType::F32);
    assert_eq!(upcast.shape(), &shape);

    Ok(())
}
