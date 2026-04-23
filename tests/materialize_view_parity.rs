#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

// Parity test for `Tensor::contiguous()` on views with non-zero view_offset.
//
// Constructs a hand-built narrow view (storage shared, custom_strides + offset)
// and verifies that materializing it produces the same bytes as the equivalent
// contiguous slice of the source data. This exercises the new
// `materialize_strided_{bf16,f32}` kernel path wired into `contiguous()`.

use flame_core::{DType, Result, Shape, Tensor};
use std::sync::Arc;

use cudarc::driver::CudaDevice;

fn cuda_device() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect("CUDA GPU required for materialize_view_parity")
}

// Build a hand-crafted narrow view of `src` along `dim`, without going through
// `Tensor::narrow` (which currently materializes). Returns a Tensor that shares
// storage with `src` but has reduced shape, source strides, and the appropriate
// element offset.
fn hand_narrow_view(src: &Tensor, dim: usize, start: usize, length: usize) -> Tensor {
    let dims = src.shape().dims();
    let mut new_dims = dims.to_vec();
    new_dims[dim] = length;

    let strides = src.strides();
    let offset = start * strides[dim];

    src.as_strided(&new_dims, &strides, offset)
        .expect("as_strided failed")
}

#[test]
fn contiguous_of_narrow_view_matches_expected_f32() -> Result<()> {
    let dev = cuda_device();
    // [2, 5, 3]: narrow dim=1 start=1 length=3 → logical [2, 3, 3]
    let shape = Shape::from_dims(&[2, 5, 3]);
    let n = shape.elem_count();
    let data: Vec<f32> = (0..n).map(|v| v as f32).collect();
    let src = Tensor::from_vec_dtype(data.clone(), shape.clone(), dev.clone(), DType::F32)?;

    let view = hand_narrow_view(&src, 1, 1, 3);
    assert_eq!(view.offset(), 1 * 3); // start * stride[1] = 1*3 = 3

    let mat = view.contiguous()?;
    assert_eq!(mat.shape().dims(), &[2, 3, 3]);
    let got = mat.to_vec()?;

    // Expected: for each batch b in 0..2 and j in 1..4 (narrow), copy 3 elems.
    let mut expected: Vec<f32> = Vec::with_capacity(2 * 3 * 3);
    for b in 0..2 {
        for j in 1..4 {
            for d in 0..3 {
                expected.push(data[b * 5 * 3 + j * 3 + d]);
            }
        }
    }
    assert_eq!(got, expected, "materialize_view f32 mismatch");
    Ok(())
}

#[test]
fn contiguous_of_narrow_view_matches_expected_bf16() -> Result<()> {
    let dev = cuda_device();
    // BF16 can represent small integers exactly up to 256, so keep values small.
    let shape = Shape::from_dims(&[2, 5, 3]);
    let n = shape.elem_count();
    let data: Vec<f32> = (0..n).map(|v| v as f32).collect();
    let src = Tensor::from_vec_dtype(data.clone(), shape.clone(), dev.clone(), DType::BF16)?;

    let view = hand_narrow_view(&src, 1, 1, 3);
    assert_eq!(view.offset(), 3);

    let mat = view.contiguous()?;
    assert_eq!(mat.shape().dims(), &[2, 3, 3]);
    // BF16 readout via to_vec() -> F32 for comparison.
    let got = mat.to_vec()?;

    let mut expected: Vec<f32> = Vec::with_capacity(2 * 3 * 3);
    for b in 0..2 {
        for j in 1..4 {
            for d in 0..3 {
                expected.push(data[b * 5 * 3 + j * 3 + d]);
            }
        }
    }
    assert_eq!(got, expected, "materialize_view bf16 mismatch");
    Ok(())
}

#[test]
fn contiguous_of_last_dim_narrow_view_bf16() -> Result<()> {
    // Last-dim narrow: strides [15,3,1], narrow dim=2 start=1 length=2 → offset=1
    let dev = cuda_device();
    let shape = Shape::from_dims(&[2, 5, 3]);
    let n = shape.elem_count();
    let data: Vec<f32> = (0..n).map(|v| v as f32).collect();
    let src = Tensor::from_vec_dtype(data.clone(), shape.clone(), dev.clone(), DType::BF16)?;

    let view = hand_narrow_view(&src, 2, 1, 2);
    assert_eq!(view.offset(), 1);
    let mat = view.contiguous()?;
    assert_eq!(mat.shape().dims(), &[2, 5, 2]);
    let got = mat.to_vec()?;

    let mut expected: Vec<f32> = Vec::with_capacity(2 * 5 * 2);
    for b in 0..2 {
        for j in 0..5 {
            for d in 1..3 {
                expected.push(data[b * 15 + j * 3 + d]);
            }
        }
    }
    assert_eq!(got, expected);
    Ok(())
}
