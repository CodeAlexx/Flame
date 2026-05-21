#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::{CudaDevice, DType, Result, Shape, Tensor};

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
fn f32_lora_b_grad_matches_reference() -> Result<()> {
    flame_core::config::set_default_dtype(DType::BF16);
    let dev = flame_core::global_cuda_device();
    let _ = CudaDevice::new(0).expect("CUDA GPU required");

    let bsz = 1usize;
    let seq = 3usize;
    let cin = 4usize;
    let rank = 2usize;
    let cout = 5usize;
    let scale = 1.25f32;

    let x_data: Vec<f32> = (0..bsz * seq * cin)
        .map(|i| (i as f32 - 5.0) * 0.03125)
        .collect();
    let a_data: Vec<f32> = (0..rank * cin)
        .map(|i| (i as f32 - 3.0) * 0.0625)
        .collect();
    let b_data = vec![0.0f32; cout * rank];
    let upstream_data: Vec<f32> = (0..bsz * seq * cout)
        .map(|i| (i as f32 - 7.0) * 0.015625)
        .collect();

    let x = Tensor::from_vec_dtype(
        x_data.clone(),
        Shape::from_dims(&[bsz, seq, cin]),
        dev.clone(),
        DType::F32,
    )?;
    let a = Tensor::from_vec_dtype(
        a_data.clone(),
        Shape::from_dims(&[rank, cin]),
        dev.clone(),
        DType::F32,
    )?;
    let b = Tensor::from_vec_dtype(
        b_data,
        Shape::from_dims(&[cout, rank]),
        dev.clone(),
        DType::F32,
    )?
    .requires_grad_(true);
    let upstream = Tensor::from_vec_dtype(
        upstream_data.clone(),
        Shape::from_dims(&[bsz, seq, cout]),
        dev.clone(),
        DType::F32,
    )?;

    let a_t = a.transpose()?.contiguous()?;
    let b_t = b.transpose()?.contiguous()?;
    let xa = x.matmul(&a_t)?;
    let out = xa.matmul(&b_t)?.mul_scalar(scale)?;
    let loss = out.mul(&upstream)?.sum()?;
    let grads = loss.backward()?;
    let grad_b = grads
        .get(b.id())
        .expect("missing grad for LoRA B")
        .to_dtype(DType::F32)?
        .to_vec_f32()?;

    let mut xa_ref = vec![0.0f32; bsz * seq * rank];
    for row in 0..(bsz * seq) {
        for r in 0..rank {
            let mut sum = 0.0f32;
            for c in 0..cin {
                sum += x_data[row * cin + c] * a_data[r * cin + c];
            }
            xa_ref[row * rank + r] = sum;
        }
    }

    let mut expected = vec![0.0f32; cout * rank];
    for o in 0..cout {
        for r in 0..rank {
            let mut sum = 0.0f32;
            for row in 0..(bsz * seq) {
                sum += xa_ref[row * rank + r] * upstream_data[row * cout + o] * scale;
            }
            expected[o * rank + r] = sum;
        }
    }

    assert_close(&grad_b, &expected, 1.0e-6);
    Ok(())
}
