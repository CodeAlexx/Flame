#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

// Phase 5b: this test previously called `bf16_elementwise::{add_bf16, mul_bf16}`
// directly. Those legacy entry points were deleted; the ops now live on the
// TensorIterator pipeline in `ops::{add_iter, mul_iter}`. The broadcast
// behaviour tested here is preserved — `build_binary_op` handles broadcast
// via stride=0 internally.
use anyhow::Result;
use flame_core::{
    tensor_iterator::ops::binary::{add_bf16_iter, mul_bf16_iter},
    DType, Device, Shape, Tensor,
};

fn make_tensor(device: &Device, data: &[f32], shape: &[usize]) -> Result<Tensor> {
    Tensor::from_vec(
        data.to_vec(),
        Shape::from_dims(shape),
        device.cuda_device_arc(),
    )?
    .to_dtype(DType::BF16)
    .map_err(Into::into)
}

fn bf16_to_f32(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_dtype(DType::F32)?.to_vec_f32()?)
}

#[test]
fn add_mul_broadcast_match_fp32() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(err) => {
            eprintln!("[skip] CUDA unavailable: {err}");
            return Ok(());
        }
    };

    let a_data: Vec<f32> = (0..8).map(|v| v as f32 * 0.5).collect();
    let b_data: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0];
    let a = make_tensor(&device, &a_data, &[2, 4])?;
    let b = make_tensor(&device, &b_data, &[1, 4])?;

    let sum_bf16 = add_bf16_iter(&a, &b)?;
    let prod_bf16 = mul_bf16_iter(&a, &b)?;

    let sum_f32 = bf16_to_f32(&sum_bf16)?;
    let prod_f32 = bf16_to_f32(&prod_bf16)?;

    for (idx, (sum, prod)) in sum_f32.iter().zip(prod_f32.iter()).enumerate() {
        let col = idx % 4;
        let expected_sum = a_data[idx] + b_data[col];
        let expected_prod = a_data[idx] * b_data[col];
        assert!((sum - expected_sum).abs() < 5e-3, "sum mismatch at {}", idx);
        assert!(
            (prod - expected_prod).abs() < 5e-3,
            "prod mismatch at {}",
            idx
        );
    }

    Ok(())
}
