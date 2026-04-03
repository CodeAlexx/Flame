#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use anyhow::Result;
use flame_core::{DType, Device, Shape, Tensor};

fn randn(shape: &[usize], device: &Device) -> Result<Tensor> {
    let arc = device.cuda_device().clone();
    Ok(Tensor::randn(Shape::from_dims(shape), 0.0, 1.0, arc)?.to_dtype(DType::BF16)?)
}

fn compare_activation<F>(input: &Tensor, f_bf16: F, f_ref: fn(f32) -> f32, name: &str) -> Result<()>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let out_bf16 = f_bf16(input)?.to_dtype(DType::F32)?.to_vec_f32()?;
    let input_f32 = input.to_dtype(DType::F32)?;
    let out_ref: Vec<f32> = input_f32.to_vec_f32()?.into_iter().map(f_ref).collect();

    let max_diff = out_bf16
        .iter()
        .zip(out_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(max_diff < 7e-3, "{name} BF16 mismatch: max diff {max_diff}");
    Ok(())
}

#[test]
fn gelu_and_silu_match_reference() -> Result<()> {
    let device = match Device::cuda(0) {
        Ok(dev) => dev,
        Err(err) => {
            eprintln!("[skip] CUDA unavailable: {err}");
            return Ok(());
        }
    };

    let x = randn(&[4, 8, 16], &device)?;

    compare_activation(
        &x,
        |t| Ok(t.gelu()?),
        |v| {
            let c = 0.7978845608_f32 * (v + 0.044715_f32 * v * v * v);
            0.5_f32 * v * (1.0_f32 + c.tanh())
        },
        "gelu",
    )?;

    compare_activation(&x, |t| Ok(t.silu()?), |v| v / (1.0 + (-v).exp()), "silu")?;

    Ok(())
}
