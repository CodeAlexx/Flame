#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use anyhow::Result;
use flame_core::device::Device;
use flame_core::tensor::Tensor;
use flame_core::{DType, Shape};
use rand::Rng;

fn random_bf16(shape: &[usize]) -> Result<Tensor> {
    let numel: usize = shape.iter().product();
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..numel).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
    let device = Device::cuda(0)?;
    let tensor = Tensor::from_vec(data, Shape::from_dims(shape), device.cuda_device_arc())?;
    Ok(tensor.to_dtype(DType::BF16)?)
}

#[test]
fn bf16_slice_matches_cpu() -> Result<()> {
    let shape = [2, 5, 3];
    let tensor = random_bf16(&shape)?;
    let axis = 1;
    let start = 1;
    let len = 3;

    let cpu_ref = tensor.to_dtype(DType::F32)?.to_vec()?;
    let slice = tensor.slice_1d_device(axis, start, len)?;
    let slice_host = slice.to_dtype(DType::F32)?.to_vec()?;

    for b in 0..shape[0] {
        for i in 0..len {
            for c in 0..shape[2] {
                let src_idx = ((b * shape[1] + (start + i)) * shape[2] + c) as usize;
                let dst_idx = ((b * len + i) * shape[2] + c) as usize;
                let ref_val = cpu_ref[src_idx];
                let got = slice_host[dst_idx];
                assert!(
                    (ref_val - got).abs() < 2e-3,
                    "slice mismatch ref={} got={}",
                    ref_val,
                    got
                );
            }
        }
    }
    Ok(())
}

#[test]
fn bf16_broadcast_matches_cpu() -> Result<()> {
    let tensor = random_bf16(&[1, 2, 3])?;
    let out = tensor.broadcast_to_device(&[4, 2, 3])?;

    let host_src = tensor.to_dtype(DType::F32)?.to_vec()?;
    let host_out = out.to_dtype(DType::F32)?.to_vec()?;

    for b in 0..4 {
        for i in 0..2 {
            for c in 0..3 {
                let dst = ((b * 2 + i) * 3 + c) as usize;
                let src = (i * 3 + c) as usize;
                assert!(
                    (host_out[dst] - host_src[src]).abs() < 2e-3,
                    "broadcast mismatch"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn bf16_repeat_axis_matches_cpu() -> Result<()> {
    let tensor = random_bf16(&[2, 1, 3])?;
    let out = tensor.repeat_axis_device(1, 4)?;

    let host_src = tensor.to_dtype(DType::F32)?.to_vec()?;
    let host_out = out.to_dtype(DType::F32)?.to_vec()?;

    for b in 0..2 {
        for rep in 0..4 {
            for c in 0..3 {
                let dst = ((b * 4 + rep) * 3 + c) as usize;
                let src = (b * 3 + c) as usize;
                assert!(
                    (host_out[dst] - host_src[src]).abs() < 2e-3,
                    "repeat mismatch"
                );
            }
        }
    }
    Ok(())
}
