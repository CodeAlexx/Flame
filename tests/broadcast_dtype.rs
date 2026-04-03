#![cfg(all(feature = "cuda", feature = "heavy_kernels", feature = "bf16_u16"))]

use flame_core::{device::Device, DType, Result, Shape, Tensor};

fn cuda_device() -> Device {
    Device::cuda(0).expect(
        "CUDA GPU required. Set CUDA_HOME=/usr/local/cuda and export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH.",
    )
}

#[test]
fn broadcast_bf16_gpu_preserves_storage() -> Result<()> {
    let dev = cuda_device().cuda_device().clone();

    let c = 16usize;
    let data: Vec<f32> = (0..c).map(|v| v as f32).collect();
    let x = Tensor::from_vec_dtype(data, Shape::from_dims(&[c]), dev.clone(), DType::BF16)?;
    assert_eq!(x.dtype(), DType::BF16);

    let target = Shape::from_dims(&[2, 3, c]);
    let y = x.broadcast_to(&target)?;
    assert_eq!(y.dtype(), DType::BF16);
    assert_eq!(y.shape(), &target);

    let host = y.to_dtype(DType::F32)?.to_vec_f32()?;
    for i in 0..host.len() {
        let want = (i % c) as f32;
        assert!(
            (host[i] - want).abs() < 1e-6,
            "broadcast mismatch at index {i}"
        );
    }

    Ok(())
}
