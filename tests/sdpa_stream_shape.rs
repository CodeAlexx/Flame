#![cfg(all(feature = "cuda", feature = "heavy_kernels", feature = "bf16_u16"))]

use env_logger;
use flame_core::device::Device;
use flame_core::shape::Shape;
use flame_core::{cuda_ops_bf16, DType, Tensor};

#[test]
#[ignore]
fn inspect_sdpa_stream_sd35_shape() -> flame_core::Result<()> {
    let _ = env_logger::builder().is_test(true).try_init();
    std::env::set_var("FLAME_SDPA_AUTOTUNE", "0");
    std::env::set_var("STREAMING_SDPA_CHUNK_MAX", "96");

    let device = Device::cuda(0)?;
    let cuda = device.cuda_device_arc();
    let shape = Shape::from_dims(&[1, 24, 4096, 64]);
    let q = Tensor::zeros_dtype(shape.clone(), DType::BF16, cuda.clone())?;
    let k = Tensor::zeros_dtype(shape.clone(), DType::BF16, cuda.clone())?;
    let v = Tensor::zeros_dtype(shape.clone(), DType::BF16, cuda.clone())?;

    match cuda_ops_bf16::sdpa_stream_bf16(&q, &k, &v, None, 96, false, None) {
        Ok(_) => println!("sdpa_stream_bf16 succeeded for SD3.5-sized probe"),
        Err(err) => println!("sdpa_stream_bf16 shape probe error: {err:?}"),
    }

    Ok(())
}
