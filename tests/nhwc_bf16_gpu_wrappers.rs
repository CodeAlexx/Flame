#[cfg(feature = "cuda")]
mod cuda_only {
    use std::sync::Arc;

    use cudarc::driver::CudaDevice;
    use flame_core::attention::GeGLU;
    use flame_core::pooling::{AvgPool2d, AvgPool2dConfig};
    use flame_core::{DType, Result, Shape, Tensor};

    type Device = Arc<CudaDevice>;

    fn nhwc_bf16(device: &Device, shape: &[usize]) -> Result<Tensor> {
        Tensor::zeros(Shape::from_dims(shape), device.clone())?.to_dtype(DType::BF16)
    }

    #[test]
    fn pooling_and_geglu_accept_bf16() -> Result<()> {
        let device: Device = CudaDevice::new(0)?;

        // NHWC pooling path
        let input = nhwc_bf16(&device, &[1, 8, 8, 32])?;
        let mut cfg = AvgPool2dConfig::new((2, 2));
        cfg.stride = Some((2, 2));
        let pool = AvgPool2d::new(cfg);
        let pooled = pool.forward(&input)?;
        assert_eq!(pooled.dtype(), DType::BF16);
        assert_eq!(pooled.shape().dims(), &[1, 4, 4, 32]);

        // GeGLU expects 2D input; collapse spatial dims and ensure BF16 round-trip
        let geglu_in = input.reshape(&[64, 32])?;
        let geglu = GeGLU::new(32, 16, device.clone())?;
        let geglu_out = geglu.forward(&geglu_in)?;
        assert_eq!(geglu_out.dtype(), DType::BF16);
        assert_eq!(geglu_out.shape().dims(), &[64, 16]);

        Ok(())
    }
}

#[cfg(not(feature = "cuda"))]
#[test]
fn pooling_and_geglu_accept_bf16() {
    eprintln!("skipped: build without cuda feature");
}
