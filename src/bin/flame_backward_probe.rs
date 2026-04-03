use anyhow::Result;
use flame_core::device::Device as FlameDevice;
use flame_core::{DType, Shape, Tensor};
use log::info;

fn main() -> Result<()> {
    let _ = env_logger::builder().is_test(false).try_init();
    let device = FlameDevice::cuda(0)?;
    let cuda = device.cuda_device_arc();

    let data: Vec<f32> = (0..4).map(|v| v as f32 / 100.0).collect();
    let tensor =
        Tensor::from_vec_dtype(data, Shape::from_dims(&[2, 2]), cuda.clone(), DType::BF16)?
            .requires_grad_(true);

    let loss = tensor.square()?.mean()?.requires_grad_(true);
    device.synchronize()?;
    info!("[flame_backward_probe] entering backward");
    let gradients = loss.backward()?;
    info!("[flame_backward_probe] backward returned");
    device.synchronize()?;

    let loss_value = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
    let grad_count = gradients.iter_fp32()?.count();
    info!("[flame_backward_probe] loss={loss_value} gradients={grad_count}");
    Ok(())
}
