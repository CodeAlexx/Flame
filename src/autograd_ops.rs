use crate::tensor::contracts::assert_nhwc_bf16_public;
use crate::{DType, Error, Result, Tensor, TensorStorage};

#[cfg(feature = "cuda")]
fn tensor_raw_ptr(t: &Tensor) -> Result<*const core::ffi::c_void> {
    use cudarc::driver::DevicePtr;
    match &t.storage {
        TensorStorage::F32 { data, .. } => Ok(*data.device_ptr() as *const core::ffi::c_void),
        TensorStorage::F16 { data, .. } => Ok(*data.device_ptr() as *const core::ffi::c_void),
        #[cfg(feature = "bf16_u16")]
        TensorStorage::BF16 { data, .. } => Ok(*data.device_ptr() as *const core::ffi::c_void),
        #[cfg(feature = "bf16_u16")]
        TensorStorage::BF16Arena { ptr, .. } => Ok(ptr.as_ptr() as *const core::ffi::c_void),
        #[cfg(feature = "bf16_u16")]
        TensorStorage::BF16View { ptr, .. } => Ok(ptr.as_ptr() as *const core::ffi::c_void),
        TensorStorage::I32 { data, .. } => Ok(*data.device_ptr() as *const core::ffi::c_void),
        TensorStorage::Bool { data, .. } => Ok(*data.device_ptr() as *const core::ffi::c_void),
        _ => Err(Error::InvalidOperation("unsupported dtype for raw ptr".into())),
    }
}

#[cfg(feature = "cuda")]
fn tensor_raw_ptr_mut(t: &mut Tensor) -> Result<*mut core::ffi::c_void> {
    use cudarc::driver::DevicePtrMut;
    match &mut t.storage {
        TensorStorage::F32 { data, .. } => {
            let slice = crate::tensor_storage::ensure_unique_slice(data)?;
            Ok(*slice.device_ptr_mut() as *mut core::ffi::c_void)
        }
        #[cfg(feature = "bf16_u16")]
        TensorStorage::BF16 { data, .. } => {
            let slice = crate::tensor_storage::ensure_unique_slice(data)?;
            Ok(*slice.device_ptr_mut() as *mut core::ffi::c_void)
        }
        _ => Err(Error::InvalidOperation("unsupported dtype for raw ptr mut".into())),
    }
}

#[cfg(feature = "cuda")]
type UnaryBackwardKernel = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    i64,
    *mut core::ffi::c_void,
) -> i32;

/// Dispatch a fused unary-activation backward CUDA kernel.
///
/// BF16→BF16 is the fast path. Anything else is cast to F32 and served by the
/// F32 kernel (then the caller is responsible for any downstream dtype match).
#[cfg(feature = "cuda")]
fn launch_unary_backward(
    op_name: &str,
    grad_output: &Tensor,
    input: &Tensor,
    bf16_kernel: UnaryBackwardKernel,
    f32_kernel: UnaryBackwardKernel,
) -> Result<Tensor> {
    use crate::device::CudaStreamRawPtrExt;
    let n = input.shape().elem_count() as i64;
    if n == 0 {
        // Zero-element: return a dtype-matching empty tensor without a launch.
        return Tensor::zeros_dtype(input.shape().clone(), input.dtype(), input.device().clone());
    }
    let device = input.device().clone();
    let stream = device.cuda_stream_raw_ptr();

    if grad_output.dtype() == DType::BF16 && input.dtype() == DType::BF16 {
        let mut out = Tensor::zeros_dtype(input.shape().clone(), DType::BF16, device)?;
        let status = unsafe {
            bf16_kernel(
                tensor_raw_ptr(grad_output)?,
                tensor_raw_ptr(input)?,
                tensor_raw_ptr_mut(&mut out)?,
                n,
                stream,
            )
        };
        if status != 0 {
            return Err(Error::Cuda(format!("{op_name} bf16 kernel failed")));
        }
        return Ok(out);
    }

    let og_f32 = if grad_output.dtype() != DType::F32 {
        grad_output.to_dtype_no_grad(DType::F32)?
    } else {
        grad_output.clone_result()?
    };
    let x_f32 = if input.dtype() != DType::F32 {
        input.to_dtype_no_grad(DType::F32)?
    } else {
        input.clone_result()?
    };
    let mut out = Tensor::zeros_dtype(x_f32.shape().clone(), DType::F32, input.device().clone())?;
    let status = unsafe {
        f32_kernel(
            tensor_raw_ptr(&og_f32)?,
            tensor_raw_ptr(&x_f32)?,
            tensor_raw_ptr_mut(&mut out)?,
            n,
            stream,
        )
    };
    if status != 0 {
        return Err(Error::Cuda(format!("{op_name} f32 kernel failed")));
    }
    Ok(out)
}

/// Backward operations for autograd
pub struct BackwardOps;

impl BackwardOps {
    /// Backward for addition: grad flows unchanged to both inputs
    pub fn add_backward(grad_output: &Tensor) -> Result<(Tensor, Tensor)> {
        assert_nhwc_bf16_public("BackwardOps::add_backward in", grad_output)?;
        let grad_lhs = grad_output.clone_result()?;
        let grad_rhs = grad_output.clone_result()?;
        assert_nhwc_bf16_public("BackwardOps::add_backward out(lhs)", &grad_lhs)?;
        assert_nhwc_bf16_public("BackwardOps::add_backward out(rhs)", &grad_rhs)?;
        Ok((grad_lhs, grad_rhs))
    }

    /// Backward for subtraction: grad flows to lhs, -grad to rhs
    pub fn sub_backward(grad_output: &Tensor) -> Result<(Tensor, Tensor)> {
        assert_nhwc_bf16_public("BackwardOps::sub_backward in", grad_output)?;
        let grad_lhs = grad_output.clone_result()?;
        let grad_rhs = grad_output.mul_scalar(-1.0)?;
        assert_nhwc_bf16_public("BackwardOps::sub_backward out(lhs)", &grad_lhs)?;
        assert_nhwc_bf16_public("BackwardOps::sub_backward out(rhs)", &grad_rhs)?;
        Ok((grad_lhs, grad_rhs))
    }

    /// Backward for element-wise multiplication
    pub fn mul_backward(grad_output: &Tensor, lhs: &Tensor, rhs: &Tensor) -> Result<(Tensor, Tensor)> {
        assert_nhwc_bf16_public("BackwardOps::mul_backward in(grad)", grad_output)?;
        assert_nhwc_bf16_public("BackwardOps::mul_backward in(lhs)", lhs)?;
        assert_nhwc_bf16_public("BackwardOps::mul_backward in(rhs)", rhs)?;
        let grad_lhs = grad_output.mul(rhs)?;
        let grad_rhs = grad_output.mul(lhs)?;
        assert_nhwc_bf16_public("BackwardOps::mul_backward out(lhs)", &grad_lhs)?;
        assert_nhwc_bf16_public("BackwardOps::mul_backward out(rhs)", &grad_rhs)?;
        Ok((grad_lhs, grad_rhs))
    }

    /// Backward for scalar multiplication
    pub fn mul_scalar_backward(grad_output: &Tensor, scalar: f32) -> Result<Tensor> {
        assert_nhwc_bf16_public("BackwardOps::mul_scalar_backward in", grad_output)?;
        let grad = grad_output.mul_scalar(scalar)?;
        assert_nhwc_bf16_public("BackwardOps::mul_scalar_backward out", &grad)?;
        Ok(grad)
    }

    /// Backward for matrix multiplication
    pub fn matmul_backward(grad_output: &Tensor, lhs: &Tensor, rhs: &Tensor) -> Result<(Tensor, Tensor)> {
        if grad_output.dtype() != DType::BF16 || lhs.dtype() != DType::BF16 || rhs.dtype() != DType::BF16 {
            return Err(crate::Error::InvalidInput(
                "BackwardOps::matmul_backward expects BF16 tensors".into(),
            ));
        }
        let rhs_t = rhs.transpose()?;
        let grad_lhs = grad_output.matmul(&rhs_t)?;

        let lhs_t = lhs.transpose()?;
        let grad_rhs = lhs_t.matmul(grad_output)?;

        if grad_lhs.dtype() != DType::BF16 {
            let grad_lhs = grad_lhs.to_dtype(DType::BF16)?;
            let grad_rhs = if grad_rhs.dtype() != DType::BF16 {
                grad_rhs.to_dtype(DType::BF16)?
            } else {
                grad_rhs
            };
            return Ok((grad_lhs, grad_rhs));
        }
        if grad_rhs.dtype() != DType::BF16 {
            let grad_rhs = grad_rhs.to_dtype(DType::BF16)?;
            return Ok((grad_lhs, grad_rhs));
        }

        Ok((grad_lhs, grad_rhs))
    }

    /// Backward for ReLU activation. d/dx ReLU(x) = 1 if x > 0 else 0.
    /// CUDA path uses the fused `flame_relu_backward_*` kernel; non-CUDA builds
    /// fall back to a host loop.
    pub fn relu_backward(grad_output: &Tensor, input: &Tensor) -> Result<Tensor> {
        assert_nhwc_bf16_public("BackwardOps::relu_backward in(grad)", grad_output)?;
        assert_nhwc_bf16_public("BackwardOps::relu_backward in(input)", input)?;

        #[cfg(feature = "cuda")]
        {
            let grad = launch_unary_backward(
                "relu_backward",
                grad_output,
                input,
                crate::cuda::ffi::flame_relu_backward_bf16,
                crate::cuda::ffi::flame_relu_backward_f32,
            )?;
            assert_nhwc_bf16_public("BackwardOps::relu_backward out", &grad)?;
            return Ok(grad);
        }

        #[cfg(not(feature = "cuda"))]
        {
            let input_data = input.to_vec()?;
            let grad_data = grad_output.to_vec()?;
            let mut result = vec![0.0f32; grad_data.len()];
            for i in 0..result.len() {
                result[i] = if input_data[i] > 0.0 { grad_data[i] } else { 0.0 };
            }
            let grad = Tensor::from_vec_dtype(
                result,
                grad_output.shape().clone(),
                grad_output.device.clone(),
                DType::BF16,
            )?;
            assert_nhwc_bf16_public("BackwardOps::relu_backward out", &grad)?;
            Ok(grad)
        }
    }

    /// Backward for square operation
    pub fn square_backward(grad_output: &Tensor, input: &Tensor) -> Result<Tensor> {
        assert_nhwc_bf16_public("BackwardOps::square_backward in(grad)", grad_output)?;
        assert_nhwc_bf16_public("BackwardOps::square_backward in(input)", input)?;
        let two_x = input.mul_scalar(2.0)?;
        let grad = grad_output.mul(&two_x)?;
        assert_nhwc_bf16_public("BackwardOps::square_backward out", &grad)?;
        Ok(grad)
    }

    /// Backward for sum operation
    pub fn sum_backward(grad_output: &Tensor, input_shape: &crate::Shape) -> Result<Tensor> {
        assert_nhwc_bf16_public("BackwardOps::sum_backward in", grad_output)?;
        let grad_val = grad_output.to_vec()?[0];
        let mut grad = Tensor::ones_dtype(
            input_shape.clone(),
            DType::BF16,
            grad_output.device.clone(),
        )?;
        grad = grad.mul_scalar(grad_val)?;
        if grad.dtype() != DType::BF16 {
            grad = grad.to_dtype(DType::BF16)?;
        }
        assert_nhwc_bf16_public("BackwardOps::sum_backward out", &grad)?;
        Ok(grad)
    }

    /// Backward for mean operation
    pub fn mean_backward(grad_output: &Tensor, input_shape: &crate::Shape) -> Result<Tensor> {
        assert_nhwc_bf16_public("BackwardOps::mean_backward in", grad_output)?;
        let n = input_shape.elem_count() as f32;
        let grad_val = grad_output.to_vec()?[0] / n;

        let mut grad = Tensor::ones_dtype(
            input_shape.clone(),
            DType::BF16,
            grad_output.device.clone(),
        )?;
        grad = grad.mul_scalar(grad_val)?;
        if grad.dtype() != DType::BF16 {
            grad = grad.to_dtype(DType::BF16)?;
        }
        assert_nhwc_bf16_public("BackwardOps::mean_backward out", &grad)?;
        Ok(grad)
    }

    /// Backward for transpose operation
    pub fn transpose_backward(grad_output: &Tensor) -> Result<Tensor> {
        if grad_output.dtype() != DType::BF16 {
            return Err(crate::Error::InvalidInput(
                "BackwardOps::transpose_backward expects BF16 tensor".into(),
            ));
        }
        let grad = grad_output.transpose()?;
        if grad.rank() == 4 {
            assert_nhwc_bf16_public("BackwardOps::transpose_backward out", &grad)?;
        }
        Ok(grad)
    }

    /// Backward for GELU activation (tanh-approx).
    /// CUDA path uses the fused `flame_gelu_backward_*` kernel.
    pub fn gelu_backward(grad_output: &Tensor, input: &Tensor) -> Result<Tensor> {
        assert_nhwc_bf16_public("BackwardOps::gelu_backward in(grad)", grad_output)?;
        assert_nhwc_bf16_public("BackwardOps::gelu_backward in(input)", input)?;

        #[cfg(feature = "cuda")]
        {
            let grad = launch_unary_backward(
                "gelu_backward",
                grad_output,
                input,
                crate::cuda::ffi::flame_gelu_backward_bf16,
                crate::cuda::ffi::flame_gelu_backward_f32,
            )?;
            assert_nhwc_bf16_public("BackwardOps::gelu_backward out", &grad)?;
            return Ok(grad);
        }

        #[cfg(not(feature = "cuda"))]
        {
            let input_data = input.to_vec()?;
            let grad_data = grad_output.to_vec()?;
            let mut result = vec![0.0f32; grad_data.len()];
            const SQRT_2_OVER_PI: f32 = 0.7978845608;
            const C: f32 = 0.044715;
            for i in 0..result.len() {
                let x = input_data[i];
                let x_sq = x * x;
                let x_cubed = x * x_sq;
                let tanh_arg = SQRT_2_OVER_PI * (x + C * x_cubed);
                let tanh_val = tanh_arg.tanh();
                let sech_sq = 1.0 - tanh_val * tanh_val;
                let grad = 0.5 * (1.0 + tanh_val)
                    + 0.5 * x * sech_sq * SQRT_2_OVER_PI * (1.0 + 3.0 * C * x_sq);
                result[i] = grad_data[i] * grad;
            }
            let grad = Tensor::from_vec_dtype(
                result,
                grad_output.shape().clone(),
                grad_output.device().clone(),
                DType::BF16,
            )?;
            assert_nhwc_bf16_public("BackwardOps::gelu_backward out", &grad)?;
            Ok(grad)
        }
    }

    /// Backward for SiLU activation. d/dx SiLU(x) = sig(x) + x*sig(x)*(1 - sig(x)).
    /// CUDA path uses the fused `flame_silu_backward_*` kernel.
    pub fn silu_backward(grad_output: &Tensor, input: &Tensor) -> Result<Tensor> {
        assert_nhwc_bf16_public("BackwardOps::silu_backward in(grad)", grad_output)?;
        assert_nhwc_bf16_public("BackwardOps::silu_backward in(input)", input)?;

        #[cfg(feature = "cuda")]
        {
            let grad = launch_unary_backward(
                "silu_backward",
                grad_output,
                input,
                crate::cuda::ffi::flame_silu_backward_bf16,
                crate::cuda::ffi::flame_silu_backward_f32,
            )?;
            assert_nhwc_bf16_public("BackwardOps::silu_backward out", &grad)?;
            return Ok(grad);
        }

        #[cfg(not(feature = "cuda"))]
        {
            let input_data = input.to_vec()?;
            let grad_data = grad_output.to_vec()?;
            let mut result = vec![0.0f32; grad_data.len()];
            for i in 0..result.len() {
                let x = input_data[i];
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                let grad = sigmoid * (1.0 + x * (1.0 - sigmoid));
                result[i] = grad_data[i] * grad;
            }
            let grad = Tensor::from_vec_dtype(
                result,
                grad_output.shape().clone(),
                grad_output.device().clone(),
                DType::BF16,
            )?;
            assert_nhwc_bf16_public("BackwardOps::silu_backward out", &grad)?;
            Ok(grad)
        }
    }

    /// Backward for Tanh activation. d/dx tanh(x) = 1 - tanh^2(x).
    /// CUDA path uses the fused `flame_tanh_backward_*` kernel.
    pub fn tanh_backward(grad_output: &Tensor, input: &Tensor) -> Result<Tensor> {
        assert_nhwc_bf16_public("BackwardOps::tanh_backward in(grad)", grad_output)?;
        assert_nhwc_bf16_public("BackwardOps::tanh_backward in(input)", input)?;

        #[cfg(feature = "cuda")]
        {
            let grad = launch_unary_backward(
                "tanh_backward",
                grad_output,
                input,
                crate::cuda::ffi::flame_tanh_backward_bf16,
                crate::cuda::ffi::flame_tanh_backward_f32,
            )?;
            assert_nhwc_bf16_public("BackwardOps::tanh_backward out", &grad)?;
            return Ok(grad);
        }

        #[cfg(not(feature = "cuda"))]
        {
            let input_data = input.to_vec()?;
            let grad_data = grad_output.to_vec()?;
            let mut result = vec![0.0f32; grad_data.len()];
            for i in 0..result.len() {
                let tanh_val = input_data[i].tanh();
                let grad = 1.0 - tanh_val * tanh_val;
                result[i] = grad_data[i] * grad;
            }
            let grad = Tensor::from_vec_dtype(
                result,
                grad_output.shape().clone(),
                grad_output.device().clone(),
                DType::BF16,
            )?;
            assert_nhwc_bf16_public("BackwardOps::tanh_backward out", &grad)?;
            Ok(grad)
        }
    }

    /// Backward for Sigmoid activation. d/dx sig(x) = sig(x) * (1 - sig(x)).
    /// CUDA path uses the fused `flame_sigmoid_backward_*` kernel.
    pub fn sigmoid_backward(grad_output: &Tensor, input: &Tensor) -> Result<Tensor> {
        assert_nhwc_bf16_public("BackwardOps::sigmoid_backward in(grad)", grad_output)?;
        assert_nhwc_bf16_public("BackwardOps::sigmoid_backward in(input)", input)?;

        #[cfg(feature = "cuda")]
        {
            let grad = launch_unary_backward(
                "sigmoid_backward",
                grad_output,
                input,
                crate::cuda::ffi::flame_sigmoid_backward_bf16,
                crate::cuda::ffi::flame_sigmoid_backward_f32,
            )?;
            assert_nhwc_bf16_public("BackwardOps::sigmoid_backward out", &grad)?;
            return Ok(grad);
        }

        #[cfg(not(feature = "cuda"))]
        {
            let input_data = input.to_vec()?;
            let grad_data = grad_output.to_vec()?;
            let mut result = vec![0.0f32; grad_data.len()];
            for i in 0..result.len() {
                let x = input_data[i];
                let s = 1.0 / (1.0 + (-x).exp());
                result[i] = grad_data[i] * s * (1.0 - s);
            }
            let grad = Tensor::from_vec_dtype(
                result,
                grad_output.shape().clone(),
                grad_output.device().clone(),
                DType::BF16,
            )?;
            assert_nhwc_bf16_public("BackwardOps::sigmoid_backward out", &grad)?;
            Ok(grad)
        }
    }

    /// Scaled dot-product attention backward via recomputation path.
    /// Shapes follow [B, H, S, D] convention for Q,K,V and [B, H, S_q, D] for dO.
    /// mask is an optional additive mask broadcastable to [B,H,S_q,S_k].
    pub fn attention_backward_recompute(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        dout: &Tensor,
        mask: Option<&Tensor>,
        scale: f32,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let kt = k.transpose_dims(2, 3)?;
        let mut logits = q.bmm(&kt)?;
        logits = logits.mul_scalar(scale)?;
        if let Some(m) = mask { logits = logits.add(m)?; }

        let attn = logits.softmax(-1)?;

        let attn_t = attn.transpose_dims(2, 3)?;
        let d_v = attn_t.bmm(dout)?;

        let vt = v.transpose_dims(2, 3)?;
        let d_attn = dout.bmm(&vt)?;

        let dattn_times_attn = d_attn.mul(&attn)?;
        let sum_term = dattn_times_attn.sum_dim_keepdim(3)?;
        let d_logits = d_attn.sub(&sum_term)?.mul(&attn)?;

        let d_logits = if let Some(_m) = mask { d_logits } else { d_logits };

        let d_q = d_logits.bmm(k)?;

        let d_logits_t = d_logits.transpose_dims(2, 3)?;
        let d_k = d_logits_t.bmm(q)?;

        Ok((d_q, d_k, d_v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CudaDevice, Shape};

    #[test]
    fn test_add_backward() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let grad_output = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[2, 2]), device)?;

        let (grad_lhs, grad_rhs) = BackwardOps::add_backward(&grad_output)?;

        assert_eq!(grad_lhs.to_vec()?, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(grad_rhs.to_vec()?, vec![1.0, 2.0, 3.0, 4.0]);

        Ok(())
    }

    #[test]
    fn test_relu_backward() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let input = Tensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0], Shape::from_dims(&[4]), device.clone())?;
        let grad_output = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], Shape::from_dims(&[4]), device)?;

        let grad_input = BackwardOps::relu_backward(&grad_output, &input)?;

        assert_eq!(grad_input.to_vec()?, vec![0.0, 1.0, 0.0, 1.0]);

        Ok(())
    }
}
