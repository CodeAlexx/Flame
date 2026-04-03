use crate::tensor::contracts::assert_nhwc_bf16_public;
use crate::{DType, Result, Tensor};

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
        // d/dx (x * y) = y, d/dy (x * y) = x
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
        // MatMul operates on rank-2 tensors; enforce BF16 at boundary but skip rank guard.
        if grad_output.dtype() != DType::BF16 || lhs.dtype() != DType::BF16 || rhs.dtype() != DType::BF16 {
            return Err(crate::Error::InvalidInput(
                "BackwardOps::matmul_backward expects BF16 tensors".into(),
            ));
        }
        // d/dA (A @ B) = grad @ B^T
        // d/dB (A @ B) = A^T @ grad
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
    
    /// Backward for ReLU activation
    pub fn relu_backward(grad_output: &Tensor, input: &Tensor) -> Result<Tensor> {
        assert_nhwc_bf16_public("BackwardOps::relu_backward in(grad)", grad_output)?;
        assert_nhwc_bf16_public("BackwardOps::relu_backward in(input)", input)?;
        // d/dx ReLU(x) = 1 if x > 0, else 0
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
    
    /// Backward for square operation
    pub fn square_backward(grad_output: &Tensor, input: &Tensor) -> Result<Tensor> {
        assert_nhwc_bf16_public("BackwardOps::square_backward in(grad)", grad_output)?;
        assert_nhwc_bf16_public("BackwardOps::square_backward in(input)", input)?;
        // d/dx (x^2) = 2x
        let two_x = input.mul_scalar(2.0)?;
        let grad = grad_output.mul(&two_x)?;
        assert_nhwc_bf16_public("BackwardOps::square_backward out", &grad)?;
        Ok(grad)
    }
    
    /// Backward for sum operation
    pub fn sum_backward(grad_output: &Tensor, input_shape: &crate::Shape) -> Result<Tensor> {
        assert_nhwc_bf16_public("BackwardOps::sum_backward in", grad_output)?;
        // Gradient of sum: broadcast grad_output to input shape
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
        // d/dx mean(x) = 1/n for each element
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
        // Gradient of transpose works for arbitrary ranks; enforce BF16 only.
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
    
    /// Backward for GELU activation
    pub fn gelu_backward(grad_output: &Tensor, input: &Tensor) -> Result<Tensor> {
        assert_nhwc_bf16_public("BackwardOps::gelu_backward in(grad)", grad_output)?;
        assert_nhwc_bf16_public("BackwardOps::gelu_backward in(input)", input)?;
        // d/dx GELU(x) = 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
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
    
    /// Backward for SiLU activation
    pub fn silu_backward(grad_output: &Tensor, input: &Tensor) -> Result<Tensor> {
        assert_nhwc_bf16_public("BackwardOps::silu_backward in(grad)", grad_output)?;
        assert_nhwc_bf16_public("BackwardOps::silu_backward in(input)", input)?;
        // d/dx SiLU(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
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
    
    /// Backward for Tanh activation
    pub fn tanh_backward(grad_output: &Tensor, input: &Tensor) -> Result<Tensor> {
        assert_nhwc_bf16_public("BackwardOps::tanh_backward in(grad)", grad_output)?;
        assert_nhwc_bf16_public("BackwardOps::tanh_backward in(input)", input)?;
        // d/dx tanh(x) = sech^2(x) = 1 - tanh^2(x)
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
        // 1) logits = (Q K^T) * scale [+ mask]
        let kt = k.transpose_dims(2, 3)?;           // [B,H,D,Sk]
        let mut logits = q.bmm(&kt)?;               // [B,H,Sq,Sk]
        logits = logits.mul_scalar(scale)?;
        if let Some(m) = mask { logits = logits.add(m)?; }

        // 2) attn = softmax(logits)
        let attn = logits.softmax(-1)?;             // [B,H,Sq,Sk]

        // 3) dV = attn^T @ dO
        let attn_t = attn.transpose_dims(2, 3)?;    // [B,H,Sk,Sq]
        let d_v = attn_t.bmm(dout)?;                // [B,H,Sk,D]

        // 4) dAttn = dO @ V^T
        let vt = v.transpose_dims(2, 3)?;           // [B,H,D,Sk]
        let d_attn = dout.bmm(&vt)?;                // [B,H,Sq,Sk]

        // 5) Softmax backward: dLogits = (dAttn - sum(dAttn*attn, -1, keepdim)) * attn
        let dattn_times_attn = d_attn.mul(&attn)?;  // [B,H,Sq,Sk]
        let sum_term = dattn_times_attn.sum_dim_keepdim(3)?; // [B,H,Sq,1]
        let d_logits = d_attn.sub(&sum_term)?.mul(&attn)?;   // [B,H,Sq,Sk]

        // 6) Optional: if caller passes a 0/1 binary mask, multiply to stop grads.
        // For additive -inf masks, softmax already zeros masked probs, so this is usually unnecessary.
        let d_logits = if let Some(_m) = mask { d_logits } else { d_logits };

        // 7) dQ = dLogits @ K
        let d_q = d_logits.bmm(k)?;                 // [B,H,Sq,D]

        // 8) dK = dLogits^T @ Q
        let d_logits_t = d_logits.transpose_dims(2, 3)?; // [B,H,Sk,Sq]
        let d_k = d_logits_t.bmm(q)?;               // [B,H,Sk,D]

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
