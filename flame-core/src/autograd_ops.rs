use crate::{Tensor, Result};

/// Backward operations for autograd
pub struct BackwardOps;

impl BackwardOps {
    /// Backward for addition: grad flows unchanged to both inputs
    pub fn add_backward(grad_output: &Tensor) -> Result<(Tensor, Tensor)> {
        let grad_lhs = grad_output.clone()?;
        let grad_rhs = grad_output.clone()?;
        Ok((grad_lhs, grad_rhs))
    }
    
    /// Backward for subtraction: grad flows to lhs, -grad to rhs
    pub fn sub_backward(grad_output: &Tensor) -> Result<(Tensor, Tensor)> {
        let grad_lhs = grad_output.clone()?;
        let grad_rhs = grad_output.mul_scalar(-1.0)?;
        Ok((grad_lhs, grad_rhs))
    }
    
    /// Backward for element-wise multiplication
    pub fn mul_backward(grad_output: &Tensor, lhs: &Tensor, rhs: &Tensor) -> Result<(Tensor, Tensor)> {
        // d/dx (x * y) = y, d/dy (x * y) = x
        let grad_lhs = grad_output.mul(rhs)?;
        let grad_rhs = grad_output.mul(lhs)?;
        Ok((grad_lhs, grad_rhs))
    }
    
    /// Backward for scalar multiplication
    pub fn mul_scalar_backward(grad_output: &Tensor, scalar: f32) -> Result<Tensor> {
        grad_output.mul_scalar(scalar)
    }
    
    /// Backward for matrix multiplication
    pub fn matmul_backward(grad_output: &Tensor, lhs: &Tensor, rhs: &Tensor) -> Result<(Tensor, Tensor)> {
        // d/dA (A @ B) = grad @ B^T
        // d/dB (A @ B) = A^T @ grad
        let rhs_t = rhs.transpose()?;
        let grad_lhs = grad_output.matmul(&rhs_t)?;
        
        let lhs_t = lhs.transpose()?;
        let grad_rhs = lhs_t.matmul(grad_output)?;
        
        Ok((grad_lhs, grad_rhs))
    }
    
    /// Backward for ReLU activation
    pub fn relu_backward(grad_output: &Tensor, input: &Tensor) -> Result<Tensor> {
        // d/dx ReLU(x) = 1 if x > 0, else 0
        let input_data = input.to_vec()?;
        let grad_data = grad_output.to_vec()?;
        
        let mut result = vec![0.0f32; grad_data.len()];
        for i in 0..result.len() {
            result[i] = if input_data[i] > 0.0 { grad_data[i] } else { 0.0 };
        }
        
        Tensor::from_vec(result, grad_output.shape().clone(), grad_output.device.clone())
    }
    
    /// Backward for square operation
    pub fn square_backward(grad_output: &Tensor, input: &Tensor) -> Result<Tensor> {
        // d/dx (x^2) = 2x
        let two_x = input.mul_scalar(2.0)?;
        grad_output.mul(&two_x)
    }
    
    /// Backward for sum operation
    pub fn sum_backward(grad_output: &Tensor, input_shape: &crate::Shape) -> Result<Tensor> {
        // Gradient of sum: broadcast grad_output to input shape
        let grad_val = grad_output.to_vec()?[0];
        Tensor::from_vec(
            vec![grad_val; input_shape.elem_count()],
            input_shape.clone(),
            grad_output.device.clone()
        )
    }
    
    /// Backward for mean operation
    pub fn mean_backward(grad_output: &Tensor, input_shape: &crate::Shape) -> Result<Tensor> {
        // d/dx mean(x) = 1/n for each element
        let n = input_shape.elem_count() as f32;
        let grad_val = grad_output.to_vec()?[0] / n;
        
        Tensor::from_vec(
            vec![grad_val; input_shape.elem_count()],
            input_shape.clone(),
            grad_output.device.clone()
        )
    }
    
    /// Backward for transpose operation
    pub fn transpose_backward(grad_output: &Tensor) -> Result<Tensor> {
        // Gradient of transpose is transpose of gradient
        grad_output.transpose()
    }
    
    /// Backward for GELU activation
    pub fn gelu_backward(grad_output: &Tensor, input: &Tensor) -> Result<Tensor> {
        // d/dx GELU(x) = 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) 
        //              + 0.5 * x * sech^2(sqrt(2/pi) * (x + 0.044715 * x^3)) * sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
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
        
        Tensor::from_vec(result, grad_output.shape().clone(), grad_output.device().clone())
    }
    
    /// Backward for SiLU activation
    pub fn silu_backward(grad_output: &Tensor, input: &Tensor) -> Result<Tensor> {
        // d/dx SiLU(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        //               = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        let input_data = input.to_vec()?;
        let grad_data = grad_output.to_vec()?;
        
        let mut result = vec![0.0f32; grad_data.len()];
        for i in 0..result.len() {
            let x = input_data[i];
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            let grad = sigmoid * (1.0 + x * (1.0 - sigmoid));
            result[i] = grad_data[i] * grad;
        }
        
        Tensor::from_vec(result, grad_output.shape().clone(), grad_output.device().clone())
    }
    
    /// Backward for Tanh activation
    pub fn tanh_backward(grad_output: &Tensor, input: &Tensor) -> Result<Tensor> {
        // d/dx tanh(x) = sech^2(x) = 1 - tanh^2(x)
        let input_data = input.to_vec()?;
        let grad_data = grad_output.to_vec()?;
        
        let mut result = vec![0.0f32; grad_data.len()];
        for i in 0..result.len() {
            let tanh_val = input_data[i].tanh();
            let grad = 1.0 - tanh_val * tanh_val;
            result[i] = grad_data[i] * grad;
        }
        
        Tensor::from_vec(result, grad_output.shape().clone(), grad_output.device().clone())
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