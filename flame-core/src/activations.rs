use crate::{Result, Tensor, Shape, CudaDevice};
use std::sync::Arc;

/// LeakyReLU activation function
/// output = max(0, x) + negative_slope * min(0, x)
pub struct LeakyReLU {
    pub negative_slope: f32,
}

impl LeakyReLU {
    pub fn new(negative_slope: f32) -> Self {
        Self { negative_slope }
    }
    
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        crate::cuda_ops::GpuOps::leaky_relu(input, self.negative_slope)
    }
}

/// ELU activation function (Exponential Linear Unit)
/// output = max(0, x) + min(0, alpha * (exp(x) - 1))
pub struct ELU {
    pub alpha: f32,
}

impl ELU {
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }
    
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        crate::cuda_ops::GpuOps::elu(input, self.alpha)
    }
}

/// PReLU activation function (Parametric ReLU)
/// output = max(0, x) + weight * min(0, x)
/// where weight is learnable per channel
pub struct PReLU {
    pub num_parameters: usize,
    pub weight: Tensor,
}

impl PReLU {
    /// Create a new PReLU layer
    /// num_parameters: number of learnable parameters (typically number of channels)
    pub fn new(num_parameters: usize, device: Arc<CudaDevice>) -> Result<Self> {
        // Initialize weights with 0.25 (common default)
        let weight = Tensor::from_vec(
            vec![0.25f32; num_parameters],
            Shape::from_dims(&[num_parameters]),
            device,
        )?;
        
        Ok(Self {
            num_parameters,
            weight,
        })
    }
    
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        crate::cuda_ops::GpuOps::prelu(input, &self.weight)
    }
}

/// ReLU activation function
/// output = max(0, x)
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        Self
    }
    
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        crate::cuda_ops::GpuOps::relu(input)
    }
}

/// GELU activation function (Gaussian Error Linear Unit)
/// output = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub struct GELU;

impl GELU {
    pub fn new() -> Self {
        Self
    }
    
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        crate::cuda_ops::GpuOps::gelu(input)
    }
}

/// SiLU activation function (Sigmoid Linear Unit, also known as Swish)
/// output = x * sigmoid(x)
pub struct SiLU;

impl SiLU {
    pub fn new() -> Self {
        Self
    }
    
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        crate::cuda_ops::GpuOps::silu(input)
    }
}

/// Tanh activation function
/// output = tanh(x)
pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Self
    }
    
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        crate::cuda_ops::GpuOps::tanh(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_activations() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let input = Tensor::randn(Shape::from_dims(&[2, 3, 4, 4]), 0.0, 1.0, device.clone())?;
        
        // Test LeakyReLU
        let leaky_relu = LeakyReLU::new(0.01);
        let output = leaky_relu.forward(&input)?;
        assert_eq!(output.shape().dims(), input.shape().dims());
        
        // Test ELU
        let elu = ELU::new(1.0);
        let output = elu.forward(&input)?;
        assert_eq!(output.shape().dims(), input.shape().dims());
        
        // Test PReLU
        let prelu = PReLU::new(3, device.clone())?;
        let output = prelu.forward(&input)?;
        assert_eq!(output.shape().dims(), input.shape().dims());
        
        // Test ReLU
        let relu = ReLU::new();
        let output = relu.forward(&input)?;
        assert_eq!(output.shape().dims(), input.shape().dims());
        
        // Test GELU
        let gelu = GELU::new();
        let output = gelu.forward(&input)?;
        assert_eq!(output.shape().dims(), input.shape().dims());
        
        // Test SiLU
        let silu = SiLU::new();
        let output = silu.forward(&input)?;
        assert_eq!(output.shape().dims(), input.shape().dims());
        
        // Test Tanh
        let tanh = Tanh::new();
        let output = tanh.forward(&input)?;
        assert_eq!(output.shape().dims(), input.shape().dims());
        
        Ok(())
    }
}