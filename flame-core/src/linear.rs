use crate::{Tensor, Result, FlameError, Shape};
use cudarc::driver::CudaDevice;
use std::sync::Arc;

/// Linear (fully connected) layer
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    /// Create a new linear layer
    pub fn new(in_features: usize, out_features: usize, bias: bool, device: &Arc<CudaDevice>) -> Result<Self> {
        // Initialize weight with Xavier/Glorot uniform
        let bound = (6.0 / (in_features + out_features) as f32).sqrt();
        let weight = Tensor::randn(
            Shape::from_dims(&[out_features, in_features]),
            0.0,
            bound,
            device.clone()
        )?.requires_grad_(true);
        
        let bias = if bias {
            Some(Tensor::zeros(
                Shape::from_dims(&[out_features]),
                device.clone()
            )?.requires_grad_(true))
        } else {
            None
        };
        
        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
        })
    }
    
    /// Get input features
    pub fn in_features(&self) -> usize {
        self.in_features
    }
    
    /// Get output features
    pub fn out_features(&self) -> usize {
        self.out_features
    }
    
    /// Forward pass
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Input shape: [..., in_features]
        // Weight shape: [out_features, in_features]
        // Output shape: [..., out_features]
        
        let input_shape = input.shape().dims();
        if input_shape[input_shape.len() - 1] != self.in_features {
            return Err(FlameError::ShapeMismatch {
                expected: Shape::from_dims(&[self.in_features]),
                got: Shape::from_dims(&[input_shape[input_shape.len() - 1]]),
            });
        }
        
        // Reshape input to 2D for matmul
        let batch_size = input_shape[..input_shape.len() - 1].iter().product::<usize>();
        let input_2d = input.reshape(&[batch_size, self.in_features])?;
        
        // Compute: input @ weight.T + bias
        let weight_t = self.weight.transpose()?;
        let mut output = input_2d.matmul(&weight_t)?;
        
        // Add bias if present
        if let Some(bias) = &self.bias {
            output = output.add_bias(bias)?;
        }
        
        // Reshape back to original batch dimensions
        let mut output_shape = input_shape[..input_shape.len() - 1].to_vec();
        output_shape.push(self.out_features);
        output.reshape(&output_shape)
    }
    
    /// Get trainable parameters
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(bias) = &self.bias {
            params.push(bias);
        }
        params
    }
}