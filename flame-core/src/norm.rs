use crate::{Tensor, Shape, Result, FlameError};
use std::sync::Arc;

/// Batch Normalization layer
pub struct BatchNorm2d {
    pub num_features: usize,
    pub eps: f32,
    pub momentum: f32,
    pub affine: bool,
    pub track_running_stats: bool,
    
    // Learnable parameters
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
    
    // Running statistics
    pub running_mean: Option<Tensor>,
    pub running_var: Option<Tensor>,
    pub num_batches_tracked: usize,
}

impl BatchNorm2d {
    /// Create a new BatchNorm2d layer
    pub fn new(
        num_features: usize,
        eps: f32,
        momentum: f32,
        affine: bool,
        track_running_stats: bool,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let (weight, bias) = if affine {
            let weight = Tensor::from_vec(
                vec![1.0f32; num_features],
                Shape::from_dims(&[num_features]),
                device.clone(),
            )?;
            let bias = Tensor::zeros(Shape::from_dims(&[num_features]), device.clone())?;
            (Some(weight), Some(bias))
        } else {
            (None, None)
        };
        
        let (running_mean, running_var) = if track_running_stats {
            let running_mean = Tensor::zeros(Shape::from_dims(&[num_features]), device.clone())?;
            let running_var = Tensor::from_vec(
                vec![1.0f32; num_features],
                Shape::from_dims(&[num_features]),
                device.clone(),
            )?;
            (Some(running_mean), Some(running_var))
        } else {
            (None, None)
        };
        
        Ok(Self {
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            weight,
            bias,
            running_mean,
            running_var,
            num_batches_tracked: 0,
        })
    }
    
    /// Forward pass for BatchNorm2d
    pub fn forward(&mut self, input: &Tensor, training: bool) -> Result<Tensor> {
        // Validate input shape: [N, C, H, W]
        let dims = input.shape().dims();
        if dims.len() != 4 {
            return Err(FlameError::InvalidOperation(
                format!("BatchNorm2d expects 4D input [N,C,H,W], got {:?}", dims)
            ));
        }
        
        let num_channels = dims[1];
        if num_channels != self.num_features {
            return Err(FlameError::InvalidOperation(
                format!("Expected {} channels, got {}", self.num_features, num_channels)
            ));
        }
        
        let (mean, var) = if training || !self.track_running_stats {
            // Calculate batch statistics
            self.calculate_batch_stats(input)?
        } else {
            // Use running statistics
            match (&self.running_mean, &self.running_var) {
                (Some(mean), Some(var)) => {
                // Create copies of the tensors
                let mean_data = mean.to_vec()?;
                let var_data = var.to_vec()?;
                let mean_copy = Tensor::from_vec(mean_data, mean.shape().clone(), mean.device.clone())?;
                let var_copy = Tensor::from_vec(var_data, var.shape().clone(), var.device.clone())?;
                (mean_copy, var_copy)
            },
                _ => return Err(FlameError::InvalidOperation(
                    "Running stats not available for evaluation mode".into()
                )),
            }
        };
        
        // Update running statistics if training
        if training && self.track_running_stats {
            self.update_running_stats(&mean, &var)?;
        }
        
        // Normalize
        let normalized = self.normalize(input, &mean, &var)?;
        
        // Apply affine transformation if enabled
        if self.affine {
            self.apply_affine(&normalized)
        } else {
            Ok(normalized)
        }
    }
    
    /// Calculate batch mean and variance
    fn calculate_batch_stats(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        let dims = input.shape().dims();
        let batch_size = dims[0];
        let num_channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        let spatial_size = height * width;
        
        let input_data = input.to_vec()?;
        let mut mean = vec![0.0f32; num_channels];
        let mut var = vec![0.0f32; num_channels];
        
        // Calculate mean
        for n in 0..batch_size {
            for c in 0..num_channels {
                let mut sum = 0.0f32;
                for h in 0..height {
                    for w in 0..width {
                        let idx = n * (num_channels * height * width) 
                                + c * (height * width) 
                                + h * width 
                                + w;
                        sum += input_data[idx];
                    }
                }
                mean[c] += sum / (batch_size * spatial_size) as f32;
            }
        }
        
        // Calculate variance
        for n in 0..batch_size {
            for c in 0..num_channels {
                let channel_mean = mean[c];
                let mut sum_sq = 0.0f32;
                for h in 0..height {
                    for w in 0..width {
                        let idx = n * (num_channels * height * width) 
                                + c * (height * width) 
                                + h * width 
                                + w;
                        let diff = input_data[idx] - channel_mean;
                        sum_sq += diff * diff;
                    }
                }
                var[c] += sum_sq / (batch_size * spatial_size) as f32;
            }
        }
        
        let mean_tensor = Tensor::from_vec(mean, Shape::from_dims(&[num_channels]), input.device.clone())?;
        let var_tensor = Tensor::from_vec(var, Shape::from_dims(&[num_channels]), input.device.clone())?;
        
        Ok((mean_tensor, var_tensor))
    }
    
    /// Update running statistics
    fn update_running_stats(&mut self, batch_mean: &Tensor, batch_var: &Tensor) -> Result<()> {
        if let (Some(running_mean), Some(running_var)) = (&mut self.running_mean, &mut self.running_var) {
            let batch_mean_data = batch_mean.to_vec()?;
            let batch_var_data = batch_var.to_vec()?;
            let mut running_mean_data = running_mean.to_vec()?;
            let mut running_var_data = running_var.to_vec()?;
            
            let momentum = self.momentum;
            let one_minus_momentum = 1.0 - momentum;
            
            for i in 0..self.num_features {
                running_mean_data[i] = one_minus_momentum * running_mean_data[i] + momentum * batch_mean_data[i];
                running_var_data[i] = one_minus_momentum * running_var_data[i] + momentum * batch_var_data[i];
            }
            
            *running_mean = Tensor::from_vec(running_mean_data, running_mean.shape().clone(), running_mean.device.clone())?;
            *running_var = Tensor::from_vec(running_var_data, running_var.shape().clone(), running_var.device.clone())?;
            
            self.num_batches_tracked += 1;
        }
        
        Ok(())
    }
    
    /// Normalize input using mean and variance
    fn normalize(&self, input: &Tensor, mean: &Tensor, var: &Tensor) -> Result<Tensor> {
        let dims = input.shape().dims();
        let batch_size = dims[0];
        let num_channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        
        let input_data = input.to_vec()?;
        let mean_data = mean.to_vec()?;
        let var_data = var.to_vec()?;
        let mut output_data = vec![0.0f32; input_data.len()];
        
        for n in 0..batch_size {
            for c in 0..num_channels {
                let channel_mean = mean_data[c];
                let channel_std = (var_data[c] + self.eps).sqrt();
                
                for h in 0..height {
                    for w in 0..width {
                        let idx = n * (num_channels * height * width) 
                                + c * (height * width) 
                                + h * width 
                                + w;
                        output_data[idx] = (input_data[idx] - channel_mean) / channel_std;
                    }
                }
            }
        }
        
        Tensor::from_vec(output_data, input.shape().clone(), input.device.clone())
    }
    
    /// Apply affine transformation
    fn apply_affine(&self, normalized: &Tensor) -> Result<Tensor> {
        match (&self.weight, &self.bias) {
            (Some(weight), Some(bias)) => {
                let dims = normalized.shape().dims();
                let batch_size = dims[0];
                let num_channels = dims[1];
                let height = dims[2];
                let width = dims[3];
                
                let normalized_data = normalized.to_vec()?;
                let weight_data = weight.to_vec()?;
                let bias_data = bias.to_vec()?;
                let mut output_data = vec![0.0f32; normalized_data.len()];
                
                for n in 0..batch_size {
                    for c in 0..num_channels {
                        let gamma = weight_data[c];
                        let beta = bias_data[c];
                        
                        for h in 0..height {
                            for w in 0..width {
                                let idx = n * (num_channels * height * width) 
                                        + c * (height * width) 
                                        + h * width 
                                        + w;
                                output_data[idx] = gamma * normalized_data[idx] + beta;
                            }
                        }
                    }
                }
                
                Tensor::from_vec(output_data, normalized.shape().clone(), normalized.device.clone())
            }
            _ => {
                // Return a copy
                let data = normalized.to_vec()?;
                Tensor::from_vec(data, normalized.shape().clone(), normalized.device.clone())
            },
        }
    }
}

/// Layer Normalization
pub struct LayerNorm {
    pub normalized_shape: Vec<usize>,
    pub eps: f32,
    pub elementwise_affine: bool,
    
    // Learnable parameters
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
}

impl LayerNorm {
    /// Create a new LayerNorm layer with default elementwise_affine=true
    pub fn new(
        normalized_shape: usize,
        eps: f32,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        Self::new_with_affine(vec![normalized_shape], eps, true, device)
    }
    
    /// Create a new LayerNorm layer with explicit parameters
    pub fn new_with_affine(
        normalized_shape: Vec<usize>,
        eps: f32,
        elementwise_affine: bool,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let num_elements: usize = normalized_shape.iter().product();
        
        let (weight, bias) = if elementwise_affine {
            let weight = Tensor::from_vec(
                vec![1.0f32; num_elements],
                Shape::from_dims(&normalized_shape),
                device.clone(),
            )?;
            let bias = Tensor::zeros(Shape::from_dims(&normalized_shape), device)?;
            (Some(weight), Some(bias))
        } else {
            (None, None)
        };
        
        Ok(Self {
            normalized_shape,
            eps,
            elementwise_affine,
            weight,
            bias,
        })
    }
    
    /// Forward pass for LayerNorm
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_dims = input.shape().dims();
        let input_shape_len = input_dims.len();
        let normalized_shape_len = self.normalized_shape.len();
        
        // Validate that normalized_shape matches the last dimensions of input
        if normalized_shape_len > input_shape_len {
            return Err(FlameError::InvalidOperation(
                "Normalized shape is larger than input shape".into()
            ));
        }
        
        let start_idx = input_shape_len - normalized_shape_len;
        for i in 0..normalized_shape_len {
            if input_dims[start_idx + i] != self.normalized_shape[i] {
                return Err(FlameError::InvalidOperation(
                    format!("Shape mismatch at dimension {}: expected {}, got {}", 
                            i, self.normalized_shape[i], input_dims[start_idx + i])
                ));
            }
        }
        
        // Calculate statistics over the normalized dimensions
        let normalized = self.normalize(input)?;
        
        // Apply affine transformation if enabled
        if self.elementwise_affine {
            self.apply_affine(&normalized)
        } else {
            Ok(normalized)
        }
    }
    
    /// Normalize input
    fn normalize(&self, input: &Tensor) -> Result<Tensor> {
        let input_data = input.to_vec()?;
        let _input_dims = input.shape().dims();
        let total_size = input_data.len();
        
        // Calculate the size of the dimensions to normalize
        let norm_size: usize = self.normalized_shape.iter().product();
        let batch_size = total_size / norm_size;
        
        let mut output_data = vec![0.0f32; total_size];
        
        for b in 0..batch_size {
            let start_idx = b * norm_size;
            let end_idx = (b + 1) * norm_size;
            
            // Calculate mean
            let mut mean = 0.0f32;
            for i in start_idx..end_idx {
                mean += input_data[i];
            }
            mean /= norm_size as f32;
            
            // Calculate variance
            let mut var = 0.0f32;
            for i in start_idx..end_idx {
                let diff = input_data[i] - mean;
                var += diff * diff;
            }
            var /= norm_size as f32;
            
            // Normalize
            let std = (var + self.eps).sqrt();
            for i in start_idx..end_idx {
                output_data[i] = (input_data[i] - mean) / std;
            }
        }
        
        Tensor::from_vec(output_data, input.shape().clone(), input.device.clone())
    }
    
    /// Apply affine transformation
    fn apply_affine(&self, normalized: &Tensor) -> Result<Tensor> {
        match (&self.weight, &self.bias) {
            (Some(weight), Some(bias)) => {
                let normalized_data = normalized.to_vec()?;
                let weight_data = weight.to_vec()?;
                let bias_data = bias.to_vec()?;
                
                let total_size = normalized_data.len();
                let norm_size = weight_data.len();
                let batch_size = total_size / norm_size;
                
                let mut output_data = vec![0.0f32; total_size];
                
                for b in 0..batch_size {
                    for i in 0..norm_size {
                        let idx = b * norm_size + i;
                        output_data[idx] = weight_data[i] * normalized_data[idx] + bias_data[i];
                    }
                }
                
                Tensor::from_vec(output_data, normalized.shape().clone(), normalized.device.clone())
            }
            _ => {
                // Return a copy
                let data = normalized.to_vec()?;
                Tensor::from_vec(data, normalized.shape().clone(), normalized.device.clone())
            },
        }
    }
}

/// Group Normalization
pub struct GroupNorm {
    pub num_groups: usize,
    pub num_channels: usize,
    pub eps: f32,
    pub affine: bool,
    
    // Learnable parameters
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
}

impl GroupNorm {
    /// Create a new GroupNorm layer with default affine=true
    pub fn new(
        num_groups: usize,
        num_channels: usize,
        eps: f32,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        Self::new_with_affine(num_groups, num_channels, eps, true, device)
    }
    
    /// Create a new GroupNorm layer with explicit affine parameter
    pub fn new_with_affine(
        num_groups: usize,
        num_channels: usize,
        eps: f32,
        affine: bool,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        if num_channels % num_groups != 0 {
            return Err(FlameError::InvalidOperation(
                format!("num_channels {} must be divisible by num_groups {}", num_channels, num_groups)
            ));
        }
        
        let (weight, bias) = if affine {
            let weight = Tensor::from_vec(
                vec![1.0f32; num_channels],
                Shape::from_dims(&[num_channels]),
                device.clone(),
            )?;
            let bias = Tensor::zeros(Shape::from_dims(&[num_channels]), device)?;
            (Some(weight), Some(bias))
        } else {
            (None, None)
        };
        
        Ok(Self {
            num_groups,
            num_channels,
            eps,
            affine,
            weight,
            bias,
        })
    }
    
    /// Forward pass for GroupNorm
    /// Input shape: [N, C, H, W]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let dims = input.shape().dims();
        if dims.len() != 4 {
            return Err(FlameError::InvalidOperation(
                format!("GroupNorm expects 4D input [N,C,H,W], got {:?}", dims)
            ));
        }
        
        let batch_size = dims[0];
        let num_channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        
        if num_channels != self.num_channels {
            return Err(FlameError::InvalidOperation(
                format!("Expected {} channels, got {}", self.num_channels, num_channels)
            ));
        }
        
        let channels_per_group = self.num_channels / self.num_groups;
        let _spatial_size = height * width;
        
        let input_data = input.to_vec()?;
        let mut output_data = vec![0.0f32; input_data.len()];
        
        // Process each sample in the batch
        for n in 0..batch_size {
            // Process each group
            for g in 0..self.num_groups {
                let start_channel = g * channels_per_group;
                let end_channel = (g + 1) * channels_per_group;
                
                // Calculate mean for the group
                let mut mean = 0.0f32;
                let mut count = 0;
                for c in start_channel..end_channel {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = n * (num_channels * height * width)
                                    + c * (height * width)
                                    + h * width
                                    + w;
                            mean += input_data[idx];
                            count += 1;
                        }
                    }
                }
                mean /= count as f32;
                
                // Calculate variance for the group
                let mut var = 0.0f32;
                for c in start_channel..end_channel {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = n * (num_channels * height * width)
                                    + c * (height * width)
                                    + h * width
                                    + w;
                            let diff = input_data[idx] - mean;
                            var += diff * diff;
                        }
                    }
                }
                var /= count as f32;
                let std = (var + self.eps).sqrt();
                
                // Normalize the group
                for c in start_channel..end_channel {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = n * (num_channels * height * width)
                                    + c * (height * width)
                                    + h * width
                                    + w;
                            output_data[idx] = (input_data[idx] - mean) / std;
                        }
                    }
                }
            }
        }
        
        let normalized = Tensor::from_vec(output_data, input.shape().clone(), input.device.clone())?;
        
        // Apply affine transformation if enabled
        if self.affine {
            self.apply_affine(&normalized)
        } else {
            Ok(normalized)
        }
    }
    
    /// Apply affine transformation
    fn apply_affine(&self, normalized: &Tensor) -> Result<Tensor> {
        match (&self.weight, &self.bias) {
            (Some(weight), Some(bias)) => {
                let dims = normalized.shape().dims();
                let batch_size = dims[0];
                let num_channels = dims[1];
                let height = dims[2];
                let width = dims[3];
                
                let normalized_data = normalized.to_vec()?;
                let weight_data = weight.to_vec()?;
                let bias_data = bias.to_vec()?;
                let mut output_data = vec![0.0f32; normalized_data.len()];
                
                for n in 0..batch_size {
                    for c in 0..num_channels {
                        let gamma = weight_data[c];
                        let beta = bias_data[c];
                        
                        for h in 0..height {
                            for w in 0..width {
                                let idx = n * (num_channels * height * width)
                                        + c * (height * width)
                                        + h * width
                                        + w;
                                output_data[idx] = gamma * normalized_data[idx] + beta;
                            }
                        }
                    }
                }
                
                Tensor::from_vec(output_data, normalized.shape().clone(), normalized.device.clone())
            }
            _ => {
                // Return a copy
                let data = normalized.to_vec()?;
                Tensor::from_vec(data, normalized.shape().clone(), normalized.device.clone())
            },
        }
    }
}

/// Instance Normalization
pub struct InstanceNorm2d {
    pub num_features: usize,
    pub eps: f32,
    pub momentum: f32,
    pub affine: bool,
    pub track_running_stats: bool,
    
    // Learnable parameters
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
    
    // Running statistics (usually not used in InstanceNorm)
    pub running_mean: Option<Tensor>,
    pub running_var: Option<Tensor>,
    pub num_batches_tracked: usize,
}

impl InstanceNorm2d {
    /// Create a new InstanceNorm2d layer
    pub fn new(
        num_features: usize,
        eps: f32,
        momentum: f32,
        affine: bool,
        track_running_stats: bool,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let (weight, bias) = if affine {
            let weight = Tensor::from_vec(
                vec![1.0f32; num_features],
                Shape::from_dims(&[num_features]),
                device.clone(),
            )?;
            let bias = Tensor::zeros(Shape::from_dims(&[num_features]), device.clone())?;
            (Some(weight), Some(bias))
        } else {
            (None, None)
        };
        
        let (running_mean, running_var) = if track_running_stats {
            let running_mean = Tensor::zeros(Shape::from_dims(&[num_features]), device.clone())?;
            let running_var = Tensor::from_vec(
                vec![1.0f32; num_features],
                Shape::from_dims(&[num_features]),
                device.clone(),
            )?;
            (Some(running_mean), Some(running_var))
        } else {
            (None, None)
        };
        
        Ok(Self {
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            weight,
            bias,
            running_mean,
            running_var,
            num_batches_tracked: 0,
        })
    }
    
    /// Forward pass for InstanceNorm2d
    /// Input shape: [N, C, H, W]
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let dims = input.shape().dims();
        if dims.len() != 4 {
            return Err(FlameError::InvalidOperation(
                format!("InstanceNorm2d expects 4D input [N,C,H,W], got {:?}", dims)
            ));
        }
        
        let batch_size = dims[0];
        let num_channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        
        if num_channels != self.num_features {
            return Err(FlameError::InvalidOperation(
                format!("Expected {} channels, got {}", self.num_features, num_channels)
            ));
        }
        
        let spatial_size = height * width;
        let input_data = input.to_vec()?;
        let mut output_data = vec![0.0f32; input_data.len()];
        
        // Process each instance (N x C) separately
        for n in 0..batch_size {
            for c in 0..num_channels {
                // Calculate mean for this instance and channel
                let mut mean = 0.0f32;
                for h in 0..height {
                    for w in 0..width {
                        let idx = n * (num_channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w;
                        mean += input_data[idx];
                    }
                }
                mean /= spatial_size as f32;
                
                // Calculate variance
                let mut var = 0.0f32;
                for h in 0..height {
                    for w in 0..width {
                        let idx = n * (num_channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w;
                        let diff = input_data[idx] - mean;
                        var += diff * diff;
                    }
                }
                var /= spatial_size as f32;
                let std = (var + self.eps).sqrt();
                
                // Normalize
                for h in 0..height {
                    for w in 0..width {
                        let idx = n * (num_channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w;
                        output_data[idx] = (input_data[idx] - mean) / std;
                    }
                }
            }
        }
        
        let normalized = Tensor::from_vec(output_data, input.shape().clone(), input.device.clone())?;
        
        // Apply affine transformation if enabled
        if self.affine {
            self.apply_affine(&normalized)
        } else {
            Ok(normalized)
        }
    }
    
    /// Apply affine transformation (same as BatchNorm2d)
    fn apply_affine(&self, normalized: &Tensor) -> Result<Tensor> {
        match (&self.weight, &self.bias) {
            (Some(weight), Some(bias)) => {
                let dims = normalized.shape().dims();
                let batch_size = dims[0];
                let num_channels = dims[1];
                let height = dims[2];
                let width = dims[3];
                
                let normalized_data = normalized.to_vec()?;
                let weight_data = weight.to_vec()?;
                let bias_data = bias.to_vec()?;
                let mut output_data = vec![0.0f32; normalized_data.len()];
                
                for n in 0..batch_size {
                    for c in 0..num_channels {
                        let gamma = weight_data[c];
                        let beta = bias_data[c];
                        
                        for h in 0..height {
                            for w in 0..width {
                                let idx = n * (num_channels * height * width)
                                        + c * (height * width)
                                        + h * width
                                        + w;
                                output_data[idx] = gamma * normalized_data[idx] + beta;
                            }
                        }
                    }
                }
                
                Tensor::from_vec(output_data, normalized.shape().clone(), normalized.device.clone())
            }
            _ => {
                // Return a copy
                let data = normalized.to_vec()?;
                Tensor::from_vec(data, normalized.shape().clone(), normalized.device.clone())
            },
        }
    }
}


/// RMS Normalization (Root Mean Square Layer Normalization)
/// Used in many modern transformer models like LLaMA, Mistral, etc.
pub struct RMSNorm {
    pub eps: f32,
    pub elementwise_affine: bool,
    pub normalized_shape: Vec<usize>,
    
    // Learnable parameters
    pub weight: Option<Tensor>,
}

impl RMSNorm {
    /// Create a new RMSNorm layer
    pub fn new(
        normalized_shape: Vec<usize>,
        eps: f32,
        elementwise_affine: bool,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let num_elements: usize = normalized_shape.iter().product();
        
        let weight = if elementwise_affine {
            Some(Tensor::from_vec(
                vec![1.0f32; num_elements],
                Shape::from_dims(&normalized_shape),
                device,
            )?)
        } else {
            None
        };
        
        Ok(Self {
            normalized_shape,
            eps,
            elementwise_affine,
            weight,
        })
    }
    
    /// Forward pass for RMSNorm
    /// RMSNorm(x) = x * weight / sqrt(mean(x^2) + eps)
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_dims = input.shape().dims();
        let input_shape_len = input_dims.len();
        let normalized_shape_len = self.normalized_shape.len();
        
        // Validate that normalized_shape matches the last dimensions of input
        if normalized_shape_len > input_shape_len {
            return Err(FlameError::InvalidOperation(
                "Normalized shape is larger than input shape".into()
            ));
        }
        
        let start_idx = input_shape_len - normalized_shape_len;
        for i in 0..normalized_shape_len {
            if input_dims[start_idx + i] != self.normalized_shape[i] {
                return Err(FlameError::InvalidOperation(
                    format!("Shape mismatch at dimension {}: expected {}, got {}", 
                            i, self.normalized_shape[i], input_dims[start_idx + i])
                ));
            }
        }
        
        // Calculate RMS normalization
        let normalized = self.rms_normalize(input)?;
        
        // Apply weight if enabled
        if self.elementwise_affine {
            self.apply_weight(&normalized)
        } else {
            Ok(normalized)
        }
    }
    
    /// Perform RMS normalization
    fn rms_normalize(&self, input: &Tensor) -> Result<Tensor> {
        let input_data = input.to_vec()?;
        let total_size = input_data.len();
        
        // Calculate the size of the dimensions to normalize
        let norm_size: usize = self.normalized_shape.iter().product();
        let batch_size = total_size / norm_size;
        
        let mut output_data = vec![0.0f32; total_size];
        
        for b in 0..batch_size {
            let start_idx = b * norm_size;
            let end_idx = (b + 1) * norm_size;
            
            // Calculate mean of squares (RMS^2)
            let mut mean_sq = 0.0f32;
            for i in start_idx..end_idx {
                mean_sq += input_data[i] * input_data[i];
            }
            mean_sq /= norm_size as f32;
            
            // Normalize by RMS
            let rms = (mean_sq + self.eps).sqrt();
            for i in start_idx..end_idx {
                output_data[i] = input_data[i] / rms;
            }
        }
        
        Tensor::from_vec(output_data, input.shape().clone(), input.device.clone())
    }
    
    /// Apply weight transformation
    fn apply_weight(&self, normalized: &Tensor) -> Result<Tensor> {
        match &self.weight {
            Some(weight) => {
                let normalized_data = normalized.to_vec()?;
                let weight_data = weight.to_vec()?;
                
                let total_size = normalized_data.len();
                let norm_size = weight_data.len();
                let batch_size = total_size / norm_size;
                
                let mut output_data = vec![0.0f32; total_size];
                
                for b in 0..batch_size {
                    for i in 0..norm_size {
                        let idx = b * norm_size + i;
                        output_data[idx] = weight_data[i] * normalized_data[idx];
                    }
                }
                
                Tensor::from_vec(output_data, normalized.shape().clone(), normalized.device.clone())
            }
            None => {
                // Return a copy
                let data = normalized.to_vec()?;
                Tensor::from_vec(data, normalized.shape().clone(), normalized.device.clone())
            },
        }
    }
}

/// RMSNorm specifically for 1D inputs (common in transformers)
pub struct RMSNorm1d {
    pub normalized_shape: usize,
    pub eps: f32,
    pub weight: Option<Tensor>,
}

impl RMSNorm1d {
    /// Create a new RMSNorm1d layer
    pub fn new(
        normalized_shape: usize,
        eps: f32,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<Self> {
        let weight = Some(Tensor::from_vec(
            vec![1.0f32; normalized_shape],
            Shape::from_dims(&[normalized_shape]),
            device,
        )?);
        
        Ok(Self {
            normalized_shape,
            eps,
            weight,
        })
    }
    
    /// Forward pass for RMSNorm1d
    /// Input shape: [..., normalized_shape]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let dims = input.shape().dims();
        let last_dim = dims[dims.len() - 1];
        
        if last_dim != self.normalized_shape {
            return Err(FlameError::InvalidOperation(
                format!("Expected last dimension {}, got {}", self.normalized_shape, last_dim)
            ));
        }
        
        // Use the general RMSNorm with the last dimension
        let rms_norm = RMSNorm {
            normalized_shape: vec![self.normalized_shape],
            eps: self.eps,
            elementwise_affine: self.weight.is_some(),
            weight: match &self.weight {
                Some(w) => Some(w.clone_result()?),
                None => None,
            },
        };
        
        rms_norm.forward(input)
    }
}
