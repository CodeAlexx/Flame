//! LoRA (Low-Rank Adaptation) implementation in FLAME

use crate::{Tensor, Shape, Result, FlameError};
use std::sync::Arc;
use cudarc::driver::CudaDevice;

/// LoRA adapter configuration
#[derive(Debug, Clone)]
pub struct LoRAConfig {
    /// Rank of the LoRA decomposition
    pub rank: usize,
    /// Scaling factor (alpha)
    pub alpha: f32,
    /// Dropout rate (0.0 = no dropout)
    pub dropout: f32,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 16.0,
            dropout: 0.0,
        }
    }
}

/// LoRA adapter layer
pub struct LoRALayer {
    /// Down projection: (rank, in_features)
    pub lora_down: Tensor,
    /// Up projection: (out_features, rank)
    pub lora_up: Tensor,
    /// Configuration
    pub config: LoRAConfig,
    /// Scaling factor
    pub scale: f32,
    /// Device
    device: Arc<CudaDevice>,
}

impl LoRALayer {
    /// Create a new LoRA layer
    pub fn new(
        in_features: usize,
        out_features: usize,
        config: LoRAConfig,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        // Initialize down projection with normal distribution
        let mut lora_down = Tensor::randn(
            Shape::from_dims(&[config.rank, in_features]),
            0.0,
            0.02,
            device.clone(),
        )?;
        
        // Initialize up projection with zeros (common practice)
        let mut lora_up = Tensor::zeros(
            Shape::from_dims(&[out_features, config.rank]),
            device.clone(),
        )?;
        
        // Set gradients required
        lora_down.requires_grad = true;
        lora_up.requires_grad = true;
        
        let scale = config.alpha / config.rank as f32;
        
        Ok(Self {
            lora_down,
            lora_up,
            config,
            scale,
            device,
        })
    }
    
    /// Apply LoRA to input
    /// output = input @ weight^T + scale * (input @ lora_down^T @ lora_up^T)
    pub fn forward(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor> {
        // Original computation: input @ weight^T
        let base_output = input.matmul(&weight.transpose()?)?;
        
        // LoRA computation: input @ lora_down^T @ lora_up^T
        let lora_output = input
            .matmul(&self.lora_down.transpose()?)?
            .matmul(&self.lora_up.transpose()?)?;
        
        // Scale and add
        let scaled_lora = lora_output.mul_scalar(self.scale)?;
        base_output.add(&scaled_lora)
    }
    
    /// Get trainable parameters
    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.lora_down, &self.lora_up]
    }
    
    /// Get parameter count
    pub fn num_parameters(&self) -> usize {
        self.lora_down.shape().elem_count() + self.lora_up.shape().elem_count()
    }
    
    /// Merge LoRA weights into the base weight
    /// Used for inference after training
    pub fn merge_weights(&self, weight: &mut Tensor) -> Result<()> {
        // Compute LoRA weight update: scale * lora_up @ lora_down
        let lora_weight = self.lora_up
            .matmul(&self.lora_down)?
            .mul_scalar(self.scale)?;
        
        // Add to original weight
        *weight = weight.add(&lora_weight)?;
        Ok(())
    }
}

/// Collection of LoRA layers for a model
pub struct LoRACollection {
    /// Map of layer name to LoRA adapter
    pub layers: std::collections::HashMap<String, LoRALayer>,
    /// Default configuration
    pub config: LoRAConfig,
    /// Device
    device: Arc<CudaDevice>,
}

impl LoRACollection {
    /// Create a new LoRA collection
    pub fn new(config: LoRAConfig, device: Arc<CudaDevice>) -> Self {
        Self {
            layers: std::collections::HashMap::new(),
            config,
            device,
        }
    }
    
    /// Add a LoRA adapter for a specific layer
    pub fn add_layer(
        &mut self,
        name: &str,
        in_features: usize,
        out_features: usize,
    ) -> Result<()> {
        let layer = LoRALayer::new(
            in_features,
            out_features,
            self.config.clone(),
            self.device.clone(),
        )?;
        self.layers.insert(name.to_string(), layer);
        Ok(())
    }
    
    /// Get a LoRA layer by name
    pub fn get(&self, name: &str) -> Option<&LoRALayer> {
        self.layers.get(name)
    }
    
    /// Get mutable LoRA layer by name
    pub fn get_mut(&mut self, name: &str) -> Option<&mut LoRALayer> {
        self.layers.get_mut(name)
    }
    
    /// Get all trainable parameters
    pub fn parameters(&self) -> Vec<&Tensor> {
        self.layers
            .values()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
    
    /// Get total parameter count
    pub fn num_parameters(&self) -> usize {
        self.layers
            .values()
            .map(|layer| layer.num_parameters())
            .sum()
    }
    
    /// Apply LoRA to a specific layer
    pub fn forward(&self, layer_name: &str, input: &Tensor, weight: &Tensor) -> Result<Tensor> {
        if let Some(lora_layer) = self.get(layer_name) {
            lora_layer.forward(input, weight)
        } else {
            // No LoRA for this layer, just do standard computation
            input.matmul(&weight.transpose()?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lora_forward() -> Result<()> {
        let device = CudaDevice::new(0)?;
        
        // Create test tensors
        let batch_size = 2;
        let in_features = 128;
        let out_features = 256;
        
        let input = Tensor::randn(
            Shape::from_dims(&[batch_size, in_features]),
            0.0,
            1.0,
            device.clone(),
        )?;
        
        let weight = Tensor::randn(
            Shape::from_dims(&[out_features, in_features]),
            0.0,
            0.1,
            device.clone(),
        )?;
        
        // Create LoRA layer
        let config = LoRAConfig {
            rank: 8,
            alpha: 8.0,
            dropout: 0.0,
        };
        
        let lora = LoRALayer::new(in_features, out_features, config, device)?;
        
        // Forward pass
        let output = lora.forward(&input, &weight)?;
        
        // Check output shape
        assert_eq!(output.shape().dims(), &[batch_size, out_features]);
        
        Ok(())
    }
    
    #[test]
    fn test_lora_collection() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let config = LoRAConfig::default();
        
        let mut collection = LoRACollection::new(config, device.clone());
        
        // Add some layers
        collection.add_layer("layer1", 128, 256)?;
        collection.add_layer("layer2", 256, 512)?;
        
        // Check parameter count
        let expected_params = 2 * (16 * 128 + 256 * 16) + 2 * (16 * 256 + 512 * 16);
        assert_eq!(collection.num_parameters(), expected_params);
        
        Ok(())
    }
}