//! Full training loop test with FLAME showing gradient modification capabilities
//! This demonstrates why FLAME is superior to Candle for training

use flame_core::{
    Tensor, Shape, Result, CudaDevice, AutogradContext, GradientMap,
    linear::Linear, conv::Conv2d, activations::relu,
    optimizers::{AdamW, OptimizerConfig},
};
use std::sync::Arc;

/// Simple CNN model for testing
struct SimpleCNN {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
    device: Arc<CudaDevice>,
}

impl SimpleCNN {
    fn new(device: Arc<CudaDevice>) -> Result<Self> {
        // Input: [batch, 1, 28, 28] -> Conv -> [batch, 32, 26, 26]
        let conv1 = Conv2d::new(1, 32, 3, 1, 0, 1, true, device.clone())?;
        
        // [batch, 32, 26, 26] -> Conv -> [batch, 64, 24, 24]
        let conv2 = Conv2d::new(32, 64, 3, 1, 0, 1, true, device.clone())?;
        
        // After flattening: [batch, 64 * 24 * 24] -> [batch, 128]
        let fc1 = Linear::new(64 * 24 * 24, 128, true, device.clone())?;
        
        // [batch, 128] -> [batch, 10]
        let fc2 = Linear::new(128, 10, true, device.clone())?;
        
        Ok(Self {
            conv1,
            conv2,
            fc1,
            fc2,
            device,
        })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Conv1 + ReLU
        let x = self.conv1.forward(x)?;
        let x = relu(&x)?;
        
        // Conv2 + ReLU
        let x = self.conv2.forward(&x)?;
        let x = relu(&x)?;
        
        // Flatten
        let batch_size = x.shape().dims()[0];
        let x = x.reshape(&[batch_size, 64 * 24 * 24])?;
        
        // FC1 + ReLU
        let x = self.fc1.forward(&x)?;
        let x = relu(&x)?;
        
        // FC2 (logits)
        self.fc2.forward(&x)
    }
    
    fn parameters(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![];
        
        params.push(&mut self.conv1.weight);
        if let Some(ref mut bias) = self.conv1.bias {
            params.push(bias);
        }
        
        params.push(&mut self.conv2.weight);
        if let Some(ref mut bias) = self.conv2.bias {
            params.push(bias);
        }
        
        params.push(&mut self.fc1.weight);
        if let Some(ref mut bias) = self.fc1.bias {
            params.push(bias);
        }
        
        params.push(&mut self.fc2.weight);
        if let Some(ref mut bias) = self.fc2.bias {
            params.push(bias);
        }
        
        params
    }
}

/// Cross entropy loss
fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    // Simplified version - just MSE for demonstration
    let probs = logits.softmax(-1)?;
    let diff = probs.sub(targets)?;
    let squared = diff.square()?;
    squared.mean()
}

fn main() -> Result<()> {
    // Initialize CUDA device
    let device = Arc::new(CudaDevice::new(0)?);
    println!("🔥 FLAME Training Loop Test - Demonstrating Gradient Modification\n");
    
    // Create model
    let mut model = SimpleCNN::new(device.clone())?;
    
    // Create optimizer with specific config
    let optimizer_config = OptimizerConfig {
        lr: 0.001,
        betas: (0.9, 0.999),
        eps: 1e-8,
        weight_decay: 0.01,
    };
    let mut optimizer = AdamW::with_config(model.parameters(), optimizer_config)?;
    
    // Training parameters
    let num_epochs = 3;
    let batch_size = 4;
    let gradient_clip_value = 1.0;
    
    println!("Training Configuration:");
    println!("- Epochs: {}", num_epochs);
    println!("- Batch size: {}", batch_size);
    println!("- Learning rate: {}", optimizer_config.lr);
    println!("- Gradient clipping: {}", gradient_clip_value);
    println!();
    
    // Training loop
    for epoch in 0..num_epochs {
        println!("=== Epoch {}/{} ===", epoch + 1, num_epochs);
        
        let mut epoch_loss = 0.0;
        let num_batches = 5; // Simulate 5 batches
        
        for batch_idx in 0..num_batches {
            // Clear gradients from previous step
            AutogradContext::clear();
            
            // Create dummy batch data
            let images = Tensor::randn(0.0, 0.1, 
                Shape::from_dims(&[batch_size, 1, 28, 28]), 
                device.clone()
            )?.requires_grad_(true);
            
            // Create one-hot encoded targets
            let mut target_data = vec![0.0; batch_size * 10];
            for i in 0..batch_size {
                let class = (batch_idx + i) % 10;
                target_data[i * 10 + class] = 1.0;
            }
            let targets = Tensor::from_vec(
                target_data,
                Shape::from_dims(&[batch_size, 10]),
                device.clone()
            )?;
            
            // Forward pass
            let logits = model.forward(&images)?;
            
            // Compute loss
            let loss = cross_entropy_loss(&logits, &targets)?;
            let loss_value = loss.to_vec()?[0];
            epoch_loss += loss_value;
            
            // Backward pass
            let mut gradients = AutogradContext::backward(&loss)?;
            
            // ===== GRADIENT MODIFICATION SECTION =====
            // This is the KEY feature that Candle cannot do!
            
            // 1. Gradient clipping by value
            let mut clipped_count = 0;
            let mut total_grad_norm = 0.0;
            
            for param in model.parameters() {
                if let Some(grad) = gradients.get(param.id) {
                    let grad_vec = grad.to_vec()?;
                    
                    // Calculate gradient norm
                    let grad_norm: f32 = grad_vec.iter()
                        .map(|g| g * g)
                        .sum::<f32>()
                        .sqrt();
                    total_grad_norm += grad_norm;
                    
                    // Clip gradients if needed
                    let needs_clipping = grad_vec.iter()
                        .any(|g| g.abs() > gradient_clip_value);
                    
                    if needs_clipping {
                        clipped_count += 1;
                        
                        // Create clipped gradient
                        let clipped_grad = grad_vec.iter()
                            .map(|g| g.clamp(-gradient_clip_value, gradient_clip_value))
                            .collect::<Vec<_>>();
                        
                        let clipped_tensor = Tensor::from_vec(
                            clipped_grad,
                            grad.shape().clone(),
                            device.clone()
                        )?;
                        
                        // Replace gradient with clipped version
                        gradients.insert(param.id, clipped_tensor);
                    }
                }
            }
            
            // 2. Gradient normalization (another example of modification)
            if total_grad_norm > 10.0 {
                println!("  Large gradient norm detected: {:.4}, normalizing...", total_grad_norm);
                
                for param in model.parameters() {
                    if let Some(grad) = gradients.get(param.id) {
                        let normalized_grad = grad.mul_scalar(10.0 / total_grad_norm)?;
                        gradients.insert(param.id, normalized_grad);
                    }
                }
            }
            
            // 3. Add gradient noise for better generalization (research technique)
            if epoch > 0 && batch_idx % 2 == 0 {
                for param in model.parameters() {
                    if let Some(grad) = gradients.get(param.id) {
                        let noise = Tensor::randn(0.0, 0.001, grad.shape().clone(), device.clone())?;
                        let noisy_grad = grad.add(&noise)?;
                        gradients.insert(param.id, noisy_grad);
                    }
                }
            }
            
            if clipped_count > 0 {
                println!("  Batch {}: Loss = {:.4}, Clipped {} gradients", 
                         batch_idx, loss_value, clipped_count);
            } else {
                println!("  Batch {}: Loss = {:.4}", batch_idx, loss_value);
            }
            
            // Optimizer step with MODIFIED gradients
            optimizer.step(&gradients)?;
        }
        
        let avg_loss = epoch_loss / num_batches as f32;
        println!("Epoch {} average loss: {:.4}\n", epoch + 1, avg_loss);
    }
    
    println!("=== Training Complete ===\n");
    println!("✅ FLAME successfully demonstrated:");
    println!("  1. Full training loop with automatic differentiation");
    println!("  2. Gradient clipping by value");
    println!("  3. Gradient normalization");
    println!("  4. Gradient noise injection");
    println!("  5. Custom gradient modifications before optimizer step");
    println!("\n🎯 These gradient modifications are IMPOSSIBLE with Candle!");
    println!("   This is why FLAME exists - to enable advanced training techniques.");
    
    Ok(())
}