# FLAME Optimizer API Documentation

This document describes FLAME's optimizer implementations and how EriDiffusion should use them for training.

## Available Optimizers

FLAME provides several optimizer implementations:

```rust
use flame_core::optimizers::{
    Adam, AdamConfig,
    SGD, SGDConfig,
    AdamW, AdamWConfig,
    Lion, LionConfig,
    RMSprop, RMSpropConfig,
};
```

## Adam Optimizer

### Basic Usage

```rust
use flame_core::optimizers::{Adam, AdamConfig};

// Default configuration
let config = AdamConfig::default();
// lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0

// Custom configuration
let config = AdamConfig {
    lr: 1e-4,
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-8,
    weight_decay: 0.01,  // L2 regularization
};

// Create optimizer
let mut optimizer = Adam::new(config);
```

### Using Adam in Training Loop

```rust
// Collect parameters with their IDs and gradients
let mut param_updates = vec![];
for (name, param) in model.named_parameters() {
    if let Some(grad) = grads.get(&param.id()) {
        param.set_grad(grad.clone())?;
        // (param_id, mutable_param_ref, gradient_ref)
        param_updates.push((param.id().0, param, grad));
    }
}

// Update parameters
optimizer.step(&mut param_updates)?;

// Reset optimizer state if needed
optimizer.reset();
```

## AdamW Optimizer (Adam with Decoupled Weight Decay)

```rust
use flame_core::optimizers::{AdamW, AdamWConfig};

let config = AdamWConfig {
    lr: 1e-4,
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-8,
    weight_decay: 0.1,    // Typically higher than Adam
    correct_bias: true,   // Bias correction
};

let mut optimizer = AdamW::new(config);

// Use same as Adam
optimizer.step(&mut param_updates)?;
```

## SGD Optimizer

```rust
use flame_core::optimizers::{SGD, SGDConfig};

let config = SGDConfig {
    lr: 0.01,
    momentum: 0.9,
    weight_decay: 1e-4,
    nesterov: true,  // Nesterov momentum
};

let mut optimizer = SGD::new(config);

// With learning rate scheduling
let base_lr = 0.01;
let current_lr = base_lr * (0.1_f32).powf(epoch as f32 / 30.0);
optimizer.set_lr(current_lr);
```

## Lion Optimizer (EvoLved Sign Momentum)

```rust
use flame_core::optimizers::{Lion, LionConfig};

// Lion typically uses lower learning rates than Adam
let config = LionConfig {
    lr: 1e-4,      // 10x lower than Adam equivalent
    beta1: 0.9,
    beta2: 0.99,
    weight_decay: 0.0,
};

let mut optimizer = Lion::new(config);
```

## Learning Rate Scheduling

### Manual Scheduling

```rust
// Cosine annealing
fn cosine_lr(initial_lr: f32, current_step: usize, total_steps: usize) -> f32 {
    let progress = current_step as f32 / total_steps as f32;
    initial_lr * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
}

// Apply schedule
let lr = cosine_lr(1e-3, step, total_steps);
optimizer.set_lr(lr);
```

### Built-in Schedulers

```rust
use flame_core::optimizers::schedulers::{CosineScheduler, LinearScheduler, ExponentialScheduler};

// Cosine annealing with warmup
let scheduler = CosineScheduler::new(
    initial_lr: 1e-3,
    min_lr: 1e-5,
    warmup_steps: 1000,
    total_steps: 10000,
);

// Linear warmup then constant
let scheduler = LinearScheduler::new(
    initial_lr: 0.0,
    target_lr: 1e-3,
    warmup_steps: 1000,
);

// Exponential decay
let scheduler = ExponentialScheduler::new(
    initial_lr: 1e-3,
    decay_rate: 0.95,
    decay_steps: 1000,
);

// Use in training loop
let lr = scheduler.get_lr(current_step);
optimizer.set_lr(lr);
```

## Gradient Clipping

### Clip by Norm

```rust
use flame_core::gradient_clip::clip_grad_norm;

// Before optimizer step
let max_norm = 1.0;
let total_norm = clip_grad_norm(&mut param_updates, max_norm)?;

if total_norm > max_norm {
    println!("Gradient norm clipped from {:.3} to {:.3}", total_norm, max_norm);
}

// Then optimize
optimizer.step(&mut param_updates)?;
```

### Clip by Value

```rust
use flame_core::gradient_clip::clip_grad_value;

// Clip gradients to [-5.0, 5.0]
clip_grad_value(&mut param_updates, 5.0)?;

optimizer.step(&mut param_updates)?;
```

## Advanced Optimizer Patterns

### Parameter Groups with Different LRs

```rust
pub struct ParameterGroup {
    params: Vec<Parameter>,
    lr_scale: f32,
}

pub struct MultiGroupOptimizer {
    base_optimizer: Adam,
    groups: Vec<ParameterGroup>,
}

impl MultiGroupOptimizer {
    pub fn step(&mut self, grads: &GradientMap) -> Result<()> {
        for group in &self.groups {
            // Scale learning rate for this group
            let original_lr = self.base_optimizer.config.lr;
            self.base_optimizer.set_lr(original_lr * group.lr_scale);
            
            // Update parameters in this group
            let mut param_updates = vec![];
            for param in &group.params {
                if let Some(grad) = grads.get(&param.id()) {
                    param.set_grad(grad.clone())?;
                    param_updates.push((param.id().0, param, grad));
                }
            }
            
            self.base_optimizer.step(&mut param_updates)?;
            
            // Restore original LR
            self.base_optimizer.set_lr(original_lr);
        }
        Ok(())
    }
}
```

### EMA (Exponential Moving Average)

```rust
use std::collections::HashMap;

pub struct EMAOptimizer {
    optimizer: Adam,
    ema_params: HashMap<TensorId, Tensor>,
    decay: f32,
}

impl EMAOptimizer {
    pub fn new(optimizer: Adam, decay: f32) -> Self {
        Self {
            optimizer,
            ema_params: HashMap::new(),
            decay,
        }
    }
    
    pub fn step(&mut self, param_updates: &mut [(usize, &mut Parameter, &Tensor)]) -> Result<()> {
        // Regular optimizer step
        self.optimizer.step(param_updates)?;
        
        // Update EMA parameters
        for (_, param, _) in param_updates {
            let param_tensor = param.tensor()?;
            
            if let Some(ema) = self.ema_params.get_mut(&param.id()) {
                // ema = decay * ema + (1 - decay) * param
                *ema = ema.mul_scalar(self.decay)?
                    .add(&param_tensor.mul_scalar(1.0 - self.decay)?)?;
            } else {
                // Initialize EMA
                self.ema_params.insert(param.id(), param_tensor);
            }
        }
        
        Ok(())
    }
    
    pub fn swap_to_ema(&mut self, params: &[Parameter]) -> Result<()> {
        // Temporarily use EMA weights for evaluation
        for param in params {
            if let Some(ema) = self.ema_params.get(&param.id()) {
                param.data.lock().unwrap().copy_(ema)?;
            }
        }
        Ok(())
    }
}
```

## Memory-Efficient Optimizers

### 8-bit Adam

```rust
use flame_core::optimizers::{Adam8bit, Adam8bitConfig};

// Uses 8-bit statistics to reduce memory usage
let config = Adam8bitConfig {
    lr: 1e-4,
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-8,
    weight_decay: 0.0,
    block_size: 256,  // Quantization block size
};

let mut optimizer = Adam8bit::new(config);
// Uses ~75% less memory for optimizer states
```

### Adafactor (Memory-Efficient Adaptive Optimizer)

```rust
use flame_core::optimizers::{Adafactor, AdafactorConfig};

let config = AdafactorConfig {
    lr: None,  // Automatic learning rate
    eps: 1e-30,
    clip_threshold: 1.0,
    decay_rate: -0.8,
    weight_decay: 0.0,
    scale_parameter: true,
    relative_step: true,
};

let mut optimizer = Adafactor::new(config);
// Uses O(n) memory instead of O(n²) for Adam
```

## Optimizer State Management

### Saving Optimizer State

```rust
use flame_core::serialization::{save_optimizer, load_optimizer};

// Save optimizer state
save_optimizer(&optimizer, "optimizer_checkpoint.pt")?;

// Save with model checkpoint
let checkpoint = Checkpoint {
    model_state: model.state_dict(),
    optimizer_state: optimizer.state_dict(),
    epoch: current_epoch,
    step: global_step,
};
save_checkpoint(&checkpoint, "full_checkpoint.pt")?;
```

### Loading Optimizer State

```rust
// Load optimizer state
let optimizer = load_optimizer::<Adam>("optimizer_checkpoint.pt")?;

// Load full checkpoint
let checkpoint = load_checkpoint("full_checkpoint.pt")?;
optimizer.load_state_dict(checkpoint.optimizer_state)?;
```

## Common Patterns for EriDiffusion

### LoRA Training Optimizer Setup

```rust
pub struct LoRATrainingConfig {
    pub base_lr: f32,
    pub text_encoder_lr_scale: f32,
    pub unet_lr_scale: f32,
}

pub fn setup_lora_optimizer(
    model: &LoRAModel,
    config: &LoRATrainingConfig
) -> Result<Adam> {
    let adam_config = AdamConfig {
        lr: config.base_lr,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
    };
    
    let mut optimizer = Adam::new(adam_config);
    
    // Different LR for different parts
    // This is handled by collecting parameters in groups
    // and scaling LR during step()
    
    Ok(optimizer)
}
```

### Training Loop with Optimizer

```rust
impl FluxLoRATrainer {
    pub fn train_epoch(&mut self, dataloader: &DataLoader) -> Result<f32> {
        let mut total_loss = 0.0;
        
        for (step, batch) in dataloader.enumerate() {
            // Zero gradients
            self.model.zero_grad();
            
            // Forward pass
            let output = self.model.forward(&batch)?;
            let loss = self.compute_loss(&output, &batch)?;
            
            // Backward pass
            let grads = loss.backward()?;
            
            // Gradient clipping
            let mut param_updates = self.collect_param_updates(&grads)?;
            clip_grad_norm(&mut param_updates, 1.0)?;
            
            // Learning rate scheduling
            let lr = self.scheduler.get_lr(self.global_step);
            self.optimizer.set_lr(lr);
            
            // Optimizer step
            self.optimizer.step(&mut param_updates)?;
            
            // Update EMA if enabled
            if let Some(ema) = &mut self.ema {
                ema.update(&self.model)?;
            }
            
            total_loss += loss.to_scalar()?;
            self.global_step += 1;
        }
        
        Ok(total_loss / dataloader.len() as f32)
    }
}
```

## Best Practices

1. **Choose appropriate optimizer**:
   - Adam/AdamW: General purpose, good for most cases
   - SGD: When you need precise control
   - Lion: Memory efficient, good for large models
   - 8-bit optimizers: For very large models

2. **Learning rate tuning**:
   - Start with recommended defaults
   - Use warmup for large models
   - Consider different LRs for different parts

3. **Gradient clipping**:
   - Always clip for stability
   - Norm clipping usually better than value clipping

4. **State management**:
   - Save optimizer state with model
   - Reset optimizer for new training runs

5. **Memory efficiency**:
   - Use 8-bit optimizers for large models
   - Consider Adafactor for extreme cases