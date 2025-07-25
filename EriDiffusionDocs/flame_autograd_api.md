# FLAME Autograd API Documentation

This document describes FLAME's automatic differentiation system and how EriDiffusion should use it for training.

## Core Concepts

FLAME uses a tape-based autograd system similar to PyTorch. Operations are recorded on a tape during the forward pass, and gradients are computed during the backward pass.

## Key Types

```rust
use flame_core::autograd::{AutogradEngine, Op, TapeEntry};
use flame_core::gradient::{GradientMap, TensorGradExt};
use flame_core::parameter::{Parameter, ParameterDict};
use flame_core::{Tensor, TensorId, Result};
```

## Parameter Management

### Creating Parameters

```rust
use flame_core::parameter::Parameter;

// Create a trainable parameter
let weight = Parameter::randn(
    Shape::from_dims(&[out_features, in_features]),
    mean: 0.0,
    std: 0.02,
    device.clone()
)?;

// Create zero-initialized parameter
let bias = Parameter::zeros(
    Shape::from_dims(&[out_features]),
    device.clone()
)?;

// From existing tensor
let tensor = Tensor::randn(shape, 0.0, 0.02, device)?;
let param = Parameter::new(tensor);
```

### Parameter Operations

```rust
// Get current tensor value (returns a clone)
let tensor = param.tensor()?;

// Get parameter ID for tracking
let id = param.id();

// Check if requires gradients
let requires_grad = param.requires_grad();

// Get current gradient (if computed)
let grad = param.grad(); // Returns Option<Tensor>

// Zero out gradients before new backward pass
param.zero_grad();

// Manual parameter update (for simple SGD)
param.update(learning_rate)?; // param = param - lr * grad

// Apply arbitrary update tensor
let update = optimizer.compute_update(&grad)?;
param.apply_update(&update)?;
```

### Parameter Collections

```rust
use flame_core::parameter::ParameterDict;

// Create parameter collection
let mut params = ParameterDict::new();

// Add parameters
params.insert("weight".to_string(), weight);
params.insert("bias".to_string(), bias);

// Access parameters
let weight = params.get("weight").unwrap();

// Iterate over all parameters
for param in params.parameters() {
    param.zero_grad();
}

// Iterate with names
for (name, param) in params.named_parameters() {
    println!("Parameter {}: {:?}", name, param.shape());
}
```

## Autograd Engine

### Setting Up Autograd

```rust
use flame_core::autograd::AutogradEngine;

// Create autograd engine for a device
let mut autograd = AutogradEngine::new(device.clone());

// Enable autograd context for operations
let ctx = AutogradContext::new(device.clone());
```

### Recording Operations

Operations are automatically recorded when tensors have `requires_grad = true`:

```rust
// These operations will be recorded
let x = Tensor::randn(shape, 0.0, 1.0, device)?.requires_grad_(true);
let w = Parameter::randn(weight_shape, 0.0, 0.02, device)?;

let y = x.matmul(&w.tensor()?)?;  // Recorded
let z = y.relu()?;                 // Recorded
let loss = z.mean()?;              // Recorded
```

### Backward Pass

```rust
// Compute gradients
let grads = loss.backward()?; // Returns GradientMap

// Access gradients for specific tensors
if let Some(x_grad) = grads.get(&x.id()) {
    println!("Gradient for x: {:?}", x_grad.shape());
}

// Set gradients on parameters
for (name, param) in model.params.named_parameters() {
    if let Some(grad) = grads.get(&param.id()) {
        param.set_grad(grad.clone())?;
    }
}
```

## Gradient Management

### GradientMap

```rust
use flame_core::gradient::GradientMap;

// GradientMap stores gradients by TensorId
let mut grad_map = GradientMap::new();

// Insert gradient
grad_map.insert(tensor.id(), gradient_tensor);

// Get gradient
let grad = grad_map.get(&tensor.id());

// Accumulate gradients (for gradient accumulation)
grad_map.accumulate(&tensor.id(), new_gradient)?;
```

### Gradient Operations

```rust
// Gradient clipping
use flame_core::gradient_clip::{clip_grad_norm, clip_grad_value};

// Clip by norm
let total_norm = clip_grad_norm(&mut grads, max_norm: 1.0)?;

// Clip by value
clip_grad_value(&mut grads, clip_value: 5.0)?;

// Manual gradient computation
let manual_grad = compute_gradient(&output, &target)?;
param.set_grad(manual_grad)?;
```

## Common Training Patterns

### Basic Training Loop

```rust
// Model with parameters
struct Model {
    linear1: Linear,
    linear2: Linear,
}

impl Model {
    fn parameters(&self) -> Vec<&Parameter> {
        let mut params = vec![];
        params.push(&self.linear1.weight);
        if let Some(bias) = &self.linear1.bias {
            params.push(bias);
        }
        params.push(&self.linear2.weight);
        if let Some(bias) = &self.linear2.bias {
            params.push(bias);
        }
        params
    }
    
    fn zero_grad(&self) {
        for param in self.parameters() {
            param.zero_grad();
        }
    }
}

// Training loop
let mut optimizer = Adam::new(AdamConfig::default());

for epoch in 0..num_epochs {
    for (input, target) in dataloader {
        // Zero gradients
        model.zero_grad();
        
        // Forward pass
        let output = model.forward(&input)?;
        let loss = criterion.forward(&output, &target)?;
        
        // Backward pass
        let grads = loss.backward()?;
        
        // Update parameters
        let mut param_updates = vec![];
        for param in model.parameters() {
            if let Some(grad) = grads.get(&param.id()) {
                param.set_grad(grad.clone())?;
                param_updates.push((param.id().0, param, grad));
            }
        }
        
        // Optimizer step
        optimizer.step(&mut param_updates)?;
    }
}
```

### Gradient Accumulation

```rust
let accumulation_steps = 4;
let mut accumulated_grads = GradientMap::new();

for step in 0..accumulation_steps {
    let output = model.forward(&batch[step])?;
    let loss = criterion.forward(&output, &target[step])?;
    let loss_scaled = loss.div_scalar(accumulation_steps as f32)?;
    
    let grads = loss_scaled.backward()?;
    
    // Accumulate gradients
    for (tensor_id, grad) in grads.iter() {
        accumulated_grads.accumulate(tensor_id, grad)?;
    }
}

// Update parameters with accumulated gradients
for param in model.parameters() {
    if let Some(grad) = accumulated_grads.get(&param.id()) {
        param.set_grad(grad.clone())?;
    }
}
optimizer.step(&mut param_updates)?;
```

### Mixed Precision Training

```rust
// Scale loss for mixed precision
let loss_scale = 1024.0;
let scaled_loss = loss.mul_scalar(loss_scale)?;

// Backward with scaled loss
let grads = scaled_loss.backward()?;

// Unscale gradients before optimizer step
for (_, grad) in grads.iter_mut() {
    *grad = grad.div_scalar(loss_scale)?;
}
```

## Important Notes

1. **Always zero gradients**: Before each backward pass, zero out previous gradients:
```rust
model.zero_grad();
```

2. **Check gradient availability**: Not all tensors may have gradients:
```rust
if let Some(grad) = grads.get(&tensor.id()) {
    // Use gradient
}
```

3. **Detach when needed**: Stop gradient flow for certain operations:
```rust
let detached = tensor.detach()?; // No gradients will flow through this
```

4. **Parameter vs Tensor**: Use `Parameter` for trainable weights, `Tensor` for activations:
```rust
// Parameters persist across batches
let weight = Parameter::randn(shape, 0.0, 0.02, device)?;

// Tensors are temporary
let activation = input.matmul(&weight.tensor()?)?;
```

5. **Gradient checkpointing**: For memory efficiency:
```rust
use flame_core::gradient_checkpointing::checkpoint;

let output = checkpoint(|| {
    expensive_forward_pass(&input)
}, &input)?;
```

## Integration Example

```rust
// EriDiffusion integration
use flame_core::{Tensor, Parameter, Result};
use flame_core::autograd::AutogradEngine;
use flame_core::optimizers::Adam;

pub struct FluxLoRATrainer {
    model: FluxModel,
    optimizer: Adam,
    device: Arc<CudaDevice>,
}

impl FluxLoRATrainer {
    pub fn train_step(&mut self, batch: &Batch) -> Result<f32> {
        // Zero gradients
        self.model.zero_grad();
        
        // Forward pass
        let noise_pred = self.model.forward(&batch.latents, &batch.timesteps)?;
        
        // Compute loss
        let loss = (noise_pred.sub(&batch.noise)?).square()?.mean()?;
        
        // Backward pass
        let grads = loss.backward()?;
        
        // Update LoRA parameters
        let mut param_updates = vec![];
        for (name, param) in self.model.lora_parameters() {
            if let Some(grad) = grads.get(&param.id()) {
                param.set_grad(grad.clone())?;
                param_updates.push((param.id().0, param, grad));
            }
        }
        
        // Optimizer step
        self.optimizer.step(&mut param_updates)?;
        
        Ok(loss.to_scalar()?)
    }
}
```