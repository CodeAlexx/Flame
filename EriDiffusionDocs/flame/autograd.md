# FLAME Automatic Differentiation

FLAME implements a dynamic automatic differentiation system that tracks operations and computes gradients automatically.

## How Autograd Works

### Computation Graph

When tensors with `requires_grad=true` are used in operations, FLAME builds a dynamic computation graph:

```rust
let x = Tensor::randn([10, 5], 0.0, 1.0, device)?.requires_grad();
let w = Tensor::randn([5, 3], 0.0, 0.1, device)?.requires_grad();
let b = Tensor::zeros([3], device)?.requires_grad();

// Forward pass builds the graph
let y = x.matmul(&w)?;        // Records MatMul operation
let z = y.add(&b)?;          // Records Add operation  
let loss = z.sum()?;         // Records Sum operation
```

The computation graph tracks:
- Operation type (MatMul, Add, Sum, etc.)
- Input tensors
- Output tensors
- Saved tensors needed for backward pass

### Backward Pass

Calling `backward()` traverses the graph in reverse:

```rust
// Compute gradients
let grads = loss.backward()?;

// Access gradients
let x_grad = grads.get(x.id).unwrap();
let w_grad = grads.get(w.id).unwrap();
let b_grad = grads.get(b.id).unwrap();
```

## Gradient Computation Rules

### Basic Operations

```rust
// Addition: grad_a = grad_output, grad_b = grad_output
let c = a.add(&b)?;

// Subtraction: grad_a = grad_output, grad_b = -grad_output
let c = a.sub(&b)?;

// Multiplication: grad_a = grad_output * b, grad_b = grad_output * a
let c = a.mul(&b)?;

// Division: grad_a = grad_output / b, grad_b = -grad_output * a / b²
let c = a.div(&b)?;
```

### Matrix Operations

```rust
// Matrix multiplication: C = A @ B
// grad_A = grad_C @ B^T
// grad_B = A^T @ grad_C
let c = a.matmul(&b)?;

// Transpose: grad_input = grad_output^T
let b = a.transpose()?;
```

### Activation Functions

```rust
// ReLU: grad_input = grad_output * (input > 0)
let y = x.relu()?;

// Sigmoid: grad_input = grad_output * sigmoid(x) * (1 - sigmoid(x))
let y = x.sigmoid()?;

// Tanh: grad_input = grad_output * (1 - tanh²(x))
let y = x.tanh()?;
```

### Reduction Operations

```rust
// Sum: grad_input = grad_output (broadcasted)
let y = x.sum()?;

// Mean: grad_input = grad_output / num_elements (broadcasted)
let y = x.mean()?;

// Sum along dimension: grad_input = grad_output (broadcasted along dim)
let y = x.sum_dim(1, false)?;
```

## Advanced Features

### Gradient Accumulation

For large batch training:

```rust
let mut accumulated_grads = GradientMap::new();

for batch in batches {
    let output = model.forward(&batch)?;
    let loss = compute_loss(&output, &batch.targets)?;
    
    // Scale loss for accumulation
    let scaled_loss = loss.mul_scalar(1.0 / accumulation_steps)?;
    let grads = scaled_loss.backward()?;
    
    // Accumulate gradients
    for (tensor_id, grad) in grads.iter() {
        match accumulated_grads.get_mut(tensor_id) {
            Some(acc_grad) => {
                *acc_grad = acc_grad.add(&grad)?;
            }
            None => {
                accumulated_grads.insert(*tensor_id, grad.clone());
            }
        }
    }
}
```

### Gradient Checkpointing

Save memory by recomputing activations during backward:

```rust
// In forward pass
let x1 = layer1.forward(&input)?;
// Don't save x1 in computation graph

// In backward pass, recompute x1
let x1_recomputed = layer1.forward(&input)?;
let grad_input = layer1.backward(&x1_recomputed, &grad_output)?;
```

### Custom Backward Functions

Implement custom gradients for operations:

```rust
impl Op {
    CustomOp {
        forward: Box<dyn Fn(&[&Tensor]) -> Result<Tensor>>,
        backward: Box<dyn Fn(&Tensor, &[&Tensor]) -> Result<Vec<Tensor>>>,
    }
}

// Usage
let custom_op = Op::CustomOp {
    forward: Box::new(|inputs| {
        // Custom forward computation
        let x = &inputs[0];
        x.mul_scalar(2.0)
    }),
    backward: Box::new(|grad_output, inputs| {
        // Custom gradient computation
        let grad_input = grad_output.mul_scalar(2.0)?;
        Ok(vec![grad_input])
    }),
};
```

### Gradient Clipping

Prevent gradient explosion:

```rust
use flame_core::gradient_clip::GradientClipper;

let clipper = GradientClipper::new(1.0); // max norm = 1.0
let clipped_grads = clipper.clip_grads(&mut grads)?;
```

### No Gradient Context

Disable gradient tracking for inference:

```rust
// Method 1: Detach tensors
let y = x.detach()?.matmul(&w.detach()?)?;

// Method 2: No grad context
{
    let _guard = AutogradContext::no_grad();
    let y = x.matmul(&w)?; // No gradient tracking
}
```

## Implementation Details

### Operation Recording

Each operation that requires gradients is recorded:

```rust
pub struct GradientEntry {
    pub op: Op,
    pub inputs: Vec<TensorId>,
    pub output: TensorId,
    pub saved_tensors: Vec<(TensorId, Tensor)>,
}
```

### Topological Sort

The backward pass uses topological sorting to ensure gradients are computed in the correct order:

```rust
fn topological_sort(entries: &[GradientEntry]) -> Vec<usize> {
    // Build dependency graph
    // Perform DFS to get reverse topological order
    // Return indices in backward computation order
}
```

### Memory Efficiency

FLAME minimizes memory usage by:
- Only saving tensors needed for backward
- Freeing intermediate gradients after use
- Using in-place operations where possible
- Supporting gradient checkpointing

## Common Patterns

### Training Loop
```rust
let optimizer = Adam::new(&parameters, 1e-4);

for epoch in 0..num_epochs {
    for batch in dataloader {
        // Forward pass
        let output = model.forward(&batch.input)?;
        let loss = loss_fn(&output, &batch.target)?;
        
        // Backward pass
        let grads = loss.backward()?;
        
        // Update parameters
        optimizer.step(&grads)?;
        
        // Clear gradients
        AutogradContext::clear();
    }
}
```

### Mixed Precision Training
```rust
// Scale loss for FP16 training
let scale = 1024.0;
let scaled_loss = loss.mul_scalar(scale)?;
let grads = scaled_loss.backward()?;

// Unscale gradients before optimizer step
for (_, grad) in grads.iter_mut() {
    *grad = grad.mul_scalar(1.0 / scale)?;
}
```

### Gradient Debugging
```rust
// Check for NaN/Inf gradients
for (tensor_id, grad) in grads.iter() {
    let grad_data = grad.to_vec()?;
    if grad_data.iter().any(|&x| x.is_nan() || x.is_infinite()) {
        println!("Invalid gradient detected for tensor {:?}", tensor_id);
    }
}

// Print gradient statistics
let grad_norm = grad.square()?.sum()?.sqrt()?;
println!("Gradient norm: {}", grad_norm.to_vec()?[0]);
```

### Second-Order Gradients
```rust
// Compute gradient of gradient (Hessian-vector products)
let x = Tensor::randn([10], 0.0, 1.0, device)?.requires_grad();
let y = x.square()?.sum()?;

// First gradient
let grads = y.backward()?;
let x_grad = grads.get(x.id).unwrap();

// Second gradient (requires gradient of gradient)
let grad_sum = x_grad.sum()?;
let second_grads = grad_sum.backward()?;
```

## Performance Considerations

### 1. Minimize Saved Tensors
Operations save only what's needed for backward:
```rust
// ReLU only needs to save mask, not full input
// Dropout only needs to save mask
// LayerNorm saves normalized input, not original
```

### 2. Clear Gradients Regularly
```rust
// After each training step
AutogradContext::clear();
```

### 3. Use Detach for Non-Training Paths
```rust
// For validation/inference
let output = model.forward(&input.detach()?)?;
```

### 4. Profile Memory Usage
```rust
// Check autograd memory usage
let memory_stats = AutogradContext::memory_stats();
println!("Saved tensors: {} MB", memory_stats.saved_memory_mb);
```

## Troubleshooting

### Common Issues

1. **Out of Memory During Backward**
   - Enable gradient checkpointing
   - Reduce batch size
   - Clear gradients more frequently

2. **Gradients are None**
   - Ensure tensors have `requires_grad=true`
   - Check that operations support gradients
   - Verify the computation graph is connected

3. **Gradient Explosion/Vanishing**
   - Use gradient clipping
   - Check initialization
   - Consider gradient scaling

4. **Incorrect Gradients**
   - Verify custom backward implementations
   - Check for in-place operations breaking the graph
   - Test with gradient checking:

```rust
// Numerical gradient checking
fn check_gradients(f: impl Fn(&Tensor) -> Result<Tensor>, x: &Tensor, eps: f32) -> Result<()> {
    let y = f(x)?;
    let analytical_grads = y.backward()?;
    
    // Compute numerical gradients
    for i in 0..x.shape().elem_count() {
        let x_plus = x.add_at_index(i, eps)?;
        let x_minus = x.sub_at_index(i, eps)?;
        
        let y_plus = f(&x_plus)?;
        let y_minus = f(&x_minus)?;
        
        let numerical_grad = (y_plus - y_minus) / (2.0 * eps);
        // Compare with analytical gradient
    }
    Ok(())
}
```