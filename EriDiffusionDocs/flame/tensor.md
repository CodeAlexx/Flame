# FLAME Tensor System

The tensor system is the foundation of FLAME, providing GPU-backed multi-dimensional arrays with automatic differentiation support.

## Tensor Basics

### Creating Tensors

```rust
use flame_core::{Tensor, Shape};
use std::sync::Arc;
use cudarc::driver::CudaDevice;

let device = Arc::new(CudaDevice::new(0)?);

// From data
let data = vec![1.0, 2.0, 3.0, 4.0];
let tensor = Tensor::from_vec(data, Shape::from_dims(&[2, 2]), device.clone())?;

// Random initialization
let normal = Tensor::randn(Shape::from_dims(&[32, 768]), 0.0, 1.0, device.clone())?;
let uniform = Tensor::rand(Shape::from_dims(&[10, 10]), device.clone())?;

// Constants
let zeros = Tensor::zeros(Shape::from_dims(&[5, 5]), device.clone())?;
let ones = Tensor::ones(Shape::from_dims(&[3, 3]), device.clone())?;
```

### Tensor Properties

```rust
let tensor = Tensor::randn(Shape::from_dims(&[32, 64, 128]), 0.0, 1.0, device)?;

// Shape information
let shape = tensor.shape(); // Shape object
let dims = shape.dims(); // &[32, 64, 128]
let numel = shape.elem_count(); // 262144

// Device and gradient
let device = tensor.device(); // Arc<CudaDevice>
let requires_grad = tensor.requires_grad; // bool

// Enable gradient computation
let tensor = tensor.requires_grad();
```

## Tensor Operations

### Arithmetic Operations

```rust
let a = Tensor::randn(Shape::from_dims(&[32, 64]), 0.0, 1.0, device.clone())?;
let b = Tensor::randn(Shape::from_dims(&[32, 64]), 0.0, 1.0, device.clone())?;

// Element-wise operations
let sum = a.add(&b)?;
let diff = a.sub(&b)?;
let prod = a.mul(&b)?;
let quot = a.div(&b)?;

// Scalar operations
let scaled = a.mul_scalar(2.0)?;
let shifted = a.add_scalar(1.0)?;

// In-place operations (create new tensor)
let c = a.add(&b)?; // Creates new tensor
```

### Matrix Operations

```rust
let a = Tensor::randn(Shape::from_dims(&[32, 64]), 0.0, 1.0, device.clone())?;
let b = Tensor::randn(Shape::from_dims(&[64, 128]), 0.0, 1.0, device.clone())?;

// Matrix multiplication
let c = a.matmul(&b)?; // Shape: [32, 128]

// Batch matrix multiplication
let batch_a = Tensor::randn(Shape::from_dims(&[10, 32, 64]), 0.0, 1.0, device.clone())?;
let batch_b = Tensor::randn(Shape::from_dims(&[10, 64, 128]), 0.0, 1.0, device.clone())?;
let batch_c = batch_a.bmm(&batch_b)?; // Shape: [10, 32, 128]

// Transpose
let transposed = a.transpose()?; // Shape: [64, 32]
```

### Shape Manipulation

```rust
let tensor = Tensor::randn(Shape::from_dims(&[2, 3, 4, 5]), 0.0, 1.0, device)?;

// Reshape
let reshaped = tensor.reshape(&[6, 20])?;

// Permute dimensions
let permuted = tensor.permute(&[0, 2, 1, 3])?; // [2, 4, 3, 5]

// Squeeze and unsqueeze
let squeezed = tensor.squeeze_dim(1)?; // Remove dimension of size 1
let unsqueezed = tensor.unsqueeze(0)?; // Add dimension at position 0

// Flatten
let flattened = tensor.flatten(1)?; // Flatten from dimension 1

// Slice
let sliced = tensor.narrow(0, 1, 1)?; // Slice dimension 0, start at 1, length 1
```

### Reduction Operations

```rust
let tensor = Tensor::randn(Shape::from_dims(&[32, 64, 128]), 0.0, 1.0, device)?;

// Global reductions
let sum = tensor.sum()?;
let mean = tensor.mean()?;
let max = tensor.max()?;
let min = tensor.min()?;

// Dimension reductions
let sum_dim = tensor.sum_dim(1, true)?; // Sum along dimension 1, keep dimension
let mean_dim = tensor.mean_dim(2, false)?; // Mean along dimension 2, remove dimension
```

### Activation Functions

```rust
let x = Tensor::randn(Shape::from_dims(&[32, 768]), -1.0, 1.0, device)?;

// Common activations
let relu = x.relu()?;
let sigmoid = x.sigmoid()?;
let tanh = x.tanh()?;
let gelu = x.gelu()?;
let silu = x.silu()?;

// Leaky ReLU
let leaky = x.leaky_relu(0.01)?;

// Softmax
let softmax = x.softmax(-1)?; // Along last dimension
```

## Broadcasting

FLAME supports NumPy-style broadcasting for element-wise operations:

```rust
let a = Tensor::randn(Shape::from_dims(&[32, 1, 128]), 0.0, 1.0, device.clone())?;
let b = Tensor::randn(Shape::from_dims(&[1, 64, 128]), 0.0, 1.0, device.clone())?;

// Broadcasts to [32, 64, 128]
let c = a.add(&b)?;
```

Broadcasting rules:
1. Dimensions are aligned from the right
2. Dimensions of size 1 can broadcast to any size
3. Missing dimensions are treated as size 1

## Memory Management

### Zero-Copy Views

Many operations return views without copying data:

```rust
let tensor = Tensor::randn(Shape::from_dims(&[100, 100]), 0.0, 1.0, device)?;

// These are all views (no data copy)
let transposed = tensor.transpose()?;
let sliced = tensor.narrow(0, 0, 50)?;
let reshaped = tensor.reshape(&[10000])?;
```

### Contiguous Memory

Some operations require contiguous memory:

```rust
let tensor = tensor.transpose()?; // Creates a view
let contiguous = tensor.contiguous()?; // Forces data copy if needed
```

## Gradient Tracking

### Enabling Gradients

```rust
// Enable gradients on creation
let x = Tensor::randn(Shape::from_dims(&[32, 10]), 0.0, 1.0, device)?
    .requires_grad();

// Enable gradients on existing tensor
let w = Tensor::randn(Shape::from_dims(&[10, 5]), 0.0, 0.1, device)?;
let w = w.requires_grad();
```

### Gradient Computation

```rust
// Forward pass
let y = x.matmul(&w)?;
let loss = y.sum()?;

// Backward pass
let grads = loss.backward()?;

// Access gradients
let x_grad = grads.get(x.id).expect("Missing gradient for x");
let w_grad = grads.get(w.id).expect("Missing gradient for w");
```

### Detaching from Graph

```rust
// Stop gradient flow
let y = x.matmul(&w)?;
let y_detached = y.detach()?; // No gradients will flow through this
```

## Performance Tips

### 1. Reuse Allocated Memory
```rust
// Bad: Allocates new memory each iteration
for i in 0..100 {
    let temp = Tensor::zeros(Shape::from_dims(&[1000, 1000]), device.clone())?;
    // ... use temp
}

// Good: Reuse memory
let mut buffer = Tensor::zeros(Shape::from_dims(&[1000, 1000]), device.clone())?;
for i in 0..100 {
    // ... use buffer, modify in place
}
```

### 2. Use Contiguous Memory for Performance-Critical Operations
```rust
let x = tensor.permute(&[2, 0, 1])?; // Creates non-contiguous view
let x = x.contiguous()?; // Ensure contiguous for subsequent operations
```

### 3. Batch Operations
```rust
// Bad: Individual operations
for i in 0..batch_size {
    let result = inputs[i].matmul(&weight)?;
}

// Good: Batched operation
let batched_input = Tensor::stack(&inputs, 0)?;
let results = batched_input.matmul(&weight)?;
```

### 4. Use Views When Possible
```rust
// Operations that return views (no copy):
// - transpose()
// - narrow()
// - squeeze()
// - unsqueeze()
// - reshape() (sometimes)
```

## Common Patterns

### Weight Initialization
```rust
// Xavier/Glorot initialization
let fan_in = 768;
let fan_out = 512;
let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
let weight = Tensor::randn(Shape::from_dims(&[fan_out, fan_in]), 0.0, std, device)?;

// He initialization
let std = (2.0 / fan_in as f32).sqrt();
let weight = Tensor::randn(Shape::from_dims(&[fan_out, fan_in]), 0.0, std, device)?;
```

### Normalization
```rust
// Compute mean and variance
let mean = tensor.mean_dim(dim, true)?;
let var = tensor.sub(&mean)?.square()?.mean_dim(dim, true)?;

// Normalize
let eps = 1e-5;
let std = var.add_scalar(eps)?.sqrt()?;
let normalized = tensor.sub(&mean)?.div(&std)?;
```

### Attention Pattern
```rust
// Scaled dot-product attention
let scale = 1.0 / (head_dim as f32).sqrt();
let scores = query.matmul(&key.transpose()?)?.mul_scalar(scale)?;
let weights = scores.softmax(-1)?;
let output = weights.matmul(&value)?;
```