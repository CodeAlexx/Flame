# FLAME Tensor API Documentation

This document describes the tensor operations API that EriDiffusion should use when working with FLAME tensors.

## Core Tensor Type

```rust
use flame_core::Tensor;
use flame_core::Shape;
use flame_core::DType;
use std::sync::Arc;
use cudarc::driver::CudaDevice;
```

### Tensor Creation

```rust
// Create tensor from shape and device
let shape = Shape::from_dims(&[batch_size, channels, height, width]);
let device = Arc::new(CudaDevice::new(0)?);

// Zeros
let zeros = Tensor::zeros(shape.clone(), device.clone())?;

// Ones
let ones = Tensor::ones(shape.clone(), device.clone())?;

// Random normal distribution
let randn = Tensor::randn(shape.clone(), mean: 0.0, std: 1.0, device.clone())?;

// Random uniform distribution
let rand = Tensor::rand(shape.clone(), low: 0.0, high: 1.0, device.clone())?;

// From data
let data = vec![1.0, 2.0, 3.0, 4.0];
let tensor = Tensor::from_vec(data, Shape::from_dims(&[2, 2]), device.clone())?;

// Constants
let scalar = Tensor::scalar(3.14, device.clone())?;
```

### Tensor Properties

```rust
// Shape operations
let shape = tensor.shape();  // Returns &Shape
let dims = shape.dims();     // Returns &[usize]
let ndim = shape.ndim();     // Number of dimensions
let numel = shape.numel();   // Total elements

// Device
let device = tensor.device(); // Returns &Arc<CudaDevice>

// Gradient tracking
let requires_grad = tensor.requires_grad;
let tensor = tensor.requires_grad_(true); // Enable gradient tracking

// Unique ID for autograd
let id = tensor.id(); // Returns TensorId
```

### Basic Operations

```rust
// Arithmetic operations
let c = a.add(&b)?;           // Element-wise addition
let c = a.sub(&b)?;           // Element-wise subtraction  
let c = a.mul(&b)?;           // Element-wise multiplication
let c = a.div(&b)?;           // Element-wise division

// Scalar operations
let c = a.add_scalar(5.0)?;
let c = a.mul_scalar(2.0)?;
let c = a.sub_scalar(1.0)?;
let c = a.div_scalar(3.0)?;

// Matrix operations
let c = a.matmul(&b)?;        // Matrix multiplication
let c = a.transpose()?;       // Transpose last two dimensions
let c = a.permute(&[0, 2, 3, 1])?; // Permute dimensions

// Reduction operations
let sum = tensor.sum()?;      // Sum all elements
let mean = tensor.mean()?;    // Mean of all elements
let sum_dim = tensor.sum_dim(dim)?; // Sum along dimension
let mean_dim = tensor.mean_dim(dim)?; // Mean along dimension

// Shape operations
let reshaped = tensor.reshape(&[batch_size, -1])?;
let viewed = tensor.view(&new_shape)?;
let flattened = tensor.flatten(start_dim, end_dim)?;
let squeezed = tensor.squeeze(dim)?;
let unsqueezed = tensor.unsqueeze(dim)?;
```

### Advanced Operations

```rust
// Activation functions
let relu = tensor.relu()?;
let gelu = tensor.gelu()?;
let silu = tensor.silu()?;
let sigmoid = tensor.sigmoid()?;
let tanh = tensor.tanh()?;
let softmax = tensor.softmax(dim)?;

// Other operations
let sqrt = tensor.sqrt()?;
let square = tensor.square()?;
let exp = tensor.exp()?;
let log = tensor.log()?;
let abs = tensor.abs()?;

// Clipping
let clipped = tensor.clamp(min, max)?;

// Indexing and slicing
let slice = tensor.slice(dim, start, end)?;
let indexed = tensor.index_select(dim, &indices)?;
let gathered = tensor.gather(dim, &indices)?;

// Concatenation and stacking
let cat = Tensor::cat(&[&t1, &t2, &t3], dim)?;
let stack = Tensor::stack(&[&t1, &t2, &t3], dim)?;

// Broadcasting
let broadcasted = tensor.broadcast_to(&target_shape)?;
```

### Memory Management

```rust
// Clone (creates new tensor)
let cloned = tensor.clone()?;

// Detach from computation graph
let detached = tensor.detach()?;

// Move to different device
let tensor_gpu = tensor.to_device(&gpu_device)?;

// Convert dtype
let fp16_tensor = tensor.to_dtype(DType::F16)?;
let fp32_tensor = tensor.to_dtype(DType::F32)?;

// Contiguous memory
let contiguous = tensor.contiguous()?;
```

### Common Pitfalls and Best Practices

1. **Always handle Results**: All tensor operations return `Result<Tensor>`, handle errors properly:
```rust
// Good
let result = tensor.add(&other)?;

// Bad - will panic on error
let result = tensor.add(&other).unwrap();
```

2. **Shape compatibility**: Ensure shapes are compatible for operations:
```rust
// Check shapes before operations
if a.shape().dims()[1] != b.shape().dims()[0] {
    return Err(FlameError::ShapeMismatch { 
        expected: a.shape().clone(),
        got: b.shape().clone()
    });
}
```

3. **Gradient tracking**: Enable gradient tracking for trainable parameters:
```rust
// For parameters that need gradients
let weight = Tensor::randn(shape, 0.0, 0.02, device)?
    .requires_grad_(true);

// For intermediate computations that don't need gradients
let fixed = tensor.detach()?;
```

4. **Device consistency**: Ensure tensors are on the same device:
```rust
// Move to same device before operations
let b_on_same_device = b.to_device(a.device())?;
let result = a.add(&b_on_same_device)?;
```

5. **Memory efficiency**: Use in-place operations when possible:
```rust
// In-place operations (modify tensor directly)
tensor.add_(&other)?;  // Note the underscore
tensor.mul_(2.0)?;
```

6. **Broadcasting rules**: FLAME follows NumPy broadcasting rules:
```rust
// Shape [1, 3, 1] can broadcast with [2, 1, 4]
// Result shape will be [2, 3, 4]
```

## Integration with EriDiffusion

When using FLAME tensors in EriDiffusion:

```rust
use flame_core::{Tensor, Shape, Result};

// In your model forward pass
impl YourModel {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Use FLAME tensor operations
        let x = self.conv1.forward(x)?;
        let x = x.relu()?;
        let x = self.norm1.forward(&x)?;
        Ok(x)
    }
}

// In your training loop
let output = model.forward(&input)?;
let loss = criterion.forward(&output, &target)?;

// Backward pass handled by autograd
let grads = loss.backward()?;
```