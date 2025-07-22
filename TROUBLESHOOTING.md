# FLAME Troubleshooting Guide

## Common Issues and Solutions

### 1. CUDA_ERROR_INVALID_IMAGE

**Error**: `CUDA_ERROR_INVALID_IMAGE, "device kernel image is invalid"`

**Cause**: Trying to load raw CUDA C code as PTX

**Solution**: 
```rust
// Wrong:
device.load_ptx(KERNEL_CODE.into(), ...) // KERNEL_CODE is CUDA C

// Right:
let ptx = compile_cuda_kernel(KERNEL_CODE, "kernel_name")?;
device.load_ptx(ptx, ...)
```

### 2. Gradient Not Found

**Error**: `Missing gradient for x`

**Common Causes**:

1. **Forgot requires_grad**:
```rust
// Wrong:
let x = Tensor::from_vec(vec![1.0, 2.0], ...)?;

// Right:
let x = Tensor::from_vec(vec![1.0, 2.0], ...)?.requires_grad_(true);
```

2. **Operation doesn't propagate requires_grad**:
```rust
// Check in operation implementation:
if self.requires_grad || other.requires_grad {
    output.requires_grad = true;
    // Record operation for autograd
}
```

3. **Gradient map mismatch**:
```rust
// Make sure you're using the gradient map from the same backward call
let grads = AutogradContext::backward(&loss)?;
let x_grad = x.grad(&grads).unwrap(); // Use same `grads`
```

### 3. Test Failures When Run Together

**Symptom**: Tests pass individually but fail in batch

**Cause**: Global autograd context pollution

**Temporary Workaround**:
```bash
# Run tests one at a time
cargo test test_name -- --test-threads=1

# Or run specific test
cargo test test_basic_gradient_flow -- --exact
```

**Permanent Fix** (to implement):
```rust
#[test]
fn my_test() {
    AutogradContext::reset(); // Clear previous state
    // ... test code ...
}
```

### 4. Lifetime Errors with Kernel Names

**Error**: `borrowed value does not live long enough`

**Cause**: `load_ptx` expects 'static lifetime

**Solution**: Use `Box::leak` for kernel names (they're loaded once):
```rust
let kernel_name_static = Box::leak(kernel_name.to_string().into_boxed_str());
device.load_ptx(ptx, kernel_name_static, &[kernel_name_static])?;
```

### 5. Shape Mismatches in Backward Pass

**Error**: `ShapeMismatch { expected: Shape { dims: [2, 3] }, got: Shape { dims: [3, 2] } }`

**Common Causes**:

1. **Wrong transpose in gradient computation**:
```rust
// For C = A @ B, gradients are:
// dA = dC @ B^T (not B)
// dB = A^T @ dC (not A)
```

2. **Missing reshape in gradient**:
```rust
// If forward does reshape, backward must undo it
let grad_input = grad_output.reshape(&original_shape)?;
```

### 6. CUDA Out of Memory

**Error**: `CUDA_ERROR_OUT_OF_MEMORY`

**Solutions**:

1. **Check for memory leaks**:
```rust
// Make sure tensors go out of scope
{
    let temp = large_computation()?;
    // temp dropped here
}
```

2. **Use smaller batch sizes during testing**

3. **Clear GPU memory**:
```bash
nvidia-smi
# Find process using GPU memory and kill if needed
```

### 7. Kernel Not Found

**Error**: `Failed to get kernel_name`

**Causes**:

1. **Kernel not compiled/loaded**:
```rust
// Ensure kernel is loaded before use
ensure_kernel(&device, "kernel_name", KERNEL_CODE)?;
```

2. **Name mismatch**:
```rust
// Kernel name in code must match name in load_ptx
extern "C" __global__ void my_kernel(...) // <- "my_kernel"
device.load_ptx(ptx, "module", &["my_kernel"]) // <- must match
```

### 8. NaN or Inf in Gradients

**Common Causes**:

1. **Division by zero in backward**:
```rust
// Add epsilon for numerical stability
let grad = grad_output.div(&(x + 1e-8))?;
```

2. **Exploding gradients**:
```rust
// Clip gradients
if let Some(grad) = tensor.grad_mut(&mut grads) {
    grad.clamp_(-1.0, 1.0)?;
}
```

### 9. Slow Kernel Compilation

**Symptom**: First operation takes seconds

**Cause**: NVRTC compilation happens on first use

**Solutions**:

1. **Pre-compile common kernels**:
```rust
// In initialization
warmup_kernels(&device)?;
```

2. **Cache compiled PTX** (future feature)

### 10. Wrong Gradient Values

**Debugging Steps**:

1. **Verify with finite differences**:
```rust
// Numerical gradient
let eps = 1e-4;
let f_plus = f(x + eps)?;
let f_minus = f(x - eps)?;
let numerical_grad = (f_plus - f_minus) / (2.0 * eps);
```

2. **Check operation implementation**:
- Is gradient formula correct?
- Are saved tensors the right ones?
- Is broadcasting handled correctly?

3. **Enable debug output**:
```rust
println!("Computing {} gradient", op_name);
println!("Grad output shape: {:?}", grad_output.shape());
println!("Saved tensor shape: {:?}", saved.shape());
```

## Debug Utilities

### Print Tensor Stats
```rust
fn debug_tensor(name: &str, t: &Tensor) {
    println!("{}: shape={:?}, requires_grad={}, device={:?}", 
             name, t.shape(), t.requires_grad(), t.device);
    if let Ok(data) = t.to_vec() {
        println!("  data: {:?}", &data[..5.min(data.len())]);
    }
}
```

### Verify Gradients
```rust
fn check_gradients(x: &Tensor, grads: &GradientMap) {
    match x.grad(grads) {
        Some(g) => println!("Gradient found: {:?}", g.to_vec()?),
        None => println!("No gradient for tensor {}", x.id()),
    }
}
```

### Track Memory Usage
```bash
watch -n 1 nvidia-smi
```

## Getting Help

1. **Check existing tests** - They show correct usage
2. **Enable debug output** - Add println! statements
3. **Isolate the issue** - Minimal reproduction
4. **Check the math** - Gradient formulas online
5. **File an issue** - With reproduction steps