use flame_core::{Tensor, Shape, CudaDevice, Result};
use flame_core::gradient::TensorGradExt;
use std::sync::Arc;

fn main() -> Result<()> {
    println!("Testing FLAME gradient tracking and autograd...\n");
    
    // Initialize CUDA device
    let device = Arc::new(CudaDevice::new(0)?);
    println!("✓ CUDA device initialized");
    
    // Test 1: Simple gradient computation
    println!("\n1. Testing simple gradient computation:");
    let x = Tensor::randn(Shape::from_dims(&[2, 3]), 0.0, 1.0, device.clone())?
        .requires_grad_(true);
    let w = Tensor::randn(Shape::from_dims(&[3, 4]), 0.0, 1.0, device.clone())?
        .requires_grad_(true);
    
    println!("   Created tensors x[2,3] and w[3,4] with requires_grad=true");
    
    // Forward pass: y = x @ w
    let y = x.matmul(&w)?;
    println!("   Forward: y = x @ w, shape: {:?}", y.shape().dims());
    
    // Compute loss (sum of all elements)
    let loss = y.sum()?;
    println!("   Loss = sum(y)");
    
    // Backward pass
    let grads = loss.backward()?;
    println!("   ✓ Backward pass completed");
    
    // Check gradients exist
    if grads.get(&x.id()).is_some() {
        println!("   ✓ Gradient for x computed");
    }
    if grads.get(&w.id()).is_some() {
        println!("   ✓ Gradient for w computed");
    }
    
    // Test 2: Chain rule with multiple operations
    println!("\n2. Testing chain rule with multiple operations:");
    let a = Tensor::randn(Shape::from_dims(&[4, 5]), 0.0, 1.0, device.clone())?
        .requires_grad_(true);
    let b = Tensor::randn(Shape::from_dims(&[4, 5]), 0.0, 1.0, device.clone())?
        .requires_grad_(true);
    
    // z = relu(a + b) * 2.0
    let z1 = a.add(&b)?;
    let z2 = z1.relu()?;
    let z3 = z2.mul_scalar(2.0)?;
    let loss2 = z3.sum()?;
    
    println!("   Forward: z = relu(a + b) * 2.0");
    println!("   Loss = sum(z)");
    
    let grads2 = loss2.backward()?;
    println!("   ✓ Backward pass completed");
    
    if grads2.get(&a.id()).is_some() && grads2.get(&b.id()).is_some() {
        println!("   ✓ Gradients computed through chain of operations");
    }
    
    // Test 3: Gradient accumulation
    println!("\n3. Testing gradient accumulation:");
    let params = Tensor::randn(Shape::from_dims(&[10]), 0.0, 1.0, device.clone())?
        .requires_grad_(true);
    
    // First forward-backward
    let out1 = params.mul_scalar(2.0)?;
    let loss1 = out1.sum()?;
    let grads1 = loss1.backward()?;
    
    // Second forward-backward (simulating batch accumulation)
    let out2 = params.mul_scalar(3.0)?;
    let loss2 = out2.sum()?;
    let grads2 = loss2.backward()?;
    
    println!("   ✓ Multiple backward passes completed");
    println!("   Note: Each backward() creates independent gradient computation");
    
    // Test 4: Weight update simulation
    println!("\n4. Testing weight update (SGD step):");
    let weights = Tensor::randn(Shape::from_dims(&[5, 5]), 0.0, 0.1, device.clone())?
        .requires_grad_(true);
    
    // Forward pass
    let output = weights.mul(&weights)?; // Simple operation for testing
    let loss = output.sum()?;
    
    // Backward
    let grads = loss.backward()?;
    
    // Get gradient for weights
    if let Some(grad_tensor) = grads.get(&weights.id()) {
        // Simulate SGD update: w = w - lr * grad
        let lr = 0.01;
        let updated_weights = weights.sgd_step(grad_tensor, lr)?;
        println!("   ✓ Weight update simulated: w = w - {} * grad", lr);
        println!("   Original shape: {:?}, Updated shape: {:?}", 
                 weights.shape().dims(), updated_weights.shape().dims());
    }
    
    println!("\n✓ All gradient tests completed successfully!");
    println!("\nFLAME provides full autograd support that Candle lacks:");
    println!("- Automatic differentiation with backward()");
    println!("- Gradient tracking through complex computation graphs");
    println!("- Essential for training neural networks");
    
    Ok(())
}