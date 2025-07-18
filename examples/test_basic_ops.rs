use flame_core::{Tensor, Shape, CudaDevice, Result};
use std::sync::Arc;

fn main() -> Result<()> {
    println!("Testing FLAME basic tensor operations...");
    
    // Initialize CUDA device
    let device = CudaDevice::new(0)?;
    println!("✓ CUDA device initialized");
    
    // Test 1: Tensor creation
    println!("\n1. Testing tensor creation:");
    let a = Tensor::randn(Shape::from_dims(&[2, 3]), 0.0, 1.0, device.clone())?;
    println!("   Created tensor a with shape: {:?}", a.shape().dims());
    
    let b = Tensor::randn(Shape::from_dims(&[2, 3]), 0.0, 1.0, device.clone())?;
    println!("   Created tensor b with shape: {:?}", b.shape().dims());
    
    // Test 2: Addition
    println!("\n2. Testing addition:");
    let c = a.add(&b)?;
    println!("   a + b completed, result shape: {:?}", c.shape().dims());
    
    // Test 3: Multiplication
    println!("\n3. Testing multiplication:");
    let d = a.mul(&b)?;
    println!("   a * b completed, result shape: {:?}", d.shape().dims());
    
    // Test 4: Scalar operations
    println!("\n4. Testing scalar operations:");
    let e = a.mul_scalar(2.0)?;
    println!("   a * 2.0 completed");
    
    let f = a.add_scalar(1.0)?;
    println!("   a + 1.0 completed");
    
    // Test 5: Activation functions
    println!("\n5. Testing activation functions:");
    let relu_out = a.relu()?;
    println!("   ReLU completed");
    
    let sigmoid_out = a.sigmoid()?;
    println!("   Sigmoid completed");
    
    // Test 6: Matrix multiplication
    println!("\n6. Testing matrix multiplication:");
    let x = Tensor::randn(Shape::from_dims(&[3, 4]), 0.0, 1.0, device.clone())?;
    let y = Tensor::randn(Shape::from_dims(&[4, 5]), 0.0, 1.0, device.clone())?;
    let z = x.matmul(&y)?;
    println!("   [3,4] x [4,5] = {:?}", z.shape().dims());
    
    // Test 7: Sum reduction
    println!("\n7. Testing sum reduction:");
    let sum = a.sum()?;
    println!("   Sum completed");
    
    // Test 8: Transpose
    println!("\n8. Testing transpose:");
    let transposed = a.transpose()?;
    println!("   Transpose completed, new shape: {:?}", transposed.shape().dims());
    
    println!("\n✓ All basic operations completed successfully!");
    
    Ok(())
}