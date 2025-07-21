use flame_core::{Tensor, Shape};
use cudarc::driver::CudaDevice;

fn main() -> flame_core::Result<()> {
    println!("=== FLAME Minimal Functionality Test ===\n");
    
    // Device creation
    let device = match CudaDevice::new(0) {
        Ok(d) => {
            println!("✅ PASS: CUDA device creation");
            d
        }
        Err(e) => {
            println!("❌ CRITICAL: Cannot create CUDA device - {:?}", e);
            return Ok(());
        }
    };
    
    // Basic tensor creation
    let x = match Tensor::randn(Shape::from_dims(&[10, 5]), 0.0, 1.0, device.clone()) {
        Ok(t) => {
            println!("✅ PASS: Random tensor creation");
            t
        }
        Err(e) => {
            println!("❌ CRITICAL: Cannot create random tensor - {:?}", e);
            return Ok(());
        }
    };
    
    // Basic arithmetic
    let y = match x.add(&x) {
        Ok(t) => {
            println!("✅ PASS: Tensor addition");
            t
        }
        Err(e) => {
            println!("❌ CRITICAL: Basic addition fails - {:?}", e);
            return Ok(());
        }
    };
    
    // Matrix multiplication test
    let w = match Tensor::randn(Shape::from_dims(&[5, 8]), 0.0, 0.1, device.clone()) {
        Ok(t) => t,
        Err(e) => {
            println!("❌ FAIL: Cannot create weight tensor - {:?}", e);
            return Ok(());
        }
    };
    
    let z = match x.matmul(&w) {
        Ok(t) => {
            println!("✅ PASS: Matrix multiplication");
            t
        }
        Err(e) => {
            println!("❌ CRITICAL: Matrix multiplication fails - {:?}", e);
            return Ok(());
        }
    };
    
    // Reduction test
    let sum_result = match z.sum() {
        Ok(t) => {
            println!("✅ PASS: Sum reduction");
            t
        }
        Err(e) => {
            println!("❌ CRITICAL: Sum reduction fails - {:?}", e);
            return Ok(());
        }
    };
    
    // Autograd test
    println!("\n--- Testing Autograd ---");
    let x_grad = Tensor::randn(Shape::from_dims(&[3, 4]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    let w_grad = Tensor::randn(Shape::from_dims(&[4, 2]), 0.0, 0.1, device.clone())?.requires_grad_(true);
    
    let y_grad = match x_grad.matmul(&w_grad) {
        Ok(t) => t,
        Err(e) => {
            println!("❌ FAIL: Autograd matmul fails - {:?}", e);
            return Ok(());
        }
    };
    
    let loss = match y_grad.sum() {
        Ok(t) => t,
        Err(e) => {
            println!("❌ FAIL: Autograd sum fails - {:?}", e);
            return Ok(());
        }
    };
    
    // Try backward pass
    match loss.backward() {
        Ok(grads) => {
            println!("✅ PASS: Backward pass completes");
            println!("   Gradients computed: {} tensors", grads.len());
            if grads.is_empty() {
                println!("   ⚠️  WARNING: No gradients returned!");
            }
        }
        Err(e) => {
            println!("❌ CRITICAL: Backward pass fails - {:?}", e);
        }
    }
    
    // Test activation functions
    println!("\n--- Testing Activations ---");
    let test_tensor = Tensor::randn(Shape::from_dims(&[10]), -1.0, 1.0, device.clone())?;
    
    match test_tensor.relu() {
        Ok(_) => println!("✅ PASS: ReLU activation"),
        Err(e) => println!("❌ FAIL: ReLU fails - {:?}", e),
    }
    
    match test_tensor.tanh() {
        Ok(_) => println!("✅ PASS: Tanh activation"),
        Err(e) => println!("❌ FAIL: Tanh fails - {:?}", e),
    }
    
    // Summary
    println!("\n=== Test Complete ===");
    Ok(())
}