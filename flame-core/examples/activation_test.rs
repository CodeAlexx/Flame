use flame_core::{Tensor, CudaDevice, Shape};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA - CudaDevice::new already returns Arc<CudaDevice>
    let device = CudaDevice::new(0)?;
    println!("CUDA device initialized");

    println!("\n=== Testing New Activation Functions ===");
    
    // Create test input
    let input = Tensor::from_vec(
        vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        Shape::from_dims(&[8]),
        device.clone()
    )?;
    
    println!("Input: {:?}", input.to_vec()?);
    
    // Test GELU
    let gelu_output = input.gelu()?;
    println!("\nGELU output: {:?}", gelu_output.to_vec()?);
    
    // Test SiLU (Swish)
    let silu_output = input.silu()?;
    println!("SiLU output: {:?}", silu_output.to_vec()?);
    
    // Test Tanh
    let tanh_output = input.tanh()?;
    println!("Tanh output: {:?}", tanh_output.to_vec()?);
    
    // Compare with ReLU
    let relu_output = input.relu()?;
    println!("ReLU output: {:?}", relu_output.to_vec()?);
    
    // Performance test
    println!("\n=== Performance Test ===");
    let large_input = Tensor::randn(
        Shape::from_dims(&[1024, 1024]),
        0.0, 1.0,
        device.clone()
    )?;
    
    // Test each activation function
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _ = large_input.relu()?;
    }
    let elapsed = start.elapsed();
    println!("ReLU: {:?} for 100 iterations", elapsed);
    
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _ = large_input.gelu()?;
    }
    let elapsed = start.elapsed();
    println!("GELU: {:?} for 100 iterations", elapsed);
    
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _ = large_input.silu()?;
    }
    let elapsed = start.elapsed();
    println!("SiLU: {:?} for 100 iterations", elapsed);
    
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _ = large_input.tanh()?;
    }
    let elapsed = start.elapsed();
    println!("Tanh: {:?} for 100 iterations", elapsed);
    
    // Test gradients
    println!("\n=== Testing Backward Pass ===");
    use flame_core::autograd_ops::BackwardOps;
    
    let grad_output = Tensor::from_vec(
        vec![1.0; 8],
        Shape::from_dims(&[8]),
        device
    )?;
    
    let gelu_grad = BackwardOps::gelu_backward(&grad_output, &input)?;
    println!("GELU gradient: {:?}", gelu_grad.to_vec()?);
    
    let silu_grad = BackwardOps::silu_backward(&grad_output, &input)?;
    println!("SiLU gradient: {:?}", silu_grad.to_vec()?);
    
    let tanh_grad = BackwardOps::tanh_backward(&grad_output, &input)?;
    println!("Tanh gradient: {:?}", tanh_grad.to_vec()?);
    
    println!("\nAll activation tests completed!");
    
    Ok(())
}