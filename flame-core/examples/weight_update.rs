use flame_core::{Tensor, CudaDevice, Shape};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA - CudaDevice::new already returns Arc<CudaDevice>
    let device = CudaDevice::new(0)?;
    println!("CUDA device initialized");

    // Create a weight tensor
    let mut weight = Tensor::randn(
        Shape::from_dims(&[128, 64]), 
        0.0, 
        0.02, 
        device.clone()
    )?;
    println!("Created weight tensor: {:?}", weight.shape());

    // Create a gradient tensor
    let gradient = Tensor::randn(
        Shape::from_dims(&[128, 64]), 
        0.0, 
        0.01, 
        device.clone()
    )?;
    println!("Created gradient tensor: {:?}", gradient.shape());

    // Get initial weight values
    let initial_weights = weight.to_vec()?;
    println!("Initial weight[0]: {}", initial_weights[0]);

    // THE KEY OPERATION - Update weights in place
    let learning_rate = 0.01;
    weight.update_weights(&gradient, learning_rate)?;
    println!("Updated weights with lr={}", learning_rate);

    // Verify weights changed
    let updated_weights = weight.to_vec()?;
    println!("Updated weight[0]: {}", updated_weights[0]);
    
    let diff = updated_weights[0] - initial_weights[0];
    println!("Difference: {}", diff);

    // Test matmul
    let input = Tensor::randn(
        Shape::from_dims(&[32, 128]), 
        0.0, 
        1.0, 
        device.clone()
    )?;
    
    let output = input.matmul(&weight)?;
    println!("Matmul output shape: {:?}", output.shape());

    println!("\nSuccess! Weight updates work on GPU!");
    Ok(())
}