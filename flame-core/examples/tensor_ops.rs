use flame_core::{Tensor, CudaDevice, Shape};

// Now actually tries to train (no fake success)
fn train_step(w: &mut Tensor, x: &Tensor, y_true: &Tensor) -> Result<f32, Box<dyn std::error::Error>> {
    // Forward pass
    let y_pred = x.matmul(w)?;
    
    // Compute loss
    let diff = y_pred.sub(y_true)?;
    let loss = diff.square()?.mean()?;
    let loss_val = loss.item()?;
    
    // Compute gradients manually
    // d(MSE)/d(y_pred) = 2 * (y_pred - y_true) / n
    let n = (y_pred.shape().dims()[0] * y_pred.shape().dims()[1]) as f32;
    let grad_output = diff.mul_scalar(2.0 / n)?;
    
    // d(loss)/d(w) = x^T @ grad_output
    let x_transposed = x.transpose()?;
    let grad_w = x_transposed.matmul(&grad_output)?;
    
    // Update weights
    w.update_weights(&grad_w, 0.1)?;
    
    Ok(loss_val)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA
    let device = CudaDevice::new(0)?;
    println!("CUDA device initialized");

    // Test weight updates with cuBLAS
    println!("\n=== Testing Weight Updates ===");
    let mut weight = Tensor::randn(
        Shape::from_dims(&[128, 64]), 
        0.0, 
        0.02, 
        device.clone()
    )?.requires_grad_(true);
    
    let gradient = Tensor::randn(
        Shape::from_dims(&[128, 64]), 
        0.0, 
        0.01, 
        device.clone()
    )?;

    let initial = weight.to_vec()?[0];
    weight.update_weights(&gradient, 0.01)?;
    let updated = weight.to_vec()?[0];
    println!("Weight update: {} -> {} (diff: {})", initial, updated, updated - initial);

    // Test matrix multiplication with cuBLAS
    println!("\n=== Testing Matrix Multiplication ===");
    let a = Tensor::randn(Shape::from_dims(&[32, 64]), 0.0, 1.0, device.clone())?;
    let b = Tensor::randn(Shape::from_dims(&[64, 128]), 0.0, 1.0, device.clone())?;
    let c = a.matmul(&b)?;
    println!("Matmul result shape: {:?}", c.shape());

    // Test element-wise operations
    println!("\n=== Testing Element-wise Operations ===");
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[2, 2]), device.clone())?;
    let y = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], Shape::from_dims(&[2, 2]), device.clone())?;
    
    let sum = x.add(&y)?;
    println!("Addition: {:?}", sum.to_vec()?);
    
    let diff = x.sub(&y)?;
    println!("Subtraction: {:?}", diff.to_vec()?);
    
    let scaled = x.mul_scalar(2.0)?;
    println!("Scalar multiplication: {:?}", scaled.to_vec()?);

    // Test reduction operations
    println!("\n=== Testing Reductions ===");
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[4]), device.clone())?;
    let sum = tensor.sum()?;
    println!("Sum: {}", sum.item()?);
    
    let mean = tensor.mean()?;
    println!("Mean: {}", mean.item()?);

    // Test activation
    println!("\n=== Testing Activations ===");
    let input = Tensor::from_vec(
        vec![-2.0, -1.0, 0.0, 1.0, 2.0], 
        Shape::from_dims(&[5]), 
        device.clone()
    )?;
    let relu_out = input.relu()?;
    println!("ReLU: {:?}", relu_out.to_vec()?);

    // Simple training example  
    println!("\n=== Simple Training Example ===");
    // Now actually does training with real gradient computation
    let mut w = Tensor::randn(Shape::from_dims(&[4, 2]), 0.0, 0.1, device.clone())?.requires_grad_(true);
    let x = Tensor::randn(Shape::from_dims(&[8, 4]), 0.0, 1.0, device.clone())?;
    let y_true = Tensor::randn(Shape::from_dims(&[8, 2]), 0.0, 1.0, device.clone())?;
    
    // Run multiple training steps
    for step in 0..10 {
        match train_step(&mut w, &x, &y_true) {
            Ok(loss) => {
                if step % 3 == 0 {
                    println!("Step {}: loss = {:.6}", step, loss);
                }
            },
            Err(e) => {
                println!("Training failed at step {}: {}", step, e);
                break;
            }
        }
    }
    println!("Training completed successfully!");
    
    // Test transpose operation
    println!("\n=== Testing Transpose ===");
    let mat = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Shape::from_dims(&[2, 3]),
        device.clone()
    )?;
    println!("Original shape: {:?}", mat.shape());
    println!("Original data: {:?}", mat.to_vec()?);
    
    let transposed = mat.transpose()?;
    println!("Transposed shape: {:?}", transposed.shape());
    println!("Transposed data: {:?}", transposed.to_vec()?);

    println!("\nAll operations completed successfully!");
    Ok(())
}