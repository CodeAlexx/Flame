use flame_core::{Tensor, CudaDevice, Shape, autograd_ops::BackwardOps};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA
    let device = CudaDevice::new(0)?;
    println!("CUDA device initialized");

    println!("\n=== Testing Autograd with Simple Neural Network ===");
    
    // Create a simple 2-layer network for XOR problem
    // Input: 4x2 (4 samples, 2 features)
    // Hidden: 2x4 (hidden layer with 4 units)
    // Output: 4x1 (1 output)
    
    // XOR dataset
    let x = Tensor::from_vec(
        vec![0.0, 0.0,
             0.0, 1.0,
             1.0, 0.0,
             1.0, 1.0],
        Shape::from_dims(&[4, 2]),
        device.clone()
    )?;
    
    let y = Tensor::from_vec(
        vec![0.0, 1.0, 1.0, 0.0],
        Shape::from_dims(&[4, 1]),
        device.clone()
    )?;
    
    // Initialize weights
    let mut w1 = Tensor::randn(Shape::from_dims(&[2, 4]), 0.0, 0.5, device.clone())?
        .requires_grad_(true);
    let mut b1 = Tensor::zeros(Shape::from_dims(&[1, 4]), device.clone())?
        .requires_grad_(true);
    
    let mut w2 = Tensor::randn(Shape::from_dims(&[4, 1]), 0.0, 0.5, device.clone())?
        .requires_grad_(true);
    let mut b2 = Tensor::zeros(Shape::from_dims(&[1, 1]), device.clone())?
        .requires_grad_(true);
    
    let learning_rate = 0.1;
    let epochs = 1000;
    
    println!("Training XOR network...");
    let start = Instant::now();
    
    for epoch in 0..epochs {
        // Forward pass
        // h = relu(x @ w1 + b1)
        let h_pre = x.matmul(&w1)?;
        let h_pre_b = add_broadcast(&h_pre, &b1)?;
        let h = h_pre_b.relu()?;
        
        // y_pred = h @ w2 + b2
        let y_pred = h.matmul(&w2)?;
        let y_pred_b = add_broadcast(&y_pred, &b2)?;
        
        // Loss = mean((y_pred - y)^2)
        let diff = y_pred_b.sub(&y)?;
        let squared = diff.square()?;
        let loss = squared.mean()?;
        
        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, loss.item()?);
        }
        
        // Backward pass (manual for now)
        // d_loss/d_squared = 1/n
        let grad_squared = BackwardOps::mean_backward(
            &Tensor::from_vec(vec![1.0], Shape::from_dims(&[1]), device.clone())?,
            squared.shape()
        )?;
        
        // d_squared/d_diff = 2 * diff
        let grad_diff = BackwardOps::square_backward(&grad_squared, &diff)?;
        
        // d_diff/d_y_pred_b = 1, d_diff/d_y = -1
        let (grad_y_pred_b, _) = BackwardOps::sub_backward(&grad_diff)?;
        
        // Gradients for second layer
        let grad_b2 = sum_to_shape(&grad_y_pred_b, b2.shape())?;
        let (grad_y_pred, _) = BackwardOps::add_backward(&grad_y_pred_b)?;
        let (grad_h, grad_w2) = BackwardOps::matmul_backward(&grad_y_pred, &h, &w2)?;
        
        // Gradients through ReLU
        let grad_h_pre_b = BackwardOps::relu_backward(&grad_h, &h_pre_b)?;
        
        // Gradients for first layer
        let grad_b1 = sum_to_shape(&grad_h_pre_b, b1.shape())?;
        let (grad_h_pre, _) = BackwardOps::add_backward(&grad_h_pre_b)?;
        let (_, grad_w1) = BackwardOps::matmul_backward(&grad_h_pre, &x, &w1)?;
        
        // Update weights
        w1.update_weights(&grad_w1, learning_rate)?;
        b1.update_weights(&grad_b1, learning_rate)?;
        w2.update_weights(&grad_w2, learning_rate)?;
        b2.update_weights(&grad_b2, learning_rate)?;
    }
    
    let elapsed = start.elapsed();
    println!("\nTraining completed in {:?}", elapsed);
    
    // Test the trained network
    println!("\n=== Testing Trained Network ===");
    let h_pre = x.matmul(&w1)?;
    let h_pre_b = add_broadcast(&h_pre, &b1)?;
    let h = h_pre_b.relu()?;
    let y_pred = h.matmul(&w2)?;
    let y_pred_b = add_broadcast(&y_pred, &b2)?;
    
    let predictions = y_pred_b.to_vec()?;
    let targets = y.to_vec()?;
    
    println!("Input -> Prediction (Target)");
    println!("[0, 0] -> {:.3} ({})", predictions[0], targets[0]);
    println!("[0, 1] -> {:.3} ({})", predictions[1], targets[1]);
    println!("[1, 0] -> {:.3} ({})", predictions[2], targets[2]);
    println!("[1, 1] -> {:.3} ({})", predictions[3], targets[3]);
    
    // Check accuracy
    let mut correct = 0;
    for i in 0..4 {
        let pred = if predictions[i] > 0.5 { 1.0 } else { 0.0 };
        if (pred - targets[i]).abs() < 0.1 {
            correct += 1;
        }
    }
    println!("\nAccuracy: {}/4", correct);
    
    Ok(())
}

// Helper function to broadcast add
fn add_broadcast(a: &Tensor, b: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
    // Simple broadcast for bias addition
    let a_data = a.to_vec()?;
    let b_data = b.to_vec()?;
    let a_shape = a.shape().dims();
    let b_shape = b.shape().dims();
    
    if b_shape.len() != 2 || b_shape[0] != 1 {
        return Err("Unsupported broadcast shape".into());
    }
    
    let mut result = vec![0.0f32; a_data.len()];
    let rows = a_shape[0];
    let cols = a_shape[1];
    
    for i in 0..rows {
        for j in 0..cols {
            result[i * cols + j] = a_data[i * cols + j] + b_data[j];
        }
    }
    
    Ok(Tensor::from_vec(result, a.shape().clone(), a.device().clone())?)
}

// Helper function to sum tensor to target shape
fn sum_to_shape(tensor: &Tensor, target_shape: &Shape) -> Result<Tensor, Box<dyn std::error::Error>> {
    let data = tensor.to_vec()?;
    let tensor_shape = tensor.shape().dims();
    let target_dims = target_shape.dims();
    
    if target_dims.len() != 2 || target_dims[0] != 1 {
        return Err("Unsupported sum shape".into());
    }
    
    let rows = tensor_shape[0];
    let cols = tensor_shape[1];
    let mut result = vec![0.0f32; cols];
    
    for i in 0..rows {
        for j in 0..cols {
            result[j] += data[i * cols + j];
        }
    }
    
    Ok(Tensor::from_vec(result, target_shape.clone(), tensor.device().clone())?)
}