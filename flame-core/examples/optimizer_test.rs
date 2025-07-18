use flame_core::{Tensor, CudaDevice, Shape, optimizers::{Adam, AdamConfig, SGD, SGDConfig}};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA - CudaDevice::new already returns Arc<CudaDevice>
    let device = CudaDevice::new(0)?;
    println!("CUDA device initialized");

    println!("\n=== Testing Adam Optimizer ===");
    
    // Create a simple linear regression problem: y = 2x + 3
    let x = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        Shape::from_dims(&[5, 1]),
        device.clone()
    )?;
    
    let y_true = Tensor::from_vec(
        vec![5.0, 7.0, 9.0, 11.0, 13.0], // 2x + 3
        Shape::from_dims(&[5, 1]),
        device.clone()
    )?;
    
    // Initialize parameters
    let mut w = Tensor::randn(Shape::from_dims(&[1, 1]), 0.0, 0.1, device.clone())?;
    let mut b = Tensor::zeros(Shape::from_dims(&[1]), device.clone())?;
    
    // Create Adam optimizer
    let mut adam = Adam::new(AdamConfig {
        lr: 0.1,
        ..Default::default()
    });
    
    println!("Initial w: {:?}, b: {:?}", w.to_vec()?, b.to_vec()?);
    
    // Training loop
    for epoch in 0..100 {
        // Forward pass: y_pred = x @ w + b
        let y_pred = x.matmul(&w)?;
        let y_pred = add_bias_broadcast(&y_pred, &b)?;
        
        // Loss: MSE = mean((y_pred - y_true)^2)
        let diff = y_pred.sub(&y_true)?;
        let loss = diff.square()?.mean()?;
        
        if epoch % 20 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, loss.item()?);
        }
        
        // Compute gradients (simplified)
        let grad_diff = diff.mul_scalar(2.0 / 5.0)?; // d/d(diff) of mean squared error
        
        // Gradient w.r.t weights: x^T @ grad_diff
        let x_t = x.transpose()?;
        let grad_w = x_t.matmul(&grad_diff)?;
        
        // Gradient w.r.t bias: sum(grad_diff)
        let grad_b = grad_diff.sum()?;
        
        // Update parameters
        adam.step(&mut vec![
            (0, &mut w, &grad_w),
            (1, &mut b, &grad_b),
        ])?;
    }
    
    println!("\nFinal w: {:?}, b: {:?}", w.to_vec()?, b.to_vec()?);
    println!("Expected w: [2.0], b: [3.0]");
    
    // Test SGD optimizer
    println!("\n=== Testing SGD Optimizer ===");
    
    // Reset parameters
    let mut w_sgd = Tensor::randn(Shape::from_dims(&[1, 1]), 0.0, 0.1, device.clone())?;
    let mut b_sgd = Tensor::zeros(Shape::from_dims(&[1]), device.clone())?;
    
    let mut sgd = SGD::new(SGDConfig {
        lr: 0.01,
        momentum: 0.9,
        ..Default::default()
    });
    
    println!("Initial w: {:?}, b: {:?}", w_sgd.to_vec()?, b_sgd.to_vec()?);
    
    for epoch in 0..200 {
        // Forward pass
        let y_pred = x.matmul(&w_sgd)?;
        let y_pred = add_bias_broadcast(&y_pred, &b_sgd)?;
        
        // Loss
        let diff = y_pred.sub(&y_true)?;
        let loss = diff.square()?.mean()?;
        
        if epoch % 40 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, loss.item()?);
        }
        
        // Gradients
        let grad_diff = diff.mul_scalar(2.0 / 5.0)?;
        let x_t = x.transpose()?;
        let grad_w = x_t.matmul(&grad_diff)?;
        let grad_b = grad_diff.sum()?;
        
        // Update
        sgd.step(&mut vec![
            (0, &mut w_sgd, &grad_w),
            (1, &mut b_sgd, &grad_b),
        ])?;
    }
    
    println!("\nFinal w: {:?}, b: {:?}", w_sgd.to_vec()?, b_sgd.to_vec()?);
    
    // Test weight decay
    println!("\n=== Testing Weight Decay ===");
    
    let mut w_decay = Tensor::from_vec(vec![5.0], Shape::from_dims(&[1]), device.clone())?;
    
    let mut adam_decay = Adam::new(AdamConfig {
        lr: 0.1,
        weight_decay: 0.1,
        ..Default::default()
    });
    
    println!("Initial weight: {:?}", w_decay.to_vec()?);
    
    // Apply optimizer with zero gradient (only weight decay should apply)
    let zero_grad = Tensor::zeros(Shape::from_dims(&[1]), device)?;
    
    for i in 0..10 {
        adam_decay.step(&mut vec![(0, &mut w_decay, &zero_grad)])?;
        if i % 2 == 0 {
            println!("Step {}: weight = {:?}", i, w_decay.to_vec()?);
        }
    }
    
    println!("\nAll optimizer tests completed!");
    
    Ok(())
}

fn add_bias_broadcast(x: &Tensor, bias: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
    let x_data = x.to_vec()?;
    let bias_data = bias.to_vec()?;
    let x_shape = x.shape().dims();
    
    let mut result = vec![0.0f32; x_data.len()];
    let rows = x_shape[0];
    
    for i in 0..rows {
        result[i] = x_data[i] + bias_data[0];
    }
    
    Ok(Tensor::from_vec(result, x.shape().clone(), x.device().clone())?)
}