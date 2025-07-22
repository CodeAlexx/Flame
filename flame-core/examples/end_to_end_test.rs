//! End-to-end test of FLAME framework
//! Tests core functionality needed for training diffusion models

use flame_core::{
    Tensor, Shape, CudaDevice, Result, 
    AutogradContext,
    optimizers::{Adam, AdamConfig},
    conv::Conv2d,
    linear::Linear,
};

fn main() -> Result<()> {
    println!("🔥 FLAME End-to-End Integration Test\n");
    
    let device = CudaDevice::new(0)?;
    
    // Test 1: Simple neural network training
    println!("Test 1: Training a simple linear model");
    {
        AutogradContext::reset();
        
        // Create simple dataset: y = 2x + 3
        let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_data = vec![5.0, 7.0, 9.0, 11.0, 13.0];
        
        // Initialize parameters
        let mut w = Tensor::from_vec(vec![0.5], Shape::from_dims(&[1, 1]), device.clone())?
            .requires_grad_(true);
        let mut b = Tensor::from_vec(vec![0.0], Shape::from_dims(&[1]), device.clone())?
            .requires_grad_(true);
        
        // Create optimizer
        let mut optimizer = Adam::new(AdamConfig {
            lr: 0.1,
            ..Default::default()
        });
        
        // Training loop
        for epoch in 0..100 {
            let mut total_loss = 0.0;
            
            for i in 0..x_data.len() {
                // Reset gradients
                AutogradContext::reset();
                
                // Forward pass
                let x = Tensor::from_vec(vec![x_data[i]], Shape::from_dims(&[1, 1]), device.clone())?;
                let y_true = Tensor::from_vec(vec![y_data[i]], Shape::from_dims(&[1, 1]), device.clone())?;
                
                let y_pred = x.matmul(&w)?.add(&b)?;
                
                // Compute loss (MSE)
                let diff = y_pred.sub(&y_true)?;
                let loss = diff.square()?.mean()?;
                
                // Backward pass
                let grads = AutogradContext::backward(&loss)?;
                
                // Update parameters
                let w_id = w.id();
                let b_id = b.id();
                
                if let (Some(w_grad), Some(b_grad)) = (grads.get(w_id), grads.get(b_id)) {
                    optimizer.step(&mut vec![
                        (0, &mut w, w_grad),
                        (1, &mut b, b_grad),
                    ])?;
                    
                    total_loss += loss.to_vec()?[0];
                }
            }
            
            if epoch % 20 == 0 {
                println!("  Epoch {}: loss={:.6}, w={:.3}, b={:.3}", 
                    epoch, 
                    total_loss / x_data.len() as f32,
                    w.to_vec()?[0], 
                    b.to_vec()?[0]
                );
            }
        }
        
        println!("  Final: w={:.3}, b={:.3} (target: w=2.0, b=3.0)", 
            w.to_vec()?[0], b.to_vec()?[0]);
        
        // Check if converged close to target
        let w_val = w.to_vec()?[0];
        let b_val = b.to_vec()?[0];
        if (w_val - 2.0).abs() < 0.1 && (b_val - 3.0).abs() < 0.1 {
            println!("  ✅ Linear model training successful!");
        } else {
            println!("  ❌ Linear model did not converge properly");
        }
    }
    
    // Test 2: Conv2D forward and backward
    println!("\nTest 2: Conv2D operations");
    {
        AutogradContext::reset();
        
        // Create a small image batch
        let batch_size = 2;
        let in_channels = 3;
        let out_channels = 8;
        let height = 32;
        let width = 32;
        let kernel_size = 3;
        
        // Random input
        let input = Tensor::randn(
            Shape::from_dims(&[batch_size, in_channels, height, width]),
            0.0, 1.0, device.clone()
        )?.requires_grad_(true);
        
        // Create conv layer
        let mut conv = Conv2d::new(in_channels, out_channels, kernel_size, 1, 1, device.clone())?;
        
        // Forward pass
        let output = conv.forward(&input)?;
        println!("  Input shape: {:?}", input.shape());
        println!("  Output shape: {:?}", output.shape());
        
        // Create dummy loss
        let loss = output.mean()?;
        
        // Backward pass
        let grads = AutogradContext::backward(&loss)?;
        
        // Check gradients exist
        if grads.get(input.id()).is_some() {
            println!("  ✅ Input gradient computed");
        } else {
            println!("  ❌ Missing input gradient");
        }
        
        if grads.get(conv.weight.id()).is_some() {
            println!("  ✅ Weight gradient computed");
        } else {
            println!("  ❌ Missing weight gradient");
        }
        
        if let Some(ref bias) = conv.bias {
            if grads.get(bias.id()).is_some() {
                println!("  ✅ Bias gradient computed");
            } else {
                println!("  ❌ Missing bias gradient");
            }
        }
    }
    
    // Test 3: Multi-layer network
    println!("\nTest 3: Multi-layer network");
    {
        AutogradContext::reset();
        
        let batch_size = 4;
        let input_dim = 10;
        let hidden_dim = 20;
        let output_dim = 5;
        
        // Create layers
        let mut fc1 = Linear::new(input_dim, hidden_dim, true, &device)?;
        let mut fc2 = Linear::new(hidden_dim, output_dim, true, &device)?;
        
        // Random input and target
        let input = Tensor::randn(
            Shape::from_dims(&[batch_size, input_dim]),
            0.0, 1.0, device.clone()
        )?.requires_grad_(true);
        
        let target = Tensor::randn(
            Shape::from_dims(&[batch_size, output_dim]),
            0.0, 1.0, device.clone()
        )?;
        
        // Forward pass
        let hidden = fc1.forward(&input)?.relu()?;
        let output = fc2.forward(&hidden)?;
        
        // Loss
        let diff = output.sub(&target)?;
        let loss = diff.square()?.mean()?;
        
        // Backward
        let grads = AutogradContext::backward(&loss)?;
        
        // Check all gradients
        let has_all_grads = 
            grads.get(input.id()).is_some() &&
            grads.get(fc1.weight.id()).is_some() &&
            grads.get(fc2.weight.id()).is_some();
        
        if has_all_grads {
            println!("  ✅ All gradients computed for multi-layer network");
        } else {
            println!("  ❌ Missing gradients in multi-layer network");
        }
        
        println!("  Loss value: {:.6}", loss.to_vec()?[0]);
    }
    
    // Test 4: Gradient clipping
    println!("\nTest 4: Gradient clipping");
    {
        AutogradContext::reset();
        
        // Create tensor with large gradient
        let x = Tensor::from_vec(vec![10.0, -20.0, 30.0], Shape::from_dims(&[3]), device.clone())?
            .requires_grad_(true);
        
        // Operation that produces large gradients
        let y = x.mul(&x)?; // x^2
        let loss = y.sum()?;
        
        let grads = AutogradContext::backward(&loss)?;
        
        if let Some(x_grad) = grads.get(x.id()) {
            let grad_vals = x_grad.to_vec()?;
            println!("  Original gradients: {:?}", grad_vals);
            
            // Note: Actual gradient clipping would be done in the optimizer
            // This is just to verify gradients are computed correctly
            let max_grad = grad_vals.iter().map(|g| g.abs()).fold(0.0f32, f32::max);
            println!("  Max gradient magnitude: {:.2}", max_grad);
            
            if max_grad > 50.0 {
                println!("  ✅ Large gradients computed correctly");
            }
        }
    }
    
    println!("\n🎉 End-to-end tests completed!");
    
    Ok(())
}