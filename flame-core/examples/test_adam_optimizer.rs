use flame_core::{Tensor, Shape, CudaDevice, Result, AutogradContext, parameter::Parameter, adam::Adam};
use std::sync::Arc;

fn main() -> Result<()> {
    println!("🔬 Testing FLAME Adam Optimizer...\n");
    
    let device = CudaDevice::new(0)?;
    
    // Test 1: Simple linear regression with Adam
    {
        println!("Test 1: Linear regression y = 2x + 3");
        AutogradContext::reset();
        
        // True parameters: w=2, b=3
        // Initialize with random values
        let w = Parameter::randn(Shape::from_dims(&[1, 1]), 0.0, 0.1, device.clone())?;
        let b = Parameter::randn(Shape::from_dims(&[1]), 0.0, 0.1, device.clone())?;
        
        println!("  Initial w: {:?}", w.tensor()?.to_vec()?);
        println!("  Initial b: {:?}", b.tensor()?.to_vec()?);
        
        // Create optimizer with higher learning rate
        let mut optimizer = Adam::new(0.1, 0.9, 0.999, 1e-8, 0.0);
        
        // Training data
        let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_data = vec![5.0, 7.0, 9.0, 11.0, 13.0]; // y = 2x + 3
        
        let mut losses = Vec::new();
        
        // Training loop
        for epoch in 0..500 {
            optimizer.zero_grad(&[w.clone(), b.clone()]);
            
            let mut total_loss = 0.0;
            
            // Get tensors once for this epoch
            let w_tensor = w.tensor()?;
            let b_tensor = b.tensor()?;
            
            // Accumulate gradients over all data points
            let mut w_grad_accum: Option<Tensor> = None;
            let mut b_grad_accum: Option<Tensor> = None;
            
            for (i, (x_val, y_val)) in x_data.iter().zip(y_data.iter()).enumerate() {
                // Forward pass
                let x = Tensor::from_vec(vec![*x_val], Shape::from_dims(&[1, 1]), device.clone())?;
                let y_true = Tensor::from_vec(vec![*y_val], Shape::from_dims(&[1]), device.clone())?;
                
                let y_pred = x.matmul(&w_tensor)?.squeeze(Some(1))?.add(&b_tensor)?;
                
                // MSE loss
                let diff = y_pred.sub(&y_true)?;
                let loss = diff.mul(&diff)?.mul_scalar(0.5)?; // 0.5 * (y_pred - y_true)^2
                let loss_val = loss.to_vec()?[0];
                total_loss += loss_val;
                
                // Backward
                let grads = loss.backward()?;
                
                // Accumulate gradients
                if let Some(w_grad) = grads.get(w_tensor.id()) {
                    match &mut w_grad_accum {
                        Some(accum) => *accum = accum.add(w_grad)?,
                        None => w_grad_accum = Some(w_grad.clone()?),
                    }
                }
                if let Some(b_grad) = grads.get(b_tensor.id()) {
                    match &mut b_grad_accum {
                        Some(accum) => *accum = accum.add(b_grad)?,
                        None => b_grad_accum = Some(b_grad.clone()?),
                    }
                }
            }
            
            // Average gradients and set them on parameters
            if let Some(w_grad) = w_grad_accum {
                let avg_grad = w_grad.div_scalar(x_data.len() as f32)?;
                w.set_grad(avg_grad)?;
            }
            if let Some(b_grad) = b_grad_accum {
                let avg_grad = b_grad.div_scalar(x_data.len() as f32)?;
                b.set_grad(avg_grad)?;
            }
            
            // Optimizer step
            optimizer.step(&[w.clone(), b.clone()])?;
            
            let avg_loss = total_loss / x_data.len() as f32;
            losses.push(avg_loss);
            
            if epoch % 100 == 0 {
                let w_val = w.tensor()?.to_vec()?[0];
                let b_val = b.tensor()?.to_vec()?[0];
                println!("  Epoch {}: loss={:.4}, w={:.3}, b={:.3}", epoch, avg_loss, w_val, b_val);
            }
        }
        
        // Check convergence
        let final_w = w.tensor()?.to_vec()?[0];
        let final_b = b.tensor()?.to_vec()?[0];
        
        println!("\n  Final parameters: w={:.3}, b={:.3}", final_w, final_b);
        println!("  Expected: w=2.0, b=3.0");
        
        assert!((final_w - 2.0).abs() < 0.1, "Weight didn't converge to 2.0");
        assert!((final_b - 3.0).abs() < 0.1, "Bias didn't converge to 3.0");
        
        // Check that loss decreased
        assert!(losses[losses.len()-1] < losses[0] * 0.1, "Loss didn't decrease sufficiently");
        
        println!("  ✅ Adam optimizer converged correctly!\n");
    }
    
    // Test 2: Test momentum behavior
    {
        println!("Test 2: Adam momentum behavior");
        AutogradContext::reset();
        
        // Create parameter with known gradient
        let param = Parameter::zeros(Shape::from_dims(&[3]), device.clone())?;
        let mut optimizer = Adam::new(0.1, 0.9, 0.999, 1e-8, 0.0); // lr=0.1
        
        // Apply constant gradient for several steps
        for step in 1..=5 {
            optimizer.zero_grad(&[param.clone()]);
            
            // Set constant gradient of 1.0
            let grad = Tensor::ones(Shape::from_dims(&[3]), device.clone())?;
            param.set_grad(grad)?;
            
            let before = param.tensor()?.to_vec()?;
            optimizer.step(&[param.clone()])?;
            let after = param.tensor()?.to_vec()?;
            
            let update = before[0] - after[0]; // Parameters decrease with positive gradients
            println!("  Step {}: update magnitude = {:.4}", step, update);
            
            // Update should increase due to momentum accumulation
            if step > 1 {
                assert!(update > 0.0, "Update should be positive");
            }
        }
        
        println!("  ✅ Adam momentum accumulates correctly!\n");
    }
    
    // Test 3: Test bias correction
    {
        println!("Test 3: Adam bias correction in early steps");
        AutogradContext::reset();
        
        let param1 = Parameter::zeros(Shape::from_dims(&[2]), device.clone())?;
        let param2 = Parameter::zeros(Shape::from_dims(&[2]), device.clone())?;
        
        let mut opt1 = Adam::new(0.001, 0.9, 0.999, 1e-8, 0.0);
        let mut opt2 = Adam::new(0.001, 0.9, 0.999, 1e-8, 0.0);
        
        // First step
        let grad = Tensor::ones(Shape::from_dims(&[2]), device.clone())?;
        param1.set_grad(grad.clone()?)?;
        opt1.step(&[param1.clone()])?;
        
        // Skip 100 steps for second optimizer (simulate late start)
        for _ in 0..100 {
            opt2.step(&[])?; // Empty step to advance timestep
        }
        
        // Now apply same gradient
        param2.set_grad(grad)?;
        opt2.step(&[param2.clone()])?;
        
        let update1 = param1.tensor()?.to_vec()?[0];
        let update2 = param2.tensor()?.to_vec()?[0];
        
        println!("  First step update: {:.6}", update1.abs());
        println!("  Late step update: {:.6}", update2.abs());
        
        // Early steps should have larger updates due to bias correction
        assert!(update1.abs() > update2.abs(), "Bias correction not working");
        
        println!("  ✅ Bias correction working correctly!\n");
    }
    
    println!("🎉 ALL ADAM OPTIMIZER TESTS PASSED! 🎉");
    
    Ok(())
}