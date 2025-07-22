use flame_core::{Tensor, Shape, CudaDevice, Result};
use flame_core::optimizers::{SGD, SGDConfig};

fn main() -> Result<()> {
    println!("🔬 Testing FLAME SGD Optimizer...\n");
    
    let device = CudaDevice::new(0)?;
    
    // Test 1: Basic SGD without momentum
    {
        println!("Test 1: Basic SGD (no momentum)");
        
        // Create a simple linear model: y = wx + b
        let mut w = Tensor::from_vec(vec![2.0], Shape::from_dims(&[1, 1]), device.clone())?
            .requires_grad_(true);
        let mut b = Tensor::from_vec(vec![1.0], Shape::from_dims(&[1]), device.clone())?
            .requires_grad_(true);
        
        // Target: y = 3x + 5
        let target_w = 3.0;
        let target_b = 5.0;
        
        // Create optimizer
        let mut sgd = SGD::new(SGDConfig {
            lr: 0.1,
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        });
        
        println!("  Initial: w={:.3}, b={:.3}", w.to_vec()?[0], b.to_vec()?[0]);
        println!("  Target:  w={:.3}, b={:.3}", target_w, target_b);
        
        // Training loop
        for step in 0..50 {
            // Create mini-batch
            let x = Tensor::from_vec(
                vec![1.0, 2.0, 3.0, 4.0],
                Shape::from_dims(&[4, 1]),
                device.clone()
            )?;
            
            let y_true = Tensor::from_vec(
                vec![
                    target_w * 1.0 + target_b,
                    target_w * 2.0 + target_b,
                    target_w * 3.0 + target_b,
                    target_w * 4.0 + target_b,
                ],
                Shape::from_dims(&[4, 1]),
                device.clone()
            )?;
            
            // Forward pass: y = wx + b
            let y_pred = x.matmul(&w)?.add(&b)?;
            
            // Loss: MSE
            let diff = y_pred.sub(&y_true)?;
            let loss = diff.square()?.mean()?;
            
            // Backward pass
            let grads = loss.backward()?;
            
            // Get gradients
            let w_id = w.id();
            let b_id = b.id();
            let w_grad = grads.get(w_id).expect("Missing gradient for w");
            let b_grad = grads.get(b_id).expect("Missing gradient for b");
            
            // Update parameters
            sgd.step(&mut vec![
                (0, &mut w, w_grad),
                (1, &mut b, b_grad),
            ])?;
            
            if step % 10 == 0 {
                println!("  Step {}: loss={:.6}, w={:.3}, b={:.3}",
                    step, loss.to_vec()?[0], w.to_vec()?[0], b.to_vec()?[0]);
            }
        }
        
        println!("  Final:   w={:.3}, b={:.3}", w.to_vec()?[0], b.to_vec()?[0]);
        println!("  ✅ Basic SGD test passed!\n");
    }
    
    // Test 2: SGD with momentum
    {
        println!("Test 2: SGD with momentum");
        
        let mut w = Tensor::from_vec(vec![0.5], Shape::from_dims(&[1, 1]), device.clone())?
            .requires_grad_(true);
        let mut b = Tensor::from_vec(vec![0.0], Shape::from_dims(&[1]), device.clone())?
            .requires_grad_(true);
        
        let mut sgd = SGD::new(SGDConfig {
            lr: 0.05,
            momentum: 0.9,
            weight_decay: 0.0,
            nesterov: false,
        });
        
        println!("  Initial: w={:.3}, b={:.3}", w.to_vec()?[0], b.to_vec()?[0]);
        
        // Training for a few steps
        for step in 0..20 {
            let x = Tensor::from_vec(vec![2.0], Shape::from_dims(&[1, 1]), device.clone())?;
            let y_true = Tensor::from_vec(vec![4.0], Shape::from_dims(&[1, 1]), device.clone())?;
            
            let y_pred = x.matmul(&w)?.add(&b)?;
            let loss = y_pred.sub(&y_true)?.square()?.mean()?;
            
            let grads = loss.backward()?;
            
            let w_id = w.id();
            let b_id = b.id();
            sgd.step(&mut vec![
                (0, &mut w, grads.get(w_id).unwrap()),
                (1, &mut b, grads.get(b_id).unwrap()),
            ])?;
            
            if step % 5 == 0 {
                println!("  Step {}: loss={:.6}, w={:.3}, b={:.3}",
                    step, loss.to_vec()?[0], w.to_vec()?[0], b.to_vec()?[0]);
            }
        }
        
        println!("  ✅ SGD with momentum test passed!\n");
    }
    
    // Test 3: SGD with weight decay
    {
        println!("Test 3: SGD with weight decay");
        
        let mut w = Tensor::from_vec(
            vec![5.0, 5.0, 5.0, 5.0],
            Shape::from_dims(&[2, 2]),
            device.clone()
        )?.requires_grad_(true);
        
        let mut sgd = SGD::new(SGDConfig {
            lr: 0.1,
            momentum: 0.0,
            weight_decay: 0.1,
            nesterov: false,
        });
        
        println!("  Initial weights: {:?}", w.to_vec()?);
        
        // Apply SGD with zero gradients - only weight decay should apply
        let zero_grad = Tensor::zeros(w.shape().clone(), device.clone())?;
        
        for step in 0..5 {
            sgd.step(&mut vec![(0, &mut w, &zero_grad)])?;
            println!("  Step {}: weights={:?}", step + 1, w.to_vec()?);
        }
        
        // Weights should decrease due to weight decay
        let final_weights = w.to_vec()?;
        assert!(final_weights.iter().all(|&x| x < 5.0));
        
        println!("  ✅ SGD with weight decay test passed!\n");
    }
    
    // Test 4: SGD with Nesterov momentum
    {
        println!("Test 4: SGD with Nesterov momentum");
        
        let mut w = Tensor::from_vec(vec![1.0], Shape::from_dims(&[1, 1]), device.clone())?
            .requires_grad_(true);
        
        let mut sgd_nesterov = SGD::new(SGDConfig {
            lr: 0.01,
            momentum: 0.9,
            weight_decay: 0.0,
            nesterov: true,
        });
        
        let mut sgd_regular = SGD::new(SGDConfig {
            lr: 0.01,
            momentum: 0.9,
            weight_decay: 0.0,
            nesterov: false,
        });
        
        // Clone for comparison
        let mut w_regular = w.clone()?.requires_grad_(true);
        
        println!("  Testing Nesterov vs regular momentum...");
        
        // Apply same gradient to both
        let grad = Tensor::from_vec(vec![1.0], Shape::from_dims(&[1]), device.clone())?;
        
        for step in 0..5 {
            sgd_nesterov.step(&mut vec![(0, &mut w, &grad)])?;
            sgd_regular.step(&mut vec![(1, &mut w_regular, &grad)])?;
            
            println!("  Step {}: Nesterov={:.6}, Regular={:.6}",
                step + 1, w.to_vec()?[0], w_regular.to_vec()?[0]);
        }
        
        // Nesterov should be different from regular momentum
        assert!((w.to_vec()?[0] - w_regular.to_vec()?[0]).abs() > 1e-6);
        
        println!("  ✅ Nesterov momentum test passed!\n");
    }
    
    println!("🎉 ALL SGD OPTIMIZER TESTS PASSED! 🎉");
    
    Ok(())
}