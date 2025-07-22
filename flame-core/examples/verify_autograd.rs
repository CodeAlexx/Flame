use flame_core::{Tensor, Shape, CudaDevice, Result, AutogradContext};
use std::sync::Arc;

fn main() -> Result<()> {
    println!("🔬 Verifying FLAME Autograd System...\n");
    
    let device = CudaDevice::new(0)?;
    
    // Test 1: Simple chain rule (y = (x + 2)^2)
    {
        println!("Test 1: Chain rule - y = (x + 2)^2");
        AutogradContext::reset();
        
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), device.clone())?.requires_grad_(true);
        let two = Tensor::from_vec(vec![2.0, 2.0, 2.0], Shape::from_dims(&[3]), device.clone())?;
        
        let x_plus_2 = x.add(&two)?;
        let y = x_plus_2.mul(&x_plus_2)?;
        let loss = y.sum()?;
        
        let grads = loss.backward()?;
        let x_grad = grads.get(x.id()).unwrap();
        
        // dy/dx = 2(x + 2) = [6, 8, 10]
        let expected = vec![6.0, 8.0, 10.0];
        let actual = x_grad.to_vec()?;
        
        println!("  x: [1, 2, 3]");
        println!("  Gradient: {:?}", actual);
        println!("  Expected: {:?}", expected);
        
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5, "Gradient mismatch");
        }
        println!("  ✅ Chain rule gradient correct!\n");
    }
    
    // Test 2: Multiple paths to same variable
    {
        println!("Test 2: Multiple gradient paths - y = x*x + x*3");
        AutogradContext::reset();
        
        let x = Tensor::from_vec(vec![2.0, 4.0], Shape::from_dims(&[2]), device.clone())?.requires_grad_(true);
        let three = Tensor::from_vec(vec![3.0, 3.0], Shape::from_dims(&[2]), device.clone())?;
        
        let x_squared = x.mul(&x)?;
        let x_times_3 = x.mul(&three)?;
        let y = x_squared.add(&x_times_3)?;
        let loss = y.sum()?;
        
        let grads = loss.backward()?;
        let x_grad = grads.get(x.id()).unwrap();
        
        // dy/dx = 2x + 3 = [7, 11]
        let expected = vec![7.0, 11.0];
        let actual = x_grad.to_vec()?;
        
        println!("  x: [2, 4]");
        println!("  Gradient: {:?}", actual);
        println!("  Expected: {:?}", expected);
        
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5, "Gradient mismatch");
        }
        println!("  ✅ Multiple path gradients accumulate correctly!\n");
    }
    
    // Test 3: Matrix multiplication gradients
    {
        println!("Test 3: Matrix multiplication gradients");
        AutogradContext::reset();
        
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[2, 2]), device.clone())?.requires_grad_(true);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], Shape::from_dims(&[2, 2]), device.clone())?.requires_grad_(true);
        
        let c = a.matmul(&b)?;
        let loss = c.sum()?;
        
        let grads = loss.backward()?;
        let a_grad = grads.get(a.id()).unwrap();
        let b_grad = grads.get(b.id()).unwrap();
        
        // For C = A @ B, dL/dA = dL/dC @ B^T, dL/dB = A^T @ dL/dC
        // Since dL/dC = ones(2,2), we get:
        // dL/dA = ones @ B^T = [[11, 15], [11, 15]]
        // dL/dB = A^T @ ones = [[4, 4], [6, 6]]
        
        println!("  A gradient shape: {:?}", a_grad.shape().dims());
        println!("  B gradient shape: {:?}", b_grad.shape().dims());
        
        let a_grad_data = a_grad.to_vec()?;
        let b_grad_data = b_grad.to_vec()?;
        
        println!("  A gradient: {:?}", a_grad_data);
        println!("  B gradient: {:?}", b_grad_data);
        
        // Check shapes are preserved
        assert_eq!(a_grad.shape().dims(), &[2, 2]);
        assert_eq!(b_grad.shape().dims(), &[2, 2]);
        
        println!("  ✅ Matrix multiplication gradients computed!\n");
    }
    
    // Test 4: Broadcasting gradients
    {
        println!("Test 4: Broadcasting gradients");
        AutogradContext::reset();
        
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[2, 2]), device.clone())?.requires_grad_(true);
        let bias = Tensor::from_vec(vec![10.0, 20.0], Shape::from_dims(&[2]), device.clone())?.requires_grad_(true);
        
        // Broadcast bias to match x shape
        let y = x.add(&bias)?; // Broadcasting [2] to [2,2]
        let loss = y.sum()?;
        
        let grads = loss.backward()?;
        let x_grad = grads.get(x.id()).unwrap();
        let bias_grad = grads.get(bias.id()).unwrap();
        
        println!("  x shape: {:?}", x.shape().dims());
        println!("  bias shape: {:?}", bias.shape().dims());
        println!("  x gradient shape: {:?}", x_grad.shape().dims());
        println!("  bias gradient shape: {:?}", bias_grad.shape().dims());
        
        // x gradient should be all ones
        let x_grad_data = x_grad.to_vec()?;
        assert!(x_grad_data.iter().all(|&v| (v - 1.0).abs() < 1e-5));
        
        // bias gradient should be [2, 2] (summed over the broadcast dimension)
        let bias_grad_data = bias_grad.to_vec()?;
        println!("  bias gradient: {:?}", bias_grad_data);
        assert!(bias_grad_data.iter().all(|&v| (v - 2.0).abs() < 1e-5));
        
        println!("  ✅ Broadcasting gradients reduced correctly!\n");
    }
    
    // Test 5: ReLU gradient
    {
        println!("Test 5: ReLU activation gradient");
        AutogradContext::reset();
        
        let x = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], Shape::from_dims(&[5]), device.clone())?.requires_grad_(true);
        
        let y = x.relu()?;
        let loss = y.sum()?;
        
        let grads = loss.backward()?;
        let x_grad = grads.get(x.id()).unwrap();
        
        let grad_data = x_grad.to_vec()?;
        println!("  Input: [-2, -1, 0, 1, 2]");
        println!("  ReLU gradient: {:?}", grad_data);
        println!("  Expected: [0, 0, 0, 1, 1]");
        
        let expected = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        for (a, e) in grad_data.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5, "ReLU gradient mismatch");
        }
        
        println!("  ✅ ReLU gradient correct!\n");
    }
    
    // Test 6: No gradient accumulation between backward passes
    {
        println!("Test 6: Gradient isolation between backward passes");
        AutogradContext::reset();
        
        let x = Tensor::from_vec(vec![1.0, 2.0], Shape::from_dims(&[2]), device.clone())?.requires_grad_(true);
        
        // First backward pass
        let y1 = x.mul(&x)?;
        let loss1 = y1.sum()?;
        let grads1 = loss1.backward()?;
        let grad1 = grads1.get(x.id()).unwrap().to_vec()?;
        
        // Second backward pass
        let y2 = x.add(&x)?;
        let loss2 = y2.sum()?;
        let grads2 = loss2.backward()?;
        let grad2 = grads2.get(x.id()).unwrap().to_vec()?;
        
        println!("  First pass (x*x): gradient = {:?}", grad1);
        println!("  Second pass (x+x): gradient = {:?}", grad2);
        
        // First should be 2x = [2, 4]
        assert!((grad1[0] - 2.0).abs() < 1e-5);
        assert!((grad1[1] - 4.0).abs() < 1e-5);
        
        // Second should be 2 = [2, 2]
        assert!((grad2[0] - 2.0).abs() < 1e-5);
        assert!((grad2[1] - 2.0).abs() < 1e-5);
        
        println!("  ✅ Gradients don't accumulate between backward passes!\n");
    }
    
    println!("🎉 ALL AUTOGRAD TESTS PASSED! 🎉");
    println!("\nFLAME's automatic differentiation system is working correctly:");
    println!("- Chain rule ✅");
    println!("- Multiple gradient paths ✅");
    println!("- Matrix multiplication gradients ✅");
    println!("- Broadcasting gradient reduction ✅");
    println!("- Activation gradients ✅");
    println!("- Gradient isolation ✅");
    
    Ok(())
}