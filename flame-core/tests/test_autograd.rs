use flame_core::{Tensor, Shape, AutogradContext};
use cudarc::driver::CudaDevice;

#[test]
fn test_simple_gradient() -> flame_core::Result<()> {
    let device = CudaDevice::new(0)?;
    AutogradContext::clear();
    
    // y = x^2, dy/dx = 2x
    let x = Tensor::from_slice(&[3.0], Shape::from_dims(&[1]), device)?.requires_grad_(true);
    let y = x.mul(&x)?;
    
    let grads = y.backward()?;
    let x_grad = grads.get(x.id()).unwrap();
    
    let grad_val = x_grad.to_vec()?[0];
    assert!((grad_val - 6.0).abs() < 1e-5, "Expected gradient 6.0, got {}", grad_val);
    
    Ok(())
}

#[test]
fn test_chain_rule() -> flame_core::Result<()> {
    let device = CudaDevice::new(0)?;
    AutogradContext::clear();
    
    // y = (x + 2)^2, dy/dx = 2(x + 2)
    let x = Tensor::from_slice(&[1.0], Shape::from_dims(&[1]), device)?.requires_grad_(true);
    let x_plus_2 = x.add_scalar(2.0)?;
    let y = x_plus_2.mul(&x_plus_2)?;
    
    let grads = y.backward()?;
    let x_grad = grads.get(x.id()).unwrap();
    
    let grad_val = x_grad.to_vec()?[0];
    assert!((grad_val - 6.0).abs() < 1e-5, "Expected gradient 6.0, got {}", grad_val);
    
    Ok(())
}

#[test]
fn test_multiple_paths() -> flame_core::Result<()> {
    let device = CudaDevice::new(0)?;
    AutogradContext::clear();
    
    // z = x*x + x*y, dz/dx = 2x + y, dz/dy = x
    let x = Tensor::from_slice(&[3.0], Shape::from_dims(&[1]), device.clone())?.requires_grad_(true);
    let y = Tensor::from_slice(&[4.0], Shape::from_dims(&[1]), device)?.requires_grad_(true);
    
    let xx = x.mul(&x)?;
    let xy = x.mul(&y)?;
    let z = xx.add(&xy)?;
    
    let grads = z.backward()?;
    
    let x_grad = grads.get(x.id()).unwrap().to_vec()?[0];
    let y_grad = grads.get(y.id()).unwrap().to_vec()?[0];
    
    // dz/dx = 2*3 + 4 = 10
    assert!((x_grad - 10.0).abs() < 1e-5, "Expected x gradient 10.0, got {}", x_grad);
    // dz/dy = 3
    assert!((y_grad - 3.0).abs() < 1e-5, "Expected y gradient 3.0, got {}", y_grad);
    
    Ok(())
}

#[test]
fn test_matmul_gradient() -> flame_core::Result<()> {
    let device = CudaDevice::new(0)?;
    AutogradContext::clear();
    
    // Simple 2x2 matrix multiplication
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[2, 2]), device.clone())?.requires_grad_(true);
    let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], Shape::from_dims(&[2, 2]), device)?.requires_grad_(true);
    
    let c = a.matmul(&b)?;
    let loss = c.sum()?;
    
    let grads = loss.backward()?;
    
    // Gradient of A should be B^T summed
    let a_grad = grads.get(a.id()).unwrap().to_vec()?;
    // B^T = [[5, 7], [6, 8]], summed across output positions
    assert!((a_grad[0] - 11.0).abs() < 1e-5); // 5 + 6
    assert!((a_grad[1] - 15.0).abs() < 1e-5); // 7 + 8
    assert!((a_grad[2] - 11.0).abs() < 1e-5); // 5 + 6
    assert!((a_grad[3] - 15.0).abs() < 1e-5); // 7 + 8
    
    Ok(())
}

#[test]
fn test_relu_gradient() -> flame_core::Result<()> {
    let device = CudaDevice::new(0)?;
    AutogradContext::clear();
    
    let x = Tensor::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0], Shape::from_dims(&[5]), device)?.requires_grad_(true);
    let y = x.relu()?;
    let loss = y.sum()?;
    
    let grads = loss.backward()?;
    let x_grad = grads.get(x.id()).unwrap().to_vec()?;
    
    // ReLU gradient is 0 for negative values, 1 for positive
    assert_eq!(x_grad[0], 0.0); // x=-2
    assert_eq!(x_grad[1], 0.0); // x=-1
    assert_eq!(x_grad[2], 0.0); // x=0
    assert_eq!(x_grad[3], 1.0); // x=1
    assert_eq!(x_grad[4], 1.0); // x=2
    
    Ok(())
}

#[test]
fn test_sum_broadcast_gradient() -> flame_core::Result<()> {
    let device = CudaDevice::new(0)?;
    AutogradContext::clear();
    
    // Test that sum properly broadcasts gradients back
    let x = Tensor::ones(Shape::from_dims(&[3, 4]), device)?.requires_grad_(true);
    let sum_x = x.sum()?;
    
    let grads = sum_x.backward()?;
    let x_grad = grads.get(x.id()).unwrap();
    
    // Gradient should be all 1s (broadcasted from scalar 1)
    let grad_values = x_grad.to_vec()?;
    for val in grad_values {
        assert!((val - 1.0).abs() < 1e-5);
    }
    
    Ok(())
}

#[test] 
fn test_mean_gradient() -> flame_core::Result<()> {
    let device = CudaDevice::new(0)?;
    AutogradContext::clear();
    
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[4]), device)?.requires_grad_(true);
    let mean_x = x.mean()?;
    
    let grads = mean_x.backward()?;
    let x_grad = grads.get(x.id()).unwrap().to_vec()?;
    
    // Mean gradient is 1/n for each element
    for val in x_grad {
        assert!((val - 0.25).abs() < 1e-5);
    }
    
    Ok(())
}