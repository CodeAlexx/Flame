//! Test automatic differentiation in FLAME

use flame_core::{Tensor, Shape, CudaDevice, AutogradContext};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA
    let device = Arc::new(CudaDevice::new(0)?);
    println!("Testing FLAME automatic differentiation...\n");
    
    // Test 1: Simple gradient flow
    println!("Test 1: Simple gradient flow (y = x^2, dy/dx = 2x)");
    {
        // Clear any previous computation graph
        AutogradContext::clear();
        
        let x = Tensor::from_vec(vec![3.0], Shape::from_dims(&[1]), device.clone())?
            .requires_grad_(true);
        
        let y = x.square()?;
        
        println!("x = 3.0");
        println!("y = x^2 = {}", y.item()?);
        
        // Compute gradients
        let grads = y.backward()?;
        
        if let Some(x_grad) = grads.get(x.id) {
            println!("dy/dx = 2x = {}", x_grad.item()?);
            println!("Expected: 6.0\n");
        }
    }
    
    // Test 2: Chain rule
    println!("Test 2: Chain rule (y = (x + 2)^2)");
    {
        AutogradContext::clear();
        
        let x = Tensor::from_vec(vec![3.0], Shape::from_dims(&[1]), device.clone())?
            .requires_grad_(true);
        
        let z = x.add_scalar(2.0)?;  // z = x + 2 = 5
        let y = z.square()?;          // y = z^2 = 25
        
        println!("x = 3.0");
        println!("z = x + 2 = {}", z.item()?);
        println!("y = z^2 = {}", y.item()?);
        
        let grads = y.backward()?;
        
        if let Some(x_grad) = grads.get(x.id) {
            println!("dy/dx = 2(x+2) = {}", x_grad.item()?);
            println!("Expected: 10.0\n");
        }
    }
    
    // Test 3: Multiple inputs
    println!("Test 3: Multiple inputs (z = x * y)");
    {
        AutogradContext::clear();
        
        let x = Tensor::from_vec(vec![3.0], Shape::from_dims(&[1]), device.clone())?
            .requires_grad_(true);
        let y = Tensor::from_vec(vec![4.0], Shape::from_dims(&[1]), device.clone())?
            .requires_grad_(true);
        
        let z = x.mul(&y)?;
        
        println!("x = 3.0, y = 4.0");
        println!("z = x * y = {}", z.item()?);
        
        let grads = z.backward()?;
        
        if let Some(x_grad) = grads.get(x.id) {
            println!("dz/dx = y = {}", x_grad.item()?);
        }
        if let Some(y_grad) = grads.get(y.id) {
            println!("dz/dy = x = {}", y_grad.item()?);
        }
        println!();
    }
    
    // Test 4: Matrix multiplication
    println!("Test 4: Matrix multiplication gradients");
    {
        AutogradContext::clear();
        
        let a = Tensor::randn(Shape::from_dims(&[2, 3]), 0.0, 1.0, device.clone())?
            .requires_grad_(true);
        let b = Tensor::randn(Shape::from_dims(&[3, 4]), 0.0, 1.0, device.clone())?
            .requires_grad_(true);
        
        let c = a.matmul(&b)?;  // [2, 4]
        let loss = c.sum()?;    // scalar
        
        println!("A shape: {:?}", a.shape().dims());
        println!("B shape: {:?}", b.shape().dims());
        println!("C = A @ B shape: {:?}", c.shape().dims());
        println!("loss = sum(C) = {}", loss.item()?);
        
        let grads = loss.backward()?;
        
        if let Some(a_grad) = grads.get(a.id) {
            println!("grad_A shape: {:?}", a_grad.shape().dims());
        }
        if let Some(b_grad) = grads.get(b.id) {
            println!("grad_B shape: {:?}", b_grad.shape().dims());
        }
        println!();
    }
    
    // Test 5: Activation functions
    println!("Test 5: Activation function gradients");
    {
        AutogradContext::clear();
        
        let x = Tensor::from_vec(vec![0.5, -0.5, 0.0], Shape::from_dims(&[3]), device.clone())?
            .requires_grad_(true);
        
        // Test ReLU
        let y_relu = x.relu()?;
        let loss_relu = y_relu.sum()?;
        
        println!("Testing ReLU:");
        println!("x = [0.5, -0.5, 0.0]");
        println!("relu(x) = {:?}", y_relu.to_vec()?);
        
        let grads = loss_relu.backward()?;
        if let Some(x_grad) = grads.get(x.id) {
            println!("d(relu)/dx = {:?}", x_grad.to_vec()?);
            println!("Expected: [1.0, 0.0, 0.0]\n");
        }
        
        // Clear for next test
        AutogradContext::clear();
        
        // Test Sigmoid
        let x2 = Tensor::from_vec(vec![0.0], Shape::from_dims(&[1]), device.clone())?
            .requires_grad_(true);
        let y_sigmoid = x2.sigmoid()?;
        
        println!("Testing Sigmoid:");
        println!("x = 0.0");
        println!("sigmoid(x) = {}", y_sigmoid.item()?);
        
        let grads = y_sigmoid.backward()?;
        if let Some(x_grad) = grads.get(x2.id) {
            println!("d(sigmoid)/dx at x=0 = {}", x_grad.item()?);
            println!("Expected: 0.25 (sigmoid(0) * (1 - sigmoid(0)))\n");
        }
    }
    
    // Test 6: no_grad context
    println!("Test 6: no_grad context");
    {
        AutogradContext::clear();
        
        let x = Tensor::from_vec(vec![2.0], Shape::from_dims(&[1]), device.clone())?
            .requires_grad_(true);
        
        // Operations in no_grad context shouldn't be recorded
        let y = {
            let _guard = AutogradContext::no_grad();
            x.square()?
        };
        
        // This should work even though y was created in no_grad
        let z = y.add_scalar(1.0)?;
        
        println!("x = 2.0 (requires_grad=true)");
        println!("In no_grad: y = x^2 = {}", y.item()?);
        println!("z = y + 1 = {}", z.item()?);
        println!("y.requires_grad = {}", y.requires_grad);
        println!("z.requires_grad = {}", z.requires_grad);
    }
    
    println!("\nAll autograd tests completed successfully!");
    Ok(())
}