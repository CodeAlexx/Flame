use flame_core::{Tensor, Shape, CudaDevice, Result, AutogradContext, TensorGradExt};

fn main() -> Result<()> {
    println!("Testing basic autograd functionality...\n");
    
    let device = CudaDevice::new(0)?;
    
    // Reset autograd
    AutogradContext::reset();
    
    // Test 1: Simple scalar multiplication
    {
        println!("Test 1: Scalar multiplication");
        let x = Tensor::from_vec(vec![2.0, 3.0], Shape::from_dims(&[2]), device.clone())?
            .requires_grad_(true);
        
        let y = x.mul_scalar(2.0)?;
        let loss = y.sum()?;
        
        let grads = AutogradContext::backward(&loss)?;
        
        if let Some(x_grad) = x.grad(&grads) {
            println!("✅ x gradient: {:?}", x_grad.to_vec()?);
        } else {
            println!("❌ No gradient for x!");
        }
    }
    
    // Test 2: Addition
    {
        println!("\nTest 2: Addition");
        AutogradContext::reset();
        
        let a = Tensor::from_vec(vec![1.0, 2.0], Shape::from_dims(&[2]), device.clone())?
            .requires_grad_(true);
        let b = Tensor::from_vec(vec![3.0, 4.0], Shape::from_dims(&[2]), device.clone())?
            .requires_grad_(true);
        
        let c = a.add(&b)?;
        let loss = c.sum()?;
        
        let grads = AutogradContext::backward(&loss)?;
        
        if let Some(a_grad) = a.grad(&grads) {
            println!("✅ a gradient: {:?}", a_grad.to_vec()?);
        } else {
            println!("❌ No gradient for a!");
        }
        
        if let Some(b_grad) = b.grad(&grads) {
            println!("✅ b gradient: {:?}", b_grad.to_vec()?);
        } else {
            println!("❌ No gradient for b!");
        }
    }
    
    // Test 3: Matrix multiplication
    {
        println!("\nTest 3: Matrix multiplication");
        AutogradContext::reset();
        
        let a = Tensor::from_vec(vec![1.0, 2.0], Shape::from_dims(&[1, 2]), device.clone())?
            .requires_grad_(true);
        let b = Tensor::from_vec(vec![3.0, 4.0], Shape::from_dims(&[2, 1]), device.clone())?
            .requires_grad_(true);
        
        let c = a.matmul(&b)?;
        let loss = c.sum()?;
        
        let grads = AutogradContext::backward(&loss)?;
        
        println!("Number of gradients: {}", grads.len());
        
        if let Some(a_grad) = a.grad(&grads) {
            println!("✅ a gradient: {:?}", a_grad.to_vec()?);
        } else {
            println!("❌ No gradient for a!");
        }
        
        if let Some(b_grad) = b.grad(&grads) {
            println!("✅ b gradient: {:?}", b_grad.to_vec()?);
        } else {
            println!("❌ No gradient for b!");
        }
    }
    
    Ok(())
}