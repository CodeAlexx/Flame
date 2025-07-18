use flame_core::{Shape, Result, FlameError};
use flame_core::tensor_autograd::Tensor;
use cudarc::driver::CudaDevice;
use std::sync::Arc;

fn main() -> Result<()> {
    println!("Testing FLAME Autograd Implementation");
    println!("=====================================\n");
    
    // Initialize CUDA device
    let device = Arc::new(CudaDevice::new(0)?);
    
    // Test 1: Simple gradient flow
    println!("Test 1: Simple gradient flow (y = 2x, dy/dx = 2)");
    {
        let x = Tensor::from_vec(vec![3.0], Shape::from_dims(&[1]), device.clone())?
            .requires_grad_(true);
        
        let y = x.mul_scalar(2.0)?;
        println!("x = 3.0, y = 2x = {}", y.item()?);
        
        y.backward()?;
        
        // Get gradient from autograd engine
        if let Some(x_id) = x.graph_id {
            if let Some(grad) = flame_core::autograd_simple::get_grad(x_id) {
                println!("dy/dx = {}", grad.item()?);
                assert!((grad.item()? - 2.0).abs() < 1e-5, "Expected gradient 2.0");
            } else {
                println!("ERROR: No gradient found for x");
            }
        }
        
        flame_core::autograd_simple::clear_graph();
    }
    
    // Test 2: Multi-path gradient
    println!("\nTest 2: Multi-path gradient (z = x + x^2)");
    {
        let x = Tensor::from_vec(vec![3.0], Shape::from_dims(&[1]), device.clone())?
            .requires_grad_(true);
        
        let x_squared = x.square()?;
        let z = x.add(&x_squared)?;
        println!("x = 3.0, z = x + x^2 = {}", z.item()?);
        
        z.backward()?;
        
        // dz/dx = 1 + 2x = 1 + 6 = 7
        if let Some(x_id) = x.graph_id {
            if let Some(grad) = flame_core::autograd_simple::get_grad(x_id) {
                println!("dz/dx = 1 + 2x = {}", grad.item()?);
                assert!((grad.item()? - 7.0).abs() < 1e-5, "Expected gradient 7.0");
            }
        }
        
        flame_core::autograd_simple::clear_graph();
    }
    
    // Test 3: Mean reduction
    println!("\nTest 3: Mean reduction");
    {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[4]), device.clone())?
            .requires_grad_(true);
        
        let mean = x.mean()?;
        println!("x = [1, 2, 3, 4], mean = {}", mean.item()?);
        
        mean.backward()?;
        
        // d(mean)/dx = 1/n = 0.25 for each element
        if let Some(x_id) = x.graph_id {
            if let Some(grad) = flame_core::autograd_simple::get_grad(x_id) {
                let grad_vec = grad.to_vec()?;
                println!("d(mean)/dx = {:?}", grad_vec);
                for g in grad_vec {
                    assert!((g - 0.25).abs() < 1e-5, "Expected gradient 0.25");
                }
            }
        }
        
        flame_core::autograd_simple::clear_graph();
    }
    
    // Test 4: Matrix multiplication
    println!("\nTest 4: Matrix multiplication");
    {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[2, 2]), device.clone())?
            .requires_grad_(true);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], Shape::from_dims(&[2, 2]), device.clone())?
            .requires_grad_(true);
        
        let c = a.matmul(&b)?;
        let loss = c.sum()?;
        
        println!("A = [[1, 2], [3, 4]]");
        println!("B = [[5, 6], [7, 8]]");
        println!("C = A @ B = [[19, 22], [43, 50]]");
        println!("loss = sum(C) = {}", loss.item()?);
        
        loss.backward()?;
        
        // Check gradients
        if let Some(a_id) = a.graph_id {
            if let Some(grad_a) = flame_core::autograd_simple::get_grad(a_id) {
                println!("dL/dA = {:?}", grad_a.to_vec()?);
                // dL/dA = 1 @ B^T = [[11, 15], [11, 15]]
                let expected_a = vec![11.0, 15.0, 11.0, 15.0];
                let actual_a = grad_a.to_vec()?;
                for (e, a) in expected_a.iter().zip(actual_a.iter()) {
                    assert!((e - a).abs() < 1e-4, "Gradient mismatch");
                }
            }
        }
        
        flame_core::autograd_simple::clear_graph();
    }
    
    // Test 5: Activation functions
    println!("\nTest 5: Activation functions");
    {
        let x = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], Shape::from_dims(&[4]), device.clone())?
            .requires_grad_(true);
        
        // Test ReLU
        let relu_out = x.relu()?;
        let relu_sum = relu_out.sum()?;
        relu_sum.backward()?;
        
        if let Some(x_id) = x.graph_id {
            if let Some(grad) = flame_core::autograd_simple::get_grad(x_id) {
                println!("ReLU gradient: {:?}", grad.to_vec()?);
                // Expected: [0, 0, 1, 1] (derivative is 0 for x<=0, 1 for x>0)
                let expected = vec![0.0, 0.0, 1.0, 1.0];
                let actual = grad.to_vec()?;
                for (e, a) in expected.iter().zip(actual.iter()) {
                    assert!((e - a).abs() < 1e-4, "ReLU gradient mismatch");
                }
            }
        }
        
        flame_core::autograd_simple::clear_graph();
    }
    
    println!("\nAll tests passed! ✅");
    println!("\nAutograd is working correctly!");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_autograd() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        
        let x = Tensor::from_vec(vec![2.0], Shape::from_dims(&[1]), device)?
            .requires_grad_(true);
        
        let y = x.mul_scalar(3.0)?;
        y.backward()?;
        
        if let Some(x_id) = x.graph_id {
            if let Some(grad) = flame_core::autograd_simple::get_grad(x_id) {
                assert!((grad.item()? - 3.0).abs() < 1e-5);
            }
        }
        
        Ok(())
    }
}