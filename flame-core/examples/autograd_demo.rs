use flame_core::{Tensor, CudaDevice, Shape};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA
    let device = CudaDevice::new(0)?;
    println!("CUDA device initialized");

    // Create tensors with gradient tracking
    let x = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0],
        Shape::from_dims(&[2, 2]),
        device.clone()
    )?.requires_grad_(true);
    
    let w = Tensor::from_vec(
        vec![0.5, -0.5, 1.0, -1.0],
        Shape::from_dims(&[2, 2]),
        device.clone()
    )?.requires_grad_(true);
    
    println!("Created input tensors with gradient tracking");
    
    // Forward pass
    let y = x.matmul(&w)?;
    println!("Forward pass: y = x @ w");
    println!("y = {:?}", y.to_vec()?);
    
    // Compute loss (sum of squares)
    let loss = y.square()?.sum()?;
    println!("Loss = sum(y^2) = {}", loss.item()?);
    
    // Manual gradient computation for demonstration
    // d(loss)/d(y) = 2*y
    let y_data = y.to_vec()?;
    let grad_y = Tensor::from_vec(
        y_data.iter().map(|&v| 2.0 * v).collect(),
        y.shape().clone(),
        device.clone()
    )?;
    
    // d(loss)/d(w) = x^T @ d(loss)/d(y)
    let x_t = x.transpose()?;
    let grad_w = x_t.matmul(&grad_y)?;
    
    println!("\nManual gradient computation:");
    println!("d(loss)/d(w) = {:?}", grad_w.to_vec()?);
    
    // Update weights - create a copy for the updated version
    let w_data = w.to_vec()?;
    let mut w_updated = Tensor::from_vec(w_data, w.shape().clone(), device.clone())?;
    w_updated.update_weights(&grad_w, 0.1)?;
    
    println!("\nWeight update with learning rate 0.1:");
    println!("Original w: {:?}", w.to_vec()?);
    println!("Updated w: {:?}", w_updated.to_vec()?);
    
    // Note: Full autograd with backward() would be:
    // loss.backward()?;
    // let grad = w.grad().unwrap();
    // w.update_weights(grad, 0.1)?;
    
    println!("\nAutograd demo completed!");
    println!("Note: Full automatic differentiation is partially implemented.");
    println!("The computation graph tracks operations, but tensor tracking needs completion.");
    
    Ok(())
}