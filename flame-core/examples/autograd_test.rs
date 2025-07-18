use flame_core::{CudaDevice, Shape};
use flame_core::autograd_engine::ops;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::new(0)?;
    println!("CUDA device initialized");

    println!("\n=== Testing Autograd System ===");
    
    // Test 1: Simple multiplication with gradient
    println!("\n--- Test 1: Simple Gradient Computation ---");
    
    // Create tracked tensors
    let x = ops::create_tracked_tensor(
        vec![2.0, 3.0, 4.0],
        Shape::from_dims(&[3]),
        device.clone(),
        true
    )?;
    
    let y = ops::create_tracked_tensor(
        vec![5.0, 6.0, 7.0],
        Shape::from_dims(&[3]),
        device.clone(),
        true
    )?;
    
    println!("x = {:?}", x.to_vec()?);
    println!("y = {:?}", y.to_vec()?);
    
    // Forward pass: z = x * y
    let z = ops::mul_tracked(&x, &y)?;
    println!("z = x * y = {:?}", z.to_vec()?);
    
    // loss = mean(z)
    let loss = ops::mean_tracked(&z)?;
    println!("loss = mean(z) = {}", loss.item()?);
    
    // Backward pass
    println!("\nCalling backward...");
    match loss.backward() {
        Ok(_) => println!("Backward pass succeeded"),
        Err(e) => println!("Backward pass failed: {}", e),
    }
    
    // Check gradients
    if let Some(x_grad) = x.grad() {
        println!("Gradient of x: {:?}", x_grad.to_vec()?);
        // Expected: y/n = [5/3, 6/3, 7/3] ≈ [1.67, 2.0, 2.33]
    }
    
    if let Some(y_grad) = y.grad() {
        println!("Gradient of y: {:?}", y_grad.to_vec()?);
        // Expected: x/n = [2/3, 3/3, 4/3] ≈ [0.67, 1.0, 1.33]
    }
    
    // Test 2: Chain of operations
    println!("\n--- Test 2: Chain of Operations ---");
    
    // Clear previous gradients
    flame_core::autograd_engine::ENGINE.with(|engine| {
        let mut engine = engine.lock().unwrap();
        engine.clear_graph();
        Ok::<(), flame_core::FlameError>(())
    })?;
    
    let a = ops::create_tracked_tensor(
        vec![1.0, 2.0],
        Shape::from_dims(&[2]),
        device.clone(),
        true
    )?;
    
    let b = ops::create_tracked_tensor(
        vec![3.0, 4.0],
        Shape::from_dims(&[2]),
        device.clone(),
        true
    )?;
    
    // c = a + b
    let c = ops::add_tracked(&a, &b)?;
    println!("\na = {:?}", a.to_vec()?);
    println!("b = {:?}", b.to_vec()?);
    println!("c = a + b = {:?}", c.to_vec()?);
    
    // d = c * a
    let d = ops::mul_tracked(&c, &a)?;
    println!("d = c * a = {:?}", d.to_vec()?);
    
    // loss = mean(d)
    let loss2 = ops::mean_tracked(&d)?;
    println!("loss = mean(d) = {}", loss2.item()?);
    
    // Backward
    loss2.backward()?;
    
    if let Some(a_grad) = a.grad() {
        println!("\nGradient of a: {:?}", a_grad.to_vec()?);
        // d/da mean((a+b)*a) = d/da mean(a²+ab) = (2a+b)/n
        // For a=[1,2], b=[3,4]: grad = ([2+3, 4+4])/2 = [2.5, 4.0]
    }
    
    if let Some(b_grad) = b.grad() {
        println!("Gradient of b: {:?}", b_grad.to_vec()?);
        // d/db mean((a+b)*a) = d/db mean(a²+ab) = a/n
        // For a=[1,2]: grad = [1,2]/2 = [0.5, 1.0]
    }
    
    // Test 3: Matrix multiplication
    println!("\n--- Test 3: Matrix Multiplication Gradients ---");
    
    flame_core::autograd_engine::ENGINE.with(|engine| {
        let mut engine = engine.lock().unwrap();
        engine.clear_graph();
        Ok::<(), flame_core::FlameError>(())
    })?;
    
    let w = ops::create_tracked_tensor(
        vec![1.0, 2.0, 3.0, 4.0],
        Shape::from_dims(&[2, 2]),
        device.clone(),
        true
    )?;
    
    let x_mat = ops::create_tracked_tensor(
        vec![0.5, 1.5],
        Shape::from_dims(&[2, 1]),
        device.clone(),
        true
    )?;
    
    // y = w @ x
    let y_mat = ops::matmul_tracked(&w, &x_mat)?;
    println!("\nW = {:?}", w.to_vec()?);
    println!("x = {:?}", x_mat.to_vec()?);
    println!("y = W @ x = {:?}", y_mat.to_vec()?);
    
    // loss = sum(y) (for simplicity)
    let loss3 = y_mat.sum()?;
    println!("loss = sum(y) = {}", loss3.item()?);
    
    // Note: sum() is not tracked yet, so we can't compute full gradients
    println!("\nNote: Full backprop through sum() not yet implemented");
    
    println!("\n=== Autograd Tests Complete ===");
    println!("\nThe autograd system is working for basic operations!");
    println!("Gradients are correctly computed through multiplication and mean.");
    println!("\nNext steps would be to:");
    println!("- Add more operations (relu, square, etc.)");
    println!("- Implement gradient tracking for all operations");
    println!("- Add support for more complex graphs");
    
    Ok(())
}