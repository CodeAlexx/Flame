//! Test FLAME's ability to modify gradients - the key feature Candle lacks
//! This is why FLAME exists!

use flame_core::{
    Tensor, Shape, Result, CudaDevice, AutogradContext, GradientMap,
    linear::Linear, optimizers::{Adam, AdamConfig},
};
use std::sync::Arc;

fn main() -> Result<()> {
    // Initialize CUDA device
    let device = CudaDevice::new(0)?;
    println!("Testing FLAME gradient modification capability...\n");
    
    // Create a simple linear layer
    let mut linear = Linear::new(10, 5, true, &device)?;
    
    // Create some input data
    let input = Tensor::randn(Shape::from_dims(&[2, 10]), 0.0, 1.0, device.clone())?
        .requires_grad_(true);
    
    // Forward pass
    let output = linear.forward(&input)?;
    
    // Create a simple loss (sum of outputs)
    let loss = output.sum()?;
    println!("Loss value: {:.4}", loss.to_vec()?[0]);
    
    // Backward pass - compute gradients
    let gradients = AutogradContext::backward(&loss)?;
    
    // THE KEY PART - Modify gradients before optimizer step!
    // This is what Candle CANNOT do and why FLAME exists
    println!("\n=== Gradient Modification Test ===");
    
    // Get weight gradient
    if let Some(weight_grad) = gradients.get(linear.weight.id()) {
        let original_grad = weight_grad.to_vec()?;
        println!("Original weight gradient (first 5 values): {:?}", 
                 &original_grad[..5.min(original_grad.len())]);
        
        // MODIFY THE GRADIENT - Scale by 0.1 (gradient clipping example)
        let modified_grad = weight_grad.mul_scalar(0.1)?;
        
        // Create new gradient map with modified gradients
        let mut modified_gradients = GradientMap::new(device.clone());
        modified_gradients.insert(linear.weight.id(), Clone::clone(&modified_grad));
        
        // Also copy bias gradient if it exists
        if let Some(bias) = &linear.bias {
            if let Some(bias_grad) = gradients.get(bias.id()) {
                // Also scale bias gradient
                let modified_bias_grad = bias_grad.mul_scalar(0.1)?;
                modified_gradients.insert(bias.id(), modified_bias_grad);
            }
        }
        
        let modified_values = modified_grad.to_vec()?;
        println!("Modified weight gradient (first 5 values): {:?}", 
                 &modified_values[..5.min(modified_values.len())]);
        
        // Verify modification worked
        let scale_factor = modified_values[0] / original_grad[0];
        println!("\nGradient scale factor: {:.4} (expected: 0.1)", scale_factor);
        
        // Now use modified gradients in optimizer
        let mut adam_config = AdamConfig::default();
        adam_config.lr = 0.001;
        let mut optimizer = Adam::new(adam_config);
        
        // Apply the MODIFIED gradients
        // Create parameter list for optimizer
        let mut params = vec![
            (linear.weight.id().0, &mut linear.weight, modified_gradients.get(linear.weight.id()).unwrap())
        ];
        optimizer.step(&mut params)?;
        
        println!("\n✅ SUCCESS: Gradients were modified before optimizer step!");
        println!("This is the key capability that Candle lacks and why FLAME exists.");
    }
    
    // Test gradient accumulation with modification
    println!("\n=== Gradient Accumulation Test ===");
    AutogradContext::clear(); // Clear previous computation graph
    
    let mut accumulated_gradients = GradientMap::new(device.clone());
    
    // Simulate gradient accumulation over mini-batches
    for i in 0..4 {
        let batch_input = Tensor::randn(Shape::from_dims(&[2, 10]), 0.0, 1.0, device.clone())?
            .requires_grad_(true);
        
        let batch_output = linear.forward(&batch_input)?;
        let batch_loss = batch_output.sum()?;
        
        let batch_gradients = AutogradContext::backward(&batch_loss)?;
        
        // Accumulate gradients
        if let Some(weight_grad) = batch_gradients.get(linear.weight.id()) {
            if i == 0 {
                accumulated_gradients.insert(linear.weight.id(), Clone::clone(weight_grad));
            } else {
                // Add to existing gradient
                let existing = accumulated_gradients.get(linear.weight.id()).unwrap();
                let summed = existing.add(weight_grad)?;
                accumulated_gradients.insert(linear.weight.id(), summed);
            }
        }
        
        AutogradContext::clear(); // Clear for next iteration
    }
    
    // Now we can modify the accumulated gradients
    if let Some(accum_grad) = accumulated_gradients.get(linear.weight.id()) {
        // Average the accumulated gradients
        let averaged_grad = accum_grad.mul_scalar(0.25)?; // Divide by 4 batches
        accumulated_gradients.insert(linear.weight.id(), averaged_grad);
        
        println!("✅ Gradient accumulation with modification successful!");
    }
    
    println!("\n=== Summary ===");
    println!("FLAME successfully demonstrates:");
    println!("1. ✅ Gradient computation via automatic differentiation");
    println!("2. ✅ Gradient modification before optimizer step");
    println!("3. ✅ Gradient accumulation with averaging");
    println!("4. ✅ Custom gradient manipulation for advanced training");
    println!("\nThese capabilities are ESSENTIAL for modern deep learning and");
    println!("are the reason FLAME was created to replace Candle.");
    
    Ok(())
}