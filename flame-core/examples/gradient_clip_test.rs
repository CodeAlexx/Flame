use flame_core::{Tensor, CudaDevice, Shape};
use flame_core::gradient_clip::{GradientClipper, GradientNormTracker, LayerWiseGradientClipper};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::new(0)?;
    println!("CUDA device initialized");

    println!("\n=== Testing Gradient Clipping ===");
    
    // Test 1: Clip by norm
    println!("\n--- Test 1: Gradient Clipping by Norm ---");
    
    // Create some large gradients
    let mut grad1 = Tensor::from_vec(
        vec![10.0, 20.0, 30.0, 40.0],
        Shape::from_dims(&[2, 2]),
        device.clone()
    )?;
    
    let mut grad2 = Tensor::from_vec(
        vec![5.0, 10.0, 15.0, 20.0],
        Shape::from_dims(&[2, 2]),
        device.clone()
    )?;
    
    let mut grad3 = Tensor::from_vec(
        vec![50.0, 60.0, 70.0, 80.0],
        Shape::from_dims(&[2, 2]),
        device.clone()
    )?;
    
    let clipper = GradientClipper::clip_by_norm(10.0);
    
    // Compute original norm
    let mut grads = vec![&mut grad1, &mut grad2, &mut grad3];
    let original_norm = clipper.compute_grad_norm(&grads)?;
    println!("Original gradient norm: {:.4}", original_norm);
    
    // Clip gradients
    let norm_before_clip = clipper.clip_grads(&mut grads)?;
    println!("Norm returned by clip_grads: {:.4}", norm_before_clip);
    
    // Verify clipping worked
    let new_norm = clipper.compute_grad_norm(&grads)?;
    println!("Gradient norm after clipping: {:.4}", new_norm);
    println!("Target norm: 10.0");
    
    // Test 2: Clip by value
    println!("\n--- Test 2: Gradient Clipping by Value ---");
    
    let mut grad_value = Tensor::from_vec(
        vec![-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0, 15.0],
        Shape::from_dims(&[2, 4]),
        device.clone()
    )?;
    
    println!("Original values: {:?}", grad_value.to_vec()?);
    
    let value_clipper = GradientClipper::clip_by_value(-3.0, 3.0);
    let mut value_grads = vec![&mut grad_value];
    value_clipper.clip_grads(&mut value_grads)?;
    
    println!("After clipping to [-3, 3]: {:?}", value_grads[0].to_vec()?);
    
    // Test 3: Adaptive clipping
    println!("\n--- Test 3: Adaptive Gradient Clipping ---");
    
    // Create gradient with outliers
    let mut grad_adaptive = Tensor::from_vec(
        vec![0.1, 0.2, 0.1, 100.0, 0.15, 0.1, 0.2, 0.1],  // One outlier
        Shape::from_dims(&[8]),
        device.clone()
    )?;
    
    println!("Gradient with outlier: {:?}", grad_adaptive.to_vec()?);
    
    let adaptive_clipper = GradientClipper::adaptive(2.0);
    let mut adaptive_grads = vec![&mut grad_adaptive];
    adaptive_clipper.clip_grads(&mut adaptive_grads)?;
    
    println!("After adaptive clipping: {:?}", adaptive_grads[0].to_vec()?);
    
    // Test 4: Layer-wise clipping
    println!("\n--- Test 4: Layer-wise Gradient Clipping ---");
    
    let mut layer1_grad = Tensor::randn(Shape::from_dims(&[128, 256]), 0.0, 0.5, device.clone())?;
    let mut layer2_grad = Tensor::randn(Shape::from_dims(&[256, 512]), 0.0, 0.1, device.clone())?;
    let mut layer3_grad = Tensor::randn(Shape::from_dims(&[512, 1024]), 0.0, 2.0, device.clone())?;
    
    let layer_clipper = LayerWiseGradientClipper::new(1.0);
    let mut layer_grads = vec![&mut layer1_grad, &mut layer2_grad, &mut layer3_grad];
    
    // Get norms before clipping
    let norms_before: Vec<f32> = layer_grads.iter()
        .map(|g| g.square().unwrap().sum().unwrap().item().unwrap().sqrt())
        .collect();
    
    println!("Layer norms before clipping: {:?}", norms_before);
    
    let norms_after = layer_clipper.clip_grads(&mut layer_grads)?;
    println!("Layer norms after clipping: {:?}", norms_after);
    
    // Test 5: Gradient norm tracking
    println!("\n--- Test 5: Gradient Norm Tracking ---");
    
    let mut tracker = GradientNormTracker::new(10);
    
    // Simulate training with varying gradient norms
    let norm_sequence = vec![
        0.5, 0.6, 0.7, 0.8, 0.9,  // Normal training
        50.0,                      // Gradient explosion
        0.001, 0.0005,            // Gradient vanishing
        1.0, 1.2                  // Recovery
    ];
    
    for (step, &norm) in norm_sequence.iter().enumerate() {
        tracker.record(norm);
        
        if tracker.is_exploding(10.0) {
            println!("Step {}: Gradient explosion detected! Norm = {}", step, norm);
        } else if tracker.is_vanishing(0.01) {
            println!("Step {}: Gradient vanishing detected! Norm = {}", step, norm);
        } else {
            println!("Step {}: Normal gradient norm = {}", step, norm);
        }
    }
    
    let stats = tracker.stats();
    println!("\nGradient statistics over window:");
    println!("  Mean: {:.4}", stats.mean);
    println!("  Std Dev: {:.4}", stats.std_dev);
    println!("  Max: {:.4}", stats.max);
    println!("  Min: {:.4}", stats.min);
    println!("  Current: {:.4}", stats.current);
    println!("  Healthy: {}", stats.is_healthy());
    
    // Test 6: Performance benchmark
    println!("\n--- Test 6: Performance Benchmark ---");
    
    // Create large gradients for performance testing
    let mut large_grads: Vec<Tensor> = (0..10)
        .map(|_| Tensor::randn(Shape::from_dims(&[1024, 1024]), 0.0, 1.0, device.clone()).unwrap())
        .collect();
    
    let mut large_grad_refs: Vec<&mut Tensor> = large_grads.iter_mut().collect();
    
    let start = std::time::Instant::now();
    let norm_clipper = GradientClipper::clip_by_norm(1.0);
    
    for _ in 0..10 {
        norm_clipper.clip_grads(&mut large_grad_refs)?;
    }
    
    let elapsed = start.elapsed();
    println!("Time for 10 gradient clipping operations on 10x[1024x1024] tensors: {:?}", elapsed);
    println!("Average time per clip: {:?}", elapsed / 10);
    
    // Calculate total parameters
    let total_params = 10 * 1024 * 1024;
    println!("Total parameters: {} ({:.1}M)", total_params, total_params as f32 / 1e6);
    
    println!("\nAll gradient clipping tests completed!");
    
    Ok(())
}