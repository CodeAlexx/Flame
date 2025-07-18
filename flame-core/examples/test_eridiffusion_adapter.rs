use flame_core::{Tensor, Shape, CudaDevice};
use flame_core::eridiffusion_adapter::{
    TensorAdapter, WeightAdapter, TrainingStateAdapter, LossAdapter,
    TrainingConfig, LossType, GradientScaler, MemoryOptimizer
};
use std::sync::Arc;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Flame EriDiffusion integration adapters...");
    
    // Initialize CUDA device
    let device = Arc::new(CudaDevice::new(0)?);
    println!("CUDA device initialized");
    
    // Test tensor adapter
    test_tensor_adapter(&device)?;
    
    // Test weight adapter
    test_weight_adapter(&device)?;
    
    // Test training state adapter
    test_training_state_adapter(&device)?;
    
    // Test loss adapter
    test_loss_adapter(&device)?;
    
    // Test gradient scaler
    test_gradient_scaler()?;
    
    // Test memory optimizer
    test_memory_optimizer(&device)?;
    
    println!("\nAll EriDiffusion adapter tests passed!");
    Ok(())
}

fn test_tensor_adapter(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing Tensor Adapter ---");
    
    let adapter = TensorAdapter::new(device.clone());
    
    let batch_size = 2;
    
    // Create test tensors
    let images = Tensor::randn(
        Shape::from_dims(&[batch_size, 3, 512, 512]),
        0.0,
        1.0,
        device.clone()
    )?;
    
    let text_embeddings = Tensor::randn(
        Shape::from_dims(&[batch_size, 77, 768]),
        0.0,
        0.02,
        device.clone()
    )?;
    
    let timesteps = Tensor::from_vec(
        vec![0.1, 0.5],
        Shape::from_dims(&[batch_size]),
        device.clone()
    )?;
    
    // Test batch preparation
    let batch = adapter.prepare_batch(&images, None, &text_embeddings, &timesteps)?;
    
    println!("Batch prepared successfully:");
    println!("- Batch size: {}", batch.batch_size);
    println!("- Images shape: {:?}", batch.images.shape().dims());
    println!("- Text embeddings shape: {:?}", batch.text_embeddings.shape().dims());
    println!("- Timesteps shape: {:?}", batch.timesteps.shape().dims());
    
    assert_eq!(batch.batch_size, batch_size);
    
    Ok(())
}

fn test_weight_adapter(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing Weight Adapter ---");
    
    let adapter = WeightAdapter::new(device.clone());
    
    // Create test weights
    let mut base_weights = HashMap::new();
    base_weights.insert(
        "layer.weight".to_string(),
        Tensor::randn(Shape::from_dims(&[768, 768]), 0.0, 0.02, device.clone())?
    );
    
    // Create LoRA weights
    let mut lora_weights = HashMap::new();
    let rank = 16;
    lora_weights.insert(
        "layer.weight.lora_down".to_string(),
        Tensor::randn(Shape::from_dims(&[rank, 768]), 0.0, 0.02, device.clone())?
    );
    lora_weights.insert(
        "layer.weight.lora_up".to_string(),
        Tensor::randn(Shape::from_dims(&[768, rank]), 0.0, 0.02, device.clone())?
    );
    
    // Test LoRA application
    let original_weight = base_weights["layer.weight"].to_vec()?;
    adapter.apply_lora(&mut base_weights, &lora_weights, 1.0)?;
    let modified_weight = base_weights["layer.weight"].to_vec()?;
    
    // Check that weight was modified
    let diff: f32 = original_weight.iter()
        .zip(modified_weight.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / original_weight.len() as f32;
    
    println!("LoRA applied successfully:");
    println!("- Average weight difference: {:.6}", diff);
    assert!(diff > 0.0, "Weights should be modified after LoRA application");
    
    Ok(())
}

fn test_training_state_adapter(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing Training State Adapter ---");
    
    let adapter = TrainingStateAdapter::new(device.clone());
    
    // Create training config
    let config = TrainingConfig {
        learning_rate: 5e-5,
        gradient_accumulation_steps: 4,
        max_steps: 1000,
        save_every: 100,
        validate_every: 50,
        use_ema: true,
        ema_decay: 0.999,
    };
    
    // Create training state
    let mut state = adapter.create_state(&config)?;
    
    println!("Initial training state:");
    println!("- Step: {}", state.step);
    println!("- Epoch: {}", state.epoch);
    println!("- Learning rate: {}", state.learning_rate);
    println!("- Best loss: {}", state.best_loss);
    
    // Simulate training steps
    for step in 1..=10 {
        let loss = 1.0 / (step as f32);  // Simulated decreasing loss
        adapter.update_state(&mut state, loss, step);
    }
    
    println!("\nAfter 10 steps:");
    println!("- Step: {}", state.step);
    println!("- Epoch: {}", state.epoch);
    println!("- Best loss: {:.4}", state.best_loss);
    
    assert_eq!(state.step, 10);
    assert!(state.best_loss < 1.0);
    
    Ok(())
}

fn test_loss_adapter(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing Loss Adapter ---");
    
    let adapter = LossAdapter::new(device.clone());
    
    let batch_size = 4;
    let channels = 4;
    let height = 64;
    let width = 64;
    
    // Create model output and target
    let model_output = Tensor::randn(
        Shape::from_dims(&[batch_size, channels, height, width]),
        0.0,
        0.1,
        device.clone()
    )?;
    
    let target = Tensor::randn(
        Shape::from_dims(&[batch_size, channels, height, width]),
        0.0,
        0.1,
        device.clone()
    )?;
    
    let timesteps = Tensor::from_vec(
        vec![0.1, 0.3, 0.5, 0.7],
        Shape::from_dims(&[batch_size]),
        device.clone()
    )?;
    
    // Test diffusion loss without SNR weighting
    let loss = adapter.compute_diffusion_loss(
        &model_output,
        &target,
        &timesteps,
        LossType::L2,
        None
    )?;
    
    println!("Loss without SNR weighting: {:.6}", loss.to_vec()?[0]);
    
    // Test diffusion loss with SNR weighting
    let loss_snr = adapter.compute_diffusion_loss(
        &model_output,
        &target,
        &timesteps,
        LossType::L2,
        Some(5.0)
    )?;
    
    println!("Loss with SNR weighting (gamma=5.0): {:.6}", loss_snr.to_vec()?[0]);
    
    // Verify loss is reduced to a single value
    assert_eq!(loss.shape().dims(), &[1]);
    assert_eq!(loss_snr.shape().dims(), &[1]);
    
    Ok(())
}

fn test_gradient_scaler() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing Gradient Scaler ---");
    
    let mut scaler = GradientScaler::new();
    
    println!("Initial scale: {}", scaler.scale);
    
    // Simulate steps without overflow
    for _ in 0..5 {
        scaler.update(false);
    }
    
    println!("Scale after 5 successful steps: {}", scaler.scale);
    
    // Simulate overflow
    scaler.update(true);
    println!("Scale after overflow: {}", scaler.scale);
    
    assert!(scaler.scale < 65536.0, "Scale should decrease after overflow");
    
    Ok(())
}

fn test_memory_optimizer(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing Memory Optimizer ---");
    
    let mut optimizer = MemoryOptimizer::new(device.clone());
    
    // Test activation caching
    let activation = Tensor::randn(
        Shape::from_dims(&[2, 512, 16, 16]),
        0.0,
        0.1,
        device.clone()
    )?;
    
    optimizer.cache_activation("layer1", activation);
    
    // Retrieve cached activation
    assert!(optimizer.get_activation("layer1").is_some());
    assert!(optimizer.get_activation("layer2").is_none());
    
    // Test memory estimation
    let batch_size = 4;
    let model_size = 1_000_000;  // 1M parameters
    let estimated_memory = optimizer.estimate_memory_usage(batch_size, model_size);
    
    println!("Memory estimation:");
    println!("- Batch size: {}", batch_size);
    println!("- Model size: {} parameters", model_size);
    println!("- Estimated memory: {} MB", estimated_memory / (1024 * 1024));
    
    // Clear cache
    optimizer.clear_cache();
    assert!(optimizer.get_activation("layer1").is_none());
    
    Ok(())
}