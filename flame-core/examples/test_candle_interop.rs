use flame_core::{Tensor, Shape, CudaDevice};
use flame_core::candle_interop::{CandleInterop, InteropTensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Flame-Candle interoperability...");
    
    // Initialize CUDA device
    let device = CudaDevice::new(0)?;
    println!("CUDA device initialized");
    
    // Create a Flame tensor
    let flame_tensor = Tensor::randn(
        Shape::from_dims(&[2, 3, 4, 4]),
        0.0,
        1.0,
        device.clone()
    )?;
    println!("Created Flame tensor with shape: {:?}", flame_tensor.shape().dims());
    
    // Convert to Candle
    println!("\n--- Converting Flame → Candle ---");
    let candle_tensor = flame_tensor.to_candle()?;
    println!("Candle tensor shape: {:?}", candle_tensor.dims());
    println!("Candle tensor device: {:?}", candle_tensor.device());
    
    // Perform a Candle operation
    println!("\n--- Performing Candle operation ---");
    let candle_result = candle_tensor.mul(&candle_tensor)?; // Square the tensor
    println!("Candle operation complete");
    
    // Convert back to Flame
    println!("\n--- Converting Candle → Flame ---");
    let flame_result = Tensor::from_candle(&candle_result, device.clone())?;
    println!("Flame tensor shape: {:?}", flame_result.shape().dims());
    
    // Verify the operation
    let original_data = flame_tensor.to_vec()?;
    let result_data = flame_result.to_vec()?;
    
    let mut max_diff = 0.0f32;
    for (i, (orig, res)) in original_data.iter().zip(result_data.iter()).enumerate() {
        let expected = orig * orig;
        let diff = (expected - res).abs();
        max_diff = max_diff.max(diff);
        
        if i < 5 {
            println!("  {} * {} = {} (got {})", orig, orig, expected, res);
        }
    }
    println!("Max difference: {}", max_diff);
    
    // Test batch conversion
    println!("\n--- Testing batch conversion ---");
    let batch: Vec<Tensor> = (0..3)
        .map(|i| {
            Tensor::from_vec(
                vec![i as f32; 6],
                Shape::from_dims(&[2, 3]),
                device.clone()
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    
    let candle_batch = CandleInterop::flame_batch_to_candle(&batch)?;
    println!("Converted {} tensors to Candle", candle_batch.len());
    
    let flame_batch = CandleInterop::candle_batch_to_flame(&candle_batch, device)?;
    println!("Converted {} tensors back to Flame", flame_batch.len());
    
    println!("\nInterop test completed successfully!");
    
    Ok(())
}