#![cfg(feature = "legacy_examples")]
#![allow(unused_imports, unused_variables, unused_mut, dead_code)]
#![cfg_attr(
    clippy,
    allow(
        clippy::unused_imports,
        clippy::useless_vec,
        clippy::needless_borrow,
        clippy::needless_clone
    )
)]

use flame_core::{
    serialization::{load_tensor, load_tensors, save_tensor, save_tensors, SerializationFormat},
    CudaDevice, Shape, Tensor,
};
use std::collections::HashMap;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA - CudaDevice::new already returns Arc<CudaDevice>
    let device = CudaDevice::new(0)?;
    println!("CUDA device initialized");

    println!("\n=== Testing Tensor Serialization ===");

    // Create test tensors
    let tensor1 = Tensor::randn(Shape::from_dims(&[3, 4, 5]), 0.0, 1.0, device.clone())?;

    let tensor2 = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Shape::from_dims(&[2, 3]),
        device.clone(),
    )?;

    println!("Created test tensors:");
    println!("  Tensor1 shape: {:?}", tensor1.shape().dims());
    println!("  Tensor2 shape: {:?}", tensor2.shape().dims());

    // Test single tensor save/load - Binary format
    println!("\n--- Testing Binary Format (Single Tensor) ---");
    let path = Path::new("/tmp/flame_tensor.bin");

    println!("Saving tensor1 to {:?}", path);
    save_tensor(&tensor1, path, SerializationFormat::Binary)?;

    println!("Loading tensor from {:?}", path);
    let loaded_tensor1 = load_tensor(path, device.clone(), SerializationFormat::Binary)?;

    println!("Verifying loaded tensor...");
    let original_data = tensor1.to_vec()?;
    let loaded_data = loaded_tensor1.to_vec()?;

    let max_diff = original_data
        .iter()
        .zip(loaded_data.iter())
        .map(|(a, b)| (a - b).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    println!("Max difference: {}", max_diff);
    assert!(max_diff < 1e-6, "Loaded tensor doesn't match original");
    println!("✓ Binary format single tensor test passed!");

    // Test multiple tensors save/load - Binary format
    println!("\n--- Testing Binary Format (Multiple Tensors) ---");
    let mut tensors = HashMap::new();
    tensors.insert("weight".to_string(), tensor1);
    tensors.insert("bias".to_string(), tensor2);

    let multi_path = Path::new("/tmp/flame_tensors.bin");
    println!("Saving multiple tensors to {:?}", multi_path);
    save_tensors(&tensors, multi_path, SerializationFormat::Binary)?;

    println!("Loading tensors from {:?}", multi_path);
    let loaded_tensors = load_tensors(multi_path, device.clone(), SerializationFormat::Binary)?;

    println!("Loaded {} tensors", loaded_tensors.len());
    for (name, tensor) in &loaded_tensors {
        println!("  {}: shape {:?}", name, tensor.shape().dims());
    }

    // Verify loaded tensors
    let loaded_weight = loaded_tensors.get("weight").unwrap();
    let loaded_bias = loaded_tensors.get("bias").unwrap();

    let weight_data = tensors.get("weight").unwrap().to_vec()?;
    let loaded_weight_data = loaded_weight.to_vec()?;
    let weight_diff = weight_data
        .iter()
        .zip(loaded_weight_data.iter())
        .map(|(a, b)| (a - b).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let bias_data = tensors.get("bias").unwrap().to_vec()?;
    let loaded_bias_data = loaded_bias.to_vec()?;
    let bias_diff = bias_data
        .iter()
        .zip(loaded_bias_data.iter())
        .map(|(a, b)| (a - b).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    println!("Weight max difference: {}", weight_diff);
    println!("Bias max difference: {}", bias_diff);
    assert!(
        weight_diff < 1e-6 && bias_diff < 1e-6,
        "Loaded tensors don't match originals"
    );
    println!("✓ Binary format multiple tensors test passed!");

    // Test SafeTensors format
    println!("\n--- Testing SafeTensors Format ---");
    let safe_path = Path::new("/tmp/flame_tensor.safetensors");

    // Create a new tensor for SafeTensors test
    let safe_tensor = Tensor::randn(Shape::from_dims(&[3, 4, 5]), 0.0, 1.0, device.clone())?;

    println!("Saving tensor to {:?} (SafeTensors format)", safe_path);
    save_tensor(&safe_tensor, safe_path, SerializationFormat::SafeTensors)?;

    println!("Loading tensor from {:?}", safe_path);
    let loaded_safe = load_tensor(safe_path, device.clone(), SerializationFormat::SafeTensors)?;

    let safe_original_data = safe_tensor.to_vec()?;
    let safe_data = loaded_safe.to_vec()?;
    let safe_diff = safe_original_data
        .iter()
        .zip(safe_data.iter())
        .map(|(a, b)| (a - b).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    println!("Max difference (SafeTensors): {}", safe_diff);
    assert!(
        safe_diff < 1e-6,
        "SafeTensors loaded tensor doesn't match original"
    );
    println!("✓ SafeTensors format test passed!");

    // Test convenience methods on Tensor
    println!("\n--- Testing Convenience Methods ---");
    let conv_path = Path::new("/tmp/flame_tensor_conv.bin");

    // Create a new tensor for convenience method test
    let conv_tensor = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Shape::from_dims(&[2, 3]),
        device.clone(),
    )?;

    println!("Using tensor.save() method");
    conv_tensor.save(conv_path)?;

    println!("Using Tensor::load() method");
    let loaded_conv = Tensor::load(conv_path, device.clone())?;

    let conv_original_data = conv_tensor.to_vec()?;
    let conv_data = loaded_conv.to_vec()?;
    let conv_diff = conv_original_data
        .iter()
        .zip(conv_data.iter())
        .map(|(a, b)| (a - b).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    println!("Max difference (convenience methods): {}", conv_diff);
    assert!(
        conv_diff < 1e-6,
        "Convenience method loaded tensor doesn't match original"
    );
    println!("✓ Convenience methods test passed!");

    // Performance test
    println!("\n=== Performance Test ===");
    let large_tensor = Tensor::randn(Shape::from_dims(&[512, 512, 4]), 0.0, 1.0, device.clone())?;

    let perf_path = Path::new("/tmp/flame_tensor_perf.bin");

    let start = std::time::Instant::now();
    large_tensor.save(perf_path)?;
    let save_time = start.elapsed();

    let start = std::time::Instant::now();
    let _ = Tensor::load(perf_path, device)?;
    let load_time = start.elapsed();

    println!("Large tensor (512x512x4 = ~4MB):");
    println!("  Save time: {:?}", save_time);
    println!("  Load time: {:?}", load_time);

    // Clean up temporary files
    println!("\n--- Cleaning up temporary files ---");
    let _ = std::fs::remove_file("/tmp/flame_tensor.bin");
    let _ = std::fs::remove_file("/tmp/flame_tensors.bin");
    let _ = std::fs::remove_file("/tmp/flame_tensor.safetensors");
    let _ = std::fs::remove_file("/tmp/flame_tensor_conv.bin");
    let _ = std::fs::remove_file("/tmp/flame_tensor_perf.bin");

    println!("\nAll serialization tests completed!");

    Ok(())
}
