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

use flame_core::regularization::{
    Dropout, Dropout2d, L1Regularization, L2Regularization, WeightStandardization,
};
use flame_core::{CudaDevice, Shape, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::new(0)?;
    println!("CUDA device initialized");

    println!("\n=== Testing Regularization Layers ===");

    // Test 1: Dropout
    println!("\n--- Test 1: Dropout ---");
    let input = Tensor::randn(Shape::from_dims(&[4, 8]), 0.0, 1.0, device.clone())?;
    let mut dropout = Dropout::new(0.5);

    // Training mode
    dropout.train(true);
    let output_train = dropout.forward(&input)?;
    println!("Input shape: {:?}", input.shape().dims());
    println!("Output shape (training): {:?}", output_train.shape().dims());

    // Check that some values are zeroed
    let input_data = input.to_vec()?;
    let output_data = output_train.to_vec()?;
    let zero_count = output_data.iter().filter(|&&x| x == 0.0).count();
    println!(
        "Percentage of zeros: {:.1}%",
        (zero_count as f32 / output_data.len() as f32) * 100.0
    );

    // Eval mode
    dropout.train(false);
    let output_eval = dropout.forward(&input)?;
    let eval_data = output_eval.to_vec()?;

    // In eval mode, should be identical to input
    let max_diff = input_data
        .iter()
        .zip(eval_data.iter())
        .map(|(a, b)| (a - b).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    println!("Max difference in eval mode: {} (should be ~0)", max_diff);

    // Test 2: Dropout2d for Conv layers
    println!("\n--- Test 2: Dropout2d ---");
    let conv_input = Tensor::randn(
        Shape::from_dims(&[2, 16, 8, 8]), // [batch, channels, height, width]
        0.0,
        1.0,
        device.clone(),
    )?;

    let mut dropout2d = Dropout2d::new(0.3);
    dropout2d.train(true);
    let conv_output = dropout2d.forward(&conv_input)?;

    println!("Conv input shape: {:?}", conv_input.shape().dims());
    println!("Conv output shape: {:?}", conv_output.shape().dims());

    // Check that entire channels are dropped
    let conv_data = conv_output.to_vec()?;
    let channel_size = 8 * 8; // height * width
    let mut dropped_channels = 0;

    for b in 0..2 {
        for c in 0..16 {
            let channel_start = b * 16 * 64 + c * 64;
            let channel_data = &conv_data[channel_start..channel_start + channel_size];
            if channel_data.iter().all(|&x| x == 0.0) {
                dropped_channels += 1;
            }
        }
    }
    println!("Dropped channels: {} out of 32 total", dropped_channels);

    // Test 3: L2 Regularization
    println!("\n--- Test 3: L2 Regularization ---");
    let weight1 = Tensor::randn(Shape::from_dims(&[64, 128]), 0.0, 0.1, device.clone())?;
    let weight2 = Tensor::randn(Shape::from_dims(&[128, 256]), 0.0, 0.1, device.clone())?;

    let l2_reg = L2Regularization::new(0.01);
    let penalty = l2_reg.penalty(&[&weight1, &weight2])?;
    println!("L2 penalty: {}", penalty.item()?);

    // Test gradient modification
    let grad = Tensor::randn(Shape::from_dims(&[64, 128]), 0.0, 0.01, device.clone())?;
    let modified_grad = l2_reg.apply_to_gradients(&weight1, &grad)?;

    let grad_norm_before = grad.square()?.sum()?.item()?.sqrt();
    let grad_norm_after = modified_grad.square()?.sum()?.item()?.sqrt();
    println!("Gradient norm before L2: {}", grad_norm_before);
    println!("Gradient norm after L2: {}", grad_norm_after);

    // Test 4: L1 Regularization
    println!("\n--- Test 4: L1 Regularization ---");
    let l1_reg = L1Regularization::new(0.001);
    let l1_penalty = l1_reg.penalty(&[&weight1, &weight2])?;
    println!("L1 penalty: {}", l1_penalty.item()?);

    // L1 should encourage sparsity
    let sparse_weight = Tensor::from_vec(
        vec![0.0, 1.0, 0.0, -1.0, 0.0, 2.0, 0.0, -2.0],
        Shape::from_dims(&[2, 4]),
        device.clone(),
    )?;
    let sparse_penalty = l1_reg.penalty(&[&sparse_weight])?;
    println!("L1 penalty for sparse weight: {}", sparse_penalty.item()?);

    // Test 5: Weight Standardization
    println!("\n--- Test 5: Weight Standardization ---");
    let conv_weight = Tensor::randn(
        Shape::from_dims(&[32, 16, 3, 3]), // [out_channels, in_channels, h, w]
        0.0,
        0.1,
        device.clone(),
    )?;

    let weight_std = WeightStandardization::new();
    let standardized = weight_std.forward(&conv_weight)?;

    // Check that each output channel is standardized
    let std_data = standardized.to_vec()?;
    let out_channels = 32;
    let channel_size = 16 * 3 * 3;

    let mut means = Vec::new();
    let mut stds = Vec::new();

    for i in 0..out_channels {
        let start = i * channel_size;
        let end = start + channel_size;
        let channel_data = &std_data[start..end];

        let mean: f32 = channel_data.iter().sum::<f32>() / channel_size as f32;
        let variance: f32 = channel_data
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / channel_size as f32;
        let std = variance.sqrt();

        means.push(mean);
        stds.push(std);
    }

    let avg_mean = means.iter().sum::<f32>() / means.len() as f32;
    let avg_std = stds.iter().sum::<f32>() / stds.len() as f32;

    println!("After standardization:");
    println!("  Average mean: {} (should be ~0)", avg_mean);
    println!("  Average std: {} (should be ~1)", avg_std);

    // Performance test
    println!("\n=== Performance Test ===");
    let large_input = Tensor::randn(Shape::from_dims(&[64, 512]), 0.0, 1.0, device)?;

    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _ = dropout.forward(&large_input)?;
    }
    let elapsed = start.elapsed();

    println!(
        "Time for 100 dropout operations on 64x512 tensor: {:?}",
        elapsed
    );
    println!("Average time per operation: {:?}", elapsed / 100);

    println!("\nAll regularization tests completed!");

    Ok(())
}
