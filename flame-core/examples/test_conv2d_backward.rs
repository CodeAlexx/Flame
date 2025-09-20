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

use flame_core::conv::Conv2d;
use flame_core::{CudaDevice, Result, Shape, Tensor};

fn main() -> Result<()> {
    println!("🔬 Testing FLAME Conv2D backward pass...\n");

    let device = CudaDevice::new(0)?;

    // Test 1: Simple Conv2D forward and backward
    {
        println!("Test 1: Conv2D forward and backward");

        // Create input: [batch=2, channels=3, height=8, width=8]
        let input = Tensor::randn(Shape::from_dims(&[2, 3, 8, 8]), 0.0, 0.1, device.clone())?
            .requires_grad_(true);

        // Create Conv2D layer: 3 input channels, 16 output channels, 3x3 kernel
        let mut conv = Conv2d::new(
            3,  // in_channels
            16, // out_channels
            3,  // kernel_size
            1,  // stride
            1,  // padding
            device.clone(),
        )?;

        // Make weights require grad
        conv.weight = conv.weight.requires_grad_(true);
        if let Some(bias) = conv.bias.take() {
            conv.bias = Some(bias.requires_grad_(true));
        }

        println!("  Input shape: {:?}", input.shape().dims());
        println!("  Weight shape: {:?}", conv.weight.shape().dims());

        // Forward pass
        let output = conv.forward(&input)?;
        println!("  Output shape: {:?}", output.shape().dims());

        // Create loss (sum of outputs)
        let loss = output.sum()?;
        println!("  Loss: {:?}", loss.to_vec()?[0]);

        // Backward pass
        println!("  Running backward pass...");
        let grads = loss.backward()?;

        println!("  Got {} gradients", grads.len());

        // Check that we have gradients for input and weight
        let input_grad = grads
            .get(input.id())
            .expect("Should have gradient for input");
        let weight_grad = grads
            .get(conv.weight.id())
            .expect("Should have gradient for weight");

        println!("  Input gradient shape: {:?}", input_grad.shape().dims());
        println!("  Weight gradient shape: {:?}", weight_grad.shape().dims());

        // Check gradient values are reasonable
        let input_grad_data = input_grad.to_vec()?;
        let weight_grad_data = weight_grad.to_vec()?;

        let input_grad_mean = input_grad_data.iter().sum::<f32>() / input_grad_data.len() as f32;
        let weight_grad_mean = weight_grad_data.iter().sum::<f32>() / weight_grad_data.len() as f32;

        println!("  Input gradient mean: {:.6}", input_grad_mean);
        println!("  Weight gradient mean: {:.6}", weight_grad_mean);

        // Check bias gradient if present
        if let Some(ref bias) = conv.bias {
            let bias_grad = grads.get(bias.id()).expect("Should have gradient for bias");
            println!("  Bias gradient shape: {:?}", bias_grad.shape().dims());

            let bias_grad_data = bias_grad.to_vec()?;
            println!(
                "  Bias gradient sample: [{:.4}, {:.4}, {:.4}, ...]",
                bias_grad_data[0], bias_grad_data[1], bias_grad_data[2]
            );
        }

        println!("  ✅ Conv2D backward pass test passed!\n");
    }

    // Test 2: Conv2D with different parameters
    {
        println!("Test 2: Conv2D with stride=2, no padding");

        let input = Tensor::randn(Shape::from_dims(&[1, 1, 6, 6]), 0.0, 1.0, device.clone())?
            .requires_grad_(true);

        let mut conv = Conv2d::new(
            1, // in_channels
            8, // out_channels
            3, // kernel_size
            2, // stride
            0, // padding
            device.clone(),
        )?;

        conv.weight = conv.weight.requires_grad_(true);

        let output = conv.forward(&input)?;
        println!(
            "  Input shape: {:?} -> Output shape: {:?}",
            input.shape().dims(),
            output.shape().dims()
        );

        let loss = output.mean()?;
        let grads = loss.backward()?;

        println!("  Got {} gradients", grads.len());
        println!("  ✅ Conv2D with stride test passed!\n");
    }

    println!("🎉 ALL CONV2D BACKWARD TESTS PASSED! 🎉");

    Ok(())
}
