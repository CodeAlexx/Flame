use flame_core::{Tensor, Shape, CudaDevice};
use flame_core::flux_blocks::{
    FluxConfig, FluxSelfAttention, DoubleStreamBlock, SingleStreamBlock
};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Flame Flux blocks...");
    
    // Initialize CUDA device
    let device = Arc::new(CudaDevice::new(0)?);
    println!("CUDA device initialized");
    
    // Test Flux self-attention
    test_flux_self_attention(&device)?;
    
    // Test double stream block
    test_double_stream_block(&device)?;
    
    // Test single stream block
    test_single_stream_block(&device)?;
    
    println!("\nAll Flux block tests passed!");
    Ok(())
}

fn test_flux_self_attention(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing FluxSelfAttention ---");
    
    let hidden_size = 768;
    let num_heads = 12;
    let seq_len = 16;
    let batch = 2;
    
    // Create attention module
    let attn = FluxSelfAttention::new(hidden_size, num_heads, true, device.clone())?;
    
    // Create inputs
    let x = Tensor::randn(
        Shape::from_dims(&[batch, seq_len, hidden_size]),
        0.0,
        0.02,
        device.clone()
    )?;
    
    // Create position embeddings
    let pe = Tensor::randn(
        Shape::from_dims(&[seq_len, hidden_size / num_heads]),
        0.0,
        0.02,
        device.clone()
    )?;
    
    // Forward pass
    let output = attn.forward(&x, &pe)?;
    println!("Flux attention output shape: {:?}", output.shape().dims());
    
    // Verify shape
    assert_eq!(output.shape().dims(), &[batch, seq_len, hidden_size]);
    
    // Check output values
    let data = output.to_vec()?;
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32).sqrt();
    println!("Output mean: {:.6}, std: {:.6}", mean, std);
    
    Ok(())
}

fn test_double_stream_block(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing DoubleStreamBlock ---");
    
    let config = FluxConfig {
        hidden_size: 768,
        num_heads: 12,
        head_dim: 64,
        mlp_ratio: 4.0,
        theta: 10_000.0,
        qkv_bias: true,
        guidance_embed: false,
    };
    
    // Create block
    let block = DoubleStreamBlock::new(&config, device.clone())?;
    
    let batch = 2;
    let img_seq = 16;
    let txt_seq = 8;
    
    // Create inputs
    let img = Tensor::randn(
        Shape::from_dims(&[batch, img_seq, config.hidden_size]),
        0.0,
        0.02,
        device.clone()
    )?;
    
    let txt = Tensor::randn(
        Shape::from_dims(&[batch, txt_seq, config.hidden_size]),
        0.0,
        0.02,
        device.clone()
    )?;
    
    let vec = Tensor::randn(
        Shape::from_dims(&[batch, config.hidden_size]),
        0.0,
        0.02,
        device.clone()
    )?;
    
    let pe = Tensor::randn(
        Shape::from_dims(&[img_seq + txt_seq, config.head_dim]),
        0.0,
        0.02,
        device.clone()
    )?;
    
    // Forward pass
    let (img_out, txt_out) = block.forward(&img, &txt, &vec, &pe)?;
    
    println!("Image output shape: {:?}", img_out.shape().dims());
    println!("Text output shape: {:?}", txt_out.shape().dims());
    
    // Verify shapes
    assert_eq!(img_out.shape().dims(), &[batch, img_seq, config.hidden_size]);
    assert_eq!(txt_out.shape().dims(), &[batch, txt_seq, config.hidden_size]);
    
    // Check residual connections
    let img_data = img.to_vec()?;
    let img_out_data = img_out.to_vec()?;
    let img_diff: f32 = img_data.iter()
        .zip(img_out_data.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / img_data.len() as f32;
    println!("Average image difference (shows residual): {:.6}", img_diff);
    
    Ok(())
}

fn test_single_stream_block(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing SingleStreamBlock ---");
    
    let config = FluxConfig {
        hidden_size: 768,
        num_heads: 12,
        head_dim: 64,
        mlp_ratio: 4.0,
        theta: 10_000.0,
        qkv_bias: true,
        guidance_embed: false,
    };
    
    // Create block
    let block = SingleStreamBlock::new(&config, device.clone())?;
    
    let batch = 2;
    let seq_len = 24; // Combined image + text
    
    // Create inputs
    let x = Tensor::randn(
        Shape::from_dims(&[batch, seq_len, config.hidden_size]),
        0.0,
        0.02,
        device.clone()
    )?;
    
    let vec = Tensor::randn(
        Shape::from_dims(&[batch, config.hidden_size]),
        0.0,
        0.02,
        device.clone()
    )?;
    
    let pe = Tensor::randn(
        Shape::from_dims(&[seq_len, config.head_dim]),
        0.0,
        0.02,
        device.clone()
    )?;
    
    // Forward pass
    let output = block.forward(&x, &vec, &pe)?;
    
    println!("Single stream output shape: {:?}", output.shape().dims());
    
    // Verify shape
    assert_eq!(output.shape().dims(), &[batch, seq_len, config.hidden_size]);
    
    // Check residual connection
    let x_data = x.to_vec()?;
    let out_data = output.to_vec()?;
    let diff: f32 = x_data.iter()
        .zip(out_data.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / x_data.len() as f32;
    println!("Average difference (shows residual): {:.6}", diff);
    
    Ok(())
}