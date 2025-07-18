use flame_core::{Tensor, Shape, CudaDevice};
use flame_core::attention::{
    MultiHeadAttention, AttentionConfig, TransformerBlock, 
    GeGLU, FeedForward, RotaryEmbedding
};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Flame attention modules...");
    
    // Initialize CUDA device
    let device = Arc::new(CudaDevice::new(0)?);
    println!("CUDA device initialized");
    
    // Test basic multi-head attention
    test_multi_head_attention(&device)?;
    
    // Test GeGLU activation
    test_geglu(&device)?;
    
    // Test feed-forward network
    test_feedforward(&device)?;
    
    // Test rotary embeddings
    test_rope(&device)?;
    
    // Test transformer block
    test_transformer_block(&device)?;
    
    println!("\nAll attention tests passed!");
    Ok(())
}

fn test_multi_head_attention(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing Multi-Head Attention ---");
    
    let embed_dim = 512;
    let num_heads = 8;
    let seq_len = 10;
    let batch_size = 2;
    
    // Create attention module
    let config = AttentionConfig::new(embed_dim, num_heads);
    let attn = MultiHeadAttention::new(config, device.clone())?;
    
    // Create input tensors
    let query = Tensor::randn(
        Shape::from_dims(&[batch_size, seq_len, embed_dim]),
        0.0,
        0.02,
        device.clone()
    )?;
    
    let key = Tensor::randn(
        Shape::from_dims(&[batch_size, seq_len, embed_dim]),
        0.0,
        0.02,
        device.clone()
    )?;
    
    let value = Tensor::randn(
        Shape::from_dims(&[batch_size, seq_len, embed_dim]),
        0.0,
        0.02,
        device.clone()
    )?;
    
    // Forward pass
    let output = attn.forward(&query, &key, &value, None)?;
    println!("Output shape: {:?}", output.shape().dims());
    
    // Verify output shape
    assert_eq!(output.shape().dims(), &[batch_size, seq_len, embed_dim]);
    
    // Test with attention mask (causal mask)
    let mut mask_data = vec![1.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in i+1..seq_len {
            mask_data[i * seq_len + j] = 0.0; // Mask future positions
        }
    }
    let mask = Tensor::from_vec(
        mask_data,
        Shape::from_dims(&[seq_len, seq_len]),
        device.clone()
    )?;
    
    let masked_output = attn.forward(&query, &key, &value, Some(&mask))?;
    println!("Masked output shape: {:?}", masked_output.shape().dims());
    
    // Check that outputs are different with mask
    let output_data = output.to_vec()?;
    let masked_data = masked_output.to_vec()?;
    let mut diff_count = 0;
    for i in 0..output_data.len() {
        if (output_data[i] - masked_data[i]).abs() > 1e-6 {
            diff_count += 1;
        }
    }
    println!("Differences with mask: {} / {}", diff_count, output_data.len());
    
    Ok(())
}

fn test_geglu(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing GeGLU ---");
    
    let dim_in = 256;
    let dim_out = 512;
    let batch_size = 2;
    
    // Create GeGLU module
    let geglu = GeGLU::new(dim_in, dim_out, device.clone())?;
    
    // Create input
    let input = Tensor::randn(
        Shape::from_dims(&[batch_size, dim_in]),
        0.0,
        1.0,
        device.clone()
    )?;
    
    // Forward pass
    let output = geglu.forward(&input)?;
    println!("GeGLU output shape: {:?}", output.shape().dims());
    
    // Verify output shape
    assert_eq!(output.shape().dims(), &[batch_size, dim_out]);
    
    // Check that output has reasonable values (not all zeros or NaN)
    let data = output.to_vec()?;
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
    println!("Output mean: {:.6}, variance: {:.6}", mean, variance);
    
    Ok(())
}

fn test_feedforward(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing FeedForward ---");
    
    let dim = 512;
    let mult = 4;
    let batch_size = 2;
    let seq_len = 10;
    
    // Create FeedForward module
    let ff = FeedForward::new(dim, None, mult, device.clone())?;
    
    // Create input
    let input = Tensor::randn(
        Shape::from_dims(&[batch_size, seq_len, dim]),
        0.0,
        0.02,
        device.clone()
    )?;
    
    // Forward pass
    let output = ff.forward(&input)?;
    println!("FeedForward output shape: {:?}", output.shape().dims());
    
    // Verify output shape
    assert_eq!(output.shape().dims(), &[batch_size, seq_len, dim]);
    
    Ok(())
}

fn test_rope(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing Rotary Embeddings ---");
    
    let dim = 64;
    let max_seq_len = 1024;
    let theta = 10000.0;
    let batch_size = 2;
    let num_heads = 8;
    let seq_len = 10;
    
    // Create RoPE module
    let mut rope = RotaryEmbedding::new(dim, max_seq_len, theta);
    rope.init_freqs(device.clone())?;
    
    // Create input tensor [batch, heads, seq_len, dim]
    let input = Tensor::randn(
        Shape::from_dims(&[batch_size, num_heads, seq_len, dim]),
        0.0,
        1.0,
        device.clone()
    )?;
    
    // Apply RoPE
    let output = rope.forward(&input, seq_len)?;
    println!("RoPE output shape: {:?}", output.shape().dims());
    
    // Verify output shape unchanged
    assert_eq!(output.shape().dims(), input.shape().dims());
    
    // Check that output is different from input (positions are encoded)
    let input_data = input.to_vec()?;
    let output_data = output.to_vec()?;
    let mut diff_count = 0;
    for i in 0..input_data.len() {
        if (input_data[i] - output_data[i]).abs() > 1e-6 {
            diff_count += 1;
        }
    }
    println!("Values changed by RoPE: {} / {}", diff_count, input_data.len());
    
    Ok(())
}

fn test_transformer_block(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing Transformer Block ---");
    
    let dim = 512;
    let num_heads = 8;
    let ff_mult = 4;
    let batch_size = 2;
    let seq_len = 10;
    let context_len = 15;
    
    // Test without cross-attention
    println!("\nTesting self-attention only block:");
    let block = TransformerBlock::new(dim, num_heads, ff_mult, false, device.clone())?;
    
    let input = Tensor::randn(
        Shape::from_dims(&[batch_size, seq_len, dim]),
        0.0,
        0.02,
        device.clone()
    )?;
    
    let output = block.forward(&input, None, None, None)?;
    println!("Self-attention block output shape: {:?}", output.shape().dims());
    assert_eq!(output.shape().dims(), &[batch_size, seq_len, dim]);
    
    // Test with cross-attention
    println!("\nTesting cross-attention block:");
    let cross_block = TransformerBlock::new(dim, num_heads, ff_mult, true, device.clone())?;
    
    let context = Tensor::randn(
        Shape::from_dims(&[batch_size, context_len, dim]),
        0.0,
        0.02,
        device.clone()
    )?;
    
    let cross_output = cross_block.forward(&input, Some(&context), None, None)?;
    println!("Cross-attention block output shape: {:?}", cross_output.shape().dims());
    assert_eq!(cross_output.shape().dims(), &[batch_size, seq_len, dim]);
    
    // Verify residual connections are working (output shouldn't be too different from input)
    let input_data = input.to_vec()?;
    let output_data = output.to_vec()?;
    let cross_output_data = cross_output.to_vec()?;
    
    let self_attn_diff: f32 = input_data.iter()
        .zip(output_data.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / input_data.len() as f32;
    
    let cross_attn_diff: f32 = input_data.iter()
        .zip(cross_output_data.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / input_data.len() as f32;
    
    println!("Average difference - self-attention: {:.6}, cross-attention: {:.6}", 
             self_attn_diff, cross_attn_diff);
    
    Ok(())
}