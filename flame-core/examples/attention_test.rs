use flame_core::{Tensor, CudaDevice, Shape, attention::{MultiHeadAttention, AttentionConfig}};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA - CudaDevice::new already returns Arc<CudaDevice>
    let device = CudaDevice::new(0)?;
    println!("CUDA device initialized");

    println!("\n=== Testing Multi-Head Attention ===");
    
    // Configuration
    let batch_size = 2;
    let seq_len = 4;
    let embed_dim = 64;
    let num_heads = 8;
    
    let config = AttentionConfig::new(embed_dim, num_heads);
    let attention = MultiHeadAttention::new(config, device.clone())?;
    
    // Create input tensors
    let query = Tensor::randn(
        Shape::from_dims(&[batch_size, seq_len, embed_dim]),
        0.0, 0.1,
        device.clone()
    )?;
    
    let key = Tensor::randn(
        Shape::from_dims(&[batch_size, seq_len, embed_dim]),
        0.0, 0.1,
        device.clone()
    )?;
    
    let value = Tensor::randn(
        Shape::from_dims(&[batch_size, seq_len, embed_dim]),
        0.0, 0.1,
        device.clone()
    )?;
    
    println!("Input shapes:");
    println!("  Query: {:?}", query.shape().dims());
    println!("  Key: {:?}", key.shape().dims());
    println!("  Value: {:?}", value.shape().dims());
    
    // Test self-attention (Q=K=V)
    let output = attention.forward(&query, &query, &query, None)?;
    println!("\nSelf-attention output shape: {:?}", output.shape().dims());
    
    // Test cross-attention
    let output_cross = attention.forward(&query, &key, &value, None)?;
    println!("Cross-attention output shape: {:?}", output_cross.shape().dims());
    
    // Test with causal mask
    println!("\n=== Testing Causal Mask ===");
    
    // Create causal mask (lower triangular)
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..=i {
            mask_data[i * seq_len + j] = 1.0;
        }
    }
    
    let causal_mask = Tensor::from_vec(
        mask_data.clone(),
        Shape::from_dims(&[seq_len, seq_len]),
        device.clone()
    )?;
    
    println!("Causal mask:");
    for i in 0..seq_len {
        let row: Vec<f32> = mask_data[i*seq_len..(i+1)*seq_len].to_vec();
        println!("  {:?}", row);
    }
    
    let output_causal = attention.forward(&query, &query, &query, Some(&causal_mask))?;
    println!("\nCausal attention output shape: {:?}", output_causal.shape().dims());
    
    // Test different sequence lengths for K,V (encoder-decoder attention)
    println!("\n=== Testing Encoder-Decoder Attention ===");
    
    let encoder_seq_len = 6;
    let encoder_states = Tensor::randn(
        Shape::from_dims(&[batch_size, encoder_seq_len, embed_dim]),
        0.0, 0.1,
        device.clone()
    )?;
    
    let output_enc_dec = attention.forward(&query, &encoder_states, &encoder_states, None)?;
    println!("Encoder-decoder attention output shape: {:?}", output_enc_dec.shape().dims());
    
    // Performance test
    println!("\n=== Performance Test ===");
    
    let large_seq_len = 128;
    let large_query = Tensor::randn(
        Shape::from_dims(&[4, large_seq_len, embed_dim]),
        0.0, 0.1,
        device.clone()
    )?;
    
    let start = std::time::Instant::now();
    for _ in 0..10 {
        let _ = attention.forward(&large_query, &large_query, &large_query, None)?;
    }
    let elapsed = start.elapsed();
    
    println!("Time for 10 forward passes (batch=4, seq_len=128): {:?}", elapsed);
    println!("Average time per forward pass: {:?}", elapsed / 10);
    
    // Test attention scores visualization
    println!("\n=== Attention Pattern Visualization ===");
    
    // Create a simple attention layer with 1 head for visualization
    let simple_config = AttentionConfig::new(8, 1);
    let simple_attention = MultiHeadAttention::new(simple_config, device.clone())?;
    
    let simple_input = Tensor::randn(
        Shape::from_dims(&[1, 4, 8]),
        0.0, 0.1,
        device
    )?;
    
    let _ = simple_attention.forward(&simple_input, &simple_input, &simple_input, None)?;
    println!("Simple attention computed successfully");
    
    println!("\nAll attention tests completed!");
    
    Ok(())
}