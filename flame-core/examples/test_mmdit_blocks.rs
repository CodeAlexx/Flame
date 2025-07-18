use flame_core::{Tensor, Shape, CudaDevice};
use flame_core::mmdit_blocks::{
    MMDiTConfig, JointAttention, MMDiTBlock, FinalLayer, get_2d_sincos_pos_embed
};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Flame MMDiT blocks for SD3.5...");
    
    // Initialize CUDA device
    let device = Arc::new(CudaDevice::new(0)?);
    println!("CUDA device initialized");
    
    // Test 2D sinusoidal position embeddings
    test_2d_sincos_pos_embed(&device)?;
    
    // Test joint attention
    test_joint_attention(&device)?;
    
    // Test MMDiT block
    test_mmdit_block(&device)?;
    
    // Test final layer
    test_final_layer(&device)?;
    
    // Test full MMDiT forward pass simulation
    test_full_mmdit_forward(&device)?;
    
    println!("\nAll MMDiT block tests passed!");
    Ok(())
}

fn test_2d_sincos_pos_embed(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing 2D Sinusoidal Position Embeddings ---");
    
    let embed_dim = 768;
    let grid_size = (16, 16);  // 16x16 patches
    
    let pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size, device.clone())?;
    
    println!("Position embedding shape: {:?}", pos_embed.shape().dims());
    assert_eq!(pos_embed.shape().dims(), &[256, embed_dim]);  // 16*16 = 256 patches
    
    // Check statistics
    let data = pos_embed.to_vec()?;
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32).sqrt();
    
    println!("Position embedding mean: {:.6}, std: {:.6}", mean, std);
    // Note: Mean won't be zero due to grid coordinates being positive
    assert!(std > 0.3 && std < 0.8);  // Should have reasonable variance
    
    Ok(())
}

fn test_joint_attention(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing Joint Attention ---");
    
    let hidden_size = 768;
    let num_heads = 12;
    let seq_len = 256;  // Image patches + text tokens
    let batch = 2;
    
    // Create joint attention
    let attn = JointAttention::new(hidden_size, num_heads, true, true, device.clone())?;
    
    // Create inputs
    let x = Tensor::randn(
        Shape::from_dims(&[batch, seq_len, hidden_size]),
        0.0,
        0.02,
        device.clone()
    )?;
    
    // Position embeddings
    let pe = get_2d_sincos_pos_embed(hidden_size / num_heads, (16, 16), device.clone())?;
    
    // Forward pass
    let output = attn.forward(&x, &pe)?;
    println!("Joint attention output shape: {:?}", output.shape().dims());
    
    // Verify shape
    assert_eq!(output.shape().dims(), &[batch, seq_len, hidden_size]);
    
    // Check output statistics
    let data = output.to_vec()?;
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32).sqrt();
    println!("Output mean: {:.6}, std: {:.6}", mean, std);
    
    Ok(())
}

fn test_mmdit_block(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing MMDiT Block ---");
    
    let config = MMDiTConfig {
        hidden_size: 768,
        num_heads: 12,
        depth: 4,
        mlp_ratio: 4.0,
        qkv_bias: false,
        qk_norm: true,
        pos_embed_max_size: 32,
    };
    
    // Create block
    let block = MMDiTBlock::new(&config, device.clone())?;
    
    let batch = 2;
    let seq_len = 256;
    
    // Create inputs
    let x = Tensor::randn(
        Shape::from_dims(&[batch, seq_len, config.hidden_size]),
        0.0,
        0.02,
        device.clone()
    )?;
    
    let pe = get_2d_sincos_pos_embed(
        config.hidden_size / config.num_heads,
        (16, 16),
        device.clone()
    )?;
    
    // Conditioning (timestep embedding)
    let c = Tensor::randn(
        Shape::from_dims(&[batch, config.hidden_size]),
        0.0,
        0.1,
        device.clone()
    )?;
    
    // Forward pass
    let output = block.forward(&x, &pe, &c)?;
    println!("MMDiT block output shape: {:?}", output.shape().dims());
    
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

fn test_final_layer(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing Final Layer ---");
    
    let hidden_size = 768;
    let patch_size = 2;
    let out_channels = 16;  // SD3.5 uses 16-channel VAE
    let batch = 2;
    let seq_len = 256;
    
    // Create final layer
    let final_layer = FinalLayer::new(hidden_size, patch_size, out_channels, device.clone())?;
    
    // Create inputs
    let x = Tensor::randn(
        Shape::from_dims(&[batch, seq_len, hidden_size]),
        0.0,
        0.02,
        device.clone()
    )?;
    
    let c = Tensor::randn(
        Shape::from_dims(&[batch, hidden_size]),
        0.0,
        0.1,
        device.clone()
    )?;
    
    // Forward pass
    let output = final_layer.forward(&x, &c)?;
    println!("Final layer output shape: {:?}", output.shape().dims());
    
    // Verify shape: [batch, seq_len, patch_size * patch_size * out_channels]
    assert_eq!(
        output.shape().dims(),
        &[batch, seq_len, patch_size * patch_size * out_channels]
    );
    
    Ok(())
}

fn test_full_mmdit_forward(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing Full MMDiT Forward Pass ---");
    
    // SD3.5 Medium configuration
    let config = MMDiTConfig {
        hidden_size: 1536,
        num_heads: 24,
        depth: 24,  // Reduced for testing
        mlp_ratio: 4.0,
        qkv_bias: false,
        qk_norm: true,
        pos_embed_max_size: 192,
    };
    
    let batch = 1;
    let img_seq_len = 1024;  // 32x32 patches
    let txt_seq_len = 154;   // T5 max length
    let total_seq_len = img_seq_len + txt_seq_len;
    
    println!("Simulating SD3.5 forward pass:");
    println!("- Hidden size: {}", config.hidden_size);
    println!("- Num heads: {}", config.num_heads);
    println!("- Image sequence length: {}", img_seq_len);
    println!("- Text sequence length: {}", txt_seq_len);
    println!("- Total sequence length: {}", total_seq_len);
    
    // Create position embeddings for 32x32 grid
    let img_pe = get_2d_sincos_pos_embed(
        config.hidden_size / config.num_heads,
        (32, 32),
        device.clone()
    )?;
    
    // For SD3.5, we need to extend position embeddings to handle text tokens
    // Text tokens don't use position embeddings (they use their own embeddings)
    // So we'll create a dummy PE tensor that covers the full sequence
    let pe_data = img_pe.to_vec()?;
    let mut full_pe_data = vec![0.0f32; total_seq_len * (config.hidden_size / config.num_heads)];
    
    // Copy image position embeddings
    for i in 0..img_seq_len {
        for j in 0..(config.hidden_size / config.num_heads) {
            full_pe_data[i * (config.hidden_size / config.num_heads) + j] = 
                pe_data[i * (config.hidden_size / config.num_heads) + j];
        }
    }
    // Text positions remain zero (they don't use sinusoidal embeddings)
    
    let full_pe = Tensor::from_vec(
        full_pe_data,
        Shape::from_dims(&[total_seq_len, config.hidden_size / config.num_heads]),
        device.clone()
    )?;
    
    // Create input (concatenated image and text embeddings)
    let x = Tensor::randn(
        Shape::from_dims(&[batch, total_seq_len, config.hidden_size]),
        0.0,
        0.02,
        device.clone()
    )?;
    
    // Timestep conditioning
    let c = Tensor::randn(
        Shape::from_dims(&[batch, config.hidden_size]),
        0.0,
        0.1,
        device.clone()
    )?;
    
    // Process through one block
    let block = MMDiTBlock::new(&config, device.clone())?;
    let output = block.forward(&x, &full_pe, &c)?;
    
    println!("Block output shape: {:?}", output.shape().dims());
    assert_eq!(output.shape().dims(), &[batch, total_seq_len, config.hidden_size]);
    
    // Process through final layer
    let final_layer = FinalLayer::new(
        config.hidden_size,
        2,  // patch_size
        16, // out_channels (SD3.5 VAE)
        device.clone()
    )?;
    
    let final_output = final_layer.forward(&output, &c)?;
    println!("Final output shape: {:?}", final_output.shape().dims());
    
    Ok(())
}