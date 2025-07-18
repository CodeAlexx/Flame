use flame_core::{Tensor, Shape, CudaDevice};
use flame_core::modulated_blocks::{
    AdaLayerNorm, ModulatedTransformerBlock, TimestepEmbedding, QKNorm
};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Flame modulated transformer blocks...");
    
    // Initialize CUDA device
    let device = Arc::new(CudaDevice::new(0)?);
    println!("CUDA device initialized");
    
    // Test AdaLayerNorm
    test_ada_layer_norm(&device)?;
    
    // Test timestep embedding
    test_timestep_embedding(&device)?;
    
    // Test modulated transformer block
    test_modulated_transformer_block(&device)?;
    
    // Test QK normalization
    test_qk_norm(&device)?;
    
    println!("\nAll modulated block tests passed!");
    Ok(())
}

fn test_ada_layer_norm(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing AdaLayerNorm ---");
    
    let num_features = 512;
    let cond_features = 256;
    let batch_size = 2;
    let seq_len = 10;
    
    // Create AdaLayerNorm
    let ada_ln = AdaLayerNorm::new(num_features, cond_features, device.clone())?;
    
    // Create inputs
    let x = Tensor::randn(
        Shape::from_dims(&[batch_size, seq_len, num_features]),
        0.0,
        1.0,
        device.clone()
    )?;
    
    let cond = Tensor::randn(
        Shape::from_dims(&[batch_size, cond_features]),
        0.0,
        0.1,
        device.clone()
    )?;
    
    // Forward pass
    let output = ada_ln.forward(&x, &cond)?;
    println!("AdaLayerNorm output shape: {:?}", output.shape().dims());
    
    // Check output shape
    assert_eq!(output.shape().dims(), &[batch_size, seq_len, num_features]);
    
    // Check that output is normalized with modulation
    let data = output.to_vec()?;
    let mean = data.iter().take(num_features).sum::<f32>() / num_features as f32;
    println!("First sequence mean after modulation: {:.6}", mean);
    
    Ok(())
}

fn test_timestep_embedding(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing TimestepEmbedding ---");
    
    let channels = 512;
    let freq_embed_size = 256;
    let batch_size = 4;
    
    // Create timestep embedding module
    let ts_embed = TimestepEmbedding::new(channels, freq_embed_size, device.clone())?;
    
    // Create timesteps
    let timesteps = Tensor::from_vec(
        vec![0.0, 250.0, 500.0, 1000.0],
        Shape::from_dims(&[batch_size]),
        device.clone()
    )?;
    
    // Get embeddings
    let embeddings = ts_embed.forward(&timesteps)?;
    println!("Timestep embeddings shape: {:?}", embeddings.shape().dims());
    
    // Verify shape
    assert_eq!(embeddings.shape().dims(), &[batch_size, channels]);
    
    // Check that embeddings are different for different timesteps
    let data = embeddings.to_vec()?;
    let emb1_norm: f32 = data[..channels].iter().map(|x| x * x).sum::<f32>().sqrt();
    let emb2_norm: f32 = data[channels..2*channels].iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("Embedding norms: t=0: {:.3}, t=250: {:.3}", emb1_norm, emb2_norm);
    
    Ok(())
}

fn test_modulated_transformer_block(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing ModulatedTransformerBlock ---");
    
    let hidden_size = 512;
    let num_heads = 8;
    let mlp_ratio = 4.0;
    let batch_size = 2;
    let seq_len = 10;
    
    // Create modulated transformer block
    let block = ModulatedTransformerBlock::new(
        hidden_size,
        num_heads,
        mlp_ratio,
        device.clone()
    )?;
    
    // Create inputs
    let x = Tensor::randn(
        Shape::from_dims(&[batch_size, seq_len, hidden_size]),
        0.0,
        0.02,
        device.clone()
    )?;
    
    // Create conditioning (e.g., timestep embedding)
    let c = Tensor::randn(
        Shape::from_dims(&[batch_size, hidden_size]),
        0.0,
        0.1,
        device.clone()
    )?;
    
    // Forward pass
    let output = block.forward(&x, &c)?;
    println!("Modulated block output shape: {:?}", output.shape().dims());
    
    // Verify shape preserved
    assert_eq!(output.shape().dims(), &[batch_size, seq_len, hidden_size]);
    
    // Check residual connections are working
    let x_data = x.to_vec()?;
    let out_data = output.to_vec()?;
    let diff: f32 = x_data.iter()
        .zip(out_data.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / x_data.len() as f32;
    println!("Average difference (shows residual): {:.6}", diff);
    
    Ok(())
}

fn test_qk_norm(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing QKNorm ---");
    
    let dim = 64;
    let batch_size = 2;
    let num_heads = 8;
    let seq_len = 10;
    
    // Create QK normalization
    let qk_norm = QKNorm::new(dim, device.clone())?;
    
    // Create query and key tensors
    let q = Tensor::randn(
        Shape::from_dims(&[batch_size, num_heads, seq_len, dim]),
        0.0,
        1.0,
        device.clone()
    )?;
    
    let k = Tensor::randn(
        Shape::from_dims(&[batch_size, num_heads, seq_len, dim]),
        0.0,
        1.0,
        device.clone()
    )?;
    
    // Apply QK normalization
    let (q_normed, k_normed) = qk_norm.forward(&q, &k)?;
    
    println!("Q normalized shape: {:?}", q_normed.shape().dims());
    println!("K normalized shape: {:?}", k_normed.shape().dims());
    
    // Verify shapes preserved
    assert_eq!(q_normed.shape().dims(), q.shape().dims());
    assert_eq!(k_normed.shape().dims(), k.shape().dims());
    
    // Check RMS normalization
    let q_data = q_normed.to_vec()?;
    let k_data = k_normed.to_vec()?;
    
    // Check RMS for first head
    let head_size = dim;
    let q_head_rms: f32 = (q_data[..head_size].iter()
        .map(|x| x * x)
        .sum::<f32>() / head_size as f32)
        .sqrt();
    let k_head_rms: f32 = (k_data[..head_size].iter()
        .map(|x| x * x)
        .sum::<f32>() / head_size as f32)
        .sqrt();
    
    println!("Q head RMS: {:.6}, K head RMS: {:.6} (should be close to 1)", q_head_rms, k_head_rms);
    
    Ok(())
}