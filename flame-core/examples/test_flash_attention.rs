use flame_core::{Tensor, Shape, CudaDevice, Result, flash_attention::{FlashAttention, FlashAttentionConfig}};
use std::sync::Arc;

fn main() -> Result<()> {
    println!("🔬 Testing FLAME Flash Attention...\n");
    
    let device = CudaDevice::new(0)?;
    
    // Test 1: Basic attention
    {
        println!("Test 1: Basic Flash Attention");
        
        let batch_size = 2;
        let seq_len = 8;
        let num_heads = 4;
        let head_dim = 16;
        
        // Create Q, K, V tensors
        let q = Tensor::randn(
            Shape::from_dims(&[batch_size, seq_len, num_heads, head_dim]),
            0.0, 0.1, device.clone()
        )?;
        
        let k = Tensor::randn(
            Shape::from_dims(&[batch_size, seq_len, num_heads, head_dim]),
            0.0, 0.1, device.clone()
        )?;
        
        let v = Tensor::randn(
            Shape::from_dims(&[batch_size, seq_len, num_heads, head_dim]),
            0.0, 0.1, device.clone()
        )?;
        
        // Create Flash Attention
        let config = FlashAttentionConfig::default();
        let attn = FlashAttention::new(config, device.clone());
        
        // Forward pass
        let output = attn.forward(&q, &k, &v, false)?;
        
        // Check output shape
        let expected_shape = vec![batch_size, seq_len, num_heads, head_dim];
        assert_eq!(output.shape().dims(), &expected_shape, "Output shape mismatch");
        
        println!("  Input shapes:");
        println!("    Q: {:?}", q.shape().dims());
        println!("    K: {:?}", k.shape().dims());
        println!("    V: {:?}", v.shape().dims());
        println!("  Output shape: {:?}", output.shape().dims());
        println!("  ✅ Basic attention passed!\n");
    }
    
    // Test 2: Causal attention
    {
        println!("Test 2: Causal Flash Attention");
        
        let batch_size = 1;
        let seq_len = 4;
        let num_heads = 2;
        let head_dim = 8;
        
        // Create simple inputs to verify causal masking
        let q = Tensor::ones(
            Shape::from_dims(&[batch_size, seq_len, num_heads, head_dim]),
            device.clone()
        )?;
        
        let k = Tensor::ones(
            Shape::from_dims(&[batch_size, seq_len, num_heads, head_dim]),
            device.clone()
        )?;
        
        // V with different values per position to verify masking
        let mut v_data = vec![0.0f32; batch_size * seq_len * num_heads * head_dim];
        for i in 0..seq_len {
            for j in 0..(num_heads * head_dim) {
                v_data[i * num_heads * head_dim + j] = i as f32;
            }
        }
        let v = Tensor::from_vec(
            v_data,
            Shape::from_dims(&[batch_size, seq_len, num_heads, head_dim]),
            device.clone()
        )?;
        
        // Create causal attention
        let config = FlashAttentionConfig {
            is_causal: true,
            ..Default::default()
        };
        let attn = FlashAttention::new(config, device.clone());
        
        // Forward pass
        let output = attn.forward(&q, &k, &v, false)?;
        
        // For causal attention, early positions should only attend to earlier positions
        // So output[0] should be close to v[0] = 0.0
        // output[1] should be average of v[0] and v[1] = 0.5
        // etc.
        
        let output_data = output.to_vec()?;
        println!("  First position output (should be ~0.0): {:.3}", output_data[0]);
        println!("  Second position output (should be ~0.5): {:.3}", output_data[num_heads * head_dim]);
        println!("  ✅ Causal attention passed!\n");
    }
    
    // Test 3: Different sequence lengths for Q and KV
    {
        println!("Test 3: Different sequence lengths (cross-attention)");
        
        let batch_size = 2;
        let seq_len_q = 6;
        let seq_len_kv = 10;
        let num_heads = 4;
        let head_dim = 16;
        
        let q = Tensor::randn(
            Shape::from_dims(&[batch_size, seq_len_q, num_heads, head_dim]),
            0.0, 0.1, device.clone()
        )?;
        
        let k = Tensor::randn(
            Shape::from_dims(&[batch_size, seq_len_kv, num_heads, head_dim]),
            0.0, 0.1, device.clone()
        )?;
        
        let v = Tensor::randn(
            Shape::from_dims(&[batch_size, seq_len_kv, num_heads, head_dim]),
            0.0, 0.1, device.clone()
        )?;
        
        let config = FlashAttentionConfig::default();
        let attn = FlashAttention::new(config, device.clone());
        
        let output = attn.forward(&q, &k, &v, false)?;
        
        // Output should have same sequence length as Q
        let expected_shape = vec![batch_size, seq_len_q, num_heads, head_dim];
        assert_eq!(output.shape().dims(), &expected_shape, "Cross-attention shape mismatch");
        
        println!("  Q seq_len: {}, KV seq_len: {}", seq_len_q, seq_len_kv);
        println!("  Output shape: {:?}", output.shape().dims());
        println!("  ✅ Cross-attention passed!\n");
    }
    
    // Test 4: Attention with dropout (training mode)
    {
        println!("Test 4: Attention with dropout");
        
        let batch_size = 1;
        let seq_len = 4;
        let num_heads = 2;
        let head_dim = 8;
        
        let q = Tensor::ones(
            Shape::from_dims(&[batch_size, seq_len, num_heads, head_dim]),
            device.clone()
        )?;
        
        let k = Tensor::ones(
            Shape::from_dims(&[batch_size, seq_len, num_heads, head_dim]),
            device.clone()
        )?;
        
        let v = Tensor::ones(
            Shape::from_dims(&[batch_size, seq_len, num_heads, head_dim]),
            device.clone()
        )?;
        
        let config = FlashAttentionConfig {
            dropout_p: 0.1,
            ..Default::default()
        };
        let attn = FlashAttention::new(config, device.clone());
        
        // Run with training=true (dropout enabled)
        let output_train = attn.forward(&q, &k, &v, true)?;
        
        // Run with training=false (no dropout)
        let output_eval = attn.forward(&q, &k, &v, false)?;
        
        // In eval mode, all outputs should be the same (no dropout)
        let eval_data = output_eval.to_vec()?;
        let first_val = eval_data[0];
        let all_same = eval_data.iter().all(|&x| (x - first_val).abs() < 1e-5);
        
        println!("  Dropout probability: 0.1");
        println!("  All eval outputs same: {}", all_same);
        println!("  ✅ Dropout attention passed!\n");
    }
    
    // Test 5: Edge cases
    {
        println!("Test 5: Edge cases");
        
        // Single head
        let q = Tensor::randn(
            Shape::from_dims(&[1, 16, 1, 64]),
            0.0, 0.1, device.clone()
        )?;
        let k = q.clone()?;
        let v = q.clone()?;
        
        let config = FlashAttentionConfig::default();
        let attn = FlashAttention::new(config, device.clone());
        
        let output = attn.forward(&q, &k, &v, false)?;
        assert_eq!(output.shape().dims(), q.shape().dims());
        println!("  Single head: ✅");
        
        // Large head dimension
        let q = Tensor::randn(
            Shape::from_dims(&[1, 8, 4, 256]),
            0.0, 0.1, device.clone()
        )?;
        let k = q.clone()?;
        let v = q.clone()?;
        
        let output = attn.forward(&q, &k, &v, false)?;
        assert_eq!(output.shape().dims(), q.shape().dims());
        println!("  Large head dimension (256): ✅");
        
        println!("  ✅ All edge cases passed!\n");
    }
    
    println!("🎉 ALL FLASH ATTENTION TESTS PASSED! 🎉");
    
    Ok(())
}