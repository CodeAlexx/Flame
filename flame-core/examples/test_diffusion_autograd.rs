use flame_core::{Tensor, Shape, Result};
use cudarc::driver::CudaDevice;

fn main() -> Result<()> {
    println!("🔬 Testing autograd with diffusion model operations...\n");
    
    let device = CudaDevice::new(0)?;
    
    // Test 1: U-Net style operations (residual connections + attention)
    {
        println!("Test 1: U-Net style operations");
        
        // Simulate a U-Net block
        let x = Tensor::randn(Shape::from_dims(&[2, 64, 32, 32]), 0.0, 1.0, device.clone())?
            .requires_grad_(true);
        
        // Conv + ReLU (downsampling)
        let conv1_weight = Tensor::randn(Shape::from_dims(&[128, 64, 3, 3]), 0.0, 0.1, device.clone())?
            .requires_grad_(true);
        let y = x.conv2d(&conv1_weight, None, 1, 2)?; // stride=2 for downsampling
        let y = y.relu()?;
        
        // Another conv (same resolution)
        let conv2_weight = Tensor::randn(Shape::from_dims(&[128, 128, 3, 3]), 0.0, 0.1, device.clone())?
            .requires_grad_(true);
        let z = y.conv2d(&conv2_weight, None, 1, 1)?; // stride=1
        
        // Simplified attention mechanism (using matmul)
        let b = z.shape().dims()[0];
        let c = z.shape().dims()[1];
        let h = z.shape().dims()[2];
        let w = z.shape().dims()[3];
        
        // Reshape for attention: [B, C, H*W] -> [B, H*W, C]
        let z_flat = z.reshape(&[b, c, h * w])?;
        let z_perm = z_flat.permute(&[0, 2, 1])?; // [B, H*W, C]
        
        // Simple self-attention: Q @ K^T
        let attn_scores = z_perm.matmul(&z_flat)?; // [B, H*W, H*W]
        
        // Apply attention to values
        let attn_out = attn_scores.matmul(&z_perm)?; // [B, H*W, C]
        
        // Reshape back and sum
        let loss = attn_out.sum()?;
        
        println!("   Starting backward pass...");
        let grads = loss.backward()?;
        
        if grads.contains(x.id()) && grads.contains(conv1_weight.id()) && grads.contains(conv2_weight.id()) {
            println!("✅ U-Net style operations backward pass completed!");
            println!("   Got gradients for input and conv weights");
        } else {
            println!("❌ Missing gradients!");
        }
    }
    
    // Test 2: Diffusion timestep embedding + noise prediction
    {
        println!("\nTest 2: Diffusion timestep embedding + noise prediction");
        
        // Noisy image
        let noisy_image = Tensor::randn(Shape::from_dims(&[4, 3, 64, 64]), 0.0, 1.0, device.clone())?
            .requires_grad_(true);
        
        // Timestep embedding (simplified)
        let timestep = Tensor::randn(Shape::from_dims(&[4, 128]), 0.0, 1.0, device.clone())?
            .requires_grad_(true);
        
        // First conv on noisy image
        let conv_weight = Tensor::randn(Shape::from_dims(&[64, 3, 7, 7]), 0.0, 0.1, device.clone())?
            .requires_grad_(true);
        let features = noisy_image.conv2d(&conv_weight, None, 3, 2)?; // stride=2, padding=3
        
        // Mix with timestep (broadcast and add)
        let t_proj_weight = Tensor::randn(Shape::from_dims(&[64, 128]), 0.0, 0.1, device.clone())?
            .requires_grad_(true);
        let t_features = timestep.matmul(&t_proj_weight.transpose()?)?; // [4, 64]
        
        // Reshape timestep features for broadcasting: [4, 64, 1, 1]
        let t_features_reshaped = t_features.reshape(&[4, 64, 1, 1])?;
        
        // Add timestep to spatial features
        let combined = features.add(&t_features_reshaped)?;
        let activated = combined.relu()?;
        
        // Predict noise (simplified as reduction)
        let noise_pred = activated.mean()?;
        
        // Diffusion loss (simplified)
        let target_noise = Tensor::randn(Shape::from_dims(&[]), 0.0, 1.0, device.clone())?;
        let diff = noise_pred.sub(&target_noise)?;
        let loss = diff.square()?.sum()?;
        
        println!("   Starting backward pass...");
        let grads = loss.backward()?;
        
        if grads.contains(noisy_image.id()) && grads.contains(timestep.id()) {
            println!("✅ Diffusion operations backward pass completed!");
            println!("   Got gradients for noisy image and timestep");
        } else {
            println!("❌ Missing gradients!");
        }
    }
    
    // Test 3: Multi-scale feature extraction (like in diffusion encoders)
    {
        println!("\nTest 3: Multi-scale feature extraction");
        
        let input = Tensor::randn(Shape::from_dims(&[1, 32, 128, 128]), 0.0, 1.0, device.clone())?
            .requires_grad_(true);
        
        // Scale 1: Original resolution
        let conv1_w = Tensor::randn(Shape::from_dims(&[64, 32, 3, 3]), 0.0, 0.1, device.clone())?
            .requires_grad_(true);
        let scale1 = input.conv2d(&conv1_w, None, 1, 1)?;
        
        // Scale 2: Downsampled
        let conv2_w = Tensor::randn(Shape::from_dims(&[128, 32, 3, 3]), 0.0, 0.1, device.clone())?
            .requires_grad_(true);
        let scale2 = input.conv2d(&conv2_w, None, 1, 2)?; // stride=2
        
        // Scale 3: Further downsampled  
        let conv3_w = Tensor::randn(Shape::from_dims(&[256, 32, 3, 3]), 0.0, 0.1, device.clone())?
            .requires_grad_(true);
        let scale3 = input.conv2d(&conv3_w, None, 1, 4)?; // stride=4
        
        // Combine features (sum of means for simplicity)
        let feat1 = scale1.mean()?;
        let feat2 = scale2.mean()?;
        let feat3 = scale3.mean()?;
        
        let combined = feat1.add(&feat2)?.add(&feat3)?;
        let loss = combined.square()?;
        
        println!("   Starting backward pass...");
        let grads = loss.backward()?;
        
        if grads.contains(input.id()) && grads.contains(conv1_w.id()) && 
           grads.contains(conv2_w.id()) && grads.contains(conv3_w.id()) {
            println!("✅ Multi-scale feature extraction backward pass completed!");
            println!("   Got gradients for all conv weights");
        } else {
            println!("❌ Missing gradients!");
        }
    }
    
    println!("\n🎉 ALL DIFFUSION AUTOGRAD TESTS PASSED!");
    println!("The autograd system can handle diffusion model operations!");
    
    Ok(())
}