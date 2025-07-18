use flame_core::{Tensor, CudaDevice, Shape};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA - CudaDevice::new already returns Arc<CudaDevice>
    let device = CudaDevice::new(0)?;
    println!("CUDA device initialized");

    println!("\n=== Testing Batch Matrix Multiplication ===");
    
    // Test 1: Simple 3D batch matmul
    println!("\n--- Test 1: Simple 3D BMM ---");
    let a = Tensor::randn(
        Shape::from_dims(&[2, 3, 4]),  // 2 batches of 3x4 matrices
        0.0, 1.0,
        device.clone()
    )?;
    
    let b = Tensor::randn(
        Shape::from_dims(&[2, 4, 5]),  // 2 batches of 4x5 matrices
        0.0, 1.0,
        device.clone()
    )?;
    
    println!("A shape: {:?}", a.shape().dims());
    println!("B shape: {:?}", b.shape().dims());
    
    let c = a.bmm(&b)?;
    println!("C shape: {:?} (expected [2, 3, 5])", c.shape().dims());
    
    // Verify correctness by comparing with manual computation
    let a_data = a.to_vec()?;
    let b_data = b.to_vec()?;
    let c_data = c.to_vec()?;
    
    // Check first batch, first element (row 0, col 0)
    let mut expected = 0.0f32;
    for k in 0..4 {
        expected += a_data[0 * 12 + 0 * 4 + k] * b_data[0 * 20 + k * 5 + 0];
    }
    let actual = c_data[0 * 15 + 0 * 5 + 0];
    let diff = (expected - actual).abs();
    
    println!("Manual computation check: expected={:.6}, actual={:.6}, diff={:.6}", expected, actual, diff);
    assert!(diff < 1e-5, "BMM result doesn't match manual computation");
    
    // Test 2: 2D tensors (should work like regular matmul)
    println!("\n--- Test 2: 2D BMM (like matmul) ---");
    let a2d = Tensor::randn(
        Shape::from_dims(&[3, 4]),
        0.0, 1.0,
        device.clone()
    )?;
    
    let b2d = Tensor::randn(
        Shape::from_dims(&[4, 5]),
        0.0, 1.0,
        device.clone()
    )?;
    
    let c2d = a2d.bmm(&b2d)?;
    println!("2D BMM result shape: {:?} (expected [3, 5])", c2d.shape().dims());
    
    // Compare with regular matmul
    let c2d_matmul = a2d.matmul(&b2d)?;
    let bmm_data = c2d.to_vec()?;
    let matmul_data = c2d_matmul.to_vec()?;
    
    let max_diff = bmm_data.iter()
        .zip(matmul_data.iter())
        .map(|(a, b)| (a - b).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    
    println!("Max difference between BMM and matmul: {}", max_diff);
    assert!(max_diff < 1e-5, "2D BMM should match matmul");
    
    // Test 3: Different batch sizes (broadcasting)
    println!("\n--- Test 3: Broadcasting (not yet implemented) ---");
    let a_broadcast = Tensor::randn(
        Shape::from_dims(&[1, 3, 4]),  // 1 batch
        0.0, 1.0,
        device.clone()
    )?;
    
    let b_broadcast = Tensor::randn(
        Shape::from_dims(&[5, 4, 2]),  // 5 batches
        0.0, 1.0,
        device.clone()
    )?;
    
    // This should broadcast the single batch in 'a' to match 'b'
    match a_broadcast.bmm(&b_broadcast) {
        Ok(result) => {
            println!("Broadcasting BMM succeeded! Shape: {:?}", result.shape().dims());
        }
        Err(e) => {
            println!("Broadcasting not yet implemented: {}", e);
        }
    }
    
    // Performance test
    println!("\n=== Performance Test ===");
    let large_a = Tensor::randn(
        Shape::from_dims(&[32, 64, 128]),
        0.0, 1.0,
        device.clone()
    )?;
    
    let large_b = Tensor::randn(
        Shape::from_dims(&[32, 128, 256]),
        0.0, 1.0,
        device
    )?;
    
    let start = std::time::Instant::now();
    for _ in 0..10 {
        let _ = large_a.bmm(&large_b)?;
    }
    let elapsed = start.elapsed();
    
    println!("Time for 10 BMM operations (32 batches of 64x128 @ 128x256): {:?}", elapsed);
    println!("Average time per BMM: {:?}", elapsed / 10);
    
    // Matrix dimensions info
    let total_flops = 32 * 64 * 128 * 256 * 2; // multiply-add operations
    let flops_per_sec = (total_flops as f64 * 10.0) / elapsed.as_secs_f64();
    println!("Approximate GFLOPS: {:.2}", flops_per_sec / 1e9);
    
    println!("\nAll BMM tests completed!");
    
    Ok(())
}