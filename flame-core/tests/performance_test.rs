use flame_core::{Tensor, CudaDevice, Shape};
use std::time::Instant;

#[test]
#[ignore] // Run with cargo test -- --ignored
fn benchmark_weight_updates() {
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    
    // Test different sizes
    let sizes = vec![(128, 128), (512, 512), (1024, 1024)];
    
    for (rows, cols) in sizes {
        println!("\nTesting {}x{} tensor:", rows, cols);
        
        let mut weight = Tensor::randn(
            Shape::from_dims(&[rows, cols]),
            0.0,
            0.02,
            device.clone()
        ).expect("Failed to create weight");
        
        let gradient = Tensor::randn(
            Shape::from_dims(&[rows, cols]),
            0.0,
            0.01,
            device.clone()
        ).expect("Failed to create gradient");
        
        // Warmup
        weight.update_weights(&gradient, 0.01).expect("Warmup failed");
        
        // Time 100 updates
        let start = Instant::now();
        for _ in 0..100 {
            weight.update_weights(&gradient, 0.01).expect("Update failed");
        }
        let elapsed = start.elapsed();
        
        println!("  100 updates took: {:?}", elapsed);
        println!("  Per update: {:?}", elapsed / 100);
        
        // Compare with theoretical memory bandwidth
        let bytes_moved = (rows * cols * 4 * 3) as f64; // read weight, read grad, write weight
        let gb_per_update = bytes_moved / 1e9;
        let updates_per_sec = 100.0 / elapsed.as_secs_f64();
        let gb_per_sec = gb_per_update * updates_per_sec;
        
        println!("  Effective bandwidth: {:.2} GB/s", gb_per_sec);
        println!("  (Note: Currently using CPU roundtrip, real GPU kernel would be ~100x faster)");
    }
}