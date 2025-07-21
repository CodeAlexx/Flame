use flame_core::{Tensor, Shape};
use cudarc::driver::CudaDevice;
use std::time::Instant;

fn main() -> flame_core::Result<()> {
    println!("FLAME Performance Reality Check\n");
    
    let device = CudaDevice::new(0)?;
    
    // Test 1: Matrix multiplication performance
    println!("1. Testing MatMul Performance:");
    let sizes = vec![(512, 512), (1024, 1024)];
    
    for (m, n) in sizes {
        let a = Tensor::randn(Shape::from_dims(&[m, n]), 0.0, 1.0, device.clone())?;
        let b = Tensor::randn(Shape::from_dims(&[n, m]), 0.0, 1.0, device.clone())?;
        
        // Warmup
        for _ in 0..3 {
            let _c = a.matmul(&b)?;
        }
        device.synchronize()?;
        
        // Benchmark
        let start = Instant::now();
        let num_runs = 10;
        for _ in 0..num_runs {
            let _c = a.matmul(&b)?;
        }
        device.synchronize()?;
        
        let elapsed = start.elapsed();
        let ms_per_op = elapsed.as_secs_f64() * 1000.0 / num_runs as f64;
        let gflops = (2.0 * m as f64 * n as f64 * m as f64 / 1e9) / (ms_per_op / 1000.0);
        
        println!("  {}x{}: {:.2} ms/op, {:.1} GFLOPS", m, n, ms_per_op, gflops);
    }
    
    // Test 2: Memory allocation stress
    println!("\n2. Testing Memory Allocation:");
    let start = Instant::now();
    let mut tensors = Vec::new();
    
    for i in 0..100 {
        let size = 1024 * 1024; // 4MB tensor
        let t = Tensor::zeros(Shape::from_dims(&[size]), device.clone())?;
        tensors.push(t);
        
        if i % 20 == 19 {
            tensors.clear(); // Test deallocation
        }
    }
    
    let alloc_time = start.elapsed();
    println!("  100 allocations in {:.2} ms", alloc_time.as_secs_f64() * 1000.0);
    
    // Test 3: Autograd functionality
    println!("\n3. Testing Autograd:");
    let x = Tensor::randn(Shape::from_dims(&[128, 64]), 0.0, 1.0, device.clone())?.requires_grad_(true);
    let w = Tensor::randn(Shape::from_dims(&[64, 32]), 0.0, 0.1, device.clone())?.requires_grad_(true);
    
    match x.matmul(&w) {
        Ok(y) => {
            match y.sum() {
                Ok(loss) => {
                    match loss.backward() {
                        Ok(_grads) => println!("  ✓ Forward and backward pass completed"),
                        Err(e) => println!("  ✗ Backward pass failed: {:?}", e),
                    }
                }
                Err(e) => println!("  ✗ Sum failed: {:?}", e),
            }
        }
        Err(e) => println!("  ✗ MatMul failed: {:?}", e),
    }
    
    println!("\nPerformance test complete.");
    Ok(())
}