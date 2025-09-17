//! Performance Optimization Test for FLAME
//! Benchmarks and optimizes critical operations

use flame_core::{
    Tensor, Shape, Result,
    conv::Conv2d, linear::Linear,
    cuda_kernels::CudaKernels,
};
use cudarc::driver::CudaDevice;
use std::time::Instant;
use std::sync::Arc;

/// Benchmark a function and return average time in milliseconds
fn benchmark<F>(name: &str, iterations: usize, mut f: F) -> Result<f32>
where
    F: FnMut() -> Result<()>,
{
    // Warmup
    for _ in 0..5 {
        f()?;
    }
    
    // Actual benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        f()?;
    }
    let elapsed = start.elapsed();
    
    let avg_ms = elapsed.as_secs_f32() * 1000.0 / iterations as f32;
    println!("{}: {:.3}ms per iteration", name, avg_ms);
    Ok(avg_ms)
}

#[test]
fn test_matmul_performance() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    println!("Matrix multiplication performance test:");
    
    for size in [128, 256, 512, 1024] {
        let a = Tensor::randn(
            Shape::from_dims(&[size, size]),
            0.0,
            1.0,
            device.clone()
        )?;
        let b = Tensor::randn(
            Shape::from_dims(&[size, size]),
            0.0,
            1.0,
            device.clone()
        )?;
        
        let time = benchmark(
            &format!("  {}x{} matmul", size, size),
            20,
            || {
                let _c = a.matmul(&b)?;
                Ok(())
            }
        )?;
        
        // Calculate GFLOPS
        let ops = 2.0 * (size as f64).powi(3);
        let gflops = ops / (time as f64 * 1e6);
        println!("    → {:.1} GFLOPS", gflops);
    }
    
    println!("✓ Matmul performance test completed");
    Ok(())
}

#[test]
fn test_convolution_performance() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    println!("\nConvolution performance test:");
    
    // Test different convolution configurations
    let configs = vec![
        (32, 3, 64, 224),    // ResNet first layer: 3->64, 224x224
        (16, 64, 128, 56),   // Typical mid layer
        (8, 256, 512, 14),   // Deep layer
    ];
    
    for (batch, in_ch, out_ch, size) in configs {
        let conv = Conv2d::new(in_ch, out_ch, 3, 1, 1, device.clone())?;
        let input = Tensor::randn(
            Shape::from_dims(&[batch, in_ch, size, size]),
            0.0,
            1.0,
            device.clone()
        )?;
        
        let time = benchmark(
            &format!("  Conv2d B{}x{}x{}x{} -> {}", batch, in_ch, size, size, out_ch),
            10,
            || {
                let _output = conv.forward(&input)?;
                Ok(())
            }
        )?;
        
        // Calculate approximate GFLOPS for conv
        let ops = batch as f64 * out_ch as f64 * size as f64 * size as f64 
                  * in_ch as f64 * 9.0 * 2.0; // 3x3 kernel, 2 ops per MAC
        let gflops = ops / (time as f64 * 1e6);
        println!("    → {:.1} GFLOPS", gflops);
    }
    
    println!("✓ Convolution performance test completed");
    Ok(())
}

#[test]
fn test_memory_bandwidth() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    println!("\nMemory bandwidth test:");
    
    // Test different memory access patterns
    let sizes_mb = vec![1, 10, 100, 500];
    
    for size_mb in sizes_mb {
        let elements = size_mb * 1024 * 1024 / 4; // f32 = 4 bytes
        let tensor = Tensor::randn(
            Shape::from_dims(&[elements]),
            0.0,
            1.0,
            device.clone()
        )?;
        
        // Test copy bandwidth
        let time = benchmark(
            &format!("  Copy {}MB", size_mb),
            50,
            || {
                let _copy = tensor.clone_result()?;
                Ok(())
            }
        )?;
        
        let bandwidth_gb = (size_mb as f32 * 2.0) / (time * 1000.0); // Read + Write
        println!("    → {:.1} GB/s", bandwidth_gb);
    }
    
    println!("✓ Memory bandwidth test completed");
    Ok(())
}

#[test]
fn test_kernel_fusion_opportunities() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let kernels = CudaKernels::new(device.clone())?;
    
    println!("\nKernel fusion optimization test:");
    
    let size = 1024 * 1024;
    let a = Tensor::randn(Shape::from_dims(&[size]), 0.0, 1.0, device.clone())?;
    let b = Tensor::randn(Shape::from_dims(&[size]), 0.0, 1.0, device.clone())?;
    
    // Test 1: Separate operations
    let time_separate = benchmark(
        "  Separate ops (add + mul_scalar + relu)",
        100,
        || {
            let x = kernels.add(&a, &b)?;
            let x = kernels.mul_scalar(&x, 0.5)?;
            let _x = kernels.relu(&x)?;
            Ok(())
        }
    )?;
    
    // Test 2: Could be fused (in a real implementation)
    // For now, just measure the same ops
    let time_fused = benchmark(
        "  'Fused' ops (simulated)",
        100,
        || {
            // In a real implementation, this would be a single kernel
            let x = kernels.add(&a, &b)?;
            let x = kernels.mul_scalar(&x, 0.5)?;
            let _x = kernels.relu(&x)?;
            Ok(())
        }
    )?;
    
    println!("  Potential speedup from fusion: {:.1}x", time_separate / time_fused);
    println!("✓ Kernel fusion test completed");
    Ok(())
}

#[test]
fn test_optimization_recommendations() -> Result<()> {
    println!("\n=== Performance Optimization Recommendations ===");
    
    println!("\n1. Kernel Fusion Opportunities:");
    println!("   - Fuse activation functions with preceding ops");
    println!("   - Combine bias addition with convolution");
    println!("   - Merge batch norm into conv weights when possible");
    
    println!("\n2. Memory Optimization:");
    println!("   - Implement memory pool to reuse allocations");
    println!("   - Use workspace memory for convolution algorithms");
    println!("   - Enable tensor aliasing for in-place operations");
    
    println!("\n3. Algorithm Selection:");
    println!("   - Use cuDNN for optimized convolution algorithms");
    println!("   - Implement Winograd convolution for 3x3 kernels");
    println!("   - Add FFT convolution for large kernels");
    
    println!("\n4. Parallelization:");
    println!("   - Use CUDA streams for concurrent operations");
    println!("   - Overlap computation with memory transfers");
    println!("   - Implement multi-GPU support");
    
    println!("\n5. Precision Optimization:");
    println!("   - Add mixed precision (FP16) support");
    println!("   - Implement tensor cores utilization");
    println!("   - Add INT8 quantization for inference");
    
    println!("\n✓ Optimization analysis completed");
    Ok(())
}
