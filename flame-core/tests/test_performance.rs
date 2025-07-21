use flame_core::{Tensor, Shape, Result};
use std::sync::Arc;
use cudarc::driver::CudaDevice;
use std::time::Instant;

/// Benchmark matrix multiplication performance
#[test]
fn benchmark_matmul_performance() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    // Test different matrix sizes
    let test_sizes = vec![
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ];
    
    println!("\nMatrix Multiplication Benchmarks:");
    println!("Size\t\tTime (ms)\tGFLOPS");
    println!("{}", "-".repeat(40));
    
    for (m, n, k) in test_sizes {
        let a = Tensor::randn(Shape::from_dims(&[m, k]), 0.0, 1.0, device.clone())?;
        let b = Tensor::randn(Shape::from_dims(&[k, n]), 0.0, 1.0, device.clone())?;
        
        // Warmup
        for _ in 0..5 {
            let _c = a.matmul(&b)?;
        }
        device.synchronize()?;
        
        // Benchmark
        let num_runs = 20;
        let start = Instant::now();
        
        for _ in 0..num_runs {
            let _c = a.matmul(&b)?;
        }
        device.synchronize()?;
        
        let elapsed = start.elapsed();
        let ms_per_op = elapsed.as_secs_f64() * 1000.0 / num_runs as f64;
        
        // Calculate GFLOPS (2*m*n*k operations per matmul)
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let gflops = (flops / 1e9) / (ms_per_op / 1000.0);
        
        println!("{}x{}x{}\t{:.2}\t\t{:.1}", m, n, k, ms_per_op, gflops);
        
        // Performance expectations (adjust based on GPU)
        let min_gflops = match m {
            128 => 100.0,
            512 => 500.0,
            1024 => 1000.0,
            2048 => 2000.0,
            _ => 100.0,
        };
        
        assert!(gflops > min_gflops, 
            "MatMul performance too low: {:.1} GFLOPS < {} GFLOPS", gflops, min_gflops);
    }
    
    println!("\n✅ MatMul performance test passed!");
    Ok(())
}

/// Benchmark convolution performance
#[test]
fn benchmark_conv2d_performance() -> Result<()> {
    let device = CudaDevice::new(0)?;
    use flame_core::conv::Conv2d;
    
    let test_configs = vec![
        (32, 3, 64, 224, 3),    // BatchSize, InChannels, OutChannels, Size, Kernel
        (16, 64, 128, 56, 3),
        (8, 128, 256, 28, 3),
        (4, 256, 512, 14, 3),
    ];
    
    println!("\nConvolution 2D Benchmarks:");
    println!("Config\t\t\t\tTime (ms)\tGFLOPS");
    println!("{}", "-".repeat(60));
    
    for (batch, in_c, out_c, size, kernel) in test_configs {
        let conv = Conv2d::new_with_bias(in_c, out_c, kernel, 1, kernel/2, device.clone(), true)?;
        let input = Tensor::randn(
            Shape::from_dims(&[batch, in_c, size, size]),
            0.0,
            1.0,
            device.clone()
        )?;
        
        // Warmup
        for _ in 0..3 {
            let _output = conv.forward(&input)?;
        }
        device.synchronize()?;
        
        // Benchmark
        let num_runs = 10;
        let start = Instant::now();
        
        for _ in 0..num_runs {
            let _output = conv.forward(&input)?;
        }
        device.synchronize()?;
        
        let elapsed = start.elapsed();
        let ms_per_op = elapsed.as_secs_f64() * 1000.0 / num_runs as f64;
        
        // Calculate approximate FLOPS
        let output_size = size; // Same due to padding
        let flops = batch as f64 * out_c as f64 * output_size as f64 * output_size as f64 
                    * kernel as f64 * kernel as f64 * in_c as f64 * 2.0;
        let gflops = (flops / 1e9) / (ms_per_op / 1000.0);
        
        println!("{}x{}x{}x{} k={}\t\t{:.2}\t\t{:.1}", 
            batch, in_c, size, size, kernel, ms_per_op, gflops);
    }
    
    println!("\n✅ Conv2D performance test passed!");
    Ok(())
}

/// Benchmark element-wise operations
#[test]
fn benchmark_elementwise_ops() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    let size = 1024 * 1024 * 16; // 16M elements (64MB)
    let shape = Shape::from_dims(&[size]);
    
    let a = Tensor::randn(shape.clone(), 0.0, 1.0, device.clone())?;
    let b = Tensor::randn(shape.clone(), 0.0, 1.0, device.clone())?;
    
    let ops = vec![
        ("add", Box::new(|a: &Tensor, b: &Tensor| a.add(b)) as Box<dyn Fn(&Tensor, &Tensor) -> Result<Tensor>>),
        ("mul", Box::new(|a: &Tensor, b: &Tensor| a.mul(b))),
        ("relu", Box::new(|a: &Tensor, _b: &Tensor| a.relu())),
        ("tanh", Box::new(|a: &Tensor, _b: &Tensor| a.tanh())),
    ];
    
    println!("\nElement-wise Operation Benchmarks (16M elements):");
    println!("Operation\tTime (ms)\tGB/s");
    println!("{}", "-".repeat(40));
    
    for (name, op) in ops {
        // Warmup
        for _ in 0..5 {
            let _result = op(&a, &b)?;
        }
        device.synchronize()?;
        
        // Benchmark
        let num_runs = 50;
        let start = Instant::now();
        
        for _ in 0..num_runs {
            let _result = op(&a, &b)?;
        }
        device.synchronize()?;
        
        let elapsed = start.elapsed();
        let ms_per_op = elapsed.as_secs_f64() * 1000.0 / num_runs as f64;
        
        // Calculate bandwidth (reading a and b, writing output)
        let bytes = size * 4 * 3; // 3 tensors, 4 bytes each
        let gb_per_sec = (bytes as f64 / 1e9) / (ms_per_op / 1000.0);
        
        println!("{}\t\t{:.2}\t\t{:.1}", name, ms_per_op, gb_per_sec);
        
        // Should achieve reasonable bandwidth (>100 GB/s on modern GPUs)
        assert!(gb_per_sec > 50.0, "{} bandwidth too low: {:.1} GB/s", name, gb_per_sec);
    }
    
    println!("\n✅ Element-wise operations performance test passed!");
    Ok(())
}

/// Benchmark reduction operations
#[test]
fn benchmark_reduction_ops() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    let test_shapes = vec![
        Shape::from_dims(&[1024, 1024]),
        Shape::from_dims(&[64, 512, 512]),
        Shape::from_dims(&[8, 128, 128, 128]),
    ];
    
    println!("\nReduction Operation Benchmarks:");
    println!("Shape\t\t\tSum (ms)\tMean (ms)");
    println!("{}", "-".repeat(50));
    
    for shape in test_shapes {
        let tensor = Tensor::randn(shape.clone(), 0.0, 1.0, device.clone())?;
        
        // Benchmark sum
        device.synchronize()?;
        let start_sum = Instant::now();
        for _ in 0..20 {
            let _sum = tensor.sum()?;
        }
        device.synchronize()?;
        let sum_time = start_sum.elapsed().as_secs_f64() * 1000.0 / 20.0;
        
        // Benchmark mean
        device.synchronize()?;
        let start_mean = Instant::now();
        for _ in 0..20 {
            let _mean = tensor.mean()?;
        }
        device.synchronize()?;
        let mean_time = start_mean.elapsed().as_secs_f64() * 1000.0 / 20.0;
        
        println!("{:?}\t\t{:.2}\t\t{:.2}", shape.dims(), sum_time, mean_time);
        
        // Reductions should be fast
        assert!(sum_time < 10.0, "Sum reduction too slow");
        assert!(mean_time < 10.0, "Mean reduction too slow");
    }
    
    println!("\n✅ Reduction operations performance test passed!");
    Ok(())
}

/// Benchmark backward pass performance
#[test]
fn benchmark_backward_performance() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    println!("\nBackward Pass Benchmarks:");
    println!("Operation\t\tForward (ms)\tBackward (ms)\tRatio");
    println!("{}", "-".repeat(60));
    
    // Test 1: Simple linear layer
    {
        let batch = 128;
        let in_features = 1024;
        let out_features = 512;
        
        let x = Tensor::randn(Shape::from_dims(&[batch, in_features]), 0.0, 1.0, device.clone())?.requires_grad();
        let w = Tensor::randn(Shape::from_dims(&[in_features, out_features]), 0.0, 0.1, device.clone())?.requires_grad();
        
        // Forward timing
        device.synchronize()?;
        let start_fwd = Instant::now();
        let y = x.matmul(&w)?;
        device.synchronize()?;
        let fwd_time = start_fwd.elapsed().as_secs_f64() * 1000.0;
        
        // Backward timing
        let loss = y.sum()?;
        device.synchronize()?;
        let start_bwd = Instant::now();
        let _grads = loss.backward()?;
        device.synchronize()?;
        let bwd_time = start_bwd.elapsed().as_secs_f64() * 1000.0;
        
        let ratio = bwd_time / fwd_time;
        println!("Linear {}x{}\t\t{:.2}\t\t{:.2}\t\t{:.1}x", 
            in_features, out_features, fwd_time, bwd_time, ratio);
        
        // Backward should be at most 3x forward
        assert!(ratio < 3.0, "Backward pass too slow relative to forward");
    }
    
    // Test 2: Complex computation graph
    {
        let size = 512;
        let x = Tensor::randn(Shape::from_dims(&[size, size]), 0.0, 1.0, device.clone())?.requires_grad();
        
        // Forward with multiple ops
        device.synchronize()?;
        let start_fwd = Instant::now();
        let h1 = x.relu()?;
        let h2 = h1.tanh()?;
        let h3 = h2.add(&x)?; // Skip connection
        let loss = h3.sum()?;
        device.synchronize()?;
        let fwd_time = start_fwd.elapsed().as_secs_f64() * 1000.0;
        
        // Backward
        device.synchronize()?;
        let start_bwd = Instant::now();
        let _grads = loss.backward()?;
        device.synchronize()?;
        let bwd_time = start_bwd.elapsed().as_secs_f64() * 1000.0;
        
        let ratio = bwd_time / fwd_time;
        println!("Complex graph\t\t{:.2}\t\t{:.2}\t\t{:.1}x", fwd_time, bwd_time, ratio);
        
        assert!(ratio < 4.0, "Complex backward pass too slow");
    }
    
    println!("\n✅ Backward pass performance test passed!");
    Ok(())
}

/// Test kernel launch overhead
#[test]
fn test_kernel_launch_overhead() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    // Small tensor to minimize computation time
    let small = Tensor::ones(Shape::from_dims(&[16]), device.clone())?;
    
    // Many small operations
    let num_ops = 1000;
    device.synchronize()?;
    let start = Instant::now();
    
    for _ in 0..num_ops {
        let _result = small.add_scalar(1.0)?;
    }
    device.synchronize()?;
    
    let elapsed = start.elapsed();
    let us_per_launch = elapsed.as_micros() as f64 / num_ops as f64;
    
    println!("\nKernel launch overhead: {:.1} µs per launch", us_per_launch);
    
    // Should be < 50µs on modern GPUs
    assert!(us_per_launch < 50.0, "Kernel launch overhead too high");
    
    println!("✅ Kernel launch overhead test passed!");
    Ok(())
}