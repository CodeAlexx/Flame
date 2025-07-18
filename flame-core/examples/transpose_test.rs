use flame_core::{Tensor, CudaDevice, Shape};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA
    let device = CudaDevice::new(0)?;
    println!("CUDA device initialized");

    println!("\n=== Testing Transpose CUDA Kernel ===");
    
    // Test 1: Simple 2x3 matrix
    let a = Tensor::from_vec(
        vec![1.0, 2.0, 3.0,
             4.0, 5.0, 6.0],
        Shape::from_dims(&[2, 3]),
        device.clone()
    )?;
    
    println!("\nOriginal matrix (2x3):");
    let a_data = a.to_vec()?;
    for i in 0..2 {
        println!("{:?}", &a_data[i*3..(i+1)*3]);
    }
    
    let a_t = a.transpose()?;
    let a_t_data = a_t.to_vec()?;
    
    println!("\nTransposed matrix (3x2):");
    for i in 0..3 {
        println!("{:?}", &a_t_data[i*2..(i+1)*2]);
    }
    
    // Test 2: Larger matrix (8x8)
    let size = 8;
    let mut data = Vec::new();
    for i in 0..size {
        for j in 0..size {
            data.push((i * size + j) as f32);
        }
    }
    
    let b = Tensor::from_vec(
        data,
        Shape::from_dims(&[size, size]),
        device.clone()
    )?;
    
    println!("\n\nTesting {}x{} matrix transpose...", size, size);
    let b_t = b.transpose()?;
    
    // Verify correctness
    let b_data = b.to_vec()?;
    let b_t_data = b_t.to_vec()?;
    
    let mut correct = true;
    for i in 0..size {
        for j in 0..size {
            let original = b_data[i * size + j];
            let transposed = b_t_data[j * size + i];
            if (original - transposed).abs() > 1e-6 {
                println!("Error at [{}, {}]: {} != {}", i, j, original, transposed);
                correct = false;
            }
        }
    }
    
    if correct {
        println!("✓ Large matrix transpose verified correct!");
    }
    
    // Test 3: Non-square matrix (4x6)
    let c = Tensor::from_vec(
        (0..24).map(|x| x as f32).collect(),
        Shape::from_dims(&[4, 6]),
        device.clone()
    )?;
    
    println!("\n\nTesting 4x6 matrix transpose...");
    let c_t = c.transpose()?;
    
    // Check shape
    let c_shape = c_t.shape().dims();
    println!("Transposed shape: {:?} (should be [6, 4])", c_shape);
    
    // Test 4: Performance comparison
    println!("\n\nPerformance test (1000 transposes of 512x512 matrix)...");
    
    let large = Tensor::randn(
        Shape::from_dims(&[512, 512]),
        0.0, 1.0,
        device.clone()
    )?;
    
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _ = large.transpose()?;
    }
    let elapsed = start.elapsed();
    
    println!("Time for 1000 transposes: {:?}", elapsed);
    println!("Average time per transpose: {:?}", elapsed / 1000);
    
    println!("\nAll transpose tests completed!");
    
    Ok(())
}