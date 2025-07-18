use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing NVRTC kernel compilation...");
    
    // Initialize CUDA device
    let device = Arc::new(CudaDevice::new(0)?);
    println!("✓ CUDA device initialized");
    
    // Simple kernel source
    let kernel_source = r#"
extern "C" __global__ void test_kernel(
    const float* a,
    const float* b,
    float* out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}"#;
    
    // Compile the kernel
    println!("\n1. Compiling kernel with NVRTC...");
    let ptx = compile_ptx(kernel_source)?;
    println!("   ✓ Kernel compiled successfully");
    
    // Load the PTX module
    println!("\n2. Loading PTX module...");
    device.load_ptx(ptx, "test_module", &["test_kernel"])?;
    println!("   ✓ PTX module loaded");
    
    // Get the function
    println!("\n3. Getting kernel function...");
    let kernel = device.get_func("test_module", "test_kernel")
        .ok_or("Failed to get kernel function")?;
    println!("   ✓ Kernel function retrieved");
    
    // Test execution
    println!("\n4. Testing kernel execution...");
    let n = 1024;
    let a = device.alloc_zeros::<f32>(n)?;
    let b = device.alloc_zeros::<f32>(n)?;
    let mut out = device.alloc_zeros::<f32>(n)?;
    
    let block_size = 256;
    let grid_size = (n + block_size - 1) / block_size;
    
    let config = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    
    unsafe {
        kernel.launch(config, (&a, &b, &mut out, n as u32))?;
    }
    
    device.synchronize()?;
    println!("   ✓ Kernel executed successfully");
    
    println!("\n✓ NVRTC test completed successfully!");
    
    Ok(())
}