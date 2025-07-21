//! Example demonstrating the type-safe kernel launcher
//! 
//! This example shows how to use the kernel launcher for safe CUDA kernel invocations.

use flame_core::{CudaDevice, Tensor, Shape, Result};
use flame_core::cuda_kernels_v2::CudaKernelsV2;
use flame_core::kernel_launcher::{KernelLauncher, templates};
use flame_core::kernel_params;
use std::sync::Arc;

fn main() -> Result<()> {
    // Initialize CUDA device
    let device = CudaDevice::new(0)?;
    println!("Using CUDA device: {:?}", device.ordinal());
    
    // Example 1: Using pre-built kernels
    example_prebuilt_kernels(&device)?;
    
    // Example 2: Custom kernel with type-safe parameters
    example_custom_kernel(&device)?;
    
    // Example 3: Complex kernel with parameter struct
    example_complex_kernel(&device)?;
    
    Ok(())
}

fn example_prebuilt_kernels(device: &Arc<CudaDevice>) -> Result<()> {
    println!("\n=== Example 1: Pre-built Kernels ===");
    
    let kernels = CudaKernelsV2::new(device.clone());
    
    // Create test tensors
    let a = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0],
        Shape::from_dims(&[2, 2]),
        device.clone()
    )?;
    
    let b = Tensor::from_vec(
        vec![5.0, 6.0, 7.0, 8.0],
        Shape::from_dims(&[2, 2]),
        device.clone()
    )?;
    
    // Test addition
    let sum = kernels.add(&a, &b)?;
    println!("a + b = {:?}", sum.to_vec()?);
    
    // Test multiplication
    let product = kernels.mul(&a, &b)?;
    println!("a * b = {:?}", product.to_vec()?);
    
    // Test ReLU
    let c = Tensor::from_vec(
        vec![-1.0, 2.0, -3.0, 4.0],
        Shape::from_dims(&[4]),
        device.clone()
    )?;
    let relu_result = kernels.relu(&c)?;
    println!("ReLU(c) = {:?}", relu_result.to_vec()?);
    
    // Test matrix multiplication
    let mat_result = kernels.matmul(&a, &b)?;
    println!("a @ b = {:?}", mat_result.to_vec()?);
    
    Ok(())
}

fn example_custom_kernel(device: &Arc<CudaDevice>) -> Result<()> {
    println!("\n=== Example 2: Custom Kernel ===");
    
    let launcher = KernelLauncher::new(device.clone());
    
    // Create a custom kernel for x^2 + 3x + 2
    let kernel_code = templates::elementwise_unary(
        "polynomial_kernel",
        "x * x + 3.0f * x + 2.0f"
    );
    
    let input = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0],
        Shape::from_dims(&[4]),
        device.clone()
    )?;
    
    let numel = input.shape().elem_count();
    let mut output = unsafe { device.alloc::<f32>(numel) }
        .map_err(|_| flame_core::FlameError::CudaDriver)?;
    
    let _ = launcher.prepare_kernel("polynomial_kernel", &kernel_code)?;
    
    let f = device.get_func("polynomial_kernel", "polynomial_kernel")
        .ok_or_else(|| flame_core::FlameError::Cuda("Failed to get polynomial_kernel".into()))?;
    
    let config = flame_core::kernel_launcher::launch_configs::elementwise(numel);
    
    unsafe {
        f.launch(config, (
            &mut output,
            &*input.data(),
            numel as i32,
        ))?;
    }
    
    let result = flame_core::cuda_kernels_v2::create_output_tensor(
        output,
        input.shape().clone(),
        device.clone()
    );
    
    println!("f(x) = x² + 3x + 2");
    println!("Input: {:?}", input.to_vec()?);
    println!("Output: {:?}", result.to_vec()?);
    
    Ok(())
}

fn example_complex_kernel(device: &Arc<CudaDevice>) -> Result<()> {
    println!("\n=== Example 3: Complex Kernel with Parameters ===");
    
    // Define a parameter structure for a weighted sum kernel
    kernel_params! {
        struct WeightedSumParams {
            alpha: f32,
            beta: f32,
            gamma: f32,
        }
    }
    
    let launcher = KernelLauncher::new(device.clone());
    
    // Kernel that computes: output = alpha * a + beta * b + gamma
    let kernel_code = r#"
struct WeightedSumParams {
    float alpha;
    float beta;
    float gamma;
};

extern "C" __global__ void weighted_sum_kernel(
    float* output,
    const float* a,
    const float* b,
    WeightedSumParams params,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = params.alpha * a[idx] + params.beta * b[idx] + params.gamma;
    }
}"#;
    
    let a = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0],
        Shape::from_dims(&[4]),
        device.clone()
    )?;
    
    let b = Tensor::from_vec(
        vec![5.0, 6.0, 7.0, 8.0],
        Shape::from_dims(&[4]),
        device.clone()
    )?;
    
    let params = WeightedSumParams {
        alpha: 0.3,
        beta: 0.7,
        gamma: 1.0,
    };
    
    let numel = a.shape().elem_count();
    let mut output = unsafe { device.alloc::<f32>(numel) }
        .map_err(|_| flame_core::FlameError::CudaDriver)?;
    
    let _ = launcher.prepare_kernel("weighted_sum_kernel", kernel_code)?;
    
    let f = device.get_func("weighted_sum_kernel", "weighted_sum_kernel")
        .ok_or_else(|| flame_core::FlameError::Cuda("Failed to get weighted_sum_kernel".into()))?;
    
    let config = flame_core::kernel_launcher::launch_configs::elementwise(numel);
    
    unsafe {
        f.launch(config, (
            &mut output,
            &*a.data(),
            &*b.data(),
            params,
            numel as i32,
        ))?;
    }
    
    let result = flame_core::cuda_kernels_v2::create_output_tensor(
        output,
        a.shape().clone(),
        device.clone()
    );
    
    println!("Weighted sum: {} * a + {} * b + {}", params.alpha, params.beta, params.gamma);
    println!("a = {:?}", a.to_vec()?);
    println!("b = {:?}", b.to_vec()?);
    println!("Result = {:?}", result.to_vec()?);
    
    Ok(())
}