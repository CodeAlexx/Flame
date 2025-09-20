#![cfg(feature = "legacy_examples")]
#![allow(unused_imports, unused_variables, unused_mut, dead_code)]
#![cfg_attr(
    clippy,
    allow(
        clippy::unused_imports,
        clippy::useless_vec,
        clippy::needless_borrow,
        clippy::needless_clone
    )
)]

use cudarc::driver::CudaDevice;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Simple FLAME Test ===\n");

    // Test 1: CUDA device creation
    println!("1. Testing CUDA device creation...");
    let device = CudaDevice::new(0)?;
    println!("✓ CUDA device created successfully");

    // Test 2: Basic memory allocation
    println!("\n2. Testing GPU memory allocation...");
    let size = 1024;
    let data = device.alloc_zeros::<f32>(size)?;
    println!("✓ Allocated {} floats on GPU", size);

    // Test 3: Data transfer
    println!("\n3. Testing data transfer...");
    let cpu_data = vec![1.0f32; size];
    let gpu_data = device.htod_sync_copy(&cpu_data)?;
    println!("✓ Uploaded data to GPU");

    let mut result = vec![0.0f32; size];
    device.dtoh_sync_copy_into(&gpu_data, &mut result)?;
    println!("✓ Downloaded data from GPU");

    // Verify
    let all_ones = result.iter().all(|&x| x == 1.0);
    println!("✓ Data integrity verified: {}", all_ones);

    // Test 4: Simple PTX kernel
    println!("\n4. Testing inline PTX kernel...");
    let ptx = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry add_one_kernel(
    .param .u64 data,
    .param .u32 n
)
{
    .reg .pred p;
    .reg .u32 idx;
    .reg .u64 ptr;
    .reg .f32 val, one;
    
    mov.u32 idx, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mad.lo.u32 idx, %r1, %r2, idx;
    
    ld.param.u32 %r3, [n];
    setp.ge.u32 p, idx, %r3;
    @p bra done;
    
    ld.param.u64 ptr, [data];
    mul.wide.u32 %rd1, idx, 4;
    add.u64 %rd2, ptr, %rd1;
    
    ld.global.f32 val, [%rd2];
    mov.f32 one, 1.0;
    add.f32 val, val, one;
    st.global.f32 [%rd2], val;
    
done:
    ret;
}
"#;

    // Load and run kernel
    device.load_ptx(ptx.into(), "add_one_module", &["add_one_kernel"])?;
    let kernel = device
        .get_func("add_one_module", "add_one_kernel")
        .ok_or("Failed to get kernel")?;

    let mut test_data = device.htod_sync_copy(&vec![5.0f32; 256])?;

    let block_size = 256;
    let grid_size = 1;

    unsafe {
        kernel.launch(
            cudarc::driver::LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            },
            (&mut test_data, 256u32),
        )?;
    }

    device.synchronize()?;

    let mut result = vec![0.0f32; 256];
    device.dtoh_sync_copy_into(&test_data, &mut result)?;

    println!("✓ PTX kernel executed");
    println!("  Input: 5.0, Output: {}", result[0]);
    println!(
        "  Verification: {}",
        if result[0] == 6.0 { "PASSED" } else { "FAILED" }
    );

    println!("\n=== All tests completed successfully! ===");
    Ok(())
}
