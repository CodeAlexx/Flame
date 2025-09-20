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

use flame_core::{CudaDevice, Result, Shape, Tensor};

fn main() -> Result<()> {
    println!("🔬 Testing FLAME sum_dims operation...\n");

    let device = CudaDevice::new(0)?;

    // Test sum_dims
    {
        println!("Test: sum_dims");

        // Create a 3x4 tensor with known values
        let data = vec![
            1.0, 2.0, 3.0, 4.0, // row 0, sum = 10
            5.0, 6.0, 7.0, 8.0, // row 1, sum = 26
            9.0, 10.0, 11.0, 12.0, // row 2, sum = 42
        ];

        let tensor = Tensor::from_vec(data, Shape::from_dims(&[3, 4]), device.clone())?;

        println!("  Input tensor shape: {:?}", tensor.shape().dims());
        println!("  Input data:");
        println!("    [1, 2, 3, 4]");
        println!("    [5, 6, 7, 8]");
        println!("    [9, 10, 11, 12]");

        // Sum along dimension 0 (sum columns)
        let sum_dim0 = tensor.sum_dims(&[0])?;
        println!("\n  Sum along dim 0: {:?}", sum_dim0.to_vec()?);
        println!("  Expected: [15.0, 18.0, 21.0, 24.0]");
        println!("  Shape after sum: {:?}", sum_dim0.shape().dims());

        // Sum along dimension 1 (sum rows)
        let sum_dim1 = tensor.sum_dims(&[1])?;
        println!("\n  Sum along dim 1: {:?}", sum_dim1.to_vec()?);
        println!("  Expected: [10.0, 26.0, 42.0]");
        println!("  Shape after sum: {:?}", sum_dim1.shape().dims());

        // Total sum
        let total_sum = tensor.sum()?;
        println!("\n  Total sum: {}", total_sum.to_vec()?[0]);
        println!("  Expected: 78.0");
    }

    Ok(())
}
