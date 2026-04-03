use cudarc::driver::CudaDevice;
use flame_core::{Shape, Tensor};

fn main() -> flame_core::Result<()> {
    println!("=== FLAME Basic Operations Test ===\n");

    let device = CudaDevice::new(0)?;

    // Test what actually works
    println!("1. Tensor Creation:");
    let a = Tensor::zeros(Shape::from_dims(&[2, 3]), device.clone())?;
    println!("   ✓ zeros: shape {:?}", a.shape().dims());

    let b = Tensor::ones(Shape::from_dims(&[2, 3]), device.clone())?;
    println!("   ✓ ones: shape {:?}", b.shape().dims());

    let c = Tensor::randn(Shape::from_dims(&[2, 3]), 0.0, 1.0, device.clone())?;
    println!("   ✓ randn: shape {:?}", c.shape().dims());

    println!("\n2. Basic Arithmetic:");
    let _ = a.add(&b)?;
    println!("   ✓ add");

    let _ = a.mul(&b)?;
    println!("   ✓ mul");

    let _ = a.add_scalar(5.0)?;
    println!("   ✓ add_scalar");

    println!("\n3. Matrix Operations:");
    let x = Tensor::randn(Shape::from_dims(&[3, 4]), 0.0, 1.0, device.clone())?;
    let y = Tensor::randn(Shape::from_dims(&[4, 5]), 0.0, 1.0, device.clone())?;
    match x.matmul(&y) {
        Ok(z) => println!(
            "   ✓ matmul: {:?} x {:?} -> {:?}",
            x.shape().dims(),
            y.shape().dims(),
            z.shape().dims()
        ),
        Err(e) => println!("   ✗ matmul failed: {:?}", e),
    }

    println!("\n4. Reductions:");
    match c.sum() {
        Ok(_) => println!("   ✓ sum"),
        Err(e) => println!("   ✗ sum failed: {:?}", e),
    }

    match c.mean() {
        Ok(_) => println!("   ✓ mean"),
        Err(e) => println!("   ✗ mean failed: {:?}", e),
    }

    println!("\n5. Activations:");
    match c.relu() {
        Ok(_) => println!("   ✓ relu"),
        Err(e) => println!("   ✗ relu failed: {:?}", e),
    }

    match c.tanh() {
        Ok(_) => println!("   ✓ tanh"),
        Err(e) => println!("   ✗ tanh failed: {:?}", e),
    }

    println!("\n6. Shape Operations:");
    let reshaped = c.reshape(&[6])?;
    println!(
        "   ✓ reshape: {:?} -> {:?}",
        c.shape().dims(),
        reshaped.shape().dims()
    );

    println!("\n=== Summary ===");
    println!("Basic tensor operations work, but many features are missing or broken.");

    Ok(())
}
