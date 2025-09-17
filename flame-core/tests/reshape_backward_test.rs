use flame_core::{Tensor, Shape, CudaDevice};

// Sanity test: grads flow through reshape -> matmul chain
#[test]
fn reshape_matmul_backward_has_param_grads() {
    // If CUDA is unavailable this test will bail; it’s intended to run on a CUDA box.
    let dev = match CudaDevice::new(0) {
        Ok(d) => d, // already Arc<CudaDevice>
        Err(_) => return, // gracefully skip on non-CUDA CI
    };

    // x: [2,3], w: [3,4]
    let x = Tensor::randn(Shape::from_dims(&[2, 3]), 0.0, 1.0, dev.clone())
        .unwrap()
        .requires_grad_(true);
    let w = Tensor::randn(Shape::from_dims(&[3, 4]), 0.0, 0.1, dev.clone())
        .unwrap()
        .requires_grad_(true);

    // Insert reshape views in the path deliberately
    let x6_1 = x.reshape(&[6, 1]).unwrap();
    let x2_3 = x6_1.reshape(&[2, 3]).unwrap();
    let y = x2_3.matmul(&w).unwrap(); // [2,4]
    let loss = y.square().unwrap().sum().unwrap();

    assert!(loss.requires_grad());

    // Backward and get gradient map
    let grads = flame_core::autograd::backward(&loss, false).unwrap();

    // Ensure w has a nonzero gradient (reshape must be on the tape)
    let w_grad = grads.get(w.id()).expect("Missing grad for w");
    let val = w_grad.square().unwrap().sum().unwrap().to_vec().unwrap();
    assert!(val[0] > 0.0, "expected non-zero gradient for w");
}
