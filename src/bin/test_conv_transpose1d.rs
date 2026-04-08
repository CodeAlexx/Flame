//! Verify `flame_core::conv1d::conv_transpose1d` against a PyTorch reference
//! saved at `/tmp/conv_transpose1d_ref.safetensors`.
//!
//! Covers:
//!   1. BigVGAN first-stage ConvT1d(k=11, s=5, p=3)
//!   2. BigVGAN mid-stage ConvT1d(k=4, s=2, p=1)
//!   3. Grouped anti-alias filter ConvT1d(k=12, s=2, groups=C)
//!   4. Explicit `output_padding=1`.

use flame_core::{conv1d::conv_transpose1d, global_cuda_device, DType, Tensor};

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    let va = a.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    let vb = b.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    if va.len() != vb.len() {
        return f32::INFINITY;
    }
    va.iter().zip(vb.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
}

fn check(
    name: &str,
    rust: &Tensor,
    expected: &Tensor,
    threshold: f32,
) {
    let r_shape = rust.shape().dims().to_vec();
    let e_shape = expected.shape().dims().to_vec();
    let err = max_abs_diff(rust, expected);
    println!("  {name}: rust {:?} vs torch {:?}, max|Δ|={err:.6}", r_shape, e_shape);
    if r_shape != e_shape {
        panic!("{name}: SHAPE MISMATCH");
    }
    if err > threshold {
        panic!("{name}: max|Δ|={err} > {threshold}");
    }
}

fn main() {
    env_logger::init();
    let device = global_cuda_device();

    let ref_map = flame_core::serialization::load_file(
        std::path::Path::new("/tmp/conv_transpose1d_ref.safetensors"),
        &device,
    )
    .expect("load /tmp/conv_transpose1d_ref.safetensors — run the Python generator first");

    println!("=== flame-core conv_transpose1d test ===");

    // Test 1: BigVGAN first stage — k=11, stride=5, padding=3
    let y1 = conv_transpose1d(
        ref_map.get("t1_x").unwrap(),
        ref_map.get("t1_w").unwrap(),
        Some(ref_map.get("t1_b").unwrap()),
        5, 3, 0, 1,
    )
    .unwrap();
    check("Test 1 k=11 s=5 p=3", &y1, ref_map.get("t1_y").unwrap(), 0.02);

    // Test 2: BigVGAN mid stage — k=4, stride=2, padding=1
    let y2 = conv_transpose1d(
        ref_map.get("t2_x").unwrap(),
        ref_map.get("t2_w").unwrap(),
        Some(ref_map.get("t2_b").unwrap()),
        2, 1, 0, 1,
    )
    .unwrap();
    check("Test 2 k=4 s=2 p=1", &y2, ref_map.get("t2_y").unwrap(), 0.02);

    // Test 3: Grouped anti-alias filter — k=12, stride=2, groups=C
    let x3 = ref_map.get("t3_x").unwrap();
    let c3 = x3.shape().dims()[1];
    let y3 = conv_transpose1d(
        x3,
        ref_map.get("t3_w").unwrap(),
        None,
        2, 0, 0, c3,
    )
    .unwrap();
    check("Test 3 k=12 s=2 groups=C", &y3, ref_map.get("t3_y").unwrap(), 0.02);

    // Test 4: output_padding=1
    let y4 = conv_transpose1d(
        ref_map.get("t2_x").unwrap(),
        ref_map.get("t2_w").unwrap(),
        Some(ref_map.get("t2_b").unwrap()),
        2, 1, 1, 1,
    )
    .unwrap();
    check("Test 4 k=4 s=2 p=1 op=1", &y4, ref_map.get("t4_y").unwrap(), 0.02);

    println!("All conv_transpose1d cases passed.");
}
