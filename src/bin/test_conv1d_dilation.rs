//! Verify `flame_core::conv1d::conv1d` handles the `dilation` parameter
//! correctly, by comparing against a PyTorch reference saved at
//! `/tmp/conv1d_dilation_ref.safetensors`.

use flame_core::{conv1d::conv1d, global_cuda_device, DType, Tensor};

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    let va = a.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    let vb = b.to_dtype(DType::F32).unwrap().to_vec().unwrap();
    assert_eq!(va.len(), vb.len());
    va.iter().zip(vb.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
}

fn check_case(ref_map: &std::collections::HashMap<String, Tensor>, dilation: usize, padding: usize, key: &str) {
    let x = ref_map.get("x").unwrap();
    let w = ref_map.get("w").unwrap();
    let b = ref_map.get("b").unwrap();
    let expected = ref_map.get(key).unwrap();
    let out = conv1d(x, w, Some(b), 1, padding, dilation, 1).unwrap();
    let err = max_abs_diff(&out, expected);
    println!(
        "  dilation={dilation} padding={padding}: rust {:?} vs torch {:?}, max|Δ|={err:.6}",
        out.shape().dims(),
        expected.shape().dims()
    );
    if out.shape().dims() != expected.shape().dims() {
        panic!("SHAPE MISMATCH for dilation={dilation}");
    }
    assert!(err < 1e-2, "dilation={dilation} max|Δ|={err} > 1e-2");
}

fn main() {
    env_logger::init();
    let device = global_cuda_device();

    let ref_map = flame_core::serialization::load_file(
        std::path::Path::new("/tmp/conv1d_dilation_ref.safetensors"),
        &device,
    )
    .expect("load /tmp/conv1d_dilation_ref.safetensors — run the Python generator first");

    println!("=== flame-core conv1d dilation test ===");
    check_case(&ref_map, 1, 1, "y_d1");
    check_case(&ref_map, 2, 2, "y_d2");
    check_case(&ref_map, 3, 3, "y_d3");
    println!("All conv1d dilation cases passed.");
}
