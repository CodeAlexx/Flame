//! Diagnostic for "missing dx" in multi_use_accumulation parity test.

use flame_core::{DType, Shape, Tensor};

#[test]
fn diag_multi_use_what_ids_in_grad_map() {
    let device = flame_core::global_cuda_device();
    const BN: usize = 32;
    const D: usize = 256;
    const M: usize = 128;

    let x_data: Vec<f32> = (0..BN * D).map(|i| (i as f32 * 0.1).sin()).collect();
    let m1_data: Vec<f32> = (0..D * M).map(|i| (i as f32 * 0.13).sin() * 0.05).collect();
    let m2_data: Vec<f32> = (0..D * M).map(|i| (i as f32 * 0.17).sin() * 0.05).collect();
    let go_data: Vec<f32> = (0..BN * M).map(|i| (i as f32 * 0.07).sin() * 0.1).collect();

    let x_f32 = Tensor::from_vec(x_data, Shape::from_dims(&[BN, D]), device.clone()).unwrap();
    let m1_f32 = Tensor::from_vec(m1_data, Shape::from_dims(&[D, M]), device.clone()).unwrap();
    let m2_f32 = Tensor::from_vec(m2_data, Shape::from_dims(&[D, M]), device.clone()).unwrap();
    let go_f32 = Tensor::from_vec(go_data, Shape::from_dims(&[BN, M]), device.clone()).unwrap();

    let x = x_f32.to_dtype(DType::BF16).unwrap().requires_grad_(true);
    let m1 = m1_f32.to_dtype(DType::BF16).unwrap();
    let m2 = m2_f32.to_dtype(DType::BF16).unwrap();
    let go = go_f32.to_dtype(DType::BF16).unwrap();

    eprintln!("x.id() = {:?}  requires_grad = {}", x.id(), x.requires_grad());
    eprintln!("m1.id() = {:?}", m1.id());
    eprintln!("m2.id() = {:?}", m2.id());
    eprintln!("go.id() = {:?}", go.id());

    let y1 = x.matmul(&m1).expect("x @ m1");
    let y2 = x.matmul(&m2).expect("x @ m2");
    eprintln!("y1.id() = {:?}  requires_grad = {}", y1.id(), y1.requires_grad());
    eprintln!("y2.id() = {:?}  requires_grad = {}", y2.id(), y2.requires_grad());

    let out = y1.add(&y2).expect("y1 + y2");
    eprintln!("out.id() = {:?}  requires_grad = {}", out.id(), out.requires_grad());

    let inter = out.mul(&go).expect("out * go");
    eprintln!("inter.id() = {:?}  requires_grad = {}", inter.id(), inter.requires_grad());

    let loss = inter.sum().expect("sum");
    eprintln!("loss.id() = {:?}  requires_grad = {}", loss.id(), loss.requires_grad());

    let grads = loss.backward().expect("backward");

    eprintln!("--- ids in gradient map ---");
    for (tid, t) in grads.iter() {
        eprintln!("  id={:?}  dtype={:?}  shape={:?}", tid, t.dtype(), t.shape().dims());
    }
    eprintln!("--- end ---");

    match grads.get(x.id()) {
        Some(_) => eprintln!("dx FOUND for x.id()={:?}", x.id()),
        None => eprintln!("dx MISSING for x.id()={:?}", x.id()),
    }
}
