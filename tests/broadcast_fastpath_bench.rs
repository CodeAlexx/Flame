#![cfg(all(feature = "cuda", feature = "bf16_u16"))]
//! Micro-benchmark / correctness check for the same-shape fast-path in
//! `make_broadcast_spec` (bf16_elementwise.rs) and `broadcast_shape_binary_op`
//! (shape.rs). Validates both produce identical output to what the slow path
//! would have produced, and reports a rough timing breakdown so we can
//! eyeball the speedup.

use flame_core::bf16_elementwise::make_broadcast_spec;
use flame_core::Shape;
use std::time::Instant;

#[test]
fn make_broadcast_spec_same_shape_is_correct() {
    // Typical diffusion transformer shapes.
    let cases: &[&[usize]] = &[
        &[1, 4096, 3840],
        &[1, 8192, 3072],
        &[2, 1024, 256, 256],
        &[1, 3840, 64, 64],
        &[16, 4096, 128],
        &[1],
        &[512, 512],
    ];

    for dims in cases {
        let fast = make_broadcast_spec(dims, dims);
        // Slow-path reference: run the same input through a dims-mismatched-then-identical
        // path won't hit the fast-path, so instead we reconstruct what the slow path
        // would emit by computing contiguous strides manually.
        let nd = dims.len();
        let mut strides = vec![0i64; nd];
        let mut s = 1i64;
        for i in (0..nd).rev() {
            let d = dims[i] as i64;
            strides[i] = if d == 1 { 0 } else { s };
            s *= d.max(1);
        }
        let expected_total: i64 = dims.iter().map(|&d| d as i64).product();

        let fast_dims: Vec<i64> = fast.out_dims.clone();
        let expected_dims: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        assert_eq!(
            fast_dims, expected_dims,
            "out_dims mismatch for {:?}",
            dims
        );
        assert_eq!(fast.a_strides, strides, "a_strides mismatch for {:?}", dims);
        assert_eq!(fast.b_strides, strides, "b_strides mismatch for {:?}", dims);
        assert_eq!(fast.total, expected_total, "total mismatch for {:?}", dims);
    }
}

#[test]
fn make_broadcast_spec_broadcast_case_still_works() {
    // Real broadcast: a=[2,4], b=[1,4]. Must go through the slow path, which
    // the fast-path must NOT swallow.
    let spec = make_broadcast_spec(&[2, 4], &[1, 4]);
    assert_eq!(spec.out_dims, vec![2, 4]);
    // a is contiguous [2,4] → strides [4, 1]; but since all dims >1, no zeroing
    assert_eq!(spec.a_strides, vec![4, 1]);
    // b is [1,4]: stride for dim=1 → 0, dim=4 → 1
    assert_eq!(spec.b_strides, vec![0, 1]);
    assert_eq!(spec.total, 8);
}

#[test]
fn shape_broadcast_same_shape_returns_self() {
    let s = Shape::from_dims(&[1, 4096, 3840]);
    let out = s.broadcast_shape_binary_op(&s).unwrap();
    assert_eq!(out.dims(), s.dims());
}

#[test]
fn shape_broadcast_actual_broadcast_still_works() {
    let a = Shape::from_dims(&[2, 4]);
    let b = Shape::from_dims(&[1, 4]);
    let out = a.broadcast_shape_binary_op(&b).unwrap();
    assert_eq!(out.dims(), &[2, 4]);
}

#[test]
fn make_broadcast_spec_bench_same_shape() {
    // 1M iterations of the hot path (same shape) — should be blazing fast
    // on the fast-path. Just prints the time so we can eyeball it.
    let dims = &[1usize, 4096, 3840];
    let iters = 1_000_000;
    let t0 = Instant::now();
    for _ in 0..iters {
        let s = make_broadcast_spec(dims, dims);
        std::hint::black_box(s);
    }
    let elapsed = t0.elapsed();
    let ns_per_call = elapsed.as_nanos() as f64 / iters as f64;
    println!(
        "make_broadcast_spec same-shape [1,4096,3840]: {:.1} ns/call ({} iters in {:?})",
        ns_per_call, iters, elapsed
    );
}

#[test]
fn shape_broadcast_bench_same_shape() {
    let a = Shape::from_dims(&[1, 4096, 3840]);
    let iters = 1_000_000;
    let t0 = Instant::now();
    for _ in 0..iters {
        let s = a.broadcast_shape_binary_op(&a).unwrap();
        std::hint::black_box(s);
    }
    let elapsed = t0.elapsed();
    let ns_per_call = elapsed.as_nanos() as f64 / iters as f64;
    println!(
        "Shape::broadcast_shape_binary_op same-shape [1,4096,3840]: {:.1} ns/call ({} iters in {:?})",
        ns_per_call, iters, elapsed
    );
}

#[test]
fn bench_slow_path_broadcast() {
    // Force the slow path with an actual broadcast case ([1,4096,3840] vs
    // [1,1,3840] — column broadcast like a bias add). Measures what we would
    // be paying WITHOUT the fast-path.
    let a = Shape::from_dims(&[1, 4096, 3840]);
    let b = Shape::from_dims(&[1, 1, 3840]);
    let iters = 1_000_000;
    let t0 = Instant::now();
    for _ in 0..iters {
        let s = a.broadcast_shape_binary_op(&b).unwrap();
        std::hint::black_box(s);
    }
    let elapsed = t0.elapsed();
    let ns = elapsed.as_nanos() as f64 / iters as f64;
    println!("Shape::broadcast_shape_binary_op REAL broadcast [1,4096,3840] vs [1,1,3840]: {:.1} ns/call", ns);

    let t1 = Instant::now();
    for _ in 0..iters {
        let s = make_broadcast_spec(a.dims(), b.dims());
        std::hint::black_box(s);
    }
    let elapsed2 = t1.elapsed();
    let ns2 = elapsed2.as_nanos() as f64 / iters as f64;
    println!(
        "make_broadcast_spec REAL broadcast [1,4096,3840] vs [1,1,3840]: {:.1} ns/call",
        ns2
    );
}
