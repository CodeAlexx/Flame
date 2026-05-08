//! Multi-tensor fused Adam kernel parity vs the per-tensor fused Adam kernel.
//!
//! Both paths share the SAME kernel math (DECOUPLED weight decay, identical
//! moment-update sequence). The only difference is the launch shape: one
//! kernel covering N tensors versus N kernels each covering one tensor.
//! After one step from the same starting state, the two paths must produce
//! BIT-IDENTICAL outputs — any drift would indicate a kernel bug, not a
//! tolerance issue.
//!
//! Test setup:
//!   - 50 BF16 params with LoRA-shaped sizes (mix of [r,dim] and [dim,r],
//!     varied dim).
//!   - F32 grads, F32 m, F32 v.
//!   - Same deterministic random data on both paths.
//!   - One Adam step.
//!   - Compare param/m/v byte-for-byte after the step.
//!
//! If the two byte-streams ever differ, the multi-tensor kernel has drifted
//! from the single-tensor kernel and the migration is unsafe.

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use cudarc::driver::{DevicePtr, DevicePtrMut};
use flame_core::adam::{adam_fused_multi_tensor_step, adam_fused_step, MultiTensorMetaCache};
use flame_core::{global_cuda_device, DType, Shape, Tensor};

mod parity_helpers;

fn make_shapes() -> Vec<Shape> {
    let mut out = Vec::with_capacity(50);
    for i in 0..50usize {
        let r = 16usize;
        let dim = 256 + (i % 5) * 128; // 256, 384, 512, 640, 768
        if i % 2 == 0 {
            out.push(Shape::from_dims(&[r, dim]));
        } else {
            out.push(Shape::from_dims(&[dim, r]));
        }
    }
    out
}

fn deterministic_data(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    // Simple LCG. We only need a stable, reproducible byte stream — not a
    // good RNG.
    let mut x = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let bits = (x >> 32) as u32;
            // Map to [-1, 1) * scale.
            let normalized = (bits as f32 / u32::MAX as f32) * 2.0 - 1.0;
            normalized * scale
        })
        .collect()
}

/// Build a parallel set of (param, grad, m, v) tensors for a list of shapes,
/// seeded so the same call returns the same byte stream every time.
fn build_state(shapes: &[Shape]) -> (Vec<Tensor>, Vec<Tensor>, Vec<Tensor>, Vec<Tensor>) {
    let dev = global_cuda_device();
    let mut params = Vec::with_capacity(shapes.len());
    let mut grads = Vec::with_capacity(shapes.len());
    let mut ms = Vec::with_capacity(shapes.len());
    let mut vs = Vec::with_capacity(shapes.len());

    for (i, shape) in shapes.iter().enumerate() {
        let n = shape.elem_count();
        // Params: small random BF16 (deterministic seed per index).
        let p_f32 = deterministic_data(n, 1000 + i as u64, 0.05);
        let p = Tensor::from_vec(p_f32, shape.clone(), dev.clone())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        params.push(p);

        // Grads: F32 with different seed.
        let g_f32 = deterministic_data(n, 2000 + i as u64, 0.01);
        grads.push(Tensor::from_vec(g_f32, shape.clone(), dev.clone()).unwrap());

        // m, v: zeros (first-step state).
        let m = Tensor::from_vec(vec![0.0_f32; n], shape.clone(), dev.clone()).unwrap();
        let v = Tensor::from_vec(vec![0.0_f32; n], shape.clone(), dev.clone()).unwrap();
        ms.push(m);
        vs.push(v);
    }

    (params, grads, ms, vs)
}

/// Snapshot a tensor's bytes (BF16 as u16, F32 as f32) for later compare.
fn snapshot_bf16(t: &Tensor) -> Vec<u16> {
    t.to_vec_bf16().unwrap()
}
fn snapshot_f32(t: &Tensor) -> Vec<f32> {
    t.to_vec_f32().unwrap()
}

#[test]
fn multi_tensor_matches_per_tensor_one_step_bit_exact() {
    let dev = global_cuda_device();
    let shapes = make_shapes();
    let n = shapes.len();

    let (lr, beta1, beta2, eps, wd) = (3e-5_f32, 0.9_f32, 0.999_f32, 1e-8_f32, 0.01_f32);
    // Step t=1 → bias correction factors as the kernels expect them.
    let bc1 = 1.0 - beta1.powi(1);
    let bc2 = 1.0 - beta2.powi(1);

    // ---------- Path A: single multi-tensor launch ----------
    let (mut p_a, g_a, mut m_a, mut v_a) = build_state(&shapes);

    // Build the 5-region packed buffer in the launcher's expected layout.
    let mut packed = Vec::with_capacity(5 * n);
    for p in &mut p_a {
        packed.push(p.as_mut_device_ptr_bf16("test:p").unwrap() as u64);
    }
    for g in &g_a {
        let s = g.as_slice_f32("test:g").unwrap();
        packed.push(*s.device_ptr());
    }
    for m in &mut m_a {
        let s = m.as_mut_slice_f32("test:m").unwrap();
        packed.push(*s.device_ptr_mut());
    }
    for v in &mut v_a {
        let s = v.as_mut_slice_f32("test:v").unwrap();
        packed.push(*s.device_ptr_mut());
    }
    for shape in &shapes {
        packed.push(shape.elem_count() as u64);
    }

    let mut cache = MultiTensorMetaCache::new();
    adam_fused_multi_tensor_step(
        &mut cache,
        &dev,
        n,
        false, // grad_is_bf16: F32 grads
        &packed,
        lr, beta1, beta2, eps, wd,
        bc1, bc2,
        None, // stoch_seed: round-to-nearest path, parity test
    )
    .expect("multi-tensor launch");

    // Force completion before reading.
    dev.synchronize().expect("sync after multi-tensor");

    let p_a_snap: Vec<Vec<u16>> = p_a.iter().map(snapshot_bf16).collect();
    let m_a_snap: Vec<Vec<f32>> = m_a.iter().map(snapshot_f32).collect();
    let v_a_snap: Vec<Vec<f32>> = v_a.iter().map(snapshot_f32).collect();

    // ---------- Path B: per-tensor launches ----------
    let (mut p_b, g_b, mut m_b, mut v_b) = build_state(&shapes);
    for i in 0..n {
        adam_fused_step(
            &mut p_b[i],
            &g_b[i],
            &mut m_b[i],
            &mut v_b[i],
            lr, beta1, beta2, eps, wd,
            bc1, bc2,
            None,
        )
        .expect("per-tensor launch");
    }
    dev.synchronize().expect("sync after per-tensor");

    let p_b_snap: Vec<Vec<u16>> = p_b.iter().map(snapshot_bf16).collect();
    let m_b_snap: Vec<Vec<f32>> = m_b.iter().map(snapshot_f32).collect();
    let v_b_snap: Vec<Vec<f32>> = v_b.iter().map(snapshot_f32).collect();

    // ---------- Compare ----------
    for i in 0..n {
        // BF16 params: bit-exact (same kernel, same input → same u16 bits).
        assert_eq!(
            p_a_snap[i], p_b_snap[i],
            "param[{i}] (shape={:?}): multi-tensor and per-tensor diverged",
            shapes[i].dims()
        );
        // F32 m: bit-exact.
        assert_eq!(
            m_a_snap[i], m_b_snap[i],
            "m[{i}] (shape={:?}) diverged",
            shapes[i].dims()
        );
        // F32 v: bit-exact.
        assert_eq!(
            v_a_snap[i], v_b_snap[i],
            "v[{i}] (shape={:?}) diverged",
            shapes[i].dims()
        );
    }
}

#[test]
fn multi_tensor_matches_per_tensor_100_steps() {
    // Run 100 sequential steps on both paths, compare at the end. This is a
    // weaker gate than bit-exact (because an early-step kernel-math drift
    // would amplify across 100 steps and might exceed BF16 ULP), but
    // guards against pathological accumulation. We compare the BF16 params
    // at the end via parity_helpers::compare_bf16 (atol=rtol=1e-2).
    let dev = global_cuda_device();
    let shapes = make_shapes();
    let n = shapes.len();

    let (lr, beta1, beta2, eps, wd) = (3e-5_f32, 0.9_f32, 0.999_f32, 1e-8_f32, 0.01_f32);

    // Path A: multi-tensor, 100 steps.
    let (mut p_a, g_a, mut m_a, mut v_a) = build_state(&shapes);
    let mut cache = MultiTensorMetaCache::new();
    for t in 1..=100u32 {
        let bc1 = 1.0 - beta1.powi(t as i32);
        let bc2 = 1.0 - beta2.powi(t as i32);
        let mut packed = Vec::with_capacity(5 * n);
        for p in &mut p_a {
            packed.push(p.as_mut_device_ptr_bf16("test:p").unwrap() as u64);
        }
        for g in &g_a {
            packed.push(*g.as_slice_f32("test:g").unwrap().device_ptr());
        }
        for m in &mut m_a {
            packed.push(*m.as_mut_slice_f32("test:m").unwrap().device_ptr_mut());
        }
        for v in &mut v_a {
            packed.push(*v.as_mut_slice_f32("test:v").unwrap().device_ptr_mut());
        }
        for shape in &shapes {
            packed.push(shape.elem_count() as u64);
        }
        adam_fused_multi_tensor_step(
            &mut cache, &dev, n, false, &packed,
            lr, beta1, beta2, eps, wd, bc1, bc2,
            None,
        )
        .unwrap();
    }
    dev.synchronize().unwrap();

    // Path B: per-tensor, 100 steps.
    let (mut p_b, g_b, mut m_b, mut v_b) = build_state(&shapes);
    for t in 1..=100u32 {
        let bc1 = 1.0 - beta1.powi(t as i32);
        let bc2 = 1.0 - beta2.powi(t as i32);
        for i in 0..n {
            adam_fused_step(
                &mut p_b[i], &g_b[i], &mut m_b[i], &mut v_b[i],
                lr, beta1, beta2, eps, wd, bc1, bc2,
                None,
            )
            .unwrap();
        }
    }
    dev.synchronize().unwrap();

    // Compare with BF16 tolerance.
    for i in 0..n {
        let r = parity_helpers::compare_bf16(&p_a[i], &p_b[i]);
        assert!(
            r.passed,
            "param[{i}] (shape={:?}) drifted across 100 steps: {}",
            shapes[i].dims(),
            r.pretty()
        );
    }
}
