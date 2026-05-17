//! Multi-tensor in-place scale vs per-tensor `mul_scalar` parity.
//!
//! `multi_tensor_scale_inplace_packed` collapses N per-parameter
//! `mul_scalar` launches (the trainer clip-grad path) into one grid-per-
//! tensor kernel launch. Math is pointwise (`x[i] * scale`) — no reduction
//! order, no associativity drift. F32 path is bit-exact; BF16 path matches
//! the per-tensor BF16 `mul_scalar` within 1 ULP because both go through
//! the same `__bfloat162float → float-multiply → __float2bfloat16` cast
//! chain.

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::ops::multi_tensor::{multi_tensor_scale_inplace_packed, MultiTensorMetaCache};
use flame_core::{global_cuda_device, DType, Shape, Tensor};

fn deterministic_data(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut x = seed
        .wrapping_mul(2862933555777941757)
        .wrapping_add(3037000493);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (x >> 32) as u32;
            let normalized = (bits as f32 / u32::MAX as f32) * 2.0 - 1.0;
            normalized * scale
        })
        .collect()
}

fn make_shapes(count: usize) -> Vec<Shape> {
    (0..count)
        .map(|i| {
            let r = 8 + (i % 3) * 4;
            let dim = 1024 + (i % 5) * 256;
            if i % 2 == 0 {
                Shape::from_dims(&[r, dim])
            } else {
                Shape::from_dims(&[dim, r])
            }
        })
        .collect()
}

/// Build an F32 packed-buffer layout (ptrs | sizes) from `tensors`.
fn pack_f32(tensors: &mut [Tensor]) -> Vec<u64> {
    let n = tensors.len();
    let mut packed: Vec<u64> = Vec::with_capacity(2 * n);
    let ptrs: Vec<u64> = tensors
        .iter_mut()
        .map(|t| t.as_mut_device_ptr_f32("mt_scale_test:f32").unwrap())
        .collect();
    packed.extend(ptrs);
    for t in tensors.iter() {
        packed.push(t.shape().elem_count() as u64);
    }
    packed
}

fn pack_bf16(tensors: &mut [Tensor]) -> Vec<u64> {
    let n = tensors.len();
    let mut packed: Vec<u64> = Vec::with_capacity(2 * n);
    let ptrs: Vec<u64> = tensors
        .iter_mut()
        .map(|t| t.as_mut_device_ptr_bf16("mt_scale_test:bf16").unwrap() as u64)
        .collect();
    packed.extend(ptrs);
    for t in tensors.iter() {
        packed.push(t.shape().elem_count() as u64);
    }
    packed
}

#[test]
fn mt_scale_f32_matches_per_tensor_bit_exact() {
    let dev = global_cuda_device();
    let shapes = make_shapes(50);
    let scale: f32 = 0.5;

    // Snapshot data used to build both sets.
    let data: Vec<Vec<f32>> = shapes
        .iter()
        .enumerate()
        .map(|(i, s)| deterministic_data(s.elem_count(), 7000 + i as u64, 0.01))
        .collect();

    // Per-tensor path: scale via mul_scalar, materialize host data.
    let per_tensor_host: Vec<Vec<f32>> = shapes
        .iter()
        .zip(data.iter())
        .map(|(s, d)| {
            let t = Tensor::from_vec(d.clone(), s.clone(), dev.clone()).unwrap();
            let scaled = t.mul_scalar(scale).unwrap();
            scaled.to_vec_f32().unwrap()
        })
        .collect();

    // Multi-tensor path: build tensors, scale in place, materialize host data.
    let mut mt_tensors: Vec<Tensor> = shapes
        .iter()
        .zip(data.iter())
        .map(|(s, d)| Tensor::from_vec(d.clone(), s.clone(), dev.clone()).unwrap())
        .collect();
    let packed = pack_f32(&mut mt_tensors);
    let mut cache = MultiTensorMetaCache::new();
    multi_tensor_scale_inplace_packed(
        &mut cache,
        &dev,
        mt_tensors.len(),
        &packed,
        scale,
        false, // F32
    )
    .expect("mt scale f32");
    let mt_host: Vec<Vec<f32>> = mt_tensors.iter().map(|t| t.to_vec_f32().unwrap()).collect();

    // Bit-exact byte comparison on the raw f32 representations.
    for (i, (a, b)) in per_tensor_host.iter().zip(mt_host.iter()).enumerate() {
        assert_eq!(
            a.len(),
            b.len(),
            "tensor[{i}] length mismatch: pt={} mt={}",
            a.len(),
            b.len()
        );
        for (j, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                av.to_bits(),
                bv.to_bits(),
                "tensor[{i}][{j}]: per-tensor={av} mt={bv} (bit-exact mismatch)"
            );
        }
    }
}

#[test]
fn mt_scale_bf16_matches_per_tensor_within_tolerance() {
    let dev = global_cuda_device();
    let shapes = make_shapes(20);
    let scale: f32 = 0.5;

    let data: Vec<Vec<f32>> = shapes
        .iter()
        .enumerate()
        .map(|(i, s)| deterministic_data(s.elem_count(), 7100 + i as u64, 0.02))
        .collect();

    // Per-tensor BF16 path.
    let per_tensor_host: Vec<Vec<f32>> = shapes
        .iter()
        .zip(data.iter())
        .map(|(s, d)| {
            let t = Tensor::from_vec(d.clone(), s.clone(), dev.clone()).unwrap();
            let tb = t.to_dtype(DType::BF16).unwrap();
            let scaled = tb.mul_scalar(scale).unwrap();
            scaled.to_dtype(DType::F32).unwrap().to_vec_f32().unwrap()
        })
        .collect();

    // Multi-tensor BF16 path.
    let mut mt_tensors: Vec<Tensor> = shapes
        .iter()
        .zip(data.iter())
        .map(|(s, d)| {
            Tensor::from_vec(d.clone(), s.clone(), dev.clone())
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        })
        .collect();
    let packed = pack_bf16(&mut mt_tensors);
    let mut cache = MultiTensorMetaCache::new();
    multi_tensor_scale_inplace_packed(
        &mut cache,
        &dev,
        mt_tensors.len(),
        &packed,
        scale,
        true, // BF16
    )
    .expect("mt scale bf16");
    let mt_host: Vec<Vec<f32>> = mt_tensors
        .iter()
        .map(|t| t.to_dtype(DType::F32).unwrap().to_vec_f32().unwrap())
        .collect();

    // BF16 has 8 mantissa bits — drift of 1 ULP is ~0.4% relative. We
    // expect *zero* drift because both paths perform the same cast chain
    // on the same inputs (`x→f32 → f32*scale → bf16`). Allow 1 ULP just
    // in case kernel compilers reorder.
    let mut max_abs: f32 = 0.0;
    let mut max_rel: f32 = 0.0;
    for (a, b) in per_tensor_host.iter().zip(mt_host.iter()) {
        for (av, bv) in a.iter().zip(b.iter()) {
            let abs = (av - bv).abs();
            let rel = abs / av.abs().max(1e-6);
            if abs > max_abs {
                max_abs = abs;
            }
            if rel > max_rel {
                max_rel = rel;
            }
        }
    }
    println!("bf16 scale parity: max_abs={max_abs:.3e} max_rel={max_rel:.3e}");
    // 1 BF16 ULP at magnitude ~0.01 is ~4e-5; scaled by 0.5 still tiny.
    assert!(max_abs <= 5e-5, "bf16 max_abs drift {max_abs} > 5e-5");
}

#[test]
fn mt_scale_scale_one_is_byte_noop() {
    let dev = global_cuda_device();
    let shape = Shape::from_dims(&[64, 256]);
    let data = deterministic_data(shape.elem_count(), 8888, 0.05);

    let mut t = Tensor::from_vec(data.clone(), shape.clone(), dev.clone()).unwrap();
    let mut mt_tensors = vec![t];
    let packed = pack_f32(&mut mt_tensors);
    let mut cache = MultiTensorMetaCache::new();
    multi_tensor_scale_inplace_packed(&mut cache, &dev, 1, &packed, 1.0, false)
        .expect("mt scale 1.0");
    t = mt_tensors.pop().unwrap();

    let out = t.to_vec_f32().unwrap();
    for (i, (orig, post)) in data.iter().zip(out.iter()).enumerate() {
        assert_eq!(
            orig.to_bits(),
            post.to_bits(),
            "scale=1.0 must be byte-noop at [{i}]: orig={orig} post={post}"
        );
    }
}

#[test]
fn mt_scale_empty_is_ok() {
    let dev = global_cuda_device();
    let mut cache = MultiTensorMetaCache::new();
    // n=0 short-circuits before allocating or launching.
    multi_tensor_scale_inplace_packed(&mut cache, &dev, 0, &[], 0.5, false).expect("empty must Ok");
    multi_tensor_scale_inplace_packed(&mut cache, &dev, 0, &[], 0.5, true)
        .expect("empty bf16 must Ok");
}

#[test]
fn mt_scale_packed_length_mismatch_errors() {
    let dev = global_cuda_device();
    let mut cache = MultiTensorMetaCache::new();
    // n=3 expects packed.len() == 6.
    let bad: Vec<u64> = vec![0; 5];
    let err = multi_tensor_scale_inplace_packed(&mut cache, &dev, 3, &bad, 0.5, false);
    assert!(err.is_err(), "length mismatch must surface an error");
}
