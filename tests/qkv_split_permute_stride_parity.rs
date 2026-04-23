//! Phase 2b-5 regression: `qkv_split_permute_bf16` produces identical
//! (Q, K, V) outputs for (A) a narrow-view qkv input and (B) the same
//! qkv materialized contig. Tests the strided path's indexing.
//!
//! Hot-path callers (linear3d output) always pass a contig qkv, so the
//! strided path is exercised only by the test. Still worth covering —
//! the one-dispatch-path discipline says the strided path must be
//! bit-correct if the dispatcher is going to route to it.

#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use flame_core::{global_cuda_device, DType, Shape, Tensor};

fn hash_fill(shape: &[usize]) -> Vec<f32> {
    let total: usize = shape.iter().product();
    (0..total)
        .map(|i| ((i as u32).wrapping_mul(2_654_435_761) % 1024) as f32 / 512.0 - 1.0)
        .collect()
}

#[test]
fn qkv_split_permute_contig_matches_strided_on_narrow_view() {
    // Build a wider tensor and narrow to get a strided qkv view.
    // B=2 so the B stride on qkv matters.
    let (b, n, heads, head_dim) = (2, 32, 4, 32);
    let qkv_width = 3 * heads * head_dim; // 384

    // Wide input with extra padding in the last dim, so narrow(2, 0, qkv_width)
    // produces a view with stride[1] = wide_width (not qkv_width).
    let wide_width = qkv_width + 64;
    let wide_data = hash_fill(&[b, n, wide_width]);
    let wide = Tensor::from_vec(wide_data, Shape::from_dims(&[b, n, wide_width]), global_cuda_device().clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let _ = global_cuda_device();

    let qkv_strided = wide.narrow(2, 0, qkv_width).unwrap();
    assert_eq!(qkv_strided.shape().dims(), &[b, n, qkv_width]);
    assert!(!qkv_strided.is_contiguous(), "narrow view should be strided");

    let (q_s, k_s, v_s) =
        flame_core::bf16_ops::qkv_split_permute_bf16(&qkv_strided, heads, head_dim).unwrap();

    let qkv_contig = qkv_strided.contiguous().unwrap();
    assert!(qkv_contig.is_contiguous());
    let (q_c, k_c, v_c) =
        flame_core::bf16_ops::qkv_split_permute_bf16(&qkv_contig, heads, head_dim).unwrap();

    for (label, a, bb) in [
        ("Q", q_s.to_vec().unwrap(), q_c.to_vec().unwrap()),
        ("K", k_s.to_vec().unwrap(), k_c.to_vec().unwrap()),
        ("V", v_s.to_vec().unwrap(), v_c.to_vec().unwrap()),
    ] {
        assert_eq!(a.len(), bb.len());
        for (i, (x, y)) in a.iter().zip(bb.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{label} strided vs contig diverges at idx {i}: strided={x} contig={y}"
            );
        }
    }
}

#[test]
fn qkv_split_permute_strided_math_check() {
    // Hand-verify permute math: q[b,h,n,d] == qkv[b, n, 0*H*D + h*D + d], etc.
    let (b, n, heads, head_dim) = (2, 8, 3, 16);
    let qkv_width = 3 * heads * head_dim;
    let wide_width = qkv_width + 32;

    let wide_data = hash_fill(&[b, n, wide_width]);
    let wide = Tensor::from_vec(wide_data.clone(), Shape::from_dims(&[b, n, wide_width]), global_cuda_device().clone())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let qkv = wide.narrow(2, 0, qkv_width).unwrap();

    let (q, k, v) = flame_core::bf16_ops::qkv_split_permute_bf16(&qkv, heads, head_dim).unwrap();
    let q_vec = q.to_vec().unwrap();
    let k_vec = k.to_vec().unwrap();
    let v_vec = v.to_vec().unwrap();

    // Rust side: q[b,h,n,d] contig = q_vec[((b*heads + h)*n + n_i)*head_dim + d].
    // Source: qkv[b, n, slot*H*D + h*D + d]; qkv is a view of wide with
    // offset=0 and stride[1]=wide_width, so the source linear index in
    // wide_data is b*n*wide_width + n_i*wide_width + slot*H*D + h*D + d.
    let hd = heads * head_dim;
    for bi in 0..b {
        for hi in 0..heads {
            for ni in 0..n {
                for di in 0..head_dim {
                    let flat_out = ((bi * heads + hi) * n + ni) * head_dim + di;
                    let src_q = bi * n * wide_width + ni * wide_width + 0 * hd + hi * head_dim + di;
                    let src_k = bi * n * wide_width + ni * wide_width + 1 * hd + hi * head_dim + di;
                    let src_v = bi * n * wide_width + ni * wide_width + 2 * hd + hi * head_dim + di;
                    let exp_q = bf16_roundtrip(wide_data[src_q]);
                    let exp_k = bf16_roundtrip(wide_data[src_k]);
                    let exp_v = bf16_roundtrip(wide_data[src_v]);
                    assert_eq!(
                        q_vec[flat_out].to_bits(),
                        exp_q.to_bits(),
                        "Q mismatch at (b={bi},h={hi},n={ni},d={di}): got={} exp={}",
                        q_vec[flat_out], exp_q
                    );
                    assert_eq!(k_vec[flat_out].to_bits(), exp_k.to_bits());
                    assert_eq!(v_vec[flat_out].to_bits(), exp_v.to_bits());
                }
            }
        }
    }
}

fn bf16_roundtrip(x: f32) -> f32 {
    let bits = x.to_bits();
    let rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16;
    f32::from_bits((rounded as u32) << 16)
}
