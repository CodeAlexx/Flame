#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

#[cfg(feature = "strict_bf16")]
use flame_core::strict;
use flame_core::{global_cuda_device, DType, Result, Shape, Tensor};

fn linspace(count: usize) -> Vec<f32> {
    (0..count).map(|i| (i as f32) * 0.5 - 1.25).collect()
}

#[test]
fn repeat_matches_reference_nd() -> Result<()> {
    let device = global_cuda_device();
    let shape = Shape::from_dims(&[2, 3, 2]);
    let data = linspace(shape.elem_count());
    let base = Tensor::from_vec_dtype(data.clone(), shape.clone(), device.clone(), DType::BF16)?;
    let repeats = [2, 1, 3];

    let repeated = base.repeat(&repeats)?;
    assert_eq!(repeated.dtype(), DType::BF16);

    let expected_len: usize = repeats
        .iter()
        .zip(shape.dims().iter())
        .map(|(r, d)| r * d)
        .product();
    assert_eq!(repeated.shape().elem_count(), expected_len);

    let input_host = base.to_vec()?;
    let repeated_host = repeated.to_vec()?;

    let in_dims = shape.dims();
    let out_dims = repeated.shape().dims();

    let mut expected = vec![0f32; repeated_host.len()];
    for idx in 0..expected.len() {
        let mut tmp = idx;
        let mut coords = vec![0usize; out_dims.len()];
        for (d, coord) in coords.iter_mut().enumerate().rev() {
            *coord = tmp % out_dims[d];
            tmp /= out_dims[d];
        }

        let mut in_index = 0usize;
        let mut stride = 1usize;
        for d in (0..in_dims.len()).rev() {
            let coord = coords[d] / repeats[d];
            in_index += coord * stride;
            stride *= in_dims[d];
        }
        expected[idx] = input_host[in_index];
    }

    for (idx, (&got, &exp)) in repeated_host.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-3,
            "repeat mismatch at {}: got {}, expected {}",
            idx,
            got,
            exp
        );
    }

    Ok(())
}

#[test]
fn repeat_all_ones_aliases_storage() -> Result<()> {
    let device = global_cuda_device();
    let shape = Shape::from_dims(&[4, 2, 3, 5]);
    let data = linspace(shape.elem_count());
    let base = Tensor::from_vec_dtype(data, shape, device.clone(), DType::BF16)?;

    let view = base.repeat(&[1, 1, 1, 1])?;
    assert_eq!(view.shape().dims(), base.shape().dims());
    assert_eq!(view.dtype(), DType::BF16);
    assert_ne!(base.id(), view.id());

    let base_host = base.to_vec()?;
    let view_host = view.to_vec()?;
    assert_eq!(base_host, view_host);

    Ok(())
}

#[cfg(feature = "strict_bf16")]
#[test]
fn repeat_panics_under_strict_for_non_bf16() -> Result<()> {
    strict::force_runtime_enforced();
    std::env::set_var("STRICT_BF16_MODE", "panic");

    let device = global_cuda_device();
    let shape = Shape::from_dims(&[2]);
    let tensor = Tensor::from_vec(vec![1.0f32, 2.0], shape, device.clone())?;

    let result = std::panic::catch_unwind(|| {
        let _ = tensor.repeat(&[2]);
    });

    strict::disable_runtime_enforced();
    std::env::remove_var("STRICT_BF16_MODE");

    assert!(result.is_err(), "repeat should panic in STRICT_BF16 mode");
    Ok(())
}
