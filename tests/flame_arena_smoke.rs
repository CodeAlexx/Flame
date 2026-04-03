#![cfg(all(feature = "cuda", feature = "bf16_u16"))]

use anyhow::Result;
use flame_core::cuda_ops_ffi::CudaStream;
use flame_core::device::{CudaStreamRawPtrExt, Device};
use flame_core::staging::{
    arena_alloc, arena_record_and_release, arena_reset, bf16_copy_async, bf16_zero_async,
    borrow_bf16_arena_buffer, borrow_bf16_arena_tensor, d2h_async, h2d_async,
};
use flame_core::{Shape, Tensor};
use half::bf16;
use std::ffi::c_void;

#[test]
fn flame_arena_smoke() -> Result<()> {
    let device = Device::cuda(0)?;
    let stream = CudaStream::from_raw(device.cuda_device().cuda_stream_raw_ptr());
    let device_idx = device.ordinal() as i32;

    // Start from a clean arena state.
    arena_reset(device_idx, &stream)?;

    let bytes: usize = 1 << 20; // 1 MiB
    let align = 128;
    let elems = bytes / 2; // bf16 elements (2 bytes each)

    let scratch_a = arena_alloc(device_idx, &stream, bytes, align)?;
    let scratch_b = arena_alloc(device_idx, &stream, bytes, align)?;

    bf16_zero_async(scratch_a as *mut _, elems, &stream)?;

    let host_src: Vec<u16> = (0..elems).map(|i| ((i * 13) % 65535) as u16).collect();
    h2d_async(
        scratch_b as *mut _,
        host_src.as_ptr() as *const _,
        bytes,
        &stream,
    )?;

    bf16_copy_async(scratch_a as *mut _, scratch_b as *const _, elems, &stream)?;

    let mut host_dst = vec![0u16; elems];
    d2h_async(
        host_dst.as_mut_ptr() as *mut _,
        scratch_a as *const _,
        bytes,
        &stream,
    )?;

    arena_record_and_release(device_idx, &stream)?;
    device.synchronize()?;
    arena_reset(device_idx, &stream)?;

    assert!(host_dst.iter().take(64).eq(host_src.iter().take(64)));

    Ok(())
}

#[test]
fn arena_bf16_tensor_clone_roundtrip() -> Result<()> {
    let device = Device::cuda(0)?;
    let stream = CudaStream::from_raw(device.cuda_device().cuda_stream_raw_ptr());
    let elems = 1024usize;
    let device_arc = device.cuda_device_arc();

    let buffer = borrow_bf16_arena_buffer(device_arc.clone(), &stream, elems, 128)?;
    let mut tensor = buffer.into_tensor(Shape::from_dims(&[1, elems]))?;

    let ptr = tensor.as_mut_device_ptr_bf16("arena_bf16_tensor_clone_roundtrip")?;
    bf16_zero_async(ptr as *mut c_void, elems, &stream)?;
    device.synchronize()?;

    let zeros_host = tensor.to_vec_bf16()?;
    assert!(zeros_host.iter().all(|&v| v == 0));

    // Cloning should materialize owning BF16 storage.
    let clone: Tensor = tensor.clone_result()?;
    assert_eq!(clone.dtype(), flame_core::DType::BF16);

    drop(tensor);
    device.synchronize()?;

    // Borrow again after drop to ensure the lease recorded successfully.
    let buffer2 = borrow_bf16_arena_buffer(device_arc.clone(), &stream, elems, 128)?;
    let mut tensor2 = buffer2.into_tensor(Shape::from_dims(&[1, elems]))?;
    let ptr2 = tensor2.as_mut_device_ptr_bf16("arena_bf16_tensor_clone_roundtrip.borrow_again")?;
    bf16_zero_async(ptr2 as *mut c_void, elems, &stream)?;
    device.synchronize()?;

    drop(tensor2);
    drop(clone);
    device.synchronize()?;

    Ok(())
}

#[test]
fn arena_bf16_tensor_helper_roundtrip() -> Result<()> {
    let device = Device::cuda(0)?;
    let stream = CudaStream::from_raw(device.cuda_device().cuda_stream_raw_ptr());
    let device_arc = device.cuda_device_arc();
    let shape = Shape::from_dims(&[8, 128]);

    let mut tensor = borrow_bf16_arena_tensor(device_arc.clone(), &stream, shape.clone(), 128)?;
    let ptr = tensor.as_mut_device_ptr_bf16("arena_bf16_tensor_helper_roundtrip")?;
    bf16_zero_async(ptr as *mut c_void, shape.elem_count(), &stream)?;
    device.synchronize()?;

    let host = tensor.to_vec_bf16()?;
    assert!(host.iter().all(|&v| v == 0));

    drop(tensor);
    arena_record_and_release(device.ordinal() as i32, &stream)?;
    device.synchronize()?;
    Ok(())
}

#[test]
fn tensor_to_vec_bf16_roundtrip() -> Result<()> {
    let device = Device::cuda(0)?;
    let cuda = device.cuda_device_arc();
    let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
    let tensor = Tensor::from_vec(data.clone(), Shape::from_dims(&[4, 4]), cuda.clone())?
        .to_dtype(flame_core::DType::BF16)?;
    let host = tensor.to_vec_bf16()?;
    assert_eq!(host.len(), data.len());
    for (expected, raw) in data.iter().zip(host.iter()) {
        let val = bf16::from_bits(*raw).to_f32();
        assert!(
            (expected - val).abs() < 1e-3,
            "expected {expected}, got {val}"
        );
    }
    Ok(())
}
