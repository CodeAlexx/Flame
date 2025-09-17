#![cfg(feature="capi")]
use crate::{Tensor, Device, DType, Shape};

#[no_mangle]
pub extern "C" fn flame_new_cuda_device(idx: i32) -> *mut Device {
    match Device::cuda(idx as usize) {
        Ok(d) => Box::into_raw(Box::new(d)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn flame_free_device(p: *mut Device) {
    if !p.is_null() { unsafe { drop(Box::from_raw(p)); } }
}

#[no_mangle]
pub extern "C" fn flame_zeros_nhwc_bf16(dev: *mut Device, b: i32, h: i32, w: i32, c: i32) -> *mut Tensor {
    let d = unsafe { &*dev };
    let res = Tensor::zeros(Shape::from_dims(&[b as usize, h as usize, w as usize, c as usize]), d.cuda_device().clone())
        .and_then(|t| t.to_dtype(DType::BF16));
    match res {
        Ok(t) => Box::into_raw(Box::new(t)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn flame_tensor_free(p: *mut Tensor) { if !p.is_null() { unsafe { drop(Box::from_raw(p)); } } }
