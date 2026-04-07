use crate::{strict, trace::trace_on, DType, Error, Result, Shape, Tensor};
use std::sync::Arc;

#[cfg(feature = "cuda")]
use crate::device::CudaStreamRawPtrExt;
#[cfg(feature = "cuda")]
use cudarc::driver::DevicePtr;

#[cfg(feature = "cuda")]
pub fn broadcast_to_impl(tensor: &Tensor, target_shape: &[i64]) -> Result<Tensor> {
    #[cfg(feature = "bf16_u16")]
    use crate::cuda::ffi::launch_broadcast_bf16;
    use crate::cuda::ffi::launch_broadcast_f32;

    strict::scope(
        "ops.broadcast_to.cuda",
        strict::GuardMode::env_default(),
        || {
            let target: Vec<usize> = target_shape
                .iter()
                .map(|&d| {
                    usize::try_from(d).map_err(|_| {
                        Error::InvalidInput(format!(
                            "broadcast_to: negative or too-large dimension {}",
                            d
                        ))
                    })
                })
                .collect::<Result<_>>()?;
            let out_shape = Shape::from_dims(&target);

            if tensor.shape() == &out_shape {
                let _clone_guard = strict::allow_clone();
                return tensor.clone_result();
            }

            let device = tensor.device();

            let dtype_in = tensor.dtype();
            let prepared = match dtype_in {
                DType::F32 => strict::allow_f32_in_kernel_scoped(|| tensor.to_dtype(DType::F32))?,
                DType::BF16 => {
                    let _guard = strict::allow_clone();
                    tensor.clone_result()?
                }
                _ => {
                    let _guard = strict::allow_clone();
                    tensor.clone_result()?
                }
            };

            let dtype = prepared.dtype();
            if crate::env_flags::sdxl_debug_shapes_enabled() {
                let ok = match dtype {
                    DType::F32 => prepared.storage_ref().try_as_slice_f32().is_ok(),
                    DType::BF16 => {
                        #[cfg(feature = "bf16_u16")]
                        {
                            prepared.storage_ref().try_as_slice_u16().is_ok()
                        }
                        #[cfg(not(feature = "bf16_u16"))]
                        {
                            true
                        }
                    }
                    _ => true,
                };
                eprintln!(
                    "[broadcast_impl] in={:?} -> out={:?} storage_ok {} shape {:?}",
                    dtype_in,
                    dtype,
                    ok,
                    prepared.shape().dims()
                );
            }

            let mut output = Tensor::zeros_dtype(out_shape.clone(), dtype, Arc::clone(device))?;

            let ndim = target.len();
            let src_dims = prepared.shape().dims();
            let src_strides = prepared.shape().strides();
            let offset = ndim - src_dims.len();

            let mut in_strides: Vec<i64> = vec![0; ndim];
            for d in 0..ndim {
                if d < offset {
                    in_strides[d] = 0;
                } else {
                    let idx = d - offset;
                    if src_dims[idx] == 1 {
                        in_strides[d] = 0;
                    } else {
                        in_strides[d] = src_strides[idx] as i64;
                    }
                }
            }

            let out_strides: Vec<i64> = out_shape.strides().into_iter().map(|s| s as i64).collect();
            let out_dims_i64: Vec<i64> = target.iter().map(|&d| d as i64).collect();
            let total = out_shape.elem_count() as i64;
            if total == 0 {
                return Ok(output);
            }

            let mut d_out_shape = unsafe { device.alloc::<i64>(ndim) }
                .map_err(|e| Error::Cuda(format!("broadcast_to: alloc out_shape failed: {:?}", e)))?;
            device
                .htod_copy_into(out_dims_i64.clone(), &mut d_out_shape)
                .map_err(|_| Error::CudaDriver)?;

            let mut d_in_strides = unsafe { device.alloc::<i64>(ndim) }.map_err(|e| {
                Error::Cuda(format!("broadcast_to: alloc in_strides failed: {:?}", e))
            })?;
            device
                .htod_copy_into(in_strides.clone(), &mut d_in_strides)
                .map_err(|_| Error::CudaDriver)?;

            let mut d_out_strides = unsafe { device.alloc::<i64>(ndim) }.map_err(|e| {
                Error::Cuda(format!("broadcast_to: alloc out_strides failed: {:?}", e))
            })?;
            device
                .htod_copy_into(out_strides.clone(), &mut d_out_strides)
                .map_err(|_| Error::CudaDriver)?;

            let stream = device.cuda_stream_raw_ptr();
            unsafe {
                match dtype {
                    DType::F32 => {
                        let src_slice =
                            prepared.storage_ref().try_as_slice_f32().map_err(|_| {
                                Error::InvalidOperation("broadcast_to: expected F32 storage".into())
                            })?;
                        let dst_slice =
                            output.storage_mut().try_as_mut_slice_f32().map_err(|_| {
                                Error::InvalidOperation("broadcast_to: expected F32 storage".into())
                            })?;
                        let src_ptr = *src_slice.device_ptr() as *const f32;
                        let dst_ptr = *dst_slice.device_ptr() as *mut f32;
                        launch_broadcast_f32(
                            src_ptr,
                            dst_ptr,
                            *d_out_shape.device_ptr() as *const i64,
                            *d_in_strides.device_ptr() as *const i64,
                            *d_out_strides.device_ptr() as *const i64,
                            ndim as i32,
                            total,
                            stream,
                        );
                    }
                    DType::BF16 => {
                        #[cfg(feature = "bf16_u16")]
                        {
                            if trace_on() {
                                eprintln!("[broadcast_impl] entering BF16 path total {}", total);
                            }
                            let src_ptr = prepared.as_device_ptr_bf16("broadcast_to:src")?
                                as *const core::ffi::c_void;
                            let dst_ptr = output.as_mut_device_ptr_bf16("broadcast_to:dst")?
                                as *mut core::ffi::c_void;
                            launch_broadcast_bf16(
                                src_ptr,
                                dst_ptr,
                                *d_out_shape.device_ptr() as *const i64,
                                *d_in_strides.device_ptr() as *const i64,
                                *d_out_strides.device_ptr() as *const i64,
                                ndim as i32,
                                total,
                                stream,
                            );
                        }
                        #[cfg(not(feature = "bf16_u16"))]
                        {
                            return Err(Error::Unsupported(
                                "broadcast_to: BF16 requires the bf16_u16 feature".into(),
                            ));
                        }
                    }
                    other => {
                        return Err(Error::Unsupported(format!(
                            "broadcast_to: dtype {:?} not supported",
                            other
                        )));
                    }
                }
            }

            Ok(output)
        },
    )
}

#[cfg(not(feature = "cuda"))]
pub fn broadcast_to_impl(tensor: &Tensor, target_shape: &[i64]) -> Result<Tensor> {
    strict::scope(
        "ops.broadcast_to.cpu",
        strict::GuardMode::env_default(),
        || {
            let target: Vec<usize> = target_shape
                .iter()
                .map(|&d| {
                    usize::try_from(d).map_err(|_| {
                        Error::InvalidInput(format!("broadcast_to: invalid dimension {}", d))
                    })
                })
                .collect::<Result<_>>()?;
            let out_shape = Shape::from_dims(&target);
            if tensor.shape() == &out_shape {
                let _clone_guard = strict::allow_clone();
                return tensor.clone_result();
            }

            let src_f32 = strict::allow_f32_in_kernel_scoped(|| tensor.to_dtype(DType::F32))?;
            let input_data = src_f32.to_vec()?;
            let mut output_data = vec![0.0f32; out_shape.elem_count()];

            let input_dims = tensor.shape().dims();
            let input_strides = tensor.shape().strides();
            let target_strides = out_shape.strides();
            let ndim = target.len();
            let offset = ndim - input_dims.len();

            for (i, out) in output_data
                .iter_mut()
                .enumerate()
                .take(out_shape.elem_count())
            {
                let mut target_idx = i;
                let mut input_idx = 0usize;

                for (d, &stride) in target_strides.iter().enumerate() {
                    let dim_idx = target_idx / stride;
                    target_idx %= stride;

                    if d >= offset {
                        let input_d = d - offset;
                        let input_dim_size = input_dims[input_d];
                        if input_dim_size > 1 {
                            input_idx += dim_idx * input_strides[input_d];
                        }
                    }
                }

                *out = input_data[input_idx];
            }

            let mut out_tensor = Tensor::from_vec_dtype(
                output_data,
                out_shape.clone(),
                tensor.device.clone(),
                DType::F32,
            )?;
            if tensor.dtype() == DType::F32 {
                Ok(out_tensor)
            } else {
                out_tensor.to_dtype(tensor.dtype())
            }
        },
    )
}
