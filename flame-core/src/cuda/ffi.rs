// Low-level CUDA FFI declarations for narrow kernels

extern "C" {
    pub fn flame_narrow_strided_launch(
        src: *const core::ffi::c_void,
        dst: *mut core::ffi::c_void,
        rank: i32,
        out_shape_host: *const i64,
        src_strides_host: *const i64,
        out_strides_host: *const i64,
        dim: i32,
        start: i64,
        elem_size: i64,
        n_elements: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub fn flame_narrow_backward_scatter_add_launch(
        grad_out: *const core::ffi::c_void,
        grad_in: *mut core::ffi::c_void,
        rank: i32,
        out_shape_host: *const i64,
        in_strides_host: *const i64,
        out_strides_host: *const i64,
        dim: i32,
        start: i64,
        elem_size: i64,
        n_elements: i64,
        dtype_tag: i32, // 0=F32,1=F16,2=BF16,3=I32
        stream: *mut core::ffi::c_void,
    ) -> i32;
}
