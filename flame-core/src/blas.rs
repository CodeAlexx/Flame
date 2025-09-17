use crate::error::FlameError;

/// Raw cuBLAS BF16 GEMM bridge – disabled in this build.
/// Provide a stub that returns CuBlas error to avoid compile-time linkage to low-level symbols.
#[allow(unused_variables)]
pub fn gemm_bf16_fp32(
    _h: *mut std::ffi::c_void,
    _op_a: i32, _op_b: i32,
    _m: i32, _n: i32, _k: i32,
    _alpha: f32,
    _a_bf16: *const std::ffi::c_void, _lda: i32,
    _b_bf16: *const std::ffi::c_void, _ldb: i32,
    _beta: f32,
    _c_bf16: *mut std::ffi::c_void, _ldc: i32,
) -> Result<(), FlameError> {
    Err(FlameError::CuBlas)
}
