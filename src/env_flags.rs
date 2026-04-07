//! Cached environment flag reads for flame-core hot paths.
//!
//! Every `std::env::var(...)` is a syscall. flame-core used to read a handful
//! of debug / fallback flags on every allocation, matmul, cast, narrow, conv,
//! broadcast, and tile call — thousands of syscalls per denoise step even
//! when the flags were not set. This module exposes each flag as an inlined
//! function that caches its first read via `OnceLock`, turning the hot-path
//! cost into a single atomic load.
//!
//! Use from any module inside flame-core:
//! ```ignore
//! use crate::env_flags::sdxl_debug_shapes_enabled;
//! if sdxl_debug_shapes_enabled() { ... }
//! ```

#[inline]
fn flag_enabled(var: &'static str, cache: &'static std::sync::OnceLock<bool>) -> bool {
    *cache.get_or_init(|| std::env::var(var).ok().as_deref() == Some("1"))
}

#[inline]
fn flag_present(var: &'static str, cache: &'static std::sync::OnceLock<bool>) -> bool {
    *cache.get_or_init(|| std::env::var(var).is_ok())
}

/// `ALLOC_LOG=1` — print a line for every large tensor allocation.
#[inline]
pub fn alloc_log_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    flag_enabled("ALLOC_LOG", &CACHED)
}

/// `FLAME_TRACE_DTYPE=1` — print every `Tensor::matmul` call with dtypes.
#[inline]
pub fn trace_dtype_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    flag_enabled("FLAME_TRACE_DTYPE", &CACHED)
}

/// `FLAME_DTYPE_TRACE=1` — print every dtype cast path.
#[inline]
pub fn dtype_trace_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    flag_enabled("FLAME_DTYPE_TRACE", &CACHED)
}

/// `SDXL_DEBUG_SHAPES=1` — debug shape-mismatch traces (narrow, tile, broadcast,
/// tensor_ext).
#[inline]
pub fn sdxl_debug_shapes_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    flag_enabled("SDXL_DEBUG_SHAPES", &CACHED)
}

/// `FLAME_TRACE_VERBOSE=1` — verbose GEMM trace (already cached inside gemm.rs;
/// this helper mirrors it for consistency).
#[inline]
pub fn trace_verbose_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    flag_enabled("FLAME_TRACE_VERBOSE", &CACHED)
}

/// `FLAME_NO_CUDNN_CONV=<anything>` — disable cuDNN conv2d fast path and
/// fall back to the custom NHWC kernel. Checked on every Conv2d::forward.
#[inline]
pub fn no_cudnn_conv() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    flag_present("FLAME_NO_CUDNN_CONV", &CACHED)
}

/// `FORCE_F32_CONV=<anything>` — force the F32 conv fallback. Checked on
/// every Conv2d::forward and Conv2d::forward_nhwc.
#[inline]
pub fn force_f32_conv() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    flag_present("FORCE_F32_CONV", &CACHED)
}

/// `FLAME_CUBLASLT_FORCE_FALLBACK=1` — force the BF16 GEMM fallback instead
/// of the cuBLASLt fast path. Checked in `gemm_bf16`.
#[inline]
pub fn cublaslt_force_fallback() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    flag_enabled("FLAME_CUBLASLT_FORCE_FALLBACK", &CACHED)
}
