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

/// `FLAME_CUDA_GRAPH=1` — enable CUDA Graph capture/replay for the
/// backward pass. Eliminates per-kernel launch overhead by recording
/// all backward kernels into a graph on step 2 and replaying on step 3+.
/// Requires fixed tape structure (same batch size / sequence length).
#[inline]
pub fn cuda_graph_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    flag_enabled("FLAME_CUDA_GRAPH", &CACHED)
}

/// `FLAME_BF16_REDUCE_LEGACY=1` — force the legacy BF16→F32 cast →
/// F32 reduce → cast-back path for `Tensor::sum` / `Tensor::mean` on
/// BF16 inputs. Default OFF (i.e., the BF16-native single-kernel
/// reduce is used). Knob exists for parity bisection against the
/// pre-2026-05-12 behavior.
#[inline]
pub fn bf16_reduce_legacy() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    flag_enabled("FLAME_BF16_REDUCE_LEGACY", &CACHED)
}

/// `FLAME_ASSERT_GRAD_FLOW=1` — panic when `diagnostics::assert_grad_flow`
/// finds any parameter whose gradient is missing or zero after backward.
/// When unset, the helper returns its report without panicking, so the
/// trainer may log at warn/info level.  Designed to catch the recurring
/// "BF16 fused inference op missing autograd registration" bug class
/// before it wastes a 3000-step training run.
#[inline]
pub fn assert_grad_flow_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    flag_enabled("FLAME_ASSERT_GRAD_FLOW", &CACHED)
}

/// `FLAME_HOT_FAST_PATH_DISABLE=1` — turn off the BF16-contiguous fast path
/// added to `Tensor::{silu,gelu,add,mul,mul_scalar}` and fall back to the
/// full `TensorIterator` dispatch chain. Rollback knob: bit-equivalent and
/// autograd-equivalent to the slow path, so flipping this on must not
/// change training output, only restore the old per-op CPU latency.
#[inline]
pub fn hot_fast_path_disabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    flag_enabled("FLAME_HOT_FAST_PATH_DISABLE", &CACHED)
}
