use libloading::{Library, Symbol};
use std::sync::OnceLock;

/// Runtime bindings for the FlashAttention shared library.
///
/// The library is discovered lazily the first time attention is invoked. We look
/// for an explicit `FLASH_ATTN_LIB` override first, then fall back to common
/// system install locations. The symbol we bind to matches the C shim provided
/// by the FlashAttention runtime (`fa_bf16_forward`).
pub struct FlashLib {
    pub fa_bf16_forward: unsafe extern "C" fn(
        q: *const core::ffi::c_void,
        k: *const core::ffi::c_void,
        v: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        batch: i32,
        heads: i32,
        q_tokens: i32,
        kv_tokens: i32,
        dim: i32,
        scale: f32,
        causal: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32,
    _lib: Library,
}

static FLASH_LIB: OnceLock<Result<FlashLib, String>> = OnceLock::new();

/// Resolve the FlashAttention shared library once per process.
pub fn get_flash() -> Result<&'static FlashLib, String> {
    FLASH_LIB
        .get_or_init(|| {
            let candidates = [
                std::env::var("FLASH_ATTN_LIB").ok(),
                Some("libflash_attn.so".to_string()),
                Some("/usr/local/lib/libflash_attn.so".to_string()),
                Some("/usr/lib/libflash_attn.so".to_string()),
            ];

            for path in candidates.into_iter().flatten() {
                if let Ok(lib) = unsafe { Library::new(&path) } {
                    unsafe {
                        let symbol: Symbol<
                            unsafe extern "C" fn(
                                *const core::ffi::c_void,
                                *const core::ffi::c_void,
                                *const core::ffi::c_void,
                                *mut core::ffi::c_void,
                                i32,
                                i32,
                                i32,
                                i32,
                                i32,
                                f32,
                                i32,
                                *mut core::ffi::c_void,
                            ) -> i32,
                        > = lib
                            .get(b"fa_bf16_forward\0")
                            .map_err(|err| err.to_string())?;

                        return Ok(FlashLib {
                            fa_bf16_forward: *symbol,
                            _lib: lib,
                        });
                    }
                }
            }

            Err("flash_attn library not found; set FLASH_ATTN_LIB to override".into())
        })
        .as_ref()
        .map_err(|err| err.clone())
}
