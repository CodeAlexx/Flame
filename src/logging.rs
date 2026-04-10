use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

/// Hot-path cache for ALLOC_LOG (read on every GEMM via `gemm_tag!`).
/// A direct `std::env::var` call is a syscall — this caches the result once
/// per process so the macro reduces to an atomic load when alloc logging is
/// disabled (the default).
#[doc(hidden)]
#[inline]
pub fn alloc_log_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("ALLOC_LOG").ok().as_deref() == Some("1"))
}

#[macro_export]
macro_rules! gemm_tag {
    ($name:expr, $m:expr, $n:expr, $dtype:expr) => {{
        if $crate::logging::alloc_log_enabled() {
            let bytes = ($m as usize * $n as usize) * $dtype.size_in_bytes();
            if bytes >= (8 << 20) {
                eprintln!(
                    "[alloc] tag=gemm/out({}) dtype={:?} shape=[{},{}] bytes={}",
                    $name, $dtype, $m, $n, bytes
                );
            }
        }
    }};
}

static LOG_ONCE_TAGS: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

pub(crate) fn log_once_internal(tag: &str, message: String) {
    let cache = LOG_ONCE_TAGS.get_or_init(|| Mutex::new(HashSet::new()));
    if let Ok(mut seen) = cache.lock() {
        if seen.insert(tag.to_string()) {
            eprintln!("{message}");
        }
    }
}

#[macro_export]
macro_rules! log_once {
    ($tag:expr, $($arg:tt)*) => {{
        $crate::logging::log_once_internal($tag, format!($($arg)*));
    }};
}
