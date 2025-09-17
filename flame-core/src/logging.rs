#[macro_export]
macro_rules! gemm_tag {
    ($name:expr, $m:expr, $n:expr, $dtype:expr) => {{
        if std::env::var("ALLOC_LOG").ok().as_deref() == Some("1") {
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

