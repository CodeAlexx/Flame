#[cfg(feature = "strict_bf16")]
mod inner {
    use crate::{Result, Shape};
    use std::cell::{Cell, RefCell};
    use std::fmt;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::OnceLock;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum GuardMode {
        Warn,
        Panic,
    }

    impl GuardMode {
        /// Called on every `strict::scope` invocation, which wraps most
        /// flame-core ops. Previously read `STRICT_BF16_MODE` via
        /// `std::env::var` on every call (one syscall per op). Now cached
        /// once via `OnceLock`.
        pub fn env_default() -> Self {
            static MODE: OnceLock<GuardMode> = OnceLock::new();
            *MODE.get_or_init(|| {
                if let Ok(value) = std::env::var("STRICT_BF16_MODE") {
                    let value = value.to_ascii_lowercase();
                    if matches!(value.as_str(), "panic" | "strict") {
                        return GuardMode::Panic;
                    }
                    if matches!(value.as_str(), "warn" | "relaxed") {
                        return GuardMode::Warn;
                    }
                }
                GuardMode::Warn
            })
        }
    }

    #[derive(Clone, Copy)]
    struct GuardContext {
        tag: &'static str,
        mode: GuardMode,
    }

    thread_local! {
        static ALLOW_CLONE_DEPTH: Cell<u32> = Cell::new(0);
        static ALLOW_F32_IN_KERNEL_DEPTH: Cell<u32> = Cell::new(0);
        static SCOPE_STACK: RefCell<Vec<GuardContext>> = RefCell::new(Vec::new());
    }

    static GRAPH_CAST_COUNT: AtomicUsize = AtomicUsize::new(0);
    static CLONE_ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
    static LAYOUT_FIX_COUNT: AtomicUsize = AtomicUsize::new(0);
    static PARAM_F32_STORE_COUNT: AtomicUsize = AtomicUsize::new(0);
    static RUNTIME_OVERRIDE: AtomicBool = AtomicBool::new(false);

    fn runtime_enforced() -> bool {
        if RUNTIME_OVERRIDE.load(Ordering::Relaxed) {
            return true;
        }
        static STRICT: OnceLock<bool> = OnceLock::new();
        *STRICT.get_or_init(|| {
            std::env::var("STRICT_BF16")
                .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "on"))
                .unwrap_or(false)
        })
    }

    fn snapshot_counts() -> [usize; 4] {
        [
            GRAPH_CAST_COUNT.load(Ordering::Relaxed),
            CLONE_ALLOC_COUNT.load(Ordering::Relaxed),
            LAYOUT_FIX_COUNT.load(Ordering::Relaxed),
            PARAM_F32_STORE_COUNT.load(Ordering::Relaxed),
        ]
    }

    pub fn force_runtime_enforced() {
        RUNTIME_OVERRIDE.store(true, Ordering::Relaxed);
    }

    pub fn disable_runtime_enforced() {
        RUNTIME_OVERRIDE.store(false, Ordering::Relaxed);
    }

    fn push_context(ctx: GuardContext) {
        SCOPE_STACK.with(|stack| stack.borrow_mut().push(ctx));
    }

    fn pop_context() -> Option<GuardContext> {
        SCOPE_STACK.with(|stack| stack.borrow_mut().pop())
    }

    fn current_context() -> GuardContext {
        SCOPE_STACK.with(|stack| {
            stack.borrow().last().copied().unwrap_or(GuardContext {
                tag: "<global>",
                mode: GuardMode::env_default(),
            })
        })
    }

    fn effective_mode(mode: GuardMode) -> GuardMode {
        match mode {
            GuardMode::Warn => GuardMode::Warn,
            GuardMode::Panic => {
                if runtime_enforced() {
                    GuardMode::Panic
                } else {
                    GuardMode::Warn
                }
            }
        }
    }

    fn violation(kind: &str, location: &'static str, shape: &Shape) {
        let context = current_context();
        let message = format!(
            "[STRICT_BF16] op={} kind={} location={} shape={:?}",
            context.tag,
            kind,
            location,
            shape.dims()
        );
        match effective_mode(context.mode) {
            GuardMode::Warn => log::warn!("{}", message),
            GuardMode::Panic => panic!("{}", message),
        }
    }

    struct ScopeGuard {
        tag: &'static str,
        start_counts: [usize; 4],
    }

    impl Drop for ScopeGuard {
        fn drop(&mut self) {
            let end = snapshot_counts();
            if let Some(popped) = pop_context() {
                debug_assert_eq!(popped.tag, self.tag, "STRICT_BF16 scope stack imbalance");
            }

            let delta = [
                end[0].saturating_sub(self.start_counts[0]),
                end[1].saturating_sub(self.start_counts[1]),
                end[2].saturating_sub(self.start_counts[2]),
                end[3].saturating_sub(self.start_counts[3]),
            ];

            let message = format!(
                "[strict] op={} f32_graph_casts={} clone_allocs={} layout_fixes={} param_f32_store={}",
                self.tag, delta[0], delta[1], delta[2], delta[3]
            );

            if delta.iter().any(|&v| v > 0) {
                log::info!("{}", message);
            } else {
                log::debug!("{}", message);
            }
        }
    }

    pub struct AllowCloneGuard {
        _private: (),
    }

    impl Drop for AllowCloneGuard {
        fn drop(&mut self) {
            ALLOW_CLONE_DEPTH.with(|depth| {
                let current = depth.get();
                debug_assert!(current > 0, "ALLOW_CLONE guard underflow");
                depth.set(current - 1);
            });
        }
    }

    pub struct AllowF32InKernelGuard {
        _private: (),
    }

    impl Drop for AllowF32InKernelGuard {
        fn drop(&mut self) {
            ALLOW_F32_IN_KERNEL_DEPTH.with(|depth| {
                let current = depth.get();
                debug_assert!(current > 0, "ALLOW_F32_IN_KERNEL guard underflow");
                depth.set(current - 1);
            });
        }
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct StrictTelemetry {
        pub f32_graph_casts: usize,
        pub clone_allocs: usize,
        pub layout_fixes: usize,
        pub param_f32_store: usize,
        pub strict_bf16: bool,
    }

    impl fmt::Display for StrictTelemetry {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                f,
                "StrictTelemetry(strict={}, f32_graph_casts={}, clone_allocs={}, layout_fixes={}, param_f32_store={})",
                self.strict_bf16,
                self.f32_graph_casts,
                self.clone_allocs,
                self.layout_fixes,
                self.param_f32_store
            )
        }
    }

    pub fn scope<F, T>(tag: &'static str, mode: GuardMode, body: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        push_context(GuardContext { tag, mode });
        let guard = ScopeGuard {
            tag,
            start_counts: snapshot_counts(),
        };
        let result = body();
        drop(guard);
        result
    }

    pub fn allow_clone() -> AllowCloneGuard {
        ALLOW_CLONE_DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
        AllowCloneGuard { _private: () }
    }

    pub fn allow_f32_in_kernel() -> AllowF32InKernelGuard {
        ALLOW_F32_IN_KERNEL_DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
        AllowF32InKernelGuard { _private: () }
    }

    pub fn record_bf16_to_f32(location: &'static str, shape: &Shape) {
        let allowed = ALLOW_F32_IN_KERNEL_DEPTH.with(|depth| depth.get() > 0);
        if allowed {
            return;
        }
        GRAPH_CAST_COUNT.fetch_add(1, Ordering::Relaxed);
        violation("f32_graph_cast", location, shape);
    }

    pub fn record_clone(location: &'static str, shape: &Shape) {
        let allowed = ALLOW_CLONE_DEPTH.with(|depth| depth.get() > 0);
        if allowed {
            return;
        }
        CLONE_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        violation("clone_alloc", location, shape);
    }

    pub fn record_reshape_clone(location: &'static str, shape: &Shape) {
        let allowed = ALLOW_CLONE_DEPTH.with(|depth| depth.get() > 0);
        if allowed {
            return;
        }
        LAYOUT_FIX_COUNT.fetch_add(1, Ordering::Relaxed);
        CLONE_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        violation("layout_fix", location, shape);
    }

    pub fn record_layout_fix(location: &'static str, shape: &Shape) {
        LAYOUT_FIX_COUNT.fetch_add(1, Ordering::Relaxed);
        violation("layout_fix", location, shape);
    }

    pub fn record_param_f32_store(location: &'static str, shape: &Shape) {
        PARAM_F32_STORE_COUNT.fetch_add(1, Ordering::Relaxed);
        violation("param_f32_store", location, shape);
    }

    pub fn telemetry_snapshot() -> StrictTelemetry {
        let counts = snapshot_counts();
        StrictTelemetry {
            f32_graph_casts: counts[0],
            clone_allocs: counts[1],
            layout_fixes: counts[2],
            param_f32_store: counts[3],
            strict_bf16: is_enabled(),
        }
    }

    pub fn reset_counters() {
        GRAPH_CAST_COUNT.store(0, Ordering::Relaxed);
        CLONE_ALLOC_COUNT.store(0, Ordering::Relaxed);
        LAYOUT_FIX_COUNT.store(0, Ordering::Relaxed);
        PARAM_F32_STORE_COUNT.store(0, Ordering::Relaxed);
    }

    pub fn is_enabled() -> bool {
        runtime_enforced()
    }

    pub fn allow_f32_in_kernel_scoped<F, T>(body: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        let _guard = allow_f32_in_kernel();
        body()
    }
}

#[cfg(not(feature = "strict_bf16"))]
mod inner {
    use crate::{Result, Shape};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum GuardMode {
        Warn,
        Panic,
    }

    impl GuardMode {
        pub fn env_default() -> Self {
            GuardMode::Warn
        }
    }

    pub struct AllowCloneGuard {
        _private: (),
    }

    pub struct AllowF32InKernelGuard {
        _private: (),
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct StrictTelemetry {
        pub f32_graph_casts: usize,
        pub clone_allocs: usize,
        pub layout_fixes: usize,
        pub param_f32_store: usize,
        pub strict_bf16: bool,
    }

    pub fn scope<F, T>(_tag: &'static str, _mode: GuardMode, body: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        body()
    }

    pub fn allow_clone() -> AllowCloneGuard {
        AllowCloneGuard { _private: () }
    }

    pub fn allow_f32_in_kernel() -> AllowF32InKernelGuard {
        AllowF32InKernelGuard { _private: () }
    }

    pub fn record_bf16_to_f32(_location: &'static str, _shape: &Shape) {}

    pub fn record_clone(_location: &'static str, _shape: &Shape) {}

    pub fn record_reshape_clone(_location: &'static str, _shape: &Shape) {}

    pub fn record_layout_fix(_location: &'static str, _shape: &Shape) {}

    pub fn record_param_f32_store(_location: &'static str, _shape: &Shape) {}

    pub fn telemetry_snapshot() -> StrictTelemetry {
        StrictTelemetry::default()
    }

    pub fn reset_counters() {}

    pub fn is_enabled() -> bool {
        false
    }

    pub fn force_runtime_enforced() {}

    pub fn disable_runtime_enforced() {}

    pub fn allow_f32_in_kernel_scoped<F, T>(body: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        body()
    }
}

pub use inner::*;
