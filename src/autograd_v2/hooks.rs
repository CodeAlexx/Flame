//! Hooks plumbing.
//!
//! Per §8 and recommended-change 13 of
//! `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md`: hooks must be part of
//! the v0 surface because both OneTrainer and SimpleTuner — flame-core's
//! parity references — use `register_full_backward_hook` /
//! `register_forward_hook` for memory orchestration (block swap, attention
//! stat collection, etc.).
//!
//! Phase 1 ships the type and accessor. Phase 2 wires dispatch at
//! `GradFn` entry/exit. Default state is empty; the no-hook fast path
//! is a pointer comparison against `Hooks::EMPTY`.

use crate::tensor::Tensor;
use std::sync::Arc;

/// Callback fired before a `GradFn::apply()`. Receives the grad_outputs
/// vector by reference so the hook may inspect (but not own) the
/// gradients flowing into the node.
pub type PreBackwardHook = Arc<dyn Fn(&[Option<Tensor>]) + Send + Sync>;

/// Callback fired after a `GradFn::apply()`. Receives the grad_inputs
/// vector by reference.
pub type PostBackwardHook = Arc<dyn Fn(&[Option<Tensor>]) + Send + Sync>;

/// Per-tensor hook: takes the gradient flowing through a tensor and
/// optionally returns a replacement gradient (PyTorch
/// `register_hook` semantics — `None` = leave unchanged).
pub type TensorHook = Arc<dyn Fn(&Tensor) -> Option<Tensor> + Send + Sync>;

#[derive(Clone)]
pub struct Hooks {
    pub pre_backward: Vec<PreBackwardHook>,
    pub post_backward: Vec<PostBackwardHook>,
    pub tensor_hooks: Vec<TensorHook>,
}

impl Hooks {
    /// Sentinel empty `Hooks` used by the default `GradFn::hooks()`
    /// implementation. Cannot be a `const` because `vec![]` isn't const
    /// in stable Rust 1.x; cached behind a `OnceLock` so the no-hook
    /// fast path is a single atomic load.
    pub fn empty_ref() -> &'static Hooks {
        static EMPTY: std::sync::OnceLock<Hooks> = std::sync::OnceLock::new();
        EMPTY.get_or_init(|| Hooks {
            pre_backward: Vec::new(),
            post_backward: Vec::new(),
            tensor_hooks: Vec::new(),
        })
    }

    pub fn new() -> Self {
        Self {
            pre_backward: Vec::new(),
            post_backward: Vec::new(),
            tensor_hooks: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.pre_backward.is_empty()
            && self.post_backward.is_empty()
            && self.tensor_hooks.is_empty()
    }
}

impl Default for Hooks {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Hooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hooks")
            .field("pre_backward", &self.pre_backward.len())
            .field("post_backward", &self.post_backward.len())
            .field("tensor_hooks", &self.tensor_hooks.len())
            .finish()
    }
}
