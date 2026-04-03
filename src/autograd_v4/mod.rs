//! Experimental Burn-style autograd implementation (v4).
//! This module is feature-gated via `autograd_v4`.

#![cfg(feature = "autograd_v4")]

pub mod gradients;
pub mod graph;
pub mod ops;

#[cfg(test)]
mod tests;

pub use gradients::Gradients;
pub use graph::{GradNode, Op};
#[cfg(feature = "sdpa_debug")]
pub use ops::register_sdpa_hooks;
pub use ops::{
    attach_backward_node, backward_v4, clear_tape, sdpa_backward, sdpa_forward, SdpaConfig,
    SdpaCtx, SdpaHooks, SdpaSave, SdpaStats,
};

/// Trait for applying gradients produced by the v4 engine.
pub trait ApplyGradV4 {
    fn apply_grads_v4(&mut self, grads: &Gradients, lr: f32);
}
