//! Autograd v2 — clean-sheet PyTorch-style DAG autograd engine.
//!
//! **Status**: Phase 0 only. This module is intentionally empty.
//!
//! Phase 0 wires the feature flag and the module path so that subsequent
//! phases can land incrementally without touching `Cargo.toml` or
//! `lib.rs`. The actual v2 implementation arrives in waves:
//!
//! - Phase 1: metadata + core types (`Edge`, `GradFn`, `SavedTensor`,
//!   `InputBuffer`, `Hooks`, weak leaf accumulator cache).
//! - Phase 2: engine skeleton (`GraphRoot`, `AccumulateGrad`,
//!   dependency counting, ready queue, nested execute, hook dispatch).
//! - Phase 3: first real ops + view backward + forward-mode AD.
//! - Phase 4: optimizer + trainer integration (BF16 grads end-to-end
//!   for BF16 params — Class A path).
//! - Phase 5: parity gate before retiring v1.
//!
//! See `docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md` (the design + handoff
//! doc) and `docs/BF16_GRAD_DECISION.md` (the Phase 0 grad-dtype
//! decision) for the full plan.
