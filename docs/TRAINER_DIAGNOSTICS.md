# Trainer diagnostics — wiring guide

Two flame-core helpers exist for catching trainer bugs that the autograd
graph would otherwise hide.  Both live in modules under `flame_core::`
and are importable from any trainer.

## `flame_core::diagnostics::assert_grad_flow`

Catches the recurring "BF16 fused inference op missing autograd
registration" bug class (memory entry
`feedback_flame_core_bf16_fused_autograd.md`).  Klein, Z-Image, Chroma,
and `rope_fused_bf16` all hit this same bug in the past — in each case
we trained for thousands of steps before noticing the LoRA was useless.

### Pattern

```rust
use flame_core::diagnostics;
use flame_core::parameter::Parameter;

// ... inside the training loop, after backward ...
let grads = loss.backward()?;

if step == 1 {   // NOT step 0 — see "When to check" below.
    let names = model.bundle.parameter_names();   // Vec<String>
    let named_refs: Vec<(&str, &Parameter)> = names
        .iter()
        .zip(params.iter())
        .map(|(n, p)| (n.as_str(), p))
        .collect();
    let report = diagnostics::assert_grad_flow(&grads, &named_refs)?;
    if report.is_clean() {
        log::info!("[grad-flow] step 1 clean ({} params)", report.ok_count);
    } else {
        log::warn!("{}", report.summary());
    }
}
```

### Behavior

- `check_grad_flow()` returns a `GradFlowReport` and never panics.
- `assert_grad_flow()` returns the same report; panics with a named-list
  summary only when `FLAME_ASSERT_GRAD_FLOW=1` is set in the environment.
- The flag is intended for CI / regression gating on trainers where the
  invariant is "every parameter gets a non-zero gradient at step 1".

### When to check — step 1, not step 0

Every LoRA-style algorithm (LoRA, LoCon, LoHa, LoKr) initializes one
factor at zero so the adapter delta `Δ = factor_a · factor_b = 0` at
step 0 — i.e. the LoRA contributes *nothing* to the forward at init.

Because of that initialization choice, backward through `loss(base + Δ)`
gives `grad_factor_a = (∂L/∂Δ) · factor_b^T = 0` (since
`factor_b = 0`), and only `factor_b` (the non-zero factor) gets a
non-zero gradient at step 0.  This is **mathematically required**, not
a bug — it matches PyTorch upstream's step-0 grad pattern verbatim.

Calling the assertion at `step == 0` therefore reports half the leaves
as "dead" on every healthy LoRA training run.  Use **step 1** instead:
after the first optimizer step has driven the zero factor off zero,
all leaves should have non-zero gradients in a correctly-wired
trainer.  A dead leaf at step 1 *is* the bug pattern this assertion
catches.

### Cost

O(P × G) where P is parameter count and G is per-param grad size.
For a typical LoRA training run (~1500 params × rank-16 matrices) this is
well under 100 ms on a 3090.  Call once at step 0 — do **not** call every
step.

### Per-trainer wire-in status

| Trainer | parameter-name accessor | wire-in status |
|---|---|---|
| `train_qwenimage` | `model.bundle.parameter_names()` (core `:100`) | DONE |
| `train_klein`     | `model.named_parameters()` (mixed legacy + Lycoris) | TODO (mechanical) |
| `train_zimage`    | `model.bundle.named_parameters()` | TODO (mechanical) |
| `train_flux`      | `bundle.named_parameters()` (legacy + Lycoris) | TODO (mechanical) |
| `train_sdxl`      | `model.named_parameters()` | TODO (mechanical) |
| `train_sd35`      | `model.named_parameters()` | TODO (mechanical) |
| `train_anima`     | `model.named_parameters()` | TODO (mechanical) |
| `train_ernie`     | `model.named_parameters()` | TODO (mechanical) |
| `train_acestep`   | `model.named_parameters()` | TODO (mechanical) |
| `train_slider_klein` | `model.named_parameters()` | TODO (mechanical) |
| `train_chroma`    | none — bundle uses `Arc<dyn AdapterModule>` keyed by `(layer_idx, target)`; needs a small synthesizer that walks `AdapterModule::named_tensors()` and prefixes with the outer key | needs accessor first |
| `train_wan22`     | none — `Wan22LoraBundle` mirrors `QwenImageLoraBundle`; add an analogous `parameter_names()` helper | needs accessor first |
| `train_ltx2`      | none — bin uses `model.parameters()` only; add `named_parameters()` plumbing | needs accessor first |

For the three trainers that need an accessor first, the pattern is the
same as `QwenImageLoraBundle::parameter_names()` at
`crates/eridiffusion-core/src/models/qwenimage.rs:100`.

### Qwen-image caveat

Qwen-image has 4 architecturally-zero block-59 txt-stream params
(`add_q_proj`, `to_add_out`, `txt_mlp.net.0.proj`, `txt_mlp.net.2`) that
will legitimately appear in the report.  Do **not** enable
`FLAME_ASSERT_GRAD_FLOW=1` for qwen runs — log the report at warn level
instead.  Klein/SDXL/SD3.5/anima/ernie/flux do not have architectural
zeros at the LoRA level and can run with the panic flag enabled in CI.

---

## `flame_core::parity::ParityHarness`

Catches the "looks-equivalent rewrite that silently broke a layer" class
(MagiHuman deinterleave incident, soul.md 2026-05-01).  Compares
intermediate activations against a PyTorch reference dump layer-by-layer
before declaring a port equivalent.

### Pattern

```rust
use flame_core::parity::{ParityHarness, ParityTolerance};

let mut h = ParityHarness::load("dumps/qwen_block0_bf16.safetensors", device)?
    .with_tolerance(ParityTolerance { min_cos: 0.9999, max_abs_ratio: 0.05 });

let q = block.attn.to_q.forward(&hidden)?;
h.compare("transformer_blocks.0.attn.to_q.output", &q)?;

let attn_out = block.attn.forward(&hidden, &encoder)?;
h.compare("transformer_blocks.0.attn.output", &attn_out)?;

// ... compare every boundary you care about ...

h.assert_clean();   // panics with the formatted report on any FAIL
// or:
println!("{}", h.report());
if !h.is_clean() { /* dump and continue */ }
```

### Producing the reference dump

Use `flame-core/scripts/dump_pytorch_layers.py` as the template.  Two
things to get right:

1. The dump keys must match the strings the Rust caller passes to
   `compare()`.  Convention is the PyTorch module path
   (`transformer_blocks.0.attn.to_q.output`); rename the keys at dump
   time if PyTorch's internal naming differs from what the Rust code
   uses.
2. The dump dtype must match what you're testing.  Save BF16 when
   measuring "does our BF16 forward match PyTorch's BF16 forward
   bit-for-bit"; save F32 when measuring "how much precision did our
   BF16 forward lose vs the F32 ground truth".

The smoke test at `flame-core/tests/parity_harness_smoke.rs` shows the
end-to-end flow with synthetic data.

### When to use which

- **Grad-flow assertion**: continuously, every training run, at step 0.
  Free insurance.
- **Parity harness**: ad-hoc, when porting a model or rewriting a
  kernel.  Run it once before declaring the port equivalent, then
  delete the call sites — it's not a continuous invariant, it's a
  pre-flight check.
