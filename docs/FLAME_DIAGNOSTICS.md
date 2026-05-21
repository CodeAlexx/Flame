# flame-core diagnostics (debugging guide)

Audience: future Claude debugging a gradient / numerical / dispatch bug
inside flame-core, with limited context. Optimized for "I have a symptom
— which probe finds the bug?"

Cross-refs: [`FLAME_INDEX.md`](./FLAME_INDEX.md) for symbol lookup,
[`FLAME_CONVENTIONS.md`](./FLAME_CONVENTIONS.md) for naming / dispatch
gotchas, [`FLAME_KERNELS.md`](./FLAME_KERNELS.md) for kernel-level perf,
[`TRAINER_DIAGNOSTICS.md`](./TRAINER_DIAGNOSTICS.md) for the trainer-side
wiring of `assert_grad_flow` / `ParityHarness`.

---

## 0. The trap meta-pattern (read first)

Source: `/home/alex/soul.md` 2026-04-25 ("the trap I built didn't confirm
my hypothesis — it overturned it") and 2026-05-20 (HiDream-O1 SDPA-bwd
localization). This is the single most useful pattern in this doc; the
rest is the tooling that makes it executable.

**Symptom shape this catches.** A parameter's grad has roughly-right
magnitude but cos similarity vs the PyTorch reference is ~0.05. The
gradient is going in the wrong direction. End-to-end loss may even
decrease (gradient noise still has descent component) so this can hide
for thousands of steps. See `MEMORY.md::feedback_flame_core_bf16_fused_autograd`.

**Procedure.**

1. **Pick the layer closest to the loss where the bug manifests.** Last
   decoder layer typically. Probing layer 0 sees cascade noise from all
   N upstream layers and tells you almost nothing about localization.
2. **Add forward-time probes at ~3-5 intermediate tensors along the
   suspect path.** Each probe records a `TensorId`. See the
   `inference-flame/src/models/hidream_o1/trap.rs` template — copy the
   pattern (static `Mutex<Option<HashMap<String, TensorId>>>`,
   `arm_probes` / `record_probe(name, id)` / `take_probes`) for new
   models.
3. **Register those IDs in `RETAINED_INTERMEDIATE_GRAD_IDS` before
   backward.** With gradient checkpointing, the outer-tape snapshot
   fires once at `autograd.rs:2058`; checkpoint recompute uses a
   separate snapshot at `autograd.rs:3784`. If your probes record IDs
   *inside* the recompute closure, use
   `AutogradContext::retain_intermediate_grads_add` from inside the
   closure (gated by `AutogradContext::is_checkpoint_recompute()`) so
   the sub-tape walk's check picks them up.
4. **Run backward, then call `take_retained_intermediate_grads()`.**
   Match returned grads to probe names by `TensorId`.
5. **Compare each probe's grad against a Python reference's
   `retain_grad()` dump** (use the `ParityHarness`, see §3 below).
6. **The first probe where cos drops from ~1.0 to <0.5 is the bug
   site.** Walk *backward* through layers: the upstream side of that
   probe is clean, the downstream side is corrupt, so the bug is in the
   ops between this probe and its predecessor.

**Concrete example (HiDream-O1, 2026-05-20).** V-path probes inside the
first decoder layer:

| Probe | Cos vs ref | Verdict |
|---|---|---|
| `attn_out` (SDPA output, pre-`o_proj` reshape) | 0.999 | Entering SDPA-bwd cleanly |
| `v_post_repeat_kv` (V tensor going into SDPA on the next iter; really tests SDPA-bwd exit) | 0.012 | Corrupt |

Localized: bug is between `attn_out` and `v_post_repeat_kv`, i.e. inside the
old `sdpa_prefix_causal_full` narrow+cat backward chain or its 2×SDPA forward
composition. The 2026-05-21 fix records `Op::PrefixCausalFullAttention` so
the forward can stay split but backward recomputes as one exact masked SDPA.
Without the trap, all you saw was `v_proj.lora_B` grad cos ~0.05 — useless.

---

## 1. Autograd / gradient probes

### `AutogradContext::retain_intermediate_grads(ids)`

- **Where**: `flame-core/src/autograd.rs:1395` (decl, doc at 1390-1394).
- **What**: Registers a `HashSet<TensorId>` whose gradients will be
  cloned into a side-table during backward (before the standard drain
  frees them). Replaces the retain set wholesale.
- **When**: Standard case — you registered probes *before* calling
  `backward()`. If you can do that, prefer this over the `_add` variant.
- **Drain**: `AutogradContext::take_retained_intermediate_grads()` at
  `autograd.rs:1420`. Returns `HashMap<TensorId, Tensor>`. Also clears
  the retain set so the next backward isn't affected.

### `AutogradContext::retain_intermediate_grads_add(ids)` (NEW 2026-05-20)

- **Where**: `flame-core/src/autograd.rs:1412` (decl, doc at 1404-1411).
- **What**: Additive variant. Extends the existing retain set (creating
  it if absent) instead of replacing. Does NOT clear
  `RETAINED_INTERMEDIATE_GRADS` (so partial state from an earlier
  backward survives — call `take_retained_intermediate_grads` to reset
  cleanly first).
- **When**: You need to register probes *during* a checkpoint recompute
  closure. The outer-tape retain snapshot fires once at
  `autograd.rs:2058` *before* the closure runs. The sub-tape walk re-
  reads the static set at `autograd.rs:3784` (search for
  `sub_retain_ids`), so additions made inside `Op::CheckpointOffloadBoundary`
  recompute land in the sub-tape's retain check at `autograd.rs:3791`.
- **Gotcha**: Without this, every probe recorded inside a recompute
  closure has its grad silently dropped on the floor.

### `AutogradContext::is_checkpoint_recompute()`

- **Where**: `flame-core/src/autograd.rs:1299`.
- **What**: Returns `true` only when called from inside a checkpoint
  recompute closure (i.e. between entering `Op::CheckpointOffloadBoundary`
  backward and exiting it). Narrower than `AutogradContext::is_recording()`.
- **When**: Use to gate probe code that should *only* register during
  recompute (so the same forward path doesn't double-record IDs on the
  original forward and on the recompute pass).

### `flame_core::diagnostics::assert_grad_flow` / `check_grad_flow`

- **Where**: `flame-core/src/diagnostics.rs:144` (panic variant), `:111`
  (report-only variant). Env flag handled at `env_flags.rs:108-118`.
- **What**: Iterates `(name, &Parameter)` pairs after `backward()` and
  reports any parameter whose grad is missing (op never recorded) or
  zero / non-finite (recorded but produces dead grad). Returns
  `GradFlowReport` with `missing`, `zero`, `ok_count` fields.
- **When**: At step 1 (not step 0 — LoRA-B is initialized to zero, so
  step 0 produces legitimately-zero grads). Catches the BF16-fused-op-
  missing-autograd-registration bug class
  (`MEMORY.md::feedback_flame_core_bf16_fused_autograd`).
- **Invocation**: `FLAME_ASSERT_GRAD_FLOW=1` env makes the panic
  variant hard-fail; otherwise it returns the report so callers can
  log at warn level. Memory says: default-on for new trainer runs
  (`MEMORY.md::feedback_grad_flow_default_on`). See
  [`TRAINER_DIAGNOSTICS.md`](./TRAINER_DIAGNOSTICS.md) for trainer-side
  wiring.
- **Cost**: O(P × G) abs-sum, ~100 ms for a 1500-param LoRA. Do NOT
  call every step.

### `FLAME_DEBUG_FINITE=1`

- **Where**: `flame-core/src/debug_finite.rs` (module). Backward
  per-op checks at `autograd.rs:2074-2084` (incoming grad) and
  `autograd.rs:2111-2123` (produced grad).
- **What**: When set, every op in backward checks its incoming
  `output_grad` and produced `input_grads` for NaN / ±Inf. First non-
  finite value logs `[FLAME_DEBUG_FINITE] ❌ bwd:<op_tag>@<id>:<role>:
  shape=... nan=... +inf=... -inf=... finite=[...]` and returns an
  `Error::Training` so the stack unwinds at the producing site.
- **When**: `loss.backward()?` is returning `Err`, or LoRA-B is
  diverging to inf, or you suspect a backward formula divides by zero.
  The site name names the op that produced the bad value (incoming-
  grad check fails → the *next-later* op in the tape is the culprit;
  produced-grad check fails → *this* op's formula).
- **Cost**: Each check is a D2H sync + scan. Deliberately expensive;
  use for one-step repros only. When the flag is unset, every check is
  a single cached atomic load — keep the probes in production code.
- **Reset**: Call `flame_core::debug_finite::reset()` at the top of
  every step. The tripwire latches after the first failure to avoid
  log floods, so without `reset()` you only get the first step's
  diagnostic.

---

## 2. SDPA debugging

`flame-core` has cuDNN flash-bwd and a decomposed-recompute fallback.
All four flags are read in the hot path; the cuDNN-bwd ones are read
*per call* (not cached) so they can be toggled mid-process from tests.

| Env flag | Where | Effect |
|---|---|---|
| `FLAME_NO_CUDNN_SDPA_BWD=1` | `autograd.rs:965` | Forces the decomposed `attention_backward_recompute` path instead of cuDNN flash-bwd. ~12 launches vs 1, much slower but useful to isolate cuDNN-specific bugs (e.g. `MEMORY.md::feedback_cudnn_sdpa_bwd_inf_grad`). |
| `FLAME_LOG_SDPA_BWD=1` | `autograd.rs:922` (`sdpa_bwd_log`) | Logs every SDPA-bwd dispatch decision (`[sdpa-bwd] disabled-by-env`, `[sdpa-bwd] bail:mask-present`, `[sdpa-bwd] bail:causal-disabled`, `[sdpa-bwd] bail:missing-o-or-stats`, etc.). Tells you which calls fire cuDNN and which fall back per layer. |
| `FLAME_CUDNN_SDPA_BWD_CAUSAL=1` | `sdpa.rs:20-25`, gated again at `autograd.rs:970` | Allow cuDNN bwd on the causal path. Default OFF — causal SDPA goes through the decomposed fallback because cuDNN's stats convention differs from PyTorch's on causal. |
| `FLAME_SDPA_FORCE_STREAM=1` | `sdpa.rs:172-175` (`force_stream_sdpa`) | Forces the streaming SDPA forward path even when the size-based heuristic would pick the non-streaming path. |
| `FLAME_SDPA_CHUNK_MAX=N` | `sdpa.rs:177-184` (`chunk_limit_from_env`) | Override the streaming chunk size. Useful when bisecting an SDPA-streaming numerical bug — chunk boundaries can hide accumulation issues. |
| `FLAME_SDPA_STREAM_THRESHOLD=N` | `sdpa.rs:187-194` | Override the size threshold that decides streaming vs non-streaming. |

**Per-call reads.** `FLAME_NO_CUDNN_SDPA_BWD` and
`FLAME_CUDNN_SDPA_BWD_CAUSAL` are intentionally not cached at the
`autograd.rs` call sites; the per-call `std::env::var` (~50 ns) is
negligible vs cuDNN's ~10-15 ms kernel and the toggleability is worth
it for parity-harness comparisons.

---

## 3. Parity (per-tensor cos comparison vs reference)

### `flame_core::parity::ParityHarness`

- **Where**: `flame-core/src/parity.rs:92` (struct), `:101` (`load`),
  `:134` (`compare`), `:242` (`report`), `:276` (`assert_clean`).
  Module doc at top of file is worth reading in full once.
- **What**: Load a `.safetensors` reference dump, then call
  `harness.compare(name, &our_tensor)` at every layer boundary. Each
  call computes cos similarity (F32), max-abs diff, mean-abs diff, and
  `max_abs / max_abs(reference)` ratio. `harness.report()` renders a
  table; `harness.assert_clean()` panics with the table on any failure.
- **Tolerance**: `ParityTolerance { min_cos: 0.9999, max_abs_ratio:
  0.05 }` by default — calibrated for BF16 fwd. Loosen for FP8 (e.g.
  `min_cos: 0.99`). Override via `.with_tolerance(...)`.
- **Reference dump**: produce with PyTorch `register_forward_hook` on
  each layer of interest, save with `safetensors.torch.save_file`. Use
  `flame-core/scripts/dump_pytorch_layers.py` as the recipe. Keys are
  arbitrary strings — convention is the PyTorch module path
  (`transformer_blocks.0.attn.to_q.output`). Keep keys stable across
  reference and Rust callers.
- **When**: Required before any "looks-equivalent" rewrite of a kernel
  / layer. `MEMORY.md::project_flame_diagnostics_parity_2026-05-09` has
  the original ship note.
- **Note**: `note: Some(reason)` means the comparison couldn't run
  (shape mismatch, missing key, dtype cast failure). `passed: false`
  with `note: None` means real numeric divergence — the interesting
  case.

### `flame_core::ops::fused_inference::fused_linear3d_native_pytorch_parity`

- **Where**: `flame-core/src/ops/fused_inference.rs:479` (Rust),
  `flame-core/src/cuda/fused_linear3d.cu:369` (`flame_linear3d_bf16_pytorch_parity`).
- **What**: Bit-exact PyTorch linear mirror. Biased calls mirror
  `at::cuda::blas::gemm_and_bias<at::BFloat16>`: default
  `fused_linear3d_native` can deviate by 1 BF16 ULP due to
  `BIAS_DATA_TYPE`, heuristic algo selection, alignment prefs, and
  layout attrs. No-bias calls delegate to `fused_linear3d_native`,
  matching PyTorch's matmul path and avoiding the slower biased
  heuristic.
- **When**: ai-toolkit / PyTorch bit-exact parity matters at the per-op
  level. HiDream-O1 TimestepEmbedder, predecoder linears, etc. For
  established baselines (Klein / Z-Image post-validation) keep using
  `fused_linear3d_native` to preserve their existing parity.
- **Cost**: ~1% perf overhead vs the non-parity variant for biased
  calls; no-bias calls use the native path.

### `CUBLASLT_LOG_LEVEL=5 CUBLASLT_LOG_FILE=/tmp/cublaslt.log`

- **What**: NVIDIA's own cuBLASLt logging. Decisive when diffing a
  per-op result against PyTorch — emits every `cublasLtMatmul`
  invocation with algo ID, workspace size, alignment prefs, epilogue,
  bias pointer state, layout attrs.
- **When**: Linear / GEMM output deviates from PyTorch by 1-2 BF16 ULPs
  with the same input. Run flame-core and PyTorch side-by-side with
  the env set, diff the two logs. The 2026-05-20 bias-add 1-ULP delta
  was found this way: PyTorch picked `algoId=13 customOption=11`,
  flame-core picked `algoId=31` because of workspace-size and
  `BIAS_POINTER`-before-heuristic differences. Fix landed as
  `fused_linear3d_native_pytorch_parity` above.

---

## 4. Allocator

### `FLAME_ALLOC_POOL=0`

- **Where**: `flame-core/src/cuda_alloc_pool.rs:61` (cached env read).
- **What**: Disables the BF16 allocator pool — every allocation goes
  direct to cudart. Pool default is ON post-Phase-2.
- **When**:
  - **Dataset prep binaries**: pool leaks ~1 GB/sample (OOM at ~75
    samples on a 62 GB box). All 5 `prepare_*.rs` set
    `FLAME_ALLOC_POOL=0` at `main()` start. See
    `MEMORY.md::feedback_prepare_bins_pool_off`.
  - **Training crash at step 2**: pool corruption in long training
    runs. Klein 9B step-2 crash 2026-05-13 was this — fix in
    `train_klein:main` mirrors the prepare-bin pattern. See
    `MEMORY.md::project_klein9b_step2_crash_isolation`.
- **Gotcha**: F32 already goes direct to cudart even with the pool
  enabled (only BF16 is cached). So `FLAME_ALLOC_POOL=0` mostly
  matters for BF16-heavy paths.

---

## 5. Misc tracing flags

All cached via `env_flags.rs`. Read once, cheap on subsequent calls.

| Env flag | Where | Effect |
|---|---|---|
| `RUST_LOG=info` | Standard `env_logger` | Trainers in EriDiffusion-v2 use `log::info!`; without this they run silent on stdout — board.db SQLite is the canonical scalar sink, but logs help during dev. See `MEMORY.md::reference_zimage_trainer_silent_stdout`. |
| `ALLOC_LOG=1` | `env_flags.rs:28` | Log every large tensor allocation. |
| `FLAME_TRACE_DTYPE=1` | `env_flags.rs:35`, also `tensor.rs:70`, `lib.rs:73` | Print every `Tensor::matmul` call with dtypes. Useful when chasing a silent BF16↔F32 cast. |
| `FLAME_DTYPE_TRACE=1` | `env_flags.rs:42` | Print every dtype cast path. |
| `FLAME_TRACE_VERBOSE=1` | `env_flags.rs:58`, `ops/gemm.rs:133` | Verbose GEMM trace. |
| `SDXL_DEBUG_SHAPES=1` | `env_flags.rs:50` | Debug shape-mismatch traces in narrow, tile, broadcast, tensor_ext. |
| `FLAME_TRACE_PERMUTE_GENERIC=1` | `cuda_kernels.rs:39` | Trace the 4-per-block `permute_generic` launches per Klein block (~120/step, 400-600 ms/step). See `MEMORY.md::project_permute_generic_residual_per_block`. |
| `FLAME_NO_CUDNN_CONV` | `env_flags.rs:67` | Disable cuDNN conv2d fast path. |
| `FORCE_F32_CONV` | `env_flags.rs:74` | Force F32 conv fallback. |
| `FLAME_CUBLASLT_FORCE_FALLBACK=1` | `env_flags.rs:84` | Force the BF16 GEMM fallback instead of cuBLASLt fast path. |
| `FLAME_BF16_REDUCE_LEGACY=1` | `env_flags.rs:103` | Force the legacy BF16→F32→reduce→cast-back path for `Tensor::sum/mean` on BF16. Parity-bisection knob against pre-2026-05-12 behavior. |
| `FLAME_HOT_FAST_PATH_DISABLE=1` | `env_flags.rs:126` | Disable the BF16-contiguous fast path on `silu / gelu / add / mul / mul_scalar`. Bit-equivalent rollback knob. |
| `FLAME_CUDA_GRAPH=1` | `env_flags.rs:92` | Enable CUDA Graph capture/replay on backward. Requires fixed tape structure. |

---

## 6. Per-model trap registries (template)

`inference-flame/src/models/hidream_o1/trap.rs` is the reference
implementation. The pattern:

```rust
// Static, mutex-protected, Option-wrapped slot. None = disarmed.
static TRAP_PROBES: Mutex<Option<HashMap<String, TensorId>>> = Mutex::new(None);

pub fn arm_probes()           { /* set Some(empty) */ }
pub fn disarm_probes()        { /* set None */ }
pub fn record_probe(name, id) { /* insert; no-op when disarmed */ }
pub fn is_armed() -> bool     { /* cheap check for hot path */ }
pub fn take_probes()          { /* return + disarm */ }
```

**Hot-path call site**: in the layer-0 forward, after each interesting
intermediate is materialized, call `trap::record_probe("<name>",
tensor.id)`. Wrap in `if trap::is_armed()` to skip the lock on
production runs.

**Checkpoint-aware**: forward runs twice for checkpointed layers (once
on the original forward, once on recompute). `record_probe` overwrites
on duplicate name, so the recompute IDs are the ones retained — which
is what you want, since `RETAINED_INTERMEDIATE_GRAD_IDS` reads at sub-
tape time (`autograd.rs:3784`). Pair with `retain_intermediate_grads_add`
from inside the recompute path, gated by `is_checkpoint_recompute()`.

**Copy for new models**. Keep traps in the model crate, not flame-core
— they're per-model scaffolding, not framework primitives.

### Isolated-repro pattern: `tests/sdpa_prefix_causal_full_grad.rs`

When the trap localizes a bug to a specific op chain (e.g. HiDream-O1
2026-05-20 narrowed dV-corrupt-on-SDPA-bwd to the structured
prefix-causal/full path), reproduce that chain *in isolation* at the
exact shapes and dtypes from the real model. Template at
`flame-core/tests/sdpa_prefix_causal_full_grad.rs`:

| Test | Coverage |
|---|---|
| `sdpa_prefix_causal_full_grad_matches_masked_sdpa` | Structured SDPA backward vs explicit-mask reference at HiDream-O1 shape (B=1, H=32, S=497, D=128, prefix=262). |
| `sdpa_prefix_causal_full_public_image_weighted_grad_matches_masked_sdpa` | Public single-op default vs explicit-mask reference for the O1 image-row loss. |
| `sdpa_prefix_causal_full_public_checkpoint_image_weighted_grad_matches_eager` | Public single-op default through checkpoint replay vs eager public default. |
| `hidream_o1_attention_block_grad` | Full attention chain: LoRA-fused Q/K/V proj + q_norm/k_norm + halfsplit RoPE + repeat_kv + structured SDPA. |
| `hidream_o1_attention_block_grad_checkpointed` | Same chain wrapped in `Op::CheckpointOffloadBoundary` with the grow-activation-cache + retain-id mechanism. |

All three produced cos=1.0 / max_abs=0 on 2026-05-20, **proving the
structural chain is autograd-clean in isolation**. When this happens,
the localized symptom in the real model is either (a) a parity-
comparison artifact (BF16 noise integrated over many ops, ordering
mismatch with the Python reference's hooks), or (b) a multi-layer
cascade not reproducible single-layer. Structural fixes landed today
(`RopeLayout` tag, `fused_linear3d_native_pytorch_parity`,
checkpoint-aware retain) are still real correctness wins regardless of
whether they close the end-to-end cos gap by themselves.

**Lesson**: clean isolated repro at the suspected site does not vindicate
the structural fixes — they were warranted by audit, not by the gap.
Keep them. The end-to-end cos gap needs a different bisect (e.g. swap
the Python parity reference to per-step grad capture rather than
per-layer hook captures).

---

## 7. Symptom → tool

Quick-deploy table. First match wins.

| Symptom | First probe to deploy |
|---|---|
| LoRA-B has zero abs-sum after backward | `assert_grad_flow` with `FLAME_ASSERT_GRAD_FLOW=1`. Missing-grad case = autograd op never recorded for the BF16-fused op; zero-grad case = path is recorded but emits zero (dtype-cast or shape-mismatch). |
| LoRA-B grad cos vs PyTorch ~0.05, magnitude roughly right | Trap meta-pattern (§0). Probe intermediates on the layer closest to loss, walk backward to first cos<1.0 site. |
| `loss.backward()?` returns Err with no clear site | `FLAME_DEBUG_FINITE=1` + `debug_finite::reset()` per step. First failing site names the producing op. |
| LoRA-B → inf | `FLAME_DEBUG_FINITE=1` — produced-grad check at `autograd.rs:2111` will name the op whose backward formula emitted inf. |
| Per-op output differs from PyTorch by 1-2 BF16 ULPs | `CUBLASLT_LOG_LEVEL=5` side-by-side with PyTorch, diff the algo / workspace / bias-pointer / layout-attr lines. For Linear specifically, swap to `fused_linear3d_native_pytorch_parity`. |
| SDPA grad has wrong direction but the rest of the layer is clean | Set `FLAME_LOG_SDPA_BWD=1` first to confirm cuDNN is firing (not falling back). Then try `FLAME_NO_CUDNN_SDPA_BWD=1` to isolate cuDNN-specific bugs from the decomposed-recompute path. |
| End-to-end forward parity fails at one layer, fine at upstream layers | `ParityHarness::compare` at every layer boundary. First failing key names the bug site. |
| OOM in dataset prep, or step-2 crash in training | `FLAME_ALLOC_POOL=0`. See §4. |
| Trainer "runs silent" — no stdout output | `RUST_LOG=info`. Logs go via `log::info!`; without env_logger initialized at info level you see nothing. |
| Suspect a permute / tile / broadcast shape bug | `SDXL_DEBUG_SHAPES=1`. |
| Suspect a silent BF16↔F32 cast | `FLAME_TRACE_DTYPE=1` or `FLAME_DTYPE_TRACE=1`. |

---

## 8. Known gotchas

- **`debug_finite` latches.** `FIRED` is a static `AtomicBool` that
  trips on the first non-finite value. Without `debug_finite::reset()`
  at the top of each step, only step 0 produces diagnostics. The
  trainer harness should call reset unconditionally.
- **`retain_intermediate_grads` does NOT survive across `backward()`
  calls.** `take_retained_intermediate_grads` clears the retain set.
  Re-register every backward.
- **Checkpoint recompute uses a separate snapshot** of
  `RETAINED_INTERMEDIATE_GRAD_IDS`. The outer-tape walk reads at
  `autograd.rs:2058`; the sub-tape walk re-reads at `autograd.rs:3784`.
  IDs registered *between* those two reads will be honored on the sub-
  tape. IDs registered after the sub-tape read won't.
- **`assert_grad_flow` at step 0 will give false positives** on every
  LoRA-style algorithm because LoRA-B is initialized to zero. Check at
  step 1 (or use `check_grad_flow` and tolerate step-0 zeros). See
  [`TRAINER_DIAGNOSTICS.md`](./TRAINER_DIAGNOSTICS.md) §"When to check".
- **Reference dumps from PyTorch CPU diverge from CUDA at BF16.** Per
  `MEMORY.md::feedback_pytorch_cpu_vs_cuda_bf16`: cos drops to ~0.5
  per layer between CPU and CUDA BF16. Always generate parity refs on
  GPU. For 24 GB OOM on full-model dumps, use per-layer GPU streaming
  (one layer in, dump, free, repeat).
- **cuBLASLt log files grow fast.** `CUBLASLT_LOG_LEVEL=5` produces
  tens of MB per training step. Use only for single-call diffs against
  PyTorch, not full runs.
- **`ParityHarness.compare` with all-zero reference returns NaN cos.**
  Cosine is undefined at zero norm. Check `passed` and `note` rather
  than just `cos > min_cos`.
