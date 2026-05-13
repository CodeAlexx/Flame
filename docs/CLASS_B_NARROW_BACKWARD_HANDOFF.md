# Class B fix — narrow_backward dtype detour

Date: 2026-05-12
Predecessor commit: `b552f61` (Class E sync removal — landed)

## What you're fixing

`flame-core/src/tensor_narrow.rs` lines 169-183 — the BF16→F32→BF16 cast detour that wraps every BF16 call to `narrow_backward_scatter_add_cuda` in 3 extra kernel launches.

Pre-Class-B per-call cost on klein 9B-sized shapes (post-Class-E): **569 µs**
Predicted post-Class-B per-call cost: **150–200 µs**
Predicted klein 9B wall savings: **~50–100 ms/step** (× every trainer that uses narrow in backward).

## The surprise — read this before designing

The kernel named `narrow_backward_scatter_add_kernel` **does not actually scatter-add**. Its body is plain byte-copy:

```c
const uint8_t* src = grad_out + tid * elem_size;
uint8_t* dst = grad_in + in_offset_elems * elem_size;
for (int64_t i = 0; i < elem_size; ++i) {
    dst[i] = src[i];   // ASSIGN, not ADD
}
```

(see `flame-core/cuda/narrow_strided.cu` lines 109-140)

It works correctly because every caller passes `grad_in` that **starts as all zeros**:

- `flame-core/src/autograd.rs:3785` → `let mut grad_in = Tensor::zeros_dtype(input_shape, grad_dtype, device)?;`
- `flame-core/src/autograd.rs:3815` (multi-axis variant) → same pattern
- `flame-core/src/autograd.rs:3832` (fallback) → same pattern

After the kernel writes, the slice region holds `grad_out`, the rest stays at 0, the engine returns this tensor as one gradient contribution, and the autograd engine handles cross-op accumulation. So the "assign" semantic is fine for current callers.

The kernel is **dtype-agnostic** — `elem_size` parameter says how many bytes per element. F32 = 4 bytes, BF16 = 2 bytes. Both copy correctly.

## What Class B actually needs

Not a new kernel. Not atomicAdd. Not F32 opmath_t accumulation. The kernel doesn't do arithmetic.

Class B is: **let BF16 pass through the existing byte-copy kernel directly, instead of casting to F32 and back.**

## The fix (minimal)

### 1. Delete the BF16-cast detour

`flame-core/src/tensor_narrow.rs` lines 169-183 — remove the entire `if grad_in.dtype() == crate::DType::BF16 { ... return Ok(()); }` block.

### 2. Extend the dtype match

`flame-core/src/tensor_narrow.rs` lines 243-262 — currently only accepts F32. Add BF16 (and any other dtypes with valid storage):

```rust
let (go_ptr, gi_ptr): (*const c_void, *mut c_void) = match grad_in.dtype() {
    crate::DType::F32 => {
        let go_slice = grad_out.storage_ref().try_as_slice_f32()?;
        let gi_slice = grad_in.storage_mut().try_as_mut_slice_f32()?;
        (*go_slice.device_ptr() as *const c_void, *gi_slice.device_ptr_mut() as *mut c_void)
    }
    crate::DType::BF16 => {
        // Pull raw device pointers from BF16 storage.
        // tensor_raw_ptr / tensor_raw_ptr_mut are the dtype-aware
        // wrappers (see autograd.rs:743-744 for the F32-or-BF16 path).
        let src_ptr = tensor_raw_ptr(grad_out)?;
        let dst_ptr = tensor_raw_ptr_mut(grad_in)?;
        (src_ptr, dst_ptr)
    }
    other => {
        return Err(Error::Unsupported(format!(
            "narrow_backward_scatter_add_cuda: dtype {:?} not supported",
            other
        )));
    }
};
```

### 3. Verify both callers

After the dtype gate is widened, both `Tensor::narrow_backward_scatter_add_cuda` (tensor_narrow.rs) and `gpu_scatter_add_narrow` (autograd.rs) should accept BF16 directly. `gpu_scatter_add_narrow` already uses `tensor_raw_ptr` — it likely worked for BF16 even before Class B but the tensor_narrow.rs wrapper was the bottleneck.

## Verification gate

1. **Microbench** — `cargo test --release --test narrow_sync_microbench --features "cuda,heavy_kernels,bf16_u16" -- --nocapture`. Per-call should drop from ~569 µs to ~150–200 µs.
2. **Bit-identical loss on klein 9B** — first 5–8 steps. The math is identical; only the path changes. If loss values diverge from `1.0516, 1.1059, 0.6619, 0.7484, 0.2050, ...` something is wrong.
3. **Four-trainer wall-time gate** (from the speed contract):
   - Klein 9B: bandwidth-bound by BlockOffloader, so wall savings will be modest (~50 ms/step max). Sync count already at 0 from Class E.
   - Zimage: not bandwidth-bound; should show measurable wall savings.
   - Ernie/Qwen: should show measurable wall savings.
   - **Watch:** if all 4 trainers show similar absolute ms savings, the narrow path is universal across architectures and Class B is fully proven framework-wide.

## What this does NOT fix

- Class A (F32 grad storage policy in `gradient.rs:208` + `autograd_v4/gradients.rs:81`). Still bf16↔f32 cast launches everywhere autograd records F32 grads. Separate workstream.
- Class C (sum_dim_keepdim_bf16 kernel geometry, 134× per-call slower than PT). Kernel rewrite. Separate workstream.
- Bandwidth-bound wall time on klein 9B in BlockOffloader mode. Not a flame-core fix — that's the offloader's design.
- The "scatter_add" naming. Despite the name, the kernel is scatter_assign. Renaming or making it a true atomic add is a separate refactor with semantic implications (would change accumulation behavior across overlapping narrows, which doesn't currently happen in flame-core trainers but might in future ops).

## Reference points

- Class E commit: `b552f61` (this commit's parent in flame-core)
- Microbench: `flame-core/tests/narrow_sync_microbench.rs`
- Speed contract handoff: `flame-core/docs/AUTOGRAD_V2_DESIGN_REVIEW_HANDOFF.md` (covers all 4 contract axes)
- Pre-fix nsys profile (baseline): `/tmp/flame_klein9b_prof_pinned/profile.PRE_FIX.nsys-rep`
- Post-Class-E nsys profile (current state): `/tmp/flame_klein9b_short/profile.nsys-rep`

## What you should commit

1. `flame-core/src/tensor_narrow.rs` — the two edits above
2. (optional) extend `flame-core/tests/narrow_sync_microbench.rs` to assert per-call < 300 µs as a regression gate
