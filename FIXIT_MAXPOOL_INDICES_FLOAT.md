# FIXIT — MaxPool2D / AvgPool kernel: indices stored as `float`

Filed 2026-05-22 by the t2i-l2p port-skeptic round on Wave 0.
**Not** introduced by Wave 0 — pre-existing landmine surfaced by audit.
**Not** currently triggered by any in-tree caller. File-and-forget until a real workload hits it.

## The bug

`flame-core/src/cuda_kernels_gpu.rs:1332-1382` — the maxpool indices buffer is typed `float`:

```cuda
// forward kernel (paraphrased)
extern "C" __global__ void maxpool2d_with_indices_kernel(
    const float* input,
    float*       output,
    float*       indices,   // <-- float, not int32/int64
    ...
) {
    ...
    indices[idx] = (float)max_idx;   // linear-index argmax stored as float
}
```

And the backward casts it back:

```cuda
// backward kernel (paraphrased), cuda_kernels_gpu.rs:1469-1478
int max_idx = (int)indices[idx];     // float → int round-trip
atomicAdd(&grad_input[max_idx], grad_output[i]);
```

Float32 cannot represent every integer above `2^24 = 16_777_216`. Above that ceiling, argmax indices get **rounded to even**, and the backward scatters the gradient to the **wrong input element**. No error, no warning — silent gradient corruption.

## Why nobody hit this yet

- `cuda_kernels_gpu.rs:1804-1817` recomputes argmax during backward by calling `maxpool2d_forward_with_indices` again (see `FIXIT_MAXPOOL_BWD_RECOMPUTE.md` if/when filed). So the indices buffer is *recomputed*, not loaded from a stored tensor. The 16M ceiling still applies, but the corruption window is contained per-backward instead of cross-call.
- All current in-tree models max out at 1M-element NCHW slices per maxpool stage. L2P at 1024² hits `[1, 256, 128, 128]` = ~4M per slice (depending on stage) — well under the ceiling.
- The 16M boundary corresponds to roughly `[1, C, 4096, 4096]` per slice or `[8, C, 1448, 1448]`. Anyone doing UHR pixel-space diffusion (L2P's 4K/8K/10K roadmap) will hit it.

## The fix

Change indices storage to `int32` (sufficient up to 2^31 = 2 billion elements per slice — covers any plausible workload):

- `cuda_kernels_gpu.rs:~1332` (kernel signature): `float* indices` → `int32_t* indices`
- `cuda_kernels_gpu.rs:~1382` (store): `indices[idx] = (float)max_idx;` → `indices[idx] = (int32_t)max_idx;`
- `cuda_kernels_gpu.rs:~1469-1478` (load): `int max_idx = (int)indices[idx];` → `int32_t max_idx = indices[idx];`
- Callers in `cuda_kernels.rs` (search `maxpool2d_forward_with_indices`): change buffer alloc from `f32` to `i32`.
- PyTorch's `max_pool2d_with_indices` uses `int64`. We can stay with `int32` since flame-core's tensor numel fits in `usize` on the host but kernel ops are bounded by `gridDim.x * blockDim.x` which is `~2^31` anyway.

Add a regression test in `flame-core/tests/maxpool2d_large_input.rs`:
- Build a `[1, 1, 5000, 5000]` BF16 input = 25M elements (above 16M ceiling).
- Place a sentinel value at a position whose linear index is > 16M, say index = 20_000_000.
- Forward through a 1×1 maxpool (degenerate but isolates the issue), backward.
- Assert grad is exactly at the sentinel position, not rounded-to-even.

## Who should pick this up

Flame-core perf/correctness backlog. Half-day job. No port currently blocks on this. Filed so it doesn't get lost when someone eventually trains a UHR pixel-space DiT and starts seeing gradient noise correlated with input scale.

## Cross-refs

- Skeptic finding F4 — `inference-flame/src/models/l2p/SKEPTIC_FINDINGS_2026-05-22.md`
- Related (also pre-existing): the maxpool backward "recompute forward" cost — file as a separate FIXIT if a profiling pass shows it bites.
