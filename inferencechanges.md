# Inference Mirror Updates — 2025-03-28

## Summary
- Mirrored the BF16 narrow-path fix from the Codex inference repository into this primary FlameCore tree.  
  - `src/tensor.rs`: `Tensor::narrow` now calls the CUDA BF16 implementation (`narrow_general_cuda`) whenever the `cuda` + `bf16_u16` features are enabled, avoiding the legacy FP32 staging path.  
  - Retained the old FP32 fallback under non-BF16 builds (wrapped in `#[cfg(not(all(feature = "cuda", feature = "bf16_u16")))]`).
- Adjusted imports in `src/tensor.rs` so `to_owning_fp32_strong` is only pulled in for the fallback configuration; BF16 builds stay clean of that helper.

## Impact
- Keeps inference tensors in true BF16 storage through `narrow`, eliminating transient FP32 allocations that caused OOMs during SD 3.5 MDiT conditioning.
- Ensures the main FlameCore repo matches the behavior already validated in the inference mirror (`codex-text/stable-diffusion.cpp/src/FlameCore_clean_repo2`).

## Next Steps
- Re-run `cargo check --features "cuda,bf16_u16"` (requires CUDA env) to confirm no regressions once the GPU slot is free.
- Cross-port any additional BF16 slice/permute fixes from the inference repo after they land.
